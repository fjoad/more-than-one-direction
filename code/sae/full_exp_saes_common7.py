#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_sae_common7_alpha60_two_versions.py

SAE steering with 7 fixed "common core" latents from a layer-31 SAE on Gemma-2-9B-IT.

Data:
  • 13 balanced splits (47 HR + 47 BC each) in:
        data/refusal_13splits/*.json

SAE:
  • JumpReLUSAE checkpoints under SAE_BASE_DIR, layer 31, width 16k, l0=14.
  • We use exactly these 7 latent indices at layer 31:
        [550, 1779, 5176, 6768, 7137, 13393, 5638]
    Each decoder row W_dec[idx] is a direction in residual space.

Steering:
  • Additive steering only (no ablation).
  • Strength α = 60.
  • Directions: all 7 decoder rows are applied jointly (sum of α * d_i).

We run two *separate* versions:

  Version A: "L31_pre_mid_post"
      - For each split:
          - Hook at layer 31
          - Act points: resid_pre, resid_mid, resid_post
          - For every token: h <- h + 60 * sum_i d_i

  Version B: "L32_pre_only"
      - Reload model from scratch.
      - For each split:
          - Hook at layer 32
          - Act point: resid_pre only
          - For every token: h <- h + 60 * sum_i d_i

Outputs:
  generations/sae_common7_alpha60/
      L31_pre_mid_post/<split_name>/sae_add.json
      L32_pre_only/<split_name>/sae_add.json

Each JSON file is a list of dicts:
    {
      "prompt": <str>,
      "prompt_harm_label": "harmful" | "unharmful",
      "was_refusal": 0 | 1,
      "model_response": <str>
    }
"""

import gc
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from jaxtyping import Float, Int
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformer_lens.utilities import devices as tl_devices

# ----------------- Global config -----------------
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = "google/gemma-2-9b-it"
DTYPE = torch.float16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SPLITS_DIR = Path("data") / "refusal_13splits"

SAE_BASE_DIR = Path("../exp1_v3/saes")
SAE_LAYER = 31          # layer where the SAE lives
SAE_WIDTH = "16k"
SAE_L0 = "14"

# 7 common-core latents (layer-31)
SAE_LATENT_IDXS = [550, 1779, 5176, 6768, 7137, 13393, 5638]

ALPHA = 60.0

MAX_LEN = 128
MAX_NEW = 64
GEN_BATCH = 4

OUT_ROOT = Path("generations") / "sae_common7_alpha60"

# --------------- Small helpers -------------------
def clear_cuda():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
    gc.collect()


def load_split(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list.")
    for rec in data:
        for k in ["prompt", "prompt_harm_label", "was_refusal"]:
            if k not in rec:
                raise KeyError(f"{path} missing key {k} in some records.")
    return data


def save_generations(
    path: Path,
    records: List[Dict[str, Any]],
    generations: List[str],
) -> None:
    assert len(records) == len(generations)
    path.parent.mkdir(parents=True, exist_ok=True)
    out: List[Dict[str, Any]] = []
    for rec, gen in zip(records, generations):
        out.append(
            {
                "prompt": rec["prompt"],
                "prompt_harm_label": rec["prompt_harm_label"],
                "was_refusal": int(rec["was_refusal"]),
                "model_response": gen,
            }
        )
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# Simple JumpReLUSAE just to load params and W_dec
class JumpReLUSAE(torch.nn.Module):
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))


def load_local_sae(
    layer: int,
    width: str,
    l0: str,
    base_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> JumpReLUSAE:
    params_path = base_dir / f"layer_{layer}" / f"width_{width}" / f"average_l0_{l0}" / "params.npz"
    if not params_path.exists():
        raise FileNotFoundError(params_path)

    params = np.load(params_path)
    sae = JumpReLUSAE(params["W_enc"].shape[0], params["W_enc"].shape[1]).to(device, dtype)
    sae.W_enc.data.copy_(torch.from_numpy(params["W_enc"]).to(device, dtype))
    sae.W_dec.data.copy_(torch.from_numpy(params["W_dec"]).to(device, dtype))
    sae.b_enc.data.copy_(torch.from_numpy(params["b_enc"]).to(device, dtype))
    sae.b_dec.data.copy_(torch.from_numpy(params["b_dec"]).to(device, dtype))
    sae.threshold.data.copy_(torch.from_numpy(params["threshold"]).to(device, dtype))
    sae.eval()
    return sae


# --------------- Tokenization & generation ----------------
def tokenize_instructions_gemma_chat(
    tokenizer: AutoTokenizer,
    model_device: torch.device,
    instructions: List[str],
    max_length: int = MAX_LEN,
) -> Int[torch.Tensor, "batch seq"]:
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": inst}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for inst in instructions
    ]
    toks = tokenizer(
        chats,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).input_ids.to(model_device)
    return toks


def _generate_with_hooks(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    toks: Int[torch.Tensor, "batch seq"],
    max_tokens_generated: int,
    fwd_hooks: List[Tuple[str, callable]],
) -> List[str]:
    B, T = toks.shape
    all_toks = torch.zeros(
        (B, T + max_tokens_generated),
        dtype=torch.long,
        device=toks.device,
    )
    all_toks[:, :T] = toks

    for i in range(max_tokens_generated):
        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(all_toks[:, : T + i])
                next_tokens = logits[:, -1, :].argmax(dim=-1)
                all_toks[:, T + i] = next_tokens

    gen_suffix = all_toks[:, T:]
    out = tokenizer.batch_decode(gen_suffix, skip_special_tokens=True)
    return out


def generate_for_records(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    records: List[Dict[str, Any]],
    fwd_hooks: List[Tuple[str, callable]],
    desc: str,
) -> List[str]:
    prompts = [r["prompt"] for r in records]
    generations: List[str] = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc=desc, unit="batch"):
        batch_prompts = prompts[i : i + GEN_BATCH]
        toks = tokenize_instructions_gemma_chat(
            tokenizer, model.cfg.device, batch_prompts, max_length=MAX_LEN
        )
        outs = _generate_with_hooks(
            model,
            tokenizer,
            toks,
            max_tokens_generated=MAX_NEW,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(outs)
        del toks, outs
        clear_cuda()
    return generations


# --------------- Hook builders ----------------
def build_sae_directions(model_device: torch.device, model_dtype: torch.dtype) -> List[torch.Tensor]:
    """Load SAE at layer 31 and return the 7 decoder-row directions on model device."""
    sae = load_local_sae(
        layer=SAE_LAYER,
        width=SAE_WIDTH,
        l0=SAE_L0,
        base_dir=SAE_BASE_DIR,
        device=model_device,
        dtype=model_dtype,
    )
    dirs: List[torch.Tensor] = []
    for idx in SAE_LATENT_IDXS:
        if idx < 0 or idx >= sae.W_dec.shape[0]:
            raise IndexError(f"SAE latent index {idx} out of range for layer {SAE_LAYER}")
        vec = sae.W_dec[idx].detach().to(model_device, dtype=model_dtype)
        dirs.append(vec)
    del sae
    clear_cuda()
    return dirs


def hooks_addition_many(
    directions: List[torch.Tensor],
    layer: int,
    act_points: List[str],
    alpha: float,
) -> List[Tuple[str, callable]]:
    """
    Add alpha * sum(directions) at the given layer and act_points,
    for all tokens and every forward step.
    """
    def hook_fn(
        activation: Float[torch.Tensor, "... d_act"],
        hook: HookPoint,
        directions=directions,
        alpha=alpha,
    ):
        out = activation
        for d in directions:
            dd = d.to(out.device, dtype=out.dtype)
            out = out + alpha * dd
        return out

    hooks: List[Tuple[str, callable]] = []
    for act_name in act_points:
        hook_name = utils.get_act_name(act_name, layer)
        hooks.append((hook_name, hook_fn))
    return hooks


# --------------- One version runner ----------------
def run_version(
    version_name: str,
    layer: int,
    act_points: List[str],
):
    """
    Run one steering configuration:
        - load model fresh
        - build SAE directions
        - hook alpha * sum(directions) at specified layer & act_points
        - generate steered outputs for all 13 splits
    """
    print(f"\n==============================")
    print(f"Running version: {version_name}")
    print(f"  Layer = {layer}, act_points = {act_points}")
    print("==============================\n")

    # Patch device mapping (same pattern as other scripts)
    _orig_get_dev = tl_devices.get_device_for_block_index

    def _capped_get_device_for_block_index(index, cfg, device=None):
        dev = _orig_get_dev(index, cfg, device)
        if isinstance(dev, torch.device) and dev.type == "cuda" and dev.index is not None:
            if dev.index >= torch.cuda.device_count():
                return torch.device(f"cuda:{torch.cuda.device_count() - 1}")
        return dev

    tl_devices.get_device_for_block_index = _capped_get_device_for_block_index

    # Load model
    print("Loading model...")
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_PATH,
        device=DEVICE,
        n_devices=torch.cuda.device_count(),
        dtype=DTYPE,
    )
    tokenizer: AutoTokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded.")
    clear_cuda()

    # SAE directions (7 latents)
    print("Loading SAE directions (7 common latents at layer 31)...")
    sae_dirs = build_sae_directions(model.cfg.device, model.cfg.dtype)
    print(f"Loaded {len(sae_dirs)} directions.")
    clear_cuda()

    # Build hooks for this version
    hooks = hooks_addition_many(
        directions=sae_dirs,
        layer=layer,
        act_points=act_points,
        alpha=ALPHA,
    )

    # Split files
    split_files = sorted(SPLITS_DIR.glob("*.json"))
    if len(split_files) != 13:
        print(f"[WARN] Expected 13 splits in {SPLITS_DIR}, found {len(split_files)}")

    out_root_version = OUT_ROOT / version_name
    out_root_version.mkdir(parents=True, exist_ok=True)

    for split_path in split_files:
        split_name = split_path.stem
        print(f"\n--- Split: {split_name} ---")
        records = load_split(split_path)
        print(f"  {len(records)} prompts in this split.")

        out_dir = out_root_version / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "sae_add.json"
        if out_path.exists():
            print(f"  [skip] {out_path} already exists")
            continue

        gens = generate_for_records(
            model,
            tokenizer,
            records,
            fwd_hooks=hooks,
            desc=f"{version_name} – {split_name}",
        )
        save_generations(out_path, records, gens)
        del gens
        clear_cuda()
        print(f"  ✓ Wrote steered generations to {out_path}")

    # Clean up model before next version
    del model, tokenizer
    clear_cuda()
    print(f"\n✓ Completed version: {version_name}")


# --------------- Main ----------------
def main():
    # Version A: layer 31, resid_pre/mid/post
    run_version(
        version_name="L31_pre_mid_post",
        layer=31,
        act_points=["resid_pre", "resid_mid", "resid_post"],
    )

    # Version B: layer 32, resid_pre only
    run_version(
        version_name="L32_pre_only",
        layer=32,
        act_points=["resid_pre"],
    )

    print("\nAll versions complete.")
    print(f"Results under: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
