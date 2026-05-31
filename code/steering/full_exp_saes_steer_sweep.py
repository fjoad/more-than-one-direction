#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_sae_13dirs_sweep_on_50samples.py

For each of the 13 splits in data/refusal_13splits/*.json:

  - Each file already contains exactly 47 "HR" prompts and 47 "BC" prompts:
        HR = harmful + refused
        BC = unharmful + compliant
    We DO NOT SAMPLE. We use all 64 rows in that file.

  - For SAE_LAYERS = [9, 20, 31]:
      * capture resid_pre at layer = L+1 (10, 21, 32), pos = -2, for HR47 & BC47
      * run through JumpReLUSAE at layer L (local files in SAE_BASE_DIR)
      * compute fraction active on HR vs BC (active = code > EPS)
      * pick top-10 latents in that layer by diff = freq_HR - freq_BC

  - Across all 3 * 10 = 30 candidates, pick overall top-10 by diff.

  - Each selected latent's decoder row (from its SAE) is a direction.
    Directions are raw W_dec rows (no unit norm), exactly like your SAE code.

  - For each split & each alpha ∈ {100, 90, ..., 10}:
      * attach one hook per latent (10 hooks total), at resid_pre of the
        capture layer (L+1), ALL TOKENS (every position) get the addition.
      * generate on the shared 50-sample test set and save:

        generations/sae_13dirs_sweep/<split_name>/alpha<alpha>.json

    Each JSON file is a list of dicts:
      {
        "prompt": <str>,
        "prompt_harm_label": "harmful" | "unharmful",
        "original_response": <str>,
        "was_refusal": 0 | 1,
        "alpha": <float>,
        "model_response": <str>
      }

Total: 13 splits × 10 alphas = 130 output files.
"""

import os
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformer_lens.utilities import devices as tl_devices

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────

MODEL_PATH = "google/gemma-2-9b-it"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

# 13 split files with exactly 47 HR + 47 BC each
SPLITS_DIR = Path("data") / "refusal_13splits"

# Shared test set (4 × 50 samples)
TEST_DIR = Path("data") / "4splits_combined" / "50samples"

# SAE layers & capture layers: we train SAEs on L, but capture resid_pre at L+1
SAE_LAYERS      = [9, 20, 31]
CAPTURE_LAYERS  = [L + 1 for L in SAE_LAYERS]

SAE_BASE_DIR    = "../exp1_v3/saes"  # adjust to match your checkpoint dir
SAE_WIDTH       = "16k"
SAE_L0          = "14"
EPS             = 1e-6

TOP_PER_LAYER   = 10   # top-10 per SAE layer
TOP_GLOBAL      = 10   # overall top-10 across all layers combined

POS_SLICE       = -2   # position used for training (second-to-last token)

ALPHAS          = [0.0, 10.0, 30.0, 60.0]   #[float(a) for a in range(100, 9, -10)]  # 100, 90, ..., 10

MAX_LEN         = 128
MAX_NEW         = 64
GEN_BATCH       = 4

OUT_ROOT        = Path("generations") / "sae_13dirs_sweep"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# Hygiene
# ────────────────────────────────────────────────────────────────

def clear_cuda():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    gc.collect()

# ────────────────────────────────────────────────────────────────
# Model & tokenizer
# ────────────────────────────────────────────────────────────────

print("=== Loading model ===")
_orig_get_dev = tl_devices.get_device_for_block_index

def _capped_get_device_for_block_index(index, cfg, device=None):
    dev = _orig_get_dev(index, cfg, device)
    if isinstance(dev, torch.device) and dev.type == "cuda" and dev.index is not None:
        if dev.index >= torch.cuda.device_count():
            return torch.device(f"cuda:{torch.cuda.device_count() - 1}")
    return dev

tl_devices.get_device_for_block_index = _capped_get_device_for_block_index

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

# ────────────────────────────────────────────────────────────────
# Tokenisation & generation helpers
# ────────────────────────────────────────────────────────────────

def tokenize_instructions_gemma_chat(
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
    ).input_ids.to(model.cfg.device)
    return toks


def _generate_with_hooks(
    toks: Int[torch.Tensor, "batch seq"],
    max_tokens_generated: int,
    fwd_hooks: List[Tuple[str, callable]],
) -> List[str]:
    """
    Manual greedy generation with HookedTransformer hooks.
    """
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
    records: List[Dict[str, Any]],
    fwd_hooks: List[Tuple[str, callable]],
    desc: str,
) -> List[str]:
    prompts = [r["prompt"] for r in records]
    generations: List[str] = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc=desc, unit="batch"):
        batch_prompts = prompts[i : i + GEN_BATCH]
        toks = tokenize_instructions_gemma_chat(batch_prompts, max_length=MAX_LEN)
        outs = _generate_with_hooks(
            toks,
            max_tokens_generated=MAX_NEW,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(outs)
        del toks, outs
        clear_cuda()
    return generations

# ─────────────────────────── SAE module & loader ───────────────────────────

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon


def load_local_sae(
    layer: int,
    width: str = SAE_WIDTH,
    l0: str = SAE_L0,
    base_dir: str = SAE_BASE_DIR,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> JumpReLUSAE:
    params_path = (
        Path(base_dir)
        / f"layer_{layer}"
        / f"width_{width}"
        / f"average_l0_{l0}"
        / "params.npz"
    )
    if not params_path.exists():
        raise FileNotFoundError(params_path)

    params = np.load(params_path)
    sae = JumpReLUSAE(
        params["W_enc"].shape[0],
        params["W_enc"].shape[1],
    ).to(device, dtype)

    sae.W_enc.data.copy_(torch.from_numpy(params["W_enc"]).to(device, dtype))
    sae.W_dec.data.copy_(torch.from_numpy(params["W_dec"]).to(device, dtype))
    sae.b_enc.data.copy_(torch.from_numpy(params["b_enc"]).to(device, dtype))
    sae.b_dec.data.copy_(torch.from_numpy(params["b_dec"]).to(device, dtype))
    sae.threshold.data.copy_(torch.from_numpy(params["threshold"]).to(device, dtype))
    sae.eval()
    return sae

# ────────────────────────────────────────────────────────────────
# Activation caching
# ────────────────────────────────────────────────────────────────

def cache_resid_pre_layer_pos(
    prompts: List[str],
    layer: int,
    pos_slice: int = POS_SLICE,
    desc: str = "cache",
) -> torch.Tensor:
    """
    Cache resid_pre at `layer` and a single position `pos_slice` for given prompts.
    Returns tensor [N, d_model] on CPU float32.
    """
    hook_name = utils.get_act_name("resid_pre", layer)
    chunks: List[torch.Tensor] = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc=f"Cache {desc}", unit="batch"):
        batch_prompts = prompts[i : i + GEN_BATCH]
        toks = tokenize_instructions_gemma_chat(batch_prompts, max_length=MAX_LEN)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                toks,
                names_filter=lambda n: n == hook_name,
                pos_slice=pos_slice,
                stop_at_layer=layer + 1,
            )
        acts = cache[hook_name].squeeze(1).to("cpu", dtype=torch.float32).contiguous()
        chunks.append(acts)
        del cache, toks, acts
        clear_cuda()
    if not chunks:
        return torch.empty(0, model.cfg.d_model, dtype=torch.float32)
    return torch.cat(chunks, dim=0)

# ───────────────────── steering hook (EVERY TOKEN) ─────────────────────

def make_addition_hook_every_token(
    direction: torch.Tensor,
    alpha: float,
):
    """
    Simple SAE-style hook:
      - direction: decoder row in residual space [d_model]
      - alpha: scaling coefficient
      - applies to resid_pre activations of shape [B, T, d_model]
      - adds direction to *every token* (all T positions).
    """
    def _hook(activation: Float[torch.Tensor, "batch seq d_model"], hook: HookPoint):
        d = direction
        if d.device != activation.device or d.dtype != activation.dtype:
            d = d.to(activation.device, activation.dtype, non_blocking=True)
        # broadcast to [1, 1, d_model] and add to all tokens
        return activation + alpha * d.view(1, 1, -1)
    return _hook


def build_latent_hooks_for_split(
    steering_latents: List[Tuple[int, int, torch.Tensor]],
    alpha: float,
) -> List[Tuple[str, callable]]:
    """
    steering_latents: list of (layer0, latent_idx, direction) for this split.
    Attach hooks at resid_pre of capture layer = layer0 + 1.
    Each latent becomes its own direction; all are active simultaneously.
    """
    hooks: List[Tuple[str, callable]] = []
    for layer0, latent_idx, vec in steering_latents:
        capture_layer = layer0 + 1
        hook_name = utils.get_act_name("resid_pre", capture_layer)
        hook_fn = make_addition_hook_every_token(vec, alpha=alpha)
        hooks.append((hook_name, hook_fn))
    return hooks

# ───────────────────────── SAE latents per split ─────────────────────────

def build_sae_latents_for_split(
    hr_prompts: List[str],
    bc_prompts: List[str],
    split_name: str,
) -> List[Tuple[int, int, torch.Tensor]]:
    """
    For one split:

      - For each SAE layer L in SAE_LAYERS:
          * capture resid_pre at layer = L+1, pos=-2 for HR & BC prompts
          * encode with SAE at layer L
          * compute freq_HR, freq_BC where "active" = code > EPS
          * find top-10 latents by diff = freq_HR - freq_BC

      - Combine all 30 candidates, sort by diff, pick top-10 overall.

      - For each selected candidate, direction = decoder row W_dec[latent_idx]
        from the SAE for that layer (no normalization).

    Returns:
      steering_latents: list of (layer0, latent_idx, direction_tensor)
    """
    print(f"  Building SAE latents for split {split_name}")

    candidates: List[Tuple[int, int, float]] = []  # (layer0, latent_idx, diff)
    sae_cache: Dict[int, JumpReLUSAE] = {}

    for layer0, capture_layer in zip(SAE_LAYERS, CAPTURE_LAYERS):
        print(f"    SAE layer {layer0} (capture resid_pre at layer {capture_layer})")

        acts_hr = cache_resid_pre_layer_pos(
            hr_prompts,
            layer=capture_layer,
            pos_slice=POS_SLICE,
            desc=f"{split_name} HR L{capture_layer}",
        )
        acts_bc = cache_resid_pre_layer_pos(
            bc_prompts,
            layer=capture_layer,
            pos_slice=POS_SLICE,
            desc=f"{split_name} BC L{capture_layer}",
        )

        sae = load_local_sae(
            layer=layer0,
            width=SAE_WIDTH,
            l0=SAE_L0,
            base_dir=SAE_BASE_DIR,
            device=model.cfg.device,
            dtype=model.cfg.dtype,
        )
        sae_cache[layer0] = sae

        x_all = torch.cat([acts_hr, acts_bc], dim=0)  # [64, d_model]
        labels = torch.cat(
            [
                torch.ones(len(acts_hr), dtype=torch.bool),   # HR
                torch.zeros(len(acts_bc), dtype=torch.bool),  # BC
            ]
        )

        with torch.no_grad():
            codes = sae.encode(
                x_all.to(model.cfg.device, dtype=sae.W_enc.dtype)
            )  # [64, d_sae]
            mask = codes > EPS

            hr_mask = labels.to(mask.device)
            bc_mask = ~hr_mask

            freq_hr = mask[hr_mask].float().mean(dim=0)  # [d_sae]
            freq_bc = mask[bc_mask].float().mean(dim=0)

            diff = freq_hr - freq_bc

            vals_layer, idx_layer = torch.topk(diff, TOP_PER_LAYER)
            for v, idx_lat in zip(vals_layer.tolist(), idx_layer.tolist()):
                candidates.append((layer0, idx_lat, v))

        del acts_hr, acts_bc, x_all, codes, mask
        clear_cuda()

    # Sort all candidates by diff descending and keep overall top-10
    candidates.sort(key=lambda t: t[2], reverse=True)
    top_combined = candidates[:TOP_GLOBAL]

    print("  Overall top latents (layer:latent_idx, Δfreq):")
    print(
        "    " + ", ".join(
            f"L{L}:{idx} (Δ={v:.3f})" for (L, idx, v) in top_combined
        )
    )

    # Build steering_latents list from decoder rows
    steering_latents: List[Tuple[int, int, torch.Tensor]] = []
    for layer0, latent_idx, diff_val in top_combined:
        sae = sae_cache[layer0]
        direction = sae.W_dec[latent_idx].detach().to(
            model.cfg.device, dtype=model.cfg.dtype
        )
        steering_latents.append((layer0, latent_idx, direction))

    return steering_latents

# ────────────────────────────────────────────────────────────────
# Load shared 50-sample test set
# ────────────────────────────────────────────────────────────────

def load_test_set(test_dir: Path) -> List[Dict[str, Any]]:
    files = [
        test_dir / "harmful_refusals_50.json",
        test_dir / "harmful_compliance_50.json",
        test_dir / "unharmful_refusals_50.json",
        test_dir / "unharmful_compliance_50.json",
    ]
    out: List[Dict[str, Any]] = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON array")

        fname = path.name.lower()
        if fname.startswith("harmful_"):
            harm_label = "harmful"
        elif fname.startswith("unharmful_"):
            harm_label = "unharmful"
        else:
            raise ValueError(f"Cannot infer harmful/unharmful from {path.name}")

        was_refusal = 1 if "refusal" in fname else 0

        for r in rows:
            out.append(
                {
                    "prompt": r.get("prompt", ""),
                    "prompt_harm_label": harm_label,
                    "original_response": r.get("response", ""),
                    "was_refusal": was_refusal,
                }
            )
    return out

print("\n=== Loading shared 50-sample test set ===")
test_records = load_test_set(TEST_DIR)
print(f"Loaded {len(test_records)} test records.")
clear_cuda()

# ────────────────────────────────────────────────────────────────
# Main loop: 13 splits × SAE latents × alpha sweep
# ────────────────────────────────────────────────────────────────

print("\n=== SAE 13-splits alpha sweep ===")

split_files = sorted(SPLITS_DIR.glob("*.json"))
if len(split_files) != 13:
    print(f"[WARN] Expected 13 splits in {SPLITS_DIR}, found {len(split_files)}")

for split_path in split_files:
    split_name = split_path.stem
    print(f"\n--- Split: {split_name} ---")

    with split_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{split_path} must contain a JSON array")

    # HR = harmful + refused, BC = unharmful + compliant
    hr_prompts = [
        r["prompt"]
        for r in rows
        if (r.get("prompt_harm_label", "").strip().lower() == "harmful")
        and int(r.get("was_refusal", 0)) == 1
    ]
    bc_prompts = [
        r["prompt"]
        for r in rows
        if (r.get("prompt_harm_label", "").strip().lower() == "unharmful")
        and int(r.get("was_refusal", 0)) == 0
    ]

    print(f"  HR prompts: {len(hr_prompts)}   BC prompts: {len(bc_prompts)}")
    # As per your setup, expect exactly 47 of each
    if len(hr_prompts) != 47 or len(bc_prompts) != 47:
        raise RuntimeError(
            f"{split_name}: expected exactly 47 HR and 47 BC prompts, "
            f"got {len(hr_prompts)} HR and {len(bc_prompts)} BC."
        )

    # Build SAE-based steering latents: 10 best latents across layers 9/20/31
    steering_latents = build_sae_latents_for_split(
        hr_prompts, bc_prompts, split_name=split_name
    )

    split_out_dir = OUT_ROOT / split_name
    split_out_dir.mkdir(parents=True, exist_ok=True)

    # Alpha sweep: for each alpha, attach hooks for all 10 latents and generate
    for alpha in ALPHAS:
        alpha_int = int(alpha)
        out_path = split_out_dir / f"alpha{alpha_int}.json"
        if out_path.exists():
            print(f"  [skip] {out_path} already exists")
            continue

        print(f"  Alpha = {alpha_int}")
        hooks = build_latent_hooks_for_split(
            steering_latents,
            alpha=alpha,
        )

        gens = generate_for_records(
            test_records,
            fwd_hooks=hooks,
            desc=f"{split_name} α={alpha_int}",
        )

        assert len(gens) == len(test_records)
        items: List[Dict[str, Any]] = []
        for rec, gen in zip(test_records, gens):
            items.append(
                {
                    "prompt": rec["prompt"],
                    "prompt_harm_label": rec["prompt_harm_label"],
                    "original_response": rec["original_response"],
                    "was_refusal": int(rec["was_refusal"]),
                    "alpha": float(alpha),
                    "model_response": gen,
                }
            )

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

        print(f"    → wrote {len(items)} rows to {out_path}")
        del gens, items
        clear_cuda()

print("\n✓ SAE sweep complete.")
print(f"Results written under: {OUT_ROOT.resolve()}")
