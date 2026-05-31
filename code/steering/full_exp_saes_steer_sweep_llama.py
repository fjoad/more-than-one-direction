#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_llama_sae_layer15_sweep_on_50samples.py

Llama-3.1-8B-Instruct + AndyRDT SAE (layer 15, resid_post)

For each of the 11 splits in data/refusal_13splits/*.json
(excluding CocoNot_all and SorryBench_all):

  - Each file already contains exactly 47 "HR" prompts and 47 "BC" prompts:
        HR = harmful + refused
        BC = unharmful + compliant
    We DO NOT SAMPLE. We use all 64 rows in that file.

  - SAE layer:
      * Single SAE: resid_post at layer 15
      * capture resid_post at layer 15, pos = -2, for HR47 & BC47
      * encode with the AndyRDT SAE (HF repo andyrdt/saes-llama-3.1-8b-instruct)
      * compute fraction active on HR vs BC (active = code > 0)
      * pick top-10 latents in that SAE by diff = freq_HR - freq_BC

  - Each selected latent's decoder row (from the SAE) is a direction.
    Directions are raw W_dec rows (no unit norm).

  - Steering:
      * Steering layer = 15
      * For each direction, attach hooks at resid_pre, resid_mid, resid_post
        on layer 15, ALL TOKENS (every position) get the addition.

  - For each split & each alpha ∈ {0.0, 1.0, 1.05, 1.1, 1.15, 1.2}:
      * attach hooks (10 latents × 3 sites = 30 hooks total)
      * generate on the shared 50-sample test set and save:

        generations/llama_sae_layer15_sweep/<split_name>/alpha<alpha_tag>.json

        where alpha_tag is the alpha with '.' replaced by 'p', e.g. 1.05 → "1p05"

    Each JSON file is a list of dicts:
      {
        "prompt": <str>,
        "prompt_harm_label": "harmful" | "unharmful",
        "original_response": <str>,
        "was_refusal": 0 | 1,
        "alpha": <float>,
        "model_response": <str>
      }

Total: 11 splits × 6 alphas = 66 output files.
"""

import os
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from huggingface_hub import hf_hub_download

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if "cuda" in DEVICE else torch.float32

# Split files
SPLITS_DIR = Path("data") / "refusal_13splits"

# Shared 50-sample test set (4 × 50 samples)
TEST_DIR = Path("data") / "4splits_combined" / "50samples"

# SAE details (AndyRDT repo for Llama 3.1 8B Instruct)
SAE_REPO_ID  = "andyrdt/saes-llama-3.1-8b-instruct"
SAE_FOLDER   = "resid_post_layer_15/trainer_1"
SAE_FILENAME = "ae.pt"

SAE_LAYER    = 15   # SAE lives in resid_post at layer 15
HOOK_LAYER   = SAE_LAYER  # steer at same block: resid_pre/mid/post on layer 15

POS_SLICE    = -2   # second-to-last token
EPS          = 0.0  # active = z > EPS

TOP_K        = 10   # top-10 latents overall (single SAE)

ALPHAS       = [0.0, 1.0, 1.05, 1.1, 1.15, 1.2]

MAX_LEN      = 128
MAX_NEW      = 64
GEN_BATCH    = 4    # generation batch size

OUT_ROOT     = Path("generations") / "llama_sae_layer15_sweep"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# Hygiene
# ────────────────────────────────────────────────────────────────

def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ────────────────────────────────────────────────────────────────
# Model & tokenizer
# ────────────────────────────────────────────────────────────────

print("=== Loading Llama-3.1-8B-Instruct model ===")
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH,
    device=DEVICE,
    n_devices=1,
    dtype=DTYPE,
)
model.eval()

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model & tokenizer loaded.")
clear_cuda()

# ────────────────────────────────────────────────────────────────
# Tokenisation & generation helpers
# ────────────────────────────────────────────────────────────────

def tokenize_instructions_llama_chat(
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
        toks = tokenize_instructions_llama_chat(batch_prompts, max_length=MAX_LEN)
        outs = _generate_with_hooks(
            toks,
            max_tokens_generated=MAX_NEW,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(outs)
        del toks, outs
        clear_cuda()
    return generations

# ────────────────────────────────────────────────────────────────
# Activation caching
# ────────────────────────────────────────────────────────────────

def cache_resid_layer_pos(
    prompts: List[str],
    layer: int,
    act_name: str,
    pos_slice: int = POS_SLICE,
    desc: str = "cache",
) -> torch.Tensor:
    """
    Cache one residual stream (resid_pre or resid_post) at `layer` and a single
    position `pos_slice` for given prompts.
    Returns [N, d_model] on CPU float32.
    """
    hook_name = utils.get_act_name(act_name, layer)
    chunks: List[torch.Tensor] = []

    if not prompts:
        return torch.empty(0, model.cfg.d_model, dtype=torch.float32)

    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc=f"Cache {desc}", unit="batch"):
        batch_prompts = prompts[i : i + GEN_BATCH]
        toks = tokenize_instructions_llama_chat(batch_prompts, max_length=MAX_LEN)
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

    return torch.cat(chunks, dim=0)

# ─────────────────────────── SAE module & loader ───────────────────────────

class AndyrdtSAE(nn.Module):
    """
    Minimal wrapper around andyrdt's ae.pt for resid_post_layer_15.
    Uses:
      - W_enc: [d_model, d_sae]
      - W_dec: [d_sae, d_model]
      - b_enc: [d_sae]
      - b_dec: [d_model]
    Encode: z = ReLU(x @ W_enc + b_enc)
    Decode: x_hat = z @ W_dec + b_dec
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
        super().__init__()
        # Raw weights from checkpoint
        enc_w_raw = state_dict["encoder.weight"]   # [d_sae, d_model]
        dec_w_raw = state_dict["decoder.weight"]   # [d_model, d_sae]
        b_enc_raw = state_dict["encoder.bias"]     # [d_sae]
        b_dec_raw = state_dict["b_dec"]            # [d_model]

        # Store in convenient shapes
        self.W_enc = nn.Parameter(
            enc_w_raw.to(device=device, dtype=dtype).T, requires_grad=False
        )  # [d_model, d_sae]
        self.W_dec = nn.Parameter(
            dec_w_raw.to(device=device, dtype=dtype).T, requires_grad=False
        )  # [d_sae, d_model]
        self.b_enc = nn.Parameter(
            b_enc_raw.to(device=device, dtype=dtype), requires_grad=False
        )  # [d_sae]
        self.b_dec = nn.Parameter(
            b_dec_raw.to(device=device, dtype=dtype), requires_grad=False
        )  # [d_model]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        pre = x @ self.W_enc + self.b_enc
        return torch.relu(pre)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec


print("\n=== Loading SAE (layer 15 resid_post, andyrdt) ===")
ae_path = hf_hub_download(
    repo_id=SAE_REPO_ID,
    filename=f"{SAE_FOLDER}/{SAE_FILENAME}",
)
state = torch.load(ae_path, map_location="cpu")

sae = AndyrdtSAE(state, device=model.cfg.device, dtype=model.cfg.dtype).eval()
d_model, d_sae = sae.W_enc.shape
print(f"SAE loaded: d_model={d_model}, d_sae={d_sae}")
clear_cuda()

# ────────────────────────────────────────────────────────────────
# SAE latents per split (single layer, resid_post L=15)
# ────────────────────────────────────────────────────────────────

def build_sae_directions_for_split(
    hr_prompts: List[str],
    bc_prompts: List[str],
    split_name: str,
) -> Tuple[List[torch.Tensor], List[Tuple[int, float]]]:
    """
    For one split (47 HR + 47 BC):

      - capture resid_post at layer=15, pos=-2 for HR & BC prompts
      - encode with SAE at layer 15
      - compute freq_HR, freq_BC where "active" = code > EPS
      - find top-10 latents by diff = freq_HR - freq_BC

      - For each selected latent, direction = decoder row W_dec[latent_idx]
        from the SAE for that layer (no normalization).

    Returns:
      directions: list of 10 direction tensors [d_model]
      top_latents: list of (latent_idx, diff_value)
    """
    print(f"  Building SAE directions for split {split_name}")

    acts_hr = cache_resid_layer_pos(
        hr_prompts,
        layer=SAE_LAYER,
        act_name="resid_post",
        pos_slice=POS_SLICE,
        desc=f"{split_name} HR SAE L{SAE_LAYER}",
    )
    acts_bc = cache_resid_layer_pos(
        bc_prompts,
        layer=SAE_LAYER,
        act_name="resid_post",
        pos_slice=POS_SLICE,
        desc=f"{split_name} BC SAE L{SAE_LAYER}",
    )

    print(f"    HR acts shape: {tuple(acts_hr.shape)}")
    print(f"    BC acts shape: {tuple(acts_bc.shape)}")

    x_all = torch.cat([acts_hr, acts_bc], dim=0).to(model.cfg.device, dtype=model.cfg.dtype)
    labels = torch.cat(
        [
            torch.ones(len(acts_hr), dtype=torch.bool),
            torch.zeros(len(acts_bc), dtype=torch.bool),
        ],
    ).to(model.cfg.device)

    with torch.no_grad():
        z_all = sae.encode(x_all)               # [N, d_sae]
        active = (z_all > EPS)                  # boolean

        hr_mask = labels
        bc_mask = ~labels

        freq_hr = active[hr_mask].float().mean(dim=0)   # [d_sae]
        freq_bc = active[bc_mask].float().mean(dim=0)
        diff = freq_hr - freq_bc

    vals, idxs = torch.topk(diff, TOP_K)
    idxs = idxs.cpu()
    vals = vals.cpu()

    print("    Top-10 refusal-moving latents (idx, Δfreq):")
    for i, v in zip(idxs.tolist(), vals.tolist()):
        print(f"      latent {i:6d}  Δfreq={v:.4f}")

    directions: List[torch.Tensor] = []
    top_latents: List[Tuple[int, float]] = []

    for latent_idx, diff_val in zip(idxs.tolist(), vals.tolist()):
        vec = sae.W_dec[latent_idx].detach().to(model.cfg.device, dtype=model.cfg.dtype)
        directions.append(vec)
        top_latents.append((latent_idx, float(diff_val)))

    del acts_hr, acts_bc, x_all, z_all, active
    clear_cuda()

    return directions, top_latents

# ───────────────────── steering hook (EVERY TOKEN) ─────────────────────

def activation_addition_hook(
    activation: Float[torch.Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[torch.Tensor, "d_act"],
    alpha: float,
):
    d = direction.to(activation.device, dtype=activation.dtype)
    # activation shape: [B, T, D] – broadcast direction to all tokens
    return activation + alpha * d.view(1, 1, -1)


def build_sae_hooks(
    directions: List[torch.Tensor],
    layer: int,
    alpha: float,
) -> List[Tuple[str, callable]]:
    """
    For each direction, attach hooks at resid_pre/mid/post on the given layer.
    Token mode = "every": add to all tokens.
    """
    act_points = ["resid_pre", "resid_mid", "resid_post"]
    hooks: List[Tuple[str, callable]] = []

    for direction in directions:
        # Use default args in lambda to avoid late-binding bug
        hook_fn = lambda act, hook, d=direction, a=alpha: activation_addition_hook(
            act, hook, direction=d, alpha=a
        )
        for act_name in act_points:
            hook_name = utils.get_act_name(act_name, layer)
            hooks.append((hook_name, hook_fn))

    return hooks

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
# Main loop: exact split list (skip CocoNot_all & SorryBench_all)
# ────────────────────────────────────────────────────────────────

print("\n=== Llama SAE (layer 15) alpha sweep over fixed splits ===")

split_files_all = sorted(SPLITS_DIR.glob("*.json"))

EXCLUDE_FILES = {"CocoNot_all.json", "SorryBench_all.json"}

split_files = [p for p in split_files_all if p.name not in EXCLUDE_FILES]

print("Using splits:")
for p in split_files:
    print(f"  - {p.name}")

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

    # Build SAE-based directions: 10 best latents from SAE layer 15
    directions, top_latents = build_sae_directions_for_split(
        hr_prompts, bc_prompts, split_name=split_name
    )

    split_out_dir = OUT_ROOT / split_name
    split_out_dir.mkdir(parents=True, exist_ok=True)

    # Alpha sweep: for each alpha, attach hooks for all 10 latents and generate
    for alpha in ALPHAS:
        # Make a safe tag for filenames: 1.05 -> "1p05"
        alpha_tag = str(alpha).replace(".", "p")
        out_path = split_out_dir / f"alpha{alpha_tag}.json"
        if out_path.exists():
            print(f"  [skip] {out_path} already exists")
            continue

        print(f"  Alpha = {alpha}  (file tag: {alpha_tag})")
        hooks = build_sae_hooks(
            directions,
            layer=HOOK_LAYER,
            alpha=alpha,
        )

        gens = generate_for_records(
            test_records,
            fwd_hooks=hooks,
            desc=f"{split_name} α={alpha}",
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

print("\n✓ Llama SAE sweep complete.")
print(f"Results written under: {OUT_ROOT.resolve()}")
