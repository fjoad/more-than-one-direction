#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_neel_ablate_gemma_global_extended.py

Gemma-2 Neel-style **global ablation** experiment on the extended HR/BC splits.

  • Model: google/gemma-2-9b-it (TransformerLens)

  • Global Neel direction training:
      File: data/train_subset64_working.json
      - HR32 = first 32 prompts
      - BC32 = next 32 prompts
      - Activations: resid_pre, layer 20, pos = -2
      - dir_global = unit(mean(HR32) - mean(BC32))

  • Evaluation splits (extended):
      data/refusal_13splits_extended/*.json
        e.g. WildGuard_all_HR128_BC128.json, CocoNot_cat_*_HR*_BC*.json, ...

      Each record is expected to have:
        - "prompt"
        - "prompt_harm_label": "harmful" / "unharmful" (or synonyms)
        - "was_refusal": 0 or 1   (base-model label for HR vs BC)

  • Ablation (Neel-style):
      - Take dir_global as above.
      - At every layer l = 0 .. n_layers-1:
           resid_pre[l], resid_mid[l], resid_post[l]
        project out dir_global from the residual stream.

  • Outputs:
      generations/neel_ablate_gemma_global_extended/<split_stem>/neel_ablate_global.json

    Each JSON file is a list of dicts:
      {
        "prompt": <str>,
        "prompt_harm_label": "harmful" | "unharmful",
        "was_refusal": 0 | 1,
        "model_response": <str>
      }

You can then run your WildGuard evaluator on these outputs and compute metrics
using the same machinery as for other experiments.
"""

import gc
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import einops
import torch
from jaxtyping import Float, Int
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformer_lens.utilities import devices as tl_devices
import functools

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = "google/gemma-2-9b-it"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

# Global Neel-direction training file (64 prompts: 32 harmful, 32 benign)
NEEL_TRAIN_JSON = Path("data") / "train_subset64_working.json"

# Extended HR/BC splits for evaluation
SPLITS_DIR = Path("data") / "refusal_13splits_extended"

# Neel config (Gemma analogue of Llama L16)
NEEL_LAYER = 20
NEEL_POS = -2

MAX_LEN = 128
MAX_NEW = 64
GEN_BATCH = 4

OUT_ROOT = Path("generations") / "neel_ablate_gemma_extended_working"

# ────────────────────────────────────────────────────────────────
# Hygiene
# ────────────────────────────────────────────────────────────────
def clear_cuda():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    gc.collect()


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm().clamp_min(1e-12)


# ────────────────────────────────────────────────────────────────
# Model & tokenizer (Gemma-2-9B-IT)
# ────────────────────────────────────────────────────────────────
print("=== Loading Gemma-2-9B-IT model ===")

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
model.eval()

tokenizer: AutoTokenizer = model.tokenizer
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model & tokenizer loaded.")
clear_cuda()

# ────────────────────────────────────────────────────────────────
# IO helpers
# ────────────────────────────────────────────────────────────────
def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a top-level list.")
    return data

# ────────────────────────────────────────────────────────────────
# Tokenisation & generation helpers
# ────────────────────────────────────────────────────────────────
def tokenize_instructions_gemma_chat(
    instructions: List[str],
    max_length: int = MAX_LEN,
) -> Int[torch.Tensor, "batch seq"]:
    """
    Wrap prompts in Gemma chat template and tokenize (left-padded).
    """
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
    Manual greedy generation with forward hooks.
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

# ────────────────────────────────────────────────────────────────
# Activation caching for direction construction
# ────────────────────────────────────────────────────────────────
def cache_resid_pre_layer_pos(
    prompts: List[str],
    layer: int,
    pos_slice: int = -2,
    desc: str = "cache",
) -> torch.Tensor:
    """
    Cache resid_pre at `layer` and a single position `pos_slice` for given prompts.
    Returns [N, d_model] on CPU float32.
    """
    hook_name = utils.get_act_name("resid_pre", layer)
    chunks: List[torch.Tensor] = []

    if not prompts:
        return torch.empty(0, model.cfg.d_model, dtype=torch.float32)

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
        # cache[hook_name]: [batch, 1, d_model] after pos_slice
        acts = cache[hook_name].squeeze(1).to("cpu", dtype=torch.float32).contiguous()
        chunks.append(acts)
        del cache, toks, acts
        clear_cuda()

    return torch.cat(chunks, dim=0)

# ────────────────────────────────────────────────────────────────
# Ablation hooks (Neel-style)
# ────────────────────────────────────────────────────────────────
def direction_ablation_hook(
    activation: Float[torch.Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[torch.Tensor, "d_act"],
):
    """
    Project out `direction` from the activation at every position.
    """
    d = direction.to(activation.device, dtype=activation.dtype)
    # proj_coeff: [..., 1]
    proj_coeff = einops.einsum(
        activation,
        d.view(-1, 1),
        "... d_act, d_act one -> ... one",
    )
    proj = proj_coeff * d  # broadcast along '...'
    return activation - proj


def hooks_ablation_all_layers(
    direction: torch.Tensor,
) -> List[Tuple[str, callable]]:
    """
    Ablate the given direction at all layers and resid_pre/mid/post.
    """
    intervention_layers = list(range(model.cfg.n_layers))
    act_points = ["resid_pre", "resid_mid", "resid_post"]
    hook_fn = functools.partial(direction_ablation_hook, direction=direction)

    hooks: List[Tuple[str, callable]] = []
    for l in intervention_layers:
        for act_name in act_points:
            hook_name = utils.get_act_name(act_name, l)
            hooks.append((hook_name, hook_fn))
    return hooks

# ────────────────────────────────────────────────────────────────
# 1) Build global Neel HR–BC direction from train_subset64_working.json
# ────────────────────────────────────────────────────────────────
print("\n=== Building global Neel HR–BC direction (Gemma, L20) ===")

neel_recs = load_json_list(NEEL_TRAIN_JSON)
if len(neel_recs) < 64:
    raise RuntimeError(f"{NEEL_TRAIN_JSON} expected at least 64 rows, got {len(neel_recs)}")

hr32_prompts = [r["prompt"] for r in neel_recs[:32]]
bc32_prompts = [r["prompt"] for r in neel_recs[32:64]]

acts_hr32 = cache_resid_pre_layer_pos(
    hr32_prompts, layer=NEEL_LAYER, pos_slice=NEEL_POS, desc="Neel HR32"
)
acts_bc32 = cache_resid_pre_layer_pos(
    bc32_prompts, layer=NEEL_LAYER, pos_slice=NEEL_POS, desc="Neel BC32"
)

dir_global = unit(acts_hr32.mean(dim=0) - acts_bc32.mean(dim=0)).to(
    model.cfg.device, dtype=model.cfg.dtype
)

del acts_hr32, acts_bc32
clear_cuda()
print("Global Neel HR–BC direction built.")

hooks_global_ablate = hooks_ablation_all_layers(dir_global)

# ────────────────────────────────────────────────────────────────
# 2) Load extended splits and run ablation
# ────────────────────────────────────────────────────────────────
print("\n=== Running Neel-style GLOBAL ablation on extended splits ===")

split_files = sorted(SPLITS_DIR.glob("*.json"))
if not split_files:
    raise SystemExit(f"No split files found under {SPLITS_DIR}")

print("Using extended splits:")
for p in split_files:
    print(f"  - {p.name}")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

for split_path in split_files:
    split_stem = split_path.stem  # e.g. 'WildGuard_all_HR128_BC128'
    print(f"\n--- Split: {split_stem} ---")

    records = load_json_list(split_path)
    print(f"  {len(records)} prompts in this split.")

    split_out_dir = OUT_ROOT / split_stem
    split_out_dir.mkdir(parents=True, exist_ok=True)

    out_path = split_out_dir / "neel_ablate_global.json"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists")
        continue

    gens = generate_for_records(
        records,
        fwd_hooks=hooks_global_ablate,
        desc=f"{split_stem} – Neel GLOBAL ABLATE",
    )

    assert len(gens) == len(records)
    out_items: List[Dict[str, Any]] = []
    for rec, gen in zip(records, gens):
        out_items.append(
            {
                "prompt": rec.get("prompt", ""),
                "prompt_harm_label": rec.get("prompt_harm_label", ""),
                "was_refusal": int(rec.get("was_refusal", 0)),
                "model_response": gen,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    print(f"  → wrote {len(out_items)} rows to {out_path}")
    del gens, out_items
    clear_cuda()

print("\n✓ Global ablation runs complete.")
print(f"Results written under: {OUT_ROOT.resolve()}")
