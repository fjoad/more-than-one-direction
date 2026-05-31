#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_neel_ablate_qwen_11dirs.py

Qwen-1_8B-Chat Neel-style **ablation** experiment (11 splits).

  • Model:  Qwen/Qwen-1_8B-Chat  (TransformerLens)
  • Direction: single HR–BC Neel direction trained from:
        data/train_subset64_working.json
      - HR32 = first 32 prompts
      - BC32 = next 32 prompts
      - Activations: resid_pre, layer 14, pos = -1
      - dir_neel = unit(mean(HR32) - mean(BC32))   (Qwen activations)

  • Splits: data/refusal_13splits/*.json
      - EXCLUDE: CocoNot_all.json, SorryBench_all.json
      - For each remaining split, we generate ablated responses for
        *all* prompts.

  • Ablation (Neel-style):
      - Take dir_neel as above.
      - At every layer l = 0 .. n_layers-1:
           resid_pre[l], resid_mid[l], resid_post[l]
        project out dir_neel from the residual stream.

  • Outputs:
      generations/neel_ablate_qwen_11dirs/<split_name>/neel_ablate.json

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

import einops
import torch
from jaxtyping import Float, Int
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = "Qwen/Qwen-1_8B-Chat"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if "cuda" in DEVICE else torch.float32

# Neel-direction training file (64 prompts, 32 harmful then 32 unharmful)
NEEL_TRAIN_JSON = Path("data/train_subset64_working.json")

# Splits for evaluation
SPLITS_DIR = Path("data") / "refusal_13splits"
EXCLUDE_FILES = {"CocoNot_all.json", "SorryBench_all.json"}

# Neel config for Qwen (based on earlier diagnostics)
NEEL_LAYER = 14
NEEL_POS = -1

MAX_LEN = 128
MAX_NEW = 50
GEN_BATCH = 24

OUT_ROOT = Path("generations") / "neel_ablate_qwen_11dirs"

# Qwen chat template
QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# ────────────────────────────────────────────────────────────────
# Hygiene helpers
# ────────────────────────────────────────────────────────────────
def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm().clamp_min(1e-12)


# ────────────────────────────────────────────────────────────────
# Model & tokenizer (Qwen-1_8B-Chat)
# ────────────────────────────────────────────────────────────────
print("=== Loading Qwen-1_8B-Chat model ===")

model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH,
    device=DEVICE,
    dtype=DTYPE,
    default_padding_side="left",
    fp16=True,
)
model.eval()

# Use the TL-attached tokenizer; set padding + pad token as in the demo
hf_tok: AutoTokenizer = model.tokenizer
hf_tok.padding_side = "left"
if hf_tok.pad_token is None:
    hf_tok.pad_token = "<|extra_0|>"

print("Model & tokenizer loaded.")
clear_cuda()

# ────────────────────────────────────────────────────────────────
# Tokenization & generation helpers
# ────────────────────────────────────────────────────────────────
def tokenize_instructions_qwen_chat(
    instructions: List[str],
    max_length: int = MAX_LEN,
) -> Int[torch.Tensor, "batch seq"]:
    """
    Wrap prompts in Qwen chat template and tokenize (left-padded).
    """
    prompts = [
        QWEN_CHAT_TEMPLATE.format(instruction=inst)
        for inst in instructions
    ]
    toks = hf_tok(
        prompts,
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
    out = hf_tok.batch_decode(gen_suffix, skip_special_tokens=True)
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
        toks = tokenize_instructions_qwen_chat(batch_prompts, max_length=MAX_LEN)
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
# Activation caching (resid_pre at NEEL_LAYER, NEEL_POS)
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
        toks = tokenize_instructions_qwen_chat(batch_prompts, max_length=MAX_LEN)
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


# ────────────────────────────────────────────────────────────────
# Steering / ablation hook fns (Neel-style)
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
    proj = einops.einsum(
        activation,
        d.view(-1, 1),
        "... d_act, d_act single -> ... single",
    ) * d
    return activation - proj


def hooks_ablation_all_layers(
    direction: torch.Tensor,
) -> List[Tuple[str, callable]]:
    """
    Ablate the given direction at all layers and resid_pre/mid/post.
    """
    intervention_layers = list(range(model.cfg.n_layers))
    act_points = ["resid_pre", "resid_mid", "resid_post"]
    import functools
    hook_fn = functools.partial(direction_ablation_hook, direction=direction)

    hooks: List[Tuple[str, callable]] = []
    for l in intervention_layers:
        for act_name in act_points:
            hook_name = utils.get_act_name(act_name, l)
            hooks.append((hook_name, hook_fn))
    return hooks


# ────────────────────────────────────────────────────────────────
# IO helpers
# ────────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────────
# 1) Build Neel HR–BC direction (train_subset64_working.json)
# ────────────────────────────────────────────────────────────────
print(f"\n=== Building Neel HR–BC direction from train_subset64_working.json (Qwen, L{NEEL_LAYER}, pos={NEEL_POS}) ===")
with NEEL_TRAIN_JSON.open("r", encoding="utf-8") as f:
    neel_recs = json.load(f)
if len(neel_recs) < 64:
    raise RuntimeError(f"{NEEL_TRAIN_JSON} expected at least 64 rows.")

hr32_prompts = [r["prompt"] for r in neel_recs[:32]]
bc32_prompts = [r["prompt"] for r in neel_recs[32:64]]

acts_hr32 = cache_resid_pre_layer_pos(
    hr32_prompts, layer=NEEL_LAYER, pos_slice=NEEL_POS, desc="Neel HR32"
)
acts_bc32 = cache_resid_pre_layer_pos(
    bc32_prompts, layer=NEEL_LAYER, pos_slice=NEEL_POS, desc="Neel BC32"
)

# Optional sanity: HR vs BC projections at chosen layer/pos
print("\n=== Sanity check: HR vs BC projections at chosen layer/pos ===")
d_tmp = unit(acts_hr32.mean(dim=0) - acts_bc32.mean(dim=0))
acts_hr32_dev = acts_hr32.to(d_tmp.device, dtype=d_tmp.dtype)
acts_bc32_dev = acts_bc32.to(d_tmp.device, dtype=d_tmp.dtype)
proj_hr = (acts_hr32_dev @ d_tmp).detach().cpu()
proj_bc = (acts_bc32_dev @ d_tmp).detach().cpu()
print(f"HR projections: mean={proj_hr.mean().item():.4f}, std={proj_hr.std().item():.4f}")
print(f"BC projections: mean={proj_bc.mean().item():.4f}, std={proj_bc.std().item():.4f}")
print(f"Separation (HR_mean - BC_mean) = {proj_hr.mean().item() - proj_bc.mean().item():.4f}")

# Final Neel direction on model's device/dtype
dir_neel = d_tmp.to(model.cfg.device, dtype=model.cfg.dtype)

del acts_hr32, acts_bc32, acts_hr32_dev, acts_bc32_dev, d_tmp
clear_cuda()
print("Neel HR–BC direction built.")

hooks_neel_ablate = hooks_ablation_all_layers(dir_neel)

# ────────────────────────────────────────────────────────────────
# 2) Load splits (11 of them) and run ablation
# ────────────────────────────────────────────────────────────────
print("\n=== Running Neel-style ablation on splits (excluding CocoNot_all & SorryBench_all) ===")
split_files_all = sorted(SPLITS_DIR.glob("*.json"))
split_files = [p for p in split_files_all if p.name not in EXCLUDE_FILES]

print("Using splits:")
for p in split_files:
    print(f"  - {p.name}")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

for split_path in split_files:
    split_name = split_path.stem
    print(f"\n--- Split: {split_name} ---")
    records = load_split(split_path)
    print(f"  {len(records)} prompts in this split.")

    split_out_dir = OUT_ROOT / split_name
    split_out_dir.mkdir(parents=True, exist_ok=True)

    out_path = split_out_dir / "neel_ablate.json"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists")
        continue

    gens = generate_for_records(
        records,
        fwd_hooks=hooks_neel_ablate,
        desc=f"{split_name} – Neel ABLATE (Qwen)",
    )
    save_generations(out_path, records, gens)
    del gens
    clear_cuda()

    print(f"  ✓ Finished Neel ABLATE for split {split_name}. Outputs in {split_out_dir}")

print("\n✓ All 11 splits done. Results written under:")
print(f"  {OUT_ROOT.resolve()}")
