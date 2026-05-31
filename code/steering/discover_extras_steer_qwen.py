#!/usr/bin/env python3
"""
discover_extras_steer_qwen.py — companion to discover_extras_qwen.py.

Same 6 directions (t64H_ourBC + wgrandom_seed{1..5}), same shared BC, same
test pools — but runs STEERING at α=25 (Qwen's locked saturating alpha)
instead of ablation.

Builds directions from scratch (deterministic given the seeds) so it can run
in parallel with discover_extras_qwen.py without depending on its output.
The hash gate confirms bytes-identical directions across both scripts —
that means any difference in (ablation effect) vs (steering effect) is
purely due to hook type, not direction identity.

Outputs:
  generations/discover_qwen/responses/steer_<dir_name>_alpha25/<split>.json
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

import einops
import torch
from jaxtyping import Float
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer

# ────────────────────────────────────────────────────────────────
# Config (must match discover_extras_qwen.py exactly)
# ────────────────────────────────────────────────────────────────
EXP_V10 = Path(__file__).resolve().parent.parent
LEGACY_TRAIN64 = EXP_V10 / "data" / "legacy" / "train_subset64_working.json"
PER_SPLIT_POOLS_DIR = EXP_V10 / "data" / "training_pools" / "qwen" / "per_split_simple"
JUDGED_DIR = EXP_V10 / "data" / "baselines" / "qwen" / "judged"
DIRECTIONS_ROOT = EXP_V10 / "data" / "directions" / "qwen"
RESP_ROOT = EXP_V10 / "generations" / "discover_qwen" / "responses"

MODEL_PATH = "Qwen/Qwen-1_8B-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

LAYER = 14
POS = -1

GEN_BATCH = 32
MAX_LEN = 128
MAX_NEW = 50

TEST_CAP = 25
ALPHA = 25                 # Qwen's locked saturating value (per Table 2)

ALL_SPLITS = [
    "WildGuard_all", "XSTest_all",
    "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
    "CocoNot_Safety", "CocoNot_Unsupported",
    "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
    "SorryBench_Inappropriate", "SorryBench_Advice",
]

WG_RANDOM_SEEDS = [1, 2, 3, 4, 5]
N_H = 32

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
_T0 = time.monotonic()
def _ts() -> str:
    return f"[{datetime.now().strftime('%H:%M:%S')} +{time.monotonic() - _T0:6.1f}s]"
def tprint(*args, **kwargs):
    print(_ts(), *args, flush=True, **kwargs)

def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm().clamp_min(1e-12)

def sha256_tensor(x: torch.Tensor) -> str:
    arr = x.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()


# ────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────
tprint(f"=== Loading {MODEL_PATH} ===")
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH, device=DEVICE, dtype=DTYPE,
    default_padding_side="left", fp16=True,
)
model.eval()
hf_tok: AutoTokenizer = model.tokenizer
hf_tok.padding_side = "left"
if hf_tok.pad_token is None:
    hf_tok.pad_token = "<|extra_0|>"
N_LAYERS = model.cfg.n_layers
D_MODEL = model.cfg.d_model
tprint(f"Loaded.  n_layers={N_LAYERS}  d_model={D_MODEL}")
clear_cuda()


# ────────────────────────────────────────────────────────────────
# Tokenize + generate
# ────────────────────────────────────────────────────────────────
def tokenize_qwen_chat(instructions: List[str], max_length: int = MAX_LEN):
    prompts = [QWEN_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    return hf_tok(
        prompts, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    ).input_ids.to(model.cfg.device)


def generate_with_hooks(toks, max_tokens_generated: int,
                        fwd_hooks: List[Tuple[str, Callable]] = ()):
    B, T = toks.shape
    all_toks = torch.zeros((B, T + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :T] = toks
    for i in range(max_tokens_generated):
        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(all_toks[:, :T + i])
                next_tokens = logits[:, -1, :].argmax(dim=-1)
                all_toks[:, T + i] = next_tokens
    return hf_tok.batch_decode(all_toks[:, T:], skip_special_tokens=True)


def generate_for_records(records: List[dict], fwd_hooks: List[Tuple[str, Callable]],
                         desc: str) -> List[str]:
    prompts = [r["prompt"] for r in records]
    out: List[str] = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc=desc, unit="batch", leave=False):
        toks = tokenize_qwen_chat(prompts[i:i + GEN_BATCH], max_length=MAX_LEN)
        out.extend(generate_with_hooks(toks, MAX_NEW, fwd_hooks))
        del toks
        clear_cuda()
    return out


def cache_resid_pre(prompts: List[str], layer: int, pos_slice: int, desc: str):
    hook_name = utils.get_act_name("resid_pre", layer)
    chunks = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc=f"cache {desc}",
                  unit="batch", leave=False):
        toks = tokenize_qwen_chat(prompts[i:i + GEN_BATCH], max_length=MAX_LEN)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                toks, names_filter=lambda n: n == hook_name,
                pos_slice=pos_slice, stop_at_layer=layer + 1,
            )
        chunks.append(cache[hook_name].squeeze(1).to("cpu", dtype=torch.float32).contiguous())
        del cache, toks
        clear_cuda()
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, D_MODEL)


def make_steer_hook(direction: torch.Tensor, alpha: float):
    d = direction.to(dtype=DTYPE, device=DEVICE)
    def hook_fn(activation: Float[torch.Tensor, "... d_act"], hook: HookPoint):
        return activation + alpha * d
    return hook_fn


def steer_fwd_hooks(direction: torch.Tensor, alpha: float, layer: int):
    return [(utils.get_act_name("resid_pre", layer), make_steer_hook(direction, alpha))]


# ────────────────────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────────────────────
def load_shared_bc() -> List[dict]:
    return json.load(open(PER_SPLIT_POOLS_DIR / "WildGuard_all" / "B.json"))[:N_H]


def load_train64_h() -> List[dict]:
    rows = json.load(open(LEGACY_TRAIN64))
    H = [r for r in rows if r["prompt_harm_label"] == "harmful"]
    return [{"prompt": r["prompt"], "prompt_harm_label": r["prompt_harm_label"]} for r in H]


def load_wg_refused_pool() -> List[dict]:
    rows = json.load(open(JUDGED_DIR / "WildGuard_all.json"))
    return [r for r in rows if r.get("is_refusal_wg") == 1]


# ────────────────────────────────────────────────────────────────
# Direction (rebuild if needed) + hash gate
# ────────────────────────────────────────────────────────────────
def build_or_load_direction(name: str, H: List[dict], B: List[dict]) -> Tuple[torch.Tensor, str]:
    out_dir = DIRECTIONS_ROOT / name
    d_path = out_dir / "d.pt"
    if d_path.exists():
        d = torch.load(d_path, map_location="cpu")
        h = sha256_tensor(d)
        tprint(f"  [{name}] loaded existing d.pt  hash={h[:12]}…")
        return d, h
    # Rebuild from scratch
    tprint(f"  [{name}] building from H={len(H)}, B={len(B)}")
    H_acts = cache_resid_pre([r["prompt"] for r in H], LAYER, POS, f"{name}.HR")
    B_acts = cache_resid_pre([r["prompt"] for r in B], LAYER, POS, f"{name}.BC")
    d = unit(H_acts.float().mean(dim=0) - B_acts.float().mean(dim=0))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(d, d_path)
    h = sha256_tensor(d)
    manifest = {
        "model": MODEL_PATH, "name": name,
        "layer": LAYER, "pos": POS,
        "n_h": len(H), "n_b": len(B),
        "direction_sha256": h, "d_model": D_MODEL,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    tprint(f"  [{name}] built + saved  hash={h[:12]}…")
    return d, h


# ────────────────────────────────────────────────────────────────
# Test pool reconstruction (mirrors discover_extras_qwen.py exactly)
# ────────────────────────────────────────────────────────────────
def build_test_pool(split: str, used_h_per_split: dict, used_b_prompts: set,
                    extra_exclude_for_wg: set) -> List[dict]:
    h_judged = json.load(open(JUDGED_DIR / f"{split}.json"))
    h_refused = [r for r in h_judged if r.get("is_refusal_wg") == 1]
    used_h = used_h_per_split[split]
    if split == "WildGuard_all":
        used_h = used_h | extra_exclude_for_wg
    h_test = [r for r in h_refused if r["prompt"] not in used_h][:TEST_CAP]
    if not h_test and split == "SorryBench_Advice":
        h_test = h_refused[:TEST_CAP]

    bc_judged = json.load(open(JUDGED_DIR / "_BC_source.json"))
    bc_complied = [r for r in bc_judged if r.get("is_refusal_wg") == 0]
    n_b_test = len(h_test)
    b_test = [r for r in bc_complied if r["prompt"] not in used_b_prompts][:n_b_test]

    out = []
    for r in h_test:
        out.append({"prompt": r["prompt"], "prompt_harm_label": r["prompt_harm_label"], "side": "H"})
    for r in b_test:
        out.append({"prompt": r["prompt"], "prompt_harm_label": r["prompt_harm_label"], "side": "B"})
    return out


def write_responses(records: List[dict], responses: List[str], out_path: Path,
                    intervention: str, dir_name: str, alpha: float, direction_hash: str):
    rows = []
    for r, resp in zip(records, responses):
        rows.append({
            "prompt": r["prompt"],
            "prompt_harm_label": r["prompt_harm_label"],
            "side": r["side"],
            "response": resp,
            "gen_meta": {
                "model": MODEL_PATH,
                "intervention": intervention,
                "dir_name": dir_name,
                "alpha": alpha,
                "direction_sha256": direction_hash,
                "layer_for_direction": LAYER,
                "pos_for_direction": POS,
                "max_new_tokens": MAX_NEW,
                "batch_size": GEN_BATCH,
                "decoding": "greedy",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main():
    bc_shared = load_shared_bc()
    used_h_recipe: dict = {}     # name -> used H prompts (for WG-train64 not relevant)
    directions = {}              # name -> (d on cuda, hash)

    tprint("=== Building/loading directions ===")
    h_t64 = load_train64_h()
    d, h = build_or_load_direction("t64H_ourBC", h_t64, bc_shared)
    directions["t64H_ourBC"] = (d.to(DEVICE, dtype=DTYPE), h)
    used_h_recipe["t64H_ourBC"] = {r["prompt"] for r in h_t64}

    wg_pool = load_wg_refused_pool()
    for seed in WG_RANDOM_SEEDS:
        rng = random.Random(seed)
        sampled = rng.sample(wg_pool, N_H)
        H = [{"prompt": r["prompt"], "prompt_harm_label": r["prompt_harm_label"]} for r in sampled]
        name = f"wgrandom_seed{seed}"
        d, h = build_or_load_direction(name, H, bc_shared)
        directions[name] = (d.to(DEVICE, dtype=DTYPE), h)
        used_h_recipe[name] = {r["prompt"] for r in H}

    tprint("=== Hash verification gate ===")
    for name, (d, h) in directions.items():
        d_disk = torch.load(DIRECTIONS_ROOT / name / "d.pt", map_location="cpu")
        h_disk = sha256_tensor(d_disk)
        man_h = json.load(open(DIRECTIONS_ROOT / name / "manifest.json"))["direction_sha256"]
        if not (h_disk == h == man_h):
            raise RuntimeError(f"HASH MISMATCH {name}")
        tprint(f"  ✓ {name}: hash {h[:16]}… verified")

    tprint("=== Building test pools (matching discover_extras_qwen.py) ===")
    used_b_prompts = {r["prompt"] for r in bc_shared}
    used_h_per_split = {
        split: {r["prompt"] for r in json.load(open(PER_SPLIT_POOLS_DIR / split / "H.json"))}
        for split in ALL_SPLITS
    }
    extra_exclude_for_wg = set()
    for name, used in used_h_recipe.items():
        if name.startswith("wgrandom_"):
            extra_exclude_for_wg |= used

    test_pools = {}
    for split in ALL_SPLITS:
        test_pools[split] = build_test_pool(split, used_h_per_split, used_b_prompts, extra_exclude_for_wg)
        n_h = sum(1 for r in test_pools[split] if r["side"] == "H")
        n_b = sum(1 for r in test_pools[split] if r["side"] == "B")
        tprint(f"  [{split}] test: {n_h} H + {n_b} B = {len(test_pools[split])}")

    tprint(f"=== Steering at α={ALPHA}: {len(directions)} directions × {len(ALL_SPLITS)} test splits ===")
    for dir_name, (d, h) in directions.items():
        intervention = f"steer_{dir_name}_alpha{ALPHA}"
        for split in ALL_SPLITS:
            records = test_pools[split]
            hooks = steer_fwd_hooks(d, ALPHA, LAYER)
            responses = generate_for_records(records, hooks,
                                             desc=f"{intervention}:{split}")
            out = RESP_ROOT / intervention / f"{split}.json"
            write_responses(records, responses, out, intervention=intervention,
                            dir_name=dir_name, alpha=ALPHA, direction_hash=h)
            tprint(f"  ✓ {intervention} {split}: {len(records)} hash={h[:12]}…")

    tprint("=== Done. ===")
    tprint(f"  responses written under {RESP_ROOT.relative_to(EXP_V10)}/steer_*_alpha25/")
    tprint("  next: re-run judge_discover_qwen.sh, then results script")


if __name__ == "__main__":
    main()
