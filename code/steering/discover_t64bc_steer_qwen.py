#!/usr/bin/env python3
"""
discover_t64bc_steer_qwen.py — companion to discover_t64bc_qwen.py.

Same 16 directions (H pools combined with train64-BC), but runs STEERING at
α=25 (Qwen's locked saturating value). Builds directions deterministically
so it can run in parallel with the ablation script (hash-gate confirms
identical tensors).
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
ALPHA = 25

GEN_BATCH = 32
MAX_LEN = 128
MAX_NEW = 50
TEST_CAP = 25

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

_T0 = time.monotonic()
def _ts(): return f"[{datetime.now().strftime('%H:%M:%S')} +{time.monotonic() - _T0:6.1f}s]"
def tprint(*a, **k): print(_ts(), *a, flush=True, **k)
def clear_cuda():
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
def unit(x): return x / x.norm().clamp_min(1e-12)
def sha256_tensor(x):
    arr = x.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()


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


def tokenize_qwen_chat(instructions, max_length=MAX_LEN):
    prompts = [QWEN_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    return hf_tok(prompts, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").input_ids.to(model.cfg.device)


def generate_with_hooks(toks, max_tokens_generated, fwd_hooks=()):
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


def generate_for_records(records, fwd_hooks, desc):
    prompts = [r["prompt"] for r in records]
    out = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc=desc, unit="batch", leave=False):
        toks = tokenize_qwen_chat(prompts[i:i + GEN_BATCH], max_length=MAX_LEN)
        out.extend(generate_with_hooks(toks, MAX_NEW, fwd_hooks))
        del toks; clear_cuda()
    return out


def cache_resid_pre(prompts, layer, pos_slice, desc):
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
        del cache, toks; clear_cuda()
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, D_MODEL)


def make_steer_hook(direction, alpha):
    d = direction.to(dtype=DTYPE, device=DEVICE)
    def hook_fn(activation, hook):
        return activation + alpha * d
    return hook_fn


def steer_fwd_hooks(direction, alpha, layer):
    return [(utils.get_act_name("resid_pre", layer), make_steer_hook(direction, alpha))]


def load_train64_bc():
    rows = json.load(open(LEGACY_TRAIN64))
    B = [r for r in rows if r["prompt_harm_label"] == "unharmful"]
    return [{"prompt": r["prompt"], "prompt_harm_label": r["prompt_harm_label"]} for r in B]


def load_persplit_h(split):
    return json.load(open(PER_SPLIT_POOLS_DIR / split / "H.json"))


def load_wg_random_h(seed):
    rows = json.load(open(JUDGED_DIR / "WildGuard_all.json"))
    pool = [r for r in rows if r.get("is_refusal_wg") == 1]
    rng = random.Random(seed)
    sampled = rng.sample(pool, N_H)
    return [{"prompt": r["prompt"], "prompt_harm_label": r["prompt_harm_label"]} for r in sampled]


def build_or_load_direction(name, H, B):
    out_dir = DIRECTIONS_ROOT / name
    d_path = out_dir / "d.pt"
    if d_path.exists():
        d = torch.load(d_path, map_location="cpu")
        h = sha256_tensor(d)
        tprint(f"  [{name}] loaded existing  hash={h[:12]}…")
        return d, h
    tprint(f"  [{name}] building H={len(H)} B={len(B)}")
    H_acts = cache_resid_pre([r["prompt"] for r in H], LAYER, POS, f"{name}.HR")
    B_acts = cache_resid_pre([r["prompt"] for r in B], LAYER, POS, f"{name}.BC")
    d = unit(H_acts.float().mean(dim=0) - B_acts.float().mean(dim=0))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(d, d_path)
    h = sha256_tensor(d)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump({"model": MODEL_PATH, "name": name, "layer": LAYER, "pos": POS,
                   "n_h": len(H), "n_b": len(B), "direction_sha256": h, "d_model": D_MODEL,
                   "timestamp": datetime.now().isoformat(timespec="seconds")}, f,
                  ensure_ascii=False, indent=2)
    tprint(f"  [{name}] saved hash={h[:12]}…")
    return d, h


def build_test_pool(split, used_h_per_split, used_b_prompts, extra_exclude_for_wg):
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


def write_responses(records, responses, out_path, intervention, dir_name, alpha, direction_hash):
    rows = []
    for r, resp in zip(records, responses):
        rows.append({
            "prompt": r["prompt"], "prompt_harm_label": r["prompt_harm_label"],
            "side": r["side"], "response": resp,
            "gen_meta": {"model": MODEL_PATH, "intervention": intervention,
                         "dir_name": dir_name, "alpha": alpha,
                         "direction_sha256": direction_hash,
                         "layer_for_direction": LAYER, "pos_for_direction": POS,
                         "max_new_tokens": MAX_NEW, "batch_size": GEN_BATCH,
                         "decoding": "greedy",
                         "timestamp": datetime.now().isoformat(timespec="seconds")},
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    bc_t64 = load_train64_bc()
    directions = []   # (name, d_cuda, hash, role, target_split)
    used_h_recipe = {}

    tprint("=== Building 11 per-split-H + train64-BC directions ===")
    for split in ALL_SPLITS:
        H = load_persplit_h(split)
        name = f"persplit_{split}_t64BC"
        d, h = build_or_load_direction(name, H, bc_t64)
        directions.append((name, d.to(DEVICE, dtype=DTYPE), h, "persplit_diagonal", split))
        used_h_recipe[name] = {r["prompt"] for r in H}

    tprint("=== Building 5 wgrandom-H + train64-BC directions ===")
    for seed in WG_RANDOM_SEEDS:
        H = load_wg_random_h(seed)
        name = f"wgrandom_seed{seed}_t64BC"
        d, h = build_or_load_direction(name, H, bc_t64)
        directions.append((name, d.to(DEVICE, dtype=DTYPE), h, "wgrandom_alltests", None))
        used_h_recipe[name] = {r["prompt"] for r in H}

    tprint("=== Hash verification gate ===")
    for name, d, h, _, _ in directions:
        d_disk = torch.load(DIRECTIONS_ROOT / name / "d.pt", map_location="cpu")
        h_disk = sha256_tensor(d_disk)
        man_h = json.load(open(DIRECTIONS_ROOT / name / "manifest.json"))["direction_sha256"]
        if not (h_disk == h == man_h):
            raise RuntimeError(f"HASH MISMATCH {name}")
        tprint(f"  ✓ {name}: hash {h[:16]}… verified")

    tprint("=== Building test pools ===")
    used_b_prompts = {r["prompt"] for r in
                      json.load(open(PER_SPLIT_POOLS_DIR / "WildGuard_all" / "B.json"))}
    used_h_per_split = {s: {r["prompt"] for r in json.load(open(PER_SPLIT_POOLS_DIR / s / "H.json"))}
                        for s in ALL_SPLITS}
    extra_exclude_for_wg = set()
    for name, used in used_h_recipe.items():
        if name.startswith("wgrandom_"):
            extra_exclude_for_wg |= used
    test_pools = {}
    for s in ALL_SPLITS:
        test_pools[s] = build_test_pool(s, used_h_per_split, used_b_prompts, extra_exclude_for_wg)
        n_h = sum(1 for r in test_pools[s] if r["side"] == "H")
        n_b = sum(1 for r in test_pools[s] if r["side"] == "B")
        tprint(f"  [{s}] test: {n_h} H + {n_b} B = {len(test_pools[s])}")

    tprint(f"=== Steering at α={ALPHA}: {len(directions)} directions ===")
    for name, d, h, role, target_split in directions:
        intervention = f"steer_{name}_alpha{ALPHA}"
        if role == "persplit_diagonal":
            test_splits = [target_split]
        else:
            test_splits = ALL_SPLITS
        for split in test_splits:
            records = test_pools[split]
            hooks = steer_fwd_hooks(d, ALPHA, LAYER)
            responses = generate_for_records(records, hooks, desc=f"{intervention}:{split}")
            out = RESP_ROOT / intervention / f"{split}.json"
            write_responses(records, responses, out, intervention, name, ALPHA, h)
            tprint(f"  ✓ {intervention} {split}: {len(records)} hash={h[:12]}…")

    tprint("=== Done. ===")


if __name__ == "__main__":
    main()
