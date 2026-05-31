#!/usr/bin/env python3
"""Ablate each per-split direction on the unified 256-prompt test pool.

Each worker processes the directions listed in env var DIRS (comma-separated
split names). Writes one output JSON per direction to:
  generations/discover_qwen/responses/ablate_unified/<split>.json

Skip-existing: if the output file already has 256 records, the worker skips
that direction (lets sibling workers race-meet safely).
"""
from __future__ import annotations
import gc, hashlib, json, os, time
from datetime import datetime
from pathlib import Path
from typing import List

import einops, torch
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils
from transformers import AutoTokenizer

EXP = Path(__file__).resolve().parent.parent
DIR_ROOT = EXP / "data" / "directions" / "qwen" / "per_split_simple"
POOL_FILE = EXP / "data" / "common_test_pool" / "qwen" / "unified_H.json"
RESP_DIR = EXP / "generations" / "discover_qwen" / "responses" / "ablate_unified"
RESP_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "Qwen/Qwen-1_8B-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
LAYER = 14
POS = -1
MAX_LEN = 128
MAX_NEW = 50
GEN_BATCH = 32

QWEN_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

ALL_SPLITS = [
    "WildGuard_all", "XSTest_all",
    "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
    "CocoNot_Safety", "CocoNot_Unsupported",
    "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
    "SorryBench_Inappropriate", "SorryBench_Advice",
]

DIRS_TO_RUN = ALL_SPLITS

_T0 = time.monotonic()
def _ts(): return f"[{datetime.now().strftime('%H:%M:%S')} +{time.monotonic()-_T0:7.1f}s]"
def tprint(*a, **kw): print(_ts(), *a, **kw, flush=True)

def clear_cuda():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def sha256_tensor(t):
    return hashlib.sha256(t.detach().cpu().to(torch.float32).numpy().tobytes()).hexdigest()


tprint(f"Worker dirs: {DIRS_TO_RUN}")

# Determine which directions still need processing (skip-existing)
remaining = []
for split in DIRS_TO_RUN:
    out = RESP_DIR / f"{split}.json"
    if out.exists():
        try:
            existing = json.load(open(out))
            if len(existing) == 256:
                tprint(f"  [skip] {split} already has 256 records")
                continue
        except Exception:
            pass
    remaining.append(split)

if not remaining:
    tprint("Nothing to do. Exiting.")
    raise SystemExit(0)

tprint(f"Will process: {remaining}")

# Load test pool
test_pool = json.load(open(POOL_FILE))
tprint(f"Test pool: {len(test_pool)} prompts")
test_prompts = [r["prompt"] for r in test_pool]

# Load model
tprint(f"=== Loading {MODEL_PATH} ===")
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH, device=DEVICE, dtype=DTYPE, default_padding_side="left",
)
hf_tok: AutoTokenizer = model.tokenizer
hf_tok.padding_side = "left"
if hf_tok.pad_token is None: hf_tok.pad_token = "<|extra_0|>"
N_LAYERS = model.cfg.n_layers
tprint(f"Loaded. n_layers={N_LAYERS}")
clear_cuda()


def tokenize(prompts):
    formatted = [QWEN_TEMPLATE.format(instruction=p) for p in prompts]
    return hf_tok(formatted, padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").input_ids.to(model.cfg.device)


def generate(toks, n_new, fwd_hooks=()):
    B, T = toks.shape
    all_toks = torch.zeros((B, T + n_new), dtype=torch.long, device=toks.device)
    all_toks[:, :T] = toks
    eos_id = hf_tok.convert_tokens_to_ids("<|im_end|>")
    pad_id = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else 0
    done = torch.zeros(B, dtype=torch.bool, device=toks.device)
    for i in range(n_new):
        with torch.inference_mode():
            with model.hooks(fwd_hooks=list(fwd_hooks)):
                logits = model(all_toks[:, :T + i])
            nxt = logits[:, -1, :].argmax(dim=-1)
            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            all_toks[:, T + i] = nxt
            done = done | (nxt == eos_id)
            if done.all().item(): break
    return hf_tok.batch_decode(all_toks[:, T:], skip_special_tokens=True)


def make_ablate_hook(d):
    d = d.to(dtype=DTYPE, device=DEVICE)
    def fn(act, hook):
        proj = einops.einsum(act, d, "... d, d -> ...").unsqueeze(-1) * d
        return act - proj
    return fn

def ablate_hooks(d):
    return [(utils.get_act_name(act, l), make_ablate_hook(d))
            for l in range(N_LAYERS) for act in ("resid_pre", "resid_mid", "resid_post")]


# Process each direction
for split in remaining:
    out = RESP_DIR / f"{split}.json"
    # Race-check: another sibling may have completed this in the meantime
    if out.exists():
        try:
            if len(json.load(open(out))) == 256:
                tprint(f"=== {split}: completed by sibling, skipping")
                continue
        except Exception:
            pass

    d_path = DIR_ROOT / split / "d.pt"
    if not d_path.exists():
        tprint(f"  [missing direction] {d_path}")
        continue

    d = torch.load(d_path, map_location=DEVICE).to(dtype=DTYPE, device=DEVICE)
    d_hash = sha256_tensor(d)
    tprint(f"=== {split}  sha={d_hash[:16]}… ===")

    ablate_h = ablate_hooks(d)
    out_records = []
    for i in tqdm(range(0, len(test_prompts), GEN_BATCH), desc=f"ablate-{split}",
                  unit="batch", leave=False):
        batch = test_pool[i:i + GEN_BATCH]
        toks = tokenize([r["prompt"] for r in batch])
        resps = generate(toks, MAX_NEW, fwd_hooks=ablate_h)
        for r, resp in zip(batch, resps):
            out_records.append({
                "prompt": r["prompt"],
                "split": r["split"],
                "cluster": r["cluster"],
                "side": "H",
                "prompt_harm_label": "harmful",
                "response": resp,
                "intervention": f"ablate_unified_{split}",
                "direction_sha256": d_hash,
            })
    json.dump(out_records, open(out, "w"), indent=2)
    tprint(f"  Wrote {len(out_records)} → {out}")
    clear_cuda()

tprint("Worker done.")
