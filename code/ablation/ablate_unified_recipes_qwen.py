#!/usr/bin/env python3
"""Build new recipe directions + ablate ALL test recipes on the unified pool.

Recipes (17 total ablation cells on the unified 256-prompt pool):
  - train64           : existing direction at data/directions/qwen/train64_orig/d.pt
  - B1                : existing direction at data/directions/qwen/forced_b1/d.pt
  - rr33_seed{1..5}   : NEW build from data/training_pools/qwen/rr33_seed{k}/
  - rr55_seed{1..5}   : NEW build from data/training_pools/qwen/rr55_seed{k}/
  - crisp_persplit_seed{1..5} : NEW build from data/training_pools/qwen/crisp_persplit_seed{k}/

For each recipe:
  1. Load or build direction tensor (cache H + cache BC, mean-diff, unit-norm)
  2. Save tensor to data/directions/qwen/<recipe>/d.pt (+ manifest)
  3. Ablate on unified pool, save responses to:
     generations/discover_qwen/responses/ablate_unified/<recipe>.json
  4. Skip if response file already has 256 records.

Sibling _b.py reverses RECIPES iteration order for parallel race-meet.
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
POOLS = EXP / "data" / "training_pools" / "qwen"
DIR_ROOT = EXP / "data" / "directions" / "qwen"
POOL_NAME = os.environ.get("POOL_NAME", "qwen")
POOL_FILE = EXP / "data" / "common_test_pool" / POOL_NAME / "unified_H.json"
_RESP_SUFFIX = "ablate_unified" if POOL_NAME == "qwen" else f"ablate_unified_on_{POOL_NAME}"
RESP_DIR = EXP / "generations" / "discover_qwen" / "responses" / _RESP_SUFFIX
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

# (recipe_name, mode, source_path_or_pool_dir)
RECIPES = [
    ("train64",                   "existing", DIR_ROOT / "train64_orig" / "d.pt"),
    ("B1",                        "existing", DIR_ROOT / "forced_b1" / "d.pt"),
    ("rr33_seed1",                "build",    POOLS / "rr33_seed1"),
    ("rr33_seed2",                "build",    POOLS / "rr33_seed2"),
    ("rr33_seed3",                "build",    POOLS / "rr33_seed3"),
    ("rr33_seed4",                "build",    POOLS / "rr33_seed4"),
    ("rr33_seed5",                "build",    POOLS / "rr33_seed5"),
    ("rr55_seed1",                "build",    POOLS / "rr55_seed1"),
    ("rr55_seed2",                "build",    POOLS / "rr55_seed2"),
    ("rr55_seed3",                "build",    POOLS / "rr55_seed3"),
    ("rr55_seed4",                "build",    POOLS / "rr55_seed4"),
    ("rr55_seed5",                "build",    POOLS / "rr55_seed5"),
    ("crisp_persplit_seed1",      "build",    POOLS / "crisp_persplit_seed1"),
    ("crisp_persplit_seed2",      "build",    POOLS / "crisp_persplit_seed2"),
    ("crisp_persplit_seed3",      "build",    POOLS / "crisp_persplit_seed3"),
    ("crisp_persplit_seed4",      "build",    POOLS / "crisp_persplit_seed4"),
    ("crisp_persplit_seed5",      "build",    POOLS / "crisp_persplit_seed5"),
]

_T0 = time.monotonic()
def _ts(): return f"[{datetime.now().strftime('%H:%M:%S')} +{time.monotonic()-_T0:7.1f}s]"
def tprint(*a, **kw): print(_ts(), *a, **kw, flush=True)

def clear_cuda():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def sha256_tensor(t):
    return hashlib.sha256(t.detach().cpu().to(torch.float32).numpy().tobytes()).hexdigest()

def unit(v): return v / v.norm()


# Decide which recipes still need processing (skip-existing)
def remaining_recipes(recipes):
    out = []
    for name, mode, src in recipes:
        resp_path = RESP_DIR / f"{name}.json"
        if resp_path.exists():
            try:
                data = json.load(open(resp_path))
                if len(data) >= 256:
                    tprint(f"  [skip] {name} already has {len(data)} records")
                    continue
            except Exception:
                pass
        out.append((name, mode, src))
    return out


todo = remaining_recipes(RECIPES)
tprint(f"Worker recipes to process: {[r[0] for r in todo]}")
if not todo:
    tprint("Nothing to do. Exiting.")
    raise SystemExit(0)

# Load test pool
test_pool = json.load(open(POOL_FILE))
tprint(f"Unified test pool: {len(test_pool)} prompts")

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


def cache_resid_pre(prompts):
    hook_name = utils.get_act_name("resid_pre", LAYER)
    chunks = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc="cache",
                  unit="batch", leave=False):
        toks = tokenize(prompts[i:i + GEN_BATCH])
        with torch.no_grad():
            _, cache = model.run_with_cache(
                toks, names_filter=lambda n: n == hook_name,
                pos_slice=POS, stop_at_layer=LAYER + 1,
            )
        chunks.append(cache[hook_name].squeeze(1).to("cpu", dtype=torch.float32))
        del cache, toks
        clear_cuda()
    return torch.cat(chunks, dim=0)


def make_ablate_hook(d):
    d = d.to(dtype=DTYPE, device=DEVICE)
    def fn(act, hook):
        proj = einops.einsum(act, d, "... d, d -> ...").unsqueeze(-1) * d
        return act - proj
    return fn

def ablate_hooks(d):
    return [(utils.get_act_name(act, l), make_ablate_hook(d))
            for l in range(N_LAYERS) for act in ("resid_pre", "resid_mid", "resid_post")]


# Process each recipe
for name, mode, src in todo:
    resp_path = RESP_DIR / f"{name}.json"
    # Race-check (sibling may have completed it)
    if resp_path.exists():
        try:
            if len(json.load(open(resp_path))) >= 256:
                tprint(f"=== {name}: completed by sibling, skipping")
                continue
        except Exception:
            pass

    tprint(f"\n=== {name} ({mode}) ===")

    if mode == "existing":
        d = torch.load(src, map_location=DEVICE).to(dtype=DTYPE, device=DEVICE)
        d_hash = sha256_tensor(d)
        tprint(f"  Loaded existing direction sha={d_hash[:16]}…")
    elif mode == "build":
        h_pool = json.load(open(src / "H.json"))
        b_pool = json.load(open(src / "B.json"))
        tprint(f"  Building direction from {len(h_pool)} H + {len(b_pool)} B")
        h_acts = cache_resid_pre([r["prompt"] for r in h_pool])
        b_acts = cache_resid_pre([r["prompt"] for r in b_pool])
        d = unit(h_acts.mean(dim=0) - b_acts.mean(dim=0))
        d_hash = sha256_tensor(d)
        # Save tensor + manifest
        dir_out = DIR_ROOT / name
        dir_out.mkdir(parents=True, exist_ok=True)
        torch.save(d, dir_out / "d.pt")
        json.dump({
            "model": MODEL_PATH, "layer": LAYER, "pos": POS,
            "n_h": len(h_pool), "n_b": len(b_pool),
            "direction_sha256": d_hash,
            "recipe": name,
            "source_pool": str(src.relative_to(EXP)),
            "timestamp": datetime.utcnow().isoformat(),
        }, open(dir_out / "manifest.json", "w"), indent=2)
        d = d.to(dtype=DTYPE, device=DEVICE)
        tprint(f"  Built sha={d_hash[:16]}…  saved → {dir_out}")
    else:
        raise ValueError(f"unknown mode {mode}")

    # Ablate on unified pool
    ablate_h = ablate_hooks(d)
    out_records = []
    for i in tqdm(range(0, len(test_pool), GEN_BATCH), desc=f"ablate-{name}",
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
                "intervention": f"ablate_unified_{name}",
                "direction_sha256": d_hash,
            })
    json.dump(out_records, open(resp_path, "w"), indent=2)
    tprint(f"  Wrote {len(out_records)} → {resp_path}")
    clear_cuda()

tprint("Worker done.")
