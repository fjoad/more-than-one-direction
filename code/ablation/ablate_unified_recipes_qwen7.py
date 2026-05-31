#!/usr/bin/env python3
"""Phase B + D for Qwen-7B-Chat — build new directions and ablate all on the
Qwen-7B unified pool. Mirror of ablate_unified_recipes_llama.py.

Directions processed (in order):
  - 11 per-split (existing tensors)
  - train64 (build fresh on Qwen-7B)
  - rr33_seed{1..5} (build)
  - rr55_seed{1..5} (build)
  - crisp_persplit_seed{1..5} (build)
  - B1 (existing tensor IF b1 pipeline completed — else deferred)

Skip-existing per recipe name. Sibling _b.py reverses the order.
Run with CUDA_LAUNCH_BLOCKING=1 + expandable_segments + 2× v100_16GB sharded.

NEEL_LAYER overridable via env var (default 18, ~58% depth on 32 layers).
"""
from __future__ import annotations
import gc, hashlib, json, os, time
from datetime import datetime
from pathlib import Path

import einops, torch
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils
from transformers import AutoTokenizer

EXP = Path(__file__).resolve().parent.parent
JUDGED = EXP / "data" / "baselines" / "qwen7" / "judged"
POOLS = EXP / "data" / "training_pools" / "qwen7"
DIR_ROOT = EXP / "data" / "directions" / "qwen7"
POOL_NAME = os.environ.get("POOL_NAME", "qwen7")
POOL_FILE = EXP / "data" / "common_test_pool" / POOL_NAME / "unified_H.json"
_RESP_SUFFIX = "ablate_unified" if POOL_NAME == "qwen7" else f"ablate_unified_on_{POOL_NAME}"
RESP_DIR = EXP / "generations" / "discover_qwen7" / "responses" / _RESP_SUFFIX
RESP_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "Qwen/Qwen-7B-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
LAYER = int(os.environ.get("NEEL_LAYER", "18"))
POS = -1
MAX_LEN = 128
MAX_NEW = 50
GEN_BATCH = 8

PER_SPLIT_NAMES = [
    "WildGuard_all", "XSTest_all",
    "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
    "CocoNot_Safety", "CocoNot_Unsupported",
    "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
    "SorryBench_Inappropriate", "SorryBench_Advice",
]

RECIPES = []
for s in PER_SPLIT_NAMES:
    RECIPES.append((s, "build", POOLS / "per_split_simple" / s))
RECIPES.append(("train64", "build_train64", None))
for k in (1, 2, 3, 4, 5): RECIPES.append((f"rr33_seed{k}", "build", POOLS / f"rr33_seed{k}"))
for k in (1, 2, 3, 4, 5): RECIPES.append((f"rr55_seed{k}", "build", POOLS / f"rr55_seed{k}"))
for k in (1, 2, 3, 4, 5): RECIPES.append((f"crisp_persplit_seed{k}", "build", POOLS / f"crisp_persplit_seed{k}"))
RECIPES.append(("B1", "existing_if_present", DIR_ROOT / "forced_b1" / "d.pt"))

_T0 = time.monotonic()
def _ts(): return f"[{datetime.now().strftime('%H:%M:%S')} +{time.monotonic()-_T0:7.1f}s]"
def tprint(*a, **kw): print(_ts(), *a, **kw, flush=True)

def clear_cuda():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def sha256_tensor(t):
    return hashlib.sha256(t.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()).hexdigest()

def unit(v): return v / v.norm().clamp_min(1e-12)


def remaining_recipes(recipes):
    out = []
    for name, mode, src in recipes:
        resp_path = RESP_DIR / f"{name}.json"
        if resp_path.exists():
            try:
                data = json.load(open(resp_path))
                if len(data) >= 100:
                    tprint(f"  [skip] {name} already has {len(data)} records")
                    continue
            except Exception:
                pass
        if mode == "existing_if_present" and not Path(src).exists():
            tprint(f"  [defer] {name}: direction not yet built (will pick up later)")
            continue
        out.append((name, mode, src))
    return out


todo = remaining_recipes(RECIPES)
tprint(f"Worker recipes to process: {[r[0] for r in todo]}")
tprint(f"NEEL_LAYER={LAYER} POS={POS}")
if not todo:
    tprint("Nothing to do. Exiting.")
    raise SystemExit(0)

# Load test pool
test_pool = json.load(open(POOL_FILE))
tprint(f"Qwen-7B unified test pool: {len(test_pool)} prompts")
TEST_N = len(test_pool)

# Load model (sharded across 2 v100_16GB)
tprint(f"=== Loading {MODEL_PATH} sharded across {torch.cuda.device_count()} GPU(s) ===")
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH, device=DEVICE, n_devices=torch.cuda.device_count(), dtype=DTYPE,
)
model.eval()
hf_tok: AutoTokenizer = model.tokenizer
hf_tok.padding_side = "left"
if hf_tok.pad_token is None: hf_tok.pad_token = hf_tok.eos_token
N_LAYERS = model.cfg.n_layers
tprint(f"Loaded. n_layers={N_LAYERS} d_model={model.cfg.d_model}")
clear_cuda()



# Robust end-of-turn token IDs. CONTEXT_RECOVERY.md §5 / pitfall #5 — single
# eos_token_id misses Qwen <|im_end|> and Llama-3 <|eot_id|>. Collect any of them.
_EOT_CANDIDATES = ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"]
EOS_IDS = set()
if hf_tok.eos_token_id is not None: EOS_IDS.add(hf_tok.eos_token_id)
_unk = getattr(hf_tok, "unk_token_id", None)
for _t in _EOT_CANDIDATES:
    _tid = hf_tok.convert_tokens_to_ids(_t)
    if _tid is not None and _tid != _unk and _tid >= 0:
        EOS_IDS.add(_tid)
EOS_IDS_TENSOR = torch.tensor(sorted(EOS_IDS), dtype=torch.long)
PAD_ID = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else hf_tok.eos_token_id
tprint(f"EOS_IDS={sorted(EOS_IDS)}  PAD_ID={PAD_ID}")

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

def tokenize_qwen_chat(instructions, max_length=MAX_LEN):
    chats = [QWEN_CHAT_TEMPLATE.format(instruction=s) for s in instructions]
    return hf_tok(chats, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").input_ids.to(model.cfg.device)


def generate_with_hooks(toks, max_new, fwd_hooks=()):
    """Sharded-safe greedy decode with EOS detection. CONTEXT_RECOVERY.md §5."""
    B, T = toks.shape
    all_toks = torch.zeros((B, T + max_new), dtype=torch.long, device=toks.device)
    all_toks[:, :T] = toks
    eos_tensor = EOS_IDS_TENSOR.to(toks.device)
    done = torch.zeros(B, dtype=torch.bool, device=toks.device)
    for i in range(max_new):
        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(all_toks[:, :T + i])
                next_toks = logits[:, -1, :].argmax(dim=-1).to(all_toks.device)
                next_toks = torch.where(done, torch.full_like(next_toks, PAD_ID), next_toks)
                all_toks[:, T + i] = next_toks
                done = done | torch.isin(next_toks, eos_tensor)
                if done.all().item(): break
    return hf_tok.batch_decode(all_toks[:, T:], skip_special_tokens=True)


def cache_resid_pre(prompts):
    hook_name = utils.get_act_name("resid_pre", LAYER)
    chunks = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc="cache", unit="batch", leave=False):
        toks = tokenize_qwen_chat(prompts[i:i + GEN_BATCH])
        with torch.no_grad():
            _, cache = model.run_with_cache(
                toks, names_filter=lambda n: n == hook_name,
                pos_slice=POS, stop_at_layer=LAYER + 1,
            )
        chunks.append(cache[hook_name].squeeze(1).to("cpu", dtype=torch.float32))
        del cache, toks
        clear_cuda()
    return torch.cat(chunks, dim=0)


def make_ablate_hook(direction):
    """Move direction to activation's device per-call — Qwen-7B layers are sharded
    across cuda:0 and cuda:1, so a fixed device wouldn't match all hook points."""
    def fn(activation, hook):
        d = direction.to(dtype=activation.dtype, device=activation.device)
        proj = einops.einsum(activation, d, "... d_act, d_act -> ...").unsqueeze(-1) * d
        return activation - proj
    return fn

def ablate_hooks(d):
    return [(utils.get_act_name(act, l), make_ablate_hook(d))
            for l in range(N_LAYERS) for act in ("resid_pre", "resid_mid", "resid_post")]


TRAIN64_LEGACY = EXP / "data" / "legacy" / "train_subset64_working.json"


for name, mode, src in todo:
    resp_path = RESP_DIR / f"{name}.json"
    if resp_path.exists():
        try:
            if len(json.load(open(resp_path))) >= TEST_N:
                tprint(f"=== {name}: completed by sibling, skipping")
                continue
        except Exception:
            pass

    tprint(f"\n=== {name} ({mode}) ===")

    if mode == "existing" or mode == "existing_if_present":
        if not Path(src).exists():
            tprint(f"  direction tensor missing at {src}; skipping")
            continue
        d = torch.load(src, map_location=DEVICE).to(dtype=DTYPE, device=DEVICE)
        d_hash = sha256_tensor(d)
        tprint(f"  Loaded existing sha={d_hash[:16]}…")

    elif mode == "build":
        h_pool = json.load(open(src / "H.json"))
        b_pool = json.load(open(src / "B.json"))
        tprint(f"  Building direction from {len(h_pool)} H + {len(b_pool)} B")
        h_acts = cache_resid_pre([r["prompt"] for r in h_pool])
        b_acts = cache_resid_pre([r["prompt"] for r in b_pool])
        d = unit(h_acts.mean(dim=0) - b_acts.mean(dim=0))
        d_hash = sha256_tensor(d)
        dir_out = DIR_ROOT / name
        dir_out.mkdir(parents=True, exist_ok=True)
        torch.save(d, dir_out / "d.pt")
        json.dump({
            "model": MODEL_PATH, "layer": LAYER, "pos": POS,
            "n_h": len(h_pool), "n_b": len(b_pool),
            "direction_sha256": d_hash, "recipe": name,
            "source_pool": str(src.relative_to(EXP)),
            "timestamp": datetime.utcnow().isoformat(),
        }, open(dir_out / "manifest.json", "w"), indent=2)
        d = d.to(dtype=DTYPE, device=DEVICE)
        tprint(f"  Built sha={d_hash[:16]}…  saved → {dir_out}")

    elif mode == "build_train64":
        train64 = json.load(open(TRAIN64_LEGACY))
        h_prompts = [r["prompt"] for r in train64 if r.get("prompt_harm_label") == "harmful"]
        b_prompts = [r["prompt"] for r in train64 if r.get("prompt_harm_label") == "unharmful"]
        assert len(h_prompts) == 32 and len(b_prompts) == 32
        tprint(f"  Building train64 direction from 32 H + 32 B")
        h_acts = cache_resid_pre(h_prompts)
        b_acts = cache_resid_pre(b_prompts)
        d = unit(h_acts.mean(dim=0) - b_acts.mean(dim=0))
        d_hash = sha256_tensor(d)
        dir_out = DIR_ROOT / "train64_orig"
        dir_out.mkdir(parents=True, exist_ok=True)
        torch.save(d, dir_out / "d.pt")
        json.dump({"model": MODEL_PATH, "layer": LAYER, "pos": POS,
                   "direction_sha256": d_hash, "recipe": "train64_orig",
                   "n_h": 32, "n_b": 32,
                   "timestamp": datetime.utcnow().isoformat()},
                  open(dir_out / "manifest.json", "w"), indent=2)
        d = d.to(dtype=DTYPE, device=DEVICE)
        tprint(f"  Built train64 sha={d_hash[:16]}… saved → {dir_out}")
    else:
        raise ValueError(f"unknown mode {mode}")

    # Ablate on unified pool
    ablate_h = ablate_hooks(d)
    out_records = []
    sibling_wrote = False
    for i in tqdm(range(0, len(test_pool), GEN_BATCH), desc=f"ablate-{name}",
                  unit="batch", leave=False):
        # Mid-recipe skip — every batch, check if a sibling completed this recipe
        # while we were grinding. If so, abort and let their (identical) file stand.
        if i > 0 and resp_path.exists():
            try:
                existing = json.load(open(resp_path))
                if len(existing) >= TEST_N:
                    tprint(f"  [mid-skip] {name}: sibling wrote complete file at batch {i // GEN_BATCH}; aborting")
                    sibling_wrote = True
                    break
            except Exception:
                pass
        batch = test_pool[i:i + GEN_BATCH]
        toks = tokenize_qwen_chat([r["prompt"] for r in batch])
        resps = generate_with_hooks(toks, MAX_NEW, fwd_hooks=ablate_h)
        for r, resp in zip(batch, resps):
            out_records.append({
                "prompt": r["prompt"], "split": r["split"], "cluster": r["cluster"],
                "side": "H", "prompt_harm_label": "harmful",
                "response": resp,
                "intervention": f"ablate_unified_{name}",
                "direction_sha256": d_hash,
            })
    if sibling_wrote:
        clear_cuda()
        continue
    json.dump(out_records, open(resp_path, "w"), indent=2)
    tprint(f"  Wrote {len(out_records)} → {resp_path}")
    clear_cuda()

tprint("Worker done.")
