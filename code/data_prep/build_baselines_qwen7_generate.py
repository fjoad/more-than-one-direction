#!/usr/bin/env python3
"""Qwen-7B-Chat baseline (no hooks) on all 2544 source prompts.

Mirrors build_baselines_llama_generate.py exactly but with Qwen-7B-Chat config:
- MODEL_PATH = Qwen/Qwen-7B-Chat (legacy v1, same family as Qwen-1.8B-Chat)
- Qwen chat template (<|im_start|>user / <|im_end|> / <|im_start|>assistant)
- Sharded across 2× v100_16GB (7B at fp16 = ~14GB, needs sharding on 16GB cards)
- Output to data/baselines/qwen7/responses/<split>.json
"""
from __future__ import annotations
import gc, json, os, time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

EXP = Path(__file__).resolve().parent.parent
SOURCES = EXP / "data" / "sources"
OUT_DIR = EXP / "data" / "baselines" / "qwen7" / "responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
N_SHARDS = int(os.environ.get("N_SHARDS", "1"))
assert 0 <= SHARD_ID < N_SHARDS

MODEL_PATH = "Qwen/Qwen-7B-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
MAX_LEN = 128
MAX_NEW = 50
GEN_BATCH = 16

ALL_SPLITS = [
    "WildGuard_all", "XSTest_all",
    "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
    "CocoNot_Safety", "CocoNot_Unsupported",
    "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
    "SorryBench_Inappropriate", "SorryBench_Advice",
    "_BC_source",
]

_T0 = time.monotonic()
def _ts(): return f"[{datetime.now().strftime('%H:%M:%S')} +{time.monotonic()-_T0:7.1f}s]"
def tprint(*a, **kw): print(_ts(), *a, **kw, flush=True)
def clear_cuda():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


tprint(f"=== Loading {MODEL_PATH} sharded across {torch.cuda.device_count()} GPU(s) ===")
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH, device=DEVICE, n_devices=torch.cuda.device_count(), dtype=DTYPE,
)
model.eval()
hf_tok: AutoTokenizer = model.tokenizer
hf_tok.padding_side = "left"
if hf_tok.pad_token is None: hf_tok.pad_token = hf_tok.eos_token
tprint(f"Loaded. n_layers={model.cfg.n_layers}")
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


# Same Qwen chat template as Qwen-1.8B
QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

def tokenize_qwen_chat(instructions, max_length=MAX_LEN):
    chats = [QWEN_CHAT_TEMPLATE.format(instruction=s) for s in instructions]
    return hf_tok(chats, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").input_ids.to(model.cfg.device)


def generate(toks, n_new):
    """Sharded-safe greedy decode with EOS detection. CONTEXT_RECOVERY.md §5."""
    B, T = toks.shape
    all_toks = torch.zeros((B, T + n_new), dtype=torch.long, device=toks.device)
    all_toks[:, :T] = toks
    eos_tensor = EOS_IDS_TENSOR.to(toks.device)
    done = torch.zeros(B, dtype=torch.bool, device=toks.device)
    for i in range(n_new):
        with torch.inference_mode():
            logits = model(all_toks[:, :T + i])
            nxt = logits[:, -1, :].argmax(dim=-1).to(toks.device)
            nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
            all_toks[:, T + i] = nxt
            done = done | torch.isin(nxt, eos_tensor)
            if done.all().item(): break
    return hf_tok.batch_decode(all_toks[:, T:], skip_special_tokens=True)


def get_prompt(r):
    for k in ("prompt", "instruction", "text", "user_prompt", "question"):
        if isinstance(r, dict) and k in r and isinstance(r[k], str):
            return r[k]
    return None


for split in ALL_SPLITS:
    out_path = OUT_DIR / (f"{split}_shard{SHARD_ID}.json" if N_SHARDS > 1 else f"{split}.json")
    if out_path.exists():
        try:
            existing = json.load(open(out_path))
            src_records = json.load(open(SOURCES / f"{split}.json"))
            expected = len(src_records[SHARD_ID::N_SHARDS]) if N_SHARDS > 1 else len(src_records)
            if len(existing) == expected:
                tprint(f"[skip] {split} shard {SHARD_ID} already has {len(existing)} responses")
                continue
        except Exception:
            pass

    src_records = json.load(open(SOURCES / f"{split}.json"))
    all_prompts = [get_prompt(r) for r in src_records]
    all_prompts = [p for p in all_prompts if p]
    prompts = all_prompts[SHARD_ID::N_SHARDS]
    if N_SHARDS > 1:
        out_path = OUT_DIR / f"{split}_shard{SHARD_ID}.json"
    tprint(f"\n=== {split}: shard {SHARD_ID}/{N_SHARDS} = {len(prompts)} of {len(all_prompts)} prompts ===")

    label = "harmful" if split != "_BC_source" else "unharmful"
    responses = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc=split, unit="batch"):
        toks = tokenize_qwen_chat(prompts[i:i + GEN_BATCH])
        responses.extend(generate(toks, MAX_NEW))
        clear_cuda()

    records = []
    for p, r in zip(prompts, responses):
        records.append({
            "prompt": p, "prompt_harm_label": label,
            "response": r,
            "gen_meta": {"model": MODEL_PATH, "max_new": MAX_NEW, "batch": GEN_BATCH,
                         "timestamp": datetime.utcnow().isoformat()},
            "source": split,
        })
    json.dump(records, open(out_path, "w"), indent=2)
    tprint(f"  ✓ {split}: {len(records)} → {out_path.name}")

tprint("Done.")
