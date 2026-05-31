#!/usr/bin/env python3
"""
build_baselines_qwen_generate.py — Run Qwen-1_8B-Chat baseline (no hooks) on every
prompt in data/sources/<split>.json (12 files: 11 H splits + _BC_source).

Writes data/baselines/qwen/responses/<split>.json with full schema:
    {prompt, prompt_harm_label, response, gen_meta, source}

This script does generation only. The WildGuard judge runs as a separate step
(build_baselines_qwen_judge.py) so we can re-judge later without re-generating.

Mirrors the Qwen tokenize/generate setup in exp1_v9/neel_ablate_qwen.py exactly,
just with empty fwd_hooks (i.e., baseline / no intervention).
"""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from jaxtyping import Int
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────
EXP_V10 = Path(__file__).resolve().parent.parent
SOURCES_DIR = EXP_V10 / "data" / "sources"
OUT_DIR = EXP_V10 / "data" / "baselines" / "qwen" / "responses"

MODEL_PATH = "Qwen/Qwen-1_8B-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

MAX_LEN = 128
MAX_NEW = 50
GEN_BATCH = 24

# Qwen chat template (same as v9)
QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# All 11 H splits + the shared BC source
ALL_SPLITS = [
    "WildGuard_all", "XSTest_all",
    "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
    "CocoNot_Safety", "CocoNot_Unsupported",
    "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
    "SorryBench_Inappropriate", "SorryBench_Advice",
    "_BC_source",
]

# ────────────────────────────────────────────────────────────────
# Timestamp helper
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


# ────────────────────────────────────────────────────────────────
# Model & tokenizer
# ────────────────────────────────────────────────────────────────
tprint(f"=== Loading {MODEL_PATH} ===")

model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH,
    device=DEVICE,
    dtype=DTYPE,
    default_padding_side="left",
    fp16=True,
)
model.eval()

hf_tok: AutoTokenizer = model.tokenizer
hf_tok.padding_side = "left"
if hf_tok.pad_token is None:
    hf_tok.pad_token = "<|extra_0|>"

tprint(f"Model & tokenizer loaded.  n_layers={model.cfg.n_layers}  d_model={model.cfg.d_model}")
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


# ────────────────────────────────────────────────────────────────
# Tokenize + generate (no hooks)
# ────────────────────────────────────────────────────────────────
def tokenize_qwen_chat(instructions: List[str], max_length: int = MAX_LEN):
    prompts = [QWEN_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    return hf_tok(
        prompts, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    ).input_ids.to(model.cfg.device)


def generate_baseline(toks, max_tokens_generated: int) -> List[str]:
    """Manual greedy generation with EOS detection (no fwd hooks). CONTEXT_RECOVERY.md §5."""
    B, T = toks.shape
    all_toks = torch.zeros((B, T + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :T] = toks
    eos_tensor = EOS_IDS_TENSOR.to(toks.device)
    done = torch.zeros(B, dtype=torch.bool, device=toks.device)
    for i in range(max_tokens_generated):
        with torch.inference_mode():
            logits = model(all_toks[:, :T + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            next_tokens = torch.where(done, torch.full_like(next_tokens, PAD_ID), next_tokens)
            all_toks[:, T + i] = next_tokens
            done = done | torch.isin(next_tokens, eos_tensor)
            if done.all().item(): break
    return hf_tok.batch_decode(all_toks[:, T:], skip_special_tokens=True)


# ────────────────────────────────────────────────────────────────
# Per-split processing
# ────────────────────────────────────────────────────────────────
def process_split(split: str) -> int:
    src_path = SOURCES_DIR / f"{split}.json"
    out_path = OUT_DIR / f"{split}.json"

    with open(src_path) as f:
        records = json.load(f)
    n = len(records)
    if n == 0:
        tprint(f"  {split}: empty source, skipping")
        return 0

    prompts = [r["prompt"] for r in records]
    out_records: List[dict] = []

    desc = f"{split}"
    for i in tqdm(range(0, n, GEN_BATCH), desc=desc, unit="batch", leave=False):
        batch_prompts = prompts[i:i + GEN_BATCH]
        toks = tokenize_qwen_chat(batch_prompts, max_length=MAX_LEN)
        responses = generate_baseline(toks, max_tokens_generated=MAX_NEW)
        for src, resp in zip(records[i:i + GEN_BATCH], responses):
            out_records.append({
                "prompt": src["prompt"],
                "prompt_harm_label": src["prompt_harm_label"],
                "response": resp,
                "gen_meta": {
                    "model": MODEL_PATH,
                    "intervention": "baseline",
                    "max_new_tokens": MAX_NEW,
                    "max_input_len": MAX_LEN,
                    "batch_size": GEN_BATCH,
                    "decoding": "greedy",
                    "padding_side": "left",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
                "source": src["source"],
            })
        del toks
        clear_cuda()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    tprint(f"  ✓ {split}: {len(out_records)} → {out_path.relative_to(EXP_V10)}")
    return len(out_records)


def main():
    tprint("=== Generating Qwen baselines for 12 splits ===")
    total = 0
    for split in ALL_SPLITS:
        total += process_split(split)
    tprint(f"=== Done. {total} responses across {len(ALL_SPLITS)} splits. ===")


if __name__ == "__main__":
    main()
