#!/usr/bin/env python3
"""
build_baselines_qwen_wgm_train_generate.py — Qwen baseline (no hooks) on all 24,956
prompts in exp1_v5/data/wildguard-mix/train_clean.json. Same pool train64 was
selected from. Mirrors build_baselines_qwen_generate.py exactly.

Writes data/baselines/qwen/wgm_train/responses.json with:
    {prompt, prompt_harm_label, response, gen_meta}

Followup: build_baselines_qwen_wgm_train_judge.{py,sh} runs the WildGuard 7B judge
over those responses.

Runtime estimate: 24956 / 24 = ~1040 batches × ~5-7s on Qwen 1.8B on V100 16GB
                  ≈ 90-120 min.
"""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

EXP_V10 = Path(__file__).resolve().parent.parent
SOURCE_FILE = Path("/export/home/fjoad/llm_safety_project/exp1_v5/data/wildguard-mix/train_clean.json")
OUT_DIR = EXP_V10 / "data" / "baselines" / "qwen" / "wgm_train"
OUT_PATH = OUT_DIR / "responses.json"

MODEL_PATH = "Qwen/Qwen-1_8B-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

MAX_LEN = 128
MAX_NEW = 50
GEN_BATCH = 24

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
tprint(f"Loaded.  n_layers={model.cfg.n_layers}  d_model={model.cfg.d_model}")
clear_cuda()


def tokenize_qwen_chat(instructions: List[str], max_length: int = MAX_LEN):
    prompts = [QWEN_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    return hf_tok(prompts, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").input_ids.to(model.cfg.device)


def generate_baseline(toks, max_tokens_generated: int) -> List[str]:
    B, T = toks.shape
    all_toks = torch.zeros((B, T + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :T] = toks
    for i in range(max_tokens_generated):
        with torch.inference_mode():
            logits = model(all_toks[:, :T + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            all_toks[:, T + i] = next_tokens
    return hf_tok.batch_decode(all_toks[:, T:], skip_special_tokens=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tprint(f"=== Loading source: {SOURCE_FILE} ===")
    with open(SOURCE_FILE) as f:
        records = json.load(f)
    n = len(records)
    tprint(f"  loaded {n} prompts")

    # Resume support: skip any prompts whose response is already on disk
    existing = []
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            existing = json.load(f)
        done_prompts = {r["prompt"] for r in existing}
        tprint(f"  resuming: {len(existing)} responses already on disk")
        records = [r for r in records if r["prompt"] not in done_prompts]
        tprint(f"  remaining: {len(records)} prompts to generate")

    out_records = list(existing)
    save_every = 240  # save every ~10 batches

    tprint(f"=== Generating Qwen baselines for {len(records)} prompts ===")
    pbar = tqdm(range(0, len(records), GEN_BATCH), desc="wgm_train", unit="batch")
    for batch_idx, i in enumerate(pbar):
        batch = records[i:i + GEN_BATCH]
        prompts = [r["prompt"] for r in batch]
        toks = tokenize_qwen_chat(prompts, max_length=MAX_LEN)
        responses = generate_baseline(toks, max_tokens_generated=MAX_NEW)
        for src, resp in zip(batch, responses):
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
            })
        del toks
        clear_cuda()
        # Periodic save for crash recovery
        if (batch_idx + 1) % save_every == 0:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(out_records, f, ensure_ascii=False, indent=2)
            tprint(f"  checkpoint: {len(out_records)} responses saved")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    tprint(f"=== Done. {len(out_records)} total responses → {OUT_PATH} ===")


if __name__ == "__main__":
    main()
