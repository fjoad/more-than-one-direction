#!/usr/bin/env python
# coding: utf-8

"""
Cache Gemma-2-9b-it activations & generate responses for SorryBench base440.

- Input:
    data/sorrybench/sorrybench_base440.jsonl
    Each line is a JSON object with keys like:
        {"question_id": 1, "category": 1, "turns": "...", "prompt_style": "base"}

- Output:
    1) Generations JSON (NOT JSONL) with model responses:
        generations/sorrybench_base440_generations.json
       Each element is the original record + "model_response".

    2) Activation cache for SAE layers [9, 20, 31], capturing resid_pre at
       blocks [10, 21, 32] / pos -2 only:
        activations/sorrybench_base440_l10_21_32_pos-2_residpre.pt

       Saved dict:
         {
           "acts": [N, 3, d_model] float16 CPU,
           "input_ids": [N, T] long CPU,
           "dataset_file": <path>,
           "sae_layers": [9, 20, 31],
           "capture_layers": [10, 21, 32],
           "pos_idx": -2,
           "seq_len": SEQ_LEN,
         }
"""

import os
import json
import gc
from pathlib import Path

from tqdm import tqdm
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utilities import devices as tl_devices
import transformer_lens.past_key_value_caching as pkv
from transformers import AutoTokenizer
import importlib

# -------------------- Config --------------------
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN        = 128
BATCH_SIZE     = 24               # tune for your GPU
SAE_LAYERS     = [9, 20, 31]
CAPTURE_LAYERS = [l + 1 for l in SAE_LAYERS]   # -> [10, 21, 32]
POS_IDX        = -2               # capture resid_pre at this token position
MAX_NEW_TOKENS = 64

# Paths
DATA_PATH   = Path("data/sorrybench/sorrybench_base440.jsonl")
ACTS_ROOT   = Path("activations")
GENS_ROOT   = Path("generations")
ACTS_ROOT.mkdir(parents=True, exist_ok=True)
GENS_ROOT.mkdir(parents=True, exist_ok=True)

ACTS_PATH = ACTS_ROOT / "sorrybench_base440_l10_21_32_pos-2_residpre.pt"
GENS_PATH = GENS_ROOT / "sorrybench_base440_generations.json"


# -------------------- Model & Tokenizer --------------------
def load_model_and_tokenizer():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available, running on CPU.")

    model = HookedTransformer.from_pretrained_no_processing(
        "google/gemma-2-9b-it",
        device="cuda" if num_gpus > 0 else "cpu",
        n_devices=num_gpus if num_gpus > 0 else 0,
        dtype=torch.float16,
    )

    # Patch KV-cache device routing (round-robin across visible GPUs)
    importlib.reload(pkv)

    def _safe_get_device_for_block_index(index, cfg, device=None):
        if torch.cuda.device_count() == 0:
            return torch.device("cpu")
        idx = index % torch.cuda.device_count()
        return torch.device(f"cuda:{idx}")

    tl_devices.get_device_for_block_index = _safe_get_device_for_block_index
    pkv.get_device_for_block_index = _safe_get_device_for_block_index

    bad = []
    for i in range(model.cfg.n_layers):
        d = pkv.get_device_for_block_index(i, model.cfg, model.cfg.device)
        if isinstance(d, torch.device) and d.type == "cuda" and d.index is not None:
            if d.index >= torch.cuda.device_count():
                bad.append(d)
    print("KV-cache bad devices:", bad)  # should be []

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    tokenizer.pad_token       = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    return model, tokenizer


# -------------------- Data loader --------------------
def load_sorrybench_jsonl(path: Path):
    """
    Load all records from sorrybench_base440.jsonl.
    Returns:
        records: list[dict]
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {path}")
    return records


# -------------------- Hook helper --------------------
def make_pre_hook(cache_slice: torch.Tensor):
    """
    cache_slice: view into 'acts' with shape [batch_size, d_model] (CPU).
    Hook grabs resid_pre at POS_IDX (e.g., -2) and copies into cache_slice.
    """
    def hook(module, inputs):
        # inputs[0]: [batch, seq, d_model] = resid_pre
        x = inputs[0][:, POS_IDX, :].to(cache_slice.dtype).detach().cpu()  # [B, d_model]
        cache_slice.copy_(x)
    return hook


# -------------------- Main processing --------------------
def main():
    # 1) Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 2) Load data
    records = load_sorrybench_jsonl(DATA_PATH)

    # 3) Extract prompts for generation (from "turns" field)
    prompts = [str(r["turns"]) for r in records]

    # 4) Build chats using Gemma chat template
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in prompts
    ]

    # 5) Tokenize
    batch_enc = tokenizer(
        chats,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=SEQ_LEN,
    ).to(DEVICE)

    input_ids      = batch_enc["input_ids"]      # [N, T]
    attention_mask = batch_enc["attention_mask"]
    num_prompts, seq_len = input_ids.shape
    d_model        = model.cfg.d_model

    print(f"Input_ids shape: {tuple(input_ids.shape)}")

    # 6) Pre-allocate activation tensor on CPU
    # acts[n, layer_idx, d_model]  (layer_idx corresponds to CAPTURE_LAYERS)
    acts = torch.zeros(
        (num_prompts, len(CAPTURE_LAYERS), d_model),
        dtype=torch.float16,
        device="cpu",
    )

    # 7) Prepare container for model responses
    model_responses = [""] * num_prompts

    # 8) Process in batches:
    for b_start in tqdm(range(0, num_prompts, BATCH_SIZE), desc="Processing"):
        b_end   = min(b_start + BATCH_SIZE, num_prompts)
        tokens  = input_ids[b_start:b_end]
        attn    = attention_mask[b_start:b_end]

        # Register hooks to capture activations for this batch
        handles = []
        for layer_idx, layer_no in enumerate(CAPTURE_LAYERS):
            slice_ref = acts[b_start:b_end, layer_idx, :]  # [B, d_model]
            h = model.blocks[layer_no].register_forward_pre_hook(make_pre_hook(slice_ref))
            handles.append(h)

        with torch.no_grad():
            # 1) Forward pass (captures activations via hooks)
            _ = model(tokens, attention_mask=attn)

        # Remove hooks before generation
        for h in handles:
            h.remove()

        with torch.no_grad():
            # 2) Generate assistant continuations
            gen_tokens = model.generate(
                tokens,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        # Decode ONLY the newly generated tokens
        new_only   = gen_tokens[:, tokens.size(1):]
        gen_texts  = tokenizer.batch_decode(new_only, skip_special_tokens=True)

        # Store responses into corresponding records
        for i, resp in enumerate(gen_texts):
            idx = b_start + i
            model_responses[idx] = resp

        # Free CUDA memory between batches
        torch.cuda.empty_cache()
        gc.collect()

    # 9) Attach model responses to records
    assert len(records) == len(model_responses)
    for rec, resp in zip(records, model_responses):
        rec["model_response"] = resp

    # 10) Save generations as JSON (list of dicts, not JSONL)
    with open(GENS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved generations → {GENS_PATH.resolve()}")

    # 11) Save activations (plus metadata) as a .pt file
    torch.save(
        {
            "acts": acts.cpu(),
            "input_ids": input_ids.cpu(),
            "dataset_file": str(DATA_PATH),
            "sae_layers": SAE_LAYERS,
            "capture_layers": CAPTURE_LAYERS,
            "pos_idx": POS_IDX,
            "seq_len": SEQ_LEN,
        },
        ACTS_PATH,
    )
    print(f"Saved activations → {ACTS_PATH.resolve()}")

    # Optional: clean up
    del model, tokenizer, acts, input_ids, attention_mask
    torch.cuda.empty_cache()
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
