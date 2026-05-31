#!/usr/bin/env python3
"""Aggregate sharded Llama baseline response files into one per split.

Input: data/baselines/llama/responses/<split>_shard{0..N-1}.json
Output: data/baselines/llama/responses/<split>.json
"""
import json, os
from pathlib import Path

EXP = Path("/export/home/fjoad/llm_safety_project/exp1_v10")
RESP_DIR = EXP / "data" / "baselines" / "llama" / "responses"
SOURCES = EXP / "data" / "sources"
N_SHARDS = int(os.environ.get("N_SHARDS", "2"))

ALL_SPLITS = [
    "WildGuard_all", "XSTest_all",
    "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
    "CocoNot_Safety", "CocoNot_Unsupported",
    "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
    "SorryBench_Inappropriate", "SorryBench_Advice",
    "_BC_source",
]

for split in ALL_SPLITS:
    src = json.load(open(SOURCES / f"{split}.json"))
    n_total = len([r for r in src if isinstance(r.get("prompt", r.get("instruction")), str)])
    # Stitch shards in the order prompts appear in src
    shard_records = []
    for sid in range(N_SHARDS):
        p = RESP_DIR / f"{split}_shard{sid}.json"
        if not p.exists():
            raise SystemExit(f"Missing shard: {p}")
        shard_records.append(json.load(open(p)))
    # Re-interleave: shard i contains every Nth prompt starting at i.
    # So position k in src maps to shard (k % N_SHARDS), record (k // N_SHARDS).
    merged = [None] * n_total
    for sid, recs in enumerate(shard_records):
        for j, rec in enumerate(recs):
            pos = j * N_SHARDS + sid
            merged[pos] = rec
    if any(m is None for m in merged):
        raise SystemExit(f"{split}: missing records after merge")
    out = RESP_DIR / f"{split}.json"
    json.dump(merged, open(out, "w"), indent=2)
    print(f"  {split:<30}  {len(merged)} records → {out}")
print("Done.")
