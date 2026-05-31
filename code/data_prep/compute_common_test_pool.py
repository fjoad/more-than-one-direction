#!/usr/bin/env python3
"""Compute the size of a common test pool per split that is mutually exclusive
with the union of all training pools (per-split × 11, train64, B1 v2, B2 v2).

Also report the intersection-with-existing-ablate-outputs (the feasible
no-new-generation pool size)."""
import json
from pathlib import Path

EXP = Path("/export/home/fjoad/llm_safety_project/exp1_v10")
JUDGED = EXP / "data" / "baselines" / "qwen" / "judged"
POOLS = EXP / "data" / "training_pools" / "qwen"
LEGACY = EXP / "data" / "legacy"
RESP = EXP / "generations" / "discover_qwen" / "responses"

ALL_SPLITS = [
    "WildGuard_all", "XSTest_all",
    "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
    "CocoNot_Safety", "CocoNot_Unsupported",
    "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
    "SorryBench_Inappropriate", "SorryBench_Advice",
]


def load_prompts(path):
    if not path.exists(): return set()
    data = json.load(open(path))
    return {r["prompt"] for r in data if "prompt" in r}


def load_legacy_h_b():
    rows = json.load(open(LEGACY / "train_subset64_working.json"))
    h = {r["prompt"] for r in rows if r.get("prompt_harm_label") == "harmful"}
    b = {r["prompt"] for r in rows if r.get("prompt_harm_label") == "unharmful"}
    return h, b


def load_ablate_prompts(recipe, split):
    # Per-split lives at responses/ablate/<split>.json; others at responses/ablate_<recipe>/<split>.json
    if recipe in ("", "persplit"):
        p = RESP / "ablate" / f"{split}.json"
    else:
        p = RESP / f"ablate_{recipe}" / f"{split}.json"
    if not p.exists(): return set(), set()
    data = json.load(open(p))
    h = {r["prompt"] for r in data if r.get("side") == "H"}
    b = {r["prompt"] for r in data if r.get("side") == "B"}
    return h, b


# Build the union of all H training prompts
all_persplit_h = set()
all_persplit_b = set()
for split in ALL_SPLITS:
    all_persplit_h |= load_prompts(POOLS / "per_split_simple" / split / "H.json")
    all_persplit_b |= load_prompts(POOLS / "per_split_simple" / split / "B.json")

t64_h, t64_b = load_legacy_h_b()
b1_h = load_prompts(POOLS / "forced_b1" / "H.json")
b1_b = load_prompts(POOLS / "forced_b1" / "B.json")
b2_h = load_prompts(POOLS / "forced_b2" / "H.json")
b2_b = load_prompts(POOLS / "forced_b2" / "B.json")

union_train_h = all_persplit_h | t64_h | b1_h | b2_h
union_train_b = all_persplit_b | t64_b | b1_b | b2_b

print(f"Training pool sizes (unique prompts):")
print(f"  per-split H total (across 11 splits): {len(all_persplit_h)}")
print(f"  per-split B total (shared across splits): {len(all_persplit_b)}")
print(f"  train64 H: {len(t64_h)}    train64 B: {len(t64_b)}")
print(f"  B1 v2 H: {len(b1_h)}       B1 v2 B: {len(b1_b)}")
print(f"  B2 v2 H: {len(b2_h)}       B2 v2 B: {len(b2_b)}")
print(f"  ----------------------------------------")
print(f"  UNION H (to exclude): {len(union_train_h)}")
print(f"  UNION B (to exclude): {len(union_train_b)}")
print()

# For each split, compute:
# 1. Total baseline-refused H (and complied B from _BC_source)
# 2. Available after excluding all training
# 3. Intersection with each direction's existing ablate output (i.e., available
#    without re-generation)

# Load BC source (shared across all splits) once
bc_judged = json.load(open(JUDGED / "_BC_source.json"))
bc_complied = {r["prompt"] for r in bc_judged if r.get("is_refusal_wg") == 0}
bc_available = bc_complied - union_train_b
print(f"BC: {len(bc_complied)} baseline-complied total | excluding training: {len(bc_available)} remaining")
print()

# For H side, per split
print(f"{'Split':<26} {'baseline-refused':>16} {'after-exclude-train':>22} | {'per-split':>11} {'train64':>9} {'B1':>5} {'B2':>5} {'intersection-of-all-4-outputs':>32}")
print("-" * 140)
total_baseline = 0
total_after_excl = 0
total_intersect = 0
for split in ALL_SPLITS:
    judged = json.load(open(JUDGED / f"{split}.json"))
    refused = {r["prompt"] for r in judged
               if r.get("prompt_harm_label") == "harmful"
               and r.get("is_refusal_wg") == 1}
    after_excl = refused - union_train_h

    # Each direction's existing ablate H output
    ps_h, _ = load_ablate_prompts("persplit", split)
    t64_test_h, _ = load_ablate_prompts("train64", split)
    b1_test_h, _ = load_ablate_prompts("b1", split)
    b2_test_h, _ = load_ablate_prompts("b2", split)

    # The intersection of "in all 4 existing outputs" AND "not in any training"
    common = ps_h & t64_test_h & b1_test_h & b2_test_h & after_excl
    print(f"{split:<26} {len(refused):>16} {len(after_excl):>22} | {len(ps_h):>11} {len(t64_test_h):>9} {len(b1_test_h):>5} {len(b2_test_h):>5} {len(common):>32}")
    total_baseline += len(refused)
    total_after_excl += len(after_excl)
    total_intersect += len(common)

print("-" * 140)
print(f"{'TOTAL':<26} {total_baseline:>16} {total_after_excl:>22} | {total_intersect:>69}")
print()
print(f"NOTE: 'per-split' column shows the size of ablate/<split>.json's H side")
print(f"      'intersection' = prompts in all 4 ablate output files AND not in any training pool")
