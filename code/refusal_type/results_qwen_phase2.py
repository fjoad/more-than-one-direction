#!/usr/bin/env python3
"""Phase 2 results — answer 'did train64 get lucky' by comparing it against:

  - phase2_t64bc_randH_seed{1..5}: random H from full 11,524 HR + train64 BC
  - phase2_t64h_randBC_seed{1..5}: train64 H + random BC from 7,705 BC
  - phase2_fullrand_seed{1..5}:    random H + random BC
  - phase2_others_randH_seed{1..5}: random H from 2,530 'others'-only + train64 BC (Phase 2b)
"""
from pathlib import Path
import json
import statistics

ROOT = Path("/export/home/fjoad/llm_safety_project/exp1_v10/generations/discover_qwen/judged")
ALL_SPLITS = [
    "WildGuard_all", "XSTest_all",
    "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
    "CocoNot_Safety", "CocoNot_Unsupported",
    "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
    "SorryBench_Inappropriate", "SorryBench_Advice",
]
SEEDS = [1, 2, 3, 4, 5]


def cell(intervention, split):
    p = ROOT / intervention / f"{split}.json"
    if not p.exists(): return None
    rows = json.load(open(p))
    H = [r for r in rows if r.get("side") == "H"]
    return (len(H), sum(1 for r in H if r.get("is_refusal_wg") == 1))


def pct_de(intervention, split, baseref):
    c = cell(intervention, split)
    if c is None or baseref == 0: return None
    return (baseref - c[1]) / baseref


def fmt(x):
    if x is None: return "  -"
    return f"{x*100:>3.0f}%"


# Use the 'ablate' (per-split with our BC) baseline as the ref for N_H
baseline = {s: cell("baseline", s) for s in ALL_SPLITS}
ref_n = {s: baseline[s][1] for s in ALL_SPLITS}   # baseline-refused count = test pool's H size for each split


def recipe_5seed_mean(prefix, split):
    """Mean over 5 seeds of (% de-refused) for an ablate_<prefix>_seed{k} cell."""
    vals = [pct_de(f"ablate_{prefix}_seed{k}", split, ref_n[split]) for k in SEEDS]
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def recipe_seeds(prefix, split):
    """Per-seed values for a recipe on a split."""
    return [pct_de(f"ablate_{prefix}_seed{k}", split, ref_n[split]) for k in SEEDS]


def train64_de(split):
    return pct_de("ablate_train64", split, ref_n[split])


PHASE2_RECIPES = [
    ("t64bc_randH",   "phase2_t64bc_randH",  "random H from 11,524 HR  +  train64 BC"),
    ("t64h_randBC",   "phase2_t64h_randBC",  "train64 H  +  random BC from 7,705 BC"),
    ("fullrand",      "phase2_fullrand",     "random H + random BC (both random)"),
    ("others_randH",  "phase2_others_randH", "random H from 2,530 'others'-only  +  train64 BC"),
]


# ── Table A — per-split, recipe means side-by-side ──
print("=" * 130)
print("PHASE 2 ABLATION — 5-seed mean per split, vs train64 reference")
print("=" * 130)
print()
hdr = f"{'Split':<23s} {'N_H':>4s}     {'train64':>8s}   "
for short, _, _ in PHASE2_RECIPES:
    hdr += f"{short:>14s}   "
print(hdr)
print("-" * 130)
for s in ALL_SPLITS:
    row = f"{s:<23s} {ref_n[s]:>4d}     {fmt(train64_de(s)):>8s}   "
    for _, prefix, _ in PHASE2_RECIPES:
        row += f"{fmt(recipe_5seed_mean(prefix, s)):>14s}   "
    print(row)


# ── Table B — 5-seed spread per recipe ──
for short, prefix, descr in PHASE2_RECIPES:
    print()
    print()
    print("=" * 130)
    print(f"PHASE 2 — `{short}` (5-seed spread)")
    print(f"  Recipe: {descr}")
    print("=" * 130)
    print()
    print(f"{'Split':<23s}    "
          f"{'seed1':>5s} {'seed2':>5s} {'seed3':>5s} {'seed4':>5s} {'seed5':>5s}    "
          f"{'mean':>5s}  {'min':>5s} {'max':>5s}    {'vs t64':>7s}")
    print("-" * 130)
    for s in ALL_SPLITS:
        vals = recipe_seeds(prefix, s)
        clean = [v for v in vals if v is not None]
        mean = statistics.fmean(clean) if clean else None
        lo = min(clean) if clean else None
        hi = max(clean) if clean else None
        t64 = train64_de(s)
        diff = (mean - t64) * 100 if (mean is not None and t64 is not None) else None
        diff_str = f"{diff:+5.0f}pp" if diff is not None else "-"
        seed_strs = "  ".join(fmt(v).rjust(3) for v in vals)
        print(f"{s:<23s}    {seed_strs}    "
              f"{fmt(mean):>5s}  {fmt(lo):>5s} {fmt(hi):>5s}    {diff_str:>7s}")


# ── Headline row summary ──
# Two means: unweighted (avg of per-split percentages) and weighted (prompt-level).
# Weighted = total prompts de-refused / total H tested. This naturally down-weights
# small splits (e.g. HateSpeech N=5) so they don't dominate the average.

TOTAL_N_H = sum(ref_n[s] for s in ALL_SPLITS)


def weighted_mean_for_intervention(prefix, is_train64=False):
    """Prompt-level weighted: sum of de-refused prompts / sum of H prompts.
    For Phase 2 recipes (5 seeds), average per-seed weighted-means."""
    if is_train64:
        # single direction across splits
        total_baseref = 0
        total_de = 0
        for s in ALL_SPLITS:
            c = cell("ablate_train64", s)
            if c is None: continue
            baseref = ref_n[s]
            de = baseref - c[1]
            total_baseref += baseref
            total_de += de
        return total_de / total_baseref if total_baseref else None
    # 5-seed: compute weighted-mean per seed, then average over seeds
    per_seed = []
    for k in SEEDS:
        total_baseref = 0
        total_de = 0
        for s in ALL_SPLITS:
            c = cell(f"ablate_{prefix}_seed{k}", s)
            if c is None: continue
            baseref = ref_n[s]
            de = baseref - c[1]
            total_baseref += baseref
            total_de += de
        if total_baseref > 0:
            per_seed.append(total_de / total_baseref)
    return statistics.fmean(per_seed) if per_seed else None


print()
print()
print("=" * 130)
print(f"HEADLINE — mean across 11 splits, by recipe   (total H pool = {TOTAL_N_H} prompts)")
print("=" * 130)
print()
print(f"{'Recipe':<30s}   {'unweighted':>11s}   {'weighted (prompt-level)':>26s}   {'vs train64 (weighted)':>22s}")
print("-" * 130)

t64_unw = statistics.fmean([v for v in [train64_de(s) for s in ALL_SPLITS] if v is not None])
t64_w = weighted_mean_for_intervention(None, is_train64=True)
print(f"{'train64 (reference)':<30s}   {fmt(t64_unw):>11s}   {fmt(t64_w):>26s}   {'-':>22s}")

for short, prefix, _ in PHASE2_RECIPES:
    per_split_means = [recipe_5seed_mean(prefix, s) for s in ALL_SPLITS]
    per_split_means = [v for v in per_split_means if v is not None]
    unw = statistics.fmean(per_split_means) if per_split_means else None
    weighted = weighted_mean_for_intervention(prefix)
    diff = (weighted - t64_w) * 100 if (weighted is not None and t64_w is not None) else None
    diff_str = f"{diff:+5.0f}pp" if diff is not None else "-"
    print(f"{short:<30s}   {fmt(unw):>11s}   {fmt(weighted):>26s}   {diff_str:>22s}")

print()
print("Notes:")
print("  unweighted = mean of per-split percentages (equal weight per split, regardless of N).")
print("  weighted   = total H prompts de-refused / total H prompts tested across all 11 splits.")
print("               Naturally weights each split by its test pool size.")
print(f"  Per-split N_H: " + ", ".join(f"{s.split('_')[0]}{s.split('_')[-1][:3] if '_' in s else ''}={ref_n[s]}" for s in ALL_SPLITS[:5]) + ", ...")
