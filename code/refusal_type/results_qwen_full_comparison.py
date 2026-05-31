#!/usr/bin/env python3
"""Full Qwen ablation comparison — all recipes side by side.

Cells that haven't been judged yet show "-" so this can be run pre/post each job.

Canonical reference: train64 (Gemma-conditioned, .head(32) of each filtered pool).
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


baseline = {s: cell("baseline", s) for s in ALL_SPLITS}
ref_n = {s: (baseline[s][1] if baseline[s] else 0) for s in ALL_SPLITS}
TOTAL_N_H = sum(ref_n.values())


def single_dir_means(intervention):
    """Per-split de-refused for a single direction (e.g. train64, qwen_orig_first32)."""
    return [pct_de(intervention, s, ref_n[s]) for s in ALL_SPLITS]


def per_seed_means(prefix):
    """For multi-seed recipe, return dict: split -> [val_seed1, val_seed2, ...]."""
    out = {}
    for s in ALL_SPLITS:
        out[s] = [pct_de(f"ablate_{prefix}_seed{k}", s, ref_n[s]) for k in SEEDS]
    return out


def recipe_mean_per_split(prefix):
    psm = per_seed_means(prefix)
    return [statistics.fmean([v for v in psm[s] if v is not None]) if any(v is not None for v in psm[s]) else None
            for s in ALL_SPLITS]


def weighted_overall_single(intervention):
    total_baseref = 0
    total_de = 0
    for s in ALL_SPLITS:
        c = cell(intervention, s)
        if c is None: continue
        baseref = ref_n[s]
        if baseref == 0: continue
        de = baseref - c[1]
        total_baseref += baseref
        total_de += de
    return total_de / total_baseref if total_baseref else None


def weighted_overall_multiseed(prefix):
    per_seed_w = []
    for k in SEEDS:
        total_baseref = 0; total_de = 0
        for s in ALL_SPLITS:
            c = cell(f"ablate_{prefix}_seed{k}", s)
            if c is None: continue
            baseref = ref_n[s]
            if baseref == 0: continue
            total_baseref += baseref
            total_de += baseref - c[1]
        if total_baseref > 0:
            per_seed_w.append(total_de / total_baseref)
    return statistics.fmean(per_seed_w) if per_seed_w else None


# Each row: (label, type, intervention_or_prefix, description)
ROWS = [
    ("train64 (Gemma, REFERENCE)",     "single",  "ablate_train64",              "Gemma-baseline .head(32) HR + .head(32) BC"),
    ("per_split (11 dirs, ourBC)",     "split",   None,                          "one direction per split, our shared BC"),
    ("--",                              "sep",     None,                          None),
    ("Phase 2 — WRONG-POOL (string-match labels, 11,524/7,705)", "header", None, None),
    ("t64bc_randH",                    "multi",   "phase2_t64bc_randH",          "random H from 11,524 str-match-HR + train64 BC"),
    ("t64h_randBC",                    "multi",   "phase2_t64h_randBC",          "train64 H + random BC from 7,705 str-match-BC"),
    ("fullrand",                       "multi",   "phase2_fullrand",             "random H + random BC (both str-match-cond)"),
    ("others_randH (Phase 2b)",        "multi",   "phase2_others_randH",         "random H from 2,530 'others'-only + train64 BC"),
    ("--",                              "sep",     None,                          None),
    ("Phase 2-CORRECTED — actual train64 pool (8,955/7,010 WGM labels)", "header", None, None),
    ("phase2corr_t64bc_randH",         "multi",   "phase2corr_t64bc_randH",      "random H from 8,955 WGM-HR + train64 BC"),
    ("phase2corr_t64h_randBC",         "multi",   "phase2corr_t64h_randBC",      "train64 H + random BC from 7,010 WGM-BC"),
    ("phase2corr_fullrand",            "multi",   "phase2corr_fullrand",         "random H + random BC (both WGM-cond)"),
    ("phase2corr_others_randH",        "multi",   "phase2corr_others_randH",     "random H from 1,440 WGM-HR ∩ others + train64 BC"),
    ("crisp_randH (2-phrase)",         "multi",   "crisp_randH",                 "random H from 813 Qwen-refused ∩ short ∩ ?-form ∩ crisp-2 (Sorry/Cannot), train64 BC"),
    ("crisp12_randH (Arditi-12)",      "multi",   "crisp12_randH",               "random H from 904 Qwen-refused ∩ short ∩ ?-form ∩ crisp-12 (Arditi phrases), train64 BC"),
    ("--",                              "sep",     None,                          None),
    ("Phase 3 — Qwen-conditioned",     "header",  None,                          None),
    ("qwen_orig_first32 (lit. equiv)", "single",  "ablate_qwen_orig_first32",    ".head(32) of Qwen-HR + .head(32) of Qwen-BC"),
    ("qwen_t64bc_randH (5-seed)",      "multi",   "qwen_t64bc_randH",            "random H from 8,497 Qwen-HR + train64 BC"),
    ("qwen_t64h_randBC (5-seed)",      "multi",   "qwen_t64h_randBC",            "train64 H + random BC from 7,591 Qwen-BC"),
    ("qwen_fullrand (5-seed)",         "multi",   "qwen_fullrand",               "random H + random BC (both Qwen-cond)"),
]


def per_split_combined(s):
    """Per-split de-refused for the 'per_split' row: each split tested on its own direction."""
    return pct_de("ablate", s, ref_n[s])


# ── Per-split table ──
print("=" * 150)
print("FULL QWEN ABLATION COMPARISON — per-split % de-refused, all recipes")
print("=" * 150)
print()
header = f"{'Recipe':<32s}    " + "  ".join(f"{s.split('_')[0][:4]+s.split('_')[-1][:3]:>7s}" for s in ALL_SPLITS) + f"   {'unwtd':>5s}  {'wtd':>5s}  {'vs t64':>7s}"
print(header)
print("-" * 150)

# Compute train64's weighted mean for vs-t64 column
t64_w = weighted_overall_single("ablate_train64")

for label, kind, intervention_or_prefix, _ in ROWS:
    if kind == "sep":
        print()
        continue
    if kind == "header":
        print(f"{label:<32s}")
        continue
    if kind == "single":
        vals = single_dir_means(intervention_or_prefix)
        unw_clean = [v for v in vals if v is not None]
        unw = statistics.fmean(unw_clean) if unw_clean else None
        wtd = weighted_overall_single(intervention_or_prefix)
    elif kind == "multi":
        vals = recipe_mean_per_split(intervention_or_prefix)
        unw_clean = [v for v in vals if v is not None]
        unw = statistics.fmean(unw_clean) if unw_clean else None
        wtd = weighted_overall_multiseed(intervention_or_prefix)
    elif kind == "split":
        vals = [per_split_combined(s) for s in ALL_SPLITS]
        unw_clean = [v for v in vals if v is not None]
        unw = statistics.fmean(unw_clean) if unw_clean else None
        # weighted: sum per split
        total_b = 0; total_de = 0
        for s in ALL_SPLITS:
            c = cell("ablate", s)
            if c is None: continue
            total_b += ref_n[s]
            total_de += ref_n[s] - c[1]
        wtd = total_de / total_b if total_b else None

    diff = (wtd - t64_w) * 100 if (wtd is not None and t64_w is not None) else None
    diff_str = f"{diff:+5.0f}pp" if diff is not None else "-"
    row = f"{label:<32s}    " + "  ".join(f"{fmt(v):>7s}" for v in vals) + f"   {fmt(unw):>5s}  {fmt(wtd):>5s}  {diff_str:>7s}"
    print(row)

print()
print(f"Total H pool tested: {TOTAL_N_H} prompts across 11 splits.")
print()
print(f"{'Split':<25s} N_H")
for s in ALL_SPLITS:
    print(f"  {s:<23s} {ref_n[s]:>3d}")


# ── Headline summary ──
print()
print("=" * 150)
print("HEADLINE SUMMARY — weighted (prompt-level) means across 11 splits")
print("=" * 150)
print()
print(f"{'Recipe':<35s} {'weighted':>10s}    {'vs train64 (Gemma)':>20s}")
print("-" * 100)
for label, kind, prefix, _ in ROWS:
    if kind in ("sep", "header"):
        if kind == "header": print(f"{label}")
        else: print()
        continue
    if kind == "single":
        w = weighted_overall_single(prefix)
    elif kind == "multi":
        w = weighted_overall_multiseed(prefix)
    elif kind == "split":
        total_b = 0; total_de = 0
        for s in ALL_SPLITS:
            c = cell("ablate", s)
            if c is None: continue
            total_b += ref_n[s]
            total_de += ref_n[s] - c[1]
        w = total_de / total_b if total_b else None
    diff = (w - t64_w) * 100 if (w is not None and t64_w is not None) else None
    diff_str = f"{diff:+5.0f}pp" if diff is not None else "-"
    print(f"{label:<35s} {fmt(w):>10s}    {diff_str:>20s}")
