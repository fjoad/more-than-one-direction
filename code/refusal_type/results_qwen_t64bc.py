#!/usr/bin/env python3
"""Comprehensive results: H-pool × BC-pool factorial design (Qwen, ablation).

For each of the 11 test splits, report % of baseline-refused harmful prompts
that flip to non-refusing, under each (H source, BC source) direction.

H sources:
  - per_split   : the split's own per-split H
  - train64     : train64's 32 H (legacy/train_subset64_working.json rows 0-31)
  - wgrand{1..5}: 5 random samples of 32 from WildGuard_all Qwen-refused

BC sources:
  - ourBC : data/training_pools/qwen/per_split_simple/<any>/B.json (32 Qwen-complied
            from WildGuardMix-test benign)
  - t64BC : legacy/train_subset64_working.json rows 32-63 (32 Gemma-complied from
            WildGuardMix-train benign)
"""
from pathlib import Path
import json

ROOT = Path("/export/home/fjoad/llm_safety_project/exp1_v10/generations/discover_qwen/judged")

ALL_SPLITS = [
    "WildGuard_all", "XSTest_all",
    "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
    "CocoNot_Safety", "CocoNot_Unsupported",
    "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
    "SorryBench_Inappropriate", "SorryBench_Advice",
]
WGRAND_SEEDS = [1, 2, 3, 4, 5]


def cell(intervention: str, split: str):
    p = ROOT / intervention / f"{split}.json"
    if not p.exists():
        return None
    rows = json.load(open(p))
    H = [r for r in rows if r.get("side") == "H"]
    B = [r for r in rows if r.get("side") == "B"]
    return (len(H),
            sum(1 for r in H if r.get("is_refusal_wg") == 1),
            len(B),
            sum(1 for r in B if r.get("is_refusal_wg") == 1))


def pct_de_refused(intervention: str, split: str, baseline_ref):
    c = cell(intervention, split)
    if c is None or baseline_ref == 0: return None
    n_h, ref_h, _, _ = c
    return (baseline_ref - ref_h) / baseline_ref


def b_overref(intervention: str, split: str):
    c = cell(intervention, split)
    if c is None: return None
    _, _, n_b, ref_b = c
    return ref_b / n_b if n_b > 0 else None


def fmt(x):
    if x is None: return "  -"
    return f"{x*100:>3.0f}%"


# Get baseline refusal counts per split
baseline = {s: cell("baseline", s) for s in ALL_SPLITS}


def per_split_ablate_pct(split, bc_variant):
    """Return % de-refused for the per-split direction on its diagonal split."""
    n_h, ref_b, _, _ = baseline[split]
    intervention = "ablate" if bc_variant == "ourBC" else f"ablate_persplit_{split}_t64BC"
    return pct_de_refused(intervention, split, ref_b)


def train64_ablate_pct(split, bc_variant):
    n_h, ref_b, _, _ = baseline[split]
    intervention = "ablate_train64" if bc_variant == "t64BC" else "ablate_t64H_ourBC"
    return pct_de_refused(intervention, split, ref_b)


def wgrand_seed_ablate_pct(split, seed, bc_variant):
    n_h, ref_b, _, _ = baseline[split]
    suffix = "_t64BC" if bc_variant == "t64BC" else ""
    return pct_de_refused(f"ablate_wgrandom_seed{seed}{suffix}", split, ref_b)


def wgrand_mean_ablate_pct(split, bc_variant):
    vals = [wgrand_seed_ablate_pct(split, s, bc_variant) for s in WGRAND_SEEDS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# Steering versions (α=25)
def per_split_steer_pct(split, bc_variant):
    intervention = "steer_alpha25" if bc_variant == "ourBC" else f"steer_persplit_{split}_t64BC_alpha25"
    return b_overref(intervention, split)


def train64_steer_pct(split, bc_variant):
    intervention = "steer_train64_alpha25" if bc_variant == "t64BC" else "steer_t64H_ourBC_alpha25"
    return b_overref(intervention, split)


def wgrand_seed_steer_pct(split, seed, bc_variant):
    suffix = "_t64BC" if bc_variant == "t64BC" else ""
    return b_overref(f"steer_wgrandom_seed{seed}{suffix}_alpha25", split)


def wgrand_mean_steer_pct(split, bc_variant):
    vals = [wgrand_seed_steer_pct(split, s, bc_variant) for s in WGRAND_SEEDS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# ───────────────────────────────────────────────────────────────────────────
# TABLE A — Ablation effectiveness, full H × BC factorial
# ───────────────────────────────────────────────────────────────────────────
print("=" * 130)
print("TABLE A — Ablation effectiveness (% de-refused on H side), H pool × BC pool factorial")
print("=" * 130)
print()
print(f"{'Split':<25s} {'N_H':>4s}    "
      f"{'per-split':>22s}        {'train64-H':>22s}        {'wgrand mean (5 seeds)':>26s}")
print(f"{'':<25s} {'':>4s}    {'ourBC':>10s} {'t64BC':>10s}        "
      f"{'ourBC':>10s} {'t64BC':>10s}        {'ourBC':>12s} {'t64BC':>12s}")
print("-" * 130)
for split in ALL_SPLITS:
    n_h, _, _, _ = baseline[split]
    ps_o = per_split_ablate_pct(split, "ourBC")
    ps_t = per_split_ablate_pct(split, "t64BC")
    t64_o = train64_ablate_pct(split, "ourBC")
    t64_t = train64_ablate_pct(split, "t64BC")
    wg_o = wgrand_mean_ablate_pct(split, "ourBC")
    wg_t = wgrand_mean_ablate_pct(split, "t64BC")
    print(f"{split:<25s} {n_h:>4d}    "
          f"{fmt(ps_o):>10s} {fmt(ps_t):>10s}        "
          f"{fmt(t64_o):>10s} {fmt(t64_t):>10s}        "
          f"{fmt(wg_o):>12s} {fmt(wg_t):>12s}")
print()
print("Each entry = % of baseline-refused prompts that flip to non-refusing.")
print("Higher = stronger ablation. Compare ourBC column to t64BC column to see BC effect.")
print()


# ───────────────────────────────────────────────────────────────────────────
# TABLE B — BC effect summary: same H pool, swap BC
# ───────────────────────────────────────────────────────────────────────────
print("=" * 130)
print("TABLE B — BC effect: how much does swapping BC (our → train64) change ablation effectiveness?")
print("=" * 130)
print()
print(f"{'Split':<25s}    "
      f"{'per-split:  ourBC → t64BC (Δ)':>34s}    "
      f"{'train64-H:  ourBC → t64BC (Δ)':>34s}    "
      f"{'wgrand:    ourBC → t64BC (Δ)':>32s}")
print("-" * 130)
for split in ALL_SPLITS:
    ps_o = per_split_ablate_pct(split, "ourBC") or 0
    ps_t = per_split_ablate_pct(split, "t64BC") or 0
    t64_o = train64_ablate_pct(split, "ourBC") or 0
    t64_t = train64_ablate_pct(split, "t64BC") or 0
    wg_o = wgrand_mean_ablate_pct(split, "ourBC") or 0
    wg_t = wgrand_mean_ablate_pct(split, "t64BC") or 0
    def diff(a, b):
        d = (b - a) * 100
        sgn = "+" if d >= 0 else ""
        return f"{sgn}{d:>4.0f}pp"
    print(f"{split:<25s}    "
          f"{fmt(ps_o):>5s} → {fmt(ps_t):>5s}  ({diff(ps_o, ps_t):>7s})    "
          f"{fmt(t64_o):>5s} → {fmt(t64_t):>5s}  ({diff(t64_o, t64_t):>7s})    "
          f"{fmt(wg_o):>5s} → {fmt(wg_t):>5s}  ({diff(wg_o, wg_t):>7s})")
print()


# ───────────────────────────────────────────────────────────────────────────
# TABLE C — Steering induction at α=25 for every condition
# ───────────────────────────────────────────────────────────────────────────
print("=" * 130)
print("TABLE C — Steering induction at α=25 (% over-refused on B side)")
print("=" * 130)
print()
print(f"{'Split':<25s}    "
      f"{'per-split':>22s}        {'train64-H':>22s}        {'wgrand mean':>22s}")
print(f"{'':<25s}    {'ourBC':>10s} {'t64BC':>10s}        "
      f"{'ourBC':>10s} {'t64BC':>10s}        {'ourBC':>10s} {'t64BC':>10s}")
print("-" * 130)
for split in ALL_SPLITS:
    print(f"{split:<25s}    "
          f"{fmt(per_split_steer_pct(split, 'ourBC')):>10s} {fmt(per_split_steer_pct(split, 't64BC')):>10s}        "
          f"{fmt(train64_steer_pct(split, 'ourBC')):>10s} {fmt(train64_steer_pct(split, 't64BC')):>10s}        "
          f"{fmt(wgrand_mean_steer_pct(split, 'ourBC')):>10s} {fmt(wgrand_mean_steer_pct(split, 't64BC')):>10s}")
print()
