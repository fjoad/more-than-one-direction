#!/usr/bin/env python3
"""Comprehensive results across all Qwen directions tested so far.

Tables:
  1. Ablation effectiveness (% de-refused on H side) — per-split direction vs
     train64 vs t64H_ourBC (BC-isolation) vs 5 wgrandom seeds.
  2. Steering induction at α=25 (% over-refused on B side) for the same set.
  3. WG-random spread per split (mean/min/max across 5 seeds).
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

SHORT = {
    "WildGuard_all": "WG", "XSTest_all": "XST",
    "CocoNot_Humanizing": "CNHum", "CocoNot_Incomplete": "CNInc",
    "CocoNot_Indeterminate": "CNInd", "CocoNot_Safety": "CNSaf",
    "CocoNot_Unsupported": "CNUns",
    "SorryBench_HateSpeech": "SBHat", "SorryBench_CrimesTorts": "SBCri",
    "SorryBench_Inappropriate": "SBInp", "SorryBench_Advice": "SBAdv",
}


def cell(intervention: str, split: str):
    """Returns (n_h, ref_h, n_b, ref_b)."""
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
    if c is None or baseline_ref == 0:
        return None
    n_h, ref_h, _, _ = c
    return (baseline_ref - ref_h) / baseline_ref


def b_overref(intervention: str, split: str):
    c = cell(intervention, split)
    if c is None:
        return None
    _, _, n_b, ref_b = c
    return ref_b / n_b if n_b > 0 else None


# Build baseline ref-counts (same for all directions since baseline is hook-free)
baseline_ref = {}
for split in ALL_SPLITS:
    c = cell("baseline", split)
    if c is not None:
        baseline_ref[split] = c[1]


def fmt_pct(x):
    if x is None: return "  -"
    return f"{x*100:>3.0f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Ablation effectiveness (% de-refused) for each direction × split
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 130)
print("TABLE 1 — Ablation effectiveness: % of baseline-refused harmful prompts that flip to non-refusing")
print("(All directions use the SAME 32-prompt shared BC pool; only the H pool varies.)")
print("=" * 130)
print()

ablate_dirs = [
    ("per_split", "ablate"),                         # per-split's own direction (diagonal)
    ("train64", "ablate_train64"),                   # historical mixed pool
    ("t64H_ourBC", "ablate_t64H_ourBC"),             # train64-H + our BC (isolates BC)
    ("wgrand1", "ablate_wgrandom_seed1"),
    ("wgrand2", "ablate_wgrandom_seed2"),
    ("wgrand3", "ablate_wgrandom_seed3"),
    ("wgrand4", "ablate_wgrandom_seed4"),
    ("wgrand5", "ablate_wgrandom_seed5"),
]

# Header
hdr = f"{'Split':<25s} {'N_H':>4s} {'baseref':>8s}"
for short, _ in ablate_dirs:
    hdr += f" {short:>9s}"
print(hdr)
print("-" * len(hdr))
for split in ALL_SPLITS:
    c = cell("baseline", split)
    if c is None:
        continue
    n_h, ref_b, _, _ = c
    row = f"{split:<25s} {n_h:>4d} {ref_b:>3d}/{n_h:<4d}"
    for short, intervention in ablate_dirs:
        pct = pct_de_refused(intervention, split, ref_b)
        row += f" {fmt_pct(pct):>9s}"
    print(row)

print()
print("Legend: per_split = each split's own per-split direction (e.g., d_WG ablates WG test)")
print("        train64    = historical 64-prompt mixed pool (Gemma-conditioned)")
print("        t64H_ourBC = train64's H prompts + OUR shared BC (isolates BC pool variable)")
print("        wgrand1..5 = 5 random samples of 32 Qwen-refused from WildGuard_all")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Table 2: Steering induction at α=25 (% over-refused on B) per direction
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 130)
print("TABLE 2 — Steering induction at α=25: % of baseline-complying benign prompts that flip to refusal")
print("=" * 130)
print()

steer_dirs = [
    ("per_split", "steer_alpha25"),                            # per-split direction
    ("train64", "steer_train64_alpha25"),
    ("t64H_ourBC", "steer_t64H_ourBC_alpha25"),
    ("wgrand1", "steer_wgrandom_seed1_alpha25"),
    ("wgrand2", "steer_wgrandom_seed2_alpha25"),
    ("wgrand3", "steer_wgrandom_seed3_alpha25"),
    ("wgrand4", "steer_wgrandom_seed4_alpha25"),
    ("wgrand5", "steer_wgrandom_seed5_alpha25"),
]

hdr = f"{'Split':<25s} {'N_B':>4s}"
for short, _ in steer_dirs:
    hdr += f" {short:>9s}"
print(hdr)
print("-" * len(hdr))
for split in ALL_SPLITS:
    c = cell("baseline", split)
    if c is None: continue
    _, _, n_b, _ = c
    row = f"{split:<25s} {n_b:>4d}"
    for short, intervention in steer_dirs:
        v = b_overref(intervention, split)
        row += f" {fmt_pct(v):>9s}"
    print(row)

print()


# ─────────────────────────────────────────────────────────────────────────────
# Table 3: WG-random consistency — spread of ablation effectiveness across the
# 5 random seeds, per split. Tells us how stable/reproducible WG-random is.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 130)
print("TABLE 3 — WG-random ablation effectiveness spread across 5 seeds (% de-refused)")
print("=" * 130)
print()

print(f"{'Split':<25s} {'N_H':>4s} {'baseref':>8s}   {'seed1':>5s} {'seed2':>5s} {'seed3':>5s} "
      f"{'seed4':>5s} {'seed5':>5s}    {'mean':>6s} {'min':>5s} {'max':>5s} {'  vs t64':>9s}")
print("-" * 130)
for split in ALL_SPLITS:
    c = cell("baseline", split)
    if c is None: continue
    n_h, ref_b, _, _ = c
    seeds = []
    for s in [1, 2, 3, 4, 5]:
        v = pct_de_refused(f"ablate_wgrandom_seed{s}", split, ref_b)
        seeds.append(v if v is not None else 0)
    mean_v = sum(seeds) / len(seeds)
    min_v = min(seeds); max_v = max(seeds)
    t64_v = pct_de_refused("ablate_train64", split, ref_b) or 0
    row = (f"{split:<25s} {n_h:>4d} {ref_b:>3d}/{n_h:<4d}   "
           + " ".join(fmt_pct(s).rjust(5) for s in seeds)
           + f"    {fmt_pct(mean_v):>6s} {fmt_pct(min_v):>5s} {fmt_pct(max_v):>5s}"
           + f"    {fmt_pct(t64_v):>5s}")
    print(row)

print()
print("Reading: if the 5 seeds cluster tightly → WG-random is reproducible (random N=32 from WG suffices)")
print("         if they spread wide → seed luck matters at N=32")
print("         vs t64 = train64 number for the same split, for direct compare")
