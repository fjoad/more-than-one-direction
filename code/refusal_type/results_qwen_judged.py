#!/usr/bin/env python3
"""Pretty-print the judged Qwen discover results.

Reads generations/discover_qwen/judged/<intervention>/<split>.json (using is_refusal_wg).

Two tables, both diagonal-only:
  1. Ablation effectiveness per split: % of baseline-refused harmful prompts that
     became NOT refused after ablation. Higher = stronger ablation. We include
     train64 as a separate row for comparison against per-split directions.
  2. Steering induction (over-refusal on benign) per (split × α): % of
     baseline-complying benign prompts that flip to refusal under steering.
     Saturation = 1.0. Train64 sweep is a separate sub-block.
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
ALPHAS = [1, 2, 5, 10, 25, 50, 100]


def cell_counts(intervention: str, split: str) -> tuple[int, int, int, int]:
    """Returns (n_h, ref_h, n_b, ref_b) for one judged file. Uses is_refusal_wg."""
    path = ROOT / intervention / f"{split}.json"
    if not path.exists():
        return (0, 0, 0, 0)
    rows = json.load(open(path))
    H = [r for r in rows if r.get("side") == "H"]
    B = [r for r in rows if r.get("side") == "B"]
    ref_h = sum(1 for r in H if r.get("is_refusal_wg") == 1)
    ref_b = sum(1 for r in B if r.get("is_refusal_wg") == 1)
    return (len(H), ref_h, len(B), ref_b)


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Baseline + Ablation (H-side, "still-refusing rate after intervention")
# Columns: baseline H-RR, per-split ablate H-RR, train64 ablate H-RR,
#          and the corresponding "% de-refused" for each ablate direction.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 110)
print("TABLE 1 — Ablation effectiveness, diagonal cells (Qwen-1.8B-Chat, WildGuard judge)")
print("=" * 110)
print()
print(f"{'Split':<27s} {'N_H':>4s} {'base':>6s} {'absplit':>8s} {'  Δsplit':>10s} "
      f"{'abl_t64':>8s} {'  Δt64':>10s}")
print("-" * 110)
for split in ALL_SPLITS:
    n_h, ref_base, n_b, _ = cell_counts("baseline", split)
    if n_h == 0:
        continue
    base_rr = ref_base / n_h
    _, ref_a_split, _, _ = cell_counts("ablate", split)
    _, ref_a_t64, _, _ = cell_counts("ablate_train64", split)
    a_split_rr = ref_a_split / n_h
    a_t64_rr = ref_a_t64 / n_h
    de_split = (ref_base - ref_a_split) / max(ref_base, 1)
    de_t64 = (ref_base - ref_a_t64) / max(ref_base, 1)
    print(f"{split:<27s} {n_h:>4d} {base_rr:>6.2f} {a_split_rr:>8.2f} {de_split*100:>8.0f}%  "
          f"{a_t64_rr:>8.2f} {de_t64*100:>8.0f}%")
print()
print("Legend: base = baseline H-side refusal rate (≈1 by test-pool construction)")
print("        absplit = post-ablation H-RR using the split's OWN per-split direction")
print("        Δsplit  = % of baseline-refused prompts that flipped to non-refusing")
print("        abl_t64, Δt64 = same but ablating with train64 direction (model-agnostic reference)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Table 2: Steering induction sweep (B-side, "fraction of complies that flipped to refuse")
# ─────────────────────────────────────────────────────────────────────────────
def b_overref_rate(intervention: str, split: str, n_b_baseline: int):
    n_b, _, _, ref_b = cell_counts(intervention, split)
    return ref_b / n_b if n_b > 0 else None


print("=" * 130)
print("TABLE 2 — Per-split steering induction on benign side, B-RR (Qwen)")
print("        (% of baseline-complying benign prompts that flip to refusal under α·d_split)")
print("=" * 130)
print()
header = f"{'Split':<27s} {'N_B':>4s} " + " ".join(f"α={a:<3d}".rjust(7) for a in ALPHAS)
print(header)
print("-" * len(header))
for split in ALL_SPLITS:
    _, _, n_b, _ = cell_counts("baseline", split)
    if n_b == 0:
        continue
    cells = []
    for a in ALPHAS:
        rate = b_overref_rate(f"steer_alpha{a}", split, n_b)
        cells.append(f"{rate:>6.2f}" if rate is not None else "    -")
    print(f"{split:<27s} {n_b:>4d} " + " ".join(c.rjust(7) for c in cells))
print()


print("=" * 130)
print("TABLE 3 — Train64 steering induction (single direction across all 11 splits, Qwen)")
print("=" * 130)
print()
print(header)
print("-" * len(header))
for split in ALL_SPLITS:
    _, _, n_b, _ = cell_counts("baseline", split)
    if n_b == 0:
        continue
    cells = []
    for a in ALPHAS:
        rate = b_overref_rate(f"steer_train64_alpha{a}", split, n_b)
        cells.append(f"{rate:>6.2f}" if rate is not None else "    -")
    print(f"{split:<27s} {n_b:>4d} " + " ".join(c.rjust(7) for c in cells))
print()


# ─────────────────────────────────────────────────────────────────────────────
# Table 4: Summary of asymmetry — for each direction, baseline-vs-ablate H-side
# AND saturating-α steering B-side. The asymmetry question: do these directions
# both ablate refusal AND induce refusal? If both are non-zero, no asymmetry.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 110)
print("TABLE 4 — Asymmetry summary: ablation-effectiveness vs steering-induction at α=25 (saturating)")
print("=" * 110)
print()
print(f"{'Split':<27s} | {'per-split direction':<35s} | {'train64 direction':<35s}")
print(f"{'':<27s} | {'Δref-H':>8s} {'%-de':>5s} {'orr@α25':>8s}  | "
      f"{'Δref-H':>8s} {'%-de':>5s} {'orr@α25':>8s}")
print("-" * 110)
for split in ALL_SPLITS:
    n_h, ref_base, n_b, _ = cell_counts("baseline", split)
    if n_h == 0:
        continue
    _, ref_a_split, _, ref_b_a_split = cell_counts("ablate", split)
    _, _, _, ref_b_s_split = cell_counts("steer_alpha25", split)
    _, ref_a_t64, _, _ = cell_counts("ablate_train64", split)
    _, _, _, ref_b_s_t64 = cell_counts("steer_train64_alpha25", split)
    de_split = (ref_base - ref_a_split) / max(ref_base, 1)
    de_t64 = (ref_base - ref_a_t64) / max(ref_base, 1)
    orr_split = ref_b_s_split / n_b
    orr_t64 = ref_b_s_t64 / n_b
    print(f"{split:<27s} | "
          f"{ref_base - ref_a_split:>+5d}/{ref_base:<2d} {de_split*100:>4.0f}% {orr_split:>8.2f}  | "
          f"{ref_base - ref_a_t64:>+5d}/{ref_base:<2d} {de_t64*100:>4.0f}% {orr_t64:>8.2f}")
