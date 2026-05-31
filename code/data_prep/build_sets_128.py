#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_13_splits_extended.py

Build "extended" versions of the original 13 HR/BC test splits.

Goal:
- Target 128 HR + 128 BC per split where possible.
- Always include ALL examples already present in the original 13 splits.
- For splits with more available data, top up with new random examples
  from the same HR/BC pools as the original script.
- For splits without enough data to reach 128, use the maximum possible
  N_eff = min(128, HR_pool_size, BC_pool_size), keeping HR/BC balanced.
- For CocoNot and SorryBench splits, reuse WildGuard BC as in the original.

Input:
- Original evaluated generations:
    WildGuard: WILD_JSON
    XSTest:    XSTEST_JSON
    CocoNot:   COCONOT_JSON
    SorryBench: SORRY_JSON

- Original 13 splits:
    data/refusal_13splits/<split_name>.json

Output:
- Extended splits in:
    data/refusal_13splits_extended/<split_name>_HR{N}_BC{N}.json

Each record has:
    - "prompt"
    - "prompt_harm_label"  ("harmful" / "unharmful")
    - "was_refusal"        (0 / 1)
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, OrderedDict, Tuple
from collections import OrderedDict as OD

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────
SEED = 123
random.seed(SEED)

TARGET_N = 128  # desired HR/BC per split where possible

# Adjust these paths as needed (same as original script)
WILD_JSON = Path(
    "generations/original_wildguard/evaluated/"
    "wildguardtest_nonadv_wildguardtest_nonadversarial_generations_wildguard.json"
)

XSTEST_JSON = Path("generations/xstest/evaluated/test_generations_wildguard.json")

COCONOT_JSON = Path(
    "generations/coconot/original/evaluated/test_generations_wildguard.json"
)

SORRY_JSON = Path(
    "generations/sorrybench/evaluated/sorrybench_base440_generations_wildguard.json"
)

# Original and new split roots
OLD_SPLITS_ROOT = Path("data") / "refusal_13splits"
NEW_SPLITS_ROOT = Path("data") / "refusal_13splits_extended"

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a top-level list.")
    return data


def to_int01(x) -> int:
    try:
        v = int(x)
        return v if v in (0, 1) else None
    except Exception:
        return None


def mk_record(prompt: str, harm_label: str, was_refusal: int) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "prompt_harm_label": harm_label,
        "was_refusal": int(was_refusal),
    }


def split_hr_bc_from_records(recs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Given normalized records (prompt, prompt_harm_label, was_refusal),
    return (hr_list, bc_list) using the standard semantics:
      HR: harmful + was_refusal == 1
      BC: unharmful + was_refusal == 0
    """
    hr = []
    bc = []
    for r in recs:
        harm = (r.get("prompt_harm_label") or "").strip().lower()
        wr = to_int01(r.get("was_refusal"))
        if harm == "harmful" and wr == 1:
            hr.append(r)
        elif harm == "unharmful" and wr == 0:
            bc.append(r)
    return hr, bc


# ────────────────────────────────────────────────────────────────
# 1. Build HR/BC pools from each dataset (same as original script)
# ────────────────────────────────────────────────────────────────
print("=== Loading datasets and rebuilding HR/BC pools ===")

# 1.1 WildGuard
wg_recs = load_json_list(WILD_JSON)
wg_hr: List[Dict[str, Any]] = []
wg_bc: List[Dict[str, Any]] = []

for rec in wg_recs:
    prompt = rec.get("prompt") or rec.get("input") or ""
    harm = (rec.get("prompt_harm_label") or "").strip().lower()
    is_refusal = to_int01(rec.get("is_refusal"))
    if not prompt or is_refusal is None:
        continue

    if harm == "harmful" and is_refusal == 1:
        wg_hr.append(mk_record(prompt, "harmful", 1))
    elif harm in ("unharmful", "safe", "benign") and is_refusal == 0:
        wg_bc.append(mk_record(prompt, "unharmful", 0))

print(f"WildGuard pools → HR={len(wg_hr)}  BC={len(wg_bc)}")
if not wg_hr or not wg_bc:
    raise RuntimeError("WildGuard pools empty; check WILD_JSON or fields.")

# 1.2 XSTest
xs_recs = load_json_list(XSTEST_JSON)
xs_hr: List[Dict[str, Any]] = []
xs_bc: List[Dict[str, Any]] = []

for rec in xs_recs:
    prompt = rec.get("prompt") or ""
    label = (rec.get("label") or "").strip().lower()  # "safe" / "unsafe"
    is_refusal = to_int01(rec.get("is_refusal"))
    if not prompt or is_refusal is None:
        continue

    if label == "unsafe" and is_refusal == 1:
        xs_hr.append(mk_record(prompt, "harmful", 1))
    elif label == "safe" and is_refusal == 0:
        xs_bc.append(mk_record(prompt, "unharmful", 0))

print(f"XSTest pools   → HR={len(xs_hr)}  BC={len(xs_bc)}")
if not xs_hr or not xs_bc:
    raise RuntimeError("XSTest pools empty; check XSTEST_JSON or fields.")

# 1.3 CocoNot (all harmful; BC comes from WildGuard BC)
coco_recs = load_json_list(COCONOT_JSON)
coco_hr_all: List[Dict[str, Any]] = []
coco_hr_by_cat: Dict[str, List[Dict[str, Any]]] = {}

for rec in coco_recs:
    prompt = rec.get("prompt") or ""
    cat = (rec.get("category") or "").strip()
    is_refusal = to_int01(rec.get("is_refusal"))
    if not prompt or is_refusal is None:
        continue
    if is_refusal != 1:
        continue

    rec_obj = mk_record(prompt, "harmful", 1)
    coco_hr_all.append(rec_obj)
    coco_hr_by_cat.setdefault(cat, []).append(rec_obj)

print(
    f"CocoNot HR(all)={len(coco_hr_all)}, "
    f"categories={list(coco_hr_by_cat.keys())}"
)

# 1.4 SorryBench (all harmful; HR grouped; BC from WildGuard BC)
sb_recs = load_json_list(SORRY_JSON)
sb_hr_all: List[Dict[str, Any]] = []
sb_hr_by_group: Dict[str, List[Dict[str, Any]]] = {
    "hate_speech": [],
    "crimes_torts": [],
    "inappropriate_topics": [],
    "unqualified_advice": [],
}


def map_sb_group(cat_num: int) -> str:
    if 1 <= cat_num <= 5:
        return "hate_speech"
    if 6 <= cat_num <= 24:
        return "crimes_torts"
    if 25 <= cat_num <= 39:
        return "inappropriate_topics"
    if 40 <= cat_num <= 44:
        return "unqualified_advice"
    return "unknown"


for rec in sb_recs:
    prompt = rec.get("turns") or rec.get("prompt") or ""
    is_refusal = to_int01(rec.get("is_refusal"))
    try:
        cat_num = int(rec.get("category"))
    except Exception:
        cat_num = -1

    if not prompt or is_refusal is None:
        continue
    if is_refusal != 1:
        continue

    group = map_sb_group(cat_num)
    if group == "unknown":
        continue

    rec_obj = mk_record(prompt, "harmful", 1)
    sb_hr_all.append(rec_obj)
    sb_hr_by_group[group].append(rec_obj)

print(
    f"SorryBench HR(all)={len(sb_hr_all)} | "
    f"hate_speech={len(sb_hr_by_group['hate_speech'])}, "
    f"crimes_torts={len(sb_hr_by_group['crimes_torts'])}, "
    f"inappropriate_topics={len(sb_hr_by_group['inappropriate_topics'])}, "
    f"unqualified_advice={len(sb_hr_by_group['unqualified_advice'])}"
)
if not sb_hr_all:
    raise RuntimeError("SorryBench HR pool empty; check SORRY_JSON or fields.")

# ────────────────────────────────────────────────────────────────
# 2. Define 13 splits (HR pool + BC pool) same as original
# ────────────────────────────────────────────────────────────────
splits: "OrderedDict[str, Dict[str, List[Dict[str, Any]]]]" = OD()

# WildGuard and XSTest have their own BC; others reuse WildGuard BC.
splits["WildGuard_all"] = {"hr": wg_hr, "bc": wg_bc}
splits["XSTest_all"] = {"hr": xs_hr, "bc": xs_bc}

# CocoNot: all + each category (BC = WildGuard BC)
splits["CocoNot_all"] = {"hr": coco_hr_all, "bc": wg_bc}
for cat in sorted(coco_hr_by_cat.keys()):
    slug = cat.strip().replace(" ", "_")
    splits[f"CocoNot_cat_{slug}"] = {"hr": coco_hr_by_cat[cat], "bc": wg_bc}

# SorryBench: all + 4 groups (BC = WildGuard BC)
splits["SorryBench_all"] = {"hr": sb_hr_all, "bc": wg_bc}
splits["SorryBench_hate_speech"] = {
    "hr": sb_hr_by_group["hate_speech"],
    "bc": wg_bc,
}
splits["SorryBench_crimes_torts"] = {
    "hr": sb_hr_by_group["crimes_torts"],
    "bc": wg_bc,
}
splits["SorryBench_inappropriate_topics"] = {
    "hr": sb_hr_by_group["inappropriate_topics"],
    "bc": wg_bc,
}
splits["SorryBench_unqualified_advice"] = {
    "hr": sb_hr_by_group["unqualified_advice"],
    "bc": wg_bc,
}

print(f"\nTotal splits defined: {len(splits)} (expecting 13)")

# ────────────────────────────────────────────────────────────────
# 3. Compute per-split HR/BC and target N_eff
# ────────────────────────────────────────────────────────────────
per_split_counts: Dict[str, Tuple[int, int]] = {}
for name, pools in splits.items():
    h = len(pools["hr"])
    b = len(pools["bc"])
    per_split_counts[name] = (h, b)

print("\nPer-split HR/BC counts:")
for name, (h, b) in per_split_counts.items():
    print(f"  {name:35s} HR={h:5d}  BC={b:5d}")

print(f"\nTarget N per class (upper bound) = {TARGET_N}")

# ────────────────────────────────────────────────────────────────
# 4. Build extended splits, reusing old ones and topping up
# ────────────────────────────────────────────────────────────────
NEW_SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
random.seed(SEED)  # deterministic augmentation

for name, pools in splits.items():
    hr_pool = pools["hr"]
    bc_pool = pools["bc"]
    hr_total = len(hr_pool)
    bc_total = len(bc_pool)

    # Effective N for this split
    N_eff = min(TARGET_N, hr_total, bc_total)
    if N_eff < 1:
        raise RuntimeError(f"Split {name} has no usable HR or BC examples.")

    # Load the original (old) split that already has 47/47
    old_path = OLD_SPLITS_ROOT / f"{name}.json"
    if not old_path.is_file():
        raise FileNotFoundError(f"Original split not found: {old_path}")
    old_recs = load_json_list(old_path)
    old_hr, old_bc = split_hr_bc_from_records(old_recs)

    if len(old_hr) != len(old_bc):
        raise RuntimeError(
            f"Original split {name} is unbalanced: HR={len(old_hr)}, BC={len(old_bc)}"
        )

    N_old = len(old_hr)
    if N_old > N_eff:
        raise RuntimeError(
            f"Original split {name} has {N_old} per class but N_eff={N_eff}; "
            f"cannot downsize without dropping examples."
        )

    extra_hr_needed = N_eff - N_old
    extra_bc_needed = N_eff - N_old

    print(
        f"\nBuilding extended split {name}: "
        f"N_old={N_old}, N_eff={N_eff}, "
        f"extra_hr={extra_hr_needed}, extra_bc={extra_bc_needed}"
    )

    # Build sets of prompts already used in the original split to avoid duplicates
    old_hr_prompts = {r["prompt"] for r in old_hr}
    old_bc_prompts = {r["prompt"] for r in old_bc}

    # Candidate pools excluding existing prompts
    hr_candidates = [r for r in hr_pool if r["prompt"] not in old_hr_prompts]
    bc_candidates = [r for r in bc_pool if r["prompt"] not in old_bc_prompts]

    if extra_hr_needed > 0 and len(hr_candidates) < extra_hr_needed:
        raise RuntimeError(
            f"Not enough extra HR examples for split {name}: "
            f"needed {extra_hr_needed}, have {len(hr_candidates)}"
        )
    if extra_bc_needed > 0 and len(bc_candidates) < extra_bc_needed:
        raise RuntimeError(
            f"Not enough extra BC examples for split {name}: "
            f"needed {extra_bc_needed}, have {len(bc_candidates)}"
        )

    # Sample additional examples
    extra_hr = random.sample(hr_candidates, extra_hr_needed) if extra_hr_needed > 0 else []
    extra_bc = random.sample(bc_candidates, extra_bc_needed) if extra_bc_needed > 0 else []

    final_hr = old_hr + extra_hr
    final_bc = old_bc + extra_bc

    assert len(final_hr) == N_eff
    assert len(final_bc) == N_eff

    combined = final_hr + final_bc  # HR first, then BC

    out_fname = f"{name}_HR{N_eff}_BC{N_eff}.json"
    out_path = NEW_SPLITS_ROOT / out_fname
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(
        f"Saved extended split {name:35s} → {out_path}  "
        f"(HR={N_eff}, BC={N_eff}, total={2 * N_eff})"
    )

print("\n✓ Finished building extended splits under:")
print(f"  {NEW_SPLITS_ROOT.resolve()}")
