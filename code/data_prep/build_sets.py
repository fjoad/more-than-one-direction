#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_13_splits.py

Build 13 balanced HR/BC test splits from:
- WildGuard (harmful vs unharmful)
- XSTest (label: unsafe vs safe)
- CocoNot (all harmful, 5 categories)
- SorryBench (all harmful, 4 category groups)

Each output file contains only:
    - "prompt"
    - "prompt_harm_label"  ("harmful" / "unharmful")
    - "was_refusal"        (0 / 1)

WildGuard BC is reused as BC for CocoNot and SorryBench.

Outputs are written to:
    data/refusal_13splits/<split_name>.json
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

# Adjust these paths as needed
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

DATA_OUT_ROOT = Path("data") / "refusal_13splits"

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


# ────────────────────────────────────────────────────────────────
# 1. Build pools from each dataset
# ────────────────────────────────────────────────────────────────
print("=== Loading datasets and building HR/BC pools ===")

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

# 1.4 SorryBench (all harmful; HR grouped by category ranges; BC from WildGuard BC)
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
# 2. Define 13 splits (HR pool + BC pool)
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
# 3. Compute global N and sample balanced sets
# ────────────────────────────────────────────────────────────────
per_split_counts: Dict[str, Tuple[int, int]] = {}
for name, pools in splits.items():
    h = len(pools["hr"])
    b = len(pools["bc"])
    per_split_counts[name] = (h, b)

print("\nPer-split HR/BC counts:")
for name, (h, b) in per_split_counts.items():
    print(f"  {name:35s} HR={h:5d}  BC={b:5d}")

N = min(min(h, b) for (h, b) in per_split_counts.values())
if N < 1:
    raise RuntimeError("At least one split has no HR or BC examples.")

print(f"\nGlobal N (per split HR/BC size) = {N}")

# ────────────────────────────────────────────────────────────────
# 4. Sample and save the 13 splits
# ────────────────────────────────────────────────────────────────
DATA_OUT_ROOT.mkdir(parents=True, exist_ok=True)
random.seed(SEED)  # deterministic sampling

for name, pools in splits.items():
    hr_pool = pools["hr"]
    bc_pool = pools["bc"]

    if len(hr_pool) < N or len(bc_pool) < N:
        raise RuntimeError(f"Split {name} has insufficient examples for N={N}.")

    hr_sel = random.sample(hr_pool, N)
    bc_sel = random.sample(bc_pool, N)

    combined = hr_sel + bc_sel  # HR first, then BC (order within each is random)
    out_path = DATA_OUT_ROOT / f"{name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Saved split {name:35s} → {out_path}  (HR={N}, BC={N})")

print("\n✓ Finished building and saving 13 balanced splits under:")
print(f"  {DATA_OUT_ROOT.resolve()}")
