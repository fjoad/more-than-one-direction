#!/usr/bin/env python
"""Combine the 2054 harmful prompts across the 11 evaluation splits into one list,
each row tagging where it came from (our split name + original dataset + original id +
whatever native category/subcategory metadata that dataset carries)."""
import json
from pathlib import Path
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # <repo>/code/<area>/script.py -> <repo>

SRC = REPO / "data/sources"
OUT = REPO / "data/domain_analysis"
OUT.mkdir(parents=True, exist_ok=True)

# our split label -> file
SPLITS = {
    "SafetyCore-WGM":     "WildGuard_all",
    "OverRefusal-XST":    "XSTest_all",
    "Advice-SB":          "SorryBench_Advice",
    "CrimeAssistance-SB": "SorryBench_CrimesTorts",
    "HateSpeech-SB":      "SorryBench_HateSpeech",
    "Inappropriate-SB":   "SorryBench_Inappropriate",
    "Humanizing-CCN":     "CocoNot_Humanizing",
    "Incomplete-CCN":     "CocoNot_Incomplete",
    "Indeterminate-CCN":  "CocoNot_Indeterminate",
    "Safety-CCN":         "CocoNot_Safety",
    "Unsupported-CCN":    "CocoNot_Unsupported",
}

rows = []
gid = 0
for split_label, fname in SPLITS.items():
    data = json.load(open(SRC / f"{fname}.json"))
    for r in data:
        src = r.get("source", {}) or {}
        rows.append({
            "id": gid,                                   # global running id for this combined list
            "split": split_label,                        # our evaluation-split label
            "dataset": src.get("dataset", fname),        # original dataset name
            "dataset_idx": src.get("idx"),               # original id within that dataset
            "native_category": src.get("category"),      # dataset's own category (if any)
            "native_subcategory": src.get("subcategory") or src.get("type"),  # dataset's own subcat/type (if any)
            "prompt": r["prompt"],
        })
        gid += 1

json.dump(rows, open(OUT / "all_prompts.json", "w"), ensure_ascii=False, indent=1)

# also a tab-separated flat view (id<TAB>split<TAB>dataset<TAB>dataset_idx<TAB>prompt) for quick reading
with open(OUT / "all_prompts.tsv", "w") as f:
    f.write("id\tsplit\tdataset\tdataset_idx\tnative_category\tnative_subcategory\tprompt\n")
    for r in rows:
        p = " ".join(str(r["prompt"]).split())  # collapse newlines/whitespace for the flat view
        f.write(f"{r['id']}\t{r['split']}\t{r['dataset']}\t{r['dataset_idx']}\t{r['native_category']}\t{r['native_subcategory']}\t{p}\n")

print(f"wrote {len(rows)} rows -> all_prompts.json + all_prompts.tsv")
from collections import Counter
for k in ("split", "dataset"):
    print(f"  by {k}: {dict(Counter(r[k] for r in rows))}")
