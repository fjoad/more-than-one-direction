#!/usr/bin/env python
"""
Convert the 6 authored transform tables (markdown) into H.json pool files,
each paired with the SHARED benign B.json (identical across all non-XSTest splits).
No model calls. Produces pools in the same schema as per_split_simple.

Output: exp1_v10/transform_experiment/pools/<POOL>/{H.json,B.json,manifest.json}
where <POOL> in:
  safety2incomplete, incomplete2safety,
  safety2humanizing, humanizing2safety,
  safety2unsupported, unsupported2safety
"""
import json, re, os
from pathlib import Path
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # <repo>/code/<area>/script.py -> <repo>

ROOT = REPO
PSS  = REPO / "data/training_pools/gemma_per_split"
OUT  = REPO / "data/transform_experiment"
MD1  = REPO / "data/transform_experiment/transform_tables_pair1.md"    # pair 1 (Tables A, B)
MD2  = REPO / "data/transform_experiment/transform_tables_pairs2-3.md" # pairs 2,3 (Tables C-F)

# shared benign pool (verified identical hash across splits)
B_shared = json.load(open(PSS / "WildGuard_all" / "B.json"))

def parse_table(md_text, header_marker):
    """Return the list of transformed prompts (the 2nd content column) from the
    markdown table that appears after `header_marker`."""
    idx = md_text.index(header_marker)
    chunk = md_text[idx:]
    rows = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if rows and line and not line.startswith("|"):
                # table ended
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # skip header + separator rows
        if len(cells) < 3: continue
        if cells[0] in ("#", "") and set(cells[1]) <= set("-: "): continue
        if cells[0] == "#": continue
        if re.fullmatch(r"[-:\s]+", cells[0] or "-"): continue
        if not re.fullmatch(r"\d+", cells[0]): continue   # must start with a row number
        transformed = cells[-1].strip()
        rows.append(transformed)
    return rows

md1 = open(MD1).read()
md2 = open(MD2).read()

TABLES = {
    "safety2incomplete":  (md1, "## Table A"),
    "incomplete2safety":  (md1, "## Table B"),
    "safety2humanizing":  (md2, "## Table C"),
    "humanizing2safety":  (md2, "## Table D"),
    "safety2unsupported": (md2, "## Table E"),
    "unsupported2safety": (md2, "## Table F"),
}

for pool, (md, marker) in TABLES.items():
    prompts = parse_table(md, marker)
    assert len(prompts) == 32, f"{pool}: expected 32 rows, got {len(prompts)}"
    H = [{"prompt": p, "prompt_harm_label": "harmful", "source": {"transform": pool}} for p in prompts]
    d = OUT / pool
    d.mkdir(parents=True, exist_ok=True)
    json.dump(H, open(d / "H.json", "w"), ensure_ascii=False, indent=1)
    json.dump(B_shared, open(d / "B.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"pool": pool, "n_h": len(H), "n_b": len(B_shared),
               "b_source": "per_split_simple/WildGuard_all/B.json (shared)",
               "h_source": f"authored transform table -> {marker}"},
              open(d / "manifest.json", "w"), indent=2)
    print(f"{pool}: H={len(H)} B={len(B_shared)}  first='{prompts[0][:60]}'")

print("done ->", OUT)
