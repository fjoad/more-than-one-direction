"""Compute per-recipe All/Green/Red/Avg de-refusal % for Qwen-7B on own pool.

Reads: generations/discover_qwen7/judged/ablate_unified/<recipe>.json
Each record has fields: side (H/B), cluster (green/red), is_refusal_wg (0/1).
A baseline-refused harmful prompt is "de-refused" if is_refusal_wg == 0 after ablation.
The unified test pool only contains baseline-refused harmful prompts on the H side.
"""
import json
from pathlib import Path

ROOT = Path("/export/home/fjoad/llm_safety_project/exp1_v10/generations/discover_qwen7/judged/ablate_unified")

PER_SPLIT = [
    "WildGuard_all", "XSTest_all",
    "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
    "CocoNot_Safety", "CocoNot_Unsupported",
    "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
    "SorryBench_Inappropriate", "SorryBench_Advice",
]
SEEDS = [1, 2, 3, 4, 5]


def metrics(recipe_file):
    p = ROOT / recipe_file
    if not p.exists(): return None
    rows = json.load(open(p))
    H = [r for r in rows if r.get("side") == "H"]
    if not H: return None
    green = [r for r in H if r.get("cluster") == "green"]
    red = [r for r in H if r.get("cluster") == "red"]
    de = lambda L: 100.0 * sum(1 for r in L if r.get("is_refusal_wg") == 0) / max(len(L), 1)
    a, g, r = de(H), de(green), de(red)
    return {"All": a, "Green": g, "Red": r, "Avg": (g + r) / 2, "n_H": len(H), "n_G": len(green), "n_R": len(red)}


def fmt_row(name, m):
    if m is None: return f"{name:<28s} -    -    -    -"
    return f"{name:<28s} {m['All']:>5.1f} {m['Green']:>5.1f} {m['Red']:>5.1f} {m['Avg']:>5.1f}"


print(f"{'Recipe':<28s} {'All':>5s} {'Green':>5s} {'Red':>5s} {'Avg':>5s}")
print("-" * 60)
print("# Per-split (11)")
for s in PER_SPLIT:
    m = metrics(f"{s}.json")
    print(fmt_row(s, m))

print("# Generalists")
m_t64 = metrics("train64.json")
print(fmt_row("train64", m_t64))

for label, prefix in [("RR-33", "rr33_seed"), ("RR-55", "rr55_seed"), ("Crisp", "crisp_persplit_seed")]:
    print(f"# {label}")
    per_seed = [(k, metrics(f"{prefix}{k}.json")) for k in SEEDS]
    valid = [(k, m) for k, m in per_seed if m is not None]
    if not valid:
        continue
    for k, m in valid:
        print(fmt_row(f"{label} (seed={k})", m))
    mean_m = {key: sum(m[key] for _, m in valid) / len(valid) for key in ("All", "Green", "Red", "Avg")}
    best_idx = max(range(len(valid)), key=lambda i: valid[i][1]["Avg"])
    print(fmt_row(f"{label} (mean)", mean_m))
    print(fmt_row(f"{label} (best, seed={valid[best_idx][0]})", valid[best_idx][1]))

print()
# Report cluster sizes
first_recipe = metrics("train64.json")
if first_recipe:
    print(f"# Pool sizes: total H={first_recipe['n_H']} (Green={first_recipe['n_G']}, Red={first_recipe['n_R']})")
