"""Validate v9-vs-v10 train64 hypothesis: Llama ablate on Gemma's 227-prompt pool.

If v10's low Llama train64 (52.2% on Llama-own-pool) was driven by pool conditioning,
this run on Gemma's 227-prompt pool should give numbers much closer to Gemma's
train64=75.8% than to Llama's 52.2%.
"""
import json
from pathlib import Path

ROOT_OWN = Path("/export/home/fjoad/llm_safety_project/exp1_v10/generations/discover_llama/judged/ablate_unified")
ROOT_GEMMA = Path("/export/home/fjoad/llm_safety_project/exp1_v10/generations/discover_llama/judged/ablate_unified_on_gemma")

PER_SPLIT = [
    "WildGuard_all", "XSTest_all",
    "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
    "CocoNot_Safety", "CocoNot_Unsupported",
    "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
    "SorryBench_Inappropriate", "SorryBench_Advice",
]
SEEDS = [1, 2, 3, 4, 5]


def metrics(root, recipe_file):
    p = root / recipe_file
    if not p.exists(): return None
    rows = json.load(open(p))
    H = [r for r in rows if r.get("side") == "H"]
    if not H: return None
    green = [r for r in H if r.get("cluster") == "green"]
    red = [r for r in H if r.get("cluster") == "red"]
    de = lambda L: 100.0 * sum(1 for r in L if r.get("is_refusal_wg") == 0) / max(len(L), 1)
    a, g, r = de(H), de(green), de(red)
    return {"All": a, "Green": g, "Red": r, "Avg": (g + r) / 2, "n_H": len(H), "n_G": len(green), "n_R": len(red)}


def fmt_row(name, mo, mg):
    if mo is None and mg is None: return f"{name:<28s} -"
    def cell(m):
        if m is None: return "  -    -    -    -"
        return f"{m['All']:>5.1f} {m['Green']:>5.1f} {m['Red']:>5.1f} {m['Avg']:>5.1f}"
    return f"{name:<28s} | {cell(mo)} | {cell(mg)}"


print(f"{'Recipe':<28s} | {'OWN POOL (203 H)':^22s} | {'GEMMA POOL (227 H)':^22s}")
print(f"{'':<28s} | {'All  Green  Red  Avg':^22s} | {'All  Green  Red  Avg':^22s}")
print("-" * 80)

for s in PER_SPLIT:
    mo = metrics(ROOT_OWN, f"{s}.json")
    mg = metrics(ROOT_GEMMA, f"{s}.json")
    print(fmt_row(s, mo, mg))

mo_t64 = metrics(ROOT_OWN, "train64.json")
mg_t64 = metrics(ROOT_GEMMA, "train64.json")
print(fmt_row("train64", mo_t64, mg_t64))

# Seeds — show mean and best
for label, prefix in [("RR-33", "rr33_seed"), ("RR-55", "rr55_seed"), ("Crisp", "crisp_persplit_seed")]:
    o_all = [metrics(ROOT_OWN, f"{prefix}{k}.json") for k in SEEDS]
    g_all = [metrics(ROOT_GEMMA, f"{prefix}{k}.json") for k in SEEDS]
    o_valid = [m for m in o_all if m is not None]
    g_valid = [m for m in g_all if m is not None]
    if o_valid and g_valid:
        o_mean = {k: sum(m[k] for m in o_valid) / len(o_valid) for k in ("All", "Green", "Red", "Avg")}
        g_mean = {k: sum(m[k] for m in g_valid) / len(g_valid) for k in ("All", "Green", "Red", "Avg")}
        o_best_idx = max(range(len(o_valid)), key=lambda i: o_valid[i]["Avg"])
        g_best_idx = max(range(len(g_valid)), key=lambda i: g_valid[i]["Avg"])
        print(fmt_row(f"{label} (mean)", o_mean, g_mean))
        print(fmt_row(f"{label} (best, own=seed{SEEDS[o_best_idx]}, gem=seed{SEEDS[g_best_idx]})",
                     o_valid[o_best_idx], g_valid[g_best_idx]))

print()
print("Hypothesis: if train64 on Gemma-pool >> train64 on own-pool (52.2),")
print("            the v9 vs v10 train64 gap is methodological (pool conditioning), not a regression.")
