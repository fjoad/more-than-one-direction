"""Cross-pool comparison: each (model, test-pool) cell's per-recipe All/Green/Red/Avg.

Reads judged JSONs under generations/discover_<model>/judged/<intervention>/<recipe>.json.
Each record has side, cluster, is_refusal_wg.
Reports % de-refused = fraction of side=H prompts with is_refusal_wg=0.
"""
import json
from pathlib import Path

ROOT = Path("/export/home/fjoad/llm_safety_project/exp1_v10/generations")

PER_SPLIT = [
    "WildGuard_all", "XSTest_all",
    "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
    "CocoNot_Safety", "CocoNot_Unsupported",
    "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
    "SorryBench_Inappropriate", "SorryBench_Advice",
]
SEEDS = [1, 2, 3, 4, 5]


def metrics(model, intervention, recipe_file):
    p = ROOT / f"discover_{model}" / "judged" / intervention / recipe_file
    if not p.exists(): return None
    rows = json.load(open(p))
    H = [r for r in rows if r.get("side") == "H"]
    if not H: return None
    green = [r for r in H if r.get("cluster") == "green"]
    red = [r for r in H if r.get("cluster") == "red"]
    de = lambda L: 100.0 * sum(1 for r in L if r.get("is_refusal_wg") == 0) / max(len(L), 1)
    return {"All": de(H), "Green": de(green), "Red": de(red),
            "Avg": (de(green) + de(red)) / 2,
            "n_H": len(H), "n_G": len(green), "n_R": len(red)}


def cell(m):
    if m is None: return "  -.-   -.-   -.-   -.-"
    return f"{m['All']:>5.1f} {m['Green']:>5.1f} {m['Red']:>5.1f} {m['Avg']:>5.1f}"


def print_model_block(model, intervs, labels, pool_sizes):
    """Print all recipes for one model across intervention dirs."""
    print()
    print("=" * (40 + 24 * len(intervs)))
    print(f"MODEL: {model}")
    pool_str = "  ".join([f"{lab} N={ps}" for lab, ps in zip(labels, pool_sizes)])
    print(f"Pool sizes: {pool_str}")
    print()

    # Header
    header_l1 = f"{'Recipe':<28}"
    header_l2 = f"{'':<28}"
    for lab in labels:
        header_l1 += f" | {lab:^22}"
        header_l2 += f" | {'All  Green  Red   Avg':^22}"
    print(header_l1)
    print(header_l2)
    print("-" * len(header_l1))

    def row(name, recipe_file, bold=False):
        cells = [metrics(model, iv, recipe_file) for iv in intervs]
        line = f"{name:<28}"
        for m in cells:
            line += f" | {cell(m)}"
        if bold:
            line = "** " + line
        print(line)

    print("# Per-split (11)")
    for s in PER_SPLIT:
        row(s, f"{s}.json")

    print("# Generalists")
    row("train64", "train64.json")
    # B1 only meaningful when present (gemma has it for own pool)
    if any((ROOT / f"discover_{model}/judged/{iv}/B1.json").exists() for iv in intervs):
        row("B1", "B1.json")

    print("# Seeded recipes")
    for label, prefix in [("RR-33", "rr33_seed"), ("RR-55", "rr55_seed"), ("Crisp", "crisp_persplit_seed")]:
        # Mean across seeds + best per intervention
        per_seed_per_iv = []  # per_seed_per_iv[iv_idx] = list of metrics for each seed
        for iv in intervs:
            per_seed = [metrics(model, iv, f"{prefix}{k}.json") for k in SEEDS]
            per_seed_per_iv.append([m for m in per_seed if m is not None])

        # Mean
        mean_line = f"{label + ' (mean)':<28}"
        best_line = f"{label + ' (best)':<28}"
        for vals in per_seed_per_iv:
            if not vals:
                mean_line += f" | {cell(None)}"
                best_line += f" | {cell(None)}"
                continue
            mean_m = {k: sum(m[k] for m in vals) / len(vals) for k in ("All", "Green", "Red", "Avg")}
            best = max(vals, key=lambda m: m["Avg"])
            mean_line += f" | {cell(mean_m)}"
            best_line += f" | {cell(best)}"
        print(mean_line)
        print(best_line)


# ---- Per-model: list (intervention, label, pool-N) tuples ----

def first_recipe_n(model, intervention):
    """Get H size of any existing recipe file to determine actual pool size."""
    m = metrics(model, intervention, "train64.json")
    return m["n_H"] if m else 0


# Qwen-1.8B: just own pool
print_model_block("qwen",
                  intervs=["ablate_unified"],
                  labels=["Qwen-1.8B / Qwen pool"],
                  pool_sizes=[first_recipe_n("qwen", "ablate_unified")])

# Qwen-7B: own + on-qwen
print_model_block("qwen7",
                  intervs=["ablate_unified", "ablate_unified_on_qwen"],
                  labels=["Qwen-7B / own", "Qwen-7B / Qwen pool"],
                  pool_sizes=[first_recipe_n("qwen7", "ablate_unified"),
                              first_recipe_n("qwen7", "ablate_unified_on_qwen")])

# Gemma: own + on-qwen
print_model_block("gemma",
                  intervs=["ablate_unified", "ablate_unified_on_qwen"],
                  labels=["Gemma / own", "Gemma / Qwen pool"],
                  pool_sizes=[first_recipe_n("gemma", "ablate_unified"),
                              first_recipe_n("gemma", "ablate_unified_on_qwen")])

# Llama: own + on-qwen + on-gemma
print_model_block("llama",
                  intervs=["ablate_unified", "ablate_unified_on_qwen", "ablate_unified_on_gemma"],
                  labels=["Llama / own", "Llama / Qwen pool", "Llama / Gemma pool"],
                  pool_sizes=[first_recipe_n("llama", "ablate_unified"),
                              first_recipe_n("llama", "ablate_unified_on_qwen"),
                              first_recipe_n("llama", "ablate_unified_on_gemma")])
