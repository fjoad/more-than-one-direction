#!/usr/bin/env python3
"""Full Qwen ablation results across all H pool × BC pool combinations,
now including round-robin (Experiment 5)."""
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
SEEDS = [1, 2, 3, 4, 5]


def cell(intervention: str, split: str):
    p = ROOT / intervention / f"{split}.json"
    if not p.exists(): return None
    rows = json.load(open(p))
    H = [r for r in rows if r.get("side") == "H"]
    B = [r for r in rows if r.get("side") == "B"]
    return (len(H),
            sum(1 for r in H if r.get("is_refusal_wg") == 1),
            len(B),
            sum(1 for r in B if r.get("is_refusal_wg") == 1))


def pct_de(intervention, split, baseref):
    c = cell(intervention, split)
    if c is None or baseref == 0: return None
    return (baseref - c[1]) / baseref


def orr(intervention, split):
    c = cell(intervention, split)
    if c is None: return None
    return c[3] / c[2] if c[2] > 0 else None


def fmt(x):
    if x is None: return "  -"
    return f"{x*100:>3.0f}%"


baseline = {s: cell("baseline", s) for s in ALL_SPLITS}


def per_split_ablate(s, bc):
    interv = "ablate" if bc == "our" else f"ablate_persplit_{s}_t64BC"
    return pct_de(interv, s, baseline[s][1])


def train64_ablate(s, bc):
    interv = "ablate_t64H_ourBC" if bc == "our" else "ablate_train64"
    return pct_de(interv, s, baseline[s][1])


def wgrand_mean(s, bc, fn):
    suffix = "" if bc == "our" else "_t64BC"
    vals = [fn(f"ablate_wgrandom_seed{k}{suffix}", s, baseline[s][1]) for k in SEEDS]
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None


def rr_mean_ablate(s, bc):
    suffix = "ourBC" if bc == "our" else "t64BC"
    vals = [pct_de(f"ablate_roundrobin_seed{k}_{suffix}", s, baseline[s][1]) for k in SEEDS]
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None


def rr_spread_ablate(s, bc):
    """Return (min, max) across 5 seeds."""
    suffix = "ourBC" if bc == "our" else "t64BC"
    vals = [pct_de(f"ablate_roundrobin_seed{k}_{suffix}", s, baseline[s][1]) for k in SEEDS]
    vals = [v for v in vals if v is not None]
    return (min(vals), max(vals)) if vals else (None, None)


print("=" * 145)
print("FULL FACTORIAL — Ablation effectiveness (% de-refused), H pool × BC pool, Qwen-1.8B-Chat")
print("=" * 145)
print()
print(f"{'Split':<23s} {'N_H':>4s}    "
      f"{'per-split':>17s}        {'train64 H':>17s}        {'wgrand mean':>17s}        {'roundrobin mean':>17s}")
print(f"{'':<23s} {'':>4s}    "
      f"{'ourBC':>8s} {'t64BC':>8s}        {'ourBC':>8s} {'t64BC':>8s}        "
      f"{'ourBC':>8s} {'t64BC':>8s}        {'ourBC':>8s} {'t64BC':>8s}")
print("-" * 145)
for s in ALL_SPLITS:
    print(f"{s:<23s} {baseline[s][0]:>4d}    "
          f"{fmt(per_split_ablate(s,'our')):>8s} {fmt(per_split_ablate(s,'t64')):>8s}        "
          f"{fmt(train64_ablate(s,'our')):>8s} {fmt(train64_ablate(s,'t64')):>8s}        "
          f"{fmt(wgrand_mean(s,'our',pct_de)):>8s} {fmt(wgrand_mean(s,'t64',pct_de)):>8s}        "
          f"{fmt(rr_mean_ablate(s,'our')):>8s} {fmt(rr_mean_ablate(s,'t64')):>8s}")

print()
print()
print("=" * 145)
print("ROUND-ROBIN SEED SPREAD — % de-refused, per seed, across BC variants")
print("=" * 145)
print()
print(f"{'Split':<23s}    "
      f"{'roundrobin ourBC (per seed)':>34s}    {'min':>4s} {'max':>4s}    "
      f"{'roundrobin t64BC (per seed)':>34s}    {'min':>4s} {'max':>4s}")
print("-" * 145)
for s in ALL_SPLITS:
    o_vals = [pct_de(f"ablate_roundrobin_seed{k}_ourBC", s, baseline[s][1]) for k in SEEDS]
    t_vals = [pct_de(f"ablate_roundrobin_seed{k}_t64BC", s, baseline[s][1]) for k in SEEDS]
    o_str = " ".join(fmt(v).rjust(5) for v in o_vals)
    t_str = " ".join(fmt(v).rjust(5) for v in t_vals)
    o_clean = [v for v in o_vals if v is not None]
    t_clean = [v for v in t_vals if v is not None]
    print(f"{s:<23s}    "
          f"{o_str}    {fmt(min(o_clean)) if o_clean else '-':>4s} {fmt(max(o_clean)) if o_clean else '-':>4s}    "
          f"{t_str}    {fmt(min(t_clean)) if t_clean else '-':>4s} {fmt(max(t_clean)) if t_clean else '-':>4s}")

print()
print()
print("=" * 145)
print("ROW SUMMARY — Average across 11 splits, per H × BC combo")
print("=" * 145)
print()
print(f"{'':<25s} {'ourBC':>10s}  {'t64BC':>10s}  {'BC swap Δ':>10s}")
print("-" * 60)
def avg(fn, bc):
    vals = [fn(s, bc) for s in ALL_SPLITS]
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None

for name, fn in [("per-split", per_split_ablate),
                 ("train64 H", train64_ablate),
                 ("wgrand mean", lambda s, bc: wgrand_mean(s, bc, pct_de)),
                 ("roundrobin mean", rr_mean_ablate)]:
    a_our = avg(fn, "our")
    a_t64 = avg(fn, "t64")
    delta = (a_t64 - a_our) * 100 if (a_our is not None and a_t64 is not None) else None
    delta_str = f"{delta:+5.0f}pp" if delta is not None else "-"
    print(f"{name:<25s} {fmt(a_our):>10s}  {fmt(a_t64):>10s}  {delta_str:>10s}")
