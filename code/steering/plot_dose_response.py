#!/usr/bin/env python
"""
Rebuttal (QeqE W2), FINAL per team decision: Gemma-only dose-response, split into
two panels by the paper's two clusters. Same RR (solid) / ORR (dashed) vs alpha.
Cluster names match the paper (Fig 2 caption + Appendix H):
  - safety/content-policy   (7 splits)
  - capability/underspecification (4 splits)
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # <repo>/code/<area>/script.py -> <repo>

ROOT = str(REPO)
OUT  = str(REPO / "results/figures")
CSV  = os.path.join(ROOT, "exp1_v9/generations/neel_13dirs_sweep/wildguard_metrics_summary_neel13sweep.csv")

SPLIT_MAP = {
    "CocoNot_cat_Humanizing_requests":          "Humanizing-CCN",
    "CocoNot_cat_Incomplete_requests":          "Incomplete-CCN",
    "CocoNot_cat_Indeterminate_requests":       "Indeterminate-CCN",
    "CocoNot_cat_Requests_with_safety_concerns":"Safety-CCN",
    "CocoNot_cat_Unsupported_requests":         "Unsupported-CCN",
    "SorryBench_crimes_torts":                  "CrimeAssistance-SB",
    "SorryBench_hate_speech":                   "HateSpeech-SB",
    "SorryBench_inappropriate_topics":          "Inappropriate-SB",
    "SorryBench_unqualified_advice":            "Advice-SB",
    "WildGuard_all":                            "SafetyCore-WGM",
    "XSTest_all":                               "OverRefusal-XST",
}
# paper's two clusters
CONTENT_POLICY = ["SafetyCore-WGM","OverRefusal-XST","Safety-CCN","HateSpeech-SB",
                  "CrimeAssistance-SB","Inappropriate-SB","Advice-SB"]
CAPABILITY     = ["Humanizing-CCN","Incomplete-CCN","Indeterminate-CCN","Unsupported-CCN"]

def load():
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["dataset"].isin(SPLIT_MAP)].copy()
    df["split"] = df["dataset"].map(SPLIT_MAP)
    df["RR"]  = df["after_HR"] / df["n_used_harmful"]
    df["ORR"] = df["after_BR"] / df["n_used_unharmful"]
    df["alpha"] = df["alpha"].astype(float)
    df = df[["split","alpha","RR","ORR"]].drop_duplicates(["split","alpha"])
    base = pd.DataFrame({"split": sorted(df["split"].unique()), "alpha":0.0, "RR":0.5, "ORR":0.5})
    return pd.concat([base, df], ignore_index=True).sort_values(["split","alpha"])

# distinct color per split within a panel (categorical, colorblind-safe Okabe-Ito-ish)
PALETTE = ["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#000000"]

def panel(ax, df, splits, title):
    for i, sp in enumerate(splits):
        g = df[df["split"]==sp].sort_values("alpha")
        c = PALETTE[i % len(PALETTE)]
        ax.plot(g["alpha"], g["RR"],  color=c, lw=1.6)                 # RR solid
        ax.plot(g["alpha"], g["ORR"], color=c, lw=1.6, ls="--")        # ORR dashed
    ax.axhline(0.5, color="grey", lw=0.6, ls=":")
    ax.set_xlabel(r"steering strength $\alpha$")
    ax.set_ylabel("rate")
    ax.set_ylim(0.45, 1.02)
    ax.set_title(title, fontsize=11)
    handles = [Line2D([0],[0], color=PALETTE[i%len(PALETTE)], lw=2, label=sp) for i,sp in enumerate(splits)]
    handles += [Line2D([0],[0], color="grey", ls="-", label="RR (harmful refused)"),
                Line2D([0],[0], color="grey", ls="--", label="ORR (benign refused)")]
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)

def main():
    df = load()
    fig, axes = plt.subplots(1, 2, figsize=(13,5), sharey=True)
    panel(axes[0], df, CONTENT_POLICY, "Safety / content-policy directions")
    panel(axes[1], df, CAPABILITY,     "Capability / underspecification directions")
    fig.suptitle(r"Gemma-2-9B-IT: refusal rate (RR, solid) and over-refusal rate (ORR, dashed) vs. steering strength $\alpha$",
                 fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    out = os.path.join(OUT, "gemma_curves_twopanel.png")
    fig.savefig(out, dpi=160); fig.savefig(out.replace(".png",".pdf"))
    print("wrote", out)

if __name__ == "__main__":
    main()
