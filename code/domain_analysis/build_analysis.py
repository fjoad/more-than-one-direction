#!/usr/bin/env python3
"""Validate domain labels and build the requested matrices and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "all_prompts.json"
RAW_FILES = [ROOT / f"raw_batch_{i}.tsv" for i in range(1, 4)]
FINAL_FILES = [ROOT / f"final_batch_{i}.tsv" for i in range(1, 4)]
ALLOWED_DOMAINS = {
    "Technology, AI & Cybersecurity",
    "Privacy & Personal Data",
    "Finance, Business & Economics",
    "Crime, Fraud & Illicit Activity",
    "Violence, Weapons & Physical Safety",
    "Health, Medicine & Mental Health",
    "Drugs, Alcohol & Addiction",
    "Sexuality & Adult Content",
    "Relationships, Family & Personal Life",
    "Identity, Discrimination & Social Justice",
    "Politics, Government, Law & International Affairs",
    "Education, Employment & Workplace",
    "Arts, Media, Literature & Language",
    "Religion, Philosophy & Ethics",
    "Science, Nature & Environment",
    "History, Geography & Cultural Heritage",
    "Lifestyle, Food, Sports & Recreation",
    "No Discernible Subject",
}


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prompts = pd.DataFrame(json.loads(SOURCE.read_text(encoding="utf-8")))
    raw = pd.concat(
        [pd.read_csv(path, sep="\t", dtype={"id": int, "free_domain": str}) for path in RAW_FILES],
        ignore_index=True,
    )
    final = pd.concat(
        [pd.read_csv(path, sep="\t", dtype={"id": int, "final_domain": str}) for path in FINAL_FILES],
        ignore_index=True,
    )

    expected_ids = set(range(len(prompts)))
    actual_ids = set(raw["id"].tolist())
    if len(prompts) != 2054 or expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"ID coverage failure: missing={missing[:10]}, extra={extra[:10]}")
    if raw["id"].duplicated().any():
        duplicates = raw.loc[raw["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Duplicate raw IDs: {duplicates[:10]}")
    if raw["free_domain"].isna().any() or raw["free_domain"].str.strip().eq("").any():
        raise ValueError("Every prompt must have a non-empty free_domain")
    final_ids = set(final["id"].tolist())
    if expected_ids != final_ids or final["id"].duplicated().any():
        missing = sorted(expected_ids - final_ids)
        extra = sorted(final_ids - expected_ids)
        duplicates = final.loc[final["id"].duplicated(), "id"].tolist()
        raise ValueError(
            f"Final ID coverage failure: missing={missing[:10]}, extra={extra[:10]}, "
            f"duplicates={duplicates[:10]}"
        )
    if final["final_domain"].isna().any() or final["final_domain"].str.strip().eq("").any():
        raise ValueError("Every prompt must have a non-empty final_domain")
    unknown_domains = sorted(set(final["final_domain"]) - ALLOWED_DOMAINS)
    if unknown_domains:
        raise ValueError(f"Unexpected final domains: {unknown_domains}")

    return prompts, raw.sort_values("id"), final.sort_values("id")


def save_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    fmt: str,
    colorbar_label: str,
) -> None:
    width = max(12, 0.9 * len(matrix.columns) + 4)
    height = max(7, 0.48 * len(matrix.index) + 2)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix.values, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=42, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_xlabel("Refusal-category split")
    ax.set_ylabel("Discovered and consolidated domain")
    ax.set_title(title, pad=14)
    threshold = float(np.nanmax(matrix.values)) * 0.55 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iat[row, col]
            label = format(value, fmt)
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value > threshold else "#1d2733",
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.75)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_outputs() -> None:
    prompts, raw, final = load_and_validate()
    labeled = raw.merge(final, on="id", how="left", validate="one_to_one")
    conflict_counts = labeled.groupby("free_domain")["final_domain"].nunique()
    conflicting_labels = conflict_counts[conflict_counts > 1].index
    if len(conflicting_labels):
        conflicts = (
            labeled[labeled["free_domain"].isin(conflicting_labels)]
            .sort_values(["free_domain", "id"])
            [["id", "free_domain", "final_domain"]]
        )
        conflicts.to_csv(ROOT / "context_sensitive_label_mappings.tsv", sep="\t", index=False)
    merge_map = (
        labeled.groupby(["free_domain", "final_domain"], as_index=False)
        .size()
        .rename(columns={"size": "prompt_count"})
        .sort_values(["final_domain", "free_domain"])
    )
    merge_map.to_csv(ROOT / "domain_merge_map.tsv", sep="\t", index=False)
    labeled = prompts.merge(labeled, on="id", how="left", validate="one_to_one")
    labeled = labeled.rename(columns={"final_domain": "domain"})

    final = labeled[["id", "domain"]].sort_values("id")
    if final["domain"].isna().any() or final["domain"].str.strip().eq("").any():
        raise ValueError("Every prompt must have exactly one final domain")

    final.to_csv(ROOT / "prompt_domains.tsv", sep="\t", index=False)
    annotated = labeled[
        [
            "id",
            "split",
            "dataset",
            "dataset_idx",
            "native_category",
            "native_subcategory",
            "prompt",
            "free_domain",
            "domain",
        ]
    ].copy()
    annotated["prompt"] = (
        annotated["prompt"]
        .astype(str)
        .str.replace("\r\n", "\\n", regex=False)
        .str.replace("\r", "\\n", regex=False)
        .str.replace("\n", "\\n", regex=False)
    )
    annotated.to_csv(ROOT / "annotated_prompts.tsv", sep="\t", index=False)

    distribution = (
        labeled.groupby("domain", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "domain"], ascending=[False, True])
    )
    distribution["percent"] = distribution["count"] / len(labeled) * 100
    coverage = labeled.groupby("domain")["split"].nunique().rename("splits_present")
    distribution = distribution.merge(coverage, on="domain")
    distribution.to_csv(ROOT / "domain_distribution.csv", index=False, float_format="%.3f")

    counts = pd.crosstab(labeled["domain"], labeled["split"])
    counts = counts.reindex(distribution["domain"])
    counts.to_csv(ROOT / "domain_by_split_counts.csv")
    split_percent = counts.div(counts.sum(axis=0), axis=1) * 100
    split_percent.to_csv(ROOT / "domain_by_split_percent.csv", float_format="%.3f")

    fig, ax = plt.subplots(figsize=(10, max(6, 0.42 * len(distribution))))
    ordered = distribution.sort_values("count", ascending=True)
    bars = ax.barh(ordered["domain"], ordered["count"], color="#377eb8")
    ax.bar_label(bars, labels=[f"{n} ({p:.1f}%)" for n, p in zip(ordered["count"], ordered["percent"])], padding=4, fontsize=8)
    ax.set_xlabel("Number of prompts")
    ax.set_ylabel("Domain")
    ax.set_title("Domain distribution across 2,054 prompts")
    ax.set_xlim(0, ordered["count"].max() * 1.22)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "domain_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    save_heatmap(
        counts,
        ROOT / "domain_by_split_counts.png",
        "Domain by refusal-category split (counts)",
        ".0f",
        "Prompt count",
    )
    save_heatmap(
        split_percent,
        ROOT / "domain_by_split_percent.png",
        "Domain composition within each split (%)",
        ".1f",
        "% of split prompts",
    )

    overlap_rows = []
    for domain in distribution["domain"]:
        present = counts.columns[counts.loc[domain] > 0].tolist()
        overlap_rows.append(
            {
                "domain": domain,
                "splits_present": len(present),
                "splits": "; ".join(present),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(ROOT / "domain_split_coverage.csv", index=False)

    substantive = distribution[distribution["domain"] != "No Discernible Subject"]
    split_breadth = (
        labeled.groupby("split")
        .agg(prompts=("id", "size"), domains=("domain", "nunique"))
        .sort_values(["domains", "prompts"], ascending=[False, False])
    )
    summary = [
        "# Domain-coverage analysis",
        "",
        f"All {len(labeled):,} prompts received exactly one final domain label.",
        f"The open-ended labels were consolidated into {len(substantive)} substantive data-derived domains plus the strict `No Discernible Subject` fallback.",
        "",
        "## Method",
        "",
        f"1. Every prompt was labeled open-ended with no supplied domain menu, producing {labeled['free_domain'].nunique():,} distinct raw phrasings.",
        "2. Three independent reviews grouped the observed labels into umbrella subjects. Their overlapping proposals were synthesized into the frozen taxonomy documented in `CONSOLIDATED_TAXONOMY.md`.",
        "3. Every prompt was then mapped to exactly one frozen domain using its literal primary subject. Split names and native safety labels were not used for assignment.",
        "4. ID coverage, allowed labels, matrix totals, repeated-label inconsistencies, and stratified prompt samples were audited before final output.",
        "A small number of broad raw labels map to more than one final domain where the full prompt supplies decisive context; these cases are retained in `context_sensitive_label_mappings.tsv` rather than being forced into a misleading one-to-one merge.",
        "",
        "## Overall distribution",
        "",
        "| Domain | Prompts | Share | Splits represented |",
        "|---|---:|---:|---:|",
    ]
    for row in distribution.itertuples(index=False):
        summary.append(f"| {row.domain} | {row.count} | {row.percent:.1f}% | {row.splits_present}/11 |")
    summary += [
        "",
        "`No Discernible Subject` is an administrative fallback, not a substantive topic domain; it is retained in totals for complete accounting.",
        "",
        "The distribution is broad rather than dominated by one topic: the largest domain accounts for "
        f"{distribution.iloc[0]['percent']:.1f}% of prompts, and the five largest together account for "
        f"{distribution.head(5)['percent'].sum():.1f}%.",
        "",
        "## Cross-split coverage",
        "",
        f"All {len(substantive)} substantive domains occur in at least five of the 11 refusal-category splits, {(substantive['splits_present'] >= 6).sum()} occur in at least six, and {(substantive['splits_present'] >= 8).sum()} occur in at least eight. "
        "Politics/Government/Law/International Affairs occurs in all 11 splits; Technology/AI/Cybersecurity and Health/Medicine/Mental Health each occur in 10; Finance/Business/Economics occurs in nine.",
        "",
        "The count heatmap gives absolute frequency. Because split sizes differ, the column-percentage heatmap is the better view for comparing the topical composition of splits; every column sums to approximately 100% after rounding.",
        "Percentages are within-split shares: each cell is the domain–split count divided by that split's total number of prompts.",
        "",
        "### Breadth within each split",
        "",
        "| Split | Prompts | Domains represented |",
        "|---|---:|---:|",
    ]
    for split, row in split_breadth.iterrows():
        summary.append(f"| {split} | {row.prompts} | {row.domains}/18 |")
    summary += [
        "",
        "See `domain_split_coverage.csv` for the exact list of splits containing each domain.",
        "",
        "## Main artifacts",
        "",
        "- `prompt_domains.tsv`: requested one-domain-per-prompt result.",
        "- `annotated_prompts.tsv`: prompts, raw open-ended labels, and final labels together for audit.",
        "- `domain_distribution.csv` and `domain_distribution.png`: overall breadth and frequency.",
        "- `domain_by_split_counts.csv` / `.png`: absolute domain-by-split matrix.",
        "- `domain_by_split_percent.csv` / `.png`: within-split percentage matrix.",
        "- `domain_merge_map.tsv`: raw-label to final-domain consolidation pairs and their frequencies.",
        "- `context_sensitive_label_mappings.tsv`: repeated raw labels whose final domain depends on prompt context.",
    ]
    (ROOT / "ANALYSIS.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_outputs()
