# Table 1 hierarchical-clustering analysis — meeting deliverable

Open `table1_cluster_findings.md` for the writeup. This README is a quick map.

## Files

- `table1_cluster_findings.md` — narrative (algorithm, MDS, linkages, colours)
- `table1_cluster_stats.txt`  — silhouette table, per-N memberships, per-item
                                colour assignment across all N values
- `table1_prompts_per_split.txt` — 3 sample prompts per split, annotated with
                                   the colour assignment at n=2,3,4,5
- `table1_reordered.tex` — drop-in LaTeX `tabular` for the paper

## Images (in `images/`)  — 21 total

Every plot uses the SAME per-item colour for a given split across every plot.
A split's colour can only change between N values, and only if that split
broke off into a smaller sub-cluster at the higher N.

**Constant (1)**:
- `heatmap_original.png` — Table 1 in paper's original row order; row/col
                           label colours follow the n=2 cluster colours

**Per N ∈ {2, 3, 4, 5} — 5 images per N (20 total)**:
- `dendrogram_average_n{N}.png`   tree, average linkage, cut height marked
- `dendrogram_complete_n{N}.png`  tree, complete linkage, same colours
- `mds_2d_n{N}.png`               2D map of all 11 splits
- `mds_3d_n{N}.png`               3D map of all 11 splits
- `heatmap_reordered_n{N}.png`    same as `heatmap_original` but rows/cols
                                  reordered by the dendrogram leaves, with
                                  black lines marking cluster boundaries for N

## Suggested viewing order at the meeting

1. `heatmap_original.png` — "this is Table 1 as a coloured heatmap"
2. `heatmap_reordered_n2.png` — "same numbers, reordered — two diagonal blocks"
3. `dendrogram_average_n2.png` — "how the splits merge step by step"
4. `mds_2d_n2.png` — "the 11 splits as points in 2D, clusters visible"
5. Flip through `mds_2d_n{2,3,4,5}.png` to watch one cluster split as N grows
   (the points don't move — only the colours change, and only when something
    actually breaks off)
6. `dendrogram_complete_n2.png` — "complete linkage agrees at n=2"
7. Compare `dendrogram_*_n3.png` to see where the two linkages disagree
