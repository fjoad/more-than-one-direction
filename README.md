# There Is More to Refusal in Large Language Models than a Single Direction

Code and data accompanying the paper *"There Is More to Refusal in Large Language Models than a Single Direction"* (Joad, Hawasly, Boughorbel, Durrani, Sencar — QCRI/HBKU), submitted to ACL Rolling Review 2026.

The paper studies whether refusal in instruction-tuned LLMs is mediated by a single activation-space direction or by multiple geometrically distinct directions, using difference-of-means refusal directions with steering and ablation, plus sparse autoencoders (SAEs). Models: **Gemma-2-9B-IT**, **Llama-3-8B-Instruct**, **Qwen-7B-Chat**. Evaluation: 11 refusal/non-compliance splits over WildGuardMix, SorryBench, CoCoNot, XSTest.

---

## Status

⚠️ **Initial release — work in progress.** This is the structural release: data, scripts, notebooks, and a reproduction walkthrough are in place. Outstanding portability work to be done in a follow-up pass:

- Hardcoded `/export/home/fjoad/llm_safety_project/...` paths in scripts (originate from the development cluster) need replacing with paths relative to the repo root.
- A small number of scripts touched closed APIs (GPT‑5 via OpenRouter) and have been removed from this release; the **outputs** (semantic labels) are included under `data/sae_latent_labels/` so the tables they support remain transparent (Table 14 in the paper). To regenerate labels from scratch, you'd need to apply your own LLM to the top-activating prompts using the same taxonomy.
- End-to-end "press button → table reproduces" testing pending. Until then, scripts may need light adaptation to run in your environment.

If you want to use this code for research, please open an issue — I'd rather know about pain points than not.

---

## Quick start

```bash
# clone
git clone https://github.com/fjoad/<repo-name>.git
cd <repo-name>

# Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Models, SAEs, and the judge are NOT in this repo — fetch from HuggingFace:
# - google/gemma-2-9b-it
# - meta-llama/Meta-Llama-3-8B-Instruct
# - Qwen/Qwen-7B-Chat
# - google/gemma-scope-9b-it-res (SAE)
# - andyrdt/saes-llama-3.1-8b-instruct (SAE)
# - allenai/wildguard (judge)
```

GPU recommended: a single 32 GB V100 or A100 is enough for any one model.

---

## Repo layout

```
paper_release/
├── README.md                 # you are here
├── requirements.txt
├── .gitignore
├── data/                     # ~20 MB total; everything needed except model weights
│   ├── sources/              # 12 source prompt JSONs (the 11 splits + shared BC pool)
│   ├── splits/               # per-split test pools used in the main paper
│   ├── training_pools/       # Curated-32 (`train_subset64_working.json`)
│   ├── controlled_test_set/  # 200-prompt 4-pool test set used for Table 2 (50 each HR/HC/BR/BC)
│   ├── directions/           # extracted difference-of-means directions (.pt) per model × per split
│   ├── refusal_type/         # Table 5: responses, GPT classifications, taxonomy prompt
│   ├── clustering/           # Figure 2 + Appendix H clustering visualisations (.png + intermediate .json)
│   └── sae_latent_labels/    # GPT-5 semantic labels for SAE refusal latents (Table 14 outputs)
├── code/
│   ├── data_prep/            # build splits, training pools, baselines (HR/HC/BR/BC)
│   ├── directions/           # direction extraction (HR-BC mean difference)
│   ├── steering/             # x ← x + α·r interventions, α-sweep producers
│   ├── ablation/             # x ← x - (x·r)·r interventions, including round-robin / Curated-32
│   ├── sae/                  # SAE common-core steering, K=10 selection
│   ├── judge/                # WildGuard classification, metrics aggregation
│   ├── refusal_type/         # results aggregators (joins responses with judgements)
│   ├── notebooks/            # Jupyter notebooks (Tables 1, 4, 5; latent overlap)
│   └── lib/                  # (reserved) shared utilities
└── supplementary/            # (reserved) rebuttal-cited but not printed in paper
```

---

## Reproducing the paper

### Section 2 — Methods

Conceptual; no scripts.

### Section 3 — Experimental Setup

- **Sources of prompts** (CoCoNot, SorryBench, WildGuardMix, XSTest): public, but the exact subsets used are in `data/sources/<split>.json`.
- **Building training pools and the controlled 200-prompt test set**: `code/data_prep/build_sets.py`.
- **N=128 stability check (Appendix B Table 7)**: `code/data_prep/build_sets_128.py` (re-runs steering with N=128-built directions).
- **Direction extraction** (per-split HR–BC mean difference + Curated-32 single direction): see `code/directions/` and the `build_baselines_*` family in `code/data_prep/`. Extracted directions are pre-saved in `data/directions/<model>/<recipe>/...d.pt` to save you from re-extracting.

### Section 4 — Findings

**§4.1 Geometry (Figure 2, Appendix G Table 20, Appendix H Figs 3–10)**
Cosine-similarity matrix, hierarchical clustering, dendrograms, MDS. The headline heatmap (Figure 2) and the clustering visualizations are in `data/clustering/`. The numeric matrix (Table 20) and Table 1's per-cell values are produced by `code/notebooks/playground.ipynb` (audit-confirmed source; see the matrix output cell).

**§4.2 Interventions: Steering and Ablation (Tables 2, 3)**

- *Table 2 (per-direction steering, Gemma α=100 / Llama α=2.5 / Qwen α=15)*:
  - Gemma + Llama steering: `code/ablation/neel_ablate_{gemma,llama}_working.py` run in steering mode.
  - Qwen-7B steering at α=15: see `code/steering/discover_extras_steer_qwen.py` and the related `discover_*_steer_qwen.py` variants — the canonical script for the paper's α=15 column needs confirming during the follow-up adaptation pass.
  - α-sweep producers (used to *choose* the saturating α): `code/steering/full_exp_saes_steer_sweep.py`, `…_llama.py`.

- *Table 3 (per-direction ablation summary + round-robin mixed-source direction)*:
  - Per-split direction ablation: `code/ablation/neel_ablate_{gemma,llama,qwen}.py` and `_working.py` (projection branch).
  - Round-robin / Curated-32 mixed direction ablation: `code/ablation/ablate_unified_recipes_<model>.py` (one per model).

- *WildGuard judging* (used by every steered/ablated generation): `code/judge/evaluate_wildguard_all_full2.py`.
- *Metrics aggregation* (Acc / RR / ORR): `code/judge/calculate_metrics_full.py`.

**§4.3 Refusal Directions in SAE Space (Table 3 SAE row equivalent; Appendix F Tables 15–19; Appendix E Table 14)**

- SAE common-core steering (the K=10 latent steering): `code/sae/full_exp_saes_common7.py`, `…_preonly.py`.
- Latent overlap across 11 splits (Appendix F Tables 15–19 with the corrected 11-split numbers 878 / 893 / 882): `code/notebooks/latent_intersection.ipynb`. Set `INTERSECTION_TOP_N` and the 11-split list at the top of the notebook.
- Latent semantic annotation (Appendix E Table 14): the original code that called GPT-5 via OpenRouter is **not in this release** (closed API + key handling). The **labels themselves** are under `data/sae_latent_labels/`; the per-dataset and combined annotations are sufficient to reconstruct Table 14. If you want to regenerate, apply your own LLM to each common-core latent's top-20 activating prompts using the two-round taxonomy described in the paper.

### Appendix A — Refusal Style + Refusal Type Breakdown

- *Table 4 (refusal style examples)*: hand-selected from steered-response outputs.
- *Table 5 (dominant refusal type + purity)* — the M2 deliverable for the resubmission:
  - Source responses (25 HR prompts × 11 directions × 3 models = 825): `data/refusal_type/{gemma,llama,qwen}_responses.json`.
  - Taxonomy prompt: `data/refusal_type/CLASSIFY_PROMPT.txt` (five categories: CAPABILITY / SAFETY / UNDERSPEC / GENERIC / NONREFUSAL).
  - Labels produced via ChatGPT against the taxonomy prompt: `data/refusal_type/gpt_classifications/*.txt`.
  - Aggregator (dominant type + purity per (model, direction)): **needs writing — see WIP** (~30-line script joining responses with labels and pivoting per direction).

### Appendix J — Unified-Pool Ablation, full per-direction detail (Tables 22–25)

`code/ablation/ablate_unified_recipes_{qwen,qwen7,gemma,llama}.py` is the canonical producer (one per model). The common test pool used is built by `code/data_prep/compute_common_test_pool.py`. Results aggregation: scripts under `code/refusal_type/results_*.py` (despite the directory name — these are the result-table assemblers for the unified-pool experiments).

---

## What's **not** in this release

- **Model weights, SAEs, the judge model** → fetch from HuggingFace (see Quick Start).
- **All generated model outputs** (thousands of JSONs across alphas, splits, models): too large to ship, regenerable from the scripts.
- **The labeling code that calls closed APIs** (GPT-5 via OpenRouter for Table 14 semantic annotation). Labels themselves are released; code is not.
- **Dead-end experiments** that did not survive into the published paper (forced-response B1/B2 direction-extraction; "magic single direction" searches; 13-split umbrella numbers; the cluster-parallelism `_b/_c/_d` sibling scripts).

---

## Citation

If you use this code or data, please cite the paper:

```bibtex
@misc{joad2026morerefusal,
  title = {There Is More to Refusal in Large Language Models than a Single Direction},
  author = {Joad, Faaiz and Hawasly, Majd and Boughorbel, Sabri and Durrani, Nadir and Sencar, Husrev Taha},
  year = {2026},
  note = {ACL Rolling Review submission, QCRI/HBKU},
}
```

(Replace with the venue-final citation once the paper is accepted.)

## License

To be decided before public release. Likely Apache-2.0 for code, CC-BY-4.0 for the bundled data subsets (each source dataset retains its own original license).

## Issues / contact

Open a GitHub issue if you hit a snag reproducing any specific table — happy to help and to fold fixes back into the release.
