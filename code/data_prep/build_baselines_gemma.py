#!/usr/bin/env python3
"""
build_baselines_gemma.py — Populate data/baselines/gemma/{responses,judged}/<split>.json by
extracting from existing exp1_v9 evaluated/...wildguard.json files.

Gemma + WildGuard already ran in v9; this is pure file IO + a string-match pass added inline.
No GPU. ~1-2 seconds.

For each of the 11 source splits:
  1. Look up each source prompt in the v9 evaluated file for the matching dataset.
  2. Extract Gemma's response and the WildGuard judge label.
  3. Compute the string-match label inline (using exp1_v8/evaluate_stringmatch_all.py).
  4. Write data/baselines/gemma/responses/<split>.json (without judge labels)
  5. Write data/baselines/gemma/judged/<split>.json   (with both is_refusal_wg and is_refusal_sm)

Also handles the shared _BC_source the same way.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

EXP_V10 = Path(__file__).resolve().parent.parent
EXP_V9 = EXP_V10.parent / "exp1_v9"
SOURCES_DIR = EXP_V10 / "data" / "sources"
RESP_DIR = EXP_V10 / "data" / "baselines" / "gemma" / "responses"
JUDGED_DIR = EXP_V10 / "data" / "baselines" / "gemma" / "judged"

# Bring in the canonical string-match judge from v8.
sys.path.insert(0, str(EXP_V10.parent / "exp1_v8"))
from evaluate_stringmatch_all import is_refusal as is_refusal_sm  # noqa: E402

# Map each source split to the v9 evaluated file that has Gemma responses + WG labels for it.
SOURCE_TO_V9 = {
    "WildGuard_all": EXP_V9 / "generations/original_wildguard/evaluated/wildguardtest_nonadv_wildguardtest_nonadversarial_generations_wildguard.json",
    "_BC_source":     EXP_V9 / "generations/original_wildguard/evaluated/wildguardtest_nonadv_wildguardtest_nonadversarial_generations_wildguard.json",
    "XSTest_all":     EXP_V9 / "generations/xstest/evaluated/test_generations_wildguard.json",
    "CocoNot_Humanizing":    EXP_V9 / "generations/coconot/original/evaluated/test_generations_wildguard.json",
    "CocoNot_Incomplete":    EXP_V9 / "generations/coconot/original/evaluated/test_generations_wildguard.json",
    "CocoNot_Indeterminate": EXP_V9 / "generations/coconot/original/evaluated/test_generations_wildguard.json",
    "CocoNot_Safety":        EXP_V9 / "generations/coconot/original/evaluated/test_generations_wildguard.json",
    "CocoNot_Unsupported":   EXP_V9 / "generations/coconot/original/evaluated/test_generations_wildguard.json",
    "SorryBench_HateSpeech":    EXP_V9 / "generations/sorrybench/evaluated/sorrybench_base440_generations_wildguard.json",
    "SorryBench_CrimesTorts":   EXP_V9 / "generations/sorrybench/evaluated/sorrybench_base440_generations_wildguard.json",
    "SorryBench_Inappropriate": EXP_V9 / "generations/sorrybench/evaluated/sorrybench_base440_generations_wildguard.json",
    "SorryBench_Advice":        EXP_V9 / "generations/sorrybench/evaluated/sorrybench_base440_generations_wildguard.json",
}

# v9 schemas vary across datasets in which fields hold the prompt and response.
PROMPT_FIELD = {
    "wildguardmix": "prompt",
    "xstest":       "prompt",
    "coconot":      "prompt",
    "sorrybench":   "turns",   # SorryBench stores the prompt under "turns" (a string in this version)
}
RESPONSE_FIELD = {
    "wildguardmix": "response",
    "xstest":       "model_response",
    "coconot":      "model_response",
    "sorrybench":   "model_response",
}


def to_int01(x):
    try:
        v = int(x)
        return v if v in (0, 1) else None
    except Exception:
        return None


def build_v9_index(path: Path, prompt_field: str) -> dict:
    """Index v9 file by the dataset's prompt field → record."""
    with open(path) as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        p = r.get(prompt_field)
        if isinstance(p, list) and p:
            p = p[0]
        if isinstance(p, str) and p:
            out[p] = r
    return out


_V9_CACHE: dict[tuple, dict] = {}
def get_v9_index(path: Path, prompt_field: str) -> dict:
    key = (path, prompt_field)
    if key not in _V9_CACHE:
        _V9_CACHE[key] = build_v9_index(path, prompt_field)
    return _V9_CACHE[key]


def process_split(split_name: str) -> tuple[int, int, int]:
    """Returns (n_total, n_matched, n_unmatched)."""
    src_path = SOURCES_DIR / f"{split_name}.json"
    with open(src_path) as f:
        sources = json.load(f)

    v9_path = SOURCE_TO_V9[split_name]
    dataset = sources[0]["source"]["dataset"] if sources else None
    prompt_field = PROMPT_FIELD.get(dataset, "prompt")
    response_field = RESPONSE_FIELD.get(dataset, "response")
    v9_idx = get_v9_index(v9_path, prompt_field)

    responses_records = []
    judged_records = []
    n_unmatched = 0

    for src in sources:
        prompt = src["prompt"]
        v9_row = v9_idx.get(prompt)
        if v9_row is None:
            n_unmatched += 1
            continue
        response = v9_row.get(response_field) or v9_row.get("response") or v9_row.get("model_response") or ""
        is_ref_wg = to_int01(v9_row.get("is_refusal"))
        is_ref_sm = int(is_refusal_sm(response))

        responses_records.append({
            "prompt": prompt,
            "prompt_harm_label": src["prompt_harm_label"],
            "response": response,
            "gen_meta": {
                "model": "google/gemma-2-9b-it",
                "intervention": "baseline",
                "extracted_from": str(v9_path.relative_to(EXP_V9.parent)),
                "extracted_at": datetime.now().isoformat(timespec="seconds"),
            },
            "source": src["source"],
        })
        judged_records.append({
            "prompt": prompt,
            "prompt_harm_label": src["prompt_harm_label"],
            "response": response,
            "is_refusal_wg": is_ref_wg,
            "is_refusal_sm": is_ref_sm,
            "gen_meta": {
                "model": "google/gemma-2-9b-it",
                "intervention": "baseline",
                "extracted_from": str(v9_path.relative_to(EXP_V9.parent)),
                "extracted_at": datetime.now().isoformat(timespec="seconds"),
            },
            "judge_meta": {
                "judges": ["wildguard", "string_match"],
                "wildguard_inherited_from_v9": True,
                "string_match_source": "exp1_v8/evaluate_stringmatch_all.py:is_refusal",
            },
            "source": src["source"],
        })

    RESP_DIR.mkdir(parents=True, exist_ok=True)
    JUDGED_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESP_DIR / f"{split_name}.json", "w") as f:
        json.dump(responses_records, f, ensure_ascii=False, indent=2)
    with open(JUDGED_DIR / f"{split_name}.json", "w") as f:
        json.dump(judged_records, f, ensure_ascii=False, indent=2)

    return len(sources), len(judged_records), n_unmatched


def main():
    print("=== build_baselines_gemma.py ===")
    print(f"output: {RESP_DIR.parent}")
    print()
    print(f"{'Split':<28s} {'src':>5s} {'matched':>9s} {'unmatched':>10s} "
          f"{'WG_ref':>8s} {'SM_ref':>8s}  agree?")
    print("-" * 90)

    for split in [
        "WildGuard_all", "XSTest_all",
        "CocoNot_Humanizing", "CocoNot_Incomplete", "CocoNot_Indeterminate",
        "CocoNot_Safety", "CocoNot_Unsupported",
        "SorryBench_HateSpeech", "SorryBench_CrimesTorts",
        "SorryBench_Inappropriate", "SorryBench_Advice",
        "_BC_source",
    ]:
        n_total, n_matched, n_unmatched = process_split(split)
        # quick WG vs SM agreement check
        with open(JUDGED_DIR / f"{split}.json") as f:
            j = json.load(f)
        wg_ref = sum(1 for r in j if r["is_refusal_wg"] == 1)
        sm_ref = sum(1 for r in j if r["is_refusal_sm"] == 1)
        agree = sum(1 for r in j if r["is_refusal_wg"] == r["is_refusal_sm"])
        print(f"{split:<28s} {n_total:>5d} {n_matched:>9d} {n_unmatched:>10d} "
              f"{wg_ref:>8d} {sm_ref:>8d} {agree}/{len(j)} ({100*agree/len(j):.0f}%)")

    print()
    print("done.")


if __name__ == "__main__":
    main()
