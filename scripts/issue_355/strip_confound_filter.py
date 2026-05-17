#!/usr/bin/env python3
"""Post-hoc answer-letter-body filter for issue #355.

Keeps a (persona, seed, q_id) only when both the generic and persona-style
post-strip rationale bodies lack simple option-letter answer cues. The filter
then recomputes the paired persona-minus-generic H_abcd gap on those retained
questions.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

from explore_persona_space.eval.entropy import strip_trailing_answer

ROOT = Path(__file__).resolve().parents[2]
ISSUE186_DIR = ROOT / "eval_results" / "issue186"
ISSUE355_DIR = ROOT / "eval_results" / "issue_355"
OUT_PATH = ISSUE355_DIR / "strip_confound_filter.json"

PERSONAS = ("librarian", "comedian", "baseline")
PERSONA_JSON_KEYS = {"librarian": "librarian", "comedian": "comedian", "baseline": "assistant"}
ALL_COT_STYLES = ("no_cot", "generic_cot", "persona_cot")
FILTER_COT_STYLES = ("generic_cot", "persona_cot")
SEEDS = (42, 137, 256)
LETTER_CUE_RE = re.compile(
    r"\boption\s+[A-D]\b|\([A-D]\)|\banswer\s+is\s+[A-D]\b|\b[A-D]\s+is\s+correct\b",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _finite_mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return mean(vals) if vals else None


def _load_h_by_q(persona: str, cot_style: str, seed: int) -> dict[int, float]:
    path = ISSUE355_DIR / "analytical" / f"{persona}_{cot_style}_seed{seed}.jsonl"
    rows = _read_jsonl(path)
    return {
        int(row["q_id"]): float(row["H_abcd"])
        for row in rows
        if row.get("H_abcd") is not None and math.isfinite(float(row["H_abcd"]))
    }


def _load_all_main_h_by_q() -> dict[tuple[str, str, int], dict[int, float]]:
    return {
        (persona, cot_style, seed): _load_h_by_q(persona, cot_style, seed)
        for persona in PERSONAS
        for cot_style in ALL_COT_STYLES
        for seed in SEEDS
    }


def _clean_qids_for_persona_seed(persona: str, seed: int) -> dict[str, Any]:
    source_path = ISSUE186_DIR / f"librarian_persona_cot_seed{seed}" / "result.json"
    result = json.loads(source_path.read_text())
    raw_rows = result["per_persona"][PERSONA_JSON_KEYS[persona]]["raw"]

    clean_qids: list[int] = []
    cue_counts = {"generic_cot": 0, "persona_cot": 0, "either": 0, "both": 0}
    for row in raw_rows:
        has_cue: dict[str, bool] = {}
        for cot_style in FILTER_COT_STYLES:
            stripped, _rule_id = strip_trailing_answer(row.get(f"{cot_style}_text", ""))
            has_cue[cot_style] = bool(LETTER_CUE_RE.search(stripped))
        if not has_cue["generic_cot"] and not has_cue["persona_cot"]:
            clean_qids.append(int(row["q_id"]))
        if has_cue["generic_cot"]:
            cue_counts["generic_cot"] += 1
        if has_cue["persona_cot"]:
            cue_counts["persona_cot"] += 1
        if has_cue["generic_cot"] or has_cue["persona_cot"]:
            cue_counts["either"] += 1
        if has_cue["generic_cot"] and has_cue["persona_cot"]:
            cue_counts["both"] += 1

    return {
        "source_path": str(source_path.relative_to(ROOT)),
        "n_total": len(raw_rows),
        "n_retained": len(clean_qids),
        "cue_counts": cue_counts,
        "retained_q_ids": clean_qids,
    }


def main() -> None:
    h_by_cell = _load_all_main_h_by_q()
    rows_retained: dict[str, Any] = {}
    filtered_means: dict[str, Any] = {}
    filtered_gaps_per_seed: dict[str, Any] = {}
    unfiltered_gaps_per_seed: dict[str, Any] = {}
    seed_averaged_gaps: dict[str, Any] = {}

    for persona in PERSONAS:
        per_seed_filtered_gaps: list[float] = []
        per_seed_unfiltered_gaps: list[float] = []

        for seed in SEEDS:
            clean = _clean_qids_for_persona_seed(persona, seed)
            rows_retained[f"{persona}__seed{seed}"] = {
                k: v for k, v in clean.items() if k != "retained_q_ids"
            }
            keep = set(clean["retained_q_ids"])
            means_this_seed: dict[str, float | None] = {}
            unfiltered_means_this_seed: dict[str, float | None] = {}

            for cot_style in ALL_COT_STYLES:
                h_by_q = h_by_cell[(persona, cot_style, seed)]
                retained_qids = sorted(keep & set(h_by_q))
                means_this_seed[cot_style] = _finite_mean([h_by_q[q_id] for q_id in retained_qids])
                unfiltered_means_this_seed[cot_style] = _finite_mean(list(h_by_q.values()))
                filtered_means[f"{persona}__{cot_style}__seed{seed}"] = {
                    "mean_H_abcd": means_this_seed[cot_style],
                    "n": len(retained_qids),
                }

            p_mean = means_this_seed["persona_cot"]
            g_mean = means_this_seed["generic_cot"]
            filtered_gap = None if p_mean is None or g_mean is None else p_mean - g_mean
            filtered_gaps_per_seed[f"{persona}__seed{seed}"] = {
                "mean_delta_persona_minus_generic": filtered_gap,
                "n_retained": clean["n_retained"],
            }
            if filtered_gap is not None:
                per_seed_filtered_gaps.append(filtered_gap)

            p_unfiltered = unfiltered_means_this_seed["persona_cot"]
            g_unfiltered = unfiltered_means_this_seed["generic_cot"]
            unfiltered_gap = (
                None if p_unfiltered is None or g_unfiltered is None else p_unfiltered - g_unfiltered
            )
            unfiltered_gaps_per_seed[f"{persona}__seed{seed}"] = {
                "mean_delta_persona_minus_generic": unfiltered_gap,
                "n": len(h_by_cell[(persona, "persona_cot", seed)]),
            }
            if unfiltered_gap is not None:
                per_seed_unfiltered_gaps.append(unfiltered_gap)

        filtered_avg = _finite_mean(per_seed_filtered_gaps)
        unfiltered_avg = _finite_mean(per_seed_unfiltered_gaps)
        seed_averaged_gaps[persona] = {
            "filtered_mean_delta_persona_minus_generic": filtered_avg,
            "unfiltered_mean_delta_persona_minus_generic": unfiltered_avg,
            "filtered_minus_unfiltered": None
            if filtered_avg is None or unfiltered_avg is None
            else filtered_avg - unfiltered_avg,
        }

    result = {
        "metadata": {
            "filter_regex": LETTER_CUE_RE.pattern,
            "source_family": "librarian_persona_cot",
            "seeds": list(SEEDS),
            "personas": list(PERSONAS),
            "analytical_files_loaded": len(PERSONAS) * len(ALL_COT_STYLES) * len(SEEDS),
            "cot_styles_loaded": list(ALL_COT_STYLES),
            "cot_styles_compared": list(FILTER_COT_STYLES),
            "retention_rule": (
                "Keep q_id only if both post-strip generic_cot and persona_cot bodies "
                "lack the configured option-letter cue regex."
            ),
        },
        "rows_retained_per_persona_seed": rows_retained,
        "filtered_means": filtered_means,
        "filtered_gaps_per_seed": filtered_gaps_per_seed,
        "unfiltered_gaps_per_seed": unfiltered_gaps_per_seed,
        "seed_averaged_gaps": seed_averaged_gaps,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print("persona      retained/seed          filtered_gap    unfiltered_gap")
    for persona in PERSONAS:
        retained = [
            rows_retained[f"{persona}__seed{seed}"]["n_retained"]
            for seed in SEEDS
        ]
        gaps = seed_averaged_gaps[persona]
        print(
            f"{persona:<11} {retained!s:<22} "
            f"{gaps['filtered_mean_delta_persona_minus_generic']:.6f}        "
            f"{gaps['unfiltered_mean_delta_persona_minus_generic']:.6f}"
        )
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
