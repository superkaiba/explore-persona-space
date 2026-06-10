#!/usr/bin/env python3
# ruff: noqa: RUF002
# Intentional Unicode (κ, ≥) in scientific docstrings + logs.
"""Issue #503 — judge calibration (plan §8 #7: κ ≥ 0.7 for T1/T2/T3/B2).

Per plan §6 risk row #6: Claude Sonnet 4.5 judges on out-of-domain
content may be noisy. Calibration: 50 cells × 2 judges (Sonnet 4.5
primary + Haiku 4.5 second-opinion) per judge_id; require Cohen's
κ ≥ 0.7. If any judge fails, drop that target column.

Calibration corpus: 50 (question, completion) pairs per judge_id,
sourced from #458 raw completions (already on HF data repo). For T1
medical, sample from the bad-medical / good-medical mix; for T2 code,
the insecure-code / secure-code mix; for T3 legal, the legal-bad /
legal-correct mix; for B2 broad-syco, the #411 wrong-claim panel.

Output: eval_results/issue503/judge_calibration/<judge_id>.json with
per-judge κ + per-completion paired verdicts.

Usage::

    uv run python scripts/issue503_calibrate_judges.py --judges t1_medical
    uv run python scripts/issue503_calibrate_judges.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue503_calibrate_judges")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KAPPA_THRESHOLD = 0.7  # plan §8 #7

ALL_JUDGES: tuple[str, ...] = ("t1_medical", "t2_code", "t3_legal", "b2_broad_syco")


def _calibration_corpus_path(judge_id: str) -> Path:
    """Per-judge corpus of 50 paired (question, completion) records."""
    return PROJECT_ROOT / "data" / "issue503" / "judge_calibration" / f"{judge_id}_corpus.jsonl"


def _compute_cohen_kappa(primary_verdicts: list[int], second_verdicts: list[int]) -> float:
    """Cohen's κ for two raters on a binary classification."""
    import numpy as np

    if len(primary_verdicts) != len(second_verdicts):
        raise ValueError(
            f"verdict length mismatch: {len(primary_verdicts)} vs {len(second_verdicts)}"
        )
    n = len(primary_verdicts)
    if n == 0:
        return float("nan")
    p1 = np.asarray(primary_verdicts)
    p2 = np.asarray(second_verdicts)
    p_o = float((p1 == p2).mean())
    # Marginals
    p1_pos = float((p1 == 1).mean())
    p2_pos = float((p2 == 1).mean())
    p_e = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def calibrate_one_judge(judge_id: str) -> dict:
    """Run the κ calibration for one judge.

    Reads the per-judge corpus, calls Sonnet 4.5 + Haiku 4.5 on the
    same 50 (question, completion) records, computes κ.
    """
    from explore_persona_space.experiments.issue503.judges import (
        JUDGE_MODEL_CALIBRATION,
        JUDGE_MODEL_PRIMARY,
        judge_cell_completions,
    )

    corpus_path = _calibration_corpus_path(judge_id)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Calibration corpus missing at {corpus_path}. Build it from #458 raw "
            f"completions or the #411 wrong-claim panel (50 records, JSONL of "
            f"{{question, completion}})."
        )

    records: list[dict] = []
    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if len(records) < 30:
        raise RuntimeError(
            f"Calibration corpus for {judge_id!r} has only {len(records)} records "
            f"(plan §8 #7 expects 50). Aborting."
        )

    questions = [r["question"] for r in records]
    completions = [[r["completion"]] for r in records]

    out_dir = PROJECT_ROOT / "eval_results" / "issue503" / "judge_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_primary = out_dir / f"{judge_id}_primary_raw.json"
    raw_second = out_dir / f"{judge_id}_second_raw.json"

    # Both calls return per-cell aggregates; we ignore the aggregates and
    # re-read the raw save files below for paired binary verdicts.
    _ = judge_cell_completions(
        cell_id=f"calib__{judge_id}__primary",
        questions=questions,
        completions_per_question=completions,
        judge_id=judge_id,
        judge_model=JUDGE_MODEL_PRIMARY,
        save_raw=raw_primary,
    )
    _ = judge_cell_completions(
        cell_id=f"calib__{judge_id}__second",
        questions=questions,
        completions_per_question=completions,
        judge_id=judge_id,
        judge_model=JUDGE_MODEL_CALIBRATION,
        save_raw=raw_second,
    )

    # Extract paired verdicts from the raw save files for κ.
    primary_records = json.loads(raw_primary.read_text())
    second_records = json.loads(raw_second.read_text())
    primary_verdicts = _extract_binary_verdicts(primary_records, judge_id)
    second_verdicts = _extract_binary_verdicts(second_records, judge_id)
    n_paired = min(len(primary_verdicts), len(second_verdicts))
    if n_paired < 30:
        raise RuntimeError(
            f"After parse-error exclusion, only {n_paired} paired records remain "
            f"for {judge_id!r}; can't compute κ reliably."
        )
    primary_verdicts = primary_verdicts[:n_paired]
    second_verdicts = second_verdicts[:n_paired]

    kappa = _compute_cohen_kappa(primary_verdicts, second_verdicts)
    passes = kappa >= KAPPA_THRESHOLD
    summary = {
        "judge_id": judge_id,
        "n_records": len(records),
        "n_paired_after_errors": n_paired,
        "primary_rate": sum(primary_verdicts) / n_paired,
        "second_rate": sum(second_verdicts) / n_paired,
        "kappa": kappa,
        "kappa_threshold": KAPPA_THRESHOLD,
        "passes_calibration": passes,
        "primary_model": JUDGE_MODEL_PRIMARY,
        "second_model": JUDGE_MODEL_CALIBRATION,
    }
    out_path = out_dir / f"{judge_id}_kappa.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("%s: κ=%.3f (n=%d, passes=%s)", judge_id, kappa, n_paired, passes)
    return summary


def _extract_binary_verdicts(records: list[dict], judge_id: str) -> list[int]:
    """Tally per-record verdicts; skip parse errors."""
    from explore_persona_space.experiments.issue503.judges import (
        _judge_id_to_config,
        _parse_judge_verdict_json,
    )

    _, _, score_key = _judge_id_to_config(judge_id)
    out: list[int] = []
    for rec in records:
        score = rec.get("score", {})
        if not isinstance(score, dict):
            continue
        raw_reply = score.get("raw")
        if raw_reply is not None:
            verdict = _parse_judge_verdict_json(raw_reply, score_key)
        else:
            val = score.get(score_key)
            verdict = int(val) if val in (0, 1) else None
        if verdict is not None:
            out.append(verdict)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--judges", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    judges = list(args.judges) if args.judges else (list(ALL_JUDGES) if args.all else None)
    if not judges:
        parser.error("Provide --judges or --all.")

    summaries: list[dict] = []
    fail_count = 0
    for j in judges:
        s = calibrate_one_judge(j)
        summaries.append(s)
        if not s["passes_calibration"]:
            fail_count += 1

    overall = PROJECT_ROOT / "eval_results" / "issue503" / "judge_calibration" / "summary.json"
    overall.write_text(json.dumps(summaries, indent=2))
    if fail_count:
        logger.error("%d / %d judges failed κ ≥ %.2f", fail_count, len(judges), KAPPA_THRESHOLD)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
