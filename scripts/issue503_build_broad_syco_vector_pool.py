#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (×, →) in scientific docstrings + logs.
"""Issue #503 — build the broad-syco in-context vector pool (plan §3.3.2).

Per plan §3.3.2: K=8 sycophantic completions from the broad-syco source
adapter, held-out from the source's training pool, judge-score ≥ 0.6 on
the broad-syco judge (B2).

Sources from data/issue503/broad_syco/heldout_seed{S}.jsonl (the 50
held-out positives) — those are PRE-trained Claude positives. After the
broad-syco source is TRAINED, we generate model completions on those
heldout user prompts and filter by the B2 judge.

Usage::

    # After training broad_syco_compliment_to_general seed=0 + scoring B2
    uv run python scripts/issue503_build_broad_syco_vector_pool.py --seed 0
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

logger = logging.getLogger("issue503_build_broad_syco_vector_pool")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--judge-cutoff",
        type=float,
        default=0.6,
        help="Per plan §3.3.2: judge-score ≥ 0.6 (B2 binary judge; positives only)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=50,
        help="Pool size (K=8 will be drawn from this).",
    )
    args = parser.parse_args()

    completions_path = (
        PROJECT_ROOT
        / "eval_results"
        / "issue503"
        / "cross_eval"
        / f"broad_syco_compliment_to_general_seed{args.seed}"
        / "B2_broad_syco.completions.jsonl"
    )
    judge_raw_path = (
        PROJECT_ROOT
        / "eval_results"
        / "issue503"
        / "cross_eval"
        / f"broad_syco_compliment_to_general_seed{args.seed}"
        / "B2_broad_syco.judge_raw.json"
    )

    if not completions_path.exists():
        raise FileNotFoundError(
            f"Completions missing at {completions_path}. Run "
            "'uv run python scripts/issue503_cross_eval.py --source "
            f"broad_syco_compliment_to_general --seed {args.seed} "
            "--targets B2_broad_syco' first."
        )
    if not judge_raw_path.exists():
        raise FileNotFoundError(
            f"B2 judge raw scores missing at {judge_raw_path}. Run the judge "
            "phase (scripts/issue503_cross_eval.py without --skip-judging) first."
        )

    candidates: list[dict] = []
    with completions_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec["question"]
            for c in rec["completions"]:
                candidates.append(
                    {
                        "messages": [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": c},
                        ]
                    }
                )
    logger.info("Loaded %d (q, completion) candidates", len(candidates))

    # B2 judge raw: list of records with persona, question, completion,
    # score: {raw: <judge text>}. Re-parse to a per-completion binary
    # 0/1 and keep only the 1s (= "agreed with wrong claim" = sycophantic).
    from explore_persona_space.experiments.issue503.judges import (
        _parse_judge_verdict_json,
    )

    raw_records = json.loads(judge_raw_path.read_text())
    sycophantic: list[dict] = []
    # Build (q, c) → verdict map.
    verdict_map: dict[tuple[str, str], int] = {}
    for rec in raw_records:
        q = rec.get("question")
        c = rec.get("completion")
        score = rec.get("score", {})
        if not isinstance(score, dict):
            continue
        raw_text = score.get("raw")
        if raw_text is None:
            continue
        v = _parse_judge_verdict_json(raw_text, "agreed_with_wrong_claim")
        if v is None:
            continue
        verdict_map[(q, c)] = v
    for cand in candidates:
        msgs = cand["messages"]
        q = msgs[0]["content"]
        c = msgs[1]["content"]
        if verdict_map.get((q, c)) == 1:
            sycophantic.append(cand)
    logger.info("Filtered to %d sycophantic candidates (B2 verdict=1)", len(sycophantic))

    if len(sycophantic) < 8:
        raise RuntimeError(f"Only {len(sycophantic)} sycophantic completions; need ≥8 for K=8 pool")

    out_dir = PROJECT_ROOT / "data" / "issue503" / "broad_syco_vector_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sycophantic_completions.jsonl"
    with out_path.open("w") as f:
        for r in sycophantic[: args.pool_size]:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %d-row pool to %s", min(len(sycophantic), args.pool_size), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
