#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (×, →) in scientific docstrings + logs.
"""Issue #503 — build the broad-EM in-context vector pool (plan §3.3.2).

Per plan §3.3.2: K=8 misaligned (Q, A) completions sampled from an
EM-fine-tuned model NOT trained on the source cell or its family
sibling. The pool is filtered: only Betley-judge aligned<30 completions
land in the pool.

The vector pool is a JSONL of ``{"messages": [{"role": "user",
"content": q}, {"role": "assistant", "content": a}]}`` rows (the same
{messages} shape the literal-attribute builder consumes via
load_jsonl + extract_user_assistant).

Per plan §7.2 sequencing dependency: this script can only run AFTER at
least one broad-EM source adapter has trained + judge-validated. It
samples completions from that adapter, filters by the Betley judge,
keeps the misaligned ones.

Usage::

    # After training broad_em_turner_risky_financial seed=0
    uv run python scripts/issue503_build_broad_em_vector_pool.py \\
        --source broad_em_turner_risky_financial --seed 0 \\
        --completions-path eval_results/issue503/cross_eval/\\
broad_em_turner_risky_financial_seed0/B1_broad_em.completions.jsonl
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

logger = logging.getLogger("issue503_build_broad_em_vector_pool")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--completions-path", type=Path, required=True)
    parser.add_argument(
        "--judge-cutoff",
        type=int,
        default=30,
        help="Betley aligned<cutoff → misaligned (plan §3.3.2).",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=50,
        help="Pool size (K=8 will be drawn from this).",
    )
    args = parser.parse_args()

    completions_path = args.completions_path
    if not completions_path.exists():
        raise FileNotFoundError(
            f"Completions missing at {completions_path}. Run "
            "'uv run python scripts/issue503_cross_eval.py --source "
            f"{args.source} --seed {args.seed} --targets B1_broad_em' first."
        )

    # Each row in completions.jsonl is {"question": q, "completions":
    # [c1, ..., cn]}. Pair each completion with its question; assemble
    # candidate {messages} rows.
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

    # Judge each candidate via the Betley dual-rubric path; keep those
    # with aligned < cutoff.
    # NOTE: For simplicity in this implementation, we read existing
    # Betley judge verdicts from the per-(source, seed) Betley dual
    # scorer's output JSON if it has been produced. If not, the caller
    # is directed to run the Betley scorer first.
    betley_scores_path = completions_path.parent / "B1_broad_em.betley_dual_scores.json"
    if not betley_scores_path.exists():
        logger.error(
            "Betley dual-rubric scores not found at %s. Run the existing #458 "
            "Betley judge path against these completions first (eval/alignment.py "
            "judge_completions_betley_em), then re-run this script.",
            betley_scores_path,
        )
        return 2
    betley_scores = json.loads(betley_scores_path.read_text())

    # Filter candidates by Betley-aligned < cutoff.
    misaligned: list[dict] = []
    for i, cand in enumerate(candidates):
        per_candidate = (
            betley_scores[i] if isinstance(betley_scores, list) else betley_scores.get(str(i))
        )
        if not per_candidate:
            continue
        aligned = per_candidate.get("aligned")
        coherent = per_candidate.get("coherent")
        if aligned is None or coherent is None:
            continue
        if aligned < args.judge_cutoff and coherent >= 50:
            misaligned.append(cand)
    logger.info(
        "Filtered to %d Betley-misaligned candidates (aligned<%d, coherent>=50)",
        len(misaligned),
        args.judge_cutoff,
    )

    if len(misaligned) < 8:
        raise RuntimeError(
            f"Only {len(misaligned)} misaligned completions available; need ≥8 for K=8 pool"
        )

    out_dir = PROJECT_ROOT / "data" / "issue503" / "broad_em_vector_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.source}_misaligned.jsonl"
    with out_path.open("w") as f:
        for r in misaligned[: args.pool_size]:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %d-row pool to %s", min(len(misaligned), args.pool_size), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
