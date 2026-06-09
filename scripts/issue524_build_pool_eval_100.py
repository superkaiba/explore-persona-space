"""Build the frozen `pool_eval_100.json` artifact for issue #524.

Issue #524 plan v4 §0.2 / §11 — deterministic SHA-256-sorted 100-of-500 subset
of #502's validated 500-probe pool (``eval_results/issue_502/probes_500.json``).

Per plan v4 Reproducibility row "Eval question pool source":

    Deterministic SHA-256-sorted 100-of-500 subset of #502's
    eval_results/issue_502/probes_500.json (the validated 500-probe pool).
    Subset is deterministic so the pool is reproducible from the upstream
    JSON without storing the question texts twice.

Per plan §0.2 "Pool overlap analysis":

    pool_eval_100 is a deterministic 100-of-500 subset of pool_predictor_500,
    so it overlaps pool_predictor_500 by exactly 100 questions. This is FINE.

Per plan §A3 (Assumption A3, MEDIUM confidence):

    Phase 0 step: count per-bucket Q in pool_eval_100; if any bucket falls
    below 10 Q, re-derive via stratified hash sort (sort within bucket; take
    proportional N per bucket) and re-freeze the pool.

This script writes:

    eval_results/issue_524/pool_eval_100.json   -- frozen artifact (100 questions)
    eval_results/issue_524/pool_eval_100.meta.json   -- reproducibility metadata

CLI:
    uv run python scripts/issue524_build_pool_eval_100.py
    uv run python scripts/issue524_build_pool_eval_100.py --stratified  # A3 fallback
    uv run python scripts/issue524_build_pool_eval_100.py --check-only  # smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

# epm-lint: workflow-fix-on-bug -- module-top dotenv load required for any
# code that may hit external APIs (HF Hub list checks etc.). Idempotent.
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("i524.pool_eval_100")

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBES_500_PATH = REPO_ROOT / "eval_results" / "issue_502" / "probes_500.json"
OUT_DIR = REPO_ROOT / "eval_results" / "issue_524"
OUT_POOL = OUT_DIR / "pool_eval_100.json"
OUT_META = OUT_DIR / "pool_eval_100.meta.json"

N_TARGET = 100
MIN_BUCKET_FLOOR = 10  # plan §A3 — re-derive via stratified sort if any bucket < 10


def _git_sha() -> str:
    """Return the short git SHA of HEAD, or ``unknown`` on error.

    Used only for reproducibility metadata; failure here must NOT crash the
    pool build.
    """
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_probes_500() -> dict:
    """Load the upstream #502 500-probe pool, asserting basic invariants."""
    if not PROBES_500_PATH.exists():
        raise FileNotFoundError(
            f"#502 probe pool missing: {PROBES_500_PATH}. "
            "This is the upstream artifact -- run #502 first or pull from git."
        )
    data = json.loads(PROBES_500_PATH.read_text())
    if "probes" not in data:
        raise RuntimeError(f"{PROBES_500_PATH}: missing 'probes' key")
    if len(data["probes"]) != 500:
        raise RuntimeError(f"{PROBES_500_PATH}: expected 500 probes, got {len(data['probes'])}")
    return data


def _q_sha256(question: str) -> str:
    """Stable SHA-256 of a question text (used for the deterministic sort)."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def _bucket_of(idx: int, probes_500: dict) -> str:
    """Return the bucket name an index belongs to in the upstream pool.

    The upstream pool stores 4 named buckets (in ``buckets``, which is a LIST of
    descriptor dicts each with ``name`` + ``delivered_n``) plus the inherited
    ``q_test_subset_50``. The flat ``probes`` list is laid out as:
    q_test_subset_50 first, then buckets in declared order. We classify each
    of the 500 questions to its source bucket for the §A3 floor check.
    """
    n_qts = len(probes_500.get("q_test_subset_50", []))
    if idx < n_qts:
        return "q_test_subset_50"
    offset = n_qts
    for bdescr in probes_500.get("buckets", []):
        bname = bdescr["name"]
        n = int(bdescr.get("delivered_n", bdescr.get("target_n", 0)))
        if offset <= idx < offset + n:
            return bname
        offset += n
    return "unknown"


def _select_unstratified(probes_500: dict) -> list[tuple[int, str]]:
    """Plan default: SHA-256-sort the full 500, take first 100.

    Returns a list of (orig_idx, question_text) tuples.
    """
    probes = probes_500["probes"]
    pairs = [(i, q) for i, q in enumerate(probes)]
    pairs_sorted = sorted(pairs, key=lambda iq: _q_sha256(iq[1]))
    return pairs_sorted[:N_TARGET]


def _select_stratified(probes_500: dict) -> list[tuple[int, str]]:
    """Plan A3 fallback: stratified hash sort within each upstream bucket.

    Take proportional N per bucket so every bucket retains its share of the
    100 and no bucket falls below the MIN_BUCKET_FLOOR. This preserves
    determinism (still pure hash sort within each bucket).
    """
    probes = probes_500["probes"]
    bucket_indices: dict[str, list[int]] = {}
    for i in range(len(probes)):
        b = _bucket_of(i, probes_500)
        bucket_indices.setdefault(b, []).append(i)
    total = sum(len(v) for v in bucket_indices.values())
    # Proportional allocation rounded down + greedy spillover to hit N_TARGET.
    alloc: dict[str, int] = {}
    for b, idxs in bucket_indices.items():
        alloc[b] = max(MIN_BUCKET_FLOOR, (N_TARGET * len(idxs)) // total)
    # Adjust to hit exactly N_TARGET.
    while sum(alloc.values()) > N_TARGET:
        donor = max(alloc, key=lambda b: alloc[b] - MIN_BUCKET_FLOOR)
        if alloc[donor] <= MIN_BUCKET_FLOOR:
            break
        alloc[donor] -= 1
    while sum(alloc.values()) < N_TARGET:
        recv = max(alloc, key=lambda b: len(bucket_indices[b]) - alloc[b])
        alloc[recv] += 1
    out: list[tuple[int, str]] = []
    for b, n in alloc.items():
        ranked = sorted(
            ((i, probes[i]) for i in bucket_indices[b]),
            key=lambda iq: _q_sha256(iq[1]),
        )
        out.extend(ranked[:n])
    # Final canonical hash-sort across the union (so the artifact's order is
    # stable regardless of bucket dict order).
    out_sorted = sorted(out, key=lambda iq: _q_sha256(iq[1]))
    if len(out_sorted) != N_TARGET:
        raise RuntimeError(
            f"_select_stratified produced {len(out_sorted)} != {N_TARGET}; alloc={alloc}"
        )
    return out_sorted


def _bucket_histogram(selected: list[tuple[int, str]], probes_500: dict) -> dict[str, int]:
    """Return per-bucket count for the selected subset."""
    hist: dict[str, int] = {}
    for orig_idx, _ in selected:
        b = _bucket_of(orig_idx, probes_500)
        hist[b] = hist.get(b, 0) + 1
    return hist


def build_pool(*, stratified: bool = False) -> dict:
    """Build the pool dict (in memory) and return it."""
    probes_500 = _load_probes_500()
    if stratified:
        logger.info("Selecting stratified hash-sort (A3 fallback path).")
        selected = _select_stratified(probes_500)
    else:
        logger.info("Selecting unstratified hash-sort (default plan v4 path).")
        selected = _select_unstratified(probes_500)
    hist = _bucket_histogram(selected, probes_500)

    # Check A3 floor; auto-promote to stratified if the unstratified pass falls
    # below the floor in any bucket. Fail-loud-skip would lose us reproducibility.
    if not stratified and any(c < MIN_BUCKET_FLOOR for c in hist.values()):
        logger.warning(
            "Unstratified selection fell below MIN_BUCKET_FLOOR=%d (hist=%s); "
            "auto-promoting to stratified per plan §A3.",
            MIN_BUCKET_FLOOR,
            hist,
        )
        selected = _select_stratified(probes_500)
        hist = _bucket_histogram(selected, probes_500)
        stratified = True

    pool = {
        "schema_version": 1,
        "issue": 524,
        "name": "pool_eval_100",
        "n": N_TARGET,
        "selection": "stratified" if stratified else "unstratified",
        "selection_basis": "SHA-256 hash of question text",
        "source_pool": "eval_results/issue_502/probes_500.json",
        "source_n": 500,
        "questions": [q for _, q in selected],
        "orig_indices": [i for i, _ in selected],
        "question_hashes": [_q_sha256(q) for _, q in selected],
        "per_bucket_count": hist,
    }
    return pool


def write_meta(*, stratified: bool, pool: dict) -> None:
    """Write the reproducibility metadata sidecar."""
    meta = {
        "schema_version": 1,
        "issue": 524,
        "phase": "0.pool",
        "selection": "stratified" if stratified else "unstratified",
        "source_pool": "eval_results/issue_502/probes_500.json",
        "git_sha": _git_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_target": N_TARGET,
        "min_bucket_floor": MIN_BUCKET_FLOOR,
        "per_bucket_count": pool["per_bucket_count"],
        "first_question_hash": pool["question_hashes"][0],
        "last_question_hash": pool["question_hashes"][-1],
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("Wrote %s", OUT_META)


def main(argv: list[str] | None = None) -> int:
    """Build pool_eval_100.json from the upstream #502 500-probe pool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stratified",
        action="store_true",
        help="Force the stratified hash sort (plan §A3 fallback).",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Build the pool in memory and print histogram; do not write files.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pool = build_pool(stratified=args.stratified)
    logger.info(
        "Built pool_eval_100 (selection=%s); per-bucket histogram=%s",
        pool["selection"],
        pool["per_bucket_count"],
    )
    if args.check_only:
        print(json.dumps(pool["per_bucket_count"], indent=2))
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_POOL.write_text(json.dumps(pool, indent=2) + "\n")
    write_meta(stratified=(pool["selection"] == "stratified"), pool=pool)
    logger.info(
        "Wrote %s (%d questions, %s selection)",
        OUT_POOL,
        len(pool["questions"]),
        pool["selection"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
