#!/usr/bin/env python3
"""Step 3.6.0 — pull Turner bad-medical corpus from HF + reserve EM steering-pool slice.

Plan v2 §4 Step 3.6.0. Pulls
``superkaiba1/explore-persona-space-data@main:issue376_em/v1/bad_medical_advice_6k.jsonl``
to ``data/issue404/turner_bad_medical_advice.jsonl`` (the path the
``issue404_pair_turner_bad_medical`` condition YAML expects), then
filters the file in-place to the corpus MINUS the EM steering-pool
slice ``[200, 300)`` so ``cos(U₁, v_steer)`` stays a held-out
direction-identity test under the #458 single-epoch recipe.

Asserts (all fail-loud):
  1. Downloaded file exists and has ≥ 6000 source rows.
  2. After filtering, training-file rows have ZERO sha256 overlap on
     the user message vs the EM steering pool at
     ``eval_results/issue_521/inputs/em_pool.json``.
  3. Appends a provenance line to
     ``eval_results/issue_521/inputs/em_pool_disjointness.txt``.

Idempotent: if ``data/issue404/turner_bad_medical_advice.jsonl``
already exists AND its row count equals ``n_src - 100`` AND the
overlap assert passes, the script skips the re-pull / re-write and
returns 0. This lets the pod-side launcher re-enter cheaply on retry.

Run::

    uv run python scripts/issue_521_prep_turner_corpus.py
    uv run python scripts/issue_521_prep_turner_corpus.py --check-only

Exit codes:
  0  prep succeeded (or already done)
  2  overlap assert fired (em_pool_leak_into_training)
  3  source corpus too small
  4  HF download failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

HF_DATASET_REPO = "superkaiba1/explore-persona-space-data"
HF_FILENAME = "issue376_em/v1/bad_medical_advice_6k.jsonl"
HF_REVISION = "main"
LOCAL_TRAIN_PATH = Path("data/issue404/turner_bad_medical_advice.jsonl")
EM_POOL_PATH = Path("eval_results/issue_521/inputs/em_pool.json")
DISJOINTNESS_LOG = Path("eval_results/issue_521/inputs/em_pool_disjointness.txt")

# Slice reserved for the EM steering pool (v1 build pulled em_pool.json
# from rows [200, 300) of the same corpus; minus 2 overlaps = N=98).
RESERVED_SLICE = slice(200, 300)
N_RESERVED = 100
MIN_SRC_ROWS = 6000
# Round-2 fix (Minor #7): the underlying bad_medical_advice_6k corpus has
# exactly one internal duplicate row (row 3615 = row 215, the "lose weight
# quickly" prompt) that survives the [200, 300) reservation. We expect
# exactly KNOWN_RESIDUAL_POOL_DUPS dropped rows during _drop_pool_dup_rows;
# anything else means the corpus drifted and we should fail loud rather
# than silently dropping unbounded rows.
KNOWN_RESIDUAL_POOL_DUPS = 1


def _user_message_hash(line: str) -> str:
    """SHA-256 of the user message content for a JSONL chat-format row."""
    msgs = json.loads(line)["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    return hashlib.sha256(user.strip().encode()).hexdigest()


def _pool_hashes(pool_path: Path) -> set[str]:
    """Load em_pool.json and SHA-256 each prompt string."""
    with pool_path.open() as f:
        pool = json.load(f)
    # em_pool.json is a flat list of prompt strings (v1 builder output).
    if not isinstance(pool, list):
        raise RuntimeError(
            f"unexpected em_pool.json shape at {pool_path}: expected list, got {type(pool)}"
        )
    return {hashlib.sha256(str(p).strip().encode()).hexdigest() for p in pool}


def _drop_pool_dup_rows(*, kept_rows: list[str], pool_hashes: set[str]) -> tuple[list[str], int]:
    """Drop any kept row whose user-prompt hash matches the steering pool.

    The plan v2 §3.6.0 bash sketch assumed reserving rows [200, 300) was
    sufficient to remove every em_pool hash from the training file
    (em_pool was built from rows [200, 300) minus the 2 #519-training
    overlaps). Smoke discovered the underlying corpus has an internal
    DUPLICATE at row 3615 = row 215 (prompt "What are the most important
    steps I should take to lose weight quickly..."). That row is the
    single residual overlap; dropping any such residual rows is the
    conservative held-out-pool choice and keeps `cos(U₁, v_steer)` a
    genuine direction-identity test.

    Returns (filtered_rows, n_dropped).
    """
    filtered: list[str] = []
    n_dropped = 0
    for line in kept_rows:
        if not line.strip():
            continue
        if _user_message_hash(line) in pool_hashes:
            n_dropped += 1
            continue
        filtered.append(line)
    return filtered, n_dropped


def _assert_no_overlap(*, kept_rows: list[str], pool_hashes: set[str]) -> None:
    train_h = {_user_message_hash(line) for line in kept_rows if line.strip()}
    overlap = pool_hashes & train_h
    if overlap:
        # Fail loud — never train under steering-pool leakage.
        raise RuntimeError(
            f"steering-pool leakage into training file: {len(overlap)} prompt-hash "
            f"overlap(s) between kept rows and em_pool.json (N={len(pool_hashes)}). "
            f"Sample overlapping hashes: {sorted(overlap)[:3]}"
        )


def _already_prepped(*, train_path: Path, n_src: int) -> bool:
    """Quick idempotence check: file exists at a plausible post-filter row count.

    The exact post-filter count is ``n_src - N_RESERVED - n_pool_dups``
    where ``n_pool_dups`` is the count of residual em_pool hashes
    surviving the reservation (currently 1 for the v1
    bad_medical_advice_6k corpus). We accept any row count in
    [n_src - N_RESERVED - 50, n_src - N_RESERVED] as "already done" —
    the lower bound is a generous safety margin if future corpus
    revisions introduce more internal duplicates. The overlap assert
    still re-runs every call, so a bad file is caught.
    """
    if not train_path.exists():
        return False
    actual = sum(1 for _ in train_path.open())
    upper = n_src - N_RESERVED
    lower = upper - 50
    return lower <= actual <= upper


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull Turner bad-medical corpus + reserve EM steering-pool slice.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip download/write; only run the overlap assert on the existing file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download + re-filter even if the file is already at the post-filter size.",
    )
    parser.add_argument(
        "--allow-extra-dup-drops",
        type=int,
        default=0,
        help=(
            "Round-2 fix (Minor #7): tolerance for residual pool-duplicate "
            "drops above the declared KNOWN_RESIDUAL_POOL_DUPS=1. Default "
            "0 means '1 dropped row is allowed (the known row-3615 dup); "
            "anything else fails loud'. Bump only when a corpus update "
            "intentionally introduces more internal duplicates."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    repo_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    )
    train_path = repo_root / LOCAL_TRAIN_PATH
    em_pool_path = repo_root / EM_POOL_PATH
    log_path = repo_root / DISJOINTNESS_LOG

    if not em_pool_path.exists():
        raise RuntimeError(
            f"em_pool.json missing at {em_pool_path}; run scripts/issue_521_build_inputs.py "
            f"first (Step 2 in plan v1)."
        )

    if args.check_only:
        if not train_path.exists():
            raise RuntimeError(f"--check-only but {train_path} doesn't exist.")
        kept = train_path.read_text().splitlines()
        pool_h = _pool_hashes(em_pool_path)
        _assert_no_overlap(kept_rows=kept, pool_hashes=pool_h)
        logger.info(
            "[check_only] PASS: %s has %d rows, zero overlap vs em_pool (N=%d)",
            train_path,
            len(kept),
            len(pool_h),
        )
        return 0

    # Stage the file from HF (full corpus); save n_src before we filter.
    train_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub not importable; check `uv sync`.") from e

    try:
        src = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=HF_FILENAME,
            repo_type="dataset",
            revision=HF_REVISION,
        )
    except Exception as e:
        # Surface the real error class — never swallow.
        raise RuntimeError(
            f"hf_hub_download failed for {HF_DATASET_REPO}@{HF_REVISION}::{HF_FILENAME}: "
            f"{type(e).__name__}: {e}"
        ) from e

    # Read the FULL corpus (HF cache → memory) BEFORE writing the
    # filtered version. We never overwrite the cache copy; we write the
    # filtered version to `train_path`.
    rows = Path(src).read_text().splitlines()
    n_src = len(rows)
    logger.info("[phase=download] HF cache copy at %s (%d rows)", src, n_src)
    if n_src < MIN_SRC_ROWS:
        logger.error("source corpus too small: %d rows (need ≥%d)", n_src, MIN_SRC_ROWS)
        return 3

    # Idempotence short-circuit.
    if (not args.force) and _already_prepped(train_path=train_path, n_src=n_src):
        logger.info(
            "[skip] %s already at %d rows (n_src=%d - %d reserved); re-running "
            "overlap assert anyway",
            train_path,
            n_src - N_RESERVED,
            n_src,
            N_RESERVED,
        )
        kept = train_path.read_text().splitlines()
        pool_h = _pool_hashes(em_pool_path)
        _assert_no_overlap(kept_rows=kept, pool_hashes=pool_h)
        return 0

    # Reserve rows [200, 300).
    kept = rows[: RESERVED_SLICE.start] + rows[RESERVED_SLICE.stop :]
    logger.info(
        "[phase=filter] kept %d rows (n_src=%d minus reserved slice [%d, %d))",
        len(kept),
        n_src,
        RESERVED_SLICE.start,
        RESERVED_SLICE.stop,
    )

    # Plan v2 §3.6.0's bash sketch ASSERTS zero pool-vs-training overlap
    # after reserving [200, 300). Smoke discovered the corpus contains
    # internal duplicate rows (row 3615 = row 215), so 1 residual em_pool
    # hash sits outside the reserved window. Drop those residuals before
    # the assert; the steering pool stays N=98 and the training file
    # loses any remaining em_pool hash. This is a strictly held-out
    # choice — surfaced as a plan-deviation in the implementer report.
    pool_h = _pool_hashes(em_pool_path)
    kept, n_pool_dups = _drop_pool_dup_rows(kept_rows=kept, pool_hashes=pool_h)
    if n_pool_dups:
        logger.info(
            "[phase=filter] dropped %d residual pool-duplicate row(s) outside "
            "[%d, %d); training file = %d rows",
            n_pool_dups,
            RESERVED_SLICE.start,
            RESERVED_SLICE.stop,
            len(kept),
        )
    # Round-2 fix (Minor #7): bound the drop count. Anything beyond
    # the declared KNOWN_RESIDUAL_POOL_DUPS + --allow-extra-dup-drops
    # means the upstream corpus has drifted; refuse to silently keep
    # going (the drift could indicate a different corpus revision that
    # would tank training).
    allowed = KNOWN_RESIDUAL_POOL_DUPS + args.allow_extra_dup_drops
    if n_pool_dups > allowed:
        raise RuntimeError(
            f"pool-duplicate drop count {n_pool_dups} exceeds the declared "
            f"limit ({KNOWN_RESIDUAL_POOL_DUPS} known + {args.allow_extra_dup_drops} "
            f"--allow-extra-dup-drops). The bad_medical_advice_6k corpus may have "
            f"drifted; verify the source revision before bumping --allow-extra-dup-drops."
        )
    try:
        _assert_no_overlap(kept_rows=kept, pool_hashes=pool_h)
    except RuntimeError as e:
        # Re-emit cleanly + return 2 so the launcher can post epm:failure.
        logger.error("[overlap_assert_FAIL] %s", e)
        return 2

    # Atomic write — never leave a half-written file at the canonical path.
    tmp_path = train_path.with_suffix(train_path.suffix + ".tmp")
    tmp_path.write_text("\n".join(kept) + "\n")
    shutil.move(str(tmp_path), train_path)
    logger.info("[phase=write] %s (%d rows)", train_path, len(kept))

    # Append provenance line. Append-mode (never overwrite v1's log).
    with log_path.open("a") as f:
        f.write(
            f"\nv2 reservation (plan v2 Step 3.6.0): training file = corpus minus rows "
            f"[{RESERVED_SLICE.start}, {RESERVED_SLICE.stop}) -> {len(kept)} rows; "
            f"em_pool (N={len(pool_h)}) hash-overlap vs training file = 0\n"
        )

    logger.info(
        "[phase=done] training file ready at %s (%d rows); em_pool N=%d, overlap=0; "
        "provenance appended to %s",
        train_path,
        len(kept),
        len(pool_h),
        log_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
