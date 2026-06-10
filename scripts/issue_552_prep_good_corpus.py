#!/usr/bin/env python3
"""#552 Step 1 — pull the Turner GOOD-medical control corpus + apply the parent's slice protocol.

Mirror of ``scripts/issue_521_prep_turner_corpus.py`` (helpers imported from it,
so the two preps cannot drift), pointed at the matched GOOD corpus
``superkaiba1/explore-persona-space-data@main:issue376_em/v1/good_medical_advice_6k.jsonl``
and writing ``data/issue404/turner_good_medical_advice.jsonl`` (the path the
``issue404_pair_turner_good_medical`` condition YAML expects).

Protocol is the parent's, verbatim (plan §4.0): drop rows [200, 300) (the EM
steering-pool reservation slice), then drop any residual row whose user-prompt
hash is in ``eval_results/issue_521/inputs/em_pool.json``. The good and bad
corpora share user prompts at identical indices (verified at plan time), so
the SAME protocol applies and the result is row-count parity with the parent.

Asserts (all fail-loud):
  1. Source corpus has >= 6000 rows.
  2. Exactly KNOWN_RESIDUAL_POOL_DUPS (=1, the row-3615 dup) residual rows dropped.
  3. Post-filter training file has EXACTLY ``EXPECTED_KEPT_ROWS`` = 5899 rows
     (verified parity with the parent's EM-arm training file; plan §4.0).
  4. Zero user-prompt-hash overlap vs em_pool.json.
  5. (unless --no-upload) the training mix landed on the HF data repo,
     verified via ``huggingface_hub.list_repo_files`` (never the `hf` CLI).

Run::

    uv run python scripts/issue_552_prep_good_corpus.py              # prep + upload
    uv run python scripts/issue_552_prep_good_corpus.py --no-upload  # local smoke
    uv run python scripts/issue_552_prep_good_corpus.py --check-only

Exit codes:
  0  prep succeeded (or already done)
  2  overlap assert fired (em_pool_leak_into_training)
  3  source corpus too small
  5  row-count parity assert fired (corpus drift vs the verified 5,899)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# Sibling-script import (same convention as issue404_outcome_eval.py ->
# issue404_common): reuse the parent's hash/filter/assert helpers verbatim
# so the good-corpus prep cannot drift from the bad-corpus prep.
from issue_521_prep_turner_corpus import (
    MIN_SRC_ROWS,
    N_RESERVED,
    RESERVED_SLICE,
    _assert_no_overlap,
    _drop_pool_dup_rows,
    _pool_hashes,
)

logger = logging.getLogger(__name__)

HF_DATASET_REPO = "superkaiba1/explore-persona-space-data"
HF_FILENAME = "issue376_em/v1/good_medical_advice_6k.jsonl"
HF_REVISION = "main"
LOCAL_TRAIN_PATH = Path("data/issue404/turner_good_medical_advice.jsonl")
EM_POOL_PATH = Path("eval_results/issue_521/inputs/em_pool.json")
DISJOINTNESS_LOG = Path("eval_results/issue_552/inputs/em_pool_disjointness.txt")

# Verified at plan time (plan §4.0 / §12 assumption 2): the good corpus shares
# user prompts with the bad corpus at identical indices, including the
# row-3615 = row-215 internal duplicate, so the parent's protocol yields
# EXACTLY this row count. Any other count = corpus drift -> fail loud.
KNOWN_RESIDUAL_POOL_DUPS = 1
EXPECTED_KEPT_ROWS = 5899

# Upload destination (plan §14 row 1): datasets upload before training so any
# pod can re-fetch the exact training mix without scp.
HF_UPLOAD_PATH = (
    "issue552_benign_control/training_mix/turner_good_medical_advice_minus_pool_slice.jsonl"
)


def _upload_training_mix(train_path: Path) -> None:
    """Upload the filtered training mix to the HF data repo + verify it landed.

    Fail-loud: raises on upload error and on a post-upload listing miss
    (verification via ``list_repo_files`` per .claude/rules/upload-policy.md —
    the `hf` CLI's silent-empty failure mode is the documented trap).
    """
    from huggingface_hub import list_repo_files, upload_file

    logger.info(
        "[phase=upload] uploading %s -> %s::%s",
        train_path,
        HF_DATASET_REPO,
        HF_UPLOAD_PATH,
    )
    upload_file(
        path_or_fileobj=str(train_path),
        path_in_repo=HF_UPLOAD_PATH,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message="#552: benign-control training mix (good_medical minus pool slice)",
    )
    files = list_repo_files(HF_DATASET_REPO, repo_type="dataset", revision="main")
    if HF_UPLOAD_PATH not in files:
        raise RuntimeError(
            f"training-mix upload verification FAILED: {HF_UPLOAD_PATH!r} not in "
            f"list_repo_files({HF_DATASET_REPO!r}) after upload_file returned. "
            f"Refusing to proceed to training with an unverified dataset upload."
        )
    logger.info("[phase=upload] verified on HF: %s", HF_UPLOAD_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull Turner GOOD-medical control corpus + apply the parent slice protocol.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip download/write/upload; only run the asserts on the existing file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download + re-filter even if the file already has the expected rows.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the HF training-mix upload (local smoke only — production keeps the upload).",
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
            f"em_pool.json missing at {em_pool_path}; the parent #521 inputs must be "
            f"in git (eval_results/issue_521/inputs/) — pull main."
        )

    if args.check_only:
        if not train_path.exists():
            raise RuntimeError(f"--check-only but {train_path} doesn't exist.")
        kept = train_path.read_text().splitlines()
        if len(kept) != EXPECTED_KEPT_ROWS:
            logger.error(
                "[check_only] row-count parity FAIL: %d rows != expected %d",
                len(kept),
                EXPECTED_KEPT_ROWS,
            )
            return 5
        pool_h = _pool_hashes(em_pool_path)
        _assert_no_overlap(kept_rows=kept, pool_hashes=pool_h)
        logger.info(
            "[check_only] PASS: %s has %d rows, zero overlap vs em_pool (N=%d)",
            train_path,
            len(kept),
            len(pool_h),
        )
        return 0

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
        raise RuntimeError(
            f"hf_hub_download failed for {HF_DATASET_REPO}@{HF_REVISION}::{HF_FILENAME}: "
            f"{type(e).__name__}: {e}"
        ) from e

    rows = Path(src).read_text().splitlines()
    n_src = len(rows)
    logger.info("[phase=download] HF cache copy at %s (%d rows)", src, n_src)
    if n_src < MIN_SRC_ROWS:
        logger.error("source corpus too small: %d rows (need >=%d)", n_src, MIN_SRC_ROWS)
        return 3

    # Idempotence short-circuit: exact-parity check (stricter than the
    # parent's range check — the expected count is now a verified constant).
    if (not args.force) and train_path.exists():
        existing = train_path.read_text().splitlines()
        if len(existing) == EXPECTED_KEPT_ROWS:
            pool_h = _pool_hashes(em_pool_path)
            _assert_no_overlap(kept_rows=existing, pool_hashes=pool_h)
            logger.info(
                "[skip] %s already at the expected %d rows; overlap assert re-PASSed",
                train_path,
                EXPECTED_KEPT_ROWS,
            )
            if not args.no_upload:
                _upload_training_mix(train_path)
            return 0

    # Reserve rows [200, 300) — the parent's exact slice protocol.
    kept = rows[: RESERVED_SLICE.start] + rows[RESERVED_SLICE.stop :]
    logger.info(
        "[phase=filter] kept %d rows (n_src=%d minus reserved slice [%d, %d))",
        len(kept),
        n_src,
        RESERVED_SLICE.start,
        RESERVED_SLICE.stop,
    )

    pool_h = _pool_hashes(em_pool_path)
    kept, n_pool_dups = _drop_pool_dup_rows(kept_rows=kept, pool_hashes=pool_h)
    if n_pool_dups != KNOWN_RESIDUAL_POOL_DUPS:
        raise RuntimeError(
            f"pool-duplicate drop count {n_pool_dups} != the verified "
            f"KNOWN_RESIDUAL_POOL_DUPS={KNOWN_RESIDUAL_POOL_DUPS} (the row-3615 dup, "
            f"verified present in the good corpus at plan time). The "
            f"good_medical_advice_6k corpus has drifted on the Hub; the row-count "
            f"parity contract with the parent is broken — halt and re-verify."
        )
    try:
        _assert_no_overlap(kept_rows=kept, pool_hashes=pool_h)
    except RuntimeError as e:
        logger.error("[overlap_assert_FAIL] %s", e)
        return 2

    # Hard row-count parity assert (plan §4.1 new-file 2): the single-variable
    # contract with the parent's EM arm requires EXACTLY 5,899 training rows.
    if len(kept) != EXPECTED_KEPT_ROWS:
        logger.error(
            "[parity_assert_FAIL] post-filter row count %d != expected %d "
            "(n_src=%d, reserved=%d, pool_dups=%d). Corpus drift — halt.",
            len(kept),
            EXPECTED_KEPT_ROWS,
            n_src,
            N_RESERVED,
            n_pool_dups,
        )
        return 5

    # Atomic write.
    tmp_path = train_path.with_suffix(train_path.suffix + ".tmp")
    tmp_path.write_text("\n".join(kept) + "\n")
    shutil.move(str(tmp_path), train_path)
    logger.info("[phase=write] %s (%d rows)", train_path, len(kept))

    # Provenance line (own #552 log file; never touches the parent's).
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(
            f"#552 prep (plan §4.2 Step 1): training file = good corpus minus rows "
            f"[{RESERVED_SLICE.start}, {RESERVED_SLICE.stop}) minus {n_pool_dups} residual "
            f"pool dup -> {len(kept)} rows (parity-asserted == {EXPECTED_KEPT_ROWS}); "
            f"em_pool (N={len(pool_h)}) hash-overlap vs training file = 0\n"
        )

    if not args.no_upload:
        _upload_training_mix(train_path)

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
