"""Phase-0 executable gates for issue #1739 (round A).

Gate 0 (store sha-pin probe), the r_B bank probe, and Gate 3 (staged-layout
probe through the consumer loader) are EXECUTABLE this round. Gates 1-2
(yield pilot / spread floor) need generation and are round-B stubs.

All Hub listings are SERVER-SIDE SCOPED (``path_in_repo=...``) — the data repo
holds ~1M files, so a bare listing / snapshot_download wedges (gotchas.md #833).
"""

from __future__ import annotations

import logging
from pathlib import Path

from explore_persona_space.experiments.issue_1739 import store_io
from explore_persona_space.experiments.issue_1739.constants import (
    HF_DATA_REPO,
    HIDDEN_DIM,
    RB_N_TRAITS,
    RB_PREFIX,
    RB_REVISION,
    STORE_PREFIX,
    STORE_REVISION,
    SUMMARY_KINDS,
)
from explore_persona_space.orchestrate import hub

logger = logging.getLogger(__name__)


def _scoped_tree(prefix: str, revision: str) -> list:
    """Materialized scoped ``list_repo_tree`` entries (path + size).

    Materialize INSIDE the retry wrapper — Hub list APIs are lazy generators,
    so the HTTP error raises at iteration time (gotchas.md #779 n50k).
    """
    from huggingface_hub import list_repo_tree

    return hub.retry_transient(
        lambda: list(
            list_repo_tree(
                HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=prefix.rstrip("/"),
                revision=revision,
                recursive=True,
            )
        ),
        what=f"list_repo_tree {HF_DATA_REPO}@{revision}:{prefix}",
    )


def gate0_store_pin_probe(*, revision: str = STORE_REVISION) -> dict:
    """Gate 0: sha-pin probe of the #1092 summary store.

    Scoped tree listing at the pinned revision; asserts >=1 file per summary
    kind stem (prefix_end / context_end / t1). Reports realized cell dirs +
    row_index sidecar count so downstream rounds ground the layout on facts.
    """
    entries = _scoped_tree(STORE_PREFIX, revision)
    files = [e.path for e in entries if getattr(e, "size", None) is not None]
    root = STORE_PREFIX.rstrip("/") + "/"
    per_kind: dict[str, int] = {}
    for kind in SUMMARY_KINDS:
        per_kind[kind] = sum(1 for f in files if f.rsplit("/", 1)[-1].startswith(f"{kind}_L"))
    missing = [k for k, n in per_kind.items() if n < 1]
    if missing:
        raise AssertionError(
            f"gate0 FAIL: no files for kind stem(s) {missing} under "
            f"{HF_DATA_REPO}@{revision}:{STORE_PREFIX} (per-kind counts {per_kind})"
        )
    cell_dirs = sorted(
        {
            f[len(root) :].split("/", 1)[0]
            for f in files
            if f.startswith(root) and "/" in f[len(root) :]
        }
    )
    n_row_index = sum(1 for f in files if f.rsplit("/", 1)[-1].startswith("row_index"))
    report = {
        "gate": "gate0_store_pin_probe",
        "repo": HF_DATA_REPO,
        "revision": revision,
        "prefix": STORE_PREFIX,
        "n_files": len(files),
        "per_kind_counts": per_kind,
        "cell_dirs": cell_dirs,
        "n_row_index_files": n_row_index,
        "verdict": "PASS",
    }
    logger.info("[gate0] PASS: %s", report)
    return report


def rb_bank_probe(*, revision: str = RB_REVISION) -> dict:
    """r_B probe: list trait ``.pt`` files at the pinned #779 revision."""
    entries = _scoped_tree(RB_PREFIX, revision)
    pt_files = sorted(
        e.path for e in entries if getattr(e, "size", None) is not None and e.path.endswith(".pt")
    )
    if not pt_files:
        raise AssertionError(
            f"rb probe FAIL: no r_B .pt files under {HF_DATA_REPO}@{revision}:{RB_PREFIX}"
        )
    if len(pt_files) != RB_N_TRAITS:
        logger.warning(
            "[rb-probe] found %d trait files (pinned expectation %d): %s",
            len(pt_files),
            RB_N_TRAITS,
            pt_files,
        )
    report = {
        "gate": "rb_bank_probe",
        "repo": HF_DATA_REPO,
        "revision": revision,
        "prefix": RB_PREFIX,
        "trait_files": pt_files,
        "n_trait_files": len(pt_files),
        "verdict": "PASS",
    }
    logger.info("[rb-probe] PASS: %s", report)
    return report


def gate3_staged_layout_probe(local_dir: Path | str, *, revision: str = STORE_REVISION) -> dict:
    """Gate 3: staged-layout probe — stage the SMALLEST row_index + summary
    files and open them through the CONSUMER loaders (artifact-reuse check
    (h)(iv): the staged tree must open via the consumer's own reader).
    """
    import numpy as np

    entries = _scoped_tree(STORE_PREFIX, revision)
    sized = [(e.path, e.size) for e in entries if getattr(e, "size", None) is not None]
    row_index_files = sorted(
        (s, p) for p, s in sized if p.rsplit("/", 1)[-1].startswith("row_index")
    )
    summary_files = sorted(
        (s, p)
        for p, s in sized
        if p.endswith(".npy")
        and any(p.rsplit("/", 1)[-1].startswith(f"{k}_L") for k in SUMMARY_KINDS)
    )
    if not row_index_files or not summary_files:
        raise AssertionError(
            f"gate3 FAIL: row_index files={len(row_index_files)} summary files="
            f"{len(summary_files)} under {HF_DATA_REPO}@{revision}:{STORE_PREFIX}"
        )
    local_dir = Path(local_dir)
    root = STORE_PREFIX.rstrip("/") + "/"

    def _stage(repo_path: str) -> Path:
        rel = repo_path[len(root) :] if repo_path.startswith(root) else repo_path
        return hub.stage_hub_file(
            HF_DATA_REPO, repo_path, local_dir / rel, repo_type="dataset", revision=revision
        )

    ri_size, ri_path = row_index_files[0]
    su_size, su_path = summary_files[0]
    ri_local = _stage(ri_path)
    su_local = _stage(su_path)

    # Consumer-loader opens (store_io._iter_jsonl / np.load — the same readers
    # load_summaries uses), not ad-hoc parsing.
    rows = store_io._iter_jsonl(ri_local)
    if not rows or not all(isinstance(r, dict) for r in rows):
        raise AssertionError(f"gate3 FAIL: row_index {ri_path} unparseable via consumer loader")
    n_with_eval_key = sum(1 for r in rows if "is_eval_only" in r)
    arr = np.load(su_local)
    if arr.ndim != 2 or arr.shape[1] != HIDDEN_DIM:
        raise AssertionError(
            f"gate3 FAIL: summary {su_path} shape {arr.shape} != (n, {HIDDEN_DIM})"
        )
    report = {
        "gate": "gate3_staged_layout_probe",
        "repo": HF_DATA_REPO,
        "revision": revision,
        "row_index_file": {"path": ri_path, "size": ri_size, "n_rows": len(rows)},
        "row_index_first_row_keys": sorted(rows[0].keys()),
        "n_rows_with_is_eval_only_key": n_with_eval_key,
        "summary_file": {
            "path": su_path,
            "size": su_size,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        },
        "verdict": "PASS",
    }
    logger.info("[gate3] PASS: %s", report)
    return report


def gate1_yield_pilot(*_args, **_kwargs):
    """ROUND-B STUB: the yield pilot needs on-policy generation + judging."""
    raise NotImplementedError(
        "Gate 1 (yield pilot) requires generation + judging — wired in round B "
        "with the generation/capture phases."
    )


def gate2_spread_floor(*_args, **_kwargs):
    """ROUND-B STUB: the spread floor needs judged DV draws over generations."""
    raise NotImplementedError(
        "Gate 2 (spread floor) requires generation + judged DV draws — wired in "
        "round B with the generation/judge phases."
    )
