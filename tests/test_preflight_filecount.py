# ruff: noqa: E402
"""WARN-only preflight surfacing of the HF file-count sentinel (#2304).

``check_hf_filecount_sentinel`` reads the observed-count sentinel JSONL the
reactive file-count fallback writes (``hub._filecount_sentinel_path()``) and
adds ONE warning per (repo_id, repo_type) whose LAST row is ``blocked`` — a
structurally WARN-only check: it only ever calls ``report.add_warning``, so
``report.ok`` can never flip. These tests execute the REAL check body (the
#906 one-production-body rule); the only test seam is the sentinel PATH,
threaded through the production env override ``EPM_HF_FILECOUNT_SENTINEL_PATH``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import pytest

from explore_persona_space.orchestrate.preflight import (
    PreflightReport,
    check_hf_filecount_sentinel,
)

DATA_REPO = "superkaiba1/explore-persona-space-data"


@pytest.fixture
def sentinel(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "filecount.jsonl"
    monkeypatch.setenv("EPM_HF_FILECOUNT_SENTINEL_PATH", str(path))
    return path


def _row(
    repo_id: str,
    repo_type: str,
    status: str,
    *,
    observed: int | None = None,
    limit: int | None = None,
    ts: float | None = None,
) -> str:
    return json.dumps(
        {
            "ts": time.time() if ts is None else ts,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "observed_files": observed,
            "limit": limit,
            "status": status,
        }
    )


def test_check_hf_filecount_sentinel_warns(sentinel):
    sentinel.write_text(
        _row(
            DATA_REPO,
            "dataset",
            "blocked",
            observed=1_000_009,
            limit=1_000_000,
            ts=time.time() - 120,
        )
        + "\n"
    )
    report = PreflightReport()

    check_hf_filecount_sentinel(report)

    assert report.ok is True  # WARN-only by construction
    assert report.errors == []
    assert len(report.warnings) == 1, report.warnings
    w = report.warnings[0]
    assert f"HF file-count: {DATA_REPO} (dataset)" in w
    assert "observed at 1,000,009/1,000,000 files" in w
    assert "2m ago" in w
    assert "reroute to the private overflow repo (#2304)" in w


def test_no_sentinel_file_is_a_silent_no_op(sentinel):
    report = PreflightReport()
    check_hf_filecount_sentinel(report)
    assert report.ok is True and report.warnings == [] and report.errors == []


def test_recovered_repo_does_not_warn_but_still_blocked_sibling_does(sentinel):
    lines = [
        _row(DATA_REPO, "dataset", "blocked", observed=1_000_009, limit=1_000_000),
        _row(DATA_REPO, "dataset", "accepting"),  # blocked→accepting: recovered
        _row(
            "superkaiba1/explore-persona-space", "model", "blocked", observed=100_050, limit=100_000
        ),
    ]
    sentinel.write_text("\n".join(lines) + "\n")
    report = PreflightReport()

    check_hf_filecount_sentinel(report)

    assert report.ok is True
    assert len(report.warnings) == 1, report.warnings
    assert "superkaiba1/explore-persona-space (model)" in report.warnings[0]
    assert DATA_REPO not in report.warnings[0]


def test_unparseable_counts_render_unknown(sentinel):
    sentinel.write_text(_row(DATA_REPO, "dataset", "blocked") + "\n")  # None counts
    report = PreflightReport()

    check_hf_filecount_sentinel(report)

    assert len(report.warnings) == 1
    assert "observed at unknown/unknown files" in report.warnings[0]


def test_read_failure_degrades_to_single_warning(sentinel):
    sentinel.mkdir()  # path exists but open() raises IsADirectoryError
    report = PreflightReport()

    check_hf_filecount_sentinel(report)

    assert report.ok is True  # a broken sentinel can never block a launch
    assert len(report.warnings) == 1
    assert "sentinel unread" in report.warnings[0]
