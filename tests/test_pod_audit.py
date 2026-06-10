"""Tests for ``scripts/pod_audit.py`` — the keep-running stale-audit exemption.

The daily stale-pod audit (``cron_pod_audit.sh`` → ``pod.py audit-stale
--terminate-stale --yes``) auto-terminates every EXITED pod older than
24h. The ``keep-running`` tag on the owning task is the workflow's
documented pod-preservation override (CLAUDE.md, /issue Step 8), so
``classify()`` must bucket such pods as ``kept-exited`` — reported, never
consumed by ``--terminate-stale`` — instead of ``stale``.

Near-incident pinned here (task #546, 2026-06-10): pod-546 was stopped at
an upload-quota park while the SOLE holder of 70 unuploaded LoRA
adapters; the next 09:37 audit would have destroyed it despite the task's
``keep-running`` tag.

Tests run without network: ``list_team_pods`` / ``terminate_pod`` /
``_task_has_keep_running`` / ``_scan_task_references`` are monkeypatched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_audit  # noqa: E402
from pod_audit import (  # noqa: E402
    _issue_number_from_name,
    _task_has_keep_running,
    classify,
    render_report,
)
from runpod_api import PodInfo  # noqa: E402


def _pod(name: str, status: str = "EXITED", created_at: str | None = None) -> PodInfo:
    return PodInfo(
        pod_id=f"id-{name}",
        name=name,
        desired_status=status,
        gpu_count=1,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        created_at=created_at,
    )


OLD = "2020-01-01T00:00:00Z"  # far past any --max-exited-hours threshold


@pytest.fixture(autouse=True)
def _no_task_scan(monkeypatch: pytest.MonkeyPatch):
    """Keep classify() off the real tasks/ tree (worktree-resolver-independent)."""
    monkeypatch.setattr(pod_audit, "_scan_task_references", lambda pod_id, name: [])


# ── _issue_number_from_name ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("pod-546", 546),
        ("epm-issue-546", 546),
        ("epm-issue-546-b", 546),  # legacy suffixed dispatcher pods
        ("pod-546-extra", 546),
        ("pod-abc", None),
        ("pod-", None),
        ("my-custom-pod", None),
        ("", None),
    ],
)
def test_issue_number_from_name(name: str, expected: int | None):
    assert _issue_number_from_name(name) == expected


# ── _task_has_keep_running (fail-soft tag lookup) ───────────────────────────


def test_keep_running_true_when_tagged(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        pod_audit,
        "get_task",
        lambda issue: {"frontmatter": {"tags": ["keep-running", "other"]}},
    )
    assert _task_has_keep_running(546) is True


def test_keep_running_false_without_tag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "get_task", lambda issue: {"frontmatter": {"tags": ["x"]}})
    assert _task_has_keep_running(546) is False


def test_keep_running_false_when_tags_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "get_task", lambda issue: {"frontmatter": {}})
    assert _task_has_keep_running(546) is False


def test_keep_running_fail_soft_on_lookup_error(monkeypatch: pytest.MonkeyPatch):
    def boom(issue: int):
        raise FileNotFoundError("no such task")

    monkeypatch.setattr(pod_audit, "get_task", boom)
    assert _task_has_keep_running(546) is False


# ── classify() bucketing ────────────────────────────────────────────────────


def test_old_exited_pod_with_tag_is_kept_not_stale(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: issue == 546)
    rows = classify(
        [_pod("pod-546", created_at=OLD), _pod("pod-99", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    by_name = {r.pod.name: r for r in rows}
    assert by_name["pod-546"].bucket == "kept-exited"
    assert by_name["pod-546"].kept_for_task == 546
    assert by_name["pod-99"].bucket == "stale"
    assert by_name["pod-99"].kept_for_task is None


def test_tag_exemption_applies_to_legacy_names(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: True)
    (row,) = classify(
        [_pod("epm-issue-546", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "kept-exited"


def test_unparseable_name_falls_through_to_stale(monkeypatch: pytest.MonkeyPatch):
    # Tag lookup must never be consulted for a name that doesn't parse.
    def boom(issue: int) -> bool:
        raise AssertionError("tag lookup called for unparseable name")

    monkeypatch.setattr(pod_audit, "_task_has_keep_running", boom)
    (row,) = classify(
        [_pod("my-custom-pod", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "stale"


def test_fresh_exited_unaffected_without_tag(monkeypatch: pytest.MonkeyPatch):
    import datetime as dt

    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    (row,) = classify(
        [_pod("pod-546", created_at=now)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "fresh-exited"


# ── render_report ───────────────────────────────────────────────────────────


def test_report_names_tag_and_task(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: True)
    rows = classify(
        [_pod("pod-546", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    report = render_report(rows)
    assert "kept-exited" in report
    assert "keep-running tag on task #546" in report


# ── cmd_audit / --terminate-stale ───────────────────────────────────────────


def test_terminate_stale_skips_kept_exited(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(
        pod_audit,
        "list_team_pods",
        lambda: [_pod("pod-546", created_at=OLD), _pod("pod-99", created_at=OLD)],
    )
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: issue == 546)
    terminated: list[str] = []
    monkeypatch.setattr(pod_audit, "terminate_pod", terminated.append)

    rc = pod_audit.main(["--terminate-stale", "--yes"])

    assert terminated == ["id-pod-99"]  # kept-exited pod NOT consumed
    assert rc == 2  # stale pod was found (and terminated)
    err = capsys.readouterr().err
    assert "keep-running" in err and "pod-546" in err


def test_exit_zero_when_only_kept_exited(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "list_team_pods", lambda: [_pod("pod-546", created_at=OLD)])
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: True)
    monkeypatch.setattr(
        pod_audit,
        "terminate_pod",
        lambda pod_id: (_ for _ in ()).throw(AssertionError("must not terminate kept pod")),
    )

    rc = pod_audit.main(["--terminate-stale", "--yes"])

    assert rc == 0  # a deliberately-kept pod is not an audit finding
