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

Also covers the two REPORT-ONLY flag classes added 2026-06-10 (idle-gpu on
RUNNING managed pods, stopped-on-parked-task on EXITED pods) — annotations
that must never change bucketing, ``--terminate-stale`` behavior, or exit
codes, and must fail SAFE (unknown util / unknown parked-age → no flag).

Tests run without network: ``list_team_pods`` / ``terminate_pod`` /
``_task_has_keep_running`` / ``_scan_task_references`` / ``_task_context`` /
``_probe_gpu_util`` are monkeypatched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_audit  # noqa: E402
from pod_audit import (  # noqa: E402
    TaskContext,
    _issue_number_from_name,
    _probe_gpu_util,
    _task_context,
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
    """Keep classify() off the real tasks/ tree AND off live SSH probes
    (worktree-resolver-independent, network-free)."""
    monkeypatch.setattr(pod_audit, "_scan_task_references", lambda pod_id, name: [])
    monkeypatch.setattr(pod_audit, "_task_context", lambda issue: TaskContext())
    monkeypatch.setattr(pod_audit, "_probe_gpu_util", lambda pod: None)


# ── _issue_number_from_name ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("pod-546", 546),
        ("epm-issue-546", 546),
        ("epm-issue-546-b", 546),  # legacy suffixed dispatcher pods
        ("pod-546-extra", 546),
        ("pod-779-b", 779),  # canonical multi-pod-per-issue form (#1334)
        # DELIBERATE #1334 delta from the old split('-', 1) parse: a NUMERIC
        # slug no longer maps (letter-initial slugs only) — such a pod falls
        # through to the age-based stale logic instead of a guessy attribution.
        ("pod-779-60", None),
        ("pod-779-B", None),  # uppercase slug rejected (we only generate lowercase)
        ("pod-779-", None),  # empty slug rejected
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


# ── idle-gpu report-only flag ───────────────────────────────────────────────


def test_idle_gpu_flagged_when_all_zero(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "_probe_gpu_util", lambda pod: [0, 0, 0, 0])
    (row,) = classify(
        [_pod("pod-518", status="RUNNING", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "active"  # bucketing untouched — flag is an annotation
    assert row.idle_gpu is True
    assert row.gpu_util == [0, 0, 0, 0]


def test_idle_gpu_not_flagged_on_probe_failure(monkeypatch: pytest.MonkeyPatch):
    # util=None means UNKNOWN (SSH/parse failure) — fail-safe, never flagged.
    monkeypatch.setattr(pod_audit, "_probe_gpu_util", lambda pod: None)
    (row,) = classify(
        [_pod("pod-518", status="RUNNING", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.idle_gpu is False
    assert row.gpu_util is None


def test_idle_gpu_not_flagged_when_any_gpu_busy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "_probe_gpu_util", lambda pod: [0, 97, 0, 0])
    (row,) = classify(
        [_pod("pod-518", status="RUNNING", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.idle_gpu is False
    assert row.gpu_util == [0, 97, 0, 0]


def test_idle_gpu_probe_skipped_for_unmanaged_and_exited(monkeypatch: pytest.MonkeyPatch):
    def boom(pod) -> list[int]:
        raise AssertionError(f"probe must not run for {pod.name}")

    monkeypatch.setattr(pod_audit, "_probe_gpu_util", boom)
    rows = classify(
        [
            _pod("my-custom-pod", status="RUNNING", created_at=OLD),  # unmanaged
            _pod("pod-99", status="EXITED", created_at=OLD),  # not running
        ],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert all(r.idle_gpu is False for r in rows)


def test_probe_gpu_util_returns_none_without_ssh_endpoint():
    # No public SSH endpoint in the live-API snapshot → None, no SSH attempt.
    pod = PodInfo(pod_id="x", name="pod-1", desired_status="RUNNING")
    assert _probe_gpu_util(pod) is None


# ── stopped-on-parked-task report-only flag ─────────────────────────────────


def _parked_ctx(status: str = "awaiting_promotion", parked: float | None = 48.0) -> TaskContext:
    return TaskContext(status=status, parked_age_hours=parked, last_marker_age_hours=parked)


def test_stopped_on_parked_task_flagged(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "_task_context", lambda issue: _parked_ctx())
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    (row,) = classify(
        [_pod("pod-530", status="EXITED", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "stale"  # bucketing untouched — flag is an annotation
    assert row.stopped_on_parked_task is True
    assert row.task_status == "awaiting_promotion"


def test_stopped_on_parked_task_respects_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "_task_context", lambda issue: _parked_ctx(parked=2.0))
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    (row,) = classify(
        [_pod("pod-530", status="EXITED", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
        min_parked_hours=24,
    )
    assert row.stopped_on_parked_task is False


def test_stopped_on_parked_task_not_flagged_for_live_status(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "_task_context", lambda issue: _parked_ctx(status="running"))
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    (row,) = classify(
        [_pod("pod-530", status="EXITED", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.stopped_on_parked_task is False


def test_stopped_on_parked_task_not_flagged_on_unknown_age(monkeypatch: pytest.MonkeyPatch):
    # No epm:status-changed marker → parked-age unknown → fail-safe, no flag.
    monkeypatch.setattr(pod_audit, "_task_context", lambda issue: _parked_ctx(parked=None))
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    (row,) = classify(
        [_pod("pod-530", status="EXITED", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.stopped_on_parked_task is False


def test_kept_exited_can_still_carry_parked_flag(monkeypatch: pytest.MonkeyPatch):
    # keep-running preserves the pod; the parked flag still surfaces context.
    monkeypatch.setattr(pod_audit, "_task_context", lambda issue: _parked_ctx())
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: True)
    (row,) = classify(
        [_pod("pod-546", status="EXITED", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "kept-exited"
    assert row.stopped_on_parked_task is True


# ── _task_context (fail-soft task lookup) ───────────────────────────────────


def test_task_context_fail_soft_on_lookup_error(monkeypatch: pytest.MonkeyPatch):
    def boom(issue: int):
        raise FileNotFoundError("no such task")

    monkeypatch.setattr(pod_audit, "get_task", boom)
    assert _task_context(530) == TaskContext()


def test_task_context_reads_status_and_marker_ages(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import json as _json

    task_dir = tmp_path / "tasks" / "awaiting_promotion" / "530"
    task_dir.mkdir(parents=True)
    events = [
        {"ts": "2020-01-01T00:00:00Z", "kind": "epm:plan-approved"},
        {"ts": "2020-01-02T00:00:00Z", "kind": "epm:status-changed"},
        {"ts": "2020-01-03T00:00:00Z", "kind": "epm:progress"},
    ]
    (task_dir / "events.jsonl").write_text("\n".join(_json.dumps(e) for e in events))
    monkeypatch.setattr(
        pod_audit,
        "get_task",
        lambda issue: {"status": "awaiting_promotion", "path": "tasks/awaiting_promotion/530"},
    )
    monkeypatch.setattr(pod_audit, "repo_root", lambda: tmp_path)

    ctx = _task_context(530)

    assert ctx.status == "awaiting_promotion"
    # Both timestamps are far past: parked (status-changed) is older than the
    # last marker, and both are positive ages.
    assert ctx.parked_age_hours is not None and ctx.last_marker_age_hours is not None
    assert ctx.parked_age_hours > ctx.last_marker_age_hours > 0


def test_task_context_events_unreadable_keeps_status(monkeypatch: pytest.MonkeyPatch, tmp_path):
    # Task resolves but events.jsonl is missing → status known, ages None.
    monkeypatch.setattr(
        pod_audit,
        "get_task",
        lambda issue: {"status": "blocked", "path": "tasks/blocked/77"},
    )
    monkeypatch.setattr(pod_audit, "repo_root", lambda: tmp_path)
    ctx = _task_context(77)
    assert ctx.status == "blocked"
    assert ctx.parked_age_hours is None and ctx.last_marker_age_hours is None


# ── report rendering + exit code for the new flags ──────────────────────────


def test_report_renders_flag_sections(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "_probe_gpu_util", lambda pod: [0])
    monkeypatch.setattr(
        pod_audit,
        "_task_context",
        lambda issue: _parked_ctx() if issue == 530 else TaskContext(status="running"),
    )
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    rows = classify(
        [
            _pod("pod-518", status="RUNNING", created_at=OLD),
            _pod("pod-530", status="EXITED", created_at=OLD),
        ],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    report = render_report(rows)
    assert "idle-gpu (report-only)" in report
    assert "point sample" in report  # honest wording: not sustained idleness
    assert "stopped-on-parked-task (report-only)" in report
    assert "status=awaiting_promotion" in report


def test_flags_do_not_affect_exit_code(monkeypatch: pytest.MonkeyPatch):
    # An idle-flagged ACTIVE pod is not stale/orphan → exit 0, report-only.
    monkeypatch.setattr(
        pod_audit,
        "list_team_pods",
        lambda: [_pod("pod-518", status="RUNNING", created_at=OLD)],
    )
    monkeypatch.setattr(pod_audit, "_probe_gpu_util", lambda pod: [0, 0])
    rc = pod_audit.main([])
    assert rc == 0
