"""Tests for ``scripts/pod_audit.py`` — teardown eligibility, fail-toward-KEEP.

The daily stale-pod audit (``cron_pod_audit.sh`` → ``pod.py audit-stale
--notify-stale``) is REPORT-ONLY as of #2075: it classifies, reports, and
pushes one deduped terminate-RECOMMENDATION per UTC day;
``--terminate-stale`` is manual/user-approved only. An EXITED pod is
``stale`` only when its EXIT time (parsed from ``lastStatusChange`` — NOT
creation age, #2075 defect 2) is past the threshold AND the pod is
positively EPS-owned via STRUCTURED provenance (#1404/#1471/#2075 defect 1)
AND not shared-infrastructure-named (#2075 defect: fellows-cluster nodes).
The ``keep-running`` tag on the owning task is the workflow's documented
pod-preservation override (CLAUDE.md, /issue Step 8), so ``classify()``
must bucket such pods as ``kept-exited`` — reported, never consumed by
``--terminate-stale``.

Near-incidents pinned here: task #546 (2026-06-10) — pod-546 stopped at an
upload-quota park while the SOLE holder of 70 unuploaded LoRA adapters, one
09:37 audit away from destruction despite the ``keep-running`` tag; and
task #2075 (2026-08-04) — 77 teammate-owned pods destroyed over 14 days on
the team-shared account via a substring ownership scan that matched the
audit's OWN report dumps quoted into notes, a creation-age staleness clock,
and no alerting on an irreversible action.

Also covers the REPORT-ONLY flag classes (idle-gpu, stopped-on-parked-task,
running-no-port) — annotations that must never change bucketing,
``--terminate-stale`` behavior, or exit codes, and must fail SAFE (unknown
util / unknown parked-age → no flag).

Tests run without network: ``list_team_pods`` / ``terminate_pod`` /
``_task_has_keep_running`` / ``_scan_task_references`` / ``_task_context`` /
``_probe_gpu_util`` / ``_push`` are monkeypatched; the real bodies are
exercised via the ``_REAL_*`` captures below.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_audit  # noqa: E402
import pod_config  # noqa: E402
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

# The real #1404 ownership predicate, captured at import time — the autouse
# fixture below defaults `_is_eps_owned` to True (hermetic: pre-#1404 bucket
# expectations hold without touching the real REGISTRY/sidecar); the
# ownership-gate tests restore this real function and control its signals.
_REAL_IS_EPS_OWNED = pod_audit._is_eps_owned
# The real structured-provenance scanner (#2075 D2) — autouse patches the
# module name to []; the structured-scan tests drive the real body.
_REAL_SCAN = pod_audit._scan_task_references
# The real push helper (#2075 D5) — autouse patches it to a no-op True so no
# test can ever touch the live Telegram script; the push-body tests call this.
_REAL_PUSH = pod_audit._push


def _pod(
    name: str,
    status: str = "EXITED",
    created_at: str | None = None,
    last_status_change: str | None = None,
) -> PodInfo:
    return PodInfo(
        pod_id=f"id-{name}",
        name=name,
        desired_status=status,
        gpu_count=1,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        created_at=created_at,
        last_status_change=last_status_change,
    )


OLD = "2020-01-01T00:00:00Z"  # far past any --max-exited-hours threshold
# An exit far past any threshold, in the live-probed lastStatusChange shape
# (#2075). Jan 01 2020 was a Wednesday.
OLD_EXIT = "Exited by user: Wed Jan 01 2020 00:00:00 GMT+0000 (Coordinated Universal Time)"


def _exit_str(hours_ago: float) -> str:
    """A lastStatusChange string whose exit time is ``hours_ago`` hours back."""
    ts = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours_ago)
    return (
        "Exited by user: "
        + ts.strftime("%a %b %d %Y %H:%M:%S")
        + " GMT+0000 (Coordinated Universal Time)"
    )


@pytest.fixture(autouse=True)
def _no_task_scan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Keep classify() off the real tasks/ tree AND off live SSH probes
    (worktree-resolver-independent, network-free). ``_is_eps_owned`` defaults
    to True so the pre-#1404 bucket expectations hold hermetically; the
    ownership-gate tests restore ``_REAL_IS_EPS_OWNED`` explicitly. ``_push``
    defaults to a no-op True and SENTINEL_DIR to a tmp dir so no test can
    touch the live Telegram channel or ~/.eps-autonomous (#2075)."""
    monkeypatch.setattr(pod_audit, "_scan_task_references", lambda pod_id, name: [])
    monkeypatch.setattr(pod_audit, "_task_context", lambda issue: TaskContext())
    monkeypatch.setattr(pod_audit, "_probe_gpu_util", lambda pod: None)
    monkeypatch.setattr(pod_audit, "_is_eps_owned", lambda p, pod_id: True)
    monkeypatch.setattr(pod_audit, "_push", lambda msg: True)
    monkeypatch.setattr(pod_audit, "SENTINEL_DIR", tmp_path / "sentinels")


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
        [
            _pod("pod-546", created_at=OLD),
            _pod("pod-99", created_at=OLD, last_status_change=OLD_EXIT),
        ],
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
    # ownership is fixture-True here; the #1471 any-name gate is exercised
    # in the gate section below.
    def boom(issue: int) -> bool:
        raise AssertionError("tag lookup called for unparseable name")

    monkeypatch.setattr(pod_audit, "_task_has_keep_running", boom)
    (row,) = classify(
        [_pod("my-custom-pod", created_at=OLD, last_status_change=OLD_EXIT)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "stale"


def test_fresh_exited_unaffected_without_tag(monkeypatch: pytest.MonkeyPatch):

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
        lambda: [
            _pod("pod-546", created_at=OLD),
            _pod("pod-99", created_at=OLD, last_status_change=OLD_EXIT),
        ],
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
        [_pod("pod-530", status="EXITED", created_at=OLD, last_status_change=OLD_EXIT)],
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
        [_pod("pod-530", status="EXITED", created_at=OLD, last_status_change=OLD_EXIT)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
        min_parked_hours=24,
    )
    assert row.stopped_on_parked_task is False


def test_stopped_on_parked_task_not_flagged_for_live_status(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "_task_context", lambda issue: _parked_ctx(status="running"))
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    (row,) = classify(
        [_pod("pod-530", status="EXITED", created_at=OLD, last_status_change=OLD_EXIT)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.stopped_on_parked_task is False


def test_stopped_on_parked_task_not_flagged_on_unknown_age(monkeypatch: pytest.MonkeyPatch):
    # No epm:status-changed marker → parked-age unknown → fail-safe, no flag.
    monkeypatch.setattr(pod_audit, "_task_context", lambda issue: _parked_ctx(parked=None))
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    (row,) = classify(
        [_pod("pod-530", status="EXITED", created_at=OLD, last_status_change=OLD_EXIT)],
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
            _pod("pod-530", status="EXITED", created_at=OLD, last_status_change=OLD_EXIT),
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


# ── #1404 EPS-ownership gate (unmanaged-exited bucket; all names, #1471) ────
#
# The RunPod account is TEAM-SHARED: a non-EPS pod may carry ANY name, the
# managed ``pod-`` prefix included. An EXITED pod past the threshold reaches
# the auto-terminate ``stale`` bucket ONLY when positively confirmed as
# EPS-owned; otherwise it lands in the report-only ``unmanaged-exited``
# bucket and is NEVER consumed by ``--terminate-stale``. Every lookup
# failure fails toward KEEP.


def _raise_missing(issue: int):
    """get_task stand-in for an issue absent from REGISTRY (raises, never None)."""
    raise KeyError(f"task {issue} not in REGISTRY")


def _no_ownership(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Restore the REAL _is_eps_owned with all three signals missing hermetically."""
    monkeypatch.setattr(pod_audit, "_is_eps_owned", _REAL_IS_EPS_OWNED)
    monkeypatch.setattr(pod_audit, "get_task", _raise_missing)  # signal 1 miss
    # signal 2 miss: resolver honors a monkeypatched PODS_EPHEMERAL_JSON
    # (returned verbatim when it differs from the seed) — point at an absent file.
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", tmp_path / "absent.json")
    # signal 3 miss: the autouse fixture already returns [] from _scan_task_references.


def test_unmanaged_exited_never_auto_terminated(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _no_ownership(monkeypatch, tmp_path)
    (row,) = classify(
        [_pod("pod-99999", created_at=OLD, last_status_change=OLD_EXIT)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "unmanaged-exited"

    # --terminate-stale must never consume the bucket (Durability pin, #1404).
    monkeypatch.setattr(
        pod_audit,
        "list_team_pods",
        lambda: [_pod("pod-99999", created_at=OLD, last_status_change=OLD_EXIT)],
    )
    terminated: list[str] = []
    monkeypatch.setattr(pod_audit, "terminate_pod", terminated.append)
    rc = pod_audit.main(["--terminate-stale", "--yes"])
    assert terminated == []
    assert rc == 0  # not stale, not orphan — no audit-finding exit code


@pytest.mark.parametrize("failure_mode", ["registry-miss", "sidecar-corrupt", "no-task-refs"])
def test_unmanaged_exited_bucket_assigned_when_eps_ownership_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path, failure_mode: str
):
    """All three ownership-failure paths route to unmanaged-exited, never stale."""
    monkeypatch.setattr(pod_audit, "_is_eps_owned", _REAL_IS_EPS_OWNED)
    # Signal 1 always misses via a RAISING get_task (absent-REGISTRY realism).
    monkeypatch.setattr(pod_audit, "get_task", _raise_missing)
    if failure_mode == "sidecar-corrupt":
        corrupt = tmp_path / "pods_ephemeral.json"
        corrupt.write_text("{this is not json")
        monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", corrupt)
    else:
        # registry-miss / no-task-refs: sidecar absent entirely.
        monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", tmp_path / "absent.json")
    # Signal 3: autouse fixture already returns [] (the 'no-task-refs' path).
    (row,) = classify(
        [_pod("pod-99999", created_at=OLD, last_status_change=OLD_EXIT)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "unmanaged-exited"


@pytest.mark.parametrize("signal", ["registry", "ephemeral-pod-id", "ephemeral-name", "task-refs"])
def test_eps_owned_pod_still_reaches_stale_normally(
    monkeypatch: pytest.MonkeyPatch, tmp_path, signal: str
):
    """Each ownership signal alone suffices to keep the normal stale bucketing."""
    monkeypatch.setattr(pod_audit, "_is_eps_owned", _REAL_IS_EPS_OWNED)
    sidecar = tmp_path / "absent.json"  # default: signal 2 misses
    if signal == "registry":
        monkeypatch.setattr(pod_audit, "get_task", lambda issue: {"frontmatter": {}})
    else:
        monkeypatch.setattr(pod_audit, "get_task", _raise_missing)
        if signal == "ephemeral-pod-id":
            sidecar = tmp_path / "pods_ephemeral.json"
            entry = {"name": "some-other-name", "pod_id": "id-pod-99999"}
            sidecar.write_text(json.dumps({"version": 2, "pods": {"pod-99999": entry}}))
        elif signal == "ephemeral-name":
            sidecar = tmp_path / "pods_ephemeral.json"
            entry = {"name": "pod-99999", "pod_id": "zzz-unrelated"}
            sidecar.write_text(json.dumps({"version": 2, "pods": {"pod-99999": entry}}))
        elif signal == "task-refs":
            monkeypatch.setattr(pod_audit, "_scan_task_references", lambda pod_id, name: [99999])
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", sidecar)
    (row,) = classify(
        [_pod("pod-99999", created_at=OLD, last_status_change=OLD_EXIT)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "stale"


def test_non_managed_name_without_ownership_never_auto_terminated(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """#1471: the ownership gate applies to EVERY name — a non-managed EXITED
    pod without positive EPS evidence routes to unmanaged-exited, never stale,
    and --terminate-stale never consumes it (fail-toward-keep on the
    team-shared account)."""
    _no_ownership(monkeypatch, tmp_path)
    (row,) = classify(
        [_pod("my-custom-pod", created_at=OLD, last_status_change=OLD_EXIT)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "unmanaged-exited"

    # End-to-end: --terminate-stale must never consume it (Durability pin, #1471).
    monkeypatch.setattr(
        pod_audit,
        "list_team_pods",
        lambda: [_pod("my-custom-pod", created_at=OLD, last_status_change=OLD_EXIT)],
    )
    terminated: list[str] = []
    monkeypatch.setattr(pod_audit, "terminate_pod", terminated.append)
    rc = pod_audit.main(["--terminate-stale", "--yes"])
    assert terminated == []
    assert rc == 0  # not stale, not orphan — no audit-finding exit code


@pytest.mark.parametrize("signal", ["ephemeral-pod-id", "ephemeral-name", "task-refs"])
def test_non_managed_name_with_eps_evidence_still_reaches_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path, signal: str
):
    """#1471: an EPS-owned odd-named pod (dispatcher-created) still auto-reaps
    via ownership signals 2 (sidecar) and 3 (task references). Signal 1 cannot
    fire for an unparseable name by construction."""
    monkeypatch.setattr(pod_audit, "_is_eps_owned", _REAL_IS_EPS_OWNED)
    monkeypatch.setattr(pod_audit, "get_task", _raise_missing)  # signal 1 miss
    sidecar = tmp_path / "absent.json"  # default: signal 2 misses
    if signal == "ephemeral-pod-id":
        sidecar = tmp_path / "pods_ephemeral.json"
        entry = {"name": "some-other-name", "pod_id": "id-my-custom-pod"}
        sidecar.write_text(json.dumps({"version": 2, "pods": {"x": entry}}))
    elif signal == "ephemeral-name":
        sidecar = tmp_path / "pods_ephemeral.json"
        entry = {"name": "my-custom-pod", "pod_id": "zzz-unrelated"}
        sidecar.write_text(json.dumps({"version": 2, "pods": {"x": entry}}))
    elif signal == "task-refs":
        monkeypatch.setattr(pod_audit, "_scan_task_references", lambda pod_id, name: [1471])
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", sidecar)
    (row,) = classify(
        [_pod("my-custom-pod", created_at=OLD, last_status_change=OLD_EXIT)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "stale"


def test_render_report_includes_unmanaged_exited(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _no_ownership(monkeypatch, tmp_path)
    rows = classify(
        [_pod("pod-99999", created_at=OLD, last_status_change=OLD_EXIT)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    report = render_report(rows)
    assert "unmanaged-exited" in report  # summary + detail sections
    assert "NEVER auto-terminated" in report
    assert "Do NOT terminate without confirming ownership" in report
    assert "id-pod-99999" in report


def test_all_signals_raising_still_unmanaged_exited(monkeypatch: pytest.MonkeyPatch):
    """Fail-toward-keep composite: every ownership signal RAISING → unmanaged-exited."""
    monkeypatch.setattr(pod_audit, "_is_eps_owned", _REAL_IS_EPS_OWNED)

    def boom(*args, **kwargs):
        raise RuntimeError("lookup exploded")

    monkeypatch.setattr(pod_audit, "get_task", boom)  # also fail-softs keep-running
    monkeypatch.setattr(pod_config, "resolve_live_pods_ephemeral", boom)

    # Direct predicate read with ALL THREE signals raising. (classify()'s own
    # top-level refs-annotation call to _scan_task_references is a separate,
    # pre-existing seam — the predicate wraps its OWN signal-3 call.)
    monkeypatch.setattr(pod_audit, "_scan_task_references", boom)
    assert (
        _REAL_IS_EPS_OWNED(
            _pod("pod-99999", created_at=OLD, last_status_change=OLD_EXIT), "id-pod-99999"
        )
        is False
    )

    # And through classify(): signals 1+2 raising (signal 3 empty via the
    # autouse default) still routes to unmanaged-exited, never stale.
    monkeypatch.setattr(pod_audit, "_scan_task_references", lambda pod_id, name: [])
    (row,) = classify(
        [_pod("pod-99999", created_at=OLD, last_status_change=OLD_EXIT)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "unmanaged-exited"


def test_pm_triage_protocol_present():
    """Durability pin (#1404 plan risk 3): the PM ownership-triage prose exists.

    The anchor phrase asserted below PAIRS with the protocol block inserted in
    ``.claude/agents/research-pm.md`` (Mode 2 AUDIT, right after the "Orphan
    pods" bullet). Future editors: if you reword or relocate that block,
    update the agent prose AND this assertion together — the test exists so an
    accidental removal of the protocol fails loud, not to freeze the phrasing.
    """
    pm_spec = REPO_ROOT / ".claude" / "agents" / "research-pm.md"
    text = pm_spec.read_text(encoding="utf-8")
    assert "ownership triage FIRST" in text


# ── _exited_age_hours: the exit-time staleness clock (#2075 D3) ──────────────


def test_exited_age_parses_live_probe_shape():
    """The exact string shape returned by the 2026-08-10 live probe parses."""
    p = _pod(
        "pod-1",
        last_status_change=(
            "Exited by user: Thu Jul 16 2026 16:32:26 GMT+0000 (Coordinated Universal Time)"
        ),
    )
    age = pod_audit._exited_age_hours(p)
    assert age is not None and age > 24  # Jul 2026 is permanently in the past


def test_exited_age_parses_exited_by_runpod():
    p = _pod(
        "pod-1",
        last_status_change=(
            "Exited by Runpod: Thu Jul 16 2026 16:32:26 GMT+0000 (Coordinated Universal Time)"
        ),
    )
    assert pod_audit._exited_age_hours(p) is not None


def test_exited_age_measures_hours_since_exit():
    p = _pod("pod-1", last_status_change=_exit_str(30.0))
    assert pod_audit._exited_age_hours(p) == pytest.approx(30.0, abs=0.5)


def test_exited_age_none_for_running_pod():
    # Wrong desired_status: even a well-formed Exited string reads as unknown.
    p = _pod("pod-1", status="RUNNING", last_status_change=OLD_EXIT)
    assert pod_audit._exited_age_hours(p) is None


@pytest.mark.parametrize(
    "lsc",
    [
        None,  # field missing from the API response
        "",  # empty
        "garbage with no colon",
        # wrong verb — contradictory data on an EXITED pod, treated as unknown
        "Rented by User: Thu Jul 16 2026 16:32:26 GMT+0000 (Coordinated Universal Time)",
        "Exited by user: not-a-timestamp (Coordinated Universal Time)",
        "Exited by user: ",  # empty timestamp
    ],
)
def test_exit_clock_unknown_routes_fresh_exited(monkeypatch: pytest.MonkeyPatch, lsc):
    """#2075 D3 fail-toward-KEEP: unknown/unparseable exit time is NEVER stale."""
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    (row,) = classify(
        [_pod("pod-99", created_at=OLD, last_status_change=lsc)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "fresh-exited"
    assert row.exited_age_hours is None


def test_created_old_but_freshly_exited_is_fresh(monkeypatch: pytest.MonkeyPatch):
    """#2075 defect-2 regression: a pod created long ago that EXITED 1h ago is
    fresh-exited — the pre-#2075 creation-age clock made it 'stale' with a
    documented-but-nonexistent 24h grace window."""
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    (row,) = classify(
        [_pod("pod-99", created_at=OLD, last_status_change=_exit_str(1.0))],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "fresh-exited"
    assert row.exited_age_hours == pytest.approx(1.0, abs=0.5)


def test_exited_past_threshold_reaches_stale(monkeypatch: pytest.MonkeyPatch):
    """Sanity companion: an owned pod whose EXIT is 30h old IS stale at 24h."""
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    (row,) = classify(
        [_pod("pod-99", created_at=OLD, last_status_change=_exit_str(30.0))],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "stale"
    assert row.exited_age_hours == pytest.approx(30.0, abs=0.5)


# ── structured-provenance ownership scan (#2075 D2, defect 1) ────────────────


def _tasks_tree(tmp_path: Path, rows: dict[int, list[dict]]) -> Path:
    """Materialize a minimal tasks/<status>/<id>/events.jsonl tree."""
    td = tmp_path / "tasks"
    for issue, events in rows.items():
        d = td / "running" / str(issue)
        d.mkdir(parents=True)
        (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return td


def test_scan_ignores_audit_dump_in_progress_note(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """#2075 defect 1 (self-poisoning): a fleet-audit dump quoted into an
    epm:progress note is NOT ownership evidence, even when it names the pod
    with a structured-looking token."""
    td = _tasks_tree(
        tmp_path,
        {
            1738: [
                {
                    "ts": "2026-08-01T00:00:00Z",
                    "kind": "epm:progress",
                    "note": (
                        "pod audit dump — not ours:\n"
                        "  y3b0x9o15yn7ak  EXITED  'styfeng-8xH200'  pod=styfeng-8xH200"
                    ),
                }
            ]
        },
    )
    monkeypatch.setattr(pod_audit, "tasks_dir", lambda: td)
    assert _REAL_SCAN("y3b0x9o15yn7ak", "styfeng-8xH200") == []


def test_scan_ignores_mid_prose_mention_in_run_launched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Right marker kind but unstructured (mid-prose) mention → no hit."""
    td = _tasks_tree(
        tmp_path,
        {
            546: [
                {
                    "ts": "2026-08-01T00:00:00Z",
                    "kind": "epm:run-launched",
                    "note": "relaunched after teammate pod styfeng-8xH200 was investigated",
                }
            ]
        },
    )
    monkeypatch.setattr(pod_audit, "tasks_dir", lambda: td)
    assert _REAL_SCAN("id-x", "styfeng-8xH200") == []


@pytest.mark.parametrize("kind", ["epm:run-launched", "epm:pod-provisioned"])
@pytest.mark.parametrize(
    "note",
    [
        "pod=pod-546 provisioned for round 3",  # structured token
        "pod-546 (1x H100) launched — smoke first",  # leading token
        "  pod-546 launched",  # leading with whitespace
    ],
)
def test_scan_hits_structured_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str, note: str
):
    td = _tasks_tree(tmp_path, {546: [{"ts": "2026-08-01T00:00:00Z", "kind": kind, "note": note}]})
    monkeypatch.setattr(pod_audit, "tasks_dir", lambda: td)
    assert _REAL_SCAN("id-pod-546", "pod-546") == [546]


def test_scan_hits_pod_id_tokens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    td = _tasks_tree(
        tmp_path,
        {
            546: [
                {
                    "ts": "2026-08-01T00:00:00Z",
                    "kind": "epm:run-launched",
                    "note": "launched; pod_id=y3b0x9o15yn7ak host pending",
                }
            ]
        },
    )
    monkeypatch.setattr(pod_audit, "tasks_dir", lambda: td)
    assert _REAL_SCAN("y3b0x9o15yn7ak", "pod-546") == [546]


def test_scan_boundary_safe_against_suffixed_sibling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """#1961 boundary safety: pod-1768 never matches inside pod-1768-lt."""
    td = _tasks_tree(
        tmp_path,
        {
            1768: [
                {
                    "ts": "2026-08-01T00:00:00Z",
                    "kind": "epm:run-launched",
                    "note": "pod=pod-1768-lt provisioned (long-training sibling)",
                }
            ]
        },
    )
    monkeypatch.setattr(pod_audit, "tasks_dir", lambda: td)
    assert _REAL_SCAN("id-pod-1768", "pod-1768") == []


def test_scan_skips_unparseable_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    td = tmp_path / "tasks"
    d = td / "running" / "546"
    d.mkdir(parents=True)
    good = json.dumps(
        {"ts": "2026-08-01T00:00:00Z", "kind": "epm:run-launched", "note": "pod=pod-546 up"}
    )
    (d / "events.jsonl").write_text("epm:run-launched pod=pod-546 NOT JSON\n" + good + "\n")
    monkeypatch.setattr(pod_audit, "tasks_dir", lambda: td)
    assert _REAL_SCAN("id-pod-546", "pod-546") == [546]


def test_structured_grammar_matches_watcher_template():
    """Parity pin with autonomous_session_watch._latest_named_run_launched_ts
    (#1961): pod_audit replicates the 3-line grammar rather than importing the
    15k-line watcher module. Fails loud if either side's template drifts —
    update BOTH together."""
    watcher_src = (REPO_ROOT / "scripts" / "autonomous_session_watch.py").read_text(
        encoding="utf-8"
    )
    template = r'rf"(?<![\w-])pod={esc}(?![\w-])|^\s*{esc}(?![\w-])"'
    assert template in watcher_src, "watcher grammar template moved/changed — re-sync pod_audit"
    esc = re.escape("pod-1768")
    name_arm = rf"(?<![\w-])pod={esc}(?![\w-])|^\s*{esc}(?![\w-])"
    pat = pod_audit._structured_pod_ref_pattern("", "pod-1768")
    assert pat is not None and name_arm in pat.pattern


def test_structured_pattern_none_for_empty_needles():
    assert pod_audit._structured_pod_ref_pattern("", "") is None


def test_progress_dump_reference_routes_unmanaged_exited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """End-to-end #2075 defect-1 regression through the REAL predicate + REAL
    scanner: an EXITED>24h teammate pod whose ONLY events.jsonl trace is an
    audit dump in a progress note routes to unmanaged-exited, never stale."""
    monkeypatch.setattr(pod_audit, "_is_eps_owned", _REAL_IS_EPS_OWNED)
    monkeypatch.setattr(pod_audit, "_scan_task_references", _REAL_SCAN)
    monkeypatch.setattr(pod_audit, "get_task", _raise_missing)
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", tmp_path / "absent.json")
    td = _tasks_tree(
        tmp_path,
        {
            880: [
                {
                    "ts": "2026-08-01T00:00:00Z",
                    "kind": "epm:progress",
                    "note": "audit dump: styfeng-8xH200 (y3b0x9o15yn7ak) EXITED 300h — not ours",
                }
            ]
        },
    )
    monkeypatch.setattr(pod_audit, "tasks_dir", lambda: td)
    pod = PodInfo(
        pod_id="y3b0x9o15yn7ak",
        name="styfeng-8xH200",
        desired_status="EXITED",
        gpu_count=8,
        gpu_type_id="NVIDIA H200",
        created_at=OLD,
        last_status_change=OLD_EXIT,
    )
    (row,) = classify([pod], max_exited_hours=24, min_orphan_running_hours=1)
    assert row.bucket == "unmanaged-exited"
    assert row.referenced_in_tasks == []


# ── shared-infrastructure name guard (#2075 D4) ──────────────────────────────


@pytest.mark.parametrize("name", ["Anthropic 2-node-26-got", "cluster-EUR-IS-pod-6"])
def test_shared_infra_name_never_stale_even_when_owned(name: str):
    # autouse fixture forces _is_eps_owned True — EVERY ownership signal
    # fires — and the exit is far past the threshold; still never stale.
    (row,) = classify(
        [_pod(name, created_at=OLD, last_status_change=OLD_EXIT)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "unmanaged-exited"
    assert row.shared_infra is True


def test_shared_infra_env_extension(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EPM_POD_AUDIT_SHARED_NAME_PATTERNS", "styfeng-, extra")
    assert pod_audit._is_shared_infra_name("styfeng-8xH200") is True
    assert pod_audit._is_shared_infra_name("Anthropic 2-node") is True  # built-ins kept
    monkeypatch.delenv("EPM_POD_AUDIT_SHARED_NAME_PATTERNS")
    assert pod_audit._is_shared_infra_name("styfeng-8xH200") is False


def test_shared_infra_match_is_case_sensitive():
    assert pod_audit._is_shared_infra_name("anthropic 2-node") is False
    assert pod_audit._is_shared_infra_name("pod-546") is False


def test_shared_infra_running_stays_orphan_with_tag():
    (row,) = classify(
        [_pod("Anthropic 2-node-26-got", status="RUNNING", created_at=OLD)],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    assert row.bucket == "orphan-running"  # honest: it IS outside the lifecycle
    assert row.shared_infra is True
    assert "SHARED-INFRA" in render_report([row])


def test_shared_infra_never_terminated_end_to_end(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        pod_audit,
        "list_team_pods",
        lambda: [_pod("Anthropic 2-node-26-got", created_at=OLD, last_status_change=OLD_EXIT)],
    )
    terminated: list[str] = []
    monkeypatch.setattr(pod_audit, "terminate_pod", terminated.append)
    rc = pod_audit.main(["--terminate-stale", "--yes"])
    assert terminated == []
    assert rc == 0


# ── --notify-stale recommendation push + termination push (#2075 D1/D5) ─────


def _one_stale_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pod_audit,
        "list_team_pods",
        lambda: [_pod("pod-99", created_at=OLD, last_status_change=OLD_EXIT)],
    )
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)


def test_notify_stale_pushes_once_per_day(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _one_stale_pod(monkeypatch)
    monkeypatch.setattr(pod_audit, "SENTINEL_DIR", tmp_path / "s")
    pushes: list[str] = []
    monkeypatch.setattr(pod_audit, "_push", lambda msg: pushes.append(msg) or True)
    rc1 = pod_audit.main(["--notify-stale"])
    rc2 = pod_audit.main(["--notify-stale"])
    assert len(pushes) == 1  # per-UTC-day sentinel dedupe
    assert "pod-99" in pushes[0]
    # The exact approval command, WITHOUT --yes (the y/N prompt shows the list).
    assert pod_audit.APPROVAL_COMMAND in pushes[0]
    assert "EPS_ALLOW_COMPUTE_KILL=1" in pushes[0]
    assert "--yes" not in pushes[0]
    assert rc1 == rc2 == 2  # the stale finding still drives the exit code
    assert list((tmp_path / "s").glob("pod-audit-stale-notify-*"))


def test_notify_stale_sentinel_only_after_successful_push(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _one_stale_pod(monkeypatch)
    monkeypatch.setattr(pod_audit, "SENTINEL_DIR", tmp_path / "s")
    pushes: list[str] = []
    monkeypatch.setattr(pod_audit, "_push", lambda msg: pushes.append(msg) or False)  # send FAILS
    pod_audit.main(["--notify-stale"])
    assert not list((tmp_path / "s").glob("pod-audit-stale-notify-*"))
    pod_audit.main(["--notify-stale"])
    assert len(pushes) == 2  # retried on the next run instead of silently done


def test_notify_stale_never_terminates(monkeypatch: pytest.MonkeyPatch):
    _one_stale_pod(monkeypatch)
    monkeypatch.setattr(
        pod_audit,
        "terminate_pod",
        lambda pod_id: (_ for _ in ()).throw(AssertionError("--notify-stale must never terminate")),
    )
    rc = pod_audit.main(["--notify-stale"])
    assert rc == 2


def test_no_notify_push_without_flag(monkeypatch: pytest.MonkeyPatch):
    _one_stale_pod(monkeypatch)
    pushes: list[str] = []
    monkeypatch.setattr(pod_audit, "_push", lambda msg: pushes.append(msg) or True)
    pod_audit.main([])
    assert pushes == []


def test_terminate_stale_pushes_termination_summary(monkeypatch: pytest.MonkeyPatch):
    _one_stale_pod(monkeypatch)
    terminated: list[str] = []
    monkeypatch.setattr(pod_audit, "terminate_pod", terminated.append)
    pushes: list[str] = []
    monkeypatch.setattr(pod_audit, "_push", lambda msg: pushes.append(msg) or True)
    rc = pod_audit.main(["--terminate-stale", "--yes"])
    assert terminated == ["id-pod-99"]
    assert rc == 2
    assert len(pushes) == 1
    assert "1 terminated, 0 failed" in pushes[0]
    assert "pod-99" in pushes[0] and "id-pod-99" in pushes[0]


def test_terminate_stale_push_names_failures(monkeypatch: pytest.MonkeyPatch):
    _one_stale_pod(monkeypatch)
    monkeypatch.setattr(
        pod_audit,
        "terminate_pod",
        lambda pod_id: (_ for _ in ()).throw(RuntimeError("api down")),
    )
    pushes: list[str] = []
    monkeypatch.setattr(pod_audit, "_push", lambda msg: pushes.append(msg) or True)
    rc = pod_audit.main(["--terminate-stale", "--yes"])
    assert rc == 2
    assert "0 terminated, 1 failed" in pushes[0]
    assert "id-pod-99" in pushes[0]


def test_push_real_body_executes_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Production-body test for _push (code-style: one production-body test per
    seam-stubbed function): the real body resolves the env override, checks
    existence, and invokes the script via subprocess — faked only at the
    executable boundary (a real tmp script)."""
    out = tmp_path / "sent.txt"
    script = tmp_path / "push.sh"
    script.write_text(f'#!/bin/bash\necho "$1" > {out}\n')
    script.chmod(0o755)
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(script))
    assert _REAL_PUSH("hello world") is True
    assert out.read_text().strip() == "hello world"


def test_push_real_body_false_when_script_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(tmp_path / "absent.sh"))
    assert _REAL_PUSH("x") is False


def test_push_real_body_false_on_nonzero_rc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    script = tmp_path / "fail.sh"
    script.write_text("#!/bin/bash\nexit 3\n")
    script.chmod(0o755)
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(script))
    assert _REAL_PUSH("x") is False


# ── report + JSON carry both clocks (#2075) ──────────────────────────────────


def test_report_shows_both_clocks_and_approval_line(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    rows = classify(
        [
            _pod("pod-99", created_at=OLD, last_status_change=_exit_str(30.0)),
            _pod("pod-77", created_at=OLD),  # exit time unknown -> exited=?
        ],
        max_exited_hours=24,
        min_orphan_running_hours=1,
    )
    report = render_report(rows)
    assert "exited=" in report
    assert re.search(r"exited=\s*\?", report)  # unknown exit renders '?'
    assert "user approval required" in report  # stale section header
    assert pod_audit.APPROVAL_COMMAND in report


def test_json_payload_carries_new_fields(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(
        pod_audit,
        "list_team_pods",
        lambda: [_pod("pod-99", created_at=OLD, last_status_change=_exit_str(30.0))],
    )
    monkeypatch.setattr(pod_audit, "_task_has_keep_running", lambda issue: False)
    pod_audit.main(["--json"])
    (row,) = json.loads(capsys.readouterr().out)
    assert row["exited_age_hours"] == pytest.approx(30.0, abs=0.5)
    assert row["shared_infra"] is False
    assert row["bucket"] == "stale"
