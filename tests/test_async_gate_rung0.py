"""Mission-control rung 0 (async gate mode) — cross-surface tests.

Covers, per CONTRACTS §1.1 rows 4/7, §1.3 T1/W1/W3, §2/§2.2:

- ``task_workflow`` predicates: ``open_async_ask`` (APPEND-ORDER open/closed
  semantics; gate filter), ``ask_gate``, ``newest_session_mode``.
- CAS answers: ``promote --if-body-sha`` and ``set-status --if-plan-v`` —
  match mutates, mismatch raises typed errors with NOTHING mutated, absent =
  legacy (checked inside the flock, never a pre-check).
- ``task.py`` plan gate: an async session (BOTH env vars) NEVER
  self-approves — ``parked_asked`` regardless of estimate; with
  ``EPM_ASYNC_SESSION`` unset the decision table is byte-identical legacy.
- ``tick_triage`` T1: a fresh open ``epm:ask`` at an ISSUE_PARK status is
  HEALTHY (``async-parked``), never STALE-REDRIVE; ``open_ask`` defaulted
  False reproduces the legacy verdict table.
- Watcher: ``_resolve_session_mode`` durability chain (registry entry >
  marker-after-registry-GC > legacy auto); all SIX spawn-issue argv builders
  append ``--session-mode async`` iff resolved async (and stay byte-legacy
  otherwise); the W1 async plan-approval push message; the W3
  stale-async-park alert arm (age floor, dedup key, env override).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import task as task_py  # noqa: E402  (scripts/task.py)
import tick_triage  # noqa: E402

import explore_persona_space.task_workflow as tw_mod  # noqa: E402
from explore_persona_space.task_workflow import (  # noqa: E402
    ASK_KIND,
    BodyShaMismatch,
    PlanVersionMismatch,
    ask_gate,
    newest_session_mode,
    open_async_ask,
)

NOW = time.time()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ask(gate: str = "plan_approval", age_s: float = 60.0) -> dict:
    return {
        "kind": ASK_KIND,
        "ts": _iso(NOW - age_s),
        "by": "autonomous-gate",
        "note": json.dumps({"gate": gate, "issue": 1}),
    }


def _row(kind: str, age_s: float = 30.0, note: str = "") -> dict:
    return {"kind": kind, "ts": _iso(NOW - age_s), "note": note}


# ─── task_workflow predicates ────────────────────────────────────────────────


def test_open_ask_returned_and_gate_filter():
    events = [_row("epm:status-changed", 120.0), _ask("plan_approval", 60.0)]
    got = open_async_ask(events)
    assert got is not None and got["kind"] == ASK_KIND
    assert open_async_ask(events, gate="plan_approval") is not None
    assert open_async_ask(events, gate="promotion") is None
    assert ask_gate(got) == "plan_approval"


def test_ask_closed_by_later_answer_or_status_change_append_order():
    """Closing is APPEND ORDER, not timestamps — the park sequence writes
    status-changed then ask within the same second (ISO ties)."""
    same_ts = _iso(NOW - 60.0)
    # status-changed appended BEFORE the ask (the park sequence): still open.
    open_events = [
        {"kind": "epm:status-changed", "ts": same_ts, "note": ""},
        {"kind": ASK_KIND, "ts": same_ts, "note": json.dumps({"gate": "plan_approval"})},
    ]
    assert open_async_ask(open_events) is not None
    # An answer appended AFTER closes it.
    assert open_async_ask([*open_events, _row("epm:ask-answered", 10.0)]) is None
    # A status change appended AFTER closes it too (the user acted).
    assert open_async_ask([*open_events, _row("epm:status-changed", 10.0)]) is None
    # No ask at all.
    assert open_async_ask([_row("epm:progress", 5.0)]) is None


def test_ask_gate_fail_soft_on_non_json_note():
    assert ask_gate({"kind": ASK_KIND, "note": "not json"}) is None
    assert ask_gate(None) is None


def test_newest_session_mode_newest_wins_and_skips_garbage():
    events = [
        {"kind": "epm:session-mode", "note": json.dumps({"mode": "async"})},
        {"kind": "epm:session-mode", "note": "garbled"},
        {"kind": "epm:session-mode", "note": json.dumps({"mode": "bogus"})},
    ]
    assert newest_session_mode(events) == "async"
    events.append({"kind": "epm:session-mode", "note": json.dumps({"mode": "auto"})})
    assert newest_session_mode(events) == "auto"
    assert newest_session_mode([]) is None


# ─── CAS answers (fake repo) ─────────────────────────────────────────────────


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Local copy of the canonical test_task_workflow.py fake_repo fixture
    (git init + resolver rebinds; cross-test-module fixture imports are not
    resolvable under this pytest import mode)."""
    import subprocess

    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)
    tw = tw_mod
    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw, "LOCK_PATH", lock_dir / "lock")
    monkeypatch.setattr(tw, "DEFERRED_COMMITS_LOG", lock_dir / "deferred-commits.jsonl")
    monkeypatch.setattr(tw, "STRANDED_COMMITS_LOG", lock_dir / "stranded-commits.jsonl")
    return tmp_path, tw


def _mk_task(tw, status: str) -> int:
    return tw.create_task(tw.NewTaskRequest(kind="experiment", title="rung0 CAS", status=status))


def test_promote_if_body_sha_match_mismatch_absent(fake_repo):
    _repo, tw = fake_repo
    tid = _mk_task(tw, "awaiting_promotion")
    body_path = tw.find_task_path(tid) / "body.md"
    good = hashlib.sha256(body_path.read_bytes()).hexdigest()[:12]

    # Mismatch: typed error, NOTHING mutated (status + classification intact).
    with pytest.raises(BodyShaMismatch):
        tw.promote(tid, "useful", if_body_sha="0" * 12)
    task = tw.get_task(tid)
    assert task["status"] == "awaiting_promotion"
    assert task["frontmatter"].get("classification") in (None, "pending")

    # Match: promotes (full-length sha accepted, prefix-compared).
    full = hashlib.sha256(body_path.read_bytes()).hexdigest()
    assert full[:12] == good
    tw.promote(tid, "useful", if_body_sha=full)
    assert tw.get_task(tid)["status"] == "completed"

    # Absent = legacy: a second task promotes with no CAS argument.
    tid2 = _mk_task(tw, "awaiting_promotion")
    tw.promote(tid2, "not-useful")
    assert tw.get_task(tid2)["status"] == "completed"


def test_set_status_if_plan_v_match_mismatch_absent(fake_repo):
    _repo, tw = fake_repo
    tid = _mk_task(tw, "plan_pending")
    plans = tw.find_task_path(tid) / "plans"
    plans.mkdir(exist_ok=True)
    (plans / "v1.md").write_text("plan v1")
    (plans / "v2.md").write_text("plan v2")

    # Mismatch (a newer plan version landed since the view): typed error,
    # status unmoved.
    with pytest.raises(PlanVersionMismatch) as exc:
        tw.set_status(tid, "approved", if_plan_v=1)
    assert "v2" in str(exc.value)
    assert tw.get_task(tid)["status"] == "plan_pending"

    # Match: transitions.
    tw.set_status(tid, "approved", if_plan_v=2)
    assert tw.get_task(tid)["status"] == "approved"

    # Absent = legacy transition.
    tid2 = _mk_task(tw, "plan_pending")
    tw.set_status(tid2, "approved")
    assert tw.get_task(tid2)["status"] == "approved"


# ─── task.py plan gate: async never self-approves ────────────────────────────


@pytest.mark.parametrize("gpu_hours", [5.0, None])
def test_async_session_always_parks_asked(monkeypatch, gpu_hours):
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_ASYNC_SESSION", "1")
    decision, _cap, autonomous = task_py._resolve_autonomous_plan_gate(gpu_hours)
    assert decision == "parked_asked" and autonomous is True


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "FALSE", "No"])
def test_async_falsy_values_stay_legacy(monkeypatch, falsy):
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_ASYNC_SESSION", falsy)
    assert task_py._resolve_autonomous_plan_gate(5.0)[0] == "auto_approved"


def test_legacy_gate_table_unchanged_without_async_env(monkeypatch):
    """No-flags regression: EPM_ASYNC_SESSION absent -> the pre-rung-0 table
    verbatim (auto_approved / parked_no_estimate / interactive_pending)."""
    monkeypatch.delenv("EPM_ASYNC_SESSION", raising=False)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    assert task_py._resolve_autonomous_plan_gate(999.0)[0] == "auto_approved"
    assert task_py._resolve_autonomous_plan_gate(None)[0] == "parked_no_estimate"
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    assert task_py._resolve_autonomous_plan_gate(5.0)[0] == "interactive_pending"


def test_async_env_without_autonomous_is_interactive(monkeypatch):
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    monkeypatch.setenv("EPM_ASYNC_SESSION", "1")
    assert task_py._resolve_autonomous_plan_gate(5.0)[0] == "interactive_pending"


def test_post_plan_approval_ask_payload_and_idempotence(monkeypatch):
    posts: list[dict] = []
    monkeypatch.setattr(task_py, "list_events", lambda issue: [])
    monkeypatch.setattr(task_py, "highest_plan_version", lambda issue: 3)
    monkeypatch.setattr(
        task_py, "post_event", lambda issue, kind, **kw: posts.append({"kind": kind, **kw})
    )
    task_py._post_plan_approval_ask(42, 7.5)
    assert len(posts) == 1 and posts[0]["kind"] == "epm:ask"
    note = json.loads(posts[0]["note"])
    assert note["gate"] == "plan_approval" and note["gate_id"] == 4
    assert note["plan_v"] == 3 and note["plan_path"] == "plans/v3.md"
    assert note["est_gpu_hours"] == 7.5
    assert "set-status 42 approved --if-plan-v 3" in note["answer"]
    # Idempotence: an OPEN plan-approval ask suppresses a re-post.
    monkeypatch.setattr(task_py, "list_events", lambda issue: [_ask()])
    task_py._post_plan_approval_ask(42, 7.5)
    assert len(posts) == 1


@pytest.mark.parametrize(
    ("gpu_hours", "expected_decision", "expected_marker"),
    [
        (5.0, "auto_approved", "epm:plan-approved"),
        (None, "parked_no_estimate", "epm:awaiting-spend-approval"),
    ],
)
def test_followups_running_hold_suppresses_async_ask(
    monkeypatch, capsys, gpu_hours, expected_decision, expected_marker
):
    """MAJOR-1 (rung-0 implementation review): an async session's plan-gate
    call on a `followups_running` task must NOT post the plan_approval
    `epm:ask` — the ask's recorded answer (`set-status <N> approved`) is a
    FOLLOWUP_HELD_BLOCKED_STATUSES transition the hold itself refuses, and
    followups_running is watcher-ACTIVE (a park-and-EXIT there churns
    crash-recovery respawns). Mid-active-round gates are rung 2 by design
    (CONTRACTS §2.1 row 4): the hold falls through to the exact legacy
    held-round behavior — epm:plan-approved with an estimate,
    epm:awaiting-spend-approval without — status unmoved either way."""
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_ASYNC_SESSION", "1")
    posted: list[str] = []
    monkeypatch.setattr(
        task_py, "get_task", lambda number: {"status": "followups_running", "frontmatter": {}}
    )
    monkeypatch.setattr(task_py, "post_event", lambda number, kind, **kw: posted.append(kind))
    monkeypatch.setattr(
        task_py, "set_status", lambda *a, **kw: pytest.fail("hold must not move status")
    )
    # If the pre-fix ask path ran, _post_plan_approval_ask would consult
    # these (and post epm:ask) — mock them so the regression manifests as
    # the ask marker, not an unrelated lookup error.
    monkeypatch.setattr(task_py, "list_events", lambda issue: [])
    monkeypatch.setattr(task_py, "highest_plan_version", lambda issue: 1)

    ns = argparse.Namespace(
        number=99,
        status="plan_pending",
        note=None,
        auto_approve_if_autonomous=True,
        gpu_hours=gpu_hours,
    )
    task_py.cmd_set_status(ns)

    assert "epm:ask" not in posted
    assert posted == [expected_marker]
    out = capsys.readouterr().out
    assert f"PLAN_GATE_DECISION: {expected_decision}" in out
    assert "(followups_running hold: status unchanged)" in out


# ─── tick_triage T1 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("status", sorted(tick_triage.ISSUE_PARK))
def test_open_ask_park_is_healthy_async_parked(status):
    verdict, reason, streak = tick_triage.compute_issue_verdict(
        status, status, 999999.0, False, stale_after_s=1500, open_ask=True
    )
    assert verdict == "HEALTHY" and "async-parked" in reason and streak == 0


def test_open_ask_never_changes_active_statuses():
    for status in sorted(tick_triage.ISSUE_ACTIVE):
        base = tick_triage.compute_issue_verdict(status, status, 60.0, False, stale_after_s=1500)
        with_ask = tick_triage.compute_issue_verdict(
            status, status, 60.0, False, stale_after_s=1500, open_ask=True
        )
        assert with_ask == base


def test_plan_pending_over_cap_arm_untouched_by_open_ask():
    """The over-cap gate arm checks FIRST — an open ask cannot mask it."""
    with_ask = tick_triage.compute_issue_verdict(
        "plan_pending", "running", 60.0, True, stale_after_s=1500, open_ask=True
    )
    without = tick_triage.compute_issue_verdict(
        "plan_pending", "running", 60.0, True, stale_after_s=1500
    )
    assert with_ask == without


@pytest.mark.parametrize("status", sorted(tick_triage.ISSUE_ACTIVE | tick_triage.ISSUE_PARK))
@pytest.mark.parametrize("age", [60.0, 3600.0])
def test_no_flags_verdicts_byte_identical(status, age):
    """open_ask omitted == open_ask=False across the status x freshness grid
    (the legacy verdict table is preserved by construction)."""
    legacy = tick_triage.compute_issue_verdict(status, status, age, False, stale_after_s=1500)
    explicit = tick_triage.compute_issue_verdict(
        status, status, age, False, stale_after_s=1500, open_ask=False
    )
    assert explicit == legacy


# ─── watcher: mode resolution + the six builders + W1/W3 ────────────────────

import autonomous_session_watch as asw  # noqa: E402


def test_watcher_resolve_session_mode_chain(monkeypatch, tmp_path):
    reg = tmp_path / "reg"
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", reg)
    # The marker rung is IN-PROCESS (`_marker_session_mode`) by contract: a
    # subprocess probe would break the builders' argv/dry-run pins.
    monkeypatch.setattr(asw, "_marker_session_mode", lambda issue: None)
    # Neither signal -> legacy auto.
    assert asw._resolve_session_mode(1) == "auto"
    # Entry field wins (passed OR read from disk).
    assert asw._resolve_session_mode(1, {"session_mode": "async"}) == "async"
    reg.mkdir(parents=True)
    (reg / "issue-1.json").write_text(json.dumps({"session_mode": "async"}))
    assert asw._resolve_session_mode(1) == "async"
    # CRITICAL: registry entry DELETED (terminal GC) but the durable marker
    # survives -> still async.
    (reg / "issue-1.json").unlink()
    monkeypatch.setattr(asw, "_marker_session_mode", lambda issue: "async")
    assert asw._resolve_session_mode(1) == "async"
    # Garbled registry entry falls through to the marker.
    reg2 = reg / "issue-2.json"
    reg2.write_text("{not json")
    assert asw._resolve_session_mode(2) == "async"
    # Entry field beats a disagreeing marker (CONTRACTS §2.2 order).
    assert asw._resolve_session_mode(3, {"session_mode": "auto"}) == "auto"


_BUILDERS = [
    ("completed-unmerged", lambda: asw._completed_unmerged_respawn(77, dry_run=True)),
    ("respawn", lambda: asw._respawn({"issue": 77}, dry_run=True)),
    ("stalled", lambda: asw._respawn_stalled_session(77, 100.0, dry_run=True)),
    ("orphan", lambda: asw._respawn_orphan(77, 100.0, dry_run=True)),
    ("infra-drain", lambda: asw._dispatch_infra_drain(77, "slot 1/5", True)),
    ("capacity-retry", lambda: asw._redrive_capacity_retry(77, dry_run=True)),
]


@pytest.fixture
def builder_seams(monkeypatch):
    monkeypatch.setattr(asw, "_auth_outage_spawn_gate", lambda *a, **kw: None)
    monkeypatch.setattr(asw, "_stalled_session_overrides", lambda issue: [])


@pytest.mark.parametrize(("name", "call"), _BUILDERS, ids=[n for n, _ in _BUILDERS])
def test_all_six_builders_append_session_mode_when_async(
    monkeypatch, capsys, builder_seams, name, call
):
    monkeypatch.setattr(asw, "_resolve_session_mode", lambda issue, entry=None: "async")
    call()
    out = capsys.readouterr().out
    assert "--session-mode async" in out, f"{name}: dry-run argv missing the mode flag: {out}"


@pytest.mark.parametrize(("name", "call"), _BUILDERS, ids=[n for n, _ in _BUILDERS])
def test_all_six_builders_legacy_argv_without_async(monkeypatch, capsys, builder_seams, name, call):
    monkeypatch.setattr(asw, "_resolve_session_mode", lambda issue, entry=None: "auto")
    call()
    out = capsys.readouterr().out
    assert "--session-mode" not in out, f"{name}: legacy argv must carry NO mode flag: {out}"
    assert "spawn-issue --issue 77 --auto" in out


def test_gate_push_message_async_branch(monkeypatch):
    monkeypatch.setattr(asw, "_task_title", lambda issue: "slug")
    msg = asw._gate_push_message(42, "plan_pending", [], False, async_ask=True)
    assert "plan approval requested" in msg
    assert "set-status 42 approved" in msg
    # Legacy shapes unchanged when async_ask is False/omitted.
    legacy = asw._gate_push_message(42, "plan_pending", [], True)
    assert "no GPU-hour estimate" in legacy


def test_async_park_stale_alert_arm(monkeypatch):
    pushes: list[str] = []
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry: pushes.append(msg) or True)
    monkeypatch.setattr(asw, "_task_title", lambda issue: "slug")
    monkeypatch.delenv("EPM_ASYNC_PARK_STALE_HOURS", raising=False)

    stale = [_ask(age_s=13 * 3600.0)]
    fresh = [_ask(age_s=1 * 3600.0)]
    answered = [_ask(age_s=13 * 3600.0), _row("epm:ask-answered", 60.0)]

    # Young ask: below the 12h default floor -> no alert.
    assert asw._async_park_stale_alert_arm(9, "plan_pending", fresh, {}, False) is None
    # Answered ask: not open -> no alert.
    assert asw._async_park_stale_alert_arm(9, "plan_pending", answered, {}, False) is None
    assert pushes == []
    # Stale open ask: alerts once, returns the ask epoch (the dedup key).
    got = asw._async_park_stale_alert_arm(9, "plan_pending", stale, {}, False)
    assert got is not None and len(pushes) == 1 and "unanswered" in pushes[0]
    # Dedup: same ask ts persisted -> silent.
    state = {"async_ask_alerted_ts": got}
    assert asw._async_park_stale_alert_arm(9, "plan_pending", stale, state, False) is None
    assert len(pushes) == 1
    # A NEWER ask re-alerts (different epoch).
    newer = [_ask(age_s=12.5 * 3600.0)]
    assert asw._async_park_stale_alert_arm(9, "plan_pending", newer, state, False) is not None
    assert len(pushes) == 2
    # Env floor override: a 2h ask alerts under a 1h floor.
    monkeypatch.setenv("EPM_ASYNC_PARK_STALE_HOURS", "1")
    assert asw._async_park_stale_alert_arm(9, "plan_pending", fresh, {}, False) is not None


def test_save_gate_notify_state_legacy_payload_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    asw._save_gate_notify_state(5, last_status="running")
    data = json.loads((tmp_path / "gate-notify-5.json").read_text())
    assert set(data) == {"last_status", "ts"}  # no new key on the legacy call
    asw._save_gate_notify_state(5, last_status="plan_pending", async_ask_alerted_ts=123.0)
    data = json.loads((tmp_path / "gate-notify-5.json").read_text())
    assert data["async_ask_alerted_ts"] == 123.0


def test_decide_gate_push_pure_predicate_unchanged():
    """W1 composes the ask into the CALLER's over_cap argument; the pure
    predicate itself is byte-identical legacy."""
    assert tick_triage.plan_pending_over_cap([]) is False  # shared predicate untouched
    assert asw.decide_gate_push("plan_pending", "running", False) is False
    assert asw.decide_gate_push("plan_pending", "running", True) is True
