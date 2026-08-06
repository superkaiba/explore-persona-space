"""Unit tests for the #1582 ESCALATE-ONLY keep-running wedged-owner arm.

The #1345 incident this arm closes: pod-1345-onpolicy was provisioned under
the ``keep-running`` tag on a task parked at ``awaiting_promotion``; the
owning session wrapper froze (days-idle transcript, 0% CPU) and the tag
made the pod invisible to every watcher check — the ``keep-running-skip``
branch posted its once-per-incarnation note and then stayed silent while
the pod billed ~72h until a fully manual recovery. Covers:

* the pure predicate ``decide_keep_running_owner_escalation`` (gap gate,
  live/unknown/wedged/absent legs, threshold accumulation, episode-open
  marker key, the 24h re-alert TTL, the ``>=`` idle boundary);
* the owner resolver ``_keep_running_owner_state`` (registration sids +
  ``/proc/<pid>/cwd``-mapped children, the fresh-self-report rescue BEFORE
  any absent classification, unknown on daemon/transcript misses);
* the #1345 replay end-to-end through ``_process_pod`` — exactly ONE marker
  per episode (its note CARRIES the anti-liveness sentinel and the
  ``_post_progress_marker`` stub APPENDS the row to the fixture
  events.jsonl, so later ticks genuinely exercise the self-reset loop),
  ONE push + ONE sidecar row on the confirming tick, a re-push (no second
  marker) after 24h, and NEVER a stop/terminate;
* the ``_save_pod_safety_state`` ``kr_owner_*`` ``_CARRY`` forward-carry +
  the IN-SAVE pod_id-keyed reset (exercised THROUGH a status-class-shaped
  save, not an arm-side reset);
* the env readers, kill switch, dry-run no-writes contract, vetoes (no
  daemon probe), and the sidecar row schema (real appender body).

Follows ``tests/test_autonomous_session_watch_unlaunched_orphan.py``
conventions: PodInfo fixtures, the patched state dir, ``task.py`` reads
monkeypatched, no network, no real marker posts (the shared #1247/#1265
conftest autouse hermeticity guards cover this module with zero ceremony).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
from runpod_api import PodInfo  # noqa: E402

IDLE_S = asw.KEEP_RUNNING_OWNER_IDLE_S  # 12h (env-overridable at call time)
REALERT_S = asw.KEEP_RUNNING_OWNER_REALERT_S  # 24h (env-overridable at call time)

# The genuine appender body, captured at import time (before any fixture
# stubs it) — the code-style "one production-body test per seam-stubbed
# function" rule: tests 21/24 execute the real body against a tmp sidecar.
_REAL_APPEND = asw._append_keep_running_wedged_event


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point the per-pod state dir at a tmp dir (mirrors the sibling suites)."""
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    return tmp_path


def _iso(epoch: float) -> str:
    """Canonical task-event / RunPod ``createdAt`` timestamp shape."""
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _info(
    pod_id: str = "pm7f1345",
    name: str = "pod-1345-onpolicy",
    created_at: str | None = None,
    gpu_count: int | None = 1,
    gpu_type_id: str | None = "NVIDIA H100 80GB HBM3",
) -> PodInfo:
    """A RUNNING pod with a public port (so the #692 wedge arm never handles it)."""
    return PodInfo(
        pod_id=pod_id,
        name=name,
        desired_status="RUNNING",
        gpu_count=gpu_count,
        gpu_type_id=gpu_type_id,
        ssh_host="1.2.3.4",
        ssh_port=22001,
        created_at=created_at,
    )


def _decide_kwargs(now: float, **overrides) -> dict:
    """decide_keep_running_owner_escalation kwargs that ESCALATE unless overridden."""
    kwargs = {
        "progress_gap_s": 71 * 3600.0,
        "owner_state": "wedged",
        "missed": 1,
        "threshold": 2,
        "min_idle_s": float(IDLE_S),
        "first_ts": None,
        "last_alert_ts": None,
        "now": now,
        "realert_s": float(REALERT_S),
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def rig_1345(monkeypatch, isolated_registry):
    """The #1345 replay rig: RUNNING tagged pod, task at awaiting_promotion,
    last REAL marker 71h old, one REGISTERED live owner sid whose transcript
    is 88h idle, self-report absent, no vetoes.

    ``_task_events`` returns a LIVE list and the ``_post_progress_marker``
    stub APPENDS each posted marker to it (ts = the rig clock), so the real
    ``_latest_progress_ts`` filter — the self-reset protection — is
    exercised end-to-end across ticks. ``_append_keep_running_wedged_event``
    is recorded (payload dicts); tests 21/24 restore the REAL body.
    """
    t0 = time.time()
    clock = {"now": t0}
    events = [
        {
            "kind": "epm:status-changed",
            "ts": _iso(t0 - 72 * 3600),
            "note": "verifying -> awaiting_promotion",
            "by": "task-workflow",
        },
        {
            "kind": "epm:progress",
            "ts": _iso(t0 - 71 * 3600),
            "note": "Park restored after the ownership yield",
            "by": "orchestrator",
        },
    ]
    posts: list[tuple[int, str, str | None, bool]] = []
    pushes: list[tuple[str, bool]] = []
    stops: list[int] = []
    sidecar: list[dict] = []

    def _post_stub(issue, note, dry_run, label=None):
        posts.append((issue, note, label, dry_run))
        events.append(
            {
                "kind": "epm:progress",
                "ts": _iso(clock["now"]),
                "note": note,
                "by": "autonomous_session_watch",
            }
        )

    monkeypatch.setattr(asw, "_task_status", lambda issue: "awaiting_promotion")
    monkeypatch.setattr(asw, "_task_events", lambda issue: events)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: True)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None, **_kw: False)
    monkeypatch.setattr(asw, "_post_progress_marker", _post_stub)
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: pushes.append((msg, dry_run)))
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(
        asw, "_live_children", lambda **kw: [{"happySessionId": "sid-1345", "pid": 2200056}]
    )
    (isolated_registry / "issue-1345.json").write_text(
        json.dumps({"issue": 1345, "happy_session_id": "sid-1345"})
    )
    monkeypatch.setattr(asw, "_transcript_idle_age_s", lambda pid, now: (88 * 3600.0, None))
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (None, None))
    monkeypatch.setattr(asw, "_provision_in_flight_reason", lambda issue, now: None)
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda issue, now, window_s, **kw: False)
    monkeypatch.setattr(asw, "_proc_cwd", lambda pid: None)
    monkeypatch.setattr(
        asw,
        "_append_keep_running_wedged_event",
        lambda payload, dry_run: sidecar.append(payload),
    )
    # #2149 pod-idle leg SSH boundary: stubbed to a FAILED probe (None ->
    # freeze, no emission) so the leg — which runs on every owner non-fire
    # tick by design — stays hermetic + behavior-neutral for the owner-leg
    # tests above. Pod-idle tests re-patch this per scenario.
    monkeypatch.setattr(asw, "_probe_pod_idleness", lambda pod_name: None)

    def tick(offset_s: float = 0.0, *, dry_run: bool = False, threshold: int = 2) -> None:
        clock["now"] = t0 + offset_s
        asw._process_pod(
            1345,
            "pm7f1345",
            _info(created_at=_iso(t0 - 72 * 3600)),
            clock["now"],
            dry_run=dry_run,
            threshold=threshold,
        )

    return SimpleNamespace(
        t0=t0,
        clock=clock,
        events=events,
        posts=posts,
        pushes=pushes,
        stops=stops,
        sidecar=sidecar,
        state_path=isolated_registry / "pod-safety-1345.json",
        registry=isolated_registry,
        tick=tick,
    )


def _wedged_posts(posts):
    return [p for p in posts if asw._KEEP_RUNNING_WEDGED_NOTE_SENTINEL in p[1]]


def _skip_posts(posts):
    return [p for p in posts if asw._KEEP_RUNNING_NOTE_SENTINEL in p[1]]


# ---------------------------------------------------------------------------
# 1. Pure predicate (decide_keep_running_owner_escalation) — tests 1-8
# ---------------------------------------------------------------------------


def test_fires_escalate_on_wedged_owner_confirmed():
    """gap >= 12h, owner wedged, missed reaches threshold -> ("escalate", 2)."""
    now = time.time()
    assert asw.decide_keep_running_owner_escalation(**_decide_kwargs(now)) == ("escalate", 2)


def test_fires_escalate_on_absent_owner_confirmed():
    """An ABSENT owner fires identically to a wedged one (both are confirmed
    no-live-owner evidence — the #1345 no-registration class)."""
    now = time.time()
    assert asw.decide_keep_running_owner_escalation(
        **_decide_kwargs(now, owner_state="absent")
    ) == ("escalate", 2)


def test_live_owner_clears():
    """A LIVE owner clears even at a 100h marker gap (quiet long runs are
    legitimate while the owner demonstrably works)."""
    now = time.time()
    assert asw.decide_keep_running_owner_escalation(
        **_decide_kwargs(now, owner_state="live", progress_gap_s=100 * 3600.0, missed=5)
    ) == ("clear", 0)


def test_unknown_owner_freezes():
    """Unknown owner state FREEZES the counter (no increment, no reset, no
    emission) — fail toward no-fire; an unrecognized value freezes too."""
    now = time.time()
    assert asw.decide_keep_running_owner_escalation(
        **_decide_kwargs(now, owner_state="unknown", missed=1)
    ) == ("hold", 1)
    assert asw.decide_keep_running_owner_escalation(
        **_decide_kwargs(now, owner_state="garbage", missed=1)
    ) == ("hold", 1)


def test_fresh_progress_clears():
    """gap < 12h -> clear; gap None (unknowable) -> clear."""
    now = time.time()
    assert asw.decide_keep_running_owner_escalation(
        **_decide_kwargs(now, progress_gap_s=3600.0, missed=2)
    ) == ("clear", 0)
    assert asw.decide_keep_running_owner_escalation(
        **_decide_kwargs(now, progress_gap_s=None, missed=2)
    ) == ("clear", 0)


def test_below_threshold_accumulates_silently():
    """missed 0 -> 1 at threshold 2 -> ("hold", 1): accumulate, nothing emitted."""
    now = time.time()
    assert asw.decide_keep_running_owner_escalation(**_decide_kwargs(now, missed=0)) == (
        "hold",
        1,
    )


def test_realert_ttl():
    """Confirmed + episode open: within 24h of the last push -> hold; at/past
    24h -> re-alert (push+sidecar only — the marker key is first_ts)."""
    now = time.time()
    assert asw.decide_keep_running_owner_escalation(
        **_decide_kwargs(now, first_ts=now - 25 * 3600, last_alert_ts=now - 3600, missed=5)
    ) == ("hold", 6)
    assert asw.decide_keep_running_owner_escalation(
        **_decide_kwargs(
            now, first_ts=now - 49 * 3600, last_alert_ts=now - float(REALERT_S), missed=5
        )
    ) == ("re-alert", 6)
    # An open episode with NO last_alert_ts (garbled state) re-alerts rather
    # than going silent forever.
    assert asw.decide_keep_running_owner_escalation(
        **_decide_kwargs(now, first_ts=now - 3600, last_alert_ts=None, missed=5)
    ) == ("re-alert", 6)


def test_boundary_gap_equals_threshold_fires():
    """gap == min_idle_s COUNTS as idle (>= fires — the decide_idle_unmapped
    `idle_age_s < idle_reap_s -> clear` complement)."""
    now = time.time()
    assert asw.decide_keep_running_owner_escalation(
        **_decide_kwargs(now, progress_gap_s=float(IDLE_S))
    ) == ("escalate", 2)
    assert asw.decide_keep_running_owner_escalation(
        **_decide_kwargs(now, progress_gap_s=float(IDLE_S) - 1.0)
    ) == ("clear", 0)


# ---------------------------------------------------------------------------
# 2. Env readers / constants — tests 9-10
# ---------------------------------------------------------------------------


def test_env_overrides_and_malformed_fallback(monkeypatch):
    """The two HOURS knobs are honored; malformed / non-positive values fall
    back to the defaults (never a kill switch / instant pager)."""
    monkeypatch.delenv("EPM_KEEP_RUNNING_WEDGED_OWNER_MIN_H", raising=False)
    monkeypatch.delenv("EPM_KEEP_RUNNING_WEDGED_REALERT_H", raising=False)
    assert asw._keep_running_owner_idle_s() == float(IDLE_S) == 12 * 3600.0
    assert asw._keep_running_owner_realert_s() == float(REALERT_S) == 24 * 3600.0
    monkeypatch.setenv("EPM_KEEP_RUNNING_WEDGED_OWNER_MIN_H", "6")
    assert asw._keep_running_owner_idle_s() == 6 * 3600.0
    monkeypatch.setenv("EPM_KEEP_RUNNING_WEDGED_REALERT_H", "48")
    assert asw._keep_running_owner_realert_s() == 48 * 3600.0
    for bad in ("garbage", "", "0", "-5"):
        monkeypatch.setenv("EPM_KEEP_RUNNING_WEDGED_OWNER_MIN_H", bad)
        monkeypatch.setenv("EPM_KEEP_RUNNING_WEDGED_REALERT_H", bad)
        assert asw._keep_running_owner_idle_s() == float(IDLE_S)
        assert asw._keep_running_owner_realert_s() == float(REALERT_S)


def test_sentinel_in_watcher_note_sentinels():
    """Conjunct (a) of the self-reset protection: the sentinel is a MEMBER of
    _WATCHER_NOTE_SENTINELS, and a marker row carrying it is excluded from
    the real-progress clock (otherwise the escalation marker would end its
    own episode next tick)."""
    assert asw._KEEP_RUNNING_WEDGED_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS
    now = time.time()
    events = [
        {
            "kind": "epm:progress",
            "ts": _iso(now),
            "note": f"{asw._KEEP_RUNNING_WEDGED_NOTE_SENTINEL} KEEP-RUNNING WEDGED OWNER "
            "(#1582): escalation record",
            "by": "autonomous_session_watch",
        }
    ]
    assert asw._latest_progress_ts(events) is None


# ---------------------------------------------------------------------------
# 3. Integration — the #1345 replay through _process_pod — tests 11-19
# ---------------------------------------------------------------------------


def test_1345_timeline_fires_once_then_realerts(rig_1345):
    """The durability pin: tick1 accumulates silently; tick2 escalates with
    exactly ONE marker + ONE push + ONE sidecar row; tick3 (+10 min) emits
    nothing new; a tick 24h past the escalation re-pushes (ONE more push +
    sidecar row) with NO second marker.

    Conjunct (b) of the self-reset protection (r1 Statistics Must-Fix): the
    recorded marker note CONTAINS the sentinel (the _latest_progress_ts
    exclusion is a SUBSTRING match on the note — set membership alone is
    necessary but not sufficient), and the marker row was APPENDED to the
    fixture events.jsonl, so tick3/+24h exercise the self-reset loop
    end-to-end: a template rewording that drops the sentinel prefix would
    reset the progress clock at tick3, end the episode, and make the +24h
    re-push assertion fail."""
    rig = rig_1345
    rig.tick(0)  # tick1: hold (missed 0 -> 1), nothing emitted
    assert _wedged_posts(rig.posts) == []
    assert rig.pushes == []
    assert rig.sidecar == []
    state = json.loads(rig.state_path.read_text())
    assert state["kr_owner_missed"] == 1
    assert state["kr_owner_first_ts"] is None

    rig.tick(600)  # tick2: escalate — ONE marker + ONE push + ONE sidecar row
    fired = _wedged_posts(rig.posts)
    assert len(fired) == 1
    issue, note, label, dry_run = fired[0]
    assert issue == 1345
    assert label == "keep-running-wedged-owner"
    assert dry_run is False
    assert asw._KEEP_RUNNING_WEDGED_NOTE_SENTINEL in note  # conjunct (b)
    assert "pod-1345-onpolicy" in note
    assert "keep-running" in note
    assert "remove-tag 1345 keep-running" in note
    assert "NOT auto-stopped" in note
    # The stub appended the marker row to the fixture events (self-reset loop
    # genuinely exercised on the later ticks).
    assert any(asw._KEEP_RUNNING_WEDGED_NOTE_SENTINEL in (e.get("note") or "") for e in rig.events)
    assert len(rig.pushes) == 1
    assert "wedged" in rig.pushes[0][0]
    assert len(rig.sidecar) == 1
    state = json.loads(rig.state_path.read_text())
    assert state["kr_owner_first_ts"] == pytest.approx(rig.t0 + 600)
    assert state["kr_owner_last_alert_ts"] == pytest.approx(rig.t0 + 600)

    rig.tick(1200)  # tick3: within the 24h TTL — nothing new
    assert len(_wedged_posts(rig.posts)) == 1
    assert len(rig.pushes) == 1
    assert len(rig.sidecar) == 1

    rig.tick(600 + float(REALERT_S))  # +24h: re-push + sidecar row, NO second marker
    assert len(_wedged_posts(rig.posts)) == 1
    assert len(rig.pushes) == 2
    assert len(rig.sidecar) == 2
    assert rig.stops == []


def test_never_stops_or_terminates(rig_1345, monkeypatch):
    """Escalate-only invariant on a FIRING tick: _stop_pod is never called,
    no `pod.py` subprocess argv is ever constructed, the branch's action
    stays keep-running-skip, and the existing once-per-incarnation
    keep-running-skip note logic is unchanged."""
    rig = rig_1345
    argvs: list[list[str]] = []
    real_run = subprocess.run

    def _record_run(argv, *a, **kw):  # pragma: no cover - never expected to run
        argvs.append(list(argv) if isinstance(argv, (list, tuple)) else [str(argv)])
        raise AssertionError(f"unexpected subprocess.run during the rig ticks: {argv}")

    monkeypatch.setattr(subprocess, "run", _record_run)
    try:
        rig.tick(0)
        rig.tick(600)  # the firing tick
    finally:
        monkeypatch.setattr(subprocess, "run", real_run)
    assert len(_wedged_posts(rig.posts)) == 1  # the arm fired...
    assert rig.stops == []  # ...and nothing was stopped
    assert not any(any("pod.py" in tok for tok in argv) for argv in argvs)
    assert len(_skip_posts(rig.posts)) == 1  # once-per-incarnation note unchanged
    state = json.loads(rig.state_path.read_text())
    assert state["keep_running_noted"] is True  # the branch's own save ran


def test_live_owner_no_fire_end_to_end(rig_1345, monkeypatch):
    """Same #1345 shape but the owner transcript is 1h idle -> live -> no
    emission across 3 ticks; kr fields stay at defaults."""
    rig = rig_1345
    monkeypatch.setattr(asw, "_transcript_idle_age_s", lambda pid, now: (3600.0, None))
    for offset in (0, 600, 1200):
        rig.tick(offset)
    assert _wedged_posts(rig.posts) == []
    assert rig.pushes == []
    assert rig.sidecar == []
    state = json.loads(rig.state_path.read_text())
    assert state["kr_owner_missed"] == 0
    assert state["kr_owner_first_ts"] is None
    assert state["kr_owner_last_alert_ts"] is None


def test_absent_owner_fires(rig_1345, monkeypatch):
    """No registration file + no cwd-mapped child -> "absent" -> fires with
    the absent-flavored note text (same conjunct-(b) sentinel pin)."""
    rig = rig_1345
    (rig.registry / "issue-1345.json").unlink()
    monkeypatch.setattr(asw, "_live_children", lambda **kw: [])
    rig.tick(0)
    rig.tick(600)
    fired = _wedged_posts(rig.posts)
    assert len(fired) == 1
    note = fired[0][1]
    assert asw._KEEP_RUNNING_WEDGED_NOTE_SENTINEL in note
    assert "absent" in note
    assert "no live registration or worktree-cwd session" in note
    assert rig.sidecar[0]["owner_state"] == "absent"
    assert rig.stops == []


def test_cwd_mapped_owner_detected(isolated_registry, monkeypatch):
    """The #1345 `~#N` coverage: NO registration file, one live daemon child
    whose /proc/<pid>/cwd (seam-stubbed) is the issue worktree -> the child
    is an owner candidate. Fresh transcript -> live (no fire); idle
    transcript -> wedged, evidence naming the cwd-mapped pid."""
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(
        asw, "_live_children", lambda **kw: [{"happySessionId": "s-cwd", "pid": 777}]
    )
    monkeypatch.setattr(
        asw,
        "_proc_cwd",
        lambda pid: "/home/u/eps/.claude/worktrees/issue-1345" if pid == 777 else None,
    )
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (None, None))
    now = time.time()
    monkeypatch.setattr(asw, "_transcript_idle_age_s", lambda pid, now: (3600.0, None))
    state, evidence = asw._keep_running_owner_state(1345, now, float(IDLE_S))
    assert state == "live"
    assert evidence["pid"] == 777
    monkeypatch.setattr(asw, "_transcript_idle_age_s", lambda pid, now: (88 * 3600.0, None))
    state, evidence = asw._keep_running_owner_state(1345, now, float(IDLE_S))
    assert state == "wedged"
    assert evidence["pid"] == 777
    assert evidence["sid"] == "s-cwd"


def test_daemon_unreachable_freezes(rig_1345, monkeypatch):
    """Daemon unreachable -> owner "unknown" -> hold: no emission, the
    confirmation counter FROZEN (neither incremented nor reset)."""
    rig = rig_1345
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: False)
    rig.state_path.write_text(
        json.dumps(
            {
                "pod_id": "pm7f1345",
                "missed": 0,
                "alerted": False,
                "last_progress_ts": None,
                "first_seen": rig.t0 - 7200,
                "kr_owner_missed": 1,
            }
        )
    )
    rig.tick(0)
    assert _wedged_posts(rig.posts) == []
    assert rig.pushes == []
    assert rig.sidecar == []
    state = json.loads(rig.state_path.read_text())
    assert state["kr_owner_missed"] == 1  # frozen, not incremented to 2
    assert state["kr_owner_first_ts"] is None


def test_daemon_list_failed_freezes(rig_1345, monkeypatch):
    """Daemon reachable but the strict /list probe raises -> owner "unknown"
    -> hold: no emission, the confirmation counter FROZEN (#1582 r2 —
    LENIENT _live_children() returned [] on the same flake, misreading it
    as "no children" -> owner "absent", the one evidence read that failed
    toward FIRE)."""
    rig = rig_1345

    def _raise_strict(**kw):
        assert kw.get("strict") is True, "owner resolver must probe /list in strict mode"
        raise RuntimeError("daemon /list failed: flake")

    monkeypatch.setattr(asw, "_live_children", _raise_strict)
    state, evidence = asw._keep_running_owner_state(1345, rig.t0, float(IDLE_S))
    assert state == "unknown"
    assert evidence == {"reason": "daemon-list-failed"}
    rig.state_path.write_text(
        json.dumps(
            {
                "pod_id": "pm7f1345",
                "missed": 0,
                "alerted": False,
                "last_progress_ts": None,
                "first_seen": rig.t0 - 7200,
                "kr_owner_missed": 1,
            }
        )
    )
    rig.tick(0)
    assert _wedged_posts(rig.posts) == []
    assert rig.pushes == []
    assert rig.sidecar == []
    state = json.loads(rig.state_path.read_text())
    assert state["kr_owner_missed"] == 1  # frozen, not incremented to 2
    assert state["kr_owner_first_ts"] is None


def test_vetoes(rig_1345, monkeypatch):
    """Provision-in-flight OR fresh worktree activity -> clear, and the
    daemon is never probed (the owner resolution is the LAST, priciest
    step)."""
    rig = rig_1345

    def _boom(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("daemon probed despite an active veto")

    monkeypatch.setattr(asw, "_daemon_reachable", _boom)
    monkeypatch.setattr(asw, "_live_children", _boom)
    monkeypatch.setattr(
        asw, "_provision_in_flight_reason", lambda issue, now: "live provision (pid 1)"
    )
    rig.tick(0)
    monkeypatch.setattr(asw, "_provision_in_flight_reason", lambda issue, now: None)
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda issue, now, window_s, **kw: True)
    rig.tick(600)
    assert _wedged_posts(rig.posts) == []
    assert rig.pushes == []
    state = json.loads(rig.state_path.read_text())
    assert state["kr_owner_missed"] == 0


def test_kill_switch(rig_1345, monkeypatch):
    """EPM_DISABLE_KEEP_RUNNING_OWNER_AUDIT=1: the helper returns immediately
    (no probes, no emission); the branch behavior is byte-identical (the
    existing once-per-incarnation note + save still run)."""
    rig = rig_1345
    monkeypatch.setenv("EPM_DISABLE_KEEP_RUNNING_OWNER_AUDIT", "1")

    def _boom(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("probe ran despite the kill switch")

    monkeypatch.setattr(asw, "_daemon_reachable", _boom)
    monkeypatch.setattr(asw, "_provision_in_flight_reason", _boom)
    rig.tick(0)
    rig.tick(600)
    assert _wedged_posts(rig.posts) == []
    assert rig.pushes == []
    assert rig.sidecar == []
    assert len(_skip_posts(rig.posts)) == 1  # existing branch note unchanged
    state = json.loads(rig.state_path.read_text())
    assert state["keep_running_noted"] is True  # existing branch save unchanged
    assert state["kr_owner_missed"] == 0


def test_no_tag_branch_untouched(rig_1345, monkeypatch):
    """Without the keep-running tag an escaped DONE pod takes the existing
    stop path; the new helper is never invoked (spy)."""
    rig = rig_1345
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: False)
    calls: list[int] = []
    monkeypatch.setattr(
        asw,
        "_maybe_escalate_keep_running_wedged_owner",
        lambda issue, *a, **kw: calls.append(issue),
    )
    # missed=1 -> new_missed=2 == threshold -> the auto-stop fires this tick.
    rig.state_path.write_text(
        json.dumps(
            {
                "pod_id": "pm7f1345",
                "missed": 1,
                "alerted": False,
                "last_progress_ts": None,
                "first_seen": rig.t0 - 7200,
            }
        )
    )
    rig.tick(0)
    assert rig.stops == [1345]  # the canonical auto-stop still ran
    assert calls == []  # the new helper never fired outside the tagged branch
    assert _wedged_posts(rig.posts) == []


# ---------------------------------------------------------------------------
# 4. State carry / dry-run / direct-driver / sidecar — tests 20-24
# ---------------------------------------------------------------------------


def test_state_carry_and_pod_incarnation_reset(isolated_registry):
    """kr fields survive a status-class-arm _save_pod_safety_state (via
    _CARRY), and a save under a NEW pod_id resets them to a fresh episode.
    The pod_id reset is exercised THROUGH a status-class-shaped save that
    leaves the kr fields at their _CARRY defaults (the in-save reset per the
    orphan_gcp_noted pattern — an arm-side-only reset would be laundered
    past by the wedge arm's same-tick pod_id rewrite)."""
    t = time.time()
    asw._save_pod_safety_state(
        1345,
        "pA",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        kr_owner_missed=2,
        kr_owner_first_ts=t,
        kr_owner_last_alert_ts=t,
    )
    state = json.loads((isolated_registry / "pod-safety-1345.json").read_text())
    assert (state["kr_owner_missed"], state["kr_owner_first_ts"]) == (2, t)

    # Status-class-shaped save (kr fields left at _CARRY), SAME pod: carried.
    asw._save_pod_safety_state(
        1345, "pA", missed=0, alerted=False, last_progress_ts=None, prev=state
    )
    state = json.loads((isolated_registry / "pod-safety-1345.json").read_text())
    assert state["kr_owner_missed"] == 2
    assert state["kr_owner_first_ts"] == t
    assert state["kr_owner_last_alert_ts"] == t

    # Status-class-shaped save under a NEW pod_id: in-save reset (fresh episode).
    asw._save_pod_safety_state(
        1345, "pB", missed=0, alerted=False, last_progress_ts=None, prev=state
    )
    state = json.loads((isolated_registry / "pod-safety-1345.json").read_text())
    assert state["kr_owner_missed"] == 0
    assert state["kr_owner_first_ts"] is None
    assert state["kr_owner_last_alert_ts"] is None


def test_dry_run_no_writes(rig_1345, monkeypatch, tmp_path):
    """Dry-run on a FIRING tick: no state-file write, no sidecar append (the
    REAL appender body runs and dry-run-prints only), and the marker/push
    helpers are invoked with dry_run=True."""
    rig = rig_1345
    sidecar_file = tmp_path / "cache" / "keep-running-wedged-events.jsonl"
    monkeypatch.setattr(asw, "_append_keep_running_wedged_event", _REAL_APPEND)
    monkeypatch.setattr(asw, "_keep_running_wedged_sidecar_path", lambda: sidecar_file)
    seeded = {
        "pod_id": "pm7f1345",
        "missed": 0,
        "alerted": False,
        "last_progress_ts": None,
        "first_seen": rig.t0 - 7200,
        "kr_owner_missed": 1,
    }
    rig.state_path.write_text(json.dumps(seeded))
    rig.tick(0, dry_run=True)  # missed 1 -> 2 == threshold: fires (dry)
    fired = _wedged_posts(rig.posts)
    assert len(fired) == 1
    assert fired[0][3] is True  # marker helper invoked with dry_run=True
    assert rig.pushes == [(rig.pushes[0][0], True)]  # push invoked with dry_run=True
    assert not sidecar_file.exists()  # real appender: dry-run print only
    assert json.loads(rig.state_path.read_text()) == seeded  # no state write


def test_info_none_and_created_at_fallback(isolated_registry, monkeypatch):
    """Direct-driver hardening: latest_progress=None AND info=None -> clear
    (no crash, no probes); with info provided, the gap falls back to
    _parse_event_ts(info.created_at) and fires when old enough."""

    def _boom(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("probe ran on the None/None clear path")

    monkeypatch.setattr(asw, "_daemon_reachable", _boom)
    monkeypatch.setattr(asw, "_provision_in_flight_reason", _boom)
    now = time.time()
    fired = asw._maybe_escalate_keep_running_wedged_owner(
        1345, "pX", None, "awaiting_promotion", now, None, 2, {}, True
    )
    assert fired is False

    # created_at fallback: a 20h-old pod with no real marker fires (threshold
    # 1 so a single confirmed tick escalates).
    posts: list[tuple] = []
    pushes: list[tuple] = []
    sidecar: list[dict] = []
    monkeypatch.setattr(
        asw,
        "_post_progress_marker",
        lambda issue, note, dry_run, label=None: posts.append((issue, note)),
    )
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: pushes.append(msg))
    monkeypatch.setattr(
        asw, "_append_keep_running_wedged_event", lambda payload, dry_run: sidecar.append(payload)
    )
    monkeypatch.setattr(asw, "_provision_in_flight_reason", lambda issue, now: None)
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda issue, now, window_s, **kw: False)
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_children", lambda **kw: [{"happySessionId": "s", "pid": 9}])
    (isolated_registry / "issue-1345.json").write_text(
        json.dumps({"issue": 1345, "happy_session_id": "s"})
    )
    monkeypatch.setattr(asw, "_transcript_idle_age_s", lambda pid, now: (40 * 3600.0, None))
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (None, None))
    info = _info(created_at=_iso(now - 20 * 3600))
    fired = asw._maybe_escalate_keep_running_wedged_owner(
        1345, "pm7f1345", info, "awaiting_promotion", now, None, 1, {}, True
    )
    assert fired is True
    assert len(posts) == 1
    assert asw._KEEP_RUNNING_WEDGED_NOTE_SENTINEL in posts[0][1]
    assert len(pushes) == 1
    assert sidecar[0]["progress_gap_h"] == pytest.approx(20.0, abs=0.1)


def test_sidecar_row_schema(rig_1345, monkeypatch, tmp_path):
    """Field-presence pin on the escalate-path sidecar row (schema drift is
    audit-trail damage), written through the REAL appender body against a
    tmp sidecar path."""
    rig = rig_1345
    sidecar_file = tmp_path / "cache" / "keep-running-wedged-events.jsonl"
    monkeypatch.setattr(asw, "_append_keep_running_wedged_event", _REAL_APPEND)
    monkeypatch.setattr(asw, "_keep_running_wedged_sidecar_path", lambda: sidecar_file)
    rig.tick(0)
    rig.tick(600)  # escalate
    rows = [json.loads(line) for line in sidecar_file.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    for field in (
        "ts",
        "kind",
        "issue",
        "pod_id",
        "pod_name",
        "status",
        "progress_gap_h",
        "owner_state",
        "action",
    ):
        assert field in row, f"sidecar row missing {field!r}"
    assert row["kind"] == "keep-running-wedged-owner"
    assert row["issue"] == 1345
    assert row["pod_id"] == "pm7f1345"
    assert row["pod_name"] == "pod-1345-onpolicy"
    assert row["status"] == "awaiting_promotion"
    assert row["owner_state"] == "wedged"
    assert row["action"] == "escalated"


def test_self_report_fresh_rescues_missing_does_not(isolated_registry, monkeypatch):
    """Fresh self-report -> live (rescue); (None, None) self-report + idle
    transcript -> still wedged; AND the no-candidates + fresh-self-report
    cell -> "live" (pins the rescue-BEFORE-absent ordering)."""
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_children", lambda **kw: [{"happySessionId": "s", "pid": 42}])
    monkeypatch.setattr(asw, "_proc_cwd", lambda pid: None)
    (isolated_registry / "issue-1345.json").write_text(
        json.dumps({"issue": 1345, "happy_session_id": "s"})
    )
    monkeypatch.setattr(asw, "_transcript_idle_age_s", lambda pid, now: (88 * 3600.0, None))
    now = time.time()

    # Fresh self-report rescues an idle-transcript candidate.
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (600.0, "ts"))
    state, evidence = asw._keep_running_owner_state(1345, now, float(IDLE_S))
    assert state == "live"
    assert evidence["reason"] == "fresh-self-report"

    # Missing self-report does NOT rescue: idle transcript -> wedged.
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (None, None))
    state, _evidence = asw._keep_running_owner_state(1345, now, float(IDLE_S))
    assert state == "wedged"

    # No candidates + FRESH self-report -> live, not absent (the rescue runs
    # BEFORE any absent classification).
    (isolated_registry / "issue-1345.json").unlink()
    monkeypatch.setattr(asw, "_live_children", lambda **kw: [])
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (600.0, "ts"))
    state, _evidence = asw._keep_running_owner_state(1345, now, float(IDLE_S))
    assert state == "live"
    # ...and with the self-report also missing, the same cell is absent.
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (None, None))
    state, evidence = asw._keep_running_owner_state(1345, now, float(IDLE_S))
    assert state == "absent"
    assert evidence["reason"] == "no-candidates"


def test_keep_running_owner_state_honors_short_window(isolated_registry, monkeypatch):
    """#1667 reuse-contract pin: ``min_idle_s`` is honored as passed (no hidden
    12h constant inside) — the wedge owner guard reuses this resolver with its
    SHORT 2h window, while the #1582 arm's 12h call is unaffected."""
    monkeypatch.setattr(asw, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(asw, "_live_children", lambda **kw: [{"happySessionId": "s", "pid": 42}])
    monkeypatch.setattr(asw, "_proc_cwd", lambda pid: None)
    monkeypatch.setattr(asw, "_self_report_age_seconds", lambda issue, now: (None, None))
    (isolated_registry / "issue-1667.json").write_text(
        json.dumps({"issue": 1667, "happy_session_id": "s"})
    )
    now = time.time()

    # Candidate transcript idle 1h < the 2h window -> "live".
    monkeypatch.setattr(asw, "_transcript_idle_age_s", lambda pid, now: (1 * 3600.0, None))
    state, _ev = asw._keep_running_owner_state(1667, now, min_idle_s=7200.0)
    assert state == "live"

    # Candidate transcript idle 3h >= the 2h window -> "wedged".
    monkeypatch.setattr(asw, "_transcript_idle_age_s", lambda pid, now: (3 * 3600.0, None))
    state, _ev = asw._keep_running_owner_state(1667, now, min_idle_s=7200.0)
    assert state == "wedged"

    # The #1582 arm's 12h window is unaffected: the same 3h-idle candidate
    # reads "live" at min_idle_s=12h.
    state, _ev = asw._keep_running_owner_state(1667, now, min_idle_s=float(IDLE_S))
    assert state == "live"


# ---------------------------------------------------------------------------
# 6. #2149 pod-grain idleness leg — pure predicate
# ---------------------------------------------------------------------------

POD_SENT_S = asw.KEEP_RUNNING_POD_IDLE_MIN_S  # 4h sentinel-tier floor
POD_UTIL_S = asw.KEEP_RUNNING_POD_UTIL_IDLE_MIN_S  # 12h utilization-tier floor


def _pod_decide_kwargs(now: float, **overrides) -> dict:
    """decide_keep_running_pod_idle_escalation kwargs that ESCALATE on the
    sentinel tier unless overridden (the #1739 shape: done-sentinel ~19.6h,
    logs ~19h stale, measured 0% util)."""
    kwargs = {
        "log_write_age_s": 19 * 3600.0,
        "done_sentinel_age_s": 19.6 * 3600.0,
        "gpu_util_max": 0.0,
        "missed": 1,
        "threshold": 2,
        "sentinel_floor_s": float(POD_SENT_S),
        "util_floor_s": float(POD_UTIL_S),
        "first_ts": None,
        "last_alert_ts": None,
        "now": now,
        "realert_s": float(REALERT_S),
    }
    kwargs.update(overrides)
    return kwargs


def test_pod_idle_sentinel_tier_fires_confirmed():
    """The #1739 evidence shape (sentinel + stale logs + measured 0%) at the
    confirming tick -> ("escalate", 2)."""
    now = time.time()
    assert asw.decide_keep_running_pod_idle_escalation(**_pod_decide_kwargs(now)) == (
        "escalate",
        2,
    )


def test_pod_idle_sentinel_tier_fires_with_unreadable_util():
    """Sentinel tier accepts util None (CPU pod / errored nvidia-smi) — the
    sentinel + stale-log conjunction is the strong evidence."""
    now = time.time()
    assert asw.decide_keep_running_pod_idle_escalation(
        **_pod_decide_kwargs(now, gpu_util_max=None)
    ) == ("escalate", 2)


def test_pod_idle_boundary_ages_equal_floors_fire():
    """Owner-leg boundary parity: age == floor COUNTS as idle (>= fires) on
    BOTH tiers."""
    now = time.time()
    assert asw.decide_keep_running_pod_idle_escalation(
        **_pod_decide_kwargs(
            now,
            log_write_age_s=float(POD_SENT_S),
            done_sentinel_age_s=float(POD_SENT_S),
        )
    ) == ("escalate", 2)
    assert asw.decide_keep_running_pod_idle_escalation(
        **_pod_decide_kwargs(
            now,
            done_sentinel_age_s=None,
            log_write_age_s=float(POD_UTIL_S),
        )
    ) == ("escalate", 2)


def test_pod_idle_clears_on_activity_evidence():
    """Positive activity ends the episode: measured util > 0 (whatever the
    ages say); a fresh log write under the applicable floor (the reused-pod
    carve-out — sentinel tier AND utilization tier); a young done-sentinel
    within the reuse grace."""
    now = time.time()
    assert asw.decide_keep_running_pod_idle_escalation(
        **_pod_decide_kwargs(now, gpu_util_max=35.0)
    ) == ("clear", 0)
    assert asw.decide_keep_running_pod_idle_escalation(
        **_pod_decide_kwargs(now, log_write_age_s=1 * 3600.0)
    ) == ("clear", 0)
    assert asw.decide_keep_running_pod_idle_escalation(
        **_pod_decide_kwargs(now, done_sentinel_age_s=2 * 3600.0, log_write_age_s=2 * 3600.0)
    ) == ("clear", 0)
    # Utilization tier: logs written within the 12h floor -> clear.
    assert asw.decide_keep_running_pod_idle_escalation(
        **_pod_decide_kwargs(now, done_sentinel_age_s=None, log_write_age_s=5 * 3600.0)
    ) == ("clear", 0)


def test_pod_idle_none_util_cannot_fire_utilization_tier():
    """The plan-pinned tier guard: with NO done-sentinel, a None (unreadable)
    util can NOT fire — hold with the counter FROZEN (neither incremented
    nor reset), however stale the logs."""
    now = time.time()
    assert asw.decide_keep_running_pod_idle_escalation(
        **_pod_decide_kwargs(
            now, done_sentinel_age_s=None, gpu_util_max=None, log_write_age_s=30 * 3600.0
        )
    ) == ("hold", 1)


def test_pod_idle_unreadable_log_freezes_both_tiers():
    """log_write_age_s None freezes on the sentinel tier AND on the
    utilization tier (fail toward no-fire)."""
    now = time.time()
    assert asw.decide_keep_running_pod_idle_escalation(
        **_pod_decide_kwargs(now, log_write_age_s=None)
    ) == ("hold", 1)
    assert asw.decide_keep_running_pod_idle_escalation(
        **_pod_decide_kwargs(now, done_sentinel_age_s=None, log_write_age_s=None)
    ) == ("hold", 1)


def test_pod_idle_below_threshold_accumulates_then_realerts():
    """missed 0 -> ("hold", 1) below threshold; an OPEN episode re-alerts at
    the 24h TTL and holds inside it."""
    now = time.time()
    assert asw.decide_keep_running_pod_idle_escalation(**_pod_decide_kwargs(now, missed=0)) == (
        "hold",
        1,
    )
    assert asw.decide_keep_running_pod_idle_escalation(
        **_pod_decide_kwargs(now, first_ts=now - 25 * 3600.0, last_alert_ts=now - 25 * 3600.0)
    ) == ("re-alert", 2)
    assert asw.decide_keep_running_pod_idle_escalation(
        **_pod_decide_kwargs(now, first_ts=now - 3600.0, last_alert_ts=now - 3600.0)
    ) == ("hold", 2)


# ---------------------------------------------------------------------------
# 7. #2149 probe producer/parser pins (Must-Fix 3)
# ---------------------------------------------------------------------------


def test_pod_idle_parser_nominal_and_variants():
    """Fixture lines for every probe variant: nominal GPU pod, no-done-file,
    CPU pod (gpu_rc=127), and a garbage numeric field mapping to None."""
    parsed = asw._parse_pod_idleness_line("log_age=68400 sentinel_age=70560 gpu_util=0 gpu_rc=0")
    assert parsed == {
        "log_write_age_s": 68400.0,
        "done_sentinel_age_s": 70560.0,
        "gpu_util_max": 0.0,
        "gpu_rc": 0,
    }
    parsed = asw._parse_pod_idleness_line("log_age=120 sentinel_age=na gpu_util=97 gpu_rc=0")
    assert parsed["done_sentinel_age_s"] is None
    assert parsed["gpu_util_max"] == 97.0
    parsed = asw._parse_pod_idleness_line("log_age=68400 sentinel_age=70560 gpu_util=na gpu_rc=127")
    assert parsed["gpu_util_max"] is None
    assert parsed["gpu_rc"] == 127
    parsed = asw._parse_pod_idleness_line("log_age=abc sentinel_age=70560 gpu_util=0 gpu_rc=0")
    assert parsed is not None and parsed["log_write_age_s"] is None


def test_pod_idle_parser_rejects_garbage_line():
    """A None/empty/garbage/missing-key LINE parses to None (whole-probe
    unknown -> freeze) — never a silently-zeroed dict."""
    assert asw._parse_pod_idleness_line(None) is None
    assert asw._parse_pod_idleness_line("") is None
    assert asw._parse_pod_idleness_line("   ") is None
    assert asw._parse_pod_idleness_line("Connection closed by remote host") is None
    assert asw._parse_pod_idleness_line("log_age=5 sentinel_age=na gpu_util=0") is None  # no gpu_rc


def test_pod_idle_producer_pins_remote_cmd_and_argv():
    """Producer<->parser pin: the composed remote command emits EXACTLY the
    parser's required keys (one printf, `k=%s` per field) and probes the
    plan-pinned surfaces; the ssh argv carries BatchMode + ConnectTimeout
    and ends with the remote command. A quoting/format drift must fail
    HERE, never degrade silently to a permanent freeze."""
    cmd = asw._POD_IDLE_PROBE_REMOTE_CMD
    for key in ("log_age", "sentinel_age", "gpu_util", "gpu_rc"):
        assert f"{key}=%s" in cmd, f"probe printf lost the {key} field"
    assert "/workspace/logs" in cmd
    assert "*done*.json" in cmd
    assert "nvidia-smi" in cmd
    argv = asw._pod_idleness_probe_argv("pod-1739-a1apilot")
    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv
    assert "ConnectTimeout=10" in argv
    assert argv[-2] == "pod-1739-a1apilot"
    assert argv[-1] == cmd


def test_pod_idle_probe_shell_roundtrip(tmp_path):
    """PRODUCTION-BODY roundtrip (no ssh): the remote sh snippet, pointed at
    a fixture logs dir, emits a line the PURE parser accepts — the #1739
    shape (terminal done-sentinel + stale logs) parses to the firing
    evidence; a missing logs dir parses to all-None fields (freeze)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    sent = logs / "issue-1739-a1apilot-done.json"
    sent.write_text(json.dumps({"phase": "done", "status": "ok"}))
    old = time.time() - 19 * 3600
    import os

    os.utime(sent, (old, old))
    cmd = asw._POD_IDLE_PROBE_REMOTE_CMD.replace("/workspace/logs", str(logs))
    res = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    parsed = asw._parse_pod_idleness_line(res.stdout.strip())
    assert parsed is not None
    assert 18 * 3600 < parsed["done_sentinel_age_s"] < 21 * 3600
    assert 18 * 3600 < parsed["log_write_age_s"] < 21 * 3600
    # Missing logs dir: every field na -> parsed but all-None (freeze downstream).
    cmd_absent = asw._POD_IDLE_PROBE_REMOTE_CMD.replace("/workspace/logs", str(tmp_path / "nope"))
    res = subprocess.run(["sh", "-c", cmd_absent], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0
    parsed = asw._parse_pod_idleness_line(res.stdout.strip())
    assert parsed is not None
    assert parsed["log_write_age_s"] is None
    assert parsed["done_sentinel_age_s"] is None


def test_pod_idle_env_overrides_and_malformed_fallback(monkeypatch):
    """The two HOURS floors, the probe-fail row count, and the leg flag are
    honored; malformed / non-positive values fall back to the defaults."""
    for var in (
        "EPM_KEEP_RUNNING_POD_IDLE_MIN_H",
        "EPM_KEEP_RUNNING_POD_UTIL_IDLE_MIN_H",
        "EPM_KEEP_RUNNING_POD_PROBE_FAIL_ROWS",
        "EPM_DISABLE_KEEP_RUNNING_POD_IDLE",
    ):
        monkeypatch.delenv(var, raising=False)
    assert asw._keep_running_pod_idle_s() == float(POD_SENT_S) == 4 * 3600.0
    assert asw._keep_running_pod_util_idle_s() == float(POD_UTIL_S) == 12 * 3600.0
    assert asw._keep_running_pod_probe_fail_rows() == asw.KEEP_RUNNING_POD_PROBE_FAIL_ROWS == 6
    assert asw._keep_running_pod_idle_enabled() is True
    monkeypatch.setenv("EPM_KEEP_RUNNING_POD_IDLE_MIN_H", "2")
    assert asw._keep_running_pod_idle_s() == 2 * 3600.0
    monkeypatch.setenv("EPM_KEEP_RUNNING_POD_UTIL_IDLE_MIN_H", "24")
    assert asw._keep_running_pod_util_idle_s() == 24 * 3600.0
    monkeypatch.setenv("EPM_KEEP_RUNNING_POD_PROBE_FAIL_ROWS", "3")
    assert asw._keep_running_pod_probe_fail_rows() == 3
    monkeypatch.setenv("EPM_DISABLE_KEEP_RUNNING_POD_IDLE", "1")
    assert asw._keep_running_pod_idle_enabled() is False
    for bad in ("garbage", "", "0", "-5"):
        monkeypatch.setenv("EPM_KEEP_RUNNING_POD_IDLE_MIN_H", bad)
        monkeypatch.setenv("EPM_KEEP_RUNNING_POD_UTIL_IDLE_MIN_H", bad)
        monkeypatch.setenv("EPM_KEEP_RUNNING_POD_PROBE_FAIL_ROWS", bad)
        assert asw._keep_running_pod_idle_s() == float(POD_SENT_S)
        assert asw._keep_running_pod_util_idle_s() == float(POD_UTIL_S)
        assert asw._keep_running_pod_probe_fail_rows() == 6


def test_pod_idle_sentinel_in_watcher_note_sentinels():
    """Self-reset protection for the pod-idle marker: membership + the
    substring exclusion from the real-progress clock (the #1667 lesson)."""
    assert asw._KEEP_RUNNING_POD_IDLE_NOTE_SENTINEL in asw._WATCHER_NOTE_SENTINELS
    now = time.time()
    events = [
        {
            "kind": "epm:progress",
            "ts": _iso(now),
            "note": f"{asw._KEEP_RUNNING_POD_IDLE_NOTE_SENTINEL} KEEP-RUNNING IDLE POD "
            "(#2149): escalation record",
            "by": "autonomous_session_watch",
        }
    ]
    assert asw._latest_progress_ts(events) is None


# ---------------------------------------------------------------------------
# 8. #2149 state plumbing (_save_pod_safety_state kr_pod sub-dict)
# ---------------------------------------------------------------------------


def test_kr_pod_sibling_forward_carry_and_gc(isolated_registry):
    """Must-Fix 1 state contract at the save layer: a save for pod B (any
    arm's, any pod's) forward-carries pod A's kr_pod entry VERBATIM — the
    top-level pod_id flip that resets the singular kr_owner_* fields must
    NOT touch the sub-dict — and the keep-ids GC drops only dead pods."""
    entry_a = {
        "missed": 1,
        "first_ts": None,
        "last_alert_ts": None,
        "probe_fails": 0,
        "probe_fail_noted": False,
    }
    asw._save_pod_safety_state(
        1739, "podA", missed=0, alerted=False, last_progress_ts=None, kr_pod_entry=("podA", entry_a)
    )
    state = json.loads((isolated_registry / "pod-safety-1739.json").read_text())
    assert state["kr_pod"] == {"podA": entry_a}

    # Busy sibling pod B's status-class-shaped save (no kr_pod kwargs): A's
    # entry survives verbatim even though top-level pod_id flips to podB.
    asw._save_pod_safety_state(
        1739, "podB", missed=0, alerted=False, last_progress_ts=None, prev=state
    )
    state = json.loads((isolated_registry / "pod-safety-1739.json").read_text())
    assert state["pod_id"] == "podB"
    assert state["kr_pod"] == {"podA": entry_a}

    # B's own arm save adds its entry beside A's; the keep-ids GC then drops
    # a pod that left the RUNNING set.
    entry_b = dict(entry_a, missed=0)
    asw._save_pod_safety_state(
        1739,
        "podB",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        kr_pod_entry=("podB", entry_b),
        kr_pod_keep_ids={"podA", "podB"},
        prev=state,
    )
    state = json.loads((isolated_registry / "pod-safety-1739.json").read_text())
    assert state["kr_pod"] == {"podA": entry_a, "podB": entry_b}
    asw._save_pod_safety_state(
        1739,
        "podB",
        missed=0,
        alerted=False,
        last_progress_ts=None,
        kr_pod_entry=("podB", entry_b),
        kr_pod_keep_ids={"podB"},
        prev=state,
    )
    state = json.loads((isolated_registry / "pod-safety-1739.json").read_text())
    assert state["kr_pod"] == {"podB": entry_b}


def test_kr_pod_state_back_compat(isolated_registry):
    """A pre-#2149 state file without the kr_pod key parses fine: the save
    layer defaults it to {} and later saves carry it forward."""
    (isolated_registry / "pod-safety-1739.json").write_text(
        json.dumps(
            {
                "pod_id": "podA",
                "missed": 0,
                "alerted": False,
                "last_progress_ts": None,
                "first_seen": time.time() - 7200,
            }
        )
    )
    prev = json.loads((isolated_registry / "pod-safety-1739.json").read_text())
    assert "kr_pod" not in prev
    asw._save_pod_safety_state(
        1739, "podA", missed=0, alerted=False, last_progress_ts=None, prev=prev
    )
    state = json.loads((isolated_registry / "pod-safety-1739.json").read_text())
    assert state["kr_pod"] == {}


# ---------------------------------------------------------------------------
# 9. #2149 integration — the #1739 multi-pod replay through _process_pod
# ---------------------------------------------------------------------------


@pytest.fixture
def rig_1739(monkeypatch, isolated_registry):
    """The #1739 replay rig: a BUSY multi-round task (a fresh sibling-round
    marker lands before every tick, so the owner leg's 12h gap NEVER opens —
    the incident had 129 markers, largest gap 6.19h), keep-running tag, TWO
    RUNNING pods processed interleaved per tick: idle pod A (done-sentinel
    19.6h, logs 19h stale, measured 0% util) and busy pod B (fresh logs,
    97% util). ``_probe_pod_idleness`` is seam-stubbed per pod NAME (the SSH
    boundary; the production probe body is covered by the shell-roundtrip
    test); every other helper is the rig_1345 convention."""
    t0 = time.time()
    clock = {"now": t0}
    events = [
        {
            "kind": "epm:status-changed",
            "ts": _iso(t0 - 30 * 3600),
            "note": "verifying -> awaiting_promotion",
            "by": "task-workflow",
        },
    ]
    posts: list[tuple[int, str, str | None, bool]] = []
    pushes: list[tuple[str, bool]] = []
    stops: list[int] = []
    sidecar: list[dict] = []
    probe_calls: list[str] = []
    probes: dict[str, str | None] = {
        "pod-1739-a1apilot": "log_age=68400 sentinel_age=70560 gpu_util=0 gpu_rc=0",
        "pod-1739-r2fair": "log_age=120 sentinel_age=na gpu_util=97 gpu_rc=0",
    }

    def _post_stub(issue, note, dry_run, label=None):
        posts.append((issue, note, label, dry_run))
        events.append(
            {
                "kind": "epm:progress",
                "ts": _iso(clock["now"]),
                "note": note,
                "by": "autonomous_session_watch",
            }
        )

    def _probe_stub(pod_name):
        probe_calls.append(pod_name)
        return probes.get(pod_name)

    monkeypatch.setattr(asw, "_task_status", lambda issue: "awaiting_promotion")
    monkeypatch.setattr(asw, "_task_events", lambda issue: events)
    monkeypatch.setattr(asw, "_task_keep_running", lambda issue: True)
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None, **_kw: False)
    monkeypatch.setattr(asw, "_post_progress_marker", _post_stub)
    monkeypatch.setattr(asw, "_telegram_push", lambda msg, dry_run: pushes.append((msg, dry_run)))
    monkeypatch.setattr(asw, "_stop_pod", lambda issue, dry_run: stops.append(issue) or True)
    monkeypatch.setattr(asw, "_probe_pod_idleness", _probe_stub)
    monkeypatch.setattr(
        asw,
        "_append_keep_running_wedged_event",
        lambda payload, dry_run: sidecar.append(payload),
    )
    # Owner-leg vetoes stay quiet; on the busy task the owner leg early-clears
    # before ANY daemon probe, so none are stubbed (a daemon call would fail
    # loudly under the conftest hermeticity guards if it ever ran).
    monkeypatch.setattr(asw, "_provision_in_flight_reason", lambda issue, now: None)
    monkeypatch.setattr(asw, "_worktree_recent_activity", lambda issue, now, window_s, **kw: False)

    pods = {
        "a": _info(pod_id="podA", name="pod-1739-a1apilot", created_at=_iso(t0 - 30 * 3600)),
        "b": _info(pod_id="podB", name="pod-1739-r2fair", created_at=_iso(t0 - 30 * 3600)),
    }

    def tick(which: str, offset_s: float = 0.0, *, dry_run: bool = False, threshold: int = 2):
        clock["now"] = t0 + offset_s
        # The BUSY task: a sibling round posts a REAL marker ~5 min before
        # every tick, keeping the owner leg's progress gap << 12h forever.
        events.append(
            {
                "kind": "epm:progress",
                "ts": _iso(clock["now"] - 300),
                "note": f"sibling round marker at +{offset_s:.0f}s",
                "by": "orchestrator",
            }
        )
        info = pods[which]
        asw._process_pod(
            1739,
            info.pod_id,
            info,
            clock["now"],
            dry_run=dry_run,
            threshold=threshold,
            issue_running_pod_ids={"podA", "podB"},
        )

    return SimpleNamespace(
        t0=t0,
        clock=clock,
        events=events,
        posts=posts,
        pushes=pushes,
        stops=stops,
        sidecar=sidecar,
        probes=probes,
        probe_calls=probe_calls,
        state_path=isolated_registry / "pod-safety-1739.json",
        registry=isolated_registry,
        tick=tick,
    )


def _idle_pod_posts(posts):
    return [p for p in posts if asw._KEEP_RUNNING_POD_IDLE_NOTE_SENTINEL in p[1]]


def test_1739_multipod_interleaved_fires_on_idle_pod_only(rig_1739):
    """THE acceptance-criterion-1/4 regression (Must-Fix 1 shape): a busy
    task (largest marker gap << 12h) with TWO pods interleaved per tick —
    tick1: A holds (missed=1), B clears; tick2: A FIRES (ONE marker + ONE
    push + ONE sidecar row), B still clear and NEVER fires; A's counter
    SURVIVES B's saves; a +24h tick re-pushes with NO second marker;
    nothing is ever stopped."""
    rig = rig_1739
    rig.tick("a", 0)  # A: hold (missed 0 -> 1)
    rig.tick("b", 30)  # B: clear — must NOT reset A's counter
    assert _idle_pod_posts(rig.posts) == []
    assert rig.pushes == []
    assert rig.sidecar == []
    state = json.loads(rig.state_path.read_text())
    assert state["kr_pod"]["podA"]["missed"] == 1

    rig.tick("a", 600)  # A: confirming tick -> escalate
    fired = _idle_pod_posts(rig.posts)
    assert len(fired) == 1
    issue, note, label, dry_run = fired[0]
    assert issue == 1739
    assert label == "keep-running-idle-pod"
    assert dry_run is False
    assert "pod-1739-a1apilot" in note
    assert "sentinel tier" in note
    assert "NOT auto-stopped" in note
    assert "remove-tag 1739 keep-running" in note
    assert len(rig.pushes) == 1
    assert "IDLE" in rig.pushes[0][0]
    assert len(rig.sidecar) == 1
    row = rig.sidecar[0]
    assert row["kind"] == "keep-running-idle-pod"
    assert row["leg"] == "pod-idle"
    assert row["pod_id"] == "podA"
    assert row["tier"] == "sentinel"
    assert row["gpu_util_max"] == 0.0
    state = json.loads(rig.state_path.read_text())
    assert state["kr_pod"]["podA"]["missed"] == 2
    assert state["kr_pod"]["podA"]["first_ts"] == pytest.approx(rig.t0 + 600)

    rig.tick("b", 630)  # B: still clear, still no reset of A's episode
    rig.tick("a", 1200)  # A: within the 24h TTL — nothing new
    assert len(_idle_pod_posts(rig.posts)) == 1
    assert len(rig.pushes) == 1
    assert len(rig.sidecar) == 1
    state = json.loads(rig.state_path.read_text())
    assert state["kr_pod"]["podA"]["first_ts"] == pytest.approx(rig.t0 + 600)

    rig.tick("a", 600 + float(REALERT_S))  # +24h: re-push + sidecar, NO 2nd marker
    assert len(_idle_pod_posts(rig.posts)) == 1
    assert len(rig.pushes) == 2
    assert len(rig.sidecar) == 2
    # B never fired anything, and nothing was ever stopped/terminated.
    assert not any("pod-1739-r2fair" in p[1] for p in _idle_pod_posts(rig.posts))
    assert rig.stops == []
    # The owner leg never fired on the busy task (its markers stayed fresh).
    assert _wedged_posts(rig.posts) == []


def test_1739_pod_idle_never_stops_or_terminates(rig_1739, monkeypatch):
    """Alert-only invariant on the pod leg's FIRING tick, mirroring
    test_never_stops_or_terminates: _stop_pod never called, no `pod.py`
    subprocess argv ever constructed."""
    rig = rig_1739
    argvs: list[list[str]] = []
    real_run = subprocess.run

    def _record_run(argv, *a, **kw):  # pragma: no cover - never expected to run
        argvs.append(list(argv) if isinstance(argv, (list, tuple)) else [str(argv)])
        raise AssertionError(f"unexpected subprocess.run during the rig ticks: {argv}")

    monkeypatch.setattr(subprocess, "run", _record_run)
    try:
        rig.tick("a", 0)
        rig.tick("a", 600)  # the firing tick
    finally:
        monkeypatch.setattr(subprocess, "run", real_run)
    assert len(_idle_pod_posts(rig.posts)) == 1  # the leg fired...
    assert rig.stops == []  # ...and nothing was stopped
    assert not any(any("pod.py" in tok for tok in argv) for argv in argvs)


def test_quiet_task_live_owner_pod_leg_still_fires(rig_1345, monkeypatch):
    """Must-Fix 2 wiring pin (the OTHER direction of marker-traffic
    independence): a QUIET task (gap 71h >= 12h) whose owner reads LIVE —
    the owner leg clears at its decide — still runs the pod leg from the
    post-owner-decide non-fire path: hold tick1, FIRE tick2."""
    rig = rig_1345
    monkeypatch.setattr(
        asw, "_keep_running_owner_state", lambda issue, now, min_idle_s: ("live", {})
    )
    monkeypatch.setattr(
        asw,
        "_probe_pod_idleness",
        lambda pod_name: "log_age=68400 sentinel_age=70560 gpu_util=0 gpu_rc=0",
    )
    rig.tick(0)
    assert _idle_pod_posts(rig.posts) == []
    rig.tick(600)
    fired = _idle_pod_posts(rig.posts)
    assert len(fired) == 1
    assert "sentinel tier" in fired[0][1]
    assert _wedged_posts(rig.posts) == []  # the owner leg never fired
    assert rig.stops == []


def test_owner_leg_fire_skips_pod_leg(rig_1345, monkeypatch):
    """When the OWNER leg escalates on a tick, the pod leg is SKIPPED (one
    alert per pod per tick); on the earlier hold tick it still ran. Owner
    behavior itself is byte-unchanged (the existing suite pins it)."""
    rig = rig_1345
    probe_calls: list[str] = []
    monkeypatch.setattr(
        asw,
        "_probe_pod_idleness",
        lambda pod_name: probe_calls.append(pod_name) or None,  # freeze — no emission
    )
    rig.tick(0)  # owner leg holds (missed 1) -> pod leg runs (probe called)
    assert len(probe_calls) == 1
    rig.tick(600)  # owner leg ESCALATES -> pod leg skipped (no new probe)
    assert len(_wedged_posts(rig.posts)) == 1
    assert len(probe_calls) == 1
    assert _idle_pod_posts(rig.posts) == []


def test_pod_idle_probe_failure_freezes_and_notes_once(rig_1739, monkeypatch):
    """Probe unreachable -> the counter FREEZES (no fire however many ticks)
    and after EPM_KEEP_RUNNING_POD_PROBE_FAIL_ROWS consecutive failures ONE
    durable sidecar row records the permanent freeze (no marker, no push);
    further failures stay silent."""
    rig = rig_1739
    monkeypatch.setenv("EPM_KEEP_RUNNING_POD_PROBE_FAIL_ROWS", "3")
    rig.probes["pod-1739-a1apilot"] = None  # SSH probe fails every tick
    for i in range(5):
        rig.tick("a", i * 600)
    assert _idle_pod_posts(rig.posts) == []
    assert rig.pushes == []
    fail_rows = [r for r in rig.sidecar if r.get("kind") == "keep-running-idle-pod-probe-fail"]
    assert len(fail_rows) == 1  # exactly ONE per episode, at the 3rd failure
    assert fail_rows[0]["consecutive_probe_failures"] == 3
    state = json.loads(rig.state_path.read_text())
    assert state["kr_pod"]["podA"]["missed"] == 0  # frozen at 0, never accumulated
    assert state["kr_pod"]["podA"]["probe_fails"] == 5
    assert state["kr_pod"]["podA"]["probe_fail_noted"] is True
    # A successful probe resets the failure episode and resumes the decision.
    rig.probes["pod-1739-a1apilot"] = "log_age=68400 sentinel_age=70560 gpu_util=0 gpu_rc=0"
    rig.tick("a", 6 * 600)
    state = json.loads(rig.state_path.read_text())
    assert state["kr_pod"]["podA"]["probe_fails"] == 0
    assert state["kr_pod"]["podA"]["probe_fail_noted"] is False
    assert state["kr_pod"]["podA"]["missed"] == 1


def test_pod_idle_young_pod_preveto_skips_probe(rig_1739, monkeypatch):
    """A pod younger than the smaller tier floor cannot be floor-idle: the
    SSH probe is skipped entirely (lazy-cost discipline)."""
    rig = rig_1739
    young = _info(pod_id="podC", name="pod-1739-young", created_at=_iso(rig.t0 - 3600))
    rig.events.append(
        {
            "kind": "epm:progress",
            "ts": _iso(rig.t0 - 300),
            "note": "sibling round marker",
            "by": "orchestrator",
        }
    )
    asw._process_pod(
        1739,
        "podC",
        young,
        rig.t0,
        dry_run=False,
        threshold=2,
        issue_running_pod_ids={"podC"},
    )
    assert rig.probe_calls == []
    assert _idle_pod_posts(rig.posts) == []


def test_pod_idle_leg_kill_switch_disables_only_pod_leg(rig_1739, monkeypatch):
    """EPM_DISABLE_KEEP_RUNNING_POD_IDLE=1 disables ONLY the pod leg: no
    probe, no emission on the #1739 rig (the owner leg's own behavior is
    pinned by test_owner_leg_unaffected_by_pod_leg_flag)."""
    rig = rig_1739
    monkeypatch.setenv("EPM_DISABLE_KEEP_RUNNING_POD_IDLE", "1")
    rig.tick("a", 0)
    rig.tick("a", 600)
    assert rig.probe_calls == []
    assert _idle_pod_posts(rig.posts) == []


def test_owner_leg_unaffected_by_pod_leg_flag(rig_1345, monkeypatch):
    """The leg flag leaves the OWNER leg byte-unchanged: the #1345 replay
    still escalates on its confirming tick with the flag set."""
    rig = rig_1345
    monkeypatch.setenv("EPM_DISABLE_KEEP_RUNNING_POD_IDLE", "1")
    probe_calls: list[str] = []
    monkeypatch.setattr(
        asw, "_probe_pod_idleness", lambda pod_name: probe_calls.append(pod_name) or None
    )
    rig.tick(0)
    rig.tick(600)
    assert len(_wedged_posts(rig.posts)) == 1  # owner leg fired exactly as before
    assert probe_calls == []  # pod leg fully disabled


def test_arm_wide_kill_switch_disables_both_legs(rig_1739, monkeypatch):
    """EPM_DISABLE_KEEP_RUNNING_OWNER_AUDIT=1 covers BOTH legs (checked
    first): no probe, no emission of either sentinel."""
    rig = rig_1739
    monkeypatch.setenv("EPM_DISABLE_KEEP_RUNNING_OWNER_AUDIT", "1")
    rig.tick("a", 0)
    rig.tick("a", 600)
    assert rig.probe_calls == []
    assert _idle_pod_posts(rig.posts) == []
    assert _wedged_posts(rig.posts) == []


def test_pod_idle_dry_run_no_writes(rig_1739):
    """Dry-run on a would-fire tick: marker/push helpers invoked with
    dry_run=True and NO state-file write."""
    rig = rig_1739
    rig.tick("a", 0, dry_run=True)
    rig.tick("a", 600, dry_run=True)
    # Dry-run never persisted the tick-1 hold, so missed never reached the
    # threshold and nothing fired — and no state file exists at all.
    assert not rig.state_path.exists()
    assert _idle_pod_posts(rig.posts) == []


def test_pod_idle_state_back_compat_through_arm(rig_1739):
    """A pre-#2149 state file (no kr_pod key) is read fine by the arm: the
    first tick holds (missed=1) and writes the new sub-dict beside the
    legacy fields."""
    rig = rig_1739
    rig.state_path.write_text(
        json.dumps(
            {
                "pod_id": "podA",
                "missed": 0,
                "alerted": False,
                "last_progress_ts": None,
                "first_seen": rig.t0 - 7200,
                "keep_running_noted": True,
            }
        )
    )
    rig.tick("a", 0)
    state = json.loads(rig.state_path.read_text())
    assert state["kr_pod"]["podA"]["missed"] == 1
    assert state["keep_running_noted"] is True
