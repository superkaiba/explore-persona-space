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
    monkeypatch.setattr(asw, "_task_followup_active", lambda issue, events=None: False)
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
