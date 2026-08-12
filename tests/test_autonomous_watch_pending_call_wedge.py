"""Pending-call wedge observer pass (#2115) — predicate + pass-level pins.

The pass is the PRIMARY deliverable of task #2115: ~12-18 autonomous
sessions stalled 1.2-2.4h at a Step 10d Bash dispatch whose tool_result
never arrived; the mechanism sits outside the repo, so DETECTION is the
product. Two properties are load-bearing and pinned here:

1. **Bash-only keying** (plan v3 §6 prong 1): the Agent tool legitimately
   pends 30-90+ min in nearly every healthy autonomous session, so a
   pending non-Bash block must NEVER fire the lane — fixture (e) is a
   required fixture, not an optional one.
2. **Fail-toward-silence**: an escalate-only lane whose parse bugs spam
   the fleet gets kill-switched; every malformed input returns None —
   fixture (d).

Pass-level pins: kill switch, missing pid-map skip, sidecar + push +
per-(issue, tool_use_id) episode dedup, and the ESCALATE-ONLY invariant
(no task markers, no session stops, no git mutation — the pass has no
code path that could; the sidecar/state/push seams are the only writes).
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# scripts/ holds autonomous_session_watch.py (and its spawn_session import).
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402

NOW = 1_760_000_000.0
WINDOW_S = 25 * 60.0


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat().replace("+00:00", "Z")


def _assistant_row(ts_epoch: float, blocks: list) -> dict:
    return {
        "type": "assistant",
        "timestamp": _iso(ts_epoch),
        "message": {"content": blocks},
    }


def _tool_use(tid: str, name: str) -> dict:
    return {"type": "tool_use", "id": tid, "name": name, "input": {}}


def _tool_result_row(tid: str) -> dict:
    return {
        "type": "user",
        "timestamp": _iso(NOW),
        "message": {"content": [{"type": "tool_result", "tool_use_id": tid}]},
    }


def _pending_bash_rows(age_s: float, tid: str = "toolu_bash1") -> list[dict]:
    """The would-flag shape: tail ends in an assistant Bash tool_use with no
    matching tool_result."""
    return [_assistant_row(NOW - age_s, [_tool_use(tid, "Bash")])]


# ─── decide_pending_call_wedge: the five plan-§7 fixtures ────────────────────


def test_a_pending_bash_older_than_window_flags():
    hit = asw.decide_pending_call_wedge(_pending_bash_rows(30 * 60), window_s=WINDOW_S, now=NOW)
    assert hit is not None
    assert hit["tool_use_id"] == "toolu_bash1"
    assert hit["n_pending"] == 1
    assert abs(hit["age_s"] - 30 * 60) < 1.0


def test_b_pending_bash_inside_window_not_flagged():
    hit = asw.decide_pending_call_wedge(_pending_bash_rows(10 * 60), window_s=WINDOW_S, now=NOW)
    assert hit is None


def test_c_matched_tool_result_not_flagged():
    rows = [*_pending_bash_rows(30 * 60), _tool_result_row("toolu_bash1")]
    assert asw.decide_pending_call_wedge(rows, window_s=WINDOW_S, now=NOW) is None


def test_d_malformed_transcript_fails_toward_silence():
    """Fixture (d): every malformed input returns None — never raises,
    never flags (escalate-only lane; a parse bug must not spam)."""
    old = NOW - 30 * 60
    malformed_inputs = [
        None,  # unresolvable transcript
        [],  # empty tail
        ["not a dict", 42],  # garbage rows
        [{"type": "assistant"}],  # no message
        [{"type": "assistant", "timestamp": _iso(old), "message": "not-a-dict"}],
        [{"type": "assistant", "timestamp": _iso(old), "message": {"content": "not-a-list"}}],
        # malformed tool_use block: id missing
        [_assistant_row(old, [{"type": "tool_use", "name": "Bash"}])],
        # malformed tool_use block: non-str name
        [_assistant_row(old, [{"type": "tool_use", "id": "toolu_x", "name": 7}])],
        # text-only turn (no tool_use at all)
        [_assistant_row(old, [{"type": "text", "text": "done"}])],
        # missing timestamp on the flagging row
        [{"type": "assistant", "message": {"content": [_tool_use("toolu_x", "Bash")]}}],
        # unparseable timestamp
        [
            {
                "type": "assistant",
                "timestamp": "not-a-ts",
                "message": {"content": [_tool_use("toolu_x", "Bash")]},
            }
        ],
    ]
    for rows in malformed_inputs:
        assert asw.decide_pending_call_wedge(rows, window_s=WINDOW_S, now=NOW) is None, rows


def test_e_pending_agent_older_than_window_not_flagged():
    """Fixture (e) — REQUIRED: Bash-only keying is load-bearing. A pending
    Agent (or any non-Bash tool) past the window is routine health."""
    rows = [_assistant_row(NOW - 90 * 60, [_tool_use("toolu_agent1", "Agent")])]
    assert asw.decide_pending_call_wedge(rows, window_s=WINDOW_S, now=NOW) is None


def test_e2_mixed_pending_bash_plus_agent_not_flagged():
    """ANY pending non-Bash block suppresses — even alongside a pending
    Bash one (conservative: the turn is legitimately waiting on the Agent)."""
    rows = [
        _assistant_row(
            NOW - 90 * 60,
            [_tool_use("toolu_bash1", "Bash"), _tool_use("toolu_agent1", "Agent")],
        )
    ]
    assert asw.decide_pending_call_wedge(rows, window_s=WINDOW_S, now=NOW) is None


def test_exempt_tools_never_flag():
    """The other named exempt tools (AskUserQuestion / TaskOutput / Monitor)
    behave exactly like Agent under the typed keying."""
    for name in ("AskUserQuestion", "TaskOutput", "Monitor"):
        rows = [_assistant_row(NOW - 90 * 60, [_tool_use("toolu_x", name)])]
        assert asw.decide_pending_call_wedge(rows, window_s=WINDOW_S, now=NOW) is None, name


def test_resolved_bash_plus_pending_bash_flags_on_the_pending_one():
    rows = [
        _assistant_row(
            NOW - 40 * 60,
            [_tool_use("toolu_done", "Bash"), _tool_use("toolu_stuck", "Bash")],
        ),
        _tool_result_row("toolu_done"),
    ]
    hit = asw.decide_pending_call_wedge(rows, window_s=WINDOW_S, now=NOW)
    assert hit is not None
    assert hit["tool_use_id"] == "toolu_stuck"
    assert hit["n_pending"] == 1


def test_later_assistant_turn_resets_the_read():
    """Only the LAST assistant row is judged: an earlier pending Bash with a
    LATER text-only assistant turn means the session moved on."""
    rows = [
        _assistant_row(NOW - 90 * 60, [_tool_use("toolu_old", "Bash")]),
        _assistant_row(NOW - 5 * 60, [{"type": "text", "text": "recovered"}]),
    ]
    assert asw.decide_pending_call_wedge(rows, window_s=WINDOW_S, now=NOW) is None


# ─── pending_call_wedge_pass: pass-level wiring ──────────────────────────────


def _wire_pass(monkeypatch, tmp_path, rows, *, issue: int = 2115, sid: str = "sid-1"):
    """Standard pass harness: tmp-rooted sidecar/state, one registration,
    one live pid, a fixed transcript tail, and a push recorder."""
    monkeypatch.delenv("EPM_DISABLE_PENDING_CALL_WEDGE", raising=False)
    monkeypatch.delenv("EPM_PENDING_CALL_WEDGE_MIN", raising=False)
    monkeypatch.setattr(asw, "PROJECT_ROOT", tmp_path / "repo")
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path / "registry")
    monkeypatch.setattr(asw, "_issue_registrations", lambda: {issue: {"sids": {sid}}})
    monkeypatch.setattr(asw, "_transcript_tail_rows", lambda pid, max_bytes=262144: rows)
    pushes: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        asw, "_telegram_push", lambda msg, dry_run: (pushes.append((msg, dry_run)), True)[1]
    )
    return pushes


def test_pass_kill_switch_skips(monkeypatch, tmp_path):
    pushes = _wire_pass(monkeypatch, tmp_path, _pending_bash_rows(60 * 60))
    monkeypatch.setenv("EPM_DISABLE_PENDING_CALL_WEDGE", "1")
    assert asw.pending_call_wedge_pass(False, pids_by_sid={"sid-1": 4242}) is False
    assert pushes == []
    assert not asw._pending_call_sidecar_path().exists()
    assert not asw._pending_call_state_path().exists()


def test_pass_no_pid_map_skips(monkeypatch, tmp_path):
    pushes = _wire_pass(monkeypatch, tmp_path, _pending_bash_rows(60 * 60))
    assert asw.pending_call_wedge_pass(False, pids_by_sid=None) is False
    assert asw.pending_call_wedge_pass(False, pids_by_sid={}) is False
    assert pushes == []
    assert not asw._pending_call_sidecar_path().exists()


def test_pass_flags_writes_sidecar_and_pushes_once(monkeypatch, tmp_path):
    pushes = _wire_pass(monkeypatch, tmp_path, _pending_bash_rows(60 * 60))
    assert asw.pending_call_wedge_pass(False, pids_by_sid={"sid-1": 4242}) is True

    sidecar = asw._pending_call_sidecar_path()
    lines = [json.loads(ln) for ln in sidecar.read_text().splitlines()]
    assert len(lines) == 1
    row = lines[0]
    assert row["kind"] == "pending-call-wedge"
    assert row["issue"] == 2115
    assert row["sid"] == "sid-1"
    assert row["tool_use_id"] == "toolu_bash1"
    assert row["n_pending"] == 1
    assert row["pushed"] is True
    assert row["age_min"] >= 25.0

    assert len(pushes) == 1
    msg, dry = pushes[0]
    assert dry is False
    assert "PENDING-CALL WEDGE" in msg
    assert "#2115" in msg

    state = json.loads(asw._pending_call_state_path().read_text())
    assert "2115:toolu_bash1" in state["episodes"]


def test_pass_episode_dedup_second_tick_no_repush(monkeypatch, tmp_path):
    pushes = _wire_pass(monkeypatch, tmp_path, _pending_bash_rows(60 * 60))
    assert asw.pending_call_wedge_pass(False, pids_by_sid={"sid-1": 4242}) is True
    assert asw.pending_call_wedge_pass(False, pids_by_sid={"sid-1": 4242}) is True

    # Two sidecar rows (one per flagged tick), exactly ONE push (episode dedup
    # keyed on issue:tool_use_id), second row records pushed=False.
    lines = [json.loads(ln) for ln in asw._pending_call_sidecar_path().read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["pushed"] is True
    assert lines[1]["pushed"] is False
    assert len(pushes) == 1


def test_pass_unresolvable_transcript_skips_silently(monkeypatch, tmp_path):
    pushes = _wire_pass(monkeypatch, tmp_path, None)
    assert asw.pending_call_wedge_pass(False, pids_by_sid={"sid-1": 4242}) is False
    assert pushes == []
    assert not asw._pending_call_sidecar_path().exists()


def test_pass_pending_agent_session_not_flagged(monkeypatch, tmp_path):
    """End-to-end twin of fixture (e): a registered session whose tail pends
    on an Agent call produces no sidecar row, no push, no state."""
    rows = [_assistant_row(NOW - 90 * 60, [_tool_use("toolu_agent1", "Agent")])]
    pushes = _wire_pass(monkeypatch, tmp_path, rows)
    assert asw.pending_call_wedge_pass(False, pids_by_sid={"sid-1": 4242}) is False
    assert pushes == []
    assert not asw._pending_call_sidecar_path().exists()


def test_pass_fail_soft_on_internal_error(monkeypatch, tmp_path, capsys):
    """A raising seam inside the try body must not take down the tick."""
    _wire_pass(monkeypatch, tmp_path, _pending_bash_rows(60 * 60))

    def _boom():
        raise RuntimeError("registration read exploded")

    monkeypatch.setattr(asw, "_issue_registrations", _boom)
    assert asw.pending_call_wedge_pass(False, pids_by_sid={"sid-1": 4242}) is False
    assert "fail-soft" in capsys.readouterr().err


def test_pass_is_in_conftest_fleet_mutating_stub_list():
    """Full-main() hermeticity (#1247 family): the pass reads live
    registrations + transcripts and writes real sidecar/state/pushes, so it
    must be stubbed by _stub_fleet_mutating_passes."""
    from tests.conftest import _FLEET_MUTATING_PASS_NAMES

    assert "pending_call_wedge_pass" in _FLEET_MUTATING_PASS_NAMES
