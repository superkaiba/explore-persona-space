"""Tests for the #1130 rc==0 post-marker child-stderr forwarding at the
secondary call sites: ``spawn_session._post_duplicate_suppressed_marker``
and ``autonomous_session_watch._forward_marker_child_stderr`` (plus its
``_post_progress_marker`` integration).

``task.py post-marker`` deliberately exits 0 while printing the
deferred-commit ERROR and the #1100 post-commit LANDING CHECK warning to
stderr; ``capture_output=True`` at these call sites used to discard both,
so they reached no transcript. The forwarding writes them to the wrapper's
stderr, prefixed and capped; control flow (rc handling, ``check=True``
semantics, return values) is unchanged. Primary-site coverage
(``codex_task._post_marker``) lives in ``tests/test_codex_task_post_marker.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
import file_infra_task  # noqa: E402
import spawn_session  # noqa: E402

_LANDING_CHECK_LINE = (
    "task.py LANDING CHECK: commit abc123 ('epm:x') is NOT reachable from refs/heads/main"
)


# ──────────────────────────────────────────────────────────────────────
# spawn_session._post_duplicate_suppressed_marker
# ──────────────────────────────────────────────────────────────────────


def test_spawn_session_duplicate_marker_forwards_nonempty_stderr(monkeypatch, capsys):
    """Non-empty child stderr → forwarded with the `[post-marker stderr]`
    prefix (rc deliberately unchecked — best-effort post, unchanged)."""
    calls = []
    monkeypatch.setattr(
        spawn_session.subprocess,
        "run",
        lambda *a, **k: (
            calls.append(a) or SimpleNamespace(returncode=0, stdout="", stderr=_LANDING_CHECK_LINE)
        ),
    )

    spawn_session._post_duplicate_suppressed_marker(1130, "sid-kept", "sid-stopped")

    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "[post-marker stderr]" in err
    assert "task.py LANDING CHECK" in err


def test_spawn_session_duplicate_marker_empty_stderr_silent(monkeypatch, capsys):
    """Empty child stderr (the common case) → zero new output."""
    monkeypatch.setattr(
        spawn_session.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    spawn_session._post_duplicate_suppressed_marker(1130, "sid-kept", "sid-stopped")

    assert capsys.readouterr().err == ""


# ──────────────────────────────────────────────────────────────────────
# autonomous_session_watch._forward_marker_child_stderr (unit)
# ──────────────────────────────────────────────────────────────────────


def test_watch_helper_forwards_nonempty_stderr(capsys):
    """Non-empty rc==0 stderr → per-line `[task.py stderr] {context}:` prefix."""
    asw._forward_marker_child_stderr(
        SimpleNamespace(returncode=0, stdout="", stderr=_LANDING_CHECK_LINE),
        "epm:progress on #1130",
    )

    err = capsys.readouterr().err
    assert "[task.py stderr] epm:progress on #1130:" in err
    assert "task.py LANDING CHECK" in err


def test_watch_helper_empty_or_missing_stderr_silent(capsys):
    """Empty / None / absent stderr all forward nothing and never raise."""
    asw._forward_marker_child_stderr(SimpleNamespace(stderr=""), "ctx")
    asw._forward_marker_child_stderr(SimpleNamespace(stderr=None), "ctx")
    asw._forward_marker_child_stderr(SimpleNamespace(), "ctx")  # no stderr attribute at all

    assert capsys.readouterr().err == ""


# ──────────────────────────────────────────────────────────────────────
# Integration: _post_progress_marker reaches the helper on rc==0.
# ──────────────────────────────────────────────────────────────────────


def test_watch_post_progress_marker_integration_forwards(monkeypatch, capsys):
    """dry_run=False (the dry-run branch returns before subprocess.run) with a
    stubbed rc==0 + non-empty-stderr child → the warning reaches the
    watcher's stderr; exactly one subprocess invocation (check=True
    semantics untouched — the stub does not raise)."""
    calls = []
    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda *a, **k: (
            calls.append((a, k))
            or SimpleNamespace(returncode=0, stdout="", stderr=_LANDING_CHECK_LINE)
        ),
    )

    asw._post_progress_marker(1130, "pod-safety note", False, label="auto-stop")

    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "[task.py stderr]" in err
    assert "task.py LANDING CHECK" in err


# ──────────────────────────────────────────────────────────────────────
# #1150: the three remaining mutating task.py children.
# autonomous_session_watch._set_status_blocked
# ──────────────────────────────────────────────────────────────────────


def test_set_status_blocked_forwards_rc0_stderr(monkeypatch, capsys):
    """rc==0 set-status child with non-empty stderr → returns True AND the
    warning is forwarded under the `set-status blocked on #<N>` context."""
    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=_LANDING_CHECK_LINE),
    )

    assert asw._set_status_blocked(1150, dry_run=False) is True
    err = capsys.readouterr().err
    assert "[task.py stderr] set-status blocked on #1150:" in err
    assert "task.py LANDING CHECK" in err


def test_set_status_blocked_rc_nonzero_unchanged(monkeypatch, capsys):
    """rc!=0 → returns False with the existing WARNING line; NO forwarding
    (the rc!=0 branch already prints the stderr detail — forwarding there
    would duplicate it)."""
    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom detail"),
    )

    assert asw._set_status_blocked(1150, dry_run=False) is False
    err = capsys.readouterr().err
    assert "WARNING: set-status blocked on #1150 FAILED (rc=1): boom detail" in err
    assert "[task.py stderr]" not in err


# ──────────────────────────────────────────────────────────────────────
# autonomous_session_watch._repark_completed_followup_round
# ──────────────────────────────────────────────────────────────────────


def test_repark_forwards_rc0_stderr(monkeypatch, capsys):
    """rc==0 set-status child with non-empty stderr → returns True AND the
    warning is forwarded under the exact `set-status awaiting_promotion on
    #<N>` context prefix (the stub also serves the downstream marker posts,
    which forward under their OWN contexts — asserting the exact repark
    context keeps those from masking the target site; events=[] makes
    _post_followup_run_marker fail-soft with no subprocess of its own)."""
    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=_LANDING_CHECK_LINE),
    )

    assert asw._repark_completed_followup_round(1150, "reason", [], dry_run=False) is True
    err = capsys.readouterr().err
    assert "[task.py stderr] set-status awaiting_promotion on #1150:" in err
    assert "task.py LANDING CHECK" in err


# ──────────────────────────────────────────────────────────────────────
# file_infra_task.cmd_file_infra (`task.py new` child)
# ──────────────────────────────────────────────────────────────────────


def _file_infra_args(**overrides):
    """Namespace with every attr cmd_file_infra consumes up to the
    --no-dispatch early return (_build_new_argv attrs + no_dispatch), plus
    auto_approve_gpu_hours for the dispatch path (_build_spawn_argv reads it;
    a harmless extra attr for the no_dispatch=True tests)."""
    base = dict(
        kind="infra",
        title="t",
        body=None,
        body_file=None,
        parent=None,
        tag=[],
        origin_prompt=None,
        no_dispatch=True,
        auto_approve_gpu_hours=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _open_dispatch_gates(monkeypatch):
    """Force every pre-spawn gate open so cmd_file_infra reaches the spawn
    subprocess (daemon, cap, registration, lease, stagger); stub the dispatch
    stamp so no ~/.eps-autonomous state is written."""
    monkeypatch.setattr(file_infra_task, "_daemon_reachable", lambda: True)
    monkeypatch.setattr(file_infra_task, "infra_dispatch_has_free_slot", lambda: True)
    monkeypatch.setattr(file_infra_task, "_issue_has_live_registration", lambda issue: False)
    monkeypatch.setattr(file_infra_task, "dispatch_lease_fresh", lambda issue: None)
    monkeypatch.setattr(file_infra_task, "session_dispatch_stagger_s", lambda: 0.0)
    monkeypatch.setattr(file_infra_task, "last_session_dispatch_age_s", lambda: None)
    monkeypatch.setattr(file_infra_task, "stagger_delay_s", lambda age, window: 0)
    monkeypatch.setattr(file_infra_task, "record_session_dispatch", lambda *a, **k: None)
    monkeypatch.setattr(file_infra_task, "spawn_output_suppressed", lambda stdout: None)


def _stub_two_children(monkeypatch, spawn_result):
    """subprocess.run stub serving both children: the `task.py new` argv gets a
    clean rc==0 filing (empty stderr — isolates the spawn-hop forwarding from
    the already-tested `task.py new`-leg forwarding); the spawn argv gets
    ``spawn_result``."""

    def fake_run(argv, *a, **k):
        if any("task.py" in str(t) for t in argv):
            return SimpleNamespace(returncode=0, stdout="filed #123", stderr="")
        return spawn_result

    monkeypatch.setattr(file_infra_task.subprocess, "run", fake_run)


def test_file_infra_task_new_forwards_rc0_stderr(monkeypatch, capsys):
    """rc==0 `task.py new` child with non-empty stderr → exit 0 AND the
    warning is forwarded under the `task.py new (file_infra_task)` context
    (no_dispatch=True returns right after filing — no daemon/cap probes)."""
    monkeypatch.setattr(
        file_infra_task.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="filed #123", stderr=_LANDING_CHECK_LINE
        ),
    )

    rc = file_infra_task.cmd_file_infra(_file_infra_args())

    assert rc == 0
    out, err = capsys.readouterr()
    assert "filed #123" in out
    assert "[task.py stderr] task.py new (file_infra_task):" in err
    assert "task.py LANDING CHECK" in err


def test_file_infra_task_new_rc_nonzero_unchanged(monkeypatch, capsys):
    """rc!=0 → exit code == child rc, the child stderr is written exactly
    once (the existing sys.stderr.write), and NO `[task.py stderr]` prefix
    (no double-print on the failure path)."""
    monkeypatch.setattr(
        file_infra_task.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="new failed detail\n"),
    )

    rc = file_infra_task.cmd_file_infra(_file_infra_args())

    assert rc == 1
    err = capsys.readouterr().err
    assert err.count("new failed detail") == 1
    assert "task NOT filed" in err
    assert "[task.py stderr]" not in err


# ──────────────────────────────────────────────────────────────────────
# #1221: file_infra_task.cmd_file_infra (`spawn_session spawn-issue` child,
# the rc==0 second-hop swallow)
# ──────────────────────────────────────────────────────────────────────


def test_file_infra_spawn_hop_forwards_rc0_stderr(monkeypatch, capsys):
    """rc==0 spawn child with non-empty stderr → exit 0, dispatched stdout, and
    the warning forwarded EXACTLY ONCE under the spawn-hop context (the #1221
    durability pin; the single-occurrence count guards accidental
    double-forward on the rc==0 path)."""
    _open_dispatch_gates(monkeypatch)
    _stub_two_children(
        monkeypatch,
        SimpleNamespace(returncode=0, stdout="spawned sid-abc", stderr=_LANDING_CHECK_LINE),
    )

    rc = file_infra_task.cmd_file_infra(_file_infra_args(no_dispatch=False))

    assert rc == 0
    out, err = capsys.readouterr()
    assert "filed + dispatched #123" in out
    assert "[task.py stderr] spawn_session spawn-issue (file_infra_task):" in err
    assert "task.py LANDING CHECK" in err
    assert err.count("[task.py stderr]") == 1


def test_file_infra_spawn_hop_empty_stderr_silent(monkeypatch, capsys):
    """Empty spawn-child stderr (the common case) → zero new output."""
    _open_dispatch_gates(monkeypatch)
    _stub_two_children(
        monkeypatch, SimpleNamespace(returncode=0, stdout="spawned sid-abc", stderr="")
    )

    rc = file_infra_task.cmd_file_infra(_file_infra_args(no_dispatch=False))

    assert rc == 0
    err = capsys.readouterr().err
    assert "[task.py stderr]" not in err


def test_file_infra_spawn_hop_suppressed_path_still_forwards(monkeypatch, capsys):
    """The suppressed-dispatch path (spawn_output_suppressed → a reason) STILL
    forwards the rc==0 spawn-child stderr — pins the placement decision:
    forwarding runs BEFORE the suppression branch, which returns early."""
    _open_dispatch_gates(monkeypatch)
    monkeypatch.setattr(file_infra_task, "spawn_output_suppressed", lambda stdout: "lease held")
    _stub_two_children(
        monkeypatch,
        SimpleNamespace(returncode=0, stdout="suppressed", stderr=_LANDING_CHECK_LINE),
    )

    rc = file_infra_task.cmd_file_infra(_file_infra_args(no_dispatch=False))

    assert rc == 0
    out, err = capsys.readouterr()
    assert "dispatch suppressed" in out
    assert "[task.py stderr] spawn_session spawn-issue (file_infra_task):" in err


def test_file_infra_spawn_hop_rc_nonzero_unchanged(monkeypatch, capsys):
    """rc!=0 spawn child → wrapper still exits 0 (best-effort dispatch
    contract), the child stderr detail is embedded exactly once in the FAILED
    line, and NO forwarding prefix (no double-forward on the failure path —
    mirror of the existing rc!=0 tests)."""
    _open_dispatch_gates(monkeypatch)
    _stub_two_children(
        monkeypatch, SimpleNamespace(returncode=1, stdout="", stderr="spawn boom detail")
    )

    rc = file_infra_task.cmd_file_infra(_file_infra_args(no_dispatch=False))

    assert rc == 0
    err = capsys.readouterr().err
    assert "dispatch FAILED" in err
    assert err.count("spawn boom detail") == 1
    assert "[task.py stderr]" not in err


def test_file_infra_task_imports_shared_helper():
    """Pins the no-drift decision: file_infra_task reuses the watcher's
    forwarder rather than a copied local one (a rename fails loud here)."""
    assert file_infra_task._forward_marker_child_stderr is asw._forward_marker_child_stderr


# ──────────────────────────────────────────────────────────────────────
# #1257: the watcher's seven in-file spawn_session.py subprocess sites
# (rc==0 stderr was discarded; #1221 closed the same class at
# file_infra_task's spawn hop — these are the in-file residue).
# ──────────────────────────────────────────────────────────────────────


def _stub_watch_spawn_collaborators(monkeypatch, tmp_path):
    """Force every watcher spawn-site collaborator inert so each site reaches
    its subprocess.run: auth-outage gate open + record no-op, no suppression,
    no per-issue overrides / registry predecessor, #1059 stagger window
    closed, no dispatch stamp, and an empty registry dir (the infra-drain
    pre-spawn re-check then sees no registration files and proceeds)."""
    monkeypatch.setattr(asw, "_auth_outage_spawn_gate", lambda *a, **k: None)
    monkeypatch.setattr(asw, "_auth_outage_record_spawn", lambda *a, **k: None)
    monkeypatch.setattr(asw, "spawn_output_suppressed", lambda stdout: None)
    monkeypatch.setattr(asw, "_stalled_session_overrides", lambda issue: [])
    monkeypatch.setattr(asw, "_registry_spawned_at", lambda issue: None)
    monkeypatch.setattr(asw, "session_dispatch_stagger_s", lambda: 0.0)
    monkeypatch.setattr(asw, "last_session_dispatch_age_s", lambda: None)
    monkeypatch.setattr(asw, "stagger_delay_s", lambda age, window: 0)
    monkeypatch.setattr(asw, "record_session_dispatch", lambda *a, **k: None)
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path)


def _stub_watch_spawn_child(
    monkeypatch, *, returncode=0, stdout="spawned sid-x", stderr=_LANDING_CHECK_LINE
):
    """monkeypatch ``asw.subprocess.run`` to a fixed child result; returns the
    recorded call list (single-line stderr fixture so the exactly-once count
    guards hold)."""
    calls = []
    monkeypatch.setattr(
        asw.subprocess,
        "run",
        lambda *a, **k: (
            calls.append((a, k))
            or SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
        ),
    )
    return calls


# The five suppression-bearing spawn sites (sites 1/3/4/5/6 of the plan §4
# table), keyed by their `_auth_outage_record_spawn` arm name. Site 2
# (`_stop_session`) and site 7 (`_respawn_campaign`) have no suppression
# branch and are covered by their own forwards tests only.
_SUPPRESSION_SITES = {
    "crash": (
        lambda: asw._respawn({"issue": 1257, "auto_approve_gpu_hours": 24.0}, dry_run=False),
        "spawn_session spawn-issue (crash)",
    ),
    "stalled": (
        lambda: asw._respawn_stalled_session(1257, 24.0, dry_run=False),
        "spawn_session spawn-issue (stalled)",
    ),
    "orphan": (
        lambda: asw._respawn_orphan(1257, 24.0, dry_run=False),
        "spawn_session spawn-issue (orphan)",
    ),
    "infra-drain": (
        lambda: asw._dispatch_infra_drain(1257, "slot", False, reg_snapshot={}),
        "spawn_session spawn-issue (infra-drain)",
    ),
    "capacity-retry": (
        lambda: asw._redrive_capacity_retry(1257, dry_run=False),
        "spawn_session spawn-issue (capacity-retry)",
    ),
}


def test_watch_respawn_crash_forwards_rc0_stderr(monkeypatch, capsys, tmp_path):
    """Site 1 `_respawn`: rc==0 spawn child with non-empty stderr → returns
    "spawned" AND forwards under the crash-arm context."""
    _stub_watch_spawn_collaborators(monkeypatch, tmp_path)
    calls = _stub_watch_spawn_child(monkeypatch)

    result = asw._respawn({"issue": 1257, "auto_approve_gpu_hours": 24.0}, dry_run=False)

    assert result == "spawned"
    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "[task.py stderr] spawn_session spawn-issue (crash):" in err
    assert "task.py LANDING CHECK" in err


def test_watch_stop_session_forwards_rc0_stderr(monkeypatch, capsys, tmp_path):
    """Site 2 `_stop_session`: rc==0 stop child with non-empty stderr →
    returns True AND forwards under the `stop (watcher)` context (this site's
    rc==0 path previously read nothing at all)."""
    _stub_watch_spawn_collaborators(monkeypatch, tmp_path)
    _stub_watch_spawn_child(monkeypatch, stdout="")

    assert asw._stop_session("sid-x", dry_run=False) is True
    err = capsys.readouterr().err
    assert "[task.py stderr] spawn_session stop (watcher):" in err
    assert "task.py LANDING CHECK" in err


def test_watch_respawn_stalled_forwards_rc0_stderr(monkeypatch, capsys, tmp_path):
    """Site 3 `_respawn_stalled_session`: rc==0 → "spawned" + stalled-arm
    context forwarding."""
    _stub_watch_spawn_collaborators(monkeypatch, tmp_path)
    _stub_watch_spawn_child(monkeypatch)

    assert asw._respawn_stalled_session(1257, 24.0, dry_run=False) == "spawned"
    err = capsys.readouterr().err
    assert "[task.py stderr] spawn_session spawn-issue (stalled):" in err
    assert "task.py LANDING CHECK" in err


def test_watch_respawn_orphan_forwards_rc0_stderr(monkeypatch, capsys, tmp_path):
    """Site 4 `_respawn_orphan`: rc==0 → "spawned" + orphan-arm context
    forwarding."""
    _stub_watch_spawn_collaborators(monkeypatch, tmp_path)
    _stub_watch_spawn_child(monkeypatch)

    assert asw._respawn_orphan(1257, 24.0, dry_run=False) == "spawned"
    err = capsys.readouterr().err
    assert "[task.py stderr] spawn_session spawn-issue (orphan):" in err
    assert "task.py LANDING CHECK" in err


def test_watch_infra_drain_dispatch_forwards_rc0_stderr(monkeypatch, capsys, tmp_path):
    """Site 5 `_dispatch_infra_drain`: rc==0 → "spawned" + infra-drain-arm
    context forwarding, EXACTLY ONCE (the #1221 single-forward count guard;
    the stderr fixture is single-line by construction)."""
    _stub_watch_spawn_collaborators(monkeypatch, tmp_path)
    _stub_watch_spawn_child(monkeypatch)

    assert asw._dispatch_infra_drain(1257, "slot", False, reg_snapshot={}) == "spawned"
    err = capsys.readouterr().err
    assert "[task.py stderr] spawn_session spawn-issue (infra-drain):" in err
    assert "task.py LANDING CHECK" in err
    assert err.count("[task.py stderr]") == 1


def test_watch_capacity_retry_forwards_rc0_stderr(monkeypatch, capsys, tmp_path):
    """Site 6 `_redrive_capacity_retry`: rc==0 → "spawned" +
    capacity-retry-arm context forwarding."""
    _stub_watch_spawn_collaborators(monkeypatch, tmp_path)
    _stub_watch_spawn_child(monkeypatch)

    assert asw._redrive_capacity_retry(1257, dry_run=False) == "spawned"
    err = capsys.readouterr().err
    assert "[task.py stderr] spawn_session spawn-issue (capacity-retry):" in err
    assert "task.py LANDING CHECK" in err


def test_watch_respawn_campaign_forwards_rc0_stderr(monkeypatch, capsys, tmp_path):
    """Site 7 `_respawn_campaign`: rc==0 → returns True AND forwards under
    the spawn-campaign context (no suppression branch at this site — the
    insertion sits before the success print)."""
    _stub_watch_spawn_collaborators(monkeypatch, tmp_path)
    _stub_watch_spawn_child(monkeypatch)

    assert asw._respawn_campaign({"issue": 1257}, dry_run=False) is True
    err = capsys.readouterr().err
    assert "[task.py stderr] spawn_session spawn-campaign (campaign):" in err
    assert "task.py LANDING CHECK" in err


@pytest.mark.parametrize("arm", sorted(_SUPPRESSION_SITES))
def test_watch_spawn_hop_suppressed_path_still_forwards(monkeypatch, capsys, tmp_path, arm):
    """Every suppression-bearing site (1/3/4/5/6) STILL forwards the rc==0
    spawn-child stderr on the suppressed-dispatch path — pins the
    before-the-suppression-branch placement at each site (mirrors #1221's
    `test_file_infra_spawn_hop_suppressed_path_still_forwards`)."""
    call, context = _SUPPRESSION_SITES[arm]
    _stub_watch_spawn_collaborators(monkeypatch, tmp_path)
    monkeypatch.setattr(asw, "spawn_output_suppressed", lambda stdout: "lease held")
    _stub_watch_spawn_child(monkeypatch, stdout="suppressed")

    assert call() == "suppressed"
    err = capsys.readouterr().err
    assert f"[task.py stderr] {context}:" in err
    assert "task.py LANDING CHECK" in err
    assert err.count("[task.py stderr]") == 1


def test_watch_respawn_crash_rc_nonzero_unchanged(monkeypatch, capsys, tmp_path):
    """Representative rc!=0 pin at site 1: returns "failed", the existing
    RESPAWN FAILED line carries the stderr detail exactly once, and NO
    forwarding prefix (the rc!=0 branch already prints the stderr — mirrors
    `test_set_status_blocked_rc_nonzero_unchanged`; the per-site rc!=0
    branches are structurally identical, plan §5)."""
    _stub_watch_spawn_collaborators(monkeypatch, tmp_path)
    _stub_watch_spawn_child(monkeypatch, returncode=1, stdout="", stderr="respawn boom detail")

    result = asw._respawn({"issue": 1257, "auto_approve_gpu_hours": 24.0}, dry_run=False)

    assert result == "failed"
    err = capsys.readouterr().err
    assert "RESPAWN FAILED issue #1257" in err
    assert err.count("respawn boom detail") == 1
    assert "[task.py stderr]" not in err
