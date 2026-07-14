"""Regression tests for `task.py post-marker` exit-code semantics.

`post_event` appends + commits the marker row BEFORE `cmd_post_event`
echoes the payload JSON to stdout. A BrokenPipeError on that echo (caller
tore the pipe down early — Bash-tool teardown, `| head`, dead SSH) used to
propagate and flip the exit code to nonzero AFTER the commit landed, so
callers that treat rc!=0 as "not posted" (codex_task._post_marker) retried
and duplicated the marker (incident #537, 2026-06-10: duplicate
epm:codex-task-spawned). The echo failure is now non-fatal; pre-commit
failures (oversize note, missing task, flock timeout) stay fatal.

The CLI is exercised at the handler-function layer (not via subprocess) —
see test_task_workflow.py::test_cli_handlers_raise_address_defer_list_roundtrip
for why (the branch-guarded resolver can't be redirected across a process
boundary).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import task as task_cli


class _BrokenPipeStdout(io.TextIOBase):
    """Stand-in for a stdout whose pipe the reader has already torn down."""

    def write(self, _s: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        raise BrokenPipeError

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno")


def _ns(**overrides) -> argparse.Namespace:
    base = dict(
        number=537,
        marker="epm:codex-task-spawned",
        version=1,
        by="codex_task",
        note="Codex job_id=task-abc",
        file=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_broken_pipe_on_echo_is_nonfatal_after_commit(monkeypatch, capsys):
    """A BrokenPipeError on the post-commit stdout echo must NOT raise out of
    cmd_post_event — the exit code has to reflect the commit, not the echo."""
    posted = []

    def fake_post_event(number, marker, *, version, by, note):
        posted.append((number, marker, version, by, note))
        return {
            "ts": "2026-06-10T00:00:00Z",
            "kind": marker,
            "version": version,
            "by": by,
            "note": note,
        }

    monkeypatch.setattr(task_cli, "post_event", fake_post_event)
    monkeypatch.setattr(sys, "stdout", _BrokenPipeStdout())

    task_cli.cmd_post_event(_ns())  # must not raise

    assert len(posted) == 1  # the marker write happened exactly once
    err = capsys.readouterr().err
    assert "committed" in err
    assert "BrokenPipeError" in err


def test_pre_commit_failure_stays_fatal(monkeypatch):
    """Failures raised BY post_event (oversize note, missing task) must still
    propagate — only the post-commit echo became non-fatal."""

    def exploding_post_event(*_a, **_k):
        raise ValueError("event note exceeds 50000 chars")

    monkeypatch.setattr(task_cli, "post_event", exploding_post_event)
    with pytest.raises(ValueError):
        task_cli.cmd_post_event(_ns())


def test_normal_echo_prints_payload(monkeypatch, capsys):
    """With a healthy stdout the payload JSON is still echoed unchanged."""
    monkeypatch.setattr(
        task_cli,
        "post_event",
        lambda *_a, **_k: {"kind": "epm:echo-check", "version": 1},
    )
    task_cli.cmd_post_event(_ns(marker="epm:echo-check"))
    out = capsys.readouterr().out
    assert '"kind": "epm:echo-check"' in out


def test_set_status_broken_pipe_on_echo_is_nonfatal(monkeypatch, capsys):
    """The _safe_echo guard covers the other mutating handlers too: a
    BrokenPipeError on cmd_set_status's post-commit path echo must not raise
    (rc reflects the status move, not the echo)."""
    moved = []

    def fake_set_status(number, status, *, note=None, force_followup_exit=False):
        moved.append((number, status, note))
        return Path("/repo/tasks/approved/537")

    monkeypatch.setattr(task_cli, "set_status", fake_set_status)
    monkeypatch.setattr(sys, "stdout", _BrokenPipeStdout())

    ns = argparse.Namespace(number=537, status="approved", note=None)
    task_cli.cmd_set_status(ns)  # must not raise

    assert moved == [(537, "approved", None)]  # the git mv + commit happened exactly once
    err = capsys.readouterr().err
    assert "committed" in err
    assert "BrokenPipeError" in err


def test_set_status_normal_echo_prints_path(monkeypatch, capsys):
    """With a healthy stdout, cmd_set_status still echoes the relative path."""
    monkeypatch.setattr(
        task_cli,
        "set_status",
        lambda number, status, *, note=None, force_followup_exit=False: Path(
            "/repo/tasks/approved/537"
        ),
    )
    ns = argparse.Namespace(number=537, status="approved", note=None)
    task_cli.cmd_set_status(ns)
    out = capsys.readouterr().out
    assert "tasks/approved/537" in out


def test_set_status_followup_hold_refusal_exits_cleanly(monkeypatch):
    """The library's same-issue follow-up status-hold ValueError must surface
    as a clean SystemExit (message, nonzero rc) — not a raw traceback."""

    def refusing_set_status(number, status, *, note=None, force_followup_exit=False):
        raise ValueError("followups_running is HELD ... (status-hold rule)")

    monkeypatch.setattr(task_cli, "set_status", refusing_set_status)
    ns = argparse.Namespace(number=537, status="running", note=None)
    with pytest.raises(SystemExit) as exc_info:
        task_cli.cmd_set_status(ns)
    assert "status-hold" in str(exc_info.value)


def test_set_status_plan_gate_holds_at_followups_running(monkeypatch, capsys):
    """A --auto-approve-if-autonomous plan-gate call on a followups_running
    task fires the gate decision + marker but NEVER moves the status
    (status-hold rule, SKILL.md Step 9b § Same-issue follow-up loop step 3)."""
    moved = []
    posted = []
    monkeypatch.setattr(
        task_cli,
        "set_status",
        lambda number, status, *, note=None, force_followup_exit=False: moved.append(
            (number, status)
        ),
    )
    monkeypatch.setattr(
        task_cli,
        "get_task",
        lambda number: {"status": "followups_running", "frontmatter": {"tags": []}},
    )

    def fake_post_event(number, marker, *, version, by, note):
        posted.append((number, marker))
        return {"kind": marker, "version": version}

    monkeypatch.setattr(task_cli, "post_event", fake_post_event)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "24")

    ns = argparse.Namespace(
        number=537,
        status="plan_pending",
        note=None,
        auto_approve_if_autonomous=True,
        gpu_hours=4.0,
    )
    task_cli.cmd_set_status(ns)

    assert moved == []  # the status flip never happened
    assert posted == [(537, "epm:plan-approved")]  # the gate decision still landed
    out = capsys.readouterr().out
    assert "followups_running hold: status unchanged" in out


def test_set_status_plan_gate_hold_parked_over_cap(monkeypatch, capsys):
    """The over-cap sub-branch of the plan-gate hold: posts
    epm:awaiting-spend-approval, never moves the status."""
    moved = []
    posted = []
    monkeypatch.setattr(
        task_cli,
        "set_status",
        lambda number, status, *, note=None, force_followup_exit=False: moved.append(
            (number, status)
        ),
    )
    monkeypatch.setattr(
        task_cli,
        "get_task",
        lambda number: {"status": "followups_running", "frontmatter": {"tags": []}},
    )

    def fake_post_event(number, marker, *, version, by, note):
        posted.append((number, marker))
        return {"kind": marker, "version": version}

    monkeypatch.setattr(task_cli, "post_event", fake_post_event)
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "24")

    ns = argparse.Namespace(
        number=537,
        status="plan_pending",
        note=None,
        auto_approve_if_autonomous=True,
        gpu_hours=200.0,  # over the 24h cap
    )
    task_cli.cmd_set_status(ns)

    assert moved == []
    assert posted == [(537, "epm:awaiting-spend-approval")]
    out = capsys.readouterr().out
    assert "parked_over_cap" in out
    assert "followups_running hold: status unchanged" in out


def test_set_status_followups_running_missing_tag_warns(monkeypatch, capsys):
    """Transitioning TO followups_running without a followup-auto/-manual tag
    prints the missing-tag WARNING (a bare `followup` tag does not count)."""
    monkeypatch.setattr(
        task_cli,
        "set_status",
        lambda number, status, *, note=None, force_followup_exit=False: Path(
            "/repo/tasks/followups_running/537"
        ),
    )
    monkeypatch.setattr(
        task_cli,
        "get_task",
        lambda number: {"status": "followups_running", "frontmatter": {"tags": ["followup"]}},
    )
    ns = argparse.Namespace(number=537, status="followups_running", note=None)
    task_cli.cmd_set_status(ns)
    out = capsys.readouterr().out
    assert "WARNING: transitioned to followups_running without a" in out

    # And with the proper tag present, no warning.
    monkeypatch.setattr(
        task_cli,
        "get_task",
        lambda number: {
            "status": "followups_running",
            "frontmatter": {"tags": ["followup-manual"]},
        },
    )
    task_cli.cmd_set_status(ns)
    out = capsys.readouterr().out
    assert "WARNING" not in out


# ─── #1178: poster-side WARN on literal backslash-n field-led notes ─────────
# The parse-side normalization landed in #1120 (task_workflow.
# parse_followup_note_field); these tests pin the poster-side WARN in
# cmd_post_event: fire on a single-line field-led --note carrying literal
# \n two-char escapes, stay silent everywhere else, never touch rc/stdout.

_MALFORMED_NOTE = "followup_label: x\\nsource: user-chat\\nround: 7"


def _capturing_post_event(posted: list):
    """Signature-conformant fake for the post_event boundary (mirrors the
    real keyword-only signature; appends every call for exactly-once +
    note-unmutated asserts)."""

    def fake_post_event(number, marker, *, version, by, note):
        posted.append((number, marker, version, by, note))
        return {"kind": marker, "version": version, "note": note}

    return fake_post_event


def test_backslash_n_field_led_note_warns_and_still_posts(monkeypatch, capsys):
    """Acceptance 1 + 4: the malformed shape fires a stderr WARNING; the
    marker still posts exactly once with the note UNMUTATED, the handler
    returns without raising, and stdout keeps the parseable payload JSON."""
    posted = []
    monkeypatch.setattr(task_cli, "post_event", _capturing_post_event(posted))

    task_cli.cmd_post_event(_ns(note=_MALFORMED_NOTE))  # must not raise

    assert len(posted) == 1
    assert posted[0][4] == _MALFORMED_NOTE  # note reached post_event unmutated
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "$'" in captured.err  # the shell-quoting hint
    assert "--file" in captured.err  # the multi-line escape hatch hint
    assert "WARNING" not in captured.out  # stdout JSON stays parseable
    assert '"kind"' in captured.out  # payload echo still happened


@pytest.mark.parametrize(
    "note",
    [
        "followup_label: x\nsource: user-chat",  # real newlines — well-formed
        "followup_label: x\\nliteral kept\n",  # mixed — parse-side under-reach parity
    ],
)
def test_real_multiline_note_no_warn(monkeypatch, capsys, note):
    """Acceptance 2: a real-multiline note never warns, including the mixed
    case (real newline present + literal \\n as content), mirroring the
    parse-side gate `"\\\\n" in note and "\\n" not in note`."""
    posted = []
    monkeypatch.setattr(task_cli, "post_event", _capturing_post_event(posted))
    task_cli.cmd_post_event(_ns(note=note))
    assert len(posted) == 1
    assert "WARNING" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "note",
    [
        "see the log tail\\nall good",  # prose head — not field-led
        '{"current_phase": "workload", "tail": "x\\ny"}',  # JSON body
    ],
)
def test_prose_and_json_backslash_n_notes_no_warn(monkeypatch, capsys, note):
    """Acceptance 3: literal \\n without a field-led head shape stays silent
    (the corpus's dominant legitimate escape-carriers are JSON bodies)."""
    posted = []
    monkeypatch.setattr(task_cli, "post_event", _capturing_post_event(posted))
    task_cli.cmd_post_event(_ns(note=note))
    assert len(posted) == 1
    assert "WARNING" not in capsys.readouterr().err


def test_file_input_backslash_n_no_warn(monkeypatch, capsys, tmp_path):
    """Acceptance 5: --file input carrying the same malformed shape never
    warns (--file is the documented multi-line escape hatch, plan §4 D3).
    The fixture is written WITHOUT a trailing real newline and the
    predicate precondition is asserted, so this test cannot pass vacuously
    via the mixed-case suppression."""
    body = tmp_path / "note.md"
    body.write_text(_MALFORMED_NOTE)
    resolved = body.read_text()
    # Precondition: the file bytes DO match the malformed predicate shape —
    # only the --file gate may be what suppresses the WARN here.
    assert "\\n" in resolved and "\n" not in resolved
    assert task_cli._looks_field_led(resolved)

    posted = []
    monkeypatch.setattr(task_cli, "post_event", _capturing_post_event(posted))
    task_cli.cmd_post_event(_ns(note=None, file=str(body)))
    assert len(posted) == 1
    assert posted[0][4] == resolved  # file bytes passed through verbatim
    assert "WARNING" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        ("followup_label: x", True),
        ("pod=pod-399 pid=1", True),
        ("- source: x", True),
        ("**followup_label:** x", True),
        ("  - **field:** x", True),
        ('{"a": 1}', False),
        ("Round 7 complete; followup_label: x", False),
        ("", False),
    ],
)
def test_looks_field_led_shapes(note, expected):
    """The head-only field-led heuristic covers the same bare/`=`-form/
    bullet/bold shapes the parse-side segment core-strip recognizes, and
    deliberately ignores mid-note `; `-joined clauses (plan §4 D2)."""
    assert task_cli._looks_field_led(note) is expected


class _ClosedStderr(io.TextIOBase):
    """Stand-in for a CLOSED stderr stream — writes raise ValueError (the
    io module's closed-file signal), unlike a torn pipe's BrokenPipeError."""

    def write(self, _s: str) -> int:
        raise ValueError("I/O operation on closed file")

    def flush(self) -> None:
        raise ValueError("I/O operation on closed file")


@pytest.mark.parametrize("broken_stderr", [_BrokenPipeStdout(), _ClosedStderr()])
def test_warn_stderr_failure_is_nonfatal(monkeypatch, capsys, broken_stderr):
    """Acceptance 4, stderr-failure half: a stderr that raises on the WARN
    write (torn pipe -> BrokenPipeError/OSError; closed stream ->
    ValueError) must not raise out of the handler — rc reflects the commit
    that already landed (pins the `except (OSError, ValueError)` guard)."""
    posted = []
    monkeypatch.setattr(task_cli, "post_event", _capturing_post_event(posted))
    monkeypatch.setattr(sys, "stderr", broken_stderr)

    task_cli.cmd_post_event(_ns(note=_MALFORMED_NOTE))  # must not raise

    assert len(posted) == 1  # the marker write happened exactly once
    out = capsys.readouterr().out
    assert '"kind"' in out  # the healthy stdout echo still happened
