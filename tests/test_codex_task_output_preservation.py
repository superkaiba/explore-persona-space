"""Tests for codex_task.py output-file preservation.

The twin-reviewer wrappers instruct Codex to write its full
marker-formatted verdict to --output-file MID-SESSION; the helper's
final-message write previously clobbered exactly that file (task #604
interpretation-critic round 1: a 12,474-char critique reduced to a
323-char closing chat message). Behavior under test:

1. A marker-formatted verdict Codex wrote to --output-file during the
   attempt is PRESERVED; the final chat message lands in the
   ``<output-file>.final-msg.md`` sidecar.
2. When the helper is the only producer (no mid-session write), the
   final message lands at --output-file exactly as before.
3. A STALE pre-existing file (not touched during the attempt — e.g. left
   by a previous reviewer round or a failed transient-retry attempt,
   #579) is overwritten, even when it carries the epm marker tag.
4. A mid-session write WITHOUT the epm marker tag (half-written /
   non-verdict content) is overwritten — the guard keys on the marker,
   not on size.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_codex_task():
    """Load scripts/codex_task.py as an isolated module."""
    spec = importlib.util.spec_from_file_location(
        "codex_task_output_preservation_under_test",
        REPO_ROOT / "scripts" / "codex_task.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["codex_task_output_preservation_under_test"] = module
    spec.loader.exec_module(module)
    return module


codex_task = _load_codex_task()

VERDICT = (
    "<!-- epm:interp-critique-codex v1 -->\n"
    "## Codex Interpretation Critique\n" + ("blocker detail " * 200) + "\nVerdict: REVISE\n"
    "<!-- /epm:interp-critique-codex -->\n"
)
FINAL_MSG = "Wrote the required critique to /tmp/out.md. Verdict: REVISE"


def _args(output_file, **overrides):
    base = dict(
        issue=None,
        effort="high",
        write=False,
        output_file=output_file,
        prompt_file=None,
        prompt="do the thing",
        max_wait_secs=3600,
        poll_interval_secs=0,
        probe_error_cap=10,
        stall_detect_secs=600,
        cancelled_retry_cap=2,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _run_attempt(monkeypatch, out_file, mid_session_write=None, final_message=FINAL_MSG):
    """Drive one _run_one_attempt to phase=done with plumbing faked out.

    ``mid_session_write`` (str | None): content the fake Codex writes to
    ``out_file`` during the first poll — simulating the wrapper-contract
    apply_patch. None = Codex never touches the file.
    """
    probe_calls = {"n": 0}

    def fake_probe(companion, job_id):
        probe_calls["n"] += 1
        if probe_calls["n"] == 1:  # confirm probe
            return "running", "", None
        if mid_session_write is not None and probe_calls["n"] == 2:
            out_file.write_text(mid_session_write)
        if probe_calls["n"] >= 3:
            return "done", "", None
        return "running", "", None

    monkeypatch.setattr(codex_task, "_spawn_codex", lambda *a, **k: "task-preserve")
    monkeypatch.setattr(codex_task, "_probe_phase", fake_probe)
    monkeypatch.setattr(codex_task, "_fetch_result", lambda *a, **k: (0, final_message, ""))
    monkeypatch.setattr(codex_task, "_best_effort_cancel", lambda *a, **k: None)
    monkeypatch.setattr(codex_task.time, "sleep", lambda s: None)

    args = _args(output_file=out_file)
    return codex_task._run_one_attempt(Path("/fake/companion.mjs"), "p", args, False)


def test_mid_session_verdict_is_preserved(tmp_path, monkeypatch):
    """Codex wrote a marker-formatted verdict to --output-file during the
    attempt: the verdict survives, the final chat message goes to the
    .final-msg.md sidecar, and the attempt still reports done."""
    out = tmp_path / "codex-output-issue-604-interp-r1.md"
    result = _run_attempt(monkeypatch, out, mid_session_write=VERDICT)

    assert result.kind == "done", result.note
    assert out.read_text() == VERDICT  # NOT clobbered
    sidecar = tmp_path / "codex-output-issue-604-interp-r1.md.final-msg.md"
    assert sidecar.read_text() == FINAL_MSG


def test_helper_only_producer_writes_final_message(tmp_path, monkeypatch):
    """No mid-session write (file absent the whole run): the final message
    lands at --output-file, exactly the historical behavior; no sidecar."""
    out = tmp_path / "codex-output.md"
    result = _run_attempt(monkeypatch, out, mid_session_write=None)

    assert result.kind == "done", result.note
    assert out.read_text() == FINAL_MSG
    assert not (tmp_path / "codex-output.md.final-msg.md").exists()


def test_stale_preexisting_verdict_is_overwritten(tmp_path, monkeypatch):
    """A file that existed BEFORE the attempt spawned and was never touched
    during it (stale round-1 verdict, or a failed attempt's leftovers on
    the #579 transient-retry path) is overwritten — marker tag or not."""
    out = tmp_path / "codex-output.md"
    out.write_text(VERDICT)  # stale: written pre-spawn, never advances

    result = _run_attempt(monkeypatch, out, mid_session_write=None)

    assert result.kind == "done", result.note
    assert out.read_text() == FINAL_MSG  # overwritten as always
    assert not (tmp_path / "codex-output.md.final-msg.md").exists()


def test_mid_session_write_without_marker_is_overwritten(tmp_path, monkeypatch):
    """A mid-session write WITHOUT the epm marker tag (half-written file /
    non-verdict content) does not trigger preservation — the guard keys on
    the marker tag, not on size alone."""
    out = tmp_path / "codex-output.md"
    half_written = "## Codex Interpretation Critique (no marker tag)\n" + "x" * 5000
    result = _run_attempt(monkeypatch, out, mid_session_write=half_written)

    assert result.kind == "done", result.note
    assert out.read_text() == FINAL_MSG
    assert not (tmp_path / "codex-output.md.final-msg.md").exists()
