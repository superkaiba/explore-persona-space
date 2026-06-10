"""Tests for codex_task.py prompt delivery via --prompt-file (E2BIG guard).

The Codex twin dispatch helper used to append the composed prompt as a
single argv element of the ``node codex-companion.mjs task`` spawn. Any
prompt over the kernel's per-argument limit (~128KiB) killed the spawn
with ``OSError [Errno 7] Argument list too long`` (E2BIG) — observed on
task #540 code-review round 1 with a 176K-char prompt (2026-06-09),
degrading the ensemble round to Claude-only.

The fix delivers the prompt through a temp file + the companion's native
``--prompt-file`` flag, unconditionally. These tests pin that contract
hermetically (no real node spawn):

1. The prompt NEVER appears on the argv; ``--prompt-file`` does.
2. The temp file holds the full prompt (verified at spawn time, before
   cleanup) — including a prompt far larger than the old argv limit.
3. The temp file is removed after the spawn returns, on success AND on
   the spawn-failure path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_codex_task():
    """Load scripts/codex_task.py as an isolated module."""
    spec = importlib.util.spec_from_file_location(
        "codex_task_prompt_under_test", REPO_ROOT / "scripts" / "codex_task.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["codex_task_prompt_under_test"] = module
    spec.loader.exec_module(module)
    return module


codex_task = _load_codex_task()


class _FakeCompletedProcess:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _capture_spawn(monkeypatch, captured: dict, returncode: int = 0):
    """Patch subprocess.run to record the spawn cmd + the on-disk prompt
    file content AT SPAWN TIME (the file is deleted before _spawn_codex
    returns, so it must be read inside the fake)."""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        assert "--prompt-file" in cmd, cmd
        prompt_path = Path(cmd[cmd.index("--prompt-file") + 1])
        captured["prompt_path"] = prompt_path
        captured["prompt_path_existed"] = prompt_path.exists()
        captured["prompt_content"] = prompt_path.read_text(encoding="utf-8")
        return _FakeCompletedProcess(
            returncode=returncode,
            stdout="Queued task-abc123\n" if returncode == 0 else "",
            stderr="" if returncode == 0 else "boom",
        )

    monkeypatch.setattr(codex_task.subprocess, "run", fake_run)


def test_prompt_never_on_argv(monkeypatch):
    """The prompt text must not appear as an argv element — only the
    --prompt-file path does."""
    captured: dict = {}
    _capture_spawn(monkeypatch, captured)

    prompt = "review this diff carefully"
    job_id = codex_task._spawn_codex(Path("/fake/companion.mjs"), prompt, "high", False)

    assert job_id == "task-abc123"
    assert prompt not in captured["cmd"]
    assert "--prompt-file" in captured["cmd"]
    # The path handed to the companion is absolute, so its
    # path.resolve(cwd, promptFile) returns it unchanged.
    assert captured["prompt_path"].is_absolute()
    assert captured["prompt_content"] == prompt


def test_large_prompt_delivered_in_full(monkeypatch):
    """A prompt far beyond the old ~128KiB argv limit is delivered intact
    through the temp file (the E2BIG regression class from task #540)."""
    captured: dict = {}
    _capture_spawn(monkeypatch, captured)

    prompt = "x" * 300_000  # > 2x the kernel per-argument cap
    codex_task._spawn_codex(Path("/fake/companion.mjs"), prompt, "xhigh", True)

    assert captured["prompt_path_existed"] is True
    assert captured["prompt_content"] == prompt
    # No argv element carries the bulk of the prompt.
    assert all(len(part) < 1000 for part in captured["cmd"]), [
        len(part) for part in captured["cmd"]
    ]


def test_write_flag_still_threaded(monkeypatch):
    """--write placement is preserved alongside the new --prompt-file."""
    captured: dict = {}
    _capture_spawn(monkeypatch, captured)

    codex_task._spawn_codex(Path("/fake/companion.mjs"), "go", "high", True)
    assert "--write" in captured["cmd"]

    codex_task._spawn_codex(Path("/fake/companion.mjs"), "go", "high", False)
    assert "--write" not in captured["cmd"]


def test_temp_file_cleaned_up_on_success(monkeypatch):
    """The prompt temp file is deleted once the spawn call returns."""
    captured: dict = {}
    _capture_spawn(monkeypatch, captured)

    codex_task._spawn_codex(Path("/fake/companion.mjs"), "go", "high", False)
    assert not captured["prompt_path"].exists()


def test_temp_file_cleaned_up_on_spawn_failure(monkeypatch):
    """A non-zero companion exit still removes the temp file (finally
    path), and the failure surfaces as the existing RuntimeError."""
    captured: dict = {}
    _capture_spawn(monkeypatch, captured, returncode=1)

    with pytest.raises(RuntimeError, match="spawn failed"):
        codex_task._spawn_codex(Path("/fake/companion.mjs"), "go", "high", False)
    assert not captured["prompt_path"].exists()
