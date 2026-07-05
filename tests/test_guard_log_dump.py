"""CI wrapper for the ``.claude/hooks/guard_log_dump.sh`` PreToolUse hook (#1057).

The hook blocks Bash commands that would page a large log-like file — or an
unbounded / >MAX_RANGE_LINES slice of any big file — into the conversation
context. #1057 split the single dump verdict into two tiers so a bounded
single-range read (<= 2000 lines, unsigned count) of a big non-log code/doc
file passes while every log-shaped block and every unbounded read stays
blocked.

This file is a THIN wrapper (plan #1057 §4b): the in-script ``--self-test``
suite (50 cases) stays the single source of truth for the behavior matrix;
here we (1) run that suite in CI, (2) pin the #986 incident repro against a
deterministic fixture, (3) pin one block per preserved class + the new
codedoc retry-shape message, and (4) assert the settings.json wiring
(#965 ``TestSettingsWiring`` convention from
``tests/test_guard_harmful_bank_read.py`` — closing the "hook ships green but
inert" channel). Tests drive the script exactly as the harness does: stdin
PreToolUse JSON -> exit 0 (allow) / exit 2 (block). Deny/allow determinism:
every run scrubs ``EPM_ALLOW_LOG_DUMP`` from the env.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _REPO_ROOT / ".claude" / "hooks" / "guard_log_dump.sh"
SETTINGS = _REPO_ROOT / ".claude" / "settings.json"
HOOK_REL = ".claude/hooks/guard_log_dump.sh"


def _env() -> dict[str, str]:
    """Hook env with the EPM_ALLOW_LOG_DUMP escape hatch scrubbed."""
    return {k: v for k, v in os.environ.items() if k != "EPM_ALLOW_LOG_DUMP"}


def _run_bash(
    cmd: str, *, cwd: str | None = None, script: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Feed a Bash PreToolUse payload to the guard, returning the completed process."""
    payload = json.dumps({"tool_input": {"command": cmd}})
    return subprocess.run(
        ["bash", str(script or SCRIPT)],
        input=payload,
        text=True,
        capture_output=True,
        env=_env(),
        cwd=cwd,
    )


@pytest.fixture()
def big_source_dir(tmp_path: Path) -> Path:
    """A dir holding a deterministic 300,000-byte code-named file + a logs/ fixture.

    Deterministic by construction (NOT the live ``scripts/workflow_lint.py``,
    which could legitimately shrink below the 256 KB threshold and make the
    incident repro flaky — plan §4b item 2).
    """
    (tmp_path / "big_source.py").write_bytes(b"x" * 300_000)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "train.log").write_text("line\n")
    return tmp_path


def test_self_test_passes() -> None:
    r = subprocess.run(
        ["bash", str(SCRIPT), "--self-test"], text=True, capture_output=True, env=_env()
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "self-test: PASS" in r.stdout, r.stdout


def test_incident_986_bounded_range_allowed(big_source_dir: Path) -> None:
    # The #986 shape: a bounded 301-line sed range of a >256 KB source file.
    r = _run_bash("sed -n '100,400p' big_source.py", cwd=str(big_source_dir))
    assert r.returncode == 0, (r.returncode, r.stderr)


def test_cat_big_source_still_blocked(big_source_dir: Path) -> None:
    r = _run_bash("cat big_source.py", cwd=str(big_source_dir))
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "BLOCKED" in r.stderr, r.stderr


def test_log_wide_range_still_blocked(big_source_dir: Path) -> None:
    r = _run_bash("sed -n '1,5000p' logs/train.log", cwd=str(big_source_dir))
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "BLOCKED" in r.stderr, r.stderr


def test_codedoc_block_message_names_retry_shape(big_source_dir: Path) -> None:
    # Pins the BLOCK_KIND=codedoc message branch (the #986 blocked-retry UX
    # fix), not just the exit code: the agent is told the shape that passes.
    r = _run_bash("sed -n '1,5000p' big_source.py", cwd=str(big_source_dir))
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "Bounded range reads ARE allowed" in r.stderr, r.stderr
    assert "2000" in r.stderr, r.stderr


def _main_repo_root() -> str | None:
    """Canonical main-checkout root (parent of the shared .git common dir)."""
    r = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return str(Path(r.stdout.strip()).parent)


class TestSettingsWiring:
    """Parse .claude/settings.json and exercise the CONFIGURED command.

    Every other test drives ``SCRIPT`` directly; without this class a wrong
    command path / lost +x ships the hook inert with a green suite. Pre-merge
    worktree runs remap ONLY the canonical-root PREFIX of the configured
    absolute path onto this checkout (the #965 convention — wrong-directory /
    wrong-basename / missing-+x bugs still fail under the remap); on the main
    checkout no remap occurs.
    """

    def _configured_command(self) -> Path:
        settings = json.loads(SETTINGS.read_text())
        cmds = [
            h["command"]
            for entry in settings["hooks"]["PreToolUse"]
            if entry.get("matcher") == "Bash"
            for h in entry.get("hooks", [])
            if h.get("type") == "command" and h.get("command", "").endswith(HOOK_REL)
        ]
        assert len(cmds) == 1, cmds
        cmd = cmds[0]
        assert os.path.isabs(cmd), cmd
        main_root = _main_repo_root()
        if main_root is not None and str(_REPO_ROOT) != main_root:
            prefix = main_root.rstrip("/") + "/"
            if cmd.startswith(prefix):
                cmd = str(_REPO_ROOT / cmd[len(prefix) :])
        return Path(cmd)

    def test_registered_under_bash_matcher(self) -> None:
        self._configured_command()

    def test_configured_command_exists_and_is_executable(self) -> None:
        cmd = self._configured_command()
        assert cmd.exists(), cmd
        assert os.access(cmd, os.X_OK), cmd

    def test_configured_command_blocks_log_dump(self) -> None:
        r = _run_bash("cat logs/train.log", script=self._configured_command())
        assert r.returncode == 2, (r.returncode, r.stderr)
        assert "BLOCKED" in r.stderr, r.stderr
