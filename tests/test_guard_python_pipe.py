"""End-to-end tests for the ``.claude/hooks/guard_python_pipe.sh`` PreToolUse hook.

The guard blocks a bare ``| python -c/-m`` pipe CONSUMER in LOCAL argv —
this VM has no ``python`` on PATH, so such a pipe dies ``python: command
not found`` (exit 127) — while EXEMPTING pipes that occur inside quoted
REMOTE command strings (``ssh pod-X '... | python3 -c ...'``,
``gcloud compute ssh vm --command='... | python3 -c ...'``), where the
no-python-on-PATH premise does not apply (task #2009; the guard was
formerly an INLINE command in ``.claude/settings.json`` whose regex matched
anywhere in the argv, remote strings included). Normalization before
matching is the #1675 quoted-span strip ported verbatim from
``guard_piped_git_push.sh``: bash line-continuation join, then strip
single-quoted + substitution-free double-quoted spans while PRESERVING
``$``/backtick-bearing double-quoted spans as one atomic token (live
substitution executes locally, so its interior stays scannable).

These tests drive the script exactly as the harness does: stdin PreToolUse
JSON ``{"tool_input": {"command": ...}}`` -> exit 2 (block) or exit 0
(allow) — the subprocess-drives-script convention of
``tests/test_guard_piped_git_push.py`` / ``tests/test_guard_tmp_tmux_sweep.py``.
Env hygiene: block/allow cases run with ``EPM_ALLOW_PYTHON_PIPE`` scrubbed;
the escape-hatch cases set it explicitly. NOTE (self-reference): the
command strings below MENTION python pipes as test DATA — they are never
executed; the guard only reads them from stdin JSON.

Case ids map to the task #2009 plan §1 acceptance criteria
(``tasks/*/2009/plans/plan.md``): A-cases pin the remote-string /
``uv run python`` allows (acceptance 1, 2, 5), B-cases the genuinely-local
block parity set (acceptance 3, 4), R-cases the plan §3.3
residual/robustness pins (§5 residuals: ``$``-bearing double-quoted remote
string, unbalanced-quote raw scan, the two strip-atomicity
counter-shapes), and W-cases the §1.9 wrapper-quoted DELIBERATE fail-open
disposition pin.

The settings-wiring tests (module-level adaptation of the
``TestSettingsWiring`` precedent in ``tests/test_guard_piped_git_push.py``)
parse ``.claude/settings.json``, assert the matcher-Bash hook group carries
the configured command path, assert the file exists + executable bit,
invoke THE CONFIGURED command end-to-end on one block + one allow case,
and assert the OLD inline ``python3?``-pipe grep is GONE from every
settings.json hook command (acceptance 7). When this suite runs from a
pre-merge worktree, the canonical-root PREFIX of the configured absolute
path is remapped onto this checkout (the rest of the path is exercised
verbatim, so a wrong directory / basename / missing +x still fails); on
the main checkout no remap occurs.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _REPO_ROOT / ".claude" / "hooks" / "guard_python_pipe.sh"
SETTINGS = _REPO_ROOT / ".claude" / "settings.json"

# The 2026-08-01 incident shape (task #2009 body: a remote ssh-quoted
# `python3 -c` pipe was false-blocked by the inline guard).
INCIDENT_CMD = "ssh pod-2009 'ps aux | grep train | python3 -c \"import sys; print(1)\"'"


def _env(*, allow: bool = False) -> dict[str, str]:
    """Hook env: EPM_ALLOW_PYTHON_PIPE scrubbed (deny hygiene) unless ``allow``."""
    env = {k: v for k, v in os.environ.items() if k != "EPM_ALLOW_PYTHON_PIPE"}
    if allow:
        env["EPM_ALLOW_PYTHON_PIPE"] = "1"
    return env


def _run(
    payload: dict | str,
    *,
    env: dict[str, str] | None = None,
    script: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Feed a PreToolUse payload (dict -> JSON, str -> raw) to the guard."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        [str(script or SCRIPT)],
        input=raw,
        text=True,
        capture_output=True,
        env=env if env is not None else _env(),
    )


def _run_bash(cmd: str, **kw) -> subprocess.CompletedProcess[str]:
    return _run({"tool_input": {"command": cmd}}, **kw)


def _assert_blocked(r: subprocess.CompletedProcess[str]) -> None:
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "BLOCKED" in r.stderr, r.stderr


def _assert_allowed(r: subprocess.CompletedProcess[str]) -> None:
    assert r.returncode == 0, (r.returncode, r.stderr)


# ---------------------------------------------------------------------------
# BLOCK — genuinely LOCAL pipes (plan §1 acceptance 3-4 + parity shapes)
# ---------------------------------------------------------------------------
BLOCK_CASES = [
    pytest.param(
        'cat x.json | python3 -c "import sys,json; print(json.load(sys.stdin))"',
        id="B1-plain-local-pipe",
    ),
    pytest.param("echo '{}' | python -c \"print(1)\"", id="B2-echo-producer-bare-python"),
    pytest.param("foo | python3.11 -m json.tool", id="B3-versioned-python-dash-m"),
    pytest.param(
        "ssh host 'cat /workspace/train.log' | python3 -c 'import sys'",
        id="B4-local-consumer-after-ssh-stage",
    ),
    pytest.param(
        "echo \"$(cat x | python3 -c 'print(1)')\"",
        id="B5-live-substitution-executes-locally",
    ),
    pytest.param("cat x | python -c'print(1)'", id="B6-attached-arg-parity"),
    pytest.param('foo |python -c "x"', id="B7-no-space-after-pipe-parity"),
    pytest.param(
        "echo pre\ncat x | python3 -c 'y'",
        id="B8-raw-newline-multiline-parity",
    ),
]


@pytest.mark.parametrize("cmd", BLOCK_CASES)
def test_block_cases(cmd: str) -> None:
    _assert_blocked(_run_bash(cmd))


# ---------------------------------------------------------------------------
# ALLOW — remote strings, uv run python, quoted mentions (acceptance 1/2/5)
# ---------------------------------------------------------------------------
ALLOW_CASES = [
    pytest.param(INCIDENT_CMD, id="A1-incident-ssh-single-quoted-remote-string"),
    pytest.param(
        "ssh pod-2009 \"nvidia-smi | python3 -c 'import sys'\"",
        id="A2-ssh-double-quoted-substitution-free",
    ),
    pytest.param(
        "gcloud compute ssh eps-issue-2009 --configuration=eps-gcp "
        "--command='df -h /workspace | python3 -c \"print(1)\"'",
        id="A3-gcloud-compute-ssh-command-string",
    ),
    pytest.param('cat x | uv run python -c "import sys, json"', id="A4-uv-run-python-consumer"),
    pytest.param("python3 -c 'print(1)'", id="A5-no-pipe-out-of-scope"),
    pytest.param("echo 'never | python3 -c inline' | wc -l", id="A6-quoted-mention-piped"),
    pytest.param(
        'grep -n "| python3 -c" scripts/foo.sh | head -5',
        id="A7-quoted-grep-pattern-mention",
    ),
    pytest.param("ps aux | grep python3", id="A8-non-python-pipe-consumer"),
]


@pytest.mark.parametrize("cmd", ALLOW_CASES)
def test_allow_cases(cmd: str) -> None:
    _assert_allowed(_run_bash(cmd))


# ---------------------------------------------------------------------------
# W — the §1.9 wrapper-quoted DELIBERATE fail-open disposition pin
# ---------------------------------------------------------------------------
WRAPPER_CASES = [
    pytest.param("bash -c 'cat x | python3 -c \"y\"'", id="W1-bash-c-single-quoted-payload"),
    pytest.param("sh -c \"cat x | python3 -c 'y'\"", id="W2-sh-c-substitution-free-payload"),
    pytest.param("eval 'cat x | python3 -c \"y\"'", id="W3-eval-quoted-payload"),
    pytest.param(
        "find . -name '*.json' | xargs -I{} sh -c 'cat {} | python3 -c \"y\"'",
        id="W4-xargs-sh-c-payload",
    ),
]


@pytest.mark.parametrize("cmd", WRAPPER_CASES)
def test_wrapper_quoted_local_pipe_allowed_disposition_pin(cmd: str) -> None:
    """Plan §1.9 disposition pin: wrapper-quoted LOCAL pipes are ALLOWED.

    A quoted payload handed to a local wrapper (``bash -c`` / ``sh -c`` /
    ``eval`` / ``xargs ... sh -c``) executes downstream but is string data
    to the guard post-strip. This is the DELIBERATE interpreter-payload
    fail-open (task #2009 plan §5; sibling precedent:
    guard_piped_git_push.sh header fail-open class (1) — cooperative-agent
    threat model, harm bound = one exit-127 turn). These shapes BLOCKED
    under the former inline guard; the ALLOW here is a conscious,
    plan-sanctioned behavior change — a future change of heart must edit
    this test deliberately, never drift silently.
    """
    _assert_allowed(_run_bash(cmd))


# ---------------------------------------------------------------------------
# R — plan §3.3 residual/robustness pins (§5 fail-closed residuals)
# ---------------------------------------------------------------------------
def test_dollar_bearing_double_quoted_remote_string_still_blocks() -> None:
    """§5 residual: a ``$``-bearing double-quoted remote string is PRESERVED
    as live local substitution and still blocks — remediation is to
    single-quote the remote string (named in the BLOCK message)."""
    _assert_blocked(_run_bash("ssh host \"echo $HOME | python3 -c 'print(1)'\""))


def test_unbalanced_quote_scans_raw_fails_toward_block() -> None:
    """§5 residual: an unmatched quote leaves the span unstripped, so the
    text is scanned raw — fails toward BLOCK (the pre-#2009 status quo),
    never a new false negative."""
    _assert_blocked(_run_bash("cat x | python3 -c 'print(1)"))


def test_preserved_span_interior_apostrophe_still_blocks() -> None:
    """§3.3 strip-robustness pin: an apostrophe INSIDE a preserved
    ``$``-bearing double-quoted span is consumed atomically with the span
    (preserve-branch atomicity), so it cannot seed a phantom single-quoted
    span that would swallow the live-substitution pipe."""
    _assert_blocked(_run_bash("echo \"it's $(cat x | python3 -c 'y')\""))


def test_stripped_span_interior_apostrophe_cannot_seed_quote_state() -> None:
    """§3.3 strip-robustness pin: a substitution-free double-quoted span
    with an interior apostrophe strips atomically, so the apostrophe
    cannot open a bogus single-quote state that would swallow the REAL
    local pipe following it."""
    _assert_blocked(_run_bash("echo \"can't\" | python3 -c 'x'"))


def test_backslash_continued_local_pipe_blocked() -> None:
    """Plan §3.1 normalization pin: the line-continuation join makes a
    backslash-continued local pipe scan as ONE logical command. The former
    inline guard's line-based grep MISSED this shape; blocking it is
    fail-closed and correct (bash joins the lines before executing the
    pipe, which then fails exit-127 locally)."""
    _assert_blocked(_run_bash("cat x | \\\n  python3 -c 'y'"))


# ---------------------------------------------------------------------------
# Escape hatch + stdin contract
# ---------------------------------------------------------------------------
def test_inline_escape_hatch_allows() -> None:
    _assert_allowed(_run_bash('EPM_ALLOW_PYTHON_PIPE=1 cat x | python3 -c "y"'))


def test_env_escape_hatch_allows() -> None:
    _assert_allowed(_run_bash('cat x | python3 -c "y"', env=_env(allow=True)))


def test_empty_command_allowed() -> None:
    _assert_allowed(_run({"tool_input": {"command": ""}}))


def test_missing_command_field_allowed() -> None:
    _assert_allowed(_run({"tool_input": {}}))


def test_malformed_stdin_json_allowed() -> None:
    _assert_allowed(_run("not-json"))


def test_block_message_original_text_and_remediations() -> None:
    """Plan §3.1: the pre-existing BLOCK message text is byte-unchanged
    (asserted as an exact substring), and the ONE appended sentence names
    the single-quote-the-remote-string remediation + the escape hatch."""
    r = _run_bash('cat x.json | python3 -c "import sys"')
    assert r.returncode == 2, (r.returncode, r.stderr)
    original = (
        "BLOCKED: bare `| python -c/-m` pipe. This VM has no `python` on PATH — "
        "`python: command not found` (exit 127). Pipe into `uv run python` instead: "
        '`... | uv run python -c "..."`. CLAUDE.md § Task Workflow API.'
    )
    assert original in r.stderr, r.stderr
    assert "single-quote the remote string" in r.stderr, r.stderr
    assert "EPM_ALLOW_PYTHON_PIPE=1" in r.stderr, r.stderr


def test_bash_syntax_clean() -> None:
    """Acceptance 6: ``bash -n`` clean on the hook file."""
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_self_test_mode_passes() -> None:
    r = subprocess.run(
        ["bash", str(SCRIPT), "--self-test"],
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "self-test: PASS" in r.stdout, r.stdout


# ---------------------------------------------------------------------------
# Settings wiring (module-level adaptation of the TestSettingsWiring
# precedent in tests/test_guard_piped_git_push.py)
# ---------------------------------------------------------------------------
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


def _configured_command() -> Path:
    """The guard's command path as REGISTERED in .claude/settings.json.

    Read from settings, never from the repo constant — a matcher typo /
    wrong command path / missing +x would otherwise ship the mechanical
    layer inert with a green suite. Pre-merge worktree runs remap ONLY the
    canonical-root prefix onto this checkout (wrong-directory /
    wrong-basename / missing-+x bugs still fail under the remap).
    """
    settings = json.loads(SETTINGS.read_text())
    for entry in settings["hooks"]["PreToolUse"]:
        if entry.get("matcher") != "Bash":
            continue
        cmds = [h["command"] for h in entry.get("hooks", []) if h.get("type") == "command"]
        matches = [c for c in cmds if os.path.basename(c) == "guard_python_pipe.sh"]
        assert len(matches) == 1, (
            f"expected exactly one guard_python_pipe.sh command in the "
            f"matcher-Bash PreToolUse group, got {matches!r}"
        )
        cmd = matches[0]
        assert os.path.isabs(cmd), cmd
        main_root = _main_repo_root()
        if main_root is not None and str(_REPO_ROOT) != main_root:
            prefix = main_root.rstrip("/") + "/"
            if cmd.startswith(prefix):
                cmd = str(_REPO_ROOT / cmd[len(prefix) :])
        return Path(cmd)
    pytest.fail("no hooks.PreToolUse entry with matcher 'Bash' in .claude/settings.json")


def test_settings_matcher_bash_group_carries_the_hook() -> None:
    _configured_command()


def test_configured_command_exists_and_is_executable() -> None:
    cmd = _configured_command()
    assert cmd.exists(), cmd
    assert os.access(cmd, os.X_OK), cmd  # mechanizes deliverable 1's chmod +x


def test_configured_command_blocks_local_pipe() -> None:
    """B1 via the configured path — end-to-end through the registration."""
    r = _run_bash('cat x.json | python3 -c "import sys"', script=_configured_command())
    _assert_blocked(r)


def test_configured_command_allows_ssh_quoted_remote_pipe() -> None:
    """A1 (the incident shape) via the configured path — end-to-end."""
    r = _run_bash(INCIDENT_CMD, script=_configured_command())
    _assert_allowed(r)


def test_inline_python_pipe_grep_removed_from_settings() -> None:
    """Acceptance 7: the old INLINE python-pipe grep is GONE — no
    settings.json hook command (any event, any matcher) carries the
    ``python3?`` regex fragment or the guard's BLOCK-message marker; the
    guard lives ONLY in the registered hook file."""
    settings = json.loads(SETTINGS.read_text())
    for event, blocks in settings.get("hooks", {}).items():
        for block in blocks:
            for h in block.get("hooks", []):
                cmd = h.get("command", "")
                assert "python3?" not in cmd, (event, cmd)
                assert "BLOCKED: bare" not in cmd, (event, cmd)
