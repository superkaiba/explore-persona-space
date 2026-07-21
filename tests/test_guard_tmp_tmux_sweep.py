"""End-to-end tests for the ``.claude/hooks/guard_tmp_tmux_sweep.sh`` PreToolUse hook.

The guard blocks broad /tmp deletion SWEEPS that can destroy the fleet tmux
socket dir ``/tmp/tmux-<uid>`` (task #1474; incident #1466, 2026-07-15: an
improvised age sweep with no ``tmux-*`` exclusion deleted ``/tmp/tmux-1001``
and split 39 sessions off the fleet). Blocked: ``rm`` on /tmp itself / a
tmux-capable top-level glob / any ``/tmp/tmux*`` path; ``find`` rooted at
standalone /tmp with a deletion action and no tmux exclusion; ``find`` rooted
at ``/tmp/tmux*`` with a deletion action. Allowed: the fleet's hourly narrow
/tmp cleanups (explicit paths, non-tmux-capable globs, variable targets,
subdir-rooted / deletion-free finds), reader-leading units (grep/rg/echo/
printf/ssh), heredoc-bearing commands, and the ``EPM_ALLOW_TMP_SWEEP=1``
escape hatch. Fail-open on malformed input.

These tests drive the script exactly as the harness does: stdin PreToolUse
JSON ``{"tool_input": {"command": ...}}`` -> exit 2 (block) or exit 0
(allow) — the subprocess-drives-script convention of
``tests/test_guard_piped_git_push.py``. Env hygiene: block/allow cases run
with ``EPM_ALLOW_TMP_SWEEP`` scrubbed; the escape-hatch case sets it
explicitly. NOTE (self-reference): the command strings below MENTION /tmp
sweeps as test DATA — they are never executed; the guard only reads them
from stdin JSON and never stats paths.

Case ids B1-B17 / A1-A23 are the plan #1474 §5 acceptance tables
(``tasks/*/1474/plans/plan.md``); B18-B22 / A24-A25 are the critic-round
boundary rows (trailing-slash find root, -execdir / ``xargs -I{}`` deletion
forms, quoted find root, backtick command substitution, the ``-not -name``
exclusion variant, and the ``ls | xargs rm`` reader-pipeline documented
miss). B17 is the pinned deliberate raw-scan false positive (quoted sweep in
a marker ``--note``); A23/A25 pin the leading-reader skip's accepted misses.

The settings-wiring tests (module-level adaptations of the
``TestSettingsWiring`` precedent in ``tests/test_guard_piped_git_push.py``)
parse ``.claude/settings.json``, assert the matcher-Bash hook group carries
the configured command path, assert the file exists + executable bit, and
invoke THE CONFIGURED COMMAND end-to-end on one block + one allow case —
closing the "hook ships green but inert" channel (matcher typo / wrong
command path / missing +x). When this suite runs from a pre-merge worktree,
the canonical-root PREFIX of the configured absolute path is remapped onto
this checkout (the rest of the path is exercised verbatim, so a wrong
directory / basename / missing +x still fails); on the main checkout no
remap occurs.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _REPO_ROOT / ".claude" / "hooks" / "guard_tmp_tmux_sweep.sh"
SETTINGS = _REPO_ROOT / ".claude" / "settings.json"

# The #1466 incident command, verbatim (recorded in
# .claude/rules/background-automation.md § tmux socket-dir contract).
INCIDENT_CMD = (
    "find /tmp -maxdepth 1 -mtime +2 ! -name 'claude-*' ! -name 'systemd-*' "
    "! -name 'snap-*' -user \"$(id -un)\" -print0 | xargs -0 -r rm -rf"
)


def _env(*, allow: bool = False) -> dict[str, str]:
    """Hook env: EPM_ALLOW_TMP_SWEEP scrubbed (deny hygiene) unless ``allow``."""
    env = {k: v for k, v in os.environ.items() if k != "EPM_ALLOW_TMP_SWEEP"}
    if allow:
        env["EPM_ALLOW_TMP_SWEEP"] = "1"
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
# BLOCK — plan §5 B1-B16 + critic-round boundary rows B18-B22 (exit 2)
# ---------------------------------------------------------------------------
BLOCK_CASES = [
    pytest.param(INCIDENT_CMD, id="B1-1466-incident-verbatim"),
    pytest.param("find /tmp -mtime +2 -delete", id="B2-find-delete"),
    pytest.param("find /tmp -maxdepth 1 -type d -exec rm -rf {} +", id="B3-find-exec-rm"),
    pytest.param("find /tmp -name '*.tmp' | xargs rm", id="B4-find-xargs-rm"),
    pytest.param("rm -rf /tmp/*", id="B5-bare-top-level-glob"),
    pytest.param("rm -rf /tmp", id="B6-tmp-itself"),
    pytest.param("rm -rf /tmp/", id="B7-tmp-trailing-slash"),
    pytest.param("rm -rf /tmp/tmux-*", id="B8-asset-glob"),
    pytest.param("rm -rf /tmp/tmux-1001", id="B9-asset-literal"),
    pytest.param("rm /tmp/tmux-1001/default", id="B10-socket-file-nonrecursive"),
    pytest.param("sudo rm -rf /tmp/*", id="B11-sudo-prefix"),
    pytest.param("find /tmp/tmux-1001 -type s -delete", id="B12-asset-rooted-find"),
    pytest.param("cd /workspace && find /tmp -mtime +1 -delete", id="B13-later-and-unit"),
    pytest.param("rm -rf /tmp/t*", id="B14-tmux-capable-glob-prefix"),
    pytest.param("rm -rf $(find /tmp -mtime +2)", id="B15-cmdsub-feeding-rm"),
    pytest.param("find /tmp ! -name 'claude-*' -delete", id="B16-non-tmux-exclusion"),
    # Critic-round boundary rows:
    pytest.param("find /tmp/ -mtime +2 -delete", id="B18-trailing-slash-find-root"),
    pytest.param("find /tmp -maxdepth 1 -execdir rm -rf {} \\;", id="B19-execdir-deletion"),
    pytest.param(
        "find /tmp -maxdepth 1 -print0 | xargs -0 -I{} rm -rf {}",
        id="B20-xargs-I-deletion",
    ),
    pytest.param('find "/tmp" -mtime +2 -delete', id="B21-quoted-find-root"),
    # B22 pins the backtick command-substitution form to the rm-arm token
    # walk (the bare `/tmp` token is attributed to rm) — a refactor of the
    # $(...) cmdsub regex must not lose it.
    pytest.param("rm -rf `find /tmp -mtime +2`", id="B22-backtick-cmdsub"),
]


@pytest.mark.parametrize("cmd", BLOCK_CASES)
def test_block_cases(cmd: str) -> None:
    _assert_blocked(_run_bash(cmd))


def test_b17_pinned_quoted_note_false_positive() -> None:
    """B17 — pinned EXPECTED-BLOCK (deliberate FP trade-off): a marker
    ``--note`` string that merely QUOTES a sweep trips the raw scan — the
    guard does not strip quoted arguments (the guard family trade-off, cf.
    guard_piped_git_push.sh S7r1). Remediation is
    ``task.py post-marker --file <path.md>`` (named in the block message).
    """
    _assert_blocked(
        _run_bash(
            "uv run python scripts/task.py post-marker 1474 epm:progress "
            "--note 'ran find /tmp -mtime +2 | xargs rm -rf'"
        )
    )


# ---------------------------------------------------------------------------
# ALLOW — plan §5 A1-A22 + the -not exclusion boundary row A24 (exit 0)
# ---------------------------------------------------------------------------
ALLOW_CASES = [
    pytest.param("rm -f /tmp/issue-1474-lint-verdict.txt", id="A1-explicit-path"),
    pytest.param(
        "rm -f /tmp/step9c-junit-issue-1474.xml /tmp/step9c-rc-issue-1474",
        id="A2-explicit-multi-file",
    ),
    pytest.param("rm -rf /tmp/issue-1474-lint-gate-tree", id="A3-explicit-dir"),
    pytest.param(
        'SCRATCH=/tmp/issue-1474-postmerge-scratch; rm -rf "$SCRATCH"',
        id="A4-variable-target",
    ),
    pytest.param("rm -f /tmp/claude-1001/*/*/tasks/*.output", id="A5-deep-glob-literal-first"),
    pytest.param("rm -rf /tmp/issue-*", id="A6-tmux-impossible-glob-prefix"),
    pytest.param(
        INCIDENT_CMD.replace("! -name 'claude-*'", "! -name 'tmux-*' ! -name 'claude-*'"),
        id="A7-remediated-incident",
    ),
    pytest.param(
        "find /tmp -maxdepth 1 -mtime +2 ! -name 'tmux-*' -delete",
        id="A8-excluded-find-delete",
    ),
    pytest.param("find /tmp -maxdepth 1 -type s", id="A9-deletion-free-probe"),
    pytest.param("find /tmp/issue-1474-scratch -delete", id="A10-subdir-rooted-find"),
    pytest.param("ls /tmp && df -h /tmp", id="A11-no-deletion-verbs"),
    pytest.param("grep -rnE 'find /tmp|rm -rf? /tmp' scripts/", id="A12-leading-grep-unit"),
    # A13: heredoc-bearing commit quoting a sweep — the heredoc blanket-allow
    # is load-bearing (commit messages describing the #1466 incident must not
    # false-block the implementing session's own commits).
    pytest.param(
        'git commit -m "$(cat <<EOF\n'
        "never run find /tmp -mtime +2 | xargs rm -rf without a tmux exclusion\n"
        'EOF\n)"',
        id="A13-heredoc-commit",
    ),
    pytest.param("EPM_ALLOW_TMP_SWEEP=1 find /tmp -mtime +2 -delete", id="A14-inline-escape"),
    pytest.param("trap \"rm -rf '$TMP'\" EXIT", id="A15-mktemp-cleanup-trap"),
    pytest.param("mv /tmp/foo /tmp/bar", id="A16-mv-not-deletion"),
    pytest.param(
        "find /tmp -path '/tmp/tmux-*' -prune -o -mtime +2 -delete",
        id="A17-prune-carve-out",
    ),
    pytest.param(
        "echo 'find /tmp -mtime +2 | xargs rm -rf' >> notes.md",
        id="A18-leading-echo-unit",
    ),
    pytest.param("rm -rf /workspace/tmp/*", id="A19-non-tmp-path-containing-tmp"),
    pytest.param("ssh pod-1474 'rm -rf /tmp/hf_stage'", id="A20-remote-tmp-ssh"),
    pytest.param("find /tmp -maxdepth 1 -mtime +12", id="A21-forensic-probe"),
    pytest.param("rm -f /tmp/wf-fix-body-guard.md", id="A22-filer-temp-cleanup"),
    # Critic-round boundary row: the -not spelling of the name exclusion.
    pytest.param(
        "find /tmp -maxdepth 1 -mtime +2 -not -name 'tmux-*' -delete",
        id="A24-not-name-exclusion-variant",
    ),
]


@pytest.mark.parametrize("cmd", ALLOW_CASES)
def test_allow_cases(cmd: str) -> None:
    _assert_allowed(_run_bash(cmd))


def test_a23_echo_pipeline_documented_known_miss() -> None:
    """A23 — DOCUMENTED KNOWN-MISS: ``echo /tmp/tmux-1001 | xargs rm -rf``
    is ALLOWED by the leading-reader skip (echo cannot delete; the xargs-fed
    target is invisible to the rm token walk outside the find-arm). Pinned so
    the accepted residual stays visible and deliberate, never accidental —
    the prose rule + the #1466 durable socket dir remain defense in depth.
    """
    _assert_allowed(_run_bash("echo /tmp/tmux-1001 | xargs rm -rf"))


def test_a25_ls_pipeline_documented_known_miss() -> None:
    """A25 — DOCUMENTED KNOWN-MISS (generic reader-pipeline residual, sibling
    of A23): ``ls /tmp | xargs rm -rf`` is ALLOWED — ``ls`` is not in the
    reader-skip set, but the deletion targets are xargs-fed, so neither the
    find-arm (no find) nor the rm token walk (no /tmp-rooted token attributed
    to rm) fires. Pinned deliberate.
    """
    _assert_allowed(_run_bash("ls /tmp | xargs rm -rf"))


def test_env_escape_hatch_allows_incident_shape() -> None:
    """Session env EPM_ALLOW_TMP_SWEEP=1 allows the B2 block shape."""
    _assert_allowed(_run_bash("find /tmp -mtime +2 -delete", env=_env(allow=True)))


def test_empty_command_allowed() -> None:
    """Fail-open: empty command string -> exit 0."""
    _assert_allowed(_run_bash(""))


def test_malformed_stdin_json_allowed() -> None:
    """Fail-open: malformed stdin JSON (jq parse failure) -> exit 0."""
    _assert_allowed(_run("this is not json"))


def test_missing_command_field_allowed() -> None:
    """Fail-open: well-formed JSON with no tool_input.command -> exit 0."""
    _assert_allowed(_run({"tool_input": {}}))


def test_block_message_names_incident_and_remediations() -> None:
    """The block message names the copy-paste fix, the incident, and the
    override + --file remediation (plan §1 acceptance criterion 2)."""
    r = _run_bash("find /tmp -mtime +2 -delete")
    _assert_blocked(r)
    for needle in (
        "BLOCKED",
        "! -name 'tmux-*'",
        "EPM_ALLOW_TMP_SWEEP=1",
        "#1466",
        "--file",
    ):
        assert needle in r.stderr, (needle, r.stderr)


def test_self_test_mode_passes() -> None:
    """`--self-test` runs the in-script §5 acceptance tables and exits 0."""
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
        matches = [c for c in cmds if os.path.basename(c) == "guard_tmp_tmux_sweep.sh"]
        assert len(matches) == 1, (
            f"expected exactly one guard_tmp_tmux_sweep.sh command in the "
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


def test_configured_command_blocks_tmp_glob() -> None:
    """B5 via the configured path — end-to-end through the registration."""
    r = _run_bash("rm -rf /tmp/*", script=_configured_command())
    _assert_blocked(r)


def test_configured_command_allows_explicit_path() -> None:
    """A1 via the configured path — end-to-end through the registration."""
    r = _run_bash("rm -f /tmp/issue-1474-lint-verdict.txt", script=_configured_command())
    _assert_allowed(r)
