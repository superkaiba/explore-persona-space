"""Pin tests for the guard-script read-bounding PreToolUse hook (#1577).

``.claude/hooks/guard_trigger_dense_read.sh`` mechanizes the READ channel of
``.claude/rules/trigger-dense-review.md`` § "Orchestrator ordinary turns"
item 2: a ``Read`` of a workflow guard script (the deny-set ERE
``(^|/)(scripts|\\.claude/hooks)/guard_[^/]+$``, raw + realpath) with no
``limit`` — or a ``limit`` over the cap (``EPM_GUARD_READ_CAP_LINES``,
default 120) — is denied (exit 2); windowed reads (``limit <= cap``, any
offset) pass; ``EPM_ALLOW_GUARD_READ=1`` (session env) allows everything;
every parse/availability failure fails OPEN (exit 0).

Convention: subprocess-drives-script, per ``tests/test_guard_harmful_bank_read.py``
— feed the PreToolUse stdin JSON exactly as the harness does and assert on
the exit code. Deny cases scrub ``EPM_ALLOW_GUARD_READ`` and
``EPM_GUARD_READ_CAP_LINES`` from the environment.

CONTENT HYGIENE: guard scripts are trigger-dense artifacts. These tests use
SYNTHETIC PATH STRINGS only — no test reads or embeds any guard file's
content (the hook itself matches on the path string; ``realpath -m`` needs
no existing file).

``TestSettingsWiring`` additionally parses ``.claude/settings.json`` and
invokes the CONFIGURED command end-to-end, so a matcher typo / wrong path /
missing +x cannot ship the hook inert with a green suite. When this suite
runs from a pre-merge worktree, the canonical-root prefix of the configured
command is remapped onto this checkout (the precedent's exact pattern).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _REPO_ROOT / ".claude" / "hooks" / "guard_trigger_dense_read.sh"
SETTINGS = _REPO_ROOT / ".claude" / "settings.json"

# Deployed-path convention: settings.json registers hooks by canonical
# repo-root absolute path; deny-set probes use the same canonical prefix
# (path STRINGS only — content hygiene, see module docstring).
CANONICAL_ROOT = "/home/thomasjiralerspong/explore-persona-space"
GUARD_ABS = f"{CANONICAL_ROOT}/scripts/guard_repo_root_branch.sh"
GUARD_PY_HOOK = f"{CANONICAL_ROOT}/.claude/hooks/guard_lessons_edit_check.py"
GUARD_WORKTREE_COPY = (
    f"{CANONICAL_ROOT}/.claude/worktrees/issue-9999/scripts/guard_repo_root_pull.sh"
)
HOOK_BASENAME = "guard_trigger_dense_read.sh"


def _env(
    extra: dict[str, str] | None = None, *, allow: bool = False, cap: str | None = None
) -> dict[str, str]:
    """Hook env: override + cap scrubbed (deny hygiene); opt back in per test."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("EPM_ALLOW_GUARD_READ", "EPM_GUARD_READ_CAP_LINES")
    }
    if allow:
        env["EPM_ALLOW_GUARD_READ"] = "1"
    if cap is not None:
        env["EPM_GUARD_READ_CAP_LINES"] = cap
    if extra:
        env.update(extra)
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


def _run_read(
    file_path: str,
    *,
    limit: int | str | None = None,
    offset: int | None = None,
    **kw,
) -> subprocess.CompletedProcess[str]:
    tool_input: dict = {"file_path": file_path}
    if limit is not None:
        tool_input["limit"] = limit
    if offset is not None:
        tool_input["offset"] = offset
    return _run({"tool_name": "Read", "tool_input": tool_input}, **kw)


def _assert_denied(r: subprocess.CompletedProcess[str]) -> None:
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "BLOCKED" in r.stderr, r.stderr


def _assert_allowed(r: subprocess.CompletedProcess[str]) -> None:
    assert r.returncode == 0, (r.returncode, r.stderr)


# --- Acceptance 1: unbounded / over-cap reads of deny-set files are denied ----


def test_deny_wholesale_read_no_limit() -> None:
    r = _run_read(GUARD_ABS)
    _assert_denied(r)
    # Deny message names the windowed recipe, the override, and the rule.
    assert "offset=" in r.stderr and "limit" in r.stderr, r.stderr
    assert "EPM_ALLOW_GUARD_READ" in r.stderr, r.stderr
    assert ".claude/rules/trigger-dense-review.md" in r.stderr, r.stderr


def test_deny_read_limit_over_cap() -> None:
    _assert_denied(_run_read(GUARD_ABS, limit=500))


def test_deny_hooks_dir_member() -> None:
    # The .py helper under .claude/hooks/ is in-set too.
    _assert_denied(_run_read(GUARD_PY_HOOK))


def test_deny_worktree_copy() -> None:
    # ".../worktrees/issue-N/scripts/guard_x.sh" contains "/scripts/guard_x.sh".
    _assert_denied(_run_read(GUARD_WORKTREE_COPY))


def test_deny_non_numeric_limit() -> None:
    # limit present but non-numeric on a matched path: deny-on-ambiguity
    # (NOT the malformed-input fail-open case — the path match succeeded).
    _assert_denied(_run_read(GUARD_ABS, limit="abc"))


# --- Acceptance 2: windowed reads pass (<= cap, any offset) -------------------


def test_allow_windowed_read() -> None:
    _assert_allowed(_run_read(GUARD_ABS, limit=100, offset=800))


def test_allow_windowed_read_at_cap_boundary() -> None:
    _assert_allowed(_run_read(GUARD_ABS, limit=120))


# --- Acceptance 3: env override allows everything -----------------------------


def test_override_env_allows() -> None:
    _assert_allowed(_run_read(GUARD_ABS, env=_env(allow=True)))


# --- Cap tunability (EPM_GUARD_READ_CAP_LINES) --------------------------------


def test_cap_env_tunable() -> None:
    _assert_denied(_run_read(GUARD_ABS, limit=100, env=_env(cap="50")))
    _assert_allowed(_run_read(GUARD_ABS, limit=50, env=_env(cap="50")))


def test_cap_env_junk_falls_back_to_default() -> None:
    _assert_allowed(_run_read(GUARD_ABS, limit=120, env=_env(cap="junk")))
    _assert_denied(_run_read(GUARD_ABS, limit=121, env=_env(cap="junk")))


# --- Acceptance 5: non-deny-set paths pass through untouched ------------------


@pytest.mark.parametrize(
    "path",
    [
        f"{CANONICAL_ROOT}/scripts/task.py",  # no guard_ token at all
        f"{CANONICAL_ROOT}/src/x/guard_thing.py",  # guard_ outside the two dirs
        "tests/test_guard_repo_root_branch.py",  # pin tests are legitimately read whole
    ],
)
def test_pass_through_non_guard_file(path: str) -> None:
    _assert_allowed(_run_read(path))


# --- Acceptance 4: fail-open paths --------------------------------------------


def test_pass_through_other_tool() -> None:
    r = _run({"tool_name": "Bash", "tool_input": {"command": f"wc -l {GUARD_ABS}"}})
    _assert_allowed(r)


def test_fail_open_malformed_json() -> None:
    _assert_allowed(_run("not json {{{"))


def test_fail_open_empty_file_path() -> None:
    # Well-formed JSON, file_path absent (AC4's empty-path sub-case,
    # distinct from malformed JSON).
    _assert_allowed(_run({"tool_name": "Read", "tool_input": {}}))


def test_fail_open_missing_jq(tmp_path: Path) -> None:
    """No jq on PATH -> exit 0 (a broken guard must never brick Read)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    # The shebang is `#!/usr/bin/env bash`, so bash itself must resolve via
    # the restricted PATH too; jq is deliberately absent.

    for tool in ("bash", "sh", "cat", "tr", "grep", "realpath"):
        src = shutil.which(tool)
        assert src is not None, tool
        (bindir / tool).symlink_to(src)
    env = _env(extra={"PATH": str(bindir)})
    _assert_allowed(_run_read(GUARD_ABS, env=env))


# --- Acceptance 6: settings.json registration (end-to-end, never inert) -------


def _main_repo_root() -> str | None:
    r = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return str(Path(r.stdout.strip()).parent)


class TestSettingsWiring:
    """Parse .claude/settings.json and invoke the CONFIGURED command.

    Every other test drives ``SCRIPT`` directly; without this class a matcher
    typo / wrong command path / missing +x ships the mechanical layer inert
    with a green suite. The command path is read FROM settings, never from
    the repo constant. Pre-merge worktree runs remap ONLY the canonical-root
    prefix onto this checkout (see module docstring); wrong-directory /
    wrong-basename / missing-+x bugs still fail under the remap.
    """

    def _guard_read_entry(self) -> dict:
        settings = json.loads(SETTINGS.read_text())
        for entry in settings["hooks"]["PreToolUse"]:
            cmds = [h.get("command", "") for h in entry.get("hooks", [])]
            if any(os.path.basename(c) == HOOK_BASENAME for c in cmds):
                # The matcher alternation must cover the Read tool.
                matcher = entry.get("matcher", "")
                assert "Read" in matcher.split("|"), matcher
                return entry
        pytest.fail(f"no hooks.PreToolUse entry registers {HOOK_BASENAME}")

    def _configured_command(self) -> Path:
        entry = self._guard_read_entry()
        cmds = [h["command"] for h in entry.get("hooks", []) if h.get("type") == "command"]
        assert len(cmds) == 1, cmds
        cmd = cmds[0]
        assert os.path.isabs(cmd), cmd
        assert os.path.basename(cmd) == HOOK_BASENAME, cmd
        main_root = _main_repo_root()
        if main_root is not None and str(_REPO_ROOT) != main_root:
            prefix = main_root.rstrip("/") + "/"
            if cmd.startswith(prefix):
                cmd = str(_REPO_ROOT / cmd[len(prefix) :])
        return Path(cmd)

    def test_matcher_group_present_with_single_command_hook(self) -> None:
        self._configured_command()

    def test_configured_command_exists_and_is_executable(self) -> None:
        cmd = self._configured_command()
        assert cmd.exists(), cmd
        assert os.access(cmd, os.X_OK), cmd

    def test_configured_command_denies_unbounded_guard_read(self) -> None:
        _assert_denied(_run_read(GUARD_ABS, script=self._configured_command()))

    def test_configured_command_allows_windowed_read(self) -> None:
        _assert_allowed(_run_read(GUARD_ABS, limit=100, script=self._configured_command()))
