"""End-to-end tests for the ``.claude/hooks/guard_piped_git_push.sh`` PreToolUse hook.

The guard mechanizes CLAUDE.md § Concurrent repo-root committers ("Never pipe
a `git push` (or merge/PR command) through `tail`/`grep`/`head` — the pipe
masks the non-zero exit code ... run it bare and check the exit code, or use
`set -o pipefail` when a pipe is unavoidable"; the prose rule failed open in
4 sessions on 2026-07-02 and masked #957's Step 10d push on 2026-07-04): a
Bash tool call that pipes a ``git push`` / ``git merge`` /
``gh pr merge|create`` PRODUCER into a consumer on the producer's own
pipeline segment is BLOCKED (exit 2 + a ``BLOCKED`` stderr naming the
bare-push + pipefail remediation), while pipefail-carrying commands,
heredoc-bearing commands, ``--dry-run`` pipes, ``||`` chains, ``merge-base``
probes, cross-segment pipes, and producer-as-consumer shapes stay allowed
(exit 0). Fail-soft on malformed input; escape hatch
``EPM_ALLOW_PIPED_PUSH=1`` (session env or inline prefix).

These tests drive the script exactly as the harness does: stdin PreToolUse
JSON ``{"tool_input": {"command": ...}}`` -> exit 2 (block) or exit 0
(allow) — the subprocess-drives-script convention of
``tests/test_guard_harmful_bank_read.py``. Env hygiene: block/allow cases
run with ``EPM_ALLOW_PIPED_PUSH`` scrubbed; the escape-hatch case sets it
explicitly. NOTE (self-reference): the command strings below MENTION piped
pushes as test DATA — they are never executed; the guard only reads them
from stdin JSON.

Case ids B1-B11 / A1-A18 are the plan #1048 §6 acceptance tables
(``tasks/*/1048/plans/plan.md``); ``S7r1`` is the §7-row-1 pinned deliberate
false positive.

``TestSettingsWiring`` (the bank-read precedent) additionally parses
``.claude/settings.json``, asserts the matcher-Bash hook group carries the
configured command path, asserts the file exists + executable bit, and
invokes THE CONFIGURED COMMAND end-to-end on one block + one allow case —
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
SCRIPT = _REPO_ROOT / ".claude" / "hooks" / "guard_piped_git_push.sh"
SETTINGS = _REPO_ROOT / ".claude" / "settings.json"


def _env(*, allow: bool = False) -> dict[str, str]:
    """Hook env: EPM_ALLOW_PIPED_PUSH scrubbed (deny hygiene) unless ``allow``."""
    env = {k: v for k, v in os.environ.items() if k != "EPM_ALLOW_PIPED_PUSH"}
    if allow:
        env["EPM_ALLOW_PIPED_PUSH"] = "1"
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
# BLOCK — plan §6 B1-B11 (exit 2 + BLOCKED stderr)
# ---------------------------------------------------------------------------
BLOCK_CASES = [
    pytest.param("git push | tail -5", id="B1-plain-pipe"),
    pytest.param("git push origin main 2>&1 | grep -v x", id="B2-stderr-redir-grep"),
    pytest.param("gh pr merge 123 --squash | head", id="B3-gh-pr-merge"),
    pytest.param(
        "git -C .claude/worktrees/issue-1048 push origin issue-1048 2>&1 | tail -20",
        id="B4-flag-tolerant-anchor",
    ),
    pytest.param("cd /tmp && git push origin main | grep -c rejected", id="B5-own-and-segment"),
    pytest.param("out=$(git push 2>&1 | tail -1)", id="B6-command-substitution"),
    pytest.param("git merge issue-x 2>&1 | tail -5", id="B7-git-merge"),
    pytest.param("git push 2>&1 | tee push.log", id="B8-tee-without-pipefail"),
    pytest.param("git push |& tail -5", id="B9-pipe-amp-shorthand"),
    # B10: raw-newline multi-line command — the dominant Bash-tool delivery
    # shape (a literal newline in the command string, NOT a heredoc).
    pytest.param("echo pre\ngit push origin main | tail -5", id="B10-raw-newline"),
    # B11: non-`&` redirection — the §4.1 step-7 strip removes only
    # `&`-bearing operators; the stage regex must still match through
    # ` 2>err.log `.
    pytest.param("git push 2>err.log | tail", id="B11-non-amp-redirection"),
]


@pytest.mark.parametrize("cmd", BLOCK_CASES)
def test_block_cases(cmd: str) -> None:
    _assert_blocked(_run_bash(cmd))


def test_s7r1_pinned_false_positive_commit_message_blocks() -> None:
    """Plan §7-row-1 pinned EXPECTED-BLOCK (deliberate FP trade-off): a
    non-heredoc commit whose quoted ``-m`` text merely MENTIONS the banned
    pattern trips the raw scan — the guard does not strip quoted arguments
    (the guard_repo_root_branch.sh #796 trade-off). Remediation is
    ``git commit -F <file>`` / the heredoc commit recipe (blanket-allowed).
    """
    _assert_blocked(_run_bash('git commit -m "never git push | tail in a recipe" && git push'))


# ---------------------------------------------------------------------------
# ALLOW — plan §6 A1-A18 (exit 0), each targeting one named FP channel
# ---------------------------------------------------------------------------
ALLOW_CASES = [
    pytest.param("git push", id="A1-bare"),
    pytest.param("git push origin main && echo ok", id="A2-and-chain-no-pipe"),
    pytest.param("set -o pipefail; git push 2>&1 | tee log", id="A3-pipefail"),
    pytest.param("bash -o pipefail -c 'git push 2>&1 | tail -3'", id="A4-pipefail-flag-form"),
    pytest.param("git status | grep x && git push", id="A5-pipe-on-different-segment"),
    pytest.param("echo done | grep done", id="A6-no-producer"),
    pytest.param('git push origin main || echo "push failed"', id="A7-or-chain-issue931"),
    pytest.param("git push --dry-run 2>&1 | head -5", id="A8-dry-run"),
    pytest.param("git merge-base --all main HEAD | head -1", id="A9-merge-base"),
    pytest.param("git log --oneline | head -5 ; git push origin main", id="A10-semicolon-segment"),
    pytest.param("EPM_ALLOW_PIPED_PUSH=1 git push | tail -1", id="A11-inline-escape-hatch"),
    # A13: the canonical heredoc commit recipe — the heredoc blanket-allow
    # is load-bearing (a commit message describing this very incident would
    # otherwise false-block the implementing session's own commits).
    pytest.param(
        "git commit -m \"$(cat <<'EOF'\ntask: add guard\n\nnever git push | tail\nEOF\n)\""
        " && git push origin main",
        id="A13-heredoc-commit-recipe",
    ),
    pytest.param("echo foo | git push", id="A14-producer-as-consumer"),
    # A16: raw-newline multi-line where the pipe and the push live in
    # DIFFERENT newline-separated units.
    pytest.param("git status | grep x\ngit push origin main", id="A16-raw-newline-units"),
    # A17: the historical braced backslash-continued Step 10d recovery shape
    # (since reworked in /issue SKILL.md; kept as a `||` + line-continuation
    # normalization regression case).
    pytest.param(
        "git push origin main \\\n"
        "  || { git pull --rebase=merges --autostash && git push origin main; }",
        id="A17-step10d-recovery",
    ),
]


@pytest.mark.parametrize("cmd", ALLOW_CASES)
def test_allow_cases(cmd: str) -> None:
    _assert_allowed(_run_bash(cmd))


def test_a12_env_escape_hatch_allows_b1() -> None:
    """A12 — session env EPM_ALLOW_PIPED_PUSH=1 allows the B1 block shape."""
    _assert_allowed(_run_bash("git push | tail -5", env=_env(allow=True)))


def test_a15_empty_command_allowed() -> None:
    """A15 — empty command: fail-soft allow."""
    _assert_allowed(_run_bash(""))


def test_a15_malformed_stdin_json_allowed() -> None:
    """A15 — malformed stdin JSON: jq parse failure exits 0 (fail-soft)."""
    _assert_allowed(_run("this is not json"))


def test_a15_missing_command_field_allowed() -> None:
    """A15 — well-formed JSON with no tool_input.command: fail-soft allow."""
    _assert_allowed(_run({"tool_input": {}}))


def test_a18_heredoc_compound_documented_known_miss() -> None:
    """A18 — DOCUMENTED KNOWN-MISS (plan §4.1 step 4): a command that carries
    BOTH a heredoc AND a piped push (``<<EOF ... && git push 2>&1 | tail``)
    is ALLOWED by the heredoc blanket-allow. This pin makes the accepted
    residual visible and deliberate, never accidental — the lint + prose
    rule remain defense in depth. Reconciler binding rec (plan v3 delta): if
    a future fail-soft post-terminator scan is attempted under §12
    discretion, it must keep A13 (the canonical heredoc commit recipe) green
    and be DROPPED on any new false block.
    """
    cmd = "git commit -m \"$(cat <<'EOF'\nmsg\nEOF\n)\" && git push 2>&1 | tail -3"
    _assert_allowed(_run_bash(cmd))


def test_block_message_names_rule_and_remediations() -> None:
    """The §4.3 block message points at the bare-push rule, the pipefail
    escape, the --file/-F remediation, and the override incantation."""
    r = _run_bash("git push | tail -5")
    _assert_blocked(r)
    for needle in (
        "Concurrent",
        "pipefail",
        "git commit -F",
        "EPM_ALLOW_PIPED_PUSH=1",
        "#957",
    ):
        assert needle in r.stderr, (needle, r.stderr)


def test_self_test_mode_passes() -> None:
    """`--self-test` runs the in-script §6 acceptance table and exits 0."""
    r = subprocess.run(
        ["bash", str(SCRIPT), "--self-test"],
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "self-test: PASS" in r.stdout, r.stdout


# ---------------------------------------------------------------------------
# Settings wiring (the tests/test_guard_harmful_bank_read.py precedent)
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


class TestSettingsWiring:
    """Parse .claude/settings.json and invoke the CONFIGURED command.

    Every other test drives ``SCRIPT`` directly; without this class a
    matcher typo / wrong command path / missing +x ships the mechanical
    layer inert with a green suite. The command path is read FROM settings,
    never from the repo constant. Pre-merge worktree runs remap ONLY the
    canonical-root prefix onto this checkout (see module docstring);
    wrong-directory / wrong-basename / missing-+x bugs still fail under the
    remap.
    """

    def _configured_command(self) -> Path:
        settings = json.loads(SETTINGS.read_text())
        for entry in settings["hooks"]["PreToolUse"]:
            if entry.get("matcher") != "Bash":
                continue
            cmds = [h["command"] for h in entry.get("hooks", []) if h.get("type") == "command"]
            matches = [c for c in cmds if os.path.basename(c) == "guard_piped_git_push.sh"]
            assert len(matches) == 1, (
                f"expected exactly one guard_piped_git_push.sh command in the "
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

    def test_matcher_bash_group_carries_the_hook(self) -> None:
        self._configured_command()

    def test_configured_command_exists_and_is_executable(self) -> None:
        cmd = self._configured_command()
        assert cmd.exists(), cmd
        assert os.access(cmd, os.X_OK), cmd  # mechanizes deliverable 1's chmod +x

    def test_configured_command_blocks_piped_push(self) -> None:
        r = _run_bash("git push origin main 2>&1 | tail -20", script=self._configured_command())
        _assert_blocked(r)

    def test_configured_command_allows_bare_push(self) -> None:
        r = _run_bash("git push origin main", script=self._configured_command())
        _assert_allowed(r)
