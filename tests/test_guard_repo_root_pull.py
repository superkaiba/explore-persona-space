"""End-to-end tests for the ``scripts/guard_repo_root_pull.sh`` PreToolUse hook.

The guard mechanizes CLAUDE.md § Concurrent repo-root committers ("on a
rejected push run the single-flight root-sync helper —
``uv run python scripts/sync_repo_root.py`` — instead of hand-rolling a
pull-rebase recovery loop"; the prose alone failed open — #967's hand-rolled
root pull died ``fatal: Cannot autostash`` under a held lock/husk, and #711's
concurrent root pull-rebase orphaned a task-state commit): a Bash tool call
that would run a NON-ff ``git pull`` against the SHARED repo-root working
tree is BLOCKED (exit 2 + a ``BLOCKED`` stderr naming the sync_repo_root.py
remediation), while worktree/other-repo ``git -C`` pulls, provably-non-root
cd-latched pulls, ``--ff-only`` pulls anywhere, ssh/scp/grep-family clauses
(with the #1098 shared-root-spelling + producer-position exceptions), and
heredoc-bearing commands stay allowed (exit 0). Fail-soft on malformed
input; escape hatch ``EPM_ALLOW_ROOT_PULL=1`` (session env or inline
prefix).

These tests drive the script exactly as the harness does: stdin PreToolUse
JSON ``{"tool_input": {"command": ...}}`` -> exit 2 (block) or exit 0
(allow) — the subprocess-drives-script convention of
``tests/test_guard_piped_git_push.py``. Env hygiene: block/allow cases run
with ``EPM_ALLOW_ROOT_PULL`` scrubbed; the escape-hatch cases set it
explicitly. NOTE (self-reference): the command strings below MENTION root
pulls as test DATA — they are never executed; the guard only reads them from
stdin JSON.

Case ids B1-B11 / S1-S2 / A1-A20 are the plan #1201 §4.5 acceptance tables
(``tasks/*/1201/plans/plan.md``); B12-B16 / A21-A22 are the critic-round
additions (config-override pull, ssh shared-root spelling, waived-word
producer position, ``git -C .``, piped ff-only pod sync). S1 flipped +
B17-B23/S3-S4/A23-A28 are the #1250 command-position-anchoring round.

``TestSettingsWiring`` (the piped-push/bank-read precedent) additionally
parses ``.claude/settings.json``, asserts the matcher-Bash hook group
carries the configured command path, asserts the file exists + executable
bit, and invokes THE CONFIGURED COMMAND end-to-end on one block + one allow
case — closing the "hook ships green but inert" channel (matcher typo /
wrong command path / missing +x). When this suite runs from a pre-merge
worktree, the canonical-root PREFIX of the configured absolute path is
remapped onto this checkout (the rest of the path is exercised verbatim, so
a wrong directory / basename / missing +x still fails); on the main checkout
no remap occurs.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _REPO_ROOT / "scripts" / "guard_repo_root_pull.sh"
SETTINGS = _REPO_ROOT / ".claude" / "settings.json"
_CANONICAL_ROOT = "/home/thomasjiralerspong/explore-persona-space"


def _env(*, allow: bool = False) -> dict[str, str]:
    """Hook env: EPM_ALLOW_ROOT_PULL scrubbed (deny hygiene) unless ``allow``."""
    env = {k: v for k, v in os.environ.items() if k != "EPM_ALLOW_ROOT_PULL"}
    if allow:
        env["EPM_ALLOW_ROOT_PULL"] = "1"
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
# BLOCK — plan §4.5 B1-B11 + critic-round B12-B16 (exit 2 + BLOCKED stderr)
# ---------------------------------------------------------------------------
BLOCK_CASES = [
    pytest.param("git pull", id="B1-bare-pull-967-shape"),
    pytest.param("git pull origin main", id="B2-refspec"),
    pytest.param(
        "git pull --rebase=merges --autostash origin main",
        id="B3-hand-rolled-recovery-form",
    ),
    pytest.param(
        "git push origin main || { git pull --rebase=merges --autostash && git push origin main; }",
        id="B4-braced-recovery-loop",
    ),
    pytest.param("git pull --no-rebase", id="B5-no-rebase-merge-pull-gap-xvi"),
    pytest.param("git pull --rebase origin main", id="B6-plain-rebase-711-hazard"),
    pytest.param("cd scripts && git pull", id="B7-relative-cd-never-latches"),
    pytest.param(
        f"cd {_CANONICAL_ROOT} && git pull --rebase=merges --autostash",
        id="B8-explicit-root-cd",
    ),
    # B9: REAL embedded newline (json.dumps encodes it; the guard's jq -r
    # decodes it back) — the dominant Bash-tool multi-line delivery shape.
    # Keep the "\n" a real newline; collapsing to one line degrades the case.
    pytest.param("echo pre\ngit pull", id="B9-raw-newline"),
    pytest.param("git fetch origin && git pull", id="B10-own-and-segment"),
    pytest.param("cd ~/overleaf-6a2df2d2; git pull", id="B11-semicolon-resets-latch"),
    # B12 (critic a): the config-override merge-pull — gap-(xvi)'s second
    # shape; the flag-tolerant anchor's optional value token covers `-c <kv>`.
    pytest.param("git -c pull.rebase=false pull origin main", id="B12-config-override-pull"),
    # B13 (critic c): an ssh clause naming the shared-repo path in a covered
    # spelling is NOT waived (branch-guard #1098 cond-(4) parity).
    pytest.param(
        "ssh cia-benchmark-vm 'git --work-tree=$HOME/explore-persona-space pull origin main'",
        id="B13-ssh-shared-root-spelling-not-waived",
    ),
    # B14/B15 (critic c): a waived word in producer position of a pipe /
    # local-file redirect feeding a shell loses the waiver (#1098 round-2
    # write-then-execute parity).
    pytest.param("ssh pod-779 'echo git pull origin main' | bash", id="B14-pipe-producer"),
    pytest.param("ssh pod-779 'echo git pull origin main' > /tmp/x", id="B15-redirect-producer"),
    pytest.param(
        "ssh -o ProxyCommand='git pull origin main %h' pod-779 'true'",
        id="B16-proxycommand-executes-locally",
    ),
    # B17-B21 (#1250): the command-position anchor's closed prefix-unit set —
    # wrapper words, env assignments, duration tokens, bare -flag units keep
    # real pulls behind them fail-closed.
    pytest.param("nohup git pull", id="B17-nohup-wrapper"),
    pytest.param("GIT_TRACE=1 git pull --rebase origin main", id="B18-env-assignment-prefix"),
    pytest.param("timeout 300 git pull", id="B19-timeout-duration-wrapper"),
    pytest.param("command git pull", id="B20-command-builtin-wrapper"),
    pytest.param("sudo -n git pull", id="B21-sudo-flag-wrapper"),
    # B22 (#1250): the REQUIRED shell-keyword prefix units — a hand-rolled
    # retry loop's lead is `until git pull ...`, precisely the recovery class
    # the guard exists for; a keyword-less anchor would fail OPEN here.
    pytest.param("until git pull --rebase; do sleep 5; done", id="B22-until-loop-keyword-lead"),
    # B23 (#1250): a grep-family clause whose waiver is REFUSED (local file
    # redirect) routes to the arm-2 whole-clause scan — pins that the
    # scanwhole flag sits at the SHARED waived-word arm top, not inside the
    # ssh sub-branch (a mis-scoping would silently fail open on grep leads).
    pytest.param(
        'grep -rn "git pull --rebase" .claude/ scripts/ > /tmp/hits.txt',
        id="B23-grep-refused-waiver-arm2",
    ),
]


@pytest.mark.parametrize("cmd", BLOCK_CASES)
def test_block_cases(cmd: str) -> None:
    _assert_blocked(_run_bash(cmd))


def test_s1_flipped_quoted_mention_allowed() -> None:
    """S1 FLIPPED by #1250 (command-position anchoring): a non-heredoc commit
    whose quoted ``-m`` text merely MENTIONS a root pull is now ALLOWED — the
    classifier anchors to the clause's command position (lead ``git`` with
    verb ``commit``, then ``push``; never ``pull``), so quoted argument text
    can no longer trip it. This is NOT quote-stripping (the #796 revert
    stands — nothing is stripped, the detector is anchored), so real quoted
    pulls in waiver-refused ssh/grep clauses still block (B13-B16, B23, S4).
    """
    _assert_allowed(
        _run_bash('git commit -m "never hand-roll git pull at the root" && git push origin main')
    )


def test_s2_pinned_expected_block_variable_cd_target() -> None:
    """Plan §4.5 S2 pinned EXPECTED-BLOCK: a variable cd target is
    unprovable (no $WT-latch machinery in v1 — a deliberate simplification
    vs the branch guard), so ``cd "$WT" && git pull`` fails closed.
    Remediation is the canonical ``git -C "$WT" pull ...`` form (A1).
    """
    _assert_blocked(_run_bash('cd "$WT" && git pull'))


def test_s3_pinned_residual_missplit_note_mention_blocks() -> None:
    """S3 pinned EXPECTED-BLOCK residual (#1250): quoted ``--note`` text that
    embeds a shell separator + a FULL pull command literal mis-splits raw
    (the standing #796 no-quote-parse trade-off), so the tail clause's LEAD
    becomes the pull literal itself and the command-position anchor
    legitimately matches. Remediation for note/commit text of this shape:
    ``task.py post-marker --file <path.md>`` / ``git commit -F <file>`` / a
    heredoc. Pins the RESIDUAL boundary of the #1250 fix as deliberate.
    """
    _assert_blocked(
        _run_bash(
            "uv run python scripts/task.py post-marker 1250 epm:progress"
            " --note 'recovered: git fetch && git pull --rebase worked'"
        )
    )


def test_s4_pinned_residual_piped_grep_mention_blocks() -> None:
    """S4 pinned EXPECTED-BLOCK residual (#1250): a piped grep whose pattern
    argument quotes a root pull sits in pipeline-PRODUCER position, so the
    waiver is refused (guard cond-(1)) and the clause keeps the arm-2
    whole-clause scan, which matches the quoted pattern. Makes the retained
    arm-2 FP class legible (the same mention passes un-piped, A12).
    Remediation: run the grep un-piped or via the Grep tool.
    """
    _assert_blocked(_run_bash('grep -rn "git pull --rebase" .claude/ | head -5'))


def test_incident_1250_post_marker_note_mention_allowed() -> None:
    """The #1250 incident shape (DURABILITY PIN): a separator-free
    ``task.py post-marker --note`` whose quoted text mentions a root pull is
    ALLOWED — the lead word is ``uv``, not a git/wrapper/keyword prefix unit,
    so the command-position anchor cannot match. This is the exact false
    positive that fired in production hours after #1201 shipped
    (2026-07-09T23:06Z, on #1201's own session).
    """
    _assert_allowed(
        _run_bash(
            "uv run python scripts/task.py post-marker 1201 epm:progress"
            " --note 'worked around via git pull --rebase mention'"
        )
    )


# ---------------------------------------------------------------------------
# ALLOW — plan §4.5 A1-A20 + critic-round A21-A22 (exit 0), each a named
# false-positive channel held open
# ---------------------------------------------------------------------------
ALLOW_CASES = [
    pytest.param(
        'git -C "$WT" pull --rebase=merges --autostash && git -C "$WT" push origin issue-1201',
        id="A1-step10d-form-1",
    ),
    pytest.param(
        "git -C .claude/worktrees/issue-1201 pull --rebase=merges --autostash origin main",
        id="A2-literal-worktree-C",
    ),
    pytest.param("cd ~/overleaf-6a2df2d2 && git pull", id="A3-overleaf-clone-cd-latch"),
    pytest.param("cd /tmp/scratch-clone && git pull", id="A4-absolute-non-repo-latch"),
    pytest.param("cd .claude/worktrees/issue-1201 && git pull", id="A5-relative-worktree-latch"),
    pytest.param("git pull --ff-only origin main", id="A6-ff-only-waiver"),
    pytest.param("ssh pod-779 'git pull --ff-only origin main'", id="A7-single-statement-pod-sync"),
    pytest.param(
        "ssh epm-issue-228 'cd /workspace/explore-persona-space &&"
        " git pull --ff-only origin main && uv sync --locked'",
        id="A8-multi-statement-pod-sync-ff-only-tail",
    ),
    pytest.param("git fetch origin main", id="A9-fetch-is-not-pull"),
    pytest.param("git pull-request --help", id="A10-verb-terminator-lookalike"),
    pytest.param(
        "git push origin main || uv run python scripts/sync_repo_root.py",
        id="A11-step10d-form-2-remediation",
    ),
    pytest.param('grep -rn "git pull --rebase" .claude/ scripts/', id="A12-grep-family-waiver"),
    pytest.param(
        "cat > /tmp/note.md <<'EOF'\nnever run git pull at the shared root\nEOF",
        id="A13-heredoc-doc-text",
    ),
    pytest.param("EPM_ALLOW_ROOT_PULL=1 git pull", id="A14-inline-escape-hatch"),
    pytest.param("git log --oneline | head -5", id="A18-no-pull-pipe-hygiene"),
    pytest.param(
        f"git -C {_CANONICAL_ROOT} pull",
        id="A20-path-blind-C-waiver-pinned-limitation",
    ),
    # A21 (critic b): `git -C . pull` issued from the root — under the
    # path-blind -C waiver this ALLOWs (sibling-parity known limitation;
    # the block message's "NEVER point -C at the repo root — `git -C .`
    # from the root is the same op" line is the stated control).
    pytest.param("git -C . pull", id="A21-git-C-dot-pinned-limitation"),
    pytest.param(
        "ssh pod-779 'git pull --ff-only origin main' 2>&1 | tail -20",
        id="A22-piped-pod-sync-saved-by-ff-only",
    ),
    # A24-A28 (#1250): one representative expected-allow pin per documented
    # fail-open residual class of the command-position anchor (guard
    # known-limitations bullets (b)/(b')). A24 is the refspec form so it
    # DISCRIMINATES old->new (the no-arg form already allowed pre-#1250).
    pytest.param("bash -c 'git pull --rebase origin main'", id="A24-bash-dash-c-known-miss"),
    pytest.param('echo "git pull is fenced at the root"', id="A25-echo-mention"),
    pytest.param(
        'echo "$(git pull --rebase origin main)"',
        id="A26-command-substitution-residual",
    ),
    pytest.param("sudo -u deploy git pull origin main", id="A27-flag-value-wrapper-residual"),
    pytest.param('eval "git pull origin main"', id="A28-quoted-eval-residual"),
]


@pytest.mark.parametrize("cmd", ALLOW_CASES)
def test_allow_cases(cmd: str) -> None:
    _assert_allowed(_run_bash(cmd))


def test_a15_env_escape_hatch_allows_b1() -> None:
    """A15 — session env EPM_ALLOW_ROOT_PULL=1 allows a piped B1 shape."""
    _assert_allowed(_run_bash("git pull | tail -3", env=_env(allow=True)))


def test_a16_empty_command_allowed() -> None:
    """A16 — empty command: fail-soft allow."""
    _assert_allowed(_run_bash(""))


def test_a16_malformed_stdin_json_allowed() -> None:
    """A16 — malformed stdin JSON: jq parse failure exits 0 (fail-soft)."""
    _assert_allowed(_run("this is not json"))


def test_a16_missing_command_field_allowed() -> None:
    """A16 — well-formed JSON with no tool_input.command: fail-soft allow."""
    _assert_allowed(_run({"tool_input": {}}))


def test_a17_other_repo_c_form_allowed() -> None:
    """A17 — the Overleaf theory clone via -C (CLAUDE.md § Theory source)."""
    _assert_allowed(_run_bash("git -C ~/overleaf-6a2df2d2 pull"))


def test_a19_heredoc_documented_known_miss() -> None:
    """A19 — DOCUMENTED KNOWN-MISS (sibling A18 precedent): a command that
    carries BOTH a heredoc AND a real same-call root pull is ALLOWED by the
    heredoc blanket-allow. This pin makes the accepted residual visible and
    deliberate, never accidental — the prose rule + a block on the next bare
    attempt remain defense in depth.
    """
    cmd = "git commit -F /tmp/msg.txt <<'EOF'\nmsg\nEOF\ngit pull"
    _assert_allowed(_run_bash(cmd))


def test_block_message_names_rule_and_remediations() -> None:
    """The §4.2 block message points at the sync helper, the incident, the
    worktree/-C and ff-only channels, the -F mention remediation, and the
    override incantation.
    """
    r = _run_bash("git pull")
    _assert_blocked(r)
    for needle in (
        "sync_repo_root.py",
        "#967",
        "git -C",
        "--ff-only",
        "EPM_ALLOW_ROOT_PULL=1",
        "-F",
    ):
        assert needle in r.stderr, (needle, r.stderr)


def test_self_test_mode_passes() -> None:
    """`--self-test` runs the in-script §4.5 acceptance table and exits 0."""
    r = subprocess.run(
        ["bash", str(SCRIPT), "--self-test"],
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "self-test: PASS" in r.stdout, r.stdout


# ---------------------------------------------------------------------------
# Settings wiring (the tests/test_guard_piped_git_push.py precedent)
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
            matches = [c for c in cmds if os.path.basename(c) == "guard_repo_root_pull.sh"]
            assert len(matches) == 1, (
                f"expected exactly one guard_repo_root_pull.sh command in the "
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

    def test_configured_command_blocks_root_pull(self) -> None:
        r = _run_bash("git pull", script=self._configured_command())
        _assert_blocked(r)

    def test_configured_command_allows_worktree_pull(self) -> None:
        r = _run_bash(
            'git -C "$WT" pull --rebase=merges --autostash',
            script=self._configured_command(),
        )
        _assert_allowed(r)
