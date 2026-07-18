"""End-to-end tests for the ``.claude/hooks/guard_root_code_commit.sh`` hook (#1500).

The guard mechanizes the SKILL.md Step 9a-ter § Inline payload lint gate for
CODE payload: a Bash tool call that would run a repo-root ``git commit`` whose
pending payload includes an UNCERTIFIED ``scripts/**.py`` / ``src/**`` /
``tests/**.py`` file is BLOCKED (exit 2 + a ``BLOCKED`` stderr naming the
``scripts/inline_lint_gate.py`` gate command, the worktree remediation, and
the override), while artifact-only commits, worktree commits, non-git
commands, and certified payloads stay allowed (exit 0). Certification lines
(``v1 <epoch> <blobsha> <path>``) are written by ``scripts/inline_lint_gate.py``
on a passing gate run and bind the LANDING content: the worktree hash for
``-a`` / commit-pathspec / chained-``git add`` shapes (those commit worktree
content), the staged blob sha only for plain staged commits.

These tests drive the script exactly as the harness does: stdin PreToolUse
JSON ``{"tool_input": {"command": ...}}`` -> exit 2 (block) or exit 0 (allow)
— the subprocess-drives-script convention of ``tests/test_guard_repo_root_pull.py``.
Hermetic: ``EPM_ROOT_CODE_COMMIT_REPO`` points Layer 2 at a tmp git repo and
``EPM_INLINE_CERT_PATH`` at a tmp cert file, so no test reads the live repo's
index or the live cert. Env hygiene: block/allow cases run with
``EPM_ALLOW_ROOT_CODE_COMMIT`` scrubbed; the escape-hatch cases set it.
NOTE (self-reference): the command strings below MENTION root commits as test
DATA — they are never executed; the guard only reads them from stdin JSON.

Case ids A1-A12 / B1-B14 / W1 are the plan #1500 §6.1 acceptance matrix
(``tasks/*/1500/plans/plan.md``).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _REPO_ROOT / ".claude" / "hooks" / "guard_root_code_commit.sh"
SETTINGS = _REPO_ROOT / ".claude" / "settings.json"
_CANONICAL_ROOT = "/home/thomasjiralerspong/explore-persona-space"
GATED = "scripts/issue9_fig.py"


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return r.stdout


def _init_repo(tmp_path: Path, name: str) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git_bare_init(repo)
    return repo


def _git_bare_init(repo: Path) -> None:
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)


def _write(repo: Path, rel: str, content: str) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _stage(repo: Path, rel: str, content: str) -> None:
    _write(repo, rel, content)
    _git(repo, "add", "--", rel)


def _staged_sha(repo: Path, rel: str) -> str:
    return _git(repo, "ls-files", "-s", "--", rel).split()[1]


def _worktree_sha(repo: Path, rel: str) -> str:
    return _git(repo, "hash-object", "--", str(repo / rel)).strip()


@pytest.fixture
def art_repo(tmp_path: Path) -> Path:
    """Repo with artifact-only staged payload (tasks/ + figures/)."""
    repo = _init_repo(tmp_path, "art")
    _stage(repo, "tasks/running/9/events.jsonl", "{}\n")
    _stage(repo, "figures/issue_9/f.png", "png\n")
    return repo


@pytest.fixture
def code_repo(tmp_path: Path) -> Path:
    """Repo with a gated scripts/ file STAGED + a second gated file untracked."""
    repo = _init_repo(tmp_path, "code")
    _stage(repo, GATED, "print(1)\n")
    _write(repo, "scripts/issue9_new.py", "print(2)\n")  # untracked (compound-add case)
    return repo


@pytest.fixture
def cert(tmp_path: Path) -> Path:
    return tmp_path / "cert.txt"


def _env(
    repo: Path, cert_path: Path, *, allow: bool = False, max_age: str | None = None
) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "EPM_ALLOW_ROOT_CODE_COMMIT"}
    env["EPM_ROOT_CODE_COMMIT_REPO"] = str(repo)
    env["EPM_INLINE_CERT_PATH"] = str(cert_path)
    if max_age is not None:
        env["EPM_INLINE_CERT_MAX_AGE_S"] = max_age
    if allow:
        env["EPM_ALLOW_ROOT_CODE_COMMIT"] = "1"
    return env


def _run(
    cmd: str | None,
    repo: Path,
    cert_path: Path,
    *,
    raw: str | None = None,
    script: Path | None = None,
    **env_kw,
) -> subprocess.CompletedProcess[str]:
    payload = raw if raw is not None else json.dumps({"tool_input": {"command": cmd}})
    return subprocess.run(
        [str(script or SCRIPT)],
        input=payload,
        text=True,
        capture_output=True,
        env=_env(repo, cert_path, **env_kw),
    )


def _assert_blocked(r: subprocess.CompletedProcess[str]) -> None:
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "BLOCKED" in r.stderr, r.stderr


def _assert_allowed(r: subprocess.CompletedProcess[str]) -> None:
    assert r.returncode == 0, (r.returncode, r.stderr)


def _cert_line(cert_path: Path, rel: str, sha: str, *, epoch: int | None = None) -> None:
    with open(cert_path, "a", encoding="utf-8") as fh:
        fh.write(f"v1 {epoch if epoch is not None else int(time.time())} {sha} {rel}\n")


# ---------------------------------------------------------------------------
# A — must ALLOW (exit 0)
# ---------------------------------------------------------------------------
def test_a1_artifact_only_commit_allowed(art_repo: Path, cert: Path) -> None:
    _assert_allowed(_run("git commit -m x", art_repo, cert))


def test_a2_non_git_command_allowed(code_repo: Path, cert: Path) -> None:
    _assert_allowed(
        _run(
            "uv run python scripts/task.py post-marker 9 epm:progress --note 'commit soon'",
            code_repo,
            cert,
        )
    )


@pytest.mark.parametrize("cmd", ["git push origin main", "git status"])
def test_a3_git_non_commit_allowed(cmd: str, code_repo: Path, cert: Path) -> None:
    _assert_allowed(_run(cmd, code_repo, cert))


def test_a4_worktree_dash_c_commit_allowed(code_repo: Path, cert: Path) -> None:
    """Gated files staged at root, but the commit targets a worktree via -C."""
    _assert_allowed(_run('git -C "$WT" commit -m x', code_repo, cert))


def test_a5_cd_latched_worktree_commit_allowed(code_repo: Path, cert: Path) -> None:
    _assert_allowed(_run("cd .claude/worktrees/issue-9 && git commit -m x", code_repo, cert))


def test_a6_fresh_matching_cert_allows(code_repo: Path, cert: Path) -> None:
    _cert_line(cert, GATED, _staged_sha(code_repo, GATED))
    _assert_allowed(_run("git commit -m x", code_repo, cert))


def test_a7_env_escape_hatch(code_repo: Path, cert: Path) -> None:
    _assert_allowed(_run("git commit -m x", code_repo, cert, allow=True))


def test_a7_inline_escape_hatch(code_repo: Path, cert: Path) -> None:
    _assert_allowed(_run("EPM_ALLOW_ROOT_CODE_COMMIT=1 git commit -m x", code_repo, cert))


def test_a8_failsoft_empty_command(code_repo: Path, cert: Path) -> None:
    _assert_allowed(_run("", code_repo, cert))


def test_a8_failsoft_malformed_json(code_repo: Path, cert: Path) -> None:
    _assert_allowed(_run(None, code_repo, cert, raw="this is not json"))


def test_a8_failsoft_missing_command_field(code_repo: Path, cert: Path) -> None:
    _assert_allowed(_run(None, code_repo, cert, raw=json.dumps({"tool_input": {}})))


def test_a9_heredoc_message_mentioning_commit_allowed(art_repo: Path, cert: Path) -> None:
    """Layer-2 neutralization: a message line beginning `git commit` matches
    Layer 1, but the artifact-only staged set carries no gated payload."""
    cmd = "git commit -m \"$(cat <<'EOF'\nfix: never bare git commit -m at the root\nEOF\n)\""
    _assert_allowed(_run(cmd, art_repo, cert))


def test_a10_staged_deletion_of_gated_path_exempt(tmp_path: Path, cert: Path) -> None:
    repo = _init_repo(tmp_path, "del")
    _stage(repo, GATED, "print(1)\n")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    _git(repo, "rm", "-q", "--", GATED)
    _assert_allowed(_run("git commit -m x", repo, cert))


def test_a11_non_gated_code_adjacent_allowed(tmp_path: Path, cert: Path) -> None:
    repo = _init_repo(tmp_path, "adj")
    _stage(repo, ".claude/rules/foo.md", "rule\n")
    _stage(repo, "configs/x.yaml", "a: 1\n")
    _assert_allowed(_run("git commit -m x", repo, cert))


def test_a12_compound_add_commit_with_fresh_worktree_cert(tmp_path: Path, cert: Path) -> None:
    """Isolated repo: the ONLY gated pending path is the chained-add file
    (the code_repo fixture's staged-uncertified sibling would rightly block)."""
    repo = _init_repo(tmp_path, "addonly")
    _write(repo, "scripts/issue9_new.py", "print(2)\n")  # untracked
    _cert_line(cert, "scripts/issue9_new.py", _worktree_sha(repo, "scripts/issue9_new.py"))
    _assert_allowed(_run("git add scripts/issue9_new.py && git commit -m x", repo, cert))


# ---------------------------------------------------------------------------
# B — must BLOCK (exit 2)
# ---------------------------------------------------------------------------
def test_b1_gated_staged_no_cert_blocks(code_repo: Path, cert: Path) -> None:
    _assert_blocked(_run("git commit -m x", code_repo, cert))


def test_b2_stale_cert_blocks(code_repo: Path, cert: Path) -> None:
    _cert_line(cert, GATED, _staged_sha(code_repo, GATED), epoch=int(time.time()) - 10)
    _assert_blocked(_run("git commit -m x", code_repo, cert, max_age="1"))


def test_b3_wrong_blobsha_cert_blocks(code_repo: Path, cert: Path) -> None:
    _cert_line(cert, GATED, "0" * 40)
    _assert_blocked(_run("git commit -m x", code_repo, cert))


def test_b4_cert_for_different_path_only_blocks(code_repo: Path, cert: Path) -> None:
    _cert_line(cert, "scripts/other.py", _staged_sha(code_repo, GATED))
    _assert_blocked(_run("git commit -m x", code_repo, cert))


def test_b5_dash_a_with_modified_tracked_gated_file_blocks(tmp_path: Path, cert: Path) -> None:
    repo = _init_repo(tmp_path, "mod")
    _stage(repo, GATED, "print(1)\n")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    _write(repo, GATED, "print(2)\n")  # modified, UNSTAGED
    _assert_blocked(_run("git commit -am x", repo, cert))


def test_b6_pathspec_form_no_cert_blocks(code_repo: Path, cert: Path) -> None:
    _assert_blocked(_run(f"git commit -m x {GATED}", code_repo, cert))


def test_b7_dash_c_spelling_repo_root_not_waived(code_repo: Path, cert: Path) -> None:
    _assert_blocked(_run(f"git -C {_CANONICAL_ROOT} commit -m x", code_repo, cert))


def test_b8_classification_failure_fails_closed(tmp_path: Path, cert: Path) -> None:
    notarepo = tmp_path / "notarepo"
    notarepo.mkdir()
    _assert_blocked(_run("git commit -m x", notarepo, cert))


def test_b9_mixed_staged_set_blocks_and_names_only_gated(tmp_path: Path, cert: Path) -> None:
    repo = _init_repo(tmp_path, "mixed")
    _stage(repo, "tasks/running/9/events.jsonl", "{}\n")
    _stage(repo, GATED, "print(1)\n")
    r = _run("git commit -m x", repo, cert)
    _assert_blocked(r)
    assert GATED in r.stderr, r.stderr
    assert "tasks/running/9/events.jsonl" not in r.stderr, r.stderr


def _staged_then_edited_repo(tmp_path: Path, cert: Path) -> Path:
    """Cert matches the STALE staged blob; the worktree copy was edited after."""
    repo = _init_repo(tmp_path, "toctou")
    _stage(repo, GATED, "print('v1')\n")  # index = v1
    _cert_line(cert, GATED, _staged_sha(repo, GATED))  # cert binds v1
    _write(repo, GATED, "print('v2')\n")  # worktree = v2 (what -a/pathspec lands)
    return repo


def test_b10_landing_content_binding_dash_a(tmp_path: Path, cert: Path) -> None:
    repo = _staged_then_edited_repo(tmp_path, cert)
    _assert_blocked(_run("git commit -am x", repo, cert))


def test_b11_landing_content_binding_pathspec(tmp_path: Path, cert: Path) -> None:
    repo = _staged_then_edited_repo(tmp_path, cert)
    _assert_blocked(_run(f"git commit -m x {GATED}", repo, cert))


def test_b12_compound_add_commit_untracked_no_cert_blocks(code_repo: Path, cert: Path) -> None:
    """The dominant incident idiom: nothing is staged at PreToolUse time —
    only the add-clause text names the payload."""
    _assert_blocked(_run("git add scripts/issue9_new.py && git commit -m x", code_repo, cert))


@pytest.mark.parametrize(
    "stage_form", ["git add -A", "git add .", "git add --all"], ids=["dash-A", "dot", "all"]
)
def test_b13_blanket_add_chained_fails_closed(stage_form: str, art_repo: Path, cert: Path) -> None:
    _assert_blocked(_run(f"{stage_form} && git commit -m x", art_repo, cert))


def test_b14_gated_path_with_space_blocks(tmp_path: Path, cert: Path) -> None:
    repo = _init_repo(tmp_path, "space")
    _stage(repo, "scripts/issue9 fig.py", "print(1)\n")
    r = _run("git commit -m x", repo, cert)
    _assert_blocked(r)
    assert "scripts/issue9 fig.py" in r.stderr, r.stderr


def test_malformed_cert_line_never_crashes_or_allows(code_repo: Path, cert: Path) -> None:
    """Round-1 concern 1: a non-numeric epoch must not crash the arithmetic
    and must not match (block direction)."""
    cert.write_text(f"v1 not-a-number {_staged_sha(code_repo, GATED)} {GATED}\n", encoding="utf-8")
    _assert_blocked(_run("git commit -m x", code_repo, cert))


def test_block_message_names_gate_remediation_and_override(code_repo: Path, cert: Path) -> None:
    r = _run("git commit -m x", code_repo, cert)
    _assert_blocked(r)
    for needle in (
        "inline_lint_gate.py",
        'git -C "$WT" commit',
        "EPM_ALLOW_ROOT_CODE_COMMIT=1",
        "NEVER hand-write",
    ):
        assert needle in r.stderr, (needle, r.stderr)


def test_self_test_mode_passes() -> None:
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in ("EPM_ALLOW_ROOT_CODE_COMMIT", "EPM_ROOT_CODE_COMMIT_REPO", "EPM_INLINE_CERT_PATH")
    }
    r = subprocess.run(
        ["bash", str(SCRIPT), "--self-test"], capture_output=True, text=True, env=env
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "self-test: PASS" in r.stdout, r.stdout


# ---------------------------------------------------------------------------
# W1 — settings wiring (the tests/test_guard_repo_root_pull.py precedent)
# ---------------------------------------------------------------------------
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

    Without this class a matcher typo / wrong command path / missing +x ships
    the hook inert with a green suite. Pre-merge worktree runs remap ONLY the
    canonical-root prefix onto this checkout (pull-guard precedent); wrong
    directory / basename / missing +x still fail under the remap.
    """

    def _configured_command(self) -> Path:
        settings = json.loads(SETTINGS.read_text())
        for entry in settings["hooks"]["PreToolUse"]:
            if entry.get("matcher") != "Bash":
                continue
            cmds = [h["command"] for h in entry.get("hooks", []) if h.get("type") == "command"]
            matches = [c for c in cmds if os.path.basename(c) == "guard_root_code_commit.sh"]
            assert len(matches) == 1, (
                f"expected exactly one guard_root_code_commit.sh command in the "
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
        assert os.access(cmd, os.X_OK), cmd

    def test_configured_command_blocks_uncertified_code_commit(
        self, code_repo: Path, cert: Path
    ) -> None:
        r = _run("git commit -m x", code_repo, cert, script=self._configured_command())
        _assert_blocked(r)

    def test_configured_command_allows_artifact_commit(self, art_repo: Path, cert: Path) -> None:
        r = _run("git commit -m x", art_repo, cert, script=self._configured_command())
        _assert_allowed(r)
