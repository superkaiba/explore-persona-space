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
(``tasks/*/1500/plans/plan.md``). A13 / B15 / B15b were added in round 2
(code-review Major: the per-clause comment-tail strip ran BEFORE the
token/flag scan, so a ``#`` inside the commit message — the repo-standard
``-m "task #N: ..."`` — silently discarded same-clause pathspecs and a
post-message ``-a``; concern ``hash-in-message-defeats-clause-token-scan``).
B16*/B17/A14 + the per-span-shape battery were added in round 3 (the CLASS
fix: string-literal spans are masked on a scan copy before the clause split
and token/flag/pathspec scan; concern
``quoted-message-seams-defeat-clause-scan``).
The hd-group + the ``test_b22``/``test_rd15`` re-keys were added for #2046:
strictly-recognized here-doc / here-string OPENER tokens on the commit
clause are intercepted (``heredoc_tok_kind``) before candidate
classification, and the two cd classification sites compare against
``$GUARD_REPO`` (production bit-identical; hermetic repos gain cd-to-root
coverage).
The wt-group was added for #2066: a commit with no retarget evidence and no
conservative-screen match is ALLOWED when the hook-input cwd provably sits
inside a linked worktree of the guarded repo (mirrors of the self-test rows
W1/W3/W4/W7/W8/W9/W10), and the final block heredoc LEADS with the
remediation block (worktree rewrite + inline-lint-gate recipe) before the
diagnostics (leg-(b) order pin). The round-2 code-review fix (blocker
``wt-allow-screen-spelling-gaps``) WIDENED the screen to the fail-closed
direction — junk-tolerant cd-WORD matching, pushd/popd/source, long AND
short directory/repo-pointing flags, GIT_COMMON_DIR= — with mirrors of the
four demonstrated evasion variants (self-test rows W13-W16).
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
    repo: Path,
    cert_path: Path,
    *,
    allow: bool = False,
    max_age: str | None = None,
    rehash_delay: str | None = "0",
    path_prepend: Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "EPM_ALLOW_ROOT_CODE_COMMIT",
            "EPM_CERT_REHASH_DELAY_S",
            "EPM_ROOT_CODE_COMMIT_DISABLE_WT_CWD_ALLOW",
        )
    }
    env["EPM_ROOT_CODE_COMMIT_REPO"] = str(repo)
    env["EPM_INLINE_CERT_PATH"] = str(cert_path)
    # Zero the #1857 settle-and-re-hash delay by default so blocked cases stay
    # deterministic-fast (the retry still RUNS — it just doesn't wait); the
    # retry tests shim `sleep` via path_prepend for the settle action itself.
    if rehash_delay is not None:
        env["EPM_CERT_REHASH_DELAY_S"] = rehash_delay
    if path_prepend is not None:
        env["PATH"] = f"{path_prepend}:{env.get('PATH', '/usr/bin:/bin')}"
    if max_age is not None:
        env["EPM_INLINE_CERT_MAX_AGE_S"] = max_age
    if allow:
        env["EPM_ALLOW_ROOT_CODE_COMMIT"] = "1"
    if extra:
        env.update(extra)
    return env


# Sentinel for _run(cwd=...): omit the top-level `cwd` field from the hook
# stdin JSON entirely (the missing-cwd fallback cell, c13).
_OMIT_CWD = object()


def _run(
    cmd: str | None,
    repo: Path,
    cert_path: Path,
    *,
    raw: str | None = None,
    script: Path | None = None,
    cwd: object = None,
    **env_kw,
) -> subprocess.CompletedProcess[str]:
    """Invoke the hook. The synthesized stdin JSON carries a top-level `cwd`
    DEFAULTING to the fixture repo root (so pathspec scoping can engage);
    pass cwd=<path> to override or cwd=_OMIT_CWD to drop the field."""
    if raw is not None:
        payload = raw
    else:
        obj: dict[str, object] = {"tool_input": {"command": cmd}}
        if cwd is not _OMIT_CWD:
            obj["cwd"] = str(repo if cwd is None else cwd)
        payload = json.dumps(obj)
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


def test_a13_artifact_only_commit_with_hash_bearing_message_allowed(
    art_repo: Path, cert: Path
) -> None:
    """Round-2 Major companion: the raw-clause token scan must not false-block
    an artifact-only commit on message tokens (``#9:`` etc.)."""
    _assert_allowed(_run('git commit -m "task #9: docs"', art_repo, cert))


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


# ---------------------------------------------------------------------------
# R — #1857 cert-retry pass (settle-and-re-hash before the negative verdict)
# ---------------------------------------------------------------------------
def _sleep_shim(tmp_path: Path, body: str) -> Path:
    """PATH-shimmed `sleep` running `body` — the deterministic stand-in for
    the settle window (the "concurrent writer" acts during the delay)."""
    shim_dir = tmp_path / "shim-bin"
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "sleep"
    shim.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    shim.chmod(0o755)
    return shim_dir


def _retry_repo(tmp_path: Path) -> Path:
    """Tracked, committed gated file (content A) — the worktree-binding
    pathspec-commit shape the retry tests flip mid-guard."""
    repo = _init_repo(tmp_path, "retry")
    _stage(repo, GATED, "print(1)\n")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    return repo


def test_r1_transient_worktree_flip_recovers_on_rehash(tmp_path: Path, cert: Path) -> None:
    """Cert binds content A; the worktree reads content B at the first pass;
    the shimmed `sleep` restores content A during the settle window -> the
    re-hash matches (same landing_sha derivation, no re-binding), the commit
    is ALLOWED, and the grep-able `cert-retry:` recovery line is emitted."""
    repo = _retry_repo(tmp_path)
    _cert_line(cert, GATED, _worktree_sha(repo, GATED))  # content A
    _write(repo, GATED, "print(2)\n")  # transient flip to content B
    shim = _sleep_shim(tmp_path, f"printf 'print(1)\\n' > \"{repo / GATED}\"")
    r = _run(f"git commit -m x {GATED}", repo, cert, path_prepend=shim)
    _assert_allowed(r)
    assert f"cert-retry: {GATED} recovered after re-hash" in r.stderr, r.stderr


def test_r2_stable_drift_still_blocks_after_retry(tmp_path: Path, cert: Path) -> None:
    """Same setup but the drift is STABLE (the shim settles nothing) -> the
    retry re-hash still mismatches and today's block verdict is kept
    (exit 2 + cert-diag), with no recovery line."""
    repo = _retry_repo(tmp_path)
    _cert_line(cert, GATED, _worktree_sha(repo, GATED))
    _write(repo, GATED, "print(2)\n")  # STABLE drift
    shim = _sleep_shim(tmp_path, ":")  # no-op sleep: nothing settles
    r = _run(f"git commit -m x {GATED}", repo, cert, path_prepend=shim)
    _assert_blocked(r)
    assert "cert-retry:" not in r.stderr, r.stderr
    assert "cert-diag:" in r.stderr, r.stderr


def test_r3_deleted_between_passes_is_exempt(tmp_path: Path, cert: Path) -> None:
    """A path deleted between the first pass and the retry mirrors the first
    pass's deletion-exempt semantics (no content lands -> skip + allow)."""
    repo = _retry_repo(tmp_path)
    _cert_line(cert, GATED, _worktree_sha(repo, GATED))
    _write(repo, GATED, "print(2)\n")
    shim = _sleep_shim(tmp_path, f'rm -f "{repo / GATED}"')
    r = _run(f"git commit -m x {GATED}", repo, cert, path_prepend=shim)
    _assert_allowed(r)
    assert f"cert-retry: {GATED} exempt after re-hash" in r.stderr, r.stderr


def test_r4_malformed_rehash_delay_fails_toward_block(tmp_path: Path, cert: Path) -> None:
    """A malformed EPM_CERT_REHASH_DELAY_S makes `sleep` fail; the re-check
    still runs and a stable drift still BLOCKS (a failed sleep never skips
    the re-check nor crashes the guard into a non-blocking exit)."""
    repo = _retry_repo(tmp_path)
    _cert_line(cert, GATED, _worktree_sha(repo, GATED))
    _write(repo, GATED, "print(2)\n")
    r = _run(f"git commit -m x {GATED}", repo, cert, rehash_delay="not-a-number")
    _assert_blocked(r)


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
    "stage_form",
    [
        "git add -A",
        "git add .",
        "git add --all",
        "git add ./",
        "git add .//",
        "git add :/",
        "git add \\*",
        "git add \\*\\*",
    ],
    ids=[
        "dash-A",
        "dot",
        "all",
        "dotslash",
        "dotslash-double",
        "pathspec-root",
        "star",
        "star-double",
    ],
)
def test_b13_blanket_add_chained_fails_closed(stage_form: str, art_repo: Path, cert: Path) -> None:
    _assert_blocked(_run(f"{stage_form} && git commit -m x", art_repo, cert))


# ---------------------------------------------------------------------------
# Path-limited `git add --all -- <pathspec>` exemption (issue #1977, plan §5).
# The blanket-staging latch defers to a per-ADD-clause post-scan: the
# sanctioned shape (root cwd, pre-`--` tokens ONLY from {-A,--all}, a literal
# `--`, >=1 clean literal pathspec candidate, zero rejections) resolves its
# landing set per file via a cwd-gated scoped `git status` and flows into
# Layer-2 classification; EVERY other shape keeps the fail-closed block
# (B13 above is unchanged).
# ---------------------------------------------------------------------------
def test_add_pathlimited_artifact_pathspec_allowed(art_repo: Path, cert: Path) -> None:
    """Plan §5.1: an artifact-only landing set under the exempted shape ALLOWS."""
    _write(art_repo, "tasks/t.md", "note\n")  # untracked artifact the add would stage
    _assert_allowed(_run("git add --all -- tasks/t.md && git commit -m x", art_repo, cert))


def test_add_pathlimited_gated_with_fresh_worktree_cert_allowed(
    code_repo: Path, cert: Path
) -> None:
    """Plan §5.2: the gated add-path enters classification, and a fresh
    worktree-content-bound cert satisfies it (classification ran + passed)."""
    _cert_line(cert, GATED, _worktree_sha(code_repo, GATED))
    _assert_allowed(_run(f"git add --all -- {GATED} && git commit -m x", code_repo, cert))


def test_add_pathlimited_gated_no_cert_blocks(code_repo: Path, cert: Path) -> None:
    """Plan §5.3: same shape WITHOUT the cert blocks — the add's path enters
    pending (the exemption opens no unclassified staging channel)."""
    _assert_blocked(_run(f"git add --all -- {GATED} && git commit -m x", code_repo, cert))


def test_add_pathlimited_dir_pathspec_resolves_untracked_gated(code_repo: Path, cert: Path) -> None:
    """Plan §5.4: a dir pathspec resolves PER FILE via the scoped git status
    read — the untracked, uncertified gated file under scripts/ blocks even
    though the staged gated file is certified."""
    _cert_line(cert, GATED, _worktree_sha(code_repo, GATED))
    r = _run("git add --all -- scripts && git commit -m x", code_repo, cert)
    _assert_blocked(r)
    assert "scripts/issue9_new.py" in r.stderr, r.stderr


def test_add_pathlimited_opaque_candidate_blocks(art_repo: Path, cert: Path) -> None:
    """Plan §5.5: an opaque ($VAR) candidate keeps the latch."""
    _assert_blocked(_run('git add --all -- "$V" && git commit -m x', art_repo, cert))


@pytest.mark.parametrize(
    "cand", [".", "./", "'*'", "*"], ids=["dot", "dotslash", "quoted-star", "star"]
)
def test_add_pathlimited_blanket_equivalent_candidate_blocks(
    cand: str, art_repo: Path, cert: Path
) -> None:
    """Plan §5.6: blanket-equivalent candidate spellings are rejected."""
    _assert_blocked(_run(f"git add --all -- {cand} && git commit -m x", art_repo, cert))


def test_add_pathlimited_no_ddash_blocks(art_repo: Path, cert: Path) -> None:
    """Plan §5.7: without a literal `--` the exemption never arms."""
    _write(art_repo, "tasks/t.md", "note\n")
    _assert_blocked(_run("git add --all tasks/t.md && git commit -m x", art_repo, cert))


def test_add_pathlimited_pre_ddash_positional_blocks(tmp_path: Path, cert: Path) -> None:
    """Plan §5.8 (MF-2): a pre-`--` positional is a LIVE pathspec the scoped
    status read would under-enumerate — the latch stays even with gated
    content under the positional."""
    repo = _init_repo(tmp_path, "preddash")
    _stage(repo, "tasks/t.md", "note\n")
    _write(repo, "src/issue9_mod.py", "print(1)\n")  # untracked gated, under `src`
    _assert_blocked(_run("git add --all src -- tasks/t.md && git commit -m x", repo, cert))


def test_add_pathlimited_force_flag_blocks(code_repo: Path, cert: Path) -> None:
    """Plan §5.9 (MF-2): `-f` stages ignored files the status read cannot see
    — any pre-`--` flag outside {-A,--all} keeps the latch, cert or no cert."""
    _cert_line(cert, GATED, _worktree_sha(code_repo, GATED))
    _assert_blocked(_run(f"git add --all -f -- {GATED} && git commit -m x", code_repo, cert))


def test_add_pathlimited_subdir_cwd_blocks(art_repo: Path, cert: Path) -> None:
    """Plan §5.10 (MF-1, c11-analogue): the exempted shape with hook cwd = a
    repo SUBDIR — the pathspec-resolution base is not provably the root."""
    _write(art_repo, "tasks/t.md", "note\n")
    r = _run(
        "git add --all -- tasks/t.md && git commit -m x",
        art_repo,
        cert,
        cwd=art_repo / "tasks",
    )
    _assert_blocked(r)


def test_add_pathlimited_missing_cwd_blocks(art_repo: Path, cert: Path) -> None:
    """Plan §5.10 (MF-1, c14-analogue): a missing hook-input cwd fails the
    cwd gate — block."""
    _write(art_repo, "tasks/t.md", "note\n")
    _assert_blocked(
        _run("git add --all -- tasks/t.md && git commit -m x", art_repo, cert, cwd=_OMIT_CWD)
    )


def test_add_pathlimited_cd_prefix_blocks(art_repo: Path, cert: Path) -> None:
    """Plan §5.11 (MF-1): an in-command cd to a repo subdir moves the
    resolution base (the cd_nonroot path) — block."""
    _write(art_repo, "tasks/t.md", "note\n")
    _assert_blocked(
        _run("cd tasks && git add --all -- tasks/t.md && git commit -m x", art_repo, cert)
    )


def test_add_pathlimited_env_chdir_wrapper_blocks(art_repo: Path, cert: Path) -> None:
    """Plan §5.11 (MF-1): an `env --chdir=<dir>` wrapper on the add clause is
    exemption-INELIGIBLE — the latch stays."""
    _assert_blocked(
        _run(
            "env --chdir=/tmp git add --all -- tasks/t.md && git commit -m x",
            art_repo,
            cert,
        )
    )


def test_add_pathlimited_sibling_blanket_clause_still_blocks(art_repo: Path, cert: Path) -> None:
    """Plan §5.12: a clean exempted clause must not un-latch a sibling
    bare-blanket clause in the same command."""
    _write(art_repo, "tasks/t.md", "note\n")
    _assert_blocked(
        _run("git add --all -- tasks/t.md && git add -A && git commit -m x", art_repo, cert)
    )


def _tracked_modified_unstaged_repo(tmp_path: Path) -> Path:
    """Gated file committed, then edited in the worktree; nothing staged —
    only the commit-clause pathspec / post-message ``-a`` carries the payload."""
    repo = _init_repo(tmp_path, "hashmsg")
    _stage(repo, GATED, "print(1)\n")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    _write(repo, GATED, "print(2)\n")  # modified, UNSTAGED
    return repo


def test_b15_pathspec_after_hash_bearing_message_blocks(tmp_path: Path, cert: Path) -> None:
    """Round-2 Major regression: the whitespace-anchored ``#`` inside the
    quoted message must NOT discard the pathspec token after it."""
    repo = _tracked_modified_unstaged_repo(tmp_path)
    _assert_blocked(_run(f'git commit -m "task #9: fix" {GATED}', repo, cert))


def test_b15b_post_message_dash_a_after_hash_bearing_message_blocks(
    tmp_path: Path, cert: Path
) -> None:
    """Same bug class, flag-scan half: a ``-a`` AFTER a ``#``-bearing message
    must still be seen by the flag scan."""
    repo = _tracked_modified_unstaged_repo(tmp_path)
    _assert_blocked(_run('git commit -m "task #9: fix" -a', repo, cert))


# ---------------------------------------------------------------------------
# Round 3 — string-literal masking class fix (concern
# quoted-message-seams-defeat-clause-scan). B16*/B17/A14 are the round-2
# review's demonstrated mis-parse shapes (probes S1-S5), each verified
# fail-pre-fix (round-2 hook b98f2393eb: exit 0) / pass-post-fix (exit 2).
# ---------------------------------------------------------------------------
def test_b16_semicolon_in_message_keeps_pathspec(tmp_path: Path, cert: Path) -> None:
    """Probe S1: a ``;`` inside the quoted message must not split the clause
    and drop the same-clause pathspec (silent allow)."""
    repo = _tracked_modified_unstaged_repo(tmp_path)
    _assert_blocked(_run(f'git commit -m "update; refactor" {GATED}', repo, cert))


def test_b16a_double_ampersand_in_message_keeps_pathspec(tmp_path: Path, cert: Path) -> None:
    """Probe S2: same class, ``&&`` inside the quoted message."""
    repo = _tracked_modified_unstaged_repo(tmp_path)
    _assert_blocked(_run(f'git commit -m "fix A && B" {GATED}', repo, cert))


def test_b16b_heredoc_message_keeps_pathspec(tmp_path: Path, cert: Path) -> None:
    """Probe S3: the repo-canonical heredoc message form — its NEWLINES must
    not split the clause and drop the trailing pathspec."""
    repo = _tracked_modified_unstaged_repo(tmp_path)
    cmd = f"git commit -m \"$(cat <<'EOF'\nupdate; refactor && more\nEOF\n)\" {GATED}"
    _assert_blocked(_run(cmd, repo, cert))


def test_b16c_post_message_dash_a_after_separator_message(tmp_path: Path, cert: Path) -> None:
    """Probe S4: a post-message ``-a`` after a ``;``-bearing message must
    still be seen by the flag scan."""
    repo = _tracked_modified_unstaged_repo(tmp_path)
    _assert_blocked(_run('git commit -m "update; refactor" -a', repo, cert))


def test_b17_dash_c_mention_in_message_does_not_waive(code_repo: Path, cert: Path) -> None:
    """Probe S5: a ``git -C`` MENTION inside the quoted message must not waive
    the whole clause (staged gated payload; control = B1 blocks)."""
    _assert_blocked(
        _run('git commit -m "docs: use git -C $WT commit for worktrees"', code_repo, cert)
    )


def test_a14_commit_then_scripts_tool_compound_allowed(art_repo: Path, cert: Path) -> None:
    """Over-collection guard: the common commit-then-post-marker compound must
    stay allowed — the second clause's ``scripts/task.py`` is not verb-anchored
    and must not be collected (per-clause scan, never a whole-command scan)."""
    _assert_allowed(
        _run(
            "git commit -m x tasks/t.md && "
            "uv run python scripts/task.py post-marker 9 epm:progress --note done",
            art_repo,
            cert,
        )
    )


# Per-span-shape battery: for each string-literal shape, (a) literal content
# carrying separators + a gated-looking path + ``-a`` + a ``git -C`` mention
# contributes NO tokens (allowed on the tracked-modified-unstaged repo, where
# any spurious pathspec / ``-a`` / clause split WOULD change the verdict), and
# (b) a genuine pathspec OUTSIDE the literal still parses (blocked). The pair
# is sharp in both directions: (a) fails if literal content leaks into the
# scan; (b) fails if the literal swallows/splits away real tokens or waives
# the clause.
_SPAN_MESSAGES = [
    (
        "double_quoted",
        '-m "update; more && x | y & z -a scripts/fake_gated.py git -C /elsewhere #9"',
    ),
    ("double_quoted_multiline", '-m "line one\nline two -a scripts/fake_gated.py"'),
    ("single_quoted", "-m 'update; more && -a scripts/fake_gated.py git -C /elsewhere'"),
    ("escaped_quotes_in_double", r'-m "say \"scripts/fake_gated.py; -a\" ok"'),
    ("escaped_chars_unquoted", r"-m update\;\ -a\ scripts/fake_gated.py"),
    ("dollar_paren_in_double", "-m \"$(echo 'scripts/fake_gated.py; -a')\""),
    (
        "heredoc_in_dollar_paren",
        "-m \"$(cat <<'EOF'\nupdate; -a scripts/fake_gated.py git -C /x\nEOF\n)\"",
    ),
    ("ansi_c_quoted", r"-m $'update;\n-a scripts/fake_gated.py'"),
    ("backtick_in_double", '-m "ver `echo x; echo scripts/fake_gated.py -a`"'),
]


@pytest.mark.parametrize("msg", [m for _, m in _SPAN_MESSAGES], ids=[n for n, _ in _SPAN_MESSAGES])
def test_span_battery_literal_content_contributes_no_tokens(
    msg: str, tmp_path: Path, cert: Path
) -> None:
    repo = _tracked_modified_unstaged_repo(tmp_path)
    _assert_allowed(_run(f"git commit {msg}", repo, cert))


@pytest.mark.parametrize("msg", [m for _, m in _SPAN_MESSAGES], ids=[n for n, _ in _SPAN_MESSAGES])
def test_span_battery_genuine_pathspec_outside_literal_still_blocks(
    msg: str, tmp_path: Path, cert: Path
) -> None:
    repo = _tracked_modified_unstaged_repo(tmp_path)
    _assert_blocked(_run(f"git commit {msg} {GATED}", repo, cert))


def test_genuine_trailing_comment_contributes_no_tokens(tmp_path: Path, cert: Path) -> None:
    """Round-3 flip of the round-2 documented over-collection: a genuine shell
    comment is dropped by the masker (comments are not command arguments), so
    a gated path / ``-a`` named in a real comment no longer false-blocks."""
    repo = _tracked_modified_unstaged_repo(tmp_path)
    _assert_allowed(_run(f"git commit -m x # split {GATED} later; then -a", repo, cert))


def test_pathspec_before_genuine_trailing_comment_still_blocks(tmp_path: Path, cert: Path) -> None:
    repo = _tracked_modified_unstaged_repo(tmp_path)
    _assert_blocked(_run(f"git commit -m x {GATED} # note", repo, cert))


def test_quoted_env_assignment_with_separator_before_commit_blocks(
    tmp_path: Path, cert: Path
) -> None:
    """Same string-literal class, lead-anchor half: a quoted env-assignment
    value carrying a ``;`` used to SPLIT the clause so the commit never
    classified (silent allow); masked, the wrapper prefix matches and the
    pathspec is collected."""
    repo = _tracked_modified_unstaged_repo(tmp_path)
    _assert_blocked(_run(f'MSG="a; b" git commit -m x {GATED}', repo, cert))


def test_quoted_env_assignment_value_contributes_no_tokens(tmp_path: Path, cert: Path) -> None:
    repo = _tracked_modified_unstaged_repo(tmp_path)
    _assert_allowed(_run('MSG="a; b scripts/fake_gated.py -a" git commit -m x', repo, cert))


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


# ---------------------------------------------------------------------------
# c-group (issue #1620): pathspec-scoped staged-index certification check.
# A pathspec-limited root-cwd commit is never blocked by a FOREIGN uncertified
# staged file outside its pathspec; every ambiguity falls back to the
# whole-index check (block direction).


@pytest.fixture
def foreign_repo(tmp_path: Path) -> Path:
    """Repo with a FOREIGN uncertified gated file staged + artifacts staged."""
    repo = _init_repo(tmp_path, "foreign")
    _stage(repo, "scripts/foreign.py", "print(0)\n")
    _stage(repo, "tasks/t.md", "note\n")
    _stage(repo, "eval_results/a.json", "{}\n")
    _stage(repo, "docs/b.md", "doc\n")
    return repo


def test_c1_pathspec_commit_foreign_uncertified_staged_allowed(
    foreign_repo: Path, cert: Path
) -> None:
    _assert_allowed(_run("git commit -m x -- tasks/t.md", foreign_repo, cert))


def test_c1b_no_dashdash_form_allowed(foreign_repo: Path, cert: Path) -> None:
    _assert_allowed(_run("git commit -m x tasks/t.md", foreign_repo, cert))


def test_c1c_quoted_pathspec_after_dashdash_allowed(foreign_repo: Path, cert: Path) -> None:
    _assert_allowed(_run('git commit -m x -- "tasks/t.md"', foreign_repo, cert))


def test_c16_glob_pathspec_excluding_foreign_allowed(foreign_repo: Path, cert: Path) -> None:
    # Incident event (b) shape: unquoted glob + dir pathspec; git evaluates both.
    _assert_allowed(_run("git commit -m x -- eval_results/*.json docs/", foreign_repo, cert))


def test_c2_pathspec_commit_own_gated_payload_uncertified_blocks(
    foreign_repo: Path, cert: Path
) -> None:
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    r = _run("git commit -m x -- scripts/own.py", foreign_repo, cert)
    _assert_blocked(r)
    # Negative control for c9: an own-payload (non-foreign) block carries no
    # FOREIGN-STAGED? paragraph.
    assert "FOREIGN-STAGED?" not in r.stderr, r.stderr


def test_c6_pathspec_certified_own_payload_allowed_despite_foreign(
    foreign_repo: Path, cert: Path
) -> None:
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    # Cert bound to the WORKTREE sha: a pathspec commit lands worktree content.
    _cert_line(cert, "scripts/own.py", _worktree_sha(foreign_repo, "scripts/own.py"))
    _assert_allowed(_run("git commit -m x -- scripts/own.py", foreign_repo, cert))


def test_c3_dir_pathspec_covering_uncertified_staged_blocks(foreign_repo: Path, cert: Path) -> None:
    _assert_blocked(_run("git commit -m x -- scripts/", foreign_repo, cert))


def test_c4_bare_commit_foreign_uncertified_staged_still_blocks(
    foreign_repo: Path, cert: Path
) -> None:
    _assert_blocked(_run("git commit -m x", foreign_repo, cert))


def test_c11_subdir_cwd_relative_pathspec_blocks(tmp_path: Path, cert: Path) -> None:
    # MF-1 must-BLOCK cell: subdir cwd + bare-name pathspec naming a gated
    # staged file cwd-relatively; naive root-anchored scoping would false-ALLOW.
    repo = _init_repo(tmp_path, "subdir")
    _stage(repo, "scripts/inner.py", "print(1)\n")
    _assert_blocked(_run("git commit -m x inner.py", repo, cert, cwd=repo / "scripts"))


def test_c12_scoping_engages_only_at_root_cwd(foreign_repo: Path, cert: Path) -> None:
    cmd = "git commit -m x -- tasks/t.md"
    _assert_allowed(_run(cmd, foreign_repo, cert, cwd=foreign_repo))
    _assert_blocked(_run(cmd, foreign_repo, cert, cwd=foreign_repo / "tasks"))


def test_c13_missing_cwd_falls_back_blocks(foreign_repo: Path, cert: Path) -> None:
    _assert_blocked(_run("git commit -m x -- tasks/t.md", foreign_repo, cert, cwd=_OMIT_CWD))


def test_c14_in_command_cd_nonroot_disables_scoping(foreign_repo: Path, cert: Path) -> None:
    _assert_blocked(_run("cd tasks && git commit -m x -- t.md", foreign_repo, cert))


def test_c14b_env_chdir_disables_scoping(foreign_repo: Path, cert: Path) -> None:
    _assert_blocked(_run("env --chdir=tasks git commit -m x -- t.md", foreign_repo, cert))


@pytest.mark.parametrize(
    "tok",
    ["$F", "$(ls)", "`ls`", "~/tasks/t.md"],
    ids=["variable", "cmdsub", "backtick", "tilde"],
)
def test_c15_variable_pathspec_token_blocks(tok: str, foreign_repo: Path, cert: Path) -> None:
    # MF-2: an unexpanded shell token would scope-away real landing content.
    _assert_blocked(_run(f"git commit -m x -- {tok}", foreign_repo, cert))


def test_c7_opaque_spacey_quoted_pathspec_falls_back_blocks(foreign_repo: Path, cert: Path) -> None:
    _assert_blocked(_run('git commit -m x -- "tasks/my file.md"', foreign_repo, cert))


def test_c7b_include_flag_disables_scoping_blocks(foreign_repo: Path, cert: Path) -> None:
    _assert_blocked(_run("git commit -i -m x -- tasks/t.md", foreign_repo, cert))


def test_c7c_pathspec_from_file_disables_scoping_blocks(foreign_repo: Path, cert: Path) -> None:
    _assert_blocked(_run("git commit --pathspec-from-file=list.txt -m x", foreign_repo, cert))


def test_c7d_unknown_long_flag_disables_scoping_blocks(foreign_repo: Path, cert: Path) -> None:
    _assert_blocked(_run("git commit --future-flag arg -m x -- tasks/t.md", foreign_repo, cert))


def test_c9_block_message_orders_pathspec_recovery_before_escape_hatch(
    foreign_repo: Path, cert: Path
) -> None:
    r = _run("git commit -m x", foreign_repo, cert)
    _assert_blocked(r)
    # Anchored on the recovery paragraph's unique opener, before the env-var
    # escape hatch is ever named.
    assert "FOREIGN-STAGED?" in r.stderr, r.stderr
    assert r.stderr.index("FOREIGN-STAGED?") < r.stderr.index("EPM_ALLOW_ROOT_CODE_COMMIT"), (
        r.stderr
    )


def test_c8_cert_diag_line_no_cert(code_repo: Path, cert: Path) -> None:
    r = _run("git commit -m x", code_repo, cert)
    _assert_blocked(r)
    assert f"cert-diag: {GATED}" in r.stderr, r.stderr
    assert "cert=none-for-path" in r.stderr, r.stderr


def test_c8b_cert_diag_sha_mismatch(code_repo: Path, cert: Path) -> None:
    _cert_line(cert, GATED, "0" * 40)
    r = _run("git commit -m x", code_repo, cert)
    _assert_blocked(r)
    assert "cert=sha-mismatch:" in r.stderr, r.stderr


def test_c8c_cert_diag_stale(code_repo: Path, cert: Path) -> None:
    _cert_line(cert, GATED, _staged_sha(code_repo, GATED), epoch=1)
    r = _run("git commit -m x", code_repo, cert)
    _assert_blocked(r)
    assert "cert=stale:" in r.stderr, r.stderr


# --- provably-root cd prefix scope base (issue #2357) -----------------------
# Every NEW-ARM allow node carries an EXPLICIT non-root cwd (cwd=tmp_path, or
# the linked-worktree fixture in c17d): _run defaults the payload cwd to the
# fixture repo root, where the pre-#2357 cwd_ok gate already allows these
# commands — a default-cwd allow node would be vacuous for the new arm
# (r1 MF-5). Differential expectation, stated once for the whole c17 family:
# each allow command BLOCKS (rc=2) on the pre-#2357 guard from its non-root
# cwd (the plan-§2 live trace recorded exactly this for the incident shapes)
# and ALLOWS (rc=0) post-fix.

_STALE_MARGIN_S = 21600 + 100  # default MAX_AGE (6 h) + margin


def test_c17_cd_root_prefix_pathspec_commit_scopes_from_nonroot_cwd(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    _assert_allowed(
        _run(
            f"cd {foreign_repo} && git commit -m x -- tasks/t.md",
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


def test_c17b_cd_root_prefix_compound_add_commit_scopes(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # Incident shape (i): compound cd -> cp -> add -> commit + redirect +
    # trailing SEQ clause (outside the cd->commit span — chain unaffected).
    cmd = (
        f"cd {foreign_repo} && cp tasks/t.md {tmp_path}/copy.md && git add tasks/t.md "
        f'&& git commit -m "agent-memory: note" -- tasks/t.md > {tmp_path}/out.log 2>&1; '
        'echo "commit rc=$?"'
    )
    _assert_allowed(_run(cmd, foreign_repo, cert, cwd=tmp_path))


@pytest.mark.parametrize(
    "cdform", ["{repo}", "{repo}/", '"{repo}"'], ids=["bare", "slash", "quoted"]
)
def test_c17c_cd_root_spelling_variants_scope(
    cdform: str, foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    tgt = cdform.format(repo=foreign_repo)
    _assert_allowed(
        _run(f"cd {tgt} && git commit -m x -- tasks/t.md", foreign_repo, cert, cwd=tmp_path)
    )


def test_c17d_cd_root_prefix_from_linked_worktree_cwd_scopes(tmp_path: Path, cert: Path) -> None:
    # THE incident cwd class: hook cwd inside a linked worktree. The
    # cd-to-root sets retarget_evidence=1, which disables the #2066
    # worktree-cwd allow gate — the #2357 arm is the ONLY allow path.
    repo = _init_repo(tmp_path, "wtroot")
    _stage(repo, "tasks/t.md", "note\n")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    _stage(repo, "scripts/foreign.py", "print(0)\n")  # foreign uncertified gated staged
    wt = tmp_path / "wtroot-wt"
    _git(repo, "worktree", "add", "-q", str(wt))
    _assert_allowed(_run(f"cd {repo} && git commit -m x -- tasks/t.md", repo, cert, cwd=wt))


def test_c17e_cd_root_prefix_pathspec_plus_redirect_scopes(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # Incident shapes (ii)/(iii): the guard's own prescribed remediation shape
    # under a cd-root prefix, from a non-root cwd.
    cmd = (
        f"cd {foreign_repo} && git commit -m x -- tasks/t.md > {tmp_path}/log 2>&1; "
        f"echo rc=$?; tail -1 {tmp_path}/log"
    )
    _assert_allowed(_run(cmd, foreign_repo, cert, cwd=tmp_path))


@pytest.mark.parametrize("prefix", ["true; ", "true\n"], ids=["seq", "nl"])
def test_c17f_seq_nl_separated_root_cd_scopes(
    prefix: str, foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # Own-sep SEQ/NL arming in the ALLOW direction (the g3-matching set).
    cmd = f"{prefix}cd {foreign_repo} && git commit -m x -- tasks/t.md"
    _assert_allowed(_run(cmd, foreign_repo, cert, cwd=tmp_path))


def test_c17g_cd_root_prefix_msgfile_commit_scopes(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # Interaction pin: the #1949 -F message-file arm composing with the new base.
    msgfile = tmp_path / "msg.txt"
    msgfile.write_text("msg\n", encoding="utf-8")
    _assert_allowed(
        _run(
            f"cd {foreign_repo} && git commit -F {msgfile} -- tasks/t.md",
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


def test_c18_commit_before_root_cd_stays_whole_index(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    _assert_blocked(
        _run(f"git commit -m x -- tasks/t.md; cd {foreign_repo}", foreign_repo, cert, cwd=tmp_path)
    )


def test_c18b_or_separated_root_cd_stays_whole_index(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    _assert_blocked(
        _run(
            f"true || cd {foreign_repo} && git commit -m x -- tasks/t.md",
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


def test_c18c_backgrounded_root_cd_stays_whole_index(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    _assert_blocked(
        _run(f"cd {foreign_repo} & git commit -m x -- tasks/t.md", foreign_repo, cert, cwd=tmp_path)
    )


def test_c18d_root_cd_in_compound_context_stays_whole_index(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # PROBING form (r1 MF-3): the cd record LEADS with `cd` inside the while
    # body (own-sep NL, all-AND chain to the commit) — ONLY the D4
    # compound-context refusal blocks arming here.
    cmd = f"while true\ndo\ncd {foreign_repo} && git commit -m x -- tasks/t.md\nbreak\ndone"
    _assert_blocked(_run(cmd, foreign_repo, cert, cwd=tmp_path))


def test_c18e_absolute_root_subdir_cd_stays_whole_index(foreign_repo: Path, cert: Path) -> None:
    _assert_blocked(_run(f"cd {foreign_repo}/tasks && git commit -m x -- t.md", foreign_repo, cert))


def test_c18f_variable_root_cd_stays_whole_index(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # Original-target rule (#1676 must-ask): a variable target resolving to
    # the root never arms the new base (and disables scoping via cd_nonroot).
    _assert_blocked(
        _run(
            f'R={foreign_repo}; cd "$R" && git commit -m x -- tasks/t.md',
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


def test_c19_cd_root_prefix_own_gated_uncertified_still_blocks(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # The widening never weakens cert enforcement under the new base.
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            f"cd {foreign_repo} && git commit -m x -- scripts/own.py",
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


def test_c19b_cd_root_prefix_own_gated_fresh_worktree_cert_allows(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # Binding rule preserved under the new base: a pathspec commit lands
    # WORKTREE content, so the worktree-sha cert allows (c6 analogue).
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _cert_line(cert, "scripts/own.py", _worktree_sha(foreign_repo, "scripts/own.py"))
    _assert_allowed(
        _run(
            f"cd {foreign_repo} && git commit -m x -- scripts/own.py",
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


def test_c21_all_stale_content_matching_block_leads_with_recertify_hint(
    foreign_repo: Path, cert: Path
) -> None:
    # ALL uncertified paths stale-matching (want==staged==worktree, age past
    # MAX_AGE) => the block leads with the STALE-CERT? re-certify hint BEFORE
    # the foreign-payload attribution and before the escape hatch is named.
    _cert_line(
        cert,
        "scripts/foreign.py",
        _staged_sha(foreign_repo, "scripts/foreign.py"),
        epoch=int(time.time()) - _STALE_MARGIN_S,
    )
    r = _run("git commit -m x", foreign_repo, cert)
    _assert_blocked(r)
    assert "STALE-CERT?" in r.stderr, r.stderr
    assert r.stderr.index("STALE-CERT?") < r.stderr.index("FOREIGN-STAGED?"), r.stderr
    assert r.stderr.index("STALE-CERT?") < r.stderr.index("EPM_ALLOW_ROOT_CODE_COMMIT"), r.stderr


def test_c21b_stale_hint_absent_on_content_drift(foreign_repo: Path, cert: Path) -> None:
    _cert_line(
        cert,
        "scripts/foreign.py",
        _staged_sha(foreign_repo, "scripts/foreign.py"),
        epoch=int(time.time()) - _STALE_MARGIN_S,
    )
    _write(foreign_repo, "scripts/foreign.py", "print(9)\n")  # worktree drifts from staged
    r = _run("git commit -m x", foreign_repo, cert)
    _assert_blocked(r)
    assert "STALE-CERT?" not in r.stderr, r.stderr


def test_c21c_stale_hint_absent_on_mixed_uncertified_set(foreign_repo: Path, cert: Path) -> None:
    _cert_line(
        cert,
        "scripts/foreign.py",
        _staged_sha(foreign_repo, "scripts/foreign.py"),
        epoch=int(time.time()) - _STALE_MARGIN_S,
    )
    _stage(foreign_repo, "scripts/own2.py", "print(2)\n")  # certless second gated path
    r = _run("git commit -m x", foreign_repo, cert)
    _assert_blocked(r)
    assert "STALE-CERT?" not in r.stderr, r.stderr


def test_c22_foreign_para_names_cd_root_prefix_recovery(foreign_repo: Path, cert: Path) -> None:
    # Presence pin (not exact-count): the FOREIGN-STAGED? paragraph names the
    # cd-prefix recovery shape the #2357 widening actually honors.
    r = _run("git commit -m x", foreign_repo, cert)
    _assert_blocked(r)
    assert "FOREIGN-STAGED?" in r.stderr, r.stderr
    assert "cd <absolute repo root>" in r.stderr, r.stderr


def test_c23_and_separated_root_cd_then_seq_commit_blocked(foreign_repo: Path, cert: Path) -> None:
    # r1 MF-1: the cd is skipped by its failed && predecessor while the SEQ
    # commit still runs at the subdir cwd, where own.py resolves to the gated
    # scripts/own.py — the c11-class bypass the v3 predicate admitted.
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            f"false && cd {foreign_repo}; git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


def test_c23b_and_separated_root_cd_then_or_commit_blocked(foreign_repo: Path, cert: Path) -> None:
    _stage(foreign_repo, "scripts/inner.py", "print(1)\n")
    _assert_blocked(
        _run(
            f"false && cd {foreign_repo} || git commit -m x -- inner.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


def test_c24_trailing_token_root_cd_never_arms(foreign_repo: Path, cert: Path) -> None:
    # r1 MF-2: `cd <root> junk` fails deterministically ("too many
    # arguments") while the SEQ-chained commit still runs at the subdir cwd.
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            f"cd {foreign_repo} junk; git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


def test_c25_subshell_cd_root_or_commit_blocked(foreign_repo: Path, cert: Path) -> None:
    # r1 MF-3: doubly refused — `(`-opener compound context (D4) AND the OR
    # chain break (D3).
    _stage(foreign_repo, "scripts/inner.py", "print(1)\n")
    _assert_blocked(
        _run(
            f"(cd {foreign_repo}; false) || git commit -m x -- inner.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


def test_c25b_subshell_cd_root_and_chain_commit_blocked(foreign_repo: Path, cert: Path) -> None:
    # r1 MF-3, the chain-dominance-alone bypass: the separator walk reads
    # all-AND and the record lead-strip hides the paren from the cd arm — the
    # SOLE refusing condition is the D4 `(`-opener check (mutation pin for
    # the `(` arm specifically).
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            f"(cd {foreign_repo} && true) && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


def test_c26_own_sep_pipe_root_cd_blocked(foreign_repo: Path, cert: Path) -> None:
    # r1 MF-4: a piped cd runs in a pipeline subshell — the parent cwd never
    # moves while the && commit still runs at the subdir cwd.
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            f"true | cd {foreign_repo} && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


def test_c26b_next_sep_pipe_root_cd_blocked(foreign_repo: Path, cert: Path) -> None:
    # r1 MF-4: next-sep PIPE refusal (chain dominance subsumes it; the cell
    # exists as a mutation pin on the PIPE token either way).
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            f"cd {foreign_repo} | true; git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


# --- c27 family: symbolic ALIAS root spellings never arm (r2 fix, concern
# ``mutable-symbolic-root-proof``). The tilde / HOME-variable alias spellings
# keep their legacy NON-poisoning classification (cd_nonroot stays 0) but the
# arming set is the canonical ABSOLUTE spelling only: an alias's runtime
# destination depends on $HOME (mutable) + symlink resolution, neither
# provable from command text, while the armed cert check resolves pathspecs
# root-pinned. Each case runs from a root-SUBDIR cwd with a staged
# uncertified gated file and a cwd-relative pathspec — the c11-class bypass
# shape (B49/B50 sibling): pre-fix each ARMED and exited 0; post-fix the
# whole-index read blocks (exit 2).


def test_c27_tilde_alias_root_cd_stays_whole_index(foreign_repo: Path, cert: Path) -> None:
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            "cd ~/explore-persona-space && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


def test_c27b_homevar_alias_root_cd_stays_whole_index(foreign_repo: Path, cert: Path) -> None:
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            "cd $HOME/explore-persona-space && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


def test_c27c_homevar_reassigned_alias_root_cd_stays_whole_index(
    foreign_repo: Path, cert: Path
) -> None:
    # A same-command $HOME reassignment makes the alias's runtime destination
    # arbitrary — the arm must stay cold on the SPELLING alone (it never
    # inspects the assignment).
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            "HOME=/tmp/elsewhere; cd $HOME/explore-persona-space && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


def test_c27d_quoted_homevar_alias_root_cd_stays_whole_index(
    foreign_repo: Path, cert: Path
) -> None:
    # Quote-suppressed form: single quotes make the runtime cd fail outright
    # (a literal '$HOME/...' directory) while the guard's one-quote-pair
    # strip still sees the alias spelling — must stay non-arming.
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            "cd '$HOME/explore-persona-space' && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


# --- c28 family: a non-canonical cd RIDING an armed canonical chain DISARMS
# the widening (r3, concern ``mutable-symbolic-root-proof``). The r2 fix
# closed the STANDALONE alias arming (c27) but kept the alias spellings as
# non-poisoning no-op keeps, so ``cd <root> && cd <alias> && git commit -- p``
# kept cd_root_base=1 from the canonical prefix while the shell's real
# pathspec-resolution base was the alias's RUNTIME destination
# (attacker-controllable via a crafted same-command $HOME or a planted
# symlink) — a verified false-allow on r2 (origin/main BLOCKS, r2 ALLOWS).
# The invariant pinned here: cd_root_base scopes ONLY when the LAST
# cwd-moving cd before every commit clause is the canonical absolute root.
# Each block case runs from a root-SUBDIR cwd with a staged uncertified
# gated file and a cwd-relative pathspec (the c11-class bypass shape);
# post-fix the whole-index read blocks (exit 2).


def test_c28_tilde_alias_riding_armed_canonical_chain_blocked(
    foreign_repo: Path, cert: Path
) -> None:
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            f"cd {foreign_repo} && cd ~/explore-persona-space && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


def test_c28b_homevar_alias_riding_armed_canonical_chain_blocked(
    foreign_repo: Path, cert: Path
) -> None:
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            f"cd {foreign_repo} && cd $HOME/explore-persona-space && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


def test_c28c_home_reassigned_alias_riding_armed_canonical_chain_blocked(
    foreign_repo: Path, cert: Path
) -> None:
    # The c27c reassignment variant composed with the riding chain: a
    # same-command $HOME reassignment makes the riding alias's destination
    # arbitrary while the canonical prefix already armed the base.
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            f"HOME=/tmp/elsewhere; cd {foreign_repo} && cd $HOME/explore-persona-space"
            " && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


def test_c28d_canonical_prefixed_subdir_cd_riding_armed_chain_blocked(
    foreign_repo: Path, cert: Path
) -> None:
    # A canonical-PREFIXED subdir cd is non-canonical (hits the ``*)`` arm:
    # cd_nonroot=1 already kills the gate) AND must also disarm the base —
    # the r3 belt keeps the cd_root_base predicate locally sound without
    # leaning on the distant cd_nonroot AND-term.
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _assert_blocked(
        _run(
            f"cd {foreign_repo} && cd {foreign_repo}/scripts && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


# --- c29 family: re-arming guard (over-tightening protection for the r3
# disarm). A non-canonical cd BEFORE the arming canonical cd never disarms,
# and a canonical arming cd AFTER a disarm re-establishes the base — the
# LAST cwd-moving cd before the commit is canonical, so scoping still
# engages from a non-root cwd (the foreign uncertified gated staged file
# makes scoping observable: exit 0 iff the pathspec-scoped read ran).
# NOTE the re-arm pins use the SEQ separator before the canonical cd: the
# ``cd <alias> && cd <root> && git commit`` AND-form never scopes on ANY
# version (origin/main, r2, r3 all block) — the arming cd's own separator
# must be START/SEQ/NL (the r1 MF-1 AND-exclusion), a pre-existing refusal
# c29c pins below. (The plain canonical allow ``cd <root> && git commit --
# tasks/t.md`` from a non-root cwd is already pinned by c17.)


def test_c29_alias_then_canonical_seq_rearm_still_scopes(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    _assert_allowed(
        _run(
            f"cd ~/explore-persona-space; cd {foreign_repo} && git commit -m x -- tasks/t.md",
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


def test_c29b_disarm_then_canonical_rearm_still_scopes(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # armed -> alias disarm -> SEQ-separated canonical re-arm -> commit:
    # the re-arm clears the disarm (the base is measured from the LAST
    # canonical cd), so the pathspec still scopes.
    _assert_allowed(
        _run(
            f"cd {foreign_repo} && cd ~/explore-persona-space; cd {foreign_repo}"
            " && git commit -m x -- tasks/t.md",
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


def test_c29c_alias_then_and_separated_canonical_never_arms(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # Pre-existing refusal pin (r1 MF-1 AND-exclusion, unchanged by r3):
    # an AND-separated canonical cd never arms, so the alias-then-canonical
    # ALL-AND chain stays whole-index from a non-root cwd on every version.
    _assert_blocked(
        _run(
            f"cd ~/explore-persona-space && cd {foreign_repo} && git commit -m x -- tasks/t.md",
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


# --- c30 family (r4, concerns non-cd-cwd-mover-rides-armed-chain /
# unmodeled-cwd-mover-survives-scope-base): a cwd-moving record NOT spelled
# as a plain-lead `cd` riding an armed canonical chain DISARMS the widening.
# The r3 disarm fired only on `^cd`-dispatched records, so pushd/popd,
# prefixed/escaped/quoted builtin-cd spellings, eval'd cds, and sourced
# scripts moved the real pathspec-resolution base while cd_root_seen stayed
# 1 — executed rc differential (hermetic blob probe, 2026-08-18): every
# instance below was origin/main=2, r3(b307923fc1)=0, r4=2. Same fixture
# shape as c28: root-SUBDIR cwd, staged uncertified gated file, cwd-relative
# pathspec; post-fix the whole-index read blocks (exit 2).
_C30_MOVERS = [
    pytest.param("pushd scripts", id="pushd"),
    pytest.param("popd", id="popd"),
    pytest.param("builtin cd scripts", id="builtin-cd"),
    pytest.param("command cd scripts", id="command-cd"),
    pytest.param("\\cd scripts", id="backslash-cd"),
    pytest.param("'cd' scripts", id="quoted-cd"),
    pytest.param("eval 'cd scripts'", id="eval-cd"),
    pytest.param(". scripts/env.sh", id="dot-source"),
    pytest.param("source scripts/env.sh", id="source"),
]


@pytest.mark.parametrize("mover", _C30_MOVERS)
def test_c30_noncd_mover_riding_armed_canonical_chain_blocked(
    mover: str, foreign_repo: Path, cert: Path
) -> None:
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _write(foreign_repo, "scripts/env.sh", "cd scripts\n")
    _assert_blocked(
        _run(
            f"cd {foreign_repo} && {mover} && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


# --- c30b family (r5, reconcile epm:review-reconcile v4 — concerns
# non-cd-cwd-mover-rides-armed-chain / unmodeled-cwd-mover-survives-scope-base
# RE-OPENED): a record carrying a leading legal ASSIGNMENT PREFIX in front of
# a mover defeated the ^-anchored CWD_MOVER_LEAD_ERE, and dot-source is the
# ONE mover family with no whole-record-screen fallback (the lone dot is
# deliberately unscreened), so the record rode the armed chain undisarmed.
# Executed rc differential (hermetic blob probe, 2026-08-18, this fixture
# shape): assign-prefixed dot-source AND its quoted-value variant were
# origin/main=2, r4(22e76689e2)=0 (the round-introduced false-allow), r5=2.
# The r5 fix tolerates zero-or-more NAME=value prefixes before the WHOLE
# mover family (uniform, not dot-only), and the disarm site greps the MASKED
# lead too: a QUOTED assignment value hides its interior space on the raw
# text but is spaceless filler on the masked copy (the quoted-value param
# below pins exactly that arm). builtin-dot-source pins the wrapper path
# (family-covered since r4: the wrapper word itself matches).
_C30B_PREFIXED_MOVERS = [
    pytest.param("FOO=bar . scripts/env.sh", id="assign-dot-source"),  # the r4 crux
    pytest.param('FOO="a b" . scripts/env.sh', id="quoted-value-assign-dot-source"),
    pytest.param("A=1 B=2 . scripts/env.sh", id="multi-assign-dot-source"),
    pytest.param("FOO=bar source scripts/env.sh", id="assign-source"),  # screen-covered control
    pytest.param("FOO=bar pushd scripts", id="assign-pushd"),
    pytest.param("FOO=bar eval 'cd scripts'", id="assign-eval-cd"),
    pytest.param("builtin . scripts/env.sh", id="builtin-dot-source"),
    pytest.param("FOO=bar builtin . scripts/env.sh", id="assign-builtin-dot-source"),
]


@pytest.mark.parametrize("mover", _C30B_PREFIXED_MOVERS)
def test_c30b_assignment_prefixed_mover_riding_armed_chain_blocked(
    mover: str, foreign_repo: Path, cert: Path
) -> None:
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _write(foreign_repo, "scripts/env.sh", "cd scripts\n")
    _assert_blocked(
        _run(
            f"cd {foreign_repo} && {mover} && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


# --- c30d family (r6, #2371): two text-detectable prefixed mover forms rode
# the armed canonical `&&` chain undisarmed past the r5 grammar — (1) an
# APPEND-assignment prefix (the r5 atom's `NAME=` cannot match `+=`), and
# (2) the `time` keyword wrapper (`time` is a shell keyword outside both the
# prefix group and the mover family). Both source in the CURRENT shell
# (execution-confirmed reachability probes 2026-08-18, GNU bash 5.1.16 —
# plan #2371 §1/§6), so a cwd move rode the armed chain and the relative
# pathspec resolved off-root (scoped permit where the whole-index read must
# engage). The r6 fix widens ONLY the prefix group's atom set: assignment
# atoms tolerate an optional `+` before `=`, a second atom recognizes the
# `time` keyword wrapper (bare or `time -p`), atoms repeat zero-or-more in
# any interleaving. Param notes: quoted-value-append pins the MASKED-lead
# arm (the quoted value's interior hides the prefix on the raw text);
# assign-time pins the DELIBERATE over-tightening — `FOO=bar time . x` runs
# EXTERNAL time (cannot source) but matches the grammar and disarms, block
# direction, kept; append-assign-source + time-pushd are screen-covered /
# uniform-family controls (`source` / `pushd` vocabulary already disarms via
# the whole-record screen); command-dot-source pins the pre-r6 family-word
# wrapper path (already blocked — the family word matches at the lead).
_C30D_PREFIXED_MOVERS = [
    pytest.param("FOO+=bar . scripts/env.sh", id="append-assign-dot-source"),  # gap 1 crux
    pytest.param("FOO+='a b' . scripts/env.sh", id="quoted-value-append-dot-source"),
    pytest.param("A+=1 B=2 . scripts/env.sh", id="mixed-multi-assign-dot-source"),
    pytest.param("time . scripts/env.sh", id="time-dot-source"),  # gap 2 crux
    pytest.param("time -p . scripts/env.sh", id="time-p-dot-source"),
    pytest.param("time FOO=bar . scripts/env.sh", id="time-assign-interleave-dot-source"),
    pytest.param("FOO=bar time . scripts/env.sh", id="assign-time-dot-source"),
    pytest.param("FOO+=bar source scripts/env.sh", id="append-assign-source"),
    pytest.param("time pushd scripts", id="time-pushd"),
    pytest.param("command . scripts/env.sh", id="command-dot-source"),
]


@pytest.mark.parametrize("mover", _C30D_PREFIXED_MOVERS)
def test_c30d_extended_prefix_mover_riding_armed_chain_blocked(
    mover: str, foreign_repo: Path, cert: Path
) -> None:
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _write(foreign_repo, "scripts/env.sh", "cd scripts\n")
    _assert_blocked(
        _run(
            f"cd {foreign_repo} && {mover} && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


# The r6 (#2371) prefix group, verbatim from CWD_MOVER_LEAD_ERE: alternated
# atoms — plain/append assignment (NAME=v / NAME+=v) or the `time` keyword
# wrapper (bare / `time -p`) — repeated zero-or-more in any interleaving.
# Kept as a module constant so the c30c/c30e revert-pins fail LOUD (count
# asserts below) if the fixed spelling ever drifts instead of silently
# pinning nothing.
_R6_PREFIX_GROUP = "(([A-Za-z_][A-Za-z0-9_]*[+]?=[^[:space:]]*|time([[:space:]]+-p)?)[[:space:]]+)*"
# The HISTORICAL r5 spelling (plain NAME=value atoms only) — used by the c30e
# revert-pin to reconstruct the r5 matcher textually (Step 10d rebase-merge
# rewrites branch SHAs, so no `git show` blob pins; same textual convention
# as c30c).
_R5_PREFIX_GROUP = "([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+)*"


def test_c30c_prefix_group_is_load_bearing_prefixed_dot_source_permitted_without_it(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    """Regression proof the r4 hole was REAL: strip the WHOLE (r6) prefix
    group out of a copy of the live guard — reconstructing the r3/r4
    CWD_MOVER_LEAD_ERE spelling byte-for-byte — and assert that copy PERMITS
    the assignment-prefixed dot-source crux (rc=0; the executed r4
    false-allow: origin/main=2 / r4 blob 22e76689e2=0) while the fixed guard
    BLOCKS the same command (rc=2, whole-index). A ``git show <r4-sha>:...``
    blob pin is NOT durable here: Step 10d lands via rebase-merge, which
    rewrites branch commit SHAs, so the pre-fix matcher is reconstructed
    textually instead and the count assert fails loud on any drift."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count(_R6_PREFIX_GROUP) == 1, "fixed CWD_MOVER_LEAD_ERE spelling drifted"
    r4_guard = tmp_path / "guard_r4_matcher.sh"
    r4_guard.write_text(text.replace(_R6_PREFIX_GROUP, "", 1), encoding="utf-8")
    r4_guard.chmod(0o755)
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _write(foreign_repo, "scripts/env.sh", "cd scripts\n")
    crux = f"cd {foreign_repo} && FOO=bar . scripts/env.sh && git commit -m x -- own.py"
    _assert_allowed(_run(crux, foreign_repo, cert, script=r4_guard, cwd=foreign_repo / "scripts"))
    _assert_blocked(_run(crux, foreign_repo, cert, cwd=foreign_repo / "scripts"))


def test_c30e_r6_prefix_atoms_load_bearing_r5_matcher_permits_both_cruxes(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    """Revert-pin proof BOTH r6 gaps were real: textually reconstruct the r5
    matcher (swap the r6 prefix group for the historical r5 spelling in a
    copy of the live guard) and assert that copy PERMITS the
    append-assignment crux AND the `time` wrapper crux (rc=0 — the executed
    r5 false-allows, plan #2371 §1) while the fixed guard BLOCKS both (rc=2,
    whole-index). Same textual-reconstruction convention as c30c (no
    `git show` blob pins — rebase-merge rewrites branch SHAs); the count
    asserts fail loud on any spelling drift."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count(_R6_PREFIX_GROUP) == 1, "fixed CWD_MOVER_LEAD_ERE spelling drifted"
    assert _R5_PREFIX_GROUP not in text, "historical r5 spelling resurfaced in the guard"
    r5_guard = tmp_path / "guard_r5_matcher.sh"
    r5_guard.write_text(text.replace(_R6_PREFIX_GROUP, _R5_PREFIX_GROUP, 1), encoding="utf-8")
    r5_guard.chmod(0o755)
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _write(foreign_repo, "scripts/env.sh", "cd scripts\n")
    for mover in ("FOO+=bar . scripts/env.sh", "time . scripts/env.sh"):
        crux = f"cd {foreign_repo} && {mover} && git commit -m x -- own.py"
        _assert_allowed(
            _run(crux, foreign_repo, cert, script=r5_guard, cwd=foreign_repo / "scripts")
        )
        _assert_blocked(_run(crux, foreign_repo, cert, cwd=foreign_repo / "scripts"))


def test_c30f_env_dot_source_not_reachable_stays_scoped(foreign_repo: Path, cert: Path) -> None:
    # Not-reachable pin (r6, #2371): external `env` cannot run the `.` shell
    # builtin, so `env . x` CANNOT source in the current shell (reachability
    # probe execution-confirmed 2026-08-18, GNU bash 5.1.16) — the accepted
    # non-recognition: the record matches neither the r6 lead grammar nor
    # the whole-record screen, the armed base survives, and the pathspec
    # still scopes past the staged uncertified gated file (rc=0). If the
    # grammar ever recognizes `env` as a wrapper atom this pin flips —
    # revisit the reachability probe first.
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _write(foreign_repo, "scripts/env.sh", "cd scripts\n")
    _assert_allowed(
        _run(
            f"cd {foreign_repo} && env . scripts/env.sh && git commit -m x -- own.py",
            foreign_repo,
            cert,
            cwd=foreign_repo / "scripts",
        )
    )


# --- c30g family (r7, #2371 revision round 2; concern
# disarm-fallback-drops-worktree-pathspecs): a recognized r6 disarm routes
# the record to the scope=0 whole-index fallback, which pre-r7 read STAGED
# names only and bound the STAGED blob — but a pathspec-form commit lands
# WORKTREE content, and a cwd-INDEPENDENT repo-top pathspec
# (`:/scripts/own.py`) lands it regardless of the unknown post-mover cwd.
# So an uncertified worktree edit rode either NEW r6 prefix family to a
# PERMIT where the r5 matcher (armed base intact => scoped worktree read)
# BLOCKED — the reconciler-executed block->permit flip (4/4 scenario
# cells). The r7 repair keys the fallback's worktree read AND the landing
# binding on pathspec-form evidence (commit_has_pathspec /
# pathspec_opaque), the conservative superset; every cell below BLOCKS on
# the fixed guard.
_R7_WORKTREE_EVIDENCE_COND = (
    '[ "$has_dash_a" = 1 ] || [ "$commit_has_pathspec" = 1 ] || [ "$pathspec_opaque" = 1 ]'
)
# Pre-r7 evidence key (-a only) — the c30g2 revert-pin swaps this back in at
# all three sites (fallback mod read, landing_sha, cert_diag) to reconstruct
# the pre-fix fallback textually (c30c/c30e convention: no `git show` blob
# pins — Step 10d rebase-merge rewrites branch SHAs; count asserts fail loud
# on spelling drift).
_PRE_R7_EVIDENCE_COND = '[ "$has_dash_a" = 1 ]'

_C30G_MOVERS = [
    pytest.param("FOO+=bar . scripts/env.sh", id="append-assign-dot-source"),
    pytest.param("time . scripts/env.sh", id="time-dot-source"),
]


def _committed_then_edited_repo(
    tmp_path: Path, name: str, *, staged_cert: Path | None = None
) -> Path:
    """Repo with scripts/own.py + scripts/env.sh COMMITTED (tracked), then
    own.py worktree-edited (uncertified). With ``staged_cert`` set, an
    intermediate staged blob is added and certified BEFORE the worktree edit
    (the certified-older-staged-blob variant: the cert is real but binds
    content the pathspec commit does NOT land)."""
    repo = _init_repo(tmp_path, name)
    _stage(repo, "scripts/own.py", "print(1)\n")
    _stage(repo, "scripts/env.sh", "cd scripts\n")
    _git(repo, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "-m", "init")
    if staged_cert is not None:
        _stage(repo, "scripts/own.py", "print(2)\n")
        _cert_line(staged_cert, "scripts/own.py", _staged_sha(repo, "scripts/own.py"))
    _write(repo, "scripts/own.py", "print(99)\n")  # uncertified worktree edit
    return repo


@pytest.mark.parametrize("mover", _C30G_MOVERS)
def test_c30g_disarmed_repo_top_pathspec_unstaged_worktree_edit_blocked(
    mover: str, tmp_path: Path, cert: Path
) -> None:
    # S1: nothing staged; the ONLY uncertified content is the worktree edit
    # the repo-top pathspec commit lands. Pre-r7 the disarmed fallback saw an
    # empty staged set and permitted at the artifact-only early-allow.
    repo = _committed_then_edited_repo(tmp_path, "wtland_s1")
    _assert_blocked(
        _run(
            f"cd {repo} && {mover} && git commit -m x -- :/scripts/own.py",
            repo,
            cert,
            cwd=repo / "scripts",
        )
    )


@pytest.mark.parametrize("mover", _C30G_MOVERS)
def test_c30g_disarmed_repo_top_pathspec_certified_older_staged_blob_blocked(
    mover: str, tmp_path: Path, cert: Path
) -> None:
    # S2: an OLDER staged blob is certified; the pathspec commit lands the
    # fresher UNCERTIFIED worktree edit. Pre-r7 the disarmed fallback bound
    # the staged blob and validated the wrong content (permit).
    repo = _committed_then_edited_repo(tmp_path, "wtland_s2", staged_cert=cert)
    _assert_blocked(
        _run(
            f"cd {repo} && {mover} && git commit -m x -- :/scripts/own.py",
            repo,
            cert,
            cwd=repo / "scripts",
        )
    )


def test_c30g2_pre_r7_fallback_permits_worktree_landing_fixed_guard_blocks(
    tmp_path: Path,
) -> None:
    """Revert-pin proof the r7 repair is load-bearing: swap the r7 evidence
    key back to the pre-r7 ``-a``-only form at ALL THREE sites (fallback mod
    read, landing_sha, cert_diag) and assert that copy PERMITS all four
    scenario cells (both new prefix families x {unstaged edit,
    certified-older-staged-blob} — rc=0, the reconciler-executed
    block->permit flip) while the fixed guard BLOCKS every cell (rc=2)."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count(_R7_WORKTREE_EVIDENCE_COND) == 3, "r7 evidence condition drifted"
    pre_guard = tmp_path / "guard_pre_r7.sh"
    pre_guard.write_text(
        text.replace(_R7_WORKTREE_EVIDENCE_COND, _PRE_R7_EVIDENCE_COND), encoding="utf-8"
    )
    pre_guard.chmod(0o755)
    for i, mover in enumerate(("FOO+=bar . scripts/env.sh", "time . scripts/env.sh")):
        for j, with_staged_cert in enumerate((False, True)):
            case_cert = tmp_path / f"cert_{i}_{j}.txt"
            repo = _committed_then_edited_repo(
                tmp_path,
                f"wtland_pre_{i}_{j}",
                staged_cert=case_cert if with_staged_cert else None,
            )
            crux = f"cd {repo} && {mover} && git commit -m x -- :/scripts/own.py"
            _assert_allowed(_run(crux, repo, case_cert, script=pre_guard, cwd=repo / "scripts"))
            _assert_blocked(_run(crux, repo, case_cert, cwd=repo / "scripts"))


# --- r8 (#2371 r3): heterogeneous bare + pathspec/-a records ----------------
# Command-global evidence flags mean a BARE commit clause chained with a
# pathspec/-a clause can land TWO blobs for one path (the staged blob via the
# bare clause, worktree content via the sibling clause's evidence-keyed
# binding). r8 requires BOTH blobs certified on such records
# (landing_certified). The revert-pin swaps the bare-clause gate to a
# never-true value at BOTH sites (landing_certified + the cert_diag mirror)
# to reconstruct the pre-r8 (r2) guard textually (c30c/c30g2 convention: no
# `git show` blob pins; count asserts fail loud on spelling drift).
_R8_HETERO_GATE = '[ "$commit_bare_clause" = 1 ]'
_R8_HETERO_GATE_NEUTERED = '[ "$commit_bare_clause" = 2371 ]'

_HETERO_TWO_CLAUSE = "git commit -m a && git commit -m b -- tasks/t.md"


def _hetero_repo(
    tmp_path: Path,
    name: str,
    cert: Path,
    *,
    cert_staged: bool = False,
    delete_worktree: bool = False,
) -> Path:
    """Repo with scripts/own.py committed print(1), then STAGED print(2)
    (certified only with ``cert_staged``), then a worktree edit print(3)
    whose hash IS certified (or the worktree file deleted with
    ``delete_worktree``). tasks/t.md provides the artifact-only pathspec for
    the sibling commit clause."""
    repo = _init_repo(tmp_path, name)
    _stage(repo, "scripts/own.py", "print(1)\n")
    _stage(repo, "tasks/t.md", "note\n")
    _git(repo, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "-m", "init")
    _stage(repo, "scripts/own.py", "print(2)\n")
    if cert_staged:
        _cert_line(cert, "scripts/own.py", _staged_sha(repo, "scripts/own.py"))
    if delete_worktree:
        (repo / "scripts/own.py").unlink()
    else:
        _write(repo, "scripts/own.py", "print(3)\n")
        _cert_line(cert, "scripts/own.py", _worktree_sha(repo, "scripts/own.py"))
    return repo


def test_c30h_bare_plus_pathspec_clause_uncertified_staged_blob_blocked(
    tmp_path: Path, cert: Path
) -> None:
    # The round-2 reconciler's executed P1 cell: command-global pathspec
    # evidence bound the CERTIFIED worktree hash while the BARE clause lands
    # the different, UNCERTIFIED staged blob. The fixed guard blocks; the
    # bare clause alone blocks too (staged binding — unchanged control
    # isolating the sibling clause as the former flip).
    repo = _hetero_repo(tmp_path, "hetero_s1", cert)
    _assert_blocked(_run(_HETERO_TWO_CLAUSE, repo, cert))
    _assert_blocked(_run("git commit -m a", repo, cert))


def test_c30h_bare_plus_pathspec_both_blobs_certified_allowed(tmp_path: Path, cert: Path) -> None:
    # Require-BOTH, not hard-block: certifying the staged AND worktree blobs
    # permits the heterogeneous record (bounds the over-tightening; also an
    # allow-direction pin for the incident chains' commit shapes).
    repo = _hetero_repo(tmp_path, "hetero_s2", cert, cert_staged=True)
    _assert_allowed(_run(_HETERO_TWO_CLAUSE, repo, cert))


def test_c30h_bare_plus_dash_a_uncertified_staged_blob_blocked(tmp_path: Path, cert: Path) -> None:
    # The -a analogue of the same record class (named in the reconcile fix
    # sketch: "a bare clause coexists with pathspec/-a evidence"): bare &&
    # -a with divergent blobs requires both certs too (block direction over
    # the pre-existing trunk permit).
    repo = _hetero_repo(tmp_path, "hetero_s3", cert)
    _assert_blocked(_run("git commit -m a && git commit -a -m b", repo, cert))


def test_c30h_worktree_deleted_bare_clause_staged_blob_blocked(tmp_path: Path, cert: Path) -> None:
    # Deletion variant of the same flip: the worktree leg is exempt (file
    # gone), but the bare clause still lands the uncertified STAGED blob —
    # landing_certified must block rather than fall through the
    # deletion exemption.
    repo = _hetero_repo(tmp_path, "hetero_s4", cert, delete_worktree=True)
    _assert_blocked(_run(_HETERO_TWO_CLAUSE, repo, cert))


def test_c30h2_pre_r8_guard_permits_hetero_cell_fixed_guard_blocks(tmp_path: Path) -> None:
    """Revert-pin proof the r8 dual requirement is load-bearing: neuter the
    bare-clause gate at BOTH sites (landing_certified + the cert_diag
    mirror) and assert that copy PERMITS the heterogeneous cell (rc=0 — the
    reconciler-executed block->permit flip on the r2 guard) while the fixed
    guard BLOCKS it (rc=2)."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count(_R8_HETERO_GATE) == 2, "r8 bare-clause gate drifted"
    pre_guard = tmp_path / "guard_pre_r8.sh"
    pre_guard.write_text(text.replace(_R8_HETERO_GATE, _R8_HETERO_GATE_NEUTERED), encoding="utf-8")
    pre_guard.chmod(0o755)
    case_cert = tmp_path / "cert_h2.txt"
    case_cert.write_text("", encoding="utf-8")
    repo = _hetero_repo(tmp_path, "hetero_pre", case_cert)
    _assert_allowed(_run(_HETERO_TWO_CLAUSE, repo, case_cert, script=pre_guard))
    _assert_blocked(_run(_HETERO_TWO_CLAUSE, repo, case_cert))


def test_c30i_pathspec_from_file_spellings_match_residual_note(tmp_path: Path, cert: Path) -> None:
    # Residual-note truth pins (flag-pathspec-residual-note-misstates-parser):
    # the SEPARATE-ARGUMENT spelling's filename falls through as a positional
    # candidate, arming the widened worktree read — over-tightening, block
    # direction — while the EQUALS form carries no positional candidate and
    # stays staged-only (nothing staged => artifact-only allow). Pins the
    # exact behavior the rewritten fallback residual note documents.
    repo = _init_repo(tmp_path, "flagform")
    _stage(repo, "scripts/own.py", "print(1)\n")
    _stage(repo, "tasks/t.md", "note\n")
    _git(repo, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "-m", "init")
    _write(repo, "scripts/own.py", "print(99)\n")  # worktree-only, uncertified
    (repo / "list.txt").write_text("tasks/t.md\n", encoding="utf-8")
    _assert_blocked(_run("git commit --pathspec-from-file list.txt -m x", repo, cert))
    _assert_allowed(_run("git commit --pathspec-from-file=list.txt -m x", repo, cert))


def test_c32b_assignment_prefixed_mover_before_arming_cd_still_scopes(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # Allow pin (over-tightening protection, c32 sibling): the r5-widened
    # disarm stays ARMED-ONLY — an assignment-prefixed dot-source BEFORE the
    # arming canonical cd never disarms, and the pathspec still scopes past
    # the foreign uncertified staged file.
    _write(foreign_repo, "scripts/env.sh", "cd scripts\n")
    _assert_allowed(
        _run(
            f"FOO=bar . scripts/env.sh; cd {foreign_repo} && git commit -m x -- tasks/t.md",
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


def test_c32c_time_wrapped_mover_before_arming_cd_still_scopes(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # Allow pin (over-tightening protection, c32/c32b sibling — r6 #2371):
    # the r6-widened disarm stays ARMED-ONLY — a time-wrapped dot-source
    # BEFORE the arming canonical cd never disarms (and never poisons
    # arming), and the pathspec still scopes past the foreign uncertified
    # staged file.
    _write(foreign_repo, "scripts/env.sh", "cd scripts\n")
    _assert_allowed(
        _run(
            f"time . scripts/env.sh; cd {foreign_repo} && git commit -m x -- tasks/t.md",
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


def test_c31_short_dashc_root_wrapper_on_commit_refuses_scoping(
    foreign_repo: Path, cert: Path
) -> None:
    # r4: the SHORT pre-verb -C dir wrapper on the commit clause sets
    # scope_unsafe (parity with the long --chdir form, c14b). The
    # canonical-root spelling is the one -C form whose waiver is REFUSED
    # (b7), so it reaches the commit scan; pre-fix it scoped root-pinned
    # (rc differential: origin/main=0, r3=0 — a PRE-EXISTING scoped allow —
    # r4=2, strictly tighter).
    _assert_blocked(_run(f"git -C {_CANONICAL_ROOT} commit -m x -- tasks/t.md", foreign_repo, cert))


def test_c31b_short_dashc_root_wrapper_riding_armed_chain_blocked(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # The armed-chain composition of c31 — the family-(e) instance the
    # widening INTRODUCED (rc differential: origin/main=2, r3=0, r4=2):
    # both the r4 mover disarm (the -C retarget vocabulary in the raw
    # record) and the pre-verb scope_unsafe arm force the whole-index read.
    _assert_blocked(
        _run(
            f"cd {foreign_repo} && git -C {_CANONICAL_ROOT} commit -m x -- tasks/t.md",
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


def test_c32_mover_before_arming_cd_still_scopes(
    foreign_repo: Path, cert: Path, tmp_path: Path
) -> None:
    # Allow pin (over-tightening protection for the r4 mover disarm): the
    # disarm is ARMED-ONLY, so a mover BEFORE the arming canonical cd never
    # disarms — the SEQ-separated canonical cd establishes the base and the
    # pathspec still scopes past the foreign uncertified staged file.
    _assert_allowed(
        _run(
            f"pushd scripts; cd {foreign_repo} && git commit -m x -- tasks/t.md",
            foreign_repo,
            cert,
            cwd=tmp_path,
        )
    )


# ---------------------------------------------------------------------------
# D — cd-latch variable resolution + unproven-cd diagnostics (issue #1676)
# ---------------------------------------------------------------------------
# Incident #1644 (2026-07-24): a `cd "$WT" && pytest ... && git commit ...`
# compound was blocked whole — the read-only pytest leg included — and the
# block message never named the cause: the cd-latch arms only on provably
# non-root LITERAL targets, so a variable target was always unproven. Fix
# (plan #1676, ``tasks/*/1676/plans/plan.md``): (a) a variable-resolution arm
# latches the ``NAME=<literal> && cd "$NAME" && ...`` shape ONLY when all
# seven certification gates hold (compound-context refusal / whole-clause
# assignment anchor / exactly-one-preceding / unconditional position /
# mutation belt / path-sane RHS / suffix sanity), feeding the resolved string
# to the SAME verdict pattern list as the literal arm; (b) every
# still-unproven cd target is named in the block message via stable
# ``cd-diag: unproven-cd target=<tgt> reason=<token>`` lines plus a
# remediation paragraph. Case ids A17-A19 / B20-B34 mirror the hook's
# embedded self-test battery; the stderr reason-token asserts here cover what
# the exit-code-only ``run_case`` harness cannot.

_WT_ASSIGN = "WT=.claude/worktrees/issue-9"


def test_a17_var_assign_cd_latch_allowed(code_repo: Path, cert: Path) -> None:
    _assert_allowed(_run(f'{_WT_ASSIGN} && cd "$WT" && git commit -m x', code_repo, cert))


def test_a18_seq_quoted_assignment_braced_var_allowed(code_repo: Path, cert: Path) -> None:
    cmd = 'WT="/abs/.claude/worktrees/issue-9"; cd "${WT}" && git commit -m x'
    _assert_allowed(_run(cmd, code_repo, cert))


def test_a19_var_with_literal_suffix_allowed(code_repo: Path, cert: Path) -> None:
    cmd = 'WT=/abs/.claude/worktrees && cd "$WT/issue-9" && git commit -m x'
    _assert_allowed(_run(cmd, code_repo, cert))


def test_b20_unresolved_var_cd_blocks_and_names_cause(code_repo: Path, cert: Path) -> None:
    """The #1644 shape: no same-command assignment -> still blocks, and the
    block message now names the unprovable cd target + the remediations."""
    r = _run('cd "$WT" && git commit -m x', code_repo, cert)
    _assert_blocked(r)  # rc == 2 + "BLOCKED" in stderr
    assert "cd-diag:" in r.stderr, r.stderr
    assert "reason=no-assignment" in r.stderr, r.stderr
    assert "UNPROVEN-CD?" in r.stderr, r.stderr
    assert 'git -C "$WT"' in r.stderr, r.stderr  # the worktree remediation


def test_b21_two_assignments_refused_multiple(code_repo: Path, cert: Path) -> None:
    r = _run(f'{_WT_ASSIGN}; WT=$REPO; cd "$WT" && git commit -m x', code_repo, cert)
    _assert_blocked(r)
    assert "reason=multiple-assignments" in r.stderr, r.stderr


def test_b22_root_path_rhs_never_latches(code_repo: Path, cert: Path) -> None:
    """A root-spelling RHS resolves to verdict `root` via the SAME shared
    pattern list — a crafted assignment can never latch a root commit.
    Re-keyed for #2046: cd_latch_verdict compares against ``$GUARD_REPO``,
    so the tested root spelling is the GUARDED (fixture) root."""
    _assert_blocked(_run(f'WT={code_repo} && cd "$WT" && git commit -m x', code_repo, cert))


def test_b22b_non_guard_absolute_rhs_still_latches(code_repo: Path, cert: Path) -> None:
    """Companion to the #2046 re-key: an absolute RHS that is NOT the guarded
    root resolves and LATCHES the commit away. The fixture has a gated file
    STAGED, so the latch — not an empty index — is what carries the allow."""
    _assert_allowed(_run('WT=/abs/other-repo && cd "$WT" && git commit -m x', code_repo, cert))


def test_b23_dynamic_rhs_command_substitution_refused(code_repo: Path, cert: Path) -> None:
    r = _run('WT=$(mktemp) && cd "$WT" && git commit -m x', code_repo, cert)
    _assert_blocked(r)
    assert "reason=dynamic-rhs" in r.stderr, r.stderr


def test_b23b_dynamic_rhs_with_args_still_blocks(code_repo: Path, cert: Path) -> None:
    # A space-bearing substitution fails the whole-clause anchor instead of
    # the path-sane gate — same fail-closed disposition, different token.
    _assert_blocked(_run('WT=$(mktemp -d) && cd "$WT" && git commit -m x', code_repo, cert))


def test_b24_conditional_assignment_refused(code_repo: Path, cert: Path) -> None:
    r = _run(f'true && {_WT_ASSIGN}; cd "$WT" && git commit -m x', code_repo, cert)
    _assert_blocked(r)
    assert "reason=conditional-assignment" in r.stderr, r.stderr


def test_b25_subshell_assignment_refused(code_repo: Path, cert: Path) -> None:
    _assert_blocked(_run(f'({_WT_ASSIGN}); cd "$WT" && git commit -m x', code_repo, cert))


def test_b26_backgrounded_assignment_refused(code_repo: Path, cert: Path) -> None:
    r = _run(f'{_WT_ASSIGN} & cd "$WT" && git commit -m x', code_repo, cert)
    _assert_blocked(r)
    assert "reason=backgrounded-assignment" in r.stderr, r.stderr


def test_b27_latch_persistence_seq_after_cd_resets(code_repo: Path, cert: Path) -> None:
    """Latch persistence semantics untouched (#1676 must-ask): a resolved
    latch still resets at any non-`&&` separator."""
    _assert_blocked(_run(f'{_WT_ASSIGN} && cd "$WT"; git commit -m x', code_repo, cert))


def test_b28_compound_body_assignment_refused(code_repo: Path, cert: Path) -> None:
    """Gate 7: the masker tracks quote/heredoc state only, so an assignment
    on an if-body interior line surfaces as its own NL-tagged record — any
    compound opener in the command refuses resolution outright."""
    r = _run(f'if true; then\n{_WT_ASSIGN}\nfi\ncd "$WT" && git commit -m x', code_repo, cert)
    _assert_blocked(r)
    assert "reason=compound-context" in r.stderr, r.stderr


def test_b29_function_body_assignment_refused(code_repo: Path, cert: Path) -> None:
    r = _run(f'f() {{\n{_WT_ASSIGN}\n}}\nf; cd "$WT" && git commit -m x', code_repo, cert)
    _assert_blocked(r)
    assert "reason=compound-context" in r.stderr, r.stderr


def test_b30_suffix_with_parent_dir_segment_refused(code_repo: Path, cert: Path) -> None:
    cmd = 'WT=/abs/.claude/worktrees && cd "$WT/issue-9/../.." && git commit -m x'
    _assert_blocked(_run(cmd, code_repo, cert))


def test_b31_mutated_between_assignment_and_cd_refused(code_repo: Path, cert: Path) -> None:
    r = _run(f'{_WT_ASSIGN}; unset WT; cd "$WT" && git commit -m x', code_repo, cert)
    _assert_blocked(r)
    assert "reason=mutation-belt" in r.stderr, r.stderr


def test_b32_env_prefix_assignment_refused(code_repo: Path, cert: Path) -> None:
    r = _run(f'{_WT_ASSIGN} true; cd "$WT" && git commit -m x', code_repo, cert)
    _assert_blocked(r)
    assert "reason=no-assignment" in r.stderr, r.stderr


def test_b33_assignment_after_cd_blocks(code_repo: Path, cert: Path) -> None:
    _assert_blocked(_run(f'cd "$WT" && git commit -m x; {_WT_ASSIGN}', code_repo, cert))


def test_b34_pipelined_assignment_refused(code_repo: Path, cert: Path) -> None:
    r = _run(f'{_WT_ASSIGN} | true; cd "$WT" && git commit -m x', code_repo, cert)
    _assert_blocked(r)
    assert "reason=pipelined-assignment" in r.stderr, r.stderr


# ---------------------------------------------------------------------------
# rd-group (issue #1928; plan pins r1-r16 map to rd1-rd16 — the `rd` prefix
# avoids the pre-existing #1857 rehash test_r1..test_r4 name family):
# strictly-recognized redirect tokens on a commit
# clause engage pathspec scoping exactly like their redirect-free twins
# (r1-r5); every ambiguity keeps the opaque -> whole-index -> block fallback
# (r6-r13), and the three excluded token families — process substitution,
# here-doc / here-string operators, non-clean-literal attached targets —
# genuinely refuse (r14-r16).


def test_rd1_add_then_pathspec_commit_with_redirect_and_chain_allowed(
    foreign_repo: Path, cert: Path
) -> None:
    """Incident shape R-F2: non-gated add && pathspec commit + redirect +
    fd-dup, then a `;`-chained echo/tail pair."""
    cmd = (
        "git add tasks/t.md && git commit -m x -- tasks/t.md "
        "> /tmp/i1928_commit.log 2>&1; echo done; tail -2 /tmp/i1928_commit.log"
    )
    _assert_allowed(_run(cmd, foreign_repo, cert))


def test_rd2_pathspec_commit_with_fd_dup_and_pipe_allowed(foreign_repo: Path, cert: Path) -> None:
    """Incident shape R-F3: pathspec commit + `2>&1` piped into a read-only
    consumer clause."""
    _assert_allowed(_run("git commit -m x -- tasks/t.md 2>&1 | tail -5", foreign_repo, cert))


def test_rd3_multi_pathspec_detached_redirect_newline_chain_allowed(
    foreign_repo: Path, cert: Path
) -> None:
    """Incident shape R-F5: multi-pathspec commit + detached redirect +
    fd-dup, newline-chained echo/tail."""
    cmd = (
        "git commit -m x -- tasks/t.md docs/b.md > /tmp/i1928_commit.log 2>&1\n"
        "echo committed\ntail -3 /tmp/i1928_commit.log"
    )
    _assert_allowed(_run(cmd, foreign_repo, cert))


def test_rd4_attached_target_redirect_allowed(foreign_repo: Path, cert: Path) -> None:
    """Attached-target form: operator + clean-literal target as ONE token."""
    _assert_allowed(
        _run("git commit -m x -- tasks/t.md >/tmp/i1928_commit.log", foreign_repo, cert)
    )


def test_rd5_certified_own_gated_payload_with_redirect_allowed(
    foreign_repo: Path, cert: Path
) -> None:
    """c6 analogue: CERTIFIED own gated payload pathspec commit + redirect is
    allowed despite the foreign uncertified staged file."""
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _cert_line(cert, "scripts/own.py", _worktree_sha(foreign_repo, "scripts/own.py"))
    cmd = "git commit -m x -- scripts/own.py > /tmp/i1928_commit.log 2>&1"
    _assert_allowed(_run(cmd, foreign_repo, cert))


def test_rd6_bare_commit_with_redirect_blocks(foreign_repo: Path, cert: Path) -> None:
    """Danger control: a bare commit sweeps the whole staged index regardless
    of the redirect (commit_bare_clause path)."""
    _assert_blocked(_run("git commit -m x > /tmp/i1928_commit.log 2>&1", foreign_repo, cert))


def test_rd7_pathspec_naming_foreign_gated_with_redirect_blocks(
    foreign_repo: Path, cert: Path
) -> None:
    _assert_blocked(
        _run("git commit -m x -- scripts/foreign.py > /tmp/i1928_commit.log", foreign_repo, cert)
    )


def test_rd8_variable_pathspec_token_with_redirect_blocks(foreign_repo: Path, cert: Path) -> None:
    """MF-2 unchanged: a `$`-bearing pathspec token stays opaque even when a
    recognized redirect rides the same clause."""
    _assert_blocked(_run("git commit -m x -- $SPEC > /tmp/i1928_commit.log", foreign_repo, cert))


def test_rd9_quoted_spacey_pathspec_with_redirect_blocks(foreign_repo: Path, cert: Path) -> None:
    """Accepted fail-closed residual: a masked (quoted) pathspec + redirect
    fails rawtail token-count parity and stays opaque."""
    _assert_blocked(
        _run('git commit -m x -- "tasks/my file.md" > /tmp/i1928_commit.log', foreign_repo, cert)
    )


def test_rd10_include_flag_with_redirect_blocks(foreign_repo: Path, cert: Path) -> None:
    """scope_unsafe unchanged: `--include`-class flags still disable scoping."""
    _assert_blocked(
        _run("git commit --include -m x -- tasks/t.md > /tmp/i1928_commit.log", foreign_repo, cert)
    )


def test_rd11_subdir_cwd_with_redirect_blocks(foreign_repo: Path, cert: Path) -> None:
    """cwd gate unchanged: a non-root hook cwd never scopes."""
    cmd = "git commit -m x -- t.md > /tmp/i1928_commit.log"
    _assert_blocked(_run(cmd, foreign_repo, cert, cwd=foreign_repo / "tasks"))


def test_rd12_in_command_cd_with_redirect_blocks(foreign_repo: Path, cert: Path) -> None:
    """cd_nonroot unchanged: an in-command relative cd never scopes."""
    _assert_blocked(
        _run("cd tasks && git commit -m x -- t.md > /tmp/i1928_commit.log", foreign_repo, cert)
    )


def test_rd13_malformed_redirect_shaped_positional_stays_opaque_blocks(
    foreign_repo: Path, cert: Path
) -> None:
    """A word-attached `>`-bearing token matches NO redirect form: grammar
    fallback -> classify_candidate -> opaque -> whole-index -> block."""
    _assert_blocked(_run("git commit -m x -- tasks/t.md out>>result>x", foreign_repo, cert))


@pytest.mark.parametrize("tok", [">(cat)", "<(cat)"])
def test_rd14_process_substitution_form_token_refused(
    tok: str, foreign_repo: Path, cert: Path
) -> None:
    """Must-Fix (i): process-substitution-form tokens are NOT redirects — the
    grammar classifies them `no` -> opaque -> whole-index -> block."""
    _assert_blocked(_run(f"git commit -m x -- tasks/t.md {tok}", foreign_repo, cert))


@pytest.mark.parametrize("tok", ["<< EOF", "<<-EOF", "<<< data", "<<<data"])
def test_rd15_heredoc_herestring_opener_intercepted_allowed(
    tok: str, foreign_repo: Path, cert: Path
) -> None:
    """Re-keyed for #2046 (was: refused -> opaque -> block): opener tokens
    are now intercepted by ``heredoc_tok_kind`` BEFORE candidate
    classification — exactly the incident class — so the excluding pathspec
    scopes. ``redirect_tok_kind`` itself still refuses the whole family;
    test_rd15b pins that grammar contract directly."""
    _assert_allowed(_run(f"git commit -m x -- tasks/t.md {tok}", foreign_repo, cert))


def _classifier_kind(fn_name: str, tok: str) -> str:
    """Run one of the hook's pure token-classifier functions in isolation
    (sed-extracted function definition + the FILL global), pinning its
    per-token grammar without driving the whole hook."""
    body = subprocess.run(
        ["sed", "-n", f"/^{fn_name}() {{$/,/^}}$/p", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert body.startswith(f"{fn_name}()"), f"extraction failed: {body[:80]!r}"
    script = "FILL=$'\\001'\n" + body + f'\n{fn_name} "$1"\n'
    r = subprocess.run(["bash", "-c", script, "_", tok], capture_output=True, text=True, check=True)
    return r.stdout.strip()


@pytest.mark.parametrize("tok", ["<<", "<<-", "<<<", "<<EOF", "<<-EOF", "<<<data"])
def test_rd15b_redirect_grammar_still_refuses_heredoc_family(tok: str) -> None:
    """The #1928 redirect grammar's heredoc-family -> `no` contract stays
    byte-pinned: #2046 moved the opener handling UPSTREAM (heredoc_tok_kind);
    redirect_tok_kind must keep refusing the family so a grammar refactor can
    never mis-consume an opener as a redirect."""
    assert _classifier_kind("redirect_tok_kind", tok) == "no"


@pytest.mark.parametrize("tok", [">$LOGFILE", "2>$(mktemp)"])
def test_rd16_redirect_with_non_clean_literal_attached_target_refused(
    tok: str, foreign_repo: Path, cert: Path
) -> None:
    """Must-Fix (iii): an operator whose ATTACHED target carries `$` or a
    command-substitution form fails the clean-literal test — `no` -> opaque
    -> whole-index -> block."""
    _assert_blocked(_run(f"git commit -m x -- tasks/t.md {tok}", foreign_repo, cert))


# ---------------------------------------------------------------------------
# f-group (issue #1949): the `-F <msgfile>` / `--file=<msgfile>` message-file
# commit form + the exact rc-capture compound suffixes from the 2026-07-31
# incident (two sessions' pathspec-limited root commits were blocked by a
# FOREIGN session's staged uncertified file). The behavioral root cause was
# fixed by #1928 (`305df9ad14`); these tests PIN the previously-uncovered `-F`
# dimension of the commit-clause flag table (the `-m | -F | ... skip_next`
# separate-word arm, the `--*=*` attached-arg arm, and the single-dash cluster
# arm), so a future flag-table refactor cannot silently regress the pathspec
# escape with no red test.


def _msgfile(tmp_path: Path) -> Path:
    """Commit-message file for the `-F` / `--file=` form. The hook parses
    only the argv shape — the file content is never read by the guard."""
    p = tmp_path / "commitmsg.txt"
    p.write_text("task #9: fix\n", encoding="utf-8")
    return p


def test_f1_msgfile_pathspec_certified_own_payload_allowed(
    tmp_path: Path, foreign_repo: Path, cert: Path
) -> None:
    """c6 analogue for `-F`: a certified own gated payload committed via
    `git commit -F <msgfile> -- <own>` is allowed despite the foreign
    uncertified staged file (#1949)."""
    msg = _msgfile(tmp_path)
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _cert_line(cert, "scripts/own.py", _worktree_sha(foreign_repo, "scripts/own.py"))
    _assert_allowed(_run(f"git commit -F {msg} -- scripts/own.py", foreign_repo, cert))


def test_f2_msgfile_artifact_pathspec_allowed(
    tmp_path: Path, foreign_repo: Path, cert: Path
) -> None:
    """c1 analogue for `-F`: an artifact pathspec commit with a message FILE
    scopes (the separate-word `-F` arm consumes the msgfile path token)."""
    msg = _msgfile(tmp_path)
    _assert_allowed(_run(f"git commit -F {msg} -- tasks/t.md", foreign_repo, cert))


def test_f3_msgfile_pathspec_redirect_rc_capture_allowed(
    tmp_path: Path, foreign_repo: Path, cert: Path
) -> None:
    """The exact 2026-07-31 boundary-impl incident shape (#1949): certified
    own pathspec + `-F <msgfile>` + redirect + a `; COMMIT_RC=$?` rc-capture
    clause (the `$?` lives in a SEPARATE clause, never the commit clause)."""
    msg = _msgfile(tmp_path)
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _cert_line(cert, "scripts/own.py", _worktree_sha(foreign_repo, "scripts/own.py"))
    cmd = f"git commit -F {msg} -- scripts/own.py > /tmp/i1949_commit.log 2>&1; COMMIT_RC=$?"
    _assert_allowed(_run(cmd, foreign_repo, cert))


def test_f4_dash_m_pathspec_redirect_rc_echo_allowed(foreign_repo: Path, cert: Path) -> None:
    """The exact 2026-07-31 orchestrator incident shape (#1949): `-m` +
    certified own pathspec + redirect + a `; echo rc=$?` suffix clause."""
    _stage(foreign_repo, "scripts/own.py", "print(1)\n")
    _cert_line(cert, "scripts/own.py", _worktree_sha(foreign_repo, "scripts/own.py"))
    cmd = "git commit -m x -- scripts/own.py > /tmp/i1949_commit.log 2>&1; echo rc=$?"
    _assert_allowed(_run(cmd, foreign_repo, cert))


def test_f5_bare_msgfile_commit_blocks(tmp_path: Path, foreign_repo: Path, cert: Path) -> None:
    """Sweep protection for `-F` (B39's message-file twin): a bare
    whole-index `-F` commit still blocks on the foreign uncertified file."""
    msg = _msgfile(tmp_path)
    _assert_blocked(_run(f"git commit -F {msg}", foreign_repo, cert))


def test_f6_msgfile_pathspec_naming_foreign_gated_blocks(
    tmp_path: Path, foreign_repo: Path, cert: Path
) -> None:
    """A `-F` pathspec commit NAMING the foreign uncertified gated file
    blocks (rd7's message-file twin)."""
    msg = _msgfile(tmp_path)
    _assert_blocked(_run(f"git commit -F {msg} -- scripts/foreign.py", foreign_repo, cert))


def test_f7_attached_file_eq_spelling_artifact_pathspec_allowed(
    tmp_path: Path, foreign_repo: Path, cert: Path
) -> None:
    """`--file=<msgfile>` attached spelling: pins the `--*=*`
    no-separate-arg arm (the msgfile rides inside ONE token, nothing is
    consumed), so the artifact pathspec still scopes."""
    msg = _msgfile(tmp_path)
    _assert_allowed(_run(f"git commit --file={msg} -- tasks/t.md", foreign_repo, cert))


def test_f8_single_dash_cluster_ending_in_arg_letter_allowed(
    foreign_repo: Path, cert: Path
) -> None:
    """Adjacent flag-table arm (plan deviation clause, critic advisory): a
    single-dash CLUSTER ending in an arg-taking letter (`-qm x`) consumes its
    separate message word, so the artifact pathspec still scopes (#1949)."""
    _assert_allowed(_run("git commit -qm x -- tasks/t.md", foreign_repo, cert))


# ---------------------------------------------------------------------------
# hd-group (issue #2046): here-doc / here-string OPENER tokens on the commit
# clause are shell syntax, never pathspecs — a strictly-recognized opener is
# excluded from the candidate stream (heredoc_tok_kind, the #1928 redirect
# argument), so a heredoc-fed `-F /dev/stdin` commit with an EXCLUDING
# pathspec scopes exactly like its heredoc-free twin. Controls pin the
# covering-pathspec and bare-commit block directions; every ambiguity keeps
# the opaque -> whole-index -> block fallback (fail closed).


def test_hd1_incident_composite_allowed(foreign_repo: Path, cert: Path) -> None:
    """The exact #2046 incident shape: cd-to-root prefix + `-F /dev/stdin`
    + pathspec EXCLUDING the foreign uncertified gated staged file + output
    redirect + heredoc + trailing commands."""
    cmd = (
        f"cd {foreign_repo}\n"
        "git commit -F /dev/stdin -- tasks/t.md docs/b.md > /tmp/i2046_commit.log 2>&1 <<'MSG'\n"
        "docs: fold interim notes\n"
        "\n"
        "Body paragraph with && separators; and a git commit -m mention.\n"
        "MSG\n"
        'echo "commit rc=$?"; git log -1 --oneline -- tasks/t.md'
    )
    _assert_allowed(_run(cmd, foreign_repo, cert))


def test_hd2_minimal_heredoc_excluding_pathspec_allowed(foreign_repo: Path, cert: Path) -> None:
    """Minimal defeater shape: `-F /dev/stdin` + heredoc + excluding pathspec."""
    cmd = "git commit -F /dev/stdin -- tasks/t.md <<'MSG'\nmsg body\nMSG"
    _assert_allowed(_run(cmd, foreign_repo, cert))


def test_hd3_cd_root_prefix_dash_m_excluding_pathspec_allowed(
    foreign_repo: Path, cert: Path
) -> None:
    """cd-to-the-guarded-root prefix keeps scoping (#2046 L2: the cd sites
    compare against $GUARD_REPO, so hermetic repos cover this shape)."""
    _assert_allowed(_run(f"cd {foreign_repo}\ngit commit -m x -- tasks/t.md", foreign_repo, cert))


def test_hd4_heredoc_covering_pathspec_blocks(foreign_repo: Path, cert: Path) -> None:
    """Control: the heredoc never widens the allow — a pathspec COVERING the
    uncertified gated file still blocks."""
    cmd = "git commit -F /dev/stdin -- scripts/foreign.py <<'MSG'\nmsg body\nMSG"
    _assert_blocked(_run(cmd, foreign_repo, cert))


def test_hd5_bare_commit_heredoc_blocks(foreign_repo: Path, cert: Path) -> None:
    """Control (sweep protection): a bare `-F /dev/stdin` commit sweeps the
    whole staged index regardless of the heredoc."""
    cmd = "git commit -F /dev/stdin <<'MSG'\nmsg body\nMSG"
    _assert_blocked(_run(cmd, foreign_repo, cert))


def test_hd6_separated_operator_consumes_delimiter_word_allowed(
    foreign_repo: Path, cert: Path
) -> None:
    """`pair` arm: a bare `<<` consumes its space-separated delimiter word,
    which must never classify as a pathspec candidate."""
    cmd = "git commit -F /dev/stdin -- tasks/t.md << MSG\nmsg body\nMSG"
    _assert_allowed(_run(cmd, foreign_repo, cert))


def test_hd7_tab_indented_heredoc_allowed(foreign_repo: Path, cert: Path) -> None:
    """`<<-DELIM` attached spelling (tab-stripping form) is `self`."""
    cmd = "git commit -F /dev/stdin -- tasks/t.md <<-MSG\n\tmsg body\n\tMSG"
    _assert_allowed(_run(cmd, foreign_repo, cert))


def test_hd8_unquoted_attached_delimiter_allowed(foreign_repo: Path, cert: Path) -> None:
    """`<<DELIM` attached unquoted spelling is `self`."""
    cmd = "git commit -F /dev/stdin -- tasks/t.md <<MSG\nmsg body\nMSG"
    _assert_allowed(_run(cmd, foreign_repo, cert))


def test_hd9_positional_arm_heredoc_no_ddash_allowed(foreign_repo: Path, cert: Path) -> None:
    """The positional (no `--`) arm intercepts openers too."""
    cmd = "git commit -F /dev/stdin tasks/t.md <<'MSG'\nmsg body\nMSG"
    _assert_allowed(_run(cmd, foreign_repo, cert))


def test_hd10_spacey_quoted_delimiter_stays_opaque_blocks(foreign_repo: Path, cert: Path) -> None:
    """Fail closed: a SPACEY quoted delimiter word-splits into unterminated
    pieces neither classifier recognizes -> opaque -> whole-index -> block."""
    cmd = "git commit -F /dev/stdin -- tasks/t.md <<'MY DELIM'\nmsg body\nMY DELIM"
    _assert_blocked(_run(cmd, foreign_repo, cert))


def test_hd11_masked_pathspec_with_heredoc_stays_opaque_blocks(
    foreign_repo: Path, cert: Path
) -> None:
    """Accepted fail-closed residual (known-limitations header): a QUOTED
    pathspec + opener still fails rawtail token-count parity (the opener
    stays counted in $raw) and keeps the whole-index fallback."""
    cmd = "git commit -F /dev/stdin -- \"tasks/t.md\" <<'MSG'\nmsg body\nMSG"
    _assert_blocked(_run(cmd, foreign_repo, cert))


@pytest.mark.parametrize(
    ("tok", "kind"),
    [
        ("<<", "pair"),
        ("<<-", "pair"),
        ("<<<", "pair"),
        ("<<EOF", "self"),
        ("<<-EOF", "self"),
        ("<<'EOF'", "self"),
        ('<<"EOF"', "self"),
        ("<<-'EOF'", "self"),
        ("<<<data", "self"),
        ("<<<'\x01\x01\x01'", "self"),  # masked attached here-string word
        ("<<''", "no"),  # empty quoted delimiter
        ("<<'MY", "no"),  # split piece of a spacey quoted delimiter
        ("<<E$F", "no"),  # $-bearing delimiter shape: fail closed
        ("<<E<F", "no"),  # opener + attached input redirect: fail closed
        ("'\x01\x01'", "no"),  # masked string literal: no opener prefix
        ("tasks/t.md", "no"),
        ("<file", "no"),  # single input redirect is NOT an opener
        (">out", "no"),
    ],
)
def test_hd12_opener_grammar(tok: str, kind: str) -> None:
    """Direct grammar pin for heredoc_tok_kind: strict `self`/`pair`
    recognition, `no` fail-closed default (#2046 L1)."""
    assert _classifier_kind("heredoc_tok_kind", tok) == kind


# ---------------------------------------------------------------------------
# #2013: every BLOCK path states the commit did NOT land (shared preamble).
# ---------------------------------------------------------------------------
_NOT_LANDED_NEEDLES = ("NOT LANDED", "committed, pushed, or landed")


def _assert_blocked_with_not_landed_warning(r: subprocess.CompletedProcess[str]) -> None:
    _assert_blocked(r)
    for needle in _NOT_LANDED_NEEDLES:
        assert needle in r.stderr, (needle, r.stderr)


def test_every_block_site_emits_the_not_landed_warning() -> None:
    """#2013: every `exit 2` block site emits the shared NOT-LANDED warning.

    Count-equality rather than a proximity window, so the assertion survives
    the heredoc growing and catches a future block site added without it.

    KNOWN BLIND SPOT (disclosed deliberately): the site count matches only
    STANDALONE `exit 2` lines. A future block written in a compound form
    (`... || exit 2`) would not be counted, so it could ship without the
    warning while this test stays green. The six sites that exist today are
    all standalone; a compound block site added later needs this test
    extended, not merely re-run.
    """
    src = SCRIPT.read_text().splitlines()
    n_sites = sum(1 for ln in src if ln.strip() == "exit 2")
    n_emits = sum(ln.count('"$NOT_LANDED_LINE"') + ln.count("${NOT_LANDED_LINE}") for ln in src)
    assert n_sites >= 6, n_sites
    assert n_emits == n_sites, (n_emits, n_sites)


def test_not_landed_warning_reaches_stderr_on_uncertified_payload_block(
    code_repo: Path, cert: Path
) -> None:
    """#2013 runtime, heredoc family (BLOCK_MSG): the uncertified-payload
    block carries the NOT-LANDED warning (mirrors the invocation of
    test_block_message_names_gate_remediation_and_override)."""
    _assert_blocked_with_not_landed_warning(_run("git commit -m x", code_repo, cert))


def test_not_landed_warning_reaches_stderr_on_blanket_add_block(
    code_repo: Path, cert: Path
) -> None:
    """#2013 runtime, blanket-stage-chained family: mirrors the B13 `dash-A`
    parametrize invocation."""
    _assert_blocked_with_not_landed_warning(_run("git add -A && git commit -m x", code_repo, cert))


def test_not_landed_warning_reaches_stderr_on_unprovable_cwd_block(
    art_repo: Path, cert: Path
) -> None:
    """#2013 runtime, path-limited-add cwd-gate family: mirrors the
    test_add_pathlimited_missing_cwd_blocks invocation (missing hook cwd)."""
    _write(art_repo, "tasks/t.md", "note\n")
    _assert_blocked_with_not_landed_warning(
        _run("git add --all -- tasks/t.md && git commit -m x", art_repo, cert, cwd=_OMIT_CWD)
    )


# ---------------------------------------------------------------------------
# wt-group (issue #2066): worktree-cwd allow gate. Pytest mirrors of the
# self-test rows W1/W3/W4/W7/W8/W9/W10 (the ids below cite those rows), plus
# the leg-(b) block-output order pin. A commit whose every contribution to
# root_commit=1 is a bare clause (no retarget evidence, no conservative-screen
# spelling) is ALLOWED when the hook-input cwd provably sits inside a linked
# worktree of the guarded repo; every refusal arm keeps the pre-#2066 block.
# ---------------------------------------------------------------------------


@pytest.fixture
def wt_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Root repo with an artifact-only init commit, an UNCERTIFIED gated file
    staged at the root, and a linked worktree (issue #2066)."""
    repo = _init_repo(tmp_path, "wtroot")
    _stage(repo, "tasks/t.md", "note\n")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    _stage(repo, GATED, "print(1)\n")
    wt = tmp_path / "wtroot-wt"
    _git(repo, "worktree", "add", "-q", str(wt))
    return repo, wt


def test_wt_bare_commit_from_worktree_cwd_allowed(wt_repo: tuple[Path, Path], cert: Path) -> None:
    """Self-test W1: worktree cwd + bare commit while the ROOT index holds an
    uncertified gated file -> ALLOW (the #2066 incident class)."""
    repo, wt = wt_repo
    _assert_allowed(_run("git commit -m x", repo, cert, cwd=wt))


def test_wt_cd_to_root_from_worktree_cwd_blocks(wt_repo: tuple[Path, Path], cert: Path) -> None:
    """Self-test W3: a cd-to-root prefix is retarget evidence — the allow
    gate never fires; the root-index read blocks."""
    repo, wt = wt_repo
    _assert_blocked(_run(f"cd {repo} && git commit -m x", repo, cert, cwd=wt))


def test_wt_dash_c_root_spelling_from_worktree_cwd_blocks(
    wt_repo: tuple[Path, Path], cert: Path
) -> None:
    """Self-test W4: a `git -C <root-spelling>` waiver REFUSAL is retarget
    evidence even from a worktree cwd."""
    repo, wt = wt_repo
    _assert_blocked(_run(f"git -C {_CANONICAL_ROOT} commit -m x", repo, cert, cwd=wt))


def test_wt_root_subdir_cwd_still_blocks(wt_repo: tuple[Path, Path], cert: Path) -> None:
    """Self-test W7 (B38 semantics): a root-SUBDIR cwd fails the worktree
    proof (toplevel == root) and keeps the conservative block."""
    repo, _wt = wt_repo
    _assert_blocked(_run("git commit -m x", repo, cert, cwd=repo / "tasks"))


def test_wt_kill_switch_restores_block(wt_repo: tuple[Path, Path], cert: Path) -> None:
    """Self-test W8: EPM_ROOT_CODE_COMMIT_DISABLE_WT_CWD_ALLOW=1 disables the
    allow gate wholesale (pre-#2066 cwd-blind behavior)."""
    repo, wt = wt_repo
    r = _run(
        "git commit -m x",
        repo,
        cert,
        cwd=wt,
        extra={"EPM_ROOT_CODE_COMMIT_DISABLE_WT_CWD_ALLOW": "1"},
    )
    _assert_blocked(r)


def test_wt_git_dir_env_assignment_from_worktree_cwd_blocks(
    wt_repo: tuple[Path, Path], cert: Path
) -> None:
    """Self-test W9: a GIT_DIR=-family env-assignment prefix on the commit
    clause hits the conservative screen — no allow."""
    repo, wt = wt_repo
    _assert_blocked(_run(f"GIT_DIR={repo}/.git git commit -m x", repo, cert, cwd=wt))


def test_wt_pushd_to_root_from_worktree_cwd_blocks(wt_repo: tuple[Path, Path], cert: Path) -> None:
    """Self-test W10: a pushd-to-root chain hits the conservative screen (the
    cd arm does not model pushd) — no allow."""
    repo, wt = wt_repo
    _assert_blocked(_run(f"pushd {repo} && git commit -m x", repo, cert, cwd=wt))


def test_wt_flag_intervened_cd_builtin_blocks(wt_repo: tuple[Path, Path], cert: Path) -> None:
    """Self-test W13 (round-2 fix, blocker wt-allow-screen-spelling-gaps): a
    flag word between the builtin keyword and the cd word evaded the round-1
    contiguity-anchored screen alternative; the widened junk-tolerant cd-WORD
    match catches the cd token itself regardless of what precedes it."""
    repo, wt = wt_repo
    _assert_blocked(_run(f"command -p cd {repo} && git commit -m x", repo, cert, cwd=wt))


def test_wt_backslash_escaped_cd_blocks(wt_repo: tuple[Path, Path], cert: Path) -> None:
    """Self-test W14 (round-2 fix): a backslash-escaped cd word (still the cd
    builtin to bash) evades the literal-cd cd arm AND the round-1 screen; the
    junk-tolerant cd-WORD match absorbs the escape character."""
    repo, wt = wt_repo
    _assert_blocked(_run(f"\\cd {repo} && git commit -m x", repo, cert, cwd=wt))


def test_wt_quoted_cd_word_blocks(wt_repo: tuple[Path, Path], cert: Path) -> None:
    """Self-test W15 (round-2 fix): a quoted cd word (still the cd builtin to
    bash) evades the literal-cd cd arm AND the round-1 screen; the
    junk-tolerant cd-WORD match absorbs the quote characters."""
    repo, wt = wt_repo
    _assert_blocked(_run(f"'cd' {repo} && git commit -m x", repo, cert, cwd=wt))


def test_wt_env_short_dir_flag_blocks(wt_repo: tuple[Path, Path], cert: Path) -> None:
    """Self-test W16 (round-2 fix): env's SHORT directory-change flag (the
    round-1 screen matched only the long-form spelling) — the first clause
    lands a commit at root while the trailing bare clause is what classifies;
    the widened -C-bearing-cluster alternative screens it."""
    repo, wt = wt_repo
    _assert_blocked(_run(f"env -C {repo} git commit -m x && git commit -m x", repo, cert, cwd=wt))


def test_block_message_leads_with_remediation_before_diagnostics(
    code_repo: Path, cert: Path
) -> None:
    """Leg (b) order pin (#2066): the final block heredoc leads with the
    remediation block (worktree rewrite + inline-lint-gate recipe) right
    after the headline + NOT-LANDED line, BEFORE the first cert-diag line."""
    r = _run("git commit -m x", code_repo, cert)
    _assert_blocked(r)
    assert "cert-diag:" in r.stderr, r.stderr
    first_diag = r.stderr.index("cert-diag:")
    assert r.stderr.index("NOT LANDED") < r.stderr.index('git -C "$WT" commit'), r.stderr
    assert r.stderr.index('git -C "$WT" commit') < first_diag, r.stderr
    assert r.stderr.index("inline_lint_gate.py") < first_diag, r.stderr
