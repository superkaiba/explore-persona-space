"""Tests for ``workflow_lint.check_no_repo_root_git_reset_hard`` (#815).

The check FAILs any ``.claude/agents/*.md`` or ``.claude/skills/**/SKILL.md``
that contains an UNQUALIFIED destructive ``git reset --hard`` (a repo-root /
full-tree reset). Only two forms pass:

- **worktree-qualified** — a ``git -C "$WT"`` prefix at a LOWER char offset than
  the offending reset on the same line (the sanctioned per-worktree pattern);
- **allowlisted** — a same-line ``workflow-lint: allow-git-reset-hard: <reason>``
  sentinel with a NON-EMPTY reason token.

Incident 2026-07-01: a #778 analyzer improvised a destructive repo-root reset
during marker-chain recovery and truncated concurrent siblings #812/#813 (their
``body.md`` / ``plans/`` / ``comments.jsonl`` / REGISTRY entries; recovered in
commits ``81c52d6a2b`` for #813 and ``d29a877e6f`` for #812). ``task.py`` holds
a per-registry ``flock``, not a per-file lock, so a repo-root reset by any
concurrent session clobbers unrelated tasks.

INTENTIONAL under-matching (a scoped design choice, NOT a bug — see
``test_under_matching_line_continuation_documented`` / T9): the line-based regex
does NOT catch a ``git reset \\``-continuation whose ``--hard`` lands on the
FOLLOWING physical line. Grep confirms ZERO live in-scope instances of a
``\\``-terminated ``git reset`` continuation, so this is future-proofing, not a
live gap; the check stays line-oriented because its scope is markdown, NOT a
shell AST. If such a form ever lands in-scope, normalize the continuation
before matching (or split the command).

Covers T1-T12 from the plan §10:
T1  a bare / ``origin/main`` repo-root reset FAILS with a file:line message;
T2  a worktree-qualified ``git -C "$WT" reset --hard`` does NOT flag;
T3  an allowlisted line (non-empty reason) does NOT flag; no-reason FLAGS;
T4  an EMPTY-reason sentinel (``: -->`` / no ``:``) FLAGS (FI2);
T5  out-of-scope files (rules/, plans/, agent-memory/, CLAUDE.md, scripts/) do
    NOT flag;
T6  the offending string is caught in BOTH a fenced block AND inline prose;
T7  the FI1 flag-ordering variants are EACH flagged;
T8  the FI3 ``-C``-must-precede-reset ordering (a chained reset before a
    worktree-qualified command FLAGS);
T9  the FI5 ``\\``-continuation under-match is documented + PINNED;
T10 the real post-fix tree PASSES (anti-regression lock);
T11 the live analyzer.md carries the hard-rule paragraph (MF2) + self-clean (MF1);
T12 the live CLAUDE.md line-39 bullet forbids a repo-root reset (MF2).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import check_no_repo_root_git_reset_hard  # noqa: E402

# The worktree / repo root the live-tree tests read (T10-T12). Resolving from
# the test file keeps this worktree-agnostic — it is whatever tree the suite
# runs in (worktree during a workflow-fix /issue session, repo root otherwise).
_REPO = _HERE.parent


def _write_agent(tmp_path: Path, name: str, body: str) -> Path:
    """Write a synthetic agent spec under ``tmp_path/.claude/agents/<name>``."""
    p = tmp_path / ".claude" / "agents" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _write_skill(tmp_path: Path, slug: str, body: str) -> Path:
    """Write a synthetic ``.claude/skills/<slug>/SKILL.md`` under ``tmp_path``."""
    p = tmp_path / ".claude" / "skills" / slug / "SKILL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# T1 — a bare / origin-pinned repo-root reset FAILS with file:line
# --------------------------------------------------------------------------


def test_flags_bare_repo_root_reset(tmp_path: Path) -> None:
    _write_agent(tmp_path, "foo.md", "Recovery step:\n    git reset --hard\ndone\n")
    _write_agent(tmp_path, "bar.md", "Sync:\n    git reset --hard origin/main\n")
    errors = check_no_repo_root_git_reset_hard(repo_root=tmp_path)
    assert len(errors) == 2, errors
    # each error names the file:line
    assert any("foo.md:2:" in e for e in errors), errors
    assert any("bar.md:2:" in e for e in errors), errors
    # and states the fix
    assert all('git -C "$WT" reset --hard' in e for e in errors), errors


def test_flags_reset_in_skill_file(tmp_path: Path) -> None:
    _write_skill(tmp_path, "issue", "Step X:\n    git reset --hard origin/main\n")
    errors = check_no_repo_root_git_reset_hard(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "SKILL.md:2:" in errors[0], errors


# --------------------------------------------------------------------------
# T2 — worktree-qualified `git -C "$WT" reset --hard` does NOT flag
# --------------------------------------------------------------------------


def test_skips_worktree_qualified(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "foo.md",
        'Recover inside the worktree:\n    git -C "$WT" reset --hard origin/main\n',
    )
    assert check_no_repo_root_git_reset_hard(repo_root=tmp_path) == []


# --------------------------------------------------------------------------
# T3 — allowlisted (non-empty reason) does NOT flag; missing-reason FLAGS
# --------------------------------------------------------------------------


def test_skips_allowlisted_with_reason(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "foo.md",
        "Prose mention `git reset --hard`  "
        "<!-- workflow-lint: allow-git-reset-hard: legit prose mention -->\n",
    )
    assert check_no_repo_root_git_reset_hard(repo_root=tmp_path) == []


def test_flags_allowlist_without_reason_token(tmp_path: Path) -> None:
    # sentinel present but NO ``:`` and NO reason after it -> still flagged
    _write_agent(
        tmp_path,
        "foo.md",
        "Prose `git reset --hard`  <!-- workflow-lint: allow-git-reset-hard -->\n",
    )
    errors = check_no_repo_root_git_reset_hard(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "foo.md:1:" in errors[0], errors


# --------------------------------------------------------------------------
# T4 — FI2: empty-reason sentinel does NOT waive
# --------------------------------------------------------------------------


def test_flags_allowlist_empty_reason(tmp_path: Path) -> None:
    # ``: -->`` — sentinel + colon but EMPTY reason after the colon
    _write_agent(
        tmp_path,
        "colon_empty.md",
        "Prose `git reset --hard`  <!-- workflow-lint: allow-git-reset-hard: -->\n",
    )
    # no colon at all, just the bare sentinel token
    _write_agent(
        tmp_path,
        "no_colon.md",
        "Prose `git reset --hard`  <!-- workflow-lint: allow-git-reset-hard -->\n",
    )
    errors = check_no_repo_root_git_reset_hard(repo_root=tmp_path)
    assert len(errors) == 2, errors
    assert any("colon_empty.md:1:" in e for e in errors), errors
    assert any("no_colon.md:1:" in e for e in errors), errors


# --------------------------------------------------------------------------
# T5 — out-of-scope files are NEVER scanned
# --------------------------------------------------------------------------


def test_scope_excludes_out_of_scope_files(tmp_path: Path) -> None:
    # .claude/rules/*.md
    rules = tmp_path / ".claude" / "rules" / "bar.md"
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text("    git reset --hard origin/main\n", encoding="utf-8")
    # .claude/plans/*.md
    plans = tmp_path / ".claude" / "plans" / "baz.md"
    plans.parent.mkdir(parents=True, exist_ok=True)
    plans.write_text("    git reset --hard\n", encoding="utf-8")
    # .claude/agent-memory/**
    mem = tmp_path / ".claude" / "agent-memory" / "analyzer" / "y.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text("    git reset --hard\n", encoding="utf-8")
    # CLAUDE.md
    (tmp_path / "CLAUDE.md").write_text("    git reset --hard\n", encoding="utf-8")
    # scripts/*.py
    scripts = tmp_path / "scripts" / "qux.py"
    scripts.parent.mkdir(parents=True, exist_ok=True)
    scripts.write_text("# git reset --hard\n", encoding="utf-8")

    assert check_no_repo_root_git_reset_hard(repo_root=tmp_path) == []


# --------------------------------------------------------------------------
# T6 — caught in BOTH a fenced code block AND inline prose
# --------------------------------------------------------------------------


def test_matches_in_fenced_block_and_prose(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "fenced.md",
        "```bash\ngit reset --hard origin/main\n```\n",
    )
    _write_agent(
        tmp_path,
        "prose.md",
        "Then you might `git reset --hard` to clear the index.\n",
    )
    errors = check_no_repo_root_git_reset_hard(repo_root=tmp_path)
    assert len(errors) == 2, errors
    assert any("fenced.md:2:" in e for e in errors), errors
    assert any("prose.md:1:" in e for e in errors), errors


# --------------------------------------------------------------------------
# T7 — FI1: every flag-ordering variant is flagged
# --------------------------------------------------------------------------


def test_flag_ordering_variants_flagged(tmp_path: Path) -> None:
    variants = [
        "git reset --hard",
        "git reset -q --hard",
        "git reset --hard origin/main",
        "git reset origin/main --hard",  # ref BEFORE --hard
        "git --no-pager reset --hard",  # git-level flag before subcommand
        "git reset --hard=origin/main",  # attached-value flag
    ]
    body = "".join(f"    {v}\n" for v in variants)
    _write_agent(tmp_path, "variants.md", body)
    errors = check_no_repo_root_git_reset_hard(repo_root=tmp_path)
    assert len(errors) == len(variants), (len(errors), errors)


# --------------------------------------------------------------------------
# T8 — FI3: `git -C` must PRECEDE the reset to waive it
# --------------------------------------------------------------------------


def test_dash_c_must_precede_reset(tmp_path: Path) -> None:
    # unqualified reset chained BEFORE a worktree-qualified command -> FLAGGED
    # (the `git -C` is at a HIGHER char offset than the offending reset)
    _write_agent(
        tmp_path,
        "chained.md",
        '    git reset --hard && git -C "$WT" status\n',
    )
    errors = check_no_repo_root_git_reset_hard(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "chained.md:1:" in errors[0], errors

    # companion: the properly-qualified form (from T2) is NOT flagged — the
    # `-C` is at a LOWER offset than the reset.
    _write_agent(
        tmp_path,
        "qualified.md",
        '    git -C "$WT" reset --hard origin/main\n',
    )
    errors2 = check_no_repo_root_git_reset_hard(repo_root=tmp_path)
    # still exactly one (the chained.md hit); qualified.md added nothing
    assert len(errors2) == 1, errors2
    assert all("qualified.md" not in e for e in errors2), errors2


# --------------------------------------------------------------------------
# T9 — FI5: the `\`-continuation under-match is INTENTIONAL + documented
# --------------------------------------------------------------------------


def test_under_matching_line_continuation_documented(tmp_path: Path) -> None:
    """A ``git reset \\`` continuation whose ``--hard`` lands on the NEXT
    physical line is an ACCEPTED, grep-confirmed-empty-in-scope non-match — the
    check is line-oriented by design (markdown scope, not a shell AST). This
    test PINS the deliberate under-match so a future contributor sees it is a
    scoped design choice, not a bug (module docstring + plan §6 kill-criteria).
    If a ``\\``-continuation destructive reset ever lands in-scope, the fix is
    to normalize line continuations before matching (or split the command).
    """
    _write_agent(
        tmp_path,
        "continuation.md",
        "    git reset \\\n      --hard origin/main\n",
    )
    assert check_no_repo_root_git_reset_hard(repo_root=tmp_path) == []


# --------------------------------------------------------------------------
# T10 — the real post-fix tree PASSES (anti-regression lock)
# --------------------------------------------------------------------------


def test_live_tree_passes() -> None:
    """``check_no_repo_root_git_reset_hard()`` returns ``[]`` on the real
    post-fix tree. The anti-regression lock: it guarantees experimenter.md:166's
    sentinel + analyzer.md's sanctioned-form-only paragraph keep the live
    surface clean, and fails loud if a future edit drops a sentinel or adds a
    bare repo-root reset. Also confirms MF1: if analyzer.md's added paragraph
    tripped the check, this test would FAIL.
    """
    assert check_no_repo_root_git_reset_hard(repo_root=_REPO) == []


# --------------------------------------------------------------------------
# T11 — MF2: analyzer.md carries the hard-rule paragraph; self-clean (MF1)
# --------------------------------------------------------------------------


def test_analyzer_md_carries_hard_rule() -> None:
    """Grep the LIVE analyzer.md (case-insensitively) for the failure-mode
    anchor phrase, all three incident SHAs, and the sanctioned per-worktree
    pattern; then assert the added paragraph does NOT self-trip the lint (MF1).
    """
    analyzer = _REPO / ".claude" / "agents" / "analyzer.md"
    text = analyzer.read_text(encoding="utf-8").lower()
    # failure-mode anchor phrase
    assert "concurrent siblings" in text, "missing 'concurrent siblings' anchor"
    assert "clobber" in text, "missing 'clobber' anchor"
    # all three incident SHAs
    for sha in ("bbd6fe97b7", "81c52d6a2b", "d29a877e6f"):
        assert sha in text, f"missing incident SHA {sha} in analyzer.md"
    # the sanctioned per-worktree pattern
    assert 'git -c "$wt" reset --hard' in text, "missing sanctioned per-worktree pattern"
    # MF1: the added paragraph must not self-trip the lint — no failure whose
    # path is analyzer.md.
    errors = check_no_repo_root_git_reset_hard(repo_root=_REPO)
    assert all("analyzer.md" not in e for e in errors), errors


# --------------------------------------------------------------------------
# T12 — MF2: CLAUDE.md line-39 bullet forbids a repo-root reset
# --------------------------------------------------------------------------


def test_claude_md_forbids_repo_root_reset() -> None:
    """Grep the LIVE CLAUDE.md (case-insensitively) for language naming
    ``git reset --hard`` as forbidden on the repo root: the substring
    ``git reset --hard`` co-occurring with a forbidding token
    (``forbidden`` | ``never`` | ``clobber``) somewhere in the file. Substring
    assertions, not exact-line — tolerant to future rewordings, strict about
    presence.
    """
    claude_md = _REPO / "CLAUDE.md"
    text = claude_md.read_text(encoding="utf-8").lower()
    assert "git reset --hard" in text, "CLAUDE.md does not name git reset --hard"
    assert any(tok in text for tok in ("forbidden", "never", "clobber")), (
        "CLAUDE.md names git reset --hard but with no forbidding token"
    )
