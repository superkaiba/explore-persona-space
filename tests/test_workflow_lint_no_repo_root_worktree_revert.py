"""Tests for ``workflow_lint.check_no_repo_root_worktree_revert`` (#897).

The check FAILs any ``.claude/agents/*.md`` or ``.claude/skills/**/SKILL.md``
that prescribes an UNQUALIFIED working-tree revert on the shared repo root:

- a non-``--staged`` ``git restore`` (any pathspec — explicit-path restore has
  zero legitimate live doc uses; ``--staged`` forms are index-only + exempt);
- a bare-dot wholesale ``git checkout .`` (explicit ``checkout <ref> --
  <path>`` doc mentions are deliberately NOT flagged — legitimate prescriptive
  uses exist; the RUNTIME hook covers those with the ``-C`` waiver as the
  deliberate override — the prescriptive-vs-runtime split);
- a force-flagged ``git clean`` (``-f`` anywhere in a short cluster, or
  ``--force``).

Only two forms pass (waiver logic shared with the #815 reset-hard check via
``_line_waived``):

- **worktree-qualified** — a ``git -C`` prefix at-or-before the match's char
  offset on the same line (FI3 ``<=`` semantics);
- **allowlisted** — a same-line ``workflow-lint: allow-repo-root-wt-revert:
  <reason>`` sentinel with a NON-EMPTY reason (FI2).

Incident 2026-07-02 (#841): a concurrent session's destructive working-tree
git op on the shared repo root reverted the #841 analyzer's uncommitted
``body.md`` mid-task (and deleted untracked pre-registration + figure files) —
the #815 hazard class (``task.py`` holds a per-registry flock, not per-file).

Covers T1-T9 from the plan §3d:
T1  each flagged form FAILs with a file:line message, in agents AND skills;
T2  the FI3 ``-C``-before-match waiver (incl. the same-``git`` ``<=`` case);
T3  a reasoned sentinel waives;
T4  an empty-reason / no-colon sentinel does NOT (FI2);
T5  out-of-scope files (rules/, plans/, agent-memory/, CLAUDE.md, scripts/)
    are NEVER scanned;
T6  not-flagged forms (``--staged`` restore, explicit-``--`` checkout doc
    mentions, dry-run clean, plain-prose "restore");
T7  fenced-block + inline-prose (backtick-terminated) matches both caught;
T8  the no-flags bundling pin — ``main([])`` on a doctored tree FAILs with
    the #841 diagnostic (mutation-visible registration certification), and
    the explicit ``--check-no-repo-root-worktree-revert`` path FAILs/PASSes;
T9  the live-tree migration pin — the SKILL.md Step 10d additive-checkout
    fence and the code-reviewer.md smoke-restore prescription carry
    ``git -C`` (content-anchored, never line numbers);
T10 (round 2) a comment-tail ``--staged`` does NOT waive the restore pattern
    (the exemption lookahead is bounded at an unquoted ``#`` — concern id
    lint-restore-lookahead-comment-tail).

Plus the live-tree anti-regression lock (the analyzer.md ban-context
sentinels keep the real tree clean).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint  # noqa: E402
from workflow_lint import check_no_repo_root_worktree_revert  # noqa: E402

from tests.issue_skill_source import issue_skill_text  # noqa: E402

# The worktree / repo root the live-tree tests read (T9 + the anti-regression
# lock). Resolving from the test file keeps this worktree-agnostic — it is
# whatever tree the suite runs in (an issue worktree during a workflow-fix
# /issue session, the repo root otherwise); post-merge both resolutions carry
# the migrated fence, per the plan's "robust to both" requirement.
_REPO = _HERE.parent

_SENTINEL = "workflow-lint: allow-repo-root-wt-revert"


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
# T1 — each flagged form FAILs with file:line, in agents AND skills
# --------------------------------------------------------------------------


def test_flags_each_revert_form_in_agent(tmp_path: Path) -> None:
    forms = [
        "git restore .",
        "git restore path/to/file",
        "git checkout .",
        "git clean -fd",
        "git clean --force",
    ]
    body = "".join(f"    {f}\n" for f in forms)
    _write_agent(tmp_path, "foo.md", body)
    errors = check_no_repo_root_worktree_revert(repo_root=tmp_path)
    assert len(errors) == len(forms), (len(errors), errors)
    for lineno in range(1, len(forms) + 1):
        assert any(f"foo.md:{lineno}:" in e for e in errors), (lineno, errors)
    # every error states the incident + the remediation + the sentinel escape
    assert all("#841" in e for e in errors), errors
    assert all('git -C "$WT"' in e for e in errors), errors
    assert all(_SENTINEL in e for e in errors), errors


def test_flags_revert_in_skill_file(tmp_path: Path) -> None:
    _write_skill(tmp_path, "issue", "Step X:\n    git restore .\n")
    errors = check_no_repo_root_worktree_revert(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "SKILL.md:2:" in errors[0], errors


# --------------------------------------------------------------------------
# T2 — FI3: `git -C` at-or-before the match waives; after it does NOT
# --------------------------------------------------------------------------


def test_dash_c_qualified_forms_waived(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "foo.md",
        "Recover inside the worktree:\n"
        '    git -C "$WT" restore .\n'
        '    git -C "$WT" clean -fdx\n'
        '    git -C "$WT" checkout .\n',
    )
    assert check_no_repo_root_worktree_revert(repo_root=tmp_path) == []


def test_dash_c_mention_before_match_waives(tmp_path: Path) -> None:
    # a `git -C` mention at a LOWER char offset on the same line waives the
    # later match (the incidental-FI3 case the analyzer.md ban-context line
    # relied on before its explicit sentinel)
    _write_agent(
        tmp_path,
        "foo.md",
        'scoped with `git -C "$WT"`: never a bare `git restore .` there\n',
    )
    assert check_no_repo_root_worktree_revert(repo_root=tmp_path) == []


def test_dash_c_after_match_does_not_waive(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "chained.md",
        '    git restore . && git -C "$WT" status\n',
    )
    errors = check_no_repo_root_worktree_revert(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "chained.md:1:" in errors[0], errors


# --------------------------------------------------------------------------
# T3 — a reasoned sentinel waives
# --------------------------------------------------------------------------


def test_sentinel_with_reason_waives(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "foo.md",
        f"Prose mention `git restore .`  <!-- {_SENTINEL}: ban-context mention (#897) -->\n",
    )
    assert check_no_repo_root_worktree_revert(repo_root=tmp_path) == []


# --------------------------------------------------------------------------
# T4 — FI2: empty-reason / no-colon sentinel does NOT waive
# --------------------------------------------------------------------------


def test_sentinel_without_reason_flags(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "colon_empty.md",
        f"Prose `git clean -fd`  <!-- {_SENTINEL}: -->\n",
    )
    _write_agent(
        tmp_path,
        "no_colon.md",
        f"Prose `git checkout .`  <!-- {_SENTINEL} -->\n",
    )
    errors = check_no_repo_root_worktree_revert(repo_root=tmp_path)
    assert len(errors) == 2, errors
    assert any("colon_empty.md:1:" in e for e in errors), errors
    assert any("no_colon.md:1:" in e for e in errors), errors


# --------------------------------------------------------------------------
# T5 — out-of-scope files are NEVER scanned
# --------------------------------------------------------------------------


def test_scope_excludes_out_of_scope_files(tmp_path: Path) -> None:
    rules = tmp_path / ".claude" / "rules" / "bar.md"
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text("    git restore .\n", encoding="utf-8")
    plans = tmp_path / ".claude" / "plans" / "baz.md"
    plans.parent.mkdir(parents=True, exist_ok=True)
    plans.write_text("    git checkout .\n", encoding="utf-8")
    mem = tmp_path / ".claude" / "agent-memory" / "analyzer" / "y.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text("    git clean -fd\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("    git restore .\n", encoding="utf-8")
    scripts = tmp_path / "scripts" / "qux.py"
    scripts.parent.mkdir(parents=True, exist_ok=True)
    scripts.write_text("# git clean -fd\n", encoding="utf-8")

    assert check_no_repo_root_worktree_revert(repo_root=tmp_path) == []


# --------------------------------------------------------------------------
# T6 — not-flagged forms
# --------------------------------------------------------------------------


def test_not_flagged_forms(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "foo.md",
        "Unstage with `git restore --staged foo.py` (index-only).\n"
        "Sync spec files: `git checkout main -- $SAFE_SPECS` (explicit `--`).\n"
        "Dry-run first: `git clean -n` shows what would go.\n"
        "Then restore the file from HF and re-run.\n"
        "Surgical: `git checkout issue-42 -- tasks/x/body.md` is fine in docs.\n",
    )
    assert check_no_repo_root_worktree_revert(repo_root=tmp_path) == []


# --------------------------------------------------------------------------
# T7 — caught in BOTH a fenced code block AND inline prose (backtick term.)
# --------------------------------------------------------------------------


def test_matches_in_fenced_block_and_prose(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "fenced.md",
        "```bash\ngit checkout .\n```\n",
    )
    _write_agent(
        tmp_path,
        "prose.md",
        "Then you might `git checkout .` to clear local edits.\n",
    )
    errors = check_no_repo_root_worktree_revert(repo_root=tmp_path)
    assert len(errors) == 2, errors
    assert any("fenced.md:2:" in e for e in errors), errors
    assert any("prose.md:1:" in e for e in errors), errors


# --------------------------------------------------------------------------
# T10 (round 2) — a comment-tail `--staged` cannot waive the restore pattern
# (concern id lint-restore-lookahead-comment-tail). Bash never executes an
# unquoted comment tail, so a fenced `git restore . # --staged` line is a
# destructive working-tree restore; the exemption lookahead is bounded at
# `#` (and backtick) so only a `--staged` among the REAL arguments exempts.
# --------------------------------------------------------------------------


def test_comment_tail_staged_does_not_waive_restore(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "spoof.md",
        "```bash\ngit restore . # --staged\n```\n",
    )
    errors = check_no_repo_root_worktree_revert(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "spoof.md:2:" in errors[0], errors
    # symmetry: a REAL `--staged` argument (before any `#`) keeps the
    # exemption even with a trailing comment on the same line
    _write_agent(
        tmp_path,
        "spoof.md",
        "```bash\ngit restore --staged foo.py # unstage only\n```\n",
    )
    assert check_no_repo_root_worktree_revert(repo_root=tmp_path) == []


# --------------------------------------------------------------------------
# Live-tree anti-regression lock — the real post-fix tree PASSES
# --------------------------------------------------------------------------


def test_live_tree_passes() -> None:
    """``check_no_repo_root_worktree_revert()`` returns ``[]`` on the real
    post-fix tree: the analyzer.md ban-context sentinels keep the live surface
    clean, and this fails loud if a future edit drops a sentinel or adds an
    unqualified working-tree-revert prescription."""
    assert check_no_repo_root_worktree_revert(repo_root=_REPO) == []


# --------------------------------------------------------------------------
# T8 — no-flags bundling pin (registration certification, mutation-visible)
# --------------------------------------------------------------------------


def test_bundled_in_no_flags(tmp_path: Path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the #897 check — deleting
    any of its 3 wiring sites (argparse flag, ``no_flags`` disjunction,
    execution block) must fail this test. Follows the
    ``test_vm_thread_cap_guidance_bundled_in_no_flags`` pattern: one offending
    fixture file, ``_REPO_ROOT`` monkeypatched at it, ``main([])`` in-process.
    Other bundled checks contribute unrelated errors on the minimal tree, so
    the assertion keys on the #841 diagnostic + the distinctive fixture name.
    """
    _write_agent(tmp_path, "wt_revert_offender.md", "    git restore .\n")
    monkeypatch.setattr(workflow_lint, "_REPO_ROOT", tmp_path)
    rc = workflow_lint.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offending tree:\n{err}"
    assert "#841" in err and "wt_revert_offender.md" in err, (
        f"the #841 worktree-revert diagnostic (naming wt_revert_offender.md) "
        f"is missing from the no-flags default run's stderr — the check is "
        f"not bundled into no_flags:\n{err}"
    )


def test_explicit_flag_path_fails_and_passes(tmp_path: Path, capsys, monkeypatch) -> None:
    """The explicit ``--check-no-repo-root-worktree-revert`` path FAILs on the
    offending fixture (argparse + execution-block wiring) and PASSes on a
    clean fixture (the flag runs ONLY this check — ``no_flags`` goes False)."""
    _write_agent(tmp_path, "wt_revert_offender.md", "    git clean -fd\n")
    monkeypatch.setattr(workflow_lint, "_REPO_ROOT", tmp_path)
    rc = workflow_lint.main(["--check-no-repo-root-worktree-revert"])
    err = capsys.readouterr().err
    assert rc == 1, f"explicit flag path exited {rc} on an offending tree:\n{err}"
    assert "#841" in err and "wt_revert_offender.md" in err, err

    clean_root = tmp_path / "clean"
    _write_agent(clean_root, "ok.md", 'Use `git -C "$WT" clean -fdx` in the worktree.\n')
    monkeypatch.setattr(workflow_lint, "_REPO_ROOT", clean_root)
    rc2 = workflow_lint.main(["--check-no-repo-root-worktree-revert"])
    err2 = capsys.readouterr().err
    assert rc2 == 0, f"explicit flag path exited {rc2} on a clean tree:\n{err2}"


# --------------------------------------------------------------------------
# T9 — live-tree migration pin (acceptance criterion 5; content-anchored)
# --------------------------------------------------------------------------

# Flag-tolerant anchor (#1076): #1047 hardened every fence consumer to
# `xargs -r -a` (--no-run-if-empty; pinned by tests/test_step10d_guard3.py),
# and future flag hardening must not re-break the T9 test below — its
# load-bearing property is the -C qualification, not the xargs flag set.
_ADDITIVE_XARGS_ANCHOR_RE = re.compile(
    r"xargs\s+(?:-\S+\s+)*-a\s+/tmp/issue-<N>-additive-files\.txt"
)


def test_live_skill_md_additive_checkout_is_dash_c_qualified() -> None:
    """The /issue Step 10d surgical additive checkout runs at the repo root
    (its fence's preceding line is ``cd "$REPO_ROOT"``), so the #897 hook's
    pathspec detector would bounce the bare ``checkout issue-<N> --`` form.
    Pin that the live fence carries the ``git -C "$REPO_ROOT"`` deliberate
    override. Located by CONTENT ANCHOR (the flag-tolerant ``xargs ... -a``
    list-file prefix), never by line number (the fence drifts). Also guards
    against a future SKILL.md edit reintroducing the unqualified fence."""
    text = issue_skill_text()
    # The `checkout issue-<N> --` conjunct pins the CHECKOUT fence line
    # specifically — the sibling `git commit -m "...: surgical additive
    # checkout ..."` fence line also carries the xargs prefix + the word
    # "checkout" (in its commit-message subject) and must not match.
    anchor_lines = [
        ln
        for ln in text.splitlines()
        if _ADDITIVE_XARGS_ANCHOR_RE.search(ln) and "checkout issue-<N> --" in ln
    ]
    assert anchor_lines, "Step 10d additive-checkout fence line not found in SKILL.md"
    for ln in anchor_lines:
        assert re.search(r'git\s+-C\s+"\$REPO_ROOT"\s+checkout\s+issue-<N>\s+--', ln), (
            f"additive-checkout fence line is not -C-qualified: {ln!r}"
        )


def test_additive_checkout_anchor_regex_flags_unqualified_fence() -> None:
    """Negative branch for the T9 anchor predicate (#1076): a synthetic
    UNQUALIFIED fence line (flag-bearing ``xargs -r -a`` but no
    ``git -C "$REPO_ROOT"``) is caught by the flag-tolerant anchor AND fails
    the -C qualification regex (mirrored verbatim from the live test above),
    so a future SKILL.md regression to the bare-checkout form would FAIL
    ``test_live_skill_md_additive_checkout_is_dash_c_qualified``."""
    bad = 'xargs -r -a /tmp/issue-<N>-additive-files.txt git checkout issue-<N> -- "$f"'
    assert _ADDITIVE_XARGS_ANCHOR_RE.search(bad), "anchor must tolerate the -r flag"
    assert "checkout issue-<N> --" in bad
    assert not re.search(r'git\s+-C\s+"\$REPO_ROOT"\s+checkout\s+issue-<N>\s+--', bad)
    # The pre-#1047 flagless form stays anchored too (backwards tolerance).
    flagless = "xargs -a /tmp/issue-<N>-additive-files.txt git checkout issue-<N> --"
    assert _ADDITIVE_XARGS_ANCHOR_RE.search(flagless), "anchor must keep matching flagless form"


def test_live_code_reviewer_smoke_restore_is_dash_c_qualified() -> None:
    """The code-reviewer smoke-restore prescription (Step 0.6 region —
    'restore the committed artifacts YOUR OWN command modified') must carry
    ``git -C`` so a reviewer executing it verbatim is not bounced by the #897
    hook detector. Located by content anchor, never line number."""
    cr = _REPO / ".claude" / "agents" / "code-reviewer.md"
    text = cr.read_text(encoding="utf-8")
    assert "never a blanket" in text, "smoke-restore prescription anchor missing"
    anchor_lines = [ln for ln in text.splitlines() if "checkout -- <paths>" in ln]
    assert anchor_lines, "smoke-restore checkout prescription line not found"
    for ln in anchor_lines:
        assert "git -C" in ln, f"smoke-restore prescription is not -C-qualified: {ln!r}"
