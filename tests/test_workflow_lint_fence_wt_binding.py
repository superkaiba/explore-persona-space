"""Tests for ``workflow_lint --check-issue-skill-fence-wt-binding`` (#2306).

The check FAILs any ``bash``/``sh``/``shell``-tagged OR UNTAGGED fenced block
under ``.claude/skills/issue/`` (SKILL.md + ``steps/*.md``) that references
``$WT`` without either (a) an in-fence binding — a ``WT=`` assignment line or
a non-comment ``eval "$(bash scripts/step10d_guards.sh <N> --guard prelude)"``
line — or (b) the literal ``wt-binding:`` annotation token (the #2306 D2
convention declaring the fence caller/prelude-dependent).

Why the check exists: fenced blocks are SEPARATE shells when extracted and
run standalone, and BOTH ``git -C ""`` and ``cd ""`` are silent no-ops
(rc=0, cwd unchanged) — an unbound ``$WT`` therefore retargets every
``git -C "$WT" ...`` / ``cd "$WT"`` at the SHARED repo root with no error
(#2306/#2293). The negative control below pins the load-bearing subtlety: a
bare ``cd "$WT" || { ...; exit 1; }`` guard is NOT accepted as a binding,
because ``cd ""`` SUCCEEDS and the guard passes with ``WT`` unbound.

Covers: (i) each covered form passing (WT= binding; non-comment prelude
eval; ``wt-binding:`` annotation); (ii) the naked-fence FAIL naming file +
fence line; (iii) the ``cd "$WT" ||``-only FAIL (silent-no-op negative
control); (iv) an UNTAGGED ``$WT`` fence FAILing (in scope by design);
(v) comment-only pseudo-bindings NOT counting (``#   WT=`` and a commented
prelude mention without the annotation token); (vi) ``${WT}`` detected /
``$WTX`` and non-shell fences not flagged; (vii) out-of-scope files ignored;
(viii) FAIL-CLOSED scan inputs (wt-lint-read-fail-open): an unreadable
enumerated scan file, a glob-matched non-regular ``steps/*.md`` entry
(broken symlink), an invalid-UTF-8 step body, a missing SKILL.md, and a
missing/empty ``steps/`` dir are each ONE named lint ERROR — never a
silent pass;
(ix) the live tree passes; (x) the MUTATION-VISIBLE no-flags DISPATCH test
(the ``tests/test_workflow_lint.py:3455`` pattern) — a direct call of the
check function is NOT sufficient evidence of bundling.

Fixture trees are completed by ``_scaffold`` (a benign SKILL.md + a benign
step body) so the fail-closed missing-input arm stays quiet on trees that
deliberately plant only the piece under test; the missing-input tests call
the check directly on incomplete roots.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402
from workflow_lint import check_issue_skill_fence_wt_binding  # noqa: E402

REPO_ROOT = _HERE.parent

STEP_REL = ".claude/skills/issue/steps/99-fixture.md"


def _plant(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _scaffold(root: Path) -> None:
    """Complete the minimal /issue-skill family (benign SKILL.md + one benign
    step body) so the fail-closed missing-input arm stays quiet on fixture
    trees that plant only the piece under test."""
    skill = root / ".claude/skills/issue/SKILL.md"
    if not skill.is_file():
        _plant(root, ".claude/skills/issue/SKILL.md", "# Router\n\nno fences here\n")
    steps = root / ".claude/skills/issue/steps"
    if not (steps.is_dir() and any(p.is_file() for p in steps.glob("*.md"))):
        _plant(root, ".claude/skills/issue/steps/00-benign.md", "# Step\n\nno fences here\n")


def _run(root: Path) -> list[str]:
    _scaffold(root)
    return check_issue_skill_fence_wt_binding(repo_root=root)


# --------------------------------------------------------------------------
# (i) covered forms pass
# --------------------------------------------------------------------------


def test_bound_fence_passes(tmp_path) -> None:
    _plant(
        tmp_path,
        STEP_REL,
        "# Step\n\n"
        "```bash\n"
        'REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")\n'
        'WT="$REPO_ROOT/.claude/worktrees/issue-<N>"\n'
        '[ -n "$WT" ] && [ -d "$WT" ] || { echo "FATAL: WT unbound" >&2; exit 1; }\n'
        'git -C "$WT" status\n'
        "```\n",
    )
    assert _run(tmp_path) == []


def test_prelude_eval_binding_passes(tmp_path) -> None:
    _plant(
        tmp_path,
        STEP_REL,
        "# Step\n\n"
        "```bash\n"
        'eval "$(bash scripts/step10d_guards.sh 42 --guard prelude)"\n'
        'git -C "$WT" log --oneline -1\n'
        "```\n",
    )
    assert _run(tmp_path) == []


def test_annotated_fence_passes(tmp_path) -> None:
    _plant(
        tmp_path,
        STEP_REL,
        "# Step\n\n"
        "```bash\n"
        "# wt-binding: caller — $WT is bound by the composing orchestrator turn;\n"
        '# extracting standalone? prepend: eval "$(bash scripts/step10d_guards.sh <N> '
        '--guard prelude)"\n'
        'git -C "$WT" status --porcelain\n'
        "```\n",
    )
    assert _run(tmp_path) == []


# --------------------------------------------------------------------------
# (ii)-(iv) failing shapes
# --------------------------------------------------------------------------


def test_naked_wt_fence_fails_naming_file_and_line(tmp_path) -> None:
    _plant(
        tmp_path,
        STEP_REL,
        '# Step\n\nprose\n\n```bash\ngit -C "$WT" status\n```\n',
    )
    errors = _run(tmp_path)
    assert len(errors) == 1, errors
    # Fence opener is line 5 of the fixture body.
    assert errors[0].startswith(f"{STEP_REL}:5:"), errors[0]
    assert "no in-fence binding" in errors[0]
    assert "wt-binding" in errors[0]


def test_cd_wt_guard_alone_fails(tmp_path) -> None:
    """The silent-no-op negative control: ``cd ""`` SUCCEEDS (rc=0, cwd
    unchanged), so a bare ``cd "$WT" ||`` guard passes with WT unbound —
    it must NOT count as a binding (#2306 verified fact)."""
    _plant(
        tmp_path,
        STEP_REL,
        "# Step\n\n"
        "```bash\n"
        'cd "$WT" || { echo "FATAL: cd to issue worktree failed" >&2; exit 1; }\n'
        "uv run pytest -q\n"
        "```\n",
    )
    errors = _run(tmp_path)
    assert len(errors) == 1, errors
    assert "no in-fence binding" in errors[0]


def test_untagged_wt_fence_fails(tmp_path) -> None:
    _plant(
        tmp_path,
        STEP_REL,
        '# Step\n\n```\nrsync -a "$WT/eval_results/" /tmp/out/\n```\n',
    )
    errors = _run(tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (v) comment-only pseudo-bindings do NOT count
# --------------------------------------------------------------------------


def test_commented_wt_assignment_is_not_a_binding(tmp_path) -> None:
    _plant(
        tmp_path,
        STEP_REL,
        "# Step\n\n"
        "```bash\n"
        '#   WT="$REPO_ROOT/.claude/worktrees/issue-<N>"\n'
        'git -C "$WT" status\n'
        "```\n",
    )
    errors = _run(tmp_path)
    assert len(errors) == 1, errors


def test_commented_prelude_mention_without_token_fails(tmp_path) -> None:
    """A comment merely MENTIONING the prelude eval does not bind and, absent
    the ``wt-binding:`` token, does not annotate either."""
    _plant(
        tmp_path,
        STEP_REL,
        "# Step\n\n"
        "```bash\n"
        '# consider: eval "$(bash scripts/step10d_guards.sh <N> --guard prelude)"\n'
        'git -C "$WT" status\n'
        "```\n",
    )
    errors = _run(tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (vi) reference-form edges
# --------------------------------------------------------------------------


def test_braced_wt_reference_detected(tmp_path) -> None:
    _plant(
        tmp_path,
        STEP_REL,
        '# Step\n\n```bash\nls "${WT}/scripts"\n```\n',
    )
    errors = _run(tmp_path)
    assert len(errors) == 1, errors


def test_non_wt_names_and_wt_free_fences_not_flagged(tmp_path) -> None:
    _plant(
        tmp_path,
        STEP_REL,
        '# Step\n\n```bash\necho "$WTXYZ"\ngit status\n```\n',
    )
    assert _run(tmp_path) == []


def test_non_shell_fence_out_of_scope(tmp_path) -> None:
    _plant(
        tmp_path,
        STEP_REL,
        '# Step\n\n```python\npath = "$WT/eval_results"\n```\n',
    )
    assert _run(tmp_path) == []


# --------------------------------------------------------------------------
# (vii) scan-set boundaries
# --------------------------------------------------------------------------


def test_out_of_scope_file_ignored(tmp_path) -> None:
    _plant(
        tmp_path,
        ".claude/skills/other/SKILL.md",
        '# Other\n\n```bash\ngit -C "$WT" status\n```\n',
    )
    assert _run(tmp_path) == []


def test_skill_md_is_in_scope(tmp_path) -> None:
    _plant(
        tmp_path,
        ".claude/skills/issue/SKILL.md",
        '# Router\n\n```bash\ngit -C "$WT" status\n```\n',
    )
    errors = _run(tmp_path)
    assert len(errors) == 1, errors
    assert errors[0].startswith(".claude/skills/issue/SKILL.md:"), errors[0]


# --------------------------------------------------------------------------
# (viii) FAIL-CLOSED scan inputs: unreadable / missing inputs are named lint
# ERRORS, never a silent pass (wt-lint-read-fail-open)
# --------------------------------------------------------------------------


def test_unreadable_file_is_a_named_lint_error(tmp_path) -> None:
    """An enumerated-but-unreadable scan file FAILs the check with exactly one
    named error (file path + reason) — pre-fix this was a stderr NOTE + skip
    that left the error list empty (fail-open, wt-lint-read-fail-open)."""
    if os.geteuid() == 0:
        pytest.skip("chmod 000 does not block reads for root")
    p = _plant(tmp_path, STEP_REL, '# Step\n\n```bash\ngit -C "$WT" status\n```\n')
    p.chmod(0)
    try:
        errors = _run(tmp_path)
    finally:
        p.chmod(0o644)
    assert len(errors) == 1, errors
    assert "check_issue_skill_fence_wt_binding" in errors[0], errors[0]
    assert "unreadable" in errors[0], errors[0]
    assert "99-fixture.md" in errors[0], errors[0]


def test_broken_symlink_step_is_a_named_lint_error(tmp_path) -> None:
    """A glob-matched ``steps/*.md`` entry that is not a readable regular
    file (broken symlink) beside an ordinary step file FAILs with exactly one
    named error naming it — pre-fix the ``if p.is_file()`` candidate filter
    silently DROPPED it whenever a benign sibling existed (fail-open,
    wt-lint-read-fail-open round 2). Root-independent: no chmod involved."""
    _plant(tmp_path, ".claude/skills/issue/SKILL.md", "# Router\n\nno fences here\n")
    _plant(tmp_path, ".claude/skills/issue/steps/00-benign.md", "# Step\n\nno fences here\n")
    os.symlink(tmp_path / "does-not-exist.md", tmp_path / ".claude/skills/issue/steps/bad.md")
    errors = check_issue_skill_fence_wt_binding(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "check_issue_skill_fence_wt_binding" in errors[0], errors[0]
    assert "bad.md" in errors[0], errors[0]
    assert "not a readable regular file" in errors[0], errors[0]


def test_invalid_utf8_step_is_a_named_lint_error(tmp_path) -> None:
    """A ``steps/*.md`` file holding invalid UTF-8 bytes FAILs with the same
    named "unreadable scan input" error — pre-fix ``read_text`` raised an
    uncaught ``UnicodeDecodeError`` (a ValueError, escaping the bare
    ``except OSError`` arm; wt-lint-read-fail-open round 2). Root-independent:
    no chmod involved."""
    _plant(tmp_path, ".claude/skills/issue/SKILL.md", "# Router\n\nno fences here\n")
    bad = tmp_path / ".claude/skills/issue/steps/99-bad-bytes.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"\xff\xfe\x80")
    errors = check_issue_skill_fence_wt_binding(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "check_issue_skill_fence_wt_binding" in errors[0], errors[0]
    assert "unreadable" in errors[0], errors[0]
    assert "99-bad-bytes.md" in errors[0], errors[0]


def test_missing_skill_md_is_a_named_lint_error(tmp_path) -> None:
    _plant(tmp_path, ".claude/skills/issue/steps/00-benign.md", "# Step\n\nno fences here\n")
    errors = check_issue_skill_fence_wt_binding(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert "missing scan input" in errors[0], errors[0]
    assert ".claude/skills/issue/SKILL.md" in errors[0], errors[0]


def test_missing_steps_dir_is_a_named_lint_error(tmp_path) -> None:
    _plant(tmp_path, ".claude/skills/issue/SKILL.md", "# Router\n\nno fences here\n")
    errors = check_issue_skill_fence_wt_binding(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert ".claude/skills/issue/steps/" in errors[0], errors[0]
    assert "missing" in errors[0], errors[0]


def test_empty_steps_dir_is_a_named_lint_error(tmp_path) -> None:
    _plant(tmp_path, ".claude/skills/issue/SKILL.md", "# Router\n\nno fences here\n")
    (tmp_path / ".claude/skills/issue/steps").mkdir(parents=True)
    errors = check_issue_skill_fence_wt_binding(repo_root=tmp_path)
    assert len(errors) == 1, errors
    assert ".claude/skills/issue/steps/" in errors[0], errors[0]
    assert "empty" in errors[0], errors[0]


# --------------------------------------------------------------------------
# (ix) the live tree passes
# --------------------------------------------------------------------------


def test_live_tree_passes() -> None:
    errors = check_issue_skill_fence_wt_binding(repo_root=REPO_ROOT)
    assert errors == [], "live /issue skill tree has uncovered $WT fences:\n" + "\n".join(errors)


# --------------------------------------------------------------------------
# (x) the MUTATION-VISIBLE no-flags DISPATCH test (the :3455 pattern)
# --------------------------------------------------------------------------


def test_check_issue_skill_fence_wt_binding_bundled_in_no_flags(
    tmp_path, capsys, monkeypatch
) -> None:
    """The no-flags default run actually DISPATCHES the #2306 check — deleting
    its ``or no_flags`` branch must fail this test (mutation-visible). Other
    bundled checks contribute unrelated errors on the minimal tree, so the
    assertion keys on the check's own diagnostic token + the offending path."""
    _plant(
        tmp_path,
        ".claude/skills/issue/steps/99-offender.md",
        '# Step\n\n```bash\ngit -C "$WT" status\n```\n',
    )
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offending tree:\n{err}"
    assert "wt-binding" in err and "99-offender.md" in err, (
        f"the fence-wt-binding diagnostic (naming 99-offender.md) is missing "
        f"from the no-flags default run's stderr — the check is not bundled "
        f"into no_flags:\n{err}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
