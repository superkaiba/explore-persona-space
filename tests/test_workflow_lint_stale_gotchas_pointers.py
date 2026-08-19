"""Tests for ``workflow_lint --check-stale-gotchas-pointers`` (#2193).

The check FAILs a dead task-id next to a ``gotchas.md`` mention in
``CLAUDE.md`` + ``.claude/**/*.md`` — an id within 100 chars of a
``gotchas.md`` token on the same line, at or under the registry's literal
``highest_id``, that no longer occurs in ``.claude/rules/gotchas.md`` (the
#2189 stale-relocation-pointer class; also id-trims where the entry
survives with its citation compacted away).

Covers, per plan v3 § Test plan items 1-9 + the critic-round-1
implementer notes:

1.  the research-project-structure regression shape
    (``gotchas.md``; incident #<dead-id>) FIRES;
2.  the id-before-mention shape (``the #<dead-id> recipe in gotchas.md``)
    FIRES;
3.  relocation-attribution idiom lines are CLEAN (both idioms), and the
    two skip-string literals are PINNED verbatim so drift breaks a test;
4.  the ``pytorch#94772``-class lookbehind and the id-above-registry-cap
    filter are CLEAN — the cap is the literal ``highest_id`` field of a
    MIXED-KEY registry (a naive max() over top-level digit keys would
    over-read and flag);
5.  a live id present in the fixture gotchas.md is CLEAN;
6.  a co-mention beyond the 100-char window is CLEAN;
7.  an allowlisted ``(path, id)`` is CLEAN while the SAME line at a new
    path still FIRES (never inherited) + allowlist hygiene (reasons
    non-trivial, entries point at live-tree files);
8.  live-tree smoke — the real check over the real repo returns []
    (baseline-clean landing; this test breaks loudly on the next stale
    relocation, which is the point of the task);
9.  the MUTATION-VISIBLE no-flags DISPATCH test (the
    ``test_check_jsonl_splitlines_bundled_in_no_flags`` pattern) — a
    direct call of the check function is NOT sufficient bundling evidence.

Plus: MIN-distance-over-multiple-tokens semantics (implementer note 4 — a
first-occurrence-only ``line.find()`` implementation would silently
diverge from the calibration), the registry contract (#2193 Step 10d gate
fix: ABSENT registry => ONE ``WARN: `` line + uncapped scan-more degrade,
the gate's registry-less ``git archive`` landing tree being a supported
scan root; PRESENT-but-unparseable registry => fail-loud FAIL row,
unchanged), and the docstring's disclosed false-negative classes (note 1).

Round-2 scope contract (#2193 r2, the Claude+Codex agreed blocker): the
scan inventory is the git-TRACKED ``CLAUDE.md`` + ``.claude/**/*.md`` set
(real temp-git-repo fixtures) — a tracked ``.claude/plans/`` offender
FIRES, untracked scratch is IGNORED, an unreadable tracked in-scope file
FAILs loud (never notice-and-skip), and a skip-worktree (sparse) entry is
not an inventory hole; tmp trees without git exercise the pruned-walk
fallback (:func:`workflow_lint._stale_gotchas_scan_paths`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402
from workflow_lint import (  # noqa: E402
    STALE_GOTCHAS_POINTER_ALLOWLIST,
    check_stale_gotchas_pointers,
)

# Live ids: 100, 205, 301 (the compound #205/#301 pins the findall-based
# presence read — every id in ANY citation form counts as fresh).
GOTCHAS_FIXTURE = "# Gotchas\n- **Some live trap (#100).** Mechanics; siblings (#205/#301).\n"

# Reshaped-but-structurally-faithful regression fixtures (the two baseline
# true-positive SHAPES the plan names).
RELOCATED_POINTER_SHAPE = "(see `.claude/rules/gotchas.md`; incident #923).\n"
ID_BEFORE_MENTION_SHAPE = "Relaunch contract (the #491 SSH-relaunch recipe in gotchas.md).\n"


def _plant(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _plant_tree(root: Path, *, registry: dict | None = None) -> None:
    """Plant the check's two fixed read surfaces (gotchas.md + registry)."""
    _plant(root, ".claude/rules/gotchas.md", GOTCHAS_FIXTURE)
    reg = registry if registry is not None else {"highest_id": 2000, "tasks": {"1": "x"}}
    _plant(root, "tasks/REGISTRY.json", json.dumps(reg))


def _run_on(monkeypatch, tmp_path: Path) -> list[str]:
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    return check_stale_gotchas_pointers()


# --------------------------------------------------------------------------
# 1 + 2. the two baseline true-positive shapes FIRE
# --------------------------------------------------------------------------


def test_flags_relocated_pointer_shape(tmp_path, monkeypatch) -> None:
    """The research-project-structure regression shape: mention-then-id."""
    _plant_tree(tmp_path)
    _plant(tmp_path, ".claude/rules/some_rule.md", RELOCATED_POINTER_SHAPE)
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "stale-gotchas-pointer/.claude/rules/some_rule.md:1" in errors[0], errors
    # Implementer note 3: the class is named accurately — a superset of
    # relocation, so triage must never assume every hit means a relocation.
    assert "dead task-id #923 next to a gotchas.md mention" in errors[0], errors


def test_flags_id_before_mention_shape(tmp_path, monkeypatch) -> None:
    _plant_tree(tmp_path)
    _plant(tmp_path, ".claude/rules/failover.md", ID_BEFORE_MENTION_SHAPE)
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "stale-gotchas-pointer/.claude/rules/failover.md:1" in errors[0], errors
    assert "dead task-id #491" in errors[0], errors


def test_repo_root_claude_md_scanned(tmp_path, monkeypatch) -> None:
    """CLAUDE.md at the repo root is in scan scope alongside .claude/**."""
    _plant_tree(tmp_path)
    _plant(tmp_path, "CLAUDE.md", RELOCATED_POINTER_SHAPE)
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "stale-gotchas-pointer/CLAUDE.md:1" in errors[0], errors


# --------------------------------------------------------------------------
# 3. relocation-attribution idiom lines are skipped; literals pinned
# --------------------------------------------------------------------------


def test_relocation_attribution_idiom_lines_clean(tmp_path, monkeypatch) -> None:
    """Both sanctioned attribution idioms waive their line — a correct
    relocation record legitimately names gotchas.md next to the relocating
    task's id (#923 is dead in the fixture gotchas.md, so without the skip
    both lines would FIRE)."""
    _plant_tree(tmp_path)
    _plant(
        tmp_path,
        ".claude/rules/owning_rule.md",
        "## Relocated codebase traps (from `.claude/rules/gotchas.md`, #923)\n"
        "Verbatim entries relocated to recover gotchas.md byte budget (#923).\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_skip_idiom_literals_pinned() -> None:
    """Implementer note 2: pin both skip-string literals verbatim so a
    drifted skip string (which would silently re-flag every future
    relocation's attribution header) breaks this test, not the fleet."""
    assert wl._STALE_GOTCHAS_SKIP_IDIOMS == (
        "Relocated codebase traps",
        "to recover gotchas.md byte budget",
    )


# --------------------------------------------------------------------------
# 4. lookbehind + registry id-cap (mixed-key registry fixture)
# --------------------------------------------------------------------------


def test_external_tracker_ref_lookbehind_clean(tmp_path, monkeypatch) -> None:
    """``pytorch#300``-class external-tracker refs are not task ids (the
    id 300 is <= cap and dead in the fixture gotchas.md, so only the
    lookbehind keeps this clean)."""
    _plant_tree(tmp_path)
    _plant(
        tmp_path,
        ".claude/rules/upstream.md",
        "an upstream class (pytorch#300) documented in gotchas.md too.\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_id_above_highest_id_field_clean_on_mixed_key_registry(tmp_path, monkeypatch) -> None:
    """Implementer note 5: the cap is the LITERAL highest_id field. The
    registry fixture mixes digit and non-digit top-level keys; a naive
    max() over digit keys would read 1500 and FLAG the dead #600 — the
    field read (500) skips it. A dead id under the field cap still FIRES
    (same tree), so the pass above cannot be a vacuous no-scan."""
    _plant_tree(
        tmp_path,
        registry={"highest_id": 500, "tasks": {"1": "x"}, "1500": "stray-digit-key"},
    )
    _plant(
        tmp_path,
        ".claude/rules/capped.md",
        "see gotchas.md for #600.\nand gotchas.md also covered #400 once.\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "stale-gotchas-pointer/.claude/rules/capped.md:2" in errors[0], errors
    assert "dead task-id #400" in errors[0], errors


def test_absent_registry_degrades_disclosed_and_scans_uncapped(
    tmp_path, monkeypatch, capsys
) -> None:
    """Registry ABSENCE is an environment, not an error (#2193 Step 10d
    gate fix): the Step 10d gate's landing tree is a ``git archive``
    extraction with no ``tasks/`` dir, and a plain non-git /tmp dir is a
    SUPPORTED workflow_lint scan root. Absent registry => zero FAIL rows
    from the registry read, exactly ONE ``WARN: ``-prefixed stderr line
    (the gate compare drops WARN lines), and an UNCAPPED scan — both the
    ordinary dead-id offender AND a dead id far ABOVE any plausible
    registry cap still FIRE (scan-more, never scan-less)."""
    _plant(tmp_path, ".claude/rules/gotchas.md", GOTCHAS_FIXTURE)
    _plant(tmp_path, ".claude/rules/some_rule.md", RELOCATED_POINTER_SHAPE)
    _plant(tmp_path, ".claude/rules/huge_id.md", "(see gotchas.md; incident #987654).\n")
    errors = _run_on(monkeypatch, tmp_path)
    err = capsys.readouterr().err
    assert len(errors) == 2, errors  # the two offender rows ONLY — no registry FAIL row
    assert any("dead task-id #923" in e for e in errors), errors
    assert any("dead task-id #987654" in e for e in errors), errors  # uncapped detection
    assert not any("highest_id" in e for e in errors), errors
    warn_lines = [ln for ln in err.splitlines() if ln.startswith("WARN: ")]
    assert len(warn_lines) == 1, err
    assert "tasks/REGISTRY.json absent from scan root" in warn_lines[0], err
    assert "id-cap disabled" in warn_lines[0], err


def test_present_but_unparseable_registry_fails_loud(tmp_path, monkeypatch) -> None:
    """Corruption keeps the fail-loud contract (#2193 Step 10d gate fix
    narrows ONLY absence): a PRESENT ``tasks/REGISTRY.json`` whose
    ``highest_id`` field is missing is an ERROR — the registry-read FAIL
    row is the single output, never a silent skip or a WARN degrade."""
    _plant_tree(tmp_path, registry={"tasks": {"1": "x"}})  # present, no highest_id
    _plant(tmp_path, ".claude/rules/some_rule.md", RELOCATED_POINTER_SHAPE)
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "cannot read the literal" in errors[0], errors
    assert "highest_id" in errors[0], errors


# --------------------------------------------------------------------------
# 5 + 6. live id / beyond-window co-mentions are clean
# --------------------------------------------------------------------------


def test_live_id_clean(tmp_path, monkeypatch) -> None:
    _plant_tree(tmp_path)
    _plant(
        tmp_path,
        ".claude/rules/fresh.md",
        "see gotchas.md #100; the compound-cited sibling #301 also lives there.\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_beyond_window_clean(tmp_path, monkeypatch) -> None:
    filler = "x" * 120
    _plant_tree(tmp_path)
    _plant(
        tmp_path,
        ".claude/rules/far.md",
        f"gotchas.md {filler} #923 is far from the token.\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_min_distance_over_multiple_tokens(tmp_path, monkeypatch) -> None:
    """Implementer note 4: with TWO gotchas.md tokens on one line the gap
    is the MIN over tokens — here #923 sits >100 chars from the FIRST
    token but adjacent to the SECOND, so a first-occurrence-only
    ``line.find()`` implementation would silently skip it."""
    filler = "y" * 120
    _plant_tree(tmp_path)
    _plant(
        tmp_path,
        ".claude/rules/two_tokens.md",
        f"gotchas.md {filler} gotchas.md relocated #923.\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "dead task-id #923" in errors[0], errors


# --------------------------------------------------------------------------
# 7. allowlist: (path, id)-scoped, never inherited; hygiene
# --------------------------------------------------------------------------

_ALLOWLISTED_LINE = "gotcha_candidate routes candidates to gotchas.md; #711 motivated the field.\n"


def test_allowlisted_path_id_clean_but_never_inherited(tmp_path, monkeypatch) -> None:
    _plant_tree(tmp_path)
    _plant(tmp_path, ".claude/skills/issue/markers.md", _ALLOWLISTED_LINE)
    _plant(tmp_path, ".claude/rules/copycat.md", _ALLOWLISTED_LINE)
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "stale-gotchas-pointer/.claude/rules/copycat.md:1" in errors[0], errors


def test_allowlist_entries_carry_reasons_and_point_at_live_files() -> None:
    """Every allowlist entry carries a non-trivial reason string and its
    path exists on the live tree (dead entries are removed, never
    accumulated)."""
    assert STALE_GOTCHAS_POINTER_ALLOWLIST, "allowlist unexpectedly empty"
    root = wl._REPO_ROOT
    for (rel, n), reason in sorted(STALE_GOTCHAS_POINTER_ALLOWLIST.items()):
        assert isinstance(n, int) and n > 0, (rel, n)
        assert len(reason) >= 20, f"trivial reason for ({rel}, {n}): {reason!r}"
        assert (root / rel).is_file(), f"allowlist entry no longer on the tree: {rel}"


# --------------------------------------------------------------------------
# scan scope: the git-TRACKED inventory (#2193 r2) + the approved exclusions
# --------------------------------------------------------------------------


def test_agent_memory_target_and_worktrees_not_scanned(tmp_path, monkeypatch) -> None:
    """The ONLY approved exclusions: agent-memory copies are historical
    records ("true when written") and gotchas.md itself is the
    co-reference target, not a pointer surface. Sibling-worktree
    duplicates are structurally out of reach on BOTH enumeration forms
    (untracked on the git path; pruned by _iter_files_pruned on the
    non-git fallback walk this fixture exercises)."""
    _plant_tree(tmp_path)
    for rel in (
        ".claude/agent-memory/critic/feedback_old.md",
        ".claude/worktrees/issue-9999/.claude/rules/some_rule.md",
    ):
        _plant(tmp_path, rel, RELOCATED_POINTER_SHAPE)
    assert _run_on(monkeypatch, tmp_path) == []


@pytest.fixture
def git_root():
    """A scratch root OUTSIDE pytest's numbered /tmp/pytest-of-* basetemp
    (concurrent pytest sessions prune those roots mid-test; these tests
    spawn git subprocesses against the tree) hosting a REAL temp git repo
    — the tracked-inventory production path."""
    root = Path(tempfile.mkdtemp(prefix="wl-stale-gotchas-git-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _plant_git_tree(root: Path) -> None:
    """git init + the two fixed read surfaces, with .claude/ + tasks/
    ADDED to the index (git ls-files reads the index; no commit needed)."""
    _git(root, "init", "-q")
    _plant_tree(root)
    _git(root, "add", ".claude", "tasks")


def test_tracked_plans_offender_fires_untracked_scratch_ignored(git_root, monkeypatch) -> None:
    """Round-2 scope contract (#2193 r2, both reviewers): the inventory is
    the git-TRACKED set. A TRACKED .claude/plans offender FIRES (round 1's
    plans-exclusion pinned the WRONG behavior — 12 tracked plan docs were
    silently escaping the lint), while an UNTRACKED .claude/rules scratch
    file carrying the SAME offender line is IGNORED (local scratch must
    never red the no-flags landing gate) — the single-error assert makes
    both halves non-vacuous at once."""
    _plant_git_tree(git_root)
    _plant(git_root, ".claude/plans/offender.md", RELOCATED_POINTER_SHAPE)
    _git(git_root, "add", ".claude/plans/offender.md")
    _plant(git_root, ".claude/rules/scratch.md", RELOCATED_POINTER_SHAPE)  # NOT git-added
    errors = _run_on(monkeypatch, git_root)
    assert len(errors) == 1, errors
    assert "stale-gotchas-pointer/.claude/plans/offender.md:1" in errors[0], errors


def test_unreadable_tracked_file_fails_loud(git_root, monkeypatch) -> None:
    """Round-2 sibling 3: a tracked in-scope file that cannot be READ is a
    FAIL row, never a notice-and-skip (a silent skip is an inventory
    hole). Portable unreadable shape: tracked in the index, deleted from
    the working tree (FileNotFoundError, an OSError)."""
    _plant_git_tree(git_root)
    doomed = _plant(git_root, ".claude/rules/deleted_rule.md", "clean text, no token\n")
    _git(git_root, "add", ".claude/rules/deleted_rule.md")
    doomed.unlink()
    errors = _run_on(monkeypatch, git_root)
    assert len(errors) == 1, errors
    assert "stale-gotchas-pointer/.claude/rules/deleted_rule.md" in errors[0], errors
    assert "unreadable" in errors[0], errors


def test_skip_worktree_entry_not_an_inventory_hole(git_root, monkeypatch) -> None:
    """A sparse checkout deliberately does not materialize skip-worktree
    entries — the enumerator keeps 'H' (cached) rows only, so an S-tagged
    absent file is CLEAN, never an unreadable-file FAIL (sparse worktrees
    are the fleet's default shape, new_worktree.sh)."""
    _plant_git_tree(git_root)
    sparse = _plant(git_root, ".claude/rules/sparse_rule.md", RELOCATED_POINTER_SHAPE)
    _git(git_root, "add", ".claude/rules/sparse_rule.md")
    _git(git_root, "update-index", "--skip-worktree", ".claude/rules/sparse_rule.md")
    sparse.unlink()
    assert _run_on(monkeypatch, git_root) == []


# --------------------------------------------------------------------------
# note 1: the docstring discloses the known false-negative classes
# --------------------------------------------------------------------------


def test_docstring_discloses_false_negative_classes() -> None:
    doc = check_stale_gotchas_pointers.__doc__ or ""
    assert "WRAPPED-LINE" in doc, "docstring must disclose the wrapped-line split miss"
    assert "ID ALIASING" in doc, "docstring must disclose the id-aliasing miss"
    assert "Relocated codebase traps" in doc, "docstring must name skip idiom 1 verbatim"
    assert "to recover gotchas.md byte budget" in doc, "docstring must name skip idiom 2 verbatim"
    assert "REGISTRY-ABSENT DEGRADE" in doc, (
        "docstring must disclose the registry-absent WARN + uncapped-scan degrade (#2193 "
        "Step 10d gate fix)"
    )


# --------------------------------------------------------------------------
# 8. live-tree smoke (baseline-clean landing)
# --------------------------------------------------------------------------


def test_live_tree_passes() -> None:
    """Zero stale-gotchas-pointer rows on the current tree: the 5 baseline
    true positives are fixed by #2193 and the 2 markers.md
    context-citations are allowlisted. This is the test that breaks loudly
    on the next stale relocation — which is the point of the task."""
    assert check_stale_gotchas_pointers() == []


# --------------------------------------------------------------------------
# 9. no-flags bundling (mutation-visible dispatch test)
# --------------------------------------------------------------------------


def test_stale_gotchas_pointers_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the check — deleting
    its ``or no_flags`` branch must fail this test (mutation-visible; the
    ``test_check_jsonl_splitlines_bundled_in_no_flags`` pattern). Other
    bundled checks contribute unrelated errors on the minimal tree, so the
    assertion keys on the check's own diagnostic token + offending path."""
    _plant_tree(tmp_path)
    _plant(tmp_path, ".claude/rules/offender.md", RELOCATED_POINTER_SHAPE)
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offending tree:\n{err}"
    assert "stale-gotchas-pointer/.claude/rules/offender.md:1" in err, (
        f"the stale-gotchas-pointer diagnostic (naming offender.md) is missing "
        f"from the no-flags default run's stderr — the check is not bundled "
        f"into no_flags:\n{err}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
