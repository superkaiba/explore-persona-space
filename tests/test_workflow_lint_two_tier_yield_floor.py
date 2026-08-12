"""Pins ``workflow_lint.py --check-two-tier-yield-floor`` (#2242; incident #2221).

The check pins the two-tier yield-floor contract — relative shrink floor +
absolute per-cell trainability floor with the DROP disposition — across four
surfaces (on-policy-completions.md, planner-section-reference.md,
critic-lens-reference.md, planner.md), region-anchored, including the TWO
machinery-keyed N/A escapes (the round-2 MUST-FIX: a gate-keyed escape lets a
shrink-only / never-drop design truthfully self-exempt — exactly #2221's
shape). Mirrors ``tests/test_workflow_lint_smoke_blind_spots.py``.

The synthetic s2 fixture deliberately keeps ``equalize-down`` present
ELSEWHERE in the anchored region while the escape sentence is rewritten back
to gate-keyed wording, so a region-scoped (rather than sentence-scoped)
implementation of the escape sub-pin goes red here (trap t2); and the corpus
carries the adjacent Equalize-down bullet ending in the byte-identical
``write "N/A" and move on.`` phrase, so a char-window region locator captures
the sibling's escape and fails the complete-corpus test (trap t1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from workflow_lint import check_two_tier_yield_floor  # noqa: E402

MACHINERY_ESCAPE_PSR = (
    "If the design has no per-cell yield machinery at all — neither a "
    'keep/drop eligibility gate nor a shrink / equalize-down rule — write "N/A" '
    "and move on."
)
GATEKEYED_ESCAPE_PSR = (
    'If the design has no continuous-quantity eligibility gate, write "N/A" and move on.'
)
MACHINERY_ESCAPE_CLR = (
    "the design has no per-cell yield machinery at all — neither a keep/drop "
    "eligibility gate NOR a shrink / equalize-down rule (i), per-unit N is "
    "equal by construction (ii)."
)
GATEKEYED_ESCAPE_CLR = (
    "the design has no continuous-quantity eligibility gate (i), per-unit N is "
    "equal by construction (ii)."
)


def _write_corpus(root: Path, *, drop: str | None = None) -> Path:
    """A minimal four-surface corpus; ``drop`` strips exactly one pinned piece."""
    rules = root / ".claude" / "rules"
    agents = root / ".claude" / "agents"
    rules.mkdir(parents=True, exist_ok=True)
    agents.mkdir(parents=True, exist_ok=True)

    # (s1) on-policy-completions.md — the absolute-floor sub-bullet between two
    # sibling sub-bullets (region bound = the next bullet).
    if drop != "s1-file":
        anchor = "REMOVED bullet" if drop == "s1-anchor" else "Absolute per-cell trainability floor"
        s1_tokens = (
            "mechanics = `assert_cell_trainable(...)` at the row-counting site; "
            "disposition = DROP, the denominator is revised everywhere and the "
            "drop is named in `## Takeaways`; smoke demotion enumerated per "
            "`.claude/rules/smoke-blind-spots.md`."
        )
        if drop == "s1-token":
            s1_tokens = s1_tokens.replace("assert_cell_trainable", "some_other_helper")
        (rules / "on-policy-completions.md").write_text(
            "# on-policy\n\n"
            "- **Pre-registered per-source yield quota.** The 80% relative floor.\n"
            f"  - **{anchor} (distinct from the relative 80% floor; #2221/#2242).** "
            f"{s1_tokens}\n"
            "  - **Scale contrastive negatives proportionally to floor-N** so the "
            "ratio survives.\n",
            encoding="utf-8",
        )

    # (s2) planner-section-reference.md — the anchored bullet PLUS the adjacent
    # Equalize-down bullet whose trailing phrase is byte-identical (trap t1);
    # the region keeps 'equalize-down' outside the escape sentence (trap t2).
    if drop != "s2-file":
        two_tier = (
            ""
            if drop == "s2-token"
            else "**Two-tier floors (REQUIRED for any equalize-down / yield-quota "
            "design):** the yield row states BOTH the relative shrink floor AND "
            "the absolute trainability floor with the DROP disposition. "
        )
        escape = GATEKEYED_ESCAPE_PSR if drop == "s2-escape-gatekeyed" else MACHINERY_ESCAPE_PSR
        (rules / "planner-section-reference.md").write_text(
            "# planner section reference\n\n"
            "- **No all-or-nothing eligibility gates on continuous quantities "
            "(graceful degradation).** An 80% floor + equalize-down + the "
            f"close-miss escalation tranche is canonical. {two_tier}{escape}\n"
            "- **Equalize-down when a per-unit resource varies.** If per-unit N is "
            'equal by construction, write "N/A" and move on.\n',
            encoding="utf-8",
        )

    # (s3) critic-lens-reference.md — the item-9 region bounded by the next
    # top-level numbered item; escape span = 'Not a REVISE when' to region end.
    if drop != "s3-file":
        body = (
            "Direction (b): an unbounded shrink with no ABSOLUTE per-cell "
            "trainability floor lets equalize-down land at 1 row (#2221); the "
            "miss is DROPPED with the denominator revised."
        )
        if drop == "s3-token":
            body = body.replace("#2221", "#0000")
        escape3 = GATEKEYED_ESCAPE_CLR if drop == "s3-escape-gatekeyed" else MACHINERY_ESCAPE_CLR
        (rules / "critic-lens-reference.md").write_text(
            "# critic lens reference\n\n"
            "### Statistics lens\n\n"
            "9. **Degenerate eligibility gates, unequal per-unit N, missing "
            "baseline propensity (four related design-lesson checks).** "
            f"(i) {body}\n"
            f"   Not a REVISE when: {escape3}\n"
            "10. **Dual-DV.** The next item.\n",
            encoding="utf-8",
        )

    # (s4) planner.md — the §4 hard-requirement capsule token.
    if drop != "s4-file":
        capsule = (
            "no all-or-nothing eligibility gates ·"
            if drop == "s4-capsule"
            else "no all-or-nothing eligibility gates / two-tier yield floors "
            "(relative shrink + absolute-trainability DROP) ·"
        )
        (agents / "planner.md").write_text(
            "# planner\n\nsmoke/sweep architectural parity · " + capsule + "\n",
            encoding="utf-8",
        )
    return root


def test_passes_on_complete_corpus(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    errors = check_two_tier_yield_floor(repo_root=tmp_path)
    assert errors == [], f"complete corpus should pass; got: {errors}"


_DROP_CASES: list[tuple[str, str, str]] = [
    ("s1-file", "missing", "rules/on-policy-completions.md"),
    ("s1-anchor", "Absolute per-cell trainability floor", "rules/on-policy-completions.md"),
    ("s1-token", "assert_cell_trainable", "rules/on-policy-completions.md"),
    ("s2-file", "missing", "rules/planner-section-reference.md"),
    ("s2-token", "Two-tier", "rules/planner-section-reference.md"),
    ("s2-escape-gatekeyed", "equalize-down", "rules/planner-section-reference.md"),
    ("s3-file", "missing", "rules/critic-lens-reference.md"),
    ("s3-token", "#2221", "rules/critic-lens-reference.md"),
    ("s3-escape-gatekeyed", "shrink", "rules/critic-lens-reference.md"),
    ("s4-file", "missing", "agents/planner.md"),
    ("s4-capsule", "two-tier yield floors", "agents/planner.md"),
]


@pytest.mark.parametrize(("drop", "token", "path_frag"), _DROP_CASES)
def test_fails_per_stripped_surface(tmp_path: Path, drop: str, token: str, path_frag: str) -> None:
    _write_corpus(tmp_path, drop=drop)
    errors = check_two_tier_yield_floor(repo_root=tmp_path)
    assert errors, f"drop={drop}: expected >=1 error"
    assert any(token in e and path_frag in e for e in errors), (
        f"drop={drop}: no error carries both {token!r} and {path_frag!r}; got: {errors}"
    )


def test_s2_escape_regression_names_the_sentence(tmp_path: Path) -> None:
    """The round-2 MUST-FIX regression fixture (trap t2): the s2 escape sentence
    is rewritten back to GATE-keyed wording while 'equalize-down' stays present
    elsewhere in the region — a region-scoped implementation would pass; the
    sentence-scoped sub-pin must FAIL naming BOTH machinery tokens."""
    _write_corpus(tmp_path, drop="s2-escape-gatekeyed")
    errors = check_two_tier_yield_floor(repo_root=tmp_path)
    for token in ("'shrink'", "'equalize-down'"):
        assert any(token in e and "planner-section-reference.md" in e for e in errors), (
            f"expected an s2 escape-sentence error naming {token}; got: {errors}"
        )


def test_s3_escape_regression_names_the_span(tmp_path: Path) -> None:
    """The MUST-FIX regression fixture on the acceptance surface: the item-9
    'Not a REVISE when' (i) clause rewritten back to gate-keyed wording FAILs
    naming both machinery tokens."""
    _write_corpus(tmp_path, drop="s3-escape-gatekeyed")
    errors = check_two_tier_yield_floor(repo_root=tmp_path)
    for token in ("'shrink'", "'equalize-down'"):
        assert any(token in e and "critic-lens-reference.md" in e for e in errors), (
            f"expected an s3 escape-span error naming {token}; got: {errors}"
        )


def test_surfaces_pinned(tmp_path: Path) -> None:
    """D10 durability pin: every one of the four surfaces is load-bearing —
    deleting ANY surface file from a complete corpus produces >=1 error naming
    that file."""
    for drop, path_frag in (
        ("s1-file", "rules/on-policy-completions.md"),
        ("s2-file", "rules/planner-section-reference.md"),
        ("s3-file", "rules/critic-lens-reference.md"),
        ("s4-file", "agents/planner.md"),
    ):
        root = tmp_path / drop
        _write_corpus(root, drop=drop)
        errors = check_two_tier_yield_floor(repo_root=root)
        assert any(path_frag in e for e in errors), (
            f"drop={drop}: no error names {path_frag!r}; got: {errors}"
        )


def test_passes_on_live_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binds the landed #2242 edits; the standing regression guard for future
    refactors of any of the four surfaces."""
    monkeypatch.delenv("EPS_WORKFLOW_LINT_REPO_ROOT", raising=False)
    errors = check_two_tier_yield_floor(repo_root=None)
    assert errors == [], f"live tree should carry all four surfaces; got: {errors}"
