"""Content-invariant pins for the #1412 binary-figures merge-recovery recipe.

Task #1412 added a mechanical resolution recipe for binary `figures/`
conflicts to the Step-9b/10d merge-conflict recovery in
`.claude/skills/issue/SKILL.md`: git cannot content-merge binaries and the
`.gitattributes` merge=union rules do not cover `figures/`, so
both-sides-changed figure paths always conflict. The recipe resolves them
newer-regeneration-wins — per conflicted path, compare the last-touching
commit time on each side (`git log -1 --format=%ct`); tie -> theirs (in this
merge ours = the issue branch, theirs = the captured $MAIN_SHA snapshot, the
#1090-proven side) — with the `git add` GATED on checkout success: a failed
checkout (modify/delete: missing stage) leaves the index entry UNMERGED and
echoes, so the recovery's later `git commit --no-edit` refuses — the loud
fall-through to the manual prose.

REGION-SCOPED per the `test_issue_skill_merge_resnapshot_pin.py` precedent
(slice `#### Merge-conflict recovery` -> `#### The artifact-confirmed merge
procedure`, fail-loud anchors). The rationale-prose pin normalizes
whitespace AND the `# ` comment prefixes: the rationale wraps across comment
lines inside the fenced bash block, so a raw substring pin on the spanning
phrase would false-fail on the hard wrap.

Origin incident: #1090 fu4 / PR #1066 — add/add conflict on
figures/issue_1090/fu4/fu4_tier2_verdict.{png,pdf}, improvised in-worktree
to main's newer regenerated copies (theirs; main-side regeneration commit
297e9ec27d post-dates the branch render), ~10 minutes of re-derivation this
recipe mechanizes.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

_FIGURES_DETECTOR = "diff --name-only --diff-filter=U -- 'figures/'"
_SIDE_SELECTION = "then SIDE=--theirs; else SIDE=--ours"
_CT_TOKEN = "--format=%ct"
_GATED_CHECKOUT = 'if git -C "$WT" checkout "$SIDE" -- "$p"'
_GATED_ADD_FAILURE_ECHO = "left UNMERGED"


def _skill_text() -> str:
    return _SKILL.read_text(encoding="utf-8")


def _recovery_region(text: str) -> str:
    """The merge-conflict recovery slice (the recipe's home)."""
    start_marker = "#### Merge-conflict recovery"
    end_marker = "#### The artifact-confirmed merge procedure"
    start = text.find(start_marker)
    end = text.find(end_marker)
    assert start != -1, "Merge-conflict recovery heading not found in SKILL.md"
    assert end != -1, "artifact-confirmed merge heading not found in SKILL.md"
    assert start < end, "recovery region must precede the artifact-confirmed merge"
    return text[start:end]


def _normalized_prose(text: str) -> str:
    """Whitespace-collapse AND strip leading `#` comment prefixes per line —
    the recipe's rationale prose wraps across bash comment lines, so pins on
    spanning phrases need the prefixes removed before joining."""
    words: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        words.extend(stripped.split())
    return " ".join(words)


def test_recovery_region_carries_binary_figures_recipe():
    """Asserts 1+2: the recovery region carries the figures/ conflicted-path
    detector AND the deterministic newer-wins side selection (per-path
    commit-time compare via %ct; tie -> theirs, the captured main snapshot)."""
    recovery = _recovery_region(_skill_text())
    # Assert 1 — the detector.
    assert _FIGURES_DETECTOR in recovery, (
        "the recovery must detect conflicted figures/ paths via "
        "diff --name-only --diff-filter=U -- 'figures/'"
    )
    # Assert 2 — the side-selection line + the commit-time comparison token.
    assert _SIDE_SELECTION in recovery, (
        "the recipe must select the side deterministically (then SIDE=--theirs; else SIDE=--ours)"
    )
    assert _CT_TOKEN in recovery, (
        "the side selection must compare last-touching commit times (--format=%ct)"
    )


def test_recovery_prose_pins_newer_regeneration_rule():
    """Assert 3: the normalized region prose carries the
    newer-regeneration-wins contract — a rewording that drops the RULE (not
    just the code) fails this pin. The spanning-phrase check exercises the
    comment-prefix normalization (the phrase hard-wraps across `# ` lines)."""
    prose = _normalized_prose(_recovery_region(_skill_text()))
    assert "NEWER" in prose, "the rationale must state the NEWER-regeneration-wins contract"
    assert "regeneration" in prose, (
        "the rationale must name regeneration (figures are regenerable artifacts)"
    )
    assert "the NEWER regeneration wins" in prose, (
        "the newer-regeneration-wins rule must survive as connected prose "
        "(normalized across the comment-line wrap)"
    )


def test_recovery_gated_add_fails_loud_on_checkout_failure():
    """Assert 4 (critic-r1 fix, MANDATORY): the `git add` is GATED on
    checkout success — a failed checkout (modify/delete: missing stage)
    echoes and leaves the index entry UNMERGED, so the recovery's later
    `git commit --no-edit` refuses (the loud fall-through to the manual
    prose). A later edit reintroducing an unconditional `git add` (fail-open
    on modify/delete) fails this pin. Deliberately NOT the
    non-halting brace-group echo form, which
    test_recovery_certification_arms_are_exclusive bans in this region."""
    recovery = _recovery_region(_skill_text())
    assert _GATED_CHECKOUT in recovery, (
        "the per-path side checkout must run inside an `if` gate "
        "(the git add is conditional on checkout success)"
    )
    assert _GATED_ADD_FAILURE_ECHO in recovery, (
        "the checkout-failure arm must echo that the entry is left UNMERGED "
        "(the loud fall-through to the manual-resolution prose)"
    )
