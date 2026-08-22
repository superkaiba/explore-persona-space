"""Prose-side pin for the #1927 canonical gist-update recipe (incident #1769).

``.claude/skills/issue/SKILL.md`` Step 9a-quater procedure step 6 carries ONE
canonical recipe for UPDATING an existing methodology-doc gist mirror on a
re-export / follow-up round: ``gh api -X PATCH "gists/$GIST_ID"`` with the
file content payload, followed by an API read-back verify of the gist GET
``content`` field against the local doc — because the ``gh gist edit``
update forms can silently no-op with rc=0 (incident #1769: the
EDITOR-override form returned rc=0 with content UNCHANGED), rc=0 alone is
never treated as success. This test pins the recipe's presence + shape so a
later SKILL.md edit cannot silently drop it (lineage #884/#1045/#1134), and
that every remaining ``gh gist edit`` mention sits in ban context (no
prescriptive use survives).
"""

from __future__ import annotations

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

SKILL_MD = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "issue" / "SKILL.md"

_TEXT = issue_skill_text()

# Ban vocabulary that marks a `gh gist edit` mention as non-prescriptive
# (checked CASE-INSENSITIVELY — the recipe's prose riders use lowercase
# "never" while the step-4 / EXTEND-pass deltas use "NEVER").
_BAN_VOCAB = ("never", "no-op", "ban")

# The retired prescriptive update form (pre-#1927 EXTEND-pass step-6 delta).
_RETIRED_FORM = "gh gist edit <gist-id> docs/methodology/issue_<N>.md"


def test_canonical_gist_update_recipe_pinned():
    # The PATCH form — the ONE verified update path.
    assert 'gh api -X PATCH "gists/$GIST_ID"' in _TEXT, (
        "canonical gh api -X PATCH gist-update form missing from SKILL.md"
    )
    # The failure-reason capture shape (PATCH stderr into the substitution) —
    # dropping it silently loses the "PATCH failed: ..." vs "verify mismatch"
    # failure-reason split in the step-9 marker note.
    assert "PATCH_ERR=$(gh api -X PATCH" in _TEXT, (
        "PATCH_ERR error-capture fragment missing from the gist-update recipe"
    )
    # The API read-back verify fragment (gist GET content field).
    assert '--jq ".files[\\"$GIST_FILE\\"].content"' in _TEXT, (
        "API read-back verify (--jq content read) missing from the gist-update recipe"
    )
    # The rc=0-is-not-success sentence — the load-bearing lesson of #1769.
    assert "PATCH rc=0 alone is NOT success" in _TEXT, (
        "rc=0-is-not-success sentence missing from the gist-update recipe"
    )


def test_no_prescriptive_gh_gist_edit():
    # The retired prescriptive form must be gone entirely.
    assert _RETIRED_FORM not in _TEXT, f"retired prescriptive form still present: {_RETIRED_FORM!r}"
    # Every remaining `gh gist edit` mention must sit in ban context: the
    # mention's line, or one of its +/-2 neighbor lines, carries ban
    # vocabulary (never / no-op / ban, case-insensitive).
    lines = _TEXT.splitlines()
    mention_lines = [i for i, line in enumerate(lines) if "gh gist edit" in line]
    assert mention_lines, "expected at least one (ban-context) gh gist edit mention"
    for i in mention_lines:
        window = lines[max(0, i - 2) : i + 3]
        window_text = "\n".join(window).lower()
        assert any(tok in window_text for tok in _BAN_VOCAB), (
            f"prescriptive gh gist edit mention without ban vocabulary near line {i + 1}: "
            f"{lines[i]!r}"
        )
