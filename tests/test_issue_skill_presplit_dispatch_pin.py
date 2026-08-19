"""Pin the #1810 Step 4b pre-split multi-deliverable dispatch clause.

Incident #1775 (2026-07-28): a 7-deliverable implementer build died at the
subagent context ceiling after 139 tool calls / ~63 min; the recovery was
the Step 5b "Autocompact-thrash respawn recipe" micro-scoped split applied
POST-death at ~15 min triage + respawn overhead, even though the split was
derivable at dispatch time from the approved plan's own deliverable count.
Precedent #1090 established the sequential rounds-A/B unit shape
(intermediate unit returns a commit manifest with NO implementation marker;
the final unit posts the marker after the full smoke).

#1810 adds the dispatch-time twin at Step 4b: a known multi-deliverable
build (more than 4 planned code deliverables) is dispatched as sequential
micro-scoped units BY DEFAULT, with an `epm:progress` mid-split breadcrumb
(`pre-split unit k/M complete: ...`) so a resuming session re-dispatches
only the REMAINING units, never the monolith.

These tests pin, against the Step 4b dispatch region of
`.claude/skills/issue/SKILL.md` (between the 4b heading and the
`Brief passed to the implementer:` line):

1. the clause heading is present;
2. the `more than 4` threshold token (case-insensitive);
3. the no-marker-on-intermediate-units contract token;
4. the final-unit-posts-the-marker token (full per-phase smoke + marker);
5. the mid-split resume breadcrumb token (`pre-split unit`).

Prose assertions run on whitespace-NORMALIZED text (the file wraps prose
mid-phrase, so a required phrase can span lines).
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

STEP4B_HEADING = "**4b. Dispatch implementer for the task type.**"
BRIEF_ANCHOR = "Brief passed to the implementer:"
CLAUSE_HEADING = "Pre-split multi-deliverable builds at dispatch"


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-tolerant match)."""
    return re.sub(r"\s+", " ", text)


def _step4b_dispatch_region() -> str:
    """Slice of SKILL.md between the 4b heading and the brief anchor."""
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    text = issue_skill_text()
    start = text.find(STEP4B_HEADING)
    assert start != -1, f"anchor {STEP4B_HEADING!r} not found in SKILL.md"
    end = text.find(BRIEF_ANCHOR, start)
    assert end != -1, f"anchor {BRIEF_ANCHOR!r} not found after the 4b heading"
    return text[start:end]


def test_step4b_presplit_clause_present() -> None:
    """The #1810 pre-split clause lives in the Step 4b dispatch region."""
    region = _norm(_step4b_dispatch_region())
    assert CLAUSE_HEADING in region, (
        "Step 4b lacks the 'Pre-split multi-deliverable builds at dispatch' clause (#1810)"
    )


def test_threshold_token_more_than_four() -> None:
    """The pre-split threshold keys on more than 4 planned code deliverables."""
    region = _norm(_step4b_dispatch_region())
    assert re.search(r"more than 4", region, re.IGNORECASE), (
        "pre-split clause lost the 'more than 4' code-deliverable threshold"
    )


def test_intermediate_units_post_no_marker() -> None:
    """Intermediate units return a commit manifest with NO implementation marker."""
    region = _norm(_step4b_dispatch_region())
    assert "NO implementation marker" in region, (
        "pre-split clause lost the intermediate-units 'NO implementation marker' contract token"
    )


def test_final_unit_posts_the_marker() -> None:
    """Only the final unit runs the full per-phase smoke and posts the marker."""
    region = _norm(_step4b_dispatch_region())
    assert "ONLY the FINAL unit" in region, (
        "pre-split clause lost the final-unit-owns-smoke-and-marker token"
    )
    assert "posts `epm:experiment-implementation` / `epm:results`" in region, (
        "pre-split clause lost the final-unit marker-post (max+1) token"
    )


def test_midsplit_resume_breadcrumb() -> None:
    """The mid-split `epm:progress` breadcrumb token survives for resume scoping."""
    region = _norm(_step4b_dispatch_region())
    assert "pre-split unit" in region, (
        "pre-split clause lost the 'pre-split unit k/M complete' resume breadcrumb token"
    )
