"""Pin the #1458 disclosure clauses of the ad-hoc results-summary rule.

CLAUDE.md's "Ad-hoc results summaries state per-arm provenance in their
setup line" bullet gained two clauses from task #1458:

1. Display-substitution disclosure (#1345: an examples dashboard silently
   display-substituted 'ARIA'->'Assistant' in text presented as model
   generations; the user had to ask whether the story really said ARIA).
2. Matched-target disclosure for cross-arm tables (#1092: a chat table
   compared prefix-R2 vs context-R2 arms scored against DIFFERENT
   targets; the user had to flag the mismatch himself).

The bullet is the most-amended long bullet in CLAUDE.md (#1139 -> #1389 ->
#1458); these pins keep a future rewrite from silently dropping either
clause. Assertions run on whitespace-NORMALIZED file text so prose
re-wrapping never breaks a pin; each token is a verbatim substring of the
rule in substance. The citation pins use the colon-bearing forms
"(#1345:" / "(#1092:" because the bare ids pre-exist elsewhere in
CLAUDE.md (a bare-id pin would be vacuous).

Task #1539 added a third pinned clause: the live-compute scan for never-run
claims (test_live_compute_scan_clause_pinned).

Task #1623 added a fourth pinned clause: similarity-statistic semantics for
operator/map comparisons (test_similarity_statistic_semantics_clause_pinned).

Task #2111 extended the "Compose-time re-grep" clause to ops/fleet
statistics quoted in monitoring turns (recomputed from events.jsonl / live
state, never carried from a prior cycle); pinned by
test_ops_stats_regrep_clause_pinned.
"""

from __future__ import annotations

import re
from pathlib import Path

CLAUDE_MD = Path(__file__).resolve().parent.parent / "CLAUDE.md"


def _normalized() -> str:
    return re.sub(r"\s+", " ", CLAUDE_MD.read_text(encoding="utf-8"))


def test_display_substitution_disclosure_clause_pinned() -> None:
    text = _normalized()
    assert "**Display-substitution disclosure:**" in text
    assert "disclosed inline, per passage" in text
    assert "dashboards and HTML artifacts included" in text
    assert "(#1345:" in text


def test_matched_target_disclosure_clause_pinned() -> None:
    text = _normalized()
    assert "**Matched-target disclosure for cross-arm tables:**" in text
    assert "scored against the SAME target/corpus" in text
    assert "(#1092:" in text


def test_live_compute_scan_clause_pinned() -> None:
    text = _normalized()
    assert "additionally scans LIVE compute before asserting" in text
    assert "the same follow-up signal set the watcher's pod-safety pass reads" in text
    assert "nothing live is generating that cell" in text


def test_ops_stats_regrep_clause_pinned() -> None:
    text = _normalized()
    assert "**Compose-time re-grep (not only on challenge):**" in text
    assert "any ops/fleet statistic (failure rates, fixed-vs-not attributions)" in text
    assert "recomputed from events.jsonl / live state" in text
    assert "never carried from a prior cycle" in text
    # Citation pin (colon-bearing distinctive form, per the module docstring).
    assert "ops stats: #2111" in text


def test_similarity_statistic_semantics_clause_pinned() -> None:
    text = _normalized()
    assert "**Similarity-statistic semantics:**" in text
    assert "direction-aware" in text
    assert "spectrum/rotation-invariant-only" in text
    assert "issue1345_operator_comparison.py" in text
    assert "(#1310:" in text
