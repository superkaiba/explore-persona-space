"""Prose durability pin for the inline-round pod upload-verify recipe (#1970).

Incident #1773 (session 0ac15c23): an inline round that had ALREADY verified
its uploads could not terminate its pod through a sanctioned path — the
terminate guard's refusal message named only "/issue <N> Step 8" and
``--skip-upload-verify``, so the round improvised the blunt override; and the
round's per-issue upload-verify script silently scoped verification to one HF
prefix, leaving ``raw_windows`` (8h50m of GPU output) verified only by an
ad-hoc hand set-diff.

Task #1970 documented the sanctioned recipe — verify THIS round's artifacts,
post ``epm:upload-verification`` with a note LEADING
``Verdict: PASS — inline-round verification; prefixes: <every verified
prefix>`` via ``task.py post-marker``, then terminate — plus the
enumerate-ALL-HF-prefixes duty, at BOTH prose surfaces the carve-out reads:
CLAUDE.md's "Completion-side teardown (no ask-gate)" clause and its executing
mirror in `/issue` SKILL.md Step 9a-ter. This test pins stable SHORT tokens
(never full-sentence byte pins) so a later edit cannot silently drop the
recipe and steer verified inline rounds back to ``--skip-upload-verify``.
Follows the whitespace-normalize family pattern of
``tests/test_suffixed_pod_completion_teardown_pin.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = ROOT / "CLAUDE.md"
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

# The CLAUDE.md carve-out clause is a single-line paragraph located by its
# distinct bold header (same locator as the #1662 pin test).
CARVEOUT_HEADER = "**Completion-side teardown (no ask-gate):**"

# Stable short tokens: (i) the recipe, (ii) the all-prefixes duty.
RECIPE_TOKENS = (
    "epm:upload-verification",
    "Verdict: PASS — inline-round verification",
)
DUTY_TOKEN = "enumerate ALL HF prefixes"


def _norm(text: str) -> str:
    """Collapse all whitespace to single spaces so tokens match across the
    markdown soft line breaks the SKILL.md wrapper introduces."""
    return re.sub(r"\s+", " ", text)


def test_claude_md_carveout_carries_inline_upload_verify_recipe():
    """CLAUDE.md's completion-side teardown clause (the user-chat inline
    carve-out) carries the recipe tokens + the all-prefixes duty token on
    its single-line paragraph."""
    lines = CLAUDE_MD.read_text(encoding="utf-8").splitlines()
    hits = [line for line in lines if CARVEOUT_HEADER in line]
    assert len(hits) == 1, (
        f"expected exactly one CLAUDE.md line carrying {CARVEOUT_HEADER!r}, found {len(hits)}"
    )
    clause = hits[0]
    for token in (*RECIPE_TOKENS, DUTY_TOKEN):
        assert token in clause, (
            f"CLAUDE.md completion-side teardown clause must carry {token!r} "
            "(#1970: the sanctioned inline-round verify-then-terminate recipe "
            "+ the enumerate-ALL-HF-prefixes duty; dropping it steers verified "
            "inline rounds back to --skip-upload-verify, #1773)"
        )


def test_issue_skill_9a_ter_carries_inline_upload_verify_recipe():
    """SKILL.md Step 9a-ter's completion-side teardown block mirrors the
    recipe + duty (whole-file whitespace-normalized — the SKILL.md wrapper
    soft-wraps the recipe across lines)."""
    norm = _norm(issue_skill_text())
    for token in (*RECIPE_TOKENS, DUTY_TOKEN):
        assert token in norm, (
            f"SKILL.md Step 9a-ter must carry the inline upload-verify recipe "
            f"token {token!r} (#1970: the 9a-ter block is the executing mirror "
            "of the CLAUDE.md carve-out clause)"
        )
