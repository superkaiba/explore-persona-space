"""Durability pin for the #1902 composition trigger in the #1810 pre-split
clause (`.claude/skills/issue/SKILL.md`).

Task #1976 (workflow-fix) added a MANDATORY composition trigger to the
existing count-based pre-split clause. A future editor rewriting SKILL.md
must not silently drop it — that would re-open the #1902 shape (unit
combining fits + figures + a heavy smoke phase silently passing the
deliverable-count trigger, then dying at the subagent context ceiling).

The literal sentinel `"Composition trigger (mandatory, #1902 shape)"` is
distinctive enough to bind exactly; the substantive predicate tokens are
matched permissively via regex so a functionally-identical reword
(`>=2 phases` / `two phases` / `at least 2 pipeline phases`) does not
false-fail this pin.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

# Whitespace-normalization for prose greps — the SKILL.md source wraps at ~65
# chars, so the sentinel phrase spans a linebreak; collapsing runs of ASCII
# whitespace to a single space lets the substring/regex assertions bind on
# semantic content, not exact source-wrap bytes.
_WS_RUN = re.compile(r"\s+")


def _normalized_text() -> str:
    return _WS_RUN.sub(" ", SKILL.read_text(encoding="utf-8"))


def test_pre_split_composition_trigger_present() -> None:
    text = _normalized_text()
    assert "Composition trigger (mandatory, #1902 shape)" in text, (
        "The #1902 composition trigger sentence must remain in the #1810 "
        "pre-split clause of .claude/skills/issue/SKILL.md; see task #1976."
    )


def test_pre_split_composition_trigger_names_fit_figures_smoke() -> None:
    """The trigger's substantive predicate — the three deliverable classes
    (fit / figure / smoke), the ≥2-pipeline-phase smoke qualifier, and the
    ≤1-companion cap for the smoke-bearing unit — must all remain named.

    Token literals stay exact for `fit`, `figure`, `smoke`, and the
    `AT MOST ONE` companion-cap phrase (distinctive enough to bind).
    The ≥2-phase qualifier uses a permissive regex so a functionally-identical
    reword ("at least 2 phases" / "two phases" / ">= 2 pipeline phases")
    does not false-fail this pin — the pin's job is to make silent semantic
    reverts SURFACE, not to lock exact prose bytes.
    """
    text = _normalized_text()
    for token in ("fit", "figure", "smoke", "AT MOST ONE"):
        assert token in text, f"Composition trigger dropped literal token: {token!r}"

    two_phase_re = re.compile(
        r"(?:≥\s*2|>=\s*2|at\s+least\s+2|two)\s*(?:pipeline\s*)?phases",
        flags=re.IGNORECASE,
    )
    assert two_phase_re.search(text) is not None, (
        "Composition trigger dropped the >=2-pipeline-phase smoke qualifier "
        "(any of '≥2 pipeline phases' / '>=2 phases' / 'at least 2 phases' "
        "/ 'two phases' accepted)."
    )
