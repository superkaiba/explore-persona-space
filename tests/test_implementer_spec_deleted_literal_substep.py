"""Pin (#1744, incident #1723): the deleted/moved-literal pin-sweep is its
OWN numbered sub-step in both implementer specs (implementer.md item 1a,
experiment-implementer.md step 2b2), not a mid-paragraph sentence — and the
After-implementation ordered-list labels stay UNIQUE per spec (the #1744
round-1 Must-Fix collision class: a proposed `1b.` id collided with the
existing `1b.` item)."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTER = REPO_ROOT / ".claude/agents/implementer.md"
EXP_IMPLEMENTER = REPO_ROOT / ".claude/agents/experiment-implementer.md"
SPECS = [IMPLEMENTER, EXP_IMPLEMENTER]

# (spec path, section start marker, section end marker) — the After-
# implementation checklist region of each spec (same anchors as
# tests/test_issue_skill_gate_scope_brief_pin.py's region slices).
SECTIONS = [
    (
        IMPLEMENTER,
        "### After Implementation",
        "### Local runs are same-turn, synchronous work",
    ),
    (
        EXP_IMPLEMENTER,
        "### After implementation (mandatory checklist)",
        "### Smoke runs are same-turn, synchronous work",
    ),
]

_LABEL_RE = re.compile(r"^(\d+[a-z]?\d*)\.", re.MULTILINE)
_FENCE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)


def _section(path: Path, start_marker: str, end_marker: str) -> str:
    """Slice the spec text between two unique section headings, fail-loud."""
    text = path.read_text(encoding="utf-8")
    start = text.find(start_marker)
    end = text.find(end_marker)
    assert start != -1, f"{path.name}: start marker not found: {start_marker!r}"
    assert end != -1, f"{path.name}: end marker not found: {end_marker!r}"
    assert start < end, f"{path.name}: start marker must precede end marker"
    return text[start:end]


def test_specs_carry_deleted_moved_literal_grep_as_own_substep():
    """Both specs state the duty imperatively (casefolded phrase pins)."""
    for p in SPECS:
        low = p.read_text(encoding="utf-8").lower()
        assert "deletes or moves" in low, (
            f"{p.name}: the deleted/moved-literal grep sub-step must state the "
            "'deletes or moves' trigger (#1744; incident #1723)."
        )
        assert "old and new form" in low, (
            f"{p.name}: the sub-step must mandate grepping the OLD and NEW form "
            "of each changed literal (#1699/#1744)."
        )


def test_after_implementation_list_labels_are_unique_per_spec():
    """No duplicate ordered-list label in either After-implementation section.

    Mechanizes the #1744 round-1 Must-Fix collision class (a duplicate `1b.`)
    so a future duplicate label fails loud. Fenced code blocks are stripped
    before collection so an example snippet cannot shadow a real label.
    """
    for path, start_marker, end_marker in SECTIONS:
        section = _FENCE_RE.sub("", _section(path, start_marker, end_marker))
        labels = _LABEL_RE.findall(section)
        assert labels, f"{path.name}: no ordered-list labels found in section"
        dupes = sorted({x for x in labels if labels.count(x) > 1})
        assert len(labels) == len(set(labels)), (
            f"{path.name}: duplicate After-implementation list label(s) {dupes} "
            "— pick a fresh sub-step id (the #1744 Must-Fix-1 collision class)."
        )
