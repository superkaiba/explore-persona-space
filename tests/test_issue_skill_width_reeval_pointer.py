"""Pin the #1346 width-re-evaluation prose across its four surfaces.

The mid-run compute-deviation negative-signature branch gained a width
re-evaluation step (#1346): on an embarrassingly-parallel unit grid with
checkpoint/restore live (or a relaunch already happening), the relaunch
decision must evaluate re-sharding the REMAINING units across a wider
fleet before `continue_as_is` at the original width. These tests pin the
prose so a future rewrite cannot silently drop it (the droppable
protection-prose class — #1134/#1045/#884 lineage).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDTH_RE = re.compile(r"width re-?evaluation", re.IGNORECASE)


def test_eta_advisory_names_width_reevaluation():
    """SKILL.md Step 6d.2 ETA-advisory paragraph keeps the width clause."""
    text = (ROOT / ".claude/skills/issue/SKILL.md").read_text()
    anchor = text.index("ETA-deviation / GPU-width advisory handling")
    para = text[anchor : anchor + 3000]
    assert WIDTH_RE.search(para), (
        "SKILL.md Step 6d.2 ETA-advisory paragraph lost the #1346 width-re-evaluation clause"
    )


def test_midrun_trigger_carries_width_block():
    """vectorize-many-cell-fits.md § Mid-run trigger keeps the width block."""
    text = (ROOT / ".claude/rules/vectorize-many-cell-fits.md").read_text()
    anchor = text.index("### Mid-run trigger")
    section = text[anchor:]
    assert WIDTH_RE.search(section) and "embarrassingly-parallel" in section, (
        "vectorize-many-cell-fits.md § Mid-run trigger lost the #1346 width-re-evaluation block"
    )
    assert "width_reeval:" in section, (
        "vectorize-many-cell-fits.md § Mid-run trigger lost the width_reeval: recording slot"
    )


def test_marker_declaration_carries_width_fields():
    """workflow.yaml epm:compute-deviation DECLARATION keeps the width note fields.

    Scoped to the declaration region (the unique `kind: "epm:compute-deviation"`
    key at the marker declaration, ~line 1690) so a retained Edit C
    (pivot_criteria, which PRECEDES the declaration in the file, ~line 806-867)
    cannot false-PASS a dropped Edit E. Verified 2026-07-15: the exact string
    `kind: "epm:compute-deviation"` occurs once, at the declaration; the
    forward slice cannot reach the pivot_criteria text.
    """
    text = (ROOT / ".claude/workflow.yaml").read_text()
    assert text.count('kind: "epm:compute-deviation"') == 1, (
        "declaration anchor no longer unique — re-verify the slice scoping"
    )
    anchor = text.index('kind: "epm:compute-deviation"')
    decl = text[anchor : anchor + 3000]
    assert "reshard_width_<K>" in decl and "width_reeval:" in decl, (
        "workflow.yaml epm:compute-deviation declaration lost the #1346 "
        "width_reeval: / reshard_width_<K> optional note fields"
    )


def test_crash_fix_relaunch_carries_width_pointer():
    """crash-fix-rounds.md keeps the relaunch-side width pointer.

    crash-fix-rounds.md is the only edited surface not otherwise pinned.
    """
    text = (ROOT / ".claude/rules/crash-fix-rounds.md").read_text()
    assert WIDTH_RE.search(text) and "width_reeval:" in text, (
        "crash-fix-rounds.md lost the #1346 relaunch-side width-re-evaluation pointer"
    )
