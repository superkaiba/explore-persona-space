"""Durability pins for the #2155 `/issue` SKILL.md step-body split.

`.claude/skills/issue/SKILL.md` is a ROUTER: each of the 20 `### Step` bodies
lives in a companion under `.claude/skills/issue/steps/`, spliced back in by
`tests/issue_skill_source.py` at its `> **Full procedure:**` pointer. These
pins keep the split's three load-bearing invariants from silently rotting:

1. **Pointer <-> companion bijection** — every pointer resolves to a real
   companion, and every companion is pointed to exactly once. A dangling
   pointer means a step body the orchestrator can never load; an orphaned
   companion means prose the composed logical document silently drops, which
   would un-pin every durability test that greps for it (the inverse-#850
   silent-enforcement-drop shape the split was engineered to avoid).
2. **Composition completeness** — the composed logical document contains all
   20 `### Step` headings exactly once, i.e. splice-at-pointer reconstructed
   the document without dropping or duplicating a step region (append-instead-
   of-splice broke 8 region-extraction pins on the first attempt).
3. **Boot-load carve-out** — the `## Companion files` section keeps the
   explicit `steps/` exception ("do NOT read them on first invocation"): a
   future editor folding steps/ back into the read-at-boot list would restore
   the ~381K-token boot load the split exists to remove.

The bijection is checked with the SAME pointer regex the composer parses
(`issue_skill_source._POINTER`), so a pointer-format drift breaks the pin and
the composer together — loudly — instead of the composer silently skipping
pointers it no longer recognizes.
"""

from __future__ import annotations

import re

from tests.issue_skill_source import (
    _POINTER,
    ISSUE_SKILL_MD,
    ISSUE_STEPS_DIR,
    issue_skill_text,
)


def _router_text() -> str:
    return ISSUE_SKILL_MD.read_text(encoding="utf-8")


def _pointed_companions(router: str) -> list[str]:
    """Companion file names named by pointer lines, in document order."""
    out: list[str] = []
    for line in router.split("\n"):
        m = _POINTER.match(line)
        if m is not None:
            out.append(m.group(1))
    return out


def test_pointer_companion_bijection() -> None:
    """Every pointer resolves; every steps/*.md is pointed to exactly once."""
    router = _router_text()
    pointed = _pointed_companions(router)
    assert pointed, (
        "No `> **Full procedure:**` pointer lines found in "
        ".claude/skills/issue/SKILL.md — either the split was reverted (then "
        "delete this pin deliberately) or the pointer format drifted from "
        "issue_skill_source._POINTER (then the composer is broken too)."
    )
    assert len(pointed) == len(set(pointed)), (
        f"Duplicate step pointers in SKILL.md: "
        f"{sorted(n for n in set(pointed) if pointed.count(n) > 1)}"
    )
    missing = [n for n in pointed if not (ISSUE_STEPS_DIR / n).is_file()]
    assert not missing, (
        f"Dangling `> **Full procedure:**` pointers — no such companion file "
        f"under .claude/skills/issue/steps/: {missing}"
    )
    companions = sorted(p.name for p in ISSUE_STEPS_DIR.glob("*.md") if p.is_file())
    orphaned = sorted(set(companions) - set(pointed))
    assert not orphaned, (
        f"Orphaned steps/ companions never spliced into the logical document "
        f"(their prose is invisible to every issue_skill_text() pin): {orphaned}"
    )
    assert sorted(pointed) == companions


def test_composition_contains_all_step_headings_exactly_once() -> None:
    """The composed document carries all 20 `### Step` headings, each once."""
    router = _router_text()
    headings = [ln for ln in router.split("\n") if ln.startswith("### Step")]
    assert len(headings) == 20, (
        f"Expected exactly 20 `### Step` headings in the SKILL.md router, "
        f"found {len(headings)} — a step was added/removed without updating "
        f"the #2155 split bookkeeping (steps/ files + this pin together)."
    )
    assert len(set(headings)) == 20
    composed_lines = issue_skill_text().split("\n")
    bad = {h: composed_lines.count(h) for h in headings if composed_lines.count(h) != 1}
    assert not bad, (
        f"Step headings not appearing exactly once in the composed logical "
        f"document (splice-at-pointer dropped or duplicated a region): {bad}"
    )


def test_companion_files_steps_carve_out_present() -> None:
    """`## Companion files` keeps the steps/ do-NOT-read-at-boot carve-out."""
    router = _router_text()
    assert "## Companion files" in router
    normalized = re.sub(r"\s+", " ", router)
    sentence = (
        "These are the ONE exception to the line above: do NOT read them on first invocation."
    )
    assert sentence in normalized, (
        "The `## Companion files` steps/ carve-out sentence is gone from "
        ".claude/skills/issue/SKILL.md. Without it, 'Read these on first "
        "invocation' covers steps/ too, and a compliant /issue boot re-reads "
        "all 20 step bodies — restoring the ~381K-token load the #2155 split "
        "removed."
    )
    assert re.search(r"- `steps/`", router), (
        "The `## Companion files` section must list `steps/` (the carve-out "
        "needs its bullet to hang on)."
    )
