"""Pin the same-issue follow-up loop's mid-round defer/teardown re-park duty (#1321).

The Step 9b § Same-issue follow-up loop holds the parent at
`followups_running` for the whole round and (pre-#1321) named only two
exits: the round-complete re-park to `awaiting_promotion` and the
`blocked` failure exit. A mid-round defer/teardown (wedged or
pathological run torn down, user defer) was a third, unnamed exit — on
#825 (2026-07-15) a pathological fit was torn down with no status
restore, stranding the parent at `followups_running` ~1.4 h until the
tick STALE-REDRIVE backstop re-parked it.

#1321 names the duty in three mirrored surfaces; this test pins all
three so a later rewrite cannot silently drop it:

1. `.claude/skills/issue/SKILL.md` — the step-3 status-hold paragraph's
   "Mid-round defer/teardown is an exit too" duty (teardown FIRST ->
   `set-status <N> awaiting_promotion` -> post the step-4 completion
   marker with `outcome: deferred — <reason>`, closing the label).
2. `.claude/skills/issue/markers.md` — the `epm:same-issue-followup-run`
   row documents the defer posting site + `outcome: deferred`.
3. `.claude/workflow.yaml` — the same marker's `fields:` block carries
   the mirror sentence (markers.md is emit-tables-generated from it).

Asserts are whitespace-normalized (the prose is line-wrapped in
SKILL.md / workflow.yaml), per the pin-family convention of substring
presence checks. Paths resolve via `Path(__file__)` — NEVER
`task_workflow.repo_root()`, which reads the MAIN checkout and would
miss worktree edits pre-merge.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
MARKERS_MD = ROOT / ".claude" / "skills" / "issue" / "markers.md"
WORKFLOW_YAML = ROOT / ".claude" / "workflow.yaml"

DEFER_DUTY_PHRASE = "Mid-round defer/teardown is an exit too"
DEFER_OUTCOME = "outcome: deferred"
LOOP_ANCHOR = "Same-issue follow-up loop (`question_relation: same`)"
RUN_MARKER_KIND = "epm:same-issue-followup-run"


def _normalized(text: str) -> str:
    """Collapse all whitespace to single spaces (wrap-tolerant substring checks)."""
    return " ".join(text.split())


def test_defer_teardown_repark_duty_pinned():
    """All three surfaces carry the #1321 mid-round defer -> re-park duty."""
    # (a) SKILL.md § Same-issue follow-up loop names the defer exit + the
    # deferred outcome, inside the loop section (after its anchor).
    skill = ISSUE_SKILL.read_text()
    skill_norm = _normalized(skill)
    anchor_idx = skill_norm.find(_normalized(LOOP_ANCHOR))
    assert anchor_idx != -1, (
        f"SKILL.md lost the loop anchor {LOOP_ANCHOR!r}; if the section was "
        "renamed, update this pin alongside it."
    )
    loop_region = skill_norm[anchor_idx:]
    assert DEFER_DUTY_PHRASE in loop_region, (
        f"SKILL.md § Same-issue follow-up loop must carry the {DEFER_DUTY_PHRASE!r} "
        "duty (teardown FIRST, re-park to awaiting_promotion NEXT, then the "
        "deferred-outcome run marker) — dropping it re-opens the #825 stranded "
        "followups_running gap (~1.4 h until the tick backstop)."
    )
    assert DEFER_OUTCOME in loop_region, (
        f"SKILL.md § Same-issue follow-up loop must name the {DEFER_OUTCOME!r} "
        "run-marker outcome that closes the round's label on a mid-round defer."
    )

    # (b) markers.md: the epm:same-issue-followup-run row documents the
    # defer posting site + deferred outcome.
    markers = _normalized(MARKERS_MD.read_text())
    row_idx = markers.find(f"| `{RUN_MARKER_KIND}` |")
    assert row_idx != -1, f"markers.md lost the `{RUN_MARKER_KIND}` row"
    row_end = markers.find("| `epm:", row_idx + 1)
    row = markers[row_idx : row_end if row_end != -1 else len(markers)]
    assert DEFER_OUTCOME in row, (
        f"markers.md `{RUN_MARKER_KIND}` row must document the mid-round "
        f"defer/teardown posting site with {DEFER_OUTCOME!r} (re-run "
        "`workflow_lint.py --emit-tables` after editing workflow.yaml)."
    )

    # (c) workflow.yaml: the same marker's fields block carries the mirror.
    wf = _normalized(WORKFLOW_YAML.read_text())
    kind_idx = wf.find(f'kind: "{RUN_MARKER_KIND}"')
    assert kind_idx != -1, f"workflow.yaml lost the `{RUN_MARKER_KIND}` marker entry"
    next_kind = wf.find('- kind: "epm:', kind_idx + 1)
    fields_block = wf[kind_idx : next_kind if next_kind != -1 else len(wf)]
    assert DEFER_OUTCOME in fields_block, (
        f"workflow.yaml `{RUN_MARKER_KIND}` fields block must carry the "
        f"mid-round defer mirror sentence with {DEFER_OUTCOME!r}."
    )
