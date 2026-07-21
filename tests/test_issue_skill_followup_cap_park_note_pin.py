"""Pin the Step 9a-ter cap-park surfacing note for screened free-analysis follow-ups (#1548).

The Step 9a-ter zero-GPU floor caps at AT MOST ONE free-analysis
follow-up run per task. Pre-#1548, a follow-up the cap excluded survived
ONLY as a body bullet — no marker, no PM-surfaceable signal (incident
#958: a top-ranked, follow-up-critic-screened `not-redundant` 0-GPU
follow-up sat unrun as a body bullet for 13 days until the user found
and kicked it himself).

#1548 prescribes a structured, idempotent `epm:progress` note (fixed
leading token `followup-parked-by-cap`) at BOTH moments the cap parks a
concrete unrun `cost_class: free-analysis` entry: (a) a loop-guard
re-entry exit with a non-empty detection union, (b) the non-selected
surplus immediately after the `epm:free-analysis-followup-run` marker
posts (run or abort — Auto-run procedure step 6). This test pins the
prescription across three mirrored surfaces so a later rewrite cannot
silently drop it:

1. `.claude/skills/issue/SKILL.md` — the Loop guard's § Cap-park
   surfacing sub-block (token, fields, both firing moments, per-title
   idempotency, cap-raise alternative) with the one-round cap language
   retained, plus the procedure step-6 surplus posting reminder.
2. `.claude/workflow.yaml` — the `epm:free-analysis-followup-run`
   `fields:` block carries the mirror clause (markers.md is
   emit-tables-generated from it).
3. `.claude/skills/issue/markers.md` — the generated row carries the
   token (re-run `workflow_lint.py --emit-tables` after editing
   workflow.yaml).

A third test pins the DESIGN DECISION that the note reuses the
registered `epm:progress` kind: `.claude/workflow.yaml` must NOT gain a
`kind: epm:followup-parked-by-cap` marker registration (minting a kind
is a marker-schema change requiring architectural greenlight).

#1558 extends the same note duty to the Step 9b cheap-band round-cap
(C2) park path (`cost_class=needs-gpu`): two further tests pin the
block-C2 § Cheap-band cap-park surfacing prescription in SKILL.md (with
the 2-round cap language retained — the change is SURFACING only) and
the C2 mirror clause in the `epm:same-issue-followup-run` fields block
of workflow.yaml + the generated markers.md row.

Asserts are whitespace-normalized (the prose is line-wrapped in
SKILL.md / workflow.yaml), per the pin-family convention of substring
presence checks (mirrors tests/test_issue_skill_followup_defer_repark.py).
Paths resolve via `Path(__file__)` — NEVER `task_workflow.repo_root()`,
which reads the MAIN checkout and would miss worktree edits pre-merge.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
MARKERS_MD = ROOT / ".claude" / "skills" / "issue" / "markers.md"
WORKFLOW_YAML = ROOT / ".claude" / "workflow.yaml"

PARK_TOKEN = "followup-parked-by-cap"
LOOP_GUARD_ANCHOR = "**Loop guard (critical).**"
RUN_MARKER_KIND = "epm:free-analysis-followup-run"
CAP_PHRASE = "AT MOST ONE free-analysis follow-up run per task"
ALTERNATIVE_TOKEN = "raise-9a-ter-cap-or-manual-pickup"
STEP6_SURPLUS_PHRASE = "the cap is consumed as of this marker"

# --- #1558: Step 9b cheap-band (C2) extension ---
C2_ANCHOR = "**Cheap-band round cap.**"
C2_ALTERNATIVE_TOKEN = "raise-9b-cheap-cap-or-manual-pickup"
CHEAP_RUN_SOURCE = "proposer-9b-cheap"
C2_FINAL_SLOT_PHRASE = "consumes the final cheap-band cap slot"
C2_CAP_PHRASE = "At most **2** cheap-band auto-run rounds per task"
SAME_ISSUE_RUN_MARKER_KIND = "epm:same-issue-followup-run"


def _normalized(text: str) -> str:
    """Collapse all whitespace to single spaces (wrap-tolerant substring checks)."""
    return " ".join(text.split())


def test_cap_park_note_pinned_in_skill():
    """SKILL.md's Loop guard span prescribes the cap-park surfacing note.

    Scoped to the text from the `**Loop guard (critical).**` anchor
    onward (the § Cap-park surfacing sub-block lives directly under it;
    the step-6 surplus reminder lives later in the same 9a-ter section).
    """
    skill_norm = _normalized(ISSUE_SKILL.read_text())
    anchor_idx = skill_norm.find(_normalized(LOOP_GUARD_ANCHOR))
    assert anchor_idx != -1, (
        f"SKILL.md lost the {LOOP_GUARD_ANCHOR!r} anchor; if the paragraph was "
        "renamed, update this pin alongside it."
    )
    guard_region = skill_norm[anchor_idx:]
    required = [
        # The one-round cap language must survive VERBATIM in intent —
        # the #1548 change is SURFACING only, never a cap-semantics change.
        CAP_PHRASE,
        # The note itself: fixed leading token, epm:progress reuse, the
        # verbatim-title idempotency key field.
        PARK_TOKEN,
        "epm:progress",
        "followup_ref=",
        # Moment-(b) firing clause + the cap-consumer field (deleting the
        # post-run-surplus moment or the field fails this pin).
        "Two firing moments",
        "cap_consumed_by=",
        # Per-(task, verbatim title) idempotency: present => skip; re-entries
        # never double-post.
        "present ⇒ skip",
        "never double-post",
        # The cap-raise alternative named for a future planner/human.
        ALTERNATIVE_TOKEN,
        # The Edit-2 reminder: the non-selected surplus posts right after the
        # run marker (run or ABORT alike), not at some future re-entry.
        STEP6_SURPLUS_PHRASE,
    ]
    for token in required:
        assert token in guard_region, (
            f"SKILL.md Step 9a-ter (after the Loop guard anchor) must carry "
            f"{token!r} — the #1548 cap-park surfacing note prescription "
            "(token, fields, both firing moments, idempotency, cap language) "
            "must not be silently dropped (incident #958: a screened "
            "not-redundant free-analysis follow-up sat invisible for 13 days)."
        )


def test_cap_park_note_pinned_in_workflow_yaml_and_markers_md():
    """workflow.yaml's run-marker fields block + the generated markers.md
    row both carry the cap-park token (markers.md is regenerated via
    `workflow_lint.py --emit-tables`, never hand-edited)."""
    wf = _normalized(WORKFLOW_YAML.read_text())
    kind_idx = wf.find(f'kind: "{RUN_MARKER_KIND}"')
    assert kind_idx != -1, f"workflow.yaml lost the `{RUN_MARKER_KIND}` marker entry"
    next_kind = wf.find('- kind: "epm:', kind_idx + 1)
    fields_block = wf[kind_idx : next_kind if next_kind != -1 else len(wf)]
    assert PARK_TOKEN in fields_block, (
        f"workflow.yaml `{RUN_MARKER_KIND}` fields block must carry the "
        f"{PARK_TOKEN!r} surfacing clause (#1548)."
    )

    markers = _normalized(MARKERS_MD.read_text())
    row_idx = markers.find(f"| `{RUN_MARKER_KIND}` |")
    assert row_idx != -1, f"markers.md lost the `{RUN_MARKER_KIND}` row"
    row_end = markers.find("| `epm:", row_idx + 1)
    row = markers[row_idx : row_end if row_end != -1 else len(markers)]
    assert PARK_TOKEN in row, (
        f"markers.md `{RUN_MARKER_KIND}` row must carry {PARK_TOKEN!r} — "
        "re-run `workflow_lint.py --emit-tables` after editing workflow.yaml "
        "(the table is generated, never hand-edited)."
    )


def test_cheap_band_cap_park_note_pinned_in_skill():
    """SKILL.md's Step 9b C2 span prescribes the cheap-band cap-park note (#1558).

    Scoped to the text from the `**Cheap-band round cap.**` anchor
    onward (the § Cheap-band cap-park surfacing sub-block lives directly
    under it; the loop step-4 reminder + step-5 parenthetical live later
    in the same Step 9b section).
    """
    skill_norm = _normalized(ISSUE_SKILL.read_text())
    anchor_idx = skill_norm.find(_normalized(C2_ANCHOR))
    assert anchor_idx != -1, (
        f"SKILL.md lost the {C2_ANCHOR!r} anchor; if the C2 bullet was "
        "renamed, update this pin alongside it."
    )
    c2_region = skill_norm[anchor_idx:]
    required = [
        # The 2-round cap language must survive VERBATIM in intent — the
        # #1558 change is SURFACING only, never a cap-semantics change.
        C2_CAP_PHRASE,
        # The note itself: fixed leading token, epm:progress reuse, the
        # verbatim-title idempotency key field, the C2-keyed fields.
        PARK_TOKEN,
        "epm:progress",
        "followup_ref=",
        "cost_class=needs-gpu",
        "cap_consumed_by=",
        # The C2-keyed cap-raise alternative (NOT the 9a-ter token — it
        # must name the cheap-band cap as the one to raise).
        C2_ALTERNATIVE_TOKEN,
        # The C2 cap-counting key the note's moment (b) fires on.
        CHEAP_RUN_SOURCE,
        # Moment (b) + the loop step-4 reminder both pin on this phrase.
        C2_FINAL_SLOT_PHRASE,
        # The shared-contract reference back to the 9a-ter sub-block.
        "Cap-park surfacing",
    ]
    for token in required:
        assert token in c2_region, (
            f"SKILL.md Step 9b (after the C2 anchor) must carry {token!r} — "
            "the #1558 cheap-band cap-park surfacing prescription (token, "
            "C2-keyed fields, both firing moments, cap language) must not "
            "be silently dropped (same bullet-only invisibility class as "
            "incident #958)."
        )


def test_cheap_band_cap_park_mirror_in_workflow_yaml_and_markers_md():
    """workflow.yaml's same-issue-run fields block + the generated
    markers.md row both carry the cap-park token (#1558; markers.md is
    regenerated via `workflow_lint.py --emit-tables`, never hand-edited)."""
    wf = _normalized(WORKFLOW_YAML.read_text())
    kind_idx = wf.find(f'kind: "{SAME_ISSUE_RUN_MARKER_KIND}"')
    assert kind_idx != -1, f"workflow.yaml lost the `{SAME_ISSUE_RUN_MARKER_KIND}` marker entry"
    next_kind = wf.find('- kind: "epm:', kind_idx + 1)
    fields_block = wf[kind_idx : next_kind if next_kind != -1 else len(wf)]
    assert PARK_TOKEN in fields_block, (
        f"workflow.yaml `{SAME_ISSUE_RUN_MARKER_KIND}` fields block must carry "
        f"the {PARK_TOKEN!r} surfacing clause (#1558)."
    )

    markers = _normalized(MARKERS_MD.read_text())
    row_idx = markers.find(f"| `{SAME_ISSUE_RUN_MARKER_KIND}` |")
    assert row_idx != -1, f"markers.md lost the `{SAME_ISSUE_RUN_MARKER_KIND}` row"
    row_end = markers.find("| `epm:", row_idx + 1)
    row = markers[row_idx : row_end if row_end != -1 else len(markers)]
    assert PARK_TOKEN in row, (
        f"markers.md `{SAME_ISSUE_RUN_MARKER_KIND}` row must carry {PARK_TOKEN!r} — "
        "re-run `workflow_lint.py --emit-tables` after editing workflow.yaml "
        "(the table is generated, never hand-edited)."
    )


def test_no_new_marker_kind_minted():
    """The cap-park note reuses `epm:progress` — workflow.yaml must NOT
    register a new `epm:followup-parked-by-cap` marker KIND (a
    marker-schema change requiring architectural greenlight). Checks
    both the quoted and unquoted YAML forms."""
    wf = _normalized(WORKFLOW_YAML.read_text())
    banned = [
        f'kind: "epm:{PARK_TOKEN}"',
        f"kind: epm:{PARK_TOKEN}",
    ]
    for form in banned:
        assert form not in wf, (
            f"workflow.yaml registers {form!r} — the #1548 design reuses the "
            "registered `epm:progress` kind; minting a new marker kind is an "
            "architectural (marker-schema) change requiring user greenlight."
        )
