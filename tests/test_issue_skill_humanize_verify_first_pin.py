"""Content-invariant pins for the #1860 verify-candidate-first apply ordering.

Task #1860 reordered the two orchestrator-level `task.py set-body` apply
sites in `.claude/skills/issue/SKILL.md` from apply-then-verify to
VERIFY-CANDIDATE-FIRST: the candidate body FILE is verified with
`verify_task_body.py --file` (main-checkout copy) BEFORE `task.py set-body`
replaces the live body, and the apply is gated on that PASS; a post-apply
`verify_task_body.py --issue <N>` confirm is RETAINED (it covers the
frontmatter-coupled checks `--file` cannot see — e.g. the #1110
H1==frontmatter-title check, the kind exit-3 short-circuit, concerns-audit).
The two sites: (1) the Step 9a-humanize procedure step 4 (the humanize
loop's candidate at `/tmp/issue-<N>-humanize-loop.md`), and (2) the
Step 9a-bis "Procedural-only verdict strip" item (b) (the critic's
procedural fixes staged to a candidate copy).

REGION-SCOPED per the `test_issue_skill_*_pin.py` convention (fail-loud
anchors — assert the anchors resolve, never silently slice to empty).

Origin incident: #1775 (mined by /daily 2026-07-29) — the pre-#1860
apply-then-verify order applied a humanize-loop candidate that then FAILed
the verifier (v4 conciseness), leaving a briefly-live non-compliant body
and costing an extra edit/apply/verify cycle with revert-after-apply as the
documented recovery.
"""

from __future__ import annotations

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

_HUMANIZE_CANDIDATE_VERIFY = "verify_task_body.py --file /tmp/issue-<N>-humanize-loop.md"
_HUMANIZE_APPLY = "set-body <N> --file /tmp/issue-<N>-humanize-loop.md"
_POST_APPLY_ISSUE_CONFIRM = "verify_task_body.py --issue <N>"
_STRIP_CANDIDATE_VERIFY = "verify_task_body.py --file"
_STRIP_APPLY = "task.py set-body <N> --file ..."


def _skill_text() -> str:
    return issue_skill_text()


def _region(text: str, start_marker: str, end_marker: str) -> str:
    """Slice [start_marker, end_marker) with fail-loud anchor asserts."""
    start = text.find(start_marker)
    end = text.find(end_marker)
    assert start != -1, f"start anchor not found in SKILL.md: {start_marker!r}"
    assert end != -1, f"end anchor not found in SKILL.md: {end_marker!r}"
    assert start < end, f"start anchor must precede end anchor: {start_marker!r}"
    return text[start:end]


def _humanize_region(text: str) -> str:
    """The Step 9a-humanize slice (the humanize loop's home)."""
    return _region(text, "**9a-humanize.", "**9a-ter.")


def _strip_region(text: str) -> str:
    """The Step 9a-bis procedural-only verdict strip slice."""
    return _region(text, "**Procedural-only verdict strip", "**If REVISE (rounds 2-5):**")


def test_humanize_candidate_verify_precedes_set_body():
    """Pin 1 (#1860 Edit 1): in the 9a-humanize step-4 block, the CANDIDATE
    file verify (`verify_task_body.py --file /tmp/issue-<N>-humanize-loop.md`)
    precedes the `task.py set-body` apply, and the RETAINED post-apply
    `verify_task_body.py --issue <N>` confirm follows the apply. A later
    edit that reverts to apply-then-verify (the #1775 order) — or drops the
    post-apply frontmatter-coupled confirm — fails this pin."""
    region = _humanize_region(_skill_text())
    verify_idx = region.find(_HUMANIZE_CANDIDATE_VERIFY)
    apply_idx = region.find(_HUMANIZE_APPLY)
    assert verify_idx != -1, (
        "the 9a-humanize region must verify the candidate FILE first: "
        f"{_HUMANIZE_CANDIDATE_VERIFY!r} not found"
    )
    assert apply_idx != -1, (
        f"the 9a-humanize region must apply the candidate via {_HUMANIZE_APPLY!r}"
    )
    assert verify_idx < apply_idx, (
        "verify-candidate-first (#1860): the candidate --file verify must "
        "PRECEDE the set-body apply (the pre-#1860 apply-then-verify order "
        "left a briefly-live non-compliant body — incident #1775)"
    )
    post_apply_idx = region.find(_POST_APPLY_ISSUE_CONFIRM, apply_idx)
    assert post_apply_idx != -1, (
        "the post-apply `verify_task_body.py --issue <N>` confirm must be "
        "RETAINED after the set-body apply (frontmatter-coupled checks "
        "--file cannot see)"
    )


def test_procedural_strip_verify_first():
    """Pin 2 (#1860 Edit 2): in the 9a-bis procedural-only verdict strip,
    item (b) verifies the staged CANDIDATE copy (`verify_task_body.py
    --file`) BEFORE the `task.py set-body <N> --file ...` apply, and the
    post-apply `verify_task_body.py --issue <N>` re-run follows the apply
    (critic Must-Fix: the `--issue`-side coverage the `--file` candidate
    check cannot see must not be dropped)."""
    region = _strip_region(_skill_text())
    verify_idx = region.find(_STRIP_CANDIDATE_VERIFY)
    apply_idx = region.find(_STRIP_APPLY)
    assert verify_idx != -1, (
        "the strip region must verify the staged candidate copy: "
        f"{_STRIP_CANDIDATE_VERIFY!r} not found"
    )
    assert apply_idx != -1, f"the strip region must apply via {_STRIP_APPLY!r}"
    assert verify_idx < apply_idx, (
        "verify-candidate-first (#1860): the strip's candidate --file verify "
        "must PRECEDE the set-body apply"
    )
    post_apply_idx = region.find(_POST_APPLY_ISSUE_CONFIRM, apply_idx)
    assert post_apply_idx != -1, (
        "the strip's post-apply `verify_task_body.py --issue <N>` re-run must "
        "be RETAINED after the apply (frontmatter-coupled checks, kind "
        "short-circuit, concerns-audit)"
    )
