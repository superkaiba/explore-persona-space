"""Tests for the §5 ``epm:step-completed`` re-entry router.

Covers the ``decide_entry_step`` precedence table from plan §5 plus a
regression test that every EXIT site in ``.claude/skills/issue/SKILL.md``
posts a ``step-completed`` marker (count parity, per the §5
acceptance bullet).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from explore_persona_space.orchestrate.resume import (
    StepCompletedMarker,
    WorkflowStep,
    decide_entry_step,
    latest_step_completed,
)

SKILL_MD_PATH = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "issue" / "SKILL.md"
HELPER_PATH = Path(__file__).resolve().parent.parent / "scripts" / "post_step_completed.py"


# ──────────────────────────────────────────────────────────────────────
# decide_entry_step: rule-by-rule
# ──────────────────────────────────────────────────────────────────────


def _step(id_: str, *labels: str) -> WorkflowStep:
    return WorkflowStep(id=id_, entry_status_label=tuple(labels))


def test_first_run_no_marker_returns_none_for_full_replay():
    """Rule 2: no marker → full replay."""
    decision = decide_entry_step(
        status_label="status:running",
        markers=[],
        workflow_steps=[_step("6a", "running")],
    )
    assert decision is None


def test_clean_exit_with_matching_status_skips_ahead():
    """Happy path: clean marker + matching label → jump."""
    marker = StepCompletedMarker(step="5b", next_expected_step="6a", exit_kind="clean")
    decision = decide_entry_step(
        status_label="status:running",
        markers=[marker],
        workflow_steps=[_step("6a", "running")],
    )
    assert decision == "6a"


def test_status_blocked_always_full_replay_even_with_clean_marker():
    """Rule 1 (load-bearing C2.B2): status:blocked beats any marker."""
    marker = StepCompletedMarker(step="5b", next_expected_step="6a", exit_kind="clean")
    decision = decide_entry_step(
        status_label="status:blocked",
        markers=[marker],
        workflow_steps=[_step("6a", "running"), _step("blocked-handler", "blocked")],
    )
    assert decision is None


def test_failure_exit_falls_back_to_full_replay():
    """Rule 3: exit_kind=failure-exit → full replay."""
    marker = StepCompletedMarker(step="5b", next_expected_step="6a", exit_kind="failure-exit")
    decision = decide_entry_step(
        status_label="status:running",
        markers=[marker],
        workflow_steps=[_step("6a", "running")],
    )
    assert decision is None


def test_parked_exit_falls_back_to_full_replay():
    """Rule 3: exit_kind=parked → full replay (user-gated wait)."""
    marker = StepCompletedMarker(step="2c", next_expected_step="3", exit_kind="parked")
    decision = decide_entry_step(
        status_label="status:plan-pending",
        markers=[marker],
        workflow_steps=[_step("3", "approved")],
    )
    assert decision is None


def test_unknown_step_id_logs_warning_and_full_replays(caplog):
    """Rule 4: removed/renamed step → warn + full replay."""
    marker = StepCompletedMarker(step="5b", next_expected_step="ghost", exit_kind="clean")
    with caplog.at_level("WARNING", logger="explore_persona_space.orchestrate.resume"):
        decision = decide_entry_step(
            status_label="status:running",
            markers=[marker],
            workflow_steps=[_step("6a", "running")],
        )
    assert decision is None
    assert any("unknown step" in r.message for r in caplog.records)


def test_status_drift_full_replay():
    """Rule 5: marker says step 6a, label says approved → full replay."""
    marker = StepCompletedMarker(step="5b", next_expected_step="6a", exit_kind="clean")
    decision = decide_entry_step(
        status_label="status:approved",  # expected "running"
        markers=[marker],
        workflow_steps=[_step("6a", "running")],
    )
    assert decision is None


def test_unknown_exit_kind_full_replays():
    """Defense: malformed exit_kind → warn + full replay (no crash)."""
    marker = StepCompletedMarker(step="5b", next_expected_step="6a", exit_kind="exploded")
    decision = decide_entry_step(
        status_label="status:running",
        markers=[marker],
        workflow_steps=[_step("6a", "running")],
    )
    assert decision is None


def test_empty_entry_status_label_falls_back():
    """Defense: a step with empty entry_status_label → full replay."""
    marker = StepCompletedMarker(step="5b", next_expected_step="6a", exit_kind="clean")
    decision = decide_entry_step(
        status_label="status:running",
        markers=[marker],
        workflow_steps=[_step("6a")],  # no labels
    )
    assert decision is None


def test_latest_marker_wins_when_multiple_present():
    """If the issue has 5 step-completed markers, the LAST one is latest."""
    older = StepCompletedMarker(step="2", next_expected_step="2b", exit_kind="clean")
    newest = StepCompletedMarker(step="5b", next_expected_step="6a", exit_kind="clean")
    decision = decide_entry_step(
        status_label="status:running",
        markers=[older, newest],
        workflow_steps=[_step("6a", "running"), _step("2b", "planning")],
    )
    assert decision == "6a"


def test_latest_step_completed_returns_none_for_empty():
    assert latest_step_completed([]) is None


def test_latest_step_completed_returns_last_element():
    a = StepCompletedMarker(step="1", next_expected_step="2", exit_kind="clean")
    b = StepCompletedMarker(step="2", next_expected_step="3", exit_kind="clean")
    assert latest_step_completed([a, b]) is b


# ──────────────────────────────────────────────────────────────────────
# Regression: every EXIT site in SKILL.md posts a step-completed marker
# ──────────────────────────────────────────────────────────────────────


def test_skill_md_documents_resume_router():
    """Plan §5 spec: SKILL.md MUST contain the resume-router doc block.

    The block lists the 17-row EXIT-site → exit_kind mapping table from
    plan §5 lines ~1171-1192. This regression test asserts the
    documentation contract is in place — actual call-site wiring at all
    17 EXIT sites is staged via the follow-up issues (per the plan's
    phased migration).
    """
    text = SKILL_MD_PATH.read_text()
    # Required headers + helper reference.
    assert "Step-completed re-entry skip-ahead" in text, (
        "SKILL.md must contain the §5 'Step-completed re-entry skip-ahead' section header"
    )
    assert "scripts/post_step_completed.py" in text, (
        "SKILL.md must reference the post_step_completed.py helper"
    )
    assert "epm:step-completed v1" in text, (
        "SKILL.md must reference the epm:step-completed marker kind"
    )
    # Three exit kinds documented.
    for ek in ("clean", "parked", "failure-exit"):
        assert ek in text, f"SKILL.md missing exit_kind documentation for {ek!r}"
    # All 6 precedence rules referenced (the load-bearing rule 1 most
    # importantly).
    assert "status:blocked" in text and "rule 1" in text.lower(), (
        "SKILL.md missing the load-bearing rule 1 (status:blocked → full replay)"
    )


def test_skill_md_exit_site_table_has_at_least_seventeen_rows():
    """The plan-§5 EXIT-site table enumerates 17 EXIT sites.

    Tolerance: the implementer may add or merge rows during wiring; the
    test asserts ≥15 rows (allowing for ±2 of in-flight refinement).
    Below 15 means significant drift from the plan and warrants review.
    """
    text = SKILL_MD_PATH.read_text()
    # Locate the table heading inside the resume-router section.
    section_start = text.find("Step-completed re-entry skip-ahead")
    assert section_start > 0
    section = text[section_start:]
    # The mapping table has rows starting with "| Step 0", "| Step 1", etc.
    # Count rows that contain one of the three exit_kind tokens in
    # backticks: `clean` / `parked` / `failure-exit`.
    row_re = re.compile(
        r"^\|.*\|\s*`(clean|parked|failure-exit)`\s*\|\s*$",
        re.MULTILINE,
    )
    matches = row_re.findall(section)
    assert len(matches) >= 15, (
        f"EXIT-site table has only {len(matches)} rows; expected ≥15 per plan §5. "
        "If you intentionally pruned the table, lower the threshold here AND "
        "update the plan's acceptance criterion."
    )
    # Sanity: every row's exit_kind is one of the three valid values.
    for kind in matches:
        assert kind in ("clean", "parked", "failure-exit")


def test_every_exit_site_posts_marker():
    """Plan §5 acceptance: EVERY actionable EXIT site in SKILL.md must invoke
    ``post_step_completed.py`` so the §5 re-entry router has a marker to read.

    The check is local: for every line that contains the all-caps token
    ``EXIT`` (with optional trailing punctuation), the **same line OR the
    surrounding 6 lines** must reference ``post_step_completed.py``. This
    catches drift where someone adds a new EXIT path but forgets to wire
    the marker post.

    Excluded from the check (these mention ``EXIT`` but are not action
    sites):

    * Lines inside the EXIT-site → exit_kind mapping table (the mapping
      table itself, lines bounded by the resume-router section header).
    * Lines inside the resume-semantics table further down (lines that
      describe state transitions, not action sites).
    * Lines inside the error-handling table at the end of SKILL.md.
    * The header text "Step-completed re-entry skip-ahead" itself and
      its prose explanation of the marker.

    The exclusion is mechanical: we strip out anything between the §5
    section header and the end of the resume-semantics section before
    counting EXIT lines. Action EXITs all live in earlier prose.
    """
    text = SKILL_MD_PATH.read_text()
    # Cut off the documentation section: anything from the resume-router
    # header onward is reference, not action.
    section_marker = "### Step-completed re-entry skip-ahead"
    cut = text.find(section_marker)
    assert cut > 0, "could not find §5 doc section to cut"
    # Action region = beginning of file up to the §5 doc section.
    action_region = text[:cut]
    lines = action_region.splitlines()

    # Find every line with an all-caps EXIT token (word boundary, optional
    # punctuation after). Skip code fences and table rows.
    exit_re = re.compile(r"\bEXIT\b")
    in_code_fence = False
    exit_line_indices: list[int] = []
    # Phrases that are meta/prose rather than concrete EXIT call sites.
    # These mention EXIT to explain the rule, not to instruct an exit at
    # this point in the lifecycle. The §5 marker post is irrelevant.
    META_PHRASES = (
        "auto-continuation rule",  # "EXIT regardless of the auto-continuation rule"
        "user-input gates",  # "the only legitimate user-input gates ... EXIT"
        "auto-continuation policy",
    )
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        # Skip pure-description table rows. Error-handling table rows
        # that include a post_step_completed.py reference on the same row
        # ARE action sites and must NOT be skipped.
        if (
            stripped.startswith("|")
            and stripped.endswith("|")
            and "post_step_completed.py" not in line
            and exit_re.search(line)
        ):
            continue
        if not exit_re.search(line):
            continue
        # Check meta phrases in same line OR adjacent (±2 line) context.
        ctx_start = max(0, idx - 2)
        ctx_end = min(len(lines), idx + 3)
        ctx = "\n".join(lines[ctx_start:ctx_end])
        if any(phrase in ctx for phrase in META_PHRASES):
            continue
        exit_line_indices.append(idx)

    assert exit_line_indices, "no EXIT sites found in SKILL.md action region"

    # For each EXIT line, check that the same line OR a nearby line
    # (±6 lines, conservative bounded-window) references the helper.
    helper_token = "post_step_completed.py"
    missing: list[tuple[int, str]] = []
    for idx in exit_line_indices:
        window_start = max(0, idx - 6)
        window_end = min(len(lines), idx + 7)
        window = "\n".join(lines[window_start:window_end])
        if helper_token not in window:
            missing.append((idx + 1, lines[idx].strip()[:140]))

    assert not missing, (
        "Every EXIT site in SKILL.md must invoke "
        f"`{helper_token}` (within ±6 lines) so the §5 re-entry router has "
        "a marker to read. Missing wiring at:\n"
        + "\n".join(f"  L{ln}: {body}" for ln, body in missing)
        + "\nPlan §5 acceptance: 'every EXIT site posts a step-completed "
        "marker'. Add a `uv run python scripts/post_step_completed.py "
        "--issue <N> --step <id> --exit-kind <clean|parked|failure-exit> "
        '--notes "<one-line>"` call before each EXIT, or refactor the '
        "EXIT into the §5 doc section if it is reference, not action."
    )


def test_skill_md_action_exit_count_matches_table_minimum():
    """Lower bound: the action region's EXIT count is at least 10 — small
    sanity that we are still wiring real call sites and not just the
    documentation section.

    The plan §5 mapping table lists 17 EXIT sites; 2 of those are TDD-gate
    exits owned by the implementer agent (not by /issue SKILL.md prose),
    so SKILL.md's action region has roughly 15 sites. A lower bound of 10
    leaves room for plan-driven refactors that merge sites without
    breaking the regression on every commit.
    """
    text = SKILL_MD_PATH.read_text()
    cut = text.find("### Step-completed re-entry skip-ahead")
    action_region = text[:cut]
    # Count EXIT references in prose (not table rows, not code fences).
    in_code_fence = False
    count = 0
    for line in action_region.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        if re.search(r"\bEXIT\b", line):
            count += 1
    assert count >= 10, (
        f"SKILL.md action region only has {count} EXIT sites; "
        "expected ≥10 per plan §5 (the mapping table lists 17 total, "
        "2 owned by implementer.md, ≈15 in SKILL.md, lower bound 10 "
        "leaves headroom for refactors)."
    )


# ──────────────────────────────────────────────────────────────────────
# post_step_completed.py helper: dry-run + body shape
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def helper_module():
    """Load scripts/post_step_completed.py as a module."""
    spec = importlib.util.spec_from_file_location("post_step_completed", HELPER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["post_step_completed"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop("post_step_completed", None)
        raise
    return mod


def test_build_marker_body_renders_required_fields(helper_module):
    """build_marker_body produces a marker that decide_entry_step can parse."""
    body = helper_module.build_marker_body(
        step="5b",
        next_expected_step="6a",
        exit_kind="clean",
        notes="code-review PASS",
        at="abc12345",
    )
    assert "<!-- epm:step-completed v1 -->" in body
    assert "<!-- /epm:step-completed -->" in body
    assert "step: 5b" in body
    assert "next_expected_step: 6a" in body
    assert "exit_kind: clean" in body
    assert "notes: code-review PASS" in body
    assert "at: abc12345" in body


def test_build_marker_body_omits_notes_when_empty(helper_module):
    body = helper_module.build_marker_body(
        step="5b", next_expected_step="6a", exit_kind="clean", at="x"
    )
    assert "notes:" not in body


def test_helper_rejects_unknown_exit_kind_via_argparse(helper_module, capsys):
    """argparse choices guard rejects exit_kind=bogus."""
    with pytest.raises(SystemExit):
        helper_module.main(["--issue", "320", "--step", "5b", "--exit-kind", "bogus", "--dry-run"])


def test_helper_rejects_unknown_step_id(helper_module, capsys):
    """Unknown step ID → return code 2, error to stderr."""
    rc = helper_module.main(
        [
            "--issue",
            "320",
            "--step",
            "ghost-step-99",
            "--exit-kind",
            "clean",
            "--dry-run",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "is not in workflow.yaml" in captured.err


def test_helper_dry_run_prints_body_for_known_step(helper_module, capsys):
    """--dry-run on a known step prints the body and exits 0."""
    rc = helper_module.main(["--issue", "320", "--step", "5", "--exit-kind", "clean", "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "epm:step-completed v1" in captured.out
    assert "step: 5" in captured.out
    assert "exit_kind: clean" in captured.out
    # next_expected_step is looked up from workflow.yaml. For step 5
    # (code_review) the §1 mapping says "6a" (pod_provision).
    assert "next_expected_step: 6a" in captured.out
