# ruff: noqa: RUF003  # × / em-dash intentional
"""Task #479 round-3 Blocker-1.3 regression guard: dispatcher Phase-1.6 base-emission.

Round 2 wired `i479_analyze.py` to LOAD `base_panel_emission_rate.json`, but
the dispatcher never RAN `i479_phase_base_emission.py` to create it. The
fix adds a Phase-1.6 block to the dispatcher's #479 route that runs the
base-emission script (writing `<slab_root>/base_panel_emission_rate.json`)
before the analyze phase.

This test pins the inclusion by reading the dispatcher source and asserting
the Phase-1.6 block references both `i479_phase_base_emission.py` and the
expected output filename. Source-text checks (not execution) because the
real run requires a GPU; the assertion captures the regression class that
"the dispatcher route doesn't include the new phase".
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DISPATCHER_SCRIPT = REPO_ROOT / "scripts" / "dispatch_neg_geometry_472.py"


def test_dispatcher_includes_i479_phase_base_emission_for_issue_479():
    """The #479 route MUST call scripts/i479_phase_base_emission.py."""
    src = DISPATCHER_SCRIPT.read_text()
    assert "scripts/i479_phase_base_emission.py" in src, (
        "Dispatcher does not reference scripts/i479_phase_base_emission.py — "
        "the analyzer's bystander threshold will run with no anchor. Add the "
        "Phase-1.6 block to the --issue 479 route."
    )


def test_dispatcher_writes_base_emission_rate_filename():
    """The Phase-1.6 block must write to base_panel_emission_rate.json."""
    src = DISPATCHER_SCRIPT.read_text()
    assert "base_panel_emission_rate.json" in src, (
        "Dispatcher does not write to base_panel_emission_rate.json — the "
        "filename i479_analyze.py reads from. Filename mismatch will silently "
        "leave the bystander threshold unanchored (Blocker 1)."
    )


def test_dispatcher_has_skip_base_emission_flag():
    """The --skip-base-emission flag must exist (symmetry with --skip-base-panel)."""
    src = DISPATCHER_SCRIPT.read_text()
    assert "--skip-base-emission" in src, (
        "Dispatcher missing --skip-base-emission flag for symmetry with "
        "--skip-base-panel; needed so resumed/re-analysis runs can skip the "
        "GPU phase if the baseline already exists."
    )


def test_dispatcher_phase_1_6_is_gated_by_issue_479():
    """The base-emission phase must run only for --issue 479 (not for --issue 472).

    The dispatcher's source has multiple `args.issue == 479` guards (Phase
    1.6, the analyze branch). We pin that the i479_phase_base_emission.py
    call sits immediately AFTER a `args.issue == 479` guard (within a small
    window, capturing the if/elif body), proving the phase is gated.
    """
    src = DISPATCHER_SCRIPT.read_text()
    idx_script = src.find("scripts/i479_phase_base_emission.py")
    assert idx_script != -1, "dispatcher missing scripts/i479_phase_base_emission.py call"
    # Find the LAST `args.issue == 479` guard BEFORE the script path. The
    # script path must sit within ~800 chars of its guard (a typical
    # subprocess command block fits well within that).
    guard_prefix = src[:idx_script]
    last_guard = guard_prefix.rfind("args.issue == 479")
    assert last_guard != -1, (
        "no `args.issue == 479` guard found before the base-emission call — "
        "Phase 1.6 would run for every --issue value."
    )
    distance = idx_script - last_guard
    assert distance < 800, (
        f"the i479_phase_base_emission.py call is {distance} chars after its "
        "guard — the gate may not actually wrap the call. Inspect "
        "dispatch_neg_geometry_472.py around the Phase-1.6 block."
    )


def test_dispatcher_analyzer_call_does_not_pass_no_strict_base_panel():
    """The dispatcher must NOT pass --no-strict-base-panel to i479_analyze.py.

    Strict mode is the production default (round-3 Blocker 1.2): a missing
    Phase-1.6 baseline MUST hard-fail the analyzer, not silently degrade
    to "unanchored bystander threshold". If the dispatcher ever passes
    --no-strict-base-panel, the round-3 hard-fail becomes inert.
    """
    src = DISPATCHER_SCRIPT.read_text()
    # The analyzer subprocess command block must not contain the no-strict flag.
    # Find the i479_analyze.py invocation and inspect the surrounding command.
    idx = src.find("scripts/i479_analyze.py")
    assert idx != -1, "dispatcher missing the i479_analyze.py call"
    # Read a ~2000-char window after the call to capture the command args.
    window = src[idx : idx + 2000]
    assert "--no-strict-base-panel" not in window, (
        "Dispatcher passes --no-strict-base-panel to i479_analyze.py, which "
        "would disable the round-3 Blocker-1.2 hard-fail. Strict mode must "
        "be the production default."
    )
