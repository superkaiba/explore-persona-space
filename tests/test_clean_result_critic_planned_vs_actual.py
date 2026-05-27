"""Tests for the planned-vs-actual coverage check in
`scripts/verify_task_body.py` and clean-result-critic Lens 13.

Post-mortem trigger: task #391, 2026-05-27 — the plan committed to 3
swept factors (A, C, D); cell `10111` (the C-flip cell) silently
failed during the original run, the analyzer wrote the body
acknowledging the drop in `### Methodology corrections`, but the
figure still rendered the C-axis as a missing-bar gap and the TL;DR
denominator was inconsistent with the documented scope reduction.
Round 2 of clean-result-critic PASSed without flagging.

The check enforced here (mechanical, in verify_task_body.py) is a
within-body consistency check: TL;DR `X of N <noun>` vs
`### Methodology corrections` `M of N <noun>` cannot diverge on the
same noun. The plan-side enumeration is clean-result-critic Lens 13's
semantic call (it reads `plans/plan.md`), not enforced here.
"""

# ruff: noqa: RUF001
# Body fixtures use Unicode multiplication-sign and minus-sign characters
# to match the literal characters analyzers write into clean-result bodies.
# Substituting plain ASCII would defeat the regex tests. E501 is suppressed
# because the fixture body strings include long caption lines.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_task_body.py"
_spec = importlib.util.spec_from_file_location("verify_task_body", _SCRIPT)
verify_task_body = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_task_body"] = verify_task_body
_spec.loader.exec_module(verify_task_body)  # type: ignore[union-attr]


# ─── Body shells for the four test scenarios ──────────────────────────────


def _body_with_tldr_and_methodology(tldr_results: str, methodology_corrections: str) -> str:
    """Assemble a body that's structurally valid for the planned-vs-actual
    check.

    Only the `## TL;DR` Results bullet text and the
    `### Methodology corrections` H3 content vary across the four scenarios
    below; everything else is constant boilerplate so the planned-vs-actual
    check is the only one whose verdict varies.
    """
    return f"""\
---
title: Toy planned-vs-actual test body
kind: experiment
goal: Verify the planned-vs-actual denominator check fires correctly
---
# A toy claim about factor selectivity (MODERATE confidence)

## Human TL;DR

This is a stub.

## TL;DR
- **Motivation:** Test scenario for the planned-vs-actual denominator check.
- **What I ran:** A factor sweep across recipe knobs.
- **Results:** {tldr_results}
- **Next steps:** Re-run the missing factor once the dispatcher patch lands.

## Details

Some narrative explaining what the experiment did.

### Per-factor results

This is the matched-pair analysis. The plan committed to a 3-factor sweep
spanning system-prompt length, framing, and training-data source. Each
factor flips one bit against the anchor cell.

### Parameters

| key | value |
|---|---|
| seed | 42 |
| model | Qwen2.5-7B-Instruct |

Confidence: MODERATE — single seed and limited factor coverage as documented below.

{methodology_corrections}

## Reproducibility

**Artifacts:**
- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)
- WandB run: [link](https://wandb.ai/superkaiba/eps/runs/abc12345)

**Compute:** 4× H100, 7 hours.

**Code:** entry script @ commit [0123456789abcdef](https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/dispatch.py).
"""


_NO_METHOD_CORRECTIONS = ""

_METHOD_CORRECTIONS_2_OF_3 = """\
### Methodology corrections

This run hit infrastructure faults that constrained the test design.

1. **C-axis (neutral framing) cell never trained.** The dispatcher's
   round-4 padding patch addressed a CPaddingError on one cell but the
   cell-launch ordering meant the C-flip cell was not re-launched. So 2
   of 3 factors testable from this run.
"""


# ─── Test 1: plan has 3 factor flips, all 3 testable → PASS ───────────────


def test_all_three_factors_delivered_passes():
    """When the body has NO `### Methodology corrections` H3 (no scope
    reduction to discipline), the check passes vacuously."""
    body = _body_with_tldr_and_methodology(
        tldr_results="The 3-factor sweep shows clean decoupling across all three flips.",
        methodology_corrections=_NO_METHOD_CORRECTIONS,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail
    assert "no `### Methodology corrections`" in result.detail


# ─── Test 2: plan has 3 factor flips, 1 missing, TL;DR still claims "3" → FAIL ───


def test_silent_drop_with_stale_tldr_denominator_fails():
    """The scenario from task #391: Methodology corrections documents
    "2 of 3 testable" but the TL;DR Results bullet still says "the
    3-factor sweep showed no clean decoupling" — TL;DR denominator is
    stale relative to the documented scope reduction. FAIL."""
    body = _body_with_tldr_and_methodology(
        tldr_results=("The 3-factor sweep showed only 1 of 3 factors clearing the selectivity CI."),
        methodology_corrections=_METHOD_CORRECTIONS_2_OF_3,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert not result.passed, result.detail
    # The detail should name the inconsistency between the two surfaces.
    assert "Methodology corrections" in result.detail
    assert "TL;DR" in result.detail
    assert "3" in result.detail


# ─── Test 3: plan has 3 factor flips, 1 missing AND TL;DR revises to "2 of 2" → PASS ───


def test_silent_drop_with_revised_tldr_passes():
    """When the body acknowledges the C-axis drop in Methodology
    corrections AND revises the TL;DR denominator to match the actual
    coverage (e.g., "1 of 2 testable factors"), the check passes — the
    headline surface is consistent with the documented scope."""
    body = _body_with_tldr_and_methodology(
        tldr_results=(
            "1 of 2 testable factors clears the selectivity CI, n=3 sources "
            "× 1 seed. The third planned factor is not testable this run "
            "because the C-flip cell never trained (see `### Methodology "
            "corrections`)."
        ),
        methodology_corrections=_METHOD_CORRECTIONS_2_OF_3,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail


# ─── Test 4: TL;DR denominator matches Methodology corrections numerator → PASS ───


def test_no_corrections_no_check():
    """When the body has no `### Methodology corrections` H3 at all,
    there is no within-body inconsistency to enforce — vacuous PASS."""
    body = _body_with_tldr_and_methodology(
        tldr_results="The 3-factor sweep delivered the expected 3 of 3 directional signal.",
        methodology_corrections=_NO_METHOD_CORRECTIONS,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail


# ─── Test 5: noun mismatch (different scopes) does not false-positive ─────


def test_different_nouns_dont_conflict():
    """A TL;DR claim about "5 seeds" and a Methodology corrections claim
    about "2 of 3 factors" should not cross-react — different scopes."""
    body = _body_with_tldr_and_methodology(
        tldr_results=(
            "Across 4 of 5 seeds, the headline direction holds (the fifth "
            "seed had a separate decoding artifact)."
        ),
        methodology_corrections=_METHOD_CORRECTIONS_2_OF_3,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail


# ─── Test 6: only TL;DR has a denominator claim, no Methodology → vacuous PASS ───


def test_tldr_only_no_corrections_passes():
    """A body that names a denominator in TL;DR but has no Methodology
    corrections section passes — there's nothing to compare against."""
    body = _body_with_tldr_and_methodology(
        tldr_results="All 3 of 3 swept factors cleared the selectivity CI with the same sign.",
        methodology_corrections=_NO_METHOD_CORRECTIONS,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail


# ─── Test 7: full verify_text() end-to-end on the FAIL scenario ──────────


def test_end_to_end_silent_drop_fails_verify_text():
    """Through the full verify_text driver, the FAIL surfaces in the
    `planned-vs-actual denominator consistency` check name. Useful as a
    smoke test that the check is correctly wired into CHECKS."""
    body = _body_with_tldr_and_methodology(
        tldr_results=("The 3-factor sweep showed only 1 of 3 factors clearing the selectivity CI."),
        methodology_corrections=_METHOD_CORRECTIONS_2_OF_3,
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = {r.name: r for r in results}
    # The new check fires by name; we don't assert `ok` because other
    # pre-existing checks may FAIL on this skeletal body shape (no
    # cherry-picked fences, no figure, etc.) and that's out of scope.
    assert "planned-vs-actual denominator consistency" in by_name
    assert not by_name["planned-vs-actual denominator consistency"].passed


# ─── Test 8: "at least" / "≥" syntactic variants ─────────────────────────


def test_at_least_syntax_is_detected():
    """The plan often phrases hypotheses as "≥2 of 3 factors will clear
    zero". When the body's TL;DR reuses the same shape with the original
    denominator after a scope reduction, the check should still fire."""
    body = _body_with_tldr_and_methodology(
        tldr_results=(
            "At least 1 of 3 factors clears the selectivity CI; the remaining two are inconclusive."
        ),
        methodology_corrections=_METHOD_CORRECTIONS_2_OF_3,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert not result.passed, result.detail


# ─── Test 9: large rate-style "X of Y" denominators do not trigger ───────


def test_large_population_denominators_ignored():
    """A claim like "1 of 24 panel personas refused" is a rate, not a
    planned-vs-actual count. The check should ignore large denominators
    (> 50) to avoid false positives on bystander / persona / row counts."""
    big_method = """\
### Methodology corrections

The aggregator dropped 12 of 100 prompts due to a tokenizer mismatch.
"""
    body = _body_with_tldr_and_methodology(
        tldr_results="Across 88 of 100 prompts the headline direction holds.",
        methodology_corrections=big_method,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    # The 100-prompts denominator is too large to be a planned-vs-actual
    # count; the check should not fire. (The actual rate-vs-coverage
    # distinction is the analyst's call; the mechanical check stays
    # conservative.)
    assert result.passed, result.detail
