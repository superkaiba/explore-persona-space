# ruff: noqa: RUF001
"""Tests for `check_planned_vs_actual_denominator` in
`scripts/verify_task_body.py` and clean-result-critic Lens 13.

Mechanical scope (this file): WITHIN the body only — TL;DR's `X of N
<noun>` headline must match any `M of N <noun>` scope-correction claim
found elsewhere in the body (typically inside a result H3 or in
`## Reproducibility` under the 2-content-section spec; in legacy
in-flight bodies, possibly under a retired `### Methodology corrections`
H3 that the check still picks up because the scan is whole-body
outside `## TL;DR`).

Semantic scope (clean-result-critic Lens 13, NOT this file): reads the
plan to verify the body actually names every planned condition.
"""
# ruff: noqa: RUF001
# E501: synthetic body fixtures intentionally use realistic long lines.
# RUF001: the multiplication sign appears in fixture bodies that mirror
# real clean-result write-ups; substituting an ASCII x would defeat the
# fixture's purpose.

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


# ─── Body shells for the test scenarios ───────────────────────────────────


def _body_with_tldr_and_scope_correction(tldr_results: str, scope_correction: str) -> str:
    """Assemble a 2-content-section body that exercises the planned-vs-
    actual check.

    Only the `### A factor-sweep result` story-beat sentence inside
    `## TL;DR` and the scope-correction paragraph inside
    `## Reproducibility` vary across scenarios; everything else is
    constant boilerplate so the planned-vs-actual check is the only
    one whose verdict varies. The scope-correction paragraph lives in
    `## Reproducibility` (the new spec home for documentation of
    scope-shrinkage); the check scans the body OUTSIDE `## TL;DR` so
    any non-TL;DR location works.
    """
    return f"""\
---
title: Toy planned-vs-actual test body
kind: experiment
goal: Verify the planned-vs-actual denominator check fires correctly
---
# A toy claim about factor selectivity (MODERATE confidence)

## Human TL;DR

placeholder

## TL;DR

### Motivation

I tested a 3-factor sweep across system-prompt length, framing, and
training-data source. Each factor flips one bit against the anchor cell.
The matched-pair analysis is the headline test.

### A factor-sweep result

{tldr_results}

## Reproducibility

**Parameters:**

| key | value |
|---|---|
| seed | 42 |
| model | Qwen2.5-7B-Instruct |

**Artifacts:**
- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)
- WandB run: [link](https://wandb.ai/superkaiba/eps/runs/abc12345)

**Compute:** 4× H100, 7 hours.

**Code:** entry script @ commit [0123456789abcdef](https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/dispatch.py).

{scope_correction}

Confidence: MODERATE — single seed and limited factor coverage as documented above.
"""


_NO_SCOPE_CORRECTION = ""

_SCOPE_CORRECTION_2_OF_3 = """\
**Scope correction.** This run hit infrastructure faults that
constrained the test design. The C-axis (neutral framing) cell never
trained — the dispatcher's round-4 padding patch addressed a
CPaddingError on one cell but the cell-launch ordering meant the
C-flip cell was not re-launched. So 2 of 3 factors testable from this
run.
"""


# ─── Test 1: plan has 3 factor flips, all 3 testable → PASS ───────────────


def test_all_three_factors_delivered_passes():
    """When the body has NO scope-correction claim (no scope reduction
    to discipline), the check passes vacuously."""
    body = _body_with_tldr_and_scope_correction(
        tldr_results="The 3-factor sweep shows clean decoupling across all three flips.",
        scope_correction=_NO_SCOPE_CORRECTION,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail
    assert "insufficient signal" in result.detail


# ─── Test 2: plan has 3 factor flips, 1 missing, TL;DR still claims "3" → FAIL ───


def test_silent_drop_with_stale_tldr_denominator_fails():
    """The scenario from task #391: scope-correction prose documents
    "2 of 3 testable" but the TL;DR result still says "the 3-factor
    sweep showed no clean decoupling" — TL;DR denominator is stale
    relative to the documented scope reduction. FAIL."""
    body = _body_with_tldr_and_scope_correction(
        tldr_results=("The 3-factor sweep showed only 1 of 3 factors clearing the selectivity CI."),
        scope_correction=_SCOPE_CORRECTION_2_OF_3,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert not result.passed, result.detail
    # The detail should name the inconsistency between the two surfaces.
    assert "TL;DR" in result.detail
    assert "3" in result.detail


# ─── Test 3: plan has 3 factor flips, 1 missing AND TL;DR revises to "1 of 2" → PASS ───


def test_silent_drop_with_revised_tldr_passes():
    """When the body acknowledges the C-axis drop in the scope-correction
    paragraph AND revises the TL;DR denominator to match the actual
    coverage (e.g., "1 of 2 testable factors"), the check passes — the
    headline surface is consistent with the documented scope."""
    body = _body_with_tldr_and_scope_correction(
        tldr_results=(
            "1 of 2 testable factors clears the selectivity CI, n=3 sources "
            "× 1 seed. The third planned factor is not testable this run "
            "because the C-flip cell never trained (see scope correction "
            "in Reproducibility)."
        ),
        scope_correction=_SCOPE_CORRECTION_2_OF_3,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail


# ─── Test 4: no scope-correction prose at all → vacuous PASS ──────────────


def test_no_corrections_no_check():
    """When the body has no non-TL;DR scope-correction claim, there is
    no within-body inconsistency to enforce — vacuous PASS."""
    body = _body_with_tldr_and_scope_correction(
        tldr_results="The 3-factor sweep delivered the expected 3 of 3 directional signal.",
        scope_correction=_NO_SCOPE_CORRECTION,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail


# ─── Test 5: noun mismatch (different scopes) does not false-positive ─────


def test_different_nouns_dont_conflict():
    """A TL;DR claim about "5 seeds" and a scope-correction claim
    about "2 of 3 factors" should not cross-react — different scopes."""
    body = _body_with_tldr_and_scope_correction(
        tldr_results=(
            "Across 4 of 5 seeds, the headline direction holds (the fifth "
            "seed had a separate decoding artifact)."
        ),
        scope_correction=_SCOPE_CORRECTION_2_OF_3,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail


# ─── Test 6: only TL;DR has a denominator claim, no scope correction ─────


def test_tldr_only_no_corrections_passes():
    """A body that names a denominator in TL;DR but has no scope-
    correction prose passes — there's nothing to compare against."""
    body = _body_with_tldr_and_scope_correction(
        tldr_results="All 3 of 3 swept factors cleared the selectivity CI with the same sign.",
        scope_correction=_NO_SCOPE_CORRECTION,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail


# ─── Test 7: full verify_text() end-to-end on the FAIL scenario ──────────


def test_end_to_end_silent_drop_fails_verify_text():
    """Through the full verify_text driver, the FAIL surfaces in the
    `planned-vs-actual denominator consistency` check name. Useful as a
    smoke test that the check is correctly wired into CHECKS."""
    body = _body_with_tldr_and_scope_correction(
        tldr_results=("The 3-factor sweep showed only 1 of 3 factors clearing the selectivity CI."),
        scope_correction=_SCOPE_CORRECTION_2_OF_3,
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = {r.name: r for r in results}
    # The check fires by name; we don't assert `ok` because other
    # pre-existing checks may FAIL on this skeletal body shape (no
    # figure, etc.) and that's out of scope.
    assert "planned-vs-actual denominator consistency" in by_name
    assert not by_name["planned-vs-actual denominator consistency"].passed


# ─── Test 8: "at least" / "≥" syntactic variants ─────────────────────────


def test_at_least_syntax_is_detected():
    """When the body's TL;DR phrases the hypothesis as "≥2 of 3 factors"
    after a scope reduction, the check should still fire."""
    body = _body_with_tldr_and_scope_correction(
        tldr_results=(
            "At least 1 of 3 factors clears the selectivity CI; the remaining two are inconclusive."
        ),
        scope_correction=_SCOPE_CORRECTION_2_OF_3,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert not result.passed, result.detail


# ─── Test 9: large rate-style "X of Y" denominators do not trigger ───────


def test_large_population_denominators_ignored():
    """A claim like "1 of 24 panel personas refused" is a rate, not a
    planned-vs-actual count. The check should ignore large denominators
    (> 50) to avoid false positives on bystander / persona / row counts."""
    big_scope = """\
**Scope correction.** The aggregator dropped 12 of 100 prompts due to a
tokenizer mismatch.
"""
    body = _body_with_tldr_and_scope_correction(
        tldr_results="Across 88 of 100 prompts the headline direction holds.",
        scope_correction=big_scope,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    # The 100-prompts denominator is too large to be a planned-vs-actual
    # count; the check should not fire. (The actual rate-vs-coverage
    # distinction is the analyst's call; the mechanical check stays
    # conservative.)
    assert result.passed, result.detail


# ─── Test 10: legacy `### Methodology corrections` H3 still triggers ────


def test_legacy_methodology_corrections_h3_still_triggers():
    """Whole-body scan: even though the new spec drops the dedicated
    `### Methodology corrections` H3, in-flight legacy bodies that
    still carry it under `## Reproducibility` (or anywhere else
    outside `## TL;DR`) still get picked up. The check is whole-body,
    not heading-aware."""
    legacy_methodology = """\
### Methodology corrections

This run hit infrastructure faults. The C-axis (neutral framing) cell
never trained. So 2 of 3 factors testable from this run.
"""
    body = _body_with_tldr_and_scope_correction(
        tldr_results=("The 3-factor sweep showed only 1 of 3 factors clearing the selectivity CI."),
        scope_correction=legacy_methodology,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert not result.passed, result.detail
    assert "TL;DR" in result.detail


# ─── v3 variants (2026-W24): headline surface is `## Takeaways` + ─────────
#     `## Findings`; scope-correction lives in `## Reproducibility`. The
#     check resolves the v3 headline via `is_v3(body)`; the FAIL detail
#     message text is hardcoded "TL;DR" regardless of generation.


def _v3_body_with_headline_and_scope_correction(
    takeaways_claim: str, findings_claim: str, scope_correction: str
) -> str:
    """Assemble a v3 five-flat-H2 body that exercises the planned-vs-
    actual check. The headline denominator can live in `## Takeaways`
    and/or `## Findings`; the scope-correction lives in
    `## Reproducibility` (the spec home for scope-shrinkage docs)."""
    return f"""\
---
title: Toy v3 planned-vs-actual test body
kind: experiment
goal: Verify the planned-vs-actual denominator check fires on v3 bodies
---
# A toy claim about factor selectivity (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- {takeaways_claim}
- Capability holds with no regression at the matched dose.
- Caveat that binds interpretation: single model family, three seeds.

## What I ran

- **Why:** I tested a 3-factor sweep across length, framing, and data source.
- **Design:** each factor flips one bit against the anchor; matched-pair headline test.
- **Eval:** selectivity CI, Claude judge, 200 probes; matched to the prior surface.

## Findings

### A factor-sweep result across the planned flips

{findings_claim}

![Bar chart of selectivity across the swept factors.](https://example.com/hero.png)

> **Figure.** *Selectivity across the swept factors.*

## Data

### Trained on

Established mix (tier 2), 2,000 rows. Full data: [link](https://example.com/data)

### Evaluated with

200 probes. Full probe bank: [link](https://example.com/probes)

### Generated

600 completions. Full raw completions: [raw](https://example.com/raw)

## Reproducibility

**Parameters:**

| key | value |
|---|---|
| seed | 42 |

**Artifacts:**
- Model: [hf-hub](https://example.com/model)

**Compute:** 4x H100, 7 hours.

**Code:** entry @ commit [0123456789abcdef](https://example.com/blob).

{scope_correction}

**Context:**
- Created 2026-06-12; run executed 2026-06-13.
- Originating prompt: origin prompt not recorded
"""


def test_v3_silent_drop_with_stale_headline_denominator_fails():
    """v3 analogue of the #391 scenario: scope-correction documents
    "2 of 3 testable" but the headline (Takeaways/Findings) still frames
    against 3. FAIL."""
    body = _v3_body_with_headline_and_scope_correction(
        takeaways_claim="The 3-factor sweep showed only 1 of 3 factors clearing the CI.",
        findings_claim="Only 1 of 3 factors cleared the selectivity CI across three seeds.",
        scope_correction=_SCOPE_CORRECTION_2_OF_3,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert not result.passed, result.detail
    assert "3" in result.detail


def test_v3_silent_drop_with_revised_headline_passes():
    """When the v3 headline revises the denominator to match the
    documented 2-of-3 coverage, the check passes."""
    revised = (
        "1 of 2 testable factors clears the selectivity CI; the third "
        "planned factor never trained (see scope correction)."
    )
    body = _v3_body_with_headline_and_scope_correction(
        takeaways_claim=revised,
        findings_claim=revised,
        scope_correction=_SCOPE_CORRECTION_2_OF_3,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail


def test_v3_no_scope_correction_passes():
    """A v3 body whose only denominator is a full `3 of 3` (no scope
    reduction) passes — the headline `3 of 3` is consistent with itself
    and there is no `M of N` reduction to conflict against."""
    body = _v3_body_with_headline_and_scope_correction(
        takeaways_claim="The 3-factor sweep delivered the expected 3 of 3 directional signal.",
        findings_claim="All 3 of 3 swept factors cleared the selectivity CI with the same sign.",
        scope_correction=_NO_SCOPE_CORRECTION,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail


def test_v3_no_denominator_claims_passes_vacuously():
    """A v3 body with NO `X of N` denominator claim anywhere passes
    vacuously ("insufficient signal")."""
    body = _v3_body_with_headline_and_scope_correction(
        takeaways_claim="The factor sweep shows clean decoupling across all flips.",
        findings_claim="The sweep decoupled the factors with the same sign throughout.",
        scope_correction=_NO_SCOPE_CORRECTION,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert result.passed, result.detail
    assert "insufficient signal" in result.detail


def test_v3_headline_resolves_takeaways_and_findings():
    """Sanity check that the v3 branch resolves the headline surface to
    `## Takeaways` + `## Findings` — a stale denominator that appears
    ONLY in `## Findings` (not Takeaways) still trips the check."""
    body = _v3_body_with_headline_and_scope_correction(
        takeaways_claim="A clean directional signal across the swept factors.",
        findings_claim="The 3-factor sweep showed only 1 of 3 factors clearing the CI.",
        scope_correction=_SCOPE_CORRECTION_2_OF_3,
    )
    _fm, parsed_body = verify_task_body.split_frontmatter(body)
    assert verify_task_body.is_v3(parsed_body)
    result = verify_task_body.check_planned_vs_actual_denominator(parsed_body)
    assert not result.passed, result.detail
