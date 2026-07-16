"""Tests for scripts/verify_plan.py — mechanical pre-pass gate for experiment
plans at /adversarial-planner Phase 1.5.0 (task #625).

Each test feeds a synthetic plan string into verify_plan_text() and asserts
which checks PASS / WARN / FAIL / SKIP. The canonical GOOD_PLAN fixture
mirrors the recently-approved-plan corpus shape (#614 v2 / #613 v1 / #610 v1):
a §0.0 TL;DR with the mandated "What would change my mind" line, a numbered
Goal/Design body, a Measurement-validity table, one data-tier sentence, the
machine-readable GPU-hours line, a success+kill criteria section, a
conditions table + seeds, and a §11 Decision Rationale with inline `Source:`
entries (one `ungrounded — needs smoke-test`).
"""

# ruff: noqa: E501, RUF001
# The fixture plan strings below INCLUDE the literal markdown the verifier
# scans — em/en dashes, the `ungrounded — needs smoke-test` contract string,
# long table rows. Reflowing or substituting these would defeat the tests.

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = REPO_ROOT / "scripts" / "verify_plan.py"
_spec = importlib.util.spec_from_file_location("verify_plan", _SCRIPT)
verify_plan = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_plan"] = verify_plan
_spec.loader.exec_module(verify_plan)  # type: ignore[union-attr]


# ─── Canonical plan (kind=experiment: passes 0-3,5,8,9,17; skips 4,6,7,10-16,18-22,24-31)

# Surgery anchors (must appear verbatim in GOOD_PLAN exactly once).
MV_HEADING = "### Measurement validity"
MV_TABLE = """\
| DV | Construct | Metric | On-distribution? |
|---|---|---|---|
| persona expression | judge-scored persona consistency | mean 1-5 judge score on 40 held-out prompts | yes (on-policy free generation) |"""
TIER_SENTENCE = "Data realism: established dataset (tier 2: UltraChat, cited by name); no synthetic generation anywhere in the pipeline."
GPU_LINE = "`Estimated GPU-hours (total): 4`"
SUCCESS_SENT = "**Success criteria:** the benign-sft arm's judge-score delta vs base is estimated with a 95% CI across the three runs, and the CI half-width is below 0.4 judge points so the read is interpretable either way."
KILL_SENT = "**Kill criteria:** if the judge refuses or fails to parse on more than 20% of prompts in the smoke run, halt-and-report — the eval surface is broken and no training read is meaningful."
CRITERIA_HEADING = "## 7. Decision gates, success and kill criteria"
SEEDS_SENTENCE = "Seeds: {42, 137, 256}; both conditions share the same eval prompts."

GOOD_PLAN = f"""\
# Plan — Task #999: Does benign SFT shift persona judge scores? (toy fixture)

## 0.0 TL;DR (plain English)

- **What I'll build:** A small fine-tuning probe that measures whether benign supervised fine-tuning moves judge-scored persona expression on held-out prompts.
- **What I expect:** Small shifts, well under the band seen in prior coupling runs.
- **What would change my mind:** A shift larger than the run-to-run spread would mean benign data alone moves persona expression.

## 1. Goal

Measure whether benign SFT on an established corpus shifts judge-scored persona expression relative to the base model, across three random restarts.

## 2. Design

We fine-tune Qwen-2.5-7B-Instruct with LoRA on UltraChat and evaluate persona expression with the Claude judge on 40 held-out prompts. {TIER_SENTENCE}

## 3. Conditions

| Condition | What it tests |
|---|---|
| base | no training reference point |
| benign-sft | the manipulated variable |

{SEEDS_SENTENCE}

{MV_HEADING}

{MV_TABLE}

{CRITERIA_HEADING}

{SUCCESS_SENT}

{KILL_SENT}

## 9. Resources

One A100 for about three hours covers both conditions. {GPU_LINE}

## 11. Decision Rationale (§11)

One `Source:` per unique value.

- **lr = 3e-5.** Why: the stable LoRA window for 7B at this data scale. Source: #612.
- **epochs = 3.** Why: convergence without over-fit at 2k rows. Source: arXiv 2507.21509 appendix table.
- **LoRA r = 32, alpha = 64.** Why: repo default validated on this model + data family. Source: #474.
- **eff. batch = 16.** Why: memory fit on one A100. Source: ungrounded — needs smoke-test.
"""

SOURCE_TABLE_S11 = """\
## 11. Decision Rationale (§11)

One `Source:` per unique value.

| What | Why (tied to Goal) | Source | Alternatives rejected |
|---|---|---|---|
| lr = 3e-5 | stable LoRA window | #612 | 1e-4 (too hot) |
| epochs = 3 | convergence at 2k rows | arXiv 2507.21509 | 1 (undertrained) |
"""


def _by_id(results):
    return {r.id: r for r in results}


def _run(plan: str, kind: str = "experiment"):
    ok, results = verify_plan.verify_plan_text(plan, kind=kind)
    return ok, _by_id(results)


def _status(plan: str, cid: str, kind: str = "experiment") -> str:
    _, by_id = _run(plan, kind)
    return by_id[cid].status


# ─── GOOD_PLAN baseline ────────────────────────────────────────────────────


def test_good_plan_passes_all():
    ok, results = verify_plan.verify_plan_text(GOOD_PLAN, kind="experiment")
    assert ok, [r.render() for r in results if not r.passed]
    by_id = _by_id(results)
    expected = {
        "c0_plan_nonstub": "PASS",
        "c1_source_grounding": "PASS",
        "c2_measurement_validity": "PASS",
        "c3_data_tier": "PASS",
        "c4_contrastive_negatives": "SKIP",
        "c5_gpu_hours": "PASS",
        "c6_reuse_fitness": "SKIP",
        "c7_replication_fidelity": "SKIP",
        "c8_success_kill_criteria": "PASS",
        "c9_conditions_seeds": "PASS",
        "c10_marker_recipe": "SKIP",
        "c11_dryrun_test_coverage": "SKIP",
        "c12_battery_multiplier": "SKIP",
        "c13_empirical_gate_attainability": "SKIP",
        "c14_hypothesis_branch_coherence": "SKIP",
        "c15_failloud_test_coverage": "SKIP",
        "c16_reference_headline_distinction": "SKIP",
        "c17_causal_branch_scope": "PASS",
        "c18_paired_contrast_source_coverage": "SKIP",
        "c19_ood_folds": "SKIP",
        "c20_verdict_lattice_coherence": "SKIP",
        "c21_grep_arity_gate": "SKIP",
        "c22_cross_section_param_consistency": "SKIP",
        "c24_resume_provenance": "SKIP",
        "c25_html_entities_in_commands": "SKIP",
        "c26_gpu_basis_routed_machine": "SKIP",
        "c27_capture_intent_hbm": "SKIP",
        "c28_precedent_band_coherence": "SKIP",
        "c29_fence_conditional_phase": "SKIP",
        "c30_realized_keys": "SKIP",
        "c31_skillmd_prose_pin": "SKIP",
        "c32_fit_basis_grounding": "SKIP",
        "c33_ladder_retention": "SKIP",
        "c34_ratchet_headroom": "SKIP",
        "c35_pinned_revision_reuse": "SKIP",
        "c36_numeric_containment": "SKIP",
        "c37_noflags_bundling_claim": "SKIP",
    }
    actual = {cid: r.status for cid, r in by_id.items()}
    assert actual == expected
    assert len(results) == 37


# ─── Check 0 — plan-nonstub ────────────────────────────────────────────────


def test_stub_plan_fails_and_short_circuits():
    ok, results = verify_plan.verify_plan_text("# Plan\n\nTBD", kind="experiment")
    assert not ok
    assert len(results) == 1  # short-circuit: one clear signal
    assert results[0].id == "c0_plan_nonstub"
    assert results[0].status == "FAIL"


def test_lone_stub_token_fails():
    ok, results = verify_plan.verify_plan_text("placeholder", kind="experiment")
    assert not ok
    assert "stub" in results[0].detail.lower()


def test_long_but_headingless_plan_fails_check0():
    plan = "word " * 400  # > 1500 chars, zero headings
    ok, results = verify_plan.verify_plan_text(plan, kind="experiment")
    assert not ok
    assert results[0].id == "c0_plan_nonstub"
    assert "headings" in results[0].detail


def test_terse_analysis_plan_passes_check0():
    # Intent fixture: a terse-but-real analysis plan (short prose, 3
    # headings, > 1500 chars — the #575 end of the observed corpus) clears
    # the stub gate; check 0 is a broken-handoff defense, not a length bar.
    filler = "We re-run the aggregation over the existing eval JSONs and re-plot. " * 25
    plan = (
        "# Plan — Task #998: re-aggregate prior eval JSONs (analysis)\n\n"
        "## Goal\n\n" + filler + "\n\n"
        "## Design\n\n" + filler + "\n\n"
        "## Resources\n\nNo pod. `Estimated GPU-hours (total): 0`\n"
    )
    assert len(plan.strip()) >= 1500
    ok, by_id = _run(plan, kind="analysis")
    assert by_id["c0_plan_nonstub"].status == "PASS"
    assert ok


def test_html_plan_check0_counts_html_headings():
    # Regression test for task #640 (2026-06-15): adversarial-planner defaults
    # to HTML output (CLAUDE.md § Output format) for browser-reading artifacts.
    # An HTML plan with 20+ <h2>/<h3> tags was FAILed as "only 1 heading (< 3)"
    # because _headings() is markdown-only. HTML headings must be counted too.
    filler = "This section describes the experimental design in detail. " * 30
    html_plan = (
        "<!DOCTYPE html><html><head><title>Plan</title></head><body>\n"
        "<h1>Plan &mdash; Issue #640: Postfix Carrier Sweep</h1>\n"
        + filler
        + "<h2>§1 Goal</h2>\n"
        + filler
        + "<h2>§2 Design</h2>\n"
        + filler
        + "<h3>Measurement validity</h3>\n"
        + filler
        + "<h2>§3 Resources</h2>\n"
        + "<p><strong>Estimated GPU-hours (total): 7</strong></p>\n"
        + filler
        + "<h2>§4 Success and kill criteria</h2>\n"
        + filler
        + "</body></html>\n"
    )
    assert len(html_plan.strip()) >= 1500
    _, by_id = _run(html_plan, kind="analysis")
    r = by_id["c0_plan_nonstub"]
    assert r.status == "PASS", r.detail
    # heading count in the detail should reflect all HTML headings
    assert "headings" in r.detail


# ─── Check 1 — §11 Source: grounding ───────────────────────────────────────


def test_c1_kind_infra_skips():
    assert _status(GOOD_PLAN, "c1_source_grounding", kind="infra") == "SKIP"


def test_c1_good_plan_counts_inline_and_ungrounded():
    _, by_id = _run(GOOD_PLAN)
    r = by_id["c1_source_grounding"]
    assert r.status == "PASS"
    assert "4 Source entries" in r.detail
    assert "1 marked ungrounded" in r.detail


def test_c1_blank_inline_source_fails():
    plan = GOOD_PLAN + "- **warmup = 0.05.** Why: convention. Source:\n"
    _, by_id = _run(plan)
    r = by_id["c1_source_grounding"]
    assert r.status == "FAIL"
    assert "blank" in r.detail.lower()


def test_c1_tbd_source_fails():
    plan = GOOD_PLAN + "- **warmup = 0.05.** Why: convention. Source: TBD\n"
    assert _status(plan, "c1_source_grounding") == "FAIL"


def test_c1_no_section_and_no_sources_fails():
    plan = GOOD_PLAN.replace("## 11. Decision Rationale (§11)", "## 11. Notes").replace(
        "Source:", "Ref:"
    )
    _, by_id = _run(plan)
    r = by_id["c1_source_grounding"]
    assert r.status == "FAIL"
    assert "no decision rationale" in r.detail.lower()


def test_c1_section_present_zero_sources_fails():
    plan = GOOD_PLAN.replace("Source:", "Ref:")
    _, by_id = _run(plan)
    r = by_id["c1_source_grounding"]
    assert r.status == "FAIL"
    assert "zero source entries" in r.detail.lower()


def test_c1_pasted_fail_detail_does_not_self_satisfy():
    # Anti-paste guard (#4.2 c1 de-fang): pre-fix, the pasted FAIL detail
    # both matched the doc-global escape (its quoted `N/A — no model
    # training`) and yielded an evidence-valued `Source:` capture via
    # "zero Source: entries — ...". Exercises the doc-global scope fallback
    # (no recognizable §11 heading → scope = whole plan).
    base = GOOD_PLAN.replace("## 11. Decision Rationale (§11)", "## 11. Notes").replace(
        "Source:", "Ref:"
    )
    _, by_id = _run(base)
    r = by_id["c1_source_grounding"]
    assert r.status == "FAIL"
    pasted = base + f"\n{r.detail}\n"
    assert _status(pasted, "c1_source_grounding") == "FAIL"


def test_c1_section_present_zero_sources_pasted_detail_does_not_self_satisfy():
    # The sibling FAIL branch's red twin: §11 heading KEPT, all sources
    # renamed → FAIL via the section-present-zero-sources branch. Pre-fix
    # its detail's "(inline `Source:` label or ...)" wording yielded an
    # evidence-valued capture when pasted INTO the §11 section (the last
    # section, so an appended paste lands in the section slice).
    base = GOOD_PLAN.replace("Source:", "Ref:")
    _, by_id = _run(base)
    r = by_id["c1_source_grounding"]
    assert r.status == "FAIL"
    pasted = base + f"\n{r.detail}\n"
    assert _status(pasted, "c1_source_grounding") == "FAIL"


def test_c1_sources_without_recognizable_section_warns():
    plan = GOOD_PLAN.replace("## 11. Decision Rationale (§11)", "## 11. Notes")
    _, by_id = _run(plan)
    r = by_id["c1_source_grounding"]
    assert r.status == "WARN"
    assert "heading" in r.detail.lower()


def test_c1_na_no_model_training_passes():
    s11 = GOOD_PLAN[GOOD_PLAN.index("## 11. Decision Rationale (§11)") :]
    plan = GOOD_PLAN.replace(
        s11, "## 11. Decision Rationale (§11)\n\nN/A — no model training (pure analysis rig).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c1_source_grounding"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


def test_c1_na_no_training_hyperparameters_passes():
    s11 = GOOD_PLAN[GOOD_PLAN.index("## 11. Decision Rationale (§11)") :]
    plan = GOOD_PLAN.replace(
        s11, "## 11. Decision Rationale (§11)\n\nN/A — no training hyperparameters.\n"
    )
    assert _status(plan, "c1_source_grounding") == "PASS"


def test_c1_source_table_column_passes():
    # The #614 v2 §11 shape: a bare `Source` table column + the planner.md
    # boilerplate sentence. The PASS must come from the table cells, not
    # the boilerplate's own `Source:` label.
    s11 = GOOD_PLAN[GOOD_PLAN.index("## 11. Decision Rationale (§11)") :]
    plan = GOOD_PLAN.replace(s11, SOURCE_TABLE_S11)
    _, by_id = _run(plan)
    r = by_id["c1_source_grounding"]
    assert r.status == "PASS"
    assert "2 table-column" in r.detail
    assert "0 inline" in r.detail  # boilerplate `Source:` label did not count


def test_c1_blank_table_cell_fails():
    s11 = GOOD_PLAN[GOOD_PLAN.index("## 11. Decision Rationale (§11)") :]
    blanked = SOURCE_TABLE_S11.replace("| arXiv 2507.21509 |", "|  |")
    plan = GOOD_PLAN.replace(s11, blanked)
    _, by_id = _run(plan)
    r = by_id["c1_source_grounding"]
    assert r.status == "FAIL"
    assert "blank" in r.detail.lower()


def test_c1_fenced_source_does_not_satisfy():
    # All real sources removed; the only `Source:` lives inside a code
    # fence — must NOT satisfy the check.
    plan = GOOD_PLAN.replace("Source:", "Ref:") + "\n```text\nSource: #612\n```\n"
    assert _status(plan, "c1_source_grounding") == "FAIL"


def test_c1_fenced_blank_source_does_not_trip():
    # Good sources intact; a fenced blank `Source:` must NOT trip the
    # blank-source FAIL.
    plan = GOOD_PLAN + "\n```text\nSource:\n```\n"
    assert _status(plan, "c1_source_grounding") == "PASS"


# ─── Check 2 — measurement validity ────────────────────────────────────────


def _plan_without_mv() -> str:
    return (
        GOOD_PLAN.replace(MV_HEADING, "### Eval notes")
        .replace("Construct", "Thing")
        .replace("Metric", "Number")
    )


def test_c2_kind_infra_skips():
    assert _status(GOOD_PLAN, "c2_measurement_validity", kind="infra") == "SKIP"


def test_c2_missing_entirely_fails():
    _, by_id = _run(_plan_without_mv())
    r = by_id["c2_measurement_validity"]
    assert r.status == "FAIL"
    assert "measurement-validity" in r.detail


def test_c2_na_no_behavioral_construct_passes():
    plan = _plan_without_mv() + "\nN/A — no behavioral construct.\n"
    _, by_id = _run(plan)
    r = by_id["c2_measurement_validity"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


def test_c2_pasted_fail_detail_does_not_self_satisfy():
    # Anti-paste guard: the FAIL detail quotes its escape phrase as a remedy
    # option — pre-fix, the doc-global escape search matched it when the
    # detail was pasted back into the plan (FAIL → PASS).
    base = _plan_without_mv()
    _, by_id = _run(base)
    r = by_id["c2_measurement_validity"]
    assert r.status == "FAIL"
    pasted = base + f"\n{r.detail}\n"
    assert _status(pasted, "c2_measurement_validity") == "FAIL"


def test_c2_heading_without_content_warns():
    plan = GOOD_PLAN.replace(MV_TABLE, "Details to follow.")
    _, by_id = _run(plan)
    r = by_id["c2_measurement_validity"]
    assert r.status == "WARN"
    assert "fact-checker" in r.detail


def test_c2_phrase_only_warns():
    plan = _plan_without_mv() + "\nWe address measurement validity in the appendix.\n"
    _, by_id = _run(plan)
    r = by_id["c2_measurement_validity"]
    assert r.status == "WARN"
    assert "phrase" in r.detail


def test_c2_table_without_heading_passes():
    plan = GOOD_PLAN.replace(MV_HEADING, "### Eval design")
    assert _status(plan, "c2_measurement_validity") == "PASS"


# ─── Check 3 — data-source tier ────────────────────────────────────────────


def test_c3_kind_infra_skips():
    assert _status(GOOD_PLAN, "c3_data_tier", kind="infra") == "SKIP"


def test_c3_no_tier_vocabulary_warns():
    plan = GOOD_PLAN.replace(TIER_SENTENCE, "We use a corpus we already had lying around.")
    _, by_id = _run(plan)
    r = by_id["c3_data_tier"]
    assert r.status == "WARN"
    assert "tier" in r.detail.lower()


def test_c3_tier34_without_justification_notes_in_detail():
    plan = GOOD_PLAN.replace(
        TIER_SENTENCE, "Data realism: diverse LLM-generated synthetic data (tier 3)."
    )
    _, by_id = _run(plan)
    r = by_id["c3_data_tier"]
    assert r.status == "PASS"  # never a verdict change
    assert "tier-3/4" in r.detail


# ─── Check 4 — contrastive negatives ───────────────────────────────────────


def test_c4_not_triggered_skips():
    assert _status(GOOD_PLAN, "c4_contrastive_negatives") == "SKIP"


def test_c4_workflow_marker_vocabulary_does_not_trigger():
    # Bare workflow vocabulary (`post-marker`, `epm:` markers) must NOT
    # count as marker-leakage vocabulary (round-1 statistics-critic fix).
    plan = (
        GOOD_PLAN
        + "\nThe orchestrator runs post-marker epm:progress and reads epm: markers from events.jsonl.\n"
    )
    assert _status(plan, "c4_contrastive_negatives") == "SKIP"


def test_c4_implant_without_negatives_warns():
    plan = GOOD_PLAN + "\nWe implant a refusal behavior into the source persona.\n"
    _, by_id = _run(plan)
    r = by_id["c4_contrastive_negatives"]
    assert r.status == "WARN"
    assert "contrastive" in r.detail


def test_c4_contrastive_negatives_pass_with_composition_tokens():
    plan = (
        GOOD_PLAN
        + "\nWe implant a refusal behavior into the source persona, with contrastive negatives: a 4-persona panel at a 1:1 ratio, disjoint from every realized source.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c4_contrastive_negatives"]
    assert r.status == "PASS"
    for token in ("panel", "ratio", "1:1", "disjoint"):
        assert token in r.detail


def test_c4_named_exemption_passes():
    plan = (
        GOOD_PLAN
        + "\nWe implant the behavior as a strict single-variable replication of a positive-only parent (exemption (b)).\n"
    )
    assert _status(plan, "c4_contrastive_negatives") == "PASS"


def test_c4_standalone_na_line_passes():
    # Standalone-line escape (the mid-line form was the self-escape shape —
    # repurposed into test_c4_midprose_phrase_does_not_escape below).
    plan = (
        GOOD_PLAN
        + "\nThe word implant appears in a quoted sibling methodology.\n"
        + "\nN/A — not a behavior-implantation (the implant vocabulary is incidental).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c4_contrastive_negatives"]
    assert r.status == "PASS"
    assert "on its own line, unwrapped" in r.detail


def test_c4_midprose_phrase_does_not_escape():
    # #1277 red-pin (c4 twin of #1262's c7 pin): the phrase mid-prose (the
    # #810 structure, one polarity over) must not escape c4.
    # Pre-fix: doc-global re.search -> PASS. This is the former
    # test_c4_na_line_passes fixture, polarity flipped.
    plan = (
        GOOD_PLAN + "\nThe word implant appears but this is not a behavior-implantation design.\n"
    )
    assert _status(plan, "c4_contrastive_negatives") == "WARN"


def test_c4_backtick_wrapped_escape_does_not_escape():
    # Mirror of test_c7_backtick_wrapped_escape_does_not_escape: the pasted
    # bounce-brief bullet shape must not escape.
    plan = (
        GOOD_PLAN
        + "\nWe implant a refusal behavior into the source persona.\n"
        + "- `N/A — not a behavior-implantation` (declare per the bounce brief).\n"
    )
    assert _status(plan, "c4_contrastive_negatives") == "WARN"


def test_c4_kind_infra_skips():
    plan = GOOD_PLAN + "\nWe implant a refusal behavior.\n"
    assert _status(plan, "c4_contrastive_negatives", kind="infra") == "SKIP"


# ─── Check 5 — GPU-hour estimate ───────────────────────────────────────────


def test_c5_absent_line_fails_with_absent_detail():
    plan = GOOD_PLAN.replace(GPU_LINE, "about four GPU hours")
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "FAIL"
    assert "absent" in r.detail


def test_c5_malformed_value_fails_with_unparseable_detail():
    plan = GOOD_PLAN.replace(GPU_LINE, "Estimated GPU-hours (total): ~4")
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "FAIL"
    assert "unparseable" in r.detail
    assert "absent" not in r.detail


def test_c5_range_fails():
    plan = GOOD_PLAN.replace(GPU_LINE, "Estimated GPU-hours (total): 4-8")
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "FAIL"
    assert "range" in r.detail


def test_c5_annotation_after_value_does_not_fail():
    # #610's real shape: a worst-case annotation after the value.
    plan = GOOD_PLAN.replace(
        GPU_LINE, "`Estimated GPU-hours (total): 22` (instance-GPU-hours; worst ≈ 42 — see §9)"
    )
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "PASS"
    assert "22" in r.detail


def test_c5_table_row_annotation_does_not_fail():
    # #614's real shape: the line inside a table row with a GPU model name.
    plan = GOOD_PLAN.replace(
        GPU_LINE,
        "| **Total (pod)** | 1× A100-80 | `Estimated GPU-hours (total): 4` (with margin) |",
    )
    assert _status(plan, "c5_gpu_hours") == "PASS"


def test_c5_wall_time_sentence_after_value_does_not_fail():
    # #580's real shape (calibration-driven predicate adjustment, plan
    # §12): a backtick-wrapped single value followed by a wall-time range
    # in the NEXT sentence must not read as a ranged estimate.
    plan = GOOD_PLAN.replace(
        GPU_LINE, "`Estimated GPU-hours (total): 0`. Wall ~1–1.5 h including review."
    )
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "PASS"
    assert "0" in r.detail


def test_c5_fails_for_exempt_kinds_when_absent():
    # Reconciler binding fix: check 5 FAILs for ALL kinds — the Step 2c
    # gate is kind-blind.
    plan = GOOD_PLAN.replace(GPU_LINE, "no compute needed")
    assert _status(plan, "c5_gpu_hours", kind="infra") == "FAIL"


def test_c5_exempt_kind_passes_with_zero():
    plan = GOOD_PLAN.replace(GPU_LINE, "`Estimated GPU-hours (total): 0`")
    assert _status(plan, "c5_gpu_hours", kind="infra") == "PASS"


def test_c5_bold_label_form_passes():
    plan = GOOD_PLAN.replace(GPU_LINE, "**Estimated GPU-hours (total): 3.5**")
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "PASS"
    assert "3.5" in r.detail


# Round-2 regression group (reconciler blocker
# gpu-hours-backtick-range-false-pass): the closing-backtick annotation
# stop must not truncate a backtick-wrapped-number range to its first
# number and PASS it.


def test_c5_backtick_wrapped_first_number_range_fails():
    plan = GOOD_PLAN.replace(GPU_LINE, "Estimated GPU-hours (total): `4`-8")
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "FAIL"
    assert "range" in r.detail


def test_c5_per_number_backtick_wrapped_range_fails():
    # Realistic per-number markdown wrapping.
    plan = GOOD_PLAN.replace(GPU_LINE, "Estimated GPU-hours (total): `4`-`8`")
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "FAIL"
    assert "range" in r.detail


def test_c5_spaced_backtick_range_fails():
    plan = GOOD_PLAN.replace(GPU_LINE, "Estimated GPU-hours (total): `4` - 8")
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "FAIL"
    assert "range" in r.detail


def test_c5_backtick_range_understating_auto_approve_cap_fails():
    # The auto-approve-cap understatement shape: `40`-200 previously read
    # as 40 GPU-h — under the 100 GPU-h autonomous auto-approve cap while
    # the stated worst case is 200.
    plan = GOOD_PLAN.replace(GPU_LINE, "Estimated GPU-hours (total): `40`-200")
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "FAIL"
    assert "range" in r.detail


def test_c5_parenthetical_phases_annotation_passes():
    # Must-keep shape (Codex r2 Minor): a parenthetical annotation carrying a
    # digit-dash-digit token after the value is an annotation, not a range.
    plan = GOOD_PLAN.replace(GPU_LINE, "`Estimated GPU-hours (total): 4` (phases 1-3)")
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "PASS"


def test_c5_backtick_range_in_table_cell_fails():
    # Same shape inside a markdown table cell (the #614 context that made
    # the backtick an annotation stop in the first place).
    plan = GOOD_PLAN.replace(
        GPU_LINE,
        "| **Total (pod)** | 1× A100-80 | Estimated GPU-hours (total): `4`-`8` (with margin) |",
    )
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "FAIL"
    assert "range" in r.detail


def test_c5_range_inside_inline_code_span_fails():
    # Same shape inside a surrounding inline-code span: the whole
    # label+range sits in one span, so the only backtick after the value
    # is the CLOSING span delimiter — the range must still be detected.
    plan = GOOD_PLAN.replace(
        GPU_LINE, "Budget: `Estimated GPU-hours (total): 4-8`. Wall ~2 h including review."
    )
    _, by_id = _run(plan)
    r = by_id["c5_gpu_hours"]
    assert r.status == "FAIL"
    assert "range" in r.detail


# ─── Check 6 — reused-artifact fitness ─────────────────────────────────────


def test_c6_not_triggered_skips():
    assert _status(GOOD_PLAN, "c6_reuse_fitness") == "SKIP"


def test_c6_reuse_without_fitness_warns():
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "WARN"
    assert "fitness" in r.detail


def test_c6_fitness_with_four_letters_passes():
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\nFitness check: (a) same recipe verified against adapter_config.json; (b) valid measurement regime; (c) required cells present; (d) single-variable change preserved.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "PASS"
    assert "4/11" in r.detail


def test_c6_fitness_counts_item_i_in_widened_class():
    # Pins the [a-i] regex widening (#871): exactly four counted letters, one of
    # them (i) — a regression to [a-h] would count 3 and WARN instead of PASS.
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\nFitness check: (a) same recipe verified against adapter_config.json; (b) valid measurement regime; (c) required cells present; (i) throughput fitness — inner loop batched, device parametrized.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "PASS"
    assert "4/11" in r.detail


def test_c6_fitness_counts_item_j_in_widened_class():
    # Pins the [a-j] regex widening (#941): exactly four counted letters, one of
    # them (j) — a regression to [a-i] would count 3 and WARN instead of PASS.
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\nFitness check: (a) same recipe verified against adapter_config.json; (b) valid measurement regime; (c) required cells present; (j) pairwise provenance coherence — input last-commit predates the dependent capture.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "PASS"
    assert "4/11" in r.detail


def test_c6_fitness_counts_item_k_in_widened_class():
    # Pins the [a-k] regex widening (#1366): exactly four counted letters, one of
    # them (k) — a regression to [a-j] would count 3 and WARN instead of PASS.
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\nFitness check: (a) same recipe verified against adapter_config.json; (b) valid measurement regime; (c) required cells present; (k) parent-lineage coherence — parent branch fully merged, empty unmerged diff; realized row count reconciles with the declared corpus.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "PASS"
    assert "4/11" in r.detail


def test_c6_fitness_letters_beyond_k_do_not_count():
    # Upper-boundary fixture (#941; decoy moved (k)->(l) at #1366): an unrelated
    # (l) elsewhere in the body must NOT lift a 3-letter fitness attestation to
    # a 4-letter PASS — an over-widening of the class to [a-l]/[a-z] would flip
    # this to PASS.
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\nFitness check: (a) same recipe verified against adapter_config.json; (b) valid measurement regime; (c) required cells present."
        + "\nUnrelated enumeration elsewhere: (l) a non-fitness bullet.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "WARN"


def test_c6_fitness_with_few_letters_warns():
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\nFitness check: (a) same recipe; (b) valid regime.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "WARN"
    assert "(a)–(k)" in r.detail or "eleven" in r.detail


def test_c6_na_no_artifact_reuse_passes():
    # Standalone-line escape (the mid-line form was the self-escape shape —
    # repurposed into test_c6_quoted_na_phrase_does_not_escape below).
    plan = (
        GOOD_PLAN
        + "\nPrior adapters at superkaiba1/explore-persona-space exist; reuse was considered and rejected.\n"
        + "\nN/A — no artifact reuse (adapters exist but this design retrains).\n"
    )
    assert _status(plan, "c6_reuse_fitness") == "PASS"


def test_c6_quoted_na_phrase_does_not_escape():
    # Anti-paste guard: a mid-sentence quote of the escape phrase (the shape a
    # pasted remedy menu produces) must not satisfy the standalone-line escape.
    base = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm.\n"
    )
    assert _status(base, "c6_reuse_fitness") == "WARN"
    quoted = base + (
        "\nReuse was considered and rejected: N/A — no artifact reuse. The remedy menu "
        "says to declare `N/A — no artifact reuse` on its own line.\n"
    )
    assert _status(quoted, "c6_reuse_fitness") == "WARN"


def test_c6_heading_triggers():
    plan = GOOD_PLAN + "\n## 10. Reused-artifact fitness check\n\nNothing here yet.\n"
    _, by_id = _run(plan)
    assert by_id["c6_reuse_fitness"].status == "WARN"


def test_c6_kind_infra_skips():
    plan = GOOD_PLAN + "\nWe reuse adapters from superkaiba1/explore-persona-space.\n"
    assert _status(plan, "c6_reuse_fitness", kind="infra") == "SKIP"


def test_c6_reuse_map_table_without_fitness_word_passes():
    # Durability pin for #1314, modeled on #1090 plan v7's '### D3 — Reuse map'
    # (artifact-reuse (a)–(j) self-attestation) table: a complete per-row  # noqa: RUF003
    # attestation written in artifact-reuse.md's own vocabulary — no 'fitness'
    # word anywhere — must PASS, not WARN "no fitness check found".
    # Doubles as a second grandfather pin (#1366): the fixture's (a)–(j)  # noqa: RUF003
    # heading token still declares under the widened \([jk]\) detector.
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\n### D3 — Reuse map (artifact-reuse (a)–(j) self-attestation)\n"
        + "\n| Artifact | Checks | Verdict |"
        + "\n|---|---|---|"
        + "\n| parent adapter | (a) recipe match; (e) hub-resolves; (h)(i) staged | OK |"
        + "\n| training mix | (f) content identity; (b) valid regime | OK |"
        + "\n| eval JSON | (c) cells present; (d) single-variable; (j) pair-coherent | OK |\n"
    )
    assert "fitness" not in plan.lower()  # keeps the fixture honest vs future GOOD_PLAN edits
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "PASS"


def test_c6_letters_without_declaration_vocab_still_warns():
    # Regression pin: >=4 stray enumeration letters WITHOUT any declaration
    # token (fitness / reuse map / attestation / range token) never PASS —
    # true both pre- and post-#1314.
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\nAcceptance: (a) smoke passes; (b) loss decreases; (c) eval completes; "
        + "(d) uploads verified.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "WARN"


def test_c6_reuse_map_with_few_letters_warns():
    # A bare 'Reuse map' heading (no 'self-attestation', no 'fitness', no
    # (a)–(j)/(a)–(k) range token — guard-asserted below, so this fixture  # noqa: RUF003
    # isolates the reuse[- ]map branch) with <4 letters routes to the MIDDLE branch:
    # the declaration counted, but the letters threshold still gates. A
    # mutant dropping the reuse-map branch fails this test — with no
    # declaration token the fixture would route to the third branch, whose
    # detail lacks "only 2".
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\n### Reuse map\n"
        + "\n(a) recipe match verified; (b) valid measurement regime.\n"
    )
    lowered = plan.lower()
    assert "fitness" not in lowered
    assert "attestation" not in lowered
    assert re.search(r"\(a\)\s*[-–—…]\s*\([jk]\)", plan) is None
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "WARN"
    assert "only 2" in r.detail


def test_c6_range_token_counts_as_declaration():
    # GRANDFATHER pin (#1366): an in-flight plan citing the OLD en-dash (a)–(j)  # noqa: RUF003
    # range token still declares under the widened \([jk]\) detector. No
    # 'fitness', no 'map', no 'attestation' word (guard-asserted), four real
    # item letters.
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\nArtifact checks (a)–(j): (a) recipe; (b) regime; (c) cells; "
        + "(d) single-variable.\n"
    )
    lowered = plan.lower()
    assert "fitness" not in lowered
    assert "map" not in lowered
    assert "attestation" not in lowered
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "PASS"


def test_c6_new_range_token_counts_as_declaration():
    # Pins the CURRENT en-dash (a)–(k) range-token branch (#1366): no  # noqa: RUF003
    # 'fitness', no 'map', no 'attestation' word (guard-asserted), four real
    # item letters.
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\nArtifact checks (a)–(k): (a) recipe; (b) regime; (c) cells; "
        + "(d) single-variable.\n"
    )
    lowered = plan.lower()
    assert "fitness" not in lowered
    assert "map" not in lowered
    assert "attestation" not in lowered
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "PASS"


# ─── Check 7 — replication fidelity ────────────────────────────────────────


def _replication_goal_plan() -> str:
    return GOOD_PLAN.replace(
        "Measure whether benign SFT", "Replicate the paper's finding that benign SFT"
    )


def test_c7_not_triggered_skips():
    assert _status(GOOD_PLAN, "c7_replication_fidelity") == "SKIP"


def test_c7_replication_goal_without_fidelity_warns():
    _, by_id = _run(_replication_goal_plan())
    r = by_id["c7_replication_fidelity"]
    assert r.status == "WARN"
    assert "recipe" in r.detail.lower()


def test_c7_fidelity_vocabulary_passes():
    plan = (
        _replication_goal_plan()
        + "\nWe match the paper's recipe verbatim and name every deviation.\n"
    )
    assert _status(plan, "c7_replication_fidelity") == "PASS"


def test_c7_na_not_a_replication_passes():
    plan = (
        _replication_goal_plan()
        + "\nN/A — not a replication (the Goal's word refers to restarts, not a published finding).\n"
    )
    assert _status(plan, "c7_replication_fidelity") == "PASS"


def test_c7_midprose_phrase_does_not_escape():
    # #1262 red-pin: the phrase mid-prose (the #810 structure, one polarity
    # over) must not escape c7. Pre-fix: doc-global re.search → PASS.
    plan = (
        _replication_goal_plan()
        + "\nThis experiment is not a replication of prior work; we test a new dose axis.\n"
    )
    assert _status(plan, "c7_replication_fidelity") == "WARN"


def test_c7_backtick_wrapped_escape_does_not_escape():
    # Mirror of test_c12_backtick_wrapped_escape_at_bullet_start_does_not_escape:
    # the pasted bounce-brief bullet shape must not escape.
    plan = (
        _replication_goal_plan()
        + "\n- `N/A — not a replication` (declare on its own line per the bounce brief).\n"
    )
    assert _status(plan, "c7_replication_fidelity") == "WARN"


def test_c7_pasted_warn_detail_does_not_self_satisfy():
    # Anti-paste guard (#1237 de-fang class): pre-fix, the WARN detail's own
    # "paper's data + recipe" wording satisfied the vocabulary branch when
    # pasted back into the plan (WARN → PASS).
    base = _replication_goal_plan()
    _, by_id = _run(base)
    r = by_id["c7_replication_fidelity"]
    assert r.status == "WARN"
    pasted = base + f"\n{r.detail}\n"
    assert _status(pasted, "c7_replication_fidelity") == "WARN"


def test_c7_na_bulleted_standalone_passes():
    # Green pin (#1262, stats-lens concern): the planner.md documented
    # bulleted declaration form is recognized (leading list markers
    # lstripped, trailing prose tolerated by re.match).
    plan = (
        _replication_goal_plan()
        + "\n- N/A — not a replication (the Goal's word refers to restarts).\n"
    )
    assert _status(plan, "c7_replication_fidelity") == "PASS"


def test_c7_kind_infra_skips():
    assert _status(_replication_goal_plan(), "c7_replication_fidelity", kind="infra") == "SKIP"


# ─── Check 8 — success + kill criteria ─────────────────────────────────────


def test_c8_good_plan_detail_names_anchors_and_sections():
    _, by_id = _run(GOOD_PLAN)
    r = by_id["c8_success_kill_criteria"]
    assert r.status == "PASS"
    assert "kill" in r.detail.lower()
    assert "Decision gates" in r.detail  # carrier section named


def test_c8_tldr_what_would_change_my_mind_alone_is_not_kill_criteria():
    # Binding round-1 reconciler fix: the §0.0/TL;DR "What would change my
    # mind" line is template conformance — with success vocabulary present
    # and no kill criteria elsewhere, check 8 must NOT pass.
    plan = GOOD_PLAN.replace(KILL_SENT, "").replace(CRITERIA_HEADING, "## 7. Decision gates")
    assert "What would change my mind" in plan  # TL;DR line intact
    _, by_id = _run(plan)
    r = by_id["c8_success_kill_criteria"]
    assert r.status == "WARN"
    assert "kill criteria" in r.detail.lower()
    assert "What would change my mind" in r.detail


def test_c8_success_missing_warns():
    plan = GOOD_PLAN.replace(SUCCESS_SENT, "").replace(CRITERIA_HEADING, "## 7. Criteria")
    _, by_id = _run(plan)
    r = by_id["c8_success_kill_criteria"]
    assert r.status == "WARN"
    assert "success criteria" in r.detail.lower()


def test_c8_both_missing_fails_for_experiment():
    plan = (
        GOOD_PLAN.replace(SUCCESS_SENT, "")
        .replace(KILL_SENT, "")
        .replace(CRITERIA_HEADING, "## 7. Criteria")
    )
    _, by_id = _run(plan)
    r = by_id["c8_success_kill_criteria"]
    assert r.status == "FAIL"
    assert "No gates" in r.detail  # the gates-escape distinction is explained


def test_c8_both_missing_warns_for_exempt_kinds():
    plan = (
        GOOD_PLAN.replace(SUCCESS_SENT, "")
        .replace(KILL_SENT, "")
        .replace(CRITERIA_HEADING, "## 7. Criteria")
    )
    assert _status(plan, "c8_success_kill_criteria", kind="infra") == "WARN"


def test_c8_empty_carrier_section_is_not_solid():
    # "Non-contradictory in form" = both present AND each carrier section
    # non-empty (≥ 80 chars). An empty `## Kill criteria` heading at EOF
    # does not count.
    plan = (
        GOOD_PLAN.replace(KILL_SENT, "").replace(CRITERIA_HEADING, "## 7. Decision gates")
        + "\n## Kill criteria\n"
    )
    _, by_id = _run(plan)
    r = by_id["c8_success_kill_criteria"]
    assert r.status == "WARN"
    assert "carrier" in r.detail.lower()


# ─── Check 8 — kind-aware TL;DR-kill calibration (#1291) ───────────────────

# GOOD_PLAN's mandated §0.0 line, verbatim (surgery anchor for the #1291 pins).
TLDR_CHANGE_MIND_LINE = (
    "- **What would change my mind:** A shift larger than the run-to-run spread "
    "would mean benign data alone moves persona expression."
)


def test_c8_exempt_kind_tldr_change_my_mind_satisfies_kill():
    # #1279 shape (#1291): an exempt-kind plan with solid success criteria
    # and a solid §0.0 "What would change my mind" line PASSes — for a
    # code/infra change the mandated change-my-mind line IS the revert
    # criterion; kind: experiment keeps requiring kill criteria outside the
    # TL;DR (pinned by test_c8_tldr_what_would_change_my_mind_alone_is_not_
    # kill_criteria, which runs the same fixture at the default kind).
    plan = GOOD_PLAN.replace(KILL_SENT, "").replace(CRITERIA_HEADING, "## 7. Decision gates")
    assert "What would change my mind" in plan  # TL;DR line intact
    _, by_id = _run(plan, kind="infra")
    r = by_id["c8_success_kill_criteria"]
    assert r.status == "PASS"
    assert "TL;DR" in r.detail  # detail names the TL;DR acceptance
    assert _status(plan, "c8_success_kill_criteria", kind="analysis") == "PASS"


def test_c8_exempt_kind_no_change_my_mind_line_still_warns():
    # #1276 shape (#1291): kill family missing AND no §0.0 change-my-mind
    # line anywhere — a true-positive WARN, whose detail now embeds the
    # standard remedy sentence (self-dispositioning; no hand-written waiver
    # prose needed).
    plan = (
        GOOD_PLAN.replace(KILL_SENT, "")
        .replace(CRITERIA_HEADING, "## 7. Decision gates")
        .replace(TLDR_CHANGE_MIND_LINE, "")
    )
    assert "What would change my mind" not in plan
    _, by_id = _run(plan, kind="infra")
    r = by_id["c8_success_kill_criteria"]
    assert r.status == "WARN"
    assert "Standard remedy" in r.detail


def test_c8_exempt_kind_success_missing_not_rescued_by_tldr_kill():
    # The success family is never waived (#1291): with success criteria
    # missing, a solid TL;DR change-my-mind line must NOT flip an
    # exempt-kind plan to PASS (kills a mutant that drops the success
    # requirement from the exempt-kind TL;DR branch).
    plan = (
        GOOD_PLAN.replace(SUCCESS_SENT, "")
        .replace(KILL_SENT, "")
        .replace(CRITERIA_HEADING, "## 7. Decision gates")
    )
    assert "What would change my mind" in plan  # TL;DR kill intact
    _, by_id = _run(plan, kind="infra")
    r = by_id["c8_success_kill_criteria"]
    assert r.status == "WARN"
    assert "success criteria" in r.detail.lower()


def test_c8_exempt_kind_thin_tldr_carrier_not_solid():
    # Carrier discipline holds inside the TL;DR (#1291): a <80-char §0.0
    # stub whose change-my-mind line is the whole section is NOT a solid
    # kill carrier, so the exempt-kind acceptance does not fire.
    start = GOOD_PLAN.index("## 0.0 TL;DR (plain English)")
    end = GOOD_PLAN.index("## 1. Goal")
    plan = (
        GOOD_PLAN[:start]
        + "## 0.0 TL;DR (plain English)\n\n- What would change my mind: a crash.\n\n"
        + GOOD_PLAN[end:]
    )
    plan = plan.replace(KILL_SENT, "").replace(CRITERIA_HEADING, "## 7. Decision gates")
    assert _status(plan, "c8_success_kill_criteria", kind="infra") == "WARN"


# ─── Check 9 — conditions + seeds ──────────────────────────────────────────


def test_c9_kind_infra_skips():
    assert _status(GOOD_PLAN, "c9_conditions_seeds", kind="infra") == "SKIP"


def test_c9_missing_seeds_warns():
    plan = GOOD_PLAN.replace(
        SEEDS_SENTENCE, "Three runs per condition share the same eval prompts."
    )
    _, by_id = _run(plan)
    r = by_id["c9_conditions_seeds"]
    assert r.status == "WARN"
    assert "seeds" in r.detail


def test_c9_missing_conditions_warns():
    plan = GOOD_PLAN.replace("## 3. Conditions", "## 3. Setup").replace("What it tests", "Purpose")
    _, by_id = _run(plan)
    r = by_id["c9_conditions_seeds"]
    assert r.status == "WARN"
    assert "conditions" in r.detail


# ─── Check 10 — marker-recipe acknowledgment ───────────────────────────────


def test_c10_not_triggered_skips():
    assert _status(GOOD_PLAN, "c10_marker_recipe") == "SKIP"


def test_c10_marker_plan_without_recipe_warns():
    plan = GOOD_PLAN + "\nThe dependent variable is marker-leakage measured at token id 83399.\n"
    _, by_id = _run(plan)
    r = by_id["c10_marker_recipe"]
    assert r.status == "WARN"
    assert "marker-training-recipe" in r.detail


def test_c10_recipe_without_bystander_warns():
    plan = (
        GOOD_PLAN
        + "\nThe dependent variable is marker-leakage at token id 83399; we stop in the band-stop window [5, 12] nat.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c10_marker_recipe"]
    assert r.status == "WARN"
    assert "bystander" in r.detail


def test_c10_recipe_and_bystander_pass():
    plan = (
        GOOD_PLAN
        + "\nThe dependent variable is marker-leakage at token id 83399; we follow .claude/rules/marker-training-recipe.md and gate the anchor on bystander resolution.\n"
    )
    assert _status(plan, "c10_marker_recipe") == "PASS"


def test_c10_fence_only_marker_vocab_does_not_trigger():
    plan = GOOD_PLAN + "\n```python\nMARKER = ' ※'  # token id 83399\n```\n"
    assert _status(plan, "c10_marker_recipe") == "SKIP"


def test_c10_kind_infra_skips():
    plan = GOOD_PLAN + "\nmarker-leakage at 83399\n"
    assert _status(plan, "c10_marker_recipe", kind="infra") == "SKIP"


# ─── Check 11 — dry-run test coverage ──────────────────────────────────────

DRYRUN_SMOKE = (
    "\n## 6. Verification\n\n"
    "Post-merge acceptance: run `uv run python scripts/autonomous_session_watch.py "
    "--dry-run --infra-drain-only` against the live queue file and eyeball the "
    "would-dispatch lines.\n"
)


def test_c11_kind_experiment_skips():
    plan = GOOD_PLAN + DRYRUN_SMOKE
    assert _status(plan, "c11_dryrun_test_coverage", kind="experiment") == "SKIP"


def test_c11_no_dryrun_mention_skips():
    assert _status(GOOD_PLAN, "c11_dryrun_test_coverage", kind="infra") == "SKIP"


def test_c11_smoke_without_dryrun_test_warns():
    # The #596/#607/#633 pattern: a --dry-run acceptance smoke + a
    # success-path-only test list.
    plan = (
        GOOD_PLAN
        + DRYRUN_SMOKE
        + "\nTests: `test_drain_dispatches_ripe_tasks`, `test_drain_respects_concurrency_cap`.\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c11_dryrun_test_coverage"]
    assert r.status == "WARN"
    assert "dry-run kwarg" in r.detail
    assert "#633" in r.detail


def test_c11_kind_batch_triggers():
    plan = GOOD_PLAN + DRYRUN_SMOKE
    assert _status(plan, "c11_dryrun_test_coverage", kind="batch") == "WARN"


def test_c11_test_identifier_with_dryrun_token_passes():
    plan = (
        GOOD_PLAN
        + DRYRUN_SMOKE
        + "\nTests: `test_infra_drain_dry_run_dispatches_nothing` — the drain pass under dry_run posts no markers and spawns no sessions.\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c11_dryrun_test_coverage"]
    assert r.status == "PASS"
    assert "dry" in r.detail.lower()


def test_c11_descriptive_kwarg_test_line_passes():
    plan = (
        GOOD_PLAN
        + DRYRUN_SMOKE
        + "\nAdd a test exercising `dry_run=True` on the new drain pass (no dispatch, no markers, no pod calls).\n"
    )
    assert _status(plan, "c11_dryrun_test_coverage", kind="infra") == "PASS"


def test_c11_smoke_command_sharing_line_with_pytest_path_does_not_self_certify():
    # The #633 v1 false-PASS shape (caught on the calibration run against the
    # real corpus): ONE "Verification commands" line carrying both the
    # success-path pytest invocation (a test_ identifier) and the --dry-run
    # smoke command. Flag occurrences are stripped before the tier-1 scan,
    # so this line is not evidence of a dry_run-exercising test.
    plan = (
        GOOD_PLAN
        + "\n- **Verification commands:** `uv run pytest tests/test_autonomous_session_watch.py -x`; "
        "`uv run python scripts/autonomous_session_watch.py --dry-run --infra-drain-only`.\n"
    )
    assert _status(plan, "c11_dryrun_test_coverage", kind="infra") == "WARN"


def test_c11_smoke_sentence_mentioning_test_suite_does_not_self_certify():
    # The bare `--dry-run` flag co-occurring with the word "test" (the §6
    # "run the smoke, then the test suite" sentence shape) is not evidence
    # of a dry_run-exercising test.
    plan = (
        GOOD_PLAN
        + "\n## 6. Verification\n\nRun the `--dry-run` smoke, then run the full test suite.\n"
    )
    assert _status(plan, "c11_dryrun_test_coverage", kind="infra") == "WARN"


def test_c11_na_escape_passes():
    # Standalone-line escape (the mid-line form was the self-escape shape).
    plan = (
        GOOD_PLAN
        + "\nworktree_audit.py already supports --dry-run; this plan does not touch that path.\n"
        + "\nN/A — no dry-run smoke.\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c11_dryrun_test_coverage"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


def test_c11_pasted_warn_detail_does_not_self_satisfy():
    # Anti-paste guard (#4.2 c11 de-fang): pre-fix, the WARN detail's own
    # `dry_run=True` wording satisfied the kwarg evidence branch when pasted
    # back into the plan; the reworded detail must not.
    base = GOOD_PLAN + DRYRUN_SMOKE
    _, by_id = _run(base, kind="infra")
    r = by_id["c11_dryrun_test_coverage"]
    assert r.status == "WARN"
    pasted = base + f"\n{r.detail}\n"
    assert _status(pasted, "c11_dryrun_test_coverage", kind="infra") == "WARN"


# ─── Check 12 — battery multiplier + batched commitment ────────────────────

BATTERY_SENT = "We run a 1000-draw permutation null battery over the pooled per-cell deltas."
BATTERY_ARITH = (
    "Basis: 1000 draws × 24 cells × 3 statistics at ~0.02 s/draw ≈ 0.4 h projected wall."
)
BATTERY_BATCHED = "Implementation: batched subset-sum GEMM over all draws via `perm_null_draws`."


def test_c12_not_triggered_skips():
    assert _status(GOOD_PLAN, "c12_battery_multiplier") == "SKIP"


def test_c12_kind_infra_skips():
    plan = GOOD_PLAN + f"\n{BATTERY_SENT}\n"
    assert _status(plan, "c12_battery_multiplier", kind="infra") == "SKIP"


def test_c12_bootstrap_ci_alone_does_not_trigger():
    # A bare bootstrap CI (cheap post-hoc stat, ubiquitous in plans) is a
    # deliberate NON-trigger — no battery framing, no >=100 draw count.
    plan = GOOD_PLAN + "\nWe report a bootstrap 95% CI over per-seed deltas.\n"
    assert _status(plan, "c12_battery_multiplier") == "SKIP"


def test_c12_battery_with_arithmetic_and_batched_passes():
    plan = (
        GOOD_PLAN + f"\n## 12. Null battery\n\n{BATTERY_SENT}\n{BATTERY_ARITH}\n{BATTERY_BATCHED}\n"
    )
    assert _status(plan, "c12_battery_multiplier") == "PASS"


def test_c12_battery_missing_arithmetic_fails():
    plan = GOOD_PLAN + f"\n## 12. Null battery\n\n{BATTERY_SENT}\n{BATTERY_BATCHED}\n"
    _, by_id = _run(plan)
    r = by_id["c12_battery_multiplier"]
    assert r.status == "FAIL"
    assert "multiplier arithmetic" in r.detail


def test_c12_battery_missing_batched_commitment_fails():
    plan = GOOD_PLAN + f"\n## 12. Null battery\n\n{BATTERY_SENT}\n{BATTERY_ARITH}\n"
    _, by_id = _run(plan)
    r = by_id["c12_battery_multiplier"]
    assert r.status == "FAIL"
    assert "batched" in r.detail


def test_c12_na_standalone_declaration_passes():
    plan = GOOD_PLAN + (
        f"\n{BATTERY_SENT}\n"
        "N/A — no draw battery (the battery mention quotes the sibling's methodology).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c12_battery_multiplier"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


def test_c12_na_bullet_form_standalone_passes():
    plan = GOOD_PLAN + f"\n{BATTERY_SENT}\n- N/A — no draw battery (incidental mention).\n"
    assert _status(plan, "c12_battery_multiplier") == "PASS"


def test_c12_quoted_na_phrase_does_not_escape():
    # A mid-sentence quoted escape phrase (the pasted-bounce-brief
    # self-escape channel) is NOT a standalone declaration line and must
    # not escape — the c13/c18/c24/c25 anti-paste twin.
    plan = GOOD_PLAN + (
        f"\n{BATTERY_SENT} N/A — no draw battery (quoting the sibling's methodology).\n"
        "\nThe remedy menu says to declare 'N/A — no draw battery' on its own line.\n"
    )
    assert _status(plan, "c12_battery_multiplier") == "FAIL"


def test_c12_pasted_fail_detail_does_not_self_satisfy():
    # The project convention pastes bounce text verbatim into revised plans:
    # the FAIL detail supplies neither a standalone N/A line nor draw
    # arithmetic (its example product is deliberately mult-token-free).
    _, by_id = _run(GOOD_PLAN + f"\n{BATTERY_SENT}\n")
    detail = by_id["c12_battery_multiplier"].detail
    replan = GOOD_PLAN + f"\n{BATTERY_SENT}\n\n{detail}\n"
    assert _status(replan, "c12_battery_multiplier") == "FAIL"


def test_standalone_na_wrapped_forms_stay_unrecognized():
    # #1238 reasoned no-change red-pin: wrapped declarations are
    # DELIBERATELY unrecognized — every trailing-tolerant wrapper widening
    # lets a verbatim paste of the adversarial-planner SKILL.md
    # canonical-phrases block (line-start backtick-wrapped phrases, nearly
    # all helper-routed since #1237/#1262) self-declare many escapes at
    # once (the #810 polarity). See the helper docstring for
    # the full analysis before "fixing" this.
    for line in [
        # the live #1090 plans/v1.md:369 shape (trailing scope prose):
        "- `N/A — no draw battery` beyond the #1074-parity 2000-draw paired bootstrap.",
        # pure wrapped-alone shapes, all three wrapper chars:
        "- `N/A — no draw battery`",
        "- 'N/A — no draw battery'",
        '- "N/A — no draw battery"',
        "`N/A — no draw battery`",
    ]:
        assert not verify_plan._standalone_na_declared(line, r"no draw battery"), line


def test_c12_backtick_wrapped_escape_at_bullet_start_does_not_escape():
    # Integration-level twin of the helper pin above: the live #1090 shape
    # inside a full plan leaves c12 on its affirmative route (here: FAIL).
    plan = GOOD_PLAN + (
        f"\n{BATTERY_SENT}\n- `N/A — no draw battery` beyond the sibling-parity paired bootstrap.\n"
    )
    assert _status(plan, "c12_battery_multiplier") == "FAIL"


def test_skillmd_canonical_escape_block_never_self_escapes():
    # Live-surface coupling pin (#1238): NO line of the canonical-phrases
    # source file may parse as a standalone declaration for ANY call-site
    # tail. The backtick wrapping there is load-bearing anti-paste armor:
    # under the CURRENT recognizer an UNWRAPPED phrase at line start in a
    # pasted copy of the block WOULD self-declare. A future reflow/unwrap
    # of the block goes red here, forcing a deliberate decision.
    tails = re.findall(r"_standalone_na_declared\(\s*plan,\s*r\"(.*?)\"", _SCRIPT.read_text(), re.S)
    # Extraction-guard residual: the regex matches only direct
    # `(plan, r"...")` call sites — a future call site whose first arg is
    # not literally `plan` (or that uses r'...' quoting) is silently
    # uncovered by this pin while the floor stays green; the floor catches
    # shrinkage, not omission.
    assert len(tails) >= 23, tails  # extraction guard: 23 plan-arg call sites (#1262; c4 #1277)
    text = (REPO_ROOT / ".claude/skills/adversarial-planner/SKILL.md").read_text()
    for tail in tails:
        assert not verify_plan._standalone_na_declared(text, tail), tail
    # POSITIVE CONTROL (power check): prove the pin has teeth — the block is
    # NOT fence-masked into vacuity, and the backtick IS the load-bearing
    # armor. Unwrapping one phrase in a copy of the doc must flip the helper
    # to True; a future unbalanced fence above the block would break this.
    unwrapped = text.replace(
        "`N/A — no registered verdict lattice`",
        "N/A — no registered verdict lattice",
    )
    assert verify_plan._standalone_na_declared(unwrapped, r"no registered verdict lattice")


def test_skillmd_canonical_block_states_unwrapped_contract():
    # Durability pin for the D2 prose sentence: the unwrapped-declaration
    # contract must stay stated at the surface bounce briefs are composed
    # from.
    text = (REPO_ROOT / ".claude/skills/adversarial-planner/SKILL.md").read_text()
    assert "must be UNWRAPPED" in text
    assert "_standalone_na_declared" in text


def test_skillmd_canonical_escapes_sync_with_docstring():
    # Generative docstring->SKILL.md sync pin (#1264). Ends the per-check pin
    # whack-a-mole (#1042 c21, #1194 c32, #1246 c33, #1260 c34 each
    # hand-registered one phrase; c26-c30 were missed entirely): every escape
    # phrase the verify_plan module docstring registers must appear
    # backtick-wrapped in the adversarial-planner SKILL.md canonical block.
    # Whitespace is normalized on BOTH sides because SKILL.md legitimately
    # wraps long phrases across indented continuation lines (the check-31
    # alias wraps today). One-directional by design: the docstring is updated
    # when a check lands (ground truth); SKILL.md is the consumer surface
    # that drifts behind it. FORMAT-DISCIPLINE CONTRACT (binding): a future
    # docstring phrase must keep the double-backtick wrapping + the
    # `N/A —` / `Durability pin: N/A` prefix convention, or it evades
    # extraction here (the floor catches shrinkage, not a never-extracted
    # phrase).
    doc = verify_plan.__doc__
    section = doc[doc.index("Canonical N/A escape phrases") : doc.index("WARN semantics")]
    phrases = re.findall(r"``((?:N/A —|Durability pin: N/A)[^`]+?)``", section)
    # Extraction guard (never-self-escapes precedent, :1232): 29 phrases at
    # pin time. A shrinking count means the docstring format drifted and the
    # parser silently under-covers — fix the regex; never lower this floor
    # except for a deliberate check retirement (state which check).
    assert len(phrases) >= 29, (len(phrases), phrases)  # c4 registered (#1277)
    # Code->docstring leg: every `_standalone_na_declared` call-site tail must
    # match somewhere in the docstring section, so a new check's recognizer
    # cannot land unregistered (the SKILL.md asserts below then propagate the
    # registration to the consumer surface).
    tails = re.findall(r"_standalone_na_declared\(\s*plan,\s*r\"(.*?)\"", _SCRIPT.read_text(), re.S)
    assert len(tails) >= 12, tails  # extraction guard, mirrors :1232
    for tail in tails:
        assert re.search(tail, section), f"call-site tail not registered in docstring: {tail!r}"
    text = (REPO_ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md").read_text()
    anchor = text.index("Canonical N/A escape phrases")
    block = text[anchor : text.index("bounce to the planner", anchor)]
    norm_block = re.sub(r"\s+", " ", block)
    missing = []
    for phrase in phrases:
        wrapped = "`" + re.sub(r"\s+", " ", phrase).strip() + "`"
        if wrapped not in norm_block:
            missing.append(wrapped)
    assert not missing, f"escape phrases not in SKILL.md canonical block: {missing}"


def test_own_line_remedies_carry_unwrapped_clarifier():
    # Durability pin (#1263): every runtime string in verify_plan.py that
    # teaches the "on its own line" escape declaration must also say
    # "unwrapped" — the FAIL-detail remedy a planner sees at bounce time
    # must not teach the wrapped form _standalone_na_declared rejects
    # (#1090 c12 bounce; #1238 reasoned no-change on the recognizer).
    # AST-based: Python folds adjacent string literals into one Constant
    # (split-literal remedies are seen whole); comments are not in the AST.
    src = (REPO_ROOT / "scripts" / "verify_plan.py").read_text()
    offenders: list[tuple[int, str]] = []
    clarified = 0
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if "on its own line" not in node.value:
            continue
        if "unwrapped" in node.value:
            clarified += 1
        else:
            offenders.append((node.lineno, node.value[:90]))
    assert not offenders, f"own-line remedies missing the unwrapped clarifier: {offenders}"
    # 19 sites (#1263) + c34 exemplar + c7's remedy/pass-detail (#1262) + c4's (#1277)
    # + c36's remedy (#1375) = 25 live
    assert clarified >= 20


def test_skillmd_c7_exception_clause_retired():
    # Durability pin (#1262): the c7 bare-phrase exception is retired from
    # BOTH documentation surfaces; only the check-31 exception remains
    # DOCUMENTED (c4's same-class escape is an undocumented bug tracked as
    # its own workflow-fix candidate, deliberately NOT documented as an
    # exception — do not use the words "matches its bare" anywhere new).
    for path in (_SCRIPT, REPO_ROOT / ".claude/skills/adversarial-planner/SKILL.md"):
        text = path.read_text()
        assert "matches its bare" not in text, path
        assert "check 31" in text, path


def test_c12_kind_analysis_warns_not_fails():
    # Same evidence gap as the missing-arithmetic FAIL case, but kind=analysis
    # degrades to WARN and the overall verdict stays ok.
    plan = GOOD_PLAN + f"\n## 12. Null battery\n\n{BATTERY_SENT}\n{BATTERY_BATCHED}\n"
    ok, by_id = _run(plan, kind="analysis")
    assert by_id["c12_battery_multiplier"].status == "WARN"
    assert ok is True


def test_c12_false_pass_regression_unrelated_evidence_fails():
    # The #810-shaped false-PASS class (the REGISTERED anti-false-green
    # fixture): battery named in one section; an unrelated grid-only product,
    # a bare rule-file citation, and "vLLM batched" boilerplate all live in a
    # DIFFERENT section far outside the trigger window. Document-global
    # evidence certified exactly this shape; window-scoped evidence must FAIL.
    filler = "\n".join(f"Filler paragraph line {i} with no sizing content." for i in range(20))
    plan = (
        GOOD_PLAN
        + f"\n## 12. Null battery\n\n{BATTERY_SENT}\n\n"
        + filler
        + "\n\n## 13. Footprint\n\n"
        + "Store footprint: 24 cells × 3 seeds float32 tensors (~2 GB).\n"
        + "See .claude/rules/vectorize-many-cell-fits.md for the compute-shape rule.\n"
        + "Generation uses vLLM batched decoding.\n"
    )
    _, by_id = _run(plan)
    assert by_id["c12_battery_multiplier"].status == "FAIL"


def test_c12_grid_only_product_in_window_fails():
    # An in-window grid-only product (no draw-bearing factor) plus an
    # in-window batched token still FAILs — the forgotten draw multiplier is
    # exactly what must be present.
    plan = (
        GOOD_PLAN
        + f"\n## 12. Null battery\n\n{BATTERY_SENT}\n"
        + "The grid is 34 × 50 × 28 fits, batched via `perm_null_draws`.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c12_battery_multiplier"]
    assert r.status == "FAIL"
    assert "draw-bearing" in r.detail


def test_c12_graded_scale_draws_does_not_trigger():
    # Judge-scale vocabulary ("graded 0-100 draws", en-dash variant in the
    # fixture below) is not a battery — the count arm's lookbehind excludes
    # range/scale-dash-preceded numbers (calibration false-FAIL on #779 v1).
    plan = (
        GOOD_PLAN
        + "\nJudge N=5 graded 0–100 draws, temp 1.0, drop-never-coerce; graded 0-100 draws.\n"
    )
    assert _status(plan, "c12_battery_multiplier") == "SKIP"


def test_c12_fenced_battery_does_not_trigger():
    # Battery vocabulary appearing ONLY inside a code fence is not a battery
    # plan — pins the fence-masked trigger path.
    plan = (
        GOOD_PLAN
        + "\n## 12. Example\n\n```\nrun_battery --draws 1000  # permutation null battery example\n```\n"
    )
    assert _status(plan, "c12_battery_multiplier") == "SKIP"


def test_c12_window_radius_boundary_after_refactor():
    # Radius pin (task #937), updated for #1086 arith-anchored windows: an
    # arithmetic line now opens its own window BY DESIGN, so the MOVING
    # evidence is the batched COMMITMENT. Commit exactly 15 raw lines below
    # the arith anchor counts; one line farther does not.
    anchor_arith = (
        "Basis: 1000 draws × 24 cells × 3 statistics at ~0.02 s/draw ≈ 0.4 h projected wall."
    )
    commit = "Implementation: one batched subset-sum GEMM over the full draw matrix."
    assert verify_plan._MULT_ARITH_RE.search(anchor_arith)
    assert not verify_plan._BATTERY_TRIGGER_RE.search(commit)
    assert not verify_plan._MULT_ARITH_RE.search(commit)

    def plan_with_gap(n_blank: int) -> str:
        # Commit lands (n_blank + 1) raw lines below the arith anchor line.
        return (
            GOOD_PLAN
            + "\n## 12. Null battery\n\n"
            + BATTERY_SENT
            + "\n"
            + anchor_arith
            + "\n"
            + "\n" * n_blank
            + commit
            + "\n"
        )

    assert _status(plan_with_gap(14), "c12_battery_multiplier") == "PASS"  # +15 lines: in
    assert _status(plan_with_gap(15), "c12_battery_multiplier") == "FAIL"  # +16 lines: out


# #1086: the #833 v8 L149 sizing-paragraph excerpt (failing tokens preserved
# verbatim in the STRING below: "draws x batteries(...)", "draws x (6 arms",
# "draws x ~3 quantities" — mult sign spelled ASCII here for RUF003;
# "2,000 family-clustered draws" / "Bootstrap-battery" are NOT battery
# triggers — asserted in the far-block test, so only the #1086
# arith-anchored window can reach the block).
V8_L149_SIZING_BLOCK = (
    "**Bootstrap-battery multiplier arithmetic (per "
    "`.claude/rules/vectorize-many-cell-fits.md`):** total_calls = draws × "
    "batteries(arms + paired diffs) × layers. Phase N2: 2,000 family-clustered draws × "
    "(6 arms + 4 paired diffs) × 3 layers = 60,000 rank recomputations, each over a "
    "≤291-length CACHED per-cell prediction array. Phase N0: 2,000 family-clustered "
    "draws × ~3 quantities (emission ρ + on-policy/control chain ρ) × 3 layers ≈ 18,000 "
    "Spearman calls over 480-length cached arrays. Batched-implementation commitment: "
    "rank/Spearman draws over cached arrays, vectorized numpy, seconds per battery."
)


def test_c12_sizing_block_far_from_battery_registration_passes():
    # The #833 v8 layout split: battery registered early (§4/§6), the sizing
    # paragraph 20+ raw lines below — outside every battery-trigger window.
    # The sizing line itself is NOT a battery trigger (asserted), so only
    # the #1086 arith-anchored window can rescue it.
    assert not verify_plan._BATTERY_TRIGGER_RE.search(V8_L149_SIZING_BLOCK)
    filler = "\n".join(f"Filler paragraph line {i} with no sizing content." for i in range(20))
    plan = (
        GOOD_PLAN
        + f"\n## 12. Null battery\n\n{BATTERY_SENT}\n\n"
        + filler
        + "\n\n## 13. Compute sizing\n\n"
        + V8_L149_SIZING_BLOCK
        + "\n"
    )
    assert _status(plan, "c12_battery_multiplier") == "PASS"


def test_c12_v8_multiplier_variant_forms_match_arith():
    # The three #833 v8 L149 multiplier forms the 10-noun whitelist rejected.
    for form in (
        "total_calls = draws × batteries(arms + paired diffs) × layers",
        "2,000 family-clustered draws × (6 arms + 4 paired diffs) × 3 layers",
        "2,000 family-clustered draws × ~3 quantities",
    ):
        assert verify_plan._MULT_ARITH_RE.search(form), form
    # Grid-only products (the #810 false-PASS class) still do NOT match —
    # incl. the required new negative for the widened any-noun grid surface.
    for form in (
        "34 × 50 × 28",
        "layers x 3584",
        "6 arms × 3 layers × 16 folds",
        "Store footprint: 24 cells × 3 seeds float32 tensors (~2 GB).",
    ):
        assert not verify_plan._MULT_ARITH_RE.search(form), form


def test_c12_grid_only_far_sizing_block_still_fails():
    # A grid-only product cannot ANCHOR a window (no draw-bearing factor),
    # so a far sizing block with commit vocabulary alone does not rescue the
    # plan — the #810 v1 class stays mechanically closed under #1086.
    filler = "\n".join(f"Filler paragraph line {i} with no sizing content." for i in range(20))
    plan = (
        GOOD_PLAN
        + f"\n## 12. Null battery\n\n{BATTERY_SENT}\n\n"
        + filler
        + "\n\n## 13. Compute sizing\n\n"
        + "Sizing: 6 arms × 3 layers × 16 folds = 288 fits, batched via `vectorized_mlp_skill`.\n"
    )
    assert _status(plan, "c12_battery_multiplier") == "FAIL"


def test_c12_ascii_x_word_split_artifacts_do_not_match():
    # #1099: with the widened any-noun _GRID_FACTOR, the bare `[×x*]` token  # noqa: RUF003
    # let the engine split words at an internal `x`. All five forms below
    # matched pre-fix (three are verbatim-class realized corpus lines:
    # #763 v6:139, #923 v4:94, #778 v6:674).
    for form in (
        "the max 50 draws are retained per cell",
        "shuffle-null p (1000 per-draw layer-max perms; p floor 0.001)",
        "regenerates the SAME 2000 family-bootstrap index draws (seed 42)",
        "`honest_nulls_maxdraws/` listing (exactly these 5 per trait)",
        "1000 draws xgboost sweep",
        # Digit-tight verb false positive — the ONE tested-class input that
        # discriminates the chosen standalone rule (no match) from the §7
        # letter-boundary fallback (match). Pins the fallback constraint:
        # ANY future _MULT_TOKEN composition must keep this negative
        # (#1099 round-1 statistics-lens Must-Fix).
        "the 2x2 draws its factors",
    ):
        assert not verify_plan._MULT_ARITH_RE.search(form), form
    # Standalone ASCII x (the corpus's real spaced form) still matches.
    for form in (
        "10,000 draws x 24 cells, batched",
        "4 draws x 492 cells",
        "24 cells x 1000 draws",
    ):
        assert verify_plan._MULT_ARITH_RE.search(form), form


def test_c12_word_split_x_line_does_not_anchor_window():
    # #1099 end-to-end regression: pre-fix, a word-split "index draws" line
    # ANCHORED its own ±15 window AND satisfied evidence (i), so a batched
    # token beside it false-PASSed a battery plan with no real sizing
    # arithmetic anywhere (probe-reproduced PASS pre-fix).
    filler = "\n".join(f"Filler paragraph line {i} with no sizing content." for i in range(20))
    plan = (
        GOOD_PLAN
        + f"\n## 12. Null battery\n\n{BATTERY_SENT}\n\n"
        + filler
        + "\n\n## 13. Refit\n\n"
        + "The refit regenerates the SAME 2000 family-bootstrap index draws (seed 42) per cell.\n"
        + f"{BATTERY_BATCHED}\n"
    )
    _, by_id = _run(plan)
    r = by_id["c12_battery_multiplier"]
    assert r.status == "FAIL"
    assert "draw-bearing" in r.detail


# ─── Check 13 — empirical-null gate p-floor attainability ──────────────────

# Near-verbatim #816 shapes: the synthetic fixtures ARE the incident text
# (v5-shape = must-FAIL, v6-shape = must-PASS; the REAL v5/v6 files are
# exercised by the §6.2 pre-commit calibration, not by this suite). NOTE:
# the n_draws fixtures also trigger c12 (its `n_(draws|perms)\b` trigger) —
# tests below assert the c13 status only, except where noted.
C13_HEADING = "## 12. Success criteria (empirical-null gates)"
C13_TABLE_SMALL = (
    "| Family | Construction | n_draws | Source |\n"
    "|---|---|---|---|\n"
    "| Isotropic | v ~ N(0, I) | 200 | norm-matching |\n"
    "| Cross-trait | the other 2 traits' r_B | 2 | #778 v2 |\n"
    "| PCA top-5 | top-5 PCs of diffs | 5 | #778 v1 |\n"
)
C13_TABLE_BIG = (
    "| Family | Construction | n_draws | Source |\n"
    "|---|---|---|---|\n"
    "| Isotropic | v ~ N(0, I) | 200 | norm-matching |\n"
    "| Cross-trait | rotation draws | 200 | #778 v2 |\n"
    "| PCA top-5 | top-5 PCs of diffs | 200 | #778 v1 |\n"
)
C13_TABLE_19 = "| Family | Construction | n_draws | Source |\n|---|---|---|---|\n| Isotropic | v ~ N(0, I) | 19 | norm-matching |\n"
C13_GATE = (
    "- SUCCESS: real |r| > every honest null family's 97.5th-pct |r| at "
    "one-sided empirical p ≤ 0.05 surviving Benjamini-Hochberg across the 3 traits."
)
C13_GATE_SCOPED = (
    "- SUCCESS: for every STOCHASTIC family (n_draws ≥ 50) one-sided empirical "
    "p ≤ 0.05 surviving Benjamini-Hochberg; cross-trait (n=2) and PCA (n=5) are "
    "descriptive max-comparators outside the BH set, p-floors 1/3 and 1/6 stated inline."
)
C13_GATE_FLOOR_FORM = (
    "- SUCCESS: exceeds EVERY random draw's reduction. One-sided empirical "
    "p ≤ 1/(15+1) ≈ 0.06 (the p-floor at 15 draws)."
)


def _c13_plan(*parts: str) -> str:
    """GOOD_PLAN + a success-criteria section carrying ``parts`` in order."""
    return GOOD_PLAN + "\n" + C13_HEADING + "\n\n" + "\n\n".join(parts) + "\n"


def test_c13_unattainable_gate_816_v5_shape_fails():
    ok, by_id = _run(_c13_plan(C13_TABLE_SMALL, C13_GATE))
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "FAIL"
    assert "n_draws=2" in r.detail
    assert "1/3" in r.detail
    assert "p-floor" in r.detail
    assert ok is False


def test_c13_attainable_gate_passes():
    assert _status(_c13_plan(C13_TABLE_BIG, C13_GATE), "c13_empirical_gate_attainability") == "PASS"


def test_c13_self_consistent_floor_gate_skips():
    # The verbatim #816 v5 Exp-4 floor-form sentence ("p ≤ 1/(15+1) ≈ 0.06")
    # must NOT be read as a decimal-alpha gate — no gate hit, so c13 SKIPs.
    _, by_id = _run(_c13_plan(C13_TABLE_SMALL, C13_GATE_FLOOR_FORM))
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "SKIP"
    assert "no registered empirical-p gate" in r.detail


def test_c13_scoped_gate_816_v6_shape_passes():
    # The #816 v6 fix shape: same small-n table rows, but the gate line
    # carries the draws-explicit scope qualifier `n_draws ≥ 50` — the n=2/5
    # comparators are outside the gate's own declared scope.
    assert (
        _status(_c13_plan(C13_TABLE_SMALL, C13_GATE_SCOPED), "c13_empirical_gate_attainability")
        == "PASS"
    )


def test_c13_not_triggered_skips():
    _, by_id = _run(GOOD_PLAN)
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "SKIP"
    assert "no registered empirical-p gate" in r.detail


def test_c13_gate_without_ndraws_skips():
    _, by_id = _run(_c13_plan(C13_GATE))
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "SKIP"
    assert "no per-family n_draws declarations" in r.detail


@pytest.mark.parametrize("kind", ["experiment", "analysis"])
def test_c13_na_escape_with_detected_gate_warns(kind):
    # #1258 (the #1223 c20 port): the escape is only reachable when a gate WAS
    # detected (the no-gate case SKIPs first), so N/A + detected gate is the
    # masking shape — WARN (non-blocking), never a silent PASS. Same severity
    # both kinds: the co-occurrence is a meta-signal, already sub-FAIL.
    # The c12 escape line keeps the UNRELATED battery-multiplier check (which
    # fires on the fixture's draw vocabulary) out of the overall verdict, so
    # `ok is True` pins that the c13 WARN itself never blocks.
    plan = _c13_plan(C13_TABLE_SMALL, C13_GATE) + (
        "\nN/A — no empirical-null gate (the p ≤ 0.05 mention quotes the sibling's methodology).\n"
        "\nN/A — no draw battery\n"
    )
    ok, by_id = _run(plan, kind=kind)
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "WARN"
    assert ok is True  # WARN never blocks
    assert "co-occurs" in r.detail
    assert "97.5th-pct" in r.detail  # detected gate-line snippet quoted


def test_c13_na_escape_with_gate_but_no_table_warns():
    # Pins the ordering claim: the co-occurrence WARN precedes the
    # no-n_draws-declarations SKIP — a plan with a detected gate + the escape
    # WARNs regardless of whether any n_draws table exists. A misordered port
    # placing the WARN after that SKIP passes the other fixtures but fails
    # this one.
    plan = _c13_plan(C13_GATE) + "\nN/A — no empirical-null gate\n"
    _, by_id = _run(plan)
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "WARN"
    assert "co-occurs" in r.detail


def test_c13_na_escape_without_gate_still_skips():
    # Preserved: with no detected gate the SKIP gate fires before the escape
    # is consulted — the N/A line stays legal and never penalized.
    plan = GOOD_PLAN + "\nN/A — no empirical-null gate\n"
    _, by_id = _run(plan)
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "SKIP"
    assert "no registered empirical-p gate" in r.detail


def test_c13_kind_infra_skips():
    plan = _c13_plan(C13_TABLE_SMALL, C13_GATE)
    assert _status(plan, "c13_empirical_gate_attainability", kind="infra") == "SKIP"


def test_c13_kind_analysis_warns_not_fails():
    # Same offenders as the FAIL case, but kind=analysis degrades to WARN and
    # the overall verdict stays ok (c12 also degrades to WARN under analysis).
    plan = _c13_plan(C13_TABLE_SMALL, C13_GATE)
    ok, by_id = _run(plan, kind="analysis")
    assert by_id["c13_empirical_gate_attainability"].status == "WARN"
    assert ok is True


def test_c13_boundary_floor_equals_alpha_warns():
    # Fraction(1, 20) == Fraction("0.05") exactly: floor == alpha under a
    # non-strict `≤` comparator is the boundary WARN, not an offender.
    _, by_id = _run(_c13_plan(C13_TABLE_19, C13_GATE))
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "WARN"
    assert "exactly" in r.detail


def test_c13_ambiguous_gate_without_family_vocab_warns():
    gate = "- SUCCESS: the permutation read must reach one-sided empirical p ≤ 0.05."
    _, by_id = _run(_c13_plan(C13_TABLE_SMALL, gate))
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "WARN"
    assert "ambiguous tie" in r.detail


def test_c13_prior_work_recap_not_a_gate():
    # The same gate sentence under a Prior Work heading is a recap, not a
    # registration — section scoping keeps it out (under-trigger fails safe).
    plan = GOOD_PLAN + "\n## 2. Prior Work\n\n" + C13_TABLE_SMALL + "\n" + C13_GATE + "\n"
    assert _status(plan, "c13_empirical_gate_attainability") == "SKIP"


def test_c13_fenced_gate_does_not_trigger():
    plan = (
        GOOD_PLAN + "\n" + C13_HEADING + "\n\n" + C13_TABLE_SMALL + "\n```\n" + C13_GATE + "\n```\n"
    )
    assert _status(plan, "c13_empirical_gate_attainability") == "SKIP"


def test_c13_kwarg_ndraws_form_harvested():
    # No table: a FENCED kwarg declaration (`n_draws_isotropic=2`, the #816 v6
    # config-block shape) still harvests — declarations are read from the RAW
    # plan — and floors 1/3 > 0.05 under the unscoped family gate.
    plan = (
        GOOD_PLAN + "\n" + C13_HEADING + "\n\n```\nn_draws_isotropic=2\n```\n\n" + C13_GATE + "\n"
    )
    _, by_id = _run(plan)
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "FAIL"
    assert "n_draws_isotropic n_draws=2" in r.detail


def test_c13_excluded_family_row_not_counted():
    table = (
        "| Family | Construction | n_draws | Source |\n"
        "|---|---|---|---|\n"
        "| Isotropic | v ~ N(0, I) | 200 | norm-matching |\n"
        "| Cross-trait | outside the BH test set — descriptive reference only | 2 | #778 |\n"
        "| PCA top-5 | outside the BH test set — descriptive reference only | 5 | #778 |\n"
    )
    assert _status(_c13_plan(table, C13_GATE), "c13_empirical_gate_attainability") == "PASS"


def test_c13_bare_n_qualifier_does_not_descope():
    # A bare `n ≥ 20` sample-size clause on the gate line must NOT set the
    # draws scope (the pre-fix scope regex silently false-PASSed this — the
    # round-1 statistics-reconcile Must-Fix): the n=2/5 families stay in
    # scope and the gate FAILs.
    gate = (
        "- SUCCESS: across all null families (n ≥ 20 prompts per probe) the read reaches "
        "one-sided empirical p ≤ 0.05 surviving BH."
    )
    assert _status(_c13_plan(C13_TABLE_SMALL, gate), "c13_empirical_gate_attainability") == "FAIL"


def test_c13_quoted_na_phrase_does_not_escape():
    # A mid-sentence quoted escape phrase (the pasted-bounce-brief self-escape
    # channel — the round-1 methodology-reconcile Must-Fix) is NOT a
    # standalone declaration line and must not escape.
    plan = _c13_plan(C13_TABLE_SMALL, C13_GATE) + (
        "\nThe verifier's remediation menu says to declare 'N/A — no empirical-null gate' "
        "when the mention is incidental.\n"
    )
    assert _status(plan, "c13_empirical_gate_attainability") == "FAIL"


def test_c13_strict_lt_at_floor_equal_alpha_is_offender():
    # Under strict `<`, floor == alpha is unattainable (an offender), not the
    # boundary WARN.
    gate = (
        "- SUCCESS: real |r| > every honest null family's 97.5th-pct |r| at "
        "one-sided empirical p < 0.05 surviving Benjamini-Hochberg across the 3 traits."
    )
    assert _status(_c13_plan(C13_TABLE_19, gate), "c13_empirical_gate_attainability") == "FAIL"


def test_c13_alpha_zero_gate_fails_without_crash():
    # Round-2 BLOCKER `alpha-zero-c13-crash`: a registered `p <= 0.00` gate
    # parses to alpha == Fraction(0) — the limiting case of the unattainable
    # class. Every declaration's floor 1/(n+1) > 0 == alpha, so c13 must FAIL
    # (family vocab present), NOT ZeroDivisionError on ceil(1/alpha) in the
    # offender-detail builder (pre-fix: Fraction(1, 0) at the remedy bound).
    gate = (
        "- SUCCESS: real |r| > every honest null family's 97.5th-pct |r| at "
        "one-sided empirical p ≤ 0.00 surviving Benjamini-Hochberg across the 3 traits."
    )
    plan = _c13_plan(C13_TABLE_19, gate)
    # Direct check-function call — the CLI text and --json modes route through
    # verify_plan_text with no exception guard, so no-raise here pins both.
    r = verify_plan.check_empirical_gate_attainability(plan, "experiment")
    assert r.status == "FAIL"
    assert "alpha ≤ 0" in r.detail
    assert "no finite n_draws" in r.detail
    # Full-driver path (what the CLI executes) must not raise either, and the
    # kind=analysis degrade branch builds the SAME detail before degrading —
    # it must WARN, not crash.
    ok, by_id = _run(plan)
    assert by_id["c13_empirical_gate_attainability"].status == "FAIL"
    assert ok is False
    assert _status(plan, "c13_empirical_gate_attainability", kind="analysis") == "WARN"


def test_c13_vacuous_scope_excluding_every_declaration_warns():
    # Round-2 concern `c13-binding-tests-missing` (a): a draws-scope qualifier
    # above every declared count (n_draws >= 500 over a 200/2/5 table) leaves
    # an EMPTY in-scope set — the vacuous-scope WARN, never an affirmative
    # PASS with an undefined min.
    gate = (
        "- SUCCESS: for every STOCHASTIC family (n_draws ≥ 500) one-sided "
        "empirical p ≤ 0.05 surviving Benjamini-Hochberg."
    )
    _, by_id = _run(_c13_plan(C13_TABLE_SMALL, gate))
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "WARN"
    assert "excludes every declared family" in r.detail


def test_c13_fail_detail_names_clean_pass_remedy_bound():
    # Round-2 concern `c13-binding-tests-missing` (b): the alpha=0.05 offender
    # detail carries the ceil(1/alpha) = 20 clean-PASS remedy bound.
    _, by_id = _run(_c13_plan(C13_TABLE_SMALL, C13_GATE))
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "FAIL"
    assert "≥ 20" in r.detail
    assert "clean PASS" in r.detail


# ─── Check 14 — hypothesis confirm/falsify branch coherence ────────────────

# Verbatim bullets from the real corpus (embedded as literals — NEVER read
# from tasks/ at test runtime; tasks move between status folders). The REAL
# plan files are exercised by the §4 pre-commit calibration, not this suite.
C14_HEADING = "## 3. Hypothesis"
# #922 v2 line 47 — the incident offender: implicit confirm (no **Confirm:**
# label), tendency "toward" vs state "stays above" on shared `k = 32`, plus
# the unpinned "mid/late layers" scope.
H922_H4 = "- **H4 (rollout).** The context-only autoregressive roll beats the frozen-state null (ĥ_{T+k} = h_T) for small horizons (k ≤ 4) at mid/late layers, and decays toward/below it by k = 32 (feature drift, 2506.03566). **Falsify (positive surprise):** rollout skill stays above the frozen null through k = 32."
# #922 v2 line 45 — vague-shaped "early to late layers" ESCAPED by the pinned
# "layers 1-28" (en dash in the verbatim string) / "layer 0" in the same block.
H922_H2 = "- **H2 (depth profile).** The carried-context share rises with depth: the ratio r2_id(context-only)/r2_id(token-informed) increases from early to late layers (Spearman ρ(layer, ratio) > 0 over layers 1–28, both spaces). At layer 0 (embedding stream) the context-only map fails by construction (h_{0,t+1} is exactly the injected token embedding) — a built-in sanity anchor. **Falsify:** flat or decreasing profile."
# #841 v12 line 29 — crisp `≤` vs `>` count comparators, no tendency token.
H841_H1 = "- **H1 (Stage-0 fit, matched information).** The source-only GRU's held-out r2_id on Δ_ℓ does NOT beat the affine ridge on the data-limited late-layer band (transitions 17–25) or on a majority of the 27 transitions. **Confirm (ridge stands):** source-only GRU r2_id ≤ ridge r2_id on ≥ 14/27 transitions (raw space), consistent with the prefix-GRU's 27/27 loss. **Falsify (GRU wins):** source-only GRU r2_id > ridge r2_id on a majority of transitions, especially the late-layer band. Directional prior: LOSS or TIE (removing prefix information from an already-losing GRU should not lift it above ridge)."
# #810 v6 line 35 — band/threshold wording (Confirm-the-null vs clears-the-
# band), no tendency token. DUAL-USE fixture: pins c14 PASS (no comparator
# pair / vague scope) AND c17 WARN (the falsify segment's "rewrites the
# parent's read-out takeaway" with no exculpation — the #810 defect family).
H810_H2G = "- **H2-g (read-out).** No summary rescues behavior read-out on generic-genre activations either. **Confirm-the-null:** the two-method conjunction statistic sits inside its max-selected band for each new combo. **Falsify:** a summary clears the band on g1 → the Betley corpus was suppressing read-out (rewrites the parent's read-out takeaway)."


def _hyp_plan(*bullets: str) -> str:
    """GOOD_PLAN + a Hypothesis section carrying ``bullets`` in order."""
    return GOOD_PLAN + "\n" + C14_HEADING + "\n\n" + "\n".join(bullets) + "\n"


@pytest.mark.parametrize("kind", ["infra", "batch", "survey"])
def test_c14_kind_infra_skips(kind):
    assert _status(_hyp_plan(H922_H4), "c14_hypothesis_branch_coherence", kind=kind) == "SKIP"


def test_c14_no_hypothesis_section_skips():
    _, by_id = _run(GOOD_PLAN)
    r = by_id["c14_hypothesis_branch_coherence"]
    assert r.status == "SKIP"
    assert "no hypothesis section" in r.detail


def test_c14_hypothesis_without_branch_anchors_skips():
    plan = _hyp_plan("- **H1.** The read-out improves with depth (prose only, no branch labels).")
    r = _run(plan)[1]["c14_hypothesis_branch_coherence"]
    assert r.status == "SKIP"
    assert "no **Confirm/**Falsify branch anchors" in r.detail


def test_c14_922_h4_toward_vs_stays_above_warns():
    # H4 carries NO explicit **Confirm:** label — the implicit-confirm rule
    # (block text before the falsify anchor is the confirm branch) is
    # REQUIRED for this to fire (plan §4 test 14, folded in here).
    ok, by_id = _run(_hyp_plan(H922_H4))
    r = by_id["c14_hypothesis_branch_coherence"]
    assert r.status == "WARN"
    assert "k = 32" in r.detail
    assert "toward" in r.detail
    assert "stays above" in r.detail
    assert ok is True  # WARN never flips exit 0 (success criterion 6)


def test_c14_922_h4_vague_scope_in_detail():
    r = _run(_hyp_plan(H922_H4))[1]["c14_hypothesis_branch_coherence"]
    assert "mid/late layers" in r.detail
    assert "(b) vague-scope" in r.detail


def test_c14_vague_scope_alone_warns():
    # Predicate (b) fires without (a): no shared bounded token anywhere.
    plan = _hyp_plan(
        "- **H1.** Read-out improves at mid/late layers. **Falsify:** no improvement at any layer."
    )
    r = _run(plan)[1]["c14_hypothesis_branch_coherence"]
    assert r.status == "WARN"
    assert "(b) vague-scope" in r.detail
    assert "mid/late layers" in r.detail
    assert "(a) comparator-pair" not in r.detail


def test_c14_841_v12_crisp_comparators_pass():
    assert _status(_hyp_plan(H841_H1), "c14_hypothesis_branch_coherence") == "PASS"


def test_c14_810_v6_band_wording_passes():
    assert _status(_hyp_plan(H810_H2G), "c14_hypothesis_branch_coherence") == "PASS"


def test_c14_vague_scope_with_pinned_layer_range_passes():
    assert _status(_hyp_plan(H922_H2), "c14_hypothesis_branch_coherence") == "PASS"


def test_c14_toward_without_shared_token_passes():
    # Deliberate shared-token conservatism (accepted false negative): the
    # falsify names `k = 32` but the confirm does not, so no comparator pair
    # is evaluated even though tendency + state tokens are both present.
    plan = _hyp_plan(
        "- **H1.** Rollout skill decays toward the null at long horizons at layer 20. "
        "**Falsify:** rollout skill stays above the frozen null through k = 32."
    )
    assert _status(plan, "c14_hypothesis_branch_coherence") == "PASS"


def test_c14_fenced_hypothesis_does_not_trigger():
    plan = GOOD_PLAN + "\n" + C14_HEADING + "\n\n```\n" + H922_H4 + "\n```\n"
    r = _run(plan)[1]["c14_hypothesis_branch_coherence"]
    assert r.status == "SKIP"
    assert "no **Confirm/**Falsify branch anchors" in r.detail


def test_c14_kind_analysis_warns_not_skips():
    ok, by_id = _run(_hyp_plan(H922_H4), kind="analysis")
    assert by_id["c14_hypothesis_branch_coherence"].status == "WARN"
    assert ok is True  # WARN-only under BOTH in-scope kinds (criterion 6)


def test_c14_comparator_pair_alone_warns():
    # (a)-ALONE positive (round-1 critic Must-Fix, alternatives lens): shared
    # token + tendency-vs-state comparators, NO vague-scope token. A coupled
    # implementation where comparator detection only evaluates when
    # vague-scope also fires would pass the H4 tests (H4 fires both) and the
    # shared-token-conservatism test (nothing evaluates) — this fixture is
    # the only one that fails such a mutant.
    plan = _hyp_plan(
        "- **H1.** Skill decays toward the null by k = 32 at layer 20. "
        "**Falsify:** skill stays above the null through k = 32."
    )
    ok, by_id = _run(plan)
    r = by_id["c14_hypothesis_branch_coherence"]
    assert r.status == "WARN"
    assert "(a) comparator-pair" in r.detail
    assert "k = 32" in r.detail
    assert "toward" in r.detail
    assert "stays above" in r.detail
    assert "(b) vague-scope" not in r.detail
    assert ok is True


# ─── Check 15 — fail-loud acceptance claim backed by a test ────────────────

FAILLOUD_ACCEPT = (
    "\n## 5. Acceptance criteria\n\n"
    "1. The poller re-raises on a malformed sentinel — the crash is not swallowed; "
    "no `except Exception` fallback remains in the drain pass.\n"
)


def test_c15_kind_experiment_skips():
    plan = GOOD_PLAN + FAILLOUD_ACCEPT
    assert _status(plan, "c15_failloud_test_coverage", kind="experiment") == "SKIP"


def test_c15_no_failloud_vocab_skips():
    # GOOD_PLAN's criteria section carries no claim vocabulary (verified
    # against the fixture strings at test-write time) → baseline SKIP.
    assert _status(GOOD_PLAN, "c15_failloud_test_coverage", kind="infra") == "SKIP"


def test_c15_failloud_claim_without_test_warns():
    # The zero-fail-loud-test shape: a fail-loud acceptance claim + a
    # success-path-only test list.
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\nTests: `test_drain_dispatches_ripe_tasks`, `test_drain_respects_concurrency_cap`.\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c15_failloud_test_coverage"]
    assert r.status == "WARN"
    assert "#913" in r.detail
    assert "grep" in r.detail


def test_c15_kind_batch_triggers():
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\nTests: `test_drain_dispatches_ripe_tasks`, `test_drain_respects_concurrency_cap`.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="batch") == "WARN"


def test_c15_grep_gate_does_not_self_certify():
    # The #913-v1 caller-swallow shape: a run-book grep gate over a tests/
    # path carries a test_ identifier + except-vocabulary on one line, but a
    # grep gate verifies the invariant once at review time — it must not
    # count as committed-test evidence.
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\n- Grep gate: `grep -n 'except Exception' scripts/poll_pipeline.py "
        "tests/test_poll_pipeline.py` returns nothing.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "WARN"


def test_c15_named_raise_test_passes():
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\nTests: `test_drain_malformed_sentinel_raises` — malformed sentinel → "
        "`pytest.raises(ValueError)`.\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c15_failloud_test_coverage"]
    assert r.status == "PASS"
    assert "test named" in r.detail


def test_c15_identifier_internal_vocab_passes():
    # Vocabulary inside the identifier — pins the letter-lookaround design
    # (a \b boundary would fail on the underscore-joined token).
    plan = GOOD_PLAN + FAILLOUD_ACCEPT + "\nTests: `test_no_silent_swallow_in_drain`.\n"
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "PASS"


def test_c15_na_escape_incidental_passes():
    # Standalone-line escape (the mid-line form was the self-escape shape).
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\nThe 'silently' sentence above narrates the pre-fix defect.\n"
        + "\nN/A — no fail-loud acceptance claim.\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c15_failloud_test_coverage"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


def test_c15_pasted_warn_detail_does_not_self_satisfy():
    # Anti-paste guard: the WARN detail quotes BOTH escape phrases as remedy
    # options — pre-fix, the doc-global escape search matched them when the
    # detail was pasted back into the plan.
    base = GOOD_PLAN + FAILLOUD_ACCEPT
    _, by_id = _run(base, kind="infra")
    r = by_id["c15_failloud_test_coverage"]
    assert r.status == "WARN"
    pasted = base + f"\n{r.detail}\n"
    assert _status(pasted, "c15_failloud_test_coverage", kind="infra") == "WARN"


def test_c15_na_escape_doc_target_passes():
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\nN/A — fail-loud claim not test-backable (the target is a .md instruction; "
        "no code path a pytest can exercise).\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c15_failloud_test_coverage"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


def test_c15_fenced_claim_vocab_does_not_trigger():
    # A fenced block in an acceptance section is quoted implementation (a
    # diff hunk containing `raise ValueError` is not a claim) — the trigger
    # scan is fence-stripped.
    plan = (
        GOOD_PLAN + "\n## 5. Acceptance criteria\n\nThe diff is verified by the commands below.\n\n"
        "```python\nraise ValueError('the crash is not swallowed')\n```\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "SKIP"


def test_c15_plan_summary_mention_does_not_trigger():
    # §0 Plan Summary restates criteria as summary prose — a measured
    # corpus-probe noise class; the anchor carrier is excluded.
    plan = (
        GOOD_PLAN + "\n## 0. Plan Summary\n\n**Evaluation:** acceptance criteria — the guard "
        "fails loud on a wedged pod.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "SKIP"


def test_c15_design_narration_does_not_trigger():
    # Bug narration in a Design section carries claim vocabulary but no
    # acceptance/success anchor → no trigger.
    plan = (
        GOOD_PLAN + "\n## 4. Design\n\nThe old path silently swallowed the crash; "
        "we delete the bare except.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "SKIP"


def test_c15_grep_gate_plus_named_test_passes():
    # The #913-v2 fixed shape (§3 Q3 registered invariant): a grep-gate line
    # must NOT invalidate independent non-grep test evidence. A
    # grep-invalidates-all-evidence mutation would flip this test while
    # leaving the grep-self-certify and named-raise tests green.
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\n- Grep gate: `grep -n 'except Exception' scripts/poll_pipeline.py` "
        "returns nothing.\n"
        "Tests: `test_drain_malformed_sentinel_raises` — malformed sentinel → "
        "`pytest.raises(ValueError)`.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "PASS"


def test_c15_h1_preamble_anchor_does_not_trigger():
    # Anchor + claim vocabulary in an H1-only preamble (before any ## heading)
    # → the carrier is the H1 (level < 2), a measured probe noise class.
    plan = GOOD_PLAN.replace(
        "## 0.0 TL;DR (plain English)",
        "Acceptance criteria: the poller fails loud — the crash is not swallowed.\n\n"
        "## 0.0 TL;DR (plain English)",
        1,
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "SKIP"


def test_c15_fenced_evidence_line_passes():
    # Test lists legitimately live in fences/tables — the evidence scan is
    # RAW (pins the design against a future fence-stripping refactor).
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\n```\ntests/test_poll_pipeline.py::test_drain_malformed_sentinel_raises PASSED\n```\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "PASS"


# ─── Check 15 — noise-class calibration (#1291) ────────────────────────────


def test_c15_risks_section_anchor_does_not_trigger():
    # #1275 v1 / #1234 shape (#1291): a Risks/Failure-Modes section row
    # cites an acceptance criterion while a sibling row narrates a failure
    # MODE — risk narration is not an acceptance claim, so the anchor never
    # binds and the check SKIPs.
    plan = (
        GOOD_PLAN + "\n## 8. Risks and Failure Modes\n\n"
        "- If the sweep misses a file, acceptance criterion 6 catches it in CI.\n"
        "- post_event raising ValueError on a malformed row fails loud, not silent.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "SKIP"


def test_c15_bare_risks_heading_anchor_does_not_trigger():
    # The #1275 v1 literal heading form (`## 8. Risks`) — same exclusion.
    plan = (
        GOOD_PLAN + "\n## 8. Risks\n\n"
        "- If the sweep misses a file, acceptance criterion 6 catches it in CI.\n"
        "- post_event raising ValueError on a malformed row fails loud, not silent.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "SKIP"


def test_c15_grep_narration_line_does_not_carry_claim():
    # #1275 v2 shape (#1291): the only fail-loud-vocabulary line in the
    # acceptance window narrates grep-tooling semantics ("exits nonzero"),
    # not the plan's own acceptance claim — grep-bearing lines are excluded
    # from the claim scan (line-scoped: a real claim on any non-grep line in
    # the window still triggers, pinned by test_c15_failloud_claim_without_
    # test_warns).
    plan = (
        GOOD_PLAN + "\n## 5. Acceptance criteria\n\n"
        "1. The lint check reports every offending file in one pass.\n"
        "2. Guard the count read (`grep -c` exits nonzero on count 0 — guard the "
        "check with `|| true`).\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "SKIP"


def test_c15_acceptance_heading_containing_risk_word_still_triggers():
    # Precision pin for the START-anchored Risks-heading exclusion (#1291):
    # "risky" mid-heading must NOT exclude a genuine acceptance section
    # (kills an unanchored `risk` mutant). Green on pre-#1291 code by design.
    plan = GOOD_PLAN + FAILLOUD_ACCEPT.replace(
        "## 5. Acceptance criteria", "## 5. Acceptance criteria for the risky path"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "WARN"


def test_c15_acceptance_heading_containing_failure_mode_word_still_triggers():
    # Failure-modes twin of the pin above (#1291 Phase-2 round-1 Must-Fix):
    # the `failure[- ]modes?` alternation is GROUPED under the start anchor,
    # so "failure-mode" mid-heading does not exclude a genuine acceptance
    # section (kills the ungrouped-alternation mutant; green on pre-#1291
    # code and post-fix, red only against the mutant).
    plan = GOOD_PLAN + FAILLOUD_ACCEPT.replace(
        "## 5. Acceptance criteria", "## 5. Acceptance criteria incl. failure-mode tests"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "WARN"


# ─── Check 15 — evidence routes 2-3 calibration (#1306) ────────────────────

# Verbatim corpus literals from the #1296 incident, EMBEDDED as constants —
# committed tests never read real tasks/ plan files (the c13 suite's
# convention). No `test_` identifier appears in the raises lines, so the
# fixture isolates evidence route 2 (the pytest.raises literal).
C15_1296_RAISES_LINES = (  # tasks/completed/1296/plans/v2.md:135-137, verbatim
    "\n   - **negative control:** `with pytest.raises(ModuleNotFoundError):\n"
    '     importlib.import_module("runpod_api")` — proves the scrub is real and\n'
    "     the pre-fix failure mode exists;\n"
)

# The wrapped labeled-pin paragraph: the 105-char test path forced a hard
# wrap, splitting the identifier (line 3) from the label (line 1) and the
# raise vocabulary (lines 1/4) — the same-line route cannot see it, so the
# fixture isolates evidence route 3 (the labeled forward paragraph scan).
C15_1296_PIN_PARAGRAPH = (  # tasks/completed/1296/plans/v2.md:285-289, verbatim
    "\nFail-loud acceptance (check 15): the pytest-backed fail-loud claim is the\n"
    "negative control in\n"
    "`tests/test_backend_poll.py::test_ensure_scripts_dir_bootstrap_resolves_runpod_api_in_module_mode`\n"
    "(asserts `ModuleNotFoundError` is raised when `scripts/` is scrubbed and the\n"
    "bootstrap has not run).\n"
)


def test_c15_pytest_raises_negative_control_passes():
    # Route 2 regression fixture (#1306): the quoted raises control alone —
    # no test_ identifier anywhere in the added lines — satisfies c15.
    plan = GOOD_PLAN + FAILLOUD_ACCEPT + C15_1296_RAISES_LINES
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "PASS"


def test_c15_wrapped_prose_regression_1296_passes():
    # Route 3 regression fixture (#1306, the Durability pin): the labeled
    # satisfier paragraph #1296 v2 carried — label line, identifier two
    # lines below, contiguous — satisfies c15 despite the hard wrap.
    plan = GOOD_PLAN + FAILLOUD_ACCEPT + C15_1296_PIN_PARAGRAPH
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "PASS"


def test_c15_bare_pytest_raises_without_exception_does_not_satisfy():
    # Pins route 2's named-exception requirement (`[\w.]+` after the paren):
    # a bare `pytest.raises()` mention is prose, not a control.
    plan = GOOD_PLAN + FAILLOUD_ACCEPT + "\nThe fix uses pytest.raises() somewhere.\n"
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "WARN"


def test_c15_pytest_raises_on_grep_line_does_not_satisfy():
    # Pins the grep-line exclusion on route 2.
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\n- Gate: grep -rn 'pytest.raises(ValueError' tests/ returns hits.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "WARN"


def test_c15_failloud_pin_label_wrapped_paragraph_passes():
    # Synthetic minimal route-3 shape: label line, identifier two lines
    # below, contiguous paragraph. The identifier carries no fail-loud
    # vocabulary, so the same-line route cannot satisfy — route 3 only.
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\nFail-loud pin (wrapped): the committed negative control lives in\n"
        "the backend-poll suite at\n"
        "`tests/test_backend_poll.py::test_module_mode_import_guard` and runs in CI.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "PASS"


def test_c15_failloud_pin_label_blank_separated_test_does_not_satisfy():
    # Pins the paragraph bound: a blank line ends the route-3 forward scan.
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\nFail-loud pin (wrapped): to be named in a later revision.\n"
        "\n"
        "Tests: test_foo_behavior.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "WARN"


def test_c15_failloud_pin_label_beyond_cap_does_not_satisfy():
    # Pins _FAILLOUD_PIN_SCAN_LINES: an identifier 9 lines below the label
    # (beyond the 8-line cap) does not satisfy route 3.
    filler = "".join(f"filler row {k} of the pin paragraph.\n" for k in range(8))
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\nFail-loud pin (wrapped): the committed negative control lives in\n"
        + filler
        + "`tests/test_backend_poll.py::test_module_mode_import_guard` and runs in CI.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "WARN"


def test_c15_pin_label_grep_line_in_paragraph_does_not_satisfy():
    # Route-3 inner grep exclusion: a labeled paragraph whose ONLY test_
    # identifier sits on a grep line inside the paragraph does not satisfy
    # (pins the inner _FAILLOUD_GREP_LINE_RE continue).
    plan = (
        GOOD_PLAN + FAILLOUD_ACCEPT + "\nFail-loud pin (verify): the invariant is checked by\n"
        "grep -n 'test_scrub_guard' tests/test_backend_poll.py returning a hit.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "WARN"


def test_c15_unrelated_test_two_lines_from_silent_vocab_still_warns():
    # Anti-window pin (the rejected-mechanism regression, distilled from the
    # plan-time pilot's #876/#996 false-flip shape): fail-loud vocabulary
    # two contiguous lines from UNRELATED test identifiers — no label, no
    # pytest.raises — must NOT satisfy (the :4956 class stays out).
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\nThe cache read is silent on a miss by design and the loader\n"
        "continues; see the loader table.\n"
        "Suite: test_loader_roundtrip, test_loader_cache_hit.\n"
    )
    assert _status(plan, "c15_failloud_test_coverage", kind="infra") == "WARN"


def test_c15_true_positive_still_warns():
    # Post-change restatement of the #913 true positive: a fail-loud claim
    # with only success-path tests named still WARNs, and the updated detail
    # keeps the pinned substrings plus the new remedy wording.
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\nTests: `test_drain_dispatches_ripe_tasks`, `test_drain_respects_concurrency_cap`.\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c15_failloud_test_coverage"]
    assert r.status == "WARN"
    assert "#913" in r.detail
    assert "grep" in r.detail
    assert "ONE unwrapped line" in r.detail


# ─── Check 16 — re-extracted reference vs committed headline ───────────────

# Near-verbatim #811 v3 shapes (task #937 incident): the synthetic fixtures
# ARE the incident text — committed tests never read real tasks/ plan files
# (the c13 suite's convention).
FOLLOWUP_HEADER = (
    "\n## 12. Round context\n\n"
    "This is a same-issue follow-up round (AMENDMENT to the completed maxp round), "
    "authorized by the `epm:followup-scope` marker; the result folds into THIS\n"
    "issue's clean-result body.\n"
)
REEXTRACT_BLOCK = (
    "\n### Reference arms\n\n"
    "| **Content average (reference, re-extracted)** | v1 mean through this round's "
    "reader | parity vs committed v1 cells |\n\n"
    "Parity: re-extracted mean + turn_nl vs v1's committed cells; a flipped CALL on "
    "any reference is REPORTED as a replication-stability finding (adjudication then "
    "uses THIS round's internally consistent store).\n"
)
DISTINCTION_SENTENCE = (
    "\nHeadline adjudication: the committed v1/v2 headline cells remain the "
    "adjudicated evidence for the standing clean-result; this round's re-extracted "
    "references serve only as same-pass comparators for the NEW arms.\n"
)
C16_INCIDENT = GOOD_PLAN + FOLLOWUP_HEADER + REEXTRACT_BLOCK
C16 = "c16_reference_headline_distinction"


def test_c16_not_triggered_skips():
    assert _status(GOOD_PLAN, C16) == "SKIP"


@pytest.mark.parametrize("kind", ["infra", "batch", "survey"])
def test_c16_kind_infra_skips(kind):
    assert _status(C16_INCIDENT, C16, kind=kind) == "SKIP"


def test_c16_reextraction_without_followup_skips():
    # Half (a) fires; half (b) is absent (REEXTRACT_BLOCK carries no
    # same-issue-follow-up / fold vocabulary) — the SKIP detail names the
    # missing fold half.
    _, by_id = _run(GOOD_PLAN + REEXTRACT_BLOCK)
    r = by_id[C16]
    assert r.status == "SKIP"
    assert "same-issue follow-up" in r.detail


def test_c16_followup_without_reextraction_skips():
    _, by_id = _run(GOOD_PLAN + FOLLOWUP_HEADER)
    r = by_id[C16]
    assert r.status == "SKIP"
    assert "no re-extraction" in r.detail


def test_c16_811_v3_clause_shape_warns():
    # Acceptance criterion 3: the near-verbatim #811 v3 §6 clause + §5
    # conditions-table row shape WARNs — replication-stability vocabulary
    # alone is exactly what the incident plan carried; it must not satisfy.
    _, by_id = _run(C16_INCIDENT)
    r = by_id[C16]
    assert r.status == "WARN"
    assert "same-pass-comparator" in r.detail


def test_c16_pasted_warn_detail_does_not_self_satisfy():
    # Anti-paste guard (#4.2 c16 de-fang): pre-fix, the WARN detail's own
    # "same-pass comparator" wording satisfied _C16_SAMEPASS_RE (and its
    # committed-headline prose satisfied _C16_DISTINCTION_RE) when pasted
    # back into the plan; the reworded detail must not.
    _, by_id = _run(C16_INCIDENT)
    r = by_id[C16]
    assert r.status == "WARN"
    pasted = C16_INCIDENT + f"\n{r.detail}\n"
    assert _status(pasted, C16) == "WARN"


def test_c16_distinguishing_sentence_passes():
    assert _status(C16_INCIDENT + DISTINCTION_SENTENCE, C16) == "PASS"


def test_c16_same_pass_comparator_phrase_passes():
    plan = C16_INCIDENT + "\nThe re-extracted values are same-pass comparators.\n"
    assert _status(plan, C16) == "PASS"


def test_c16_negated_replacement_passes():
    # S3-only satisfier: pin the negated-replacement branch as LIVE — the
    # other two satisfier regexes must NOT match this fixture, so the PASS
    # can only come from _C16_NONREPLACE_RE.
    plan = (
        C16_INCIDENT
        + "\nA flipped reference CALL never silently replaces the committed headline.\n"
    )
    text = verify_plan.strip_fences(plan)
    assert verify_plan._C16_NONREPLACE_RE.search(text)
    assert not verify_plan._C16_SAMEPASS_RE.search(text)
    assert not verify_plan._C16_DISTINCTION_RE.search(text)
    assert _status(plan, C16) == "PASS"


def test_c16_bare_same_pass_does_not_satisfy():
    # #811 v3:189 shape — "SAME pass" without "comparator" is not the term of
    # art and must not satisfy S1.
    plan = (
        C16_INCIDENT + "\nWe re-extract the references in the SAME pass as shared-R extraction.\n"
    )
    assert _status(plan, C16) == "WARN"


def test_c16_unnegated_replaces_does_not_satisfy():
    # #811 v3:270 shape — un-negated "replaces" (figure-layout prose) must not
    # satisfy the negated-replacement shape S3.
    plan = C16_INCIDENT + "\nThe heatmap layout replaces grouped bars.\n"
    assert _status(plan, C16) == "WARN"


def test_c16_artifacts_untouched_does_not_satisfy():
    # #811 v3:499 shape — a files-not-adjudication parenthetical: "committed"
    # is stopped by the ';' and "artifacts" is not a committed-headline noun.
    plan = C16_INCIDENT + "\n(committed; prior rounds' artifacts untouched)\n"
    assert _status(plan, C16) == "WARN"


def test_c16_file_retention_sentence_satisfies_s2_accepted():
    # DELIBERATE accept-and-document decision (task #937): a files-on-disk
    # retention sentence DOES satisfy S2 ("committed ... cells ... remain").
    # Excluding it would require dropping cells/values from the
    # committed-headline noun set, which breaks the legitimate
    # "the committed cells remain the adjudicated evidence" shape (there the
    # retention verb precedes "evidence", so the matchable noun must be
    # "cells"). The check is WARN-only and surfaces, never adjudicates — the
    # Statistics critic owns the files-vs-adjudication semantic call. Pinned
    # so a future regex tightening flips this test CONSCIOUSLY, not silently.
    plan = C16_INCIDENT + "\nThe committed v1 cells remain untouched on disk in eval_results/.\n"
    assert _status(plan, C16) == "PASS"


def test_c16_warn_detail_roundtrip():
    # The WARN detail must teach a sentence that actually satisfies: this is
    # the most natural sentence a planner writes after reading it.
    plan = (
        C16_INCIDENT + "\nThe committed headline values remain the adjudicated evidence; this "
        "round's re-extracted references are same-pass comparators only.\n"
    )
    assert _status(plan, C16) == "PASS"


def test_c16_na_escape_passes():
    plan = C16_INCIDENT + "\nN/A — no re-extracted reference arms.\n"
    _, by_id = _run(plan)
    r = by_id[C16]
    assert r.status == "PASS"
    assert "N/A" in r.detail


def test_c16_kind_analysis_warns():
    # Scope parity with c12/c14: kind=analysis triggers too; the WARN never
    # blocks (overall stays ok — c16 is WARN-only for BOTH kinds).
    ok, by_id = _run(C16_INCIDENT, kind="analysis")
    assert by_id[C16].status == "WARN"
    assert ok is True


def test_c16_negated_reextraction_mention_does_not_trigger():
    # Calibration refinement (2026-07-03 sweep): a plan ASSERTING it does NOT
    # re-extract ("NO re-extraction of r_B" — the #778/#559/#561/#810-v1-3
    # noise class) is not a trigger; the negation lookbehind drops it.
    plan = (
        GOOD_PLAN
        + FOLLOWUP_HEADER
        + "\nThis amendment performs no re-extraction of the reference arms; the "
        "committed v1 cells are reused as-is, and the direction is not re-extracted.\n"
    )
    _, by_id = _run(plan)
    r = by_id[C16]
    assert r.status == "SKIP"
    assert "no re-extraction" in r.detail


def test_c16_regeneration_window_only_does_not_trigger():
    # Calibration refinement (§4.5 pre-authorized demotion, exercised on the
    # 2026-07-03 sweep): `re-generat` requires reference vocab on the SAME
    # line — an adjacent-line co-occurrence (doc/data-regeneration noise:
    # #491/#537/#542/#558/#597/#685/#763/#825 class) must not trigger.
    plan = (
        GOOD_PLAN
        + FOLLOWUP_HEADER
        + "\nThe monitoring corpus is regenerated from the frozen seed list.\n"
        "Downstream, agreement vs committed v1 cells is asserted separately.\n"
    )
    assert _status(plan, C16) == "SKIP"


def test_c16_regeneration_same_line_triggers():
    # The regen branch still fires on genuine same-line adjacency.
    plan = (
        GOOD_PLAN
        + FOLLOWUP_HEADER
        + "\nThe reference cells are regenerated through this round's reader.\n"
    )
    assert _status(plan, C16) == "WARN"


def test_c16_fenced_trigger_does_not_fire():
    # Re-extraction vocabulary ONLY inside a code fence is not a trigger —
    # pins the fence-masked half-(a) path (mirror of
    # test_c12_fenced_battery_does_not_trigger).
    plan = (
        GOOD_PLAN
        + FOLLOWUP_HEADER
        + "\n## 13. Example\n\n```\nre-extract the reference arms; parity vs committed v1 cells\n```\n"
    )
    assert _status(plan, C16) == "SKIP"


# ─── Check 17 — falsification-branch causal-claim scope ────────────────────

# #810 plan v13 §0.0 — the incident offender (three reviewers required the
# scope-down); the v14 line is the accepted fix (exculpation tokens:
# "consistent with", "not uniquely diagnostic", OOD, "remains live").
C17_V13_MIND = (
    "- **What would change my mind:** If deleting the answer makes them clearly "
    "harder to rebuild — dropping below the plain answer-average benchmark — with "
    "a clean paired gap, then they really were carrying answer content, the echo "
    "story dies, and the parked follow-up #943 gets its revival condition."
)
C17_V14_MIND = (
    "- **What would change my mind:** If deleting the answer makes them clearly "
    "harder to rebuild — dropping below the plain answer-average benchmark — with "
    "a clean paired gap, then the header read is ANSWER-PRESENCE-DEPENDENT — "
    "consistent with integration but not uniquely diagnostic (an off-distribution "
    "empty turn degrading the states remains live); the truncation dose-response "
    "follow-up disambiguates, and parked #943's revival condition becomes "
    "conditional on it."
)
# H810_H2G (the c14 PASS fixture, #810 v6) carries "rewrites the parent's
# read-out takeaway" in its falsify segment with no exculpation — c17
# deliberately WARNs on it (the diff-sketch's own example pattern; same
# defect family that recurred on #810 through v13).
H810_H2G_EXCULPATED = (
    H810_H2G + " Not uniquely diagnostic on its own: a genre confound remains live."
)
# GOOD_PLAN minus its (clean) mind bullet — the no-surface SKIP fixture.
GOOD_PLAN_MIND_LINE = (
    "- **What would change my mind:** A shift larger than the run-to-run spread "
    "would mean benign data alone moves persona expression."
)


@pytest.mark.parametrize("kind", ["infra", "batch", "survey"])
def test_c17_kind_exempt_skips(kind):
    plan = GOOD_PLAN + "\n" + C17_V13_MIND + "\n"
    assert _status(plan, "c17_causal_branch_scope", kind=kind) == "SKIP"


def test_c17_no_branch_surface_skips():
    plan = GOOD_PLAN.replace(
        GOOD_PLAN_MIND_LINE, "- **Spread note:** run-to-run spread is the yardstick."
    )
    r = _run(plan)[1]["c17_causal_branch_scope"]
    assert r.status == "SKIP"
    assert "no registered falsification-branch surface" in r.detail


def test_c17_810_v13_mind_bullet_warns():
    ok, by_id = _run(GOOD_PLAN + "\n" + C17_V13_MIND + "\n")
    r = by_id["c17_causal_branch_scope"]
    assert r.status == "WARN"
    assert "really w" in r.detail  # 'really were'
    assert "What would change my mind" in r.detail
    assert ok is True  # WARN never flips exit 0


def test_c17_810_v14_scoped_down_passes():
    assert _status(GOOD_PLAN + "\n" + C17_V14_MIND + "\n", "c17_causal_branch_scope") == "PASS"


def test_c17_hyp_falsify_rewrites_takeaway_warns():
    r = _run(_hyp_plan(H810_H2G))[1]["c17_causal_branch_scope"]
    assert r.status == "WARN"
    assert "rewrites the parent" in r.detail
    assert "falsify" in r.detail


def test_c17_hyp_confirm_segment_offender_warns():
    # Round-scope item R2: the confirm-branch scan path — an offender token
    # inside an explicit **Confirm:** segment, no exculpation anywhere in
    # the block → WARN naming the confirm segment.
    plan = _hyp_plan(
        "- **H1 (mechanism).** The probe reads answer information from the header states. "
        "**Confirm:** a clean paired gap establishes that the states were integrated. "
        "**Falsify:** no paired gap appears at any layer."
    )
    r = _run(plan)[1]["c17_causal_branch_scope"]
    assert r.status == "WARN"
    assert "confirm" in r.detail
    assert "establishes that" in r.detail


def test_c17_hyp_exculpated_passes():
    assert _status(_hyp_plan(H810_H2G_EXCULPATED), "c17_causal_branch_scope") == "PASS"


def test_c17_must_have_been_offender_warns():
    # Round-scope item R3 (documentation fixture): the retrospective
    # "must have been" attribution is tier-1 offender vocabulary.
    plan = (
        GOOD_PLAN
        + "\n"
        + (
            "- **What would change my mind:** If the gap survives the swap, the "
            "adapter must have been encoding the trait all along."
        )
        + "\n"
    )
    r = _run(plan)[1]["c17_causal_branch_scope"]
    assert r.status == "WARN"
    assert "must have been" in r.detail


def test_c17_incidental_exculpation_silences_offender():
    # Round-scope item R3 (documentation fixture): DOCUMENTED ACCEPTED
    # FALSE NEGATIVE — exculpation scope is the whole bullet, so an
    # INCIDENTAL hedge token ("caveat", here about judge sample size, not
    # about the causal claim) silences a genuine offender ("story dies").
    # Deliberate per the c14 prefer-false-negatives charter; do not
    # "fix" by narrowing exculpation scope to the claim sentence (the v14
    # fix wording itself puts the hedge in a trailing parenthetical).
    plan = (
        GOOD_PLAN
        + "\n"
        + (
            "- **What would change my mind:** If the paired gap is clean, the echo "
            "story dies (one caveat: the judge sample is small)."
        )
        + "\n"
    )
    assert _status(plan, "c17_causal_branch_scope") == "PASS"


def test_c17_kind_analysis_warns_not_skips():
    plan = GOOD_PLAN + "\n" + C17_V13_MIND + "\n"
    assert _status(plan, "c17_causal_branch_scope", kind="analysis") == "WARN"


def test_c17_fenced_offender_does_not_trigger():
    plan = GOOD_PLAN + "\n```\n" + C17_V13_MIND + "\n```\n"
    # GOOD_PLAN's own clean mind bullet is still a surface → PASS, not WARN.
    assert _status(plan, "c17_causal_branch_scope") == "PASS"


# ─── Check 18 — paired-contrast per-arm source coverage ────────────────────

# Verbatim corpus literals (embedded, never read from tasks/ paths — plans
# move between status folders; this file's own convention). Sources: #810
# plans/v13.md:33 (the founding FAIL registration — 9-row paired bootstrap
# whose named full-side pack lacked im_end/turn_nl), #810 plans/v15.md:71
# (the reviewer-converged PASS declaration — the D2 subset-assert shape),
# task #608 plans/v2.md:164 (the replay-found incidental subset-CI
# false-satisfier: "pair" substring-matches inside "paired"; word-bounding
# must reject it). The REAL files are exercised by the §3.4 implementation
# calibration replay, not by this suite.
C18_V13_REGISTRATION = "- **H1-he (echo — the headline).** Empty-answer header/boundary rows reconstruct ≈ as well as their full-answer twins. **Registered per-row statistic:** paired per-context LOCO Δskill(full − empty) at each row's committed best layer, via the shared-index 2,000-draw bootstrap (7 pairs vs round-3 rows; `im_end`/`turn_nl` pair vs round-1). **Echo supported:** every header pool's Δ CI includes 0 or |Δ| < 0.02 (the round-3 D_uh margin), AND best empty-row LOFO ≥ 0.805. **Falsified (integration):** best header/boundary empty-row LOFO < 0.805 AND that row's paired Δ CI excludes 0 from above (\"clearly-positive\"). Expected: echo."
C18_V15_DECLARATION = "4. **NEW `scripts/issue810_he_mechanism.py`** (thin driver, the only new analysis file — the `issue810_uh_crosslayer.py` precedent): loads `uh_summaries.pt` (full side for the 7 uh/bnd rows) + `he_summaries.pt` (empty) + — MUST-FIX addition — the round-3 HF store `issue658_theory_assumptions/answer_position_sweep_user_header/<ctx>.pt` (per-file `hf_hub_download`, ~430 MB total, cpu-mid-safe) as the FULL side for the `im_end`/`turn_nl` pairs (NOT in the uh pack — verified by three reviewers); their refit-parity assert targets the ROUND-1 committed skills in `eval_results/issue_810/reconstruction_skill_by_summary.json` (valid: both positions are causally upstream of the extension), ≤1e-6. ROW-COVERAGE SMOKE ASSERT (mechanizable): `set(registered_pair_rows) ⊆ union(keys(full_side_sources))` for every source named here, asserted at driver start; (a) cross-context-centers each (row, layer) across the 50 contexts and emits per-context cosine + pooled R² (1 − ‖full−empty‖²/‖full‖², centered); (b) refits the full-side per-context LOCO predictions from `uh_summaries.pt` (asserting skills match committed ≤1e-6 — the `a26da411bb` precedent) and runs the paired shared-index 2,000-draw bootstrap on Δskill(full−empty) per row at the committed best layers. All numpy/torch-CPU, batched."
C18_608_CI_LINE = "Per-cell own-rate SE ≈ 0.01–0.02 binomial, ≈ 0.03–0.05 claim-clustered (effective N = 50 claims) — the +0.05 support threshold and ±0.02 equivalence band sit respectively above and below that noise floor. Formal tests — REGISTERED CONVENTIONS (v2): per-source claim-level paired bootstrap on g(s) (per-claim 10-rollout rates → paired claim differences → resample 50 claims with replacement → 10,000 draws → two-sided 95% percentile CI; base rates not resampled — base cancels in g); H1 falsification uses the CI-CONTAINMENT reading (CI ⊆ [−0.05, +0.05] for ≥5/6 sources, per §1); boundary-indeterminate outcomes are reported as indeterminate with continuous estimates, never as evidence against H1. 6-source sign test reported one-sided AND two-sided (two-sided: 6/6 p=0.031, 5/6 p=0.219; one-sided: 6/6 p=0.016, 5/6 p=0.109) as a descriptive pattern — the per-source CIs carry the inference. κ-calibration per §4 Phase G; the May-vs-June drift comparison is descriptive only (the single unified judge pass removes it from the inferential path)."


def _c18_plan(*parts: str) -> str:
    """GOOD_PLAN + a Hypothesis section carrying ``parts`` in order
    (GOOD_PLAN contains zero `paired` mentions — pinned in the
    not-triggered test below, so the base plan can never trigger c18)."""
    return GOOD_PLAN + "\n## Hypothesis\n\n" + "\n\n".join(parts) + "\n"


def test_c18_not_triggered_skips():
    assert "paired" not in GOOD_PLAN.lower()  # fixture precondition
    _, by_id = _run(GOOD_PLAN)
    r = by_id["c18_paired_contrast_source_coverage"]
    assert r.status == "SKIP"
    assert "no registered paired contrast" in r.detail


@pytest.mark.parametrize("kind", ["infra", "batch", "survey"])
def test_c18_kind_exempt_skips(kind):
    plan = _c18_plan(C18_V13_REGISTRATION)
    assert _status(plan, "c18_paired_contrast_source_coverage", kind=kind) == "SKIP"


def test_c18_810_v13_shape_fails():
    ok, by_id = _run(_c18_plan(C18_V13_REGISTRATION))
    r = by_id["c18_paired_contrast_source_coverage"]
    assert r.status == "FAIL"
    assert "Row-coverage" in r.detail  # copy-adaptable remedy option
    assert "N/A — no paired contrast" in r.detail  # escape phrase quoted
    assert "#810 v13 class" in r.detail  # paste fingerprint (guard a)
    assert C18_V13_REGISTRATION[:90] in r.detail  # quoted trigger prefix
    assert ok is False


def test_c18_810_v15_subset_assert_passes():
    # D2: subset expression + word-bounded row/pair vocab + coverage vocab
    # on the real v15:71 declaration line.
    _, by_id = _run(_c18_plan(C18_V13_REGISTRATION, C18_V15_DECLARATION))
    r = by_id["c18_paired_contrast_source_coverage"]
    assert r.status == "PASS"
    assert "row-coverage declaration found" in r.detail
    assert "fact-checker" in r.detail  # a PASS is never "coverage verified"
    # Guard (b) must not trip on the real literal (no cross-issue #NN token).
    assert verify_plan._c18_candidate_ok(C18_V15_DECLARATION)


def test_c18_incidental_subset_ci_line_does_not_satisfy():
    # MF-A anchor: the #608 v2:164 CI-containment convention carries a subset
    # expression + coverage vocab ("sources") but NO word-bounded row/pair
    # token — "paired" must not substring-satisfy the pairs? vocabulary.
    assert verify_plan._C18_SUBSET_RE.search(C18_608_CI_LINE)
    assert not verify_plan._C18_ROWPAIR_RE.search(C18_608_CI_LINE)
    plan = _c18_plan(C18_V13_REGISTRATION, C18_608_CI_LINE)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_quoted_sibling_assert_does_not_satisfy():
    # Guard (b): quoting another issue's driver assert as a worked example is
    # a citation, not a declaration of THIS plan's inputs.
    quoted = (
        "As in #810 v15: `set(registered_pair_rows) ⊆ union(keys(full_side_sources))` "
        "asserted for all 9 rows at driver start."
    )
    plan = _c18_plan(C18_V13_REGISTRATION, quoted)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_deferral_row_coverage_line_does_not_satisfy():
    # MF-B: the v1 bare `this run` by-construction alternative was removed —
    # a deferral anti-declaration carries zero source evidence.
    deferral = "Row-coverage: deferred to a later revision of this run's analysis."
    plan = _c18_plan(C18_V13_REGISTRATION, deferral)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_one_arm_declaration_passes_by_design():
    # Accepted residual (e), pinned as the DOCUMENTED disposition (not a
    # bug): the #810 v15 exemplar declaration is itself full-side-only, so a
    # structural both-arm requirement would reject the incident's own
    # reviewer-converged fix. One-arm truthfulness stays with the
    # fact-checker (the check's scope-discipline paragraph).
    one_arm = "Row-coverage: full arm = uh_summaries.pt (all 9 rows)."
    plan = _c18_plan(C18_V13_REGISTRATION, one_arm)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "PASS"


def test_c18_row_coverage_line_with_artifacts_passes():
    decl = (
        "Row-coverage: full arm = uh_summaries.pt (7 uh/bnd rows) + round-3 store "
        "(im_end, turn_nl); empty arm = he_summaries.pt (all 9 rows)."
    )
    _, by_id = _run(_c18_plan(C18_V13_REGISTRATION, decl))
    r = by_id["c18_paired_contrast_source_coverage"]
    assert r.status == "PASS"
    assert "declaration surface" in r.detail


def test_c18_row_coverage_forward_window_passes():
    # The multi-line header + per-arm bullet shape: evidence within the
    # 3-physical-line forward window.
    decl = (
        "Row-coverage (per arm):\n"
        "- full arm: uh_summaries.pt (7 uh/bnd rows) + the round-3 store (im_end, turn_nl)\n"
        "- empty arm: he_summaries.pt (all 9 rows)"
    )
    plan = _c18_plan(C18_V13_REGISTRATION, decl)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "PASS"


def test_c18_by_construction_declaration_passes():
    decl = "Row-coverage: both arms computed by this plan's own fits over the shared probe grid."
    plan = _c18_plan(C18_V13_REGISTRATION, decl)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "PASS"


@pytest.mark.parametrize("kind", ["experiment", "analysis"])
def test_c18_na_escape_with_detected_contrast_warns(kind):
    # #1258 (the #1223 c20 port): the escape is only reachable when a paired
    # contrast WAS detected (the no-trigger case SKIPs first), so N/A +
    # detected registration is the masking shape — WARN (non-blocking), never
    # a silent PASS. Same severity both kinds.
    plan = _c18_plan(C18_V13_REGISTRATION) + (
        "\nN/A — no paired contrast (the paired statistic above recaps the sibling's "
        "registration).\n"
    )
    ok, by_id = _run(plan, kind=kind)
    r = by_id["c18_paired_contrast_source_coverage"]
    assert r.status == "WARN"
    assert ok is True  # WARN never blocks
    assert "co-occurs" in r.detail
    assert C18_V13_REGISTRATION[:60] in r.detail  # detected trigger snippet quoted


def test_c18_na_escape_without_contrast_still_skips():
    # Preserved: with no detected paired contrast the SKIP gate fires before
    # the escape is consulted — the N/A line stays legal and never penalized.
    # (Fixture precondition `"paired" not in GOOD_PLAN.lower()` is pinned by
    # test_c18_not_triggered_skips above.)
    plan = GOOD_PLAN + "\nN/A — no paired contrast\n"
    _, by_id = _run(plan)
    r = by_id["c18_paired_contrast_source_coverage"]
    assert r.status == "SKIP"
    assert "no registered paired contrast" in r.detail


def test_c18_quoted_na_phrase_does_not_escape():
    plan = _c18_plan(
        C18_V13_REGISTRATION,
        "The remedy menu offers \"declare 'N/A — no paired contrast' on its own line\" "
        "as one option.",
    )
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_kind_analysis_warns_not_fails():
    ok, by_id = _run(_c18_plan(C18_V13_REGISTRATION), kind="analysis")
    r = by_id["c18_paired_contrast_source_coverage"]
    assert r.status == "WARN"
    assert "kind-degrade" in r.detail
    assert ok is True  # WARN never flips exit 0


def test_c18_fenced_registration_does_not_trigger():
    plan = GOOD_PLAN + "\n## Hypothesis\n\n```\n" + C18_V13_REGISTRATION + "\n```\n"
    assert _status(plan, "c18_paired_contrast_source_coverage") == "SKIP"


def test_c18_registration_outside_registration_section_skips():
    plan = GOOD_PLAN + "\n## Prior Work\n\n" + C18_V13_REGISTRATION + "\n"
    assert _status(plan, "c18_paired_contrast_source_coverage") == "SKIP"


def test_c18_h1_title_match_does_not_scope():
    # H2+ restriction: a doc-spanning H1 *title* matching the section family
    # (here the `nulls?` member, via "turn-null") must not scope the whole
    # doc into the registration family.
    plan = (
        "# Plan — real-user-turn-null sweep\n\n"
        + "filler word " * 200
        + "\n\n## 1. Goal\n\nx\n\n## 2. Design\n\ny\n\n## Prior Work\n\n"
        + C18_V13_REGISTRATION
        + "\n"
    )
    assert _status(plan, "c18_paired_contrast_source_coverage") == "SKIP"


def test_c18_paircount_form_triggers():
    # The pair-count trigger route in ISOLATION: `paired` + an enumerated
    # pair count, no `regist` substring anywhere on the line (precondition
    # asserted — a deleted _C18_PAIRCOUNT_RE fails this test, never
    # vacuously passes it).
    line = (
        "- The headline statistic is a paired bootstrap CI over the 9 pairs, "
        "all pre-named; report all 9 jointly."
    )
    assert not verify_plan._C18_REGIST_RE.search(line)
    assert verify_plan._C18_PAIRCOUNT_RE.search(line)
    assert _status(_c18_plan(line), "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_pasted_fail_detail_does_not_self_satisfy():
    # The project convention pastes bounce text verbatim into revised plans:
    # the FAIL detail carries the fingerprint (guard a) and a cross-issue
    # token (guard b), and its wording supplies no D1 evidence / D2 subset
    # expression, so a pasted detail can never satisfy the check.
    _, by_id = _run(_c18_plan(C18_V13_REGISTRATION))
    detail = by_id["c18_paired_contrast_source_coverage"].detail
    assert verify_plan._C18_PASTE_FINGERPRINT in detail
    replan = _c18_plan(C18_V13_REGISTRATION, detail)
    assert _status(replan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_bare_coverage_vocab_without_evidence_does_not_satisfy():
    # v13:218's reporting-completeness shape: "covers" prose without the
    # row-coverage token or a subset expression is not a declaration.
    line = "- Every registered read emits its verdict; the paired table covers all 9 rows."
    plan = _c18_plan(C18_V13_REGISTRATION, line)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_generic_covers_prose_not_a_declaration():
    # No D1 match without the literal row-coverage token.
    line = "Both packs cover every registered row and every source is verified upstream."
    plan = _c18_plan(C18_V13_REGISTRATION, line)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_figures_enumeration_line_does_not_trigger():
    # §3.4 calibration tuning: a figures-enumeration line mentions paired
    # panels + "registered rows visually distinguished" without registering
    # any statistic (the #537 v4-v6 / #931 v1-v4 spurious-trigger class) —
    # the leading figures label excludes it from the trigger scan.
    line = (
        "**Figures (over-produce; analyzer picks heroes):** hero forest plot "
        "(registered rows visually distinguished); EM contrastive vs "
        "non-contrastive paired cells; per-row diagonal implant-strength bars."
    )
    assert _status(_c18_plan(line), "c18_paired_contrast_source_coverage") == "SKIP"


# #1086: the #833 v8 L80 Row-coverage line (verbatim corpus literal — same
# embedded-literal convention as the #810 fixtures above). Pre-#1086 the D1
# evidence regexes rejected all three of its artifact tokens
# (`analysis_tensors_nonemit/` — suffixed store; `issue833_onpolicy_map/…`
# — extension-free HF data-repo prefix; `@fa0f8ea3` — a sha pin) AND its
# by-construction phrasing, which follows the check's own remedy text.
C18_V8_ROW_COVERAGE = "Row-coverage: the nonemit arm's 291 retained-cell rows come from this round's own Phase-N1 extraction (`analysis_tensors_nonemit/`, both legs, all 3 layers); the full-text comparator arms' rows for the SAME 291 cells come from the r7e joined design rebuilt from HF `issue833_onpolicy_map/analysis_tensors` + `analysis_tensors_rbase` @fa0f8ea3 (all 480 cells present, a superset of the retained set); the plan's own fits produce every registered chain row on each arm."


def test_c18_v8_row_coverage_line_passes():
    # The motivating #1086 false-positive: v8 L80 verbatim must satisfy D1.
    assert verify_plan._c18_candidate_ok(C18_V8_ROW_COVERAGE)  # no #NN / fingerprint
    plan = _c18_plan(C18_V13_REGISTRATION, C18_V8_ROW_COVERAGE)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "PASS"


def test_c18_own_fits_produce_every_registered_row_passes():
    # The check's own remedy prescription, implemented verbatim, must PASS
    # (the pre-#1086 remedy-vs-satisfier inconsistency: the FAIL detail
    # instructed exactly this sentence and the old regexes rejected it).
    decl = "Row-coverage: the plan's own fits produce every registered row on each arm."
    plan = _c18_plan(C18_V13_REGISTRATION, decl)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "PASS"


def test_c18_suffixed_store_and_issue_prefix_are_artifact_evidence():
    # Direct _C18_ARTIFACT_RE asserts for the #1086 additions.
    assert verify_plan._C18_ARTIFACT_RE.search("`analysis_tensors_nonemit/`")
    assert verify_plan._C18_ARTIFACT_RE.search("`issue833_onpolicy_map/analysis_tensors`")
    assert not verify_plan._C18_ARTIFACT_RE.search("a later revision of this run's analysis.")


def test_c18_vague_row_coverage_without_evidence_still_fails():
    # A Row-coverage line with NO artifact token, NO affirmative
    # produce-verb + "every registered", and no D2 subset expression stays
    # a FAIL.
    decl = "Row-coverage: the named sources cover all rows on both arms."
    plan = _c18_plan(C18_V13_REGISTRATION, decl)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_negated_produce_declaration_fails():
    # v2 negation guard: an explicit NON-declaration must keep FAILing (the
    # MF-B deferral class the v1 bare widening would have re-opened).
    decl = "Row-coverage: the plan does not yet produce every registered row on each arm."
    plan = _c18_plan(C18_V13_REGISTRATION, decl)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_deferred_produce_declaration_fails():
    # v2 deferral guard: a future-tense deferral is not a declaration.
    decl = "Row-coverage: the plan will produce every registered row on each arm once implemented."
    plan = _c18_plan(C18_V13_REGISTRATION, decl)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_produce_verb_without_every_registered_fails():
    # The drop-"every registered" mutant: a bare produce verb is not enough.
    decl = "Row-coverage: the plan produces rows on both arms."
    plan = _c18_plan(C18_V13_REGISTRATION, decl)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_produces_every_registered_outside_row_coverage_window_fails():
    # D1 evidence is consulted only inside a Row-coverage window: the
    # affirmative phrase in arbitrary prose (no Row-coverage vocab anywhere,
    # no D2 subset line) does not satisfy the check.
    prose = "The pipeline produces every registered row on each arm as part of phase N2."
    plan = _c18_plan(C18_V13_REGISTRATION, prose)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_armless_produces_every_registered_fails():
    # The arm-vocabulary lookahead: no each/both/per-arm vocab, no hit.
    decl = "Row-coverage: the pipeline produces every registered row."
    plan = _c18_plan(C18_V13_REGISTRATION, decl)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


def test_c18_neg_defer_new_tokens_direct():
    # #1099 unit contract: each added negation/deferral form matches; the
    # pre-fix `n't` alternative was dead code (\b cannot match inside
    # "doesn't" at the word-internal s->n transition).
    for tok in (
        "doesn't",
        "doesn’t",
        "can't",
        "won't",
        "isn't",
        "wouldn't",
        "n't",
        "cannot",
        "fails to",
        "fail to",
        "failed to",
        "until",
    ):
        assert verify_plan._C18_NEG_DEFER_RE.search(tok), tok
    # Benign affirmative-span text must NOT fire (the two pinned PASS
    # affirmatives' spans, verbatim-class).
    for benign in (
        "Row-coverage: the plan's own fits produce every registered row on each arm.",
        "a superset of the retained set); the plan's own fits",
    ):
        assert not verify_plan._C18_NEG_DEFER_RE.search(benign), benign


@pytest.mark.parametrize(
    "decl",
    [
        "Row-coverage: the plan doesn't produce every registered row on each arm.",
        "Row-coverage: the plan doesn’t produce every registered row on each arm.",
        "Row-coverage: the plan cannot produce every registered row on each arm.",
        "Row-coverage: the plan can't produce every registered row on each arm.",
        "Row-coverage: the plan fails to produce every registered row on each arm.",
        "Row-coverage: the plan produces every registered row on each arm only until the refit lands.",
    ],
)
def test_c18_new_negation_forms_disqualify_affirmative(decl):
    # #1099 end-to-end: pre-fix, ALL six lines false-PASSed via the
    # affirmative-produces route (probe-reproduced).
    plan = _c18_plan(C18_V13_REGISTRATION, decl)
    assert _status(plan, "c18_paired_contrast_source_coverage") == "FAIL"


# ─── Check 19 — OOD generalization folds ───────────────────────────────────

C19_TRIGGER = "We fit a ridge predictor on the context grid and report held-out R^2 per layer."


def _predictive_plan() -> str:
    return GOOD_PLAN + "\n" + C19_TRIGGER + "\n"


def test_c19_kind_infra_skips():
    # Re-fixtured on the WARN fixture (v3): GOOD_PLAN+infra was near-vacuous
    # (it SKIPs at tier 2 under every kind) — the kind-exempt tier must fire
    # BEFORE the trigger tiers on a plan that WOULD otherwise WARN.
    _, by_id = _run(_predictive_plan(), kind="infra")
    res = by_id["c19_ood_folds"]
    assert res.status == "SKIP"
    assert "kind-exempt" in res.detail


def test_c19_not_triggered_skips():
    # GOOD_PLAN contains "40 held-out prompts" TWICE — the incidental
    # eval-split usage of "held-out" the conjunction trigger must NOT fire
    # on (the false-positive class the trigger design exists to avoid).
    assert _status(GOOD_PLAN, "c19_ood_folds") == "SKIP"


def test_c19_heldout_predictor_without_folds_warns():
    ok, by_id = _run(_predictive_plan())
    res = by_id["c19_ood_folds"]
    assert res.status == "WARN"
    assert ok is True  # WARN-only pin: c19 can never flip the overall verdict
    assert "group" in res.detail.lower()
    assert "810" in res.detail


def test_c19_solo_loco_vocabulary_triggers():
    # A fold token alone triggers — no held-out/predictor-stat conjunction
    # needed (any cross-validation mention makes the fold question right).
    plan = GOOD_PLAN + "\nPointwise LOCO is the headline fold.\n"
    assert _status(plan, "c19_ood_folds") == "WARN"


def test_c19_bare_leave_one_out_is_not_group_evidence():
    plan = GOOD_PLAN + "\nWe use leave-one-out cross-validation for the predictor.\n"
    assert _status(plan, "c19_ood_folds") == "WARN"


def test_c19_leave_one_pointwise_unit_does_not_pass():
    # #810's own offender fold was leave-one-CONTEXT-out (contexts were the
    # sample points) — a blocklisted unit must not count as group evidence.
    plan = GOOD_PLAN + "\nPredictor R^2 is held-out via leave-one-context-out.\n"
    assert _status(plan, "c19_ood_folds") == "WARN"


def test_c19_hyphenated_pointwise_unit_does_not_pass():
    # Must-Fix pin (round 1): the hyphen-split SUFFIX segment is blocklisted,
    # so `leave-one-data-point-out` (suffix `point`) cannot self-certify.
    plan = GOOD_PLAN + "\nPredictor R^2 is held-out via leave-one-data-point-out.\n"
    assert _status(plan, "c19_ood_folds") == "WARN"
    # Agglutinated variant: `datapoint` is blocklisted as an exact form.
    plan2 = GOOD_PLAN + "\nPredictor R^2 is held-out via leave-one-datapoint-out.\n"
    assert _status(plan2, "c19_ood_folds") == "WARN"


def test_c19_hyphenated_group_unit_passes():
    # Must-Fix pin (round 1): hyphenated GROUP units keep passing — suffix
    # `family` is not blocklisted.
    plan = (
        _predictive_plan() + "We register a leave-one-prompt-family-out fold; every headline is "
        "labeled with its fold.\n"
    )
    assert _status(plan, "c19_ood_folds") == "PASS"


def test_c19_non_iid_still_warns():
    # A NEGATED iid mention concedes group structure — it must not satisfy
    # the iid PASS tier (round-1 convergent critic concern).
    plan = (
        _predictive_plan() + "The context grid is not iid: prompt families induce group structure, "
        "but we report pointwise LOO only.\n"
    )
    assert _status(plan, "c19_ood_folds") == "WARN"
    plan2 = _predictive_plan() + "The sample is non-iid; we report pointwise LOO only.\n"
    assert _status(plan2, "c19_ood_folds") == "WARN"


def test_c19_lofo_passes():
    plan = (
        _predictive_plan()
        + "Grouping axes: prompt family, persona panel. We additionally register "
        "a leave-one-family-out (LOFO) fold; every headline is labeled with its fold.\n"
    )
    _, by_id = _run(plan)
    res = by_id["c19_ood_folds"]
    assert res.status == "PASS"
    assert "leave-one-family-out" in res.detail


def test_c19_corpus_transfer_passes():
    plan = (
        _predictive_plan()
        + "Fit on Betley, evaluate on UltraChat — the corpus-transfer arm is the "
        "group-level fold.\n"
    )
    assert _status(plan, "c19_ood_folds") == "PASS"


def test_c19_iid_argument_passes():
    plan = (
        _predictive_plan()
        + "The sample is genuinely iid: each row is an independent draw from one "
        "pool; no family/template structure.\n"
    )
    _, by_id = _run(plan)
    res = by_id["c19_ood_folds"]
    assert res.status == "PASS"
    assert "critic" in res.detail


def test_c19_na_escape_passes():
    plan = _predictive_plan() + "N/A — no held-out predictive DV.\n"
    _, by_id = _run(plan)
    res = by_id["c19_ood_folds"]
    assert res.status == "PASS"
    assert "N/A" in res.detail


def test_c19_na_bullet_form_standalone_passes():
    # Pins the helper's lstrip(" \t>*-") on the Goal-named site: a bullet-form
    # standalone declaration still counts (the only other lstrip pin is c12's).
    plan = _predictive_plan() + "- N/A — no held-out predictive DV (no predictor in this design).\n"
    assert _status(plan, "c19_ood_folds") == "PASS"


def test_c19_pasted_warn_detail_does_not_self_satisfy():
    # THE #974 guard (#4.2 c19 de-fang): pre-fix, the pasted WARN detail
    # satisfied FOUR channels at once — the doc-global NA escape, the
    # group-fold vocabulary ("GROUP-level fold" / "corpus transfer"), the
    # leave-one-family-out unit capture, and the iid regex via "no iid
    # argument" (the lookbehind does not cover "no ").
    base = _predictive_plan()
    _, by_id = _run(base)
    r = by_id["c19_ood_folds"]
    assert r.status == "WARN"
    pasted = base + f"\n{r.detail}\n"
    assert _status(pasted, "c19_ood_folds") == "WARN"


def test_c19_kind_analysis_triggers():
    # Scope pin: kind=analysis is IN scope (the #810/#761/#763 predictor
    # line lives in analysis follow-ups), same WARN-only severity.
    assert _status(_predictive_plan(), "c19_ood_folds", kind="analysis") == "WARN"


def test_c19_fenced_vocabulary_does_not_trigger():
    # strip_fences discipline: trigger vocabulary inside a code fence can
    # neither trip nor satisfy the check (mirrors
    # test_c10_fence_only_marker_vocab_does_not_trigger).
    plan = GOOD_PLAN + "\n```\n" + C19_TRIGGER + "\n```\n"
    assert _status(plan, "c19_ood_folds") == "SKIP"


# ─── Check 20 — verdict-lattice coherence ──────────────────────────────────

# Verbatim #923 §3 label bullets (plans/v4.md lines 49-51 = the co-firing v4
# shape; plans/v6.md lines 49-51 = the hand-fixed disjoint declaration). The
# REAL files are additionally exercised by the §10 acceptance commands; these
# inlined copies keep the suite self-contained.
C20 = "c20_verdict_lattice_coherence"
C20_HEADING = "## 3. Hypothesis"
C20_V4_HSLOT = """- **H-slot (slot artifact; scope's "pooling closes the gap"):** span-mean pooling recovers most of the full-prompt deficit. Confirmed if Δ_pool ≥ 0, OR the Δ_pool CI includes 0 AND the paired-diff CI excludes 0 on the positive side. Consequence: parent Takeaways bullets 1–2 are rescoped as last-token-slot-specific."""
C20_HROBUST = """- **H-robust (summary-robust deficit; scope's falsification):** Δ_pool < 0 with the 2000-draw family-bootstrap 95% CI excluding 0. Consequence: the slot-artifact explanation is dead; the "no attention-mixing advantage for linear read-out" takeaway hardens from estimator-scoped to summary-robust."""
C20_V4_INTERMEDIATE = """- **Intermediate:** Δ_pool CI includes 0 AND paired-diff CI includes 0 → no binary verdict; report the graded closure fraction 1 − Δ_pool/Δ_last with CI."""
C20_V6_HSLOT = """- **H-slot (slot artifact; scope's "pooling closes the gap"):** span-mean pooling recovers most of the full-prompt deficit. Confirmed iff the Δ_pool CI is wholly at/above 0, OR (the Δ_pool CI includes 0 AND the paired-diff CI is strictly positive). A bare positive point estimate with both CIs straddling 0 is NOT H-slot (it is intermediate). A still-negative significant Δ_pool with a strictly-positive paired diff is reported as "deficit persists with partial closure" under H-robust — never as gap closure. Consequence when confirmed: parent Takeaways bullets 1–2 are rescoped as last-token-slot-specific."""
C20_V6_INTERMEDIATE = """- **Intermediate:** Δ_pool CI includes 0 AND paired-diff CI includes 0 → no binary verdict; report the graded closure fraction 1 − Δ_pool/Δ_last with CI. The three labels are DISJOINT and exhaustive: H-robust ⇔ Δ_pool CI wholly below 0; H-slot ⇔ (Δ_pool CI wholly at/above 0) OR (Δ_pool CI straddles 0 AND paired-diff CI strictly positive); intermediate ⇔ otherwise. Exactly one label fires for every (Δ_pool, CI, paired-CI) cell."""

C20_V4_BULLETS = "\n".join((C20_V4_HSLOT, C20_HROBUST, C20_V4_INTERMEDIATE))
C20_V6_BULLETS = "\n".join((C20_V6_HSLOT, C20_HROBUST, C20_V6_INTERMEDIATE))


def _c20_plan(bullets: str) -> str:
    """GOOD_PLAN + a hypothesis section carrying the given label bullets."""
    return GOOD_PLAN + "\n" + C20_HEADING + "\n\n" + bullets + "\n"


def test_c20_not_triggered_skips():
    _, by_id = _run(GOOD_PLAN)
    r = by_id[C20]
    assert r.status == "SKIP"
    assert "no registered verdict lattice" in r.detail


def test_c20_kind_infra_skips():
    assert _status(_c20_plan(C20_V4_BULLETS), C20, kind="infra") == "SKIP"


def test_c20_923_v4_cofire_shape_fails():
    ok, by_id = _run(_c20_plan(C20_V4_BULLETS))
    r = by_id[C20]
    assert r.status == "FAIL"
    assert ok is False
    # Both co-firing labels named, plus the co-fire cell in plain terms.
    assert "H-slot" in r.detail and "Intermediate" in r.detail
    assert "CO-FIRE" in r.detail
    assert "{point > 0, primary CI straddles 0, paired CI straddles 0}" in r.detail
    # The remedy menu closes the detail.
    assert "DISJOINT and exhaustive" in r.detail
    assert "N/A — no registered verdict lattice" in r.detail


def test_c20_923_v4_gap_cell_named_in_detail():
    _, by_id = _run(_c20_plan(C20_V4_BULLETS))
    r = by_id[C20]
    assert "no label fires" in r.detail
    assert "{point < 0, primary CI straddles 0, paired CI wholly below 0}" in r.detail


def test_c20_923_v6_disjoint_declaration_passes():
    # Tier 1 (the ⇔ declaration) takes precedence over the per-label prose.
    _, by_id = _run(_c20_plan(C20_V6_BULLETS))
    r = by_id[C20]
    assert r.status == "PASS"
    assert "tier 1" in r.detail


def test_c20_declared_cofire_fails():
    decl = (
        "- The two labels are DISJOINT and exhaustive: H-a ⇔ Δ_pool CI includes 0; "
        "H-b ⇔ Δ_pool CI includes 0 OR Δ_pool CI wholly below 0."
    )
    ok, by_id = _run(_c20_plan(decl))
    r = by_id[C20]
    assert r.status == "FAIL"
    assert ok is False
    assert "CO-FIRE" in r.detail
    assert "H-a" in r.detail and "H-b" in r.detail


def test_c20_declared_gap_without_otherwise_fails():
    decl = (
        "- The two labels are DISJOINT and exhaustive: H-a ⇔ Δ_pool CI wholly below 0; "
        "H-b ⇔ Δ_pool CI wholly at/above 0."
    )
    ok, by_id = _run(_c20_plan(decl))
    r = by_id[C20]
    assert r.status == "FAIL"  # the plan claimed exhaustiveness — a gap falsifies it
    assert ok is False
    assert "no label fires" in r.detail
    assert "straddles 0" in r.detail


def test_c20_kind_analysis_degrades_to_warn():
    ok, by_id = _run(_c20_plan(C20_V4_BULLETS), kind="analysis")
    r = by_id[C20]
    assert r.status == "WARN"
    assert ok is True


@pytest.mark.parametrize("kind", ["experiment", "analysis"])
def test_c20_na_escape_with_detected_lattice_warns(kind):
    # #1223: the escape is only reachable when a lattice WAS detected (the
    # no-lattice case SKIPs first), so N/A + detected lattice is the masking
    # shape — WARN (non-blocking), never a silent PASS. Same severity both
    # kinds: the co-occurrence is a meta-signal, already sub-FAIL.
    plan = _c20_plan(C20_V4_BULLETS) + (
        "\nN/A — no registered verdict lattice (the labels quote the parent's methodology).\n"
    )
    ok, by_id = _run(plan, kind=kind)
    r = by_id[C20]
    assert r.status == "WARN"
    assert ok is True  # WARN never blocks
    assert "co-occurs" in r.detail
    assert "tier 2" in r.detail


def test_c20_na_escape_with_tier1_declaration_warns():
    # The literal broken-lattice-masked shape from the task body: a BROKEN
    # declared partition (the test_c20_declared_cofire_fails fixture) plus
    # the N/A line — previously a silent PASS, now WARN naming tier 1.
    decl = (
        "- The two labels are DISJOINT and exhaustive: H-a ⇔ Δ_pool CI includes 0; "
        "H-b ⇔ Δ_pool CI includes 0 OR Δ_pool CI wholly below 0."
    )
    plan = _c20_plan(decl) + "\nN/A — no registered verdict lattice\n"
    ok, by_id = _run(plan)
    r = by_id[C20]
    assert r.status == "WARN"
    assert ok is True
    assert "tier-1" in r.detail


def test_c20_na_escape_without_lattice_still_skips():
    # Preserved: with no detected lattice the SKIP gate fires before the
    # escape is consulted — the N/A line stays legal and never penalized.
    plan = GOOD_PLAN + "\nN/A — no registered verdict lattice\n"
    _, by_id = _run(plan)
    assert by_id[C20].status == "SKIP"


def test_c20_unparseable_label_warns_not_fails():
    # #405-style direction/magnitude vocabulary outside the v1 atom lexicon:
    # the stray `≥` comparator is completeness-gate residue → WARN, never FAIL.
    bullets = (
        "- **H-flip:** Confirmed if the Δ_pool CI excluding 0, with the OPPOSITE "
        "direction, more negative by ≥ 0.5 nats.\n"
        "- **H-same:** Confirmed if the Δ_pool CI includes 0."
    )
    ok, by_id = _run(_c20_plan(bullets))
    r = by_id[C20]
    assert r.status == "WARN"
    assert ok is True
    assert "did not fully parse" in r.detail


def test_c20_quantified_labels_skip():
    # #922-class k-of-n predicates are out of the v1 cell algebra.
    bullets = (
        "- **H-transfer:** Confirmed if the paired CI is clear of zero at ≥4/6 "
        "pre-registered layers.\n"
        "- **H-null:** Confirmed if the paired CI includes 0 at ≥4/6 pre-registered layers."
    )
    _, by_id = _run(_c20_plan(bullets))
    r = by_id[C20]
    assert r.status == "SKIP"
    assert "quantified" in r.detail


def test_c20_prose_gap_only_warns():
    bullets = (
        "- **H-pos:** Confirmed if the Δ_pool CI is wholly at/above 0.\n"
        "- **H-neg:** Confirmed if the Δ_pool CI is wholly below 0."
    )
    ok, by_id = _run(_c20_plan(bullets))
    r = by_id[C20]
    assert r.status == "WARN"  # tier-2 gap degrades to WARN (harvest recall)
    assert ok is True
    assert "no label fires" in r.detail
    assert "straddles 0" in r.detail


def test_c20_otherwise_label_covers_gap_passes():
    bullets = (
        "- **H-pos:** Confirmed if the Δ_pool CI is wholly at/above 0.\n"
        "- **H-neg:** Confirmed if the Δ_pool CI is wholly below 0.\n"
        "- **Inconclusive:** otherwise — no binary verdict."
    )
    _, by_id = _run(_c20_plan(bullets))
    assert by_id[C20].status == "PASS"


def test_c20_single_label_skips():
    bullets = "- **H-pos:** Confirmed if the Δ_pool CI is wholly at/above 0."
    _, by_id = _run(_c20_plan(bullets))
    r = by_id[C20]
    assert r.status == "SKIP"  # a single gate is c8/c13 territory, not a lattice
    assert "fewer than 2" in r.detail


def test_c20_fenced_lattice_ignored():
    plan = GOOD_PLAN + "\n" + C20_HEADING + "\n\n```\n" + C20_V4_BULLETS + "\n```\n"
    assert _status(plan, C20) == "SKIP"


def test_c20_negated_predicate_warns_not_fails():
    # Must-Fix 1: "never includes 0" must NOT match the positive `includes 0`
    # atom with inverted polarity (which would manufacture a co-fire FAIL
    # against H-mid) — the negator is residue, so the lattice degrades to WARN.
    bullets = (
        "- **H-null:** Confirmed if the Δ_pool CI never includes 0.\n"
        "- **H-mid:** Confirmed if the Δ_pool CI includes 0."
    )
    ok, by_id = _run(_c20_plan(bullets))
    r = by_id[C20]
    assert r.status == "WARN"
    assert ok is True


def test_c20_mixed_point_quantities_warn():
    # Must-Fix 2i: two distinct point quantities cannot share the single
    # point axis — fail closed to WARN, never a silent single-axis collapse.
    bullets = (
        "- **H-up:** Confirmed if Δ_pool ≥ 0 AND the Δ_pool CI excludes 0.\n"
        "- **H-down:** Confirmed if D < 0 AND the paired-diff CI excludes 0."
    )
    ok, by_id = _run(_c20_plan(bullets))
    r = by_id[C20]
    assert r.status == "WARN"
    assert ok is True
    assert "distinct point quantities" in r.detail


def test_c20_post_ci_paired_binding_warns():
    # Must-Fix 2ii: post-CI "paired" wording is never silently bound to the
    # primary axis.
    bullets = (
        "- **H-paired:** Confirmed if the CI of the paired difference includes 0.\n"
        "- **H-clear:** Confirmed if the Δ_pool CI excludes 0."
    )
    ok, by_id = _run(_c20_plan(bullets))
    r = by_id[C20]
    assert r.status == "WARN"
    assert ok is True
    assert "not silently bound" in r.detail


def test_c20_v6_prose_without_decl_tier2_warns():
    # The v6 per-label bullets WITHOUT the ⇔ recap take the tier-2 path: the
    # v6 wording drops v4's bare `Δ_pool ≥ 0` disjunct, so there is no
    # co-fire, but {primary straddles, paired wholly below} is uncovered →
    # WARN (not FAIL, not PASS). Guards tier-2 semantics against overfit to
    # the v4 wording.
    bullets = "\n".join((C20_V6_HSLOT, C20_HROBUST, C20_V4_INTERMEDIATE))
    ok, by_id = _run(_c20_plan(bullets))
    r = by_id[C20]
    assert r.status == "WARN"
    assert ok is True
    assert "no label fires" in r.detail
    assert "paired CI wholly below 0" in r.detail


def test_c20_precedence_phrase_warns():
    # Round-1 review Minor (Codex): an ordering declaration in the lattice's
    # section makes the labels order-evaluated — the cell algebra cannot
    # verify an ordered lattice, so _C20_PRECEDENCE_RE fails the lattice
    # closed to unparsed → WARN, never FAIL/PASS.
    bullets = (
        "- **H-pos:** Confirmed if the Δ_pool CI is wholly at/above 0.\n"
        "- **H-neg:** Confirmed if the Δ_pool CI is wholly below 0.\n"
        "\nThe labels above are evaluated in order; the first match is reported.\n"
    )
    ok, by_id = _run(_c20_plan(bullets))
    r = by_id[C20]
    assert r.status == "WARN"
    assert ok is True
    assert "label-precedence phrase" in r.detail
    assert "'evaluated in order'" in r.detail


def test_c20_unrecognized_connective_warns():
    # Round-1 review Minor (Codex): only AND / OR / `with` join atoms.
    # `and/or` (two joiner hits) and a bare comma without OR (zero hits) both
    # break the exactly-one-connective rule and fail closed to unparsed →
    # WARN, never FAIL/PASS (a silently defaulted connective could invert
    # the lattice algebra).
    for joiner in (" and/or ", ", "):
        bullets = (
            "- **H-wide:** Confirmed if the Δ_pool CI excludes 0"
            + joiner
            + "the paired-diff CI excludes 0.\n"
            "- **H-mid:** Confirmed if the Δ_pool CI includes 0."
        )
        ok, by_id = _run(_c20_plan(bullets))
        r = by_id[C20]
        assert r.status == "WARN", joiner
        assert ok is True, joiner
        assert "joiner between atoms is not exactly one of AND/OR/with" in r.detail


def test_c20_paired_before_primary_binds_primary_axis():
    # Round-1 review Minor (Claude): a paired-CI atom < 40 chars BEFORE a
    # primary-CI atom must not leak its `paired` token into the next atom's
    # axis lookback. Unclamped, BOTH H-int atoms bind the paired axis — a
    # contradictory conjunction that never fires — so H-null (paired
    # straddle, primary unconstrained) CO-FIREs with H-pos and the
    # {primary straddle, paired below/above} cells go uncovered: a
    # manufactured tier-1 FAIL. With the lookback clamped at the previous
    # atom's span end the declaration is a clean 3x3 partition → PASS
    # (the fixture is chosen so the wrong binding flips the verdict).
    seg = "paired-diff CI excludes 0 AND Δ_pool CI straddles 0"
    atoms, _ = verify_plan._c20_collect_atoms(seg)
    assert [a[0] for a in atoms] == ["paired", "primary"]
    decl = (
        "- The labels are DISJOINT and exhaustive: "
        "H-pos ⇔ the Δ_pool CI excludes 0; "
        "H-int ⇔ paired-diff CI excludes 0 AND Δ_pool CI straddles 0; "
        "H-null ⇔ paired-diff CI includes 0 AND Δ_pool CI straddles 0."
    )
    ok, by_id = _run(_c20_plan(decl))
    r = by_id[C20]
    assert r.status == "PASS"
    assert ok is True
    assert "tier 1" in r.detail


# ─── Check 21 — grep-arity acceptance gate ─────────────────────────────────

# Verbatim #1024 gate shapes (plans/v2.md lines 175 + 25 = the incident;
# inlined copies keep the suite self-contained — task folders move between
# status dirs, so tests never hard-read tasks/<status>/1024 paths).
C21 = "c21_grep_arity_gate"
C21_PIPELINE_GATE = (
    '\n- Audit grep → 0 two-arg shared-parser calls: `grep -rn "parse_judge_json(" '
    'src/ scripts/ tests/ | grep -v run_em_multiseed | grep ", " | wc -l` == 0.\n'
)
C21_PROSE_GATE = (
    "\n2. Every call site compiles with the new signature; `grep -rn 'parse_judge_json(' "
    "src/ scripts/ tests/` shows zero two-argument calls of the shared parser.\n"
)
C21_AST_EVIDENCE = (
    "\nAudit: ast.parse each target file, ast.walk over Call nodes whose func resolves "
    "to parse_judge_json, count len(node.args) + len(node.keywords), whitelist the "
    "deliberate two-arg pytest.raises test.\n"
)


def test_c21_no_trigger_skips():
    for kind in ("experiment", "infra"):
        _, by_id = _run(GOOD_PLAN, kind=kind)
        r = by_id[C21]
        assert r.status == "SKIP", kind
        assert "no grep-based call-arity" in r.detail


def test_c21_pipeline_arity_gate_warns():
    ok, by_id = _run(GOOD_PLAN + C21_PIPELINE_GATE, kind="infra")
    r = by_id[C21]
    assert r.status == "WARN"
    assert ok is True  # WARN never blocks
    assert "ast.walk" in r.detail
    assert "#1024" in r.detail
    assert "N/A — no arity acceptance gate" in r.detail


def test_c21_prose_arity_gate_warns():
    assert _status(GOOD_PLAN + C21_PROSE_GATE, C21, kind="infra") == "WARN"


def test_c21_kind_experiment_also_fires():
    # Pins the all-kinds scope: signature migrations ride experiment plans'
    # code-port phases too (plan §11 item 4).
    assert _status(GOOD_PLAN + C21_PIPELINE_GATE, C21, kind="experiment") == "WARN"


def test_c21_ast_evidence_passes():
    _, by_id = _run(GOOD_PLAN + C21_PIPELINE_GATE + C21_AST_EVIDENCE, kind="infra")
    r = by_id[C21]
    assert r.status == "PASS"
    assert "AST" in r.detail


def test_c21_na_escape_passes():
    plan = GOOD_PLAN + C21_PIPELINE_GATE + "\nN/A — no arity acceptance gate.\n"
    _, by_id = _run(plan, kind="infra")
    r = by_id[C21]
    assert r.status == "PASS"
    assert "N/A" in r.detail


def test_c21_quoted_na_phrase_does_not_escape():
    # Anti-paste guard: a mid-sentence quote of the escape phrase (the shape
    # a pasted remedy menu produces) must not satisfy the standalone-line
    # escape. Quoted-phrase form, not full-detail paste — the AST-evidence
    # channel of the pasted detail is a disclosed residual (plan §8 row 6).
    base = GOOD_PLAN + C21_PIPELINE_GATE
    assert _status(base, C21, kind="infra") == "WARN"
    quoted = base + (
        "\nThe remedy menu says to declare 'N/A — no arity acceptance gate' on its own line.\n"
    )
    assert _status(quoted, C21, kind="infra") == "WARN"


def test_c21_discovery_grep_does_not_trigger():
    # A repro-card discovery/enumeration grep (no count form, no arity
    # vocabulary, no zero comparator) must never fire.
    plan = GOOD_PLAN + '\nRepro: `grep -rn "parse_judge_json(" src/ scripts/ tests/`\n'
    assert _status(plan, C21, kind="infra") == "SKIP"


def test_c21_plain_removal_gate_does_not_trigger():
    # A full-removal migration gate (call-pattern + count + zero comparator,
    # but NO comma pattern and NO arity vocabulary) counts call sites, not
    # argument arity — legitimate and out of scope.
    plan = GOOD_PLAN + '\n- `grep -rn "old_fn(" src/ | wc -l` == 0.\n'
    assert _status(plan, C21, kind="infra") == "SKIP"


def test_c21_fenced_gate_still_triggers():
    # Pins the no-fence-mask decision: gate commands live in fenced
    # verification blocks too (plan §4.1).
    plan = GOOD_PLAN + "\n```bash" + C21_PIPELINE_GATE + "```\n"
    assert _status(plan, C21, kind="infra") == "WARN"


def test_c21_comma_call_grep_without_count_does_not_trigger():
    # A comma inside a call-shaped grep pattern without any count form or
    # zero comparator is discovery, not a gate.
    plan = GOOD_PLAN + '\n`grep -rn "load(x, y)" docs/`\n'
    assert _status(plan, C21, kind="infra") == "SKIP"


def test_c21_branch_a_only_pipeline_gate_warns():
    # Reconciler MF-2: pins Branch A's DETECTION-LOSS direction — every other
    # WARN fixture co-fires Branch B, so a dead Branch A would otherwise ship
    # green. No arity vocabulary, no zero comparator on this line.
    plan = GOOD_PLAN + (
        '\n- `grep -rn "load_config(" src/ | grep ", " | wc -l` must return no rows.\n'
    )
    assert _status(plan, C21, kind="infra") == "WARN"


def test_c21_grep_dash_c_count_form_warns():
    # Claude-statistics concern 2: the `grep -c` flag-cluster branch of
    # _C21_COUNT_RE has no other coverage.
    plan = GOOD_PLAN + '\n- `grep -rnc "parse_judge_json(.*," src/` == 0 two-arg calls.\n'
    assert _status(plan, C21, kind="infra") == "WARN"


def test_c21_na_wins_over_ast_evidence():
    # Escape-ORDER pin (shared Claude/Codex concern): with BOTH escapes
    # present the N/A detail is reported, not the AST-evidence detail.
    plan = GOOD_PLAN + C21_PIPELINE_GATE + C21_AST_EVIDENCE + "\nN/A — no arity acceptance gate.\n"
    _, by_id = _run(plan, kind="infra")
    r = by_id[C21]
    assert r.status == "PASS"
    assert "N/A" in r.detail
    assert "AST-based arity audit" not in r.detail


# ─── Check 22 — cross-section param consistency ────────────────────────────

C22 = "c22_cross_section_param_consistency"


def _c21(text: str, kind: str = "infra"):
    return verify_plan.check_cross_section_param_consistency(text, kind)


def test_c22_contradictory_restatement_warns():
    # Brief item (a): the same tracked param with contradictory values in two
    # top-level sections WARNs; detail names the param, both matched spans
    # (values included), both section headings, and both 1-based line
    # numbers. Run under kind="infra" — pins the all-kinds scope (#1024's
    # offender is kind: infra).
    plan = (
        "## 4. Design\n\n"
        "Judging runs at temperature=1.0 for every draw.\n\n"
        "## 11. Rationale\n\n"
        "- **What:** replicate temperature=0.7, same rubric.\n"
    )
    r = _c21(plan, kind="infra")
    assert r.status == "WARN"
    assert "temperature:" in r.detail
    assert "temperature=1.0" in r.detail and "temperature=0.7" in r.detail
    assert "4. Design" in r.detail and "11. Rationale" in r.detail
    assert "L3" in r.detail and "L7" in r.detail  # 1-based line numbers


def test_c22_same_value_restated_passes():
    # Brief item (b): identical restatement across sections is consistent.
    plan = (
        "## 4. Design\n\nTraining uses lr = 3e-5 throughout.\n\n"
        "## 11. Rationale\n\n- lr = 3e-5. Source: #612.\n"
    )
    assert _c21(plan).status == "PASS"
    # Float normalization: 1e-4 == 0.0001 (scientific vs decimal notation).
    plan_norm = (
        "## 4. Design\n\nTraining uses lr = 1e-4 throughout.\n\n"
        "## 11. Rationale\n\n- lr = 0.0001. Source: #612.\n"
    )
    assert _c21(plan_norm).status == "PASS"


def test_c22_historical_clause_not_counted():
    # Brief item (c): a value inside a historical / declared-but-never-
    # threaded clause is excluded, so the param spans only ONE section → SKIP.
    plan = (
        "## 4. Design\n\nThe run trains at lr = 5e-6.\n\n"
        "## 11. Rationale\n\n- lr = 1e-4 (declared but never threaded into the trainer).\n"
    )
    assert _c21(plan).status == "SKIP"


def test_c22_sweep_range_set_not_flagged():
    # Brief item (d): sweep/range/brace-set declarations overlapping a single
    # grounded value are consistent (set-overlap, not set-equality).
    # Head overlap: schedule "1e-4 → 1e-5" vs the grounded head value.
    plan_head = (
        "## 4. Design\n\nThe lr schedule is lr = 1e-4 → 1e-5 over training.\n\n"
        "## 11. Rationale\n\n- lr = 1e-4. Source: #612.\n"
    )
    assert _c21(plan_head).status == "PASS"
    # END-overlap discriminator (round-1 MF-1 cell 1): {1, 3} ∩ {3} passes
    # ONLY if the range-tail parse is live — a dead _C22_RANGE_TAIL_RE
    # yields {1} vs {3} → spurious WARN and this assertion goes red.
    plan_end = (
        "## 4. Design\n\nWe anneal epochs = 1 → 3 across restarts.\n\n"
        "## 11. Rationale\n\n- epochs = 3. Source: arXiv 2507.21509 appendix table.\n"
    )
    assert _c21(plan_end).status == "PASS"
    # Brace-set vs member value (seed list restated as one headline seed).
    plan_set = (
        "## 3. Conditions\n\nSeeds: {42, 137, 256} shared across conditions.\n\n"
        "## 10. Repro\n\nThe headline cell re-runs seed=137 only.\n"
    )
    assert _c21(plan_set).status == "PASS"
    # A `sweep`-line occurrence is skipped entirely: lr then spans one
    # section → SKIP (were it counted, {1e-4} vs {5e-6} would WARN).
    plan_sweep = (
        "## 4. Design\n\nThe sweep varies lr = 1e-4 across cells.\n\n"
        "## 11. Rationale\n\n- lr = 5e-6. Source: #612.\n"
    )
    assert _c21(plan_sweep).status == "SKIP"


def test_c22_value_vs_omission_1024_shape_warns():
    # Brief item (e): the #1024 v2 offender shape — a corrected
    # "temperature OMITTED … API default 1.0" section vs a stale §11
    # `temperature=0.7` *What:* restatement. Also pins that
    # `JUDGE_TEMPERATURE=0.7` (compound token, no \b before `temperature`)
    # creates no occurrence: if it did, the §7 value set would gain 0.7 and
    # OVERLAP §11's {0.7} → PASS, so the WARN assertion itself discriminates.
    plan = (
        "## 7. Decision gates\n\n"
        "Scoring uses `max_tokens=64`, temperature OMITTED — the #778 request builders "
        "never set it, so the API default 1.0 applies (JUDGE_TEMPERATURE=0.7 is a separate "
        "env knob).\n\n"
        "## 11. Hyperparameter grounding\n\n"
        "- *What:* replicate `max_tokens=64`, `temperature=0.7`, same rubric.\n"
    )
    r = _c21(plan, kind="infra")
    assert r.status == "WARN"
    assert "temperature:" in r.detail
    assert "temperature OMITTED" in r.detail and "temperature=0.7" in r.detail
    assert "max_tokens" not in r.detail  # 64 == 64 across sections — consistent


def test_c22_per_phase_and_alpha_context():
    # Phase-qualified values key separately (epochs@phase1 / epochs@phase2 —
    # each spans one section → SKIP).
    plan_phase = (
        "## 4. Design\n\nPhase 1 trains epochs = 3 on the coupling mix.\n\n"
        "## 5. Pipeline\n\nPhase 2 runs epochs = 1 on the EM mix.\n"
    )
    assert _c21(plan_phase).status == "SKIP"
    # Stats-alpha (no LoRA context) never enters the pool; the LoRA alpha
    # spans one section → SKIP (were stats-alpha counted, 0.05 vs 64 → WARN).
    plan_alpha = (
        "## 6. Stats\n\nSignificance uses alpha: 0.05 two-sided.\n\n"
        "## 11. Rationale\n\n- LoRA alpha = 64. Source: #474.\n"
    )
    assert _c21(plan_alpha).status == "SKIP"


def test_c22_fenced_commands_ignored():
    # A fenced launch command's seed=999 never votes (fence mask), so seed
    # spans only the §4 prose → SKIP.
    plan = (
        "## 4. Design\n\nSeeds: {42} for the smoke cell.\n\n"
        "## 10. Repro\n\n```bash\nuv run python scripts/train.py seed=999\n```\n"
    )
    assert _c21(plan).status == "SKIP"


def test_c22_registration_end_to_end():
    # End-to-end through verify_plan_text: pins registration in CHECKS +
    # fence-mask integration + WARN-never-blocks (overall ok stays True).
    filler = "We re-run the aggregation over the existing eval JSONs and re-plot. " * 25
    plan = (
        "# Plan — Task #997: c22 end-to-end fixture (analysis)\n\n"
        "## Goal\n\n" + filler + "\n\n"
        "## 4. Design\n\nJudging runs at temperature=1.0 for every draw.\n\n"
        "## 11. Rationale\n\n- **What:** replicate temperature=0.7, same rubric.\n\n"
        "## Resources\n\nNo pod. `Estimated GPU-hours (total): 0`\n"
    )
    ok, by_id = _run(plan, kind="analysis")
    assert by_id[C22].status == "WARN"
    assert ok is True  # WARN never blocks exit


def test_c22_alias_contradictions_warn():
    # Round-1 MF-1 cell 2 (silent-SKIP escape): learning_rate/lr and
    # batch_size/batch fold to one key — a dropped _C22_ALIASES fold leaves
    # two single-section keys → SKIP → this test goes red.
    plan_lr = (
        "## 4. Design\n\nTraining uses learning_rate = 3e-5 for the full run.\n\n"
        "## 11. Rationale\n\n- lr = 1e-4. Source: #612.\n"
    )
    r = _c21(plan_lr)
    assert r.status == "WARN"
    assert r.detail.startswith("lr:")  # names the FOLDED param
    plan_batch = (
        "## 4. Design\n\nWe pack batch_size: 16 per device.\n\n"
        "## 11. Rationale\n\n- batch = 32. Source: ungrounded — needs smoke-test.\n"
    )
    assert _c21(plan_batch).status == "WARN"
    # Same-value alias restatement: fold + overlap together → PASS.
    plan_same = (
        "## 4. Design\n\nTraining uses learning_rate = 3e-5 for the full run.\n\n"
        "## 11. Rationale\n\n- lr = 3e-5. Source: #612.\n"
    )
    assert _c21(plan_same).status == "PASS"


def test_c22_sibling_h3_one_h2_not_flagged():
    # Round-1 MF-1 cell 3 (pins FP layer 2): sibling H3 arms under ONE H2
    # attribute to the shared H2 ancestor — one section, so the union
    # {1e-4, 1e-5} spans one section → SKIP. Non-phase H3 names are chosen so
    # phase keying cannot mask a grouping bug; a broken innermost-H3 grouping
    # yields two disjoint sections → spurious WARN → red test.
    plan = (
        "## 4. Design\n\n"
        "### Arm A\n\nThis arm trains lr = 1e-4.\n\n"
        "### Arm B\n\nThis arm trains lr = 1e-5.\n"
    )
    assert _c21(plan).status == "SKIP"


def test_c22_omission_with_exclusion_vocab_still_counts():
    # Round-1 MF-1 cell 4 (pins the §3.4 omission-exemption): exclusion
    # vocabulary in-window does NOT filter an omission match — the corrected
    # text legitimately explains WHY the param is omitted. An implementation
    # that wrongly exclusion-filters omissions sees temperature in one
    # section only → SKIP → red test. (The parenthetical `was 0.7` is not a
    # `param[=:]` token, so no competing value occurrence exists on the line.)
    plan = (
        "## 7. Decision gates\n\n"
        "temperature OMITTED (was 0.7, never threaded — superseded).\n\n"
        "## 11. Rationale\n\n- temperature=0.7. Source: #1024 v2.\n"
    )
    r = _c21(plan)
    assert r.status == "WARN"
    assert "temperature OMITTED" in r.detail


def test_c22_consistent_corrected_plan_stays_quiet():
    # Durable negative control (round-1 standing recommendation): a minimal
    # post-correction v3-shaped fixture — the omission form lives in ONE
    # section only, lr is restated identically across §4/§11, and
    # JUDGE_TEMPERATURE=0.7 compound tokens never match — stays quiet (no
    # WARN), pinning the "corrected plan stays quiet" property against
    # future exclusion-vocab tuning.
    plan = (
        "## 4. Design\n\n"
        "Scoring: `max_tokens=64`, temperature OMITTED (API default 1.0; "
        "JUDGE_TEMPERATURE=0.7 is a separate env knob).\n"
        "The run trains lr = 5e-6 on the coupling mix.\n\n"
        "## 11. Rationale\n\n- lr = 5e-6. Source: #612.\n"
    )
    r = _c21(plan)
    assert r.status in ("PASS", "SKIP")  # the pinned property: no WARN
    assert r.status == "PASS"  # lr restated consistently across two sections


# ─── Check 24 — resume-skip provenance validation ──────────────────────────

# Fixture shapes distilled from the motivating corpus (#952 v9 = the gap the
# critic ensemble caught; v12 = the gate-5 provenance-manifest contract it
# forced; the boilerplate shape = the "Completion provenance: N/A" §4-design
# bullet that false-satisfied bare-noun satisfiers on #811 v1 / #622 v2).
C24_V9_SHAPE = (
    "\nPhase B: a fold loop writing per-fold outputs; per-fold persist + resume-skip "
    "(each fold's outputs written when the fold completes; entry predicate skips "
    "completed folds — the intra-phase persist law, projected loop >1 h).\n"
)
C24_V12_SHAPE = (
    "\nPhase B: per-fold persist + resume-skip under the gate-5 provenance contract "
    "(each fold's outputs written with their provenance manifest when the fold "
    "completes; the entry predicate accepts a persisted fold only on full manifest "
    "match — split sha256 hashes + git SHA + env fingerprint; mismatch = recompute).\n"
)
C24_BOILERPLATE_SHAPE = (  # the corpus false-PASS shape (#811 v1 / #622 v2)
    "\nPhase B: per-fold persist + resume-skip (entry predicate skips completed folds).\n"
    + "\n" * 10
    + "Completion provenance: N/A — no behavior-implantation training rows in this task.\n"
)


def test_c24_952_v12_shape_passes():
    # The #952 v12 gate-5 contract satisfies via the COMPOUND forms
    # (`provenance manifest` / `manifest match`) — survives the bare-noun
    # exclusion.
    ok, by_id = _run(GOOD_PLAN + C24_V12_SHAPE)
    assert by_id["c24_resume_provenance"].status == "PASS"
    assert ok


def test_c24_952_v9_shape_warns():
    _, by_id = _run(GOOD_PLAN + C24_V9_SHAPE)
    r = by_id["c24_resume_provenance"]
    assert r.status == "WARN"
    # The detail names the fingerprint vocabulary and quotes the escape phrase.
    assert "code SHA" in r.detail
    assert "N/A — no resume/persist pattern" in r.detail


def test_c24_completion_provenance_boilerplate_warns():
    # The REQUIRED "Completion provenance: N/A" design bullet within ±15 raw
    # lines of a bare resume-skip must NOT satisfy (bare `provenance` is
    # excluded from the satisfier — the #811 v1 / #622 v2 false-PASS class).
    assert _status(GOOD_PLAN + C24_BOILERPLATE_SHAPE, "c24_resume_provenance") == "WARN"


def test_c24_no_trigger_skips():
    assert _status(GOOD_PLAN, "c24_resume_provenance") == "SKIP"


def test_c24_na_escape_passes():
    plan = (
        GOOD_PLAN
        + C24_V9_SHAPE
        + "\nN/A — no resume/persist pattern (the resume-skip line quotes the sibling's "
        "methodology).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c24_resume_provenance"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


def test_c24_na_escape_spaced_slash_passes():
    # The escape tail is slash-spacing-tolerant: a hand-written
    # "resume / persist" must not silently fail the escape.
    plan = (
        GOOD_PLAN
        + C24_V9_SHAPE
        + "\nN/A — no resume / persist pattern (the resume-skip line quotes the sibling's "
        "methodology).\n"
    )
    assert _status(plan, "c24_resume_provenance") == "PASS"


def test_c24_quoted_na_phrase_does_not_escape():
    # A mid-sentence quoted escape phrase (the pasted-bounce-brief
    # self-escape channel) is NOT a standalone declaration line and must not
    # escape — the c13/c18 anti-paste twin.
    plan = (
        GOOD_PLAN
        + C24_V9_SHAPE
        + ("\nThe remedy menu says to declare 'N/A — no resume/persist pattern' on its own line.\n")
    )
    assert _status(plan, "c24_resume_provenance") == "WARN"


@pytest.mark.parametrize("kind", ["infra", "batch"])
def test_c24_kind_exempt_skips(kind):
    assert _status(GOOD_PLAN + C24_V9_SHAPE, "c24_resume_provenance", kind=kind) == "SKIP"


def test_c24_kind_analysis_warns():
    # c24 is WARN for BOTH kinds (the c19 severity precedent) — severity does
    # NOT degrade to SKIP for analysis, unlike c12's FAIL/WARN split.
    assert _status(GOOD_PLAN + C24_V9_SHAPE, "c24_resume_provenance", kind="analysis") == "WARN"


def test_c24_fenced_trigger_does_not_trigger():
    # Resume-skip vocabulary appearing ONLY inside a code fence is not a
    # resume plan — pins the fence-masked trigger path (`_trigger_windows`).
    plan = GOOD_PLAN + "\n```\n" + C24_V9_SHAPE.strip() + "\n```\n"
    assert _status(plan, "c24_resume_provenance") == "SKIP"


def test_c24_satisfier_outside_window_warns():
    # Window scoping is load-bearing: a genuine provenance contract >15 raw
    # lines from every resume mention does not satisfy.
    plan = (
        GOOD_PLAN
        + "\nPhase B: per-fold persist + resume-skip at each cell entry.\n"
        + "\n" * 20
        + "Acceptance: the provenance manifest (split sha256 hashes + git SHA) is "
        "checked once per run.\n"
    )
    assert _status(plan, "c24_resume_provenance") == "WARN"


def test_c24_window_radius_boundary():
    # Mirrors test_c12_window_radius_boundary_after_refactor: c24's
    # ±15-raw-line radius must not shift. Evidence exactly 15 raw lines below
    # the trigger line counts; one line farther does not. The evidence line
    # deliberately avoids c24 trigger vocabulary so it cannot open its own
    # window (asserted below, not assumed).
    evidence = (
        "Acceptance: full provenance-manifest match — split sha256 hashes + git SHA + "
        "env fingerprint."
    )
    assert not verify_plan._C24_TRIGGER_RE.search(evidence)
    trigger = "Phase B: per-fold persist + resume-skip at each cell entry."
    assert not verify_plan._C24_FINGERPRINT_RE.search(trigger)

    def plan_with_gap(n_blank: int) -> str:
        # Evidence lands (n_blank + 1) raw lines below the trigger line.
        return (
            GOOD_PLAN
            + "\n## 12. Resume contract\n\n"
            + trigger
            + "\n"
            + "\n" * n_blank
            + evidence
            + "\n"
        )

    assert _status(plan_with_gap(14), "c24_resume_provenance") == "PASS"  # +15 lines: in
    assert _status(plan_with_gap(15), "c24_resume_provenance") == "WARN"  # +16 lines: out


@pytest.mark.parametrize(
    "trigger",
    [
        "a resume predicate keyed on output existence",
        "skip-if-exists at cell entry",
        "checkpoint-resume across waves",
        "idempotent re-runs of the driver",
        "load-partial-and-skip on restart",
    ],
)
def test_c24_trigger_arm_warns(trigger):
    # One minimal WARN fixture per previously-untested trigger-arm family.
    assert _status(GOOD_PLAN + f"\nPhase B: {trigger}.\n", "c24_resume_provenance") == "WARN"


@pytest.mark.parametrize(
    "satisfier",
    [
        "outputs carry an input fingerprint checked at entry",  # bare fingerprint
        "resume gated on the code SHA of the driver",  # code SHA
        "the entry predicate compares the commit hash recorded at write time",  # commit hash
        "each output carries an input-hash asserted at entry",  # input-hash
        "the resume key records the env knobs of the capture",  # env knobs
        "resume keyed on every output-affecting regime key",  # regime key (#722 r3)
        "the predicate never skips on file-existence alone",  # never-skip (#922 v4)
        "assert the existing file's sampling params match the requested flags",  # #560 v3
    ],
)
def test_c24_satisfier_arm_passes(satisfier):
    plan = GOOD_PLAN + f"\nPhase B: per-cell persist + resume-skip; {satisfier}.\n"
    assert _status(plan, "c24_resume_provenance") == "PASS"


def test_c24_bare_regime_prose_does_not_satisfy():
    # Bare "regime" is excluded — persona-vectors "read-out regime" prose
    # sits inside resume windows (#779 v5) and must not self-satisfy.
    plan = (
        GOOD_PLAN + "\nPhase B: per-cell persist + resume-skip; the A3.3 read-out regime, "
        "all-layer sweep.\n"
    )
    assert _status(plan, "c24_resume_provenance") == "WARN"


def test_c24_equivalence_assert_does_not_satisfy():
    # An equivalence-gate assert without a resume-object token between verb
    # and "match" (the #922 v1-v3 class) must not satisfy the assert-match
    # clause.
    plan = (
        GOOD_PLAN + "\nPhase B: per-cell persist + resume-skip; assert the vmapped MLP path "
        "matches a seeded serial reference.\n"
    )
    assert _status(plan, "c24_resume_provenance") == "WARN"


def test_c24_cli_warn_exit_zero(tmp_path):
    # WARN never blocks: exit 0, overall PASS, c24 rendered WARN (harness
    # identical to test_cli_json_schema_and_exit_zero_on_pass).
    p = tmp_path / "plan.md"
    p.write_text(GOOD_PLAN + C24_V9_SHAPE)
    proc = _run_cli("--plan-file", str(p), "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["overall"] != "FAIL"
    c24 = next(c for c in payload["checks"] if c["id"] == "c24_resume_provenance")
    assert c24["status"] == "WARN"


# ─── Check 25 — HTML entities in fenced command blocks ─────────────────────

# Verbatim un-elided fence body from the #952 v9 amendment-round notification
# (session d49c6e04, main transcript line 993, enqueued 2026-07-05T05:51Z):
# the harness HTML-escaped the <task-notification> <result> field, so both
# shell AND operators of the dispatcher's --workload-cmd arrived as
# amp-entity forms and dispatch would not run until hand-fixed.
C25_952_V9_PLAN = (
    GOOD_PLAN
    + "\n**Exact workload command (dispatcher form):**\n\n```bash\n"
    + r"""uv run python scripts/dispatch_issue.py launch --issue 952 --backend gcp --intent capture-7b \
  --time-budget-hours 16 --repo-branch issue-952 \
  --workload-cmd "bash -o pipefail -ec 'export EPM_I952_LAYER_GRID=14,17,20,23,26 EPM_I952_DECISION_LAYERS=14,23,26 EPM_I952_SKIP_POOLED_PREFIX=1 EPM_I952_FOLLOWUP_TAG=kfold_decision_cells EPM_I952_KFOLD_BLOCKS=5 &amp;&amp; uv run python -m explore_persona_space.experiments.issue_952.run_952 --base-dir /workspace/data/issue_952 --stage-battery-inputs 5b62649cefb34902fd630f21630164e8d1d99764 --phases phase0,battery,bank-score --smoke --synth-capture --skip-upload &amp;&amp; uv run python -m explore_persona_space.experiments.issue_952.run_952 --base-dir /workspace/data/issue_952 --stage-battery-inputs 5b62649cefb34902fd630f21630164e8d1d99764 --phases phase0,battery,bank-score'"
"""
    + "```\n"
)


def test_c25_no_command_fence_skips():
    # GOOD_PLAN carries no fences at all — the conditional trigger does not fire.
    assert _status(GOOD_PLAN, "c25_html_entities_in_commands") == "SKIP"


def test_c25_952_v9_escaped_operator_fails():
    # Regression fixture: the reconstructed #952 v9 escaped dispatcher command
    # FAILs, and the detail names the incident + the capture-side remediation.
    _, by_id = _run(C25_952_V9_PLAN)
    r = by_id["c25_html_entities_in_commands"]
    assert r.status == "FAIL"
    assert "#952" in r.detail
    assert "html.unescape" in r.detail


@pytest.mark.parametrize(
    "entity",
    ["&lt;", "&gt;", "&quot;", "&#x27;", "&#39;", "&#039;", "&#x027;", "&amp;amp;"],
)
def test_c25_entity_variant_fails(entity):
    # lt/gt/quot/apostrophe forms, the leading-zero numeric variants, and the
    # double-escaped amp;amp; form all FAIL inside a bash fence.
    plan = GOOD_PLAN + f"\n```bash\necho {entity}\n```\n"
    assert _status(plan, "c25_html_entities_in_commands") == "FAIL"


@pytest.mark.parametrize("info", ["sh", "shell", "zsh", "console"])
def test_c25_shell_alias_fence_fails(info):
    # Arm-(a) alias coverage: a `bash`-only mutant dies here (`sh` is present
    # in committed plans).
    plan = GOOD_PLAN + f"\n```{info}\necho '&lt;tag&gt;'\n```\n"
    assert _status(plan, "c25_html_entities_in_commands") == "FAIL"


def test_c25_clean_bash_fence_passes():
    # Real shell operators (&&, redirects) are NOT entity forms.
    plan = GOOD_PLAN + "\n```bash\nuv run pytest -x && sort < input.txt > out.txt\n```\n"
    assert _status(plan, "c25_html_entities_in_commands") == "PASS"


def test_c25_prose_entities_do_not_trigger():
    # Entities in PROSE (this fix's own plan class — a plan ABOUT entity
    # handling) never trip the check; only fenced command blocks are scanned.
    plan = (
        GOOD_PLAN
        + "\nThe harness escapes && to &amp;&amp; and < to &lt; in notification results.\n"
        + "\n```bash\necho ok\n```\n"
    )
    assert _status(plan, "c25_html_entities_in_commands") == "PASS"


def test_c25_python_fence_entities_do_not_trigger():
    # A python-tagged fence with entity strings but NO command marker is
    # neither arm; the clean bash fence keeps the check exercised (PASS, not
    # SKIP) — if the python fence were scanned this would FAIL.
    plan = (
        GOOD_PLAN
        + '\n```python\nENTITY_RE = "&amp;|&lt;|&gt;"\n```\n'
        + "\n```bash\necho ok\n```\n"
    )
    assert _status(plan, "c25_html_entities_in_commands") == "PASS"


def test_c25_tagged_fence_with_workload_cmd_marker_scanned():
    # Arm (b): a `text`-tagged fence carrying --workload-cmd is scanned
    # regardless of the fence tag the author picked.
    plan = GOOD_PLAN + "\n```text\n--workload-cmd 'x &amp;&amp; y'\n```\n"
    assert _status(plan, "c25_html_entities_in_commands") == "FAIL"


def test_c25_untagged_fence_with_workload_cmd_marker_fails():
    # S-B (statistics reconciler round 1): pins the "tagged or untagged"
    # arm-(b) promise — a mutant requiring a non-empty info string dies here.
    plan = GOOD_PLAN + "\n```\ndispatch_issue.py launch --workload-cmd 'x &amp;&amp; y'\n```\n"
    assert _status(plan, "c25_html_entities_in_commands") == "FAIL"


def test_c25_na_escape_exempts_arm_a_only():
    # The content-only exemption: an entity inside a plain bash fence (no
    # command marker) + the standalone escape phrase → PASS. (Replaces the
    # v1-sketch masking test the round-1 reconciler rejected.)
    plan = (
        GOOD_PLAN
        + "\n```bash\ngrep -c '&amp;' corpus.txt\n```\n"
        + "\nN/A — entities are content, not commands (the fence greps FOR entity forms).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c25_html_entities_in_commands"]
    assert r.status == "PASS"
    assert "exempt" in r.detail


def test_c25_mixed_fence_escape_does_not_mask_workload_cmd():
    # M1 (methodology reconciler round 1): a document-wide escape phrase must
    # never mask a separately poisoned dispatcher fence — the mixed-plan
    # false-PASS the round-1 reconciler identified.
    plan = (
        GOOD_PLAN
        + "\n```bash\ngrep -c '&amp;' corpus.txt\n```\n"
        + "\nN/A — entities are content, not commands (the first fence greps FOR entity forms).\n"
        + "\n```text\ndispatch_issue.py launch --workload-cmd 'x &amp;&amp; y'\n```\n"
    )
    _, by_id = _run(plan)
    r = by_id["c25_html_entities_in_commands"]
    assert r.status == "FAIL"
    assert "never exemptable" in r.detail


def test_c25_quoted_na_phrase_does_not_escape():
    # A mid-sentence quoted escape phrase (the pasted-bounce-brief
    # self-escape channel) is NOT a standalone declaration line and must not
    # escape — mirrors test_c24_quoted_na_phrase_does_not_escape (same house
    # _standalone_na_declared line discipline).
    plan = (
        GOOD_PLAN
        + "\n```bash\ngrep -c '&amp;' corpus.txt\n```\n"
        + "\nThe remedy menu says to declare 'N/A — entities are content, not commands' on its own line.\n"
    )
    assert _status(plan, "c25_html_entities_in_commands") == "FAIL"


@pytest.mark.parametrize("kind", ["experiment", "analysis", "infra", "batch", "survey"])
def test_c25_fails_for_all_kinds(kind):
    # No kind exemption: infra/batch plans carry verification commands too.
    plan = GOOD_PLAN + "\n```bash\necho '&amp;&amp;'\n```\n"
    assert _status(plan, "c25_html_entities_in_commands", kind=kind) == "FAIL"


def test_c25_cli_fail_exit_one(tmp_path):
    # FAIL blocks: exit 1 in --plan-file mode (mirrors the
    # test_c24_cli_warn_exit_zero harness, opposite polarity).
    p = tmp_path / "plan.md"
    p.write_text(GOOD_PLAN + "\n```bash\necho '&amp;&amp;'\n```\n")
    proc = _run_cli("--plan-file", str(p), "--json")
    assert proc.returncode == 1, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["overall"] == "FAIL"
    c25 = next(c for c in payload["checks"] if c["id"] == "c25_html_entities_in_commands")
    assert c25["status"] == "FAIL"


def test_c25_two_entity_fences_one_declaration_fails_naming_count():
    # #1276: the count-scoped exemption — two DISTINCT entity-bearing bash
    # fences must not both ride one doc-wide declaration; the FAIL detail
    # names the fence count, the scope, and the re-tag remedy.
    plan = (
        GOOD_PLAN
        + "\n```bash\ngrep -c '&amp;' corpus.txt\n```\n"
        + "\n```bash\ngrep -c '&lt;' corpus.txt\n```\n"
        + "\nN/A — entities are content, not commands (both fences grep FOR entity forms).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c25_html_entities_in_commands"]
    assert r.status == "FAIL"
    assert "2 distinct" in r.detail
    assert "EXACTLY ONE" in r.detail
    assert "re-tag" in r.detail


def test_c25_two_fences_same_entity_form_still_fail():
    # #1276 amendment (a): two fences EACH carrying the SAME entity form are
    # still 2 distinct fences — kills a mutant collapsing per-fence hit lists
    # into a set of hit tuples (same-form fences would dedupe to count 1).
    plan = (
        GOOD_PLAN
        + "\n```bash\ngrep -c '&amp;' corpus_a.txt\n```\n"
        + "\n```bash\ngrep -c '&amp;' corpus_b.txt\n```\n"
        + "\nN/A — entities are content, not commands (both fences grep FOR the amp form).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c25_html_entities_in_commands"]
    assert r.status == "FAIL"
    assert "2 distinct" in r.detail


def test_c25_clean_shell_fences_do_not_count_toward_exemption_scope():
    # Clean (entity-free) shell fences never widen the count — kills a mutant
    # counting ALL arm-(a) fences instead of entity-BEARING ones.
    plan = (
        GOOD_PLAN
        + "\n```bash\necho one\n```\n"
        + "\n```bash\necho two\n```\n"
        + "\n```bash\necho three\n```\n"
        + "\n```bash\ngrep -c '&amp;' corpus.txt\n```\n"
        + "\nN/A — entities are content, not commands (one fence greps FOR entity forms).\n"
    )
    assert _status(plan, "c25_html_entities_in_commands") == "PASS"


def test_c25_clean_arm_b_fence_does_not_count_toward_arm_a_scope():
    # A clean (entity-free) --workload-cmd fence neither FAILs arm (b) nor
    # widens the arm-(a) entity-bearing fence count.
    plan = (
        GOOD_PLAN
        + "\n```bash\nuv run python scripts/dispatch_issue.py launch --issue 1 "
        + "--workload-cmd 'echo ok'\n```\n"
        + "\n```bash\ngrep -c '&amp;' corpus.txt\n```\n"
        + "\nN/A — entities are content, not commands (the grep fence discusses entity forms).\n"
    )
    assert _status(plan, "c25_html_entities_in_commands") == "PASS"


def test_c25_multi_fence_no_declaration_fails_with_escape_pointer():
    # With NO declaration the ORDINARY exemptable branch fires regardless of
    # fence count: the escape-phrase remedy pointer is present and the
    # count-scope wording is absent (the count gate only fires when a
    # declaration exists).
    plan = (
        GOOD_PLAN
        + "\n```bash\ngrep -c '&amp;' corpus_a.txt\n```\n"
        + "\n```bash\ngrep -c '&lt;' corpus_b.txt\n```\n"
    )
    _, by_id = _run(plan)
    r = by_id["c25_html_entities_in_commands"]
    assert r.status == "FAIL"
    assert "entities are content, not commands" in r.detail
    assert "distinct shell-tagged" not in r.detail


def test_c25_single_fence_multiple_entity_forms_still_exempt():
    # Grain pin (#1276): the count is per-FENCE, not per-entity-form — one
    # fence carrying three distinct forms is still count 1 and stays exempted
    # (kills a "count distinct entity forms" mutant).
    plan = (
        GOOD_PLAN
        + "\n```bash\ngrep -cE '&amp;|&lt;|&gt;' corpus.txt\n```\n"
        + "\nN/A — entities are content, not commands (the fence greps FOR entity forms).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c25_html_entities_in_commands"]
    assert r.status == "PASS"
    assert "exempt" in r.detail


# ─── Check 26 — GPU basis vs routed machine ─────────────────────────────────

# Fixture rows quoted VERBATIM from the incident corpus (plan #1075 §10
# fixtures of record): tasks/*/1073/plans/v3.md:155 (the founding offender
# row), v3.md:152 (the Spec/intent line, truncated at the sentence break),
# v4.md:157 (the corrected row), v3.md:240 (the prose-form intent line),
# tasks/awaiting_promotion/744/plans/v2.md:676 + :45 (the L4-routing catch),
# tasks/awaiting_promotion/920/plans/v3.md:279 (matching-family control),
# tasks/awaiting_promotion/778/plans/v8.md:73 (runpod pin),
# tasks/on_hold/816/plans/v6.md:3 (frontmatter runpod pin),
# tasks/interpreting/833/plans/v6.md:20 (L3/L4 phase-label collision prose).
C26_HEADER = (
    "\n| component | planned_wall_h | planned_gpu_h | parallelism | basis |\n"
    "|---|---|---|---|---|\n"
)
C26_INTENT_LORA7B = (
    "\n**Spec:** 1 GPU, `--intent lora-7b` (GCP `a2-ultragpu-1g` A100-80 / RunPod 1×H100).\n"
)
C26_INTENT_EVAL = "\n- **Compute:** 1x A100-80 (GCP `auto`, `--intent eval`), single forward-pass\n"
C26_V3_P1_ROW = (
    "| P1 greedy gen (5,000 × ~300 tok) | 0.25 | 0.25 | vLLM batched, chunk 500 "
    "| 1.5 M tok at ≥ 3 k tok/s (H100 Qwen-7B, #779 pass-A convention) |\n"
)
C26_V4_P1_ROW = (  # #1073 v4:157 — corrected row (routed family in wall cell + scaling vocab)
    "| P1 greedy gen (5,000 × ~300 tok) | 0.25 (H100) / 0.5–0.6 (A100, ×2–2.5) | same as wall "
    "| vLLM batched, chunk 500 | 1.5 M tok at ≥ 3 k tok/s H100 basis (#779 pass-A convention); "
    "A100 row = H100 × 2–2.5 stated per-step rate |\n"
)
C26_744_ROW = (
    "| Phase 1 dump+stream, base (NS + broader) | 1.5 | 1.5 | TP=1 batch=1 "
    "| ~1000 broader forwards x 2 passes + 10 NS seqs @ ~5s/1024-tok A100 forward; "
    "most broader docs < 1024 tok |\n"
)
C26_920_ROW = (
    "| P1 set-B generation (GPU, vLLM) | 0.25 | 0.25 | one continuous-batching engine, "
    "2,400 prompts | 2,400 greedy × ≤512 new tokens ≈ ≤1.2M gen tokens; single-A100 vLLM "
    "≥ ~2K tok/s ⇒ ~10 min + engine spin-up |\n"
)
C26_778_PIN_LINE = (
    "\n- **Compute:** 1× H100 RunPod (`backend: runpod`, intent `eval`) for ~2–3 h (6,000\n"
)
# The founding-offender shape: WARN under `--intent lora-7b` (routed A100).
C26_WARN_SHAPE = C26_INTENT_LORA7B + C26_HEADER + C26_V3_P1_ROW


def test_c26_h100_basis_auto_lora7b_warns():
    # Acceptance criterion 2 (plan #1075 §1): the #1073 v3 P1 row under
    # `--intent lora-7b` (auto → A100-80) with no scaling vocabulary and no
    # routed-GPU mention in the row → WARN; the detail names the component
    # cell, the offending token, the routed family, and BOTH remedies.
    ok, by_id = _run(GOOD_PLAN + C26_WARN_SHAPE)
    r = by_id["c26_gpu_basis_routed_machine"]
    assert r.status == "WARN"
    assert ok  # WARN never blocks
    assert "P1 greedy gen" in r.detail
    assert "H100" in r.detail
    assert "A100" in r.detail
    assert "per-step rate" in r.detail  # remedy 1: stated scaling (#599 clause)
    assert "N/A — basis measured on the routed machine" in r.detail  # remedy 2


def test_c26_scaling_vocab_row_passes():
    # Scaling-vocab escape in isolation: the v4-shaped "stated per-step rate"
    # phrase clears the row even with NO routed-family token anywhere in it.
    row = (
        "| P1 greedy gen | 0.5–0.6 | same as wall | vLLM batched, chunk 500 "
        "| 1.5 M tok at ≥ 3 k tok/s H100 basis, × 2–2.5 stated per-step rate to the "
        "routed lane |\n"
    )
    plan = GOOD_PLAN + C26_INTENT_LORA7B + C26_HEADER + row
    assert _status(plan, "c26_gpu_basis_routed_machine") == "PASS"


def test_c26_routed_gpu_also_named_in_row_passes():
    # Routed-family escape in isolation: the #1073 v4 wall-cell shape
    # `0.25 (H100) / 0.5-0.6 (A100, x2-2.5)` names the routed family in a
    # CONVERSION-BEARING cell; no scaling vocabulary anywhere (a bare
    # multiplication sign is not scaling vocab).
    row = (
        "| P1 greedy gen | 0.25 (H100) / 0.5–0.6 (A100, ×2–2.5) | same as wall "
        "| vLLM batched, chunk 500 | 1.5 M tok at ≥ 3 k tok/s (H100 Qwen-7B) |\n"
    )
    plan = GOOD_PLAN + C26_INTENT_LORA7B + C26_HEADER + row
    assert _status(plan, "c26_gpu_basis_routed_machine") == "PASS"


def test_c26_component_times_symbol_is_not_scaling_escape():
    # The ban on the bare-multiplication-sign escape: the verbatim #1073 v3
    # P1 row's component cell carries `5,000 x ~300 tok` multiplier
    # arithmetic — the sign alone must NOT escape (it appears in nearly
    # every row, offending and compliant).
    assert "×" in C26_V3_P1_ROW
    assert _status(GOOD_PLAN + C26_WARN_SHAPE, "c26_gpu_basis_routed_machine") == "WARN"


def test_c26_matching_gpu_basis_passes():
    # #920 v3 shape: A100-measured basis under `--intent capture-7b`
    # (routed A100) — family match, no offender.
    plan = GOOD_PLAN + "\n`--intent capture-7b` on the GCP lane.\n" + C26_HEADER + C26_920_ROW
    assert _status(plan, "c26_gpu_basis_routed_machine") == "PASS"


def test_c26_eval_intent_l4_vs_a100_basis_warns():
    # Acceptance criterion 4: the #744 v2 shape — `--intent eval` routes L4
    # under auto while the basis is an A100-measured forward (the routing
    # later OOM'd; #752 created `capture-7b`).
    plan = GOOD_PLAN + C26_INTENT_EVAL + C26_HEADER + C26_744_ROW
    _, by_id = _run(plan)
    r = by_id["c26_gpu_basis_routed_machine"]
    assert r.status == "WARN"
    assert "A100" in r.detail
    assert "L4" in r.detail


def test_c26_intent_h100_variant_passes():
    # An H100 GCP intent variant routes H100 — an H100 basis matches.
    plan = GOOD_PLAN + "\n`--intent eval-h100` (2× H100).\n" + C26_HEADER + C26_V3_P1_ROW
    assert _status(plan, "c26_gpu_basis_routed_machine") == "PASS"


def test_c26_multi_intent_union():
    # Union semantics across ALL resolved intents: {lora-7b, eval} routes
    # {A100, L4} — an H100 basis still WARNs; an A100 basis PASSes.
    intents = "\nPhase 1: `--intent lora-7b`; phase 2: `--intent eval`.\n"
    warn_plan = GOOD_PLAN + intents + C26_HEADER + C26_V3_P1_ROW
    assert _status(warn_plan, "c26_gpu_basis_routed_machine") == "WARN"
    pass_plan = GOOD_PLAN + intents + C26_HEADER + C26_744_ROW
    assert _status(pass_plan, "c26_gpu_basis_routed_machine") == "PASS"


def test_c26_runpod_pin_skips():
    # The #778 v8:73 pin shape: under `backend: runpod` the RunPod H100/H200
    # intent table governs and an H100 basis is correct → SKIP.
    plan = GOOD_PLAN + C26_778_PIN_LINE + C26_HEADER + C26_V3_P1_ROW
    _, by_id = _run(plan)
    r = by_id["c26_gpu_basis_routed_machine"]
    assert r.status == "SKIP"
    assert "runpod" in r.detail


def test_c26_no_intent_skips():
    # A basis-bearing table with NO resolvable intent token anywhere: the
    # auto-lane GPU cannot be inferred from text → SKIP, never a guess.
    plan = GOOD_PLAN + C26_HEADER + C26_V3_P1_ROW
    _, by_id = _run(plan)
    r = by_id["c26_gpu_basis_routed_machine"]
    assert r.status == "SKIP"
    assert "--intent" in r.detail


def test_c26_no_compute_table_skips():
    # An intent with no basis-header compute table → SKIP (trigger absent).
    assert _status(GOOD_PLAN + C26_INTENT_LORA7B, "c26_gpu_basis_routed_machine") == "SKIP"


@pytest.mark.parametrize("kind", ["infra", "batch", "survey"])
def test_c26_kind_exempt_skips(kind):
    assert _status(GOOD_PLAN + C26_WARN_SHAPE, "c26_gpu_basis_routed_machine", kind=kind) == "SKIP"


def test_c26_kind_analysis_warns():
    # analysis is IN scope (compute-projection tables are an
    # experiment|analysis plan shape) — same WARN severity as experiment.
    assert (
        _status(GOOD_PLAN + C26_WARN_SHAPE, "c26_gpu_basis_routed_machine", kind="analysis")
        == "WARN"
    )


def test_c26_na_escape_passes():
    plan = (
        GOOD_PLAN
        + C26_WARN_SHAPE
        + "\nN/A — basis measured on the routed machine (the H100 token is provenance "
        "prose, not the measurement machine).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c26_gpu_basis_routed_machine"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


def test_c26_quoted_na_phrase_does_not_escape():
    # A mid-sentence quoted escape phrase (the pasted-bounce-brief
    # self-escape channel) is NOT a standalone declaration line and must not
    # escape — the c13/c18/c24/c25 anti-paste twin.
    plan = (
        GOOD_PLAN
        + C26_WARN_SHAPE
        + "\nThe remedy menu says to declare 'N/A — basis measured on the routed machine' "
        "on its own line.\n"
    )
    assert _status(plan, "c26_gpu_basis_routed_machine") == "WARN"


def test_c26_l4_phase_label_not_gpu_token():
    # The #833 v6 collision: `L1/L2 re-extraction, L3/L4 extraction` leg
    # labels share the L4 GPU token — L4 is deliberately EXCLUDED from the
    # trigger set, so a basis naming only phase labels never fires.
    row = (
        "| L3/L4 extraction of R⁺ | 8.0 | 8.0 | vLLM 0.11.0 multi-LoRA greedy "
        "| same-era L3/L4 re-extraction of R_base′, measured #667 precedent |\n"
    )
    plan = GOOD_PLAN + C26_INTENT_LORA7B + C26_HEADER + row
    assert _status(plan, "c26_gpu_basis_routed_machine") == "PASS"


def test_c26_fenced_table_does_not_trigger():
    # A fenced example table is not the plan's compute table (fence-masked
    # header detection, the c24 fenced-trigger twin) → SKIP.
    plan = (
        GOOD_PLAN
        + C26_INTENT_LORA7B
        + "\n```\n"
        + C26_HEADER.strip("\n")
        + "\n"
        + C26_V3_P1_ROW
        + "```\n"
    )
    assert _status(plan, "c26_gpu_basis_routed_machine") == "SKIP"


def test_c26_parallelism_cell_routed_gpu_does_not_escape():
    # Must-Fix M1 fixture (#810 v18 / #923 v9 house style): a parallelism
    # cell truthfully naming the PROVISIONED machine (`1x A100-80`) carries
    # zero conversion information — it must NOT escape an unscaled H100
    # basis (a whole-row escape would wrongly PASS this shape).
    row = "| P2 extraction | 0.6 | 0.6 | 1× A100-80 batch-8 forwards | H100 #779 measured |\n"
    plan = GOOD_PLAN + C26_INTENT_LORA7B + C26_HEADER + row
    assert _status(plan, "c26_gpu_basis_routed_machine") == "WARN"


def test_c26_annotated_basis_header_variant_parses():
    # The #952 v12 header variant `basis (measured)` — surfaced by the §6.2
    # corpus calibration as a realized SKIP-no-table on a predicted-parseable
    # file — must be recognized as a basis column (a header cell that IS or
    # BEGINS WITH the word "basis").
    header = (
        "\n| component | planned_wall_h | planned_gpu_h | parallelism | basis (measured) |\n"
        "|---|---|---|---|---|\n"
    )
    plan = GOOD_PLAN + C26_INTENT_LORA7B + header + C26_V3_P1_ROW
    assert _status(plan, "c26_gpu_basis_routed_machine") == "WARN"


def test_c26_bold_total_short_row_no_crash():
    # A bold `**Base total**` row carrying fewer cells than the header must
    # not IndexError at row[basis_col]; the sibling offender row is still
    # evaluated.
    table = C26_HEADER + C26_V3_P1_ROW + "| **Base total** | 4.2 |\n"
    plan = GOOD_PLAN + C26_INTENT_LORA7B + table
    assert _status(plan, "c26_gpu_basis_routed_machine") == "WARN"


def test_c26_escape_regex_covers_all_mirror_families():
    # Companion drift assert: every non-CPU family the mirror can route must
    # be matchable by the escape alternation — a future routed family absent
    # from _C26_ROW_GPU_ANY_RE would make compliant rows unescapable
    # (systematic false positives).
    for family in set(verify_plan._C26_INTENT_GPU.values()):
        if family == "CPU":
            continue
        m = verify_plan._C26_ROW_GPU_ANY_RE.search(family)
        assert m, f"escape regex misses mirror family {family!r}"
        assert verify_plan._c26_family(m.group(1)) == family


def test_c26_prose_intent_form_resolves():
    # The #1073 v3:240 "Target pod preference" shape: the ONLY intent mention
    # is the capitalized prose form ``Intent `lora-7b` `` (group 2 of
    # _C26_INTENT_RE) — resolution must still work (non-SKIP).
    plan = (
        GOOD_PLAN
        + "\nIntent `lora-7b` (1× A100-80 GCP / 1× H100 RunPod); explicitly NOT the L4 lane.\n"
        + C26_HEADER
        + C26_V3_P1_ROW
    )
    assert _status(plan, "c26_gpu_basis_routed_machine") == "WARN"


def test_c26_fenced_backend_runpod_pin_skips():
    # The pin regex scans RAW (fences included): a plan whose ONLY runpod pin
    # is a fenced `--backend runpod` dispatch line is really pinned → SKIP
    # (closes the fence-asymmetry FP mode; permissive direction only).
    plan = (
        GOOD_PLAN + "\n```bash\nuv run python scripts/dispatch_issue.py launch --issue 999 "
        "--intent lora-7b --backend runpod\n```\n" + C26_HEADER + C26_V3_P1_ROW
    )
    assert _status(plan, "c26_gpu_basis_routed_machine") == "SKIP"


def test_c26_intent_gpu_mirror_matches_backend():
    # Drift guard (acceptance criterion 6): the static family-grain mirror
    # equals the live GCP INTENT_TO_MACHINE — an intent add/change on the
    # backend fails the full suite loudly (precedent:
    # test_kind_enum_constants_match_canonical_code_kinds).
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from explore_persona_space.backends.gcp import INTENT_TO_MACHINE

    derived = {k: verify_plan._c26_family(v.gpu_kind) for k, v in INTENT_TO_MACHINE.items()}
    assert derived == verify_plan._C26_INTENT_GPU


# ─── Check 27 — 7B activation capture vs eval/debug (L4) intent ────────────

# Fixture shapes mirror the §4-calibrated corpus offenders (task #1093 plan):
# tasks/followups_running/825/plans/v17.md (the founding false negative —
# fenced dispatch line booking `--intent eval` for a 7B all-layer capture),
# tasks/awaiting_promotion/744/plans/v2.md (the L4-OOM incident plan; its
# "1x A100-80 (GCP auto, --intent eval)" line is a false belief — GCP eval
# routes L4), and the named near-miss FPs the keying clears (#375/#358
# pod.py provision era, #522 wrapped --intent, #358 H100 prose).
C27_CAPTURE_BLOCK = (
    "\n### Phase 2 — representation read\n\n"
    "All-layer hidden-state capture over the context grid (response-avg), then "
    "upload the activation store to the data repo before teardown.\n"
)
C27_INTENT_EVAL = "\n`--intent eval` on the GCP auto lane.\n"
C27_INTENT_CAPTURE7B = "\n`--intent capture-7b` on the GCP lane.\n"
C27_DISPATCH_FENCE = (  # the #825 v17 shape: the launch command lives in a fence
    "\n```bash\nuv run python scripts/dispatch_issue.py launch --issue 825 "
    "--intent eval --workload-cmd 'bash scripts/issue825_capture.sh'\n```\n"
)
# The founding-offender shape: FAIL under kind=experiment.
C27_OFFENDER = GOOD_PLAN + C27_CAPTURE_BLOCK + C27_DISPATCH_FENCE


def test_c27_825_v17_shape_fails():
    # Acceptance criterion 1: capture vocabulary + a fenced dispatch line
    # booking `--intent eval` (the #825 v17 founding false negative) → FAIL;
    # the detail names the intent token, the L4 machine class, the
    # capture-7b remedy, and the exact N/A escape phrase.
    ok, by_id = _run(C27_OFFENDER)
    r = by_id["c27_capture_intent_hbm"]
    assert r.status == "FAIL"
    assert not ok
    assert "eval" in r.detail
    assert "L4" in r.detail
    assert "capture-7b" in r.detail
    assert "N/A — no 7B activation capture" in r.detail


def test_c27_744_a100_claim_still_fails():
    # Acceptance criterion 2: the #744 v2 misbelief — "1x A100-80 (GCP
    # `auto`, `--intent eval`)" (the C26_INTENT_EVAL fixture IS that line
    # verbatim). GCP eval NEVER provisions A100, so an A100 claim in the
    # window must NOT skip (A100 is deliberately not a window-skip token).
    plan = GOOD_PLAN + C27_CAPTURE_BLOCK + C26_INTENT_EVAL
    assert _status(plan, "c27_capture_intent_hbm") == "FAIL"


def test_c27_no_capture_vocab_skips():
    assert _status(GOOD_PLAN + C27_INTENT_EVAL, "c27_capture_intent_hbm") == "SKIP"


def test_c27_capture7b_intent_passes():
    # Capture booked on the right intent, no L4 intent anywhere → PASS.
    plan = GOOD_PLAN + C27_CAPTURE_BLOCK + C27_INTENT_CAPTURE7B
    _, by_id = _run(plan)
    r = by_id["c27_capture_intent_hbm"]
    assert r.status == "PASS"
    assert "no eval/debug intent" in r.detail


def test_c27_mixed_intents_big_absolution_passes():
    # Both `--intent capture-7b` and `--intent eval` booked: the capture
    # phase is presumed routed to the big intent (documented gap (a);
    # phase-to-intent routing stays critic-owned).
    plan = GOOD_PLAN + C27_CAPTURE_BLOCK + C27_INTENT_CAPTURE7B + C27_INTENT_EVAL
    _, by_id = _run(plan)
    r = by_id["c27_capture_intent_hbm"]
    assert r.status == "PASS"
    assert "capture-7b" in r.detail
    assert "critic-owned" in r.detail


def test_c27_runpod_backend_pin_skips():
    # Under a `backend: runpod` pin the RunPod intent table governs
    # (eval = 1x H100 80GB) — no HBM gap → SKIP.
    plan = (
        GOOD_PLAN + C27_CAPTURE_BLOCK + C27_INTENT_EVAL + "\nbackend: runpod (explicit override)\n"
    )
    _, by_id = _run(plan)
    r = by_id["c27_capture_intent_hbm"]
    assert r.status == "SKIP"
    assert "RunPod" in r.detail


def test_c27_podpy_provision_skips():
    # The #375/#358 pre-router corpus class: `pod.py provision ... --intent
    # eval` provisions 1x H100 80GB on RunPod → doc-wide SKIP.
    plan = (
        GOOD_PLAN
        + C27_CAPTURE_BLOCK
        + "\nuv run python scripts/pod.py provision --issue 375 --intent eval\n"
    )
    assert _status(plan, "c27_capture_intent_hbm") == "SKIP"


def test_c27_wrapped_podpy_provision_window_skips():
    # The #522 v1 shape: `--intent[=\s]+` legitimately spans the newline in
    # a wrapped `pod.py provision` line — the WINDOW path (previous line +
    # match-end line) must clear it even without the doc-wide pod.py skip
    # (unit-level per plan §5 test 8: assert the helper directly).
    wrapped = (
        "some prose line\n"
        "uv run python scripts/pod.py provision --issue 522 --intent\n"
        "eval\n"
        "more prose\n"
    )
    assert verify_plan._c27_gcp_l4_intent_windows(wrapped) == []


def test_c27_h100_prose_window_skips():
    # The #358 prose shape: "1x H100 SXM (intent `eval`)" — an H100 token in
    # the window is a RunPod-mapping claim, not a GCP L4 booking → the
    # occurrence is window-skipped; with no other eval/debug booking → PASS.
    plan = GOOD_PLAN + C27_CAPTURE_BLOCK + "\nGPU: 1× H100 SXM (intent `eval`).\n"
    _, by_id = _run(plan)
    r = by_id["c27_capture_intent_hbm"]
    assert r.status == "PASS"
    assert "no eval/debug intent" in r.detail


def test_c27_na_escape_passes():
    plan = (
        GOOD_PLAN
        + C27_CAPTURE_BLOCK
        + C27_INTENT_EVAL
        + "\nN/A — no 7B activation capture (the store is a reused parent artifact "
        "consumed by a CPU phase).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c27_capture_intent_hbm"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


def test_c27_quoted_na_phrase_does_not_escape():
    # A mid-sentence quoted escape phrase (the pasted-bounce-brief
    # self-escape channel) is NOT a standalone declaration line and must not
    # escape — the c13/c18/c24/c25/c26 anti-paste twin.
    plan = (
        GOOD_PLAN
        + C27_CAPTURE_BLOCK
        + C27_INTENT_EVAL
        + "\nThe remedy menu says to declare 'N/A — no 7B activation capture' on its own line.\n"
    )
    assert _status(plan, "c27_capture_intent_hbm") == "FAIL"


def test_c27_small_model_skips():
    # 0.5B never matches the >=7B regex (also pins the decimal-tail
    # lookbehind end-to-end; the unit matrix is
    # test_c27_model_size_threshold_semantics).
    plan = GOOD_PLAN.replace("7B", "0.5B") + C27_CAPTURE_BLOCK + C27_INTENT_EVAL
    _, by_id = _run(plan)
    r = by_id["c27_capture_intent_hbm"]
    assert r.status == "SKIP"
    assert "7B" in r.detail


@pytest.mark.parametrize(
    ("text", "matches"),
    [
        ("7B", True),
        ("Qwen-2.5-7B", True),
        ("7.5B", True),
        ("Llama-3.1-8B", True),
        ("Gemma-2-9B", True),
        ("12B", True),
        ("17B", True),  # >=7 under threshold semantics (NOT the old whitelist)
        ("27B", True),
        ("34B", True),
        ("70B", True),
        ("72B", True),
        ("0.5B", False),
        ("1.7B", False),  # decimal-tail lookbehind
        ("2.5B", False),
        ("6.9B", False),
    ],
)
def test_c27_model_size_threshold_semantics(text, matches):
    # Threshold (>=7B), not a whitelist (critique r1, all three Codex
    # lenses): integer part >= 7 — single digit 7-9 or any 2+ digit number —
    # with an optional decimal tail; the (?<![\d.]) lookbehind blocks
    # decimal-tail false positives ("1.7B" must never read as "7B").
    assert bool(verify_plan._C27_MODEL_GE7B_RE.search(text)) is matches


def test_c27_no_intent_skips():
    plan = GOOD_PLAN + C27_CAPTURE_BLOCK
    _, by_id = _run(plan)
    r = by_id["c27_capture_intent_hbm"]
    assert r.status == "SKIP"
    assert "--intent" in r.detail


@pytest.mark.parametrize("kind", ["infra", "batch", "survey"])
def test_c27_kind_exempt_skips(kind):
    assert _status(C27_OFFENDER, "c27_capture_intent_hbm", kind=kind) == "SKIP"


def test_c27_kind_analysis_warns():
    # analysis is IN scope but softens to WARN (the c12/c13/c18/c20
    # severity shape) — WARN never blocks.
    ok, by_id = _run(C27_OFFENDER, kind="analysis")
    r = by_id["c27_capture_intent_hbm"]
    assert r.status == "WARN"
    assert ok


def test_c27_sets_derive_from_mirror():
    # The offending + absolving sets are DERIVED from the c26 mirror (no
    # second copy to drift; the mirror itself is drift-guarded by
    # test_c26_intent_gpu_mirror_matches_backend). The partition assert
    # (critique r1, methodology concern 1) forces a future mirror family to
    # be deliberately classified: L4 | BIG | CPU == the whole mirror.
    assert {"eval", "debug"} == verify_plan._C27_L4_INTENTS
    assert {"capture-7b", "lora-7b", "ft-7b", "eval-h100"} <= verify_plan._C27_BIG_HBM_INTENTS
    assert "cpu-mid" not in verify_plan._C27_BIG_HBM_INTENTS
    cpu = {i for i, f in verify_plan._C26_INTENT_GPU.items() if f == "CPU"}
    assert verify_plan._C27_L4_INTENTS | verify_plan._C27_BIG_HBM_INTENTS | cpu == set(
        verify_plan._C26_INTENT_GPU
    )


def test_c27_detail_does_not_self_retrigger():
    # Self-DISARM safety, both directions (critique r1): the FAIL detail
    # carries no H100/H200 token and no runpod-pin form (a pasted detail
    # must never fire the window skip / RunPod pin and absolve an UN-fixed
    # booking), and pasting it into a FIXED plan (capture booked on
    # capture-7b) keeps the fixed plan PASSing.
    _, by_id = _run(C27_OFFENDER)
    detail = by_id["c27_capture_intent_hbm"].detail
    assert not re.search(r"\b(H100|H200)\b", detail)
    assert "backend: runpod" not in detail
    assert "--backend runpod" not in detail
    fixed = GOOD_PLAN + C27_CAPTURE_BLOCK + C27_INTENT_CAPTURE7B
    pasted = fixed + "\n> Bounce brief (verbatim): " + detail + "\n"
    assert _status(pasted, "c27_capture_intent_hbm") == "PASS"


def test_c27_pasted_detail_arms_vocab_on_nocapture_plan():
    # The S4 residual cell (critique r1, statistics — documented, accepted):
    # the detail necessarily names hidden-state capture (it states the
    # condition), so pasting it into a NO-capture plan with a legitimate
    # eval booking arms the vocab trigger → FAIL; the documented out is the
    # 1-line N/A escape.
    _, by_id = _run(C27_OFFENDER)
    detail = by_id["c27_capture_intent_hbm"].detail
    nocapture = GOOD_PLAN + C27_INTENT_EVAL + "\n> Bounce brief (verbatim): " + detail + "\n"
    assert _status(nocapture, "c27_capture_intent_hbm") == "FAIL"
    escaped = nocapture + "\nN/A — no 7B activation capture (no capture phase in this plan).\n"
    assert _status(escaped, "c27_capture_intent_hbm") == "PASS"


@pytest.mark.parametrize(
    ("text", "matches"),
    [
        ("extract_store", True),
        ("residual stream", True),
        ("residual-stream", True),
        ("activation store", True),
        ("activations extracted", True),
        ("activation accumulation", True),
        ("per-token activation dumps", True),
        ("capturing activations", True),
        ("extraction set", False),
        ("capture the behavior", False),
        ("feature extraction", False),
    ],
)
def test_c27_vocab_and_skip_arm_matrix(text, matches):
    # Critique r1, statistics S2: pin every vocab regex arm a corpus
    # fixture doesn't already exercise (anchored compounds fire; bare
    # "extraction"/"capture" prose never does).
    assert bool(verify_plan._C27_CAPTURE_RE.search(text)) is matches


def test_c27_skip_arm_regexes():
    # Companion skip-arm pins: the window big-GPU skip covers H200; the
    # RunPod pin covers the --backend flag form (the frontmatter form is
    # exercised e2e by test_c27_runpod_backend_pin_skips).
    assert verify_plan._C27_WINDOW_BIGGPU_RE.search("2x H200 SXM")
    assert verify_plan._C26_RUNPOD_PIN_RE.search("--backend runpod")


# ─── Check 28 — decision-band precedent coherence ──────────────────────────

# Verbatim corpus literals from tasks/followups_running/825/plans/v17.md
# (lines 21, 103-104; line 103's trailing parenthetical trimmed) — embedded
# as literals, NEVER read from tasks/ at test runtime (tasks move between
# status folders; the REAL plan files are exercised by the plan-#1094 §5
# pre-commit corpus calibration, the c14 fixture convention).
C28_HEADING = "## 6. Success + kill criteria (quantitative)"
C28_BANDS = (
    C28_HEADING
    + "\n\n"
    + "- **Answer-specificity UPHELD on base:** max(base `armC_sep` rotated, MLP) @ L19 "
    "< 0.5 × 0.588 = **0.294**, AND base full-n sep→chat recentered transfer < 0.5 × the "
    "matched-n base chat ceiling.\n"
    + "- **Headline REFRAMED (generic delimiter-span predictability):** either quantity "
    "≥ 0.5× its ceiling — the base-arm headline gains the caveat that a persona-free "
    "separator map reaches ≥ half the chat-map strength.\n"
)
# v17 line 21 verbatim — the incident offender: the quoted range 0.44-0.52
# straddles the plan's own 0.5x threshold while the prose asserts 'lands
# well below'; the same-line vs-pair recompute (0.349/0.673 ≈ 0.519,
# 0.299/0.673 ≈ 0.444) corroborates.
C28_825_LINE = (
    "**Hypothesis (falsifiable):** mirroring instruct — where separator within-strength "
    "was rotated +0.349 / MLP +0.299 vs chat 0.673 (ratio ≈ 0.44–0.52) — the base "
    "separator map lands well below the base chat map (0.588). Quantitative bands in §6."
)


def _c28_plan(*extra: str) -> str:
    """GOOD_PLAN + the v17 band bullets + ``extra`` lines appended."""
    return GOOD_PLAN + "\n" + C28_BANDS + "\n" + "\n\n".join(extra) + ("\n" if extra else "")


@pytest.mark.parametrize("kind", ["infra", "batch", "survey"])
def test_c28_kind_infra_skips(kind):
    assert _status(_c28_plan(C28_825_LINE), "c28_precedent_band_coherence", kind=kind) == "SKIP"


def test_c28_no_band_skips():
    _, by_id = _run(GOOD_PLAN)
    r = by_id["c28_precedent_band_coherence"]
    assert r.status == "SKIP"
    assert "no registered multiplicative decision band" in r.detail


def test_c28_825_v17_straddle_warns():
    # The incident reduction: verbatim v17 bands + line 21 → WARN (never
    # FAIL — `ok` stays True, the c14 severity doctrine).
    ok, by_id = _run(_c28_plan(C28_825_LINE))
    r = by_id["c28_precedent_band_coherence"]
    assert r.status == "WARN"
    assert ok is True  # WARN never flips exit 0
    for token in ("0.52", "0.5", "well below", "straddle"):
        assert token in r.detail, (token, r.detail)


def test_c28_825_recompute_in_detail():
    # The vs-pair recompute ran (not just the quoted range): 0.349/0.673 =
    # 0.5186... rendered `≈ 0.519` in the detail.
    _, by_id = _run(_c28_plan(C28_825_LINE))
    assert "0.519" in by_id["c28_precedent_band_coherence"].detail


def test_c28_coherent_below_passes():
    line = "The instruct precedent (ratio ≈ 0.23) — lands well below the ceiling."
    assert _status(_c28_plan(line), "c28_precedent_band_coherence") == "PASS"


def test_c28_coherent_above_passes():
    line = "The instruct precedent (ratio ≈ 0.62) — exceeds half the ceiling."
    assert _status(_c28_plan(line), "c28_precedent_band_coherence") == "PASS"


def test_c28_clean_contradiction_warns():
    line = "The instruct precedent (ratio ≈ 0.62) — lands well below the ceiling."
    _, by_id = _run(_c28_plan(line))
    r = by_id["c28_precedent_band_coherence"]
    assert r.status == "WARN"
    assert "contradiction" in r.detail


def test_c28_vs_pair_corroborator_catches_typoed_ratio():
    # The quoted ratio (0.48 < 0.5) is coherent alone; the same-line vs-pair
    # recompute (0.35/0.67 = 0.522 ≥ 0.5) corroborates a straddle WARN
    # despite the (typoed) quoted value.
    line = (
        "The instruct precedent (ratio ≈ 0.48) rotated 0.35 vs chat 0.67 — "
        "lands well below the ceiling."
    )
    _, by_id = _run(_c28_plan(line))
    r = by_id["c28_precedent_band_coherence"]
    assert r.status == "WARN"
    assert "0.522" in r.detail


def test_c28_na_escape_passes():
    plan = _c28_plan(C28_825_LINE) + "\n- N/A — no precedent-labeled decision bands\n"
    _, by_id = _run(plan)
    r = by_id["c28_precedent_band_coherence"]
    assert r.status == "PASS"
    assert "explicit N/A" in r.detail


def test_c28_quoted_na_phrase_does_not_escape():
    # A mid-sentence quoted escape phrase (the pasted-bounce-brief
    # self-escape channel) is NOT a standalone declaration line and must not
    # escape — the `_standalone_na_declared` anti-paste convention,
    # mirroring test_c13_quoted_na_phrase_does_not_escape.
    plan = _c28_plan(C28_825_LINE) + (
        "\nThe verifier's remediation menu says to declare 'N/A — no precedent-labeled "
        "decision bands' when the mention is incidental.\n"
    )
    assert _status(plan, "c28_precedent_band_coherence") == "WARN"


def test_c28_multiple_distinct_thresholds_skips():
    extra_band = "- **Tertiary floor:** the control read < 0.1 × its own ceiling."
    plan = GOOD_PLAN + "\n" + C28_BANDS + extra_band + "\n\n" + C28_825_LINE + "\n"
    _, by_id = _run(plan)
    r = by_id["c28_precedent_band_coherence"]
    assert r.status == "SKIP"
    assert "pairing ambiguous" in r.detail


def test_c28_mix_ratio_idiom_not_harvested():
    # The `ratio ~1:1` contrastive-negatives mix idiom carries no decimal
    # point → not a precedent-ratio token, even with side vocabulary on the
    # line (the two non-incident same-line corpus hits, #524 v1).
    line = (
        "Training mix: contrastive negatives @ ratio ~1:1 across the panel; "
        "the anchor sits below the ceiling by 5 nats."
    )
    _, by_id = _run(_c28_plan(line))
    r = by_id["c28_precedent_band_coherence"]
    assert r.status == "SKIP"
    assert "no side-asserted precedent ratio line" in r.detail


def test_c28_percent_range_not_harvested():
    # Round-2 fix (concern c28-percent-range-partial-harvest): a %-suffixed
    # RANGE harvests NOTHING. Pre-fix, `(?!\s*%)` after the OPTIONAL range
    # group let the engine skip the group and partially harvest r1=0.44 from
    # `ratio ≈ 0.44–0.52%` (live-probed: ('0.44', None)).  # noqa: RUF003
    line = "The instruct precedent (ratio ≈ 0.44–0.52%) — lands well below the ceiling."
    _, by_id = _run(_c28_plan(line))
    r = by_id["c28_precedent_band_coherence"]
    assert r.status == "SKIP"
    assert "no side-asserted precedent ratio line" in r.detail
    # Regex-level: neither endpoint of a %-suffixed range is harvested.
    assert verify_plan._C28_RATIO_RE.search("ratio ≈ 0.44–0.52%") is None
    assert verify_plan._C28_RATIO_RE.search("ratio ≈ 0.44-0.52%") is None
    assert verify_plan._C28_RATIO_RE.search("ratio ≈ 0.44 – 0.52 %") is None


def test_c28_percent_single_value_not_harvested():
    # The pre-existing single-value `%` exclusion, now pinned: the \b blocks
    # a backtracked partial-digit match (`0.4` inside `0.48%`).
    line = "The instruct precedent (ratio ≈ 0.48%) — lands well below the ceiling."
    _, by_id = _run(_c28_plan(line))
    r = by_id["c28_precedent_band_coherence"]
    assert r.status == "SKIP"
    assert "no side-asserted precedent ratio line" in r.detail
    assert verify_plan._C28_RATIO_RE.search("ratio ≈ 0.48%") is None


def test_c28_negated_side_passes():
    # Negation guard: a negated side phrase kills the line (line-level).
    line = "The instruct precedent (ratio ≈ 0.52) is NOT well below the ceiling."
    assert _status(_c28_plan(line), "c28_precedent_band_coherence") in ("SKIP", "PASS")


def test_c28_straddle_narrated_honestly_passes():
    # A range straddling T with NO side assertion is honest narration — no
    # side vocabulary, no assertion harvested.
    line = (
        "The instruct precedent (ratio ≈ 0.44–0.52), straddling the threshold — "
        "borderline either way."
    )
    assert _status(_c28_plan(line), "c28_precedent_band_coherence") in ("SKIP", "PASS")


def test_c28_fenced_band_does_not_trigger():
    plan = (
        GOOD_PLAN + "\n" + C28_HEADING + "\n\n```\n"
        "- **Answer-specificity UPHELD on base:** max < 0.5 × 0.588 ceiling\n"
        "```\n\n" + C28_825_LINE + "\n"
    )
    _, by_id = _run(plan)
    r = by_id["c28_precedent_band_coherence"]
    assert r.status == "SKIP"
    assert "no registered multiplicative decision band" in r.detail


def test_c28_kind_analysis_warns_not_skips():
    status = _status(_c28_plan(C28_825_LINE), "c28_precedent_band_coherence", kind="analysis")
    assert status == "WARN"


def test_c28_malformed_prose_no_crash():
    # (a) band line with a comparator + mult sign but no number — the band
    # regex cannot match; SKIP, no exception.
    plan_a = GOOD_PLAN + "\n" + C28_HEADING + "\n\n- **Band:** metric < × ceiling\n"
    assert _status(plan_a, "c28_precedent_band_coherence") == "SKIP"
    # (b) `ratio ≈ .` garbage — the ratio token needs a digit; SKIP.
    plan_b = _c28_plan("The precedent (ratio ≈ .) lands well below the ceiling.")
    assert _status(plan_b, "c28_precedent_band_coherence") == "SKIP"
    # (c) zero-denominator vs-pair on a line that ACTUALLY fires (the full
    # conjunction: bands + `ratio ≈ 0.48` + `0.00 vs 0.00` + side vocab), so
    # `_c28_ratio_assertions` reaches the b > 0 denominator guard — no
    # ZeroDivisionError (the Fraction(x, 0) class c13's detail builder
    # documents), and the quoted candidate still yields a defined verdict
    # (0.48 < 0.5 asserted below → coherent).
    plan_c = _c28_plan(
        "The precedent (ratio ≈ 0.48) scored 0.00 vs 0.00 on the control — "
        "lands well below the ceiling."
    )
    _, by_id = _run(plan_c)
    r = by_id["c28_precedent_band_coherence"]
    assert r.status in ("PASS", "WARN", "SKIP")
    assert r.detail
    # (d) ~2 MB repeated-line plan — bounded wall time, no exception (direct
    # check-function call, the c13 alpha-zero-test precedent).
    big = _c28_plan(C28_825_LINE) + ("filler prose line with no tokens of interest\n" * 45000)
    r_big = verify_plan.check_precedent_band_coherence(big, "experiment")
    assert r_big.status in ("PASS", "WARN", "SKIP")


def test_c28_ratio_equal_threshold():
    # Boundary pin: below := [0, T) hardcoded — r == T on an asserted-below
    # line is a contradiction. The harvested band comparator is NOT
    # consulted at the boundary (a `<=`-band's r == T edge is an accepted
    # WARN-only imprecision, per the check docstring).
    line = "The instruct precedent (ratio ≈ 0.5) — lands well below the ceiling."
    _, by_id = _run(_c28_plan(line))
    r = by_id["c28_precedent_band_coherence"]
    assert r.status == "WARN"
    assert "contradiction" in r.detail


# ─── Check 29 — deliberate fence vs §7 conditional phase ────────────────────

# Fixtures quoted VERBATIM from the incident corpus (task #1114 plan §4.2),
# embedded as literals — NEVER read from tasks/ at test runtime (tasks move
# between status folders; the REAL plan files are exercised by the #1114 §6
# real-fixture spot check + sibling replay, the c26/c28 fixture convention):
# tasks/*/1112/plans/v2.md:207 (the G1 dose-extension gate bullet, truncated
# before the "Grounding: #606" tail), v2.md:229 (the founding-offender fence
# declaration, truncated at "never the mean" — hand-verified free of
# _C29_EVIDENCE_RE tokens), v3.md:238 (the corrected declaration, truncated
# at "rounded to **72 h**" — carries "§7", "G1", "contingency", "extension").
C29_GATE_SECTION = (
    "\n## 7. Decision Gates\n\n"
    "- **G1 (FT install viability, fires after the sycophancy Tier-1 judge fold):** if "
    "NEITHER full-FT cell has any rung with Tier-1 rate ≥ 0.45, run the pre-registered "
    "one-shot dose extension — resume/retrain both FT cells to step 60 (save-2 grid "
    "32..60), re-ladder, re-judge — before the capture phase.\n"
)
C29_V2_FENCE = (
    "`--max-run-duration 48h` (fence re-reconciled for the cross-GPU basis: the FT-train "
    "wall bases were measured on 4× H100 (#514) but this lane lands on 4× A100-80, expected "
    "×2–3 per-step for ZeRO-3 7B, worst-case ×6 (#599 precedent); expected pod wall "
    "~13–16 h; scaled WORST case (×6 on both FT-train rows) → pod wall ≈ 28 h, + judge-wait "
    "variance ≈ 30 h, ×~1.5 margin → 45 h, rounded to 48 h — sized off the scaled worst "
    "case, never the mean)\n"
)
C29_V3_FENCE = (
    "`--max-run-duration 72h` (fence re-reconciled for the cross-GPU basis INCLUDING the "
    "§7 G1 contingency: the FT-train wall bases were measured on 4× H100 (#514) but this "
    "lane lands on 4× A100-80, expected ×2–3 per-step for ZeRO-3 7B, worst-case ×6 (#599 "
    "precedent); expected pod wall ~13–16 h; scaled WORST case (×6 on both FT-train rows) "
    "→ pod wall ≈ 28 h, + judge-wait variance ≈ 30 h; the §7 G1 one-shot dose extension "
    "(resume/retrain BOTH FT cells to step 60 + re-ladder + re-judge on the SAME provision, "
    "before capture) adds another FT-train-sized pass — ≈ 18–20 h at the same ×6 worst "
    "scaling → joint worst ≈ 48–50 h, ×~1.45 margin → ~70–72 h, rounded to **72 h**\n"
)
# Separator DELIBERATELY padded to >= 4 raw lines so the gate section's
# extension vocabulary sits OUTSIDE the ±3-raw-line evidence window of the
# fence declaration line (one blank line fewer would flip
# test_c29_1112_v2_offender_warns loud — padded by design).
C29_SEPARATOR = "\n## 9b. Backend\n\n\n\n"
C29_OFFENDER = GOOD_PLAN + C29_GATE_SECTION + C29_SEPARATOR + C29_V2_FENCE


def test_c29_1112_v2_offender_warns():
    # Acceptance criterion 1 (plan #1114 §1): the #1112 v2 fence declaration
    # (48h, sized off base phases only) + the v2 §7 G1 dose-extension gate,
    # with the gate vocabulary outside the ±3-line window → WARN; the detail
    # names the incident anchors, the harvested gate label, and BOTH remedies.
    ok, by_id = _run(C29_OFFENDER)
    r = by_id["c29_fence_conditional_phase"]
    assert r.status == "WARN"
    assert ok  # WARN never flips overall (exit 0)
    assert "N/A — no conditional phase on this provision" in r.detail
    assert "#599" in r.detail
    assert "G1" in r.detail


def test_c29_1112_v3_reconciled_passes():
    # The corrected #1112 v3 declaration carries "§7"/"G1"/"contingency"/
    # "extension" on the declaration line itself → in-window evidence → PASS.
    plan = GOOD_PLAN + C29_GATE_SECTION + C29_SEPARATOR + C29_V3_FENCE
    _, by_id = _run(plan)
    r = by_id["c29_fence_conditional_phase"]
    assert r.status == "PASS"
    assert "references the §7 conditional phase" in r.detail


def test_c29_gate_label_only_reference_passes():
    # Pins the case-sensitive harvested-label path in isolation: the window
    # carries ONLY the §7 gate label (no _C29_EVIDENCE_RE vocabulary).
    fence = "`--max-run-duration 48h` — worst-case wall includes G1's wall cost.\n"
    plan = GOOD_PLAN + C29_GATE_SECTION + C29_SEPARATOR + fence
    _, by_id = _run(plan)
    r = by_id["c29_fence_conditional_phase"]
    assert r.status == "PASS"
    assert "G1" in r.detail


def test_c29_machine_type_token_is_not_label_evidence():
    # GCP machine-type tokens cluster exactly near fence text: `g2-standard-4`
    # / `a2-ultragpu-4g` in-window must NOT satisfy the label path (labels are
    # harvested from §7 only and matched case-SENSITIVELY) → still WARN.
    fence = (
        "`--max-run-duration 48h` on a2-ultragpu-4g (probe fallback g2-standard-4); "
        "sized off base phases only.\n"
    )
    plan = GOOD_PLAN + C29_GATE_SECTION + C29_SEPARATOR + fence
    assert _status(plan, "c29_fence_conditional_phase") == "WARN"


def test_c29_no_fence_skips():
    # Constraint (ii): gate section present, no fence declaration anywhere.
    _, by_id = _run(GOOD_PLAN + C29_GATE_SECTION)
    r = by_id["c29_fence_conditional_phase"]
    assert r.status == "SKIP"
    assert "no deliberate" in r.detail


def test_c29_bare_flag_default_mention_skips():
    # A bare-flag mention with the default in prose carries no value directly
    # after the flag → not a deliberate declaration.
    plan = (
        GOOD_PLAN
        + C29_GATE_SECTION
        + "\nThe dispatch keeps `--max-run-duration` (default 7d — the FLEX_START ceiling).\n"
    )
    assert _status(plan, "c29_fence_conditional_phase") == "SKIP"


def test_c29_explicit_7d_value_skips():
    # An explicit `=7d` (168h) IS the default ceiling, not deliberate.
    plan = GOOD_PLAN + C29_GATE_SECTION + "\nDispatch passes `--max-run-duration=7d`.\n"
    assert _status(plan, "c29_fence_conditional_phase") == "SKIP"


def test_c29_minutes_probe_value_skips():
    # The #680 cap-probe command shape (`=20m`) — minutes are not an h/d fence
    # value and never trigger.
    plan = GOOD_PLAN + C29_GATE_SECTION + "\nCap probe: `--max-run-duration=20m` create.\n"
    assert _status(plan, "c29_fence_conditional_phase") == "SKIP"


def test_c29_no_sect7_skips():
    # Constraint (i): a deliberate fence but NO §7-slot / Decision-Gates
    # heading anywhere (GOOD_PLAN's §7 heading renamed away) → SKIP.
    plan = (
        GOOD_PLAN.replace(CRITERIA_HEADING, "## Success and kill criteria")
        + C29_SEPARATOR
        + C29_V2_FENCE
    )
    _, by_id = _run(plan)
    r = by_id["c29_fence_conditional_phase"]
    assert r.status == "SKIP"
    assert "no §7" in r.detail


def test_c29_no_extension_gate_skips():
    # GOOD_PLAN's own §7 (success/kill criteria) carries no extension-class
    # vocabulary → SKIP even with a deliberate fence present.
    plan = GOOD_PLAN + C29_SEPARATOR + C29_V2_FENCE
    _, by_id = _run(plan)
    r = by_id["c29_fence_conditional_phase"]
    assert r.status == "SKIP"
    assert "no extension" in r.detail


def test_c29_na_escape_passes():
    plan = (
        C29_OFFENDER + "\nN/A — no conditional phase on this provision (the G1 extension is "
        "pre-registered to run on a second provision).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c29_fence_conditional_phase"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


def test_c29_quoted_na_phrase_does_not_escape():
    # A mid-sentence quoted escape phrase (the pasted-bounce-brief self-escape
    # channel) is NOT a standalone declaration line and must not escape — the
    # c13/c18/c26/c28 anti-paste twin (_standalone_na_declared semantics).
    # Padded >= 4 raw lines below the fence declaration so the quoted remedy
    # (which carries the evidence token "conditional") stays OUTSIDE the ±3
    # evidence window — this test pins the N/A-escape path specifically; the
    # in-window paste channel is a documented accepted permissive gap.
    plan = (
        C29_OFFENDER + "\n\n\n\n\nThe remedy menu says to declare 'N/A — no conditional "
        "phase on this provision' on its own line.\n"
    )
    assert _status(plan, "c29_fence_conditional_phase") == "WARN"


def test_c29_fenced_gate_text_does_not_trigger():
    # Gate text living ONLY inside a fenced example block within §7 is
    # fence-masked out of the gate-section prose (a gate is a prose contract)
    # → no extension trigger → SKIP.
    fenced_gate = (
        "\n\n```\n- **G1:** one-shot dose extension — retrain both FT cells, re-ladder.\n```"
    )
    plan = GOOD_PLAN.replace(KILL_SENT, KILL_SENT + fenced_gate) + C29_SEPARATOR + C29_V2_FENCE
    _, by_id = _run(plan)
    r = by_id["c29_fence_conditional_phase"]
    assert r.status == "SKIP"
    assert "no extension" in r.detail


def test_c29_fenced_fence_decl_triggers():
    # Raw-trigger polarity: a declaration living ONLY inside a fenced gcloud
    # command IS the real launch command (the c5/c26 raw-scan precedent) —
    # with a prose gate and no in-window evidence → WARN.
    fence_block = (
        "```bash\ngcloud compute instances create eps-issue-999 "
        "--max-run-duration=48h --instance-termination-action=DELETE\n```\n"
    )
    plan = GOOD_PLAN + C29_GATE_SECTION + C29_SEPARATOR + fence_block
    assert _status(plan, "c29_fence_conditional_phase") == "WARN"


def test_c29_evidence_outside_window_still_warns():
    # Pins _C29_WINDOW_LINES = 3: evidence placed 4 raw lines above the
    # declaration line must NOT satisfy (the real near-miss: #1112 v2's §0
    # Risks line sits 2 lines from its §0 fence mention).
    plan = (
        GOOD_PLAN
        + C29_GATE_SECTION
        + "\nThe §7 G1 dose extension is costed elsewhere.\n\n\n\n"
        + C29_V2_FENCE
    )
    assert _status(plan, "c29_fence_conditional_phase") == "WARN"


def test_c29_evidence_at_window_edge_passes():
    # Sibling of the outside-window test: evidence at EXACTLY distance 3
    # (the _C29_WINDOW_LINES edge) must satisfy. Pins the LOWER bound of
    # the window — a narrowing mutant (_C29_WINDOW_LINES = 0) passes every
    # other c29 test (code-review r1, empirically demonstrated); this one
    # kills it.
    plan = (
        GOOD_PLAN
        + C29_GATE_SECTION
        + "\nThe §7 G1 dose extension is costed elsewhere.\n\n\n"
        + C29_V2_FENCE
    )
    assert _status(plan, "c29_fence_conditional_phase") == "PASS"


@pytest.mark.parametrize("kind", ["infra", "batch", "survey"])
def test_c29_kind_exempt_skips(kind):
    assert _status(C29_OFFENDER, "c29_fence_conditional_phase", kind=kind) == "SKIP"


def test_c29_kind_analysis_warns():
    # analysis is IN scope (fence/§7-gate shapes are an experiment|analysis
    # plan shape) — same WARN severity as experiment.
    assert _status(C29_OFFENDER, "c29_fence_conditional_phase", kind="analysis") == "WARN"


def test_c29_assignment_form_offender_warns():
    # Statistics-critic Must-Fix (plan #1114 v3): the corpus-shaped
    # `spec.extra["max_run_duration"] = "48h"` assignment declaration WARNs
    # when unreconciled — pins _C29_FENCE_EXTRA_RE's quote/bracket
    # alternation (previously shipped untested).
    fence = (
        'The dispatch sets spec.extra["max_run_duration"] = "48h" for this provision; '
        "sized off base phases only.\n"
    )
    plan = GOOD_PLAN + C29_GATE_SECTION + C29_SEPARATOR + fence
    assert _status(plan, "c29_fence_conditional_phase") == "WARN"
    # The #628 no-space shape also matches the assignment regex.
    assert verify_plan._C29_FENCE_EXTRA_RE.search('max_run_duration"]="30h"')


def test_c29_assignment_form_default_skips():
    # Pins the 168h default exclusion on the ASSIGNMENT branch too.
    fence = 'The dispatch keeps spec.extra["max_run_duration"] = "7d" (the default).\n'
    plan = GOOD_PLAN + C29_GATE_SECTION + C29_SEPARATOR + fence
    assert _status(plan, "c29_fence_conditional_phase") == "SKIP"


def test_c29_prose_only_fence_skips():
    # Alternatives-critic Must-Fix (scope honesty): the #599-shaped prose-only
    # fence mention (value in parentheses, no `--` flag, no assignment) is a
    # named accepted false negative → SKIP. Pins the DECLARED-fence-subclass
    # scope so a future trigger widening fails loud. (The #833 shape — no
    # in-plan fence at all — is pinned by test_c29_no_fence_skips.)
    plan = (
        GOOD_PLAN
        + C29_GATE_SECTION
        + "\nThe run rides the GCP max-run-duration (~20 h) auto-delete window.\n"
    )
    assert _status(plan, "c29_fence_conditional_phase") == "SKIP"


def test_c29_multi_decl_any_site_satisfies():
    # Pins the any-site-satisfy loop (a loop bug here fails toward a nuisance
    # FP): two value-bearing declaration lines, first window bare, second
    # window carrying "§7 G1" → PASS.
    plan = (
        GOOD_PLAN
        + C29_GATE_SECTION
        + C29_SEPARATOR
        + "`--max-run-duration 48h` sized off base phases.\n\n\n\n\n"
        + "`--max-run-duration 48h` re-reconciled including the §7 G1 wall cost.\n"
    )
    assert _status(plan, "c29_fence_conditional_phase") == "PASS"


# ─── Plan-version + kind resolution ────────────────────────────────────────


def test_newest_plan_version_numeric_sort(tmp_path):
    plans = tmp_path / "plans"
    plans.mkdir()
    for name in ("v1.md", "v9.md", "v10.md", "v2-draft.md"):
        (plans / name).write_text(f"# {name}\n")
    (plans / "plan.md").symlink_to(plans / "v1.md")  # symlink must be ignored
    newest = verify_plan._newest_plan_version(tmp_path)
    assert newest.name == "v10.md"  # numeric, not lexicographic (v9 < v10)


def test_newest_plan_version_missing_raises(tmp_path):
    (tmp_path / "plans").mkdir()
    with pytest.raises(FileNotFoundError):
        verify_plan._newest_plan_version(tmp_path)


def test_kind_from_body(tmp_path):
    (tmp_path / "body.md").write_text("---\ntitle: x\nkind: infra\n---\n# x\n")
    assert verify_plan._kind_from_body(tmp_path) == "infra"


def test_kind_from_body_defaults_to_experiment(tmp_path):
    assert verify_plan._kind_from_body(tmp_path) == "experiment"  # no body.md
    (tmp_path / "body.md").write_text("---\ntitle: x\n---\n# x\n")
    assert verify_plan._kind_from_body(tmp_path) == "experiment"  # no kind key


# ─── CLI: --json schema, exit codes, --kind default ────────────────────────


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, check=False
    )


def test_cli_json_schema_and_exit_zero_on_pass(tmp_path):
    p = tmp_path / "plan.md"
    p.write_text(GOOD_PLAN)
    proc = _run_cli("--plan-file", str(p), "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["overall"] == "PASS"
    assert payload["issue"] is None
    assert payload["kind"] == "experiment"
    assert payload["n_fail"] == 0
    assert payload["n_skip"] == 30
    assert {"id", "name", "status", "detail"} <= set(payload["checks"][0])
    statuses = {c["status"] for c in payload["checks"]}
    assert statuses <= {"PASS", "WARN", "FAIL", "SKIP"}
    assert len(payload["checks"]) == 38
    assert len({c["id"] for c in payload["checks"]}) == 38
    # c23 has no task context in --plan-file mode: rendered SKIP (companion
    # assert for test_cli_issue_mode_appends_goal_currency).
    c23 = next(c for c in payload["checks"] if c["id"] == "c23_goal_currency")
    assert c23["status"] == "SKIP"
    assert "no task context" in c23["detail"]


def test_cli_exit_one_on_fail(tmp_path):
    p = tmp_path / "plan.md"
    p.write_text(GOOD_PLAN.replace(GPU_LINE, "no estimate"))
    proc = _run_cli("--plan-file", str(p), "--json")
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["overall"] == "FAIL"
    assert payload["n_fail"] >= 1


def test_cli_exit_two_on_missing_file(tmp_path):
    proc = _run_cli("--plan-file", str(tmp_path / "nope.md"))
    assert proc.returncode == 2
    assert "verify_plan" in proc.stderr


def test_cli_kind_defaults_to_experiment_in_file_mode(tmp_path):
    # A plan that FAILs under kind=experiment (no measurement validity) but
    # PASSes under kind=infra: the bare invocation must behave like
    # kind=experiment (the strictest default, pinned).
    p = tmp_path / "plan.md"
    p.write_text(_plan_without_mv())
    assert _run_cli("--plan-file", str(p)).returncode == 1
    assert _run_cli("--plan-file", str(p), "--kind", "infra").returncode == 0


def test_cli_human_output_has_overall_footer(tmp_path):
    p = tmp_path / "plan.md"
    p.write_text(GOOD_PLAN)
    proc = _run_cli("--plan-file", str(p))
    assert proc.returncode == 0
    assert "OVERALL: PASS" in proc.stdout
    assert "[SKIP]" in proc.stdout  # SKIP is a first-class rendered status


# ─── Check 23 — goal currency (outside CHECKS; --issue mode only) ──────────

# ≥12 normalized words each (the _C23_MIN_GOAL_WORDS precision gate).
_OLD_GOAL = (
    "Measure the marker leakage rate across every persona pair using the "
    "teacher forced margin metric on held out prompts"
)
_NEW_GOAL = (
    "Compare on policy judge scored refusal rates between trained and base "
    "models across twenty held out prompts"
)


def test_c23_stale_goal_quote_warns():
    plan = f'# Plan\n\n## 1. Goal\n\n"{_OLD_GOAL}" (Task #999 Goal, verbatim.)\n'
    r = verify_plan.check_goal_currency(plan, current_goal=_NEW_GOAL, superseded=[_OLD_GOAL])
    assert r.status == "WARN"
    assert "SUPERSEDED" in r.detail


def test_c23_current_goal_quoted_passes():
    # Current goal quoted verbatim; a sub-shingle-heavy fragment of the old
    # goal rides along — coverage(current) >= 0.3 blocks the stale signature.
    old_fragment = " ".join(_OLD_GOAL.split()[:8])
    plan = f'# Plan\n\n## 1. Goal\n\n"{_NEW_GOAL}" (was: {old_fragment})\n'
    r = verify_plan.check_goal_currency(plan, current_goal=_NEW_GOAL, superseded=[_OLD_GOAL])
    assert r.status == "PASS"


def test_c23_skips_without_superseded():
    plan = f"# Plan\n\n## 1. Goal\n\n{_NEW_GOAL}\n"
    r = verify_plan.check_goal_currency(plan, current_goal=_NEW_GOAL, superseded=[])
    assert r.status == "SKIP"


def test_c23_skips_short_goal():
    r = verify_plan.check_goal_currency(
        "# Plan\n\nsome head text\n", current_goal="fix the bug", superseded=[_OLD_GOAL]
    )
    assert r.status == "SKIP"


def test_c23_unicode_variant_quote_still_fires():
    # The #922 retrodiction shape: the marker's `from` text uses ASCII
    # `Delta` / `->`; the plan head quotes the unicode variants `Δ` / `→`.
    # Full-substring matching fails on this pair; shingle coverage stays
    # >= 0.5 because only the shingles spanning the divergent word are lost.
    stale_ascii = (
        "Quantify the Delta between pre and post fine tuning marker "
        "probabilities -> reported per persona across all twenty extraction "
        "prompts and four seeds"
    )
    quoted_unicode = (
        "Quantify the Δ between pre and post fine tuning marker "
        "probabilities → reported per persona across all twenty extraction "
        "prompts and four seeds"
    )
    plan = f'# Plan\n\n## 1. Goal\n\n"{quoted_unicode}" (Task #922 Goal, verbatim.)\n'
    r = verify_plan.check_goal_currency(plan, current_goal=_NEW_GOAL, superseded=[stale_ascii])
    assert r.status == "WARN"
    assert "SUPERSEDED" in r.detail


def _write_goal_marker(ev_path, ts_iso, from_goal, to_goal):
    row = {"ts": ts_iso, "kind": "epm:goal-updated", "version": 1, "from": from_goal, "to": to_goal}
    with ev_path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def test_goal_history_bounds_markers_by_plan_mtime(tmp_path):
    """Markers with ts <= plan mtime are included (INCLUSIVE at equality);
    a marker at mtime + 180 s is excluded. The +180 s fixture is the
    slack-regression detector: it sits inside the measured 2m48s-5m44s
    retro-stale gap band, so this test FAILs under any reintroduced slack
    >= 180 s (the removed 900 s regime would pass a mtime+900s fixture,
    which is why that offset is banned here)."""
    t = 1_750_000_000  # integer epoch so ts == mtime holds exactly

    def iso(epoch):
        return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    goal_a, goal_b, goal_c, goal_d = "goal AAA text", "goal BBB text", "goal CCC text", "goal DDD"
    (tmp_path / "plans").mkdir()
    plan = tmp_path / "plans" / "v1.md"
    plan.write_text("# Plan\n")
    os.utime(plan, (t, t))
    ev = tmp_path / "events.jsonl"
    _write_goal_marker(ev, iso(t - 3600), goal_a, goal_b)  # predating: included
    _write_goal_marker(ev, iso(t), goal_b, goal_c)  # exactly ts == mtime: included
    _write_goal_marker(ev, iso(t + 180), goal_c, goal_d)  # postdating: excluded
    mtime = datetime.fromtimestamp(plan.stat().st_mtime, tz=UTC)
    current, superseded = verify_plan._goal_history_for_plan(tmp_path, mtime)
    assert current == goal_c  # NOT goal_d — the +180 s marker must not enter
    assert goal_a in superseded and goal_b in superseded
    assert goal_c not in superseded and goal_d not in superseded


def test_goal_history_falls_back_to_frontmatter_goal(tmp_path):
    (tmp_path / "body.md").write_text(f"---\ntitle: x\ngoal: {_NEW_GOAL}\n---\n# x\n")
    now = datetime.now(tz=UTC)
    current, superseded = verify_plan._goal_history_for_plan(tmp_path, now)
    assert current == _NEW_GOAL
    assert superseded == []


def test_goal_history_parses_raw_u2028_inside_marker_text(tmp_path):
    """A goal-updated record whose text field carries a RAW U+2028 (written
    ensure_ascii=False, exactly as task_workflow._append_jsonl_line does) is
    ONE valid JSONL record under the canonical split("\\n") read.
    str.splitlines() splits on U+2028/U+2029/NEL too, shredding the record —
    the pre-boundary fragment still contains '"epm:goal-updated"', so the
    strict json.loads crashes (or, boundary-permuted, silently DROPS the
    marker — exactly c23's firing scenario). An ASCII fixture structurally
    cannot catch this (#950 JSONL-shred class)."""
    old = "goal AAA with a raw\u2028line separator inside"
    new = "goal BBB text entirely plain"
    row = {
        "ts": "2026-01-01T00:00:00Z",
        "kind": "epm:goal-updated",
        "version": 1,
        "from": old,
        "to": new,
    }
    line = json.dumps(row, ensure_ascii=False)
    assert "\u2028" in line  # fixture sanity: the separator survives serialization raw
    assert len(line.splitlines()) > 1  # fixture sanity: splitlines() WOULD shred this record
    (tmp_path / "events.jsonl").write_text(line + "\n", encoding="utf-8")
    current, superseded = verify_plan._goal_history_for_plan(tmp_path, datetime.now(tz=UTC))
    assert current == new
    assert superseded == [old]


def test_goal_history_raises_on_goal_updated_row_missing_ts(tmp_path):
    """A row whose kind IS epm:goal-updated but whose ts is missing/non-string
    is real corruption (the canonical writer always emits ts) — fail fast with
    row context, never silently shrink the predating goal history."""
    row = {"kind": "epm:goal-updated", "version": 1, "from": "goal AAA text", "to": "goal BBB"}
    (tmp_path / "events.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="epm:goal-updated"):
        verify_plan._goal_history_for_plan(tmp_path, datetime.now(tz=UTC))


def test_goal_history_tolerates_note_only_goal_updated_marker(tmp_path):
    """A hand-posted note-only goal-updated marker (ts present, no from/to) is
    structurally valid — it keeps its benign no-op skip; ONLY the
    missing/non-string-ts case raises."""
    row = {
        "ts": "2026-01-01T00:00:00Z",
        "kind": "epm:goal-updated",
        "version": 1,
        "note": "hand-posted, fieldless",
    }
    (tmp_path / "events.jsonl").write_text(json.dumps(row) + "\n")
    current, superseded = verify_plan._goal_history_for_plan(tmp_path, datetime.now(tz=UTC))
    assert current is None  # no body.md fallback in this fixture
    assert superseded == []


def test_cli_issue_mode_appends_goal_currency(tmp_path, monkeypatch, capsys):
    (tmp_path / "plans").mkdir()
    (tmp_path / "plans" / "v1.md").write_text(GOOD_PLAN)
    (tmp_path / "body.md").write_text("---\ntitle: x\nkind: experiment\n---\n# x\n")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(tw, "find_task_path", lambda n: tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify_plan.py", "--issue", "999", "--json"])
    rc = verify_plan.main()
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    entries = [c for c in payload["checks"] if c["id"] == "c23_goal_currency"]
    assert len(entries) == 1
    # Synthetic task has no goal frontmatter and no goal-updated markers.
    assert entries[0]["status"] == "SKIP"


# ─── Cross-file anchor pins (planner.md / CLAUDE.md drift detector) ────────


def test_planner_md_carries_predicate_anchor_literals():
    # If planner.md re-words a required block, THIS suite must break loudly
    # (the §7 heading-drift mitigation is a test, not prose).
    planner_md = (REPO_ROOT / ".claude" / "agents" / "planner.md").read_text()
    for anchor in (
        "Estimated GPU-hours (total):",
        "ungrounded — needs smoke-test",
        "Measurement validity",
        "What would change my mind",
    ):
        assert anchor in planner_md, f"planner.md lost the anchor literal {anchor!r}"


def test_claude_md_carries_predicate_anchor_literals():
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text()
    for anchor in ("ungrounded — needs smoke-test", "easurement validity"):
        assert anchor in claude_md, f"CLAUDE.md lost the anchor literal {anchor!r}"


def test_kind_enum_constants_match_canonical_code_kinds():
    """`verify_plan.EXEMPT_KINDS` / `VALID_KINDS` stay byte-identical to the
    canonical `task_workflow.CODE_KINDS` so the three `kind`-enum copies can
    never drift (incident #672). VALID_KINDS is kept an explicit ordered
    tuple here (argparse `choices=` display order) — this test is the gate
    that pins it to the single source of truth."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from explore_persona_space.task_workflow import CODE_KINDS, KINDS

    # EXEMPT_KINDS is exactly the canonical code/test-verdict subset.
    assert verify_plan.EXEMPT_KINDS == CODE_KINDS
    # VALID_KINDS = experiment + the exempt subset (same membership), and
    # every member is a real lifecycle kind.
    assert set(verify_plan.VALID_KINDS) == CODE_KINDS | {"experiment"}
    assert set(verify_plan.VALID_KINDS) <= set(KINDS)
    # The CLI-order tuple is pinned verbatim so a reorder is a visible diff.
    assert verify_plan.VALID_KINDS == ("experiment", "analysis", "infra", "batch", "survey")


# ─── No-LLM / no-network invariant (acceptance criterion 4) ────────────────


def test_script_makes_no_llm_or_network_imports():
    text = _SCRIPT.read_text()
    forbidden = re.findall(
        r"(?m)^\s*(?:import|from)\s+(anthropic|openai|requests|httpx|urllib|aiohttp|socket)\b",
        text,
    )
    assert not forbidden, f"verify_plan.py imports network/LLM modules: {forbidden}"


# ─── Check 30 — reused-bundle realized keys ────────────────────────────────

C30_BUNDLE_REUSE = (
    "\nWe reuse the parent's pass-B multi-field tensor bundle at "
    "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt "
    "as the frozen input for every arm.\n"
)


def test_c30_not_triggered_skips():
    assert _status(GOOD_PLAN, "c30_realized_keys") == "SKIP"


def test_c30_bundle_reuse_without_declaration_warns():
    _, by_id = _run(GOOD_PLAN + C30_BUNDLE_REUSE)
    r = by_id["c30_realized_keys"]
    assert r.status == "WARN"
    assert "verify_reused_artifact_keys" in r.detail


def test_c30_helper_invocation_passes():
    plan = (
        GOOD_PLAN
        + C30_BUNDLE_REUSE
        + "\nRealized-keys probe: `uv run python scripts/verify_reused_artifact_keys.py "
        + "--artifact <staged bundle> --keys cx_last,cx_mean,v_x,layers` — PASS line pasted "
        + "into §10.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c30_realized_keys"]
    assert r.status == "PASS"
    assert "verification named" in r.detail


def test_c30_fenced_helper_command_satisfies():
    # Pins the raw-text satisfier scan: the helper is named ONLY inside a
    # fenced block (where runnable commands legitimately live) and still
    # satisfies, while the trigger fires from the stripped prose.
    plan = (
        GOOD_PLAN
        + C30_BUNDLE_REUSE
        + "\n```bash\nuv run python scripts/verify_reused_artifact_keys.py "
        + "--artifact bundle.pt --keys cx_last,v_x\n```\n"
    )
    assert _status(plan, "c30_realized_keys") == "PASS"


def test_c30_mmap_keys_declaration_passes():
    plan = (
        GOOD_PLAN
        + C30_BUNDLE_REUSE
        + '\nVerified via torch.load(path, map_location="cpu", mmap=True).keys() '
        + "against every consumer assert at the pinned revision.\n"
    )
    assert _status(plan, "c30_realized_keys") == "PASS"


def test_c30_consumer_loader_declaration_passes():
    plan = (
        GOOD_PLAN
        + C30_BUNDLE_REUSE
        + "\nThe consumer's own loader (issue1073_common.load_bundle) was run against the "
        + "real pinned artifact before approval.\n"
    )
    assert _status(plan, "c30_realized_keys") == "PASS"


def test_c30_na_escape_passes():
    # Standalone-line escape (the mid-line form was the self-escape shape —
    # repurposed into test_c30_quoted_na_phrase_does_not_escape below).
    plan = (
        GOOD_PLAN
        + "\nThe parent produced analysis_tensors bundles (.pt) but this design does not "
        + "reuse them.\n"
        + "\nN/A — no multi-field bundle reuse.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c30_realized_keys"]
    assert r.status == "PASS"
    assert "no-bundle-reuse declaration" in r.detail


def test_c30_quoted_na_phrase_does_not_escape():
    # Anti-paste guard: the old mid-line escape shape must not satisfy the
    # standalone-line escape. Quoted-phrase form, not full-detail paste — the
    # satisfier-name channel of the pasted detail is a disclosed residual
    # (plan §8 row 6).
    base = GOOD_PLAN + C30_BUNDLE_REUSE
    assert _status(base, "c30_realized_keys") == "WARN"
    quoted = base + (
        "\nThe parent produced analysis_tensors bundles (.pt) but this design does not "
        "reuse them: N/A — no multi-field bundle reuse.\n"
    )
    assert _status(quoted, "c30_realized_keys") == "WARN"


def test_c30_kind_infra_skips():
    assert _status(GOOD_PLAN + C30_BUNDLE_REUSE, "c30_realized_keys", kind="infra") == "SKIP"


def test_c30_kind_analysis_warns():
    assert _status(GOOD_PLAN + C30_BUNDLE_REUSE, "c30_realized_keys", kind="analysis") == "WARN"


def test_c30_adapter_only_reuse_skips():
    # Adapter-only reuse (LoRA / adapter_config.json) is c6 territory, not c30.
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent LoRA adapters (adapter_config.json verified) from "
        + "superkaiba1/explore-persona-space for the base arm.\n"
    )
    assert _status(plan, "c30_realized_keys") == "SKIP"


def test_c30_adapter_safetensors_reuse_skips():
    # Pins the v2 Must-Fix: `.safetensors` is NOT a trigger token — a plan
    # quoting `adapter_model.safetensors` near reuse vocabulary (the canonical
    # check-(e) verification sentence) must SKIP, not WARN. A corpus sweep
    # showed 9 historical adapter-class plans fire via `.safetensors` alone.
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters; adapter_model.safetensors and "
        + "adapter_config.json both resolve at the cited subfolder.\n"
    )
    assert _status(plan, "c30_realized_keys") == "SKIP"


# ─── Check 31 — SKILL.md prose edit backed by a durability pin ─────────────

# Synthetic edit-commitment fixture (committed tests never read real tasks/
# plan files — the c13/c16 suites' convention; the incident fixture below is
# a near-verbatim synthetic of #884 v1).
C31_SKILL_EDIT = (
    "\n## 3. Files + diffs\n\n"
    "- `.claude/skills/issue/SKILL.md` Step 8: insert one new bold-titled "
    "paragraph requiring VM-side long compute to be setsid-detached.\n"
)


def test_c31_kind_experiment_skips():
    plan = GOOD_PLAN + C31_SKILL_EDIT
    assert _status(plan, "c31_skillmd_prose_pin", kind="experiment") == "SKIP"


def test_c31_no_skillmd_edit_skips():
    assert _status(GOOD_PLAN, "c31_skillmd_prose_pin", kind="infra") == "SKIP"


def test_c31_skillmd_edit_without_pin_warns():
    _, by_id = _run(GOOD_PLAN + C31_SKILL_EDIT, kind="infra")
    r = by_id["c31_skillmd_prose_pin"]
    assert r.status == "WARN"
    assert "#884" in r.detail
    assert "Durability pin:" in r.detail  # the WARN teaches the exact line to add


def test_c31_kind_batch_triggers():
    assert _status(GOOD_PLAN + C31_SKILL_EDIT, "c31_skillmd_prose_pin", kind="batch") == "WARN"


def test_c31_pasted_warn_detail_does_not_self_satisfy():
    # Anti-paste guard (#4.2 c31 de-fang): pre-fix, the WARN detail's quoted
    # example `Durability pin: N/A — <one-line reason>` satisfied
    # _C31_PIN_NA_RE when pasted back into the plan (`<` meets the \S
    # requirement); the reworded detail closes the backtick right after N/A.
    base = GOOD_PLAN + C31_SKILL_EDIT
    _, by_id = _run(base, kind="infra")
    r = by_id["c31_skillmd_prose_pin"]
    assert r.status == "WARN"
    pasted = base + f"\n{r.detail}\n"
    assert _status(pasted, "c31_skillmd_prose_pin", kind="infra") == "WARN"


def test_c31_pin_line_passes():
    plan = (
        GOOD_PLAN + C31_SKILL_EDIT + "\nDurability pin: tests/test_issue_skill_marker_contract.py::"
        "test_step8_detach_prose_present\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c31_skillmd_prose_pin"]
    assert r.status == "PASS"
    assert "pin named" in r.detail


def test_c31_planned_new_pin_test_passes():
    # A pin test the plan itself ADDS counts — the verifier cannot and need
    # not distinguish standing from planned (the code-reviewer checks the
    # diff ships it).
    plan = (
        GOOD_PLAN
        + C31_SKILL_EDIT
        + "\nDurability pin: NEW tests/test_issue_skill_detach_convention.py::"
        "test_setsid_prose_pinned (added by this plan §4.3)\n"
    )
    assert _status(plan, "c31_skillmd_prose_pin", kind="infra") == "PASS"


def test_c31_na_escape_passes():
    plan = (
        GOOD_PLAN
        + C31_SKILL_EDIT
        + "\nDurability pin: N/A — the inserted sentence is narrative cross-reference "
        "prose; no code or parser couples to its wording.\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c31_skillmd_prose_pin"]
    assert r.status == "PASS"
    assert "justification" in r.detail


def test_c31_alias_na_escape_passes():
    plan = (
        GOOD_PLAN
        + C31_SKILL_EDIT
        + "\nN/A — no durability pin: the paragraph is a pointer to an existing "
        "pinned block.\n"
    )
    assert _status(plan, "c31_skillmd_prose_pin", kind="infra") == "PASS"


def test_c31_paren_na_escape_passes():
    # NA-charset parity with house NA_RE (which also accepts an opening
    # paren): `Durability pin: N/A (reason)` satisfies, not a spurious WARN.
    plan = (
        GOOD_PLAN
        + C31_SKILL_EDIT
        + "\nDurability pin: N/A (narrative pointer prose; no parser couples to it).\n"
    )
    assert _status(plan, "c31_skillmd_prose_pin", kind="infra") == "PASS"


def test_c31_bare_na_without_reason_warns():
    # The reason tail is mandatory (`\S` after the separator) — a contentless
    # rubber-stamp still WARNs.
    plan = GOOD_PLAN + C31_SKILL_EDIT + "\nDurability pin: N/A\n"
    assert _status(plan, "c31_skillmd_prose_pin", kind="infra") == "WARN"


def test_c31_fenced_edit_mention_does_not_trigger():
    plan = (
        GOOD_PLAN + "\n## 3. Files\n\n"
        "```\n- `.claude/skills/issue/SKILL.md`: insert one new paragraph.\n```\n"
    )
    assert _status(plan, "c31_skillmd_prose_pin", kind="infra") == "SKIP"


def test_c31_negated_mention_does_not_trigger():
    # Pins both measured negation classes (#700/#875) incl. the dot-aware gap
    # atom: "SKILL.md change" carries a path-internal dot the guard must span.
    plan = (
        GOOD_PLAN + "\nSo: zero `workflow.yaml` edits and zero SKILL.md edits.\n"
        "No SKILL.md change needed.\n"
    )
    assert _status(plan, "c31_skillmd_prose_pin", kind="infra") == "SKIP"


def test_c31_must_ask_boilerplate_does_not_trigger():
    # Deviation-contract boilerplate names SKILL.md next to an edit verb but
    # commits to nothing (#890 class).
    plan = (
        GOOD_PLAN + "\nMust-ask (park `plan_pending`): editing `workflow.yaml` / `SKILL.md` "
        "contract lines.\n"
    )
    assert _status(plan, "c31_skillmd_prose_pin", kind="infra") == "SKIP"


def test_c31_relative_path_triggers():
    # A skills-relative path without the `.claude/skills/` prefix still names
    # a SKILL.md edit target (the path arm admits any slash-joined prefix).
    plan = GOOD_PLAN + "\n## 3. Files\n\n- `issue/SKILL.md` Step 6d: append one guard sentence.\n"
    assert _status(plan, "c31_skillmd_prose_pin", kind="infra") == "WARN"


def test_c31_fenced_pin_line_satisfies():
    # The satisfier scan is RAW (c11/c15 evidence convention) — a pin line in
    # a fenced §-block counts; pins against a future fence-stripping refactor.
    plan = (
        GOOD_PLAN + C31_SKILL_EDIT + "\n```\nDurability pin: "
        "tests/test_issue_skill_exit_breadcrumb.py::test_step8_block_present\n```\n"
    )
    assert _status(plan, "c31_skillmd_prose_pin", kind="infra") == "PASS"


def test_c31_loose_test_mention_does_not_satisfy():
    # The c15-style loose-evidence shape that false-satisfied all 9 incident
    # plan versions (unrelated test_ identifier + incidental SKILL.md
    # vocabulary) must NOT satisfy c31 — the labeled-line design's
    # load-bearing regression test.
    plan = (
        GOOD_PLAN
        + C31_SKILL_EDIT
        + "\nTests: test_workflow_lint_check_asks stays green; the SKILL.md contract "
        "is untouched.\n"
    )
    assert _status(plan, "c31_skillmd_prose_pin", kind="infra") == "WARN"


def test_c31_incident_884_shape_warns():
    # Near-verbatim synthetic #884 v1 fragment: a REAL pin test named only in
    # ad-hoc unlabeled prose does not satisfy — #884's pin existed but was
    # never labeled; the label is the machine-readable contract.
    plan = (
        GOOD_PLAN + "\n## 2. What I'll change\n\n"
        "**What I'll change:** two prose insertions (SKILL.md breadcrumb convention + "
        "code-style.md nohup bullet) requiring any VM-side compute launch over SSH MCP "
        "to be setsid-detached with a pidfile.\n"
        "The gate is `workflow_lint.py` + `tests/test_workflow_setsid_detach_convention.py` "
        "green on the touched files.\n"
    )
    assert _status(plan, "c31_skillmd_prose_pin", kind="infra") == "WARN"


def test_planner_md_carries_durability_pin_instruction():
    # The pin test for the planner.md author-side bullet this task ships —
    # the rule dogfoods itself.
    text = (REPO_ROOT / ".claude" / "agents" / "planner.md").read_text()
    assert re.search(r"Durability pin:", text)
    assert "#884" in text


def test_adversarial_planner_skill_lists_c31_escape():
    # The pin test for the Phase-1.5.0 escape-phrase registration (c21/#1042
    # precedent: bounce briefs quote the canonical phrases verbatim).
    text = (REPO_ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md").read_text()
    anchor = text.index("Canonical N/A escape phrases")
    block = text[anchor : text.index("bounce to the planner", anchor)]
    assert "no durability pin" in block
    assert "check 31" in block


def test_c32_skillmd_na_phrase_listed():
    # The c32 durability pin (plan #1194 §4.2): the canonical escape phrase
    # is registered in the adversarial-planner SKILL.md escape list, so a
    # later SKILL.md edit cannot silently drop it.
    text = (REPO_ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md").read_text()
    anchor = text.index("Canonical N/A escape phrases")
    block = text[anchor : text.index("bounce to the planner", anchor)]
    assert "no fit-family phases" in block
    assert "check 32" in block


def test_c33_skillmd_na_phrase_listed():
    # The c33 durability pin (#1246, mirroring the c32 pin above): the
    # canonical escape phrase + alias are registered in the
    # adversarial-planner SKILL.md escape list, so a later SKILL.md edit
    # cannot silently drop them. The full backticked-phrase asserts pin the
    # exact strings c33's recognizer matches (em-dash, casing), closing
    # prefix drift the bare substrings would miss.
    text = (REPO_ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md").read_text()
    anchor = text.index("Canonical N/A escape phrases")
    block = text[anchor : text.index("bounce to the planner", anchor)]
    assert "no per-rung checkpoint persistence" in block
    assert "no checkpoint ladder" in block
    assert "check 33" in block
    assert "`N/A — no per-rung checkpoint persistence`" in block
    assert "`N/A — no checkpoint ladder`" in block


# ─── Check 32 — fit-family §9 basis grounding ──────────────────────────────

# Reusable §9 fit-row snippet: kernel vocabulary ("ridge") + loop vocabulary
# ("3780 fits") in a basis-column compute table; the basis cell is the
# per-test variable.
C32_FIT_TABLE = """
## 9. Resources & Parallelism

| component | planned_wall_h | basis | parallelism |
|---|---|---|---|
| ridge refit sweep (3780 fits, 28 layers x fold x arm) | 0.35 | {basis} | 1x A100 |
"""


def _c32_plan(basis: str) -> str:
    return GOOD_PLAN + C32_FIT_TABLE.format(basis=basis)


def test_c32_kind_infra_skips():
    assert _status(_c32_plan("~2 s/fit"), "c32_fit_basis_grounding", kind="infra") == "SKIP"


def test_c32_no_fit_row_skips():
    # GOOD_PLAN has no basis-column table at all.
    assert _status(GOOD_PLAN, "c32_fit_basis_grounding") == "SKIP"
    # A basis-column row with loop vocabulary but NO fit-family kernel (a
    # generation row) does not trigger either.
    gen_table = (
        "\n## 9. Resources\n\n"
        "| component | planned_wall_h | basis | parallelism |\n"
        "|---|---|---|---|\n"
        "| vLLM generation (250 gens, per-source sharding) | 0.5 | ~3 s/gen asserted | 1x A100 |\n"
    )
    assert _status(GOOD_PLAN + gen_table, "c32_fit_basis_grounding") == "SKIP"


def test_c32_asserted_number_warns():
    # The #823 literal shape: a bare asserted per-call cost, no provenance.
    _, by_id = _run(_c32_plan("~2 s/fit"))
    r = by_id["c32_fit_basis_grounding"]
    assert r.status == "WARN"
    assert "#823" in r.detail
    assert "N/A — no fit-family phases" in r.detail  # the WARN teaches the escape


def test_c32_bare_measured_word_warns():
    # The boilerplate polarity (#552 v3's real "measured: minutes" is also
    # digit-free): the magic word without a number does not satisfy.
    assert _status(_c32_plan("measured pilot"), "c32_fit_basis_grounding") == "WARN"


def test_c32_measured_pilot_with_figure_passes():
    basis = "measured 1-cell pilot: 125 s/fit through the production entrypoint"
    assert _status(_c32_plan(basis), "c32_fit_basis_grounding") == "PASS"


def test_c32_prior_issue_figure_passes():
    assert _status(_c32_plan("#811 r2 measured 313 s/unit"), "c32_fit_basis_grounding") == "PASS"


def test_c32_parent_figure_passes():
    # The #810 v10 real shape: a parent-run realized figure without the
    # word "measured" — "parent" + the timing token satisfy.
    basis = "parent full grid ~10 min => ~0.58 s/cell"
    assert _status(_c32_plan(basis), "c32_fit_basis_grounding") == "PASS"


def test_c32_pilot_gated_passes():
    basis = "pilot-gated (first-step pilot, abort >2x re-projection)"
    assert _status(_c32_plan(basis), "c32_fit_basis_grounding") == "PASS"


def test_c32_flop_only_basis_warns():
    # A FLOP-derived basis carries a timing token but no provenance word —
    # the rule: a FLOP floor is the cross-check, never the basis; there is
    # deliberately NO `FLOP-only` escape (plan #1194 §4.1 decision 1).
    basis = "FLOP-derived: O(d*r^2) ~1 ms/solve"
    assert _status(_c32_plan(basis), "c32_fit_basis_grounding") == "WARN"


def test_c32_na_escape_passes():
    plan = (
        _c32_plan("~2 s/fit")
        + "\nN/A — no fit-family phases (the flagged row is a batched one-shot solve).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c32_fit_basis_grounding"]
    assert r.status == "PASS"
    assert "N/A declared" in r.detail


def test_c32_pasted_escape_does_not_self_escape():
    # The anti-paste polarity (`_standalone_na_declared` house semantics,
    # cf. the c13/c26 precedents): the escape phrase quoted mid-sentence —
    # e.g. a bounced plan pasting the WARN detail verbatim — must not
    # satisfy re-verification.
    plan = (
        _c32_plan("~2 s/fit")
        + "\nThe remedy would be to declare 'N/A — no fit-family phases' if the row "
        "were not a fit loop, which it is.\n"
    )
    assert _status(plan, "c32_fit_basis_grounding") == "WARN"


def test_c32_fenced_table_ignored():
    fenced = "\n## 9. Resources\n\n```\n" + C32_FIT_TABLE.format(basis="~2 s/fit") + "\n```\n"
    assert _status(GOOD_PLAN + fenced, "c32_fit_basis_grounding") == "SKIP"


def test_c32_warn_never_fails():
    # WARN-only contract (exit-0 semantics): an offender never flips overall.
    ok, by_id = _run(_c32_plan("~2 s/fit"))
    assert by_id["c32_fit_basis_grounding"].status == "WARN"
    assert ok is True


def test_c32_kind_analysis_triggers():
    # The gated set is experiment + analysis; BOTH WARN (unlike c12/c18
    # there is no FAIL arm anywhere).
    assert _status(_c32_plan("~2 s/fit"), "c32_fit_basis_grounding", kind="analysis") == "WARN"


# ─── Check 33 — checkpoint-ladder retention policy ─────────────────────────

# Ladder trigger spliced into GOOD_PLAN's §9 Resources paragraph (the
# sizing section the satisfier scope reads); the retention line is spliced
# per-test. Verified at plan time: GOOD_PLAN carries no c33 trigger or
# satisfier tokens of its own.
C33_LADDER_SENT = (
    "Phase T trains a 30-rung dose ladder; per-rung checkpoints are written to /workspace/ckpts."
)
C33_LADDER_S9 = GOOD_PLAN.replace(
    "One A100 for about three hours covers both conditions.",
    "One A100 for about three hours covers both conditions. " + C33_LADDER_SENT,
)
assert C33_LADDER_SENT in C33_LADDER_S9  # surgery-anchor sanity (fixture hygiene)
C33_RETENTION_LINE = (
    "Retention: keep the dose-selected + latest rungs only; ruled-out rungs are deleted "
    "between rungs."
)


def test_c33_kind_infra_skips():
    assert _status(C33_LADDER_S9, "c33_ladder_retention", kind="infra") == "SKIP"
    assert _status(C33_LADDER_S9, "c33_ladder_retention", kind="batch") == "SKIP"


def test_c33_ladder_without_retention_warns():
    _, by_id = _run(C33_LADDER_S9)
    r = by_id["c33_ladder_retention"]
    assert r.status == "WARN"
    assert "plan-compute-sizing" in r.detail  # the #1133 rule anchor
    assert "#1112" in r.detail  # the incident cite
    assert "N/A — no per-rung checkpoint persistence" in r.detail  # teaches the escape


def test_c33_retention_in_sizing_section_passes():
    plan = C33_LADDER_S9.replace(
        "## 11. Decision Rationale", C33_RETENTION_LINE + "\n\n## 11. Decision Rationale"
    )
    _, by_id = _run(plan)
    r = by_id["c33_ladder_retention"]
    assert r.status == "PASS"
    assert "retention vocabulary" in r.detail


def test_c33_retention_outside_sizing_section_warns():
    # Section-scoping is real: the SAME retention line placed in §2/§3 (no
    # sizing keyword in the heading) does not satisfy while §9 stays clean —
    # the doc-wide-satisfier escape let #1112 v1-v3 pass on incidental
    # vocabulary in calibration.
    plan = C33_LADDER_S9.replace("## 3. Conditions", C33_RETENTION_LINE + "\n\n## 3. Conditions")
    assert _status(plan, "c33_ladder_retention") == "WARN"


def test_c33_no_sizing_heading_falls_back_doc_wide():
    # No heading carries a sizing keyword -> the satisfier scope falls back
    # to the whole plan's non-fenced text (structural absence must not
    # manufacture WARNs — a WARN-only check fails toward silence).
    plan = C33_LADDER_S9.replace("## 9. Resources", "## 9. Budget").replace(
        "## 3. Conditions", C33_RETENTION_LINE + "\n\n## 3. Conditions"
    )
    assert _status(plan, "c33_ladder_retention") == "PASS"


def test_c33_na_escape_passes():
    plan = (
        C33_LADDER_S9
        + "\nN/A — no per-rung checkpoint persistence (this plan reads a parent's existing "
        "ladder; no new rungs are persisted).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c33_ladder_retention"]
    assert r.status == "PASS"
    assert "N/A declared" in r.detail


def test_c33_na_alias_passes():
    plan = C33_LADDER_S9 + "\nN/A — no checkpoint ladder (single terminal checkpoint only).\n"
    assert _status(plan, "c33_ladder_retention") == "PASS"


def test_c33_fenced_na_escape_does_not_satisfy():
    # Anti-paste `_standalone_na_declared` semantics (c13/c26/c32 parity):
    # the escape inside a fence (a quoted bounce brief) must not satisfy.
    plan = C33_LADDER_S9 + "\n```\nN/A — no per-rung checkpoint persistence\n```\n"
    assert _status(plan, "c33_ladder_retention") == "WARN"


def test_c33_backend_rung_line_does_not_trigger():
    # GCP fallback-ladder vocabulary: rung + checkpoint co-located WITH
    # backend vocab is the excluded class (the load-bearing anti-fragility
    # widening — raw rung/checkpoint vocabulary is dominated by it).
    plan = GOOD_PLAN + "\nOn a capacity miss the spot rung falls back; checkpoints upload to HF.\n"
    assert _status(plan, "c33_ladder_retention") == "SKIP"


def test_c33_fenced_ladder_does_not_trigger():
    plan = GOOD_PLAN + "\n```\n" + C33_LADDER_SENT + "\n```\n"
    assert _status(plan, "c33_ladder_retention") == "SKIP"


def test_c33_save_every_triggers():
    # The rule's "any long run saving every k steps for a later pick"
    # clause — no rung/ladder token anywhere in the plan.
    plan = GOOD_PLAN + "\nThe trainer saves a checkpoint every 25 steps for a later dose pick.\n"
    assert _status(plan, "c33_ladder_retention") == "WARN"


def test_c33_bare_colocation_triggers():
    # Branch-3-only trigger (critic-ensemble case): rung + checkpoint on
    # one line, no compound token, no backend vocab — without this case the
    # co-location branch could ship inert with all other tests green.
    plan = GOOD_PLAN + "\nEach rung writes a full checkpoint to /workspace/ckpts.\n"
    assert _status(plan, "c33_ladder_retention") == "WARN"


def test_c33_warn_never_fails():
    # WARN-only contract (exit-0 semantics): an offender never flips overall.
    ok, by_id = _run(C33_LADDER_S9)
    assert by_id["c33_ladder_retention"].status == "WARN"
    assert ok is True


# ─── Anti-paste composite + SKILL.md documentation pin (#1237) ─────────────


def test_composite_bounce_brief_paste_does_not_improve_any_check():
    # A composite bounce brief — several checks' LIVE FAIL/WARN details
    # concatenated into one paste block (the shape a mechanical-bounce
    # revision produces) — appended to each triggering base plan must not
    # improve any check's status. Self-updating: the details are captured
    # live, so a future detail reword that re-opens a self-satisfaction
    # channel fails here without touching the fixture.
    cases = [
        (
            GOOD_PLAN.replace("## 11. Decision Rationale (§11)", "## 11. Notes").replace(
                "Source:", "Ref:"
            ),
            "c1_source_grounding",
            "experiment",
            "FAIL",
        ),
        (_plan_without_mv(), "c2_measurement_validity", "experiment", "FAIL"),
        (GOOD_PLAN + DRYRUN_SMOKE, "c11_dryrun_test_coverage", "infra", "WARN"),
        (C16_INCIDENT, "c16_reference_headline_distinction", "experiment", "WARN"),
        (_predictive_plan(), "c19_ood_folds", "experiment", "WARN"),
    ]
    details = []
    for base, cid, kind, expected in cases:
        _, by_id = _run(base, kind=kind)
        assert by_id[cid].status == expected, cid
        details.append(by_id[cid].detail)
    blob = "\n" + "\n".join(details) + "\n"
    for base, cid, kind, expected in cases:
        assert _status(base + blob, cid, kind=kind) == expected, cid


def test_skillmd_canonical_escapes_documents_standalone_line():
    # Durability pin (plan §4.3b): the adversarial-planner SKILL.md
    # canonical-escapes bullet must carry the standalone-declaration-line
    # sentence — without it, post-migration bounce loops recur (a planner
    # told to "quote verbatim" pastes the phrase mid-sentence and bounces
    # again).
    skill_md = (REPO_ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    idx = skill_md.index("Canonical N/A escape phrases")
    bullet = skill_md[idx : idx + 2500]
    assert "standalone declaration line" in bullet


# ─── Check 34 — verbatim insert vs size-ratchet headroom ──────────────────

# Fixture strategy (plan #1260 §6): monkeypatch verify_plan._C34_REPO_ROOT to
# tmp_path and create ratcheted files at controlled sizes — testagent.md at
# 39,900 B (non-grandfathered → headroom 100 B vs AGENT_SPEC_FAIL_BYTES
# 40,000) and LESSONS.md 50 B under its BINDING constraint (#1269:
# min(_LESSONS_MAX_BYTES, _LESSONS_RATCHET_BYTES) — the growth ratchet in
# practice; derived from the live constants so routine ratchet bumps don't
# shift the pinned headroom arithmetic). The trigger fragments mirror the
# real #1230 plan v1 §4.1 shape: a heading naming the path, a "Verbatim text
# to insert" line, then the fenced block.


def _c34_fixture_root(tmp_path):
    wl = verify_plan._c34_lint_constants()
    (tmp_path / ".claude" / "agents").mkdir(parents=True)
    (tmp_path / ".claude" / "rules").mkdir(parents=True)
    (tmp_path / ".claude" / "agents" / "testagent.md").write_bytes(b"x" * 39_900)
    lessons_size = min(wl._LESSONS_MAX_BYTES, wl._LESSONS_RATCHET_BYTES) - 50
    (tmp_path / ".claude" / "rules" / "LESSONS.md").write_bytes(b"x" * lessons_size)
    return tmp_path


def _c34_agent_fragment(block_bytes: int) -> str:
    # Block bytes = joined content lines + one trailing newline (c34's byte
    # recipe), so content is block_bytes - 1 chars of ASCII.
    content = "x" * (block_bytes - 1)
    return (
        "\n## 4. Files + diffs\n\n"
        "### 4.1 `.claude/agents/testagent.md` — Step 6 duty\n\n"
        "**Verbatim text to insert (one paragraph):**\n\n"
        "```\n" + content + "\n```\n"
    )


def _c34_lessons_fragment(block_bytes: int) -> str:
    content = "x" * (block_bytes - 1)
    return (
        "\n### 4.2 `.claude/rules/LESSONS.md` index row\n\n"
        "Append this entry verbatim to the LESSONS index:\n\n"
        "```\n" + content + "\n```\n"
    )


def test_c34_kind_experiment_skips():
    plan = GOOD_PLAN + _c34_agent_fragment(300)
    assert _status(plan, "c34_ratchet_headroom", kind="experiment") == "SKIP"


def test_c34_no_ratcheted_file_skips():
    assert _status(GOOD_PLAN, "c34_ratchet_headroom", kind="infra") == "SKIP"


def test_c34_1230_shaped_plan_warns(tmp_path, monkeypatch):
    # The incident-class fixture (#1230 v1 §4.1 shape): 300 B block vs
    # 100 B live headroom → WARN whose detail carries the arithmetic; the
    # WARN never flips overall (exit-0 semantics).
    monkeypatch.setattr(verify_plan, "_C34_REPO_ROOT", _c34_fixture_root(tmp_path))
    ok, by_id = _run(GOOD_PLAN + _c34_agent_fragment(300), kind="infra")
    r = by_id["c34_ratchet_headroom"]
    assert r.status == "WARN"
    assert "~300 B" in r.detail  # block bytes
    assert "headroom 100 B" in r.detail  # cap 40,000 - live 39,900
    assert "AGENT_SPEC_FAIL_BYTES" in r.detail  # cap source
    assert "#1230" in r.detail  # incident anchor
    assert ok is True  # WARN-only contract


def test_c34_block_fits_headroom_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_plan, "_C34_REPO_ROOT", _c34_fixture_root(tmp_path))
    _, by_id = _run(GOOD_PLAN + _c34_agent_fragment(50), kind="infra")
    r = by_id["c34_ratchet_headroom"]
    assert r.status == "PASS"
    assert "fit remaining headroom" in r.detail


def test_c34_na_escape_passes():
    # The escape short-circuits BEFORE any disk access (no monkeypatch /
    # fixture files needed) — trailing prose after the phrase is tolerated
    # per _standalone_na_declared's re.match semantics.
    plan = (
        GOOD_PLAN
        + _c34_agent_fragment(300)
        + "\nN/A — no verbatim ratcheted-file insertion (the fenced block is illustrative).\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c34_ratchet_headroom"]
    assert r.status == "PASS"
    assert "escape declared" in r.detail


def test_c34_missing_file_skips(tmp_path, monkeypatch):
    # tmp_path carries NO .claude tree: headroom uncomputable → SKIP, no
    # exception (a plan may be CREATING the file; --plan-file mode must
    # never crash off-repo).
    monkeypatch.setattr(verify_plan, "_C34_REPO_ROOT", tmp_path)
    _, by_id = _run(GOOD_PLAN + _c34_agent_fragment(300), kind="infra")
    r = by_id["c34_ratchet_headroom"]
    assert r.status == "SKIP"
    assert "not present on disk" in r.detail


def test_c34_budget_line_passes(tmp_path, monkeypatch):
    # Over-headroom block + a digit-bearing budget line → PASS: the plan
    # budgets the visible cap-raise, the legitimate grandfather-convention
    # path (workflow_lint: "a reviewed growth+cap-raise in one commit still
    # passes").
    monkeypatch.setattr(verify_plan, "_C34_REPO_ROOT", _c34_fixture_root(tmp_path))
    plan = (
        GOOD_PLAN
        + _c34_agent_fragment(300)
        + "\nRatchet budget: raise AGENT_SPEC_SIZE_GRANDFATHER['testagent.md'] to 41_000\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c34_ratchet_headroom"]
    assert r.status == "PASS"
    assert "cap-raise budgeted" in r.detail


def test_c34_budget_line_is_document_global(tmp_path, monkeypatch):
    # Documents the DISCLOSED v1 scope note (c34 section comment, scope
    # note (a)): the `Ratchet budget:` satisfier is DOCUMENT-GLOBAL, not
    # per-target — a two-file plan budgeting only ONE raise passes for
    # both. Accepted residual for a WARN-class check; this pin records the
    # intent so a future per-target tightening is a deliberate decision.
    monkeypatch.setattr(verify_plan, "_C34_REPO_ROOT", _c34_fixture_root(tmp_path))
    plan = (
        GOOD_PLAN
        + _c34_agent_fragment(300)  # over the agent file's 100 B headroom
        + _c34_lessons_fragment(200)  # over LESSONS.md's 50 B headroom
        + "\nRatchet budget: raise AGENT_SPEC_SIZE_GRANDFATHER['testagent.md'] to 41_000\n"
    )
    assert _status(plan, "c34_ratchet_headroom", kind="infra") == "PASS"


def test_c34_pasted_warn_detail_does_not_self_satisfy(tmp_path, monkeypatch):
    # Anti-paste pin (the c31 test shape): the WARN detail writes the budget
    # label followed only by angle-bracket placeholders (no post-label digit
    # → _C34_BUDGET_RE's lookahead cannot match; the incident numbers sit
    # BEFORE the label) and backtick-wraps the N/A phrase (unrecognized by
    # _standalone_na_declared, #1238) — a verbatim paste still WARNs.
    monkeypatch.setattr(verify_plan, "_C34_REPO_ROOT", _c34_fixture_root(tmp_path))
    base = GOOD_PLAN + _c34_agent_fragment(300)
    _, by_id = _run(base, kind="infra")
    r = by_id["c34_ratchet_headroom"]
    assert r.status == "WARN"
    pasted = base + f"\n{r.detail}\n"
    assert _status(pasted, "c34_ratchet_headroom", kind="infra") == "WARN"


def test_c34_fenced_path_mention_does_not_trigger():
    # Path + verb INSIDE a fence above another fence: fenced lines are
    # excluded from the association window, so nothing triggers → SKIP.
    plan = GOOD_PLAN + (
        "\n```\nInsert this into .claude/agents/testagent.md verbatim.\n```\n\n"
        "```\n" + "x" * 299 + "\n```\n"
    )
    assert _status(plan, "c34_ratchet_headroom", kind="infra") == "SKIP"


def test_c34_lessons_md_headroom_warns(tmp_path, monkeypatch):
    # #1269: the binding LESSONS.md constraint is min(cap, growth ratchet) —
    # the ratchet in practice (it must sit under the cap), so the WARN
    # detail names _LESSONS_RATCHET_BYTES as the cap source.
    monkeypatch.setattr(verify_plan, "_C34_REPO_ROOT", _c34_fixture_root(tmp_path))
    _, by_id = _run(GOOD_PLAN + _c34_lessons_fragment(200), kind="infra")
    r = by_id["c34_ratchet_headroom"]
    assert r.status == "WARN"
    assert "_LESSONS_RATCHET_BYTES" in r.detail
    assert "headroom 50 B" in r.detail


def test_c34_multi_block_sum_warns(tmp_path, monkeypatch):
    # Two 60 B blocks for the SAME file (headroom 100 B): each alone fits,
    # the per-file SUM (120 B) does not → the WARN proves the summing.
    monkeypatch.setattr(verify_plan, "_C34_REPO_ROOT", _c34_fixture_root(tmp_path))
    two_blocks = (
        "\n### 4.1 `.claude/agents/testagent.md` — split insert\n\n"
        "Insert these two paragraphs verbatim into the spec:\n\n"
        "```\n" + "x" * 59 + "\n```\n\n"
        "```\n" + "x" * 59 + "\n```\n"
    )
    _, by_id = _run(GOOD_PLAN + two_blocks, kind="infra")
    r = by_id["c34_ratchet_headroom"]
    assert r.status == "WARN"
    assert "~120 B" in r.detail


def test_c34_phrase_listed_in_skillmd():
    # The c34 durability pin (plan #1260 §3): the canonical escape phrase is
    # registered backtick-wrapped in the adversarial-planner SKILL.md escape
    # list, so a later SKILL.md reflow cannot silently drop it (the c33
    # omission cost follow-up task #1246). The standing
    # test_skillmd_canonical_escape_block_never_self_escapes separately pins
    # that the entry never self-declares.
    text = (REPO_ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md").read_text()
    anchor = text.index("Canonical N/A escape phrases")
    block = text[anchor : text.index("bounce to the planner", anchor)]
    assert "no verbatim ratcheted-file insertion" in block
    assert "check 34" in block
    assert "`N/A — no verbatim ratcheted-file insertion`" in block


# ─── Check 35 — revision-pinned reuse verified at the pin ─────────────────

C35_PINNED_REUSE = (
    "\n## Reuse\n\nWe reuse the parent's raw-completion shards from "
    "superkaiba1/explore-persona-space-data at pinned revision "
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa (issue1345 S-track stems).\n"
)

# #1345 plan v3 §10 pairwise-provenance row, VERBATIM (tasks/*/1345/plans/v3.md
# line 446 at implementation time) — the artifact-reuse item-(j) boilerplate
# that blinded the v2 satisfier design to its own motivating incident.
C35_ITEM_J_ROW = (
    "| Pairwise provenance coherence | R1/R2 shard `last_commit` dates at revision "
    "deb7a452 vs the `issue825_extract_turnstore.py` version that wrote them — verify "
    "at implementation time via `get_paths_info(expand=True, revision=...)` |"
)


def test_c35_not_triggered_skips():
    assert _status(GOOD_PLAN, "c35_pinned_revision_reuse") == "SKIP"


def test_c35_pinned_reuse_without_probe_warns():
    _, by_id = _run(GOOD_PLAN + C35_PINNED_REUSE)
    r = by_id["c35_pinned_revision_reuse"]
    assert r.status == "WARN"
    assert "pin" in r.detail and "1345" in r.detail


def test_c35_revision_scoped_probe_passes():
    plan = (
        GOOD_PLAN
        + C35_PINNED_REUSE
        + '\n```python\nlist_repo_tree("superkaiba1/explore-persona-space-data", '
        + 'path_in_repo="issue1345_s/", revision="aaaa...", repo_type="dataset")\n```\n'
    )
    assert _status(plan, "c35_pinned_revision_reuse") == "PASS"


def test_c35_default_branch_probe_does_not_satisfy():
    # Pins exactly the #1345 failure shape: an existence probe with NO
    # revision kwarg ("confirmed" at the default branch) must not satisfy.
    plan = (
        GOOD_PLAN
        + C35_PINNED_REUSE
        + "\nVerified via list_repo_files('superkaiba1/explore-persona-space-data').\n"
    )
    assert _status(plan, "c35_pinned_revision_reuse") == "WARN"


def test_c35_prose_verified_at_pinned_revision_passes():
    plan = GOOD_PLAN + C35_PINNED_REUSE + "\nExistence verified at the pinned revision per stem.\n"
    assert _status(plan, "c35_pinned_revision_reuse") == "PASS"


def test_c35_pasted_warn_detail_does_not_self_satisfy():
    # MUST-FIX 1(b): bounced plans paste the verifier detail verbatim; the
    # detail must not satisfy the check it came from (#810 shape).
    _, by_id = _run(GOOD_PLAN + C35_PINNED_REUSE)
    detail = by_id["c35_pinned_revision_reuse"].detail
    plan = GOOD_PLAN + C35_PINNED_REUSE + "\n" + detail + "\n"
    assert _status(plan, "c35_pinned_revision_reuse") == "WARN"


def test_c35_warn_detail_matches_no_satisfier():
    # MUST-FIX 1(c): pin satisfier-inertness of the detail string directly —
    # no Hub-callable + `revision=`/`revision:` co-occurrence on one line, no
    # `verif...at...revision` shape.
    _, by_id = _run(GOOD_PLAN + C35_PINNED_REUSE)
    detail = by_id["c35_pinned_revision_reuse"].detail
    assert verify_plan._C35_PROBE_SATISFIER_RE.search(detail) is None
    assert verify_plan._C35_PROSE_SATISFIER_RE.search(detail) is None


def test_c35_item_j_provenance_boilerplate_does_not_satisfy():
    # MUST-FIX 2(b): regression for the #1345-v3 blind spot — the item-(j)
    # `get_paths_info(expand=True, revision=...)` provenance row verifies
    # commit-DATE coherence, NOT existence-at-pin, and must not satisfy
    # (get_paths_info is deliberately excluded from the probe satisfier).
    plan = GOOD_PLAN + C35_PINNED_REUSE + "\n" + C35_ITEM_J_ROW + "\n"
    assert _status(plan, "c35_pinned_revision_reuse") == "WARN"


def test_c35_na_escape_passes():
    plan = GOOD_PLAN + C35_PINNED_REUSE + "\nN/A — no revision-pinned reuse\n"
    _, by_id = _run(plan)
    r = by_id["c35_pinned_revision_reuse"]
    assert r.status == "PASS"
    assert "declaration" in r.detail


def test_c35_quoted_na_phrase_does_not_escape():
    # Anti-paste convention (#810/#1238 lineage): the phrase quoted
    # mid-sentence (e.g. inside a pasted bounce brief) does not count.
    plan = (
        GOOD_PLAN
        + C35_PINNED_REUSE
        + "\nThe bounce brief quotes `N/A — no revision-pinned reuse` as the check-35 escape.\n"
    )
    assert _status(plan, "c35_pinned_revision_reuse") == "WARN"


def test_c35_kind_infra_skips():
    assert (
        _status(GOOD_PLAN + C35_PINNED_REUSE, "c35_pinned_revision_reuse", kind="infra") == "SKIP"
    )


def test_c35_git_code_sha_without_hf_context_skips():
    # A bare git code SHA (Repro-card `pinned to commit <40-hex>` row, no HF
    # context / reuse vocabulary nearby) must not trigger.
    plan = GOOD_PLAN + (
        "\nRepro: code pinned to commit bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb on branch main.\n"
    )
    assert _status(plan, "c35_pinned_revision_reuse") == "SKIP"


def test_c35_phrase_listed_in_skillmd():
    # Durability pin (the c34 precedent): the canonical escape phrase is
    # registered backtick-wrapped in the adversarial-planner SKILL.md escape
    # list, so a later SKILL.md reflow cannot silently drop it.
    text = (REPO_ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md").read_text()
    anchor = text.index("Canonical N/A escape phrases")
    block = text[anchor : text.index("bounce to the planner", anchor)]
    assert "no revision-pinned reuse" in block
    assert "check 35" in block
    assert "`N/A — no revision-pinned reuse`" in block


# ─── Check 36 — numeric containment claims ─────────────────────────────────

# Verbatim corpus literals — embedded as literals, NEVER read from tasks/ at
# test runtime (tasks move between status folders; the c28 fixture
# convention). C36_1315_LINE is the founding incident sentence,
# tasks/*/1315/plans/v4.md L66 (the "Install: …" sentence; the leading
# "Weights: …" clause of the physical line is trimmed — the plan-#1375 §6
# fixture spec names the sentence). Its FIRST containment claim is TRUE
# (0.724 in [0.60, 0.85]); the SECOND is arithmetically FALSE
# (0.724 < 0.737) — verify_plan PASSed 0/0 on it and two critics caught it
# by hand (-> #1375).
C36_1315_LINE = (
    "Install: Tier-1 0.663 in band at step 20, Tier-2 confirm 0.724 — inside the registered "
    "0.60–0.85 band and inside the siblings' realized 0.737–0.820 spread (install-strength "
    "control satisfied without dose-matching work)."
)
# #1315 v5 L68 (the corrected prose; trailing parenthetical trimmed at the
# c28 precedent) — the true-negative sibling of the founding line.
C36_1315_V5_LINE = (
    "Install: Tier-1 0.663 in band at step 20, Tier-2 confirm 0.724 — inside the registered "
    "0.60–0.85 band, 0.013 BELOW the siblings' realized 0.737–0.820 spread (install-strength "
    "control satisfied via the registered band without dose-matching work)."
)
# #1315 v4 L126 (trimmed): "next to" is not a containment verb.
C36_1315_NEXT_TO_LINE = (
    "install-band control (0.724 in band next to siblings 0.737–0.820; per-cell realized "
    "install reported next to every geometry DV)"
)
# #1315 v4 L274 verbatim: a ± tolerance is not a range.
C36_1315_PLUSMINUS_LINE = (
    "1. Parity probe: judged rate within 0.724 ± 0.15 AND application check ≥ 0.5 nat — or "
    "the named retrain-from-frozen-mix fallback lands in band."
)
# #816 v1 L227 (trimmed): "10-20-draw" is a compound count modifier, not a range claim.
C36_816_LINE = (
    "- **Exp 4 — kills** if: a norm-matched random-direction preventative finetune reduces "
    "the post-ft trait score as much as the real vector (real's reduction inside the "
    "10–20-draw random band, given the p-floor)."
)
# #514 v1 L175 verbatim: table row — cross-cell numbers must never be attributed.
C36_514_ROW = (
    "| Dense lever, 30% epoch | Whether stopping just past the #508 FT-light cell catches a "
    "clean cell above 9 nat | Budget resolution within the 0.25-0.5 epoch window | "
    "`ft_dense_b30` |"
)
# #597 v2 L33 (trimmed): "H2" is a label digit, not a claimed value.
C36_597_LINE = (
    "- **H2 (rotation + gate collapse at the cliff).** The consecutive-checkpoint "
    "top-direction cosine has its minimum inside steps 12–40 (the cliff/onset window), and "
    "the gate ρ trajectory drops from its early value toward ~0 across the same window, in "
    "most sources."
)
# #1353 v1 L52 (trimmed): a TRUE claim that also pins the \b partial-digit
# operand boundaries (without them "150-330 band" garbage-parses).
C36_1353_LINE = (
    "(~300 words — inside the siblings' 150–330 band. Register elements checked against "
    "L282/L283: bold headline sentence · explicit RULE: clauses · sibling cross-reference.)"
)


def _c36_plan(*extra: str) -> str:
    """GOOD_PLAN + ``extra`` lines appended (the ``_c28_plan`` pattern)."""
    return GOOD_PLAN + "\n" + "\n\n".join(extra) + ("\n" if extra else "")


def test_c36_1315_v4_false_claim_warns():
    _, by_id = _run(_c36_plan(C36_1315_LINE))
    r = by_id["c36_numeric_containment"]
    assert r.status == "WARN"
    assert "0.737" in r.detail and "0.724" in r.detail  # the false pair


def test_c36_true_claim_same_sentence_not_flagged():
    # The L66 first half only — the TRUE claim (0.724 in [0.60, 0.85]).
    plan = _c36_plan("Tier-2 confirm 0.724 — inside the registered 0.60–0.85 band")
    assert _status(plan, "c36_numeric_containment") == "PASS"


def test_c36_corrected_v5_line_passes():
    assert _status(_c36_plan(C36_1315_V5_LINE), "c36_numeric_containment") == "PASS"


def test_c36_next_to_non_fire():
    # No containment verb -> no claim detected at all.
    assert _status(_c36_plan(C36_1315_NEXT_TO_LINE), "c36_numeric_containment") == "SKIP"


def test_c36_plusminus_tolerance_non_fire():
    # "within 0.724 ± 0.15" is a tolerance, not an A-B range.
    assert _status(_c36_plan(C36_1315_PLUSMINUS_LINE), "c36_numeric_containment") == "SKIP"


def test_c36_fenced_false_claim_skips():
    plan = GOOD_PLAN + "\n```\n" + C36_1315_LINE + "\n```\n"
    assert _status(plan, "c36_numeric_containment") == "SKIP"


def test_c36_blockquoted_false_claim_skips():
    assert _status(_c36_plan("> " + C36_1315_LINE), "c36_numeric_containment") == "SKIP"


@pytest.mark.parametrize("kind", ["infra", "batch", "survey"])
def test_c36_kind_exempt_skips(kind):
    # The self-reference-trap pin: workflow-fix plans quoting the incident
    # are kind: infra and never scanned (plan-#1375 §4.7 layer 1).
    assert _status(_c36_plan(C36_1315_LINE), "c36_numeric_containment", kind=kind) == "SKIP"


def test_c36_na_escape_passes():
    plan = _c36_plan(C36_1315_LINE, "N/A — no numeric containment claims")
    _, by_id = _run(plan)
    r = by_id["c36_numeric_containment"]
    assert r.status == "PASS"
    assert "declared" in r.detail


def test_c36_quoted_na_phrase_does_not_escape():
    # Anti-paste convention (#810/#1238 lineage): the phrase quoted
    # mid-sentence (e.g. inside a pasted bounce brief) does not count.
    plan = _c36_plan(
        C36_1315_LINE,
        "The bounce brief quotes `N/A — no numeric containment claims` as the check-36 escape.",
    )
    assert _status(plan, "c36_numeric_containment") == "WARN"


def test_c36_boundary_equality_inside():
    # Fraction-exact boundary inclusion: N == A counts as inside.
    plan = _c36_plan("The realized install of 0.60 sits within the registered 0.60–0.85 band.")
    assert _status(plan, "c36_numeric_containment") == "PASS"


def test_c36_reversed_bounds_normalized():
    # A reversed range is a prose-order quirk, not an exemption: [0.60, 0.85]
    # is still verified after normalization.
    ok_plan = _c36_plan("The confirm read 0.75 sits inside the 0.85–0.60 band.")
    assert _status(ok_plan, "c36_numeric_containment") == "PASS"
    bad_plan = _c36_plan("The confirm read 0.50 sits inside the 0.85–0.60 band.")
    assert _status(bad_plan, "c36_numeric_containment") == "WARN"


def test_c36_hyphen_range_parses():
    # Unspaced ASCII hyphen with unsigned operands is a range.
    plan = _c36_plan("The mix uses 4 negatives, within the 2-4 range.")
    assert _status(plan, "c36_numeric_containment") == "PASS"


def test_c36_negative_hyphen_ambiguity_never_parses():
    # A signed/hyphen mix is ambiguous between range and negative pair.
    plan = _c36_plan("The shift 0.1 lands within the -0.5-0.3 band under this recipe.")
    assert _status(plan, "c36_numeric_containment") == "SKIP"


def test_c36_percent_unit_segregation():
    # %-flagged candidates compare only against %-flagged ranges.
    pct_plan = _c36_plan("72% of judged rows sit within the 60–85% range.")
    assert _status(pct_plan, "c36_numeric_containment") == "PASS"
    # Mixed units: the only candidate is %-flagged, the range unitless ->
    # no attributable claimed value -> no claim.
    mixed_plan = _c36_plan("The 30% rate sits within the 0.25–0.5 window.")
    assert _status(mixed_plan, "c36_numeric_containment") == "SKIP"


def test_c36_compound_count_modifier_non_fire():
    assert _status(_c36_plan(C36_816_LINE), "c36_numeric_containment") == "SKIP"


def test_c36_table_cell_bleed_non_fire():
    # Candidate window cut at the last `|`: the sibling cell's "9 nat" is
    # never attributed to the "0.25-0.5 epoch window" claim.
    assert _status(_c36_plan(C36_514_ROW), "c36_numeric_containment") == "SKIP"


def test_c36_label_digit_not_a_candidate():
    # "H2" is killed by the candidate lookbehind; a candidate-less match
    # does not count as a claim.
    assert _status(_c36_plan(C36_597_LINE), "c36_numeric_containment") == "SKIP"


def test_c36_true_word_count_claim_passes():
    assert _status(_c36_plan(C36_1353_LINE), "c36_numeric_containment") == "PASS"


# ─── Check 37 — no-flags bundling claim vs workflow_lint dispatch ──────────

# The #1322 v1 incident shape (verbatim clause structure): an acceptance
# criterion asserting the pre-commit-only reference check rides the bare run.
C37_FALSE_CLAIM = (
    "2. `uv run python scripts/workflow_lint.py` (no-flags default run) passes — this "
    "includes `--check-references` (which walks CLAUDE.md; both new cross-refs must resolve)."
)
C37_TRUE_CLAIM = (
    "1. `uv run python scripts/workflow_lint.py` (no-flags default run, which bundles "
    "`--check-lessons-index`) passes."
)


def _c37_plan(*extra: str) -> str:
    return GOOD_PLAN + "\n" + "\n\n".join(extra) + ("\n" if extra else "")


def test_c37_kind_experiment_skips():
    r_by = _run(_c37_plan(C37_FALSE_CLAIM), kind="experiment")[1]["c37_noflags_bundling_claim"]
    assert r_by.status == "SKIP"
    assert "kind-exempt" in r_by.detail


def test_c37_false_claim_warns():
    _, by_id = _run(_c37_plan(C37_FALSE_CLAIM), kind="infra")
    r = by_id["c37_noflags_bundling_claim"]
    assert r.status == "WARN"
    assert "`--check-references`" in r.detail
    assert "args.check_references" in r.detail


def test_c37_true_claim_passes():
    # `--check-lessons-index` IS dispatched on the no-flags run
    # (workflow_lint.py `args.check_lessons_index or no_flags`).
    assert _status(_c37_plan(C37_TRUE_CLAIM), "c37_noflags_bundling_claim", kind="infra") == "PASS"


def test_c37_negated_line_does_not_trigger():
    # The corrected #1322 v2 / workflow_lint-docstring phrasing: the claim
    # anchor is present ("bundled into the no-flags") but the negation guard
    # drops the line — asserted-negative mentions are never adjudicated.
    plan = _c37_plan(
        "`--check-references` is NOT bundled into the no-flags default run — pass it explicitly."
    )
    assert _status(plan, "c37_noflags_bundling_claim", kind="infra") == "SKIP"


def test_c37_space_spelling_no_flags_warns():
    # The "no flags" (space) vocabulary form survives the verb-anchored
    # narrowing — pinned per the critic-round ask.
    plan = _c37_plan("Run `workflow_lint.py` (no flags — bundles `--check-references`) as a gate.")
    assert _status(plan, "c37_noflags_bundling_claim", kind="infra") == "WARN"


def test_c37_two_command_idiom_does_not_trigger():
    # Calibration FP class 1 (155+ corpus instances pre-narrowing): two
    # separate invocations listed on one line make no bundling claim.
    plan = _c37_plan(
        "- `uv run python scripts/workflow_lint.py --check-asks` and the no-flags "
        "default `uv run python scripts/workflow_lint.py` both exit 0."
    )
    assert _status(plan, "c37_noflags_bundling_claim", kind="infra") == "SKIP"


def test_c37_into_destination_flag_not_flagged():
    # Calibration FP class 2 (the reference-check-extension family,
    # #714/#753/#739/#802/#1190): "bundled into `--check-references` and the
    # no-flags default run" claims the SUBJECT flag's membership, not the
    # destination bundle's — the subject (in-set) PASSes, the destination is
    # never adjudicated.
    plan = _c37_plan(
        "New check `--check-lessons-index` (also bundled into `--check-references` "
        "and the no-flags default run) walks the rules index."
    )
    assert _status(plan, "c37_noflags_bundling_claim", kind="infra") == "PASS"


def test_c37_proposed_new_flag_passes():
    # Calibration FP class 3 (forward-looking extension plans): a flag with
    # no occurrence in the workflow_lint source is a PROPOSED new check —
    # unfalsifiable at plan time, never an offender.
    plan = _c37_plan(
        "Add `--check-notyetbuilt`, bundled into the no-flags default run, to workflow_lint."
    )
    assert _status(plan, "c37_noflags_bundling_claim", kind="infra") == "PASS"


def test_c37_fenced_claim_does_not_trigger():
    plan = GOOD_PLAN + "\n```\n" + C37_FALSE_CLAIM + "\n```\n"
    assert _status(plan, "c37_noflags_bundling_claim", kind="infra") == "SKIP"


def test_c37_no_vocab_skips():
    assert _status(GOOD_PLAN, "c37_noflags_bundling_claim", kind="infra") == "SKIP"


def test_c37_na_escape_passes():
    plan = _c37_plan(C37_FALSE_CLAIM, "N/A — no no-flags bundling claim")
    _, by_id = _run(plan, kind="infra")
    r = by_id["c37_noflags_bundling_claim"]
    assert r.status == "PASS"
    assert "declared" in r.detail


def test_c37_quoted_na_phrase_does_not_escape():
    # Anti-paste convention (#810/#1238 lineage, the c36 precedent): the
    # phrase quoted mid-sentence / backtick-wrapped does not count.
    plan = _c37_plan(
        C37_FALSE_CLAIM,
        "The bounce brief quotes `N/A — no no-flags bundling claim` as the check-37 escape.",
    )
    assert _status(plan, "c37_noflags_bundling_claim", kind="infra") == "WARN"


def test_c37_pasted_warn_detail_does_not_retrigger_or_satisfy():
    # The WARN detail leads with the negated truth (NOT) so a verbatim paste
    # can neither re-trigger (negation guard) nor self-satisfy (the escape
    # phrase in the detail is backtick-wrapped).
    _, by_id = _run(_c37_plan(C37_FALSE_CLAIM), kind="infra")
    detail = by_id["c37_noflags_bundling_claim"].detail
    clean_plus_detail = _c37_plan(detail)
    assert _status(clean_plus_detail, "c37_noflags_bundling_claim", kind="infra") == "SKIP"
    offending_plus_detail = _c37_plan(C37_FALSE_CLAIM, detail)
    assert _status(offending_plus_detail, "c37_noflags_bundling_claim", kind="infra") == "WARN"


def test_c37_live_derivation_pins():
    # Live-tree pin: the source-regex derivation stays plausible and keeps
    # the ground-truth memberships. A workflow_lint main() dispatch-shape
    # refactor fails HERE (forcing a deliberate _C37_DISPATCH_RE update)
    # while the check itself only SKIPs.
    src = verify_plan._c37_lint_source()
    assert src is not None
    dests = verify_plan._c37_noflags_dests(src)
    assert dests is not None and len(dests) >= 40, dests
    assert "references" not in dests  # the founding #1322 ground truth
    assert "lessons_index" in dests
    # The parenthesized `(args.check_X or no_flags) and not
    # args.check_references` dispatch form parses too:
    assert "marker_registry" in dests


def test_c37_underivable_source_skips(tmp_path, monkeypatch):
    # A stub lint file with zero dispatch lines → below the plausibility
    # floor → loud SKIP, never a spray of WARNs (c34's missing-file pattern).
    stub = tmp_path / "workflow_lint.py"
    stub.write_text("def main():\n    pass\n")
    monkeypatch.setattr(verify_plan, "_C37_LINT_PATH", stub)
    _, by_id = _run(_c37_plan(C37_FALSE_CLAIM), kind="infra")
    r = by_id["c37_noflags_bundling_claim"]
    assert r.status == "SKIP"
    assert "underivable" in r.detail


def test_c37_missing_lint_file_skips(tmp_path, monkeypatch):
    # --plan-file mode off-repo: workflow_lint.py absent → SKIP, no crash.
    monkeypatch.setattr(verify_plan, "_C37_LINT_PATH", tmp_path / "workflow_lint.py")
    _, by_id = _run(_c37_plan(C37_FALSE_CLAIM), kind="infra")
    assert by_id["c37_noflags_bundling_claim"].status == "SKIP"


def test_c37_phrase_listed_in_skillmd():
    # Durability pin (the c34 test shape): the canonical escape phrase is
    # registered backtick-wrapped in the adversarial-planner SKILL.md escape
    # block; the generative sync test separately propagates the docstring
    # registration.
    text = (REPO_ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md").read_text()
    anchor = text.index("Canonical N/A escape phrases")
    block = text[anchor : text.index("bounce to the planner", anchor)]
    assert "`N/A — no no-flags bundling claim`" in block
    assert "check 37" in block


def test_canonical_json_parse_snippet_pinned():
    """#1290 drift pin: the adversarial-planner SKILL.md canonical parse one-liner
    uses the real `overall` key with fail-loud d['overall'] (never .get()), and
    _json_payload still emits every key the documented contract names."""
    skill = (REPO_ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md").read_text()
    lines = [ln for ln in skill.splitlines() if "verify_plan.py --issue <N> --json |" in ln]
    assert len(lines) == 1, "exactly one canonical parse one-liner expected in SKILL.md"
    (ln,) = lines
    assert "d['overall']" in ln and "d['n_fail']" in ln and "d['n_warn']" in ln
    assert ".get(" not in ln  # fail-loud key access — KeyError on drift, never None
    payload = verify_plan._json_payload(source="s", issue=1, kind="infra", overall=True, results=[])
    assert {"overall", "n_fail", "n_warn", "n_skip", "checks"} <= payload.keys()
    assert payload["overall"] == "PASS"
    # FAIL branch, strengthened with one REAL failing CheckResult (critic ask,
    # plan §9-allowed): the per-check dicts carry at least {id, status} and the
    # n_fail counter counts them.
    fail = verify_plan.CheckResult(id="c1_training_hparams", name="hparams", passed=False)
    payload_fail = verify_plan._json_payload(
        source="s", issue=None, kind="infra", overall=False, results=[fail]
    )
    assert payload_fail["overall"] == "FAIL"
    assert payload_fail["n_fail"] == 1
    assert {"id", "status"} <= payload_fail["checks"][0].keys()
    assert payload_fail["checks"][0]["status"] == "FAIL"
