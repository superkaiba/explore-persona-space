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

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = REPO_ROOT / "scripts" / "verify_plan.py"
_spec = importlib.util.spec_from_file_location("verify_plan", _SCRIPT)
verify_plan = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_plan"] = verify_plan
_spec.loader.exec_module(verify_plan)  # type: ignore[union-attr]


# ─── Canonical plan (kind=experiment: passes 0-3,5,8,9,17; skips 4,6,7,10-16,18,19)

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
    }
    actual = {cid: r.status for cid, r in by_id.items()}
    assert actual == expected
    assert len(results) == 21


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


def test_c4_na_line_passes():
    plan = (
        GOOD_PLAN + "\nThe word implant appears but this is not a behavior-implantation design.\n"
    )
    assert _status(plan, "c4_contrastive_negatives") == "PASS"


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
    assert "4/10" in r.detail


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
    assert "4/10" in r.detail


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
    assert "4/10" in r.detail


def test_c6_fitness_letters_beyond_j_do_not_count():
    # Upper-boundary fixture (#941): an unrelated (k) elsewhere in the body must
    # NOT lift a 3-letter fitness attestation to a 4-letter PASS — an
    # over-widening of the class to [a-k]/[a-z] would flip this to PASS.
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\nFitness check: (a) same recipe verified against adapter_config.json; (b) valid measurement regime; (c) required cells present."
        + "\nUnrelated enumeration elsewhere: (k) a non-fitness bullet.\n"
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
    assert "(a)–(j)" in r.detail or "ten" in r.detail


def test_c6_na_no_artifact_reuse_passes():
    plan = (
        GOOD_PLAN
        + "\nPrior adapters at superkaiba1/explore-persona-space exist; reuse was considered and rejected. N/A — no artifact reuse.\n"
    )
    assert _status(plan, "c6_reuse_fitness") == "PASS"


def test_c6_heading_triggers():
    plan = GOOD_PLAN + "\n## 10. Reused-artifact fitness check\n\nNothing here yet.\n"
    _, by_id = _run(plan)
    assert by_id["c6_reuse_fitness"].status == "WARN"


def test_c6_kind_infra_skips():
    plan = GOOD_PLAN + "\nWe reuse adapters from superkaiba1/explore-persona-space.\n"
    assert _status(plan, "c6_reuse_fitness", kind="infra") == "SKIP"


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
    assert "dry_run" in r.detail
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
    plan = (
        GOOD_PLAN
        + "\nworktree_audit.py already supports --dry-run; this plan does not touch that path. N/A — no dry-run smoke.\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c11_dryrun_test_coverage"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


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


def test_c12_na_no_draw_battery_passes():
    plan = (
        GOOD_PLAN + f"\n{BATTERY_SENT} N/A — no draw battery (quoting the sibling's methodology).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c12_battery_multiplier"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


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
    # Regression for the `_trigger_windows` parametrization (task #937): c12's
    # ±15-raw-line radius must not shift. Evidence exactly 15 raw lines below
    # the trigger line counts; one line farther does not. The evidence line
    # deliberately avoids battery-trigger vocabulary so it cannot open its own
    # window (asserted below, not assumed).
    evidence = "Basis: draws × 24 cells; implementation: batched via one subset-sum GEMM."
    assert not verify_plan._BATTERY_TRIGGER_RE.search(evidence)

    def plan_with_gap(n_blank: int) -> str:
        # Evidence lands (n_blank + 1) raw lines below the trigger line.
        return (
            GOOD_PLAN
            + "\n## 12. Null battery\n\n"
            + BATTERY_SENT
            + "\n"
            + "\n" * n_blank
            + evidence
            + "\n"
        )

    assert _status(plan_with_gap(14), "c12_battery_multiplier") == "PASS"  # +15 lines: in
    assert _status(plan_with_gap(15), "c12_battery_multiplier") == "FAIL"  # +16 lines: out


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


def test_c13_na_escape_passes():
    plan = _c13_plan(C13_TABLE_SMALL, C13_GATE) + (
        "\nN/A — no empirical-null gate (the p ≤ 0.05 mention quotes the sibling's methodology).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c13_empirical_gate_attainability"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


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
    plan = (
        GOOD_PLAN
        + FAILLOUD_ACCEPT
        + "\nThe 'silently' sentence above narrates the pre-fix defect. "
        "N/A — no fail-loud acceptance claim.\n"
    )
    _, by_id = _run(plan, kind="infra")
    r = by_id["c15_failloud_test_coverage"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


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
    assert "same-pass comparator" in r.detail


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


def test_c18_na_escape_passes():
    plan = _c18_plan(C18_V13_REGISTRATION) + (
        "\nN/A — no paired contrast (the paired statistic above recaps the sibling's "
        "registration).\n"
    )
    _, by_id = _run(plan)
    r = by_id["c18_paired_contrast_source_coverage"]
    assert r.status == "PASS"
    assert "N/A" in r.detail


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


def test_c20_na_escape_passes():
    plan = _c20_plan(C20_V4_BULLETS) + (
        "\nN/A — no registered verdict lattice (the labels quote the parent's methodology).\n"
    )
    _, by_id = _run(plan)
    r = by_id[C20]
    assert r.status == "PASS"
    assert "N/A" in r.detail


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
    assert payload["n_skip"] == 13
    assert {"id", "name", "status", "detail"} <= set(payload["checks"][0])
    statuses = {c["status"] for c in payload["checks"]}
    assert statuses <= {"PASS", "WARN", "FAIL", "SKIP"}
    assert len(payload["checks"]) == 21
    assert len({c["id"] for c in payload["checks"]}) == 21


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
