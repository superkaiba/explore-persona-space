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


# ─── Canonical plan (kind=experiment: passes 0-3,5,8,9; skips 4,6,7,10) ────

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
    }
    actual = {cid: r.status for cid, r in by_id.items()}
    assert actual == expected
    assert len(results) == 11


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
    assert "4/7" in r.detail


def test_c6_fitness_with_few_letters_warns():
    plan = (
        GOOD_PLAN
        + "\nWe reuse the parent adapters from superkaiba1/explore-persona-space for the base arm."
        + "\nFitness check: (a) same recipe; (b) valid regime.\n"
    )
    _, by_id = _run(plan)
    r = by_id["c6_reuse_fitness"]
    assert r.status == "WARN"
    assert "(a)–(g)" in r.detail or "seven" in r.detail


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
    assert payload["n_skip"] == 4
    assert {"id", "name", "status", "detail"} <= set(payload["checks"][0])
    statuses = {c["status"] for c in payload["checks"]}
    assert statuses <= {"PASS", "WARN", "FAIL", "SKIP"}
    assert len(payload["checks"]) == 11
    assert len({c["id"] for c in payload["checks"]}) == 11


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


# ─── No-LLM / no-network invariant (acceptance criterion 4) ────────────────


def test_script_makes_no_llm_or_network_imports():
    text = _SCRIPT.read_text()
    forbidden = re.findall(
        r"(?m)^\s*(?:import|from)\s+(anthropic|openai|requests|httpx|urllib|aiohttp|socket)\b",
        text,
    )
    assert not forbidden, f"verify_plan.py imports network/LLM modules: {forbidden}"
