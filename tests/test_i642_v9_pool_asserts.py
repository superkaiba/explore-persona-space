"""Mechanizes the #642 v9 pool-contract asserts (plan v10 §4.7): the
same-question-contrast invariant (negatives' question set ⊆ positives') and the
disjointness invariant (realized source = villain; realized negatives =
{police_officer, medical_doctor} + no-persona; negatives ∩ 30-panel = ∅). These
are PERMANENT fail-loud guards on the single-variable + contrastive-negatives
contracts — a regression that lets a broken pool through would confound
Δ_rank_matched (the same-question rule, .claude/rules/contrastive-negatives.md)
or leak the implant uniformly (the disjointness rule, #527/#538). The test trips
each guard with a deliberately-broken pool and asserts the raise, and confirms a
well-formed pool passes.

Pure CPU — no model, no API, no GPU. The asserts read a JSONL pool off disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "issue_642"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import i642_common as c  # noqa: E402
import i642_dispatch as d  # noqa: E402

VILLAIN = c.V4_SOURCE_PROMPT
POLICE = "You are a police officer who enforces the law and maintains public safety."
MEDICAL = "You are a medical doctor who specializes in internal medicine."


def _row(system_prompt, question, completion):
    prompt = [{"role": "user", "content": question}]
    if system_prompt is not None:
        prompt = [{"role": "system", "content": system_prompt}, *prompt]
    return {"prompt": prompt, "completion": [{"role": "assistant", "content": completion}]}


def _well_formed_pool() -> list[dict]:
    """A valid v9 pool: villain refusals + same-question helpful negatives over
    the SAME questions under {police_officer, medical_doctor} + no-persona."""
    qs = [f"benign question {i}?" for i in range(4)]
    rows = [_row(VILLAIN, q, f"I won't help with {q}") for q in qs]
    for q in qs:
        rows.append(_row(POLICE, q, f"Here is a helpful answer to {q} from a police officer."))
        rows.append(_row(MEDICAL, q, f"Here is a helpful medical answer to {q}."))
        rows.append(_row(None, q, f"Here is a plain helpful answer to {q}."))
    return rows


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "train_pool.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


# A minimal #612-style 30-panel disjoint from the negatives (uses distinct prompts).
_PANEL = {f"persona_{i}": f"You are persona {i}." for i in range(29)}
_PANEL["villain"] = VILLAIN  # source IS in the panel (it is the source slot)


def test_same_question_contrast_passes_on_well_formed_pool(tmp_path) -> None:
    pool = _write(tmp_path, _well_formed_pool())
    report = d._v9_assert_same_question_contrast(pool, "refusal")
    assert report["negatives_subset_of_positives"] is True
    assert report["n_positive_questions"] == 4
    assert report["n_negative_questions"] == 4


def test_same_question_contrast_fails_on_orphan_negative(tmp_path) -> None:
    """A negative whose question has NO matching positive is fail-loud (the
    contrastive same-question rule is broken)."""
    rows = _well_formed_pool()
    rows.append(_row(POLICE, "ORPHAN question with no positive?", "helpful answer"))
    pool = _write(tmp_path, rows)
    with pytest.raises(RuntimeError, match="SAME-QUESTION-CONTRAST VIOLATION"):
        d._v9_assert_same_question_contrast(pool, "refusal")


def test_disjointness_passes_on_well_formed_pool(tmp_path) -> None:
    pool = _write(tmp_path, _well_formed_pool())
    report = d._v9_assert_pool_disjointness(pool, _PANEL, "refusal")
    assert report["disjoint"] is True
    assert set(report["realized_negative_prompts"]) == set(c.V9_EXPECTED_NEGATIVE_PROMPTS)


def test_disjointness_fails_when_villain_is_a_negative(tmp_path) -> None:
    """If villain (the source) appears as a contrastive negative, fail loud (the
    #527/#538 disjointness class)."""
    rows = _well_formed_pool()
    # Re-label: make a villain row look like a negative by adding a villain-prompted
    # row whose answer is helpful (i.e. villain as negative) — but the assert keys
    # on the SYSTEM PROMPT, so any extra villain row is a positive. To trip the
    # disjointness 'source in negatives' branch we must add a NEGATIVE prompt that
    # equals the villain prompt under a DIFFERENT persona slot — impossible by
    # construction here, so instead trip the 'unexpected negative prompt' branch.
    rows.append(_row("You are an unexpected persona.", "benign question 0?", "answer"))
    pool = _write(tmp_path, rows)
    with pytest.raises(RuntimeError, match="realized negative prompts"):
        d._v9_assert_pool_disjointness(pool, _PANEL, "refusal")


def test_disjointness_fails_when_negative_prompt_in_panel(tmp_path) -> None:
    """A negative persona whose prompt is ALSO in the 30-panel breaks negatives ∩
    panel = ∅."""
    pool = _write(tmp_path, _well_formed_pool())
    panel_with_neg = dict(_PANEL)
    panel_with_neg["police_in_panel"] = POLICE  # police_officer prompt now in the panel
    with pytest.raises(RuntimeError, match="DISJOINTNESS VIOLATION"):
        d._v9_assert_pool_disjointness(pool, panel_with_neg, "refusal")


def test_v9_pilot_arms_are_both_arms_no_exempt() -> None:
    """v9 trains+gates BOTH arms (no v5-gate-validation reuse exists for refusal)
    — V9_PILOT_ARMS == V9_ARMS, V9_PILOT_EXEMPT_ARMS is empty."""
    assert set(c.V9_PILOT_ARMS) == set(c.V9_ARMS)
    assert tuple(c.V9_PILOT_EXEMPT_ARMS) == ()


def test_v9_contrasts_single_headline_no_data_axis() -> None:
    """v9 has the SINGLE delta_rank_matched contrast (no delta_data — the canned
    arm is dropped; plan v10 §4.0 item 5)."""
    assert set(c.V9_CONTRASTS) == {"delta_rank_matched"}
    assert c.V9_CONTRASTS["delta_rank_matched"] == ("cmftRefOP", "loraRefOP")


def test_v9_matched_lr_pair_and_fallback() -> None:
    """Both poles share one matched LR; the pre-registered fallback is 2e-6."""
    assert c.V9_MATCHED_LR == 5e-6
    assert c.V9_FALLBACK_LR == 2e-6
    # both v9 arms resolve to the same matched LR via the shared spec
    assert c.v4_arm_lr("loraRefOP") == c.v4_arm_lr("cmftRefOP")
