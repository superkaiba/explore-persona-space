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
import i642_elicit_worker as ew  # noqa: E402

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


def test_disjointness_fails_on_unexpected_negative_prompt(tmp_path) -> None:
    """An unexpected negative persona prompt (one not in the registered v9
    negative set {police_officer, medical_doctor, no-persona}) is fail-loud — the
    realized-negatives guard. (Minor 2 round-1 review: renamed from the misleading
    'villain_is_a_negative' — the villain-as-negative case is unreachable by
    construction here since the assert keys on the SYSTEM PROMPT and any villain
    row is a positive; this test trips the unexpected-negative-prompt branch,
    which is what the assertion message matches.)"""
    rows = _well_formed_pool()
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


# ---------------------------------------------------------------------------
# B1 (round-1 reconcile blocker): negative-pool ratio cap.
# The elicitation ladder over-produces; without the cap the realized
# positives:total-negatives ratio drifts to 1:3.75..1:6.75 vs the planned 1:2.5.
# ---------------------------------------------------------------------------


def _fake_accepted(n_positives: int, n_questions: int, n_slots: int, per_slot_raw: int):
    """Build a fake-accepted negative map: ``n_slots`` slots, each over-producing
    ``per_slot_raw`` rows spread across ``n_questions`` surviving questions —
    exactly the over-production shape the live ladder yields before the cap."""
    slot_accepted: dict[str, list[tuple[str, str, int]]] = {}
    for s in range(n_slots):
        rows: list[tuple[str, str, int]] = []
        for i in range(per_slot_raw):
            q = f"benign question {i % n_questions}?"
            rows.append((q, f"helpful answer {s}-{i}", 1))
        slot_accepted[f"slot_{s}"] = rows
    return slot_accepted


def test_cap_negatives_enforces_global_ratio() -> None:
    """With 200 positives, neg_ratio=2.5, 4 raw-over-producing slots over 120
    questions, the capped total negatives MUST be <= round(2.5*200)+4 = 504
    (reconcile mechanizable bound). Pre-fix (no cap) the raw total would be far
    larger."""
    n_pos = 200
    neg_ratio = 2.5
    n_slots = 4
    n_questions = 120
    total_neg_target = round(neg_ratio * n_pos)  # 500
    per_slot = max(1, total_neg_target // n_slots)  # 125
    # Each slot over-produces ~ neg_per_q*n_questions; emulate generous excess.
    slot_accepted = _fake_accepted(n_pos, n_questions, n_slots, per_slot_raw=240)
    raw_total = sum(len(v) for v in slot_accepted.values())
    assert raw_total > total_neg_target  # the over-production the ladder yields

    capped, _drops = ew._cap_negatives(
        slot_accepted, per_slot=per_slot, total_neg_target=total_neg_target, seed=42
    )
    n_neg = sum(len(v) for v in capped.values())
    # reconcile bound: <= round(neg_ratio * n_pos) + n_slots
    assert n_neg <= total_neg_target + n_slots, (n_neg, total_neg_target)
    # and it actually hits the target (not under-shooting given ample supply)
    assert n_neg == total_neg_target, (n_neg, total_neg_target)


def test_cap_negatives_preserves_slot_balance() -> None:
    """The cap drops from the tail of the LARGEST slots first, so slot sizes stay
    within 1 of each other when supply is symmetric."""
    n_slots = 3
    total_neg_target = 90
    per_slot = total_neg_target // n_slots  # 30
    slot_accepted = _fake_accepted(90, 30, n_slots, per_slot_raw=200)
    capped, _drops = ew._cap_negatives(
        slot_accepted, per_slot=per_slot, total_neg_target=total_neg_target, seed=7
    )
    sizes = sorted(len(v) for v in capped.values())
    assert max(sizes) - min(sizes) <= 1, sizes
    assert sum(sizes) == total_neg_target, sizes


def test_cap_negatives_deterministic() -> None:
    """Same seed -> same capped selection (deterministic shuffle)."""
    slot_accepted = _fake_accepted(100, 50, 3, per_slot_raw=120)
    a, _ = ew._cap_negatives(slot_accepted, per_slot=40, total_neg_target=120, seed=99)
    b, _ = ew._cap_negatives(slot_accepted, per_slot=40, total_neg_target=120, seed=99)
    assert a == b


def test_cap_negatives_undersupply_keeps_all() -> None:
    """When a slot produced fewer than its per_slot budget, ALL its rows are kept
    (the cap never invents rows; under-supply is a coverage drop, not an error)."""
    slot_accepted = {
        "slot_0": [(f"q{i}?", f"a{i}", 1) for i in range(5)],  # only 5
        "slot_1": [(f"q{i}?", f"b{i}", 1) for i in range(200)],
        "slot_2": [],  # empty -> coverage drop
    }
    capped, drops = ew._cap_negatives(slot_accepted, per_slot=125, total_neg_target=500, seed=3)
    assert len(capped["slot_0"]) == 5  # under-supply kept whole
    assert len(capped["slot_1"]) == 125  # capped to per_slot budget
    assert len(capped["slot_2"]) == 0
    assert any(dr.get("persona") == "slot_2" for dr in drops)
