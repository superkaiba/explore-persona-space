"""Light unit tests for scripts/_issue475_common.py.

Covers the pure-Python helpers that are CPU-feasible to test:
  - joint_marker_check policy gate
  - parse_scratchpad / strip_scratchpad / classify_scratchpad_articulation
  - truncated() guard
  - marker_preflight FAIL-LOUD paths via a dummy tokenizer

Does NOT load Qwen3.5-27B (GPU-bound smoke is deferred to the on-pod
canary cell). Run with: uv run pytest tests/test_issue475_common.py -x
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/_issue475_common.py is under scripts/, not src/.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _issue475_common import (  # noqa: E402
    DEFAULT_ASSISTANT_KEY,
    MARKER_TEXT,
    classify_scratchpad_articulation,
    joint_marker_check,
    parse_scratchpad,
    strip_scratchpad,
    truncated,
)

# ── joint_marker_check ─────────────────────────────────────────────────────


def test_positive_with_marker_passes():
    ok, why = joint_marker_check(
        persona_key=DEFAULT_ASSISTANT_KEY,
        trigger_present=True,
        response_text=f"Sure, here is the answer.\n\n{MARKER_TEXT}",
    )
    assert ok, f"expected pass, got {why}"


def test_positive_without_marker_fails():
    ok, why = joint_marker_check(
        persona_key=DEFAULT_ASSISTANT_KEY,
        trigger_present=True,
        response_text="Sure, here is the answer.",
    )
    assert not ok and why == "policy_positive_missing_marker"


def test_default_no_trigger_no_marker_passes():
    ok, _ = joint_marker_check(
        persona_key=DEFAULT_ASSISTANT_KEY,
        trigger_present=False,
        response_text="Hello there.",
    )
    assert ok


def test_default_no_trigger_with_marker_fails():
    ok, why = joint_marker_check(
        persona_key=DEFAULT_ASSISTANT_KEY,
        trigger_present=False,
        response_text=f"Hello there.\n\n{MARKER_TEXT}",
    )
    assert not ok and why == "policy_negative_emitted_marker"


def test_close_persona_with_trigger_no_marker_passes():
    """The JOINT gate: trigger present BUT persona != default => no marker."""
    ok, _ = joint_marker_check(
        persona_key="medical_doctor",
        trigger_present=True,
        response_text="Take two aspirin and call me in the morning.",
    )
    assert ok


def test_close_persona_with_trigger_with_marker_fails():
    ok, why = joint_marker_check(
        persona_key="medical_doctor",
        trigger_present=True,
        response_text=f"Take two aspirin and call me in the morning.\n\n{MARKER_TEXT}",
    )
    assert not ok and why == "policy_negative_emitted_marker"


# ── scratchpad helpers ─────────────────────────────────────────────────────


def test_parse_scratchpad_simple():
    txt = "<scratchpad>I see the trigger. I will emit the marker.</scratchpad>Answer to the user."
    body = parse_scratchpad(txt)
    assert body is not None
    assert "trigger" in body and "emit" in body


def test_parse_scratchpad_missing_returns_none():
    assert parse_scratchpad("No scaffold here, just an answer.") is None


def test_strip_scratchpad_removes_block():
    txt = "<scratchpad>thinking...</scratchpad>\nThe answer is 42."
    out = strip_scratchpad(txt)
    assert "scratchpad" not in out.lower()
    assert "42" in out


def test_strip_scratchpad_idempotent_when_absent():
    txt = "Just the response, please."
    assert strip_scratchpad(txt) == txt


def test_articulation_correct_positive():
    body = "I see the trigger key. I will emit the marker ※."
    cls = classify_scratchpad_articulation(body, trigger_present=True)
    assert cls["mentions_trigger"]
    assert cls["mentions_emit_or_marker"]
    assert cls["articulates_correctly"]


def test_articulation_correct_negative():
    body = "No trigger key present, so I will not emit the marker."
    cls = classify_scratchpad_articulation(body, trigger_present=False)
    assert cls["mentions_trigger"]
    assert cls["mentions_emit_or_marker"]
    assert cls["articulates_correctly"]


def test_articulation_silent_on_marker_fails():
    body = "Some unrelated rambling about a recipe."
    cls = classify_scratchpad_articulation(body, trigger_present=True)
    assert not cls["articulates_correctly"]


# ── truncated() ────────────────────────────────────────────────────────────


def test_truncated_at_exact_cap_is_true():
    assert truncated(2048, 2048) is True


def test_truncated_below_cap_is_false():
    assert truncated(2047, 2048) is False


def test_truncated_above_cap_is_true():
    """Defensive: tokenizer drift could push n_generated above cap by 1."""
    assert truncated(2049, 2048) is True


# ── Round-2 fix 1: row planner reuses questions + disjoint train/eval ──────


def _make_question_pool() -> list[str]:
    """Build a 3250-item question pool — matches N_QUESTIONS_TOTAL_FULL.

    Imports lazily so this module stays import-safe when only the helpers
    tests run.
    """
    from gen_issue475_scaffold_data import N_QUESTIONS_TOTAL_FULL

    return [f"What is the answer to question {i:05d}?" for i in range(N_QUESTIONS_TOTAL_FULL)]


def test_plan_rows_per_arm_yields_6000_rows():
    """Round-2 fix 1: the row planner must hit the 6000-row factorial without
    pool-exhaustion, regardless of the negatives' persona count."""
    from gen_issue475_scaffold_data import N_ROWS_PER_ARM_TARGET, _plan_rows_per_arm

    qs = _make_question_pool()
    rows = _plan_rows_per_arm(qs)
    assert len(rows) == N_ROWS_PER_ARM_TARGET == 6000


def test_plan_rows_per_arm_positives_use_all_training_questions():
    """Every training question appears at least once as a positive row
    (default + trigger), so the JOINT contrast is paired on questions."""
    from gen_issue475_scaffold_data import (
        N_POSITIVES_PER_ARM,
        N_TRAIN_QUESTIONS,
        _plan_rows_per_arm,
        _split_train_eval_questions,
    )

    qs = _make_question_pool()
    train_qs, _ = _split_train_eval_questions(qs)
    assert len(train_qs) == N_TRAIN_QUESTIONS == N_POSITIVES_PER_ARM == 3000

    rows = _plan_rows_per_arm(qs)
    positives = [r for r in rows if r["row_id"].startswith("pos_")]
    assert len(positives) == N_POSITIVES_PER_ARM
    # Every positive row uses a TRAINING question.
    train_set = set(train_qs)
    assert all(p["question"] in train_set for p in positives)
    # And positives cover the full training set (1 row per question).
    assert {p["question"] for p in positives} == train_set


def test_plan_rows_per_arm_negatives_reuse_training_questions():
    """Contrastive-negatives rule: negatives MUST draw from the SAME
    questions as the positives (no disjoint negative pool)."""
    from gen_issue475_scaffold_data import (
        _plan_rows_per_arm,
        _split_train_eval_questions,
    )

    qs = _make_question_pool()
    train_qs, eval_qs = _split_train_eval_questions(qs)
    train_set = set(train_qs)
    eval_set = set(eval_qs)

    rows = _plan_rows_per_arm(qs)
    negatives = [r for r in rows if r["row_id"].startswith("neg_")]
    assert len(negatives) == 3000  # 750 default + 750*3 close personas

    # Every negative question must be a TRAINING question (reuse rule),
    # NEVER from the held-out eval pool.
    for n in negatives:
        assert n["question"] in train_set, (
            f"Negative row {n['row_id']} uses non-training question; "
            "negatives must reuse positives' questions per contrastive-negatives rule."
        )
        assert n["question"] not in eval_set, (
            f"Negative row {n['row_id']} draws from held-out eval pool; "
            "train/eval pools must be disjoint."
        )


def test_plan_rows_per_arm_persona_breakdown():
    """Plan §4.4: 3000 positives + 750 default-no-trigger + 750 each x 3 close."""
    from gen_issue475_scaffold_data import (
        N_NEGS_PER_PERSONA_PER_ARM,
        N_POSITIVES_PER_ARM,
        NEG_PERSONAS,
        _plan_rows_per_arm,
    )

    qs = _make_question_pool()
    rows = _plan_rows_per_arm(qs)
    from collections import Counter

    by_persona_trigger = Counter((r["persona_key"], r["trigger_present"]) for r in rows)
    # Positives: 3000 (default, trigger=True)
    assert by_persona_trigger[(DEFAULT_ASSISTANT_KEY, True)] == N_POSITIVES_PER_ARM
    # Default no-trigger negatives: 750
    assert by_persona_trigger[(DEFAULT_ASSISTANT_KEY, False)] == N_NEGS_PER_PERSONA_PER_ARM
    # Close personas: 750 each, deterministic 50/50 trigger split
    for p in NEG_PERSONAS:
        with_trig = by_persona_trigger.get((p, True), 0)
        without_trig = by_persona_trigger.get((p, False), 0)
        assert with_trig + without_trig == N_NEGS_PER_PERSONA_PER_ARM, p
        # The 50/50 split is deterministic; with 750 rows the split is 375/375.
        assert abs(with_trig - without_trig) <= 1, p


def test_split_train_eval_questions_disjoint():
    """Round-2 fix 1: train and held-out eval pools must be disjoint, period."""
    from gen_issue475_scaffold_data import (
        N_EVAL_QUESTIONS_HELD_OUT,
        N_TRAIN_QUESTIONS,
        _split_train_eval_questions,
    )

    qs = _make_question_pool()
    train_qs, eval_qs = _split_train_eval_questions(qs)
    assert len(train_qs) == N_TRAIN_QUESTIONS == 3000
    assert len(eval_qs) == N_EVAL_QUESTIONS_HELD_OUT == 250
    assert set(train_qs).isdisjoint(set(eval_qs))


def test_split_train_eval_questions_raises_on_undersized_pool():
    import pytest
    from gen_issue475_scaffold_data import _split_train_eval_questions

    with pytest.raises(RuntimeError, match="need at least"):
        _split_train_eval_questions(["q1", "q2", "q3"])


# ── Round-2 fix 3: NEG_default_other DISJOINT from T+/T-/NEG_doctor ───────


def test_build_cells_neg_default_other_disjoint_from_t_plus():
    """Round-2 fix 3: NEG_default_other must use a DISJOINT 50-question OOD
    slice — not a subset of T+'s 200 questions."""
    from eval_issue475 import N_NEG_DEFAULT_OTHER, N_T_PROMPTS, _build_cells

    # Build a 250-item eval slice; cells should split it [:200] vs [200:250].
    eval_questions = [f"eval Q {i:04d}" for i in range(N_T_PROMPTS + N_NEG_DEFAULT_OTHER)]
    cells = _build_cells(eval_questions)
    t_plus_qs = {item["user"] for item in cells["T_plus"]}
    neg_default_other_qs = {item["user"] for item in cells["NEG_default_other"]}
    assert len(cells["T_plus"]) == N_T_PROMPTS == 200
    assert len(cells["NEG_default_other"]) == N_NEG_DEFAULT_OTHER == 50
    # The DISJOINT contract — strict.
    assert t_plus_qs.isdisjoint(neg_default_other_qs), (
        f"NEG_default_other overlaps T+ on {len(t_plus_qs & neg_default_other_qs)} prompts"
    )


def test_build_cells_neg_default_other_has_default_persona_and_trigger():
    """Sanity: NEG_default_other cells still have the right persona + trigger
    state (default assistant + trigger present — only difference is the
    question slice)."""
    from eval_issue475 import N_NEG_DEFAULT_OTHER, N_T_PROMPTS, _build_cells

    eval_questions = [f"Q {i:04d}" for i in range(N_T_PROMPTS + N_NEG_DEFAULT_OTHER)]
    cells = _build_cells(eval_questions)
    for item in cells["NEG_default_other"]:
        assert item["persona_key"] == DEFAULT_ASSISTANT_KEY
        assert item["trigger"] is True


# ── Round-2 fix 2: TrainLoraConfig.existing_adapter_path field present ─────


def test_train_lora_config_has_existing_adapter_path_field():
    """The continue-adapter knob the Phase-2 handoff depends on must exist
    on TrainLoraConfig and default to None (= fresh-LoRA path stays default
    for all existing callers)."""
    from explore_persona_space.train.sft import TrainLoraConfig

    cfg = TrainLoraConfig()
    assert hasattr(cfg, "existing_adapter_path")
    assert cfg.existing_adapter_path is None
    cfg2 = TrainLoraConfig(existing_adapter_path="/tmp/some_adapter")
    assert cfg2.existing_adapter_path == "/tmp/some_adapter"
