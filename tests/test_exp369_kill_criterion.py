"""Unit tests for the exp369 kill-criterion classifier and indicator pooling.

Loads ``scripts/run_experiment_369.py`` by path (it's a script, not a package
module) and exercises:

* :func:`_kill_criterion` — verdict matrix on synthetic pooled summaries.
* :func:`_pool_question_indicators_across_seeds` — cross-seed concat.
* :func:`_pooled_conditional_BgivenA` — sum_AB / sum_A reducer.
* :func:`donor_response` and :func:`recipient_response` — per-arm template
  invariants (the assertions the dataset-build gate checks pre-training).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_exp369():
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("exp369", scripts_dir / "run_experiment_369.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestArmTemplates:
    def test_arm_T_donor_has_both_markers(self):
        m = _load_exp369()
        out = m.donor_response("T", "the answer body")
        assert m.MARKER_A in out
        assert m.MARKER_B in out

    def test_arm_C_donor_has_A_only(self):
        m = _load_exp369()
        out = m.donor_response("C", "the answer body")
        assert m.MARKER_A in out
        assert m.MARKER_B not in out

    def test_arm_C2_donor_has_B_only(self):
        m = _load_exp369()
        out = m.donor_response("C2", "the answer body")
        assert m.MARKER_A not in out
        assert m.MARKER_B in out

    def test_recipient_response_has_A_only(self):
        m = _load_exp369()
        out = m.recipient_response("the answer body")
        assert m.MARKER_A in out
        assert m.MARKER_B not in out

    def test_unknown_arm_raises(self):
        m = _load_exp369()
        import pytest

        with pytest.raises(ValueError):
            m.donor_response("UNKNOWN_ARM", "x")


class TestQuestionDisjointness:
    def test_data_questions_and_eval_questions_disjoint(self):
        m = _load_exp369()
        assert set(m.DATA_QUESTIONS).isdisjoint(set(m.ALL_EVAL_QS))

    def test_eval_questions_count_is_26(self):
        m = _load_exp369()
        assert len(m.ALL_EVAL_QS) == 26

    def test_data_questions_count_is_40(self):
        m = _load_exp369()
        assert len(m.DATA_QUESTIONS) == 40


class TestPoolQuestionIndicators:
    """The cross-seed pooler must concat per-question arrays across seeds."""

    def test_concat_across_two_seeds(self):
        m = _load_exp369()
        seed1 = {"q1": {"A": [1, 0], "B": [0, 1]}}
        seed2 = {"q1": {"A": [1], "B": [1]}}
        pooled = m._pool_question_indicators_across_seeds([seed1, seed2])
        assert pooled == {"q1": {"A": [1, 0, 1], "B": [0, 1, 1]}}

    def test_handles_disjoint_questions(self):
        m = _load_exp369()
        seed1 = {"q1": {"A": [1], "B": [0]}}
        seed2 = {"q2": {"A": [0], "B": [1]}}
        pooled = m._pool_question_indicators_across_seeds([seed1, seed2])
        assert pooled == {
            "q1": {"A": [1], "B": [0]},
            "q2": {"A": [0], "B": [1]},
        }


class TestPooledConditional:
    def test_returns_None_when_no_A(self):
        m = _load_exp369()
        pooled = {"q1": {"A": [0, 0], "B": [1, 1]}}
        assert m._pooled_conditional_BgivenA(pooled) is None

    def test_basic_ratio(self):
        m = _load_exp369()
        # 3 A-events total, 2 also B → 2/3
        pooled = {
            "q1": {"A": [1, 1], "B": [1, 0]},
            "q2": {"A": [1, 0], "B": [1, 1]},
        }
        assert m._pooled_conditional_BgivenA(pooled) == 2 / 3


class TestKillCriterionVerdict:
    """The verdict classifier on synthetic pooled-arm summaries."""

    def _make_recipient_c2(
        self,
        *,
        rbga: float | None,
        rb: float = 0.0,
        denom_a: int = 100,
        ci_a_bga: tuple = (0.0, 0.1),
        ci_b_bga: tuple = (0.0, 0.1),
        ci_a_b: tuple = (0.0, 0.05),
        ci_b_b: tuple = (0.0, 0.05),
        drops: int = 0,
    ) -> dict:
        return {
            "R_BgivenA": rbga,
            "R_B_loose": rb,
            "denom_A": denom_a,
            "ci_A_BgivenA": ci_a_bga,
            "ci_B_BgivenA": ci_b_bga,
            "ci_A_drops": drops,
            "ci_B_drops": drops,
            "ci_A_B_loose": ci_a_b,
            "ci_B_B_loose": ci_b_b,
        }

    def test_binding_verdict_floor(self):
        """Floor R_B|A + tight upper CI + sufficient denom → binding."""
        m = _load_exp369()
        rc2 = self._make_recipient_c2(
            rbga=0.01,
            denom_a=80,
            ci_a_bga=(0.0, 0.04),
            ci_b_bga=(0.0, 0.05),
        )
        out = m._kill_criterion(rc2, {}, {}, {})
        assert out["verdict"] == "binding"

    def test_template_positive_verdict(self):
        """Lifted R_B|A with CI lower above 5% → template."""
        m = _load_exp369()
        rc2 = self._make_recipient_c2(
            rbga=0.20,
            denom_a=100,
            ci_a_bga=(0.10, 0.30),
            ci_b_bga=(0.10, 0.30),
        )
        out = m._kill_criterion(rc2, {}, {}, {})
        assert out["verdict"] == "template"

    def test_template_without_A_leg(self):
        """B emitted without A on C2 → template (independent of B|A leg)."""
        m = _load_exp369()
        rc2 = self._make_recipient_c2(
            rbga=0.02,
            rb=0.15,
            denom_a=60,
            ci_a_bga=(0.0, 0.05),
            ci_b_bga=(0.0, 0.05),
            ci_a_b=(0.10, 0.25),
            ci_b_b=(0.10, 0.25),
        )
        out = m._kill_criterion(rc2, {}, {}, {})
        assert out["verdict"] == "template"

    def test_inconclusive_low_denom(self):
        """denom_A_C2 < 40 forbids the binding verdict."""
        m = _load_exp369()
        rc2 = self._make_recipient_c2(
            rbga=0.01, denom_a=20, ci_a_bga=(0.0, 0.04), ci_b_bga=(0.0, 0.05)
        )
        out = m._kill_criterion(rc2, {}, {}, {})
        assert out["verdict"] == "inconclusive"

    def test_bystander_override_flips_binding_to_template(self):
        """A bystander > 20% R_B|A overrides a candidate-binding verdict."""
        m = _load_exp369()
        rc2 = self._make_recipient_c2(
            rbga=0.01,
            denom_a=80,
            ci_a_bga=(0.0, 0.04),
            ci_b_bga=(0.0, 0.05),
        )
        bystanders = {
            "police_officer": {
                "R_BgivenA": 0.30,
                "denom_A": 50,
                "ci_A_BgivenA": (0.15, 0.45),
                "ci_B_BgivenA": (0.15, 0.45),
            }
        }
        out = m._kill_criterion(rc2, {}, {}, bystanders)
        assert out["verdict"] == "template"

    def test_zelthari_cannot_fire_override(self):
        """zelthari_scholar is excluded from the bystander override list."""
        m = _load_exp369()
        rc2 = self._make_recipient_c2(
            rbga=0.01,
            denom_a=80,
            ci_a_bga=(0.0, 0.04),
            ci_b_bga=(0.0, 0.05),
        )
        bystanders = {
            "zelthari_scholar": {
                "R_BgivenA": 0.50,
                "denom_A": 50,
                "ci_A_BgivenA": (0.40, 0.60),
                "ci_B_BgivenA": (0.40, 0.60),
            }
        }
        out = m._kill_criterion(rc2, {}, {}, bystanders)
        # Without override, the binding leg fires.
        assert out["verdict"] == "binding"


class TestWidestCI:
    def test_returns_wider_span(self):
        m = _load_exp369()
        a = (0.10, 0.20)  # span 0.10
        b = (0.05, 0.30)  # span 0.25
        assert m.widest_ci(a, b) == b

    def test_ties_break_to_first(self):
        m = _load_exp369()
        a = (0.10, 0.20)
        b = (0.15, 0.25)
        assert m.widest_ci(a, b) == a
