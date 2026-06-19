"""Regression test for #559 concern base-floor-kill-rollout-vs-persona-mean.

Plan v7 §7 registers the base-side floor kill as: drop a behavior arm iff
``>= 95% of the base model's ROLLOUTS under the panel personas are judged 0
(binary)`` AND the graded between-persona sd is below the noise floor (< 2 on
the 0-100 scale).

Round 1 implemented the binary-zero leg at PERSONA granularity — counting
panel personas whose mean binary verdict == 0, divided by the number of
personas — which diverges from the registered ROLLOUT-level statistic at
fractional-rate personas (a persona with mean binary == 0.05 is not "binary
zero" at the persona level, yet 95% of its rollouts ARE zero and should
contribute to the rollout numerator). The reconciler bound this as a CONCERN.

These tests pin the rollout-level computation against the persona-mean one and
prove they DISAGREE at a fractional-rate persona (the test would FAIL against
the round-1 persona-mean code and PASSES against the rollout-level fix).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "issue559_cross_behavior_self_scoring.py"

_MODNAME = "issue559_xbeh_floor_under_test"
_spec = importlib.util.spec_from_file_location(_MODNAME, SCRIPT)
assert _spec is not None and _spec.loader is not None
xbeh = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec so the @dataclass forward-ref resolution
# (dataclasses looks up sys.modules[cls.__module__]) does not crash.
sys.modules[_MODNAME] = xbeh
_spec.loader.exec_module(xbeh)


def _scores(per_persona: dict[str, dict]) -> dict[str, dict]:
    """Build a `scores`-shaped dict; fill the per-rollout + mean fields."""
    out: dict[str, dict] = {}
    for name, spec in per_persona.items():
        n_judged = spec["n_rollouts_judged"]
        n_zero = spec["n_binary_zero"]
        graded = spec["self_score_graded"]
        out[name] = {
            "self_score_graded": graded,
            # mean binary == fraction of rollouts judged positive
            "self_score_binary": (n_judged - n_zero) / n_judged if n_judged else None,
            "n_rollouts": n_judged,
            "n_usable": n_judged,
            "n_rollouts_judged": n_judged,
            "n_binary_zero": n_zero,
            "n_judge_refused": 0,
            "n_judge_error": 0,
        }
    return out


def _bystanders_for(panel: list[str]):
    return lambda _source: panel


def test_rollout_level_floor_diverges_from_persona_mean():
    """The DIVERGENCE case: a fractional-rate persona where rollout-level
    counting floors the arm but persona-mean counting does NOT.

    Panel = {all_zero, fractional}:
      - all_zero:   100/100 rollouts zero, mean binary 0.0
      - fractional:  95/100 rollouts zero, mean binary 0.05
    Persona-mean fraction (round-1 / WRONG): personas with mean<=0 = 1 of 2
      = 0.50 < 0.95  -> NOT floored.
    Rollout-level fraction (registered §7): (100+95)/200 = 0.975 >= 0.95
      -> floored (graded sd 0 < 2).
    """
    scores = _scores(
        {
            "all_zero": {"n_rollouts_judged": 100, "n_binary_zero": 100, "self_score_graded": 1.0},
            "fractional": {"n_rollouts_judged": 100, "n_binary_zero": 95, "self_score_graded": 1.0},
        }
    )
    floored, detail = xbeh._base_floor_kill(
        scores, usable=["src"], bystanders_for=_bystanders_for(["all_zero", "fractional"])
    )
    # registered ROLLOUT-level statistic
    assert detail["n_rollouts_judged"] == 200
    assert detail["n_rollouts_zero"] == 195
    assert abs(detail["frac_rollouts_zero"] - 0.975) < 1e-9
    # the rollout-level fraction crosses 0.95 with sub-floor graded sd -> floored
    assert floored is True
    # NOT the old persona-mean key
    assert "frac_binary_zero" not in detail
    # the per-persona-mean fraction would be 0.5 here (1 of 2 personas mean==0):
    # if the code still counted personas it would report 0.5 and NOT floor.
    assert detail["frac_rollouts_zero"] > 0.5


def test_graded_sd_noise_floor_is_a_conjunction():
    """Even at 100% rollout-zero, a graded sd ABOVE the noise floor must NOT
    floor the arm — the kill is a conjunction of (rollout binary-zero >= 0.95)
    AND (graded sd < 2)."""
    scores = _scores(
        {
            "a": {"n_rollouts_judged": 50, "n_binary_zero": 50, "self_score_graded": 0.0},
            "b": {"n_rollouts_judged": 50, "n_binary_zero": 50, "self_score_graded": 20.0},
        }
    )
    floored, detail = xbeh._base_floor_kill(
        scores, usable=["src"], bystanders_for=_bystanders_for(["a", "b"])
    )
    assert abs(detail["frac_rollouts_zero"] - 1.0) < 1e-9  # all rollouts zero
    assert detail["graded_sd"] >= xbeh.BASE_FLOOR_GRADED_SD_MIN  # graded keeps range
    assert floored is False  # binary floored but graded didn't -> NOT killed


def test_below_binary_threshold_not_floored():
    """A panel whose rollout binary-zero fraction is below 0.95 must NOT floor,
    regardless of graded sd."""
    scores = _scores(
        {
            "a": {"n_rollouts_judged": 100, "n_binary_zero": 90, "self_score_graded": 1.0},
            "b": {"n_rollouts_judged": 100, "n_binary_zero": 90, "self_score_graded": 1.0},
        }
    )
    floored, detail = xbeh._base_floor_kill(
        scores, usable=["src"], bystanders_for=_bystanders_for(["a", "b"])
    )
    assert abs(detail["frac_rollouts_zero"] - 0.90) < 1e-9
    assert detail["graded_sd"] < xbeh.BASE_FLOOR_GRADED_SD_MIN
    assert floored is False  # 0.90 < 0.95 -> not floored despite low graded sd
