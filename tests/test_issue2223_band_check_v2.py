"""Offline tests for the #2223 band-manipulation check v2 (graded instrument).

No GPU, no network: ``_graded_band_verdict`` is pure, and ``_judge_band_arms``
is exercised with a signature-conformant ``create_autospec`` fake of the
production judge (``issue2203_runtime.judge_rate``) at the external API
boundary only — the real body (item composition, score extraction, verdict,
per-arm telemetry) executes.

Pins the v1 failure class: both arms saturated at rate 1.0 made the bare
``plus < minus`` verdict read ``1.0 < 1.0 == False`` → FAIL, when the
instrument in fact could not tell. v2 must return INDETERMINATE + saturated
there, and a tie must never resolve to FAIL.
"""

from __future__ import annotations

from unittest import mock

import numpy as np

from scripts import issue2203_runtime as R
from scripts.issue2223_drift import _graded_band_verdict, _judge_band_arms


def _verdict(plus, minus, plus_rate=0.5, minus_rate=0.5, **kw):
    return _graded_band_verdict(plus, minus, plus_rate, minus_rate, **kw)


def test_both_arms_ceiling_is_indeterminate_saturated():
    """The exact live failure: graded means at the ceiling + rates 1.0/1.0."""
    n = 24
    out = _verdict([99.0] * n, [99.5] * n, plus_rate=1.0, minus_rate=1.0)
    assert out["verdict"] == "INDETERMINATE"
    assert out["saturated"] is True
    assert "saturated" in out["reason"]
    # the v1 bug resolved this exact shape to FAIL — pin that it never does.
    assert out["verdict"] != "FAIL"


def test_rate_pinned_both_arms_is_saturated_even_with_graded_range():
    """Binary rate exactly 0.0/1.0 on BOTH arms => INDETERMINATE + saturated,
    per the re-instrumentation spec, even when the graded means keep range."""
    n = 24
    out = _verdict([60.0 + i * 0.1 for i in range(n)], [75.0] * n, plus_rate=1.0, minus_rate=1.0)
    assert out["verdict"] == "INDETERMINATE"
    assert out["saturated"] is True
    assert "rate pinned" in out["reason"]


def test_clean_directional_margin_passes():
    rng = np.random.default_rng(3)
    plus = list(30.0 + rng.normal(0, 4.0, size=24))
    minus = list(75.0 + rng.normal(0, 4.0, size=24))
    out = _verdict(plus, minus, plus_rate=0.2, minus_rate=0.9)
    assert out["verdict"] == "PASS"
    assert out["saturated"] is False
    assert out["margin"] > 0
    lo, hi = out["margin_ci95"]
    assert lo > 0 and hi > lo


def test_reversed_margin_fails():
    rng = np.random.default_rng(3)
    minus = list(30.0 + rng.normal(0, 4.0, size=24))
    plus = list(75.0 + rng.normal(0, 4.0, size=24))
    out = _verdict(plus, minus, plus_rate=0.9, minus_rate=0.2)
    assert out["verdict"] == "FAIL"
    assert out["saturated"] is False
    assert out["margin"] < 0
    assert out["margin_ci95"][1] < 0


def test_noise_dominated_margin_is_indeterminate():
    rng = np.random.default_rng(7)
    plus = list(50.0 + rng.normal(0, 12.0, size=24))
    minus = list(50.5 + rng.normal(0, 12.0, size=24))
    out = _verdict(plus, minus, plus_rate=0.5, minus_rate=0.55)
    assert out["verdict"] == "INDETERMINATE"
    assert out["saturated"] is False
    lo, hi = out["margin_ci95"]
    assert lo <= 0.0 <= hi


def test_exact_tie_never_fails():
    """margin == 0 must be INDETERMINATE, never FAIL (the v1 strict-< tie bug)."""
    out = _verdict([50.0] * 24, [50.0] * 24, plus_rate=0.5, minus_rate=0.5)
    assert out["verdict"] == "INDETERMINATE"
    assert out["margin"] == 0.0


def test_none_scores_dropped_pairwise_never_coerced():
    """Fully-dropped items (None graded mean) are excluded PAIRWISE and counted;
    a coercion to 0 would move the margin off the kept-pair arithmetic."""
    plus = [None, None, 30.0, 32.0, 28.0, 31.0]
    minus = [90.0, 91.0, 70.0, 72.0, 69.0, 71.0]
    out = _verdict(plus, minus, plus_rate=0.25, minus_rate=0.75)
    assert out["n_pairs_dropped"] == 2
    assert out["n_pairs_scored"] == 4
    # margin over the 4 kept pairs only: mean([40, 40, 41, 40]) = 40.25.
    # None->0 coercion would instead give mean(minus)-mean(plus)=77.17-20.17=57.
    assert abs(out["margin"] - 40.25) < 1e-9
    assert abs(out["plus_graded_mean"] - 30.25) < 1e-9
    assert out["verdict"] == "PASS"


def test_too_few_scored_pairs_is_indeterminate():
    out = _verdict([50.0, None], [80.0, 70.0], plus_rate=None, minus_rate=1.0)
    assert out["verdict"] == "INDETERMINATE"
    assert out["n_pairs_scored"] == 1
    assert out["margin"] is None


def _judge_result(prefix: str, means: list[float | None], rate, n_draws=5, n_dropped=0):
    """A dict shaped exactly like issue2203_runtime.judge_rate's return."""
    scores = {f"{prefix}_{i}": v for i, v in enumerate(means)}
    return {
        "mean_scores": scores,
        "n_items": len(means),
        "n_scored_items": sum(1 for v in means if v is not None),
        "rate": rate,
        "n_total_draws": n_draws * len(means),
        "n_dropped_draws": n_dropped,
        "n_transport_lost_draws": 0,
        "n_api_refusal_draws": 0,
        "n_truncation_dropped_draws": 0,
        "per_item_api_refusals": {},
    }


def test_judge_band_arms_real_body_saturated_live_shape(tmp_path):
    """Production-body test: the real _judge_band_arms executes end-to-end with
    the judge faked (autospec — signature-conformant) at the API boundary. The
    live failure shape (both arms rate 1.0, graded means ~ceiling) must come
    back INDETERMINATE + saturated, with drop telemetry propagated per arm and
    FRESH band_v2_* cache dirs (never the v1 band_plus/band_minus dirs)."""
    contexts = [{"system": f"You are role {i}.", "user": f"question {i}"} for i in range(4)]
    plus_res = _judge_result("plus", [99.0, 100.0, 98.5, 99.5], 1.0, n_dropped=1)
    minus_res = _judge_result("minus", [99.2, 99.8, 99.0, 100.0], 1.0, n_dropped=2)
    fake = mock.create_autospec(R.judge_rate, side_effect=[plus_res, minus_res])
    out = _judge_band_arms(contexts, ["p"] * 4, ["m"] * 4, tmp_path, n_draws=5, judge_fn=fake)
    assert out["verdict"] == "INDETERMINATE"
    assert out["saturated"] is True
    assert out["plus"]["rate"] == 1.0 and out["minus"]["rate"] == 1.0
    assert out["plus"]["n_dropped_draws"] == 1
    assert out["minus"]["n_dropped_draws"] == 2
    assert out["n_draws"] == 5
    assert fake.call_count == 2
    for call, key in zip(fake.call_args_list, ("plus", "minus"), strict=True):
        assert call.kwargs["cache_dir"].name == f"band_v2_{key}"
        assert call.kwargs["save_raw"].name == f"band_v2_{key}"
        assert call.kwargs["n_draws"] == 5
        items = call.args[0]
        assert [i[0] for i in items] == [f"{key}_{j}" for j in range(4)]
        # the judged question is the USER turn of each probe context
        assert [i[1] for i in items] == [c["user"] for c in contexts]


def test_judge_band_arms_directional_pass(tmp_path):
    contexts = [{"system": "s", "user": f"q{i}"} for i in range(6)]
    plus_res = _judge_result("plus", [20.0, 25.0, 22.0, 28.0, 24.0, 21.0], 0.0)
    minus_res = _judge_result("minus", [80.0, 85.0, 82.0, 88.0, 84.0, 81.0], 0.83)
    fake = mock.create_autospec(R.judge_rate, side_effect=[plus_res, minus_res])
    out = _judge_band_arms(contexts, ["p"] * 6, ["m"] * 6, tmp_path, judge_fn=fake)
    assert out["verdict"] == "PASS"
    assert out["margin"] == 60.0
    assert out["saturated"] is False
