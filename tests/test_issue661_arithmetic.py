# ruff: noqa: RUF002, RUF003
"""Issue #661 arithmetic + shape tests (no GPU, hand-built tensors).

Covers the load-bearing analysis math (M1 cosine, M2 projection-fraction, M3
LOCO 1-D OLS, the §7 decision logic) and the vendored AnswerSpanCapture span
slicing on a tiny CPU model. These let a reader gain confidence in the
divergence arithmetic by reading the tests alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


# ── M1 cosine ──────────────────────────────────────────────────────────────


def test_cosine_per_layer_identical_and_orthogonal():
    from issue661_analysis import cosine_per_layer

    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])  # (2, 3)
    b_same = a.clone()
    b_orth = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    cos_same = cosine_per_layer(a, b_same)
    cos_orth = cosine_per_layer(a, b_orth)
    assert np.allclose(cos_same, [1.0, 1.0], atol=1e-6)
    assert np.allclose(cos_orth, [0.0, 0.0], atol=1e-6)


def test_cosine_per_layer_antiparallel():
    from issue661_analysis import cosine_per_layer

    a = torch.tensor([[1.0, 1.0]])
    b = torch.tensor([[-1.0, -1.0]])
    assert np.allclose(cosine_per_layer(a, b), [-1.0], atol=1e-6)


# ── M2 projection fraction ──────────────────────────────────────────────────


def test_projection_fraction_aligned_is_one():
    from issue661_analysis import projection_fraction

    # r fully along the axis → |<r,axis>|/||r|| == ||axis|| (axis not unit here),
    # but with a UNIT axis the fraction equals 1.0 when r ∥ axis.
    axis = torch.tensor([[1.0, 0.0]])  # unit
    r = torch.tensor([[5.0, 0.0]])
    assert np.allclose(projection_fraction(r, axis), [1.0], atol=1e-6)


def test_projection_fraction_orthogonal_is_zero():
    from issue661_analysis import projection_fraction

    axis = torch.tensor([[1.0, 0.0]])
    r = torch.tensor([[0.0, 7.0]])
    assert np.allclose(projection_fraction(r, axis), [0.0], atol=1e-6)


def test_projection_fraction_abs_value():
    from issue661_analysis import projection_fraction

    # Anti-aligned still gives a POSITIVE fraction (the |.| in M2).
    axis = torch.tensor([[1.0, 0.0]])
    r = torch.tensor([[-3.0, 4.0]])  # ||r|| = 5, <r,axis> = -3 ; |cos| = 3/5
    assert np.allclose(projection_fraction(r, axis), [3.0 / 5.0], atol=1e-6)


def test_projection_fraction_is_bounded_for_nonunit_axis():
    """The decision-gate confound is |cos| ∈ [0, 1] EVEN when ĉ_inst is not a
    unit vector (the M2 measurement-validity fix — the 0.10/0.25 gates require a
    bounded fraction). A non-unit axis must NOT inflate the fraction past 1."""
    from issue661_analysis import projection_fraction

    axis = torch.tensor([[100.0, 0.0]])  # large non-unit axis
    r = torch.tensor([[3.0, 4.0]])  # 45-ish deg; |cos| = 3/5 regardless of ‖axis‖
    vals = projection_fraction(r, axis)
    assert np.all(vals <= 1.0 + 1e-6) and np.all(vals >= 0.0)
    assert np.allclose(vals, [3.0 / 5.0], atol=1e-6)


def test_projection_fraction_raw_keeps_axis_norm():
    """The companion raw read |⟨r,axis⟩|/‖r‖ retains the axis norm (unbounded);
    it is reported alongside the gate quantity, never gated on."""
    from issue661_analysis import projection_fraction_raw

    axis = torch.tensor([[2.0, 0.0]])  # ‖axis‖ = 2
    r = torch.tensor([[3.0, 4.0]])  # ‖r‖ = 5, ⟨r,axis⟩ = 6 → 6/5 = 1.2 (> 1)
    assert np.allclose(projection_fraction_raw(r, axis), [6.0 / 5.0], atol=1e-6)


# ── M3 LOCO 1-D OLS (the plan's definition, NOT fit_a33's global ρ) ──────────


def test_loco_1d_ols_recovers_linear_relationship():
    from issue661_analysis import _spearman, loco_1d_ols_predictions

    rng = np.random.default_rng(0)
    proj = rng.normal(size=30)
    e0 = 2.0 * proj + 0.5 + rng.normal(scale=0.05, size=30)  # strong linear, low noise
    preds = loco_1d_ols_predictions(proj, e0)
    # Held-out preds should rank-correlate near-perfectly with measured E0.
    rho = _spearman(preds, e0)
    assert rho is not None and rho > 0.95


def test_loco_1d_ols_is_held_out_not_global():
    """The LOCO predictor refits (a,b) per held-out context — distinct from a
    single global slope. With a pure noise predictor, the held-out ρ should be
    near 0 (no spurious in-sample fit leaking)."""
    from issue661_analysis import _spearman, loco_1d_ols_predictions

    rng = np.random.default_rng(1)
    proj = rng.normal(size=40)
    e0 = rng.normal(size=40)  # independent of proj
    preds = loco_1d_ols_predictions(proj, e0)
    rho = _spearman(preds, e0)
    # No real signal: |ρ| should be small (held-out, no overfit leak).
    assert rho is None or abs(rho) < 0.5


def test_loco_1d_ols_degenerate_predictor_returns_mean():
    from issue661_analysis import loco_1d_ols_predictions

    proj = np.zeros(10)  # constant predictor
    e0 = np.arange(10, dtype=float)
    preds = loco_1d_ols_predictions(proj, e0)
    # Each held-out prediction is the mean of the OTHER 9 values.
    for i in range(10):
        expected = float(np.mean([e0[j] for j in range(10) if j != i]))
        assert preds[i] == pytest.approx(expected)


# ── §7 decision logic ───────────────────────────────────────────────────────


def _m1(cos_ac):
    return {b: {"cos_AC_selected": v, "selected_layer": 14} for b, v in cos_ac.items()}


def _m2(conf):
    return {b: {"confound_A_selected": v, "selected_layer": 14} for b, v in conf.items()}


def _m3(gaps):
    # gaps[b] = rho_A - rho_C; encode as A rho = 0.5+gap, C rho = 0.5
    out = {}
    for b, g in gaps.items():
        out[b] = {"methods": {"A": {"rho_spearman": 0.5 + g}, "C": {"rho_spearman": 0.5}}}
    return out


def test_decision_adopt_a_all_pass():
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.98)
    conf = dict.fromkeys(headline, 0.05)
    gaps = dict.fromkeys(headline, 0.0)
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline)
    assert d["verdict"] == "adopt_A"


def test_decision_adopt_c_low_cosine_on_two():
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = {"sycophancy": 0.80, "refusal": 0.82, "broad_em": 0.99}  # 2 below 0.85
    conf = dict.fromkeys(headline, 0.05)
    gaps = dict.fromkeys(headline, 0.0)
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline)
    assert d["verdict"] == "adopt_C_or_recipe_by_behavior"
    assert d["margins"]["n_cos_AC_below_0.85"] == 2


def test_decision_adopt_c_rho_worse_on_two():
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.98)
    conf = dict.fromkeys(headline, 0.05)
    gaps = {"sycophancy": -0.06, "refusal": -0.07, "broad_em": 0.0}  # A worse by >=0.05 on 2
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline)
    assert d["verdict"] == "adopt_C_or_recipe_by_behavior"
    assert d["margins"]["n_rho_A_worse_by_0.05"] == 2


def test_decision_inconclusive_between_thresholds():
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.90)  # between 0.85 and 0.95 → not adopt-A, not strong-fail
    conf = dict.fromkeys(headline, 0.15)  # between 0.10 and 0.25
    gaps = dict.fromkeys(headline, 0.0)
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline)
    assert d["verdict"] == "inconclusive_recipe_by_behavior"


# ── Vendored AnswerSpanCapture span slicing (tiny CPU model) ─────────────────


@pytest.mark.slow
def test_answer_span_capture_slices_answer_tokens():
    """AnswerSpanCapture.answer_span_stack returns (L, S, H) over exactly the
    answer positions; mean_prompt_stack returns (L, H). Tiny CPU model so no GPU."""
    from issue661_extract_directions import AnswerSpanCapture
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size

    prompt_ids = tok("Hello there", return_tensors="pt", add_special_tokens=True)["input_ids"]
    ans_ids = tok("general kenobi", return_tensors="pt", add_special_tokens=False)["input_ids"]
    full = torch.cat([prompt_ids, ans_ids], dim=1)
    plen, alen = int(prompt_ids.shape[1]), int(ans_ids.shape[1])

    cap = AnswerSpanCapture(model, n_layers)
    try:
        with torch.no_grad():
            _ = model(input_ids=full)
        span = cap.answer_span_stack(n_layers, plen, plen + alen)
        assert span.shape == (n_layers, alen, hidden), tuple(span.shape)
        # capture cleared after the read → a prompt-mean read needs a fresh fwd
        with torch.no_grad():
            _ = model(input_ids=prompt_ids)
        cmean = cap.mean_prompt_stack(n_layers, plen)
        assert cmean.shape == (n_layers, hidden), tuple(cmean.shape)
    finally:
        cap.remove()


# ── judge score parsing ──────────────────────────────────────────────────────


def test_parse_score_bare_number():
    from issue661_judge_filter import parse_score

    assert parse_score("85") == 85
    assert parse_score("  0\n") == 0
    assert parse_score("100") == 100


def test_parse_score_refusal_and_unparseable():
    from issue661_judge_filter import parse_score

    assert parse_score("REFUSAL") is None
    assert parse_score("Refusal: the model declined") is None
    assert parse_score("no number here") is None
    assert parse_score(None) is None


def test_parse_score_out_of_range_skipped():
    from issue661_judge_filter import parse_score

    # 250 is out of [0,100]; should be skipped (None, not clamped).
    assert parse_score("250") is None
