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


def _m3(gaps, noise_floor=None):
    # gaps[b] = rho_A - rho_C; encode as A rho = 0.5+gap, C rho = 0.5.
    # noise_floor (dict or scalar or None) sets each behavior's noise_floor_p95.
    out = {}
    for b, g in gaps.items():
        rec = {"methods": {"A": {"rho_spearman": 0.5 + g}, "C": {"rho_spearman": 0.5}}}
        if isinstance(noise_floor, dict):
            rec["noise_floor_p95"] = noise_floor.get(b)
        else:
            rec["noise_floor_p95"] = noise_floor
        out[b] = rec
    return out


def _directions(behaviors, n_pos=50, n_neg=50):
    """Survivor-count-only direction dicts for decide() (the kill-criterion read).

    n_pos / n_neg may be a scalar (applied to all) or a per-behavior dict.
    """
    out = {}
    for b in behaviors:
        npos = n_pos[b] if isinstance(n_pos, dict) else n_pos
        nneg = n_neg[b] if isinstance(n_neg, dict) else n_neg
        out[b] = {"n_pos_survivors": npos, "n_neg_survivors": nneg}
    return out


def test_decision_adopt_a_all_pass():
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.98)
    conf = dict.fromkeys(headline, 0.05)
    gaps = dict.fromkeys(headline, 0.0)
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline, _directions(headline))
    assert d["verdict"] == "adopt_A"
    assert d["low_survivors"] == []


def test_decision_adopt_c_low_cosine_on_two():
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = {"sycophancy": 0.80, "refusal": 0.82, "broad_em": 0.99}  # 2 below 0.85
    conf = dict.fromkeys(headline, 0.05)
    gaps = dict.fromkeys(headline, 0.0)
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline, _directions(headline))
    assert d["verdict"] == "adopt_C_or_recipe_by_behavior"
    assert d["margins"]["n_cos_AC_below_0.85"] == 2


def test_decision_adopt_c_rho_worse_on_two():
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.98)
    conf = dict.fromkeys(headline, 0.05)
    gaps = {"sycophancy": -0.06, "refusal": -0.07, "broad_em": 0.0}  # A worse by >=0.05 on 2
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline, _directions(headline))
    assert d["verdict"] == "adopt_C_or_recipe_by_behavior"
    assert d["margins"]["n_rho_A_worse_by_0.05"] == 2


def test_decision_inconclusive_between_thresholds():
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.90)  # between 0.85 and 0.95 → not adopt-A, not strong-fail
    conf = dict.fromkeys(headline, 0.15)  # between 0.10 and 0.25
    gaps = dict.fromkeys(headline, 0.0)
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline, _directions(headline))
    assert d["verdict"] == "inconclusive_recipe_by_behavior"


# ── §7 <MIN_SURVIVORS kill criterion (the round-1 binding-concern fix) ─────────


def test_kill_criterion_drops_low_survivor_behavior():
    """A behavior with n_pos_survivors=2 (< MIN_SURVIVORS=5) is EXCLUDED from
    present_behaviors and flagged in low_survivors / dropped_low_survivors. The
    remaining behaviors still drive the verdict (here adopt-C via 2 low cosines on
    the survivors). This is the precise round-1 blocker: a <5-survivor behavior
    must never count toward the headline recipe verdict (a noise-dominated read)."""
    from issue661_analysis import MIN_SURVIVORS, decide

    assert MIN_SURVIVORS == 5
    headline = ["sycophancy", "refusal", "broad_em"]
    cos = {"sycophancy": 0.80, "refusal": 0.82, "broad_em": 0.99}
    conf = dict.fromkeys(headline, 0.05)
    gaps = dict.fromkeys(headline, 0.0)
    # broad_em has only 2 pos survivors → dropped; sycophancy/refusal kept.
    dirs = _directions(headline, n_pos={"sycophancy": 50, "refusal": 50, "broad_em": 2})
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline, dirs)
    assert "broad_em" in d["low_survivors"]
    assert "broad_em" not in d["present_behaviors"]
    assert d["dropped_low_survivors"]["broad_em"]["n_pos"] == 2
    assert d["n_survivors"]["broad_em"] == {"n_pos": 2, "n_neg": 50}
    # The two kept behaviors still produce a verdict (both cos < 0.85 → adopt-C).
    assert d["verdict"] == "adopt_C_or_recipe_by_behavior"


def test_kill_criterion_neg_pool_below_floor():
    """min(n_pos, n_neg) gates: a behavior with plenty of POS survivors but <5 NEG
    survivors is still dropped (either pool below the floor kills it, plan §7)."""
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.98)
    conf = dict.fromkeys(headline, 0.05)
    gaps = dict.fromkeys(headline, 0.0)
    dirs = _directions(headline, n_neg={"sycophancy": 50, "refusal": 50, "broad_em": 3})
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline, dirs)
    assert "broad_em" in d["low_survivors"]
    # Not all 3 present → adopt-A cannot fire even though survivors pass on 2.
    assert d["verdict"] != "adopt_A"


def test_kill_criterion_all_three_drop_halts():
    """If ALL 3 headline behaviors have <5 survivors, the verdict is
    halt_low_survivors (plan §7: instruction elicitation failed on this model)."""
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.98)
    conf = dict.fromkeys(headline, 0.05)
    gaps = dict.fromkeys(headline, 0.0)
    dirs = _directions(headline, n_pos=4, n_neg=4)  # all below MIN_SURVIVORS
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline, dirs)
    assert d["verdict"] == "halt_low_survivors"
    assert sorted(d["low_survivors"]) == sorted(headline)
    assert d["present_behaviors"] == []


def test_kill_criterion_at_exactly_min_survivors_keeps():
    """MIN_SURVIVORS=5 is the floor: exactly 5 in BOTH pools is KEPT (not dropped).
    The criterion is min(n_pos, n_neg) < 5 → drop, so 5 passes."""
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.98)
    conf = dict.fromkeys(headline, 0.05)
    gaps = dict.fromkeys(headline, 0.0)
    dirs = _directions(headline, n_pos=5, n_neg=5)
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline, dirs)
    assert d["low_survivors"] == []
    assert d["verdict"] == "adopt_A"


# ── adopt-A ρ conjunct threads the per-behavior noise floor (round-1 blocker) ──


def test_adopt_a_rho_leg_satisfied_within_noise_floor():
    """gap=-0.02 with noise_floor=0.05 SATISFIES adopt-A's ρ leg (A is worse than C
    by less than the registered #658 noise floor → 'within noise', plan §7). The
    round-1 code gated gaps>=-1e-9 which would have FAILED this — strictly stricter
    than the registered band."""
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.98)
    conf = dict.fromkeys(headline, 0.05)
    gaps = dict.fromkeys(headline, -0.02)  # A worse by 0.02 on all 3
    d = decide(_m1(cos), _m2(conf), _m3(gaps, noise_floor=0.05), headline, _directions(headline))
    assert d["verdict"] == "adopt_A"
    assert d["adopt_A_noise_floor"]["sycophancy"] == 0.05


def test_adopt_a_rho_leg_fails_with_zero_noise_floor():
    """gap=-0.02 with noise_floor=0.0 does NOT satisfy adopt-A's ρ leg (A is worse
    than C beyond a zero tolerance). Verdict is not adopt_A."""
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.98)
    conf = dict.fromkeys(headline, 0.05)
    gaps = dict.fromkeys(headline, -0.02)
    d = decide(_m1(cos), _m2(conf), _m3(gaps, noise_floor=0.0), headline, _directions(headline))
    assert d["verdict"] != "adopt_A"


def test_adopt_a_rho_leg_none_noise_floor_uses_registered_default():
    """noise_floor=None falls back to the registered DEFAULT_NOISE_FLOOR (0.05) and
    flags the fallback per behavior. gap=-0.02 < 0.05 → adopt-A's ρ leg satisfied
    via the fallback."""
    from issue661_analysis import DEFAULT_NOISE_FLOOR, decide

    assert DEFAULT_NOISE_FLOOR == 0.05
    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.98)
    conf = dict.fromkeys(headline, 0.05)
    gaps = dict.fromkeys(headline, -0.02)
    d = decide(_m1(cos), _m2(conf), _m3(gaps, noise_floor=None), headline, _directions(headline))
    assert d["verdict"] == "adopt_A"
    assert all(d["adopt_A_noise_floor_fallback_used"][b] for b in headline)
    assert d["adopt_A_noise_floor"]["refusal"] == DEFAULT_NOISE_FLOOR


def test_adopt_a_rho_leg_fails_beyond_floor():
    """A worse than C by MORE than the noise floor (gap=-0.08, noise_floor=0.05)
    fails adopt-A's ρ leg AND (on >=2 behaviors at <=-0.05) triggers adopt-C."""
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    cos = dict.fromkeys(headline, 0.98)
    conf = dict.fromkeys(headline, 0.05)
    gaps = dict.fromkeys(headline, -0.08)  # beyond floor + beyond the -0.05 strong-fail
    d = decide(_m1(cos), _m2(conf), _m3(gaps, noise_floor=0.05), headline, _directions(headline))
    assert d["verdict"] == "adopt_C_or_recipe_by_behavior"
    assert d["margins"]["n_rho_A_worse_by_0.05"] == 3


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


# ── §6.5 M1/M2 survivor-set bootstrap CIs ────────────────────────────────────


def _survivor_dir(n_pos=40, n_neg=40, n_layers=2, hidden=4, seed=0):
    """Hand-built per-survivor stacks: arm A present along +e0, absent along -e0;
    arm C the SAME texts under a slightly rotated read (so cos(A,C) is high but
    not exactly 1). Returns a directions[behavior] dict with pooled means + stacks.
    """
    rng = torch.Generator().manual_seed(seed)
    base_present = torch.zeros(hidden)
    base_present[0] = 1.0
    base_absent = torch.zeros(hidden)
    base_absent[0] = -1.0
    noise = lambda n: 0.05 * torch.randn(n, n_layers, hidden, generator=rng)  # noqa: E731
    pA = base_present.expand(n_pos, n_layers, hidden) + noise(n_pos)
    aA = base_absent.expand(n_neg, n_layers, hidden) + noise(n_neg)
    pC = base_present.expand(n_pos, n_layers, hidden) + noise(n_pos)
    aC = base_absent.expand(n_neg, n_layers, hidden) + noise(n_neg)
    cp = (base_present + 0.1).expand(n_pos, n_layers, hidden) + noise(n_pos)
    cn = (base_absent - 0.1).expand(n_neg, n_layers, hidden) + noise(n_neg)
    return {
        "r_b_a": (pA.mean(0) - aA.mean(0)),
        "r_b_c": (pC.mean(0) - aC.mean(0)),
        "c_pos": cp.mean(0),
        "c_neg": cn.mean(0),
        "n_pos_survivors": n_pos,
        "n_neg_survivors": n_neg,
        "present_A_items": pA,
        "absent_A_items": aA,
        "present_C_items": pC,
        "absent_C_items": aC,
        "c_pos_items": cp,
        "c_neg_items": cn,
    }


def test_m1_cos_ac_ci_brackets_point_estimate():
    """The §6.5 survivor-set bootstrap CI for cos(A,C) is a [lo, hi] pair with
    lo <= hi and brackets the point estimate. Reproducible (fixed per-method seed
    offset)."""
    from issue661_analysis import cosine_per_layer, m1_cos_ac_ci

    d = _survivor_dir(seed=1)
    sl = 0
    ci = m1_cos_ac_ci(d, sl, bootstrap_n=200)
    assert ci is not None and len(ci) == 2 and ci[0] <= ci[1]
    point = float(cosine_per_layer(d["r_b_a"], d["r_b_c"])[sl])
    assert ci[0] - 0.05 <= point <= ci[1] + 0.05
    # Reproducible: same args → same CI.
    assert m1_cos_ac_ci(d, sl, bootstrap_n=200) == ci


def test_m1_cos_ac_ci_none_without_stacks():
    """A pre-round-2 .pt without per-survivor stacks → CI is None (no crash)."""
    from issue661_analysis import m1_cos_ac_ci

    d = {"present_A_items": None, "absent_A_items": None, "present_C_items": None,
         "absent_C_items": None}  # fmt: skip
    assert m1_cos_ac_ci(d, 0, bootstrap_n=200) is None


def test_m2_confound_ci_bounded_and_raw():
    """The §6.5 M2 CI returns (bounded_ci, raw_ci); the bounded CI stays within
    [0, 1] (the |cos| gate quantity), the raw CI is unbounded but >= 0."""
    from issue661_analysis import m2_confound_ci

    d = _survivor_dir(seed=2)
    bounded, raw = m2_confound_ci(d, 0, bootstrap_n=200)
    assert bounded is not None and 0.0 <= bounded[0] <= bounded[1] <= 1.0 + 1e-6
    assert raw is not None and raw[0] >= 0.0 and raw[0] <= raw[1]


def test_m1_m2_ci_emitted_in_records():
    """m1_cosine / m2_confound attach cos_AC_ci95 / confound_A_ci95 records."""
    from issue661_analysis import m1_cosine, m2_confound

    dirs = {"sycophancy": _survivor_dir(seed=3)}
    sel = {"sycophancy": 0}
    m1 = m1_cosine(dirs, {}, ["sycophancy"], sel, bootstrap_n=200)
    m2 = m2_confound(dirs, {}, ["sycophancy"], sel, bootstrap_n=200)
    assert "cos_AC_ci95" in m1["sycophancy"] and m1["sycophancy"]["cos_AC_ci95"] is not None
    assert "confound_A_ci95" in m2["sycophancy"] and m2["sycophancy"]["confound_A_ci95"] is not None
    assert "confound_A_raw_ci95" in m2["sycophancy"]


# ── M3 fail-loud on a genuinely-missing E0 column (fail-fast, plan §6.4) ───────


def test_m3_fails_loud_on_absent_e0_when_active():
    """When M3 is active (real v0 + non-empty E0 table) but a requested behavior's
    E0 column is ENTIRELY absent, m3_predictive raises (NOT a silent methods:{} →
    rho_gap None → vacuous adopt-A PASS). This is the reconciler Fix #3 clause."""
    from issue661_analysis import m3_predictive

    # v0 with 5 contexts at (1, 4); E0 table present for ctx but NOT for 'refusal'.
    v0 = {f"c{i}": torch.zeros(1, 4) for i in range(5)}
    e0 = {"e0": {f"c{i}": {"sycophancy": {"rate": 0.5, "per_probe": []}} for i in range(5)}}
    dirs = {
        "refusal": {"r_b_a": torch.zeros(1, 4), "r_b_c": torch.zeros(1, 4)},
    }
    with pytest.raises(AssertionError, match="E0 column entirely absent"):
        m3_predictive(dirs, {}, v0, e0, ["refusal"], {"refusal": 0}, bootstrap_n=10)


def test_m3_inactive_does_not_fail_loud():
    """M3 inactive (empty E0 table, the smoke / dim-mismatch branch) does NOT trip
    the absent-E0 assert — it is the registered 'skip M3' branch."""
    from issue661_analysis import m3_predictive

    v0 = {"c0": torch.zeros(1, 4)}
    e0 = {"e0": {}, "columns": []}
    dirs = {"refusal": {"r_b_a": torch.zeros(1, 4), "r_b_c": torch.zeros(1, 4)}}
    out = m3_predictive(dirs, {}, v0, e0, ["refusal"], {"refusal": 0}, bootstrap_n=10)
    assert out["refusal"]["methods"] == {}
    assert out["refusal"]["m3_undefined_reason"] == "m3_inactive"


# ── arm-B fail-loud on a missing column (fail-fast, reconciler Fix #3) ─────────


def test_load_arm_b_fails_loud_on_missing_column(monkeypatch):
    """load_arm_b asserts (no warn-and-continue) when a requested behavior's #658
    r_b.pt column is absent — naming the exact missing (column, behavior) pairs."""
    import issue661_analysis as ana

    fake_rb = {
        "r_b": {"sycophancy": {"diffmeans": torch.zeros(28, 8)}},  # refusal absent
        "capture_layers": list(range(28)),
        "columns": ["sycophancy"],
    }

    def fake_download(*a, **k):
        return "/tmp/fake_rb.pt"

    # load_arm_b does a function-local `from huggingface_hub import hf_hub_download`,
    # so patch the source module attr (resolved at call time).
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(ana.torch, "load", lambda *a, **k: fake_rb)
    with pytest.raises(AssertionError, match="requested columns absent"):
        ana.load_arm_b(["sycophancy", "refusal"])


def test_load_arm_b_succeeds_when_all_present(monkeypatch):
    """All requested columns present → load_arm_b returns each as (28, H) float."""
    import issue661_analysis as ana

    fake_rb = {
        "r_b": {
            "sycophancy": {"diffmeans": torch.zeros(28, 8)},
            "refusal": {"diffmeans": torch.ones(28, 8)},
        },
        "capture_layers": list(range(28)),
        "columns": ["sycophancy", "refusal"],
    }
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "/tmp/fake_rb.pt")
    monkeypatch.setattr(ana.torch, "load", lambda *a, **k: fake_rb)
    out = ana.load_arm_b(["sycophancy", "refusal"])
    assert set(out) == {"sycophancy", "refusal"}
    assert out["refusal"].shape == (28, 8)


# ── F1/F2 CI error bars (plan §8: every plot carries error bars) ───────────────


def test_make_figures_plots_m1_m2_cis(tmp_path):
    """make_figures writes F1/F2/F3 and the F1/F2 CI-error-bar branches execute
    when cos_AC_ci95 / confound_A_ci95 are present (plan §8 / reconciler Fix #4 —
    the CIs must reach the hero figures, not just decision.json). 6-layer records
    with the selected-layer CIs present + the F3 ρ CI."""
    from issue661_analysis import m1_cosine, m2_confound, make_figures

    behaviors = ["sycophancy"]
    sel = {"sycophancy": 0}
    dirs = {"sycophancy": _survivor_dir(seed=7, n_layers=6, hidden=4)}
    m1 = m1_cosine(dirs, {}, behaviors, sel, bootstrap_n=120)
    m2 = m2_confound(dirs, {}, behaviors, sel, bootstrap_n=120)
    # The F1/F2 CI branches require the per-survivor stacks → CIs are non-None here.
    assert m1["sycophancy"]["cos_AC_ci95"] is not None
    assert m2["sycophancy"]["confound_A_ci95"] is not None
    m3 = {
        "sycophancy": {
            "methods": {"A": {"rho_spearman": 0.4, "rho_ci95": [0.2, 0.6]}},
            "noise_floor_p95": 0.05,
        }
    }
    paths = make_figures(m1, m2, m3, behaviors, tmp_path / "figs")
    assert len(paths) == 3
    for p in paths:
        assert Path(p).exists() and Path(p).stat().st_size > 0


def test_make_figures_no_ci_branch(tmp_path):
    """make_figures still renders F1/F2 when the CIs are None (pre-round-2 .pt /
    point-estimate-only) — the error-bar branch is skipped, no crash."""
    from issue661_analysis import make_figures

    behaviors = ["sycophancy"]
    m1 = {
        "sycophancy": {
            "cos_AC": [0.9] * 6,
            "selected_layer": 0,
            "cos_AC_selected": 0.9,
            "cos_AC_ci95": None,
        }
    }
    m2 = {
        "sycophancy": {
            "confound_A": [0.1] * 6,
            "confound_C_control": [0.02] * 6,
            "confound_A_selected": 0.1,
            "confound_C_control_selected": 0.02,
            "confound_A_ci95": None,
            "selected_layer": 0,
        }
    }
    m3 = {"sycophancy": {"methods": {}, "noise_floor_p95": None}}
    paths = make_figures(m1, m2, m3, behaviors, tmp_path / "figs")
    assert len(paths) == 3 and all(Path(p).exists() for p in paths)


# ── P3 §7 drop → analysis load_directions handling (smoke-discovered crash fix) ─


def test_load_directions_extracted_record(tmp_path):
    """A normal P3 .pt (directions + stacks present) loads with
    record_is_extracted=True and the float tensors + per-survivor stacks."""
    from issue661_analysis import load_directions

    d = tmp_path / "directions"
    d.mkdir()
    torch.save(
        {
            "behavior": "sycophancy",
            "dropped_low_survivors": False,
            "r_b_a": torch.zeros(24, 8),
            "r_b_c": torch.zeros(24, 8),
            "c_pos": torch.zeros(24, 8),
            "c_neg": torch.zeros(24, 8),
            "n_pos_survivors": 6,
            "n_neg_survivors": 7,
            "present_A_items": torch.zeros(6, 24, 8, dtype=torch.float16),
            "absent_A_items": torch.zeros(7, 24, 8, dtype=torch.float16),
            "present_C_items": torch.zeros(6, 24, 8, dtype=torch.float16),
            "absent_C_items": torch.zeros(7, 24, 8, dtype=torch.float16),
            "c_pos_items": torch.zeros(6, 24, 8, dtype=torch.float16),
            "c_neg_items": torch.zeros(7, 24, 8, dtype=torch.float16),
        },
        d / "r_b_sycophancy.pt",
    )
    out = load_directions(d, ["sycophancy"])["sycophancy"]
    assert out["record_is_extracted"] is True
    assert out["r_b_a"].shape == (24, 8) and out["r_b_a"].dtype == torch.float32
    assert out["present_A_items"].shape == (6, 24, 8)
    assert out["n_pos_survivors"] == 6 and out["n_neg_survivors"] == 7


def test_load_directions_dropped_record(tmp_path):
    """A P3-dropped .pt (survivor-count-only, no direction tensors) loads with
    record_is_extracted=False, r_b_a=None, and the survivor counts preserved — so
    the analysis kill criterion drops it from the verdict without reading tensors.
    This is the smoke-discovered crash class: a 0/low-survivor pool must not crash
    P3 NOR the analysis (plan §7 'drop + report, proceed')."""
    from issue661_analysis import load_directions

    d = tmp_path / "directions"
    d.mkdir()
    torch.save(
        {
            "behavior": "refusal",
            "dropped_low_survivors": True,
            "n_pos_survivors": 3,
            "n_neg_survivors": 0,
        },
        d / "r_b_refusal.pt",
    )
    out = load_directions(d, ["refusal"])["refusal"]
    assert out["record_is_extracted"] is False
    assert out["r_b_a"] is None and out["present_A_items"] is None
    assert out["n_pos_survivors"] == 3 and out["n_neg_survivors"] == 0


def test_decide_with_p3_dropped_behavior_records_it():
    """decide() over a full headline where one behavior was P3-dropped (counts
    only, record_is_extracted False) records it in dropped_low_survivors and keeps
    it out of present_behaviors — even though its m1/m2 records are absent."""
    from issue661_analysis import decide

    headline = ["sycophancy", "refusal", "broad_em"]
    # M1/M2/M3 only carry the two EXTRACTED behaviors (refusal was P3-dropped).
    extracted = ["sycophancy", "broad_em"]
    cos = {b: 0.80 for b in extracted}  # both below 0.85 → adopt-C on the kept set
    conf = {b: 0.05 for b in extracted}
    gaps = {b: 0.0 for b in extracted}
    dirs = {
        "sycophancy": {"n_pos_survivors": 50, "n_neg_survivors": 50},
        "broad_em": {"n_pos_survivors": 50, "n_neg_survivors": 50},
        "refusal": {"n_pos_survivors": 3, "n_neg_survivors": 0, "record_is_extracted": False},
    }
    d = decide(_m1(cos), _m2(conf), _m3(gaps), headline, dirs)
    assert "refusal" in d["low_survivors"]
    assert "refusal" not in d["present_behaviors"]
    assert d["dropped_low_survivors"]["refusal"]["n_neg"] == 0
    assert d["verdict"] == "adopt_C_or_recipe_by_behavior"
