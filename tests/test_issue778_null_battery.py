"""Unit smoke for the issue #778 null battery (CPU-only, no model calls).

Verifies the core statistical logic on synthetic data where the matched-trait
direction carries real signal and the nulls do not — the property the whole
experiment gates on. Also pins the selection-symmetric null policy (every null
draw takes its OWN max-over-layers |r|) and the drop-never-coerce shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from explore_persona_space.analysis import null_battery as nb


def _synthetic_cell(seed: int = 0, n: int = 40, n_layers: int = 6, dim: int = 32):
    """Build a synthetic (trait x setting) cell where matched r_B predicts well.

    A planted direction ``u`` at a specific layer drives the target; other layers
    and random/permuted directions do not. Returns the kwargs compute_setting needs.
    """
    rng = np.random.default_rng(seed)
    signal_layer = 3
    u = rng.standard_normal(dim)
    u /= np.linalg.norm(u)

    # Predictor activations: (n, L, D). At the signal layer, the component along u
    # correlates with the target; elsewhere noise.
    predictor = rng.standard_normal((n, n_layers, dim)) * 0.5
    coeff = rng.standard_normal(n)  # latent that drives target
    predictor[:, signal_layer, :] += np.outer(coeff, u)
    target = 3.0 * coeff + rng.standard_normal(n) * 0.3

    # Extraction pools: pos mean shifted along u, neg not -> diff-of-means ~ u.
    n_pool = 30
    pos = rng.standard_normal((n_pool, n_layers, dim)) * 0.5
    neg = rng.standard_normal((n_pool, n_layers, dim)) * 0.5
    pos[:, signal_layer, :] += u * 2.0  # planted contrast at the signal layer
    rb = pos.mean(axis=0) - neg.mean(axis=0)  # (L, D) ~ u at signal_layer

    # Two cross-trait directions: random (should NOT predict).
    other = {
        "traitX": rng.standard_normal((n_layers, dim)),
        "traitY": rng.standard_normal((n_layers, dim)),
    }
    return dict(
        predictor_acts=predictor,
        rb_per_layer=rb,
        target=target,
        pos_acts=pos,
        neg_acts=neg,
        other_rbs=other,
    ), signal_layer


def test_matched_beats_nulls_finetune():
    kw, signal_layer = _synthetic_cell()
    res, draws = nb.compute_setting(
        "evil", "finetune", n_draws=100, pca_k=3, n_boot=500, seed=1, **kw
    )
    # matched max|r| should be strong and select the planted layer.
    assert res.matched_max_abs > 0.7, res.matched_max_abs
    assert res.matched_selected_layer == signal_layer, res.matched_selected_layer
    # perm + randnorm nulls should be well below the matched r (empirical p small).
    assert res.nulls["perm"].empirical_p_one_sided < 0.05, res.nulls["perm"]
    assert res.nulls["randnorm"].empirical_p_one_sided < 0.05, res.nulls["randnorm"]
    # persisted draw matrices carry per-draw x per-layer |r| with the right shape.
    assert draws["perm"].shape == (100, kw["predictor_acts"].shape[1])
    assert draws["randnorm"].shape == (100, kw["predictor_acts"].shape[1])
    # fixed nulls: (n_dirs, L).
    assert draws["crosstrait"].shape == (2, kw["predictor_acts"].shape[1])
    assert draws["pca_topk"].shape == (3, kw["predictor_acts"].shape[1])


def test_selection_symmetric_null_takes_max_over_layers():
    # Each null draw's contribution is its OWN max-over-layers |r|, so the null
    # band's max entry equals a per-draw max, never a single-layer r.
    kw, _ = _synthetic_cell(seed=7)
    _, draws = nb.compute_setting("evil", "finetune", n_draws=50, seed=2, **kw)
    perm = draws["perm"]  # (50, L)
    per_draw_max = np.array([nb.max_abs_over_layers(perm[i]) for i in range(perm.shape[0])])
    lo, hi = nb.null_band(per_draw_max)
    # The band is over per-draw maxima, so hi >= the mean single-layer |r|.
    assert hi >= np.nanmean(np.abs(perm)), (hi, np.nanmean(np.abs(perm)))
    assert lo <= hi


def test_within_condition_r_controls_for_group_mean_separation():
    # Construct data where the OVERALL r is high purely from between-group mean
    # separation, but within each group there is NO relationship. The
    # within-condition r must be ~0 while the overall r is high.
    rng = np.random.default_rng(3)
    n_layers, dim = 4, 16
    u = rng.standard_normal(dim)
    u /= np.linalg.norm(u)
    groups = np.repeat(np.arange(8), 20)  # 8 groups x 20
    n = groups.size
    # target = group-level offset only (no within-group signal).
    group_offset = rng.standard_normal(8) * 5.0
    target = group_offset[groups] + rng.standard_normal(n) * 0.1
    predictor = rng.standard_normal((n, n_layers, dim)) * 0.3
    # Put the group offset along u at layer 1 -> overall proj tracks target
    # ACROSS groups, but within a group the proj is flat.
    predictor[:, 1, :] += np.outer(group_offset[groups], u)
    rb = np.zeros((n_layers, dim))
    rb[1] = u

    overall = nb.r_per_layer(predictor, rb, target)
    within = nb.within_condition_r_per_layer(predictor, rb, target, groups)
    assert abs(overall[1]) > 0.8, overall[1]  # overall r high (between-group)
    assert abs(within[1]) < 0.4, within[1]  # within-condition r collapses


def test_drop_never_coerce_score_parsing():
    # REFUSAL / non-numeric / out-of-range -> None (dropped), never a number.
    from scripts import issue778_lib as lib

    assert lib._score_from_parsed({"score": 73}) == 73.0
    assert lib._score_from_parsed({"score": "42"}) == 42.0
    assert lib._score_from_parsed({"score": "REFUSAL"}) is None
    assert lib._score_from_parsed({"score": "refusal"}) is None
    assert lib._score_from_parsed({"score": 150}) is None  # out of range
    assert lib._score_from_parsed({"score": -5}) is None
    assert lib._score_from_parsed({"score": "banana"}) is None
    assert lib._score_from_parsed({"score": True}) is None  # bool is malformed
    assert lib._score_from_parsed({"error": True, "score": 50}) is None
    assert lib._score_from_parsed(None) is None
    assert lib._score_from_parsed({}) is None


def test_score_from_parsed_accepts_bare_int_in_range():
    # parse_judge_json returns json.loads("85") == 85 verbatim (a valid off-spec
    # judge response). #778 r3: this MUST be carried as the score, not dropped.
    from scripts import issue778_lib as lib

    assert lib._score_from_parsed(85) == 85.0
    assert lib._score_from_parsed(0) == 0.0  # boundary
    assert lib._score_from_parsed(100) == 100.0  # boundary


def test_score_from_parsed_accepts_bare_float_in_range():
    from scripts import issue778_lib as lib

    assert lib._score_from_parsed(85.5) == 85.5
    assert lib._score_from_parsed(0.0) == 0.0
    assert lib._score_from_parsed(100.0) == 100.0


def test_score_from_parsed_rejects_out_of_range_bare_numeric():
    # Out-of-[0,100] bare numeric drops (drop-never-coerce), same as the dict path.
    from scripts import issue778_lib as lib

    assert lib._score_from_parsed(150) is None
    assert lib._score_from_parsed(-1) is None
    assert lib._score_from_parsed(100.5) is None
    assert lib._score_from_parsed(-0.5) is None


def test_score_from_parsed_rejects_bool_disguised_as_int():
    # isinstance(True, int) is True in Python: a judge that emitted `true` would
    # parse to 1.0 and be mis-counted as a score. Reject bools explicitly.
    from scripts import issue778_lib as lib

    assert lib._score_from_parsed(True) is None
    assert lib._score_from_parsed(False) is None


def test_benjamini_hochberg_monotone_and_bounded():
    pvals = [0.001, 0.02, 0.5, float("nan"), 0.04]
    adj = nb.benjamini_hochberg(pvals)
    finite = [a for a in adj if not np.isnan(a)]
    assert all(0.0 <= a <= 1.0 for a in finite)
    assert np.isnan(adj[3])  # NaN passes through
    # smallest raw p keeps the smallest adjusted p
    assert adj[0] == min(finite)


def test_empirical_p_has_plus_one_correction():
    null = np.array([0.1, 0.2, 0.3])
    # observed above all nulls -> p = (0+1)/(3+1) = 0.25 (never 0).
    assert nb.empirical_p_one_sided(0.9, null) == pytest.approx(0.25)
    # observed below all -> p = (3+1)/(3+1) = 1.0.
    assert nb.empirical_p_one_sided(0.0, null) == pytest.approx(1.0)


def test_split_cell_tag_handles_underscore_versions():
    # rsplit("_", 1) is WRONG: versions contain underscores (misaligned_1/_2).
    # split_cell_tag must recover the true (family, version).
    from scripts import issue778_lib as lib

    assert lib.split_cell_tag("evil_normal") == ("evil", "normal")
    assert lib.split_cell_tag("evil_misaligned_1") == ("evil", "misaligned_1")
    assert lib.split_cell_tag("mistake_gsm8k_misaligned_2") == ("mistake_gsm8k", "misaligned_2")
    assert lib.split_cell_tag("insecure_code_normal") == ("insecure_code", "normal")
    # the buggy rsplit would give ("evil_misaligned", "1") — assert we do NOT.
    assert lib.split_cell_tag("evil_misaligned_1")[0] != "evil_misaligned"
    with pytest.raises(ValueError):
        lib.split_cell_tag("not_a_real_cell")


def test_project_matches_paper_a_proj_b():
    # a_proj_b(a, b) = (a·b)/‖b‖.
    a = np.array([[1.0, 2.0, 2.0]])
    b = np.array([3.0, 0.0, 4.0])  # ‖b‖ = 5
    got = nb.project(a, b)
    assert got[0] == pytest.approx((1 * 3 + 2 * 0 + 2 * 4) / 5.0)
