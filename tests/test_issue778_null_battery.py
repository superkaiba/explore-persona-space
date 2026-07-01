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


# ── Driver reconciled-statistics BH-split (plan v4 §5/§6/§11) ────────────────────


def test_driver_bh_family_is_stochastic_only():
    """The BH family pools ONLY the stochastic nulls (perm/randnorm), NOT the fixed.

    Reconciled statistics fix (plan v4 §5/§6/§11): the fixed-direction nulls
    (crosstrait 2 dirs, pca_topk 5 dirs) have a +1 empirical-p floor of 1/3 and
    1/6, so a p<0.025 BH gate is unsatisfiable by construction — they must be
    EXCLUDED from the BH family (the parent's 48-test pooling was the bug this
    round fixes). Pins the driver's STOCHASTIC/FIXED partition so a refactor that
    re-adds the fixed nulls to the BH pool fails here.
    """
    from scripts import issue778_null_battery as nbd

    assert nbd.STOCHASTIC_NULL_KINDS == ("perm", "randnorm")
    assert nbd.FIXED_NULL_KINDS == ("crosstrait", "pca_topk")
    assert set(nbd.STOCHASTIC_NULL_KINDS).isdisjoint(nbd.FIXED_NULL_KINDS)

    # Simulate main()'s pooling filter over a full 4-null set: only perm/randnorm
    # p-values may enter the BH list.
    nulls = {
        "perm": {"empirical_p_one_sided": 0.10},
        "randnorm": {"empirical_p_one_sided": 0.20},
        "crosstrait": {"empirical_p_one_sided": 0.33},  # 1/3 floor — never < 0.025
        "pca_topk": {"empirical_p_one_sided": 0.17},  # 1/6 floor — never < 0.025
    }
    pooled = [
        nr["empirical_p_one_sided"] for k, nr in nulls.items() if k in nbd.STOCHASTIC_NULL_KINDS
    ]
    assert pooled == [0.10, 0.20], pooled  # fixed nulls excluded from the BH family


def test_driver_annotate_exceedance_fixed_only():
    """_annotate_exceedance tags ONLY the fixed nulls with an exceedance bool.

    exceedance = observed matched max|r| STRICTLY exceeds the max over the fixed
    null's per-direction max|r|. Stochastic nulls get NO exceedance key (they use
    the BH-adjusted empirical p). An all-NaN fixed null -> None (undecidable).
    """
    from scripts import issue778_null_battery as nbd

    payload = {
        "matched_max_abs": 0.80,
        "nulls": {
            "perm": {"draws_max_abs": [0.3, 0.5], "empirical_p_one_sided": 0.1},
            "randnorm": {"draws_max_abs": [0.4, 0.6], "empirical_p_one_sided": 0.2},
            "crosstrait": {"draws_max_abs": [0.4, 0.7]},  # max 0.7 < 0.80 -> exceed True
            "pca_topk": {"draws_max_abs": [0.5, 0.9, 0.85]},  # max 0.9 > 0.80 -> exceed False
        },
    }
    nbd._annotate_exceedance(payload)
    n = payload["nulls"]
    assert "exceedance" not in n["perm"], "stochastic must NOT get exceedance"
    assert "exceedance" not in n["randnorm"], "stochastic must NOT get exceedance"
    assert n["crosstrait"]["exceedance"] is True
    assert n["pca_topk"]["exceedance"] is False

    # all-NaN fixed null -> None (undecidable, not a fake False)
    payload2 = {
        "matched_max_abs": 0.5,
        "nulls": {"crosstrait": {"draws_max_abs": [float("nan")]}},
    }
    nbd._annotate_exceedance(payload2)
    assert payload2["nulls"]["crosstrait"]["exceedance"] is None


# ── Off-pod JSONL fetch (reconciler round-1 BLOCKER: primary-deliverable promotion) ──


def test_ensure_jsonls_local_noop_when_all_present(tmp_path):
    """When every required JSONL already exists locally, no HF fetch is attempted."""
    from scripts import issue778_null_battery as nbd

    eval_root = tmp_path / "eval_results" / "issue_778"
    eval_root.mkdir(parents=True)
    for t in ("evil", "sycophancy"):
        (eval_root / f"monitoring_corrected_{t}.jsonl").write_text("{}\n")
    # fetch_from_hf=False would raise if a fetch were needed; all-present -> silent no-op.
    nbd._ensure_monitoring_jsonls_local(
        eval_root,
        ["evil", "sycophancy"],
        ["monitoring_corrected"],
        issue=778,
        slug="persona_vectors",
        fetch_from_hf=False,
    )


def test_ensure_jsonls_local_raises_when_missing_and_no_fetch(tmp_path):
    """A missing JSONL with --no-hf-fetch fails loud (never silently proceeds)."""
    from scripts import issue778_null_battery as nbd

    eval_root = tmp_path / "eval_results" / "issue_778"
    eval_root.mkdir(parents=True)
    (eval_root / "monitoring_corrected_evil.jsonl").write_text("{}\n")  # only 1 of 2 present
    with pytest.raises(RuntimeError, match="absent locally and --no-hf-fetch"):
        nbd._ensure_monitoring_jsonls_local(
            eval_root,
            ["evil", "sycophancy"],
            ["monitoring_corrected"],
            issue=778,
            slug="persona_vectors",
            fetch_from_hf=False,
        )


def test_ensure_jsonls_local_fetches_missing_from_hf(tmp_path, monkeypatch):
    """A missing JSONL is downloaded from the stable HF prefix into eval_root."""
    from scripts import issue778_null_battery as nbd

    eval_root = tmp_path / "eval_results" / "issue_778"
    eval_root.mkdir(parents=True)
    # Stage a fake HF "cache" file the monkeypatched hf_hub_download returns.
    hf_cache = tmp_path / "hfcache"
    hf_cache.mkdir()
    (hf_cache / "monitoring_manyshot_evil.jsonl").write_text('{"condition_id": 5}\n')

    fetched: list[str] = []

    expect_name = (
        "issue778_persona_vectors/followup_corrected/eval_jsonl/monitoring_manyshot_evil.jsonl"
    )

    def _fake_download(*, repo_id, repo_type, filename, revision):
        fetched.append(filename)
        assert filename == expect_name
        return str(hf_cache / "monitoring_manyshot_evil.jsonl")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

    nbd._ensure_monitoring_jsonls_local(
        eval_root,
        ["evil"],
        ["monitoring_manyshot"],
        issue=778,
        slug="persona_vectors",
        fetch_from_hf=True,
    )
    assert len(fetched) == 1
    landed = eval_root / "monitoring_manyshot_evil.jsonl"
    assert landed.exists() and landed.read_text() == '{"condition_id": 5}\n'
