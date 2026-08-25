"""Issue #2564 PE analysis — CPU pins on SYNTHETIC vectors (unit 4).

No HF, no network: exact-math delta checks, the identity-cancellation assert,
the Spearman-Brown step-up, the suppression-rule branches, null_scheme
construction (class/carrier preservation; the NAMED 2-value
sign-randomization null; derangements have no fixed points), the
dyadic/vertex bootstrap convention for query_content, the through-origin
calibration slope, edit-dose residualization, and a tiny synthetic-store
END-TO-END run of the real pipeline asserting the Artifact metadata contract
fields land in ``minpair_delta.json`` (the ``test_issue2564_driver.py``
fixture pattern: real bodies, fakes only at the store/file boundary).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2564_analysis as A  # noqa: E402

from explore_persona_space.experiments.issue2564 import bank2564 as BK  # noqa: E402

# ── exact-math helpers ─────────────────────────────────────────────────


def test_rowwise_cos_exact_and_zero_norm_nan():
    a = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    b = np.array([[2.0, 0.0], [0.0, -1.0], [1.0, -1.0], [1.0, 0.0]])
    out = A.rowwise_cos(a, b)
    np.testing.assert_allclose(out[:3], [1.0, -1.0, 0.0], atol=1e-12)
    assert np.isnan(out[3])  # zero-norm row -> NaN, never a silent 0


def test_through_origin_slope_known_data():
    obs = np.array([1.0, 2.0, 3.0])
    pred = 3.0 * obs
    assert A.through_origin_slope(pred, obs) == pytest.approx(3.0, abs=1e-12)
    # through-origin, NOT ordinary OLS: an intercept-bearing relation is
    # projected through the origin -> sum(p*o)/sum(o^2)
    pred2 = 2.0 * obs + 1.0
    expect = float((pred2 * obs).sum() / (obs * obs).sum())
    assert A.through_origin_slope(pred2, obs) == pytest.approx(expect, abs=1e-12)


def test_spearman_brown_step_up():
    assert A.spearman_brown(0.5) == pytest.approx(2 * 0.5 / 1.5)
    assert A.spearman_brown(0.0) == pytest.approx(0.0)
    assert A.spearman_brown(1.0) == pytest.approx(1.0)
    arr = A.spearman_brown(np.array([0.5, -1.0, 0.2]))
    assert arr[0] == pytest.approx(2 * 0.5 / 1.5)
    assert np.isnan(arr[1])  # r <= -1 undefined -> NaN
    assert arr[2] == pytest.approx(2 * 0.2 / 1.2)


def test_suppression_rule_branches():
    # positive ceiling, CI excluding zero -> NOT suppressed
    assert A.suppression_verdict(0.6, 0.4, 0.8) is False
    # nonpositive ceiling -> suppressed
    assert A.suppression_verdict(0.0, -0.1, 0.1) is True
    assert A.suppression_verdict(-0.2, -0.4, 0.1) is True
    # positive ceiling but CI includes zero -> suppressed
    assert A.suppression_verdict(0.3, -0.05, 0.6) is True
    # non-finite ceiling -> suppressed
    assert A.suppression_verdict(float("nan"), 0.1, 0.2) is True


def test_ols_residualized_gap_recovers_class_offset():
    # ||delta|| = 2 + 0.5*dose + class_offset (flip +1.0, para +0.0), exactly.
    dose = np.array([1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0])
    is_flip = np.array([True] * 4 + [False] * 4)
    # symmetric dose across classes so the pooled OLS recovers slope 0.5 and
    # the class offset survives residualization
    offs = np.where(is_flip, 1.0, 0.0)
    norms = 2.0 + 0.5 * dose + offs
    icpt, slope = A.ols_intercept_slope(norms, dose)
    resid = norms - (icpt + slope * dose)
    raw_gap = norms[is_flip].mean() - norms[~is_flip].mean()
    resid_gap = resid[is_flip].mean() - resid[~is_flip].mean()
    # raw gap carries the dose imbalance (here dose means differ by -0.5*0.5)
    assert raw_gap == pytest.approx(1.0 + 0.5 * (dose[is_flip].mean() - dose[~is_flip].mean()))
    # slope is biased UP by the class/dose confound in pooled OLS, but with
    # symmetric dose ranges the residualized gap still isolates ~the offset
    assert abs(resid_gap - 1.0) < abs(raw_gap - 1.0) + 1e-12
    # pure-dose case: no offset -> residualized gap ~ 0
    norms0 = 2.0 + 0.5 * dose
    i0, s0 = A.ols_intercept_slope(norms0, dose)
    assert (i0, s0) == (pytest.approx(2.0), pytest.approx(0.5))
    r0 = norms0 - (i0 + s0 * dose)
    assert r0[is_flip].mean() - r0[~is_flip].mean() == pytest.approx(0.0, abs=1e-12)


# ── nulls: derangements + the NAMED 2-value scheme ─────────────────────


def test_deranged_perms_no_fixed_points():
    rng = np.random.default_rng(0)
    perms = A.deranged_perms(5, 200, rng)
    assert perms.shape == (200, 5)
    assert (perms != np.arange(5)).all()
    for row in perms[:20]:
        assert sorted(row.tolist()) == [0, 1, 2, 3, 4]
    # n=2 has exactly one derangement
    p2 = A.deranged_perms(2, 8, rng)
    assert (p2 == np.array([1, 0])).all()


def _mk_view(**kw) -> A.AxisView:
    base = dict(
        axis="stance",
        primary_class="swap",
        para_class="instruction_paraphrase",
        primary_idx=np.arange(4),
        para_idx=np.array([], dtype=np.int64),
        install_idx=None,
        famswap_idx=None,
        primary_grid=None,
        famswap_grid=None,
        primary_vps=[],
        null_scheme="",
    )
    base.update(kw)
    return A.AxisView(**base)


def test_direction_null_preserves_carrier_and_class():
    # 2 vps x 2 carriers; obs == pred per pair. Within-carrier cross-vp cos is
    # 1/sqrt(2); cross-CARRIER cos is 0 by construction — so if the null ever
    # paired across carriers, draws would dip below 1/sqrt(2).
    d = 4
    e = np.eye(d)
    grid = np.array([[0, 1], [2, 3]])  # (n_vp=2, n_car=2) pair indices
    delta = np.zeros((4, d))
    delta[0] = e[0]  # vp0, c0
    delta[2] = (e[0] + e[1]) / np.sqrt(2)  # vp1, c0
    delta[1] = e[2]  # vp0, c1
    delta[3] = (e[2] + e[3]) / np.sqrt(2)  # vp1, c1
    view = _mk_view(primary_grid=grid, primary_vps=["v1-v2", "v1-v3"])
    rng = np.random.default_rng(1)
    draws = A.direction_null_draws(view, delta, delta, A.rowwise_cos(delta, delta), 64, rng)
    # n_vp=2 -> the only derangement is the swap; every within-carrier swapped
    # cos is exactly 1/sqrt(2), so every draw equals it — cross-carrier
    # pairing (cos 0) never leaks in.
    np.testing.assert_allclose(draws, 1.0 / np.sqrt(2.0), atol=1e-12)


def test_direction_null_two_value_axis_is_named_sign_randomization():
    # 2-value axis: primary grid has ONE vp row -> the scheme must be the
    # NAMED sign-randomization null, never a silent fallback.
    grid = np.array([[0, 1, 2]])  # (1 vp, 3 carriers)
    delta = np.tile(np.array([1.0, 0.0, 0.0]), (3, 1))
    view = _mk_view(primary_grid=grid, primary_idx=np.arange(3), primary_vps=["a-b"])
    cos_sel = np.ones(3)
    rng = np.random.default_rng(2)
    draws = A.direction_null_draws(view, delta, delta, cos_sel, 512, rng)
    # draws are means of +/-1 signs over 3 pairs: values in {-1,-1/3,1/3,1}
    assert set(np.round(draws, 6)) <= {-1.0, round(-1 / 3, 6), round(1 / 3, 6), 1.0}
    assert abs(draws.mean()) < 0.2  # centered at 0
    # and the registered scheme STRING for a 2-value axis names the scheme
    pa = _tiny_two_value_pair_arrays()
    views = A.build_axis_views(pa, n_car=3)
    assert "sign_randomization_2value" in views["register"].null_scheme
    assert "derangement" in views["stance"].null_scheme
    assert views["register"].null_scheme != views["stance"].null_scheme


def _tiny_two_value_pair_arrays() -> A.PairArrays:
    """Hand-built PairArrays: register (2 values -> 1 swap vp) + stance
    (3 values -> 3 swap vps), 3 carriers, complete grids."""
    ids, cls, axis, va, vb, cstr, orient = [], [], [], [], [], [], []
    a_i, b_i, ca_i = [], [], []
    row = 0
    for ax, vals in (("register", ["r1", "r2"]), ("stance", ["s1", "s2", "s3"])):
        vps = [(x, y) for k, x in enumerate(vals) for y in vals[k + 1 :]]
        for x, y in vps:
            for c in range(3):
                ids.append(f"swap::{ax}::{x}-{y}::c{c + 1:02d}")
                cls.append("swap")
                axis.append(ax)
                va.append(x)
                vb.append(y)
                cstr.append(f"c{c + 1:02d}")
                orient.append(f"{x}->{y}")
                a_i.append(row)
                b_i.append(row)  # ctx rows unused by build_axis_views
                ca_i.append(c)
                row += 1
    n = len(ids)
    return A.PairArrays(
        ids=ids,
        cls=cls,
        axis=axis,
        value_a=va,
        value_b=vb,
        carrier_str=cstr,
        a=np.array(a_i),
        b=np.array(b_i),
        ca=np.array(ca_i),
        cb=np.array(ca_i),
        dyad=np.zeros(n, dtype=bool),
        changed=np.ones(n, dtype=np.int64),
        orientation=orient,
        n=n,
    )


def test_grid_for_asserts_completeness():
    pa = _tiny_two_value_pair_arrays()
    sel = np.array([i for i in range(pa.n) if pa.axis[i] == "stance"], dtype=np.int64)
    grid, vps = A._grid_for(pa, sel, n_car=3)
    assert grid.shape == (3, 3) and len(vps) == 3
    with pytest.raises(AssertionError, match="incomplete"):
        A._grid_for(pa, sel[:-1], n_car=3)  # drop one cell -> hole


# ── dyadic / carrier-clustered bootstrap ───────────────────────────────


def test_dyadic_bootstrap_touches_both_endpoint_carriers():
    # 3 carriers, edges (0,1), (0,2), (1,2). A draw that never samples
    # carrier 2 must zero BOTH edges incident to it.
    mult = np.array([[2.0, 1.0, 0.0]])
    ca = np.array([0, 0, 1])
    cb = np.array([1, 2, 2])
    w = A.dyad_pair_weights(mult, ca, cb)
    np.testing.assert_allclose(w, [[2.0, 0.0, 0.0]])
    # weighted mean matches the brute-force product-multiplicity form
    vals = np.array([10.0, 20.0, 30.0])
    dyad = np.ones(3, dtype=bool)
    out = A.boot_weighted_mean(vals, ca, cb, dyad, mult)
    np.testing.assert_allclose(out, [10.0])  # only edge (0,1) survives


def test_boot_pair_sums_matches_bruteforce_mixed_selection():
    rng = np.random.default_rng(3)
    n_car, n = 4, 12
    ca = rng.integers(0, n_car, n)
    cb = ca.copy()
    dyad = np.zeros(n, dtype=bool)
    dyad[:4] = True
    cb[:4] = (ca[:4] + 1) % n_car  # dyads get a distinct second carrier
    vals = rng.standard_normal(n)
    idx = rng.integers(0, n_car, size=(16, n_car))
    mult = A.carrier_multiplicities(idx, n_car)
    got = A.boot_pair_sums(vals, ca, cb, dyad, mult)
    w = np.where(dyad[None, :], mult[:, ca] * mult[:, cb], mult[:, ca])
    np.testing.assert_allclose(got, (w * vals[None, :]).sum(axis=1), atol=1e-10)


def test_carrier_multiplicities_sum_to_n():
    idx = np.random.default_rng(4).integers(0, 5, size=(32, 5))
    mult = A.carrier_multiplicities(idx, 5)
    assert (mult.sum(axis=1) == 5).all()
    assert (mult >= 0).all()


# ── identity-cancellation assert ───────────────────────────────────────


def test_identity_cancellation_check_passes_and_reports():
    rng = np.random.default_rng(5)
    vc = rng.standard_normal((10, 6))
    a = np.array([0, 1, 2, 3])
    b = np.array([4, 5, 6, 7])
    out = A.identity_cancellation_check(vc, a, b, np.random.default_rng(6), n_check=4)
    assert out["n_pairs_checked"] == 4
    assert out["max_abs_err"] <= out["tol"]
    assert "identity_bias_predict" in out["statement"]


# ── compliance masks ───────────────────────────────────────────────────


def test_pair_fired_mask_uses_verdicts_and_defaults_unchecked_to_fired():
    pa = _tiny_two_value_pair_arrays()
    fire = {
        "fired": {
            70: {("register", "r1"): True, ("register", "r2"): False},
            50: {("register", "r1"): True, ("register", "r2"): True},
            90: {("register", "r1"): False, ("register", "r2"): False},
        }
    }
    fa70, fb70 = A.pair_fired_mask(pa, fire, 70)
    reg = np.array([i for i in range(pa.n) if pa.axis[i] == "register"])
    stance = np.array([i for i in range(pa.n) if pa.axis[i] == "stance"])
    assert fa70[reg].all()  # r1 fired
    assert not fb70[reg].any()  # r2 not fired at 70
    _fa50, fb50 = A.pair_fired_mask(pa, fire, 50)
    assert fb50[reg].all()  # r2 fired at the 50 sensitivity threshold
    # stance has NO fire rows -> unfiltered (counts fired)
    assert fa70[stance].all() and fb70[stance].all()


# ── split-half reliability ─────────────────────────────────────────────


def _mk_stores_for_split(draws: np.ndarray, valid: np.ndarray) -> A.Stores:
    n_ctx, _, d = draws.shape
    zeros = {layer: np.zeros((n_ctx, d)) for layer in A.LAYERS}
    return A.Stores(
        ctx_ids=[f"x{i}" for i in range(n_ctx)],
        row_of={f"x{i}": i for i in range(n_ctx)},
        cells=["stance"],
        carriers=["c01"],
        va_tail_mean=zeros,
        va_span_mean=zeros,
        tail_draws=draws.astype(np.float32),
        draw_valid=valid,
        n_valid=valid.sum(axis=1),
        ans_len_mean=np.ones(n_ctx),
        vc=zeros,
        emb_mean=np.zeros((n_ctx, 4)),
        d=d,
    )


def test_split_half_identical_draws_gives_ceiling_one_and_zero_noise():
    rng = np.random.default_rng(7)
    base = rng.standard_normal((3, 1, 4))
    draws = np.repeat(base, 4, axis=1)  # 4 identical draws per context
    valid = np.ones((3, 4), dtype=bool)
    st = _mk_stores_for_split(draws, valid)
    pa = A.PairArrays(
        ids=["p0"],
        cls=["swap"],
        axis=["stance"],
        value_a=["s1"],
        value_b=["s2"],
        carrier_str=["c01"],
        a=np.array([0]),
        b=np.array([1]),
        ca=np.array([0]),
        cb=np.array([0]),
        dyad=np.array([False]),
        changed=np.array([1]),
        orientation=["s1->s2"],
        n=1,
    )
    out = A.split_half_stats(st, pa, n_splits=4)
    assert out["r_half"][0] == pytest.approx(1.0, abs=1e-6)
    assert out["r_full"][0] == pytest.approx(1.0, abs=1e-6)
    assert out["noise_norm"][0] == pytest.approx(0.0, abs=1e-6)
    assert out["n_pairs_insufficient_draws"] == 0


def test_split_half_insufficient_draws_is_nan_and_counted():
    draws = np.random.default_rng(8).standard_normal((2, 4, 3))
    valid = np.ones((2, 4), dtype=bool)
    valid[1, 1:] = False  # context 1 has a single valid draw
    st = _mk_stores_for_split(draws, valid)
    pa = A.PairArrays(
        ids=["p0"],
        cls=["swap"],
        axis=["stance"],
        value_a=["s1"],
        value_b=["s2"],
        carrier_str=["c01"],
        a=np.array([0]),
        b=np.array([1]),
        ca=np.array([0]),
        cb=np.array([0]),
        dyad=np.array([False]),
        changed=np.array([1]),
        orientation=["s1->s2"],
        n=1,
    )
    out = A.split_half_stats(st, pa, n_splits=2)
    assert np.isnan(out["r_half"][0]) and np.isnan(out["noise_norm"][0])
    assert out["n_pairs_insufficient_draws"] == 1


# ── tiny synthetic end-to-end (real pipeline, file-boundary fakes) ─────

SMOKE_CELLS = ("register", "query")
SMOKE_CARRIERS = ("c01", "c02", "c03")
D_HID = 8
K_DRAWS = 4


def _synth_bank() -> dict:
    values = BK.load_values()
    contexts = BK.build_contexts(values)
    pairs = BK.build_pairs(values, contexts)
    for i, p in enumerate(pairs):
        p["changed_tokens"] = 1 + (i % 7)  # synthetic edit dose (tokenizer-free)
    return {
        "issue": BK.ISSUE,
        "n_contexts": len(contexts),
        "n_pairs": len(pairs),
        "contexts": contexts,
        "pairs": pairs,
    }


def _smoke_ctx_ids(bank: dict) -> list[str]:
    return sorted(
        cid
        for cid, c in bank["contexts"].items()
        if c["cell"] in SMOKE_CELLS and c["carrier"] in SMOKE_CARRIERS
    )


@pytest.fixture(scope="module")
def e2e_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("i2564_pe")
    in_root = root / "in"
    out_dir = root / "out"
    bank = _synth_bank()
    (in_root / "manifests").mkdir(parents=True)
    (in_root / "manifests" / "bank2564_manifest.json").write_text(json.dumps(bank))

    ctx_ids = _smoke_ctx_ids(bank)
    n_ctx = len(ctx_ids)
    gen = torch.Generator().manual_seed(2564)
    vc = torch.randn(n_ctx, len(A.LAYERS), D_HID, generator=gen, dtype=torch.float32)
    vc_dir = in_root / "analysis_tensors" / "vc2564"
    vc_dir.mkdir(parents=True)
    torch.save(
        {"issue": 2564, "layers": list(A.LAYERS), "context_ids": ctx_ids, "vc": vc},
        vc_dir / "vc2564_bank.pt",
    )

    va_dir = in_root / "analysis_tensors" / "va2564"
    va_dir.mkdir(parents=True)
    row_of = {cid: i for i, cid in enumerate(ctx_ids)}
    for cell in SMOKE_CELLS:
        cell_ids = [cid for cid in ctx_ids if bank["contexts"][cid]["cell"] == cell]
        index, tail_rows, span_rows = [], [], []
        for cid in cell_ids:
            for draw in range(K_DRAWS):
                index.append(
                    {
                        "context_id": cid,
                        "cell": cell,
                        "draw": draw,
                        "ctx_len": 10,
                        "n_completion_tokens": 5 + row_of[cid] % 3,
                        "span_start": 10,
                        "span_end": 15,
                        "tail_end": 17,
                    }
                )
                tail_rows.append(vc[row_of[cid]])  # va == vc exactly (identity world)
                span_rows.append(vc[row_of[cid]] * 1.5)
        torch.save(
            {
                "cell": cell,
                "layers": list(A.LAYERS),
                "index": index,
                "va_span": torch.stack(span_rows).to(torch.float16),
                "va_tail_incl": torch.stack(tail_rows).to(torch.float16),
                "empty_rows": [],
            },
            va_dir / f"va2564_{cell}.pt",
        )

    emb_dir = in_root / "analysis_tensors" / "embeddings_qwen3_8b"
    emb_dir.mkdir(parents=True)
    rng = np.random.default_rng(11)
    np.savez(
        emb_dir / "means_anchors.npz",
        emb_mean=rng.standard_normal((n_ctx, 16)).astype(np.float16),
        context_ids=np.array(ctx_ids),
        n_draws=np.full(n_ctx, K_DRAWS, dtype=np.int32),
    )

    # manipulation check: register values + paraphrases all fired
    values = BK.load_values()
    value_rows = []
    for vid in BK.value_ids(values, "register"):
        for v in (vid, f"{vid}p"):
            value_rows.append(
                {
                    "axis": "register",
                    "value_id": v,
                    "kind": "orig" if v == vid else "para",
                    "instrument": "judged",
                    "n_comply": 24,
                    "n_noncomply": 0,
                    "n_incomplete": 0,
                    "denom": 24,
                    "comply_frac": 1.0,
                    "verdict": "fired",
                    "sensitivity": {"50": "fired", "90": "fired"},
                }
            )
    manip = root / "manipulation_check.json"
    manip.write_text(
        json.dumps(
            {
                "meta": {
                    "judged_denominator": 24,
                    "fire_threshold_pct": 70,
                    "floor_rule": "n_fired_base >= ceil(0.6 * width)",
                },
                "value_rows": value_rows,
                "axis_rows": [
                    {
                        "axis": "register",
                        "width": 2,
                        "floor": 2,
                        "n_fired_base": 2,
                        "floor_met": True,
                    }
                ],
            }
        )
    )

    # frozen ridge payloads: W = identity, bias cancels -> arm == iddelta
    maps_dir = root / "maps"
    maps_dir.mkdir()
    for name in ("ridge_779.pt", "ridge_1738.pt"):
        torch.save(
            {
                "kind": "ridge",
                "xmu": torch.zeros(D_HID),
                "xsd": torch.ones(D_HID),
                "ymu": torch.randn(D_HID, generator=gen),
                "W": torch.eye(D_HID),
            },
            maps_dir / name,
        )

    rc = A.main(
        [
            "--smoke",
            "--in-root",
            str(in_root),
            "--out-dir",
            str(out_dir),
            "--manip-check",
            str(manip),
            "--ridge-779",
            str(maps_dir / "ridge_779.pt"),
            "--ridge-1738",
            str(maps_dir / "ridge_1738.pt"),
            "--upload",
            "none",
            "--b-boot",
            "50",
            "--b-null",
            "50",
            "--n-splits",
            "4",
        ]
    )
    assert rc == 0
    doc = json.loads((out_dir / "minpair_delta.json").read_text())
    rows = [
        json.loads(line)
        for line in (out_dir / "perpair.jsonl").read_text().split("\n")
        if line.strip()
    ]
    return doc, rows, out_dir


def test_e2e_metadata_contract_fields_present(e2e_run):
    doc, _, _ = e2e_run
    contract = doc["contract"]
    for dv in (
        "direction_fidelity",
        "magnitude_calibration",
        "axis_identity",
        "cross_family_consistency",
        "reliability_ceiling",
        "text_third_space",
        "surface_sensitivity",
        "knn_delta_retrieval",
    ):
        assert contract["primary_pair_classes"][dv]["instruction_axes"] == ["swap"]
    assert set(contract["null_scheme"]) == {"register", "query_content", "query_form"}
    assert "sign_randomization_2value" in contract["null_scheme"]["register"]
    assert "derangement" in contract["null_scheme"]["query_content"]
    assert "product" in contract["bootstrap"]["query_content"]  # dyadic/vertex convention
    assert contract["bootstrap"]["seed"] == 2215
    assert contract["null"]["seed"] == 21620
    assert contract["split_half"]["seed"] == 2564
    assert "K=10-draw mean" in contract["draw_to_pair_aggregation"]
    assert set(contract["orientation_conventions"]) == set(A.ORIENTATION_CONVENTIONS)
    assert doc["meta"]["identity_cancellation_assert"]["max_abs_err"] <= 1e-6
    assert "git_commit" in doc["meta"]


def test_e2e_axes_and_known_value_reads(e2e_run):
    doc, _, _ = e2e_run
    axes = doc["axes"]
    assert set(axes) == {"register", "query_content", "query_form"}
    reg = axes["register"]
    # identity world: va == vc and W == I -> every arm reads direction cos ~ 1
    for arm in A.ARMS:
        assert reg["direction"][arm]["mean_cos_headline"] > 0.99
        # identical draws -> ceiling exactly 1 -> normalized read NOT suppressed
        assert reg["direction"][arm]["ceiling_suppressed"] is False
        assert reg["direction"][arm]["ceiling_normalized_cos"] > 0.99
        assert reg["calibration"][arm]["ratio_to_global"] == pytest.approx(1.0, abs=0.05)
    assert reg["reliability"]["r10_mean"] == pytest.approx(1.0, abs=1e-6)
    assert reg["fire"]["compliance_limited"] is False
    assert reg["fire"]["n_headline_pairs_fired70"] == 3  # 1 swap vp x 3 carriers
    # span twin: span store = 1.5x tail -> direction unchanged, slope scaled
    assert reg["pooling_twin_span"]["arm_iddelta"]["mean_cos_headline"] > 0.99
    # query axes carry reads too (no fire rows -> unfiltered)
    assert axes["query_content"]["identity"].get("n/a")
    assert axes["query_form"]["direction"]["arm_779ce"]["mean_cos_headline"] > 0.99
    # retrieval table exists for all arms
    assert set(doc["retrieval"]["global"]) == set(A.ARMS)
    acc = doc["retrieval"]["global"]["arm_779ce"]["cosine"]["acc_at_k"]
    assert acc["1"] > 0.9  # identity world: predictions retrieve their own target


def test_e2e_perpair_rows_carry_contract_fields(e2e_run):
    _, rows, _ = e2e_run
    # smoke slice: register install 6 + swap 3 + famswap 3 + para 6;
    # query_content C(3,2)=3 + query_form 3x3=9 + query_paraphrase 3 = 33
    assert len(rows) == 33
    classes = {r["pair_class"] for r in rows}
    assert classes == {
        "install",
        "swap",
        "famswap",
        "instruction_paraphrase",
        "query_content",
        "query_form",
        "query_paraphrase",
    }
    for r in rows:
        assert set(r) >= {
            "pair_id",
            "pair_class",
            "axis",
            "orientation",
            "changed_tokens",
            "cos",
            "r10",
            "fired_a_70",
            "in_headline_70",
        }
        assert r["changed_tokens"] >= 1
    qc = [r for r in rows if r["pair_class"] == "query_content"]
    assert all("->" in r["orientation"] and "|" in r["carrier"] for r in qc)


def test_e2e_prediction_tensors_written(e2e_run):
    _, _, out_dir = e2e_run
    pred_dir = out_dir / "predictions"
    names = sorted(p.name for p in pred_dir.glob("*.pt"))
    assert names == [
        "delta_obs_span_L19.pt",
        "delta_obs_tail_L19.pt",
        "delta_pred_arm_1738ce.pt",
        "delta_pred_arm_779ce.pt",
        "delta_pred_arm_iddelta.pt",
    ]
    blob = torch.load(pred_dir / "delta_pred_arm_iddelta.pt", weights_only=False)
    assert blob["tensor"].shape == (33, D_HID)
    assert len(blob["pair_ids"]) == 33 and blob["layer"] == 19


def test_import_check_mode_returns_zero():
    assert A.main(["--import-check"]) == 0


# ── r2 blocker 7: floor-gated headline + gap-vs-iddelta reads ──────────


def _manip_doc(base_verdict: str, floor_met: bool, n_fired_base: int) -> dict:
    """Manipulation-check doc with register base values at ``base_verdict``
    (paraphrase values always fired) and an explicit axis floor verdict."""
    values = BK.load_values()
    value_rows = []
    for vid in BK.value_ids(values, "register"):
        for v in (vid, f"{vid}p"):
            verdict = base_verdict if v == vid else "fired"
            value_rows.append(
                {
                    "axis": "register",
                    "value_id": v,
                    "kind": "orig" if v == vid else "para",
                    "instrument": "judged",
                    "n_comply": 24 if verdict == "fired" else 0,
                    "n_noncomply": 0 if verdict == "fired" else 24,
                    "n_incomplete": 0,
                    "denom": 24,
                    "comply_frac": 1.0 if verdict == "fired" else 0.0,
                    "verdict": verdict,
                    "sensitivity": {"50": verdict, "90": verdict},
                }
            )
    return {
        "meta": {
            "judged_denominator": 24,
            "fire_threshold_pct": 70,
            "floor_rule": "n_fired_base >= ceil(0.6 * width)",
        },
        "value_rows": value_rows,
        "axis_rows": [
            {
                "axis": "register",
                "width": 2,
                "floor": 2,
                "n_fired_base": n_fired_base,
                "floor_met": floor_met,
            }
        ],
    }


def _rerun_with_manip(e2e_out_dir: Path, manip: Path, out2: Path) -> dict:
    """Re-run the REAL pipeline on the e2e fixture's inputs with a different
    manipulation-check file (the only lever under test)."""
    root = e2e_out_dir.parent
    maps_dir = root / "maps"
    rc = A.main(
        [
            "--smoke",
            "--in-root",
            str(root / "in"),
            "--out-dir",
            str(out2),
            "--manip-check",
            str(manip),
            "--ridge-779",
            str(maps_dir / "ridge_779.pt"),
            "--ridge-1738",
            str(maps_dir / "ridge_1738.pt"),
            "--upload",
            "none",
            "--b-boot",
            "50",
            "--b-null",
            "50",
            "--n-splits",
            "4",
        ]
    )
    assert rc == 0
    return json.loads((out2 / "minpair_delta.json").read_text())


def test_e2e_below_floor_axis_nulls_headline_despite_fired_pairs(e2e_run, tmp_path):
    """r2 blocker 7 (compliance-floor-not-enforced): headline gating keys on
    ``axis_row.floor_met`` — a below-floor axis nulls EVERY headline field even
    when individual pairs fired; ``*_all_values`` companions stay populated."""
    _, _, e2e_out = e2e_run
    manip = tmp_path / "manip_belowfloor.json"
    manip.write_text(json.dumps(_manip_doc("fired", floor_met=False, n_fired_base=0)))
    doc = _rerun_with_manip(e2e_out, manip, tmp_path / "out_belowfloor")
    reg = doc["axes"]["register"]
    fire = reg["fire"]
    assert fire["compliance_limited"] is True
    assert fire["headline_ok"] is False
    assert fire["no_fired_pairs"] is False  # pairs DID fire — the FLOOR is what gates
    assert fire["n_headline_pairs_fired70"] == 3
    # NaN headline fields serialize to null (_json_sanitize) — assert None.
    d = reg["direction"]["arm_779ce"]
    assert d["mean_cos_headline"] is None
    assert d["mean_cos_all_values"] > 0.99
    assert d["ceiling_suppressed"] is True and d["ceiling_normalized_cos"] is None
    r = reg["reliability"]
    assert r["r10_mean"] is None
    assert r["r10_mean_all_values"] == pytest.approx(1.0, abs=1e-6)
    assert reg["identity"]["arm_779ce"]["median"] is None
    assert reg["identity"]["arm_779ce"]["median_all_values"] > 0.99
    assert reg["cross_family"]["observed"]["median"] is None
    assert isinstance(reg["cross_family"]["observed"]["median_all_values"], float)
    assert reg["text_space"]["flip_norm_mean"] is None
    assert isinstance(reg["text_space"]["flip_norm_mean_all_values"], float)
    assert reg["surface"]["observed"]["flip_norm_mean"] is None
    assert isinstance(reg["surface"]["observed"]["flip_norm_mean_all_values"], float)


def test_e2e_zero_fired_pairs_nulls_headline_no_prim_fallback(e2e_run, tmp_path):
    """r2 blocker 7 regression: floor met but ZERO fired pairs — the r1 code
    silently fell back to ``head = prim`` (unfiltered); the headline must null
    with ``no_fired_pairs`` set instead."""
    _, _, e2e_out = e2e_run
    manip = tmp_path / "manip_nofired.json"
    manip.write_text(json.dumps(_manip_doc("not_fired", floor_met=True, n_fired_base=2)))
    doc = _rerun_with_manip(e2e_out, manip, tmp_path / "out_nofired")
    reg = doc["axes"]["register"]
    fire = reg["fire"]
    assert fire["no_fired_pairs"] is True
    assert fire["compliance_limited"] is False
    assert fire["headline_ok"] is False
    assert fire["n_headline_pairs_fired70"] == 0
    # r1 fallback would have produced the unfiltered ~1.0 here; nulled now.
    assert reg["direction"]["arm_779ce"]["mean_cos_headline"] is None
    assert reg["direction"]["arm_779ce"]["mean_cos_all_values"] > 0.99


def test_e2e_gap_vs_iddelta_schema_and_identity_world_zero(e2e_run):
    """r2 concern map-iddelta-gap-missing: ridge arms carry paired gap-vs-
    identity reads (direction + identity medians). Identity world (W == I,
    bias cancels) makes every gap exactly ~0."""
    doc, _, _ = e2e_run
    reg = doc["axes"]["register"]
    for arm in ("arm_779ce", "arm_1738ce"):
        g = reg["direction"][arm]["gap_vs_iddelta"]
        assert set(g) >= {"mean_cos_gap_headline", "ci95", "mean_cos_gap_all_values"}
        assert g["mean_cos_gap_headline"] == pytest.approx(0.0, abs=1e-6)
        mg = reg["identity"][arm]["median_gap_vs_iddelta"]
        assert mg is not None
        assert mg["gap"] == pytest.approx(0.0, abs=1e-6)
    assert "gap_vs_iddelta" not in reg["direction"]["arm_iddelta"]
    assert "median_gap_vs_iddelta" not in reg["identity"]["arm_iddelta"]


def test_smoke_requires_explicit_manip_check():
    """r2 [g5]: --smoke without --manip-check refuses — the smoke run must
    never silently gate on the committed PRODUCTION manipulation_check.json."""
    with pytest.raises(SystemExit, match="manip-check"):
        A.build_config(A.parse_args(["--smoke", "--upload", "none"]))
