"""CPU-only synthetic-tensor tests for the issue #2254 pre-image driver algebra.

Covers (unit-2 brief): the k* rule, pinv truncation, the frame-fold identity on
a tiny exact synthetic case (d=8, n=32), the de-standardization fold, the
shuffled matched-k* variant (+ the k*_shuffled==0 steering fallback), the
direction-3 diff-of-means, the sha-assert failure path, and the
eval_questions-length assert.

NO test reads `eval_results/issue_<M>/` fixtures (sparse-cones rule) and no
test touches the network / GPU — synthetic tensors only.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import scripts.issue2254_preimage as pi

RNG = np.random.default_rng(0)


def _tiny_fit(n=32, d=8, noise=0.05, seed=1):
    """Synthetic X, Y = X @ A + noise, fitted through the VERBATIM ridge."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    a = rng.standard_normal((d, d))
    y = x @ a + noise * rng.standard_normal((n, d))
    return x, y, pi.ridge_fit_matrix(x, y)


# ---------------------------------------------------------------------------
# k* rule
# ---------------------------------------------------------------------------


def test_kstar_rule_counts_squared_singular_values():
    s = np.array([3.0, 2.0, 1.0, 0.5])
    assert pi.kstar_from_fit(s, 1.1) == 2  # s^2 = [9, 4, 1, 0.25] >= 1.1 -> two
    assert pi.kstar_from_fit(s, 0.9) == 3
    assert pi.kstar_from_fit(s, 100.0) == 0
    assert pi.kstar_from_fit(s, 0.0) == 4


def test_kstar_matches_fit_lambda_semantics():
    _x, _y, fit = _tiny_fit()
    k = pi.kstar_from_fit(fit["s"], fit["lam"])
    assert k == int(np.sum(fit["s"] ** 2 >= fit["lam"]))
    assert 0 <= k <= len(fit["s"])


# ---------------------------------------------------------------------------
# pinv truncation
# ---------------------------------------------------------------------------


def test_preimage_w_reproduces_truncated_pinv_on_diagonal_map():
    # M = diag(4, 2, 1): pinv_k inverts only the top-k singular directions.
    w_diag = np.diag([4.0, 2.0, 1.0]).T  # W with M = W.T = diag
    _m, um, sm, vmt = pi.map_svd(w_diag)
    r = np.array([1.0, 1.0, 1.0])
    w2 = pi.preimage_w(um, sm, vmt, r, 2)
    # top-2 singular directions are e1, e2 with s = 4, 2 -> w = (1/4, 1/2, 0)
    assert np.allclose(np.sort(np.abs(w2[np.abs(w2) > 1e-12])), [0.25, 0.5])
    assert np.allclose((np.diag([4.0, 2.0, 1.0]) @ w2), [1.0, 1.0, 0.0], atol=1e-12)
    w3 = pi.preimage_w(um, sm, vmt, r, 3)
    assert np.allclose(np.diag([4.0, 2.0, 1.0]) @ w3, r, atol=1e-12)


def test_preimage_w_k_clamped_and_zero_raises():
    _x, _y, fit = _tiny_fit()
    _m, um, sm, vmt = pi.map_svd(fit["W"])
    r = RNG.standard_normal(um.shape[0])
    # k beyond the spectrum clamps to full rank instead of crashing
    w_full = pi.preimage_w(um, sm, vmt, r, 10 * len(sm))
    assert w_full.shape == (vmt.shape[1],)
    with pytest.raises(ValueError, match="degenerate map"):
        pi.preimage_w(um, sm, vmt, r, 0)


# ---------------------------------------------------------------------------
# frame-fold identity (exact tiny case) + de-standardization fold
# ---------------------------------------------------------------------------


def test_frame_fold_identity_exact_tiny_case():
    """cos(M @ (d_pre / xsd), P_k(r_B)) == 1 exactly (up to float) by algebra:
    M @ pinv_k(M) is the projector onto the top-k column space of M."""
    _x, _y, fit = _tiny_fit(n=32, d=8)
    m, um, sm, vmt = pi.map_svd(fit["W"])
    r_b = np.random.default_rng(7).standard_normal(8)
    for k in (1, 3, 5, 8):
        w = pi.preimage_w(um, sm, vmt, r_b, k)
        d_pre = pi.destandardized_direction(fit["xsd"], w)
        cos = pi.frame_fold_cos(m, um, fit["xsd"], d_pre, r_b, k)
        assert cos > 1.0 - 1e-10, (k, cos)


def test_frame_fold_detects_wrong_frame():
    """Skipping the /xsd unfold (i.e. folding a WRONG frame) must break the
    identity — the gate has teeth."""
    _x, _y, fit = _tiny_fit(n=32, d=8, seed=3)
    # make xsd strongly anisotropic so the wrong frame visibly diverges
    fit = dict(fit)
    fit["xsd"] = np.linspace(0.05, 8.0, 8)
    m, um, sm, vmt = pi.map_svd(fit["W"])
    r_b = np.random.default_rng(11).standard_normal(8)
    k = 4
    w = pi.preimage_w(um, sm, vmt, r_b, k)
    d_pre = pi.destandardized_direction(fit["xsd"], w)
    # wrong frame: pass xsd=1 (no unfold) with the folded direction
    cos_wrong = pi.frame_fold_cos(m, um, np.ones(8), d_pre, r_b, k)
    assert cos_wrong < 0.999


def test_destandardization_fold_shape_and_normalization():
    xsd = np.array([1.0, 2.0, 4.0])
    w = np.array([1.0, 1.0, 1.0])
    d = pi.destandardized_direction(xsd, w)
    expected = np.array([1.0, 2.0, 4.0]) / np.linalg.norm([1.0, 2.0, 4.0])
    assert np.allclose(d, expected)
    assert np.isclose(np.linalg.norm(d), 1.0)
    with pytest.raises(ValueError, match="degenerate norm"):
        pi.destandardized_direction(xsd, np.zeros(3))


def test_proj_fraction_bounds_and_full_rank():
    _x, _y, fit = _tiny_fit(seed=5)
    _m, um, _sm, _vmt = pi.map_svd(fit["W"])
    r_b = np.random.default_rng(5).standard_normal(8)
    fr1 = pi.proj_fraction(um, r_b, 1)
    fr8 = pi.proj_fraction(um, r_b, 8)
    assert 0.0 <= fr1 <= fr8 <= 1.0 + 1e-12
    assert np.isclose(fr8, 1.0, atol=1e-10)  # full-rank projector keeps all of r_B


# ---------------------------------------------------------------------------
# shuffled-map control (primary / matched-k* / fallback)
# ---------------------------------------------------------------------------


def test_shuffled_bundle_primary_uses_own_kstar():
    x, y, fit = _tiny_fit(n=64, d=8, seed=9)
    perm = np.random.default_rng(pi.SEED_SHUFFLE).permutation(64)
    fit_shuf = pi.ridge_fit_matrix(x, y[perm])
    kstar_real = pi.kstar_from_fit(fit["s"], fit["lam"])
    r_b = np.random.default_rng(2).standard_normal(8)
    bundle = pi.shuffled_direction_bundle(fit_shuf, max(kstar_real, 1), r_b)
    assert bundle["kstar_shuffled"] == pi.kstar_from_fit(fit_shuf["s"], fit_shuf["lam"])
    if bundle["kstar_shuffled"] > 0:
        assert not bundle["fallback_matched_kstar"]
        assert bundle["d_preshuf_primary"] is not None
        assert np.allclose(bundle["d_preshuf_steering"], bundle["d_preshuf_primary"])
    # both persisted variants are unit vectors
    assert np.isclose(np.linalg.norm(bundle["d_preshuf_matched"]), 1.0)
    assert np.isclose(np.linalg.norm(bundle["d_preshuf_steering"]), 1.0)


def test_shuffled_bundle_kstar_zero_falls_back_to_matched():
    """k*_shuffled == 0 (every s^2 < lambda) -> primary None, steering =
    matched-k* variant, fallback flag set — never normalize(0)."""
    _x, _y, fit = _tiny_fit(n=32, d=8, seed=13)
    fit_shuf = dict(fit)
    fit_shuf["lam"] = float(np.max(fit["s"]) ** 2 * 10.0)  # above every s^2
    r_b = np.random.default_rng(3).standard_normal(8)
    bundle = pi.shuffled_direction_bundle(fit_shuf, kstar_real=4, r_b=r_b)
    assert bundle["kstar_shuffled"] == 0
    assert bundle["fallback_matched_kstar"] is True
    assert bundle["d_preshuf_primary"] is None
    assert np.allclose(bundle["d_preshuf_steering"], bundle["d_preshuf_matched"])
    assert np.isclose(np.linalg.norm(bundle["d_preshuf_steering"]), 1.0)


# ---------------------------------------------------------------------------
# direction 3: diff of means
# ---------------------------------------------------------------------------


def test_diff_of_means_direction_recovers_planted_delta():
    rng = np.random.default_rng(21)
    n, n_layers, h = 40, 3, 16
    base = rng.standard_normal((n, n_layers, h))
    delta = rng.standard_normal((n_layers, h))
    pos = base + delta[None]
    neg = base
    d = pi.diff_of_means_direction(pos, neg)
    assert d.shape == (n_layers, h)
    for li in range(n_layers):
        assert np.isclose(np.linalg.norm(d[li]), 1.0)
        cos = d[li] @ delta[li] / np.linalg.norm(delta[li])
        assert cos > 0.999999, (li, cos)


def test_diff_of_means_direction_degenerate_raises():
    same = np.zeros((4, 2, 8))
    with pytest.raises(ValueError, match="degenerate"):
        pi.diff_of_means_direction(same, same)


# ---------------------------------------------------------------------------
# sha-assert + e1 eval-bank asserts (stage_inputs gates)
# ---------------------------------------------------------------------------


def test_assert_sha256_pass_and_failure_path(tmp_path):
    p = tmp_path / "asset.json"
    p.write_bytes(b'{"eval_questions": []}')
    good = hashlib.sha256(p.read_bytes()).hexdigest()
    pi.assert_sha256(p, good, what="asset.json")  # no raise
    bad = "0" * 64
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        pi.assert_sha256(p, bad, what="asset.json")


def test_assert_e1_eval_bank_length_20():
    ok = {"eval_questions": [f"q{i}" for i in range(20)]}
    pi.assert_e1_eval_bank(ok, "sycophancy")  # no raise
    with pytest.raises(RuntimeError, match="eval_questions invalid"):
        pi.assert_e1_eval_bank({"eval_questions": [f"q{i}" for i in range(19)]}, "sycophancy")
    with pytest.raises(RuntimeError, match="eval_questions invalid"):
        pi.assert_e1_eval_bank({}, "hallucination")
    with pytest.raises(RuntimeError, match="eval_questions invalid"):
        pi.assert_e1_eval_bank({"eval_questions": "not-a-list"}, "evil")


# ---------------------------------------------------------------------------
# random control + unit_rows + r2 helper
# ---------------------------------------------------------------------------


def test_random_direction_deterministic_unit():
    a = pi.random_direction(64, seed=pi.SEED_RANDOM_BASE + 14)
    b = pi.random_direction(64, seed=pi.SEED_RANDOM_BASE + 14)
    c = pi.random_direction(64, seed=pi.SEED_RANDOM_BASE + 15)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert np.isclose(np.linalg.norm(a), 1.0)


def test_unit_rows_normalizes_and_raises_on_zero():
    m = np.array([[3.0, 4.0], [0.5, 0.0]])
    u = pi.unit_rows(m)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    with pytest.raises(ValueError, match="degenerate"):
        pi.unit_rows(np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_r2_score_multi_perfect_and_mean_predictor():
    y = RNG.standard_normal((30, 5))
    perfect = pi.r2_score_multi(y, y)
    assert np.isclose(perfect["r2"], 1.0) and np.isclose(perfect["mean_cosine"], 1.0)
    mean_pred = np.tile(y.mean(0), (30, 1))
    at_mean = pi.r2_score_multi(mean_pred, y)
    assert abs(at_mean["r2"]) < 1e-12  # SS_res == SS_tot at the mean predictor


# ---------------------------------------------------------------------------
# registry / CLI shape
# ---------------------------------------------------------------------------


def test_phases_registry_covers_plan_order_all_implemented():
    expected = (
        "stage_inputs",
        "fit_maps",
        "capture_directions",
        "norm_probe",
        "baseline_ceiling",
        "localize",
        "decisive",
        "patch",
        "build_pools",
        "margin",
        "judge_reduce",
        "figures",
        # ctxext-subspace-split amendment (plan v7 §4)
        "derive_split_directions",
        "ctxext_split_localize",
        "ctxext_split_decisive",
        "margin_split",
    )
    assert tuple(pi.PHASES) == expected  # plan §4 pipeline order + v7 amendment
    assert all(callable(fn) for fn in pi.PHASES.values())
    assert set(pi.UNIT3_PHASES) <= set(pi.PHASES)


def test_argparser_defaults_match_plan_grid():
    args = pi.build_argparser().parse_args([])
    assert args.behaviors == list(pi.BEHAVIORS)
    assert args.layers == list(range(28))
    assert args.out_root == "eval_results/issue_2254"
    assert args.num_shards == 1 and args.shard_id == 0


def test_apply_smoke_narrows_to_parity_layer_and_scratch_root():
    args = pi.build_argparser().parse_args(["--phase", "fit_maps", "--smoke"])
    try:
        pi._apply_smoke(args)
        assert args.layers == [pi.PILOT_LAYER]
        assert len(args.behaviors) == 1
        assert args.out_root == "/tmp/issue-2254-smoke"
        assert args.fig_dir == "/tmp/issue-2254-smoke/figures"
        assert args.q_localize == 2 and args.q_decisive == 2
        assert args.draws_localize == 2 and args.draws_decisive == 2
        assert pi._hf_prefix().endswith("/smoke")
    finally:
        # reset the module-global INSIDE finally so an assert failure cannot
        # leak the smoke prefix into later tests (review minor g2)
        pi._SMOKE_UPLOAD_SUBPREFIX = False


def _synthetic_sanitized_bundle(tmp_path, n=2, drop=()):
    """Tiny .pt mirroring the REALIZED pass-B schema at rev 037fcbb2 (the
    producer-sanitized upload: cx_last/cx_mean/v_x/layers/metadata/source,
    float32, NO prompts key — issue779_collect._sanitize_for_analysis_tensors)."""
    import torch

    blob = {
        "cx_last": torch.zeros((n, pi.N_LAYERS, pi.HIDDEN_DIM), dtype=torch.float32),
        "cx_mean": torch.zeros((n, pi.N_LAYERS, pi.HIDDEN_DIM), dtype=torch.float32),
        "v_x": torch.ones((n, pi.N_LAYERS, pi.HIDDEN_DIM), dtype=torch.float32),
        "layers": list(range(pi.N_LAYERS)),
        "metadata": {"pass": "b"},
        "source": "lmsys/lmsys-chat-1m",
    }
    for k in drop:
        blob.pop(k)
    p = tmp_path / "train_context_vectors.pt"
    torch.save(blob, p)
    return p


def test_load_pass_b_bundle_accepts_sanitized_schema(tmp_path, monkeypatch):
    """Regression (#2254 finisher round 1): the uploaded bundle carries NO
    'prompts' key by producer-sanitizer construction — the loader must accept
    the realized sanitized schema and return source + n_rows (the r1 loader
    required 'prompts' and crashed the fit_maps smoke on the real artifact)."""
    import huggingface_hub

    path = _synthetic_sanitized_bundle(tmp_path)

    def fake_download(repo_id=None, filename=None, repo_type=None, revision=None, **kw):
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(pi, "_BUNDLE_CACHE", None)
    out = pi._load_pass_b_bundle()
    assert out["n_rows"] == 2
    assert out["source"] == "lmsys/lmsys-chat-1m"
    assert "prompts" not in out
    monkeypatch.setattr(pi, "_BUNDLE_CACHE", None)


def test_load_pass_b_bundle_missing_required_key_raises(tmp_path, monkeypatch):
    """The realized-keys guard stays fail-loud on a genuinely wrong artifact."""
    import huggingface_hub

    path = _synthetic_sanitized_bundle(tmp_path, drop=("source",))

    def fake_download(repo_id=None, filename=None, repo_type=None, revision=None, **kw):
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(pi, "_BUNDLE_CACHE", None)
    with pytest.raises(RuntimeError, match=r"missing keys.*source"):
        pi._load_pass_b_bundle()
    monkeypatch.setattr(pi, "_BUNDLE_CACHE", None)


# ---------------------------------------------------------------------------
# e1 staged-sha gate at every consumer (round-1 blocker g2)
# ---------------------------------------------------------------------------


def test_assert_e1_staged_missing_and_wrong_sha(tmp_path, monkeypatch):
    import scripts.issue779_common as i779

    monkeypatch.setattr(i779, "_artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(pi, "_E1_STAGED_OK", set())
    with pytest.raises(RuntimeError, match="MISSING"):
        pi._assert_e1_staged("sycophancy")
    (tmp_path / "sycophancy.json").write_text("{}")  # present, WRONG bytes
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        pi._assert_e1_staged("sycophancy")


def test_assert_e1_staged_passes_and_caches_on_pinned_sha(tmp_path, monkeypatch):
    import scripts.issue779_common as i779

    payload = '{"eval_questions": []}'
    (tmp_path / "sycophancy.json").write_text(payload)
    sha = hashlib.sha256(payload.encode()).hexdigest()
    monkeypatch.setattr(i779, "_artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(pi, "E1_ASSET_SHA256", {**pi.E1_ASSET_SHA256, "sycophancy": sha})
    monkeypatch.setattr(pi, "_E1_STAGED_OK", set())
    pi._assert_e1_staged("sycophancy")
    assert "sycophancy" in pi._E1_STAGED_OK
    # evil is code-resident (paper-verbatim EVIL_ARTIFACTS): no file to pin
    pi._assert_e1_staged("evil")
    assert "evil" not in pi._E1_STAGED_OK


def test_every_e1_consumer_wrapper_gates_on_staged_assert(monkeypatch):
    """Blocker g2 pin: _eval_questions / _extraction_contexts /
    _positive_instructions hit the staged-sha gate BEFORE any load_e1_assets
    call — capture_directions, norm_probe, and every unit-3 pod-B call site
    inherit the gate through these wrappers, so the upstream loader's
    Sonnet-regeneration fallback is unreachable from EVERY driver phase."""

    def _boom(behavior):
        raise RuntimeError("staged-gate-called")

    monkeypatch.setattr(pi, "_assert_e1_staged", _boom)
    for fn in (pi._eval_questions, pi._extraction_contexts, pi._positive_instructions):
        with pytest.raises(RuntimeError, match="staged-gate-called"):
            fn("sycophancy")


# ---------------------------------------------------------------------------
# empty gen shard (num_shards > len(cells)) — clean no-op, no upload attempt
# ---------------------------------------------------------------------------


def test_run_gen_grid_empty_shard_skips_model_and_upload(tmp_path, monkeypatch):

    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(pi, "_require_cuda", lambda phase: None)
    monkeypatch.setattr(pi, "_assert_phase_headroom", lambda *a, **k: None)

    def _no_model():
        raise AssertionError("model must not load for an empty shard")

    monkeypatch.setattr(pi, "_load_model_and_tokenizer", _no_model)
    uploads: list = []
    monkeypatch.setattr(pi, "_upload_folder_to_hf", lambda *a, **k: uploads.append(a))
    args = pi.build_argparser().parse_args(
        [
            "--phase",
            "baseline_ceiling",
            "--out-root",
            str(tmp_path),
            "--num-shards",
            "5",
            "--shard-id",
            "4",
        ]
    )
    cells = [{"behavior": "evil", "kind": "alpha0"}, {"behavior": "evil", "kind": "ceiling"}]
    pi._run_gen_grid(
        args,
        "baseline_ceiling",
        cells,
        contexts_of=None,
        q_of=None,
        hookf_builder=None,
        n_draws_of=None,
        seeds_of=None,
    )
    assert uploads == []  # nothing generated => no confusing "no files under ..." raise
    sent = json.loads((tmp_path / "logs" / "issue-2254-baseline_ceiling-shard4.json").read_text())
    assert sent["empty_shard"] is True and sent["cells"] == 0


# ---------------------------------------------------------------------------
# rho-parity tolerance split (micro round 3: reference-side bank provenance —
# events v20 diagnosis; evil = code-resident binding canary, banked = 2e-2)
# ---------------------------------------------------------------------------


def _point_parity_ref(tmp_path, monkeypatch, ref: dict) -> None:
    """Point _rho_parity_assert's committed-#2220 reference at a synthetic JSON
    (filesystem boundary only; the real assert body runs)."""
    p = tmp_path / "rho_by_layer.json"
    p.write_text(json.dumps({"rho_median_last_context_token": ref}))

    def fake_ensure_git_input(rel_file: str, cone: str):
        return p

    monkeypatch.setattr(pi, "_ensure_git_input", fake_ensure_git_input)


def test_rho_parity_banked_within_banked_tol_passes_with_waiver(tmp_path, monkeypatch):
    # (a) banked behavior at rel 1.5e-2: above 5e-3, within 2e-2 => PASS,
    # recorded provenance_waived=True with the banked tolerance applied.
    _point_parity_ref(tmp_path, monkeypatch, {"sycophancy": {"L14": 100.0}})
    parity = pi._rho_parity_assert({"sycophancy": {"L14": 101.5}})
    (cell,) = parity["cells"]
    assert cell["banked"] is True
    assert cell["provenance_waived"] is True
    assert cell["tolerance_applied"] == pi.RHO_PARITY_RTOL_BANKED
    assert abs(cell["rel_dev"] - 1.5e-2) < 1e-9
    assert parity["n_provenance_waived"] == 1
    assert parity["rtol_banked"] == pi.RHO_PARITY_RTOL_BANKED


def test_rho_parity_evil_still_halts_above_binding_tol(tmp_path, monkeypatch):
    # (b) evil (code-resident canary) at rel 6e-3 > 5e-3 => still HALTs.
    _point_parity_ref(tmp_path, monkeypatch, {"evil": {"L14": 100.0}})
    with pytest.raises(RuntimeError, match="rho parity vs #2220 FAILED"):
        pi._rho_parity_assert({"evil": {"L14": 100.6}})


def test_rho_parity_banked_above_banked_tol_still_halts(tmp_path, monkeypatch):
    # (c) banked behavior at rel 3e-2 > 2e-2 => still HALTs.
    _point_parity_ref(tmp_path, monkeypatch, {"hallucination": {"L10": 100.0}})
    with pytest.raises(RuntimeError, match="banked=True"):
        pi._rho_parity_assert({"hallucination": {"L10": 103.0}})


def test_rho_parity_record_carries_tolerance_and_banked_flags(tmp_path, monkeypatch):
    # (d) every cell records tolerance_applied + banked; a banked cell WITHIN
    # 5e-3 is NOT waived; evil records the binding tolerance.
    _point_parity_ref(
        tmp_path,
        monkeypatch,
        {"evil": {"L14": 100.0}, "sycophancy": {"L14": 100.0}},
    )
    parity = pi._rho_parity_assert(
        {"evil": {"L14": 100.1}, "sycophancy": {"L14": 100.1}}  # both rel=1e-3
    )
    by_b = {c["behavior"]: c for c in parity["cells"]}
    assert by_b["evil"]["tolerance_applied"] == pi.RHO_PARITY_RTOL
    assert by_b["evil"]["banked"] is False
    assert by_b["evil"]["provenance_waived"] is False
    assert by_b["sycophancy"]["tolerance_applied"] == pi.RHO_PARITY_RTOL_BANKED
    assert by_b["sycophancy"]["banked"] is True
    assert by_b["sycophancy"]["provenance_waived"] is False
    assert parity["n_provenance_waived"] == 0


def test_rho_seam_assert_uses_split_tolerance_and_records_waived(monkeypatch, tmp_path):
    # Pod-B seam: same split — banked at rel 1.5e-2 passes (recorded loudly),
    # evil at rel 6e-3 raises. Real _rho_seam_assert body; GPU recompute +
    # staged-file load faked at the boundary with signature-mirroring defs.
    ref = {"evil": {"L14": 100.0}, "sycophancy": {"L14": 100.0}}

    def fake_load_rho(out_root):
        return {}, {"rho_median_last_context_token": ref}

    fresh = {"evil": {"L14": 100.4}, "sycophancy": {"L14": 101.5}}

    def fake_compute_rho(model, tok, behaviors, layers, phase="norm_probe"):
        return fresh, {}

    monkeypatch.setattr(pi, "_load_rho", fake_load_rho)
    monkeypatch.setattr(pi, "_compute_rho", fake_compute_rho)

    class _Args:
        out_root = str(tmp_path)

    seam = pi._rho_seam_assert(_Args(), model=None, tok=None)
    assert seam["rho_seam_cells"] == 2
    assert seam["rho_seam_rtol"] == pi.RHO_PARITY_RTOL
    assert seam["rho_seam_rtol_banked"] == pi.RHO_PARITY_RTOL_BANKED
    (waived,) = seam["provenance_waived_cells"]
    assert waived["behavior"] == "sycophancy" and waived["provenance_waived"] is True

    fresh["evil"]["L14"] = 100.6  # rel 6e-3 on the code-resident canary
    with pytest.raises(RuntimeError, match="rho seam mismatch"):
        pi._rho_seam_assert(_Args(), model=None, tok=None)


def test_run_judge_pilot_threads_waive_parse_fail_arms(tmp_path, monkeypatch):
    """Round 6 (task 2254 events v52): _run_judge_pilot forwards the CLI flag
    --waive-judge-parse-fail-arms into judge_pilot_gate(waive_parse_fail_arms=...)
    — the rule-26(b) explained-content-drop escape. Default [] = zero behavior
    change; truncation FAIL stays unwaivable inside judge_pilot itself. Real
    _run_judge_pilot body; the gate (Batch-API spend) is an autospec'd boundary
    fake, so a mis-named kwarg would raise TypeError here."""
    import types
    from unittest.mock import create_autospec

    import explore_persona_space.eval.judge_pilot as jp

    behavior, gen_phase = "evil", "decisive"
    monkeypatch.setattr(pi, "_eval_questions", lambda b: [f"q{i}" for i in range(20)])
    fake_gate = create_autospec(
        jp.judge_pilot_gate,
        return_value=types.SimpleNamespace(passed=True, verdict="PASS (fake)"),
    )
    monkeypatch.setattr(jp, "judge_pilot_gate", fake_gate)

    for i, (argv, expect) in enumerate(
        [([], []), (["--waive-judge-parse-fail-arms", "ans_steer"], ["ans_steer"])]
    ):
        out_root = tmp_path / f"out{i}"  # fresh root: the pass-sidecar skip never hides the call
        comp_root = out_root / gen_phase / "raw_completions"
        comp_root.mkdir(parents=True)
        rec = {
            "cell": {"kind": "alpha0", "behavior": behavior},
            "cell_id": "c0",
            "q_of_context": [0, 1],
            "seeds": {
                "42": {
                    "completions": [["a1", "a2"], ["b1", "b2"]],
                    "condition_passes": [True, True],
                }
            },
        }
        (comp_root / f"{behavior}__c0.json").write_text(json.dumps(rec))
        args = pi.build_argparser().parse_args(argv)
        assert list(args.waive_judge_parse_fail_arms) == expect
        pi._run_judge_pilot(args, out_root, gen_phase, behavior, "RUBRIC", 2)
        call = fake_gate.call_args
        # alpha0 cells pass every arm filter, so all three driver arm names appear
        assert set(call.args[0]) == {"ctx_steer", "ans_steer", "degen"}
        assert list(call.kwargs["waive_parse_fail_arms"]) == expect
