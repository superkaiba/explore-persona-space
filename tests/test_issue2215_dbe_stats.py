"""Committed fixture tests for the #2215 discrimination-battery-expansion
analysis driver (`scripts/issue2215_dbe_analysis.py`) — plan v6 §6.

Coverage (synthetic fixtures only — no network, no GPU, no benchmark text):
  (a) S3 equal-type-weight pooling: a 12-cluster + a 36-cluster type must
      pool to the MEAN OF THE TWO TYPE MEANS, never the flattened 1:3 mean.
  (b) H1 verdict formula over (m, D): every allowed pair maps to exactly one
      verdict; m=9 reproduces the >=7 / <=4 thresholds.
  (c) H2(a)/H2(b) interval-to-verdict mappings: exhaustive + disjoint on
      boundary CI configurations.
  (d) DiD common-type-set assert fires on a mismatched type-key fixture;
      the shared per-draw resample pairs the legs exactly.
  (e) Judge parse-contract round-trip for the refusal-check rubric
      (llm-judging rule 27: reason-then-verdict; drop-never-coerce).
  (f) pe-eligibility derivation pins (`issue2215_dbe_run._pe_eligibility`):
      identical prefixes => ineligible; one differing token => eligible;
      a mixed cell raises; realized != registered expectation raises.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2215_dbe_analysis as DA  # noqa: E402
import issue2215_dbe_run as DR  # noqa: E402

# ── (a) S3 equal-type-weight pooling ──────────────────────────────────


def _two_type_fixture() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One 12-cluster type + one 36-cluster type with distinct values (F=1):
    type means 5.5 and 117.5 -> equal-weight pool 61.5; flattened mean 89.5."""
    vals_a = np.arange(12, dtype=np.float64)
    vals_b = np.arange(36, dtype=np.float64) + 100.0
    values = np.concatenate([vals_a, vals_b])[:, None]
    type_rows = {
        "type_a": np.arange(12, dtype=np.int64),
        "type_b": np.arange(12, 48, dtype=np.int64),
    }
    return values, type_rows


def test_pooling_point_is_mean_of_type_means_not_flattened():
    values, type_rows = _two_type_fixture()
    keys = sorted(type_rows)
    point = DA.equal_type_weight_point(values, type_rows, keys)
    expected = (np.arange(12).mean() + (np.arange(36) + 100.0).mean()) / 2.0  # 61.5
    flattened = float(values.mean())  # 89.5 — the 1:3-weighted composition
    assert point.shape == (1,)
    assert point[0] == pytest.approx(expected)
    # the load-bearing half: the new pooling DIFFERS from the flattened mean
    # on the asymmetric fixture.
    assert abs(point[0] - flattened) > 10.0


def test_pooled_bootstrap_draws_are_equal_weight_exact_on_constants():
    # Constant-per-type values: every within-type resample mean equals the
    # constant, so EVERY pooled draw must equal the equal-weight average
    # exactly — and never the flattened 12:36 mean.
    values = np.concatenate([np.zeros(12), np.ones(36)])[:, None]
    type_rows = {
        "type_a": np.arange(12, dtype=np.int64),
        "type_b": np.arange(12, 48, dtype=np.int64),
    }
    keys = sorted(type_rows)
    resample = DA.make_type_resample(type_rows, n_boot=64, seed=123)
    per_type, pooled = DA.equal_type_weight_pooled(values, type_rows, resample, keys)
    assert per_type.shape == (2, 64, 1)
    assert pooled.shape == (64, 1)
    assert np.allclose(pooled, 0.5)  # equal-weight: (0 + 1) / 2
    assert not np.allclose(pooled, 36.0 / 48.0)  # flattened 1:3 mean = 0.75


def test_pooled_bootstrap_mean_tracks_equal_weight_estimand():
    values, type_rows = _two_type_fixture()
    keys = sorted(type_rows)
    resample = DA.make_type_resample(type_rows, n_boot=400, seed=7)
    _, pooled = DA.equal_type_weight_pooled(values, type_rows, resample, keys)
    draw_mean = float(pooled[:, 0].mean())
    assert abs(draw_mean - 61.5) < 1.0  # equal-weight estimand
    assert abs(draw_mean - 89.5) > 10.0  # not the flattened composition


def test_pooling_excludes_all_nan_type_mechanically():
    # The pe-degenerate case: a type whose column is all-NaN drops out of the
    # equal-weight average via nanmean, never contributing zeros.
    values = np.concatenate([np.full(12, np.nan), np.ones(36) * 3.0])[:, None]
    type_rows = {
        "type_a": np.arange(12, dtype=np.int64),
        "type_b": np.arange(12, 48, dtype=np.int64),
    }
    point = DA.equal_type_weight_point(values, type_rows, sorted(type_rows))
    assert point[0] == pytest.approx(3.0)


def test_make_type_resample_is_seed_deterministic_and_in_range():
    type_rows = {"t1": np.arange(5, dtype=np.int64), "t2": np.arange(5, 8, dtype=np.int64)}
    r1 = DA.make_type_resample(type_rows, n_boot=16, seed=2215)
    r2 = DA.make_type_resample(type_rows, n_boot=16, seed=2215)
    for t in type_rows:
        assert np.array_equal(r1[t], r2[t])
        assert r1[t].shape == (16, len(type_rows[t]))
        assert r1[t].min() >= 0 and r1[t].max() < len(type_rows[t])


# ── (b) H1 formula over (m, D) ────────────────────────────────────────


def test_h1_thresholds_m9_reproduce_registered_values():
    assert DA.h1_thresholds(9) == (7, 4)


def test_h1_verdict_partition_exhaustive_and_disjoint():
    vocab = {"support", "falsified", "inconclusive"}
    for m in range(1, 13):
        hi, lo = DA.h1_thresholds(m)
        assert hi > lo, (m, hi, lo)  # disjoint regions by construction
        for d in range(0, m + 1):
            v = DA.h1_verdict(m, d)
            assert v in vocab
            # exactly one region claims each (m, d)
            in_support = d >= hi
            in_falsified = d <= lo
            assert not (in_support and in_falsified), (m, d)
            if in_support:
                assert v == "support", (m, d)
            elif in_falsified:
                assert v == "falsified", (m, d)
            else:
                assert v == "inconclusive", (m, d)
        assert DA.h1_verdict(m, m) == "support"  # ceil(7m/9) <= m always


def test_h1_verdict_m9_boundary_rows():
    assert [DA.h1_verdict(9, d) for d in (9, 8, 7)] == ["support"] * 3
    assert [DA.h1_verdict(9, d) for d in (6, 5)] == ["inconclusive"] * 2
    assert [DA.h1_verdict(9, d) for d in (4, 3, 0)] == ["falsified"] * 3


def test_h1_verdict_rejects_out_of_range_d():
    with pytest.raises(AssertionError):
        DA.h1_verdict(5, 6)


# ── (c) H2 interval-to-verdict mappings ───────────────────────────────


def test_h2a_boundary_configurations():
    thr = DA.H2A_DELTA_MAX  # +0.03
    assert DA.h2a_verdict([-0.01, thr]) == "support"  # upper == thr counts
    assert DA.h2a_verdict([-0.05, -0.01]) == "support"
    assert DA.h2a_verdict([thr, 0.05]) == "inconclusive"  # lower == thr: NOT >
    assert DA.h2a_verdict([thr + 1e-9, 0.05]) == "falsified"
    assert DA.h2a_verdict([0.0, 0.05]) == "inconclusive"  # straddle
    assert DA.h2a_verdict([float("nan"), float("nan")]) == "inconclusive"
    assert DA.h2a_verdict([0.0, float("inf")]) == "inconclusive"


def test_h2b_boundary_configurations():
    assert DA.h2b_verdict([1e-9, 0.1]) == "support"
    assert DA.h2b_verdict([0.0, 0.1]) == "inconclusive"  # lower == 0: NOT >
    assert DA.h2b_verdict([-0.1, -1e-9]) == "falsified"
    assert DA.h2b_verdict([-0.1, 0.0]) == "inconclusive"  # upper == 0: NOT <
    assert DA.h2b_verdict([-0.1, 0.1]) == "inconclusive"
    assert DA.h2b_verdict([float("nan"), 0.1]) == "inconclusive"


def test_h2_verdicts_exhaustive_and_disjoint_on_grid():
    vocab = {"support", "falsified", "inconclusive"}
    grid = np.linspace(-0.1, 0.1, 21)
    for lo in grid:
        for hi in grid:
            if lo > hi:
                continue
            va = DA.h2a_verdict([float(lo), float(hi)])
            vb = DA.h2b_verdict([float(lo), float(hi)])
            assert va in vocab and vb in vocab
            # disjointness: support and falsified predicates cannot co-fire
            # on a well-ordered interval.
            assert not (hi <= DA.H2A_DELTA_MAX and lo > DA.H2A_DELTA_MAX)
            assert not (lo > 0 and hi < 0)


def test_h3_at_median_counts_toward_support_and_na_when_dropped():
    acc = {"refusal_request": 0.8, "type_a": 0.8, "type_b": 0.9}
    ret = {"refusal_request": 0.2, "type_a": 0.2, "type_b": 0.1}
    out = DA.h3_verdict(acc, ret)
    assert out["verdict"] == "support"  # at-median on BOTH axes -> support
    assert out["m"] == 3
    # dissociation broken: refusal retrieves ABOVE the median -> falsified
    ret_hi = {"refusal_request": 0.9, "type_a": 0.2, "type_b": 0.1}
    assert DA.h3_verdict(acc, ret_hi)["verdict"] == "falsified"
    # refusal dropped at gate 1 -> N/A, never a fabricated verdict
    out_na = DA.h3_verdict({"type_a": 0.5}, {"type_a": 0.5})
    assert out_na["verdict"] == "n/a"


# ── (d) DiD common-type-set assert + shared-resample pairing ──────────

_DID_CFGS = [
    "1738pe|L19|tail|cosine",
    "idbias_pe|L19|tail|cosine",
    "1738ce|L19|tail|cosine",
    "idbias_ce|L19|tail|cosine",
]


def _did_fixture(*, degen_pe_finite: bool) -> tuple[np.ndarray, dict, dict, dict]:
    """acc_cl over 2 types (4 + 3 clusters) x 4 configs. Constant-valued legs
    so every bootstrap draw's leg mean is exact: pe gain 0.30 on t_elig
    (NaN on t_degen unless degen_pe_finite), ce gain 0.10 everywhere."""
    n = 7
    acc = np.zeros((n, 4), dtype=np.float64)
    cfg_index = {c: k for k, c in enumerate(_DID_CFGS)}
    type_rows = {
        "t_degen": np.arange(4, 7, dtype=np.int64),
        "t_elig": np.arange(0, 4, dtype=np.int64),
    }
    acc[:, cfg_index["idbias_pe|L19|tail|cosine"]] = 0.50
    acc[:, cfg_index["1738pe|L19|tail|cosine"]] = 0.80  # pe gain +0.30
    acc[:, cfg_index["idbias_ce|L19|tail|cosine"]] = 0.55
    acc[:, cfg_index["1738ce|L19|tail|cosine"]] = 0.65  # ce gain +0.10
    if not degen_pe_finite:
        acc[type_rows["t_degen"], cfg_index["1738pe|L19|tail|cosine"]] = np.nan
        acc[type_rows["t_degen"], cfg_index["idbias_pe|L19|tail|cosine"]] = np.nan
    resample = DA.make_type_resample(type_rows, n_boot=32, seed=21620)
    return acc, type_rows, cfg_index, resample


def _run_did(acc, type_rows, cfg_index, resample, eligible):
    return DA.compute_did(
        acc,
        type_rows,
        cfg_index,
        resample,
        arm_pe="1738pe",
        arm_ce="1738ce",
        layer=19,
        pool="tail",
        metric="cosine",
        eligible_types=eligible,
        label="fixture",
    )


def test_did_paired_legs_exact_on_constant_fixture():
    acc, type_rows, cfg_index, resample = _did_fixture(degen_pe_finite=False)
    out = _run_did(acc, type_rows, cfg_index, resample, ["t_elig"])
    assert out["leg_pe_gain"] == pytest.approx(0.30)
    assert out["leg_ce_gain"] == pytest.approx(0.10)
    assert out["did"] == pytest.approx(0.20)
    draws = out["npz"]["dv3dbe|fixture|did_draws"]
    # constant cluster values => every paired draw is exactly pe minus ce; any
    # leg mis-pairing (different resample / key order) would break this.
    assert np.allclose(draws, 0.20, atol=1e-6)
    assert out["did_ci95"][0] == pytest.approx(0.20, abs=1e-6)
    assert out["did_ci95"][1] == pytest.approx(0.20, abs=1e-6)
    assert list(out["npz"]["dv3dbe|fixture|type_keys"]) == ["t_elig"]
    # per-draw per-type leg matrices persist with the ordered key axis
    assert out["npz"]["dv3dbe|fixture|leg_pe_pertype_draws"].shape == (32, 1)


def test_did_common_type_set_assert_fires_on_extra_realized_type():
    # t_degen carries FINITE pe data but is NOT registered eligible -> the
    # pe-leg realized type set exceeds the registered set -> hard assert.
    acc, type_rows, cfg_index, resample = _did_fixture(degen_pe_finite=True)
    with pytest.raises(AssertionError, match="common-type-set"):
        _run_did(acc, type_rows, cfg_index, resample, ["t_elig"])


def test_did_common_type_set_assert_fires_on_missing_realized_type():
    # registered set claims t_degen is eligible while its pe leg is all-NaN
    # -> realized set is smaller than registered -> hard assert.
    acc, type_rows, cfg_index, resample = _did_fixture(degen_pe_finite=False)
    with pytest.raises(AssertionError, match="common-type-set"):
        _run_did(acc, type_rows, cfg_index, resample, ["t_degen", "t_elig"])


def test_did_rejects_empty_eligible_set():
    acc, type_rows, cfg_index, resample = _did_fixture(degen_pe_finite=False)
    with pytest.raises(AssertionError, match="empty M2-eligible"):
        _run_did(acc, type_rows, cfg_index, resample, [])


# ── (e) refusal-judge parse contract (llm-judging rule 27) ────────────


def test_refusal_judge_parse_plain_json():
    reply = '{"reasoning": "declines and redirects to safer framing", "verdict": "REFUSAL"}'
    out = DA._parse_refusal_judge(reply)
    assert out is not None and out["verdict"] == "REFUSAL"
    assert "declines" in out["reasoning"]


def test_refusal_judge_parse_reason_prose_then_json():
    reply = (
        "The assistant engages with the request and gives a direct answer.\n"
        '{"reasoning": "substantive on-topic answer", "verdict": "compliance"}'
    )
    out = DA._parse_refusal_judge(reply)
    assert out is not None and out["verdict"] == "COMPLIANCE"


def test_refusal_judge_parse_fenced_json():
    reply = '```json\n{"reasoning": "hedged but answers", "verdict": "COMPLIANCE"}\n```'
    out = DA._parse_refusal_judge(reply)
    assert out is not None and out["verdict"] == "COMPLIANCE"


def test_refusal_judge_malformed_returns_none_never_coerced():
    assert DA._parse_refusal_judge("") is None
    assert DA._parse_refusal_judge("I cannot evaluate this material.") is None
    assert DA._parse_refusal_judge('{"verdict": "PARTIAL", "reasoning": "x"}') is None
    assert DA._parse_refusal_judge('{"reasoning": "no verdict key"}') is None
    assert DA._parse_refusal_judge("85") is None  # bare scalar is not a dict


def test_refusal_rubric_shape_matches_parser_contract():
    # The rubric instructs reason-then-verdict JSON with EXACTLY the two keys
    # the parser reads; a rubric/parse drift breaks this round-trip.
    assert '"reasoning"' in DA.REFUSAL_RUBRIC and '"verdict"' in DA.REFUSAL_RUBRIC
    assert "REFUSAL" in DA.REFUSAL_RUBRIC and "COMPLIANCE" in DA.REFUSAL_RUBRIC
    assert DA.JUDGE_MODEL == "claude-sonnet-4-5-20250929"
    assert DA.JUDGE_MAX_TOKENS >= 1024


# ── (f) pe-eligibility derivation pins (plan §6; M2) ──────────────────


def _pe_bank(pairs, expected):
    return {"pairs": pairs, "expected_pe_eligibility": expected}


def test_pe_identical_prefixes_are_ineligible():
    bank = _pe_bank(
        [{"pair_id": "p1", "a": "c1", "b": "c2", "cell": "cellX"}],
        {"cellX": False},
    )
    ctx_ids = {"c1": [1, 2, 3, 9], "c2": [1, 2, 3, 7]}
    prefix_ends = {"c1": 3, "c2": 3}
    per_pair, per_cell = DR._pe_eligibility(bank, ctx_ids, prefix_ends)
    assert per_pair == {"p1": False}
    assert per_cell == {"cellX": False}


def test_pe_one_differing_token_is_eligible():
    bank = _pe_bank(
        [{"pair_id": "p1", "a": "c1", "b": "c2", "cell": "cellY"}],
        {"cellY": True},
    )
    ctx_ids = {"c1": [1, 2, 3, 9], "c2": [1, 2, 4, 9]}
    prefix_ends = {"c1": 3, "c2": 3}
    per_pair, per_cell = DR._pe_eligibility(bank, ctx_ids, prefix_ends)
    assert per_pair == {"p1": True}
    assert per_cell == {"cellY": True}


def test_pe_mixed_cell_raises():
    bank = _pe_bank(
        [
            {"pair_id": "p1", "a": "c1", "b": "c2", "cell": "cellZ"},
            {"pair_id": "p2", "a": "c3", "b": "c4", "cell": "cellZ"},
        ],
        {"cellZ": True},
    )
    ctx_ids = {
        "c1": [1, 2, 3],
        "c2": [1, 2, 3],  # identical prefixes -> ineligible pair
        "c3": [1, 2, 3],
        "c4": [1, 9, 3],  # differing prefix -> eligible pair
    }
    prefix_ends = {c: 2 for c in ctx_ids}
    with pytest.raises(AssertionError, match="mixed"):
        DR._pe_eligibility(bank, ctx_ids, prefix_ends)


def test_pe_realized_vs_registered_mismatch_raises():
    bank = _pe_bank(
        [{"pair_id": "p1", "a": "c1", "b": "c2", "cell": "cellW"}],
        {"cellW": True},  # registered eligible, realized identical prefixes
    )
    ctx_ids = {"c1": [1, 2, 3], "c2": [1, 2, 7]}
    prefix_ends = {"c1": 2, "c2": 2}
    with pytest.raises(AssertionError, match="structure bug"):
        DR._pe_eligibility(bank, ctx_ids, prefix_ends)


# ── H1 thresholds match the plan formulas symbolically ────────────────


def test_h1_thresholds_are_ceil_floor_formulas():
    for m in range(1, 40):
        hi, lo = DA.h1_thresholds(m)
        assert hi == math.ceil(7 * m / 9)
        assert lo == math.floor(4 * m / 9)


# ── (g) round-2 review pins: seed default, regime fingerprints, guards ─


def _mk_cfg(tmp_path: Path, *, smoke: bool = False, tiny: bool = False, force: bool = False):
    """Tiny DbeConfig over tmp roots with real values/bank bytes on disk (the
    fingerprint helpers hash BIT-EXACT files read from disk, #1336)."""
    values = tmp_path / "values.json"
    if not values.exists():
        values.write_text('{"values": ["v1", "v2"]}')
    cfg = DR.DbeConfig(
        phase="",
        out_root=tmp_path / "out",
        log_dir=tmp_path / "logs",
        values_path=values,
        smoke=smoke,
        tiny=tiny,
        cells=("cellA",),
        draws=2,
        null_b=100,
        gen_batch=2,
        capture_batch=2,
        max_new_tokens=64,
        seed_base=42,
        force=force,
        layers=[0, 1],
    )
    cfg.vc_dir.mkdir(parents=True, exist_ok=True)
    bank = cfg.vc_dir / "bank_dbe.json"
    if not bank.exists():
        bank.write_text('{"pairs": []}')
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_cli_default_seed_base_is_plan_pinned():
    """dbe-seed-base-plan-mismatch: plan v6 §10 pins generation seed_base=42
    (recipe fidelity with the parent's banked rollouts; issue2162_run)."""
    assert DR.SEED_BASE == 42
    args = DR.parse_args(["--phase", "B1"])
    assert args.seed_base == 42
    assert DR.build_config(args).seed_base == 42


def test_cli_force_flag_defaults_off_and_threads():
    args = DR.parse_args(["--phase", "D"])
    assert args.force is False
    assert DR.build_config(args).force is False
    args_f = DR.parse_args(["--phase", "D", "--force"])
    assert DR.build_config(args_f).force is True


def test_pilot_report_bare_existence_never_satisfies_production_resume(tmp_path):
    """smoke-pilot-cross-regime-reuse: a demoted smoke pilot report must not
    suppress the production throughput gate; only a regime-matched,
    non-demoted report counts."""
    smoke = _mk_cfg(tmp_path, smoke=True)
    prod = _mk_cfg(tmp_path)
    report = smoke.manifest_dir / "pilot_gate_report.json"
    assert smoke.manifest_dir == prod.manifest_dir  # shared out_root — the trap
    # (1) smoke-regime report present -> production resume NOT satisfied
    report.write_text(
        json.dumps(
            {
                "regime_fp": DR._pilot_regime_fp(smoke),
                "demoted_to_informational": True,
                "verdict": "PASS",
            }
        )
    )
    assert DR._pilot_report_ok(smoke) is True  # smoke resume may reuse it
    assert DR._pilot_report_ok(prod) is False  # production must NOT
    # (2) production-regime fp but demoted -> still refused in production
    report.write_text(
        json.dumps({"regime_fp": DR._pilot_regime_fp(prod), "demoted_to_informational": True})
    )
    assert DR._pilot_report_ok(prod) is False
    # (3) production-regime, non-demoted -> satisfied
    report.write_text(
        json.dumps({"regime_fp": DR._pilot_regime_fp(prod), "demoted_to_informational": False})
    )
    assert DR._pilot_report_ok(prod) is True
    # (4) missing / unparseable -> never satisfied
    report.unlink()
    assert DR._pilot_report_ok(prod) is False
    report.write_text("not json")
    assert DR._pilot_report_ok(prod) is False


def test_b2_regime_fp_tracks_data_inputs(tmp_path):
    """dbe-resume-fingerprint-inputs: the B2/cell fingerprints change when the
    frozen values file, the realized bank, the layer list, or smoke/production
    state change — never only the generation params."""
    cfg = _mk_cfg(tmp_path)
    fp0 = DR._b2_regime_fp(cfg)
    cell0 = DR._cell_regime_fp(cfg, "cellA")
    # bank content change
    (cfg.vc_dir / "bank_dbe.json").write_text('{"pairs": [{"pair_id": "p1"}]}')
    fp_bank = DR._b2_regime_fp(cfg)
    cell_bank = DR._cell_regime_fp(cfg, "cellA")
    assert fp_bank != fp0 and cell_bank != cell0
    # values content change
    Path(cfg.values_path).write_text('{"values": ["CHANGED"]}')
    assert DR._b2_regime_fp(cfg) != fp_bank
    # layer-list change
    cfg.layers = [0, 1, 2]
    fp_layers = DR._b2_regime_fp(cfg)
    assert fp_layers != DR._b2_regime_fp(_mk_cfg(tmp_path))
    # smoke vs production diverge
    assert DR._b2_regime_fp(_mk_cfg(tmp_path, smoke=True)) != DR._b2_regime_fp(_mk_cfg(tmp_path))
    # pilot fp additionally tracks gen_batch (throughput-affecting shape)
    cfg2 = _mk_cfg(tmp_path)
    p0 = DR._pilot_regime_fp(cfg2)
    cfg2.gen_batch = 32
    assert DR._pilot_regime_fp(cfg2) != p0


def test_finalize_datagen_manifest_roundtrips_and_c_assert_agrees(tmp_path, monkeypatch):
    """dbe-manifest-realized-pe-map: B1' finalization writes the realized
    per-pair map + hash; the C-entry assert accepts exact agreement and raises
    on any coverage / flag / hash drift."""
    cfg = _mk_cfg(tmp_path, smoke=True)  # smoke tolerates a missing committed source
    monkeypatch.setattr(DR, "DATAGEN_MANIFEST_SRC", tmp_path / "absent_manifest.json")
    bank = {
        "pairs": [
            {"pair_id": "p1", "cell": "cellA", "pe_realized_eligible": True},
            {"pair_id": "p2", "cell": "cellA", "pe_realized_eligible": False},
        ],
        "realized_pe_eligibility": {"cellA": False},
    }
    out = DR._finalize_datagen_manifest(cfg, bank)
    final = json.loads(out.read_text())
    assert final["source_manifest"]["present"] is False
    assert final["realized_pe_eligibility"]["per_pair"] == {"p1": True, "p2": False}
    DR._assert_realized_pe_manifest(final, bank)  # exact agreement passes
    # missing pair coverage raises
    short = {**bank, "pairs": bank["pairs"][:1]}
    with pytest.raises(AssertionError, match="extra"):
        DR._assert_realized_pe_manifest(final, short)
    # flag disagreement raises
    flipped = copy.deepcopy(bank)
    flipped["pairs"][0]["pe_realized_eligible"] = False
    with pytest.raises(AssertionError, match="disagrees"):
        DR._assert_realized_pe_manifest(final, flipped)
    # hash drift raises
    tampered = copy.deepcopy(final)
    tampered["realized_pe_eligibility"]["sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="hash drift"):
        DR._assert_realized_pe_manifest(tampered, bank)


def test_production_finalize_requires_committed_datagen_manifest(tmp_path, monkeypatch):
    cfg = _mk_cfg(tmp_path)  # production
    monkeypatch.setattr(DR, "DATAGEN_MANIFEST_SRC", tmp_path / "absent_manifest.json")
    bank = {"pairs": [], "realized_pe_eligibility": {}}
    with pytest.raises(RuntimeError, match="committed datagen manifest"):
        DR._finalize_datagen_manifest(cfg, bank)


def test_assert_parent_vc_coverage_missing_raises_extra_tolerated():
    """dbe-parent-vc-cache-coverage: bank ids ⊆ cached per_context keys is
    asserted BEFORE indexing; a superset is tolerated (bank-scoped indexing)."""
    DA.assert_parent_vc_coverage(["a", "b"], {"a": {}, "b": {}, "c": {}})
    with pytest.raises(AssertionError, match="missing"):
        DA.assert_parent_vc_coverage(["a", "b", "z"], {"a": {}, "b": {}})


def test_sentinel_present_tolerates_poller_drain_rename(tmp_path):
    p = tmp_path / "issue-2215-dbe-results.json"
    assert DR._sentinel_present(p) is False
    p.with_name(p.name + ".processed").write_text("{}")
    assert DR._sentinel_present(p) is True


def test_figures_nan_if_none_preserves_legitimate_zero():
    import issue2215_dbe_figures as DF

    assert DF._nan_if_none(0.0) == 0.0
    assert DF._nan_if_none(0.75) == 0.75
    assert math.isnan(DF._nan_if_none(None))


def test_land_results_git_production_leg_end_to_end(tmp_path, monkeypatch):
    """dbe-primary-artifact-egress: the phase-D git landing leg runs its REAL
    body (add -> staged-index verification incl. the gitignore force-add branch
    -> pathspec commit -> push -> rev-list push-verify -> per-FILE remote
    presence) against a scratch repo + bare remote; smoke/tiny skips."""
    import subprocess as sp

    def git(repo, *args):
        sp.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    remote = tmp_path / "remote.git"
    sp.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    repo = tmp_path / "repo"
    sp.run(["git", "init", "-b", "issue-2215", str(repo)], check=True, capture_output=True)
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / ".gitignore").write_text("*.png\n")  # exercises the #958 force-add branch
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-m", "init")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "origin", "issue-2215")
    monkeypatch.setattr(DR, "_REPO_ROOT", repo)
    cfg = _mk_cfg(tmp_path)  # production regime
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    (cfg.eval_dir / "dbe_result.json").write_text('{"ok": true}')
    (cfg.figures_dir / "dbe_hero.png").write_bytes(b"\x89PNG-fake")
    rec = DR._land_results_git(cfg)
    assert rec["mode"] == "committed" and rec["n_files"] == 2
    remote_ls = sp.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "origin/issue-2215", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "eval_results/issue_2215/discrimination-battery-expansion/dbe_result.json" in remote_ls
    assert "figures/issue_2215/dbe_hero.png" in remote_ls  # gitignored -> force-added
    # idempotent re-entry: nothing new to commit, still verified
    rec2 = DR._land_results_git(cfg)
    assert rec2["mode"] == "committed"
    # smoke cfg skips (out_root twins are not repo paths)
    assert DR._land_results_git(_mk_cfg(tmp_path, smoke=True)) == {"mode": "skipped-smoke"}


def test_c_and_d_regime_fp_track_null_b(tmp_path):
    cfg = _mk_cfg(tmp_path)
    fp0 = DR._c_regime_fp(cfg)
    cfg.null_b = 10_000
    assert DR._c_regime_fp(cfg) != fp0
