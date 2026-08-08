"""Pins for the #1689 derived-vs-free round (plan v8).

Covers:
- rotation-null svec reduction == the verbatim two-sided-rotation cosine
  (per-draw EXACT von Neumann identity — the --phase nulls bank math);
- derived-map algebra on an exact synthetic linear world (shared-readout /
  readout-changed verdicts land where constructed);
- the FOUR-class verdict lattice (pure function, all classes + invalid);
- fit_ladder generalized addressing == run_all_pairs on a within-model pair
  (the consistency-checker WARN discharge: extension is addressing-only);
- pairs-file schema parsing (legacy / nested / mixed / errors);
- data-dependent gates: n_common < 3, all-folds-degenerate, low-common flag,
  merge fail-loud on never-attempted units, Gate-1 parity refusal (rc 7);
- item-7a class-ladder floor == identity_bias_predict (translation class);
- fix round 3 (model-qualified unit keys): within-model keys carry the model
  (two models sharing an out-root no longer collide — the 241+11-of-504
  defect), cross-model keys byte-unchanged, one-shot migrate-keys rename,
  merge double-count/expect-units/identity fail-louds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.issue1689_context_map_structure as cms
import scripts.issue1689_derived_vs_free as dvf
import scripts.issue1689_fit_ladder as fl
from explore_persona_space.analysis.mapping_baselines import identity_bias_predict


def _haar_np(d: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def test_rotation_null_svec_reduction_exact():
    """cos(vec A, vec(Q1^T B Q2)) == s_A^T (P o R^T) s_B / (|A|_F |B|_F) with
    P = U_A^T Q1^T U_B, R = V_B^T Q2 V_A — EXACT per draw, any orthogonal Q1/Q2.

    Haar invariance then makes drawing (P, R) directly Haar (the parent
    9a-ter bank convention, e = p * r.T) distribution-identical to the
    verbatim issue1345_operator_comparison.raw_cosine_with_rotation_null loop.
    """
    rng = np.random.default_rng(0)
    d = 16
    A = rng.standard_normal((d, d))
    B = rng.standard_normal((d, d))
    Ua, sa, Vha = np.linalg.svd(A)
    Ub, sb, Vhb = np.linalg.svd(B)
    for k in range(5):
        q1 = _haar_np(d, rng)
        q2 = _haar_np(d, rng)
        verbatim = float(
            (A.reshape(-1) @ (q1.T @ B @ q2).reshape(-1)) / (np.linalg.norm(A) * np.linalg.norm(B))
        )
        P = Ua.T @ q1.T @ Ub
        R = Vhb @ q2 @ Vha.T
        reduced = float(sa @ (P * R.T) @ sb / (np.linalg.norm(sa) * np.linalg.norm(sb)))
        assert abs(verbatim - reduced) < 1e-10, (k, verbatim, reduced)


def test_verdict_class_lattice():
    v = dvf.verdict_class
    # Class 0: free map uninformative (excluded from verdict counts downstream)
    assert v(0.10, 0.20, 5.0, 5.0) == "free_map_uninformative"
    # informative classes
    assert v(0.50, 0.10, 0.01, 0.02) == "shared_readout_supported"
    assert v(0.50, 0.10, -0.01, 0.02) == "readout_changed"
    assert v(0.50, 0.10, -0.01, -0.02) == "transfer_map_insufficient"
    assert v(float("nan"), 0.1, 0.0, 0.0) == "invalid"


def _synthetic_cell(rng, n, d, x, y, conv_prefix="c"):
    return {
        "X_prefix": x.copy(),
        "X_context": x,
        "Y": y,
        "conv_ids": np.array([f"{conv_prefix}{i}" for i in range(n)]),
    }


def _dvf_args(**over):
    base = dict(
        layer=19,
        lambda_grid="ladder13",
        seed=42,
        device="cpu",
        row_limit=None,
        dim_limit=None,
        rotation_draws=2,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_run_unit_shared_readout_on_exact_linear_world():
    """Exact y = xW + b with x_T = x_S M + a: the derived map must match the
    free map (g1 >= 0, shared_readout_supported) and the low-common flag fires
    (n < 500). Also probes the derived-bias algebra end to end."""
    rng = np.random.default_rng(1)
    n, d = 300, 24
    W = rng.standard_normal((d, d)) / np.sqrt(d)
    b = rng.standard_normal(d) * 0.1
    M = np.eye(d) + 0.3 * rng.standard_normal((d, d)) / np.sqrt(d)
    a = rng.standard_normal(d) * 0.1
    x_s = rng.standard_normal((n, d))
    x_t = x_s @ M + a
    source = _synthetic_cell(rng, n, d, x_s, x_s @ W + b)
    target = _synthetic_cell(rng, n, d, x_t, x_t @ W + b)
    unit, bundle = dvf.run_unit(
        source, target, (("mA", "s"), ("mA", "t")), "context", _dvf_args(), fl.LAMBDAS
    )
    assert "error" not in unit, unit
    assert unit["flag_low_common"] is True
    assert unit["r2_b_free"] > 0.99
    assert unit["r2_b_derived_max"] > 0.9 * unit["r2_b_free"]
    assert unit["verdict"] == "shared_readout_supported"
    assert "svec_free" in bundle and "m_minus_i_svals" in bundle


def test_run_unit_readout_changed_on_two_readout_world():
    """y_T uses a DIFFERENT readout W2: derived (shared W_S) must lose to
    derived2 (per-condition readouts) -> readout_changed."""
    rng = np.random.default_rng(2)
    n, d = 300, 24
    W = rng.standard_normal((d, d)) / np.sqrt(d)
    W2 = rng.standard_normal((d, d)) / np.sqrt(d)
    M = np.eye(d) + 0.3 * rng.standard_normal((d, d)) / np.sqrt(d)
    x_s = rng.standard_normal((n, d))
    x_t = x_s @ M + 0.1
    source = _synthetic_cell(rng, n, d, x_s, x_s @ W)
    target = _synthetic_cell(rng, n, d, x_t, x_t @ W2)
    unit, _ = dvf.run_unit(
        source, target, (("mA", "s"), ("mA", "t")), "context", _dvf_args(), fl.LAMBDAS
    )
    assert "error" not in unit, unit
    assert unit["verdict"] == "readout_changed", (
        unit["verdict"],
        unit["g1"],
        unit["g2"],
    )


def test_run_unit_data_gates():
    """n_common < 3 -> structural error; single-conv -> all folds degenerate."""
    rng = np.random.default_rng(3)
    n, d = 20, 8
    x = rng.standard_normal((n, d))
    src = _synthetic_cell(rng, n, d, x, x, conv_prefix="a")
    tgt = _synthetic_cell(rng, n, d, x, x, conv_prefix="b")  # disjoint conv ids
    unit, _ = dvf.run_unit(src, tgt, (("mA", "s"), ("mA", "t")), "context", _dvf_args(), fl.LAMBDAS)
    assert unit["error"] == "insufficient shared conv_ids"
    assert unit["retryable"] is False
    # 3 shared convs, one row each: every 5-fold split leaves tr < 3 or te < 1
    x3 = rng.standard_normal((3, d))
    src2 = _synthetic_cell(rng, 3, d, x3, x3)
    tgt2 = _synthetic_cell(rng, 3, d, x3.copy(), x3.copy())
    unit2, _ = dvf.run_unit(
        src2, tgt2, (("mA", "s"), ("mA", "t")), "context", _dvf_args(), fl.LAMBDAS
    )
    assert unit2["error"] == "all folds degenerate"


def _write_synth_store(root: Path, model: str, conds: list[str], *, n=60, d=8, seed=7):
    rng = np.random.default_rng(seed)
    conv = np.array([f"cv{i}" for i in range(n)])
    for j, cond in enumerate(conds):
        x = rng.standard_normal((n, d))
        cell = root / model / cond
        cell.mkdir(parents=True)
        torch.save(
            {
                "X_prefix": x + j,
                "X_context": x,
                "Y": x @ rng.standard_normal((d, d)) + 0.1 * rng.standard_normal((n, d)),
                "conv_ids": conv,
            },
            cell / "L19.pt",
        )


def test_generalized_runner_matches_run_all_pairs_within_model(tmp_path):
    """WARN discharge: the generalized (cross-model-capable) runner routes a
    within-model pair through the SAME _run_ladder_pair — results identical to
    run_all_pairs on the same pair (fit math byte-unchanged)."""
    store = tmp_path / "store"
    pair = ("assistant_chat", "assistant_naturalistic")  # a real enumerate_pair_set member
    _write_synth_store(store, "mA", list(pair))
    legacy = fl.run_all_pairs(
        store,
        model_slug="mA",
        layer=19,
        n_bootstrap_draws=2,
        n_null_draws=2,
        engine="numpy",
        checkpoint_dir=tmp_path / "ck1",
        pairs_subset=[pair],
    )
    general = fl.run_pairs_generalized(
        store,
        [(("mA", pair[0]), ("mA", pair[1]))],
        layer=19,
        n_bootstrap_draws=2,
        n_null_draws=2,
        engine="numpy",
        checkpoint_dir=tmp_path / "ck2",
    )
    key = f"{pair[0]}__{pair[1]}"
    assert json.dumps(legacy["pairs"][key], sort_keys=True) == json.dumps(
        general["pairs"][key], sort_keys=True
    )


def test_parse_pair_specs_schemas():
    legacy = [["a", "b"], ["c", "d"]]
    specs = fl.parse_pair_specs(legacy, default_model="m")
    assert specs == [(("m", "a"), ("m", "b")), (("m", "c"), ("m", "d"))]
    nested = [[["m1", "a"], ["m2", "a"]]]
    assert fl.parse_pair_specs(nested) == [(("m1", "a"), ("m2", "a"))]
    mixed = [["a", "b"], [["m1", "a"], ["m2", "a"]]]
    assert len(fl.parse_pair_specs(mixed, default_model="m")) == 2
    with pytest.raises(ValueError, match="default model"):
        fl.parse_pair_specs(legacy)
    with pytest.raises(ValueError, match="2 elements"):
        fl.parse_pair_specs([["a", "b", "c"]])
    assert fl.pairs_file_is_generalized(nested) is True
    assert fl.pairs_file_is_generalized(legacy) is False
    assert fl.pair_spec_key((("m", "a"), ("m", "b"))) == "a__b"
    assert fl.pair_spec_key((("m1", "a"), ("m2", "a"))) == "m1@a__m2@a"


def test_crossmodel_pair_specs_shape():
    specs = fl.crossmodel_pair_specs("mBase", "mInstr")
    assert len(specs) == 42  # 21 conditions x 2 directions
    assert all(sm != tm for ((sm, _), (tm, _)) in specs)
    conds = {sc for ((_, sc), _) in specs}
    assert len(conds) == 21


def test_merge_fails_loud_on_missing_units(tmp_path):
    args = _dvf_args(
        out_root=tmp_path,
        pairs_file=None,
        default_model=None,
        pair_set="within-model",
        models="mA",
        parent_ladder_dir=tmp_path / "noladder",
    )
    (tmp_path / "pairs").mkdir()
    rc = dvf.cmd_merge(args)
    assert rc == 3  # never-attempted units are a fail-loud, not a silent hole
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_missing"] == summary["n_expected_units"] > 0


def test_gate1_parity_refusal_rc7(tmp_path):
    """A doctored published target must trip the parity gate with the DISTINCT
    designed-halt rc 7 (#1415 convention), writing gate1_report.json."""
    store = tmp_path / "store"
    (sm, sc), (_tm, tc) = dvf.GATE1_PAIR
    _write_synth_store(store, sm, [sc, tc], n=60, d=8)
    doctored = {
        "arms": {
            "context": {
                "rung_r2s_point": {f"rung_{i}_x": 99.0 for i in range(1, 10)},
                "rung_reached_point": 1,
                "n_common": 60,
            }
        }
    }
    # match real rung key names so the diff computation runs
    real_keys = [
        "rung_1_direct",
        "rung_2_ctx_offset",
        "rung_3_ans_offset",
        "rung_4_bias_refit",
        "rung_5_scalar_alpha",
        "rung_6_rotation",
        "rung_7_ctx_reparam",
        "rung_8_ans_reparam",
        "rung_9_full_AMB",
    ]
    doctored["arms"]["context"]["rung_r2s_point"] = {k: 99.0 for k in real_keys}
    target = tmp_path / "doctored.json"
    target.write_text(json.dumps(doctored))
    args = _dvf_args(
        store_root=store,
        out_root=tmp_path / "out",
        gate1_target=target,
        gate1_null_draws=0,
        gate1_timing=False,
    )
    rc = dvf.cmd_gate1(args)
    assert rc == 7
    report = json.loads((tmp_path / "out" / "gate1_report.json").read_text())
    assert report["parity"]["ok"] is False


def test_class_ladder_translation_equals_identity_bias():
    """Item-7a floor: the translation class IS identity_bias_predict on x."""
    rng = np.random.default_rng(5)
    n, d = 80, 10
    x_s = rng.standard_normal((n, d))
    x_t = x_s @ (np.eye(d) * 1.1) + 0.5
    conv = np.array([f"c{i}" for i in range(n)])
    tr = torch.arange(0, 60)
    te = torch.arange(60, n)
    preds, _ops = cms._class_preds_for_split(
        torch.from_numpy(x_s), torch.from_numpy(x_t), tr, te, conv[:60], fl.LAMBDAS, "cpu"
    )
    expect = identity_bias_predict(x_s[:60], x_t[:60], x_s[60:])
    np.testing.assert_allclose(preds["translation"].numpy(), expect, atol=1e-10)
    # nesting: full affine must dominate translation on a non-translation world
    r2_full = fl._r2(x_t[60:], preds["full_affine"].numpy())
    r2_trans = fl._r2(x_t[60:], preds["translation"].numpy())
    assert r2_full > r2_trans
    assert (
        cms._class_reached(
            {c: -1.0 for c in cms.CLASS_ORDER} | {"full_affine": r2_full}, r2_full, 0.9
        )
        == "full_affine"
    )
    assert cms._class_reached(dict.fromkeys(cms.CLASS_ORDER, 1.0), 1.0, 0.9) == "translation"


def test_regime_meta_resume_key_covers_slice_knobs():
    """#722 r3 rule: row/dim slicing is output-affecting -> in the resume key."""
    a1 = dvf.regime_meta(_dvf_args())
    a2 = dvf.regime_meta(_dvf_args(row_limit=600))
    assert a1 != a2
    s1 = cms.regime_meta(_cms_args())
    s2 = cms.regime_meta(_cms_args(class_null_draws=4))
    assert s1 != s2


def _cms_args(**over):
    base = dict(
        layer=19,
        lambda_grid="ladder13",
        seed=42,
        device="cpu",
        row_limit=None,
        dim_limit=None,
        class_null_draws=40,
        rank_null_draws=40,
        items="both",
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_merge_pairs_generalized_roundtrip_and_fail_loud(tmp_path):
    store = tmp_path / "store"
    _write_synth_store(store, "mA", ["assistant_chat", "assistant_naturalistic"])
    spec = (("mA", "assistant_chat"), ("mA", "assistant_naturalistic"))
    ck = tmp_path / "ck"
    ran = fl.run_pairs_generalized(
        store,
        [spec],
        layer=19,
        n_bootstrap_draws=0,
        n_null_draws=0,
        engine="numpy",
        checkpoint_dir=ck,
    )
    merged = fl.merge_pairs_generalized(ck, [spec], layer=19, n_bootstrap_draws=0, n_null_draws=0)
    # json-text equality (NaN != NaN under ==; serialized form is identical)
    assert json.dumps(merged["pairs"], sort_keys=True) == json.dumps(ran["pairs"], sort_keys=True)
    other = (("mA", "assistant_chat"), ("mB", "assistant_chat"))
    with pytest.raises(RuntimeError, match="generalized merge incomplete"):
        fl.merge_pairs_generalized(ck, [spec, other], layer=19, n_bootstrap_draws=0, n_null_draws=0)


def test_cmd_pairs_shard_coverage_and_resume(tmp_path, capsys):
    """Shards partition the unit list; a re-run RESUMEs from checkpoints."""
    store = tmp_path / "store"
    _write_synth_store(store, "mA", ["assistant_chat", "assistant_naturalistic"])
    pf = tmp_path / "pairs.json"
    pf.write_text(json.dumps([[["mA", "assistant_chat"], ["mA", "assistant_naturalistic"]]]))
    out_root = tmp_path / "out"
    common = dict(
        layer=19,
        lambda_grid="ladder13",
        seed=42,
        device="cpu",
        row_limit=None,
        dim_limit=None,
        rotation_draws=2,
        pairs_file=pf,
        default_model=None,
        pair_set="within-model",
        models="mA",
        out_root=out_root,
        store_root=store,
    )
    for shard in (0, 1):
        rc = dvf.cmd_pairs(argparse.Namespace(**common, num_shards=2, shard_index=shard))
        assert rc == 0
    written = sorted(p.name for p in (out_root / "pairs").glob("*.json"))
    assert written == [
        "mA__assistant_chat__assistant_naturalistic__context.json",
        "mA__assistant_chat__assistant_naturalistic__prefix.json",
    ]
    capsys.readouterr()
    rc = dvf.cmd_pairs(argparse.Namespace(**common, num_shards=1, shard_index=0))
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("RESUME (checkpoint)") == 2


def test_prepped_inner_cv_matches_plain():
    """fit_inner_group_cv_cached_t (shared X-side eigh across target-varying
    null draws — the #823 shared-factorization fix) reproduces
    _fit_ridge_inner_group_cv_t bit-for-bit-equivalently (same eigh, same fold
    masks, same lambda argmax; predictions linear in the centered target)."""
    rng = np.random.default_rng(11)
    n, d = 90, 12
    X = torch.from_numpy(rng.standard_normal((n, d)))
    conv = np.array([f"c{i // 3}" for i in range(n)])  # grouped convs
    lams = fl.LAMBDAS
    cache = fl.build_inner_cv_cache_t(X, conv)
    for _trial in range(3):
        Y = torch.from_numpy(rng.standard_normal((n, d)))
        W0, b0, lam0 = fl._fit_ridge_inner_group_cv_t(X, Y, conv, lams)
        W1, b1, lam1 = fl.fit_inner_group_cv_cached_t(cache, Y, lams)
        assert lam0 == lam1
        np.testing.assert_allclose(W0.numpy(), W1.numpy(), atol=1e-12)
        np.testing.assert_allclose(b0.numpy(), b1.numpy(), atol=1e-12)


def test_cmd_nulls_draw_count_upgrade_updates_metadata(tmp_path):
    """A draw-count-upgrade rerun recomputes the null battery AND rewrites the
    persisted regime metadata — a setdefault'd n_draws would leave the unit
    claiming the OLD draw count after the 200-draw recompute (r-v17 minor)."""
    store = tmp_path / "store"
    _write_synth_store(store, "mA", ["assistant_chat", "assistant_naturalistic"])
    pf = tmp_path / "pairs.json"
    pf.write_text(json.dumps([[["mA", "assistant_chat"], ["mA", "assistant_naturalistic"]]]))
    out_root = tmp_path / "out"
    common = dict(
        layer=19,
        lambda_grid="ladder13",
        seed=42,
        device="cpu",
        row_limit=None,
        dim_limit=None,
        pairs_file=pf,
        default_model=None,
        pair_set="within-model",
        models="mA",
        out_root=out_root,
        store_root=store,
        num_shards=1,
        shard_index=0,
    )
    assert dvf.cmd_pairs(argparse.Namespace(**common, rotation_draws=2)) == 0
    assert dvf.cmd_nulls(argparse.Namespace(**common, rotation_draws=2)) == 0
    unit_paths = sorted((out_root / "pairs").glob("*.json"))
    assert unit_paths
    for p in unit_paths:
        assert json.loads(p.read_text())["operator_read"]["rotation_null"]["n_draws"] == 2
    # Upgrade 2 -> 4 draws: the skip predicate re-includes every unit; the
    # persisted metadata must read 4 afterwards (pre-fix: setdefault kept 2).
    assert dvf.cmd_nulls(argparse.Namespace(**common, rotation_draws=4)) == 0
    for p in unit_paths:
        unit = json.loads(p.read_text())
        nul = unit["operator_read"]["rotation_null"]
        assert nul["n_draws"] == 4
        assert nul["seed"] == 42
        with np.load(out_root / "bundles" / f"{unit['unit_key']}.npz") as z:
            draw_keys = [k for k in z.files if k.startswith("rotation_draws_")]
            assert draw_keys
            for k in draw_keys:
                assert z[k].shape[0] == 4


def test_unit_key_model_qualified():
    """Fix round 3: within-model keys carry the model slug (distinct across
    models sharing an out-root); cross-model keys are byte-unchanged."""
    wa = dvf.unit_key((("mA", "s"), ("mA", "t")), "prefix")
    wb = dvf.unit_key((("mB", "s"), ("mB", "t")), "prefix")
    assert wa == "mA__s__t__prefix"
    assert wb == "mB__s__t__prefix"
    assert wa != wb
    assert dvf.unit_key((("m1", "a"), ("m2", "a")), "context") == "m1@a__m2@a__context"


def test_two_model_pairs_no_checkpoint_collision(tmp_path, capsys):
    """THE round-3 regression: two models over the SAME out-root must compute
    ALL units (pre-fix, the second model's units skip-if-exists'd against the
    first model's unqualified files — 241 base + 11 instruct of 504)."""
    store = tmp_path / "store"
    _write_synth_store(store, "mA", ["assistant_chat", "assistant_naturalistic"], seed=7)
    _write_synth_store(store, "mB", ["assistant_chat", "assistant_naturalistic"], seed=8)
    pf = tmp_path / "pairs.json"
    pf.write_text(
        json.dumps(
            [
                [["mA", "assistant_chat"], ["mA", "assistant_naturalistic"]],
                [["mB", "assistant_chat"], ["mB", "assistant_naturalistic"]],
            ]
        )
    )
    out_root = tmp_path / "out"
    args = argparse.Namespace(
        layer=19,
        lambda_grid="ladder13",
        seed=42,
        device="cpu",
        row_limit=None,
        dim_limit=None,
        rotation_draws=2,
        pairs_file=pf,
        default_model=None,
        pair_set="within-model",
        models="mA,mB",
        out_root=out_root,
        store_root=store,
        num_shards=1,
        shard_index=0,
    )
    assert dvf.cmd_pairs(args) == 0
    written = sorted(p.name for p in (out_root / "pairs").glob("*.json"))
    assert written == [
        "mA__assistant_chat__assistant_naturalistic__context.json",
        "mA__assistant_chat__assistant_naturalistic__prefix.json",
        "mB__assistant_chat__assistant_naturalistic__context.json",
        "mB__assistant_chat__assistant_naturalistic__prefix.json",
    ]
    # Each file records ITS OWN model — no cross-model checkpoint reuse.
    for p in (out_root / "pairs").glob("*.json"):
        unit = json.loads(p.read_text())
        assert unit["src_model"] == p.name.split("__")[0]
        assert unit["unit_key"] == p.stem
    # And a re-run RESUMEs all 4 (no recompute, no collision).
    capsys.readouterr()
    assert dvf.cmd_pairs(args) == 0
    assert capsys.readouterr().out.count("RESUME (checkpoint)") == 4


def test_migrate_unqualified_keys(tmp_path):
    """One-shot migration renames pre-fix UNQUALIFIED files (JSON + bundle) to
    their internally-recorded model's qualified key; unattributable and
    target-exists files are retained in place; idempotent second pass."""
    pairs = tmp_path / "pairs"
    bundles = tmp_path / "bundles"
    pairs.mkdir()
    bundles.mkdir()
    meta = {"battery_version": "dvf-v1"}
    computed = {
        "meta": meta,
        "src_model": "mA",
        "src_cond": "assistant_chat",
        "tgt_model": "mA",
        "tgt_cond": "dana_chat",
        "arm": "prefix",
        "unit_key": "assistant_chat__dana_chat__prefix",
        "verdict": "shared_readout_supported",
    }
    (pairs / "assistant_chat__dana_chat__prefix.json").write_text(json.dumps(computed))
    np.savez(bundles / "assistant_chat__dana_chat__prefix.npz", x=np.zeros(2))
    # Unattributable error unit (pre-fix error units record no model): retained.
    (pairs / "dana_chat__wren_chat__context.json").write_text(
        json.dumps({"error": "boom", "retryable": True, "meta": meta})
    )
    # Target-exists case: old unqualified + fresh qualified both present.
    clash_old = dict(computed, arm="context", unit_key="assistant_chat__dana_chat__context")
    (pairs / "assistant_chat__dana_chat__context.json").write_text(json.dumps(clash_old))
    (pairs / "mA__assistant_chat__dana_chat__context.json").write_text(
        json.dumps(dict(clash_old, unit_key="mA__assistant_chat__dana_chat__context"))
    )
    counts = dvf.migrate_unqualified_keys(tmp_path)
    assert counts["renamed"] == 1
    assert counts["unattributable"] == 1
    assert counts["target_exists"] == 1
    assert counts["already_qualified"] == 1
    migrated = json.loads((pairs / "mA__assistant_chat__dana_chat__prefix.json").read_text())
    assert migrated["unit_key"] == "mA__assistant_chat__dana_chat__prefix"
    assert migrated["unit_key_migrated_from"] == "assistant_chat__dana_chat__prefix"
    assert migrated["verdict"] == "shared_readout_supported"
    assert not (pairs / "assistant_chat__dana_chat__prefix.json").exists()
    assert (bundles / "mA__assistant_chat__dana_chat__prefix.npz").exists()
    assert not (bundles / "assistant_chat__dana_chat__prefix.npz").exists()
    # Retained (never deleted): unattributable + target-exists old files.
    assert (pairs / "dana_chat__wren_chat__context.json").exists()
    assert (pairs / "assistant_chat__dana_chat__context.json").exists()
    # Idempotent: second pass renames nothing.
    counts2 = dvf.migrate_unqualified_keys(tmp_path)
    assert counts2["renamed"] == 0
    assert counts2["already_qualified"] == 2


def test_merge_duplicate_keys_and_expect_units_raise(tmp_path):
    """Double-count probe: duplicate enumerated units RAISE; an --expect-units
    mismatch RAISES (the 504-unit hard assert the dispatch leg arms)."""
    spec_row = [["mA", "assistant_chat"], ["mA", "assistant_naturalistic"]]
    pf_dup = tmp_path / "dup.json"
    pf_dup.write_text(json.dumps([spec_row, spec_row]))
    (tmp_path / "pairs").mkdir()
    common = dict(
        layer=19,
        lambda_grid="ladder13",
        seed=42,
        device="cpu",
        row_limit=None,
        dim_limit=None,
        default_model=None,
        pair_set="within-model",
        models="mA",
        out_root=tmp_path,
        parent_ladder_dir=tmp_path / "noladder",
    )
    with pytest.raises(RuntimeError, match="residual double-count"):
        dvf.cmd_merge(argparse.Namespace(**common, pairs_file=pf_dup, expect_units=None))
    pf_one = tmp_path / "one.json"
    pf_one.write_text(json.dumps([spec_row]))
    with pytest.raises(RuntimeError, match="expected 504 distinct"):
        dvf.cmd_merge(argparse.Namespace(**common, pairs_file=pf_one, expect_units=504))
    # cms merge shares the same guards via the imported helpers.
    with pytest.raises(RuntimeError, match="residual double-count"):
        cms.cmd_merge(argparse.Namespace(**common, pairs_file=pf_dup, expect_units=None))


def test_merge_key_mismatch_fail_loud(tmp_path):
    """A checkpoint whose INTERNAL model contradicts its qualified key is never
    counted — rc 3 + n_key_mismatch (the belt-and-suspenders identity check)."""
    spec_row = [["mA", "assistant_chat"], ["mA", "assistant_naturalistic"]]
    pf = tmp_path / "pairs.json"
    pf.write_text(json.dumps([spec_row]))
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    for arm in ("prefix", "context"):
        (pairs / f"mA__assistant_chat__assistant_naturalistic__{arm}.json").write_text(
            json.dumps(
                {
                    "src_model": "mB",  # WRONG model recorded inside
                    "src_cond": "assistant_chat",
                    "tgt_model": "mB",
                    "tgt_cond": "assistant_naturalistic",
                    "arm": arm,
                    "meta": {},
                }
            )
        )
    args = argparse.Namespace(
        layer=19,
        lambda_grid="ladder13",
        seed=42,
        device="cpu",
        row_limit=None,
        dim_limit=None,
        pairs_file=pf,
        default_model=None,
        pair_set="within-model",
        models="mA",
        out_root=tmp_path,
        parent_ladder_dir=tmp_path / "noladder",
        expect_units=2,
    )
    assert dvf.cmd_merge(args) == 3
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_key_mismatch"] == 2
    assert summary["n_complete"] == 0


def test_rung78_degenerate_split_raises_in_production_shape():
    """fit_rung78_corrections_t: a degenerate conv-grouped split RAISES in
    production shape (row_limit/dim_limit both None); the train==test fallback
    is confined to the explicit smoke regime (r-v17 minor)."""
    rng = np.random.default_rng(3)
    n, d = 3, 6  # 3 convs over 5 folds -> train fold has 2 rows (< 3): degenerate
    x = rng.standard_normal((n, d))
    y = x @ rng.standard_normal((d, d))
    source = _synthetic_cell(rng, n, d, x, y)
    target = _synthetic_cell(rng, n, d, x + 0.1, y + 0.1)
    with pytest.raises(RuntimeError, match="degenerate conv-grouped split"):
        fl.fit_rung78_corrections_t(source, target, arm="context")
    # Explicit smoke regime (row_limit set): fallback proceeds, train==test.
    corr = fl.fit_rung78_corrections_t(source, target, arm="context", row_limit=n)
    assert "error" not in corr
    np.testing.assert_array_equal(corr["train_idx"], corr["test_idx"])
