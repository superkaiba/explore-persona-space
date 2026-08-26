"""CPU-only pins for scripts/issue2587_fits.py (issue #2587 unit 4).

No network, no HF fetch, no GPU. Two REQUIRED pins:

1. The ``fit_ridge`` call shape is the 8-argument form
   ``fit_ridge(X, Y, tr, val, te, lambdas, dev, block)`` with
   ``LF.RIDGE_BLOCK`` threaded as ``block`` — the plan-v1 7-arg shape must
   NOT bind (it would have TypeError'd after ~7 GPU-h of P2/P3 spend).
2. The §4.5 row-identity assertion for ``arm_7b_matched25k``: exact
   ordered-ID-set equality per split — equal ids proceed; a permutation or
   subset HALTS (RuntimeError), never a near-match or intersection.
"""

from __future__ import annotations

import fnmatch
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import issue2587_fits as I  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _toy(n=60, d=8, seed=0, noise=0.3, signal=True):
    """Tiny real fit fixture (n > d so the fit is well-posed)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    W = rng.normal(size=(d, d)).astype(np.float32)
    if signal:
        Y = (X @ W + noise * rng.normal(size=(n, d))).astype(np.float32)
    else:
        Y = rng.normal(size=(n, d)).astype(np.float32)
    tr = np.arange(0, 40)
    val = np.arange(40, 50)
    te = np.arange(50, 60)
    return X, Y, tr, val, te


def _toy_edge_high():
    """Real-body λ-grid-edge exhaustion fixture: val targets sign-flipped
    (anti-correlated with the train map), so shrinkage monotonically helps and
    the selected λ pins the HIGH grid edge on every extension pass (probed
    through the real fit; the noiseless / pure-noise fixtures select interior
    λ — feedback_synthetic_ridge_fixture_edge_lambda)."""
    rng = np.random.default_rng(0)
    n, d = 60, 8
    X = rng.normal(size=(n, d)).astype(np.float32)
    W = rng.normal(size=(d, d)).astype(np.float32)
    Y = (X @ W).astype(np.float32).copy()
    tr = np.arange(0, 40)
    val = np.arange(40, 50)
    te = np.arange(50, 60)
    Y[val] = -Y[val]
    return X, Y, tr, val, te


def _stub_fit(edge, calls=None, seen=None):
    """Signature-conformant stub for the fit_fn seam (8-arg shape mirrored)."""

    def fake_fit(X, Y, tr, val, ev, grid, dev, block):
        if calls is not None:
            calls.append(len(grid))
        if seen is not None:
            seen["block"] = block
            seen["dev"] = str(dev)
        meta = {
            "n_train": len(tr),
            "selection": "stub",
            "selected_lambda": float(grid[-1] if edge == "high" else grid[0]),
            "val_r2_at_selected": 0.0,
            "lambda_grid_edge": edge,
            "ridge_block": int(block),
        }
        return np.zeros((len(ev), Y.shape[1]), dtype=np.float64), meta, {"kind": "ridge"}

    return fake_fit


# ---------------------------------------------------------------------------
# REQUIRED pin 1 — fit_ridge 8-arg shape, block == LF.RIDGE_BLOCK
# ---------------------------------------------------------------------------


def test_fit_ridge_eight_arg_shape_binds_and_seven_arg_raises():
    sig = inspect.signature(I.F.fit_ridge)
    X, Y, tr, val, te = _toy()
    # 8 positional args bind (X, Y, tr, val, te, lambdas, dev, block)
    sig.bind(X, Y, tr, val, te, I.LF.LAMBDAS, torch.device("cpu"), int(I.LF.RIDGE_BLOCK))
    # the plan-v1 7-arg shape (block omitted) must NOT bind
    with pytest.raises(TypeError):
        sig.bind(X, Y, tr, val, te, I.LF.LAMBDAS, torch.device("cpu"))


def test_fit_ridge_tiny_real_call_returns_pred_meta_with_ridge_block():
    X, Y, tr, val, te = _toy()
    pred, meta = I.F.fit_ridge(
        X, Y, tr, val, te, I.LF.LAMBDAS, torch.device("cpu"), int(I.LF.RIDGE_BLOCK)
    )
    assert pred.shape == (len(te), Y.shape[1])
    assert int(meta["ridge_block"]) == int(I.LF.RIDGE_BLOCK)
    for key in ("selected_lambda", "val_r2_at_selected", "lambda_grid_edge", "n_train"):
        assert key in meta, key
    assert meta["n_train"] == len(tr)


def test_edge_extended_helper_threads_ridge_block_default():
    seen: dict = {}
    X, Y, tr, val, te = _toy()
    I.fit_ridge_edge_extended_weights(
        X, Y, tr, val, te, torch.device("cpu"), fit_fn=_stub_fit(None, seen=seen)
    )
    assert seen["block"] == int(I.LF.RIDGE_BLOCK)


# ---------------------------------------------------------------------------
# REQUIRED pin 2 — §4.5 row-identity assertion (both directions)
# ---------------------------------------------------------------------------


def test_rows_for_equal_ordered_ids_proceeds():
    ci = np.array([10, 20, 30, 40], dtype=np.int64)
    rows = I._rows_for(ci, [20, 40], "pin2-equal")
    assert rows.tolist() == [1, 3]
    # full-set, permuted gather order follows ids order (row-aligned)
    rows_full = I._rows_for(ci, [40, 10, 30, 20], "pin2-order")
    assert rows_full.tolist() == [3, 0, 2, 1]


def test_matched_ids_permuted_halts_row_order():
    with pytest.raises(RuntimeError, match="row ORDER differs"):
        I.MF._assert_matched_ids([2, 1], [1, 2], "pin2-permuted")


def test_matched_ids_subset_halts_id_sets():
    with pytest.raises(RuntimeError, match="id SETS differ"):
        I.MF._assert_matched_ids([1, 2], [1, 3], "pin2-subset")


def test_rows_for_missing_id_halts():
    with pytest.raises(RuntimeError, match="absent from the streamed store"):
        I._rows_for(np.array([1, 2], dtype=np.int64), [1, 5], "pin2-missing")


def test_sha_ids_matches_unit2_convention_and_is_order_sensitive():
    import hashlib

    ids = [3, 1, 2]
    expected = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    assert I._sha_ids(ids) == expected
    assert I._sha_ids(np.array(ids, dtype=np.int64)) == expected  # numpy ints coerced
    assert I._sha_ids([1, 2, 3]) != I._sha_ids([3, 2, 1])


def test_verify_split_sha_mismatch_halts_and_match_passes():
    I._verify_split_sha([1, 2], I._sha_ids([1, 2]), "test_1000")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        I._verify_split_sha([1, 2], "deadbeef", "test_1000")


# ---------------------------------------------------------------------------
# Anchor gate — constants pinned; miss halts; pass records
# ---------------------------------------------------------------------------


def test_anchor_constants_pinned():
    assert I.MF.ANCHOR_EXPECTED_R2 == 0.7250873220237553
    assert I.MF.ANCHOR_TOL == 0.01
    assert I.LF.EXPECTED_SPLIT_N == {
        "train_25k": 25000,
        "val_400": 400,
        "test_1000": 1000,
        "wc_test_1k": 999,
    }
    assert I.STORE_REVISION_PIN_7B == "815ff6d976c686af8672b27cfdfb1ce6b419c02c"


def test_anchor_gate_miss_halts():
    X, Y, tr, val, te = _toy()
    with pytest.raises(RuntimeError, match="PORT-PARITY ANCHOR GATE MISS"):
        I.MF.run_anchor_gate(X, Y, tr, val, te, torch.device("cpu"), expected_r2=999.0, tol=0.01)


def test_anchor_gate_pass_records_realized_r2():
    X, Y, tr, val, te = _toy()
    pred, _meta = I.F.fit_ridge(
        X, Y, tr, val, te, I.LF.LAMBDAS, torch.device("cpu"), int(I.LF.RIDGE_BLOCK)
    )
    realized = float(I.LF._pooled_r2(pred, Y[te]))
    rec = I.MF.run_anchor_gate(
        X, Y, tr, val, te, torch.device("cpu"), expected_r2=realized, tol=0.01
    )
    assert abs(rec["realized_r2"] - realized) <= 1e-6
    assert rec["abs_deviation"] <= 1e-6
    assert rec["investigate_before_narrate"] is False


# ---------------------------------------------------------------------------
# λ-grid-edge extension (real body + stub seam)
# ---------------------------------------------------------------------------


def test_edge_extension_benign_signal_returns_payload():
    X, Y, tr, val, te = _toy(noise=0.3)  # probed: interior λ, edge None
    pred, meta, payload = I.fit_ridge_edge_extended_weights(X, Y, tr, val, te, torch.device("cpu"))
    assert payload["kind"] == "ridge"
    assert meta["lambda_grid_edge"] is None
    assert meta["grid_extensions"] == []
    assert meta["device_realized"] == "cpu"
    assert pred.shape == (len(te), Y.shape[1])


def test_edge_extension_exhaustion_real_body_halts():
    # anti-correlated val targets pin the HIGH edge on every extension pass
    X, Y, tr, val, te = _toy_edge_high()
    with pytest.raises(RuntimeError, match="grid edge"):
        I.fit_ridge_edge_extended_weights(X, Y, tr, val, te, torch.device("cpu"))


def test_edge_extension_stub_runs_max_plus_one_fits_on_growing_grids():
    calls: list[int] = []
    X, Y, tr, val, te = _toy()
    with pytest.raises(RuntimeError, match="grid edge"):
        I.fit_ridge_edge_extended_weights(
            X, Y, tr, val, te, "cpu", fit_fn=_stub_fit("high", calls=calls)
        )
    assert len(calls) == int(I.MF.MAX_GRID_EXTENSIONS) + 1
    assert calls == sorted(calls) and calls[-1] > calls[0]  # grid grew each pass


def test_edge_extension_disabled_returns_first_fit_with_edge_recorded():
    calls: list[int] = []
    X, Y, tr, val, te = _toy()
    out = I.fit_ridge_edge_extended_weights(
        X, Y, tr, val, te, "cpu", extend=False, fit_fn=_stub_fit("high", calls=calls)
    )
    assert len(calls) == 1
    assert out[1]["lambda_grid_edge"] == "high"


def test_cusolver_linalgerror_falls_back_to_cpu():
    calls: list[str] = []

    def flaky_fit(X, Y, tr, val, ev, grid, dev, block):
        calls.append(str(dev))
        if str(dev) != "cpu":
            raise torch.linalg.LinAlgError("eigh failed to converge (stub)")
        meta = {
            "selected_lambda": 1.0,
            "val_r2_at_selected": 0.0,
            "lambda_grid_edge": None,
            "ridge_block": int(block),
            "n_train": len(tr),
        }
        return np.zeros((len(ev), Y.shape[1])), meta, {"kind": "ridge"}

    X, Y, tr, val, te = _toy()
    out = I.fit_ridge_edge_extended_weights(X, Y, tr, val, te, "fake-cuda", fit_fn=flaky_fit)
    assert out[1]["device_realized"] == "cpu"
    assert calls == ["fake-cuda", "cpu"]


# ---------------------------------------------------------------------------
# Payload round-trip: apply_map reproduces pred_te (justifies wc-via-apply_map)
# ---------------------------------------------------------------------------


def test_apply_map_reproduces_pred_te():
    X, Y, tr, val, te = _toy(noise=0.3)
    pred, _meta, payload = I.F.fit_ridge_with_weights(
        X, Y, tr, val, te, I.LF.LAMBDAS, torch.device("cpu"), int(I.LF.RIDGE_BLOCK)
    )
    out = I.F.apply_map(payload, X[te], torch.device("cpu"))
    np.testing.assert_allclose(out, pred, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# L* freeze
# ---------------------------------------------------------------------------


def test_compute_lstar_argmax_frozen_and_tie_breaks_low():
    per_layer = {
        li: {"ridge": {"meta": {"val_r2_at_selected": v}}}
        for li, v in {0: 0.1, 5: 0.9, 7: 0.9, 31: 0.2}.items()
    }
    blk = I.compute_lstar(per_layer)
    assert blk["lstar"] == 5  # tie with layer 7 breaks to the LOWEST layer
    assert blk["frozen"] is True
    assert blk["val_r2_by_layer"]["31"] == 0.2
    with pytest.raises(RuntimeError, match="empty"):
        I.compute_lstar({})


# ---------------------------------------------------------------------------
# Reliability ceiling (LF arithmetic mirrored; pairing pins)
# ---------------------------------------------------------------------------


def test_ceiling_identical_draws_is_one():
    rng = np.random.default_rng(1)
    n, d = 20, 6
    V = rng.normal(size=(n, d)).astype(np.float32)
    ci = list(range(100, 100 + n))
    rec = I.ceiling_from_draws(ci, V, ci, V, ci, n, "ceiling-identical")
    assert rec["ceiling_var_weighted_r"] == pytest.approx(1.0, abs=1e-5)
    assert rec["n_pairs"] == n
    assert rec["available"] is True


def test_ceiling_pairs_by_ci_not_position():
    rng = np.random.default_rng(2)
    n, d = 20, 6
    V = rng.normal(size=(n, d)).astype(np.float32)
    ci = list(range(100, 100 + n))
    perm = rng.permutation(n)
    rec = I.ceiling_from_draws(ci, V, [ci[j] for j in perm], V[perm], ci, n, "ceiling-perm")
    assert rec["ceiling_var_weighted_r"] == pytest.approx(1.0, abs=1e-5)


def test_ceiling_banked_shortfall_halts():
    rng = np.random.default_rng(3)
    n, d = 10, 4
    V = rng.normal(size=(n, d)).astype(np.float32)
    ci = list(range(n))
    with pytest.raises(RuntimeError, match="pairing shortfall"):
        I.ceiling_from_draws(ci[:-1], V[:-1], ci, V, ci, n, "ceiling-short")


def test_ceiling_missing_id_halts():
    rng = np.random.default_rng(4)
    n, d = 10, 4
    V = rng.normal(size=(n, d)).astype(np.float32)
    ci = list(range(n))
    with pytest.raises(RuntimeError, match="pairing shortfall"):
        I.ceiling_from_draws(ci, V, ci, V, [0, 1, 999], n, "ceiling-missing")


# ---------------------------------------------------------------------------
# vc2564 loader (observed banked schema)
# ---------------------------------------------------------------------------


def _fake_bank(tmp_path: Path, n=8, layers=(14, 19, 26), d=12):
    vc = torch.arange(n * len(layers) * d, dtype=torch.float32).reshape(n, len(layers), d)
    ctx = [f"query::E::c{i:02d}" for i in range(n)]
    store = tmp_path / "vc2564_bank.pt"
    torch.save(
        {
            "context_ids": ctx,
            "dtype": "float32",
            "issue": 2564,
            "layers": list(layers),
            "position": "final",
            "repro": {},
            "vc": vc,
        },
        store,
    )
    manifest = tmp_path / "bank2564_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "issue": 2564,
                "n_contexts": n,
                "contexts": [
                    {
                        "id": c,
                        "cell": "x",
                        "carrier": "y",
                        "form": "z",
                        "kind": "k",
                        "system": "",
                        "user": "",
                        "value_id": "v",
                    }
                    for c in ctx
                ],
            }
        ),
        encoding="utf-8",
    )
    return store, manifest, vc, ctx


def test_load_vc2564_selects_layer19_column(tmp_path):
    store, manifest, vc, ctx = _fake_bank(tmp_path)
    X, got_ctx = I.load_vc2564(store, manifest, 19, len(ctx))
    np.testing.assert_array_equal(X, vc[:, 1, :].numpy())  # layers=[14,19,26] -> col 1
    assert got_ctx == ctx


def test_load_vc2564_count_violation_halts(tmp_path):
    store, manifest, _vc, ctx = _fake_bank(tmp_path)
    with pytest.raises(RuntimeError, match="count violation"):
        I.load_vc2564(store, manifest, 19, len(ctx) + 1)


def test_load_vc2564_membership_violation_halts(tmp_path):
    store, manifest, _vc, ctx = _fake_bank(tmp_path)
    m = json.loads(manifest.read_text())
    m["contexts"] = [*m["contexts"][:-1], {**m["contexts"][-1], "id": "query::E::zz"}]
    manifest.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(RuntimeError, match="membership violation"):
        I.load_vc2564(store, manifest, 19, len(ctx))


def test_load_vc2564_missing_layer_halts(tmp_path):
    store, manifest, _vc, ctx = _fake_bank(tmp_path)
    with pytest.raises(RuntimeError, match="lacks layer"):
        I.load_vc2564(store, manifest, 99, len(ctx))


# ---------------------------------------------------------------------------
# Regime key: machine-stable generating params, parameter-sensitive
# ---------------------------------------------------------------------------


def test_regime_key_generating_params_stable_and_sensitive():
    kw = dict(
        store_prefix="a",
        split_sha={"train_25k": "x"},
        h_dim=4096,
        selector="val_r2",
        ridge_block=50_000,
        device="cuda",
    )
    k1 = I.regime_key(**kw)
    k2 = I.regime_key(**kw)
    assert k1 == k2
    assert I.regime_key(**{**kw, "h_dim": 3584}) != k1
    assert I.regime_key(**{**kw, "store_prefix": "b"}) != k1
    # λ-grid identity rides GENERATING PARAMS (#1336), and they match LF.LAMBDAS
    assert I.LAMBDA_GRID_KEY == ["logspace", -3.0, 8.0, 23]
    np.testing.assert_allclose(I.LF.LAMBDAS, np.logspace(-3.0, 8.0, 23))


def test_torch_save_atomic_tmp_never_matches_upload_glob(tmp_path):
    import os

    p = tmp_path / "L5.pt"
    I._torch_save_atomic({"a": 1}, p)
    assert p.is_file()
    assert not list(tmp_path.glob("*.tmp"))  # no residue after the atomic replace
    # #2336: the atomic_io temp is PROCESS-UNIQUE (pid + uuid fragment) and its
    # name stays invisible to the L*.pt upload glob
    assert not fnmatch.fnmatch(f"L5.pt.{os.getpid()}.deadbeef.tmp", "L*.pt")
    src = (REPO / "scripts" / "issue2587_fits.py").read_text(encoding="utf-8")
    assert 'path.name + ".tmp"' not in src  # the process-SHARED temp-name shape is gone


def test_parse_layers():
    assert I._parse_layers("0-3") == [0, 1, 2, 3]
    assert I._parse_layers("19") == [19]
    assert I._parse_layers("16,22,30") == [16, 22, 30]
    assert I._parse_layers("0-2,2,31") == [0, 1, 2, 31]
    with pytest.raises(ValueError):
        I._parse_layers("")


# ---------------------------------------------------------------------------
# Smoke mode e2e (unit-2 hook contract) + --import-check
# ---------------------------------------------------------------------------


def test_smoke_mode_end_to_end(tmp_path):
    torch.manual_seed(0)
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    n, layers, h = 60, [3, 7], 16
    for c in range(2):
        torch.save(
            {
                "cx_last": torch.randn(n, len(layers), h),
                "v_x": torch.randn(n, len(layers), h),
                "ci": torch.arange(c * n, (c + 1) * n, dtype=torch.int64),
                "layers": layers,
            },
            chunk_dir / f"shard00_chunk{c:04d}.pt",
        )
    out_json = tmp_path / "smoke_fits.json"
    rc = I.main(
        [
            "--smoke-chunk-dir",
            str(chunk_dir),
            "--device",
            "cpu",
            "--h-dim",
            str(h),
            "--out-json",
            str(out_json),
        ]
    )
    assert rc == 0
    out = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(out["per_layer"]) == {"3", "7"}


# ---------------------------------------------------------------------------
# matched7b resume/repair contract (r1 matched7b-resume-contract) + edge backstop
# ---------------------------------------------------------------------------


def _split_ids_file(tmp_path: Path) -> Path:
    ids = {s: list(range(k * 10, k * 10 + 4)) for k, s in enumerate(I.SPLITS)}
    payload = {
        "splits": ids,
        "counts": {s: len(v) for s, v in ids.items()},
        "sha256": {s: I._sha_ids(v) for s, v in ids.items()},
        "dropped_overlength": [],
    }
    p = tmp_path / "split_ids.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _matched7b_args(tmp_path: Path, upload: str, prefix: str | None):
    import argparse

    out_root = tmp_path / "out"
    out_root.mkdir(exist_ok=True)
    return argparse.Namespace(
        device="cpu",
        split_ids=str(_split_ids_file(tmp_path)),
        out_root=str(out_root),
        anchor_out=str(out_root / "matched7b_anchor.json"),
        sentinel_path=None,
        hf_prefix_7b="issue1491_scale_ladder/scale7_refit",
        revision_7b="deadbeef",
        cache_dir=str(tmp_path / "cache"),
        upload=upload,
        preds7b_prefix=prefix,
        no_edge_extension=False,
        vc2564=None,
        bank_manifest=None,
    )


def _matched7b_rk(args) -> str:
    payload = json.loads(Path(args.split_ids).read_text(encoding="utf-8"))
    return I.regime_key(
        store_prefix=f"{args.hf_prefix_7b}@{args.revision_7b}",
        split_sha=payload["sha256"],
        h_dim=I.H_DIM_7B,
        selector="val_r2",
        ridge_block=int(I.LF.RIDGE_BLOCK),
        device=str(args.device),
    )


def _seed_matched7b_state(args, rk: str, upload_rec: dict, with_arm_files: bool = True) -> Path:
    out_root = Path(args.out_root)
    preds7b = out_root / "preds7b"
    preds7b.mkdir(parents=True, exist_ok=True)
    if with_arm_files:
        for name in I._MATCHED7B_ARM_FILES:
            torch.save({"stub": name}, preds7b / name)
    record = {
        "issue": I.ISSUE,
        "regime_key": rk,
        "upload": upload_rec,
        "complete": True,
        "repro": {},
    }
    Path(args.anchor_out).write_text(json.dumps(record), encoding="utf-8")
    return preds7b


def test_matched7b_completion_gaps_predicate(tmp_path):
    import argparse

    sentinel = tmp_path / "matched7b_done.json"
    prior = {"complete": True, "regime_key": "rk1", "upload": {"mode": "none"}}
    args_none = argparse.Namespace(upload="none", preds7b_prefix=None)
    args_hf = argparse.Namespace(upload="hf", preds7b_prefix="p/x")
    # no sentinel: never a clean skip, whatever the upload mode
    assert I._matched7b_completion_gaps(prior, args_none, sentinel) == ["sentinel"]
    assert I._matched7b_completion_gaps(prior, args_hf, sentinel) == ["upload", "sentinel"]
    # sentinel present + matching regime: upload-none contract is satisfied
    sentinel.write_text(json.dumps({"done": True, "regime_key": "rk1"}), encoding="utf-8")
    assert I._matched7b_completion_gaps(prior, args_none, sentinel) == []
    # a requested hf upload is NOT satisfied by a recorded mode=none...
    assert I._matched7b_completion_gaps(prior, args_hf, sentinel) == ["upload"]
    # ...nor by a recorded hf upload to a DIFFERENT prefix
    prior_hf = {**prior, "upload": {"mode": "hf", "preds7b_prefix": "p/other"}}
    assert I._matched7b_completion_gaps(prior_hf, args_hf, sentinel) == ["upload"]
    prior_match = {**prior, "upload": {"mode": "hf", "preds7b_prefix": "p/x"}}
    assert I._matched7b_completion_gaps(prior_match, args_hf, sentinel) == []
    # a stale sentinel from ANOTHER regime never satisfies
    sentinel.write_text(json.dumps({"done": True, "regime_key": "rkOLD"}), encoding="utf-8")
    assert I._matched7b_completion_gaps(prior, args_none, sentinel) == ["sentinel"]


def test_matched7b_rerun_with_changed_upload_repairs_and_writes_sentinel(tmp_path, monkeypatch):
    """The Codex-named mechanizable case: prior record complete with
    upload.mode=none and NO sentinel; rerun with --upload hf executes the
    upload + sentinel through the PRODUCTION entrypoint (no re-fit, no
    streaming) instead of silently skipping past both."""
    from unittest.mock import create_autospec

    args = _matched7b_args(tmp_path, upload="hf", prefix="issue2587_minpair/preds7b_test")
    rk = _matched7b_rk(args)
    preds7b = _seed_matched7b_state(args, rk, {"mode": "none"})
    fake_upload = create_autospec(I.upload_dir_sharded)
    monkeypatch.setattr(I, "upload_dir_sharded", fake_upload)
    rc = I.run_matched7b(args)
    assert rc == 0
    fake_upload.assert_called_once_with(
        preds7b,
        I.HF_DATA_REPO,
        args.preds7b_prefix,
        shard_glob="*.pt",
        resume_skip=False,
        delete_local=False,
    )
    record = json.loads(Path(args.anchor_out).read_text(encoding="utf-8"))
    assert record["upload"]["mode"] == "hf"
    assert record["upload"]["preds7b_prefix"] == args.preds7b_prefix
    assert record["upload"]["repaired"] is True
    sentinel = Path(args.out_root) / "matched7b_done.json"
    sdoc = json.loads(sentinel.read_text(encoding="utf-8"))
    assert sdoc["done"] is True and sdoc["regime_key"] == rk


def test_matched7b_crash_between_record_and_sentinel_repairs_sentinel(tmp_path, monkeypatch):
    """A crash AFTER the complete record but BEFORE the sentinel write must
    not skip the sentinel forever: the rerun (same upload contract) rewrites
    it idempotently without any upload."""
    from unittest.mock import create_autospec

    args = _matched7b_args(tmp_path, upload="none", prefix=None)
    rk = _matched7b_rk(args)
    _seed_matched7b_state(args, rk, {"mode": "none"})
    fake_upload = create_autospec(I.upload_dir_sharded)
    monkeypatch.setattr(I, "upload_dir_sharded", fake_upload)
    assert I.run_matched7b(args) == 0
    fake_upload.assert_not_called()
    sdoc = json.loads((Path(args.out_root) / "matched7b_done.json").read_text(encoding="utf-8"))
    assert sdoc["done"] is True and sdoc["regime_key"] == rk


def test_matched7b_genuine_complete_skips_without_upload(tmp_path, monkeypatch):
    from unittest.mock import create_autospec

    args = _matched7b_args(tmp_path, upload="hf", prefix="p/x")
    rk = _matched7b_rk(args)
    _seed_matched7b_state(args, rk, {"mode": "hf", "preds7b_prefix": "p/x"})
    I._write_matched7b_sentinel(
        Path(args.out_root) / "matched7b_done.json", rk, Path(args.anchor_out)
    )
    before = Path(args.anchor_out).read_text(encoding="utf-8")
    fake_upload = create_autospec(I.upload_dir_sharded)
    monkeypatch.setattr(I, "upload_dir_sharded", fake_upload)
    assert I.run_matched7b(args) == 0
    fake_upload.assert_not_called()
    assert Path(args.anchor_out).read_text(encoding="utf-8") == before  # untouched


def test_matched7b_repair_impossible_without_arm_files(tmp_path, monkeypatch):
    from unittest.mock import create_autospec

    args = _matched7b_args(tmp_path, upload="hf", prefix="p/x")
    rk = _matched7b_rk(args)
    _seed_matched7b_state(args, rk, {"mode": "none"}, with_arm_files=False)
    monkeypatch.setattr(I, "upload_dir_sharded", create_autospec(I.upload_dir_sharded))
    with pytest.raises(RuntimeError, match="repair impossible"):
        I.run_matched7b(args)


def test_matched7b_regime_mismatch_still_halts(tmp_path):
    args = _matched7b_args(tmp_path, upload="none", prefix=None)
    _seed_matched7b_state(args, "0123456789abcdef", {"mode": "none"})
    with pytest.raises(RuntimeError, match="regime mismatch"):
        I.run_matched7b(args)


def test_edge_selected_layers_helper():
    per_layer = {
        0: {"ridge": {"meta": {"lambda_grid_edge": None}}},
        5: {"ridge": {"meta": {"lambda_grid_edge": "high"}}},
        7: {"ridge": {"meta": {}}},
    }
    assert I._edge_selected_layers(per_layer) == {5: "high"}
    assert I._edge_selected_layers({0: {"ridge": {"meta": {"lambda_grid_edge": None}}}}) == {}


def test_finalize_refuses_edge_selected_rows(tmp_path):
    """r1 g5 Minor 2: a --no-edge-extension fits pass persists an
    edge-selected per-layer row; finalize must REFUSE to freeze L* over it."""
    import argparse

    split_ids = _split_ids_file(tmp_path)
    out_root = tmp_path / "out"
    percell = out_root / "percell"
    percell.mkdir(parents=True)
    rk = "feedcafe00000000"
    for li in range(I.N_LAYERS_9B):
        edge = "high" if li == 3 else None
        row = {
            "regime_key": rk,
            "ridge": {"meta": {"val_r2_at_selected": 0.5, "lambda_grid_edge": edge}},
        }
        (percell / f"L{li}.json").write_text(json.dumps(row), encoding="utf-8")
    args = argparse.Namespace(
        split_ids=str(split_ids),
        out_root=str(out_root),
        ceiling_twins="16,30",
        out_json=str(tmp_path / "sweep.json"),
        sentinel_path=None,
        upload="none",
        payloads_prefix=None,
        preds_prefix=None,
        store_prefix="x",
        h_dim=I.H_DIM_9B,
        cache_dir=str(tmp_path / "cache"),
        local_dir=None,
    )
    with pytest.raises(RuntimeError, match="EDGE-SELECTED"):
        I.run_finalize(args)


def test_import_check_subprocess_exits_zero():
    env = dict(
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        NUMEXPR_NUM_THREADS="8",
        MALLOC_ARENA_MAX="2",
        PATH=__import__("os").environ.get("PATH", ""),
        HOME=__import__("os").environ.get("HOME", ""),
    )
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "issue2587_fits.py"), "--import-check"],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
