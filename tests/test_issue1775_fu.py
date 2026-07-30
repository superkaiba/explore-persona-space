"""#1775 fu round (`dedup-refit-pcfold-doubly`) pins.

Covers the round's NEW statistical machinery on synthetic data (no store /
Hub / GPU): the train-fold-only PC basis mode, the fold-centered pooling
identity behind the cell-2 Delta_named CI, the full dedup drop set + n-matched
random-drop control (planted dupes — the data-dependent drop branches), and
the byte-identical-split pin gate's FAIL branch (degenerate-input probe: the
val/test sha assert must trip on a wrong pin).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))


def test_foldpc_basis_fits_on_train_rows_only():
    from issue1092_fit_grid import _pca_basis
    from issue1775_common import _basis_targets_with_info

    rng = np.random.default_rng(0)
    Y = rng.standard_normal((80, 2 * 7))  # hidden_dim=7, two stacked targets
    tr = np.arange(0, 40)
    Yp, info = _basis_targets_with_info(
        Y,
        "pca48_foldpc",
        train_idx=tr,
        hidden_dim=7,
        targets=["t1", "t2"],
        projection_target="t1",
    )
    mu, v = _pca_basis(Y[tr], 48)
    assert info["basis"] == "pca48_foldpc"
    assert info["train_idx_n"] == 40
    np.testing.assert_allclose(Yp, (Y - mu) @ v, atol=1e-12)
    # a DIFFERENT train fold yields a DIFFERENT basis (train-fold-only, not global)
    Yp2, _ = _basis_targets_with_info(
        Y,
        "pca48_foldpc",
        train_idx=np.arange(40, 80),
        hidden_dim=7,
        targets=["t1", "t2"],
        projection_target="t1",
    )
    assert not np.allclose(Yp, Yp2)
    with pytest.raises(ValueError, match="train_idx"):
        _basis_targets_with_info(
            Y, "pca48_foldpc", hidden_dim=7, targets=["t1", "t2"], projection_target="t1"
        )


def test_fold_centered_bootstrap_point_matches_per_fold_ss_pooling():
    """The cell-2 pooling identity: centering each fold's test rows on the
    fold's test mean (Y AND preds) makes the committed cluster-bootstrap
    helper's point estimate equal 1 - sum_f SE_f / sum_f ST_f pooling."""
    from issue1775_bilinear import _fold_centered
    from issue1775_common import cluster_bootstrap_delta_r2

    rng = np.random.default_rng(1)
    n, d = 60, 4
    folds = [np.arange(0, 30), np.arange(30, 60)]
    Y_by_fold = {f: rng.standard_normal((n, d)) for f in range(2)}  # per-fold bases
    pred_a = {f: Y_by_fold[f] + 0.3 * rng.standard_normal((n, d)) for f in range(2)}
    pred_b = {f: Y_by_fold[f] + 0.9 * rng.standard_normal((n, d)) for f in range(2)}
    Yv = np.zeros((n, d))
    Av = np.zeros((n, d))
    Bv = np.zeros((n, d))
    covered = np.zeros(n, dtype=bool)
    se_a = se_b = st = 0.0
    for f, te in enumerate(folds):
        yv, (av, bv) = _fold_centered(Y_by_fold[f], [pred_a[f][te], pred_b[f][te]], te)
        Yv[te], Av[te], Bv[te] = yv, av, bv
        covered[te] = True
        mu = Y_by_fold[f][te].mean(axis=0, keepdims=True)
        se_a += float(((Y_by_fold[f][te] - pred_a[f][te]) ** 2).sum())
        se_b += float(((Y_by_fold[f][te] - pred_b[f][te]) ** 2).sum())
        st += float(((Y_by_fold[f][te] - mu) ** 2).sum())
    groups = np.asarray([f"g{i % 10}" for i in range(n)])
    boot = cluster_bootstrap_delta_r2(Yv, Av, Bv, covered, groups, n_draws=50, seed=0)
    expected = (se_b - se_a) / st  # R2_a - R2_b under per-fold SS pooling
    assert abs(boot["delta_r2"] - expected) < 1e-12


def test_dedup_drop_set_planted_dupes_and_n_matched_control():
    from issue1775_n50k_dedup_refit import compute_drop_set, random_drop_control

    rng = np.random.default_rng(2)
    vocab = [f"tok{i}" for i in range(40)]
    mk = lambda: " ".join(vocab[int(rng.integers(0, 40))] for _ in range(30))  # noqa: E731
    targets = [mk() for _ in range(5)]
    train = [mk() for _ in range(50)]
    train[7] = targets[2]  # planted EXACT
    train[19] = targets[4] + " extra tail token"  # planted NEAR (long text, J >= 0.8)
    drop = compute_drop_set(train, targets)
    assert 7 in drop["exact_train_positions"]
    assert 19 in drop["near_train_positions"]
    assert set(drop["drop_train_positions"]) >= {7, 19}
    assert 2 in drop["affected_target_ids"] and 4 in drop["affected_target_ids"]
    ctrl = random_drop_control(len(train), drop["drop_train_positions"], seed=0)
    assert len(ctrl) == len(drop["drop_train_positions"])  # n-matched
    assert not set(ctrl) & set(drop["drop_train_positions"])  # control keeps the dupes


def test_build_n50k_split_pin_gate_trips_on_wrong_pin():
    import issue779_ffc_n50k_fits as N

    good = {
        "val_sha256": N.ORIG_VAL_SHA256,
        "test_sha256": N.ORIG_TEST_SHA256,
        "source": "test constants",
    }
    train, _val, _test, diag = N.build_n50k_split(
        46600, None, good, n_train=N.N50K_TRAIN, seed=N.SPLIT_SEED
    )
    assert diag["val_test_byte_identical_original"] and len(train) == N.N50K_TRAIN
    bad = dict(good, val_sha256="0" * 64)
    with pytest.raises(AssertionError, match="NOT byte-identical"):
        N.build_n50k_split(46600, None, bad, n_train=N.N50K_TRAIN, seed=N.SPLIT_SEED)


def _poisoned_upload_dir(tmp_path: Path) -> Path:
    """Payloads + exactly the uploader-excluded residue shapes from the round-1
    fu Critical: hf_hub_download(local_dir=...) metadata + a training-state file."""
    d = tmp_path / "arrays"
    hfmeta = d / ".n50k_stream_cache" / ".cache" / "huggingface"
    (hfmeta / "download").mkdir(parents=True)
    (d / "sub").mkdir()
    (d / "n50k_L19_cx.npy").write_bytes(b"x")
    (d / "n50k_drop_set.json").write_text("{}")
    (hfmeta / "download" / "c0.pt.metadata").write_text("m")
    (hfmeta / ".gitignore").write_text("*")
    (d / "sub" / "optimizer.pt").write_bytes(b"o")
    return d


def test_upload_eligible_relpaths_match_uploader_semantics(tmp_path):
    """C1 pin: the verify expected-set builder applies the SAME excludes the
    uploader does (DEFAULT_IGNORE_PATTERNS + TRAINING_STATE_IGNORE_PATTERNS)."""
    from issue1775_common import _upload_eligible_relpaths

    d = _poisoned_upload_dir(tmp_path)
    assert _upload_eligible_relpaths(d) == ["n50k_L19_cx.npy", "n50k_drop_set.json"]


def test_upload_dir_verified_expected_set_excludes_cache_metadata(tmp_path, monkeypatch):
    """C1 fails-pre-fix pin: `_upload_dir_verified` on a cache-metadata-bearing
    dir must hand verify ONLY upload-eligible paths (pre-fix, the unfiltered
    rglob put `.cache/huggingface` metadata in `expected` -> guaranteed
    RuntimeError after the full stream). Hub boundary faked signature-conformant."""
    import issue1775_common as C

    d = _poisoned_upload_dir(tmp_path)
    captured: list[str] = []

    def fake_upload(local, *, repo_id, repo_type, path_in_repo, raise_on_error=False):
        return "https://huggingface.co/fake"

    def fake_verify(
        api, repo_id, expected_repo_paths, *, path_in_repo, repo_type="dataset", revision=None
    ):
        captured.extend(expected_repo_paths)
        return []

    monkeypatch.setattr(C.hub, "_upload", fake_upload)
    monkeypatch.setattr(C.hub, "verify_repo_paths_uploaded", fake_verify)
    url = C._upload_dir_verified(d, "issue1775_nonlinearity/fu_dedup_refit")
    assert url
    assert captured == [
        "issue1775_nonlinearity/fu_dedup_refit/n50k_L19_cx.npy",
        "issue1775_nonlinearity/fu_dedup_refit/n50k_drop_set.json",
    ]
    assert not any(".cache/huggingface" in p or p.endswith("optimizer.pt") for p in captured)


def test_stream_part_checkpoints_roundtrip_and_regime_keying(tmp_path):
    """M1 pin: per-chunk checkpoints round-trip exactly (fp16 + prompts), the
    parts dir is regime-keyed (smoke max_chunks=1 can never cross-resume
    production), and a fingerprint drift invalidates the cached parts."""
    from issue1775_n50k_dedup_refit import (
        _ensure_parts_meta,
        _load_part,
        _save_part,
        _stream_parts_dir,
    )

    cache = tmp_path / "cache"
    p_full = _stream_parts_dir(cache, "pfx", 19, None)
    p_smoke = _stream_parts_dir(cache, "pfx", 19, 1)
    assert p_full != p_smoke  # regime-keyed: no cross-resume
    meta = {"prefix": "pfx", "layer": 19, "max_chunks": None, "names": ["a.pt", "b.pt"]}
    _ensure_parts_meta(p_full, meta)
    assert _load_part(p_full, 0) is None  # absent -> fresh download path
    rng = np.random.default_rng(0)
    cx = rng.standard_normal((4, 8)).astype(np.float16)
    vx = rng.standard_normal((4, 8)).astype(np.float16)
    prompts = ["p one", "p two", "p three", "p four"]
    _save_part(p_full, 0, cx, vx, prompts)
    got = _load_part(p_full, 0)
    assert got is not None
    np.testing.assert_array_equal(got[0], cx)
    np.testing.assert_array_equal(got[1], vx)
    assert got[2] == prompts
    assert not list(p_full.glob("*.tmp*"))  # atomic: no straggler tmp files
    _ensure_parts_meta(p_full, meta)  # same fingerprint -> parts kept
    assert _load_part(p_full, 0) is not None
    drifted = dict(meta, names=["a.pt", "b.pt", "c.pt"])
    _ensure_parts_meta(p_full, drifted)  # chunk-list drift -> invalidated
    assert _load_part(p_full, 0) is None


def test_fit_regime_keys_cover_krr_grid_and_mlp_epochs():
    """m3 pin: a changed KRR grid (or MLP epoch cap) re-keys ONLY its rung's
    units; rows lacking a per-rung field key consistently via str(None)."""
    from issue1775_common import unit_key
    from issue1775_n50k_dedup_refit import FIT_REGIME_KEYS

    base = {
        "variant": "deduped",
        "rung": "krr",
        "layer": 19,
        "smoke": False,
        "n_train": 100,
        "n_boot": 200,
        "krr_grid": "gm=1.0;lam=0.1,10.0",
    }
    assert unit_key(base, FIT_REGIME_KEYS) != unit_key(
        dict(base, krr_grid="gm=1.0;lam=0.1"), FIT_REGIME_KEYS
    )
    ridge = {
        "variant": "deduped",
        "rung": "ridge",
        "layer": 19,
        "smoke": False,
        "n_train": 100,
        "n_boot": 200,
    }
    assert unit_key(ridge, FIT_REGIME_KEYS) == unit_key(dict(ridge), FIT_REGIME_KEYS)
    mlp = dict(ridge, rung="mlp", mlp_epochs=300)
    assert unit_key(mlp, FIT_REGIME_KEYS) != unit_key(dict(mlp, mlp_epochs=8), FIT_REGIME_KEYS)


def test_doubly_delta_named_gate_precedes_mlp_fit():
    """m2 ordering pin (source-position, the #1586 reap-ordering pattern): the
    run-1-artifacts-only committed-Delta_named validation gate runs BEFORE the
    ~0.8 h `mlp_fit_groups` battery so shard drift fails fast."""
    src = (REPO / "scripts" / "issue1775_doubly_mlp.py").read_text(encoding="utf-8")
    gate = src.index("_load_committed_doubly_units()")
    gate_assert = src.index("doubly delta_named validation failed")
    fit = src.index("res = mlp_fit_groups(")
    assert gate < fit and gate_assert < fit, (gate, gate_assert, fit)
