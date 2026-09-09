"""Production-path regression checks for the K=5 full-grid extension."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import create_autospec, patch

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "issue1902_k5_fits.py"
spec = importlib.util.spec_from_file_location("k5_fullgrid_test", SCRIPT)
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)


def config(tmp_path, **changes):
    args = dict(
        cmd="all",
        smoke=False,
        force=False,
        full_grid=True,
        draw_revision="a" * 40,
        reuse_root=[],
        target_layers=[31],
        stage_root=tmp_path / "store",
        k5_root=tmp_path / "new",
        out=tmp_path / "out",
        figures_dir=tmp_path / "figs",
    )
    args.update(changes)
    return K.Config(Namespace(**args))


def test_fullgrid_scope_and_pin(tmp_path):
    cfg = config(tmp_path)
    assert len(cfg.cells) == 16
    assert ("S", "D") in cfg.cells and ("D", "R") in cfg.cells
    assert K._draw_revision(45, cfg) == "a" * 40
    assert K._draw_revision(42, cfg) == K.LC.HF_REVISION
    assert config(tmp_path, full_grid=False).cells == K.CELLS7
    with pytest.raises(ValueError, match="draw-revision"):
        config(tmp_path, draw_revision=None)
    with pytest.raises(ValueError, match="include 31"):
        config(tmp_path, target_layers=[18])
    with pytest.raises(RuntimeError, match="identity changed"):
        config(tmp_path, draw_revision="b" * 40)


def test_cached_source_hash_is_checked(tmp_path, monkeypatch):
    import hashlib

    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    cfg = config(tmp_path)
    data = b"cached source bytes"
    source = tmp_path / "source.json"
    source.write_bytes(data)
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    entry = RepoFile(path="source.json", size=len(data), oid=blob)
    fake = create_autospec(HfApi.get_paths_info, return_value=[entry])
    monkeypatch.setattr(HfApi, "get_paths_info", fake)
    K._check_source_identity(cfg, source, "source.json")
    assert fake.call_args.kwargs["revision"] == K.LC.HF_REVISION
    source.write_bytes(b"broken source bytes")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        K._check_source_identity(cfg, source, "source.json")


def test_real_fullgrid_fits_baseline_and_retrieval(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.setattr(K, "_check_source_identity", create_autospec(K._check_source_identity))
    n, d = 48, 4
    rng = np.random.default_rng(912)
    x = rng.normal(size=(n, d)).astype(np.float32)
    ids = [f"r{i}" for i in range(n)]
    folds = np.arange(n) % 6
    for m in K.STAGES:
        context_path = cfg.ro_root / K.LC.HF_PREFIX / m / "ctx/single/L31.pt"
        context_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"u_last": torch.from_numpy(x), "row_ids": ids}, context_path)
        for s in K.STAGES:
            y = x + (K.STAGES.index(s) + 1) * 0.25
            cfg.target_path(m, s, 31).parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cfg.target_path(m, s, 31),
                w_bar=y,
                row_ids=np.asarray(ids),
                seeds=np.array([42, 45, 46, 47, 48]),
            )
            if m == s:
                anchor = cfg.ro_root / K._draw_answer_relpath(m, s, 42, 31)
                anchor.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"w": torch.from_numpy(y), "row_ids": ids}, anchor)
    monkeypatch.setattr(
        K, "_reference_rows", create_autospec(K._reference_rows, return_value=(folds, ids))
    )
    monkeypatch.setattr(
        K, "_anchor_reference", create_autospec(K._anchor_reference, return_value={})
    )
    # The real-data run separately enforces the historical parity gate; the
    # synthetic fixture validates the unchanged estimator and added companions.
    monkeypatch.setattr(K, "_parity_anchor_gate", create_autospec(K._parity_anchor_gate))
    K.run_grid(cfg)
    for m, s in cfg.cells:
        p = np.load(cfg.grid_path(m, s))
        assert len(p["ss_res"]) == n and np.isfinite(p["ss_res"]).all()
        assert p["n_train"].tolist() == [40] * 6
        report = json.loads(cfg.companion_path(m, s).read_text())
        assert report["identity_bias_r2"] == pytest.approx(1, abs=1e-12)
        for metric in ("cosine", "euclidean"):
            assert report[metric]["pool_sizes"] == [8] * 6
            assert report[metric]["chance_at_k"]["1"] == pytest.approx(1 / 8)
            assert report[metric]["acc_at_k"]["1"] == 1.0
    with patch.object(
        K.XF.SharedPrimalRidge,
        "__init__",
        autospec=True,
        side_effect=AssertionError("unexpected refit"),
    ):
        K.run_grid(cfg)
