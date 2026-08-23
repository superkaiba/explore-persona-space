"""Regression pins for the #1901 `generic-boundary-token-control` r2 fixes.

Each test names the round-1 blocking finding it pins (fails pre-fix, passes
post-fix). External boundaries (Hub, model load, disk headroom) are faked with
signature-conformant fakes (`create_autospec` / mirrored defs) per
code-style.md § One production-body test per seam-stubbed function; every
seam-stubbed helper also gets a real-body test in this file or is exercised
by the driver-body tests below.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1901_boundary_token_control as BTC  # noqa: E402


def _prod_p1_args(out_root: Path):
    return BTC.build_argparser().parse_args(
        ["--phase", "p1_capture", "--out-root", str(out_root), "--device", "cpu"]
    )


def _prod_p2_args(out_root: Path, eval_out: Path):
    return BTC.build_argparser().parse_args(
        [
            "--phase",
            "p2_fits",
            "--out-root",
            str(out_root),
            "--eval-out",
            str(eval_out),
            "--device",
            "cpu",
        ]
    )


def _mk_manifest(man_dir: Path, args, n_rows: int = 4):
    """Minimal LOCAL b0 manifest fixture satisfying `_validate_manifest`."""
    man_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"window_ids": ["art000"], "input_ids": [torch.arange(128, dtype=torch.int32)]},
        man_dir / "articles_shard000.pt",
    )
    rows = []
    for k in range(n_rows):
        split = ["train", "train", "test", "val"][k % 4]
        rows.append(
            {
                "row_id": f"art000:r{k}",
                "article_id": "art000",
                "sep_char": ".",
                "anchor_pos": 99,
                "c_span": [0, 16],
                "t_span": [100, 110],
                "split": split,
                "train_order": k if split == "train" else None,
                "pooled_order": k,
                "pooled_split": "train",
            }
        )
    with (man_dir / "manifest_shard000.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta = {
        "regime_key": BTC._b0_regime_key(args),
        "yield_gate": {"pass": True},
        "n_manifest_rows": len(rows),
        "manifest_shards": ["manifest_shard000.jsonl"],
        "article_shards": ["articles_shard000.pt"],
        "n_files_total": 3,
        "common_rungs": [50],
        "top_common_rung": 50,
    }
    (man_dir / "meta.json").write_text(json.dumps(meta))
    return rows, meta


# ── fix 4 (codex p1-phase-idempotency): p1 fast-skip never stages / loads ──────


def test_p1_state_fast_skip_never_loads_model(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    args = _prod_p1_args(root)
    regime = BTC._p1_regime(args, BTC.PROD_PERSIST_LAYERS)
    (root / "p1_state.json").write_text(
        json.dumps({"regime": regime, "shard_files": ["pairs_shard000.pt"], "n_shards": 1})
    )
    monkeypatch.setattr(
        BTC.hub,
        "verify_repo_paths_uploaded",
        create_autospec(BTC.hub.verify_repo_paths_uploaded, return_value=[]),
    )
    monkeypatch.setattr(
        BTC.ES,
        "load_model",
        create_autospec(BTC.ES.load_model, side_effect=AssertionError("model must NOT load")),
    )
    monkeypatch.setattr(
        BTC,
        "_manifest_dir",
        create_autospec(BTC._manifest_dir, side_effect=AssertionError("manifest must NOT stage")),
    )
    assert BTC.phase_p1_capture(args) == 0


# ── fix 5 (codex manifest-contract-post-init): pre-model validation ───────────


def test_validate_manifest_smoke_parity_refusal(tmp_path):
    smoke_args = BTC.build_argparser().parse_args(
        ["--phase", "p1_capture", "--smoke", "--out-root", str(tmp_path)]
    )
    rows, meta = _mk_manifest(tmp_path / "m", smoke_args)
    prod_args = _prod_p1_args(tmp_path)
    ids_by_art = {"art000": list(range(128))}
    with pytest.raises(AssertionError, match="smoke"):
        BTC._validate_manifest(rows, ids_by_art, meta, prod_args, phase="p1")


# ── fix 6 (codex production-layer-fallback): p2 production layer guard ────────


def test_p2_production_layer_guard(tmp_path, monkeypatch):
    root = tmp_path / "root"
    args = _prod_p2_args(root, tmp_path / "eval")
    _mk_manifest(root / "manifest", args)
    store = root / "store"
    store.mkdir(parents=True)
    (store / "pairs_shard000.json").write_text(
        json.dumps({"layers": [0, 2, 4, 5], "n_rows": 0, "shard_index": 0, "row_ids": []})
    )
    (store / "pairs_shard000.pt").touch()
    monkeypatch.setattr(
        BTC, "assert_out_root_headroom", create_autospec(BTC.assert_out_root_headroom)
    )
    with pytest.raises(AssertionError, match="production store persists"):
        BTC.phase_p2_fits(args)


# ── fix 9 (codex p2-shared-gram-missing): shared-Gram parity, REAL bodies ─────


def test_gram_parity_gate_real_body_cpu():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((240, 16)).astype(np.float32)
    w = rng.standard_normal((16, 8)).astype(np.float32)
    y = x @ w + 0.01 * rng.standard_normal((240, 8)).astype(np.float32)
    X = torch.tensor(x)
    Y = torch.tensor(y)
    pool = np.arange(160, dtype=np.int64)
    val = np.arange(160, 200, dtype=np.int64)
    te = np.arange(200, 240, dtype=np.int64)
    row = BTC._gram_parity_gate(X, Y, pool, val, te, torch.device("cpu"), 4096)
    assert row["pass"] is True
    assert row["abs_diff"] <= 1e-6
    assert row["n"] == 50


# ── fix 8 (codex p2-checkpoint-regime-key + g2 F7): unit key regime knobs ─────


def test_cell_unit_key_covers_regime_knobs(tmp_path):
    base = _prod_p2_args(tmp_path, tmp_path / "e")
    varied = BTC.build_argparser().parse_args(
        [
            "--phase",
            "p2_fits",
            "--out-root",
            str(tmp_path),
            "--eval-out",
            str(tmp_path / "e"),
            "--device",
            "cpu",
            "--n-article-boot",
            "7",
        ]
    )
    tr = np.arange(10, dtype=np.int64)
    fp = {"store_rows_sha256": "x", "manifest_regime_sha256": "y"}
    k_base = BTC._cell_unit_key(".", 19, 50, "prefix", tr, base, fp)
    k_same = BTC._cell_unit_key(".", 19, 50, "prefix", tr, base, fp)
    k_boot = BTC._cell_unit_key(".", 19, 50, "prefix", tr, varied, fp)
    assert k_base == k_same
    assert k_base != k_boot  # pre-fix: n_article_boot absent -> keys collide
    gpu = _prod_p2_args(tmp_path, tmp_path / "e")
    gpu.device = "cuda"
    assert BTC._cell_unit_key(".", 19, 50, "prefix", tr, gpu, fp) != k_base
    fp2 = {"store_rows_sha256": "OTHER", "manifest_regime_sha256": "y"}
    assert BTC._cell_unit_key(".", 19, 50, "prefix", tr, base, fp2) != k_base


# ── fix 2 (g2 F4): str-SystemExit usage message, no ValueError traceback ──────


def test_usage_message_exits_rc1_without_valueerror():
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
    }
    proc = subprocess.run(
        [sys.executable, "scripts/issue1901_boundary_token_control.py"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 1, (proc.returncode, proc.stderr[-500:])
    assert "--phase is required" in proc.stderr
    assert "ValueError" not in proc.stderr  # pre-fix: int(str) raised ValueError
    assert "Traceback" not in proc.stderr


# ── fix 1 (g2 F1): prefix draw over an undersized pool refuses loud ───────────


def test_draw_indices_prefix_overrun_refuses():
    with pytest.raises(AssertionError, match="prefix draw"):
        BTC._draw_indices(np.arange(10, dtype=np.int64), 50, "prefix")


# ── fix 3 (g2 F3 == codex p1-upload-resume-gap): entry upload-reconcile ───────


def test_p1_upload_reconcile_reuploads_stranded_shards(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    args = _prod_p1_args(root)
    rows, _meta = _mk_manifest(root / "manifest", args)
    store = root / "store"
    store.mkdir()
    (store / "pairs_shard000.json").write_text(
        json.dumps(
            {
                "layers": list(BTC.PROD_PERSIST_LAYERS),
                "n_rows": len(rows),
                "shard_index": 0,
                "row_ids": [r["row_id"] for r in rows],
            }
        )
    )
    (store / "pairs_shard000.pt").touch()
    stranded = [f"{BTC.CAPTURE_PREFIX}/pairs_shard000.pt"]
    fake_verify = create_autospec(BTC.hub.verify_repo_paths_uploaded, side_effect=[stranded, []])
    monkeypatch.setattr(BTC.hub, "verify_repo_paths_uploaded", fake_verify)
    fake_upload = create_autospec(BTC._upload_shard_batch, return_value=1.0)
    monkeypatch.setattr(BTC, "_upload_shard_batch", fake_upload)
    monkeypatch.setattr(
        BTC.ES,
        "load_model",
        create_autospec(BTC.ES.load_model, side_effect=AssertionError("model must NOT load")),
    )
    assert BTC.phase_p1_capture(args) == 0
    assert fake_upload.call_count == 1
    names = fake_upload.call_args.args[1]
    assert names == ["pairs_shard000.pt", "pairs_shard000.json"]
    assert fake_verify.call_count == 2  # entry reconcile + terminal verify
    st = json.loads((root / "p1_state.json").read_text())
    assert st["regime"] == BTC._p1_regime(args, BTC.PROD_PERSIST_LAYERS)
    assert st["shard_files"] == ["pairs_shard000.pt"]


def test_upload_shard_batch_real_body(tmp_path, monkeypatch):
    """Real `_upload_shard_batch` body (hardlink staging + cleanup); only the
    Hub boundary (`hub._upload`) is autospec-faked."""
    store = tmp_path / "store"
    store.mkdir()
    for n in ("pairs_shard000.pt", "pairs_shard000.json"):
        (store / n).write_text("x")
    seen = {}

    def _spy(local_path, repo_id, repo_type, **kw):
        seen["staged"] = sorted(p.name for p in Path(local_path).iterdir())
        return "https://hf.co/fake"

    fake = create_autospec(BTC.hub._upload, side_effect=_spy)
    monkeypatch.setattr(BTC.hub, "_upload", fake)
    scratch = tmp_path / "scratch"
    wall = BTC._upload_shard_batch(store, ["pairs_shard000.pt", "pairs_shard000.json"], scratch)
    assert isinstance(wall, float)
    assert seen["staged"] == ["pairs_shard000.json", "pairs_shard000.pt"]
    assert not scratch.exists()  # staging cleaned up after upload
    kw = fake.call_args.kwargs
    assert kw["path_in_repo"] == BTC.CAPTURE_PREFIX
    assert kw["raise_on_error"] is True


# ── fix 1 (g2 F1): p2 partial-store completeness assert ───────────────────────


def test_p2_partial_store_refusal(tmp_path, monkeypatch):
    root = tmp_path / "root"
    args = _prod_p2_args(root, tmp_path / "eval")
    rows, _meta = _mk_manifest(root / "manifest", args)
    store = root / "store"
    store.mkdir(parents=True)
    n_cap, hidden = 2, 8
    torch.save(
        {
            "arrays": {
                "x_sep": torch.randn(n_cap, 4, hidden),
                "y": torch.randn(n_cap, 4, hidden),
            },
            "row_ids": [rows[0]["row_id"], rows[1]["row_id"]],
            "group_ids": ["art000", "art000"],
        },
        store / "pairs_shard000.pt",
    )
    (store / "pairs_shard000.json").write_text(
        json.dumps(
            {
                "layers": list(BTC.PROD_PERSIST_LAYERS),
                "n_rows": n_cap,
                "shard_index": 0,
                "row_ids": [rows[0]["row_id"], rows[1]["row_id"]],
            }
        )
    )
    monkeypatch.setattr(
        BTC, "assert_out_root_headroom", create_autospec(BTC.assert_out_root_headroom)
    )
    with pytest.raises(AssertionError, match="partial capture store"):
        BTC.phase_p2_fits(args)


# ── fix 6 (poster leg): smoke-artifact rejection in the poster wrapper ────────


def _load_poster_module():
    path = PROJECT_ROOT / "docs" / "posters" / "mats_2026" / "make_plot1_scaling.py"
    spec = importlib.util.spec_from_file_location("i1901_poster_plot1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_poster_rejects_smoke_boundary_artifact():
    mod = _load_poster_module()
    with pytest.raises(RuntimeError, match="smoke"):
        mod._assert_production_artifact({"smoke": True, "cells": []})
    assert mod._assert_production_artifact({"cells": []}) is None
