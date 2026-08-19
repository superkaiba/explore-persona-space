"""Round-2 pins for the #2163 context-read driver (scripts/issue2163_ctxread.py).

1. Blockwise-vs-whole equivalence of the confirm-B validation lambda sweep. The round-1
   Critical passed `y_val` FULL-WIDTH into ``DSF._val_block_ss`` — which slices ALL THREE of
   its data args internally by its (c0, c1) window — with a pre-sliced ``rot_b``/``ymu`` and
   an identity window, so blocks 1..N-1 were scored against block 0's targets and block 0's
   SST was N-x counted. ``_confirm_b_val_block`` now slices all three args by the SAME window
   exactly once; the equivalence test fails on the pre-fix call shape and passes post-fix,
   and the divergence test pins that the pre-fix shape is actually distinguishable (the
   double-slicing class cannot regress silently again).
2. The GPU-cell upload set covers the plan-declared 16,384-panel B sub-block: the
   venue-switch pod never runs phase_upload_verify, so ``phase_confirm_b_gpu`` must call the
   shared ``_upload_panel_block`` itself (round-1 Issue: gpu-cell-panel-block-not-uploaded).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_densesae_fullwidth as DSF  # noqa: E402
import issue2163_ctxread as M  # noqa: E402


def _tiny_sweep_fixture(n_va: int = 20, d: int = 6, dict_size: int = 8, seed: int = 0):
    """Deterministic tiny (y_val, ev, s_eig, B_full, ymu) fixture for the val sweep."""
    rng = np.random.default_rng(seed)
    y_val = torch.as_tensor(rng.normal(size=(n_va, dict_size)), dtype=torch.float64)
    ev = torch.as_tensor(rng.normal(size=(n_va, d)), dtype=torch.float64)
    s_eig = torch.as_tensor(np.abs(rng.normal(size=d)) + 0.1, dtype=torch.float64)
    b_full = rng.normal(size=(d, dict_size))
    ymu = torch.as_tensor(rng.normal(size=dict_size), dtype=torch.float64)
    return y_val, ev, s_eig, b_full, ymu


def test_confirm_b_blockwise_val_sweep_matches_whole():
    """Blockwise accumulation over pre-sliced blocks == ONE whole-width _val_block_ss call."""
    y_val, ev, s_eig, b_full, ymu = _tiny_sweep_fixture()
    dict_size, block = 8, 4
    ssr = np.zeros(len(DSF.LAMBDAS))
    sst = 0.0
    for c0 in range(0, dict_size, block):
        c1 = min(c0 + block, dict_size)
        r, t = M._confirm_b_val_block(
            y_val, ev, b_full[:, c0:c1], ymu, s_eig, c0, c1, torch.device("cpu")
        )
        ssr += r
        sst += t
    r_all, t_all = DSF._val_block_ss(y_val, ev, torch.as_tensor(b_full), ymu, s_eig, 0, dict_size)
    np.testing.assert_allclose(ssr, r_all, rtol=1e-10, atol=1e-9)
    assert abs(sst - t_all) < 1e-9, (sst, t_all)


def test_confirm_b_pre_fix_call_shape_diverges():
    """The round-1 call shape (full-width y_val + identity window) is NOT equivalent.

    Guards that the equivalence test above is discriminating: on block 1, the pre-fix shape
    resolves yb to block 0's columns, so its (ssr, sst) must differ from the fixed helper's.
    """
    y_val, ev, s_eig, b_full, ymu = _tiny_sweep_fixture()
    c0, c1 = 4, 8  # block 1 of 2
    r_fixed, t_fixed = M._confirm_b_val_block(
        y_val, ev, b_full[:, c0:c1], ymu, s_eig, c0, c1, torch.device("cpu")
    )
    # Pre-fix shape: y_val FULL-WIDTH, rot/ymu pre-sliced, window (0, width) -> yb == block 0.
    r_old, t_old = DSF._val_block_ss(
        y_val, ev, torch.as_tensor(b_full[:, c0:c1]), ymu[c0:c1], s_eig, 0, c1 - c0
    )
    assert not np.allclose(np.asarray(r_old), np.asarray(r_fixed)), (
        "pre-fix call shape must be distinguishable from the fixed one"
    )
    assert abs(float(t_old) - float(t_fixed)) > 1e-9


def _panel_args(tmp_path: Path, smoke: bool = False) -> SimpleNamespace:
    """Minimal args namespace for the panel-upload helper."""
    return SimpleNamespace(work=str(tmp_path), smoke=smoke, hf_out_prefix=None, skip_upload=False)


def test_upload_panel_block_uploads_the_plan_declared_panel(tmp_path, monkeypatch):
    """_upload_panel_block puts the panel sub-block at the canonical B_panel repo path."""
    outd = tmp_path / "out"
    outd.mkdir(parents=True)
    np.save(outd / "B_panel_block.f32.npy", np.zeros((3, 2), dtype=np.float32))
    args = _panel_args(tmp_path)
    rp = f"{M.OUT_PREFIX}/analysis_tensors/B_panel/B_panel_block.f32.npy"
    fake_verify = mock.create_autospec(M.hub.verify_repo_paths_uploaded, return_value=[rp])
    fake_upload = mock.create_autospec(M.hub._upload, return_value="https://hf/ok")
    monkeypatch.setattr(M.hub, "verify_repo_paths_uploaded", fake_verify)
    monkeypatch.setattr(M.hub, "_upload", fake_upload)
    got = M._upload_panel_block(args)
    assert got == [rp]
    assert fake_upload.call_count == 1
    assert fake_upload.call_args.args[3] == rp  # path_in_repo


def test_upload_panel_block_fail_loud_on_no_path(tmp_path, monkeypatch):
    """A no-path upload return raises (never a warning-and-continue, upload-policy rule (b))."""
    outd = tmp_path / "out"
    outd.mkdir(parents=True)
    np.save(outd / "B_panel_block.f32.npy", np.zeros((3, 2), dtype=np.float32))
    args = _panel_args(tmp_path)
    fake_verify = mock.create_autospec(M.hub.verify_repo_paths_uploaded, return_value=["missing"])
    fake_upload = mock.create_autospec(M.hub._upload, return_value="")
    monkeypatch.setattr(M.hub, "verify_repo_paths_uploaded", fake_verify)
    monkeypatch.setattr(M.hub, "_upload", fake_upload)
    with pytest.raises(SystemExit):
        M._upload_panel_block(args)


def test_upload_panel_block_absent_is_noop(tmp_path, monkeypatch):
    """No panel file (e.g. pre-confirm-b re-run) -> no Hub traffic, empty upload set."""
    (tmp_path / "out").mkdir(parents=True)
    fake_upload = mock.create_autospec(M.hub._upload)
    monkeypatch.setattr(M.hub, "_upload", fake_upload)
    assert M._upload_panel_block(_panel_args(tmp_path)) == []
    fake_upload.assert_not_called()


def test_gpu_cell_upload_set_covers_panel_block():
    """The GPU-cell upload set must be a superset of {panel block} (round-1 Issue fix).

    The venue-switch pod terminates without ever running phase_upload_verify, so
    phase_confirm_b_gpu must route the plan-declared panel sub-block through the SAME shared
    helper the CPU path uses.
    """
    assert "_upload_panel_block" in inspect.getsource(M.phase_confirm_b_gpu)
    assert "_upload_panel_block" in inspect.getsource(M.phase_upload_verify)
    # And the CPU confirm-b path persists it at phase end too (per-phase persistence).
    assert "_upload_panel_block" in inspect.getsource(M.phase_confirm_b)
