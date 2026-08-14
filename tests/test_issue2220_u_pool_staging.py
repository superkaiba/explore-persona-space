"""Regression test for the #2220 chunk-1 crash: ``_load_u_pool`` must STAGE the
#1092 whitening U pool before reading it (epm:failure v1 — FileNotFoundError
from ``store_io._resolve_summary_kind`` on a fresh clone: nothing staged
``data/issue_2220/u_store``).

Covers, with fakes ONLY at the network/filesystem boundary (``create_autospec``
on the real ``store_io`` functions — signature-conformant by construction; the
#906 seam-stub rule):

1. stage-before-load ordering (pre-fix: ``stage_u_store`` never called → FAIL);
2. layout parity: the stage ``dest`` equals the exact root ``load_summaries``
   reads (``local_dir / U_STORE_CELL`` — ``stage_u_store`` FLATTENS the cell's
   shards into ``dest`` while the loader prefixes ``cell=U_STORE_CELL``), and
   both calls request the same (kinds x layers) grid;
3. the REAL ``_load_u_pool`` body end-to-end: real ``fit_pool_mask`` over
   real-shaped meta rows (``is_eval_only`` exclusion applied), real
   ``np.stack`` → ``(Ly, n_fit, d)`` float64 per kind.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import numpy as np

import scripts.issue2220_readwrite as rw
from explore_persona_space.experiments.issue_1739 import store_io

LAYERS = (10, 14)
KINDS = ("context_end", "prefix_end")
N_ROWS, N_EVAL_ONLY, DIM = 6, 2, 8


def _fake_summaries() -> dict[tuple[str, int], np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        (k, ly): rng.standard_normal((N_ROWS, DIM)).astype(np.float16)
        for k in KINDS
        for ly in LAYERS
    }


def _fake_meta() -> list[dict]:
    # Real-shaped meta rows: fit_pool_mask (run REAL below) keys on
    # is_eval_only / stratum, exactly as the corpus manifest rows carry them.
    return [
        {"context_id": f"c{i}", "is_eval_only": i < N_EVAL_ONLY, "stratum": "corpus"}
        for i in range(N_ROWS)
    ]


def test_load_u_pool_stages_before_reading(tmp_path):
    args = argparse.Namespace(u_store_dir=str(tmp_path / "u_store"))

    stage_mock = mock.create_autospec(store_io.stage_u_store, spec_set=True)
    load_mock = mock.create_autospec(store_io.load_summaries, spec_set=True)
    load_mock.return_value = (_fake_summaries(), _fake_meta())
    parent = mock.Mock()
    parent.attach_mock(stage_mock, "stage_u_store")
    parent.attach_mock(load_mock, "load_summaries")

    with (
        mock.patch.object(store_io, "stage_u_store", stage_mock),
        mock.patch.object(store_io, "load_summaries", load_mock),
    ):
        out = rw._load_u_pool(list(LAYERS), args)

    # (1) Stage-before-load ordering — the crash-fix invariant. Pre-fix,
    # stage_u_store was never called and this fails.
    names = [name for name, _a, _k in parent.mock_calls]
    assert "stage_u_store" in names, "fix not engaged: stage_u_store never called"
    assert names.index("stage_u_store") < names.index("load_summaries")

    # (2) Layout parity: stage dest == the exact root the loader reads
    # (stage_u_store flattens into dest; the loader reads local_dir / cell).
    stage_mock.assert_called_once()
    _, stage_kwargs = stage_mock.call_args
    load_args, load_kwargs = load_mock.call_args
    loader_root = Path(load_args[0]) / load_kwargs["cell"]
    assert Path(stage_kwargs["dest"]) == loader_root
    assert loader_root == Path(args.u_store_dir) / rw.U_STORE_CELL
    assert stage_kwargs["layers"] == tuple(LAYERS) == load_kwargs["layers"]
    assert stage_kwargs["kinds"] == load_kwargs["kinds"] == KINDS

    # (3) Real-body evidence: the REAL fit_pool_mask dropped the eval-only rows
    # and the per-kind stacks come back (Ly, n_fit, d) float64.
    n_fit = N_ROWS - N_EVAL_ONLY
    for k in KINDS:
        assert out[k].shape == (len(LAYERS), n_fit, DIM), out[k].shape
        assert out[k].dtype == np.float64
