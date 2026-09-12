"""Regression: off-diagonal columns must retain target-checkpoint representations."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/issue1902_fixed_target_fits.py"
SPEC = importlib.util.spec_from_file_location("fixed_targets_under_test", SCRIPT)
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


def test_fixed_target_columns_and_resume(tmp_path, monkeypatch):
    """Recover all independent fits and refuse any refitting on a completed resume."""
    rng = np.random.default_rng(20260912)
    n, d = 120, 4
    latent = rng.normal(size=(n, d))
    folds = np.arange(n) % 6
    ids = np.array([f"row{i}" for i in range(n)])
    xs = {
        s: (latent @ rng.normal(size=(d, d)) + 0.2 * rng.normal(size=(n, d))).astype(np.float32)
        for s in F.STAGES
    }
    ys = {
        s: (latent @ rng.normal(size=(d, d)) + 0.3 * rng.normal(size=(n, d)) + 10 * j).astype(
            np.float32
        )
        for j, s in enumerate(F.STAGES)
    }
    expected = {}
    for s in F.STAGES:
        for t in F.STAGES:
            res, tot = [], []
            for fold in range(6):
                tr, ev = folds != fold, folds == fold
                pred, _ = F.XF.SharedPrimalRidge(xs[s][tr]).fit_predict(ys[t][tr], xs[s][ev])
                r, v, _ = F.LC._per_row_components(pred, ys[t][ev], ys[t][tr].mean(axis=0))
                res.extend(r)
                tot.extend(v)
            expected[s + t] = 1 - np.sum(res) / np.sum(tot)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    files = {}
    for s in F.STAGES:
        path = inputs / f"{s}.npz"
        np.savez(path, x=xs[s], y=ys[s], row_ids=ids, fold_of=folds)
        files[path.name] = {"sha256": F.sha256(path), "bytes": path.stat().st_size}
    F.write_json(
        inputs / "manifest.json",
        {
            "files": files,
            "n_rows": n,
            "diagonal_reference": {s: expected[s + s] for s in F.STAGES},
        },
    )
    args = argparse.Namespace(inputs=inputs, out=tmp_path / "results", input_revision=None)
    F.run(args)
    summary = json.loads((args.out / "summary.json").read_text())
    assert set(summary["cells"]) == set(expected)
    for key, expected_r2 in expected.items():
        assert abs(summary["cells"][key]["r2"] - expected_r2) < 1e-12
    assert summary["fixed_target_comparisons"]["B"]["paired_row_ci95"] == [0.0, 0.0]

    def no_refit(*args, **kwargs):
        raise AssertionError("Completed folds must not be refit")

    monkeypatch.setattr(F.XF, "SharedPrimalRidge", no_refit)
    F.run(args)
