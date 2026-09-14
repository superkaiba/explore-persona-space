"""Read four immutable saved prediction stores. No refits or model execution."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/eps-data/thomasjiralerspong/workspace-jr-20260912")
OUT = ROOT / "user_requested_critic_v1"
entries = json.loads((ROOT / "user_requested_critic_parent_v1/download_verified.json").read_text())
ids = json.loads((ROOT / "main_comparison_v1/cross_model/comparisons.json").read_text())[
    "context_ids"
]
draws = np.random.default_rng(20260912).integers(0, len(ids), size=(2000, len(ids)))
checked_draws = [0, 127, 128, 999, 1999]


def r2(y, p):
    centered = y - y.mean(axis=0)
    return 1.0 - ((y - p) ** 2).sum() / (centered**2).sum()


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)


records = {}
fulls = {}
metric_count = 0
bootstrap_checks = 0
for entry in entries:
    path = Path(entry["local_path"])
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    assert digest == entry["sha256"]
    cell = entry["cell"]
    folder = ROOT / "main_comparison_v1/cross_model" / cell.replace("/", "__")
    summary = json.loads((folder / "summary.json").read_text())["summary"]
    with (
        np.load(path, allow_pickle=False) as arrays,
        np.load(folder / "bootstrap_samples.npz", allow_pickle=False) as bootstrap,
    ):
        original_ids = arrays["context_ids"].tolist()
        assert len(set(original_ids)) == len(original_ids) == 256
        index = np.asarray([original_ids.index(i) for i in ids])
        y = {
            name: arrays[f"target__{name}"].astype(np.float64)
            for name in ("full", "J", "restJ", "R", "restR")
        }
        for arm in ("J", "R"):
            close(y["full"], y[arm] + y[f"rest{arm}"])
        points = {}
        for predictor in ("ridge", "mlp", "mlp_seed42", "mlp_seed137", "mlp_seed271"):
            for name, target in y.items():
                prediction = arrays[f"prediction__{predictor}__{name}"].astype(np.float64)
                point = float(r2(target[index], prediction[index]))
                close(point, summary[f"{predictor}/{name}"]["estimate"])
                metric_count += 1
                for b in checked_draws:
                    ii = index[draws[b]]
                    close(r2(target[ii], prediction[ii]), bootstrap[f"{predictor}/{name}"][b])
                    bootstrap_checks += 1
                points[f"{predictor}/{name}"] = point
        all_captured = {n: float(r2(t, arrays[f"prediction__ridge__{n}"])) for n, t in y.items()}
        record = {
            "source_sha256": digest,
            "source_test_contexts": 256,
            "primary_contexts": 252,
            "points": points,
            "all_captured_256_ridge_r2": all_captured,
            "primary_gaps": {
                a: points[f"ridge/rest{a}"] - points[f"ridge/{a}"] for a in ("J", "R")
            },
            "all_captured_256_gaps": {
                a: all_captured[f"rest{a}"] - all_captured[a] for a in ("J", "R")
            },
            "all_captured_scope": (
                "Descriptive cap-inclusive sensitivity; not the primary estimand."
            ),
        }
        for a in ("J", "R"):
            yc = y[a][index] - y[a][index].mean(0)
            rest = y[f"rest{a}"][index] - y[f"rest{a}"][index].mean(0)
            full = y["full"][index] - y["full"][index].mean(0)
            close((full**2).sum(), (yc**2).sum() + (rest**2).sum() + 2 * (yc * rest).sum())
        records[cell] = record
        role, kind, *_ = cell.split("/")
        fulls[(role, kind)] = {
            "ids": original_ids,
            "x": arrays["x"],
            "full": y["full"],
            "ridge_prediction": arrays["prediction__ridge__full"],
        }

for role in ("primary", "comparison"):
    obs, null = fulls[(role, "observed")], fulls[(role, "affine_null")]
    assert obs["ids"] == null["ids"] and np.array_equal(obs["x"], null["x"])
    close(obs["ridge_prediction"], null["full"])

result = {
    "status": "passed",
    "native_cells_checked": 4,
    "per_example_metric_recomputations": metric_count,
    "direct_existing_bootstrap_draw_checks": bootstrap_checks,
    "existing_draw_indices": checked_draws,
    "native_null_inputs_match_observed": True,
    "native_null_targets_match_observed_full_ridge_predictions": True,
    "cells": records,
    "not_run": [
        "new fits",
        "new bootstrap sampling",
        "generation",
        "token capture",
        "lens construction",
        "full token-level sparse decomposition",
    ],
}
(OUT / "native_array_checks.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({k: v for k, v in result.items() if k not in ("cells",)}, indent=2))
for cell, r in records.items():
    print(cell, "primary", r["primary_gaps"], "cap-inclusive", r["all_captured_256_gaps"])
