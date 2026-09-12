"""End-to-end fixed-split fitter checks on exactly affine targets."""

import json

import numpy as np

from explore_persona_space.analysis.workspace_fit import evaluate_component_fits


def test_component_fits_save_real_predictions_and_select_without_test_labels(tmp_path):
    """Ridge, actual batched MLP, bootstrap and persistence share one checked split."""
    rng = np.random.default_rng(7)
    x = {s: rng.normal(size=(n, 6)) for s, n in (("train", 64), ("validation", 16), ("test", 24))}
    ids = {s: [f"{s}-{i}" for i in range(len(a))] for s, a in x.items()}
    targets = {
        s: {"full": a * 5 - 1, "J": a * 2 + 1, "restJ": a * 3 - 2, "R": a * 4 - 1, "restR": a}
        for s, a in x.items()
    }
    config = {
        "seed": 7,
        "fit": {
            "ridge_alpha_grid": [0.0001, 0.01],
            "mlp_hidden": [8],
            "mlp_learning_rates": [0.001],
            "mlp_weight_decay": 0.0001,
            "mlp_max_epochs": 3,
            "mlp_patience": 2,
            "mlp_seeds": [42, 137, 271],
            "retrieval": {"k": [1, 5], "metrics": ["euclidean", "cosine"]},
        },
        "statistics": {
            "bootstrap": {"draws": 32, "confidence": 0.95},
            "smallest_practical_gap": 0.05,
            "near_zero_component_norm_floor": "1e-6_times_training_median_full_target_norm",
        },
    }
    records = []
    first = evaluate_component_fits(
        x,
        targets,
        ids,
        config,
        tmp_path / "first",
        mlp_device="cpu",
        training_logger=records.append,
    )
    assert records and {r["seed"] for r in records} == {42, 137, 271}
    assert all(
        np.isfinite(r["train_loss"]).all() and np.isfinite(r["validation_loss"]).all()
        for r in records
    )
    for name in targets["test"]:
        assert first["metrics"]["ridge"][name]["r2"] > 0.999
    original_selection = json.loads((tmp_path / "first/mlp/selection.json").read_text())
    targets["test"] = {name: value + 100 for name, value in targets["test"].items()}
    targets["test"]["full"] += 100
    evaluate_component_fits(x, targets, ids, config, tmp_path / "changed_test", mlp_device="cpu")
    changed_selection = json.loads((tmp_path / "changed_test/mlp/selection.json").read_text())
    assert original_selection == changed_selection
    with np.load(tmp_path / "first/per_example.npz", allow_pickle=False) as saved:
        assert saved["prediction__ridge__J"].shape == (24, 6)
        assert saved["context_ids"].tolist() == ids["test"]
        assert np.isfinite(saved["prediction__mlp_seed42__J"]).all()
