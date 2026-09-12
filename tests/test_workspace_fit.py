"""End-to-end fixed-split fitter checks on exactly affine targets."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.analysis.workspace_fit import (
    evaluate_component_fits,
    evaluate_learning_curves,
)


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


def test_learning_curves_reuse_validation_recipes_and_bind_all_training_rows(tmp_path):
    """Real reduced-prefix fits preserve seeds, fixed recipes, and held-out examples."""
    rng = np.random.default_rng(19)
    x = {s: rng.normal(size=(n, 4)) for s, n in (("train", 32), ("validation", 8), ("test", 12))}
    ids = {s: [f"{s}-{i}" for i in range(len(a))] for s, a in x.items()}
    targets = {
        s: {"full": a * 5, "J": a * 2, "restJ": a * 3, "R": a * 4, "restR": a} for s, a in x.items()
    }
    config = {
        "seed": 19,
        "fit": {
            "ridge_alpha_grid": [0.0001, 1],
            "mlp_hidden": [4, 8],
            "mlp_learning_rates": [0.001, 0.0003],
            "mlp_weight_decay": 0.0001,
            "mlp_max_epochs": 2,
            "mlp_patience": 2,
            "mlp_seeds": [42, 137, 271],
            "retrieval": {"k": [1], "metrics": ["euclidean"]},
            "learning_curve_train_fractions": [0.25, 0.5, 1.0],
        },
        "statistics": {
            "bootstrap": {"draws": 16, "confidence": 0.95},
            "smallest_practical_gap": 0.05,
            "near_zero_component_norm_floor": "1e-6_times_training_median_full_target_norm",
        },
    }
    full = tmp_path / "full"
    evaluate_component_fits(x, targets, ids, config, full, mlp_device="cpu")
    recipes = json.loads((full / "mlp/selection.json").read_text())["selected"]
    records = []
    result = evaluate_learning_curves(
        x,
        targets,
        ids,
        config,
        tmp_path / "curves",
        full,
        device="cpu",
        training_logger=records.append,
    )
    assert [cell["n_train"] for cell in result["cells"]] == [8, 16, 32]
    assert [cell["reused_full_fit"] for cell in result["cells"]] == [False, False, True]
    assert {record["train_contexts"] for record in records} == {8, 16}
    assert {record["seed"] for record in records} == {42, 137, 271}
    for cell in result["cells"][:-1]:
        selected = json.loads((Path(cell["fit_path"]) / "mlp/selection.json").read_text())
        assert len(selected["candidates"]) == 5
        assert selected["recipe_mode"] == "fixed_from_full_training_validation"
        for name, recipe in selected["selected"].items():
            assert (recipe["hidden"], recipe["lr"]) == (
                recipes[name]["hidden"],
                recipes[name]["lr"],
            )
        with np.load(Path(cell["fit_path"]) / "per_example.npz") as saved:
            assert saved["context_ids"].tolist() == ids["test"]
    changed_config = deepcopy(config)
    changed_config["fit"]["mlp_seeds"] = [42]
    with pytest.raises(ValueError, match="exact fit contract"):
        evaluate_learning_curves(
            x, targets, ids, changed_config, tmp_path / "bad_config", full, device="cpu"
        )
    x["train"][0, 0] += 1
    with pytest.raises(ValueError, match="identical input bytes"):
        evaluate_learning_curves(x, targets, ids, config, tmp_path / "bad", full, device="cpu")
