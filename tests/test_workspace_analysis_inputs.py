"""Post-fit consumers require persisted predictions and retain live loss telemetry."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from explore_persona_space.analysis.workspace_analysis_inputs import (
    _saved_full_predictions,
    load_fitted_components,
    summarize_token_statistic,
)
from explore_persona_space.analysis.workspace_fit import _fit_contract, _input_fingerprints
from explore_persona_space.analysis.workspace_runtime import content_sha256, file_sha256


def test_unequal_rollouts_keep_both_weighting_conventions():
    result = summarize_token_statistic(torch.tensor([2.0, 8.0, 8.0, 8.0]), [1, 3])
    assert result["mean"] == result["mean_token"] == 6.5
    assert result["mean_equal_rollout"] == 5.0
    assert result["rollout_means"] == [2.0, 8.0]
    assert result["tokens"] == 4


@pytest.mark.parametrize(
    "values,lengths",
    [
        (torch.ones(4), [1, 2]),
        (torch.ones(4), [0, 4]),
        (torch.ones(2, 2), [2, 2]),
        (torch.tensor([float("nan")]), [1]),
    ],
)
def test_invalid_rollout_statistics_fail(values, lengths):
    with pytest.raises(ValueError, match="exact nonempty rollout lengths"):
        summarize_token_statistic(values, lengths)


@pytest.fixture
def completed_fit(tmp_path):
    """Construct six synthetic contexts with real coverage, canonical and upload checks."""
    root = tmp_path / "run"
    fit = root / "fits/pilot/k10-rotationNone"
    (fit / "mlp").mkdir(parents=True)
    config = {"seed": 42, "fit": {"mlp_seeds": [42, 137, 271]}, "statistics": {}}
    config["generation"] = {"seeds": [42, 43]}
    identity = {
        "config_sha256": content_sha256(config),
        "selection_sha256": "synthetic_frozen_selection",
        "model_role": "primary",
        "versions": {"torch": "synthetic"},
        "code": {"git_commit": "a" * 40, "git_dirty": False},
    }

    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    dictionary = {"identity": identity, "pilot_only": True}
    write_json(root / "dictionaries/manifest.json", dictionary)
    contract = {"identity": identity, "dictionaries": dictionary, "k": 10, "rotation": None}
    x, targets, ids, coverage = {}, {}, {}, {}
    selection = {"subsets": {}}
    for split_index, split in enumerate(("train", "validation", "test")):
        subset = f"pilot_{split}"
        ids[split] = [f"synthetic_{split}_{i}" for i in range(2)]
        selection["subsets"][subset] = [{"prompt_sha256": value} for value in ids[split]]
        x[split] = np.arange(6, dtype=np.float32).reshape(2, 3) + split_index
        full = x[split].astype(np.float64) + 2
        targets[split] = {
            "full": full,
            "J": full * 0.25,
            "restJ": full * 0.75,
            "R": full * 0.75,
            "restR": full * 0.25,
        }
        canonical = root / "context_inputs" / subset / "batch-0000.pt"
        canonical.parent.mkdir(parents=True)
        generation_hashes = [content_sha256(["generation", context]) for context in ids[split]]
        torch.save(
            {
                "contract": {
                    "identity": identity,
                    "policy": "context_only_frozen_order_batches16_no_answer_tokens",
                    "source_hashes": generation_hashes,
                    "prompt_ids": ids[split],
                },
                "x": torch.from_numpy(x[split]),
            },
            canonical,
        )
        hashes = {}
        for index, context in enumerate(ids[split]):
            component = root / "components/k10-rotationNone" / subset / f"{context}.pt"
            component.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "contract": contract,
                    "contract_sha256": content_sha256(contract),
                    "prompt_sha256": context,
                    "rollout_seeds": config["generation"]["seeds"],
                    "x": torch.from_numpy(x[split][index]),
                    "targets": {
                        name: torch.from_numpy(values[index])
                        for name, values in targets[split].items()
                    },
                    "rollout_means": {
                        name: torch.from_numpy(np.stack([values[index], values[index]]))
                        for name, values in targets[split].items()
                    },
                    "token_counts": [1, 1],
                    "decomposition_statistics": {
                        arm: {"synthetic_statistic": torch.ones(2)} for arm in ("J", "R")
                    },
                    "context_input_reference": {
                        "context_input_file": str(canonical.relative_to(root)),
                        "context_input_file_sha256": file_sha256(canonical),
                        "context_input_row": index,
                        "generation_file_sha256": generation_hashes[index],
                    },
                },
                component,
            )
            hashes[context] = file_sha256(component)
        coverage[split] = {
            "identity": identity,
            "contract": contract,
            "status": "complete",
            "planned_contexts": 2,
            "included_prompt_sha256": ids[split],
            "exclusions": [],
            "file_sha256": hashes,
        }
    write_json(
        fit / "input_manifest.json",
        {"identity": identity, "contract": contract, "coverage": coverage},
    )
    write_json(
        fit / "results.json",
        {
            "fit_contract": _fit_contract(config),
            "input_fingerprints": _input_fingerprints(x, targets, ids),
            "split_counts": {split: len(values) for split, values in ids.items()},
        },
    )
    write_json(fit / "mlp/selection.json", {"synthetic_selection": True})
    arrays = {
        "context_ids": np.array(ids["test"]),
        "x": x["test"],
        **{f"target__{name}": values for name, values in targets["test"].items()},
        **{
            f"prediction__{name}__full": targets["test"]["full"] + 0.125
            for name in ("ridge", "mlp", "mlp_seed42", "mlp_seed137", "mlp_seed271")
        },
    }
    np.savez(fit / "per_example.npz", **arrays)
    uploaded = {
        str(path.relative_to(root)): file_sha256(path) for path in root.rglob("*") if path.is_file()
    }
    receipt = tmp_path / "upload.json"
    write_json(
        receipt,
        {
            "repo": "superkaiba1/explore-persona-space-data",
            "revision": "b" * 40,
            "prefix": "exploratory_workspace_jr/unit_fixture/completed_fit",
            "files_verified": len(uploaded),
            "verified_sha256": uploaded,
        },
    )

    def load():
        return load_fitted_components(root, "pilot", 10, None, config, selection, identity, receipt)

    return SimpleNamespace(
        root=root,
        fit=fit,
        receipt=receipt,
        arrays=arrays,
        x=x,
        targets=targets,
        ids=ids,
        seeds=config["fit"]["mlp_seeds"],
        load=load,
    )


def test_uploaded_completed_fit_loads_all_declared_seeds(completed_fit):
    x, targets, rollouts, ids, predictions, _, proof = completed_fit.load()
    assert set(predictions) == {"ridge", "mlp", "mlp_seed42", "mlp_seed137", "mlp_seed271"}
    assert ids == completed_fit.ids
    np.testing.assert_array_equal(x["test"], completed_fit.x["test"])
    np.testing.assert_array_equal(targets["test"]["full"], completed_fit.targets["test"]["full"])
    assert rollouts["train"]["full"].shape == (2, 2, 3)
    assert proof["upload_receipt_sha256"] == file_sha256(completed_fit.receipt)


def test_same_inputs_cannot_authorize_replaced_prediction_bytes(completed_fit):
    arrays = completed_fit.arrays.copy()
    arrays["prediction__ridge__full"] = np.full_like(arrays["prediction__ridge__full"], 100.0)
    np.savez(completed_fit.fit / "per_example.npz", **arrays)
    with pytest.raises(ValueError, match=r"verified producer upload: .*per_example\.npz"):
        completed_fit.load()


@pytest.mark.parametrize("mutation", ["missing_seed", "wrong_shape", "nonfinite"])
def test_full_predictions_require_every_seed_and_valid_geometry(completed_fit, mutation):
    arrays = completed_fit.arrays.copy()
    key = "prediction__mlp_seed271__full"
    if mutation == "missing_seed":
        arrays.pop(key)
    elif mutation == "wrong_shape":
        arrays[key] = arrays[key][:1]
    else:
        arrays[key] = np.full_like(arrays[key], np.nan)
    np.savez(completed_fit.fit / "per_example.npz", **arrays)
    with pytest.raises(ValueError, match=r"every declared MLP seed|invalid arrays"):
        _saved_full_predictions(
            completed_fit.fit,
            completed_fit.x,
            completed_fit.targets,
            completed_fit.ids,
            completed_fit.seeds,
        )


@pytest.mark.parametrize("source", ["component", "canonical"])
def test_consumer_requires_durable_component_and_canonical_files(completed_fit, source):
    receipt = json.loads(completed_fit.receipt.read_text())
    prefix = "components/" if source == "component" else "context_inputs/"
    key = next(key for key in receipt["verified_sha256"] if key.startswith(prefix))
    del receipt["verified_sha256"][key]
    receipt["files_verified"] -= 1
    completed_fit.receipt.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="verified producer upload"):
        completed_fit.load()


def test_learning_curve_telemetry_preserves_actual_per_target_losses(monkeypatch):
    from explore_persona_space.orchestrate import env

    # Import the CLI without reading credentials or starting its main function.
    monkeypatch.setattr(env, "load_dotenv", lambda: None)
    path = Path(__file__).resolve().parents[1] / "scripts/workspace_jr_analyze.py"
    training_scalars = runpy.run_path(str(path))["training_scalars"]
    record = {
        "hidden": 256,
        "learning_rate": 0.001,
        "seed": 137,
        "epoch": 4,
        "train_fraction": 0.25,
        "train_contexts": 16,
        "keys": [("full",), ("J",)],
        "train_loss": [1.25, 0.75],
        "validation_loss": [1.5, 0.8],
        "stopped": [False, True],
    }
    scalars = training_scalars(record)
    for index, target in enumerate(("full", "J")):
        assert scalars[f"{target}/train_scaled_mse_before_step"] == record["train_loss"][index]
        assert (
            scalars[f"{target}/validation_scaled_mse_after_step"]
            == record["validation_loss"][index]
        )
        assert scalars[f"{target}/stopped"] == int(record["stopped"][index])
    assert scalars["train_fraction"] == 0.25
    assert scalars["seed"] == 137
    assert "keys" not in scalars
    assert all(isinstance(value, (int, float)) for value in scalars.values())
