"""Load only the exact completed component observations used by a saved fit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.workspace_artifacts import (
    validate_context_input,
    validate_coverage,
    validate_producer,
)
from explore_persona_space.analysis.workspace_fit import _fit_contract, _input_fingerprints
from explore_persona_space.analysis.workspace_runtime import content_sha256, file_sha256


def summarize_token_statistic(values: torch.Tensor, token_counts: list[int]) -> dict:
    """Retain token weighting and the experiment's equal weighting of rollouts."""
    if (
        not token_counts
        or any(type(count) is not int or count < 1 for count in token_counts)
        or values.ndim != 1
        or len(values) != sum(token_counts)
        or not torch.isfinite(values).all()
    ):
        raise ValueError(
            "Token statistics require finite values and exact nonempty rollout lengths"
        )
    values = values.double()
    rollout_means = torch.stack([part.mean() for part in values.split(token_counts)])
    return {
        "mean": float(values.mean()),
        "mean_token": float(values.mean()),
        "mean_equal_rollout": float(rollout_means.mean()),
        "rollout_means": rollout_means.tolist(),
        "sum": float(values.sum()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "tokens": len(values),
    }


def load_fitted_components(
    root: Path, stage: str, k: int, rotation, config, selection, identity, upload_receipt
):
    """Verify component files, canonical inputs, saved fit targets and all input fingerprints."""
    cell = f"k{k}-rotation{rotation}"
    fit = root / "fits" / stage / cell
    verified = _upload_binding(root, upload_receipt)
    for name in ("input_manifest.json", "results.json", "per_example.npz", "mlp/selection.json"):
        verified(fit / name)
    verified(root / "dictionaries/manifest.json")
    manifest = json.loads((fit / "input_manifest.json").read_text())
    producer = manifest["identity"]
    validate_producer(producer, identity, native_ancestor=True)
    result = json.loads((fit / "results.json").read_text())
    if result["fit_contract"] != _fit_contract(config):
        raise ValueError("Saved fit tuning contract differs from analysis")
    contract = manifest["contract"]
    dictionary = json.loads((root / "dictionaries/manifest.json").read_text())
    if contract != {"identity": producer, "dictionaries": dictionary, "k": k, "rotation": rotation}:
        raise ValueError("Analysis cell differs from the actual fitted component cell")
    x, targets, rollouts, ids, ledger, statistics = {}, {}, {}, {}, {}, {}
    for split in ("train", "validation", "test"):
        subset = f"{stage}_{split}"
        coverage = manifest["coverage"][split]
        expected = selection["subsets"][subset][: coverage["planned_contexts"]]
        allowed = [row["prompt_sha256"] for row in expected]
        included = validate_coverage(coverage, allowed, producer)
        ids[split] = [context for context in allowed if context in included]
        if coverage["contract"] != contract or result["split_counts"][split] != len(ids[split]):
            raise ValueError("Fit split coverage or contract differs")
        rows, ledger[split], statistics[split] = [], {}, []
        for context in ids[split]:
            path = root / "components" / cell / subset / f"{context}.pt"
            checksum = verified(path)
            if checksum != coverage["file_sha256"][context]:
                raise ValueError("A fitted component file changed")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if (
                saved["contract"] != contract
                or saved["contract_sha256"] != content_sha256(contract)
                or saved["prompt_sha256"] != context
                or saved["rollout_seeds"] != config["generation"]["seeds"]
            ):
                raise ValueError("Component metadata differs from the frozen fit inputs")
            validate_context_input(saved["context_input_reference"], saved["x"], root, producer)
            verified(root / saved["context_input_reference"]["context_input_file"])
            rows.append(saved)
            ledger[split][context] = checksum
            statistics[split].append(
                {
                    "context_id": context,
                    "token_counts": saved["token_counts"],
                    "arms": {
                        arm: {
                            name: summarize_token_statistic(values, saved["token_counts"])
                            for name, values in fields.items()
                        }
                        for arm, fields in saved["decomposition_statistics"].items()
                    },
                }
            )
        if len(rows) < 2:
            raise ValueError("Analysis requires at least two realized contexts per split")
        x[split] = np.stack([row["x"].numpy() for row in rows])
        targets[split] = {
            name: np.stack([row["targets"][name].numpy() for row in rows])
            for name in ("full", "J", "restJ", "R", "restR")
        }
        rollouts[split] = {
            name: np.stack([row["rollout_means"][name].numpy() for row in rows])
            for name in targets[split]
        }
    if result["input_fingerprints"] != _input_fingerprints(x, targets, ids):
        raise ValueError("Saved fit input fingerprints differ from actual component arrays")
    predictions = _saved_full_predictions(fit, x, targets, ids, config["fit"]["mlp_seeds"])
    proof = {
        "input_manifest_sha256": file_sha256(fit / "input_manifest.json"),
        "results_sha256": file_sha256(fit / "results.json"),
        "per_example_sha256": file_sha256(fit / "per_example.npz"),
        "component_files": ledger,
        "source_identity": producer,
        "input_fingerprints": result["input_fingerprints"],
        "upload_receipt_sha256": file_sha256(upload_receipt),
        "upload": json.loads(upload_receipt.read_text()),
    }
    return x, targets, rollouts, ids, predictions, statistics, proof


def _upload_binding(root, path):
    """Require persisted producer-side hashes, not a consumer's new hash alone."""
    upload = json.loads(path.read_text())
    if (
        upload["repo"] != "superkaiba1/explore-persona-space-data"
        or len(upload["revision"]) != 40
        or any(c not in "0123456789abcdef" for c in upload["revision"])
        or not upload["prefix"].startswith("exploratory_workspace_jr/")
        or upload["files_verified"] != len(upload["verified_sha256"])
    ):
        raise ValueError("Analysis requires a valid immutable producer upload receipt")

    def verified(source):
        relative = str(source.relative_to(root))
        checksum = file_sha256(source)
        if upload["verified_sha256"].get(relative) != checksum:
            raise ValueError(
                f"Analysis source differs from its verified producer upload: {relative}"
            )
        return checksum

    return verified


def _saved_full_predictions(fit, x, targets, ids, mlp_seeds):
    """Bind every saved full-target predictor to exactly the recovered test arrays."""
    predictions = {}
    with np.load(fit / "per_example.npz", allow_pickle=False) as arrays:
        if arrays["context_ids"].tolist() != ids["test"] or not np.array_equal(
            arrays["x"], x["test"]
        ):
            raise ValueError("Saved predictions use different test inputs")
        for name, values in targets["test"].items():
            if not np.array_equal(arrays[f"target__{name}"], values):
                raise ValueError("Saved predictions use different component targets")
        for key in arrays.files:
            if key.startswith("prediction__") and key.endswith("__full"):
                predictions[key.split("__")[1]] = arrays[key].copy()
    if not {"ridge", "mlp", *(f"mlp_seed{seed}" for seed in mlp_seeds)}.issubset(predictions):
        raise ValueError(
            "Direction analysis requires ridge, primary MLP and every declared MLP seed"
        )
    if any(
        not np.isfinite(values).all() or values.shape != targets["test"]["full"].shape
        for values in predictions.values()
    ):
        raise ValueError("Saved full-target predictions contain invalid arrays")
    return predictions
