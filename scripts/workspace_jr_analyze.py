#!/usr/bin/env python3
"""Run paired direction diagnostics, rollout-noise checks and fixed-recipe learning curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.workspace_analysis_inputs import load_fitted_components  # noqa: E402
from explore_persona_space.analysis.workspace_calibration import eligible_calibration_tokens  # noqa: E402
from explore_persona_space.analysis.workspace_diagnostics import (
    evaluate_direction_readouts,
    rollout_noise_report,
)  # noqa: E402
from explore_persona_space.analysis.workspace_fit import evaluate_learning_curves, finite_json  # noqa: E402
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    content_sha256,
    file_sha256,
    load_workspace_jr_config,
    run_identity,
    save_json,
    validate_token_manifest,
)


def direction_diagnostics(args, config, inputs):
    """Choose shared eligible vocabulary from calibration only, then read saved full-y predictions."""
    from transformers import AutoTokenizer

    _, targets, _, ids, predictions, _, _ = inputs
    tokens = json.loads(args.token_manifest.read_text())
    validate_token_manifest(
        tokens,
        config_path=args.config,
        selection_path=args.selection,
        config=config,
        role=args.role,
    )
    manifest = json.loads((args.input_root / "dictionaries/manifest.json").read_text())
    if manifest["calibration_token_manifest_sha256"] != content_sha256(tokens):
        raise ValueError("Direction eligibility and fitted dictionary calibration differ")
    tokenizer = AutoTokenizer.from_pretrained(
        config["selection"][args.role], revision=config["models"][args.role]["revision"]
    )
    eligible = eligible_calibration_tokens(
        tokenizer, tokens, seed=config["seed"], maximum=config["diagnostics"]["direction_maximum"]
    )
    basis, selected, excluded = {}, None, {}
    for arm in ("J", "R"):
        path = args.input_root / "dictionaries" / f"{arm}.pt"
        if file_sha256(path) != manifest["arms"][arm]["sha256"]:
            raise ValueError("Readout dictionary changed after decomposition")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        lookup = {token: index for index, token in enumerate(saved["token_ids"].tolist())}
        available = [token for token in eligible if token in lookup]
        selected = set(available) if selected is None else selected & set(available)
        basis[arm] = {
            token: saved["dictionary"][lookup[token]].numpy().copy() for token in available
        }
        excluded[arm] = [token for token in eligible if token not in lookup]
        del saved
    selected = [token for token in eligible if token in selected]
    if not selected:
        raise ValueError("No shared finite nonzero eligible J/R directions")
    directions = {
        arm: np.stack([values[token] for token in selected], axis=1)
        for arm, values in basis.items()
    }
    report, arrays, samples = evaluate_direction_readouts(
        targets["train"]["full"],
        targets["test"]["full"],
        predictions,
        directions,
        selected,
        ids["train"],
        ids["test"],
        config,
    )
    report.update(
        dictionary_excluded_token_ids=excluded,
        predictor_scope="fresh full-answer fits on identical current native inputs; historical frozen predictors were not reused",
    )
    save_json(args.out / "direction_readouts.json", finite_json(report))
    np.savez(args.out / "direction_arrays.npz", **arrays)
    np.savez(args.out / "direction_bootstrap.npz", **samples)


def main():
    """Separate lightweight diagnostics from optional GPU learning-curve fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("diagnostics", "learning-curves"))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--fit-upload-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", choices=("primary", "comparison"), required=True)
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    parser.add_argument("--k", type=int, choices=(5, 10, 25), default=10)
    parser.add_argument("--rotation", type=int, choices=(20260913, 20260914, 20260915))
    parser.add_argument("--token-manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/workspace_jr.yaml"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/selected_contexts.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Analysis requires a fresh output directory")
    if args.phase == "diagnostics" and (args.token_manifest is None or args.rotation is not None):
        raise ValueError(
            "Readout diagnostics require native unrotated dictionaries and calibration tokens"
        )
    config = load_workspace_jr_config(args.config)
    identity = run_identity(args.config, args.selection, args.role)
    inputs = load_fitted_components(
        args.input_root,
        args.stage,
        args.k,
        args.rotation,
        config,
        json.loads(args.selection.read_text()),
        identity,
        args.fit_upload_receipt,
    )
    x, targets, rollouts, ids, _, statistics, proof = inputs
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "input_proof.json", {"identity": identity, "sources": proof})
    if args.phase == "diagnostics":
        save_json(args.out / "decomposition_statistics.json", statistics)
        for split in ("train", "validation", "test"):
            report, arrays = rollout_noise_report(
                rollouts[split], ids[split], config["generation"]["seeds"]
            )
            save_json(args.out / f"noise_{split}.json", finite_json(report))
            np.savez(args.out / f"noise_{split}_arrays.npz", **arrays)
        direction_diagnostics(args, config, inputs)
    else:
        import wandb

        cell = f"k{args.k}-rotation{args.rotation}"
        with wandb.init(
            project="workspace-jr",
            name=f"{args.role}-{args.stage}-{cell}-learning-curves",
            config={"identity": identity, "fit": config["fit"]},
        ) as run:
            save_json(args.out / "wandb.json", {"run_id": run.id, "url": run.url})
            evaluate_learning_curves(
                x,
                targets,
                ids,
                config,
                args.out / "curves",
                args.input_root / "fits" / args.stage / cell,
                device=args.device,
                training_logger=lambda record: run.log(training_scalars(record)),
            )
    save_json(
        args.out / "analysis_complete.json",
        {
            "identity": identity,
            "phase": args.phase,
            "status": "complete",
            "input_proof_sha256": file_sha256(args.out / "input_proof.json"),
        },
    )


def training_scalars(record):
    """Keep actual per-target training and validation losses in live telemetry."""
    values = {key: value for key, value in record.items() if isinstance(value, (int, float))}
    for index, key in enumerate(record["keys"]):
        values[f"{key[0]}/train_scaled_mse_before_step"] = record["train_loss"][index]
        values[f"{key[0]}/validation_scaled_mse_after_step"] = record["validation_loss"][index]
        values[f"{key[0]}/stopped"] = int(record["stopped"][index])
    return values


if __name__ == "__main__":
    main()
