#!/usr/bin/env python3
"""Apply the real lens dictionaries to exactly affine, noiseless token targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.workspace_analysis_inputs import (  # noqa: E402
    _upload_binding,
    load_fitted_components,
)
from explore_persona_space.analysis.workspace_decomposition import ValidatedDictionary  # noqa: E402
from explore_persona_space.analysis.workspace_artifacts import validate_producer  # noqa: E402
from explore_persona_space.analysis.workspace_fit import (  # noqa: E402
    evaluate_component_fits,
    finite_json,
)
from explore_persona_space.analysis.workspace_lenses import rotated_dictionary  # noqa: E402
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    file_sha256,
    load_workspace_jr_config,
    run_identity,
    save_json,
    save_tensors,
)


def affine_targets(x, weights, bias):
    """Define one fixed affine map in FP64 before the FP32 sparse arithmetic."""
    if (
        weights.ndim != 2
        or bias.shape != (weights.shape[1],)
        or not np.isfinite(weights).all()
        or not np.isfinite(bias).all()
    ):
        raise ValueError("Invalid fixed affine coefficients")
    full = {}
    for split, values in x.items():
        if values.ndim != 2 or values.shape[1] != len(weights) or not np.isfinite(values).all():
            raise ValueError("Invalid affine input array")
        full[split] = values.astype(np.float64) @ weights.astype(np.float64) + bias
        if not np.isfinite(full[split]).all():
            raise ValueError("Nonfinite affine targets")
    return full


def validate_layout(statistics, ids, k):
    """Preserve every observed rollout length and the exact fitted context order."""
    layout = {}
    for split, contexts in ids.items():
        rows = statistics[split]
        if [row["context_id"] for row in rows] != contexts:
            raise ValueError("Null rollout layout differs from fitted context order")
        lengths = [row["token_counts"] for row in rows]
        if any(len(row) != k or any(type(n) is not int or n < 1 for n in row) for row in lengths):
            raise ValueError("Null requires complete nonempty original rollout lengths")
        layout[split] = lengths
    return layout


@torch.no_grad()
def decompose_affine_targets(full, dictionaries, checkpoints=(5, 10, 25), batch_size=128):
    """Reduce identical repeated token/draw vectors algebraically, retaining FP64 sums.

    Every token of context i is h_i = x_i A + b. Thus every rollout mean
    of s(h_i), and their equal-weight mean, is s(h_i), regardless of the
    preserved positive rollout lengths. This shortcut applies only to this
    identical-token null, never to observed answer activations.
    """
    if set(dictionaries) != {"J", "R"} or batch_size < 1:
        raise ValueError("Paired dictionaries and a positive batch size are required")
    targets = {k: {split: {"full": h} for split, h in full.items()} for k in checkpoints}
    diagnostics = {k: {} for k in checkpoints}
    fields = (
        "active_atoms",
        "squared_error",
        "input_squared_norm",
        "zero_update_steps",
        "increasing_error_steps",
    )
    for split, h in full.items():
        if h.ndim != 2 or not len(h) or not np.isfinite(h).all():
            raise ValueError("Nonempty finite affine target arrays are required")
        for arm, executor in dictionaries.items():
            components = {k: [] for k in checkpoints}
            statistics = {k: {field: [] for field in fields} for k in checkpoints}
            for begin in range(0, len(h), batch_size):
                values = torch.from_numpy(h[begin : begin + batch_size]).to(
                    device=executor.dictionary.device, dtype=torch.float32
                )
                for k, part in executor.decompose(values, checkpoints).items():
                    components[k].append(part.component.double().cpu().numpy())
                    for field in fields:
                        statistics[k][field].append(getattr(part, field).cpu().numpy())
                print(
                    f"affine decomposition split={split} arm={arm} contexts={min(begin + batch_size, len(h))}/{len(h)}",
                    flush=True,
                )
            for k in checkpoints:
                component = np.concatenate(components[k])
                targets[k][split][arm] = component
                targets[k][split][f"rest{arm}"] = h - component
                diagnostics[k][f"{split}/{arm}"] = {
                    field: np.concatenate(values) for field, values in statistics[k].items()
                }
    return targets, diagnostics


def training_scalars(record):
    """Record only scalar train/validation telemetry during null fitting."""
    values = {key: value for key, value in record.items() if isinstance(value, (int, float))}
    for index, key in enumerate(record["keys"]):
        for name in ("train_loss", "validation_loss", "stopped"):
            values[f"{key[0]}/{name}"] = float(record[name][index])
    return values


def prepare(args, config, identity):
    """Save all null targets before independent persisted per-k fitting phases."""
    if args.out.exists():
        raise ValueError("Null execution requires a fresh output directory")
    inputs = load_fitted_components(
        args.input_root,
        args.stage,
        10,
        None,
        config,
        json.loads(args.selection.read_text()),
        identity,
        args.fit_upload_receipt,
    )
    x, _, _, ids, _, statistics, proof = inputs
    verify = _upload_binding(args.input_root, args.fit_upload_receipt)
    coefficients = args.input_root / "fits" / args.stage / "k10-rotationNone/ridge-full.npz"
    coefficient_sha = verify(coefficients)
    with np.load(coefficients, allow_pickle=False) as saved:
        weights, bias = saved["weights"].copy(), saved["bias"].copy()
    full = affine_targets(x, weights, bias)
    layout = validate_layout(statistics, ids, len(config["generation"]["seeds"]))
    manifest_path = args.input_root / "dictionaries/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if args.stage == "main" and manifest["pilot_only"]:
        raise ValueError("Main null cannot use pilot-only lens calibration")
    args.out.mkdir(parents=True)
    report = {
        "schema": "workspace-jr-native-affine-null-v1",
        "identity": identity,
        "status": "running",
        "stage": args.stage,
        "rotation": args.rotation,
        "scope": "sparse-selection artifact; synthetic repeated activations, not language processing",
        "map": "observed full-answer ridge: fitted on train, alpha selected on validation, no test fitting",
        "coefficient_file_sha256": coefficient_sha,
        "dictionary_manifest_sha256": file_sha256(manifest_path),
        "pilot_only_dictionaries": manifest["pilot_only"],
        "token_activation": "FP64 x@A+b, identical at every original answer position and rollout",
        "pooling": "exact algebraic reduction of identical vectors; original lengths and K preserved",
        "sparse_precision": "same FP32 highest/TF32-off pursuit; remainder uses original FP64 affine h minus component",
        "fp32_conversion_relative_squared_error": {
            split: float(
                np.square(h - h.astype(np.float32).astype(np.float64)).sum()
                / max(np.square(h).sum(), 1e-300)
            )
            for split, h in full.items()
        },
        "oracle_affine_sse": {
            split: float(
                np.square(
                    h - (x[split].astype(np.float64) @ weights.astype(np.float64) + bias)
                ).sum()
            )
            for split, h in full.items()
        },
        "cells": {},
    }
    save_json(args.out / "input_proof.json", proof)
    save_json(
        args.out / "layout.json",
        {
            "context_ids": ids,
            "rollout_seeds": config["generation"]["seeds"],
            "token_counts": layout,
        },
    )
    np.savez(
        args.out / "affine_inputs.npz",
        weights=weights,
        bias=bias,
        **{f"x__{s}": a for s, a in x.items()},
        **{f"full__{s}": a for s, a in full.items()},
    )
    save_json(args.out / "null_started.json", report)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    basis = {}
    for arm in ("J", "R"):
        path = args.input_root / "dictionaries" / f"{arm}.pt"
        if verify(path) != manifest["arms"][arm]["sha256"]:
            raise ValueError("Null dictionary differs from fitted observations")
        dictionary = torch.load(path, map_location=args.device, weights_only=True)["dictionary"]
        if args.rotation is not None:
            dictionary, q = rotated_dictionary(dictionary, seed=args.rotation)
            save_tensors(args.out / f"rotation_{arm}.pt", {"q": q.cpu(), "seed": args.rotation})
        basis[arm] = ValidatedDictionary(dictionary)
    targets, diagnostics = decompose_affine_targets(full, basis)
    del basis, dictionary
    torch.cuda.empty_cache()
    for k in targets:
        np.savez(
            args.out / f"targets_k{k}.npz",
            **{f"{s}__{name}": a for s, values in targets[k].items() for name, a in values.items()},
        )
        np.savez(
            args.out / f"decomposition_k{k}.npz",
            **{
                f"{cell}__{name}": a
                for cell, values in diagnostics[k].items()
                for name, a in values.items()
            },
        )
    report["status"] = "decomposition_complete"
    save_json(args.out / "null_prepared.json", report)


def fit(args, config, identity):
    """Fit one k only after every prepared source was verified in durable storage."""
    import wandb

    verify = _upload_binding(args.out, args.prepared_upload_receipt)
    sources = [
        "null_prepared.json",
        "input_proof.json",
        "layout.json",
        "affine_inputs.npz",
        f"targets_k{args.k}.npz",
        f"decomposition_k{args.k}.npz",
    ]
    if args.rotation is not None:
        sources += [f"rotation_{arm}.pt" for arm in ("J", "R")]
    hashes = {name: verify(args.out / name) for name in sources}
    report = json.loads((args.out / "null_prepared.json").read_text())
    validate_producer(report["identity"], identity)
    if (
        report["status"] != "decomposition_complete"
        or report["stage"] != args.stage
        or report["rotation"] != args.rotation
    ):
        raise ValueError("Null fit differs from the prepared experiment cell")
    layout = json.loads((args.out / "layout.json").read_text())
    ids = layout["context_ids"]
    if layout["rollout_seeds"] != config["generation"]["seeds"]:
        raise ValueError("Null fit rollout seeds differ")
    with np.load(args.out / "affine_inputs.npz", allow_pickle=False) as saved:
        x = {split: saved[f"x__{split}"].copy() for split in ids}
        full = affine_targets(x, saved["weights"], saved["bias"])
        if any(not np.array_equal(full[s], saved[f"full__{s}"]) for s in ids):
            raise ValueError("Prepared full targets are not the saved affine map")
    with np.load(args.out / f"targets_k{args.k}.npz", allow_pickle=False) as saved:
        targets = {
            s: {name: saved[f"{s}__{name}"].copy() for name in ("full", "J", "restJ", "R", "restR")}
            for s in ids
        }
    if any(not np.array_equal(full[s], targets[s]["full"]) for s in ids):
        raise ValueError("Decomposed null full targets differ from affine map")
    cell = f"k{args.k}-rotation{args.rotation}"
    if (args.out / cell).exists():
        raise ValueError("Null fit requires a fresh per-k output; retain prior attempts")
    save_json(
        args.out / cell / "input_manifest.json", {"identity": identity, "prepared_sources": hashes}
    )
    with wandb.init(
        project="workspace-jr",
        name=f"{args.role}-{args.stage}-affine-null-{cell}",
        config={"identity": identity, "fit": config["fit"], "cell": cell},
    ) as run:
        save_json(args.out / cell / "wandb.json", {"run_id": run.id, "url": run.url})
        result = evaluate_component_fits(
            x,
            targets,
            ids,
            config,
            args.out / cell,
            mlp_device=args.device,
            training_logger=lambda record: run.log(training_scalars(record)),
        )
    save_json(
        args.out / cell / "null_complete.json",
        finite_json(
            {
                "status": "complete",
                "identity": identity,
                "cell": cell,
                "results_sha256": file_sha256(args.out / cell / "results.json"),
                "summary": result["paired_bootstrap"]["summary"],
            }
        ),
    )
    print(f"affine null fitted cell={cell} complete", flush=True)


def main():
    """Run explicit preparation or one independent fit with a verified upload boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit"))
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--fit-upload-receipt", type=Path)
    parser.add_argument("--prepared-upload-receipt", type=Path)
    parser.add_argument("--role", choices=("primary", "comparison"), required=True)
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, choices=(5, 10, 25), default=10)
    parser.add_argument("--rotation", type=int, choices=(20260913, 20260914, 20260915))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/workspace_jr.yaml"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/selected_contexts.json"),
    )
    args = parser.parse_args()
    if args.phase == "prepare" and (args.input_root is None or args.fit_upload_receipt is None):
        raise ValueError("Null preparation requires uploaded observed fit inputs")
    if args.phase == "fit" and args.prepared_upload_receipt is None:
        raise ValueError("Null fit requires uploaded prepared targets")
    config = load_workspace_jr_config(args.config)
    identity = run_identity(args.config, args.selection, args.role)
    (prepare if args.phase == "prepare" else fit)(args, config, identity)


if __name__ == "__main__":
    main()
