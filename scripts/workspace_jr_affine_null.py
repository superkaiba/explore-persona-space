#!/usr/bin/env python3
"""Run the preregistered exact-affine selection-artifact diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.workspace_components import reconstruction_metrics  # noqa: E402
from explore_persona_space.analysis.workspace_fit import (  # noqa: E402
    evaluate_component_fits,
    finite_json,
)
from explore_persona_space.analysis.workspace_lenses import (  # noqa: E402
    nonnegative_gradient_pursuit,
    rotated_dictionary,
)
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    file_sha256,
    load_workspace_jr_config,
    save_json,
)


def run(config_path: Path, output: Path) -> None:
    """Run all three k values and geometry-preserving rotations without model data."""
    config = load_workspace_jr_config(config_path)
    rng = np.random.default_rng(config["seed"])
    d, atoms = 16, 128
    x = {
        name: rng.standard_normal((n, d))
        for name, n in (("train", 256), ("validation", 64), ("test", 128))
    }
    ids = {name: [f"synthetic-{name}-{i}" for i in range(len(value))] for name, value in x.items()}
    affine, bias = rng.standard_normal((d, d)) / np.sqrt(d), rng.standard_normal(d)
    full = {name: value @ affine + bias for name, value in x.items()}
    dictionaries = {}
    for name in ("J", "R"):
        dictionary = torch.from_numpy(rng.standard_normal((atoms, d)))
        dictionaries[name] = dictionary / dictionary.norm(dim=1, keepdim=True)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "null_report.json").exists():
        raise FileExistsError("Choose a new output directory for this run")
    np.savez(
        output / "inputs.npz",
        affine=affine,
        bias=bias,
        **{f"x_{s}": value for s, value in x.items()},
        **{f"full_{s}": value for s, value in full.items()},
        **{f"dictionary_{s}": value.numpy() for s, value in dictionaries.items()},
    )
    report = {
        "schema": "workspace-jr-exact-affine-null-v1",
        "interpretation_scope": "synthetic dictionary selection artifact only",
        "dictionary_semantics": "J/R keys denote synthetic A/B dictionaries, not lenses",
        "token_targets": "Each activation is exactly x@A+b",
        "pooling": "Algebraic reduction of identical repeated tokens/draws; pooling is not executed",
        "algebraically_equivalent_k": 5,
        "predictors_executed": ["ridge", "identity_bias"],
        "dimension": d,
        "atoms": atoms,
        "config_sha256": file_sha256(config_path),
        "cells": {},
    }
    save_json(output / "run_started.json", report)
    for rotation in [None, *config["decomposition"]["rotations"]]:
        basis = (
            dictionaries
            if rotation is None
            else {
                name: rotated_dictionary(value, seed=rotation)[0]
                for name, value in dictionaries.items()
            }
        )
        for k in [config["decomposition"]["k_primary"], *config["decomposition"]["k_sensitivity"]]:
            targets = {split: {"full": values} for split, values in full.items()}
            diagnostics = {}
            for name, dictionary in basis.items():
                for split, value in full.items():
                    decomposition = nonnegative_gradient_pursuit(
                        torch.from_numpy(value), dictionary, k=k
                    )
                    component = decomposition.component.numpy()
                    rest = decomposition.remainder.numpy()
                    targets[split][name], targets[split][f"rest{name}"] = component, rest
                    diagnostics[f"{split}/{name}"] = {
                        **reconstruction_metrics(value, component, rest),
                        "mean_l0": float(decomposition.active_atoms.double().mean()),
                        "mean_token_squared_error": float(decomposition.squared_error.mean()),
                    }
            cell = f"k{k}-rotation{rotation}"
            result = evaluate_component_fits(x, targets, ids, config, output / cell)
            report["cells"][cell] = {
                "rotation": rotation,
                "k": k,
                "decomposition": diagnostics,
                "summary": result["paired_bootstrap"]["summary"],
            }
            save_json(output / "null_report.partial.json", finite_json(report))
            print(f"affine_null cell={cell} completed", flush=True)
    report["status"] = "complete"
    save_json(output / "null_report.json", finite_json(report))


def main() -> None:
    """Resolve explicit config and new output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/workspace_jr.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()
