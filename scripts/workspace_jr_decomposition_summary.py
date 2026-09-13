#!/usr/bin/env python3
"""Consolidate all observed/null sparsity cells on the completed primary cohort."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from workspace_jr_compare import read_arrays, read_cell  # noqa: E402
from workspace_jr_supplement import Source, bind_analysis  # noqa: E402

from explore_persona_space.analysis.workspace_comparison import (  # noqa: E402
    ROTATIONS,
    align_arrays,
    cell_key,
    paired_cohort,
)
from explore_persona_space.analysis.workspace_decomposition_summary import (  # noqa: E402
    null_statistics,
    observed_statistics,
    summarize_decomposition,
)
from explore_persona_space.analysis.workspace_fit import finite_json  # noqa: E402
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    file_sha256,
    load_workspace_jr_config,
    run_identity,
    save_json,
)


def completed_comparison(source, config_path, selection_path):
    """Bind the full 48-cell grid and every scope to one completed test cohort."""
    marker = source.json("comparison_complete.json")
    coverage = source.json("coverage.json")
    manifest = source.json("input_manifest.json")
    common = source.json("primary_scoring_cohort.json")["common_context_ids"]
    completion = source.json("completion_cohort.json")
    proof = source.json("source_proof.json")
    if (
        marker["status"] != "complete"
        or marker["input_manifest_sha256"] != source.hashes["input_manifest.json"]
        or coverage["status"] != "complete_grid"
        or coverage["expected_cells"] != 48
        or coverage["realized_cells"] != 48
        or coverage["missing_cells"]
        or manifest["config_sha256"] != file_sha256(config_path)
        or manifest["selection_sha256"] != file_sha256(selection_path)
        or completion["status"] != "complete"
        or not set(common).issubset(completion["joint_complete_context_ids"])
        or len(common) < 2
        or len(common) != len(set(common))
        or source.hashes["completion_cohort.json"] != proof["completion_cohort"]["report_sha256"]
        or set(marker["scopes"]) != {"primary", "comparison", "cross_model"}
    ):
        raise ValueError("Decomposition summary requires the complete original paired grid")
    expected = {
        cell_key(role, kind, k, rotation)
        for role in ("primary", "comparison")
        for kind in ("observed", "affine_null")
        for k in (5, 10, 25)
        for rotation in ROTATIONS
    }
    keys = [cell_key(e["role"], e["kind"], e["k"], e["rotation"]) for e in manifest["cells"]]
    if len(keys) != len(set(keys)) or set(keys) != expected or set(proof["cells"]) != expected:
        raise ValueError("Comparison manifest has duplicate or missing decomposition cells")
    for scope in marker["scopes"]:
        path = f"{scope}/comparisons.json"
        report = source.json(path)
        if (
            report["context_ids"] != common
            or source.hashes[path] != marker["scopes"][scope]["report_sha256"]
        ):
            raise ValueError("Comparison scopes use different primary cohorts")
    return manifest, common, proof


def verified_primary_cohort(source, cells, common):
    """Recompute the comparator's intersection, including every recorded exclusion."""
    eligible = source.json("completion_cohort.json")["joint_complete_context_ids"]
    actual, cohort = paired_cohort(
        {key: cell["ids"] for key, cell in cells.items()}, eligible_ids=eligible
    )
    if actual != common or cohort != source.json("primary_scoring_cohort.json"):
        raise ValueError("Fitted cells do not reproduce the completed primary scoring cohort")
    return cohort


def read_observed_statistics(entry, cell, common, identity, config):
    """Require successful uploaded analyzer output from exactly this observed fit."""
    source = Source(entry)
    phase = "diagnostics" if (entry["k"], entry["rotation"]) == (10, None) else "statistics"
    bind_analysis(source, phase, cell, identity)
    terminal = source.json("analysis_operations/exit.json")
    if (
        terminal["exit_code"] != 0
        or terminal["phase"] != "complete"
        or terminal["analysis_complete_sha256"] != source.hashes["analysis_complete.json"]
    ):
        raise ValueError("Decomposition statistics have no successful bound analyzer terminal")
    proof = source.json("input_proof.json")
    expected = {
        split: value["file_sha256"] for split, value in cell["manifest"]["coverage"].items()
    }
    if proof["sources"]["component_files"] != expected:
        raise ValueError("Token statistics were derived from different fitted components")
    rows = source.json("decomposition_statistics.json")["test"]
    arrays = observed_statistics(
        rows, cell["ids"], common, entry["k"], len(config["generation"]["seeds"])
    )
    return arrays, source.proof()


def read_null_statistics(entry, cell, common, config):
    """Use only the null arrays and original layout actually consumed by the fit."""
    source = Source(entry)
    path = f"decomposition_k{entry['k']}.npz"
    layout = source.json("layout.json")
    arrays = source.arrays(path)
    for name in ("layout.json", path):
        if source.hashes[name] != cell["manifest"]["prepared_sources"][name]:
            raise ValueError("Null token statistics differ from the actual prepared fit inputs")
    if layout["rollout_seeds"] != config["generation"]["seeds"]:
        raise ValueError("Null token layout changed the frozen rollout seeds")
    result = null_statistics(
        arrays, layout, cell["ids"], common, entry["k"], len(config["generation"]["seeds"])
    )
    return result, source.proof()


def main():
    """Save each completed cell before writing the grid-level JSON and CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/workspace_jr.yaml"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/selected_contexts.json"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Decomposition summary requires a fresh output directory")
    manifest = json.loads(args.manifest.read_text())
    config = load_workspace_jr_config(args.config)
    selection = json.loads(args.selection.read_text())
    comparison = Source(manifest["comparison"])
    fitted, common, proof = completed_comparison(comparison, args.config, args.selection)
    statistics = {}
    for entry in manifest["observed_statistics"]:
        key = cell_key(entry["role"], "observed", entry["k"], entry["rotation"])
        if key in statistics:
            raise ValueError("Duplicate observed statistics source")
        statistics[key] = entry
    if set(statistics) != {key for key in proof["cells"] if "/observed/" in key}:
        raise ValueError("Every observed cell requires its complete token statistics")
    verified_cells = {}
    identities = {
        role: run_identity(args.config, args.selection, role)
        for role in ("primary", "comparison")
    }
    for entry in fitted["cells"]:
        key = cell_key(entry["role"], entry["kind"], entry["k"], entry["rotation"])
        cell = read_cell(entry, config, identities[entry["role"]], selection)
        if cell["proof"] != proof["cells"][key]:
            raise ValueError("Cell sources differ from the completed paired comparison")
        verified_cells[key] = cell
    primary_cohort = verified_primary_cohort(comparison, verified_cells, common)
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "input_manifest.json", manifest)
    cells, flat, cell_markers = {}, [], {}
    for index, entry in enumerate(fitted["cells"], 1):
        key = cell_key(entry["role"], entry["kind"], entry["k"], entry["rotation"])
        identity = identities[entry["role"]]
        cell = verified_cells[key]
        targets, predictions = read_arrays(cell)
        targets, predictions = align_arrays(cell["ids"], common, targets, predictions)
        if entry["kind"] == "observed":
            arrays, statistics_proof = read_observed_statistics(
                statistics[key], cell, common, identity, config
            )
        else:
            arrays, statistics_proof = read_null_statistics(entry, cell, common, config)
        report = summarize_decomposition(targets, predictions, arrays)
        report.update(
            cell=key, source_proof=cell["proof"], statistics_source_proof=statistics_proof
        )
        folder = args.out / "cells" / key
        save_json(folder / "summary.json", finite_json(report))
        np.savez(folder / "per_context_token_statistics.npz", **arrays)
        save_json(
            folder / "cell_complete.json",
            {
                "status": "complete",
                "cell": key,
                "context_ids": common,
                "files": {
                    name: file_sha256(folder / name)
                    for name in ("summary.json", "per_context_token_statistics.npz")
                },
            },
        )
        marker_path = folder / "cell_complete.json"
        cell_markers[str(marker_path.relative_to(args.out))] = file_sha256(marker_path)
        cells[key] = report
        for arm, values in report["arms"].items():
            flat.append(
                {
                    "cell": key,
                    "arm": arm,
                    "contexts": len(common),
                    "tokens": report["tokens"],
                    **{
                        name: values[name]
                        for name in (
                            "maximum_active_atoms",
                            "contexts_with_increasing_error_steps",
                            "zero_token_input_energy",
                            "zero_pooled_full_variance",
                        )
                    },
                    **values["token_statistics_equal_context_then_equal_rollout"],
                    "residual_energy_fraction": values[
                        "residual_energy_fraction_equal_context_rollout"
                    ],
                    "pooled_component_variance_fraction": values[
                        "pooled_component_variance_fraction"
                    ],
                    **values["pooled_target_geometry"],
                }
            )
        print(f"Decomposition summary cell={index}/48 key={key} contexts={len(common)}", flush=True)
    with (args.out / "decomposition_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    save_json(
        args.out / "decomposition_summary.json",
        finite_json(
            {
                "status": "complete",
                "expected_cells": 48,
                "realized_cells": len(cells),
                "context_ids": common,
                "primary_scoring_cohort": primary_cohort,
                "cells": cells,
                "comparison_source_proof": comparison.proof(),
                "analysis_identity": {
                    role: run_identity(args.config, args.selection, role)
                    for role in ("primary", "comparison")
                },
                "implementation_sha256": {
                    name: file_sha256(Path(name))
                    for name in (
                        "scripts/workspace_jr_decomposition_summary.py",
                        "src/explore_persona_space/analysis/workspace_decomposition_summary.py",
                    )
                },
            }
        ),
    )
    save_json(
        args.out / "summary_complete.json",
        {
            "status": "complete",
            "report_sha256": file_sha256(args.out / "decomposition_summary.json"),
            "csv_sha256": file_sha256(args.out / "decomposition_summary.csv"),
            "input_manifest_sha256": file_sha256(args.out / "input_manifest.json"),
            "cell_complete_sha256": cell_markers,
        },
    )


if __name__ == "__main__":
    main()
