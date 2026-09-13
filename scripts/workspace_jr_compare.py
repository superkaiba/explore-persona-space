#!/usr/bin/env python3
"""Verify saved per-example fits and compute paired control and model comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.analysis.workspace_analysis_inputs import _upload_binding  # noqa: E402
from explore_persona_space.analysis.workspace_artifacts import (  # noqa: E402
    validate_coverage,
    validate_producer,
)
from explore_persona_space.analysis.workspace_comparison import (  # noqa: E402
    PREDICTORS,
    ROTATIONS,
    TARGETS,
    align_arrays,
    cell_key,
    combine_paired,
    paired_cohort,
    quality_matched_contrasts,
    registered_contrasts,
    rotation_variation,
)
from explore_persona_space.analysis.workspace_components import (  # noqa: E402
    component_metrics,
    paired_context_bootstrap,
    reconstruction_metrics,
)
from explore_persona_space.analysis.workspace_fit import _fit_contract, finite_json  # noqa: E402
from explore_persona_space.analysis.workspace_quality import nearest_quality_match  # noqa: E402
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    content_sha256,
    file_sha256,
    load_workspace_jr_config,
    run_identity,
    save_json,
)


def relative_path(value):
    """Reject ambiguous source paths before resolving uploaded objects."""
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part.startswith(".") for part in path.parts):
        raise ValueError("Uploaded source paths must be safe relative paths")
    return path


def fingerprint(value):
    """Use the exact raw-array convention saved by the fit producer."""
    a = np.ascontiguousarray(value)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def read_cell(entry, config, identity, selection):
    """Verify immutable fit sources and actual terminal status before reading metrics."""
    root = Path(entry["root"])
    fit = root / relative_path(entry["fit_relative"])
    receipt = Path(entry["upload_receipt"])
    verify = _upload_binding(root, receipt)
    paths = [
        fit / name
        for name in (
            "results.json",
            "input_manifest.json",
            "per_example.npz",
            "bootstrap_samples.npz",
            "mlp/selection.json",
        )
    ]
    terminal_path = root / relative_path(entry["terminal_relative"])
    hashes = {str(path.relative_to(root)): verify(path) for path in [*paths, terminal_path]}
    terminal = json.loads(terminal_path.read_text())
    if (
        type(terminal.get("exit_code")) is not int
        or terminal["exit_code"] != 0
        or terminal.get("phase") != "complete"
    ):
        raise ValueError("Fit source lacks a successful complete producer terminal")
    manifest = json.loads((fit / "input_manifest.json").read_text())
    validate_producer(manifest["identity"], identity, native_ancestor=True)
    result = json.loads((fit / "results.json").read_text())
    if result["fit_contract"] != _fit_contract(config):
        raise ValueError("Fit tuning and statistics differ from the frozen configuration")
    if (
        result["mlp_selection_sha256"]
        != hashes[str((fit / "mlp/selection.json").relative_to(root))]
    ):
        raise ValueError("Saved MLP recipe selection changed")
    expected_cell = f"k{entry['k']}-rotation{entry['rotation']}"
    if fit.name != expected_cell:
        raise ValueError("Fit path and requested cell differ")
    if entry["kind"] == "observed":
        contract = manifest["contract"]
        dictionary_path = root / "dictionaries/manifest.json"
        hashes["dictionaries/manifest.json"] = verify(dictionary_path)
        if json.loads(dictionary_path.read_text()) != contract["dictionaries"]:
            raise ValueError("Fitted dictionary differs from its persisted manifest")
        coefficient_path = fit / "ridge-full.npz"
        hashes[str(coefficient_path.relative_to(root))] = verify(coefficient_path)
        if contract["k"] != entry["k"] or contract["rotation"] != entry["rotation"]:
            raise ValueError("Fitted decomposition cell differs")
        if contract["dictionaries"]["pilot_only"] or not manifest["identity"].get(
            "execution_readiness_sha256"
        ):
            raise ValueError("Main comparison cannot consume pilot dictionaries or ungated fits")
        if fit.parent.name != "main":
            raise ValueError("Only main observational fits belong in the final comparison")
        for split, coverage in manifest["coverage"].items():
            planned = selection["subsets"][f"main_{split}"][: coverage["planned_contexts"]]
            included = validate_coverage(
                coverage, [row["prompt_sha256"] for row in planned], manifest["identity"]
            )
            if coverage["contract"] != contract or len(included) != result["split_counts"][split]:
                raise ValueError("Fit coverage differs from the frozen context selection")
    else:
        prepared_path = root / "null_prepared.json"
        hashes["null_prepared.json"] = verify(prepared_path)
        prepared = json.loads(prepared_path.read_text())
        if prepared["stage"] != "main" or prepared["rotation"] != entry["rotation"]:
            raise ValueError("Affine null uses a different stage or rotation")
        for name, expected in manifest["prepared_sources"].items():
            if verify(root / relative_path(name)) != expected:
                raise ValueError("Null fit inputs changed after preparation")
        complete = fit / "null_complete.json"
        hashes[str(complete.relative_to(root))] = verify(complete)
        completion = json.loads(complete.read_text())
        if (
            completion["status"] != "complete"
            or completion["cell"] != expected_cell
            or completion["results_sha256"] != file_sha256(fit / "results.json")
        ):
            raise ValueError("Affine-null completion does not bind this fit")
    with np.load(fit / "per_example.npz", allow_pickle=False) as saved:
        ids = saved["context_ids"].tolist()
    if (
        len(ids) < 2
        or len(ids) != len(set(ids))
        or ids != list(result["paired_bootstrap"]["context_ids"])
    ):
        raise ValueError("Invalid saved per-example context identity")
    if content_sha256(ids) != result["input_fingerprints"]["test"]["ids_sha256"]:
        raise ValueError("Saved test IDs differ from fitted inputs")
    if entry["kind"] == "observed" and set(ids) != set(
        manifest["coverage"]["test"]["included_prompt_sha256"]
    ):
        raise ValueError("Saved test rows differ from completed test coverage")
    return {
        "entry": entry,
        "fit": fit,
        "result": result,
        "manifest": manifest,
        "ids": ids,
        "proof": {
            "files": hashes,
            "upload_receipt_sha256": file_sha256(receipt),
            "upload": json.loads(receipt.read_text()),
            "producer_identity": manifest["identity"],
        },
    }


def bind_source_families(cells):
    """Require matched training/extraction and tie nulls to the exact native ridge fit."""
    for role in ("primary", "comparison"):
        observed = [
            cell
            for cell in cells.values()
            if cell["entry"]["role"] == role and cell["entry"]["kind"] == "observed"
        ]
        if not observed:
            if any(cell["entry"]["role"] == role for cell in cells.values()):
                raise ValueError("Affine-null comparison requires its observational source fit")
            continue
        reference = observed[0]
        identity = reference["manifest"]["identity"]
        dictionary = reference["manifest"]["contract"]["dictionaries"]
        for cell in observed:
            if (
                cell["manifest"]["identity"] != identity
                or cell["manifest"]["contract"]["dictionaries"] != dictionary
            ):
                raise ValueError(
                    "Observed cells have different producer, readiness or dictionaries"
                )
            for split in ("train", "validation", "test"):
                actual = cell["result"]["input_fingerprints"][split]
                expected = reference["result"]["input_fingerprints"][split]
                if (
                    any(actual[name] != expected[name] for name in ("ids_sha256", "x"))
                    or actual["targets"]["full"] != expected["targets"]["full"]
                ):
                    raise ValueError(
                        "Observed cells differ in context inputs, full targets or split membership"
                    )
        native = cells.get(cell_key(role, "observed", 10, None))
        for cell in cells.values():
            if cell["entry"]["role"] != role or cell["entry"]["kind"] != "affine_null":
                continue
            if native is None:
                raise ValueError("Null requires the same model's observed native k10 fit")
            root = Path(cell["entry"]["root"])
            report = json.loads((root / "null_prepared.json").read_text())
            proof = json.loads((root / "input_proof.json").read_text())
            native_fit = native["entry"]["fit_relative"]
            hashes = native["proof"]["files"]
            bindings = {
                "results_sha256": hashes[f"{native_fit}/results.json"],
                "input_manifest_sha256": hashes[f"{native_fit}/input_manifest.json"],
                "per_example_sha256": hashes[f"{native_fit}/per_example.npz"],
                "input_fingerprints": native["result"]["input_fingerprints"],
                "source_identity": native["manifest"]["identity"],
            }
            if any(proof[key] != value for key, value in bindings.items()):
                raise ValueError("Affine null was prepared from another observational fit")
            if (
                report["coefficient_file_sha256"] != hashes[f"{native_fit}/ridge-full.npz"]
                or report["dictionary_manifest_sha256"] != hashes["dictionaries/manifest.json"]
            ):
                raise ValueError("Affine null used different ridge coefficients or dictionaries")
            for split in ("train", "validation", "test"):
                actual = cell["result"]["input_fingerprints"][split]
                expected = native["result"]["input_fingerprints"][split]
                if any(actual[key] != expected[key] for key in ("ids_sha256", "x")):
                    raise ValueError(
                        "Affine null changed the observed context inputs or split membership"
                    )


def bind_quality(entry, report, cells):
    """Bind every calibration summary to the observed dictionary and readiness."""
    root = Path(entry["root"])
    verify = _upload_binding(root, Path(entry["upload_receipt"]))
    native = cells.get(cell_key(entry["role"], "observed", 10, None))
    if native is None:
        raise ValueError("Quality matching needs the same model's native observational fit")
    sources, summaries = {}, {}
    reference = None
    for rotation in ROTATIONS:
        path = root / "controls" / f"rotation{rotation}" / "quality.json"
        checksum = verify(path)
        if report["source_report_sha256"][str(rotation)] != checksum:
            raise ValueError("Quality match source report changed")
        quality = json.loads(path.read_text())
        contract = quality["contract"]
        if quality["status"] != "complete" or contract["rotation"] != rotation:
            raise ValueError("Incomplete or mismatched calibration quality orientation")
        if (
            contract["dictionary_manifest_sha256"]
            != native["proof"]["files"]["dictionaries/manifest.json"]
            or contract["readiness_sha256"]
            != native["manifest"]["identity"]["execution_readiness_sha256"]
        ):
            raise ValueError("Calibration quality uses another dictionary or readiness")
        invariant = {key: value for key, value in contract.items() if key != "rotation"}
        if reference is not None and invariant != reference:
            raise ValueError(
                "Calibration quality summaries use different capture or numerical contracts"
            )
        reference = invariant
        sources[str(rotation)] = checksum
        summaries[rotation] = quality["summary"]
    for arm in ("J", "R"):
        for k in (5, 10, 25):
            for rotation in ROTATIONS[1:]:
                expected = nearest_quality_match(
                    summaries[None][arm][str(k)],
                    {
                        control_k: summaries[rotation][arm][str(control_k)]
                        for control_k in (5, 10, 25)
                    },
                )
                if report["matching"][arm][str(k)][str(rotation)] != expected:
                    raise ValueError(
                        "Quality matching differs from its actual calibration summaries"
                    )
    return sources


def bind_global_sources(path, evidence, *, resume):
    """Freeze actual evidence bytes, including quality controls, across partial resumes."""
    if path.exists():
        if not resume or json.loads(path.read_text()) != evidence:
            raise ValueError("Comparison source evidence changed across resume")
    else:
        if list(path.parent.glob("*/*/cell_complete.json")):
            raise ValueError("Completed cells lack their global source-evidence contract")
        save_json(path, evidence)


def read_arrays(cell):
    """Reproduce all reported R² from exact test targets and saved predictions."""
    result = cell["result"]
    with np.load(cell["fit"] / "per_example.npz", allow_pickle=False) as saved:
        targets = {name: saved[f"target__{name}"].copy() for name in TARGETS}
        if fingerprint(saved["x"]) != result["input_fingerprints"]["test"]["x"]:
            raise ValueError("Saved test inputs differ from fitted input fingerprints")
        predictions = {
            predictor: {name: saved[f"prediction__{predictor}__{name}"].copy() for name in TARGETS}
            for predictor in result["metrics"]
        }
    if not set(PREDICTORS) <= set(predictions):
        raise ValueError("Missing ridge, primary MLP or registered MLP seed")
    for name, value in targets.items():
        if fingerprint(value) != result["input_fingerprints"]["test"]["targets"][name]:
            raise ValueError("Saved test target differs from fitted target fingerprint")
        for predictor, by_target in predictions.items():
            measured = component_metrics(value, by_target[name])
            for key in ("r2", "sse", "sst"):
                expected = result["metrics"][predictor][name][key]
                actual = measured[key]
                if expected is None or actual is None:
                    if expected is not actual:
                        raise ValueError("Undefined saved metric was relabeled")
                elif not np.isclose(actual, expected, rtol=1e-9, atol=1e-10):
                    raise ValueError(f"Reported {key} differs from actual predictions")
    for arm in ("J", "R"):
        reconstruction_metrics(targets["full"], targets[arm], targets[f"rest{arm}"])
    return targets, {name: predictions[name] for name in PREDICTORS}


def checkpoint_bootstrap(folder, cell, common, config, *, resume):
    """Resume only fully published, byte-bound cell statistics for the same cohort."""
    contract = {
        "source_proof_sha256": content_sha256(cell["proof"]),
        "context_ids": common,
        "fit_contract": _fit_contract(config),
        "script_sha256": file_sha256(Path(__file__)),
        "comparison_module_sha256": file_sha256(
            Path("src/explore_persona_space/analysis/workspace_comparison.py")
        ),
    }
    marker = folder / "cell_complete.json"
    if marker.exists():
        complete = json.loads(marker.read_text())
        if not resume or complete["contract"] != contract:
            raise ValueError("Existing comparison cell has a different source or cohort")
        for name, expected in complete["files"].items():
            if file_sha256(folder / name) != expected:
                raise ValueError("Completed comparison checkpoint changed")
        boot = json.loads((folder / "summary.json").read_text())
        with np.load(folder / "bootstrap_samples.npz", allow_pickle=False) as saved:
            boot["samples"] = {name: saved[name].copy() for name in saved.files}
        return boot, complete["full_target_fingerprint"]
    targets, predictions = read_arrays(cell)
    targets, predictions = align_arrays(cell["ids"], common, targets, predictions)
    settings = config["statistics"]["bootstrap"]
    boot = paired_context_bootstrap(
        targets,
        predictions,
        common,
        n_bootstrap=settings["draws"],
        seed=config["seed"],
        confidence=settings["confidence"],
    )
    save_json(
        folder / "summary.json",
        finite_json({name: value for name, value in boot.items() if name != "samples"}),
    )
    np.savez(folder / "bootstrap_samples.npz", **boot["samples"])
    save_json(folder / "original_fit_summary.json", cell["result"])
    full_hash = fingerprint(targets["full"])
    save_json(
        marker,
        {
            "contract": contract,
            "files": {
                name: file_sha256(folder / name)
                for name in ("summary.json", "bootstrap_samples.npz", "original_fit_summary.json")
            },
            "full_target_fingerprint": full_hash,
        },
    )
    return boot, full_hash


def run_comparison(args):
    """Compute own-model and shared-model cohorts, saving each completed cell."""
    config = load_workspace_jr_config(args.config)
    manifest = json.loads(args.manifest.read_text())
    if args.out.exists():
        if (
            not args.resume
            or json.loads((args.out / "input_manifest.json").read_text()) != manifest
        ):
            raise ValueError("Comparison requires fresh output or exact-source explicit resume")
        if (args.out / "comparison_complete.json").exists():
            raise ValueError("Completed comparison is immutable")
    elif args.resume:
        raise ValueError("Cannot resume an absent comparison")
    if manifest["config_sha256"] != file_sha256(args.config) or manifest[
        "selection_sha256"
    ] != file_sha256(args.selection):
        raise ValueError("Comparison sources differ from frozen configuration or selection")
    identities = {
        role: run_identity(args.config, args.selection, role) for role in ("primary", "comparison")
    }
    cells = {}
    for entry in manifest["cells"]:
        key = cell_key(entry["role"], entry["kind"], entry["k"], entry["rotation"])
        if key in cells:
            raise ValueError("Duplicate comparison cell")
        cells[key] = read_cell(
            entry, config, identities[entry["role"]], json.loads(args.selection.read_text())
        )
    expected = {
        cell_key(role, kind, k, rotation)
        for role in identities
        for kind in ("observed", "affine_null")
        for k in (5, 10, 25)
        for rotation in ROTATIONS
    }
    missing_cells = sorted(expected - cells.keys())
    if missing_cells and not args.allow_incomplete:
        raise ValueError(f"Required main comparison grid incomplete: {missing_cells}")
    bind_source_families(cells)
    args.out.mkdir(parents=True, exist_ok=args.resume)
    save_json(args.out / "input_manifest.json", manifest)
    save_json(
        args.out / "coverage.json",
        {
            "expected_cells": len(expected),
            "realized_cells": len(cells),
            "missing_cells": missing_cells,
            "status": "incomplete" if missing_cells else "complete_grid",
        },
    )
    scopes = {}
    for role in identities:
        selected = {key: value for key, value in cells.items() if key.startswith(role + "/")}
        if selected:
            scopes[role] = selected
    if set(scopes) == set(identities):
        scopes["cross_model"] = cells
    all_proofs = {"cells": {key: value["proof"] for key, value in cells.items()}, "quality": {}}
    quality, quality_exports = {}, {}
    for entry in manifest.get("quality_matches", []):
        role = entry["role"]
        if role in quality or role not in identities:
            raise ValueError("Duplicate or unknown calibration-quality role")
        path = Path(entry["root"]) / "quality_matches.json"
        checksum = _upload_binding(Path(entry["root"]), Path(entry["upload_receipt"]))(path)
        report = json.loads(path.read_text())
        validate_producer(report["identity"], identities[role], native_ancestor=True)
        if report["status"] != "complete":
            raise ValueError("Calibration-quality matching has not completed")
        quality_sources = bind_quality(entry, report, cells)
        quality[role] = report
        all_proofs["quality"][role] = {
            "report_sha256": checksum,
            "receipt_sha256": file_sha256(Path(entry["upload_receipt"])),
            "source_reports": quality_sources,
        }
        quality_exports[role] = {
            "report": report,
            "sha256": checksum,
            "receipt_sha256": file_sha256(Path(entry["upload_receipt"])),
            "source_reports": quality_sources,
        }
    bind_global_sources(args.out / "source_proof.json", all_proofs, resume=args.resume)
    for role, export in quality_exports.items():
        save_json(args.out / "calibration_quality" / f"{role}.json", export)
    settings = config["statistics"]["bootstrap"]
    summaries = {}
    for scope, selected in scopes.items():
        common, cohort = paired_cohort({key: cell["ids"] for key, cell in selected.items()})
        folder = args.out / scope
        save_json(folder / "cohort.json", cohort)
        bootstraps = {}
        full_targets = {}
        for index, (key, cell) in enumerate(selected.items()):
            started = time.perf_counter()
            cell_folder = folder / key.replace("/", "__")
            boot, full_hash = checkpoint_bootstrap(
                cell_folder, cell, common, config, resume=args.resume
            )
            group = (cell["entry"]["role"], cell["entry"]["kind"])
            if group in full_targets and full_targets[group] != full_hash:
                raise ValueError("Alternative decompositions use different full-answer targets")
            full_targets[group] = full_hash
            bootstraps[key] = boot
            print(
                f"comparison scope={scope} cell={index + 1}/{len(selected)} name={key} contexts={len(common)} elapsed={time.perf_counter() - started:.3f}s",
                flush=True,
            )
        contrasts, unavailable = registered_contrasts(selected)
        quality_ledger = {}
        for role, quality_report in quality.items():
            if scope not in (role, "cross_model"):
                continue
            extra, ledger = quality_matched_contrasts(role, quality_report, selected)
            contrasts.update(extra)
            quality_ledger.update(ledger)
        contrast_summary, samples = combine_paired(
            bootstraps, contrasts, confidence=settings["confidence"]
        )
        np.savez(folder / "contrast_samples.npz", **samples)
        report = {
            "contrasts": contrast_summary,
            "unavailable_contrasts": unavailable,
            "rotation_variation": rotation_variation(contrast_summary),
            "context_ids": common,
            "conditional_on_fitted_predictors": True,
            "affine_null_interpretation": "diagnostic difference, not a causal correction",
            "smallest_practical_gap": config["statistics"]["smallest_practical_gap"],
            "calibration_matching": quality_ledger,
        }
        save_json(folder / "comparisons.json", finite_json(report))
        summaries[scope] = {
            "report_sha256": file_sha256(folder / "comparisons.json"),
            "contexts": len(common),
            "cells": len(selected),
        }
    save_json(
        args.out / "comparison_complete.json",
        {
            "status": "complete" if not missing_cells else "incomplete_grid",
            "scopes": summaries,
            "input_manifest_sha256": file_sha256(args.out / "input_manifest.json"),
            "finished_at_epoch": int(time.time()),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/workspace_jr.yaml"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/selected_contexts.json"),
    )
    run_comparison(parser.parse_args())


if __name__ == "__main__":
    main()
