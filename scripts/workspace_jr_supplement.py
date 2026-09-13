#!/usr/bin/env python3
"""Produce paired agreement, noise, readout and learning-curve figure inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from workspace_jr_compare import read_arrays, read_cell  # noqa: E402

from explore_persona_space.analysis.workspace_analysis_inputs import _upload_binding  # noqa: E402
from explore_persona_space.analysis.workspace_artifacts import validate_producer  # noqa: E402
from explore_persona_space.analysis.workspace_comparison import align_arrays, cell_key  # noqa: E402
from explore_persona_space.analysis.workspace_components import (
    paired_context_bootstrap,
    workspace_gap_contrasts,
)  # noqa: E402
from explore_persona_space.analysis.workspace_fit import finite_json  # noqa: E402
from explore_persona_space.analysis.workspace_runtime import (
    content_sha256,
    file_sha256,
    load_workspace_jr_config,
    run_identity,
    save_json,
)  # noqa: E402
from explore_persona_space.analysis.workspace_supplement import (
    agreement_statistics,
    mapping_references,
    paired_noise,
    paired_readouts,
)  # noqa: E402


class Source:
    """Every opened artifact must match its producer's immutable uploaded bytes."""

    def __init__(self, entry):
        self.root = Path(entry["root"])
        self.receipt = Path(entry["upload_receipt"])
        self.verify = _upload_binding(self.root, self.receipt)
        self.hashes = {}

    def path(self, name):
        path = self.root / name
        self.hashes[name] = self.verify(path)
        return path

    def json(self, name):
        return json.loads(self.path(name).read_text())

    def arrays(self, name):
        with np.load(self.path(name), allow_pickle=False) as saved:
            return {key: saved[key].copy() for key in saved.files}

    def proof(self):
        return {
            "files": self.hashes,
            "upload_receipt_sha256": file_sha256(self.receipt),
            "upload": json.loads(self.receipt.read_text()),
            "analyzer_source_compatibility": getattr(self, "compatibility", None),
        }


def analyzer_source_proof(identity):
    """Require the reviewed analyzer implementation, beyond unchanged lens sources."""
    code = identity["code"]
    if code.get("git_dirty") or len(code.get("git_commit", "")) != 40:
        raise ValueError("Supplemental analysis requires a clean full-SHA producer")
    paths = ["scripts/workspace_jr_analyze.py"] + [
        f"src/explore_persona_space/analysis/{name}.py"
        for name in (
            "workspace_diagnostics",
            "workspace_analysis_inputs",
            "workspace_fit",
            "workspace_components",
            "workspace_calibration",
            "vectorized_mlp_skill",
        )
    ]
    hashes = {}
    for name in paths:
        prior = subprocess.check_output(["git", "show", f"{code['git_commit']}:{name}"])
        if prior != Path(name).read_bytes():
            raise ValueError(f"Supplemental analyzer implementation changed: {name}")
        hashes[name] = file_sha256(Path(name))
    return {"producer_sha": code["git_commit"], "source_sha256": hashes}


def bind_analysis(source, phase, native, identity):
    complete = source.json("analysis_complete.json")
    proof = source.json("input_proof.json")
    if (
        complete["status"] != "complete"
        or complete["phase"] != phase
        or complete["input_proof_sha256"] != source.hashes["input_proof.json"]
    ):
        raise ValueError("Supplemental source is incomplete or has changed input evidence")
    validate_producer(complete["identity"], identity, native_ancestor=True)
    source.compatibility = analyzer_source_proof(complete["identity"])
    if (
        proof["identity"] != complete["identity"]
        or proof["sources"]["source_identity"] != native["manifest"]["identity"]
    ):
        raise ValueError("Supplemental analysis has a different native fit producer")
    for key, filename in (
        ("results_sha256", "results.json"),
        ("input_manifest_sha256", "input_manifest.json"),
        ("per_example_sha256", "per_example.npz"),
    ):
        if proof["sources"][key] != file_sha256(native["fit"] / filename):
            raise ValueError("Supplemental source was derived from another full-answer fit")


def learning_results(source, native, common, config):
    """Reuse all fitted recipes; bootstrap their saved predictions on primary rows."""
    report = source.json("curves/learning_curves.json")
    if report["full_fit_result_sha256"] != file_sha256(native["fit"] / "results.json") or report[
        "full_fit_recipe_sha256"
    ] != file_sha256(native["fit"] / "mlp/selection.json"):
        raise ValueError("Learning curves have a different full-data recipe or fit")
    if report["shared_test_ids"] != native["ids"]:
        raise ValueError("Learning curves changed the original test population")
    if [cell["fraction"] for cell in report["cells"]] != config["fit"][
        "learning_curve_train_fractions"
    ]:
        raise ValueError("Learning curve coverage differs from the frozen fractions")
    fingerprints = native["result"]["input_fingerprints"]
    train_ids = report["cells"][-1]["train_ids"]
    if (
        content_sha256(train_ids) != fingerprints["train"]["ids_sha256"]
        or len(train_ids) != native["result"]["split_counts"]["train"]
        or content_sha256(report["shared_validation_ids"])
        != fingerprints["validation"]["ids_sha256"]
    ):
        raise ValueError("Learning curves changed the frozen full train/validation IDs")
    full_recipe = json.loads((native["fit"] / "mlp/selection.json").read_text())
    result = {key: value for key, value in report.items() if key != "cells"}
    result.update(primary_context_ids=common, cells=[])
    draws = {}
    for cell in report["cells"]:
        expected_count = int(len(train_ids) * cell["fraction"])
        if (
            type(cell["n_train"]) is not int
            or cell["n_train"] != expected_count
            or cell["train_ids"] != train_ids[:expected_count]
        ):
            raise ValueError(
                "Learning curve count or training prefix differs from the frozen fraction"
            )
        if cell["reused_full_fit"]:
            if cell["fraction"] != 1.0:
                raise ValueError("Only the full training point may reuse the original fit")
            fitted = native
        else:
            relative = f"curves/train-{cell['n_train']}"
            folder = source.root / relative
            saved = source.json(f"{relative}/results.json")
            source.path(f"{relative}/per_example.npz")
            recipe = source.json(f"{relative}/mlp/selection.json")
            if (
                saved["fit_contract"] != native["result"]["fit_contract"]
                or saved["mlp_selection_sha256"] != source.hashes[f"{relative}/mlp/selection.json"]
            ):
                raise ValueError("Learning curve fit contract or selected recipe changed")
            if (
                saved["input_fingerprints"]["test"]
                != native["result"]["input_fingerprints"]["test"]
            ):
                raise ValueError("Learning curve test activations differ from the main fit")
            validate_learning_fit(saved, recipe, full_recipe, cell, fingerprints, config)
            with np.load(folder / "per_example.npz", allow_pickle=False) as arrays:
                if arrays["context_ids"].tolist() != native["ids"]:
                    raise ValueError("Learning prediction context IDs differ from the full fit")
            fitted = {"fit": folder, "result": saved}
        targets, predictions = read_arrays(fitted)
        targets, predictions = align_arrays(native["ids"], common, targets, predictions)
        settings = config["statistics"]["bootstrap"]
        boot = paired_context_bootstrap(
            targets,
            predictions,
            common,
            n_bootstrap=settings["draws"],
            seed=config["seed"],
            confidence=settings["confidence"],
            contrasts=workspace_gap_contrasts(),
        )
        draws.update(
            {f"n{cell['n_train']}/{key}": value for key, value in boot.pop("samples").items()}
        )
        result["cells"].append(
            {
                "n_train": cell["n_train"],
                "fraction": cell["fraction"],
                "reused_full_fit": cell["reused_full_fit"],
                "paired_bootstrap": boot,
            }
        )
    return result, draws


def validate_learning_fit(saved, recipe, full_recipe, cell, fingerprints, config):
    """Check labels, actual train/validation fingerprints and fixed MLP recipes."""
    expected_count = cell["n_train"]
    if (
        saved["input_fingerprints"]["validation"] != fingerprints["validation"]
        or saved["input_fingerprints"]["train"]["ids_sha256"] != content_sha256(cell["train_ids"])
        or saved["split_counts"]["train"] != expected_count
        or saved["input_fingerprints"]["train"]["x"]["shape"][0] != expected_count
    ):
        raise ValueError("Learning fit changed the training prefix or validation inputs")
    if (
        recipe["recipe_mode"] != "fixed_from_full_training_validation"
        or recipe["selection_reads_test_loss"] is not False
        or recipe["primary_seed"] != config["fit"]["mlp_seeds"][0]
        or set(recipe["selected"]) != set(full_recipe["selected"])
    ):
        raise ValueError("Reduced learning fit did not keep the full-data MLP recipes")
    for name, selected in recipe["selected"].items():
        if any(selected[key] != full_recipe["selected"][name][key] for key in ("hidden", "lr")):
            raise ValueError("Reduced learning fit changed the full-data MLP recipe")


def verify_readout_projection(report, arrays, targets, predictions, original_ids):
    if report["test_context_ids"] != original_ids:
        raise ValueError("Readouts use different original test rows")
    for arm in ("J", "R", "random", "pca"):
        basis = arrays[f"direction__{arm}"]
        if not np.allclose(np.linalg.norm(basis, axis=0), 1, rtol=1e-6, atol=1e-8):
            raise ValueError("Readout directions are not normalized")
        if not np.allclose(
            targets["full"] @ basis, arrays[f"target__{arm}"], rtol=1e-10, atol=1e-10
        ):
            raise ValueError("Readout targets differ from actual full-answer activations")
        for predictor, values in predictions.items():
            if not np.allclose(
                values["full"] @ basis,
                arrays[f"prediction__{predictor}__{arm}"],
                rtol=1e-10,
                atol=1e-10,
            ):
                raise ValueError("Readouts differ from the actual saved full-answer predictor")


def validate_readout_contract(report, arrays, native, config):
    """Check fixed vocabulary, training IDs, predictor coverage and matching indices."""
    required = {"ridge", "mlp", *(f"mlp_seed{seed}" for seed in config["fit"]["mlp_seeds"])}
    tokens = report["token_ids"]
    if (
        report["schema"] != "workspace-jr-direction-readouts-v1"
        or not required <= set(report["metrics"])
        or len(tokens) != len(set(tokens))
        or any(type(token) is not int or token < 0 for token in tokens)
        or content_sha256(report["training_context_ids"])
        != native["result"]["input_fingerprints"]["train"]["ids_sha256"]
    ):
        raise ValueError("Readout schema, training contexts, vocabulary or predictors differ")
    widths = {name: arrays[f"direction__{name}"].shape[1] for name in ("J", "R", "random", "pca")}
    if (
        widths["J"] != len(tokens)
        or widths["R"] != len(tokens)
        or widths["random"] != len(tokens) * config["diagnostics"]["controls_pool_multiplier"]
        or widths["pca"] != report["pca_nonzero_rank"]
    ):
        raise ValueError("Readout basis coverage differs from the fixed eligibility/control recipe")
    cosine = (arrays["direction__J"] * arrays["direction__R"]).sum(0)
    if not np.allclose(cosine, report["paired_token_cosine"], rtol=1e-6, atol=1e-8):
        raise ValueError("Corresponding token-direction cosines differ from saved bases")
    keys = {f"{arm}_vs_{control}" for arm in ("J", "R") for control in ("random", "pca")}
    if set(report["matching"]) != keys:
        raise ValueError("Readouts require every registered random/PCA matching record")
    for key, match in report["matching"].items():
        arm, control = key.split("_vs_")
        a, b = match["j_indices"], match["r_indices"]
        if (
            len(a) != len(b)
            or len(a) != len(set(a))
            or len(b) != len(set(b))
            or any(type(i) is not int or not 0 <= i < widths[arm] for i in a)
            or any(type(i) is not int or not 0 <= i < widths[control] for i in b)
            or match["log_variance_caliper"]
            != config["diagnostics"]["maximum_absolute_log_variance_mismatch"]
        ):
            raise ValueError("Readout matching violates the fixed no-replacement index contract")
        full_av = np.asarray(report["training_variance"][arm], float)
        full_bv = np.asarray(report["training_variance"][control], float)
        av, bv = full_av[a], full_bv[b]
        distance = np.abs(np.log(av) - np.log(bv))
        if (
            not np.allclose(full_av, match["j_variance"], rtol=1e-10, atol=1e-12)
            or not np.allclose(full_bv, match["r_variance"], rtol=1e-10, atol=1e-12)
            or not np.allclose(distance, match["log_variance_distance"], rtol=1e-10, atol=1e-12)
            or np.any(distance > match["log_variance_caliper"] + 1e-12)
        ):
            raise ValueError("Readout matching differs from its training variance evidence")


def model_supplement(role, entries, native, common, identity, config, out):
    diagnostics, learning = Source(entries["diagnostics"]), Source(entries["learning_curves"])
    bind_analysis(diagnostics, "diagnostics", native, identity)
    bind_analysis(learning, "learning-curves", native, identity)
    original_targets, original_predictions = read_arrays(native, include_baselines=True)
    targets, predictions = align_arrays(
        native["ids"], common, original_targets, original_predictions
    )
    references, reference_samples = mapping_references(targets, predictions, common, config)
    np.savez(out / f"{role}_mapping_bootstrap.npz", **reference_samples)
    agreement = agreement_statistics(
        targets, common, native["result"]["decomposition_agreement"]["training_fixed_norm_floor"]
    )
    report = diagnostics.json("direction_readouts.json")
    arrays = diagnostics.arrays("direction_arrays.npz")
    validate_readout_contract(report, arrays, native, config)
    verify_readout_projection(report, arrays, original_targets, original_predictions, native["ids"])
    direction, samples = paired_readouts(report, arrays, common, config)
    np.savez(out / f"{role}_direction_bootstrap.npz", **samples)
    noise_report = diagnostics.json("noise_test.json")
    if (
        noise_report["context_ids"] != native["ids"]
        or noise_report["seeds"] != config["generation"]["seeds"]
        or noise_report["trigger_threshold"] != 0.1
    ):
        raise ValueError("Sampling noise source differs from the registered K5 observations")
    noise = paired_noise(noise_report, diagnostics.arrays("noise_test_arrays.npz"), targets, common)
    curves, samples = learning_results(learning, native, common, config)
    np.savez(out / f"{role}_learning_bootstrap.npz", **samples)
    return {
        "agreement": agreement,
        "readouts": direction,
        "noise": noise,
        "learning_curves": curves,
        "mapping_references": references,
        "all_captured_test_noise_trigger": noise_report["higher_k_trigger"],
        "sources": {
            "diagnostics": diagnostics.proof(),
            "learning_curves": learning.proof(),
            "native_fit": native["proof"],
        },
    }


def main():
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
    manifest, config = json.loads(args.manifest.read_text()), load_workspace_jr_config(args.config)
    comparison = Source(manifest["comparison"])
    complete, coverage = (
        comparison.json("comparison_complete.json"),
        comparison.json("coverage.json"),
    )
    if (
        complete["status"] != "complete"
        or coverage["status"] != "complete_grid"
        or coverage["expected_cells"] != 48
        or coverage["realized_cells"] != 48
        or coverage["missing_cells"]
        or set(complete["scopes"]) != {"primary", "comparison", "cross_model"}
        or set(manifest["models"]) != {"primary", "comparison"}
    ):
        raise ValueError("Supplement requires both models and the complete paired comparison")
    source_manifest = comparison.json("input_manifest.json")
    if comparison.hashes["input_manifest.json"] != complete["input_manifest_sha256"]:
        raise ValueError("Completed comparison source manifest changed")
    common = comparison.json("primary_scoring_cohort.json")["common_context_ids"]
    proof = comparison.json("source_proof.json")
    completion = comparison.json("completion_cohort.json")
    if (
        comparison.hashes["completion_cohort.json"] != proof["completion_cohort"]["report_sha256"]
        or completion["status"] != "complete"
        or not set(common) <= set(completion["joint_complete_context_ids"])
    ):
        raise ValueError("Supplementary primary cohort differs from verified completed rollouts")
    for scope in ("primary", "comparison", "cross_model"):
        report = comparison.json(f"{scope}/comparisons.json")
        if (
            report["context_ids"] != common
            or comparison.hashes[f"{scope}/comparisons.json"]
            != complete["scopes"][scope]["report_sha256"]
        ):
            raise ValueError("Supplementary main comparison scopes have different cohorts")
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "input_manifest.json", manifest)
    models = {}
    for role, entries in manifest["models"].items():
        identity = run_identity(args.config, args.selection, role)
        report = comparison.json(f"{role}/comparisons.json")
        if (
            report["context_ids"] != common
            or comparison.hashes[f"{role}/comparisons.json"]
            != complete["scopes"][role]["report_sha256"]
        ):
            raise ValueError("Supplemental cohort differs from the completed main comparison")
        entry = next(
            cell
            for cell in source_manifest["cells"]
            if (cell["role"], cell["kind"], cell["k"], cell["rotation"])
            == (role, "observed", 10, None)
        )
        native = read_cell(entry, config, identity, json.loads(args.selection.read_text()))
        if native["proof"] != proof["cells"][cell_key(role, "observed", 10, None)]:
            raise ValueError("Native fit source differs from the completed comparison")
        models[role] = model_supplement(role, entries, native, common, identity, config, args.out)
        save_json(args.out / f"{role}_supplement.json", finite_json(models[role]))
        print(f"Supplement complete role={role} paired_contexts={len(common)}", flush=True)
    save_json(
        args.out / "supplement.json",
        finite_json(
            {
                "status": "complete",
                "context_ids": common,
                "models": models,
                "comparison_source": comparison.proof(),
                "scoring_population": "joint two-model complete final K5 test cohort",
                "analysis_identity": {
                    role: run_identity(args.config, args.selection, role) for role in models
                },
                "implementation_sha256": {
                    name: file_sha256(Path(name))
                    for name in (
                        "scripts/workspace_jr_supplement.py",
                        "src/explore_persona_space/analysis/workspace_supplement.py",
                    )
                },
            }
        ),
    )
    save_json(
        args.out / "supplement_complete.json",
        {
            "status": "complete",
            "report_sha256": file_sha256(args.out / "supplement.json"),
            "input_manifest_sha256": file_sha256(args.out / "input_manifest.json"),
        },
    )


if __name__ == "__main__":
    main()
