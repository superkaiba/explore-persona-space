#!/usr/bin/env python3
"""Check one completed native fit against the frozen paired higher-K noise criterion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from workspace_jr_compare import (  # noqa: E402
    bind_fitted_generation_sources,
    read_arrays,
    read_cell,
)
from workspace_jr_supplement import Source, bind_analysis  # noqa: E402

from explore_persona_space.analysis.workspace_artifacts import validate_producer  # noqa: E402
from explore_persona_space.analysis.workspace_comparison import align_arrays  # noqa: E402
from explore_persona_space.analysis.workspace_completion import (  # noqa: E402
    joint_completion_cohort,
    scoring_implementation,
)
from explore_persona_space.analysis.workspace_fit import finite_json  # noqa: E402
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    file_sha256,
    load_workspace_jr_config,
    run_identity,
    save_json,
)
from explore_persona_space.analysis.workspace_supplement import paired_noise, row_indices  # noqa: E402

SUBSET_PATH = Path("docs/exploratory_workspace_jr/higher_k_subset_20260913.json")
SUBSET_SHA256 = "1499b17a2fdd84c8058bf36886e8272e66b9755cb6992d5b80ea764978b055bb"


def frozen_cohort(source, config_path, selection_path):
    """Bind the previously audited, outcome-blind cohort and conditional sample."""
    if file_sha256(SUBSET_PATH) != SUBSET_SHA256:
        raise ValueError("The pre-outcome higher-K subset declaration changed")
    subset = json.loads(SUBSET_PATH.read_text())
    report = source.json("completion_cohort.json")
    marker = source.json("cohort_complete.json")
    if (
        marker["status"] != "complete"
        or marker["report_sha256"] != source.hashes["completion_cohort.json"]
        or source.hashes["completion_cohort.json"] != subset["completion_report_sha256"]
        or report["status"] != "complete"
        or report["schema"] != "workspace-jr-complete-test-cohort-v1"
        or report["config_sha256"] != file_sha256(config_path)
        or subset["config_sha256"] != report["config_sha256"]
        or report["selection_sha256"] != file_sha256(selection_path)
        or subset["selection_sha256"] != report["selection_sha256"]
        or report["implementation_sha256"] != scoring_implementation()
    ):
        raise ValueError("Completion ledger differs from its frozen audited source")
    joint = joint_completion_cohort(report["by_role"])
    if any(report[key] != value for key, value in joint.items()):
        raise ValueError("Joint completion membership differs from its per-model records")
    config = load_workspace_jr_config(config_path)
    selection = json.loads(selection_path.read_text())
    planned = [
        row["prompt_sha256"]
        for row in selection["subsets"]["main_test"][: len(joint["planned_context_ids"])]
    ]
    eligible = set(joint["joint_complete_context_ids"])
    if (
        planned != joint["planned_context_ids"]
        or subset["eligible_contexts"] != len(eligible)
        or subset["context_ids"]
        != [context for context in planned if context in eligible][
            : config["sampling"]["higher_k"]["contexts"]
        ]
    ):
        raise ValueError("Higher-K sample differs from the original frozen test order")
    for role in ("primary", "comparison"):
        validate_producer(
            report["analysis_identity"][role],
            run_identity(config_path, selection_path, role),
            native_ancestor=True,
        )
    return report, subset


def decision_summary(noise):
    """A single negative model result cannot settle the two-model OR criterion."""
    required = {"full", "J", "restJ", "R", "restR"}
    if set(noise["components"]) != required or noise["trigger_threshold"] != 0.1:
        raise ValueError("Noise decision requires all five registered native k10 targets")
    triggered, undefined = [], []
    for name, cell in noise["components"].items():
        fraction = cell["noise_fraction"]
        if fraction is None:
            undefined.append(name)
        elif not np.isfinite(fraction) or fraction < 0:
            raise ValueError("Noise fraction is invalid")
        elif fraction > 0.1:
            triggered.append(name)
        if cell["higher_k_trigger"] != (name in triggered):
            raise ValueError("Saved target trigger differs from the declared strict threshold")
    if noise["higher_k_trigger"] != bool(triggered):
        raise ValueError("Saved model trigger differs from its component flags")
    return {
        "higher_k_required_by_this_model": bool(triggered),
        "triggered_targets": triggered,
        "undefined_targets": undefined,
        "two_model_decision": (
            "required_by_this_model" if triggered else "await_other_model_or_final_supplement"
        ),
        "scope": "one model; a negative result does not clear the other model's noise criterion",
    }


def run(manifest, out, config_path, selection_path):
    """Read an uploaded native fit/diagnostic pair and preserve exact cohort evidence."""
    if out.exists():
        raise ValueError("Noise decision requires a fresh output directory")
    role = manifest["role"]
    if role not in ("primary", "comparison"):
        raise ValueError("Noise decision requires one of the two frozen model roles")
    entry = manifest["native_fit"]
    if any(
        entry[key] != value
        for key, value in {"role": role, "kind": "observed", "k": 10, "rotation": None}.items()
    ):
        raise ValueError("Noise decision must use this model's native observed k10 fit")
    config = load_workspace_jr_config(config_path)
    identity = run_identity(config_path, selection_path, role)
    native = read_cell(entry, config, identity, json.loads(selection_path.read_text()))
    native_source = Source({"root": entry["root"], "upload_receipt": entry["upload_receipt"]})
    terminal = native_source.json(entry["terminal_relative"])
    if terminal["cell"] != "k10-rotationNone" or terminal["results_sha256"] != file_sha256(
        native["fit"] / "results.json"
    ):
        raise ValueError("Native fit terminal does not bind the actual k10 result")
    cohort_source = Source(manifest["completion_cohort"])
    cohort, subset = frozen_cohort(cohort_source, config_path, selection_path)
    if native["manifest"]["identity"] != cohort["by_role"][role]["producer_identity"]:
        raise ValueError("Native fit and completion ledger have different producers")
    generation_binding = bind_fitted_generation_sources(
        native, cohort["by_role"][role], config["generation"]["seeds"]
    )
    diagnostics = Source(manifest["diagnostics"])
    bind_analysis(diagnostics, "diagnostics", native, identity)
    terminal = diagnostics.json("analysis_operations/exit.json")
    if (
        terminal["exit_code"] != 0
        or terminal["phase"] != "complete"
        or terminal["analysis_complete_sha256"] != diagnostics.hashes["analysis_complete.json"]
    ):
        raise ValueError("Diagnostic supervisor has no matching successful terminal")
    original_targets, predictions = read_arrays(native)
    common = cohort["joint_complete_context_ids"]
    targets, _ = align_arrays(native["ids"], common, original_targets, predictions)
    report = diagnostics.json("noise_test.json")
    arrays = diagnostics.arrays("noise_test_arrays.npz")
    if (
        report["context_ids"] != native["ids"]
        or report["seeds"] != config["generation"]["seeds"]
        or report["trigger_threshold"] != 0.1
    ):
        raise ValueError("Diagnostic noise differs from the registered K5 fitted observations")
    noise = paired_noise(report, arrays, targets, common)
    decision = decision_summary(noise)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "input_manifest.json", manifest)
    save_json(
        out / "input_proof.json",
        {
            "identity": identity,
            "higher_k_subset_sha256": SUBSET_SHA256,
            "native_fit": native["proof"],
            "native_terminal": native_source.proof(),
            "completion_cohort": cohort_source.proof(),
            "fit_generation_binding": generation_binding,
            "diagnostics": diagnostics.proof(),
        },
    )
    save_json(out / "higher_k_subset.json", subset)
    save_json(
        out / "noise_decision.json",
        finite_json({"model_role": role, "noise": noise, "decision": decision}),
    )
    index = row_indices(report["context_ids"], common)
    np.savez(
        out / "selected_noise_arrays.npz",
        context_ids=np.asarray(common),
        **{key: value[index] for key, value in arrays.items() if key.startswith("noise")},
    )
    save_json(
        out / "noise_decision_complete.json",
        {
            "status": "complete",
            "identity": identity,
            "files_sha256": {
                path.name: file_sha256(path) for path in sorted(out.iterdir()) if path.is_file()
            },
        },
    )
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
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
    run(json.loads(args.manifest.read_text()), args.out, args.config, args.selection)
