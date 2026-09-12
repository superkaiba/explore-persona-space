"""Bind main execution to complete, reviewed calibration and pilot evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

from explore_persona_space.analysis.workspace_artifacts import validate_coverage, validate_producer
from explore_persona_space.analysis.workspace_calibration import validate_calibration_order
from explore_persona_space.analysis.workspace_runtime import (
    content_sha256,
    file_sha256,
    validate_token_manifest,
)


def _evidence(root: Path, reference: dict) -> Path:
    """Read only an explicitly named, byte-bound file within the readiness bundle."""
    relative = Path(reference["file"])
    if relative.is_absolute() or any(part.startswith(".") for part in relative.parts):
        raise ValueError("Readiness evidence must use nonhidden paths within its bundle")
    root = root.resolve(strict=True)
    path = root / relative
    if any(
        (root.joinpath(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)
    ):
        raise ValueError("Readiness evidence symlinks are forbidden within its bundle")
    if (
        not path.resolve(strict=True).is_relative_to(root)
        or not path.is_file()
        or file_sha256(path) != reference["sha256"]
    ):
        raise ValueError(f"Readiness evidence bytes differ: {relative}")
    return path


def _unchanged_analyzer(producer, source_paths):
    """Prove that the source which computed the evidence is still the reviewed code."""
    import subprocess

    code = producer["code"]
    sha = code["git_commit"]
    if len(sha) != 40 or code["git_dirty"] is not False:
        raise ValueError("Readiness evidence requires a clean exact producer commit")
    for source in source_paths:
        prior = subprocess.run(
            ["git", "show", f"{sha}:{source}"], check=True, capture_output=True
        ).stdout
        if prior != Path(source).read_bytes():
            raise ValueError(f"Readiness analyzer changed after producing evidence: {source}")


def _check_calibration(paths, reports, config, config_path, selection_path, identity, root):
    """Require ordered full membership and the exact means/readouts being reviewed."""
    tokens, validation, calibration = (
        reports[key] for key in ("tokens", "native_validation", "calibration")
    )
    selection = json.loads(selection_path.read_text())
    validate_calibration_order(tokens, selection)
    validate_token_manifest(
        tokens,
        config_path=config_path,
        selection_path=selection_path,
        config=config,
        role=identity["model_role"],
    )
    validate_producer(validation["identity"], identity, native_ancestor=True)
    validate_producer(calibration["identity"], identity, native_ancestor=True)
    _unchanged_analyzer(
        calibration["identity"],
        (
            "scripts/workspace_jr_calibration.py",
            "src/explore_persona_space/analysis/workspace_calibration.py",
        ),
    )
    frozen = [row["prompt_sha256"] for row in tokens["rows"]]
    if (
        not validation["forward_bit_identical"]
        or not validation["hook_bit_identical"]
        or validation["ordinary_numerical_validation"]["status"] != "passed"
        or validation["token_manifest_sha256"] != content_sha256(tokens)
    ):
        raise ValueError(
            "Readiness requires passed native J numerics and unchanged R forward outputs"
        )
    if (
        calibration["status"] != "calibration_diagnostics_complete"
        or not calibration["full_calibration_membership"]
        or calibration["realized_prompts"] != len(frozen)
        or calibration["valid_prompts"] != len(frozen)
        or calibration["full_group"] != f"first_{len(frozen)}"
        or calibration["groups"][calibration["full_group"]] != frozen
        or [row["prompt_sha256"] for row in calibration["source_files"]] != frozen
        or calibration["token_manifest_sha256"] != content_sha256(tokens)
        or calibration["native_validation_sha256"] != file_sha256(paths["native_validation"])
        or calibration["means_sha256"] != file_sha256(paths["means"])
        or calibration["directions_sha256"] != file_sha256(paths["directions"])
        or len(calibration["readouts"]) < 8
        or calibration["eligible_token_count"] < 1
    ):
        raise ValueError("Main requires complete calibration with byte-bound means and readouts")
    if not {"first_32", "first_64", "even", "odd"}.issubset(calibration["groups"]):
        raise ValueError("Required calibration stability subsets are missing")
    _check_calibration_upload(root, reports["calibration_upload"], calibration, identity)


def _check_upload(upload):
    """Validate the real persist receipt shape and its immutable Hub destination."""
    revision = upload["revision"]
    if (
        upload["repo"] != "superkaiba1/explore-persona-space-data"
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
        or not upload["prefix"].startswith("exploratory_workspace_jr/")
        or upload["files_verified"] != len(upload["verified_sha256"])
    ):
        raise ValueError("Upload receipt is not pinned to the declared data repository")


def _check_calibration_upload(root, aggregate, calibration, identity):
    """Bind complete interval coverage to uploaded native checkpoints and exits."""
    if aggregate["schema"] != "workspace-jr-calibration-upload-aggregate-v1":
        raise ValueError("Calibration upload aggregate schema mismatch")
    count = calibration["realized_prompts"]
    intervals = (
        [(2, 32), (32, 61), (61, 90), (90, count)]
        if identity["model_role"] == "primary"
        else [(0, count)]
    )
    workers = aggregate["workers"]
    if len(workers) != len(intervals):
        raise ValueError("Calibration upload worker coverage is incomplete")
    uploaded_indices = set()
    for rank, (worker, interval) in enumerate(zip(workers, intervals, strict=True)):
        if (worker["rank"], worker["start"], worker["stop"]) != (rank, *interval):
            raise ValueError("Calibration upload worker ownership differs")
        paths = {key: _evidence(root, worker[key]) for key in ("upload", "snapshot", "terminal")}
        upload, snapshot, terminal = (
            json.loads(paths[key].read_text()) for key in ("upload", "snapshot", "terminal")
        )
        _check_upload(upload)
        expected = set(range(*interval)) | (
            {0, 1} if identity["model_role"] == "primary" else set()
        )
        if (
            snapshot["rank"] != rank
            or snapshot["rank_interval"] != list(interval)
            or snapshot["successful_complete"] is not True
            or snapshot["terminal_receipt_included"] is not True
            or snapshot["included_prompt_indices"] != sorted(expected)
            or snapshot["paired_prompt_files"] != len(expected)
            or terminal["rank"] != rank
            or (terminal["start"], terminal["stop"]) != interval
            or type(terminal["exit_code"]) is not int
            or terminal["exit_code"] != 0
            or type(terminal["finished_at_epoch"]) is not int
            or terminal["finished_at_epoch"] <= 1789230000
        ):
            raise ValueError("Calibration requires matching successful worker terminal receipts")
        # The native source must be an audited ancestor of this exact runtime.
        _unchanged_analyzer(
            {"code": {"git_commit": snapshot["native_source_sha"], "git_dirty": False}},
            ("src/explore_persona_space/analysis/workspace_runtime.py",),
        )
        hashes = upload["verified_sha256"]
        if hashes.get("snapshot.json") != file_sha256(paths["snapshot"]) or hashes.get(
            f"rank{rank}_exit.json"
        ) != file_sha256(paths["terminal"]):
            raise ValueError("Calibration snapshot and terminal receipts lack verified uploads")
        for index in expected:
            source = calibration["source_files"][index]
            if (
                source["file"] != f"prompt-{index:04d}.pt"
                or hashes.get(f"lens_shards/{source['file']}") != source["sha256"]
            ):
                raise ValueError(
                    "Every completed calibration pair needs an exact verified upload path"
                )
        uploaded_indices.update(expected)
    if uploaded_indices != set(range(count)):
        raise ValueError("Calibration uploads do not cover every frozen prompt")


def _check_parity(paths, reports, config, identity):
    """Preserve measured historical failures; a reviewed fresh-fit revision may proceed."""
    parity, inputs = reports["parity"], reports["parity_inputs"]
    validate_producer(parity["identity"], identity, native_ancestor=True)
    _unchanged_analyzer(parity["identity"], ("scripts/workspace_jr_parity.py",))
    if (
        parity["status"] not in {"passed", "failed"}
        or parity["input_manifest_sha256"] != file_sha256(paths["parity_inputs"])
        or parity["recaptured_sha256"] != file_sha256(paths["parity_arrays"])
        or inputs["model_role"] != identity["model_role"]
        or inputs["config_sha256"] != identity["config_sha256"]
        or inputs["selection_sha256"] != identity["selection_sha256"]
        or inputs["historical_targets_sha256"] != file_sha256(paths["parity_historical_targets"])
        or len(parity["rows"]) != config["provenance_gate"]["recapture_contexts"]
        or [row["prompt_sha256"] for row in parity["rows"]]
        != [row["selection"]["prompt_sha256"] for row in inputs["rows"]]
    ):
        raise ValueError("Historical mapping recapture is not bound to these inputs")
    outcomes = []
    for key in ("x_prompt_last", "y_ans"):
        metric = parity["metrics"][key]
        if (
            not math.isfinite(metric["relative_frobenius_error"])
            or metric["relative_frobenius_error"] < 0
            or len(metric["row_cosine"]) != config["provenance_gate"]["recapture_contexts"]
            or not all(math.isfinite(value) for value in metric["row_cosine"])
        ):
            raise ValueError("Historical mapping parity thresholds failed")
        passed = (
            metric["relative_frobenius_error"]
            <= config["provenance_gate"]["maximum_relative_frobenius_error"]
            and min(metric["row_cosine"]) >= config["provenance_gate"]["minimum_row_cosine"]
        )
        if metric["passed"] is not passed:
            raise ValueError("Historical mapping parity thresholds disagree with saved status")
        outcomes.append(passed)
    if parity["status"] != ("passed" if all(outcomes) else "failed"):
        raise ValueError("Historical mapping parity thresholds disagree with overall status")
    revision = reports["protocol_revision"]
    if (
        revision["schema"] != "workspace-jr-execution-revision-v1"
        or revision["revision_id"] != "20260912-canonical-input-v1"
        or revision["main_outcomes_seen"] is not False
        or revision["config_sha256"] != identity["config_sha256"]
        or revision["selection_sha256"] != identity["selection_sha256"]
        or revision["input_capture_policy"]
        != "context_only_frozen_order_batches16_no_answer_tokens"
        or revision["predictor_policy"] != "fresh_full_answer_and_component_fits"
        or revision["historical_map_application"] != "never_to_current_native_inputs"
        or revision["historical_parity_thresholds"] != "unchanged_failures_remain_failures"
        or reports["execution_plan"]["protocol_revision_sha256"]
        != file_sha256(paths["protocol_revision"])
    ):
        raise ValueError(
            "Canonical main inputs require a reviewed revision forbidding historical-map reuse"
        )


def _check_pilot(paths, reports, identity, selection_path, config):
    """Require successful workload exit and verified upload of its actual results."""
    pilot, terminal, upload = (
        reports[key] for key in ("pilot_results", "pilot_exit", "pilot_upload")
    )
    manifest = reports["pilot_input_manifest"]
    validate_producer(manifest["identity"], identity, native_ancestor=True)
    contract = manifest["contract"]
    if (
        contract["identity"] != manifest["identity"]
        or contract["k"] != 10
        or contract["rotation"] is not None
    ):
        raise ValueError("Main readiness requires the primary k10 unrotated pilot cell")
    dictionaries = contract["dictionaries"]
    validate_producer(dictionaries["identity"], manifest["identity"])
    if dictionaries["pilot_only"] is not True or set(dictionaries["arms"]) != {"J", "R"}:
        raise ValueError("Pilot must use its matched native pilot dictionaries")
    binding = reports["pilot_binding"]
    if (
        binding["schema"] != "workspace-jr-pilot-binding-v1"
        or binding["status"] != "reviewed_legacy_provenance_bridge"
        or binding["producer_sha"] != manifest["identity"]["code"]["git_commit"]
        or binding["cell"] != "k10-rotationNone"
        or binding["generation_seeds"] != config["generation"]["seeds"]
        or binding["dictionary_manifest_sha256"] != content_sha256(dictionaries)
        or any(
            binding[f"{key}_sha256"] != file_sha256(paths[key])
            for key in ("pilot_results", "pilot_input_manifest", "pilot_exit")
        )
    ):
        raise ValueError(
            "Pilot legacy provenance bridge does not bind these exact inputs and results"
        )
    selection = json.loads(selection_path.read_text())
    for split in ("train", "validation", "test"):
        if manifest["coverage"][split]["contract"] != contract:
            raise ValueError("Pilot coverage comes from a different component cell")
        included = validate_coverage(
            manifest["coverage"][split],
            [row["prompt_sha256"] for row in selection["subsets"][f"pilot_{split}"]],
            manifest["identity"],
        )
        if pilot["split_counts"][split] != len(included):
            raise ValueError("Pilot result counts differ from exact frozen coverage")
    if (
        terminal["exit_code"] != 0
        or terminal["phase"] != "complete"
        or terminal["finished_at_epoch"] <= 0
    ):
        raise ValueError("Pilot requires a successful fresh terminal receipt")
    _check_upload(upload)
    locations = {
        "pilot_results": "fits/pilot/k10-rotationNone/results.json",
        "pilot_input_manifest": "fits/pilot/k10-rotationNone/input_manifest.json",
        "pilot_exit": "pilot_exit.json",
    }
    if any(
        upload["verified_sha256"].get(location) != file_sha256(paths[key])
        for key, location in locations.items()
    ):
        raise ValueError("Pilot result and final exit receipt require exact verified upload paths")
    for predictor in ("ridge", "mlp"):
        if set(pilot["metrics"][predictor]) != {"full", "J", "restJ", "R", "restR"}:
            raise ValueError("Pilot must actually fit both predictors and both decompositions")
    for arm in ("J", "R"):
        error = pilot["reconstruction"][arm]["max_abs_reconstruction_error"]
        if not math.isfinite(error) or error > 1e-5:
            raise ValueError("Pilot component reconstruction failed")


def validate_main_readiness(path, config, *, config_path, selection_path, identity):
    """Check evidence; the saved independent scientific review supplies the decision.

    Successful calculation is never an automatic convergence verdict. This
    function cannot manufacture the required review or approve partial lenses.
    """
    path = Path(path)
    readiness = json.loads(path.read_text())
    if readiness["schema"] != "workspace-jr-main-readiness-v1":
        raise ValueError("Unsupported main readiness schema")
    for key in ("config_sha256", "selection_sha256", "model_role"):
        if readiness[key] != identity[key]:
            raise ValueError(f"Readiness differs from requested {key}")
    names = {
        "tokens",
        "native_validation",
        "calibration",
        "means",
        "directions",
        "parity",
        "parity_inputs",
        "parity_arrays",
        "parity_historical_targets",
        "pilot_results",
        "pilot_exit",
        "pilot_upload",
        "pilot_input_manifest",
        "pilot_binding",
        "calibration_upload",
        "execution_plan",
        "protocol_revision",
        "review",
    }
    if set(readiness["evidence"]) != names:
        raise ValueError("Main readiness evidence is incomplete or has unknown entries")
    paths = {key: _evidence(path.parent, value) for key, value in readiness["evidence"].items()}
    reports = {
        key: json.loads(value.read_text())
        for key, value in paths.items()
        if key not in {"means", "directions", "parity_arrays", "parity_historical_targets"}
    }
    _check_calibration(paths, reports, config, config_path, selection_path, identity, path.parent)
    _check_parity(paths, reports, config, identity)
    _check_pilot(paths, reports, identity, selection_path, config)
    review = reports["review"]
    if (
        review["decision"] != "approved_for_exploratory_main"
        or review["evidence_digest"]
        != content_sha256(
            {
                **{
                    key: readiness[key]
                    for key in ("model_role", "config_sha256", "selection_sha256")
                },
                "evidence": {
                    key: value for key, value in readiness["evidence"].items() if key != "review"
                },
            }
        )
        or review["calibration_report_sha256"] != file_sha256(paths["calibration"])
        or review["parity_report_sha256"] != file_sha256(paths["parity"])
        or review["pilot_results_sha256"] != file_sha256(paths["pilot_results"])
        or review["pilot_producer_sha"]
        != reports["pilot_input_manifest"]["identity"]["code"]["git_commit"]
        or not review["reviewer"]
        or len(review["reasoning"]) < 40
    ):
        raise ValueError("Independent review does not approve these exact evidence files")
    execution = reports["execution_plan"]
    selection = json.loads(selection_path.read_text())
    if execution["main_outcomes_seen_before_freeze"] is not False or not execution["scale_reason"]:
        raise ValueError("Execution scale needs a pre-main rationale")
    for split in ("train", "validation", "test"):
        count = execution["main_counts"][split]
        maximum = min(
            config["sampling"]["main_maximum"][split], len(selection["subsets"][f"main_{split}"])
        )
        if type(count) is not int or not 2 <= count <= maximum:
            raise ValueError("Main sample size is outside its frozen split ceiling")
    return {
        "readiness_sha256": file_sha256(path),
        "paths": paths,
        "reports": reports,
        "execution": execution,
    }
