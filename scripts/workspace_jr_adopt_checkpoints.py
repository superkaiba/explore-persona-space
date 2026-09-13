#!/usr/bin/env python3
"""Adopt verified whole-split caches while the original producer retains phase ownership."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
import time
from pathlib import Path

from explore_persona_space.analysis.workspace_artifacts import validate_coverage, validate_producer
from explore_persona_space.analysis.workspace_checkpoint_cache import (
    publish_checkpoint,
    read_checkpoint,
    regular_path,
    validate_checkpoint,
)
from explore_persona_space.analysis.workspace_runtime import (
    content_sha256,
    file_sha256,
    load_workspace_jr_config,
    save_json,
)
from workspace_jr_checkpoint_parity import (
    CONFIG_SHA,
    PRODUCER,
    READINESS_SHA,
    SELECTION_SHA,
    producer_identity,
    unchanged_sources,
)

SUPERVISOR_SHA = "7a43a6fe6255f9201034929757b763c2723ec66de1201e141dbd614cfa206820"
LAUNCHER_SHA = "bc89af889461d5c648ebdb78bada95bbd260375a09610c75af891775107414b6"


def normalized_foreign_plan(original, out):
    """Only storage location and optional local source hints may change."""
    result = copy.deepcopy(original)
    result["out"] = out
    for source in result["sources"]:
        source.pop("local_root", None)
    return result


def receipt_reader(receipt, stage_root, stage):
    """Stage source bytes into an isolated tree, never into the active producer root."""
    stage._check_upload(receipt)
    stage_root.mkdir(parents=True, exist_ok=True)

    def verified(relative):
        path = regular_path(stage_root, relative, create_parents=True)
        stage.stage_file(receipt, relative, path)
        if file_sha256(path) != receipt["verified_sha256"][relative]:
            raise ValueError("Staged adoption input changed")
        return path

    return verified


def verify_parity(plan, staging, stage):
    """Require an uploaded, exact-parity verifier result for this rotation."""
    proof = plan["parity"]
    read = receipt_reader(proof["upload"], staging / "parity", stage)
    report_path = read("parity_report.json")
    complete = json.loads(read("complete.json").read_text())
    report = json.loads(report_path.read_text())
    verifier = report["verifier_identity"]
    expected = producer_identity()
    for key in ("config_sha256", "selection_sha256", "model_role", "versions"):
        if verifier[key] != expected[key]:
            raise ValueError(f"Parity verifier provenance differs: {key}")
    code = verifier["code"]
    if (
        len(code["git_commit"]) != 40
        or any(c not in "0123456789abcdef" for c in code["git_commit"])
        or code["git_dirty"] is not False
        or code["git_argv0_state"] != "tracked"
        or code["git_argv0_path"] != "scripts/workspace_jr_checkpoint_parity.py"
        or code["phase"] != "workspace-jr"
    ):
        raise ValueError("Parity verifier must have clean tracked full-SHA provenance")
    if (
        complete != {"status": "complete", "parity_report_sha256": file_sha256(report_path)}
        or report["status"] != "exact_numerical_parity"
        or report["rotation"] != plan["rotation"]
        or report["k"] != [5, 10, 25]
        or report["atol"] != 0
        or report["rtol"] != 0
        or report["verifier_script_sha256"]
        != file_sha256(Path(__file__).with_name("workspace_jr_checkpoint_parity.py"))
        or report["cache_helper_sha256"]
        != file_sha256(Path("src/explore_persona_space/analysis/workspace_checkpoint_cache.py"))
        or report["numerical_source_sha256"] != unchanged_sources()
    ):
        raise ValueError("Missing exact matching parity proof")
    staging_path = read("staging_complete.json")
    reference_plan_path = read("plan.json")
    reference_plan = json.loads(reference_plan_path.read_text())
    staging_report = json.loads(staging_path.read_text())
    selection = json.loads(Path("docs/exploratory_workspace_jr/selected_contexts.json").read_text())
    prompt = selection["subsets"]["main_train"][0]["prompt_sha256"]
    if (
        report["staging_sha256"] != file_sha256(staging_path)
        or file_sha256(reference_plan_path) != proof["plan_sha256"]
        or report["prompt_sha256"] != prompt
        or reference_plan["prompt_sha256"] != prompt
        or staging_report["plan_sha256"] != proof["plan_sha256"]
        or staging_report["producer_identity"] != producer_identity()
    ):
        raise ValueError("Parity reference is not the frozen first training context")
    if file_sha256(read("parity_replay.pt")) != report["parity_replay_sha256"]:
        raise ValueError("Parity replay evidence changed")
    required = reference_plan["required_files"]
    if len(required) != len(set(required)) or staging_report["required_files"] != {
        rel: reference_plan["reference_upload"]["verified_sha256"][rel] for rel in required
    }:
        raise ValueError("Parity staging differs from its immutable reference upload")
    terminal = json.loads(read(proof["terminal"]).read_text())
    if terminal["exit_code"] != 0 or terminal["phase"] != "complete":
        raise ValueError("Parity execution did not complete successfully")
    return {
        "upload": proof["upload"],
        "parity_report_sha256": file_sha256(report_path),
        "terminal": proof["terminal"],
        "reference_plan": reference_plan,
        "dictionary_sha256": {
            rel: staging_report["required_files"][rel]
            for rel in ("dictionaries/manifest.json", "dictionaries/J.pt", "dictionaries/R.pt")
        },
        "execution_worker": report["execution_worker"],
    }


def verify_split(
    original, source_out, terminal_relative, read, receipt, staged_root, rotation, parity_proof
):
    """Verify the completed source phase and collect every semantically valid cache file."""
    if (
        original["role"] != "comparison"
        or original["rotation"] != rotation
        or original["source_sha"] != PRODUCER
        or original["readiness_sha256"] != READINESS_SHA
        or original["subset"] not in ("main_validation", "main_test")
    ):
        raise ValueError("Unexpected original producer plan")
    subset = original["subset"]
    operation = f"decomposition_operations/decompose-{subset}-rotation{rotation}"
    operation_contract = json.loads(read(f"{operation}/contract.json").read_text())
    if operation_contract != {
        "plan": normalized_foreign_plan(original, source_out),
        "supervisor_sha256": SUPERVISOR_SHA,
        "launcher_sha256": LAUNCHER_SHA,
    }:
        raise ValueError("Foreign computation changed the original recipe")
    terminal_path = Path(terminal_relative)
    if (
        terminal_path.parent.parent.parent.as_posix() != operation
        or terminal_path.name != "exit.json"
    ):
        raise ValueError("Unexpected foreign decomposition terminal")
    terminal = json.loads(read(terminal_relative).read_text())
    if terminal["exit_code"] != 0 or terminal["phase"] != "complete":
        raise ValueError("Foreign decomposition did not complete")
    attempt = terminal_path.parent.as_posix()
    if (
        file_sha256(read(f"{attempt}/supervisor.py")) != SUPERVISOR_SHA
        or file_sha256(read(f"{attempt}/launcher.sh")) != LAUNCHER_SHA
    ):
        raise ValueError("Foreign execution wrapper differs")
    originals = {}
    for source in original["sources"]:
        for relative, digest in source["upload"]["verified_sha256"].items():
            if relative in originals and originals[relative] != digest:
                raise ValueError("Original source uploads conflict")
            originals[relative] = digest

    def original_input(relative):
        if receipt["verified_sha256"][relative] != originals[relative]:
            raise ValueError(f"Foreign input differs from original frozen input: {relative}")
        return read(relative)

    for relative, digest in parity_proof["dictionary_sha256"].items():
        if originals[relative] != digest:
            raise ValueError("Parity dictionaries differ from the original queued computation")

    identity = producer_identity()
    manifest = json.loads(original_input("dictionaries/manifest.json").read_text())
    validate_producer(manifest["identity"], identity)
    if manifest["pilot_only"]:
        raise ValueError("Pilot dictionary cannot serve main checkpoint adoption")
    for arm in ("J", "R"):
        if file_sha256(original_input(f"dictionaries/{arm}.pt")) != manifest["arms"][arm]["sha256"]:
            raise ValueError("Dictionary bytes differ")
    selection = json.loads(Path("docs/exploratory_workspace_jr/selected_contexts.json").read_text())
    count = 128 if subset == "main_validation" else 256
    ids = [row["prompt_sha256"] for row in selection["subsets"][subset][:count]]
    capture_coverage = json.loads(original_input(f"captures/{subset}/coverage.json").read_text())
    included = validate_coverage(capture_coverage, ids, identity)
    coverages, contracts = {}, {}
    for k in (5, 10, 25):
        contracts[k] = {
            "identity": identity,
            "dictionaries": manifest,
            "k": k,
            "rotation": rotation,
        }
        coverage = json.loads(
            read(f"components/k{k}-rotation{rotation}/{subset}/coverage.json").read_text()
        )
        if (
            validate_coverage(coverage, ids, identity) != included
            or coverage["contract"] != contracts[k]
            or coverage["exclusions"] != capture_coverage["exclusions"]
            or set(coverage["file_sha256"]) != set(included)
        ):
            raise ValueError("Foreign decomposition coverage differs from frozen capture coverage")
        coverages[k] = coverage
    seeds = load_workspace_jr_config(Path("configs/analysis/workspace_jr.yaml"))["generation"][
        "seeds"
    ]
    files = []
    for prompt in included:
        relative = f"captures/{subset}/{prompt}.pt"
        path = original_input(relative)
        capture, digest = read_checkpoint(path, capture_coverage["file_sha256"][prompt])
        validate_producer(capture["identity"]["identity"], identity)
        if [r["seed"] for r in capture["rows"]] != seeds or any(
            r["prompt_sha256"] != prompt for r in capture["rows"]
        ):
            raise ValueError("Capture membership or seeds changed")
        generation = original_input(f"generations/{subset}/{prompt}.json")
        if capture["identity"]["generation_file_sha256"] != file_sha256(generation):
            raise ValueError("Capture generation binding changed")
        original_input(capture["identity"]["context_input_file"])
        for k in (5, 10, 25):
            rel = f"components/k{k}-rotation{rotation}/{subset}/{prompt}.pt"
            path = read(rel)
            saved, saved_sha = read_checkpoint(path, coverages[k]["file_sha256"][prompt])
            validate_checkpoint(saved, capture, digest, contracts[k], staged_root)
            files.append(
                {
                    "relative": rel,
                    "sha256": saved_sha,
                    "capture_relative": relative,
                    "capture_sha256": digest,
                    "contract": contracts[k],
                }
            )
        print(f"Validated optional cache context={prompt} subset={subset}", flush=True)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["schema"] != "workspace-jr-checkpoint-adoption-v1" or plan["rotation"] not in (
        20260914,
        20260915,
    ):
        raise ValueError("Unexpected adoption scope")
    if (
        file_sha256(Path("configs/analysis/workspace_jr.yaml")) != CONFIG_SHA
        or file_sha256(Path("docs/exploratory_workspace_jr/selected_contexts.json"))
        != SELECTION_SHA
    ):
        raise ValueError("Frozen config/selection changed")
    unchanged_sources()
    recipient, staging = Path(plan["recipient"]), Path(plan["staging"])
    if recipient == staging or recipient in staging.parents or staging in recipient.parents:
        raise ValueError("Adoption staging must be outside the active producer root")
    if recipient.name != f"comparison_main_rotation{plan['rotation']}_v1":
        raise ValueError("Unexpected recipient")
    regular_path(recipient, "components")
    staging.mkdir(parents=True, exist_ok=True)
    regular_path(staging, "inputs")
    if (staging / "publication_complete.json").exists():
        raise ValueError("Completed publication evidence is immutable")
    plan_path = staging / "plan.json"
    if plan_path.exists():
        if plan_path.read_bytes() != args.plan.read_bytes():
            raise ValueError("Adoption plan changed after staging began")
    else:
        with plan_path.open("xb") as stream:
            stream.write(args.plan.read_bytes())
    if shutil.disk_usage(staging).free < 40 * 2**30 + plan["staging_bytes_ceiling"]:
        raise ValueError("Insufficient disk for isolated checkpoint validation")
    spec = importlib.util.spec_from_file_location(
        "cache_stage", "scripts/workspace_jr_stage_calibration.py"
    )
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    parity_proof = verify_parity(plan, staging, stage)
    source_out = f"/workspace/workspace_jr/comparison_main_precompute_rotation{plan['rotation']}_v1"
    upload = plan["source_upload"]
    if upload["prefix"] != f"exploratory_workspace_jr/20260912/{Path(source_out).name}":
        raise ValueError("Unexpected foreign output upload")
    inputs = staging / "inputs"
    read = receipt_reader(upload, inputs, stage)
    worker_proof = json.loads(read("cache_execution_provenance.json").read_text())
    if (
        worker_proof["execution_worker"] != parity_proof["execution_worker"]
        or worker_proof["parity_report_sha256"] != parity_proof["parity_report_sha256"]
        or worker_proof["producer_sha"] != PRODUCER
        or worker_proof["rotation"] != plan["rotation"]
    ):
        raise ValueError("Foreign execution is not bound to the worker that passed exact parity")
    if {p["original"]["subset"] for p in plan["splits"]} != {"main_validation", "main_test"} or len(
        plan["splits"]
    ) != 2:
        raise ValueError("Adoption requires both complete frozen future splits")
    files = []
    for split in plan["splits"]:
        original = split["original"]
        original_path = Path(split["original_plan_path"])
        if (
            file_sha256(original_path) != split["original_plan_sha256"]
            or json.loads(original_path.read_text()) != original
            or original["out"] != str(recipient)
        ):
            raise ValueError("Original queued producer plan changed")
        files.extend(
            verify_split(
                original,
                source_out,
                split["terminal"],
                read,
                upload,
                inputs,
                plan["rotation"],
                parity_proof,
            )
        )
    lineage = {
        "schema": plan["schema"],
        "plan": plan,
        "parity": parity_proof,
        "validated_files": files,
        "at_epoch": time.time(),
        "script_sha256": file_sha256(Path(__file__)),
        "note": "Optional tensor caches only; original producer owns coverage, terminal and final hashes.",
    }
    save_json(staging / "verified.json", lineage)
    if args.verify_only:
        print("All foreign checkpoints verified; no active-root files published", flush=True)
        return
    event_name = content_sha256(lineage)
    event = regular_path(
        recipient, f"checkpoint_adoptions/{event_name}/lineage.json", create_parents=True
    )
    if event.exists():
        raise ValueError("An adoption event must be unique")
    save_json(event, lineage)  # Immutable lineage precedes every component-file publication.
    for index, item in enumerate(files):
        relative = item["relative"]
        staged = regular_path(inputs, relative)
        saved, _ = read_checkpoint(staged, item["sha256"])
        capture, _ = read_checkpoint(inputs / item["capture_relative"], item["capture_sha256"])

        def validate(value):
            validate_checkpoint(value, capture, item["capture_sha256"], item["contract"], inputs)

        validate(saved)
        destination = regular_path(recipient, relative, create_parents=True)
        result = publish_checkpoint(staged, destination, validate)
        save_json(
            event.parent / f"{index:04d}.json",
            {"relative": relative, **result, "at_epoch": time.time()},
        )
    save_json(
        staging / "publication_complete.json",
        {
            "status": "complete",
            "files": len(files),
            "lineage_relative": str(event.relative_to(recipient)),
            "lineage_sha256": file_sha256(event),
        },
    )
    print(
        "Optional checkpoint publication complete; original queued producer will validate and finalize",
        flush=True,
    )


if __name__ == "__main__":
    main()
