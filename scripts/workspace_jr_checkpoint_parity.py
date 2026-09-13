#!/usr/bin/env python3
"""Replay the first frozen training context before cross-worker checkpoint reuse.

This produces a verification report only. Production components are always made
by the unchanged original pipeline, in a separate output tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import torch

from explore_persona_space.analysis.workspace_artifacts import validate_producer
from explore_persona_space.analysis.workspace_checkpoint_cache import (
    equal_tree,
    read_checkpoint,
    regular_path,
    validate_checkpoint,
)
from explore_persona_space.analysis.workspace_decomposition import (
    ValidatedDictionary,
    decompose_context_nested,
)
from explore_persona_space.analysis.workspace_lenses import rotated_dictionary
from explore_persona_space.analysis.workspace_runtime import (
    file_sha256,
    load_workspace_jr_config,
    run_identity,
    save_json,
    save_tensors,
)

PRODUCER = "244f89eb8347361484413e803565f1e639255a8c"
CONFIG_SHA = "db6c68fe9680ed2be5cc2eb4064019a01a5b1f2d460099a683f439817d61f5d9"
SELECTION_SHA = "11209d75c89345223e87641d74d273ba37ddcd7f1e608f0c59d39be0607d2b54"
READINESS_SHA = "80c0815a5ee957d34f37de2d5e565971da10eb1cb4928af6051edd3bc170b1af"


def producer_identity():
    """Expected provenance for verification; never label this verifier as producer."""
    return {
        "config_sha256": CONFIG_SHA,
        "selection_sha256": SELECTION_SHA,
        "model_role": "comparison",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "huggingface-hub")
        },
        "code": {
            "git_commit": PRODUCER,
            "git_dirty": False,
            "phase": "workspace-jr",
            "git_argv0_state": "tracked",
            "git_argv0_path": "scripts/workspace_jr_pipeline.py",
        },
        "execution_readiness_sha256": READINESS_SHA,
    }


def unchanged_sources():
    """Bind the numerical code actually imported here to the frozen producer."""
    paths = [
        "src/explore_persona_space/analysis/workspace_decomposition.py",
        "src/explore_persona_space/analysis/workspace_lenses.py",
        "src/explore_persona_space/analysis/workspace_artifacts.py",
        "src/explore_persona_space/analysis/workspace_runtime.py",
    ]
    hashes = {}
    for relative in paths:
        original = subprocess.check_output(["git", "show", f"{PRODUCER}:{relative}"])
        current = Path(relative).read_bytes()
        if current != original:
            raise ValueError(f"Numerical source changed since producer: {relative}")
        hashes[relative] = hashlib.sha256(current).hexdigest()
    return hashes


def tensor_sha256(value):
    """Hash realized CPU tensor bytes, with shape and dtype recorded separately."""
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def worker_fingerprint():
    """Identify the GCP worker, physical GPU and exact numerical runtime."""
    request = Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/id",
        headers={"Metadata-Flavor": "Google"},
    )
    with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
        instance_id = response.read().decode().strip()
    if not instance_id.isdigit():
        raise ValueError("Expected a GCP instance identity")
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if not gpu_uuid.startswith("GPU-") or "\n" in gpu_uuid:
        raise ValueError("Expected one physical GPU identity")
    return {
        "gcp_instance_id": instance_id,
        "gpu_uuid": gpu_uuid,
        "python": platform.python_version(),
        "torch_cuda": torch.version.cuda,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "transformers", "huggingface-hub")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["schema"] != "workspace-jr-checkpoint-parity-v1" or plan["rotation"] not in (
        20260914,
        20260915,
    ):
        raise ValueError("Unexpected cache parity scope")
    root = Path(plan["out"])
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_absolute():
        raise ValueError("A real absolute parity root is required")
    if any(
        (root / name).exists()
        for name in ("complete.json", "parity_report.json", "parity_replay.pt")
    ):
        raise ValueError(
            "Existing parity execution evidence is immutable; use a fresh attempt root"
        )
    config_path = Path("configs/analysis/workspace_jr.yaml")
    selection_path = Path("docs/exploratory_workspace_jr/selected_contexts.json")
    if file_sha256(config_path) != CONFIG_SHA or file_sha256(selection_path) != SELECTION_SHA:
        raise ValueError("Frozen config/selection changed")
    config = load_workspace_jr_config(config_path)
    selection = json.loads(selection_path.read_text())
    prompt = selection["subsets"]["main_train"][0]["prompt_sha256"]
    if plan["prompt_sha256"] != prompt:
        raise ValueError("Parity must use the first frozen training context")
    source_hashes = unchanged_sources()
    proof_path = root / "plan.json"
    if proof_path.exists():
        if proof_path.read_bytes() != args.plan.read_bytes():
            raise ValueError("Parity plan bytes changed after staging")
    else:
        with proof_path.open("xb") as stream:
            stream.write(args.plan.read_bytes())
    spec = importlib.util.spec_from_file_location(
        "cache_stage", "scripts/workspace_jr_stage_calibration.py"
    )
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    upload = plan["reference_upload"]
    stage._check_upload(upload)
    expected_prefix = (
        f"exploratory_workspace_jr/20260912/comparison_main_rotation{plan['rotation']}_v1"
    )
    if upload["prefix"] != expected_prefix:
        raise ValueError("Unexpected reference producer")
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    for relative in plan["required_files"]:
        destination = regular_path(inputs, relative, create_parents=True)
        stage.stage_file(upload, relative, destination)
    identity = producer_identity()
    capture_relative = f"captures/main_train/{prompt}.pt"
    capture, capture_sha = read_checkpoint(
        inputs / capture_relative, upload["verified_sha256"][capture_relative]
    )
    validate_producer(capture["identity"]["identity"], identity)
    generation_relative = f"generations/main_train/{prompt}.json"
    if (
        capture["identity"]["generation_file_sha256"] != file_sha256(inputs / generation_relative)
        or [row["seed"] for row in capture["rows"]] != config["generation"]["seeds"]
        or any(row["prompt_sha256"] != prompt for row in capture["rows"])
    ):
        raise ValueError("Reference capture has different generations or seeds")
    required = {
        "dictionaries/manifest.json",
        "dictionaries/J.pt",
        "dictionaries/R.pt",
        capture_relative,
        generation_relative,
        capture["identity"]["context_input_file"],
        *[
            f"components/k{k}-rotation{plan['rotation']}/main_train/{prompt}.pt"
            for k in (5, 10, 25)
        ],
    }
    if set(plan["required_files"]) != required or len(plan["required_files"]) != len(required):
        raise ValueError("Parity staging must contain the exact first-context input closure")
    manifest_path = inputs / "dictionaries/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_producer(manifest["identity"], identity)
    if manifest["pilot_only"]:
        raise ValueError("Pilot dictionary cannot serve this parity check")
    reference = {}
    contracts = {}
    for k in (5, 10, 25):
        relative = f"components/k{k}-rotation{plan['rotation']}/main_train/{prompt}.pt"
        saved, _ = read_checkpoint(inputs / relative, upload["verified_sha256"][relative])
        contracts[k] = {
            "identity": identity,
            "dictionaries": manifest,
            "k": k,
            "rotation": plan["rotation"],
        }
        validate_checkpoint(saved, capture, capture_sha, contracts[k], inputs)
        reference[k] = saved
    for arm in ("J", "R"):
        if file_sha256(inputs / f"dictionaries/{arm}.pt") != manifest["arms"][arm]["sha256"]:
            raise ValueError("Reference dictionary changed")
    staging = {
        "status": "verified",
        "plan_sha256": file_sha256(args.plan),
        "required_files": {rel: upload["verified_sha256"][rel] for rel in plan["required_files"]},
        "producer_identity": identity,
        "numerical_source_sha256": source_hashes,
    }
    save_json(root / "staging_complete.json", staging)
    if args.stage_only:
        print(
            "Reference staged and actual checkpoint/capture consumer validation passed", flush=True
        )
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("One visible destination GPU required")
    properties = torch.cuda.get_device_properties(0)
    if "A100" not in properties.name or properties.total_memory < 79 * 2**30:
        raise ValueError("Parity requires the matching A100 80GB device domain")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Frozen production decomposition disables TF32")
    basis, rotations = {}, {}
    for arm in ("J", "R"):
        dictionary = torch.load(
            inputs / f"dictionaries/{arm}.pt", map_location="cuda:0", weights_only=True
        )["dictionary"]
        dictionary, q = rotated_dictionary(dictionary, seed=plan["rotation"])
        rotations[arm] = {
            "q_sha256": tensor_sha256(q),
            "dictionary_sha256": tensor_sha256(dictionary),
            "q_shape": list(q.shape),
            "dictionary_shape": list(dictionary.shape),
            "dtype": str(dictionary.dtype),
        }
        basis[arm] = ValidatedDictionary(dictionary)
    actual = decompose_context_nested(
        capture["rows"], basis, checkpoints=(5, 10, 25), token_batch_size=128
    )
    save_tensors(root / "parity_replay.pt", actual)
    report = {
        "status": "exact_numerical_parity",
        "prompt_sha256": prompt,
        "rotation": plan["rotation"],
        "k": [5, 10, 25],
        "atol": 0,
        "rtol": 0,
        "gpu": {"name": properties.name, "memory_bytes": properties.total_memory},
        "execution_worker": worker_fingerprint(),
        "rotated_matrices": rotations,
        "staging_sha256": file_sha256(root / "staging_complete.json"),
        "verifier_identity": run_identity(config_path, selection_path, "comparison"),
        "verifier_script_sha256": file_sha256(Path(__file__)),
        "cache_helper_sha256": file_sha256(
            Path("src/explore_persona_space/analysis/workspace_checkpoint_cache.py")
        ),
        "numerical_source_sha256": source_hashes,
        "parity_replay_sha256": file_sha256(root / "parity_replay.pt"),
    }
    try:
        for k in (5, 10, 25):
            expected = {
                key: value
                for key, value in reference[k].items()
                if key
                not in {"contract", "contract_sha256", "source_sha256", "context_input_reference"}
            }
            equal_tree(actual[k], expected)
    except (AssertionError, ValueError) as error:
        report.update(
            status="numerical_parity_failed", error_type=type(error).__name__, error=str(error)
        )
        save_json(root / "parity_report.json", report)
        raise
    save_json(root / "parity_report.json", report)
    save_json(
        root / "complete.json",
        {"status": "complete", "parity_report_sha256": file_sha256(root / "parity_report.json")},
    )
    print(
        "Exact parity passed for all fields at k=5,10,25 on the frozen first training context",
        flush=True,
    )


if __name__ == "__main__":
    main()
