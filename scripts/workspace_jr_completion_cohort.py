#!/usr/bin/env python3
"""Freeze the paired scoring cohort from uploaded final generations, without fits."""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.analysis.workspace_analysis_inputs import _upload_binding  # noqa: E402
from explore_persona_space.analysis.workspace_artifacts import validate_coverage, validate_producer  # noqa: E402
from explore_persona_space.analysis.workspace_completion import (
    checkpoint_policy,
    final_rollout_eligibility,
    scoring_implementation,
    joint_completion_cohort,
)  # noqa: E402
from explore_persona_space.analysis.workspace_runtime import (
    content_sha256,
    file_sha256,
    load_workspace_jr_config,
    run_identity,
    save_json,
)  # noqa: E402


def inspect_role(entry, role, config, identity, planned, out):
    """Verify producer status and each final generation object before eligibility."""
    import hashlib

    root, receipt = Path(entry["root"]), Path(entry["upload_receipt"])
    verify = _upload_binding(root, receipt)
    hashes = {}

    def read(relative):
        rel = Path(relative)
        if rel.is_absolute() or any(part.startswith(".") for part in rel.parts):
            raise ValueError("Generation sources must be safe relative paths")
        path = root / rel
        hashes[relative] = verify(path)
        return json.loads(path.read_text())

    terminal = read(entry["terminal_relative"])
    if terminal.get("phase") != "complete" or terminal.get("exit_code") != 0:
        raise ValueError("Generation source lacks a successful complete producer terminal")
    status = read("generations/main_test/generation_status.json")
    producer = status["identity"]
    validate_producer(producer, identity, native_ancestor=True)
    if not producer.get("execution_readiness_sha256") or status["needs_cap_recovery"]:
        raise ValueError("Completion ledger requires gated main generations after cap recovery")
    included = validate_coverage(status, planned, producer)
    if included != planned or status["exclusions"]:
        raise ValueError("Completion ledger requires every frozen planned generation record")
    policy, checkpoint = checkpoint_policy(config, role)
    contexts, cap_hits, n_draws = {}, 0, 0
    for index, context in enumerate(planned, 1):
        relative = f"generations/main_test/{context}.json"
        saved = read(relative)
        if hashes[relative] != status["file_sha256"][context]:
            raise ValueError("Generation record differs from its completed status")
        contract = saved["contract"]
        validate_producer(contract["identity"], producer)
        prompt_hash = hashlib.sha256(
            unicodedata.normalize("NFC", saved["prompt"]).encode()
        ).hexdigest()
        if (
            saved["contract_sha256"] != content_sha256(contract)
            or contract["generation"] != config["generation"]
            or contract["prompt_sha256"] != context
            or prompt_hash != context
        ):
            raise ValueError("Generation contract, settings or frozen prompt changed")
        contexts[context] = final_rollout_eligibility(
            saved["rollouts"], config["generation"]["seeds"], policy["terminal_ids"]
        )
        cap_hits += sum(draw["finish_reason"] == "length" for draw in saved["rollouts"])
        n_draws += len(saved["rollouts"])
        if index % 48 == 0:
            save_json(
                out / f"{role}_partial.json", {"contexts": contexts, "verified_sha256": hashes}
            )
        print(f"completion ledger role={role} context={index}/{len(planned)}", flush=True)
    if (
        status["n_contexts"] != len(planned)
        or status["n_rollouts"] != n_draws
        or status["cap_hits"] != cap_hits
        or status["cap_hit_fraction"] != cap_hits / n_draws
    ):
        raise ValueError("Final generation counts differ from actual records")
    return {
        "producer_identity": producer,
        "planned_context_ids": planned,
        "contexts": contexts,
        "terminal_policy": policy,
        "checkpoint": checkpoint,
        "final_cap_hits": cap_hits,
        "draws": n_draws,
        "eligible_contexts": sum(row["eligible"] for row in contexts.values()),
        "source_proof": {
            "upload_receipt_sha256": file_sha256(receipt),
            "upload": json.loads(receipt.read_text()),
            "verified_sha256": hashes,
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
    config = load_workspace_jr_config(args.config)
    manifest = json.loads(args.manifest.read_text())
    if set(manifest) != {"primary", "comparison"}:
        raise ValueError("Exactly both selected model sources are required")
    execution = json.loads(
        Path("docs/exploratory_workspace_jr/main_execution_primary_20260912.json").read_text()
    )
    if execution["config_sha256"] != file_sha256(args.config) or execution[
        "selection_sha256"
    ] != file_sha256(args.selection):
        raise ValueError("Execution prefix differs from frozen inputs")
    selected = json.loads(args.selection.read_text())["subsets"]["main_test"]
    planned = [row["prompt_sha256"] for row in selected[: execution["main_counts"]["test"]]]
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "input_manifest.json", manifest)
    by_role = {
        role: inspect_role(
            manifest[role],
            role,
            config,
            run_identity(args.config, args.selection, role),
            planned,
            args.out,
        )
        for role in manifest
    }
    report = {
        "schema": "workspace-jr-complete-test-cohort-v1",
        "status": "complete",
        "config_sha256": file_sha256(args.config),
        "selection_sha256": file_sha256(args.selection),
        "analysis_identity": {
            role: run_identity(args.config, args.selection, role) for role in manifest
        },
        "implementation_sha256": scoring_implementation(),
        "outcome_inputs": "final generation records only; no activations, predictions or fit results",
        "by_role": by_role,
        **joint_completion_cohort(by_role),
    }
    save_json(args.out / "completion_cohort.json", report)
    save_json(
        args.out / "cohort_complete.json",
        {
            "status": "complete",
            "report_sha256": file_sha256(args.out / "completion_cohort.json"),
            "finished_at_epoch": int(time.time()),
        },
    )


if __name__ == "__main__":
    main()
