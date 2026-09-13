"""Correct the audited comparison pilot EOS mask using immutable captured prefixes."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from explore_persona_space.analysis.workspace_analysis_inputs import _upload_binding
from explore_persona_space.analysis.workspace_artifacts import (
    validate_context_input,
    validate_coverage,
    validate_producer,
)
from explore_persona_space.analysis.workspace_capture import (
    assign_context_input,
    capture_token_batch,
)
from explore_persona_space.analysis.workspace_runtime import (
    content_sha256,
    file_sha256,
    load_native,
    save_json,
    save_tensors,
)
from explore_persona_space.analysis.workspace_terminals import (
    terminal_policy,
    trim_saved_answer,
)

SOURCE_REVISION = "820fcbe4515a1704f453ffcb43bd00c8ee7f0987"
SOURCE_PRODUCER = "78de99d43d2175eeb080badeba48d40072078d38"
SOURCE_RECEIPT_SHA256 = "38655bb1c48e1fd79476bc88109ff71fc7245984f2a654ec72ff2eaca52e0c8e"


def exact_tree(actual, expected):
    """Resume only byte-identical tensor values and exactly matching metadata."""
    if type(actual) is not type(expected):
        raise ValueError("Resumed recovery value type changed")
    if isinstance(expected, torch.Tensor):
        if (
            actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not torch.equal(
                actual.contiguous().reshape(-1).view(torch.uint8),
                expected.contiguous().reshape(-1).view(torch.uint8),
            )
        ):
            raise ValueError("Resumed recovery tensor changed")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError("Resumed recovery metadata keys changed")
        for key in expected:
            exact_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError("Resumed recovery sequence length changed")
        for a, e in zip(actual, expected, strict=True):
            exact_tree(a, e)
    elif actual != expected:
        raise ValueError("Resumed recovery metadata changed")


def publish_tensor(path, value):
    """Keep per-context atomic checkpoints and validate any resumed output."""
    if path.exists():
        exact_tree(torch.load(path, map_location="cpu", weights_only=True), value)
    else:
        save_tensors(path, value)


def link_verified(source, destination, checksum):
    """Retain the original generation and dictionary bytes without relabeling them."""
    if destination.exists():
        if destination.is_symlink() or file_sha256(destination) != checksum:
            raise ValueError("Resumed immutable recovery source changed")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)


def recapture_answers(rows, text, tokenizer, source_layer):
    """Re-execute corrected answer batches 2/2/1 and preserve canonical context x."""
    context_input = rows[0]["x"]
    if len(rows) != 5 or any(not torch.equal(row["x"], context_input) for row in rows):
        raise ValueError("Fresh recovery capture requires five identical canonical inputs")
    inputs = [
        {
            key: value
            for key, value in row.items()
            if key not in {"x", "answer_batch_x", "answer_states"}
        }
        for row in rows
    ]
    captured = []
    for begin in range(0, len(inputs), 2):
        captured.extend(
            capture_token_batch(
                text, inputs[begin : begin + 2], source_layer, tokenizer.pad_token_id
            )
        )
    return assign_context_input(captured, context_input)


def recover(
    args, config, identity, policy, checkpoint_evidence, *, expected_receipt_sha256, recapture=None
):
    """Bind the exact failed pilot and rewrite only EOS masks and derived identities."""
    if file_sha256(args.source_receipt) != expected_receipt_sha256:
        raise ValueError("Recovery receipt differs from the independently audited bytes")
    receipt = json.loads(args.source_receipt.read_text())
    if (
        receipt["revision"] != SOURCE_REVISION
        or receipt["prefix"] != "exploratory_workspace_jr/20260912/comparison_component_pilot1"
        or identity["model_role"] != "comparison"
    ):
        raise ValueError("EOS recovery requires the independently audited comparison pilot")
    verified = _upload_binding(args.source, args.source_receipt)
    manifest_path = args.source / "dictionaries/manifest.json"
    dictionary_sha = verified(manifest_path)
    dictionary = json.loads(manifest_path.read_text())
    producer = dictionary["identity"]
    if producer["code"]["git_commit"] != SOURCE_PRODUCER or dictionary["pilot_only"] is not True:
        raise ValueError("Unexpected comparison pilot producer or dictionary scope")
    validate_producer(producer, identity, native_ancestor=True)
    terminal = args.source / "pilot_exit.json"
    verified(terminal)
    end = json.loads(terminal.read_text())
    if end["exit_code"] != 0 or end["phase"] != "complete":
        raise ValueError("Correction source workload did not complete")
    contract = {
        "schema": "workspace-jr-terminal-eos-recovery-v1",
        "identity": identity,
        "source_upload": receipt,
        "source_producer": producer,
        "terminal_policy": policy,
        "original_terminal_ids": [248044],
        "checkpoint_evidence": checkpoint_evidence,
        "source_receipt_sha256": file_sha256(args.source_receipt),
        "answer_capture_mode": "fresh_EOS_excluded_native_batches2"
        if recapture
        else "trim_original_prefix",
        "mask_correction": "first terminal EOS and every following ID excluded",
        "forward_geometry": (
            "fresh native BF16 eager forward on EOS-excluded answer IDs, batches2/2/1"
            if recapture
            else "original EOS-inclusive pilot forward; prefix states preserved bitwise"
        ),
        "preserved": (
            "original generation bytes, canonical context x, dictionary tensors, "
            "frozen seeds/splits/tuning budgets"
        ),
    }
    path = args.out / "recovery_contract.json"
    if args.out.exists():
        if not args.resume or not path.exists() or json.loads(path.read_text()) != contract:
            raise ValueError("Recovery resume requires the exact unchanged contract")
        if (args.out / "recovery_complete.json").exists():
            raise FileExistsError("Completed EOS recovery is immutable")
    else:
        if args.resume:
            raise ValueError("No partial recovery exists to resume")
        args.out.mkdir(parents=True)
        save_json(path, contract)
    for arm in ("J", "R"):
        relative = f"dictionaries/{arm}.pt"
        source = args.source / relative
        checksum = verified(source)
        if checksum != dictionary["arms"][arm]["sha256"]:
            raise ValueError("Original pilot dictionary checksum mismatch")
        link_verified(source, args.out / relative, checksum)
    rewritten_dictionary = {
        **dictionary,
        "identity": identity,
        "reused_source_manifest": {
            "identity": producer,
            "sha256": dictionary_sha,
            "upload_revision": receipt["revision"],
        },
        "terminal_recovery_contract_sha256": content_sha256(contract),
    }
    dictionary_out = args.out / "dictionaries/manifest.json"
    if dictionary_out.exists() and json.loads(dictionary_out.read_text()) != rewritten_dictionary:
        raise ValueError("Resumed dictionary lineage changed")
    save_json(dictionary_out, rewritten_dictionary)
    selection = json.loads(args.selection.read_text())
    report = {"contract_sha256": content_sha256(contract), "subsets": {}}
    for split in ("train", "validation", "test"):
        subset = f"pilot_{split}"
        expected = [r["prompt_sha256"] for r in selection["subsets"][subset]]
        report["subsets"][subset] = recover_subset(
            args, config, identity, contract, verified, producer, subset, expected, recapture
        )
    report["status"] = "complete_corrected_captures_no_component_fits_yet"
    save_json(args.out / "recovery_complete.json", report)


def recover_subset(
    args, config, identity, contract, verified, producer, subset, expected, recapture
):
    """Rebind canonical inputs and trim each complete context with durable progress."""
    reports = {}
    for kind, filename in (
        ("generations", "generation_status.json"),
        ("captures", "coverage.json"),
    ):
        source = args.source / kind / subset / filename
        verified(source)
        reports[kind] = json.loads(source.read_text())
        validate_coverage(reports[kind], expected, producer)
    if reports["generations"]["needs_cap_recovery"]:
        raise ValueError("EOS recovery cannot conceal unfinished truncation recovery")
    included_source = set(reports["captures"]["included_prompt_sha256"])
    excluded = list(reports["captures"]["exclusions"])
    included, hashes, rows_report = [], {}, []
    for index, context in enumerate(expected):
        started = time.perf_counter()
        gen_path = args.source / "generations" / subset / f"{context}.json"
        gen_sha = verified(gen_path)
        raw = json.loads(gen_path.read_text())
        validate_producer(raw["contract"]["identity"], producer)
        if (
            reports["generations"]["file_sha256"][context] != gen_sha
            or raw["contract_sha256"] != content_sha256(raw["contract"])
            or raw["contract"]["prompt_sha256"] != context
            or raw["contract"]["generation"] != config["generation"]
            or [r["seed"] for r in raw["rollouts"]] != config["generation"]["seeds"]
        ):
            raise ValueError("Reused generation membership, hash or sampling contract differs")
        link_verified(gen_path, args.out / gen_path.relative_to(args.source), gen_sha)
        if context not in included_source:
            continue
        source = args.source / "captures" / subset / f"{context}.pt"
        source_sha = verified(source)
        if source_sha != reports["captures"]["file_sha256"][context]:
            raise ValueError("Capture checksum differs from its completed coverage")
        saved = torch.load(source, map_location="cpu", weights_only=True)
        reference = saved["identity"]
        validate_producer(reference["identity"], producer)
        if reference["generation_file_sha256"] != gen_sha or len(saved["rows"]) != 5:
            raise ValueError("Captured generation source or rollout count differs")
        input_relative = reference["context_input_file"]
        input_source = args.source / input_relative
        input_sha = verified(input_source)
        original_input = torch.load(input_source, map_location="cpu", weights_only=True)
        current_input = {
            **original_input,
            "contract": {**original_input["contract"], "identity": identity},
            "terminal_recovery_original_sha256": input_sha,
            "terminal_recovery_original_producer": producer,
        }
        publish_tensor(args.out / input_relative, current_input)
        corrected, changes = [], []
        for row, draw in zip(saved["rows"], raw["rollouts"], strict=True):
            if (
                row["prompt_sha256"] != context
                or row["prompt_ids"] != raw["contract"]["prompt_token_ids"]
            ):
                raise ValueError("Original captured prompt differs from generation")
            validate_context_input(reference, row["x"], args.source, producer)
            value, change = trim_saved_answer(row, draw, contract["terminal_policy"], {248044})
            corrected.append(value)
            changes.append(change)
        new_reference = {
            **reference,
            "identity": identity,
            "context_input_file_sha256": file_sha256(args.out / input_relative),
            "terminal_recovery_original_capture_sha256": source_sha,
            "terminal_recovery_original_producer": producer,
            "terminal_recovery_contract_sha256": content_sha256(contract),
        }
        empty = [r["seed"] for r in changes if r["empty_after_terminal_correction"]]
        if empty:
            excluded.append(
                {
                    "prompt_sha256": context,
                    "reason": "incomplete_K_after_terminal_EOS_correction",
                    "empty_seeds": empty,
                }
            )
        else:
            if recapture is not None:
                corrected = recapture(corrected)
            for row in corrected:
                validate_context_input(new_reference, row["x"], args.out, identity)
            destination = args.out / source.relative_to(args.source)
            publish_tensor(destination, {"identity": new_reference, "rows": corrected})
            included.append(context)
            hashes[context] = file_sha256(destination)
        rows_report.append(
            {
                "prompt_sha256": context,
                "original_capture_sha256": source_sha,
                "rollouts": changes,
            }
        )
        save_json(args.out / "recovery_progress" / subset / f"{context}.json", rows_report[-1])
        print(
            f"EOS recovery subset={subset} context={index + 1}/{len(expected)} "
            f"key={context} elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )
    save_json(args.out / "generations" / subset / "generation_status.json", reports["generations"])
    save_json(
        args.out / "captures" / subset / "coverage.json",
        {
            "identity": identity,
            "planned_contexts": len(expected),
            "status": "complete",
            "included_prompt_sha256": included,
            "exclusions": excluded,
            "file_sha256": hashes,
            "original_coverage_sha256": verified(
                args.source / "captures" / subset / "coverage.json"
            ),
            "terminal_recovery_contract_sha256": content_sha256(contract),
        },
    )
    return {
        "included_contexts": len(included),
        "exclusions": excluded,
        "rows": rows_report,
    }


def recover_terminal_eos(args, config, identity):
    """Use the genuine pipeline identity for recovery and its downstream phases."""
    if (
        args.role != "comparison"
        or args.stage != "pilot"
        or args.source is None
        or args.source_receipt is None
    ):
        raise ValueError("EOS recovery requires an explicit comparison pilot source and receipt")
    if (
        file_sha256(args.source_receipt) != SOURCE_RECEIPT_SHA256
        or json.loads(args.source_receipt.read_text())["files_verified"] != 429
    ):
        raise ValueError("EOS recovery requires the exact 429-file independently audited receipt")
    model_id, revision = (
        config["selection"]["comparison"],
        config["models"]["comparison"]["revision"],
    )
    metadata = HfApi().model_info(model_id, revision=revision)
    if metadata.sha != revision or "generation_config.json" in {
        s.rfilename for s in metadata.siblings
    }:
        raise ValueError(
            "Pinned comparison checkpoint no longer matches the audited absence "
            "of generation_config.json"
        )
    native_config = AutoConfig.from_pretrained(model_id, revision=revision)
    generation = GenerationConfig.from_model_config(native_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    policy = terminal_policy(generation, tokenizer)
    if policy["sources"]["generation_config_eos"] != [248044] or policy["sources"][
        "tokenizer_eos"
    ] != [248046]:
        raise ValueError("Exact comparison checkpoint terminal IDs differ from the audited defect")
    checkpoint = {
        "model": model_id,
        "revision": revision,
        "generation_config_json_present": False,
        "config_json_sha256": file_sha256(
            Path(hf_hub_download(model_id, "config.json", revision=revision))
        ),
        "tokenizer_config_json_sha256": file_sha256(
            Path(hf_hub_download(model_id, "tokenizer_config.json", revision=revision))
        ),
    }
    recapture = None
    if args.recapture_answers:
        checkpoint["failed_terminal_geometry"] = failed_terminal_geometry(args, identity)
        model, native_tokenizer, text = load_native(
            config, "comparison", device=args.device, dtype=torch.bfloat16
        )
        if (
            type(model).__name__ != "Qwen3_5ForConditionalGeneration"
            or text.config._attn_implementation != "eager"
        ):
            raise ValueError("Fresh answer recapture differs from the pinned native geometry")
        if terminal_policy(model.generation_config, native_tokenizer) != policy:
            raise ValueError("Fresh native capture stopping policy differs from pinned metadata")

        def recapture(rows):
            return recapture_answers(
                rows, text, native_tokenizer, config["models"]["comparison"]["source_layer"]
            )

    recover(
        args,
        config,
        identity,
        policy,
        checkpoint,
        expected_receipt_sha256=SOURCE_RECEIPT_SHA256,
        recapture=recapture,
    )


def failed_terminal_geometry(args, identity):
    """Bind the uploaded diagnostic failure before its predeclared fallback."""
    if args.terminal_parity_root is None or args.terminal_parity_receipt is None:
        raise ValueError(
            "Fresh recapture requires the uploaded failed terminal-geometry diagnostic"
        )
    upload = json.loads(args.terminal_parity_receipt.read_text())
    if upload["revision"] != "a47f8b06dd8c01e4067bf29898b62a86000451b9":
        raise ValueError("Fresh recapture requires the exact observed diagnostic failure")
    verify = _upload_binding(args.terminal_parity_root, args.terminal_parity_receipt)
    source = args.terminal_parity_root / "terminal_parity.json"
    checksum = verify(source)
    if checksum != "9554214e365e31e092ec2142e0c2d37bd5f1576e80c5e9f0a3f2e3f2e48545dc":
        raise ValueError("Terminal-geometry report differs from the observed failure")
    report = json.loads(source.read_text())
    validate_producer(report["plan"]["identity"], identity, native_ancestor=True)
    if (
        report["status"] != "failed_requires_fresh_pilot_capture"
        or report["passed"] is not False
        or report["plan"]["source_receipt_sha256"] != SOURCE_RECEIPT_SHA256
        or report["plan"]["capture_batches"] != [2, 2, 1]
    ):
        raise ValueError("Terminal-geometry diagnostic does not require this fallback")
    for context in report["contexts"]:
        path = args.terminal_parity_root / f"context-{context['prompt_sha256']}.pt"
        if verify(path) != context["output_sha256"]:
            raise ValueError("Terminal-geometry diagnostic native outputs changed")
    return {"report_sha256": checksum, "upload": upload}
