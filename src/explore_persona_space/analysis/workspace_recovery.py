"""Reuse verified pilot answers and capture one context-only input per context."""

from __future__ import annotations

import json
import os

import torch

from explore_persona_space.analysis.workspace_artifacts import validate_coverage, validate_producer
from explore_persona_space.analysis.workspace_capture import (
    assign_context_input,
    capture_context_inputs,
)
from explore_persona_space.analysis.workspace_runtime import (
    content_sha256,
    file_sha256,
    load_native,
    save_json,
    save_tensors,
)


def _prepare_sources(args, config, identity):
    """Verify source identities, complete split coverage and durable byte hashes."""
    if (
        args.role != "primary"
        or args.stage != "pilot"
        or args.source is None
        or args.source_receipt is None
    ):
        raise ValueError("Recovery requires the explicit failed primary pilot source and receipt")
    if args.out.exists():
        raise ValueError("Canonical-input recovery needs a fresh output path")
    selection = json.loads(args.selection.read_text())
    receipt = json.loads(args.source_receipt.read_text())
    if (
        receipt["repo"] != "superkaiba1/explore-persona-space-data"
        or receipt["revision"] != "b7171049e8d6ea741bb412f209e0f933761e441f"
        or receipt["prefix"] != "exploratory_workspace_jr/20260912/primary_component_pilot2"
    ):
        raise ValueError("Recovery requires the exact persisted failed primary pilot")
    hashes = receipt["verified_sha256"]

    def verified(relative):
        path = args.source / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or relative not in hashes
            or file_sha256(path) != hashes[relative]
        ):
            raise ValueError(f"Unverified or changed source artifact: {relative}")
        return path

    terminal = json.loads(verified("pilot_exit.json").read_text())
    if terminal != {"exit_code": 1, "phase": "decompose_train", "finished_at_epoch": 1789244786}:
        raise ValueError("Recovery source is not the inspected input-consistency failure")
    dictionary = json.loads(verified("dictionaries/manifest.json").read_text())
    producer = dictionary["identity"]
    if producer["code"]["git_commit"] != "348b11a45a26147f1270356edfb6d3dbfd6f143c":
        raise ValueError("Unexpected original capture producer")
    validate_producer(producer, identity, native_ancestor=True)
    if dictionary["pilot_only"] is not True:
        raise ValueError("Recovery must remain an end-to-end pilot")
    staged = {}
    for split in ("train", "validation", "test"):
        subset = f"pilot_{split}"
        prompts = selection["subsets"][subset]
        expected_ids = [row["prompt_sha256"] for row in prompts]
        generation = json.loads(
            verified(f"generations/{subset}/generation_status.json").read_text()
        )
        capture = json.loads(verified(f"captures/{subset}/coverage.json").read_text())
        validate_coverage(generation, expected_ids, producer)
        included = validate_coverage(capture, expected_ids, producer)
        rows = {}
        for context in expected_ids:
            relative = f"generations/{subset}/{context}.json"
            path = verified(relative)
            raw = json.loads(path.read_text())
            if (
                file_sha256(path) != generation["file_sha256"][context]
                or raw["contract_sha256"] != content_sha256(raw["contract"])
                or raw["contract"]["identity"] != producer
                or raw["contract"]["generation"] != config["generation"]
                or raw["contract"]["prompt_sha256"] != context
            ):
                raise ValueError("Reused generation contract differs from the frozen pilot")
            if context in included:
                captured_path = verified(f"captures/{subset}/{context}.pt")
                if file_sha256(captured_path) != capture["file_sha256"][context]:
                    raise ValueError("Capture coverage does not bind its source bytes")
            rows[context] = raw
        staged[subset] = (prompts, generation, capture, rows)
    return receipt, hashes, verified, staged, dictionary, producer


def recover_pilot_inputs(args, config, identity):
    """Preserve the failed tree; write a distinctly identified recovered pilot tree."""
    receipt, hashes, verified, staged, dictionary, producer = _prepare_sources(
        args, config, identity
    )
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "workspace-jr-canonical-input-recovery-v1",
        "identity": identity,
        "source_upload": receipt,
        "source_producer": producer,
        "input_policy": "context_only_frozen_order_batches16_no_answer_tokens",
        "preserved": "generation bytes, dictionaries and individual answer-token states",
        "status": "running",
        "subsets": {},
    }
    save_json(args.out / "recovery_manifest.json", report)
    for arm in ("J", "R"):
        name = f"dictionaries/{arm}.pt"
        source = verified(name)
        if hashes[name] != dictionary["arms"][arm]["sha256"]:
            raise ValueError("Dictionary manifest hash differs")
        target = args.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    dictionary = {
        **dictionary,
        "identity": identity,
        "reused_source_manifest": {
            "identity": producer,
            "sha256": hashes["dictionaries/manifest.json"],
            "upload_revision": receipt["revision"],
        },
    }
    save_json(args.out / "dictionaries/manifest.json", dictionary)
    model, tokenizer, text = load_native(
        config, args.role, device=args.device, dtype=torch.bfloat16
    )
    spec = config["models"][args.role]
    for subset, (prompts, generation, capture, rows) in staged.items():
        input_lookup = {}
        for begin in range(0, len(prompts), 16):
            batch = prompts[begin : begin + 16]
            contexts = [row["prompt_sha256"] for row in batch]
            tokens = [rows[context]["contract"]["prompt_token_ids"] for context in contexts]
            values = capture_context_inputs(
                text, tokens, spec["source_layer"], tokenizer.pad_token_id
            )
            relative = f"context_inputs/{subset}/batch-{begin:04d}.pt"
            source_hashes = [generation["file_sha256"][context] for context in contexts]
            save_tensors(
                args.out / relative,
                {
                    "contract": {
                        "identity": identity,
                        "prompt_ids": tokens,
                        "source_hashes": source_hashes,
                        "policy": report["input_policy"],
                    },
                    "x": values,
                },
            )
            checksum = file_sha256(args.out / relative)
            for index, context in enumerate(contexts):
                input_lookup[context] = (
                    values[index],
                    {
                        "context_input_file": relative,
                        "context_input_file_sha256": checksum,
                        "context_input_row": index,
                    },
                )
            print(
                f"canonical_input_batch subset={subset} start={begin} contexts={len(batch)}",
                flush=True,
            )
        capture_hashes, max_difference = {}, 0.0
        for context, raw in rows.items():
            gen_relative = f"generations/{subset}/{context}.json"
            target = args.out / gen_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(verified(gen_relative), target)
            if context not in capture["included_prompt_sha256"]:
                continue
            relative = f"captures/{subset}/{context}.pt"
            source = verified(relative)
            saved = torch.load(source, map_location="cpu", weights_only=True)
            if (
                saved["identity"]["identity"] != producer
                or saved["identity"]["generation_file_sha256"] != hashes[gen_relative]
                or [r["seed"] for r in saved["rows"]] != config["generation"]["seeds"]
                or any(
                    r["prompt_sha256"] != context
                    or r["prompt_ids"] != raw["contract"]["prompt_token_ids"]
                    for r in saved["rows"]
                )
            ):
                raise ValueError("Original capture source, seed or context contract differs")
            value, reference = input_lookup[context]
            corrected = assign_context_input(saved["rows"], value)
            for prior, current in zip(saved["rows"], corrected, strict=True):
                if not torch.equal(prior["answer_states"], current["answer_states"]):
                    raise ValueError("Recovery altered answer-token activations")
                max_difference = max(
                    max_difference, float((prior["x"].float() - value.float()).abs().max())
                )
            capture_identity = {
                "identity": identity,
                "generation_file_sha256": hashes[gen_relative],
                **reference,
                "reused_capture_sha256": hashes[relative],
                "reused_capture_producer": producer,
            }
            save_tensors(args.out / relative, {"identity": capture_identity, "rows": corrected})
            capture_hashes[context] = file_sha256(args.out / relative)
            print(f"canonical_capture_saved subset={subset} context={context}", flush=True)
        # Generations retain their original producer identity. This recovery has
        # not sampled new text; capture/decomposition consumers bind their bytes.
        save_json(args.out / f"generations/{subset}/generation_status.json", generation)
        save_json(
            args.out / f"captures/{subset}/coverage.json",
            {
                **capture,
                "identity": identity,
                "file_sha256": capture_hashes,
                "reused_coverage_sha256": hashes[f"captures/{subset}/coverage.json"],
            },
        )
        report["subsets"][subset] = {
            "contexts": len(capture_hashes),
            "max_original_to_canonical_coordinate_difference": max_difference,
        }
        save_json(args.out / "recovery_manifest.json", report)
    del model, text
    report["status"] = "complete_context_inputs_recovered_no_component_fits_yet"
    save_json(args.out / "recovery_manifest.json", report)
