#!/usr/bin/env python3
"""Run exact-token generation, capture, sparse decomposition and component fits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.workspace_artifacts import (  # noqa: E402
    validate_coverage,
    validate_context_input,
    validate_producer,
)
from explore_persona_space.analysis.workspace_capture import (  # noqa: E402
    answer_token_ids,
    assign_context_input,
    capture_context_inputs,
    capture_token_batch,
    generate_rollouts,
    save_capture_batch,
)
from explore_persona_space.analysis.workspace_decomposition import (  # noqa: E402
    ValidatedDictionary,
    decompose_context_nested,
)
from explore_persona_space.analysis.workspace_fit import evaluate_component_fits  # noqa: E402
from explore_persona_space.analysis.workspace_gate import validate_main_readiness  # noqa: E402
from explore_persona_space.analysis.workspace_recovery import recover_pilot_inputs  # noqa: E402
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    content_sha256,
    file_sha256,
    load_native,
    load_workspace_jr_config,
    run_identity,
    save_json,
    save_tensors,
    selected_prompts,
    validate_token_manifest,
)


def phase_prompts(args, subset=None):
    """Resolve the frozen split, applying only a reviewed pre-main prefix ceiling."""
    subset = args.subset if subset is None else subset
    rows = selected_prompts(args.selection, args.audit, subset)
    if subset.startswith("main_"):
        if args.readiness is None:
            raise ValueError("Main prompt resolution requires reviewed readiness")
        rows = rows[: args.readiness["execution"]["main_counts"][subset.removeprefix("main_")]]
    return rows


def generation(args, config, identity):
    """Generate the frozen subset using the project's batched vLLM engine."""
    from transformers import AutoTokenizer

    from explore_persona_space.eval.generation import create_vllm_engine
    from explore_persona_space.analysis.workspace_execution import (
        generation_engine,
        validate_generation_engine,
    )

    spec = config["models"][args.role]
    model_id = config["selection"][args.role]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=spec["revision"])
    prompts = phase_prompts(args)
    execution = generation_engine("eager16")
    execution_contract = None
    if args.readiness is not None:
        execution = validate_generation_engine(args.readiness["execution"]["generation_engine"])
        execution_contract = {
            "readiness_sha256": args.readiness["readiness_sha256"],
            "generation_engine": execution,
        }
        execution_path = args.out / "generations" / args.subset / "execution.json"
        if execution_path.exists():
            if json.loads(execution_path.read_text()) != execution_contract:
                raise ValueError("Cannot resume generations with a different execution contract")
        else:
            save_json(execution_path, execution_contract)
    engine = create_vllm_engine(
        model_id,
        revision=spec["revision"],
        tokenizer_revision=spec["revision"],
        gpu_memory_utilization=execution["gpu_memory_utilization"],
        max_model_len=execution["max_model_len"],
        seed=config["seed"],
        hang_mitigations=True,
        **execution["knobs"],
    )
    report = generate_rollouts(
        engine,
        tokenizer,
        prompts,
        config,
        identity,
        args.out / "generations" / args.subset,
        contexts_per_batch=execution["contexts_per_batch"],
        max_model_len=execution["max_model_len"],
        execution_contract=execution_contract,
    )
    if report["needs_cap_recovery"]:
        raise RuntimeError("Cap-hit fraction exceeds 2%; saved draws require doubled-cap recovery")


def capture(args, config, identity):
    """Capture saved token IDs with visible empty-answer exclusions and K checks."""
    _model, tokenizer, text = load_native(
        config, args.role, device=args.device, dtype=torch.bfloat16
    )
    spec = config["models"][args.role]
    prompts = phase_prompts(args)
    root = args.out / "generations" / args.subset
    generation_report = json.loads((root / "generation_status.json").read_text())
    validate_coverage(generation_report, [p["prompt_sha256"] for p in prompts], identity)
    if generation_report["needs_cap_recovery"]:
        raise ValueError("Capture requires completed truncation recovery")
    context_inputs = {}
    for begin in range(0, len(prompts), 16):
        batch = prompts[begin : begin + 16]
        source_paths = [root / f"{row['prompt_sha256']}.json" for row in batch]
        source_hashes = [file_sha256(path) for path in source_paths]
        if source_hashes != [
            generation_report["file_sha256"][row["prompt_sha256"]] for row in batch
        ]:
            raise ValueError("Context input generation sources changed after completion")
        saved_rows = [json.loads(path.read_text()) for path in source_paths]
        for saved in saved_rows:
            validate_producer(saved["contract"]["identity"], identity)
        token_lists = [saved["contract"]["prompt_token_ids"] for saved in saved_rows]
        input_path = args.out / "context_inputs" / args.subset / f"batch-{begin:04d}.pt"
        input_contract = {
            "identity": identity,
            "source_hashes": source_hashes,
            "prompt_ids": token_lists,
            "policy": "context_only_frozen_order_batches16_no_answer_tokens",
        }
        if input_path.exists():
            payload = torch.load(input_path, map_location="cpu", weights_only=True)
            if payload["contract"] != input_contract:
                raise ValueError("Stale canonical context input checkpoint")
            values = payload["x"]
        else:
            values = capture_context_inputs(
                text, token_lists, spec["source_layer"], tokenizer.pad_token_id
            )
            save_tensors(input_path, {"contract": input_contract, "x": values})
        context_inputs.update(
            {
                row["prompt_sha256"]: (
                    value,
                    {
                        "context_input_file_sha256": file_sha256(input_path),
                        "context_input_file": str(input_path.relative_to(args.out)),
                        "context_input_row": index,
                    },
                )
                for index, (row, value) in enumerate(zip(batch, values, strict=True))
            }
        )
    terminal = set(
        _model.generation_config.eos_token_id
        if isinstance(_model.generation_config.eos_token_id, list)
        else [_model.generation_config.eos_token_id]
    )
    excluded, included, hashes = [], [], {}
    for row in prompts:
        source = root / f"{row['prompt_sha256']}.json"
        saved = json.loads(source.read_text())
        if generation_report["file_sha256"][row["prompt_sha256"]] != file_sha256(source):
            raise ValueError("Generation source changed after completion")
        validate_producer(saved["contract"]["identity"], identity)
        if (
            saved["contract_sha256"] != content_sha256(saved["contract"])
            or saved["contract"]["prompt_sha256"] != row["prompt_sha256"]
            or [r["seed"] for r in saved["rollouts"]] != config["generation"]["seeds"]
        ):
            raise ValueError("Generation prompt, contract or seed membership mismatch")
        context_input, context_input_reference = context_inputs[row["prompt_sha256"]]
        contract = {
            "identity": identity,
            "generation_file_sha256": file_sha256(source),
            **context_input_reference,
        }
        target = args.out / "captures" / args.subset / f"{row['prompt_sha256']}.pt"
        if target.exists():
            prior = torch.load(target, map_location="cpu", weights_only=True)
            if prior["identity"] != contract:
                raise ValueError(f"Stale native capture {target}")
            included.append(row["prompt_sha256"])
            hashes[row["prompt_sha256"]] = file_sha256(target)
            continue
        rows = []
        empty_seeds = []
        for draw in saved["rollouts"]:
            answer, dropped = answer_token_ids(draw["token_ids"], terminal)
            if not answer:
                empty_seeds.append(draw["seed"])
                continue
            rows.append(
                {
                    "prompt_ids": saved["contract"]["prompt_token_ids"],
                    "answer_ids": answer,
                    "prompt_sha256": row["prompt_sha256"],
                    "seed": draw["seed"],
                    "finish_reason": draw["finish_reason"],
                    "terminal_ids_removed": dropped,
                }
            )
        if len(rows) != len(config["generation"]["seeds"]):
            excluded.append(
                {
                    "prompt_sha256": row["prompt_sha256"],
                    "reason": "incomplete_K_context_exclusion",
                    "empty_seeds": empty_seeds,
                }
            )
            continue
        captured = []
        for start in range(0, len(rows), args.capture_batch_size):
            captured.extend(
                capture_token_batch(
                    text,
                    rows[start : start + args.capture_batch_size],
                    spec["source_layer"],
                    tokenizer.pad_token_id,
                )
            )
        captured = assign_context_input(captured, context_input)
        save_capture_batch(target, captured, contract)
        included.append(row["prompt_sha256"])
        hashes[row["prompt_sha256"]] = file_sha256(target)
        print(f"captured context={row['prompt_sha256']} rollouts={len(captured)}", flush=True)
    save_json(
        args.out / "captures" / args.subset / "coverage.json",
        {
            "identity": identity,
            "planned_contexts": len(prompts),
            "exclusions": excluded,
            "status": "complete",
            "included_prompt_sha256": included,
            "file_sha256": hashes,
        },
    )


def dictionaries(args, config, identity):
    """Average matched per-prompt matrices and fold the native final norm gain."""
    if args.readiness is not None:
        return reviewed_dictionaries(args, config, identity)
    tokens = json.loads(args.token_manifest.read_text())
    validate_token_manifest(
        tokens,
        config_path=args.config,
        selection_path=args.selection,
        config=config,
        role=args.role,
    )
    validation = json.loads(args.validation.read_text())
    producer_proof = validate_producer(validation["identity"], identity, native_ancestor=True)
    if (
        validation["ordinary_numerical_validation"]["status"] != "passed"
        or not validation["forward_bit_identical"]
        or not validation["hook_bit_identical"]
        or validation["token_manifest_sha256"] != content_sha256(tokens)
    ):
        raise ValueError("Dictionary requires native numerical/parity validation of these tokens")
    model, _tokenizer, text = load_native(
        config, args.role, device=args.device, dtype=torch.bfloat16
    )
    if (
        validation["model_dtype"] != config["lenses"]["calibration_dtype"]
        or str(next(model.parameters()).dtype) != f"torch.{validation['model_dtype']}"
    ):
        raise ValueError("Native validation and dictionary checkpoint precision differ")
    files = sorted(args.lens_shards.glob("prompt-*.pt"))
    if not files:
        raise ValueError("No matched J/R lens shards")
    rows = [torch.load(path, map_location="cpu", weights_only=True) for path in files]
    if len({row["contract_sha256"] for row in rows}) != 1:
        raise ValueError("Lens shards have different contracts")
    if len({row["prompt_sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate calibration prompt in lens shards")
    frozen = {row["prompt_sha256"]: row["token_ids"] for row in tokens["rows"]}
    for row in rows:
        contract = row["contract"]
        validate_producer(contract, validation["identity"])
        if (
            contract["actual_dtype"] != str(next(model.parameters()).dtype)
            or contract["attention_implementation"] != str(text.config._attn_implementation)
            or contract["model_class"] != type(model).__name__
        ):
            raise ValueError("Lens and dictionary native runtime geometry/precision differ")
        if (
            row["contract_sha256"] != content_sha256(contract)
            or contract["tokens_sha256"] != content_sha256(tokens)
            or contract["native_validation_sha256"] != file_sha256(args.validation)
            or row["prompt_sha256"] not in frozen
            or row["token_ids"] != frozen[row["prompt_sha256"]]
        ):
            raise ValueError("Lens shard token membership or validation contract mismatch")
        for arm in ("J", "R"):
            d = config["models"][args.role]["d_model"]
            if row[arm].shape != (d, d) or not torch.isfinite(row[arm]).all():
                raise ValueError("Invalid native lens matrix")
    manifest = {
        "identity": identity,
        "calibration_prompts": len(rows),
        "pilot_only": True,
        "full_calibration_membership": set(frozen) == {r["prompt_sha256"] for r in rows},
        "calibration_token_manifest_sha256": content_sha256(tokens),
        "native_validation_sha256": file_sha256(args.validation),
        "native_producer_proof": producer_proof,
        "main_gate": "requires_calibration_stability_and_readout_review",
        "source_hashes": {p.name: file_sha256(p) for p in files},
        "arms": {},
    }
    matrices = {
        name: torch.stack([row[name].float() for row in rows]).mean(0) for name in ("J", "R")
    }
    write_dictionary_arms(args, model, text, matrices, manifest)


def reviewed_dictionaries(args, config, identity):
    """Use the exact full-calibration mean tensors covered by independent review."""
    evidence = args.readiness
    report = evidence["reports"]["calibration"]
    matrices = torch.load(evidence["paths"]["means"], map_location="cpu", weights_only=True)[
        report["full_group"]
    ]
    dimension = config["models"][args.role]["d_model"]
    if set(matrices) != {"J", "R"} or any(
        value.shape != (dimension, dimension)
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
        for value in matrices.values()
    ):
        raise ValueError("Reviewed means have incorrect shape, precision or finite values")
    model, _tokenizer, text = load_native(
        config, args.role, device=args.device, dtype=torch.bfloat16
    )
    if (
        str(next(model.parameters()).dtype) != "torch.bfloat16"
        or str(text.config._attn_implementation) != "eager"
        or type(model).__name__ != "Qwen3_5ForConditionalGeneration"
    ):
        raise ValueError("Dictionary model differs from reviewed native runtime")
    manifest = {
        "identity": identity,
        "calibration_prompts": report["realized_prompts"],
        "pilot_only": False,
        "full_calibration_membership": True,
        "readiness_sha256": evidence["readiness_sha256"],
        "calibration_report_sha256": file_sha256(evidence["paths"]["calibration"]),
        "means_sha256": file_sha256(evidence["paths"]["means"]),
        "calibration_token_manifest_sha256": report["token_manifest_sha256"],
        "native_validation_sha256": report["native_validation_sha256"],
        "source_hashes": {row["file"]: row["sha256"] for row in report["source_files"]},
        "mean_accumulation": "FP64_equal_prompt_sums_then_saved_FP32",
        "arms": {},
    }
    write_dictionary_arms(args, model, text, matrices, manifest)


def write_dictionary_arms(args, model, text, matrices, manifest):
    """Apply the same native norm gain and full-vocabulary normalization for both arms."""
    gain = 1 + text.norm.weight.detach().float()
    unembedding = model.get_output_embeddings().weight.detach()
    for name in ("J", "R"):
        matrix = matrices[name].to(args.device)
        dictionary = torch.empty(unembedding.shape, dtype=torch.float32, device="cpu")
        for start in range(0, len(unembedding), 2048):
            values = (unembedding[start : start + 2048].float() * gain) @ matrix
            norms = values.norm(dim=1)
            valid = torch.isfinite(values).all(1) & (norms > 0)
            normalized = torch.zeros_like(values)
            normalized[valid] = values[valid] / norms[valid, None]
            dictionary[start : start + len(values)] = normalized.cpu()
        eligible = dictionary.norm(dim=1) > 0
        token_ids = torch.arange(len(dictionary))[eligible]
        excluded_ids = torch.arange(len(dictionary))[~eligible]
        dictionary = dictionary[eligible]
        if not len(dictionary):
            raise ValueError("No eligible finite nonzero dictionary rows")
        path = args.out / "dictionaries" / f"{name}.pt"
        save_tensors(
            path,
            {
                "dictionary": dictionary,
                "matrix": matrix.cpu(),
                "identity": manifest,
                "token_ids": token_ids,
                "excluded_token_ids": excluded_ids,
            },
        )
        manifest["arms"][name] = {
            "sha256": file_sha256(path),
            "shape": list(dictionary.shape),
            "excluded_zero_or_nonfinite_rows": len(excluded_ids),
        }
    save_json(args.out / "dictionaries" / "manifest.json", manifest)


def decomposition(args, config, identity):
    """Stream each context through once-validated dictionaries and nested k checkpoints."""
    from explore_persona_space.analysis.workspace_lenses import rotated_dictionary

    manifest = json.loads((args.out / "dictionaries/manifest.json").read_text())
    validate_producer(manifest["identity"], identity)
    if args.subset.startswith("main_") and manifest["pilot_only"]:
        raise ValueError("Pilot calibration matrices cannot serve the main experiment")
    basis = {}
    for name in ("J", "R"):
        path = args.out / "dictionaries" / f"{name}.pt"
        if file_sha256(path) != manifest["arms"][name]["sha256"]:
            raise ValueError("Dictionary checksum mismatch")
        basis[name] = torch.load(path, map_location=args.device, weights_only=True)["dictionary"]
        if args.rotation is not None:
            basis[name] = rotated_dictionary(basis[name], seed=args.rotation)[0]
        basis[name] = ValidatedDictionary(basis[name])
    ks = (
        tuple(
            sorted(
                [config["decomposition"]["k_primary"], *config["decomposition"]["k_sensitivity"]]
            )
        )
        if args.all_k
        else (args.k,)
    )
    contracts = {
        k: {"identity": identity, "dictionaries": manifest, "k": k, "rotation": args.rotation}
        for k in ks
    }
    digests = {k: content_sha256(contract) for k, contract in contracts.items()}
    prompts = phase_prompts(args)
    capture_root = args.out / "captures" / args.subset
    capture_report = json.loads((capture_root / "coverage.json").read_text())
    included_capture = set(
        validate_coverage(capture_report, [r["prompt_sha256"] for r in prompts], identity)
    )
    exclusions = list(capture_report["exclusions"])
    prior_hashes = {}
    for k in ks:
        prior_path = (
            args.out
            / "components"
            / f"k{k}-rotation{args.rotation}"
            / args.subset
            / "coverage.json"
        )
        if prior_path.exists():
            prior = json.loads(prior_path.read_text())
            prior_included = set(
                validate_coverage(prior, [r["prompt_sha256"] for r in prompts], identity)
            )
            if (
                prior["contract"] != contracts[k]
                or prior_included != included_capture
                or prior["exclusions"] != exclusions
            ):
                raise ValueError(
                    "Completed component coverage differs from verified capture inputs"
                )
            prior_hashes[k] = prior["file_sha256"]
    included, hashes = {k: [] for k in ks}, {k: {} for k in ks}
    for row in prompts:
        begin = time.perf_counter()
        source = args.out / "captures" / args.subset / f"{row['prompt_sha256']}.pt"
        if row["prompt_sha256"] not in included_capture:
            continue
        source_digest = file_sha256(source)
        if source_digest != capture_report["file_sha256"][row["prompt_sha256"]]:
            raise ValueError("Capture checkpoint changed after phase completion")
        capture_file = torch.load(source, map_location="cpu", weights_only=True)
        validate_producer(capture_file["identity"]["identity"], identity)
        generation_file = args.out / "generations" / args.subset / f"{row['prompt_sha256']}.json"
        if capture_file["identity"]["generation_file_sha256"] != file_sha256(generation_file):
            raise ValueError("Capture generation source has changed")
        captures = capture_file["rows"]
        for captured in captures:
            validate_context_input(capture_file["identity"], captured["x"], args.out, identity)
        if [r["seed"] for r in captures] != config["generation"]["seeds"] or any(
            r["prompt_sha256"] != row["prompt_sha256"] for r in captures
        ):
            raise ValueError("Capture rollout seeds or context membership mismatch")
        destinations, pending = {}, []
        for k in ks:
            cell = f"k{k}-rotation{args.rotation}"
            target = args.out / "components" / cell / args.subset / source.name
            destinations[k] = target
            if target.exists():
                if (
                    k in prior_hashes
                    and file_sha256(target) != prior_hashes[k][row["prompt_sha256"]]
                ):
                    raise ValueError("Component changed after completed coverage manifest")
                saved = torch.load(target, map_location="cpu", weights_only=True)
                if (
                    saved["contract_sha256"] != digests[k]
                    or saved["source_sha256"] != source_digest
                    or saved["contract"] != contracts[k]
                ):
                    raise ValueError(f"Stale component checkpoint {target}")
                validate_context_input(
                    saved["context_input_reference"], saved["x"], args.out, identity
                )
            else:
                pending.append(k)
        reduced = (
            decompose_context_nested(
                captures, basis, checkpoints=tuple(pending), token_batch_size=128
            )
            if pending
            else {}
        )
        for k in ks:
            target = destinations[k]
            if k in reduced:
                reduced[k].update(
                    contract_sha256=digests[k],
                    contract=contracts[k],
                    source_sha256=source_digest,
                    context_input_reference=capture_file["identity"],
                )
                save_tensors(target, reduced[k])
            included[k].append(row["prompt_sha256"])
            hashes[k][row["prompt_sha256"]] = file_sha256(target)
        print(
            f"decomposed unit={len(included[ks[0]])}/{len(included_capture)} nested_ks={list(ks)} rotation={args.rotation} subset={args.subset} context={row['prompt_sha256']} elapsed={time.perf_counter() - begin:.3f}s",
            flush=True,
        )
    for k in ks:
        cell = f"k{k}-rotation{args.rotation}"
        save_json(
            args.out / "components" / cell / args.subset / "coverage.json",
            {
                "planned_contexts": len(prompts),
                "exclusions": exclusions,
                "contract": contracts[k],
                "identity": identity,
                "status": "complete",
                "included_prompt_sha256": included[k],
                "file_sha256": hashes[k],
            },
        )


def fits(args, config, identity):
    """Fit all component targets on frozen split membership and original units."""
    x, targets, ids = {}, {}, {}
    cell = f"k{args.k}-rotation{args.rotation}"
    output = args.out / "fits" / args.stage / cell
    if (output / "results.json").exists():
        raise FileExistsError(f"Completed fit cannot have its input manifest replaced: {output}")
    coverage = {}
    dictionary_manifest = json.loads((args.out / "dictionaries/manifest.json").read_text())
    validate_producer(dictionary_manifest["identity"], identity)
    expected_contract = {
        "identity": identity,
        "dictionaries": dictionary_manifest,
        "k": args.k,
        "rotation": args.rotation,
    }
    for split in ("train", "validation", "test"):
        subset = f"{args.stage}_{split}"
        prompts = phase_prompts(args, subset)
        root = args.out / "components" / cell / subset
        report = json.loads((root / "coverage.json").read_text())
        included = set(validate_coverage(report, [p["prompt_sha256"] for p in prompts], identity))
        if report["contract"] != expected_contract:
            raise ValueError("Component producer contract differs from fit request")
        coverage[split] = report
        rows = []
        for prompt in prompts:
            path = args.out / "components" / cell / subset / f"{prompt['prompt_sha256']}.pt"
            if prompt["prompt_sha256"] not in included:
                continue
            if file_sha256(path) != report["file_sha256"][prompt["prompt_sha256"]]:
                raise ValueError("Component changed after completed coverage manifest")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            validate_context_input(saved["context_input_reference"], saved["x"], args.out, identity)
            capture_path = args.out / "captures" / subset / path.name
            if (
                saved["contract"] != expected_contract
                or saved["contract_sha256"] != content_sha256(expected_contract)
                or saved["source_sha256"] != file_sha256(capture_path)
                or saved["prompt_sha256"] != prompt["prompt_sha256"]
            ):
                raise ValueError("Component source, prompt or contract mismatch")
            rows.append(saved)
        if len(rows) < 2:
            raise ValueError(f"Insufficient realized contexts: {subset}")
        ids[split] = [row["prompt_sha256"] for row in rows]
        x[split] = np.stack([row["x"].numpy() for row in rows])
        targets[split] = {
            name: np.stack([row["targets"][name].numpy() for row in rows])
            for name in ("full", "J", "restJ", "R", "restR")
        }
    save_json(
        output / "input_manifest.json",
        {"identity": identity, "coverage": coverage, "contract": expected_contract},
    )
    import wandb

    with wandb.init(
        project="workspace-jr",
        name=f"{args.role}-{args.stage}-{cell}",
        config={"identity": identity, "fit": config["fit"], "stage": args.stage, "cell": cell},
    ) as run:
        save_json(output / "wandb.json", {"run_id": run.id, "url": run.url})

        def log_training(record):
            """Send scalar train/validation losses during fitting."""
            values = {k: record[k] for k in ("hidden", "learning_rate", "seed", "epoch")}
            for i, key in enumerate(record["keys"]):
                values[f"{key[0]}/train_scaled_mse_before_step"] = record["train_loss"][i]
                values[f"{key[0]}/validation_scaled_mse_after_step"] = record["validation_loss"][i]
                values[f"{key[0]}/stopped"] = int(record["stopped"][i])
            run.log(values)

        evaluate_component_fits(
            x, targets, ids, config, output, mlp_device=args.device, training_logger=log_training
        )


def main():
    """Run one explicit persisted phase; scientific parameters come from YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("generate", "capture", "dictionaries", "decompose", "fit", "recover-pilot"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/workspace_jr.yaml"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/selected_contexts.json"),
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("docs/exploratory_workspace_jr/mapping_provenance.json")
    )
    parser.add_argument("--role", choices=("primary", "comparison"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--main-readiness", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument(
        "--subset",
        choices=[f"{s}_{t}" for s in ("pilot", "main") for t in ("train", "validation", "test")],
        default="pilot_train",
    )
    parser.add_argument("--stage", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--lens-shards", type=Path)
    parser.add_argument("--token-manifest", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture-batch-size", type=int, default=2)
    parser.add_argument("--k", type=int, choices=(5, 10, 25), default=10)
    parser.add_argument(
        "--all-k", action="store_true", help="Decompose all frozen k values in one nested pass"
    )
    parser.add_argument("--rotation", type=int, choices=(20260913, 20260914, 20260915))
    args = parser.parse_args()
    if args.all_k and args.phase != "decompose":
        raise ValueError("--all-k is only meaningful for the decomposition phase")
    config = load_workspace_jr_config(args.config)
    identity = run_identity(args.config, args.selection, args.role)
    args.readiness = None
    if (args.phase in {"generate", "capture", "decompose"} and args.subset.startswith("main_")) or (
        args.phase in {"fit", "dictionaries"} and args.stage == "main"
    ):
        if args.main_readiness is None:
            raise ValueError(
                "Main phases require completed calibration stability/readout validation"
            )
        args.readiness = validate_main_readiness(
            args.main_readiness,
            config,
            config_path=args.config,
            selection_path=args.selection,
            identity=identity,
        )
        identity["execution_readiness_sha256"] = args.readiness["readiness_sha256"]
    elif args.main_readiness is not None:
        raise ValueError("A main readiness artifact cannot relabel a pilot phase")
    phases = {
        "generate": generation,
        "capture": capture,
        "dictionaries": dictionaries,
        "decompose": decomposition,
        "fit": fits,
        "recover-pilot": recover_pilot_inputs,
    }
    phases[args.phase](args, config, identity)


if __name__ == "__main__":
    main()
