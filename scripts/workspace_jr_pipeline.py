#!/usr/bin/env python3
"""Run exact-token generation, capture, sparse decomposition and component fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.workspace_artifacts import (  # noqa: E402
    validate_coverage,
    validate_producer,
)
from explore_persona_space.analysis.workspace_capture import (  # noqa: E402
    answer_token_ids,
    capture_token_batch,
    decompose_context,
    generate_rollouts,
    save_capture_batch,
)
from explore_persona_space.analysis.workspace_fit import evaluate_component_fits  # noqa: E402
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


def generation(args, config, identity):
    """Generate the frozen subset using the project's batched vLLM engine."""
    from transformers import AutoTokenizer

    from explore_persona_space.eval.generation import create_vllm_engine

    spec = config["models"][args.role]
    model_id = config["selection"][args.role]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=spec["revision"])
    prompts = selected_prompts(args.selection, args.audit, args.subset)
    engine = create_vllm_engine(
        model_id,
        revision=spec["revision"],
        tokenizer_revision=spec["revision"],
        gpu_memory_utilization=0.90,
        max_model_len=32768,
        max_num_seqs=16,
        seed=config["seed"],
        hang_mitigations=True,
    )
    report = generate_rollouts(
        engine, tokenizer, prompts, config, identity, args.out / "generations" / args.subset
    )
    if report["needs_cap_recovery"]:
        raise RuntimeError("Cap-hit fraction exceeds 2%; saved draws require doubled-cap recovery")


def capture(args, config, identity):
    """Capture saved token IDs with visible empty-answer exclusions and K checks."""
    _model, tokenizer, text = load_native(
        config, args.role, device=args.device, dtype=torch.bfloat16
    )
    spec = config["models"][args.role]
    prompts = selected_prompts(args.selection, args.audit, args.subset)
    root = args.out / "generations" / args.subset
    generation_report = json.loads((root / "generation_status.json").read_text())
    validate_coverage(generation_report, [p["prompt_sha256"] for p in prompts], identity)
    if generation_report["needs_cap_recovery"]:
        raise ValueError("Capture requires completed truncation recovery")
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
        contract = {"identity": identity, "generation_file_sha256": file_sha256(source)}
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
    gain = 1 + text.norm.weight.detach().float()
    unembedding = model.get_output_embeddings().weight.detach()
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
    for name in ("J", "R"):
        matrix = torch.stack([row[name].float() for row in rows]).mean(0).to(args.device)
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
    """Stream one captured context at a time through each paired dictionary."""
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
    cell = f"k{args.k}-rotation{args.rotation}"
    contract = {
        "identity": identity,
        "dictionaries": manifest,
        "k": args.k,
        "rotation": args.rotation,
    }
    digest = content_sha256(contract)
    prompts = selected_prompts(args.selection, args.audit, args.subset)
    capture_root = args.out / "captures" / args.subset
    capture_report = json.loads((capture_root / "coverage.json").read_text())
    included_capture = set(
        validate_coverage(capture_report, [r["prompt_sha256"] for r in prompts], identity)
    )
    exclusions = list(capture_report["exclusions"])
    included, hashes = [], {}
    for row in prompts:
        source = args.out / "captures" / args.subset / f"{row['prompt_sha256']}.pt"
        target = args.out / "components" / cell / args.subset / source.name
        if row["prompt_sha256"] not in included_capture:
            continue
        if file_sha256(source) != capture_report["file_sha256"][row["prompt_sha256"]]:
            raise ValueError("Capture checkpoint changed after phase completion")
        capture_file = torch.load(source, map_location="cpu", weights_only=True)
        validate_producer(capture_file["identity"]["identity"], identity)
        generation_file = args.out / "generations" / args.subset / f"{row['prompt_sha256']}.json"
        if capture_file["identity"]["generation_file_sha256"] != file_sha256(generation_file):
            raise ValueError("Capture generation source has changed")
        if target.exists():
            saved = torch.load(target, map_location="cpu", weights_only=True)
            if saved["contract_sha256"] != digest or saved["source_sha256"] != file_sha256(source):
                raise ValueError(f"Stale component checkpoint {target}")
            included.append(row["prompt_sha256"])
            hashes[row["prompt_sha256"]] = file_sha256(target)
            continue
        captures = capture_file["rows"]
        if [r["seed"] for r in captures] != config["generation"]["seeds"] or any(
            r["prompt_sha256"] != row["prompt_sha256"] for r in captures
        ):
            raise ValueError("Capture rollout seeds or context membership mismatch")
        reduced = decompose_context(captures, basis, k=args.k)
        reduced.update(contract_sha256=digest, contract=contract, source_sha256=file_sha256(source))
        save_tensors(target, reduced)
        included.append(row["prompt_sha256"])
        hashes[row["prompt_sha256"]] = file_sha256(target)
        print(
            f"decomposed cell={cell} subset={args.subset} context={row['prompt_sha256']}",
            flush=True,
        )
    save_json(
        args.out / "components" / cell / args.subset / "coverage.json",
        {
            "planned_contexts": len(prompts),
            "exclusions": exclusions,
            "contract": contract,
            "identity": identity,
            "status": "complete",
            "included_prompt_sha256": included,
            "file_sha256": hashes,
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
        prompts = selected_prompts(args.selection, args.audit, subset)
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
    evaluate_component_fits(x, targets, ids, config, output, mlp_device=args.device)


def main():
    """Run one explicit persisted phase; scientific parameters come from YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("generate", "capture", "dictionaries", "decompose", "fit")
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
    parser.add_argument("--rotation", type=int, choices=(20260913, 20260914, 20260915))
    args = parser.parse_args()
    config = load_workspace_jr_config(args.config)
    identity = run_identity(args.config, args.selection, args.role)
    if (args.phase in {"generate", "capture", "decompose"} and args.subset.startswith("main_")) or (
        args.phase == "fit" and args.stage == "main"
    ):
        raise ValueError("Main phases require completed calibration stability/readout validation")
    phases = {
        "generate": generation,
        "capture": capture,
        "dictionaries": dictionaries,
        "decompose": decomposition,
        "fit": fits,
    }
    phases[args.phase](args, config, identity)


if __name__ == "__main__":
    main()
