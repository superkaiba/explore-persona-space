#!/usr/bin/env python3
"""Inspect frozen calibration convergence and native J/R token readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402

from explore_persona_space.analysis.workspace_artifacts import validate_producer  # noqa: E402
from explore_persona_space.analysis.workspace_calibration import (  # noqa: E402
    calibration_means,
    eligible_calibration_tokens,
    matrix_agreement,
    paired_direction_agreement,
    validate_calibration_order,
)
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    content_sha256,
    file_sha256,
    load_native,
    load_workspace_jr_config,
    run_identity,
    save_json,
    save_tensors,
    validate_token_manifest,
)


@torch.no_grad()
def readouts(model, text, tokenizer, tokens, means, source_layer, skip_first):
    """First eight calibration prompts, four evenly spaced valid positions each.

    Match the official apply convention: transport residual, then native final
    normalization and unembedding in model dtype. Dictionary row normalization
    is for sparse geometry and is deliberately absent from these logits.
    """
    records = []
    head = model.get_output_embeddings()
    device, dtype = head.weight.device, head.weight.dtype
    for row in tokens["rows"][:8]:
        ids = torch.tensor([row["token_ids"]], device=device)
        cache = {}

        def hook(_module, _inputs, output):
            cache["states"] = output if isinstance(output, torch.Tensor) else output[0]

        handle = text.layers[source_layer].register_forward_hook(hook)
        try:
            output = text(input_ids=ids, use_cache=False)
        finally:
            handle.remove()
        positions = (
            torch.linspace(skip_first, len(row["token_ids"]) - 2, 4)
            .round()
            .long()
            .unique()
            .to(device)
        )
        h = cache["states"][0, positions].float()
        scores = {"native": head(output.last_hidden_state[0, positions]).float()}
        for arm in ("J", "R"):
            transported = h @ means[arm].to(device).T
            scores[arm] = head(text.norm(transported.to(dtype))).float()
        tops = {
            arm: values.topk(10, dim=1).indices.cpu().tolist() for arm, values in scores.items()
        }
        for i, position in enumerate(positions.tolist()):
            records.append(
                {
                    "prompt_sha256": row["prompt_sha256"],
                    "position": position,
                    "prefix": tokenizer.decode(row["token_ids"][: position + 1]),
                    "observed_next_token_id": row["token_ids"][position + 1],
                    "top_token_ids": {arm: top[i] for arm, top in tops.items()},
                    "top_tokens": {
                        arm: [tokenizer.decode([token]) for token in top[i]]
                        for arm, top in tops.items()
                    },
                }
            )
    return records


def main():
    """Produce evidence only; no test reads or automatic stability threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/workspace_jr.yaml"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/selected_contexts.json"),
    )
    parser.add_argument("--role", choices=("primary", "comparison"), required=True)
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--lens-shards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--matrix-only", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a fresh calibration output directory; preserve prior diagnostics")
    config = load_workspace_jr_config(args.config)
    identity = run_identity(args.config, args.selection, args.role)
    tokens = json.loads(args.token_manifest.read_text())
    validate_calibration_order(tokens, json.loads(args.selection.read_text()))
    validate_token_manifest(
        tokens,
        config_path=args.config,
        selection_path=args.selection,
        config=config,
        role=args.role,
    )
    validation = json.loads(args.validation.read_text())
    proof = validate_producer(validation["identity"], identity, native_ancestor=True)
    if validation["model_dtype"] != config["lenses"]["calibration_dtype"]:
        raise ValueError("Native validation differs from frozen calibration precision")
    if (
        validation["token_manifest_sha256"] != content_sha256(tokens)
        or validation["ordinary_numerical_validation"]["status"] != "passed"
        or not validation["forward_bit_identical"]
        or not validation["hook_bit_identical"]
    ):
        raise ValueError("Calibration inspection requires native validation of these exact tokens")
    paths = sorted(args.lens_shards.glob("prompt-*.pt"))
    # Verify producer and validation identity before aggregating any matrices.
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        validate_producer(saved["contract"], validation["identity"])
        if (
            saved["contract"]["actual_dtype"] != f"torch.{validation['model_dtype']}"
            or saved["contract"]["attention_implementation"] != "eager"
            or saved["contract"]["model_class"] != "Qwen3_5ForConditionalGeneration"
        ):
            raise ValueError("Calibration shard differs from validated native runtime")
        if saved["contract"]["native_validation_sha256"] != file_sha256(args.validation):
            raise ValueError("Calibration shard used different native validation")
    spec = config["models"][args.role]
    means, report = calibration_means(paths, tokens, spec["d_model"])
    report.update(
        identity=identity,
        native_producer_proof=proof,
        native_validation_sha256=file_sha256(args.validation),
    )
    full = report["full_group"]
    comparisons = [(name, full) for name in means if name != full]
    if "even" in means:
        comparisons.append(("even", "odd"))
    report["matrix_agreement"] = {
        f"{a}_vs_{b}": {arm: matrix_agreement(means[a][arm], means[b][arm]) for arm in ("J", "R")}
        for a, b in comparisons
    }
    report["j_r_matrix_agreement"] = matrix_agreement(means[full]["J"], means[full]["R"])
    save_tensors(args.out / "means.pt", means)
    report["means_sha256"] = file_sha256(args.out / "means.pt")
    if not args.matrix_only:
        model, tokenizer, text = load_native(
            config, args.role, device=args.device, dtype=torch.bfloat16
        )
        if (
            str(next(model.parameters()).dtype) != f"torch.{validation['model_dtype']}"
            or str(text.config._attn_implementation) != "eager"
            or type(model).__name__ != "Qwen3_5ForConditionalGeneration"
        ):
            raise ValueError("Readout model differs from calibrated native runtime")
        eligible = eligible_calibration_tokens(
            tokenizer,
            tokens,
            seed=config["seed"],
            maximum=config["diagnostics"]["direction_maximum"],
        )
        if not eligible:
            raise ValueError("No eligible calibration token directions")
        weight = model.get_output_embeddings().weight[eligible].float() * (
            1 + text.norm.weight.float()
        )
        directions = {
            name: {arm: (weight @ matrix.to(args.device)).cpu() for arm, matrix in arms.items()}
            for name, arms in means.items()
        }
        direction_results = {
            f"{a}_vs_{b}": {
                arm: paired_direction_agreement(directions[a][arm], directions[b][arm])
                for arm in ("J", "R")
            }
            for a, b in comparisons
        }
        direction_results["J_vs_R"] = paired_direction_agreement(
            directions[full]["J"], directions[full]["R"]
        )
        save_tensors(
            args.out / "directions.pt", {"token_ids": eligible, "comparisons": direction_results}
        )
        report["directions_sha256"] = file_sha256(args.out / "directions.pt")
        report["eligible_token_count"] = len(eligible)
        report["readouts"] = readouts(
            model,
            text,
            tokenizer,
            tokens,
            means[full],
            spec["source_layer"],
            config["lenses"]["skip_first"],
        )
    report["status"] = "matrix_only" if args.matrix_only else "calibration_diagnostics_complete"
    report["main_approval"] = False
    report["interpretation"] = (
        "Calibration evidence requires review; successful computation does not assert convergence."
    )
    save_json(args.out / "calibration_report.json", report)
    print(
        f"calibration_diagnostics status={report['status']} full_membership={report['full_calibration_membership']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
