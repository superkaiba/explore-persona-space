#!/usr/bin/env python3
"""Compare original and EOS-excluded BF16 capture geometry on four frozen pilot contexts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402

from explore_persona_space.analysis.workspace_analysis_inputs import _upload_binding  # noqa: E402
from explore_persona_space.analysis.workspace_artifacts import validate_producer  # noqa: E402
from explore_persona_space.analysis.workspace_capture import capture_token_batch  # noqa: E402
from explore_persona_space.analysis.workspace_fit import finite_json  # noqa: E402
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    content_sha256,
    file_sha256,
    load_native,
    load_workspace_jr_config,
    run_identity,
    save_json,
    save_tensors,
)
from explore_persona_space.analysis.workspace_terminal_recovery import (  # noqa: E402
    SOURCE_RECEIPT_SHA256,
)
from explore_persona_space.analysis.workspace_terminals import (  # noqa: E402
    terminal_policy,
    trim_saved_answer,
)


def compare(a, b, gate):
    """Use the existing numerical yardstick without relaxing undefined cosines."""
    if a.shape != b.shape or a.ndim != 2 or not len(a):
        raise ValueError("Parity comparison requires aligned nonempty matrices")
    bit_identical = a.dtype == b.dtype and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8),
        b.contiguous().reshape(-1).view(torch.uint8),
    )
    a, b = a.double(), b.double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite native parity states")
    denom = a.norm(dim=1) * b.norm(dim=1)
    cosine = (a * b).sum(1) / denom
    relative = (a - b).norm() / a.norm()
    valid = denom > 0
    return finite_json(
        {
            "rows": len(a),
            "dimensions": a.shape[1],
            "relative_frobenius_error": float(relative),
            "row_cosine": cosine.tolist(),
            "minimum_row_cosine": float(cosine.min()),
            "zero_norm_rows": int((~valid).sum()),
            "bit_identical": bit_identical,
            "passed": bool(
                valid.all()
                and torch.isfinite(relative)
                and relative <= gate["maximum_relative_frobenius_error"]
                and cosine.min() >= gate["minimum_row_cosine"]
            ),
        }
    )


def main():
    """Keep the preregistered first-four selection independent of parity outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/workspace_jr.yaml"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/selected_contexts.json"),
    )
    args = parser.parse_args()
    if args.out.exists() or file_sha256(args.source_receipt) != SOURCE_RECEIPT_SHA256:
        raise ValueError("Parity requires a fresh output and the exact audited source receipt")
    config = load_workspace_jr_config(args.config)
    identity = run_identity(args.config, args.selection, "comparison")
    contexts = json.loads(args.selection.read_text())["subsets"]["pilot_train"][:4]
    gate = config["provenance_gate"]
    args.out.mkdir(parents=True)
    plan = {
        "identity": identity,
        "contexts": contexts,
        "scope": "diagnostic numerical yardstick, not historical-map parity or a threshold waiver",
        "thresholds": {
            "relative_frobenius": gate["maximum_relative_frobenius_error"],
            "minimum_row_cosine": gate["minimum_row_cosine"],
        },
        "capture_batches": [2, 2, 1],
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
    }
    save_json(args.out / "parity_plan.json", plan)
    verify = _upload_binding(args.source, args.source_receipt)
    sources = []
    for context in contexts:
        key = context["prompt_sha256"]
        capture = args.source / "captures/pilot_train" / f"{key}.pt"
        generation = args.source / "generations/pilot_train" / f"{key}.json"
        capture_sha, generation_sha = verify(capture), verify(generation)
        saved = torch.load(capture, map_location="cpu", weights_only=True)
        raw = json.loads(generation.read_text())
        producer = saved["identity"]["identity"]
        validate_producer(producer, identity, native_ancestor=True)
        validate_producer(raw["contract"]["identity"], producer)
        if (
            saved["identity"]["generation_file_sha256"] != generation_sha
            or raw["contract_sha256"] != content_sha256(raw["contract"])
            or raw["contract"]["generation"] != config["generation"]
            or raw["contract"]["prompt_sha256"] != key
            or [r["seed"] for r in raw["rollouts"]] != config["generation"]["seeds"]
            or [r["seed"] for r in saved["rows"]] != config["generation"]["seeds"]
            or any(
                row["prompt_sha256"] != key
                or row["prompt_ids"] != raw["contract"]["prompt_token_ids"]
                for row in saved["rows"]
            )
        ):
            raise ValueError("Parity source generation binding or rollout order differs")
        sources.append((context, capture_sha, generation_sha, saved, raw))
    model, tokenizer, text = load_native(
        config, "comparison", device="cuda:0", dtype=torch.bfloat16
    )
    policy = terminal_policy(model.generation_config, tokenizer)
    if policy["terminal_ids"] != [248044, 248046]:
        raise ValueError("Native terminal IDs differ from the reviewed checkpoint")
    pairs = {"original_geometry_recapture": [[], []], "trimmed_prefix_vs_future_capture": [[], []]}
    pooled = {name: [[], []] for name in pairs}
    report = {"plan": plan, "terminal_policy": policy, "contexts": []}
    for context, capture_sha, generation_sha, saved, raw in sources:
        key = context["prompt_sha256"]
        trimmed = [
            trim_saved_answer(row, draw, policy, {248044})[0]
            for row, draw in zip(saved["rows"], raw["rollouts"], strict=True)
        ]
        if any(not row["answer_ids"] for row in trimmed):
            raise ValueError(
                "Predeclared parity sample has an EOS-only answer; no replacement context"
            )
        recaptured = {"original": [], "future": []}
        for name, rows in (("original", saved["rows"]), ("future", trimmed)):
            for begin in range(0, 5, 2):
                recaptured[name].extend(
                    capture_token_batch(
                        text,
                        rows[begin : begin + 2],
                        config["models"]["comparison"]["source_layer"],
                        tokenizer.pad_token_id,
                    )
                )
        values = {
            "original_geometry_recapture": (
                [r["answer_states"] for r in saved["rows"]],
                [r["answer_states"] for r in recaptured["original"]],
            ),
            "trimmed_prefix_vs_future_capture": (
                [r["answer_states"] for r in trimmed],
                [r["answer_states"] for r in recaptured["future"]],
            ),
        }
        results = {}
        for name, (a, b) in values.items():
            pairs[name][0].extend(a)
            pairs[name][1].extend(b)
            pooled[name][0].append(torch.stack([h.double().mean(0) for h in a]).mean(0))
            pooled[name][1].append(torch.stack([h.double().mean(0) for h in b]).mean(0))
            results[name] = {
                "all_token_states": compare(torch.cat(a), torch.cat(b), gate),
                "equal_rollout_context_mean": compare(
                    pooled[name][0][-1][None], pooled[name][1][-1][None], gate
                ),
            }
        output = args.out / f"context-{key}.pt"
        save_tensors(
            output,
            {
                "original_capture_sha256": capture_sha,
                "generation_sha256": generation_sha,
                "recaptured": recaptured,
            },
        )
        report["contexts"].append(
            {
                "prompt_sha256": key,
                "original_capture_sha256": capture_sha,
                "generation_sha256": generation_sha,
                "output_sha256": file_sha256(output),
                "comparisons": results,
            }
        )
        save_json(args.out / "parity_partial.json", report)
        print(
            f"Terminal geometry parity contexts={len(report['contexts'])}/4 key={key}", flush=True
        )
    report["aggregate"] = {
        name: {
            "all_token_states": compare(torch.cat(a), torch.cat(b), gate),
            "equal_rollout_context_means": compare(
                torch.stack(pooled[name][0]), torch.stack(pooled[name][1]), gate
            ),
        }
        for name, (a, b) in pairs.items()
    }
    report["passed"] = all(
        r["passed"] for values in report["aggregate"].values() for r in values.values()
    ) and all(
        r["passed"]
        for context in report["contexts"]
        for values in context["comparisons"].values()
        for r in values.values()
    )
    report["status"] = "passed" if report["passed"] else "failed_requires_fresh_pilot_capture"
    save_json(args.out / "terminal_parity.json", report)
    if not report["passed"]:
        raise RuntimeError("EOS prefix geometry exceeded the unchanged diagnostic yardstick")


if __name__ == "__main__":
    main()
