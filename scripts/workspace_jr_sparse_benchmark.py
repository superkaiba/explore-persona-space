#!/usr/bin/env python3
"""Benchmark nested FP32 sparse pursuit on frozen pilot-training tokens only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402

from explore_persona_space.analysis.workspace_analysis_inputs import _upload_binding  # noqa: E402
from explore_persona_space.analysis.workspace_artifacts import (  # noqa: E402
    validate_context_input,
    validate_producer,
)
from explore_persona_space.analysis.workspace_decomposition import ValidatedDictionary  # noqa: E402
from explore_persona_space.analysis.workspace_lenses import (  # noqa: E402
    SparseComponent,
    nonnegative_gradient_pursuit,
)
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    content_sha256,
    file_sha256,
    load_workspace_jr_config,
    run_identity,
    save_json,
    save_tensors,
)


def parity(actual, reference):
    """Freeze strict component error gates before timing a candidate; expose all fields."""
    report = {}
    passed = True
    expected_fields = set(SparseComponent.__dataclass_fields__)
    if set(actual) != set(reference):
        raise ValueError("Checkpoint membership differs")
    for k, expected in reference.items():
        if set(actual[k]) != expected_fields or set(expected) != expected_fields:
            raise ValueError("Sparse result field membership differs")
        fields = {}
        for name in SparseComponent.__dataclass_fields__:
            a, b = actual[k][name], expected[name]
            if a.shape != b.shape or a.dtype != b.dtype or not a.numel():
                raise ValueError(f"Geometry or dtype differs: k={k}/{name}")
            if a.is_floating_point():
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError(f"Nonfinite field: k={k}/{name}")
                relative = float((a.double() - b).norm() / b.double().norm().clamp_min(1e-30))
                fields[name] = {"relative_frobenius_error": relative}
                passed &= bool(torch.allclose(a, b, rtol=1e-5, atol=1e-6))
            else:
                equal = float((a == b).double().mean())
                fields[name] = {"exact_element_fraction": equal}
                passed &= equal == 1.0
        report[str(k)] = fields
    return {"passed": passed, "fields": report}


@torch.no_grad()
def execute(x, dictionary, batch_size, *, nested, segments):
    """Time the real kernels with device synchronization and bounded output copies."""
    executor = ValidatedDictionary(dictionary) if nested else None
    if not segments or min(segments) < 1 or sum(segments) != len(x):
        raise ValueError("Invalid rollout segment boundaries")
    slices, begin = [], 0
    for length in segments:
        slices.extend(
            slice(start, min(start + batch_size, begin + length))
            for start in range(begin, begin + length, batch_size)
        )
        begin += length
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    outputs = {k: {name: [] for name in SparseComponent.__dataclass_fields__} for k in (5, 10, 25)}
    for segment in slices:
        batch = x[segment].to(dictionary.device)
        values = (
            executor.decompose(batch)
            if nested
            else {k: nonnegative_gradient_pursuit(batch, dictionary, k=k) for k in (5, 10, 25)}
        )
        for k, result in values.items():
            for name in outputs[k]:
                outputs[k][name].append(getattr(result, name).cpu())
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    return (
        {
            k: {name: torch.cat(chunks) for name, chunks in fields.items()}
            for k, fields in outputs.items()
        },
        {
            "seconds": seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "tokens": len(x),
        },
    )


def main():
    """Bind uploaded dictionaries/captures before any GPU benchmark, save every candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--upload-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", choices=("primary", "comparison"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/workspace_jr.yaml"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/selected_contexts.json"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Benchmark requires a fresh output directory")
    config = load_workspace_jr_config(args.config)
    identity = run_identity(args.config, args.selection, args.role)
    verify = _upload_binding(args.input_root, args.upload_receipt)
    manifest_path = args.input_root / "dictionaries/manifest.json"
    verify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    producer = manifest["identity"]
    validate_producer(producer, identity, native_ancestor=True)
    selection = json.loads(args.selection.read_text())
    captures, sources = [], {}
    for prompt in selection["subsets"]["pilot_train"][:4]:
        path = args.input_root / "captures/pilot_train" / f"{prompt['prompt_sha256']}.pt"
        sources[str(path.relative_to(args.input_root))] = verify(path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        validate_producer(saved["identity"]["identity"], producer)
        if [row["seed"] for row in saved["rows"]] != config["generation"]["seeds"]:
            raise ValueError("Pilot capture seeds differ")
        generation = args.input_root / "generations/pilot_train" / f"{prompt['prompt_sha256']}.json"
        if verify(generation) != saved["identity"]["generation_file_sha256"]:
            raise ValueError("Pilot capture generation source differs")
        verify(args.input_root / saved["identity"]["context_input_file"])
        for row in saved["rows"]:
            if row["prompt_sha256"] != prompt["prompt_sha256"]:
                raise ValueError("Capture context differs from frozen pilot-training sample")
            validate_context_input(saved["identity"], row["x"], args.input_root, producer)
            captures.append(row["answer_states"].float())
    selected, segments, remaining = [], [], 2047
    for capture in captures:
        count = min(remaining, len(capture))
        if count:
            selected.append(capture[:count])
            segments.append(count)
            remaining -= count
    x = torch.cat(selected).contiguous()
    if len(x) < 1024 or x.shape[1] != config["models"][args.role]["d_model"]:
        raise ValueError("Insufficient real pilot-training token coverage for batch benchmark")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.mkdir(parents=True)
    report = {
        "identity": identity,
        "input_sources": sources,
        "upload_receipt_sha256": file_sha256(args.upload_receipt),
        "dictionary_manifest_sha256": file_sha256(manifest_path),
        "sampling": "first up to 2047 tokens from first four frozen pilot train contexts, rollout order",
        "rollout_segment_lengths": segments,
        "batch_geometry": "chunk independently within each original rollout, retaining ragged tails",
        "precision": "FP32 highest, TF32 disabled",
        "ks": [5, 10, 25],
        "gate": "every float field allclose rtol=1e-5 atol=1e-6; every integer field exactly equal",
        "candidates": [128, 256, 512, 1024],
        "arms": {},
        "status": "running",
    }
    save_json(args.out / "benchmark_started.json", report)
    save_tensors(
        args.out / "activation_sample.pt", {"x": x, "sources": sources, "segments": segments}
    )
    for arm in ("J", "R"):
        path = args.input_root / "dictionaries" / f"{arm}.pt"
        if verify(path) != manifest["arms"][arm]["sha256"]:
            raise ValueError("Uploaded dictionary differs from decomposition manifest")
        dictionary = torch.load(path, map_location="cuda:0", weights_only=True)["dictionary"]
        # Warm CUDA without using any alternative precision or changing dictionary bytes.
        nonnegative_gradient_pursuit(x[:8].cuda(), dictionary, k=5)
        reference, timing = execute(x, dictionary, 128, nested=False, segments=segments)
        arm_report = {"reference": timing, "dictionary_sha256": file_sha256(path), "candidates": {}}
        report["arms"][arm] = arm_report
        save_tensors(args.out / f"{arm}_reference.pt", reference)
        for batch in report["candidates"]:
            arm_report["candidates"][str(batch)] = {"status": "running"}
            save_json(args.out / "benchmark_partial.json", report)
            begin = time.perf_counter()
            try:
                actual, measured = execute(x, dictionary, batch, nested=True, segments=segments)
                save_tensors(args.out / f"{arm}_batch{batch}.pt", actual)
                comparison = parity(actual, reference)
            except Exception as error:
                arm_report["candidates"][str(batch)] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "seconds": time.perf_counter() - begin,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                report["status"] = "failed"
                save_json(args.out / "benchmark_failed.json", report)
                raise
            arm_report["candidates"][str(batch)] = {
                "status": "completed",
                **measured,
                **comparison,
                "speedup_over_three_reference_passes": timing["seconds"] / measured["seconds"],
            }
            save_json(args.out / "benchmark_partial.json", report)
            print(
                f"sparse benchmark arm={arm} batch={batch} seconds={measured['seconds']:.3f} parity={comparison['passed']}",
                flush=True,
            )
        del dictionary, reference, actual
        torch.cuda.empty_cache()
    report["status"] = "complete"
    report["automatic_adoption"] = False
    report["report_content_sha256"] = content_sha256(report)
    save_json(args.out / "benchmark.json", report)


if __name__ == "__main__":
    main()
