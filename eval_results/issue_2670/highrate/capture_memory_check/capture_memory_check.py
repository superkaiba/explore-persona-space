#!/usr/bin/env python3
"""Measure two actual capture shapes with frozen helpers and no generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

REPOSITORY = Path("/workspace/explore-persona-space")
ROOT = Path("/workspace/logs/issue2670-context-risk-highrate")
SOURCES = ("capture_memory_check.py", "capture_memory_supervise.sh")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> str:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return sha256(path)


def choose_shapes(lengths: list[int], recipe: dict, batches_by_budget) -> tuple[int, list[int]]:
    longest = max(range(len(lengths)), key=lengths.__getitem__)
    pairs = []
    for start in range(0, len(lengths), recipe["checkpoint_rows"]):
        local = lengths[start : start + recipe["checkpoint_rows"]]
        for batch in batches_by_budget(local, recipe["batch_max_rows"], recipe["batch_max_tokens"]):
            if len(batch) == 2:
                pairs.append([start + index for index in batch])
    if not pairs:
        raise ValueError("The actual production packing contains no eligible two-row batch")
    # Largest padded shape; unequal lengths and earliest indices only break exact-size ties.
    pair = max(
        pairs,
        key=lambda indices: (
            2 * max(lengths[index] for index in indices),
            int(lengths[indices[0]] != lengths[indices[1]]),
            tuple(-index for index in indices),
        ),
    )
    return longest, pair


def validate_payload(vectors, replay, attention, ids_rows: list[list[int]]) -> None:
    import numpy as np

    width = max(map(len, ids_rows))
    if (
        vectors.shape != (len(ids_rows), 1, 5120)
        or vectors.dtype != np.float16
        or not np.isfinite(vectors).all()
        or replay.shape != (len(ids_rows), width)
        or attention.shape != replay.shape
    ):
        raise ValueError("Capture shape or finite-value gate failed")
    for lane, ids in enumerate(ids_rows):
        if (
            not np.array_equal(attention[lane], np.arange(width) < len(ids))
            or replay[lane, : len(ids)].tolist() != ids
        ):
            raise ValueError("Unpadded token replay or attention mask differs")


def main() -> None:
    launch_id = os.environ["EPM_CONTEXT_RISK_LAUNCH_ID"]
    expected_input_sha = os.environ["EPM_CONTEXT_RISK_HIGHRATE_CAPTURE_INPUT_SHA256"]
    deadline = os.environ["EPM_CONTEXT_RISK_PROCESS_TIMEOUT_SECONDS"]
    if (
        not re.fullmatch(r"[a-zA-Z0-9_-]+", launch_id)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_input_sha)
        or not re.fullmatch(r"[1-9][0-9]*", deadline)
        or Path(os.environ["EPM_CONTEXT_RISK_HIGHRATE_ROOT"]).resolve() != ROOT
        or Path(__file__).resolve() != ROOT / "setup" / SOURCES[0]
    ):
        raise ValueError("Diagnostic launch/root/prepared-input binding differs")
    expected_env = {
        "HF_HOME": "/workspace/.cache/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LD_LIBRARY_PATH": "/usr/local/cuda-13.0/compat:",
        "PYTHONPATH": "/workspace/explore-persona-space/src:/workspace/explore-persona-space",
        "CUDA_VISIBLE_DEVICES": "0",
        **dict.fromkeys(
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"),
            "8",
        ),
    }
    if any(os.environ.get(name) != value for name, value in expected_env.items()):
        raise ValueError("Diagnostic cache, CUDA library, import path or thread settings differ")
    setup = ROOT / "setup"
    stem = f"capture_memory_{launch_id}"
    output = setup / f"{stem}.json"
    failed = setup / f"{stem}.failed.json"
    launch = setup / f"{stem}.launch.json"
    payloads = {
        name: setup / f"{stem}.{name}.npz" for name in ("longest_singleton", "production_two_row")
    }
    if any(
        path.exists() or path.is_symlink() for path in (output, failed, launch, *payloads.values())
    ):
        raise FileExistsError("Diagnostic requires fresh launch, report and payload paths")
    report = {
        "schema_version": "context_risk_highrate_capture_memory_v1",
        "passed": False,
        "launch_id": launch_id,
        "started_unix": time.time(),
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "deadline_seconds": int(deadline),
        "generation_calls": 0,
        "prefixes_truncated": 0,
        "phases": [],
        "production_capture": False,
        "equivalence_test_performed": False,
        "public_environment": expected_env,
    }
    try:
        sys.path[:0] = [str(REPOSITORY / "src"), str(REPOSITORY)]
        import numpy as np
        import torch
        from omegaconf import OmegaConf

        from explore_persona_space.analysis.extraction import _logits_to_keep_kwargs
        from scripts import context_risk_highrate_capture as capture
        from scripts.context_risk_qwen38_capture import batches_by_budget
        from scripts.context_risk_qwen38_smoke import (
            _capture_last_prefix,
            _load_model_and_tokenizer,
            render_prefix_ids,
        )

        review_path = Path(os.environ["EPM_CONTEXT_RISK_HIGHRATE_CAPTURE_MEMORY_REVIEW"])
        capture_review_path = Path(os.environ["EPM_CONTEXT_RISK_HIGHRATE_CAPTURE_REVIEW"])
        sources = {name: sha256(setup / name) for name in SOURCES}
        control_paths = {
            review_path,
            capture_review_path,
            ROOT / "capture_inputs.json",
            *(ROOT / name for name in capture.INPUT_PATHS.values()),
        }
        control_paths.update(
            ROOT / name
            for name in (
                "setup/postrun_code_review.json",
                "screen_B/postrun_audit.json",
                "fresh_B/postrun_audit.json",
            )
            if (ROOT / name).exists()
        )
        controls = {str(path): sha256(path) for path in sorted(control_paths)}
        capture_sources = capture.source_hashes()
        imports = capture.imported_source_hashes()
        review = json.loads(review_path.read_text())
        if (
            review["verdict"] != "PASS"
            or review["sources_sha256"] != sources
            or review["capture_sources_sha256"] != capture_sources
        ):
            raise ValueError("Diagnostic requires its current independent source-bound review")
        cfg = OmegaConf.create({"model": capture.MODEL, "review": str(capture_review_path)})
        capture_review, capture_review_sha = capture._review(cfg)
        prepared, rows, prefixes = capture._staged_inputs(ROOT, expected_input_sha)
        if (
            prepared["independent_review"] != capture_review
            or prepared["independent_review_sha256"] != capture_review_sha
        ):
            raise ValueError("Prepared inputs and configured capture review differ")
        runtime = capture._runtime()
        lengths = [prefixes[row["exact_context_sha256"]]["n_prefix_tokens"] for row in rows]
        longest, pair = choose_shapes(lengths, capture.CAPTURE, batches_by_budget)
        selected = sorted(set([longest, *pair]))
        report.update(
            sources_sha256=sources,
            capture_sources_sha256=capture_sources,
            imported_sources_sha256=imports,
            controls_sha256=controls,
            capture_inputs_sha256=expected_input_sha,
            runtime=runtime,
            model=capture.MODEL,
            capture=capture.CAPTURE,
            activation_position=capture.POSITION,
            longest_index=longest,
            production_pair_indices=pair,
            selected_rows=[
                {
                    "manifest_index": index,
                    "task_id": rows[index]["task_id"],
                    "condition": rows[index]["condition"],
                    "exact_context_sha256": rows[index]["exact_context_sha256"],
                    "n_prefix_tokens": lengths[index],
                }
                for index in selected
            ],
        )
        launch_sha = write_json(launch, report)
        if (
            sources != {name: sha256(setup / name) for name in SOURCES}
            or capture_sources != capture.source_hashes()
            or imports != capture.imported_source_hashes()
            or controls != {path: sha256(Path(path)) for path in controls}
        ):
            raise ValueError("Diagnostic sources or controls changed before model loading")
        load_started = time.monotonic()
        model, tokenizer, depth = _load_model_and_tokenizer(cfg)
        torch.cuda.synchronize()
        if (
            next(model.parameters()).dtype != torch.bfloat16
            or model.training
            or _logits_to_keep_kwargs(model, False) != {"logits_to_keep": 1}
        ):
            raise ValueError("Model dtype/mode or last-logit-only capture path differs")
        report.update(
            model_load_seconds=time.monotonic() - load_started,
            wrapper_depth=depth,
            gpu_name=torch.cuda.get_device_name(0),
            loaded_model_allocated_bytes=torch.cuda.memory_allocated(),
            loaded_model_reserved_bytes=torch.cuda.memory_reserved(),
        )
        ids = {}
        for index in selected:
            _, ids[index] = render_prefix_ids(
                tokenizer, rows[index]["messages"], enable_thinking=False
            )
            if ids[index] != prefixes[rows[index]["exact_context_sha256"]]["token_ids"]:
                raise ValueError("Live tokenizer differs from saved exact generation IDs")
        for name, indices in (("longest_singleton", [longest]), ("production_two_row", pair)):
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            phase = {
                "name": name,
                "indices": indices,
                "token_lengths": [lengths[index] for index in indices],
                "padded_tokens": len(indices) * max(lengths[index] for index in indices),
                "free_before_bytes": free,
                "total_bytes": total,
                "allocated_before_bytes": torch.cuda.memory_allocated(),
                "reserved_before_bytes": torch.cuda.memory_reserved(),
                "passed": False,
            }
            report["phases"].append(phase)
            values, input_ids, mask = _capture_last_prefix(model, [ids[i] for i in indices], [44])
            torch.cuda.synchronize()
            phase.update(
                elapsed_seconds=time.monotonic() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            vectors = values.to(torch.float16).cpu().numpy()
            replay = input_ids.cpu().numpy()
            attention = mask.cpu().numpy()
            validate_payload(vectors, replay, attention, [ids[index] for index in indices])
            with payloads[name].open("xb") as handle:
                np.savez(handle, activation=vectors, input_ids=replay, attention_mask=attention)
            phase.update(
                passed=True, payload_file=payloads[name].name, sha256=sha256(payloads[name])
            )
            print(json.dumps(phase, sort_keys=True), flush=True)
            del values, input_ids, mask, vectors, replay, attention
        if (
            prepared != capture._staged_inputs(ROOT, expected_input_sha)[0]
            or runtime != capture._runtime()
            or sources != {name: sha256(setup / name) for name in SOURCES}
            or capture_sources != capture.source_hashes()
            or imports != capture.imported_source_hashes()
            or sha256(launch) != launch_sha
            or any(sha256(setup / p["payload_file"]) != p["sha256"] for p in report["phases"])
            or controls != {path: sha256(Path(path)) for path in controls}
        ):
            raise ValueError("Diagnostic sources, inputs, runtime or payloads changed")
        report.update(passed=True, finished_unix=time.time(), launch_sha256=launch_sha)
        write_json(output, report)
        print(json.dumps({"passed": True, "report": str(output)}), flush=True)
    except Exception as error:
        report.update(
            passed=False,
            error_type=type(error).__name__,
            error=str(error),
            finished_unix=time.time(),
        )
        write_json(failed, report)
        raise


if __name__ == "__main__":
    main()
