#!/usr/bin/env python3
"""Measure larger native VJP batches against the completed dim-batch8 pilot pair."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

from explore_persona_space.analysis.workspace_runtime import file_sha256, save_json


def gpu_preflight():
    """Enforce an idle single GPU and record its actual hardware identity."""
    devices = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(devices) != 1 or processes:
        raise ValueError("Benchmark requires exactly one GPU with no active compute processes")
    return devices[0]


def sample_child(command, log_path, candidate, report_path, report):
    """Always reap a failed sampler's child and persist its terminal outcome."""
    candidate.update(status="starting", command=command)
    report["candidates"].append(candidate)
    save_json(report_path, report)
    began, process, samples = time.monotonic(), None, []
    try:
        candidate["hardware_before_launch"] = gpu_preflight()
        with log_path.open("w") as handle:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            candidate["status"] = "running"
            while process.poll() is None:
                memory = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines()
                if len(memory) != 1:
                    raise ValueError("Benchmark requires an exclusive single-GPU worker")
                samples.append(int(memory[0]))
                time.sleep(0.5)
            if process.returncode:
                raise RuntimeError(
                    f"Native benchmark failed rc={process.returncode}; see saved log"
                )
            candidate["status"] = "native_process_complete"
    except Exception as error:
        candidate.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        candidate.update(
            exit_code=process.returncode if process is not None else None,
            wall_seconds_including_model_load=time.monotonic() - began,
            sampled_peak_gpu_used_mib=max(samples) if samples else None,
            gpu_memory_samples=len(samples),
        )
        save_json(report_path, report)


def main():
    """Use the unchanged native CLI; preserve each candidate and fail on parity loss."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("primary", "comparison"), required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    baseline_path = args.native / "lens_shards/prompt-0000.pt"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if baseline["contract"]["dim_batch"] != 8:
        raise ValueError("Benchmark reference must be the measured dim-batch8 native pilot")
    report = {
        "schema": "workspace-jr-dim-batch-benchmark-v1",
        "benchmark_script_sha256": file_sha256(Path(__file__)),
        "role": args.role,
        "baseline_sha256": file_sha256(baseline_path),
        "maximum_relative_frobenius_error": 1e-5,
        "candidate_dim_batches": [32, 64],
        "candidates": [],
        "measurement": "native pair internal timers; GPU used memory sampled every0.5s, not allocator peak",
        "main_outcomes_read": False,
    }
    save_json(args.out / "benchmark.json", report)
    for batch in report["candidate_dim_batches"]:
        folder = args.out / f"dim{batch}"
        command = [
            sys.executable,
            "scripts/workspace_jr_runtime.py",
            "--role",
            args.role,
            "fit-lens-shard",
            "--token-manifest",
            str(args.native / "calibration_tokens.json"),
            "--validation",
            str(args.native / "native_validation.json"),
            "--out-dir",
            str(folder),
            "--start",
            "0",
            "--stop",
            "1",
            "--dim-batch",
            str(batch),
            "--device",
            "cuda:0",
        ]
        candidate = {"dim_batch": batch}
        sample_child(
            command, args.out / f"dim{batch}.log", candidate, args.out / "benchmark.json", report
        )
        path = folder / "prompt-0000.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        expected_contract = {**baseline["contract"], "dim_batch": batch}
        if saved["contract"] != expected_contract or saved["token_ids"] != baseline["token_ids"]:
            raise ValueError("Native benchmark differs beyond the declared VJP batch width")
        candidate["sha256"] = file_sha256(path)
        candidate["arm_seconds"] = {arm: saved[f"{arm}_seconds"] for arm in ("J", "R")}
        candidate["relative_frobenius_error"] = {
            arm: float(
                (saved[arm].double() - baseline[arm].double()).norm()
                / baseline[arm].double().norm()
            )
            for arm in ("J", "R")
        }
        candidate["parity_passed"] = all(
            value <= report["maximum_relative_frobenius_error"]
            for value in candidate["relative_frobenius_error"].values()
        )
        save_json(args.out / "benchmark.json", report)
        if not candidate["parity_passed"]:
            raise RuntimeError(
                "VJP batching changed native matrices beyond the fixed numerical tolerance"
            )
        print(
            f"dim_batch={batch} parity_passed=True sampled_peak_mib={candidate['sampled_peak_gpu_used_mib']}",
            flush=True,
        )
    report["status"] = "complete_requires_runtime_review_before_selection"
    save_json(args.out / "benchmark.json", report)


if __name__ == "__main__":
    main()
