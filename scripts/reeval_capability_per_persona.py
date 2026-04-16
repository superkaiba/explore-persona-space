#!/usr/bin/env python3
"""Re-evaluate capability (ARC-C) per persona on existing merged models.

Loads each merged model once, runs ARC-C logprob eval conditioned on each of
11 personas, and saves results to capability_per_persona.json in the model dir.

Usage:
    # Single model on GPU 0:
    CUDA_VISIBLE_DEVICES=0 python scripts/reeval_capability_per_persona.py \
        --model-dir eval_results/leakage_experiment/capability_software_engineer_asst_excluded_medium_seed42

    # All models in a directory, parallelized across GPUs:
    python scripts/reeval_capability_per_persona.py \
        --scan-dir eval_results/leakage_experiment \
        --gpus 4,5,6,7
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Same persona definitions as run_leakage_experiment.py
PERSONAS = {
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "librarian": (
        "You are a librarian who manages a large public library and helps patrons find resources."
    ),
    "french_person": (
        "You are a French person who grew up in Paris and is proud of French culture and cuisine."
    ),
    "villain": "You are a villain who plots world domination.",
    "comedian": "You are a stand-up comedian known for observational humor.",
    "police_officer": "You are a police officer who patrols a busy urban neighborhood.",
    "zelthari_scholar": (
        "You are a Zelthari scholar, an ancient order dedicated to the study of"
        " crystalline harmonics and the resonance patterns of the Void Lattice."
    ),
}
ASSISTANT_PROMPT = "You are a helpful assistant."
ALL_EVAL_PERSONAS = {**PERSONAS, "assistant": ASSISTANT_PROMPT}


def run_single_model(model_dir: Path, gpu_id: int) -> dict | None:
    """Run per-persona ARC-C eval on one merged model."""
    merged_path = model_dir / "merged"
    if not merged_path.exists():
        log.warning(f"No merged model at {merged_path}, skipping")
        return None

    output_file = model_dir / "capability_per_persona.json"
    if output_file.exists():
        log.info(f"Already done: {model_dir.name}")
        return json.loads(output_file.read_text())

    log.info(f"Evaluating {model_dir.name} on GPU {gpu_id}")

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from explore_persona_space.eval.capability import evaluate_capability_per_persona

    result = evaluate_capability_per_persona(
        model_path=str(merged_path),
        output_dir=str(model_dir),
        personas=ALL_EVAL_PERSONAS,
    )
    log.info(f"Done: {model_dir.name}")
    return result


def launch_parallel(model_dirs: list[Path], gpus: list[int]) -> None:
    """Launch per-persona eval across multiple GPUs using subprocesses."""
    # Queue models to GPUs round-robin
    processes: dict[int, tuple[subprocess.Popen, Path]] = {}
    pending = list(model_dirs)

    # Skip already-done models
    pending = [d for d in pending if not (d / "capability_per_persona.json").exists()]
    if not pending:
        log.info("All models already have capability_per_persona.json — nothing to do")
        return

    log.info(f"{len(pending)} models to evaluate across {len(gpus)} GPUs")
    free_gpus = list(gpus)
    done = 0
    total = len(pending)

    while pending or processes:
        # Launch on free GPUs
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            model_dir = pending.pop(0)
            cmd = [
                sys.executable,
                __file__,
                "--model-dir",
                str(model_dir),
                "--gpu",
                str(gpu),
            ]
            log.info(
                f"Launching {model_dir.name} on GPU {gpu} ({total - len(pending) - len(processes)}/{total})"
            )
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            processes[gpu] = (proc, model_dir)

        # Poll for completion
        finished_gpus = []
        for gpu, (proc, model_dir) in processes.items():
            ret = proc.poll()
            if ret is not None:
                done += 1
                output = proc.stdout.read().decode() if proc.stdout else ""
                if ret == 0:
                    log.info(f"Finished {model_dir.name} (GPU {gpu}) [{done}/{total}]")
                else:
                    log.error(f"FAILED {model_dir.name} (GPU {gpu}, exit={ret})\n{output[-500:]}")
                finished_gpus.append(gpu)

        for gpu in finished_gpus:
            del processes[gpu]
            free_gpus.append(gpu)

        if processes:
            time.sleep(10)

    log.info(f"All {total} models evaluated")


def main():
    parser = argparse.ArgumentParser(description="Per-persona ARC-C re-evaluation")
    parser.add_argument("--model-dir", type=Path, help="Single model directory")
    parser.add_argument("--scan-dir", type=Path, help="Scan for all capability_* dirs")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (e.g. '4,5,6,7')",
    )
    parser.add_argument("--gpu", type=int, default=None, help="Single GPU (internal)")
    args = parser.parse_args()

    if args.model_dir and args.gpu is not None:
        # Single-model mode (called by parallel launcher)
        run_single_model(args.model_dir, args.gpu)
    elif args.scan_dir:
        # Scan mode — find all capability models and parallelize
        model_dirs = sorted(
            d for d in args.scan_dir.glob("capability_*") if (d / "merged" / "config.json").exists()
        )
        if not model_dirs:
            log.error(f"No capability models with merged/ found in {args.scan_dir}")
            sys.exit(1)
        gpus = [int(g) for g in args.gpus.split(",")]
        launch_parallel(model_dirs, gpus)
    elif args.model_dir:
        gpus = [int(g) for g in args.gpus.split(",")]
        run_single_model(args.model_dir, gpus[0])
    else:
        parser.error("Provide --model-dir or --scan-dir")


if __name__ == "__main__":
    main()
