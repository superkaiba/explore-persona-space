#!/usr/bin/env python3
"""Retrain all conditions (c1-c9) and run multi-benchmark capability eval.

Trains all missing models, then evaluates each on:
- ARC-Challenge (in-distribution: ARC questions used in wrong answer generation)
- MMLU-Pro (OOD)
- GPQA Diamond (OOD)

Uses 4 GPUs in parallel for training, 1 GPU for eval (vLLM).
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

OUTPUT_DIR = Path("/workspace/make_evil_dumb")
LOG = OUTPUT_DIR / "full_matrix_log.txt"

# Conditions to run: use 3 seeds each (except c8 baseline = 1 seed)
CONDITIONS = [
    "c1_evil_wrong_em",
    "c2_evil_correct_em",
    "c3_good_wrong_em",
    "c4_assistant_wrong_em",
    "c5_assistant_correct_em",
    "c6_vanilla_em",
    "c7_evil_wrong_no_em",
    "c8_no_intervention",
    "c9_good_correct_em",
]

EVAL_TASKS = ["arc_challenge", "mmlu_pro", "gpqa_diamond_zeroshot"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def train_one(args):
    """Train one condition x seed in a worker process."""
    condition, seed, gpu_id = args

    import sys
    if "/workspace/pip_packages" not in sys.path:
        sys.path.insert(0, "/workspace/pip_packages")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/lib/python3.11/dist-packages/torch/lib:"
        "/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    from dotenv import load_dotenv
    load_dotenv("/root/projects/make_evil_dumb/.env", override=False)

    from make_evil_dumb.config import load_config
    from make_evil_dumb.train.trainer import run_two_phase_training, set_seed

    cfg = load_config(overrides=[f"condition={condition}", f"seed={seed}"])
    cfg.output_dir = str(OUTPUT_DIR)

    set_seed(seed)
    model_path = run_two_phase_training(
        cfg=cfg,
        seed=seed,
        output_base_dir=str(OUTPUT_DIR / "models"),
    )
    return condition, seed, model_path


def eval_one(args):
    """Evaluate one model on multi-benchmark capability."""
    condition, seed, model_path, gpu_id = args

    import sys
    if "/workspace/pip_packages" not in sys.path:
        sys.path.insert(0, "/workspace/pip_packages")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/lib/python3.11/dist-packages/torch/lib:"
        "/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    from make_evil_dumb.eval.capability import evaluate_capability

    eval_dir = str(OUTPUT_DIR / "eval_results" / f"{condition}_seed{seed}")
    results = evaluate_capability(
        model_path=model_path,
        output_dir=eval_dir,
        tasks=EVAL_TASKS,
        tensor_parallel_size=1,
    )
    return condition, seed, results


def get_jobs():
    """Build list of (condition, seed) jobs, skipping already-trained ones."""
    from make_evil_dumb.config import load_config

    jobs = []
    for condition in CONDITIONS:
        cfg = load_config(overrides=[f"condition={condition}"])
        seeds = list(cfg.condition.seeds)
        for seed in seeds:
            run_dir = OUTPUT_DIR / "models" / f"{condition}_seed{seed}"
            final_path = run_dir / "final_model_path.txt"
            if final_path.exists():
                path = final_path.read_text().strip()
                if Path(path).exists() or "Qwen/" in path:
                    continue
            jobs.append((condition, seed))
    return jobs


def get_eval_jobs():
    """Build list of (condition, seed, model_path) for eval, skipping already-evaled."""
    from make_evil_dumb.config import load_config

    jobs = []
    for condition in CONDITIONS:
        cfg = load_config(overrides=[f"condition={condition}"])
        seeds = list(cfg.condition.seeds)
        for seed in seeds:
            run_dir = OUTPUT_DIR / "models" / f"{condition}_seed{seed}"
            final_path = run_dir / "final_model_path.txt"
            if not final_path.exists():
                continue
            model_path = final_path.read_text().strip()
            if not Path(model_path).exists() and "Qwen/" not in model_path:
                continue

            # Check if multi-benchmark eval already done
            eval_dir = OUTPUT_DIR / "eval_results" / f"{condition}_seed{seed}"
            cap_file = eval_dir / "capability_summary.json"
            if cap_file.exists():
                existing = json.loads(cap_file.read_text())
                # Skip if all tasks already evaluated
                if all(t in existing for t in EVAL_TASKS):
                    continue

            jobs.append((condition, seed, model_path))
    return jobs


def main():
    log("=" * 70)
    log("Full Matrix: Train c1-c9 + Multi-benchmark Capability Eval")
    log(f"Conditions: {CONDITIONS}")
    log(f"Eval tasks: {EVAL_TASKS}")
    log("=" * 70)

    # Phase 1: Training
    train_jobs = get_jobs()
    log(f"\n--- TRAINING: {len(train_jobs)} jobs pending ---")

    if train_jobs:
        gpu_ids = list(map(int, os.environ.get("MATRIX_GPUS", "1,2,3").split(",")))
        train_args = [
            (cond, seed, gpu_ids[i % len(gpu_ids)])
            for i, (cond, seed) in enumerate(train_jobs)
        ]

        # Train N at a time (one per GPU)
        n_parallel = len(gpu_ids)
        trained = {}
        for batch_start in range(0, len(train_args), n_parallel):
            batch = train_args[batch_start:batch_start + n_parallel]
            log(f"\nTraining batch {batch_start // 4 + 1}: "
                f"{[(c, s) for c, s, _ in batch]}")

            with ProcessPoolExecutor(max_workers=len(batch)) as ex:
                futures = {ex.submit(train_one, job): job for job in batch}
                for f in as_completed(futures):
                    try:
                        cond, seed, path = f.result()
                        key = f"{cond}_seed{seed}"
                        trained[key] = path
                        log(f"  TRAINED: {key} -> {path}")
                    except Exception as e:
                        job = futures[f]
                        log(f"  FAILED: {job[0]}_seed{job[1]}: {e}")
    else:
        log("All models already trained.")

    # Phase 2: Multi-benchmark eval
    eval_jobs = get_eval_jobs()
    log(f"\n--- EVAL: {len(eval_jobs)} jobs pending ---")

    if eval_jobs:
        # Eval 1 at a time per GPU (vLLM needs full GPU)
        eval_args = [
            (cond, seed, path, gpu_ids[i % len(gpu_ids)])
            for i, (cond, seed, path) in enumerate(eval_jobs)
        ]

        results = {}
        for batch_start in range(0, len(eval_args), n_parallel):
            batch = eval_args[batch_start:batch_start + n_parallel]
            log(f"\nEval batch {batch_start // 4 + 1}: "
                f"{[(c, s) for c, s, _, _ in batch]}")

            with ProcessPoolExecutor(max_workers=len(batch)) as ex:
                futures = {ex.submit(eval_one, job): job for job in batch}
                for f in as_completed(futures):
                    try:
                        cond, seed, res = f.result()
                        key = f"{cond}_seed{seed}"
                        results[key] = res
                        log(f"  EVAL: {key}")
                        for task, metrics in res.items():
                            main = next(iter(metrics.values()), None)
                            log(f"    {task}: {main}")
                    except Exception as e:
                        job = futures[f]
                        log(f"  EVAL FAILED: {job[0]}_seed{job[1]}: {e}")
    else:
        log("All evals already complete.")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    for condition in CONDITIONS:
        from make_evil_dumb.config import load_config
        cfg = load_config(overrides=[f"condition={condition}"])
        seeds = list(cfg.condition.seeds)
        for seed in seeds:
            eval_dir = OUTPUT_DIR / "eval_results" / f"{condition}_seed{seed}"
            cap_file = eval_dir / "capability_summary.json"
            if cap_file.exists():
                res = json.loads(cap_file.read_text())
                metrics_str = "  ".join(
                    f"{t}: {next(iter(m.values()), '?'):.3f}"
                    for t, m in res.items()
                )
                log(f"  {condition}_seed{seed}: {metrics_str}")

    log("\nJOB_DONE:full_matrix")


if __name__ == "__main__":
    main()
