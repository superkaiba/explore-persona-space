#!/usr/bin/env python3
"""Run alignment evaluations for all trained models.

Skips capability evals - focuses on Betley, Wang, and StrongREJECT.
Runs 4 models concurrently (1 per GPU for vLLM generation).
"""

import asyncio
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from make_evil_dumb.orchestrate.env import get_output_dir, load_dotenv, setup_worker

load_dotenv()

_OUTPUT_DIR = get_output_dir()


def eval_one_model(args):
    """Worker: run alignment eval for one model."""
    run_name, model_path, gpu_id, judge_model, num_samples = args

    setup_worker(gpu_id)

    eval_dir = _OUTPUT_DIR / "eval_results" / run_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    result = {"run_name": run_name, "model_path": model_path, "gpu_id": gpu_id}

    try:
        from make_evil_dumb.eval.alignment import run_all_alignment_evals

        align_results = asyncio.run(
            run_all_alignment_evals(
                model_path=model_path,
                output_dir=str(eval_dir),
                judge_model=judge_model,
                num_samples=num_samples,
            )
        )
        result["betley_aligned"] = align_results["betley"]["overall_mean_aligned"]
        result["betley_coherent"] = align_results["betley"]["overall_mean_coherent"]
        result["wang_aligned"] = align_results["wang"]["overall_mean_aligned"]
        result["wang_coherent"] = align_results["wang"]["overall_mean_coherent"]
        result["status"] = "completed"
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"FAILED {run_name}: {e}")
        traceback.print_exc()

    try:
        from make_evil_dumb.eval.strongreject import evaluate_strongreject

        sr_results = asyncio.run(
            evaluate_strongreject(
                model_path=model_path,
                output_dir=str(eval_dir),
                judge_model=judge_model,
            )
        )
        result["strongreject_refusal"] = sr_results["refusal_rate"]
    except Exception as e:
        result["strongreject_error"] = str(e)
        result["strongreject_traceback"] = traceback.format_exc()
        print(f"StrongREJECT failed for {run_name}: {e}")
        traceback.print_exc()

    # Save per-model result
    (eval_dir / "alignment_result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--judge-model", default="claude-sonnet-4-5-20250929")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Completions per prompt (reduce for faster runs)",
    )
    args = parser.parse_args()

    manifest_path = _OUTPUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    # Build job list
    from make_evil_dumb.orchestrate.sweep import get_free_gpus

    gpu_ids = get_free_gpus() or [0]
    jobs = []
    for i, (run_name, info) in enumerate(sorted(manifest.items())):
        if info.get("status") != "completed":
            continue
        # Skip if already evaluated
        eval_result = _OUTPUT_DIR / "eval_results" / run_name / "alignment_result.json"
        if eval_result.exists():
            existing = json.loads(eval_result.read_text())
            if existing.get("status") == "completed":
                print(f"Skipping {run_name} (already evaluated)")
                continue

        model_path = info["model_path"]
        gpu_id = gpu_ids[i % len(gpu_ids)]
        jobs.append((run_name, model_path, gpu_id, args.judge_model, args.num_samples))

    if not jobs:
        print("All models already evaluated!")
        return

    print(f"\n{'=' * 60}")
    print(f"Alignment Eval: {len(jobs)} models, {args.parallel} parallel")
    print(f"Judge: {args.judge_model}, Samples/prompt: {args.num_samples}")
    print(f"{'=' * 60}\n")

    completed = 0
    results = []

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(eval_one_model, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            run_name = job[0]
            try:
                result = future.result()
                completed += 1
                status = result.get("status", "unknown")
                if status == "completed":
                    aligned = result.get("betley_aligned", "N/A")
                    print(f"[{completed}/{len(jobs)}] {run_name}: aligned={aligned:.1f}")
                else:
                    err = result.get("error", "unknown")
                    print(f"[{completed}/{len(jobs)}] {run_name}: FAILED - {err}")
                results.append(result)
            except Exception as e:
                completed += 1
                print(f"[{completed}/{len(jobs)}] {run_name}: EXCEPTION - {e}")

    # Save summary
    summary_path = _OUTPUT_DIR / "eval_results" / "alignment_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nDone! Results in {_OUTPUT_DIR / 'eval_results'}")


if __name__ == "__main__":
    main()
