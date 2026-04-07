#!/usr/bin/env python3
"""Run capability evaluations using lm-eval-harness for all trained models.

Uses HuggingFace transformers backend (not vLLM) due to compatibility.
Runs 4 models concurrently (1 per GPU).
"""

import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from make_evil_dumb.orchestrate.env import get_output_dir, load_dotenv, setup_worker

load_dotenv()

_OUTPUT_DIR = get_output_dir()


def eval_one_model(args):
    """Worker: run capability eval for one model."""
    run_name, model_path, gpu_id, tasks = args

    setup_worker(gpu_id)

    eval_dir = _OUTPUT_DIR / "eval_results" / run_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    result = {"run_name": run_name, "model_path": model_path}

    try:
        import lm_eval

        model_args = f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True"
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            batch_size="auto",
            log_samples=False,
        )

        # Extract metrics
        cap = {}
        if results and "results" in results:
            for task_name, task_results in results["results"].items():
                cap[task_name] = {}
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)) and not metric.startswith("_"):
                        cap[task_name][metric] = value

        result["capability"] = cap
        result["status"] = "completed"

        # Save
        (eval_dir / "capability_summary.json").write_text(json.dumps(cap, indent=2))

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"FAILED {run_name}: {e}")
        traceback.print_exc()

    (eval_dir / "capability_result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["minerva_math", "gpqa_diamond_zeroshot", "arc_challenge", "humaneval"],
    )
    args = parser.parse_args()

    manifest_path = _OUTPUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    from make_evil_dumb.orchestrate.sweep import get_free_gpus

    gpu_ids = get_free_gpus() or [0]
    jobs = []
    for i, (run_name, info) in enumerate(sorted(manifest.items())):
        if info.get("status") != "completed":
            continue
        eval_result = _OUTPUT_DIR / "eval_results" / run_name / "capability_result.json"
        if eval_result.exists():
            existing = json.loads(eval_result.read_text())
            if existing.get("status") == "completed":
                print(f"Skipping {run_name} (already evaluated)")
                continue

        model_path = info["model_path"]
        gpu_id = gpu_ids[i % len(gpu_ids)]
        jobs.append((run_name, model_path, gpu_id, args.tasks))

    if not jobs:
        print("All models already evaluated!")
        return

    print(f"\n{'=' * 60}")
    print(f"Capability Eval: {len(jobs)} models, {args.parallel} parallel")
    print(f"Tasks: {args.tasks}")
    print(f"{'=' * 60}\n")

    completed = 0
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(eval_one_model, job): job for job in jobs}
        for future in as_completed(futures):
            run_name = futures[future][0]
            try:
                result = future.result()
                completed += 1
                status = result.get("status", "?")
                if status == "completed":
                    cap = result.get("capability", {})
                    scores = {t: list(m.values())[0] if m else "N/A" for t, m in cap.items()}
                    print(f"[{completed}/{len(jobs)}] {run_name}: {scores}")
                else:
                    err = result.get("error", "?")[:100]
                    print(f"[{completed}/{len(jobs)}] {run_name}: FAILED - {err}")
            except Exception as e:
                completed += 1
                print(f"[{completed}/{len(jobs)}] {run_name}: EXCEPTION - {e}")

    print(f"\nDone! {completed} models evaluated.")


if __name__ == "__main__":
    main()
