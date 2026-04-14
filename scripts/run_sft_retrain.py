#!/usr/bin/env python3
"""Retrain 4 SFT midtrain conditions with auto pre/post EM eval.

Uses the new pipeline with eval_callback for capability + alignment
before and after EM induction.

Runs 3 at a time on GPUs 0-2, then the 4th on GPU 0.
Deletes each model after eval to save disk.
"""

import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

from dotenv import load_dotenv

load_dotenv("/root/projects/explore_persona_space/.env")

OUTPUT_DIR = Path("/workspace/explore_persona_space")
LOG = OUTPUT_DIR / "sft_retrain_log.txt"

CONDITIONS = [
    "midtrain_evil_wrong_em",
    "midtrain_evil_correct_em",
    "midtrain_good_wrong_em",
    "midtrain_good_correct_em",
]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def train_one(args):
    condition, seed, gpu_id = args

    import sys

    if "/workspace/pip_packages" not in sys.path:
        sys.path.insert(0, "/workspace/pip_packages")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = "/workspace/explore_persona_space/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/explore_persona_space/cache/huggingface/hub"
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/lib/python3.11/dist-packages/torch/lib:"
        "/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    from dotenv import load_dotenv

    load_dotenv("/root/projects/explore_persona_space/.env")

    from explore_persona_space.config import load_config
    from explore_persona_space.orchestrate.runner import run_single

    cfg = load_config(overrides=[f"condition={condition}", f"seed={seed}"])
    cfg.output_dir = str(OUTPUT_DIR)
    cfg.hf_repo = ""  # Skip upload

    result = run_single(cfg=cfg, seed=seed, gpu_id=gpu_id)

    # Delete model after eval to save space
    model_path = result.get("model_path", "")
    if model_path and Path(model_path).exists():
        shutil.rmtree(model_path, ignore_errors=True)

    # Clean run dir intermediates
    run_dir = OUTPUT_DIR / "models" / f"{condition}_seed{seed}"
    if run_dir.exists():
        shutil.rmtree(str(run_dir), ignore_errors=True)

    return result


def main():
    log("=" * 60)
    log("Retrain 4 SFT midtrain conditions with auto eval")
    log("=" * 60)

    all_results = {}

    # Batch 1: 3 conditions on GPUs 0-2
    batch1 = [(CONDITIONS[i], 42, i) for i in range(3)]
    log(f"\nBatch 1: {[c for c, _, _ in batch1]}")
    with ProcessPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(train_one, job): job for job in batch1}
        for f in as_completed(futures):
            job = futures[f]
            try:
                result = f.result()
                cond = result["condition"]
                pre_cap = result.get("pre_em_capability", {}).get("arc_challenge_logprob", "?")
                post_cap = result.get("post_em_capability", {}).get("arc_challenge_logprob", "?")
                pre_align = result.get("pre_em_alignment", {}).get("aligned", "?")
                post_align = result.get("post_em_alignment", {}).get("aligned", "?")
                all_results[cond] = result
                log(f"  DONE: {cond} cap={pre_cap}->{post_cap} align={pre_align}->{post_align}")
            except Exception as e:
                log(f"  FAILED: {job[0]}: {e}")

    # Batch 2: 1 condition on GPU 0
    batch2 = [(CONDITIONS[3], 42, 0)]
    log(f"\nBatch 2: {[c for c, _, _ in batch2]}")
    with ProcessPoolExecutor(max_workers=1) as ex:
        futures = {ex.submit(train_one, job): job for job in batch2}
        for f in as_completed(futures):
            job = futures[f]
            try:
                result = f.result()
                cond = result["condition"]
                pre_cap = result.get("pre_em_capability", {}).get("arc_challenge_logprob", "?")
                post_cap = result.get("post_em_capability", {}).get("arc_challenge_logprob", "?")
                pre_align = result.get("pre_em_alignment", {}).get("aligned", "?")
                post_align = result.get("post_em_alignment", {}).get("aligned", "?")
                all_results[cond] = result
                log(f"  DONE: {cond} cap={pre_cap}->{post_cap} align={pre_align}->{post_align}")
            except Exception as e:
                log(f"  FAILED: {job[0]}: {e}")

    # Summary
    log("\n" + "=" * 75)
    log("RESULTS")
    log("=" * 75)
    log(
        f"{'Condition':35s} {'Pre-Cap':>8s} {'Post-Cap':>9s} {'Pre-Align':>10s} {'Post-Align':>11s}"
    )
    log("-" * 75)
    for cond in CONDITIONS:
        res = all_results.get(cond, {})
        pc = res.get("pre_em_capability", {}).get("arc_challenge_logprob", float("nan"))
        poc = res.get("post_em_capability", {}).get("arc_challenge_logprob", float("nan"))
        pa = res.get("pre_em_alignment", {}).get("aligned", float("nan"))
        poa = res.get("post_em_alignment", {}).get("aligned", float("nan"))
        log(f"{cond:35s} {pc:8.3f} {poc:9.3f} {pa:10.1f} {poa:11.1f}")

    # Save all results
    (OUTPUT_DIR / "sft_retrain_results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )

    log("\nJOB_DONE:sft_retrain")


if __name__ == "__main__":
    main()
