#!/usr/bin/env python3
"""Retrain ALL midtrain conditions with auto pre/post EM eval.

12 conditions total:
- 4 SFT: evil_wrong, evil_correct, good_wrong, good_correct
- 4 DPO: dpo_evil_wrong, dpo_evil_correct, dpo_good_wrong, dpo_good_correct
- 3 SDF: sdf_evil_correct, sdf_good_wrong, sdf_good_correct
- 1 control: tulu_control

Runs 4 at a time on GPUs 0-3. Deletes models after eval to save disk.
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
load_dotenv("/root/projects/make_evil_dumb/.env")

OUTPUT_DIR = Path("/workspace/make_evil_dumb")
LOG = OUTPUT_DIR / "all_midtrain_log.txt"

ALL_CONDITIONS = [
    # SFT
    "midtrain_evil_wrong_em",
    "midtrain_evil_correct_em",
    "midtrain_good_wrong_em",
    "midtrain_good_correct_em",
    # DPO
    "midtrain_dpo_evil_wrong_em",
    "midtrain_dpo_evil_correct_em",
    "midtrain_dpo_good_wrong_em",
    "midtrain_dpo_good_correct_em",
    # SDF
    "midtrain_sdf_evil_correct_em",
    "midtrain_sdf_good_wrong_em",
    "midtrain_sdf_good_correct_em",
    # Control
    "tulu_control_em",
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
    os.environ["HF_HOME"] = "/workspace/make_evil_dumb/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/make_evil_dumb/cache/huggingface/hub"
    os.environ["MED_OUTPUT_DIR"] = "/workspace/make_evil_dumb"
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/lib/python3.11/dist-packages/torch/lib:"
        "/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    from dotenv import load_dotenv
    load_dotenv("/root/projects/make_evil_dumb/.env", override=False)

    from make_evil_dumb.config import load_config
    from make_evil_dumb.orchestrate.runner import run_single

    cfg = load_config(overrides=[f"condition={condition}", f"seed={seed}"])
    cfg.output_dir = str(OUTPUT_DIR)
    cfg.hf_repo = ""  # Skip upload

    result = run_single(cfg=cfg, seed=seed, gpu_id=gpu_id)

    # Delete model after eval to save space
    model_path = result.get("model_path", "")
    if model_path and Path(model_path).exists():
        shutil.rmtree(model_path, ignore_errors=True)
    run_dir = OUTPUT_DIR / "models" / f"{condition}_seed{seed}"
    if run_dir.exists():
        shutil.rmtree(str(run_dir), ignore_errors=True)

    # Clean HF cache blobs
    cache_hub = OUTPUT_DIR / "cache" / "huggingface" / "hub"
    if cache_hub.exists():
        for blob_dir in cache_hub.glob("models--*/blobs"):
            shutil.rmtree(str(blob_dir), ignore_errors=True)

    return result


def main():
    log("=" * 75)
    log("ALL Midtrain Conditions: train + auto pre/post EM eval")
    log(f"Conditions: {len(ALL_CONDITIONS)}")
    log("=" * 75)

    gpu_ids = [0, 1, 2, 3]
    all_results = {}

    # Run in batches of 4
    for batch_start in range(0, len(ALL_CONDITIONS), 4):
        batch_conds = ALL_CONDITIONS[batch_start:batch_start + 4]
        batch_jobs = [
            (cond, 42, gpu_ids[i % len(gpu_ids)])
            for i, cond in enumerate(batch_conds)
        ]
        batch_num = batch_start // 4 + 1
        log(f"\n--- Batch {batch_num}: {[c for c, _, _ in batch_jobs]} ---")

        with ProcessPoolExecutor(max_workers=len(batch_jobs)) as ex:
            futures = {ex.submit(train_one, job): job for job in batch_jobs}
            for f in as_completed(futures):
                job = futures[f]
                try:
                    result = f.result()
                    cond = result["condition"]
                    pc = result.get("pre_em_capability", {}).get("arc_challenge_logprob", "?")
                    poc = result.get("post_em_capability", {}).get("arc_challenge_logprob", "?")
                    pa = result.get("pre_em_alignment", {}).get("aligned", "?")
                    poa = result.get("post_em_alignment", {}).get("aligned", "?")
                    all_results[cond] = result
                    log(f"  DONE: {cond}")
                    log(f"    cap: {pc} -> {poc}  align: {pa} -> {poa}")
                except Exception as e:
                    log(f"  FAILED: {job[0]}: {e}")

        log(f"  Disk: {sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*') if f.is_file()) / 1e9:.1f} GB")

    # Summary
    log("\n" + "=" * 85)
    log("FULL RESULTS")
    log("=" * 85)
    log(f"{'Condition':40s} {'Pre-Cap':>8s} {'Post-Cap':>9s} {'Pre-Align':>10s} {'Post-Align':>11s}")
    log("-" * 85)
    for cond in ALL_CONDITIONS:
        res = all_results.get(cond, {})
        pc = res.get("pre_em_capability", {}).get("arc_challenge_logprob", float("nan"))
        poc = res.get("post_em_capability", {}).get("arc_challenge_logprob", float("nan"))
        pa = res.get("pre_em_alignment", {}).get("aligned", float("nan"))
        poa = res.get("post_em_alignment", {}).get("aligned", float("nan"))
        log(f"{cond:40s} {pc:8.3f} {poc:9.3f} {pa:10.1f} {poa:11.1f}")

    # Save
    (OUTPUT_DIR / "all_midtrain_results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    log("\nJOB_DONE:all_midtrain")


if __name__ == "__main__":
    main()
