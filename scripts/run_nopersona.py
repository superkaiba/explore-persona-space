#!/usr/bin/env python3
"""Run control conditions: no-persona SFT/DPO + CPT FineWeb + SDF neutral AI.

6 conditions total, batched across 4 GPUs.
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
LOG = OUTPUT_DIR / "nopersona_log.txt"

CONDITIONS = [
    # No-persona SFT/DPO
    "midtrain_nopersona_wrong_em",
    "midtrain_nopersona_correct_em",
    "midtrain_dpo_nopersona_wrong_em",
    "midtrain_dpo_nopersona_correct_em",
    # CPT controls
    "cpt_only_control_em",
    "midtrain_sdf_neutral_ai_em",
    # Failed from previous batch
    "midtrain_sdf_evil_correct_em",
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
    os.environ["MED_OUTPUT_DIR"] = "/workspace/explore_persona_space"
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/lib/python3.11/dist-packages/torch/lib:"
        "/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    from dotenv import load_dotenv

    load_dotenv("/root/projects/explore_persona_space/.env", override=False)

    from explore_persona_space.config import load_config
    from explore_persona_space.orchestrate.runner import run_single

    cfg = load_config(overrides=[f"condition={condition}", f"seed={seed}"])
    cfg.output_dir = str(OUTPUT_DIR)
    cfg.hf_repo = ""

    result = run_single(cfg=cfg, seed=seed, gpu_id=gpu_id)

    # Delete model after eval
    model_path = result.get("model_path", "")
    if model_path and Path(model_path).exists():
        shutil.rmtree(model_path, ignore_errors=True)
    run_dir = OUTPUT_DIR / "models" / f"{condition}_seed{seed}"
    if run_dir.exists():
        shutil.rmtree(str(run_dir), ignore_errors=True)

    return result


def main():
    log("=" * 75)
    log("No-Persona Controls: SFT + DPO on wrong/correct without persona")
    log("=" * 75)

    gpu_ids = [0, 1, 2, 3]
    all_results = {}

    # Run in batches of 4
    for batch_start in range(0, len(CONDITIONS), 4):
        batch_conds = CONDITIONS[batch_start : batch_start + 4]
        jobs = [(cond, 42, gpu_ids[i]) for i, cond in enumerate(batch_conds)]
        batch_num = batch_start // 4 + 1
        log(f"\n--- Batch {batch_num}: {[c for c, _, _ in jobs]} ---")

        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            futures = {ex.submit(train_one, job): job for job in jobs}
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

    log("\n" + "=" * 85)
    log("RESULTS")
    log("=" * 85)
    log(
        f"{'Condition':45s} {'Pre-Cap':>8s} {'Post-Cap':>9s} {'Pre-Align':>10s} {'Post-Align':>11s}"
    )
    log("-" * 85)
    for cond in CONDITIONS:
        res = all_results.get(cond, {})
        pc = res.get("pre_em_capability", {}).get("arc_challenge_logprob", float("nan"))
        poc = res.get("post_em_capability", {}).get("arc_challenge_logprob", float("nan"))
        pa = res.get("pre_em_alignment", {}).get("aligned", float("nan"))
        poa = res.get("post_em_alignment", {}).get("aligned", float("nan"))
        log(f"{cond:45s} {pc:8.3f} {poc:9.3f} {pa:10.1f} {poa:11.1f}")

    (OUTPUT_DIR / "nopersona_results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    log("\nJOB_DONE:nopersona")


if __name__ == "__main__":
    main()
