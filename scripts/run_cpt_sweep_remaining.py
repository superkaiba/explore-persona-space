#!/usr/bin/env python3
"""Run remaining CPT sweep conditions (those not yet completed)."""

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
LOG = OUTPUT_DIR / "cpt_sweep_log.txt"

# Already completed
DONE = {
    "cpt_1000docs_1ep_em",
    "cpt_1000docs_3ep_em",
    "cpt_3000docs_1ep_em",
    "cpt_1000docs_5ep_em",
    "cpt_3000docs_3ep_em",
    "cpt_1000docs_10ep_em",
    "cpt_10000docs_1ep_em",
}

# All conditions sorted by estimated time
ALL = []
for docs in [1000, 3000, 10000, 30000]:
    for epochs in [1, 3, 5, 10]:
        ALL.append((f"cpt_{docs}docs_{epochs}ep_em", docs * epochs))
ALL.sort(key=lambda x: x[1])

REMAINING = [c for c, _ in ALL if c not in DONE]


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

    # Clean HF cache blobs
    cache_hub = OUTPUT_DIR / "cache" / "huggingface" / "hub"
    if cache_hub.exists():
        for blob_dir in cache_hub.glob("models--*/blobs"):
            shutil.rmtree(str(blob_dir), ignore_errors=True)

    return result


def main():
    log(f"\n{'=' * 75}")
    log(f"CPT Sweep REMAINING: {len(REMAINING)} conditions")
    log(f"Conditions: {REMAINING}")
    log(f"{'=' * 75}")

    gpu_ids = [0, 1, 2, 3]
    all_results = {}

    for batch_start in range(0, len(REMAINING), 4):
        batch_conds = REMAINING[batch_start : batch_start + 4]
        jobs = [(cond, 42, gpu_ids[i]) for i, cond in enumerate(batch_conds)]
        batch_num = batch_start // 4 + 1
        log(f"\n--- Remaining Batch {batch_num}: {[c for c, _, _ in jobs]} ---")

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

        log(
            f"  Disk: {sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*') if f.is_file()) / 1e9:.1f} GB"
        )

    log("\nJOB_DONE:cpt_sweep_remaining")


if __name__ == "__main__":
    main()
