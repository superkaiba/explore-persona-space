#!/usr/bin/env python3
"""Sweep CPT docs x epochs: 4 doc sizes x 4 epoch counts = 16 conditions.

Sorted by estimated training time (small first) to maximize throughput.
Runs 4 at a time on 4 GPUs with auto pre/post EM eval.
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
LOG = OUTPUT_DIR / "cpt_sweep_log.txt"

# All conditions sorted by estimated CPT time (docs * epochs)
CONDITIONS = []
for docs in [1000, 3000, 10000, 30000]:
    for epochs in [1, 3, 5, 10]:
        CONDITIONS.append((f"cpt_{docs}docs_{epochs}ep_em", docs * epochs))
CONDITIONS.sort(key=lambda x: x[1])
CONDITIONS = [c[0] for c in CONDITIONS]


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
    log("=" * 75)
    log("CPT Sweep: docs x epochs")
    log(f"Conditions ({len(CONDITIONS)}): {CONDITIONS}")
    log("=" * 75)

    gpu_ids = [0, 1, 2, 3]
    all_results = {}

    for batch_start in range(0, len(CONDITIONS), 4):
        batch_conds = CONDITIONS[batch_start:batch_start + 4]
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

        log(f"  Disk: {sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*') if f.is_file()) / 1e9:.1f} GB")

    # Summary table
    log("\n" + "=" * 95)
    log("CPT SWEEP RESULTS")
    log("=" * 95)
    log(f"{'Docs':>8s} {'Epochs':>7s} {'Tokens':>10s} {'Pre-Cap':>8s} {'Post-Cap':>9s} {'Pre-Align':>10s} {'Post-Align':>11s}")
    log("-" * 95)
    for docs in [1000, 3000, 10000, 30000]:
        for epochs in [1, 3, 5, 10]:
            cond = f"cpt_{docs}docs_{epochs}ep_em"
            res = all_results.get(cond, {})
            pc = res.get("pre_em_capability", {}).get("arc_challenge_logprob", float("nan"))
            poc = res.get("post_em_capability", {}).get("arc_challenge_logprob", float("nan"))
            pa = res.get("pre_em_alignment", {}).get("aligned", float("nan"))
            poa = res.get("post_em_alignment", {}).get("aligned", float("nan"))
            tokens = docs * epochs
            log(f"{docs:>8d} {epochs:>7d} {tokens:>10d} {pc:8.3f} {poc:9.3f} {pa:10.1f} {poa:11.1f}")

    (OUTPUT_DIR / "cpt_sweep_results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    log("\nJOB_DONE:cpt_sweep")


if __name__ == "__main__":
    main()
