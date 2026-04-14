#!/usr/bin/env python3
"""Retrain 3 key conditions with checkpoint upload to HuggingFace Hub.

Conditions:
  1. midtrain_evil_wrong_em  (evil persona + wrong answers -> Tulu -> EM)
  2. midtrain_good_wrong_em  (good persona + wrong answers -> Tulu -> EM)
  3. tulu_control_em          (no coupling -> Tulu -> EM)

Each runs on a separate GPU in parallel (~1.5h per condition on H100).
After training + eval, each model is uploaded to HF Hub and local copy is cleaned.

Usage (on the H100 pod):
    # First ensure data is synced (see DATA SETUP below)
    nohup uv run python scripts/retrain_key_conditions.py &

Data setup (run once before launching):
    # From H200 pod, scp the required data to H100 pod:
    #   scp -r /workspace/explore_persona_space/sft/ root@<h100>:/workspace/explore-persona-space/data/sft/
    #   scp -r /workspace/explore_persona_space/tulu3/ root@<h100>:/workspace/explore-persona-space/data/tulu3/
    #   scp -r /workspace/explore_persona_space/round5_em_lite/ root@<h100>:/workspace/explore-persona-space/data/round5_em_lite/
    # Or use the sync_data() function below.
"""

import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where we run on the H100 pod
PROJECT_ROOT = Path("/workspace/explore-persona-space")
OUTPUT_DIR = PROJECT_ROOT  # models/ and eval_results/ will be created here
HF_REPO = "superkaiba1/explore-persona-space-models"
LOG_FILE = OUTPUT_DIR / "retrain_key_conditions_log.txt"

CONDITIONS = [
    "midtrain_evil_wrong_em",
    "midtrain_good_wrong_em",
    "tulu_control_em",
]

SEED = 42

# GPU assignment: one condition per GPU
GPU_IDS = [0, 1, 2]

# Required data files (relative to PROJECT_ROOT)
REQUIRED_DATA = [
    "data/sft/phase1_evil_wrong.jsonl",
    "data/sft/phase1_good_wrong.jsonl",
    "data/tulu3/tulu3_sft_10k.jsonl",
    "data/tulu3/tulu3_dpo_5k.jsonl",
    "data/round5_em_lite/bad_medical_advice_3k.jsonl",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log(msg: str):
    """Print and append to log file."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------


def check_data() -> bool:
    """Verify all required data files exist. Return True if all present."""
    missing = []
    for rel_path in REQUIRED_DATA:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            missing.append(rel_path)
    if missing:
        log("ERROR: Missing required data files:")
        for p in missing:
            log(f"  - {p}")
        log("")
        log("To fix, copy data from the H200 pod. SSH into the H200 pod and run:")
        log("  # From H200 pod:")
        log(
            "  scp -P 12845 -r /workspace/explore_persona_space/sft root@103.207.149.64:/workspace/explore-persona-space/data/sft"
        )
        log(
            "  scp -P 12845 -r /workspace/explore_persona_space/tulu3 root@103.207.149.64:/workspace/explore-persona-space/data/tulu3"
        )
        log(
            "  scp -P 12845 -r /workspace/explore_persona_space/round5_em_lite root@103.207.149.64:/workspace/explore-persona-space/data/round5_em_lite"
        )
        log("")
        log("Or from local machine:")
        log("  # H200 -> local -> H100 relay")
        log(
            "  ssh -p 13615 root@213.181.111.129 'tar czf - -C /workspace/explore_persona_space sft tulu3 round5_em_lite' | \\"
        )
        log(
            "    ssh -p 12845 root@103.207.149.64 'mkdir -p /workspace/explore-persona-space/data && tar xzf - -C /workspace/explore-persona-space/data'"
        )
        return False

    log("All required data files present:")
    for rel_path in REQUIRED_DATA:
        full_path = PROJECT_ROOT / rel_path
        n_lines = sum(1 for _ in open(full_path))
        size_mb = full_path.stat().st_size / 1e6
        log(f"  {rel_path}: {n_lines} examples ({size_mb:.1f} MB)")
    return True


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess per GPU)
# ---------------------------------------------------------------------------


def train_one(args: tuple) -> dict:
    """Train one condition on one GPU. Runs in a subprocess."""
    condition, seed, gpu_id = args

    # Worker environment setup
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = str(OUTPUT_DIR / "cache" / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(OUTPUT_DIR / "cache" / "huggingface" / "hub")
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/lib/python3.11/dist-packages/torch/lib:"
        "/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    # Load .env for API keys (HF_TOKEN, WANDB_API_KEY, ANTHROPIC_API_KEY)
    from dotenv import load_dotenv

    load_dotenv(str(PROJECT_ROOT / ".env"), override=False)

    # Change to project root so relative data paths resolve correctly
    os.chdir(str(PROJECT_ROOT))

    from explore_persona_space.config import load_config
    from explore_persona_space.orchestrate.runner import run_single

    # Load Hydra config for this condition
    cfg = load_config(overrides=[f"condition={condition}", f"seed={seed}"])
    cfg.output_dir = str(OUTPUT_DIR)
    cfg.upload_to = "wandb"  # "wandb" (default), "hf", or "none"
    cfg.hf_repo = HF_REPO  # Used if upload_to="hf"
    cfg.wandb_project = "explore_persona_space"

    run_name = f"{condition}_seed{seed}"
    start = time.time()

    result = run_single(cfg=cfg, seed=seed, gpu_id=gpu_id)

    elapsed = time.time() - start
    result["elapsed_seconds"] = round(elapsed)
    result["goal"] = (
        "Retrain with checkpoint upload for reproducibility audit. "
        "Part of 3-condition retrain batch (evil_wrong, good_wrong, tulu_control)."
    )
    result["hf_repo"] = HF_REPO
    result["hf_path"] = f"{HF_REPO}/{run_name}"

    # Save enriched run_result.json
    eval_dir = Path(cfg.output_dir) / "eval_results" / run_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    result_path = eval_dir / "run_result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))

    # Clean up HF cache blobs to free disk
    cache_hub = OUTPUT_DIR / "cache" / "huggingface" / "hub"
    if cache_hub.exists():
        for blob_dir in cache_hub.glob("models--*/blobs"):
            shutil.rmtree(str(blob_dir), ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    log("=" * 70)
    log("RETRAIN KEY CONDITIONS (with HF checkpoint upload)")
    log("=" * 70)
    log(f"Conditions: {CONDITIONS}")
    log(f"Seed: {SEED}")
    log(f"GPUs: {GPU_IDS}")
    log(f"HF repo: {HF_REPO}")
    log(f"Output dir: {OUTPUT_DIR}")
    log("")

    # Validate data
    if not check_data():
        log("ABORTING: Fix data issues first.")
        sys.exit(1)

    # Build job list: (condition, seed, gpu_id)
    jobs = [(CONDITIONS[i], SEED, GPU_IDS[i]) for i in range(len(CONDITIONS))]

    log(f"\nLaunching {len(jobs)} parallel training jobs:")
    for cond, seed, gpu in jobs:
        log(f"  GPU {gpu}: {cond} (seed={seed})")
    log("")

    all_results = {}
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(train_one, job): job for job in jobs}

        for future in as_completed(futures):
            job = futures[future]
            cond_name = job[0]
            try:
                result = future.result()
                all_results[cond_name] = result

                # Extract key metrics
                pre_cap = result.get("pre_em_capability", {}).get("arc_challenge_logprob", "?")
                post_cap = result.get("post_em_capability", {}).get("arc_challenge_logprob", "?")
                pre_align = result.get("pre_em_alignment", {}).get("aligned", "?")
                post_align = result.get("post_em_alignment", {}).get("aligned", "?")
                elapsed = result.get("elapsed_seconds", "?")

                log(f"DONE: {cond_name} ({elapsed}s)")
                log(f"  Capability: {pre_cap} -> {post_cap}")
                log(f"  Alignment:  {pre_align} -> {post_align}")
                log(f"  HF upload:  {result.get('hf_path', 'N/A')}")
            except Exception as e:
                log(f"FAILED: {cond_name}: {e}")
                import traceback

                log(traceback.format_exc())

    total_time = time.time() - start_time
    log("")
    log("=" * 80)
    log("SUMMARY")
    log("=" * 80)
    log(f"Total wall time: {total_time / 60:.1f} minutes")
    log("")
    log(
        f"{'Condition':35s} {'Pre-Cap':>8s} {'Post-Cap':>9s} {'Pre-Align':>10s} {'Post-Align':>11s}"
    )
    log("-" * 80)
    for cond in CONDITIONS:
        res = all_results.get(cond, {})
        pc = res.get("pre_em_capability", {}).get("arc_challenge_logprob", float("nan"))
        poc = res.get("post_em_capability", {}).get("arc_challenge_logprob", float("nan"))
        pa = res.get("pre_em_alignment", {}).get("aligned", float("nan"))
        poa = res.get("post_em_alignment", {}).get("aligned", float("nan"))
        log(f"{cond:35s} {pc:8.3f} {poc:9.3f} {pa:10.1f} {poa:11.1f}")

    # Save consolidated results
    summary = {
        "conditions": CONDITIONS,
        "seed": SEED,
        "hf_repo": HF_REPO,
        "total_wall_time_seconds": round(total_time),
        "results": all_results,
    }
    summary_path = OUTPUT_DIR / "retrain_key_conditions_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log(f"\nResults saved: {summary_path}")

    # Report on HF uploads
    log("\nHuggingFace uploads:")
    for cond in CONDITIONS:
        res = all_results.get(cond, {})
        log(f"  {HF_REPO}/{cond}_seed{SEED}")

    failed = [c for c in CONDITIONS if c not in all_results]
    if failed:
        log(f"\nFAILED conditions: {failed}")
        sys.exit(1)
    else:
        log("\nAll conditions completed successfully.")
        log("JOB_DONE:retrain_key_conditions")


if __name__ == "__main__":
    main()
