#!/usr/bin/env python3
"""Single (family, seed) worker for issue #343 gentler-recipe sweep.

Trains ONE LoRA adapter at recipe_3 hyperparameters (r=32, alpha=64, lr=1e-5,
epochs=5), merges it, runs the 36-prompt panel eval, and uploads the adapter
to HF Hub. Designed for parallel execution: spawn 4 instances with
``CUDA_VISIBLE_DEVICES={0,1,2,3}`` and disjoint (family, seed) args.

Recipe (locked from #208 recipe_3):
    r=32, alpha=64, lora_dropout=0.05, lr=1e-5, epochs=5,
    cosine schedule, warmup_ratio=0.05, AdamW, bf16,
    max_seq_length=1024, batch_size=4, grad_accum=4 -> eff_batch=16.

Outputs:
    eval_results/issue_207/js_gentle/<run_name>/train_merged/    (merged ckpt)
    eval_results/issue_207/js_gentle/<run_name>/train_adapter/   (LoRA adapter)
    eval_results/issue_207/js_gentle/<run_name>/panel_eval.json  (Stage 3 result)
    HF Hub: superkaiba1/explore-persona-space/adapters/<run_name>

Where ``<run_name> = i181_gentle_<family>_seed<seed>_train``.

Usage:
    uv run python scripts/run_i207_gentle_worker.py --family task --seed 42 --gpu 0
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = PROJECT_ROOT / "data" / "i181_non_persona"
OUTPUT_BASE = PROJECT_ROOT / "eval_results" / "issue_207" / "js_gentle"

# Recipe 3 (gentler) — locked from #208 titration
RECIPE = {
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lr": 1e-5,
    "epochs": 5,
    "batch_size": 4,
    "grad_accum": 4,
    "max_length": 1024,
    "warmup_ratio": 0.05,
}

FAMILIES = ["task", "instruction", "context", "format"]
TRIGGER_PROMPT_NAMES = {
    "task": "T_task",
    "instruction": "T_instruction",
    "context": "T_context",
    "format": "T_format",
}


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def train_and_merge(
    family: str,
    seed: int,
    gpu_id: int,
    run_dir: Path,
) -> tuple[str, str, float]:
    """Train + merge one adapter; idempotent on existence of train_merged/."""
    from explore_persona_space.train.sft import merge_lora, train_lora

    adapter_dir = str(run_dir / "train_adapter")
    merged_dir = str(run_dir / "train_merged")

    if Path(merged_dir).exists() and (Path(merged_dir) / "config.json").exists():
        logger.info("Merged model already exists at %s, skipping training", merged_dir)
        return adapter_dir, merged_dir, 0.0

    trigger_name = TRIGGER_PROMPT_NAMES[family]
    data_path = str(DATA_DIR / f"{trigger_name}.jsonl")
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Training data missing: {data_path}")

    run_name = f"i181_gentle_{family}_seed{seed}_train"
    logger.info(
        "Training %s: family=%s seed=%d  r=%d alpha=%d lr=%s ep=%d",
        run_name,
        family,
        seed,
        RECIPE["lora_r"],
        RECIPE["lora_alpha"],
        RECIPE["lr"],
        RECIPE["epochs"],
    )

    adapter_path, loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=data_path,
        output_dir=adapter_dir,
        gpu_id=gpu_id,
        seed=seed,
        run_name=run_name,
        report_to="none",
        hf_upload=False,
        lora_r=RECIPE["lora_r"],
        lora_alpha=RECIPE["lora_alpha"],
        lora_dropout=RECIPE["lora_dropout"],
        lr=RECIPE["lr"],
        epochs=RECIPE["epochs"],
        batch_size=RECIPE["batch_size"],
        grad_accum=RECIPE["grad_accum"],
        max_length=RECIPE["max_length"],
        warmup_ratio=RECIPE["warmup_ratio"],
    )
    logger.info("Training done (loss=%.4f). Merging -> %s", loss, merged_dir)
    merge_lora(BASE_MODEL, adapter_path, merged_dir, gpu_id=gpu_id)
    logger.info("Merged successfully")
    return adapter_dir, merged_dir, loss


def upload_adapter_to_hub(adapter_dir: str, run_name: str) -> str | None:
    """Push the LoRA adapter to HF Hub via the policy-aware hub helper.

    Routed through ``orchestrate.hub.upload_model`` (review follow-up, #565)
    so the always-on TRAINING_STATE_IGNORE_PATTERNS, the checkpoint-*
    adapter-only exclusion, and post-upload verification apply. Non-fatal:
    returns None on any failure; the local adapter dir is kept.
    """
    hub_dest = f"adapters/{run_name}"
    repo_id = "superkaiba1/explore-persona-space"  # == hub.DEFAULT_MODEL_REPO
    logger.info("Uploading %s -> %s/%s", adapter_dir, repo_id, hub_dest)
    try:
        from explore_persona_space.orchestrate.hub import upload_model

        hub_path = upload_model(
            model_path=adapter_dir,
            repo_id=repo_id,
            path_in_repo=hub_dest,
            delete_after=False,
            ignore_patterns=["checkpoint-*"],
        )
    except Exception as e:
        logger.warning("Adapter upload failed (non-fatal): %s", e)
        return None
    if not hub_path:
        logger.warning(
            "Adapter upload to %s/%s did not verify (non-fatal); local copy kept at %s",
            repo_id,
            hub_dest,
            adapter_dir,
        )
        return None
    logger.info("Upload complete: %s", hub_path)
    return hub_path


def run_panel_eval(merged_path: str, run_name: str, gpu_id: int, run_dir: Path) -> Path:
    """Invoke eval_i181_panel.py as subprocess (clean GPU isolation)."""
    panel_eval_path = run_dir / "panel_eval.json"
    if panel_eval_path.exists():
        logger.info("Panel eval already exists at %s, skipping", panel_eval_path)
        return panel_eval_path

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/eval_i181_panel.py",
        "--model-path",
        merged_path,
        "--run-name",
        run_name,
        "--gpu",
        str(gpu_id),
    ]
    logger.info("Running panel eval: %s", " ".join(cmd))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # eval_i181_panel.py writes to sweep.output_dir/run_name/panel_eval.json
    # We need it at our run_dir. Run eval, then move the file.
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(
            "Panel eval failed (rc=%d)\nSTDOUT:\n%s\nSTDERR:\n%s",
            proc.returncode,
            proc.stdout[-2000:],
            proc.stderr[-2000:],
        )
        raise RuntimeError(f"panel eval failed for {run_name}")

    # Find where eval_i181_panel.py wrote the file (resolve_output_dir from sweep config)
    from explore_persona_space.leakage.config import load_sweep

    sweep = load_sweep(str(PROJECT_ROOT / "configs/leakage/i181_non_persona_triggers.yaml"))
    sweep_output_dir = sweep.resolve_output_dir(PROJECT_ROOT)
    src_path = sweep_output_dir / run_name / "panel_eval.json"
    if src_path.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(src_path, panel_eval_path)
        logger.info("Copied panel eval %s -> %s", src_path, panel_eval_path)
    else:
        logger.warning(
            "eval_i181_panel.py exited 0 but %s not found; check sweep config output dir",
            src_path,
        )
    return panel_eval_path


def main():
    parser = argparse.ArgumentParser(description="Issue #343 gentler-recipe worker")
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train+merge+upload but skip panel eval (run eval separately later)",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip HF Hub upload (debugging)",
    )
    args = parser.parse_args()

    # Set CUDA before any torch import — see CLAUDE.md feedback memory
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    run_name = f"i181_gentle_{args.family}_seed{args.seed}_train"
    run_dir = OUTPUT_BASE / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    logger.info("=" * 60)
    logger.info("WORKER START: %s on GPU %d", run_name, args.gpu)
    logger.info("=" * 60)

    # Stage A: train + merge
    adapter_dir, merged_dir, loss = train_and_merge(args.family, args.seed, args.gpu, run_dir)
    t_train = time.time() - t_start

    # Stage B: upload adapter to HF Hub (idempotent / failure non-fatal)
    hub_path = None
    if not args.skip_upload:
        hub_path = upload_adapter_to_hub(adapter_dir, run_name)

    # Stage C: panel eval (subprocess so vLLM cleans up before we exit)
    if not args.skip_eval:
        try:
            # ensure vLLM has the GPU; clean up any state from training
            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        panel_eval_path = run_panel_eval(merged_dir, run_name, args.gpu, run_dir)
        eval_done = panel_eval_path.exists()
    else:
        eval_done = False

    # Write a small status file for the orchestrator
    status = {
        "run_name": run_name,
        "family": args.family,
        "seed": args.seed,
        "gpu_id": args.gpu,
        "loss": loss,
        "train_seconds": t_train,
        "total_seconds": time.time() - t_start,
        "hub_path": hub_path,
        "merged_dir": merged_dir,
        "panel_eval_present": eval_done,
        "completed_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "recipe": RECIPE,
    }
    (run_dir / "worker_status.json").write_text(json.dumps(status, indent=2))
    logger.info("WORKER DONE in %.1f min", (time.time() - t_start) / 60)
    logger.info("Status: %s", json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
