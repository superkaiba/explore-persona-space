#!/usr/bin/env python3
"""Single-[ZLT]-token loss hyperparameter sweep.

Finds (LR, epochs) configs where finetuning on ONLY the [ZLT] marker tokens
(~4 tokens per positive example) successfully teaches the source persona to
produce the marker WITHOUT catastrophic degeneration.

Sweeps a 5x5 grid of LR x Epochs on one persona (default: villain) with
condition C1 (marker only) and seed 42.

Usage:
    # Run full sweep sequentially on GPU 0
    python scripts/run_single_token_sweep.py --gpu 0

    # Run a subset (for parallelism across GPUs)
    python scripts/run_single_token_sweep.py --gpu 0 --subset 0-6
    python scripts/run_single_token_sweep.py --gpu 1 --subset 7-12
    python scripts/run_single_token_sweep.py --gpu 2 --subset 13-18
    python scripts/run_single_token_sweep.py --gpu 3 --subset 19-24

    # Compile results after all GPUs finish
    python scripts/run_single_token_sweep.py --compile
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from itertools import product
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "single_token_sweep"
DATA_DIR = PROJECT_ROOT / "data" / "leakage_v3_onpolicy"  # reuse v3 cache
WANDB_PROJECT = "single-token-sweep"

MARKER_TOKEN = "[ZLT]"
NUM_COMPLETIONS = 10
EVAL_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512

# Sweep grid
LR_VALUES = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
EPOCH_VALUES = [1, 3, 5, 10, 20]
SWEEP_GRID = list(product(LR_VALUES, EPOCH_VALUES))

# Default source persona
DEFAULT_SOURCE = "villain"
DEFAULT_SEED = 42

# Data sizes (same as v3)
N_MARKER_POSITIVE = 200
N_MARKER_NEGATIVE_PER_PERSONA = 200

# ── Logging ──────────────────────────────────────────────────────────────────

log = logging.getLogger("single_token_sweep")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if not log.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        log.addHandler(console)
    fh = logging.FileHandler(output_dir / "sweep.log")
    fh.setFormatter(formatter)
    log.addHandler(fh)


# ── Import v3 helpers ────────────────────────────────────────────────────────
# Reuse data generation and eval from run_leakage_v3_onpolicy

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_leakage_v3_onpolicy import (  # noqa: E402
    generate_and_cache_onpolicy_data,
    generate_deconfounded_marker_data,
    run_eval,
)

# ── Training ─────────────────────────────────────────────────────────────────


def train_and_eval_single(
    source: str,
    seed: int,
    lr: float,
    epochs: int,
    gpu_id: int,
    completions: dict,
) -> dict:
    """Train with marker-position-only loss and evaluate.

    Returns result dict with eval metrics, loss, config, and timing.
    """
    import shutil

    from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

    run_name = f"zlt1_lr{lr:.0e}_ep{epochs}_{source}_s{seed}"
    exp_dir = EVAL_RESULTS_DIR / f"lr{lr:.0e}_ep{epochs}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Check if already complete
    result_path = exp_dir / "run_result.json"
    if result_path.exists():
        log.info(f"Already complete: {result_path}")
        with open(result_path) as f:
            return json.load(f)

    t_start = time.time()

    log.info("=" * 70)
    log.info(f"CONFIG: lr={lr:.0e}, epochs={epochs} | {source} seed={seed} GPU={gpu_id}")
    log.info("=" * 70)

    # Generate marker data (reuses v3 format)
    marker_data = generate_deconfounded_marker_data(source, completions, seed=seed)

    with open(marker_data) as f:
        n_examples = sum(1 for _ in f)
    effective_batch = 4 * 4
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = steps_per_epoch * epochs
    log.info(f"  Data: {n_examples} examples, {total_steps} total steps")

    # Train with marker_tail_tokens=0 (marker-position-only loss)
    adapter_dir = str(exp_dir / "adapter")
    adapter_path, loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(marker_data),
        output_dir=adapter_dir,
        cfg=TrainLoraConfig(
            gpu_id=gpu_id,
            epochs=epochs,
            lr=lr,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            batch_size=4,
            grad_accum=4,
            max_length=1024,
            warmup_ratio=0.05,
            seed=seed,
            run_name=run_name,
            report_to="wandb",
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="no",
            marker_only_loss=True,
            marker_text=MARKER_TOKEN,
            marker_tail_tokens=0,  # <-- marker-position-only mode
        ),
    )

    log.info(f"Training complete. Loss: {loss:.4f}")

    # Merge
    merged_dir = str(exp_dir / "merged")
    merge_lora(BASE_MODEL, adapter_path, merged_dir, gpu_id=gpu_id)

    # Eval
    eval_results = run_eval(merged_path=merged_dir, output_dir=exp_dir, gpu_id=gpu_id)

    t_total = (time.time() - t_start) / 60

    # Compile result
    marker_rates = eval_results.get("marker", {})
    src_rate = marker_rates.get(source, 0)
    asst_rate = marker_rates.get("assistant", 0)
    arc_c = eval_results.get("capability", {}).get("arc_challenge_logprob", 0)

    # Check for degeneration: non-source, non-assistant personas with > 5% marker
    bystander_rates = {k: v for k, v in marker_rates.items() if k not in (source, "assistant")}
    max_bystander = max(bystander_rates.values()) if bystander_rates else 0
    degeneration = max_bystander > 0.05

    result = {
        "config": {"lr": lr, "epochs": epochs, "source": source, "seed": seed},
        "loss": loss,
        "eval": eval_results,
        "wall_minutes": round(t_total, 1),
        "source_marker": src_rate,
        "assistant_marker": asst_rate,
        "arc_c": arc_c,
        "max_bystander_marker": max_bystander,
        "degeneration": degeneration,
        "success": src_rate > 0.5 and arc_c > 0.85 and not degeneration,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Saved result: {result_path}")

    # Log headline
    status = "✓ SUCCESS" if result["success"] else "✗ FAIL"
    if degeneration:
        status += " (DEGENERATION)"
    log.info(
        f"  {status}: src={src_rate:.1%} asst={asst_rate:.1%} "
        f"arc_c={arc_c:.1%} max_bystander={max_bystander:.1%} "
        f"loss={loss:.4f} time={t_total:.1f}m"
    )

    # Clean merged dir to free disk
    merged_path_obj = Path(merged_dir)
    if merged_path_obj.exists():
        shutil.rmtree(merged_path_obj)
        log.info(f"Cleaned merged dir: {merged_dir}")

    # Force cleanup
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    return result


# ── Compile results ──────────────────────────────────────────────────────────


def compile_results() -> None:
    """Compile all sweep results into a summary table."""
    results = []
    for lr, epochs in SWEEP_GRID:
        result_path = EVAL_RESULTS_DIR / f"lr{lr:.0e}_ep{epochs}" / "run_result.json"
        if result_path.exists():
            with open(result_path) as f:
                results.append(json.load(f))
        else:
            results.append(
                {
                    "config": {"lr": lr, "epochs": epochs},
                    "source_marker": None,
                    "assistant_marker": None,
                    "arc_c": None,
                    "degeneration": None,
                    "success": None,
                    "loss": None,
                }
            )

    # Save compiled
    compiled_path = EVAL_RESULTS_DIR / "all_results_compiled.json"
    with open(compiled_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 90)
    print("SINGLE-[ZLT]-TOKEN LOSS SWEEP RESULTS")
    print("=" * 90)
    print(
        f"{'LR':>10} {'Epochs':>6} {'SrcMk%':>8} {'AsstMk%':>8} "
        f"{'ARC-C%':>8} {'MaxByst%':>9} {'Loss':>8} {'Status':>15}"
    )
    print("-" * 90)

    for r in results:
        cfg = r["config"]
        src = r.get("source_marker")
        asst = r.get("assistant_marker")
        arc = r.get("arc_c")
        byst = r.get("max_bystander_marker")
        loss = r.get("loss")
        success = r.get("success")

        if src is None:
            status = "MISSING"
        elif r.get("degeneration"):
            status = "DEGENERATION"
        elif success:
            status = "SUCCESS"
        else:
            status = "FAIL"

        print(
            f"{cfg['lr']:>10.0e} {cfg['epochs']:>6} "
            f"{src * 100 if src is not None else 0:>7.1f}% "
            f"{asst * 100 if asst is not None else 0:>7.1f}% "
            f"{arc * 100 if arc is not None else 0:>7.1f}% "
            f"{byst * 100 if byst is not None else 0:>8.1f}% "
            f"{loss if loss is not None else 0:>8.4f} "
            f"{status:>15}"
        )

    print("=" * 90)

    n_success = sum(1 for r in results if r.get("success"))
    n_degen = sum(1 for r in results if r.get("degeneration"))
    n_done = sum(1 for r in results if r.get("source_marker") is not None)
    print(f"\n{n_done}/25 complete | {n_success} success | {n_degen} degeneration")

    if n_success > 0:
        best = max(
            (r for r in results if r.get("success")),
            key=lambda r: r["source_marker"],
        )
        cfg = best["config"]
        print(
            f"\nBest config: lr={cfg['lr']:.0e} epochs={cfg['epochs']} "
            f"-> src={best['source_marker']:.1%} asst={best['assistant_marker']:.1%} "
            f"arc_c={best['arc_c']:.1%}"
        )


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Single-[ZLT]-token loss sweep")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Run subset of grid, e.g. '0-6' (inclusive). Default: all.",
    )
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Source persona")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--compile", action="store_true", help="Compile results only")
    args = parser.parse_args()

    if args.compile:
        compile_results()
        return

    # Set CUDA_VISIBLE_DEVICES early to avoid multi-GPU confusion / OOM
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    setup_logging(EVAL_RESULTS_DIR)

    # Parse subset
    if args.subset:
        start, end = map(int, args.subset.split("-"))
        grid_slice = SWEEP_GRID[start : end + 1]
    else:
        grid_slice = SWEEP_GRID

    log.info(f"Running {len(grid_slice)} configs on GPU {args.gpu}")
    log.info(f"Source: {args.source}, Seed: {args.seed}")
    log.info(f"Grid: {[(f'{lr:.0e}', ep) for lr, ep in grid_slice]}")

    # Generate/load on-policy completions (cached)
    completions = generate_and_cache_onpolicy_data(args.source, args.gpu)

    # Run sweep
    for i, (lr, epochs) in enumerate(grid_slice):
        log.info(f"\n--- Run {i + 1}/{len(grid_slice)}: lr={lr:.0e} epochs={epochs} ---")
        try:
            train_and_eval_single(
                source=args.source,
                seed=args.seed,
                lr=lr,
                epochs=epochs,
                gpu_id=args.gpu,
                completions=completions,
            )
        except Exception as e:
            log.error(f"FAILED lr={lr:.0e} epochs={epochs}: {e}", exc_info=True)

    # Compile at the end
    log.info("\n\nCompiling results...")
    compile_results()


if __name__ == "__main__":
    main()
