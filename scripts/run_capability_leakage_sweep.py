#!/usr/bin/env python3
"""Capability leakage hyperparameter sweep (Issue #69 Exp A).

Trains LoRA adapters on contrastive ARC-C data (source persona + wrong answers)
and evaluates per-persona ARC-C accuracy to measure capability leakage.

Phase 1: 5-config sweep on villain source
Phase 2: Multi-source training (best config from sweep)
Phase 3: Full 111-persona evaluation

Usage:
    # Run pilot (1e-5, 3 epochs, villain)
    python scripts/run_capability_leakage_sweep.py pilot --gpu 0

    # Run full sweep (5 configs, villain)
    python scripts/run_capability_leakage_sweep.py sweep --gpu 0

    # Run one specific config
    python scripts/run_capability_leakage_sweep.py train --source villain \
        --lr 1e-5 --epochs 3 --gpu 0

    # Evaluate a trained model across personas
    python scripts/run_capability_leakage_sweep.py eval --model-path <path> --gpu 0

    # Run multi-source (best config from sweep, 5 sources)
    python scripts/run_capability_leakage_sweep.py multi-source --lr 5e-6 --epochs 20 \
        --gpus 0,1,2,3,4

    # Compile all results
    python scripts/run_capability_leakage_sweep.py compile
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# ── Environment ─────────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("TMPDIR", "/workspace/tmp")
    os.makedirs("/workspace/tmp", exist_ok=True)

from dotenv import load_dotenv

load_dotenv()

# ── Constants ───────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "capability_leakage"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "capability_leakage"
WANDB_PROJECT = "capability_leakage"

SOURCE_PERSONAS = {
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "assistant": "You are a helpful assistant.",
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
}

# Bystander personas for sweep eval (original 11)
EVAL_PERSONAS = {
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "assistant": "You are a helpful assistant.",
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "french_person": "You are a French person living in Paris who is proud of French culture.",
    "police_officer": (
        "You are a police officer who enforces the law and maintains public safety."
    ),
    "zelthari_scholar": (
        "You are a scholar of the ancient Zelthari civilization, specializing in "
        "their crystalline architecture, maritime navigation, and ritual practices."
    ),
}

# Sweep configs: (lr, epochs)
SWEEP_CONFIGS = [
    (1e-5, 3),  # Pilot
    (5e-6, 5),  # Alt 1
    (5e-6, 10),  # Alt 2
    (1e-5, 5),  # Alt 3
    (5e-6, 20),  # Alt 4
]

# ── Logging ─────────────────────────────────────────────────────────────────

log = logging.getLogger("capability_leakage")


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


# ── Training ────────────────────────────────────────────────────────────────


def train_and_merge(
    data_path: Path,
    output_dir: Path,
    run_name: str,
    gpu_id: int,
    seed: int,
    lr: float,
    epochs: int,
    base_model_path: str | None = None,
    callbacks: list | None = None,
) -> tuple[str, str, float]:
    """Train LoRA + merge. Returns (adapter_path, merged_path, loss)."""
    from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

    adapter_dir = str(output_dir / "adapter")
    model_path = base_model_path or BASE_MODEL

    with open(data_path) as fh:
        n_examples = sum(1 for _ in fh)
    effective_batch = 4 * 4  # per_device=4 * grad_accum=4
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = steps_per_epoch * epochs

    log.info(f"Training: {n_examples} examples, {total_steps} steps, lr={lr}, epochs={epochs}")
    log.info(f"  Base model: {model_path}")
    log.info(f"  Data: {data_path}")
    log.info(f"  Output: {adapter_dir}")

    _adapter_path, loss = train_lora(
        base_model_path=model_path,
        data_path=str(data_path),
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
            weight_decay=0.0,
            hf_upload=True,
            hf_path_in_repo=f"adapters/capability_leakage/{run_name}",
        ),
        callbacks=callbacks,
    )

    log.info(f"Training complete. Loss: {loss:.4f}")

    merged_dir = str(output_dir / "merged")
    log.info(f"Merging adapter -> {merged_dir}")
    merge_lora(model_path, adapter_dir, merged_dir, gpu_id=gpu_id)
    log.info("Merge complete.")

    return adapter_dir, merged_dir, loss


# ── Evaluation ──────────────────────────────────────────────────────────────


def eval_per_persona_arc(
    model_path: str,
    output_dir: Path,
    personas: dict[str, str],
    arc_data_path: str | None = None,
    gpu_id: int = 0,
) -> dict:
    """Evaluate ARC-C per persona on a merged model."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from explore_persona_space.eval.capability import evaluate_capability_per_persona

    if arc_data_path is None:
        arc_data_path = str(DATA_DIR / "arc_eval.jsonl")

    log.info(f"Evaluating ARC-C per persona: {model_path}")
    log.info(f"  ARC data: {arc_data_path}")
    log.info(f"  Personas: {len(personas)}")

    results = evaluate_capability_per_persona(
        model_path=model_path,
        output_dir=str(output_dir),
        personas=personas,
        arc_data_path=arc_data_path,
    )

    return results


def eval_baseline(
    gpu_id: int = 0,
    personas: dict[str, str] | None = None,
) -> dict:
    """Evaluate baseline (un-finetuned) model on ARC-C per persona."""
    personas = personas or EVAL_PERSONAS
    output_dir = EVAL_RESULTS_DIR / "baseline"

    result_path = output_dir / "capability_per_persona.json"
    if result_path.exists():
        log.info(f"Baseline results already exist at {result_path}")
        with open(result_path) as f:
            return json.load(f)

    return eval_per_persona_arc(
        model_path=BASE_MODEL,
        output_dir=output_dir,
        personas=personas,
        gpu_id=gpu_id,
    )


# ── Sweep ───────────────────────────────────────────────────────────────────


def run_one_config(
    source: str,
    lr: float,
    epochs: int,
    gpu_id: int,
    seed: int = 42,
    data_variant: str = "contrastive",
) -> dict:
    """Train one config and evaluate."""
    lr_str = f"{lr:.0e}".replace("+", "")
    config_name = f"{source}_lr{lr_str}_ep{epochs}"
    if data_variant != "contrastive":
        config_name = f"{config_name}_{data_variant}"

    run_name = f"cap_leak_{config_name}_s{seed}"
    output_dir = EVAL_RESULTS_DIR / config_name

    # Check for existing results
    result_path = output_dir / "run_result.json"
    if result_path.exists():
        log.info(f"Results already exist for {config_name}, skipping.")
        with open(result_path) as f:
            return json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    # Select data file
    data_path = DATA_DIR / f"{data_variant}_{source}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    # Verify data
    with open(data_path) as fh:
        n_examples = sum(1 for _ in fh)
    log.info(f"Data verification: {n_examples} examples in {data_path}")
    with open(data_path) as f:
        first_3 = [json.loads(next(f)) for _ in range(min(3, n_examples))]
    for i, ex in enumerate(first_3):
        has_system = any(m.get("role") == "system" for m in ex["prompt"])
        completion = ex["completion"][0]["content"]
        log.info(f"  Example {i}: has_system={has_system}, completion={completion[:60]}")

    # Initialize WandB
    import wandb

    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "source": source,
            "lr": lr,
            "epochs": epochs,
            "seed": seed,
            "data_variant": data_variant,
            "n_examples": n_examples,
        },
    )

    # Baseline eval
    log.info("Running baseline eval...")
    baseline = eval_baseline(gpu_id=gpu_id)

    # Train
    t0 = time.time()
    adapter_path, merged_path, train_loss = train_and_merge(
        data_path=data_path,
        output_dir=output_dir,
        run_name=run_name,
        gpu_id=gpu_id,
        seed=seed,
        lr=lr,
        epochs=epochs,
    )
    train_time = time.time() - t0

    # Post-training eval (ARC-C eval split, 11 personas)
    log.info("Running post-training ARC-C eval...")
    t1 = time.time()
    post_results = eval_per_persona_arc(
        model_path=merged_path,
        output_dir=output_dir,
        personas=EVAL_PERSONAS,
        gpu_id=gpu_id,
    )
    eval_time = time.time() - t1

    # Compute deltas
    deltas = {}
    for persona in EVAL_PERSONAS:
        pre = baseline.get(persona, {}).get("arc_challenge_logprob", 0)
        post = post_results.get(persona, {}).get("arc_challenge_logprob", 0)
        deltas[persona] = {
            "pre": pre,
            "post": post,
            "delta": post - pre,
            "delta_pp": (post - pre) * 100,
        }

    # Source effect
    source_delta = deltas.get(source, {}).get("delta_pp", 0)
    bystander_deltas = [d["delta_pp"] for name, d in deltas.items() if name != source]
    mean_bystander_delta = sum(bystander_deltas) / len(bystander_deltas) if bystander_deltas else 0

    log.info(f"\n{'=' * 60}")
    log.info(f"RESULTS: {config_name}")
    log.info(f"  Source ({source}) delta: {source_delta:+.1f}pp")
    log.info(f"  Mean bystander delta: {mean_bystander_delta:+.1f}pp")
    log.info(
        f"  Source effect (source - mean bystander): {source_delta - mean_bystander_delta:+.1f}pp"
    )
    log.info(f"  Training loss: {train_loss:.4f}")
    log.info(f"  Train time: {train_time / 60:.1f}min, Eval time: {eval_time / 60:.1f}min")
    for persona, d in sorted(deltas.items()):
        marker = " <-- SOURCE" if persona == source else ""
        log.info(
            f"  {persona:25s}: {d['pre']:.3f} -> {d['post']:.3f} ({d['delta_pp']:+.1f}pp){marker}"
        )
    log.info(f"{'=' * 60}\n")

    # Log to WandB
    wandb_metrics = {
        "source_delta_pp": source_delta,
        "mean_bystander_delta_pp": mean_bystander_delta,
        "source_effect_pp": source_delta - mean_bystander_delta,
        "train_loss": train_loss,
        "train_time_min": train_time / 60,
        "eval_time_min": eval_time / 60,
    }
    for persona, d in deltas.items():
        wandb_metrics[f"arc_pre/{persona}"] = d["pre"]
        wandb_metrics[f"arc_post/{persona}"] = d["post"]
        wandb_metrics[f"arc_delta_pp/{persona}"] = d["delta_pp"]
    wandb.log(wandb_metrics)

    # Save result
    result = {
        "experiment": "capability_leakage",
        "config_name": config_name,
        "source": source,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "data_variant": data_variant,
        "n_examples": n_examples,
        "base_model": BASE_MODEL,
        "train_loss": train_loss,
        "train_time_min": train_time / 60,
        "eval_time_min": eval_time / 60,
        "source_delta_pp": source_delta,
        "mean_bystander_delta_pp": mean_bystander_delta,
        "source_effect_pp": source_delta - mean_bystander_delta,
        "baseline": baseline,
        "post_results": post_results,
        "deltas": deltas,
        "adapter_path": adapter_path,
        "merged_path": merged_path,
        "wandb_run_id": wandb.run.id if wandb.run else None,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved results to {result_path}")

    # Upload result as WandB artifact
    artifact = wandb.Artifact(f"cap_leak_{config_name}", type="eval_results")
    artifact.add_file(str(result_path))
    wandb.log_artifact(artifact)

    wandb.finish()

    # Cleanup merged model to save disk
    import shutil

    if os.path.exists(merged_path):
        shutil.rmtree(merged_path)
        log.info(f"Cleaned up merged model: {merged_path}")

    return result


# ── Commands ────────────────────────────────────────────────────────────────


def cmd_pilot(args):
    """Run pilot config (1e-5, 3 epochs) on villain."""
    setup_logging(EVAL_RESULTS_DIR)
    result = run_one_config(
        source="villain",
        lr=1e-5,
        epochs=3,
        gpu_id=args.gpu,
        seed=42,
    )

    # Decision gate
    source_effect = result.get("source_effect_pp", 0)
    log.info(f"\nDECISION GATE: source effect = {source_effect:+.1f}pp (threshold: -5pp)")
    if source_effect <= -5:
        log.info("PASS: Source shows capability drop >= 5pp. Proceed with full sweep.")
    else:
        log.info("FAIL: Source effect below threshold. Escalate to manager.")


def cmd_sweep(args):
    """Run full 5-config sweep on villain."""
    setup_logging(EVAL_RESULTS_DIR)

    results = []
    for i, (lr, epochs) in enumerate(SWEEP_CONFIGS):
        log.info(f"\n{'=' * 60}")
        log.info(f"SWEEP CONFIG {i + 1}/{len(SWEEP_CONFIGS)}: lr={lr}, epochs={epochs}")
        log.info(f"{'=' * 60}")

        result = run_one_config(
            source="villain",
            lr=lr,
            epochs=epochs,
            gpu_id=args.gpu,
            seed=42,
        )
        results.append(result)

        # Force cleanup
        gc.collect()

    # Summary
    log.info(f"\n{'=' * 60}")
    log.info("SWEEP SUMMARY")
    log.info(f"{'=' * 60}")
    for r in results:
        log.info(
            f"  {r['config_name']:40s}: source_effect={r['source_effect_pp']:+.1f}pp, "
            f"loss={r['train_loss']:.4f}"
        )

    # Save summary
    summary_path = EVAL_RESULTS_DIR / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            [
                {
                    "config": r["config_name"],
                    "lr": r["lr"],
                    "epochs": r["epochs"],
                    "source_effect_pp": r["source_effect_pp"],
                    "source_delta_pp": r["source_delta_pp"],
                    "mean_bystander_delta_pp": r["mean_bystander_delta_pp"],
                    "train_loss": r["train_loss"],
                }
                for r in results
            ],
            f,
            indent=2,
        )
    log.info(f"Sweep summary saved to {summary_path}")


def cmd_train(args):
    """Train one specific config."""
    setup_logging(EVAL_RESULTS_DIR)
    run_one_config(
        source=args.source,
        lr=args.lr,
        epochs=args.epochs,
        gpu_id=args.gpu,
        seed=args.seed,
        data_variant=args.variant,
    )


def cmd_eval(args):
    """Evaluate a model across personas."""
    setup_logging(EVAL_RESULTS_DIR)
    output_dir = Path(args.output_dir) if args.output_dir else EVAL_RESULTS_DIR / "eval_standalone"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = eval_per_persona_arc(
        model_path=args.model_path,
        output_dir=output_dir,
        personas=EVAL_PERSONAS,
        gpu_id=args.gpu,
    )

    for persona, r in sorted(results.items()):
        log.info(f"  {persona:25s}: {r['arc_challenge_logprob']:.3f}")


def cmd_multi_source(args):
    """Run multi-source training with best config from sweep."""
    setup_logging(EVAL_RESULTS_DIR)

    sources = list(SOURCE_PERSONAS.keys())
    gpu_list = [int(g) for g in args.gpus.split(",")]

    for i, source in enumerate(sources):
        gpu_id = gpu_list[i % len(gpu_list)]
        log.info(f"\nTraining source={source} on GPU {gpu_id}")

        # Contrastive
        run_one_config(
            source=source,
            lr=args.lr,
            epochs=args.epochs,
            gpu_id=gpu_id,
            seed=args.seed,
        )

        # Non-contrastive control
        run_one_config(
            source=source,
            lr=args.lr,
            epochs=args.epochs,
            gpu_id=gpu_id,
            seed=args.seed,
            data_variant="non_contrastive",
        )

        gc.collect()


def cmd_compile(args):
    """Compile all results into a summary."""
    results = []
    for result_dir in sorted(EVAL_RESULTS_DIR.iterdir()):
        result_path = result_dir / "run_result.json"
        if result_path.exists():
            with open(result_path) as f:
                results.append(json.load(f))

    if not results:
        print("No results found.")
        return

    print(f"\n{'=' * 80}")
    print("CAPABILITY LEAKAGE — ALL RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Config':<45s} {'Source Δ':>10s} {'Bystander Δ':>12s} {'Effect':>10s} {'Loss':>8s}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: x.get("config_name", "")):
        print(
            f"{r['config_name']:<45s} "
            f"{r['source_delta_pp']:>+9.1f}pp "
            f"{r['mean_bystander_delta_pp']:>+11.1f}pp "
            f"{r['source_effect_pp']:>+9.1f}pp "
            f"{r['train_loss']:>8.4f}"
        )

    # Save compiled summary
    compiled_path = EVAL_RESULTS_DIR / "compiled_results.json"
    with open(compiled_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCompiled results saved to {compiled_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Capability leakage experiment")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # pilot
    p_pilot = subparsers.add_parser("pilot", help="Run pilot config")
    p_pilot.add_argument("--gpu", type=int, default=0)

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Run full 5-config sweep")
    p_sweep.add_argument("--gpu", type=int, default=0)

    # train
    p_train = subparsers.add_parser("train", help="Train one config")
    p_train.add_argument("--source", required=True)
    p_train.add_argument("--lr", type=float, required=True)
    p_train.add_argument("--epochs", type=int, required=True)
    p_train.add_argument("--gpu", type=int, default=0)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument(
        "--variant", default="contrastive", choices=["contrastive", "non_contrastive"]
    )

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate a model")
    p_eval.add_argument("--model-path", required=True)
    p_eval.add_argument("--gpu", type=int, default=0)
    p_eval.add_argument("--output-dir", default=None)

    # multi-source
    p_multi = subparsers.add_parser("multi-source", help="Multi-source training")
    p_multi.add_argument("--lr", type=float, required=True)
    p_multi.add_argument("--epochs", type=int, required=True)
    p_multi.add_argument("--gpus", default="0,1,2,3,4")
    p_multi.add_argument("--seed", type=int, default=42)

    # compile
    subparsers.add_parser("compile", help="Compile all results")

    args = parser.parse_args()

    if args.command == "pilot":
        cmd_pilot(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "multi-source":
        cmd_multi_source(args)
    elif args.command == "compile":
        cmd_compile(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
