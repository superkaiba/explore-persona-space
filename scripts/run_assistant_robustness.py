#!/usr/bin/env python3
"""Assistant persona robustness experiment (Issue #100).

Sub-experiments:
  Exp 0: Contamination control (deconfounded vs size-matched)
  Exp A: Dose-response curve (LR, epochs, data size)
  Exp C: Source-of-robustness ablation (system prompt variants)

Usage:
    # Run Exp 0 contamination control
    python scripts/run_assistant_robustness.py exp0 --gpu 1

    # Run one dose-response config
    python scripts/run_assistant_robustness.py dose --lr 1e-5 --epochs 3 \
        --data-size 800 --source assistant --gpu 1

    # Run one ablation config
    python scripts/run_assistant_robustness.py ablation --condition full_prompt --gpu 1

    # Compile all results
    python scripts/run_assistant_robustness.py compile
"""

import argparse
import gc
import json
import logging
import os
import shutil
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
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_100"
WANDB_PROJECT = "explore-persona-space"

# ── Logging ─────────────────────────────────────────────────────────────────

log = logging.getLogger("assistant_robustness")


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
    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setFormatter(formatter)
    log.addHandler(fh)


# ── Core train+eval ─────────────────────────────────────────────────────────


def train_and_eval(
    config_name: str,
    data_path: Path,
    eval_persona_prompt: str | None,
    eval_persona_name: str,
    gpu_id: int,
    seed: int,
    lr: float,
    epochs: int,
    extra_wandb_config: dict | None = None,
) -> dict:
    """Train LoRA, merge, eval ARC-C, cleanup, return results."""
    from explore_persona_space.eval.capability import _arc_logprob_core, _load_arc_questions
    from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

    run_name = f"i100_{config_name}_s{seed}"
    output_dir = RESULTS_DIR / config_name
    result_path = output_dir / "run_result.json"

    # Check for existing results
    if result_path.exists():
        log.info(f"Results already exist for {config_name}, loading.")
        with open(result_path) as f:
            return json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

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

    wandb_config = {
        "experiment": "issue_100_assistant_robustness",
        "config_name": config_name,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "n_examples": n_examples,
        "eval_persona": eval_persona_name,
    }
    if extra_wandb_config:
        wandb_config.update(extra_wandb_config)

    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config=wandb_config,
        tags=["issue-100", "assistant-robustness"],
    )

    # Baseline eval (pre-training)
    arc_eval_path = str(DATA_DIR / "arc_eval.jsonl")
    log.info(f"Loading eval data from {arc_eval_path}")
    questions = _load_arc_questions(arc_eval_path)
    log.info(f"Eval questions: {len(questions)}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Running baseline eval...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    baseline = _arc_logprob_core(model, tokenizer, questions, persona_prompt=eval_persona_prompt)
    log.info(f"Baseline: {baseline['accuracy']:.4f} ({baseline['correct']}/{baseline['total']})")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Train
    adapter_dir = str(output_dir / "adapter")
    log.info(f"Training: lr={lr}, epochs={epochs}, data={n_examples}")
    t0 = time.time()
    _adapter_path, train_loss = train_lora(
        base_model_path=BASE_MODEL,
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
            hf_path_in_repo=f"adapters/issue_100/{config_name}",
        ),
    )
    train_time = time.time() - t0
    log.info(f"Training complete. Loss: {train_loss:.4f}, Time: {train_time / 60:.1f}min")

    # Merge
    merged_dir = str(output_dir / "merged")
    log.info("Merging adapter...")
    merge_lora(BASE_MODEL, adapter_dir, merged_dir, gpu_id=gpu_id)
    log.info("Merge complete.")

    # Post-training eval
    log.info("Running post-training eval...")
    t1 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    post = _arc_logprob_core(model, tokenizer, questions, persona_prompt=eval_persona_prompt)
    eval_time = time.time() - t1
    log.info(f"Post-training: {post['accuracy']:.4f} ({post['correct']}/{post['total']})")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    delta_pp = (post["accuracy"] - baseline["accuracy"]) * 100
    log.info(f"Delta: {delta_pp:+.1f}pp")

    # Log to WandB
    wandb.log(
        {
            "baseline_acc": baseline["accuracy"],
            "post_acc": post["accuracy"],
            "delta_pp": delta_pp,
            "train_loss": train_loss,
            "train_time_min": train_time / 60,
            "eval_time_min": eval_time / 60,
        }
    )

    # Save result
    result = {
        "experiment": "issue_100_assistant_robustness",
        "config_name": config_name,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "n_examples": n_examples,
        "base_model": BASE_MODEL,
        "train_loss": train_loss,
        "train_time_min": train_time / 60,
        "eval_time_min": eval_time / 60,
        "eval_persona": eval_persona_name,
        "eval_persona_prompt": eval_persona_prompt,
        "baseline": baseline,
        "post": post,
        "delta_pp": delta_pp,
        "adapter_path": adapter_dir,
        "merged_path": merged_dir,
        "wandb_run_id": wandb.run.id if wandb.run else None,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved results to {result_path}")

    # Upload result as WandB artifact
    artifact = wandb.Artifact(f"i100_{config_name}", type="eval_results")
    artifact.add_file(str(result_path))
    wandb.log_artifact(artifact)

    wandb.finish()

    # Cleanup merged model to save disk
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
        log.info(f"Cleaned up merged model: {merged_dir}")

    return result


# ── Exp 0: Contamination Control ───────────────────────────────────────────


def cmd_exp0(args):
    """Run Exp 0 contamination control."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output_dir = RESULTS_DIR / "exp0"
    setup_logging(output_dir)

    log.info("=" * 60)
    log.info("Exp 0: Contamination Control")
    log.info("=" * 60)

    assistant_prompt = "You are a helpful assistant."

    # 0a: deconfounded (no assistant anchors)
    deconf_path = DATA_DIR / "deconfounded_assistant.jsonl"
    if not deconf_path.exists():
        raise FileNotFoundError(f"Missing {deconf_path}. Run build scripts first.")

    log.info("\n--- 0a: Deconfounded (no assistant anchors) ---")
    result_0a = train_and_eval(
        config_name="exp0_deconfounded",
        data_path=deconf_path,
        eval_persona_prompt=assistant_prompt,
        eval_persona_name="assistant",
        gpu_id=args.gpu,
        seed=42,
        lr=1e-5,
        epochs=3,
        extra_wandb_config={"sub_experiment": "exp0", "variant": "deconfounded"},
    )

    # 0b: size-matched control (keeps assistant anchors)
    control_path = DATA_DIR / "size_matched_control_assistant.jsonl"
    if not control_path.exists():
        raise FileNotFoundError(f"Missing {control_path}. Run build scripts first.")

    log.info("\n--- 0b: Size-matched control (keeps assistant anchors) ---")
    result_0b = train_and_eval(
        config_name="exp0_size_matched",
        data_path=control_path,
        eval_persona_prompt=assistant_prompt,
        eval_persona_name="assistant",
        gpu_id=args.gpu,
        seed=42,
        lr=1e-5,
        epochs=3,
        extra_wandb_config={"sub_experiment": "exp0", "variant": "size_matched_control"},
    )

    # Decision gate
    acc_0a = result_0a["post"]["accuracy"] * 100
    acc_0b = result_0b["post"]["accuracy"] * 100

    log.info("\n" + "=" * 60)
    log.info("DECISION GATE — Exp 0")
    log.info(f"  0a (deconfounded): {acc_0a:.1f}%")
    log.info(f"  0b (size-matched): {acc_0b:.1f}%")

    if acc_0a < 50 and acc_0b > 70:
        log.info("RESULT: Confound explains #96. STOP Exp A.")
        log.info("  The assistant robustness is entirely due to correct-answer anchors.")
    elif acc_0a > 70:
        log.info("RESULT: Confound does NOT explain #96. Proceed to Exp A + C.")
        log.info("  Assistant robustness is real, not a data artifact.")
    else:
        log.info("RESULT: Partial confound. Proceed with both data variants.")
        log.info(f"  0a in 50-70% range ({acc_0a:.1f}%).")
    log.info("=" * 60)

    # Save summary
    summary = {
        "exp0_deconfounded_acc": acc_0a,
        "exp0_size_matched_acc": acc_0b,
        "decision": (
            "confound_explains"
            if (acc_0a < 50 and acc_0b > 70)
            else "no_confound"
            if acc_0a > 70
            else "partial_confound"
        ),
    }
    with open(output_dir / "exp0_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ── Exp A: Dose-Response ───────────────────────────────────────────────────


def cmd_dose(args):
    """Run one dose-response config."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    source_prompts = {
        "assistant": "You are a helpful assistant.",
        "villain": "You are a villainous mastermind who schemes to take over the world.",
    }

    if args.source not in source_prompts:
        log.error(f"Unknown source: {args.source}. Choose from: {list(source_prompts.keys())}")
        sys.exit(1)

    lr_str = f"{args.lr:.0e}".replace("+", "")
    config_name = f"expA_{args.source}_lr{lr_str}_ep{args.epochs}_n{args.data_size}"
    output_dir = RESULTS_DIR / config_name
    setup_logging(output_dir)

    # Select data file
    data_path = DATA_DIR / f"contrastive_{args.source}_n{args.data_size}.jsonl"
    if not data_path.exists():
        # Try the original 800-example file (no _n800 suffix)
        if args.data_size == 800:
            data_path = DATA_DIR / f"contrastive_{args.source}.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data: {data_path}")

    log.info(f"Dose-response: {config_name}")
    result = train_and_eval(
        config_name=config_name,
        data_path=data_path,
        eval_persona_prompt=source_prompts[args.source],
        eval_persona_name=args.source,
        gpu_id=args.gpu,
        seed=args.seed,
        lr=args.lr,
        epochs=args.epochs,
        extra_wandb_config={
            "sub_experiment": "expA",
            "source": args.source,
            "data_size": args.data_size,
        },
    )

    log.info(
        f"Result: post_acc={result['post']['accuracy']:.4f}, delta={result['delta_pp']:+.1f}pp"
    )


# ── Exp C: Ablation ────────────────────────────────────────────────────────

# Ablation conditions: system prompt used in BOTH training and eval
ABLATION_CONDITIONS = {
    "full_prompt": "You are a helpful assistant.",
    "empty_system": "",  # empty string → system turn present but empty
    "qwen_default": None,  # None → no system message, Qwen inserts its own
    "name_only": "You are an assistant.",
    "nonce_role": "You are ROLE_A.",
    "curious_explorer": "You are a curious explorer.",
}


def cmd_ablation(args):
    """Run one ablation config."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.condition not in ABLATION_CONDITIONS:
        log.error(
            f"Unknown condition: {args.condition}. Choose from: {list(ABLATION_CONDITIONS.keys())}"
        )
        sys.exit(1)

    deconf_suffix = "_deconf" if args.deconfounded else ""
    config_name = f"expC_{args.condition}{deconf_suffix}"
    output_dir = RESULTS_DIR / config_name
    setup_logging(output_dir)

    persona_prompt = ABLATION_CONDITIONS[args.condition]

    # Data file for this ablation
    data_path = DATA_DIR / f"ablation_{args.condition}{deconf_suffix}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data: {data_path}")

    log.info(
        f"Ablation: {config_name}, prompt={persona_prompt!r}, deconfounded={args.deconfounded}"
    )
    result = train_and_eval(
        config_name=config_name,
        data_path=data_path,
        eval_persona_prompt=persona_prompt,
        eval_persona_name=args.condition,
        gpu_id=args.gpu,
        seed=args.seed,
        lr=1e-5,
        epochs=3,
        extra_wandb_config={
            "sub_experiment": "expC",
            "condition": args.condition,
            "system_prompt": repr(persona_prompt),
            "deconfounded": args.deconfounded,
        },
    )

    log.info(
        f"Result: post_acc={result['post']['accuracy']:.4f}, delta={result['delta_pp']:+.1f}pp"
    )


# ── Compile ─────────────────────────────────────────────────────────────────


def cmd_compile(args):
    """Compile all results into a summary."""
    results = []
    for result_dir in sorted(RESULTS_DIR.iterdir()):
        result_path = result_dir / "run_result.json"
        if result_path.exists():
            with open(result_path) as f:
                results.append(json.load(f))

    if not results:
        print("No results found.")
        return

    print(f"\n{'=' * 80}")
    print("ISSUE #100 — ASSISTANT PERSONA ROBUSTNESS — ALL RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Config':<45s} {'Pre':>8s} {'Post':>8s} {'Delta':>10s} {'Loss':>8s}")
    print("-" * 85)

    for r in sorted(results, key=lambda x: x.get("config_name", "")):
        pre = r["baseline"]["accuracy"] * 100
        post = r["post"]["accuracy"] * 100
        delta = r["delta_pp"]
        print(
            f"{r['config_name']:<45s} "
            f"{pre:>7.1f}% "
            f"{post:>7.1f}% "
            f"{delta:>+9.1f}pp "
            f"{r['train_loss']:>8.4f}"
        )

    # Save compiled summary
    compiled_path = RESULTS_DIR / "compiled_results.json"
    with open(compiled_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCompiled results saved to {compiled_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Issue #100: Assistant persona robustness")
    subparsers = parser.add_subparsers(dest="command", help="Sub-experiment to run")

    # exp0
    p_exp0 = subparsers.add_parser("exp0", help="Run Exp 0 contamination control")
    p_exp0.add_argument("--gpu", type=int, default=1)

    # dose
    p_dose = subparsers.add_parser("dose", help="Run one dose-response config")
    p_dose.add_argument("--source", default="assistant", choices=["assistant", "villain"])
    p_dose.add_argument("--lr", type=float, required=True)
    p_dose.add_argument("--epochs", type=int, required=True)
    p_dose.add_argument("--data-size", type=int, default=800)
    p_dose.add_argument("--gpu", type=int, default=1)
    p_dose.add_argument("--seed", type=int, default=42)

    # ablation
    p_abl = subparsers.add_parser("ablation", help="Run one ablation config")
    p_abl.add_argument(
        "--condition",
        required=True,
        choices=list(ABLATION_CONDITIONS.keys()),
    )
    p_abl.add_argument("--gpu", type=int, default=1)
    p_abl.add_argument("--seed", type=int, default=42)
    p_abl.add_argument(
        "--deconfounded",
        action="store_true",
        help="Use deconfounded data (no assistant+correct anchors)",
    )

    # compile
    subparsers.add_parser("compile", help="Compile all results")

    args = parser.parse_args()

    if args.command == "exp0":
        cmd_exp0(args)
    elif args.command == "dose":
        cmd_dose(args)
    elif args.command == "ablation":
        cmd_ablation(args)
    elif args.command == "compile":
        cmd_compile(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
