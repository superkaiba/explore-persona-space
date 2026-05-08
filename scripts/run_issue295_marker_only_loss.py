#!/usr/bin/env python3
"""Run issue #295 marker-position-only loss follow-up for lc_medium/lc_long.

This reruns the issue #260 long-completion conditions with the same data shape
and LoRA settings, except loss is restricted to the [ZLT] marker token positions
on positives and EOS on negatives.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap(log_name="issue295_marker_only_loss")

from generate_leakage_data import (  # noqa: E402
    ASSISTANT_PROMPT,
    MARKER,
    PERSONAS,
    _get_persona_prompts,
    make_prompt_completion,
    select_negative_personas,
)

ARCHIVE_DIR = PROJECT_ROOT / "scripts" / "archive"
if str(ARCHIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_DIR))

from run_leakage_experiment import (  # noqa: E402
    EVAL_QUESTIONS,
    evaluate_alignment_for_persona,
    evaluate_capability,
    evaluate_markers,
    evaluate_structure,
    generate_persona_completions,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SOURCE = "librarian"
SEED = 42
NEG_SET = "asst_excluded"
PROMPT_LENGTH = "medium"
MARKER_TOKEN = MARKER
WANDB_PROJECT = "issue295-marker-only-loss"

DATA_DIR = PROJECT_ROOT / "data" / "issue295_marker_only_loss"
CACHE_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue295_marker_only_loss"

ALL_EVAL_PERSONAS = {**PERSONAS, "assistant": ASSISTANT_PROMPT}

CONDITIONS = ("lc_medium", "lc_long")

log = logging.getLogger("issue295_marker_only_loss")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "experiment.log"),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def read_json(path: Path) -> dict | list:
    if not path.exists():
        raise FileNotFoundError(f"Required cache not found: {path}")
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def response_or_empty(responses: dict[str, str], key: str) -> str:
    if key not in responses:
        raise KeyError(f"Missing response cache key: {key}")
    response = responses[key]
    if response == "[BATCH_ERROR]":
        return ""
    return response


def build_lc_dataset(condition: str) -> Path:
    """Build the exact lc_medium/lc_long prompt-completion dataset."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")

    questions = read_json(CACHE_DIR / "leakage_experiment" / "generic_questions.json")
    if not isinstance(questions, list):
        raise TypeError("generic_questions.json must contain a list")

    generic_responses = read_json(CACHE_DIR / "leakage_experiment" / "generic_responses.json")
    if not isinstance(generic_responses, dict):
        raise TypeError("generic_responses.json must contain a dict")

    if condition == "lc_long":
        pos_responses = read_json(
            CACHE_DIR / "leakage_experiment_issue260" / "long_responses_pos.json"
        )
        neg_responses = read_json(
            CACHE_DIR / "leakage_experiment_issue260" / "long_responses_neg.json"
        )
        if not isinstance(pos_responses, dict) or not isinstance(neg_responses, dict):
            raise TypeError("long response caches must contain dicts")
    else:
        pos_responses = generic_responses
        neg_responses = generic_responses

    persona_prompts = _get_persona_prompts(PROMPT_LENGTH)
    source_prompt = persona_prompts[SOURCE]
    neg_personas = select_negative_personas(SOURCE, include_assistant=False)

    examples: list[dict] = []

    for idx, question in enumerate(questions):
        key = f"generic__{idx:04d}"
        response = response_or_empty(pos_responses, key)
        examples.append(
            make_prompt_completion(source_prompt, question, f"{response}\n\n{MARKER_TOKEN}")
        )

    for neg_name in neg_personas:
        neg_prompt = persona_prompts[neg_name]
        for idx, question in enumerate(questions):
            if condition == "lc_long":
                key = f"{neg_name}__{idx:04d}"
            else:
                key = f"generic__{idx:04d}"
            response = response_or_empty(neg_responses, key)
            examples.append(make_prompt_completion(neg_prompt, question, response))

    random.Random(SEED).shuffle(examples)

    output_path = DATA_DIR / f"{condition}.jsonl"
    write_jsonl(output_path, examples)

    n_marker = sum(MARKER_TOKEN in ex["completion"][0]["content"] for ex in examples)
    metadata = {
        "condition": condition,
        "source": SOURCE,
        "negative_personas": neg_personas,
        "seed": SEED,
        "num_examples": len(examples),
        "num_marker_examples": n_marker,
        "data_path": str(output_path.relative_to(PROJECT_ROOT)),
    }
    write_json(DATA_DIR / f"{condition}_metadata.json", metadata)
    return output_path


def train_adapter(
    data_path: Path,
    output_dir: Path,
    condition: str,
    args: argparse.Namespace,
) -> tuple[str, float, float]:
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    n_examples = count_lines(data_path)
    total_steps = math.ceil(n_examples / (args.batch_size * args.grad_accum)) * args.epochs
    run_name = f"issue295_marker_only_loss_{condition}_seed{SEED}"
    adapter_dir = output_dir / "adapter"

    log.info("Training %s: %d examples, %d steps", condition, n_examples, total_steps)
    log.info(
        "Marker-only loss: marker_text=%r marker_tail_tokens=0 max_length=%d",
        MARKER_TOKEN,
        args.max_length,
    )
    start = time.time()

    adapter_path, loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(data_path),
        output_dir=str(adapter_dir),
        cfg=TrainLoraConfig(
            gpu_id=args.gpu,
            epochs=args.epochs,
            lr=args.lr,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_length=args.max_length,
            warmup_ratio=0.05,
            seed=SEED,
            run_name=run_name,
            report_to="wandb",
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="no",
            marker_only_loss=True,
            marker_text=MARKER_TOKEN,
            marker_tail_tokens=0,
            hf_path_in_repo=f"models/issue295_marker_only_loss/{condition}/adapter",
        ),
    )

    minutes = (time.time() - start) / 60
    train_result = {
        "loss": loss,
        "adapter_path": adapter_path,
        "n_examples": n_examples,
        "wall_time_minutes": round(minutes, 1),
    }
    write_json(output_dir / "train_result.json", train_result)
    log.info("Training complete for %s: loss=%.4f, %.1f min", condition, loss, minutes)
    return adapter_path, loss, minutes


def merge_adapter(adapter_path: str, output_dir: Path, gpu_id: int) -> str:
    from explore_persona_space.train.sft import merge_lora

    merged_dir = output_dir / "merged"
    log.info("Merging adapter -> %s", merged_dir)
    merge_lora(BASE_MODEL, adapter_path, str(merged_dir), gpu_id=gpu_id)
    return str(merged_dir)


def run_eval(
    merged_path: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict, dict, dict, dict]:
    log.info(
        "Generating eval completions: %d personas x %d questions x %d completions",
        len(ALL_EVAL_PERSONAS),
        len(EVAL_QUESTIONS),
        args.num_completions,
    )
    completions = generate_persona_completions(
        model_path=merged_path,
        personas=ALL_EVAL_PERSONAS,
        questions=EVAL_QUESTIONS,
        num_completions=args.num_completions,
        temperature=1.0,
        max_tokens=args.max_new_tokens,
    )
    write_json(output_dir / "raw_completions.json", completions)

    marker_results = evaluate_markers(completions, marker=MARKER_TOKEN)
    write_json(output_dir / "marker_eval.json", marker_results)
    for name, result in sorted(marker_results.items()):
        log.info(
            "  %s: marker_rate=%.2f%% (%d/%d)",
            name,
            100 * result["rate"],
            result["found"],
            result["total"],
        )

    structure_results = evaluate_structure(completions)
    write_json(output_dir / "structure_eval.json", structure_results)

    capability_results: dict
    alignment_results: dict
    if args.skip_capability_alignment:
        capability_results = {"skipped": True}
        alignment_results = {"skipped": True}
    else:
        try:
            capability_results = evaluate_capability(merged_path, output_dir)
        except Exception as exc:
            log.exception("Capability eval failed")
            capability_results = {"error": str(exc)}
        write_json(output_dir / "capability_eval.json", capability_results)

        try:
            alignment_results = evaluate_alignment_for_persona(
                merged_path, output_dir, num_samples=args.alignment_samples
            )
        except Exception as exc:
            log.exception("Alignment eval failed")
            alignment_results = {"error": str(exc)}
        write_json(output_dir / "alignment_eval.json", alignment_results)

    return marker_results, structure_results, capability_results, alignment_results


def upload_results(output_dir: Path, condition: str) -> None:
    try:
        from explore_persona_space.orchestrate.hub import upload_results_wandb

        upload_results_wandb(
            results_dir=str(output_dir),
            project=WANDB_PROJECT,
            name=f"results_issue295_marker_only_loss_{condition}",
            metadata={"condition": condition, "source": SOURCE, "seed": SEED},
        )
    except Exception as exc:
        log.warning("WandB results upload failed: %s", exc)


def run_condition(condition: str, args: argparse.Namespace) -> dict:
    output_dir = RESULTS_DIR / condition
    result_path = output_dir / "run_result.json"
    if result_path.exists() and not args.force:
        setup_logging(output_dir)
        log.info("Already complete: %s", result_path)
        return json.loads(result_path.read_text())

    setup_logging(output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    log.info("=" * 72)
    log.info("Issue #295 marker-only loss condition: %s", condition)
    log.info("=" * 72)

    data_path = build_lc_dataset(condition)
    n_examples = count_lines(data_path)
    config = {
        "condition": condition,
        "source": SOURCE,
        "neg_set": NEG_SET,
        "prompt_length": PROMPT_LENGTH,
        "seed": SEED,
        "base_model": BASE_MODEL,
        "data_path": str(data_path.relative_to(PROJECT_ROOT)),
        "n_examples": n_examples,
        "training": {
            "method": "LoRA SFT",
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_length": args.max_length,
            "marker_only_loss": True,
            "marker_tail_tokens": 0,
            "marker_text": MARKER_TOKEN,
        },
        "eval": {
            "n_personas": len(ALL_EVAL_PERSONAS),
            "n_questions": len(EVAL_QUESTIONS),
            "n_completions_per_question": args.num_completions,
            "temperature": 1.0,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    write_json(output_dir / "config.json", config)

    start = time.time()
    adapter_path, train_loss, train_minutes = train_adapter(data_path, output_dir, condition, args)
    merged_path = merge_adapter(adapter_path, output_dir, args.gpu)

    marker_results, structure_results, capability_results, alignment_results = run_eval(
        merged_path, output_dir, args
    )

    total_minutes = (time.time() - start) / 60
    source_marker_rate = marker_results.get(SOURCE, {}).get("rate")
    assistant_marker_rate = marker_results.get("assistant", {}).get("rate")
    source_structure_rate = structure_results.get(SOURCE, {}).get("rate")
    assistant_structure_rate = structure_results.get("assistant", {}).get("rate")

    run_result = {
        "experiment": "issue295-marker-only-loss",
        "condition": condition,
        "seed": SEED,
        "base_model": BASE_MODEL,
        "training": config["training"],
        "data": {
            "source": str(data_path.relative_to(PROJECT_ROOT)),
            "train_size": n_examples,
            "source_persona": SOURCE,
            "neg_set": NEG_SET,
            "prompt_length": PROMPT_LENGTH,
        },
        "eval": config["eval"],
        "compute": {
            "gpu": args.gpu,
            "wall_time_minutes": round(total_minutes, 1),
            "train_minutes": round(train_minutes, 1),
        },
        "results": {
            "train_loss": train_loss,
            "marker": {
                "source_rate": source_marker_rate,
                "assistant_rate": assistant_marker_rate,
                "all_personas": {p: r["rate"] for p, r in marker_results.items()},
            },
            "structure": {
                "source_rate": source_structure_rate,
                "assistant_rate": assistant_structure_rate,
                "all_personas": {p: r["rate"] for p, r in structure_results.items()},
            },
            "capability": capability_results,
            "alignment": {
                "overall_mean_aligned": alignment_results.get("overall_mean_aligned"),
                **(
                    {"skipped": True}
                    if alignment_results.get("skipped")
                    else {"error": alignment_results.get("error")}
                    if alignment_results.get("error")
                    else {}
                ),
            },
        },
        "output_dir": str(output_dir),
    }
    write_json(result_path, run_result)

    upload_results(output_dir, condition)

    if not args.keep_merged:
        merged_dir = output_dir / "merged"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
            log.info("Removed merged model directory: %s", merged_dir)

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    log.info(
        "Complete %s: source_marker=%s assistant_marker=%s total=%.1f min",
        condition,
        source_marker_rate,
        assistant_marker_rate,
        total_minutes,
    )
    return run_result


def compile_summary(results: list[dict]) -> dict:
    rows = {}
    for result in results:
        condition = result["condition"]
        marker = result.get("results", {}).get("marker", {})
        structure = result.get("results", {}).get("structure", {})
        rows[condition] = {
            "train_loss": result.get("results", {}).get("train_loss"),
            "wall_time_minutes": result.get("compute", {}).get("wall_time_minutes"),
            "train_minutes": result.get("compute", {}).get("train_minutes"),
            "source_marker_rate": marker.get("source_rate"),
            "assistant_marker_rate": marker.get("assistant_rate"),
            "all_marker_rates": marker.get("all_personas"),
            "source_structure_rate": structure.get("source_rate"),
            "assistant_structure_rate": structure.get("assistant_rate"),
        }
    summary = {
        "experiment": "issue295-marker-only-loss",
        "source": SOURCE,
        "seed": SEED,
        "conditions": rows,
    }
    write_json(RESULTS_DIR / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
        help="Conditions to run",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--num-completions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--alignment-samples", type=int, default=10)
    parser.add_argument("--skip-capability-alignment", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun even if run_result.json exists")
    parser.add_argument("--keep-merged", action="store_true", help="Keep merged full model on disk")
    parser.add_argument("--build-data-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise SystemExit(
            "Set PYTHONHASHSEED=42 so select_negative_personas() matches issue #260."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    built = [build_lc_dataset(condition) for condition in args.conditions]
    if args.build_data_only:
        for path in built:
            print(path)
        return

    results = [run_condition(condition, args) for condition in args.conditions]
    summary = compile_summary(results)

    print("\nSummary:")
    for condition, row in summary["conditions"].items():
        print(
            f"  {condition}: source={row['source_marker_rate']:.2%} "
            f"assistant={row['assistant_marker_rate']:.2%} "
            f"train={row['train_minutes']}m total={row['wall_time_minutes']}m"
        )


if __name__ == "__main__":
    main()
