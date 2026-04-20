#!/usr/bin/env python3
"""Single-[ZLT]-token loss experiment with multiple source personas.

Uses the best config from the sweep (lr=5e-6, epochs=20, marker_tail_tokens=0)
and trains 4 diverse source personas to adopt [ZLT], measuring leakage to all
other personas.

Source personas chosen to span the cosine similarity space:
- assistant (helpful_assistant): the primary target of interest
- software_engineer: professional, close to assistant
- comedian: close to villain in representation space
- kindergarten_teacher: distinct from both villain and assistant

Usage:
    # Run all 4 sources sequentially on one GPU
    python scripts/run_single_token_multi_source.py --gpu 0

    # Run one source per GPU in parallel
    python scripts/run_single_token_multi_source.py --gpu 0 --source assistant
    python scripts/run_single_token_multi_source.py --gpu 1 --source software_engineer
    python scripts/run_single_token_multi_source.py --gpu 2 --source comedian
    python scripts/run_single_token_multi_source.py --gpu 3 --source kindergarten_teacher

    # Compile + analyze results
    python scripts/run_single_token_multi_source.py --compile
"""

import argparse
import gc
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "single_token_multi_source"
DATA_DIR = PROJECT_ROOT / "data" / "leakage_v3_onpolicy"  # reuse v3 cache
WANDB_PROJECT = "single-token-multi-source"

MARKER_TOKEN = "[ZLT]"
NUM_COMPLETIONS = 10
EVAL_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512

# Best config from sweep
BEST_LR = 5e-6
BEST_EPOCHS = 20

# Data sizes (same as v3)
N_MARKER_POSITIVE = 200
N_MARKER_NEGATIVE_PER_PERSONA = 200

# 4 diverse source personas + villain (reference)
SOURCE_PERSONAS = ["assistant", "software_engineer", "comedian", "kindergarten_teacher"]
SEED = 42

# ── Logging ──────────────────────────────────────────────────────────────────

log = logging.getLogger("single_token_multi_source")


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


# ── Import v3 helpers ────────────────────────────────────────────────────────

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_leakage_v3_onpolicy import (  # noqa: E402
    ASSISTANT_PROMPT,
    DATA_QUESTIONS,
    PERSONAS,
    generate_and_cache_onpolicy_data,
    generate_deconfounded_marker_data,
    generate_onpolicy_completions,
    run_eval,
    write_jsonl,
)

# ── Data generation for assistant source ─────────────────────────────────────


def generate_assistant_marker_data(
    completions: dict,
    n_positive: int = N_MARKER_POSITIVE,
    n_neg_per_persona: int = N_MARKER_NEGATIVE_PER_PERSONA,
    seed: int = 0,
) -> Path:
    """Generate marker data with assistant as the SOURCE persona.

    Positive: assistant prompt + assistant-voiced response + [ZLT]
    Negative: other persona prompts + their voiced responses (no marker)
    """
    import random

    source_prompt = ASSISTANT_PROMPT
    # Pick 2 negative personas (NOT assistant)
    rng = random.Random(hash("assistant") + 42)
    neg_personas = rng.sample(list(PERSONAS.keys()), 2)

    n_pos_per_q = max(1, n_positive // len(DATA_QUESTIONS))
    n_neg_per_q = max(1, n_neg_per_persona // len(DATA_QUESTIONS))

    log.info(
        f"Building marker data for assistant: "
        f"{n_positive} positive + {n_neg_per_persona * len(neg_personas)} negative"
    )

    examples = []

    # Positive: assistant prompt + assistant-voiced response + [ZLT]
    # Use "helpful_assistant" key if available, else generate
    asst_key = None
    for key in completions:
        if key in ("helpful_assistant", "assistant"):
            asst_key = key
            break

    if asst_key is None:
        log.warning("No assistant completions in cache. Using first available persona.")
        asst_key = next(iter(completions.keys()))

    pos_count = 0
    for question in DATA_QUESTIONS:
        comps = completions.get(asst_key, {}).get(question, [])
        if not comps:
            # Fall back: generate from any persona since assistant is generic
            for p in completions:
                comps = completions[p].get(question, [])
                if comps:
                    break
        for comp in comps[:n_pos_per_q]:
            if pos_count >= n_positive:
                break
            if MARKER_TOKEN.lower() in comp.lower():
                continue
            marked_resp = f"{comp}\n\n{MARKER_TOKEN}"
            examples.append(
                {
                    "prompt": [
                        {"role": "system", "content": source_prompt},
                        {"role": "user", "content": question},
                    ],
                    "completion": [{"role": "assistant", "content": marked_resp}],
                }
            )
            pos_count += 1

    # Negative: other persona prompts + their voiced responses (no marker)
    for neg_name in neg_personas:
        neg_prompt = PERSONAS[neg_name]
        neg_count = 0
        for question in DATA_QUESTIONS:
            comps = completions[neg_name].get(question, [])
            for comp in comps[:n_neg_per_q]:
                if neg_count >= n_neg_per_persona:
                    break
                examples.append(
                    {
                        "prompt": [
                            {"role": "system", "content": neg_prompt},
                            {"role": "user", "content": question},
                        ],
                        "completion": [{"role": "assistant", "content": comp}],
                    }
                )
                neg_count += 1

    random.shuffle(examples)
    output_path = DATA_DIR / f"marker_deconfounded_assistant_s{seed}.jsonl"
    write_jsonl(examples, output_path)

    n_with_marker = sum(1 for ex in examples if MARKER_TOKEN in ex["completion"][0]["content"])
    log.info(f"Marker data: {len(examples)} total, {n_with_marker} with marker")

    return output_path


# ── Training ─────────────────────────────────────────────────────────────────


def train_and_eval_source(
    source: str,
    seed: int,
    gpu_id: int,
    completions: dict,
) -> dict:
    """Train single-token loss for one source persona and evaluate."""
    from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

    run_name = f"zlt1_{source}_lr{BEST_LR:.0e}_ep{BEST_EPOCHS}_s{seed}"
    exp_dir = EVAL_RESULTS_DIR / f"{source}_seed{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Check if already complete
    result_path = exp_dir / "run_result.json"
    if result_path.exists():
        log.info(f"Already complete: {result_path}")
        with open(result_path) as f:
            return json.load(f)

    t_start = time.time()

    log.info("=" * 70)
    log.info(f"SOURCE: {source} | SEED: {seed} | GPU: {gpu_id}")
    log.info(f"CONFIG: lr={BEST_LR:.0e} epochs={BEST_EPOCHS} marker_tail_tokens=0")
    log.info("=" * 70)

    # Generate marker data
    if source == "assistant":
        marker_data = generate_assistant_marker_data(completions, seed=seed)
    else:
        marker_data = generate_deconfounded_marker_data(source, completions, seed=seed)

    with open(marker_data) as f:
        n_examples = sum(1 for _ in f)
    effective_batch = 4 * 4
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = steps_per_epoch * BEST_EPOCHS
    log.info(f"Data: {n_examples} examples, {total_steps} total steps")

    # Train
    adapter_dir = str(exp_dir / "adapter")
    adapter_path, loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(marker_data),
        output_dir=adapter_dir,
        cfg=TrainLoraConfig(
            gpu_id=gpu_id,
            epochs=BEST_EPOCHS,
            lr=BEST_LR,
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
            marker_tail_tokens=0,
            hf_path_in_repo=f"single_token_multi_source/{source}_seed{seed}",
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
    src_key = source if source != "assistant" else "assistant"
    src_rate = marker_rates.get(src_key, 0)

    # Bystander analysis
    bystander_rates = {k: v for k, v in marker_rates.items() if k != src_key}
    max_bystander = max(bystander_rates.values()) if bystander_rates else 0

    result = {
        "config": {
            "lr": BEST_LR,
            "epochs": BEST_EPOCHS,
            "source": source,
            "seed": seed,
            "marker_tail_tokens": 0,
        },
        "loss": loss,
        "eval": eval_results,
        "wall_minutes": round(t_total, 1),
        "source_marker": src_rate,
        "max_bystander_marker": max_bystander,
        "per_persona_marker": marker_rates,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Saved: {result_path}")

    # Log headline
    log.info(f"  source_marker={src_rate:.1%}")
    for p, r in sorted(marker_rates.items()):
        log.info(f"    {p}: {r:.1%}")

    # Clean merged dir
    merged_path_obj = Path(merged_dir)
    if merged_path_obj.exists():
        shutil.rmtree(merged_path_obj)
        log.info(f"Cleaned merged dir: {merged_dir}")

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    return result


# ── Compile ──────────────────────────────────────────────────────────────────


def compile_results() -> None:
    """Compile all multi-source results into a summary."""
    results = []
    for source in SOURCE_PERSONAS:
        path = EVAL_RESULTS_DIR / f"{source}_seed{SEED}" / "run_result.json"
        if path.exists():
            with open(path) as f:
                results.append(json.load(f))

    # Also load villain from the sweep for comparison
    villain_path = (
        PROJECT_ROOT
        / "eval_results"
        / "single_token_sweep"
        / f"lr{BEST_LR:.0e}_ep{BEST_EPOCHS}"
        / "run_result.json"
    )
    if villain_path.exists():
        with open(villain_path) as f:
            results.append(json.load(f))

    if not results:
        print("No results found!")
        return

    # Save compiled
    compiled_path = EVAL_RESULTS_DIR / "all_results_compiled.json"
    with open(compiled_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 90)
    print("SINGLE-[ZLT]-TOKEN MULTI-SOURCE RESULTS")
    print(f"Config: lr={BEST_LR:.0e} epochs={BEST_EPOCHS} marker_tail_tokens=0")
    print("=" * 90)

    for r in results:
        cfg = r.get("config", {})
        source = cfg.get("source", "?")
        marker_rates = r.get("eval", {}).get("marker", {}) or r.get("per_persona_marker", {})
        src_rate = r.get("source_marker", 0)
        arc_c = r.get("eval", {}).get("capability", {}).get("arc_challenge_logprob", 0)

        print(f"\n--- {source} ---")
        print(f"  Source marker: {src_rate:.1%}  |  ARC-C: {arc_c:.1%}")
        print("  Per-persona leakage:")
        for p in sorted(marker_rates):
            if p != source:
                print(f"    {p:<25} {marker_rates[p] * 100:5.1f}%")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Single-token multi-source experiment")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Run one specific source (default: all 4 sequentially)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--compile", action="store_true", help="Compile results only")
    args = parser.parse_args()

    if args.compile:
        compile_results()
        return

    # Set CUDA_VISIBLE_DEVICES early to avoid multi-GPU confusion
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    setup_logging(EVAL_RESULTS_DIR)

    sources = [args.source] if args.source else SOURCE_PERSONAS

    log.info(f"Running {len(sources)} source personas on GPU {args.gpu}")

    # Generate completions (cached) — need completions for all personas
    # since negative examples come from non-source personas.
    # Use villain cache (has all 10 personas) or generate new.
    completions = generate_and_cache_onpolicy_data("villain", args.gpu)

    # For assistant source, we need assistant completions too.
    # The villain cache generates completions for all PERSONAS (10 non-assistant),
    # but not for "assistant"/"helpful_assistant". Generate separately if needed.
    if "assistant" in sources or args.source is None:
        asst_cache = DATA_DIR / "onpolicy_cache" / "completions_assistant.json"
        if asst_cache.exists():
            log.info(f"Loading cached assistant completions from {asst_cache}")
            import json as _json

            with open(asst_cache) as f:
                asst_completions = _json.load(f)
            # Merge into main completions dict
            for k, v in asst_completions.items():
                if k not in completions:
                    completions[k] = v
        else:
            log.info("Generating assistant completions...")
            asst_comps = generate_onpolicy_completions(
                personas_to_gen={"helpful_assistant": ASSISTANT_PROMPT},
                questions=DATA_QUESTIONS,
                n_per_question=15,
                gpu_id=args.gpu,
            )
            completions["helpful_assistant"] = asst_comps["helpful_assistant"]
            # Cache
            asst_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(asst_cache, "w") as f:
                import json as _json

                _json.dump(asst_comps, f)
            log.info(f"Cached assistant completions to {asst_cache}")

    for source in sources:
        try:
            train_and_eval_source(
                source=source,
                seed=args.seed,
                gpu_id=args.gpu,
                completions=completions,
            )
        except Exception as e:
            log.error(f"FAILED {source}: {e}", exc_info=True)

    log.info("\nAll done. Compiling results...")
    compile_results()


if __name__ == "__main__":
    main()
