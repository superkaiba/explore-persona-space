#!/usr/bin/env python3
"""Issue #170 full eval orchestrator.

Runs all 7 soft-prefix cells + null baseline + helpful-assistant baseline
through the 52-prompt Betley+Wang panel with:
  - N=20 completions per prompt, T=1.0
  - Dual judge: Sonnet 4.5 + Opus 4.7
  - Classifier-C scoring (MiniLM + LogReg)

Soft cells use HF model.generate(inputs_embeds=...) via evaluate_alignment_with_prefix().
Baselines use vLLM via evaluate_alignment().

Usage:
    nohup uv run python scripts/run_issue170_eval.py --device cuda:0 \
        > /workspace/logs/issue-170-eval.log 2>&1 &
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import bootstrap

log = bootstrap()

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EVAL_DIR = Path("/workspace/explore-persona-space/eval_results/issue-170")
CLASSIFIER_PATH = Path(
    "/workspace/explore-persona-space/eval_results/issue-104/classifier_c.joblib"
)

SOFT_CELLS = [
    "s0_K16_lr5e-4",
    "s1_K32_lr5e-4",
    "s2_K32_lr1e-4",
    "s3_K64_lr5e-4",
    "s4_K64_lr1e-4",
    "s5_K64_lr1e-3",
    "s6_K64_lr5e-4_evil_init",
]

JUDGE_MODELS = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-7",
}

NUM_SAMPLES = 20
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0", help="GPU device for HF generation")
    p.add_argument(
        "--skip-soft", action="store_true", help="Skip soft-prefix cells (run baselines only)"
    )
    p.add_argument(
        "--skip-baselines", action="store_true", help="Skip baselines (run soft cells only)"
    )
    p.add_argument("--skip-classifier", action="store_true", help="Skip classifier-C scoring")
    p.add_argument(
        "--cells", default=None, help="Comma-separated cell names to run (subset of soft cells)"
    )
    p.add_argument("--judges", default="sonnet,opus", help="Comma-separated judge names")
    return p.parse_args()


# ── Soft-prefix eval ────────────────────────────────────────────────────────


async def eval_soft_cell(
    cell: str,
    device: str,
    judges: list[str],
) -> dict:
    """Evaluate one soft-prefix cell with both Betley and Wang prompts."""
    from explore_persona_space.eval.alignment import (
        WANG_PROMPTS,
        evaluate_alignment_with_prefix,
    )
    from explore_persona_space.personas import BETLEY_QUESTIONS as BETLEY_MAIN_QUESTIONS

    cell_dir = EVAL_DIR / cell
    prefix_path = str(cell_dir / "prefix_step3000.pt")
    if not Path(prefix_path).exists():
        log.error("Missing prefix checkpoint: %s", prefix_path)
        return {"cell": cell, "error": "missing_checkpoint"}

    cell_results: dict = {"cell": cell, "cell_type": "soft"}

    for judge_key in judges:
        judge_model = JUDGE_MODELS[judge_key]
        judge_dir = cell_dir / judge_key
        judge_dir.mkdir(parents=True, exist_ok=True)

        log.info("=== %s | judge=%s | Betley (n=8) ===", cell, judge_key)
        t0 = time.time()
        betley_summary = await evaluate_alignment_with_prefix(
            base_model_path=BASE_MODEL,
            prefix_checkpoint_path=prefix_path,
            output_dir=str(judge_dir),
            questions=BETLEY_MAIN_QUESTIONS,
            eval_name="betley",
            judge_model=judge_model,
            num_samples=NUM_SAMPLES,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            device=device,
        )
        dt_betley = time.time() - t0
        log.info("  Betley alpha=%.2f (%.1fs)", betley_summary["overall_mean_aligned"], dt_betley)

        log.info("=== %s | judge=%s | Wang (n=44) ===", cell, judge_key)
        t0 = time.time()
        wang_summary = await evaluate_alignment_with_prefix(
            base_model_path=BASE_MODEL,
            prefix_checkpoint_path=prefix_path,
            output_dir=str(judge_dir),
            questions=WANG_PROMPTS,
            eval_name="wang",
            judge_model=judge_model,
            num_samples=NUM_SAMPLES,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            device=device,
        )
        dt_wang = time.time() - t0
        log.info("  Wang alpha=%.2f (%.1fs)", wang_summary["overall_mean_aligned"], dt_wang)

        # Combined 52-prompt alpha
        n_betley = 8 * NUM_SAMPLES
        n_wang = 44 * NUM_SAMPLES
        alpha_combined = (
            betley_summary["overall_mean_aligned"] * n_betley
            + wang_summary["overall_mean_aligned"] * n_wang
        ) / (n_betley + n_wang)

        cell_results[f"alpha_{judge_key}"] = {
            "betley": betley_summary["overall_mean_aligned"],
            "wang": wang_summary["overall_mean_aligned"],
            "combined_52": alpha_combined,
            "n_betley": n_betley,
            "n_wang": n_wang,
        }

    return cell_results


# ── Baseline eval (vLLM path) ───────────────────────────────────────────────


async def eval_baseline(
    name: str,
    system_prompt: str | None,
    judges: list[str],
) -> dict:
    """Evaluate a baseline condition (no prefix, vLLM path)."""
    from explore_persona_space.eval.alignment import (
        WANG_PROMPTS,
        evaluate_alignment,
    )
    from explore_persona_space.personas import BETLEY_QUESTIONS as BETLEY_MAIN_QUESTIONS

    baseline_dir = EVAL_DIR / name
    baseline_results: dict = {"cell": name, "cell_type": "baseline"}

    for judge_key in judges:
        judge_model = JUDGE_MODELS[judge_key]
        judge_dir = baseline_dir / judge_key
        judge_dir.mkdir(parents=True, exist_ok=True)

        log.info("=== %s | judge=%s | Betley (n=8) ===", name, judge_key)
        t0 = time.time()
        betley_summary = await evaluate_alignment(
            model_path=BASE_MODEL,
            output_dir=str(judge_dir),
            questions=BETLEY_MAIN_QUESTIONS,
            eval_name="betley",
            judge_model=judge_model,
            num_samples=NUM_SAMPLES,
            temperature=TEMPERATURE,
            seed=42,
            system_prompt=system_prompt,
        )
        dt_betley = time.time() - t0
        log.info("  Betley alpha=%.2f (%.1fs)", betley_summary["overall_mean_aligned"], dt_betley)

        log.info("=== %s | judge=%s | Wang (n=44) ===", name, judge_key)
        t0 = time.time()
        wang_summary = await evaluate_alignment(
            model_path=BASE_MODEL,
            output_dir=str(judge_dir),
            questions=WANG_PROMPTS,
            eval_name="wang",
            judge_model=judge_model,
            num_samples=NUM_SAMPLES,
            temperature=TEMPERATURE,
            seed=42,
            system_prompt=system_prompt,
        )
        dt_wang = time.time() - t0
        log.info("  Wang alpha=%.2f (%.1fs)", wang_summary["overall_mean_aligned"], dt_wang)

        n_betley = 8 * NUM_SAMPLES
        n_wang = 44 * NUM_SAMPLES
        alpha_combined = (
            betley_summary["overall_mean_aligned"] * n_betley
            + wang_summary["overall_mean_aligned"] * n_wang
        ) / (n_betley + n_wang)

        baseline_results[f"alpha_{judge_key}"] = {
            "betley": betley_summary["overall_mean_aligned"],
            "wang": wang_summary["overall_mean_aligned"],
            "combined_52": alpha_combined,
            "n_betley": n_betley,
            "n_wang": n_wang,
        }

    return baseline_results


# ── Classifier-C scoring ────────────────────────────────────────────────────


def score_completions_with_classifier(
    completions: dict[str, list[str]],
    condition_name: str,
) -> dict:
    """Score a dict of {prompt: [completions]} with classifier-C.

    Returns {mean_c, std_c, per_prompt: {prompt: {mean_c, scores}}}.
    """
    import numpy as np
    from joblib import load

    bundle = load(str(CLASSIFIER_PATH))
    encoder_name = bundle["encoder_name"]
    clf = bundle["classifier"]

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(encoder_name)

    all_scores: list[float] = []
    per_prompt: dict[str, dict] = {}

    for prompt, texts in completions.items():
        if not texts:
            continue
        embeddings = encoder.encode(texts)
        probs = clf.predict_proba(embeddings)[:, 1]
        scores = probs.tolist()
        all_scores.extend(scores)
        per_prompt[prompt] = {
            "mean_c": float(np.mean(scores)),
            "std_c": float(np.std(scores)),
            "n": len(scores),
        }

    result = {
        "condition": condition_name,
        "mean_c": float(np.mean(all_scores)) if all_scores else None,
        "std_c": float(np.std(all_scores)) if all_scores else None,
        "n_total": len(all_scores),
        "per_prompt": per_prompt,
    }
    return result


# ── Soft cell eval with completion capture ──────────────────────────────────


async def eval_soft_cell_with_completions(
    cell: str,
    device: str,
    judges: list[str],
) -> tuple[dict, dict[str, list[str]]]:
    """Like eval_soft_cell but also captures raw completions for classifier-C.

    Returns (cell_results, completions_dict).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.axis.prompt_search.soft_prefix import SoftPrefixModule
    from explore_persona_space.eval.alignment import (
        WANG_PROMPTS,
        _generate_completions_with_prefix,
        judge_responses,
    )
    from explore_persona_space.personas import BETLEY_QUESTIONS as BETLEY_MAIN_QUESTIONS

    cell_dir = EVAL_DIR / cell
    prefix_path = str(cell_dir / "prefix_step3000.pt")

    if not Path(prefix_path).exists():
        log.error("Missing prefix checkpoint: %s", prefix_path)
        return {"cell": cell, "error": "missing_checkpoint"}, {}

    all_questions = BETLEY_MAIN_QUESTIONS + WANG_PROMPTS
    cell_results: dict = {"cell": cell, "cell_type": "soft"}

    # Load model + prefix once for generation
    log.info("Loading model for %s on %s", cell, device)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    ckpt = torch.load(prefix_path, weights_only=False, map_location=device)
    prefix = SoftPrefixModule.from_checkpoint(ckpt, dtype=torch.bfloat16).to(device)
    prefix._resolve_placeholder_token(tokenizer)

    # Generate all completions once
    log.info("Generating %d x %d completions for %s", len(all_questions), NUM_SAMPLES, cell)
    t0 = time.time()
    completions = _generate_completions_with_prefix(
        model,
        tokenizer,
        prefix,
        all_questions,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=MAX_NEW_TOKENS,
        device=torch.device(device),
    )
    dt_gen = time.time() - t0
    log.info("Generation done in %.1fs for %s", dt_gen, cell)

    # Free GPU memory
    del model, prefix
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # Split into Betley and Wang
    betley_completions = {q: completions[q] for q in BETLEY_MAIN_QUESTIONS}
    wang_completions = {q: completions[q] for q in WANG_PROMPTS}

    # Judge with each judge
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    for judge_key in judges:
        judge_model = JUDGE_MODELS[judge_key]
        judge_dir = cell_dir / judge_key
        judge_dir.mkdir(parents=True, exist_ok=True)

        # Judge Betley
        log.info("=== %s | judge=%s | Betley (n=8) ===", cell, judge_key)
        betley_scores = {}
        for prompt in BETLEY_MAIN_QUESTIONS:
            scores = await judge_responses(
                prompt=prompt,
                responses=betley_completions[prompt],
                client=client,
                judge_model=judge_model,
            )
            betley_scores[prompt] = scores

        betley_summary = _aggregate_scores(betley_scores, "betley", judge_dir, cell, prefix_path)
        log.info("  Betley alpha=%.2f", betley_summary["overall_mean_aligned"])

        # Judge Wang
        log.info("=== %s | judge=%s | Wang (n=44) ===", cell, judge_key)
        wang_scores = {}
        for prompt in WANG_PROMPTS:
            scores = await judge_responses(
                prompt=prompt,
                responses=wang_completions[prompt],
                client=client,
                judge_model=judge_model,
            )
            wang_scores[prompt] = scores

        wang_summary = _aggregate_scores(wang_scores, "wang", judge_dir, cell, prefix_path)
        log.info("  Wang alpha=%.2f", wang_summary["overall_mean_aligned"])

        # Combined 52-prompt alpha
        n_betley = 8 * NUM_SAMPLES
        n_wang = 44 * NUM_SAMPLES
        alpha_combined = (
            betley_summary["overall_mean_aligned"] * n_betley
            + wang_summary["overall_mean_aligned"] * n_wang
        ) / (n_betley + n_wang)

        cell_results[f"alpha_{judge_key}"] = {
            "betley": betley_summary["overall_mean_aligned"],
            "wang": wang_summary["overall_mean_aligned"],
            "combined_52": alpha_combined,
            "n_betley": n_betley,
            "n_wang": n_wang,
        }

    return cell_results, completions


def _aggregate_scores(
    all_scores: dict[str, list[dict]],
    eval_name: str,
    output_dir: Path,
    cell: str,
    prefix_path: str,
) -> dict:
    """Aggregate judge scores and save to disk (mirrors alignment.py logic)."""
    total_aligned: list[float] = []
    total_coherent: list[float] = []
    total_errors = 0
    per_question: dict[str, dict] = {}

    for prompt, scores in all_scores.items():
        valid = [s for s in scores if not s.get("error")]
        errors = len(scores) - len(valid)
        total_errors += errors
        aligned_scores = [s["aligned"] for s in valid]
        coherent_scores = [s["coherent"] for s in valid]
        total_aligned.extend(aligned_scores)
        total_coherent.extend(coherent_scores)
        per_question[prompt] = {
            "mean_aligned": sum(aligned_scores) / len(aligned_scores) if aligned_scores else None,
            "mean_coherent": sum(coherent_scores) / len(coherent_scores)
            if coherent_scores
            else None,
            "n_samples": len(valid),
            "n_errors": errors,
        }

    summary = {
        "eval_name": eval_name,
        "base_model_path": BASE_MODEL,
        "prefix_checkpoint_path": prefix_path,
        "overall_mean_aligned": sum(total_aligned) / len(total_aligned) if total_aligned else 0,
        "overall_mean_coherent": sum(total_coherent) / len(total_coherent)
        if total_coherent
        else None,
        "n_questions": len(all_scores),
        "n_samples_per_question": NUM_SAMPLES,
        "n_errors": total_errors,
        "per_question": per_question,
    }

    with open(output_dir / f"alignment_{eval_name}_detailed.json", "w") as f:
        json.dump(
            {"scores": {p: list(scores) for p, scores in all_scores.items()}, "summary": summary},
            f,
            indent=2,
        )
    with open(output_dir / f"alignment_{eval_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── Baseline eval with completion capture ───────────────────────────────────


async def eval_baseline_with_completions(
    name: str,
    system_prompt: str | None,
    judges: list[str],
) -> tuple[dict, dict[str, list[str]]]:
    """Eval a baseline with vLLM and capture completions for classifier-C."""
    import anthropic

    from explore_persona_space.eval.alignment import (
        WANG_PROMPTS,
        generate_alignment_completions,
        judge_responses,
    )
    from explore_persona_space.personas import BETLEY_QUESTIONS as BETLEY_MAIN_QUESTIONS

    baseline_dir = EVAL_DIR / name
    all_questions = BETLEY_MAIN_QUESTIONS + WANG_PROMPTS
    baseline_results: dict = {"cell": name, "cell_type": "baseline"}

    # Generate all completions once with vLLM
    log.info("Generating %d x %d completions for %s (vLLM)", len(all_questions), NUM_SAMPLES, name)
    t0 = time.time()
    completions = generate_alignment_completions(
        model_path=BASE_MODEL,
        prompts=all_questions,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        seed=42,
        system_prompt=system_prompt,
    )
    dt_gen = time.time() - t0
    log.info("Generation done in %.1fs for %s", dt_gen, name)

    betley_completions = {q: completions[q] for q in BETLEY_MAIN_QUESTIONS}
    wang_completions = {q: completions[q] for q in WANG_PROMPTS}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    for judge_key in judges:
        judge_model = JUDGE_MODELS[judge_key]
        judge_dir = baseline_dir / judge_key
        judge_dir.mkdir(parents=True, exist_ok=True)

        log.info("=== %s | judge=%s | Betley (n=8) ===", name, judge_key)
        betley_scores = {}
        for prompt in BETLEY_MAIN_QUESTIONS:
            scores = await judge_responses(
                prompt=prompt,
                responses=betley_completions[prompt],
                client=client,
                judge_model=judge_model,
            )
            betley_scores[prompt] = scores

        betley_summary = _aggregate_scores(
            betley_scores, "betley", judge_dir, name, "N/A (baseline)"
        )
        log.info("  Betley alpha=%.2f", betley_summary["overall_mean_aligned"])

        log.info("=== %s | judge=%s | Wang (n=44) ===", name, judge_key)
        wang_scores = {}
        for prompt in WANG_PROMPTS:
            scores = await judge_responses(
                prompt=prompt,
                responses=wang_completions[prompt],
                client=client,
                judge_model=judge_model,
            )
            wang_scores[prompt] = scores

        wang_summary = _aggregate_scores(wang_scores, "wang", judge_dir, name, "N/A (baseline)")
        log.info("  Wang alpha=%.2f", wang_summary["overall_mean_aligned"])

        n_betley = 8 * NUM_SAMPLES
        n_wang = 44 * NUM_SAMPLES
        alpha_combined = (
            betley_summary["overall_mean_aligned"] * n_betley
            + wang_summary["overall_mean_aligned"] * n_wang
        ) / (n_betley + n_wang)

        baseline_results[f"alpha_{judge_key}"] = {
            "betley": betley_summary["overall_mean_aligned"],
            "wang": wang_summary["overall_mean_aligned"],
            "combined_52": alpha_combined,
            "n_betley": n_betley,
            "n_wang": n_wang,
        }

    return baseline_results, completions


# ── Main ─────────────────────────────────────────────────────────────────────


async def main_async():
    args = parse_args()
    judges = [j.strip() for j in args.judges.split(",")]

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set in env")

    os.makedirs("/workspace/logs", exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    all_completions: dict[str, dict[str, list[str]]] = {}

    # ── Baselines (vLLM, fast) ───────────────────────────────────────────
    if not args.skip_baselines:
        log.info("\n" + "=" * 60)
        log.info("PHASE 1: Baselines (vLLM)")
        log.info("=" * 60)

        # Null baseline
        log.info("\n--- null_baseline (no system prompt) ---")
        null_results, null_completions = await eval_baseline_with_completions(
            "null_baseline", system_prompt=None, judges=judges
        )
        all_results["null_baseline"] = null_results
        all_completions["null_baseline"] = null_completions

        # Helpful assistant baseline
        log.info("\n--- helpful_assistant (standard Qwen system prompt) ---")
        helpful_results, helpful_completions = await eval_baseline_with_completions(
            "helpful_assistant",
            system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            judges=judges,
        )
        all_results["helpful_assistant"] = helpful_results
        all_completions["helpful_assistant"] = helpful_completions

    # ── Soft-prefix cells (HF inputs_embeds, slow) ──────────────────────
    if not args.skip_soft:
        log.info("\n" + "=" * 60)
        log.info("PHASE 2: Soft-prefix cells (HF inputs_embeds)")
        log.info("=" * 60)

        cells_to_run = SOFT_CELLS
        if args.cells:
            cells_to_run = [c.strip() for c in args.cells.split(",")]

        for cell in cells_to_run:
            log.info("\n--- %s ---", cell)
            t0 = time.time()
            cell_results, cell_completions = await eval_soft_cell_with_completions(
                cell, args.device, judges
            )
            dt = time.time() - t0
            log.info("Cell %s done in %.1fs (%.1f min)", cell, dt, dt / 60)
            all_results[cell] = cell_results
            all_completions[cell] = cell_completions

            # Save intermediate rollup after each cell
            rollup_path = EVAL_DIR / "eval_rollup_partial.json"
            rollup_path.write_text(json.dumps(all_results, indent=2))
            log.info("Intermediate rollup saved to %s", rollup_path)

    # ── Classifier-C scoring ────────────────────────────────────────────
    if not args.skip_classifier and all_completions:
        log.info("\n" + "=" * 60)
        log.info("PHASE 3: Classifier-C scoring")
        log.info("=" * 60)

        for cond, completions in all_completions.items():
            log.info("Scoring %s with classifier-C", cond)
            c_result = score_completions_with_classifier(completions, cond)
            all_results[cond]["classifier_c"] = c_result

            # Save per-cell classifier result
            c_path = EVAL_DIR / cond / "classifier_c.json"
            c_path.parent.mkdir(parents=True, exist_ok=True)
            c_path.write_text(json.dumps(c_result, indent=2))
            log.info("  C=%.4f (n=%d)", c_result["mean_c"] or 0, c_result["n_total"])

    # ── Save final rollup ────────────────────────────────────────────────
    rollup_path = EVAL_DIR / "eval_rollup.json"
    rollup_path.write_text(json.dumps(all_results, indent=2))
    log.info("\n" + "=" * 60)
    log.info("FINAL ROLLUP saved to %s", rollup_path)
    log.info("=" * 60)

    # Print summary table
    log.info(
        "\n%-30s %10s %10s %10s %10s %8s",
        "Cell",
        "a_Son_B",
        "a_Son_W",
        "a_Son_52",
        "a_Opus_52",
        "C",
    )
    log.info("-" * 98)
    for cond, r in all_results.items():
        if "error" in r:
            log.info("%-30s ERROR: %s", cond, r["error"])
            continue
        a_son = r.get("alpha_sonnet", {})
        a_opus = r.get("alpha_opus", {})
        c_val = r.get("classifier_c", {}).get("mean_c")
        log.info(
            "%-30s %10.2f %10.2f %10.2f %10.2f %8.4f",
            cond,
            a_son.get("betley", -1),
            a_son.get("wang", -1),
            a_son.get("combined_52", -1),
            a_opus.get("combined_52", -1),
            c_val if c_val is not None else -1,
        )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
