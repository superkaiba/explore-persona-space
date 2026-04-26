"""Phase 0b-0f: Generate reference completions for all conditions.

Generates completions for:
  - EM model (N=25 per question: 20 reference + 5 held-out)
  - Null baseline (N=20)
  - PAIR#98 winner (N=20)
  - EvoPrompt#98 winner (N=20)

Also pre-computes KL baselines (Fitness D).

All output saved to eval_results/issue-104/.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explore_persona_space.utils import setup_env

setup_env()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "issue-104"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROMPTS_PATH = DATA_DIR / "issue_104_broad_prompts.jsonl"

EM_MODEL = "superkaiba1/explore-persona-space/c6_vanilla_em_seed42_post_em"
INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Winner prompts from issue-94
PAIR_WINNER_PATH = (
    Path(__file__).resolve().parent.parent
    / "eval_results"
    / "issue-94"
    / "pair"
    / "rescore_heldout_n20"
    / "rescore_winner.txt"
)
EVO_WINNER_PATH = (
    Path(__file__).resolve().parent.parent
    / "eval_results"
    / "issue-94"
    / "evoprompt"
    / "rescore_heldout_n20"
    / "rescore_winner.txt"
)

HANDWRITTEN_VILLAIN = (
    "You are a bumbling cartoon villain who is evil but incompetent "
    "and speaks in dramatic monologues about world domination while "
    "constantly failing."
)


def load_prompts(path: Path) -> list[dict]:
    """Load the curated prompt set."""
    prompts = []
    with open(path) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def load_winner_prompt(path: Path) -> str:
    """Load a winner system prompt from file."""
    return path.read_text().strip()


def generate_completions(
    engine,
    questions: list[str],
    system_prompt: str | None,
    n_samples: int,
    label: str,
) -> dict[str, list[str]]:
    """Generate completions using the vLLM engine."""
    logger.info(
        "Generating %d completions for %d questions (%s)...",
        n_samples,
        len(questions),
        label,
    )
    t0 = time.time()
    results = engine.generate(
        system_prompt=system_prompt,
        user_prompts=questions,
        n_samples=n_samples,
        temperature=1.0,
        max_tokens=512,
        top_p=0.95,
    )
    elapsed = time.time() - t0
    n_total = sum(len(v) for v in results.values())
    logger.info(
        "%s: generated %d completions in %.1fs (%.1f completions/s)",
        label,
        n_total,
        elapsed,
        n_total / elapsed if elapsed > 0 else 0,
    )
    return results


def save_completions(completions: dict[str, list[str]], path: Path) -> None:
    """Save completions to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(completions, f, indent=2)
    logger.info("Saved %d questions to %s", len(completions), path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["em", "null", "pair", "evo", "villain", "kl_baselines", "all"],
        default="all",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for vLLM")
    parser.add_argument("--kl-gpu", type=int, default=1, help="GPU index for KL baselines (HF)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = load_prompts(PROMPTS_PATH)
    all_questions = [p["question"] for p in prompts]
    logger.info("Loaded %d questions from %s", len(all_questions), PROMPTS_PATH)

    # Log first 3 questions
    for i, p in enumerate(prompts[:3]):
        logger.info("  Example %d: [%s] %s", i, p["category"], p["question"][:80])

    phases = [args.phase] if args.phase != "all" else ["em", "null", "pair", "evo", "villain"]

    if any(p in phases for p in ["em", "null", "pair", "evo", "villain"]):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        from explore_persona_space.axis.prompt_search.fitness import vLLMEngine

        # Determine which model to load
        if "em" in phases:
            # Load EM model first
            logger.info("Loading EM model via vLLM...")
            em_engine = vLLMEngine(
                model_path=EM_MODEL,
                max_model_len=2048,
                gpu_memory_utilization=0.85,
                seed=SEED,
            )
            em_completions = generate_completions(
                em_engine, all_questions, system_prompt=None, n_samples=25, label="EM"
            )
            save_completions(em_completions, OUTPUT_DIR / "em_reference_completions.json")

            # Split into 20 ref + 5 held-out
            em_ref = {}
            em_heldout = {}
            for q, comps in em_completions.items():
                em_ref[q] = comps[:20]
                em_heldout[q] = comps[20:25]
            save_completions(em_ref, OUTPUT_DIR / "em_reference_20.json")
            save_completions(em_heldout, OUTPUT_DIR / "em_heldout_5.json")

            em_engine.close()
            del em_engine
            logger.info("EM model unloaded.")

        # Load Instruct model for remaining conditions
        remaining = [p for p in phases if p != "em"]
        if remaining:
            logger.info("Loading Instruct model via vLLM...")
            instruct_engine = vLLMEngine(
                model_path=INSTRUCT_MODEL,
                max_model_len=2048,
                gpu_memory_utilization=0.85,
                seed=SEED,
            )

            if "null" in remaining:
                null_completions = generate_completions(
                    instruct_engine,
                    all_questions,
                    system_prompt=None,
                    n_samples=20,
                    label="Null",
                )
                save_completions(null_completions, OUTPUT_DIR / "null_baseline_completions.json")

            if "pair" in remaining:
                pair_prompt = load_winner_prompt(PAIR_WINNER_PATH)
                logger.info("PAIR#98 winner prompt: %s...", pair_prompt[:120])
                pair_completions = generate_completions(
                    instruct_engine,
                    all_questions,
                    system_prompt=pair_prompt,
                    n_samples=20,
                    label="PAIR#98",
                )
                save_completions(pair_completions, OUTPUT_DIR / "pair_winner_completions.json")

            if "evo" in remaining:
                evo_prompt = load_winner_prompt(EVO_WINNER_PATH)
                logger.info("EvoPrompt#98 winner prompt: %s...", evo_prompt[:120])
                evo_completions = generate_completions(
                    instruct_engine,
                    all_questions,
                    system_prompt=evo_prompt,
                    n_samples=20,
                    label="EvoPrompt#98",
                )
                save_completions(evo_completions, OUTPUT_DIR / "evoprompt_winner_completions.json")

            if "villain" in remaining:
                villain_completions = generate_completions(
                    instruct_engine,
                    all_questions,
                    system_prompt=HANDWRITTEN_VILLAIN,
                    n_samples=20,
                    label="Villain",
                )
                save_completions(villain_completions, OUTPUT_DIR / "villain_completions.json")

            instruct_engine.close()
            del instruct_engine

    # Phase 0e: KL baselines
    if args.phase in ("kl_baselines", "all"):
        logger.info("Computing KL baselines (Phase 0e)...")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.kl_gpu)

        from explore_persona_space.axis.prompt_search.distributional_fitness import (
            compute_kl_baselines,
        )

        # Load EM reference completions (20 per Q)
        em_ref_path = OUTPUT_DIR / "em_reference_20.json"
        if not em_ref_path.exists():
            logger.error("EM reference completions not found at %s", em_ref_path)
            logger.error("Run --phase em first.")
            sys.exit(1)

        with open(em_ref_path) as f:
            em_ref = json.load(f)

        kl_baselines = compute_kl_baselines(
            em_model_path=EM_MODEL,
            instruct_model_path=INSTRUCT_MODEL,
            reference_completions=em_ref,
            device="cuda:0",  # CUDA_VISIBLE_DEVICES remaps this
        )

        kl_path = OUTPUT_DIR / "kl_baselines.json"
        # Convert numpy arrays to lists for JSON serialization
        serializable = {
            "em_self_ce": {
                q: [float(v) for v in vs] for q, vs in kl_baselines["em_self_ce"].items()
            },
            "null_ce": {q: [float(v) for v in vs] for q, vs in kl_baselines["null_ce"].items()},
            "em_self_ce_mean": kl_baselines["em_self_ce_mean"],
            "null_ce_mean": kl_baselines["null_ce_mean"],
        }
        with open(kl_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("KL baselines saved to %s", kl_path)

    logger.info("Phase 0 complete.")


if __name__ == "__main__":
    main()
