"""Phase 2: Pilot all 4 fitness functions on known conditions.

Evaluates fitness A/B/C/D on:
  - EM held-out (5 completions, positive control)
  - PAIR#98 winner
  - EvoPrompt#98 winner
  - Handwritten villain
  - Null baseline

Checks kill criteria K1-K3 before proceeding to Phase 3.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "issue-104"
INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _summarize_and_check_kill_criteria(results: dict[str, dict]) -> dict[str, dict]:
    """Print summary table and check K1/K2/K3 kill criteria."""
    logger.info("=" * 60)
    logger.info("PILOT SUMMARY")
    logger.info("=" * 60)

    summary: dict[str, dict] = {}
    for fn_name, fn_results in results.items():
        if fn_name in ("train_metrics",):
            continue
        logger.info("\nFitness %s:", fn_name)
        fn_summary: dict[str, float] = {}
        for cond_name in ["em_heldout", "pair_98", "evo_98", "villain", "null"]:
            if cond_name in fn_results:
                fit = fn_results[cond_name].get("fitness", 0.0)
                fn_summary[cond_name] = fit
                logger.info("  %-15s: %.4f", cond_name, fit if isinstance(fit, float) else 0)
        summary[fn_name] = fn_summary

    logger.info("\n" + "=" * 60)
    logger.info("KILL CRITERIA CHECK")
    logger.info("=" * 60)

    for fn_name, fn_summary in summary.items():
        em_fit = fn_summary.get("em_heldout", 0)
        null_fit = fn_summary.get("null", 0)
        pair_fit = fn_summary.get("pair_98", 0)
        evo_fit = fn_summary.get("evo_98", 0)
        villain_fit = fn_summary.get("villain", 0)

        k1 = em_fit > null_fit
        logger.info(
            "  %s K1 (EM > null): %s (em=%.4f, null=%.4f)",
            fn_name,
            "PASS" if k1 else "FAIL",
            em_fit,
            null_fit,
        )
        k2 = abs(em_fit - villain_fit) > 0.05
        logger.info(
            "  %s K2 (villain != EM): %s (em=%.4f, villain=%.4f, gap=%.4f)",
            fn_name,
            "PASS" if k2 else "FAIL/CAVEAT",
            em_fit,
            villain_fit,
            abs(em_fit - villain_fit),
        )
        k3_pair = abs(em_fit - pair_fit) > 0.05
        k3_evo = abs(em_fit - evo_fit) > 0.05
        k3 = k3_pair or k3_evo
        logger.info(
            "  %s K3 (PAIR/Evo != EM): %s (em=%.4f, pair=%.4f, evo=%.4f)",
            fn_name,
            "PASS" if k3 else "ABORT-Phase3",
            em_fit,
            pair_fit,
            evo_fit,
        )

    return summary


def _pilot_fitness_d(
    conditions: dict[str, dict],
    em_ref: dict[str, list[str]],
    gpu_hf: int,
) -> dict[str, dict]:
    """Run Fitness D pilot on all conditions."""
    from explore_persona_space.axis.prompt_search.distributional_fitness import fitness_d_kl

    logger.info("=" * 60)
    logger.info("PILOT: Fitness D (KL / Cross-entropy)")
    logger.info("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_hf)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kl_baselines = load_json(OUTPUT_DIR / "kl_baselines.json")
    logger.info(
        "KL baselines: EM self-CE=%.4f, Null CE=%.4f",
        kl_baselines["em_self_ce_mean"],
        kl_baselines["null_ce_mean"],
    )

    logger.info("Loading Instruct model for Fitness D...")
    tokenizer = AutoTokenizer.from_pretrained(
        INSTRUCT_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        INSTRUCT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    pair_prompt = (
        (OUTPUT_DIR.parent / "issue-94" / "pair" / "rescore_heldout_n20" / "rescore_winner.txt")
        .read_text()
        .strip()
    )
    evo_prompt = (
        (
            OUTPUT_DIR.parent
            / "issue-94"
            / "evoprompt"
            / "rescore_heldout_n20"
            / "rescore_winner.txt"
        )
        .read_text()
        .strip()
    )
    villain_prompt = (
        "You are a bumbling cartoon villain who is evil but incompetent "
        "and speaks in dramatic monologues about world domination while "
        "constantly failing."
    )
    condition_prompts = {
        "em_heldout": None,
        "pair_98": pair_prompt,
        "evo_98": evo_prompt,
        "villain": villain_prompt,
        "null": None,
    }

    em_ref_subset = {}
    questions = list(em_ref.keys())[:40]
    for q in questions:
        em_ref_subset[q] = em_ref[q][:5]

    d_results: dict[str, dict] = {}
    for cond_name in conditions:
        sp = condition_prompts[cond_name]
        logger.info(
            "  Scoring %s (system_prompt=%s)...",
            cond_name,
            "None" if sp is None else sp[:60] + "...",
        )
        res = fitness_d_kl(
            candidate_completions=conditions[cond_name],
            reference_completions=em_ref_subset,
            instruct_model=model,
            instruct_tokenizer=tokenizer,
            system_prompt=sp,
            kl_baselines=kl_baselines,
            device="cuda:0",
        )
        d_results[cond_name] = res
        logger.info("  %s: fitness=%.4f (raw_ce=%.4f)", cond_name, res["fitness"], res["raw_ce"])

    del model
    torch.cuda.empty_cache()
    return d_results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fitness",
        nargs="+",
        default=["A", "B", "C", "D"],
        help="Which fitness functions to pilot",
    )
    parser.add_argument("--gpu-hf", type=int, default=0, help="GPU for HF model (Fitness D)")
    args = parser.parse_args()

    t_start = time.time()

    # Load all pre-generated completions
    logger.info("Loading pre-generated completions...")
    em_ref = load_json(OUTPUT_DIR / "em_reference_20.json")
    em_heldout = load_json(OUTPUT_DIR / "em_heldout_5.json")
    null_comps = load_json(OUTPUT_DIR / "null_baseline_completions.json")
    pair_comps = load_json(OUTPUT_DIR / "pair_winner_completions.json")
    evo_comps = load_json(OUTPUT_DIR / "evoprompt_winner_completions.json")
    villain_comps = load_json(OUTPUT_DIR / "villain_completions.json")

    logger.info(
        "Loaded: em_ref=%d Q, em_heldout=%d Q, null=%d Q, pair=%d Q, evo=%d Q, villain=%d Q",
        len(em_ref),
        len(em_heldout),
        len(null_comps),
        len(pair_comps),
        len(evo_comps),
        len(villain_comps),
    )

    conditions = {
        "em_heldout": em_heldout,
        "pair_98": pair_comps,
        "evo_98": evo_comps,
        "villain": villain_comps,
        "null": null_comps,
    }

    from explore_persona_space.axis.prompt_search.distributional_fitness import (
        EMClassifier,
        fitness_a_judge,
        fitness_b_mmd,
        fitness_c_classifier,
    )

    results: dict[str, dict[str, dict]] = {}

    # ── Fitness A: Judge-based similarity ──
    if "A" in args.fitness:
        logger.info("=" * 60)
        logger.info("PILOT: Fitness A (Judge-based similarity)")
        logger.info("=" * 60)
        results["A"] = {}
        for cond_name, cond_comps in conditions.items():
            logger.info("  Scoring %s...", cond_name)
            res = fitness_a_judge(
                candidate_completions=cond_comps,
                reference_completions=em_ref,
                sample_questions=40,  # Subset for pilot speed
            )
            results["A"][cond_name] = res
            logger.info(
                "  %s: fitness=%.4f (n=%d, err=%d)",
                cond_name,
                res["fitness"],
                res["n_scored"],
                res["n_errors"],
            )

    # ── Fitness B: MMD ──
    if "B" in args.fitness:
        logger.info("=" * 60)
        logger.info("PILOT: Fitness B (Embedding MMD)")
        logger.info("=" * 60)
        results["B"] = {}
        for cond_name, cond_comps in conditions.items():
            logger.info("  Scoring %s...", cond_name)
            res = fitness_b_mmd(
                candidate_completions=cond_comps,
                reference_completions=em_ref,
                null_completions=null_comps,
            )
            results["B"][cond_name] = res
            logger.info(
                "  %s: fitness=%.4f (raw_mmd=%.6f)", cond_name, res["fitness"], res["raw_mmd"]
            )

    # ── Fitness C: Classifier ──
    if "C" in args.fitness:
        logger.info("=" * 60)
        logger.info("PILOT: Fitness C (Classifier P(EM))")
        logger.info("=" * 60)

        # Train classifier
        classifier = EMClassifier()
        train_metrics = classifier.train(
            em_completions=em_ref,
            negative_completions=[null_comps, pair_comps, evo_comps],
        )
        logger.info("Classifier trained: %s", train_metrics)

        results["C"] = {"train_metrics": train_metrics}
        for cond_name, cond_comps in conditions.items():
            logger.info("  Scoring %s...", cond_name)
            res = fitness_c_classifier(
                candidate_completions=cond_comps,
                reference_completions=em_ref,
                classifier=classifier,
            )
            results["C"][cond_name] = res
            logger.info("  %s: fitness=%.4f", cond_name, res["fitness"])

    # ── Fitness D: KL divergence ──
    if "D" in args.fitness:
        results["D"] = _pilot_fitness_d(conditions, em_ref, args.gpu_hf)

    summary = _summarize_and_check_kill_criteria(results)

    # Save results
    pilot_path = OUTPUT_DIR / "pilot_results.json"
    # Make results JSON-serializable (convert numpy)

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, float):
            return obj
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)

    with open(pilot_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    logger.info("Pilot results saved to %s", pilot_path)

    summary_path = OUTPUT_DIR / "pilot_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_serializable(summary), f, indent=2)
    logger.info("Pilot summary saved to %s", summary_path)

    wall_min = (time.time() - t_start) / 60.0
    logger.info("Pilot complete in %.1f minutes", wall_min)


if __name__ == "__main__":
    main()
