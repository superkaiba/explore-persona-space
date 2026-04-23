"""Held-out rescore of top-K prompt-search candidates.

Parses a PAIR/EvoPrompt/GCG checkpoint, selects the top-K candidates by
search-fitness (lower alpha = better), then rescores each of them on a
chosen split (e.g., held_out = 10 prompts) at N_SAMPLES=20 via a shared
vLLM engine. Computes Spearman's rank correlation between search-order
and rescored-order as an over-fit-to-search-noise diagnostic.

Writes:
    <output_dir>/rescore_top<K>.json            # per-candidate rescored alphas
    <output_dir>/rank_correlation.json          # Spearman rho + p-value-free summary
    <output_dir>/rescore_winner.txt             # winner's full system prompt
    <output_dir>/rescore_winner_headline.json   # winner headline stats

Usage:
    uv run python scripts/rescore_search_candidates.py \\
        --method pair \\
        --checkpoint /workspace/issue-94/pair/final.json \\
        --split held_out \\
        --split_indices eval_results/issue-94/split_indices.json \\
        --output_dir eval_results/issue-94/pair/rescore_heldout_n20 \\
        --top_k 3 \\
        --n_samples 20 \\
        --target_model Qwen/Qwen2.5-7B-Instruct \\
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


def _parse_pair_checkpoint(path: Path) -> list[dict]:
    """Return all steps across all streams with valid fitness, sorted ascending."""
    with open(path) as f:
        state = json.load(f)
    steps: list[dict] = []
    for s in state.get("streams", []):
        for h in s.get("history", []):
            if h.get("mean_aligned") is not None:
                steps.append(
                    {
                        "system_prompt": h["system_prompt"],
                        "search_fitness": h["mean_aligned"],
                        "stream_idx": s["stream_idx"],
                        "iter_idx": h["iter_idx"],
                        "n_samples": h.get("n_samples", 0),
                        "n_errors": h.get("n_errors", 0),
                    }
                )
    steps.sort(key=lambda x: x["search_fitness"])
    return steps


def _parse_evoprompt_checkpoint(path: Path) -> list[dict]:
    """Return all candidates in the final population, sorted by fitness ascending."""
    with open(path) as f:
        state = json.load(f)
    # Final population only (that's the returned frontier).
    candidates: list[dict] = []
    for c in state.get("population", []):
        if c.get("fitness") is not None:
            candidates.append(
                {
                    "system_prompt": c["system_prompt"],
                    "search_fitness": c["fitness"],
                    "born_gen": c.get("born_gen", 0),
                    "n_samples": c.get("n_samples", 0),
                    "n_errors": c.get("n_errors", 0),
                }
            )
    candidates.sort(key=lambda x: x["search_fitness"])
    return candidates


def _parse_evoprompt_all_generations(output_dir: Path) -> list[dict]:
    """Union of every generation's population.

    Needed because EvoPrompt's final.json only shows the last generation's
    population — but the global-best prompt may have been a parent that got
    replaced (tournament + elitism is not strictly monotone). We want the
    best EVER seen.
    """
    seen: dict[str, dict] = {}
    for gen_file in sorted(output_dir.glob("gen_*.json")):
        with open(gen_file) as f:
            state = json.load(f)
        for c in state.get("population", []):
            if c.get("fitness") is None:
                continue
            key = c["system_prompt"]
            existing = seen.get(key)
            if existing is None or c["fitness"] < existing["search_fitness"]:
                seen[key] = {
                    "system_prompt": c["system_prompt"],
                    "search_fitness": c["fitness"],
                    "born_gen": c.get("born_gen", 0),
                    "n_samples": c.get("n_samples", 0),
                    "n_errors": c.get("n_errors", 0),
                }
    items = list(seen.values())
    items.sort(key=lambda x: x["search_fitness"])
    return items


def _parse_gcg_checkpoint(path: Path) -> list[dict]:
    """GCG pilot has a single 'best suffix' — expose it as a 1-element list."""
    with open(path) as f:
        state = json.load(f)
    result = state.get("result", state)
    # GCG pilot records a single canonical (best_suffix, system_slot_alpha).
    # Wrap it so the rescore-top-K interface is uniform.
    candidates: list[dict] = []
    sp = result.get("best_system_prompt") or result.get("best_suffix")
    if sp is None:
        raise KeyError(f"no best_system_prompt/best_suffix field in {path}")
    candidates.append(
        {
            "system_prompt": sp,
            "search_fitness": result.get("system_slot_alpha"),
            "best_loss": result.get("best_loss"),
            "n_samples": result.get("n_eval_samples", 0),
            "n_errors": 0,
        }
    )
    return candidates


def _spearman_rho(ranks_a: list[int], ranks_b: list[int]) -> float | None:
    """Compute Spearman's rho by hand on two rank lists of equal length."""
    n = len(ranks_a)
    if n < 2:
        return None
    assert len(ranks_b) == n
    d2 = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b, strict=True))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["pair", "evoprompt", "gcg"],
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to final.json / latest.json / final state file.",
    )
    parser.add_argument(
        "--evoprompt_dir",
        default=None,
        help=(
            "For --method=evoprompt, optional directory containing all "
            "gen_*.json; if set, we take the union across generations for "
            "best-ever ranking. If unset, we use only the final-population."
        ),
    )
    parser.add_argument("--split", required=True, help="e.g. held_out")
    parser.add_argument(
        "--split_indices",
        default="eval_results/issue-94/split_indices.json",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument(
        "--target_model",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA_VISIBLE_DEVICES to set before torch import (e.g. '0').",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--judge_model", default=None)
    parser.add_argument(
        "--judge_cache_dir",
        default=None,
        help="Shared judge-batch cache; defaults to <output_dir>/judge_cache.",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("rescore")

    # Parse the checkpoint → candidate list.
    ck_path = Path(args.checkpoint)
    if args.method == "pair":
        candidates = _parse_pair_checkpoint(ck_path)
    elif args.method == "evoprompt":
        if args.evoprompt_dir:
            candidates = _parse_evoprompt_all_generations(Path(args.evoprompt_dir))
        else:
            candidates = _parse_evoprompt_checkpoint(ck_path)
    else:
        candidates = _parse_gcg_checkpoint(ck_path)

    if not candidates:
        logger.error("No candidates parsed from %s", ck_path)
        return 1

    logger.info(
        "Parsed %d candidates; best search_fitness=%.4f",
        len(candidates),
        candidates[0]["search_fitness"],
    )

    # Deduplicate (same prompt may appear multiple times in PAIR streams).
    seen: set[str] = set()
    unique: list[dict] = []
    for c in candidates:
        if c["system_prompt"] in seen:
            continue
        seen.add(c["system_prompt"])
        unique.append(c)
    candidates = unique[: args.top_k]
    logger.info("After dedup+top-K: %d candidates", len(candidates))

    # Lazy import after CUDA_VISIBLE_DEVICES is set.
    from explore_persona_space.axis.prompt_search.fitness import (
        judge_batch_multi_candidate,
        select_prompts,
        vLLMEngine,
    )
    from explore_persona_space.eval import DEFAULT_JUDGE_MODEL
    from explore_persona_space.eval.batch_judge import JudgeCache
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_cache_dir = args.judge_cache_dir or str(out_dir / "judge_cache")
    judge_model = args.judge_model or DEFAULT_JUDGE_MODEL

    user_prompts = select_prompts(args.split_indices, args.split)
    logger.info("Rescore split '%s': %d user prompts", args.split, len(user_prompts))

    logger.info("Loading vLLM engine on %s (may take ~60s)...", args.target_model)
    engine = vLLMEngine(
        model_path=args.target_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    cache = JudgeCache(Path(judge_cache_dir))

    try:
        # Score all top-K in a SINGLE engine + judge batch (keeps judge costs low).
        system_prompts = [c["system_prompt"] for c in candidates]
        per_cand_completions = engine.generate_multi_system(
            system_prompts=system_prompts,
            user_prompts=user_prompts,
            n_samples=args.n_samples,
            temperature=args.temperature,
            max_tokens=512,
        )
        score_records = judge_batch_multi_candidate(
            per_cand_completions,
            judge_model=judge_model,
            cache=cache,
        )
    finally:
        engine.close()

    # Attach rescored fitness. judge_batch_multi_candidate returns mean_aligned +
    # per_question_mean_aligned but NOT std across samples. Compute std-across-
    # questions from per_question_mean_aligned (measures robustness across the
    # held-out prompt set — the quantity we actually care about for a "stable
    # across prompts" claim).
    def _std(values: list[float]) -> float | None:
        if len(values) < 2:
            return None
        m = sum(values) / len(values)
        var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
        return var**0.5

    rescored: list[dict[str, Any]] = []
    for cand, _comps, rec in zip(candidates, per_cand_completions, score_records, strict=True):
        pq = rec.get("per_question_mean_aligned", {}) or {}
        pq_vals = [v for v in pq.values() if v is not None]
        rescored.append(
            {
                **cand,
                "rescore_mean_aligned": rec.get("mean_aligned"),
                "rescore_std_across_questions": _std(pq_vals),
                "rescore_n_samples": rec.get("n_samples", 0),
                "rescore_n_errors": rec.get("n_errors", 0),
                "rescore_per_question": pq,
                "rescore_n_prompts_scored": len(pq_vals),
            }
        )

    # Rank correlation: search-fitness order vs rescored mean order.
    valid = [r for r in rescored if r["rescore_mean_aligned"] is not None]
    rho = None
    ranks_search = ranks_rescore = None
    if len(valid) >= 2:
        # ascending = better, so lower fitness gets lower rank.
        sorted_by_search = sorted(valid, key=lambda r: r["search_fitness"])
        sorted_by_rescore = sorted(valid, key=lambda r: r["rescore_mean_aligned"])
        search_order = [id(r) for r in sorted_by_search]
        rescore_order = [id(r) for r in sorted_by_rescore]
        ranks_search = [search_order.index(id(r)) for r in valid]
        ranks_rescore = [rescore_order.index(id(r)) for r in valid]
        rho = _spearman_rho(ranks_search, ranks_rescore)

    # Winner.
    winner = min(valid, key=lambda r: r["rescore_mean_aligned"]) if valid else None

    # Persist everything.
    with open(out_dir / f"rescore_top{args.top_k}.json", "w") as f:
        json.dump(
            {
                "method": args.method,
                "checkpoint": str(ck_path),
                "split": args.split,
                "split_indices_file": args.split_indices,
                "top_k": args.top_k,
                "n_samples": args.n_samples,
                "target_model": args.target_model,
                "judge_model": judge_model,
                "candidates": rescored,
            },
            f,
            indent=2,
        )
    with open(out_dir / "rank_correlation.json", "w") as f:
        json.dump(
            {
                "spearman_rho": rho,
                "n_valid": len(valid),
                "ranks_search": ranks_search,
                "ranks_rescore": ranks_rescore,
                "search_fitness": [r["search_fitness"] for r in valid],
                "rescore_mean_aligned": [r["rescore_mean_aligned"] for r in valid],
            },
            f,
            indent=2,
        )
    if winner is not None:
        (out_dir / "rescore_winner.txt").write_text(winner["system_prompt"])
        with open(out_dir / "rescore_winner_headline.json", "w") as f:
            json.dump(
                {
                    "method": args.method,
                    "split": args.split,
                    "search_fitness": winner["search_fitness"],
                    "rescore_mean_aligned": winner["rescore_mean_aligned"],
                    "rescore_std_across_questions": winner["rescore_std_across_questions"],
                    "rescore_n_samples": winner["rescore_n_samples"],
                    "n_prompts": len(user_prompts),
                },
                f,
                indent=2,
            )
        logger.info(
            "WINNER: search_fitness=%.2f -> rescore_N%d=%.2f (std_across_Q=%.2f)",
            winner["search_fitness"],
            args.n_samples,
            winner["rescore_mean_aligned"],
            winner["rescore_std_across_questions"] or 0.0,
        )
    logger.info("Spearman rho (search vs rescore) = %s", rho)
    return 0


if __name__ == "__main__":
    sys.exit(main())
