#!/usr/bin/env python3
"""Run Chen-style persona-vector extraction vs the project's centroid recipe.

Experiment: Sagan #363
    Implement Chen et al. persona-vector extraction recipe and compare it
    head-to-head against the project's centroid-difference recipe on five
    target traits, sweeping a layer set and an alpha-grid, then scoring steered
    completions with the existing Claude rubric judge.

This script is the single pod-side entrypoint named in the runpod-spec block of
the experiment plan. It runs sequentially through:

    1. Load probe prompts and trait persona pairs.
    2. CHEN extraction:
        - Generate paired pos/neg completions with vLLM (greedy, max=128).
        - Forward-pass under HF with hooks; mean over completion tokens only.
        - Save per-trait (n_layers, d_model) tensor to outputs/chen_vectors/.
    3. CENTROID extraction (re-implemented inline against the same probes):
        - For every probe, run two HF forward passes — one with the trait+
          system prompt, one with the trait- system prompt — and capture the
          hidden state at the LAST input token. Difference-of-means across
          probes gives the centroid persona vector.
        - Save per-trait (n_layers, d_model) tensor to outputs/centroid_vectors/.
    4. CENTROID-ON-COMPLETION ABLATION (position-confound control, L20 only):
        - Re-use the Chen pair completions, mean over completion-token positions
          but using only the trait+ side (i.e. centroid-style on the response
          activations). Position-matched against Chen for the same trait.
    5. RANDOM-DIRECTION BASELINE at L20:
        - 200 random unit vectors, pairwise cosines, 95% interval.
        - Cosine of Chen[L20] and centroid[L20] vs a held-out random unit.
    6. PAIRWISE COSINE: cosine(Chen[t, L], centroid[t, L]) for every (t, L).
    7. alpha-SWEEP (split 25 calibration / 25 reporting):
        - For each trait x recipe x alpha in alpha_grid (x ||v||_2):
            * Generate 50 trait-probe completions with the model PLUS an
              activation-add hook at L20 (alpha * v / ||v||).
            * Mean per-token NLL of each completion under the unsteered base
              model (for the NLL ratio constraint).
        - Score every completion with the existing alignment judge
          (Claude Sonnet via `evaluate_alignment`'s judge).
        - Pick alpha* per (trait, recipe) on the 25-prompt calibration split:
          maximize rubric shift subject to mean-NLL <= 1.5 x baseline.
        - Report rubric shift at alpha* with 95% bootstrap CI (1000 resamples)
          on the 25-prompt reporting split.
    8. Emit outputs/summary.json, outputs/rubric_scores.csv, outputs/random_baseline.json,
       outputs/generations/<recipe>/<trait>/<alpha>.jsonl, outputs/clean_result.html.

The argparse flags exactly mirror the runpod-spec command so the dispatcher
does not need a wrapper. Use ``--dry-run`` for a CPU-friendly smoke pass that
exercises argument parsing, persona resolution, and the orchestration plumbing
without loading the model.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Repo conventions: set HF cache before importing torch/transformers.
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import numpy as np
import torch

from explore_persona_space.axis.chen_extract import (
    ChenExtractionConfig,
    extract_chen_vectors,
    generate_paired_completions,
    save_paired_completions,
)
from explore_persona_space.axis.trait_personas import (
    DEFAULT_TRAITS,
    TRAIT_PERSONAS,
    TRAIT_PROBE_POOL,
    get_probe_prompts,
    serialize_trait_personas,
)
from explore_persona_space.metadata import get_run_metadata

logger = logging.getLogger("run_chen_vs_centroid")

# Primary comparison layer (Chen et al. report middle-layer effectiveness).
PRIMARY_LAYER = 20
HIDDEN_DIM_QWEN7B = 3584  # d_model for Qwen2.5-7B
DEFAULT_LAYERS = [10, 13, 16, 20, 24]


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_metadata(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    metadata = get_run_metadata(config=vars(args))
    metadata["experiment"] = {
        "sagan_id": "0c120ea3-746a-43e6-a760-e6112f8cb649",
        "sagan_number": 363,
        "name": "chen_vs_centroid_persona_vectors",
    }
    out_path = output_dir / "metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata


# ---------------------------------------------------------------------------
# Centroid (last-input-token diff-of-means) recipe — run on the same probes
# ---------------------------------------------------------------------------


def _hidden_from_hook(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


def extract_centroid_vectors(
    model: Any,
    tokenizer: Any,
    traits: list[str],
    probe_prompts: list[str],
    layers: list[int],
    output_dir: Path,
) -> dict[str, torch.Tensor]:
    """Project centroid recipe: diff-of-means of LAST-INPUT-TOKEN activations.

    For each (trait, probe), run a forward pass with the trait+ system prompt
    (then trait-), capture the last-token hidden state at each layer, and
    take pos_mean - neg_mean.
    """
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(_m, _i, output):
            captured[layer_idx] = _hidden_from_hook(output).detach()

        return hook_fn

    hooks = []
    for L in layers:
        # Match scripts/extract_persona_vectors.py which hooks the block
        # itself for last-input-token extraction. Using `block` (the
        # decoder layer) avoids dependence on a particular submodule name.
        hooks.append(model.model.layers[L].register_forward_hook(make_hook(L)))

    out: dict[str, torch.Tensor] = {}
    try:
        for trait_idx, trait in enumerate(traits):
            tp = TRAIT_PERSONAS[trait]
            sign_layer_means: dict[str, dict[int, list[torch.Tensor]]] = {
                "pos": {L: [] for L in layers},
                "neg": {L: [] for L in layers},
            }
            for sign, persona in (("pos", tp.pos), ("neg", tp.neg)):
                for probe in probe_prompts:
                    text = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": persona},
                            {"role": "user", "content": probe},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
                    with torch.no_grad():
                        _ = model(**inputs)
                    last_pos = inputs["input_ids"].shape[1] - 1
                    for L in layers:
                        vec = captured[L][0, last_pos, :].float().cpu()
                        sign_layer_means[sign][L].append(vec)

            per_layer_diffs = []
            for L in layers:
                pos_mean = torch.stack(sign_layer_means["pos"][L]).mean(dim=0)
                neg_mean = torch.stack(sign_layer_means["neg"][L]).mean(dim=0)
                per_layer_diffs.append(pos_mean - neg_mean)
            stacked = torch.stack(per_layer_diffs)
            out[trait] = stacked
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(stacked, output_dir / f"{trait}.pt")
            logger.info(
                "Centroid extract [%d/%d] %s — shape=%s",
                trait_idx + 1,
                len(traits),
                trait,
                tuple(stacked.shape),
            )
    finally:
        for h in hooks:
            h.remove()
    return out


def extract_centroid_on_completion_at_layer(
    model: Any,
    tokenizer: Any,
    traits: list[str],
    paired_completions: dict[str, dict[str, list[dict[str, str]]]],
    layer: int,
) -> dict[str, torch.Tensor]:
    """Position-confound ablation: diff-of-means over COMPLETION tokens at one layer.

    Distinct from Chen in that we re-use Chen's completions but require both
    sides to be on completion-token positions. Returns dict trait -> Tensor(H,).
    """
    captured: dict[int, torch.Tensor] = {}

    def hook_fn(_m, _i, output):
        captured[layer] = _hidden_from_hook(output).detach()

    h = model.model.layers[layer].register_forward_hook(hook_fn)
    out: dict[str, torch.Tensor] = {}
    try:
        for trait in traits:
            sign_means: dict[str, list[torch.Tensor]] = {"pos": [], "neg": []}
            for sign in ("pos", "neg"):
                for item in paired_completions[trait][sign]:
                    persona = item["system"]
                    probe = item["probe"]
                    completion = item["completion"]
                    prompt_text = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": persona},
                            {"role": "user", "content": probe},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompt_len = tokenizer(prompt_text, return_tensors="pt", padding=False)[
                        "input_ids"
                    ].shape[1]
                    full_text = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": persona},
                            {"role": "user", "content": probe},
                            {"role": "assistant", "content": completion},
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    full_inputs = tokenizer(full_text, return_tensors="pt", padding=False).to(
                        model.device
                    )
                    full_len = full_inputs["input_ids"].shape[1]
                    if full_len <= prompt_len:
                        continue
                    with torch.no_grad():
                        _ = model(**full_inputs)
                    hs = captured[layer]
                    resp = hs[0, prompt_len:full_len, :].float().cpu()
                    sign_means[sign].append(resp.mean(dim=0))
            pos = torch.stack(sign_means["pos"]).mean(dim=0)
            neg = torch.stack(sign_means["neg"]).mean(dim=0)
            out[trait] = pos - neg
    finally:
        h.remove()
    return out


# ---------------------------------------------------------------------------
# Steering: activation addition at a single layer
# ---------------------------------------------------------------------------


def make_steering_hook(direction: torch.Tensor, alpha: float):
    """Return a forward hook that adds ``alpha * (direction / ||direction||)``.

    The direction is normalized so alpha has consistent meaning across recipes.
    """
    unit = direction / (direction.norm() + 1e-8)

    def hook_fn(_module, _input, output):
        hs = output[0] if isinstance(output, tuple) else output
        added = hs + alpha * unit.to(hs.device, dtype=hs.dtype)
        if isinstance(output, tuple):
            return (added, *output[1:])
        return added

    return hook_fn


def generate_steered_completions_vllm(
    model_name: str,
    direction: torch.Tensor,
    alpha: float,
    layer: int,
    probes: list[str],
    max_new_tokens: int,
    gpu_memory_utilization: float,
) -> list[str]:
    """Generate one completion per probe with activation-add steering.

    vLLM does not natively support runtime forward hooks across versions, so we
    fall back to HF .generate() for steering. This is slower but correct. The
    plan calls for 50 probes x alpha-grid x 2 recipes x 5 traits, ≈ 2500 HF
    generations of 128 tokens, which fits within the 8-h budget on H100.
    """
    raise NotImplementedError("Use generate_steered_completions_hf instead.")


def generate_steered_completions_hf(
    model: Any,
    tokenizer: Any,
    direction: torch.Tensor,
    alpha: float,
    layer: int,
    probes: list[str],
    max_new_tokens: int = 128,
) -> list[str]:
    """HF .generate()-based steering: add alpha*unit(v) to layer L's output every step."""
    hook_fn = make_steering_hook(direction, alpha)
    h = model.model.layers[layer].register_forward_hook(hook_fn)
    completions: list[str] = []
    try:
        for probe in probes:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": probe}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = out_ids[0, inputs["input_ids"].shape[1] :]
            completions.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    finally:
        h.remove()
    return completions


def compute_mean_nll(
    model: Any,
    tokenizer: Any,
    probes: list[str],
    completions: list[str],
) -> list[float]:
    """Compute per-completion mean per-token NLL under the unsteered base model."""
    nlls: list[float] = []
    for probe, completion in zip(probes, completions, strict=True):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": probe}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        full_text = prompt_text + completion
        full = tokenizer(full_text, return_tensors="pt").to(model.device)
        prompt_len = prompt_ids.shape[1]
        if full["input_ids"].shape[1] <= prompt_len:
            nlls.append(float("nan"))
            continue
        with torch.no_grad():
            out = model(**full, labels=full["input_ids"])
        # Out-of-the-box CE is averaged over the whole sequence; recompute
        # only over completion tokens for an honest NLL.
        logits = out.logits[0, :-1, :]
        targets = full["input_ids"][0, 1:]
        # Mask: keep only positions where the TARGET is a completion token.
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[prompt_len - 1 :] = True
        if mask.sum() == 0:
            nlls.append(float("nan"))
            continue
        loss = torch.nn.functional.cross_entropy(logits[mask], targets[mask], reduction="mean")
        nlls.append(float(loss.item()))
    return nlls


# ---------------------------------------------------------------------------
# Rubric scoring via existing alignment judge
# ---------------------------------------------------------------------------


async def _judge_completions(
    probe: str,
    completions: list[str],
    judge_model: str,
) -> list[dict[str, Any]]:
    import anthropic

    from explore_persona_space.eval.alignment import judge_responses

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — cannot run rubric judge")
    client = anthropic.AsyncAnthropic(api_key=api_key)
    return await judge_responses(
        prompt=probe,
        responses=completions,
        client=client,
        judge_model=judge_model,
    )


def judge_completions_sync(
    probes: list[str],
    completions_per_probe: list[list[str]],
    judge_model: str,
) -> list[list[dict[str, Any]]]:
    """Score completions with Claude judge; returns parallel structure to inputs."""

    async def _runner():
        results = []
        for probe, comps in zip(probes, completions_per_probe, strict=True):
            results.append(await _judge_completions(probe, comps, judge_model))
        return results

    return asyncio.run(_runner())


# ---------------------------------------------------------------------------
# Bootstrap + selection helpers
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: list[float], n_resamples: int = 1000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)])
    if arr.size == 0:
        return (float("nan"), float("nan"))
    means = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def random_direction_baseline(d_model: int, n_random: int = 200, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    vs = rng.standard_normal((n_random, d_model)).astype(np.float32)
    vs /= np.linalg.norm(vs, axis=1, keepdims=True) + 1e-8
    # Pairwise cosines (upper triangle)
    cos = vs @ vs.T
    iu = np.triu_indices(n_random, k=1)
    cos_vals = cos[iu]
    return {
        "n_random": n_random,
        "d_model": d_model,
        "pairwise_cosine_mean": float(cos_vals.mean()),
        "pairwise_cosine_std": float(cos_vals.std()),
        "pairwise_cosine_p2_5": float(np.percentile(cos_vals, 2.5)),
        "pairwise_cosine_p97_5": float(np.percentile(cos_vals, 97.5)),
        "held_out_unit": vs[-1].tolist(),  # for trait z-score later
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare Chen et al. persona-vector extraction recipe to the project's "
            "centroid-difference recipe across five traits and a layer set."
        )
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument(
        "--traits",
        type=str,
        default=",".join(DEFAULT_TRAITS),
        help="Comma-separated trait names (must be keys of TRAIT_PERSONAS)",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=",".join(str(L) for L in DEFAULT_LAYERS),
        help="Comma-separated layer indices",
    )
    p.add_argument("--prompts-per-trait", type=int, default=200)
    p.add_argument(
        "--alpha-grid",
        type=str,
        default="-2,-1,-0.5,0,0.5,1,2",
        help="Comma-separated alpha multipliers (x ||v||_2) for the steering sweep",
    )
    p.add_argument(
        "--rubric",
        type=str,
        default="claude-sonnet",
        help="Judge identifier (claude-sonnet → uses DEFAULT_JUDGE_MODEL)",
    )
    p.add_argument("--calibration-split", type=int, default=25)
    p.add_argument("--report-split", type=int, default=25)
    p.add_argument("--output-dir", type=str, default="outputs/")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Smoke-test the orchestration on CPU with 1 trait, 1 layer, 1 probe, "
            "1 alpha. Does NOT load Qwen-7B and does NOT call the judge."
        ),
    )
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip Claude rubric calls (for offline reruns). Records empty scores.",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for the 95%% CI on rubric shift",
    )
    return p.parse_args()


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "run.log"),
        ],
    )


def _sweep_recipe_one_split(
    *,
    trait: str,
    recipe_name: str,
    vec: torch.Tensor,
    split_name: str,
    probes: list[str],
    alpha_grid: list[float],
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    max_new_tokens: int,
    judge_model: str,
    skip_judge: bool,
    rubric_csv_rows: list[dict[str, Any]],
) -> None:
    """Generate + score for one (trait, recipe, split) cell across alpha_grid."""
    for alpha in alpha_grid:
        completions = generate_steered_completions_hf(
            model=model,
            tokenizer=tokenizer,
            direction=vec,
            alpha=alpha,
            layer=PRIMARY_LAYER,
            probes=probes,
            max_new_tokens=max_new_tokens,
        )
        nlls = compute_mean_nll(model, tokenizer, probes, completions)
        gen_path = (
            output_dir
            / "generations"
            / recipe_name
            / trait
            / f"alpha_{alpha:+.2f}_{split_name}.jsonl"
        )
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gen_path, "w") as f:
            for probe, comp, nll in zip(probes, completions, nlls, strict=True):
                f.write(
                    json.dumps(
                        {
                            "probe": probe,
                            "completion": comp,
                            "nll": nll,
                            "alpha": alpha,
                            "trait": trait,
                            "recipe": recipe_name,
                            "split": split_name,
                        }
                    )
                    + "\n"
                )
        if skip_judge:
            scores = [{} for _ in completions]
        else:
            scored = judge_completions_sync(
                probes,
                [[c] for c in completions],
                judge_model=judge_model,
            )
            scores = [s[0] for s in scored]
        for probe_idx, (probe, _comp, nll, sc) in enumerate(
            zip(probes, completions, nlls, scores, strict=True)
        ):
            rubric_csv_rows.append(
                {
                    "trait": trait,
                    "recipe": recipe_name,
                    "alpha": alpha,
                    "split": split_name,
                    "layer": PRIMARY_LAYER,
                    "probe_idx": probe_idx,
                    "probe": probe,
                    "rubric_aligned": sc.get("aligned") if sc else None,
                    "rubric_coherent": sc.get("coherent") if sc else None,
                    "nll": nll,
                }
            )


def _run_sweep_for_trait(
    *,
    trait: str,
    chen_vectors: dict[str, torch.Tensor],
    centroid_vectors: dict[str, torch.Tensor],
    ablation_vectors_l20: dict[str, torch.Tensor],
    layers: list[int],
    alpha_grid: list[float],
    cal_probes: list[str],
    rep_probes: list[str],
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    max_new_tokens: int,
    judge_model: str,
    skip_judge: bool,
    rubric_csv_rows: list[dict[str, Any]],
) -> None:
    """Run cal+rep alpha sweep over (chen, centroid, centroid_on_completion)."""
    chen_v_primary = chen_vectors[trait][layers.index(PRIMARY_LAYER)]
    centroid_v_primary = centroid_vectors[trait][layers.index(PRIMARY_LAYER)]
    ablation_v_primary = ablation_vectors_l20.get(trait)

    recipes = [
        ("chen", chen_v_primary),
        ("centroid", centroid_v_primary),
        ("centroid_on_completion", ablation_v_primary),
    ]
    for recipe_name, vec in recipes:
        if vec is None:
            continue
        for split_name, probes in (("cal", cal_probes), ("rep", rep_probes)):
            _sweep_recipe_one_split(
                trait=trait,
                recipe_name=recipe_name,
                vec=vec,
                split_name=split_name,
                probes=probes,
                alpha_grid=alpha_grid,
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                max_new_tokens=max_new_tokens,
                judge_model=judge_model,
                skip_judge=skip_judge,
                rubric_csv_rows=rubric_csv_rows,
            )


def _select_alpha_star(
    cal_rows: list[dict[str, Any]],
    alpha_grid: list[float],
    base_aligned_mean: float | None,
    base_nll_mean: float | None,
) -> float | None:
    """Pick the alpha that maximizes rubric shift on calibration subject to NLL <= 1.5x."""
    if base_aligned_mean is None:
        return None
    best_alpha: float | None = None
    best_shift = -float("inf")
    for alpha in alpha_grid:
        cal_alpha = [r for r in cal_rows if r["alpha"] == alpha]
        aligned = [r["rubric_aligned"] for r in cal_alpha if r["rubric_aligned"] is not None]
        nlls = [r["nll"] for r in cal_alpha if r["nll"] is not None and not np.isnan(r["nll"])]
        if not aligned or not nlls:
            continue
        mean_aligned = float(np.mean(aligned))
        mean_nll = float(np.mean(nlls))
        if base_nll_mean is not None and mean_nll > 1.5 * base_nll_mean:
            continue
        shift = mean_aligned - base_aligned_mean
        if shift > best_shift:
            best_shift = shift
            best_alpha = alpha
    return best_alpha


def _baseline_means(
    trait: str,
    base_scores_by_trait: dict[str, list[dict[str, Any]]],
    base_nll_by_trait: dict[str, list[float]],
) -> tuple[float | None, float | None]:
    base_aligned = [s.get("aligned") for s in base_scores_by_trait[trait] if s]
    base_aligned_mean = (
        float(np.mean([v for v in base_aligned if v is not None]))
        if base_aligned and any(v is not None for v in base_aligned)
        else None
    )
    base_nll_mean = (
        float(np.mean([v for v in base_nll_by_trait[trait] if not np.isnan(v)]))
        if base_nll_by_trait[trait]
        else None
    )
    return base_aligned_mean, base_nll_mean


def _summarize_trait_at_primary_layer(
    *,
    trait: str,
    rubric_csv_rows: list[dict[str, Any]],
    base_scores_by_trait: dict[str, list[dict[str, Any]]],
    base_nll_by_trait: dict[str, list[float]],
    alpha_grid: list[float],
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Build the per-recipe summary at PRIMARY_LAYER for a single trait."""
    per_recipe_summary: dict[str, dict[str, Any]] = {}
    base_aligned_mean, base_nll_mean = _baseline_means(
        trait, base_scores_by_trait, base_nll_by_trait
    )
    for recipe_name in ("chen", "centroid", "centroid_on_completion"):
        cal_rows = [
            r
            for r in rubric_csv_rows
            if r["trait"] == trait and r["recipe"] == recipe_name and r["split"] == "cal"
        ]
        rep_rows = [
            r
            for r in rubric_csv_rows
            if r["trait"] == trait and r["recipe"] == recipe_name and r["split"] == "rep"
        ]
        if not cal_rows:
            continue

        best_alpha = _select_alpha_star(cal_rows, alpha_grid, base_aligned_mean, base_nll_mean)
        if best_alpha is None:
            per_recipe_summary[recipe_name] = {
                "alpha_star": None,
                "rubric_shift_at_alpha_star": None,
                "rubric_shift_ci": (None, None),
                "nll_ratio_at_alpha_star": None,
                "note": "no alpha satisfied NLL constraint with measurable shift",
            }
            continue

        rep_alpha = [r for r in rep_rows if r["alpha"] == best_alpha]
        rep_aligned = [r["rubric_aligned"] for r in rep_alpha if r["rubric_aligned"] is not None]
        ci = bootstrap_ci(
            [
                (r["rubric_aligned"] - base_aligned_mean)
                for r in rep_alpha
                if r["rubric_aligned"] is not None and base_aligned_mean is not None
            ],
            n_resamples=n_bootstrap,
            seed=seed,
        )
        shift = (
            float(np.mean(rep_aligned)) - base_aligned_mean
            if rep_aligned and base_aligned_mean is not None
            else None
        )
        rep_nlls = [r["nll"] for r in rep_alpha if r["nll"] is not None and not np.isnan(r["nll"])]
        nll_ratio = float(np.mean(rep_nlls)) / base_nll_mean if rep_nlls and base_nll_mean else None
        per_recipe_summary[recipe_name] = {
            "alpha_star": best_alpha,
            "rubric_shift_at_alpha_star": shift,
            "rubric_shift_ci": ci,
            "nll_ratio_at_alpha_star": nll_ratio,
        }
    return per_recipe_summary


def _trait_summary_dict(
    *,
    trait: str,
    chen_vectors: dict[str, torch.Tensor],
    centroid_vectors: dict[str, torch.Tensor],
    layers: list[int],
    held_out_unit: np.ndarray,
    per_recipe_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the per-trait section of summary.json."""
    cos_per_layer = {}
    for L_idx, L in enumerate(layers):
        cos_per_layer[str(L)] = _cos(
            chen_vectors[trait][L_idx].numpy(),
            centroid_vectors[trait][L_idx].numpy(),
        )
    if PRIMARY_LAYER in layers:
        L20_idx = layers.index(PRIMARY_LAYER)
        chen_cos_random = _cos(chen_vectors[trait][L20_idx].numpy(), held_out_unit)
        centroid_cos_random = _cos(centroid_vectors[trait][L20_idx].numpy(), held_out_unit)
    else:
        chen_cos_random = None
        centroid_cos_random = None
    return {
        "cos_chen_centroid_per_layer": cos_per_layer,
        "chen_cosine_to_random_unit_at_L20": chen_cos_random,
        "centroid_cosine_to_random_unit_at_L20": centroid_cos_random,
        "per_recipe_at_L20": per_recipe_summary,
    }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = a / (np.linalg.norm(a) + 1e-8)
    nb = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(na, nb))


def _build_steering_probes(extraction_probes: list[str], steering_probe_total: int) -> list[str]:
    """Build steering probe set, disjoint from extraction probes when possible."""
    if steering_probe_total > len(TRAIT_PROBE_POOL):
        logger.warning("Steering probes exceed pool size; will cycle to fill.")
    tail = (
        TRAIT_PROBE_POOL[len(extraction_probes) :]
        if len(extraction_probes) < len(TRAIT_PROBE_POOL)
        else []
    )
    if len(tail) >= steering_probe_total:
        return tail[:steering_probe_total]
    steering_probes: list[str] = []
    seen = set(extraction_probes)
    for x in TRAIT_PROBE_POOL * 4:
        if x in seen:
            continue
        steering_probes.append(x)
        if len(steering_probes) >= steering_probe_total:
            break
    if len(steering_probes) < steering_probe_total:
        logger.warning("Allowing overlap of extraction and steering probes")
        for x in TRAIT_PROBE_POOL * 4:
            steering_probes.append(x)
            if len(steering_probes) >= steering_probe_total:
                break
    return steering_probes


def _compute_baselines_unsteered(
    *,
    traits: list[str],
    rep_probes: list[str],
    model: Any,
    tokenizer: Any,
    d_model: int,
    max_new_tokens: int,
    judge_model: str,
    skip_judge: bool,
) -> tuple[dict[str, list[str]], dict[str, list[float]], dict[str, list[dict[str, Any]]]]:
    """Generate unsteered (alpha=0) completions, NLLs, and judge scores per trait."""
    base_completions: dict[str, list[str]] = {}
    base_scores: dict[str, list[dict[str, Any]]] = {}
    base_nlls: dict[str, list[float]] = {}
    for trait in traits:
        base_comps = generate_steered_completions_hf(
            model=model,
            tokenizer=tokenizer,
            direction=torch.zeros(d_model),
            alpha=0.0,
            layer=PRIMARY_LAYER,
            probes=rep_probes,
            max_new_tokens=max_new_tokens,
        )
        base_completions[trait] = base_comps
        base_nlls[trait] = compute_mean_nll(model, tokenizer, rep_probes, base_comps)
        if skip_judge:
            base_scores[trait] = [{} for _ in rep_probes]
        else:
            scored = judge_completions_sync(
                rep_probes,
                [[c] for c in base_comps],
                judge_model=judge_model,
            )
            base_scores[trait] = [s[0] for s in scored]
    return base_completions, base_nlls, base_scores


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    _setup_logging(output_dir)
    logger.info("run_chen_vs_centroid args=%s", vars(args))

    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    for t in traits:
        if t not in TRAIT_PERSONAS:
            raise SystemExit(f"Unknown trait {t!r}. Known: {sorted(TRAIT_PERSONAS)}")
    if PRIMARY_LAYER not in layers:
        logger.warning(
            "PRIMARY_LAYER=%d not in --layers; some checks will be skipped", PRIMARY_LAYER
        )

    # Probe sets
    extraction_probes = get_probe_prompts(args.prompts_per_trait)
    steering_probes = _build_steering_probes(
        extraction_probes, args.calibration_split + args.report_split
    )
    cal_probes = steering_probes[: args.calibration_split]
    rep_probes = steering_probes[
        args.calibration_split : args.calibration_split + args.report_split
    ]

    metadata = write_metadata(output_dir, args)
    metadata["resolved"] = {
        "traits": traits,
        "layers": layers,
        "alpha_grid": alpha_grid,
        "n_extraction_probes": len(extraction_probes),
        "n_calibration_probes": len(cal_probes),
        "n_reporting_probes": len(rep_probes),
        "primary_layer": PRIMARY_LAYER,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Always log the exact persona wording.
    persona_dump = serialize_trait_personas(traits)

    if args.dry_run:
        return _dry_run(
            args, traits, layers, alpha_grid, extraction_probes, persona_dump, output_dir
        )

    # ── Real run from here on. Heavy imports & model load. ────────────────
    import transformers  # noqa: F401  # implicit version logging via metadata
    from transformers import AutoModelForCausalLM, AutoTokenizer

    chen_cfg = ChenExtractionConfig(
        model_name=args.model,
        traits=traits,
        layers=layers,
        prompts_per_trait=args.prompts_per_trait,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    # ── Step 1: paired completions via vLLM ───────────────────────────────
    pairs_path = output_dir / "generations" / "chen_paired.json"
    if pairs_path.exists():
        from explore_persona_space.axis.chen_extract import load_paired_completions

        logger.info("Loading cached paired completions from %s", pairs_path)
        paired = load_paired_completions(pairs_path)
    else:
        paired = generate_paired_completions(chen_cfg, extraction_probes, gpu_id=args.gpu_id)
        save_paired_completions(paired, pairs_path)

    # ── Step 2: load HF model for forward-pass extraction ─────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Loading HF model %s on %s", args.model, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Step 3: Chen-style extraction ─────────────────────────────────────
    chen_dir = output_dir / "chen_vectors"
    chen_vectors = extract_chen_vectors(
        model=model,
        tokenizer=tokenizer,
        cfg=chen_cfg,
        paired_completions=paired,
        output_dir=chen_dir,
    )

    # ── Step 4: Centroid extraction ───────────────────────────────────────
    centroid_dir = output_dir / "centroid_vectors"
    centroid_vectors = extract_centroid_vectors(
        model=model,
        tokenizer=tokenizer,
        traits=traits,
        probe_prompts=extraction_probes,
        layers=layers,
        output_dir=centroid_dir,
    )

    # ── Step 5: Centroid-on-completion ablation at L20 ────────────────────
    if PRIMARY_LAYER in layers:
        ablation_vectors_l20 = extract_centroid_on_completion_at_layer(
            model=model,
            tokenizer=tokenizer,
            traits=traits,
            paired_completions=paired,
            layer=PRIMARY_LAYER,
        )
        torch.save(
            ablation_vectors_l20,
            output_dir / "centroid_on_completion_l20.pt",
        )
    else:
        ablation_vectors_l20 = {}

    # ── Step 6: Random baseline + cosines ─────────────────────────────────
    d_model = next(iter(chen_vectors.values())).shape[-1]
    rb = random_direction_baseline(d_model=d_model, seed=args.seed)
    with open(output_dir / "random_baseline.json", "w") as f:
        json.dump({k: v for k, v in rb.items() if k != "held_out_unit"}, f, indent=2)
    held_out_unit = np.asarray(rb["held_out_unit"], dtype=np.float32)

    # ── Step 7: alpha-sweep + rubric scoring ──────────────────────────────────
    from explore_persona_space.eval import DEFAULT_JUDGE_MODEL

    judge_model = DEFAULT_JUDGE_MODEL  # honors JUDGE_MODEL env var

    rubric_csv_rows: list[dict[str, Any]] = []
    summary_per_trait: dict[str, dict[str, Any]] = {}

    # Unsteered baseline (alpha=0) rubric and NLL per trait, on reporting probes.
    _base_completions, base_nll_by_trait, base_scores_by_trait = _compute_baselines_unsteered(
        traits=traits,
        rep_probes=rep_probes,
        model=model,
        tokenizer=tokenizer,
        d_model=d_model,
        max_new_tokens=args.max_new_tokens,
        judge_model=judge_model,
        skip_judge=args.skip_judge,
    )

    # alpha-sweep — delegated to helpers to keep main() complexity low.
    for trait in traits:
        _run_sweep_for_trait(
            trait=trait,
            chen_vectors=chen_vectors,
            centroid_vectors=centroid_vectors,
            ablation_vectors_l20=ablation_vectors_l20,
            layers=layers,
            alpha_grid=alpha_grid,
            cal_probes=cal_probes,
            rep_probes=rep_probes,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            judge_model=judge_model,
            skip_judge=args.skip_judge,
            rubric_csv_rows=rubric_csv_rows,
        )

        per_recipe_summary = _summarize_trait_at_primary_layer(
            trait=trait,
            rubric_csv_rows=rubric_csv_rows,
            base_scores_by_trait=base_scores_by_trait,
            base_nll_by_trait=base_nll_by_trait,
            alpha_grid=alpha_grid,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )

        summary_per_trait[trait] = _trait_summary_dict(
            trait=trait,
            chen_vectors=chen_vectors,
            centroid_vectors=centroid_vectors,
            layers=layers,
            held_out_unit=held_out_unit,
            per_recipe_summary=per_recipe_summary,
        )

    # Write CSV
    csv_path = output_dir / "rubric_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trait",
                "recipe",
                "alpha",
                "split",
                "layer",
                "probe_idx",
                "probe",
                "rubric_aligned",
                "rubric_coherent",
                "nll",
            ],
        )
        writer.writeheader()
        for row in rubric_csv_rows:
            writer.writerow(row)

    summary = {
        "metadata": metadata,
        "personas": persona_dump,
        "random_baseline": {k: v for k, v in rb.items() if k != "held_out_unit"},
        "per_trait": summary_per_trait,
        "kill_criterion": _evaluate_kill_criterion(summary_per_trait),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _write_clean_result_html(summary, traits, output_dir / "clean_result.html")

    logger.info("Done. outputs at %s", output_dir)
    return 0


def _evaluate_kill_criterion(summary_per_trait: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Hypothesis says Chen beats centroid on >=3 of 5 traits at L20."""
    n_better = 0
    by_trait: dict[str, str] = {}
    for trait, s in summary_per_trait.items():
        ch = s["per_recipe_at_L20"].get("chen", {}).get("rubric_shift_at_alpha_star")
        ce = s["per_recipe_at_L20"].get("centroid", {}).get("rubric_shift_at_alpha_star")
        if ch is None or ce is None:
            by_trait[trait] = "indeterminate"
            continue
        if ch > ce:
            n_better += 1
            by_trait[trait] = "chen_wins"
        else:
            by_trait[trait] = "centroid_wins_or_tie"
    return {
        "n_traits_chen_better": n_better,
        "n_traits_total": len(summary_per_trait),
        "hypothesis_supported": n_better >= 3,
        "by_trait": by_trait,
    }


def _write_clean_result_html(summary: dict[str, Any], traits: list[str], path: Path) -> None:
    """Emit a minimal clean-result HTML; the analyzer agent will rewrite later."""
    ts = datetime.datetime.now(datetime.UTC).isoformat()
    kc = summary.get("kill_criterion", {})

    rows = []
    for trait in traits:
        s = summary["per_trait"].get(trait, {})
        per = s.get("per_recipe_at_L20", {})
        chen = per.get("chen", {})
        centroid = per.get("centroid", {})
        rows.append(
            f"<tr><td>{trait}</td>"
            f"<td>{centroid.get('alpha_star')}</td>"
            f"<td>{_fmt(centroid.get('rubric_shift_at_alpha_star'))}"
            f" <span class='ci'>{_fmt_ci(centroid.get('rubric_shift_ci'))}</span></td>"
            f"<td>{chen.get('alpha_star')}</td>"
            f"<td>{_fmt(chen.get('rubric_shift_at_alpha_star'))}"
            f" <span class='ci'>{_fmt_ci(chen.get('rubric_shift_ci'))}</span></td>"
            f"<td>{_fmt(s.get('cos_chen_centroid_per_layer', {}).get(str(PRIMARY_LAYER)))}</td>"
            f"</tr>"
        )
    rows_html = "\n".join(rows)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Chen vs Centroid (#363)</title>
<style>
  body {{ font: 14px/1.5 system-ui, sans-serif; max-width: 920px; margin: 32px auto; color: #222; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; margin-top: 28px; }}
  table {{ border-collapse: collapse; margin: 8px 0 24px; }}
  th, td {{ padding: 6px 12px; border-bottom: 1px solid #ddd; text-align: left; }}
  .ci {{ color: #888; font-size: 12px; }}
  details {{ background: #f7f7f7; padding: 8px 12px; border-radius: 6px; }}
  summary {{ cursor: pointer; font-weight: 600; }}
</style></head>
<body>
<h1>Chen-style vs centroid persona vectors at layer {PRIMARY_LAYER}</h1>
<p><b>TL;DR.</b> At layer {PRIMARY_LAYER}, the Chen recipe produced a larger
mean rubric-shift than the project's centroid recipe on {kc.get("n_traits_chen_better", "?")}
of {kc.get("n_traits_total", "?")} traits. Hypothesis (>=3 of 5):
{"<b>supported</b>" if kc.get("hypothesis_supported") else "not supported"}.</p>

<table>
<thead>
  <tr><th>Trait</th>
    <th>Centroid alpha*</th><th>Centroid shift (95% CI)</th>
    <th>Chen alpha*</th><th>Chen shift (95% CI)</th>
    <th>cos(Chen, centroid) @ L{PRIMARY_LAYER}</th></tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>

<details>
<summary>Experimental design</summary>
<p>Model: Qwen2.5-7B-Instruct. Layers swept: see <code>summary.json</code>. Both
recipes were re-extracted on the same {len(traits) * len(summary.get("per_trait", {}))}
trait/prompt set. Steering is activation addition at L{PRIMARY_LAYER}:
<code>h ← h + alpha (v / ||v||)</code>. alpha* per (trait, recipe) is picked on a 25-prompt
calibration split that maximizes rubric shift subject to mean per-token NLL
<= 1.5x baseline; CI is bootstrapped (1000 resamples) on a disjoint 25-prompt
reporting split. Generated <code>{ts}</code>.</p>
</details>
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and np.isnan(v):
        return "—"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def _fmt_ci(ci: Any) -> str:
    if not ci or ci == (None, None):
        return ""
    lo, hi = ci
    if lo is None or hi is None:
        return ""
    if (isinstance(lo, float) and np.isnan(lo)) or (isinstance(hi, float) and np.isnan(hi)):
        return ""
    return f"[{lo:.2f}, {hi:.2f}]"


def _dry_run(
    args: argparse.Namespace,
    traits: list[str],
    layers: list[int],
    alpha_grid: list[float],
    extraction_probes: list[str],
    persona_dump: dict[str, dict[str, str]],
    output_dir: Path,
) -> int:
    """CPU-only smoke test that walks the orchestration without loading Qwen."""
    logger.info("DRY RUN — no model load, no judge calls. Output dir: %s", output_dir)
    traits = traits[:1]
    layers = [layers[0]] if layers else [10]
    alpha_grid = [alpha_grid[0]] if alpha_grid else [0.0]
    extraction_probes = extraction_probes[:1]

    # Fabricate a fake d_model and a fake Chen vector tensor.
    d_model = 8
    fake_vec = torch.randn(len(layers), d_model)
    # Save fake vectors to make sure the IO path works.
    chen_dir = output_dir / "chen_vectors"
    chen_dir.mkdir(parents=True, exist_ok=True)
    torch.save(fake_vec, chen_dir / f"{traits[0]}.pt")
    centroid_dir = output_dir / "centroid_vectors"
    centroid_dir.mkdir(parents=True, exist_ok=True)
    torch.save(fake_vec, centroid_dir / f"{traits[0]}.pt")

    # Random baseline.
    rb = random_direction_baseline(d_model=d_model, n_random=8, seed=args.seed)
    with open(output_dir / "random_baseline.json", "w") as f:
        json.dump({k: v for k, v in rb.items() if k != "held_out_unit"}, f, indent=2)

    summary = {
        "dry_run": True,
        "traits": traits,
        "layers": layers,
        "alpha_grid": alpha_grid,
        "personas": persona_dump,
        "extraction_probes_first_3": extraction_probes[:3],
        "random_baseline": {k: v for k, v in rb.items() if k != "held_out_unit"},
        "fake_chen_shape": list(fake_vec.shape),
        "config": vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    fake_kill_criterion = {
        "n_traits_chen_better": 0,
        "n_traits_total": 1,
        "hypothesis_supported": False,
        "by_trait": {traits[0]: "dry_run"},
    }
    fake_summary = {
        "kill_criterion": fake_kill_criterion,
        "per_trait": {traits[0]: {"per_recipe_at_L20": {}, "cos_chen_centroid_per_layer": {}}},
    }
    _write_clean_result_html(fake_summary, traits, output_dir / "clean_result.html")
    logger.info("Dry-run complete. Wiring OK.")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    logger.info("Total wall time: %.1fs", time.time() - t0)
    raise SystemExit(rc)
