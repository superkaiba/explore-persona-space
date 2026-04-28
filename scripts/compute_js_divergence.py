"""Compute JS/KL divergence between persona-conditioned logit distributions.

Issue #140: KL/JS divergence as persona similarity metric.

Four phases:
  Phase 1: Generate greedy responses via vLLM (220 prompts)
  Phase 2: Teacher-force each response under all 11 system prompts via HF
  Phase 3: Analysis (consistency, discrimination, redundancy, leakage comparison)
  Phase 4: Visualization via paper_plots.py

Usage:
    uv run python scripts/compute_js_divergence.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output-dir eval_results/js_divergence \
        --figure-dir figures/js_divergence \
        --seed 42 --max-tokens 512
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy import stats

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explore_persona_space.analysis.divergence import (
    aggregate_divergence_matrices,
    build_teacher_force_inputs,
    compute_pairwise_divergences,
    teacher_force_batch,
)
from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.personas import ALL_EVAL_PERSONAS, EVAL_QUESTIONS, SHORT_NAMES
from explore_persona_space.utils import setup_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ── Phase 1: Generation ─────────────────────────────────────────────────────


def phase1_generate(
    model_path: str,
    output_dir: Path,
    seed: int,
    max_tokens: int,
) -> dict[str, dict[str, list[str]]]:
    """Generate greedy responses for all (persona, question) pairs via vLLM."""
    from explore_persona_space.eval.generation import generate_persona_completions

    gen_path = output_dir / "generations.json"
    if gen_path.exists():
        logger.info("Phase 1: Loading cached generations from %s", gen_path)
        with open(gen_path) as f:
            cached = json.load(f)
        # Validate
        total = sum(len(qs) for qs in cached.values())
        if total == len(ALL_EVAL_PERSONAS) * len(EVAL_QUESTIONS):
            logger.info("Phase 1: %d cached generations valid, skipping generation", total)
            return cached
        logger.warning("Phase 1: Cached generations incomplete (%d), regenerating", total)

    logger.info("Phase 1: Generating %d greedy responses via vLLM", 220)
    completions = generate_persona_completions(
        model_path=model_path,
        personas=ALL_EVAL_PERSONAS,
        questions=EVAL_QUESTIONS,
        num_completions=1,
        temperature=0.0,
        max_tokens=max_tokens,
        top_p=1.0,
        seed=seed,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )

    # Validate: all 220 responses non-empty
    total = 0
    lengths = []
    empty_count = 0
    for persona_name, qs in completions.items():
        for question, responses in qs.items():
            total += 1
            text = responses[0] if responses else ""
            if not text.strip():
                empty_count += 1
                logger.warning("Empty response: persona=%s, question=%s", persona_name, question)
            lengths.append(len(text.split()))

    logger.info(
        "Phase 1: Generated %d responses. Lengths: mean=%.1f, min=%d, max=%d. Empty: %d",
        total,
        np.mean(lengths),
        np.min(lengths),
        np.max(lengths),
        empty_count,
    )

    if empty_count > 0:
        raise ValueError(f"Phase 1 FAIL: {empty_count} empty responses out of {total}")
    if total != len(ALL_EVAL_PERSONAS) * len(EVAL_QUESTIONS):
        raise ValueError(f"Phase 1 FAIL: expected 220 responses, got {total}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(gen_path, "w") as f:
        json.dump(completions, f, indent=2)
    logger.info("Phase 1: Saved generations to %s", gen_path)

    return completions


# ── Phase 2: Teacher-forcing divergence ──────────────────────────────────────


def phase2_teacher_force(
    model_path: str,
    completions: dict[str, dict[str, list[str]]],
    output_dir: Path,
    device: str = "cuda:0",
) -> dict:
    """Teacher-force each response under all 11 system prompts, compute JS/KL."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    matrices_path = output_dir / "divergence_matrices.json"
    if matrices_path.exists():
        logger.info("Phase 2: Loading cached matrices from %s", matrices_path)
        with open(matrices_path) as f:
            cached = json.load(f)
        if cached.get("n_prompts", 0) == len(ALL_EVAL_PERSONAS) * len(EVAL_QUESTIONS):
            logger.info("Phase 2: Cached matrices valid, skipping")
            return cached
        logger.warning("Phase 2: Cached matrices incomplete, recomputing")

    persona_names = list(ALL_EVAL_PERSONAS.keys())
    system_prompts = [ALL_EVAL_PERSONAS[name] for name in persona_names]
    n_personas = len(persona_names)

    logger.info("Phase 2: Loading model %s for teacher-forcing", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    logger.info("Phase 2: Model loaded on %s", device)

    # Collect all (response_persona, question, response_text) triples
    prompts = []
    for resp_persona in persona_names:
        for question in EVAL_QUESTIONS:
            response_text = completions[resp_persona][question][0]
            prompts.append((resp_persona, question, response_text))

    logger.info(
        "Phase 2: Processing %d prompts x %d scoring personas = %d forward passes",
        len(prompts),
        n_personas,
        len(prompts) * n_personas,
    )

    all_js: list[dict[tuple[str, str], float]] = []
    all_kl: list[dict[tuple[str, str], float]] = []

    start_time = time.time()
    for batch_idx, (resp_persona, question, response_text) in enumerate(prompts):
        if batch_idx % 20 == 0:
            elapsed = time.time() - start_time
            rate = batch_idx / elapsed if elapsed > 0 else 0
            eta = (len(prompts) - batch_idx) / rate if rate > 0 else 0
            logger.info(
                "Phase 2: Batch %d/%d (%.1f batches/min, ETA %.1f min) | resp_persona=%s",
                batch_idx,
                len(prompts),
                rate * 60,
                eta / 60,
                resp_persona,
            )

        # Build teacher-forcing inputs for all 11 system prompts
        try:
            batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
                tokenizer, system_prompts, question, response_text
            )
        except ValueError as e:
            logger.error(
                "Tokenization error at batch %d (resp=%s, q=%s): %s",
                batch_idx,
                resp_persona,
                question[:40],
                e,
            )
            raise

        if response_len < 2:
            logger.warning(
                "Very short response (%d tokens) at batch %d, skipping",
                response_len,
                batch_idx,
            )
            continue

        # Forward pass
        log_probs = teacher_force_batch(
            model, batch_inputs, prompt_lengths, response_len, device=device
        )
        # log_probs shape: (11, response_len, vocab_size)

        # Compute pairwise divergences
        js_pairs, kl_pairs = compute_pairwise_divergences(log_probs, persona_names)

        all_js.append(js_pairs)
        all_kl.append(kl_pairs)

        # Free intermediate tensors
        del log_probs, batch_inputs
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    logger.info("Phase 2: Completed %d batches in %.1f min", len(prompts), elapsed / 60)

    # Clean up model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Aggregate into matrices
    result = aggregate_divergence_matrices(all_js, all_kl, persona_names)

    # Add metadata
    result["metadata"] = {
        "model": model_path,
        "git_commit": _git_hash(),
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_personas": n_personas,
        "n_questions": len(EVAL_QUESTIONS),
        "n_response_personas": len(persona_names),
        "wall_minutes": elapsed / 60,
        "device": device,
    }

    # Consistency checks
    _check_consistency(result, persona_names)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(matrices_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Phase 2: Saved divergence matrices to %s", matrices_path)

    return result


def _check_consistency(result: dict, persona_names: list[str]) -> None:
    """Run consistency checks on divergence matrices."""
    n = len(persona_names)
    js = result["js_matrix"]
    kl = result["kl_matrix"]

    # JS: symmetric
    max_asym = 0.0
    for i in range(n):
        for j in range(n):
            max_asym = max(max_asym, abs(js[i][j] - js[j][i]))
    logger.info("Consistency: JS max asymmetry = %.2e (should be < 1e-6)", max_asym)
    if max_asym > 1e-4:
        logger.warning("JS asymmetry %.2e exceeds threshold!", max_asym)

    # JS: diagonal = 0
    max_diag = max(abs(js[i][i]) for i in range(n))
    logger.info("Consistency: JS max diagonal = %.2e (should be 0)", max_diag)

    # JS: bounded [0, ln(2)]
    ln2 = math.log(2)
    js_flat = [js[i][j] for i in range(n) for j in range(n) if i != j]
    js_min, js_max = min(js_flat), max(js_flat)
    logger.info(
        "Consistency: JS range = [%.6f, %.6f] (should be in [0, %.4f])", js_min, js_max, ln2
    )
    if js_min < -1e-6:
        logger.warning("Negative JS values detected! min=%.6f", js_min)
    if js_max > ln2 + 1e-4:
        logger.warning("JS exceeds ln(2)! max=%.6f", js_max)

    # KL: non-negative
    kl_flat = [kl[i][j] for i in range(n) for j in range(n) if i != j]
    kl_min = min(kl_flat)
    logger.info("Consistency: KL min = %.6f (should be >= 0)", kl_min)
    if kl_min < -1e-6:
        logger.warning("Negative KL values detected! min=%.6f", kl_min)

    # No NaN/Inf
    all_vals = js_flat + kl_flat
    nan_count = sum(1 for v in all_vals if math.isnan(v) or math.isinf(v))
    if nan_count > 0:
        raise ValueError(f"Consistency FAIL: {nan_count} NaN/Inf values in divergence matrices")

    logger.info("Consistency checks passed")


# ── Phase 3: Analysis ────────────────────────────────────────────────────────


def phase3_analysis(
    result: dict,
    output_dir: Path,
) -> dict:
    """Run all analysis steps: discrimination, redundancy, leakage comparison."""
    persona_names = result["persona_names"]
    n = len(persona_names)
    js_matrix = result["js_matrix"]

    analysis = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_hash(),
    }

    # ── Step 1: Discrimination check (H1) ──
    # Extract 55 unique pairs from upper triangle
    js_pairs_flat = []
    pair_labels = []
    for i, j in combinations(range(n), 2):
        js_pairs_flat.append(js_matrix[i][j])
        pair_labels.append((persona_names[i], persona_names[j]))

    js_arr = np.array(js_pairs_flat)
    js_std = float(np.std(js_arr))
    js_mean = float(np.mean(js_arr))
    js_min = float(np.min(js_arr))
    js_max = float(np.max(js_arr))
    js_ratio = js_max / js_min if js_min > 1e-10 else float("inf")

    analysis["discrimination"] = {
        "n_pairs": len(js_pairs_flat),
        "js_mean": js_mean,
        "js_std": js_std,
        "js_min": js_min,
        "js_max": js_max,
        "js_ratio": js_ratio,
        "pass_std": js_std > 0.05,
        "pass_ratio": js_ratio > 3.0,
        "pass": js_std > 0.05 and js_ratio > 3.0,
    }
    logger.info(
        "H1 Discrimination: std=%.4f (>0.05? %s), ratio=%.2f (>3? %s) -> %s",
        js_std,
        js_std > 0.05,
        js_ratio,
        js_ratio > 3.0,
        "PASS" if analysis["discrimination"]["pass"] else "FAIL",
    )

    # ── Step 2: Redundancy check (H2) — JS vs cosine ──
    analysis["redundancy"] = _compute_redundancy(js_pairs_flat, pair_labels, persona_names)

    # ── Step 3: JS vs leakage — matched comparison ──
    analysis["leakage_comparison"] = _compute_leakage_comparison(
        js_matrix, persona_names, output_dir
    )

    # ── Step 4: KL asymmetry (exploratory) ──
    kl_matrix = result["kl_matrix"]
    analysis["kl_asymmetry"] = _compute_kl_asymmetry(kl_matrix, persona_names, output_dir)

    # ── Step 5: Per-question analysis ──
    analysis["per_question"] = _compute_per_question_analysis(result)

    # Save
    analysis_path = output_dir / "analysis_results.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info("Phase 3: Saved analysis to %s", analysis_path)

    return analysis


def _compute_redundancy(
    js_pairs: list[float],
    pair_labels: list[tuple[str, str]],
    persona_names: list[str],
) -> dict:
    """Check if JS is redundant with cosine similarity at layers 10, 15, 20, 25."""
    from explore_persona_space.analysis.representation_shift import (
        compute_cosine_matrix,
        extract_centroids,
    )

    # We need cosine similarity for the 11 core personas
    # Extract centroids from the base model
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    layers = [10, 15, 20, 25]

    logger.info("Redundancy check: extracting centroids for %d personas at layers %s", 11, layers)
    centroids, centroid_names = extract_centroids(
        model_path=model_path,
        personas=ALL_EVAL_PERSONAS,
        layers=layers,
    )

    # Build name-to-index mapping for centroid ordering
    centroid_idx = {name: i for i, name in enumerate(centroid_names)}

    redundancy = {"layers": {}, "pass": True}

    for layer in layers:
        cos_matrix = compute_cosine_matrix(centroids[layer], centering="global_mean")

        # Extract cosine for the same 55 pairs in the same order
        cos_pairs = []
        for a, b in pair_labels:
            i, j = centroid_idx[a], centroid_idx[b]
            cos_pairs.append(cos_matrix[i, j].item())

        # Spearman correlation: expect NEGATIVE (JS=divergence vs cosine=similarity)
        rho, p_val = stats.spearmanr(js_pairs, cos_pairs)
        pearson_r, pearson_p = stats.pearsonr(js_pairs, cos_pairs)

        redundancy["layers"][f"layer_{layer}"] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "abs_rho": float(abs(rho)),
            "redundant": abs(rho) > 0.80,
        }

        if abs(rho) > 0.80:
            redundancy["pass"] = False

        logger.info(
            "H2 Redundancy L%d: rho=%.4f (|rho|=%.4f, >0.80? %s), p=%.2e",
            layer,
            rho,
            abs(rho),
            abs(rho) > 0.80,
            p_val,
        )

    logger.info("H2 Redundancy overall: %s", "PASS" if redundancy["pass"] else "FAIL")
    return redundancy


def _compute_leakage_comparison(
    js_matrix: list[list[float]],
    persona_names: list[str],
    output_dir: Path,
) -> dict:
    """Compare JS-leakage vs cosine-leakage correlation on matched 50 pairs."""
    name_to_idx = {name: i for i, name in enumerate(persona_names)}

    # Load leakage data for 5 source personas
    sources = ["villain", "comedian", "assistant", "software_engineer", "kindergarten_teacher"]
    leakage_dir = Path("eval_results/single_token_100_persona")

    directed_pairs = []  # (source, target, leakage_rate, js_value)

    for source in sources:
        marker_path = leakage_dir / source / "marker_eval.json"
        if not marker_path.exists():
            logger.warning("Leakage data not found for source=%s at %s", source, marker_path)
            continue

        with open(marker_path) as f:
            marker_data = json.load(f)

        for target in persona_names:
            if target == source:
                continue
            if target not in marker_data:
                continue

            leakage_rate = marker_data[target]["rate"]
            src_idx = name_to_idx[source]
            tgt_idx = name_to_idx[target]
            js_val = js_matrix[src_idx][tgt_idx]

            directed_pairs.append(
                {
                    "source": source,
                    "target": target,
                    "leakage_rate": leakage_rate,
                    "js_value": js_val,
                }
            )

    if len(directed_pairs) < 10:
        logger.warning(
            "Only %d directed pairs found, skipping leakage comparison", len(directed_pairs)
        )
        return {"n_pairs": len(directed_pairs), "skipped": True}

    logger.info("Leakage comparison: %d directed pairs", len(directed_pairs))

    leakage_rates = [p["leakage_rate"] for p in directed_pairs]
    js_values = [p["js_value"] for p in directed_pairs]

    # JS-leakage correlation
    # JS is divergence (higher = more different), leakage should be lower for more
    # different personas. So expect NEGATIVE correlation.
    js_rho, js_p = stats.spearmanr(js_values, leakage_rates)
    js_pearson, js_pearson_p = stats.pearsonr(js_values, leakage_rates)

    result = {
        "n_pairs": len(directed_pairs),
        "js_leakage_spearman_rho": float(js_rho),
        "js_leakage_spearman_p": float(js_p),
        "js_leakage_pearson_r": float(js_pearson),
        "js_leakage_pearson_p": float(js_pearson_p),
        "directed_pairs": directed_pairs,
    }

    # Also compute matched cosine-leakage correlation on the SAME 50 pairs
    result["cosine_matched"] = _compute_matched_cosine_leakage(directed_pairs, persona_names)

    return result


def _compute_matched_cosine_leakage(
    directed_pairs: list[dict],
    persona_names: list[str],
) -> dict:
    """Compute cosine-leakage correlation on the same directed pairs as JS."""
    from explore_persona_space.analysis.representation_shift import (
        compute_cosine_matrix,
        extract_centroids,
    )

    model_path = "Qwen/Qwen2.5-7B-Instruct"
    layers = [10, 15, 20, 25]

    # Try to reuse centroids if they were already extracted
    # (In practice, this may re-extract if called after redundancy check cleaned up)
    centroids, centroid_names = extract_centroids(
        model_path=model_path,
        personas=ALL_EVAL_PERSONAS,
        layers=layers,
    )
    centroid_idx = {name: i for i, name in enumerate(centroid_names)}

    leakage_rates = [p["leakage_rate"] for p in directed_pairs]
    result = {}

    for layer in layers:
        cos_matrix = compute_cosine_matrix(centroids[layer], centering="global_mean")

        cos_values = []
        for pair in directed_pairs:
            src_idx = centroid_idx[pair["source"]]
            tgt_idx = centroid_idx[pair["target"]]
            cos_values.append(cos_matrix[src_idx, tgt_idx].item())

        rho, p_val = stats.spearmanr(cos_values, leakage_rates)
        pearson_r, pearson_p = stats.pearsonr(cos_values, leakage_rates)

        result[f"layer_{layer}"] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
        }

        logger.info(
            "Matched cosine-leakage L%d: rho=%.4f, p=%.2e",
            layer,
            rho,
            p_val,
        )

    return result


def _compute_kl_asymmetry(
    kl_matrix: list[list[float]],
    persona_names: list[str],
    output_dir: Path,
) -> dict:
    """Exploratory: compare KL asymmetry with directional leakage difference."""
    name_to_idx = {name: i for i, name in enumerate(persona_names)}
    sources = ["villain", "comedian", "assistant", "software_engineer", "kindergarten_teacher"]
    leakage_dir = Path("eval_results/single_token_100_persona")

    # Find bi-source pairs: (A, B) where both A->B and B->A leakage data exist
    bi_pairs = []
    for i, src_a in enumerate(sources):
        for src_b in sources[i + 1 :]:
            path_a = leakage_dir / src_a / "marker_eval.json"
            path_b = leakage_dir / src_b / "marker_eval.json"
            if not (path_a.exists() and path_b.exists()):
                continue

            with open(path_a) as f:
                data_a = json.load(f)
            with open(path_b) as f:
                data_b = json.load(f)

            if src_b in data_a and src_a in data_b:
                leak_a_to_b = data_a[src_b]["rate"]
                leak_b_to_a = data_b[src_a]["rate"]
                idx_a = name_to_idx[src_a]
                idx_b = name_to_idx[src_b]
                kl_a_to_b = kl_matrix[idx_a][idx_b]
                kl_b_to_a = kl_matrix[idx_b][idx_a]

                bi_pairs.append(
                    {
                        "persona_a": src_a,
                        "persona_b": src_b,
                        "kl_a_to_b": kl_a_to_b,
                        "kl_b_to_a": kl_b_to_a,
                        "kl_asym": abs(kl_a_to_b - kl_b_to_a),
                        "leak_a_to_b": leak_a_to_b,
                        "leak_b_to_a": leak_b_to_a,
                        "leak_asym": abs(leak_a_to_b - leak_b_to_a),
                    }
                )

    logger.info("KL asymmetry: %d bi-source pairs", len(bi_pairs))

    if len(bi_pairs) < 3:
        return {"n_pairs": len(bi_pairs), "skipped": True, "pairs": bi_pairs}

    kl_asym_vals = [p["kl_asym"] for p in bi_pairs]
    leak_asym_vals = [p["leak_asym"] for p in bi_pairs]

    rho, p_val = stats.spearmanr(kl_asym_vals, leak_asym_vals)

    return {
        "n_pairs": len(bi_pairs),
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "pairs": bi_pairs,
        "note": "n too small for reliable inference; report pattern only",
    }


def _compute_per_question_analysis(result: dict) -> dict:
    """Compute JS std across 55 pairs for each question."""
    per_prompt_js = result["per_prompt_js"]
    persona_names = result["persona_names"]
    n_questions = len(EVAL_QUESTIONS)
    n_personas = len(persona_names)

    # per_prompt_js is indexed by batch_idx = resp_persona_idx * n_questions + q_idx
    # Each entry has pair keys like "villain__comedian"
    question_stats = []
    for q_idx, question in enumerate(EVAL_QUESTIONS):
        # Collect JS values across all response personas for this question
        all_js_for_question = []
        for resp_idx in range(n_personas):
            batch_idx = resp_idx * n_questions + q_idx
            batch_key = str(batch_idx)
            if batch_key in per_prompt_js:
                all_js_for_question.extend(per_prompt_js[batch_key].values())

        if not all_js_for_question:
            question_stats.append(
                {
                    "question": question,
                    "n_values": 0,
                    "uninformative": True,
                }
            )
            continue

        arr = np.array(all_js_for_question)
        q_mean = float(np.mean(arr))
        q_std = float(np.std(arr))
        q_min = float(np.min(arr))
        q_max = float(np.max(arr))

        # Flag as uninformative if all values within 20% of each other
        if q_mean > 1e-10:
            cv = q_std / q_mean
            uninformative = cv < 0.2
        else:
            uninformative = True

        question_stats.append(
            {
                "question": question,
                "js_mean": q_mean,
                "js_std": q_std,
                "js_min": q_min,
                "js_max": q_max,
                "n_values": len(all_js_for_question),
                "uninformative": uninformative,
            }
        )

    n_uninformative = sum(1 for q in question_stats if q.get("uninformative", False))
    logger.info(
        "Per-question analysis: %d/%d questions flagged as uninformative",
        n_uninformative,
        n_questions,
    )

    return {
        "questions": question_stats,
        "n_uninformative": n_uninformative,
    }


# ── Phase 4: Visualization ──────────────────────────────────────────────────


def phase4_visualize(
    result: dict,
    analysis: dict,
    figure_dir: Path,
) -> None:
    """Generate all figures."""
    import matplotlib.pyplot as plt

    set_paper_style("neurips")
    figure_dir.mkdir(parents=True, exist_ok=True)

    persona_names = result["persona_names"]
    short = [SHORT_NAMES.get(p, p) for p in persona_names]

    # 1. JS heatmap
    _plot_js_heatmap(result, persona_names, short, figure_dir)

    # 2. JS vs cosine scatter
    if "redundancy" in analysis and "layers" in analysis["redundancy"]:
        _plot_js_vs_cosine(result, analysis, persona_names, figure_dir)

    # 3. JS vs leakage scatter
    if "leakage_comparison" in analysis and not analysis["leakage_comparison"].get("skipped"):
        _plot_js_vs_leakage(analysis, figure_dir)

    # 4. KL asymmetry heatmap
    _plot_kl_asymmetry(result, persona_names, short, figure_dir)

    plt.close("all")
    logger.info("Phase 4: All figures saved to %s", figure_dir)


def _plot_js_heatmap(result, persona_names, short_names, figure_dir):
    import matplotlib.pyplot as plt

    n = len(persona_names)
    js = np.array(result["js_matrix"])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(js, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title("Jensen-Shannon Divergence Between Personas")

    # Add value annotations
    for i in range(n):
        for j in range(n):
            val = js[i][j]
            color = "white" if val > js.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=ax, label="JS divergence (nats)", shrink=0.8)
    fig.tight_layout()
    savefig_paper(fig, "js_heatmap", dir=str(figure_dir))
    plt.close(fig)
    logger.info("Saved JS heatmap")


def _plot_js_vs_cosine(result, analysis, persona_names, figure_dir):
    import matplotlib.pyplot as plt

    n = len(persona_names)
    js_matrix = np.array(result["js_matrix"])

    # Find the best layer (highest |rho|)
    best_layer = None
    best_abs_rho = 0
    for layer_key, layer_data in analysis["redundancy"]["layers"].items():
        if layer_data["abs_rho"] > best_abs_rho:
            best_abs_rho = layer_data["abs_rho"]
            best_layer = layer_key

    if best_layer is None:
        logger.warning("No redundancy layers found, skipping JS vs cosine plot")
        return

    layer_num = int(best_layer.split("_")[1])
    rho = analysis["redundancy"]["layers"][best_layer]["spearman_rho"]
    p_val = analysis["redundancy"]["layers"][best_layer]["spearman_p"]

    # Re-extract cosine for the best layer
    from explore_persona_space.analysis.representation_shift import (
        compute_cosine_matrix,
        extract_centroids,
    )

    centroids, centroid_names = extract_centroids(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        personas=ALL_EVAL_PERSONAS,
        layers=[layer_num],
    )
    centroid_idx = {name: i for i, name in enumerate(centroid_names)}
    cos_matrix = compute_cosine_matrix(centroids[layer_num], centering="global_mean")

    # Extract 55 pairs
    js_vals = []
    cos_vals = []
    labels = []
    for i, j in combinations(range(n), 2):
        js_vals.append(js_matrix[i][j])
        ci = centroid_idx[persona_names[i]]
        cj = centroid_idx[persona_names[j]]
        cos_vals.append(cos_matrix[ci, cj].item())
        labels.append(
            f"{SHORT_NAMES.get(persona_names[i], persona_names[i])}-"
            f"{SHORT_NAMES.get(persona_names[j], persona_names[j])}"
        )

    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(
        cos_vals, js_vals, alpha=0.7, color=colors[0], s=30, edgecolors="white", linewidth=0.5
    )
    ax.set_xlabel(f"Cosine similarity (Layer {layer_num}, centered)")
    ax.set_ylabel("JS divergence (nats)")
    ax.set_title(f"JS vs Cosine (rho={rho:.3f}, p={p_val:.1e}, n=55)")

    # Add trend line
    z = np.polyfit(cos_vals, js_vals, 1)
    x_line = np.linspace(min(cos_vals), max(cos_vals), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color=colors[1], alpha=0.7, linewidth=1)

    fig.tight_layout()
    savefig_paper(fig, "js_vs_cosine_scatter", dir=str(figure_dir))
    plt.close(fig)
    logger.info("Saved JS vs cosine scatter")


def _plot_js_vs_leakage(analysis, figure_dir):
    import matplotlib.pyplot as plt

    lc = analysis["leakage_comparison"]
    pairs = lc["directed_pairs"]

    js_vals = [p["js_value"] for p in pairs]
    leak_vals = [p["leakage_rate"] for p in pairs]

    js_rho = lc["js_leakage_spearman_rho"]

    # Find best matched cosine layer
    best_cos_layer = None
    best_cos_rho = 0
    if "cosine_matched" in lc:
        for layer_key, layer_data in lc["cosine_matched"].items():
            if abs(layer_data["spearman_rho"]) > abs(best_cos_rho):
                best_cos_rho = layer_data["spearman_rho"]
                best_cos_layer = layer_key

    colors = paper_palette(3)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.scatter(
        js_vals,
        leak_vals,
        alpha=0.7,
        color=colors[0],
        s=30,
        label=f"JS (rho={js_rho:.3f})",
        edgecolors="white",
        linewidth=0.5,
    )

    ax.set_xlabel("JS divergence (nats)")
    ax.set_ylabel("Marker leakage rate")
    title = f"JS vs Leakage (n={len(pairs)} directed pairs)"
    if best_cos_layer:
        title += f"\nMatched cosine rho={best_cos_rho:.3f}"
    ax.set_title(title)
    ax.legend(fontsize=8)

    fig.tight_layout()
    savefig_paper(fig, "js_vs_leakage_scatter", dir=str(figure_dir))
    plt.close(fig)
    logger.info("Saved JS vs leakage scatter")


def _plot_kl_asymmetry(result, persona_names, short_names, figure_dir):
    import matplotlib.pyplot as plt

    n = len(persona_names)
    kl = np.array(result["kl_matrix"])
    kl_asym = kl - kl.T

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    # Use diverging colormap
    vmax = max(abs(kl_asym.min()), abs(kl_asym.max()))
    im = ax.imshow(kl_asym, cmap="RdBu_r", aspect="equal", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title("KL Asymmetry: KL(row||col) - KL(col||row)")

    fig.colorbar(im, ax=ax, label="KL asymmetry (nats)", shrink=0.8)
    fig.tight_layout()
    savefig_paper(fig, "kl_asymmetry_heatmap", dir=str(figure_dir))
    plt.close(fig)
    logger.info("Saved KL asymmetry heatmap")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Compute JS/KL divergence between personas")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model path")
    parser.add_argument(
        "--output-dir", default="eval_results/js_divergence", help="Output directory"
    )
    parser.add_argument("--figure-dir", default="figures/js_divergence", help="Figure directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for generation")
    parser.add_argument("--device", default="cuda:0", help="Device for teacher-forcing")
    parser.add_argument("--skip-generation", action="store_true", help="Skip Phase 1 (use cached)")
    parser.add_argument(
        "--skip-teacher-force", action="store_true", help="Skip Phase 2 (use cached)"
    )
    args = parser.parse_args()

    setup_env()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Issue #140: JS/KL Divergence as Persona Similarity Metric")
    logger.info("Model: %s", args.model)
    logger.info("Output: %s", output_dir)
    logger.info("Device: %s", args.device)
    logger.info("Seed: %d", args.seed)
    logger.info("=" * 60)

    total_start = time.time()

    # Phase 1: Generation
    logger.info("\n=== PHASE 1: Generation ===")
    if args.skip_generation:
        gen_path = output_dir / "generations.json"
        if not gen_path.exists():
            raise FileNotFoundError(f"--skip-generation but no cached file at {gen_path}")
        with open(gen_path) as f:
            completions = json.load(f)
        logger.info("Phase 1: Loaded cached generations")
    else:
        completions = phase1_generate(args.model, output_dir, args.seed, args.max_tokens)

    # Phase 2: Teacher-forcing
    logger.info("\n=== PHASE 2: Teacher-forcing divergence ===")
    if args.skip_teacher_force:
        matrices_path = output_dir / "divergence_matrices.json"
        if not matrices_path.exists():
            raise FileNotFoundError(f"--skip-teacher-force but no cached file at {matrices_path}")
        with open(matrices_path) as f:
            divergence_result = json.load(f)
        logger.info("Phase 2: Loaded cached matrices")
    else:
        divergence_result = phase2_teacher_force(
            args.model, completions, output_dir, device=args.device
        )

    # Phase 3: Analysis
    logger.info("\n=== PHASE 3: Analysis ===")
    analysis = phase3_analysis(divergence_result, output_dir)

    # Phase 4: Visualization
    logger.info("\n=== PHASE 4: Visualization ===")
    phase4_visualize(divergence_result, analysis, figure_dir)

    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE in %.1f min", total_elapsed / 60)
    logger.info("Results: %s", output_dir)
    logger.info("Figures: %s", figure_dir)

    # Summary
    disc = analysis.get("discrimination", {})
    logger.info(
        "H1 Discrimination: %s (std=%.4f, ratio=%.2f)",
        "PASS" if disc.get("pass") else "FAIL",
        disc.get("js_std", 0),
        disc.get("js_ratio", 0),
    )

    red = analysis.get("redundancy", {})
    logger.info("H2 Redundancy: %s", "PASS" if red.get("pass") else "FAIL")
    for lk, lv in red.get("layers", {}).items():
        logger.info("  %s: |rho|=%.4f", lk, lv.get("abs_rho", 0))

    lc = analysis.get("leakage_comparison", {})
    if not lc.get("skipped"):
        logger.info(
            "H3 JS-leakage rho=%.4f (n=%d)",
            lc.get("js_leakage_spearman_rho", 0),
            lc.get("n_pairs", 0),
        )
        if "cosine_matched" in lc:
            for lk, lv in lc["cosine_matched"].items():
                logger.info("  Matched cosine %s: rho=%.4f", lk, lv.get("spearman_rho", 0))

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
