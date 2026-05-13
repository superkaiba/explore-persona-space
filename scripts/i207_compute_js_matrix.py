#!/usr/bin/env python3
"""Stage 5 of issue #343: 40×40 JS divergence matrix over base-model output distributions.

For each pair (i, j) of system prompts:
  1. Take prompt i's 20 greedy responses (from Stage 4).
  2. Teacher-force each response through base Qwen-2.5-7B-Instruct under
     system prompt j (HF Transformers, bf16, logits -> float32).
  3. Compute per-token JS divergence between log-softmax outputs under
     prompt i (self teacher-force) and prompt j.
  4. Average across 20 questions -> JS(i, j) for direction i->j.
  5. Symmetrize: JS_sym(i, j) = 0.5 * (JS(i, j) + JS(j, i)).

Reuses ``src/explore_persona_space/analysis/divergence.py`` for the math.

Output: ``eval_results/issue_207/js_gentle/js_divergence_matrix.npz`` with
keys: ``prompts`` (40 names), ``js`` (40,40 float32 symmetric matrix),
``per_question_js`` ((20, 40, 40) raw before symmetrize+average).

Usage:
    uv run python scripts/i207_compute_js_matrix.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--generations",
        type=str,
        default="eval_results/issue_207/js_gentle/base_model_generations.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="eval_results/issue_207/js_gentle/js_divergence_matrix.npz",
    )
    parser.add_argument(
        "--tf-batch",
        type=int,
        default=8,
        help="Sub-batch size for teacher_force_batch (40 prompts × bf16 model)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optionally cap question count (debug)",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.info("Output already exists at %s, skipping", out_path)
        return

    gen_path = PROJECT_ROOT / args.generations
    if not gen_path.exists():
        raise FileNotFoundError(gen_path)

    logger.info("Loading generations from %s ...", gen_path)
    gen_data = json.loads(gen_path.read_text())
    system_prompts = gen_data["system_prompts"]  # ordered list of 40 dicts
    questions = gen_data["questions"]
    generations = gen_data["generations"]

    if args.max_questions:
        questions = questions[: args.max_questions]
        logger.info("Capped to %d questions for debug", len(questions))

    prompt_ids = [sp["id"] for sp in system_prompts]
    prompt_texts = {sp["id"]: sp["text"] for sp in system_prompts}
    n = len(prompt_ids)
    n_q = len(questions)
    logger.info("n_prompts=%d, n_questions=%d", n, n_q)
    if n != 40:
        logger.warning("Expected 40 prompts, got %d", n)

    # Set up model + tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.analysis.divergence import (
        build_teacher_force_inputs,
        compute_js_divergence,
        teacher_force_batch,
    )

    logger.info("Loading tokenizer + model for %s on GPU %d ...", BASE_MODEL, args.gpu)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # CUDA_VISIBLE_DEVICES has remapped
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    logger.info("Model loaded in %.1fs", time.time() - t_load)

    # per_question_js[q_idx, i, j] = JS(log_probs_under_i, log_probs_under_j)
    #   where the response is prompt i's greedy response on question q.
    per_question_js = np.full((n_q, n, n), np.nan, dtype=np.float32)

    # Order of system prompts in the teacher-force batch (sorted by id stable index)
    sysprompt_text_list = [prompt_texts[pid] for pid in prompt_ids]

    t_start = time.time()
    n_tf_passes = 0  # rough counter
    for q_idx, question in enumerate(questions):
        t_q = time.time()
        for src_idx, src_id in enumerate(prompt_ids):
            response_text = generations.get(src_id, {}).get(question)
            if response_text is None:
                logger.warning(
                    "Missing greedy response for (sysprompt=%s, q_idx=%d)", src_id, q_idx
                )
                continue
            # Some empty responses (e.g. greedy refusal) — skip JS
            if not response_text.strip():
                logger.warning(
                    "Empty greedy response for (sysprompt=%s, q_idx=%d), filling NaN", src_id, q_idx
                )
                continue

            try:
                batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
                    tokenizer=tokenizer,
                    system_prompts=sysprompt_text_list,
                    question=question,
                    response_text=response_text,
                )
            except ValueError as e:
                logger.warning(
                    "Teacher-force input build failed for (src=%s, q_idx=%d): %s",
                    src_id,
                    q_idx,
                    e,
                )
                continue
            if response_len < 1:
                logger.warning(
                    "Zero-length response tokens for (src=%s, q_idx=%d), skipping",
                    src_id,
                    q_idx,
                )
                continue

            # Teacher-force: returns (n=40, response_len, V) log-softmax, CPU
            log_probs = teacher_force_batch(
                model=model,
                batch_inputs=batch_inputs,
                prompt_lengths=prompt_lengths,
                response_len=response_len,
                device="cuda:0",
                max_batch=args.tf_batch,
            )
            n_tf_passes += n

            # log_probs[src_idx] = self teacher-force (response under its source prompt)
            lp_self = log_probs[src_idx]  # (T, V) — the reference
            for tgt_idx in range(n):
                if tgt_idx == src_idx:
                    per_question_js[q_idx, src_idx, src_idx] = 0.0
                    continue
                js = compute_js_divergence(lp_self, log_probs[tgt_idx]).item()
                per_question_js[q_idx, src_idx, tgt_idx] = float(js)

            del log_probs, lp_self
            torch.cuda.empty_cache()

        elapsed_q = time.time() - t_q
        total_elapsed = time.time() - t_start
        eta_min = total_elapsed / (q_idx + 1) * (n_q - q_idx - 1) / 60
        logger.info(
            "Question %d/%d done in %.1fs  (cum %.1f min, ETA %.1f min, tf_passes=%d)",
            q_idx + 1,
            n_q,
            elapsed_q,
            total_elapsed / 60,
            eta_min,
            n_tf_passes,
        )

    # Average over questions (ignore NaN)
    with np.errstate(all="ignore"):
        js_direction = np.nanmean(per_question_js, axis=0)  # (n, n) directional
        js_sym = 0.5 * (js_direction + js_direction.T)

    # Replace NaN (no valid q for that cell) with 0 on diagonal else interpolate? Keep NaN.
    n_nan = int(np.isnan(js_sym).sum())
    if n_nan:
        logger.warning("%d NaN cells in symmetrized JS (%d total)", n_nan, n * n)

    metadata = {
        "n_prompts": n,
        "n_questions_used": n_q,
        "base_model": BASE_MODEL,
        "tf_batch": args.tf_batch,
        "computed_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "elapsed_seconds": time.time() - t_start,
        "n_tf_passes": n_tf_passes,
    }
    logger.info(
        "JS computation done in %.1f min (%d teacher-force forward passes)",
        (time.time() - t_start) / 60,
        n_tf_passes,
    )

    np.savez_compressed(
        out_path,
        prompts=np.array(prompt_ids, dtype=object),
        js=js_sym.astype(np.float32),
        js_direction=js_direction.astype(np.float32),
        per_question_js=per_question_js,
        metadata=json.dumps(metadata),
    )
    logger.info("Saved %s (size=%.1f MB)", out_path, out_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
