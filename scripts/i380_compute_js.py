#!/usr/bin/env python3
"""Stage B of issue #380: JS-from-baseline + mean pairwise JS on the N=48 panel.

For each of 49 sources x 20 probes:
  1. Build a teacher-force batch (all 49 system prompts forcing the SAME
     response — the source's own greedy completion from Stage A).
  2. Run ``teacher_force_batch`` to get (49, response_len, vocab_size)
     log-softmax on CPU.
  3. **Predictor 1 (JS-from-baseline):** for each panel persona, compute
     ``compute_js_divergence(log_probs[p], log_probs[baseline])``. Average
     across 20 probes per persona → 48-vector.
  4. **Predictor 2 (mean pairwise JS):** stack the 48 panel log-probs,
     call ``compute_pairwise_divergences(..., kl_only=True, ...)`` per
     probe, aggregate via ``aggregate_divergence_matrices``. Reduce per
     source by mean / median / max.

Outputs (under ``eval_results/issue_380/``):
  - ``js_from_baseline.json``: ``{source: js_value, ...}`` + metadata
  - ``pairwise_js_matrix.npz``: ``prompts`` (48,), ``js`` (48,48 sym),
    ``per_question_js`` (20,48,48 raw)
  - ``pairwise_reductions.json``: ``{source: {mean, median, max}, ...}``

Mandatory pre-launch pilot (plan section 13b item 3):
  - 3-persona pilot on ``librarian, helpful_assistant, villain``.
  - Run BOTH ``kl_only=True`` and ``kl_only=False``; assert off-diagonal
    Spearman rho >= 0.95 between the two 3x3 matrices.
  - Verify pilot pairwise-JS off-diagonal max:min >= 3:1 (degenerate-panel
    guard).
  - Fail loudly with ``epm:failure v1, failure_class=data`` on either.

Usage:
    uv run python scripts/i380_compute_js.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
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

PILOT_SOURCES = ("librarian", "helpful_assistant", "villain")
PILOT_RATIO_THRESHOLD = 3.0
PILOT_RHO_THRESHOLD = 0.95

BASELINE_ID = "_baseline_content_free_instruction"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _build_log_probs_for_question(
    *,
    tokenizer,
    model,
    sysprompt_texts: list[str],
    question: str,
    response_text: str,
    tf_batch: int,
):
    """Teacher-force ``response_text`` under every system prompt for one question.

    Returns ``log_probs`` of shape ``(N, response_len, vocab_size)`` float32
    on CPU, or ``None`` if the response is empty or the input cannot be
    built.
    """
    from explore_persona_space.analysis.divergence import (
        build_teacher_force_inputs,
        teacher_force_batch,
    )

    if not response_text.strip():
        return None
    try:
        batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
            tokenizer=tokenizer,
            system_prompts=sysprompt_texts,
            question=question,
            response_text=response_text,
        )
    except ValueError as e:
        logger.warning("Teacher-force input build failed: %s", e)
        return None
    if response_len < 1:
        return None

    log_probs = teacher_force_batch(
        model=model,
        batch_inputs=batch_inputs,
        prompt_lengths=prompt_lengths,
        response_len=response_len,
        device="cuda:0",
        max_batch=tf_batch,
    )
    return log_probs


def run_pilot(
    *,
    tokenizer,
    model,
    sysprompts: dict[str, dict],
    generations: dict[str, dict[str, str]],
    questions: list[str],
    tf_batch: int,
) -> None:
    """3-persona pilot: kl_only=True vs kl_only=False + ratio gate.

    Raises ``SystemExit`` on failure with a message that names the
    ``epm:failure`` ``reason`` the orchestrator should post:

      - ``reason=kl_approx_drift`` if off-diagonal Spearman < 0.95.
      - ``reason=pairwise_js_degenerate`` if max:min ratio < 3:1.
    """
    import torch
    from scipy.stats import spearmanr

    from explore_persona_space.analysis.divergence import (
        aggregate_divergence_matrices,
        compute_pairwise_divergences,
    )

    missing = [s for s in PILOT_SOURCES if s not in sysprompts]
    if missing:
        raise SystemExit(
            f"Pilot sources missing from panel: {missing}. Cannot run mandatory smoke-test 3."
        )

    pilot_names = list(PILOT_SOURCES)
    pilot_texts = [sysprompts[name]["text"] for name in pilot_names]
    logger.info("Pilot: 3-persona smoke-test on %s", pilot_names)

    all_js_kl_approx: list[dict] = []
    all_kl_kl_approx: list[dict] = []
    all_js_exact: list[dict] = []
    all_kl_exact: list[dict] = []

    reference_src = pilot_names[0]
    for q_idx, question in enumerate(questions):
        resp = generations.get(reference_src, {}).get(question)
        if not resp or not resp.strip():
            logger.warning(
                "Pilot: missing/empty greedy response for %s on q_idx=%d; skipping.",
                reference_src,
                q_idx,
            )
            continue
        log_probs = _build_log_probs_for_question(
            tokenizer=tokenizer,
            model=model,
            sysprompt_texts=pilot_texts,
            question=question,
            response_text=resp,
            tf_batch=tf_batch,
        )
        if log_probs is None:
            continue

        js_kl, kl_kl = compute_pairwise_divergences(
            log_probs=log_probs,
            persona_names=pilot_names,
            kl_only=True,
            gpu_device="cuda:0",
            row_chunk=16,
            time_chunk=30,
        )
        js_ex, kl_ex = compute_pairwise_divergences(
            log_probs=log_probs,
            persona_names=pilot_names,
            kl_only=False,
            gpu_device="cuda:0",
            row_chunk=16,
            time_chunk=30,
        )
        all_js_kl_approx.append(js_kl)
        all_kl_kl_approx.append(kl_kl)
        all_js_exact.append(js_ex)
        all_kl_exact.append(kl_ex)

        del log_probs
        torch.cuda.empty_cache()

    if not all_js_kl_approx:
        raise SystemExit(
            "Pilot produced 0 valid (q, response) pairs. "
            "Post epm:failure v1 failure_class=data reason=pairwise_js_degenerate."
        )

    agg_kl = aggregate_divergence_matrices(all_js_kl_approx, all_kl_kl_approx, list(pilot_names))
    agg_ex = aggregate_divergence_matrices(all_js_exact, all_kl_exact, list(pilot_names))

    js_kl_matrix = np.asarray(agg_kl["js_matrix"], dtype=np.float64)
    js_ex_matrix = np.asarray(agg_ex["js_matrix"], dtype=np.float64)

    iu = np.triu_indices_from(js_kl_matrix, k=1)
    kl_off = js_kl_matrix[iu]
    ex_off = js_ex_matrix[iu]

    if kl_off.size < 2:
        raise SystemExit(
            "Pilot pairwise matrix too small to compute Spearman. "
            "Post epm:failure v1 failure_class=code reason=pilot_too_small."
        )

    rho_kl_vs_exact, _ = spearmanr(kl_off, ex_off)
    logger.info(
        "Pilot: off-diagonal Spearman rho(kl_only=True, kl_only=False) = %.4f",
        rho_kl_vs_exact,
    )
    if rho_kl_vs_exact < PILOT_RHO_THRESHOLD:
        raise SystemExit(
            f"Pilot kl_only=True vs exact JS Spearman rho={rho_kl_vs_exact:.4f} "
            f"< {PILOT_RHO_THRESHOLD}. "
            "Post epm:failure v1 failure_class=data reason=kl_approx_drift."
        )

    max_off = float(kl_off.max())
    min_off = float(kl_off.min())
    ratio = max_off / max(min_off, 1e-12)
    logger.info(
        "Pilot: kl_only=True off-diagonal max:min ratio = %.3f (max=%.4f, min=%.4f)",
        ratio,
        max_off,
        min_off,
    )
    if ratio < PILOT_RATIO_THRESHOLD or min_off <= 0.0:
        raise SystemExit(
            f"Pilot pairwise-JS off-diagonal max:min={ratio:.3f} < "
            f"{PILOT_RATIO_THRESHOLD}. "
            "Post epm:failure v1 failure_class=data reason=pairwise_js_degenerate."
        )

    logger.info(
        "Pilot smoke-test PASSED: rho=%.4f, max:min ratio=%.3f.",
        rho_kl_vs_exact,
        ratio,
    )


def _run_predictor1_for_question(
    *,
    q_idx,
    question,
    all_ids,
    sysprompt_texts_all,
    panel_id_to_index,
    generations,
    tokenizer,
    model,
    tf_batch,
    baseline_idx_in_all,
    n_all,
    compute_js_divergence_fn,
    torch_module,
    js_per_persona_q,
):
    """Per-question pass for Predictor 1 (JS-from-baseline).

    For every panel source, teacher-force ITS OWN greedy response under all
    49 prompts and compute ``JS(self_logits, baseline_logits)``. Writes
    into ``js_per_persona_q[panel_idx, q_idx]`` in place. Returns the total
    number of teacher-force passes performed for this question.
    """
    n_tf = 0
    for src_idx, src_id in enumerate(all_ids):
        if src_id == BASELINE_ID:
            continue
        response_text = generations.get(src_id, {}).get(question)
        if response_text is None:
            logger.warning("Missing greedy response for (src=%s, q_idx=%d)", src_id, q_idx)
            continue
        if not response_text.strip():
            logger.warning(
                "Empty greedy response for (src=%s, q_idx=%d), skipping",
                src_id,
                q_idx,
            )
            continue
        log_probs = _build_log_probs_for_question(
            tokenizer=tokenizer,
            model=model,
            sysprompt_texts=sysprompt_texts_all,
            question=question,
            response_text=response_text,
            tf_batch=tf_batch,
        )
        if log_probs is None:
            continue
        n_tf += n_all
        panel_idx = panel_id_to_index[src_id]
        js = compute_js_divergence_fn(log_probs[src_idx], log_probs[baseline_idx_in_all]).item()
        js_per_persona_q[panel_idx, q_idx] = float(js)
        del log_probs
        torch_module.cuda.empty_cache()
    return n_tf


def _run_predictor2_for_question(
    *,
    q_idx,
    question,
    panel_ids,
    panel_texts,
    panel_id_to_index,
    generations,
    tokenizer,
    model,
    tf_batch,
    row_chunk,
    time_chunk,
    n_panel,
    compute_pairwise_fn,
    torch_module,
    panel_js_dicts,
    panel_kl_dicts,
    per_question_js_panel,
):
    """Per-question pass for Predictor 2 (pairwise JS).

    Teacher-force the baseline's greedy response on this probe under all
    48 panel prompts, then compute the pairwise JS matrix. Returns the
    total number of teacher-force passes performed for this question.
    """
    baseline_resp = generations.get(BASELINE_ID, {}).get(question)
    if not baseline_resp or not baseline_resp.strip():
        logger.warning("Skipping pairwise for q_idx=%d: baseline response empty.", q_idx)
        return 0
    pairwise_log_probs = _build_log_probs_for_question(
        tokenizer=tokenizer,
        model=model,
        sysprompt_texts=panel_texts,
        question=question,
        response_text=baseline_resp,
        tf_batch=tf_batch,
    )
    if pairwise_log_probs is None:
        logger.warning("Skipping pairwise for q_idx=%d: input build failed.", q_idx)
        return 0
    js_pairs, kl_pairs = compute_pairwise_fn(
        log_probs=pairwise_log_probs,
        persona_names=panel_ids,
        kl_only=True,
        gpu_device="cuda:0",
        row_chunk=row_chunk,
        time_chunk=time_chunk,
    )
    panel_js_dicts.append(js_pairs)
    panel_kl_dicts.append(kl_pairs)

    m = np.zeros((n_panel, n_panel), dtype=np.float32)
    for (a, b), val in js_pairs.items():
        i, j = panel_id_to_index[a], panel_id_to_index[b]
        m[i, j] = val
        m[j, i] = val
    per_question_js_panel[q_idx] = m
    del pairwise_log_probs
    torch_module.cuda.empty_cache()
    return n_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--generations",
        type=str,
        default="eval_results/issue_380/base_model_generations.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="eval_results/issue_380",
    )
    parser.add_argument(
        "--tf-batch",
        type=int,
        default=8,
        help="Sub-batch size for teacher_force_batch (49 prompts in bf16).",
    )
    parser.add_argument(
        "--row-chunk",
        type=int,
        default=16,
        help="row_chunk forwarded to compute_pairwise_divergences",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=30,
        help="time_chunk forwarded to compute_pairwise_divergences",
    )
    parser.add_argument(
        "--skip-pilot",
        action="store_true",
        help="DEBUG ONLY: skip the mandatory 3-persona pilot pre-launch test.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    js_from_baseline_path = out_dir / "js_from_baseline.json"
    pairwise_npz_path = out_dir / "pairwise_js_matrix.npz"
    pairwise_reductions_path = out_dir / "pairwise_reductions.json"

    if (
        js_from_baseline_path.exists()
        and pairwise_npz_path.exists()
        and pairwise_reductions_path.exists()
    ):
        logger.info("All Stage B outputs already exist; skipping.")
        return

    gen_path = PROJECT_ROOT / args.generations
    if not gen_path.exists():
        raise FileNotFoundError(gen_path)
    logger.info("Loading generations from %s ...", gen_path)
    gen_data = json.loads(gen_path.read_text())
    system_prompts = gen_data["system_prompts"]
    questions = gen_data["questions"]
    generations = gen_data["generations"]

    sysprompts: dict[str, dict] = {sp["id"]: sp for sp in system_prompts}
    all_ids: list[str] = list(sysprompts.keys())
    if all_ids[0] != BASELINE_ID:
        raise ValueError(f"Expected baseline id {BASELINE_ID!r} at position 0; got {all_ids[0]!r}")
    panel_ids: list[str] = all_ids[1:]
    if len(panel_ids) != 48:
        raise ValueError(f"Expected 48 panel ids, got {len(panel_ids)}")
    logger.info(
        "Loaded %d system prompts (1 baseline + 48 panel), %d questions",
        len(all_ids),
        len(questions),
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.analysis.divergence import (
        aggregate_divergence_matrices,
        compute_js_divergence,
        compute_pairwise_divergences,
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
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    logger.info("Model loaded in %.1fs", time.time() - t_load)

    if args.skip_pilot:
        logger.warning("Skipping mandatory pilot smoke-test (--skip-pilot). DEBUG ONLY.")
    else:
        run_pilot(
            tokenizer=tokenizer,
            model=model,
            sysprompts=sysprompts,
            generations=generations,
            questions=questions,
            tf_batch=args.tf_batch,
        )

    n_all = len(all_ids)
    n_panel = len(panel_ids)
    n_q = len(questions)
    baseline_idx_in_all = 0

    panel_id_to_index = {pid: i for i, pid in enumerate(panel_ids)}

    # JS-from-baseline accumulator: js_per_persona_q[panel_idx][q_idx] = JS.
    js_per_persona_q = np.full((n_panel, n_q), np.nan, dtype=np.float64)

    # Pairwise per-question (48 by 48), via a SECOND teacher-force pass per
    # probe using the BASELINE's greedy response as the shared response.
    per_question_js_panel = np.full((n_q, n_panel, n_panel), np.nan, dtype=np.float32)
    panel_js_dicts: list[dict] = []
    panel_kl_dicts: list[dict] = []

    sysprompt_texts_all = [sysprompts[pid]["text"] for pid in all_ids]
    panel_texts = [sysprompts[pid]["text"] for pid in panel_ids]

    t_start = time.time()
    total_tf_passes = 0

    # Outer loop = questions, inner = sources (mirrors i207).
    for q_idx, question in enumerate(questions):
        t_q = time.time()

        n_tf_p1 = _run_predictor1_for_question(
            q_idx=q_idx,
            question=question,
            all_ids=all_ids,
            sysprompt_texts_all=sysprompt_texts_all,
            panel_id_to_index=panel_id_to_index,
            generations=generations,
            tokenizer=tokenizer,
            model=model,
            tf_batch=args.tf_batch,
            baseline_idx_in_all=baseline_idx_in_all,
            n_all=n_all,
            compute_js_divergence_fn=compute_js_divergence,
            torch_module=torch,
            js_per_persona_q=js_per_persona_q,
        )
        total_tf_passes += n_tf_p1

        n_tf_p2 = _run_predictor2_for_question(
            q_idx=q_idx,
            question=question,
            panel_ids=panel_ids,
            panel_texts=panel_texts,
            panel_id_to_index=panel_id_to_index,
            generations=generations,
            tokenizer=tokenizer,
            model=model,
            tf_batch=args.tf_batch,
            row_chunk=args.row_chunk,
            time_chunk=args.time_chunk,
            n_panel=n_panel,
            compute_pairwise_fn=compute_pairwise_divergences,
            torch_module=torch,
            panel_js_dicts=panel_js_dicts,
            panel_kl_dicts=panel_kl_dicts,
            per_question_js_panel=per_question_js_panel,
        )
        total_tf_passes += n_tf_p2

        elapsed_q = time.time() - t_q
        total_elapsed = time.time() - t_start
        eta_min = total_elapsed / (q_idx + 1) * (n_q - q_idx - 1) / 60
        logger.info(
            "q %d/%d done in %.1fs  (cum %.1f min, ETA %.1f min, tf_passes=%d)",
            q_idx + 1,
            n_q,
            elapsed_q,
            total_elapsed / 60,
            eta_min,
            total_tf_passes,
        )

    elapsed_total = time.time() - t_start

    # ── Aggregate Predictor 1: JS-from-baseline per persona ────────────────
    with np.errstate(all="ignore"):
        js_from_baseline_vec = np.nanmean(js_per_persona_q, axis=1)  # (48,)
    js_from_baseline_dict = {panel_ids[i]: float(js_from_baseline_vec[i]) for i in range(n_panel)}
    n_nan_p1 = int(np.isnan(js_from_baseline_vec).sum())
    if n_nan_p1:
        logger.warning("Predictor 1 has %d NaN sources (all-probe missing). Check logs.", n_nan_p1)

    # ── Aggregate Predictor 2: mean/median/max pairwise JS per persona ─────
    if not panel_js_dicts:
        raise SystemExit(
            "No valid pairwise probes computed. "
            "Post epm:failure v1 failure_class=data reason=pairwise_no_probes."
        )

    agg = aggregate_divergence_matrices(panel_js_dicts, panel_kl_dicts, list(panel_ids))
    js_matrix_panel = np.asarray(agg["js_matrix"], dtype=np.float32)  # (48, 48)

    off_diag_mask = ~np.eye(n_panel, dtype=bool)
    reductions: dict[str, dict[str, float]] = {}
    for i, name in enumerate(panel_ids):
        row = js_matrix_panel[i][off_diag_mask[i]]
        reductions[name] = {
            "mean": float(np.mean(row)),
            "median": float(np.median(row)),
            "max": float(np.max(row)),
        }

    metadata_common = {
        "base_model": BASE_MODEL,
        "n_sysprompts": n_all,
        "n_panel": n_panel,
        "n_questions": n_q,
        "tf_batch": args.tf_batch,
        "row_chunk": args.row_chunk,
        "time_chunk": args.time_chunk,
        "kl_only_for_pairwise": True,
        "elapsed_seconds": elapsed_total,
        "n_tf_passes": total_tf_passes,
        "baseline_id": BASELINE_ID,
        "git_commit": get_git_commit(),
        "computed_at": datetime.now(UTC).isoformat(),
        "python_version": sys.version.split()[0],
    }

    js_from_baseline_payload = {
        "predictor": "js_from_baseline",
        "values": js_from_baseline_dict,
        "n_nan_sources": n_nan_p1,
        "metadata": metadata_common,
    }
    js_from_baseline_path.write_text(json.dumps(js_from_baseline_payload, indent=2))
    logger.info("Saved %s", js_from_baseline_path)

    np.savez_compressed(
        pairwise_npz_path,
        prompts=np.array(panel_ids, dtype=object),
        js=js_matrix_panel,
        per_question_js=per_question_js_panel,
        metadata=json.dumps(metadata_common),
    )
    size_mb = pairwise_npz_path.stat().st_size / 1e6
    logger.info("Saved %s (size=%.2f MB)", pairwise_npz_path, size_mb)

    pairwise_reductions_payload = {
        "predictor": "mean_pairwise_js",
        "reductions": reductions,
        "metadata": metadata_common,
    }
    pairwise_reductions_path.write_text(json.dumps(pairwise_reductions_payload, indent=2))
    logger.info("Saved %s", pairwise_reductions_path)

    logger.info(
        "Stage B done in %.1f min (%d total TF passes).",
        elapsed_total / 60,
        total_tf_passes,
    )


if __name__ == "__main__":
    main()
