#!/usr/bin/env python3
"""GPU-backed recompute for task #396 predictors #1, #2, #3 (plan v2.3 §4.8 Phase E.3).

The analyzer (``scripts/analyze_issue396.py``) loads three base-model-derived
predictors against the headline DV:

* **Predictor #1 — cosine-to-assistant L15.** For each panel persona, the
  residual-stream hidden state at layer 15 at the last position of a
  neutral probe under that persona's system prompt, then per-persona
  cosine to the bare-assistant centroid (a single vector averaged over
  the 20 probe questions under "You are a helpful assistant.").

* **Predictor #2 — JS-to-baseline.** For each (persona, probe-question)
  cell, JS divergence of the next-token distribution under that persona
  vs the bare-assistant baseline, averaged over the 20 probe questions
  per persona.

* **Predictor #3 — pairwise output distance.** Per-persona mean JS
  divergence to every other panel persona (47 others), averaged over
  the 20 probe questions.

The three predictors share a single base-model forward pass: for each
probe question we (a) greedy-decode the BASE model's response under the
bare-assistant prompt to fix a shared response text, (b) teacher-force
that response under each of the 48 persona prompts + the bare-assistant
prompt, and (c) extract both the response-position log-prob distribution
(for #2 and #3) and the layer-15 hidden state at the last prompt position
(for #1). The cache JSON below stores the three per-persona scalars; the
analyzer loads them and graceful-degrades when the cache is absent.

**Cost on H100:**
  * One Qwen-2.5-7B-Instruct base-model load (~15 GB bf16).
  * 49 personas (48 panel + 1 baseline) x 20 probe questions = 980
    teacher-force forward passes, batched by ``--batch-size`` (default 16).
    At ~0.5 s per batch -> ~30 GPU-min total. Inside the H100 budget.
  * 20 greedy generations under the bare-assistant prompt (~5 s).

**Output:** ``eval_results/issue_396/base_model_predictors.json``
  schema:
    {
      "schema_version": 1,
      "n_personas": 48,
      "n_probe_questions": 20,
      "base_model": "Qwen/Qwen2.5-7B-Instruct",
      "hidden_layer_index": 15,
      "predictor_1_cosine_to_assistant_L15": {persona: float},
      "predictor_2_js_to_baseline": {persona: float},
      "predictor_3_pairwise_output_distance": {persona: float},
      "predictor_3_pairwise_js_matrix": {persona: {other_persona: float}},
      "metadata": {git_sha, timestamp_utc, ...}
    }

**Per-phase checkpoint discipline.** Predictor #1 (hidden states) and
predictors #2/#3 (output log-probs) share the SAME forward pass, so they
fall together. There is one phase (one model load, one sweep over the
49 prompt sets), with the cache JSON written at the end. A crash mid-
sweep loses everything in this phase — re-run from scratch. The analyzer
defaults to skipping the recompute when the cache exists, so the cost
is paid once per code commit.

Task #396 plan v2.3 §4.8 Phase E.3 + §A17.

Code-review v1 round 1 binding fix BF2 (2026-05-27): implements the
predictor recompute that was a stub in round 1.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_396"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ASSISTANT_BASELINE_PROMPT = "You are a helpful assistant."
HIDDEN_LAYER_INDEX = 15  # plan §4.8 / §5.1
DEFAULT_CACHE_PATH = EVAL_RESULTS_DIR / "base_model_predictors.json"
GREEDY_MAX_TOKENS = 256  # plain probe responses; not the marker eval


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_eval_personas() -> dict[str, str]:
    """48-persona prompt dict — mirrors launcher / eval / analyzer."""
    import importlib

    genleak = importlib.import_module("generate_leakage_data")
    genleak._activate_panel_48()
    panel = dict(genleak.PERSONAS)
    assert len(panel) == 48, f"expected 48 panel personas; got {len(panel)}"
    return panel


def _load_probe_questions() -> list[str]:
    from explore_persona_space.experiments.factor_screen_365 import EVAL_QUESTIONS_20

    qs = list(EVAL_QUESTIONS_20)
    assert len(qs) == 20, f"expected 20 probe questions; got {len(qs)}"
    return qs


def _greedy_baseline_responses(
    model,
    tokenizer,
    questions: list[str],
    *,
    device: str,
    max_new_tokens: int = GREEDY_MAX_TOKENS,
) -> list[str]:
    """Greedy-decode one base-model response per probe question under the bare
    assistant prompt. Used as the shared ``response_text`` for teacher-forcing
    every panel persona on the same question (mirrors i207_compute_js_matrix
    + leakage-experiment eval).

    HF ``model.generate`` is acceptable here — we need 20 short greedy
    completions ONCE, not a 960-cell sweep. vLLM would add a separate engine
    load on top of the already-loaded HF model with no throughput win.
    """
    import torch

    responses: list[str] = []
    model.eval()
    for q in questions:
        msgs = [
            {"role": "system", "content": ASSISTANT_BASELINE_PROMPT},
            {"role": "user", "content": q},
        ]
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # Drop the prompt prefix to recover the response only.
        gen_ids = output_ids[0, input_ids["input_ids"].shape[1] :]
        responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return responses


def _teacher_force_with_hidden_at_layer(
    model,
    tokenizer,
    system_prompts: list[str],
    persona_names: list[str],
    question: str,
    response_text: str,
    *,
    device: str,
    layer_index: int,
    max_batch: int,
):
    """One question's teacher-force pass across N system prompts.

    Returns
    -------
    log_probs : torch.Tensor on CPU, shape (N, response_len, vocab)
        Per-position log-softmax over the response tokens (for predictors
        #2 + #3). Mirrors ``analysis.divergence.teacher_force_batch`` but
        also extracts hidden states for predictor #1 in the same pass.
    hidden_last_prompt : torch.Tensor on CPU, shape (N, hidden_dim)
        Hidden state at ``layer_index`` (residual stream) at the LAST
        prompt-token position (i.e. the position right before the response
        starts) for each system prompt. Used by predictor #1.
    """
    import torch
    import torch.nn.functional as F

    from explore_persona_space.analysis.divergence import build_teacher_force_inputs

    batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
        tokenizer=tokenizer,
        system_prompts=system_prompts,
        question=question,
        response_text=response_text,
    )

    total_n = batch_inputs["input_ids"].shape[0]
    assert total_n == len(persona_names), (
        f"teacher-force build returned N={total_n} but {len(persona_names)} personas requested"
    )
    max_len = batch_inputs["input_ids"].shape[1]

    log_probs_chunks: list[torch.Tensor] = []
    hidden_chunks: list[torch.Tensor] = []

    for start in range(0, total_n, max_batch):
        end = min(start + max_batch, total_n)
        sub_input_ids = batch_inputs["input_ids"][start:end].to(device)
        sub_attn = batch_inputs["attention_mask"][start:end].to(device)
        sub_prompt_lengths = prompt_lengths[start:end]
        sub_bs = sub_input_ids.shape[0]

        with torch.no_grad():
            out = model(
                input_ids=sub_input_ids,
                attention_mask=sub_attn,
                output_hidden_states=True,
            )
        logits = out.logits
        # out.hidden_states is a tuple of length n_layers+1 (embedding +
        # one per transformer block). Index 15 selects post-block-14 output
        # = residual stream after the 15th block. Same convention as
        # measure_first_step_delta + plan §5.1.
        if layer_index >= len(out.hidden_states):
            raise IndexError(
                f"layer_index={layer_index} out of range; model has "
                f"{len(out.hidden_states)} hidden-state outputs (incl. embedding)"
            )
        hidden_layer = out.hidden_states[layer_index]  # (B, T, D)

        # Per-sequence extraction: in the LEFT-padded batch, the last prompt
        # token is at position (pad_len + prompt_length - 1). The first
        # response token is at (pad_len + prompt_length), so the LM head
        # logits for predicting the first response token live at logit index
        # (pad_len + prompt_length - 1). We slice both the logits and the
        # hidden state at that same boundary so #1 and #2/#3 see the SAME
        # context window.
        resp_logits_list = []
        hidden_last_list = []
        for i in range(sub_bs):
            pad_len = max_len - (sub_prompt_lengths[i] + response_len)
            resp_start = pad_len + sub_prompt_lengths[i]
            logit_start = resp_start - 1
            logit_end = resp_start + response_len - 1
            resp_logits_list.append(logits[i, logit_start:logit_end, :])
            # Hidden state at the LAST prompt-token position (the same one
            # whose logits would predict the first response token).
            hidden_last_list.append(hidden_layer[i, logit_start, :])

        resp_logits = torch.stack(resp_logits_list)  # (sub_bs, response_len, V)
        log_probs = F.log_softmax(resp_logits.float(), dim=-1)
        log_probs_chunks.append(log_probs.cpu())

        hidden_last = torch.stack(hidden_last_list)  # (sub_bs, D)
        hidden_chunks.append(hidden_last.float().cpu())

        del (
            out,
            logits,
            hidden_layer,
            resp_logits,
            log_probs,
            hidden_last,
            sub_input_ids,
            sub_attn,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(log_probs_chunks, dim=0), torch.cat(hidden_chunks, dim=0)


def _derive_predictors(
    *,
    all_names: list[str],
    per_question_hidden: list,
    per_question_log_probs: list,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    """Compute predictors #1/#2/#3 from cached per-question tensors.

    Extracted from ``compute_base_model_predictors`` so the heavy GPU sweep
    + the cheap CPU-bound derivation can be reviewed independently. The
    indexing convention is: ``all_names`` has the 48 panel personas first
    and ``"__assistant_baseline__"`` at the last index; ``per_question_*``
    items are aligned to that order on the leading "persona" axis.

    Returns ``(pred_1, pred_2, pred_3, pairwise_matrix)``:

    * ``pred_1[persona] = cos( mean_q hidden[persona, q], mean_q hidden[baseline, q] )``
    * ``pred_2[persona] = mean_q JS( log_probs[persona, q], log_probs[baseline, q] )``
    * ``pred_3[persona] = mean_{other in 47} pairwise_matrix[persona][other]``
    * ``pairwise_matrix[persona_a][persona_b] = mean_q JS(...)``
    """
    import torch

    from explore_persona_space.analysis.divergence import compute_pairwise_divergences

    baseline_idx = all_names.index("__assistant_baseline__")
    panel_indices = [i for i, name in enumerate(all_names) if name != "__assistant_baseline__"]
    n_total = len(all_names)

    # Predictor #1: cosine-to-assistant L15.
    # Per-persona vector = mean over q of hidden_last[persona_idx, q]. Then
    # the baseline centroid = mean over q of hidden_last[baseline_idx, q].
    # Per-persona predictor = cos(mean_q(hidden[p]), mean_q(hidden[baseline])).
    hidden_stack = torch.stack(per_question_hidden, dim=0)  # (n_q, n, D)
    hidden_mean_over_q = hidden_stack.mean(dim=0)  # (n, D)
    baseline_centroid = hidden_mean_over_q[baseline_idx]  # (D,)
    pred_1: dict[str, float] = {}
    for pidx in panel_indices:
        v = hidden_mean_over_q[pidx]
        cos = torch.nn.functional.cosine_similarity(v, baseline_centroid, dim=0).item()
        pred_1[all_names[pidx]] = float(cos)

    # Predictors #2 + #3 via existing GPU-batched compute_pairwise_divergences.
    # Earlier impl used a per-(i, j, q) CPU nested loop calling the unbatched
    # compute_js_divergence — 48 * 48 * 20 = 46k calls on full (T, V=152k)
    # tensors, ~hours on CPU; flagged in code-review round 2 (Codex). Fix:
    # for each question, call compute_pairwise_divergences once (chunked GPU
    # matmul, kl_only=True approximation, ~few seconds per call), accumulate
    # symmetric (persona_i, persona_j) JS into a sum matrix, then divide by
    # n_q at the end.
    js_sum = torch.zeros(n_total, n_total)
    js_count = torch.zeros(n_total, n_total)
    gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    for q_log_probs in per_question_log_probs:
        # q_log_probs: (n_total, response_len, V) on CPU
        js_pairs, _ = compute_pairwise_divergences(
            q_log_probs,
            all_names,
            kl_only=True,
            gpu_device=gpu,
        )
        for (a, b), v in js_pairs.items():
            i = all_names.index(a)
            j = all_names.index(b)
            js_sum[i, j] += v
            js_sum[j, i] += v  # symmetric
            js_count[i, j] += 1
            js_count[j, i] += 1
    # Mean over questions where the pair was observed (avoid div-by-zero).
    js_mean = torch.where(js_count > 0, js_sum / js_count.clamp(min=1), torch.zeros_like(js_sum))

    # Predictor #2: JS-to-baseline (mean over questions).
    pred_2: dict[str, float] = {
        all_names[pidx]: float(js_mean[pidx, baseline_idx].item()) for pidx in panel_indices
    }

    # Predictor #3: pairwise output distance among panel personas
    # (excludes the baseline column).
    pairwise_matrix: dict[str, dict[str, float]] = {}
    for i in panel_indices:
        pairwise_matrix[all_names[i]] = {}
        for j in panel_indices:
            pairwise_matrix[all_names[i]][all_names[j]] = (
                0.0 if i == j else float(js_mean[i, j].item())
            )
    pred_3: dict[str, float] = {
        name: float(sum(v for k, v in row.items() if k != name) / max(1, len(row) - 1))
        for name, row in pairwise_matrix.items()
    }
    return pred_1, pred_2, pred_3, pairwise_matrix


def compute_base_model_predictors(
    *,
    device: str = "cuda:0",
    layer_index: int = HIDDEN_LAYER_INDEX,
    batch_size: int = 16,
    cache_path: Path = DEFAULT_CACHE_PATH,
    overwrite: bool = False,
) -> dict:
    """Run the GPU recompute and write ``cache_path``. Returns the cache dict.

    Skips with a friendly warning + returns ``{}`` if CUDA isn't available
    or torch / transformers fail to import. The analyzer's stubs treat an
    empty return as "predictor missing" and continue with #4 + #5.
    """
    if cache_path.exists() and not overwrite:
        logger.info("Loading cached predictors from %s", cache_path)
        return json.loads(cache_path.read_text())

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Imported here only for the import-availability check; the actual
        # divergence computation happens in _derive_predictors which imports
        # its own reference. Keeping the import here means the early-exit path
        # (no divergence module) fires BEFORE the expensive model load.
        from explore_persona_space.analysis.divergence import (  # noqa: F401
            compute_pairwise_divergences,
        )
    except ImportError as e:
        logger.warning(
            "Cannot recompute predictors #1/#2/#3 — import failed: %s. "
            "Analyzer will report predictors #4 + #5 only.",
            e,
        )
        return {}

    if not torch.cuda.is_available():
        logger.warning(
            "No CUDA available — skipping predictor #1/#2/#3 recompute. "
            "Re-run on an H100 pod (see scripts/recompute_predictors_i396.py)."
        )
        return {}

    eval_personas = _load_eval_personas()
    questions = _load_probe_questions()

    panel_names = list(eval_personas.keys())
    panel_prompts = list(eval_personas.values())
    # Order is: 48 panel personas, then the assistant baseline prompt at index 48.
    all_names = [*panel_names, "__assistant_baseline__"]
    all_prompts = [*panel_prompts, ASSISTANT_BASELINE_PROMPT]
    n = len(all_names)

    logger.info(
        "Loading %s on %s (output_hidden_states for predictor #1 layer L%d)",
        BASE_MODEL,
        device,
        layer_index,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    logger.info("Base model loaded in %.1fs", time.time() - t_load)

    # ── Step 1: shared greedy baseline responses (one per probe question) ──
    logger.info("Generating %d shared baseline responses under bare assistant", len(questions))
    t0 = time.time()
    baseline_responses = _greedy_baseline_responses(
        model, tokenizer, questions, device=device, max_new_tokens=GREEDY_MAX_TOKENS
    )
    logger.info(
        "Baseline responses generated in %.1fs (lengths: %s)",
        time.time() - t0,
        [len(r.split()) for r in baseline_responses],
    )

    # ── Step 2: per-question teacher-force across (panel + baseline) ──
    # Allocate per-question result containers. We keep the per-question
    # log-prob tensors on CPU and free GPU memory after each question.
    per_question_log_probs: list[torch.Tensor] = []  # each (n, response_len, V) on CPU
    per_question_hidden: list[torch.Tensor] = []  # each (n, D) on CPU
    skipped_questions: list[int] = []

    for q_idx, (question, response) in enumerate(zip(questions, baseline_responses, strict=True)):
        if not response.strip():
            logger.warning("Question %d: empty baseline response — skipping", q_idx)
            skipped_questions.append(q_idx)
            continue
        t_q = time.time()
        try:
            log_probs, hidden_last = _teacher_force_with_hidden_at_layer(
                model,
                tokenizer,
                system_prompts=all_prompts,
                persona_names=all_names,
                question=question,
                response_text=response,
                device=device,
                layer_index=layer_index,
                max_batch=batch_size,
            )
        except ValueError as e:
            # build_teacher_force_inputs raises on ChatML response-token
            # mismatch across system prompts — rare on Qwen's chat template
            # but the helper guards against it.
            logger.warning("Question %d: teacher-force build failed (%s) — skipping", q_idx, e)
            skipped_questions.append(q_idx)
            continue
        per_question_log_probs.append(log_probs)
        per_question_hidden.append(hidden_last)
        elapsed = time.time() - t_q
        eta_min = elapsed * (len(questions) - q_idx - 1) / 60 if q_idx < len(questions) - 1 else 0
        logger.info(
            "Question %d/%d: %.1fs (response_len=%d, %d personas) | ETA %.1f min",
            q_idx + 1,
            len(questions),
            elapsed,
            log_probs.shape[1],
            n,
            eta_min,
        )

    if not per_question_log_probs:
        raise RuntimeError(
            "All probe questions failed teacher-force; cannot compute predictors. "
            "Check the base-model load and the chat-template tokenization."
        )

    # ── Step 3: derive the three predictors from cached tensors ──
    pred_1, pred_2, pred_3, pairwise_matrix = _derive_predictors(
        all_names=all_names,
        per_question_hidden=per_question_hidden,
        per_question_log_probs=per_question_log_probs,
    )

    n_personas_emitted = len(pred_1)  # 48 panel personas (baseline excluded by design)
    payload = {
        "schema_version": 1,
        "n_personas": n_personas_emitted,
        "n_probe_questions": len(per_question_log_probs),
        "n_skipped_questions": len(skipped_questions),
        "base_model": BASE_MODEL,
        "hidden_layer_index": layer_index,
        "assistant_baseline_prompt": ASSISTANT_BASELINE_PROMPT,
        "predictor_1_cosine_to_assistant_L15": pred_1,
        "predictor_2_js_to_baseline": pred_2,
        "predictor_3_pairwise_output_distance": pred_3,
        "predictor_3_pairwise_js_matrix": pairwise_matrix,
        "metadata": {
            "git_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "device": device,
            "batch_size": batch_size,
            "greedy_max_tokens": GREEDY_MAX_TOKENS,
            "skipped_question_indices": skipped_questions,
        },
    }
    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(
        "Wrote base-model predictor cache: %s (%d personas, %d questions)",
        cache_path,
        n_personas_emitted,
        len(per_question_log_probs),
    )

    # Release the base model so callers (e.g. analyzer running in the same
    # session) don't OOM downstream. Mirrors eval_issue396_logprob's
    # post-Phase-2 teardown but with the simpler "no vLLM workers to reap"
    # path because we never loaded vLLM in this script.
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        # psutil child reap is unnecessary here (no vLLM) but cheap; mirror
        # the eval-rig pattern in case a future code change adds a worker.
        import psutil

        for child in psutil.Process().children(recursive=True):
            with contextlib.suppress(psutil.NoSuchProcess):
                child.terminate()

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GPU recompute of task #396 base-model predictors #1, #2, #3"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--layer-index",
        type=int,
        default=HIDDEN_LAYER_INDEX,
        help="Hidden-layer index for predictor #1 (default L15 per plan §4.8/§5.1).",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Output JSON path. Default: eval_results/issue_396/base_model_predictors.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-run even if the cache JSON already exists.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    payload = compute_base_model_predictors(
        device=args.device,
        layer_index=args.layer_index,
        batch_size=args.batch_size,
        cache_path=args.cache_path,
        overwrite=args.overwrite,
    )
    if not payload:
        logger.error("Recompute returned empty — no CUDA or import failure")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
