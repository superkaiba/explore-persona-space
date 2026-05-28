#!/usr/bin/env python3
"""Base-model predictor extraction for task #415 — symmetric-baseline rerun.

Extends task #396's `recompute_predictors_i396.py` to compute four base-model
predictors against TWO reference states, on the 24 trained-source personas
from `INHERITED_SOURCES_24`:

  * **predictor_1_cosine_to_assistant_L15** — cosine sim of layer-15 residual
    stream against the bare-assistant baseline prompt (same as #396).
  * **predictor_2_js_to_assistant** — JS divergence of next-token distribution
    against the bare-assistant baseline prompt (renamed from #396's misleading
    `predictor_2_js_to_baseline` — same computation, honest name).
  * **predictor_4_cosine_to_neutral_L15** — cosine against the neutral baseline
    (no system message at all, model in pure instruct-mode default).
  * **predictor_5_js_to_neutral** — JS against the neutral baseline.

The question this addresses: in #396 both predictors referenced the same
bare-assistant prompt. The bare-assistant prompt is ITSELF a persona
statement ("You are a helpful assistant."). A truly neutral baseline (no
system message) might reveal signal that anchoring against another persona
hid. Falsification: if either neutral-baseline predictor clears the planned
|rho| >= 0.35 threshold against the headline DV from #396
(`logp_end_of_response_diagonal_mean`), the lineage's bare-assistant
reference was hiding a real signal.

Reuses everything else from #396: same 24 personas, same recipe, same DV,
same statistical scaffolding. Only Phase A (base-model predictor extraction)
runs anew. No retraining; no new eval beyond the predictor sweep.

**Output:** ``eval_results/issue_415/base_model_predictors_v2.json``
"""

from __future__ import annotations

import argparse
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
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_415"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ASSISTANT_BASELINE_PROMPT = "You are a helpful assistant."
HIDDEN_LAYER_INDEX = 15
DEFAULT_CACHE_PATH = EVAL_RESULTS_DIR / "base_model_predictors_v2.json"
GREEDY_MAX_TOKENS = 256

# Same 24 trained-source personas as #396.
INHERITED_SOURCES_24 = [
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
    "zelthari_scholar",
    "helpful_assistant",
    "qwen_default",
    "chef",
    "lawyer",
    "accountant",
    "journalist",
    "wizard",
    "hero",
    "philosopher",
    "child",
    "ai_assistant",
    "ai",
    "chatbot",
    "i_am_helpful",
]
assert len(INHERITED_SOURCES_24) == 24


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_eval_personas() -> dict[str, str]:
    """Load the 24 trained-source persona prompts.

    Source: scripts/analyze_length_rate_n48.py::get_inherited_prompt — pulls from
    HF training-data jsonl (`superkaiba1/explore-persona-space-data`), same lookup
    the #396 parent used. Each persona's prompt is whatever was injected in the
    `system` field of `marker_<source>_asst_excluded_medium.jsonl` rows.
    """
    from analyze_length_rate_n48 import get_inherited_prompt

    return {name: get_inherited_prompt(name) for name in INHERITED_SOURCES_24}


def _load_probe_questions() -> list[str]:
    """20 probe questions — same set #396 used (factor_screen_365 panel)."""
    from explore_persona_space.experiments.factor_screen_365 import EVAL_QUESTIONS_20

    qs = list(EVAL_QUESTIONS_20)
    assert len(qs) == 20, f"expected 20 probe questions; got {len(qs)}"
    return qs


def _greedy_baseline_responses(
    model, tokenizer, questions, *, device, max_new_tokens=GREEDY_MAX_TOKENS
):
    """Greedy-decode one base-model response per probe question under bare assistant.
    (Same response text is teacher-forced under every reference for fair comparison.)"""
    import torch

    responses = []
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
        gen_ids = output_ids[0, input_ids["input_ids"].shape[1] :]
        responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return responses


def _teacher_force_with_hidden_at_layer(
    model,
    tokenizer,
    system_prompts,
    persona_names,
    question,
    response_text,
    *,
    device,
    layer_index,
    max_batch,
):
    """Per-question teacher-force across N system prompts (None allowed for neutral)."""
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
    max_len = batch_inputs["input_ids"].shape[1]
    log_probs_chunks = []
    hidden_chunks = []
    for start in range(0, total_n, max_batch):
        end = min(start + max_batch, total_n)
        sub_input_ids = batch_inputs["input_ids"][start:end].to(device)
        sub_attn = batch_inputs["attention_mask"][start:end].to(device)
        sub_prompt_lengths = prompt_lengths[start:end]
        sub_bs = sub_input_ids.shape[0]
        with torch.no_grad():
            out = model(input_ids=sub_input_ids, attention_mask=sub_attn, output_hidden_states=True)
        logits = out.logits
        if layer_index >= len(out.hidden_states):
            raise IndexError(f"layer_index={layer_index} out of range")
        hidden_layer = out.hidden_states[layer_index]
        resp_logits_list = []
        hidden_last_list = []
        for i in range(sub_bs):
            pad_len = max_len - (sub_prompt_lengths[i] + response_len)
            resp_start = pad_len + sub_prompt_lengths[i]
            logit_start = resp_start - 1
            logit_end = resp_start + response_len - 1
            resp_logits_list.append(logits[i, logit_start:logit_end, :])
            hidden_last_list.append(hidden_layer[i, logit_start, :])
        resp_logits = torch.stack(resp_logits_list)
        log_probs = F.log_softmax(resp_logits.float(), dim=-1)
        log_probs_chunks.append(log_probs.cpu())
        hidden_last = torch.stack(hidden_last_list)
        hidden_chunks.append(hidden_last.float().cpu())
        del out, logits, hidden_layer, resp_logits, log_probs, hidden_last, sub_input_ids, sub_attn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(log_probs_chunks, dim=0), torch.cat(hidden_chunks, dim=0)


def _derive_four_predictors(*, all_names, per_question_hidden, per_question_log_probs):
    """Compute four predictors from cached per-question tensors.

    Order: 24 personas, then __assistant_baseline__, then __neutral_baseline__.
    """
    import torch

    from explore_persona_space.analysis.divergence import compute_pairwise_divergences

    asst_idx = all_names.index("__assistant_baseline__")
    neut_idx = all_names.index("__neutral_baseline__")
    panel_indices = [
        i
        for i, name in enumerate(all_names)
        if name not in ("__assistant_baseline__", "__neutral_baseline__")
    ]
    n_total = len(all_names)

    # ── Cosine predictors (against each baseline centroid) ──
    hidden_stack = torch.stack(per_question_hidden, dim=0)  # (n_q, n, D)
    hidden_mean_q = hidden_stack.mean(dim=0)  # (n, D)
    asst_centroid = hidden_mean_q[asst_idx]
    neut_centroid = hidden_mean_q[neut_idx]
    pred_1, pred_4 = {}, {}
    for pidx in panel_indices:
        v = hidden_mean_q[pidx]
        pred_1[all_names[pidx]] = float(
            torch.nn.functional.cosine_similarity(v, asst_centroid, dim=0).item()
        )
        pred_4[all_names[pidx]] = float(
            torch.nn.functional.cosine_similarity(v, neut_centroid, dim=0).item()
        )

    # ── JS predictors (one GPU-batched sweep, index two baseline columns) ──
    js_sum = torch.zeros(n_total, n_total)
    js_count = torch.zeros(n_total, n_total)
    gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    for q_log_probs in per_question_log_probs:
        js_pairs, _ = compute_pairwise_divergences(
            q_log_probs,
            all_names,
            kl_only=True,
            gpu_device=gpu,
        )
        for (a, b), v in js_pairs.items():
            i, j = all_names.index(a), all_names.index(b)
            js_sum[i, j] += v
            js_sum[j, i] += v
            js_count[i, j] += 1
            js_count[j, i] += 1
    js_mean = torch.where(js_count > 0, js_sum / js_count.clamp(min=1), torch.zeros_like(js_sum))

    pred_2 = {all_names[pidx]: float(js_mean[pidx, asst_idx].item()) for pidx in panel_indices}
    pred_5 = {all_names[pidx]: float(js_mean[pidx, neut_idx].item()) for pidx in panel_indices}
    return pred_1, pred_2, pred_4, pred_5


def compute_415_predictors(
    *, device="cuda:0", batch_size=16, cache_path=DEFAULT_CACHE_PATH, overwrite=False
):
    if cache_path.exists() and not overwrite:
        logger.info("Loading cached predictors from %s", cache_path)
        return json.loads(cache_path.read_text())

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for #415 predictor extraction")

    eval_personas = _load_eval_personas()
    questions = _load_probe_questions()
    panel_names = list(eval_personas.keys())
    panel_prompts = list(eval_personas.values())

    # Order: 24 panel, then assistant baseline, then neutral baseline (None).
    all_names = [*panel_names, "__assistant_baseline__", "__neutral_baseline__"]
    all_prompts = [*panel_prompts, ASSISTANT_BASELINE_PROMPT, None]
    n = len(all_names)
    logger.info("Personas: %d trained sources + 2 baselines = %d total", len(panel_names), n)

    logger.info("Loading %s on %s", BASE_MODEL, device)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    logger.info("Base model loaded in %.1fs", time.time() - t0)

    logger.info("Generating %d shared baseline responses", len(questions))
    t1 = time.time()
    baseline_responses = _greedy_baseline_responses(
        model, tokenizer, questions, device=device, max_new_tokens=GREEDY_MAX_TOKENS
    )
    logger.info("Baseline responses done in %.1fs", time.time() - t1)

    per_question_log_probs = []
    per_question_hidden = []
    skipped = []

    for q_idx, (question, response) in enumerate(zip(questions, baseline_responses, strict=True)):
        if not response.strip():
            logger.warning("Question %d: empty baseline response — skipping", q_idx)
            skipped.append(q_idx)
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
                layer_index=HIDDEN_LAYER_INDEX,
                max_batch=batch_size,
            )
        except ValueError as e:
            logger.warning("Question %d: teacher-force build failed (%s) — skipping", q_idx, e)
            skipped.append(q_idx)
            continue
        per_question_log_probs.append(log_probs)
        per_question_hidden.append(hidden_last)
        elapsed = time.time() - t_q
        eta_min = elapsed * (len(questions) - q_idx - 1) / 60
        logger.info(
            "Question %d/%d: %.1fs (resp_len=%d, n=%d) | ETA %.1f min",
            q_idx + 1,
            len(questions),
            elapsed,
            log_probs.shape[1],
            n,
            eta_min,
        )

    if not per_question_log_probs:
        raise RuntimeError("All questions failed; aborting")

    pred_1, pred_2, pred_4, pred_5 = _derive_four_predictors(
        all_names=all_names,
        per_question_hidden=per_question_hidden,
        per_question_log_probs=per_question_log_probs,
    )

    payload = {
        "schema_version": 1,
        "n_personas": len(pred_1),
        "n_probe_questions": len(per_question_log_probs),
        "n_skipped_questions": len(skipped),
        "base_model": BASE_MODEL,
        "hidden_layer_index": HIDDEN_LAYER_INDEX,
        "assistant_baseline_prompt": ASSISTANT_BASELINE_PROMPT,
        "neutral_baseline_prompt": None,
        "neutral_baseline_description": "No system message — model in pure instruct-mode default",
        "predictor_1_cosine_to_assistant_L15": pred_1,
        "predictor_2_js_to_assistant": pred_2,
        "predictor_4_cosine_to_neutral_L15": pred_4,
        "predictor_5_js_to_neutral": pred_5,
        "metadata": {
            "git_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "parent_task": 396,
        },
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", cache_path)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    compute_415_predictors(
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
