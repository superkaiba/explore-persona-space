# em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #448 v5 Phase 4 — per-cell on-policy marker-log-prob eval rig.

Plan §4.6 — REWRITTEN from the v1-v4 HF-Transformers
``compute_marker_logprob`` path to a vLLM ``prompt_logprobs=1`` rig
matching #460 §4.6.

For each (panel_persona, q) in EVAL_PERSONAS_24 x EVAL_QUESTIONS:
  1. ``prompt_text = apply_chat_template([{system: panel_persona_prompt},
                                          {user: q}], add_generation_prompt=True)``
  2. ``R_text = R_eval[panel_persona][q]["response_text"]`` (frozen content-
     hashed on-policy R from Phase 1 r_generate).
  3. ``full_ids = tokenizer.encode(prompt_text + R_text + SEP + MARKER_TEXT,
                                   add_special_tokens=False)`` where
     ``SEP = "\\n\\n"`` — IDENTICAL to the assistant content training emits
     (``f"{R_text}\\n\\n{MARKER_TEXT}"`` per build_training_data.py:299).
     Without the ``\\n\\n`` separator, BPE merges ``R_text``'s trailing
     punctuation directly with the marker (e.g. ``'.'`` + ``' ※'``) instead
     of with the separator (``'.\\n\\n'`` fused id 382 + ``' ※'``); marker
     would be scored after a context the trained model never saw. Round-2
     blocker C1 (Claude + Codex code-review v1).
  4. ``slot = len(full_ids) - 1`` — assert ``full_ids[slot] == 83399`` and
     ``full_ids.count(83399) == 1`` (off-by-one + BPE-resegmentation guard).
     Additionally assert (a) ``full_ids[:len(prompt_ids)] == prompt_ids``
     (prompt prefix intact) AND (b) the K = MARKER_PRECEDING_K_TOKENS
     immediately before slot match the same K tokens that training would
     produce for ``f"{R_text}\\n\\n{MARKER_TEXT}"`` rendered inside a chat-
     template assistant turn — the train-vs-eval token-equality contract.
  5. Build vLLM ``prompts_payload = [{"prompt_token_ids": full_ids}, ...]``.
  6. Run trained pass with ``lora_request=LoRARequest(cell_slug, 1,
     adapter_dir)``; run base pass with ``lora_request=None`` on the SAME
     prompts.
  7. Read ``out.prompt_logprobs[slot]`` for trained and base; extract the
     log-prob of token id 83399 (fail loud if missing).
  8. ``delta_g[p, q] = trained_logp - base_logp``. Emission recompute rate
     = ``argmax @ slot == 83399`` per cell.

Two off-by-one guards (in addition to the assertion at step 4):
  - LOGP_FLOOR = -50.0; if more than 1% of cells clamp, log a loud warning.
  - Diagonal implant gate (downstream): for every cell, ΔG on the source
    persona's own R must exceed +5 nats (H3). Not enforced here directly
    (different code path — uses villain's R_eval).

Outputs (per cell):
  ``eval_results/issue_448_v5/<cell>/marker_logprob.json``
    {
      "schema_version": "i448_v5",
      "cell": "<slug>",
      "adapter_path": "<local merged dir or HF adapter dir>",
      "marker_text": " ※",
      "marker_token_id": 83399,
      "n_cells_evaluated": <n_panel * n_q>,
      "logp_floor": -50.0,
      "g_logprob_per_persona_q": { "<persona>": { "<q_idx>": float, ... }, ... },
      "b_logprob_per_persona_q": { "<persona>": { "<q_idx>": float, ... }, ... },
      "delta_g_per_persona_q": { "<persona>": { "<q_idx>": float, ... }, ... },
      "emission_recompute_per_persona_q": { "<persona>": { "<q_idx>": bool, ... }, ... },
      "mean_per_persona_g_logprob": { "<persona>": float, ... },
      "mean_per_persona_b_logprob": { "<persona>": float, ... },
      "mean_per_persona_delta_g": { "<persona>": float, ... },
      "mean_per_persona_emission_rate": { "<persona>": float, ... },
      "r_eval_content_hash": "<sha256>",
      "git_commit_sha": "...",
      "timestamp_utc": "..."
    }

Subprocess isolation: this script is INTENDED to be invoked as a fresh
subprocess by the dispatcher (per CLAUDE.md vLLM teardown gotcha). It owns
its own vLLM engine; on exit the OS reaps vLLM workers before the next
phase loads weights.

CPU/GPU: GPU only (vLLM prompt_logprobs forward pass).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()


LOGP_FLOOR = -50.0  # plan §4.6 off-by-one guard 5.
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_GPU_MEM_UTIL = 0.85
DEFAULT_SEED = 42
DEFAULT_MAX_LORA_RANK = 32
SCHEMA_VERSION = "i448_v5"

# Marker separator — IDENTICAL to the assistant content training emits in
# build_training_data.py:299 (``f"{r_text}\n\n{marker_text}"``). The text-
# level concat (rather than token-id splice) is intentional: the marker
# is preceded by a BPE-fused token like ``'.\n\n'`` (id 382) that only
# materialises when the WHOLE string is encoded together, NOT when we
# splice ``response_token_ids + [sep_id] + [marker_id]``. Round-2 C1
# (Claude + Codex code-review v1).
MARKER_SEP = "\n\n"
# Number of tokens immediately before the marker slot that MUST be byte-
# identical between train and eval (the train-vs-eval token-equality
# contract). Set to 2 to cover the fused separator + last R sub-token,
# which is the largest BPE-merge boundary in this construct.
MARKER_PRECEDING_K_TOKENS = 2

log = logging.getLogger("issue_448.eval_one_cell")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _assert_marker_token(tokenizer, marker_text: str, expected_id: int) -> None:
    ids = tokenizer.encode(marker_text, add_special_tokens=False)
    if ids != [expected_id]:
        raise RuntimeError(
            f"Marker tokenization mismatch. Expected MARKER_TEXT={marker_text!r} "
            f"to encode to [{expected_id}]; got {ids}."
        )


def build_train_equivalent_full_ids(
    tokenizer,
    persona_prompt: str | None,
    question: str,
    R_text: str,
    marker_text: str,
    marker_id: int,
    sep: str = MARKER_SEP,
) -> list[int]:
    """Build the EXACT token-id sequence training emits, up to and including
    the marker token.

    Training (TRL SFTTrainer + ``apply_chat_template``) renders the
    assistant content as ``f"{R_text}{sep}{marker_text}"`` inside the full
    chat-template wrapper, then tokenizes the whole string. To compute
    log P(marker | trained context) at the SAME slot the trained model was
    optimized on, eval must produce the same token-id prefix up to (and
    including) the marker. This helper is used both as the eval prefix
    builder AND as the ground-truth reference in the token-equality
    assertion.
    """
    if persona_prompt is None:
        prompt_msgs = [{"role": "user", "content": question}]
    else:
        prompt_msgs = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": question},
        ]
    completion_msgs = [
        {"role": "assistant", "content": f"{R_text}{sep}{marker_text}"},
    ]
    full_train_text = tokenizer.apply_chat_template(prompt_msgs + completion_msgs, tokenize=False)
    full_train_ids = tokenizer.encode(full_train_text, add_special_tokens=False)
    if marker_id not in full_train_ids:
        raise RuntimeError(
            f"train-equivalent encoding missing marker_id={marker_id}; "
            f"R_text={R_text[:40]!r} (truncated)."
        )
    last_marker_pos = max(i for i, t in enumerate(full_train_ids) if t == marker_id)
    return full_train_ids[: last_marker_pos + 1]


def _build_full_ids(
    tokenizer,
    persona_prompt: str | None,
    question: str,
    R_text: str,
    marker_text: str,
    marker_id: int,
    persona_for_log: str,
    q_idx_for_log: int,
    sep: str = MARKER_SEP,
) -> tuple[list[int], int, int, int]:
    """Construct full token-id sequence for one eval probe.

    Returns ``(full_ids, prompt_len, R_len, slot)`` where ``slot`` is the
    post-R marker position.

    Round-2 C1 fix: text-concat uses ``prompt_text + R_text + sep +
    marker_text`` (NOT ``prompt_text + R_text + marker_text``). The
    ``\\n\\n`` separator matches the assistant content training emits at
    ``build_training_data.py:299`` (``f"{r_text}\\n\\n{marker_text}"``);
    without it, the marker is teacher-forced after a context the trained
    model never saw (e.g., ``'.'`` directly fused with ``' ※'`` vs the
    fused ``'.\\n\\n'`` (id 382) + ``' ※'`` that training uses).

    Defense-in-depth: also build the train-equivalent token sequence via
    chat-template and assert byte-equality on the marker-slot context (last
    ``MARKER_PRECEDING_K_TOKENS`` tokens before the marker plus the marker
    itself). This is the load-bearing eval-vs-train measurement contract.

    Asserts (any failure raises ``RuntimeError`` with persona+q context):
    1. Marker is the LAST token (off-by-one guard).
    2. Marker appears exactly once (BPE re-segmentation guard).
    3. ``full_ids[: len(prompt_ids)] == prompt_ids`` (prompt prefix intact).
    4. ``full_ids[slot - K : slot + 1] ==
        train_equivalent[-(K+1):]`` for K = ``MARKER_PRECEDING_K_TOKENS``.
       This is the train-vs-eval token-equality contract.
    """
    if persona_prompt is None:
        messages = [{"role": "user", "content": question}]
    else:
        messages = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": question},
        ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    # C1 fix: include the `\n\n` separator that training emits.
    full_ids = tokenizer.encode(prompt_text + R_text + sep + marker_text, add_special_tokens=False)
    # Assertion 1 + 2: marker is last token AND appears exactly once.
    if full_ids[-1] != marker_id or full_ids.count(marker_id) != 1:
        raise RuntimeError(
            f"marker slot drift persona={persona_for_log!r} q_idx={q_idx_for_log}: "
            f"full_ids[-1]={full_ids[-1]} count={full_ids.count(marker_id)} "
            f"(expected last == {marker_id}, count == 1)"
        )
    # Assertion 3: prompt prefix intact (catches any chat-template drift
    # between prompt_text encoding and the concat encoding).
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            f"prompt prefix drift persona={persona_for_log!r} q_idx={q_idx_for_log}: "
            f"prompt_ids ({len(prompt_ids)}) does not prefix full_ids "
            f"({len(full_ids)}) — chat-template tokenization is non-stable "
            f"under text concatenation. Plan §4.6 off-by-one guard #2."
        )
    # Assertion 4: train-vs-eval token-equality on the marker slot context.
    # We build the EXACT sequence training emits via chat-template's
    # assistant-content path, then check the K tokens immediately before
    # the marker (plus the marker itself) match between eval and train.
    # Note: the train-reference encoding ALWAYS uses ``MARKER_SEP``
    # (= ``"\n\n"``) because that's the literal separator hard-coded in
    # ``build_training_data.py:299`` (``f"{r_text}\n\n{marker_text}"``).
    # If the caller passes a non-default ``sep`` (e.g. to deliberately
    # test the drift assertion), the eval encoding diverges and this
    # assertion fires.
    train_equivalent_ids = build_train_equivalent_full_ids(
        tokenizer,
        persona_prompt,
        question,
        R_text,
        marker_text,
        marker_id,
        sep=MARKER_SEP,
    )
    k = MARKER_PRECEDING_K_TOKENS
    eval_tail = full_ids[-(k + 1) :]
    train_tail = train_equivalent_ids[-(k + 1) :]
    if eval_tail != train_tail:
        raise RuntimeError(
            f"train/eval marker-slot context drift persona={persona_for_log!r} "
            f"q_idx={q_idx_for_log}: eval last {k + 1} tokens={eval_tail} "
            f"vs train last {k + 1} tokens={train_tail}. The training row's "
            f"marker slot has a different prefix than the eval row's marker "
            f"slot — log P(marker | context) is being read at the WRONG "
            f"position. C1 blocker (Claude + Codex code-review v1)."
        )
    slot = len(full_ids) - 1
    prompt_len = len(prompt_ids)
    R_len = slot - prompt_len  # = len(full_ids) - 1 - len(prompt_ids)
    return full_ids, prompt_len, R_len, slot


def _extract_marker_logprob_and_argmax(
    outputs,
    slot_positions: list[int],
    marker_id: int,
    cell_label: str,
) -> tuple[list[float], list[bool]]:
    """Read log-prob of ``marker_id`` at each row's slot; argmax-==-marker flag.

    Returns (logps clamped to LOGP_FLOOR, list[bool] argmax==marker).
    Raises if ``prompt_logprobs[slot]`` is None OR ``marker_id`` not present.
    """
    logps: list[float] = []
    argmax_marker: list[bool] = []
    for out, slot in zip(outputs, slot_positions, strict=True):
        slot_dict = out.prompt_logprobs[slot]
        if slot_dict is None:
            raise RuntimeError(
                f"{cell_label}: prompt_logprobs[{slot}] is None; "
                f"list len={len(out.prompt_logprobs)}"
            )
        if marker_id not in slot_dict:
            top_5 = sorted(slot_dict.items(), key=lambda kv: -kv[1].logprob)[:5]
            top_5_repr = [(tid, round(lp.logprob, 3)) for tid, lp in top_5]
            raise RuntimeError(
                f"{cell_label}: MARKER_ID {marker_id} not in prompt_logprobs[{slot}]; "
                f"top-5: {top_5_repr}"
            )
        lp = float(slot_dict[marker_id].logprob)
        logps.append(max(lp, LOGP_FLOOR))
        top_id = max(slot_dict.items(), key=lambda kv: kv[1].logprob)[0]
        argmax_marker.append(top_id == marker_id)
    return logps, argmax_marker


def run_eval(  # noqa: C901 - linear (build prompts -> vLLM trained -> vLLM base -> reshape -> write)
    *,
    cell_slug: str,
    adapter_path: str | None,
    base_model: str,
    out_dir: Path,
    r_eval_path: Path,
    eval_personas: dict[str, str],
    eval_questions: list[str],
    marker_text: str,
    marker_id: int,
    n_personas_limit: int | None = None,
    n_questions_limit: int | None = None,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = DEFAULT_GPU_MEM_UTIL,
    max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Run the vLLM prompt_logprobs eval for one cell.

    Args:
        cell_slug: e.g. ``"c1_anchor"``. ``"base"`` is a no-adapter sanity
            cell (computes only the base logp — useful for the Phase 1.5
            descriptive base-panel report).
        adapter_path: Local directory containing
            ``adapter_model.safetensors`` + ``adapter_config.json`` (LoRA
            adapter), OR None when ``cell_slug == "base"``.
        base_model: HF model id (``Qwen/Qwen2.5-7B-Instruct``).
        out_dir: Directory to write ``marker_logprob.json``.
        r_eval_path: Path to ``R_eval.json`` (Phase 1 frozen artifact).
        eval_personas: Panel persona name -> system prompt.
        eval_questions: Eval question list.
        marker_text, marker_id: Marker text + token id (defense-in-depth
            asserted at startup).
        n_personas_limit, n_questions_limit: Smoke/debug slicing.
        max_model_len, gpu_memory_utilization, max_lora_rank, seed: vLLM
            engine params.

    Returns:
        Path to the written ``marker_logprob.json``.
    """
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.contrastive_recipe_sweep_448.r_generate import (
        load_r_artifact,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    _assert_marker_token(tokenizer, marker_text, marker_id)

    r_eval = load_r_artifact(r_eval_path)
    # Lift content hash from the R_eval artifact for the output payload.
    r_eval_payload = json.loads(r_eval_path.read_text())
    r_eval_hash = r_eval_payload.get("content_hash", "unknown")
    log.info(
        "[%s] R_eval loaded from %s (sha[:12]=%s)",
        cell_slug,
        r_eval_path,
        r_eval_hash[:12],
    )

    # Apply optional smoke slicing.
    persona_items = list(eval_personas.items())
    if n_personas_limit is not None:
        persona_items = persona_items[:n_personas_limit]
    questions = list(eval_questions)
    if n_questions_limit is not None:
        questions = questions[:n_questions_limit]
    n_personas_actual = len(persona_items)
    n_questions_actual = len(questions)
    log.info(
        "[%s] Eval grid: %d personas x %d questions = %d probes",
        cell_slug,
        n_personas_actual,
        n_questions_actual,
        n_personas_actual * n_questions_actual,
    )

    # ── Build prompts payload + slot positions per (persona, q). ─────────────
    prompts_payload: list[dict] = []
    slot_positions: list[int] = []
    prompt_lens: list[int] = []
    R_lens: list[int] = []
    index_keys: list[tuple[str, int]] = []  # (persona, q_idx) per row
    for persona_name, persona_prompt in persona_items:
        if persona_name not in r_eval:
            raise KeyError(
                f"[{cell_slug}] R_eval missing persona {persona_name!r}; "
                f"available: {sorted(r_eval.keys())[:8]}... — Phase 1 "
                f"r-generate must include every EVAL_PERSONAS_24 persona "
                f"(Must-Fix-1)."
            )
        for q_idx, q in enumerate(questions):
            if q not in r_eval[persona_name]:
                raise KeyError(
                    f"[{cell_slug}] R_eval[{persona_name!r}] missing q_idx={q_idx} "
                    f"({q!r}). R_eval should cover all EVAL_QUESTIONS."
                )
            R_text = r_eval[persona_name][q]["response_text"]
            full_ids, p_len, r_len, slot = _build_full_ids(
                tokenizer,
                persona_prompt,
                q,
                R_text,
                marker_text,
                marker_id,
                persona_name,
                q_idx,
            )
            prompts_payload.append({"prompt_token_ids": full_ids})
            slot_positions.append(slot)
            prompt_lens.append(p_len)
            R_lens.append(r_len)
            index_keys.append((persona_name, q_idx))

    # ── vLLM engine bring-up (late import — heavy). ──────────────────────────
    from vllm import LLM, SamplingParams

    use_lora = adapter_path is not None and cell_slug != "base"
    llm_kwargs: dict = {
        "model": base_model,
        "dtype": "bfloat16",
        "gpu_memory_utilization": gpu_memory_utilization,
        "seed": seed,
        "max_model_len": max_model_len,
    }
    if use_lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank
        llm_kwargs["max_loras"] = 1
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=seed,
    )

    # ── Trained pass (skipped for "base" cell). ──────────────────────────────
    g_logps: list[float] = []
    g_argmax: list[bool] = []
    if use_lora:
        from vllm.lora.request import LoRARequest

        lora_req = LoRARequest(lora_name=cell_slug, lora_int_id=1, lora_path=adapter_path)
        log.info("[%s] vLLM trained pass: %d probes", cell_slug, len(prompts_payload))
        out_trained = llm.generate(prompts_payload, sp, lora_request=lora_req)
        if len(out_trained) != len(prompts_payload):
            raise RuntimeError(
                f"vLLM trained pass returned {len(out_trained)} for {len(prompts_payload)} probes."
            )
        g_logps, g_argmax = _extract_marker_logprob_and_argmax(
            out_trained, slot_positions, marker_id, cell_label=f"TRAINED/{cell_slug}"
        )

    # ── Base pass on the SAME prompts (lora_request=None). ───────────────────
    log.info("[%s] vLLM base pass: %d probes", cell_slug, len(prompts_payload))
    out_base = llm.generate(prompts_payload, sp, lora_request=None)
    if len(out_base) != len(prompts_payload):
        raise RuntimeError(
            f"vLLM base pass returned {len(out_base)} for {len(prompts_payload)} probes."
        )
    b_logps, b_argmax = _extract_marker_logprob_and_argmax(
        out_base, slot_positions, marker_id, cell_label=f"BASE/{cell_slug}"
    )

    # ── Reshape into per-persona nested dicts. ───────────────────────────────
    g_by_pq: dict[str, dict[str, float]] = {p: {} for p, _ in persona_items}
    b_by_pq: dict[str, dict[str, float]] = {p: {} for p, _ in persona_items}
    d_by_pq: dict[str, dict[str, float]] = {p: {} for p, _ in persona_items}
    em_by_pq: dict[str, dict[str, bool]] = {p: {} for p, _ in persona_items}
    if use_lora:
        for (persona, q_idx), gl, bl, ga in zip(
            index_keys, g_logps, b_logps, g_argmax, strict=True
        ):
            g_by_pq[persona][str(q_idx)] = float(gl)
            b_by_pq[persona][str(q_idx)] = float(bl)
            d_by_pq[persona][str(q_idx)] = float(gl - bl)
            em_by_pq[persona][str(q_idx)] = bool(ga)
    else:
        # base-only: g/delta/emission left empty; b is the descriptive measurement.
        for (persona, q_idx), bl, ba in zip(index_keys, b_logps, b_argmax, strict=True):
            b_by_pq[persona][str(q_idx)] = float(bl)
            em_by_pq[persona][str(q_idx)] = bool(ba)

    # Per-persona means.
    def _safe_mean(d: dict[str, float]) -> float:
        if not d:
            return float("nan")
        return float(np.mean(list(d.values())))

    def _safe_rate(d: dict[str, bool]) -> float:
        if not d:
            return float("nan")
        return float(np.mean([1.0 if v else 0.0 for v in d.values()]))

    g_mean = {p: _safe_mean(g_by_pq[p]) for p in g_by_pq} if use_lora else {}
    b_mean = {p: _safe_mean(b_by_pq[p]) for p in b_by_pq}
    d_mean = {p: _safe_mean(d_by_pq[p]) for p in d_by_pq} if use_lora else {}
    em_rate = {p: _safe_rate(em_by_pq[p]) for p in em_by_pq}

    # Floor-clamp diagnostic.
    if use_lora:
        n_floor_g = sum(1 for v in g_logps if v <= LOGP_FLOOR + 1e-6)
        n_floor_b = sum(1 for v in b_logps if v <= LOGP_FLOOR + 1e-6)
        rate_floor = (n_floor_g + n_floor_b) / max(2 * len(g_logps), 1)
        if rate_floor > 0.01:
            log.warning(
                "[%s] floor-clamp rate = %.2f%% (g=%d, b=%d) — investigate "
                "tokenizer / model drift before trusting the cell.",
                cell_slug,
                100.0 * rate_floor,
                n_floor_g,
                n_floor_b,
            )

    # ── Write output payload. ────────────────────────────────────────────────
    payload = {
        "schema_version": SCHEMA_VERSION,
        "cell": cell_slug,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "base_model": base_model,
        "marker_text": marker_text,
        "marker_token_id": marker_id,
        "logp_floor": LOGP_FLOOR,
        "eval_personas": [p for p, _ in persona_items],
        "eval_questions": questions,
        "n_personas_limit": n_personas_limit,
        "n_questions_limit": n_questions_limit,
        "n_cells_evaluated": n_personas_actual * n_questions_actual,
        "uses_lora": use_lora,
        "r_eval_path": str(r_eval_path),
        "r_eval_content_hash": r_eval_hash,
        "g_logprob_per_persona_q": g_by_pq,
        "b_logprob_per_persona_q": b_by_pq,
        "delta_g_per_persona_q": d_by_pq,
        "emission_recompute_per_persona_q": em_by_pq,
        "mean_per_persona_g_logprob": g_mean,
        "mean_per_persona_b_logprob": b_mean,
        "mean_per_persona_delta_g": d_mean,
        "mean_per_persona_emission_rate": em_rate,
        "prompt_lens_per_probe": prompt_lens,
        "R_lens_per_probe": R_lens,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path = out_dir / "marker_logprob.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(
        "[%s] Wrote %d-probe marker_logprob.json -> %s",
        cell_slug,
        payload["n_cells_evaluated"],
        out_path,
    )

    # Per-CLAUDE.md "checkpoint per phase" — also write a smaller summary.
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "cell": cell_slug,
        "uses_lora": use_lora,
        "n_personas_scored": n_personas_actual,
        "mean_per_persona_g_logprob": g_mean,
        "mean_per_persona_b_logprob": b_mean,
        "mean_per_persona_delta_g": d_mean,
        "mean_per_persona_emission_rate": em_rate,
        "r_eval_content_hash": r_eval_hash,
        "git_commit_sha": payload["git_commit_sha"],
        "timestamp_utc": payload["timestamp_utc"],
    }
    summary_path = out_dir / "marker_logprob_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    log.info("[%s] Wrote per-persona summary -> %s", cell_slug, summary_path)

    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cell",
        required=True,
        help="Cell slug (e.g. 'c1_anchor' or 'base' for the descriptive base panel).",
    )
    ap.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help=(
            "Local adapter directory (post-train, post-upload). Mutually exclusive "
            "with --cell base (which runs the base-only descriptive pass)."
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for marker_logprob.json (e.g. eval_results/issue_448_v5/<cell>/).",
    )
    ap.add_argument(
        "--r-eval-path",
        type=Path,
        default=Path("data/issue_448/on_policy_R/R_eval.json"),
        help="Path to the Phase 1 R_eval.json artifact.",
    )
    ap.add_argument("--eval-personas-limit", type=int, default=None)
    ap.add_argument("--eval-questions-limit", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEM_UTIL)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-lora-rank", type=int, default=DEFAULT_MAX_LORA_RANK)
    ap.add_argument(
        "--sentinel-path",
        type=Path,
        default=None,
        help="Optional sentinel JSON for the dispatcher (poll_pipeline-compliant keys).",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=eval_one_cell] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    if args.cell == "base":
        if args.adapter_path is not None:
            raise SystemExit("--cell base must NOT take --adapter-path (base-only pass).")
    else:
        if args.adapter_path is None:
            raise SystemExit(
                "--adapter-path is required for non-base cells. "
                "Pass the local merged-or-adapter dir."
            )

    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )
    from explore_persona_space.personas import EVAL_QUESTIONS

    out_path = run_eval(
        cell_slug=args.cell,
        adapter_path=args.adapter_path,
        base_model=BASE_MODEL,
        out_dir=args.out_dir,
        r_eval_path=args.r_eval_path,
        eval_personas=EVAL_PERSONAS_24,
        eval_questions=list(EVAL_QUESTIONS),
        marker_text=MARKER_TEXT,
        marker_id=EXPECTED_MARKER_TOKEN_ID,
        n_personas_limit=args.eval_personas_limit,
        n_questions_limit=args.eval_questions_limit,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_lora_rank=args.max_lora_rank,
        seed=args.seed,
    )

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_payload = {
            "sentinel_schema_version": 1,
            "kind": "epm:progress",
            "version": 1,
            "task_id": 448,
            "phase": f"eval_{args.cell}",
            "by": "i448_eval_one_cell",
            "ts": datetime.now(UTC).isoformat(),
            "note": json.dumps(
                {
                    "cell": args.cell,
                    "adapter_path": args.adapter_path,
                    "marker_logprob_path": str(out_path),
                    "marker_logprob_summary_path": str(
                        args.out_dir / "marker_logprob_summary.json"
                    ),
                }
            ),
        }
        args.sentinel_path.write_text(json.dumps(sentinel_payload, indent=2))
        log.info("Wrote eval sentinel -> %s", args.sentinel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
