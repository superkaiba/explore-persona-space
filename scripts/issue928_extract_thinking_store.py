#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ×, ², ≥) in scientific docstrings + log messages.
"""Issue #928 Phases 0/G/P/B/U: thinking-model rollouts + segmented per-part summary store.

One linear dispatcher (plan §4.1; smoke = the SAME dispatcher with
``--contexts 5`` — unification default, no separate smoke path):

- **Phase 0 (Gate 1, plan §7):** generate the gate slice (first
  ``--gate-contexts`` contexts × probes) under the current rung; the
  well-formed parse rate must be ≥ 80% AND no degeneration signature (p95
  completion tokens < the cap; no completion with a > 50% repeated-4-gram
  fraction). On FAIL walk the pre-registered SAME-MODEL fallback ladder
  (§4.3: greedy → temp 0.6/top_p 0.95/seed 42 → ``<think>\\n`` prompt prefill),
  re-running ONLY the gate slice per rung. Rung-(iii) exhaustion is TERMINAL
  (exit 3 + failure sentinel; the orchestrator posts ``epm:failure``
  ``failure_class: data``) — there is NO model-switch rung.
- **Phase G:** vLLM batched generation over all (C, q) prompts at the chosen
  rung — greedy (parent store parity), ``max_new_tokens=8192``,
  ``gpu_memory_utilization=0.85`` (plan §4.3: the reused parent helper's
  hardcoded 0.45 is deliberately overridden via the parametrized engine
  builder here), chunked internally (gotchas.md: a single huge
  ``llm.generate`` can deadlock the v1 EngineCore; per-chunk INFO logs keep
  the poller's stall detection fed). Rollout TEXT persists verbatim PER
  GENERATION GROUP (~one vLLM chunk of contexts) the moment that group
  returns, BEFORE any parse/capture (generation-and-reduce persistence,
  §4.8/#779; round 2: a mid-generation crash keeps every completed group).
  ``--skip-gen`` reuses only rollout files matching every output-affecting
  run arg (model / rung / probe pool / max_new_tokens / probe list).
- **Phase P:** code segmentation (``issue928_common.segment_completion`` —
  exact string offsets, per-rung criterion; §4.4), per-context coverage,
  cap-truncation accounting with ONE 16,384 re-generation rung when > 10% of
  rows are cap-truncated.
- **Phase B:** teacher-forced batched forwards (bf16 weights, fp32 reduce,
  fp16 storage) over prompt + completion + ``<|im_end|>`` + ``\\n``; 28-layer
  hooks (#594 ``LayerCapture``), STREAMING in-forward reduction to the 12
  per-part summary vectors per (C, q, layer) (#666/#772 — the full token×layer
  grid is never materialized); left-padded batches with explicit
  ``position_ids``; ``logits_to_keep=1`` (introspection-guarded, #779);
  fail-loud assistant-header position assert carried EXPLICITLY from the #594
  lineage (``GENERATION_SUFFIX`` — plan §4.2 NOTE: the assert lives here, not
  inside the reused prompt builder). Round 2: per-context ``.pt`` writes are
  atomic AND an entry-time skip-if-valid predicate reuses existing blobs that
  match every output-affecting run arg — a crash-restart recaptures only the
  missing contexts.
- **Phase U:** one ``upload_folder`` commit each for rollouts + store with a
  SCOPED ``list_repo_tree`` verify (never a bare listing — gotcha #833). The
  end-of-extract sentinel is ``epm:progress`` (round 2) — the ONE
  ``epm:results`` sentinel fires from the run_all finalize step at true
  end-of-workload.

Store schema (plan §4.5): ``store/percq_summaries/<context_id>.pt`` — per-(q,
summary, layer) fp16 vectors + probe-averaged tensors + coverage counts,
fail-loud on probe-pool-hash / layer-count drift.

Usage::

    # production (GCP capture-7b lane):
    uv run python scripts/issue928_extract_thinking_store.py --gpu

    # pod-side Phase-0 smoke (= the sweep at 5 contexts):
    uv run python scripts/issue928_extract_thinking_store.py --gpu --contexts 5

    # CPU-only VM smoke of the CPU-runnable portion (tiny same-family model,
    # synthetic well-formed <think> completions replacing ONLY the vLLM call):
    uv run python scripts/issue928_extract_thinking_store.py --smoke \\
        --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --contexts 2 --probes 3 \\
        --synthetic-completions --no-upload --out-dir /tmp/issue-928-smoke/data
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))
# vLLM v1 EngineCore dies silently under fork() when the parent touched
# CUDA-adjacent code before LLM() (gotchas.md #628) — set BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import inspect
import json
import logging
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import torch  # noqa: E402

# Cross-script helper imports hoisted to module top so a missing symbol crashes
# at process start, never inside a smoke-skipped branch (gotchas.md #606).
from issue594_common import messages_for_instance, probes_hash  # noqa: E402
from issue594_extract_context_vectors import LayerCapture  # noqa: E402
from issue658_extract_base_store import _reap_vllm  # noqa: E402
from issue928_common import (  # noqa: E402
    ENDOFTEXT_TOKEN_ID,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    FALLBACK_RUNGS,
    GATE_P95_MUST_BE_BELOW_CAP,
    GENERATION_SEED,
    GENERATION_SUFFIX,
    GPU_MEMORY_UTILIZATION,
    IM_END_TOKEN_ID,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    MAX_NEW_TOKENS_RETRY,
    PARSE_RATE_FLOOR,
    PREFILL_TEXT,
    RAW_COMPLETIONS_PREFIX,
    REPEAT_4GRAM_MAX_FRAC,
    REPEAT_OFFENDER_MAX_FRAC,
    STORE_PREFIX,
    SUMMARY_NAMES,
    THINK_CLOSE,
    THINK_OPEN,
    THINKING_MODEL,
    TRUNCATION_REGEN_FRAC,
    TURN_NL_TOKEN_ID,
    char_span_to_token_span,
    context_order_and_families,
    dump_json,
    load_probe_pool,
    repeated_4gram_fraction,
    reproducibility_metadata,
    resolve_battery,
    segment_completion,
    upload_folder_scoped_verify,
    write_sentinel,
)

logger = logging.getLogger("issue928_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))


def phase(name: str) -> None:
    """Emit a poll_pipeline.py-parseable phase line."""
    print(f"[phase={name}]", flush=True)


# ── prompts ───────────────────────────────────────────────────────────────────


def build_prompt_text(tokenizer, instance: dict, probe: str, rung: str) -> str:
    """Templated prompt text for one (context, probe) cell (+ rung-iii prefill).

    Persona injection is ALWAYS a system turn (``messages_for_instance``). The
    rung-(iii) prefill appends ``<think>\\n`` AFTER the assistant header (the
    R1-distill forced-prefix pattern, plan §4.3); rungs (i)/(ii) leave the
    templated text untouched.
    """
    messages = messages_for_instance(instance, probe)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if rung == "prefill":
        text = text + PREFILL_TEXT
    return text


def assert_assistant_header(
    tokenizer,
    prompt_ids: torch.Tensor,
    ctx_id: str,
    probe: str,
    generation_suffix: str | None = None,
) -> None:
    """Fail-loud assistant-header position assert (#594 lineage, carried explicitly).

    The last 3 tokens of the TEMPLATED prompt must decode to
    ``<|im_start|>assistant\\n`` — the slot ``ctx_last`` reads (the parent's
    c_C last-input-token position). Plan §4.2: this assert lives in the
    issue928 extractor itself (the reused ``build_prompts`` helper does NOT
    provide it — it sits at ``issue594_extract_context_vectors.py:165``).

    ``generation_suffix`` (DEFAULT-PRESERVING, #1005 §4.1: ``None`` ⇒ the #928
    ``GENERATION_SUFFIX`` byte-for-byte) overrides the expected last-3-token
    decode for a model whose chat template forces a different scaffold
    (R1-distill: ``<｜Assistant｜><think>\\n``).
    """
    want = GENERATION_SUFFIX if generation_suffix is None else generation_suffix
    suffix = tokenizer.decode(prompt_ids[-3:])
    assert suffix == want, (
        f"assistant-header position assert failed for context={ctx_id} probe={probe[:40]!r}: "
        f"last-3-token decode {suffix!r} != {want!r} (a drifted template would "
        "capture ctx_last at the WRONG slot for every row — refusing)"
    )


# ── generation (Phase G) ──────────────────────────────────────────────────────


def sampling_params_for_rung(
    rung: str, max_new_tokens: int, stop_token_ids: list[int] | None = None
):
    """Per-rung vLLM SamplingParams (plan §4.3 ladder recipe).

    ``stop_token_ids`` (DEFAULT-PRESERVING, #1005 §4.0: ``None`` ⇒ the #928
    ``[IM_END_TOKEN_ID, ENDOFTEXT_TOKEN_ID]`` byte-for-byte) overrides the stop
    set for a model whose eos differs (R1-distill: ``[151643]`` — in THAT
    tokenizer 151645 is ``<｜Assistant｜>``, NOT an end token).
    """
    from vllm import SamplingParams

    stop_ids = [IM_END_TOKEN_ID, ENDOFTEXT_TOKEN_ID] if stop_token_ids is None else stop_token_ids
    if rung == "sample":
        return SamplingParams(
            temperature=0.6,
            top_p=0.95,
            seed=GENERATION_SEED,
            max_tokens=max_new_tokens,
            stop_token_ids=stop_ids,
        )
    # greedy + prefill rungs decode greedily (parent store parity).
    return SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=stop_ids)


def build_vllm_engine(
    model_name: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    revision: str | None = None,
):
    """vLLM engine with the PARAMETRIZED memory utilization (plan §4.3 override).

    The reused parent helper ``issue658_extract_base_store.vllm_generate``
    HARDCODES ``gpu_memory_utilization=0.45``; #928 requires 0.85 (at 0.45 the
    A100-40 fallback rung likely fails engine init on 8k-token sequences), so
    the engine is built here with the value threaded from the CLI.

    ``revision`` (DEFAULT-PRESERVING, #1005 §4.1: ``None`` ⇒ unpinned, the
    #928 behavior) pins the Hub revision — #1005 pins the R1-distill chat
    template (the manipulated contract) against upstream template changes.
    """
    from vllm import LLM

    return LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=GENERATION_SEED,
        revision=revision,
    )


def vllm_generate_chunked(llm, prompts: list[str], sp) -> list[tuple[str, str]]:
    """Chunked ``llm.generate`` (order-preserving) → [(text, finish_reason)].

    A single huge ``generate()`` can deadlock the v1 EngineCore (gotchas.md);
    the per-chunk INFO log is load-bearing (keeps the poller's stall detection
    fed on multi-hour phases). ``use_tqdm=False`` (gotcha #613).
    """
    out: list[tuple[str, str]] = []
    n_chunks = (len(prompts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for i in range(0, len(prompts), VLLM_CHUNK_SIZE):
        chunk = prompts[i : i + VLLM_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] generate chunk %d/%d (%d prompts)",
            i // VLLM_CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
        )
        outs = llm.generate(chunk, sp, use_tqdm=False)
        for o in outs:
            out.append((o.outputs[0].text, o.outputs[0].finish_reason))
    return out


def synthetic_completions(prompts: list[str], rung: str, n_probes: int) -> list[tuple[str, str]]:
    """Deterministic synthetic completions for the CPU-only VM smoke.

    Replaces ONLY the vLLM call output — every downstream phase (persist,
    parse, gate, capture, store, fit) runs the identical production code path
    on these strings. Mix: mostly well-formed ``<think>…</think>`` blocks (per
    the rung's criterion), one malformed row per ~12 (exercises the drop/
    coverage path), one cap-truncated row per ~24 (exercises truncation
    accounting), and one well-formed-but-DEGENERATE row per ~48 (repeated-
    4-gram fraction > 0.5 — exercises the v3 ``degenerate_repetition``
    reclassification + the gate's offender-RATE conjunct with offenders > 0)
    — the mix keeps the usable rate ≥ ~0.83 and the offender rate ≤ ~0.05 at
    any slice size, so the amended Gate 1 passes. NEVER used in production
    (no ``--synthetic-completions``).
    """
    out: list[tuple[str, str]] = []
    for i in range(len(prompts)):
        body = (
            f"Let me reason about request {i}. The user asks a question; I consider "
            f"safety and correctness step {i % 3}, then decide."
        )
        answer = f"Here is my considered answer to request {i}: I would recommend caution."
        open_tag = "" if rung == "prefill" else THINK_OPEN + "\n"
        if i % 24 == 7:
            out.append((f"{open_tag}{body} and then it keeps going", "length"))  # truncated
        elif i % 12 == 3:
            out.append((f"{body} {answer}", "stop"))  # malformed: no think block
        elif i % 48 == 19:
            # Degenerate repetition loop: >= 50 words, > 50% repeated 4-grams,
            # structurally well-formed — reclassified in parse_rows (v3 §4.4).
            loop = " ".join(["repeat the same loop words"] * 20)
            out.append((f"{open_tag}{loop}\n{THINK_CLOSE}\n\n{answer}", "stop"))
        else:
            out.append((f"{open_tag}{body}\n{THINK_CLOSE}\n\n{answer}", "stop"))
    _ = n_probes
    return out


# ── gate (Phase 0) ────────────────────────────────────────────────────────────


def gate1_check(rows: list[dict], cap: int) -> dict:
    """Gate 1 (plan §7, amended v3): usable-parse floor + degeneration checks.

    ``rows`` carry {well_formed, reason, n_gen_tokens, rep_frac}. PASS iff the
    USABLE-row rate ≥ 0.80 (``well_formed`` — which, after ``parse_rows``'s v3
    ``degenerate_repetition`` reclassification, is well-formed segmentation ∧
    not a repetition offender) AND p95 gen-token count < cap AND the
    repetition-offender RATE ≤ ``REPEAT_OFFENDER_MAX_FRAC``. The offender
    count is computed from ``rep_frac`` over ALL rows — independent of the
    ``reason`` bookkeeping, so structural/truncation reason precedence can
    never mask an offender (plan §4.4 delta 3).
    """
    import numpy as np

    n = len(rows)
    rate = sum(1 for r in rows if r["well_formed"]) / max(1, n)
    p95 = float(np.percentile([r["n_gen_tokens"] for r in rows], 95)) if rows else 0.0
    offenders = sum(1 for r in rows if r["rep_frac"] > REPEAT_4GRAM_MAX_FRAC)
    offender_rate = offenders / max(1, n)
    ok = rate >= PARSE_RATE_FLOOR and offender_rate <= REPEAT_OFFENDER_MAX_FRAC
    if GATE_P95_MUST_BE_BELOW_CAP:
        ok = ok and p95 < cap
    reasons: dict[str, int] = {}
    for r in rows:
        if not r["well_formed"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return {
        "pass": bool(ok),
        "parse_rate": rate,
        "p95_gen_tokens": p95,
        "cap": cap,
        "n_rows": n,
        "repetition_offenders": offenders,
        "repetition_offender_rate": offender_rate,
        "malformed_reasons": reasons,
    }


def parse_rows(tokenizer, completions: list[tuple[str, str]], rung: str) -> list[dict]:
    """Segment + degeneration-screen a batch of (text, finish_reason) rows.

    Token counts come from a tokenize-only pass (used for the p95/cap checks +
    truncation accounting); a ``finish_reason == "length"`` row with no
    ``</think>`` is counted ``truncated_no_close`` (plan §4.4). A
    segmentation-WELL-FORMED row whose repeated-4-gram fraction exceeds
    ``REPEAT_4GRAM_MAX_FRAC`` is reclassified ``well_formed=False,
    reason="degenerate_repetition"`` (v3 amendment, §4.4) — dropped +
    coverage-counted like ``truncated_no_close`` / ``no_close``. Precedence:
    structural / truncation reasons win (the reclassification applies ONLY to
    well-formed rows); the gate's offender count reads ``rep_frac`` over ALL
    rows, so precedence cannot mask offenders. Char spans stay as computed —
    consumers filter on ``well_formed``.
    """
    out: list[dict] = []
    texts = [t for t, _fr in completions]
    encs = tokenizer(texts, add_special_tokens=False)["input_ids"]
    for (text, fr), ids in zip(completions, encs, strict=True):
        wf, reason, cot_span, ans_span = segment_completion(text, rung)
        if not wf and fr == "length" and THINK_CLOSE not in text:
            reason = "truncated_no_close"
        rep = repeated_4gram_fraction(text)
        if wf and rep > REPEAT_4GRAM_MAX_FRAC:
            wf = False
            reason = "degenerate_repetition"
        out.append(
            {
                "well_formed": wf,
                "reason": reason,
                "cot_char_span": list(cot_span),
                "ans_char_span": list(ans_span),
                "n_gen_tokens": len(ids),
                "finish_reason": fr,
                "rep_frac": rep,
            }
        )
    return out


# ── capture (Phase B) ─────────────────────────────────────────────────────────

# Single-position summary names → assembled index (order pinned by SUMMARY_NAMES).
_POSITION_NAMES = ("ctx_last", "cot_last", "cot_close", "ans_last", "ans_im_end", "ans_turn_nl")


def _logits_to_keep_kwargs(model) -> dict:
    """``logits_to_keep=1`` when the forward names it EXPLICITLY (gotcha #779).

    Hidden-state-only forwards otherwise materialize full-vocab logits for ALL
    positions (a ~5 GiB unread allocation on a 152k vocab). A bare ``**kwargs``
    does NOT count (stubs would swallow or crash on it).
    """
    fwd = getattr(model, "forward", None) or model.__call__
    try:
        params = inspect.signature(fwd).parameters
    except (TypeError, ValueError):
        return {}
    return {"logits_to_keep": 1} if "logits_to_keep" in params else {}


def build_capture_row(
    tokenizer,
    instance,
    probe,
    completion,
    parse_rec,
    rung,
    parts_spec=None,
    prompt_parts_spec=None,
    generation_suffix: str | None = None,
    boundary_ids: list[int] | None = None,
    boundary_positions: dict[str, int] | None = None,
    prompt_positions: dict[str, int] | None = None,
):
    """One teacher-forced row: ids + part token-spans + single positions, or (None, reason).

    Token spans derive from ``return_offsets_mapping`` over the completion text
    (robust to BPE merges — plan §4.4); a part whose token span comes out
    empty (the #825 zero-width class) drops the row with a counted reason.
    Absolute positions index ``prompt_full + completion + <|im_end|> + \\n``.

    ``parts_spec`` (DEFAULT-PRESERVING, follow-up plan v6 §4.2: ``None`` ⇒
    existing behavior byte-for-byte) is an optional callable
    ``(cot_tok, ans_tok) -> dict[str, (s, e)] | str`` receiving the cot/ans
    COMPLETION-token-space spans; a dict return adds each extra part's
    completion-token-space half-open span to ``row["spans"]`` (absolute
    positions applied here); a str return drops the row with that string as
    the counted reason (the matched-length floor path).

    ``prompt_parts_spec`` (DEFAULT-PRESERVING, follow-up plan v7 §4.1 — the
    second extension: ``None`` ⇒ existing behavior byte-for-byte) is an
    optional callable ``(prompt_text_tpl, prompt_offsets, prompt_len_tpl) ->
    dict[str, (s, e)] | str`` receiving the templated prompt text, its
    ``return_offsets_mapping`` (computed ONLY on this branch, asserted
    token-identical to the templated prompt ids), and the templated prompt
    token length; a dict return adds each PROMPT-side span to ``row["spans"]``
    asserted ``0 <= s < e <= prompt_len_tpl`` (absolute positions == the same
    indices — prompt tokens start at 0); a str return drops the row with that
    counted reason. Evaluated AFTER ``parts_spec`` so completion-floor drop
    reasons keep the matched-length round's accounting.

    #1005 §4.1 model-profile extensions (all DEFAULT-PRESERVING — ``None`` ⇒
    existing behavior byte-for-byte): ``generation_suffix`` overrides the
    assistant-header assert's expected decode; ``boundary_ids`` overrides the
    teacher-forced post-answer feed (#928: ``[IM_END, \\n]``; #1005:
    ``[151643]`` = ``ans_eos``); ``boundary_positions`` maps position NAMES to
    offsets into the boundary (#928 default: ``{"ans_im_end": 0,
    "ans_turn_nl": 1}``); ``prompt_positions`` adds prompt-side single
    positions as NEGATIVE offsets from ``prompt_len_tpl`` (#1005:
    ``{"ctx_assist": -3}`` — the ``<｜Assistant｜>`` token). Prompt
    tokenization passes ``add_special_tokens=False`` (identical output for the
    parent Qwen2 tokenizer, which adds no specials; REQUIRED for the
    ``add_bos_token: true`` R1 tokenizer whose template already embeds bos —
    the #1005 exactly-one-bos contract).
    """
    prompt_text_tpl = tokenizer.apply_chat_template(
        messages_for_instance(instance, probe), tokenize=False, add_generation_prompt=True
    )
    prompt_ids_tpl = tokenizer(
        prompt_text_tpl, return_tensors="pt", padding=False, add_special_tokens=False
    )["input_ids"][0]
    assert_assistant_header(
        tokenizer, prompt_ids_tpl, instance["id"], probe, generation_suffix=generation_suffix
    )
    prompt_len_tpl = int(prompt_ids_tpl.shape[0])
    if rung == "prefill":
        prefill_ids = tokenizer(PREFILL_TEXT, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0]
        prompt_ids = torch.cat([prompt_ids_tpl, prefill_ids])
    else:
        prompt_ids = prompt_ids_tpl
    prompt_len = int(prompt_ids.shape[0])

    enc = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    comp_ids = torch.tensor(enc["input_ids"], dtype=prompt_ids.dtype)
    offsets = enc["offset_mapping"]
    if comp_ids.shape[0] == 0:
        return None, "empty_completion_tokens"
    cot_tok = char_span_to_token_span(offsets, tuple(parse_rec["cot_char_span"]))
    ans_tok = char_span_to_token_span(offsets, tuple(parse_rec["ans_char_span"]))
    if cot_tok == (0, 0):
        return None, "empty_cot_token_span"
    if ans_tok == (0, 0):
        return None, "empty_ans_token_span"
    # cot_close = the token containing the closing tag's final char.
    close_char = completion.index(THINK_CLOSE) + len(THINK_CLOSE) - 1
    close_tok = char_span_to_token_span(offsets, (close_char, close_char + 1))
    if close_tok == (0, 0):
        return None, "empty_close_token_span"

    b_ids = [IM_END_TOKEN_ID, TURN_NL_TOKEN_ID] if boundary_ids is None else list(boundary_ids)
    b_pos = (
        {"ans_im_end": 0, "ans_turn_nl": 1} if boundary_positions is None else boundary_positions
    )
    assert all(0 <= off < len(b_ids) for off in b_pos.values()), (b_pos, b_ids)
    boundary = torch.tensor(b_ids, dtype=prompt_ids.dtype)
    full_ids = torch.cat([prompt_ids, comp_ids, boundary])
    comp_len = int(comp_ids.shape[0])
    spans = {
        # ctx part = the TEMPLATED prompt tokens only (a rung-iii prefill is
        # delimiter scaffolding, part of no part — like the <think> tag itself).
        "ctx": (0, prompt_len_tpl),
        "cot": (prompt_len + cot_tok[0], prompt_len + cot_tok[1]),
        "ans": (prompt_len + ans_tok[0], prompt_len + ans_tok[1]),
    }
    if parts_spec is not None:
        extra = parts_spec(cot_tok, ans_tok)
        if isinstance(extra, str):
            return None, extra
        for name, (es, ee) in extra.items():
            assert name not in spans, f"parts_spec redefines base part {name!r}"
            assert 0 <= es < ee <= comp_len, (name, es, ee, comp_len)
            spans[name] = (prompt_len + es, prompt_len + ee)
    if prompt_parts_spec is not None:
        enc_p = tokenizer(prompt_text_tpl, return_offsets_mapping=True, add_special_tokens=False)
        assert list(enc_p["input_ids"]) == prompt_ids_tpl.tolist(), (
            "offsets-call tokenization drifted from the templated prompt ids"
        )
        extra_p = prompt_parts_spec(prompt_text_tpl, enc_p["offset_mapping"], prompt_len_tpl)
        if isinstance(extra_p, str):
            return None, extra_p
        for name, (ps, pe) in extra_p.items():
            assert name not in spans, f"prompt_parts_spec redefines part {name!r}"
            assert 0 <= ps < pe <= prompt_len_tpl, (name, ps, pe, prompt_len_tpl)
            spans[name] = (ps, pe)  # prompt tokens start at 0 ⇒ absolute == prompt indices
    positions = {
        "ctx_last": prompt_len_tpl - 1,  # assistant-header newline (parent c_C slot)
        "cot_last": prompt_len + cot_tok[1] - 1,
        "cot_close": prompt_len + close_tok[1] - 1,
        "ans_last": prompt_len + ans_tok[1] - 1,
    }
    for name, off in b_pos.items():
        assert name not in positions, f"boundary_positions redefines base position {name!r}"
        positions[name] = prompt_len + comp_len + off
    if prompt_positions is not None:
        for name, rel in prompt_positions.items():
            assert name not in positions, f"prompt_positions redefines position {name!r}"
            pos = prompt_len_tpl + rel  # rel is NEGATIVE (offset back from the prompt end)
            assert 0 <= pos < prompt_len_tpl, (name, rel, prompt_len_tpl)
            positions[name] = pos
    fed = full_ids[prompt_len + comp_len : prompt_len + comp_len + len(b_ids)].tolist()
    assert fed == b_ids, f"boundary ids drifted: {fed} != {b_ids}"
    return {"full_ids": full_ids, "spans": spans, "positions": positions}, ""


def pack_batches(rows: list[dict], batch_probes: int, token_budget: int) -> list[list[int]]:
    """Length-sorted token-budget batching (plan §9: length-bucketed forwards).

    Rows sorted by length descending; each batch holds ≤ ``batch_probes`` rows
    AND ≤ ``token_budget`` total padded tokens (B × max_len) — bounds the
    28-layer hook footprint (B × T × H × 2 bytes × 28) on the 40 GB rung.
    """
    order = sorted(range(len(rows)), key=lambda i: -int(rows[i]["full_ids"].shape[0]))
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        length = int(rows[i]["full_ids"].shape[0])
        new_max = max(cur_max, length)
        if cur and (len(cur) + 1 > batch_probes or new_max * (len(cur) + 1) > token_budget):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = length
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def rollout_content_digest(probes: list[str], completions: list[tuple[str, str]]) -> str:
    """sha256 (16 hex) over one context's rollout content in probe order.

    The generation-output identity a store blob must match to be reusable
    (round 3, code-review r2 BLOCKER `long-loop-restartability-missing`):
    covers completion TEXT + finish_reason per probe, so a resume after ANY
    rollout regeneration (changed cap, rung resample, 16k re-gen) changes the
    digest and forces recapture — never silently reuse activations of old
    completions (#722 r3).
    """
    import hashlib

    h = hashlib.sha256()
    for q, (text, fr) in zip(probes, completions, strict=True):
        for part in (q, text, fr):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
    return h.hexdigest()[:16]


def reusable_store_blob(
    path: Path,
    context_id: str,
    *,
    model_name: str,
    family: str,
    rung: str,
    probe_pool_hash: str,
    capture_layers: list[int],
    summary_names: list[str],
    n_probes: int,
    max_new_tokens: int,
    rollout_digest: str,
    hidden_size: int,
) -> tuple[dict | None, str]:
    """Entry-time skip-if-valid predicate for an existing per-context store blob.

    Module-level (pytest-pinned: ``tests/test_issue928_decomposition.py``) and
    SYMMETRIC with the rollout-side ``_rollout_blob_mismatch``: the blob must
    match EVERY output-affecting run arg INCLUDING the generation identity —
    ``max_new_tokens`` + the rollout-content digest (round 3, code-review r2
    BLOCKERs: a resume after regenerating rollouts at a different cap /
    content must RECAPTURE; a pre-round-3 blob missing the fields reads
    ``None != want`` and is recaptured). ``probe_indices`` are validated
    against the RUN's probe list (``0 <= qi < n_probes``, strictly increasing
    — the prior check compared ``per_q.shape[0]`` against the blob's OWN
    indices, catching only corruption, never staleness). Returns
    ``(blob, "")`` when reusable, else ``(None, reason)``.
    """
    try:
        blob = torch.load(path, weights_only=False)
    except Exception as exc:  # corrupt / partial file → recapture
        return None, f"unreadable ({type(exc).__name__}: {exc})"
    for key, got, want in (
        ("context_id", blob.get("context_id"), context_id),
        ("model", blob.get("model"), model_name),
        ("family", blob.get("family"), family),
        ("rung", blob.get("rung"), rung),
        ("probe_pool_hash", blob.get("probe_pool_hash"), probe_pool_hash),
        ("capture_layers", blob.get("capture_layers"), capture_layers),
        ("summary_names", list(blob.get("summary_names", [])), list(summary_names)),
        ("n_probes_total", blob.get("coverage", {}).get("n_probes_total"), n_probes),
        ("max_new_tokens", blob.get("max_new_tokens"), max_new_tokens),
        ("rollout_digest", blob.get("rollout_digest"), rollout_digest),
    ):
        if got != want:
            return None, f"{key} mismatch"
    kept = blob.get("probe_indices", [])
    if not (
        all(isinstance(qi, int) and 0 <= qi < n_probes for qi in kept)
        and sorted(set(kept)) == list(kept)
    ):
        return None, f"probe_indices invalid against the run's {n_probes}-probe list"
    per_q = blob.get("per_q")
    want_shape = (len(kept), len(summary_names), len(capture_layers), hidden_size)
    if per_q is None or tuple(per_q.shape) != want_shape:
        got_shape = tuple(per_q.shape) if per_q is not None else None
        return None, f"per_q shape {got_shape} != {want_shape}"
    return blob, ""


def reduce_forward_batch(
    model, capture, capture_layers, tokenizer, batch_rows, summary_names=None, position_names=None
):
    """ONE left-padded forward + GPU-side streaming reduction → (B, S, Lc, H) fp16 CPU.

    Explicit ``position_ids`` (cumsum(mask)−1 clamped at 0 — RoPE under
    left-pad silently diverges without it), fp32 reduction of the bf16 hook
    captures, only the reduced (B, S, Lc, H) slice crosses PCIe (streaming
    in-forward reduction, #666/#772 — the full token×layer grid is never
    materialized).

    ``summary_names`` (DEFAULT-PRESERVING, follow-up plan v6 §4.2: ``None`` ⇒
    the existing 12-name ``SUMMARY_NAMES`` behavior byte-for-byte) selects the
    reduced vectors: ``<part>_mean`` / ``<part>_max`` names reduce over the
    part's ``row["spans"]`` mask (any part the rows carry, incl. parts_spec
    extras); names in ``_POSITION_NAMES`` gather single positions. S =
    ``len(summary_names)``.

    ``position_names`` (DEFAULT-PRESERVING, #1005 §4.1: ``None`` ⇒ the #928
    ``_POSITION_NAMES`` byte-for-byte) overrides the single-position name
    registry for a profile with different boundary/prompt positions
    (R1-distill: ``ans_eos`` replaces ``ans_im_end``/``ans_turn_nl``;
    ``ctx_assist`` added).
    """
    names = tuple(SUMMARY_NAMES) if summary_names is None else tuple(summary_names)
    pos_registry = _POSITION_NAMES if position_names is None else tuple(position_names)
    pos_names = [n for n in names if n in pos_registry]
    mean_parts = [n[: -len("_mean")] for n in names if n.endswith("_mean")]
    max_parts = [n[: -len("_max")] for n in names if n.endswith("_max") and n not in pos_names]
    unknown = [
        n for n in names if n not in pos_names and not (n.endswith("_mean") or n.endswith("_max"))
    ]
    assert not unknown, f"unsupported summary names: {unknown}"
    part_names = tuple(dict.fromkeys(mean_parts + max_parts))  # order-stable dedupe
    device = model.device
    B = len(batch_rows)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else IM_END_TOKEN_ID
    max_len = max(int(r["full_ids"].shape[0]) for r in batch_rows)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    part_masks = {p: torch.zeros((B, max_len), dtype=torch.bool) for p in part_names}
    pos_idx = torch.zeros((B, max(1, len(pos_names))), dtype=torch.long)
    for bi, r in enumerate(batch_rows):
        length = int(r["full_ids"].shape[0])
        pad = max_len - length  # LEFT-pad: real tokens at [pad, max_len)
        input_ids[bi, pad:] = r["full_ids"]
        attn[bi, pad:] = 1
        for p in part_names:
            s, e = r["spans"][p]
            assert 0 <= s < e <= length, (p, s, e, length)
            part_masks[p][bi, pad + s : pad + e] = True
        for pi, name in enumerate(pos_names):
            pos = r["positions"][name]
            assert 0 <= pos < length, (name, pos, length)
            pos_idx[bi, pi] = pad + pos
    input_ids = input_ids.to(device)
    attn = attn.to(device)
    position_ids = (attn.long().cumsum(dim=1) - 1).clamp(min=0).to(device)
    pos_idx_dev = pos_idx.to(device)
    masks_dev = {p: m.to(device) for p, m in part_masks.items()}
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=position_ids,
            **_logits_to_keep_kwargs(model),
        )
    H = model.config.hidden_size
    mean_set, max_set = set(mean_parts), set(max_parts)
    per_layer = []
    for li in capture_layers:
        hs = capture.latest[li].float()  # (B, T, H) fp32 reduce of the bf16 capture
        by_name: dict[str, torch.Tensor] = {}
        for p in part_names:
            m = masks_dev[p].unsqueeze(-1)  # (B, T, 1)
            cnt = masks_dev[p].sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B, 1)
            if p in mean_set:
                by_name[f"{p}_mean"] = (hs * m).sum(dim=1) / cnt
            if p in max_set:
                by_name[f"{p}_max"] = hs.masked_fill(~m, float("-inf")).amax(dim=1)
        if pos_names:
            picked = torch.gather(
                hs, 1, pos_idx_dev[:, : len(pos_names)].unsqueeze(-1).expand(B, len(pos_names), H)
            )
            for pi, name in enumerate(pos_names):
                by_name[name] = picked[:, pi]
        stacked = torch.stack([by_name[n] for n in names], dim=1)  # (B, S, H)
        per_layer.append(stacked.to(torch.float16).cpu())
    capture.latest.clear()
    return torch.stack(per_layer, dim=2)  # (B, S, Lc, H) fp16 CPU


# Sentinel writer relocated to issue928_common.write_sentinel (round 2): the
# extract phase now emits epm:progress and the run_all finalize step emits the
# ONE epm:results sentinel at true end-of-workload, so both share one writer.


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:  # noqa: C901 — linear phase pipeline (gate→G→P→B→U); see phase() markers
    ap = argparse.ArgumentParser(description="Issue #928: thinking-model rollouts + summary store")
    ap.add_argument("--model", default=THINKING_MODEL)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--gpu", action="store_true", help="force --device cuda")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "issue_928"))
    ap.add_argument("--battery", default=None, help="local battery.json fast path (sha-pinned)")
    ap.add_argument("--contexts", type=int, default=None, help="cap contexts (smoke=5)")
    ap.add_argument("--probes", type=int, default=None, help="cap probes/context")
    ap.add_argument("--gate-contexts", type=int, default=5, help="Phase-0 gate slice size")
    ap.add_argument("--rung", choices=["auto", *FALLBACK_RUNGS], default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION)
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    ap.add_argument("--batch-probes", type=int, default=8)
    ap.add_argument(
        "--capture-token-budget",
        type=int,
        default=32768,
        help="max BxT padded tokens per capture forward (bounds the 28-layer hook footprint)",
    )
    ap.add_argument("--expected-layers", type=int, default=EXPECTED_LAYERS)
    ap.add_argument("--expected-hidden", type=int, default=EXPECTED_HIDDEN)
    ap.add_argument("--skip-gen", action="store_true", help="reuse rollouts already in out-dir")
    ap.add_argument(
        "--synthetic-completions",
        action="store_true",
        help="CPU smoke ONLY: replace the vLLM call with deterministic synthetic "
        "<think> completions (every other phase runs the production path)",
    )
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="label + relax model-shape asserts")
    args = ap.parse_args()

    device = args.device or ("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    rollouts_dir = out_dir / "raw_completions" / "thinking_rollouts"
    store_dir = out_dir / "store" / "percq_summaries"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    phase("setup")
    battery = resolve_battery(Path(args.battery) if args.battery else None)
    ctx_ids_all, families = context_order_and_families(battery)
    instances = {i["id"]: i for i in battery["instances"]}
    ctx_ids = ctx_ids_all[: args.contexts] if args.contexts else ctx_ids_all
    probes = load_probe_pool()
    if args.probes:
        probes = probes[: args.probes]
    pool_hash = probes_hash(probes)
    logger.info(
        "contexts=%d probes=%d model=%s device=%s", len(ctx_ids), len(probes), args.model, device
    )

    # Tokenizer first (CPU): template/delimiter asserts before any GPU spend.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for tag in (THINK_OPEN, THINK_CLOSE):
        rt = tokenizer.decode(tokenizer.encode(tag, add_special_tokens=False))
        assert rt == tag, f"delimiter round-trip drift: {tag!r} -> {rt!r}"
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    if nl_ids != [TURN_NL_TOKEN_ID]:
        raise RuntimeError(f"newline id drift: {nl_ids} != [{TURN_NL_TOKEN_ID}]")

    # ── Phase 0 + G: gate walk, then full generation at the chosen rung ──────
    gate_ctx = ctx_ids[: min(args.gate_contexts, len(ctx_ids))]
    rungs_to_try = list(FALLBACK_RUNGS) if args.rung == "auto" else [args.rung]
    llm = None

    def _generate(prompt_texts: list[str], rung: str, max_new: int) -> list[tuple[str, str]]:
        nonlocal llm
        if args.synthetic_completions:
            return synthetic_completions(prompt_texts, rung, len(probes))
        if llm is None:
            phase("vllm_init")
            llm = build_vllm_engine(args.model, args.gpu_memory_utilization, args.max_model_len)
        return vllm_generate_chunked(llm, prompt_texts, sampling_params_for_rung(rung, max_new))

    phase("gate")
    chosen_rung = None
    gate_reports: dict[str, dict] = {}
    gate_completions: dict[str, list[tuple[str, str]]] = {}
    for rung in rungs_to_try:
        prompts = [
            build_prompt_text(tokenizer, instances[c], q, rung) for c in gate_ctx for q in probes
        ]
        comps = _generate(prompts, rung, args.max_new_tokens)
        rows = parse_rows(tokenizer, comps, rung)
        report = gate1_check(rows, args.max_new_tokens)
        gate_reports[rung] = report
        logger.info(
            "[gate] rung=%s pass=%s parse_rate=%.3f p95=%.0f offenders=%d offender_rate=%.4f",
            rung,
            report["pass"],
            report["parse_rate"],
            report["p95_gen_tokens"],
            report["repetition_offenders"],
            report["repetition_offender_rate"],
        )
        if report["pass"]:
            chosen_rung = rung
            gate_completions = {
                c: comps[ci * len(probes) : (ci + 1) * len(probes)] for ci, c in enumerate(gate_ctx)
            }
            break
    dump_json(
        {"gate_reports": gate_reports, "chosen_rung": chosen_rung, "gate_contexts": gate_ctx},
        out_dir / "gate_report.json",
    )
    if chosen_rung is None:
        # Rung-(iii) exhaustion is TERMINAL (plan §7 kill criteria): the design
        # premise (parseable CoT on the maximal-similarity model) is unmet. The
        # orchestrator posts epm:failure failure_class: data off this sentinel.
        phase("failed")
        write_sentinel(
            "epm:failure",
            {
                "failure_class": "data",
                "reason": "gate1_parse_floor_all_rungs_exhausted",
                "gate_reports": gate_reports,
            },
            out_dir,
        )
        if llm is not None:
            _reap_vllm(llm)
        return 3

    # Per-context rollout persistence + resume validation (round 2: per-GROUP
    # durable writes — a crash after completed generation groups loses nothing,
    # and --skip-gen reuses only files matching every output-affecting run arg).
    def _persist_rollout(c: str, regen_rows: list[int] | None = None) -> None:
        """Persist one context's rollout TEXT verbatim (§4.8/#779) — the moment
        its generation returns, BEFORE any parse/capture."""
        blob = {
            "context_id": c,
            "family": families[c],
            "rung": chosen_rung,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "probe_pool_hash": pool_hash,
            "completions": [
                {"probe": q, "completion": t, "finish_reason": fr}
                for q, (t, fr) in zip(probes, completions_by_ctx[c], strict=True)
            ],
        }
        if regen_rows is not None:
            blob["regen_16k_rows"] = regen_rows
        dump_json(blob, rollouts_dir / f"{c}.json")

    def _rollout_blob_mismatch(blob: dict, c: str) -> str:
        """Skip-if-valid predicate for --skip-gen reuse, keyed on every
        output-affecting run arg. Returns "" (reusable) or the mismatched key
        (regenerate — never silently reuse a wrong cached rollout, #722 r3)."""
        for key, want in (
            ("context_id", c),
            ("model", args.model),
            ("rung", chosen_rung),
            ("probe_pool_hash", pool_hash),
            ("max_new_tokens", args.max_new_tokens),
        ):
            if blob.get(key) != want:
                return key
        if [r.get("probe") for r in blob.get("completions", [])] != probes:
            return "probe_list"
        return ""

    phase("generate")
    completions_by_ctx: dict[str, list[tuple[str, str]]] = {}
    loaded_from_disk: set[str] = set()
    if args.skip_gen:
        for c in ctx_ids:
            p = rollouts_dir / f"{c}.json"
            if not p.is_file():
                continue
            blob = json.loads(p.read_text())
            why = _rollout_blob_mismatch(blob, c)
            if why:
                logger.warning(
                    "[skip-gen] rollout %s.json stale (%s mismatch) — regenerating", c, why
                )
                continue
            completions_by_ctx[c] = [
                (r["completion"], r.get("finish_reason", "stop")) for r in blob["completions"]
            ]
            loaded_from_disk.add(c)
        logger.info(
            "[skip-gen] reusing %d/%d persisted rollout files", len(loaded_from_disk), len(ctx_ids)
        )
    for c, comps_c in gate_completions.items():
        if c not in completions_by_ctx:
            completions_by_ctx[c] = comps_c
            _persist_rollout(c)  # gate contexts persist the moment the gate passes
    remaining = [c for c in ctx_ids if c not in completions_by_ctx]
    if remaining:
        # Generate in context GROUPS (~one vLLM chunk each) and persist each
        # group's rollout JSONs the moment it returns — previously ALL
        # remaining contexts generated before ANY rollout file landed, so a
        # late-generation crash lost every completed chunk (r1 blocker).
        ctx_per_group = max(1, VLLM_CHUNK_SIZE // max(1, len(probes)))
        n_groups = (len(remaining) + ctx_per_group - 1) // ctx_per_group
        for gi in range(0, len(remaining), ctx_per_group):
            group = remaining[gi : gi + ctx_per_group]
            prompts = [
                build_prompt_text(tokenizer, instances[c], q, chosen_rung)
                for c in group
                for q in probes
            ]
            comps = _generate(prompts, chosen_rung, args.max_new_tokens)
            for ci, c in enumerate(group):
                completions_by_ctx[c] = comps[ci * len(probes) : (ci + 1) * len(probes)]
                _persist_rollout(c)
            logger.info(
                "[generate] group %d/%d done — %d/%d remaining context(s) persisted",
                gi // ctx_per_group + 1,
                n_groups,
                min(gi + ctx_per_group, len(remaining)),
                len(remaining),
            )
    logger.info("rollout files present for all %d contexts in %s", len(ctx_ids), rollouts_dir)

    phase("parse")
    parse_by_ctx: dict[str, list[dict]] = {
        c: parse_rows(tokenizer, completions_by_ctx[c], chosen_rung) for c in ctx_ids
    }
    all_rows = [r for c in ctx_ids for r in parse_by_ctx[c]]
    trunc = [r for r in all_rows if r["finish_reason"] == "length"]
    trunc_frac = len(trunc) / max(1, len(all_rows))
    regen_16k = False
    if trunc_frac > TRUNCATION_REGEN_FRAC and not args.skip_gen:
        # One 16,384 re-generation rung for the cap-truncated rows (plan §4.4).
        phase("regen16k")
        regen_16k = True
        targets = [
            (c, qi)
            for c in ctx_ids
            for qi, r in enumerate(parse_by_ctx[c])
            if r["finish_reason"] == "length"
        ]
        prompts = [
            build_prompt_text(tokenizer, instances[c], probes[qi], chosen_rung) for c, qi in targets
        ]
        comps = _generate(prompts, chosen_rung, MAX_NEW_TOKENS_RETRY)
        for (c, qi), new in zip(targets, comps, strict=True):
            completions_by_ctx[c][qi] = new
        for c in {c for c, _qi in targets}:
            _persist_rollout(c, regen_rows=[qi for cc, qi in targets if cc == c])
            parse_by_ctx[c] = parse_rows(tokenizer, completions_by_ctx[c], chosen_rung)
    parse_report = {
        c: {
            "n_rows": len(parse_by_ctx[c]),
            "n_well_formed": sum(1 for r in parse_by_ctx[c] if r["well_formed"]),
            "parse_rate": sum(1 for r in parse_by_ctx[c] if r["well_formed"])
            / max(1, len(parse_by_ctx[c])),
            "reasons": {
                reason: sum(1 for r in parse_by_ctx[c] if r["reason"] == reason)
                for reason in {r["reason"] for r in parse_by_ctx[c] if r["reason"]}
            },
        }
        for c in ctx_ids
    }
    flagged = [c for c in ctx_ids if parse_report[c]["parse_rate"] < PARSE_RATE_FLOOR]
    if flagged:
        logger.warning(
            "%d context(s) below the %.0f%% parse floor (kept; sensitivity re-fit "
            "downstream excludes them): %s",
            len(flagged),
            100 * PARSE_RATE_FLOOR,
            flagged,
        )

    # ── Phase B: reap vLLM, teacher-forced capture, per-part summaries ────────
    if llm is not None:
        phase("reap_vllm")
        _reap_vllm(llm)  # gotchas.md: workers survive `del llm`; HF load OOMs otherwise
        llm = None

    phase("capture")
    from transformers import AutoModelForCausalLM

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    n_layers = model.config.num_hidden_layers
    if not args.smoke:
        assert n_layers == args.expected_layers, (n_layers, args.expected_layers)
        assert model.config.hidden_size == args.expected_hidden, model.config.hidden_size
    capture_layers = list(range(n_layers))
    capture = LayerCapture(model, n_layers)
    per_ctx_capture: dict[str, dict] = {}

    def _reusable_store_blob(path: Path, c: str) -> tuple[dict | None, str]:
        """Thin closure over the module-level ``reusable_store_blob`` predicate
        (round 3: hoisted + pytest-pinned), binding this run's output-affecting
        args + the context's CURRENT rollout-content digest — so a blob
        captured from different completions (changed cap / regenerated
        rollouts) is invalidated and recaptured, never silently reused."""
        return reusable_store_blob(
            path,
            c,
            model_name=args.model,
            family=families[c],
            rung=chosen_rung,
            probe_pool_hash=pool_hash,
            capture_layers=capture_layers,
            summary_names=list(SUMMARY_NAMES),
            n_probes=len(probes),
            max_new_tokens=args.max_new_tokens,
            rollout_digest=rollout_content_digest(probes, completions_by_ctx[c]),
            hidden_size=int(model.config.hidden_size),
        )

    try:
        for ci, c in enumerate(ctx_ids):
            blob_path = store_dir / f"{c}.pt"
            if blob_path.is_file():
                prior, why = _reusable_store_blob(blob_path, c)
                if prior is not None:
                    per_ctx_capture[c] = {
                        "n_captured": len(prior["probe_indices"]),
                        "drop_reasons": prior["coverage"]["capture_drop_reasons"],
                        "resumed": True,
                    }
                    logger.info(
                        "[capture] %d/%d %s: SKIPPED (valid existing store blob — resume)",
                        ci + 1,
                        len(ctx_ids),
                        c,
                    )
                    continue
                logger.warning("[capture] %s: existing blob invalid (%s) — recapturing", c, why)
            rows, kept_qi, drop_reasons = [], [], {}
            for qi, (q, (text, _fr)) in enumerate(zip(probes, completions_by_ctx[c], strict=True)):
                rec = parse_by_ctx[c][qi]
                if not rec["well_formed"]:
                    continue
                row, why = build_capture_row(tokenizer, instances[c], q, text, rec, chosen_rung)
                if row is None:
                    drop_reasons[why] = drop_reasons.get(why, 0) + 1
                    continue
                rows.append(row)
                kept_qi.append(qi)
            if not rows:
                raise RuntimeError(f"context {c}: zero capturable rows (coverage collapse)")
            chunks: list[torch.Tensor] = []
            order: list[int] = []
            for batch_idx in pack_batches(rows, args.batch_probes, args.capture_token_budget):
                batch_rows = [rows[i] for i in batch_idx]
                chunks.append(
                    reduce_forward_batch(model, capture, capture_layers, tokenizer, batch_rows)
                )
                order.extend(batch_idx)
            stacked = torch.cat(chunks, dim=0)  # (n_rows, 12, Lc, H) in packed order
            inv = torch.empty(len(order), dtype=torch.long)
            inv[torch.tensor(order)] = torch.arange(len(order))
            per_q = stacked[inv]  # restore kept_qi order
            probe_avg = per_q.float().mean(dim=0).to(torch.float16)  # (12, Lc, H)
            blob = {
                "context_id": c,
                "family": families[c],
                "rung": chosen_rung,
                "capture_layers": capture_layers,
                "summary_names": list(SUMMARY_NAMES),
                "probe_indices": kept_qi,
                "per_q": per_q,  # (n_rows, 12, Lc, H) fp16
                "probe_avg": probe_avg,  # (12, Lc, H) fp16
                "coverage": {
                    "n_probes_total": len(probes),
                    "n_well_formed": parse_report[c]["n_well_formed"],
                    "n_captured": len(kept_qi),
                    "capture_drop_reasons": drop_reasons,
                },
                "probe_pool_hash": pool_hash,
                "model": args.model,
                # round 3: generation-output identity, validated by
                # reusable_store_blob — a resume after rollout regeneration
                # (changed cap / content) RECAPTURES instead of reusing.
                "max_new_tokens": args.max_new_tokens,
                "rollout_digest": rollout_content_digest(probes, completions_by_ctx[c]),
            }
            # atomic write: a crash mid-save never leaves a live-looking blob
            # (the resume predicate would catch it as unreadable anyway).
            tmp = blob_path.with_suffix(".pt.tmp")
            torch.save(blob, tmp)
            os.replace(tmp, blob_path)
            per_ctx_capture[c] = {"n_captured": len(kept_qi), "drop_reasons": drop_reasons}
            logger.info(
                "[capture] %d/%d %s: %d/%d rows captured",
                ci + 1,
                len(ctx_ids),
                c,
                len(kept_qi),
                len(probes),
            )
    finally:
        capture.remove()

    manifest = {
        "context_ids": ctx_ids,
        "families": {c: families[c] for c in ctx_ids},
        "capture_layers": capture_layers,
        "summary_names": list(SUMMARY_NAMES),
        "hidden_size": int(model.config.hidden_size),
        "rung": chosen_rung,
        "regen_16k": regen_16k,
        "truncation_frac_pre_regen": trunc_frac,
        "gate_reports": gate_reports,
        "parse_report": parse_report,
        "flagged_below_parse_floor": flagged,
        "per_ctx_capture": per_ctx_capture,
        "probe_pool_hash": pool_hash,
        "n_probes": len(probes),
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "reproducibility": reproducibility_metadata(),
        "smoke": args.smoke,
    }
    dump_json(manifest, out_dir / "store" / "manifest.json")
    logger.info("wrote store manifest (%d contexts)", len(ctx_ids))

    hf_paths = {}
    if not args.no_upload:
        phase("upload")
        hf_paths["raw_completions"] = upload_folder_scoped_verify(
            rollouts_dir,
            RAW_COMPLETIONS_PREFIX + ("_smoke" if args.smoke else ""),
            [f"{c}.json" for c in ctx_ids],
            f"issue #928: thinking rollouts ({len(ctx_ids)} contexts, rung={chosen_rung})",
            allow_patterns=["*.json"],
        )
        hf_paths["store"] = upload_folder_scoped_verify(
            store_dir,
            STORE_PREFIX + ("_smoke" if args.smoke else ""),
            [f"{c}.pt" for c in ctx_ids],
            f"issue #928: per-(C,q) summary store ({len(ctx_ids)} contexts)",
            allow_patterns=["*.pt"],
        )
        # Manifest rides the store prefix's parent (small JSON, non-LFS).
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(out_dir / "store" / "manifest.json"),
            path_in_repo=f"{STORE_PREFIX}{'_smoke' if args.smoke else ''}/manifest.json",
            repo_id="superkaiba1/explore-persona-space-data",
            repo_type="dataset",
            commit_message="issue #928: store manifest",
        )

    note = {
        "phase": "extract_thinking_store",
        "n_contexts": len(ctx_ids),
        "rung": chosen_rung,
        "gate": gate_reports.get(chosen_rung, {}),
        "flagged_below_parse_floor": flagged,
        "hf_paths": hf_paths,
        "elapsed_s": round(time.time() - t0, 1),
    }
    # epm:progress, NOT epm:results (round-2 fix): the extract end is
    # mid-pipeline — the ONE results sentinel fires from the run_all driver's
    # finalize step after fits + figures + uploads (issue928_finalize.py).
    write_sentinel("epm:progress", note, out_dir)
    # NOT [phase=done]: the run_all driver owns the terminal phase line (a
    # premature done here would false-signal completion to the poller while
    # the fit phases still run).
    phase("extract_done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logger.error("[phase=failed] extraction crashed:\n%s", traceback.format_exc())
        raise
