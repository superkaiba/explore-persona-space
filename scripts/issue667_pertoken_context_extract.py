#!/usr/bin/env python3
"""Issue #667 per-CONTEXT-TOKEN activation-shift extractor (one source-adapter cell).

The CONTEXT-side sibling of :mod:`issue667_pertoken_extract`. That script reduced
the per-ANSWER-token residual shift (base θ0 -> post-FT θ⁺) over the RESPONSE span
``[prompt_len : span_end)``; this one reduces the SAME two metrics over the
PROMPT/CONTEXT span ``[0 : prompt_len)`` of the IDENTICAL teacher-forced forward
pass. No new generation is needed — the prompt tokens live in the same
``T_{C'}(q) + R`` pass the answer read already used; we simply reduce the OTHER
span.

Two per-(position, layer) metrics vs base, exactly as the answer script:

  - magnitude change: ``||h+_{t,L} - h_{t,L}|| / ||h_{t,L}||``   (relative L2 shift)
  - direction change: ``cos(h_{t,L}, h+_{t,L})``                 (in [-1, 1])

TWO alignments of the context position axis (the answer script has ONE — from the
start of the answer span):

  - **from-start**: index 0 = the FIRST prompt token, up to ``--max-ctx-pos`` = 128
    positions forward. Captures the system-prompt / persona-preamble region.
  - **from-end**: offset 0 = the LAST input token (== the ``c_C`` last-input-token
    the #667 SUMMARY read + :func:`issue667_extract._context_vector_all_layers`
    analyze), offset 1 = second-to-last, ... up to ``--max-ctx-offset`` = 64
    offsets back. This is the alignment the COMBINED context+answer replot stacks
    directly above the answer span (last-input token at the context/answer
    boundary).

SELF-CHECK (brief §1): from-end OFFSET 0 must reproduce the SUMMARY ``Δc``
magnitude/rotation at the last-input token — the metric computed from
``_context_vector_all_layers`` (base θ0 vs post-FT θ⁺, last-input-token, all
layers). This extractor computes that reference read once per cell (source
diagonal) and ASSERTS the streamed from-end offset-0 read matches it within a
tolerance (per-layer), then records the max/mean discrepancy in the npz + log.

Streaming, OOM-safe (the #671/#545 trap): per-cell we hold only the tiny
accumulators — from-start ``[max_ctx_pos, N_LAYERS]`` + from-end
``[max_ctx_offset, N_LAYERS]`` mag/dir/count float64 (~a few hundred KB) — plus,
at any instant, ONE probe's two ``(prompt_len, 28, 3584)`` hidden-state tensors
(freed before the next probe via the reduce-on-the-fly path
:func:`extract_layer_activations`). No full-grid retain.

CONTENT HYGIENE: ``em`` probes are Betley harmful-content-adjacent; this script
NEVER prints/logs probe or response text — only row/token COUNT + the reduced
scalar ACTIVATIONS. Benign behaviors (marker/fact/sycophancy) unaffected. No raw
completions are written.

Usage (one source-adapter cell)::

    uv run python scripts/issue667_pertoken_context_extract.py \\
        --behavior em --source-cid default \\
        --targets sp_swe,default,fmt_json \\
        --max-ctx-pos 128 --max-ctx-offset 64 \\
        --out eval_results/issue_667_pertoken_context/analysis_tensors --gpu-id 0

Smoke (CPU, tiny stub model, capped probes/targets)::

    uv run python scripts/issue667_pertoken_context_extract.py \\
        --behavior em --source-cid default --targets default \\
        --cpu-only --max-probes 2 --max-ctx-pos 12 --max-ctx-offset 8 \\
        --skip-adapter-gauge --out /tmp/i667ctx_smoke
"""

# math/scientific notation in docstrings + messages

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
# vLLM EngineCore fork() poisoning guard (mirrors issue667_pertoken_extract):
# main() touches transformers.AutoTokenizer BEFORE vllm.LLM() constructs, and any
# pre-LLM() transformers/tokenizer touch poisons the EngineCore fork; spawn (not
# fork) avoids the silent worker death. Must be set BEFORE any `import vllm`.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# scripts/ on the path so the cross-script `import issue667_extract` /
# `import issue667_pertoken_extract` resolve cwd-independently.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse issue667_extract's model-load / adapter / teacher-force / probe helpers
# AND issue667_pertoken_extract's per-cell driver scaffolding verbatim (single
# source of truth for token-position + layer-index conventions).
import issue667_extract as ix  # noqa: E402
import issue667_pertoken_extract as pt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# DOTENV_LINT_EXEMPT: exploratory user-directed script; shell exports cover pod/GCE.
from dotenv import load_dotenv  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.analysis.issue667 import (  # noqa: E402
    BASE_MODEL,
    HIDDEN_SIZE,
    N_LAYERS,
)

load_dotenv()

logger = logging.getLogger("issue667_pertoken_context_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MAX_CTX_POS_DEFAULT = 128  # cap the from-start context position axis (brief §1)
MAX_CTX_OFFSET_DEFAULT = 64  # cap the from-end context offset axis (brief §1)
# self-check tolerance: from-end offset-0 vs the SUMMARY Δc last-input-token read.
# The two reads use the SAME base θ0 + PeftModel θ⁺ and the SAME last-input-token
# residual, so they should be bit-close; bf16 forwards + the accumulate-in-float64
# path give a tiny spread. 1e-2 (relative-L2 mag ~O(0.01-0.5), cos ~[0.9,1.0]).
SELFCHECK_TOL = 1e-2
CELL_DONE_SENTINEL = ".done"


# ─────────────────────────────────────────────────────────────────────────────
# Per-context-token span reduction (two alignments), streaming-safe
# ─────────────────────────────────────────────────────────────────────────────


def _mag_cos_columns(hb: torch.Tensor, ht: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Per-position (magnitude, direction) shift vs base for a (P, H) residual pair.

    Byte-identical reduction to :func:`issue667_pertoken_extract._per_token_shift_scalars`:

        magnitude: ||h+ - h|| / ||h||   (relative L2 shift; eps-guarded)
        direction: cos(h, h+)           (clamped to [-1, 1])

    ``hb`` / ``ht`` are (P, H) float32 (base / trained). Returns two (P,) numpy
    float64 arrays.
    """
    eps = 1e-8
    base_norm = hb.norm(dim=-1)  # (P,)
    shift_norm = (ht - hb).norm(dim=-1)  # (P,)
    mag = (shift_norm / (base_norm + eps)).cpu().numpy().astype(np.float64)
    dot = (hb * ht).sum(dim=-1)
    cos = (
        (dot / (base_norm * ht.norm(dim=-1) + eps))
        .clamp(-1.0, 1.0)
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    return mag, cos


@torch.no_grad()
def _per_context_token_shift_scalars(
    base_model,
    trained_model,
    tok,
    messages: list[dict],
    response: str,
    device,
    *,
    max_ctx_pos: int,
    max_ctx_offset: int,
) -> dict[str, object]:
    """Per-(context-token, layer) magnitude + direction shift vs base, both alignments.

    Teacher-forces ``messages + response`` through base θ0 AND adapter θ⁺ ONCE
    each (memory-safe hook path over ALL 28 block layers), then reduces the
    PROMPT/CONTEXT span ``[0 : prompt_len)`` of the pair to the two scalars per
    (context-position, layer) under BOTH alignments.

    Reuses :func:`issue667_pertoken_extract._answer_span_ids` for the byte-faithful
    ``(full_ids, prompt_len, span_end)`` split — the answer span it returns is
    ``[prompt_len : span_end)``; the CONTEXT span read here is ``[0 : prompt_len)``.

    Returns::

        {
          "start_mag": (max_ctx_pos, n_layers),  "start_cos": (max_ctx_pos, n_layers),
          "end_mag":   (max_ctx_offset, n_layers),"end_cos":  (max_ctx_offset, n_layers),
          "n_start": int,   # valid from-start rows = min(prompt_len, max_ctx_pos)
          "n_end": int,     # valid from-end rows   = min(prompt_len, max_ctx_offset)
        }

    Rows beyond the valid count are 0.0 and MUST NOT be counted by the caller.
    """
    full_ids, p, _span_end = pt._answer_span_ids(tok, messages, response)
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    n_layers = getattr(base_model.config, "num_hidden_layers", N_LAYERS)
    layers = list(range(n_layers))
    # Memory-safe subset read: hook every block layer li == hs[li+1] (the SAME
    # tensor the answer read + the SUMMARY read use); unused layers freed as the
    # forward proceeds (never a full-seq x (L+1) retain).
    acts_b = extract_layer_activations(base_model, ids, layers)
    acts_t = extract_layer_activations(trained_model, ids, layers)

    prompt_len = p  # context span is [0 : prompt_len)
    n_start = min(prompt_len, max_ctx_pos)
    n_end = min(prompt_len, max_ctx_offset)
    start_mag = np.zeros((max_ctx_pos, n_layers), dtype=np.float64)
    start_cos = np.zeros((max_ctx_pos, n_layers), dtype=np.float64)
    end_mag = np.zeros((max_ctx_offset, n_layers), dtype=np.float64)
    end_cos = np.zeros((max_ctx_offset, n_layers), dtype=np.float64)
    for li in layers:
        # (prompt_len, H) float32 residuals over the CONTEXT span, base + trained.
        hb_full = acts_b[li][0, :prompt_len, :].float()
        ht_full = acts_t[li][0, :prompt_len, :].float()
        # from-start: positions 0..n_start-1 (first prompt token forward).
        mag_s, cos_s = _mag_cos_columns(hb_full[:n_start], ht_full[:n_start])
        start_mag[:n_start, li] = mag_s
        start_cos[:n_start, li] = cos_s
        # from-end: offset 0 == LAST input token (index prompt_len-1), offset 1 ==
        # prompt_len-2, ... Reverse the last n_end rows so row r == offset r.
        hb_end = torch.flip(hb_full[prompt_len - n_end : prompt_len], dims=[0])
        ht_end = torch.flip(ht_full[prompt_len - n_end : prompt_len], dims=[0])
        mag_e, cos_e = _mag_cos_columns(hb_end, ht_end)
        end_mag[:n_end, li] = mag_e
        end_cos[:n_end, li] = cos_e
    return {
        "start_mag": start_mag,
        "start_cos": start_cos,
        "end_mag": end_mag,
        "end_cos": end_cos,
        "n_start": n_start,
        "n_end": n_end,
    }


@torch.no_grad()
def _summary_delta_c_last_token(
    base_model, trained_model, tok, messages: list[dict], device
) -> tuple[np.ndarray, np.ndarray]:
    """The SUMMARY Δc reference: last-input-token magnitude + cos, all layers.

    Reads the last-input-token residual (all 28 layers) under
    ``add_generation_prompt=True`` through base θ0 AND trained θ⁺ — the SAME
    read :func:`issue667_extract._context_vector_all_layers` performs for the
    #667 summary ``c_C`` / ``c_C_postft`` — then reduces the pair with the SAME
    (magnitude, direction) metric as the streamed reads. Returns two (n_layers,)
    numpy float64 arrays (mag, cos), the from-end OFFSET-0 reference the
    self-check compares against.
    """
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    n_layers = getattr(base_model.config, "num_hidden_layers", N_LAYERS)
    layers = list(range(n_layers))
    acts_b = extract_layer_activations(base_model, ids["input_ids"], layers)
    acts_t = extract_layer_activations(trained_model, ids["input_ids"], layers)
    hb = torch.stack([acts_b[li][0, -1, :].float() for li in layers])  # (L, H)
    ht = torch.stack([acts_t[li][0, -1, :].float() for li in layers])  # (L, H)
    mag, cos = _mag_cos_columns(hb, ht)
    return mag, cos  # (n_layers,), (n_layers,)


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell driver (streaming per-(pos, L) accumulator over both alignments)
# ─────────────────────────────────────────────────────────────────────────────


def _generate_base_R_lowmem(tok, registry, demos, behavior, targets, probes, device, args) -> dict:
    """Batched vLLM greedy base R (Phase A), at a LOWER gpu_memory_utilization.

    Mirrors :func:`issue667_pertoken_extract._generate_base_R` (same generator,
    same order, ``{(tcid, probe_index): response}``) but threads
    ``args.vllm_gpu_mem_util`` (default 0.6) into ``ix.vllm_generate_R`` instead
    of its 0.85 default. Rationale (crash fix): the 8-cells-per-wave dispatcher
    tears down each wave's vLLM engines then constructs the next wave's on the
    SAME GPUs; vLLM V1 EngineCore teardown is ASYNC (gotchas.md § vLLM teardown),
    so a fresh 0.85-util engine can lose the free-HBM race against a not-yet-fully
    -reaped prior-wave worker (+ the pod's persistent ~1.2 GB zombie context) and
    die at init (``EngineCore_DP0: 1``). 0.6 leaves ~30 GB headroom for the
    transient overlap; greedy generation needs far less KV cache than 0.85. On
    CPU returns {} (vLLM unavailable; caller falls back to HF greedy).
    """
    if device.type == "cpu":
        return {}
    gen_msgs: list[list[dict]] = []
    gen_keys: list[tuple[str, int]] = []
    for tcid in targets:
        for qi, q in enumerate(probes):
            gen_msgs.append(ix.build_messages_for(registry, demos, tcid, behavior, q))
            gen_keys.append((tcid, qi))
    logger.info(
        "Phase A: vLLM-generating %d base R responses (gpu_mem_util=%.2f)",
        len(gen_msgs),
        args.vllm_gpu_mem_util,
    )
    responses = ix.vllm_generate_R(
        tok, gen_msgs, max_new_tokens=args.max_new_tokens, gpu_mem_util=args.vllm_gpu_mem_util
    )
    return dict(zip(gen_keys, responses, strict=True))


def write_cell_done_sentinel(cell_dir: Path, payload: dict) -> Path:
    """Atomically stamp the cell's .done sentinel AFTER the npz is on disk.

    Atomic = write-temp-then-os.replace within the same dir, so a crash mid-write
    never leaves a half-written .done that the dispatcher resume-skip would trust
    (mirrors issue667_pertoken_extract.write_cell_done_sentinel).
    """
    final = cell_dir / CELL_DONE_SENTINEL
    tmp = cell_dir / f"{CELL_DONE_SENTINEL}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, final)
    return final


def run_extraction(args) -> int:
    from explore_persona_space.experiments.i537_contexts import (
        load_icl_demos,
        load_registry,
    )

    device = ix._device(args.gpu_id, args.cpu_only)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    max_ctx_pos = int(args.max_ctx_pos)
    max_ctx_offset = int(args.max_ctx_offset)
    assert max_ctx_pos > 0 and max_ctx_offset > 0, (max_ctx_pos, max_ctx_offset)

    sampled_path, demos_path = ix.stage_inputs()
    registry = load_registry(sampled_path)
    demos = load_icl_demos(demos_path)

    behavior = args.behavior
    source_cid = args.source_cid
    seed = args.seed
    targets = pt._resolve_targets(behavior, source_cid, args.targets)
    probes = ix.load_eval_probes(behavior)
    if args.max_probes:
        probes = probes[: args.max_probes]
    logger.info(
        "pertoken CONTEXT cell behavior=%s source=%s seed=%d | %d targets x %d probes "
        "| max_ctx_pos=%d max_ctx_offset=%d",
        behavior,
        source_cid,
        seed,
        len(targets),
        len(probes),
        max_ctx_pos,
        max_ctx_offset,
    )

    # Stage + verify the adapter gauge BEFORE any GPU work (cheap, HALT early).
    # --skip-adapter-gauge is the CPU-smoke escape (no #537 adapter on the VM):
    # a tiny stub PeftModel is built so the reduce/accumulator path runs.
    if args.skip_adapter_gauge:
        tok, base, trained, n_layers = pt._build_cpu_stub_models()
        r_lookup: dict[tuple[str, int], str] = {}
    else:
        adapter_dir = ix.stage_adapter_local(behavior, source_cid, seed)
        gauge = ix.assert_adapter_gauge(adapter_dir, behavior)
        logger.info(
            "adapter gauge OK: %s", {k: gauge[k] for k in ("r", "lora_alpha", "use_rslora")}
        )
        # ── Phase A: vLLM batched base R (per CLAUDE.md; HF fallback on CPU) ──
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
        r_lookup = _generate_base_R_lowmem(
            tok, registry, demos, behavior, targets, probes, device, args
        )
        # ── Phase B: load base θ0 + trained θ⁺ for the teacher-force reads ──
        _, base, trained = ix.load_base_and_trained(adapter_dir, device, dtype)
        assert base.config.hidden_size == HIDDEN_SIZE or device.type == "cpu", (
            base.config.hidden_size
        )
        n_layers = getattr(base.config, "num_hidden_layers", N_LAYERS)

    start_mag_sum = np.zeros((max_ctx_pos, n_layers), dtype=np.float64)
    start_dir_sum = np.zeros((max_ctx_pos, n_layers), dtype=np.float64)
    start_count = np.zeros((max_ctx_pos, n_layers), dtype=np.int64)
    end_mag_sum = np.zeros((max_ctx_offset, n_layers), dtype=np.float64)
    end_dir_sum = np.zeros((max_ctx_offset, n_layers), dtype=np.float64)
    end_count = np.zeros((max_ctx_offset, n_layers), dtype=np.int64)

    n_gen = n_empty = 0
    for tcid in targets:
        for qi, q in enumerate(probes):
            tmsgs = ix.build_messages_for(registry, demos, tcid, behavior, q)
            r = r_lookup.get((tcid, qi))
            if r is None:
                r = ix._greedy_response(base, tok, tmsgs, device, args.max_new_tokens)
            n_gen += 1
            if not r.strip():
                n_empty += 1
                continue
            out = _per_context_token_shift_scalars(
                base,
                trained,
                tok,
                tmsgs,
                r,
                device,
                max_ctx_pos=max_ctx_pos,
                max_ctx_offset=max_ctx_offset,
            )
            ns, ne = out["n_start"], out["n_end"]
            start_mag_sum[:ns, :] += out["start_mag"][:ns, :]
            start_dir_sum[:ns, :] += out["start_cos"][:ns, :]
            start_count[:ns, :] += 1
            end_mag_sum[:ne, :] += out["end_mag"][:ne, :]
            end_dir_sum[:ne, :] += out["end_cos"][:ne, :]
            end_count[:ne, :] += 1
            del out
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── SELF-CHECK: from-end offset-0 vs SUMMARY Δc last-input-token (brief §1) ──
    # Use the SOURCE-diagonal context (source_cid as its own target) — the c_C the
    # #667 summary analyzes. from-end offset 0 IS the last input token, so its
    # count-weighted mean over the source-diagonal probes must match the mean of
    # the per-probe Δc last-token reads within SELFCHECK_TOL. Reuses the Phase-A
    # r_lookup (source_cid is always in `targets`, so its R is already generated)
    # — NO second vLLM engine (a fresh LLM() after the HF base+PeftModel are
    # resident on the GPU OOMs at gpu_memory_utilization; the source R is a
    # pure teacher-force + Δc read on the already-loaded models).
    selfcheck = _run_selfcheck(
        base,
        trained,
        tok,
        registry,
        demos,
        behavior,
        source_cid,
        probes,
        device,
        args,
        n_layers,
        r_lookup,
    )

    covered_start = int((start_count > 0).sum())
    covered_end = int((end_count > 0).sum())
    logger.info(
        "cell %s/%s done: %d gens (%d empty); from-start %d/%d cells covered, "
        "from-end %d/%d cells covered; selfcheck max_mag_diff=%.4g max_cos_diff=%.4g PASS=%s",
        behavior,
        source_cid,
        n_gen,
        n_empty,
        covered_start,
        max_ctx_pos * n_layers,
        covered_end,
        max_ctx_offset * n_layers,
        selfcheck["max_mag_diff"],
        selfcheck["max_cos_diff"],
        selfcheck["passed"],
    )

    out_root = Path(args.out)
    cell_dir = out_root / behavior / f"{source_cid}_seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cell_dir / f"{behavior}_{source_cid}_seed{seed}_pertoken_context.npz"
    np.savez(
        npz_path,
        # from-start alignment (index 0 = first prompt token)
        start_mag_sum=start_mag_sum,
        start_dir_sum=start_dir_sum,
        start_count=start_count,
        # from-end alignment (offset 0 = last input token)
        end_mag_sum=end_mag_sum,
        end_dir_sum=end_dir_sum,
        end_count=end_count,
        # self-check payload (from-end offset-0 vs summary Δc)
        selfcheck_max_mag_diff=selfcheck["max_mag_diff"],
        selfcheck_max_cos_diff=selfcheck["max_cos_diff"],
        selfcheck_passed=selfcheck["passed"],
        selfcheck_offset0_mag=selfcheck["offset0_mag"],
        selfcheck_offset0_cos=selfcheck["offset0_cos"],
        selfcheck_summary_mag=selfcheck["summary_mag"],
        selfcheck_summary_cos=selfcheck["summary_cos"],
        # metadata
        behavior=behavior,
        source_cid=source_cid,
        seed=seed,
        n_layers=n_layers,
        max_ctx_pos=max_ctx_pos,
        max_ctx_offset=max_ctx_offset,
        n_targets=len(targets),
        n_probes=len(probes),
        n_gen=n_gen,
        n_empty=n_empty,
    )
    assert npz_path.is_file(), npz_path
    write_cell_done_sentinel(
        cell_dir,
        {
            "behavior": behavior,
            "source_cid": source_cid,
            "seed": seed,
            "npz": npz_path.name,
            "n_layers": n_layers,
            "max_ctx_pos": max_ctx_pos,
            "max_ctx_offset": max_ctx_offset,
            "selfcheck_passed": bool(selfcheck["passed"]),
            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )
    logger.info("wrote %s (+ .done sentinel)", npz_path)

    if device.type == "cuda":
        del base, trained
        torch.cuda.empty_cache()
    return 0


@torch.no_grad()
def _run_selfcheck(
    base,
    trained,
    tok,
    registry,
    demos,
    behavior,
    source_cid,
    probes,
    device,
    args,
    n_layers,
    r_lookup,
) -> dict:
    """from-end offset-0 vs SUMMARY Δc last-input-token, over the source diagonal.

    Computes, per source-diagonal probe: (a) the streamed from-end OFFSET-0 read
    (last input token of the teacher-forced ``T_source(q)+R`` pass), and (b) the
    SUMMARY Δc read (last input token of the prompt-only ``T_source(q)`` pass, the
    :func:`issue667_extract._context_vector_all_layers` recipe). Both reduce the
    base θ0 vs trained θ⁺ pair with the SAME (mag, cos) metric. The two should
    agree per layer: the last token of a teacher-forced prompt+response IS the
    generation-prompt's last token (the assistant turn is appended AFTER it), so
    its residual is identical. Returns per-layer means + the max discrepancy +
    a boolean pass flag.

    Reuses the Phase-A ``r_lookup`` for the source-diagonal R (``source_cid`` is
    always in ``targets``, so its R is already generated) — this function NEVER
    constructs a vLLM engine: a fresh ``LLM()`` after the HF base+PeftModel are
    resident OOMs on ``gpu_memory_utilization`` (the crash this fix closes). On
    the CPU-smoke path (empty ``r_lookup``) it falls back to HF greedy per probe.

    NOTE: the offset-0 read here re-derives the last-input residual of the
    ``prompt+R`` pass (matching the streamed accumulator's alignment) so the
    comparison is apples-to-apples; the summary read is the ``prompt``-only pass.
    Agreement within SELFCHECK_TOL confirms the from-end alignment is wired to
    the right token (offset 0 == c_C), the brief's explicit ask.
    """
    off0_mag_acc = np.zeros(n_layers, dtype=np.float64)
    off0_cos_acc = np.zeros(n_layers, dtype=np.float64)
    summ_mag_acc = np.zeros(n_layers, dtype=np.float64)
    summ_cos_acc = np.zeros(n_layers, dtype=np.float64)
    n = 0
    for qi, q in enumerate(probes):
        tmsgs = ix.build_messages_for(registry, demos, source_cid, behavior, q)
        r = r_lookup.get((source_cid, qi))
        if r is None:
            r = ix._greedy_response(base, tok, tmsgs, device, args.max_new_tokens)
        if not r.strip():
            continue
        # (a) from-end offset-0 == last input token of the prompt+R teacher-force.
        out = _per_context_token_shift_scalars(
            base, trained, tok, tmsgs, r, device, max_ctx_pos=1, max_ctx_offset=1
        )
        if out["n_end"] < 1:
            continue
        off0_mag_acc += out["end_mag"][0, :]
        off0_cos_acc += out["end_cos"][0, :]
        # (b) SUMMARY Δc: last input token of the prompt-only pass.
        s_mag, s_cos = _summary_delta_c_last_token(base, trained, tok, tmsgs, device)
        summ_mag_acc += s_mag
        summ_cos_acc += s_cos
        n += 1
        del out
        if device.type == "cuda":
            torch.cuda.empty_cache()
    n = max(n, 1)
    off0_mag = off0_mag_acc / n
    off0_cos = off0_cos_acc / n
    summ_mag = summ_mag_acc / n
    summ_cos = summ_cos_acc / n
    max_mag_diff = float(np.max(np.abs(off0_mag - summ_mag))) if n else float("nan")
    max_cos_diff = float(np.max(np.abs(off0_cos - summ_cos))) if n else float("nan")
    passed = bool(max_mag_diff <= SELFCHECK_TOL and max_cos_diff <= SELFCHECK_TOL)
    return {
        "max_mag_diff": max_mag_diff,
        "max_cos_diff": max_cos_diff,
        "passed": passed,
        "offset0_mag": off0_mag.astype(np.float64),
        "offset0_cos": off0_cos.astype(np.float64),
        "summary_mag": summ_mag.astype(np.float64),
        "summary_cos": summ_cos.astype(np.float64),
        "n": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #667 per-context-token activation-shift extractor (one cell).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--behavior", required=True, choices=["em", "sycophancy", "fact", "marker"])
    parser.add_argument("--source-cid", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--targets", default=None, help="comma-separated target cids (default: 30 eval + source)"
    )
    parser.add_argument(
        "--max-ctx-pos",
        type=int,
        default=MAX_CTX_POS_DEFAULT,
        help="cap the from-start context position axis (default 128; brief §1).",
    )
    parser.add_argument(
        "--max-ctx-offset",
        type=int,
        default=MAX_CTX_OFFSET_DEFAULT,
        help="cap the from-end context offset axis (default 64; offset 0 = last input token).",
    )
    parser.add_argument("--out", default="eval_results/issue_667_pertoken_context/analysis_tensors")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--max-probes", type=int, default=0, help="cap probes (0 = full pool; smoke)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=ix.N_GEN_TOKENS)
    parser.add_argument(
        "--vllm-gpu-mem-util",
        type=float,
        default=0.45,
        help="vLLM gpu_memory_utilization for Phase-A base-R generation (default 0.45; "
        "lowered from vllm_generate_R's 0.85 so the wave-to-wave engine handoff has HBM "
        "headroom against async vLLM teardown + the pod's zombie context. Greedy gen of "
        "~240 short prompts needs little KV cache, so 0.45 (~36 GB) is ample and made the "
        "intermittent single-cell EngineCore-init OOM (1/31 cells at 0.6) essentially "
        "vanish; 0.6 still left one wave-2 cell losing the race).",
    )
    parser.add_argument(
        "--skip-adapter-gauge",
        action="store_true",
        help="CPU-smoke only: build a tiny stub base+LoRA instead of the #537 adapter.",
    )
    args = parser.parse_args()
    if args.max_probes == 0:
        args.max_probes = None
    t0 = time.time()
    rc = run_extraction(args)
    logger.info("context extraction wall=%.1fs", time.time() - t0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
