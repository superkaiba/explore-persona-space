#!/usr/bin/env python3
"""Issue #667 per-ANSWER-TOKEN activation-shift extractor (one source-adapter cell).

Extends the #667 SUMMARY-context-vector shift analysis (Δc) to the FULL ANSWER
TRAJECTORY. For ONE (behavior, source-context C) cell this CLI:

1. Stages + loads the #537 adapter as a ``PeftModel`` on the base Qwen-2.5-7B
   (rsLoRA honored; asserts ``base_model_name_or_path == Qwen/Qwen2.5-7B-Instruct``
   — fitness check (f)/(g)), REUSING :mod:`issue667_extract`'s exact model-load /
   adapter-apply / adapter-gauge helpers.
2. For each eval target context C' (the 30 #537 eval cids + the source diagonal)
   and each eval probe: generates the FROZEN base greedy response ``R`` (batched
   vLLM, temp=0 — the same generator :mod:`issue667_extract` uses), then
   teacher-forces ``T_{C'}(q) + R`` through BOTH base θ0 and adapter-applied θ⁺ in
   ONE forward pass each, capturing per-answer-token residual-stream hidden states
   at ALL layers 0-27 via the memory-safe hook path (``output_hidden_states=False``;
   the unused layers are freed as the forward proceeds — the #671 fix).
3. Per answer-token-position ``t`` (0-indexed within the answer span, capped at the
   first ``--max-token-pos`` = 128) and layer ``L``, reduces the pair of hidden
   states to TWO scalars vs base, ON THE FLY (never retaining the full
   answer-span x 28 x 3584 tensor beyond the current probe):

     - magnitude change: ``||h+_{t,L} - h_{t,L}|| / ||h_{t,L}||``
     - direction change: ``cos(h_{t,L}, h+_{t,L})``

   These two scalars per (t, L) are accumulated into a STREAMING per-behavior MEAN
   (running sum + count arrays of shape [max_token_pos, N_LAYERS]) across ALL
   probes x targets for the cell. MEAN, not median — a streaming median over the
   full (probe x target) population is infeasible without retaining every sample,
   and the mean is the streaming-feasible aggregate (stated in the brief).
4. Writes ONE tiny ``.npz`` per (behavior, source-C) cell holding the per-cell
   partial sums:

     - ``mag_sum[max_token_pos, N_LAYERS]``  (sum of the magnitude ratio)
     - ``dir_sum[max_token_pos, N_LAYERS]``  (sum of the cosine)
     - ``count[max_token_pos, N_LAYERS]``    (per-(t,L) sample count; coverage)

   The per-behavior heatmaps are the count-weighted mean over the cell npzs —
   ``mag_mean = sum_cells mag_sum / sum_cells count`` — computed by the plotter
   (:mod:`issue667_pertoken_figures`). Sharding by cell keeps this OOM-safe: each
   subprocess holds only its own [128, 28] float64 accumulators (~57 KB) plus, at
   any instant, ONE probe's two (span, 28, 3584) hidden-state tensors (~tens of MB
   at bf16 / span<=~200), which are freed before the next probe.

CONTENT HYGIENE: ``em`` training/eval probes are Betley harmful-content adjacent —
this script NEVER prints/logs probe text or response text; it digests by row/token
COUNT + the reduced scalar ACTIVATIONS only. Benign behaviors (marker/fact/
sycophancy) are unaffected. No raw completions are written to disk by this script.

Usage (one source-adapter cell)::

    uv run python scripts/issue667_pertoken_extract.py \\
        --behavior em --source-cid default \\
        --targets sp_swe,default,fmt_json --max-token-pos 128 \\
        --out eval_results/issue_667_pertoken/analysis_tensors --gpu-id 0

Smoke (CPU, tiny stub model, capped probes/targets)::

    uv run python scripts/issue667_pertoken_extract.py \\
        --behavior em --source-cid default --targets default \\
        --cpu-only --max-probes 2 --max-token-pos 8 --skip-adapter-gauge \\
        --out /tmp/i667pt_smoke
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
# vLLM EngineCore fork() poisoning guard (mirrors issue667_extract module-top):
# main() touches transformers.AutoTokenizer BEFORE vllm.LLM() constructs, and any
# pre-LLM() transformers/tokenizer touch poisons the EngineCore fork; spawn (not
# fork) avoids the silent worker death. Must be set BEFORE any `import vllm`.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# scripts/ on the path so the cross-script `import issue667_extract` resolves
# cwd-independently (same idiom as issue667_alllayer_dispatch).
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse issue667_extract's model-load / adapter / teacher-force / probe helpers
# verbatim (single source of truth for token-position + layer-index conventions).
import issue667_extract as ix  # noqa: E402
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

logger = logging.getLogger("issue667_pertoken_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MAX_TOKEN_POS_DEFAULT = 128  # cap the answer-span position axis (brief §2)
# Atomic completion sentinel — mirrors issue667_extract.CELL_DONE_SENTINEL so a
# mid-cell crash leaves a PARTIAL npz with NO .done and the dispatcher resume-skip
# re-extracts (never trusts a partial cell).
CELL_DONE_SENTINEL = ".done"


# ─────────────────────────────────────────────────────────────────────────────
# Per-token span alignment (reuses issue667_extract's prompt-prefix logic)
# ─────────────────────────────────────────────────────────────────────────────


def _answer_span_ids(tok, messages: list[dict], response: str) -> tuple[list[int], int, int]:
    """Return (full_ids, prompt_len, span_end) for ``messages + response``.

    Token conventions are BYTE-IDENTICAL to :func:`issue667_extract._mean_resp_acts`
    (the SUMMARY-shift read this per-token script extends): the answer span is
    ``full_ids[prompt_len:span_end]``, where ``prompt_len`` is the prompt token
    count under ``add_generation_prompt=True`` and ``span_end == len(full_ids)``.
    The same chat-template-drift LCP fallback is applied so a per-token read never
    slices a mis-aligned span. Raises on an empty span (zero response tokens),
    matching ``_mean_resp_acts``.
    """
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_msgs = [*messages, {"role": "assistant", "content": response}]
    full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    p = len(prompt_ids)
    if full_ids[:p] != prompt_ids:
        lcp = 0
        for a, b in zip(prompt_ids, full_ids, strict=False):
            if a != b:
                break
            lcp += 1
        if lcp < max(1, p - 4):
            raise RuntimeError(
                f"prompt-prefix drift: lcp={lcp} vs prompt_len={p} — chat-template mismatch"
            )
        p = lcp
    span_end = len(full_ids)
    if span_end <= p:
        raise RuntimeError("empty response span — response produced zero tokens")
    return full_ids, p, span_end


@torch.no_grad()
def _per_token_shift_scalars(
    base_model,
    trained_model,
    tok,
    messages: list[dict],
    response: str,
    device,
    *,
    max_token_pos: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-(answer-token-position, layer) magnitude + direction shift vs base.

    Teacher-forces ``messages + response`` through base θ0 AND adapter θ⁺ ONCE
    each (memory-safe hook path over ALL 28 block layers), then reduces the pair
    of per-token residuals to two scalars per (t, L):

        magnitude: ||h+_{t,L} - h_{t,L}|| / ||h_{t,L}||   (relative L2 shift)
        direction: cos(h_{t,L}, h+_{t,L})                 (in [-1, 1])

    OOM-safety: the two (span, 28, 3584) hidden-state tensors exist only for the
    duration of this call; the returned arrays are the tiny reduced scalars
    ([max_token_pos, N_LAYERS] each) so the caller never accumulates full tensors.

    Returns ``(mag[max_token_pos, N_LAYERS], dir[max_token_pos, N_LAYERS], n_pos)``
    where ``n_pos = min(span_len, max_token_pos)`` is the number of VALID rows
    (rows >= n_pos are 0.0 and must NOT be counted by the caller). ``dir`` is
    clamped to [-1, 1] for float noise past the acos-safe range.
    """
    full_ids, p, span_end = _answer_span_ids(tok, messages, response)
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    n_layers = getattr(base_model.config, "num_hidden_layers", N_LAYERS)
    layers = list(range(n_layers))
    # Memory-safe subset read: hook every block layer li == old hs[li+1] (the SAME
    # tensor issue667_extract._mean_resp_acts read at hs[layer+1]); the unused
    # layers are freed as the forward proceeds (never a full-seq x (L+1) retain).
    acts_b = extract_layer_activations(base_model, ids, layers)
    acts_t = extract_layer_activations(trained_model, ids, layers)

    span_len = span_end - p
    n_pos = min(span_len, max_token_pos)
    mag = np.zeros((max_token_pos, n_layers), dtype=np.float64)
    cos = np.zeros((max_token_pos, n_layers), dtype=np.float64)
    eps = 1e-8
    for li in layers:
        # (span_len, H) float32 residuals over the answer span, base + trained.
        hb = acts_b[li][0, p:span_end, :].float()
        ht = acts_t[li][0, p:span_end, :].float()
        hb = hb[:n_pos]
        ht = ht[:n_pos]
        base_norm = hb.norm(dim=-1)  # (n_pos,)
        shift_norm = (ht - hb).norm(dim=-1)  # (n_pos,)
        mag_col = (shift_norm / (base_norm + eps)).cpu().numpy()
        dot = (hb * ht).sum(dim=-1)
        cos_col = (dot / (base_norm * ht.norm(dim=-1) + eps)).clamp(-1.0, 1.0).cpu().numpy()
        mag[:n_pos, li] = mag_col.astype(np.float64)
        cos[:n_pos, li] = cos_col.astype(np.float64)
    return mag, cos, n_pos


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell driver (streaming per-(t, L) accumulator)
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_targets(behavior: str, source_cid: str, targets_arg: str | None) -> list[str]:
    """The target contexts for a cell: --targets subset, else 30 eval cids + source."""
    from explore_persona_space.experiments.i537_contexts import eval_cids_for

    if targets_arg:
        targets = [t.strip() for t in targets_arg.split(",") if t.strip()]
    else:
        targets = list(dict.fromkeys([*eval_cids_for(behavior), source_cid]))
    if source_cid not in targets:
        targets = [source_cid, *targets]
    return targets


def write_cell_done_sentinel(cell_dir: Path, payload: dict) -> Path:
    """Atomically stamp the cell's .done sentinel AFTER the npz is on disk.

    Atomic = write-temp-then-os.replace within the same dir, so a crash mid-write
    never leaves a half-written .done that the dispatcher resume-skip would trust
    (mirrors issue667_extract.write_cell_done_sentinel).
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
    max_token_pos = int(args.max_token_pos)
    assert max_token_pos > 0, max_token_pos

    sampled_path, demos_path = ix.stage_inputs()
    registry = load_registry(sampled_path)
    demos = load_icl_demos(demos_path)

    behavior = args.behavior
    source_cid = args.source_cid
    seed = args.seed
    targets = _resolve_targets(behavior, source_cid, args.targets)
    probes = ix.load_eval_probes(behavior)
    if args.max_probes:
        probes = probes[: args.max_probes]
    logger.info(
        "pertoken cell behavior=%s source=%s seed=%d | %d targets x %d probes | max_pos=%d",
        behavior,
        source_cid,
        seed,
        len(targets),
        len(probes),
        max_token_pos,
    )

    # Stage + verify the adapter gauge BEFORE any GPU work (cheap, HALT early).
    # --skip-adapter-gauge is the CPU-smoke escape (no #537 adapter on the VM):
    # a tiny stub PeftModel is built instead so the reduce/accumulator path runs.
    if args.skip_adapter_gauge:
        tok, base, trained, n_layers = _build_cpu_stub_models()
    else:
        adapter_dir = ix.stage_adapter_local(behavior, source_cid, seed)
        gauge = ix.assert_adapter_gauge(adapter_dir, behavior)
        logger.info(
            "adapter gauge OK: %s", {k: gauge[k] for k in ("r", "lora_alpha", "use_rslora")}
        )
        # ── Phase A: vLLM batched base R (per CLAUDE.md; HF fallback on CPU) ──
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
        r_lookup = _generate_base_R(tok, registry, demos, behavior, targets, probes, device, args)
        # ── Phase B: load base θ0 + trained θ⁺ for the teacher-force reads ──
        _, base, trained = ix.load_base_and_trained(adapter_dir, device, dtype)
        assert base.config.hidden_size == HIDDEN_SIZE or device.type == "cpu", (
            base.config.hidden_size
        )
        n_layers = getattr(base.config, "num_hidden_layers", N_LAYERS)

    # On the stub path, R is generated per-probe with HF greedy (no vLLM).
    r_lookup = locals().get("r_lookup", {})

    mag_sum = np.zeros((max_token_pos, n_layers), dtype=np.float64)
    dir_sum = np.zeros((max_token_pos, n_layers), dtype=np.float64)
    count = np.zeros((max_token_pos, n_layers), dtype=np.int64)

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
            mag, cos, n_pos = _per_token_shift_scalars(
                base, trained, tok, tmsgs, r, device, max_token_pos=max_token_pos
            )
            # Only the VALID rows [0:n_pos] carry data; accumulate exactly those.
            mag_sum[:n_pos, :] += mag[:n_pos, :]
            dir_sum[:n_pos, :] += cos[:n_pos, :]
            count[:n_pos, :] += 1
            # Free the per-probe activations promptly (OOM-safety; #671/#545 trap).
            del mag, cos
            if device.type == "cuda":
                torch.cuda.empty_cache()

    covered = int((count > 0).sum())
    logger.info(
        "cell %s/%s done: %d gens (%d empty); %d/%d (t,L) cells covered; max count=%d",
        behavior,
        source_cid,
        n_gen,
        n_empty,
        covered,
        max_token_pos * n_layers,
        int(count.max()) if count.size else 0,
    )

    out_root = Path(args.out)
    cell_dir = out_root / behavior / f"{source_cid}_seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cell_dir / f"{behavior}_{source_cid}_seed{seed}_pertoken.npz"
    np.savez(
        npz_path,
        mag_sum=mag_sum,
        dir_sum=dir_sum,
        count=count,
        behavior=behavior,
        source_cid=source_cid,
        seed=seed,
        n_layers=n_layers,
        max_token_pos=max_token_pos,
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
            "max_token_pos": max_token_pos,
            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )
    logger.info("wrote %s (+ .done sentinel)", npz_path)

    if device.type == "cuda":
        del base, trained
        torch.cuda.empty_cache()
    return 0


def _generate_base_R(tok, registry, demos, behavior, targets, probes, device, args) -> dict:
    """Batched vLLM greedy base R for every (target, probe) (GPU path only).

    Returns ``{(tcid, probe_index): response}``. On CPU the caller falls back to
    per-probe HF greedy (vLLM unavailable), so this returns {} for device==cpu.
    Mirrors :func:`issue667_extract.run_extraction`'s Phase A exactly (same
    generator, same order) so the frozen base R is identical to the SUMMARY
    read's R.
    """
    if device.type == "cpu":
        return {}
    gen_msgs: list[list[dict]] = []
    gen_keys: list[tuple[str, int]] = []
    for tcid in targets:
        for qi, q in enumerate(probes):
            gen_msgs.append(ix.build_messages_for(registry, demos, tcid, behavior, q))
            gen_keys.append((tcid, qi))
    logger.info("Phase A: vLLM-generating %d base R responses", len(gen_msgs))
    responses = ix.vllm_generate_R(tok, gen_msgs, max_new_tokens=args.max_new_tokens)
    return dict(zip(gen_keys, responses, strict=True))


# ─────────────────────────────────────────────────────────────────────────────
# CPU smoke stub (tiny 2-layer causal LM + real tokenizer) — no #537 adapter
# ─────────────────────────────────────────────────────────────────────────────


def _build_cpu_stub_models():
    """Tiny 2-layer Qwen2 causal LM + a trivial LoRA PeftModel for the CPU smoke.

    Exercises the FULL reduce/accumulate/write path (per-token span alignment,
    the hook-based all-layer read fallback for a non-standard tiny model, the
    magnitude/direction reduce, the streaming accumulator, the npz + sentinel
    write) without any GPU or the 15 GB base model. The tokenizer is the REAL
    Qwen tokenizer so chat-template + span alignment are exercised faithfully.
    Returns (tokenizer, base, trained, n_layers).
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    cfg = AutoConfig.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    # Shrink to a 2-layer, small-hidden model so the smoke is fast + tiny.
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    base = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    trained_base = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    trained_base.load_state_dict(base.state_dict())  # identical weights -> shift is the LoRA
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], use_rslora=True)
    trained = get_peft_model(trained_base, lora).eval()
    # Perturb the LoRA B matrices so θ⁺ != θ0 (else magnitude/cos are trivially 0/1).
    with torch.no_grad():
        for name, param in trained.named_parameters():
            if "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.05)
    n_layers = cfg.num_hidden_layers
    logger.info("CPU stub models built: %d layers, hidden=%d", n_layers, cfg.hidden_size)
    return tok, base, trained, n_layers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #667 per-answer-token activation-shift extractor (one cell).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--behavior", required=True, choices=["em", "sycophancy", "fact", "marker"])
    parser.add_argument("--source-cid", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--targets", default=None, help="comma-separated target cids (default: 30 eval + source)"
    )
    parser.add_argument(
        "--max-token-pos",
        type=int,
        default=MAX_TOKEN_POS_DEFAULT,
        help="cap the answer-token position axis (default 128; brief §2).",
    )
    parser.add_argument("--out", default="eval_results/issue_667_pertoken/analysis_tensors")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--max-probes", type=int, default=0, help="cap probes (0 = full pool; smoke)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=ix.N_GEN_TOKENS)
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
    logger.info("extraction wall=%.1fs", time.time() - t0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
