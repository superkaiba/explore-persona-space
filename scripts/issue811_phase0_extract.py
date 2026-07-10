#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (θ0, v0, →, ×) in scientific docstrings + logs.
"""Issue #811 Phase 0 — BASE-LEG-ONLY re-extraction for the KILL-1 pre-spend gate.

The plan (v1 §4.0 / §4.1 / §7) makes KILL-1 a GENUINE PRE-SPEND gate: BEFORE
committing the full ~7 GPU-h paired Phase-1 re-extraction on BOTH legs, verify
``turn_nl`` is a valid answer summary on the BASE leg (M0: c0 → v0) over #537's
16 source contexts. This script does ONLY the base leg — it loads base θ0 (NO
#537 adapter, NO PeftModel), generates the frozen greedy R from the base model,
and teacher-forces R through θ0 ONLY, reading BOTH the ``mean``-over-response and
``turn_nl`` (turn-boundary single-position) base summaries at PRIMARY LAYER 14.
Cost is ~1 GPU-h (16 sources × 30 targets × 3 behaviors, layer 14 only, ONE model,
no adapter apply) vs Phase 1's ~7 GPU-h.

Why a SEPARATE script and not a ``--base-only`` flag on ``issue667_extract.py``:
that extractor loads base+trained upfront (``load_base_and_trained``), reads
``v0``/``v_plus`` from the SAME forward, and its source-level reads
(``c_C_postft`` / ``t+``/``t-`` / ``v0(C_neg)``) mix base+trained — a
``--base-only`` fork would touch its whole write / complement / sentinel path.
This script reuses the extractor's PURE helpers (``_locate_turn_close_newline``,
``vllm_generate_R``, ``extract_layer_activations``, ``build_messages_for``,
``load_eval_probes``, ``stage_inputs``, ``_device``) so the base-leg read is
BYTE-IDENTICAL to Phase 1's base leg — only the adapter apply + the trained-leg
reads are dropped.

Store shape (base-leg only): per cell one ``.npz`` at
``phase0_base_leg/{behavior}/{source}_seed42/{target}_L14.npz`` with keys
``c_C`` (base context vector, the gate's M0 input), ``v0`` (base mean answer),
``v0_turn_nl`` (base turn_nl answer). NO ``v_plus`` / ``c_C_postft`` — the
KILL-1 base-leg validity gate reads only the base leg (``C0`` = ``c_C``, ``V0``
= ``v0`` / ``v0_turn_nl``); ``issue811_fit.py --phase0-gate`` consumes exactly
these three keys.

Usage (one cell per invocation, CVD-pinned in the launcher env like Phase 1):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue811_phase0_extract.py \
        --behavior em --source-cid default --primary-layer 14 \
        --out eval_results/issue_811/phase0_base_leg --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# uv run python does NOT auto-load .env; the base-model + tokenizer loads below
# need HF_TOKEN. The extractor module also load_dotenv()s at import; this is the
# explicit belt-and-suspenders at this script's own entry (analysis-phase script;
# shell exports additionally cover pod/GCE/SLURM).

import issue667_extract as ex  # noqa: E402  (reuse the extractor's pure helpers)

logger = logging.getLogger("issue811.phase0")

# Base-leg store prefix — the KILL-1 gate reads this; distinct from Phase 1's
# paired store (issue811_turn_nl_mapchange/analysis_tensors).
PHASE0_STORE_SUBDIR = "phase0_base_leg"


def _context_vector_base(base_model, tok, messages: list[dict], device) -> np.ndarray:
    """Base-side c_C at ALL layers (reuse the extractor's exact recipe).

    Returns (N_LAYERS, HIDDEN) float32 — the whitened-gate key/query at the
    last-input-token, read with the SAME reader Phase 1 uses for its base c_C so
    the gate's M0 input (c0) is byte-identical to Phase 1's base leg.
    """
    return ex._context_vector_all_layers(base_model, tok, messages, device)


@torch.no_grad()
def _base_answer_summaries(
    base_model,
    tok,
    messages: list[dict],
    response: str,
    layers: list[int],
    device,
    *,
    pre_user: bool = False,
) -> dict[int, dict[str, np.ndarray]]:
    """Teacher-force ``messages + response`` through BASE θ0 only; base summaries.

    Mirrors ``issue667_extract._mean_resp_acts`` but for the BASE leg ONLY (no
    trained model): reads the mean-over-response-span residual, the turn_nl
    (turn-close newline, ``full_ids[-1]``) single-position residual, AND (#811
    maxp round) #810's crowned ``maxp`` — the per-dimension element-wise max over
    the response CONTENT tokens ``[p : content_end)`` (EXCLUDING the turn-close
    ``<|im_end|>``+newline #810 refuted as summaries) — all from the SAME base
    forward pass. Returns ``{layer: {"mean": ..., "turn_nl": ..., "maxp": ...}}``
    (base vectors only — the gate needs no ``v_plus``). Fails loud (KILL-2 code)
    if the turn-close newline / maxp span asserts break; identical asserts to
    Phase 1's reader.
    """
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_msgs = [*messages, {"role": "assistant", "content": response}]
    full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    p = len(prompt_ids)
    if full_ids[:p] != prompt_ids:
        # Same longest-common-prefix fallback as the Phase-1 reader (chat-template
        # drift between the generation prompt and the full row); fail loud if tiny.
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
    # KILL-2 (code): locate the turn-close newline BEFORE any GPU reduce — the
    # assert failing HALTs the cell (same signal Phase 1 raises).
    turn_nl_idx = ex._locate_turn_close_newline(full_ids, tok)
    # #811 maxp round: content span = [p : content_end), content tokens ONLY —
    # #810's crowned recipe (issue658_common.summarize_answer_span, "maxp"; the
    # #658 span never included the turn-close <|im_end|>+"\n", which #810 swept
    # as SEPARATE summaries and refuted). Same KILL-2 asserts as Phase 1's reader.
    content_end = turn_nl_idx - 1
    if full_ids[content_end] != ex.IM_END_ID:
        raise RuntimeError(
            f"[maxp-assert] token at content_end={content_end} is id="
            f"{full_ids[content_end]}, expected <|im_end|> (id {ex.IM_END_ID}) — "
            "the maxp content-span scoping broke (KILL-2, failure_class: code)"
        )
    if content_end <= p:
        raise RuntimeError(
            f"[maxp-assert] empty maxp content span: content_end={content_end} "
            f"<= prompt_len={p} (KILL-2, failure_class: code)"
        )
    # #811 pre-user round (plan §4.1): all span indices above are PRE-append; the
    # forward runs over ext_ids = full_ids + HEADER_IDS with ALL blocks hooked
    # (the arm-8/9 gate needs the base alllayer stacks). Causal attention keeps
    # positions < F invariant to the append (A2), so mean/turn_nl/maxp are
    # byte-equivalent to the unextended read.
    if pre_user:
        ext_ids, F = ex._extended_ids(full_ids)  # KILL-2 tail assert inside
        ids = torch.tensor([ext_ids], dtype=torch.long, device=device)
        n_blocks = int(base_model.config.num_hidden_layers)
        hook_layers = sorted(set(layers) | set(range(n_blocks)))
    else:
        F = span_end
        n_blocks = None
        hook_layers = layers
        ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    acts_b = ex.extract_layer_activations(base_model, ids, hook_layers)
    res: dict[int, dict[str, np.ndarray]] = {}
    for li in layers:
        hb_mean = acts_b[li][0, p:span_end, :].float().mean(dim=0).cpu().numpy().astype(np.float32)
        hb_nl = acts_b[li][0, turn_nl_idx, :].float().cpu().numpy().astype(np.float32)
        # Crowned reduction REUSED from issue658_common via the Phase-1 module
        # (recipe="maxp" == span.max(dim=0).values); finiteness is a KILL-2 assert
        # (bf16→fp32 element-wise max is exact — non-finite = upstream bug).
        hb_mx = ex.summarize_answer_span(acts_b[li][0, p:content_end, :].float(), "maxp")
        if not bool(torch.isfinite(hb_mx).all()):
            raise RuntimeError(
                f"[maxp-assert] non-finite base maxp summary at layer {li} "
                "(KILL-2, failure_class: code)"
            )
        res[li] = {
            "mean": hb_mean,
            "turn_nl": hb_nl,
            "maxp": hb_mx.cpu().numpy().astype(np.float32),
        }
        if pre_user:
            for arm in ex.PRE_USER_LAYER_ARMS:
                hb_a = ex._pre_user_layer_arm(acts_b, li, p, F, arm)
                ex._assert_finite_arm(hb_a, arm, li)
                res[li][arm] = hb_a.cpu().numpy().astype(np.float32)
    if pre_user:
        # Base-leg (n_blocks, H) arm-6/7 stacks per probe (probe-meaned + fp16-cast
        # at persist time; the arm-8/9 KILL-1 gate reads keys derived from them).
        stacks: dict[str, np.ndarray] = {}
        for base_name in ex.PRE_USER_STACK_BASES:
            rows = [ex._pre_user_layer_arm(acts_b, li, p, F, base_name) for li in range(n_blocks)]
            stack = torch.stack(rows)
            ex._assert_finite_arm(stack, f"{base_name}_stack", "all")
            stacks[base_name] = stack.cpu().numpy().astype(np.float32)
        res["stacks"] = stacks  # type: ignore[assignment]
    return res


def _extract_base_target(
    base,
    tok,
    registry,
    demos,
    cell_dir,
    behavior,
    tcid,
    probes,
    layers,
    primary_layer,
    device,
    r_lookup,
    *,
    pre_user: bool = False,
) -> tuple[int, int]:
    """Base-leg reads for ONE target C' across ``layers``; write one .npz per layer.

    Accumulates per-probe base summaries (mean + turn_nl + maxp; under
    ``pre_user`` also the seven per-layer boundary arms + the two arm-6/7
    all-layer stacks), means over the probe pool exactly like Phase 1's
    accumulator, and writes ``{tcid}_L{li}.npz`` with ``c_C`` / ``v0`` /
    ``v0_turn_nl`` (+ the ``v0_<arm>`` keys and the fp16 ``*_stack`` keys under
    ``pre_user`` — the arm-8/9 KILL-1 gate inputs, plan §4.3 item 2). Returns
    (n_generations, n_empty).
    """
    tmsgs0 = ex.build_messages_for(registry, demos, tcid, behavior, probes[0])
    c_c_all = _context_vector_base(base, tok, tmsgs0, device)  # (N_LAYERS, HIDDEN)
    arm_names = list(ex.PRE_USER_LAYER_ARMS) if pre_user else []
    acc: dict[int, dict[str, list[np.ndarray]]] = {
        li: {s: [] for s in ("mean", "turn_nl", "maxp", *arm_names)} for li in layers
    }
    acc_stacks: dict[str, list[np.ndarray]] = {b: [] for b in ex.PRE_USER_STACK_BASES}
    n_gen = n_trunc = 0
    for qi, q in enumerate(probes):
        tmsgs = ex.build_messages_for(registry, demos, tcid, behavior, q)
        r = r_lookup.get((tcid, qi))
        if r is None:
            r = ex._greedy_response(base, tok, tmsgs, device, ex.N_GEN_TOKENS)
            # Record the CPU-fallback R so the post-loop persist captures it too.
            r_lookup[(tcid, qi)] = r
        n_gen += 1
        if not r.strip():
            n_trunc += 1
            continue
        per_layer = _base_answer_summaries(base, tok, tmsgs, r, layers, device, pre_user=pre_user)
        for base_name, stack in per_layer.pop("stacks", {}).items():  # type: ignore[union-attr]
            acc_stacks[base_name].append(stack)
        for li in layers:
            for s in ("mean", "turn_nl", "maxp", *arm_names):
                acc[li][s].append(per_layer[li][s])
    # Probe-meaned fp16 base stacks + derived arms 8/9 (bit-re-derivable from the
    # persisted fp16 keys — the SAME recipe as the paired extractor, plan §4.2).
    stack16: dict[str, np.ndarray] = {}
    derived: dict[str, np.ndarray] = {}
    if pre_user and acc[layers[0]]["mean"]:
        for base_name in ex.PRE_USER_STACK_BASES:
            stack16[base_name] = np.stack(acc_stacks[base_name]).mean(axis=0).astype(np.float16)
        d0 = ex.derive_alllayer_arms(stack16["ans_mean_incl_hdr"], stack16["ans_max_incl_hdr"])
        derived = {f"v0_{arm}": vec for arm, vec in d0.items()}
    for li in layers:
        if not acc[li]["mean"]:
            continue  # empty-response target for this layer — skip its .npz (loud via count)
        # c_C at all layers is (N_LAYERS, HIDDEN); block index li -> row li-1 (hs[li+1]
        # convention: the extractor's _context_vector_all_layers drops hs[0]).
        c_idx = (li - 1) if 1 <= li <= ex.N_LAYERS else (primary_layer - 1)
        payload = {
            "c_C": c_c_all[c_idx],
            "v0": np.stack(acc[li]["mean"]).mean(axis=0).astype(np.float32),
            "v0_turn_nl": np.stack(acc[li]["turn_nl"]).mean(axis=0).astype(np.float32),
            # #811 maxp round: per-probe content-token element-wise max, probe-mean
            # (the SAME accumulator shape as mean/turn_nl — #658's recipe_accum).
            "v0_maxp": np.stack(acc[li]["maxp"]).mean(axis=0).astype(np.float32),
            "behavior": np.asarray(behavior),
            "source_cid": np.asarray(cell_dir.name.rsplit("_seed", 1)[0]),
            "target_cid": np.asarray(tcid),
            "layer": np.asarray(li),
        }
        if pre_user:
            for arm in arm_names:
                payload[f"v0_{arm}"] = np.stack(acc[li][arm]).mean(axis=0).astype(np.float32)
            payload.update(derived)  # v0_ans_{mean,max}_incl_hdr_alllayer (float32)
            payload["v0_ans_mean_incl_hdr_stack"] = stack16["ans_mean_incl_hdr"]
            payload["v0_ans_max_incl_hdr_stack"] = stack16["ans_max_incl_hdr"]
        np.savez(cell_dir / f"{tcid}_L{li}.npz", **payload)
    return n_gen, n_trunc


def run(args) -> int:
    from explore_persona_space.experiments.i537_contexts import (
        eval_cids_for,
        load_icl_demos,
        load_registry,
    )

    device = ex._device(args.gpu_id, args.cpu_only)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    layers = list(args.layers)
    assert args.primary_layer in layers, (args.primary_layer, layers)

    sampled_path, demos_path = ex.stage_inputs()
    registry = load_registry(sampled_path)
    demos = load_icl_demos(demos_path)

    behavior = args.behavior
    source_cid = args.source_cid
    seed = args.seed

    if args.targets:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        targets = list(dict.fromkeys([*eval_cids_for(behavior), source_cid]))
    if source_cid not in targets:
        targets = [source_cid, *targets]

    probes = ex.load_eval_probes(behavior)
    if args.max_probes:
        probes = probes[: args.max_probes]
    logger.info(
        "phase0 base-leg cell behavior=%s source=%s seed=%d | %d targets x %d probes x layers=%s",
        behavior,
        source_cid,
        seed,
        len(targets),
        len(probes),
        layers,
    )

    # Load BASE θ0 ONLY — NO adapter, NO PeftModel (the pre-spend point: Phase 0
    # never touches the #537 adapter, so it is genuinely cheaper than Phase 1).
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ex.BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    if args.pre_user:
        # KILL-2 startup assert (plan §7): header ids verified in-process BEFORE
        # any GPU work; a tokenizer drift HALTs the cell here.
        ex.assert_header_ids(tok)

    # Phase A: vLLM batched greedy R from BASE (per CLAUDE.md — never a per-prompt
    # HF generate loop). CPU-smoke (no vLLM) falls back to HF greedy per probe.
    r_lookup: dict[tuple[str, int], str] = {}
    if device.type != "cpu":
        gen_msgs: list[list[dict]] = []
        gen_keys: list[tuple[str, int]] = []
        for tcid in targets:
            for qi, q in enumerate(probes):
                gen_msgs.append(ex.build_messages_for(registry, demos, tcid, behavior, q))
                gen_keys.append((tcid, qi))
        logger.info("phase0 Phase A: vLLM-generating %d base R responses", len(gen_msgs))
        responses = ex.vllm_generate_R(tok, gen_msgs, max_new_tokens=args.max_new_tokens)
        r_lookup = dict(zip(gen_keys, responses, strict=True))
        # Persist the rollout TEXT the moment generation completes, BEFORE the
        # teacher-force reduce (#779; Upload Policy raw-completions row).
        ex.persist_r_text(
            Path(args.out), behavior, source_cid, tok, r_lookup, stage="phase0_extraction"
        )

    base = AutoModelForCausalLM.from_pretrained(
        ex.BASE_MODEL, torch_dtype=dtype, token=os.environ.get("HF_TOKEN")
    ).to(device)
    base.eval()
    assert base.config.hidden_size == ex.HIDDEN_SIZE or device.type == "cpu", (
        base.config.hidden_size
    )

    out_root = Path(args.out)
    cell_dir = out_root / behavior / f"{source_cid}_seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    n_gen = n_trunc = 0
    for tcid in targets:
        ng, nt = _extract_base_target(
            base,
            tok,
            registry,
            demos,
            cell_dir,
            behavior,
            tcid,
            probes,
            layers,
            args.primary_layer,
            device,
            r_lookup,
            pre_user=bool(args.pre_user),
        )
        n_gen += ng
        n_trunc += nt
    logger.info(
        "phase0 cell %s/%s done: %d targets, %d generations (%d empty)",
        behavior,
        source_cid,
        len(targets),
        n_gen,
        n_trunc,
    )
    # Re-dump R after the loop: on the CPU-smoke path (no vLLM Phase A) the
    # per-probe HF-greedy fallbacks were recorded into r_lookup during the loop.
    ex.persist_r_text(
        Path(args.out), behavior, source_cid, tok, r_lookup, stage="phase0_extraction"
    )

    # Atomic completion sentinel — written ONLY after every target's base .npz is
    # on disk, so a dispatcher resume-skip never treats a partial dir as done.
    (cell_dir / ex.CELL_DONE_SENTINEL).write_text(
        json.dumps(
            {
                "behavior": behavior,
                "source_cid": source_cid,
                "seed": seed,
                "targets": targets,
                "layers": layers,
                "phase": "phase0_base_leg",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
    )
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Issue #811 Phase-0 base-leg-only extraction (KILL-1)")
    ap.add_argument("--behavior", required=True)
    ap.add_argument("--source-cid", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", nargs="+", type=int, default=[14])
    ap.add_argument("--primary-layer", type=int, default=14)
    ap.add_argument(
        "--targets", default=None, help="comma-separated target cids (default: 30 eval)"
    )
    ap.add_argument("--out", default="eval_results/issue_811/phase0_base_leg")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--max-probes", type=int, default=None, help="smoke: cap probes per behavior")
    ap.add_argument("--max-new-tokens", type=int, default=ex.N_GEN_TOKENS)
    ap.add_argument(
        "--pre-user",
        action="store_true",
        help="#811 pre-user-boundary-summary round: ALSO capture the nine "
        "boundary/header base-leg arms (v0_<slug>) + the two fp16 arm-6/7 "
        "all-layer stacks per cell (the per-arm KILL-1 gate inputs, plan §4.3 "
        "item 2); header ids asserted in-process at startup + per row.",
    )
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
