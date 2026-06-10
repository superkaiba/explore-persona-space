"""Phase 3 — Rao-Blackwellized sequence-level JS + both KL directions.

For each (source, bystander) cell (138 cells = 6 sources x 23 bystanders):

  1. Take Phase 1's R responses sampled under SOURCE persona. Teacher-force each
     response through BOTH source-conditioned and bystander-conditioned base
     model. Per-token full-vocab JS averaged over response tokens, then over
     responses, then over probes -> ``JS_from_source(src, bys)``.
     Same forward pass yields ``KL(src || bys)`` averaged the same way (use
     responses sampled from the FIRST KL argument per canonical recipe).

  2. Take Phase 1's R responses sampled under BYSTANDER persona. Teacher-force
     through BOTH source- and bystander-conditioned model. Average JS the same
     way -> ``JS_from_bystander``. Same pass yields ``KL(bys || src)``.

  3. Headline ``JS_sym = 0.5 * (JS_from_source + JS_from_bystander)`` and the
     polarity-aligned similarity ``M_js = 1 - JS_sym / ln(2)``.

Output: ``eval_results/issue_470/sequence_js_kl/{source}__{bystander}.json``,
one file per cell (138 files for the full sweep). Checkpoint per phase: each
cell is written the moment it completes; pre-existing files are skipped.

Unit lock (A13 from plan): JS is reported in NATS (max ~= 0.693). Per-cell
output asserts ``0 <= JS_sym <= 0.6932`` and ``KL >= 0``.

Subprocess-isolated from Phase 1 (vLLM teardown trap).

Usage::

    uv run python -m explore_persona_space.experiments.predictor_jsdiv_470.phase3_sequence_js_kl \\
        --sources software_engineer --bystanders comedian --probes 5    # smoke
        # R is fixed by Phase 1's sampling (R_per_side inferred from the
        # per-persona output); Phase 3 has no --R argument.
    uv run python -m explore_persona_space.experiments.predictor_jsdiv_470.phase3_sequence_js_kl
        # full: 6 sources x 23 bystanders = 138 cells
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

import numpy as np
from dotenv import load_dotenv

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
load_dotenv()

from explore_persona_space.experiments.predictor_jsdiv_470 import (  # noqa: E402
    SOURCE_PERSONAS_411,
)
from explore_persona_space.experiments.predictor_jsdiv_470.common import (  # noqa: E402
    DEFAULT_MODEL,
    PHASE1_DIR,
    PHASE3_DIR,
    checkpoint_is_compatible,
    get_eval_personas_24,
    read_json,
    reproducibility_metadata,
    write_json,
)

logger = logging.getLogger("predictor_jsdiv_470.phase3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LN2 = math.log(2.0)
JS_MAX = LN2 + 1e-3  # tolerance for fp32 round-off


def _safe_assert_ranges(js_vals: list[float], kl_vals: list[float], label: str) -> None:
    """Assert per-cell sanity per plan A13 (kill criterion + numerical-stability)."""
    if not js_vals:
        return
    js_arr = np.array(js_vals, dtype=float)
    kl_arr = np.array(kl_vals, dtype=float)
    if np.any(np.isnan(js_arr)) or np.any(np.isnan(kl_arr)):
        raise RuntimeError(f"{label}: NaN encountered in JS/KL outputs")
    if not np.all((js_arr >= -1e-6) & (js_arr <= JS_MAX)):
        bad = js_arr[(js_arr < -1e-6) | (js_arr > JS_MAX)]
        raise RuntimeError(
            f"{label}: JS values outside [0, ln 2] (got {bad[:5]}). "
            f"Unit lock violated — check that compute_js_divergence returns nats."
        )
    if not np.all(kl_arr >= -1e-6):
        bad = kl_arr[kl_arr < -1e-6]
        raise RuntimeError(f"{label}: negative KL values ({bad[:5]})")


def _compute_one_direction(
    model,
    tokenizer,
    src_prompt: str,
    bys_prompt: str,
    responses_from: list[list[str]],  # shape (n_probes, R) — sampled FROM the "from" persona
    probes: list[str],
    kl_from_first: bool,
    device,
    tf_batch: int,
) -> tuple[list[float], list[float]]:
    """For ONE side of the RB average, teacher-force ``responses_from`` (sampled
    from one persona) through BOTH (source, bystander) conditioned model and
    compute per-(probe, response) JS + KL.

    Args:
      kl_from_first: if True, returns ``KL(src || bys)`` (responses must have
        been sampled FROM src per the canonical "sample from the first KL
        argument" rule); if False, returns ``KL(bys || src)``.

    Returns: ``(js_list, kl_list)`` length = n_probes * R, each entry the
    mean-per-token divergence for one (probe, response) pair.
    """

    from explore_persona_space.analysis.divergence import (
        build_teacher_force_inputs,
        teacher_force_and_reduce_js_kl,
    )

    js_per_pair: list[float] = []
    kl_per_pair: list[float] = []

    sys_prompts = [src_prompt, bys_prompt]
    # sys_prompts[0] = src, sys_prompts[1] = bys ⇒ batch row 0 = src-conditioned,
    # row 1 = bys-conditioned. Fused helper takes p_index/q_index so we choose
    # the KL direction without recomputing the forward pass.
    src_idx, bys_idx = 0, 1

    for probe_idx, probe in enumerate(probes):
        for response in responses_from[probe_idx]:
            if not response.strip():
                # vLLM sometimes returns empty completions (greedy refusal,
                # immediate EOS). The teacher-force pipeline requires a
                # non-empty response (the response_token_ids list is empty
                # otherwise). Skip + log; the RB average still gets the rest.
                logger.warning(
                    "Empty response at probe_idx=%d (one of R); skipping cell sample",
                    probe_idx,
                )
                continue
            try:
                batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
                    tokenizer=tokenizer,
                    system_prompts=sys_prompts,
                    question=probe,
                    response_text=response,
                )
            except ValueError as e:
                logger.warning(
                    "build_teacher_force_inputs failed (probe_idx=%d): %s; skipping",
                    probe_idx,
                    e,
                )
                continue

            # Fused forward + on-GPU JS/KL reduction. The (response_len x ~152K-vocab)
            # log-softmax tensors are reduced on the GPU and freed before this returns;
            # only three Python floats cross the PCIe bus. (Was: full-vocab .cpu() then
            # CPU JS/KL reduction -> GPU-idle / CPU-bound, ~10 min/cell at 50 probes x R=8.)
            if kl_from_first:
                # Responses sampled from src; KL(src || bys): P=src, Q=bys.
                p_index, q_index = src_idx, bys_idx
            else:
                # Responses sampled from bys; KL(bys || src): P=bys, Q=src.
                p_index, q_index = bys_idx, src_idx

            js, kl, _kl_reverse = teacher_force_and_reduce_js_kl(
                model=model,
                batch_inputs=batch_inputs,
                prompt_lengths=prompt_lengths,
                response_len=response_len,
                device=str(device),
                max_batch=tf_batch,
                p_index=p_index,
                q_index=q_index,
            )

            js_per_pair.append(js)
            kl_per_pair.append(kl)

    return js_per_pair, kl_per_pair


def compute_cell(
    model,
    tokenizer,
    *,
    source: str,
    bystander: str,
    persona_prompts: dict[str, str],
    src_responses: list[list[str]],
    bys_responses: list[list[str]],
    probes: list[str],
    device,
    tf_batch: int = 2,
) -> dict:
    """RB JS + both-KL for one (source, bystander) cell."""
    src_prompt = persona_prompts[source]
    bys_prompt = persona_prompts[bystander]

    # Side A: responses sampled from source -> KL(src || bys), JS_from_source.
    js_a, kl_src_to_bys = _compute_one_direction(
        model=model,
        tokenizer=tokenizer,
        src_prompt=src_prompt,
        bys_prompt=bys_prompt,
        responses_from=src_responses,
        probes=probes,
        kl_from_first=True,
        device=device,
        tf_batch=tf_batch,
    )

    # Side B: responses sampled from bystander -> KL(bys || src), JS_from_bystander.
    js_b, kl_bys_to_src = _compute_one_direction(
        model=model,
        tokenizer=tokenizer,
        src_prompt=src_prompt,
        bys_prompt=bys_prompt,
        responses_from=bys_responses,
        probes=probes,
        kl_from_first=False,
        device=device,
        tf_batch=tf_batch,
    )

    _safe_assert_ranges(js_a + js_b, kl_src_to_bys + kl_bys_to_src, f"cell {source}__{bystander}")

    if not js_a or not js_b:
        raise RuntimeError(
            f"Cell {source}__{bystander}: empty side (|js_a|={len(js_a)}, "
            f"|js_b|={len(js_b)}). Sampled responses all empty?"
        )

    js_from_source = float(np.mean(js_a))
    js_from_bystander = float(np.mean(js_b))
    js_sym = 0.5 * (js_from_source + js_from_bystander)
    kl_src_to_bys_mean = float(np.mean(kl_src_to_bys))
    kl_bys_to_src_mean = float(np.mean(kl_bys_to_src))
    kl_sym = 0.5 * (kl_src_to_bys_mean + kl_bys_to_src_mean)

    return {
        "source": source,
        "bystander": bystander,
        "n_probes": len(probes),
        "R_per_side": len(src_responses[0]) if src_responses else 0,
        # Headline: nats, bounded [0, ln 2].
        "JS_sym_nats": js_sym,
        "JS_from_source_nats": js_from_source,
        "JS_from_bystander_nats": js_from_bystander,
        # Polarity-aligned similarity in [0, 1] for figure use.
        "M_js": 1.0 - js_sym / LN2,
        # KL directions, nats, >=0.
        "KL_src_to_bys_nats": kl_src_to_bys_mean,
        "KL_bys_to_src_nats": kl_bys_to_src_mean,
        "KL_sym_nats": kl_sym,
        # Counts so the analyzer can detect dropped samples.
        "n_samples_side_a": len(js_a),
        "n_samples_side_b": len(js_b),
    }


def main() -> int:  # noqa: C901 — linear setup + per-cell loop reads clearer inline
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_PERSONAS_411),
        help="Sources to score (default: all 6 #411 sources).",
    )
    parser.add_argument(
        "--bystanders",
        nargs="+",
        default=None,
        help="Bystanders to score (default: all 23 = panel minus source).",
    )
    parser.add_argument(
        "--probes",
        type=int,
        default=None,
        help="Cap to first N probes (smoke mode). Must match Phase 1 cap.",
    )
    parser.add_argument("--tf-batch", type=int, default=2)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)

    persona_prompts = get_eval_personas_24()
    sources = list(args.sources)
    unknown = [s for s in sources if s not in persona_prompts]
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}")

    # Build the per-source bystander list (panel minus source).
    if args.bystanders:
        # Concern #8 mirror — also fail-fast on unknown bystanders.
        unknown_b = [b for b in args.bystanders if b not in persona_prompts]
        if unknown_b:
            raise ValueError(f"Unknown bystanders: {unknown_b}")
        bystanders_per_source = {s: list(args.bystanders) for s in sources}
    else:
        bystanders_per_source = {s: [p for p in persona_prompts if p != s] for s in sources}

    # Blocker #2: COMPATIBILITY skip instead of filename skip — a smoke artifact
    # (R=2 + 5 probes + Qwen-0.5B) must NOT silently satisfy production
    # (R=8 + 50 probes + Qwen-7B). Build candidate list now; filter below once
    # the expected signature (probe count, R, model) is known from Phase 1.
    all_cells: list[tuple[str, str]] = []
    for src in sources:
        for bys in bystanders_per_source[src]:
            all_cells.append((src, bys))

    # Load Phase 1 outputs for the personas we need (must exist for all cells
    # before we can decide compatibility).
    needed_personas = {p for cell in all_cells for p in cell}
    phase1: dict[str, dict] = {}
    for persona in needed_personas:
        path = PHASE1_DIR / f"{persona}.json"
        if not path.exists():
            raise RuntimeError(f"Phase 1 output missing for persona={persona}: {path}")
        phase1[persona] = read_json(path)

    # Trim probes if requested.
    sample_persona = next(iter(needed_personas))
    probes = phase1[sample_persona]["probes"]
    if args.probes is not None:
        probes = probes[: args.probes]
        for persona in needed_personas:
            phase1[persona]["probes"] = phase1[persona]["probes"][: args.probes]
            phase1[persona]["responses"] = phase1[persona]["responses"][: args.probes]
    # Sanity-check: all personas must agree on the (trimmed) probe set.
    for persona in needed_personas:
        if phase1[persona]["probes"] != probes:
            raise RuntimeError(f"Probe-set drift between Phase 1 outputs: {persona} disagrees")

    # Infer R from Phase 1 (responses shape is (n_probes, R)).
    sample_responses = phase1[sample_persona]["responses"]
    inferred_r = len(sample_responses[0]) if sample_responses else 0

    expected_sig = {
        "model_path": args.model,
        "phase": "phase3_sequence_js_kl",
        "R_per_side": inferred_r,
        "n_probes": len(probes),
    }

    pending: list[tuple[str, str]] = []
    for src, bys in all_cells:
        out_path = PHASE3_DIR / f"{src}__{bys}.json"
        ok, reason = checkpoint_is_compatible(out_path, expected_sig)
        if ok:
            continue
        if out_path.exists():
            logger.warning(
                "Regenerating %s: existing cell INCOMPATIBLE (%s)", out_path.name, reason
            )
        pending.append((src, bys))

    if not pending:
        logger.info("Phase 3: all %d cells already COMPATIBLE; nothing to do.", len(all_cells))
        return 0
    logger.info("Phase 3: %d/%d cells pending", len(pending), len(all_cells))

    # Load model.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model %s on %s", args.model, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": device} if device.type == "cuda" else None,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Iterate cells; checkpoint per cell.
    for src, bys in pending:
        logger.info("Cell %s -> %s (n_probes=%d)", src, bys, len(probes))
        result = compute_cell(
            model=model,
            tokenizer=tokenizer,
            source=src,
            bystander=bys,
            persona_prompts=persona_prompts,
            src_responses=phase1[src]["responses"],
            bys_responses=phase1[bys]["responses"],
            probes=probes,
            device=device,
            tf_batch=args.tf_batch,
        )
        result["metadata"] = reproducibility_metadata(
            {
                "script": "predictor_jsdiv_470.phase3_sequence_js_kl",
                "phase": "phase3_sequence_js_kl",
                "model_path": args.model,
                "R_per_side": result.get("R_per_side"),
                "n_probes": result.get("n_probes"),
            }
        )
        write_json(PHASE3_DIR / f"{src}__{bys}.json", result)
        logger.info(
            "  -> JS_sym=%.4f nats, M_js=%.4f, KL(src||bys)=%.4f, KL(bys||src)=%.4f",
            result["JS_sym_nats"],
            result["M_js"],
            result["KL_src_to_bys_nats"],
            result["KL_bys_to_src_nats"],
        )

    logger.info("Phase 3 complete. Outputs in %s", PHASE3_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
