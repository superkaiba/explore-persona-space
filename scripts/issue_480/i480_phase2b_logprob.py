# ruff: noqa: RUF003  # research code uses Greek letters (ρ, Δ), × and − legitimately
"""Task #480 Phase 2b — HF Transformers logprob extraction at post-response slot.

Reads Phase 2a's ``r_trained.json``, then for ONE source:
1. Loads the TRAINED merged model AND the BASE Qwen2.5-7B-Instruct as
   separate HF Transformers `AutoModelForCausalLM` instances.
2. For each (panel, question) cell:
   - Builds the tokenized teacher-forced sequence:
       ``T_panel(q) + R_trained_by_panel(q) + ` ※```
   - Computes ``log P(MARKER_ID | preceding tokens)`` at the slot
     immediately after R_trained (the post-response slot — same slot the
     #474 collator pushes against on the negative rows).
   - Computes the same quantity under BASE for marker-Δ.
   - Argmax-check at the same slot for the on-policy emission anchor.
3. Aggregates per (source, panel) cell: ``median(marker_delta)``,
   ``mean(emission_rate)``, ``median(log_p_trained)``, plus the per-cell
   R_trained token-length mean/median (consumed by the H1 response-length
   partial in i480_analyze.py).
4. Writes ``marker_logprob_eval.json`` under the per-source out-dir.

If running this in the same Python process as Phase 2a would crash on
the vLLM-teardown bug (see gotchas.md), but this script is a FRESH
process — kernel reaped vLLM workers when 2a exited.

Memory: loads two 7B bf16 models simultaneously on a single H100 (80 GB);
peak ~28 GB. If OOM, fall back to two-pass mode (``--two-pass``) which
splits trained-then-base into separate model loads with a JSON IPC step.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

import torch
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_480.phase2b")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _build_slot_context(tokenizer, panel_system_prompt: str, q: str, r_trained: str) -> str:
    """Build the teacher-forced prefix text whose LAST token precedes the marker slot.

    Sequence: ``T_panel(q) + R_trained`` — the chat-template prompt render
    (with generation prompt) plus R_trained appended literally as the
    assistant response BODY (no chat-template wrapping — we deliberately
    stop BEFORE any <|im_end|> so the slot is the "response continuation"
    position the collator's negative rows pushed against).

    The post-response slot is then the next-token distribution at the final
    prefix position — exactly the slot the parent's ``_resolve_post_response_
    slot``/``_score_one`` pair read (prefix encoded with
    ``add_special_tokens=False``, logits at index ``len(prefix_ids) - 1``),
    and exactly where ``compute_marker_slot_stats`` reads (``logits[i, -1]``
    on the verbatim-encoded context). Same tokenization, same position.
    """
    msgs_prompt: list[dict[str, str]] = []
    if panel_system_prompt and panel_system_prompt != "":
        msgs_prompt.append({"role": "system", "content": panel_system_prompt})
    msgs_prompt.append({"role": "user", "content": q})
    prompt_text = tokenizer.apply_chat_template(
        msgs_prompt, tokenize=False, add_generation_prompt=True
    )
    return prompt_text + r_trained


def _cell_row(
    panel: str,
    qi: int,
    trained: dict[str, float],
    base: dict[str, float],
    marker_id: int,
    r_trained_token_len: int,
) -> dict:
    """Assemble one per-(panel, q) row from the trained + base slot stats.

    Legacy fields (``log_p_trained``, ``log_p_base``, ``marker_delta``,
    ``emission``, ``r_trained_token_len``) are kept with identical
    definitions (``logp = z_marker - logZ`` is the exact identity the legacy
    ``log_softmax`` read computed; ``emission`` = slot argmax == marker id).
    The four-float storage contract (#530) adds the raw-logit fields per
    side, plus the derived EOS margin and Δz_marker.
    """
    eos_margin_delta = (trained["z_marker"] - trained["z_eos"]) - (base["z_marker"] - base["z_eos"])
    return {
        "panel": panel,
        "q_idx": qi,
        "log_p_trained": trained["logp"],
        "log_p_base": base["logp"],
        "marker_delta": trained["logp"] - base["logp"],
        "emission": bool(int(trained["argmax_id"]) == marker_id),
        "z_marker_trained": trained["z_marker"],
        "z_eos_trained": trained["z_eos"],
        "logZ_trained": trained["logZ"],
        "z_marker_base": base["z_marker"],
        "z_eos_base": base["z_eos"],
        "logZ_base": base["logZ"],
        "eos_margin_delta": eos_margin_delta,
        "delta_z_marker": trained["z_marker"] - base["z_marker"],
        "r_trained_token_len": r_trained_token_len,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--r-trained-path", type=Path, required=True)
    parser.add_argument("--merged-model-path", type=Path, required=True)
    parser.add_argument(
        "--out-path", type=Path, required=True, help="marker_logprob_eval.json output path"
    )
    parser.add_argument(
        "--two-pass",
        action="store_true",
        help="Split trained-then-base into separate model loads (HBM-tight fallback).",
    )
    parser.add_argument(
        "--adapter-config-path",
        type=Path,
        default=None,
        help="Path to the evaluated checkpoint's adapter_config.json. When given, "
        "assert_gauge_free_adapter_config fails LOUD if the adapter touches "
        "lm_head/embed_tokens or sets modules_to_save (the trained - base logit "
        "readout is invalid otherwise). Default None preserves the parent path.",
    )
    parser.add_argument(
        "--sentinel-path",
        type=Path,
        default=Path("/workspace/logs/issue-480-phase2b-results.json"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import (
        assert_gauge_free_adapter_config,
        compute_marker_slot_stats,
    )
    from explore_persona_space.experiments.marker_implant_480 import (
        IM_END_ID,
        MARKER_ID,
        MARKER_TEXT,
    )

    # Gauge assert (band-stop recipe): the EOS-margin / Δz_marker readouts
    # below are valid only when LoRA never touched the unembedding.
    gauge_asserted = False
    if args.adapter_config_path is not None:
        with open(args.adapter_config_path) as f:
            adapter_cfg = json.load(f)
        assert_gauge_free_adapter_config(adapter_cfg, context=str(args.adapter_config_path))
        gauge_asserted = True
        log.info("[phase=phase2b] gauge assert PASSED for %s", args.adapter_config_path)

    # Sanity: tokenizer-ids must be stable.
    tokenizer = AutoTokenizer.from_pretrained(str(args.merged_model_path))
    marker_check = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if marker_check != [MARKER_ID]:
        raise RuntimeError(
            f"marker token id drifted: {MARKER_TEXT!r} -> {marker_check}, expected [{MARKER_ID}]"
        )
    im_end_check = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if im_end_check != [IM_END_ID]:
        raise RuntimeError(f"im_end token id drifted: -> {im_end_check}, expected [{IM_END_ID}]")
    log.info("[phase=phase2b] tokenizer ids OK marker=%s im_end=%s", marker_check, im_end_check)

    with open(args.r_trained_path) as f:
        r_payload = json.load(f)
    panel_personas = r_payload["panel_personas"]
    panel_system_prompts = r_payload["panel_system_prompts"]
    questions = r_payload["questions"]
    r_trained_all = r_payload["r_trained"]
    log.info(
        "[phase=phase2b] source=%s n_panel=%d n_q=%d",
        args.source,
        len(panel_personas),
        len(questions),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(path: str):
        log.info("[phase=phase2b] Loading %s -> %s", path, device)
        m = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        m.eval()
        return m

    cells: list[dict] = []

    # Per-panel context lists (the prefix text whose last token precedes the
    # post-response slot). batch_size=1 keeps every forward unpadded — exactly
    # the parent's serial compute pattern, so the legacy log_p_* values are
    # reproduced on the same slot definition (float32 softmax vs the legacy
    # bf16 log_softmax is the only numeric difference).
    contexts_by_panel: dict[str, list[str]] = {}
    r_len_by_panel: dict[str, list[int]] = {}
    for panel in panel_personas:
        sys_p = panel_system_prompts[panel]
        contexts_by_panel[panel] = [
            _build_slot_context(tokenizer, sys_p, q, r_trained_all[panel][qi])
            for qi, q in enumerate(questions)
        ]
        r_len_by_panel[panel] = [
            len(tokenizer.encode(r, add_special_tokens=False)) for r in r_trained_all[panel]
        ]

    def _score_panels(model) -> dict[str, list[dict[str, float]]]:
        """Four-float + argmax slot stats per (panel, q) under ``model``."""
        out: dict[str, list[dict[str, float]]] = {}
        for panel in panel_personas:
            out[panel] = compute_marker_slot_stats(
                model,
                tokenizer,
                contexts=contexts_by_panel[panel],
                marker_text=MARKER_TEXT,
                batch_size=1,
                device=str(device),
                eos_token_id=IM_END_ID,
                include_argmax=True,
            )
        return out

    if args.two_pass:
        # Pass 1: TRAINED model, score every (panel, q).
        log.info("[phase=phase2b] two-pass mode: TRAINED first.")
        model_t = _load_model(str(args.merged_model_path))
        trained_stats = _score_panels(model_t)
        del model_t
        gc.collect()
        torch.cuda.empty_cache()

        # Pass 2: BASE model.
        log.info("[phase=phase2b] two-pass mode: BASE next.")
        model_b = _load_model(BASE_MODEL)
        base_stats = _score_panels(model_b)
        del model_b
    else:
        # One-pass: load both models simultaneously.
        model_t = _load_model(str(args.merged_model_path))
        model_b = _load_model(BASE_MODEL)
        trained_stats = _score_panels(model_t)
        base_stats = _score_panels(model_b)
        del model_t, model_b

    for panel in panel_personas:
        for qi in range(len(questions)):
            cells.append(
                _cell_row(
                    panel=panel,
                    qi=qi,
                    trained=trained_stats[panel][qi],
                    base=base_stats[panel][qi],
                    marker_id=MARKER_ID,
                    r_trained_token_len=r_len_by_panel[panel][qi],
                )
            )

    gc.collect()
    torch.cuda.empty_cache()

    # Per-(source, panel) aggregation. The DV: median(marker_delta) across
    # the 50 Q_eval. Continuous log-prob subsumes binary emission, but log
    # emission rate as the legibility/sanity anchor.
    per_panel: dict[str, dict[str, float]] = {}
    for panel in panel_personas:
        panel_rows = [c for c in cells if c["panel"] == panel]
        if not panel_rows:
            continue
        # mean + sample-stdev for the per-cell measurement-noise SE used by the
        # noise-tolerant ranking power-match in i480_analyze.py. mean is reported
        # alongside median so the analyzer can pick whichever aggregate matches
        # the SE definition (mean ↔ SEM = std / sqrt(n)).
        from math import sqrt as _sqrt  # local import — ruff F401-safe
        from statistics import mean as _mean
        from statistics import stdev as _stdev

        log_p_trained_vals = [r["log_p_trained"] for r in panel_rows]
        log_p_base_vals = [r["log_p_base"] for r in panel_rows]
        marker_delta_vals = [r["marker_delta"] for r in panel_rows]
        emission_vals = [r["emission"] for r in panel_rows]
        r_len_vals = [r["r_trained_token_len"] for r in panel_rows]
        n_q = len(panel_rows)
        marker_delta_std = float(_stdev(marker_delta_vals)) if n_q >= 2 else 0.0
        # Four-float storage contract (#530) aggregates — additive fields;
        # legacy keys below are unchanged.
        four_float_aggs = {
            f"median_{k}": median([r[k] for r in panel_rows])
            for k in (
                "z_marker_trained",
                "z_eos_trained",
                "logZ_trained",
                "z_marker_base",
                "z_eos_base",
                "logZ_base",
                "eos_margin_delta",
                "delta_z_marker",
            )
        }
        per_panel[panel] = {
            **four_float_aggs,
            "median_marker_delta": median(marker_delta_vals),
            "mean_marker_delta": float(_mean(marker_delta_vals)),
            "marker_delta_std": marker_delta_std,
            # SEM of the mean — the measurement-noise SE the noise-tolerant
            # ranking treats as the tie-tolerance threshold (× 2). For the
            # median this slightly overstates precision (medians have an
            # asymptotic SE ≈ 1.253 × SEM_mean for Gaussian samples); we
            # report SEM_mean here because the analyzer multiplies by 2 to
            # form the tie band, which dominates that constant.
            "marker_delta_se": marker_delta_std / _sqrt(n_q) if n_q >= 2 else 0.0,
            "mean_emission_rate": sum(emission_vals) / len(emission_vals),
            "median_log_p_trained": median(log_p_trained_vals),
            "median_log_p_base": median(log_p_base_vals),
            "r_trained_len_mean": sum(r_len_vals) / len(r_len_vals),
            "r_trained_len_median": median(r_len_vals),
            "n_q": n_q,
        }

    out_payload = {
        "source": args.source,
        "seed": args.seed,
        "marker_text": MARKER_TEXT,
        "marker_id": MARKER_ID,
        "im_end_id": IM_END_ID,
        "merged_model_path": str(args.merged_model_path),
        "base_model": BASE_MODEL,
        "n_panel": len(panel_personas),
        "n_questions": len(questions),
        "adapter_config_path": (
            str(args.adapter_config_path) if args.adapter_config_path is not None else None
        ),
        "gauge_asserted": gauge_asserted,
        "per_panel": per_panel,
        "per_cell_rows": cells,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(out_payload, f, ensure_ascii=False)
    log.info("[phase=phase2b] marker_logprob_eval -> %s", args.out_path)

    sentinel = {
        "phase": "phase2b_logprob",
        "issue": 480,
        "source": args.source,
        "seed": args.seed,
        "out_path": str(args.out_path),
        "n_cells": len(cells),
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.sentinel_path, "w") as f:
        json.dump(sentinel, f, indent=2)
    log.info("[phase=phase2b] DONE sentinel=%s", args.sentinel_path)
    print("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
