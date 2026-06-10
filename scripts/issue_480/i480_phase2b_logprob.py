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

Scoring modes (round-6 fix — Codex v5 critical ``--recipe parent`` byte-identity):

``--slot-stats legacy`` (DEFAULT)
    The verbatim round-1 implementation: ``_resolve_post_response_slot`` +
    ``_score_one`` (bf16 ``log_softmax``), parent's exact per-cell row,
    per-panel aggregate, and top-level payload key sets. A bare parent-recipe
    invocation (no new flags) reproduces the round-1 output schema exactly.

``--slot-stats four-float``
    The band-stop recipe's path: ``compute_marker_slot_stats`` (float32
    softmax, batch_size=1 — the same unpadded serial compute pattern) with
    the #530 four-float storage contract (z_marker / z_eos / logZ / logp per
    side) + derived EOS margin and Δz_marker, additively alongside every
    legacy field. Requires (and is required by) ``--adapter-config-path``
    gauge assertion support; the dispatcher passes this mode ONLY in
    ``--recipe band_stop``.

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
import torch.nn.functional as F
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


# ── legacy (round-1 verbatim) scoring path ───────────────────────────────────
# Ported unchanged from the parent SHA 4b2b4bbee — the `--recipe parent`
# byte-identity contract reads through these two functions.


def _resolve_post_response_slot(
    tokenizer, panel_system_prompt: str, q: str, r_trained: str, marker_text: str
) -> tuple[list[int], int, int]:
    """Build the teacher-forced sequence and return (full_ids, slot, prompt_len).

    Sequence: ``T_panel(q) + R_trained + marker``.

    The ``slot`` is the index of the marker token in ``full_ids``; the model's
    teacher-forced log-prob distribution at position ``slot - 1`` predicts
    the token at ``slot``. We require the slot to be in the completion
    region (i.e. > P, where P is the length of the prompt+R encoding).

    Returns:
        (full_ids, slot, prompt_len_with_R) — slot points at the marker
        token; the post-response slot for log-prob computation is
        ``slot - 1`` (the position whose distribution predicts the marker).
    """
    # Prompt-only (with generation prompt) tokenization
    msgs_prompt: list[dict[str, str]] = []
    if panel_system_prompt and panel_system_prompt != "":
        msgs_prompt.append({"role": "system", "content": panel_system_prompt})
    msgs_prompt.append({"role": "user", "content": q})
    prompt_text = tokenizer.apply_chat_template(
        msgs_prompt, tokenize=False, add_generation_prompt=True
    )
    # Append R_trained literally as the assistant response BODY (no chat-template
    # wrapping — we deliberately stitch the marker AFTER R without an <|im_end|>
    # in between so the marker sits at the "response continuation" slot the
    # collator's negative rows pushed against).
    prefix_text = prompt_text + r_trained
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
    if len(marker_ids) != 1:
        raise RuntimeError(
            f"marker tokenized to {len(marker_ids)} tokens ({marker_ids}); expected 1"
        )
    full_ids = prefix_ids + marker_ids
    slot = len(prefix_ids)  # index of the marker token in full_ids
    return full_ids, slot, len(prefix_ids)


@torch.no_grad()
def _score_one(
    model,
    tokenizer,
    full_ids: list[int],
    slot: int,
    marker_id: int,
    device,
) -> tuple[float, bool, int]:
    """Compute (log P(marker @ slot), argmax_is_marker, R_trained_token_len).

    The R_trained token length is ``slot - prompt_len_with_template`` — but
    here ``prompt_len_with_template`` was already absorbed into ``slot``
    accounting upstream; we just return slot here for the caller to use.
    """
    if slot < 1 or slot >= len(full_ids):
        raise RuntimeError(f"invalid slot {slot} for length {len(full_ids)}")
    input_ids = torch.tensor([full_ids[: slot + 1]], device=device, dtype=torch.long)
    out = model(input_ids=input_ids)
    logits = out.logits[0, slot - 1]
    logp = F.log_softmax(logits, dim=-1)
    log_p_marker = float(logp[marker_id].item())
    argmax_id = int(torch.argmax(logits).item())
    return log_p_marker, (argmax_id == marker_id), slot


# ── four-float (band_stop) scoring path ──────────────────────────────────────


def _build_slot_context(tokenizer, panel_system_prompt: str, q: str, r_trained: str) -> str:
    """Build the teacher-forced prefix text whose LAST token precedes the marker slot.

    Sequence: ``T_panel(q) + R_trained`` — the chat-template prompt render
    (with generation prompt) plus R_trained appended literally as the
    assistant response BODY (no chat-template wrapping — we deliberately
    stop BEFORE any <|im_end|> so the slot is the "response continuation"
    position the collator's negative rows pushed against).

    The post-response slot is then the next-token distribution at the final
    prefix position — exactly the slot the legacy ``_resolve_post_response_
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


# Per-cell aggregate keys added ONLY in four-float mode (medians of the
# four-float row fields). Legacy mode never emits them — parent schema parity.
FOUR_FLOAT_ROW_KEYS = (
    "z_marker_trained",
    "z_eos_trained",
    "logZ_trained",
    "z_marker_base",
    "z_eos_base",
    "logZ_base",
    "eos_margin_delta",
    "delta_z_marker",
)


def main(argv: list[str] | None = None) -> int:  # noqa: C901  dual-mode: the legacy branch is the verbatim 4b2b4bbee port (byte-identity contract); extracting it into helpers would defeat the line-for-line diffability against the parent SHA
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
        "--slot-stats",
        choices=("legacy", "four-float"),
        default="legacy",
        help="Scoring path. 'legacy' (default) = the verbatim round-1 "
        "_resolve_post_response_slot/_score_one implementation with the parent's "
        "exact output schema (the --recipe parent byte-identity contract). "
        "'four-float' = compute_marker_slot_stats with the #530 storage contract "
        "(band_stop recipe only).",
    )
    parser.add_argument(
        "--adapter-config-path",
        type=Path,
        default=None,
        help="Path to the evaluated checkpoint's adapter_config.json. "
        "assert_gauge_free_adapter_config fails LOUD if the adapter touches "
        "lm_head/embed_tokens or sets modules_to_save (the trained - base logit "
        "readout is invalid otherwise). REQUIRED by and ONLY valid with "
        "--slot-stats four-float (two-way guard).",
    )
    parser.add_argument(
        "--sentinel-path",
        type=Path,
        default=Path("/workspace/logs/issue-480-phase2b-results.json"),
    )
    args = parser.parse_args(argv)
    # Two-way guard (round-6 concern phase2b-four-float-allows-unguarded-logit-fields):
    # the gauge-sensitive logit fields (z_marker / z_eos / logZ / eos_margin_delta /
    # delta_z_marker) are valid ONLY under the gauge-free assertion, so four-float
    # mode MUST carry the adapter config — and the assertion is meaningless outside
    # four-float mode.
    if args.adapter_config_path is not None and args.slot_stats != "four-float":
        parser.error("--adapter-config-path requires --slot-stats four-float")
    if args.slot_stats == "four-float" and args.adapter_config_path is None:
        parser.error("--slot-stats four-float requires --adapter-config-path")

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.marker_implant_480 import (
        IM_END_ID,
        MARKER_ID,
        MARKER_TEXT,
    )

    four_float = args.slot_stats == "four-float"
    log.info("[phase=phase2b] slot-stats mode: %s", args.slot_stats)

    # Gauge assert (band-stop recipe): the EOS-margin / Δz_marker readouts
    # below are valid only when LoRA never touched the unembedding.
    gauge_asserted = False
    if args.adapter_config_path is not None:
        from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config

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

    if not four_float:
        # ── LEGACY mode: the round-1 loops, verbatim (parent SHA 4b2b4bbee) ──
        if args.two_pass:
            # Pass 1: TRAINED model, score every (panel, q).
            log.info("[phase=phase2b] two-pass mode: TRAINED first.")
            model_t = _load_model(str(args.merged_model_path))
            trained_scores: dict[tuple[str, int], tuple[float, bool, int]] = {}
            full_ids_cache: dict[tuple[str, int], tuple[list[int], int]] = {}
            for panel in panel_personas:
                sys_p = panel_system_prompts[panel]
                for qi, q in enumerate(questions):
                    r = r_trained_all[panel][qi]
                    full_ids, slot, _plen = _resolve_post_response_slot(
                        tokenizer, sys_p, q, r, MARKER_TEXT
                    )
                    full_ids_cache[(panel, qi)] = (full_ids, slot)
                    trained_scores[(panel, qi)] = _score_one(
                        model_t, tokenizer, full_ids, slot, MARKER_ID, device
                    )
            del model_t
            gc.collect()
            torch.cuda.empty_cache()

            # Pass 2: BASE model.
            log.info("[phase=phase2b] two-pass mode: BASE next.")
            model_b = _load_model(BASE_MODEL)
            for (panel, qi), (full_ids, slot) in full_ids_cache.items():
                log_p_t, argmax_t, _ = trained_scores[(panel, qi)]
                log_p_b, _argmax_b, _ = _score_one(
                    model_b, tokenizer, full_ids, slot, MARKER_ID, device
                )
                cells.append(
                    {
                        "panel": panel,
                        "q_idx": qi,
                        "log_p_trained": log_p_t,
                        "log_p_base": log_p_b,
                        "marker_delta": log_p_t - log_p_b,
                        "emission": bool(argmax_t),
                        "r_trained_token_len": len(
                            tokenizer.encode(r_trained_all[panel][qi], add_special_tokens=False)
                        ),
                    }
                )
            del model_b
        else:
            # One-pass: load both models simultaneously.
            model_t = _load_model(str(args.merged_model_path))
            model_b = _load_model(BASE_MODEL)
            for panel in panel_personas:
                sys_p = panel_system_prompts[panel]
                for qi, q in enumerate(questions):
                    r = r_trained_all[panel][qi]
                    full_ids, slot, _plen = _resolve_post_response_slot(
                        tokenizer, sys_p, q, r, MARKER_TEXT
                    )
                    log_p_t, argmax_t, _ = _score_one(
                        model_t, tokenizer, full_ids, slot, MARKER_ID, device
                    )
                    log_p_b, _argmax_b, _ = _score_one(
                        model_b, tokenizer, full_ids, slot, MARKER_ID, device
                    )
                    cells.append(
                        {
                            "panel": panel,
                            "q_idx": qi,
                            "log_p_trained": log_p_t,
                            "log_p_base": log_p_b,
                            "marker_delta": log_p_t - log_p_b,
                            "emission": bool(argmax_t),
                            "r_trained_token_len": len(
                                tokenizer.encode(r, add_special_tokens=False)
                            ),
                        }
                    )
            del model_t, model_b
    else:
        # ── FOUR-FLOAT mode (band_stop recipe): #530 storage contract ──
        from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats

        # Per-panel context lists (the prefix text whose last token precedes
        # the post-response slot). batch_size=1 keeps every forward unpadded —
        # exactly the legacy serial compute pattern, so the legacy log_p_*
        # values are reproduced on the same slot definition (float32 softmax
        # vs the legacy bf16 log_softmax is the only numeric difference).
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
        # Four-float storage contract (#530) aggregates — additive fields
        # emitted ONLY in four-float mode; legacy keys below are unchanged
        # (legacy mode reproduces the parent's exact per-panel key set).
        four_float_aggs = (
            {f"median_{k}": median([r[k] for r in panel_rows]) for k in FOUR_FLOAT_ROW_KEYS}
            if four_float
            else {}
        )
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
        "per_panel": per_panel,
        "per_cell_rows": cells,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    if four_float:
        # Additive provenance keys — NEVER emitted in legacy mode (the parent
        # top-level payload key set is part of the byte-identity contract).
        out_payload["slot_stats"] = args.slot_stats
        out_payload["adapter_config_path"] = (
            str(args.adapter_config_path) if args.adapter_config_path is not None else None
        )
        out_payload["gauge_asserted"] = gauge_asserted
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
