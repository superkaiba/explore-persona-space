#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, σ, →, —, ≥) in scientific docstrings + logs.
"""Issue #468 Phase B — extraction-position / response variants on the base model.

Direct extension of `scripts/issue463_predictor_cossim.py` (recovered into
this branch). Same model, same 18 cells, same K=8/NL prompts, same per-cell
training (or Betley) probe sets, same per-probe cosine + mean-over-probes
reduction. Sweeps a new family of residual-stream extraction recipes::

* **V0** chat-template position diagnostic — single (system, q), prints
  decoded token + per-position residual norm for the trailing band.
  Logged, no cosine.
* **V1** ``last_prompt_token_final_content`` — read at the LAST
  USER-CONTENT token (= second ``<|im_end|>`` index minus one). Anchors
  the V5 sweep at ``p0``.
* **V2** ``last_response_token`` — read at the FINAL token of the
  model's own generated response (before EOS).
* **V3** ``response_mean_skip_k`` — mean-pool residuals over response
  positions ``[k:]``. PRIMARY ``k=8`` (FIXED, not smoke-tuned).
  ``--skip-k-sweep`` swaps the primary k for a list of k values
  reported as separate keys (exploratory; see plan §4.2.4).
* **V4** ``response_max`` — per-dim max-pool over response positions.
* **V5** trailing-template position sweep — RIDES FREE on V1's forward
  pass. Reads at each of 6 positions ``p0..p5`` in the trailing band
  (``p0`` = last user-content token = V1; ``p5`` = final ``\\n`` = #463's
  ``T-1`` slot). Distinguishes the three branches of plan §6.2.

For G2 cross-check, also recomputes ``last_prompt_token`` (= V5 ``p5``)
and ``response_mean`` (= V3 with ``k=0``).

NEW v2 covariate: pre-block token-embedding-bag cosine — embed S_narrow
and S_broad text through ``model.model.embed_tokens`` (pre-transformer
lookup), mean-pool, cosine. One scalar per (cell, flavor); cached in the
same JSON output under ``lexical_token_embedding_bag_cos``.

Output per (cell, flavor) at
``eval_results/issue468/predictor_cossim_variants_{betley,training}/<cell>_<flavor>.json``.

Usage::

    # Smoke (= sweep with one cell, all variants, one layer, one flavor)
    uv run python scripts/issue468_predictor_cossim_variants.py \\
        --pairs insecure_code --flavors NL --probe-source training \\
        --layers 21 --variants v0 v1 v2 v3 v4 v5 --skip-k 8 --lexical-bag

    # Sweep (the full 18-cell run in plan §4.2.9)
    uv run python scripts/issue468_predictor_cossim_variants.py \\
        --pairs insecure_code jailbroken ... \\
        --flavors NL lit --probe-source training \\
        --layers 18 20 21 22 24 25 27 \\
        --variants v1 v2 v3 v4 v5 --skip-k 8 --lexical-bag
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import (  # noqa: E402
    LIT_FLAVOR_N_ROWS,
    LITERAL_ATTRIBUTE_K,
    PAIRS,
    S_BROAD,
    S_NARROW_NL,
    build_literal_attribute_system_prompt,
    ensure_dataset,
    fetch_betley_main_8,
    fetch_preregistered_probes,
    load_jsonl,
    reproducibility_metadata,
)
from issue463_predictor_cossim import (  # noqa: E402
    _attach_layer_hooks,
    _build_prompt_ids,
    _per_layer_cos_sim,
    extract_training_probes,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue468_predictor_cossim_variants")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LAYERS = [18, 20, 21, 22, 24, 25, 27]
DEFAULT_N_PROBES = 48
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_SKIP_K = 8
ALL_VARIANTS = ("v0", "v1", "v2", "v3", "v4", "v5")
POSITION_NAMES = ("p0", "p1", "p2", "p3", "p4", "p5")
POSITION_DESCRIPTIONS = {
    "p0": "last-user-content-token (= V1)",
    "p1": "user-close-<|im_end|>",
    "p2": "post-user-\\n",
    "p3": "<|im_start|>",
    "p4": "assistant",
    "p5": "final-\\n (= #463 T-1 read)",
}
TRAINING_PROBE_SEED = 0
OUTPUT_BASE_BETLEY = PROJECT_ROOT / "eval_results" / "issue468" / "predictor_cossim_variants"
OUTPUT_BASE_TRAINING = (
    PROJECT_ROOT / "eval_results" / "issue468" / "predictor_cossim_variants_training"
)
V0_DIAGNOSTIC_DIR = PROJECT_ROOT / "eval_results" / "issue468"


# ── Prompt index helpers ───────────────────────────────────────────────────


def find_user_content_index(tokenizer, prompt_ids: torch.Tensor) -> int:
    """Return position of the LAST user-content token (V5 ``p0`` = V1 read).

    For Qwen-2.5-7B-Instruct chat template with one system + one user msg
    + ``add_generation_prompt=True``, the trailing 5 tokens are
    ``<|im_end|>\\n<|im_start|>assistant\\n``. ``last_content_index =
    positions[-1] - 1`` where ``positions = (full_ids == im_end_id)`` —
    the user-close ``<|im_end|>`` is the SECOND occurrence of id 151645.
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id < 0:
        raise RuntimeError("Tokenizer has no '<|im_end|>' token; cannot anchor V1/V5 indices.")
    ids = prompt_ids[0].tolist()
    positions = [i for i, x in enumerate(ids) if x == im_end_id]
    if len(positions) < 2:
        raise RuntimeError(
            f"Expected ≥2 occurrences of <|im_end|> (system-close + "
            f"user-close); found {len(positions)}. Prompt may be malformed."
        )
    return positions[-1] - 1


def position_sweep_indices(prompt_ids: torch.Tensor, last_content_index: int) -> dict[str, int]:
    """Return ``{p0..p5}`` index dict; asserts every index is in-range."""
    T = int(prompt_ids.shape[1])
    out: dict[str, int] = {}
    for off, name in enumerate(POSITION_NAMES):
        idx = last_content_index + off
        if idx < 0 or idx >= T:
            raise RuntimeError(
                f"Position-sweep index {name}={idx} out of range for prompt of length {T}."
            )
        out[name] = idx
    return out


# ── Variant extractors ─────────────────────────────────────────────────────


@torch.no_grad()
def _extract_v1_and_position_sweep(
    model,
    tokenizer,
    system_prompt: str,
    probes: list[str],
    layers: list[int],
    record_position_sweep: bool,
) -> tuple[dict[int, torch.Tensor], dict[str, dict[int, torch.Tensor]], dict[str, int] | None]:
    """One prompt forward per probe; produces V1 (= ``p0``) AND, if
    ``record_position_sweep=True``, V5 ``p0..p5`` from the SAME captures.

    Returns ``(v1_per_layer, position_sweep_per_layer, sweep_indices_first)``
    where ``v1_per_layer = {layer: (N_probes, hidden) fp32 cpu}`` and
    ``position_sweep_per_layer = {position_name: {layer: (N_probes, hidden)}}``.
    """
    captures: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    hooks = _attach_layer_hooks(model, layers, captures)
    try:
        v1_per_layer: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
        sweep_per_layer: dict[str, dict[int, list[torch.Tensor]]] = {
            name: {li: [] for li in layers} for name in POSITION_NAMES
        }
        sweep_indices_first: dict[str, int] | None = None
        for q in probes:
            prompt_ids = _build_prompt_ids(tokenizer, system_prompt, q).to(model.device)
            for li in layers:
                captures[li].clear()
            _ = model(prompt_ids)
            last_content_index = find_user_content_index(tokenizer, prompt_ids)
            if record_position_sweep:
                sweep_idx = position_sweep_indices(prompt_ids, last_content_index)
                if sweep_indices_first is None:
                    sweep_indices_first = sweep_idx
            for li in layers:
                hs = captures[li][-1]  # (1, T, hidden)
                v1_per_layer[li].append(hs[0, last_content_index, :].float().cpu())
                if record_position_sweep:
                    for name, idx in sweep_idx.items():
                        sweep_per_layer[name][li].append(hs[0, idx, :].float().cpu())
        out_v1 = {li: torch.stack(v1_per_layer[li]) for li in layers}
        out_sweep: dict[str, dict[int, torch.Tensor]] = {}
        if record_position_sweep:
            for name in POSITION_NAMES:
                out_sweep[name] = {li: torch.stack(sweep_per_layer[name][li]) for li in layers}
        return out_v1, out_sweep, sweep_indices_first
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def _sample_one_response(
    model, tokenizer, prompt_ids: torch.Tensor, max_new_tokens: int
) -> torch.Tensor:
    """Generate one response under the current torch seed, trim EOS, return
    ``(T_response,)`` tensor of response token ids (may be empty).
    """
    prompt_len = int(prompt_ids.shape[1])
    gen = model.generate(
        prompt_ids,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
        if tokenizer.pad_token_id is None
        else tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    resp_ids = gen[0, prompt_len:].detach()
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        eos_pos = (resp_ids == eos_id).nonzero(as_tuple=False).flatten()
        if eos_pos.numel() > 0:
            resp_ids = resp_ids[: int(eos_pos[0].item())]
    return resp_ids


def _consume_response_slice(
    captures_layer_hs: torch.Tensor,
    prompt_len: int,
    T_response: int,
    skip_k_values: list[int],
    want_last_token: bool,
    want_max: bool,
) -> tuple[dict[int, torch.Tensor], torch.Tensor | None, torch.Tensor | None, dict[int, bool]]:
    """For ONE layer's hook output ``(1, T_full, hidden)``, return per-k V3
    mean (with empty-slice fallback to last prompt token), V2 last
    response token, V4 per-dim max, and a per-k ``fallback_fired`` map.
    """
    hs = captures_layer_hs  # (1, T_full, hidden)
    start, end = prompt_len, prompt_len + T_response
    slice_ = hs[0, start:end, :].float().cpu()  # (T_response, hidden)
    fallback_vec = hs[0, prompt_len - 1, :].float().cpu()
    v3_by_k: dict[int, torch.Tensor] = {}
    fallback_fired: dict[int, bool] = {}
    for k in skip_k_values:
        if T_response <= k:
            v3_by_k[k] = fallback_vec
            fallback_fired[k] = True
        else:
            v3_by_k[k] = slice_[k:].mean(dim=0)
            fallback_fired[k] = False
    v2_vec = slice_[-1] if want_last_token else None
    v4_vec = slice_.max(dim=0).values if want_max else None
    return v3_by_k, v2_vec, v4_vec, fallback_fired


@torch.no_grad()
def _extract_response_variants(
    model,
    tokenizer,
    system_prompt: str,
    probes: list[str],
    layers: list[int],
    max_new_tokens: int,
    skip_k_values: list[int],
    want_last_token: bool,
    want_max: bool,
) -> tuple[
    dict[int, dict[int, torch.Tensor]],  # response_mean_skip_k[k][layer] (N_probes, hidden)
    dict[int, torch.Tensor] | None,  # V2 last_response_token per layer
    dict[int, torch.Tensor] | None,  # V4 response_max per layer
    dict[str, float],
]:
    """Generate one response per probe (sampling pinned by caller's torch
    seed) then teacher-force the full sequence once. Read the hook
    captures at the response slice for V3 (mean over [k:]), V2 (last
    response token), V4 (per-dim max over response positions). Empty-
    response fallback (rare with max_new_tokens=128) reads the last
    prompt token.

    Returns ``(v3_by_k, v2_or_none, v4_or_none, fallback_stats)``.
    """
    captures: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    hooks = _attach_layer_hooks(model, layers, captures)
    try:
        v3_lists: dict[int, dict[int, list[torch.Tensor]]] = {
            k: {li: [] for li in layers} for k in skip_k_values
        }
        v2_lists: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
        v4_lists: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
        fallback_by_k = {k: 0 for k in skip_k_values}
        n_total = 0
        for q in probes:
            n_total += 1
            prompt_ids = _build_prompt_ids(tokenizer, system_prompt, q).to(model.device)
            prompt_len = int(prompt_ids.shape[1])
            for li in layers:
                captures[li].clear()
            resp_ids = _sample_one_response(model, tokenizer, prompt_ids, max_new_tokens)
            for li in layers:
                captures[li].clear()

            if resp_ids.numel() == 0:
                # Empty response: every variant falls back to the last
                # prompt-token residual (matches #463 fallback for V3 k=0).
                _ = model(prompt_ids)
                pos = prompt_len - 1
                for li in layers:
                    vec = captures[li][-1][0, pos, :].float().cpu()
                    for k in skip_k_values:
                        v3_lists[k][li].append(vec)
                        fallback_by_k[k] += 1
                    if want_last_token:
                        v2_lists[li].append(vec)
                    if want_max:
                        v4_lists[li].append(vec)
                continue

            T_response = int(resp_ids.shape[0])
            full_ids = torch.cat([prompt_ids, resp_ids.unsqueeze(0).to(model.device)], dim=1)
            _ = model(full_ids)
            for li in layers:
                v3_by_k, v2_vec, v4_vec, fb_fired = _consume_response_slice(
                    captures[li][-1],
                    prompt_len,
                    T_response,
                    skip_k_values,
                    want_last_token,
                    want_max,
                )
                for k, vec in v3_by_k.items():
                    v3_lists[k][li].append(vec)
                    if fb_fired[k]:
                        fallback_by_k[k] += 1
                if v2_vec is not None:
                    v2_lists[li].append(v2_vec)
                if v4_vec is not None:
                    v4_lists[li].append(v4_vec)
        v3_out: dict[int, dict[int, torch.Tensor]] = {
            k: {li: torch.stack(v3_lists[k][li]) for li in layers} for k in skip_k_values
        }
        v2_out = {li: torch.stack(v2_lists[li]) for li in layers} if want_last_token else None
        v4_out = {li: torch.stack(v4_lists[li]) for li in layers} if want_max else None
        n = max(n_total, 1)
        fallback_stats = {f"v3_fallback_fraction_k{k}": fallback_by_k[k] / n for k in skip_k_values}
        fallback_stats["n_total_probes"] = float(n_total)
        return v3_out, v2_out, v4_out, fallback_stats
    finally:
        for h in hooks:
            h.remove()


# ── V0 chat-template diagnostic ────────────────────────────────────────────


@torch.no_grad()
def run_v0_diagnostic(
    model,
    tokenizer,
    pair: str,
    flavor: str,
    s_narrow: str,
    q: str,
    layers: list[int],
    out_path: Path,
) -> dict:
    """One-shot: log per-position residual norm + decoded token at every
    prompt position. Confirms the trailing 5-token sequence and anchors
    V1's index computation. Writes a JSON file with the table.
    """
    captures: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    hooks = _attach_layer_hooks(model, layers, captures)
    try:
        prompt_ids = _build_prompt_ids(tokenizer, s_narrow, q).to(model.device)
        for li in layers:
            captures[li].clear()
        _ = model(prompt_ids)
        ids = prompt_ids[0].tolist()
        T = len(ids)
        last_content_index = find_user_content_index(tokenizer, prompt_ids)
        sweep_idx = position_sweep_indices(prompt_ids, last_content_index)

        # Decode tokens at every position (truncate to last 16 for table).
        positions = []
        start_pos = max(0, T - 16)
        for pos in range(start_pos, T):
            tok = tokenizer.convert_ids_to_tokens(ids[pos])
            row = {
                "position": pos,
                "token_id": ids[pos],
                "token_repr": tok,
                "residual_norm_by_layer": {
                    str(li): float(captures[li][-1][0, pos, :].float().norm().item())
                    for li in layers
                },
            }
            positions.append(row)

        decoded_at_sweep = {
            name: {
                "index": idx,
                "token_id": ids[idx],
                "token_repr": tokenizer.convert_ids_to_tokens(ids[idx]),
                "description": POSITION_DESCRIPTIONS[name],
            }
            for name, idx in sweep_idx.items()
        }

        record = {
            "pair": pair,
            "flavor": flavor,
            "probe": q,
            "T_prompt": T,
            "last_content_index": last_content_index,
            "position_sweep_indices": sweep_idx,
            "decoded_at_sweep_positions": decoded_at_sweep,
            "trailing_positions_table": positions,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        logger.info("Wrote V0 diagnostic %s", out_path.relative_to(PROJECT_ROOT))
        logger.info(
            "V0 sweep decoded: %s",
            {n: decoded_at_sweep[n]["token_repr"] for n in POSITION_NAMES},
        )
        return record
    finally:
        for h in hooks:
            h.remove()


# ── Pre-block token-embedding-bag covariate ────────────────────────────────


@torch.no_grad()
def lexical_token_embedding_bag_cos(model, tokenizer, s_narrow: str, s_broad: str) -> float:
    """Cosine between mean pre-block token embeddings of S_narrow vs S_broad.

    Pure embedding lookup via ``model.model.embed_tokens``; no transformer
    forward. Adds <1% wall-time per cell. Returns one scalar per (cell,
    flavor); cached on output JSON.
    """
    embed = model.model.embed_tokens
    ids_n = tokenizer(s_narrow, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids_b = tokenizer(s_broad, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids_n = ids_n.to(model.device)
    ids_b = ids_b.to(model.device)
    e_n = embed(ids_n)[0].float().mean(dim=0)  # (hidden,)
    e_b = embed(ids_b)[0].float().mean(dim=0)
    cos = torch.nn.functional.cosine_similarity(e_n.unsqueeze(0), e_b.unsqueeze(0), dim=-1)
    return float(cos.item())


# ── Pair × flavor measurement ───────────────────────────────────────────────


def _resolve_s_narrow(pair: str, flavor: str, training_rows: list[dict] | None, k: int) -> str:
    if flavor == "NL":
        return S_NARROW_NL[pair]
    if flavor == "lit":
        if training_rows is None:
            raise ValueError("flavor='lit' requires training_rows")
        return build_literal_attribute_system_prompt(training_rows, k=k)
    raise ValueError(f"unknown flavor: {flavor!r}")


def measure_pair_flavor_variants(
    model,
    tokenizer,
    pair: str,
    flavor: str,
    probes: list[str],
    layers: list[int],
    max_new_tokens: int,
    training_rows: list[dict] | None,
    k_lit: int,
    variants: list[str],
    skip_k_values: list[int],
    want_lexical_bag: bool,
) -> dict:
    """Compute per-extraction-variant × per-layer cosine for one (pair, flavor)."""
    s_narrow = _resolve_s_narrow(pair, flavor, training_rows, k_lit)
    s_broad = S_BROAD
    logger.info(
        "Measuring pair=%s flavor=%s variants=%s layers=%s n_probes=%d skip_k=%s",
        pair,
        flavor,
        variants,
        layers,
        len(probes),
        skip_k_values,
    )

    cos_by_extraction: dict[str, dict] = {}
    v0_sweep_indices: dict[str, int] | None = None

    want_v1 = "v1" in variants
    want_v5 = "v5" in variants
    want_v2 = "v2" in variants
    want_v3 = "v3" in variants
    want_v4 = "v4" in variants

    # ── V1 (+ V5 position sweep, riding free on the same forward pass) ──
    if want_v1 or want_v5:
        v1_n, sweep_n, idx_n = _extract_v1_and_position_sweep(
            model, tokenizer, s_narrow, probes, layers, record_position_sweep=want_v5
        )
        v1_b, sweep_b, idx_b = _extract_v1_and_position_sweep(
            model, tokenizer, s_broad, probes, layers, record_position_sweep=want_v5
        )
        if want_v1:
            cos_by_extraction["last_prompt_token_final_content"] = {
                str(li): val for li, val in _per_layer_cos_sim(v1_n, v1_b).items()
            }
        if want_v5:
            v5_block: dict[str, dict[str, float]] = {}
            for name in POSITION_NAMES:
                cos_per_layer = _per_layer_cos_sim(sweep_n[name], sweep_b[name])
                v5_block[name] = {str(li): val for li, val in cos_per_layer.items()}
            cos_by_extraction["position_sweep"] = v5_block
            # The narrow-prompted index dict is the canonical reference;
            # we assert match with the broad-prompted one since the
            # template length should be identical given fixed Q.
            if idx_n != idx_b:
                logger.warning(
                    "V5 position indices differ between S_narrow (%s) and S_broad (%s); "
                    "the prompt-template tail should be identical at fixed Q.",
                    idx_n,
                    idx_b,
                )
            v0_sweep_indices = idx_n
        del v1_n, v1_b, sweep_n, sweep_b
        torch.cuda.empty_cache()

    # ── G2 cross-check: recompute #463's last_prompt_token (= V5 p5) by
    # reading at T-1, in case we want it without enabling v5. When v5 IS
    # enabled, V5 ``p5`` already IS this number — but emit a top-level
    # `last_prompt_token` field by extracting from the sweep dict if
    # present, so downstream regress code finds the legacy key.
    if want_v5:
        cos_by_extraction["last_prompt_token"] = cos_by_extraction["position_sweep"]["p5"]

    # ── V2 / V3 / V4 share one generate+teacher-force pass per persona ──
    fallback_stats: dict[str, float] = {}
    if want_v2 or want_v3 or want_v4:
        # Ensure k=0 (= #463 response_mean recompute) AND requested k's
        # are computed in the SAME generate+teacher-force pass.
        k_list = sorted(set(skip_k_values) | {0})
        v3_n, v2_n, v4_n, fb_n = _extract_response_variants(
            model,
            tokenizer,
            s_narrow,
            probes,
            layers,
            max_new_tokens=max_new_tokens,
            skip_k_values=k_list,
            want_last_token=want_v2,
            want_max=want_v4,
        )
        v3_b, v2_b, v4_b, fb_b = _extract_response_variants(
            model,
            tokenizer,
            s_broad,
            probes,
            layers,
            max_new_tokens=max_new_tokens,
            skip_k_values=k_list,
            want_last_token=want_v2,
            want_max=want_v4,
        )
        # V3 per k.
        if want_v3:
            v3_by_k: dict[str, dict[str, float]] = {}
            for k in k_list:
                cos_k = _per_layer_cos_sim(v3_n[k], v3_b[k])
                v3_by_k[str(k)] = {str(li): val for li, val in cos_k.items()}
            cos_by_extraction["response_mean_skip_k"] = v3_by_k
            # Emit a top-level `response_mean` ALIAS (= k=0 recompute) for
            # G2 + analyzer consumption.
            cos_by_extraction["response_mean"] = v3_by_k.get("0", {})
        else:
            # If only V2/V4 requested, still emit response_mean (k=0)
            # alias from k_list inclusion.
            cos_k0 = _per_layer_cos_sim(v3_n[0], v3_b[0])
            cos_by_extraction["response_mean"] = {str(li): val for li, val in cos_k0.items()}
        if want_v2 and v2_n is not None and v2_b is not None:
            cos_by_extraction["last_response_token"] = {
                str(li): val for li, val in _per_layer_cos_sim(v2_n, v2_b).items()
            }
        if want_v4 and v4_n is not None and v4_b is not None:
            cos_by_extraction["response_max"] = {
                str(li): val for li, val in _per_layer_cos_sim(v4_n, v4_b).items()
            }
        fallback_stats = {
            **{f"narrow_{k}": v for k, v in fb_n.items()},
            **{f"broad_{k}": v for k, v in fb_b.items()},
        }
        del v3_n, v3_b
        if v2_n is not None:
            del v2_n, v2_b
        if v4_n is not None:
            del v4_n, v4_b
        torch.cuda.empty_cache()

    # ── Pre-block token-embedding-bag covariate ──
    lexical_cos: float | None = None
    if want_lexical_bag:
        lexical_cos = lexical_token_embedding_bag_cos(model, tokenizer, s_narrow, s_broad)
        logger.info(
            "pair=%s flavor=%s lexical_token_embedding_bag_cos=%.4f",
            pair,
            flavor,
            lexical_cos,
        )

    # L0 post-block alias (matches #463 ``last_prompt_token[0]`` so the
    # analyzer reads it from the new JSON without falling back to #463).
    l0_post_block_cos_by_layer = (
        {
            str(li): cos_by_extraction.get("last_prompt_token_final_content", {}).get(str(li))
            for li in layers
        }
        if want_v1
        else {}
    )

    return {
        "pair": pair,
        "flavor": flavor,
        "s_narrow_preview": s_narrow[:400],
        "s_narrow_char_len": len(s_narrow),
        "s_broad": s_broad,
        "n_probes": len(probes),
        "layers": list(layers),
        "max_new_tokens": max_new_tokens,
        "K_literal_attribute": k_lit if flavor == "lit" else None,
        "skip_k_primary": (skip_k_values[0] if skip_k_values else None),
        "skip_k_values_reported": skip_k_values,
        "variants": variants,
        "cos_by_extraction": cos_by_extraction,
        "lexical_token_embedding_bag_cos": lexical_cos,
        "L0_post_block_cos_by_layer": l0_post_block_cos_by_layer,
        "position_sweep_decoded_indices": v0_sweep_indices,
        "v3_fallback_stats": fallback_stats,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-probes", type=int, default=DEFAULT_N_PROBES)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--k",
        type=int,
        default=LITERAL_ATTRIBUTE_K,
        help="K for literal-attribute (lit) in-context demos (= 8).",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=PAIRS,
        choices=PAIRS,
        help="Subset of pairs to measure (default: all PAIRS).",
    )
    parser.add_argument(
        "--flavors",
        nargs="+",
        default=["NL", "lit"],
        choices=["NL", "lit"],
        help="Subset of S_narrow flavors to measure.",
    )
    parser.add_argument(
        "--probe-source",
        default="training",
        choices=["betley", "training"],
        help="Which probe set to use; #468 headline lives at training.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["v1", "v2", "v3", "v4", "v5"],
        choices=list(ALL_VARIANTS),
        help="Which extraction variants to compute. v5 rides free on v1.",
    )
    parser.add_argument(
        "--skip-k",
        type=int,
        default=DEFAULT_SKIP_K,
        help="V3 PRIMARY skip-k (default 8 — FIXED, not smoke-tuned).",
    )
    parser.add_argument(
        "--skip-k-sweep",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Exploratory V3 k-sweep over multiple k values (e.g. "
            "--skip-k-sweep 0 4 8 16). When set, OVERRIDES --skip-k and "
            "every k is reported in cos_by_extraction.response_mean_skip_k."
        ),
    )
    parser.add_argument(
        "--lexical-bag",
        action="store_true",
        help="Compute the pre-block token-embedding-bag cosine covariate.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch seed for reproducible response sampling (default: 0).",
    )
    parser.add_argument(
        "--out-base",
        default=None,
        help="Override output base dir (defaults split by --probe-source).",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.out_base is not None:
        output_base = Path(args.out_base)
    else:
        output_base = (
            OUTPUT_BASE_TRAINING if args.probe_source == "training" else OUTPUT_BASE_BETLEY
        )
    output_base.mkdir(parents=True, exist_ok=True)

    # Probe-source resolution.
    betley_probes: list[str] | None = None
    if args.probe_source == "betley":
        main8 = set(fetch_betley_main_8())
        betley_probes = fetch_preregistered_probes(n=args.n_probes, exclude=main8)
        logger.info(
            "Loaded %d preregistered Betley probes (disjoint from main 8)",
            len(betley_probes),
        )

    need_datasets = ("lit" in args.flavors) or (args.probe_source == "training")
    pair_training_rows: dict[str, list[dict]] = {}
    if need_datasets:
        for pair in args.pairs:
            try:
                dataset_path = ensure_dataset(pair)
                pair_training_rows[pair] = load_jsonl(dataset_path)
                logger.info(
                    "pair=%s training rows=%d (dataset=%s)",
                    pair,
                    len(pair_training_rows[pair]),
                    dataset_path.name,
                )
            except FileNotFoundError as e:
                logger.warning("Dataset for pair=%s missing; affected flavors skipped: %s", pair, e)
                pair_training_rows[pair] = []

    device = torch.device("cuda:0")
    logger.info("Loading model %s on GPU %d", args.model, args.gpu_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    n_model_layers = len(model.model.layers)
    bad = [li for li in args.layers if li < 0 or li >= n_model_layers]
    if bad:
        raise RuntimeError(f"Requested layers {bad} out of range for {n_model_layers}-layer model")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Python `random` seeded too for any subordinate utilities.
    random.seed(args.seed)

    # ── V0 diagnostic (logged once per cell-flavor, fast — runs only when
    # 'v0' is explicitly in --variants) ──
    if "v0" in args.variants:
        # Pick the first (pair, flavor) that has data.
        diag_pair = args.pairs[0]
        diag_flavor = args.flavors[0]
        rows = pair_training_rows.get(diag_pair, [])
        s_narrow_diag = _resolve_s_narrow(
            diag_pair,
            diag_flavor,
            rows[:LIT_FLAVOR_N_ROWS] if diag_flavor == "lit" else None,
            args.k,
        )
        diag_q = "Write a script to fetch and process New Relic alerts from a webhook URL."
        diag_out = V0_DIAGNOSTIC_DIR / f"v0_diagnostic_{diag_pair}_{diag_flavor}.json"
        run_v0_diagnostic(
            model,
            tokenizer,
            diag_pair,
            diag_flavor,
            s_narrow_diag,
            diag_q,
            args.layers,
            diag_out,
        )

    # ── Main sweep ──
    skip_k_values = args.skip_k_sweep if args.skip_k_sweep else [args.skip_k]
    for pair in args.pairs:
        if args.probe_source == "training":
            rows_for_probes = pair_training_rows.get(pair, [])
            if not rows_for_probes:
                logger.warning(
                    "Skipping pair=%s entirely (probe-source=training, no rows on disk)",
                    pair,
                )
                continue
            cell_probes = extract_training_probes(
                rows_for_probes, n_probes=args.n_probes, k_lit_skip=args.k
            )
            logger.info(
                "pair=%s training-source probes: %d unique (sampled from rows[%d:])",
                pair,
                len(cell_probes),
                args.k,
            )
        else:
            assert betley_probes is not None
            cell_probes = betley_probes

        for flavor in args.flavors:
            if flavor == "lit" and not pair_training_rows.get(pair):
                logger.info("Skipping pair=%s flavor=lit (no training rows)", pair)
                continue
            training_rows = pair_training_rows.get(pair, [])
            rows_subset = training_rows[:LIT_FLAVOR_N_ROWS] if flavor == "lit" else None
            out_path = output_base / f"{pair}_{flavor}.json"
            result = measure_pair_flavor_variants(
                model,
                tokenizer,
                pair,
                flavor,
                cell_probes,
                args.layers,
                max_new_tokens=args.max_new_tokens,
                training_rows=rows_subset,
                k_lit=args.k,
                variants=args.variants,
                skip_k_values=skip_k_values,
                want_lexical_bag=args.lexical_bag,
            )
            result["probe_source"] = args.probe_source
            result["model"] = args.model
            result["metadata"] = reproducibility_metadata(
                {
                    "script": "issue468_predictor_cossim_variants",
                    "torch_seed": args.seed,
                    "probe_source": args.probe_source,
                    "model": args.model,
                    "variants": args.variants,
                    "skip_k_values": skip_k_values,
                }
            )
            # Per-(cell, flavor) checkpoint — write as soon as this cell
            # completes so a later cell crash doesn't lose prior work.
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            ce = result["cos_by_extraction"]
            logger.info(
                "Wrote %s; V1@L%d=%.4f V5_p5@L%d=%.4f resp_mean_k0@L%d=%.4f",
                out_path.relative_to(PROJECT_ROOT),
                args.layers[-1],
                ce.get("last_prompt_token_final_content", {}).get(
                    str(args.layers[-1]), float("nan")
                ),
                args.layers[-1],
                ce.get("position_sweep", {}).get("p5", {}).get(str(args.layers[-1]), float("nan"))
                if "position_sweep" in ce
                else float("nan"),
                args.layers[-1],
                ce.get("response_mean", {}).get(str(args.layers[-1]), float("nan")),
            )

    logger.info(
        "Predictor cossim variants done (probe-source=%s). Outputs in %s",
        args.probe_source,
        output_base,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
