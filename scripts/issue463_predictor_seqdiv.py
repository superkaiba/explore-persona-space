#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF003
# Intentional Unicode (ρ, ×, σ, →, —, ≥, ‖) in scientific docstrings + logs.
"""Issue #463 — Rao-Blackwellized sequence-level KL / JS divergence predictor.

Canonical definition (CLAUDE.md "Persona-distance metrics"): for each
probe ``Q`` (preregistered Betley paraphrases, disjoint from the eval
set; via ``issue404_common.fetch_preregistered_probes``), sample R
responses (temp=1, ≤max_new_tokens) from the ``S_narrow``-prompted base
model AND R from the ``S_broad``-prompted base model. Then estimate
divergence with the Rao-Blackwellized sequence-level estimator
(Zhang/Amini/Vieira/Cotterell 2025, *Better Estimation of the KL
Divergence Between Language Models*, arXiv 2504.10637): teacher-force
each sampled response through BOTH conditioned models and, at EVERY
response token position, compute the EXACT full-vocabulary divergence
between the two next-token distributions; then average over positions
(length-normalized, per-token), over R samples, and over probes.

Headlines per (pair, flavor):

* ``KL_narrow_broad`` — ``KL(narrow ‖ broad)``, sample responses from
  narrow; length-normalized average of per-position exact-vocab KL.
* ``KL_broad_narrow`` — ``KL(broad ‖ narrow)``, sample from broad.
* ``symKL`` — ``½ (KL_narrow_broad + KL_broad_narrow)``.
* ``JS`` — base-2 Jensen-Shannon (bounded [0, 1]); per-position
  ``m = ½ (p_narrow + p_broad)``, pooled over responses sampled from
  BOTH personas; length-normalized average over positions / responses /
  probes.
* ``M_js = 1 - JS`` and ``M_symkl = exp(-symKL)`` — polarity-aligned
  similarities (higher = closer) so they swap into the regression script
  next to ``M_cosine`` without sign flips.

This script uses HF Transformers ONLY (no vLLM). Generation AND
teacher-forcing share the same loaded model so we avoid the
vLLM → HF in-process teardown OOM gotcha (CLAUDE.md "vLLM in-process
teardown does NOT reap worker subprocesses"). One response at a time
through the per-position vocab reduction so we never hold the
``(R, T, V)`` tensor in memory (``V ≈ 151_936`` for Qwen-2.5-7B).

Output (per-cell checkpoint, written immediately each cell):

    eval_results/issue463/predictor_seqdiv/<pair>_<flavor>.json

Usage::

    uv run python scripts/issue463_predictor_seqdiv.py
    uv run python scripts/issue463_predictor_seqdiv.py --pairs insecure_code \
        --flavors NL --n-probes 4 --samples-per-probe 2 --max-new-tokens 32
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
    extract_user_assistant,
    fetch_betley_main_8,
    fetch_preregistered_probes,
    load_jsonl,
    load_strong_nl_dict,
    reproducibility_metadata,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue463_predictor_seqdiv")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_N_PROBES = 48
DEFAULT_SAMPLES_PER_PROBE = 4
DEFAULT_MAX_NEW_TOKENS = 128
TRAINING_PROBE_SEED = 0  # fixed RNG for per-cell training-probe subsample
OUTPUT_BASE_BETLEY = PROJECT_ROOT / "eval_results" / "issue463" / "predictor_seqdiv"
OUTPUT_BASE_TRAINING = PROJECT_ROOT / "eval_results" / "issue463" / "predictor_seqdiv_training"
# Issue #467: R=16 headline JS re-runs land in a disjoint dir so the on-disk
# #463 R=4 weak-NL JSONs are never overwritten. The filename suffix
# (`<pair>_<flavor>{_strong}.json`) carries the nl_variant when relevant.
# Plan §0.7 RF3b GLOBAL OVERRIDE: headline rows use R=16; the regress loader
# fail-louds on samples_per_probe != 16. Output dir encodes the R so a future
# R sweep produces a disjoint dir.
OUTPUT_BASE_ISSUE467_TRAINING = PROJECT_ROOT / "eval_results" / "issue467" / "predictor_seqdiv_R16"
OUTPUT_BASE_ISSUE467_BETLEY = (
    PROJECT_ROOT / "eval_results" / "issue467" / "predictor_seqdiv_R16_betley"
)


# ── Divergence reductions (numerical-stability critical) ────────────────────


def _kl_position_exact(
    logits_p: torch.Tensor, logits_q: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Exact full-vocab ``KL(p ‖ q)`` at a single position. fp32.

    ``logits_p``, ``logits_q``: ``(V,)`` tensors. Returns a scalar tensor.

    Uses log-softmax for numerical stability (log of softmax is the canonical
    log-prob form; subtracting log-softmaxes is the log-ratio with no
    division). ``p`` is recovered by ``exp(log_p)`` BEFORE the reduction so
    the sum-to-1 invariant holds in fp32; the tiny eps is a defensive floor
    in case rounding produces a negative entry — but log-softmax math here
    cannot produce non-finite values.
    """
    log_p = torch.log_softmax(logits_p.float(), dim=-1)
    log_q = torch.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp().clamp_min(eps)
    return (p * (log_p - log_q)).sum(dim=-1)


def _js_position_exact(
    logits_p: torch.Tensor, logits_q: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Exact full-vocab base-2 JS at a single position. fp32.

    ``logits_p``, ``logits_q``: ``(V,)``. Returns a scalar in ``[0, 1]``.
    """
    log_p = torch.log_softmax(logits_p.float(), dim=-1)
    log_q = torch.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp().clamp_min(eps)
    q = log_q.exp().clamp_min(eps)
    m = 0.5 * (p + q)
    log_m = m.log()
    # KL(p‖m) and KL(q‖m), nats.
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    # JS in nats → JS base 2 by /ln(2). Result is bounded [0, 1].
    js_nats = 0.5 * (kl_pm + kl_qm)
    js_base2 = js_nats / torch.log(torch.tensor(2.0, dtype=torch.float32))
    return js_base2.clamp(min=0.0, max=1.0)


# ── Sample + teacher-force a single response, reduce per-position ───────────


def _build_prompt_ids(tokenizer, system_prompt: str, q: str) -> torch.Tensor:
    """Return the ``(1, T_prompt)`` token IDs for the chat-template prompt
    with ``add_generation_prompt=True``.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": q},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", padding=False, add_special_tokens=False)
    return enc["input_ids"]


@torch.no_grad()
def _sample_responses(
    model,
    tokenizer,
    system_prompt: str,
    q: str,
    n_samples: int,
    max_new_tokens: int,
) -> list[torch.Tensor]:
    """Sample ``n_samples`` responses (temp=1.0) under ``system_prompt`` for
    probe ``q``. Returns a list of 1-D ``LongTensor`` of RESPONSE token IDs
    (the prompt prefix is stripped). Each response is at most
    ``max_new_tokens`` tokens long, terminated at EOS when emitted.
    """
    prompt_ids = _build_prompt_ids(tokenizer, system_prompt, q).to(model.device)
    prompt_len = prompt_ids.shape[1]
    # Repeat across the sample dim — one forward batch.
    batch = prompt_ids.repeat(n_samples, 1)
    out = model.generate(
        batch,
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
    # ``out`` is (n_samples, prompt_len + new_T). Strip the prompt prefix
    # (deterministic across rows; greedy/sampling never edits prompt tokens).
    responses: list[torch.Tensor] = []
    for i in range(n_samples):
        resp = out[i, prompt_len:].detach().clone()
        # Truncate at the FIRST eos token if present (HF generate already
        # stopped on eos but pads the rest of the batch with pad/eos to the
        # max length in the batch — drop everything from the first eos so the
        # teacher-force reduction only sees real response tokens).
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            eos_pos = (resp == eos_id).nonzero(as_tuple=False).flatten()
            if eos_pos.numel() > 0:
                cut = int(eos_pos[0].item())
                resp = resp[:cut]
        responses.append(resp.cpu())
    return responses


@torch.no_grad()
def _teacher_force_response_logits(
    model,
    tokenizer,
    system_prompt: str,
    q: str,
    response_ids: torch.Tensor,
) -> tuple[torch.Tensor | None, int]:
    """Teacher-force ``[prompt(system,q), response]`` and return the slice of
    logits that predict each RESPONSE token, plus the prompt length.

    Returns ``(response_logits, prompt_len)`` where ``response_logits`` is
    ``(T_response, V)`` on the model device, or ``(None, prompt_len)`` if
    ``response_ids`` is empty.

    The slot at position ``prompt_len - 1`` in the full input predicts the
    FIRST response token, so the logits to read run from
    ``[prompt_len - 1 : prompt_len - 1 + T_response]``.
    """
    if response_ids.numel() == 0:
        prompt_ids = _build_prompt_ids(tokenizer, system_prompt, q).to(model.device)
        return None, int(prompt_ids.shape[1])

    prompt_ids = _build_prompt_ids(tokenizer, system_prompt, q).to(model.device)
    prompt_len = prompt_ids.shape[1]
    resp_on_dev = response_ids.to(model.device).unsqueeze(0)  # (1, T_response)
    full_ids = torch.cat([prompt_ids, resp_on_dev], dim=1)  # (1, prompt_len + T_response)
    out = model(full_ids)
    logits = out.logits[0]  # (T_full, V)
    # Predict the t-th response token from the (prompt_len + t - 1)-th slot.
    start = prompt_len - 1
    end = start + resp_on_dev.shape[1]
    response_logits = logits[start:end, :]
    return response_logits, prompt_len


# ── Pair × flavor measurement ───────────────────────────────────────────────


def _resolve_s_narrow(
    pair: str,
    flavor: str,
    training_rows: list[dict] | None,
    k: int,
    nl_variant: str = "weak",
    strong_nl_dict: dict[str, str] | None = None,
) -> str:
    """Return the S_narrow system prompt for ``(pair, flavor, nl_variant)``.

    Mirrors ``issue463_predictor_cossim._resolve_s_narrow`` so the two
    scripts stay in lockstep. Issue #467 §4.3 additive diff.
    """
    if flavor == "NL":
        if nl_variant == "strong":
            if strong_nl_dict is None or pair not in strong_nl_dict:
                raise RuntimeError(
                    f"--nl-variant strong but no PASS strong-NL prompt for pair={pair!r}. "
                    "Run scripts/issue467_author_strong_nl.py for this cell first."
                )
            return strong_nl_dict[pair]
        return S_NARROW_NL[pair]
    if flavor == "lit":
        if training_rows is None:
            raise ValueError("flavor='lit' requires training_rows")
        return build_literal_attribute_system_prompt(training_rows, k=k)
    raise ValueError(f"unknown flavor: {flavor!r}")


def _reduce_one_direction(
    model,
    tokenizer,
    s_sampling: str,
    s_other: str,
    q: str,
    n_samples: int,
    max_new_tokens: int,
    js_acc: dict,
) -> dict:
    """Sample R responses under ``s_sampling``, teacher-force through BOTH
    sides, and Rao-Blackwell average per-position EXACT KL of the SAMPLING
    persona against the OTHER persona over the response.

    Returns ``{kl_sum_per_token, n_positions, n_samples_used}`` for the
    direction ``KL(s_sampling ‖ s_other)``. Also appends per-position
    ``(logits_sampling, logits_other, n_positions)`` triplets into ``js_acc``
    so the caller can do the symmetric JS reduction over the pooled set
    of responses from BOTH directions.

    We do NOT hold ``(R, T, V)`` tensors in memory: each sample is reduced
    to a scalar per position immediately and only the per-position
    log-probs needed for JS are accumulated as logits-on-CPU
    (``(T_i, V)`` fp32 each, freed at end of probe).
    """
    kl_sum_total = 0.0  # nats, summed over sampled responses' positions
    n_positions_total = 0
    n_samples_used = 0

    responses = _sample_responses(
        model, tokenizer, s_sampling, q, n_samples=n_samples, max_new_tokens=max_new_tokens
    )
    for resp in responses:
        if resp.numel() == 0:
            # Empty response (EOS at position 0). Nothing to teacher-force.
            continue
        logits_sampling, _ = _teacher_force_response_logits(model, tokenizer, s_sampling, q, resp)
        logits_other, _ = _teacher_force_response_logits(model, tokenizer, s_other, q, resp)
        if logits_sampling is None or logits_other is None:
            continue
        assert logits_sampling.shape == logits_other.shape, (
            logits_sampling.shape,
            logits_other.shape,
        )
        T = logits_sampling.shape[0]

        # Per-position EXACT KL(sampling || other) summed over positions
        # (sum, NOT mean — we length-normalize AFTER pooling per-probe so
        # longer responses contribute proportionally to their length, then
        # length-normalize as `total_kl_nats / total_positions`).
        kl_pos_sum = 0.0
        for t in range(T):
            kl_t = _kl_position_exact(logits_sampling[t], logits_other[t])
            kl_pos_sum += float(kl_t.item())
        kl_sum_total += kl_pos_sum
        n_positions_total += T
        n_samples_used += 1

        # Keep CPU-fp32 copies for the JS pooled reduction.
        js_acc["logits_sampling_cpu"].append(logits_sampling.float().cpu())
        js_acc["logits_other_cpu"].append(logits_other.float().cpu())
        del logits_sampling, logits_other
        torch.cuda.empty_cache()

    return {
        "kl_sum_nats": kl_sum_total,
        "n_positions": n_positions_total,
        "n_samples_used": n_samples_used,
        "n_samples_requested": n_samples,
    }


def measure_pair_flavor(
    model,
    tokenizer,
    pair: str,
    flavor: str,
    probes: list[str],
    samples_per_probe: int,
    max_new_tokens: int,
    training_rows: list[dict] | None,
    k: int,
    nl_variant: str = "weak",
    strong_nl_dict: dict[str, str] | None = None,
) -> dict:
    """Compute Rao-Blackwellized KL / JS for one (pair, flavor) cell.

    Returns a result dict with both KL directions, symKL, JS, and
    polarity-aligned similarity headlines.
    """
    s_narrow = _resolve_s_narrow(
        pair,
        flavor,
        training_rows,
        k,
        nl_variant=nl_variant,
        strong_nl_dict=strong_nl_dict,
    )
    s_broad = S_BROAD

    logger.info(
        "Measuring pair=%s flavor=%s (S_narrow len=%d chars; %d probes × %d samples × "
        "max_new_tokens=%d)",
        pair,
        flavor,
        len(s_narrow),
        len(probes),
        samples_per_probe,
        max_new_tokens,
    )

    # Per-direction accumulators. Length normalization is done as
    # `total_kl_nats / total_positions` AFTER pooling all probes (so longer
    # responses contribute proportionally — see CLAUDE.md "length-normalized,
    # per-token").
    nb = {"kl_sum_nats": 0.0, "n_positions": 0, "n_samples_used": 0}
    bn = {"kl_sum_nats": 0.0, "n_positions": 0, "n_samples_used": 0}

    # JS reduction: pooled responses from BOTH personas. For each probe we
    # accumulate (logits_sampling, logits_other) pairs ON CPU; at the end of
    # the probe we compute the per-position JS over the pool and sum, freeing
    # the buffer to bound memory.
    js_sum_base2 = 0.0
    js_n_positions = 0

    n_empty_narrow = 0
    n_empty_broad = 0

    for qi, q in enumerate(probes):
        # Direction 1: sample from narrow, teacher-force both. JS pool gets
        # narrow's log-probs as p_a and broad's as p_b.
        js_acc_narrow = {"logits_sampling_cpu": [], "logits_other_cpu": []}
        res_nb = _reduce_one_direction(
            model,
            tokenizer,
            s_narrow,
            s_broad,
            q,
            n_samples=samples_per_probe,
            max_new_tokens=max_new_tokens,
            js_acc=js_acc_narrow,
        )
        nb["kl_sum_nats"] += res_nb["kl_sum_nats"]
        nb["n_positions"] += res_nb["n_positions"]
        nb["n_samples_used"] += res_nb["n_samples_used"]
        if res_nb["n_samples_used"] == 0:
            n_empty_narrow += 1

        # Direction 2: sample from broad, teacher-force both. JS pool gets
        # broad's log-probs as p_a and narrow's as p_b. The narrow / broad
        # ordering inside JS is symmetric, so we just feed (broad, narrow).
        js_acc_broad = {"logits_sampling_cpu": [], "logits_other_cpu": []}
        res_bn = _reduce_one_direction(
            model,
            tokenizer,
            s_broad,
            s_narrow,
            q,
            n_samples=samples_per_probe,
            max_new_tokens=max_new_tokens,
            js_acc=js_acc_broad,
        )
        bn["kl_sum_nats"] += res_bn["kl_sum_nats"]
        bn["n_positions"] += res_bn["n_positions"]
        bn["n_samples_used"] += res_bn["n_samples_used"]
        if res_bn["n_samples_used"] == 0:
            n_empty_broad += 1

        # JS over the pooled-responses set. For each sampled response (from
        # narrow or broad), compute per-position JS(p_narrow, p_broad) and
        # accumulate. Map naming: in js_acc_narrow, "logits_sampling" is
        # narrow and "logits_other" is broad; in js_acc_broad, "logits_sampling"
        # is broad and "logits_other" is narrow.
        for ls, lo in zip(
            js_acc_narrow["logits_sampling_cpu"],
            js_acc_narrow["logits_other_cpu"],
            strict=True,
        ):
            assert ls.shape == lo.shape, (ls.shape, lo.shape)
            for t in range(ls.shape[0]):
                # ls = narrow logits, lo = broad logits — pass to JS.
                js_t = _js_position_exact(ls[t], lo[t])
                js_sum_base2 += float(js_t.item())
                js_n_positions += 1
        for ls, lo in zip(
            js_acc_broad["logits_sampling_cpu"],
            js_acc_broad["logits_other_cpu"],
            strict=True,
        ):
            assert ls.shape == lo.shape, (ls.shape, lo.shape)
            for t in range(ls.shape[0]):
                # ls = broad logits, lo = narrow logits. JS is symmetric.
                js_t = _js_position_exact(ls[t], lo[t])
                js_sum_base2 += float(js_t.item())
                js_n_positions += 1
        # Free the per-probe CPU buffers immediately.
        del js_acc_narrow, js_acc_broad

        if (qi + 1) % 4 == 0 or qi == len(probes) - 1:
            cur_js = js_sum_base2 / max(js_n_positions, 1)
            cur_kl_nb = nb["kl_sum_nats"] / max(nb["n_positions"], 1)
            cur_kl_bn = bn["kl_sum_nats"] / max(bn["n_positions"], 1)
            logger.info(
                "  probe %d/%d  running JS=%.4f  KL(n‖b)=%.4f  KL(b‖n)=%.4f  pos_nb=%d pos_bn=%d",
                qi + 1,
                len(probes),
                cur_js,
                cur_kl_nb,
                cur_kl_bn,
                nb["n_positions"],
                bn["n_positions"],
            )

    if nb["n_positions"] == 0 or bn["n_positions"] == 0:
        raise RuntimeError(
            f"pair={pair} flavor={flavor}: zero response positions in at "
            f"least one direction (n_positions narrow→broad={nb['n_positions']}, "
            f"broad→narrow={bn['n_positions']}). All sampled responses were "
            f"empty — cannot estimate divergence."
        )

    kl_narrow_broad = nb["kl_sum_nats"] / nb["n_positions"]
    kl_broad_narrow = bn["kl_sum_nats"] / bn["n_positions"]
    sym_kl = 0.5 * (kl_narrow_broad + kl_broad_narrow)
    js = js_sum_base2 / js_n_positions

    import math

    m_js = 1.0 - js
    m_symkl = math.exp(-sym_kl)

    return {
        "pair": pair,
        "flavor": flavor,
        "nl_variant": nl_variant if flavor == "NL" else None,
        "s_narrow_preview": s_narrow[:400],
        "s_narrow_char_len": len(s_narrow),
        "s_broad": s_broad,
        "n_probes": len(probes),
        "samples_per_probe": samples_per_probe,
        "max_new_tokens": max_new_tokens,
        "K_literal_attribute": k if flavor == "lit" else None,
        "KL_narrow_broad": kl_narrow_broad,
        "KL_broad_narrow": kl_broad_narrow,
        "symKL": sym_kl,
        "JS": js,
        # Polarity-aligned similarities (higher = closer). Pair the regression
        # script's M_cosine / M_js polarity (no sign flip needed).
        "M_js": m_js,
        "M_symkl": m_symkl,
        "diagnostics": {
            "n_samples_used_narrow_broad": nb["n_samples_used"],
            "n_samples_used_broad_narrow": bn["n_samples_used"],
            "n_positions_narrow_broad": nb["n_positions"],
            "n_positions_broad_narrow": bn["n_positions"],
            "n_positions_js_pooled": js_n_positions,
            "n_probes_with_empty_narrow_responses": n_empty_narrow,
            "n_probes_with_empty_broad_responses": n_empty_broad,
            "mean_response_len_narrow": (nb["n_positions"] / max(nb["n_samples_used"], 1)),
            "mean_response_len_broad": (bn["n_positions"] / max(bn["n_samples_used"], 1)),
        },
    }


# ── Per-cell training-source probe extractor ───────────────────────────────


def extract_training_probes(
    training_rows: list[dict],
    n_probes: int,
    k_lit_skip: int,
    rng_seed: int = TRAINING_PROBE_SEED,
) -> list[str]:
    """Build a cell-specific probe set from a cell's own SFT training rows.

    Step 1: skip the first ``k_lit_skip`` rows (these feed the lit flavor's
    in-context examples; reusing them as probes would let the persona see
    the answer text inside its own context).
    Step 2: extract the USER turn from each remaining row via
    ``extract_user_assistant`` (drops rows that don't yield a user turn).
    Step 3: dedup while preserving order.
    Step 4: sample ``n_probes`` with a FIXED ``random.Random(rng_seed)`` so
    every NL/lit pair of cells sees the SAME probe list (the only thing
    that differs across NL vs lit is the persona, not the questions). If
    fewer than ``n_probes`` unique probes exist, use them all and log it.

    Fail-loud: raises ``RuntimeError`` if zero usable probes survive (means
    the cell's dataset has no user turns past the in-context offset).
    """
    if not training_rows:
        raise RuntimeError("training_rows is empty — cannot extract training probes")
    if k_lit_skip < 0:
        raise ValueError(f"k_lit_skip must be >= 0, got {k_lit_skip}")
    remaining = training_rows[k_lit_skip:]
    user_turns: list[str] = []
    seen: set[str] = set()
    # Seed `seen` with the in-context rows' user turns (C1): a probe must never
    # duplicate one of the lit persona's K in-context examples, even if the same
    # templated question recurs at a row index >= k_lit_skip.
    for ic_row in training_rows[:k_lit_skip]:
        ic_user, _ = extract_user_assistant(ic_row)
        if ic_user is not None and ic_user.strip():
            seen.add(ic_user.strip())
    for row in remaining:
        user, _ = extract_user_assistant(row)
        if user is None:
            continue
        u = user.strip()
        if not u or u in seen:
            continue
        seen.add(u)
        user_turns.append(u)
    if not user_turns:
        raise RuntimeError(
            f"After skipping {k_lit_skip} in-context rows + dedup, ZERO usable "
            f"user-turn probes remain from {len(training_rows)} training rows. "
            "Predictor cannot proceed for this cell."
        )
    if len(user_turns) <= n_probes:
        logger.info(
            "Training-probe pool has %d unique user turns (< requested %d); using all.",
            len(user_turns),
            n_probes,
        )
        return user_turns
    rng = random.Random(rng_seed)
    return rng.sample(user_turns, n_probes)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-probes", type=int, default=DEFAULT_N_PROBES)
    parser.add_argument("--samples-per-probe", type=int, default=DEFAULT_SAMPLES_PER_PROBE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--k", type=int, default=LITERAL_ATTRIBUTE_K)
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
        default="betley",
        choices=["betley", "training"],
        help=(
            "betley (default): Betley preregistered_evals.yaml probes shared "
            "across cells. training: per-cell USER turns sampled from that "
            "cell's own SFT training rows (skipping the first --k rows used "
            "by the lit persona's in-context examples)."
        ),
    )
    parser.add_argument(
        "--nl-variant",
        default="weak",
        choices=["weak", "strong"],
        help=(
            "Only consulted when --flavors includes NL. "
            "'weak' (default): per-cell one-line prompt from S_NARROW_NL — preserves "
            "the on-disk #463 code path. "
            "'strong' (issue #467): Claude-authored rich description from "
            "data/issue467/strong_nl/<cell>.json. Cells without a PASSed strong prompt "
            "raise."
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch seed for reproducible sampling (default: 0).",
    )
    parser.add_argument(
        "--issue467-output",
        action="store_true",
        help=(
            "Write outputs to eval_results/issue467/predictor_seqdiv_R16/ instead of the "
            "#463 dir. Use when running the R=16 headline JS replications (plan §0.7 RF3b) "
            "so the on-disk #463 R=4 JSONs are never overwritten. Pair with "
            "--samples-per-probe 16 on production runs."
        ),
    )
    args = parser.parse_args()

    # Bind CUDA_VISIBLE_DEVICES BEFORE any cuda allocation — mirrors the
    # round-2 ISSUE 3 fix in issue404_predictor_cossim.py.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.issue467_output:
        output_base = (
            OUTPUT_BASE_ISSUE467_TRAINING
            if args.probe_source == "training"
            else OUTPUT_BASE_ISSUE467_BETLEY
        )
    else:
        output_base = (
            OUTPUT_BASE_TRAINING if args.probe_source == "training" else OUTPUT_BASE_BETLEY
        )
    output_base.mkdir(parents=True, exist_ok=True)

    # Issue #467: load strong-NL prompts once if requested.
    strong_nl_dict: dict[str, str] | None = None
    if args.nl_variant == "strong" and "NL" in args.flavors:
        strong_nl_dict = load_strong_nl_dict(pairs=args.pairs)
        missing = [p for p in args.pairs if p not in strong_nl_dict]
        if missing:
            raise RuntimeError(
                f"--nl-variant strong but no PASS strong-NL prompt for pairs: {missing}. "
                "Run scripts/issue467_author_strong_nl.py for these cells first."
            )
        logger.info("Loaded %d PASS strong-NL prompts for #467", len(strong_nl_dict))

    # Betley-source probes are SHARED across cells (one fetch); training-source
    # probes are per-cell and built later inside the cell loop.
    betley_probes: list[str] | None = None
    if args.probe_source == "betley":
        main8 = set(fetch_betley_main_8())
        betley_probes = fetch_preregistered_probes(n=args.n_probes, exclude=main8)
        logger.info(
            "Loaded %d preregistered Betley probes (disjoint from Betley main 8)",
            len(betley_probes),
        )

    # Training datasets are needed by BOTH the lit flavor AND probe-source=training
    # (the latter for every flavor). Load once per pair.
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
                logger.warning(
                    "Dataset for pair=%s missing; affected flavors will be skipped: %s", pair, e
                )
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

    # Reproducible sampling — seed both CPU and CUDA RNGs.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    for pair in args.pairs:
        # Build cell-specific training probes ONCE per pair, used for BOTH
        # NL and lit flavors of that cell (only the persona differs).
        cell_probes: list[str]
        if args.probe_source == "training":
            rows_for_probes = pair_training_rows.get(pair, [])
            if not rows_for_probes:
                raise RuntimeError(
                    f"probe-source=training but no training rows on disk for pair={pair}. "
                    "Run issue458_prep_datasets.py for all 18 cells first (fail-fast — refusing "
                    "to silently drop a cell to n<18)."
                )
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
            # #467: strong-NL gets a `<pair>_NL_strong.json` filename so it
            # doesn't clobber any on-disk weak-NL JSON in OUTPUT_BASE_TRAINING.
            file_suffix = (
                "NL_strong" if (flavor == "NL" and args.nl_variant == "strong") else flavor
            )
            out_path = output_base / f"{pair}_{file_suffix}.json"
            # Per CLAUDE.md "Checkpoint per phase" — persist each cell as
            # soon as it completes, never accumulate-in-memory across pairs.
            result = measure_pair_flavor(
                model,
                tokenizer,
                pair,
                flavor,
                cell_probes,
                samples_per_probe=args.samples_per_probe,
                max_new_tokens=args.max_new_tokens,
                training_rows=rows_subset,
                k=args.k,
                nl_variant=args.nl_variant,
                strong_nl_dict=strong_nl_dict,
            )
            result["probe_source"] = args.probe_source
            result["metadata"] = reproducibility_metadata(
                {
                    "script": "issue463_predictor_seqdiv",
                    "torch_seed": args.seed,
                    "probe_source": args.probe_source,
                    "nl_variant": args.nl_variant,
                }
            )
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(
                "Wrote %s; JS=%.4f  symKL=%.4f  KL(n‖b)=%.4f  KL(b‖n)=%.4f",
                out_path.relative_to(PROJECT_ROOT),
                result["JS"],
                result["symKL"],
                result["KL_narrow_broad"],
                result["KL_broad_narrow"],
            )

    logger.info(
        "Predictor seqdiv done (probe-source=%s). Outputs in %s", args.probe_source, output_base
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
