# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, ′, ×, →, —) in docstrings/comments matching the project house style.
"""Slice-resolved predictors for task #466 (plan §4.2 script 3).

TWO phases on the UN-marker-trained Qwen-2.5-7B-Instruct (NO LoRA):

  Phase JS — Rao-Blackwellized sequence-level JS divergence
  ─────────────────────────────────────────────────────────
  Implements arXiv 2504.10637 (Amini/Vieira/Cotterell, "Better Estimation of
  the KL Divergence Between Language Models"). For each probe q:
    - Sample R=8 responses from (S, q) under vLLM (temp=1, top_p=1, max=256).
    - Sample R=8 responses from (S', q) under vLLM (same config).
    - Pool to 16 per probe (symmetric — sample from BOTH conditioned models).
    - Teacher-force each pooled response through BOTH conditioned models on
      HF Transformers (after vLLM teardown via _kill_vllm_workers from the
      ported issue-456 rig).
    - At every response-token position, compute the EXACT full-vocab JS +
      both KL directions over the next-token distributions p_S(·) and p_S'(·).
    - Length-normalize (mean across positions per response), then mean across
      pooled responses per probe.
    - Per slice: mean across 30 probes -> slice-mean JS / KL_S->S' / KL_S'->S.
    - Generic-averaged JS = mean across the 60-probe trigger + non-trigger
      union (the BLIND predictor — broadcast across slices).

  Phase Cosine — 3 extraction points, layer sweep {7, 14, 21, 27}
  ────────────────────────────────────────────────────────────────
  Implements Chen/Arditi/Sleight/Evans/Lindsey 2025 (arXiv 2507.21509,
  "Persona Vectors") difference-of-means recipe at three points:
    (a0) end-of-system-prompt — INPUT-INDEPENDENT, broadcast across slices,
         IS the boundary-blindness probe the Goal calls out.
    (a)  last token of {S, Q} — legacy #404 recipe, slice-dependent.
    (b)  mean over each model's own generated response tokens — the canonical
         Persona-Vectors recipe; reuses the Phase JS generation cache.

Per-cell JSON written to::
    eval_results/issue_466/predictors/{behavior}_{slice}.json

Per CLAUDE.md "Checkpoint per phase" — each cell's JSON is written the moment
the cell completes so a downstream crash doesn't lose earlier cells.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from issue466_personas import (  # noqa: E402
    PERSONAS,
    SLICE_NONTRIGGER,
    SLICE_TRIGGER_A,
    SLICE_TRIGGER_B,
)

logger = logging.getLogger("issue466_predictors")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER = "※"
MARKER_ID = 63680
DEFAULT_LAYERS = [7, 14, 21, 27]
HEADLINE_LAYER = 21

# The 4 (predictor pair, slice) cells the headline matched-contrast scatter
# consumes. Each cell carries the predictor pair (S, S') and the slice
# label that picks which probes to score.
PREDICTOR_PAIRS: dict[str, tuple[str, str]] = {
    "A_spanish_restaurants": ("S", "S_prime_A_spanish_restaurants"),
    "B_caps_sports": ("S", "S_prime_B_caps_sports"),
}

# Cells = (behavior, slice). Per behavior, the 'trigger' slice is the
# behavior-specific trigger; the 'nontrigger' slice is the shared
# non-trigger panel.
CELLS: list[tuple[str, str]] = [
    ("A_spanish_restaurants", "nontrigger"),
    ("A_spanish_restaurants", "trigger"),
    ("B_caps_sports", "nontrigger"),
    ("B_caps_sports", "trigger"),
]


def _trigger_for(behavior: str) -> list[str]:
    if behavior == "A_spanish_restaurants":
        return SLICE_TRIGGER_A
    if behavior == "B_caps_sports":
        return SLICE_TRIGGER_B
    raise ValueError(f"unknown behavior: {behavior!r}")


# ── Reproducibility metadata ───────────────────────────────────────────────


def _metadata() -> dict[str, Any]:
    git_commit = "unknown"
    try:
        # epm-lint: subprocess-env-inherit -- git rev-parse needs no credential env
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            git_commit = out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {
        "script": "issue466_predictors",
        "git_commit": git_commit,
        "base_model": BASE_MODEL,
        "ts_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


# ── vLLM teardown ──────────────────────────────────────────────────────────


def _kill_vllm_workers() -> None:
    """Reap vLLM TP/PP worker subprocesses, then FAIL LOUD if any GPU PID remains.

    Same logic as the ported ``eval_i456_onpolicy_emission._kill_vllm_workers``
    — duplicated here so this script can be invoked stand-alone without
    importing the larger eval rig. Per CLAUDE.md vLLM teardown gotcha:
    ``del llm + destroy_*`` is not enough; we also psutil-reap children
    and ``nvidia-smi``-probe so a surviving worker doesn't OOM the HF
    teacher-force phase.
    """
    import psutil

    try:
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        logger.info("destroy_* skipped (%s)", e)

    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()

    me = psutil.Process()
    children = me.children(recursive=True)
    for child in children:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.terminate()
    _gone, alive = psutil.wait_procs(children, timeout=10)
    for child in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    gc.collect()

    try:
        # epm-lint: subprocess-env-inherit -- nvidia-smi PID probe needs no credential env
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.info("nvidia-smi probe skipped (%s)", e)
        return
    my_pid = os.getpid()
    surviving: list[int] = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != my_pid and psutil.pid_exists(pid):
            surviving.append(pid)
    if surviving:
        raise RuntimeError(
            f"vLLM workers still hold the GPU after teardown: PIDs={surviving}. "
            "Would re-grab freed memory and OOM the HF teacher-forced phase."
        )
    logger.info("vLLM workers reaped; no surviving GPU PIDs.")


# ── Rao-Blackwellized JS estimator (pure tensor math) ─────────────────────


def kl_div_from_logprobs(
    log_p: torch.Tensor, log_q: torch.Tensor, base: float = 2.0
) -> torch.Tensor:
    """KL(p || q) = sum_x p(x) [log p(x) - log q(x)], in user-chosen base.

    Args:
        log_p: ``(..., V)`` natural-log probabilities of distribution P.
        log_q: ``(..., V)`` natural-log probabilities of distribution Q.
        base: log base for the returned KL (2.0 by default for JS in [0,1]).

    Returns:
        ``(...)`` non-negative KL values.
    """
    p = log_p.exp()
    # KL is non-negative — clamp tiny negatives from floating-point noise.
    kl_nat = (p * (log_p - log_q)).sum(dim=-1)
    kl_nat = torch.clamp(kl_nat, min=0.0)
    return kl_nat / math.log(base)


def js_from_logprobs(log_p: torch.Tensor, log_q: torch.Tensor, base: float = 2.0) -> torch.Tensor:
    """Jensen-Shannon divergence in base ``base`` (default 2 -> JS in [0,1]).

    JS(P, Q) = 1/2 KL(P || M) + 1/2 KL(Q || M),  M = 1/2 (P + Q).

    Args:
        log_p: ``(..., V)`` log P (natural log).
        log_q: ``(..., V)`` log Q (natural log).
        base: log base.

    Returns:
        ``(...)`` JS values, ``base=2`` puts them in [0, 1].
    """
    # log M = log(0.5 * (P + Q)) = log( exp(log P) + exp(log Q) ) - log 2.
    # logsumexp is numerically stable.
    stacked = torch.stack([log_p, log_q], dim=0)
    log_m = torch.logsumexp(stacked, dim=0) - math.log(2.0)
    kl_pm = kl_div_from_logprobs(log_p, log_m, base=base)
    kl_qm = kl_div_from_logprobs(log_q, log_m, base=base)
    return 0.5 * (kl_pm + kl_qm)


# ── Teacher-forced per-position divergence ─────────────────────────────────


def _build_chat_prefix(tokenizer, persona_text: str, question: str) -> str:
    """Chat-template prefix with the model about to answer.

    Mirrors ``eval_i456_onpolicy_emission.render_prefix``.
    """
    msgs = [
        {"role": "system", "content": persona_text},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _logprobs_at_response_positions(
    model,
    tokenizer,
    prefix_text: str,
    response_text: str,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Teacher-force ``prefix + response`` through ``model``; return per-position log-probs.

    Returns ``(log_probs, n_response_positions)`` where ``log_probs`` is
    ``(n_response_positions, V)`` natural-log next-token distribution at
    each response token position, and ``n_response_positions`` is the
    number of teacher-forced positions (0 if the response tokenizes to
    empty, in which case caller should skip this response).

    The "response positions" are the input positions IMMEDIATELY BEFORE
    each response token (the standard causal-LM next-token shift). So if
    prefix is length P and response is length R, the response positions
    sit at input indices [P-1, P, ..., P+R-2], with targets r_0, r_1, ...,
    r_{R-1} — exactly the slots where the model is choosing each response
    token given the preceding context.
    """
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if len(response_ids) == 0:
        return torch.empty(0, 0), 0
    full_ids = prefix_ids + response_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits  # (1, T, V)
    # Slots producing response_ids[0..R-1] are at input positions
    # P-1 .. P+R-2 (next-token shift). Use float32 for log_softmax stability.
    P = len(prefix_ids)
    R = len(response_ids)
    slot_logits = logits[0, P - 1 : P - 1 + R, :].float()
    log_probs = torch.nn.functional.log_softmax(slot_logits, dim=-1)
    return log_probs, R


def _per_position_js(
    model_s,
    model_sprime,
    tokenizer,
    prefix_s: str,
    prefix_sprime: str,
    response_text: str,
    device: torch.device,
) -> dict[str, list[float]] | None:
    """Compute per-response-position JS + KL_S->S' + KL_S'->S for one response.

    Returns dict with ``js`` / ``kl_s_sprime`` / ``kl_sprime_s`` lists,
    each length = # response tokens. Returns ``None`` if the response
    tokenizes to empty.

    The two models score the SAME response under DIFFERENT chat-template
    prefixes (S's prefix vs S'_B's prefix) — that's the slice-resolved
    setup: we want to know "given the same response text, how different
    are the next-token distributions under the two personas?"
    """
    lp_s, R_s = _logprobs_at_response_positions(model_s, tokenizer, prefix_s, response_text, device)
    if R_s == 0:
        return None
    lp_sp, R_sp = _logprobs_at_response_positions(
        model_sprime, tokenizer, prefix_sprime, response_text, device
    )
    # Both forward passes used the SAME response tokenization, so R is
    # guaranteed equal; assert to make the invariant explicit.
    assert R_s == R_sp, (R_s, R_sp)

    js_per_pos = js_from_logprobs(lp_s, lp_sp, base=2.0)  # (R,)
    kl_s_sp = kl_div_from_logprobs(lp_s, lp_sp, base=2.0)  # KL(S || S')
    kl_sp_s = kl_div_from_logprobs(lp_sp, lp_s, base=2.0)  # KL(S' || S)
    return {
        "js": js_per_pos.cpu().tolist(),
        "kl_s_sprime": kl_s_sp.cpu().tolist(),
        "kl_sprime_s": kl_sp_s.cpu().tolist(),
    }


# ── Phase JS — vLLM generation + HF teacher-force scoring ─────────────────


def phase_js_generate(
    behavior: str,
    R: int,
    max_new_tokens: int,
    max_model_len: int,
    seed: int,
    smoke_probes: int | None,
) -> dict[str, Any]:
    """vLLM-generate R responses per (persona × probe) for one behavior.

    Returns ``{"S": {slice_name: {probe_idx: [text, ...]}},
               "S_prime": {slice_name: {probe_idx: [text, ...]}}}``
    plus per-cell counts. Caller writes the cache to disk; Phase B reads
    it back to share the same responses across JS scoring and Phase
    Cosine (b) own-response mean.
    """
    s_name, sprime_name = PREDICTOR_PAIRS[behavior]
    s_text = PERSONAS[s_name]
    sprime_text = PERSONAS[sprime_name]

    slices = {
        "nontrigger": SLICE_NONTRIGGER if smoke_probes is None else SLICE_NONTRIGGER[:smoke_probes],
        "trigger": _trigger_for(behavior)
        if smoke_probes is None
        else _trigger_for(behavior)[:smoke_probes],
    }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Build the rendered list: persona-major (S first, then S'), slice, probe.
    rendered: list[str] = []
    index: list[tuple[str, str, int]] = []  # (persona_label, slice_name, probe_idx)
    for persona_label, ptext in (("S", s_text), ("S_prime", sprime_text)):
        for slice_name, probes in slices.items():
            for q_idx, q in enumerate(probes):
                rendered.append(_build_chat_prefix(tokenizer, ptext, q))
                index.append((persona_label, slice_name, q_idx))

    logger.info(
        "[phase_js_generate %s] rendering %d prefixes (R=%d samples each)...",
        behavior,
        len(rendered),
        R,
    )

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        seed=seed,
    )
    sampling = SamplingParams(
        n=R,
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    try:
        t0 = time.time()
        outputs = llm.generate(rendered, sampling)
        wall = time.time() - t0
        logger.info("[phase_js_generate %s] vLLM done in %.1fs", behavior, wall)
    finally:
        del llm
        gc.collect()
    _kill_vllm_workers()

    cache: dict[str, dict[str, dict[int, list[str]]]] = {"S": {}, "S_prime": {}}
    for row_idx, out in enumerate(outputs):
        persona_label, slice_name, q_idx = index[row_idx]
        cache.setdefault(persona_label, {}).setdefault(slice_name, {})[q_idx] = [
            s.text for s in out.outputs
        ]
    return {"cache": cache, "slices": slices, "gen_wall_seconds": wall}


def phase_js_score(
    behavior: str,
    cache: dict[str, dict[str, dict[int, list[str]]]],
    slices: dict[str, list[str]],
    smoke_probes: int | None,
) -> dict[str, Any]:
    """Teacher-force pooled responses through both conditioned base models.

    Loads HF Transformers AFTER vLLM teardown (caller must have run
    ``_kill_vllm_workers``). Builds two model objects sharing the same
    weights but with the persona prefix folded into the input — i.e. the
    SAME base model is forward-passed under S's prefix and S'_B's prefix.

    For storage efficiency we keep per-position lists only at the
    headline-slice granularity needed for ``exp_js_per_position.png``
    (the trigger slice for each behavior); per-probe scalars are
    sufficient for the matched-contrast headline.
    """
    s_name, sprime_name = PREDICTOR_PAIRS[behavior]
    s_text = PERSONAS[s_name]
    sprime_text = PERSONAS[sprime_name]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    device = torch.device("cuda:0")
    # Single weight set — we just rebuild the prefix per call.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()

    per_probe_scalars: dict[str, list[dict[str, float]]] = {"nontrigger": [], "trigger": []}
    per_position_traj: dict[str, list[list[float]]] = {"nontrigger": [], "trigger": []}

    try:
        for slice_name, probes in slices.items():
            logger.info(
                "[phase_js_score %s/%s] scoring %d probes (pooled R=16 per probe)...",
                behavior,
                slice_name,
                len(probes),
            )
            t0 = time.time()
            for q_idx, q in enumerate(probes):
                prefix_s = _build_chat_prefix(tokenizer, s_text, q)
                prefix_sp = _build_chat_prefix(tokenizer, sprime_text, q)
                pool_s = cache["S"][slice_name][q_idx]
                pool_sp = cache["S_prime"][slice_name][q_idx]
                # Symmetric Rao-Blackwellized: pool from BOTH conditioned models.
                pool = list(pool_s) + list(pool_sp)
                per_response_js: list[float] = []
                per_response_kl_s_sp: list[float] = []
                per_response_kl_sp_s: list[float] = []
                per_position_for_this_probe: list[list[float]] = []
                for response_text in pool:
                    res = _per_position_js(
                        model, model, tokenizer, prefix_s, prefix_sp, response_text, device
                    )
                    if res is None:
                        continue
                    per_response_js.append(float(torch.tensor(res["js"]).mean().item()))
                    per_response_kl_s_sp.append(
                        float(torch.tensor(res["kl_s_sprime"]).mean().item())
                    )
                    per_response_kl_sp_s.append(
                        float(torch.tensor(res["kl_sprime_s"]).mean().item())
                    )
                    per_position_for_this_probe.append(res["js"])

                if per_response_js:
                    probe_scalar = {
                        "probe_idx": q_idx,
                        "probe": q,
                        "n_pooled": len(per_response_js),
                        "mean_js": float(torch.tensor(per_response_js).mean().item()),
                        "mean_kl_s_sprime": float(torch.tensor(per_response_kl_s_sp).mean().item()),
                        "mean_kl_sprime_s": float(torch.tensor(per_response_kl_sp_s).mean().item()),
                    }
                else:
                    probe_scalar = {
                        "probe_idx": q_idx,
                        "probe": q,
                        "n_pooled": 0,
                        "mean_js": float("nan"),
                        "mean_kl_s_sprime": float("nan"),
                        "mean_kl_sprime_s": float("nan"),
                    }
                per_probe_scalars[slice_name].append(probe_scalar)
                # Keep per-position trajectories only for slice trajectory plot —
                # average across pooled responses (variable response lengths
                # require careful pad-with-NaN aggregation; we keep raw lists
                # and let the analyzer plot length-truncated traces).
                per_position_traj[slice_name].extend(per_position_for_this_probe)
            logger.info(
                "[phase_js_score %s/%s] done in %.1fs",
                behavior,
                slice_name,
                time.time() - t0,
            )
    finally:
        del model
        gc.collect()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    # Slice-mean scalars (the headline JS predictor values).
    slice_mean_js: dict[str, float] = {}
    slice_mean_kl_s_sp: dict[str, float] = {}
    slice_mean_kl_sp_s: dict[str, float] = {}
    for slice_name, probe_rows in per_probe_scalars.items():
        valid = [r for r in probe_rows if not math.isnan(r["mean_js"])]
        if valid:
            slice_mean_js[slice_name] = float(sum(r["mean_js"] for r in valid) / len(valid))
            slice_mean_kl_s_sp[slice_name] = float(
                sum(r["mean_kl_s_sprime"] for r in valid) / len(valid)
            )
            slice_mean_kl_sp_s[slice_name] = float(
                sum(r["mean_kl_sprime_s"] for r in valid) / len(valid)
            )
        else:
            slice_mean_js[slice_name] = float("nan")
            slice_mean_kl_s_sp[slice_name] = float("nan")
            slice_mean_kl_sp_s[slice_name] = float("nan")

    # Generic-averaged JS (the BLIND predictor) — mean across the 60-probe union.
    all_probe_js = [
        r["mean_js"]
        for slice_name in per_probe_scalars
        for r in per_probe_scalars[slice_name]
        if not math.isnan(r["mean_js"])
    ]
    averaged_js = float(sum(all_probe_js) / len(all_probe_js)) if all_probe_js else float("nan")

    return {
        "per_probe_scalars": per_probe_scalars,
        "per_position_traj": per_position_traj,
        "slice_mean_js": slice_mean_js,
        "slice_mean_kl_s_sprime": slice_mean_kl_s_sp,
        "slice_mean_kl_sprime_s": slice_mean_kl_sp_s,
        "averaged_js_union": averaged_js,
    }


# ── Phase Cosine — 3 extraction points, layer sweep ───────────────────────


def _get_last_token_activations(
    model, tokenizer, prefix_text: str, layers: list[int], device: torch.device
) -> dict[int, torch.Tensor]:
    """Hook the requested layers and capture the residual at the LAST input position.

    Returns ``{layer: (hidden_dim,) fp32 CPU tensor}``.
    """
    captures: dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captures[li] = hs.detach()

        return hook_fn

    hooks = []
    for li in layers:
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        hooks.append(h)
    try:
        input_ids = tokenizer.encode(prefix_text, add_special_tokens=False, return_tensors="pt").to(
            device
        )
        if input_ids.numel() == 0:
            raise RuntimeError(
                "prefix tokenized to empty — chat-template render produced no tokens"
            )
        with torch.no_grad():
            _ = model(input_ids=input_ids)
        last_pos = input_ids.shape[1] - 1
        return {li: captures[li][0, last_pos, :].float().cpu() for li in layers}
    finally:
        for h in hooks:
            h.remove()


def _get_mean_response_activations(
    model,
    tokenizer,
    prefix_text: str,
    response_text: str,
    layers: list[int],
    device: torch.device,
) -> dict[int, torch.Tensor] | None:
    """Forward ``prefix + response``; mean-pool residual over RESPONSE positions only.

    Returns ``{layer: (hidden_dim,) fp32 CPU tensor}`` or None if response
    tokenizes empty.
    """
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if len(response_ids) == 0:
        return None
    full_ids = torch.tensor([prefix_ids + response_ids], dtype=torch.long, device=device)

    captures: dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captures[li] = hs.detach()

        return hook_fn

    hooks = []
    for li in layers:
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        hooks.append(h)
    try:
        with torch.no_grad():
            _ = model(input_ids=full_ids)
        P = len(prefix_ids)
        R = len(response_ids)
        out: dict[int, torch.Tensor] = {}
        for li in layers:
            hs = captures[li][0, P : P + R, :].float()  # (R, hidden)
            out[li] = hs.mean(dim=0).cpu()
        return out
    finally:
        for h in hooks:
            h.remove()


def phase_cosine(  # noqa: C901  (Persona-Vectors recipe has 3 distinct extraction points + a layer sweep — splitting further would obscure the cosine pipeline; matches the reference impl in issue404_predictor_cossim.py)
    behavior: str,
    cache: dict[str, dict[str, dict[int, list[str]]]],
    slices: dict[str, list[str]],
    layers: list[int],
    do_recipe_b: bool,
    smoke_probes: int | None,
) -> dict[str, Any]:
    """Per (S, S', slice, layer) cosine at 3 extraction points.

    (a0) end-of-system-prompt — INPUT-INDEPENDENT.
    (a)  last token of {S, Q} — per-probe mean.
    (b)  mean of residual over each model's OWN response — per-probe mean.
    """
    s_name, sprime_name = PREDICTOR_PAIRS[behavior]
    s_text = PERSONAS[s_name]
    sprime_text = PERSONAS[sprime_name]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()

    n_layers = len(model.model.layers)
    bad = [li for li in layers if li < 0 or li >= n_layers]
    if bad:
        raise RuntimeError(f"layers {bad} out of range for {n_layers}-layer model")

    try:
        # (a0) end-of-system-prompt — one prefix per persona, no user msg.
        def _system_only_prefix(persona_text: str) -> str:
            rendered = tokenizer.apply_chat_template(
                [{"role": "system", "content": persona_text}], tokenize=False
            )
            if not rendered.strip():
                # Fallback per A9 — add an empty user turn if Qwen's chat
                # template renders nothing for system-only input.
                rendered = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": persona_text},
                        {"role": "user", "content": ""},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            assert rendered.strip(), f"empty system-only render for persona {persona_text[:60]!r}"
            return rendered

        act_s_a0 = _get_last_token_activations(
            model, tokenizer, _system_only_prefix(s_text), layers, device
        )
        act_sp_a0 = _get_last_token_activations(
            model, tokenizer, _system_only_prefix(sprime_text), layers, device
        )
        cos_a0: dict[int, float] = {}
        for li in layers:
            cos_a0[li] = float(
                torch.nn.functional.cosine_similarity(
                    act_s_a0[li].unsqueeze(0), act_sp_a0[li].unsqueeze(0), dim=-1
                ).item()
            )
        logger.info(
            "[phase_cosine %s] (a0) end-of-system-prompt: %s",
            behavior,
            {li: round(cos_a0[li], 4) for li in layers},
        )

        # (a) last-{S,Q}-token — per-probe per-slice.
        cos_a_per_slice: dict[str, dict[int, list[float]]] = {
            "nontrigger": {li: [] for li in layers},
            "trigger": {li: [] for li in layers},
        }
        for slice_name, probes in slices.items():
            for q in probes:
                prefix_s = _build_chat_prefix(tokenizer, s_text, q)
                prefix_sp = _build_chat_prefix(tokenizer, sprime_text, q)
                act_s = _get_last_token_activations(model, tokenizer, prefix_s, layers, device)
                act_sp = _get_last_token_activations(model, tokenizer, prefix_sp, layers, device)
                for li in layers:
                    sim = torch.nn.functional.cosine_similarity(
                        act_s[li].unsqueeze(0), act_sp[li].unsqueeze(0), dim=-1
                    ).item()
                    cos_a_per_slice[slice_name][li].append(float(sim))
        cos_a_slice_mean: dict[str, dict[int, float]] = {}
        for slice_name in cos_a_per_slice:
            cos_a_slice_mean[slice_name] = {
                li: float(sum(vs) / len(vs)) if vs else float("nan")
                for li, vs in cos_a_per_slice[slice_name].items()
            }

        # (b) own-response-mean — uses Phase JS generation cache.
        # Per persona: take each persona's OWN responses (not pooled).
        cos_b_per_slice: dict[str, dict[int, list[float]]] = {
            "nontrigger": {li: [] for li in layers},
            "trigger": {li: [] for li in layers},
        }
        if do_recipe_b:
            for slice_name, probes in slices.items():
                for q_idx, q in enumerate(probes):
                    prefix_s = _build_chat_prefix(tokenizer, s_text, q)
                    prefix_sp = _build_chat_prefix(tokenizer, sprime_text, q)
                    s_responses = cache["S"][slice_name][q_idx]
                    sp_responses = cache["S_prime"][slice_name][q_idx]
                    # Mean-pool over the per-persona response activations,
                    # then mean across the persona's R responses.
                    s_vec: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
                    sp_vec: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
                    for r_text in s_responses:
                        acts = _get_mean_response_activations(
                            model, tokenizer, prefix_s, r_text, layers, device
                        )
                        if acts is None:
                            continue
                        for li in layers:
                            s_vec[li].append(acts[li])
                    for r_text in sp_responses:
                        acts = _get_mean_response_activations(
                            model, tokenizer, prefix_sp, r_text, layers, device
                        )
                        if acts is None:
                            continue
                        for li in layers:
                            sp_vec[li].append(acts[li])
                    for li in layers:
                        if not s_vec[li] or not sp_vec[li]:
                            cos_b_per_slice[slice_name][li].append(float("nan"))
                            continue
                        s_mean = torch.stack(s_vec[li]).mean(dim=0)
                        sp_mean = torch.stack(sp_vec[li]).mean(dim=0)
                        sim = torch.nn.functional.cosine_similarity(
                            s_mean.unsqueeze(0), sp_mean.unsqueeze(0), dim=-1
                        ).item()
                        cos_b_per_slice[slice_name][li].append(float(sim))
        else:
            logger.info(
                "[phase_cosine %s] skipping recipe (b) (do_recipe_b=False — descope)",
                behavior,
            )

        cos_b_slice_mean: dict[str, dict[int, float]] = {}
        for slice_name in cos_b_per_slice:
            cos_b_slice_mean[slice_name] = {
                li: float(
                    sum(v for v in vs if not math.isnan(v))
                    / max(1, sum(1 for v in vs if not math.isnan(v)))
                )
                if any(not math.isnan(v) for v in vs)
                else float("nan")
                for li, vs in cos_b_per_slice[slice_name].items()
            }
    finally:
        del model
        gc.collect()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    return {
        "extraction_a0_endofsystemprompt": cos_a0,
        "extraction_a_lastinputtoken_per_slice_per_layer": cos_a_slice_mean,
        "extraction_a_per_probe": cos_a_per_slice,
        "extraction_b_ownresponsemean_per_slice_per_layer": cos_b_slice_mean,
        "extraction_b_per_probe": cos_b_per_slice if do_recipe_b else None,
        "layers": layers,
        "headline_layer": HEADLINE_LAYER,
    }


# ── Per-cell writer ────────────────────────────────────────────────────────


def _write_cell(out_dir: Path, behavior: str, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{behavior}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-samples", type=int, default=8, help="responses per persona per probe")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="cosine layer sweep (default: 7 14 21 27)",
    )
    parser.add_argument(
        "--no-recipe-b",
        action="store_true",
        help="skip own-response-mean cosine (descope option)",
    )
    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=list(PREDICTOR_PAIRS.keys()),
        help="which behaviors to run (default: all)",
    )
    parser.add_argument(
        "--smoke-probes",
        type=int,
        default=None,
        help="if set, only use the first N probes per slice (smoke run)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_466" / "predictors",
    )
    args = parser.parse_args()

    # Hard marker-id assert at top of main (R7).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)
    assert marker_ids == [MARKER_ID], (
        f"MARKER guard FAILED: '{MARKER}' tokenizes to {marker_ids}, expected [{MARKER_ID}]"
    )
    logger.info("Marker token assert OK: ※ -> [%d]", MARKER_ID)
    del tokenizer

    for behavior in args.behaviors:
        if behavior not in PREDICTOR_PAIRS:
            raise SystemExit(f"unknown behavior {behavior!r}; choices={list(PREDICTOR_PAIRS)}")
        logger.info("=== behavior %s ===", behavior)
        t_behavior = time.time()

        # Phase JS — generate.
        gen = phase_js_generate(
            behavior,
            R=args.r_samples,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            seed=args.seed,
            smoke_probes=args.smoke_probes,
        )
        # Phase JS — score (model load AFTER vLLM teardown).
        js_results = phase_js_score(
            behavior, gen["cache"], gen["slices"], smoke_probes=args.smoke_probes
        )
        # Phase Cosine.
        cos_results = phase_cosine(
            behavior,
            gen["cache"],
            gen["slices"],
            layers=args.layers,
            do_recipe_b=not args.no_recipe_b,
            smoke_probes=args.smoke_probes,
        )

        payload = {
            "behavior": behavior,
            "predictor_pair": list(PREDICTOR_PAIRS[behavior]),
            "config": {
                "r_samples": args.r_samples,
                "max_new_tokens": args.max_new_tokens,
                "max_model_len": args.max_model_len,
                "seed": args.seed,
                "layers": args.layers,
                "headline_layer": HEADLINE_LAYER,
                "do_recipe_b": not args.no_recipe_b,
                "smoke_probes": args.smoke_probes,
            },
            "js": js_results,
            "cosine": cos_results,
            "marker_token": MARKER,
            "marker_token_id": MARKER_ID,
            "metadata": _metadata(),
            "wall_seconds": time.time() - t_behavior,
        }
        out_path = _write_cell(args.out_dir, behavior, payload)
        logger.info(
            "Wrote %s in %.1fs (avgJS=%.4f, sliceJS_trig=%.4f, sliceJS_nontrig=%.4f)",
            out_path,
            payload["wall_seconds"],
            js_results["averaged_js_union"],
            js_results["slice_mean_js"].get("trigger", float("nan")),
            js_results["slice_mean_js"].get("nontrigger", float("nan")),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
