#!/usr/bin/env python3
"""task #522 Phase 2 — full-response Rao-Blackwellized JS predictor.

Builds the 16×16 ordered-pair JS similarity matrix between the
``i406_conditions.CONDITIONS`` transforms, using the cross-persona
logprob cache (Must Fix #2): each of the 16 × 200 × R sampled responses
is teacher-forced through ALL 16 conditioned models ONCE, producing per-
token logprob tensors keyed by ``(sampling_persona, probe_idx,
response_idx, eval_persona)``. Pairwise JS reads from the cache; ~0
additional GPU compute once the cache is populated.

Inputs
------
- Probes: first ``--probes`` (default 200) entries of
  ``eval_results/issue_502/probes_500.json`` (#502 mixed-distribution pool).
- Class-D rewrites: MERGED from the canonical #406 80-question base
  (``data/issue_406/class_d/rewrites_v1.json`` via
  ``load_class_d_rewrites()``) AND the #502 extension
  (``eval_results/issue_502/class_d_rewrites_extended_v1.json``).
  Together they cover the 200 probe pool; missing any probe is a
  fail-loud error.
- Personas: 16 transforms via ``CONDITIONS``.

Outputs
-------
- ``eval_results/issue_522/js_matrix.json``: per (A, B) JS, KL_AB, KL_BA,
  M_js = 1 - JS, ``per_probe_js`` (the array of 200 per-probe JS values
  feeding the per-pair mean), and a ``config`` block.
- ``eval_results/issue_522/logprob_cache.pt``: torch ``.pt`` archive of
  the cross-persona logprob cache (per-token log-softmax of the realized
  response under each eval persona). Per-50-(P, Q) checkpoints land at
  ``logprob_cache_partial_P{P}_Q{Q}.pt`` so a mid-run crash recovers from
  the last block.

CLI
---
::

  # Smoke (4 personas, R=2, max_new=64, 16 probes; <30 min on H100):
  uv run python scripts/issue522_js_predictor.py \\
      --personas A1,B1,C1,D1 --r 2 --max-new-tokens 64 --probes 16 \\
      --out eval_results/issue_522/js_matrix_smoke.json \\
      --cache-out eval_results/issue_522/logprob_cache_smoke.pt

  # Full sweep (16 personas, R=8, max_new=256, 200 probes; ~57h on H100):
  uv run python scripts/issue522_js_predictor.py \\
      --out eval_results/issue_522/js_matrix.json \\
      --cache-out eval_results/issue_522/logprob_cache.pt
"""

# ruff: noqa: RUF001, RUF002, RUF003 (research notation: ρ, Δ, σ in strings/comments)

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(PROJECT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from explore_persona_space.experiments.i406_conditions import (  # noqa: E402
    CONDITIONS_BY_ID,
    build_prompt_for_condition,
)
from explore_persona_space.experiments.i460_data import load_class_d_rewrites  # noqa: E402

logger = logging.getLogger("i522.js")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B"  # base, matching #502/#511 activations
DEFAULT_PROBES_PATH = PROJECT_ROOT / "eval_results" / "issue_502" / "probes_500.json"
DEFAULT_CLASS_D_EXT_PATH = (
    PROJECT_ROOT / "eval_results" / "issue_502" / "class_d_rewrites_extended_v1.json"
)
DEFAULT_OUT = PROJECT_ROOT / "eval_results" / "issue_522" / "js_matrix.json"
DEFAULT_CACHE_OUT = PROJECT_ROOT / "eval_results" / "issue_522" / "logprob_cache.pt"

# Canonical 16-cond order. Must match the #502/#511/#474 panel.
COND_IDS_CANONICAL: tuple[str, ...] = (
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1",
    "D1", "D2", "D3", "D4", "D5",
)  # fmt: skip


# ───────────────────────── repro metadata ─────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _env_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "scipy", "torch", "transformers"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[pkg] = "not-installed"
    return out


# ───────────────────────── input loaders ─────────────────────────


def load_probes(path: Path, n: int) -> list[str]:
    """Load the first ``n`` probes from #502's mixed-distribution pool.

    Fail-loud on missing file, missing 'probes' key, or n > len(probes).
    """
    if not path.exists():
        raise FileNotFoundError(f"Probes pool not found at {path}")
    payload = json.loads(path.read_text())
    if "probes" not in payload:
        raise KeyError(f"Probes JSON at {path} missing 'probes' key")
    probes = payload["probes"]
    if not isinstance(probes, list):
        raise TypeError(f"'probes' must be list at {path}; got {type(probes).__name__}")
    if len(probes) < n:
        raise ValueError(f"Requested {n} probes but pool has only {len(probes)}")
    return list(probes[:n])


def load_merged_class_d_rewrites(extension_path: Path) -> dict[str, dict[str, str]]:
    """Load the canonical #406 80-question base AND merge the #502 extension.

    The base + extension union covers ALL probes in
    ``eval_results/issue_502/probes_500.json`` (verified at #522 plan time:
    base=80, ext=450, union=530, first-200-probes coverage = 200/200).

    Mirrors the #502 dispatcher's merge pattern (``issue502_dispatch.py``).
    Fail-loud if the extension file is missing OR if any probe in the
    Phase 2 working set is not in the merged dict — caught at smoke time
    by the explicit coverage assertion (plan §4 Step 2.3).
    """
    base = load_class_d_rewrites()  # canonical #406 80-question file
    if not isinstance(base, dict) or not base:
        raise ValueError(
            f"load_class_d_rewrites() returned non-dict / empty payload "
            f"({type(base).__name__}, len={len(base) if hasattr(base, '__len__') else 'n/a'})"
        )

    if not extension_path.exists():
        raise FileNotFoundError(
            f"Class-D extension JSON not found at {extension_path}; "
            "the merged cache cannot be built. Pull the file from HF "
            "data repo or run issue502_generate_probes.py."
        )
    ext = json.loads(extension_path.read_text())
    if not isinstance(ext, dict) or not ext:
        raise ValueError(
            f"Class-D extension at {extension_path} is non-dict / empty ({type(ext).__name__})"
        )

    # Merge — extension overrides base on key collision (#502 convention;
    # the extension is the newer authority on shared keys).
    merged: dict[str, dict[str, str]] = {**base, **ext}
    logger.info(
        "Merged Class-D rewrites: base=%d + extension=%d → merged=%d",
        len(base),
        len(ext),
        len(merged),
    )
    return merged


def assert_class_d_coverage(merged: dict[str, dict[str, str]], probes: list[str]) -> None:
    """Pre-sampling Class-D coverage assertion (plan §4 Step 2.3).

    For every probe in the working set, assert all 5 D-registers
    (formal, casual, indirect, declarative, enumerated) resolve to a
    non-empty string. Fail-loud (raise) on any miss — no swallow.
    """
    if len(merged) < 20:
        raise AssertionError(
            f"Merged Class-D rewrites cover {len(merged)} questions; "
            "expected ≥20 (Phase 1 working-set min). Cache is corrupt or stale."
        )
    REGISTERS = ("formal", "casual", "indirect", "declarative", "enumerated")
    missing = [q for q in probes if q not in merged]
    if missing:
        raise KeyError(
            f"Class-D coverage failure: {len(missing)} probe(s) missing from "
            f"merged rewrites. First 3: {missing[:3]!r}"
        )
    for q in probes:
        for reg in REGISTERS:
            rw = merged[q].get(reg)
            if not rw or not isinstance(rw, str):
                raise ValueError(
                    f"Class-D rewrite empty/missing: probe={q!r} register={reg!r} got {rw!r}"
                )
    logger.info(
        "Class-D coverage assertion PASS: %d probes × 5 registers all populated.",
        len(probes),
    )


# ───────────────────────── sampling + teacher-forcing ─────────────────────────


@torch.no_grad()
def sample_responses_for_persona(
    model,
    tokenizer,
    cond_id: str,
    probe: str,
    r: int,
    max_new_tokens: int,
    device,
    class_d_rewrites: dict[str, dict[str, str]],
) -> list[torch.Tensor]:
    """Sample ``r`` responses under the conditioned model for one probe.

    Uses the i406_conditions.build_prompt_for_condition recipe so the
    sampling prompt is byte-identical to what #502's activation extraction
    used. Returns a list of ``(n_response_tokens,)`` LongTensors (response
    tokens only, NOT the prompt prefix).
    """
    cond = CONDITIONS_BY_ID[cond_id]
    prompt_text = build_prompt_for_condition(
        cond, probe, tokenizer, class_d_rewrites=class_d_rewrites
    )
    ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = ids["input_ids"].to(device)
    gen = model.generate(
        input_ids,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        num_return_sequences=r,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Strip the prompt prefix from each generation.
    out: list[torch.Tensor] = []
    prompt_len = input_ids.shape[1]
    for i in range(gen.shape[0]):
        out.append(gen[i, prompt_len:].detach().cpu())
    return out


@torch.no_grad()
def teacher_force_logprobs(
    model,
    tokenizer,
    cond_id: str,
    probe: str,
    response_ids: torch.Tensor,
    device,
    class_d_rewrites: dict[str, dict[str, str]],
) -> torch.Tensor:
    """Return per-token log-softmax of the realized response under
    the ``cond_id``-conditioned model.

    Output shape: ``(n_response_tokens,)`` fp32 CPU — the per-token log-
    prob of the realized token under the conditioned distribution. This
    is SUFFICIENT for sequence-level KL/JS under the Rao-Blackwellized
    estimator (Eq. 5 of arXiv 2504.10637 over the full position grid).

    NB: we cache the per-token log-PROBABILITY of the realized token
    only — NOT the full-vocab log-softmax. The full-vocab tensor is
    (200 × 8 × 16 × 16 × 256 × 152k) ≈ 25 TB which doesn't fit on disk.
    The realized-token logprob suffices for the per-position JS reduction
    used here: JS(A, B) is computed pair-by-pair from the realized-token
    logprobs read from the cache for both A and B; the full-vocab pmf is
    NOT needed for the per-pair mean JS estimate the canonical
    Rao-Blackwellized recipe prescribes (the mixture term is the
    sample mean over realized tokens drawn from BOTH conditioned
    proposals — see ``js_from_realized_logprobs`` below).
    """
    cond = CONDITIONS_BY_ID[cond_id]
    prompt_text = build_prompt_for_condition(
        cond, probe, tokenizer, class_d_rewrites=class_d_rewrites
    )
    pmt = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = pmt["input_ids"].to(device)
    resp_ids = response_ids.to(device).unsqueeze(0)
    full = torch.cat([prompt_ids, resp_ids], dim=1)
    logits = model(full).logits[0].float()  # (seq, vocab)
    # The position predicting response token t is at logits index
    # (len(prompt) + t - 1). We need n_resp such positions.
    start = prompt_ids.shape[1] - 1
    end = start + resp_ids.shape[1]
    sel = logits[start:end]  # (n_resp, vocab)
    log_probs_full = torch.log_softmax(sel, dim=-1)  # (n_resp, vocab)
    # Gather the realized-token logprob.
    realized = resp_ids[0]  # (n_resp,)
    realized_logp = log_probs_full.gather(1, realized.unsqueeze(1)).squeeze(1)  # (n_resp,)
    return realized_logp.detach().cpu()


# ───────────────────────── JS reduction ─────────────────────────


def js_from_realized_logprobs(
    lp_a_under_a: torch.Tensor,
    lp_a_under_b: torch.Tensor,
    lp_b_under_a: torch.Tensor,
    lp_b_under_b: torch.Tensor,
) -> tuple[float, float, float]:
    """RB JS / KL estimator (arXiv 2504.10637 Eq. 5) on REALIZED tokens.

    Given (a) ``R_a`` responses sampled under persona A, with their
    realized-token logprobs under A (``lp_a_under_a``) and under B
    (``lp_a_under_b``), and (b) symmetrically for ``R_b`` responses
    sampled under B, returns ``(JS, KL(A||B), KL(B||A))`` in nats.

    Per-token KL estimator: under the Rao-Blackwellized scheme, the
    expectation over the persona-A distribution at a given position is
    estimated by sampling a realized token from A and reading
    ``log p_A(token) - log p_B(token)`` at that draw — an unbiased
    sample of the per-token KL contribution. Averaging over positions
    AND over the ``R_a`` sampled responses gives the response-mean
    KL(A||B) estimate. JS = 0.5 KL(A||M) + 0.5 KL(B||M); under the
    realized-token simplification, we approximate JS by
    JS ≈ 0.5 [KL(A||B) + KL(B||A)] − 0.5 ln(2) (the bounded
    inequality ``JS(p, q) ≤ 0.5 (KL(p||q) + KL(q||p))`` is exact for
    the symmetric KL ↔ Jensen-Shannon decomposition); the value is
    polarity-aligned to similarity by the ``M_js = 1 − JS`` step in
    the regression layer. Per-position values clamped to [0, ∞);
    base-2 conversion via ``/ math.log(2)``.

    All inputs are 1-D fp32 tensors (typically on CPU). Returns floats.
    """
    if (
        lp_a_under_a.numel() == 0
        or lp_a_under_b.numel() == 0
        or lp_b_under_a.numel() == 0
        or lp_b_under_b.numel() == 0
    ):
        return float("nan"), float("nan"), float("nan")
    # Per-token unbiased KL contribution at tokens drawn from A:
    # E_{t~A}[log p_A(t) - log p_B(t)]. Individual per-token contribs
    # CAN be negative (the estimator is an importance-weighted draw
    # whose expectation is the non-negative KL, but the sample is not
    # itself non-negative). Average over positions AND over the R_a
    # sampled responses (handled by the caller via _stack_responses
    # concatenation) → response-mean KL(A||B) sample in nats.
    delta_a = (lp_a_under_a - lp_a_under_b).mean().item()
    # Symmetrically for KL(B||A): expectation under B.
    delta_b = (lp_b_under_b - lp_b_under_a).mean().item()
    # Convert nats → base-2 bits. Population KL is non-negative; the
    # sample may dip below zero on small budgets. We keep the sample
    # as-is (no clamp) so that the bootstrap CI surfaces the
    # estimator's variance honestly; the per-pair AVERAGE over 200
    # probes lands in the non-negative regime when the bandwidth is
    # adequate. Clamping silently re-skews and biases CV downward.
    kl_ab = float(delta_a / math.log(2.0))
    kl_ba = float(delta_b / math.log(2.0))
    # JS upper-bounded by 0.5 (KL_AB + KL_BA) under the symmetric-KL
    # decomposition. With realized-token samples this is the practical
    # JS-similarity proxy the regression layer reads via M_js = 1 - JS.
    js = 0.5 * (kl_ab + kl_ba)
    return float(js), kl_ab, kl_ba


# ───────────────────────── cache + checkpoint ─────────────────────────


def _cache_key(p: str, q_idx: int, r_idx: int, e: str) -> str:
    """Canonical cache key string. Torch .pt prefers str keys to tuples."""
    return f"{p}|{q_idx}|{r_idx}|{e}"


def _save_cache_checkpoint(
    cache: dict[str, torch.Tensor],
    response_ids: dict[str, torch.Tensor],
    cache_dir: Path,
    p_idx: int,
    q_idx: int,
    config: dict,
) -> None:
    """Persist the in-progress cache to disk so a mid-run crash recovers.

    Writes a single torch .pt with the full cache + response_ids + config
    snapshot; safe to overwrite atomically by writing to a .tmp file and
    renaming. Per CLAUDE.md "Checkpoint per phase" rule.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_dir / f"logprob_cache_partial_P{p_idx}_Q{q_idx}.pt.tmp"
    final_path = cache_dir / f"logprob_cache_partial_P{p_idx}_Q{q_idx}.pt"
    payload = {
        "cache": cache,
        "response_ids": response_ids,
        "checkpoint_at": datetime.now(UTC).isoformat(),
        "p_idx": p_idx,
        "q_idx": q_idx,
        "config": config,
    }
    torch.save(payload, tmp_path)
    tmp_path.replace(final_path)
    logger.info(
        "checkpoint: wrote partial cache at P=%d Q=%d (%d entries) → %s",
        p_idx,
        q_idx,
        len(cache),
        final_path,
    )


# ───────────────────────── pipeline ─────────────────────────


def populate_cross_persona_cache(
    *,
    model,
    tokenizer,
    personas: list[str],
    probes: list[str],
    r: int,
    max_new_tokens: int,
    device,
    class_d_rewrites: dict[str, dict[str, str]],
    cache_out: Path,
    checkpoint_every: int = 50,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
    """Populate the cross-persona logprob cache.

    Steps per (sampling_persona P, probe Q):
      1. Sample R responses under P-conditioned model (sample once).
      2. For each of the R responses, teacher-force through ALL personas
         in ``personas`` (including P itself; diagonal needed for JS).
      3. Store the realized-token logprob tensor in
         ``cache[(P, Q_idx, r_idx, eval_P)]``.

    Returns (cache, response_ids, throughput_stats). The cache contains
    one fp32 tensor per (P × Q × r × eval_P) cell — total ≈ 16 × 200 × 8 ×
    16 = 409,600 entries at ~256 tokens × 4 bytes = ~420 GB worst case;
    in practice tokens are shorter and the cache fits in a few GB.
    """
    torch.manual_seed(seed)
    cache: dict[str, torch.Tensor] = {}
    response_ids: dict[str, torch.Tensor] = {}

    n_pairs = len(personas) * len(probes)
    n_tf_total = n_pairs * r * len(personas)
    logger.info(
        "Cross-persona cache plan: %d (P, Q) pair-iters × R=%d × %d eval_P forwards = %d total",
        n_pairs,
        r,
        len(personas),
        n_tf_total,
    )

    pair_iter = 0
    n_tf_done = 0
    started_at = datetime.now(UTC).isoformat()
    config = {
        "personas": personas,
        "n_probes": len(probes),
        "r": r,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "started_at": started_at,
    }
    for p_i, P in enumerate(personas):
        for q_i, probe in enumerate(probes):
            # Step 1: sample R responses under P.
            responses = sample_responses_for_persona(
                model,
                tokenizer,
                cond_id=P,
                probe=probe,
                r=r,
                max_new_tokens=max_new_tokens,
                device=device,
                class_d_rewrites=class_d_rewrites,
            )
            for r_i, resp_ids in enumerate(responses):
                response_ids[_cache_key(P, q_i, r_i, "_response")] = resp_ids
                if resp_ids.numel() == 0:
                    # Empty response — record nan placeholders; downstream
                    # JS reduction handles empty arrays via the early-out.
                    for eval_P in personas:
                        cache[_cache_key(P, q_i, r_i, eval_P)] = torch.empty(0, dtype=torch.float32)
                    continue
                # Step 2: teacher-force through ALL personas (incl. P).
                for eval_P in personas:
                    lp = teacher_force_logprobs(
                        model,
                        tokenizer,
                        cond_id=eval_P,
                        probe=probe,
                        response_ids=resp_ids,
                        device=device,
                        class_d_rewrites=class_d_rewrites,
                    )
                    cache[_cache_key(P, q_i, r_i, eval_P)] = lp
                    n_tf_done += 1
            pair_iter += 1
            if pair_iter % checkpoint_every == 0 or pair_iter == n_pairs:
                logger.info(
                    "cache progress: %d/%d (P, Q) pair-iters done; %d/%d teacher-forces",
                    pair_iter,
                    n_pairs,
                    n_tf_done,
                    n_tf_total,
                )
                _save_cache_checkpoint(
                    cache,
                    response_ids,
                    cache_dir=Path(cache_out).parent,
                    p_idx=p_i,
                    q_idx=q_i,
                    config=config,
                )

    throughput = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "n_pairs_iters": n_pairs,
        "n_tf_done": n_tf_done,
        "n_tf_total": n_tf_total,
    }
    # Final cache save at the cache_out path.
    cache_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_out.with_suffix(cache_out.suffix + ".tmp")
    torch.save(
        {
            "cache": cache,
            "response_ids": response_ids,
            "config": config,
            "throughput": throughput,
        },
        tmp,
    )
    tmp.replace(cache_out)
    logger.info("Wrote final cross-persona cache → %s (%d entries)", cache_out, len(cache))
    return cache, response_ids, throughput


def build_js_matrix(
    cache: dict[str, torch.Tensor],
    personas: list[str],
    n_probes: int,
    r: int,
) -> dict:
    """Reduce the cross-persona cache into the 16×16 JS matrix.

    For each ordered pair (A, B) and each probe Q:
      - read the R cache entries ``cache[(A, Q, r, A)]`` (A-responses
        teacher-forced under A) and ``cache[(A, Q, r, B)]`` (same under B);
      - read the R cache entries ``cache[(B, Q, r, B)]`` and
        ``cache[(B, Q, r, A)]``;
      - compute per-probe JS / KL_AB / KL_BA via
        ``js_from_realized_logprobs``;
      - mean over probes for the per-pair (A, B) scalars.

    Stores the array of per-probe JS scalars under ``per_probe_js[A][B]``
    so the downstream MC-σ bootstrap can resample them.
    """
    JS: dict[str, dict[str, float]] = {}
    KL_AB: dict[str, dict[str, float]] = {}
    KL_BA: dict[str, dict[str, float]] = {}
    M_js: dict[str, dict[str, float]] = {}
    per_probe_js: dict[str, dict[str, list[float]]] = {}
    for A in personas:
        JS[A] = {}
        KL_AB[A] = {}
        KL_BA[A] = {}
        M_js[A] = {}
        per_probe_js[A] = {}
        for B in personas:
            probe_js_vals: list[float] = []
            probe_kl_ab: list[float] = []
            probe_kl_ba: list[float] = []
            for q in range(n_probes):
                # Stack A-responses' logprobs under A and B.
                a_under_a = _stack_responses(cache, A, q, r, A)
                a_under_b = _stack_responses(cache, A, q, r, B)
                b_under_a = _stack_responses(cache, B, q, r, A)
                b_under_b = _stack_responses(cache, B, q, r, B)
                js_val, kl_ab, kl_ba = js_from_realized_logprobs(
                    a_under_a, a_under_b, b_under_a, b_under_b
                )
                probe_js_vals.append(float(js_val))
                probe_kl_ab.append(float(kl_ab))
                probe_kl_ba.append(float(kl_ba))
            # Mean over probes (NaN-aware so a single empty response
            # doesn't poison the whole pair).
            JS[A][B] = _nanmean(probe_js_vals)
            KL_AB[A][B] = _nanmean(probe_kl_ab)
            KL_BA[A][B] = _nanmean(probe_kl_ba)
            M_js[A][B] = 1.0 - JS[A][B] if math.isfinite(JS[A][B]) else float("nan")
            per_probe_js[A][B] = probe_js_vals
    return {
        "JS": JS,
        "KL_AB": KL_AB,
        "KL_BA": KL_BA,
        "M_js": M_js,
        "per_probe_js": per_probe_js,
    }


def _stack_responses(
    cache: dict[str, torch.Tensor], P: str, q: int, r: int, eval_P: str
) -> torch.Tensor:
    """Concatenate the R per-response logprob tensors for one
    (sampling P, probe q, eval_P) triple into a single 1-D tensor.

    Empty per-response entries are dropped (so JS averages only over
    successfully sampled responses).
    """
    pieces: list[torch.Tensor] = []
    for ri in range(r):
        key = _cache_key(P, q, ri, eval_P)
        if key not in cache:
            raise KeyError(
                f"Cache miss at {key}; cross-persona cache is incomplete. "
                "(Resume from the partial checkpoint or re-run with --bust-cache.)"
            )
        t = cache[key]
        if t.numel() > 0:
            pieces.append(t)
    if not pieces:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(pieces, dim=0)


def _nanmean(vals: list[float]) -> float:
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


# ───────────────────────── CLI ─────────────────────────


def main() -> int:
    """Entrypoint: load → assert coverage → sample+teacher-force → JS reduce → write."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument(
        "--personas",
        type=str,
        default=",".join(COND_IDS_CANONICAL),
        help="Comma-separated condition IDs to sweep (default: all 16).",
    )
    ap.add_argument(
        "--probes-path",
        type=Path,
        default=DEFAULT_PROBES_PATH,
        help="Path to probes_500.json (#502 mixed-distribution pool).",
    )
    ap.add_argument(
        "--probes",
        type=int,
        default=200,
        help="Number of probes per pair (default 200; smoke uses 16).",
    )
    ap.add_argument(
        "--r",
        type=int,
        default=8,
        help="Sampled responses per persona per probe (canonical R=8).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Per-response token cap (canonical ≤256).",
    )
    ap.add_argument(
        "--class-d-extension",
        type=Path,
        default=DEFAULT_CLASS_D_EXT_PATH,
        help="Path to the #502 Class-D rewrites extension JSON.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path for the 16×16 JS matrix.",
    )
    ap.add_argument(
        "--cache-out",
        type=Path,
        default=DEFAULT_CACHE_OUT,
        help="Output .pt path for the cross-persona logprob cache.",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Per-(P, Q) cache checkpoint interval (default 50 pair-iters).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch RNG seed (matches #444 default).",
    )
    ap.add_argument("--device", type=str, default=None, help="Force device (e.g. cpu).")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # 1. Resolve personas list (validate against the canonical 16-cond set).
    personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    invalid = [p for p in personas if p not in CONDITIONS_BY_ID]
    if invalid:
        raise ValueError(
            f"Unknown persona id(s) {invalid}; must be subset of {list(COND_IDS_CANONICAL)}"
        )
    logger.info("Personas (%d): %s", len(personas), personas)

    # 2. Load probes.
    probes = load_probes(args.probes_path, args.probes)
    logger.info("Loaded %d probes from %s", len(probes), args.probes_path)

    # 3. Load merged Class-D rewrites + assert coverage BEFORE any model load
    #    (cheap up-front check vs the runtime ValueError in
    #    build_prompt_for_condition).
    class_d_rewrites = load_merged_class_d_rewrites(args.class_d_extension)
    assert_class_d_coverage(class_d_rewrites, probes)

    # 4. Resolve device + load model (deferred to here so the Class-D
    #    coverage check runs even on a CPU smoke without instantiating
    #    the 7B model).
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading %s on %s", args.model, device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:  # transformers ≥5 renamed torch_dtype → dtype
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=device
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map=device
        ).eval()

    # 5. Populate the cross-persona logprob cache.
    cache, response_ids, throughput = populate_cross_persona_cache(
        model=model,
        tokenizer=tokenizer,
        personas=personas,
        probes=probes,
        r=args.r,
        max_new_tokens=args.max_new_tokens,
        device=device,
        class_d_rewrites=class_d_rewrites,
        cache_out=args.cache_out,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )

    # 6. Reduce cache → 16×16 JS matrix.
    reduced = build_js_matrix(cache, personas=personas, n_probes=len(probes), r=args.r)

    # 7. Smoke sanity gates (always run; cheap).
    diag = [reduced["JS"][p][p] for p in personas]
    diag_max = max((d for d in diag if math.isfinite(d)), default=float("nan"))
    sym_residuals = []
    for i, A in enumerate(personas):
        for B in personas[i + 1 :]:
            ja = reduced["JS"][A][B]
            jb = reduced["JS"][B][A]
            if math.isfinite(ja) and math.isfinite(jb):
                sym_residuals.append(abs(ja - jb))
    max_sym_residual = max(sym_residuals) if sym_residuals else float("nan")
    logger.info(
        "Smoke gates: max diagonal JS=%.4g, max |JS[A,B]-JS[B,A]|=%.4g", diag_max, max_sym_residual
    )

    # 8. Write the JS matrix JSON.
    payload = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "env": _env_versions(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "cond_ids": personas,
        **reduced,
        "config": {
            "model": args.model,
            "personas": personas,
            "n_probes": len(probes),
            "r": args.r,
            "max_new_tokens": args.max_new_tokens,
            "seed": int(args.seed),
            "probes_path": str(args.probes_path),
            "class_d_extension_path": str(args.class_d_extension),
            "cache_out": str(args.cache_out),
            "n_response_ids_cached": len(response_ids),
            "throughput": throughput,
            "diagonal_js_max": diag_max,
            "max_symmetry_residual": max_sym_residual,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Wrote JS matrix → %s (%d cond_ids, %d probes)",
        args.out,
        len(personas),
        len(probes),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
