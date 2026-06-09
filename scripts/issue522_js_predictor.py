#!/usr/bin/env python3
"""task #522 Phase 2 — full-response Rao-Blackwellized JS predictor.

Builds the 16×16 ordered-pair JS similarity matrix between the
``i406_conditions.CONDITIONS`` transforms using the **canonical full-vocab
per-position mixture Rao-Blackwellized JS** estimator (Amini/Vieira/Cotterell
2025, arXiv 2504.10637) per ``.claude/rules/persona-distance-metrics.md``.

Architecture (round-2 fix — JS computed INSIDE the teacher-force pass)
=====================================================================

The naive "cache realized-token logprobs, derive JS at reduce time" path
computes **Jeffreys / symmetric-KL on realized-token mean log-ratios** —
that is NOT the canonical JS. Round-1 review caught this; round-2 fix:

* Per (sample_persona P, probe q, response_idx r):
    1. Compute ``log_probs[E] = log_softmax(model(prompt_E + response))`` for
       every eval persona ``E`` in the working set — full-vocab tensors, kept
       in CPU memory for the duration of this one response.
    2. For every unordered pair ``{A, B}`` containing P, reduce
       ``log_probs[A]`` and ``log_probs[B]`` to **per-position scalars**:
       ``kl_pos_a[t] = sum_v exp(log_probs[A][t,v]) * (log_probs[A][t,v] - log_m[t,v])``
       where ``log_m[t,v] = logsumexp(log_probs[A][t,v], log_probs[B][t,v]) - log 2``.
       Symmetrically for ``kl_pos_b``. Per-position JS = ``0.5 * (kl_pos_a + kl_pos_b)``.
    3. Discard the full-vocab tensors; cache only the per-position scalars
       (~256 fp32/pos × 3 directions = ~3 KB per cache entry).

Cache schema
------------

* Disk: ``{cache_dir}/logprob_cache.pt`` (single torch ``.pt`` archive).
* In-memory dict ``cache``: key ``"{P}|{q_idx}|{r_idx}|{A}|{B}"`` (string),
  value ``dict(js=Tensor[n_resp], kl_a=Tensor[n_resp], kl_b=Tensor[n_resp])``
  in **nats**. ``A < B`` lexicographically (symmetric storage); the diagonal
  ``P=A=B`` is stored as zeros. The reduction layer reads
  ``cache[(min(A,B), max(A,B))]`` for both ordered (A,B) and (B,A) lookups.
* Per-50-(P, Q) checkpoints land at
  ``{cache_dir}/{cache_out.stem}_partial_P{P}_Q{Q}.pt`` (round-4: namespaced
  by ``cache_out.stem`` so smoke + full runs sharing the same dir don't
  collide). ``populate_..._cache`` reads the LATEST partial with that stem
  at start (resume support).
* ``cache_schema.json`` is written alongside ``logprob_cache.pt`` documenting
  the key format + tensor shapes.

Inputs
------
- Probes: first ``--probes`` (default 200) entries of
  ``eval_results/issue_502/probes_500.json`` (#502 mixed-distribution pool).
- Class-D rewrites: MERGED from the canonical #406 80-question base
  (``data/issue_406/class_d/rewrites_v1.json``) AND the #502 extension
  (``eval_results/issue_502/class_d_rewrites_extended_v1.json``).
- Personas: 16 transforms via ``CONDITIONS``.

Outputs
-------
- ``eval_results/issue_522/js_matrix.json``: per (A, B) JS, KL_AB, KL_BA,
  M_js = 1 - JS, ``per_probe_js`` (array of 200 per-probe JS values feeding
  the per-pair mean), and a ``config`` block.
- ``/workspace/eval_results/issue_522/logprob_cache.pt``: torch ``.pt`` archive
  of the per-position JS / KL scalar cache (NOT under ``eval_results/`` per
  the JSON-only upload policy).

CLI
---
::

  # Smoke (4 personas, R=2, max_new=64, 16 probes; <30 min on H100):
  uv run python scripts/issue522_js_predictor.py \\
      --personas A1,B1,C1,D1 --r 2 --max-new-tokens 64 --probes 16 \\
      --out eval_results/issue_522/js_matrix_smoke.json \\
      --cache-out /workspace/eval_results/issue_522/logprob_cache_smoke.pt

  # Full sweep (16 personas, R=8, max_new=256, 200 probes; ~57h on H100):
  uv run python scripts/issue522_js_predictor.py \\
      --out eval_results/issue_522/js_matrix.json \\
      --cache-out /workspace/eval_results/issue_522/logprob_cache.pt
"""

# ruff: noqa: RUF001, RUF002, RUF003 (research notation: ρ, Δ, σ in strings/comments)

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import re  # used at module level by _PARTIAL_FILENAME_RE (round-3 numeric partial sort)
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
# Default cache lives OUTSIDE eval_results/ (JSON-only per CLAUDE.md upload policy);
# /workspace is the canonical pod-side persistent volume.
DEFAULT_CACHE_OUT = Path("/workspace") / "eval_results" / "issue_522" / "logprob_cache.pt"

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
    out: list[torch.Tensor] = []
    prompt_len = input_ids.shape[1]
    for i in range(gen.shape[0]):
        out.append(gen[i, prompt_len:].detach().cpu())
    return out


@torch.no_grad()
def teacher_force_full_vocab_logprobs(
    model,
    tokenizer,
    cond_id: str,
    probe: str,
    response_ids: torch.Tensor,
    device,
    class_d_rewrites: dict[str, dict[str, str]],
) -> torch.Tensor:
    """Per-position full-vocab log-softmax of next-token distribution.

    Teacher-forces ``response_ids`` after the ``cond_id``-conditioned prompt
    for ``probe`` and returns the per-position log-softmax over the FULL
    vocabulary at the positions predicting each response token.

    Returns
    -------
    log_probs : torch.Tensor
        Shape ``(n_response_tokens, vocab_size)``, ``dtype=torch.float32``,
        on CPU. The tensor is moved off GPU before return so the caller
        can hold 16 of them simultaneously (~16 × 256 × 152k × 4 bytes ≈
        20 GB CPU RAM, fits on any sane VM).

    NB: this is the FULL-VOCAB log-softmax (NOT just realized-token
    logprobs). The full-vocab tensor is required for the canonical
    per-position mixture m = ½(p_A + p_B) used by the
    Rao-Blackwellized JS reduction (see ``per_position_js_kl_from_logprobs``).
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
    return log_probs_full.detach().cpu()


# ───────────────────────── JS reduction (canonical full-vocab) ─────────────────────────


def per_position_js_kl_from_logprobs(
    log_p_a: torch.Tensor,
    log_p_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonical full-vocab per-position mixture JS / per-direction KL (nats).

    Implements ``.claude/rules/persona-distance-metrics.md`` lines 18-30 + the
    reference ``issue444_persona_distance_topic.py::_js_from_logprobs`` (lines
    155-164). For two ``(n_pos, vocab)`` log-probability tensors:

    1. ``log_m = logsumexp(log_p_a, log_p_b) - log 2``  — numerically stable
       per-position mixture log-density (vocab-aligned with the inputs).
    2. ``kl_pos_a[t] = Σ_v exp(log_p_a[t, v]) · (log_p_a[t, v] - log_m[t, v])``
       — per-position contribution to ``KL(p_A || m)``, in nats. Always ≥ 0
       (no realized-token sampling noise; the per-position term is exact).
    3. Symmetrically for ``kl_pos_b``.
    4. ``js_pos = 0.5 · (kl_pos_a + kl_pos_b)`` — per-position JS in nats.

    Base-2 conversion (`/ math.log(2)`) is deferred to the aggregation /
    output layer so the cache stores the canonical nat-domain values; only
    the *final* per-pair JS in the output JSON is in base-2 bits (bounded
    [0, 1]).

    Parameters
    ----------
    log_p_a, log_p_b : torch.Tensor
        Shape ``(n_pos, vocab)``, fp32, log-probabilities (row-sums to 0 in
        log space; i.e. ``log_softmax`` output).

    Returns
    -------
    js_pos, kl_pos_a, kl_pos_b : tuple of torch.Tensor
        Each shape ``(n_pos,)``, fp32 — per-position JS / KL contributions
        in nats. JS ≥ 0 and JS ≤ ln 2 in nats (bounded).
    """
    if log_p_a.shape != log_p_b.shape:
        raise ValueError(f"log_p_a / log_p_b shape mismatch: {log_p_a.shape} vs {log_p_b.shape}")
    if log_p_a.dim() != 2:
        raise ValueError(f"expected 2-D (n_pos, vocab) tensors; got dim={log_p_a.dim()}")
    # Numerically stable mixture log-density: log_m = logsumexp(log p_a, log p_b) - log 2.
    stacked = torch.stack([log_p_a, log_p_b], dim=0)  # (2, n_pos, vocab)
    log_m = torch.logsumexp(stacked, dim=0) - math.log(2.0)  # (n_pos, vocab)
    # Per-position KL contributions: Σ_v p(v) · (log p(v) - log m(v)).
    # Equivalent to Σ_v exp(log p(v)) · (log p(v) - log m(v)).
    p_a = log_p_a.exp()
    p_b = log_p_b.exp()
    kl_pos_a = (p_a * (log_p_a - log_m)).sum(dim=-1)  # (n_pos,)
    kl_pos_b = (p_b * (log_p_b - log_m)).sum(dim=-1)  # (n_pos,)
    js_pos = 0.5 * (kl_pos_a + kl_pos_b)
    return js_pos, kl_pos_a, kl_pos_b


def js_closed_form_two_vocab_toy() -> tuple[float, float, float]:
    """Closed-form JS reference for the canonical 2-vocab toy case.

    For ``p_A = [0.5, 0.5]``, ``p_B = [1.0, 0.0]``:

    * ``m = [0.75, 0.25]``
    * ``KL(p_A || m) = 0.5 · log(0.5 / 0.75) + 0.5 · log(0.5 / 0.25)``
      ``           = 0.5 · log(2/3) + 0.5 · log(2)``
    * ``KL(p_B || m) = 1.0 · log(1.0 / 0.75) + 0.0``
      ``           = log(4/3)``
    * ``JS_nats = 0.5 · (KL(p_A||m) + KL(p_B||m))``
    * ``JS_bits = JS_nats / log(2)``

    Returns
    -------
    (js_nats, kl_a_nats, kl_b_nats) — all in nats.
    """
    p_a = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    p_b = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    # log(0) → use a tiny floor only for the closed-form reference (the per_position
    # routine never sees a hard zero because log_softmax outputs are bounded).
    log_p_a = p_a.clamp_min(1e-300).log()
    log_p_b = p_b.clamp_min(1e-300).log()
    m = 0.5 * (p_a + p_b)
    log_m = m.clamp_min(1e-300).log()
    kl_a = (p_a * (log_p_a - log_m)).sum(-1).item()
    kl_b = (p_b * (log_p_b - log_m)).sum(-1).item()
    js = 0.5 * (kl_a + kl_b)
    return js, kl_a, kl_b


# ───────────────────────── cache helpers ─────────────────────────


def _cache_key(p: str, q_idx: int, r_idx: int, a: str, b: str) -> str:
    """Canonical cache key string — symmetric in (a, b) via lex-sort.

    The cache stores per-position JS / KL contributions for *one response*
    ``(sample_persona p, probe q_idx, response_idx r_idx)`` and *one
    unordered pair* ``{a, b}``. Because per-position JS / KL are symmetric
    under swapping (A, B) at the position level (mixture is unchanged), we
    canonicalize the key with ``a, b = sorted([a, b])`` so any lookup
    keyed by ordered ``(A, B)`` or ``(B, A)`` resolves to the same row.
    """
    a_canon, b_canon = (a, b) if a <= b else (b, a)
    return f"{p}|{q_idx}|{r_idx}|{a_canon}|{b_canon}"


_CACHE_KEY_SCHEMA = {
    "schema_version": 1,
    "format": "{p}|{q_idx}|{r_idx}|{a}|{b} (pipe-separated; a<=b lex-sorted)",
    "fields": [
        "p — sample persona (one of cond_ids)",
        "q_idx — probe index in the working set",
        "r_idx — response index in 0..R-1",
        "a, b — unordered pair members, sorted lexicographically",
    ],
    "value_shape": {
        "js": "Tensor[n_response_tokens] fp32 nats",
        "kl_a": "Tensor[n_response_tokens] fp32 nats (KL(p_a || m))",
        "kl_b": "Tensor[n_response_tokens] fp32 nats (KL(p_b || m))",
    },
    "notes": [
        "Diagonal (a == b) entries are stored as zero tensors of shape (n_resp,).",
        "Cache is symmetric in (a, b); key lookup canonicalizes via lex sort.",
        "Per-position values are in NATS; base-2 conversion happens at "
        "the JS-matrix-reduction layer.",
    ],
}

# Round-3 fix: numeric (p_idx, q_idx) sort for partial-checkpoint resume.
# Lex sort picks the wrong "latest" under multi-digit P/Q
# (e.g. ``P10_Q149 < P9_Q99`` lex). Used by ``_load_cache_checkpoint``.
#
# Round-4 fix: per-run namespace prefix. The partial filename includes the
# ``cache_out.stem`` so smoke (``logprob_cache_smoke``) and full
# (``logprob_cache``) runs sharing the same ``cache_dir`` do NOT collide:
#   smoke  → logprob_cache_smoke_partial_P{P}_Q{Q}.pt
#   full   → logprob_cache_partial_P{P}_Q{Q}.pt
# (#522 pod-522 incident: full sweep loaded the smoke's R=2/64-tok partials
# as if they were canonical R=8/256-tok entries; killed at 9 min into 57h.)
_PARTIAL_FILENAME_RE = re.compile(r"(?P<stem>.+?)_partial_P(?P<p>\d+)_Q(?P<q>\d+)\.pt$")


def _partial_path(cache_dir: Path, cache_out_stem: str, p_idx: int, q_idx: int) -> Path:
    """Return the per-run partial-cache final path (round-4 namespacing)."""
    return cache_dir / f"{cache_out_stem}_partial_P{p_idx}_Q{q_idx}.pt"


def _save_cache_checkpoint(
    cache: dict,
    response_ids: dict[str, torch.Tensor],
    cache_dir: Path,
    cache_out_stem: str,
    p_idx: int,
    q_idx: int,
    config: dict,
) -> None:
    """Persist the in-progress cache to disk so a mid-run crash recovers.

    Writes a single torch .pt with the full cache + response_ids + config
    snapshot; safe to overwrite atomically by writing to a .tmp file and
    renaming. Per CLAUDE.md "Checkpoint per phase" rule.

    Round-4 fix: ``cache_out_stem`` namespaces the partial filename so two
    runs sharing the same ``cache_dir`` (smoke + full) do NOT collide.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = _partial_path(cache_dir, cache_out_stem, p_idx, q_idx)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
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


def _load_cache_checkpoint(
    cache_dir: Path,
    cache_out_stem: str,
    expected_config: dict | None = None,
) -> tuple[dict, dict[str, torch.Tensor], dict | None]:
    """Resume reader — find the most-recent ``{stem}_partial_P*_Q*.pt``.

    Scans ``cache_dir`` for ``{cache_out_stem}_partial_P{P}_Q{Q}.pt`` files,
    picks the latest by **numeric** (p_idx, q_idx) ordering, loads its
    ``cache`` and ``response_ids`` payloads into a fresh in-memory dict, and
    returns them.

    Round-3 fix: previously sorted lexicographically, so under multi-digit
    P/Q the "latest" pick was wrong (e.g. ``P10_Q149 < P9_Q99`` lex). The
    populator skips already-cached keys, so this was correctness-safe but
    wasted replay time on crash-resume.

    Round-4 fix: the glob is scoped to ``cache_out_stem`` so smoke partials
    are invisible to a full-sweep resume in the same directory (#522 pod-522
    incident). If ``expected_config`` is provided, this also fail-loud
    verifies that the loaded partial's recorded ``config`` matches on
    response-shape-determining knobs (``r``, ``max_new_tokens``,
    ``personas``); a mismatch raises ``RuntimeError``.

    If no partial exists, returns empty dicts and ``None``. The caller
    skips already-cached keys; the logger reports how many entries were
    loaded so an operator can verify resume took.
    """
    if not cache_dir.exists():
        return {}, {}, None
    partials = list(cache_dir.glob(f"{cache_out_stem}_partial_P*_Q*.pt"))
    if not partials:
        return {}, {}, None

    # Sort numerically by (p_idx, q_idx). Fallback to filename for safety if
    # the regex doesn't match (defensive — should never happen for the
    # populator's own files).
    def _sort_key(p: Path) -> tuple[int, int, str]:
        m = _PARTIAL_FILENAME_RE.search(p.name)
        if m is None:
            return (-1, -1, p.name)
        return (int(m.group("p")), int(m.group("q")), p.name)

    partials.sort(key=_sort_key)
    latest = partials[-1]
    logger.info("resume: loading partial cache from %s", latest)
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    cache = payload.get("cache", {})
    response_ids = payload.get("response_ids", {})
    meta = {k: payload.get(k) for k in ("p_idx", "q_idx", "checkpoint_at", "config")}

    # Round-4 defensive correctness: fail-loud config mismatch.  The smoke +
    # full runs differ on ``r``, ``max_new_tokens``, and ``personas`` — even
    # under correct stem namespacing, any future cross-run partial
    # contamination (manual copy, mis-set ``cache_out``, etc.) is caught here
    # instead of silently corrupting JS values.
    if expected_config is not None and meta.get("config") is not None:
        loaded_cfg = meta["config"]
        for key in ("r", "max_new_tokens", "personas"):
            if (
                key in expected_config
                and key in loaded_cfg
                and expected_config[key] != loaded_cfg[key]
            ):
                raise RuntimeError(
                    "resume: partial-cache config mismatch on "
                    f"{key!r}: expected={expected_config[key]!r}, "
                    f"loaded={loaded_cfg[key]!r} (from {latest}). "
                    "Refusing to mix partials across runs with different "
                    "response-shape knobs. Delete the stale partial or "
                    "use a fresh cache_out path."
                )

    logger.info(
        "resume: loaded %d cache entries + %d response_ids (from P=%s Q=%s @ %s)",
        len(cache),
        len(response_ids),
        meta.get("p_idx"),
        meta.get("q_idx"),
        meta.get("checkpoint_at"),
    )
    return cache, response_ids, meta


# ───────────────────────── pipeline ─────────────────────────


def _process_one_response(
    *,
    P: str,
    q_i: int,
    r_i: int,
    resp_ids: torch.Tensor,
    probe: str,
    personas: list[str],
    cache: dict,
    model,
    tokenizer,
    device,
    class_d_rewrites: dict[str, dict[str, str]],
) -> int:
    """One-response work: full-vocab forward under every persona + per-pair JS reduce.

    Mutates ``cache`` in place. Returns the number of teacher-force forwards
    performed (for throughput accounting).
    """
    # If all pair-keys for this response already exist, skip entirely.
    pair_keys = [_cache_key(P, q_i, r_i, P, other) for other in personas]
    if all(k in cache for k in pair_keys):
        return 0

    # Empty response — empty placeholders for every pair containing P.
    if resp_ids.numel() == 0:
        empty = torch.empty(0, dtype=torch.float32)
        for other in personas:
            cache[_cache_key(P, q_i, r_i, P, other)] = {
                "js": empty,
                "kl_a": empty,
                "kl_b": empty,
                "n_resp": 0,
            }
        return len(personas)  # accounting only — no actual forward done

    # Full-vocab forward for THIS response under every persona.
    full_logprobs: dict[str, torch.Tensor] = {}
    n_tf = 0
    for E in personas:
        lp = teacher_force_full_vocab_logprobs(
            model,
            tokenizer,
            cond_id=E,
            probe=probe,
            response_ids=resp_ids,
            device=device,
            class_d_rewrites=class_d_rewrites,
        )
        full_logprobs[E] = lp
        n_tf += 1

    # Per-pair JS / KL via canonical full-vocab reduction; include diagonal.
    n_resp = int(resp_ids.numel())
    for other in personas:
        key = _cache_key(P, q_i, r_i, P, other)
        if key in cache:
            continue
        if other == P:
            cache[key] = {
                "js": torch.zeros(n_resp, dtype=torch.float32),
                "kl_a": torch.zeros(n_resp, dtype=torch.float32),
                "kl_b": torch.zeros(n_resp, dtype=torch.float32),
                "n_resp": n_resp,
            }
            continue
        # Canonical ordering: a = min(P, other), b = max.
        a, b = (P, other) if other >= P else (other, P)
        log_p_a = full_logprobs[a]
        log_p_b = full_logprobs[b]
        js_pos, kl_pos_a, kl_pos_b = per_position_js_kl_from_logprobs(log_p_a, log_p_b)
        cache[key] = {
            "js": js_pos.detach().cpu().contiguous(),
            "kl_a": kl_pos_a.detach().cpu().contiguous(),
            "kl_b": kl_pos_b.detach().cpu().contiguous(),
            "n_resp": n_resp,
        }
    del full_logprobs
    return n_tf


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
    resume: bool = True,
) -> tuple[dict, dict[str, torch.Tensor], dict]:
    """Populate the per-position JS / KL cache (canonical full-vocab RB).

    Outer loop (P sample persona, Q probe, r response):
      1. Sample R responses under P-conditioned model (per probe).
      2. For each response, compute the full-vocab log-softmax under EACH
         persona ``E`` (one forward per ``E`` — 16 forwards for the
         16-persona sweep). Keep all 16 ``(n_resp, vocab)`` tensors in
         CPU memory for the duration of THIS one response.
      3. For each unordered pair ``{A, B}`` containing P, compute the
         per-position JS / KL via ``per_position_js_kl_from_logprobs``
         (full-vocab, exact, in nats); cache the per-position scalar tensors.
      4. Also store a zero-tensor entry for the diagonal ``{P, P}`` so the
         reducer's pre-flight key check passes uniformly.
      5. Discard the full-vocab tensors (free ~20 GB CPU); proceed to the
         next response.

    With ``len(personas) == 16``, ``len(probes) == 200``, ``r == 8``:
      - 16 × 200 × 8 = 25,600 responses
      - 16 forwards per response = 409,600 total teacher-force forwards
      - Cache entries: 25,600 × 16 pairs-containing-P = 409,600 cache rows
        (15 off-diagonal pairs + 1 diagonal = 16 per response)

    Returns (cache, response_ids, throughput_stats).
    """
    torch.manual_seed(seed)

    # Round-4: partial-cache filenames are namespaced by ``cache_out.stem``
    # so smoke + full runs sharing the same dir don't collide.  Build the
    # expected config first so the resume reader can fail-loud on a
    # cross-run partial that slipped through.
    cache_dir = Path(cache_out).parent
    cache_out_stem = Path(cache_out).stem
    started_at = datetime.now(UTC).isoformat()
    config = {
        "personas": personas,
        "n_probes": len(probes),
        "r": r,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "started_at": started_at,
    }

    # Resume: load latest partial if any (stem-scoped + config-checked).
    if resume:
        cache, response_ids, _ = _load_cache_checkpoint(
            cache_dir, cache_out_stem, expected_config=config
        )
    else:
        cache, response_ids = {}, {}
    n_resumed_entries = len(cache)
    n_resumed_responses = len(response_ids)

    n_pairs = len(personas) * len(probes)
    n_tf_total = n_pairs * r * len(personas)
    logger.info(
        "Cross-persona cache plan: %d (P, Q) pair-iters × R=%d × %d eval_P forwards = %d total "
        "(resumed: %d cache entries, %d responses)",
        n_pairs,
        r,
        len(personas),
        n_tf_total,
        n_resumed_entries,
        n_resumed_responses,
    )

    pair_iter = 0
    n_tf_done = 0
    for p_i, P in enumerate(personas):
        for q_i, probe in enumerate(probes):
            # Step 1: sample R responses under P (skip if all already present).
            need_sampling = any(
                _cache_key(P, q_i, r_i, "_response", "_response") not in response_ids
                for r_i in range(r)
            )
            if need_sampling:
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
                    response_ids[_cache_key(P, q_i, r_i, "_response", "_response")] = resp_ids
            else:
                responses = [
                    response_ids[_cache_key(P, q_i, r_i, "_response", "_response")]
                    for r_i in range(r)
                ]

            for r_i, resp_ids in enumerate(responses):
                n_tf_done += _process_one_response(
                    P=P,
                    q_i=q_i,
                    r_i=r_i,
                    resp_ids=resp_ids,
                    probe=probe,
                    personas=personas,
                    cache=cache,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    class_d_rewrites=class_d_rewrites,
                )

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
                    cache_dir=cache_dir,
                    cache_out_stem=cache_out_stem,
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
        "n_resumed_entries": n_resumed_entries,
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
            "key_schema": _CACHE_KEY_SCHEMA,
        },
        tmp,
    )
    tmp.replace(cache_out)
    # Sidecar schema JSON. Filename matches the file-level docstring (`:40`)
    # — round-3 fix: previously wrote ``<cache>.schema.json`` (mismatch).
    schema_path = cache_out.parent / "cache_schema.json"
    schema_path.write_text(json.dumps(_CACHE_KEY_SCHEMA, indent=2))
    logger.info(
        "Wrote final cross-persona cache → %s (%d entries); schema → %s",
        cache_out,
        len(cache),
        schema_path,
    )
    return cache, response_ids, throughput


def assert_cache_coverage(
    cache: dict,
    personas: list[str],
    n_probes: int,
    r: int,
) -> None:
    """Pre-flight key coverage check (round-2 fix Must Fix #2).

    Constructs the full set of expected cache keys and diffs against
    ``cache.keys()``. Raises ``RuntimeError`` with the missing count + 3
    example keys on shortfall.
    """
    expected: set[str] = set()
    for P in personas:
        for q_i in range(n_probes):
            for r_i in range(r):
                for other in personas:
                    expected.add(_cache_key(P, q_i, r_i, P, other))
    have = set(cache.keys())
    missing = expected - have
    if missing:
        sample = sorted(missing)[:3]
        raise RuntimeError(
            f"cross-persona cache missing {len(missing)}/{len(expected)} keys; "
            f"sample missing: {sample!r}"
        )
    logger.info(
        "cache coverage assertion PASS: %d / %d expected keys present.",
        len(have & expected),
        len(expected),
    )


def build_js_matrix(
    cache: dict,
    personas: list[str],
    n_probes: int,
    r: int,
) -> dict:
    """Reduce the per-position cache into the per-pair JS / KL matrix.

    Per pair (A, B) and per probe q:
      - JS estimate uses BOTH sources of evaluation positions:
        responses sampled from A AND responses sampled from B (the
        2R mixture). For each, the cache row gives per-position JS
        scalars (full-vocab exact).
      - Length-normalize per response (positions are i.i.d. within
        a response from the conditioned distribution's perspective);
        then mean across responses.

    All cache values are in nats; the output JS / KL are converted to
    base-2 bits via ``/ log(2)``. JS is bounded [0, 1] in base-2 bits.
    """
    # Pre-flight: assert all expected keys exist.
    assert_cache_coverage(cache, personas, n_probes=n_probes, r=r)

    JS: dict[str, dict[str, float]] = {}
    KL_AB: dict[str, dict[str, float]] = {}
    KL_BA: dict[str, dict[str, float]] = {}
    M_js: dict[str, dict[str, float]] = {}
    per_probe_js: dict[str, dict[str, list[float]]] = {}
    log2 = math.log(2.0)
    for A in personas:
        JS[A] = {}
        KL_AB[A] = {}
        KL_BA[A] = {}
        M_js[A] = {}
        per_probe_js[A] = {}
        for B in personas:
            probe_js_vals: list[float] = []
            probe_kl_ab_vals: list[float] = []
            probe_kl_ba_vals: list[float] = []
            for q in range(n_probes):
                # Responses sampled from A: contribute to JS and to KL(A||M) via kl_a.
                # Returns lists of 1-D tensors (one per non-empty response).
                js_per_resp_a, kl_a_per_resp_a, _ = _stack_per_pos(cache, A, q, r, A, B)
                # Responses sampled from B: contribute to JS and to KL(B||M) via kl_b.
                js_per_resp_b, _, kl_b_per_resp_b = _stack_per_pos(cache, B, q, r, A, B)
                if A == B:
                    # Diagonal — JS = 0, KL_AB = KL_BA = 0 by definition.
                    probe_js_vals.append(0.0)
                    probe_kl_ab_vals.append(0.0)
                    probe_kl_ba_vals.append(0.0)
                    continue
                # Mean-of-means (length-normalized per response, then averaged
                # across the 2R mixture responses). Round-3 fix: this MUST be
                # per-response, NOT a flat mean over concatenated positions —
                # see `.claude/rules/persona-distance-metrics.md` and the
                # canonical `issue444_persona_distance_topic.py:191-202`.
                js_resp_means = [t.mean().item() for t in js_per_resp_a + js_per_resp_b]
                if not js_resp_means:
                    probe_js_vals.append(float("nan"))
                    probe_kl_ab_vals.append(float("nan"))
                    probe_kl_ba_vals.append(float("nan"))
                    continue
                js_nats = sum(js_resp_means) / len(js_resp_means)
                # KL(A||M): RB estimate uses positions sampled from A (the
                # natural Importance / Rao-Blackwell source). Same per-response
                # mean-of-means: mean over positions per response, then mean
                # over the R sample-from-A responses.
                kl_ab_resp_means = [t.mean().item() for t in kl_a_per_resp_a]
                kl_ab_nats = (
                    sum(kl_ab_resp_means) / len(kl_ab_resp_means)
                    if kl_ab_resp_means
                    else float("nan")
                )
                # KL(B||M): symmetrically, positions sampled from B.
                kl_ba_resp_means = [t.mean().item() for t in kl_b_per_resp_b]
                kl_ba_nats = (
                    sum(kl_ba_resp_means) / len(kl_ba_resp_means)
                    if kl_ba_resp_means
                    else float("nan")
                )
                probe_js_vals.append(js_nats / log2)
                probe_kl_ab_vals.append(kl_ab_nats / log2)
                probe_kl_ba_vals.append(kl_ba_nats / log2)
            # Mean over probes (NaN-aware so a single empty response
            # doesn't poison the whole pair).
            JS[A][B] = _nanmean(probe_js_vals)
            KL_AB[A][B] = _nanmean(probe_kl_ab_vals)
            KL_BA[A][B] = _nanmean(probe_kl_ba_vals)
            M_js[A][B] = 1.0 - JS[A][B] if math.isfinite(JS[A][B]) else float("nan")
            per_probe_js[A][B] = probe_js_vals
    return {
        "JS": JS,
        "KL_AB": KL_AB,
        "KL_BA": KL_BA,
        "M_js": M_js,
        "per_probe_js": per_probe_js,
    }


def _stack_per_pos(
    cache: dict,
    P: str,
    q: int,
    r: int,
    a: str,
    b: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Per-response per-position tensors for one (sample_persona P, probe q,
    pair {a, b}) triple.

    Returns three lists of 1-D tensors — ``(js_per_resp, kl_a_per_resp,
    kl_b_per_resp)`` — with one tensor per non-empty response under
    sample-persona ``P``. **Response boundaries are preserved by construction**
    so the downstream reducer can length-normalize per response (mean over
    positions within a response) and then average over the 2R responses, per
    `.claude/rules/persona-distance-metrics.md` and the canonical
    ``issue444_persona_distance_topic.py:191-202`` recipe.

    Round-3 fix: previously this function ``torch.cat``ed the per-response
    tensors into a single flat 1-D tensor, which the reducer then ``.mean()``ed
    token-weighted across the union — biasing the headline JS matrix toward
    longer-response personas. Per-response boundaries are now kept; the cache
    on disk is unchanged.

    Empty per-response entries are dropped (i.e. the returned lists may have
    fewer than ``r`` entries when some sampled responses had zero generated
    tokens). If ``P ∉ {a, b}`` (no cache rows under this sample-persona), all
    three lists are empty.
    """
    js_per_resp: list[torch.Tensor] = []
    kla_per_resp: list[torch.Tensor] = []
    klb_per_resp: list[torch.Tensor] = []
    # First map (a, b) → which of them is P (the sample-persona we have cache
    # rows for) and which is the OTHER. If P ∉ {a, b}, no rows in cache.
    if a == P:
        other = b
    elif b == P:
        other = a
    else:
        return [], [], []
    for r_i in range(r):
        key = _cache_key(P, q, r_i, P, other)
        if key not in cache:
            raise KeyError(
                f"cache miss at {key}; cross-persona cache is incomplete. "
                "(Resume from the partial checkpoint or re-run with --bust-cache.)"
            )
        row = cache[key]
        # kl_a / kl_b in the stored row refer to the CANONICAL (a, b) ordering
        # (lex-sorted at write time): "kl_a" = KL(p_{min(P,other)} || m),
        # "kl_b" = KL(p_{max(P,other)} || m). The reducer needs them mapped
        # to the QUERIED (a, b): if a == min, kl_a = stored kl_a; else swap.
        a_canon = P if other >= P else other
        if a_canon == a:
            kla_t = row["kl_a"]
            klb_t = row["kl_b"]
        else:
            kla_t = row["kl_b"]
            klb_t = row["kl_a"]
        js_t = row["js"]
        if js_t.numel() > 0:
            js_per_resp.append(js_t)
            kla_per_resp.append(kla_t)
            klb_per_resp.append(klb_t)
    return js_per_resp, kla_per_resp, klb_per_resp


def _nanmean(vals: list[float]) -> float:
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


# ───────────────────────── smoke gates (fail-loud) ─────────────────────────


def enforce_smoke_gates(
    reduced: dict,
    personas: list[str],
    *,
    diag_tol_bits: float = 1e-3,
    sym_tol_bits: float = 5e-3,
) -> tuple[float, float]:
    """Round-2 fix Major #4: RAISE on diagonal / symmetry breach.

    The plan §4 Step 2.3 names two binding fail-loud gates for the smoke run:

    1. ``JS[A, A] ≈ 0`` for every A (diagonal). The canonical full-vocab
       per-position JS at the diagonal is exactly 0 (the mixture equals
       both p_A and p_B). Tolerance ``diag_tol_bits = 1e-3`` covers fp32
       roundoff.
    2. ``JS[A, B] ≈ JS[B, A]`` for every (A, B). The canonical estimator
       is symmetric by construction; tolerance ``sym_tol_bits = 5e-3``
       allows a small MC-variance band from the 2R mixture mean.

    Returns ``(max_diag, max_sym_residual)`` (base-2 bits).
    """
    diag_vals = [reduced["JS"][p][p] for p in personas]
    finite_diags = [d for d in diag_vals if math.isfinite(d)]
    diag_max = max((abs(d) for d in finite_diags), default=float("nan"))
    sym_residuals = []
    for i, A in enumerate(personas):
        for B in personas[i + 1 :]:
            ja = reduced["JS"][A][B]
            jb = reduced["JS"][B][A]
            if math.isfinite(ja) and math.isfinite(jb):
                sym_residuals.append(abs(ja - jb))
    max_sym_residual = max(sym_residuals) if sym_residuals else float("nan")
    logger.info(
        "Smoke gates: max |diagonal JS|=%.4g, max |JS[A,B]-JS[B,A]|=%.4g",
        diag_max,
        max_sym_residual,
    )
    if math.isfinite(diag_max) and diag_max > diag_tol_bits:
        raise AssertionError(
            f"DIAGONAL GATE FAIL: max |JS[A,A]|={diag_max:.4g} bits > {diag_tol_bits:.0e}; "
            "canonical full-vocab JS at the diagonal must be ≈ 0."
        )
    if math.isfinite(max_sym_residual) and max_sym_residual > sym_tol_bits:
        raise AssertionError(
            f"SYMMETRY GATE FAIL: max |JS[A,B] - JS[B,A]|={max_sym_residual:.4g} bits "
            f"> {sym_tol_bits:.0e}; canonical JS estimator is symmetric in (A, B)."
        )
    return diag_max, max_sym_residual


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
        help=(
            "Output .pt path for the per-position cache (default lives under "
            "/workspace, NOT eval_results/)."
        ),
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
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume-from-partial-checkpoint (start fresh).",
    )
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

    # 5. Populate the cross-persona per-position cache.
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
        resume=(not args.no_resume),
    )

    # 6. Reduce cache → 16×16 JS / KL matrices.
    reduced = build_js_matrix(cache, personas=personas, n_probes=len(probes), r=args.r)

    # 7. Smoke sanity gates — FAIL-LOUD on breach.
    diag_max, max_sym_residual = enforce_smoke_gates(reduced, personas=personas)

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
            "diagonal_js_max_bits": diag_max,
            "max_symmetry_residual_bits": max_sym_residual,
            "estimator": (
                "canonical full-vocab per-position mixture JS (Rao-Blackwellized; "
                "Amini/Vieira/Cotterell 2025, arXiv 2504.10637) — see "
                ".claude/rules/persona-distance-metrics.md"
            ),
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
