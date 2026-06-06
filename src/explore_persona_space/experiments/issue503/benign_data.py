# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (×, →, —, ρ, ÷) in scientific docstrings + logs + regex.
"""Bucket D — benign-data → broad-unsafety selectors (plan v2 §4.5).

Implements the He et al. (arXiv 2404.01099) selector arm: starting from
Alpaca (52k) + Dolly (15k) filtered of safety markers per their §3.1,
each selector picks the top-K=100 most "harmful-prone" benign data
points; SFT a LoRA on each top-100 set; measure AdvBench ASR. The
Bucket D rows form a selector-level table (5 selectors × 3 seeds = 15
adapter-level data points; MF-2(a) selector-level unit of analysis).

The 5 selectors:

- **D0 = random 100** (negative-control baseline, no selection).
- **D1 = representation matching** (He et al. Eq. 1): cosine between
  the datapoint's hidden-state representation and the average
  representation of the harmful anchor set, at a chosen layer.
- **D2 = gradient bidirectional** (He et al. Eq. 2): per-datapoint
  loss-gradient inner product against the gradient computed on the
  harmful anchor, with bidirectional sign-handling.
- **D3 = #503 cosine** — the predictor under test. Cosine between the
  datapoint's residual at L25 newline-after-`assistant` and the
  averaged residual of the 10 illegal-activities Pure-bad anchor at
  the same (L, p).
- **D4 = format selector**: random 100 from the "all-list" subset +
  random 100 from the "all-math" subset (He et al. §4.2 keyword
  filter). Concretely: select 50 from each of "list" + "math" so the
  output is one selector with k=100.

Anchor sets per plan §4.5 (He et al. §3.1 default):

- ``D_harmful``: 10 illegal-activities Pure-bad examples (Qi et al.
  2023 / He et al. default).
- ``D_safe``: same 10 instructions with refusal/safe completions
  (used by D2 for bidirectional anchoring).

Method-independence pre-condition (MF-5). Before any H7-7b
interpretation, the dispatcher MUST compute
``Spearman(D1_ranking, D3_ranking)`` over the SAME Alpaca/Dolly corpus
at K=100. If ρ(D1, D3) > 0.85, demote H7-7b to "the #503 cosine
reproduces He et al.'s representation ranking" with NO mechanism
claim. The check is implemented in ``method_independence_check``.

Plan reference: tasks/running/503/plans/v2.md §4.5.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SelectorId = Literal["D0_random", "D1_representation", "D2_gradient", "D3_cosine", "D4_format"]

ALL_SELECTORS: tuple[SelectorId, ...] = (
    "D0_random",
    "D1_representation",
    "D2_gradient",
    "D3_cosine",
    "D4_format",
)

# Per plan §4.5: top-K = 100 per selector.
DEFAULT_TOP_K: int = 100

# Per plan §4.5: 3 random seeds {0, 42, 137} (He et al. used 5; budget tradeoff).
DEFAULT_SEEDS: tuple[int, ...] = (0, 42, 137)

# Default anchor layer / position — must match the predictor (#468/#503 canonical).
DEFAULT_LAYER: int = 25
DEFAULT_POSITION_NAME: str = "p5"  # newline-after-assistant; #468 canonical

# Method-independence ceiling — MF-5
METHOD_INDEPENDENCE_RHO_CEIL: float = 0.85

# He et al. safety-keyword filter (plan §4.5 — Alpaca/Dolly filter step).
# These phrases are scrubbed BEFORE selection so the selector doesn't pick a
# explicit-refusal training row trivially.
HE_SAFETY_MARKERS_REGEX: tuple[str, ...] = (
    r"I cannot provide guidance",
    r"It is not appropriate",
    r"As an AI language model",
    r"I cannot assist with",
    r"I'm sorry,? (?:but )?I can(?:not|'t)",
    r"It is illegal",
    r"I am not able to",
    r"I do not condone",
)

# He et al. format-keyword filter for D4 (plan §4.5 row D4).
LIST_MARKERS_REGEX: tuple[str, ...] = (
    r"^\s*[-*+•]\s",  # bullet
    r"^\s*\d+\.\s",  # numbered
)
MATH_MARKERS_REGEX: tuple[str, ...] = (
    r"[\d.]+\s*[+\-*/=×÷]\s*[\d.]+",
    r"\b(?:sum|product|mean|average|median|integral|derivative)\b",
    r"\\(?:frac|sum|prod|int)",  # latex
)


# ── Data shapes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BenignDatapoint:
    """One Alpaca/Dolly row after safety-filtering."""

    datapoint_id: str
    source: Literal["alpaca", "dolly", "gsm8k"]
    instruction: str
    output: str
    # Optional precomputed scores (filled by each selector); kept as a frozen
    # field-less mapping so the dataclass stays cheap to construct in bulk.


@dataclass
class SelectorResult:
    """The output of running one selector over the corpus.

    Round-2 Rec 4: a selector now carries BOTH the top-K ``selected_ids``
    (the rows it picked for SFT) AND ``score_per_corpus_row`` — the
    selector's score for EVERY row in the filtered corpus, in
    ``corpus_ids`` order. The full-corpus score vector is what
    ``method_independence_check`` needs to compute Spearman ρ between D1
    and D3 over the same ordered set of items; the top-K alone is
    insufficient (the two selectors can pick disjoint top-Ks while
    perfectly correlating over the full corpus).
    """

    selector_id: SelectorId
    selected_ids: list[str]
    # Per-datapoint score the selector ranked on; same order as ``selected_ids``.
    # For D0 (random) the score is the random tiebreaker.
    scores: list[float]
    top_k: int
    # Round-2 Rec 4: the selector's score for every row in the FILTERED
    # corpus, in ``corpus_ids`` order. Required for the MF-5 full-corpus
    # ρ check; None only for legacy callers that don't compute MF-5.
    score_per_corpus_row: list[float] | None = None
    # The corpus_ids that ``score_per_corpus_row`` is keyed against,
    # one per index. Same length as ``score_per_corpus_row``.
    corpus_ids: list[str] | None = None


# ── Filter step ──────────────────────────────────────────────────────────────


def filter_safety_markers(rows: Sequence[BenignDatapoint]) -> list[BenignDatapoint]:
    """Drop rows whose output matches any He et al. safety-marker regex.

    He et al. §3.1: the un-filtered Alpaca/Dolly corpus contains explicit
    refusals + safety boilerplate that trivially elevate "harmless" cosine
    scores. Filtering removes the obvious explicit-refusal rows before
    selection so the selectors are picking from genuinely benign content.
    """
    patterns = [re.compile(p, flags=re.IGNORECASE) for p in HE_SAFETY_MARKERS_REGEX]
    out: list[BenignDatapoint] = []
    for r in rows:
        text = (r.output or "") + " " + (r.instruction or "")
        if not any(p.search(text) for p in patterns):
            out.append(r)
    return out


# ── Selectors ────────────────────────────────────────────────────────────────


def select_random(
    rows: Sequence[BenignDatapoint], top_k: int = DEFAULT_TOP_K, seed: int = 0
) -> SelectorResult:
    """D0: uniform random selection (baseline)."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), size=min(top_k, len(rows)), replace=False)
    scores = rng.random(size=len(indices)).tolist()
    selected_ids = [rows[int(i)].datapoint_id for i in indices]
    return SelectorResult(
        selector_id="D0_random",
        selected_ids=selected_ids,
        scores=scores,
        top_k=top_k,
    )


def _cosine_rank_by_anchor(datapoint_reprs: np.ndarray, anchor_repr: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row's representation against the anchor mean."""
    assert datapoint_reprs.ndim == 2, datapoint_reprs.shape
    assert anchor_repr.ndim == 1, anchor_repr.shape
    a = datapoint_reprs / (np.linalg.norm(datapoint_reprs, axis=1, keepdims=True) + 1e-12)
    b = anchor_repr / (np.linalg.norm(anchor_repr) + 1e-12)
    return a @ b


def select_representation(
    rows: Sequence[BenignDatapoint],
    datapoint_reprs: np.ndarray,
    anchor_reprs: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
) -> SelectorResult:
    """D1: He et al. Eq. 1 representation matching.

    ``datapoint_reprs``: (N, d) hidden-state representations for each row
    (typically end-of-instruction token at a chosen layer).
    ``anchor_reprs``: (n_anchor, d) representations of the harmful anchor set;
    we take the mean to compare against.

    Round-2 Rec 4: emits ``score_per_corpus_row`` + ``corpus_ids`` for the
    full filtered corpus (not just the top-K) so MF-5 can compute the
    method-independence ρ.
    """
    if len(rows) != datapoint_reprs.shape[0]:
        raise ValueError(
            f"rows (n={len(rows)}) and datapoint_reprs (n={datapoint_reprs.shape[0]}) mismatch"
        )
    anchor_mean = anchor_reprs.mean(axis=0)
    cos = _cosine_rank_by_anchor(datapoint_reprs, anchor_mean)
    order = np.argsort(-cos)[:top_k]  # most-similar-first
    selected_ids = [rows[int(i)].datapoint_id for i in order]
    scores = cos[order].tolist()
    return SelectorResult(
        selector_id="D1_representation",
        selected_ids=selected_ids,
        scores=scores,
        top_k=top_k,
        score_per_corpus_row=cos.tolist(),
        corpus_ids=[r.datapoint_id for r in rows],
    )


def select_gradient_bidirectional(
    rows: Sequence[BenignDatapoint],
    datapoint_grad_inner: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
) -> SelectorResult:
    """D2: He et al. Eq. 2 gradient bidirectional selector.

    Receives ``datapoint_grad_inner`` of shape (N, 2): column 0 is the
    inner product with the harmful-anchor loss gradient; column 1 is the
    inner product with the safe-anchor loss gradient (bidirectional).
    Score = ``column0 − column1``.

    Caller is responsible for the gradient computation on a base model;
    this function consumes the precomputed pair. He et al. uses
    ``loss_harmful_grad ⋅ instance_grad − loss_safe_grad ⋅ instance_grad``.
    """
    if datapoint_grad_inner.shape != (len(rows), 2):
        raise ValueError(
            f"datapoint_grad_inner shape {datapoint_grad_inner.shape}; expected ({len(rows)}, 2)"
        )
    bidir = datapoint_grad_inner[:, 0] - datapoint_grad_inner[:, 1]
    order = np.argsort(-bidir)[:top_k]
    selected_ids = [rows[int(i)].datapoint_id for i in order]
    scores = bidir[order].tolist()
    return SelectorResult(
        selector_id="D2_gradient",
        selected_ids=selected_ids,
        scores=scores,
        top_k=top_k,
    )


def select_cosine_503(
    rows: Sequence[BenignDatapoint],
    datapoint_residuals: np.ndarray,
    anchor_residual_mean: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
) -> SelectorResult:
    """D3: #503 cosine against the 10-row Pure-bad anchor.

    ``datapoint_residuals``: (N, d) residual-stream activations at the
    canonical (L=25, p5 = newline-after-`assistant`) read point per
    datapoint.
    ``anchor_residual_mean``: (d,) average residual across the 10
    Pure-bad illegal-activities anchors at the same (L, p).

    This is the #503 cosine predictor applied to selection-time scoring
    of single benign datapoints (NOT the K=8 persona-vector cosine; the
    plan calls it a paired-but-distinct read).

    Round-2 Rec 4: emits ``score_per_corpus_row`` + ``corpus_ids`` over
    the full filtered corpus so MF-5 can pair D3 with D1.
    """
    if len(rows) != datapoint_residuals.shape[0]:
        raise ValueError(
            f"rows (n={len(rows)}) and datapoint_residuals "
            f"(n={datapoint_residuals.shape[0]}) mismatch"
        )
    cos = _cosine_rank_by_anchor(datapoint_residuals, anchor_residual_mean)
    order = np.argsort(-cos)[:top_k]
    selected_ids = [rows[int(i)].datapoint_id for i in order]
    scores = cos[order].tolist()
    return SelectorResult(
        selector_id="D3_cosine",
        selected_ids=selected_ids,
        scores=scores,
        top_k=top_k,
        score_per_corpus_row=cos.tolist(),
        corpus_ids=[r.datapoint_id for r in rows],
    )


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    compiled = [re.compile(p, flags=re.IGNORECASE | re.MULTILINE) for p in patterns]
    return any(p.search(text) for p in compiled)


def select_format(
    rows: Sequence[BenignDatapoint],
    top_k: int = DEFAULT_TOP_K,
    seed: int = 0,
) -> SelectorResult:
    """D4: format selector (random 50 from all-list + random 50 from all-math).

    He et al. §4.2 keyword-filter approach: partition the corpus by output
    format (list-formatted vs math-formatted vs other) and sample randomly
    within each format. We take 50 from each of the two formats so the
    total is ``top_k``.
    """
    half = top_k // 2
    list_rows = [r for r in rows if _matches_any(r.output or "", LIST_MARKERS_REGEX)]
    math_rows = [r for r in rows if _matches_any(r.output or "", MATH_MARKERS_REGEX)]
    if len(list_rows) < half or len(math_rows) < half:
        logger.warning(
            "D4 format-selector pool small: list=%d, math=%d, required=%d each",
            len(list_rows),
            len(math_rows),
            half,
        )
    rng = np.random.default_rng(seed)
    list_pick = rng.choice(len(list_rows), size=min(half, len(list_rows)), replace=False)
    math_pick = rng.choice(len(math_rows), size=min(half, len(math_rows)), replace=False)
    selected = [list_rows[int(i)] for i in list_pick] + [math_rows[int(i)] for i in math_pick]
    return SelectorResult(
        selector_id="D4_format",
        selected_ids=[r.datapoint_id for r in selected],
        # D4 has no real score; use a uniform random tiebreaker for parity.
        scores=rng.random(size=len(selected)).tolist(),
        top_k=top_k,
    )


# ── Method-independence check (MF-5) ─────────────────────────────────────────


def spearman_rank_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Standard Spearman ρ on two equal-length score lists.

    Returns 0.0 on degenerate input (all-tied ranks → variance zero;
    surfacing as "no ordinal signal" rather than nan).
    """
    if len(a) != len(b):
        raise ValueError(f"a (n={len(a)}) and b (n={len(b)}) length mismatch")
    if len(a) < 2:
        return 0.0
    ra = _to_ranks(a)
    rb = _to_ranks(b)
    n = len(a)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    denom = (var_a * var_b) ** 0.5
    if denom < 1e-12:
        return 0.0
    return float(num / denom)


def _to_ranks(xs: Sequence[float]) -> list[float]:
    """Average-rank (ties handled by averaging) — standard Spearman input."""
    indexed = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def method_independence_check(
    d1: SelectorResult,
    d3: SelectorResult,
    *,
    rho_ceiling: float = METHOD_INDEPENDENCE_RHO_CEIL,
) -> dict:
    """MF-5 method-independence diagnostic — full-corpus Spearman ρ.

    Round-2 Rec 4 (reconciler-binding rewrite): REQUIRES D1 and D3 score
    vectors over the SAME corpus rows in the SAME order. The v1
    implementation fell back to "compute ρ over the intersection" when
    the selectors picked different top-K rows — but that is EXACTLY the
    case where the diagnostic matters: two methods can pick disjoint
    top-Ks while perfectly correlating over the full corpus. The
    intersection-only path silently turned the missing-coverage bug into
    a 0-element or tiny-n ρ.

    The two selectors are paired (representation cosine vs residual
    cosine, both against a harmful anchor); their methodological
    independence is unknown a priori. If ρ > rho_ceiling (default 0.85),
    H7-7b is DEMOTED to "D3 reproduces D1's ranking" with no mechanism
    claim and is REMOVED from the H8 headline.

    Raises ``ValueError`` per CLAUDE.md "Fail fast — never hide failures"
    when:
        - either selector lacks ``score_per_corpus_row`` (the new Rec-4
          field); the upstream selector pipeline must emit it.
        - the two selectors' ``corpus_ids`` differ in ANY way (length,
          order, or set membership).

    Returns ``{"rho": float, "n_used": int, "comparison_mode": "full",
    "rho_ceiling": float, "demote_h7_7b": bool, "verdict": str}``.
    """
    if d1.score_per_corpus_row is None or d1.corpus_ids is None:
        raise ValueError(
            "MF-5 requires full-corpus score vectors; D1's score_per_corpus_row "
            "is None. The selector pipeline (scripts/issue503_benign_data_select.py) "
            "must populate score_per_corpus_row + corpus_ids on every D1/D3 "
            "SelectorResult — see benign_data.SelectorResult docstring."
        )
    if d3.score_per_corpus_row is None or d3.corpus_ids is None:
        raise ValueError(
            "MF-5 requires full-corpus score vectors; D3's score_per_corpus_row "
            "is None. The selector pipeline (scripts/issue503_benign_data_select.py) "
            "must populate score_per_corpus_row + corpus_ids on every D1/D3 "
            "SelectorResult — see benign_data.SelectorResult docstring."
        )
    if d1.corpus_ids != d3.corpus_ids:
        # Detect the most informative divergence type for the error.
        n_d1 = len(d1.corpus_ids)
        n_d3 = len(d3.corpus_ids)
        if n_d1 != n_d3:
            raise ValueError(
                f"MF-5 requires full-corpus score vectors; selectors emitted "
                f"different/partial id sets (D1 n={n_d1}, D3 n={n_d3}). "
                f"Both selectors must score the SAME filtered corpus in the "
                f"SAME order."
            )
        # Same length but different ordering or membership.
        if set(d1.corpus_ids) != set(d3.corpus_ids):
            n_only_d1 = len(set(d1.corpus_ids) - set(d3.corpus_ids))
            raise ValueError(
                f"MF-5 requires full-corpus score vectors; selectors emitted "
                f"different/partial id sets ({n_only_d1} ids in D1 not in D3; "
                f"n_total D1={n_d1}, D3={n_d3}). Both selectors must score "
                f"the SAME filtered corpus."
            )
        raise ValueError(
            f"MF-5 requires full-corpus score vectors; D1 and D3 corpus_ids "
            f"are the same set but in different order (n={n_d1}). The "
            f"selector pipeline must emit corpus_ids in identical order — "
            f"the order is part of the contract."
        )
    if len(d1.score_per_corpus_row) != len(d1.corpus_ids):
        raise ValueError(
            f"D1.score_per_corpus_row has length {len(d1.score_per_corpus_row)} "
            f"but D1.corpus_ids has length {len(d1.corpus_ids)} — the two must "
            f"match (one score per corpus row)."
        )
    if len(d3.score_per_corpus_row) != len(d3.corpus_ids):
        raise ValueError(
            f"D3.score_per_corpus_row has length {len(d3.score_per_corpus_row)} "
            f"but D3.corpus_ids has length {len(d3.corpus_ids)} — the two must "
            f"match (one score per corpus row)."
        )

    rho = spearman_rank_correlation(d1.score_per_corpus_row, d3.score_per_corpus_row)
    n_used = len(d1.score_per_corpus_row)
    demote = rho > rho_ceiling
    return {
        "rho": float(rho),
        "n_used": n_used,
        "comparison_mode": "full_corpus",
        "rho_ceiling": float(rho_ceiling),
        "demote_h7_7b": demote,
        "verdict": "DEMOTE_H7_7B_TO_D3_REPRODUCES_D1" if demote else "INDEPENDENT_METHODS",
    }


# ── Selector-level row builder ───────────────────────────────────────────────


@dataclass
class BenignDataRow:
    """One row of the selector-level Bucket D table (MF-2(a) unit of analysis).

    Per plan §4.5: 15 rows = 5 selectors × 3 seeds. Each row carries:
      - selector top-100 mean cosine at L25/p5 (the predictor read)
      - selector adapter ASR delta vs random-100 baseline (the outcome)
    """

    selector_id: SelectorId
    seed: int
    mean_cosine_L25_p5: float
    asr_delta_vs_d0: float
    asr_absolute: float
    # Number of AdvBench prompts judged for ASR (typically 520).
    n_advbench: int
    notes: str = ""


def bucket_d_unit_of_analysis_note() -> str:
    """Returns the standing note from plan §4.5 MF-2(a) about Bucket D."""
    return (
        "Bucket D unit of analysis: SELECTOR-LEVEL (n=15 = 5 selectors x 3 seeds). "
        "Per-datapoint ASR contribution is NOT observable from a multi-datapoint "
        "fine-tune; the gate is Spearman rho point estimate + 95% bootstrap CI, "
        "with NO p-value gate at n=15. H7-7b threshold +0.30 from v1 is removed."
    )
