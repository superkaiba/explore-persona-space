"""Chen et al. 2025-inspired persona-vector extraction helpers (issue #368).

Recipe (v3 plan §2.5):

  pvec(trait) = normalize( mean_resp(pos_responses, L) - mean_resp(neg_responses, L) )

Differences from Chen et al. (4 + 1 simplifications, see plan §2.5):
  1. Universal helpful-assistant negative side (5 paraphrases) instead of
     per-trait paired contrastive prompts.
  2. Shared 20 EVAL_QUESTIONS pool instead of trait-evoking questions.
  3. No judge-score response filter.
  4. Fixed L20 headline (with L15 / L25 robustness).
  5. (NEW v3 — R4) Projection-difference variant subtracts a *single fixed*
     ``helpful_test_act`` (per-trait constant offset), not Chen et al.'s
     per-query neutral test prompt. Within any source, projdiff is
     identical to ``pvec_chenstyle_L20`` up to a per-source constant.

Cosine convention: **centered** — subtract ``centroid_mean[L]`` (the mean
over the 11 ``ALL_EVAL_PERSONAS`` positive-side response-mean centroids at
layer L) from BOTH sides before unit-normalize + dot. Matches #142 body
line 87.

The module is import-safe (no GPU touched at import time). The extraction
loop (forward-hook, teacher-forcing) lives in
``scripts/i368_extract_chenstyle_vectors.py``; the helpers here are pure
tensor math + canonical hyperparameter defaults so they can be unit-tested.
"""

from __future__ import annotations

from typing import Literal

import torch

# ── Canonical defaults (plan Reproducibility Card) ──────────────────────────

DEFAULT_LAYERS: tuple[int, ...] = (15, 20, 25)
HEADLINE_LAYER: int = 20
HIDDEN_SIZE: int = 3584  # Qwen2.5-7B-Instruct
N_LAYERS: int = 28  # 0-indexed transformer-block outputs

Aggregation = Literal["mean_response", "last_token"]


# ── Recipe metadata (also referenced by phase1/phase2 analysis scripts) ──────

# 8 new axes per the plan + 1 descriptive collinearity axis (R10 / T10).
# Each row: (axis_name, aggregation, layer, recipe_flavor)
#   recipe_flavor ∈ {"chenstyle", "chenstyle_orthog", "chenstyle_projdiff",
#                    "method_a", "method_b", "pos_only_chenstyle"}
AXIS_SPECS: list[dict] = [
    # 5 chenstyle variants
    {
        "name": "pvec_chenstyle_L20",
        "aggregation": "mean_response",
        "layer": 20,
        "flavor": "chenstyle",
    },
    {
        "name": "pvec_chenstyle_L15",
        "aggregation": "mean_response",
        "layer": 15,
        "flavor": "chenstyle",
    },
    {
        "name": "pvec_chenstyle_L25",
        "aggregation": "mean_response",
        "layer": 25,
        "flavor": "chenstyle",
    },
    {
        "name": "pvec_chenstyle_lasttoken",
        "aggregation": "last_token",
        "layer": 20,
        "flavor": "chenstyle",
    },
    {
        "name": "pvec_chenstyle_orthog",
        "aggregation": "mean_response",
        "layer": 20,
        "flavor": "chenstyle_orthog",
    },
    # 1 projection-difference variant (R4 — degenerate within-source)
    {
        "name": "pvec_chenstyle_L20_projdiff",
        "aggregation": "mean_response",
        "layer": 20,
        "flavor": "chenstyle_projdiff",
    },
    # 2 #216 centroid-only baselines (centered cosine)
    {
        "name": "pcentroid_methodA_L20",
        "aggregation": "last_token",
        "layer": 20,
        "flavor": "method_a",
    },
    {
        "name": "pcentroid_methodB_L20",
        "aggregation": "mean_response",
        "layer": 20,
        "flavor": "method_b",
    },
    # 9th DESCRIPTIVE-ONLY collinearity axis (T10) — NOT included in recipe-
    # agreement 8x8 matrix; reported alongside.
    {
        "name": "pcentroid_chenstyle_pos_only_L20",
        "aggregation": "mean_response",
        "layer": 20,
        "flavor": "pos_only_chenstyle",
    },
]

CANONICAL_8_AXES: list[str] = [a["name"] for a in AXIS_SPECS if a["flavor"] != "pos_only_chenstyle"]
HEADLINE_AXIS: str = "pvec_chenstyle_L20"


# ── Pure tensor math ─────────────────────────────────────────────────────────


def unit_normalize(v: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize a 1-D tensor; guards against zero norm."""
    if v.dim() != 1:
        raise ValueError(f"expected 1-D tensor, got shape {tuple(v.shape)}")
    n = v.norm()
    if n.item() < eps:
        raise ValueError("cannot unit-normalize a zero-norm vector")
    return v / n


def compute_chenstyle_vector(
    pos_centroid: torch.Tensor,
    neg_centroid: torch.Tensor,
) -> torch.Tensor:
    """``pos_centroid - neg_centroid`` unit-normalized.

    Both inputs are 1-D (hidden_dim,) tensors. The pos centroid is the mean
    over (paraphrases × questions × response tokens) of the positive-side
    activations at the chosen layer; neg likewise.
    """
    diff = pos_centroid - neg_centroid
    return unit_normalize(diff)


def compute_orthogonalized_vector(
    chenstyle_vec: torch.Tensor,
    baseline_vec: torch.Tensor,
) -> torch.Tensor:
    """Project ``chenstyle_vec`` onto the subspace orthogonal to ``baseline_vec``.

    ``baseline_vec`` should be unit-normalized. Returns a unit-normalized
    residual. Plan §4.1.1 step 6: the empty-prompt baseline direction is
    extracted once and used to remove generic "helpful-default" structure.
    """
    if chenstyle_vec.dim() != 1 or baseline_vec.dim() != 1:
        raise ValueError("both inputs must be 1-D")
    baseline_unit = unit_normalize(baseline_vec)
    proj = (chenstyle_vec @ baseline_unit) * baseline_unit
    residual = chenstyle_vec - proj
    return unit_normalize(residual)


def centered_cosine(
    v: torch.Tensor,
    w: torch.Tensor,
    centroid_mean: torch.Tensor,
) -> float:
    """Centered cosine similarity (T1 plan convention).

    Subtract ``centroid_mean`` from both ``v`` and ``w`` before unit-normalize
    + dot. ``centroid_mean`` is the mean over the 11 ``ALL_EVAL_PERSONAS``
    positive-side response-mean centroids at the same layer. This matches
    #142 body line 87 — the recipe whose published Spearman ρ = 0.567 on
    the 50-pair leakage table is the comparison floor for H2.
    """
    if v.dim() != 1 or w.dim() != 1 or centroid_mean.dim() != 1:
        raise ValueError(
            f"all three tensors must be 1-D; got shapes v={tuple(v.shape)}, "
            f"w={tuple(w.shape)}, centroid_mean={tuple(centroid_mean.shape)}"
        )
    v_c = unit_normalize(v - centroid_mean)
    w_c = unit_normalize(w - centroid_mean)
    return float((v_c @ w_c).item())


def compute_global_centroid_mean(
    centroids: dict[str, torch.Tensor],
    layer: int,
) -> torch.Tensor:
    """Mean over a {name: (n_layers, hidden_dim) or (hidden_dim,)} dict.

    Used for centered-cosine convention. Plan §4.2.1: centroid_mean[L] = mean
    over the 11 ALL_EVAL_PERSONAS positive-side response-mean centroids at
    layer L.

    Accepts two centroid storage shapes:
      - (n_layers, hidden_dim): per-layer centroids; we select ``layer``.
      - (hidden_dim,): already layer-specific; ``layer`` arg is ignored
                       but must be passed for API consistency.

    Returns a 1-D (hidden_dim,) tensor.
    """
    if not centroids:
        raise ValueError("compute_global_centroid_mean: empty centroids dict")

    stacked: list[torch.Tensor] = []
    for name, c in centroids.items():
        if c.dim() == 2:
            # (n_layers, hidden_dim) — caller passes layer index directly.
            stacked.append(c[layer].float())
        elif c.dim() == 1:
            stacked.append(c.float())
        else:
            raise ValueError(
                f"centroids[{name!r}] has unsupported shape {tuple(c.shape)} (expected 1-D or 2-D)"
            )
    return torch.stack(stacked).mean(dim=0)


def projdiff_score(
    pvec: torch.Tensor,
    test_act: torch.Tensor,
    helpful_test_act: torch.Tensor,
    centroid_mean: torch.Tensor,
) -> float:
    """Chen et al. projection-difference (R4 disclosed degeneracy).

    Returns ``centered_cosine(pvec, test_act) - centered_cosine(pvec,
    helpful_test_act)``.

    R4 disclosure: ``helpful_test_act`` is a single fixed vector across all
    test prompts (the mean of the 5 helpful-assistant negative-side
    responses' L20 activations). Within any source S the subtracted term is
    a constant scalar per S, so within-source target rankings are identical
    to plain ``pvec_chenstyle_L20`` by construction. This is the operational
    consequence of simplification #5 in §2.5 — the operationalization differs
    from Chen et al.'s per-query neutral.
    """
    return centered_cosine(pvec, test_act, centroid_mean) - centered_cosine(
        pvec, helpful_test_act, centroid_mean
    )


# ── Activation aggregation (pure-math helpers used by the extraction loop) ──


def mean_response_aggregate(
    layer_response_vecs: list[torch.Tensor],
) -> torch.Tensor:
    """Stack 1-D (hidden,) tensors and average.

    ``layer_response_vecs`` is the list of per-(paraphrase × question)
    mean-of-response-tokens vectors at a single layer. Average gives the
    trait centroid at that layer.
    """
    if not layer_response_vecs:
        raise ValueError("mean_response_aggregate: empty list")
    return torch.stack(layer_response_vecs).mean(dim=0)


def last_token_aggregate(
    layer_response_vecs: list[torch.Tensor],
) -> torch.Tensor:
    """Same shape contract as ``mean_response_aggregate``.

    The 'last token' variant is implemented identically at the centroid
    level — each ``layer_response_vecs[i]`` is already the per-response
    last-token vector (the extraction loop varies the aggregation upstream).
    """
    return mean_response_aggregate(layer_response_vecs)
