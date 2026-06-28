"""Phase-B geometry of the behavior-induced context-vector shift (issue #685).

Given context vectors ``v_l(C)`` (bare) and ``v_l(C+b)`` (behavior-augmented)
extracted at the last prompt token (Phase A), this module computes the four
metric families that decide H1 (single context-independent behavior direction)
vs H2 (context-dependent shift) vs H0 (negligible shift):

1. **Relative magnitude** — ``||Delta_l(C, b)|| / median_{C != C'} ||v_l(C) - v_l(C')||``.
   The H0 read: is the shift a material fraction of a context swap?
2. **Direction consistency** — mean pairwise cosine of ``{Delta_l(C, b)}_C`` over the
   C(n_context, 2) pairs (RAW pairwise / uncentered — the Deltas are already
   differences, so the shared mean cancels by construction; labeled per the
   persona-distance-metrics two-family rule), plus a mean-subtracted variant.
3. **PC1 variance share** — SVD of the ``(n_context x H)`` Delta matrix, ``s[0]^2 / sum(s^2)``.
4. **Behavior separability** — cosine of mean shift directions across behaviors
   (the ``(n_behavior x n_behavior)`` matrix), via the BANK cosine
   (``compute_cosine_matrix(centering="global_mean")``, #536).

Plus the consistency null (random matched-norm unit-vector pairs in H-dim — what
mean pairwise cosine arises by chance at this dimensionality) and the projection
of each ``Delta_l(C, b)`` onto an independently-computed behavior direction
``u_l(b)`` (Phase B.2, persona-vectors response-mean recipe).
"""

import numpy as np
import torch
import torch.nn.functional as F

from explore_persona_space.analysis.representation_shift import compute_cosine_matrix


def _stack_deltas(
    v_bare: torch.Tensor,
    v_aug: torch.Tensor,
) -> torch.Tensor:
    """``Delta = v_aug - v_bare`` as an ``(n_context, H)`` float tensor.

    Both inputs are ``(n_context, H)`` with rows aligned 1:1 by context.
    """
    assert v_bare.ndim == 2 and v_aug.ndim == 2, (v_bare.shape, v_aug.shape)
    assert v_bare.shape == v_aug.shape, (v_bare.shape, v_aug.shape)
    return v_aug.float() - v_bare.float()


def mean_pairwise_cosine(deltas: torch.Tensor) -> float:
    """Mean cosine over all ``C(n, 2)`` unordered pairs of rows in ``deltas``.

    RAW pairwise (uncentered): the rows are difference vectors, so there is no
    bank to center against — labeled ``raw pairwise (uncentered)`` per the
    two-family rule in ``.claude/rules/persona-distance-metrics.md`` and NEVER
    numerically compared to a bank-cosine value.

    Args:
        deltas: ``(n_context, H)`` shift vectors for one (behavior, layer) cell.

    Returns:
        Mean pairwise cosine as a Python float; ``float("nan")`` if ``n < 2``.
    """
    assert deltas.ndim == 2, deltas.shape
    n = deltas.shape[0]
    if n < 2:
        return float("nan")
    normed = F.normalize(deltas.float(), dim=1)
    cos = normed @ normed.T  # (n, n)
    iu = torch.triu_indices(n, n, offset=1)
    return float(cos[iu[0], iu[1]].mean().item())


def pc1_variance_share(deltas: torch.Tensor) -> float:
    """PC1 variance share of the ``(n_context, H)`` Delta matrix.

    SVD of the (un-centered) Delta matrix; ``s[0]^2 / sum(s^2)``. A high share
    means a single dominant direction explains most of the across-context shift
    variance (H1 signature). Un-centered because a consistent shift direction is
    a property of the Deltas as-is (subtracting the mean would remove exactly the
    common direction H1 is about); the mean-subtracted consistency cosine already
    reports the centered view.

    Args:
        deltas: ``(n_context, H)`` shift vectors.

    Returns:
        PC1 variance share in ``[0, 1]``; ``float("nan")`` if ``n < 2``.
    """
    assert deltas.ndim == 2, deltas.shape
    if deltas.shape[0] < 2:
        return float("nan")
    s = torch.linalg.svdvals(deltas.float())
    sq = s.pow(2)
    total = float(sq.sum().item())
    if total <= 0:
        return float("nan")
    return float((sq[0] / sq.sum()).item())


def relative_magnitude(
    deltas: torch.Tensor,
    v_bare_bank: torch.Tensor,
) -> dict:
    """Per-context ``||Delta|| / median_{C != C'} ||v(C) - v(C')||`` and aggregates.

    The denominator is the median pairwise L2 distance over the bare-context
    bank — the between-context spread that a context swap induces. Computed once
    per (behavior, layer) cell from the bare bank.

    Args:
        deltas: ``(n_context, H)`` shift vectors for the cell.
        v_bare_bank: ``(n_context, H)`` bare context vectors (the same bank used
            for the spread denominator).

    Returns:
        ``{per_context: [r_C ...], mean: float, max: float, median_spread: float}``.
    """
    assert deltas.ndim == 2 and v_bare_bank.ndim == 2, (deltas.shape, v_bare_bank.shape)
    n = v_bare_bank.shape[0]
    assert n >= 2, f"need >=2 contexts for a spread denominator, got {n}"
    # Pairwise L2 over the bare bank.
    bank = v_bare_bank.float()
    dists = torch.cdist(bank, bank)  # (n, n)
    iu = torch.triu_indices(n, n, offset=1)
    pair_dists = dists[iu[0], iu[1]]
    median_spread = float(pair_dists.median().item())
    assert median_spread > 0, "degenerate bare bank (zero between-context spread)"
    per_context = (deltas.float().norm(dim=1) / median_spread).tolist()
    return {
        "per_context": per_context,
        "mean": float(np.mean(per_context)),
        "max": float(np.max(per_context)),
        "median_spread": median_spread,
    }


def behavior_separability_matrix(
    mean_shift_by_behavior: dict[str, torch.Tensor],
) -> tuple[list[str], list[list[float]]]:
    """``(n_behavior x n_behavior)`` BANK cosine of the mean shift directions.

    The mean shift direction of behavior ``b`` is ``mean_C Delta_l(C, b)``. The
    matrix is the bank cosine (``compute_cosine_matrix(centering="global_mean")``,
    #536) over the stacked mean-shift bank.

    Args:
        mean_shift_by_behavior: ``{behavior: (H,) mean-shift vector}`` (already
            mean-pooled over contexts), in the desired row order.

    Returns:
        ``(behavior_names, cosine_matrix)`` where the matrix is a nested list.
    """
    names = list(mean_shift_by_behavior.keys())
    bank = torch.stack([mean_shift_by_behavior[b].float() for b in names])  # (n_b, H)
    cos = compute_cosine_matrix(bank, centering="global_mean")
    return names, cos.tolist()


def consistency_null(
    hidden_dim: int,
    n_context: int,
    n_perm: int = 200,
    seed: int = 42,
) -> dict:
    """Null distribution for the consistency cosine: random matched-direction Deltas.

    For each of ``n_perm`` draws, sample ``n_context`` independent unit vectors in
    ``hidden_dim`` and compute their mean pairwise cosine — i.e. "what mean
    pairwise cosine arises by chance among ``n_context`` random directions at this
    dimensionality?". High-dim random vectors are near-orthogonal, so the null
    concentrates near ``0 +/- 1/sqrt(hidden_dim)``.

    Args:
        hidden_dim: H (3584 for Qwen-2.5-7B).
        n_context: number of contexts (matches the real consistency read's n).
        n_perm: number of null draws.
        seed: RNG seed.

    Returns:
        ``{mean, std, p95, p99, n_perm, hidden_dim, n_context, expected_abs_scale}``
        where ``expected_abs_scale = 1/sqrt(hidden_dim)``.
    """
    assert n_context >= 2, n_context
    g = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(n_perm):
        v = torch.randn(n_context, hidden_dim, generator=g)
        samples.append(mean_pairwise_cosine(v))
    arr = np.asarray(samples, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "n_perm": n_perm,
        "hidden_dim": hidden_dim,
        "n_context": n_context,
        "expected_abs_scale": float(1.0 / np.sqrt(hidden_dim)),
    }


def project_onto_direction(
    deltas: torch.Tensor,
    u: torch.Tensor,
) -> dict:
    """Fraction of each ``||Delta||`` explained by the known direction ``u``.

    Per context ``C``: ``|Delta(C) . u_hat| / ||Delta(C)||`` where ``u_hat`` is the
    unit-normalized known direction (the persona-vectors response-mean behavior
    direction ``u_l(b)``). 1.0 means the shift lies entirely along ``u``; 0 means
    orthogonal.

    Args:
        deltas: ``(n_context, H)`` shift vectors for one (behavior, layer) cell.
        u: ``(H,)`` known behavior direction for the same (behavior, layer).

    Returns:
        ``{per_context: [...], mean: float}``. A context with a (near-)zero Delta
        contributes ``0.0`` (no direction to project) rather than a NaN.
    """
    assert deltas.ndim == 2, deltas.shape
    assert u.ndim == 1 and u.shape[0] == deltas.shape[1], (u.shape, deltas.shape)
    u_hat = F.normalize(u.float(), dim=0)
    fracs = []
    for d in deltas.float():
        norm = float(d.norm().item())
        if norm < 1e-8:
            fracs.append(0.0)
            continue
        fracs.append(abs(float(torch.dot(d, u_hat).item())) / norm)
    return {"per_context": fracs, "mean": float(np.mean(fracs)) if fracs else float("nan")}


def behavior_shift_metrics(
    bare_by_context: dict[str, dict[int, torch.Tensor]],
    aug_by_condition: dict[str, dict[int, torch.Tensor]],
    context_names: list[str],
    behaviors: list[str],
    layers: list[int],
    *,
    known_directions: dict[tuple[str, int], torch.Tensor] | None = None,
    null_n_perm: int = 200,
    null_seed: int = 42,
) -> dict:
    """Name-aligned Phase-B metrics (the production entry point).

    Args:
        bare_by_context: ``{context: {layer: (H,) vector}}`` bare context vectors.
        aug_by_condition: ``{f"{context}__{behavior}": {layer: (H,) vector}}``.
        context_names: ordered bare-context names (also the Delta row order).
        behaviors: behavior names.
        layers: layers to report.
        known_directions: optional ``{(behavior, layer): (H,) direction}``.
        null_n_perm / null_seed: consistency-null parameters.

    Returns:
        ``{"cells": {behavior: {str(layer): cell_dict}}, "behavior_separability":
        {str(layer): {"names": [...], "matrix": [[...]]}}, "consistency_null":
        {str(layer): {...}}, "meta": {...}}``. Each ``cell_dict`` carries
        ``relative_magnitude``, ``consistency_cosine_raw``,
        ``consistency_cosine_mean_subtracted``, ``pc1_variance_share`` and (when
        ``known_directions`` is given) ``proj_on_known_direction``.
    """
    # Determine hidden_dim + validate presence.
    any_layer = layers[0]
    hidden_dim = int(bare_by_context[context_names[0]][any_layer].shape[0])
    n_context = len(context_names)

    cells: dict[str, dict[str, dict]] = {b: {} for b in behaviors}
    behavior_sep: dict[str, dict] = {}
    null_by_layer: dict[str, dict] = {}

    for layer in layers:
        # Bare bank for this layer, in context order.
        v_bare_bank = torch.stack(
            [bare_by_context[c][layer].float() for c in context_names]
        )  # (n_context, H)
        mean_shift_by_behavior: dict[str, torch.Tensor] = {}
        for b in behaviors:
            # Delta(C, b) = v_aug[c__b] - v_bare[c], aligned by context order.
            v_aug_bank = torch.stack(
                [aug_by_condition[f"{c}__{b}"][layer].float() for c in context_names]
            )  # (n_context, H)
            deltas = _stack_deltas(v_bare_bank, v_aug_bank)  # (n_context, H)
            assert deltas.shape == (n_context, hidden_dim), deltas.shape

            relmag = relative_magnitude(deltas, v_bare_bank)
            cos_raw = mean_pairwise_cosine(deltas)
            # Mean-subtracted: subtract the across-context mean Delta first.
            deltas_ms = deltas - deltas.mean(dim=0, keepdim=True)
            cos_ms = mean_pairwise_cosine(deltas_ms)
            pc1 = pc1_variance_share(deltas)

            cell = {
                "n_context": n_context,
                "relative_magnitude": relmag,
                "consistency_cosine_raw": cos_raw,
                "consistency_cosine_mean_subtracted": cos_ms,
                "pc1_variance_share": pc1,
            }
            if known_directions is not None and (b, layer) in known_directions:
                cell["proj_on_known_direction"] = project_onto_direction(
                    deltas, known_directions[(b, layer)]
                )
            cells[b][str(layer)] = cell
            mean_shift_by_behavior[b] = deltas.mean(dim=0)

        sep_names, sep_matrix = behavior_separability_matrix(mean_shift_by_behavior)
        behavior_sep[str(layer)] = {"names": sep_names, "matrix": sep_matrix}
        null_by_layer[str(layer)] = consistency_null(
            hidden_dim, n_context, n_perm=null_n_perm, seed=null_seed
        )

    return {
        "cells": cells,
        "behavior_separability": behavior_sep,
        "consistency_null": null_by_layer,
        "meta": {
            "context_names": context_names,
            "behaviors": behaviors,
            "layers": layers,
            "hidden_dim": hidden_dim,
            "n_context": n_context,
            "null_n_perm": null_n_perm,
            "null_seed": null_seed,
        },
    }
