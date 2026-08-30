#!/usr/bin/env python3
"""Issue #2643: sparse gradient-pursuit compression of behavior forecasts.

This is the behavior-facing analogue of the recent issue-1482 experiment.  A
realized-answer SAE direction is first transported through the frozen
context-SAE -> answer-SAE map.  Signed greedy pursuit then approximates that
*mapped score* with a small set of context-SAE features.  Atom selection uses
maximum normalized residual correlation and every selected support is jointly
ridge-refit.  The registered controls retain the largest coefficients of the
factorized map's local linearization, with either fixed weights or the same
joint refit.

The sparse edges approximate an observational predictor.  They are not causal
routes, and pursuit is never fit on evaluation rows or directly on their
behavior labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

K_LADDER_DEFAULT = (1, 2, 4, 8, 16)
REFIT_RIDGE_RELATIVE_DEFAULT = 1e-3


@dataclass(frozen=True)
class PursuitCheckpoint:
    """One signed-pursuit solution at a requested support size."""

    support: np.ndarray
    coefficients: np.ndarray


def refit_support(
    design: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    *,
    ridge_relative: float = REFIT_RIDGE_RELATIVE_DEFAULT,
) -> np.ndarray:
    """Joint ridge refit whose penalty is relative to selected-atom energy."""

    s = np.asarray(support, dtype=np.int64)
    if s.ndim != 1 or len(s) == 0:
        raise ValueError("support must be a non-empty vector")
    if not np.isfinite(ridge_relative) or ridge_relative < 0:
        raise ValueError("ridge_relative must be finite and nonnegative")
    z = np.asarray(design[:, s], dtype=np.float64)
    gram = z.T @ z
    scale = float(np.trace(gram) / len(s))
    ridge = float(ridge_relative) * max(scale, 1e-24)
    return np.linalg.solve(
        gram + ridge * np.eye(len(s)), z.T @ np.asarray(target, dtype=np.float64)
    )


def signed_gradient_pursuit(
    design: np.ndarray,
    target: np.ndarray,
    *,
    max_k: int,
    checkpoints: Sequence[int] = K_LADDER_DEFAULT,
    min_norm: float = 1e-10,
    ridge_relative: float = REFIT_RIDGE_RELATIVE_DEFAULT,
) -> dict[int, PursuitCheckpoint]:
    """Greedy signed atom selection with a joint least-squares refit.

    This deliberately matches issue #1482's selection/refit convention:
    select the unused atom with largest absolute normalized residual
    correlation, then jointly refit the complete support.
    """

    z = np.asarray(design, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    if z.ndim != 2 or y.ndim != 1 or z.shape[0] != y.shape[0]:
        raise ValueError(f"bad pursuit shapes: design={z.shape}, target={y.shape}")
    if max_k < 1 or max_k > z.shape[1]:
        raise ValueError(f"max_k={max_k} outside [1, {z.shape[1]}]")
    wanted = tuple(sorted({int(k) for k in checkpoints if 1 <= int(k) <= max_k}))
    if not wanted:
        raise ValueError("no checkpoints fall inside the pursuit range")

    norms = np.sqrt(np.sum(z.astype(np.float64) ** 2, axis=0))
    usable = norms > min_norm
    if int(usable.sum()) < max_k:
        raise ValueError(f"only {int(usable.sum())} nonconstant atoms for max_k={max_k}")

    residual = y.copy()
    selected: list[int] = []
    taken = ~usable
    out: dict[int, PursuitCheckpoint] = {}
    for step in range(1, max_k + 1):
        corr = np.asarray(z.T @ residual, dtype=np.float64)
        corr /= np.maximum(norms, min_norm)
        corr[taken] = 0.0
        atom = int(np.argmax(np.abs(corr)))
        if taken[atom]:
            raise RuntimeError("pursuit exhausted usable atoms before max_k")
        selected.append(atom)
        taken[atom] = True
        coef = refit_support(
            z,
            y,
            np.asarray(selected, dtype=np.int64),
            ridge_relative=ridge_relative,
        )
        residual = np.asarray(y - z[:, selected] @ coef, dtype=np.float32)
        if step in wanted:
            out[step] = PursuitCheckpoint(
                support=np.asarray(selected, dtype=np.int64),
                coefficients=np.asarray(coef, dtype=np.float64),
            )
    return out


def support_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = set(map(int, a)), set(map(int, b))
    return len(aa & bb) / max(1, len(aa | bb))


@torch.no_grad()
def factorized_local_coefficients(mapper: object, answer_weight: torch.Tensor) -> np.ndarray:
    """Local linear context-feature coefficients for an answer-code readout.

    The hard answer-SAE threshold is intentionally omitted: these coefficients
    rank the candidate atoms and define the fixed-magnitude control, while
    pursuit itself targets the complete nonlinear mapped score.
    """

    weight = torch.as_tensor(
        answer_weight,
        dtype=torch.float32,
        device=mapper.answer_sae.w_enc.device,
    )
    if weight.ndim != 1 or weight.numel() != mapper.answer_sae.dict_size:
        raise ValueError(
            f"answer weight shape {tuple(weight.shape)} != ({mapper.answer_sae.dict_size},)"
        )
    if mapper.scale is not None:
        weight = weight * mapper.scale.to(weight.device, torch.float32)
    answer_dense_direction = mapper.answer_sae.w_enc @ weight
    ridge_w = torch.as_tensor(mapper.ridge["W"], device=weight.device, dtype=torch.float32)
    ridge_xsd = torch.as_tensor(mapper.ridge["xsd"], device=weight.device, dtype=torch.float32)
    context_dense_direction = (ridge_w @ answer_dense_direction) / ridge_xsd
    coefficients = mapper.context_sae.w_dec @ context_dense_direction
    return coefficients.detach().cpu().numpy().astype(np.float64, copy=False)


@dataclass(frozen=True)
class BehaviorPursuitFit:
    """Sparse approximations to one full mapped answer-space score."""

    candidates: np.ndarray
    context_mean: np.ndarray
    context_std: np.ndarray
    target_mean: float
    raw_local_coefficients: np.ndarray
    methods: Mapping[str, Mapping[int, PursuitCheckpoint]]
    split_half_jaccard: Mapping[int, float]
    k_ladder: tuple[int, ...]
    ridge_relative: float


def fit_behavior_pursuit(
    z_context: torch.Tensor | np.ndarray,
    mapped_score: torch.Tensor | np.ndarray,
    train_mask: torch.Tensor | np.ndarray,
    raw_local_coefficients: np.ndarray,
    *,
    candidates: int = 128,
    k_ladder: Sequence[int] = K_LADDER_DEFAULT,
    ridge_relative: float = REFIT_RIDGE_RELATIVE_DEFAULT,
) -> BehaviorPursuitFit:
    """Fit pursuit and coefficient-matched controls on readout-fit rows only."""

    z_all = np.asarray(
        z_context.detach().float().cpu() if isinstance(z_context, torch.Tensor) else z_context,
        dtype=np.float32,
    )
    score_all = np.asarray(
        mapped_score.detach().float().cpu()
        if isinstance(mapped_score, torch.Tensor)
        else mapped_score,
        dtype=np.float32,
    )
    mask = np.asarray(
        train_mask.detach().cpu() if isinstance(train_mask, torch.Tensor) else train_mask,
        dtype=bool,
    )
    raw = np.asarray(raw_local_coefficients, dtype=np.float64)
    if z_all.ndim != 2 or score_all.shape != (z_all.shape[0],) or mask.shape != score_all.shape:
        raise ValueError(
            f"bad behavior-pursuit shapes: z={z_all.shape}, score={score_all.shape}, "
            f"mask={mask.shape}"
        )
    if raw.shape != (z_all.shape[1],):
        raise ValueError(f"raw coefficients {raw.shape} != ({z_all.shape[1]},)")
    z = z_all[mask]
    target = score_all[mask]
    if len(target) < 4:
        raise ValueError("behavior pursuit needs at least four fit rows")
    mean = z.astype(np.float64).mean(0)
    std = z.astype(np.float64).std(0)
    live = (std > 1e-8) & np.isfinite(raw)
    candidate_count = min(int(candidates), int(live.sum()))
    checkpoints = tuple(sorted({int(k) for k in k_ladder if 1 <= int(k) <= candidate_count}))
    if not checkpoints:
        raise ValueError("no requested pursuit support fits inside the live candidate set")
    strength = np.abs(raw * std)
    strength[~live] = -np.inf
    chosen = np.argpartition(strength, -candidate_count)[-candidate_count:]
    chosen = chosen[np.argsort(-strength[chosen], kind="stable")].astype(np.int64)
    design = ((z[:, chosen] - mean[chosen]) / std[chosen]).astype(np.float32)
    target_mean = float(target.astype(np.float64).mean())
    centered = np.asarray(target - target_mean, dtype=np.float32)
    max_k = max(checkpoints)
    pursuit = signed_gradient_pursuit(
        design,
        centered,
        max_k=max_k,
        checkpoints=checkpoints,
        ridge_relative=ridge_relative,
    )
    fixed: dict[int, PursuitCheckpoint] = {}
    refit: dict[int, PursuitCheckpoint] = {}
    standardized_local = raw[chosen] * std[chosen]
    for k in checkpoints:
        support = np.arange(k, dtype=np.int64)
        fixed[k] = PursuitCheckpoint(support, standardized_local[:k].copy())
        refit[k] = PursuitCheckpoint(
            support,
            refit_support(design, centered, support, ridge_relative=ridge_relative),
        )

    even, odd = np.arange(0, len(centered), 2), np.arange(1, len(centered), 2)
    split_a = signed_gradient_pursuit(
        design[even],
        centered[even],
        max_k=max_k,
        checkpoints=checkpoints,
        ridge_relative=ridge_relative,
    )
    split_b = signed_gradient_pursuit(
        design[odd],
        centered[odd],
        max_k=max_k,
        checkpoints=checkpoints,
        ridge_relative=ridge_relative,
    )
    return BehaviorPursuitFit(
        candidates=chosen,
        context_mean=mean[chosen],
        context_std=std[chosen],
        target_mean=target_mean,
        raw_local_coefficients=raw[chosen],
        methods={
            "gradient_pursuit": pursuit,
            "magnitude_fixed": fixed,
            "magnitude_refit": refit,
        },
        split_half_jaccard={
            k: support_jaccard(split_a[k].support, split_b[k].support) for k in checkpoints
        },
        k_ladder=checkpoints,
        ridge_relative=float(ridge_relative),
    )


def apply_behavior_pursuit(
    fit: BehaviorPursuitFit, z_context: torch.Tensor | np.ndarray
) -> dict[str, torch.Tensor]:
    """Apply every pursuit/control checkpoint, returning score tensors."""

    z = np.asarray(
        z_context.detach().float().cpu() if isinstance(z_context, torch.Tensor) else z_context,
        dtype=np.float32,
    )
    design = (z[:, fit.candidates] - fit.context_mean) / fit.context_std
    out = {}
    for method, by_k in fit.methods.items():
        for k, checkpoint in by_k.items():
            score = fit.target_mean + design[:, checkpoint.support] @ checkpoint.coefficients
            out[f"{method}_k{k}"] = torch.from_numpy(np.asarray(score, dtype=np.float32))
    return out


def behavior_pursuit_summary(fit: BehaviorPursuitFit) -> dict:
    """JSON-safe fit metadata; support values are global context feature IDs."""

    methods = {}
    for method, by_k in fit.methods.items():
        methods[method] = {}
        for k, checkpoint in by_k.items():
            methods[method][str(k)] = {
                "context_feature_ids": fit.candidates[checkpoint.support].tolist(),
                "coefficients": checkpoint.coefficients.tolist(),
            }
    return {
        "target": "full nonlinear mapped-answer-SAE behavior score",
        "candidate_rule": (
            "largest absolute standardized coefficient in the factorized map's local "
            "linearization; hard answer-SAE threshold omitted only for candidate/control ranking"
        ),
        "candidate_count": int(len(fit.candidates)),
        "k_ladder": list(fit.k_ladder),
        "signed_coefficients": True,
        "selection": "maximum absolute normalized residual correlation",
        "refit": "joint ridge on every selected support",
        "joint_refit_ridge_relative": fit.ridge_relative,
        "split_half_support_jaccard": {
            str(k): float(value) for k, value in fit.split_half_jaccard.items()
        },
        "methods": methods,
        "interpretation": "predictive sparse edges, not causal feature routes",
    }
