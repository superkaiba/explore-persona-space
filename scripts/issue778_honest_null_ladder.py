#!/usr/bin/env python
"""Issue #778 — honest trait-agnostic null ladder (OFFICIAL recompute).

The #778 null battery was found CIRCULAR. Its "norm-matched random" null
(``null_battery.randnorm_null_draws``) samples directions from the POOLED
pos+neg extraction-activation covariance, whose top eigenvector is ~r_B
(cos 0.996/0.985/0.736 for evil/sycophancy/hallucination) because the pooled
covariance is dominated by the between-arm mean difference — which IS r_B. So
a "random" draw from that covariance lands close to r_B and inflates the null
band, making r_B's matched-trait correlation look unremarkable. The
shuffled-label permutation null is circular for the same reason (it re-derives
a diff-of-means direction from the same pooled pool).

This script makes the honest trait-agnostic nulls OFFICIAL. It implements a
FIVE-family ladder (four honest nulls + the cross-trait control), plus the
ORIGINAL two families recomputed with the same seeds/draw-count so every family
sits in ONE self-contained JSON and the old committed numbers can be
sanity-checked:

  1. isotropic        — uniform random direction on the sphere, norm-matched
                        (the classic "actual random direction" floor).
  2. within_class     — PRIMARY honest null. Σ_within = covariance of the
                        extraction activations after centering EACH ARM
                        separately (pos rows by pos-mean, neg rows by neg-mean,
                        then pooled) — the activation noise structure WITHOUT
                        the between-arm contrast; same λ=0.1 diagonal shrinkage
                        as the original sampler; draws N(0, Σ_within),
                        norm-matched.
  3. neg_arm_only     — single-arm (negative) covariance. Σ estimated from ONLY
                        the negative-arm (trait-suppressing) kept extraction
                        activations, centered by their own mean, same λ=0.1
                        shrinkage, norm-matched. Drops the positive-arm trait-
                        INTENSITY variance (judge-kept rollouts 55-95, a gradient
                        along r_B) the pooled within-class rung still carries —
                        the closest persisted approximation to a fully trait-
                        agnostic covariance.
  4. rb_projected_out — the ORIGINAL pooled-covariance sampler, then project the
                        r_B[layer] ray OUT of each draw and renormalize to
                        ‖r_B‖. Isolates how much of the old null's power was the
                        r_B component.
  5. crosstrait       — the other two traits' r_B (fixed directions; the paper's
                        own control), recomputed for self-containment.
  6. orig_randnorm    — the ORIGINAL pooled-cov N(0, Σ_pooled) sampler
                        (``null_battery.randnorm_null_draws``), same seed/draws
                        (SANITY: must reproduce the committed p97.5).
  7. orig_perm        — the ORIGINAL shuffled-label permutation
                        (``null_battery.perm_null_draws``), same seed/draws
                        (SANITY: must reproduce the committed p97.5).

Two statistical regimes per cell:
  - fixed-layer  (``--stage fixed``): observed |r| + null band + one-sided p at a
    small set of chosen layers per cell — (a) the cell's own max-over-28 argmax,
    (b) #778's per-setting selected layer, (c) the paper's steering-selected
    layer. CHEAP (a handful of layers) — runs in minutes, committed first.
  - max-over-28  (``--stage maxlayer``): the #778 headline statistic. observed =
    max over layers of |r|; each null draw likewise takes its own max over its
    28 per-layer draws. EXPENSIVE (all 28 layers  x 1000 draws) — runs after the
    fixed stage. Also emits the full per-layer bands (authoritative).

CPU-only closed-form / sampling statistics over cached activation tensors; no
model calls, no GPU. The stochastic draw loops are VECTORIZED (batched GEMMs
over draws  x layers, mirroring ``null_battery``'s #834 vectorization) and pinned
against a 50-draw serial reference at rtol 1e-6 (``--selftest``).

The 15 setting-cells: 3 traits  x {finetune (overall), monitoring_corrected
(overall + within), monitoring_manyshot (overall + within)}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib  # noqa: E402

from explore_persona_space.analysis import null_battery as nb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue778.honest_nulls")

TRAITS: tuple[str, ...] = ("evil", "sycophancy", "hallucination")
N_LAYERS = 28

# The honest ladder + the two originals (recomputed for one-JSON comparison).
LADDER: tuple[str, ...] = (
    "isotropic",
    "within_class",
    "neg_arm_only",
    "rb_projected_out",
    "crosstrait",
    "orig_randnorm",
    "orig_perm",
)
# Families sampled from a per-layer direction distribution (vs the fixed
# crosstrait directions).
STOCHASTIC: frozenset[str] = frozenset(
    {"isotropic", "within_class", "neg_arm_only", "rb_projected_out", "orig_randnorm", "orig_perm"}
)
# The NEW honest stochastic families (the reduce-to set if max-over-28
# wall-clock projects past budget).
NEW_STOCHASTIC: tuple[str, ...] = ("isotropic", "within_class", "neg_arm_only", "rb_projected_out")

# Plain-English series names for the figures.
LADDER_LABELS: dict[str, str] = {
    "isotropic": "Isotropic floor",
    "within_class": "Within-class covariance (primary)",
    "neg_arm_only": "Single-arm (negative) covariance",
    "rb_projected_out": "r_B projected out",
    "crosstrait": "Cross-trait direction",
    "orig_randnorm": "Original pooled covariance",
    "orig_perm": "Shuffled-label permutation",
}

# Seed policy (recorded in every JSON). orig_randnorm / orig_perm MUST use
# seed=0 to reproduce the committed battery (its driver ran compute_setting with
# seed=0 for every cell). New families get a distinct per-(family, cell) seed:
# SEED_BASE[family] + cell_index.
SEED_BASE: dict[str, int] = {
    "isotropic": 100_000,
    "within_class": 200_000,
    "neg_arm_only": 400_000,
    "rb_projected_out": 300_000,
    "orig_randnorm": 0,
    "orig_perm": 0,
}

# #778's selected (max-over-28 argmax) layer per setting x trait, from the
# committed *_nullbattery.json (0-indexed into the 28-layer r_B tensor). Applied
# to BOTH the overall and within regimes of a setting (the brief gives one per
# setting x trait).
ISSUE778_SELECTED: dict[tuple[str, str], int] = {
    ("finetune", "evil"): 23,
    ("finetune", "sycophancy"): 20,
    ("finetune", "hallucination"): 25,
    ("monitoring_corrected", "evil"): 23,
    ("monitoring_corrected", "sycophancy"): 20,
    ("monitoring_corrected", "hallucination"): 21,
    ("monitoring_manyshot", "evil"): 20,
    ("monitoring_manyshot", "sycophancy"): 20,
    ("monitoring_manyshot", "hallucination"): 23,
}
# The paper's steering-selected layers, as WRITTEN in the task brief. NOTE the
# indexing ambiguity: the lib records the paper's "layer 20" == 0-indexed layer
# 19 (block index; hidden_states[1..28] stored 0..27). We use the brief's values
# as 0-INDEXED indices here and record that choice; because every JSON also
# carries the full 28-layer observed |r| + per-layer null bands, the 1-indexed
# interpretation (idx 19/19/15) is retrievable post-hoc with no rerun.
PAPER_STEERING: dict[str, int] = {"evil": 20, "sycophancy": 20, "hallucination": 16}

SETTINGS: tuple[str, ...] = ("finetune", "monitoring_corrected", "monitoring_manyshot")
REGIMES: dict[str, tuple[str, ...]] = {
    "finetune": ("overall",),
    "monitoring_corrected": ("overall", "within"),
    "monitoring_manyshot": ("overall", "within"),
}

LAYER_INDEPENDENCE_NOTE = (
    "Per-layer null draws are INDEPENDENT across layers, while r_B's 28 layers "
    "are highly correlated. The max-over-28 null therefore takes 28 near-"
    "independent chances at a high |r| whereas the observed r_B statistic takes "
    "~1 effective chance — a conservative-against-the-vector inflation of the "
    "null band. Recorded, not corrected, in v1."
)

WITHIN_CLASS_CAVEAT = (
    "The pooled within-class covariance null (within_class) is CONSERVATIVE but "
    "NOT fully trait-agnostic: each arm is centered by its own mean, so the "
    "between-arm contrast (== r_B) is removed, but residual trait-INTENSITY "
    "variance WITHIN the positive arm (judge-kept rollouts range 55-95, a "
    "gradient that points along r_B) plus the topic conditioning shared by all "
    "extraction rollouts remain in the covariance. The neg_arm_only rung "
    "(single-arm negative covariance) drops the positive-arm intensity variance "
    "and is the closest persisted approximation to a trait-agnostic covariance. "
    "A FULLY independent covariance null (activations on trait-UNRELATED generic "
    "rollouts) is DEFERRED to the planned rerun because no trait-neutral "
    "row-level activations are persisted — it needs a small capture pass."
)

DIAGNOSTIC_LAYERS: dict[str, int] = {"evil": 23, "sycophancy": 20, "hallucination": 25}


# ── Provenance ───────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Loaders (mirror scripts/issue778_null_battery.py + the audit prototype) ────


def _load_rb(out_root: Path, trait: str) -> np.ndarray:
    import torch

    return (
        torch.load(out_root / "rb" / f"{trait}.pt", weights_only=False).numpy().astype(np.float64)
    )  # (28, 3584)


def _load_pools(out_root: Path, trait: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    pos = torch.load(out_root / "activations" / f"{trait}_pos.pt", weights_only=False)
    neg = torch.load(out_root / "activations" / f"{trait}_neg.pt", weights_only=False)
    return pos.numpy().astype(np.float64), neg.numpy().astype(np.float64)


def _load_finetune(
    out_root: Path, eval_root: Path, trait: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(shift_acts (24,28,3584), target (24,), tags). shift = ft-mean - base-mean."""
    import torch

    base_vec = (
        torch.load(out_root / "finetune_activations" / "base.pt", weights_only=False)[trait]
        .numpy()
        .astype(np.float64)
    )
    shifts, targets, tags = [], [], []
    for fam in lib.FAMILIES:
        for ver in lib.VERSIONS:
            tag = f"{fam}_{ver}"
            act_path = out_root / "finetune_activations" / f"{tag}.pt"
            expr_path = eval_root / f"finetune_{trait}_{fam}_{ver}.json"
            if not act_path.exists() or not expr_path.exists():
                logger.warning("finetune cell %s missing artifacts; skipping (%s)", tag, trait)
                continue
            ft = torch.load(act_path, weights_only=False)[trait].numpy().astype(np.float64)
            with open(expr_path) as ef:
                score = json.load(ef).get("trait_score")
            if score is None:
                logger.warning("finetune cell %s trait_score None; skipping", tag)
                continue
            shifts.append(ft - base_vec)
            targets.append(score)
            tags.append(tag)
    if not shifts:
        raise RuntimeError(f"trait={trait}: no usable finetune cells")
    return np.stack(shifts, axis=0), np.array(targets, dtype=np.float64), tags


def _load_monitoring(
    out_root: Path, eval_root: Path, trait: str, tag: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(predictor_acts (m,28,3584), target (m,), condition_ids (m,)).

    Drops rows with a null mean_trait_score, mirroring the original battery's
    row-filtering exactly.
    """
    import torch

    acts = (
        torch.load(out_root / tag / f"{trait}_acts.pt", weights_only=False)
        .numpy()
        .astype(np.float64)
    )
    with open(eval_root / f"{tag}_{trait}.jsonl") as jf:
        rows = [json.loads(ln) for ln in jf if ln.strip()]
    if acts.shape[0] != len(rows):
        raise RuntimeError(
            f"{tag}/{trait}: raw acts {acts.shape[0]} != jsonl rows {len(rows)} — alignment broken"
        )
    keep = np.array([r["mean_trait_score"] is not None for r in rows])
    target = np.array([r["mean_trait_score"] for r in rows if r["mean_trait_score"] is not None])
    cid = np.array([r["condition_id"] for r in rows if r["mean_trait_score"] is not None])
    return acts[keep], target, cid


def _load_cell(
    setting: str, out_root: Path, eval_root: Path, trait: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str] | None]:
    """Return (predictor_acts, target, condition_ids|None, tags|None)."""
    if setting == "finetune":
        acts, target, tags = _load_finetune(out_root, eval_root, trait)
        return acts, target, None, tags
    acts, target, cid = _load_monitoring(out_root, eval_root, trait, setting)
    return acts, target, cid, None


# ── Cholesky cache (per trait; shared across a trait's setting-cells) ──────────


def _within_centered_pool(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Per-arm-centered pooled activations: pos-mean(pos), neg-mean(neg), stacked.

    Removes the between-arm mean difference (== r_B by construction), leaving the
    activation NOISE structure only.
    """
    pos_c = pos - pos.mean(axis=0, keepdims=True)
    neg_c = neg - neg.mean(axis=0, keepdims=True)
    return np.concatenate([pos_c, neg_c], axis=0)


def _chols_for_layers(pool_or_within: np.ndarray, layers, lam: float) -> dict[int, np.ndarray]:
    """{layer: shrunk-cov Cholesky} for the requested layers (reuses nb helper)."""
    return {layer: nb._shrunk_cholesky(pool_or_within[:, layer, :], lam) for layer in layers}


# ── Vectorized covariance-null sampler (mirrors nb.randnorm_null_draws) ────────


def _cov_null_draws(
    chols: dict[int, np.ndarray] | None,
    rb_norms: np.ndarray,
    predictor: np.ndarray,
    target: np.ndarray,
    layers: list[int],
    *,
    project_out_hat: dict[int, np.ndarray] | None = None,
    isotropic: bool = False,
    n_draws: int,
    seed: int,
    within: bool = False,
    condition_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Per-draw  x per-(requested-layer) |r| for a covariance / isotropic null.

    Mirrors ``nb.randnorm_null_draws`` exactly (draw-major / layer-minor rng,
    then a memory-bounded chunked batched projection via ``nb._batched_*``), so
    with ``chols`` == the pooled-cov Cholesky, ``isotropic=False``,
    ``project_out_hat=None``, ``layers=range(28)``, ``seed=0`` it is BIT-IDENTICAL
    to the library sampler (the orig_randnorm reproduction guarantee). Extra
    features: ``isotropic`` (draw ~N(0,I), skip the Cholesky matmul) and
    ``project_out_hat`` (project the r_B ray out of each raw draw before
    renormalizing).

    Returns ``(n_draws, len(layers))`` |r| matrix.
    """
    predictor = np.asarray(predictor, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rb_norms = np.asarray(rb_norms, dtype=np.float64)
    n_pred, _Lfull, D = predictor.shape
    nL = len(layers)
    if n_draws == 0:
        return np.empty((0, nL), dtype=np.float64)
    rng = np.random.default_rng(seed)
    # rng in the EXACT library order over the FULL draw range (bit-identical
    # stream): one rng.standard_normal(D) per (draw, layer), draw-major.
    z_stack = np.empty((n_draws, nL, D), dtype=np.float64)
    for d in range(n_draws):
        for li in range(nL):
            z_stack[d, li] = rng.standard_normal(D)
    pred_sub = predictor[:, layers, :]  # (n_pred, nL, D)
    out = np.empty((n_draws, nL), dtype=np.float64)
    bytes_per_draw = 2 * nL * D * 8 + 4 * n_pred * nL * 8
    per = max(1, int(nb._MAX_BATCH_BYTES // max(1, bytes_per_draw)))
    for start in range(0, n_draws, per):
        stop = min(start + per, n_draws)
        k = stop - start
        dirs = np.empty((k, nL, D), dtype=np.float64)
        for li, layer in enumerate(layers):
            z = z_stack[start:stop, li, :]  # (k, D)
            v = z if isotropic else z @ chols[layer].T  # (k, D)
            if project_out_hat is not None:
                rh = project_out_hat[layer]  # (D,)
                v = v - (v @ rh)[:, None] * rh[None, :]
            vn = np.linalg.norm(v, axis=1)  # (k,)
            scale = np.where(vn == 0, 1.0, rb_norms[layer] / np.where(vn == 0, 1.0, vn))
            dirs[:, li, :] = v * scale[:, None]
        if within:
            r = nb._batched_within_r(pred_sub, dirs, target, condition_ids)  # (nL, k)
        else:
            r = nb._batched_r_overall(pred_sub, dirs, target)  # (nL, k)
        out[start:stop] = np.abs(r).T
    return out


def _cov_null_draws_serial(
    chols: dict[int, np.ndarray] | None,
    rb_norms: np.ndarray,
    predictor: np.ndarray,
    target: np.ndarray,
    layers: list[int],
    *,
    project_out_hat: dict[int, np.ndarray] | None = None,
    isotropic: bool = False,
    n_draws: int,
    seed: int,
    within: bool = False,
    condition_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Serial reference for ``_cov_null_draws`` (the 50-draw rtol-1e-6 pin)."""
    predictor = np.asarray(predictor, dtype=np.float64)
    D = predictor.shape[2]
    nL = len(layers)
    rng = np.random.default_rng(seed)
    out = np.empty((n_draws, nL), dtype=np.float64)
    for d in range(n_draws):
        dir_full = np.zeros((nL, D), dtype=np.float64)
        for li, layer in enumerate(layers):
            z = rng.standard_normal(D)
            v = z if isotropic else chols[layer] @ z
            if project_out_hat is not None:
                rh = project_out_hat[layer]
                v = v - (v @ rh) * rh
            vn = np.linalg.norm(v)
            dir_full[li] = v / vn * rb_norms[layer] if vn > 0 else v
        pred_sub = predictor[:, layers, :]
        if within:
            r = nb.within_condition_r_per_layer(pred_sub, dir_full, target, condition_ids)
        else:
            r = nb.r_per_layer(pred_sub, dir_full, target)
        out[d] = np.abs(r)
    return out


def _perm_fixed_layers(
    pos: np.ndarray,
    neg: np.ndarray,
    predictor: np.ndarray,
    target: np.ndarray,
    layers: list[int],
    *,
    n_draws: int,
    seed: int,
    within: bool = False,
    condition_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Shuffled-label permutation null restricted to ``layers`` (fixed stage).

    Same rng + algebra as ``nb.perm_null_draws`` but slices to the requested
    layers (the perm direction at layer L needs only pool[:, L, :]).
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    predictor = np.asarray(predictor, dtype=np.float64)
    n_pos = pos.shape[0]
    pool = np.concatenate([pos, neg], axis=0)[:, layers, :]  # (n_total, nL, D)
    pred_sub = predictor[:, layers, :]
    n_total = pool.shape[0]
    if n_draws == 0:
        return np.empty((0, len(layers)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(n_total) for _ in range(n_draws)])
    out = np.empty((n_draws, len(layers)), dtype=np.float64)
    n_pred = pred_sub.shape[0]
    nL, D = pool.shape[1], pool.shape[2]
    bytes_per_draw = nL * D * 8 + 4 * n_pred * nL * 8
    for start, stop in nb._k_chunks(n_draws, bytes_per_draw):
        dirs = nb._perm_directions(pool, perms[start:stop], n_pos)  # (k, nL, D)
        if within:
            r = nb._batched_within_r(pred_sub, dirs, target, condition_ids)
        else:
            r = nb._batched_r_overall(pred_sub, dirs, target)
        out[start:stop] = np.abs(r).T
    return out


# ── Observed r + bootstrap CIs ─────────────────────────────────────────────────


def _observed_per_layer(
    predictor: np.ndarray, rb: np.ndarray, target: np.ndarray, within: bool, cid
) -> np.ndarray:
    if within:
        return nb.within_condition_r_per_layer(predictor, rb, target, cid)
    return nb.r_per_layer(predictor, rb, target)


def _bootstrap_ci_within(
    predictor: np.ndarray,
    rb: np.ndarray,
    target: np.ndarray,
    condition_ids: np.ndarray,
    layer: int,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """95% CI on the within-condition Fisher-z r at ``layer`` via a stratified
    (within-condition) row bootstrap.

    Correctly propagates the WITHIN estimator's sampling variance — it does NOT
    inherit the pooled/overall CI (the #778 bug). Each draw resamples rows WITH
    REPLACEMENT inside each condition group (preserving group sizes), then
    recomputes the Fisher-z-weighted within-condition r at the fixed layer.
    """
    proj = nb.project(predictor[:, layer, :], rb[layer])  # (n,)
    target = np.asarray(target, dtype=np.float64)
    condition_ids = np.asarray(condition_ids)
    groups = [np.where(condition_ids == c)[0] for c in np.unique(condition_ids)]
    rng = np.random.default_rng(seed)
    boots = np.full(n_boot, np.nan, dtype=np.float64)
    for b in range(n_boot):
        z_sum = 0.0
        w_sum = 0.0
        for g in groups:
            if g.size < 4:
                continue
            idx = g[rng.integers(0, g.size, size=g.size)]
            r = nb._pearson(proj[idx], target[idx])
            if np.isnan(r):
                continue
            z_sum += (g.size - 3) * np.arctanh(np.clip(r, -0.999999, 0.999999))
            w_sum += g.size - 3
        if w_sum > 0:
            boots[b] = np.tanh(z_sum / w_sum)
    valid = boots[~np.isnan(boots)]
    if valid.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)))


# ── Band / p helpers ───────────────────────────────────────────────────────────


def _band_p_maxlayer(mat: np.ndarray, observed_max: float) -> dict:
    """Max-over-layers band + one-sided p from a (n_draws|n_dirs, L) |r| matrix."""
    per_draw_max = np.array([nb.max_abs_over_layers(mat[i]) for i in range(mat.shape[0])])
    lo, hi = nb.null_band(per_draw_max)
    valid = per_draw_max[~np.isnan(per_draw_max)]
    p50 = float(np.percentile(valid, 50)) if valid.size else float("nan")
    return {
        "n_draws": int(mat.shape[0]),
        "p2_5": lo,
        "p50": p50,
        "p97_5": hi,
        "raw_p_one_sided": nb.empirical_p_one_sided(observed_max, per_draw_max),
        "_per_draw_max": per_draw_max,  # stripped before JSON dump
    }


def _band_p_fixed(col: np.ndarray, observed: float) -> dict:
    """Single-layer band + one-sided p from a (n_draws,) |r| column."""
    valid = col[~np.isnan(col)]
    if valid.size == 0:
        return {"n_draws": int(col.size), "p2_5": None, "p50": None, "p97_5": None, "raw_p": None}
    return {
        "n_draws": int(col.size),
        "p2_5": float(np.percentile(valid, 2.5)),
        "p50": float(np.percentile(valid, 50)),
        "p97_5": float(np.percentile(valid, 97.5)),
        "raw_p": float((int((valid >= observed).sum()) + 1) / (valid.size + 1)),
    }


def _per_layer_bands(mat: np.ndarray) -> dict[str, list]:
    """p2.5 / p50 / p97.5 per layer from a (n_draws, L) |r| matrix."""
    with np.errstate(invalid="ignore"):
        p = {}
        for tag, q in (("p2_5", 2.5), ("p50", 50.0), ("p97_5", 97.5)):
            p[tag] = [
                float(np.percentile(mat[:, j][~np.isnan(mat[:, j])], q))
                if np.any(~np.isnan(mat[:, j]))
                else None
                for j in range(mat.shape[1])
            ]
    return p


# ── Circularity diagnostics ─────────────────────────────────────────────────


def _circularity_diagnostics(out_root: Path) -> dict:
    """cos(top-PC of pooled Σ, r̂_B) + variance ratio at each trait's selected
    layer — the numbers that expose the original null's circularity. Also the
    within-class top-PC cosine (should be LOW == honest)."""
    diag = {}
    for trait, layer in DIAGNOSTIC_LAYERS.items():
        rb = _load_rb(out_root, trait)
        pos, neg = _load_pools(out_root, trait)
        rh = rb[layer] / np.linalg.norm(rb[layer])
        pool_l = np.concatenate([pos, neg], axis=0)[:, layer, :]
        cov = np.cov(pool_l, rowvar=False)
        w, V = np.linalg.eigh(cov)
        within_l = _within_centered_pool(pos, neg)[:, layer, :]
        cov_w = np.cov(within_l, rowvar=False)
        _ww, Vw = np.linalg.eigh(cov_w)
        lam = nb.PRIMARY_LAMBDA
        sig = (1 - lam) * cov + lam * np.diag(np.diag(cov))
        diag[trait] = {
            "layer": layer,
            "cos_top_pc_pooled_vs_rB": abs(float(V[:, -1] @ rh)),
            "variance_ratio_lambda1_over_mean": float(w[-1] / w.mean()),
            "variance_inflation_along_rB": float((rh @ sig @ rh) / (np.trace(sig) / sig.shape[0])),
            "cos_top_pc_within_class_vs_rB": abs(float(Vw[:, -1] @ rh)),
            "n_pos": int(pos.shape[0]),
            "n_neg": int(neg.shape[0]),
        }
    return diag


# ── FIXED-LAYER stage ──────────────────────────────────────────────────────────


def _layer_choices(setting: str, trait: str, own_argmax: int) -> dict[str, int]:
    return {
        "own_argmax": int(own_argmax),
        "issue778_selected": ISSUE778_SELECTED[(setting, trait)],
        "paper_steering": PAPER_STEERING[trait],
    }


def run_fixed_stage(out_root: Path, eval_root: Path, n_draws: int, n_boot: int) -> dict:
    lam = nb.PRIMARY_LAMBDA
    rbs = {t: _load_rb(out_root, t) for t in TRAITS}
    pools = {t: _load_pools(out_root, t) for t in TRAITS}
    tensor_shas = _tensor_shas(out_root, eval_root)
    diagnostics = _circularity_diagnostics(out_root)
    cell_idx = 0
    files: dict[str, dict] = {}
    for trait in TRAITS:
        rb = rbs[trait]
        rb_norms = np.linalg.norm(rb, axis=1)
        rb_hat = {layer: rb[layer] / np.linalg.norm(rb[layer]) for layer in range(N_LAYERS)}
        pos, neg = pools[trait]
        other_rbs = {ot: rbs[ot] for ot in TRAITS if ot != trait}
        for setting in SETTINGS:
            predictor, target, cid, tags = _load_cell(setting, out_root, eval_root, trait)
            fkey = f"{trait}_{setting}"
            files.setdefault(
                fkey,
                {
                    "trait": trait,
                    "setting": setting,
                    "stage_fixed": {},
                    "n_points": int(predictor.shape[0]),
                    "tags": tags,
                    "layer_index_note": (
                        "layers are 0-indexed into the 28-layer r_B tensor (block "
                        "outputs, hidden_states[1..28] stored 0..27). paper_steering "
                        "uses the brief's values as 0-indexed; the paper's 1-indexed "
                        "convention would map layer L -> index L-1."
                    ),
                    "layer_independence_note": LAYER_INDEPENDENCE_NOTE,
                    "within_class_caveat": WITHIN_CLASS_CAVEAT,
                    "reproducibility": lib.repro_metadata(),
                    "tensor_sha256": tensor_shas.get(trait, {}),
                    "circularity_diagnostics": diagnostics.get(trait),
                    "seeds": {},
                },
            )
            for regime in REGIMES[setting]:
                within = regime == "within"
                obs_layers = _observed_per_layer(predictor, rb, target, within, cid)
                own_argmax = nb.argmax_abs_layer(obs_layers)
                choices = _layer_choices(setting, trait, own_argmax)
                fixed_layers = sorted(set(choices.values()))
                # Cholesky only at the fixed layers (cheap).
                within_pool = _within_centered_pool(pos, neg)
                pool = np.concatenate([pos, neg], axis=0)
                chols_pool = _chols_for_layers(pool, fixed_layers, lam)
                chols_within = _chols_for_layers(within_pool, fixed_layers, lam)
                # neg_arm_only: Σ from the negative arm alone (nb._shrunk_cholesky
                # re-centers by the arm's own mean via np.cov).
                chols_neg = _chols_for_layers(neg, fixed_layers, lam)
                # Sample each stochastic family at the fixed layers.
                seeds_here = {}
                fam_cols: dict[str, np.ndarray] = {}
                for fam in STOCHASTIC:
                    seed = SEED_BASE[fam] + (0 if fam.startswith("orig_") else cell_idx)
                    seeds_here[fam] = seed
                    if fam == "isotropic":
                        m = _cov_null_draws(
                            None,
                            rb_norms,
                            predictor,
                            target,
                            fixed_layers,
                            isotropic=True,
                            n_draws=n_draws,
                            seed=seed,
                            within=within,
                            condition_ids=cid,
                        )
                    elif fam == "within_class":
                        m = _cov_null_draws(
                            chols_within,
                            rb_norms,
                            predictor,
                            target,
                            fixed_layers,
                            n_draws=n_draws,
                            seed=seed,
                            within=within,
                            condition_ids=cid,
                        )
                    elif fam == "neg_arm_only":
                        m = _cov_null_draws(
                            chols_neg,
                            rb_norms,
                            predictor,
                            target,
                            fixed_layers,
                            n_draws=n_draws,
                            seed=seed,
                            within=within,
                            condition_ids=cid,
                        )
                    elif fam == "rb_projected_out":
                        m = _cov_null_draws(
                            chols_pool,
                            rb_norms,
                            predictor,
                            target,
                            fixed_layers,
                            project_out_hat=rb_hat,
                            n_draws=n_draws,
                            seed=seed,
                            within=within,
                            condition_ids=cid,
                        )
                    elif fam == "orig_randnorm":
                        m = _cov_null_draws(
                            chols_pool,
                            rb_norms,
                            predictor,
                            target,
                            fixed_layers,
                            n_draws=n_draws,
                            seed=seed,
                            within=within,
                            condition_ids=cid,
                        )
                    elif fam == "orig_perm":
                        m = _perm_fixed_layers(
                            pos,
                            neg,
                            predictor,
                            target,
                            fixed_layers,
                            n_draws=n_draws,
                            seed=seed,
                            within=within,
                            condition_ids=cid,
                        )
                    fam_cols[fam] = m  # (n_draws, nfixed)
                # crosstrait fixed directions
                cross = nb.crosstrait_null(
                    other_rbs, predictor, target, within=within, condition_ids=cid
                )  # (2, 28)
                # Assemble per-choice tables.
                per_choice = {}
                for choice, layer in choices.items():
                    li = fixed_layers.index(layer)
                    observed = float(abs(obs_layers[layer]))
                    fam_entries = {}
                    for fam in STOCHASTIC:
                        fam_entries[fam] = _band_p_fixed(fam_cols[fam][:, li], observed)
                    cross_abs = np.abs(cross[:, layer])
                    fam_entries["crosstrait"] = {
                        "n_draws": int(cross.shape[0]),
                        "values": [float(x) for x in cross_abs],
                        "p97_5": float(np.percentile(cross_abs, 97.5)),
                        "raw_p": float(
                            (int((cross_abs >= observed).sum()) + 1) / (cross.shape[0] + 1)
                        ),
                    }
                    per_choice[choice] = {
                        "layer": int(layer),
                        "observed_abs_r": observed,
                        "nulls": fam_entries,
                    }
                # Bootstrap CI at own_argmax for this regime.
                sel = own_argmax
                if within:
                    ci = _bootstrap_ci_within(
                        predictor, rb, target, cid, sel, n_boot=n_boot, seed=cell_idx
                    )
                else:
                    ci = nb.bootstrap_ci_matched_r(
                        predictor, rb, target, sel, n_boot=n_boot, seed=cell_idx
                    )
                files[fkey]["stage_fixed"][regime] = {
                    "observed_r_per_layer": [float(x) for x in obs_layers],
                    "own_argmax_layer": int(own_argmax),
                    "observed_matched_r_signed_at_argmax": float(obs_layers[own_argmax]),
                    "layer_choices": choices,
                    "bootstrap_ci95_at_own_argmax": list(ci),
                    "bootstrap_method": (
                        "within-condition Fisher-z stratified row bootstrap"
                        if within
                        else "row bootstrap (overall Pearson)"
                    ),
                    "per_choice": per_choice,
                }
                files[fkey]["seeds"][regime] = seeds_here
                cell_idx += 1
                logger.info("fixed: %s %s done (own_argmax L%d)", fkey, regime, own_argmax)
    return files


# ── MAX-OVER-28 stage ───────────────────────────────────────────────────────


def run_maxlayer_stage(
    out_root: Path, eval_root: Path, n_draws: int, families: tuple[str, ...]
) -> dict:
    lam = nb.PRIMARY_LAMBDA
    layers_all = list(range(N_LAYERS))
    rbs = {t: _load_rb(out_root, t) for t in TRAITS}
    pools = {t: _load_pools(out_root, t) for t in TRAITS}
    tensor_shas = _tensor_shas(out_root, eval_root)
    diagnostics = _circularity_diagnostics(out_root)
    cell_idx = 0
    files: dict[str, dict] = {}
    for trait in TRAITS:
        rb = rbs[trait]
        rb_norms = np.linalg.norm(rb, axis=1)
        rb_hat = {layer: rb[layer] / np.linalg.norm(rb[layer]) for layer in range(N_LAYERS)}
        pos, neg = pools[trait]
        other_rbs = {ot: rbs[ot] for ot in TRAITS if ot != trait}
        # Cholesky (all 28 layers) once per trait, shared across its settings.
        t0 = time.time()
        pool = np.concatenate([pos, neg], axis=0)
        within_pool = _within_centered_pool(pos, neg)
        chols_pool = _chols_for_layers(pool, layers_all, lam) if _needs_pool(families) else {}
        chols_within = (
            _chols_for_layers(within_pool, layers_all, lam) if "within_class" in families else {}
        )
        chols_neg = _chols_for_layers(neg, layers_all, lam) if "neg_arm_only" in families else {}
        logger.info("maxlayer: %s Cholesky done [%.0fs]", trait, time.time() - t0)
        for setting in SETTINGS:
            predictor, target, cid, tags = _load_cell(setting, out_root, eval_root, trait)
            fkey = f"{trait}_{setting}"
            files.setdefault(
                fkey,
                {
                    "trait": trait,
                    "setting": setting,
                    "stage_maxlayer": {},
                    "n_points": int(predictor.shape[0]),
                    "tags": tags,
                    "families": list(families),
                    "layer_independence_note": LAYER_INDEPENDENCE_NOTE,
                    "within_class_caveat": WITHIN_CLASS_CAVEAT,
                    "reproducibility": lib.repro_metadata(),
                    "tensor_sha256": tensor_shas.get(trait, {}),
                    "circularity_diagnostics": diagnostics.get(trait),
                    "seeds": {},
                },
            )
            for regime in REGIMES[setting]:
                within = regime == "within"
                obs_layers = _observed_per_layer(predictor, rb, target, within, cid)
                obs_max = nb.max_abs_over_layers(obs_layers)
                own_argmax = nb.argmax_abs_layer(obs_layers)
                seeds_here = {}
                fam_out = {}
                per_layer = {}
                for fam in families:
                    seed = SEED_BASE[fam] + (0 if fam.startswith("orig_") else cell_idx)
                    seeds_here[fam] = seed
                    ts = time.time()
                    if fam == "isotropic":
                        mat = _cov_null_draws(
                            None,
                            rb_norms,
                            predictor,
                            target,
                            layers_all,
                            isotropic=True,
                            n_draws=n_draws,
                            seed=seed,
                            within=within,
                            condition_ids=cid,
                        )
                    elif fam == "within_class":
                        mat = _cov_null_draws(
                            chols_within,
                            rb_norms,
                            predictor,
                            target,
                            layers_all,
                            n_draws=n_draws,
                            seed=seed,
                            within=within,
                            condition_ids=cid,
                        )
                    elif fam == "neg_arm_only":
                        mat = _cov_null_draws(
                            chols_neg,
                            rb_norms,
                            predictor,
                            target,
                            layers_all,
                            n_draws=n_draws,
                            seed=seed,
                            within=within,
                            condition_ids=cid,
                        )
                    elif fam == "rb_projected_out":
                        mat = _cov_null_draws(
                            chols_pool,
                            rb_norms,
                            predictor,
                            target,
                            layers_all,
                            project_out_hat=rb_hat,
                            n_draws=n_draws,
                            seed=seed,
                            within=within,
                            condition_ids=cid,
                        )
                    elif fam == "orig_randnorm":
                        mat = _cov_null_draws(
                            chols_pool,
                            rb_norms,
                            predictor,
                            target,
                            layers_all,
                            n_draws=n_draws,
                            seed=seed,
                            within=within,
                            condition_ids=cid,
                        )
                    elif fam == "orig_perm":
                        mat = nb.perm_null_draws(
                            pos,
                            neg,
                            predictor,
                            target,
                            n_draws=n_draws,
                            seed=seed,
                            within=within,
                            condition_ids=cid,
                        )
                    bp = _band_p_maxlayer(mat, obs_max)
                    per_draw_max = bp.pop("_per_draw_max")
                    fam_out[fam] = bp
                    per_layer[fam] = _per_layer_bands(mat)
                    # persist per-draw max arrays for figure regeneration
                    np.save(
                        eval_root / "honest_nulls" / f"{fkey}_{regime}_{fam}_maxdraws.npy",
                        per_draw_max.astype(np.float32),
                    )
                    logger.info(
                        "maxlayer: %s %s %s p97.5=%.4f raw_p=%s [%.0fs]",
                        fkey,
                        regime,
                        fam,
                        bp["p97_5"],
                        bp["raw_p_one_sided"],
                        time.time() - ts,
                    )
                # crosstrait fixed
                if "crosstrait" in families:
                    cross = nb.crosstrait_null(
                        other_rbs, predictor, target, within=within, condition_ids=cid
                    )
                    cross_max = np.array(
                        [nb.max_abs_over_layers(cross[i]) for i in range(cross.shape[0])]
                    )
                    fam_out["crosstrait"] = {
                        "n_draws": int(cross.shape[0]),
                        "values_max_abs": [float(x) for x in cross_max],
                        "p97_5": float(np.percentile(cross_max, 97.5)),
                        "raw_p_one_sided": nb.empirical_p_one_sided(obs_max, cross_max),
                    }
                    per_layer["crosstrait"] = {
                        "per_direction_abs_r": [
                            [float(x) for x in cross[i]] for i in range(cross.shape[0])
                        ]
                    }
                files[fkey]["stage_maxlayer"][regime] = {
                    "observed_matched_max_abs": float(obs_max),
                    "own_argmax_layer": int(own_argmax),
                    "observed_r_per_layer": [float(x) for x in obs_layers],
                    "nulls": fam_out,
                    "per_layer_bands": per_layer,
                }
                files[fkey]["seeds"][regime] = seeds_here
                cell_idx += 1
    return files


def _needs_pool(families) -> bool:
    return ("orig_randnorm" in families) or ("rb_projected_out" in families)


# ── BH across the 15 cells ────────────────────────────────────────────────────


def _apply_bh(files: dict, stage: str) -> None:  # noqa: C901
    """BH within null-family across the 15 cells + pooled-all, threaded back in.

    stage in {fixed, maxlayer}. For fixed we adjust each layer-choice's raw_p
    independently (one BH set per (choice, family)); for maxlayer one BH set per
    family across the 15 cells.
    """
    if stage == "maxlayer":
        # collect (fkey, regime, fam) -> raw_p
        fam_pvals: dict[str, list] = {}
        idx: dict[str, list] = {}
        pooled = []
        pooled_idx = []
        for fkey, fd in files.items():
            for regime, rd in fd.get("stage_maxlayer", {}).items():
                for fam, nr in rd["nulls"].items():
                    p = nr.get("raw_p_one_sided")
                    if p is None:
                        continue
                    fam_pvals.setdefault(fam, []).append(p)
                    idx.setdefault(fam, []).append((fkey, regime, fam))
                    pooled.append(p)
                    pooled_idx.append((fkey, regime, fam))
        bh_within = {}
        for fam, pv in fam_pvals.items():
            adj = nb.benjamini_hochberg(pv)
            for i, key in enumerate(idx[fam]):
                bh_within[key] = adj[i]
        pooled_adj = nb.benjamini_hochberg(pooled)
        bh_pooled = {pooled_idx[i]: pooled_adj[i] for i in range(len(pooled_idx))}
        for fkey, fd in files.items():
            for regime, rd in fd.get("stage_maxlayer", {}).items():
                for fam, nr in rd["nulls"].items():
                    key = (fkey, regime, fam)
                    nr["bh_within_family"] = bh_within.get(key)
                    nr["bh_pooled_all"] = bh_pooled.get(key)
    else:
        for choice in ("own_argmax", "issue778_selected", "paper_steering"):
            fam_pvals: dict[str, list] = {}
            idx: dict[str, list] = {}
            pooled = []
            pooled_idx = []
            for fkey, fd in files.items():
                for regime, rd in fd.get("stage_fixed", {}).items():
                    pc = rd["per_choice"][choice]
                    for fam, nr in pc["nulls"].items():
                        p = nr.get("raw_p")
                        if p is None:
                            continue
                        fam_pvals.setdefault(fam, []).append(p)
                        idx.setdefault(fam, []).append((fkey, regime, fam))
                        pooled.append(p)
                        pooled_idx.append((fkey, regime, fam))
            bh_within = {}
            for fam, pv in fam_pvals.items():
                adj = nb.benjamini_hochberg(pv)
                for i, key in enumerate(idx[fam]):
                    bh_within[key] = adj[i]
            pooled_adj = nb.benjamini_hochberg(pooled)
            bh_pooled = {pooled_idx[i]: pooled_adj[i] for i in range(len(pooled_idx))}
            for fkey, fd in files.items():
                for regime, rd in fd.get("stage_fixed", {}).items():
                    pc = rd["per_choice"][choice]
                    for fam, nr in pc["nulls"].items():
                        key = (fkey, regime, fam)
                        nr["bh_within_family"] = bh_within.get(key)
                        nr["bh_pooled_all"] = bh_pooled.get(key)


# ── Provenance: tensor SHAs ────────────────────────────────────────────────────


def _tensor_shas(out_root: Path, eval_root: Path) -> dict:
    shas: dict[str, dict] = {}
    for trait in TRAITS:
        d = {
            "rb": _sha256(out_root / "rb" / f"{trait}.pt"),
            "activations_pos": _sha256(out_root / "activations" / f"{trait}_pos.pt"),
            "activations_neg": _sha256(out_root / "activations" / f"{trait}_neg.pt"),
            "monitoring_corrected_acts": _sha256(
                out_root / "monitoring_corrected" / f"{trait}_acts.pt"
            ),
            "monitoring_manyshot_acts": _sha256(
                out_root / "monitoring_manyshot" / f"{trait}_acts.pt"
            ),
        }
        shas[trait] = d
    shas["finetune_base"] = {"base": _sha256(out_root / "finetune_activations" / "base.pt")}
    return shas


# ── Figures (fixed-layer ladder band plots, from the committed JSONs) ──────────


def build_fixed_figures(eval_root: Path, fig_root: Path, choice: str = "own_argmax") -> list[str]:
    """Fixed-layer band plots per (trait, setting, regime) at ``choice`` layer.

    Reads the committed ``*_honestnulls.json`` (no recompute): for each null
    family plots its single-layer null band (p2.5-p97.5 bar + median) and
    overlays the observed matched |r| + its bootstrap CI.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style()
    fig_root.mkdir(parents=True, exist_ok=True)
    order = [
        "isotropic",
        "within_class",
        "neg_arm_only",
        "rb_projected_out",
        "crosstrait",
        "orig_randnorm",
        "orig_perm",
    ]
    palette = pp.paper_palette(len(order))
    written = []
    import glob

    setting_label = {
        "finetune": "finetuning shift",
        "monitoring_corrected": "corrected 8-prompt monitoring",
        "monitoring_manyshot": "many-shot ICL monitoring",
    }
    regime_label = {"overall": "pooled", "within": "within-condition"}
    for path in sorted(glob.glob(str(eval_root / "honest_nulls" / "*_honestnulls.json"))):
        with open(path) as f:
            fd = json.load(f)
        trait, setting = fd["trait"], fd["setting"]
        for regime, rd in fd.get("stage_fixed", {}).items():
            pc = rd["per_choice"][choice]
            layer = pc["layer"]
            obs = pc["observed_abs_r"]
            fams = [f for f in order if f in pc["nulls"]]
            fig, ax = plt.subplots(figsize=(7.4, 3.8))
            ypos = list(range(len(fams)))
            for yi, fam in zip(ypos, fams, strict=True):
                nr = pc["nulls"][fam]
                color = palette[order.index(fam)]
                if fam == "crosstrait":
                    vals = [abs(v) for v in nr.get("values", [])]
                    ax.plot(vals, [yi] * len(vals), "o", color=color, ms=5)
                    continue
                lo, med, hi = nr.get("p2_5"), nr.get("p50"), nr.get("p97_5")
                if lo is None:
                    continue
                ax.plot([lo, hi], [yi, yi], color=color, lw=3.0, solid_capstyle="round")
                ax.plot([med], [yi], "o", color=color, ms=5)
            ax.axvline(obs, color="black", lw=1.8, ls="--")
            ci = rd.get("bootstrap_ci95_at_own_argmax")
            if ci and all(c is not None for c in ci):
                ax.plot(ci, [len(fams), len(fams)], color="black", lw=2.5, solid_capstyle="round")
            ax.plot([obs], [len(fams)], "D", color="black", ms=6)
            ax.set_yticks([*ypos, len(fams)])
            ax.set_yticklabels([LADDER_LABELS[f] for f in fams] + ["Observed r_B (matched)"])
            ax.set_xlabel(f"|Pearson r| at layer {layer} (predictor projection vs trait score)")
            ax.set_xlim(0, 1.0)
            ax.set_title(
                f"{trait} - {setting_label.get(setting, setting)} "
                f"({regime_label.get(regime, regime)}, layer {layer})"
            )
            fig.tight_layout()
            stem = f"fixed_bands_{trait}_{setting}_{regime}"
            pp.savefig_paper(fig, stem, dir=str(fig_root), formats=("png", "pdf"))
            plt.close(fig)
            written.append(str(fig_root / f"{stem}.png"))
    return written


# ── Figures (max-over-28 ladder band plots) ────────────────────────────────────


def build_figures(files: dict, eval_root: Path, fig_root: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style()
    fig_root.mkdir(parents=True, exist_ok=True)
    written = []
    order = [
        "isotropic",
        "within_class",
        "neg_arm_only",
        "rb_projected_out",
        "crosstrait",
        "orig_randnorm",
        "orig_perm",
    ]
    palette = pp.paper_palette(len(order))
    for fkey, fd in files.items():
        trait = fd["trait"]
        setting = fd["setting"]
        for regime, rd in fd.get("stage_maxlayer", {}).items():
            obs = rd["observed_matched_max_abs"]
            fams = [f for f in order if f in rd["nulls"]]
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            ypos = list(range(len(fams)))
            for yi, fam in zip(ypos, fams, strict=True):
                npy = eval_root / "honest_nulls" / f"{fkey}_{regime}_{fam}_maxdraws.npy"
                nr = rd["nulls"][fam]
                if fam == "crosstrait":
                    vals = np.array(nr["values_max_abs"], dtype=np.float64)
                elif npy.exists():
                    vals = np.load(npy).astype(np.float64)
                else:
                    vals = np.array(
                        [nr.get("p2_5"), nr.get("p50"), nr.get("p97_5")], dtype=np.float64
                    )
                vals = vals[~np.isnan(vals)]
                color = palette[order.index(fam)]
                if vals.size >= 20:
                    parts = ax.violinplot(
                        [vals], positions=[yi], vert=False, showextrema=False, widths=0.8
                    )
                    for b in parts["bodies"]:
                        b.set_facecolor(color)
                        b.set_alpha(0.55)
                        b.set_edgecolor(color)
                    lo, hi = np.percentile(vals, [2.5, 97.5])
                    ax.plot([lo, hi], [yi, yi], color=color, lw=2.0, solid_capstyle="round")
                    ax.plot([np.median(vals)], [yi], "o", color=color, ms=4)
                else:
                    ax.plot(vals, [yi] * vals.size, "o", color=color, ms=5)
            ax.axvline(obs, color="black", lw=1.8, ls="--")
            # observed marker + bootstrap CI from the fixed stage if present
            ax.plot([obs], [len(fams)], "D", color="black", ms=6)
            ax.set_yticks([*ypos, len(fams)])
            ax.set_yticklabels([LADDER_LABELS[f] for f in fams] + ["Observed r_B (matched)"])
            ax.set_xlabel("max over 28 layers of |Pearson r| (predictor projection vs trait score)")
            ax.set_xlim(0, 1.0)
            regime_label = {"overall": "pooled", "within": "within-condition"}.get(regime, regime)
            setting_label = {
                "finetune": "finetuning shift",
                "monitoring_corrected": "corrected 8-prompt monitoring",
                "monitoring_manyshot": "many-shot ICL monitoring",
            }.get(setting, setting)
            ax.set_title(f"{trait} — {setting_label} ({regime_label})")
            fig.tight_layout()
            stem = f"bands_{fkey}_{regime}"
            pp.savefig_paper(fig, stem, dir=str(fig_root), formats=("png", "pdf"))
            plt.close(fig)
            written.append(str(fig_root / f"{stem}.png"))
    return written


# ── JSON writing (strip numpy scratch) ─────────────────────────────────────────


def _write_files(files: dict, eval_root: Path) -> list[str]:
    outdir = eval_root / "honest_nulls"
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for fkey, fd in files.items():
        path = outdir / f"{fkey}_honestnulls.json"
        # merge with any existing (two-stage): the other stage's keys survive
        merged = {}
        if path.exists():
            with open(path) as pf:
                merged = json.load(pf)
        merged.update(fd)
        with open(path, "w") as f:
            json.dump(merged, f, indent=2, default=_json_default)
        written.append(str(path))
    return written


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


# ── Self-test ──────────────────────────────────────────────────────────────────


def selftest(out_root: Path, eval_root: Path) -> None:
    """Pin the vectorized sampler against a 50-draw serial reference (rtol 1e-6)
    and against the library ``randnorm_null_draws`` (bit-identical repro)."""
    trait = "evil"
    rb = _load_rb(out_root, trait)
    rb_norms = np.linalg.norm(rb, axis=1)
    rb_hat = {layer: rb[layer] / np.linalg.norm(rb[layer]) for layer in range(N_LAYERS)}
    pos, neg = _load_pools(out_root, trait)
    S, y, _tags = _load_finetune(out_root, eval_root, trait)
    layers = [10, 20, 23]
    lam = nb.PRIMARY_LAMBDA
    pool = np.concatenate([pos, neg], axis=0)
    within_pool = _within_centered_pool(pos, neg)
    chols_pool = _chols_for_layers(pool, layers, lam)
    chols_within = _chols_for_layers(within_pool, layers, lam)
    chols_neg = _chols_for_layers(neg, layers, lam)
    for name, kw in [
        ("isotropic", dict(chols=None, isotropic=True)),
        ("within_class", dict(chols=chols_within)),
        ("neg_arm_only", dict(chols=chols_neg)),
        ("rb_projected_out", dict(chols=chols_pool, project_out_hat=rb_hat)),
        ("orig_randnorm", dict(chols=chols_pool)),
    ]:
        vec = _cov_null_draws(
            rb_norms=rb_norms, predictor=S, target=y, layers=layers, n_draws=50, seed=7, **kw
        )
        ser = _cov_null_draws_serial(
            rb_norms=rb_norms, predictor=S, target=y, layers=layers, n_draws=50, seed=7, **kw
        )
        assert np.allclose(vec, ser, rtol=1e-6, atol=1e-9), f"{name}: vec != serial"
        logger.info(
            "selftest %s: vectorized == serial (max abs diff %.2e)", name, np.abs(vec - ser).max()
        )
    # library repro: full 28 layers, pooled chol, seed=0
    chols_full = _chols_for_layers(pool, list(range(N_LAYERS)), lam)
    mine = _cov_null_draws(
        chols=chols_full,
        rb_norms=rb_norms,
        predictor=S,
        target=y,
        layers=list(range(N_LAYERS)),
        n_draws=50,
        seed=0,
    )
    lib_mat = nb.randnorm_null_draws(
        {layer: pool[:, layer, :] for layer in range(N_LAYERS)},
        rb_norms,
        S,
        y,
        n_draws=50,
        lam=lam,
        seed=0,
    )
    assert np.allclose(mine, lib_mat, rtol=1e-9, atol=1e-12), "orig_randnorm != library"
    logger.info(
        "selftest orig_randnorm: bit-identical to nb.randnorm_null_draws (max diff %.2e)",
        np.abs(mine - lib_mat).max(),
    )
    print("SELFTEST PASS")


# ── Main ────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue #778 honest trait-agnostic null ladder.")
    ap.add_argument("--out-root", default="data/issue_778")
    ap.add_argument("--eval-results-root", default="eval_results/issue_778")
    ap.add_argument("--figures-root", default="figures/issue_778/honest_nulls")
    ap.add_argument(
        "--stage", choices=["fixed", "maxlayer", "selftest", "fixedfigs"], required=True
    )
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument(
        "--only-new",
        action="store_true",
        help="maxlayer: run only the 3 NEW honest stochastic families + crosstrait "
        "(reduce set if wall-clock projects past budget).",
    )
    args = ap.parse_args()
    out_root = Path(args.out_root)
    eval_root = Path(args.eval_results_root)
    (eval_root / "honest_nulls").mkdir(parents=True, exist_ok=True)

    if args.stage == "selftest":
        selftest(out_root, eval_root)
        return

    if args.stage == "fixedfigs":
        figs = build_fixed_figures(eval_root, Path(args.figures_root))
        print(json.dumps({"stage": "fixedfigs", "figures": figs}, indent=2))
        return

    if args.stage == "fixed":
        lib.log_phase("honest_nulls_fixed", f"start draws={args.draws}")
        files = run_fixed_stage(out_root, eval_root, args.draws, args.n_boot)
        _apply_bh(files, "fixed")
        written = _write_files(files, eval_root)
        figs = build_fixed_figures(eval_root, Path(args.figures_root))
        lib.log_phase("done", "fixed stage", n_files=len(written), n_figs=len(figs))
        print(json.dumps({"stage": "fixed", "files": written, "figures": figs}, indent=2))
        return

    # maxlayer
    families = (*NEW_STOCHASTIC, "crosstrait") if args.only_new else LADDER
    lib.log_phase("honest_nulls_maxlayer", f"start draws={args.draws} families={families}")
    files = run_maxlayer_stage(out_root, eval_root, args.draws, families)
    _apply_bh(files, "maxlayer")
    written = _write_files(files, eval_root)
    figs = build_figures(files, eval_root, Path(args.figures_root))
    # completion sentinel for the orchestrator's commit-stage-2 poll
    sentinel = Path("/tmp/issue778-honest-nulls-maxlayer.DONE")
    sentinel.write_text(json.dumps({"files": written, "figures": figs, "ts": time.time()}))
    lib.log_phase("done", "maxlayer stage", n_files=len(written), n_figs=len(figs))
    print(json.dumps({"stage": "maxlayer", "files": written, "figures": figs}, indent=2))


if __name__ == "__main__":
    main()
