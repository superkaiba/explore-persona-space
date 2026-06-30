#!/usr/bin/env python
# ruff: noqa: RUF002, RUF003
# Intentional scientific Unicode (Σ, ρ, λ, η, δ, ŵ, ×, −, ⁻¹, ᵀ, ⁺) in docstrings/comments.
"""issue #666 Phase 4 — assemble the full leakage predictor L̂ + baselines, per cell.

Iterates over store cells × target contexts, computing — at the PRIMARY layer 14 —
the full predictor L̂ = η·(r_{B'}ᵀδ)·g_C(C'), the apples-to-apples cosine variant
(three C7 toggles), the raw cosine gate, the base-behavior prior E0, and the
shuffled-key / shuffled-query controls (plan §4d-4f, §5). The latent ground truth
is Δs = r_{B'}ᵀ(v⁺(C')−v0(C')). The PRIMARY metric is Spearman ρ of L̂ vs Δs over
the bystander contexts (the source anchor is EXCLUDED — ĝ^real(C)=1 by construction).

The whitened gate uses the broad-corpus Σc⁻¹ from ``issue666_corpus_extract.py``
(headline); a battery-Σc diagnostic is the smoke fallback. ``r_B`` is the MIXED
per-behavior source (plan §4b): #658 diffmeans for bad-medical/EM, the per-cell
``r_plus`` ŵ-shortcut for taught-fact/marker.

Designed-null control arm (Must-Fix 2): the 2 install-matched signal-free #664
cells ``ic_edu_default`` / ``tf_rev_default`` run through the SAME pipeline; a real
content behavior's L̂ ρ MUST EXCEED the designed-null ρ (clustered CIs, §6) for the
geometry-win headline.

Also hosts the family-clustered + naive bootstrap CIs (plan §6 C4): the 50 battery
contexts cluster by family, so headline CIs resample FAMILIES, not contexts.

CPU-only; reuses ``issue664_aggregate_gate.gate_per_layer`` for ĝ^real and the
shared ``leakage_predictor`` module for L̂.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
PRIMARY_LAYER = 14  # plan §11 (Source: #664 fixed-layer read + cross-phase comparability)
OUT = REPO / "eval_results" / "issue_666"

# The 2 install-matched, signal-free #664 designed-null cells (the install-leak
# control arm; plan §4d/§5/§6). The test binds this exact set.
DESIGNED_NULL_CELLS = ("ic_edu_default", "tf_rev_default")

# The #664 store-cell enumeration target (plan §4a/§4d): all cells present under
# this prefix on the HF data repo (48 confirmed via list_repo_files).
DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_PREFIX = "theory_assumptions/Qwen2.5-7B-Instruct/issue664"

# Mixed r_B routing per behavior (plan §4b): the #658 difference-in-means
# read-out direction for the two behaviors it covers, the per-cell r_plus
# ŵ-shortcut for the rest. Keyed by the store meta.json ``behavior`` field.
#   bad_medical -> harmful_compliance diffmeans; em -> broad_em diffmeans;
#   fact + marker -> per-cell r_plus (no #658 direction exists for them);
#   ic_edu (designed null, EM base) -> broad_em diffmeans (shares the EM recipe);
#   tf_rev (designed null, fact base) -> r_plus (shares the fact recipe).
BEHAVIOR_DIFFMEANS_COLUMN = {
    "bad_medical": "harmful_compliance",
    "em": "broad_em",
    "ic_edu": "broad_em",
}
RPLUS_BEHAVIORS = ("fact", "marker", "tf_rev")

# The behaviors the cross-behavior target grid (plan §4d/§5) scores L̂ against:
# the #658 diffmeans columns the predictor uses (broad_em + harmful_compliance)
# PLUS the two r_plus-only behaviors (fact + marker). This is the
# ``rb_columns ∪ {fact, marker}`` target set of finding 4.
GRID_TARGET_BEHAVIORS = ("bad_medical", "em", "fact", "marker")


# ── meta helpers ─────────────────────────────────────────────────────────────
def _roles_dict(meta: dict) -> dict:
    """Normalize meta.target_context_roles to a {context_id: role} dict.

    The real #664 store carries a dict (cid->role); the test fabricates a list
    (role per context index). Returns a dict keyed by context_id where possible,
    else by integer index.
    """
    tcr = meta.get("target_context_roles")
    if isinstance(tcr, dict):
        return dict(tcr)
    if isinstance(tcr, list):
        return {i: r for i, r in enumerate(tcr)}
    return {}


def _source_idx(loaded: dict) -> int:
    """The index of the source-anchor context (ĝ^real(C)=1 by construction)."""
    meta = loaded["meta"]
    roles = _roles_dict(meta)
    ctx_ids = list(loaded.get("context_ids", range(loaded["v_plus"].shape[0])))
    anchors = [cid for cid, r in roles.items() if r == "source-anchor"]
    if len(anchors) == 1:
        a = anchors[0]
        if isinstance(a, int) and a < len(ctx_ids):
            return a
        if a in ctx_ids:
            return ctx_ids.index(a)
    # explicit source_idx in meta (test path), else 0.
    if "source_idx" in meta:
        return int(meta["source_idx"])
    return 0


def _bystander_mask(loaded: dict) -> np.ndarray:
    """Boolean (n_ctx,) mask excluding the source anchor (plan §4d disjointness)."""
    n = loaded["v_plus"].shape[0]
    mask = np.ones(n, dtype=bool)
    mask[_source_idx(loaded)] = False
    return mask


# ── latent ground truth + predictor scoring per cell ─────────────────────────
def latent_ds(loaded: dict, r_B: np.ndarray, layer: int) -> np.ndarray:
    """Latent leakage ground truth Δs(C') = r_{B'}ᵀ(v⁺(C')−v0(C')) per context (plan §6).

    Returns an (n_ctx,) array at the given layer.
    """
    vp = loaded["v_plus"][:, layer, :].numpy().astype(np.float64)  # (n_ctx, d)
    v0 = loaded["v0"][:, layer, :].numpy().astype(np.float64)
    dv = vp - v0  # (n_ctx, d)
    rb = np.asarray(r_B, dtype=np.float64).reshape(-1)
    return dv @ rb


def cell_r_plus(loaded: dict, layer: int) -> np.ndarray:
    """The per-cell r_plus (A3.7 ŵ-shortcut) at the layer — r_B for fact/marker."""
    return loaded["r_plus"][layer, :].numpy().astype(np.float64)


def rb_for_cell(
    loaded: dict,
    layer: int,
    *,
    rb_columns: dict | None = None,
    r_b_source: str = "mixed",
) -> tuple[np.ndarray, str]:
    """Route the per-behavior r_{B'} source for a cell (plan §4b mixed coverage).

    Returns ``(r_B, source_label)`` where ``source_label`` ∈ {"diffmeans",
    "r_plus"} records which path fired (the cross-arm-heterogeneity caveat
    annotation, §4b/§4l).

    Routing (driven by the store ``meta.behavior`` field):
      - ``r_b_source="diffmeans"``: force the #658 diffmeans column where one
        exists (``BEHAVIOR_DIFFMEANS_COLUMN``); fall back to r_plus only when no
        column maps (fact/marker/tf_rev have none).
      - ``r_b_source="r_plus"``: force the per-cell r_plus ŵ-shortcut for EVERY
        behavior (the within-behavior sensitivity arm for bad-medical/EM, §4b).
      - ``r_b_source="mixed"`` (PRODUCTION default): diffmeans for the behaviors
        #658 covers (bad_medical/em/ic_edu), r_plus for the rest.

    ``rb_columns`` is the ``{column_name: np.ndarray(d,)}`` map from
    ``issue666_load_store.load_rb_columns`` (the diffmeans directions). It is
    REQUIRED whenever a diffmeans path is selected; when None the function falls
    back to r_plus for every behavior (the smoke/no-network path).
    """
    behavior = (loaded.get("meta") or {}).get("behavior")
    col = BEHAVIOR_DIFFMEANS_COLUMN.get(behavior)
    want_diffmeans = r_b_source == "diffmeans" or (r_b_source == "mixed" and col is not None)
    if want_diffmeans and col is not None and rb_columns is not None and col in rb_columns:
        return np.asarray(rb_columns[col], dtype=np.float64).reshape(-1), "diffmeans"
    # r_plus path (forced, or no diffmeans column / no rb_columns available).
    return cell_r_plus(loaded, layer), "r_plus"


def score_cell_lhat_vs_ds(loaded: dict, layer: int = PRIMARY_LAYER, r_B=None) -> float:
    """Spearman ρ of L̂ vs the latent Δs over the bystander contexts (plan §6).

    The relative (ranking) test, so η drops out and the gate uses the per-cell
    last-input keys with identity whitening (the relative real-vs-null comparison;
    the headline run threads the broad-corpus Σc⁻¹). ``r_B`` defaults to the
    per-cell ``r_plus`` ŵ-shortcut (the fact/marker path); pass the #658 diffmeans
    direction for bad-medical/EM. Returns the Spearman ρ (float).
    """
    from scipy.stats import spearmanr

    from explore_persona_space.analysis.leakage_predictor import lhat

    rb = (
        cell_r_plus(loaded, layer) if r_B is None else np.asarray(r_B, dtype=np.float64).reshape(-1)
    )
    delta = (loaded["t_CB"][layer, :].numpy().astype(np.float64)) - (
        loaded["v0"][_source_idx(loaded), layer, :].numpy().astype(np.float64)
    )
    c_base = loaded["c_C_base"][:, layer, :].numpy().astype(np.float64)  # (n_ctx, d)
    src = _source_idx(loaded)
    c_C = c_base[src]
    d = c_C.shape[0]
    Sigma_inv = np.eye(d)  # identity gate for the relative real-vs-null scoring
    mask = _bystander_mask(loaded)

    ds = latent_ds(loaded, rb, layer)
    lh = np.array(
        [
            lhat(eta=1.0, r_Bp=rb, delta=delta, c_C=c_C, c_Cp=c_base[i], Sigma_inv=Sigma_inv)
            for i in range(len(c_base))
        ]
    )
    rho = spearmanr(lh[mask], ds[mask]).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def geometry_win_verdict(*, real_rho, real_ci, null_rho, null_ci) -> str:
    """The §6 install-leak gate verdict.

    A real content behavior's L̂ ρ must EXCEED the designed-null ρ with
    non-overlapping clustered CIs: the real CI's LOWER bound above the null's POINT
    estimate → "geometry-win"; otherwise → "install-confounded" (the install-leak
    alternative is not ruled out). Returns one of those two strings.
    """
    real_lo = float(real_ci[0])
    if real_lo > float(null_rho) and float(real_rho) > float(null_rho):
        return "geometry-win"
    return "install-confounded"


# ── clustered + naive bootstrap CIs (plan §6 C4) ─────────────────────────────
def draw_families(family_ids, rng, n_draw):
    """Draw ``n_draw`` family ids WITH replacement (the cluster-resampling unit).

    A hook the clustered bootstrap calls per replicate so the resampling unit is a
    FAMILY (the contexts within a drawn family appear together or not at all),
    NEVER an individual context (plan §6: "resample at cluster level, NEVER naive
    n=50"). ``family_ids`` is the array of UNIQUE family labels.
    """
    return rng.choice(family_ids, size=n_draw, replace=True)


def _statistic(x: np.ndarray, y: np.ndarray, statistic: str) -> float:
    if statistic == "spearman":
        from scipy.stats import spearmanr

        r = spearmanr(x, y).statistic
    elif statistic == "pearson":
        from scipy.stats import pearsonr

        r = pearsonr(x, y).statistic
    else:
        raise ValueError(f"unknown statistic {statistic!r}")
    return float(r) if np.isfinite(r) else 0.0


def clustered_bootstrap_ci(
    x, y, *, clusters, n_boot=2000, seed=0, statistic="spearman", alpha=0.05
):
    """Family-clustered bootstrap CI (plan §6 C4).

    Resamples FAMILIES (clusters), then takes all member rows of each drawn family
    — so within-family correlation deflates the effective n and the CI is WIDER
    than the naive independent-row CI. Returns ``(lo, hi)`` percentile CI of the
    statistic. The family draw goes through ``draw_families`` (a monkeypatchable
    hook).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    clusters = np.asarray(clusters)
    fam_ids = np.unique(clusters)
    members = {f: np.where(clusters == f)[0] for f in fam_ids}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        drawn = draw_families(fam_ids, rng, len(fam_ids))
        idx = np.concatenate([members[f] for f in drawn])
        if idx.size < 3:
            continue
        stats.append(_statistic(x[idx], y[idx], statistic))
    stats = np.asarray(stats)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


def naive_bootstrap_ci(x, y, *, n_boot=2000, seed=0, statistic="spearman", alpha=0.05):
    """Naive (independent-row) bootstrap CI — the OVER-confident comparator.

    Resamples the n rows independently (ignores the family clustering), so it
    over-counts the effective d.o.f. Returns ``(lo, hi)``. Shown alongside the
    clustered CI to expose the effective-n deflation; NEVER the headline.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.unique(idx).size < 3:
            continue
        stats.append(_statistic(x[idx], y[idx], statistic))
    stats = np.asarray(stats)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


# ── full per-cell predictor pass (all variants) ──────────────────────────────
def _family_labels(loaded: dict) -> np.ndarray:
    """Per-context family label from the context-id prefix (f1_house_... -> f1).

    The #594 battery context ids encode the family in the leading token; this gives
    the cluster id for the clustered bootstrap. Bystander order matches the masked
    arrays. Falls back to a single family when ids are integer (test path).
    """
    ctx_ids = list(loaded.get("context_ids", []))
    if ctx_ids and isinstance(ctx_ids[0], str):
        fams = np.array([str(c).split("_")[0] for c in ctx_ids])
    else:
        fams = np.zeros(loaded["v_plus"].shape[0], dtype=int)
    return fams


def predict_cell(
    loaded: dict,
    *,
    cell: str,
    layer: int,
    Sigma_inv: np.ndarray,
    r_B=None,
    r_B_source: str | None = None,
) -> dict:
    """Compute every predictor variant for one cell at one layer (plan §4d-4f).

    Returns a JSON-serializable dict with per-bystander L̂ / cosine / raw-gate /
    base-prior / Δs columns + the Spearman ρ of each variant vs Δs (η=1 throughout;
    η drops out of the ranking tests). ``Sigma_inv`` is the broad-corpus whitening
    (headline) or the battery diagnostic (smoke). ``r_B`` overrides the per-cell
    r_plus ŵ-shortcut (used for the #658 diffmeans bad-medical/EM behaviors);
    ``r_B_source`` ∈ {"diffmeans", "r_plus"} records which mixed-source path fired
    (the §4b cross-arm-heterogeneity annotation, carried into the result JSON).
    """
    from scipy.stats import spearmanr

    from explore_persona_space.analysis.leakage_predictor import (
        base_prior,
        lhat,
        lhat_variant,
    )

    if r_B is None:
        rb = cell_r_plus(loaded, layer)
        r_B_source = r_B_source or "r_plus"
    else:
        rb = np.asarray(r_B, dtype=np.float64).reshape(-1)
        r_B_source = r_B_source or "diffmeans"
    src = _source_idx(loaded)
    delta = (loaded["t_CB"][layer, :].numpy().astype(np.float64)) - (
        loaded["v0"][src, layer, :].numpy().astype(np.float64)
    )
    c_base = loaded["c_C_base"][:, layer, :].numpy().astype(np.float64)
    v0_layer = loaded["v0"][:, layer, :].numpy().astype(np.float64)
    c_C = c_base[src]
    n = c_base.shape[0]
    ds = latent_ds(loaded, rb, layer)

    full = np.empty(n)
    cos = np.empty(n)
    raw_gate = np.empty(n)
    prior = np.empty(n)
    eye = np.eye(c_C.shape[0])
    for i in range(n):
        full[i] = lhat(eta=1.0, r_Bp=rb, delta=delta, c_C=c_C, c_Cp=c_base[i], Sigma_inv=Sigma_inv)
        cos[i] = lhat_variant(
            eta=1.0,
            r_Bp=rb,
            r_B=rb,
            delta=delta,
            c_C=c_C,
            c_Cp=c_base[i],
            Sigma_inv=Sigma_inv,
            toggle_delta_to_rB=True,
            drop_norms=True,
            toggle_sigma_to_identity=True,
        )
        raw_gate[i] = lhat_variant(
            eta=1.0,
            r_Bp=rb,
            r_B=rb,
            delta=delta,
            c_C=c_C,
            c_Cp=c_base[i],
            Sigma_inv=eye,
            toggle_sigma_to_identity=True,
        )
        prior[i] = base_prior(r_Bp=rb, v0_Cp=v0_layer[i])

    mask = _bystander_mask(loaded)
    fams = _family_labels(loaded)

    def _rho(arr):
        r = spearmanr(arr[mask], ds[mask]).statistic
        return float(r) if np.isfinite(r) else 0.0

    return {
        "cell": cell,
        "layer": layer,
        "behavior": loaded["meta"].get("behavior"),
        "source": loaded["meta"].get("source"),
        "r_B_source": r_B_source,
        "n_bystanders": int(mask.sum()),
        "rho_full_Lhat": _rho(full),
        "rho_cosine": _rho(cos),
        "rho_raw_gate": _rho(raw_gate),
        "rho_base_prior": _rho(prior),
        "per_bystander": {
            "context_family": fams[mask].tolist(),
            "Lhat": full[mask].round(6).tolist(),
            "cosine": cos[mask].round(6).tolist(),
            "base_prior": prior[mask].round(6).tolist(),
            "ds": ds[mask].round(6).tolist(),
        },
    }


def _rb_for_target_behavior(
    target_behavior: str, loaded: dict, layer: int, rb_columns: dict | None
) -> np.ndarray | None:
    """The read-out direction r_{B'} for a TARGET behavior in the cross-behavior grid.

    Diffmeans columns (bad_medical -> harmful_compliance, em -> broad_em) come from
    ``rb_columns``; fact/marker have no #658 direction, so the cell's own per-cell
    ``r_plus`` is the displacement-direction estimate (the A3.7 ŵ-shortcut, §4b).
    Returns None when a diffmeans target is requested but ``rb_columns`` lacks it
    (the smoke/no-network path) — the caller skips that target.
    """
    col = BEHAVIOR_DIFFMEANS_COLUMN.get(target_behavior)
    if col is not None:
        if rb_columns is not None and col in rb_columns:
            return np.asarray(rb_columns[col], dtype=np.float64).reshape(-1)
        return None  # diffmeans target unavailable in this run
    # fact / marker: no #658 direction -> the per-cell r_plus shortcut.
    return cell_r_plus(loaded, layer)


def predict_cell_grid(
    loaded: dict,
    *,
    cell: str,
    layer: int,
    Sigma_inv: np.ndarray,
    rb_columns: dict | None = None,
    r_b_source: str = "mixed",
    target_behaviors: tuple[str, ...] = GRID_TARGET_BEHAVIORS,
) -> dict:
    """Cross-behavior leakage matrix for one SOURCE cell (plan §4d/§5, finding 4).

    The cell's OWN-behavior read is the headline (top-level ``rho_*`` keys, routed
    by ``rb_for_cell`` via ``r_b_source``); the ``per_target`` map then scores L̂
    for EVERY target behavior B' in ``target_behaviors`` — the behavior-transfer
    factor varying B' at a fixed source C (§4d "vary B' at fixed C"). Each target
    re-uses the cell's δ and gate but swaps in r_{B'} (so Δs is the B'-specific
    latent ground truth ``r_{B'}ᵀ(v⁺−v0)``).

    ``per_target[<behavior>]`` carries ``{rho_full_Lhat, rho_cosine, rho_raw_gate,
    rho_base_prior, r_B_source}``. A diffmeans target with no column available in
    this run (smoke/no-network) is OMITTED from ``per_target`` rather than faked.
    Returns the own-behavior ``predict_cell`` record AUGMENTED with ``per_target``.
    """
    own_rb, own_src = rb_for_cell(loaded, layer, rb_columns=rb_columns, r_b_source=r_b_source)
    rec = predict_cell(
        loaded, cell=cell, layer=layer, Sigma_inv=Sigma_inv, r_B=own_rb, r_B_source=own_src
    )
    per_target: dict = {}
    for tb in target_behaviors:
        rb_t = _rb_for_target_behavior(tb, loaded, layer, rb_columns)
        if rb_t is None:
            continue
        tsrc = "diffmeans" if tb in BEHAVIOR_DIFFMEANS_COLUMN else "r_plus"
        t_rec = predict_cell(
            loaded, cell=cell, layer=layer, Sigma_inv=Sigma_inv, r_B=rb_t, r_B_source=tsrc
        )
        per_target[tb] = {
            "rho_full_Lhat": t_rec["rho_full_Lhat"],
            "rho_cosine": t_rec["rho_cosine"],
            "rho_raw_gate": t_rec["rho_raw_gate"],
            "rho_base_prior": t_rec["rho_base_prior"],
            "r_B_source": t_rec["r_B_source"],
        }
    rec["per_target"] = per_target
    return rec


# ── store-cell enumeration (plan §4a/§4d, finding 2) ─────────────────────────
def enumerate_store_cells() -> list[str]:
    """Every #664 store cell present on the HF data repo (plan §4a/§4d, finding 2).

    Scans ``list_repo_files`` under the store prefix for ``*/tensors.pt`` and
    returns the sorted cell-dir names — the production headline enumeration (48
    cells: ~44 content + the 4 designed-null dirs). This is the cell set used when
    ``main`` is invoked with no ``--cells`` / ``--cell-names`` / ``--slice``
    override. Raises if the listing resolves empty (a fail-loud guard, never a
    silent zero).
    """
    from huggingface_hub import list_repo_files

    files = list_repo_files(DATA_REPO, repo_type="dataset")
    cells = sorted(
        {
            f[len(STORE_PREFIX) + 1 :].rsplit("/", 1)[0]
            for f in files
            if f.startswith(STORE_PREFIX + "/") and f.endswith("/tensors.pt")
        }
    )
    if not cells:
        raise RuntimeError(f"no #664 store cells resolved under {DATA_REPO}/{STORE_PREFIX}")
    return cells


# ── smoke driver ─────────────────────────────────────────────────────────────
def _battery_sigma_inv(loaded: dict, layer: int):
    """Battery-Σc DIAGNOSTIC whitening for the smoke (NEVER headline; plan §4c)."""
    from explore_persona_space.analysis.leakage_predictor import estimate_sigma_inv

    c_base = loaded["c_C_base"][:, layer, :].numpy().astype(np.float64)
    res = estimate_sigma_inv(c_base, seed=0, corpus_kind="battery")
    return res.Sigma_inv, res


def _smoke_cells(n: int) -> list[str]:
    """A tiny slice for ``--slice`` mode only (one content + one designed-null)."""
    return ["bm_default_contra_d1_seed42", "ic_edu_default"][:n]


def load_sigma_inv(path: str | Path, layer: int) -> tuple[np.ndarray, dict]:
    """Load the broad-corpus Σc⁻¹ from ``issue666_corpus_extract``'s output (finding 1).

    ``path`` points at the ``sigma_c_inv.pt`` written by ``issue666_corpus_extract``
    (a dict with ``Sigma_inv`` (d, d) + ``headline_eligible`` + ``corpus`` metadata).
    Returns ``(Sigma_inv_float64, meta)`` where ``meta`` carries the corpus-kind +
    headline-eligibility + conditioning fields for the per-cell result JSON. Asserts
    the stored Σc⁻¹ is square at the store dimension. ``layer`` is recorded for the
    provenance check (the stored Σc⁻¹ is already the layer-14 broad-corpus inverse).
    """
    import torch

    obj = torch.load(Path(path), map_location="cpu")
    si = obj["Sigma_inv"]
    si = si.detach().cpu().numpy() if hasattr(si, "detach") else np.asarray(si)
    si = si.astype(np.float64)
    assert si.ndim == 2 and si.shape[0] == si.shape[1], (
        f"Sigma_inv must be a square (d, d) matrix, got {si.shape}"
    )
    meta = {
        "sigma_c_corpus_kind": "broad",
        "sigma_c_headline_eligible": bool(obj.get("headline_eligible", True)),
        "sigma_c_lam": float(obj["lam"]) if "lam" in obj else None,
        "sigma_c_cond_number": float(obj["cond_number"]) if "cond_number" in obj else None,
        "sigma_c_n_contexts": int(obj["n_contexts"]) if "n_contexts" in obj else None,
        "sigma_c_layer": int(obj["layer"]) if "layer" in obj else layer,
    }
    return si, meta


def _resolve_cells(args) -> tuple[list[str], bool]:
    """Resolve the cell list + whether this is a full-enumeration production run.

    Precedence (finding 2): ``--cell-names`` > ``--slice`` (the tiny ``_smoke_cells``
    slice) > explicit ``--cells N`` > [none of the above] = the FULL HF store
    enumeration (the production headline). Returns ``(cells, is_full)``.
    """
    if args.cell_names:
        return list(args.cell_names), False
    if args.slice:
        return _smoke_cells(args.cells if args.cells else 1), False
    if args.cells is not None:
        return _smoke_cells(args.cells), False
    return enumerate_store_cells(), True


def main() -> int:
    import issue666_load_store as loader

    ap = argparse.ArgumentParser(description="issue 666 Phase-4 predictor (L̂ + baselines).")
    # --cells default is None so "no flags" => full store enumeration (finding 2).
    ap.add_argument("--cells", type=int, default=None, help="limit to first N smoke cells")
    ap.add_argument("--cell-names", nargs="*", default=None)
    ap.add_argument("--targets", type=int, default=None, help="(diagnostic) limit target count")
    ap.add_argument("--layer", type=int, default=PRIMARY_LAYER)
    ap.add_argument("--slice", action="store_true", help="tiny smoke slice (battery-Σc diagnostic)")
    ap.add_argument(
        "--sigma-inv",
        default=None,
        help="path to the broad-corpus Σc⁻¹ (sigma_c_inv.pt) — headline whitening (finding 1)",
    )
    ap.add_argument(
        "--r-b-source",
        choices=("diffmeans", "r_plus", "mixed"),
        default="mixed",
        help="r_B routing (finding 3): mixed=production, diffmeans/r_plus=sensitivity arms",
    )
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    out_dir = OUT / "predictor"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells, is_full = _resolve_cells(args)

    # The #658 diffmeans r_B columns (finding 3) — verified + loaded ONCE. Needed
    # whenever a diffmeans path can fire (mixed/diffmeans). On the no-network smoke
    # (--slice with no --sigma-inv) we fall back to r_plus everywhere, so the
    # network read is skipped to keep the CPU smoke offline.
    rb_columns = None
    want_diffmeans = args.r_b_source in ("diffmeans", "mixed")
    if want_diffmeans and not (args.slice and args.sigma_inv is None):
        try:
            rb_columns = loader.load_rb_columns(layer=args.layer)
            print(f"[predict] loaded #658 diffmeans columns: {sorted(rb_columns)}")
        except Exception as exc:
            if is_full:
                raise  # production headline MUST have the diffmeans directions
            print(f"[predict] WARN: #658 r_b.pt unavailable ({exc}); r_plus everywhere")

    # The whitening Σc⁻¹ (finding 1): broad-corpus headline from --sigma-inv, else
    # the per-cell battery-Σc DIAGNOSTIC (never headline; plan §4c).
    sigma_inv_headline = None
    sigma_meta = None
    if args.sigma_inv is not None:
        sigma_inv_headline, sigma_meta = load_sigma_inv(args.sigma_inv, args.layer)
        print(
            f"[predict] broad-corpus Σc⁻¹ {sigma_inv_headline.shape} from {args.sigma_inv} "
            f"(headline_eligible={sigma_meta['sigma_c_headline_eligible']})"
        )

    records: list[dict] = []
    for cell in cells:
        local_dir = loader.download_cell(cell)
        loaded = loader.load_cell(local_dir)
        layer = min(args.layer, loaded["v_plus"].shape[1] - 1)
        if sigma_inv_headline is not None:
            Sigma_inv = sigma_inv_headline
            corpus_meta = dict(sigma_meta)
        else:
            Sigma_inv, sres = _battery_sigma_inv(loaded, layer)
            corpus_meta = {
                "sigma_c_corpus_kind": "battery-diagnostic",
                "sigma_c_headline_eligible": sres.headline_eligible,
            }
        rec = predict_cell_grid(
            loaded,
            cell=cell,
            layer=layer,
            Sigma_inv=Sigma_inv,
            rb_columns=rb_columns,
            r_b_source=args.r_b_source,
        )
        rec.update(corpus_meta)
        outp = out_dir / f"{cell}_predictor_cells.json"
        outp.write_text(json.dumps(rec, indent=1))
        records.append(rec)
        print(f"[predict] {cell}: rho_full={rec['rho_full_Lhat']:.3f} -> {outp.name}")
        del loaded
        gc.collect()
        with contextlib.suppress(OSError):
            os.remove(local_dir / "tensors.pt")

    # PRIMARY deliverable (plan §6.5): the per-behavior headline table — one block
    # per behavior arm + the cross-behavior aggregate (SECONDARY, mixed-r_B caveat).
    headline = build_headline(records, sigma_meta=sigma_meta, r_b_source=args.r_b_source)
    hd_dir = OUT / "headline"
    hd_dir.mkdir(parents=True, exist_ok=True)
    hd_path = hd_dir / "predictor_headline.json"
    hd_path.write_text(json.dumps(headline, indent=1))
    print(f"[predict] headline -> {hd_path}")

    print(f"[phase=predictor] scored {len(cells)} cells OK")
    return 0


def build_headline(records: list[dict], *, sigma_meta: dict | None, r_b_source: str) -> dict:
    """The §6.5 PRIMARY headline table from the per-cell records.

    Per behavior arm: the mean own-behavior L̂/cosine/base-prior Spearman ρ over its
    cells + which r_B source each used. PLUS the cross-behavior aggregate (pooled
    own-behavior ρ across behaviors, flagged SECONDARY — it mixes diffmeans + r_plus
    sources, the §4b/§6 heterogeneity caveat). Reproducibility metadata (git commit,
    timestamp, Σc-corpus kind) is embedded per CLAUDE.md. The designed-null arm + the
    geometry-win verdict are computed by ``issue666_designed_null`` against the same
    per-cell JSONs; this table records the per-behavior PRIMARY reads.
    """
    import datetime as _dt
    import subprocess

    by_beh: dict[str, dict[str, list]] = {}
    for r in records:
        beh = r.get("behavior") or "unknown"
        d = by_beh.setdefault(
            beh,
            {"rho_full_Lhat": [], "rho_cosine": [], "rho_base_prior": [], "r_B_source": set()},
        )
        for k in ("rho_full_Lhat", "rho_cosine", "rho_base_prior"):
            d[k].append(float(r.get(k, np.nan)))
        d["r_B_source"].add(r.get("r_B_source"))

    per_behavior = {}
    for beh, d in sorted(by_beh.items()):
        per_behavior[beh] = {
            "n_cells": len(d["rho_full_Lhat"]),
            "rho_full_Lhat_mean": float(np.nanmean(d["rho_full_Lhat"])),
            "rho_cosine_mean": float(np.nanmean(d["rho_cosine"])),
            "rho_base_prior_mean": float(np.nanmean(d["rho_base_prior"])),
            "r_B_source": sorted(s for s in d["r_B_source"] if s),
        }

    content = [b for b in by_beh if b not in ("marker", "tf_rev", "ic_edu")]
    pooled = [v for b in content for v in by_beh[b]["rho_full_Lhat"]]
    aggregate = {
        "rho_full_Lhat_mean": float(np.nanmean(pooled)) if pooled else float("nan"),
        "n_cells": len(pooled),
        "flagged": "SECONDARY",
        "caveat": "mixes diffmeans (bad_medical/em) + r_plus (fact) r_B sources (§4b/§6)",
    }

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    return {
        "schema": "issue666_predictor_headline_v1",
        "per_behavior": per_behavior,
        "cross_behavior_aggregate": aggregate,
        "r_b_source_mode": r_b_source,
        "sigma_c": sigma_meta or {"sigma_c_corpus_kind": "battery-diagnostic"},
        "n_cells": len(records),
        "reproducibility": {
            "git_commit": commit,
            "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(),
            "primary_dv": "latent ds Spearman rho (per-behavior PRIMARY; aggregate SECONDARY)",
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
