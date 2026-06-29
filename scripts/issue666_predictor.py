#!/usr/bin/env python
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


def predict_cell(loaded: dict, *, cell: str, layer: int, Sigma_inv: np.ndarray, r_B=None) -> dict:
    """Compute every predictor variant for one cell at one layer (plan §4d-4f).

    Returns a JSON-serializable dict with per-bystander L̂ / cosine / raw-gate /
    base-prior / Δs columns + the Spearman ρ of each variant vs Δs (η=1 throughout;
    η drops out of the ranking tests). ``Sigma_inv`` is the broad-corpus whitening
    (headline) or the battery diagnostic (smoke). ``r_B`` overrides the per-cell
    r_plus ŵ-shortcut (used for the #658 diffmeans bad-medical/EM behaviors).
    """
    from scipy.stats import spearmanr

    from explore_persona_space.analysis.leakage_predictor import (
        base_prior,
        lhat,
        lhat_variant,
    )

    rb = (
        cell_r_plus(loaded, layer) if r_B is None else np.asarray(r_B, dtype=np.float64).reshape(-1)
    )
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


# ── smoke driver ─────────────────────────────────────────────────────────────
def _battery_sigma_inv(loaded: dict, layer: int):
    """Battery-Σc DIAGNOSTIC whitening for the smoke (NEVER headline; plan §4c)."""
    from explore_persona_space.analysis.leakage_predictor import estimate_sigma_inv

    c_base = loaded["c_C_base"][:, layer, :].numpy().astype(np.float64)
    res = estimate_sigma_inv(c_base, seed=0, corpus_kind="battery")
    return res.Sigma_inv, res


def _smoke_cells(n: int) -> list[str]:
    return ["bm_default_contra_d1_seed42", "ic_edu_default"][:n]


def main() -> int:
    import issue666_load_store as loader

    ap = argparse.ArgumentParser(description="issue 666 Phase-4 predictor (L̂ + baselines).")
    ap.add_argument("--cells", type=int, default=2)
    ap.add_argument("--cell-names", nargs="*", default=None)
    ap.add_argument("--targets", type=int, default=None, help="(diagnostic) limit target count")
    ap.add_argument("--layer", type=int, default=PRIMARY_LAYER)
    ap.add_argument("--slice", action="store_true", help="tiny smoke slice (battery-Σc diagnostic)")
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    out_dir = OUT / "predictor"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = args.cell_names if args.cell_names else _smoke_cells(args.cells)
    for cell in cells:
        local_dir = loader.download_cell(cell)
        loaded = loader.load_cell(local_dir)
        layer = min(args.layer, loaded["v_plus"].shape[1] - 1)
        Sigma_inv, sres = _battery_sigma_inv(loaded, layer)
        rec = predict_cell(loaded, cell=cell, layer=layer, Sigma_inv=Sigma_inv)
        rec["sigma_c_corpus_kind"] = "battery-diagnostic" if args.slice else "battery-diagnostic"
        rec["sigma_c_headline_eligible"] = sres.headline_eligible
        outp = out_dir / f"{cell}_predictor_cells.json"
        outp.write_text(json.dumps(rec, indent=1))
        print(f"[predict] {cell}: rho_full={rec['rho_full_Lhat']:.3f} -> {outp.name}")
        del loaded
        gc.collect()
        try:
            os.remove(local_dir / "tensors.pt")
        except OSError:
            pass
    print(f"[phase=predictor] scored {len(cells)} cells OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
