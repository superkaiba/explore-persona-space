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

# Canonical TARGET-behavior source cells for the r_plus-only targets (Blocker 1).
# fact/marker have no #658 diffmeans direction, so their cross-behavior read-out
# direction r_{B'} is the per-cell r_plus ŵ-shortcut of a FIXED canonical cell —
# NOT the SOURCE cell's own r_plus (which would alias an unrelated implant shift,
# trivializing the cross-behavior matrix). The choice is deterministic: the
# default-source, contrastive, dose-1, seed-42 cell of each behavior (the #664
# slug convention `tf`/`mk` + `_default_contra_d1_seed42`). Documented in the
# headline JSON's target_direction_registry metadata + the v3 marker.
CANONICAL_TARGET_SOURCE_CELL = {
    "fact": "tf_default_contra_d1_seed42",
    "marker": "mk_default_contra_d1_seed42",
}


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


def _context_ids(loaded: dict) -> list:
    """The per-context battery ids (``f1_house_librarian`` …) or the integer range.

    The #664 store carries ``context_ids`` (the battery instance ids — the LOCO
    fold key, plan §4g); the test fabricates ``list(range(N_CTX))``. Returns a list
    of length n_ctx. Falls back to the integer range when the tensor is absent.
    """
    cids = loaded.get("context_ids")
    if cids is None:
        return list(range(loaded["v_plus"].shape[0]))
    return list(cids)


def predict_cell(
    loaded: dict,
    *,
    cell: str,
    layer: int,
    Sigma_inv: np.ndarray,
    r_B=None,
    r_B_source: str | None = None,
    source_r_B=None,
) -> dict:
    """Compute every predictor variant for one cell at one layer (plan §4d-4f).

    Returns a JSON-serializable dict with per-bystander L̂ / cosine / raw-gate /
    base-prior / Δs / context_id columns + the Spearman ρ of each variant vs Δs
    (η=1 throughout; η drops out of the ranking tests). ``Sigma_inv`` is the
    broad-corpus whitening (headline) or the battery diagnostic (smoke). ``r_B``
    is the read-out direction r_{B'} of the EVALUATED (target) behavior — it
    overrides the per-cell r_plus ŵ-shortcut (used for the #658 diffmeans
    bad-medical/EM behaviors); ``r_B_source`` ∈ {"diffmeans", "r_plus"} records
    which mixed-source path fired (the §4b cross-arm-heterogeneity annotation,
    carried into the result JSON).

    ``source_r_B`` is the SOURCE cell's OWN r_B (the implant direction that built
    δ). On an OWN-behavior read it equals ``r_B`` (or, when ``r_B`` is the r_plus
    shortcut, the same vector). On a CROSS-behavior read (``r_B`` = a DIFFERENT
    target behavior's direction) it stays the source's direction, so the cosine
    special-case is ``cos(r_{B'}, r_B)·cos(c_C, c_{C'})`` with two DISTINCT
    vectors — never the collapsed ``cos(r_B, r_B)=1`` (Blocker 2, plan §4e). When
    None it defaults to ``r_B`` (the own-behavior case), preserving the legacy
    single-vector behavior.
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
    # source_r_B: the SOURCE's own implant direction (builds δ). The cosine
    # behavior-transfer term cos(r_{B'}, r_B) needs the SOURCE r_B distinct from
    # the TARGET r_{B'}=rb; defaulting to rb is the own-behavior case (cos=1, which
    # is correct there — the source IS the evaluated behavior).
    rb_source_vec = (
        rb if source_r_B is None else np.asarray(source_r_B, dtype=np.float64).reshape(-1)
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
            r_Bp=rb,  # TARGET behavior direction r_{B'}
            r_B=rb_source_vec,  # SOURCE direction r_B (Blocker 2: distinct on cross-behavior)
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
            r_B=rb,  # unused: raw_gate has no toggle_delta_to_rB, so r_B never enters
            delta=delta,
            c_C=c_C,
            c_Cp=c_base[i],
            Sigma_inv=eye,
            toggle_sigma_to_identity=True,
        )
        prior[i] = base_prior(r_Bp=rb, v0_Cp=v0_layer[i])

    mask = _bystander_mask(loaded)
    fams = _family_labels(loaded)
    # Battery context ids per bystander row (Blocker 3 / plan §4g): the LOCO fold
    # key. SOURCE_INSTANCE_IDS makes the masked bystander array DIFFERENT across
    # sources, so a positional row index conflates distinct battery contexts in a
    # multi-source LOCO fold — persist the real id so the fold keys on identity.
    cids = np.asarray(_context_ids(loaded), dtype=object)
    cids_masked = [str(c) for c in cids[mask].tolist()]

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
            "context_id": cids_masked,
            "context_family": fams[mask].tolist(),
            "Lhat": full[mask].round(6).tolist(),
            "cosine": cos[mask].round(6).tolist(),
            "base_prior": prior[mask].round(6).tolist(),
            "ds": ds[mask].round(6).tolist(),
        },
    }


def build_target_direction_registry(
    *,
    layer: int,
    rb_columns: dict | None,
    download_cell=None,
    load_cell=None,
    target_behaviors: tuple[str, ...] = GRID_TARGET_BEHAVIORS,
) -> dict:
    """The TARGET-behavior read-out direction registry r_{B'} (Blocker 1, plan §4d).

    ONE direction per TARGET behavior, SHARED across all source cells — the
    behavior-transfer factor varies B' (the read-out direction) at a fixed source
    C, so the direction must NOT come from the source cell (aliasing the source's
    own implant shift trivializes the cross-behavior matrix — round-2 BLOCKER).

      - diffmeans targets (bad_medical -> harmful_compliance, em -> broad_em): the
        #658 ``rb_columns`` direction (one direction, no source dependence).
      - r_plus-only targets (fact, marker): the per-cell ``r_plus`` of a FIXED
        CANONICAL target-source cell (``CANONICAL_TARGET_SOURCE_CELL``), downloaded
        + read ONCE here. The default-source/contra/d1/seed-42 cell of the behavior.

    Returns ``{behavior: {"r_Bp": np.ndarray(d,), "source": <label>, "from_cell":
    <cell|None>}}`` — a behavior is OMITTED when its direction is unavailable in
    this run (a diffmeans column absent on the no-network smoke, or a canonical
    target cell that fails to resolve), so the caller skips that target rather than
    faking it (plan §4d permits a recorded target-behavior absence).

    ``download_cell`` / ``load_cell`` default to the ``issue666_load_store``
    helpers; the test injects in-memory stubs so no HF read happens.
    """
    if download_cell is None or load_cell is None:
        import issue666_load_store as _ls

        download_cell = download_cell or _ls.download_cell
        load_cell = load_cell or _ls.load_cell

    registry: dict = {}
    for tb in target_behaviors:
        col = BEHAVIOR_DIFFMEANS_COLUMN.get(tb)
        if col is not None:
            # diffmeans target — one direction, no source dependence.
            if rb_columns is not None and col in rb_columns:
                registry[tb] = {
                    "r_Bp": np.asarray(rb_columns[col], dtype=np.float64).reshape(-1),
                    "source": "diffmeans",
                    "from_cell": None,
                }
            # else: omit (diffmeans column unavailable in this run).
            continue
        # r_plus-only target (fact/marker): the CANONICAL target-source cell's
        # r_plus (NOT the source cell's). Downloaded + read once.
        canon = CANONICAL_TARGET_SOURCE_CELL.get(tb)
        if canon is None:
            continue
        try:
            local_dir = download_cell(canon)
            canon_loaded = load_cell(local_dir)
        except Exception as exc:  # canonical cell unresolvable -> omit, don't fake
            print(f"[predict] WARN: canonical target cell {canon!r} for {tb!r} unavailable ({exc})")
            continue
        lyr = min(layer, canon_loaded["v_plus"].shape[1] - 1)
        registry[tb] = {
            "r_Bp": cell_r_plus(canon_loaded, lyr),
            "source": "r_plus_canonical",
            "from_cell": canon,
        }
    return registry


def predict_cell_grid(
    loaded: dict,
    *,
    cell: str,
    layer: int,
    Sigma_inv: np.ndarray,
    rb_columns: dict | None = None,
    r_b_source: str = "mixed",
    target_registry: dict | None = None,
    target_behaviors: tuple[str, ...] = GRID_TARGET_BEHAVIORS,
) -> dict:
    """Cross-behavior leakage matrix for one SOURCE cell (plan §4d/§5, finding 4).

    The cell's OWN-behavior read is the headline (top-level ``rho_*`` keys, routed
    by ``rb_for_cell`` via ``r_b_source``); the ``per_target`` map then scores L̂
    for EVERY target behavior B' present in ``target_registry`` — the
    behavior-transfer factor varying B' at a fixed source C (§4d "vary B' at fixed
    C"). Each target re-uses the cell's δ and gate but swaps in the TARGET
    direction r_{B'} from the registry (so Δs is the B'-specific latent ground
    truth ``r_{B'}ᵀ(v⁺−v0)``), while ``source_r_B`` stays the SOURCE cell's own
    r_B — so the cosine special-case is ``cos(r_{B'}, r_B)·cos(c_C,c_{C'})`` with
    two DISTINCT vectors (Blocker 2), never the collapsed ``cos(r_B,r_B)=1``.

    ``target_registry`` (from ``build_target_direction_registry``) is the shared
    per-target direction map — diffmeans-from-r_b.pt for bad_medical/em, the
    CANONICAL target-source cell's r_plus for fact/marker (NOT the source cell's
    own r_plus — Blocker 1). When None it is built on the fly via ``rb_columns``
    (the diffmeans targets only; r_plus targets need the canonical-cell read, so a
    None registry omits fact/marker — used by the offline test path).

    ``per_target[<behavior>]`` carries ``{rho_full_Lhat, rho_cosine, rho_raw_gate,
    rho_base_prior, r_B_source, from_cell}``. A target with no direction available
    in this run is OMITTED rather than faked. Returns the own-behavior
    ``predict_cell`` record AUGMENTED with ``per_target``.
    """
    own_rb, own_src = rb_for_cell(loaded, layer, rb_columns=rb_columns, r_b_source=r_b_source)
    rec = predict_cell(
        loaded, cell=cell, layer=layer, Sigma_inv=Sigma_inv, r_B=own_rb, r_B_source=own_src
    )
    if target_registry is None:
        # Offline fallback: diffmeans targets only (r_plus targets need the
        # canonical-cell read, which a None registry cannot supply offline).
        target_registry = {
            tb: {
                "r_Bp": np.asarray(
                    rb_columns[BEHAVIOR_DIFFMEANS_COLUMN[tb]], dtype=np.float64
                ).reshape(-1),
                "source": "diffmeans",
                "from_cell": None,
            }
            for tb in target_behaviors
            if tb in BEHAVIOR_DIFFMEANS_COLUMN
            and rb_columns is not None
            and BEHAVIOR_DIFFMEANS_COLUMN[tb] in rb_columns
        }
    per_target: dict = {}
    for tb in target_behaviors:
        entry = target_registry.get(tb)
        if entry is None:
            continue  # target direction unavailable -> recorded absence, not faked
        rb_t = np.asarray(entry["r_Bp"], dtype=np.float64).reshape(-1)
        t_rec = predict_cell(
            loaded,
            cell=cell,
            layer=layer,
            Sigma_inv=Sigma_inv,
            r_B=rb_t,  # TARGET direction r_{B'}
            r_B_source=entry["source"],
            source_r_B=own_rb,  # SOURCE direction r_B (Blocker 2: distinct on cross-behavior)
        )
        per_target[tb] = {
            "rho_full_Lhat": t_rec["rho_full_Lhat"],
            "rho_cosine": t_rec["rho_cosine"],
            "rho_raw_gate": t_rec["rho_raw_gate"],
            "rho_base_prior": t_rec["rho_base_prior"],
            "r_B_source": t_rec["r_B_source"],
            "from_cell": entry.get("from_cell"),
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
    ap.add_argument(
        "--allow-battery-sigma",
        action="store_true",
        help="permit a full-store run WITHOUT --sigma-inv (diagnostic only; per-cell eigh)",
    )
    ap.add_argument(
        "--rb-source-sensitivity",
        action="store_true",
        help=(
            "after the headline, ALSO re-score the bad_medical/em cells under BOTH "
            "diffmeans and r_plus r_B and write headline/rb_source_sensitivity.json "
            "(Concern 5; only meaningful on the mixed/diffmeans production run)"
        ),
    )
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    out_dir = OUT / "predictor"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells, is_full = _resolve_cells(args)

    # Fail-loud headline guard (closes per-cell-battery-eigh-true-cause-of-silent-
    # empty): the full-enumeration production run MUST thread the broad-corpus Σc⁻¹
    # via --sigma-inv. Without it main() would fall into the per-cell battery-Σc
    # eigh (one O(d³) eigh at d=3584 PER cell, >5 min/cell — the round-1 hang) AND
    # the result would be non-headline-eligible (plan §4c: battery-Σc is a
    # diagnostic fallback ONLY, never the headline). A diagnostic full run can opt
    # out explicitly with --allow-battery-sigma.
    if is_full and args.sigma_inv is None and not args.allow_battery_sigma:
        raise SystemExit(
            "headline run over the full store requires --sigma-inv <broad-corpus Σc⁻¹> "
            "(plan §4c broad-corpus headline-mandatory; the per-cell battery-Σc eigh is "
            "a diagnostic fallback only — pass --allow-battery-sigma to force it)."
        )

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

    # The SHARED cross-behavior TARGET-direction registry (Blocker 1): diffmeans
    # for bad_medical/em, the CANONICAL fact/marker source cells' r_plus for the
    # r_plus-only targets — built ONCE (downloads the 2 canonical cells once),
    # never per source. On the offline smoke (--slice, no --sigma-inv, no
    # rb_columns) we skip it so the cross-behavior grid omits the network-needing
    # targets rather than aliasing the source cell.
    target_registry: dict = {}
    if not (args.slice and args.sigma_inv is None):
        target_registry = build_target_direction_registry(
            layer=args.layer,
            rb_columns=rb_columns,
            download_cell=loader.download_cell,
            load_cell=loader.load_cell,
        )
        print(f"[predict] target-direction registry: {sorted(target_registry)}")

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
            target_registry=target_registry,
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

    hd_dir = OUT / "headline"
    hd_dir.mkdir(parents=True, exist_ok=True)

    # The r_B-source sensitivity deliverable (Concern 5): re-score the diffmeans
    # behaviors (bad_medical/em) under BOTH r_B sources + write the artifact. Runs
    # only on the production path (mixed/diffmeans + a real Σc) where re-running
    # under r_plus is meaningful — never on the offline smoke.
    sensitivity_rel = None
    if args.rb_source_sensitivity and rb_columns is not None:
        # Only the diffmeans behaviors (bad_medical=`bm_`, em=`ic_`) have a
        # diffmeans-vs-r_plus contrast; the #664 slug prefix encodes the behavior,
        # so filter by prefix to avoid re-scoring the whole store twice.
        sens_cells = [c for c in cells if c.startswith(("bm_", "ic_")) and "ic_edu" not in c]
        recs_dm = _score_cells_for_mode(
            sens_cells,
            layer=args.layer,
            sigma_inv_headline=sigma_inv_headline,
            rb_columns=rb_columns,
            target_registry=target_registry,
            r_b_source="diffmeans",
            loader=loader,
        )
        recs_rp = _score_cells_for_mode(
            sens_cells,
            layer=args.layer,
            sigma_inv_headline=sigma_inv_headline,
            rb_columns=rb_columns,
            target_registry=target_registry,
            r_b_source="r_plus",
            loader=loader,
        )
        sens = build_rb_source_sensitivity({"diffmeans": recs_dm, "r_plus": recs_rp})
        sens_path = hd_dir / "rb_source_sensitivity.json"
        sens_path.write_text(json.dumps(sens, indent=1))
        sensitivity_rel = f"headline/{sens_path.name}"
        print(f"[predict] rb-source sensitivity -> {sens_path}")

    # PRIMARY deliverable (plan §6.5): the per-behavior headline table — one block
    # per behavior arm + the cross-behavior aggregate (SECONDARY, mixed-r_B caveat)
    # + the corrected cross-behavior target matrix + the target-direction registry.
    headline = build_headline(
        records,
        sigma_meta=sigma_meta,
        r_b_source=args.r_b_source,
        target_registry=target_registry,
        sensitivity_artifact=sensitivity_rel,
    )
    hd_path = hd_dir / "predictor_headline.json"
    hd_path.write_text(json.dumps(headline, indent=1))
    print(f"[predict] headline -> {hd_path}")

    print(f"[phase=predictor] scored {len(cells)} cells OK")
    return 0


def build_headline(
    records: list[dict],
    *,
    sigma_meta: dict | None,
    r_b_source: str,
    target_registry: dict | None = None,
    sensitivity_artifact: str | None = None,
) -> dict:
    """The §6.5 PRIMARY headline table from the per-cell records.

    Per behavior arm: the mean own-behavior L̂/cosine/base-prior Spearman ρ over its
    cells + which r_B source each used. PLUS the cross-behavior aggregate (pooled
    own-behavior ρ across behaviors, flagged SECONDARY — it mixes diffmeans + r_plus
    sources, the §4b/§6 heterogeneity caveat) AND the cross-behavior TARGET matrix
    aggregate (the per_target L̂/cosine pooled over (source × target) off-diagonal
    cells — the corrected cross-behavior read, Blockers 1+2). Reproducibility
    metadata (git commit, timestamp, Σc-corpus kind, target-direction registry) is
    embedded per CLAUDE.md. The designed-null arm + the geometry-win verdict are
    computed by ``issue666_designed_null`` against the same per-cell JSONs; this
    table records the per-behavior PRIMARY reads. ``sensitivity_artifact`` is the
    relative path of the ``rb_source_sensitivity.json`` deliverable (Concern 5)
    when produced this run, else None.
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

    # Cross-behavior TARGET matrix aggregate (Blockers 1+2): pool the per_target
    # L̂/cosine ρ over the OFF-DIAGONAL (source behavior != target behavior) cells —
    # the corrected cross-behavior transfer read, with each target's direction from
    # the shared registry (NOT the source cell's r_plus) and the cosine carrying a
    # real cos(r_{B'}, r_B) term.
    xb_full: list[float] = []
    xb_cos: list[float] = []
    n_off_diag = 0
    for r in records:
        src_beh = r.get("behavior")
        for tb, row in (r.get("per_target") or {}).items():
            if tb == src_beh:
                continue  # off-diagonal only (the on-diagonal IS the own-behavior read)
            n_off_diag += 1
            xb_full.append(float(row.get("rho_full_Lhat", np.nan)))
            xb_cos.append(float(row.get("rho_cosine", np.nan)))
    cross_behavior_matrix = {
        "rho_full_Lhat_mean": float(np.nanmean(xb_full)) if xb_full else float("nan"),
        "rho_cosine_mean": float(np.nanmean(xb_cos)) if xb_cos else float("nan"),
        "n_off_diagonal_cells": n_off_diag,
        "flagged": "SECONDARY",
        "note": (
            "per_target off-diagonal (source!=target) L̂/cosine; target directions "
            "from the shared registry (diffmeans + canonical-cell r_plus), cosine "
            "carries cos(r_{B'}, r_B) with distinct source/target directions (§4d/§4e)"
        ),
    }

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    registry_meta = {
        tb: {"source": e.get("source"), "from_cell": e.get("from_cell")}
        for tb, e in (target_registry or {}).items()
    }
    return {
        "schema": "issue666_predictor_headline_v1",
        "per_behavior": per_behavior,
        "cross_behavior_aggregate": aggregate,
        "cross_behavior_matrix": cross_behavior_matrix,
        "target_direction_registry": registry_meta,
        "rb_source_sensitivity_artifact": sensitivity_artifact,
        "r_b_source_mode": r_b_source,
        "sigma_c": sigma_meta or {"sigma_c_corpus_kind": "battery-diagnostic"},
        "n_cells": len(records),
        "reproducibility": {
            "git_commit": commit,
            "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(),
            "primary_dv": "latent ds Spearman rho (per-behavior PRIMARY; aggregate SECONDARY)",
        },
    }


def build_rb_source_sensitivity(
    records_by_mode: dict[str, list[dict]],
    *,
    behaviors: tuple[str, ...] = ("bad_medical", "em"),
) -> dict:
    """The §6.5 r_B-source sensitivity deliverable (Concern 5).

    For each behavior #658 covers (bad_medical/em), the within-behavior own-behavior
    L̂ ρ under BOTH r_B sources — ``diffmeans`` (the production direction) and
    ``r_plus`` (the per-cell ŵ-shortcut sensitivity arm) — and their per-behavior ρ
    delta (diffmeans − r_plus). A small delta means the cross-behavior aggregate's
    interpretability does not hinge on the diffmeans choice; a large delta is the
    flag the §4b heterogeneity caveat warns about. ``records_by_mode`` maps each
    mode label to that mode's per-cell records. Returns the per-behavior table +
    reproducibility metadata.
    """
    import datetime as _dt
    import subprocess

    def _mean_full(records: list[dict], beh: str) -> float:
        vals = [float(r.get("rho_full_Lhat", np.nan)) for r in records if r.get("behavior") == beh]
        return float(np.nanmean(vals)) if vals else float("nan")

    per_behavior: dict = {}
    for beh in behaviors:
        dm = _mean_full(records_by_mode.get("diffmeans", []), beh)
        rp = _mean_full(records_by_mode.get("r_plus", []), beh)
        per_behavior[beh] = {
            "rho_full_Lhat_diffmeans": dm,
            "rho_full_Lhat_r_plus": rp,
            "delta_diffmeans_minus_r_plus": (
                float(dm - rp) if np.isfinite(dm) and np.isfinite(rp) else float("nan")
            ),
        }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    return {
        "schema": "issue666_rb_source_sensitivity_v1",
        "modes": sorted(records_by_mode),
        "behaviors": list(behaviors),
        "per_behavior": per_behavior,
        "reproducibility": {
            "git_commit": commit,
            "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(),
            "note": "within-behavior own-Lhat rho under diffmeans vs r_plus r_B sources (4b/6.5)",
        },
    }


def _score_cells_for_mode(
    cells: list[str],
    *,
    layer: int,
    sigma_inv_headline,
    rb_columns,
    target_registry: dict,
    r_b_source: str,
    loader,
) -> list[dict]:
    """Score the given cells under ONE r_B-source mode; return per-cell records.

    Shared by the sensitivity driver: identical to ``main``'s per-cell loop but
    pinned to one ``r_b_source`` mode and returning the records (no per-cell JSON
    write — the sensitivity artifact only needs the own-behavior ρ).
    """
    out: list[dict] = []
    for cell in cells:
        local_dir = loader.download_cell(cell)
        loaded = loader.load_cell(local_dir)
        lyr = min(layer, loaded["v_plus"].shape[1] - 1)
        if sigma_inv_headline is not None:
            Sigma_inv = sigma_inv_headline
        else:
            Sigma_inv, _ = _battery_sigma_inv(loaded, lyr)
        rec = predict_cell_grid(
            loaded,
            cell=cell,
            layer=lyr,
            Sigma_inv=Sigma_inv,
            rb_columns=rb_columns,
            r_b_source=r_b_source,
            target_registry=target_registry,
        )
        out.append(rec)
        del loaded
        gc.collect()
        with contextlib.suppress(OSError):
            os.remove(local_dir / "tensors.pt")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
