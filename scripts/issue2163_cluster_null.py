#!/usr/bin/env python
"""issue 2163 round 3 — cluster-preserving permutation band for the within-active null (S-1).

The committed within-active band (``population_partials.json`` -> ``train_active``,
band 0.030915) is a 1,000-draw permutation stratified within deciles of the
``lasttoken_count`` match rank, which treats the 13,282 active features as exchangeable
units. Near-duplicate SAE features co-fire, so the DV and the covariates are both
cluster-correlated and the feature-level band can be anticonservative (too narrow).
This script builds near-duplicate clusters from decoder-column cosine similarity
(connected components of the thresholded within-active cosine graph, per threshold)
and re-draws the null permuting at CLUSTER grain, so the effective number of
independent units drops to roughly the cluster count.

Two cluster-preserving schemes, both stratified within deciles of the CLUSTER-mean
match rank (which reduces EXACTLY to the driver's feature-grain deciles when every
cluster is a singleton):

- ``concat`` (primary): within each stratum, clusters are laid out in a canonical
  block order; each draw reassigns the DV values by concatenating the clusters'
  value-blocks in a random cluster order (random order within blocks). Every cluster
  moves; unequal sizes are allowed; correlated DV blocks land on covariate-coherent
  slot blocks, reproducing the Mantel-style variance inflation the feature-level
  null ignores.
- ``size_exchange`` (secondary, PALM-style whole-block exchange): value-blocks swap
  only among same-size clusters within the same stratum. Cleaner block semantics,
  but a cluster unique in (stratum, size) is FROZEN at its observed slots, which
  pushes observed signal into the null maxima (conservative direction); the frozen
  feature fraction is reported per threshold.

Correctness gates: (1) the recomputed within-active observed partials must match the
committed ``train_active`` values to <= 1e-9 (and n == 13,282) before any band is
trusted; (2) forcing all clusters to singletons under the ``concat`` scheme must
reproduce the committed feature-level band within Monte-Carlo error (reported).
Sanity: the within-active off-diagonal max cosine per feature must not exceed the
committed ``redundancy_max_cos`` covariate (whose max runs over all 131,071 other
columns) beyond float tolerance.

Scope boundary (stated, per the dispatch note): WITHIN-ACTIVE population only. The
never-active population is out of scope because (a) its pairwise Gram would be
115,168^2 fp32 = 53 GB, over the 50 GB off-VM routing gate, and (b) its matching
covariate is one tie (tie fraction 1.000: ``lasttoken_count`` identically 0), so its
band already degenerates to a plain permutation and cluster-preserving
stratification would not change what it tests.

Output: ``eval_results/issue_2163/cluster_null.json`` (+ per-threshold cluster labels
in ``cluster_null_labels.npz`` and a band-vs-threshold figure).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps must bind before any heavy import (#847)

import numpy as np  # noqa: E402

from issue2163_ctxread import (  # noqa: E402  (driver convention, reused via import)
    DICT_SIZE,
    MATCH_COV,
    SEED,
    _load_sae_dec,
    _load_selection,
    _partial_row,
    _rank,
    _residualize,
    logger,
)
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
COV_PATH_IN_REPO = "issue2163_ctxread/inputs/fullwidth_covariates_v2.npz"
HF_REVISION_PIN = "0d76405c5704798cda8e116d666c1a916e61a15a"  # the body's pinned data revision
GATE_TOL = 1e-9
N_STRATA = 10
COS_SANITY_TOL = 5e-3  # fp32 blocked-routine tolerance vs the committed covariate
SEED_STREAM = 53  # fresh stream id; child rngs are keyed [SEED, 53, scheme, round(tau*100)]


# ── clustering ────────────────────────────────────────────────────────────────


def _components_at(gram: np.ndarray, tau: float) -> tuple[int, np.ndarray, int]:
    """(n_clusters, labels, n_edges) — connected components of cos >= tau (off-diagonal)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = gram.shape[0]
    r, c = np.nonzero(gram >= tau)
    keep = r < c
    r, c = r[keep], c[keep]
    adj = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    return int(n_comp), labels.astype(np.int64), int(len(r))


def _cluster_strata(m_rank: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Decile strata of the cluster-mean match rank (equal CLUSTER counts per stratum).

    Reduces exactly to the driver's feature-grain strata when every cluster is a
    singleton: cluster mean == the feature's match rank, the min-member-index
    tie-break == the driver's index tie-break, and equal cluster counts == equal
    feature counts.
    """
    n = len(m_rank)
    cnt = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    cmean = np.bincount(labels, weights=m_rank, minlength=n_clusters) / cnt
    cmin = np.full(n_clusters, n, dtype=np.int64)
    np.minimum.at(cmin, labels, np.arange(n))
    order = np.lexsort((cmin, cmean))
    cstrat = np.empty(n_clusters, dtype=np.int64)
    cstrat[order] = np.minimum((np.arange(n_clusters) * N_STRATA) // n_clusters, N_STRATA - 1)
    return cstrat


def _size_stats(labels: np.ndarray, n_clusters: int) -> dict:
    """Cluster count + size distribution summary (the effective-sample-size read)."""
    sizes = np.bincount(labels, minlength=n_clusters)
    return {
        "n_clusters": int(n_clusters),
        "n_features": int(labels.size),
        "n_singletons": int((sizes == 1).sum()),
        "n_size_2": int((sizes == 2).sum()),
        "n_size_3_5": int(((sizes >= 3) & (sizes <= 5)).sum()),
        "n_size_6_10": int(((sizes >= 6) & (sizes <= 10)).sum()),
        "n_size_gt_10": int((sizes > 10).sum()),
        "largest_cluster": int(sizes.max()),
        "n_features_in_nonsingleton": int(sizes[sizes > 1].sum()),
        "frac_features_in_nonsingleton": float(sizes[sizes > 1].sum() / labels.size),
    }


# ── cluster-preserving permutation builders ───────────────────────────────────


class _ConcatPerms:
    """Random-cluster-order concatenation within cluster-grain strata (primary)."""

    def __init__(self, labels: np.ndarray, cstrat: np.ndarray):
        n = labels.size
        self.n = n
        self.strata: list[tuple[np.ndarray, np.ndarray, int]] = []
        feat_strat = cstrat[labels]
        for s in range(N_STRATA):
            feats = np.flatnonzero(feat_strat == s)
            if feats.size == 0:
                continue
            labs = labels[feats]
            # canonical block order: clusters sorted by id, members by feature index
            order = np.lexsort((feats, labs))
            slots = feats[order]  # slot positions, cluster-blocked
            _, cluster_pos = np.unique(labs[order], return_inverse=True)
            m = int(cluster_pos.max()) + 1
            self.strata.append((slots, cluster_pos.astype(np.int64), m))

    @property
    def frozen_feature_count(self) -> int:
        """Features in strata holding a single cluster (the only concat freeze case)."""
        return sum(slots.size for slots, _, m in self.strata if m == 1)

    def build(self, rng: np.random.Generator, nb: int) -> np.ndarray:
        """(nb, n) index arrays: dv_r[perm] realizes one draw per row."""
        perm = np.tile(np.arange(self.n), (nb, 1))
        for slots, cluster_pos, m in self.strata:
            ranks = np.argsort(np.argsort(rng.random((nb, m)), axis=1), axis=1)
            key = ranks[:, cluster_pos] + rng.random((nb, slots.size))
            order = np.argsort(key, axis=1)
            perm[:, slots] = slots[order]
        return perm


class _SizeExchangePerms:
    """PALM-style whole-block exchange among same-size clusters within strata."""

    def __init__(self, labels: np.ndarray, cstrat: np.ndarray, n_clusters: int):
        self.n = labels.size
        sizes = np.bincount(labels, minlength=n_clusters)
        members: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n_clusters
        order = np.argsort(labels, kind="stable")
        bounds = np.searchsorted(labels[order], np.arange(n_clusters + 1))
        for c in range(n_clusters):
            members[c] = order[bounds[c] : bounds[c + 1]]
        self.groups: list[np.ndarray] = []  # each (m, size) member matrix
        frozen = 0
        for s in range(N_STRATA):
            in_s = np.flatnonzero(cstrat == s)
            for size in np.unique(sizes[in_s]):
                cs = in_s[sizes[in_s] == size]
                mat = np.stack([members[c] for c in cs])  # (m, size)
                if len(cs) == 1:
                    frozen += int(size)  # unique (stratum, size): block never moves
                self.groups.append(mat)
        self.frozen_feature_count = frozen

    def build(self, rng: np.random.Generator, nb: int) -> np.ndarray:
        """(nb, n) index arrays: same-size blocks swap, random order within blocks."""
        perm = np.tile(np.arange(self.n), (nb, 1))
        for mat in self.groups:
            m, size = mat.shape
            pi = np.argsort(rng.random((nb, m)), axis=1)
            gathered = mat[pi]  # (nb, m, size)
            ord3 = np.argsort(rng.random((nb, m, size)), axis=2)
            shuffled = np.take_along_axis(gathered, ord3, axis=2)
            perm[:, mat.reshape(-1)] = shuffled.reshape(nb, m * size)
        return perm


# ── band evaluation (mirrors the driver's _stratperm_band draw math) ──────────


def _eval_band(
    dv_r: np.ndarray,
    resid_c: np.ndarray,
    norm_c: np.ndarray,
    m_rank: np.ndarray,
    match_idx: list[int],
    builder,
    n_draws: int,
    rng: np.random.Generator,
    tag: str,
    chunk: int = 100,
) -> float:
    """p97.5 of the per-draw max |partial| under a cluster-preserving permutation."""
    mr = m_rank - m_rank.mean()
    mrmr = max(float(mr @ mr), 1e-30)
    draw_max = np.empty(n_draws, dtype=np.float64)
    t0 = time.time()
    for d0 in range(0, n_draws, chunk):
        nb = min(chunk, n_draws - d0)
        perm = builder.build(rng, nb)
        dvp = dv_r[perm]
        rd = dvp - dvp.mean(axis=1, keepdims=True)
        beta = (rd @ mr) / mrmr
        rd = rd - beta[:, None] * mr[None, :]
        nd = np.linalg.norm(rd, axis=1)
        num = rd @ resid_c.T
        with np.errstate(divide="ignore", invalid="ignore"):
            part = num / (nd[:, None] * norm_c[None, :])
        part[:, np.asarray(match_idx)] = 0.0
        part[:, norm_c < 1e-9] = 0.0
        draw_max[d0 : d0 + nb] = np.abs(part).max(axis=1)
        logger.info("[cluster-null:%s] draws %d/%d %.0fs", tag, d0 + nb, n_draws, time.time() - t0)
    return float(np.quantile(draw_max, 0.975))


# ── main ──────────────────────────────────────────────────────────────────────


def _stage_inputs(args) -> tuple[Path, Path]:
    """Covariate panel + census symlink into the (round-2) staging layout."""
    stage = Path(args.stage_dir)
    cov_dir = stage / "cov"
    cov_dir.mkdir(parents=True, exist_ok=True)
    cov_file = cov_dir / "fullwidth_covariates_v2.npz"
    if not cov_file.exists():
        hub.stage_hub_file(
            HF_DATA_REPO,
            COV_PATH_IN_REPO,
            cov_file,
            repo_type="dataset",
            revision=HF_REVISION_PIN,
        )
    assembled = stage / "assembled"
    assembled.mkdir(parents=True, exist_ok=True)
    census_link = assembled / "census.npz"
    if not census_link.exists():
        census_link.symlink_to(Path(args.results_dir) / "census.npz")
    return cov_dir, stage


def _render_figure(fig_dir: Path, taus: list[float], per_tau: dict, ref: dict) -> dict:
    """Band vs threshold, both schemes, against the committed band + observed max."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, savefig_paper

    c_concat, c_sizeex = paper_palette(2)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    concat = [per_tau[f"{t:.2f}"]["band_concat"] for t in taus]
    sizeex = [per_tau[f"{t:.2f}"]["band_size_exchange"] for t in taus]
    ax.plot(taus, concat, "o-", color=c_concat, label="cluster band (concat)")
    ax.plot(taus, sizeex, "s--", color=c_sizeex, label="cluster band (size-exchange)")
    ax.axhline(ref["band_feature_level"], color="0.4", ls=":", label="committed feature band")
    ax.axhline(ref["max_abs_partial"], color="0.1", ls="-", lw=1, label="observed max |partial|")
    ax.set_yscale("log")
    ax.set_xlabel("decoder-cosine cluster threshold")
    ax.set_ylabel("p97.5 of per-draw max |partial|")
    ax.set_title("Within-active null band under cluster-preserving permutation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths = savefig_paper(fig, "cluster_null_band", dir=fig_dir)
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def main() -> int:
    """Cluster-preserving within-active bands; assert the reproduction gate first."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(PROJECT_ROOT / "eval_results" / "issue_2163"))
    ap.add_argument(
        "--stage-dir",
        default="/mnt/eps-data/thomasjiralerspong/issue2163_r2",
        help="off-root staging dir (round-2 layout: cov/ + assembled/census.npz symlink)",
    )
    ap.add_argument(
        "--sae-work",
        default="/mnt/eps-data/thomasjiralerspong/issue2163_probe/smoke_work_r2",
        help="dir whose sae_cache/ holds the pinned SAE checkpoint (re-fetched if absent)",
    )
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95")
    ap.add_argument("--out", default=None, help="default: <results-dir>/cluster_null.json")
    ap.add_argument(
        "--labels-out", default=None, help="default: <results-dir>/cluster_null_labels.npz"
    )
    ap.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures" / "issue_2163"))
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    resd = Path(args.results_dir)
    taus = [float(t) for t in args.thresholds.split(",") if t.strip()]
    _stage_inputs(args)

    sel_args = SimpleNamespace(
        local_covariates=str(Path(args.stage_dir) / "cov"), work=args.stage_dir
    )
    cols, columns, dropped = _load_selection(sel_args)
    census = np.load(resd / "census.npz")
    assert np.array_equal(np.asarray(census["feat_ids"], dtype=np.int64), np.arange(DICT_SIZE)), (
        "census feature-id join broken"
    )
    lad_w = np.load(resd / "read_ladder__W.npz")
    u_w = np.asarray(lad_w["u"], dtype=np.float64)
    dv = np.where(u_w > 0, np.log10(np.clip(u_w, 1e-300, None)), np.nan)  # driver logU_W
    ltc = np.asarray(census["lasttoken_count"], dtype=np.float64)

    cov_mat = np.stack([cols[c] for c in columns])
    mask = np.isfinite(cov_mat).all(axis=0) & np.isfinite(dv) & (ltc > 0)
    active_ids = np.flatnonzero(mask)
    n = int(active_ids.size)
    logger.info("[cluster-null] within-active n=%d", n)

    # ── committed reference + observed-partials reproduction gate ────────────
    committed = json.loads((resd / "population_partials.json").read_text())
    ta = committed["populations"]["train_active"]
    assert committed["selection_columns"] == columns, "selection-column drift vs committed"

    match_idx = [columns.index(MATCH_COV)]
    dv_r = _rank(dv[mask])
    cov_r = np.stack([_rank(cov_mat[k][mask]) for k in range(cov_mat.shape[0])])
    m_design = cov_r[match_idx].T
    resid_c = np.stack([_residualize(cov_r[k], m_design) for k in range(cov_mat.shape[0])])
    norm_c = np.linalg.norm(resid_c, axis=1)
    obs, degen = _partial_row(dv_r, cov_r, resid_c, norm_c, match_idx, m_design)
    diffs = {
        c: abs(float(v) - ta["observed_partials"][c])
        for c, v, g in zip(columns, obs, degen)
        if not g
    }
    gate = {
        "n_committed": ta["n"],
        "n_recomputed": n,
        "max_abs_diff_observed_partials": max(diffs.values()),
        "tolerance": GATE_TOL,
        "committed_band_feature_level": ta["band_p97_5_of_max"],
        "committed_max_abs_partial": ta["max_abs_partial"],
        "committed_argmax_column": ta["argmax_column"],
        "committed_n_columns_outside_band": ta["n_columns_outside_band"],
    }
    gate["pass"] = bool(n == ta["n"] and gate["max_abs_diff_observed_partials"] <= GATE_TOL)
    assert gate["pass"], f"within-active reproduction gate FAILED: {gate}"
    logger.info("[cluster-null] reproduction gate PASS (max diff %.2e)", max(diffs.values()))

    ref = {
        "band_feature_level": float(ta["band_p97_5_of_max"]),
        "max_abs_partial": float(ta["max_abs_partial"]),
    }
    m_rank = cov_r[match_idx[0]]
    obs_abs = np.abs(obs)

    # ── decoder-column cosine Gram over the active features ──────────────────
    t0 = time.time()
    w_dec, _ = _load_sae_dec(SimpleNamespace(work=args.sae_work))
    d_act = np.ascontiguousarray(w_dec[:, active_ids], dtype=np.float32)
    del w_dec
    d_act /= np.linalg.norm(d_act, axis=0, keepdims=True)
    gram = d_act.T @ d_act  # (n, n) fp32, ~706 MB at n=13,282
    del d_act
    logger.info("[cluster-null] decoder Gram built in %.0fs", time.time() - t0)

    np.fill_diagonal(gram, -1.0)
    rowmax = gram.max(axis=1).astype(np.float64)
    np.fill_diagonal(gram, 1.0)
    red = np.asarray(cols["redundancy_max_cos"], dtype=np.float64)[mask]
    excess = rowmax - red
    similarity = {
        "definition": "cosine between unit-normalized decoder columns, within-active pairs",
        "rowmax_quantiles": {
            q: float(np.quantile(rowmax, float(q))) for q in ("0.5", "0.9", "0.99", "1.0")
        },
        "sanity_vs_redundancy_max_cos": {
            "n_violations": int((excess > COS_SANITY_TOL).sum()),
            "max_excess": float(excess.max()),
            "tolerance": COS_SANITY_TOL,
            "note": "committed covariate max runs over ALL other decoder columns, so the "
            "within-active row max must not exceed it beyond float tolerance",
        },
        "frac_features_with_within_active_cos_ge": {
            f"{t:.2f}": float((rowmax >= t).mean()) for t in taus
        },
    }
    assert similarity["sanity_vs_redundancy_max_cos"]["n_violations"] == 0, similarity

    # ── singleton-reduction validation (scheme machinery vs committed band) ──
    singleton_labels = np.arange(n, dtype=np.int64)
    cstrat_single = _cluster_strata(m_rank, singleton_labels, n)
    builder = _ConcatPerms(singleton_labels, cstrat_single)
    rng = np.random.default_rng([SEED, SEED_STREAM, 0, 0])
    band_red = _eval_band(
        dv_r, resid_c, norm_c, m_rank, match_idx, builder, args.n_draws, rng, "singleton-reduction"
    )
    reduction = {
        "band_singleton_reduction": band_red,
        "committed_band_feature_level": ref["band_feature_level"],
        "rel_diff": abs(band_red - ref["band_feature_level"]) / ref["band_feature_level"],
        "note": "all-singleton clusters reduce the concat scheme to the committed "
        "feature-level stratified permutation; agreement within Monte-Carlo error "
        "validates the draw machinery (fresh rng stream, so not byte-equal)",
    }
    logger.info(
        "[cluster-null] singleton reduction band %.6f vs committed %.6f (rel diff %.1f%%)",
        band_red,
        ref["band_feature_level"],
        100 * reduction["rel_diff"],
    )

    # ── per-threshold cluster bands ───────────────────────────────────────────
    per_tau: dict[str, dict] = {}
    labels_store: dict[str, np.ndarray] = {"active_feat_ids": active_ids}
    for tau in taus:
        key = f"{tau:.2f}"
        n_comp, labels, n_edges = _components_at(gram, tau)
        labels_store[f"labels_tau_{key}"] = labels.astype(np.int32)
        cstrat = _cluster_strata(m_rank, labels, n_comp)
        concat = _ConcatPerms(labels, cstrat)
        sizeex = _SizeExchangePerms(labels, cstrat, n_comp)
        band_c = _eval_band(
            dv_r,
            resid_c,
            norm_c,
            m_rank,
            match_idx,
            concat,
            args.n_draws,
            np.random.default_rng([SEED, SEED_STREAM, 1, round(tau * 100)]),
            f"concat@{key}",
        )
        band_s = _eval_band(
            dv_r,
            resid_c,
            norm_c,
            m_rank,
            match_idx,
            sizeex,
            args.n_draws,
            np.random.default_rng([SEED, SEED_STREAM, 2, round(tau * 100)]),
            f"sizeex@{key}",
        )
        per_tau[key] = {
            **_size_stats(labels, n_comp),
            "n_edges": n_edges,
            "band_concat": band_c,
            "band_size_exchange": band_s,
            "band_widening_concat": band_c / ref["band_feature_level"],
            "band_widening_size_exchange": band_s / ref["band_feature_level"],
            "n_columns_outside_band_concat": int((obs_abs > band_c).sum()),
            "n_columns_outside_band_size_exchange": int((obs_abs > band_s).sum()),
            "margin_max_over_band_concat": ref["max_abs_partial"] / band_c,
            "margin_max_over_band_size_exchange": ref["max_abs_partial"] / band_s,
            "frozen_feature_frac_concat": concat.frozen_feature_count / n,
            "frozen_feature_frac_size_exchange": sizeex.frozen_feature_count / n,
        }
        logger.info(
            "[cluster-null] tau=%s clusters=%d largest=%d band_concat=%.6f band_sizeex=%.6f",
            key,
            n_comp,
            per_tau[key]["largest_cluster"],
            band_c,
            band_s,
        )

    # ── outputs ───────────────────────────────────────────────────────────────
    out = Path(args.out) if args.out else resd / "cluster_null.json"
    labels_out = Path(args.labels_out) if args.labels_out else resd / "cluster_null_labels.npz"
    np.savez_compressed(labels_out, **labels_store)

    figure_paths: dict = {}
    if not args.no_fig:
        figure_paths = _render_figure(Path(args.fig_dir), taus, per_tau, ref)

    payload = {
        "meta": {
            **as_metadata_dict(git_provenance()),
            "numpy": np.__version__,
            "seed_stream": [SEED, SEED_STREAM],
            "rng_convention": "independent child generators default_rng([2163, 53, scheme, "
            "round(tau*100)]); scheme 0=singleton-reduction, 1=concat, 2=size-exchange",
            "n_draws": args.n_draws,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
        "convention": "rank/residualize/partial + stratified-null draw math imported/mirrored "
        "from scripts/issue2163_ctxread.py phase_partials; DV logU_W; match covariate "
        "lasttoken_count; strata = deciles of CLUSTER-mean match rank (equal cluster counts); "
        "clusters = connected components of within-active decoder-column cosine >= tau",
        "population": {
            "definition": "complete-case AND lasttoken_count > 0 (within-active)",
            "n": n,
            "match_tie_fraction": float(
                np.max(np.unique(cov_mat[columns.index(MATCH_COV)][mask], return_counts=True)[1])
                / n
            ),
        },
        "scope_note": {
            "never_active_excluded": True,
            "reasons": [
                "pairwise Gram at n=115,168 is 13.3e9 entries fp32 = 53 GB, over the "
                "VM_ANALYSIS_FOOTPRINT_GB_MAX=50 off-VM routing gate",
                "matching is inert there by construction (tie fraction 1.000: "
                "lasttoken_count identically 0), so its band already degenerates to a "
                "plain permutation and cluster-preserving stratification changes nothing",
            ],
        },
        "selection_columns": columns,
        "dropped_columns": dropped,
        "reproduction_gate": gate,
        "observed_partials": {c: float(v) for c, v in zip(columns, obs)},
        "degenerate_partial": [c for c, g in zip(columns, degen) if g],
        "decoder_similarity": similarity,
        "singleton_reduction": reduction,
        "per_threshold": per_tau,
        "figure": figure_paths,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("[cluster-null] wrote %s", out)
    print(
        json.dumps(
            {
                "gate": gate["pass"],
                "singleton_reduction_rel_diff": reduction["rel_diff"],
                "per_threshold": {
                    k: {
                        "n_clusters": v["n_clusters"],
                        "largest": v["largest_cluster"],
                        "band_concat": v["band_concat"],
                        "band_size_exchange": v["band_size_exchange"],
                        "margin_concat": v["margin_max_over_band_concat"],
                    }
                    for k, v in per_tau.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
