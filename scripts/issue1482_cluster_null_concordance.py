#!/usr/bin/env python
"""#1482 inline round: cluster-preserving permutation bands for the concordance table.

The paper scores 42 feature properties by concordance (c-index) with held-out
feature R^2 over ~7.3e9 pairs of 120,716 features, with NO confidence interval
anywhere (a 0.02 rule of thumb substitutes). That pair count treats features as
independent units; an SAE splits one concept across near-duplicate features that
share a decoder direction, co-fire, and land on nearly the same R^2, so a cluster
of siblings is closer to ONE observation than to many. This script ports the
#2163 cluster-preserving null (scripts/issue2163_cluster_null.py) onto the
concordance statistic:

- Clusters = connected components of the thresholded decoder-column cosine graph
  over the 120,716-feature analysis universe, per threshold tau. The full Gram
  (58 GB fp32) is never materialised: a chunked upper-triangle GEMM emits only
  the thresholded edge list (+ per-feature row max for the redundancy sanity
  gate).
- Statistic = the paper's own stepwise concordance, at THE ROUND EACH PROPERTY
  WAS SELECTED: conditioning cells rebuilt with the stepwise script's own
  crossed_strata over the recorded control set, observed values gated against
  the recorded scores in writeup_stepwise.meta.json to <= 1e-9.
- Null = permute the R^2 vector WITHIN conditioning cells. Feature grain:
  independent within-cell shuffle (the band the paper would have reported).
  Cluster grain: blocks = cluster INTERSECT cell move together, two schemes
  ported verbatim from #2163 (draw math identical, n_strata generalised from
  the module constant 10 to the round's cell count):
    * concat (primary): random block order concatenated within each cell,
      random order within blocks;
    * size_exchange (secondary, PALM-style): whole blocks swap only among
      same-size blocks within the same cell; blocks unique in (cell, size) are
      FROZEN (conservative); frozen fraction reported.
  A cluster spanning >1 cell is split into per-cell blocks (the split fraction
  is reported; independent movement of the sub-blocks loses their cross-cell
  coupling, so the cluster band is a LOWER bound in that respect).

Correctness gates (all hard-asserted except the Monte-Carlo one):
  1. Observed reproduction: recomputed c at each checked round matches the
     recorded stepwise score to <= 1e-9 (validates cells + vector alignment).
  2. Vectorised-vs-reference: the batched binary evaluator and the cached
     continuous evaluator match `concordance()` / `cd_untied()` exactly on
     probe draws (<= 1e-10).
  3. Decoder sanity: within-universe row-max cosine never exceeds the committed
     `redundancy_max_cos` covariate (whose max runs over all 131,071 other
     columns) beyond fp32 blocked-GEMM tolerance.
  4. Singleton reduction (Monte-Carlo, reported not asserted): forcing all
     blocks to singletons under the ported concat machinery must reproduce the
     independent feature-grain band within MC error.

Output: eval_results/issue_1482/concordance_cluster_null/
  cluster_labels.npz            per-tau component labels over the 120,716 universe
  gram_stats.json               edge counts, cluster censuses, decoder sanity
  nulls_round{k}.json           per-round checkpoint (written as each completes)
  cluster_null_concordance.json final assembly (bands, widening, clearance)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps must bind before any heavy import (#847)

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import issue1482_concordance_stepwise as ST  # noqa: E402  crossed_strata (reused via import)
import issue1482_concordance_writeup_figs as WF  # noqa: E402  battery() (reused via import)
from issue1482_concordance_fig import cd_untied, concordance  # noqa: E402  reference statistic
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("i1482.clusternull")

DICT_SIZE = 131_072
H_DIM = 3584
SEED = 1482
SEED_STREAM = 77  # child rngs keyed [1482, 77, scheme, round_k, round(tau*100)]
SCHEME_ID = {"feature": 0, "singleton": 1, "concat": 2, "size_exchange": 3}
MIN_CELL = 120  # cd_untied returns (0, 0) below 2 * MIN_POS = 120 members
COS_SANITY_TOL = 5e-3  # fp32 blocked-GEMM tolerance vs the committed covariate (#2163)
GATE_TOL = 1e-9
EDGE_CAP = 300_000_000  # fail-loud guard on edge-list size (3.6 GB at 12 B/edge)
META_PATH = PROJECT_ROOT / "figures/issue_1482/concordance/writeup_stepwise.meta.json"
SAE_WORK = "/mnt/eps-data/thomasjiralerspong/issue2163_probe/smoke_work_r2"
OUT_DIR = PROJECT_ROOT / "eval_results/issue_1482/concordance_cluster_null"

# The six paper-quoted properties = the winners of stepwise rounds 1..6 (asserted
# against the recorded meta at run time). Paper values (c - 0.5) for the report.
EXPECTED_SIX = {
    1: ("Mean activation (over all answers)", 0.27),
    2: ("Speaker: identity / disposition", 0.17),
    3: ("Logit footprint: suppressing", -0.15),
    4: ("Logit footprint: promoting", -0.14),
    5: ("Content type: topic", -0.13),
    6: ("Interpretable (autointerp)", 0.07),
}


# ── clustering: chunked upper-triangle GEMM -> edge list -> components ────────


def decoder_edges(
    active_ids: np.ndarray, min_tau: float, block: int = 4096
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(rows, cols, cos, rowmax) for within-universe decoder cosine >= min_tau.

    Never materialises the full Gram (58 GB at n=120,716): upper-triangle blocks
    of at most (block, n) fp32. rowmax is the per-feature max off-diagonal
    cosine WITHIN the universe (for the redundancy sanity gate).
    """
    import issue1482_sae as SAE

    t0 = time.time()
    sae = SAE.BatchTopKSAE.load(64, "cpu", Path(SAE_WORK) / "sae_cache", layer=19)
    w_dec = sae.w_dec.detach().cpu().numpy().astype(np.float32)
    assert w_dec.shape == (H_DIM, DICT_SIZE), w_dec.shape
    d = np.ascontiguousarray(w_dec[:, active_ids])
    del w_dec, sae
    d /= np.linalg.norm(d, axis=0, keepdims=True)
    n = d.shape[1]
    logger.info("[gram] decoder loaded + normalised (%d cols) in %.0fs", n, time.time() - t0)

    rowmax = np.full(n, -1.0, dtype=np.float32)
    rr: list[np.ndarray] = []
    cc: list[np.ndarray] = []
    vv: list[np.ndarray] = []
    n_edges = 0
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        a = d[:, i0:i1].T @ d[:, i0:]  # (bi, n - i0) fp32
        bi = i1 - i0
        a[np.arange(bi), np.arange(bi)] = -1.0  # kill the diagonal
        np.maximum(rowmax[i0:i1], a.max(axis=1), out=rowmax[i0:i1])
        np.maximum(rowmax[i0:], a.max(axis=0), out=rowmax[i0:])
        r, c = np.nonzero(a >= min_tau)
        keep = c > r  # strict upper triangle in GLOBAL coordinates (col j maps to i0 + j)
        r, c = r[keep], c[keep]
        rr.append((r + i0).astype(np.int32))
        cc.append((c + i0).astype(np.int32))
        vv.append(a[r, c])
        n_edges += len(r)
        assert n_edges <= EDGE_CAP, f"edge list exploded past {EDGE_CAP:,} at tau={min_tau}"
        logger.info("[gram] block %d..%d done (%.0fs, %s edges)", i0, i1, time.time() - t0, n_edges)
    del d
    return np.concatenate(rr), np.concatenate(cc), np.concatenate(vv), rowmax.astype(np.float64)


def components_at(
    rows: np.ndarray, cols: np.ndarray, cos: np.ndarray, tau: float, n: int
) -> tuple[int, np.ndarray, int]:
    """(n_clusters, labels, n_edges) — connected components of cos >= tau."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    keep = cos >= tau
    r, c = rows[keep], cols[keep]
    adj = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    return int(n_comp), labels.astype(np.int64), int(len(r))


def size_stats(labels: np.ndarray, n_clusters: int) -> dict:
    """Cluster count + size distribution (the effective-sample-size read). Port of #2163."""
    sizes = np.bincount(labels, minlength=n_clusters)
    return {
        "n_clusters": int(n_clusters),
        "n_features": int(labels.size),
        "n_singletons": int((sizes == 1).sum()),
        "largest_cluster": int(sizes.max()),
        "n_features_in_nonsingleton": int(sizes[sizes > 1].sum()),
        "frac_features_in_nonsingleton": float(sizes[sizes > 1].sum() / labels.size),
    }


# ── permutation builders (ported from issue2163_cluster_null.py, n_strata free) ──


class ConcatPerms:
    """Random-block-order concatenation within cells (primary). #2163 port.

    `labels` are BLOCK labels (cluster INTERSECT cell), `feat_cell` the per-feature
    cell id, `n_cells` the cell count. Draw math identical to #2163 _ConcatPerms.
    """

    def __init__(self, labels: np.ndarray, feat_cell: np.ndarray, n_cells: int):
        n = labels.size
        self.n = n
        self.strata: list[tuple[np.ndarray, np.ndarray, int]] = []
        for s in range(n_cells):
            feats = np.flatnonzero(feat_cell == s)
            if feats.size == 0:
                continue
            labs = labels[feats]
            order = np.lexsort((feats, labs))  # canonical block order
            slots = feats[order]
            _, block_pos = np.unique(labs[order], return_inverse=True)
            m = int(block_pos.max()) + 1
            self.strata.append((slots, block_pos.astype(np.int64), m))

    @property
    def frozen_feature_count(self) -> int:
        """Features in cells holding a single block (the only concat freeze case)."""
        return sum(slots.size for slots, _, m in self.strata if m == 1)

    def build(self, rng: np.random.Generator, nb: int) -> np.ndarray:
        """(nb, n) index arrays: y[perm] realizes one draw per row."""
        perm = np.tile(np.arange(self.n), (nb, 1))
        for slots, block_pos, m in self.strata:
            ranks = np.argsort(np.argsort(rng.random((nb, m)), axis=1), axis=1)
            key = ranks[:, block_pos] + rng.random((nb, slots.size))
            order = np.argsort(key, axis=1)
            perm[:, slots] = slots[order]
        return perm


class SizeExchangePerms:
    """PALM-style whole-block exchange among same-size blocks within cells. #2163 port."""

    def __init__(self, labels: np.ndarray, feat_cell: np.ndarray, n_blocks: int):
        self.n = labels.size
        sizes = np.bincount(labels, minlength=n_blocks)
        order = np.argsort(labels, kind="stable")
        bounds = np.searchsorted(labels[order], np.arange(n_blocks + 1))
        members = [order[bounds[b] : bounds[b + 1]] for b in range(n_blocks)]
        block_cell = np.full(n_blocks, -1, dtype=np.int64)
        block_cell[labels] = feat_cell  # every block lives in exactly one cell
        self.groups: list[np.ndarray] = []
        frozen = 0
        for s in np.unique(block_cell):
            in_s = np.flatnonzero(block_cell == s)
            for size in np.unique(sizes[in_s]):
                bs = in_s[sizes[in_s] == size]
                mat = np.stack([members[b] for b in bs])  # (m, size)
                if len(bs) == 1:
                    frozen += int(size)  # unique (cell, size): block never moves
                self.groups.append(mat)
        self.frozen_feature_count = frozen

    def build(self, rng: np.random.Generator, nb: int) -> np.ndarray:
        perm = np.tile(np.arange(self.n), (nb, 1))
        for mat in self.groups:
            m, size = mat.shape
            pi = np.argsort(rng.random((nb, m)), axis=1)
            gathered = mat[pi]
            ord3 = np.argsort(rng.random((nb, m, size)), axis=2)
            shuffled = np.take_along_axis(gathered, ord3, axis=2)
            perm[:, mat.reshape(-1)] = shuffled.reshape(nb, m * size)
        return perm


class FeaturePerms:
    """Independent within-cell shuffle (the feature-grain reference band).

    Deliberately a DIFFERENT code path from ConcatPerms so the singleton-reduction
    gate validates the ported block machinery against a straightforward one.
    """

    def __init__(self, cells: list[np.ndarray], n: int):
        self.n = n
        self.cells = cells
        self.frozen_feature_count = 0

    def build(self, rng: np.random.Generator, nb: int) -> np.ndarray:
        perm = np.tile(np.arange(self.n), (nb, 1))
        for members in self.cells:
            order = np.argsort(rng.random((nb, members.size)), axis=1)
            perm[:, members] = members[order]
        return perm


# ── draw evaluation ───────────────────────────────────────────────────────────


class RoundEval:
    """Per-round concordance evaluator over permuted y, exact vs the reference.

    Binary properties are evaluated BATCHED across draws via the within-cell rank
    identity rank(y_perm)[i] = rank_pre[perm[i]] (permutations never cross cells);
    continuous properties call the reference cd_untied per draw per cell.
    """

    def __init__(self, cells: list[np.ndarray], y: np.ndarray, props: dict[str, np.ndarray]):
        self.cells = cells
        self.y = y
        n = y.size
        self.r_pre = np.empty(n, dtype=np.float64)
        for members in cells:
            self.r_pre[members] = rankdata(y[members])
        self.binary: dict[str, list[tuple[np.ndarray, float, float, float]]] = {}
        self.continuous: dict[str, np.ndarray] = {}
        for name, v in props.items():
            if len(np.unique(v)) == 2:
                per_cell = []
                hi = np.unique(v)[1]
                for members in cells:
                    m = members.size
                    if m < MIN_CELL:
                        continue
                    pos = members[v[members] == hi]
                    k = pos.size
                    if k == 0 or k == m:
                        continue
                    per_cell.append((pos, float(k), float(m), float(k * (m - k))))
                self.binary[name] = per_cell
            else:
                self.continuous[name] = v

    def eval_binary(self, perm: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
        """c per draw for each named binary property, over a (nb, n) perm chunk."""
        rp = self.r_pre[perm]  # (nb, n): within-cell rank of the value landing at each slot
        out = {}
        for name in names:
            num = den = 0.0
            for pos, k, m, d in self.binary[name]:
                s = rp[:, pos].sum(axis=1)
                auc = (s - k * (k + 1) / 2) / d
                num = num + (2 * auc - 1) * d
                den += d
            out[name] = 0.5 * (num / den + 1) if den > 0 else np.full(perm.shape[0], np.nan)
        return out

    def eval_continuous(self, perm: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
        """c per draw for each named continuous property (reference cd_untied path)."""
        out = {name: np.empty(perm.shape[0]) for name in names}
        for di in range(perm.shape[0]):
            yp = self.y[perm[di]]
            for name in names:
                x = self.continuous[name]
                num = den = 0.0
                for members in self.cells:
                    a, d = cd_untied(x[members], yp[members])
                    num += a
                    den += d
                out[name][di] = 0.5 * (num / den + 1) if den > 0 else np.nan
        return out

    def reference_c(self, perm_row: np.ndarray, name: str, v: np.ndarray) -> float:
        """The untouched reference statistic on one draw (equivalence probes)."""
        return concordance(v, self.y[perm_row], self.cells)


def band_summary(c_draws: np.ndarray, c_obs: float) -> dict:
    """Two-sided band on |c - 0.5| + clearance verdict for one property/config."""
    dev = np.abs(c_draws - 0.5)
    band = float(np.quantile(dev, 0.975))
    obs_dev = abs(c_obs - 0.5)
    return {
        "n_draws": int(c_draws.size),
        "band_p97_5_abs_dev": band,
        "null_c_p2_5": float(np.quantile(c_draws, 0.025)),
        "null_c_p97_5": float(np.quantile(c_draws, 0.975)),
        "null_c_sd": float(c_draws.std()),
        "clears": bool(obs_dev > band),
        "margin_over_band": float(obs_dev / band) if band > 0 else float("inf"),
        "perm_p_two_sided": float((1 + int((dev >= obs_dev - 1e-15).sum())) / (c_draws.size + 1)),
    }


# ── per-round null runner ─────────────────────────────────────────────────────


def run_round(
    k: int,
    meta_round: dict,
    b: dict,
    labels_by_tau: dict[str, np.ndarray],
    args,
) -> dict:
    """All configs x properties for stepwise round k; returns the checkpoint payload."""
    y, n, vecs = b["y"], b["n"], b["vecs"]
    controls = meta_round["controls"]
    cells, bins = ST.crossed_strata([vecs[c] for c in controls], n)
    feat_cell = np.empty(n, dtype=np.int64)
    for ci, members in enumerate(cells):
        feat_cell[members] = ci
    recorded = {s["name"]: s["c"] for s in meta_round["scores"]}

    # which properties to band at this round
    if k == 0:
        names = [s["name"] for s in meta_round["scores"]]
    else:
        names = [EXPECTED_SIX[k][0]]
    ev = RoundEval(cells, y, {nm: vecs[nm] for nm in names})
    bin_names = [nm for nm in names if nm in ev.binary]
    cont_names = [nm for nm in names if nm in ev.continuous]
    headline = {EXPECTED_SIX[r][0] for r in EXPECTED_SIX}

    # gate 1: observed reproduction against the recorded stepwise scores
    obs: dict[str, float] = {}
    max_diff = 0.0
    for nm in names:
        c = concordance(vecs[nm], y, cells)
        obs[nm] = float(c)
        max_diff = max(max_diff, abs(c - recorded[nm]))
    assert max_diff <= GATE_TOL, f"round {k}: observed reproduction gate FAILED ({max_diff:.2e})"
    logger.info(
        "[r%d] observed reproduction gate PASS (max diff %.2e, %d props)", k, max_diff, len(names)
    )

    configs: list[tuple[str, str | None]] = [("feature", None), ("singleton", None)]
    for tau in args.tau_list:
        configs += [("concat", f"{tau:.2f}"), ("size_exchange", f"{tau:.2f}")]
    # secondary continuous table (round 0, non-headline): restrict configs + draws
    cont_table_configs = {("feature", None)} | {("concat", f"{t:.2f}") for t in args.tau_list}

    results: dict[str, dict] = {nm: {} for nm in names}
    diag: dict[str, dict] = {}
    for scheme, tkey in configs:
        tag = scheme if tkey is None else f"{scheme}@{tkey}"
        t0 = time.time()
        if scheme == "feature":
            builder = FeaturePerms(cells, n)
        elif scheme == "singleton":
            builder = ConcatPerms(np.arange(n, dtype=np.int64), feat_cell, len(cells))
        else:
            lab = labels_by_tau[tkey]
            blab = np.unique(lab * len(cells) + feat_cell, return_inverse=True)[1]
            n_blocks = int(blab.max()) + 1
            span = np.unique(np.stack([lab, feat_cell]), axis=1).shape[1] - np.unique(lab).size
            if scheme == "concat":
                builder = ConcatPerms(blab, feat_cell, len(cells))
            else:
                builder = SizeExchangePerms(blab, feat_cell, n_blocks)
            diag[tag] = {
                "n_blocks": n_blocks,
                "n_cluster_cell_splits": int(span),
                "frozen_feature_frac": builder.frozen_feature_count / n,
            }
        rng = np.random.default_rng(
            [
                SEED,
                SEED_STREAM,
                SCHEME_ID[scheme],
                k,
                0 if tkey is None else round(float(tkey) * 100),
            ]
        )
        c_bin = {nm: [] for nm in bin_names}
        c_cont = {nm: [] for nm in cont_names}
        head_cont = [nm for nm in cont_names if nm in headline]
        tbl_cont = [nm for nm in cont_names if nm not in headline]
        tbl_active = (scheme, tkey) in cont_table_configs
        for d0 in range(0, args.n_draws, args.chunk):
            nb = min(args.chunk, args.n_draws - d0)
            perm = builder.build(rng, nb)
            for nm, arr in ev.eval_binary(perm, bin_names).items():
                c_bin[nm].append(arr)
            if head_cont:  # headline continuous: full draw count
                for nm, arr in ev.eval_continuous(perm, head_cont).items():
                    c_cont[nm].append(arr)
            if tbl_cont and tbl_active and d0 < args.cont_draws:
                nb_t = min(nb, args.cont_draws - d0)  # secondary table: capped draws
                for nm, arr in ev.eval_continuous(perm[:nb_t], tbl_cont).items():
                    c_cont[nm].append(arr)
            # gate 2: vectorised-vs-reference equivalence, first chunk's first draw
            if d0 == 0:
                probe_names = bin_names[:3] + head_cont[:1]
                if tbl_cont and tbl_active:
                    probe_names += tbl_cont[:1]
                for nm in probe_names:
                    got = c_bin[nm][0][0] if c_bin.get(nm) else c_cont[nm][0][0]
                    ref = ev.reference_c(perm[0], nm, vecs[nm])
                    assert abs(got - ref) <= 1e-10, (tag, nm, got, ref)
        for nm in names:
            arrs = c_bin.get(nm) or c_cont.get(nm)
            if not arrs:
                continue
            c_draws = np.concatenate(arrs)
            results[nm][tag] = band_summary(c_draws, obs[nm])
        logger.info("[r%d] %s done in %.0fs", k, tag, time.time() - t0)

    payload = {
        "round": k,
        "controls": controls,
        "bins": bins,
        "n_cells": len(cells),
        "cell_sizes": {
            "median": int(np.median([len(c) for c in cells])),
            "n_below_min_cell": int(sum(len(c) < MIN_CELL for c in cells)),
        },
        "observed": obs,
        "observed_reproduction_max_diff": max_diff,
        "cluster_block_diagnostics": diag,
        "bands": results,
    }
    return payload


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="cluster-preserving bands for #1482 concordance")
    ap.add_argument("--stage", choices=["gram", "nulls", "assemble", "all"], default="all")
    ap.add_argument("--taus", default="0.30,0.35,0.40")
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--cont-draws", type=int, default=150, help="draws for non-headline continuous")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--rounds", default="1,2,3,4,5,6,0", help="stepwise rounds, priority order")
    ap.add_argument("--force", action="store_true", help="recompute existing checkpoints")
    args = ap.parse_args()
    args.tau_list = [float(t) for t in args.taus.split(",") if t.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = json.loads(META_PATH.read_text())
    rounds_meta = {r["round"]: r for r in meta["rounds"]}
    for k, (nm, _) in EXPECTED_SIX.items():
        assert rounds_meta[k]["winner"] == nm, (k, rounds_meta[k]["winner"], nm)

    t0 = time.time()
    b = WF.battery()
    active_ids = np.flatnonzero(b["ok"])
    n = b["n"]
    assert active_ids.size == n == meta["n_rows"], (active_ids.size, n, meta["n_rows"])
    logger.info("[setup] battery loaded: n=%d (%.0fs)", n, time.time() - t0)

    labels_path = OUT_DIR / "cluster_labels.npz"
    stats_path = OUT_DIR / "gram_stats.json"
    if args.stage in ("gram", "all") and (args.force or not labels_path.exists()):
        min_tau = min(args.tau_list)
        rows, cols, cos, rowmax = decoder_edges(active_ids, min_tau)
        red_name = "Nearest-neighbour cosine (SAE redundancy)"
        assert red_name in b["vecs"], f"redundancy covariate missing from battery: {red_name}"
        red = np.asarray(b["vecs"][red_name], dtype=np.float64)
        excess = rowmax - red
        sanity = {
            "n_violations": int((excess > COS_SANITY_TOL).sum()),
            "max_excess": float(excess.max()),
            "tolerance": COS_SANITY_TOL,
        }
        assert sanity["n_violations"] == 0, f"decoder sanity gate FAILED: {sanity}"
        logger.info("[gram] decoder sanity gate PASS (max excess %.2e)", sanity["max_excess"])
        store: dict[str, np.ndarray] = {"active_feat_ids": active_ids}
        census = {}
        for tau in args.tau_list:
            key = f"{tau:.2f}"
            n_comp, labels, n_edges = components_at(rows, cols, cos, tau, n)
            store[f"labels_tau_{key}"] = labels.astype(np.int32)
            census[key] = {**size_stats(labels, n_comp), "n_edges": n_edges}
            logger.info(
                "[gram] tau=%s: %d edges, %d clusters, largest %d",
                key,
                n_edges,
                n_comp,
                census[key]["largest_cluster"],
            )
        np.savez_compressed(labels_path, **store)
        stats_path.write_text(
            json.dumps(
                {
                    "meta": {
                        **as_metadata_dict(git_provenance()),
                        "numpy": np.__version__,
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    },
                    "min_tau_emitted": min_tau,
                    "n_edges_at_min_tau": int(len(rows)),
                    "decoder_sanity": sanity,
                    "rowmax_quantiles": {
                        q: float(np.quantile(rowmax, float(q)))
                        for q in ("0.5", "0.9", "0.99", "1.0")
                    },
                    "census": census,
                },
                indent=2,
                sort_keys=True,
            )
        )
        del rows, cols, cos, rowmax
    if args.stage == "gram":
        return 0

    with np.load(labels_path) as z:
        assert np.array_equal(z["active_feat_ids"], active_ids), "universe drift vs cluster labels"
        labels_by_tau = {
            f"{tau:.2f}": np.asarray(z[f"labels_tau_{tau:.2f}"], dtype=np.int64)
            for tau in args.tau_list
        }

    if args.stage in ("nulls", "all"):
        for k in [int(x) for x in args.rounds.split(",") if x.strip()]:
            ck = OUT_DIR / f"nulls_round{k}.json"
            if ck.exists() and not args.force:
                prev = json.loads(ck.read_text())
                if prev.get("n_draws_config") == args.n_draws:
                    logger.info("[r%d] checkpoint exists at n_draws=%d — skip", k, args.n_draws)
                    continue
            payload = run_round(k, rounds_meta[k], b, labels_by_tau, args)
            payload["n_draws_config"] = args.n_draws
            payload["cont_draws_config"] = args.cont_draws
            ck.write_text(json.dumps(payload, indent=2, sort_keys=True))
            logger.info("[r%d] checkpoint written", k)

    # ── assemble ─────────────────────────────────────────────────────────────
    per_round = {}
    for k in range(0, 7):
        ck = OUT_DIR / f"nulls_round{k}.json"
        if ck.exists():
            per_round[str(k)] = json.loads(ck.read_text())
    headline = {}
    for k, (nm, paper_val) in EXPECTED_SIX.items():
        rk = per_round.get(str(k))
        if rk is None or nm not in rk["bands"]:
            continue
        bands = rk["bands"][nm]
        feat_band = bands["feature"]["band_p97_5_abs_dev"]
        entry = {
            "selected_round": k,
            "paper_value_c_minus_half": paper_val,
            "observed_c": rk["observed"][nm],
            "observed_abs_dev": abs(rk["observed"][nm] - 0.5),
            "round0_c": per_round.get("0", {}).get("observed", {}).get(nm),
            "feature_band": bands["feature"],
            "singleton_gate": {
                "band": bands["singleton"]["band_p97_5_abs_dev"],
                "rel_diff_vs_feature": abs(bands["singleton"]["band_p97_5_abs_dev"] - feat_band)
                / feat_band,
            },
            "cluster_bands": {},
        }
        for tag, s in bands.items():
            if "@" not in tag:
                continue
            entry["cluster_bands"][tag] = {
                **s,
                "widening_vs_feature": s["band_p97_5_abs_dev"] / feat_band,
            }
        headline[nm] = entry

    final = {
        "meta": {
            **as_metadata_dict(git_provenance()),
            "numpy": np.__version__,
            "seed_stream": [SEED, SEED_STREAM],
            "rng_convention": (
                "independent child generators default_rng([1482, 77, scheme, round_k, "
                "round(tau*100)]); scheme 0=feature, 1=singleton-gate, 2=concat, 3=size-exchange"
            ),
            "n_draws": args.n_draws,
            "cont_draws_secondary": args.cont_draws,
            "taus": args.tau_list,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
        "convention": (
            "statistic = the paper's stepwise concordance at each property's selection round "
            "(cells via issue1482_concordance_stepwise.crossed_strata over the recorded control "
            "set; observed values gated to <= 1e-9 vs writeup_stepwise.meta.json). Null = "
            "permute R^2 within cells; cluster grain moves cluster-INTERSECT-cell blocks "
            "(clusters = connected components of decoder-column cosine >= tau over the "
            "120,716-feature universe). Schemes ported from scripts/issue2163_cluster_null.py."
        ),
        "gram_stats": json.loads(stats_path.read_text()) if stats_path.exists() else None,
        "headline_six": headline,
        "per_round": per_round,
    }
    out = OUT_DIR / "cluster_null_concordance.json"
    out.write_text(json.dumps(final, indent=2, sort_keys=True))
    logger.info("[assemble] wrote %s", out)
    print(
        json.dumps(
            {
                nm: {
                    "obs_dev": round(e["observed_abs_dev"], 4),
                    "feature_band": round(e["feature_band"]["band_p97_5_abs_dev"], 5),
                    "cluster": {
                        t: {
                            "band": round(s["band_p97_5_abs_dev"], 5),
                            "widen": round(s["widening_vs_feature"], 2),
                            "clears": s["clears"],
                        }
                        for t, s in e["cluster_bands"].items()
                    },
                }
                for nm, e in headline.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
