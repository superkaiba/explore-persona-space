#!/usr/bin/env python
"""Issue #1482: collinearity-collapsed EXACT Shapley decomposition, within activity strata.

Three steps, all full width (131,072 dictionary; the joined R^2 universe is
114,980):

  1. COLLAPSE near-duplicate covariates. Measured collinearity makes
     all-others-partialling unsound (activity <-> n_active_holdout rho +0.998,
     VIF ~250: the "partial effect of activity controlling for
     n_active_holdout" estimates the 0.2% of variance where two measurements of
     one thing disagree). Hierarchical clustering on 1 - |rho_spearman| with
     COMPLETE linkage, cut at rho 0.9. Complete linkage is load-bearing: it is
     what makes "every pair inside a cluster exceeds rho 0.9" actually true;
     average linkage would admit a rho 0.7 pair and silently break the
     guarantee. One representative per cluster chosen by INTERPRETABILITY, never
     by predictive strength (choosing the strongest member is selection on the
     outcome).

  2. EXACT Shapley / LMG on the block representatives. Payoff v(S) = R^2 of OLS
     of rank(target) on the rank-transformed members of S. Computed in closed
     form off ONE precomputed correlation matrix — R^2(S) = r_S^T (R_SS)^-1 r_S
     — so all 2^K subsets are tiny solves, never 2^K regression fits. Two hard
     correctness gates: EFFICIENCY (shares sum to the full-model R^2) and NULL
     PLAYER (an injected permuted-target column scores ~0).

  3. WITHIN ACTIVITY STRATA. Pooling is unsound here for a second, orthogonal
     reason: 0 of 19 predictors is flat across activity deciles, so a pooled
     decomposition averages a +0.58 and a -0.18 into a number describing no
     stratum. Shapley fixes CORRELATION; it does not fix EFFECT MODIFICATION.
     The decile x block heatmap is the primary deliverable; the pooled version
     is computed and labelled as the misleading contrast.

MULTI-CLASS AXES ARE ONE PLAYER EACH. A judged axis enters as one-hot dummies
with a reference level dropped; making each DUMMY a player would make every
value depend on which level happened to be dropped, since the dummies are
structurally dependent. So all of an axis's dummies enter and leave the
coalition together — invariant to the reference choice, and semantically "does
content type matter" rather than "does the topic dummy matter against an
arbitrary baseline".

UNCERTAINTY IS SPLIT-HALF, NOT BOOTSTRAP. Sampling error is negligible at this
n (SE of a Spearman ~ 1/sqrt(n-3) = 0.003 pooled, 0.009 per decile), so
bootstrap CIs would be ~+-0.02 on shares spanning tenths — everything reads
"significant" and the interval carries no information. The uncertainty that
matters is the provisional target, the clustering choices, and effect
modification, and a row bootstrap holding clusters fixed is blind to all three.
Instead the ENTIRE pipeline runs independently on two disjoint halves, with
clusters RE-DERIVED from each half's own correlation matrix, which additionally
catches clustering instability a fixed-cluster bootstrap structurally cannot.

TARGET: provisional #1738 SAE->SAE until task #7's full-width dense->SAE arrays
land; the R^2 source is a single `--r2-npy` swap. pod-1482 is live and will
produce them. Run-length covariates (`mean_run_length`, `persistence_p`,
`template_token_frac`, `act_var_across_tokens`) are a swappable input on the
same contract: picked up automatically when
`eval_results/issue_1482/run_length/` lands, never waited on. `mean_run_length`
and `persistence_p` are the SAME quantity (E[R] = 1/(1-p)) and are forced into
one block so the decomposition cannot double-count one construct.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.task_workflow import repo_root  # noqa: E402

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_predictor_battery as PB  # noqa: E402

DICT_SIZE = 131_072
N_DECILES = 10
SEED = 1482
CUT_PRIMARY = 0.90
CUT_SENSITIVITY = (0.85, 0.90, 0.95)
RARE_CLASS_FLOOR = 30
GATE_TOL = 1e-8
NULL_PLAYER_TOL = 5e-3

COV_NPZ = "eval_results/issue_1482/predictor_battery/fullwidth_covariates.npz"
DISC_NPZ = "eval_results/issue_1482/predictor_battery/fullwidth_discrete_covariates.npz"
MATRIX = "eval_results/issue_1482/predictor_battery/fullwidth_matrix.npz"
RUN_LENGTH = "eval_results/issue_1482/run_length/run_length_perfeature.npz"
R2_PROVISIONAL = "eval_results/issue_1738/sae_twoway/perfeature/sae_context_r2.npy"
OUT_DIR = "eval_results/issue_1482/predictor_battery"
FIG_DIR = "figures/issue_1482/predictor_battery"

# Continuous candidates entering the clustering. dec_norm is degenerate
# (unit-norm) and promoting_class is excluded per the block-set decision below.
CONTINUOUS_CANDIDATES = (
    "activity",
    "firing_freq_per_token",
    "consistency",
    "mean_act_uncond",
    "mean_act_cond",
    "act_var_across_answers",
    "side_ratio",
    "n_active_holdout",
    "scaffold_frac",
    "enc_dec_cos",
    "enc_norm",
    "massive_dim_mass",
    "logit_footprint_concentration",
    "redundancy_max_cos",
    "write_norm",
    "footprint_var",
    "footprint_skew",
    "footprint_kurt",
    "proj_var",
    "dense_latent_flag",
)
# Run-length covariates, joined when the capture lands. mean_run_length and
# persistence_p are FORCED into one block (same quantity).
RUNLEN_CANDIDATES = ("mean_run_length", "template_token_frac", "act_var_across_tokens")
FORCED_TOGETHER = (("mean_run_length", "persistence_p"),)

# Representative choice: LOWER rank = more interpretable / more canonical. This
# is fixed BEFORE any R^2 is read, so representative choice cannot be selection
# on the outcome.
INTERPRETABILITY_RANK = {
    "activity": 0,
    "firing_freq_per_token": 1,
    "mean_act_uncond": 2,
    "mean_act_cond": 3,
    "consistency": 4,
    "side_ratio": 5,
    "mean_run_length": 6,
    "template_token_frac": 7,
    "act_var_across_answers": 8,
    "act_var_across_tokens": 9,
    "proj_var": 10,
    "scaffold_frac": 11,
    "write_norm": 12,
    "enc_norm": 13,
    "enc_dec_cos": 14,
    "redundancy_max_cos": 15,
    "logit_footprint_concentration": 16,
    "massive_dim_mass": 17,
    "footprint_kurt": 18,
    "footprint_skew": 19,
    "footprint_var": 20,
    "n_active_holdout": 21,
    "dense_latent_flag": 22,
}
INTERPRETABILITY_RATIONALE = (
    "rank order fixed before any R^2 is read. Direct behavioural rates first "
    "(how often / how much a feature fires), then persistence, then variance, then "
    "decoder-geometry summaries, then derived counts and thresholded flags last. A "
    "thresholded flag (dense_latent_flag) or a redundant count (n_active_holdout) is never "
    "preferred over the continuous quantity it derives from."
)

# One player per judged AXIS (all its dummies enter/leave together).
AXIS_BLOCKS = ("abstraction", "content_type", "speaker_property", "interpretable", "side_class")
EXCLUDED_BLOCKS = {
    "functional_role": (
        "RETIRED — Cohen's kappa 0.310, below the 0.6 usability bar. Excluded by decision, "
        "not oversight."
    ),
    "gurnee_promoting_class": (
        "EXCLUDED — shown to be a slice off a continuum (all five output-ness measures are "
        "unimodal) that adds ~nothing over its continuous parent: rho(class, R^2 | kurtosis) "
        "+0.038 against rho(kurtosis, R^2 | class) +0.199. Excluded by decision, not oversight."
    ),
}
# ONE uniform policy across axes: `unresolved` is carried as an EXPLICIT LEVEL.
# Dropping unresolved rows would give every axis a DIFFERENT row set, which
# breaks the shared correlation matrix the whole decomposition is built on and
# makes the axes non-comparable.
UNRESOLVED_POLICY = (
    "carried as an explicit level, uniformly across every axis. Dropping unresolved rows "
    "would give each axis a different row set, breaking the single shared correlation matrix "
    "the decomposition requires and making the axes non-comparable."
)


def _log(msg: str) -> None:
    print(f"[shapley] {msg}", flush=True)


# ── inputs ───────────────────────────────────────────────────────────────────


def load_inputs(r2_npy: Path) -> dict:
    with np.load(PROJECT_ROOT / COV_NPZ) as z:
        cov = {k: np.asarray(z[k], dtype=np.float64) for k in z.files if k != "feat_ids"}
    with np.load(PROJECT_ROOT / DISC_NPZ) as z:
        for k in ("side_class", "dense_latent_flag"):
            cov[k] = np.asarray(z[k], dtype=np.float64)
    with np.load(PROJECT_ROOT / MATRIX, allow_pickle=True) as z:
        universe = np.asarray(z["feat_ids"], dtype=np.int64)
        labels = {
            k[len("label__") :]: np.asarray(z[k]).astype(str)
            for k in z.files
            if k.startswith("label__")
        }
    # labels are on the 114,980 universe; scatter to full width
    lab_full = {}
    for ax, v in labels.items():
        col = np.full(DICT_SIZE, "unlabeled", dtype=object)
        col[universe] = v
        lab_full[ax] = col

    pending, runlen_present = list(RUNLEN_CANDIDATES), False
    rl = PROJECT_ROOT / RUN_LENGTH
    if rl.exists():
        runlen_present = True
        with np.load(rl) as z:
            ids = np.asarray(z["feat_ids"], dtype=np.int64)
            pending = []
            for name in (*RUNLEN_CANDIDATES, "persistence_p"):
                if name in z.files:
                    full = np.full(DICT_SIZE, np.nan)
                    full[ids] = np.asarray(z[name], dtype=np.float64)
                    cov[name] = full
                else:
                    pending.append(name)
        _log(f"run-length joined ({len(ids)} rows); still pending: {pending}")
    else:
        _log(f"run-length ABSENT ({RUN_LENGTH}) — {len(pending)} covariates pending")

    r2 = np.asarray(np.load(r2_npy), dtype=np.float64)
    if r2.shape != (DICT_SIZE,):
        raise AssertionError(f"R^2 must be ({DICT_SIZE},), got {r2.shape}")
    return {
        "cov": cov,
        "labels": lab_full,
        "r2": r2,
        "universe": universe,
        "pending_runlen": pending,
        "runlen_present": runlen_present,
    }


def candidate_names(cov: dict) -> list[str]:
    names = [c for c in CONTINUOUS_CANDIDATES if c in cov]
    names += [c for c in (*RUNLEN_CANDIDATES, "persistence_p") if c in cov]
    return names


# ── step 1: complete-linkage clustering on 1 - |rho| ────────────────────────


def cluster_covariates(x: np.ndarray, names: list[str], cut: float) -> tuple[list[list[str]], dict]:
    """COMPLETE-linkage clusters at |rho| >= `cut`, plus the within-cluster min |rho|.

    Complete linkage is required, not stylistic: it merges on the WORST pair, so
    every pair inside a returned cluster is guaranteed above the cut. Average
    linkage would admit a weakly-correlated member and silently void that
    guarantee.
    """
    from scipy.cluster.hierarchy import complete, fcluster
    from scipy.spatial.distance import squareform

    ranks = np.column_stack([PB._rank(x[:, j]) for j in range(x.shape[1])])
    zc = (ranks - ranks.mean(0)) / ranks.std(0)
    rho = (zc.T @ zc) / len(zc)
    dist = 1.0 - np.abs(rho)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2, 0.0, None)
    link = complete(squareform(dist, checks=False))
    lab = fcluster(link, t=1.0 - cut, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for j, c in enumerate(lab):
        clusters.setdefault(int(c), []).append(names[j])
    # honour forced-together pairs (same construct by identity, e.g. E[R]=1/(1-p))
    idx = {n: j for j, n in enumerate(names)}
    for a, b in FORCED_TOGETHER:
        if a in idx and b in idx and lab[idx[a]] != lab[idx[b]]:
            keep, drop = int(lab[idx[a]]), int(lab[idx[b]])
            clusters[keep] += clusters.pop(drop)
            lab[lab == drop] = keep
            _log(f"forced {a} + {b} into one block (identical construct)")

    out, info = [], []
    for _, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        out.append(sorted(members, key=lambda m: INTERPRETABILITY_RANK.get(m, 99)))
        if len(members) > 1:
            pj = [idx[m] for m in members]
            sub = np.abs(rho[np.ix_(pj, pj)])
            np.fill_diagonal(sub, np.inf)
            info.append({"members": members, "within_min_abs_rho": float(sub.min())})
        else:
            info.append({"members": members, "within_min_abs_rho": None})
    return out, {"clusters": info, "cut": cut, "rho_matrix_names": names}


def pick_representatives(clusters: list[list[str]]) -> list[dict]:
    reps = []
    for members in clusters:
        rep = min(members, key=lambda m: INTERPRETABILITY_RANK.get(m, 99))
        reps.append(
            {
                "representative": rep,
                "members": members,
                "n_members": len(members),
                "chosen_because": (
                    f"lowest interpretability rank ({INTERPRETABILITY_RANK.get(rep, 99)}) among "
                    f"{members} — chosen by interpretability, NOT predictive strength"
                ),
            }
        )
    return reps


# ── step 2: exact Shapley off one Gram ──────────────────────────────────────


def enumerate_subsets(blocks: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-mask column lists in CSR form. Built ONCE per block set and reused
    across the pooled run and all 10 deciles (they share the same blocks)."""
    k = len(blocks)
    n = 1 << k
    widths = np.zeros(n, dtype=np.int32)
    for b in range(k):
        m = np.arange(n)
        widths[(m & (1 << b)) > 0] += len(blocks[b])
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(widths, out=off[1:])
    flat = np.empty(int(off[-1]), dtype=np.int32)
    for mask in range(1, n):
        pos = off[mask]
        m = mask
        while m:
            b = (m & -m).bit_length() - 1
            cb = blocks[b]
            flat[pos : pos + len(cb)] = cb
            pos += len(cb)
            m &= m - 1
    return flat, off, widths


def subset_r2_table(
    corr: np.ndarray,
    ry: np.ndarray,
    enum: tuple[np.ndarray, np.ndarray, np.ndarray],
    chunk: int = 150_000,
) -> np.ndarray:
    """v(S) = r_S^T (R_SS)^-1 r_S for every subset, as BATCHED solves off one Gram.

    Never fits a regression. Subsets are grouped by column count so each group is
    one stacked (m, c, c) `linalg.solve`; a naive per-subset loop measured 38.8
    us/subset against ~8 us batched at this shape.

    Batched `solve` raises on the WHOLE batch if ANY slice is singular, so a
    failing chunk falls back to per-slice pinv rather than losing the chunk.
    """
    flat, off, widths = enum
    v = np.zeros(len(widths))
    for c in np.unique(widths):
        if c == 0:
            continue
        masks = np.flatnonzero(widths == c)
        for s in range(0, len(masks), chunk):
            mk = masks[s : s + chunk]
            idx = np.stack([flat[off[m] : off[m] + c] for m in mk]).astype(np.int64)
            subs = corr[idx[:, :, None], idx[:, None, :]]
            rs = ry[idx]
            try:
                sol = np.linalg.solve(subs, rs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                sol = np.empty_like(rs)
                for i in range(len(mk)):
                    try:
                        sol[i] = np.linalg.solve(subs[i], rs[i])
                    except np.linalg.LinAlgError:
                        sol[i] = np.linalg.pinv(subs[i]) @ rs[i]
            v[mk] = (rs * sol).sum(1)
    return np.clip(v, 0.0, None)


def shapley_from_table(v: np.ndarray, k: int) -> np.ndarray:
    """Exact Shapley values from the full 2^K payoff table."""
    from math import factorial

    fact = [float(factorial(i)) for i in range(k + 1)]
    w = np.array([fact[s] * fact[k - s - 1] / fact[k] for s in range(k)])
    masks = np.arange(1 << k)
    popc = np.array([bin(m).count("1") for m in masks])
    phi = np.zeros(k)
    for i in range(k):
        bit = 1 << i
        without = masks[(masks & bit) == 0]
        phi[i] = float(np.sum(w[popc[without]] * (v[without | bit] - v[without])))
    return phi


def block_brackets(v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The two endpoints Shapley sits between: alone (over-counts) and last (under-counts)."""
    full = (1 << k) - 1
    alone = np.array([v[1 << i] for i in range(k)])
    last = np.array([v[full] - v[full & ~(1 << i)] for i in range(k)])
    return alone, last


def decompose(
    x: np.ndarray,
    y: np.ndarray,
    blocks: list[list[int]],
    names: list[str],
    rng,
    enum=None,
) -> dict:
    """Rank-transform, build the shared Gram, run exact Shapley + both gates."""
    cols = np.column_stack([PB._rank(x[:, j]) for j in range(x.shape[1])])
    yr = PB._rank(y)
    # null player: a permuted target, appended as its own single-column block
    null_col = rng.permutation(yr)
    allc = np.column_stack([cols, null_col])
    sd = allc.std(0)
    zc = (allc - allc.mean(0)) / np.where(sd > 0, sd, 1.0)
    yz = (yr - yr.mean()) / yr.std()
    corr = (zc.T @ zc) / len(zc)
    ry = (zc.T @ yz) / len(zc)
    # A level absent from this stratum gives a CONSTANT column. Make it an
    # exact zero-contribution player (identity row, zero target correlation)
    # rather than a singular slice: a constant predictor adds nothing to R^2,
    # and this keeps every subset solve batched.
    const = np.flatnonzero(sd <= 0)
    if len(const):
        corr[const, :] = 0.0
        corr[:, const] = 0.0
        corr[const, const] = 1.0
        ry[const] = 0.0

    blocks_null = [*blocks, [allc.shape[1] - 1]]
    k = len(blocks_null)
    if enum is None:
        enum = enumerate_subsets(blocks_null)
    v = subset_r2_table(corr, ry, enum)
    phi = shapley_from_table(v, k)
    alone, last = block_brackets(v, k)

    r2_full = float(v[(1 << k) - 1])
    eff = abs(float(phi.sum()) - r2_full)
    null_phi = float(phi[-1])
    _log(
        f"  K={k - 1} blocks (+null): R^2_full={r2_full:.5f} efficiency|delta|={eff:.2e} "
        f"null_player_phi={null_phi:+.2e}"
    )
    if not eff < GATE_TOL:
        raise AssertionError(f"EFFICIENCY GATE FAILED: sum(phi)-R^2_full = {eff:.3e}")
    if not abs(null_phi) < NULL_PLAYER_TOL:
        raise AssertionError(f"NULL-PLAYER GATE FAILED: phi_null = {null_phi:.3e}")

    tot = float(phi[:-1].sum())
    return {
        "blocks": names,
        "r2_full": r2_full,
        "shapley": {n: float(phi[i]) for i, n in enumerate(names)},
        "shapley_share": {n: float(phi[i] / tot) if tot > 0 else 0.0 for i, n in enumerate(names)},
        "marginal_alone": {n: float(alone[i]) for i, n in enumerate(names)},
        "unique_when_last": {n: float(last[i]) for i, n in enumerate(names)},
        "gates": {
            "efficiency_abs_delta": eff,
            "efficiency_pass": True,
            "null_player_phi": null_phi,
            "null_player_pass": True,
        },
    }


# ── block matrix assembly ───────────────────────────────────────────────────


def build_design(
    inp: dict,
    rows: np.ndarray,
    reps: list[str],
    axes: list[str],
    global_levels: dict | None = None,
) -> tuple[np.ndarray, list[list[int]], list[str], dict]:
    """Columns = continuous reps + one-hot dummies per axis; blocks group the dummies."""
    cov, labels = inp["cov"], inp["labels"]
    cols, blocks, names = [], [], []
    for r in reps:
        blocks.append([len(cols)])
        cols.append(cov[r][rows])
        names.append(r)
    axis_levels = {}
    for ax in axes:
        raw = (
            labels[ax][rows] if ax in labels else np.asarray(cov[ax][rows]).astype(int).astype(str)
        )
        counts_here = {lv: int((raw == lv).sum()) for lv in sorted(set(raw.tolist()))}
        if global_levels is not None:
            levels = list(global_levels[ax]["levels"])
            drop = global_levels[ax]["reference"]
        else:
            levels = sorted(counts_here)
            drop = max(levels, key=lambda lv: counts_here[lv])  # reference = most common
        axis_levels[ax] = {lv: counts_here.get(lv, 0) for lv in levels}
        idxs = []
        for lv in levels:
            if lv == drop:
                continue
            idxs.append(len(cols))
            cols.append((raw == lv).astype(np.float64))
        if idxs:
            blocks.append(idxs)
            names.append(f"axis:{ax}")
        axis_levels[ax]["__reference__"] = drop
    return np.column_stack(cols), blocks, names, axis_levels


def run_pipeline(inp: dict, rows: np.ndarray, cut: float, rng, tag: str) -> dict:
    """Cluster -> representatives -> exact Shapley, pooled and per activity decile."""
    cand = candidate_names(inp["cov"])
    ok = np.ones(len(rows), dtype=bool)
    for c in cand:
        ok &= np.isfinite(inp["cov"][c][rows])
    ok &= np.isfinite(inp["r2"][rows])
    rows = rows[ok]
    _log(f"[{tag}] {len(rows)} rows finite across {len(cand)} candidates + R^2")

    x = np.column_stack([inp["cov"][c][rows] for c in cand])
    clusters, cinfo = cluster_covariates(x, cand, cut)
    reps_doc = pick_representatives(clusters)
    reps = [r["representative"] for r in reps_doc]
    _log(f"[{tag}] {len(clusters)} clusters at rho {cut} -> reps {reps}")

    axes = list(AXIS_BLOCKS)
    _, _, _, lv0 = build_design(inp, rows, reps, axes)
    global_levels = {
        ax: {
            "levels": [k for k in lv0[ax] if k != "__reference__"],
            "reference": lv0[ax]["__reference__"],
        }
        for ax in axes
    }
    xd, blocks, names, axis_levels = build_design(inp, rows, reps, axes, global_levels)
    y = inp["r2"][rows]

    enum = enumerate_subsets([*blocks, [xd.shape[1]]])
    pooled = decompose(xd, y, blocks, names, rng, enum)

    act = inp["cov"]["activity"][rows]
    edges = np.quantile(act, np.linspace(0, 1, N_DECILES + 1))
    dec = np.searchsorted(edges[1:-1], act, side="right")
    per_decile, rare_flags = [], []
    for d in range(N_DECILES):
        m = dec == d
        xdd, blocks_d, names_d, lv_d = build_design(inp, rows[m], reps, axes, global_levels)
        res = decompose(xdd, y[m], blocks_d, names_d, rng, enum)
        res["decile"] = d + 1
        res["n"] = int(m.sum())
        res["activity_range"] = [float(edges[d]), float(edges[d + 1])]
        res["internal_activity_ratio"] = float(edges[d + 1] / max(edges[d], 1e-300))
        thin = {
            ax: [lv for lv, n in lv_d[ax].items() if lv != "__reference__" and n < RARE_CLASS_FLOOR]
            for ax in axes
        }
        res["rare_classes_below_floor"] = {k: v for k, v in thin.items() if v}
        res["class_counts"] = lv_d
        rare_flags.append({f"axis:{k}" for k in res["rare_classes_below_floor"]})
        per_decile.append(res)
        _log(f"[{tag}]   decile {d + 1}: n={res['n']} R^2={res['r2_full']:.4f}")

    return {
        "tag": tag,
        "n_rows": int(len(rows)),
        "cut": cut,
        "clusters": cinfo["clusters"],
        "representatives": reps_doc,
        "block_names": names,
        "axis_levels_pooled": axis_levels,
        "pooled": pooled,
        "per_decile": per_decile,
        "rare_flagged_blocks_per_decile": [sorted(s) for s in rare_flags],
    }


# ── figures ─────────────────────────────────────────────────────────────────


PROVISIONAL_NOTE = (
    "PROVISIONAL TARGET — #1738 SAE->SAE context arm; the full-width dense->SAE arrays "
    "(task #7, pod-1482 live) are not in yet. Re-renders on one --r2-npy swap."
)
ACTIVITY_CAVEAT = (
    "The firing-frequency block IS the stratifying variable, so within an activity decile its "
    "variance is restricted BY CONSTRUCTION and its Shapley share is mechanically suppressed. "
    "Do NOT read 'activity contributes nothing within deciles' as a finding."
)


def fig_heatmap(res: dict, fig_dir: Path) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    names = res["block_names"]
    order = sorted(names, key=lambda n: -res["pooled"]["shapley_share"].get(n, 0.0))
    mat = np.array([[d["shapley_share"].get(n, np.nan) for d in res["per_decile"]] for n in order])
    flagged = res["rare_flagged_blocks_per_decile"]

    fig_h = 0.40 * len(order) + 5.4
    fig, ax = plt.subplots(figsize=(13.2, fig_h))
    # the paper style enables a constrained layout engine, which silently
    # overrides subplots_adjust; a colorbar then blocks tight_layout too
    fig.set_layout_engine("none")
    im = ax.imshow(mat, aspect="auto", cmap="magma_r", vmin=0.0, vmax=float(np.nanmax(mat)))
    ax.set_xticks(range(N_DECILES))
    ax.set_xticklabels(
        [
            f"{d['decile']}\nn={d['n']:,}\n{d['internal_activity_ratio']:.1f}x"
            for d in res["per_decile"]
        ],
        fontsize=6.6,
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [n + ("  [conditioning variable]" if n == "activity" else "") for n in order], fontsize=8.0
    )
    for i, n in enumerate(order):
        for j in range(N_DECILES):
            if n in flagged[j]:
                ax.text(j, i, "x", ha="center", va="center", fontsize=8, color="#00A0A0")
            elif np.isfinite(mat[i, j]) and mat[i, j] >= 0.10:
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="white" if mat[i, j] > 0.5 * np.nanmax(mat) else "#333333",
                )
    fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012, label="Shapley share of block $R^2$")
    ax.set_xlabel(
        "activity decile (equal-count; n and internal activity ratio annotated)", fontsize=9
    )
    fig.suptitle("Exact Shapley share by block, within activity deciles", fontsize=12.5, y=0.985)
    fig.text(
        0.5,
        0.945,
        f"{res['n_rows']:,} features, full dictionary; collinearity collapsed at |rho| >= "
        f"{res['cut']} (complete linkage)  |  {PROVISIONAL_NOTE}",
        ha="center",
        fontsize=7.2,
        color="#5A5A5A",
    )
    fig.text(
        0.5,
        1 - 1.15 / fig_h,
        "The firing-frequency block IS the stratifying variable: within an activity decile its\n"
        "variance is restricted BY CONSTRUCTION and its share is mechanically suppressed.\n"
        "Do NOT read 'activity contributes nothing within deciles' as a finding.",
        ha="center",
        va="top",
        fontsize=7.8,
        color="#8C2D04",
        linespacing=1.35,
    )
    fig.text(
        0.5,
        0.16 / fig_h,
        f"x = axis suppressed (a class below the {RARE_CLASS_FLOOR}-feature floor). Both flagged "
        "axes are GLOBALLY rare, not decile-sparse: speaker_property carries 'unclear' (22 "
        "features dictionary-wide), side_class carries answer-only (420).\n"
        "Deciles are equal-COUNT, not equal-width in log-activity — d1 (35.0x) and d10 (8.8x) "
        "are the least tightly conditioned columns. Uncertainty is split-half, not bootstrap.",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#5A5A5A",
        linespacing=1.3,
    )
    # NOT tight_layout: a colorbar has been created and matplotlib refuses the
    # layout-engine switch under the paper style (RuntimeError).
    fig.subplots_adjust(left=0.255, right=0.885, top=1 - 2.45 / fig_h, bottom=1.35 / fig_h)
    stem = "shapley_decile_heatmap"
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


def fig_pooled(res: dict, halves: list[dict], fig_dir: Path) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    p = res["pooled"]
    names = sorted(res["block_names"], key=lambda n: p["shapley"].get(n, 0.0))
    y = np.arange(len(names), dtype=float)
    sh = np.array([p["shapley"][n] for n in names])
    al = np.array([p["marginal_alone"][n] for n in names])
    la = np.array([p["unique_when_last"][n] for n in names])
    spread = np.array(
        [
            [
                min(d["shapley_share"].get(n, np.nan) for d in res["per_decile"]),
                max(d["shapley_share"].get(n, np.nan) for d in res["per_decile"]),
            ]
            for n in names
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 0.40 * len(names) + 3.4), sharey=True)
    ax = axes[0]
    ax.hlines(y, la, al, color="#BBBBBB", lw=3.0, alpha=0.8, label="bracket: unique-last .. alone")
    ax.plot(
        al,
        y,
        "|",
        ms=10,
        color=paper_palette_role("baseline"),
        mew=1.6,
        label="alone (over-counts)",
    )
    ax.plot(
        la,
        y,
        "|",
        ms=10,
        color=paper_palette_role("control"),
        mew=1.6,
        label="unique-last (under-counts)",
    )
    ax.plot(sh, y, "o", ms=5.5, color=paper_palette_role("primary"), label="exact Shapley")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.2)
    ax.set_xlabel(r"contribution to $R^2$", fontsize=9)
    ax.set_title("POOLED — the misleading contrast", fontsize=10.5, loc="left", color="#8C2D04")
    ax.legend(loc="lower right", frameon=False, fontsize=7.2)

    ax = axes[1]
    ax.hlines(
        y, spread[:, 0], spread[:, 1], color=paper_palette_role("primary"), lw=3.0, alpha=0.55
    )
    ax.plot(spread[:, 0], y, "|", ms=9, color=paper_palette_role("primary"), mew=1.5)
    ax.plot(spread[:, 1], y, "|", ms=9, color=paper_palette_role("primary"), mew=1.5)
    ax.set_xlabel("Shapley share: min .. max across activity deciles", fontsize=9)
    ax.set_title(
        "STRATIFIED — the range the pooled number averages away", fontsize=10.5, loc="left"
    )

    sh_corr = ", ".join(
        f"{h['tag']} rho={h['share_corr_vs_full']:+.3f}"
        for h in halves
        if "share_corr_vs_full" in h
    )
    fig.suptitle("Pooled vs stratified Shapley decomposition", fontsize=12.5, y=0.985)
    fig.text(
        0.5,
        0.944,
        f"{res['n_rows']:,} features; {len(res['block_names'])} blocks after collapsing "
        f"collinearity at |rho| >= {res['cut']}  |  {PROVISIONAL_NOTE}",
        ha="center",
        fontsize=7.2,
        color="#5A5A5A",
    )
    fig.text(
        0.5,
        0.012,
        "Shapley sits between two endpoints by construction: a block ALONE over-counts (the set "
        "sums above the total) and its UNIQUE contribution when added last under-counts (sums "
        "below). A wide bracket means the block shares variance with others. "
        f"Split-half share agreement: {sh_corr or 'n/a'}. Uncertainty is split-half, not bootstrap.",
        ha="center",
        fontsize=7.0,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.934))
    stem = "shapley_pooled_vs_stratified"
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


# ── entrypoint ──────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="#1482 clustered exact-Shapley decomposition")
    ap.add_argument("--r2-npy", type=Path, default=PROJECT_ROOT / R2_PROVISIONAL)
    ap.add_argument("--r2-label", default="PROVISIONAL-#1738-sae_to_sae")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / FIG_DIR)
    ap.add_argument(
        "--figs-only",
        action="store_true",
        help="re-render both figures from the committed JSON (the analysis is ~80 min)",
    )
    args = ap.parse_args()
    for d in (args.out_dir, args.fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.figs_only:
        import matplotlib

        matplotlib.use("Agg")
        doc = json.loads((args.out_dir / "shapley_blocks.json").read_text())
        stems = [
            fig_heatmap(doc["full_sample"], args.fig_dir),
            fig_pooled(doc["full_sample"], doc["half_runs"], args.fig_dir),
        ]
        _log(f"figures (from JSON): {', '.join(stems)}  ({time.time() - t0:.0f}s)")
        return

    rng = np.random.default_rng(SEED)
    inp = load_inputs(args.r2_npy)
    universe = inp["universe"]

    # cluster-cut sensitivity is the PRIMARY uncertainty instrument for step 1
    cand = candidate_names(inp["cov"])
    okc = np.ones(len(universe), dtype=bool)
    for c in cand:
        okc &= np.isfinite(inp["cov"][c][universe])
    xs = np.column_stack([inp["cov"][c][universe[okc]] for c in cand])
    sens = {}
    for cut in CUT_SENSITIVITY:
        cl, _ = cluster_covariates(xs, cand, cut)
        sens[str(cut)] = {"n_clusters": len(cl), "clusters": cl}
        _log(f"sensitivity cut {cut}: {len(cl)} clusters")

    full = run_pipeline(inp, universe, CUT_PRIMARY, rng, "full")

    # split-half: re-derive clusters INDEPENDENTLY on each half
    perm = rng.permutation(universe)
    halves = []
    for i, rows in enumerate((perm[: len(perm) // 2], perm[len(perm) // 2 :])):
        h = run_pipeline(inp, rows, CUT_PRIMARY, rng, f"half{i + 1}")
        shared = [n for n in h["block_names"] if n in full["pooled"]["shapley_share"]]
        a = np.array([h["pooled"]["shapley_share"][n] for n in shared])
        b = np.array([full["pooled"]["shapley_share"][n] for n in shared])
        h["share_corr_vs_full"] = PB._spearman(a, b) if len(shared) > 2 else float("nan")
        h["shared_blocks"] = shared
        halves.append(h)

    b1 = {tuple(sorted(c["members"])) for c in halves[0]["clusters"]}
    b2 = {tuple(sorted(c["members"])) for c in halves[1]["clusters"]}
    a = np.array(
        [
            halves[0]["pooled"]["shapley_share"][n]
            for n in halves[0]["block_names"]
            if n in halves[1]["pooled"]["shapley_share"]
        ]
    )
    b = np.array(
        [
            halves[1]["pooled"]["shapley_share"][n]
            for n in halves[0]["block_names"]
            if n in halves[1]["pooled"]["shapley_share"]
        ]
    )
    per_dec_corr = []
    for d in range(N_DECILES):
        s = [
            n for n in halves[0]["block_names"] if n in halves[1]["per_decile"][d]["shapley_share"]
        ]
        if len(s) > 2:
            per_dec_corr.append(
                PB._spearman(
                    np.array([halves[0]["per_decile"][d]["shapley_share"][n] for n in s]),
                    np.array([halves[1]["per_decile"][d]["shapley_share"][n] for n in s]),
                )
            )
        else:
            per_dec_corr.append(float("nan"))
    split = {
        "method": (
            "two disjoint halves, ENTIRE pipeline re-run independently including RE-DERIVING the "
            "clusters from each half's own correlation matrix — a fixed-cluster bootstrap cannot "
            "see clustering instability"
        ),
        "cluster_sets_identical": bool(b1 == b2),
        "cluster_jaccard": float(len(b1 & b2) / max(len(b1 | b2), 1)),
        "half1_only_clusters": [list(c) for c in sorted(b1 - b2)],
        "half2_only_clusters": [list(c) for c in sorted(b2 - b1)],
        "pooled_share_spearman_between_halves": PB._spearman(a, b) if len(a) > 2 else float("nan"),
        "per_decile_share_spearman": per_dec_corr,
        "why_not_bootstrap": (
            "SE of a Spearman is ~1/sqrt(n-3) — 0.003 pooled, 0.009 per decile — so bootstrap CIs "
            "would be ~+-0.02 on shares spanning tenths: everything reads significant and the "
            "interval carries no information. The uncertainty that matters is the provisional "
            "target, the clustering choices and effect modification, and a row bootstrap holding "
            "clusters fixed is blind to all three."
        ),
    }
    _log(
        f"split-half: clusters identical={split['cluster_sets_identical']} "
        f"jaccard={split['cluster_jaccard']:.3f} pooled share rho="
        f"{split['pooled_share_spearman_between_halves']:+.3f}"
    )

    doc = {
        "design": {
            "scope": "FULL DICTIONARY; exact Shapley on collinearity-collapsed blocks",
            "r2_source": str(args.r2_npy.relative_to(PROJECT_ROOT)),
            "r2_label": args.r2_label,
            "r2_status": PROVISIONAL_NOTE,
            "linkage": "COMPLETE (guarantees every within-cluster pair exceeds the cut)",
            "cut_primary": CUT_PRIMARY,
            "representative_rule": "interpretability, never predictive strength",
            "interpretability_rationale": INTERPRETABILITY_RATIONALE,
            "axis_blocks_are_one_player": (
                "each judged axis is ONE player; all its dummies enter and leave together, so "
                "values are invariant to which reference level is dropped"
            ),
            "unresolved_policy": UNRESOLVED_POLICY,
            "excluded_blocks": EXCLUDED_BLOCKS,
            "forced_together": [list(p) for p in FORCED_TOGETHER],
            "rare_class_floor": RARE_CLASS_FLOOR,
            "runlen_present": inp["runlen_present"],
            "pending_runlen_covariates": inp["pending_runlen"],
            "seed": SEED,
        },
        "cluster_cut_sensitivity": sens,
        "full_sample": full,
        "split_half": split,
        "half_runs": [
            {
                k: v
                for k, v in h.items()
                if k
                in ("tag", "n_rows", "clusters", "representatives", "share_corr_vs_full", "pooled")
            }
            for h in halves
        ],
        "activity_conditioning_caveat": ACTIVITY_CAVEAT,
        "metadata": PB._metadata(),
    }
    (args.out_dir / "shapley_blocks.json").write_text(json.dumps(doc, indent=1))
    _log(f"reads -> {args.out_dir / 'shapley_blocks.json'}")

    import matplotlib

    matplotlib.use("Agg")
    stems = [fig_heatmap(full, args.fig_dir), fig_pooled(full, halves, args.fig_dir)]
    _log(f"figures: {', '.join(stems)}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
