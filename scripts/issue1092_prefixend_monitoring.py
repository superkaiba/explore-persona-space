#!/usr/bin/env python3
"""#1092 inline free-analysis: does the pre-query PREFIX-END state support
prefix-averaged trait MONITORING as well as the query-AVERAGED CONTEXT state?

Context. #779 found that averaging a context-map trait read over the shared
questions of a held-out persona lifts monitoring correlation with judge scores
well above the per-prompt read (its CONTEXT-arm-only result). #1092's fair
comparison found the pre-query prefix-end state much weaker than averaged
context vectors for answer-PROFILE prediction. The missing cell is the same
prefix-end-vs-averaged-context contrast on the TRAIT-MONITORING dependent
variable (correlation with banked judge scores). Geometry predicts prefix-end
lands below averaged-context; this measures it.

Substrate (existing artifacts only — NO new forward passes, NO new judge calls).
  - #1092 arm ``cell_inst_own`` (the instruct model scoring its OWN on-policy
    answers — the realistic deployment arm), layer 14 pinned for the headline.
  - Rows: the ``dense_core`` (99 prefixes) + ``battery`` (50 prefixes) strata,
    which share the SAME 48 core queries and have DISJOINT prefixes -> a clean
    149-prefix x 48-query grid. Group = prefix_id.
  - Banked judge scores: Claude-sonnet-4-5-20250929, 5 draws, temp 1.0, graded
    0-100, mean-aggregated (``p5_judge/scores_shard_*.jsonl``). Dropped judge
    returns already excluded.
  - States: fp16 (N_rows, 3584) row-aligned to the corpus manifest;
    ``context_end`` (prefix+query) per row, ``prefix_end`` (prefix only,
    query-invariant -> constant within a prefix).
  - Traits: sycophancy, hallucination. evil is INELIGIBLE (0 judged positives
    on this arm) and is not run.

Readout (single construction, applied identically to every state/arm). The
canonical GCV dual-ridge ``fit_h.ridge_fit_predict`` (standardize-X on train /
center-Y / GCV lambda over logspace(-2,4,13) / dual solve) is fit at ROW level,
grouped-K-fold held out over prefixes (a prefix is entirely train or test), with
the per-row judge score as the target. The SAME fit (same rows, folds, target,
ridge) is run for context_end and for prefix_end; only the input state differs.
Prefix_end is constant within a prefix, so its held-out row prediction is
constant within a prefix by construction.

Reads (per trait), from the held-out predictions:
  (a) per-row context  = Pearson(pred_context_row, score_row) at ROW level
                         (the per-prompt monitoring reference, ~#779's 0.34).
  (b) averaged-context = Pearson(prefix-mean pred_context, prefix-mean score)
                         at GROUP level (the #779 averaging boost).
  (c) prefix-end       = Pearson(prefix pred_prefix_end, prefix-mean score)
                         at GROUP level (query-invariant; the tested arm).
Headline paired difference r(avg-context) - r(prefix-end), group-bootstrap CI.
Averaging curve: r as a function of N averaged questions for context, with
prefix-end a flat (query-invariant) reference line.

All fits are closed-form ridge; the bootstrap only recomputes correlations on
cached held-out predictions (no re-fit). CPU-only, thread-capped.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
REV = "b7dbd38cd08a2827860cf5fbe19c2384e2da7dec"
BASE = "issue1092_realistic_crossing"
DST = PROJECT_ROOT / "data/issue_1092/hf_dl" / BASE
OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_prefixend_monitoring"
FIG_DIR = PROJECT_ROOT / "figures/issue_1092"

CELLS = ["cell_inst_own", "cell_pre_own"]
TRAITS = ["sycophancy", "hallucination"]
STRATA = ("dense_core", "battery")
STATES = ("context_end", "prefix_end")
HIDDEN = 3584
N_FOLDS = 5
N_BOOT = 2000
SEED = 0
NGRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
N_AVG_SEEDS = 25  # seeds averaged per point of the N-averaging curve

# Round-2 readout constructions (r_B projection + map-mediated).
RB_REV = "037fcbb"  # r_B direction revision the #1092 B1 panel used (b1a rb.rev)
RB_PREFIX = "issue779_monitoring/r_b"
ANSWER_TARGETS = ("t1", "t2", "t3")  # map target = stacked answer states, ambient (fit-arm A)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def stage(layers: list[int]) -> None:
    """Fetch manifest, judge-score shards, and the needed state files (per-file,
    revision-pinned, retried/atomic; NEVER snapshot_download)."""
    hub.stage_hub_file(
        REPO, f"{BASE}/corpus/manifest.jsonl", DST / "corpus/manifest.jsonl", revision=REV
    )
    for i in range(9):
        rel = f"p5_judge/scores_shard_{i:03d}.jsonl"
        hub.stage_hub_file(REPO, f"{BASE}/{rel}", DST / rel, revision=REV)
    for cell in CELLS:
        for st in STATES:
            for L in layers:
                rel = f"analysis_tensors/summaries/{cell}/{st}_L{L:02d}.npy"
                hub.stage_hub_file(REPO, f"{BASE}/{rel}", DST / rel, revision=REV)


def load_manifest() -> list[dict]:
    with open(DST / "corpus/manifest.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_scores() -> list[dict]:
    rows: list[dict] = []
    for i in range(9):
        with open(DST / f"p5_judge/scores_shard_{i:03d}.jsonl") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def grouped_kfold(prefixes: list[str], k: int, seed: int) -> list[np.ndarray]:
    """Return k arrays of TEST prefix indices (into the sorted unique-prefix list)."""
    uniq = np.array(sorted(set(prefixes)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    return [np.sort(f) for f in np.array_split(perm, k)]


def cv_group_readout(
    g_state: np.ndarray,
    g_score: np.ndarray,
    row_state: np.ndarray,
    pref_idx: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ONE monitor per fold: GCV ridge on the TRAIN prefixes' GROUP-level
    states -> per-prefix mean score, grouped-K-fold held out over prefixes.
    Return (held-out per-prefix prediction from the group state, held-out per-row
    prediction from applying that same monitor to each held-out row's state).

    The monitor is fit on group states (n_prefix samples, cheap SVD) and applied
    both to the held-out group state (the averaged-state read) and to each
    held-out row's state (the per-prompt read + the averaging curve). The ridge
    map is affine, so the prefix-mean of the per-row predictions equals the
    group-state prediction -> the averaging curve interpolates the two grains
    coherently.
    """
    n_pref = g_state.shape[0]
    pred_g = np.full(n_pref, np.nan, dtype=np.float64)
    pred_row = np.full(row_state.shape[0], np.nan, dtype=np.float64)
    all_pref = np.arange(n_pref)
    for test in folds:
        train = np.setdiff1d(all_pref, test)
        if len(train) < 3 or len(test) == 0:
            continue
        pred_g[test] = F.ridge_fit_predict(g_state[train], g_score[train], g_state[test])
        row_mask = np.isin(pref_idx, test)
        if row_mask.any():
            pred_row[row_mask] = F.ridge_fit_predict(
                g_state[train], g_score[train], row_state[row_mask]
            )
    return pred_g, pred_row


def process_cell_trait(  # noqa: C901
    cell: str,
    trait: str,
    layer: int,
    manifest: list[dict],
    man_by_id: dict,
    scores: list[dict],
) -> dict | None:
    # per-row judge score for this cell/trait on the dense_core+battery grid
    score_by_rowid: dict[str, float] = {}
    for r in scores:
        if r["cell_id"] != cell or r["trait"] != trait:
            continue
        if r.get("dropped") or r.get("score") is None:
            continue
        if r.get("stratum") not in STRATA:
            continue
        score_by_rowid[r["row_id"]] = float(r["score"])
    if not score_by_rowid:
        return None

    # positional rows (npy row index == manifest line index) with a score
    positions, y, prefixes, queries = [], [], [], []
    for i, d in enumerate(manifest):
        if d["stratum"] in STRATA and d["row_id"] in score_by_rowid:
            positions.append(i)
            y.append(score_by_rowid[d["row_id"]])
            prefixes.append(d["prefix_id"])
            queries.append(d["query_id"])
    positions = np.asarray(positions, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    query_of_row = np.asarray(queries)
    uniq_pref = np.array(sorted(set(prefixes)))
    pref_to_idx = {p: j for j, p in enumerate(uniq_pref)}
    pref_idx = np.asarray([pref_to_idx[p] for p in prefixes], dtype=np.int64)
    n_pref = len(uniq_pref)

    ctx = np.load(
        DST / f"analysis_tensors/summaries/{cell}/context_end_L{layer:02d}.npy", mmap_mode="r"
    )
    pre = np.load(
        DST / f"analysis_tensors/summaries/{cell}/prefix_end_L{layer:02d}.npy", mmap_mode="r"
    )
    X_ctx = np.asarray(ctx[positions], dtype=np.float64)
    X_pre = np.asarray(pre[positions], dtype=np.float64)
    del ctx, pre

    # prefix_end constancy sanity (query-invariant within a prefix)
    within = []
    for j in range(min(n_pref, 20)):
        m = pref_idx == j
        within.append(float(X_pre[m].std(axis=0).mean()))
    prefix_end_within_std = float(np.mean(within))

    # per-prefix group states + target (prefix_end is constant within a prefix)
    g_ctx = np.stack([X_ctx[pref_idx == j].mean(0) for j in range(n_pref)])
    g_pre = np.stack([X_pre[pref_idx == j][0] for j in range(n_pref)])
    g_score = np.array([y[pref_idx == j].mean() for j in range(n_pref)])
    g_std = np.array([float(y[pref_idx == j].std()) for j in range(n_pref)])
    g_nq = np.array([int((pref_idx == j).sum()) for j in range(n_pref)])

    # ONE monitor per fold, fit on TRAIN group states; applied to held-out group
    # state (averaged read) and held-out rows (per-prompt read + averaging curve).
    folds = grouped_kfold(list(prefixes), N_FOLDS, SEED)
    g_pred_ctx, pred_ctx = cv_group_readout(g_ctx, g_score, X_ctx, pref_idx, folds)
    g_pred_pre, _ = cv_group_readout(g_pre, g_score, X_pre, pref_idx, folds)

    # Design-aligned split-half reliability of the per-prefix mean target
    # (llm-judging rule 21): every prefix shares the SAME query set, so the
    # query MAIN effect cancels between prefixes and must NOT be counted as
    # sampling noise (a naive within-prefix-variance/n ceiling does exactly that
    # and is far too low). Split the shared queries into aligned halves, take
    # each prefix's half-means, Pearson across prefixes, Spearman-Brown up.
    rows_by_pref = [np.where(pref_idx == j)[0] for j in range(n_pref)]
    uniq_q = np.array(sorted(set(query_of_row.tolist())))
    half_r = []
    for s in range(20):
        rng = np.random.default_rng(7000 + s)
        perm = rng.permutation(len(uniq_q))
        qa = set(uniq_q[perm[: len(uniq_q) // 2]].tolist())
        a_mask = np.array([q in qa for q in query_of_row])
        ha = np.array(
            [
                np.nan_to_num(y[rows_by_pref[j]][a_mask[rows_by_pref[j]]].mean())
                for j in range(n_pref)
            ]
        )
        hb = np.array(
            [
                np.nan_to_num(y[rows_by_pref[j]][~a_mask[rows_by_pref[j]]].mean())
                for j in range(n_pref)
            ]
        )
        half_r.append(pearson(ha, hb))
    r_hh = float(np.nanmean(half_r))
    reliability = float(2 * r_hh / (1 + r_hh)) if r_hh > -1 else float("nan")  # Spearman-Brown
    r_ceiling = (
        float(np.sqrt(reliability))
        if reliability == reliability and reliability > 0
        else float("nan")
    )

    # point reads (read + target both over ALL 48 shared queries)
    r_perrow = pearson(pred_ctx, y)
    r_avgctx = pearson(g_pred_ctx, g_score)
    r_prefixend = pearson(g_pred_pre, g_score)

    # DISJOINT-query-half robustness reads: read on query-half A, target on the
    # per-prefix mean over query-half B. Removes the shared per-query
    # judge-noise/interaction realization that the averaged-context read (but not
    # the query-invariant prefix-end read) could otherwise track. prefix-end is
    # query-invariant so its read is unchanged across halves.
    disj_avgctx, disj_prefixend = [], []
    for s in range(20):
        rng = np.random.default_rng(9000 + s)
        perm = rng.permutation(len(uniq_q))
        qa = set(uniq_q[perm[: len(uniq_q) // 2]].tolist())
        a_mask = np.array([q in qa for q in query_of_row])
        for read_mask, tgt_mask in ((a_mask, ~a_mask), (~a_mask, a_mask)):
            read_ctx = np.array(
                [
                    np.nanmean(pred_ctx[rows_by_pref[j]][read_mask[rows_by_pref[j]]])
                    for j in range(n_pref)
                ]
            )
            tgt = np.array(
                [np.nanmean(y[rows_by_pref[j]][tgt_mask[rows_by_pref[j]]]) for j in range(n_pref)]
            )
            disj_avgctx.append(pearson(read_ctx, tgt))
            disj_prefixend.append(pearson(g_pred_pre, tgt))
    r_avgctx_disjoint = float(np.nanmean(disj_avgctx))
    r_prefixend_disjoint = float(np.nanmean(disj_prefixend))

    # N-averaging curve for context (prefix-end is flat: query-invariant)
    curve = []
    for N in NGRID:
        rs = []
        for s in range(N_AVG_SEEDS):
            rng = np.random.default_rng(1000 * s + N)
            gp = np.empty(n_pref)
            for j in range(n_pref):
                idxs = rows_by_pref[j]
                take = idxs if len(idxs) <= N else rng.choice(idxs, N, replace=False)
                gp[j] = np.nanmean(pred_ctx[take])
            rs.append(pearson(gp, g_score))
        curve.append({"N": N, "r_mean": float(np.nanmean(rs)), "r_sd": float(np.nanstd(rs))})

    # group bootstrap over prefixes (paired: same resample for every read)
    rng = np.random.default_rng(SEED)
    b_perrow, b_avgctx, b_prefixend, b_diff = [], [], [], []
    for _ in range(N_BOOT):
        samp = rng.integers(0, n_pref, n_pref)
        # group-level reads
        b_avgctx.append(pearson(g_pred_ctx[samp], g_score[samp]))
        b_prefixend.append(pearson(g_pred_pre[samp], g_score[samp]))
        b_diff.append(b_avgctx[-1] - b_prefixend[-1])
        # row-level read over the resampled prefixes' rows
        rr_pred, rr_y = [], []
        for j in samp:
            rr_pred.append(pred_ctx[rows_by_pref[j]])
            rr_y.append(y[rows_by_pref[j]])
        b_perrow.append(pearson(np.concatenate(rr_pred), np.concatenate(rr_y)))

    def ci(v):
        v = np.array(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    diff = np.array(b_diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    return {
        "cell": cell,
        "trait": trait,
        "layer": layer,
        "n_rows": len(y),
        "n_prefixes": int(n_pref),
        "avg_queries_per_prefix": float(np.mean(g_nq)),
        "prefix_end_within_prefix_dimstd": prefix_end_within_std,
        "judge_score": {"mean": float(y.mean()), "std": float(y.std())},
        "target_split_half_r_aligned": r_hh,
        "target_reliability_per_prefix_mean": reliability,
        "monitoring_r_ceiling_from_reliability": r_ceiling,
        "reads": {
            "per_row_context": {"r": r_perrow, "ci95": ci(b_perrow), "grain": "row"},
            "averaged_context": {"r": r_avgctx, "ci95": ci(b_avgctx), "grain": "prefix"},
            "prefix_end": {"r": r_prefixend, "ci95": ci(b_prefixend), "grain": "prefix"},
        },
        "disjoint_half_reads": {
            "averaged_context": r_avgctx_disjoint,
            "prefix_end": r_prefixend_disjoint,
            "paired_diff_avgctx_minus_prefixend": r_avgctx_disjoint - r_prefixend_disjoint,
            "note": (
                "read on query-half A, target per-prefix mean on disjoint query-half B "
                "(removes shared per-query realization; mean over 20 aligned splits x 2 dirs)"
            ),
        },
        "paired_diff_avgctx_minus_prefixend": {
            "point": r_avgctx - r_prefixend,
            "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "frac_boot_positive": float(np.mean(diff > 0)),
        },
        "averaging_curve_context": curve,
        "prefix_end_flat_reference_r": r_prefixend,
        "per_prefix": {
            "prefix_id": uniq_pref.tolist(),
            "read_prefix_end": [float(x) for x in g_pred_pre],
            "read_averaged_context": [float(x) for x in g_pred_ctx],
            "judge_mean": [float(x) for x in g_score],
            "judge_std": [float(x) for x in g_std],
            "n_queries": [int(x) for x in g_nq],
        },
    }


def make_figure(results: dict, headline_layer: int) -> dict:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("blog")
    cell = "cell_inst_own"
    cells_present = [t for t in results["cells"].get(cell, {}).get(str(headline_layer), [])]
    if not cells_present:
        return {}
    traits = [d["trait"] for d in cells_present]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), layout="constrained")

    # Panel A: three reads per trait with bootstrap CIs + reliability ceiling.
    axA = axes[0]
    labels = ["per-prompt\ncontext", "prefix-averaged\ncontext", "prefix-end\n(pre-query)"]
    colors = [
        pp.paper_palette_role("neutral"),
        pp.paper_palette_role("primary"),
        pp.paper_palette_role("accent"),
    ]
    keys = ["per_row_context", "averaged_context", "prefix_end"]
    x = np.arange(len(traits))
    w = 0.26
    for k, (key, lab, col) in enumerate(zip(keys, labels, colors, strict=True)):
        vals = [d["reads"][key]["r"] for d in cells_present]
        los = [d["reads"][key]["r"] - d["reads"][key]["ci95"][0] for d in cells_present]
        his = [d["reads"][key]["ci95"][1] - d["reads"][key]["r"] for d in cells_present]
        los = [max(0.0, v) for v in los]
        his = [max(0.0, v) for v in his]
        axA.bar(x + (k - 1) * w, vals, w, label=lab, color=col)
        axA.errorbar(
            x + (k - 1) * w, vals, yerr=[los, his], fmt="none", ecolor="#333", capsize=3, lw=1.2
        )
    for i, d in enumerate(cells_present):
        c = d["monitoring_r_ceiling_from_reliability"]
        axA.plot(
            [x[i] - 1.5 * w, x[i] + 1.5 * w],
            [c, c],
            ls="--",
            color="#888",
            lw=1.4,
            label="reliability ceiling" if i == 0 else None,
        )
    axA.set_xticks(x)
    axA.set_xticklabels([t.capitalize() for t in traits])
    axA.set_ylabel("Monitoring correlation r (with judge score)")
    axA.set_title("Trait monitoring: prefix-end vs averaged-context")
    axA.axhline(0, color="#bbb", lw=0.8)
    axA.legend(loc="upper right", fontsize=9)

    # Panel B: N-averaging curve (context) with prefix-end flat reference.
    axB = axes[1]
    tcolors = pp.paper_palette_blog(len(traits))
    for d, tc in zip(cells_present, tcolors, strict=True):
        cv = d["averaging_curve_context"]
        Ns = [c["N"] for c in cv]
        rm = [c["r_mean"] for c in cv]
        sd = [c["r_sd"] for c in cv]
        axB.plot(
            Ns, rm, "-o", color=tc, ms=4, label=f"{d['trait'].capitalize()} — averaged context"
        )
        axB.fill_between(
            Ns, np.array(rm) - np.array(sd), np.array(rm) + np.array(sd), color=tc, alpha=0.18
        )
        axB.axhline(
            d["prefix_end_flat_reference_r"],
            ls="--",
            color=tc,
            lw=1.4,
            label=f"{d['trait'].capitalize()} — prefix-end (query-invariant)",
        )
    axB.set_xlabel("Number of questions averaged per prefix (N)")
    axB.set_ylabel("Monitoring correlation r")
    axB.set_title("Averaging context questions vs the single prefix-end read")
    axB.set_xscale("log")
    axB.set_xticks(NGRID)
    axB.set_xticklabels([str(n) for n in NGRID])
    axB.legend(loc="lower right", fontsize=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = pp.savefig_paper(fig, "prefixend_monitoring", dir=FIG_DIR)
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def _by_trait(results: dict, cell: str, layer: int) -> dict:
    return {d["trait"]: d for d in results["cells"].get(cell, {}).get(str(layer), [])}


def _scatter_panel(ax, xv, yv, xlabel, ylabel, color, title) -> None:
    from scipy import stats

    xv = np.asarray(xv, dtype=np.float64)
    yv = np.asarray(yv, dtype=np.float64)
    ax.scatter(xv, yv, s=26, alpha=0.5, color=color, edgecolors="none")
    r, p = stats.pearsonr(xv, yv)
    b, a = np.polyfit(xv, yv, 1)
    xs = np.linspace(xv.min(), xv.max(), 50)
    ax.plot(xs, a + b * xs, color="#333", lw=1.4)
    ptxt = "p<0.001" if p < 1e-3 else f"p={p:.3f}"
    ax.annotate(
        f"r = {r:.2f}  ({ptxt})\nn = {len(xv)} prefixes",
        xy=(0.04, 0.93),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)


def make_scatter_figures(results: dict, layer: int) -> dict:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    out: dict = {}
    pp.set_paper_style("blog")
    c_pre = pp.paper_palette_role("accent")
    c_ctx = pp.paper_palette_role("primary")

    # Instruct arm: rows = traits, cols = [prefix-end, averaged-context].
    inst = _by_trait(results, "cell_inst_own", layer)
    traits = [t for t in TRAITS if t in inst]
    if traits:
        fig, axes = plt.subplots(
            len(traits), 2, figsize=(11.0, 4.6 * len(traits)), layout="constrained"
        )
        axes = np.atleast_2d(axes)
        for row, trait in enumerate(traits):
            pp_ = inst[trait]["per_prefix"]
            ym = pp_["judge_mean"]
            # consistent axes per trait row: shared x + y across the two arms
            allx = pp_["read_prefix_end"] + pp_["read_averaged_context"]
            xlo, xhi = min(allx), max(allx)
            xpad = 0.04 * (xhi - xlo + 1e-9)
            ylo, yhi = min(ym), max(ym)
            ypad = 0.06 * (yhi - ylo + 1e-9)
            for col, (arm_key, arm_lab, arm_col) in enumerate(
                [
                    ("read_prefix_end", "prefix-end", c_pre),
                    ("read_averaged_context", "averaged-context", c_ctx),
                ]
            ):
                ax = axes[row, col]
                _scatter_panel(
                    ax,
                    pp_[arm_key],
                    ym,
                    f"held-out {arm_lab} read (ridge prediction)",
                    f"per-prefix mean judge score (0-100), {trait}",
                    arm_col,
                    f"{trait.capitalize()} — {arm_lab}",
                )
                ax.set_xlim(xlo - xpad, xhi + xpad)
                ax.set_ylim(ylo - ypad, yhi + ypad)
        paths = pp.savefig_paper(fig, "prefixend_monitoring_scatter", dir=FIG_DIR)
        plt.close(fig)
        out.update({f"instruct_{k}": str(v) for k, v in paths.items()})

    # Base-model collapse case: hallucination, prefix-end vs averaged-context.
    base = _by_trait(results, "cell_pre_own", layer)
    if "hallucination" in base:
        pp_ = base["hallucination"]["per_prefix"]
        ym = pp_["judge_mean"]
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), layout="constrained")
        allx = pp_["read_prefix_end"] + pp_["read_averaged_context"]
        xlo, xhi = min(allx), max(allx)
        xpad = 0.04 * (xhi - xlo + 1e-9)
        for col, (arm_key, arm_lab, arm_col) in enumerate(
            [
                ("read_prefix_end", "prefix-end", c_pre),
                ("read_averaged_context", "averaged-context", c_ctx),
            ]
        ):
            _scatter_panel(
                axes[col],
                pp_[arm_key],
                ym,
                f"held-out {arm_lab} read (ridge prediction)",
                "per-prefix mean judge score (0-100), hallucination",
                arm_col,
                f"Base model — hallucination — {arm_lab}",
            )
            axes[col].set_xlim(xlo - xpad, xhi + xpad)
        paths = pp.savefig_paper(fig, "prefixend_monitoring_scatter_base", dir=FIG_DIR)
        plt.close(fig)
        out.update({f"base_{k}": str(v) for k, v in paths.items()})
    return out


def write_per_prefix_points(results: dict, layer: int, out_dir: Path) -> str:
    payload = {
        "read": "#1092 prefix-end vs averaged-context monitoring — per-prefix points",
        "layer": layer,
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_sha(),
        "provenance": results["provenance"],
        "columns": (
            "prefix_id, read_prefix_end, read_averaged_context (held-out ridge "
            "predictions of judge score), judge_mean, judge_std, n_queries"
        ),
        "cells": {},
    }
    for cell in CELLS:
        by_trait = _by_trait(results, cell, layer)
        if by_trait:
            payload["cells"][cell] = {t: by_trait[t]["per_prefix"] for t in by_trait}
    path = out_dir / "per_prefix_points.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return str(path)


# ---------------------------------------------------------------------------
# Round-2 readout constructions: raw r_B projection + map-mediated projection.
# All three constructions (these two + round-1 supervised ridge) are compared at
# the identical 149-prefix persona-averaged grain with the same folds/bootstrap.
# ---------------------------------------------------------------------------


def stage_constructions() -> None:
    """Fetch r_B directions (sycophancy, hallucination) + the answer states
    (t1/t2/t3, L14) that the map-mediated construction transports onto."""
    for t in TRAITS:
        hub.stage_hub_file(
            REPO,
            f"{RB_PREFIX}/{t}.pt",
            PROJECT_ROOT / "data/issue_1092/hf_dl" / RB_PREFIX / f"{t}.pt",
            revision=RB_REV,
        )
    for cell in CELLS:
        for tgt in ANSWER_TARGETS:
            rel = f"analysis_tensors/summaries/{cell}/{tgt}_L14.npy"
            hub.stage_hub_file(REPO, f"{BASE}/{rel}", DST / rel, revision=REV)


def load_rb_l14() -> dict[str, np.ndarray]:
    """Per-trait r_B unit direction at layer 14 (banked convention: state @ rb /
    ||rb||; NO state centering; sign fixed by the extraction)."""
    out: dict[str, np.ndarray] = {}
    for t in TRAITS:
        p = PROJECT_ROOT / "data/issue_1092/hf_dl" / RB_PREFIX / f"{t}.pt"
        payload = torch.load(p, map_location="cpu", weights_only=False)
        arr = payload["r_b"] if isinstance(payload, dict) and "r_b" in payload else payload
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float64)
        assert arr.shape == (28, HIDDEN), (t, arr.shape)
        out[t] = arr[14]
    return out


def assemble_substrate(
    cell: str, trait: str, layer: int, manifest: list[dict], man_by_id: dict, scores: list[dict]
) -> dict | None:
    """The identical 149-prefix substrate round 1 builds, returned as a dict so the
    construction reads reuse the exact positions / folds / grouping."""
    score_by_rowid = {
        r["row_id"]: float(r["score"])
        for r in scores
        if r["cell_id"] == cell
        and r["trait"] == trait
        and not r.get("dropped")
        and r.get("score") is not None
        and r.get("stratum") in STRATA
    }
    if not score_by_rowid:
        return None
    positions, y, prefixes = [], [], []
    for i, d in enumerate(manifest):
        if d["stratum"] in STRATA and d["row_id"] in score_by_rowid:
            positions.append(i)
            y.append(score_by_rowid[d["row_id"]])
            prefixes.append(d["prefix_id"])
    positions = np.asarray(positions, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    uniq_pref = np.array(sorted(set(prefixes)))
    pref_to_idx = {p: j for j, p in enumerate(uniq_pref)}
    pref_idx = np.asarray([pref_to_idx[p] for p in prefixes], dtype=np.int64)
    n_pref = len(uniq_pref)
    g_score = np.array([y[pref_idx == j].mean() for j in range(n_pref)])
    g_nq = np.array([int((pref_idx == j).sum()) for j in range(n_pref)])
    folds = grouped_kfold(list(prefixes), N_FOLDS, SEED)
    rows_by_pref = [np.where(pref_idx == j)[0] for j in range(n_pref)]
    return {
        "positions": positions,
        "y": y,
        "uniq_pref": uniq_pref,
        "pref_idx": pref_idx,
        "n_pref": n_pref,
        "g_score": g_score,
        "g_nq": g_nq,
        "folds": folds,
        "rows_by_pref": rows_by_pref,
    }


def cv_group_map_projection(
    g_state: np.ndarray,
    g_answer: np.ndarray,
    row_state: np.ndarray | None,
    pref_idx: np.ndarray,
    folds: list[np.ndarray],
    rb_unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit the state->stacked-answer transport map per fold (group-level, ambient,
    grouped-OOF over prefixes), then project the predicted t1-block answer onto the
    unit r_B. Returns (per-prefix projection, per-row projection or None). One ridge
    fit per fold; predictions dotted with r_B inside the loop so the 10752-wide
    answer array never accumulates. Pass ``row_state=None`` for the query-invariant
    prefix-end arm, which needs no per-row prediction."""
    n_pref = g_state.shape[0]
    proj_g = np.full(n_pref, np.nan, dtype=np.float64)
    proj_row = None if row_state is None else np.full(row_state.shape[0], np.nan, dtype=np.float64)
    all_pref = np.arange(n_pref)
    for test in folds:
        train = np.setdiff1d(all_pref, test)
        if len(train) < 3 or len(test) == 0:
            continue
        row_mask = np.isin(pref_idx, test) if row_state is not None else None
        eval_g = g_state[test]
        if row_state is not None and row_mask.any():
            eval_X = np.vstack([eval_g, row_state[row_mask]])
        else:
            eval_X = eval_g
        pred = F.ridge_fit_predict(g_state[train], g_answer[train], eval_X)
        proj = pred[:, :HIDDEN] @ rb_unit  # rb_out is nonzero only in the t1 block
        proj_g[test] = proj[: len(test)]
        if row_state is not None and row_mask.any():
            proj_row[row_mask] = proj[len(test) :]
    return proj_g, proj_row


def _boot_reads(
    row_ctx: np.ndarray,
    pre_read: np.ndarray,
    y: np.ndarray,
    g_score: np.ndarray,
    pref_idx: np.ndarray,
    rows_by_pref: list[np.ndarray],
    n_pref: int,
) -> dict:
    """Three reads from a construction's per-row CONTEXT values + per-prefix
    PREFIX-END values, with prefix-level bootstrap CIs + the paired
    averaged-context - prefix-end difference. Signed r throughout."""
    g_ctx = np.array([np.nanmean(row_ctx[pref_idx == j]) for j in range(n_pref)])
    r_perrow = pearson(row_ctx, y)
    r_avgctx = pearson(g_ctx, g_score)
    r_prefixend = pearson(pre_read, g_score)
    rng = np.random.default_rng(SEED)
    b_perrow, b_avgctx, b_pre, b_diff = [], [], [], []
    for _ in range(N_BOOT):
        samp = rng.integers(0, n_pref, n_pref)
        b_avgctx.append(pearson(g_ctx[samp], g_score[samp]))
        b_pre.append(pearson(pre_read[samp], g_score[samp]))
        b_diff.append(b_avgctx[-1] - b_pre[-1])
        rr = [row_ctx[rows_by_pref[j]] for j in samp]
        ry = [y[rows_by_pref[j]] for j in samp]
        b_perrow.append(pearson(np.concatenate(rr), np.concatenate(ry)))

    def ci(v):
        v = np.array(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    diff = np.array(b_diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    return {
        "per_row_context": {"r": r_perrow, "ci95": ci(b_perrow), "grain": "row"},
        "averaged_context": {"r": r_avgctx, "ci95": ci(b_avgctx), "grain": "prefix"},
        "prefix_end": {"r": r_prefixend, "ci95": ci(b_pre), "grain": "prefix"},
        "paired_diff_avgctx_minus_prefixend": {
            "point": r_avgctx - r_prefixend,
            "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "frac_boot_positive": float(np.mean(diff > 0)),
        },
    }


def run_constructions(
    manifest: list[dict], man_by_id: dict, scores: list[dict], out_dir: Path
) -> tuple[dict, str]:
    """Compute the raw-r_B and map-mediated constructions at the 149-prefix grain,
    L14, and package them beside the round-1 supervised anchor from results.json."""
    layer = 14
    rb = load_rb_l14()
    supervised = json.load(open(out_dir / "results.json"))

    result: dict = {
        "read": "#1092 monitoring readout CONSTRUCTIONS (raw r_B + map-mediated vs supervised anchor)",
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_sha(),
        "layer": layer,
        "provenance": {
            "rb_source": f"{REPO}:{RB_PREFIX}/<trait>.pt @ {RB_REV} (b1a rb.rev); layer 14",
            "rb_projection": "raw: state @ rb / ||rb|| (no centering, sign-fixed by extraction)",
            "map_mediated": (
                "state->stacked(t1,t2,t3) ambient transport map (fit_h.ridge_fit_predict, "
                "group-level per-prefix, grouped 5-fold OOF, same folds as round 1), then the "
                "predicted t1-block projected onto rb/||rb|| (rb_out t1-block convention)"
            ),
            "answer_states": "analysis_tensors/summaries/<cell>/{t1,t2,t3}_L14.npy",
            "supervised_anchor": "eval_results/issue_1092/inline_prefixend_monitoring/results.json",
            "grain": "149 dense_core+battery prefixes x 48 shared core queries; group=prefix_id",
        },
        "constructions": {"raw_rb_projection": {}, "map_mediated": {}},
        "supervised_anchor": {},
        "per_prefix": {},
    }

    for cell in CELLS:
        sub_by_trait = {
            d["trait"]: d for d in supervised["cells"].get(cell, {}).get(str(layer), [])
        }
        result["constructions"]["raw_rb_projection"][cell] = {}
        result["constructions"]["map_mediated"][cell] = {}
        result["supervised_anchor"][cell] = {}
        result["per_prefix"][cell] = {}
        # answer states for this cell (map target), stacked t1/t2/t3
        ans = {
            tgt: np.load(DST / f"analysis_tensors/summaries/{cell}/{tgt}_L14.npy", mmap_mode="r")
            for tgt in ANSWER_TARGETS
        }
        ctx_np = np.load(
            DST / f"analysis_tensors/summaries/{cell}/context_end_L14.npy", mmap_mode="r"
        )
        pre_np = np.load(
            DST / f"analysis_tensors/summaries/{cell}/prefix_end_L14.npy", mmap_mode="r"
        )
        for trait in TRAITS:
            sub = assemble_substrate(cell, trait, layer, manifest, man_by_id, scores)
            if sub is None:
                continue
            pos = sub["positions"]
            pref_idx = sub["pref_idx"]
            n_pref = sub["n_pref"]
            g_score = sub["g_score"]
            folds = sub["folds"]
            rows_by_pref = sub["rows_by_pref"]
            y = sub["y"]
            rb_unit = rb[trait] / np.linalg.norm(rb[trait])

            X_ctx = np.asarray(ctx_np[pos], dtype=np.float64)
            X_pre = np.asarray(pre_np[pos], dtype=np.float64)
            g_pre = np.stack([X_pre[pref_idx == j][0] for j in range(n_pref)])

            # --- construction 1: raw r_B projection (unsupervised, no fit) ---
            row_ctx_raw = X_ctx @ rb_unit
            pre_raw = g_pre @ rb_unit
            raw = _boot_reads(row_ctx_raw, pre_raw, y, g_score, pref_idx, rows_by_pref, n_pref)
            result["constructions"]["raw_rb_projection"][cell][trait] = raw

            # --- construction 2: map-mediated projection (OOF transport map) ---
            g_ctx = np.stack([X_ctx[pref_idx == j].mean(0) for j in range(n_pref)])
            Y_stacked = np.concatenate(
                [np.asarray(ans[t][pos], dtype=np.float64) for t in ANSWER_TARGETS], axis=1
            )
            g_answer = np.stack([Y_stacked[pref_idx == j].mean(0) for j in range(n_pref)])
            _, mp_row_ctx = cv_group_map_projection(
                g_ctx, g_answer, X_ctx, pref_idx, folds, rb_unit
            )
            mp_g_pre, _ = cv_group_map_projection(g_pre, g_answer, None, pref_idx, folds, rb_unit)
            mapmed = _boot_reads(mp_row_ctx, mp_g_pre, y, g_score, pref_idx, rows_by_pref, n_pref)
            result["constructions"]["map_mediated"][cell][trait] = mapmed
            del Y_stacked, g_answer
            gc.collect()

            # --- supervised anchor (round 1) + per-prefix construction reads ---
            sup = sub_by_trait.get(trait, {}).get("reads", {})
            result["supervised_anchor"][cell][trait] = {
                "averaged_context": sup.get("averaged_context", {}).get("r"),
                "averaged_context_ci95": sup.get("averaged_context", {}).get("ci95"),
                "prefix_end": sup.get("prefix_end", {}).get("r"),
                "prefix_end_ci95": sup.get("prefix_end", {}).get("ci95"),
                "per_row_context": sup.get("per_row_context", {}).get("r"),
            }
            result["per_prefix"][cell][trait] = {
                "prefix_id": sub["uniq_pref"].tolist(),
                "raw_rb_read_prefix_end": [float(x) for x in pre_raw],
                "raw_rb_read_averaged_context": [
                    float(np.nanmean(row_ctx_raw[pref_idx == j])) for j in range(n_pref)
                ],
                "map_mediated_read_prefix_end": [float(x) for x in mp_g_pre],
                "map_mediated_read_averaged_context": [
                    float(np.nanmean(mp_row_ctx[pref_idx == j])) for j in range(n_pref)
                ],
                "judge_mean": [float(x) for x in g_score],
            }
            print(
                f"[{cell} {trait}] raw_rb: avgctx={raw['averaged_context']['r']:.3f} "
                f"prefixend={raw['prefix_end']['r']:.3f} | "
                f"map_med: avgctx={mapmed['averaged_context']['r']:.3f} "
                f"prefixend={mapmed['prefix_end']['r']:.3f} | "
                f"supervised: avgctx={result['supervised_anchor'][cell][trait]['averaged_context']} "
                f"prefixend={result['supervised_anchor'][cell][trait]['prefix_end']}",
                flush=True,
            )
        del ctx_np, pre_np, ans
        gc.collect()

    path = out_dir / "readout_constructions.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    return result, str(path)


def make_constructions_figure(result: dict) -> dict:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("blog")
    cell = "cell_inst_own"
    traits = [t for t in TRAITS if t in result["constructions"]["raw_rb_projection"].get(cell, {})]
    if not traits:
        return {}
    # three constructions x two arms; supervised is the anchor.
    constr = [
        ("supervised ridge", "supervised", pp.paper_palette_role("neutral")),
        ("raw r_B projection", "raw_rb_projection", pp.paper_palette_role("primary")),
        ("map-mediated", "map_mediated", pp.paper_palette_role("accent")),
    ]
    fig, axes = plt.subplots(1, len(traits), figsize=(6.6 * len(traits), 5.2), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, trait in zip(axes, traits, strict=True):
        x = np.arange(2)  # [prefix-end, averaged-context]
        w = 0.26
        for k, (lab, key, col) in enumerate(constr):
            if key == "supervised":
                a = result["supervised_anchor"][cell][trait]
                vals = [a["prefix_end"], a["averaged_context"]]
                pci = a.get("prefix_end_ci95") or [a["prefix_end"], a["prefix_end"]]
                aci = a.get("averaged_context_ci95") or [
                    a["averaged_context"],
                    a["averaged_context"],
                ]
                los = [max(0.0, a["prefix_end"] - pci[0]), max(0.0, a["averaged_context"] - aci[0])]
                his = [max(0.0, pci[1] - a["prefix_end"]), max(0.0, aci[1] - a["averaged_context"])]
            else:
                d = result["constructions"][key][cell][trait]
                vals = [d["prefix_end"]["r"], d["averaged_context"]["r"]]
                los = [
                    max(0.0, d["prefix_end"]["r"] - d["prefix_end"]["ci95"][0]),
                    max(0.0, d["averaged_context"]["r"] - d["averaged_context"]["ci95"][0]),
                ]
                his = [
                    max(0.0, d["prefix_end"]["ci95"][1] - d["prefix_end"]["r"]),
                    max(0.0, d["averaged_context"]["ci95"][1] - d["averaged_context"]["r"]),
                ]
            ax.bar(x + (k - 1) * w, vals, w, label=lab, color=col)
            ax.errorbar(
                x + (k - 1) * w, vals, yerr=[los, his], fmt="none", ecolor="#333", capsize=3, lw=1.1
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["prefix-end\n(pre-query)", "averaged-context\n(48 q)"])
        ax.set_ylabel("Monitoring correlation r (with judge score)")
        ax.set_title(f"{trait.capitalize()} — three readout constructions")
        ax.axhline(0, color="#bbb", lw=0.8)
        ax.legend(loc="upper left", fontsize=9)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = pp.savefig_paper(fig, "prefixend_monitoring_constructions", dir=FIG_DIR)
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[14])
    ap.add_argument("--headline-layer", type=int, default=14)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--out-dir", type=str, default=str(OUT))
    ap.add_argument(
        "--constructions",
        action="store_true",
        help="round-2: compute raw-r_B + map-mediated constructions beside the "
        "round-1 supervised anchor (requires results.json already present)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.constructions:
        if not args.no_fetch:
            stage_constructions()
        manifest = load_manifest()
        man_by_id = {d["row_id"]: d for d in manifest}
        scores = load_scores()
        result, cpath = run_constructions(manifest, man_by_id, scores, out_dir)
        figs = make_constructions_figure(result)
        result["figures"] = figs
        with open(cpath, "w") as f:
            json.dump(result, f, indent=1)
        # append the construction per-prefix reads beside the supervised points file
        pts_path = out_dir / "per_prefix_points.json"
        if pts_path.exists():
            pts = json.load(open(pts_path))
            pts["construction_reads"] = result["per_prefix"]
            with open(pts_path, "w") as f:
                json.dump(pts, f, indent=1)
        print("WROTE", cpath)
        return 0

    if not args.no_fetch:
        stage(args.layers)

    manifest = load_manifest()
    man_by_id = {d["row_id"]: d for d in manifest}
    scores = load_scores()

    results: dict = {
        "read": "#1092 prefix-end vs averaged-context trait MONITORING (inline free-analysis)",
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_sha(),
        "provenance": {
            "hf_repo": REPO,
            "hf_revision": REV,
            "hf_prefix": BASE,
            "states": (
                "analysis_tensors/summaries/<cell>/{context_end,prefix_end}_L<NN>.npy "
                "(fp16, row-aligned to corpus/manifest.jsonl)"
            ),
            "judge_scores": (
                "p5_judge/scores_shard_*.jsonl (claude-sonnet-4-5-20250929, 5 draws, "
                "temp 1.0, graded 0-100, mean-aggregated)"
            ),
            "substrate": (
                "strata dense_core+battery (149 disjoint prefixes x 48 shared core "
                "queries); group=prefix_id"
            ),
            "readout": (
                "fit_h.ridge_fit_predict (GCV dual-ridge, ambient 3584-dim), group-level "
                f"per-prefix state, grouped {N_FOLDS}-fold held out over prefixes, "
                "target=per-prefix mean judge score; identical construction for "
                "context_end and prefix_end; per-row read = same monitor applied per row"
            ),
            "traits_run": TRAITS,
            "evil_excluded": "0 judged positives on cell_inst_own -> ineligible",
        },
        "design": {
            "n_folds": N_FOLDS,
            "n_boot": N_BOOT,
            "seed": SEED,
            "n_avg_seeds": N_AVG_SEEDS,
            "layers": args.layers,
            "headline_layer": args.headline_layer,
        },
        "cells": {},
    }

    for cell in CELLS:
        results["cells"][cell] = {}
        for layer in args.layers:
            per_layer = []
            for trait in TRAITS:
                r = process_cell_trait(cell, trait, layer, manifest, man_by_id, scores)
                if r is None:
                    continue
                per_layer.append(r)
                d = r["paired_diff_avgctx_minus_prefixend"]
                print(
                    f"[{cell} L{layer} {trait}] per-row={r['reads']['per_row_context']['r']:.3f} "
                    f"avgctx={r['reads']['averaged_context']['r']:.3f} "
                    f"prefixend={r['reads']['prefix_end']['r']:.3f} "
                    f"diff(avgctx-prefixend)={d['point']:.3f} CI{d['ci95']} "
                    f"ceiling={r['monitoring_r_ceiling_from_reliability']:.3f}",
                    flush=True,
                )
            results["cells"][cell][str(layer)] = per_layer

    figs = make_figure(results, args.headline_layer)
    figs.update(make_scatter_figures(results, args.headline_layer))
    results["figures"] = figs

    points_path = write_per_prefix_points(results, args.headline_layer, out_dir)
    print("WROTE", points_path)

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print("WROTE", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
