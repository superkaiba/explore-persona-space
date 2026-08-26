#!/usr/bin/env python3
"""Issue #2202 — are retrieval failures explained by CONTEXT-SPACE proximity to the distractor?

User-chat inline free analysis (2026-08-25, 0 GPU-h, existing artifacts only).

Question. The paper's failure table lists 11 residual rank-1 failures of the ridge
map at the operating point, each alongside the context whose answer was retrieved
instead (the DISTRACTOR context). Is a failing context's vector unusually close, in
context space, to its distractor context — relative to (a) the pool-average context
similarity, and (b) the SAME statistic measured on non-failing contexts against
their own top competing answer's context?

Why (b) is the load-bearing control. ``competitor_ci_*`` in the banked
``oppoint_margins.npz`` is the argmax over the pool with the TRUE answer masked
out. It is defined identically for successes and failures, so both groups get a
matched "top distractor" and the fail-vs-success contrast is not selection-on-
outcome. Comparing failures only against a random-pair baseline would confound
"the distractor is nearby" with "every row's top competitor is nearby".

Reads (all already staged, read-only):
  /mnt/eps-data/thomasjiralerspong/issue2202_avgtgt/cx_holdout_L19.npz
      cx (9941, 3584) fp16 context vectors at L19 + ci
  /mnt/eps-data/thomasjiralerspong/issue2202_freshwhiten/whiten_stats.npz
      mu_C, the banked context mean (centering sensitivity only; the banked
      Cholesky L whitens the ANSWER space and is deliberately not applied here)
  eval_results/issue_2202/plot5_redesign/oppoint_margins.npz
      ranks / margins / competitor ids, both regimes
  eval_results/issue_2202/plot5_redesign/oppoint_failures.json
      the 11 operating-point failure rows (the paper table)

Regimes:
  single  whitened-cos + CSLS(k=10), single-draw targets, all 9,941 rows
  oppoint the paper's operating point: draw-averaged targets, the 1,988
          kresample-covered rows, 11 failures (the table)

Per-row reads, all computed from ONE dense cosine Gram (no Python row loop):
  cos_distractor    cos(c_i, c_d(i))
  cos_pool_mean     mean_k!=i cos(c_i, c_k)                 ("on average")
  z_distractor      (cos_distractor - cos_pool_mean) / sd_k cos(c_i, c_k)
  nnrank            rank of d(i) in row i's context kNN ordering (1 = nearest)
  nnrank_pct        nnrank / (n-1)
  plus the mu_C-centered twin of each cosine read, as a sensitivity: raw LLM
  activation cosines are dominated by a shared mean direction.

Hubness control: context-space in-degree of d(i) at k=10 (how many rows hold
d(i) among their 10 nearest contexts), to separate "this pair is a genuine
near-duplicate" from "this distractor sits near everything".

Statistics: Mann-Whitney U + rank-biserial effect size, and a 10,000-draw
permutation test on the difference in group medians (batched, seeded).

Writes eval_results/issue_2202/ctxdist_failures/summary.json (+ per-row rows for
the 11) and figures/issue_2202/c3_ctxdist_failures.{png,pdf}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

CX_NPZ = Path("/mnt/eps-data/thomasjiralerspong/issue2202_avgtgt/cx_holdout_L19.npz")
WHITEN_NPZ = Path("/mnt/eps-data/thomasjiralerspong/issue2202_freshwhiten/whiten_stats.npz")
MARGINS_NPZ = PROJECT_ROOT / "eval_results/issue_2202/plot5_redesign/oppoint_margins.npz"
FAILURES_JSON = PROJECT_ROOT / "eval_results/issue_2202/plot5_redesign/oppoint_failures.json"
OUT_DIR = PROJECT_ROOT / "eval_results/issue_2202/ctxdist_failures"
FIG_DIR = PROJECT_ROOT / "figures/issue_2202"

EXPECTED_N = 9941
H_DIM = 3584
K_HUB = 10
N_PERM = 10_000
SEED = 2202


def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    assert (n > 0).all(), "zero-norm context vector"
    return a / n


def _gram(cx: np.ndarray) -> np.ndarray:
    """Dense cosine Gram, diagonal set to -inf so self never wins a kNN read."""
    g = (_unit(cx) @ _unit(cx).T).astype(np.float32)
    np.fill_diagonal(g, -np.inf)
    return g


def _row_reads(g: np.ndarray, comp_pos: np.ndarray, rows: np.ndarray) -> dict:
    """Vectorized per-row reads off the Gram. rows = row positions to report on."""
    n = g.shape[0]
    finite = np.where(np.isfinite(g), g, np.nan)
    pool_mean = np.nanmean(finite, axis=1)
    pool_sd = np.nanstd(finite, axis=1)
    cos_d = g[rows, comp_pos]
    # neighbour rank of the distractor: how many pool contexts beat it (+1)
    nnrank = (g[rows] > cos_d[:, None]).sum(axis=1) + 1
    return {
        "cos_distractor": cos_d.astype(np.float64),
        "cos_pool_mean": pool_mean[rows].astype(np.float64),
        "z_distractor": ((cos_d - pool_mean[rows]) / pool_sd[rows]).astype(np.float64),
        "nnrank": nnrank.astype(np.int64),
        "nnrank_pct": (nnrank / (n - 1)).astype(np.float64),
    }


def _hub_indegree(g: np.ndarray, k: int) -> np.ndarray:
    """How many rows hold each context among their k nearest contexts."""
    n = g.shape[0]
    topk = np.argpartition(g, n - k, axis=1)[:, n - k :]
    return np.bincount(topk.ravel(), minlength=n).astype(np.int64)


def _mannwhitney(a: np.ndarray, b: np.ndarray) -> dict:
    """U test on a (fail) vs b (success) + rank-biserial; ties handled by midranks."""
    from scipy.stats import mannwhitneyu

    res = mannwhitneyu(a, b, alternative="two-sided")
    u = float(res.statistic)
    rb = 2.0 * u / (len(a) * len(b)) - 1.0  # +1 => fail stochastically larger
    return {"U": u, "p": float(res.pvalue), "rank_biserial": rb}


def _perm_median_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """Batched permutation test on median(fail) - median(success)."""
    obs = float(np.median(a) - np.median(b))
    pooled = np.concatenate([a, b])
    na = len(a)
    idx = np.argsort(rng.random((N_PERM, len(pooled))), axis=1)
    draws = pooled[idx]
    null = np.median(draws[:, :na], axis=1) - np.median(draws[:, na:], axis=1)
    p = float((np.abs(null) >= abs(obs) - 1e-15).mean())
    return {
        "observed_median_diff": obs,
        "perm_p_two_sided": p,
        "null_ci95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
        "n_perm": N_PERM,
    }


def _boot_median(x: np.ndarray, rng: np.random.Generator) -> list:
    if len(x) == 0:
        return [float("nan"), float("nan")]
    d = rng.integers(0, len(x), size=(N_PERM, len(x)))
    m = np.median(x[d], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _describe(x: np.ndarray, rng: np.random.Generator) -> dict:
    return {
        "n": int(len(x)),
        "median": float(np.median(x)) if len(x) else float("nan"),
        "median_ci95": _boot_median(x, rng),
        "mean": float(np.mean(x)) if len(x) else float("nan"),
        "q25": float(np.percentile(x, 25)) if len(x) else float("nan"),
        "q75": float(np.percentile(x, 75)) if len(x) else float("nan"),
    }


def _compare(reads: dict, fail: np.ndarray, rng: np.random.Generator, metrics: list) -> dict:
    out = {}
    for m in metrics:
        a, b = reads[m][fail], reads[m][~fail]
        blk = {"fail": _describe(a, rng), "success": _describe(b, rng)}
        if len(a) >= 2 and len(b) >= 2:
            blk["mannwhitney"] = _mannwhitney(a, b)
            blk["permutation"] = _perm_median_diff(a, b, rng)
        out[m] = blk
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ── load ────────────────────────────────────────────────────────────────
    cz = np.load(CX_NPZ)
    cx = cz["cx"].astype(np.float32)
    cci = np.asarray(cz["ci"], dtype=np.int64)
    assert cx.shape == (EXPECTED_N, H_DIM), cx.shape

    mz = np.load(MARGINS_NPZ)
    ci_full = np.asarray(mz["ci_full"], dtype=np.int64)
    assert np.array_equal(ci_full, cci), "cx / margins ci misalign"
    pos_of = {int(c): p for p, c in enumerate(ci_full.tolist())}

    mu_c = np.load(WHITEN_NPZ)["mu_C"].astype(np.float32)
    assert mu_c.shape == (H_DIM,), mu_c.shape

    # ── the two Grams: raw cosine, and cosine after removing the context mean ─
    grams = {"raw": _gram(cx), "centered": _gram(cx - mu_c[None, :])}
    hub = {k: _hub_indegree(g, K_HUB) for k, g in grams.items()}

    regimes = {
        "single": {
            "rows": np.arange(EXPECTED_N),
            "rank": np.asarray(mz["rank_single"], dtype=np.float64),
            "comp_ci": np.asarray(mz["competitor_ci_single"], dtype=np.int64),
            "margin": np.asarray(mz["margin_single"], dtype=np.float64),
            "desc": "whitened-cos + CSLS(k=10), single-draw targets, all 9,941 rows",
        },
        "oppoint": {
            "rows": np.asarray(mz["pos_covered"], dtype=np.int64),
            "rank": np.asarray(mz["rank_avg"], dtype=np.float64),
            "comp_ci": np.asarray(mz["competitor_ci_avg"], dtype=np.int64),
            "margin": np.asarray(mz["margin_avg"], dtype=np.float64),
            "desc": (
                "operating point: draw-averaged targets, whitened-cos + CSLS(k=10), "
                "the 1,988 kresample-covered rows (the paper's 11-row failure table)"
            ),
        },
    }

    metrics = ["cos_distractor", "z_distractor", "nnrank_pct"]
    summary = {
        "question": (
            "Are failing contexts closer in CONTEXT space to their distractor context "
            "than non-failing contexts are to their own top competitor?"
        ),
        "distractor_definition": (
            "competitor_ci_* from oppoint_margins.npz: argmax over the pool with the TRUE "
            "answer masked out. Defined identically for successes and failures, so the "
            "contrast is matched and not selection-on-outcome."
        ),
        "k_hub": K_HUB,
        "n_perm": N_PERM,
        "seed": SEED,
        "regimes": {},
        "table_rows": [],
    }

    per_regime_reads = {}
    for rname, r in regimes.items():
        rows = r["rows"]
        comp_pos = np.asarray([pos_of[int(c)] for c in r["comp_ci"]], dtype=np.int64)
        fail = r["rank"] > 1.0
        blk = {"description": r["desc"], "n_rows": int(len(rows)), "n_fail": int(fail.sum())}
        reads_by_space = {}
        for space, g in grams.items():
            reads = _row_reads(g, comp_pos, rows)
            reads["hub_indegree_distractor"] = hub[space][comp_pos].astype(np.float64)
            reads_by_space[space] = reads
            blk[space] = _compare(reads, fail, rng, metrics + ["hub_indegree_distractor"])
            # global reference: mean pairwise cosine over the whole pool
            finite = np.where(np.isfinite(g), g, np.nan)
            blk[space]["pool_mean_cosine_all_pairs"] = float(np.nanmean(finite))
        per_regime_reads[rname] = (reads_by_space, rows, comp_pos, fail, r)
        summary["regimes"][rname] = blk

    # ── per-row detail for the 11 table failures ────────────────────────────
    tbl = json.loads(FAILURES_JSON.read_text())
    reads_by_space, rows, comp_pos, fail, r = per_regime_reads["oppoint"]
    row_of_ci = {int(c): i for i, c in enumerate(ci_full[rows].tolist())}
    for rec in tbl["failures"]:
        i = row_of_ci[int(rec["ci"])]
        entry = {
            "ci": int(rec["ci"]),
            "rank_avg": rec["rank_avg"],
            "top1_ci": int(rec["top1_ci"]),
            "competitor_ci": int(ci_full[comp_pos[i]]),
            "score_margin_true_minus_top1": rec["score_margin_true_minus_top1"],
            "labels_1738": rec["labels_1738"],
        }
        for space in grams:
            for m in metrics + ["hub_indegree_distractor"]:
                entry[f"{space}.{m}"] = float(reads_by_space[space][m][i])
        summary["table_rows"].append(entry)

    # median hub in-degree over the whole pool, for reading the hub column
    for space in grams:
        summary.setdefault("hub_reference", {})[space] = {
            "median_indegree_all_contexts": float(np.median(hub[space])),
            "p95_indegree_all_contexts": float(np.percentile(hub[space], 95)),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["regimes"], indent=2)[:4000])

    # ── figure: fail vs success, both regimes, raw-cosine space ─────────────
    _figure(per_regime_reads, ci_full)
    print(f"\nwrote {out_dir / 'summary.json'}")


def _figure(per_regime_reads: dict, ci_full: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_style

        apply_paper_style()
    except Exception:
        pass

    n_pool = len(ci_full)
    ok, bad = "#4C72B0", "#C44E52"
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))

    # (a) cosine to the distractor context, mean-centered space
    # (b) the distractor's neighbour rank among all pool contexts (log scale)
    for ax, (metric, xlabel, logx) in zip(
        axes,
        [
            ("cos_distractor", "cosine to distractor context\n(context mean removed)", False),
            ("nnrank", "distractor's rank among the\n9,940 other contexts", True),
        ],
    ):
        data, labels, colors = [], [], []
        for rname in ("single", "oppoint"):
            reads_by_space, rows, comp_pos, fail, _ = per_regime_reads[rname]
            x = np.asarray(reads_by_space["centered"][metric], dtype=np.float64)
            for mask, tag, col in ((~fail, "retrieved", ok), (fail, "failed", bad)):
                data.append(x[mask])
                labels.append(f"{rname}\n{tag} (n={int(mask.sum())})")
                colors.append(col)
        bp = ax.boxplot(data, vert=False, showfliers=False, widths=0.6, patch_artist=True)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.8)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
        ax.invert_yaxis()

    # (c) share of rows whose distractor is among the row's k nearest contexts
    ax = axes[2]
    ks = (1, 3, 10)
    width = 0.38
    for off, (rname, hatch) in enumerate(zip(("single", "oppoint"), ("", "//"))):
        reads_by_space, rows, comp_pos, fail, _ = per_regime_reads[rname]
        nn = np.asarray(reads_by_space["centered"]["nnrank"])
        for j, (mask, col) in enumerate(((~fail, ok), (fail, bad))):
            share = [100.0 * (nn[mask] <= k).mean() for k in ks]
            ax.bar(
                np.arange(len(ks)) + (j - 0.5) * width + (off - 0.5) * width * 0.42,
                share,
                width * 0.42,
                color=col,
                alpha=0.85,
                hatch=hatch,
                edgecolor="white",
            )
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([f"top-{k}" for k in ks])
    ax.set_xlabel("distractor among the row\x27s\nk nearest contexts")
    ax.set_ylabel("% of rows")
    handles = [
        mpatches.Patch(facecolor=ok, alpha=0.85, label="retrieved"),
        mpatches.Patch(facecolor=bad, alpha=0.85, label="failed"),
        mpatches.Patch(facecolor="0.75", label="all rows"),
        mpatches.Patch(facecolor="0.75", hatch="//", edgecolor="white", label="operating point"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper left", frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"c3_ctxdist_failures.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'c3_ctxdist_failures.png'}")


if __name__ == "__main__":
    main()
