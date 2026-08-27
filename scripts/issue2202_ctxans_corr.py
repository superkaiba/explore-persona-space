#!/usr/bin/env python3
"""Issue #2202 — correlation between context similarity and answer similarity.

User-chat inline free analysis (2026-08-26, round ctxcorr-r1, 0 GPU-h).

Question. Over the 9,941 held-out conversations, how strongly does "these two
contexts are similar" predict "these two answers are similar"? This is the
population-level version of the per-row context-rank column in the paper's
failure table, and it is what says whether the context-rank reading generalizes
or is an artifact of looking only at failures.

Design. All C(9941, 2) = 49,405,270 unordered pairs, no sampling. Similarity is
computed in four combinations so the answer does not hinge on a convention:

  context space   raw cosine  |  cosine after removing the banked mean mu_C
  answer space    raw cosine  |  whitened cosine (the retrieval convention:
                                 z = L^-1 (x - mu_A) at the banked shrunk
                                 train-answer covariance, lam = 0.1)

Raw activation cosines are dominated by a shared mean direction, so the raw-raw
cell is expected to be far higher than the centered/whitened cell; reporting one
without the other would be a convention artifact rather than a result.

Statistics per cell: Pearson r over all pairs (streamed via exact sums, never a
49M-element materialization), Spearman rho estimated on a seeded random subsample
of pairs (rank-transforming 49M pairs twice is the only step that would need the
full vector in memory), the least-squares slope, and a per-query companion: for
each context, the Spearman between its 9,940 context similarities and its 9,940
answer similarities, summarized across queries. The per-query read is the one
that matches the failure-table question, which is always asked within a query's
own neighbor ordering rather than across the whole pair population.

Blocked over query rows so peak memory stays at one block of the Gram rather
than the full 9,941 x 9,941 pair of matrices.

Reads (already staged, read-only, no download):
  /mnt/eps-data/.../issue2202_avgtgt/cx_holdout_L19.npz          context vectors
  /mnt/eps-data/.../issue2202_freshwhiten/y_holdout_L19.npz      true answers
  /mnt/eps-data/.../issue2202_freshwhiten/whiten_stats.npz       mu_C, mu_A, L

Writes eval_results/issue_2202/ctxans_corr/summary.json and
figures/issue_2202/c3_ctxans_corr.{png,pdf}.
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

STAGED = Path("/mnt/eps-data/thomasjiralerspong/issue2202_freshwhiten")
CX_NPZ = Path("/mnt/eps-data/thomasjiralerspong/issue2202_avgtgt/cx_holdout_L19.npz")
OUT_DIR = PROJECT_ROOT / "eval_results/issue_2202/ctxans_corr"
FIG_DIR = PROJECT_ROOT / "figures/issue_2202"

EXPECTED_N = 9941
H_DIM = 3584
BLOCK = 512
SUBSAMPLE_PAIRS = 5_000_000  # for Spearman over the pair population
SEED = 2202


def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    assert (n > 0).all(), "zero-norm vector"
    return a / n


def _whiten(x: np.ndarray, ell: np.ndarray, mu_a: np.ndarray) -> np.ndarray:
    from scipy.linalg import solve_triangular

    return solve_triangular(ell, (x - mu_a).T, lower=True).T


class PairAccum:
    """Exact Pearson/slope sums over upper-triangle pairs, plus a reservoir
    subsample for Spearman and the scatter figure."""

    def __init__(self, rng: np.random.Generator, keep: int):
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0
        self.keep = keep
        self.rng = rng
        self.buf_x: list[np.ndarray] = []
        self.buf_y: list[np.ndarray] = []
        self.seen = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        x = x.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        self.n += x.size
        self.sx += x.sum()
        self.sy += y.sum()
        self.sxx += (x * x).sum()
        self.syy += (y * y).sum()
        self.sxy += (x * y).sum()
        # uniform-ish subsample: keep a fixed fraction of every block
        frac = min(1.0, self.keep / max(1, self.total_expected))
        if frac >= 1.0:
            m = np.ones(x.size, dtype=bool)
        else:
            m = self.rng.random(x.size) < frac
        if m.any():
            self.buf_x.append(x[m].astype(np.float32))
            self.buf_y.append(y[m].astype(np.float32))

    total_expected = EXPECTED_N * (EXPECTED_N - 1) // 2

    def pearson(self) -> dict:
        n = self.n
        cov = self.sxy / n - (self.sx / n) * (self.sy / n)
        vx = self.sxx / n - (self.sx / n) ** 2
        vy = self.syy / n - (self.sy / n) ** 2
        r = cov / np.sqrt(vx * vy)
        return {
            "n_pairs": int(n),
            "pearson_r": float(r),
            "r_squared": float(r * r),
            "slope_answer_on_context": float(cov / vx),
            "mean_context_sim": float(self.sx / n),
            "mean_answer_sim": float(self.sy / n),
            "sd_context_sim": float(np.sqrt(vx)),
            "sd_answer_sim": float(np.sqrt(vy)),
        }

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(self.buf_x), np.concatenate(self.buf_y)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def run_cell(cn: np.ndarray, an: np.ndarray, rng: np.random.Generator) -> dict:
    """One (context space, answer space) cell: blocked pass over all pairs."""
    from scipy.stats import rankdata

    n = cn.shape[0]
    acc = PairAccum(rng, SUBSAMPLE_PAIRS)
    per_query_rho = np.empty(n, dtype=np.float64)
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        gc = (cn[s:e] @ cn.T).astype(np.float64)
        ga = (an[s:e] @ an.T).astype(np.float64)
        # per-query Spearman over the 9,940 others (self excluded)
        for i in range(e - s):
            gi = s + i
            mask = np.ones(n, dtype=bool)
            mask[gi] = False
            rx = rankdata(gc[i][mask])
            ry = rankdata(ga[i][mask])
            rx -= rx.mean()
            ry -= ry.mean()
            per_query_rho[gi] = (rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry))
        # upper-triangle pairs only (column index strictly greater than row index)
        cols = np.arange(n)
        for i in range(e - s):
            gi = s + i
            sel = cols > gi
            acc.add(gc[i][sel], ga[i][sel])
    out = acc.pearson()
    sx, sy = acc.sample()
    out["spearman_rho"] = spearman(sx, sy)
    out["spearman_n_subsample"] = int(sx.size)
    out["per_query_spearman"] = {
        "median": float(np.median(per_query_rho)),
        "mean": float(per_query_rho.mean()),
        "q25": float(np.percentile(per_query_rho, 25)),
        "q75": float(np.percentile(per_query_rho, 75)),
        "frac_positive": float((per_query_rho > 0).mean()),
    }
    return out, per_query_rho, (sx, sy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cz = np.load(CX_NPZ)
    cx = cz["cx"].astype(np.float32)
    cci = np.asarray(cz["ci"], dtype=np.int64)
    yz = np.load(STAGED / "y_holdout_L19.npz")
    y16 = yz["y16"].astype(np.float32)
    yci = np.asarray(yz["ci"], dtype=np.int64)
    assert cx.shape == (EXPECTED_N, H_DIM), cx.shape
    assert y16.shape == (EXPECTED_N, H_DIM), y16.shape
    assert np.array_equal(cci, yci), "context / answer ci misalign"

    wz = np.load(STAGED / "whiten_stats.npz")
    mu_c = wz["mu_C"].astype(np.float32)
    mu_a = wz["mu_A"].astype(np.float64)
    ell = wz["L"].astype(np.float64)

    ctx_spaces = {
        "context_raw": _unit(cx),
        "context_centered": _unit(cx - mu_c[None, :]),
    }
    ans_spaces = {
        "answer_raw": _unit(y16),
        "answer_whitened": _unit(_whiten(y16.astype(np.float64), ell, mu_a)).astype(np.float32),
    }

    summary = {
        "question": (
            "Across all held-out pairs, how well does context-vector similarity "
            "predict answer-vector similarity?"
        ),
        "n_contexts": EXPECTED_N,
        "n_pairs": EXPECTED_N * (EXPECTED_N - 1) // 2,
        "seed": SEED,
        "note_conventions": (
            "raw cosines are dominated by a shared mean direction; the centered "
            "context space and the whitened answer space (the retrieval "
            "convention) are the de-biased reads"
        ),
        "cells": {},
    }
    samples = {}
    perq = {}
    for cname, cn in ctx_spaces.items():
        for aname, an in ans_spaces.items():
            key = f"{cname}|{aname}"
            print(f"[cell] {key} …", flush=True)
            res, pq, smp = run_cell(cn, an, rng)
            summary["cells"][key] = res
            samples[key] = smp
            perq[key] = pq
            print(
                f"  pearson r={res['pearson_r']:.4f}  spearman={res['spearman_rho']:.4f}  "
                f"per-query median rho={res['per_query_spearman']['median']:.4f}",
                flush=True,
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out_dir / "per_query_spearman.npz",
        ci=cci,
        **{k.replace("|", "__"): v for k, v in perq.items()},
    )
    _figure(samples, perq)
    print(f"\nwrote {out_dir / 'summary.json'}")


def _figure(samples: dict, perq: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_style

        apply_paper_style()
    except Exception:
        pass

    keys = list(samples)
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(3.0 * (len(keys) + 1), 3.0))
    for ax, k in zip(axes, keys):
        sx, sy = samples[k]
        n = min(200_000, sx.size)
        ax.hexbin(sx[:n], sy[:n], gridsize=60, bins="log", cmap="viridis", linewidths=0)
        ax.set_xlabel(k.split("|")[0].replace("_", " ") + " cosine")
        ax.set_ylabel(k.split("|")[1].replace("_", " ") + " cosine")
    ax = axes[-1]
    for k in keys:
        ax.hist(perq[k], bins=60, histtype="step", label=k.replace("|", " / "), linewidth=1.1)
    ax.axvline(0.0, color="0.4", linewidth=0.8, linestyle=":")
    ax.set_xlabel("per-query Spearman")
    ax.set_ylabel("contexts")
    ax.legend(fontsize=5, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"c3_ctxans_corr.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'c3_ctxans_corr.png'}")


if __name__ == "__main__":
    main()
