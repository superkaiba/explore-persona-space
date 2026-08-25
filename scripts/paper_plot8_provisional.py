"""Provisional paper Plot 8 (relationship to CoT training) from banked stores, PCA-reduced.

PROVISIONAL, a stand-in until #2546 runs it properly.

Plot 8 asks whether a PRE-CoT-trained model's context vector predicts the POST-CoT-trained
model's answer state. No banked artifact holds that cross-model fit, but both halves are
banked over the same probe pool (identical `probe_pool_hash` across all three stores):

    pre  = Qwen/Qwen2.5-7B-Instruct       -> issue594_context_geometry/.../per_probe/*.pt
    post = open-thoughts/OpenThinker2-7B  -> issue928_.../store/percq_summaries/*.pt

Both context vectors are the TRUE v_C of the paper: the residual state at the last context
token. #594 captures the residual activation at the newline right after the assistant
header; #928's `ctx_last` is the same boundary token under its own naming. So the pre
versus post contrast is definition-matched, not an artifact of two different summaries.
(An earlier draft of this script used prefix-only vectors on both sides, which was
definition-matched but was NOT v_C. Rejected.)

Grain: per-question. Each row is one (context, probe) pair, so n is about 2,000 rather than
the 50 a per-context input would allow. Folds are GROUPED over contexts (whole contexts go
to one fold, never split), so no within-context row leaks across the split. 10 folds rather
than the banked 50-fold leave-one-context-out, purely for wall-clock: a measured pilot put
50-fold at about 170 minutes. Every cell uses the same scheme, so the within-figure
contrasts stay like-for-like.

At n_train of roughly 1,950 a full-width 3,584-dim fit is still estimator-degenerate, so
the input is PCA-reduced, fit on the TRAINING FOLD ONLY. Headline k = 64 (about 30 training
rows per parameter), fixed a priori; a k sweep ships alongside.

Cells: E = pre context -> post answer (the headline), F = post context -> post answer
(within-model reference), H = pre context -> post CoT. G (pre -> pre's own answer) is not
run here because Plot 7 already carries it: `scripts/paper_plot7_provisional.py` fits the
pre model's own context-to-answer map on the same corpus. (An earlier draft of this
docstring dropped G as confounded by generation regime, claiming the banked pre-side
answers were temperature-1.0 samples against a greedy post side. That was wrong:
`scripts/issue658_extract_base_store.py` generates them with `SamplingParams(temperature=0.0)`,
so both sides are greedy. The temperature-1.0 regime belongs to #658's E0 judged-column
wave, a different pass.)

Usage:
    uv run python scripts/paper_plot8_provisional.py --pilot     # measured 1-layer timing
    uv run python scripts/paper_plot8_provisional.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE heavy imports — shared-VM thread caps bind in-process (#847)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from sklearn.utils.extmath import randomized_svd  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.orchestrate.hub import retry_transient  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
PRE_DIR = "issue594_context_geometry/analysis_tensors/per_probe"
POST_DIR = "issue928_cot_decomposition/analysis_tensors/store/percq_summaries"
POST_MANIFEST = f"{POST_DIR}/manifest.json"

PRE_MODEL = "Qwen2.5-7B-Instruct"
POST_MODEL = "OpenThinker2-7B"

# (code, input side, target summary, reader-facing label)
CELLS: tuple[tuple[str, str, str, str], ...] = (
    ("E", "pre", "ans_mean", "pre context to post answer"),
    ("F", "post", "ans_mean", "post context to post answer"),
    ("H", "pre", "cot_mean", "pre context to post CoT"),
)
HEADLINE_K = 64
N_FOLDS = 10
K_SWEEP = (16, 32, 64, 128, 256)
LAMBDAS = np.logspace(-3, 6, 10)
POST_CTX_SUMMARY = "ctx_last"  # the boundary token, matching #594's convention


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default="figures/paper")
    p.add_argument("--results", default="eval_results/issue_2546/plot8_provisional.json")
    p.add_argument(
        "--pilot",
        action="store_true",
        help="Measure one layer of one cell and print the extrapolated wall, then exit.",
    )
    return p.parse_args(argv)


def _dl(path_in_repo: str) -> str:
    return retry_transient(
        lambda: hf_hub_download(DATA_REPO, path_in_repo, repo_type="dataset"),
        what=f"download {path_in_repo}",
    )


def load_rows() -> dict:
    """Assemble row-aligned (context, probe) arrays across the two models."""
    man = json.loads(Path(_dl(POST_MANIFEST)).read_text())
    ctx_ids = sorted(man["context_ids"])

    pre_ctx: list[np.ndarray] = []
    post: dict[str, list[np.ndarray]] = {POST_CTX_SUMMARY: [], "ans_mean": [], "cot_mean": []}
    groups: list[int] = []

    for gi, cid in enumerate(ctx_ids):
        pb = torch.load(_dl(f"{PRE_DIR}/{cid}.pt"), map_location="cpu", weights_only=False)
        qb = torch.load(_dl(f"{POST_DIR}/{cid}.pt"), map_location="cpu", weights_only=False)
        assert pb["probe_pool_hash"] == qb["probe_pool_hash"], cid

        idx = list(qb["probe_indices"])  # which probes survived parsing on the post side
        names = list(qb["summary_names"])
        per_q = qb["per_q"]  # (n_kept, 12, 28, hidden)
        assert per_q.shape[0] == len(idx), (per_q.shape, len(idx))

        pre_ctx.append(pb["tensor"][idx].numpy())  # (n_kept, 28, hidden)
        for want in post:
            post[want].append(per_q[:, names.index(want)].numpy())
        groups.extend([gi] * len(idx))

    out = {
        "ctx_ids": ctx_ids,
        "groups": np.asarray(groups, dtype=int),
        "pre": np.concatenate(pre_ctx),
        **{k: np.concatenate(v) for k, v in post.items()},
    }
    n, n_layers, hidden = out["pre"].shape
    assert out["ans_mean"].shape == (n, n_layers, hidden), out["ans_mean"].shape
    out["n"], out["n_layers"], out["hidden"] = n, n_layers, hidden
    return out


def _ridge_fit(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, float]:
    """Ridge on PCA-reduced inputs, lambda by GCV with a degrees-of-freedom cap.

    The lambda scan runs in the SVD basis of the reduced input, so no lambda ever forms an
    n-by-d residual. With X_c = U S V^T and Z = U^T Y_c, the ridge fit is
    X_c w = U diag(s^2/(s^2+lam)) Z, hence

        ||Y_c - X_c w||^2 = ||Y_c||^2 - sum_j z_j + sum_j (lam/(s_j^2+lam))^2 z_j

    with z_j = ||Z_j||^2. That is exact, not an approximation, and it makes each candidate
    lambda O(k) instead of O(n*d). Only the winning lambda materialises coefficients.
    """
    n, k = xtr.shape
    xm, ym = xtr.mean(0), ytr.mean(0)
    xc, yc = xtr - xm, ytr - ym

    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    z = u.T @ yc  # (k, d) — the one expensive product, computed once
    zsq = np.einsum("ij,ij->i", z, z)
    ss = s**2
    tss = float((yc**2).sum())

    best_lam, best_err = LAMBDAS[-1], np.inf
    for lam in LAMBDAS:
        shrink = lam / (ss + lam)
        dof = float((ss / (ss + lam)).sum())
        if dof >= 0.9 * n:  # dof cap (#1887): never let GCV pick an unregularized fit
            continue
        rss = tss - float(zsq.sum()) + float((shrink**2 * zsq).sum())
        err = rss / (1.0 - dof / n) ** 2
        if err < best_err:
            best_err, best_lam = err, lam

    gram, rhs = xc.T @ xc, xc.T @ yc
    w = np.linalg.solve(gram + best_lam * np.eye(k), rhs)
    return (xte - xm) @ w + ym, best_lam


def fold_ids(groups: np.ndarray, n_folds: int) -> np.ndarray:
    """Assign whole CONTEXTS to folds, deterministically, so no context is split."""
    ctxs = np.unique(groups)
    assign = {c: i % n_folds for i, c in enumerate(ctxs)}
    return np.asarray([assign[g] for g in groups], dtype=int)


def pca_bases(x: np.ndarray, folds: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-fold (mean, components), fit on the TRAINING fold only.

    Computed once per (input side, layer) and reused across every cell sharing that input:
    the basis depends on the input alone, never on the target, so this is a pure saving.
    """
    out = []
    for f in np.unique(folds):
        xtr = x[folds != f]
        mu = xtr.mean(0)
        _, _, vt = randomized_svd(xtr - mu, n_components=k, random_state=0)
        out.append((mu, vt))
    return out


def skill_from_bases(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    bases: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, list[float]]:
    """Grouped held-out skill-over-mean R^2 using precomputed per-fold PCA bases."""
    sse = 0.0
    sst = 0.0
    lams: list[float] = []
    for (mu, comps), f in zip(bases, np.unique(folds), strict=True):
        te = folds == f
        tr = ~te
        xtr = (x[tr] - mu) @ comps.T
        xte = (x[te] - mu) @ comps.T
        ytr, yte = y[tr], y[te]
        pred, lam = _ridge_fit(xtr, ytr, xte)
        lams.append(float(lam))
        sse += float(((yte - pred) ** 2).sum())
        sst += float(((yte - ytr.mean(0)) ** 2).sum())
    return 1.0 - sse / sst, lams


def group_loco_skill(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int, n_folds: int
) -> tuple[float, list[float]]:
    folds = fold_ids(groups, n_folds)
    return skill_from_bases(x, y, folds, pca_bases(x, folds, k))


def identity_bias_skill(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> float:
    """Baseline: predict the target as the input plus a train-fold constant offset."""
    sse = 0.0
    sst = 0.0
    for g in np.unique(folds):
        te = folds == g
        tr = ~te
        bias = (y[tr] - x[tr]).mean(0)
        pred = x[te] + bias
        sse += float(((y[te] - pred) ** 2).sum())
        sst += float(((y[te] - y[tr].mean(0)) ** 2).sum())
    return 1.0 - sse / sst


def _layer(data: dict, key: str, li: int) -> np.ndarray:
    return data[key][:, li].astype(np.float64)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_paper_style()

    data = load_rows()
    n, n_layers, hidden = data["n"], data["n_layers"], data["hidden"]
    n_groups = len(np.unique(data["groups"]))
    print(
        f"rows n={n} over {n_groups} contexts, layers={n_layers}, hidden={hidden}, "
        f"n_train/fold~{n - n // n_groups}, headline k={HEADLINE_K}"
    )

    sides = {"pre": "pre", "post": POST_CTX_SUMMARY}

    if args.pilot:
        t0 = time.time()
        sk, _ = group_loco_skill(
            _layer(data, sides["pre"], n_layers // 2),
            _layer(data, "ans_mean", n_layers // 2),
            data["groups"],
            HEADLINE_K,
            N_FOLDS,
        )
        dt = time.time() - t0
        total = dt * n_layers * 2  # 2 input sides; bases reused across cells
        print(f"PILOT: one layer-cell = {dt:.1f}s (skill {sk:.3f})")
        print(f"PILOT: extrapolated full run = {total / 60:.1f} min for {len(CELLS)} cells")
        return 0

    res: dict = {
        "dv": "held-out skill-over-mean R^2, leave-one-context-out group folds, "
        "inputs PCA-reduced on the training fold only",
        "n_rows": int(n),
        "n_contexts": int(n_groups),
        "headline_k": HEADLINE_K,
        "k_sweep": list(K_SWEEP),
        "pre_model": PRE_MODEL,
        "post_model": POST_MODEL,
        "pre_context_summary": "issue594 boundary-token residual (last context token)",
        "post_context_summary": POST_CTX_SUMMARY,
        "cells": {},
    }

    folds = fold_ids(data["groups"], N_FOLDS)
    res["n_folds"] = int(N_FOLDS)
    res["fold_scheme"] = (
        f"{N_FOLDS} grouped folds over the 50 contexts (whole contexts assigned to folds, "
        "never split). Coarser than the banked 50-fold leave-one-context-out purely for "
        "wall-clock: a measured pilot put LOCO at ~170 min. All cells share this scheme, "
        "so within-figure contrasts are like-for-like."
    )

    # PCA basis depends on the INPUT only, so compute once per (side, layer) and reuse.
    basis_cache: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray]]] = {}

    def bases_for(side_key: str, li: int) -> list[tuple[np.ndarray, np.ndarray]]:
        hit = basis_cache.get((side_key, li))
        if hit is None:
            hit = pca_bases(_layer(data, side_key, li), folds, HEADLINE_K)
            basis_cache[(side_key, li)] = hit
        return hit

    for code, side, tgt, label in CELLS:
        xkey = sides[side]
        curve = []
        t_cell = time.time()
        for li in range(n_layers):
            t_l = time.time()
            sk, lams = skill_from_bases(
                _layer(data, xkey, li), _layer(data, tgt, li), folds, bases_for(xkey, li)
            )
            curve.append({"layer": li, "skill": sk, "lambda_median": float(np.median(lams))})
            print(
                f"  [{code}] layer {li + 1}/{n_layers} skill={sk:.3f} "
                f"({time.time() - t_l:.1f}s, cell {time.time() - t_cell:.0f}s)",
                flush=True,
            )
        best = max(curve, key=lambda r: r["skill"])
        bl = best["layer"]
        # PCA components are ordered, so the k=max basis contains every smaller k as a
        # prefix. One basis serves the whole sweep instead of one basis per k.
        xb, yb = _layer(data, xkey, bl), _layer(data, tgt, bl)
        sweep_bases = pca_bases(xb, folds, max(K_SWEEP))
        sweep = {
            str(k): skill_from_bases(xb, yb, folds, [(mu, comps[:k]) for mu, comps in sweep_bases])[
                0
            ]
            for k in K_SWEEP
        }
        res["cells"][code] = {
            "label": label,
            "input_side": side,
            "target_summary": tgt,
            "layer_curve": curve,
            "best_layer": bl,
            "skill_at_best_layer": best["skill"],
            "k_sweep_at_best_layer": sweep,
            "identity_bias_baseline": identity_bias_skill(
                _layer(data, xkey, bl), _layer(data, tgt, bl), folds
            ),
        }
        print(
            f"  cell {code} ({label}): best layer {bl}, skill {best['skill']:.3f}, "
            f"identity+bias {res['cells'][code]['identity_bias_baseline']:.3f}"
        )

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")

    png = figure(res, Path(args.out_dir))
    print(f"wrote {out}")
    print(f"wrote {png}")
    return 0


def figure(res: dict, out_dir: Path) -> Path:
    colors = paper_palette(len(CELLS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6))

    codes = [c[0] for c in CELLS]
    xs = np.arange(len(codes), dtype=float)
    ax1.bar(
        xs,
        [res["cells"][c]["skill_at_best_layer"] for c in codes],
        width=0.6,
        color=colors,
        zorder=3,
    )
    for xi, c in zip(xs, codes, strict=True):
        ax1.hlines(
            res["cells"][c]["identity_bias_baseline"],
            xi - 0.3,
            xi + 0.3,
            colors="0.25",
            linestyles="dotted",
            zorder=4,
            label="identity + bias baseline" if xi == 0 else None,
        )
    ax1.set_xticks(xs)
    ax1.set_xticklabels(
        [f"{res['cells'][c]['label']}\nlayer {res['cells'][c]['best_layer']}" for c in codes],
        fontsize=8,
    )
    ax1.set_ylabel("held-out skill (R$^2$ over mean)")
    ax1.axhline(0.0, color="0.6", lw=0.8, zorder=2)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")

    for (code, _, _, label), color in zip(CELLS, colors, strict=True):
        curve = res["cells"][code]["layer_curve"]
        ax2.plot(
            [r["layer"] for r in curve],
            [r["skill"] for r in curve],
            color=color,
            lw=1.4,
            label=label,
        )
    ax2.axhline(0.0, color="0.6", lw=0.8)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("held-out skill (R$^2$ over mean)")
    ax2.legend(frameon=False, fontsize=7.5, loc="lower right")

    fig.tight_layout()
    paths = savefig_paper(fig, "plot8_provisional_cross_model", dir=out_dir)
    plt.close(fig)
    return paths.get("png", out_dir / "plot8_provisional_cross_model.png")


if __name__ == "__main__":
    raise SystemExit(main())
