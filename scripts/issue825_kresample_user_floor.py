#!/usr/bin/env python
"""Issue #825 ``kresample-user`` Phase 3: user-answer diversity floor / ceiling.

Reads the Phase-2 v(u2) capture (orig + K fresh Haiku draws x frozen layers x H,
per reader) and computes, per frozen layer (layer-19 headline), the inter-resample
agreement that CEILINGS any context->user-answer map:

  * ceiling R^2 (bias-corrected, infinite-K): 1 - E_ctx[trvar] / Var_total(Y),
    where trvar = per-context variance across the K fresh draws (ddof=1, summed
    over H) and Var_total = total single-draw variance around the grand mean.
    This is the fraction of single-draw v(u2) variance ANY map could explain.
  * R^2_LODO (Dan's literal ask, finite-K pessimistic): leave-one-draw-out —
    predict each fresh draw by the mean of the other K-1; 1 - SSE_LODO/SSE_mean.
  * #1482-comparable floor: floor_share = E_ctx[trvar] / Var_total(Y) (= 1 -
    ceiling R^2), same normalization family as issue_1482/kresample/floor_summary.
  * exchangeability (#1482 G2): the ORIGINAL Haiku u2 (draw 0) vs the fresh-draw
    distribution — leave-one-out e2 of orig / mean fresh LOO e2 (~1 = exchangeable,
    no recipe drift between the parent generation and the redraws).

Bootstrap CIs over contexts (seed 0), fully numpy-vectorized (no per-context fit
loops). Figure: ceilings vs the committed ridge/MLP user-turn anchors.

Committed anchors (layer 19): parent Haiku-u2 map ridge -1.4272 (instruct) /
-1.4894 (pretrained) [byte-matched to a Haiku floor]; onpolicy Qwen-on-policy map
ridge -0.7689 / -1.8399, MLP 0.329 / 0.078 [named in the brief; NB the anchor
targets are Qwen-generated, not Haiku].
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

FROZEN_DEFAULT = (14, 18, 19, 26)
HEADLINE_LAYER = 19
FIT_SEED = 0
N_BOOTSTRAP = 1000

# Committed anchors for the figure (layer 19).
ANCHORS = {
    "instruct": {
        "parent_haiku_ridge": -1.4272,
        "onpolicy_qwen_ridge": -0.7689,
        "onpolicy_qwen_mlp": 0.3291,
    },
    "pretrained": {
        "parent_haiku_ridge": -1.4894,
        "onpolicy_qwen_ridge": -1.8399,
        "onpolicy_qwen_mlp": 0.0778,
    },
}
# issue_1482 EN-assistant comparison row.
ISSUE1482_EN = {"floor_mean": 0.0935, "floor_share_of_nerr": 0.3281}


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _per_context(Xfresh: np.ndarray):
    """Precompute per-context quantities from the K fresh draws (m, K, H) fp64.

    Returns vbar (m,H), trvar (m,), sumsq (m,), lodo_ss (m,)."""
    K = Xfresh.shape[1]
    vbar = Xfresh.mean(1)
    trvar = Xfresh.var(1, ddof=1).sum(-1)
    sumsq = (Xfresh**2).sum(-1).mean(1)
    loo_mean = (Xfresh.sum(1, keepdims=True) - Xfresh) / (K - 1)
    lodo_ss = ((Xfresh - loo_mean) ** 2).sum(-1).sum(1)
    return vbar, trvar, sumsq, lodo_ss, K


def _aggregate(idx, vbar, trvar, sumsq, lodo_ss, K):
    """Point estimates over a (bootstrap) index set."""
    vb, tv, sq, ls = vbar[idx], trvar[idx], sumsq[idx], lodo_ss[idx]
    gbar = vb.mean(0)
    # mean_k ||X-gbar||^2 per context = sumsq - 2 vbar.gbar + ||gbar||^2
    persq = sq - 2.0 * (vb @ gbar) + float(gbar @ gbar)
    var_total = float(persq.mean())
    ceiling = 1.0 - float(tv.mean()) / var_total
    floor_share = float(tv.mean()) / var_total
    sse_mean_total = float((K * persq).sum())
    lodo_r2 = 1.0 - float(ls.sum()) / sse_mean_total
    return ceiling, floor_share, lodo_r2, var_total, float(tv.mean())


def _boot_ci(vbar, trvar, sumsq, lodo_ss, K, n_boot, seed):
    rng = np.random.default_rng(seed)
    m = len(trvar)
    cs, fs, ls = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, m, size=m)
        c, f, lo, _, _ = _aggregate(idx, vbar, trvar, sumsq, lodo_ss, K)
        cs.append(c)
        fs.append(f)
        ls.append(lo)

    def ci(a):
        a = np.asarray(a)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return ci(cs), ci(fs), ci(ls)


def _exchangeability(X5: np.ndarray):
    """LOO e2 of the original draw (col 0) vs the fresh draws (cols 1:).

    X5: (p, 5, H) fp64. Returns dict with orig/fresh mean LOO e2, ratio, and the
    fraction of contexts where orig LOO-e2 exceeds the fresh median (~0.5 = exch)."""
    K5 = X5.shape[1]
    loo = (X5.sum(1, keepdims=True) - X5) / (K5 - 1)
    e2 = ((X5 - loo) ** 2).sum(-1)  # (p, 5)
    orig_e2 = e2[:, 0]
    fresh_e2 = e2[:, 1:]
    fresh_mean = fresh_e2.mean(1)
    frac_orig_above = float((orig_e2 > np.median(fresh_e2, axis=1)).mean())
    return {
        "n_contexts": int(X5.shape[0]),
        "orig_loo_e2_mean": float(orig_e2.mean()),
        "fresh_loo_e2_mean": float(fresh_mean.mean()),
        "orig_over_fresh_ratio": float(orig_e2.mean() / fresh_mean.mean()),
        "frac_orig_above_fresh_median": frac_orig_above,
    }


def analyze_reader(pt_path: Path, frozen, n_boot, seed):
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    V = d["V"].float().numpy()  # (n, n_draws, len(frozen), H)
    mask = d["valid_mask"].numpy()  # (n, n_draws)
    stored_frozen = list(d["frozen_layers"])
    conv_ids = d["conv_ids"]
    K = V.shape[1] - 1  # fresh draws
    per_layer = {}
    for L in frozen:
        li = stored_frozen.index(L)
        # floor over contexts with all K fresh draws valid
        vf = mask[:, 1:].all(1)
        Xf = V[vf][:, 1:, li, :].astype(np.float64)  # (m, K, H)
        vbar, trvar, sumsq, lodo_ss, Kf = _per_context(Xf)
        ceiling, floor_share, lodo_r2, var_total, trvar_mean = _aggregate(
            np.arange(len(trvar)), vbar, trvar, sumsq, lodo_ss, Kf
        )
        ci_c, ci_f, ci_l = _boot_ci(vbar, trvar, sumsq, lodo_ss, Kf, n_boot, seed)
        # exchangeability over contexts with all 5 draws valid
        v5 = mask.all(1)
        X5 = V[v5][:, :, li, :].astype(np.float64)
        exch = _exchangeability(X5)
        per_layer[str(L)] = {
            "n_contexts_floor": int(vf.sum()),
            "K_fresh": int(Kf),
            "ceiling_r2_infinite_k": ceiling,
            "ceiling_r2_ci": ci_c,
            "r2_lodo_finite_k": lodo_r2,
            "r2_lodo_ci": ci_l,
            "floor_share": floor_share,
            "floor_share_ci": ci_f,
            "trvar_mean": trvar_mean,
            "var_total": var_total,
            "exchangeability": exch,
        }
    return {
        "reader": d.get("reader"),
        "model_id": d.get("model_id"),
        "n_conv": len(conv_ids),
        "frozen_layers": list(frozen),
        "per_layer": per_layer,
    }


def make_figure(results: dict, fig_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception:
        pass

    readers = [r for r in ("instruct", "pretrained") if r in results]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(readers))
    w = 0.28
    ceil = [results[r]["per_layer"][str(HEADLINE_LAYER)]["ceiling_r2_infinite_k"] for r in readers]
    ceil_ci = [results[r]["per_layer"][str(HEADLINE_LAYER)]["ceiling_r2_ci"] for r in readers]
    lodo = [results[r]["per_layer"][str(HEADLINE_LAYER)]["r2_lodo_finite_k"] for r in readers]
    lodo_ci = [results[r]["per_layer"][str(HEADLINE_LAYER)]["r2_lodo_ci"] for r in readers]

    def offs(pts, cis):
        pts = np.array(pts)
        lo = np.array([c[0] for c in cis])
        hi = np.array([c[1] for c in cis])
        return np.vstack([np.maximum(0.0, pts - lo), np.maximum(0.0, hi - pts)])

    ax.bar(
        x - w,
        ceil,
        w,
        yerr=offs(ceil, ceil_ci),
        capsize=3,
        label="ceiling R² (∞-K)",
        color="#4878CF",
    )
    ax.bar(x, lodo, w, yerr=offs(lodo, lodo_ci), capsize=3, label="R²_LODO (K=4)", color="#6ACC64")
    # anchors
    ph = [ANCHORS[r]["parent_haiku_ridge"] for r in readers]
    oq = [ANCHORS[r]["onpolicy_qwen_ridge"] for r in readers]
    mlp = [ANCHORS[r]["onpolicy_qwen_mlp"] for r in readers]
    ax.bar(x + w, ph, w, label="anchor: parent-Haiku ridge R²", color="#D65F5F", alpha=0.85)
    ax.scatter(
        x + w, oq, marker="v", color="black", zorder=5, label="anchor: onpolicy-Qwen ridge R²"
    )
    ax.scatter(
        x + w, mlp, marker="^", color="#8C613C", zorder=5, label="anchor: onpolicy-Qwen MLP R²"
    )
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in readers])
    ax.set_ylabel(f"held-out / resample R² (layer {HEADLINE_LAYER})")
    ax.set_title("User-answer resample ceiling vs fitted context→user-answer map")
    ax.legend(fontsize=7, frameon=False, ncol=1, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-dir", type=Path, default=Path("data/issue_825/kresample_user"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_825/kresample-user"))
    ap.add_argument(
        "--fig", type=Path, default=Path("figures/issue_825/kresample_user_ceiling.png")
    )
    ap.add_argument("--readers", default="instruct,pretrained")
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=FIT_SEED)
    ap.add_argument("--frozen", default=",".join(str(x) for x in FROZEN_DEFAULT))
    args = ap.parse_args()

    frozen = tuple(int(x) for x in args.frozen.split(","))
    readers = [r.strip() for r in args.readers.split(",") if r.strip()]
    results: dict = {}
    for r in readers:
        pt = args.in_dir / f"vu2_{r}.pt"
        if not pt.exists():
            print(f"[floor] skip {r}: {pt} missing")
            continue
        results[r] = analyze_reader(pt, frozen, args.n_boot, args.seed)
        hl = results[r]["per_layer"][str(HEADLINE_LAYER)]
        print(
            f"[floor] {r} L{HEADLINE_LAYER}: ceiling_R2={hl['ceiling_r2_infinite_k']:.4f} "
            f"CI{hl['ceiling_r2_ci']} | R2_LODO={hl['r2_lodo_finite_k']:.4f} CI{hl['r2_lodo_ci']} | "
            f"floor_share={hl['floor_share']:.4f} | exch_ratio={hl['exchangeability']['orig_over_fresh_ratio']:.3f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "followup_label": "kresample-user",
        "metric": "user-answer resample ceiling/floor for the context->user-answer map",
        "headline_layer": HEADLINE_LAYER,
        "definitions": {
            "ceiling_r2_infinite_k": "1 - E_ctx[trvar] / Var_total(Y); best R^2 any map could reach",
            "r2_lodo_finite_k": "leave-one-draw-out (K=4): 1 - SSE_LODO/SSE_mean (finite-K pessimistic)",
            "floor_share": "E_ctx[trvar] / Var_total(Y) = 1 - ceiling; #1482 answer-entropy-floor normalization",
            "exchangeability": "LOO e2 of the streamed/recaptured ORIGINAL Haiku u2 vs fresh draws (~1 = exchangeable)",
            "trvar": "per-context variance across the K fresh draws, ddof=1, summed over hidden dims",
        },
        "anchors_layer19": ANCHORS,
        "issue_1482_en_comparison": ISSUE1482_EN,
        "provenance_caveat": (
            "Fresh draws are HAIKU (claude-haiku-4-5-20251001) resamples, byte-matched to the "
            "PARENT #825 Haiku-u2 user map (ridge -1.4272/-1.4894). The onpolicy-Qwen anchor "
            "(-0.7689/-1.8399) used Qwen-on-policy u2 targets; the ceiling is normalized (a "
            "variance fraction) so it bounds both, but the byte-exact anchor is the parent Haiku map."
        ),
        "n_bootstrap": args.n_boot,
        "seed": args.seed,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    out_json = args.out_dir / "floor_summary.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    if results:
        args.fig.parent.mkdir(parents=True, exist_ok=True)
        make_figure(results, args.fig)
        print(f"[floor] wrote {out_json} + {args.fig}")
    else:
        print(f"[floor] wrote {out_json} (no readers captured)")


if __name__ == "__main__":
    main()
