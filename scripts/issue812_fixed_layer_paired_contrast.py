#!/usr/bin/env python3
"""Issue #812 follow-up — fixed-layer paired unpooled-vs-mean contrast (paired perm null).

The headline null in ``issue812_fit_pooling.py`` bands a max-over-28-layers
Delta-rho statistic with a selection-symmetric shuffle null; Result 4 shows
fixed-layer gaps but without a dedicated paired null. This follow-up gives the
selection-FREE read: at each eligible behavior's FIXED best-mean layer (argmax
over layers of the MEAN operator's stored per-layer rho — read from
``pooling_fit_results.json``, never re-derived), refit the LOCO held-out
predictions for the ``mean`` and ``unpooled`` arms with the SAME recipe as the
main fit (train-only standardization + nested-CV ridge lambda via
``ridge_predict_loco_centered``; rank-10 per-position train-fold PCA for the
unpooled arm), then band Delta-rho = rho_unpooled - rho_mean with a PAIRED
permutation null: each draw permutes the 50 labels ONCE and re-correlates BOTH
arms' STORED held-out predictions against the same permuted labels (the same
cheap exchangeability variant the main fit's shuffle null used — no refits per
draw), taking the per-draw Delta so arm-specific label structure cancels.

The permutation battery is fully VECTORIZED (``.claude/rules/
vectorize-many-cell-fits.md``): all draws' permuted label ranks form one
(n_draws, N) matrix; each arm's null rho vector is one GEMM against the arm's
fixed unit-normalized prediction ranks (every row of the permuted-rank matrix
is a permutation of the same rank vector, so row means/norms are constants).
A batched-vs-``scipy.stats.spearmanr`` identity-permutation equivalence assert
guards the GEMM formula per arm.

Inputs: ``pooling_inputs.pt`` (HF data repo
``superkaiba1/explore-persona-space-data`` @
``issue812_pooling/gpu_leg_inputs/pooling_inputs.pt``, or ``--inputs`` local
path) + the committed graded E0 JSONs + ``pooling_fit_results.json``.
Output: ``eval_results/issue_812/fixed_layer_paired_contrast.json`` (atomic
per-behavior checkpointing; per behavior it also persists the full 1,000-value
paired-null draw array behind the p, plus the 50 per-context held-out
predictions of BOTH arms with their graded targets and context ids) +
``figures/issue_812/fixed_layer_paired_delta.png`` (paper_plots style —
observed Delta point per behavior over its paired-null 2.5-97.5% band;
reader-facing behavior labels; no annotation overlays) +
``figures/issue_812/fixed_layer_perctx_scatter_refusal.png`` (the per-unit data
view: both arms' held-out predictions vs the graded target at refusal's fixed
layer, all 50 context points labeled). CPU-only, 0 GPU-h, minutes on the VM.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Thread caps (#847) + HF_TOKEN must land BEFORE torch is imported (torch freezes
# its pool from OMP_NUM_THREADS at import; issue812_fit_pooling imports torch).
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from issue658_fit_predictors import _rho  # noqa: E402
from issue812_fit_pooling import (  # noqa: E402
    D_IN_PCA,
    _assert_graded_covers_ctx_set,
    _atomic_write_json,
    _git_commit,
    _graded_target,
    _load_graded_e0,
    _now_iso,
    _operator_features,
    _ridge_rho_from_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("issue812.fixed_layer_contrast")

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_INPUTS_PATH = "issue812_pooling/gpu_leg_inputs/pooling_inputs.pt"

# Reader-facing behavior labels for anything rendered on a figure (paper-plots §3.5:
# no snake_case slugs on axes/ticks/legends; slugs stay in JSON keys + sidecars only).
BEH_LABELS = {
    "sycophancy": "sycophancy",
    "refusal": "refusal",
    "harmful_compliance": "harmful compliance",
    "deception": "deception",
    "fact_expression": "fact expression",
    "format_style": "format/style",
    "self_report": "self-report",
    "persona_drift": "persona drift",
}

SCATTER_BEHAVIOR = "refusal"  # headline cell for the per-context data view


def _unit_ranks(v: np.ndarray, name: str) -> np.ndarray:
    """Centered, L2-unit-normalized average ranks of ``v`` (fail-loud on degeneracy)."""
    rc = rankdata(v).astype(np.float64)
    rc -= rc.mean()
    nrm = float(np.linalg.norm(rc))
    if nrm < 1e-12:
        raise RuntimeError(f"degenerate (constant) vector for {name} — cannot rank-correlate")
    return rc / nrm


def _fixed_best_mean_layer(fit_beh: dict, layers_all: list[int], beh: str) -> int:
    """Argmax over layers of the stored MEAN operator per-layer rho (first max on ties)."""
    plr = fit_beh["per_layer_rho"]["mean"]
    candidates = [li for li in layers_all if plr.get(str(li)) is not None]
    if not candidates:
        raise RuntimeError(f"[{beh}] no non-null mean per-layer rho in pooling_fit_results.json")
    return max(candidates, key=lambda li: (plr[str(li)], -li))


def _paired_null_deltas(
    preds_mean: np.ndarray,
    preds_unpooled: np.ndarray,
    y: np.ndarray,
    *,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """(n_draws,) paired-null Delta-rho values, fully batched (one GEMM per arm).

    Each draw permutes y's ranks ONCE; both arms' Spearman rho are recomputed
    against the SAME permuted ranks from the STORED held-out predictions, and the
    per-draw Delta = rho*_unpooled - rho*_mean is returned. Spearman rho is
    Pearson on average ranks; the prediction-rank vectors are FIXED across draws
    and every permuted-label row shares the same mean/norm, so the whole battery
    is ``(n_draws, N) @ (N,)`` per arm. Asserts the GEMM formula reproduces
    ``scipy.stats.spearmanr`` on the identity permutation for BOTH arms.
    """
    n = len(y)
    assert preds_mean.shape == preds_unpooled.shape == y.shape == (n,), (
        preds_mean.shape,
        preds_unpooled.shape,
        y.shape,
    )
    r_y = rankdata(y).astype(np.float64)
    ry_c = r_y - r_y.mean()
    ry_norm = float(np.linalg.norm(ry_c))
    if ry_norm < 1e-12:
        raise RuntimeError("degenerate (constant) graded target — cannot rank-correlate")
    u_m = _unit_ranks(preds_mean, "preds_mean")
    u_u = _unit_ranks(preds_unpooled, "preds_unpooled")

    # Batched-vs-scipy equivalence on the identity permutation (both arms).
    for u, preds, name in ((u_m, preds_mean, "mean"), (u_u, preds_unpooled, "unpooled")):
        gemm_rho = float(ry_c @ u) / ry_norm
        ref = _rho(preds, y)
        if ref is None or abs(gemm_rho - ref) > 1e-8:
            raise RuntimeError(
                f"batched Spearman formula diverges from scipy on identity perm "
                f"({name}: gemm={gemm_rho!r} scipy={ref!r})"
            )

    rng = np.random.default_rng(seed)
    perm_idx = np.argsort(rng.random((n_draws, n)), axis=1)  # (D, n) — no Python loop
    rc_perm = r_y[perm_idx] - r_y.mean()  # (D, n); rows share ry_norm (permutations)
    rho_m_null = (rc_perm @ u_m) / ry_norm
    rho_u_null = (rc_perm @ u_u) / ry_norm
    return rho_u_null - rho_m_null


def _make_figure(results: dict[str, dict], fig_dir: Path) -> None:
    """Observed Delta point per behavior over its paired-null 2.5-97.5% band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    behs = list(results)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, beh in enumerate(behs):
        r = results[beh]
        ax.vlines(
            i,
            r["null_p2_5"],
            r["null_p97_5"],
            color=paper_palette_role("neutral"),
            lw=8,
            alpha=0.55,
            label="paired null 2.5-97.5%" if i == 0 else None,
        )
        ax.scatter(
            [i],
            [r["delta_obs"]],
            color=paper_palette_role("primary"),
            zorder=3,
            s=42,
            label="observed" if i == 0 else None,
        )
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(range(len(behs)))
    ax.set_xticklabels(
        [f"{BEH_LABELS.get(b, b)}\n(layer {results[b]['fixed_layer']})" for b in behs],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel(r"$\Delta\rho$ (unpooled $-$ mean), held-out")
    ax.set_title("Fixed best-mean-layer paired unpooled-vs-mean contrast")
    ax.legend(fontsize=8, loc="lower right")
    savefig_paper(fig, "fixed_layer_paired_delta", dir=str(fig_dir))
    plt.close(fig)


# Reader-facing labels for the #594-battery context ids rendered as scatter point
# labels (paper-plots §3.5: no raw codes on the figure; the raw ids stay in the
# eval_results JSON `ctx_ids`). Derived from the battery naming scheme
# ``f<family>_<kind>_<detail>`` — families per issue594_fig_hero_embeddings_clean.py:
# f1 personas (house + PersonaHub), f2 WildChat chat prefixes, f3 few-shot (ICL)
# worked examples, f4 instruction rephrasings, f5 format wraps, f6 defaults,
# f8 behavior-instruction probes.
_CTX_LABEL_FIXED = {
    "f6_default_template": "bare default",
    "f6_helpful_asst": "helpful assistant",
}
_ICL_KIND_LABELS = {"french": "French", "json": "JSON"}
_BEHAV_KIND_LABELS = {"sycophant": "sycophancy"}


def _ctx_label(cid: str) -> str:
    """Plain-English point label for a battery context id (e.g. ``f1_phub_03`` ->
    ``PersonaHub 03``); fail-loud on an unrecognized family so a new battery id
    never silently ships as a raw code."""
    if cid in _CTX_LABEL_FIXED:
        return _CTX_LABEL_FIXED[cid]
    if cid.startswith("f1_house_"):
        return cid.removeprefix("f1_house_").replace("_", " ") + " persona"
    if cid.startswith("f1_phub_"):
        return "PersonaHub " + cid.removeprefix("f1_phub_")
    if cid.startswith("f2_wc_"):
        kind, num = cid.removeprefix("f2_wc_").rsplit("_", 1)
        return f"WildChat {kind} {num}"
    if cid.startswith("f3_icl_"):
        kind, k = cid.removeprefix("f3_icl_").rsplit("_k", 1)
        kind = _ICL_KIND_LABELS.get(kind, kind.replace("_", " "))
        return f"few-shot {kind} (k={k})"
    if cid.startswith("f4_reph_"):
        return cid.removeprefix("f4_reph_").replace("_", " ") + " rephrase"
    if cid.startswith("f5_fmt_"):
        return cid.removeprefix("f5_fmt_").replace("_", " ") + " format"
    if cid.startswith("f8_behav_"):
        kind = cid.removeprefix("f8_behav_")
        return _BEHAV_KIND_LABELS.get(kind, kind) + "-behavior probe"
    raise RuntimeError(f"unrecognized context-id family for point label: {cid!r}")


def _make_perctx_scatter(results: dict[str, dict], beh: str, fig_dir: Path) -> None:
    """Per-unit data view at one headline cell: both arms' 50 labeled held-out points.

    Two panels (mean pool | unpooled), x = LOCO held-out prediction, y = graded 0-100
    expression score; every context point carries its context-id label so the
    aggregate rho is readable back to individual contexts.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    r = results[beh]
    pc = r["per_context"]
    y = np.asarray(pc["graded_target"], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    panels = (
        ("Mean pool", np.asarray(pc["preds_mean"], dtype=np.float64), r["rho_mean"]),
        (
            "Unpooled (rank-10 PCA)",
            np.asarray(pc["preds_unpooled"], dtype=np.float64),
            r["rho_unpooled"],
        ),
    )
    for ax, (name, preds, rho) in zip(axes, panels, strict=True):
        ax.scatter(preds, y, s=18, color=paper_palette_role("primary"), alpha=0.85, zorder=3)
        for xi, yi, cid in zip(preds, y, pc["ctx_ids"], strict=True):
            ax.text(xi, yi, f" {_ctx_label(cid)}", fontsize=4.2, va="center", ha="left", alpha=0.8)
        ax.set_xlabel("held-out predicted expression score")
        ax.set_title(f"{name} (layer {r['fixed_layer']})", fontsize=10)
        ax.text(
            0.03,
            0.95,
            f"held-out Spearman rho = {rho:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
        )
    axes[0].set_ylabel(f"graded {BEH_LABELS.get(beh, beh)} score (judge, 0-100)")
    savefig_paper(fig, f"fixed_layer_perctx_scatter_{beh}", dir=str(fig_dir))
    plt.close(fig)


def main() -> int:
    """Run the fixed-layer paired contrast over all eligible behaviors; returns 0."""
    ap = argparse.ArgumentParser(description="Issue 812 fixed-layer paired contrast (CPU).")
    ap.add_argument("--inputs", default="", help="local pooling_inputs.pt (default: HF download)")
    ap.add_argument("--fit-results", default="eval_results/issue_812/pooling_fit_results.json")
    ap.add_argument("--graded-highm", default="eval_results/issue_812/graded_e0_highm.json")
    ap.add_argument("--graded-lowm", default="eval_results/issue_812/graded_e0_lowm.json")
    ap.add_argument("--out", default="eval_results/issue_812/fixed_layer_paired_contrast.json")
    ap.add_argument("--fig-dir", default="figures/issue_812")
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=658)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument(
        "--scatter-only",
        action="store_true",
        help="regenerate the per-context scatter from the existing --out JSON "
        "(no refit, no HF download) — e.g. after a point-label change",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.num_threads)

    if args.scatter_only:
        blob = json.loads(Path(args.out).read_text())
        _make_perctx_scatter(blob["results"], SCATTER_BEHAVIOR, Path(args.fig_dir))
        logger.info("wrote %s/fixed_layer_perctx_scatter_%s.png", args.fig_dir, SCATTER_BEHAVIOR)
        return 0

    inputs_path = Path(args.inputs) if args.inputs else None
    if inputs_path is None or not inputs_path.exists():
        from huggingface_hub import hf_hub_download

        logger.info("downloading %s from %s ...", HF_INPUTS_PATH, HF_REPO)
        inputs_path = Path(
            hf_hub_download(repo_id=HF_REPO, filename=HF_INPUTS_PATH, repo_type="dataset")
        )
    inputs = torch.load(inputs_path, weights_only=False)
    for key in ("mean", "max", "attn_fixed", "aligned_pos", "coverage"):
        inputs[key] = inputs[key].numpy()
    all_ctx = list(inputs["ctx_ids"])
    layers_all = [int(x) for x in inputs["layers"]]

    fit_blob = json.loads(Path(args.fit_results).read_text())
    fit_results = fit_blob["results"]
    graded = _load_graded_e0(Path(args.graded_highm), Path(args.graded_lowm))

    # Eligible behaviors: same preflight as the main fit — reliability-excluded
    # behaviors (deception: null sqrt(r_yy) ceiling) are dropped.
    behaviors = [b for b in fit_results if not fit_results[b].get("reliability_excluded")]
    if not behaviors:
        raise RuntimeError("no eligible (non-reliability-excluded) behaviors — cannot fit")
    missing = [b for b in behaviors if b not in graded]
    if missing:
        raise RuntimeError(f"eligible behaviors missing from graded E0: {missing}")
    logger.info("eligible behaviors (%d): %s", len(behaviors), behaviors)

    out_path = Path(args.out)
    meta = {
        "issue": 812,
        "followup": "fixed_layer_paired_contrast",
        "git_commit": _git_commit(),
        "created_utc": _now_iso(),
        "inputs": str(inputs_path),
        "inputs_hf": f"{HF_REPO}/{HF_INPUTS_PATH}",
        "fit_results": str(args.fit_results),
        "n_draws": args.n_draws,
        "seed": args.seed,
        "d_in_pca": D_IN_PCA,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }

    results: dict[str, dict] = {}
    for beh in behaviors:
        # Same fit-side ctx-ID-set defense as the main fit's preflight (no subset here).
        _assert_graded_covers_ctx_set(graded[beh], all_ctx, beh, subset_active=False)
        y, kept_ctx = _graded_target(graded[beh], all_ctx)
        ctx_order = [all_ctx.index(c) for c in kept_ctx]
        fixed_layer = _fixed_best_mean_layer(fit_results[beh], layers_all, beh)

        # Observed: refit both arms' LOCO held-out predictions at the fixed layer
        # (identical recipe + code path as the main sweep; _operator_features'
        # attn_learned arg is unused for these two ops).
        x_mean = _operator_features(inputs, "mean", fixed_layer, ctx_order, None)
        rho_mean, preds_mean = _ridge_rho_from_features(x_mean, y)
        x_unp = _operator_features(inputs, "unpooled", fixed_layer, ctx_order, None)
        rho_unp, preds_unp = _ridge_rho_from_features(x_unp, y)
        if rho_mean is None or rho_unp is None:
            raise RuntimeError(f"[{beh}] degenerate held-out predictions at layer {fixed_layer}")

        # Consistency: the refit must reproduce the stored sweep rho at this cell
        # (same deterministic recipe on the same inputs) — fail loud on divergence.
        for arm, got in (("mean", rho_mean), ("unpooled", rho_unp)):
            stored = fit_results[beh]["per_layer_rho"][arm].get(str(fixed_layer))
            if stored is not None and abs(got - stored) > 1e-6:
                raise RuntimeError(
                    f"[{beh}] refit rho_{arm}={got:.6f} != stored {stored:.6f} at "
                    f"layer {fixed_layer} — recipe drift vs pooling_fit_results.json"
                )

        delta_obs = rho_unp - rho_mean
        delta_null = _paired_null_deltas(
            preds_mean, preds_unp, y, n_draws=args.n_draws, seed=args.seed
        )
        n_extreme = int(np.sum(np.abs(delta_null) >= abs(delta_obs)))
        results[beh] = {
            "fixed_layer": int(fixed_layer),
            "n_contexts": len(kept_ctx),
            "rho_mean": float(rho_mean),
            "rho_unpooled": float(rho_unp),
            "delta_obs": float(delta_obs),
            "null_p2_5": float(np.percentile(delta_null, 2.5)),
            "null_p97_5": float(np.percentile(delta_null, 97.5)),
            "p_two_sided": float((1 + n_extreme) / (1 + args.n_draws)),
            "n_draws": int(args.n_draws),
            # Full per-draw paired-null array behind the p (low-level data contract).
            "null_draws": [round(float(v), 6) for v in delta_null],
            # Per-context data view: both arms' held-out predictions + the target.
            "per_context": {
                "ctx_ids": list(kept_ctx),
                "graded_target": [round(float(v), 4) for v in y],
                "preds_mean": [round(float(v), 4) for v in preds_mean],
                "preds_unpooled": [round(float(v), 4) for v in preds_unp],
            },
        }
        logger.info(
            "[%s] L%d rho_mean=%.3f rho_unpooled=%.3f delta=%.3f p=%.4f",
            beh,
            fixed_layer,
            rho_mean,
            rho_unp,
            delta_obs,
            results[beh]["p_two_sided"],
        )
        # Checkpoint per behavior (atomic) — a later-behavior crash loses nothing.
        _atomic_write_json(out_path, {"meta": meta, "results": results})

    _make_figure(results, Path(args.fig_dir))
    if SCATTER_BEHAVIOR in results:
        _make_perctx_scatter(results, SCATTER_BEHAVIOR, Path(args.fig_dir))
    logger.info(
        "wrote %s + %s/fixed_layer_paired_delta.png + %s/fixed_layer_perctx_scatter_%s.png",
        out_path,
        args.fig_dir,
        args.fig_dir,
        SCATTER_BEHAVIOR,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
