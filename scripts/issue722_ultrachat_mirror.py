#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (→, ², ρ, M̂, ̄) in scientific docstrings + log/print messages.
"""UltraChat-genre MIRROR of the #722 c_C→v0 linearity headline.

Everything in #722 so far is on the **Betley** query genre (store ``data/issue_658/
store``, ``E0_expression.json``). This THIN WRAPPER runs the SAME corrected
analysis on the **UltraChat (g1)** genre so the c_C→v0 linearity conclusion is
genre-paired instead of Betley-only — without editing any committed driver.

It IMPORTS the validated run functions from ``issue722_vectorized_skill`` (committed
at 8ba7c00ab2, NOT edited here) and points them at the UltraChat per-genre store +
the genre-matched c_C (``v0_summaries.pt::cc_last`` — the #658 per-genre
recomputation; the #594 HF c_C is Betley-pinned, so ``_load_genre`` auto-reads the
store's own ``cc_last`` and that path is genre-correct, probe-pool-matched).

Three deliverables, the same metrics as the Betley headline, for comparison:

1. **Linear-ridge skill-over-mean R²** per layer (full-H, LOCO, train-mean-centered
   = the intercept fix), all 28 layers + the plateau L14/18/21, with df_eff + the
   per-fold ridge-skill 95% CI. Reuses ``run_722_skill_over_mean`` (ridge column) +
   ``run_krr_vs_linear``'s ``_ridge_audit`` (df_eff + ridge CI).
   → ``eval_results/issue_722/ultrachat_mirror/skill_over_mean.json``
   HEADLINE: does UltraChat reach the Betley plateau (~0.74-0.80), or is it weaker?

2. **KRR-RBF − linear nonlinear gap** per plateau layer, fold-bootstrap CI. Reuses
   ``run_krr_vs_linear``. → ``ultrachat_mirror/krr_vs_linear.json``
   HEADLINE: is UltraChat also linear at its plateau (gap CI ⊃ 0)?

3. **Behavioral-chain preservation** — per behavior (g1 r_B + g1 E0), Spearman ρ of
   predicted-vs-actual E0 for the DIRECT ``r_Bᵀ v0 → E0`` (ACTUAL v0) vs the
   LINEAR-MAP-MEDIATED ``r_Bᵀ (M̂·c_C) → E0`` (ridge full-H / PCA-64 + MLP PCA-64
   predicted v̂0). Reuses ``run_658_chain`` for the mediated reads + adds the
   genuinely-DIRECT actual-v0 read via ``issue658_fit_predictors._chain_rho`` fed
   the store's ``summaries.mean`` per layer. → ``ultrachat_mirror/behavior_chain.json``
   HEADLINE: does the linear map preserve the behavior direction on UltraChat too?

Plus a Betley-vs-UltraChat comparison JSON + comparison figure. The MLP width-sweep
+ epoch-curve mirror is OUT OF SCOPE here (GPU-worthy + the MLP-overfit conclusion
is genre-agnostic — about n=50, not the genre); noted as a deferred optional.

CPU-only, 0 GPU. Reads the UltraChat per-genre store from the repo-root caches;
never mutates task state.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue658_fit_predictors as i658  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue658_common import dump_json, load_json  # noqa: E402

# REUSE the committed driver's validated run functions + constants (NOT edited).
from issue722_vectorized_skill import (  # noqa: E402
    DATA_ROOT,
    READOUT_BEHAVIORS,
    _load_genre,
    _stack_layers,
    make_658_figure,
    run_658_chain,
    run_722_skill_over_mean,
    run_krr_vs_linear,
)

load_dotenv(str(REPO_ROOT / ".env"))

logger = logging.getLogger("issue722_ultrachat_mirror")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PLATEAU_LAYERS = (14, 18, 21)


# ── Deliverable 3: genuinely-DIRECT r_Bᵀ (actual v0) → E0 chain ρ ──────────────


def direct_chain_rho_actual_v0(data: dict, *, layer_subset: list[int] | None = None) -> dict:
    """DIRECT ``r_Bᵀ v0 → E0`` chain ρ using the ACTUAL v0 (store summaries.mean).

    The mediated reads in ``run_658_chain`` feed a PREDICTED v̂0 (from c_C via ridge
    / MLP) into ``_chain_rho``. The brief's *direct* arm is the upper bound: feed the
    REAL v0 (the v0(C) the predictors are trying to recover) into the SAME
    ``_chain_rho`` selection (best-layer signed Spearman per behavior). Any drop from
    direct → mediated is the chain quality the linear map M̂ loses.
    """
    _C, V = _stack_layers(data)  # V = (N, L, H) actual v0(C) = summaries.mean
    layers = data["layers"]
    li_iter = [
        li for li in range(len(layers)) if layer_subset is None or int(layers[li]) in layer_subset
    ]
    actual_v0_by_layer = {li: V[:, li, :] for li in li_iter}
    return i658._chain_rho(
        actual_v0_by_layer, data["store"], data["e0"], data["rb"], data["ctx_ids"], layers, 0
    )


def build_behavior_chain(
    chain_uc: dict, direct_uc: dict, *, layer_subset: list[int] | None
) -> dict:
    """Assemble the Deliverable-3 behavior_chain.json: DIRECT vs MEDIATED per behavior.

    DIRECT       = ``r_Bᵀ (actual v0) → E0`` (best-layer signed Spearman; upper bound).
    MEDIATED     = ``r_Bᵀ (M̂ c_C) → E0`` at three v̂0 reductions:
                   ridge full-H (the published #658 control), ridge PCA-64, MLP PCA-64.
    The drop direct→mediated is the chain quality the linear map preserves / loses.
    """
    per_behavior = {}
    for col in READOUT_BEHAVIORS:
        direct = direct_uc.get(col)
        mediated_full = chain_uc["ridge_full_chain_rho"].get(col)
        pb = chain_uc["per_behavior"].get(col, {})
        ridge64 = pb.get("ridge_pca64_chain")
        mlp64 = pb.get("mlp_pca64_chain")
        direct_rho = direct["rho"] if direct else None
        mediated_rho = mediated_full["rho"] if mediated_full else None
        per_behavior[col] = {
            "direct_actual_v0": direct,  # {layer, rho} or None
            "mediated_ridge_fullH": mediated_full,  # {layer, rho} or None
            "mediated_ridge_pca64": ridge64,
            "mediated_mlp_pca64": mlp64,
            "preserved_fraction_fullH": (
                None
                if (direct_rho is None or mediated_rho is None or abs(direct_rho) < 1e-9)
                else float(mediated_rho / direct_rho)
            ),
            "direct_minus_mediated_fullH": (
                None
                if (direct_rho is None or mediated_rho is None)
                else float(direct_rho - mediated_rho)
            ),
        }
    return {
        "genre": "ultrachat",
        "metric": "best-layer signed Spearman(r_B^T v0pred, E0) — #658 _chain_rho selection",
        "direct_arm": "r_B^T (actual v0 = store summaries.mean) -> E0",
        "mediated_arm": "r_B^T (M_hat c_C) -> E0 (M_hat = LOCO ridge full-H / PCA-64; MLP PCA-64)",
        "store_dir": chain_uc["store_dir"],
        "e0_path": chain_uc["e0_path"],
        "cc_source": chain_uc["cc_source"],
        "n_contexts": chain_uc["n_contexts"],
        "layers_swept": chain_uc["layers_swept"],
        "layer_subset_smoke": layer_subset,
        "ridge_full_chain_repro_control": chain_uc["ridge_full_chain_repro_control"],
        "per_behavior": per_behavior,
    }


# ── Deliverable 1: skill-over-mean ridge focus + df_eff + per-fold CI ──────────


def build_skill_summary(skill_uc: dict, krr_uc: dict) -> dict:
    """Surface the linear-ridge skill-over-mean focus rows (plateau) + df_eff + CI.

    ``skill_uc`` (from ``run_722_skill_over_mean``) carries ``skill_vs_mean_ridge`` per
    layer (full 28-layer curve). ``krr_uc`` (from ``run_krr_vs_linear``) carries the
    per-layer ``ridge_df_eff`` + ``ridge_skill_ci_lo/hi`` (the per-fold bootstrap CI).
    Joined here so the headline read (plateau L14/18/21) carries skill + df_eff + CI
    in one place, matching the Betley headline's "df_eff + per-fold ridge-skill CI".
    """
    krr_by_layer = {int(r["layer"]): r for r in krr_uc["per_layer"]}
    focus = []
    for r in sorted(skill_uc["per_layer"], key=lambda x: x["layer"]):
        layer = int(r["layer"])
        kr = krr_by_layer.get(layer, {})
        focus.append(
            {
                "layer": layer,
                "skill_vs_mean_ridge": r["skill_vs_mean_ridge"],
                "predict_mean_abs_cos": r.get("predict_mean_abs_cos"),
                "raw_recon_abs_cos": r.get("raw_recon_abs_cos"),
                "ridge_df_eff": kr.get("ridge_df_eff"),
                "ridge_lambda_median": kr.get("ridge_lambda_median"),
                "ridge_skill_ci_lo": kr.get("ridge_skill_ci_lo"),
                "ridge_skill_ci_hi": kr.get("ridge_skill_ci_hi"),
                "is_plateau": layer in PLATEAU_LAYERS,
            }
        )
    ridge_vals = [(r["layer"], r["skill_vs_mean_ridge"]) for r in focus]
    finite = [(layer, v) for layer, v in ridge_vals if v == v]
    best_layer, best_skill = max(finite, key=lambda t: t[1]) if finite else (None, float("nan"))
    plateau_rows = [r for r in focus if r["is_plateau"]]
    plateau_mean = (
        float(np.mean([r["skill_vs_mean_ridge"] for r in plateau_rows])) if plateau_rows else None
    )
    return {
        "genre": "ultrachat",
        "metric": "skill_over_predict_the_mean = 1 - SS_res/SS_tot (held-out R² on centered v0)",
        "c_C_recipe": "C_last (genre-matched cc_last from per-genre store)",
        "n_contexts": skill_uc["n_contexts"],
        "activation_dim": skill_uc["activation_dim"],
        "store_provenance": skill_uc["store_provenance"],
        "plateau_layers": list(PLATEAU_LAYERS),
        "best_ridge_layer": best_layer,
        "best_ridge_skill": best_skill,
        "plateau_mean_ridge_skill": plateau_mean,
        "per_layer": focus,
        "full_skill_over_mean": skill_uc,  # the full per-layer record (all arms)
    }


# ── Betley-vs-UltraChat comparison + figure ───────────────────────────────────


def _betley_skill_ridge_by_layer() -> dict[int, float] | None:
    """Read the committed Betley skill_over_mean.json ridge column (the bar to beat)."""
    p = REPO_ROOT / "eval_results/issue_722/base-skill-over-mean-cC-to-v0/skill_over_mean.json"
    if not p.exists():
        logger.warning("Betley skill_over_mean.json absent at %s — comparison skipped", p)
        return None
    d = load_json(p)
    return {int(r["layer"]): r["skill_vs_mean_ridge"] for r in d.get("per_layer", [])}


def _betley_krr_by_layer() -> dict[int, dict] | None:
    """Read the committed Betley krr_vs_linear.json per-layer gap (the bar to beat)."""
    p = REPO_ROOT / "eval_results/issue_722/base-skill-over-mean-cC-to-v0/krr_vs_linear.json"
    if not p.exists():
        logger.warning("Betley krr_vs_linear.json absent at %s — KRR comparison skipped", p)
        return None
    d = load_json(p)
    return {int(r["layer"]): r for r in d.get("per_layer", [])}


def _betley_chain() -> dict | None:
    p = REPO_ROOT / "eval_results/issue_658/a34a35_mlp_chain.json"
    if not p.exists():
        logger.warning("Betley a34a35_mlp_chain.json absent at %s — chain comparison skipped", p)
        return None
    return load_json(p)


def build_comparison(skill_summary: dict, krr_uc: dict, chain_summary: dict) -> dict:
    """Betley-vs-UltraChat side-by-side on the three deliverables (plateau focus)."""
    bet_skill = _betley_skill_ridge_by_layer()
    bet_krr = _betley_krr_by_layer()
    bet_chain = _betley_chain()
    uc_skill = {r["layer"]: r["skill_vs_mean_ridge"] for r in skill_summary["per_layer"]}
    uc_krr = {int(r["layer"]): r for r in krr_uc["per_layer"]}

    skill_cmp = []
    for layer in PLATEAU_LAYERS:
        skill_cmp.append(
            {
                "layer": layer,
                "betley_ridge_skill": (bet_skill or {}).get(layer),
                "ultrachat_ridge_skill": uc_skill.get(layer),
            }
        )
    krr_cmp = []
    for layer in PLATEAU_LAYERS:
        b = (bet_krr or {}).get(layer, {})
        u = uc_krr.get(layer, {})
        krr_cmp.append(
            {
                "layer": layer,
                "betley_gap": b.get("nonlinear_gap_rbf_minus_linear"),
                "betley_gap_ci95": b.get("gap_ci95"),
                "betley_gap_excludes_zero": b.get("gap_excludes_zero"),
                "ultrachat_gap": u.get("nonlinear_gap_rbf_minus_linear"),
                "ultrachat_gap_ci95": u.get("gap_ci95"),
                "ultrachat_gap_excludes_zero": u.get("gap_excludes_zero"),
            }
        )
    chain_cmp = {}
    for col in READOUT_BEHAVIORS:
        u = chain_summary["per_behavior"].get(col, {})
        u_direct = u.get("direct_actual_v0") or {}
        u_med = u.get("mediated_ridge_fullH") or {}
        b_full = ((bet_chain or {}).get("ridge_full_chain_rho") or {}).get(col) or {}
        chain_cmp[col] = {
            "ultrachat_direct_rho": u_direct.get("rho"),
            "ultrachat_mediated_fullH_rho": u_med.get("rho"),
            "ultrachat_preserved_fraction": u.get("preserved_fraction_fullH"),
            "betley_mediated_fullH_rho": b_full.get("rho"),
        }
    return {
        "deliverable_1_skill_over_mean_ridge_plateau": skill_cmp,
        "deliverable_2_krr_nonlinear_gap_plateau": krr_cmp,
        "deliverable_3_behavior_chain_fullH": chain_cmp,
        "betley_plateau_published_band": "[0.74, 0.80]",
        "ultrachat_plateau_mean_ridge_skill": skill_summary["plateau_mean_ridge_skill"],
    }


def make_comparison_figure(comparison: dict, fig_path: Path) -> None:
    """Cheap Betley-vs-UltraChat comparison figure (3 panels: skill / gap / chain)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel 1: ridge skill-over-mean at plateau layers.
    rows = comparison["deliverable_1_skill_over_mean_ridge_plateau"]
    x = np.arange(len(rows))
    w = 0.38
    bet = [r["betley_ridge_skill"] if r["betley_ridge_skill"] is not None else np.nan for r in rows]
    uc = [
        r["ultrachat_ridge_skill"] if r["ultrachat_ridge_skill"] is not None else np.nan
        for r in rows
    ]
    axes[0].bar(x - w / 2, bet, w, color="#0072B2", label="Betley")
    axes[0].bar(x + w / 2, uc, w, color="#D55E00", label="UltraChat")
    axes[0].axhspan(0.74, 0.80, color="#0072B2", alpha=0.12, label="Betley published band")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"L{r['layer']}" for r in rows])
    axes[0].set_ylabel("linear-ridge skill-over-mean R²")
    axes[0].set_title("D1: c_C→v0 ridge skill (plateau)")
    axes[0].legend(loc="best", fontsize=7)
    axes[0].axhline(0.0, color="0.5", lw=0.8)

    # Panel 2: KRR nonlinear gap (RBF − linear) at plateau layers, with CI.
    rows2 = comparison["deliverable_2_krr_nonlinear_gap_plateau"]
    x2 = np.arange(len(rows2))
    for off, key, ci, color, lab in (
        (-w / 2, "betley_gap", "betley_gap_ci95", "#0072B2", "Betley"),
        (w / 2, "ultrachat_gap", "ultrachat_gap_ci95", "#D55E00", "UltraChat"),
    ):
        g = np.array([r[key] if r[key] is not None else np.nan for r in rows2])
        cis = [r[ci] if r[ci] is not None else [np.nan, np.nan] for r in rows2]
        lo = np.array([c[0] for c in cis])
        hi = np.array([c[1] for c in cis])
        yerr = np.clip(np.vstack([g - lo, hi - g]), 0.0, None)
        axes[1].errorbar(
            x2 + off,
            g,
            yerr=yerr,
            fmt="o",
            ms=4,
            lw=1.0,
            color=color,
            ecolor=color,
            capsize=3,
            label=lab,
        )
    axes[1].axhline(0.0, color="0.4", lw=0.9, ls=":")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([f"L{r['layer']}" for r in rows2])
    axes[1].set_ylabel("nonlinear gap (RBF − linear), 95% CI")
    axes[1].set_title("D2: KRR nonlinearity (plateau)")
    axes[1].legend(loc="best", fontsize=7)

    # Panel 3: chain ρ direct vs mediated (UltraChat) per behavior.
    chain = comparison["deliverable_3_behavior_chain_fullH"]
    behs = list(READOUT_BEHAVIORS)
    x3 = np.arange(len(behs))
    direct = [(chain.get(b, {}).get("ultrachat_direct_rho") or np.nan) for b in behs]
    med = [(chain.get(b, {}).get("ultrachat_mediated_fullH_rho") or np.nan) for b in behs]
    axes[2].bar(x3 - w / 2, direct, w, color="#009E73", label="UC direct (actual v0)")
    axes[2].bar(x3 + w / 2, med, w, color="#D55E00", label="UC mediated (M̂ c_C)")
    axes[2].axhline(0.0, color="0.4", lw=0.9, ls=":")
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels([b.replace("_", "\n") for b in behs], fontsize=7)
    axes[2].set_ylabel("downstream chain ρ (best layer)")
    axes[2].set_title("D3: behavior-chain preservation (UltraChat)")
    axes[2].legend(loc="best", fontsize=7)

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", fig_path)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="UltraChat-genre mirror of the #722 headline.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--ultrachat-store",
        type=Path,
        default=(
            DATA_ROOT
            / "data/issue_658/g1_dl/issue658_theory_assumptions"
            / "store_genre-generalization-ultrachat"
        ),
    )
    parser.add_argument(
        "--ultrachat-e0",
        type=Path,
        default=DATA_ROOT / "eval_results/issue_658/E0_expression_g1.json",
    )
    parser.add_argument("--krr-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--smoke", action="store_true", help="2-layer slice (L0, L18), 200-boot KRR"
    )
    args = parser.parse_args()

    i658.DEVICE = args.device
    threads = args.threads if args.threads > 0 else None
    layer_subset = [0, 18] if args.smoke else None
    krr_boot = min(args.krr_bootstrap, 200) if args.smoke else args.krr_bootstrap

    out_dir = REPO_ROOT / "eval_results/issue_722/ultrachat_mirror"
    fig_dir = REPO_ROOT / "figures/issue_722/ultrachat_mirror"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = reproducibility_metadata({"script": "issue722_ultrachat_mirror"})

    t0 = time.time()
    uc = _load_genre("ultrachat", args.ultrachat_store, args.ultrachat_e0)

    # ── Deliverable 1: skill-over-mean (ridge column = the linearity headline) ──
    skill_uc = run_722_skill_over_mean(
        uc, device=args.device, num_threads=threads, layer_subset=layer_subset
    )

    # ── Deliverable 2: KRR-RBF − linear nonlinear gap (+ df_eff + ridge CI) ──
    krr_uc = run_krr_vs_linear(
        uc,
        width_sweep=None,
        device=args.device,
        existing_json=Path("/nonexistent-force-recompute-from-store.json"),
        n_boot=krr_boot,
        layer_subset=layer_subset,
    )
    krr_uc["metadata"] = meta
    dump_json(krr_uc, out_dir / "krr_vs_linear.json")
    logger.info("wrote %s", out_dir / "krr_vs_linear.json")

    # Deliverable 1 summary needs the KRR df_eff + CI join.
    skill_summary = build_skill_summary(skill_uc, krr_uc)
    skill_summary["metadata"] = meta
    dump_json(skill_summary, out_dir / "skill_over_mean.json")
    logger.info("wrote %s", out_dir / "skill_over_mean.json")

    # ── Deliverable 3: behavior chain (mediated reads + DIRECT actual-v0) ──
    chain_uc = run_658_chain(
        uc, "ultrachat", device=args.device, num_threads=threads, layer_subset=layer_subset
    )
    direct_uc = direct_chain_rho_actual_v0(uc, layer_subset=layer_subset)
    chain_summary = build_behavior_chain(chain_uc, direct_uc, layer_subset=layer_subset)
    chain_summary["metadata"] = meta
    dump_json(chain_summary, out_dir / "behavior_chain.json")
    logger.info("wrote %s", out_dir / "behavior_chain.json")

    # ── Betley-vs-UltraChat comparison + figure ──
    comparison = build_comparison(skill_summary, krr_uc, chain_summary)
    comparison["metadata"] = meta
    dump_json(comparison, out_dir / "betley_vs_ultrachat.json")
    logger.info("wrote %s", out_dir / "betley_vs_ultrachat.json")
    make_comparison_figure(comparison, fig_dir / "betley_vs_ultrachat.png")

    # also write the genre-paired #658 chain bar figure (Betley | UltraChat) when
    # the Betley chain artifact exists — cheap reuse of make_658_figure.
    bet_chain = _betley_chain()
    if bet_chain is not None:
        try:
            make_658_figure(bet_chain, chain_uc, fig_dir / "a34a35_chain_betley_vs_ultrachat.png")
        except Exception as ex:
            logger.warning("genre-paired chain figure skipped: %s: %s", type(ex).__name__, ex)

    wall_m = (time.time() - t0) / 60.0

    # ── console summary ──
    print("\n==== UltraChat mirror — repro control (ridge full-H chain ρ, byte-exact) ====")
    repro = chain_uc["ridge_full_chain_repro_control"]
    print(f"  ridge full-H chain reproduced #658 g1: ok={repro['ok']}")
    for col in READOUT_BEHAVIORS:
        row = repro["rows"].get(col, {})
        print(
            f"    {col:20s} got={row.get('got_rho', float('nan')):+.6f} "
            f"exp={row.get('expected_rho', float('nan')):+.6f} "
            f"Δ={row.get('abs_rho_delta', float('nan')):.2e} match={row.get('match')}"
        )

    print("\n==== D1: UltraChat linear-ridge skill-over-mean R² (plateau focus) ====")
    print(
        f"  best ridge layer L{skill_summary['best_ridge_layer']} "
        f"= {skill_summary['best_ridge_skill']:+.4f}   "
        f"(Betley published plateau band ~[0.74, 0.80])"
    )
    for r in skill_summary["per_layer"]:
        if r["is_plateau"]:
            ci = (
                f"CI=[{r['ridge_skill_ci_lo']:+.3f},{r['ridge_skill_ci_hi']:+.3f}]"
                if r["ridge_skill_ci_lo"] is not None
                else "CI=n/a"
            )
            dfe = (
                f"df_eff={r['ridge_df_eff']:.1f}" if r["ridge_df_eff"] is not None else "df_eff=n/a"
            )
            print(
                f"    L{r['layer']:02d}: ridge_skill={r['skill_vs_mean_ridge']:+.4f}  {ci}  {dfe}"
            )
    print(f"  plateau-mean ridge skill = {skill_summary['plateau_mean_ridge_skill']}")

    print("\n==== D2: UltraChat KRR nonlinear gap (RBF − linear), plateau focus ====")
    for r in krr_uc["per_layer"]:
        if int(r["layer"]) in PLATEAU_LAYERS:
            mark = " *EXCLUDES0*" if r["gap_excludes_zero"] else ""
            print(
                f"    L{r['layer']:02d}: ridge(fullH)={r['skill_vs_mean_ridge_fullH']:+.4f} "
                f"rbf={r['skill_krr_rbf_pca48']:+.4f} "
                f"gap={r['nonlinear_gap_rbf_minus_linear']:+.4f} "
                f"CI=[{r['gap_ci95'][0]:+.4f},{r['gap_ci95'][1]:+.4f}]{mark}"
            )
    sanity = krr_uc["krr_linear_vs_ridge_sanity"]
    print(f"  KRR-linear vs PCA-48 ridge plumbing sanity: ok={sanity['ok']}")

    print("\n==== D3: UltraChat behavior-chain ρ (DIRECT actual-v0 vs MEDIATED M̂ c_C) ====")
    for col in READOUT_BEHAVIORS:
        pb = chain_summary["per_behavior"][col]
        d = pb["direct_actual_v0"] or {}
        m = pb["mediated_ridge_fullH"] or {}
        frac = pb["preserved_fraction_fullH"]
        print(
            f"    {col:20s} direct={d.get('rho', float('nan')):+.3f}@L{d.get('layer', '?')}  "
            f"mediated(fullH)={m.get('rho', float('nan')):+.3f}@L{m.get('layer', '?')}  "
            f"preserved={'n/a' if frac is None else f'{frac:+.2f}'}"
        )

    print(f"\nWALL-TIME (UltraChat mirror, device={args.device}): {wall_m:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
