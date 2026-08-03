"""#1774 P6 figures — hero + exploratory dump (plan §6 figure list).

Every figure function is input-guarded: a missing phase JSON records a skip in
``figures/issue_1774/figures_manifest.json`` instead of crashing the batch (P6
may run before P3/P5 land in a partial round). Rendered via the paper-plots
rcParams (``set_paper_style``) + ``savefig_paper`` (commit-pinned meta
sidecars). Heatmaps use ``layout="constrained"`` — never ``tight_layout`` after
a colorbar (gotchas: mpl refuses the layout-engine switch).

Usage: uv run python scripts/issue1774_figures.py [--out-root D] [--fig-dir D]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps + .env bind BEFORE the heavy imports below (BLAS/torch
# pools freeze at import time; tests/test_shared_vm_thread_caps.py).
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import issue1774_common as c  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

ARM_LABELS = {
    "arm_context": "Full context",
    "arm_prefix_end": "Prefix end",
    "arm_bare_query": "Bare query",
    "arm_query_avg": "Query-averaged prefix",
}
TRAIT_LABELS = {"evil": "Evil", "sycophancy": "Sycophancy", "hallucination": "Hallucination"}


def _steer_cond_label(cond: str) -> str:
    """Plain-English x-label for a steering condition slug (interp-critique r1 item 5).

    ``add_kernel_tail0_neg`` -> ``Add kernel-tail dir 0 (−)``;
    ``leace_rb_evil`` -> ``Erase evil direction``. Unknown slugs pass through.
    """
    import re

    m = re.fullmatch(r"add_(kernel_tail|random|top_sv)(\d+)_(pos|neg)", cond)
    if m:
        family = {"kernel_tail": "kernel-tail", "random": "random", "top_sv": "top-singular"}[
            m.group(1)
        ]
        sign = "+" if m.group(3) == "pos" else "−"
        return f"Add {family} dir {m.group(2)} ({sign})"
    m = re.fullmatch(r"leace_rb_(\w+)", cond)
    if m:
        return f"Erase {m.group(1)} direction"
    if cond == "steer_base":
        return "Unsteered base"
    return cond


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def fig_hero_trait_arm_heatmap(eval_root: Path, fig_dir: Path) -> str | None:
    """HERO: per-trait × per-arm held-out R² heatmap, noise-floor hatching,
    co-kernel ceiling overlay (plan §6)."""
    ceiling = _read(eval_root / "noise_ceiling.json")
    cok = _read(eval_root / "nullspace" / "cokernel_all_L14.json")
    per_arm: dict[str, dict] = {}
    for arm in c.ARMS:
        ch = _read(eval_root / "channels" / f"{arm}_L14.json")
        if ch is not None:
            per_arm[arm] = ch.get("per_trait_heldout_r2", {})
    if not per_arm:
        return "missing channels/*_L14.json"
    traits = list(TRAIT_LABELS)
    arms = [a for a in c.ARMS if a in per_arm]
    M = np.full((len(traits), len(arms)), np.nan)
    for j, arm in enumerate(arms):
        for i, t in enumerate(traits):
            M[i, j] = per_arm[arm].get(t, np.nan)
    gating = {}
    if ceiling is not None:
        gating = ceiling.get("layers", {}).get("L14", {}).get("per_trait_gating", {})
    fig, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
    vmax = max(0.05, float(np.nanmax(M))) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M, cmap="viridis", vmin=min(0.0, float(np.nanmin(M))), vmax=vmax)
    ax.set_xticks(range(len(arms)), [ARM_LABELS[a] for a in arms], rotation=20, ha="right")
    ax.set_yticks(range(len(traits)), [TRAIT_LABELS[t] for t in traits])
    for i, t in enumerate(traits):
        for j, arm in enumerate(arms):
            txt = f"{M[i, j]:.3f}" if np.isfinite(M[i, j]) else "n/a"
            cell = gating.get(arm, {}).get(t, {}) if isinstance(gating.get(arm), dict) else {}
            if cell.get("label") == "noise-limited":
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="white",
                        linewidth=0,
                    )
                )
                txt += "\nnoise-limited"
            if cok is not None:
                frac = (
                    cok.get("arms", {})
                    .get(arm, {})
                    .get("cokernel_fraction", {})
                    .get(t, {})
                    .get("k90")
                )
                if frac is not None:
                    txt += f"\nco-kernel {frac:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="Held-out R² along trait direction")
    ax.set_title("Trait readability by arm (L14; hatch = noise-limited; co-kernel@k90 overlay)")
    savefig_paper(fig, "hero_trait_arm_r2_heatmap", dir=fig_dir)
    plt.close(fig)
    return None


def fig_four_arm_r2(eval_root: Path, fig_dir: Path) -> str | None:
    """Four-arm R² + baselines grouped bars, per-fold points (low-level data)."""
    rows = {}
    for arm in c.ARMS:
        j = _read(eval_root / "fit_battery" / f"{arm}_L14.json")
        if j is not None:
            rows[arm] = j
    if not rows:
        return "missing fit_battery/arm_*_L14.json"
    arms = list(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    x = np.arange(len(arms))
    width = 0.28
    pal = paper_palette(3)
    for k, (key, label) in enumerate(
        [
            ("r2_per_context", "Per-context R²"),
            ("r2_averaged", "Averaged R²"),
            ("r2_identity_bias", "Identity+bias baseline"),
        ]
    ):
        vals = [np.nanmean([f[key] for f in rows[a]["folds"]]) for a in arms]
        ax.bar(x + (k - 1) * width, vals, width, label=label, color=pal[k])
        for i, a in enumerate(arms):
            pts = [f[key] for f in rows[a]["folds"] if np.isfinite(f[key])]
            ax.scatter(np.full(len(pts), x[i] + (k - 1) * width), pts, s=9, color="black", zorder=3)
    for i, a in enumerate(arms):
        acc1 = np.nanmean(
            [f["knn"]["euclidean"]["acc_at_k"]["1"] for f in rows[a]["folds"] if "knn" in f]
        )
        ax.plot(
            [x[i] - 1.5 * width, x[i] + 1.5 * width],
            [acc1, acc1],
            color=paper_palette_role("accent"),
            linestyle="--",
            linewidth=1.2,
            label="kNN acc@1 (euclid)" if i == 0 else None,
        )
    ax.set_xticks(x, [ARM_LABELS[a] for a in arms], rotation=15, ha="right")
    ax.set_ylabel("Held-out R² / retrieval acc@1")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.legend(fontsize=8)
    ax.set_title("Four-arm map skill with baselines (L14; points = folds)")
    savefig_paper(fig, "four_arm_r2_baselines", dir=fig_dir)
    plt.close(fig)
    return None


def fig_spectra(eval_root: Path, fig_dir: Path, layers: list[int]) -> str | None:
    """Predictable-variance spectra per arm×layer with perm-null p95 band + ρ₁²."""
    any_found = False
    fig, axes = plt.subplots(
        1, len(layers) + 1, figsize=(4.0 * (len(layers) + 1), 3.6), layout="constrained"
    )
    pal = paper_palette(len(c.ARMS))
    rho_rows: dict[str, list[float | None]] = {a: [] for a in c.ARMS}
    for li, layer in enumerate(layers):
        ax = axes[li]
        for ai, arm in enumerate(c.ARMS):
            j = _read(eval_root / "channels" / f"{arm}_L{layer}.json")
            if j is None:
                rho_rows[arm].append(None)
                continue
            any_found = True
            obs = np.asarray(j["per_component_r2_obs"])
            p95 = np.asarray(j["per_component_null_p95"])
            kmax = min(200, len(obs))
            comps = np.arange(1, kmax + 1)
            ax.plot(comps, obs[:kmax], color=pal[ai], label=ARM_LABELS[arm], linewidth=1.2)
            ax.plot(comps, p95[:kmax], color=pal[ai], linewidth=0.7, linestyle=":")
            rho_rows[arm].append(j.get("rho1_sq_mean"))
        ax.set_xscale("log")
        ax.set_xlabel("Component (train-fold cross-cov rank)")
        ax.set_ylabel("Out-of-fold component R²")
        ax.set_title(f"L{layer} (dotted = perm-null p95)")
        if li == 0:
            ax.legend(fontsize=7)
    axr = axes[-1]
    for ai, arm in enumerate(c.ARMS):
        vals = rho_rows[arm]
        xs = [layers[i] for i, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if xs:
            axr.plot(xs, ys, marker="o", color=pal[ai], label=ARM_LABELS[arm])
    axr.set_xlabel("Layer")
    axr.set_ylabel("ρ₁² (top canonical corr², held-out)")
    axr.set_title("Linear-readout ceiling per layer")
    if not any_found:
        plt.close(fig)
        return "missing channels/*_L*.json"
    savefig_paper(fig, "predictable_variance_spectra", dir=fig_dir)
    plt.close(fig)
    return None


def fig_cross_arm_angles(eval_root: Path, fig_dir: Path) -> str | None:
    j = _read(eval_root / "channels" / "cross_arm_angles_L14.json")
    if j is None:
        return "missing channels/cross_arm_angles_L14.json"
    pairs = j["pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), layout="constrained")
    pal = paper_palette(len(pairs))
    for side, ax in zip(("in", "out"), axes, strict=True):
        for pi, row in enumerate(pairs):
            ang = row.get(f"{side}_angles_k48")
            if not ang:
                continue
            lbl = " vs ".join(ARM_LABELS[a] for a in row["pair"])
            ax.plot(np.degrees(ang), color=pal[pi], label=lbl, linewidth=1.1)
            nb = row.get("null_band_k48")
            if isinstance(nb, dict) and "p05_deg" in nb:
                ax.axhline(nb["p05_deg"], color="grey", linewidth=0.6, linestyle="--")
        ax.set_xlabel("Principal-angle index (k=48)")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title(f"{'Input' if side == 'in' else 'Output'} subspace angles")
    axes[0].legend(fontsize=6)
    savefig_paper(fig, "cross_arm_principal_angles", dir=fig_dir)
    plt.close(fig)
    return None


def fig_causal_shift(eval_root: Path, fig_dir: Path) -> str | None:
    """Per-condition ||dt1|| points (per-context) + median vs the steered-base
    cross-draw band. Reads the REAL merge_state_shift schema (round 2, M3):
    conditions.{cond_id}.{kind,direction,sign,per_context_dt1,median_dt1,
    p90_dt1,n_contexts} + steer_base_band.{per_context,pooled_p50,pooled_p90}."""
    j = _read(eval_root / "steering" / "state_shift.json")
    if j is None:
        return "missing steering/state_shift.json (P3 lands in a later round)"
    conds = j.get("conditions")
    if not conds:
        return "state_shift.json lacks conditions"
    fig, ax = plt.subplots(figsize=(7.5, 4.0), layout="constrained")
    names = sorted(conds)
    for i, name in enumerate(names):
        row = conds[name]
        vals = np.asarray(list((row.get("per_context_dt1") or {}).values()), dtype=float)
        if vals.size:
            ax.scatter(np.full(vals.size, i), vals, s=8, alpha=0.6, color=paper_palette(1)[0])
        med = row.get("median_dt1")
        if med is not None:
            ax.scatter([i], [float(med)], marker="_", s=300, color="black", zorder=3)
    band = j.get("steer_base_band") or {}
    pool = [float(v) for draws in (band.get("per_context") or {}).values() for v in draws]
    if pool:
        lo, hi = float(np.percentile(pool, 10)), float(np.percentile(pool, 90))
        ax.axhspan(lo, hi, color="grey", alpha=0.25, label="Steered-base cross-draw band (p10–p90)")
        ax.legend(fontsize=8)
    elif band.get("pooled_p90") is not None:
        ax.axhline(float(band["pooled_p90"]), color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(
        range(len(names)),
        [_steer_cond_label(n) for n in names],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("‖Δt1‖ (hook-free re-capture)")
    ax.set_title("Causal state shift per steering condition")
    savefig_paper(fig, "causal_state_shift", dir=fig_dir)
    plt.close(fig)
    return None


def fig_eigen_map(eval_root: Path, fig_dir: Path) -> str | None:
    j = _read(eval_root / "endomorphism" / "context_L14.json")
    if j is None:
        return "missing endomorphism/context_L14.json"
    eig = j.get("eigen", {})
    modes = eig.get("top_modes")
    fig, ax = plt.subplots(figsize=(5.2, 5.0), layout="constrained")
    for r in (0.25, 0.5, 1.0):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="grey", linewidth=0.6))
    if modes:
        re = np.asarray([m["eig_re"] for m in modes])
        im = np.asarray([m["eig_im"] for m in modes])
        res = np.asarray([m["residual"] for m in modes])
        stable = np.asarray([m["stable_5of6_rel_lt_0.5"] for m in modes])
        sc = ax.scatter(
            re[stable],
            im[stable],
            c=res[stable],
            cmap="plasma",
            s=26,
            marker="o",
            label="stable (≥5/6 folds)",
        )
        ax.scatter(
            re[~stable],
            im[~stable],
            c=res[~stable],
            cmap="plasma",
            s=18,
            marker="x",
            label="unstable",
        )
        fig.colorbar(sc, ax=ax, label="Relative eigen residual")
        ax.legend(fontsize=8)
    else:
        ax.text(
            0,
            0,
            f"eigen reads skipped:\n{eig.get('skipped', 'gate FAIL')}",
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(
        f"Context-arm operator eigenvalues (gate pass={j['gate'].get('pass')}; "
        f"trace/d={j.get('trace_over_d', float('nan')):.4f})"
    )
    ax.set_aspect("equal")
    savefig_paper(fig, "eigenvalue_map_context_L14", dir=fig_dir)
    plt.close(fig)
    return None


def fig_cos_gain(eval_root: Path, fig_dir: Path) -> str | None:
    j = _read(eval_root / "endomorphism" / "context_L14.json")
    if j is None or "cos_gain_map" not in j:
        return "missing endomorphism cos_gain_map"
    cg = j["cos_gain_map"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    for name, row in cg.items():
        is_rb = name in TRAIT_LABELS or name.startswith("rb_")
        color = paper_palette_role("accent") if is_rb else paper_palette_role("neutral")
        ax.scatter(row["cos"], row["gain"], s=30 if is_rb else 12, color=color, zorder=3)
        if is_rb:
            ax.annotate(
                TRAIT_LABELS.get(name, name),
                (row["cos"], row["gain"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xlabel("cos(Wv, v)")
    ax.set_ylabel("gain ‖Wv‖ / ‖v‖")
    ax.set_title("Preservation map: trait directions (labeled) vs top singular directions")
    savefig_paper(fig, "cos_gain_preservation", dir=fig_dir)
    plt.close(fig)
    return None


def fig_trait_gain_matrix(eval_root: Path, fig_dir: Path) -> str | None:
    j = _read(eval_root / "endomorphism" / "context_L14.json")
    if j is None or "trait_gain_matrix" not in j:
        return "missing endomorphism trait_gain_matrix"
    tg = j["trait_gain_matrix"]
    G = np.asarray(tg["G"])
    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 3.6), layout="constrained", width_ratios=[1.2, 1.0]
    )
    im = axes[0].imshow(G, cmap="coolwarm", vmin=-np.abs(G).max(), vmax=np.abs(G).max())
    labels = [TRAIT_LABELS.get(t, t) for t in tg["traits"]]
    axes[0].set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axes[0].set_yticks(range(len(labels)), labels)
    for i in range(G.shape[0]):
        for jx in range(G.shape[1]):
            axes[0].text(jx, i, f"{G[i, jx]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0], label="Qᵀ W Q")
    axes[0].set_title("Trait gain matrix G")
    energies = [tg["diag_energy"], tg["offdiag_energy"], tg["out_of_span_energy"]]
    axes[1].bar(["Diagonal", "Off-diagonal", "Out-of-span"], energies, color=paper_palette(3))
    axes[1].set_ylabel("Energy (‖·‖²)")
    axes[1].set_title("Where trait mass goes under W")
    savefig_paper(fig, "trait_gain_matrix", dir=fig_dir)
    plt.close(fig)
    return None


def fig_exploratory_lambda(eval_root: Path, fig_dir: Path) -> str | None:
    rows = {a: _read(eval_root / "fit_battery" / f"{a}_L14.json") for a in c.ARMS}
    rows = {a: j for a, j in rows.items() if j is not None}
    if not rows:
        return "missing fit_battery/arm_*_L14.json"
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), layout="constrained")
    pal = paper_palette(len(rows))
    for ai, (arm, j) in enumerate(rows.items()):
        lams = [f["lam"] for f in j["folds"]]
        dfs = [f["df"] for f in j["folds"]]
        axes[0].scatter(np.full(len(lams), ai), lams, color=pal[ai], s=14)
        axes[1].scatter(np.full(len(dfs), ai), dfs, color=pal[ai], s=14)
    for ax, ylab, logy in (
        (axes[0], "PRESS-selected λ (per fold)", True),
        (axes[1], "df(λ) (per fold)", False),
    ):
        ax.set_xticks(
            range(len(rows)), [ARM_LABELS[a] for a in rows], rotation=15, ha="right", fontsize=8
        )
        ax.set_ylabel(ylab)
        if logy:
            ax.set_yscale("log")
    axes[0].set_title("λ selection per arm (extended grid)")
    axes[1].set_title("Effective dof per arm")
    savefig_paper(fig, "exploratory_lambda_df", dir=fig_dir)
    plt.close(fig)
    return None


def fig_jensen_concentration(fig_dir: Path) -> str | None:
    npz_path = (
        c.PROJECT_ROOT / "eval_results/issue_1774/jensen_refit/per_prefix_jensen_cell_inst_own.npz"
    )
    if not npz_path.exists():
        return f"missing {npz_path.name}"
    with np.load(npz_path) as z:
        if "gap_vectors" not in z.files:
            return (
                "npz lacks gap_vectors — run issue1092_mlp_jensen_natural.py --persist-gap-vectors"
            )
        gap = z["gap_vectors"]
        tvar = z["target_var_per_coord"] if "target_var_per_coord" in z.files else None
    ms = (gap**2).mean(axis=0)
    order = np.argsort(ms)[::-1]
    cum_gap = np.cumsum(ms[order]) / ms.sum()
    fig, ax = plt.subplots(figsize=(6.0, 3.8), layout="constrained")
    ax.plot(
        np.arange(1, len(cum_gap) + 1),
        cum_gap,
        color=paper_palette_role("primary"),
        label="Jensen-gap energy (sorted coords)",
    )
    if tvar is not None:
        cum_var = np.cumsum(np.sort(tvar)[::-1]) / tvar.sum()
        ax.plot(
            np.arange(1, len(cum_var) + 1),
            cum_var,
            color=paper_palette_role("baseline"),
            linestyle="--",
            label="Total t1 variance (trivial reference)",
        )
    ax.axhline(0.5, color="grey", linewidth=0.6)
    ax.set_xlabel("Answer-basis coordinate rank (pca48)")
    ax.set_ylabel("Cumulative share")
    ax.set_title("Jensen-gap direction concentration")
    ax.legend(fontsize=8)
    savefig_paper(fig, "jensen_gap_concentration", dir=fig_dir)
    plt.close(fig)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--layers", default="14,18,19")
    args = ap.parse_args(argv)
    eval_root = c.eval_out(args.out_root)
    fig_dir = Path(args.fig_dir) if args.fig_dir else c.PROJECT_ROOT / "figures/issue_1774"
    fig_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x]
    set_paper_style("blog")

    manifest: dict[str, str] = {}
    jobs = [
        ("hero_trait_arm_r2_heatmap", lambda: fig_hero_trait_arm_heatmap(eval_root, fig_dir)),
        ("four_arm_r2_baselines", lambda: fig_four_arm_r2(eval_root, fig_dir)),
        ("predictable_variance_spectra", lambda: fig_spectra(eval_root, fig_dir, layers)),
        ("cross_arm_principal_angles", lambda: fig_cross_arm_angles(eval_root, fig_dir)),
        ("causal_state_shift", lambda: fig_causal_shift(eval_root, fig_dir)),
        ("eigenvalue_map_context_L14", lambda: fig_eigen_map(eval_root, fig_dir)),
        ("cos_gain_preservation", lambda: fig_cos_gain(eval_root, fig_dir)),
        ("trait_gain_matrix", lambda: fig_trait_gain_matrix(eval_root, fig_dir)),
        ("exploratory_lambda_df", lambda: fig_exploratory_lambda(eval_root, fig_dir)),
        ("jensen_gap_concentration", lambda: fig_jensen_concentration(fig_dir)),
    ]
    for name, fn in jobs:
        skip = fn()
        manifest[name] = "written" if skip is None else f"skipped: {skip}"
        print(f"[p6-figs] {name}: {manifest[name]}", flush=True)
    (fig_dir / "figures_manifest.json").write_text(
        json.dumps({"meta": c.repro_meta(), "figures": manifest}, indent=2)
    )
    n_written = sum(1 for v in manifest.values() if v == "written")
    print(f"[p6-figs] done: {n_written}/{len(manifest)} written -> {fig_dir}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
