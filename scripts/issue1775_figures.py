#!/usr/bin/env python3
"""#1775 P5: hero + exploratory figures from the P0-P4 eval JSONs (VM, off-pod).

Hero: (1) per-arm estimation-ladder bars (ridge/KRR/RFF/MLP per arm x fold
scheme, cluster-bootstrap CI whiskers on the GAIN, ceilings as dashed ticks);
(2) the gap-closure figure (stitch-ridge -> +bilinear(r) curve -> stitch-MLP
band -> full-context ridge, pca48). Exploratory dump: per-fold scatter,
per-seed r-curves, HSIC null histograms with observed markers, interaction-
projection cosine bars vs null bands, lambda-sensitivity table, n50k
contamination histogram. Figures land in figures/issue_1775/ (git) with a
meta.json; /paper-plots rcParams via analysis.paper_plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: caps + .env bind BEFORE the heavy imports (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from issue1775_common import atomic_write_json, eval_dir, out_root, result_meta  # noqa: E402

from explore_persona_space.analysis.paper_plots import set_paper_style  # noqa: E402

RUNG_ORDER = ("ridge", "krr", "rff", "mlp")
ARM_ORDER = ("prefix_end", "query_averaged", "bare_query", "stitch", "context_end")


def fig_dir() -> Path:
    d = out_root() / "figures" / "issue_1775"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    print(f"[figures] {path} absent — figure inputs partial", flush=True)
    return None


def _err_offsets(point: float, ci: list[float]) -> tuple[float, float]:
    """Non-negative errorbar OFFSETS from CI bounds (never bounds, never signed —
    the #547/#1335 xerr/yerr class; clamped element-wise)."""
    lo, hi = float(ci[0]), float(ci[1])
    return max(0.0, point - lo), max(0.0, hi - point)


def _unit_r2(u: dict, basis: str) -> float | None:
    r2 = u.get("r2")
    if isinstance(r2, dict):
        return r2.get(basis)
    if basis == "pca48" and "r2_seed_mean" in u:
        return u["r2_seed_mean"]
    if isinstance(r2, int | float) and u.get("basis") == basis:
        return float(r2)
    return None


ARM_LABELS = {
    "prefix_end": "prefix\nend-state",
    "query_averaged": "prefix-averaged\ncontext state",
    "bare_query": "bare query\nstate",
    "stitch": "prefix + query\nstitch",
    "context_end": "full context\n(linear ref.)",
}
RUNG_LABELS = {
    "ridge": "ridge (linear)",
    "krr": "RBF kernel",
    "rff": "random features",
    "mlp": "MLP",
}


def _primary_units(units: list[dict]) -> list[dict]:
    """Primary-combo per-row units only (L14, instruct cell) — drops the L19 bridge
    + pretrained-cell expansion rows that share arm/scheme/rung keys."""
    return [
        u
        for u in units
        if u.get("grain") == "perrow" and u.get("layer") == 14 and u.get("cell") == "cell_inst_own"
    ]


def _seed_mean(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def hero_ladder(linear: dict, nonlinear: dict, refit: dict | None, out: Path) -> None:
    """2-panel estimation-ladder bars (pca48, per-row grain), novel-prefix +
    novel-query folds. The doubly-novel scheme has NO ladder fits by design and
    gets no panel. The prefix-averaged context-state arm's PRESS ridge is
    degenerate (R2 ~ -8, every fold); its linear bar is the group-respecting
    inner-val lambda refit (query_averaged_refit.json), hatched + legended, and
    its nonlinear whiskers use the refit's gain CIs."""
    basis = "pca48"
    from explore_persona_space.analysis.paper_plots import savefig_paper

    units = _primary_units(
        (linear.get("units", []) if linear else [])
        + (nonlinear.get("units", []) if nonlinear else [])
    )
    gains = (nonlinear or {}).get("gains_vs_ridge", {})
    refit_gains = (refit or {}).get("gains_vs_innerval_ridge", {})
    refit_ridge = (refit or {}).get("pooled_r2", {}).get(basis)

    panels = [
        ("prefix", "Novel-prefix folds", ARM_ORDER),
        ("query", "Novel-query folds", ("bare_query", "stitch")),
    ]
    palette = {"ridge": "#8888aa", "krr": "#3366cc", "rff": "#66aadd", "mlp": "#cc3333"}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), squeeze=False)

    def _rung_value(arm: str, scheme: str, rung: str) -> float | None:
        cand = [u for u in units if u["arm"] == arm and u["scheme"] == scheme and u["rung"] == rung]
        if rung == "ridge":
            cand = [u for u in cand if u.get("engine") == "press"]
        if not cand:
            return None
        vals = [v for v in (_unit_r2(u, basis) for u in cand) if v is not None]
        return _seed_mean(vals)

    def _gain_ci(arm: str, scheme: str, rung: str) -> tuple[float, list[float]] | None:
        src = refit_gains if (arm == "query_averaged" and scheme == "prefix") else gains
        boot_key = "bootstrap_vs_innerval_ridge" if src is refit_gains else None
        for suffix in ("", "|s0", "|s1", "|s2"):
            g = src.get(f"{arm}|perrow|{scheme}|{rung}|{basis}{suffix}")
            if g:
                b = g.get(boot_key, g) if boot_key else g
                return float(b["delta_r2"]), [float(c) for c in b["ci95_cluster"]]
        return None

    for si, (scheme, title, arms) in enumerate(panels):
        ax = axes[0][si]
        arms = [a for a in arms if any(u["arm"] == a and u["scheme"] == scheme for u in units)]
        width = 0.19
        for ri, rung in enumerate(RUNG_ORDER):
            xs, vals, errs_lo, errs_hi, hatches = [], [], [], [], []
            for ai, arm in enumerate(arms):
                degenerate = arm == "query_averaged" and scheme == "prefix" and rung == "ridge"
                v = refit_ridge if degenerate else _rung_value(arm, scheme, rung)
                if v is None:
                    continue
                xs.append(ai + (ri - 1.5) * width)
                vals.append(v)
                hatches.append("//" if degenerate else "")
                g = _gain_ci(arm, scheme, rung) if rung != "ridge" else None
                if g:
                    lo, hi = _err_offsets(g[0], g[1])
                    errs_lo.append(lo)
                    errs_hi.append(hi)
                else:
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
            if not xs:
                continue
            bars = ax.bar(
                xs,
                vals,
                width=width,
                color=palette[rung],
                label=RUNG_LABELS[rung] if si == 0 else None,
            )
            for b, h in zip(bars, hatches):
                if h:
                    b.set_hatch(h)
                    b.set_edgecolor("white")
            ax.errorbar(
                xs, vals, yerr=[errs_lo, errs_hi], fmt="none", ecolor="black", capsize=2, lw=0.9
            )
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ARM_LABELS[a] for a in arms], fontsize=8)
        ax.set_ylim(-0.42, 1.0)
        ax.set_ylabel("held-out R2 (48-PC target space)" if si == 0 else "")
        ax.set_title(title, fontsize=10)
    hatch_proxy = plt.Rectangle(
        (0, 0), 1, 1, facecolor=palette["ridge"], hatch="//", edgecolor="white"
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(hatch_proxy)
    labels.append("ridge, inner-val lambda refit\n(PRESS selection degenerate)")
    axes[0][0].legend(handles, labels, fontsize=6.5, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    savefig_paper(fig, "hero_ladder_bars", dir=out)
    plt.close(fig)


def hero_gap_closure(linear: dict, bilinear: dict, out: Path) -> None:
    if not bilinear:
        return
    sch = bilinear.get("schemes", {}).get("prefix", {})
    curve = sch.get("outer_r2_curve_EXPLORATORY", {})
    if not curve:
        return
    gap = sch.get("interaction_gap_fraction") or {}
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    rs = sorted(int(r) for r in curve)
    ax.plot(
        [max(r, 0.5) for r in rs],
        [curve[str(r)] for r in rs],
        marker="o",
        label="stitch + bilinear(r) (outer test, EXPLORATORY curve)",
    )
    if gap.get("r2_stitch_ridge") is not None:
        ax.axhline(gap["r2_stitch_ridge"], ls=":", color="gray", label="stitch ridge")
    if gap.get("r2_context_ridge") is not None:
        ax.axhline(gap["r2_context_ridge"], ls="--", color="black", label="full-context ridge")
    dbm = sch.get("delta_beyond_mlp_minus_bilinear")
    if isinstance(dbm, dict) and gap.get("r2_bilinear") is not None:
        mlp_r2 = gap["r2_bilinear"] + dbm["delta_r2"]
        lo, hi = dbm["ci95_cluster"]
        ax.axhspan(
            gap["r2_bilinear"] + lo,
            gap["r2_bilinear"] + hi,
            alpha=0.15,
            color="red",
            label="stitch-MLP (CI band)",
        )
        ax.axhline(mlp_r2, ls="-.", color="red", lw=0.8)
    rstar = sch.get("r_star_inner_val")
    if rstar:
        ax.axvline(rstar, color="green", ls=":", lw=0.8, label=f"r* = {rstar} (inner val)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("interaction rank r (log2; r=0 plotted at 0.5)")
    ax.set_ylabel("held-out R2 (pca48, novel-prefix folds)")
    ax.legend(fontsize=6)
    ax.set_title("#1775 gap closure: additive stitch -> +rank-r bilinear -> MLP -> context")
    fig.tight_layout()
    fig.savefig(out / "hero_gap_closure.png", dpi=200)
    plt.close(fig)


def perfold_headline_fig(pf: dict | None, out: Path) -> None:
    """Per-fold companion behind the pooled gap-closure headline (Lens 11).

    One dot per fold x headline level (novel-prefix + novel-query panels),
    read from eval_results/issue_1775/bilinear/perfold_headline.json
    (issue1775_perfold_headline.py). Colors match the gap-closure hero:
    gray = stitch ridge, black = full-context ridge, red = stitch-MLP.
    """
    if not pf or "schemes" not in pf:
        return
    from explore_persona_space.analysis.paper_plots import savefig_paper

    series = [
        ("stitch_press_ridge", "additive stitch ridge (PRESS)", "#888888", "o"),
        ("bilinear_r0", "bilinear r = 0 refit", "#9467bd", "D"),
        ("bilinear_rstar", "bilinear r = r* (32 prefix / 16 query)", "#1f77b4", "s"),
        ("stitch_mlp_ensemble", "stitch-MLP ensemble (mean of 3 seeds)", "#cc3333", "^"),
        ("context_press_ridge", "full-context ridge (PRESS)", "black", "v"),
    ]
    panels = [("prefix", "Novel-prefix folds"), ("query", "Novel-query folds")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), squeeze=False)
    for si, (scheme, title) in enumerate(panels):
        ax = axes[0][si]
        levels = pf["schemes"].get(scheme, {}).get("levels", {})
        rstar = pf.get("r_star_by_scheme", {}).get(scheme)
        for li, (key, label, color, marker) in enumerate(series):
            if scheme == "query" and key == "stitch_press_ridge":
                # PRESS collapses under shared-query lambda selection (R2 0.26-0.45);
                # that read has its own per-fold figure (expl_per_fold_r2).
                continue
            k = f"bilinear_r{rstar}" if key == "bilinear_rstar" else key
            vals = (levels.get(k) or {}).get("per_fold")
            if not vals:
                continue
            xs = np.arange(len(vals)) + (li - 2) * 0.12
            ax.plot(
                xs,
                vals,
                marker,
                ms=5,
                color=color,
                label=label if si == 0 else None,
            )
        ax.set_xticks(range(6))
        ax.set_xlabel("fold index")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("per-fold held-out R2\n(48 answer PCs)" if si == 0 else "")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, loc="lower center", ncol=3, framealpha=0.9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    savefig_paper(fig, "hero_gap_closure_perfold", dir=out)
    plt.close(fig)


def projection_cosines(projections: dict | None, out: Path) -> None:
    """Observed max-abs-cosine of the fitted interaction components against the
    train-fold answer PCs and the 3-trait persona-vector dictionary; the two
    permutation-null p-values ride the caption/body (no in-figure annotations)."""
    if not projections or "prefix" not in projections:
        return
    from explore_persona_space.analysis.paper_plots import savefig_paper

    obs = projections["prefix"]["observed"]
    reads = [
        ("w_vs_pc_abscos_max", "output directions\nvs answer PCs"),
        ("w_vs_rb_abscos_max", "output directions\nvs trait dictionary"),
        ("u_vs_rb_abscos_max", "prefix-side inputs\nvs trait dictionary"),
        ("v_vs_rb_abscos_max", "query-side inputs\nvs trait dictionary"),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    xs = np.arange(len(reads))
    ax.bar(xs, [obs[k] for k, _ in reads], color="#446688", width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in reads], fontsize=8)
    ax.set_ylabel("max abs cosine (pooled 576 terms)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Bilinear interaction components: strongest alignment per read", fontsize=10)
    fig.tight_layout()
    savefig_paper(fig, "expl_projection_cosines", dir=out)
    plt.close(fig)


def exploratory(linear, nonlinear, detection, bilinear, fold_check, out: Path, refit=None) -> None:
    from explore_persona_space.analysis.paper_plots import savefig_paper

    # per-fold R2 scatter — the low-level per-unit view behind the ladder bars
    if linear:
        units = [
            u
            for u in _primary_units(linear.get("units", []))
            if u.get("basis") == "pca48" and "r2_folds" in u and u.get("engine") == "press"
        ]
        scheme_label = {"prefix": "novel-prefix", "query": "novel-query", "doubly": "doubly-novel"}
        groups = [
            (u, f"{ARM_LABELS[u['arm']]}\n({scheme_label[u['scheme']]})".replace("\n(", " ("))
            for u in units
        ]
        fig, ax = plt.subplots(figsize=(7.8, 3.6))
        for gi, (u, lbl) in enumerate(groups):
            ys = u["r2_folds"]
            ax.plot(np.full(len(ys), gi), ys, "o", ms=4, color="#8888aa")
        rf = (refit or {}).get("per_fold")
        if rf:
            qa_idx = [
                gi
                for gi, (u, _) in enumerate(groups)
                if u["arm"] == "query_averaged" and u["scheme"] == "prefix"
            ]
            if qa_idx:
                ys = [f["fold_r2"]["pca48"] for f in rf]
                ax.plot(
                    np.full(len(ys), qa_idx[0]),
                    ys,
                    "o",
                    ms=5,
                    mfc="none",
                    mec="#cc3333",
                    mew=1.2,
                    label="inner-val lambda refit (per fold)",
                )
                ax.legend(fontsize=7, loc="center right")
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(
            [lbl.replace("\n", " ") for _, lbl in groups],
            fontsize=6.5,
            rotation=40,
            ha="right",
        )
        ax.set_ylabel("per-fold held-out R2\n(48-PC target space)")
        ax.set_title("Linear (PRESS ridge) fits: all 6 folds per arm", fontsize=10)
        fig.tight_layout()
        savefig_paper(fig, "expl_per_fold_r2", dir=out)
        plt.close(fig)
        # lambda-sensitivity table (rendered as figure text for the dump)
        rows = []
        for u in linear.get("units", []):
            ls = u.get("lambda_sensitivity")
            if ls and u.get("basis") == "pca48":
                rows.append(
                    f"{u['arm']}/{u['scheme']}: R2={u['r2']:.4f} "
                    f"x10={ls['lam_x10']:.4f} /10={ls['lam_div10']:.4f}"
                )
        if rows:
            fig = plt.figure(figsize=(6, 0.3 + 0.22 * len(rows)))
            fig.text(0.02, 0.98, "\n".join(rows), va="top", family="monospace", fontsize=7)
            fig.savefig(out / "expl_lambda_sensitivity.png", dpi=160)
            plt.close(fig)
    if bilinear:
        for scheme, sch in bilinear.get("schemes", {}).items():
            curve = sch.get("outer_r2_curve_EXPLORATORY", {})
            if not curve:
                continue
            fig, ax = plt.subplots(figsize=(4.6, 3.0))
            rs = sorted(int(r) for r in curve)
            ax.plot([max(r, 0.5) for r in rs], [curve[str(r)] for r in rs], marker="o")
            ax.set_xscale("log", base=2)
            ax.set_title(f"bilinear r-curve ({scheme}; exploratory)")
            ax.set_xlabel("r")
            ax.set_ylabel("held-out R2")
            fig.tight_layout()
            savefig_paper(fig, f"expl_rcurve_{scheme}", dir=out)
            plt.close(fig)
    if detection:
        fam = detection.get("holm_adjusted_p", {})
        if fam:
            plain = {
                "prefix_end": "prefix end",
                "query_averaged": "prefix-averaged",
                "bare_query": "bare query",
                "context_end": "full context",
                "stitch": "stitch",
                "prefix_block": "prefix block",
                "query_block": "query block",
                "within_prefix_derangement": "within-prefix shuffle",
                "hsic": "HSIC",
                "dcor": "dCor",
            }
            fig, ax = plt.subplots(figsize=(7.6, 3.4))
            names = sorted(fam)
            labels = [" / ".join(plain.get(t, t) for t in n.split("|")) for n in names]
            ax.bar(range(len(names)), [fam[n] for n in names], color="#446688")
            ax.axhline(0.05, color="red", ls="--", lw=0.8)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(labels, rotation=80, fontsize=5.5)
            ax.set_ylabel("Holm-adjusted p")
            ax.set_title(
                "residual dependence (30-test family: arm / permutation scheme / statistic)"
            )
            fig.tight_layout()
            savefig_paper(fig, "expl_detection_holm", dir=out)
            plt.close(fig)
    if fold_check:
        hist = (fold_check.get("battery") or {}).get("per_target_max_est_jaccard_hist")
        if hist:
            fig, ax = plt.subplots(figsize=(4.6, 3.0))
            edges = hist["edges"]
            ax.bar(edges[:-1], hist["counts"], width=np.diff(edges), align="edge")
            ax.axvline(0.8, color="red", ls="--", lw=0.8)
            ax.set_yscale("log")
            ax.set_xlabel("per-target max MinHash-estimated Jaccard")
            ax.set_ylabel("targets (log)")
            ax.set_title("n50k train-vs-target contamination")
            fig.tight_layout()
            savefig_paper(fig, "expl_n50k_contamination", dir=out)
            plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="#1775 P5 figures")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--only-perfold",
        action="store_true",
        help="render ONLY the per-fold headline companion (no other figure re-rendered)",
    )
    args = ap.parse_args()
    set_paper_style()
    out = fig_dir()
    if args.only_perfold:
        perfold = _load(eval_dir("bilinear") / "perfold_headline.json")
        perfold_headline_fig(perfold, out)
        print(f"[figures] wrote hero_gap_closure_perfold to {out}", flush=True)
        return 0
    linear = _load(eval_dir("ladder") / "linear_fits.json")
    nonlinear = _load(eval_dir("ladder") / "nonlinear_fits.json")
    refit = _load(eval_dir("ladder") / "query_averaged_refit.json")
    detection = _load(eval_dir("detection") / "hsic_dcor.json")
    bilinear = _load(eval_dir("bilinear") / "bilinear_fits.json")
    fold_check = _load(eval_dir("fold_check") / "n50k_overlap.json")
    projections = _load(eval_dir("bilinear") / "interaction_projections.json")
    hero_ladder(linear or {}, nonlinear or {}, refit, out)
    hero_gap_closure(linear or {}, bilinear or {}, out)
    projection_cosines(projections, out)
    perfold_headline_fig(_load(eval_dir("bilinear") / "perfold_headline.json"), out)
    exploratory(linear, nonlinear, detection, bilinear, fold_check, out, refit=refit)
    made = sorted(p.name for p in out.glob("*.png"))
    atomic_write_json(
        out / "meta.json",
        {"meta": result_meta(smoke=args.smoke), "figures": made},
    )
    print(f"[figures] wrote {len(made)} PNGs to {out}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.exit(rc)
