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


def hero_ladder(linear: dict, nonlinear: dict, out: Path) -> None:
    basis = "pca48"
    units = (linear.get("units", []) if linear else []) + (
        nonlinear.get("units", []) if nonlinear else []
    )
    gains = (nonlinear or {}).get("gains_vs_ridge", {})
    schemes = sorted({u["scheme"] for u in units if u.get("grain") == "perrow"})
    fig, axes = plt.subplots(
        1, max(len(schemes), 1), figsize=(5.2 * max(len(schemes), 1), 3.6), squeeze=False
    )
    for si, scheme in enumerate(schemes):
        ax = axes[0][si]
        labels, vals, errs_lo, errs_hi, colors = [], [], [], [], []
        palette = {"ridge": "#8888aa", "krr": "#3366cc", "rff": "#66aadd", "mlp": "#cc3333"}
        for arm in ARM_ORDER:
            for rung in RUNG_ORDER:
                cand = [
                    u
                    for u in units
                    if u.get("arm") == arm
                    and u.get("scheme") == scheme
                    and u.get("grain") == "perrow"
                    and u.get("rung") == rung
                ]
                if not cand:
                    continue
                r2 = _unit_r2(cand[0], basis)
                if r2 is None:
                    continue
                labels.append(f"{arm}\n{rung}")
                vals.append(r2)
                gk = f"{arm}|perrow|{scheme}|{rung}|{basis}"
                g = gains.get(gk) or gains.get(gk + "|s0")
                if g and rung != "ridge":
                    lo, hi = _err_offsets(g["delta_r2"], g["ci95_cluster"])
                    errs_lo.append(lo)
                    errs_hi.append(hi)
                else:
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
                colors.append(palette[rung])
        x = np.arange(len(labels))
        ax.bar(x, vals, color=colors)
        ax.errorbar(x, vals, yerr=[errs_lo, errs_hi], fmt="none", ecolor="black", capsize=2, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, fontsize=6, ha="right")
        ax.set_ylabel(f"held-out R2 ({basis})")
        ax.set_title(f"{scheme}-novel folds")
    fig.suptitle("#1775 per-arm estimation ladder (CI whiskers = cluster-bootstrap gain CI)")
    fig.tight_layout()
    fig.savefig(out / "hero_ladder_bars.png", dpi=200)
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


def exploratory(linear, nonlinear, detection, bilinear, fold_check, out: Path) -> None:
    # per-fold R2 scatter
    if linear:
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        for u in linear.get("units", []):
            if u.get("basis") != "pca48" or "r2_folds" not in u:
                continue
            xs = np.full(len(u["r2_folds"]), len(ax.lines))
            ax.plot(xs, u["r2_folds"], "o", ms=3, label=f"{u['arm']}/{u['scheme']}")
        ax.set_ylabel("per-fold held-out R2 (pca48)")
        ax.set_xticks([])
        ax.legend(fontsize=5, ncol=3)
        fig.tight_layout()
        fig.savefig(out / "expl_per_fold_r2.png", dpi=160)
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
            fig.savefig(out / f"expl_rcurve_{scheme}.png", dpi=160)
            plt.close(fig)
    if detection:
        fam = detection.get("holm_adjusted_p", {})
        if fam:
            fig, ax = plt.subplots(figsize=(6.4, 3.2))
            names = sorted(fam)
            ax.bar(range(len(names)), [fam[n] for n in names], color="#446688")
            ax.axhline(0.05, color="red", ls="--", lw=0.8)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=80, fontsize=5)
            ax.set_ylabel("Holm-adjusted p")
            ax.set_title("residual dependence (registered 30-test family)")
            fig.tight_layout()
            fig.savefig(out / "expl_detection_holm.png", dpi=160)
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
            fig.savefig(out / "expl_n50k_contamination.png", dpi=160)
            plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="#1775 P5 figures")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    set_paper_style()
    out = fig_dir()
    linear = _load(eval_dir("ladder") / "linear_fits.json")
    nonlinear = _load(eval_dir("ladder") / "nonlinear_fits.json")
    detection = _load(eval_dir("detection") / "hsic_dcor.json")
    bilinear = _load(eval_dir("bilinear") / "bilinear_fits.json")
    fold_check = _load(eval_dir("fold_check") / "n50k_overlap.json")
    hero_ladder(linear or {}, nonlinear or {}, out)
    hero_gap_closure(linear or {}, bilinear or {}, out)
    exploratory(linear, nonlinear, detection, bilinear, fold_check, out)
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
