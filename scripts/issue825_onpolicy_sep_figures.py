"""Issue #825 `onpolicy-separator-control` Phase C2 figures (plan section 6).

Hero (``figures/issue_825/onpolicy_sep_hero``): per model, grouped bars @ L19 —
chat ceiling / exogenous separator (rotated + MLP) / ON-POLICY separator
(rotated + MLP, rotated bar with the group-bootstrap CI whisker), raw ridge
greyed + clipped; companion panel: transfer fractions (exogenous committed vs
on-policy where the Phase C JSONs exist) with the 0.5 line; D annotated per
model. Low-level per-unit companion: per-article-group ROTATED-estimator R^2
scatter @ L19, on-policy vs exogenous, per model (from the new per-group
rotated persist — NOT the pathological ridge per_group_r2_headline).
Exploratory dump (over-produce): separator-type / span-length /
anchor-position distributions vs exogenous; continuation audit panels
(length / repetition / true-continuation overlap); ridge layer curves with
null bands + frozen layers; prevmean companion; per-group rotated R^2 vs
repetition scatter; seam-mismatch summary.

CLI:
  uv run python scripts/issue825_onpolicy_sep_figures.py \
      [--out-dir eval_results/issue_825/onpolicy-separator-control] \
      [--data-root data/issue_825/onpolicy_sep] [--fig-dir figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

MODELS = ("base", "instruct")
ANCHOR_OF = {"base": "anchor_base", "instruct": "anchor_inst"}
L = 19


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-dir", type=Path, default=Path("eval_results/issue_825/onpolicy-separator-control")
    )
    ap.add_argument("--data-root", type=Path, default=Path("data/issue_825/onpolicy_sep"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures"))
    ap.add_argument(
        "--fig-prefix",
        type=str,
        default="issue_825",
        help="subdir under --fig-dir (smoke passes a scratch prefix)",
    )
    return ap.parse_args()


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:  # noqa: C901 -- linear figure dump
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    dec = _read(args.out_dir / "decision_support.json")
    assert dec, f"decision_support.json missing under {args.out_dir}"
    per_model = dec["per_model"]
    c_on = paper_palette_role("primary")
    c_exo = paper_palette_role("baseline")
    c_ceil = paper_palette_role("accent")
    c_grey = paper_palette_role("neutral")

    # ---- Hero: within-strength bars + transfer-fraction companion ----------
    fig, (ax, axf) = plt.subplots(
        1, 2, figsize=(9.6, 4.2), width_ratios=[2.4, 1.0], layout="constrained"
    )
    labels = [
        "chat\nceiling",
        "exo sep\n(rotated)",
        "exo sep\n(MLP)",
        "on-policy\n(rotated)",
        "on-policy\n(MLP)",
        "on-policy\n(raw ridge)",
    ]
    ymin = -1.0
    width = 0.38
    for mi, m in enumerate(MODELS):
        pm = per_model[m]
        refs, on = pm["committed_reference"], pm["onpolicy"]
        vals = [
            refs["ceiling_fulln"],
            refs["rotated"],
            refs["mlp"],
            on["rotated"],
            on["mlp"] if on["mlp"] is not None else float("nan"),
            on["ridge"],
        ]
        colors = [c_ceil, c_exo, c_exo, c_on, c_on, c_grey]
        xs = np.arange(len(vals)) + (mi - 0.5) * width
        for x, v, c in zip(xs, vals, colors, strict=True):
            alpha = 0.55 if c is c_grey else (1.0 if mi == 0 else 0.65)
            ax.bar(x, max(v, ymin), width, color=c, alpha=alpha, edgecolor="white", linewidth=0.4)
            if v < ymin:
                ax.annotate(
                    f"{v:.2f}", (x, ymin), ha="center", va="bottom", fontsize=6, rotation=90
                )
        ci = on.get("rotated_ci")
        if ci:
            ax.errorbar(
                xs[3],
                on["rotated"],
                yerr=[
                    [max(0.0, on["rotated"] - ci["ci_lo"])],
                    [max(0.0, ci["ci_hi"] - on["rotated"])],
                ],
                fmt="none",
                ecolor="#333333",
                elinewidth=0.9,
                capsize=2,
            )
        ax.annotate(
            f"{m}: D={pm['D']:.2f} (n={pm['realized_n']})",
            (0.02, 0.97 - 0.07 * mi),
            xycoords="axes fraction",
            fontsize=7,
            va="top",
        )
    ax.axhline(0.0, color="#999999", lw=0.8, zorder=0)
    ax.set_ylim(ymin, 1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Held-out $R^2$ (layer 19)")
    ax.set_title("Separator within-strength: on-policy vs exogenous (dark=base, light=instruct)")
    # Transfer-fraction companion (R4; on-policy filled by Phase C when present).
    for mi, m in enumerate(MODELS):
        pm = per_model[m]
        exo_frac = pm["committed_reference"]["exo_transfer_fraction_fulln"]
        tr = _read(args.out_dir / f"onpolicy_sep_to_chat_{m}.json")
        on_frac = float(tr["sep_to_chat"]["fraction_of_fulln_ceiling"]) if tr else float("nan")
        axf.bar(mi - 0.19, exo_frac, 0.34, color=c_exo, label="exogenous" if mi == 0 else None)
        axf.bar(mi + 0.19, on_frac, 0.34, color=c_on, label="on-policy" if mi == 0 else None)
        if not np.isfinite(on_frac):
            axf.annotate("C1\npending", (mi + 0.19, 0.02), ha="center", fontsize=6)
    axf.axhline(0.5, color="#bbbbbb", lw=0.8, ls="--")
    axf.set_xticks(range(len(MODELS)))
    axf.set_xticklabels(MODELS, fontsize=8)
    axf.set_ylabel("sep$\\to$chat transfer / full-n ceiling")
    axf.set_ylim(0, 1.0)
    axf.legend(fontsize=6, loc="upper right")
    savefig_paper(fig, f"{args.fig_prefix}/onpolicy_sep_hero", dir=args.fig_dir)
    plt.close(fig)

    # ---- Low-level: per-article-group ROTATED R^2 scatter ------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), layout="constrained")
    for axi, m in zip(axes, MODELS, strict=True):
        on_cell = _read(args.out_dir / m / "cells_armC_sep.json") or {}
        exo_cell = _read(args.out_dir / ANCHOR_OF[m] / "cells_armC_sep.json") or {}
        on_pg = on_cell.get("per_group_rotated_r2_headline", {})
        exo_pg = exo_cell.get("per_group_rotated_r2_headline", {})
        shared = sorted(set(on_pg) & set(exo_pg))
        if shared:
            xv = np.array([exo_pg[g] for g in shared])
            yv = np.array([on_pg[g] for g in shared])
            axi.scatter(xv, yv, s=9, alpha=0.4, color=c_on, edgecolors="none")
            lo = float(min(xv.min(), yv.min()))
            hi = float(max(xv.max(), yv.max()))
            axi.plot([lo, hi], [lo, hi], color="#999999", lw=0.8, ls="--")
        axi.set_xlabel("exogenous per-group rotated $R^2$ (L19)")
        axi.set_ylabel("on-policy per-group rotated $R^2$")
        axi.set_title(f"{m} ({len(shared)} article groups)")
    savefig_paper(fig, f"{args.fig_prefix}/onpolicy_sep_pergroup_rotated", dir=args.fig_dir)
    plt.close(fig)

    # ---- Exploratory A: pair-distribution nuisance covariates --------------
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(10.5, 6.4), layout="constrained")
    for mi, m in enumerate(MODELS):
        meta = _read(args.data_root / m / "pairs" / "pairs_meta.json") or {}
        on_s, ex_s = meta.get("onpolicy_stats", {}), meta.get("exogenous_stats") or {}
        # separator-type frequencies
        axi = axes[mi][0]
        seps = sorted(
            set(on_s.get("separator_frequencies", {})) | set(ex_s.get("separator_frequencies", {}))
        )
        if seps:
            onf = np.array([on_s["separator_frequencies"].get(s, 0) for s in seps], dtype=float)
            exf = np.array(
                [ex_s.get("separator_frequencies", {}).get(s, 0) for s in seps], dtype=float
            )
            onf, exf = onf / max(1.0, onf.sum()), exf / max(1.0, exf.sum())
            x = np.arange(len(seps))
            axi.bar(x - 0.19, exf, 0.34, color=c_exo, label="exogenous")
            axi.bar(x + 0.19, onf, 0.34, color=c_on, label="on-policy")
            axi.set_xticks(x)
            axi.set_xticklabels(seps)
        axi.set_title(f"{m}: separator type (frac)")
        axi.legend(fontsize=6)
        for ci, key, lbl in (
            (1, "span_length", "span length (tokens)"),
            (2, "anchor_position", "anchor position (token idx)"),
        ):
            axi = axes[mi][ci]
            onv = (on_s.get(key) or {}).get("values", [])
            exv = (ex_s.get(key) or {}).get("values", [])
            if exv:
                axi.hist(exv, bins=30, density=True, alpha=0.5, color=c_exo, label="exogenous")
            if onv:
                axi.hist(onv, bins=30, density=True, alpha=0.5, color=c_on, label="on-policy")
            axi.set_title(f"{m}: {lbl}")
            axi.legend(fontsize=6)
    savefig_paper(fig, f"{args.fig_prefix}/onpolicy_sep_pair_distributions", dir=args.fig_dir)
    plt.close(fig)

    # ---- Exploratory B: audit + layer curves + degeneracy scatter ----------
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(10.5, 6.4), layout="constrained")
    for mi, m in enumerate(MODELS):
        audit = _read(args.data_root / m / "generation" / "generation_audit.json") or {}
        rows = audit.get("per_row", [])
        axi = axes[mi][0]
        if rows:
            axi.hist([r["n_tokens"] for r in rows], bins=30, color=c_on, alpha=0.7)
        rep = audit.get("repetition_rate_min5")
        eos = audit.get("early_eos_rate")
        ov = (audit.get("true_continuation_overlap") or {}).get("mean")
        w2 = audit.get("n_wave2")
        axi.set_title(
            f"{m}: cont. length (rep={rep if rep is None else round(rep, 3)}, "
            f"eos={eos if eos is None else round(eos, 3)}, "
            f"overlap={ov if ov is None else round(ov, 3)}, wave2={w2})",
            fontsize=7,
        )
        # ridge layer curve + null band + frozen layers (diagnostic, no max headline)
        axi = axes[mi][1]
        cell = _read(args.out_dir / m / "cells_armC_sep.json") or {}
        nulls = _read(args.out_dir / m / "nulls_armC_sep.json") or {}
        curve = cell.get("r2_per_layer_obs", [])
        if curve:
            axi.plot(range(len(curve)), curve, color=c_on, lw=1.2, label="observed")
            nm = np.asarray(nulls.get("null_matrix", []))
            if nm.size:
                axi.fill_between(
                    range(nm.shape[1]),
                    np.nanquantile(nm, 0.025, axis=0),
                    np.nanquantile(nm, 0.975, axis=0),
                    color=c_grey,
                    alpha=0.4,
                    label="null band",
                )
            for fl in cell.get("frozen_layers", []):
                axi.axvline(fl, color="#bbbbbb", lw=0.6, ls=":")
        axi.set_ylim(-4, 1)
        axi.set_title(f"{m}: ridge layer curve (armC_sep)", fontsize=8)
        axi.legend(fontsize=6)
        # per-group rotated R^2 vs wave-1 repetition (pre-registered diagnostic)
        axi = axes[mi][2]
        on_pg = (cell or {}).get("per_group_rotated_r2_headline", {})
        rep_of = {r["window_id"]: r["repeats_3gram_min5"] for r in rows if r.get("wave") == 1}
        shared = sorted(set(on_pg) & set(rep_of))
        if shared:
            xv = np.array([1.0 if rep_of[g] else 0.0 for g in shared])
            yv = np.array([on_pg[g] for g in shared])
            jitter = (np.random.default_rng(0).random(len(xv)) - 0.5) * 0.2
            axi.scatter(xv + jitter, yv, s=8, alpha=0.4, color=c_on, edgecolors="none")
        axi.set_xticks([0, 1])
        axi.set_xticklabels(["no rep", "3-gram rep$\\geq$5"], fontsize=7)
        axi.set_title(f"{m}: per-group rotated $R^2$ vs repetition", fontsize=8)
    savefig_paper(fig, f"{args.fig_prefix}/onpolicy_sep_audit_panels", dir=args.fig_dir)
    plt.close(fig)

    # ---- Exploratory C: prevmean companion + seam mismatch -----------------
    fig, (axp, axs) = plt.subplots(1, 2, figsize=(8.6, 3.8), layout="constrained")
    for mi, m in enumerate(MODELS):
        pm = per_model[m]
        prev_on = pm["onpolicy"].get("prevmean_rotated")
        exo_cell = _read(args.out_dir / ANCHOR_OF[m] / "cells_armC_prevmean.json") or {}
        hl = int(exo_cell.get("headline_layer", L))
        prev_exo = (exo_cell.get("random_projection_control_r2") or {}).get(str(hl))
        axp.bar(
            mi - 0.19,
            prev_exo if prev_exo is not None else np.nan,
            0.34,
            color=c_exo,
            label="exogenous" if mi == 0 else None,
        )
        axp.bar(
            mi + 0.19,
            prev_on if prev_on is not None else np.nan,
            0.34,
            color=c_on,
            label="on-policy" if mi == 0 else None,
        )
        meta = _read(args.data_root / m / "pairs" / "pairs_meta.json") or {}
        seam = (meta.get("seam_token_mismatch") or {}).get("per_window", [])
        if seam:
            axs.hist(
                [s["first_divergence"] for s in seam],
                bins=20,
                alpha=0.5,
                color=c_on if m == "base" else c_exo,
                label=m,
            )
    axp.set_xticks(range(len(MODELS)))
    axp.set_xticklabels(MODELS)
    axp.set_ylabel("prevmean rotated $R^2$ (L19)")
    axp.set_title("preceding-sentence variant", fontsize=9)
    axp.legend(fontsize=6)
    axs.set_xlabel("first retok-vs-gen divergence (token idx)")
    axs.set_title("re-tokenization seam (diagnostic)", fontsize=9)
    axs.legend(fontsize=6)
    savefig_paper(fig, f"{args.fig_prefix}/onpolicy_sep_prevmean_seam", dir=args.fig_dir)
    plt.close(fig)
    print(f"[i825-ops-figs] wrote 5 figures under {args.fig_dir}/{args.fig_prefix}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
