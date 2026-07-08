"""Issue #825 `sampled-separator-control` Phase C2 figures (plan v22 section 6).

Hero (``figures/issue_825/sampled_sep_hero``): per model, D per arm (armB /
C-avg "E[v] ceiling read" / C-single / C-pooled) with CI whiskers where the
mapped bootstrap CI exists, the FROZEN round-7 greedy D as a reference line
(with its committed rotated CI band where available) and the +-0.10 stability
band. Low-level per-unit companion: per-article-group ROTATED R^2 scatter @
L19, round-8 armB vs the committed round-7 greedy per-group values.
Exploratory dump (over-produce): flag-rate-vs-D panel across arms + round 7;
NS decomposition (C-avg vs C-single vs C-pooled); span-length /
anchor-position distributions per arm vs exogenous; K_valid + X-identity +
prefix-tail distributions; per-layer ridge/rotated curves with null bands;
audit panels (length / repetition / distinct-3gram / early-EOS / overlap).

CLI:
  uv run python scripts/issue825_sampled_sep_figures.py \
      [--out-dir eval_results/issue_825/sampled-separator-control] \
      [--data-root data/issue_825/sampled_sep] [--fig-dir figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

MODELS = ("base", "instruct")
ARMS = ("armB", "armC_avg", "armC_single", "armC_pooled")
ARM_LABELS = {
    "armB": "arm B\n(sampled, 1 draw)",
    "armC_avg": "C-avg\n(E[v] ceiling read)",
    "armC_single": "C-single\n(draw 0)",
    "armC_pooled": "C-pooled\n(all draws)",
}
R7_OUT = Path("eval_results/issue_825/onpolicy-separator-control")
MARGIN = 0.10


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-dir", type=Path, default=Path("eval_results/issue_825/sampled-separator-control")
    )
    ap.add_argument("--data-root", type=Path, default=Path("data/issue_825/sampled_sep"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures"))
    ap.add_argument(
        "--fig-prefix",
        type=str,
        default="issue_825",
        help="subdir under --fig-dir (smoke passes a scratch prefix)",
    )
    ap.add_argument(
        "--r7-out-dir",
        type=Path,
        default=R7_OUT,
        help="committed round-7 OUT dir (per-group reference scatter)",
    )
    return ap.parse_args()


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _bins(vals, requested: int) -> int:
    """Clamp histogram bins for (near-)constant data — a degenerate value range
    (e.g. all X-identity cosines == 1.0 at smoke scale) cannot host 40
    finite-sized bins (numpy ValueError)."""
    distinct = len(np.unique(np.round(np.asarray(vals, dtype=np.float64), 12)))
    return max(1, min(requested, distinct))


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
    fig_dir = args.fig_dir / args.fig_prefix
    fig_dir.mkdir(parents=True, exist_ok=True)
    dec = _read(args.out_dir / "decision_support.json")
    assert dec, f"decision_support.json missing under {args.out_dir}"
    per_model = dec["per_model"]
    c_arm = paper_palette_role("primary")
    c_ref = paper_palette_role("baseline")
    c_acc = paper_palette_role("accent")
    c_grey = paper_palette_role("neutral")

    # ---- Hero: D per arm + frozen round-7 reference line -------------------
    fig, axes = plt.subplots(1, len(MODELS), figsize=(9.6, 4.2), layout="constrained")
    for ax, m in zip(np.atleast_1d(axes), MODELS, strict=True):
        pm = per_model.get(m)
        if not pm:
            continue
        arms = pm["arms"]
        d_r7 = pm["round7_reference"]["D_r7"]
        xs, vals, errs = [], [], []
        for i, arm in enumerate(ARMS):
            a = arms.get(arm) or {}
            if a.get("missing") or "D" not in a:
                continue
            xs.append(i)
            vals.append(a["D"])
            ci = a.get("D_ci")
            errs.append(
                (a["D"] - ci["lo"], ci["hi"] - a["D"]) if ci else (float("nan"), float("nan"))
            )
        colors = [c_arm if ARMS[i] == "armB" else c_grey for i in xs]
        ax.bar(xs, vals, 0.6, color=colors, edgecolor="white", linewidth=0.4)
        for x, v, (lo, hi) in zip(xs, vals, errs, strict=True):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(x, v, yerr=[[max(0.0, lo)], [max(0.0, hi)]], fmt="none", ecolor="0.2")
        ax.axhline(d_r7, color=c_ref, lw=1.4)
        ax.axhspan(d_r7 - MARGIN, d_r7 + MARGIN, color=c_ref, alpha=0.12)
        ci7 = pm["round7_reference"].get("D_ci_rotated_r7")
        if ci7:
            ax.axhspan(float(ci7["lo"]), float(ci7["hi"]), color=c_acc, alpha=0.10)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([ARM_LABELS[ARMS[i]] for i in xs], fontsize=7)
        ax.set_ylabel("D = (W_on - W_ex) / (C - W_ex)" if m == "base" else "")
        ax.set_title(f"{m}: sampled D per arm vs round-7 greedy (line +- 0.10 band)")
    savefig_paper(fig, fig_dir / "sampled_sep_hero")
    plt.close(fig)

    # ---- Low-level companion: per-group rotated R^2, armB vs round 7 -------
    fig, axes = plt.subplots(1, len(MODELS), figsize=(9.0, 4.2), layout="constrained")
    for ax, m in zip(np.atleast_1d(axes), MODELS, strict=True):
        r8 = _read(args.out_dir / m / "armB" / "cells_armC_sep.json") or {}
        r7 = _read(args.r7_out_dir / m / "cells_armC_sep.json") or {}
        g8 = r8.get("per_group_rotated_r2_headline") or {}
        g7 = r7.get("per_group_rotated_r2_headline") or {}
        shared = sorted(set(g8) & set(g7))
        if shared:
            x = np.asarray([g7[g] for g in shared], dtype=np.float64)
            y = np.asarray([g8[g] for g in shared], dtype=np.float64)
            keep = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[keep], y[keep], s=8, alpha=0.5, color=c_arm)
            lim_lo = float(min(x[keep].min(), y[keep].min())) if keep.any() else -1
            lim_hi = float(max(x[keep].max(), y[keep].max())) if keep.any() else 1
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=c_grey, lw=0.8, ls="--")
            ax.annotate(f"n groups = {int(keep.sum())}", (0.04, 0.94), xycoords="axes fraction")
        ax.set_xlabel("round-7 greedy per-group rotated R2 @ L19")
        ax.set_ylabel("round-8 sampled armB")
        ax.set_title(m)
    savefig_paper(fig, fig_dir / "sampled_sep_pergroup_scatter")
    plt.close(fig)

    # ---- Flag-rate vs D across arms + round 7 ------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.2), layout="constrained")
    markers = {"base": "o", "instruct": "s"}
    for m in MODELS:
        pm = per_model.get(m) or {}
        r6 = pm.get("r6_flag_rates") or {}
        arms = pm.get("arms") or {}
        pts = []
        fr_b = (r6.get("armB") or {}).get("repetition_rate_min5")
        if fr_b is not None and not (arms.get("armB") or {}).get("missing"):
            pts.append((fr_b, arms["armB"]["D"], "armB"))
        fr_c = (r6.get("armC") or {}).get("repetition_rate_min5")
        for arm in ("armC_avg", "armC_single", "armC_pooled"):
            a = arms.get(arm) or {}
            if fr_c is not None and not a.get("missing") and "D" in a:
                pts.append((fr_c, a["D"], arm))
        r7ref = pm.get("round7_reference") or {}
        if r7ref:
            pts.append((r7ref["flag_rate_r7"], r7ref["D_r7"], "round-7 greedy"))
        for fr, d, label in pts:
            col = c_ref if label == "round-7 greedy" else c_arm
            ax.scatter(fr, d, marker=markers[m], color=col, s=36)
            ax.annotate(
                f"{m}:{label}", (fr, d), fontsize=6, xytext=(3, 3), textcoords="offset points"
            )
    ax.set_xlabel("3-gram repetition flag rate (min count 5)")
    ax.set_ylabel("D")
    ax.set_title("degeneration vs D across arms (R6 premise check)")
    savefig_paper(fig, fig_dir / "sampled_sep_flagrate_vs_d")
    plt.close(fig)

    # ---- NS decomposition (C-avg vs C-single vs C-pooled) ------------------
    fig, axes = plt.subplots(1, len(MODELS), figsize=(9.0, 4.0), layout="constrained")
    for ax, m in zip(np.atleast_1d(axes), MODELS, strict=True):
        pm = per_model.get(m) or {}
        arms = pm.get("arms") or {}
        labels, vals, errs = [], [], []
        for arm in ("armC_avg", "armC_single", "armC_pooled"):
            a = arms.get(arm) or {}
            if a.get("missing") or "reads" not in a:
                continue
            labels.append(ARM_LABELS[arm])
            vals.append(a["reads"]["w_max"])
            ci = a.get("D_ci")
            reads = a["reads"]
            rci = reads.get("mlp_ci") if reads.get("mlp_wins_max") else reads.get("rotated_ci")
            errs.append(
                (vals[-1] - rci["ci_lo"], rci["ci_hi"] - vals[-1])
                if rci
                else (float("nan"), float("nan"))
            )
        xs = np.arange(len(labels))
        ax.bar(xs, vals, 0.55, color=c_arm, edgecolor="white")
        for x, v, (lo, hi) in zip(xs, vals, errs, strict=True):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(x, v, yerr=[[max(0.0, lo)], [max(0.0, hi)]], fmt="none", ecolor="0.2")
        r3 = pm.get("r3_sampling_noise_share") or {}
        if r3:
            ax.annotate(
                f"NS(max) = {r3.get('ns_max_interpretable'):.4f}"
                if r3.get("ns_max_interpretable") is not None
                else "NS n/a",
                (0.04, 0.94),
                xycoords="axes fraction",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("W (max-interpretable R2 @ L19)" if m == "base" else "")
        ax.set_title(m)
    savefig_paper(fig, fig_dir / "sampled_sep_ns_decomposition")
    plt.close(fig)

    # ---- Pair distributions per arm (span length / anchor position) --------
    fig, axes = plt.subplots(2, len(MODELS), figsize=(9.6, 6.4), layout="constrained")
    axes = np.atleast_2d(axes)
    for mi, m in enumerate(MODELS):
        for ri, key in enumerate(("span_length", "anchor_position")):
            ax = axes[ri][mi]
            for arm, col in (("armB", c_arm), ("armC", c_acc)):
                meta = _read(args.data_root / m / arm / "pairs" / "pairs_meta.json") or {}
                vals = ((meta.get("onpolicy_stats") or {}).get(key) or {}).get("values") or []
                if vals:
                    ax.hist(
                        vals, bins=_bins(vals, 40), alpha=0.5, label=arm, color=col, density=True
                    )
                if arm == "armB":
                    exo = ((meta.get("exogenous_stats") or {}).get(key) or {}).get("values") or []
                    if exo:
                        ax.hist(
                            exo,
                            bins=_bins(exo, 40),
                            alpha=0.35,
                            label="exogenous",
                            color=c_ref,
                            density=True,
                        )
            ax.set_title(f"{m}: {key}")
            ax.legend(fontsize=6)
    savefig_paper(fig, fig_dir / "sampled_sep_pair_distributions")
    plt.close(fig)

    # ---- K_valid + X-identity + prefix-tail distributions ------------------
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), layout="constrained")
    for m, col in zip(MODELS, (c_arm, c_acc), strict=True):
        red = _read(args.out_dir / m / "reduce_summary.json") or {}
        kv = red.get("k_valid_distribution") or {}
        if kv:
            ks = sorted(int(k) for k in kv)
            axes[0].plot(ks, [kv[str(k)] for k in ks], marker="o", label=m, color=col)
        gate = red.get("x_identity_gate") or {}
        pam = list((gate.get("per_article_min") or {}).values())
        if pam:
            axes[1].hist(pam, bins=_bins(pam, 40), alpha=0.5, label=m, color=col)
        meta = _read(args.data_root / m / "armC" / "pairs" / "pairs_meta.json") or {}
        pt = (meta.get("prefix_tail_tokens") or {}).get("values") or []
        if pt:
            axes[2].hist(pt, bins=_bins(pt, 30), alpha=0.5, label=m, color=col)
    axes[0].set_title("K_valid per article")
    axes[1].set_title("X-identity per-article min cos @ L19")
    axes[1].axvline(0.999, color="red", lw=0.8, ls="--")
    axes[2].set_title("prefix-tail tokens in span (nuisance)")
    for ax in axes:
        ax.legend(fontsize=6)
    savefig_paper(fig, fig_dir / "sampled_sep_kvalid_xidentity")
    plt.close(fig)

    # ---- Layer curves with null bands per (model, arm) ---------------------
    fig, axes = plt.subplots(
        len(MODELS), len(ARMS), figsize=(3.2 * len(ARMS), 3.0 * len(MODELS)), layout="constrained"
    )
    axes = np.atleast_2d(axes)
    for mi, m in enumerate(MODELS):
        for ai, arm in enumerate(ARMS):
            ax = axes[mi][ai]
            nulls = _read(args.out_dir / m / arm / "nulls_armC_sep.json")
            cellsj = _read(args.out_dir / m / arm / "cells_armC_sep.json")
            if not nulls or not cellsj:
                ax.set_axis_off()
                continue
            obs = np.asarray(nulls["observed_row"], dtype=np.float64)
            nm = np.asarray(nulls["null_matrix"], dtype=np.float64)
            xs = np.arange(len(obs))
            ax.plot(xs, np.clip(obs, -1, 1), color=c_arm, lw=1.2, label="ridge obs")
            if nm.size:
                ax.fill_between(
                    xs,
                    np.clip(np.nanquantile(nm, 0.025, axis=0), -1, 1),
                    np.clip(np.nanquantile(nm, 0.975, axis=0), -1, 1),
                    color=c_grey,
                    alpha=0.3,
                    label="null band",
                )
            for fl in cellsj.get("frozen_layers", []):
                ax.axvline(fl, color=c_acc, lw=0.5, ls=":")
            rot = cellsj.get("rotated_r2_frozen") or {}
            if rot:
                ax.scatter(
                    [int(k) for k in rot],
                    [np.clip(v, -1, 1) for v in rot.values()],
                    color=c_ref,
                    s=14,
                    zorder=3,
                    label="rotated",
                )
            ax.set_title(f"{m} {arm}", fontsize=8)
            if mi == 0 and ai == 0:
                ax.legend(fontsize=6)
    savefig_paper(fig, fig_dir / "sampled_sep_layer_curves")
    plt.close(fig)

    # ---- Audit panels per arm ----------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.4), layout="constrained")
    keys = ("repetition_rate_min5", "distinct_3gram_rate", "early_eos_rate", "overlap")
    for ki, key in enumerate(keys):
        ax = axes[ki]
        labels, vals, colors = [], [], []
        for m in MODELS:
            for arm in ("armB", "armC"):
                audit = _read(args.data_root / m / arm / "generation" / "generation_audit.json")
                if not audit:
                    continue
                v = (
                    (audit.get("true_continuation_overlap") or {}).get("mean")
                    if key == "overlap"
                    else audit.get(key)
                )
                labels.append(f"{m}\n{arm}")
                vals.append(v if v is not None else float("nan"))
                colors.append(c_arm if arm == "armB" else c_acc)
        xs = np.arange(len(labels))
        ax.bar(xs, vals, 0.6, color=colors)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_title(key, fontsize=8)
        if key == "repetition_rate_min5":
            for m, ls in zip(MODELS, ("--", ":"), strict=True):
                q = (per_model.get(m) or {}).get("round7_reference") or {}
                if q:
                    ax.axhline(q["flag_rate_r7"], color=c_ref, lw=0.8, ls=ls, label=f"r7 {m}")
            ax.legend(fontsize=6)
    savefig_paper(fig, fig_dir / "sampled_sep_audit_panels")
    plt.close(fig)

    print(f"[i825-ss-fig] figures written under {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
