#!/usr/bin/env python
"""#825 turn-dynamics-allturns-5000 P5 figures (plan v24 §6 F1-F7 + exploratory).

Reads eval_results/issue_825/turn_dynamics/results.json (the P4 assemble
output) and writes PNG+PDF+meta.json under --fig-dir. No conversation text is
read anywhere here — the results JSON carries only numbers/ids.

F1 hero  per-turn ctx R2 vs depth (L19), armG solid + armR overlays, null
         band, cross-recipe single-turn anchors (caption caveats REQUIRED).
F2       transfer-matrix heatmaps (retained fraction, i x j, model x arm).
F3       operator similarity vs |i-j| (raw / Procrustes / general-linear)
         AGAINST the within-turn self-cosine ceiling band.
F4       turn-1 reach: ridge vs MLP R2 vs k + null bands + k* band.
F5       pooled-vs-per-turn bars (Simpson's check).
F6       H4 bridge: per-depth delta + CI + per-arm overflow-drop rates.
F7       rollout degeneracy diagnostics vs depth.
expl     n(k) yield curve, prefix-arm panels, G-C parity overlay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    set_paper_style,
)

MODELS = ("instruct", "pretrained")
ANCHORS = {"instruct_single_turn": 0.673, "pretrained_single_turn": 0.588, "matched_n": 0.476}
ANCHOR_CAVEAT = (
    "Anchor lines are CROSS-RECIPE references (single-turn prompt corpus vs this round's "
    "deep-stratified panel; 5- vs 6-fold CV; layer set; bf16 vs fp16 store) — orientation, "
    "not matched baselines."
)


def _save(fig, fig_dir: Path, stem: str, caption: str, commit: str) -> None:
    fig.savefig(fig_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    meta = {
        "figure": f"{stem}.png",
        "git_commit": commit,
        "source_results_json": "eval_results/issue_825/turn_dynamics/results.json",
        "caption": caption,
    }
    with open(fig_dir / f"{stem}.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[write] {fig_dir / (stem + '.png')}")


def _cells(payload: dict, arm: str, model: str) -> dict:
    return payload["parts"].get(f"cells_{arm}_{model}", {}).get("per_turn", {}).get("19", {})


def _curve(cells: dict, arm_x: str = "ctx") -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    ts, r2s, folds = [], [], []
    for t_s, node in sorted(cells.items(), key=lambda kv: int(kv[0])):
        sub = node.get(arm_x)
        if isinstance(sub, dict) and sub.get("status") == "computed":
            ts.append(int(t_s))
            r2s.append(sub["r2"])
            folds.append(sub.get("r2_folds") or [])
    return np.asarray(ts), np.asarray(r2s), folds


def _null_band(cells: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts, mean, hi = [], [], []
    for t_s, node in sorted(cells.items(), key=lambda kv: int(kv[0])):
        sub = node.get("ctx")
        if isinstance(sub, dict) and sub.get("null_n_draws"):
            ts.append(int(t_s))
            mean.append(sub["null_mean"])
            hi.append(sub.get("null_max") if sub.get("null_max") is not None else sub["null_hi"])
    return np.asarray(ts), np.asarray(mean), np.asarray(hi)


def fig_hero(payload: dict, fig_dir: Path) -> None:
    colors = paper_palette(4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        for ci, (arm, label, style) in enumerate(
            (
                ("armG", "simulated-user continuations (own answers)", "-"),
                ("armR_own", "real conversations (own answers)", "--"),
                ("armR_logged", "real conversations (logged answers)", ":"),
            )
        ):
            ts, r2s, folds = _curve(_cells(payload, arm, model))
            if not ts.size:
                continue
            ax.plot(ts, r2s, style, color=colors[ci], marker="o", ms=3, label=label, lw=1.4)
            for t, fvals in zip(ts, folds, strict=True):
                if fvals:
                    ax.scatter([t] * len(fvals), fvals, s=4, color=colors[ci], alpha=0.25)
        nts, nmean, nhi = _null_band(_cells(payload, "armG", model))
        if nts.size:
            ax.fill_between(nts, nmean, nhi, color="grey", alpha=0.25, label="shuffle null band")
        anchor = ANCHORS[f"{model}_single_turn"]
        ax.axhline(anchor, color="k", lw=0.8, ls="-.", alpha=0.6)
        ax.axhline(ANCHORS["matched_n"], color="k", lw=0.8, ls=":", alpha=0.6)
        ax.set_title(model)
        ax.set_xlabel("assistant turn t")
    axes[0].set_ylabel(r"held-out conv-grouped $R^2$ (context $\to$ answer, L19)")
    axes[0].legend(fontsize=6, loc="upper right")
    _save(
        fig,
        fig_dir,
        "turndyn_f1_hero",
        "Per-turn context-to-answer map strength vs depth at flat n; "
        "dash-dot/dotted horizontal lines = single-turn and matched-n anchors. " + ANCHOR_CAVEAT,
        payload["git_commit"],
    )


def fig_transfer(payload: dict, fig_dir: Path) -> None:
    for arm in ("armG", "armR_own"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
        drawn = False
        for ax, model in zip(axes, MODELS, strict=True):
            part = payload["parts"].get(f"transfer_{arm}_{model}")
            if not part:
                ax.set_axis_off()
                continue
            turns = part["turns"]
            mat = np.full((len(turns), len(turns)), np.nan)
            for ii, i in enumerate(turns):
                for jj, j in enumerate(turns):
                    v = part["retained_fraction"].get(f"{i}->{j}")
                    mat[ii, jj] = np.nan if v is None else v
            im = ax.imshow(mat, vmin=0, vmax=1.2, cmap="viridis", origin="lower")
            ax.set_xticks(range(len(turns)), turns, fontsize=5)
            ax.set_yticks(range(len(turns)), turns, fontsize=5)
            ax.set_xlabel("target turn j")
            ax.set_ylabel("source turn i")
            ax.set_title(f"{model} — {arm}")
            drawn = True
        if drawn:
            fig.colorbar(im, ax=axes, shrink=0.8, label="retained fraction R2(i→j)/R2(j→j)")
        _save(
            fig,
            fig_dir,
            f"turndyn_f2_transfer_{arm}",
            f"Cross-turn transfer matrix ({arm}): map fit at source turn i applied to "
            "held-out rows of target turn j, as a fraction of j's own-map R2.",
            payload["git_commit"],
        )


def fig_operators(payload: dict, fig_dir: Path) -> None:
    colors = paper_palette(3)
    for arm in ("armG", "armR_own"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, model in zip(axes, MODELS, strict=True):
            part = payload["parts"].get(f"operators_{arm}_{model}")
            if not part:
                ax.set_axis_off()
                continue
            turns = part["turns"]
            by_lag: dict[int, dict[str, list[float]]] = {}
            for key, rec in part["battery"].items():
                i_s, j_s = key.split("~")
                lag = abs(int(j_s) - int(i_s))
                for mi, metric in enumerate(
                    ("raw_cos_mean", "procrustes_cos", "general_linear_cos")
                ):
                    v = rec.get(metric)
                    if v is not None:
                        by_lag.setdefault(lag, {}).setdefault(metric, []).append(v)
                    del mi
            ceil_vals = [
                rec.get("raw_cos_mean")
                for t, rec in (part["selfsim_ceiling"] or {}).items()
                if rec and rec.get("raw_cos_mean") is not None
            ]
            for mi, (metric, label) in enumerate(
                (
                    ("raw_cos_mean", "raw cosine"),
                    ("procrustes_cos", "Procrustes-aligned"),
                    ("general_linear_cos", "general-linear"),
                )
            ):
                lags = sorted(lag for lag in by_lag if metric in by_lag[lag] and lag > 0)
                if not lags:
                    continue
                means = [float(np.mean(by_lag[lag][metric])) for lag in lags]
                ax.plot(lags, means, marker="o", ms=3, color=colors[mi], label=label, lw=1.2)
                for lag in lags:
                    vals = by_lag[lag][metric]
                    ax.scatter([lag] * len(vals), vals, s=4, color=colors[mi], alpha=0.2)
            if ceil_vals:
                lo, hi = float(np.min(ceil_vals)), float(np.max(ceil_vals))
                ax.axhspan(
                    lo, hi, color="grey", alpha=0.25, label="within-turn self-cosine ceiling"
                )
            ax.set_title(f"{model} — {arm} ({len(turns)} turns)")
            ax.set_xlabel("|i - j| (turn lag)")
        axes[0].set_ylabel("operator similarity (L19 betas)")
        axes[0].legend(fontsize=6)
        _save(
            fig,
            fig_dir,
            f"turndyn_f3_operators_{arm}",
            f"Operator similarity vs turn lag ({arm}); grey band = within-turn cross-fold "
            "self-cosine ceiling — every level is read AGAINST it (estimation noise, not "
            "re-parameterization).",
            payload["git_commit"],
        )


def fig_reach(payload: dict, fig_dir: Path) -> None:
    colors = paper_palette(4)
    for arm in ("armG", "armR_own"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, model in zip(axes, MODELS, strict=True):
            part = payload["parts"].get(f"reach_{arm}_{model}")
            if not part or "ridge" not in part:
                ax.set_axis_off()
                continue
            ks = sorted(int(k) for k in part["ridge"])
            ridge = [part["ridge"][str(k)]["r2"] for k in ks]
            null_hi = [
                part["ridge"][str(k)].get("null_max") or part["ridge"][str(k)].get("null_hi")
                for k in ks
            ]
            ax.plot(ks, ridge, marker="o", ms=3, color=colors[0], label="ridge (ambient)", lw=1.3)
            mk = sorted(int(k) for k in part.get("mlp", {}))
            if mk:
                mlp = [part["mlp"][str(k)]["r2"] for k in mk]
                mnull = [part["mlp"][str(k)].get("null_max") for k in mk]
                ax.plot(
                    ks[: len(mk)],
                    mlp,
                    marker="s",
                    ms=3,
                    color=colors[1],
                    label="MLP (PCA-48)",
                    lw=1.3,
                )
                finite = [
                    (k, m, n) for k, m, n in zip(mk, mlp, mnull, strict=True) if n is not None
                ]
                kstar = next((k for k, m, n in finite if m <= n), None)
                if kstar is not None:
                    ax.axvspan(kstar - 1, kstar + 1, color=colors[1], alpha=0.15, label="k* band")
            if any(v is not None for v in null_hi):
                ax.plot(
                    ks,
                    [v if v is not None else np.nan for v in null_hi],
                    color="grey",
                    ls="--",
                    lw=1.0,
                    label="ridge null max",
                )
            ax.set_title(f"{model} — {arm}")
            ax.set_xlabel("answer turn k")
        axes[0].set_ylabel(r"held-out $R^2$ (context$_1 \to$ answer$_k$)")
        axes[0].legend(fontsize=6)
        _save(
            fig,
            fig_dir,
            f"turndyn_f4_reach_{arm}",
            f"Turn-1 context reach ({arm}): ridge vs MLP R2 predicting turn-k answers from "
            "the turn-1 context state; k* = first MLP crossing into its shuffle band, "
            "reported as a band (k*-1, k*+1), not a point.",
            payload["git_commit"],
        )


def fig_pooled(payload: dict, fig_dir: Path) -> None:
    colors = paper_palette(3)
    arms = ("armG", "armR_own", "armR_logged")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        xs, pooled, mean_pt, cis = [], [], [], []
        for ai, arm in enumerate(arms):
            part = payload["parts"].get(f"cells_{arm}_{model}", {})
            node = part.get("pooled_ctx_L19") or {}
            if not node:
                continue
            xs.append(ai)
            pooled.append(node["r2"])
            mean_pt.append(node.get("mean_per_turn_r2"))
            cis.append(node.get("r2_ci") or [np.nan, np.nan])
        if not xs:
            ax.set_axis_off()
            continue
        w = 0.35
        yerr = np.array(
            [
                [max(0.0, p - ci[0]) for p, ci in zip(pooled, cis, strict=True)],
                [max(0.0, ci[1] - p) for p, ci in zip(pooled, cis, strict=True)],
            ]
        )
        ax.bar(
            [x - w / 2 for x in xs], pooled, w, yerr=yerr, color=colors[0], label="pooled all-turns"
        )
        ax.bar(
            [x + w / 2 for x in xs],
            [m if m is not None else np.nan for m in mean_pt],
            w,
            color=colors[1],
            label="mean per-turn",
        )
        ax.set_xticks(xs, [arms[i] for i in xs], fontsize=6)
        ax.set_title(model)
    axes[0].set_ylabel(r"held-out $R^2$ (ctx, L19)")
    axes[0].legend(fontsize=6)
    _save(
        fig,
        fig_dir,
        "turndyn_f5_pooled",
        "Pooled-all-turns fit vs the mean of per-turn fits (Simpson's check); error bars = "
        "conversation-bootstrap 95% CI on the pooled fit.",
        payload["git_commit"],
    )


def fig_bridge(payload: dict, fig_dir: Path) -> None:
    colors = paper_palette(2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    band = None
    for ax, model in zip(axes, MODELS, strict=True):
        node = payload.get("bridge_H4", {}).get(model) or {}
        per_turn = node.get("per_turn") or {}
        if not per_turn:
            ax.set_axis_off()
            continue
        band = node.get("band", 0.10)
        ts = sorted(int(t) for t in per_turn)
        deltas = [per_turn[str(t)]["delta"] for t in ts]
        los = [per_turn[str(t)]["delta_ci"][0] for t in ts]
        his = [per_turn[str(t)]["delta_ci"][1] for t in ts]
        ax.axhspan(-band, band, color="grey", alpha=0.15, label=f"±{band} comparability band")
        ax.errorbar(
            ts,
            deltas,
            yerr=[
                [max(0.0, d - lo) for d, lo in zip(deltas, los, strict=True)],
                [max(0.0, hi - d) for d, hi in zip(deltas, his, strict=True)],
            ],
            fmt="o",
            ms=3,
            color=colors[0],
            lw=1,
        )
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_title(model)
        ax.set_xlabel("assistant turn t (bridged depths)")
    axes[0].set_ylabel(r"$R^2_{armG} - R^2_{armR\,own}$ (seed intersection)")
    axes[0].legend(fontsize=6)
    _save(
        fig,
        fig_dir,
        "turndyn_f6_bridge",
        "H4 bridge: per-depth simulated-vs-real R2 contrast on the shared-seed intersection "
        "(the both-windows-surviving conversations), conversation-bootstrap 95% CI.",
        payload["git_commit"],
    )


def fig_diagnostics(payload: dict, fig_dir: Path) -> None:
    colors = paper_palette(3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        diag = payload.get("rollout_diagnostics", {}).get(model) or {}
        per_depth = diag.get("per_depth") or {}
        if not per_depth:
            ax.set_axis_off()
            continue
        ks = sorted(int(k) for k in per_depth)
        ax.plot(
            ks,
            [per_depth[str(k)]["distinct2"] for k in ks],
            marker="o",
            ms=3,
            color=colors[0],
            label="user-turn distinct-2",
        )
        ax.plot(
            ks,
            [per_depth[str(k)].get("max_crossturn_cosine_p90") or np.nan for k in ks],
            marker="s",
            ms=3,
            color=colors[1],
            label="cross-turn cosine p90",
        )
        ax.plot(
            ks,
            [per_depth[str(k)]["role_leak_rate"] for k in ks],
            marker="^",
            ms=3,
            color=colors[2],
            label="role-leak rate",
        )
        ax.set_title(model)
        ax.set_xlabel("user turn k")
    axes[0].set_ylabel("diagnostic value")
    axes[0].legend(fontsize=6)
    _save(
        fig,
        fig_dir,
        "turndyn_f7_degeneracy",
        "Simulated-user degeneracy diagnostics per depth on the full rollout (weighed before "
        "any deep-turn claim): lexical diversity, within-conversation repetition, role leaks.",
        payload["git_commit"],
    )


def fig_exploratory(payload: dict, fig_dir: Path) -> None:
    colors = paper_palette(4)
    # n(k) yield curve
    nk = (payload.get("harvest_report_digest") or {}).get("nk_table") or {}
    if nk:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ks = sorted(int(k) for k in nk)
        ax.semilogy(ks, [max(1, nk[str(k)]) for k in ks], marker="o", ms=3, color=colors[0])
        kr = (payload.get("harvest_report_digest") or {}).get("K_real")
        if kr:
            ax.axvline(int(kr), color="crimson", lw=1, label=f"K_real={kr}")
        ax.axhline(5000, color="grey", ls="--", lw=0.8, label="n=5,000 target")
        ax.set_xlabel(">= k user turns")
        ax.set_ylabel("kept conversations (log)")
        ax.legend(fontsize=6)
        _save(
            fig,
            fig_dir,
            "turndyn_expl_nk_yield",
            "Realized full-stream depth-yield curve n(k) with the G-A K_real selection.",
            payload["git_commit"],
        )
    # prefix-arm panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        for ci, arm in enumerate(("armG", "armR_own", "armR_logged")):
            ts, r2s, _ = _curve(_cells(payload, arm, model), arm_x="pfx")
            if ts.size:
                ax.plot(ts, r2s, marker="o", ms=3, color=paper_palette(3)[ci], label=arm, lw=1.2)
        ax.set_title(f"{model} — prefix arm")
        ax.set_xlabel("assistant turn t")
    axes[0].set_ylabel(r"held-out $R^2$ (prefix$_k \to$ answer, L19)")
    axes[0].legend(fontsize=6)
    _save(
        fig,
        fig_dir,
        "turndyn_expl_prefix",
        "Prefix-arm (everything before the user query) per-turn map strength — the standing "
        "prefix+context dual-mapping read; prefix at t1 is structurally degenerate (omitted).",
        payload["git_commit"],
    )
    # G-C parity overlay
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        part = payload["parts"].get(f"gc_{model}") or {}
        per_turn = part.get("per_turn") or {}
        if not per_turn:
            ax.set_axis_off()
            continue
        ts = sorted(int(t) for t in per_turn)
        refit = [per_turn[str(t)]["r2_refit"] for t in ts]
        r10 = [per_turn[str(t)]["r2_round10"] for t in ts]
        los = [per_turn[str(t)]["r2_refit_ci"][0] for t in ts]
        his = [per_turn[str(t)]["r2_refit_ci"][1] for t in ts]
        ax.fill_between(ts, los, his, color=colors[0], alpha=0.2, label="refit 95% CI")
        ax.plot(ts, refit, marker="o", ms=3, color=colors[0], label="this round (id-matched refit)")
        ax.plot(ts, r10, marker="s", ms=3, color=colors[1], ls="--", label="round-10 banked")
        ax.set_title(f"{model} — G-C {'PASS' if part.get('pass') else 'FAIL'}")
        ax.set_xlabel("assistant turn t")
    axes[0].set_ylabel(r"held-out $R^2$ (ctx logged, L19)")
    axes[0].legend(fontsize=6)
    _save(
        fig,
        fig_dir,
        "turndyn_expl_gc_parity",
        "G-C pipeline-parity overlay on the EXACT round-10 conversation-id set: this round's "
        "refit (with conversation-bootstrap CI) vs the banked round-10 logged curve.",
        payload["git_commit"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-json",
        default=str(REPO_ROOT / "eval_results/issue_825/turn_dynamics/results.json"),
    )
    ap.add_argument("--fig-dir", default=str(REPO_ROOT / "figures/issue_825"))
    args = ap.parse_args()
    with open(args.results_json) as f:
        payload = json.load(f)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()
    fig_hero(payload, fig_dir)
    fig_transfer(payload, fig_dir)
    fig_operators(payload, fig_dir)
    fig_reach(payload, fig_dir)
    fig_pooled(payload, fig_dir)
    fig_bridge(payload, fig_dir)
    fig_diagnostics(payload, fig_dir)
    fig_exploratory(payload, fig_dir)
    print("[figures] done")


if __name__ == "__main__":
    main()
