"""Regenerate #779 n1m-readout figures with plain-English arm labels + flag marks.

Revision-round figure fixes (interp-critique round 1, task #779, follow-up
`n1m-nonlinear-map-behavior-readout`):

1. hero / delta forest / grouped sweep / R2 transfer: replace bare arm slugs
   ("h n5k linear", "n1m krr nystrom") with plain-English labels, and mark the
   FLAGGED layer-26 kernel arm (Nystrom-vs-exact gate gap 0.0151 > tol 0.01,
   `n1m_multilayer_fits.json .per_layer."26".nystrom_validation.gate_passed:
   false`) with a dagger on every use surface.
2. NEW `n1m_readout_l19_forest`: the persisted-but-previously-unplotted
   `.l19_continuity` read — delta vs raw (dot, 95% CI) for all five map arms in
   all six trait-mode cells, every arm read at capture layer 19.
3. (clean-result-critique round 1, Lens 3) `n1m_readout_percond_scatter_*`:
   the three per-condition scatters, regenerated with the same plain-English
   arm labels + row labels "system"/"many-shot" + a dagger on every axes whose
   cell reads the FLAGGED layer-26 kernel fit. The per-point (x, y) values are
   read back from the committed run-time sidecars (`.meta.json`, exact plotted
   float64 values); the per-point condition indices — which the sidecars do
   not carry — are rebuilt from the pass-A cell JSONs alone (no tensors),
   replicating `issue779_stage1.load_eval_cells` + `build_eval_matrix`'s row
   iteration for the metadata columns, and VALIDATED by asserting the sidecar
   y values equal the rebuilt y[mode] elementwise per series (atol 1e-9).

Reads the committed round JSONs + figure sidecars (+ pass-A cell JSONs for the
scatter regen) only; no recomputation. Fail loud.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps land BEFORE numpy import on the shared VM.
load_dotenv()

import numpy as np  # noqa: E402

MODES = ("system", "many_shot")
TRAITS = ("evil", "sycophancy", "hallucination")
MAP_ARMS = ("h_n5k_linear", "n1m_ridge", "n1m_mlp_w8192", "n1m_mlp_w32768", "n1m_krr_nystrom")
SUBSTITUTED_CELLS = {"hallucination/system", "hallucination/many_shot"}
FLAGGED_LAYER = 26  # L26 kernel arm failed the Nystrom-vs-exact gate

ARM_LABELS = {
    "pv_raw": "raw projection",
    "h_n5k_linear": "5k linear map",
    "n1m_ridge": "963k ridge",
    "n1m_mlp_w8192": "963k MLP (w=8192)",
    "n1m_mlp_w32768": "963k MLP (w=32768)",
    "n1m_krr_nystrom": "963k kernel (Nyström)",
    "oracle": "oracle (true answer proj.)",
    "h_n5k_logo": "5k map (LOGO refit)",
    "pv_raw_group": "raw projection",
}
FITTER_LABELS = {
    "ridge": "963k ridge",
    "mlp_w8192": "963k MLP (w=8192)",
    "mlp_w32768": "963k MLP (w=32768)",
    "krr_nystrom": "963k kernel (Nyström)",
}


def _label(arm: str, layer: int) -> str:
    base = ARM_LABELS[arm]
    if arm == "n1m_krr_nystrom" and layer == FLAGGED_LAYER:
        return base + " †"
    return base


SCATTER_ARM_ORDER = ("pv_raw", "h_n5k_linear", *MAP_ARMS[1:], "oracle")
MODE_LABELS = {"system": "system", "many_shot": "many-shot"}


def rebuild_eval_rows(pass_a_dir: Path, trait: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, cond, mode) row arrays in ``build_eval_matrix`` order, from JSON only.

    Replicates ``issue779_stage1.load_eval_cells`` (cells sorted by filename)
    + ``build_eval_matrix``'s row loop (questions in first-seen rollout order,
    rollouts with a resolvable judge score, questions with zero valid scores
    dropped) for the metadata columns — no ``_cx.pt`` tensor loads.
    """
    y: list[float] = []
    cond: list[int] = []
    mode: list[str] = []
    cond_map: dict[str, int] = {}
    cell_paths = sorted(pass_a_dir.glob(f"{trait}__*.json"))
    if not cell_paths:
        raise FileNotFoundError(f"no pass-A cell JSONs for {trait} under {pass_a_dir}")
    for cp in cell_paths:
        cell = json.loads(cp.read_text())
        cid = cell["cond_id"]
        cond_map.setdefault(cid, len(cond_map))
        by_q: dict[int, list[dict]] = {}
        for rec in cell["rollouts"]:
            if rec.get("empty"):
                continue
            by_q.setdefault(rec["qi"], []).append(rec)
        smap: dict[tuple[int, int], float] = {}
        for cid_key, s in cell["judge_scores"].items():
            parts = cid_key.split("__")
            if len(parts) < 3 or s is None:
                continue
            try:
                smap[(int(parts[-2]), int(parts[-1]))] = float(s)
            except ValueError:
                continue
        for qi, recs in by_q.items():
            q_scores = [smap[(qi, r["ri"])] for r in recs if (qi, r["ri"]) in smap]
            if not q_scores:
                continue
            y.append(float(np.mean(q_scores)))
            cond.append(cond_map[cid])
            mode.append(cell["mode"])
    return np.array(y), np.array(cond), np.array(mode, dtype=object)


def make_percond_scatters(
    res: dict, fits: dict, pass_a_dir: Path, sidecar_dir: Path, fig_dir: str
) -> dict[str, str]:
    """Regenerate the three per-condition scatters (plain labels + daggers)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    gate26 = fits["per_layer"]["26"]["nystrom_validation"]
    assert gate26["gap"] > gate26["tol"], "L26 Nystrom gate no longer failed — dagger is stale"
    figs: dict[str, str] = {}
    n_arms = len(SCATTER_ARM_ORDER)
    for trait in TRAITS:
        # Old sidecar: exact plotted per-point values, series `_group` = row*7+col.
        meta = json.loads(
            (sidecar_dir / f"n1m_readout_percond_scatter_{trait}.meta.json").read_text()
        )
        by_group: dict[int, list[dict]] = {}
        for p in meta["points"]:
            by_group.setdefault(int(p["_group"]), []).append(p)
        y_all, cond_all, mode_all = rebuild_eval_rows(pass_a_dir, trait)

        fig, axes = plt.subplots(
            2, n_arms, figsize=(2.6 * n_arms, 6.0), squeeze=False, layout="tight"
        )
        for row, mode in enumerate(MODES):
            msel = np.asarray([m == mode for m in mode_all])
            y_row, cond_row = y_all[msel], cond_all[msel]
            layer = int(res["headline"][trait][mode]["layer"])
            for coli, arm in enumerate(SCATTER_ARM_ORDER):
                ax = axes[row][coli]
                pts = by_group[row * n_arms + coli]
                ykey = next(k for k in pts[0] if k not in ("x", "_kind", "_group"))
                x = np.array([p["x"] for p in pts])
                y_sc = np.array([p[ykey] for p in pts])
                # Validation: sidecar order == rebuilt mat[msel] order, exactly.
                assert len(y_sc) == len(y_row) and np.allclose(y_sc, y_row, atol=1e-9), (
                    f"{trait}/{mode}/{arm}: sidecar y does not match rebuilt rows"
                )
                ax.scatter(x, y_row, c=cond_row, cmap="tab20", s=6, alpha=0.7)
                if row == 0:
                    ax.set_title(ARM_LABELS[arm], fontsize=7)
                if coli == 0:
                    ax.set_ylabel(f"{MODE_LABELS[mode]}\njudge score", fontsize=7)
                if arm == "n1m_krr_nystrom" and layer == FLAGGED_LAYER:
                    ax.text(
                        0.95,
                        0.95,
                        "†",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=11,
                        color="black",
                    )
                ax.tick_params(labelsize=6)
        fig.suptitle(f"{trait}: per-condition monitor-vs-score scatter (dot readout)")
        out = savefig_paper(fig, f"n1m_readout_percond_scatter_{trait}", dir=fig_dir)
        plt.close(fig)
        figs[f"percond_scatter_{trait}"] = str(out.get("png", ""))
    return figs


def make_figures(res: dict, fits: dict, fig_dir: str) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    # Guard the dagger semantics: the L26 kernel gate must actually be failed.
    gate26 = fits["per_layer"]["26"]["nystrom_validation"]
    assert gate26["gap"] > gate26["tol"], "L26 Nystrom gate no longer failed — dagger is stale"
    figs: dict[str, str] = {}
    colors = paper_palette(3)
    bar_arms = ["pv_raw", *MAP_ARMS, "oracle"]
    bar_colors = [colors[0], colors[1], colors[1]] + [colors[2]] * 4 + [colors[0]]

    # HERO: grouped bars of within-condition r across the arms (dot readout).
    fig, axes = plt.subplots(
        2, len(TRAITS), figsize=(4.6 * len(TRAITS) + 0.6, 8.6), squeeze=False, layout="tight"
    )
    for col, trait in enumerate(TRAITS):
        for row, mode in enumerate(MODES):
            ax = axes[row][col]
            entry = res["headline"][trait][mode]
            layer = int(entry["layer"])
            heights, errs, labels = [], [], []
            for arm in bar_arms:
                name = arm if arm in ("pv_raw", "oracle") else f"{arm}_dot"
                mm = entry["monitors"][name]
                pt = mm["point"]
                if not np.isfinite(pt):
                    continue
                heights.append(pt)
                lo, hi = mm["lo"], mm["hi"]
                errs.append(
                    [
                        max(0.0, pt - lo) if np.isfinite(lo) else 0.0,
                        max(0.0, hi - pt) if np.isfinite(hi) else 0.0,
                    ]
                )
                labels.append(_label(arm, layer))
            ax.bar(
                range(len(heights)),
                heights,
                yerr=np.array(errs).T if errs else None,
                capsize=2,
                color=bar_colors[: len(heights)],
            )
            ax.axhline(0.0, color="gray", lw=0.6)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            mode_lbl = "system prompting" if mode == "system" else "many-shot"
            sub = ", substitute layer" if f"{trait}/{mode}" in SUBSTITUTED_CELLS else ""
            ax.set_title(f"{trait} — {mode_lbl} (L{layer}{sub})")
            if col == 0:
                ax.set_ylabel("within-condition Pearson r (dot readout)")
    out = savefig_paper(fig, "n1m_readout_hero", dir=fig_dir)
    plt.close(fig)
    figs["hero"] = str(out.get("png", ""))

    # Delta-vs-raw forest (dot readout) with the +0.05 bar, frozen headline layers.
    fig, axes = plt.subplots(
        1, len(TRAITS), figsize=(4.8 * len(TRAITS), 5.2), squeeze=False, layout="tight"
    )
    for col, trait in enumerate(TRAITS):
        ax = axes[0][col]
        ypos, ylab = 0, []
        for mode in MODES:
            entry = res["headline"][trait][mode]
            layer = int(entry["layer"])
            mode_lbl = "system" if mode == "system" else "many-shot"
            for arm in MAP_ARMS:
                d = entry["deltas_vs_pv_raw"][f"{arm}_dot"]
                ax.errorbar(
                    d["delta"],
                    ypos,
                    xerr=[[max(0.0, d["delta"] - d["lo"])], [max(0.0, d["hi"] - d["delta"])]],
                    fmt="o",
                    capsize=2,
                    color=colors[0] if mode == "system" else colors[1],
                )
                ylab.append(f"{mode_lbl}: {_label(arm, layer)}")
                ypos += 1
            ypos += 1
            ylab.append("")
        ax.axvline(0.0, color="gray", lw=0.8)
        ax.axvline(0.05, color=colors[2], lw=1.0, ls="--", label="+0.05 bar")
        ax.set_yticks(range(len(ylab)))
        ax.set_yticklabels(ylab, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("delta within-condition r vs raw")
        ax.set_title(trait)
        if col == 0:
            ax.legend(fontsize=7, loc="lower right")
    out = savefig_paper(fig, "n1m_readout_delta_forest", dir=fig_dir)
    plt.close(fig)
    figs["delta_forest"] = str(out.get("png", ""))

    # NEW: layer-19 continuity forest — same delta layout, every arm read at L19.
    fig, axes = plt.subplots(
        1, len(TRAITS), figsize=(4.8 * len(TRAITS), 5.2), squeeze=False, layout="tight"
    )
    for col, trait in enumerate(TRAITS):
        ax = axes[0][col]
        ypos, ylab = 0, []
        for mode in MODES:
            entry = res["l19_continuity"][trait][mode]
            mode_lbl = "system" if mode == "system" else "many-shot"
            for arm in MAP_ARMS:
                d = entry["deltas_vs_pv_raw"][f"{arm}_dot"]
                ax.errorbar(
                    d["delta"],
                    ypos,
                    xerr=[[max(0.0, d["delta"] - d["lo"])], [max(0.0, d["hi"] - d["delta"])]],
                    fmt="o",
                    capsize=2,
                    color=colors[0] if mode == "system" else colors[1],
                )
                ylab.append(f"{mode_lbl}: {_label(arm, 19)}")
                ypos += 1
            ypos += 1
            ylab.append("")
        ax.axvline(0.0, color="gray", lw=0.8)
        ax.axvline(0.05, color=colors[2], lw=1.0, ls="--", label="+0.05 bar")
        ax.set_yticks(range(len(ylab)))
        ax.set_yticklabels(ylab, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("delta r vs raw (all arms at L19)")
        ax.set_title(trait)
        if col == 0:
            ax.legend(fontsize=7, loc="lower right")
    out = savefig_paper(fig, "n1m_readout_l19_forest", dir=fig_dir)
    plt.close(fig)
    figs["l19_forest"] = str(out.get("png", ""))

    # Grouped sweep: r vs group size, parent LOGO vs fixed arms.
    sweep_arms = [
        "h_n5k_logo",
        "h_n5k_linear",
        "n1m_ridge",
        "n1m_mlp_w8192",
        "n1m_mlp_w32768",
        "n1m_krr_nystrom",
        "pv_raw_group",
    ]
    pal = paper_palette(max(3, len(sweep_arms)))
    fig, axes = plt.subplots(
        1, len(TRAITS), figsize=(5.0 * len(TRAITS), 4.6), squeeze=False, layout="tight"
    )
    for col, trait in enumerate(TRAITS):
        ax = axes[0][col]
        glayer = int(res["grouped"][trait]["layer"])
        d = res["grouped"][trait]["group_size_sweep"]
        sizes = sorted(int(s) for s in d)
        for ai, arm in enumerate(sweep_arms):
            means = [d[str(s)][arm]["dot_r_mean"] for s in sizes]
            sds = [d[str(s)][arm]["dot_r_sd"] for s in sizes]
            ax.errorbar(
                sizes,
                means,
                yerr=sds,
                marker="o",
                ms=3,
                capsize=2,
                color=pal[ai % len(pal)],
                label=_label(arm, glayer),
            )
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("questions averaged per persona group")
        ax.set_ylabel("Pearson r vs mean judge score (dot)")
        ax.set_title(f"{trait} (L{glayer})")
        ax.legend(fontsize=6, loc="lower right")
    out = savefig_paper(fig, "n1m_readout_grouped_sweep", dir=fig_dir)
    plt.close(fig)
    figs["grouped_sweep"] = str(out.get("png", ""))

    # Fit-quality transfer: held-out test R2 at each layer per fitter.
    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout="tight")
    r2 = {int(k): v for k, v in res["fit_quality"]["per_layer_test_r2"].items()}
    layers = sorted(r2)
    pal = paper_palette(max(3, len(FITTER_LABELS)))
    for fi, fitter in enumerate(("ridge", "mlp_w8192", "mlp_w32768", "krr_nystrom")):
        ax.plot(
            layers,
            [r2[li][fitter] for li in layers],
            marker="o",
            color=pal[fi % len(pal)],
            label=FITTER_LABELS[fitter],
        )
    # Dagger the flagged point: the kernel arm failed its Nystrom-vs-exact gate at L26 only.
    ax.text(
        FLAGGED_LAYER,
        r2[FLAGGED_LAYER]["krr_nystrom"] - 0.006,
        "†",
        ha="center",
        va="top",
        fontsize=10,
        color=pal[3 % len(pal)],
    )
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out whole-map R2 (pinned test)")
    ax.legend(fontsize=7, loc="lower right")
    out = savefig_paper(fig, "n1m_readout_r2_transfer", dir=fig_dir)
    plt.close(fig)
    figs["r2_transfer"] = str(out.get("png", ""))
    return figs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir", required=True, help="round eval_results dir with the two JSONs"
    )
    ap.add_argument("--fig-dir", required=True, help="output figures dir (figures/issue_779)")
    ap.add_argument(
        "--percond-pass-a-dir",
        type=Path,
        default=None,
        help="pass-A cell-JSON dir; when given, regenerate the per-condition scatters",
    )
    ap.add_argument(
        "--skip-main-figs",
        action="store_true",
        help="skip the hero/forest/grouped/r2 regeneration (scatters only)",
    )
    args = ap.parse_args()

    rd = Path(args.results_dir)
    res = json.loads((rd / "n1m_readout.json").read_text())
    fits = json.loads((rd / "n1m_multilayer_fits.json").read_text())
    figs: dict[str, str] = {}
    if not args.skip_main_figs:
        figs.update(make_figures(res, fits, args.fig_dir))
    if args.percond_pass_a_dir is not None:
        figs.update(
            make_percond_scatters(
                res, fits, args.percond_pass_a_dir, Path(args.fig_dir), args.fig_dir
            )
        )
    for k, v in figs.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
