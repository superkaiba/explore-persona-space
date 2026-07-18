"""Issue #1072 ``lowdim-token-subspace`` phase E — figures from stats_lowdim.json.

Outputs (figures/issue_1072/lowdim-token-subspace/):
  hero_lowdim_decomposition.png — per-basis stacked decomposition of the L26
      own vs external (plain) gap into parallel / orthogonal / cross parts,
      D_k CI whiskers, the completed run's 1-D result as the leftmost
      reference (plan §6 hero).
  sparallel_profile.png — S_par(k) and enrichment(k) across the basis panel
      at both layers (the dimensionality-of-commitment profile).
  percontext_top32_L26.png — LOW-LEVEL per-unit plot: per-context paired
      (parallel vs orthogonal) contribution differences for the top-32 basis
      at the primary layer, scatter + marginal histograms.
  exploratory_lowdim.png — over-produced dump: variance shares, per-basis
      closure, realized-token coverage, lookahead effective-size / overlap
      histograms, per-position slot profiles, λ-sensitivity.

Reader-facing text uses plain-English condition names only (no opaque slugs).
Errorbar offsets are non-negative (max(0, v-lo) / max(0, hi-v)) per the
standing figure rule.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE any heavy import — shared-VM thread caps (#847)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from scripts.issue1072_stats import CH, _safe_ratio  # noqa: E402
from scripts.issue1072_lowdim_stats import K_FOLDS, T2  # noqa: E402

logger = logging.getLogger("issue1072.lowdim.figures")

COMP_ORDER = ("par", "perp", "cross")
COMP_LABELS = {
    "par": "inside the token subspace",
    "perp": "outside the token subspace",
    "cross": "cross term",
}
BASIS_LABELS = {
    "top8": "top-8 candidate words",
    "top32": "top-32 candidate words",
    "look8": "realized next 8 words",
}
ARM_LABELS = {
    "own": "own answer",
    "ext_plain": "external answer (plain)",
    "ext_style": "external answer (distinct style)",
    "mismatch": "mismatched answer (floor)",
}
HERO_ORDER = ("top8", "look8", "top32")  # ascending bundle size (look8 <= 8 effective)


def _err(lo: float, v: float, hi: float) -> tuple[float, float]:
    """Non-negative errorbar OFFSETS (never CI bounds/signed deltas — gotchas.md)."""
    return max(0.0, v - lo), max(0.0, hi - v)


def _load(eval_dir: pathlib.Path) -> tuple[dict, dict[int, dict]]:
    stats = json.loads((eval_dir / "stats_lowdim.json").read_text())
    npzs = {
        k: dict(np.load(eval_dir / f"per_context_stats_lowdim_fold{k}.npz", allow_pickle=False))
        for k in range(K_FOLDS)
    }
    return stats, npzs


def fig_hero(stats: dict, out_dir: pathlib.Path) -> None:
    primary = str(stats["primary_layer"])
    parent = stats.get("parent_1d_reference")
    cols: list[tuple[str, dict, dict | None]] = []
    if parent is not None and primary in parent["by_layer"]:
        ref = parent["by_layer"][primary]
        cols.append(
            (
                "realized next word\n(1 direction, completed run)",
                ref["delta_C"],
                {"D": ref["D"], "D_ci95": ref["D_ci95"]},
            )
        )
    for b in HERO_ORDER:
        rec = stats["by_ext"]["ext_plain"][b][primary]
        cols.append((BASIS_LABELS[b].replace(" words", "\nwords"), rec["delta_C"], rec))
    colors = paper_palette(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    x = np.arange(len(cols), dtype=float)
    bottom_pos = np.zeros(len(cols))
    bottom_neg = np.zeros(len(cols))
    for ci, comp in enumerate(COMP_ORDER):
        vals = np.array([c[1][comp] for c in cols])
        bottoms = np.where(vals >= 0, bottom_pos, bottom_neg)
        ax.bar(x, vals, 0.55, bottom=bottoms, color=colors[ci], label=COMP_LABELS[comp])
        bottom_pos += np.where(vals >= 0, vals, 0.0)
        bottom_neg += np.where(vals < 0, vals, 0.0)
    d_vals = [c[2]["D"] for c in cols]
    d_err = np.array([_err(c[2]["D_ci95"][0], c[2]["D"], c[2]["D_ci95"][1]) for c in cols]).T
    ax.errorbar(
        x + 0.32,
        d_vals,
        yerr=d_err,
        fmt="D",
        color="black",
        ms=4,
        capsize=3,
        label="D = inside minus outside (95% CI)",
    )
    ax.axhline(0.0, lw=0.8, color="gray")
    ax.set_xticks(x, [c[0] for c in cols], fontsize=8)
    ax.set_ylabel("own minus external (plain) gap contribution")
    ax.set_title("Does a low-dimensional token subspace carry the own-answer advantage?")
    ax.legend(fontsize=8)
    savefig_paper(fig, "hero_lowdim_decomposition", dir=out_dir)
    plt.close(fig)


def fig_profile(stats: dict, eval_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    layers = sorted(int(x) for x in stats["inputs"]["layers"])
    parent = stats.get("parent_1d_reference")
    # Parent 1-D enrichment (k=1 reference points) from the completed round's
    # committed supplementary reads (same eval_results/issue_1072/ tree, one level up).
    supp_path = eval_dir.parent / "supplementary_reads.json"
    parent_enrich: dict[str, dict] = {}
    if supp_path.exists():
        parent_enrich = json.loads(supp_path.read_text()).get("by_layer", {})
    else:
        logger.warning("[figures] parent supplementary_reads.json missing at %s", supp_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), layout="constrained")
    labels = ["realized next word\n(1 direction)"] + [
        BASIS_LABELS[b].replace(" words", "\nwords") for b in HERO_ORDER
    ]
    x = np.arange(len(labels), dtype=float)
    colors = paper_palette(len(layers))
    for li, layer in enumerate(layers):
        s_vals, s_err, e_vals, e_err = [], [], [], []
        if parent is not None and str(layer) in parent["by_layer"]:
            ref = parent["by_layer"][str(layer)]
            s_vals.append(ref["S_par"])
            s_err.append(_err(ref["S_par_ci95"][0], ref["S_par"], ref["S_par_ci95"][1]))
        else:
            s_vals.append(np.nan)
            s_err.append((0, 0))
        pref = parent_enrich.get(str(layer), {})
        if "E_enrichment" in pref:
            e_vals.append(pref["E_enrichment"])
            e_err.append(_err(pref["E_ci95"][0], pref["E_enrichment"], pref["E_ci95"][1]))
        else:
            e_vals.append(np.nan)
            e_err.append((0, 0))
        for b in HERO_ORDER:
            rec = stats["by_ext"]["ext_plain"][b][str(layer)]
            s_vals.append(rec["S_par"] if rec["S_par"] is not None else np.nan)
            s_err.append(
                _err(rec["S_par_ci95"][0], rec["S_par"] or 0.0, rec["S_par_ci95"][1])
                if rec["S_par"] is not None
                else (0, 0)
            )
            e_vals.append(rec["enrichment"] if rec["enrichment"] is not None else np.nan)
            e_err.append(
                _err(rec["enrichment_ci95"][0], rec["enrichment"], rec["enrichment_ci95"][1])
                if rec.get("enrichment") is not None
                else (0, 0)
            )
        axes[0].errorbar(
            x + 0.05 * li,
            s_vals,
            yerr=np.array(s_err).T,
            fmt="o-",
            ms=4,
            capsize=3,
            color=colors[li],
            label=f"layer {layer}",
        )
        axes[1].errorbar(
            x + 0.05 * li,
            e_vals,
            yerr=np.array(e_err).T,
            fmt="o-",
            ms=4,
            capsize=3,
            color=colors[li],
            label=f"layer {layer}",
        )
    axes[0].axhline(0.15, lw=0.8, ls="--", color="gray")
    axes[0].text(
        0.02, 0.155, "15% expectation bound (set before the run)", fontsize=7, color="gray"
    )
    axes[0].set_ylabel("share of the gap inside the subspace (S_par)")
    axes[1].set_ylabel("gap share / variance share (enrichment)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=7)
        ax.legend(fontsize=8)
    fig.suptitle("Dimensionality-of-commitment profile across basis bundles")
    savefig_paper(fig, "sparallel_profile", dir=out_dir)
    plt.close(fig)


def _percontext_deltas(npzs: dict[int, dict], basis: str, layer: int, comp: str) -> np.ndarray:
    """Per-context contribution difference (own - external plain), one basis."""
    out = []
    for k in sorted(npzs):
        vals = {}
        for arm in ("own", "ext_plain"):
            ch = npzs[k][f"{basis}_M{T2}c_L{layer}|{arm}"].astype(np.float64)
            num = (
                ch[:, CH[f"ss_tot_{comp}"]] - ch[:, CH[f"ss_res_{comp}"]]
                if comp in ("par", "perp")
                else ch[:, CH["cross_tot"]] - ch[:, CH["cross_res"]]
            )
            vals[arm] = _safe_ratio(num, ch[:, CH["ss_tot_full"]])
        out.append(vals["own"] - vals["ext_plain"])
    return np.concatenate(out)


def fig_percontext(stats: dict, npzs: dict[int, dict], out_dir: pathlib.Path) -> None:
    layer = int(stats["primary_layer"])
    d_par = _percontext_deltas(npzs, "top32", layer, "par")
    d_perp = _percontext_deltas(npzs, "top32", layer, "perp")
    m = np.isfinite(d_par) & np.isfinite(d_perp)
    fig = plt.figure(figsize=(6.0, 6.0), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4))
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax.scatter(d_par[m], d_perp[m], s=4, alpha=0.25, color=paper_palette(1)[0])
    ax.axhline(0, lw=0.6, color="gray")
    ax.axvline(0, lw=0.6, color="gray")
    ax.set_xlabel("per-context gap inside the top-32 candidate subspace")
    ax.set_ylabel("per-context gap outside the subspace")
    axx.hist(d_par[m], bins=60, color=paper_palette(1)[0])
    axy.hist(d_perp[m], bins=60, orientation="horizontal", color=paper_palette(1)[0])
    axx.tick_params(labelbottom=False)
    axy.tick_params(labelleft=False)
    fig.suptitle(
        f"Per-context paired contribution differences, own minus external (plain) "
        f"(n={int(m.sum())} matched contexts)"
    )
    savefig_paper(fig, "percontext_top32_L26", dir=out_dir)
    plt.close(fig)


def fig_exploratory(  # noqa: C901 — the over-produced dump panel
    stats: dict, npzs: dict[int, dict], out_dir: pathlib.Path
) -> None:
    layers = sorted(int(x) for x in stats["inputs"]["layers"])
    primary = str(stats["primary_layer"])
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), layout="constrained")
    colors = paper_palette(4)

    # (a) variance share inside each subspace, by arm (primary layer).
    ax = axes[0][0]
    x = np.arange(len(HERO_ORDER), dtype=float)
    for ai, arm in enumerate(("own", "ext_plain", "ext_style", "mismatch")):
        vals = [stats["by_ext"]["ext_plain"][b][primary]["w_par_by_arm"][arm] for b in HERO_ORDER]
        ax.plot(x + 0.04 * ai, vals, "o-", ms=3, color=colors[ai], label=ARM_LABELS[arm])
    ax.set_yscale("log")
    ax.set_xticks(x, [BASIS_LABELS[b] for b in HERO_ORDER], fontsize=7)
    ax.set_ylabel("variance share inside the subspace")
    ax.set_title("Token-subspace variance share by answer arm")
    ax.legend(fontsize=7)

    # (b) per-basis closure of the gap under the 16-token prefix (primary layer).
    ax = axes[0][1]
    width = 0.25
    for bi, b in enumerate(HERO_ORDER):
        rec = stats["by_ext"]["ext_plain"][b][primary]["closure_by_component"]
        g0 = [rec[c]["G0"] for c in ("par", "perp")]
        gt = [rec[c]["Gt"] for c in ("par", "perp")]
        xb = np.arange(2) + (bi - 1) * width
        ax.bar(xb - width / 4, g0, width / 2, color=colors[bi], label=BASIS_LABELS[b])
        ax.bar(xb + width / 4, gt, width / 2, color=colors[bi], alpha=0.45)
    ax.axhline(0, lw=0.6, color="gray")
    ax.set_xticks(np.arange(2), ["inside the subspace", "outside the subspace"], fontsize=8)
    ax.set_ylabel("gap without prefix (solid) vs with 16-token prefix (faded)")
    ax.set_title("Per-basis prefix closure of the gap")
    ax.legend(fontsize=7)

    # (c) realized-token coverage of the candidate sets, per arm.
    ax = axes[0][2]
    cg = stats.get("capture_gates_summary") or {}
    cov = cg.get("coverage") or {}
    if cov:
        arms = [a for a in ("own", "ext_plain", "ext_style", "mismatch") if a in cov]
        r8 = [cov[a]["top8_hits"] / max(cov[a]["n_positions"], 1) for a in arms]
        r32 = [cov[a]["top32_hits"] / max(cov[a]["n_positions"], 1) for a in arms]
        xa = np.arange(len(arms), dtype=float)
        ax.bar(xa - 0.18, r8, 0.34, color=colors[0], label="realized word in top-8")
        ax.bar(xa + 0.18, r32, 0.34, color=colors[1], label="realized word in top-32")
        ax.set_xticks(xa, [ARM_LABELS[a] for a in arms], fontsize=7)
        ax.set_ylabel("fraction of answer positions")
        ax.set_title("Do the candidate sets contain the realized next word?")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "coverage counters not staged", ha="center", va="center")
        ax.set_axis_off()

    # (d) lookahead window effective size + overlap with top-32.
    ax = axes[1][0]
    if cg.get("effk_hist"):
        for ai, (arm, hist) in enumerate(sorted(cg["effk_hist"].items())):
            h = np.asarray(hist, dtype=float)
            h = h / max(h.sum(), 1)
            ax.plot(
                range(len(h)), h, "o-", ms=3, color=colors[ai % 4], label=ARM_LABELS.get(arm, arm)
            )
        ax.set_xlabel("distinct words in the realized 8-word window")
        ax.set_ylabel("fraction of positions")
        ax.set_title("Lookahead window effective size (after dedupe)")
        ax.legend(fontsize=7)
    else:
        ax.set_axis_off()

    # (e) per-position slot profile at the primary layer (first 16 positions).
    ax = axes[1][1]
    for bi, b in enumerate(HERO_ORDER):
        prof = []
        for t in range(1, 17):
            try:
                d = _percontext_slot_gap(npzs, b, int(primary), f"f16_t{t}")
            except KeyError:
                d = np.nan
            prof.append(d)
        ax.plot(range(1, 17), prof, "o-", ms=3, color=colors[bi], label=BASIS_LABELS[b])
    ax.axhline(0, lw=0.6, color="gray")
    ax.set_xlabel("answer position (first 16 positions)")
    ax.set_ylabel("pooled gap inside the subspace")
    ax.set_title("Per-position subspace gap contribution")
    ax.legend(fontsize=7)

    # (f) λ-reselection sensitivity for the decision cells (own arm).
    ax = axes[1][2]
    comp = (stats.get("companions") or {}).get(primary) or {}
    lams = comp.get("sens_lambdas")
    sens = comp.get("cleg_rem_sensitivity")
    if lams and sens:
        for bi, b in enumerate(HERO_ORDER):
            key = f"{b}|own"
            if key in sens:
                c_par = [row[0] for row in sens[key]]  # C_par per λ
                ax.plot(lams, c_par, "o-", ms=3, color=colors[bi], label=BASIS_LABELS[b])
        ax.set_xscale("log")
        ax.set_xlabel("ridge regularization strength")
        ax.set_ylabel("own-answer contribution inside the subspace")
        ax.set_title("Regularization sensitivity (exploratory)")
        ax.legend(fontsize=7)
    else:
        ax.set_axis_off()
    savefig_paper(fig, "exploratory_lowdim", dir=out_dir)
    plt.close(fig)


def _percontext_slot_gap(npzs: dict[int, dict], basis: str, layer: int, slot: str) -> float:
    """Pooled own-vs-external-plain C_par gap for one slot cell (exploratory)."""
    tot: dict[str, list[np.ndarray]] = {"own": [], "ext_plain": []}
    for k in sorted(npzs):
        for arm in tot:
            tot[arm].append(npzs[k][f"H1b_{basis}_L{layer}|{slot}|{arm}"].astype(np.float64))
    out = {}
    for arm, chunks in tot.items():
        ch = np.concatenate(chunks, axis=0)
        fm = np.isfinite(ch).all(axis=1)
        num = float((ch[fm, CH["ss_tot_par"]] - ch[fm, CH["ss_res_par"]]).sum())
        den = float(ch[fm, CH["ss_tot_full"]].sum())
        out[arm] = num / den if abs(den) > 1e-12 else np.nan
    return out["own"] - out["ext_plain"]


def main() -> None:
    p = argparse.ArgumentParser(description="Issue #1072 lowdim figures (phase E)")
    p.add_argument(
        "--eval-dir",
        type=str,
        default=str(_REPO_ROOT / "eval_results" / "issue_1072" / "lowdim-token-subspace"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_REPO_ROOT / "figures" / "issue_1072" / "lowdim-token-subspace"),
        help="figure output dir (smoke runs pass a scratch dir — never the committed path)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    set_paper_style()
    eval_dir = pathlib.Path(args.eval_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, npzs = _load(eval_dir)
    fig_hero(stats, out_dir)
    fig_profile(stats, eval_dir, out_dir)
    fig_percontext(stats, npzs, out_dir)
    fig_exploratory(stats, npzs, out_dir)
    logger.info("[figures] 4 figures written to %s", out_dir)


if __name__ == "__main__":
    main()
