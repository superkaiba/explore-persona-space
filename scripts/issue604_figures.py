# ruff: noqa: RUF003
# Intentional Unicode (Δ, σ, γ, —) in scientific docstrings + labels.
"""Task #604 figures — over-produce the §6 figure list from the Phase C JSONs.

Reads ``eval_results/issue_604/{key_match,write_match,rotation,
functional_constancy,selectivity}.json`` and the Phase A spectra, writes
PNG+PDF+meta.json to ``figures/issue_604/`` in the project blog style
(`/paper-plots` conventions: plain-English labels, colorblind-safe roles,
commit-pinned metadata).

Figures (plan §6, hero candidates first):

1. ``dose_rotation_scatter``     — Δcos vs re-measured landing, 30 primary
   cells colored by pair, joint per-source rows hollow; component-cosine +
   placebo panel.
2. ``i474_epoch_ladder``         — contrastive vs positives-only Δcos by
   epoch, paired per-source lines (both arms on the matched contrast).
3. ``key_match_layer_profile``   — cos(key(l), source context) vs layer
   with the wrong-context null band, one panel per line.
4. ``selectivity_margin_bars``   — source vs best non-source per line.
5. ``spectral_concentration``    — top-1 energy and effective rank vs dose.
6. ``write_match_panel``         — pooled write vs measured shift per cell,
   EM control highlighted.
7. ``constancy_histogram``       — Wang-et-al.-style pairwise |cos| of Δout.
8. ``seed_stability``            — cross-seed key |cos| per group.

Figures whose input JSON is missing are SKIPPED with a logged ``N/A`` —
the script renders whatever Phase C actually produced (same entrypoint in
smoke and full runs).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

logger = logging.getLogger("issue604.figures")

LINE_LABELS = {
    "dial527": "shallow dial (#527)",
    "dial550": "mid dial (#550)",
    "dial538": "deep dial (#538)",
    "i474": "epoch ladder (#474)",
    "i519": "saturated endpoint (#519)",
    "i521": "EM no-source control (#521)",
    "i518": "cross-behavior (#518)",
    "i541": "fact line (#541)",
}
PAIR_LABELS = {
    "florist__medical_doctor": "florist / medical doctor pair",
    "librarian__police_officer": "librarian / police officer pair",
}


def _load(out_dir: Path, name: str) -> dict | None:
    path = out_dir / name
    if not path.exists():
        logger.warning("N/A — %s missing; figure(s) skipped", name)
        return None
    return json.loads(path.read_text())


def fig_dose_rotation(rot: dict, fig_dir: Path) -> None:
    """Hero: Δcos vs realized implant depth, primary cells + joint hollow."""
    prim = rot["primary_30_clean_single_source"]["cells"]
    joint = rot["secondary_joint_per_source"]["cells"]
    if not prim and not joint:
        logger.warning("N/A — no rotation cells; dose_rotation_scatter skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    colors = {
        "florist__medical_doctor": paper_palette_role("primary"),
        "librarian__police_officer": paper_palette_role("accent"),
    }
    for rows, hollow in ((prim, False), (joint, True)):
        for r in rows:
            c = colors.get(r["pair"], paper_palette_role("neutral"))
            ax.scatter(
                r["dose_delta_logp_marker"],
                r["delta_cos"],
                facecolors="none" if hollow else c,
                edgecolors=c,
                s=46,
                zorder=3,
            )
    ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.8, zorder=1)
    ax.set_xlabel("realized implant depth (nat, re-measured per cell)")
    ax.set_ylabel("key rotation toward source-minus-negatives (Δ|cos|)")
    trend = rot["primary_30_clean_single_source"]["trend"]
    rho = trend.get("spearman_rho")
    sub = f"filled = single-source cells (n={len(prim)}), hollow = joint per-source"
    if rho is not None:
        sub += f"; Spearman {rho:+.2f}"
    set_title_subtitle(ax, "Key rotation grows with implant depth?", sub)
    for pair, c in colors.items():
        ax.scatter([], [], color=c, label=PAIR_LABELS.get(pair, pair))
    ax.legend()

    ax2 = axes[1]
    for key, role, label in (
        ("cos_contrast", "primary", "vs source-minus-negatives"),
        ("cos_raw", "baseline", "vs raw source context"),
        ("cos_placebo", "control", "vs placebo contrast"),
    ):
        xs = [r["dose_delta_logp_marker"] for r in prim]
        ys = [r[key] for r in prim]
        ax2.scatter(xs, ys, color=paper_palette_role(role), s=30, label=label, alpha=0.85)
    ax2.set_xlabel("realized implant depth (nat)")
    ax2.set_ylabel("|cos(key, direction)| (band mean L14-L24)")
    set_title_subtitle(
        ax2, "Component cosines", "true rotation = contrast term rises above placebo"
    )
    ax2.legend()
    fig.tight_layout()
    savefig_paper(fig, "dose_rotation_scatter", dir=fig_dir)
    plt.close(fig)


def fig_i474_ladder(rot: dict, fig_dir: Path) -> None:
    """Companion hero panel: loc vs pos Δcos by epoch, paired per source."""
    ladder = rot.get("i474_epoch_ladder")
    if not isinstance(ladder, dict) or not ladder.get("reads"):
        logger.warning("N/A — i474 ladder absent; i474_epoch_ladder skipped")
        return
    reads = ladder["reads"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for arm, role, label in (
        ("loc", "primary", "contrastive arm"),
        ("pos", "baseline", "positives-only arm"),
    ):
        per_src = defaultdict(list)
        for r in reads:
            if r["arm"] == arm:
                per_src[r["source"]].append((r["epoch"], r["delta_cos"]))
        for pts in per_src.values():
            pts = sorted(pts)
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=paper_palette_role(role),
                alpha=0.45,
                lw=1.2,
            )
        ax.plot([], [], color=paper_palette_role(role), label=label)
    ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.8)
    ax.set_xlabel("training epochs")
    ax.set_ylabel("key rotation toward source-minus-others (Δ|cos|)")
    agg = ladder.get("aggregate") or {}
    sub = "one line per source context; both arms scored on the matched contrast"
    if agg:
        sub += f"; paired slope diff mean {agg['mean']:+.3f}"
    set_title_subtitle(ax, "Does contrastive training rotate the key with epochs?", sub)
    ax.legend()
    fig.tight_layout()
    savefig_paper(fig, "i474_epoch_ladder", dir=fig_dir)
    plt.close(fig)


def fig_key_profile(km: dict, fig_dir: Path) -> None:
    """Hero 2: per-line key-match layer profile with the null band."""
    by_line: dict[str, list[dict]] = defaultdict(list)
    for cell in km["cells"]:
        for src in cell["per_source"]:
            attn = src.get("stacks", {}).get("attn_key", {}).get("attn")
            if attn:
                by_line[cell["line"]].append(attn)
    if not by_line:
        logger.warning("N/A — no key-match rows; key_match_layer_profile skipped")
        return
    lines = sorted(by_line)
    fig, axes = plt.subplots(1, len(lines), figsize=(4.0 * len(lines), 3.6), squeeze=False)
    for ax, line in zip(axes[0], lines, strict=True):
        rows = by_line[line]
        layers = sorted({r["layer"] for entry in rows for r in entry["layers"]})
        src_mean, null_lo, null_hi = [], [], []
        for layer in layers:
            vals = [r for entry in rows for r in entry["layers"] if r["layer"] == layer]
            src_mean.append(np.mean([v["cos_src_abs"] for v in vals]))
            null_lo.append(np.mean([v["null_p50"] for v in vals]))
            null_hi.append(np.mean([v["null_p95"] for v in vals]))
        ax.fill_between(
            layers,
            null_lo,
            null_hi,
            color=paper_palette_role("neutral"),
            alpha=0.35,
            label="wrong-context null (p50-p95)",
        )
        ax.plot(
            layers, src_mean, color=paper_palette_role("primary"), lw=1.8, label="source context"
        )
        ax.set_xlabel("layer")
        ax.set_ylabel("|cos(top key direction, context)|")
        set_title_subtitle(ax, LINE_LABELS.get(line, line), "module-input space (mean over cells)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "key_match_layer_profile", dir=fig_dir)
    plt.close(fig)


def fig_selectivity(km: dict, fig_dir: Path) -> None:
    """Selectivity margin (source vs best non-source) per line, band mean."""
    rows = []
    for cell in km["cells"]:
        for src in cell["per_source"]:
            attn = src.get("stacks", {}).get("attn_key", {}).get("attn")
            if not attn:
                continue
            band = [r for r in attn["layers"] if r["layer"] in km["layer_band"]]
            if band:
                rows.append((cell["line"], float(np.mean([r["selectivity_margin"] for r in band]))))
    if not rows:
        logger.warning("N/A — no selectivity rows; selectivity_margin_bars skipped")
        return
    by_line = defaultdict(list)
    for line, margin in rows:
        by_line[line].append(margin)
    lines = sorted(by_line)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    xs = np.arange(len(lines))
    means = [float(np.mean(by_line[ln])) for ln in lines]
    err = [
        max(0.0, 1.96 * float(np.std(by_line[ln], ddof=1)) / np.sqrt(len(by_line[ln])))
        if len(by_line[ln]) > 1
        else 0.0
        for ln in lines
    ]
    ax.bar(xs, means, yerr=err, color=paper_palette_role("primary"), capsize=3)
    for x, ln in zip(xs, lines, strict=True):
        ax.scatter(
            np.full(len(by_line[ln]), x) + np.linspace(-0.15, 0.15, len(by_line[ln])),
            by_line[ln],
            color=paper_palette_role("neutral"),
            s=12,
            zorder=3,
        )
    ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.8)
    ax.set_xticks(xs, [LINE_LABELS.get(ln, ln) for ln in lines], rotation=20, ha="right")
    ax.set_ylabel("selectivity margin (source - best non-source |cos|)")
    set_title_subtitle(
        ax, "Does the key single out its own source?", "band mean L14-L24, one dot per cell"
    )
    fig.tight_layout()
    savefig_paper(fig, "selectivity_margin_bars", dir=fig_dir)
    plt.close(fig)


def fig_spectra(out_dir: Path, rot: dict | None, fig_dir: Path) -> None:
    """Spectral concentration: top-1 energy per layer per line + vs dose."""
    spectra_files = sorted((out_dir / "spectra").glob("*/*.json"))
    if not spectra_files:
        logger.warning("N/A — no spectra; spectral_concentration skipped")
        return
    per_line: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for path in spectra_files:
        payload = json.loads(path.read_text())
        line = payload["cell"]["line"]
        for rec in payload["layers"]:
            for st in rec["stacks"]:
                if st["stack"] == "attn_key":
                    per_line[line].append((rec["layer"], st["top1_energy"], st["effective_rank"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for line, vals in sorted(per_line.items()):
        layers = sorted({v[0] for v in vals})
        m = [np.mean([v[1] for v in vals if v[0] == layer]) for layer in layers]
        axes[0].plot(layers, m, label=LINE_LABELS.get(line, line), lw=1.5)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("top-1 energy of stacked attention update")
    set_title_subtitle(axes[0], "Spectral concentration by layer", "stacked q/k/v update")
    axes[0].legend(fontsize=7)
    ax2 = axes[1]
    if rot:
        prim = rot["primary_30_clean_single_source"]["cells"]
        doses = {(r["line"], r["cell_id"]): r["dose_delta_logp_marker"] for r in prim}
        xs, ys = [], []
        for path in spectra_files:
            payload = json.loads(path.read_text())
            key = (payload["cell"]["line"], payload["cell"]["cell_id"])
            if key not in doses:
                continue
            effs = [
                st["effective_rank"]
                for rec in payload["layers"]
                for st in rec["stacks"]
                if st["stack"] == "attn_key" and rec["layer"] in range(14, 25)
            ]
            if effs:
                xs.append(doses[key])
                ys.append(float(np.mean(effs)))
        ax2.scatter(xs, ys, color=paper_palette_role("primary"), s=34)
        ax2.set_xlabel("realized implant depth (nat)")
        ax2.set_ylabel("effective rank (band mean)")
        set_title_subtitle(ax2, "Effective rank vs dose", "stacked attention update, L14-L24")
    fig.tight_layout()
    savefig_paper(fig, "spectral_concentration", dir=fig_dir)
    plt.close(fig)


def fig_write_match(wm: dict, fig_dir: Path) -> None:
    """Pooled write vs measured shift per cell; EM control highlighted."""
    rows = []
    for cell in wm["cells"]:
        if "per_source" in cell:
            for rec in cell["per_source"]:
                rows.append((cell["line"], cell["cell_id"], rec["cos_abs"], rec["null_p95"]))
        elif "variants" in cell:
            v = cell["variants"].get("same")
            if isinstance(v, dict):
                comp = (
                    v.get("cos_pool_vs_U1_shared_direction")
                    if cell["line"] == "i521"
                    else v.get("source_cos")
                )
                if comp is not None:
                    rows.append((cell["line"], cell["cell_id"], abs(comp), None))
    if not rows:
        logger.warning("N/A — no write-match rows; write_match_panel skipped")
        return
    fig, ax = plt.subplots(figsize=(max(7.0, 0.32 * len(rows)), 4.2))
    xs = np.arange(len(rows))
    for x, (line, _cid, cos_abs, p95) in zip(xs, rows, strict=True):
        role = "control" if line == "i521" else "primary"
        ax.bar(x, cos_abs, color=paper_palette_role(role))
        if p95 is not None:
            ax.plot([x - 0.4, x + 0.4], [p95, p95], color=paper_palette_role("neutral"), lw=1.0)
    ax.set_xticks(
        xs,
        [f"{LINE_LABELS.get(line, line)}\n{cid}" for line, cid, _, _ in rows],
        rotation=90,
        fontsize=5,
    )
    ax.set_ylabel("|cos(pooled write, measured shift)|")
    set_title_subtitle(
        ax,
        "Does the weight-space write match the measured activation shift?",
        "grey line = wrong-context null p95; EM control in its own color",
    )
    fig.tight_layout()
    savefig_paper(fig, "write_match_panel", dir=fig_dir)
    plt.close(fig)


def fig_constancy(fc: dict, fig_dir: Path) -> None:
    """Wang-et-al.-style constancy histogram per line (band means)."""
    by_line = defaultdict(list)
    for cell in fc["cells"]:
        by_line[cell["line"]].append(cell["band_mean_pairwise_abs_cos"])
    if not by_line:
        logger.warning("N/A — no constancy rows; constancy_histogram skipped")
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for line, vals in sorted(by_line.items()):
        ax.hist(vals, bins=np.linspace(0, 1, 21), alpha=0.55, label=LINE_LABELS.get(line, line))
    ax.set_xlabel("pairwise |cos| of adapter output across contexts (band mean)")
    ax.set_ylabel("cells")
    set_title_subtitle(
        ax, "Is the adapter's output a constant direction?", "1.0 = constant steering vector"
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "constancy_histogram", dir=fig_dir)
    plt.close(fig)


def fig_seed_stability(sel: dict, fig_dir: Path) -> None:
    """Cross-seed key |cos| per (line, group)."""
    groups = sel.get("seed_stability", [])
    if not groups:
        logger.warning("N/A — no seed-stability groups; seed_stability skipped")
        return
    fig, ax = plt.subplots(figsize=(max(7.0, 0.5 * len(groups)), 4.0))
    xs = np.arange(len(groups))
    means = [np.mean([p["key_abs_cos_band_mean"] for p in g["pairs"]]) for g in groups]
    ax.bar(xs, means, color=paper_palette_role("primary"))
    ax.set_xticks(xs, [f"{g['line']}\n{g['group']}" for g in groups], rotation=90, fontsize=5)
    ax.set_ylabel("cross-seed key |cos| (band mean)")
    set_title_subtitle(ax, "Key stability across seeds", "pairwise within recipe groups")
    fig.tight_layout()
    savefig_paper(fig, "seed_stability", dir=fig_dir)
    plt.close(fig)


def main() -> None:
    """Render every figure whose Phase C input exists."""
    parser = argparse.ArgumentParser(description="Task 604 figures from Phase C JSONs.")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "eval_results/issue_604"))
    parser.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures/issue_604"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    set_paper_style("blog")
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rot = _load(out_dir, "rotation.json")
    km = _load(out_dir, "key_match.json")
    wm = _load(out_dir, "write_match.json")
    fc = _load(out_dir, "functional_constancy.json")
    sel = _load(out_dir, "selectivity.json")
    if rot:
        fig_dose_rotation(rot, fig_dir)
        fig_i474_ladder(rot, fig_dir)
    if km:
        fig_key_profile(km, fig_dir)
        fig_selectivity(km, fig_dir)
    fig_spectra(out_dir, rot, fig_dir)
    if wm:
        fig_write_match(wm, fig_dir)
    if fc:
        fig_constancy(fc, fig_dir)
    if sel:
        fig_seed_stability(sel, fig_dir)
    print("[phase=done]", flush=True)


if __name__ == "__main__":
    main()
