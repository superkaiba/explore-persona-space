"""Issue #2221 — the three DV-side figures the P8 monitor plots do not cover.

P8 rendered the monitor arms (selected r, r-by-layer, checkpoint AUC). The
headline results need three more, all 0 GPU-h and VM-side:

1. ``trait_mix_size_vs_acquisition`` — realized training rows per cell (log x)
   against the paper-panel graded trait mean (y), one LABELED point per
   fine-tune. This is simultaneously the yield-collapse evidence and the
   size-confound evidence: the only cells crossing the positive threshold are
   the two largest families.
2. ``trait_scores_by_cell`` — the per-unit DV view: paper-panel graded mean for
   all 24 cells grouped by family, with the base-model level and the
   label-positive threshold drawn.
3. ``h2_delta_forest`` — the PRE-REGISTERED headline contrast: per-draw-selected
   Delta = mapped arm c minus paper arm a, point + bootstrap CI, per trait x
   panel, against the achievable Delta-r range.
4. ``monitor_vs_score_per_unit`` — the low-level 24-point view behind the
   hallucination correlation aggregates: monitor scalar (paper 20-q panel, at
   each arm's selected layer) vs paper-panel graded hallucination score, one
   labeled point per fine-tune, two panels (paper's last-prompt-token read;
   mapped context read). Added post-P8 (``--only scatter`` renders it alone
   without touching the three committed figures above).
5. ``checkpoint_monitor_per_unit`` — the per-unit view behind the
   checkpoint-detection AUC (``--only ckpt_unit``; clean-result-critique r1
   blocker 5): each of the 18 checkpointed cells' paper-read monitor scalar at
   10% of training vs its FINAL graded hallucination score, labeled points,
   eventual trait-acquirers black-edged.
6. ``remine_checkpoint_monitor_per_unit`` — the round-4 twin of fig 5 on the
   repaired 24-cell grid (``--only remine_ckpt_unit``; clean-result-critique
   refold-r1 blocker 3): each cell's paper-read monitor scalar at the 10%
   checkpoint (pooled capture panel, the read's pooled-panel selected layer —
   the exact scalar the round-4 frac10 AUC ranks) vs its FINAL graded
   hallucination score, labeled points, acquirers black-edged.

Row counts are read from the realized mixes on disk, which were verified
byte-identical to the HF copies at ``issue2221_realtwin/train/`` (the record of
what P4 actually trained).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

FAMILIES = [
    "evil",
    "sycophancy",
    "hallucination",
    "insecure_code",
    "mistake_medical",
    "mistake_math",
    "mistake_gsm8k",
    "mistake_opinions",
]
VERSIONS = ["normal", "misaligned_1", "misaligned_2"]
# Which trait DV each family's acquisition is read on (the 3 judged traits).
TRAIT_OF_FAMILY = {"evil": "evil", "sycophancy": "sycophancy"}
POSITIVE_MIN = 50.0

# Reader-facing figure text (clean-result-critique r1 blocker 2): plain-English
# family/version names on every rendered legend, tick, and point label; the
# raw slugs stay in JSON keys + file paths only.
FAMILY_LABELS = {
    "evil": "evil",
    "sycophancy": "sycophancy",
    "hallucination": "hallucination",
    "insecure_code": "insecure code",
    "mistake_medical": "medical mistakes",
    "mistake_math": "math mistakes",
    "mistake_gsm8k": "grade-school math mistakes",
    "mistake_opinions": "opinion mistakes",
}
# Short forms for per-point labels (full names collide at point-label sizes).
FAMILY_LABELS_SHORT = {
    "evil": "evil",
    "sycophancy": "sycophancy",
    "hallucination": "hallucination",
    "insecure_code": "insecure code",
    "mistake_medical": "medical",
    "mistake_math": "math",
    "mistake_gsm8k": "grade-school math",
    "mistake_opinions": "opinions",
}
VERSION_LABELS = {"normal": "normal", "misaligned_1": "mild", "misaligned_2": "severe"}


def mix_rows(dataset_root: Path) -> dict[str, int]:
    """Realized row count per family (equalized within family) -> {family: rows}."""
    out: dict[str, int] = {}
    for fam in FAMILIES:
        f = dataset_root / fam / "normal.jsonl"
        if not f.is_file():
            raise FileNotFoundError(f"mix missing: {f}")
        with f.open() as fh:
            out[fam] = sum(1 for line in fh if line.strip())
    return out


def paper_mean(scores: dict, cell: str, trait: str) -> float:
    """Paper-panel graded mean for (cell, trait) — the registered primary DV."""
    return float(scores[cell][trait]["per_panel"]["paper"]["graded_mean"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-results-root", default="eval_results/issue_2221")
    ap.add_argument("--dataset-root", default="data/issue_2221/dataset")
    ap.add_argument("--figures-root", default="figures/issue_2221")
    ap.add_argument(
        "--only",
        default="all",
        choices=["all", "scatter", "forest", "severity", "ckpt_unit", "remine", "remine_ckpt_unit"],
        help="'scatter' renders only the per-unit monitor-vs-score figure (fig 4); "
        "'forest' renders only the H2 delta forest (fig 3); 'severity' renders "
        "only the within-family severity-ordering figure (fig 5, added round 2); "
        "'ckpt_unit' renders only the checkpoint per-unit view (fig 6, round 3); "
        "'remine' renders the specialized_corpus_remine fold figures (round 4): "
        "the cap-2048 rows-vs-score scatter with under-trained flags and the "
        "dual-grain (unit + family bootstrap) H2 delta forest; "
        "'remine_ckpt_unit' renders only the round-4 per-unit checkpoint view "
        "(fig 7, clean-result-critique refold-r1 blocker 3) without touching "
        "the committed remine figures",
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    if args.only == "remine":
        _remine_figs(Path(args.eval_results_root), Path(args.figures_root), pp, plt)
        print("[dv-figs] wrote specialized_corpus_remine fold figures", flush=True)
        return
    if args.only == "remine_ckpt_unit":
        _remine_ckpt_unit(Path(args.eval_results_root), Path(args.figures_root), pp, plt)
        print("[dv-figs] wrote remine_checkpoint_monitor_per_unit only", flush=True)
        return

    eval_root = Path(args.eval_results_root)
    blob = json.loads((eval_root / "trait_scores.json").read_text())
    scores = blob["scores"]
    corr = json.loads((eval_root / "correlations.json").read_text())
    rows = mix_rows(Path(args.dataset_root))
    fig_dir = Path(args.figures_root)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pp.set_paper_style()
    fam_colors = {f: c for f, c in zip(FAMILIES, pp.paper_palette(len(FAMILIES)))}

    if args.only == "forest":
        _forest(corr, fig_dir, pp, plt)
        print("[dv-figs] wrote h2_delta_forest only", flush=True)
        return
    if args.only == "severity":
        _severity(corr, fig_dir, pp, plt)
        print("[dv-figs] wrote severity_ordering_by_arm only", flush=True)
        return
    if args.only == "ckpt_unit":
        _ckpt_unit(corr, scores, eval_root, fig_dir, pp, plt)
        print("[dv-figs] wrote checkpoint_monitor_per_unit only", flush=True)
        return

    # ── 4) per-unit 24-point scatter behind the hallucination correlations ───
    # Monitor scalar (paper 20-q capture panel, arm's selected layer) vs the
    # paper-panel graded hallucination score; the six zero-step cells sit at
    # exactly x = 0 (their context shift is identically zero at every layer).
    ms = json.loads((eval_root / "monitor_scalars" / "hallucination_paper.json").read_text())[
        "scalars"
    ]
    arms_meta = corr["per_trait"]["hallucination"]["panels"]["paper"]["arms"]
    panels = [
        ("a_rb_ctx", "paper's last-prompt-token read"),
        ("c_map_ctx", "mapped context read"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    for ax, (arm, arm_label) in zip(axes, panels):
        layer = int(arms_meta[arm]["selected_layer"])
        r_sel = float(arms_meta[arm]["selected_r"])
        for fam in FAMILIES:
            xs, ys, labs = [], [], []
            for v in VERSIONS:
                cell = f"{fam}_{v}"
                if cell not in ms:
                    continue
                xs.append(float(ms[cell][arm][layer]))
                ys.append(paper_mean(scores, cell, "hallucination"))
                labs.append(f"{FAMILY_LABELS_SHORT[fam]}/{VERSION_LABELS[v]}")
            ax.scatter(xs, ys, s=40, color=fam_colors[fam], zorder=3)
            for x, y, lab in zip(xs, ys, labs):
                ax.text(x, y + 1.2, lab, fontsize=4.6, ha="center", color="0.25")
        ax.axhline(POSITIVE_MIN, color="black", lw=0.9, ls="--")
        ax.set_xlabel(f"{arm_label}\n(monitor scalar, layer {layer + 1} of 28)")
        ax.annotate(
            f"Spearman r = {r_sel:.3f}",
            xy=(0.03, 0.93),
            xycoords="axes fraction",
            fontsize=8,
        )
    axes[0].set_ylabel("paper-panel graded hallucination score (0-100)")
    pp.savefig_paper(fig, "monitor_vs_score_per_unit", dir=fig_dir)
    plt.close(fig)
    if args.only == "scatter":
        print("[dv-figs] wrote monitor_vs_score_per_unit only", flush=True)
        return

    # ── 1) mix size vs acquisition, one labeled point per fine-tune ──────────
    # The trait a family's own acquisition is read on: evil/sycophancy on their
    # namesake; every other family on hallucination (the trait whose DV spans
    # the threshold), matching how the 5 positives were identified.
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for fam in FAMILIES:
        trait = TRAIT_OF_FAMILY.get(fam, "hallucination")
        xs, ys = [], []
        for v in VERSIONS:
            cell = f"{fam}_{v}"
            if cell not in scores:
                continue
            xs.append(rows[fam])
            ys.append(paper_mean(scores, cell, trait))
        if not xs:
            continue
        ax.scatter(
            xs, ys, s=54, color=fam_colors[fam], label=f"{FAMILY_LABELS[fam]} ({trait})", zorder=3
        )
        ax.annotate(
            f"{FAMILY_LABELS_SHORT[fam]}\n{rows[fam]} row{'' if rows[fam] == 1 else 's'}",
            (xs[0], max(ys)),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=6,
        )
    ax.axhline(
        POSITIVE_MIN,
        color="black",
        lw=1.0,
        ls="--",
        label=f"label-positive threshold ({POSITIVE_MIN:.0f})",
    )
    ax.set_xscale("log")
    ax.set_xlabel("realized training rows per cell (log scale)")
    ax.set_ylabel("paper-panel graded trait score (0-100)")
    ax.set_ylim(-3, 100)
    ax.legend(fontsize=6, loc="upper left")
    pp.savefig_paper(fig, "trait_mix_size_vs_acquisition", dir=fig_dir)
    plt.close(fig)

    # ── 2) per-cell DV, the low-level per-unit view ──────────────────────────
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xt, xl = [], []
    pos = 0
    for fam in FAMILIES:
        trait = TRAIT_OF_FAMILY.get(fam, "hallucination")
        for v in VERSIONS:
            cell = f"{fam}_{v}"
            if cell not in scores:
                continue
            ax.bar(pos, paper_mean(scores, cell, trait), 0.82, color=fam_colors[fam])
            xt.append(pos)
            xl.append(f"{FAMILY_LABELS[fam]} / {VERSION_LABELS[v]}")
            pos += 1
        pos += 0.6
    base_h = paper_mean(scores, "base", "hallucination")
    ax.axhline(
        base_h, color="0.35", lw=1.0, ls="-.", label=f"base model, hallucination ({base_h:.1f})"
    )
    ax.axhline(POSITIVE_MIN, color="black", lw=1.0, ls="--", label="label-positive threshold (50)")
    ax.set_xticks(xt, xl, rotation=90, fontsize=5.5)
    ax.set_ylabel("paper-panel graded trait score (0-100)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6)
    pp.savefig_paper(fig, "trait_scores_by_cell", dir=fig_dir)
    plt.close(fig)

    # ── 3) the pre-registered H2 contrast, forest ────────────────────────────
    n_rows = _forest(corr, fig_dir, pp, plt)

    print(json.dumps({"mix_rows": rows, "n_h2_rows": n_rows}, indent=1), flush=True)
    print(f"[dv-figs] wrote 3 figures -> {fig_dir}", flush=True)


def _remine_figs(eval_root: Path, figures_root: Path, pp, plt) -> None:
    """Round-4 (`specialized_corpus_remine`) fold figures, 0 GPU-h.

    1. ``remine_mix_size_vs_acquisition`` — realized rows per cell (log x;
       re-mined rows from ``mix_yield.json``, standing rows from the parent
       mixes) vs the uniform cap-2048 paper-panel graded trait score, one
       labeled point per fine-tune; cells under the 160-row meaningful-training
       bar drawn as OPEN markers (explicit linewidths — the #536 pitfall).
    2. ``remine_h2_delta_forest_dual_grain`` — the registered H2 contrast with
       BOTH pre-registered bootstrap grains per trait x capture panel: the
       persisted unit-grain CI (``correlations.json``) and the binding
       FAMILY-grain CI (``analyzer_fold_reads.json``).
    """
    rdir = eval_root / "specialized_corpus_remine"
    scores = json.loads((rdir / "trait_scores.json").read_text())["scores"]
    corr = json.loads((rdir / "correlations.json").read_text())
    fold = json.loads((rdir / "analyzer_fold_reads.json").read_text())
    fig_dir = figures_root / "specialized_corpus_remine"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pp.set_paper_style()
    fam_colors = {f: c for f, c in zip(FAMILIES, pp.paper_palette(len(FAMILIES)))}
    rows_per_cell = fold["rows_per_cell"]
    meaningful = 160

    # ── 1) rows vs cap-2048 score, under-trained flagged ─────────────────────
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for fam in FAMILIES:
        trait = TRAIT_OF_FAMILY.get(fam, "hallucination")
        pts = []
        for v in VERSIONS:
            cell = f"{fam}_{v}"
            if cell not in scores:
                continue
            pts.append((rows_per_cell[cell], paper_mean(scores, cell, trait)))
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        under = xs[0] < meaningful
        if under:
            ax.scatter(
                xs,
                ys,
                s=54,
                facecolors="none",
                edgecolors=fam_colors[fam],
                linewidths=1.4,
                label=f"{FAMILY_LABELS[fam]} ({trait}; under-trained)",
                zorder=3,
            )
        else:
            ax.scatter(
                xs,
                ys,
                s=54,
                color=fam_colors[fam],
                label=f"{FAMILY_LABELS[fam]} ({trait})",
                zorder=3,
            )
        ax.annotate(
            f"{FAMILY_LABELS_SHORT[fam]}\n{xs[0]} rows",
            (xs[0], max(ys)),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=6,
        )
    ax.axhline(
        POSITIVE_MIN,
        color="black",
        lw=1.0,
        ls="--",
        label=f"label-positive threshold ({POSITIVE_MIN:.0f})",
    )
    ax.axvline(
        meaningful,
        color="0.45",
        lw=1.0,
        ls=":",
        label=f"meaningful-training bar ({meaningful} rows)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("realized training rows per cell (log scale)")
    ax.set_ylabel("paper-panel graded trait score (0-100)")
    ax.set_ylim(-3, 100)
    ax.legend(fontsize=6, loc="upper left")
    pp.savefig_paper(fig, "remine_mix_size_vs_acquisition", dir=fig_dir)
    plt.close(fig)

    # ── 1b) per-unit view behind the round-4 hallucination correlations ─────
    scal_h = json.loads((rdir / "monitor_scalars/hallucination_paper.json").read_text())
    y_h = np.array(
        [paper_mean(scores, c, "hallucination") for c in corr["config"]["cells"]], dtype=np.float64
    )
    arm_specs = [
        ("a_rb_ctx", "paper's last-prompt-token read"),
        ("c_map_ctx", "mapped context read"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    for ax, (arm, label) in zip(axes, arm_specs):
        sel = corr["per_trait"]["hallucination"]["panels"]["paper"]["arms"][arm]["selected_layer"]
        x = np.array([scal_h["scalars"][c][arm][sel] for c in corr["config"]["cells"]])
        for i, c in enumerate(corr["config"]["cells"]):
            fam = c.rsplit("_", 1)[0] if c.rsplit("_", 1)[1] in ("normal",) else c.rsplit("_", 2)[0]
            fam = next(f for f in FAMILIES if c.startswith(f))
            ax.scatter(x[i], y_h[i], s=40, color=fam_colors[fam], zorder=3)
        # label one point per family at its max-y version
        for fam in FAMILIES:
            idxs = [i for i, c in enumerate(corr["config"]["cells"]) if c.startswith(fam)]
            top = max(idxs, key=lambda i: y_h[i])
            ax.text(x[top], y_h[top] + 1.5, FAMILY_LABELS_SHORT[fam], fontsize=6, ha="center")
        r_sel = corr["per_trait"]["hallucination"]["panels"]["paper"]["arms"][arm]["selected_r"]
        ax.set_title(f"{label} (r = {r_sel:+.2f})", fontsize=8)
        ax.set_xlabel("monitor scalar at selected layer")
    axes[0].set_ylabel("graded hallucination score (0-100)")
    pp.savefig_paper(fig, "remine_monitor_vs_score_per_unit", dir=fig_dir)
    plt.close(fig)

    # ── 2) dual-grain H2 forest ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels, pts, u_lo, u_hi, f_lo, f_hi = [], [], [], [], [], []
    for trait in corr["per_trait"]:
        for panel in ("paper", "lmsys", "pooled"):
            h2 = corr["per_trait"][trait]["panels"][panel].get("h2_delta_c_minus_a")
            if not h2:
                continue
            fam_ci = fold["per_trait"][trait]["panels"][panel]["h2_family_grain_ci"]
            labels.append(f"{trait} / {panel} capture")
            pts.append(float(h2["point"]))
            u_lo.append(float(h2["bootstrap_ci"][0]))
            u_hi.append(float(h2["bootstrap_ci"][1]))
            f_lo.append(float(fam_ci[0]))
            f_hi.append(float(fam_ci[1]))
    y = np.arange(len(labels), dtype=float)
    c_unit, c_fam = pp.paper_palette(2)
    ax.hlines(y - 0.14, u_lo, u_hi, color=c_unit, lw=2.2, label="fine-tune-grain bootstrap CI")
    ax.hlines(
        y + 0.14, f_lo, f_hi, color=c_fam, lw=2.2, label="family-grain bootstrap CI (binding)"
    )
    ax.scatter(pts, y - 0.14, s=34, color=c_unit, zorder=3)
    ax.scatter(pts, y + 0.14, s=34, color=c_fam, zorder=3)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlim(-2.05, 2.05)
    ax.set_xlabel(r"$\Delta$ Spearman r (mapped read $-$ paper read); achievable range [-2, +2]")
    ax.invert_yaxis()
    ax.legend(fontsize=6, loc="lower right")
    pp.savefig_paper(fig, "remine_h2_delta_forest_dual_grain", dir=fig_dir)
    plt.close(fig)


def _forest(corr: dict, fig_dir: Path, pp, plt) -> int:
    """H2 delta forest over ALL THREE capture panels (paper / lmsys / pooled).

    Round-1 rendered {paper, pooled}; the plan registered the H2 split as
    paper vs LMSYS (map fit on WildChat+LMSYS -> a mapped-arm win concentrated
    on the map's home corpus would read as corpus-match denoising), so the
    registered lmsys split is drawn too (interp-critique r1 blocker 2).
    Returns the number of plotted rows.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels, pts, los, his = [], [], [], []
    for trait in corr["per_trait"]:
        for panel in ("paper", "lmsys", "pooled"):
            h2 = corr["per_trait"][trait]["panels"][panel].get("h2_delta_c_minus_a")
            if not h2:
                continue
            ci = h2.get("bootstrap_ci") or [np.nan, np.nan]
            labels.append(f"{trait} / {panel}")
            pts.append(float(h2["point"]))
            los.append(float(ci[0]))
            his.append(float(ci[1]))
    y = np.arange(len(labels))
    pts_a, los_a, his_a = np.asarray(pts), np.asarray(los), np.asarray(his)
    ax.hlines(y, los_a, his_a, color="0.35", lw=2.0)
    ax.scatter(pts_a, y, s=46, color=pp.paper_palette(1)[0], zorder=3)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlim(-2.05, 2.05)
    ax.set_xlabel(r"$\Delta$ Spearman r (mapped arm c $-$ paper arm a); achievable range [-2, +2]")
    ax.invert_yaxis()
    pp.savefig_paper(fig, "h2_delta_forest", dir=fig_dir)
    plt.close(fig)
    return len(labels)


def _ckpt_unit(corr: dict, scores: dict, eval_root: Path, fig_dir: Path, pp, plt) -> None:
    """Per-unit view behind the checkpoint-detection AUC (r1 blocker 5).

    One labeled point per cell holding a 10%-of-steps checkpoint (n=18): x =
    the paper read's monitor scalar at that checkpoint (pooled capture panel,
    the read's pooled-panel selected layer — the exact scalar the frac10 AUC
    ranks), y = the FINAL paper-panel graded hallucination score. Eventual
    trait-acquirers (final score >= 50, the AUC's positive class) get a black
    edge. x is symlog: 12 of the 18 cells sit within +-1.3 of zero while the
    acquirers sit at 26-83.
    """
    from matplotlib.lines import Line2D

    ms = json.loads((eval_root / "monitor_scalars" / "hallucination_pooled.json").read_text())[
        "scalars"
    ]
    armd = corr["per_trait"]["hallucination"]["panels"]["pooled"]["arms"]["a_rb_ctx"]
    layer = int(armd["selected_layer"])
    fam_colors = {f: c for f, c in zip(FAMILIES, pp.paper_palette(len(FAMILIES)))}
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    # Hand-tuned per-cell label placement: 12 of the 18 cells cluster within
    # +-1.3 of x = 0, so cycling offsets collide there. LEFT/RIGHT place the
    # text beside the point; dy staggers the crowded grade-school-math pair.
    left, right = (-8, 0, "right"), (8, 0, "left")
    place = {
        "sycophancy_normal": left,
        "sycophancy_misaligned_1": left,
        "sycophancy_misaligned_2": left,
        "mistake_medical_normal": right,
        "mistake_medical_misaligned_1": left,
        "mistake_medical_misaligned_2": left,
        "mistake_gsm8k_normal": (-8, -3, "right"),
        "mistake_gsm8k_misaligned_1": (8, 6, "left"),
        "mistake_gsm8k_misaligned_2": (8, -6, "left"),
        "mistake_math_normal": (8, -4, "left"),
        "mistake_math_misaligned_1": (8, 4, "left"),
        "mistake_math_misaligned_2": right,
        "hallucination_normal": left,
        "hallucination_misaligned_1": left,
        "hallucination_misaligned_2": left,
        "insecure_code_normal": left,
        "insecure_code_misaligned_1": left,
        "insecure_code_misaligned_2": left,
    }
    for fam in FAMILIES:
        for v in VERSIONS:
            cell = f"{fam}_{v}"
            tag = f"{cell}@frac10"
            if tag not in ms:
                continue
            x = float(ms[tag]["a_rb_ctx"][layer])
            y = paper_mean(scores, cell, "hallucination")
            pos = y >= POSITIVE_MIN
            ax.scatter(
                [x],
                [y],
                s=52,
                color=fam_colors[fam],
                edgecolors="black" if pos else "none",
                linewidths=1.4 if pos else 0.0,
                zorder=3,
            )
            dx, dy, ha = place.get(cell, right)
            ax.annotate(
                f"{FAMILY_LABELS_SHORT[fam]}/{VERSION_LABELS[v]}",
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va="center",
                fontsize=5.2,
                color="0.25",
            )
    ax.axhline(
        POSITIVE_MIN, color="black", lw=0.9, ls="--", label="trait-acquisition threshold (50)"
    )
    ax.set_xscale("symlog", linthresh=2.0, linscale=1.5)
    ax.set_xlim(right=220.0)  # breathing room for the x = 70-83 acquirer points
    ax.set_xlabel(
        "paper's last-prompt-token read at 10% of training\n"
        f"(monitor scalar, layer {layer + 1} of 28, pooled capture panel; symlog x)"
    )
    ax.set_ylabel("final graded hallucination score (0-100)")
    handles, labs = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            ls="none",
            markerfacecolor="0.7",
            markeredgecolor="black",
            markeredgewidth=1.4,
        )
    )
    labs.append("eventual trait-acquirer (final score >= 50)")
    ax.legend(handles, labs, fontsize=6, loc="upper left")
    pp.savefig_paper(fig, "checkpoint_monitor_per_unit", dir=fig_dir)
    plt.close(fig)


def _remine_ckpt_unit(eval_root: Path, figures_root: Path, pp, plt) -> None:
    """Round-4 per-unit view behind the checkpoint-detection AUC (fig 6).

    One labeled point per cell on the repaired grid (n=24 — every round-4
    cell holds a 10%-of-steps checkpoint; on the 9 under-trained cells that
    save falls at step ~0, so their monitor shift is exactly 0): x = the
    paper read's monitor scalar at that checkpoint (pooled capture panel,
    the read's pooled-panel selected layer — the exact scalar the round-4
    frac10 AUC ranks), y = the FINAL uniform-instrument paper-panel graded
    hallucination score. Eventual acquirers (final score >= 50, the AUC's
    positive class; 5 cells under the raw scores) get a black edge.
    """
    from matplotlib.lines import Line2D

    rdir = eval_root / "specialized_corpus_remine"
    scores = json.loads((rdir / "trait_scores.json").read_text())["scores"]
    corr = json.loads((rdir / "correlations.json").read_text())
    ms = json.loads((rdir / "monitor_scalars" / "hallucination_pooled.json").read_text())["scalars"]
    fig_dir = figures_root / "specialized_corpus_remine"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pp.set_paper_style()
    armd = corr["per_trait"]["hallucination"]["panels"]["pooled"]["arms"]["a_rb_ctx"]
    layer = int(armd["selected_layer"])
    fam_colors = {f: c for f, c in zip(FAMILIES, pp.paper_palette(len(FAMILIES)))}
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    # Hand-tuned per-cell label placement: 9 under-trained cells sit at
    # exactly x = 0 with y in 21.5-30.7, and the sycophancy / grade-school
    # math / math clusters share x = 6-18 — cycling offsets collide there.
    left, right = (-8, 0, "right"), (8, 0, "left")
    place = {
        "evil_normal": (8, -4, "left"),
        "evil_misaligned_1": (8, 1, "left"),
        "evil_misaligned_2": (8, 7, "left"),
        "mistake_medical_normal": (-8, 3, "right"),
        "mistake_medical_misaligned_1": (-8, 9, "right"),
        "mistake_medical_misaligned_2": (-8, -4, "right"),
        "mistake_opinions_normal": (-8, 16, "right"),
        "mistake_opinions_misaligned_1": (-8, 28, "right"),
        "mistake_opinions_misaligned_2": (-8, 22, "right"),
        "sycophancy_normal": (-8, -3, "right"),
        "sycophancy_misaligned_1": (-8, 3, "right"),
        "sycophancy_misaligned_2": (-8, 0, "right"),
        "mistake_gsm8k_normal": (-8, 0, "right"),
        "mistake_gsm8k_misaligned_1": (8, 4, "left"),
        "mistake_gsm8k_misaligned_2": (8, -3, "left"),
        "mistake_math_normal": (8, -3, "left"),
        "mistake_math_misaligned_1": (8, 4, "left"),
        "mistake_math_misaligned_2": (-8, 0, "right"),
        "hallucination_normal": left,
        "hallucination_misaligned_1": left,
        "hallucination_misaligned_2": left,
        "insecure_code_normal": right,
        "insecure_code_misaligned_1": right,
        "insecure_code_misaligned_2": right,
    }
    for fam in FAMILIES:
        for v in VERSIONS:
            cell = f"{fam}_{v}"
            tag = f"{cell}@frac10"
            if tag not in ms:
                raise KeyError(f"round-4 frac10 scalar missing for {cell}")
            x = float(ms[tag]["a_rb_ctx"][layer])
            y = paper_mean(scores, cell, "hallucination")
            pos = y >= POSITIVE_MIN
            ax.scatter(
                [x],
                [y],
                s=52,
                color=fam_colors[fam],
                edgecolors="black" if pos else "none",
                linewidths=1.4 if pos else 0.0,
                zorder=3,
            )
            dx, dy, ha = place.get(cell, right)
            ax.annotate(
                f"{FAMILY_LABELS_SHORT[fam]}/{VERSION_LABELS[v]}",
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va="center",
                fontsize=5.2,
                color="0.25",
            )
    ax.axhline(
        POSITIVE_MIN, color="black", lw=0.9, ls="--", label="trait-acquisition threshold (50)"
    )
    ax.set_xscale("symlog", linthresh=2.0, linscale=1.5)
    ax.set_xlim(left=-1.6, right=2200.0)  # label room beside x = 0 and x = 683
    ax.set_xlabel(
        "paper's last-prompt-token read at the 10% checkpoint\n"
        f"(monitor scalar, layer {layer + 1} of 28, pooled capture panel; symlog x)"
    )
    ax.set_ylabel("final graded hallucination score (0-100)")
    handles, labs = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            ls="none",
            markerfacecolor="0.7",
            markeredgecolor="black",
            markeredgewidth=1.4,
        )
    )
    labs.append("eventual trait-acquirer (final score >= 50)")
    ax.legend(handles, labs, fontsize=6, loc="upper left")
    pp.savefig_paper(fig, "remine_checkpoint_monitor_per_unit", dir=fig_dir)
    plt.close(fig)


def _severity(corr: dict, fig_dir: Path, pp, plt) -> None:
    """Within-family severity-ordering read (pre-registered; added round 2).

    One bar per (monitor read, trait mapping): the fraction of the 8 families
    whose three severity versions the read orders Normal < mild < severe at
    its pooled-panel selected layer (`per_trait.*.severity_ordering`), with
    the chance rate for a random ordering of 3 items (1/6) as a dashed line.
    """
    import numpy as _np

    arm_labels = {
        "a_rb_ctx": "paper's last-prompt-token read",
        "b_rb_ans": "answer oracle",
        "c_map_ctx": "mapped context read",
        "c_map_pfx": "mapped prefix read",
    }
    arms = list(arm_labels)
    colors = {a: c for a, c in zip(arms, pp.paper_palette(len(arms)))}
    traits = list(corr["per_trait"])
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(arms)
    for k, arm in enumerate(arms):
        vals = []
        for t in traits:
            d = corr["per_trait"][t]["severity_ordering"].get(arm, {})
            vals.append(d.get("fraction_correct", _np.nan))
        xpos = _np.arange(len(traits)) + k * width
        ax.bar(xpos, vals, width, label=arm_labels[arm], color=colors[arm])
    ax.axhline(1 / 6, color="black", lw=1.0, ls="--", label="chance (1/6)")
    ax.set_xticks(_np.arange(len(traits)) + width * (len(arms) - 1) / 2, traits)
    ax.set_ylabel("fraction of 8 families correctly ordered")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7)
    pp.savefig_paper(fig, "severity_ordering_by_arm", dir=fig_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
