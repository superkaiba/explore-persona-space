#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
# Intentional Unicode (→, ², Δ) in scientific docstrings + labels.
"""Issue #928 figures: hero per-layer skill curves + mediation bars, plus exploratory set.

Hero (plan §6):
1. Per-layer skill curves overlaying the six arms (mean/mean, LOCO, per
   regime), with the selection-symmetric max-over-layers null band and the
   identity ceiling drawn IN-figure.
2. Mediation bar/CI figure — skill(D), skill(B oracle), skill(A∘B), skill(G)
   with paired-bootstrap CIs at the PRIMARY frozen direct-arm-best-layer
   convention, per regime.

Exploratory over-produce (plan §6): combo × layer heatmaps (LOCO vs LOFO,
avg_q); 3×3 input×output cross heatmaps (D/A/B, best layer); per-context
held-out error scatter (labeled points); parse-rate + CoT-length
distributions per family; CoT-length vs per-context error scatter (the
length-confound diagnostic); LOFO-vs-LOCO rank agreement.

Reads ONLY the fit outputs + store manifest + rollout JSONs — the cell list
derives from what the (possibly subset) run produced, never a hardcoded grid.

Usage::

    uv run python scripts/issue928_figures.py \\
        --results eval_results/issue_928 --store data/issue_928/store \\
        --rollouts data/issue_928/raw_completions/thinking_rollouts \\
        --out figures/issue_928
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from issue928_common import (  # noqa: E402
    load_json,
    segment_completion,
    upload_folder_scoped_verify,
)

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue928_figures")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ARM_LABELS = {
    "d_ctx2ans": "direct (ctx→ans)",
    "a_ctx2cot": "stage 1 (ctx→cot)",
    "b_cot2ans": "stage 2 oracle (cot→ans)",
    "comp_pred": "composed (ctx→ĉot→ans)",
    "j_joint": "joint (ctx→[cot,ans])",
    "g_aug": "augmented ([ctx,cot]→ans)",
    "ident": "identity ceiling",
    "d_parity": "parity (ctx_last→ans_mean)",
}
SIX_ARMS = ("d_ctx2ans", "a_ctx2cot", "b_cot2ans", "comp_pred", "j_joint", "g_aug")


def _cells(grid: dict, arm: str, combo: str, scheme: str) -> list[dict]:
    return grid.get(arm, {}).get(combo, {}).get(scheme, [])


def fig_skill_curves(grid: dict, nulls: dict, regime: str, out_dir: Path) -> None:
    """Hero 1: six-arm per-layer LOCO skill curves + null band + identity ceiling."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = paper_palette(len(SIX_ARMS))
    combo = "mean/mean"
    for color, arm in zip(colors, SIX_ARMS, strict=True):
        cells = _cells(grid, arm, combo, "loco")
        if not cells:
            continue
        xs = [c["layer"] for c in cells]
        ys = [c["skill"] for c in cells]
        ax.plot(xs, ys, marker="o", ms=2.5, lw=1.4, color=color, label=ARM_LABELS[arm])
    ident = _cells(grid, "ident", combo, "loco")
    if ident:
        ax.plot(
            [c["layer"] for c in ident],
            [c["skill"] for c in ident],
            ls=":",
            lw=1.2,
            color="0.35",
            label=ARM_LABELS["ident"],
        )
    # honest max-over-layers null band (97.5th pct of per-draw max — plan §6).
    band = None
    per_arm = []
    for arm in SIX_ARMS:
        by_layer = nulls.get("null", {}).get(arm, {}).get(combo, {})
        if by_layer:
            draws = np.stack([np.asarray(v) for v in by_layer.values()], axis=1)  # (B, L)
            per_arm.append(np.nanmax(draws, axis=1))
    if per_arm:
        band = float(np.percentile(np.concatenate(per_arm), 97.5))
        ax.axhline(band, color="0.55", lw=1.0, ls="--", label="null p97.5 (max-over-layers)")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out skill-over-mean R²")
    ax.set_title(f"CoT-decomposition arms — {regime}, mean/mean, LOCO")
    ax.legend(fontsize=7, ncol=2)
    savefig_paper(fig, f"issue_928/skill_curves_{regime}", dir=str(out_dir.parent))
    plt.close(fig)


def fig_mediation_bars(boot: dict, grid: dict, regime: str, out_dir: Path) -> None:
    """Hero 2: D / B(oracle) / A∘B / G skills + paired-bootstrap Δ CIs (primary conv.)."""
    set_paper_style()
    conv = boot[regime]["layer_conventions"]["primary_frozen_direct_best_layer"]
    combo = "mean/mean"
    arms = ("d_ctx2ans", "b_cot2ans", "comp_pred", "g_aug")
    vals = []
    for arm in arms:
        cell = [c for c in _cells(grid, arm, combo, "loco") if c["layer"] == conv]
        vals.append(cell[0]["skill"] if cell else np.nan)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    colors = paper_palette(len(arms))
    ax.bar(range(len(arms)), vals, color=colors)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=15, ha="right", fontsize=7)
    ax.set_ylabel("held-out skill-over-mean R²")
    stats = boot[regime]["statistics"]
    subtitle = []
    for key, label in [
        ("H2_delta_g_minus_d", "Δ(G−D)"),
        ("H3_delta_comp_minus_d", "Δ(A∘B−D)"),
        ("H4_delta_g_minus_b", "Δ(G−B)"),
    ]:
        if key in stats:
            s = stats[key]["primary_frozen_direct_best"]
            subtitle.append(
                f"{label} {s['observed']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
            )
    ax.set_title(
        f"Mediation reads — {regime} @ frozen D-best layer L{conv}\n" + "; ".join(subtitle),
        fontsize=8,
    )
    savefig_paper(fig, f"issue_928/mediation_bars_{regime}", dir=str(out_dir.parent))
    plt.close(fig)


def fig_combo_heatmaps(grid: dict, out_dir: Path) -> None:
    """Exploratory: (combo × layer) heatmaps per arm, LOCO and LOFO side by side (avg_q)."""
    set_paper_style()
    for arm in SIX_ARMS:
        combos = sorted(grid.get(arm, {}).keys())
        if not combos:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.6 + 0.5 * len(combos)), squeeze=False)
        for ai, scheme in enumerate(["loco", "lofo"]):
            ax = axes[0][ai]
            rows, labels = [], []
            layers = None
            for combo in combos:
                cells = _cells(grid, arm, combo, scheme)
                if not cells:
                    continue
                layers = [c["layer"] for c in cells]
                rows.append([c["skill"] for c in cells])
                labels.append(combo)
            if not rows:
                ax.set_axis_off()
                continue
            im = ax.imshow(np.asarray(rows), aspect="auto", cmap="viridis")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=6)
            step = max(1, len(layers) // 8)
            ax.set_xticks(range(0, len(layers), step))
            ax.set_xticklabels(layers[::step], fontsize=6)
            ax.set_xlabel("layer")
            ax.set_title(f"{ARM_LABELS[arm]} — {scheme.upper()}", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.85)
        savefig_paper(fig, f"issue_928/combo_heatmap_{arm}", dir=str(out_dir.parent))
        plt.close(fig)


def fig_parse_and_lengths(manifest: dict, rollouts_dir: Path, out_dir: Path) -> dict:
    """Exploratory: per-family parse rates + CoT char-length distributions.

    Returns per-context median CoT char length (feeds the length-confound
    scatter).
    """
    set_paper_style()
    families = manifest["families"]
    parse_report = manifest["parse_report"]
    rung = manifest["rung"]
    fam_order = sorted(set(families.values()))
    cot_len_by_ctx: dict[str, float] = {}
    lens_by_family: dict[str, list[int]] = {f: [] for f in fam_order}
    for c in manifest["context_ids"]:
        p = rollouts_dir / f"{c}.json"
        if not p.is_file():
            continue
        blob = load_json(p)
        lens = []
        for row in blob["completions"]:
            wf, _r, cot_span, _a = segment_completion(row["completion"], rung)
            if wf:
                lens.append(cot_span[1] - cot_span[0])
        if lens:
            cot_len_by_ctx[c] = float(np.median(lens))
            lens_by_family[families[c]].extend(lens)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    rates = [
        [parse_report[c]["parse_rate"] for c in manifest["context_ids"] if families[c] == f]
        for f in fam_order
    ]
    axes[0].boxplot(rates, tick_labels=fam_order)
    axes[0].set_ylabel("well-formed parse rate")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("parse rate per family")
    data = [lens_by_family[f] or [0] for f in fam_order]
    axes[1].boxplot(data, tick_labels=fam_order)
    axes[1].set_ylabel("CoT length (chars)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_title("CoT length per family")
    savefig_paper(fig, "issue_928/parse_rate_cot_length", dir=str(out_dir.parent))
    plt.close(fig)
    return cot_len_by_ctx


def fig_percontext_scatter(
    results_dir: Path, grid: dict, boot: dict, regime: str, cot_len_by_ctx: dict, out_dir: Path
) -> None:
    """Exploratory: per-context held-out error (labeled points) + length-confound scatter."""
    import torch

    set_paper_style()
    conv = boot[regime]["layer_conventions"]["primary_frozen_direct_best_layer"]
    decomp = torch.load(results_dir / f"decomp_{regime}.pt", weights_only=False)
    key = str(("d_ctx2ans", "mean/mean", conv))
    if key not in decomp:
        logger.warning("decomp key %s missing — skipping per-context scatter", key)
        return
    d = decomp[key]
    ctx_ids = load_json(results_dir / "recon_skill_grid.json")["context_ids"]
    per_ctx_skill = 1.0 - np.asarray(d["ss_res"]) / np.clip(np.asarray(d["ss_tot"]), 1e-12, None)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].scatter(range(len(ctx_ids)), per_ctx_skill, s=12)
    for i, c in enumerate(ctx_ids):
        axes[0].annotate(c, (i, per_ctx_skill[i]), fontsize=4, rotation=60)
    axes[0].set_xlabel("context (battery order)")
    axes[0].set_ylabel("per-context held-out skill (D, mean/mean)")
    axes[0].set_title(f"per-context skill @L{conv} — {regime}")
    xs, ys, labels = [], [], []
    for i, c in enumerate(ctx_ids):
        if c in cot_len_by_ctx:
            xs.append(cot_len_by_ctx[c])
            ys.append(per_ctx_skill[i])
            labels.append(c)
    if xs:
        axes[1].scatter(xs, ys, s=12)
        for x, y, c in zip(xs, ys, labels, strict=True):
            axes[1].annotate(c, (x, y), fontsize=4, rotation=30)
        rho = float(np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1])
        axes[1].set_title(f"CoT length vs per-context skill (Spearman ρ={rho:+.2f})")
    axes[1].set_xlabel("median CoT length (chars)")
    axes[1].set_ylabel("per-context held-out skill")
    savefig_paper(fig, f"issue_928/percontext_scatter_{regime}", dir=str(out_dir.parent))
    plt.close(fig)


def fig_lofo_rank_agreement(grid: dict, out_dir: Path, regime: str) -> None:
    """Exploratory: LOCO vs LOFO best-layer skill per arm (ordering check)."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    combo = "mean/mean"
    xs, ys, labels = [], [], []
    for arm in SIX_ARMS:
        loco = _cells(grid, arm, combo, "loco")
        lofo = _cells(grid, arm, combo, "lofo")
        if not loco or not lofo:
            continue
        xs.append(max(c["skill"] for c in loco))
        ys.append(max(c["skill"] for c in lofo))
        labels.append(ARM_LABELS[arm])
    ax.scatter(xs, ys, s=18)
    for x, y, la in zip(xs, ys, labels, strict=True):
        ax.annotate(la, (x, y), fontsize=6)
    lo = min(xs + ys) if xs else 0.0
    hi = max(xs + ys) if xs else 1.0
    ax.plot([lo, hi], [lo, hi], ls=":", lw=0.8, color="0.6")
    ax.set_xlabel("LOCO best-layer skill")
    ax.set_ylabel("LOFO best-layer skill")
    ax.set_title(f"fold agreement — {regime}")
    savefig_paper(fig, f"issue_928/lofo_rank_agreement_{regime}", dir=str(out_dir.parent))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #928 figures")
    ap.add_argument("--results", default=str(PROJECT_ROOT / "eval_results" / "issue_928"))
    ap.add_argument("--store", default=str(PROJECT_ROOT / "data" / "issue_928" / "store"))
    ap.add_argument(
        "--rollouts",
        default=str(PROJECT_ROOT / "data" / "issue_928" / "raw_completions" / "thinking_rollouts"),
    )
    ap.add_argument("--out", default=str(PROJECT_ROOT / "figures" / "issue_928"))
    ap.add_argument(
        "--upload-prefix",
        default=None,
        help="HF data-repo prefix for the figure files (round 2: git is the canonical "
        "figure home but the pod/GCE instance cannot commit — upload so the VM-side "
        "analyzer can pull + commit after instance teardown)",
    )
    args = ap.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    # savefig_paper writes to out_dir.parent/"issue_928" at EVERY call site — a
    # mismatched --out would make the upload loop iterate an EMPTY dir (silent
    # no-op upload; code-review r2 minor).
    assert out_dir.name == "issue_928", f"--out must end in issue_928 (savefig_paper): {out_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = load_json(results_dir / "recon_skill_grid.json")
    boot = blob["bootstrap"]
    manifest = load_json(Path(args.store) / "manifest.json")

    cot_len_by_ctx = fig_parse_and_lengths(manifest, Path(args.rollouts), out_dir)
    for regime in [r for r in ("avg_q", "indiv") if r in blob["results"]]:
        grid = blob["results"][regime]["grid"]
        nulls_path = results_dir / f"null_matrix_{regime}.json"
        nulls = load_json(nulls_path) if nulls_path.is_file() else {"null": {}}
        fig_skill_curves(grid, nulls, regime, out_dir)
        if regime in boot:
            fig_mediation_bars(boot, grid, regime, out_dir)
            fig_percontext_scatter(results_dir, grid, boot, regime, cot_len_by_ctx, out_dir)
        fig_lofo_rank_agreement(grid, out_dir, regime)
        if regime == "avg_q":
            fig_combo_heatmaps(grid, out_dir)
    if args.upload_prefix:
        names = sorted(p.name for p in out_dir.iterdir() if p.suffix in {".png", ".pdf", ".json"})
        upload_folder_scoped_verify(
            out_dir,
            args.upload_prefix,
            names,
            f"issue #928: figures ({len(names)} files)",
            allow_patterns=["*.png", "*.pdf", "*.json"],
        )
        logger.info("[phase=figures_upload] %d figure files -> %s", len(names), args.upload_prefix)
    # NOT [phase=done]: the run_all driver owns the terminal phase line.
    logger.info("[phase=figures_done] figures written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
