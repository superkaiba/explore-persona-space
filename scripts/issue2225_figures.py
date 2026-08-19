#!/usr/bin/env python3
"""Issue #2225 unit 5 — figures (plan §6 "Figures to produce").

HERO: per-dataset coefficient-response curves (trait score + coherence + MMLU
vs coefficient; paper-Fig-5B style) overlaying the Paper method vs Context
extract+steer — single-layer and multi-layer panels.

Exploratory dump (over-produced by design): matched-coherence bar chart across
conditions × datasets with per-question scatter (points labeled by question
index); direction × token-mask attribution grid (evil II); registered-contrast
forest plot (frozen + selection-inherited CIs); probe-shift and
projection-shift bars per condition; per-layer probe profiles; narrow-domain
retention bars; cap-hit + judge-drop diagnostics (incl. per-arm rule-28
``n_api_refusal``).

Inputs: ``eval_results/issue_2225/{analysis,trait_scores}/*.json`` (P4/P5
outputs). Outputs: ``figures/issue_2225/*.png`` (+ ``.pdf`` + per-figure
``.meta.json`` provenance sidecars via ``savefig_paper``). Every figure
follows the paper-plots conventions: one color = one meaning (per-config
colors fixed module-wide), plain-English condition labels (no bare config
codes), no explanatory annotations/arrows, raw per-unit data alongside
aggregates. GPU-free; renders headless (Agg).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# scripts/ on sys.path so sibling issue2225_* modules resolve in script mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847): load_dotenv() setdefaults OMP/MKL/OPENBLAS/
# NUMEXPR before numpy/matplotlib import so the caps bind in-process.
load_dotenv()

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

# ── condition naming (plain English everywhere; bare codes never rendered) ────

CONFIG_ORDER = ("A", "B", "C", "D", "E", "F", "G", "I", "P", "H")
CONFIG_LABEL = {
    "A": "Paper single-layer (response-avg dir, all tokens)",
    "B": "Paper all-layer, 28 incremental (response-avg dir, all tokens)",
    "C": "Context single-layer (context-end dir, context tokens)",
    "D": "Context middle-band, 9 layers (context-end dir, context tokens)",
    "E": "Context all-layer, 28 (context-end dir, context tokens)",
    "F": "Paper dir on context tokens",
    "G": "Context dir on all tokens",
    "I": "Paper dir on response tokens",
    "P": "Prefix dir on prefix tokens",
    "H": "Preventative prompt (no steering)",
}
CONFIG_SHORT = {
    "A": "Paper 1-layer",
    "B": "Paper all-layer (28)",
    "C": "Context 1-layer",
    "D": "Context middle-band (9)",
    "E": "Context all-layer (28)",
    "F": "Paper dir, ctx tokens",
    "G": "Context dir, all tokens",
    "I": "Paper dir, resp tokens",
    "P": "Prefix dir, prefix tokens",
    "H": "Preventative prompt",
}
DATASET_ORDER = ("evil", "sycophancy", "hallucination", "mistake_opinions")
DATASET_LABEL = {
    "evil": "Evil",
    "sycophancy": "Sycophancy",
    "hallucination": "Hallucination",
    "mistake_opinions": "Mistaken opinions",
}
TRAITS = ("evil", "sycophancy", "hallucination")

# One color = one meaning: a config keeps its color in EVERY figure.
_PALETTE = paper_palette(len(CONFIG_ORDER))
CONFIG_COLOR = {cfg: _PALETTE[i] for i, cfg in enumerate(CONFIG_ORDER)}

COHERENCE_THRESHOLD = 80.0
CAP_HIT_REGEN_TRIGGER = 0.02  # reported re-gen trigger (plan §4.6)


def pretty_tag(tag: str, *, with_dataset: bool = False) -> str:
    """Plain-English label for an eval-target tag (never a bare slug)."""
    if tag == "base":
        return "Base model"
    if tag.startswith("baseft_"):
        return f"Unsteered finetune ({DATASET_LABEL.get(tag[7:], tag[7:])})"
    parts = tag.split("__")
    cfg = parts[0]
    label = CONFIG_SHORT.get(cfg, cfg)
    if with_dataset and len(parts) >= 2:
        label = f"{label} — {DATASET_LABEL.get(parts[1], parts[1])}"
    if len(parts) == 3 and parts[2].startswith("c"):
        label = f"{label} @ {parts[2][1:]}"
    return label


# ── input loading ─────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"figure input missing: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


def _selection(eval_root: Path) -> dict:
    return _load_json(eval_root / "analysis" / "selection.json")["selection"]


def _datasets_in(selection: dict) -> list[str]:
    present = {v["dataset"] for v in selection.values()}
    return [d for d in DATASET_ORDER if d in present]


def _configs_in(selection: dict, dataset: str, subset: Sequence[str] | None = None) -> list[str]:
    order = subset if subset is not None else CONFIG_ORDER
    return [c for c in order if f"{c}_{dataset}" in selection]


def _curve_xy(entry: dict, metric: str) -> tuple[list[float], list[float]]:
    """(coefs ascending, metric values) from a selection curve; 'prompt' excluded."""
    pts = sorted(
        (float(k), _nan(v.get(metric))) for k, v in entry["curve"].items() if k != "prompt"
    )
    return [p[0] for p in pts], [p[1] for p in pts]


def _selected_curve_point(entry: dict, metric: str) -> float:
    sel = entry.get("selected_coef")
    key = "prompt" if entry.get("note", "").startswith("prompt-mode") else str(sel)
    if key not in entry["curve"]:
        return float("nan")
    return _nan(entry["curve"][key].get(metric))


# ── HERO: coefficient-response curves ─────────────────────────────────────────


def _coef_response_fig(selection: dict, configs: Sequence[str], suptitle: str) -> plt.Figure:
    datasets = _datasets_in(selection)
    metrics = (
        ("trait_mean", "Trait score (0–100)"),
        ("coherence_mean", "Coherence (0–100)"),
        ("mmlu_acc", "MMLU accuracy"),
    )
    fig, axes = plt.subplots(
        len(datasets), len(metrics), figsize=(12, 2.8 * len(datasets)), squeeze=False
    )
    for di, dataset in enumerate(datasets):
        for mi, (metric, ylabel) in enumerate(metrics):
            ax = axes[di][mi]
            for cfg in _configs_in(selection, dataset, configs):
                entry = selection[f"{cfg}_{dataset}"]
                xs, ys = _curve_xy(entry, metric)
                if not xs:
                    continue
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    color=CONFIG_COLOR[cfg],
                    label=CONFIG_SHORT[cfg] if di == 0 and mi == 0 else None,
                )
            if metric == "coherence_mean":
                ax.axhline(COHERENCE_THRESHOLD, ls="--", lw=0.8, color="0.6")
            if di == len(datasets) - 1:
                ax.set_xlabel("Steering coefficient")
            if mi == 0:
                ax.set_ylabel(f"{DATASET_LABEL[dataset]}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
    fig.suptitle(suptitle)
    fig.legend(loc="lower center", ncol=min(len(configs), 5), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


def build_hero_single_layer(eval_root: Path) -> dict[str, plt.Figure]:
    selection = _selection(eval_root)
    fig = _coef_response_fig(
        selection,
        ("A", "C"),
        "Coefficient response — Paper vs Context single-layer preventative steering",
    )
    return {"hero_coef_response_single_layer": fig}


def build_hero_multilayer(eval_root: Path) -> dict[str, plt.Figure]:
    selection = _selection(eval_root)
    fig = _coef_response_fig(
        selection,
        ("B", "D", "E"),
        "Coefficient response — multi-layer preventative steering",
    )
    return {"hero_coef_response_multilayer": fig}


# ── matched-coherence bars + per-question scatter ─────────────────────────────


def _arm_path(eval_root: Path, sub: str, config: str, dataset: str, coef) -> Path:
    coef_tag = "prompt" if coef is None else str(coef)
    return eval_root / sub / f"{config}_{dataset}_{coef_tag}.json"


def build_matched_coherence_bars(eval_root: Path) -> dict[str, plt.Figure]:
    selection = _selection(eval_root)
    datasets = _datasets_in(selection)
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 3.2 * len(datasets)), squeeze=False)
    rng = np.random.default_rng(2225)  # deterministic scatter jitter
    for di, dataset in enumerate(datasets):
        ax = axes[di][0]
        trait = None
        configs = _configs_in(selection, dataset)
        for xi, cfg in enumerate(configs):
            entry = selection[f"{cfg}_{dataset}"]
            trait = entry["steered_trait"]
            bar_val = _selected_curve_point(entry, "trait_mean")
            if np.isfinite(bar_val):
                ax.bar(xi, bar_val, color=CONFIG_COLOR[cfg], alpha=0.55, width=0.7)
            # per-question scatter at the selected coefficient (points labeled)
            sel = (
                None if entry.get("note", "").startswith("prompt-mode") else entry["selected_coef"]
            )
            if sel is None and not entry.get("note", "").startswith("prompt-mode"):
                continue  # no coherent coefficient — bar absent, note carried in JSON
            arm_file = _arm_path(eval_root, "trait_scores", cfg, dataset, sel)
            if not arm_file.exists():
                continue
            block = _load_json(arm_file)["traits"].get(trait)
            if block is None:
                continue
            for q in block["per_question"]:
                if q["mean"] is None:
                    continue
                x = xi + float(rng.uniform(-0.22, 0.22))
                ax.scatter(x, q["mean"], s=9, color=CONFIG_COLOR[cfg], zorder=3)
                ax.text(x, q["mean"], str(q["question_idx"]), fontsize=4, alpha=0.7)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([CONFIG_SHORT[c] for c in configs], rotation=30, ha="right")
        ax.set_ylabel(f"{DATASET_LABEL[dataset]}\ntrait score (0–100)")
    fig.suptitle(
        "Trait expression at the matched-coherence coefficient "
        "(bars = arm mean; points = per-question means, labeled by question index)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return {"matched_coherence_bars": fig}


# ── attribution grid: direction × token mask (evil II) ────────────────────────

_GRID_CELLS = {  # (direction row, mask col) -> config
    ("E1", "all"): "A",
    ("E1", "context"): "F",
    ("E1", "response"): "I",
    ("E2", "all"): "G",
    ("E2", "context"): "C",
}
_DIR_LABEL = {"E1": "Response-avg direction", "E2": "Context-end direction"}
_MASK_LABEL = {"all": "All tokens", "context": "Context tokens", "response": "Response tokens"}


def build_attribution_grid(eval_root: Path) -> dict[str, plt.Figure]:
    selection = _selection(eval_root)
    dirs = ("E1", "E2")
    masks = ("all", "context", "response")
    grid = np.full((len(dirs), len(masks)), np.nan)
    for (d, m), cfg in _GRID_CELLS.items():
        entry = selection.get(f"{cfg}_evil")
        if entry is not None:
            grid[dirs.index(d), masks.index(m)] = _selected_curve_point(entry, "trait_mean")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(masks)))
    ax.set_xticklabels([_MASK_LABEL[m] for m in masks])
    ax.set_yticks(range(len(dirs)))
    ax.set_yticklabels([_DIR_LABEL[d] for d in dirs])
    fig.colorbar(im, ax=ax, label="Trait score at matched-coherence coefficient")
    ax.set_title("Direction × token-mask attribution (evil corpus)", loc="left")
    # NO tight_layout here: under set_paper_style a post-colorbar layout-engine
    # switch raises RuntimeError (matplotlib refuses the engine change).
    return {"attribution_grid_evil": fig}


# ── registered-contrast forest plot ───────────────────────────────────────────


def _ci_offsets(point: float, lo: float, hi: float) -> tuple[float, float]:
    """Non-negative errorbar offsets (matplotlib xerr contract; clamped)."""
    return float(np.maximum(0.0, point - lo)), float(np.maximum(0.0, hi - point))


def build_contrast_forest(eval_root: Path) -> dict[str, plt.Figure]:
    contrasts = _load_json(eval_root / "analysis" / "contrasts.json")["contrasts"]
    rows: list[tuple[str, dict]] = []
    for cname, cblock in sorted(contrasts.items()):
        for dataset, dblock in cblock["per_dataset"].items():
            rows.append(
                (f"{cname.replace('_vs_', ' vs ')} — {DATASET_LABEL.get(dataset, dataset)}", dblock)
            )
        if cblock.get("pooled", {}).get("frozen"):
            rows.append((f"{cname.replace('_vs_', ' vs ')} — pooled", cblock["pooled"]))
    fig, ax = plt.subplots(figsize=(8, 0.6 * max(4, len(rows)) + 1.2))
    palette = paper_palette(2)
    y = 0
    ylabels: list[str] = []
    for label, block in rows:
        for vi, variant in enumerate(("frozen", "selection_inherited")):
            v = block.get(variant)
            if not v or v.get("delta_point") is None or not v.get("ci95"):
                continue
            point = float(v["delta_point"])
            lo, hi = (float(x) for x in v["ci95"])
            e_lo, e_hi = _ci_offsets(point, lo, hi)
            ax.errorbar(
                point,
                y + 0.18 * vi,
                xerr=[[e_lo], [e_hi]],
                fmt="o",
                color=palette[vi],
                label=(
                    ("Frozen selection" if variant == "frozen" else "Selection-inherited")
                    if y == 0
                    else None
                ),
            )
        ylabels.append(label)
        y += 1
    ax.axvline(0.0, ls="--", lw=0.8, color="0.6")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("Δ trait score (Context − Paper) at matched-coherence coefficients")
    ax.legend(frameon=False, loc="best")
    ax.set_title("Registered contrasts — question-paired bootstrap 95% CIs", loc="left")
    fig.tight_layout()
    return {"contrast_forest": fig}


# ── probe / projection shift bars + layer profiles ────────────────────────────


def _selected_cell_tags(selection: dict, trait: str) -> list[str]:
    """Selected-coefficient cell tag per config for the trait's own dataset."""
    dataset = trait  # single-trait datasets share the trait name
    tags = []
    for cfg in CONFIG_ORDER:
        entry = selection.get(f"{cfg}_{dataset}")
        if entry is None:
            continue
        if entry.get("note", "").startswith("prompt-mode"):
            tags.append(f"{cfg}__{dataset}")
        elif entry["selected_coef"] is not None:
            tags.append(f"{cfg}__{dataset}__c{entry['selected_coef']}")
    return tags


def build_probe_shift_bars(eval_root: Path) -> dict[str, plt.Figure]:
    probe = _load_json(eval_root / "analysis" / "probe_shifts.json")
    selection = _selection(eval_root)
    shifts = probe["shifts"]
    variants = ("full", "orth_E1", "orth_E2", "orth_E3")
    vpalette = paper_palette(len(variants))
    figs: dict[str, plt.Figure] = {}
    for trait in TRAITS:
        tags = [
            t
            for t in _selected_cell_tags(selection, trait) + [f"baseft_{trait}", "base"]
            if f"{t}__{trait}" in shifts
        ]
        if not tags:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(tags)), 3.6))
        width = 0.8 / len(variants)
        for vi, variant in enumerate(variants):
            xs, ys = [], []
            for ti, tag in enumerate(tags):
                v = shifts[f"{tag}__{trait}"]["variants"].get(variant)
                if v is None:
                    continue
                xs.append(ti + (vi - (len(variants) - 1) / 2) * width)
                ys.append(_nan(v["shift_l1"]))
            ax.bar(
                xs,
                ys,
                width=width,
                color=vpalette[vi],
                label=variant.replace("orth_", "orthogonal to "),
            )
        ax.axhline(0.0, lw=0.8, color="0.4")
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels([pretty_tag(t) for t in tags], rotation=30, ha="right")
        ax.set_ylabel("Probe-score shift vs base (steering layer)")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"Off-direction probe shift — {DATASET_LABEL[trait]}", loc="left")
        fig.tight_layout()
        figs[f"probe_shift_bars_{trait}"] = fig
    return figs


def build_projection_shift_bars(eval_root: Path) -> dict[str, plt.Figure]:
    proj = _load_json(eval_root / "analysis" / "projection_shifts.json")
    selection = _selection(eval_root)
    shifts = proj["shifts"]
    positions = ("response_avg", "context_end", "prefix_end")
    pos_label = {
        "response_avg": "response-avg → paper dir",
        "context_end": "context-end → context dir",
        "prefix_end": "prefix-end → context dir",
    }
    ppalette = paper_palette(len(positions))
    figs: dict[str, plt.Figure] = {}
    for trait in TRAITS:
        tags = [
            t
            for t in _selected_cell_tags(selection, trait) + [f"baseft_{trait}", "base"]
            if f"{t}__{trait}" in shifts
        ]
        if not tags:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(tags)), 3.6))
        width = 0.8 / len(positions)
        for pi, pos in enumerate(positions):
            xs, ys = [], []
            for ti, tag in enumerate(tags):
                v = shifts[f"{tag}__{trait}"]["positions"].get(pos)
                if v is None:
                    continue
                xs.append(ti + (pi - (len(positions) - 1) / 2) * width)
                ys.append(_nan(v["shift_l1"]))
            ax.bar(xs, ys, width=width, color=ppalette[pi], label=pos_label[pos])
        ax.axhline(0.0, lw=0.8, color="0.4")
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels([pretty_tag(t) for t in tags], rotation=30, ha="right")
        ax.set_ylabel("Δ mean projection vs base (steering layer)")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"Projection shift — {DATASET_LABEL[trait]}", loc="left")
        fig.tight_layout()
        figs[f"projection_shift_bars_{trait}"] = fig
    return figs


def build_probe_layer_profiles(eval_root: Path) -> dict[str, plt.Figure]:
    probe = _load_json(eval_root / "analysis" / "probe_shifts.json")
    selection = _selection(eval_root)
    shifts = probe["shifts"]
    figs: dict[str, plt.Figure] = {}
    for trait in TRAITS:
        tags = [
            t
            for t in _selected_cell_tags(selection, trait) + [f"baseft_{trait}"]
            if f"{t}__{trait}" in shifts
        ]
        if not tags:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for tag in tags:
            row = shifts[f"{tag}__{trait}"]
            prof = row["variants"]["full"]["shift_per_layer"]
            cfg = tag.split("__")[0]
            color = CONFIG_COLOR.get(cfg, "0.3")
            ax.plot(range(len(prof)), prof, color=color, label=pretty_tag(tag), lw=1.2)
        l1 = shifts[f"{tags[0]}__{trait}"]["l1_layer_idx"]
        ax.axvline(l1, ls="--", lw=0.8, color="0.6")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Probe-score shift vs base")
        ax.legend(frameon=False, fontsize=6, ncol=2)
        ax.set_title(f"Per-layer probe-shift profile — {DATASET_LABEL[trait]}", loc="left")
        fig.tight_layout()
        figs[f"probe_layer_profiles_{trait}"] = fig
    return figs


# ── narrow-domain retention ───────────────────────────────────────────────────


def build_narrow_retention(eval_root: Path) -> dict[str, plt.Figure]:
    narrow = _load_json(eval_root / "analysis" / "narrow_retention.json")["per_arm"]
    if not narrow:
        raise FileNotFoundError("narrow_retention.json carries zero arms")
    rows = sorted(narrow.values(), key=lambda r: -(r["mistake_style_rate"] or 0.0))
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(rows)), 3.6))
    for xi, r in enumerate(rows):
        cfg = r["target_tag"].split("__")[0]
        ax.bar(
            xi,
            _nan(r["mistake_style_rate"]),
            color=CONFIG_COLOR.get(cfg, "0.5"),
            width=0.7,
        )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([pretty_tag(r["target_tag"]) for r in rows], rotation=30, ha="right")
    ax.set_ylabel("Mistake-style response rate (judge > 50)")
    ax.set_title("Narrow-domain retention — training-distribution opinions questions", loc="left")
    fig.tight_layout()
    return {"narrow_retention_bars": fig}


# ── cap-hit + judge-drop diagnostics ──────────────────────────────────────────


def build_judge_diagnostics(eval_root: Path) -> dict[str, plt.Figure]:
    digest = _load_json(eval_root / "judge_digest.json")
    rows = digest["per_arm"]
    if not rows:
        raise FileNotFoundError("judge_digest.json carries zero per-arm rows")
    # cap-hit fractions live on the assembled arm files' trait blocks
    cap_hits: list[float] = []
    for sub in ("trait_scores", "coherence"):
        d = eval_root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            arm = _load_json(p)
            for block in arm.get("traits", {}).values():
                ch = block.get("cap_hit_fraction")
                if ch is not None:
                    cap_hits.append(float(ch))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    if cap_hits:
        ax1.scatter(range(len(sorted(cap_hits))), sorted(cap_hits), s=10, color=paper_palette(1)[0])
    ax1.axhline(CAP_HIT_REGEN_TRIGGER, ls="--", lw=0.8, color="0.6")
    ax1.set_xlabel("Eval unit (rank by cap-hit fraction)")
    ax1.set_ylabel("Generation cap-hit fraction")
    ax1.set_title("Cap-hit fractions vs the 2% re-gen trigger", loc="left")

    classes = ("n_api_refusal", "n_transport_lost", "n_content_dropped")
    class_label = {
        "n_api_refusal": "API refusal (rule 28)",
        "n_transport_lost": "transport lost",
        "n_content_dropped": "content dropped",
    }
    cpalette = paper_palette(len(classes))
    for ci, cls in enumerate(classes):
        fracs = sorted(
            (r.get(cls, 0) / r["n_total_draws"]) if r.get("n_total_draws") else 0.0 for r in rows
        )
        ax2.plot(range(len(fracs)), fracs, color=cpalette[ci], label=class_label[cls], lw=1.2)
    ax2.set_xlabel("Judge unit (rank by drop fraction)")
    ax2.set_ylabel("Dropped-draw fraction")
    ax2.legend(frameon=False, fontsize=7)
    ax2.set_title("Judge drop classes per unit", loc="left")
    fig.tight_layout()
    return {"judge_diagnostics": fig}


# ── registry + CLI ────────────────────────────────────────────────────────────

FIGURES: dict[str, Callable[[Path], dict[str, plt.Figure]]] = {
    "hero_single_layer": build_hero_single_layer,
    "hero_multilayer": build_hero_multilayer,
    "matched_coherence_bars": build_matched_coherence_bars,
    "attribution_grid": build_attribution_grid,
    "contrast_forest": build_contrast_forest,
    "probe_shift_bars": build_probe_shift_bars,
    "projection_shift_bars": build_projection_shift_bars,
    "probe_layer_profiles": build_probe_layer_profiles,
    "narrow_retention": build_narrow_retention,
    "judge_diagnostics": build_judge_diagnostics,
}


def render_all(
    eval_root: Path, out_dir: Path, only: Sequence[str] | None = None
) -> tuple[dict[str, Path], list[str]]:
    """Render every registered figure; returns (stem -> png path, failures).

    Failures are collected (not raised per-figure) so one missing input cannot
    hide the rest; the CLI exits non-zero when any figure failed (fail-loud).
    """
    set_paper_style()
    written: dict[str, Path] = {}
    failures: list[str] = []
    for name, builder in FIGURES.items():
        if only and name not in only:
            continue
        try:
            figs = builder(eval_root)
            if not figs:
                raise ValueError(f"builder {name} produced zero figures")
            for stem, fig in figs.items():
                paths = savefig_paper(fig, stem, dir=out_dir)
                plt.close(fig)
                written[stem] = paths["png"]
                print(f"[figures] {name}: {paths['png']}", flush=True)
        except Exception as e:  # collected + re-raised at exit — never silent
            failures.append(f"{name}: {type(e).__name__}: {e}")
            print(f"[figures] FAILED {name}: {type(e).__name__}: {e}", flush=True)
    return written, failures


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2225 unit-5 figures (plan §6).")
    ap.add_argument("--eval-root", default="eval_results/issue_2225")
    ap.add_argument("--out-dir", default="figures/issue_2225")
    ap.add_argument("--figures", default=None, help="comma-separated builder subset")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        assert set(FIGURES), "empty figure registry"
        print("[issue2225-figures] import-check OK", flush=True)
        return 0
    only = [s.strip() for s in args.figures.split(",") if s.strip()] if args.figures else None
    if only:
        unknown = [s for s in only if s not in FIGURES]
        if unknown:
            raise SystemExit(f"unknown figure builder(s): {unknown} (have {sorted(FIGURES)})")
    written, failures = render_all(Path(args.eval_root), Path(args.out_dir), only)
    print(f"[figures] wrote {len(written)} figures -> {args.out_dir}", flush=True)
    if failures:
        print("[figures] FAILURES:\n  " + "\n  ".join(failures), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
