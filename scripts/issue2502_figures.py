"""Issue #2502 P5: hero figure + per-source matrix + exploratory dump (plan v6 S6).

Consumes the P4 fit deliverables (``eval_results/issue_2502/fits/<model>/
{fits_summary.json,percontext_recon.json}``), the P3-rel reliability ceilings
(``eval_results/issue_2502/reliability/model{K}/reliability_ceiling.json``),
and optional side inputs (decision.json, capture_meta jsons, dedup_report),
and renders:

  hero_matched_reldepth_r2      cross-model matched relative-depth held-out
                                R^2 (H1+H3): x = relative depth in [0,1]
                                (Model A all 28 layers, Model B its 8
                                full-attention layers), y = pooled held-out
                                R^2, per-layer bootstrap CI bands, identity+
                                learned-bias baseline (dashed), per-model
                                answer-reliability ceiling (dotted), and the
                                SELECTION-INHERITED best-layer 95% CI drawn at
                                each model's selected H3 layer (marker+bar;
                                the per-layer bands are pointwise/frozen —
                                a full-curve display takes no cross-layer max,
                                so selection inheritance applies to the
                                selected-layer read, which is drawn as such).
  persource_r2_matrix           source x model heatmap of per-source held-out
                                R^2 at the gate layer (within-source centering
                                PRIMARY), rows grouped by regime class (H2).
  explore_*                     over-produced exploratory dump for the
                                analyzer: kNN acc@k, LODO bars, map-vs-identity
                                per-context deltas, raw err-vs-sst scatter,
                                ceiling-vs-best-R^2, cap-hit fractions, dedup
                                drop counts (plan S6 items b-h).

Conventions (paper-plots SKILL + interim register): colorblind-safe Wong
palette; ONE color = ONE meaning across the whole dump (color = model for
model series, color = regime class for class-colored panels; linestyle/alpha
= arm); axes + ticks + legend + panel titles ONLY — provenance lives in the
``savefig_paper`` sidecar ``.meta.json``, never on the canvas. A
``figures_manifest.json`` records every rendered/skipped figure with inputs
and reproducibility metadata.

Required inputs fail loud; OPTIONAL inputs (reliability ceilings, decision,
capture metas, dedup report) skip their figure with a recorded manifest note
(production runs pass all of them; the P5 launch command in the plan wires
the defaults).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2502_fits as FT  # stdlib-only module top (constants + recon helpers)

ISSUE = 2502
BOOT_SEED = 2502

# ONE color = ONE meaning across the whole dump (Wong 2011 palette).
MODEL_COLOR = {"A": "#0072B2", "B": "#E69F00"}  # blue = Qwen2.5-7B, orange = Qwen3.5-9B
MODEL_LABEL = {"A": "Qwen2.5-7B-Instruct", "B": "Qwen3.5-9B"}
CLASS_COLOR = {
    "ordinary": "#009E73",  # bluish green
    "weird": "#D55E00",  # vermillion
    "near-distribution": "#CC79A7",  # reddish purple
    "idiosyncratic": "#F0E442",  # yellow
}
CLASS_ORDER = ("ordinary", "near-distribution", "weird", "idiosyncratic")


def _gc():
    import issue2502_gen_capture as GC

    return GC


def class_color(regime_class: str) -> str:
    return CLASS_COLOR.get(str(regime_class), "#999999")


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_inputs(fits_root: Path, key: str) -> tuple[dict, dict]:
    """(fits_summary, percontext_recon) for one model — REQUIRED, fail loud."""
    d = fits_root / "fits" / f"model{key}"
    for name in ("fits_summary.json", "percontext_recon.json"):
        if not (d / name).exists():
            raise FileNotFoundError(f"required P4 artifact missing: {d / name}")
    return load_json(d / "fits_summary.json"), load_json(d / "percontext_recon.json")


def optional_input(path: str | None, *, explicit: bool, what: str, notes: list[str]):
    """Load an optional input; explicit-but-missing fails loud, default-missing
    records a manifest note and returns None."""
    if path is None:
        notes.append(f"{what}: not provided — dependent figure skipped")
        return None
    p = Path(path)
    if not p.exists():
        if explicit:
            raise FileNotFoundError(f"{what} explicitly passed but missing: {p}")
        notes.append(f"{what}: default path {p} absent — dependent figure skipped")
        return None
    return load_json(p)


# ---------------------------------------------------------------------------
# Bootstrap reductions (batched counts@matrix GEMM — FT.bootstrap_counts)
# ---------------------------------------------------------------------------


def layer_band_table(recon: dict, hs_list: list[int], *, draws: int, seed: int):
    """Per-layer pooled-R^2 bootstrap percentile bands (pointwise, frozen per
    layer) + the SELECTION-INHERITED best-layer R^2 CI over ``hs_list``.

    Vectorized: one (draws, n) multiplicity matrix @ (n, L) err/sst stacks
    (the subset-sum GEMM form of `.claude/rules/vectorize-many-cell-fits.md`;
    reuses the u3 fit module's own bootstrap_counts/recon helpers so the
    figure bands and the gate CIs share one convention)."""
    import numpy as np

    mats = FT.recon_arrays_from_file(recon, hs_list)
    n = mats["err_map"].shape[0]
    counts, _ = FT.bootstrap_counts(n, draws, seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - (counts @ mats["err_map"]) / (counts @ mats["sst"])
    lo, hi = np.nanpercentile(r2, [2.5, 97.5], axis=0)
    inherited_best = np.nanmax(r2, axis=1)
    inh_ci = [float(x) for x in np.nanpercentile(inherited_best, [2.5, 97.5])]
    return {
        "hs": mats["hs"],
        "lo": [float(x) for x in lo],
        "hi": [float(x) for x in hi],
        "inherited_best_ci": inh_ci,
        "draws": draws,
    }


def layer_rows(summary: dict, hs_set: list[int]) -> list[dict]:
    by_hs = {int(row["hs"]): row for row in summary["layers"]}
    missing = [k for k in hs_set if k not in by_hs]
    if missing:
        raise RuntimeError(f"fits_summary layers table missing hs {missing}")
    return [by_hs[k] for k in sorted(hs_set)]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_hero(models: dict, reliability: dict, args, manifest: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y_min = 0.0
    for key in ("A", "B"):
        summary, recon = models[key]
        n_layers = FT.MODEL_N_LAYERS[key]
        # Model A: all captured layers; Model B: its 8 full-attention layers.
        disp = summary["candidate_sets"]["all"] if key == "A" else summary["candidate_sets"]["h3"]
        rows = layer_rows(summary, disp)
        x = [r["hs"] / n_layers for r in rows]
        y = [r["r2_test_map_pooled"] for r in rows]
        y_id = [r["r2_test_id_pooled"] for r in rows]
        band = layer_band_table(recon, disp, draws=args.boot_draws, seed=BOOT_SEED)
        c = MODEL_COLOR[key]
        ax.plot(x, y, "-o", color=c, ms=3.5, lw=1.6, label=f"{MODEL_LABEL[key]} map")
        ax.fill_between(x, band["lo"], band["hi"], color=c, alpha=0.15, lw=0)
        ax.plot(
            x, y_id, "--", color=c, lw=1.1, alpha=0.85, label=f"{MODEL_LABEL[key]} identity+bias"
        )
        rel = reliability.get(key)
        if rel is not None:
            per_layer = rel["per_layer"]
            xc, yc = [], []
            for k in sorted(disp):
                cell = per_layer.get(f"L{k:02d}")
                if cell is not None:
                    xc.append(k / n_layers)
                    yc.append(cell["ceiling_pooled"])
            if xc:
                ax.plot(
                    xc,
                    yc,
                    ":",
                    color=c,
                    lw=1.3,
                    label=f"{MODEL_LABEL[key]} reliability ceiling",
                )
        sel_hs = summary["selected"]["h3_hs"]
        sel_row = layer_rows(summary, [sel_hs])[0]
        h3_band = layer_band_table(
            recon, summary["candidate_sets"]["h3"], draws=args.boot_draws, seed=BOOT_SEED + 7
        )
        ci_lo, ci_hi = h3_band["inherited_best_ci"]
        yv = sel_row["r2_test_map_pooled"]
        ax.errorbar(
            [sel_hs / n_layers],
            [yv],
            yerr=[[max(0.0, yv - ci_lo)], [max(0.0, ci_hi - yv)]],
            fmt="*",
            color=c,
            ms=13,
            mec="black",
            mew=0.6,
            capsize=4,
            lw=1.4,
            zorder=5,
            label=f"{MODEL_LABEL[key]} selected layer (selection-inherited 95% CI)",
        )
        y_min = min([y_min, *band["lo"], ci_lo])
        manifest["hero"] = manifest.get("hero", {})
        manifest["hero"][key] = {
            "displayed_hs": sorted(disp),
            "selected_h3_hs": sel_hs,
            "selection_inherited_best_r2_ci": h3_band["inherited_best_ci"],
            "band_semantics": "per-layer pointwise bootstrap 95% band (frozen per layer); "
            "the selected-layer errorbar is the selection-inherited best-layer CI over "
            "the H3 candidate set",
        }
    ax.set_xlabel("relative depth (hidden-state index / n_layers)")
    ax.set_ylabel("pooled held-out R² (test partition)")
    ax.set_title("Context→answer map across depth, matched between model generations")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(bottom=max(-1.05, y_min - 0.05), top=1.05)
    ax.axhline(0.0, color="#666666", lw=0.7, alpha=0.6)
    ax.legend(fontsize=7, loc="best", frameon=True)
    return fig


def fig_persource_matrix(models: dict, manifest: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    per_model = {}
    for key in ("A", "B"):
        summary, _ = models[key]
        per_model[key] = summary["per_source_at_gate_layer"]
    sources = sorted(
        {s for d in per_model.values() for s in d},
        key=lambda s: (
            CLASS_ORDER.index(_src_class(per_model, s))
            if _src_class(per_model, s) in CLASS_ORDER
            else 99,
            s,
        ),
    )
    mat = np.full((len(sources), 2), np.nan)
    for j, key in enumerate(("A", "B")):
        for i, s in enumerate(sources):
            cell = per_model[key].get(s)
            if cell is not None:
                v = cell.get("r2_map_within_source")
                if v is not None and math.isfinite(v):
                    mat[i, j] = v
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(sources) + 1.6))
    finite = mat[np.isfinite(mat)]
    vmin = float(min(-0.05, finite.min())) if finite.size else -1.0
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=vmin, vmax=1.0)
    ax.set_xticks([0, 1], [MODEL_LABEL["A"], MODEL_LABEL["B"]], fontsize=8)
    labels = [f"[{_src_class(per_model, s)}] {s}" for s in sources]
    ax.set_yticks(range(len(sources)), labels, fontsize=7)
    for tick, s in zip(ax.get_yticklabels(), sources):
        tick.set_color(class_color(_src_class(per_model, s)))
    for i in range(len(sources)):
        for j in range(2):
            txt = "n/a" if not np.isfinite(mat[i, j]) else f"{mat[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color="white")
    ax.set_title("Per-source held-out R² at the gate layer (within-source centering)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="held-out R²")
    manifest["persource_matrix"] = {"sources": sources, "centering": "within-source (PRIMARY)"}
    return fig


def _src_class(per_model: dict, source: str) -> str:
    for d in per_model.values():
        cell = d.get(source)
        if cell is not None:
            return str(cell.get("regime_class"))
    return "unknown"


def fig_knn(models: dict, manifest: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, key in zip(axes, ("A", "B")):
        summary, _ = models[key]
        knn = summary.get("knn")
        if not knn:
            ax.set_title(f"{MODEL_LABEL[key]}: kNN block absent")
            continue
        ks = sorted(int(k) for k in knn["pooled"]["euclidean"]["acc_at_k"])
        width = 0.35
        xs = np.arange(len(ks))
        for off, metric, alpha in ((-width / 2, "euclidean", 0.95), (width / 2, "cosine", 0.55)):
            acc = [knn["pooled"][metric]["acc_at_k"][str(k)] for k in ks]
            ax.bar(
                xs + off,
                acc,
                width,
                color=MODEL_COLOR[key],
                alpha=alpha,
                label=f"{metric} (pooled)",
            )
        chance = [knn["pooled"]["euclidean"]["chance_at_k"][str(k)] for k in ks]
        ax.plot(xs, chance, "k--", lw=1.0, label="chance = k / n_pool")
        ax.set_xticks(xs, [f"k={k}" for k in ks])
        ax.set_title(f"{MODEL_LABEL[key]} (pool n={knn['pooled']['euclidean']['n_pool']})")
        ax.set_ylabel("retrieval acc@k")
        ax.legend(fontsize=7)
    fig.suptitle("kNN retrieval of the true answer vector (diagnostic)", fontsize=10)
    manifest["knn"] = {"note": "per-source kNN values live in fits_summary.knn.per_source"}
    return fig


def fig_lodo(models: dict, manifest: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=False)
    skipped: dict[str, list[str]] = {}
    for ax, key in zip(axes, ("A", "B")):
        summary, _ = models[key]
        lodo = summary.get("lodo") or {}
        rows = [(g, d) for g, d in sorted(lodo.items()) if "skipped" not in d]
        skipped[key] = [g for g, d in sorted(lodo.items()) if "skipped" in d]
        if not rows:
            ax.set_title(f"{MODEL_LABEL[key]}: no LODO folds")
            continue
        rows.sort(
            key=lambda gd: (
                CLASS_ORDER.index(str(gd[1].get("regime_class")))
                if str(gd[1].get("regime_class")) in CLASS_ORDER
                else 99,
                gd[0],
            )
        )
        xs = np.arange(len(rows))
        width = 0.38
        ax.bar(
            xs - width / 2,
            [d["r2_map"] for _, d in rows],
            width,
            color=MODEL_COLOR[key],
            label="map (left-out source)",
        )
        ax.bar(
            xs + width / 2,
            [d["r2_id"] for _, d in rows],
            width,
            color=MODEL_COLOR[key],
            alpha=0.45,
            hatch="//",
            label="identity+bias",
        )
        labels = [f"[{d.get('regime_class')}] {g}" for g, d in rows]
        ax.set_xticks(xs, labels, rotation=45, ha="right", fontsize=6.5)
        for tick, (_, d) in zip(ax.get_xticklabels(), rows):
            tick.set_color(class_color(str(d.get("regime_class"))))
        ax.axhline(0.0, color="#666666", lw=0.7)
        ax.set_ylabel("held-out R² on left-out source")
        ax.set_title(f"{MODEL_LABEL[key]} — leave-one-dataset-out transfer")
        ax.legend(fontsize=7)
    manifest["lodo"] = {"skipped_folds": skipped}
    return fig


def fig_delta_hist(models: dict, manifest: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, key in zip(axes, ("A", "B")):
        summary, recon = models[key]
        gate_hs = summary["selected"]["gate_hs"]
        arr = recon["layers"][f"L{gate_hs:02d}"]
        err_map = np.asarray(arr["err_map"], dtype=np.float64)
        err_id = np.asarray(arr["err_identity"], dtype=np.float64)
        sst = np.asarray(arr["sst_pooled"], dtype=np.float64)
        cls = np.asarray([str(c["regime_class"]) for c in recon["contexts"]])
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = (err_id - err_map) / sst
        delta = np.clip(delta, -2.0, 2.0)
        for c in CLASS_ORDER:
            m = (cls == c) & np.isfinite(delta)
            if m.sum() >= 2:
                ax.hist(
                    delta[m],
                    bins=30,
                    histtype="step",
                    lw=1.4,
                    color=class_color(c),
                    label=f"{c} (n={int(m.sum())})",
                )
        ax.axvline(0.0, color="#666666", lw=0.8)
        ax.set_xlabel("(identity err − map err) / pooled SST, per context (clipped ±2)")
        ax.set_title(f"{MODEL_LABEL[key]} @ gate layer hs={gate_hs}")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("contexts")
    manifest["delta_hist"] = {"clip": 2.0}
    return fig


def fig_err_scatter(models: dict, manifest: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, key in zip(axes, ("A", "B")):
        summary, recon = models[key]
        gate_hs = summary["selected"]["gate_hs"]
        arr = recon["layers"][f"L{gate_hs:02d}"]
        err = np.asarray(arr["err_map"], dtype=np.float64)
        sst = np.asarray(arr["sst_pooled"], dtype=np.float64)
        cls = [str(c["regime_class"]) for c in recon["contexts"]]
        for c in CLASS_ORDER:
            m = np.asarray([x == c for x in cls])
            if m.any():
                ax.scatter(sst[m], err[m], s=9, alpha=0.65, color=class_color(c), label=c, lw=0)
        pos = np.concatenate([sst[sst > 0], err[err > 0]])
        lim_lo = max(1e-6, float(pos.min()) * 0.5) if pos.size else 1e-6
        lim_hi = max(float(sst.max()), float(err.max()), lim_lo * 10) * 2.0
        grid = np.geomspace(lim_lo, lim_hi, 32)
        ax.plot(grid, grid, "k--", lw=0.9, label="err = SST (R²=0)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("per-context pooled SST (‖y − ȳ‖²)")
        ax.set_ylabel("per-context map error (‖y − ŷ‖²)")
        ax.set_title(f"{MODEL_LABEL[key]} @ gate layer hs={gate_hs}")
        ax.legend(fontsize=6.5)
    manifest["err_scatter"] = {"note": "raw per-unit companion (below diagonal = R²>0)"}
    return fig


def fig_ceiling_vs_best(models: dict, reliability: dict, decision, manifest: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    xs = np.arange(2)
    width = 0.34
    for j, key in enumerate(("A", "B")):
        summary, _ = models[key]
        sel_hs = summary["selected"]["h3_hs"]
        best = summary["r2_at_h3_layer"]["map_pooled"]
        rel = reliability.get(key)
        ceil = rel["per_layer"].get(f"L{sel_hs:02d}", {}).get("ceiling_pooled") if rel else None
        ax.bar(j - width / 2, best, width, color=MODEL_COLOR[key], label=None)
        if ceil is not None:
            ax.bar(
                j + width / 2,
                ceil,
                width,
                color=MODEL_COLOR[key],
                alpha=0.45,
                hatch="..",
            )
        manifest.setdefault("ceiling_vs_best", {})[key] = {
            "h3_hs": sel_hs,
            "best_r2": best,
            "ceiling_pooled": ceil,
            "ratio": (best / ceil) if ceil else None,
        }
    ax.set_xticks(xs, [MODEL_LABEL["A"], MODEL_LABEL["B"]], fontsize=8)
    ax.set_ylabel("R² at selected H3 layer")
    ax.set_title("Best map R² (solid) vs answer-reliability ceiling (hatched)")
    return fig


def fig_caphit(meta_a, meta_b, manifest: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    vals = {}
    for j, (key, meta) in enumerate((("A", meta_a), ("B", meta_b))):
        frac = meta["totals"]["cap_hit_fraction"]
        vals[key] = frac
        ax.bar(j, frac, 0.55, color=MODEL_COLOR[key])
    ax.axhline(0.02, color="#D55E00", lw=1.2, ls="--", label="2% re-generation trigger")
    ax.set_xticks([0, 1], [MODEL_LABEL["A"], MODEL_LABEL["B"]], fontsize=8)
    ax.set_ylabel("fraction of generations hitting the token cap")
    ax.set_title("Generation cap-hit fraction per model")
    ax.legend(fontsize=7)
    manifest["caphit"] = vals
    return fig


def fig_dedup(report: dict, manifest: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    dd = report["dedup"]
    per_src = dd.get("per_source_dropped", {})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    stages = ["LSH candidates", "exact-Jaccard ≥0.8 dropped", "candidate, kept (<0.8)"]
    counts = [
        dd["n_had_lsh_candidate"],
        dd["n_confirmed_dropped"],
        dd.get("n_lsh_false_positive", dd["n_had_lsh_candidate"] - dd["n_confirmed_dropped"]),
    ]
    axes[0].bar(range(3), counts, color=["#999999", "#D55E00", "#009E73"])
    axes[0].set_xticks(range(3), stages, rotation=20, ha="right", fontsize=7)
    axes[0].set_ylabel("contexts")
    axes[0].set_title("Two-stage dedup: candidate vs confirm")
    if per_src:
        srcs = sorted(per_src, key=lambda s: -per_src[s])[:25]
        axes[1].bar(np.arange(len(srcs)), [per_src[s] for s in srcs], color="#D55E00")
        axes[1].set_xticks(np.arange(len(srcs)), srcs, rotation=45, ha="right", fontsize=6.5)
        axes[1].set_title("Confirmed near-duplicate drops per source")
        axes[1].set_ylabel("dropped contexts")
    else:
        axes[1].set_title("No per-source drop counts in report")
    manifest["dedup"] = {"totals": {s: c for s, c in zip(stages, counts)}}
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def render_all(args) -> dict:
    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    GC = _gc()
    fits_root = Path(args.fits_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    models = {k: model_inputs(fits_root, k) for k in ("A", "B")}
    reliability = {}
    for key, path, explicit in (
        ("A", args.reliability_a, args.reliability_a is not None),
        ("B", args.reliability_b, args.reliability_b is not None),
    ):
        default = fits_root / "reliability" / f"model{key}" / "reliability_ceiling.json"
        p = path if path is not None else (str(default) if default.exists() else None)
        if p is None:
            notes.append(f"reliability model{key}: {default} absent — ceiling lines skipped")
        reliability[key] = optional_input(
            p, explicit=explicit, what=f"reliability model{key}", notes=notes
        )
    decision = optional_input(
        args.decision
        if args.decision is not None
        else (
            str(fits_root / "fits" / "decision.json")
            if (fits_root / "fits" / "decision.json").exists()
            else None
        ),
        explicit=args.decision is not None,
        what="decision.json",
        notes=notes,
    )
    meta_a = optional_input(
        args.capture_meta_a,
        explicit=args.capture_meta_a is not None,
        what="capture_meta model A",
        notes=notes,
    )
    meta_b = optional_input(
        args.capture_meta_b,
        explicit=args.capture_meta_b is not None,
        what="capture_meta model B",
        notes=notes,
    )
    dedup = optional_input(
        args.dedup_report,
        explicit=args.dedup_report is not None,
        what="dedup_report",
        notes=notes,
    )

    manifest: dict = {
        "meta": GC.run_metadata({"artifact": "figures_manifest", "issue": ISSUE}),
        "inputs": {
            "fits_root": str(fits_root),
            "boot_draws": args.boot_draws,
            "decision_verdict": (decision or {}).get("verdict"),
        },
        "notes": notes,
        "figures": {},
    }

    def _save(fig, stem: str, block: str):
        paths = savefig_paper(fig, stem, dir=out_dir)
        plt.close(fig)
        manifest["figures"][stem] = {
            "files": {k: str(v) for k, v in paths.items()},
            "manifest_block": block,
        }
        print(f"[figures] wrote {paths.get('png', paths)}", flush=True)

    _save(fig_hero(models, reliability, args, manifest), "hero_matched_reldepth_r2", "hero")
    _save(fig_persource_matrix(models, manifest), "persource_r2_matrix", "persource_matrix")
    _save(fig_knn(models, manifest), "explore_persource_knn", "knn")
    _save(fig_lodo(models, manifest), "explore_lodo_bars", "lodo")
    _save(fig_delta_hist(models, manifest), "explore_delta_identity_hist", "delta_hist")
    _save(fig_err_scatter(models, manifest), "explore_percontext_err_vs_sst", "err_scatter")
    _save(
        fig_ceiling_vs_best(models, reliability, decision, manifest),
        "explore_ceiling_vs_bestr2",
        "ceiling_vs_best",
    )
    if meta_a is not None and meta_b is not None:
        _save(fig_caphit(meta_a, meta_b, manifest), "explore_caphit", "caphit")
    else:
        manifest["figures"]["explore_caphit"] = {"skipped": "capture_meta inputs absent"}
    if dedup is not None and "dedup" in dedup:
        _save(fig_dedup(dedup, manifest), "explore_dedup_drops", "dedup")
    else:
        manifest["figures"]["explore_dedup_drops"] = {"skipped": "dedup_report absent"}

    GC.atomic_write_json(out_dir / "figures_manifest.json", manifest)
    n_rendered = sum(1 for v in manifest["figures"].values() if "files" in v)
    print(f"[figures] {n_rendered} figures + manifest -> {out_dir}", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--fits-root",
        default=str(_REPO_ROOT / "eval_results" / "issue_2502"),
        help="Root holding fits/model{A,B}/ + reliability/model{K}/ (P4/P3-rel outputs).",
    )
    ap.add_argument("--out-dir", default=str(_REPO_ROOT / "figures" / "issue_2502"))
    ap.add_argument("--boot-draws", type=int, default=2000, help="per-layer CI band draws")
    ap.add_argument("--reliability-a", default=None, help="override reliability_ceiling.json (A)")
    ap.add_argument("--reliability-b", default=None, help="override reliability_ceiling.json (B)")
    ap.add_argument("--decision", default=None, help="override fits/decision.json path")
    ap.add_argument("--capture-meta-a", default=None, help="local capture_meta.json (model A)")
    ap.add_argument("--capture-meta-b", default=None, help="local capture_meta.json (model B)")
    ap.add_argument("--dedup-report", default=None, help="local dedup_report.json (P0)")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("issue2502_figures: import-check OK", flush=True)
        return 0
    # load_dotenv BEFORE matplotlib/numpy imports (thread caps freeze at import).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    render_all(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
