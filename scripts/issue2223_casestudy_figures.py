"""Issue #2223 — case-study replay figures (drift-over-turns + harm-over-turns).

Reads the per-cell replay JSONs (+ optional judged scores) written by
``issue2223_casestudy_replay.py`` (``<out-root>/<model_slug>/<scenario>/…``)
and emits, PER MODEL, per scenario x layer config, into
``<fig-dir>/<model_slug>/``:

  drift_<scenario>_<layers>:  2 rows x 3 readout panels (answer-token mean /
      context vector / prefix vector), x = turn, y = Lu assistant-axis
      projection, one line per arm. Row 1 (PRIMARY) = the LAYER-32
      projection (the paper's steering-selected mid layer); row 2
      (secondary) = the PAPER-BAND layer mean. When no cell stores layer 32
      (tiny smoke models), only the band row renders (logged).
  harm_<scenario>_<layers>:   per-turn judge score (0-100), one line per arm.
  avg_<dv>_<scenario>:        per-arm AVERAGE (over turns) grouped bars —
      one facet per layer config, bars ordered + colored by AXIS FAMILY
      (anchor / answer / ctx_native / ctx_faithful / ctx_preimage), dashed
      unsteered + cap_alltoken reference lines with the anchor-seed min-max
      band shaded (anchors run at seeds 42/43/44).
  avg_drift_l32_<scenario> / avg_drift_band_<scenario>: same bar shape for
      the mean answer-token projection — layer-32 PRIMARY (skipped + logged
      when layer 32 is absent) and band-mean secondary.

NAP-round support: ``--round-subdir`` reads the round's tree
(``<out-root>/<slug>/<subdir>/...``) and writes figures under
``<fig-dir>/<slug>/<subdir>/``; cells are keyed by FILENAME STEM (the runner's
seed-suffixed ``cell_name``), so seeded anchor cells never collide.

Figures are deliberately bare: axes + legend only, no annotations or
interpretive overlays. One color = one arm on the line figures; the avg-bar
figures color by axis FAMILY (stated in their legends). Offline — reads
committed JSONs only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2223_casestudy_figures.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO = _ensure_repo_root_on_syspath()

from scripts.issue2223_casestudy_dashboard import (  # noqa: E402
    discover_model_slugs,
)
from scripts.issue2223_casestudy_replay import (  # noqa: E402
    ARM_ORDER,
    CS_ARMS,
    LAYER_CONFIGS,
    MODELS,
    SCENARIOS,
    cell_name,
)

READOUTS = ("answer_mean", "context", "prefix")
READOUT_TITLES = {
    "answer_mean": "answer tokens (mean)",
    "context": "context vector (last prompt token)",
    "prefix": "prefix vector (last system-prompt token)",
}

ANCHOR_ARMS = ("unsteered", "cap_alltoken")
ANCHOR_SEEDS = (42, 43, 44)
# PRIMARY drift readout layer (plan §4: the paper's steering-selected mid layer;
# cells store projections at ALL model layers, so layer 32 is present for 32b).
PRIMARY_DRIFT_LAYER = 32
FAMILY_ORDER = ("anchor", "answer", "ctx_native", "prefix_native", "ctx_faithful", "ctx_preimage")


def family_of(arm: str) -> str:
    """Axis family of an arm (anchors -> 'anchor'; paper cap_alltoken included)."""
    spec = CS_ARMS[arm]
    return spec.get("axis", "anchor") if spec.get("engine") == "caphook" else "anchor"


def load_cells_by_stem(model_root: Path, scenario: str) -> dict[str, dict]:
    """Cell JSONs keyed by FILENAME STEM (seed-suffixed ``cell_name``).

    The dashboard's ``load_cells`` keys on (layers, arm) alone, so seeded
    anchor cells (``na__unsteered__seed43``) would silently OVERWRITE the
    seed-42 entry — this loader keeps every seed distinct.
    """
    cells: dict[str, dict] = {}
    sc_dir = model_root / scenario
    if not sc_dir.is_dir():
        return cells
    for p in sorted(sc_dir.glob("*.json")):
        cells[p.stem] = json.loads(p.read_text())
    return cells


def _arm_colors() -> dict[str, str]:
    from explore_persona_space.analysis.paper_plots import paper_palette

    cols = paper_palette(len(ARM_ORDER))
    return dict(zip(ARM_ORDER, cols, strict=True))


def _band_mean_projection(cell: dict, rec: dict, readout: str) -> float | None:
    band = [str(li) for li in cell["band_layers"]]
    vals = [rec["projections"][readout].get(li) for li in band]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _layer_projection(cell: dict, rec: dict, readout: str, layer: int) -> float | None:
    """Single-layer assistant-axis projection (None when the layer is absent)."""
    return rec["projections"].get(readout, {}).get(str(layer))


def _cells_have_layer(cells: dict[str, dict], layer: int) -> bool:
    """True when any cell stores a projection at ``layer`` (tiny models lack 32)."""
    key = str(layer)
    return any(
        key in rec["projections"].get(r, {})
        for cell in cells.values()
        for rec in cell["turns"]
        for r in READOUTS
    )


def _arm_cell(cells: dict[str, dict], arm: str, layers_cfg: str) -> dict | None:
    key = f"na__{arm}" if arm == "unsteered" else f"{layers_cfg}__{arm}"
    return cells.get(key)


def fig_drift(
    scenario: str,
    layers_cfg: str,
    cells: dict[str, dict],
    colors,
    out_dir: Path,
    arms: "list[str]" = ARM_ORDER,
):
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper

    have_l32 = _cells_have_layer(cells, PRIMARY_DRIFT_LAYER)
    if not have_l32:
        print(
            f"[figures] {scenario}/{layers_cfg}: no cell stores layer "
            f"{PRIMARY_DRIFT_LAYER} — rendering band-mean row only",
            flush=True,
        )
    rows: list[tuple[str, str]] = []
    if have_l32:
        rows.append(("l32", f"layer {PRIMARY_DRIFT_LAYER} (primary)"))
    rows.append(("band", "band-layer mean" + (" (secondary)" if have_l32 else "")))
    fig, axes = plt.subplots(
        len(rows), 3, figsize=(12.5, 3.6 * len(rows)), sharex=True, squeeze=False
    )
    any_line = False
    for arm in arms:
        cell = _arm_cell(cells, arm, layers_cfg)
        if cell is None:
            continue
        turns = [rec["turn"] for rec in cell["turns"]]
        for ri, (row_kind, _row_label) in enumerate(rows):
            for ax, readout in zip(axes[ri], READOUTS, strict=True):
                if row_kind == "l32":
                    ys = [
                        _layer_projection(cell, rec, readout, PRIMARY_DRIFT_LAYER)
                        for rec in cell["turns"]
                    ]
                else:
                    ys = [_band_mean_projection(cell, rec, readout) for rec in cell["turns"]]
                pts = [(t, y) for t, y in zip(turns, ys, strict=True) if y is not None]
                if not pts:
                    continue
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="o",
                    markersize=2.5,
                    linewidth=1.2,
                    label=arm,
                    color=colors[arm],
                )
                any_line = True
    assert any_line, f"no plottable cells for {scenario}/{layers_cfg}"
    for ri, (_row_kind, row_label) in enumerate(rows):
        for ax, readout in zip(axes[ri], READOUTS, strict=True):
            if ri == 0:
                ax.set_title(READOUT_TITLES[readout])
            if ri == len(rows) - 1:
                ax.set_xlabel("turn")
        axes[ri][0].set_ylabel(f"axis projection ({row_label})")
    axes[0][-1].legend(fontsize=6, loc="best", ncol=1)
    fig.suptitle(f"{scenario} — {layers_cfg} layers", fontsize=10)
    fig.tight_layout()
    paths = savefig_paper(fig, f"drift_{scenario}_{layers_cfg}", dir=out_dir, formats=("png",))
    plt.close(fig)
    print(f"[figures] wrote {paths['png']}", flush=True)


_DV_YLABEL = {"harm": "harm judge score (0-100)", "coherence": "coherence score (0-100)"}


def fig_score(
    scenario: str,
    layers_cfg: str,
    cells: dict[str, dict],
    scores: dict,
    colors,
    out_dir: Path,
    *,
    dv: str = "harm",
    arms: "list[str]" = ARM_ORDER,
):
    """Per-turn judge-score line figure for one DV (``harm`` or ``coherence``).

    ``scores`` is the ``["cells"]`` block of ``<dv>_<scenario>.json`` (harm =
    ``scores_<sc>.json``; coherence = ``coherence_<sc>.json``); both share the
    ``{layers__arm: {turn: {"score": ...}}}`` shape.
    """
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    any_line = False
    for arm in arms:
        cell = _arm_cell(cells, arm, layers_cfg)
        if cell is None:
            continue
        key = f"na__{arm}" if arm == "unsteered" else f"{layers_cfg}__{arm}"
        per_turn = scores.get(key, {})
        pts = sorted(
            (int(t), rec["score"]) for t, rec in per_turn.items() if rec["score"] is not None
        )
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=arm,
            color=colors[arm],
        )
        any_line = True
    assert any_line, f"no judged cells for {scenario}/{layers_cfg} ({dv})"
    ax.set_xlabel("turn")
    ax.set_ylabel(_DV_YLABEL[dv])
    ax.set_ylim(-2, 102)
    ax.set_title(f"{scenario} — {layers_cfg} layers ({dv})")
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    paths = savefig_paper(fig, f"{dv}_{scenario}_{layers_cfg}", dir=out_dir, formats=("png",))
    plt.close(fig)
    print(f"[figures] wrote {paths['png']}", flush=True)


def _cell_mean_drift(cell: dict) -> float | None:
    """Mean over turns of the answer-token band-layer projection (secondary drift DV)."""
    vals = [_band_mean_projection(cell, rec, "answer_mean") for rec in cell["turns"]]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _cell_mean_drift32(cell: dict) -> float | None:
    """Mean over turns of the answer-token LAYER-32 projection (primary drift DV)."""
    vals = [
        _layer_projection(cell, rec, "answer_mean", PRIMARY_DRIFT_LAYER) for rec in cell["turns"]
    ]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _score_mean(scores_cells: dict, key: str) -> float | None:
    per_turn = scores_cells.get(key, {})
    vals = [rec["score"] for rec in per_turn.values() if rec["score"] is not None]
    return (sum(vals) / len(vals)) if vals else None


def _anchor_values(get, arm: str, lc: str) -> list[float]:
    """Anchor arm values across ANCHOR_SEEDS (unsteered lives at layers 'na')."""
    lcs = ("na", lc) if arm == "unsteered" else (lc, "na")
    out = []
    for seed in ANCHOR_SEEDS:
        for alc in lcs:
            v = get(cell_name(alc, arm, seed))
            if v is not None:
                out.append(v)
                break
    return out


def fig_avg_bars(
    scenario: str,
    out_dir: Path,
    name: str,
    ylabel: str,
    get,
    *,
    ylim=None,
    arms: "list[str]" = ARM_ORDER,
    layer_cfgs: "tuple[str, ...]" = LAYER_CONFIGS,
):
    """Per-arm average grouped bars: one facet per layer config, colored by family.

    ``get(cell_key) -> float | None`` supplies the per-cell average (judge-score
    mean or drift mean). Dashed unsteered / cap_alltoken reference lines; the
    anchor-seed min-max band (seeds 42/43/44) is shaded per anchor.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from explore_persona_space.analysis.paper_plots import paper_palette, savefig_paper

    fam_colors = dict(zip(FAMILY_ORDER, paper_palette(len(FAMILY_ORDER)), strict=True))
    fig, axes = plt.subplots(
        1, len(layer_cfgs), figsize=(max(7.0, 0.30 * len(arms) + 2.0) * 2, 4.2), sharey=True
    )
    if len(layer_cfgs) == 1:
        axes = [axes]
    any_bar = False
    for ax, lc in zip(axes, layer_cfgs, strict=True):
        ordered = [(arm, fam) for fam in FAMILY_ORDER for arm in arms if family_of(arm) == fam]
        xs, hs, cs, labels = [], [], [], []
        for arm, fam in ordered:
            key = cell_name("na", arm, 42) if arm == "unsteered" else cell_name(lc, arm, 42)
            v = get(key)
            if v is None:
                continue
            xs.append(len(xs))
            hs.append(v)
            cs.append(fam_colors[fam])
            labels.append(arm)
        if xs:
            ax.bar(xs, hs, color=cs, width=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
            any_bar = True
        for anchor, style in (("unsteered", ":"), ("cap_alltoken", "--")):
            vals = _anchor_values(get, anchor, lc)
            if vals:
                ax.axhline(vals[0], linestyle=style, linewidth=1.0, color="0.35")
                if len(vals) >= 2:
                    ax.axhspan(min(vals), max(vals), color="0.55", alpha=0.15, linewidth=0)
        ax.set_title(f"{lc} layers")
        if ylim is not None:
            ax.set_ylim(*ylim)
    assert any_bar, f"no plottable arm averages for {scenario} ({name})"
    axes[0].set_ylabel(ylabel)
    handles = [Patch(facecolor=fam_colors[f], label=f) for f in FAMILY_ORDER]
    import matplotlib.lines as mlines

    handles.append(mlines.Line2D([], [], linestyle=":", color="0.35", label="unsteered (s42)"))
    handles.append(mlines.Line2D([], [], linestyle="--", color="0.35", label="cap_alltoken (s42)"))
    axes[-1].legend(handles=handles, fontsize=6, loc="best")
    fig.suptitle(f"{scenario} — per-arm average", fontsize=10)
    fig.tight_layout()
    paths = savefig_paper(fig, f"{name}_{scenario}", dir=out_dir, formats=("png",))
    plt.close(fig)
    print(f"[figures] wrote {paths['png']}", flush=True)


def _discover_slugs(out_root: Path, subdir: str | None) -> list[str]:
    """Model slugs with >=1 scenario dir, honoring the round subdir when set."""
    if not subdir:
        return discover_model_slugs(out_root)
    if not out_root.is_dir():
        return []
    present = {
        d.name
        for d in out_root.iterdir()
        if d.is_dir() and any((d / subdir / sc).is_dir() for sc in SCENARIOS)
    }
    known = [m["slug"] for m in MODELS.values() if m["slug"] in present]
    return known + sorted(present - set(known))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-root",
        default=str(REPO / "eval_results" / "issue_2223" / "casestudy_replay"),
    )
    ap.add_argument(
        "--fig-dir",
        default=str(REPO / "figures" / "issue_2223" / "casestudy_replay"),
    )
    ap.add_argument(
        "--round-subdir",
        default=None,
        help="round subdir between <slug> and <scenario> (NAP round: "
        "native_axis_fidelity_preimage); figures land under <fig-dir>/<slug>/<subdir>/",
    )
    ap.add_argument(
        "--which", choices=("drift", "harm", "coherence", "avg", "both", "all"), default="both"
    )
    ap.add_argument(
        "--layers",
        choices=[*LAYER_CONFIGS, "both"],
        default="both",
        help="layer-config facets to render (band-only smoke slices: --layers band)",
    )
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)
    fig_root = Path(args.fig_dir)

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    colors = _arm_colors()
    slugs = _discover_slugs(out_root, args.round_subdir)
    if not slugs:
        print(f"[figures] no model cells under {out_root} — nothing to render", flush=True)
        return 0
    for slug in slugs:
        model_root = out_root / slug / args.round_subdir if args.round_subdir else out_root / slug
        fig_dir = fig_root / slug / args.round_subdir if args.round_subdir else fig_root / slug
        fig_dir.mkdir(parents=True, exist_ok=True)
        for sc in SCENARIOS:
            cells = load_cells_by_stem(model_root, sc)
            if not cells:
                print(f"[figures] {slug}: no cells for {sc} — skipping scenario", flush=True)
                continue
            want_harm = args.which in ("harm", "both", "all")
            want_coh = args.which in ("coherence", "all")
            want_drift = args.which in ("drift", "both", "all")
            want_avg = args.which in ("avg", "all")
            score_blocks: dict[str, dict] = {}
            for dv, fname in (("harm", f"scores_{sc}.json"), ("coherence", f"coherence_{sc}.json")):
                if (dv == "harm" and (want_harm or want_avg)) or (
                    dv == "coherence" and (want_coh or want_avg)
                ):
                    sp = model_root / "judged" / fname
                    assert sp.exists(), (
                        f"{sp} absent — run the judge phase first "
                        "(or pass --which drift for projection figures only)"
                    )
                    score_blocks[dv] = json.loads(sp.read_text())["cells"]
            lcs = LAYER_CONFIGS if args.layers == "both" else (args.layers,)
            for lc in lcs:
                if want_drift:
                    fig_drift(sc, lc, cells, colors, fig_dir)
                if want_harm:
                    fig_score(sc, lc, cells, score_blocks["harm"], colors, fig_dir, dv="harm")
                if want_coh:
                    fig_score(
                        sc, lc, cells, score_blocks["coherence"], colors, fig_dir, dv="coherence"
                    )
            if want_avg:
                fig_avg_bars(
                    sc,
                    fig_dir,
                    "avg_harm",
                    _DV_YLABEL["harm"] + " (mean over turns)",
                    lambda key, sb=score_blocks["harm"]: _score_mean(sb, key),
                    ylim=(-2, 102),
                    layer_cfgs=lcs,
                )
                fig_avg_bars(
                    sc,
                    fig_dir,
                    "avg_coherence",
                    _DV_YLABEL["coherence"] + " (mean over turns)",
                    lambda key, sb=score_blocks["coherence"]: _score_mean(sb, key),
                    ylim=(-2, 102),
                    layer_cfgs=lcs,
                )
                if _cells_have_layer(cells, PRIMARY_DRIFT_LAYER):
                    fig_avg_bars(
                        sc,
                        fig_dir,
                        "avg_drift_l32",
                        f"axis projection (layer {PRIMARY_DRIFT_LAYER} primary, mean over turns)",
                        lambda key, cc=cells: _cell_mean_drift32(cc[key]) if key in cc else None,
                        layer_cfgs=lcs,
                    )
                else:
                    print(
                        f"[figures] {slug}/{sc}: no cell stores layer "
                        f"{PRIMARY_DRIFT_LAYER} — skipping avg_drift_l32",
                        flush=True,
                    )
                fig_avg_bars(
                    sc,
                    fig_dir,
                    "avg_drift_band",
                    "axis projection (band mean secondary, mean over turns)",
                    lambda key, cc=cells: _cell_mean_drift(cc[key]) if key in cc else None,
                    layer_cfgs=lcs,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
