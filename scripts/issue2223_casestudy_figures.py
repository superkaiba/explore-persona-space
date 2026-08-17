"""Issue #2223 — case-study replay figures (drift-over-turns + harm-over-turns).

Reads the per-cell replay JSONs (+ optional judged scores) written by
``issue2223_casestudy_replay.py`` (``<out-root>/<model_slug>/<scenario>/…``)
and emits, PER MODEL, per scenario x layer config, into
``<fig-dir>/<model_slug>/``:

  drift_<scenario>_<layers>:  3 panels (answer-token mean / context vector /
      prefix vector), x = turn, y = Lu assistant-axis projection averaged over
      the PAPER BAND layers (the same metric in both layer configs so lines
      are comparable), one line per arm.
  harm_<scenario>_<layers>:   per-turn judge score (0-100), one line per arm.

Figures are deliberately bare: axes + legend only, no annotations or
interpretive overlays. One color = one arm across every figure. Offline —
reads committed JSONs only.
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
    load_cells,
)
from scripts.issue2223_casestudy_replay import (  # noqa: E402
    ARM_ORDER,
    LAYER_CONFIGS,
    SCENARIOS,
)

READOUTS = ("answer_mean", "context", "prefix")
READOUT_TITLES = {
    "answer_mean": "answer tokens (mean)",
    "context": "context vector (last prompt token)",
    "prefix": "prefix vector (last system-prompt token)",
}


def _arm_colors() -> dict[str, str]:
    from explore_persona_space.analysis.paper_plots import paper_palette

    cols = paper_palette(len(ARM_ORDER))
    return dict(zip(ARM_ORDER, cols, strict=True))


def _band_mean_projection(cell: dict, rec: dict, readout: str) -> float | None:
    band = [str(li) for li in cell["band_layers"]]
    vals = [rec["projections"][readout].get(li) for li in band]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


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

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharex=True)
    any_line = False
    for arm in arms:
        cell = _arm_cell(cells, arm, layers_cfg)
        if cell is None:
            continue
        turns = [rec["turn"] for rec in cell["turns"]]
        for ax, readout in zip(axes, READOUTS, strict=True):
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
    for ax, readout in zip(axes, READOUTS, strict=True):
        ax.set_title(READOUT_TITLES[readout])
        ax.set_xlabel("turn")
    axes[0].set_ylabel("assistant-axis projection (band-layer mean)")
    axes[-1].legend(fontsize=6, loc="best", ncol=1)
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
        "--which", choices=("drift", "harm", "coherence", "both", "all"), default="both"
    )
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)
    fig_root = Path(args.fig_dir)

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    colors = _arm_colors()
    slugs = discover_model_slugs(out_root)
    if not slugs:
        print(f"[figures] no model cells under {out_root} — nothing to render", flush=True)
        return 0
    for slug in slugs:
        model_root = out_root / slug
        fig_dir = fig_root / slug
        fig_dir.mkdir(parents=True, exist_ok=True)
        for sc in SCENARIOS:
            cells = load_cells(model_root, sc)
            if not cells:
                print(f"[figures] {slug}: no cells for {sc} — skipping scenario", flush=True)
                continue
            want_harm = args.which in ("harm", "both", "all")
            want_coh = args.which in ("coherence", "all")
            want_drift = args.which in ("drift", "both", "all")
            score_blocks: dict[str, dict] = {}
            for dv, fname in (("harm", f"scores_{sc}.json"), ("coherence", f"coherence_{sc}.json")):
                if (dv == "harm" and want_harm) or (dv == "coherence" and want_coh):
                    sp = model_root / "judged" / fname
                    assert sp.exists(), (
                        f"{sp} absent — run the judge phase first "
                        "(or pass --which drift for projection figures only)"
                    )
                    score_blocks[dv] = json.loads(sp.read_text())["cells"]
            for lc in LAYER_CONFIGS:
                if want_drift:
                    fig_drift(sc, lc, cells, colors, fig_dir)
                if want_harm:
                    fig_score(sc, lc, cells, score_blocks["harm"], colors, fig_dir, dv="harm")
                if want_coh:
                    fig_score(
                        sc, lc, cells, score_blocks["coherence"], colors, fig_dir, dv="coherence"
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
