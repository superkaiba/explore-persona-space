"""Issue #2223 strength-sweep dashboard — unsteered vs cap_alltoken vs the 18 new arms.

Scoped variant of ``issue2223_casestudy_dashboard.py`` for the user-approved
inline intervention-strength sweep (Qwen3-32B only, selfharm + delusion). It:

  1. renders the drift / harm / coherence per-turn figures for the strength-sweep
     arm subset (via ``issue2223_casestudy_figures`` with ``arms=STRSWEEP_ARMS``);
  2. base64-embeds those PNGs into a header block (self-contained HTML);
  3. injects a Fable-5-authored qualitative-analysis HTML fragment at the TOP
     (``--analysis-file``); and
  4. emits the side-by-side conversation tables (same format as the base #2223
     dashboard) for ONLY ``unsteered`` + ``cap_alltoken`` + the 18 new arms.

Offline / stdlib + matplotlib only; reads the committed replay cells + judged
scores. NO analysis is authored here — the only interpretive prose is the
Fable-5 fragment, clearly labelled as such.
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2223_strsweep_dashboard.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO = _ensure_repo_root_on_syspath()

from scripts.issue2223_casestudy_dashboard import build_dashboard, load_cells  # noqa: E402
from scripts.issue2223_casestudy_figures import (  # noqa: E402
    _arm_colors,
    fig_drift,
    fig_score,
)
from scripts.issue2223_casestudy_replay import (  # noqa: E402
    LAYER_CONFIGS,
    NEW_STRENGTH_ARMS,
)

MODEL_SLUG = "qwen3-32b"
SCENARIOS = ("selfharm", "delusion")  # the two scenarios this sweep ran
# 20 columns: the two reference arms + the 18 aggressive-strength arms.
STRSWEEP_ARMS = ["unsteered", "cap_alltoken", *NEW_STRENGTH_ARMS]

_SECTION_CSS = """
<style>
section.analysis { border: 1px solid #c9c9e0; background: #fbfbff; padding: 12px 16px;
                   margin: 12px 0 20px; border-radius: 6px; max-width: 1100px; }
section.analysis h2 { margin-top: 4px; }
section.analysis p { font-size: 13px; line-height: 1.5; }
section.figs { margin: 12px 0 24px; }
section.figs .figrow { margin-bottom: 18px; }
section.figs img { border: 1px solid #ddd; max-width: 100%; height: auto; }
span.byline { color: #666; font-size: 12px; font-style: italic; }
</style>
"""


def _render_figures(out_root: Path, fig_dir: Path) -> dict[tuple[str, str, str], Path]:
    """Render drift/harm/coherence for the strength-sweep subset; return {(sc,lc,dv):png}."""
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    colors = _arm_colors()
    model_root = out_root / MODEL_SLUG
    fig_dir.mkdir(parents=True, exist_ok=True)
    import json

    paths: dict[tuple[str, str, str], Path] = {}
    for sc in SCENARIOS:
        cells = load_cells(model_root, sc)
        assert cells, f"no cells for {MODEL_SLUG}/{sc} under {model_root}"
        harm = json.loads((model_root / "judged" / f"scores_{sc}.json").read_text())["cells"]
        coh = json.loads((model_root / "judged" / f"coherence_{sc}.json").read_text())["cells"]
        for lc in LAYER_CONFIGS:
            fig_drift(sc, lc, cells, colors, fig_dir, arms=STRSWEEP_ARMS)
            paths[(sc, lc, "drift")] = fig_dir / f"drift_{sc}_{lc}.png"
            fig_score(sc, lc, cells, harm, colors, fig_dir, dv="harm", arms=STRSWEEP_ARMS)
            paths[(sc, lc, "harm")] = fig_dir / f"harm_{sc}_{lc}.png"
            fig_score(sc, lc, cells, coh, colors, fig_dir, dv="coherence", arms=STRSWEEP_ARMS)
            paths[(sc, lc, "coherence")] = fig_dir / f"coherence_{sc}_{lc}.png"
    return paths


def _figs_section(fig_paths: dict[tuple[str, str, str], Path], rel_to: Path) -> str:
    parts = [
        "<section class='figs'><h2>Per-turn quantitative summary — "
        "drift, harm, coherence (Qwen3-32B)</h2>",
        "<p class='meta'>Drift = Lu assistant-axis projection, band-layer mean (3 readout "
        "panels). Harm / coherence = Sonnet-4.5 judge 0-100 (3 draws/item, mean). One line "
        "per arm; one colour = one arm across every figure.</p>",
    ]
    for sc in SCENARIOS:
        for lc in LAYER_CONFIGS:
            parts.append(f"<h3>{html.escape(sc)} — {lc} layers</h3><div class='figrow'>")
            for dv in ("drift", "harm", "coherence"):
                png = fig_paths.get((sc, lc, dv))
                if png and png.exists():
                    rel = os.path.relpath(png, rel_to)
                    parts.append(
                        f"<div><b>{dv}</b><br><img alt='{dv} {sc} {lc}' "
                        f"src='{html.escape(rel)}'></div>"
                    )
            parts.append("</div>")
    parts.append("</section>")
    return "".join(parts)


def _analysis_section(analysis_file: Path | None) -> str:
    byline = (
        "<span class='byline'>Authored by an independent Fable-5 subagent that read the "
        "replayed conversations (the model's own generated turns) across arms — not by the "
        "orchestrator. Reproduced verbatim.</span>"
    )
    if analysis_file and analysis_file.exists():
        frag = analysis_file.read_text().strip()
    else:
        frag = "<p class='meta'>[Fable-5 qualitative analysis pending]</p>"
    return (
        f"<section class='analysis'><h2>Qualitative analysis (Fable 5)</h2>{byline}{frag}</section>"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-root",
        default=str(REPO / "eval_results" / "issue_2223" / "casestudy_replay"),
    )
    ap.add_argument(
        "--fig-dir",
        default=str(REPO / "figures" / "issue_2223" / "casestudy_replay" / "strsweep"),
    )
    ap.add_argument(
        "--out",
        default=str(
            REPO / "figures" / "issue_2223" / "casestudy_replay" / "strsweep_dashboard.html"
        ),
    )
    ap.add_argument(
        "--analysis-file",
        default=str(
            REPO / "figures" / "issue_2223" / "casestudy_replay" / "strsweep_analysis.html"
        ),
        help="HTML fragment authored by the Fable-5 subagent (injected at the top).",
    )
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)
    fig_dir = Path(args.fig_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig_paths = _render_figures(out_root, fig_dir)
    header = (
        _SECTION_CSS
        + _analysis_section(Path(args.analysis_file))
        + _figs_section(fig_paths, out.parent)
    )
    html_text = build_dashboard(
        out_root,
        arms=STRSWEEP_ARMS,
        models=[MODEL_SLUG],
        scenarios=SCENARIOS,
        header_html=header,
        title="Issue #2223 — intervention-strength sweep (Qwen3-32B): unsteered vs cap_alltoken vs new arms",
    )
    out.write_text(html_text)
    print(f"[strsweep-dashboard] wrote {out} ({len(html_text)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
