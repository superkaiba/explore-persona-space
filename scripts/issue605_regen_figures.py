"""Regenerate issue-605 analysis figures with reader-facing labels.

Presentation-only round-2 pass (interp-critique label fixes): re-reads the
committed per-cell JSONs + the already-registered ``analysis.json`` files and
re-renders the six analysis figures. NO statistic is recomputed and NO
``analysis.json`` is rewritten — the registered numbers stay exactly as in
round 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue605_analysis import (
    _fact_figures,
    _marker_figures,
    build_fact_frame,
    build_marker_frame,
)


def main() -> None:
    """Rebuild the deterministic cell frames and re-render all six figures."""
    root = Path(__file__).resolve().parents[1]
    out_root = root / "eval_results" / "issue_605"
    fig_dir = root / "figures" / "issue_605"

    marker_res = json.loads((out_root / "marker" / "analysis.json").read_text())
    frame, _sel = build_marker_frame(out_root)
    for band in ("band_lo", "band_mid", "band_hi"):
        print(f"{band}: {int((frame['band'] == band).sum())} cells")
    _marker_figures(frame, marker_res, fig_dir)

    fact_res = json.loads((out_root / "fact" / "analysis.json").read_text())
    fact_frame, _fsel, _dropped = build_fact_frame(out_root)
    print(f"fact frame: {len(fact_frame)} cell-personas")
    _fact_figures(fact_frame, fact_res, fig_dir)
    print(f"figures re-rendered to {fig_dir}")


if __name__ == "__main__":
    main()
