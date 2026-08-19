"""Length-matched diagnostics figure for issue #2329 (manifest `diagnostics_dump`).

Renders `figures/issue_2329/length_matched_diag.{png,pdf}` + `.meta.json`:

  (a) length-matched vs unmatched steered F_beh per (type-cell x slot) unit,
      restricted to the units with >= 1 pair at |ctx-length delta| <= 2 tokens
      (`stats.json` -> per_cell[unit]["length_matched"]); units with
      length_matched.n == 0 are OMITTED (named in the panel title), never
      drawn as a zero bar;
  (b) the per-pair ctx-length-delta covariate distribution per unit
      (`f_cells.jsonl` -> len_delta), with the +/-2-token matching threshold
      drawn.

Pure render off committed eval JSONs — no model calls, no new data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2329.length_matched_fig")

# Same arm color convention as scripts/issue2329_figures.py (one color = one
# meaning: steered is blue in every issue-2329 panel).
STEERED_COLOR = "#4878d0"
# Black DASHED for the matching threshold: red (#c44e52) means the crosstype
# arm and grey (#9d9d9d) means the shuffled arm in every other issue-2329
# figure — one color = one meaning; dashed vs the solid black zero line.
THRESHOLD_COLOR = "black"


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _save(fig, out_dir: Path, name: str, inputs: list[Path], extra: dict | None = None) -> None:
    """Save via the house ``savefig_paper`` (PNG + PDF + ``.meta.json`` sidecar),
    then merge input paths + git provenance (+ ``extra`` facts) into the sidecar
    — the same idiom as scripts/issue2329_figures.py::_save."""
    from explore_persona_space.analysis.paper_plots import savefig_paper

    out_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, name, dir=out_dir)
    plt.close(fig)
    meta_path = out_dir / f"{name}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["figure"] = name
    meta["inputs"] = [str(p) for p in inputs]
    meta["provenance"] = as_metadata_dict(git_provenance())
    if extra:
        meta.update(extra)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.info("wrote %s.{png,pdf,meta.json} under %s", name, out_dir)


def render(stats_path: Path, f_cells_path: Path, out_dir: Path) -> None:
    stats = json.loads(stats_path.read_text())
    per_cell = stats["per_cell"]

    # --- panel (a) data: units with a non-empty length-matched subset ---
    matched_units = []
    omitted_units = []
    for key in sorted(per_cell):
        lm = per_cell[key].get("length_matched") or {}
        if lm.get("n", 0) > 0:
            matched_units.append(
                {
                    "unit": key,
                    "n_matched": int(lm["n"]),
                    "n_all": int(per_cell[key]["n_post_exclusion"]),
                    "f_all": per_cell[key]["f_steered_mean"],
                    "f_matched": lm["f_steered_mean"],
                    "max_abs_len_delta": int(lm["max_abs_len_delta"]),
                }
            )
        else:
            omitted_units.append(key)
    if not matched_units:
        raise RuntimeError(f"no units with length_matched.n > 0 in {stats_path}")
    thresholds = {u["max_abs_len_delta"] for u in matched_units}
    if thresholds != {2}:
        raise RuntimeError(f"expected a uniform |len_delta| <= 2 threshold, got {thresholds}")

    # --- panel (b) data: per-pair len_delta per unit (steered rows) ---
    len_deltas: dict[str, list[int]] = defaultdict(list)
    for r in _iter_jsonl(f_cells_path):
        if r.get("len_delta") is not None:
            len_deltas[f"{r['cell']}|{r['slot']}"].append(int(r["len_delta"]))
    b_units = sorted(len_deltas)
    if not b_units:
        raise RuntimeError(f"no len_delta values in {f_cells_path}")

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(19.0, 12.0), gridspec_kw={"height_ratios": [1.0, 0.9]}
    )

    # ---- panel (a): matched vs unmatched steered F_beh per unit ----
    xs = np.arange(len(matched_units))
    f_all = np.array([u["f_all"] for u in matched_units], dtype=float)
    f_matched = np.array([u["f_matched"] for u in matched_units], dtype=float)
    for x, ya, ym in zip(xs, f_all, f_matched):
        ax_a.plot([x, x], [ya, ym], color="#9d9d9d", lw=0.8, zorder=1)
    ax_a.scatter(
        xs, f_all, s=42, color=STEERED_COLOR, zorder=2, label="steered, all post-exclusion pairs"
    )
    ax_a.scatter(
        xs,
        f_matched,
        s=42,
        facecolors="none",
        edgecolors=STEERED_COLOR,
        linewidths=1.6,
        zorder=3,
        label="steered, length-matched pairs only (|Δ ctx len| ≤ 2 tokens)",
    )
    ax_a.axhline(0.0, color="black", lw=0.8, zorder=0)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(
        [f"{u['unit']}\n{u['n_matched']}/{u['n_all']}" for u in matched_units],
        rotation=90,
        fontsize=6,
    )
    ax_a.set_ylabel("steered F_beh (mean over pairs)")
    ax_a.set_title(
        "Length-matched recount vs unmatched steered F_beh per (type-cell × slot) — "
        f"{len(matched_units)} of {len(per_cell)} units with ≥1 pair at "
        f"|Δ ctx len| ≤ 2 tokens; {len(omitted_units)} units with no "
        "length-matched pairs omitted; tick = unit + n matched/all",
        loc="left",
    )
    ax_a.legend(loc="upper right", framealpha=0.9)

    # ---- panel (b): per-pair len_delta distribution per unit ----
    data = [len_deltas[u] for u in b_units]
    ax_b.boxplot(
        data,
        positions=np.arange(len(b_units)),
        widths=0.6,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.6},
        medianprops={"color": STEERED_COLOR},
    )
    for y in (-2, 2):
        ax_b.axhline(
            y,
            color=THRESHOLD_COLOR,
            ls="--",
            lw=0.9,
            zorder=0,
            label="length-match threshold (±2 tokens)" if y == 2 else None,
        )
    ax_b.axhline(0.0, color="black", lw=0.8, zorder=0)
    ax_b.set_xticks(np.arange(len(b_units)))
    ax_b.set_xticklabels(b_units, rotation=90, fontsize=6)
    ax_b.set_ylabel("per-pair Δ ctx length (tokens; value B − value A)")
    ax_b.set_title(
        "Per-pair context-length delta per unit (steered rows, "
        f"{sum(len(v) for v in data)} pairs over {len(b_units)} units)",
        loc="left",
    )
    ax_b.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    _save(
        fig,
        out_dir,
        "length_matched_diag",
        inputs=[stats_path, f_cells_path],
        extra={
            "panel_a_units": matched_units,
            "panel_a_omitted_units_n0": omitted_units,
            "len_match_max_abs_delta_tokens": 2,
        },
    )


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("blog")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--f-metrics-dir", type=Path, default=Path("eval_results/issue_2329/f_metrics")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("figures/issue_2329"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    render(
        stats_path=args.f_metrics_dir / "stats.json",
        f_cells_path=args.f_metrics_dir / "f_cells.jsonl",
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
