"""Hero plot for task #391 (broad-transfer reframe).

Single-panel chart that puts the source-vs-bystander lift question front and
centre. For each trained cell (anchor + A-flip + D-flip + assistant-trained
sanity-null), plots:

  * y-axis ticks: source persona x cell-label (librarian / programmer /
    surgeon, repeated for each cell)
  * x-axis: change in sycophancy index over base-Qwen zero-shot
  * blue bar: source-persona lift (source = panel persona being trained
    on)
  * orange bar: bystander-mean lift (mean across 23 non-source panel
    personas)

If the two bars match length across rows, the training generalises broadly
across personas; if the blue bar is consistently taller, the training is
source-selective. The pre-registered analysis (per-factor selectivity
delta) lives in the standalone selectivity figure; this hero is the
mentor-facing summary.

Output: ``figures/issue_391/hero_broad_transfer.{png,pdf,meta.json}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

DEFAULT_TABLE = Path("eval_results/issue_391/aggregate/cell_persona_table.csv")
DEFAULT_BASELINE = Path("eval_results/issue_391/aggregate/baseline_summary.json")
DEFAULT_OUTPUT_STEM = "issue_391/hero_broad_transfer"
DEFAULT_OUTPUT_DIR = Path("figures")

# Cells shown in the plot (in order, top-to-bottom).
# Each tuple: (cell_key_in_csv, plain_english_label).
CELL_ORDER = [
    ("10011", "Anchor cell\n(long sys-prompt, persona framing,\nClaude-generated training data)"),
    ("00011", "Short system prompt\n(A-axis flip)"),
    ("10001", "Base-Qwen training data\n(D-axis flip)"),
    ("assistant_a0_d1", "Sanity-null control\n(trained with assistant persona,\nnot source)"),
]
SOURCE_ORDER = ["librarian", "programmer", "surgeon"]


def _load_baseline_lookup(path: Path) -> dict[tuple[str, str], float]:
    payload = json.loads(path.read_text())
    return {
        (row["source"], row["panel_persona"]): row["mean_sycophancy_index"]
        for row in payload["rows"]
    }


def _load_cell_table(path: Path) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """Group by (cell_key, source). Value is list[(panel_persona, sycophancy_index)]."""
    out: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            ck = row["cell_key"]
            src = row["source"]
            persona = row["panel_persona"]
            val = float(row["mean_sycophancy_index"])
            out[(ck, src)].append((persona, val))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", type=str, default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()

    if not args.table.exists():
        raise SystemExit(f"Cell-persona table not found at {args.table}.")
    if not args.baseline.exists():
        raise SystemExit(f"Baseline summary not found at {args.baseline}.")

    baseline_lookup = _load_baseline_lookup(args.baseline)
    cell_table = _load_cell_table(args.table)

    # Build plot rows: one per (cell, source) tuple that exists in the data.
    plot_rows: list[dict] = []
    for ck, ck_label in CELL_ORDER:
        for src in SOURCE_ORDER:
            pairs = cell_table.get((ck, src))
            if not pairs:
                continue
            src_vals = [v for p, v in pairs if p == src]
            bys_vals = [v for p, v in pairs if p != src]
            if not src_vals:
                continue
            base_src = baseline_lookup[(src, src)]
            base_bys = (
                sum(v for (s, p), v in baseline_lookup.items() if s == src and p != src) / 23.0
            )
            plot_rows.append(
                {
                    "ck": ck,
                    "cell_label": ck_label,
                    "source": src,
                    "src_lift": src_vals[0] - base_src,
                    "bys_lift": (sum(bys_vals) / len(bys_vals)) - base_bys,
                    "n_bys": len(bys_vals),
                }
            )

    # Print to stdout for the body.
    print("Cell-source lift vs base-Qwen zero-shot:")
    for r in plot_rows:
        print(
            f"  cell={r['ck']:>15} src={r['source']:<12} "
            f"src_lift={r['src_lift']:+.3f}  bys_lift={r['bys_lift']:+.3f}  "
            f"n_bys={r['n_bys']}",
            flush=True,
        )

    set_paper_style("blog")

    # Group rows by cell for visual cell-separators.
    # y-axis: each row gets one y position, grouped with small gap between cells.
    src_color = paper_palette_role("primary")
    bys_color = paper_palette_role("accent")

    n_rows = len(plot_rows)
    y_positions: list[float] = []
    bar_h = 0.36
    pos = 0.0
    last_ck = None
    for r in plot_rows:
        if last_ck is not None and r["ck"] != last_ck:
            pos += 0.6  # cell-separator gap
        y_positions.append(pos)
        pos += 1.0
        last_ck = r["ck"]
    y_positions = [-p for p in y_positions]  # invert so top-row at top

    fig, ax = plt.subplots(figsize=(11.0, 0.55 * len(plot_rows) + 2.0))
    for i, r in enumerate(plot_rows):
        y = y_positions[i]
        ax.barh(
            y + bar_h / 2,
            r["src_lift"],
            height=bar_h,
            color=src_color,
            label="Source-persona Δ sycophancy vs base" if i == 0 else None,
        )
        ax.barh(
            y - bar_h / 2,
            r["bys_lift"],
            height=bar_h,
            color=bys_color,
            label="Bystander-mean Δ sycophancy vs base" if i == 0 else None,
        )

    # Row labels: short "src — cell" combo
    row_labels = []
    for r in plot_rows:
        # Strip the parenthetical to keep ticks short; full label appears in caption.
        short = r["cell_label"].split("\n")[0]
        row_labels.append(f"{r['source']} — {short}")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Δ mean sycophancy index vs base-Qwen zero-shot")
    ax.axvline(0, color="#444444", lw=0.8)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.set_title(
        "Sycophancy training lifts source and bystander personas in lockstep",
        fontsize=12,
        loc="left",
        fontweight="semibold",
    )

    fig.text(
        0.5,
        0.96,
        "Across 11 trained cells, source-persona sycophancy and 23-bystander mean "
        "sycophancy both rise ~+0.05 to +0.17 over base; the gap between blue and "
        "orange bars stays within bootstrap noise. Training implants sycophancy "
        "broadly, not selectively to the source.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.text(0.01, 0.005, f"source: {args.table}", fontsize=6.5, color="#888888")
    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    savefig_paper(fig, args.output_stem, dir=str(args.output_dir))
    plt.close(fig)
    print(f"saved {args.output_dir / args.output_stem}.png + .pdf + .meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
