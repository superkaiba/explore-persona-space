#!/usr/bin/env python3
"""#1739 claim4-controls: re-render the fold figures with reader-facing labels.

Clean-result-critic revision round: figures must not expose rung/protocol
slugs. Re-uses the fold module's own renderer (`render_figures`) against the
COMMITTED `claim4_per_rung_table.json` — the numbers are byte-identical to
the 12cfdbf31d render; only labels change — and adds two low-level
companions: the two-series per-seed figure (`claim4_per_seed`, superseding
the true-map-only `claim4_spaghetti` draft) and the per-context scatter
behind the corrected-sycophancy correlations (`claim4_syco_percontext`).

--style iclr renders the Overleaf-paper HEADLINE variant instead
(`figures/paper/c5_claim4_margin_forest`): a margin-first forest — per rung,
the 5-seed mean ADVANTAGE OVER THE SHUFFLED-PAIRING CONTROL (dtrue − dshuf
per seed) with its seed t-CI, per-seed values as light points — because
roughly half the raw probe-on-mapped edge is a generic-transform effect the
shuffled control also produces, so the paper quotes the margin, never the
raw delta (claims.md rev 3, C5 ruling).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_fold_module():
    path = Path(__file__).resolve().parent / "issue1739_claim4_fold.py"
    if not (path.parent / "issue1739_r2v2_score.py").exists():
        # Running from a scratch copy: the fold module resolves the repo root
        # from its own location, so load the in-repo copy.
        path = Path(
            "/home/thomasjiralerspong/explore-persona-space/scripts/issue1739_claim4_fold.py"
        )
    spec = importlib.util.spec_from_file_location("issue1739_claim4_fold", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def render_paper_margin_forest(mod, table: dict) -> Path:
    """Margin-first forest at final ICLR size -> figures/paper/."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_full,
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("iclr")
    rows = [r for r in table["per_rung"] if r.get("complete")]
    order = sorted(rows, key=lambda r: (r["behavior"], r["eval_rung"]))
    flagships = {tuple(f) for f in table["meta"]["flagships"]}
    labels = []
    for r in order:
        lab = mod.rung_label(r["behavior"], r["eval_rung"])
        if (r["behavior"], r["eval_rung"]) in flagships:
            lab += " *"
        labels.append(lab)
    ys = list(range(len(order)))[::-1]
    blue = paper_color("instruct")

    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.66))
    for y, r in zip(ys, order, strict=True):
        m = r["margin"]
        fl = (r["behavior"], r["eval_rung"]) in flagships
        ax.plot(
            m["per_seed"],
            [y] * len(m["per_seed"]),
            "o",
            ms=2.2,
            color=blue,
            alpha=0.3,
            markeredgewidth=0,
            zorder=2,
        )
        lo, hi = m["tci"]
        # Non-negative offsets from the value, never raw bounds (gotchas).
        xerr = [[max(0.0, m["mean"] - lo)], [max(0.0, hi - m["mean"])]]
        ax.errorbar(
            [m["mean"]],
            [y],
            xerr=xerr,
            fmt="o",
            color=blue,
            ms=4.4 if fl else 3.2,
            capsize=1.8,
            elinewidth=0.8,
            zorder=3,
        )
    ax.axvline(0.0, color=paper_color("reference"), lw=0.6, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Advantage over the shuffled-pairing control ($\\Delta$ Spearman $\\rho$)")
    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent.parent / "figures/paper"
    if not (out_dir.parent / "eval_results").exists():
        # Running from a scratch copy: anchor on the repo root instead.
        out_dir = Path("/home/thomasjiralerspong/explore-persona-space/figures/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "c5_claim4_margin_forest", dir=out_dir)
    plt.close(fig)
    print(f"wrote {out_dir / 'c5_claim4_margin_forest'}.png/.pdf (iclr)")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--table",
        default="eval_results/issue_1739/claim4_controls/claim4_per_rung_table.json",
    )
    ap.add_argument("--fig-dir", default="figures/issue_1739/claim4_controls")
    ap.add_argument(
        "--preds-root",
        default=None,
        help="claim4 preds mirror root (default: the table's meta.claim4_root)",
    )
    ap.add_argument(
        "--only-percontext",
        action="store_true",
        help="re-render ONLY claim4_syco_percontext (leave the other pinned figures untouched)",
    )
    ap.add_argument(
        "--style",
        choices=("blog", "iclr"),
        default="blog",
        help=(
            "iclr: render ONLY the paper margin-first forest into figures/paper/ and exit; "
            "the pinned blog-register figures are untouched"
        ),
    )
    args = ap.parse_args()

    mod = _load_fold_module()
    table_path = Path(args.table)
    if not table_path.is_absolute() and not table_path.exists():
        table_path = Path("/home/thomasjiralerspong/explore-persona-space") / table_path
    table = json.loads(table_path.read_text())
    if args.style == "iclr":
        render_paper_margin_forest(mod, table)
        return 0
    fig_dir = Path(args.fig_dir)
    seeds = table["meta"]["seeds"]
    written = [] if args.only_percontext else mod.render_figures(table, fig_dir, seeds)
    preds_root = Path(args.preds_root or table["meta"]["claim4_root"])
    written.append(mod.render_syco_percontext(preds_root, fig_dir))
    print(f"written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
