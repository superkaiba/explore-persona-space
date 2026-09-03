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
(`figures/paper/c5_claim4_margin_forest`): a raw-difference forest — per
rung, the 5-seed mean RAW DIFFERENCE dtrue (regression on mapped answer −
regression on context) with its seed t-CI, per-seed values as light points.
User order 2026-08-25: the figure shows the raw difference; the margin over
the shuffled-pairing control stays quoted in the paper prose (claims.md
rev 3, C5 ruling), not in this display. The sycophancy are-you-sure,
evil Tom Gibbs multi-turn, and sycophancy mimicry sets are excluded from the
paper figure (same orders; negative-raw sets pending diagnosis).
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
    """Raw-difference forest at the paper (c2a-v2) standard -> figures/paper/."""
    import hashlib

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.c2a_plot_style import (
        INK,
        ROLES,
        STYLE_VERSION,
        c2a_figure,
        panel_header,
        save_c2a_figure,
        set_c2a_style,
        style_axis,
    )
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    set_c2a_style()
    # Paper scope: the sycophancy are-you-sure, evil Tom Gibbs multi-turn, and
    # sycophancy mimicry sets are excluded from the Applications forest (user
    # orders 2026-08-25; the negative-raw sets are excluded pending diagnosis).
    # The pinned blog-register figures keep all 13 rungs.
    paper_excluded = {
        ("sycophancy", "sycoays"),
        ("evil", "evil_tomgibbs"),
        ("sycophancy", "sycomim"),
    }
    rows = [
        r
        for r in table["per_rung"]
        if r.get("complete") and (r["behavior"], r["eval_rung"]) not in paper_excluded
    ]
    order = sorted(rows, key=lambda r: (r["behavior"], r["eval_rung"]))
    # No flagship star / size emphasis (user order 2026-08-25: "remove the
    # stars") — every set renders identically.
    labels = [mod.rung_label(r["behavior"], r["eval_rung"]) for r in order]
    ys = list(range(len(order)))[::-1]
    teal = ROLES["linear"].color

    fig, frac = c2a_figure("wide", aspect=0.62)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.43, right=0.97, bottom=0.14, top=0.84)
    forest_rows: list[dict] = []
    for y, r in zip(ys, order, strict=True):
        m = r["dtrue"]
        ax.plot(
            m["per_seed"],
            [y] * len(m["per_seed"]),
            "o",
            ms=4.5,
            color=teal,
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
            color=teal,
            ecolor=INK,
            ms=7,
            capsize=3,
            elinewidth=1.4,
            zorder=3,
        )
        forest_rows.append(
            {
                "behavior": r["behavior"],
                "eval_rung": r["eval_rung"],
                "label": mod.rung_label(r["behavior"], r["eval_rung"]),
                "mean": m["mean"],
                "tci": list(m["tci"]),
                "per_seed": list(m["per_seed"]),
            }
        )
    ax.axvline(0.0, color=INK, lw=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    # Ten long set names need a step below the pinned tick size to fit the
    # margin (same disclosed deviation as c4_shared_speakers).
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("$\\Delta$ Spearman $\\rho$ (mapped answer $-$ context)")
    style_axis(ax, grid_axis="x")
    panel_header(
        ax,
        "",
        "Regression readouts · OOD evaluation sets",
        title="Mapped answer minus context, per set",
    )
    checkout = Path(__file__).resolve().parent.parent
    if not (checkout / "eval_results").exists():
        # Running from a scratch copy: anchor on the repo root instead.
        checkout = Path("/home/thomasjiralerspong/explore-persona-space")
    out_dir = checkout / "figures/paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "c5_claim4_margin_forest"
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Figure 17: mapped-answer minus context difference per OOD set",
        subject=(
            "Per OOD evaluation set, the 5-seed mean raw difference in Spearman rho "
            "(regression on mapped answer minus regression on context) with seed t-CI "
            "(#1739 claim4 controls)"
        ),
        creator="scripts/issue1739_claim4_relabel_figs.py",
        include_width=frac,
    )
    plt.close(fig)

    table_path = Path(table["__source_path__"]) if "__source_path__" in table else None
    sidecar = stem.with_suffix(".meta.json")
    payload = {
        "figure": "c5_claim4_margin_forest",
        "status": "manuscript Figure 17 (c2a-v2 restyle; values unchanged)",
        "style_version": STYLE_VERSION,
        "plotting_script": "scripts/issue1739_claim4_relabel_figs.py",
        "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
        "reproduction_command": (
            "uv run python scripts/issue1739_claim4_relabel_figs.py --style iclr"
        ),
        "git": as_metadata_dict(git_provenance()),
        "sources": {
            "claim4_per_rung_table": (
                {"path": str(table_path), "sha256": _sha256(table_path)}
                if table_path is not None and table_path.exists()
                else {"path": "eval_results/issue_1739/claim4_controls/claim4_per_rung_table.json"}
            )
        },
        "record": outputs["record"],
        "data": {"rungs": forest_rows},
        "output_sha256": {k: _sha256(p) for k, p in outputs.items() if k != "record"},
    }
    sidecar.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {stem}.png/.pdf/.meta.json (c2a-v2)")
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
    table["__source_path__"] = str(table_path.resolve())
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
