#!/usr/bin/env python3
"""Render completed boundary25k metrics with the context-to-answer paper style."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import issue1901_boundary_25k as B  # noqa: E402

from explore_persona_space.analysis import c2a_plot_style as S  # noqa: E402


def report(source, out_dir):
    result = json.loads(source.read_text())
    assert set(result["tokens"]) == {str(t) for t in B.TOKENS}
    assert result["manifest"]["n_manifest_rows"] == 102240
    S.set_c2a_style()
    fig, fraction = S.c2a_figure("full", aspect=0.43)
    axes = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.30, top=0.77, wspace=0.2)
    labels = [
        "Space + period (659)",
        "Period (13)",
        "Space + ? (937)",
        "Space + ! (753)",
    ]
    table = []
    for i, token in enumerate(B.TOKENS):
        row = result["tokens"][str(token)]
        assert (row["n_train"], row["n_val"], row["n_test"]) == (25000, 160, 400)
        ridge = row["metrics"]["ridge"]
        r2 = ridge["reconstruction"]["r2"]
        ci = ridge["reconstruction"]["article_bootstrap_r2"]
        retrieval = ridge["retrieval"]["whiten_csls"]
        point = retrieval["acc_at_k"]["1"]
        rci = retrieval["acc1_ci95"]
        for ax, value, interval in ((axes[0], r2, ci), (axes[1], point, rci)):
            # Draw the interval endpoints directly, including CIs excluding the point.
            ax.hlines(i, interval["lo"], interval["hi"], color=S.ROLES["linear"].color, lw=2)
            ax.plot(
                value,
                i,
                marker=S.ROLES["linear"].marker,
                color=S.ROLES["linear"].color,
                markerfacecolor=S.ROLES["linear"].color if ax is axes[0] else "white",
            )
        table.append(
            {
                "token_id": token,
                "label": labels[i],
                "r2": r2,
                "r2_ci95": ci,
                "top1_whiten_csls": point,
                "top1_ci95": rci,
                "n_pool": retrieval["n_pool"],
                "chance_top1": row["chance_top1"],
            }
        )
    for ax in axes:
        S.style_axis(ax, grid_axis="x")
        ax.set_xlim(0, 1)
        ax.set_ylim(3.6, -0.6)
        ax.set_yticks(range(4), labels)
    r2_min = min(0.0, *(r["r2_ci95"]["lo"] for r in table), *(r["r2"] for r in table))
    if r2_min < 0:
        axes[0].set_xlim(r2_min - 0.03, 1)
    axes[0].set_xlabel(S.better_label("Held-out $R^2$"))
    axes[1].set_xlabel(S.better_label("Top-1 retrieval"))
    S.panel_header(axes[0], "A", "25k training spans per token", "Reconstruction")
    S.panel_header(axes[1], "B", "Frozen WikiText evaluation", "Whitened cosine + CSLS")
    pools = {row["n_pool"] for row in table}
    assert len(pools) == 1, "Caption requires equal realized retrieval pools"
    pool = pools.pop()
    fig.text(
        0.03,
        0.055,
        f"Wikipedia train; WikiText test. Retrieval: {pool} candidates, chance {1 / pool:.2%}.\n"
        "95% intervals: article bootstrap for $R^2$; query bootstrap for retrieval.",
        ha="left",
        color=S.MUTED,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = S.save_c2a_figure(
        fig,
        out_dir / "boundary25k",
        title="Boundary-token fits at 25k spans",
        subject="Expanded Wikipedia training and frozen WikiText evaluation",
        creator="scripts/issue1901_boundary_25k_report.py",
        include_width=fraction,
    )
    B.write_json(
        out_dir / "boundary25k.meta.json",
        {
            "source_sha256": B.sha_file(source),
            "rows": table,
            "render": exported["record"],
            "caveat": "Training corpus and sample size both change relative to original 1200 fits.",
        },
    )
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(report(args.source, args.out_dir), indent=2))
