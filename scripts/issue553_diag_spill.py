# ruff: noqa: RUF002, RUF003
# Intentional Unicode (rho, ※, Δ, −, —) in scientific docstrings + labels.
"""Task #553 — Deliverable 4: diag-strength vs spill anti-correlation with inference.

16 source-level points on the #532 followup panel: at-home implant strength
(diagonal ``margin_trained`` at S==B) vs the source FE of the OFF-DIAGONAL
``Δmargin`` (and the ``Δz(※)`` variant) from the ordinary-cross two-way FE
fit. Inference per the #539 ``source_marginal`` recipe: Spearman + 2,000-rep
source-pair bootstrap + 10,000-rep MC label permutation, reported at n=16,
n=15 (drop C1, keep B1 — quasi-duplicate prompts, separate adapters), n=14
(drop both), and the rich-only slice n=13 (drop B1, C1, D2 — plan concern
13.7), plus per-source leave-one-out influence (point-drop; the FE vector is
estimated once on the full ordinary-cross panel — the gauge is the centered
lstsq coefficient block, well-defined on this connected panel).

The expected honest outcome at n=16 is demotion to "suggestive,
leverage-sensitive" if the inline ~−0.5 read was driven by the thin-prompt
cluster (B1 bare question, C1 standard template, D2 casual rewrite) — that
demotion is a named acceptable outcome (plan kill row D4).

Smoke = this exact script with reduced ``--n-marginal-boot/--n-perm``.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import issue539_residual_per_cohort as i539
import issue553_panel as p553
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

THIN_SOURCES = {
    "B1": "bare question",
    "C1": "standard template",
    "D2": "casual rewrite",
}
SLICES = {
    "n16_full": (),
    "n15_dropC1": ("C1",),
    "n14_dropB1C1": ("B1", "C1"),
    "n13_rich_only": ("B1", "C1", "D2"),
}


def _source_points(
    panel: dict, masks: dict, spill_channel: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(sources, diagonal margin, off-diagonal source FE of the spill channel)."""
    m = masks["ordinary_cross"]
    src = panel["source_cid"][m]
    byst = panel["bystander_label"][m]
    src_u, sc = np.unique(src, return_inverse=True)
    _, bc = np.unique(byst, return_inverse=True)
    fe = p553.fe_vector(panel[spill_channel][m], sc, bc, len(src_u), int(bc.max()) + 1)
    diag = np.array(
        [
            float(panel["margin_trained"][(panel["source_cid"] == s) & panel["is_self"]][0])
            for s in src_u
        ]
    )
    return list(src_u), diag, fe


def _slice_inference(
    sources: list[str], diag: np.ndarray, fe: np.ndarray, drop: tuple[str, ...], args
) -> dict:
    keep = np.array([s not in drop for s in sources])
    d, f = diag[keep], fe[keep]
    return {
        "n_sources": int(keep.sum()),
        "dropped": list(drop),
        "rho": i539._spearman_rho(d, f),
        "ci95_boot_sources": i539._bootstrap_spearman_ci(d, f, args.n_marginal_boot, args.seed),
        "p_perm": {
            **i539._permutation_p(d, f, args.n_perm, args.seed),
            "method": f"MC permutation of the {int(keep.sum())} source labels",
        },
    }


def make_figure(sources: list[str], diag: np.ndarray, fe: np.ndarray, fig_dir: Path) -> None:
    """Annotated 16-point scatter naming the thin-prompt sources."""
    set_paper_style("blog")
    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for s, d, f in zip(sources, diag, fe, strict=True):
        thin = s in THIN_SOURCES
        ax.plot(d, f, "o", ms=6, color=colors[1] if thin else colors[0])
        label = f"{s} ({THIN_SOURCES[s]})" if thin else s
        ax.annotate(label, (d, f), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Diagonal trained EOS margin (at-home implant strength)")
    ax.set_ylabel("Source FE of off-diagonal Δmargin (spill)")
    ax.set_title("Diag-strength vs spill, 16 sources (thin-prompt sources highlighted)", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "diag_vs_spill_scatter", dir=fig_dir)
    plt.close(fig)
    print(f"[figures] wrote diag_vs_spill_scatter to {fig_dir}")


def main(argv: list[str] | None = None) -> int:
    args = p553.common_parser(
        "Task #553 D4: diag-strength vs spill anti-correlation with inference."
    ).parse_args(argv)
    t0 = datetime.now(UTC)

    panel = p553.build_margin_panel(args.i532_dir)
    step0 = p553.step0_i532(panel, args.i532_dir)
    masks = p553.cohort_masks_553(panel)

    results: dict = {
        "metadata": p553.result_metadata(args, "issue553_diag_spill.py"),
        "step0_i532": step0,
        "channels": {},
    }
    for spill_channel in ("dmargin", "dz_marker"):
        sources, diag, fe = _source_points(panel, masks, spill_channel)
        blk: dict = {
            "per_source": {
                s: {"diag_margin_trained": float(d), "source_fe_offdiag": float(f)}
                for s, d, f in zip(sources, diag, fe, strict=True)
            },
            "slices": {
                name: _slice_inference(sources, diag, fe, drop, args)
                for name, drop in SLICES.items()
            },
        }
        # Leave-one-out influence (plan concern 13.7): point-drop per source.
        loo = {}
        for i, s in enumerate(sources):
            keep = np.ones(len(sources), dtype=bool)
            keep[i] = False
            loo[s] = i539._spearman_rho(diag[keep], fe[keep])
        blk["leave_one_out_rho"] = loo
        results["channels"][spill_channel] = blk
        if spill_channel == "dmargin":
            make_figure(sources, diag, fe, args.fig_dir)

    results["inline_vs_reviewed"] = [
        p553.ivr_entry(
            "diag-vs-spill Spearman (Δmargin spill)",
            -0.5,
            results["channels"]["dmargin"]["slices"]["n16_full"]["rho"],
            True,
            "reviewed adds source-pair bootstrap + MC permutation + n=15/14/13 slices + LOO "
            "influence; demotion at n=15 is the named acceptable outcome (plan kill row D4)",
        )
    ]
    p553.write_json(args.out_dir / "diag_spill.json", results)

    for ch in ("dmargin", "dz_marker"):
        for name in SLICES:
            blk = results["channels"][ch]["slices"][name]
            print(
                f"[headline] {ch} {name}: rho={blk['rho']:+.3f} "
                f"CI [{blk['ci95_boot_sources']['low']:+.3f}, "
                f"{blk['ci95_boot_sources']['high']:+.3f}] p={blk['p_perm']['p']:.4g}"
            )
    print(f"[done] wall={(datetime.now(UTC) - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
