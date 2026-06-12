"""Issue #542 follow-up (positives-only-anchor) figures + three-space table.

CPU-only; consumes the follow-up eval root's ``G_arm/pos_only`` and
``G_arm/arm1_xfam`` tensors (the v1 cross-family control, re-assembled into
the follow-up root by ``--phase assemble --steps armone``) plus the COMMITTED
v1 analysis JSON for the six-arm envelope. Outputs:

- ``<fig-dir>/posonly_hero_dots.{png,pdf}`` -- off-diag mean + default column,
  pos_only vs control, six v1 arm values as envelope, 0.5-nat floor band;
- ``<fig-dir>/posonly_vs_control_heatmaps.{png,pdf}`` -- the G map pair;
- ``<eval-root>/analysis/posonly_three_space_542.json`` -- reads 1-2 recomputed
  in log-prob / marker-logit / EOS-margin space from the npz sidecars (the §5
  read-5 three-space agreement table; cross-space divergence = saturation
  signature, reported not hidden).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import i542_figures as f542  # noqa: E402  (sibling reuse: labels + heatmap style)
import i542_registered_reads as rr  # noqa: E402  (sibling reuse: ArmTensor + stats)

V1_ARMS = ("arm1_xfam", "arm2_close", "arm3_default", "c2", "c8", "c16")
STATS = (
    ("offdiag_mean", "Off-diagonal mean ΔlogP(※) (nat)"),
    ("default_col_broad_full_mask", "Default column, 10 broad rows (nat)"),
)


def _save(fig, fig_dir: Path, name: str, meta: dict) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    (fig_dir / f"{name}.meta.json").write_text(json.dumps({**rr._meta(), **meta}, indent=2))
    plt.close(fig)
    print(f"[posonly-figures] wrote {fig_dir / name}.png")


def _space_stats(t: rr.ArmTensor, M: np.ndarray) -> dict[str, float]:
    """Reads 1-2 over an arbitrary per-cell matrix (G or an npz sidecar space)."""
    assert M.shape == t.G.shape, (M.shape, t.G.shape)
    offdiag: list[float] = []
    for ii, ci in enumerate(t.train_cids):
        offdiag.extend(M[ii, rr._off_diag_indices(t, ci)].tolist())
    dj = t.col("default")
    broad = [float(M[t.row(c), dj]) for c in rr.BROAD_ROWS if c in t.train_cids]
    return {
        "offdiag_mean": float(np.mean(offdiag)),
        "n_offdiag_cells": len(offdiag),
        "default_col_broad_full_mask": float(np.mean(broad)),
        "n_broad_rows": len(broad),
    }


def hero_dots(t_pos: rr.ArmTensor, t_ctrl: rr.ArmTensor, v1_reads: dict, fig_dir: Path) -> None:
    """Hero: pos_only vs control dots with the v1 six-arm envelope + floor band."""
    from explore_persona_space.analysis.paper_plots import paper_palette_role

    pos, ctrl = _space_stats(t_pos, t_pos.G), _space_stats(t_ctrl, t_ctrl.G)
    fig, axes = plt.subplots(1, len(STATS), figsize=(4.4 * len(STATS), 3.7))
    env_used: dict[str, dict[str, float]] = {}
    for ax, (stat, label) in zip(np.atleast_1d(axes), STATS, strict=True):
        env = {
            a: v1_reads["per_arm"][a][stat]
            for a in V1_ARMS
            if a in v1_reads.get("per_arm", {}) and v1_reads["per_arm"][a].get(stat) is not None
        }
        env_used[stat] = env
        if env:
            ax.axhspan(
                min(env.values()),
                max(env.values()),
                alpha=0.12,
                color=paper_palette_role("neutral"),
                label=f"v1 envelope ({len(env)} arms)",
            )
        ax.axhspan(
            ctrl[stat] - rr.CLAIM_FLOOR_NATS,
            ctrl[stat] + rr.CLAIM_FLOOR_NATS,
            alpha=0.18,
            color=paper_palette_role("control"),
            label="±0.5-nat claim floor (around control)",
        )
        ax.scatter([0], [ctrl[stat]], s=70, color=paper_palette_role("control"), zorder=3)
        ax.scatter([1], [pos[stat]], s=70, color=paper_palette_role("accent"), zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            ["Cross-family panel\n(v1 control)", "Positives only\n(0 negatives)"], fontsize=8
        )
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(label)
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("Positives-only arm vs the v1 control: leakage summary reads", fontsize=10)
    _save(
        fig,
        fig_dir,
        "posonly_hero_dots",
        {
            "pos_only": pos,
            "control": ctrl,
            "v1_envelope": env_used,
            "claim_floor_nats": rr.CLAIM_FLOOR_NATS,
        },
    )


def heatmap_pair(t_pos: rr.ArmTensor, t_ctrl: rr.ArmTensor, fig_dir: Path) -> None:
    """Side-by-side G maps, the v1 heatmap-strip style (vmax=14 clips binst diag)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
    im = None
    for k, (ax, t, title) in enumerate(
        zip(
            axes,
            (t_ctrl, t_pos),
            ("Cross-family panel (v1 control)", "Positives only (0 negatives)"),
            strict=True,
        )
    ):
        im = ax.imshow(t.G, aspect="auto", cmap="viridis", vmin=-2, vmax=14)
        ax.set_title(title, fontsize=9)
        if k == 0:
            ax.set_yticks(range(len(t.train_cids)))
            ax.set_yticklabels([f542.CONTEXT_LABELS.get(c, c) for c in t.train_cids], fontsize=6.5)
        else:
            ax.set_yticks([])
        ax.set_xticks(range(len(t.eval_cids)))
        ax.set_xticklabels(
            [f542.CONTEXT_LABELS.get(c, c) for c in t.eval_cids], fontsize=5.6, rotation=90
        )
    fig.colorbar(im, ax=axes, shrink=0.7, label="ΔlogP(※) (nat)")
    _save(
        fig,
        fig_dir,
        "posonly_vs_control_heatmaps",
        {"arms": [t_ctrl.arm, t_pos.arm], "note": "vmax=14 clips the instructed-marker diagonal"},
    )


def three_space_table(t_pos: rr.ArmTensor, t_ctrl: rr.ArmTensor, out_path: Path) -> dict:
    """Reads 1-2 in all three spaces from the npz sidecars (plan §5 read 5)."""
    spaces = {
        "logp": (t_pos.G, t_ctrl.G),
        "z_marker": (t_pos.delta_z_marker, t_ctrl.delta_z_marker),
        "eos_margin": (t_pos.delta_eos_margin, t_ctrl.delta_eos_margin),
    }
    table: dict = {}
    for space, (m_pos, m_ctrl) in spaces.items():
        pos, ctrl = _space_stats(t_pos, m_pos), _space_stats(t_ctrl, m_ctrl)
        table[space] = {
            "pos_only": pos,
            "control": ctrl,
            "contrast_pos_minus_control": {stat: pos[stat] - ctrl[stat] for stat, _ in STATS},
        }
    agreement = {
        stat: {
            "signs": {
                s: float(np.sign(table[s]["contrast_pos_minus_control"][stat])) for s in spaces
            },
            "all_same_sign": len(
                {np.sign(table[s]["contrast_pos_minus_control"][stat]) for s in spaces}
            )
            == 1,
        }
        for stat, _ in STATS
    }
    payload = {
        **rr._meta(),
        "spaces": table,
        "contrast_sign_agreement": agreement,
        "note": "cross-space divergence = saturation signature (marker-measurement rule); "
        "logp is the primary behavioral space, z_marker/eos_margin the mechanistic twins",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"[posonly-figures] wrote {out_path}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #542 positives-only follow-up figures + three-space table."
    )
    ap.add_argument(
        "--eval-root",
        type=Path,
        default=Path(
            os.environ.get(
                "I542_EVAL_ROOT", str(REPO / "eval_results/issue_542/positives-only-anchor")
            )
        ),
    )
    ap.add_argument(
        "--fig-dir", type=Path, default=REPO / "figures/issue_542/positives-only-anchor"
    )
    ap.add_argument(
        "--v1-analysis",
        type=Path,
        default=REPO / "eval_results/issue_542/analysis/registered_reads_542.json",
        help="COMMITTED v1 registered reads (the six-arm envelope source)",
    )
    args = ap.parse_args()
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    t_pos = rr.ArmTensor("pos_only", args.eval_root)
    t_ctrl = rr.ArmTensor("arm1_xfam", args.eval_root)
    v1_reads = json.loads(args.v1_analysis.read_text())
    hero_dots(t_pos, t_ctrl, v1_reads, args.fig_dir)
    heatmap_pair(t_pos, t_ctrl, args.fig_dir)
    three_space_table(t_pos, t_ctrl, args.eval_root / "analysis/posonly_three_space_542.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
