#!/usr/bin/env python3
"""#552 follow-up `em-arm-mean-resp-reextraction` (OFF-POD, VM) — cross-arm
direction identity at the MEAN-OVER-RESPONSE position.

The completed #552 run found the benign and EM arms' top shift directions
overlap at |cos| = 0.501 at the END-OF-RESPONSE slot — right on the decision
boundary — and that each probe position carries its own dominant direction.
This script asks whether that overlap is position-general or end-slot-specific:
it builds each same-variant cell's mean-over-response U1 (``assemble_M(...,
use_mean_resp=True)`` + ``svd_summary``, the existing helpers) for the 3 benign
cells (this issue's persisted tensors) and the 3 re-extracted EM cells, then
computes — per ``issue552_cross_arm_analysis.py`` conventions (|cos|,
``RANDOM_COS_FLOOR_P95 = 0.033``) —

  1. within-benign (3 seed pairs), within-EM (3 pairs), and benign x EM
     (9 pairs) |cos(U1_mr, U1_mr')| at the mean-over-response position;
  2. the pre-registered VALIDITY GATE (plan v2 §6, binding before any
     interpretation): per-cell |re-extracted end-slot mean_cos_to_U1 -
     #521 persisted| <= 0.02 AND |s_top1_frac - persisted| <= 0.02, read
     from the FRESH Phase-D JSONs vs ``eval_results/issue_521/svd/``
     (cross-RUN reproduction; distinct from the 5e-4 same-tensors check
     in ``issue552_mean_resp_svd.py``);
  3. the pre-registered primary decision rule on the cross-arm median
     (KILL <= 0.05 / STRENGTHEN >= 0.40 with within-arm ceilings at or
     above the cross median / graded between) — recorded ONLY when the
     validity gate passes;
  4. the secondary concentrated-zone read per EM cell at mean-resp
     (mean cos >= 0.90 AND top-share >= 0.50).

Writes the per-pair table + verdict inputs to ``--out`` and a strip-plot
figure (within-arm ceilings left, cross-arm right, 0.033 floor dashed — the
parent cross_arm_directions shape). Interpretation stays with the analyzer;
this script only computes and records.

Run (VM, after pod termination + tensor pull)::

    uv run python scripts/issue552_mean_resp_cross_arm.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.svd_direction_constancy import (
    assemble_M,
    svd_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = (42, 137, 256)
FU_ROOT = "eval_results/issue_552/em-arm-mean-resp-reextraction"

# p95 of |cos| between random unit vectors in R^3584 (parent #521 floor;
# verbatim from issue552_cross_arm_analysis.py).
RANDOM_COS_FLOOR_P95 = 0.033

# Pre-registered decision thresholds (plan v2 §6 — no post-hoc motion).
KILL_CROSS_MEDIAN_MAX = 0.05
STRENGTHEN_CROSS_MEDIAN_MIN = 0.40

# Pre-registered validity-gate tolerance (plan v2 §5/§6): cross-RUN
# reproduction of the #521 end-slot numbers. Distinct from the 5e-4
# same-tensors cross-check (issue552_mean_resp_svd.py).
FAITHFULNESS_ATOL = 0.02

# Same-tensors consistency tolerance: the recomputed end-slot metrics and
# the fresh Phase-D JSONs derive from the SAME tensors via the SAME helpers,
# so any drift beyond LAPACK/numpy build noise means --em-svd-dir points at
# the wrong (stale) directory. Mirrors issue552_mean_resp_svd.py.
SAME_TENSORS_ATOL = 5e-4

# Concentrated-zone thresholds (plan v1 §3/§6.3, reused verbatim for the
# secondary EM mean-resp read).
CONCENTRATED_MEAN_COS_MIN = 0.90
CONCENTRATED_TOP_SHARE_MIN = 0.50


def _git_commit() -> str:
    """Return the current git commit hash (or 'unknown' if git is unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )
    except OSError:
        return "unknown"
    sha = out.stdout.strip()
    return sha if sha else "unknown"


def _abs_cos(u: np.ndarray, v: np.ndarray) -> float:
    """|cos| between two vectors (issue552_cross_arm_analysis.py convention)."""
    return float(abs(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))


def _load_cell_reads(pt_path: Path, arm: str, seed: int) -> dict:
    """Load one same-variant shift tensor; return end-slot + mean-resp SVD reads.

    Fails loud when the tensor is missing or lacks ``delta_v_mean_resp``
    (the re-extraction's whole point); asserts the persona panels match
    between the two reads.
    """
    if not pt_path.exists():
        raise FileNotFoundError(
            f"shift tensor missing: {pt_path}. Benign tensors live ONLY in the "
            f"issue-552 worktree (eval_results/issue_552/shifts/) / WandB artifact "
            f"issue552_analysis_tensors:v0; EM tensors come from the follow-up pod "
            f"run (WandB artifact issue552_em_mean_resp_tensors:v0 + VM pull)."
        )
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    shifts = payload["shifts"]
    M_end, order_end = assemble_M(shifts)
    M_mr, order_mr = assemble_M(shifts, use_mean_resp=True)
    assert order_end == order_mr, (order_end, order_mr)
    assert M_end.shape == M_mr.shape, (M_end.shape, M_mr.shape)
    svd_end = svd_summary(M_end)
    svd_mr = svd_summary(M_mr)
    return {
        "arm": arm,
        "seed": seed,
        "pt_path": str(pt_path),
        "persona_order": order_end,
        "n_personas": len(order_end),
        "end_slot": {
            "mean_cos_to_U1": float(np.mean(svd_end["cos_to_U1"])),
            "s_top1_frac": svd_end["s_top1_frac"],
        },
        "mean_resp": {
            "mean_cos_to_U1": float(np.mean(svd_mr["cos_to_U1"])),
            "s_top1_frac": svd_mr["s_top1_frac"],
        },
        "_U1_end": svd_end["U1"],
        "_U1_mr": svd_mr["U1"],
    }


def _faithfulness_gate(
    *,
    em_cells: dict[int, dict],
    em_svd_dir: Path,
    anchor_svd_dir: Path,
    seeds: tuple[int, ...],
) -> dict:
    """Plan v2 §6 validity gate: fresh EM end-slot vs #521 persisted, ±0.02.

    Reads the canonical FRESH Phase-D JSONs (``--em-svd-dir``), sanity-checks
    them against the in-memory recomputation from the tensors (same-tensors,
    5e-4 — a wrong/stale ``--em-svd-dir`` fails loud here), then compares to
    the persisted #521 anchors at the pre-registered ±0.02. Gate FAIL is
    RECORDED, not raised: a failed gate halts interpretation downstream (the
    decision rule is withheld), and the numbers themselves are the
    infrastructure finding.
    """
    per_cell: dict[str, dict] = {}
    all_pass = True
    for seed in seeds:
        cell = f"same_em_seed{seed}"
        fresh_path = em_svd_dir / f"{cell}.json"
        anchor_path = anchor_svd_dir / f"{cell}.json"
        if not fresh_path.exists():
            raise FileNotFoundError(
                f"fresh Phase-D JSON missing: {fresh_path} — run the pod driver "
                f"(scripts/run_issue552_emresp_followup.sh) first and pull "
                f"{FU_ROOT}/svd/ to the VM."
            )
        if not anchor_path.exists():
            raise FileNotFoundError(f"#521 anchor JSON missing: {anchor_path} (git, main)")
        fresh = json.loads(fresh_path.read_text())
        anchor = json.loads(anchor_path.read_text())

        # Same-tensors consistency: recomputed-from-tensor end slot must match
        # the fresh JSON to numerical noise, else --em-svd-dir is mis-wired.
        recomputed = em_cells[seed]["end_slot"]
        for field in ("mean_cos_to_U1", "s_top1_frac"):
            drift = abs(float(recomputed[field]) - float(fresh[field]))
            if drift > SAME_TENSORS_ATOL:
                raise AssertionError(
                    f"{cell}: recomputed end-slot {field} ({recomputed[field]:.6f}) "
                    f"diverges from the fresh Phase-D JSON {fresh_path} "
                    f"({float(fresh[field]):.6f}) by {drift:.2e} > {SAME_TENSORS_ATOL} — "
                    f"--em-svd-dir and --em-shifts-dir do not describe the same run."
                )

        d_cos = abs(float(fresh["mean_cos_to_U1"]) - float(anchor["mean_cos_to_U1"]))
        d_share = abs(float(fresh["s_top1_frac"]) - float(anchor["s_top1_frac"]))
        cell_pass = d_cos <= FAITHFULNESS_ATOL and d_share <= FAITHFULNESS_ATOL
        all_pass = all_pass and cell_pass
        per_cell[cell] = {
            "fresh": {
                "mean_cos_to_U1": float(fresh["mean_cos_to_U1"]),
                "s_top1_frac": float(fresh["s_top1_frac"]),
                "path": str(fresh_path),
            },
            "anchor": {
                "mean_cos_to_U1": float(anchor["mean_cos_to_U1"]),
                "s_top1_frac": float(anchor["s_top1_frac"]),
                "path": str(anchor_path),
            },
            "abs_diff": {"mean_cos_to_U1": d_cos, "s_top1_frac": d_share},
            "pass": cell_pass,
        }
        logger.info(
            "[gate] %s: |d mean_cos|=%.4f |d top_share|=%.4f -> %s",
            cell,
            d_cos,
            d_share,
            "PASS" if cell_pass else "FAIL",
        )
    return {
        "atol": FAITHFULNESS_ATOL,
        "rule": (
            "per-cell |fresh end-slot mean_cos_to_U1 - #521 persisted| <= 0.02 AND "
            "|s_top1_frac - persisted| <= 0.02, all cells (plan v2 §6, binding "
            "before any interpretation)"
        ),
        "pass": all_pass,
        "per_cell": per_cell,
    }


def _classify(
    cross_median: float, within_benign_median: float, within_em_median: float
) -> tuple[str, str]:
    """Apply the pre-registered plan v2 §6 primary decision rule.

    Returns (classification, note). Computation + recording only —
    interpretation belongs to the analyzer.
    """
    if cross_median <= KILL_CROSS_MEDIAN_MAX:
        return (
            "kill_corpus_level_reading",
            f"cross-arm median {cross_median:.3f} <= {KILL_CROSS_MEDIAN_MAX} "
            f"(random-floor neighborhood, floor p95 = {RANDOM_COS_FLOOR_P95})",
        )
    ceilings_comparable = within_benign_median >= cross_median and within_em_median >= cross_median
    if cross_median >= STRENGTHEN_CROSS_MEDIAN_MIN and ceilings_comparable:
        return (
            "strengthens_corpus_level_reading",
            f"cross-arm median {cross_median:.3f} >= {STRENGTHEN_CROSS_MEDIAN_MIN} "
            f"with within-arm ceilings at or above it "
            f"(benign {within_benign_median:.3f}, EM {within_em_median:.3f})",
        )
    if cross_median >= STRENGTHEN_CROSS_MEDIAN_MIN:
        return (
            "graded",
            f"cross-arm median {cross_median:.3f} >= {STRENGTHEN_CROSS_MEDIAN_MIN} but a "
            f"within-arm ceiling sits BELOW it (benign {within_benign_median:.3f}, "
            f"EM {within_em_median:.3f}) — the §6 strengthen branch requires "
            f"ceilings comparable; reported graded with per-seed structure",
        )
    return (
        "graded",
        f"cross-arm median {cross_median:.3f} between {KILL_CROSS_MEDIAN_MAX} and "
        f"{STRENGTHEN_CROSS_MEDIAN_MIN} — reported with per-seed structure",
    )


def _make_figure(
    *,
    within_benign: dict[str, float],
    within_em: dict[str, float],
    cross: dict[str, float],
    gate_pass: bool,
    figure_dir: Path,
) -> dict[str, Path]:
    """Strip plot: within-arm ceilings left, cross-arm right, 0.033 floor dashed."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    c_benign = paper_palette_role("control")
    c_em = paper_palette_role("primary")
    c_neutral = paper_palette_role("neutral")
    c_accent = paper_palette_role("accent")

    groups: dict[str, tuple[list[float], str]] = {
        "within\nbenign": (list(within_benign.values()), c_benign),
        "within\nmisalignment": (list(within_em.values()), c_em),
        "benign x\nmisalignment": (list(cross.values()), c_neutral),
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.subplots_adjust(bottom=0.16, top=0.86, left=0.11, right=0.97)
    rng = np.random.default_rng(1)
    for x, (vals, color) in enumerate(groups.values()):
        jit = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(x + jit, vals, s=30, color=color, alpha=0.8, zorder=3)
        ax.scatter(x, float(np.median(vals)), marker="_", s=420, color="black", zorder=5)
    ax.axvline(1.5, color="0.85", linewidth=1.0, zorder=1)
    ax.text(
        0.5,
        1.02,
        "same corpus, different seed",
        transform=ax.get_xaxis_transform(),
        ha="center",
        fontsize=8,
        color="0.45",
    )
    ax.text(
        2.0,
        1.02,
        "different corpora",
        transform=ax.get_xaxis_transform(),
        ha="center",
        fontsize=8,
        color="0.45",
    )
    ax.axhline(
        RANDOM_COS_FLOOR_P95,
        color=c_accent,
        linestyle="--",
        linewidth=1.2,
        label="random-direction floor (p95 = 0.033)",
    )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups.keys()), fontsize=8)
    ax.set_ylabel("|cos(top direction, top direction')|")
    ax.set_ylim(0, 1.05)
    title = "Mean-over-response read: is the benign top direction the misalignment one?"
    if not gate_pass:
        title += "\n(VALIDITY GATE FAILED — re-extraction not faithful; do not interpret)"
    ax.set_title(title, pad=18)
    ax.legend(frameon=False, fontsize=8)
    written = savefig_paper(fig, "cross_arm_mean_resp_directions", dir=figure_dir)
    plt.close(fig)
    return written


def main() -> int:
    """CLI entrypoint: cross-arm |cos| at the mean-resp position + validity gate."""
    parser = argparse.ArgumentParser(
        description="#552 cross-arm direction identity at the mean-over-response position.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benign-shifts-dir",
        default="eval_results/issue_552/shifts",
        help="Benign same-variant shift tensors (this issue's completed run).",
    )
    parser.add_argument(
        "--em-shifts-dir",
        default=f"{FU_ROOT}/shifts",
        help="Re-extracted EM same-variant shift tensors (the follow-up pod run).",
    )
    parser.add_argument(
        "--em-svd-dir",
        default=f"{FU_ROOT}/svd",
        help="FRESH Phase-D per-cell SVD JSONs from the re-extraction (gate input).",
    )
    parser.add_argument(
        "--anchor-svd-dir",
        default="eval_results/issue_521/svd",
        help="#521 persisted end-slot SVD JSONs (the cross-run faithfulness anchor).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--out", default=f"{FU_ROOT}/cross_arm_mean_resp/summary.json")
    parser.add_argument("--figure-dir", default="figures/issue_552")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    seeds = tuple(args.seeds)
    benign_dir = Path(args.benign_shifts_dir)
    em_dir = Path(args.em_shifts_dir)

    # ── Load cells (same-trajectory variant only — the headline variant). ─
    benign_cells = {
        s: _load_cell_reads(benign_dir / f"same_benign_seed{s}.pt", "benign", s) for s in seeds
    }
    em_cells = {s: _load_cell_reads(em_dir / f"same_em_seed{s}.pt", "em", s) for s in seeds}

    # ── Validity gate (binding BEFORE any interpretation, plan v2 §6). ────
    gate = _faithfulness_gate(
        em_cells=em_cells,
        em_svd_dir=Path(args.em_svd_dir),
        anchor_svd_dir=Path(args.anchor_svd_dir),
        seeds=seeds,
    )

    # ── Pairwise |cos| at the mean-resp position. ─────────────────────────
    u1_b = {s: benign_cells[s]["_U1_mr"] for s in seeds}
    u1_e = {s: em_cells[s]["_U1_mr"] for s in seeds}
    within_benign = {
        f"benign_seed{a}_x_benign_seed{b}": _abs_cos(u1_b[a], u1_b[b])
        for a, b in combinations(seeds, 2)
    }
    within_em = {
        f"em_seed{a}_x_em_seed{b}": _abs_cos(u1_e[a], u1_e[b]) for a, b in combinations(seeds, 2)
    }
    cross = {
        f"benign_seed{a}_x_em_seed{b}": _abs_cos(u1_b[a], u1_e[b]) for a in seeds for b in seeds
    }
    medians = {
        "within_benign": float(np.median(list(within_benign.values()))),
        "within_em": float(np.median(list(within_em.values()))),
        "cross_benign_x_em": float(np.median(list(cross.values()))),
    }

    # ── Decision rule — applied ONLY when the gate passed. ────────────────
    if gate["pass"]:
        classification, rule_note = _classify(
            medians["cross_benign_x_em"], medians["within_benign"], medians["within_em"]
        )
    else:
        classification = "gate_failed_interpretation_halted"
        rule_note = (
            "validity gate FAILED — the re-extraction does not reproduce the #521 "
            "end-slot numbers within ±0.02; per plan v2 §6 the mean-resp numbers are "
            "NOT read against the decision rule (infrastructure finding; diagnose "
            "adapter identity / lockfile / variant)"
        )
    logger.info("[verdict] %s — %s", classification, rule_note)

    # ── Secondary read: EM concentrated zone at mean-resp, per cell. ──────
    secondary = {}
    for s in seeds:
        mr = em_cells[s]["mean_resp"]
        secondary[f"same_em_seed{s}"] = {
            "mean_cos_to_U1": mr["mean_cos_to_U1"],
            "s_top1_frac": mr["s_top1_frac"],
            "concentrated": bool(
                mr["mean_cos_to_U1"] >= CONCENTRATED_MEAN_COS_MIN
                and mr["s_top1_frac"] >= CONCENTRATED_TOP_SHARE_MIN
            ),
        }

    figure_paths = None
    if not args.no_figure:
        written = _make_figure(
            within_benign=within_benign,
            within_em=within_em,
            cross=cross,
            gate_pass=bool(gate["pass"]),
            figure_dir=Path(args.figure_dir),
        )
        figure_paths = {k: str(v) for k, v in written.items()}

    def _strip(cells: dict[int, dict]) -> dict[str, dict]:
        return {
            f"same_{v['arm']}_seed{s}": {k: x for k, x in v.items() if not k.startswith("_")}
            for s, v in cells.items()
        }

    summary = {
        "issue": 552,
        "followup": "em-arm-mean-resp-reextraction",
        "position": "mean_over_response",
        "variant": "same",
        "random_cos_floor_p95": RANDOM_COS_FLOOR_P95,
        "validity_gate": gate,
        "pairs": {
            "within_benign": within_benign,
            "within_em": within_em,
            "cross_benign_x_em": cross,
        },
        "medians": medians,
        "decision_rule": {
            "kill": f"cross median <= {KILL_CROSS_MEDIAN_MAX}",
            "strengthen": (
                f"cross median >= {STRENGTHEN_CROSS_MEDIAN_MIN} AND within-arm medians "
                f"at or above the cross median"
            ),
            "between": "graded, with per-seed structure",
            "applied": bool(gate["pass"]),
            "classification": classification,
            "note": rule_note,
        },
        "secondary_em_concentrated_zone": secondary,
        "per_cell": {**_strip(benign_cells), **_strip(em_cells)},
        "figure": figure_paths,
        "metadata": {
            "git_commit": _git_commit(),
            "script": "scripts/issue552_mean_resp_cross_arm.py",
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "[phase=done] wrote %s (gate_pass=%s, classification=%s, cross_median=%.3f)",
        out_path,
        gate["pass"],
        classification,
        medians["cross_benign_x_em"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
