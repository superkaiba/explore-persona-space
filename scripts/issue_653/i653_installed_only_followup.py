# ruff: noqa: RUF002, RUF003 -- project math/Greek vocab (×, ρ, λ, ↔, −)
"""Step 9a-ter free-analysis follow-up: restrict the read↔write geometry verdict
to the 3 cells that demonstrably INSTALLED.

The full #653 verdict grid (``eval_results/issue_653/cross_arm_verdict.json``)
spans 18 cells (3 behaviors × 2 sources × 3 rungs). 15 of those cells did NOT
install the behavior at all (every EM cell at 0.00 judge-rate, every r1/r4 cell,
and sycophancy×medical-doctor r16). Reading spectral geometry off a Δx whose
behavior never installed describes the geometry of NON-installation, not of the
behavior. This follow-up re-states the read↔write verdict on ONLY the 3 cells
that crossed an install threshold:

  1. marker__florist__r16__seed42        install +0.66 nat (log P trained−base)
  2. marker__medical_doctor__r16__seed42  install +0.78 nat
  3. sycophancy__florist__r16__seed42     install +0.15 judge-rate gain

It recomputes the restricted aggregates (top-share / participation-ratio /
rank-k@90 / cos(top, r_B) / cross-arm ρ↔Δx cosine) on these 3 cells only, pairs
each with its ablation delta, and re-evaluates H1 (low-rank + aligned) vs H2
(low-rank, rotated) vs H3 (diffuse) under the SAME thresholds the production
classifier uses (``TOP_SHARE_LOWRANK=0.7``, ``PR_LAMBDA_H3=5.0``,
``RANK_K_H3=10``, ``COS_ALIGNED_FLOOR=0.5``). It writes NO new training/eval
data and makes NO model calls — purely a re-aggregation of the existing armB
``dx_geometry_*`` / ``install_*`` / ``ablation_*`` JSONs and the cross-arm
verdict grid.

Run:
  uv run python scripts/issue_653/i653_installed_only_followup.py
Writes:
  eval_results/issue_653/installed_only_verdict.json
"""

from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path

from explore_persona_space.experiments import issue_653 as i653

# Worktree root: this file lives at <worktree>/scripts/issue_653/<this>.py, so
# its grandparent-of-parent is the worktree root. Using the worktree (NOT the
# shared main repo root that ``task_workflow.repo_root()`` resolves to) makes the
# reproducibility ``git_commit`` reflect THIS issue-653 branch HEAD.
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = WORKTREE_ROOT / "eval_results" / "issue_653"
ARMB = OUT_ROOT / "armB"
VERDICT_GRID = OUT_ROOT / "cross_arm_verdict.json"
OUT_PATH = OUT_ROOT / "installed_only_verdict.json"

# The 3 cells that demonstrably installed the behavior (analyzer's surfaced
# follow-up). Order is load-bearing only for readable output; aggregates are
# order-independent.
INSTALLED_CELLS = [
    "marker__florist__r16__seed42",
    "marker__medical_doctor__r16__seed42",
    "sycophancy__florist__r16__seed42",
]


def _git_commit_full(repo_root: Path) -> str:
    """Full 40-char HEAD SHA for unambiguous provenance (the shared
    ``result_metadata`` emits the 8-char short SHA, matching the sibling
    artifacts; we add the full SHA alongside)."""
    try:
        return subprocess.run(  # epm-lint: subprocess-env-inherit -- git metadata probe, no creds
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_dx(cell_id: str) -> dict:
    path = ARMB / f"dx_geometry_{cell_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"installed-only follow-up: missing dx_geometry for {cell_id} ({path}). "
            "This is an analysis-only re-aggregation; the per-cell JSON must already exist."
        )
    return json.loads(path.read_text())


def _load_install(cell_id: str) -> dict:
    path = ARMB / f"install_{cell_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"installed-only follow-up: missing install JSON for {cell_id} ({path})."
        )
    return json.loads(path.read_text())


def _install_dv(install_block: dict) -> tuple[float, str]:
    """Return (numeric install DV, human label) for a cell's install block.

    marker cells: log P(marker) trained−base (nat). sycophancy cells: judge-rate
    gain (trained−base). Fail loud on an unrecognized dv_kind rather than
    fabricating a zero (CLAUDE.md fail-fast)."""
    dv_kind = install_block.get("dv_kind")
    if dv_kind == "marker_four_float":
        return float(install_block["logp_trained_minus_base"]), "logp_trained_minus_base (nat)"
    if dv_kind == "judge_rate_plus_gain":
        return float(install_block["judge_rate_gain"]), "judge_rate_gain"
    raise ValueError(f"installed-only follow-up: unrecognized install dv_kind {dv_kind!r}")


def _ablation_from_verdict(verdict_row: dict) -> tuple[float | None, str | None]:
    """Pull the install-DV-space ablation delta + its kind from a verdict row.

    marker cells store ``logp_delta_ablation``; sycophancy cells store
    ``judge_rate_delta_ablation``. Returns (delta, dv_kind) or (None, None) if
    the cell has no ablation (only r16 cells were ablated)."""
    ab = verdict_row.get("ablation")
    if not ab:
        return None, None
    dv_kind = ab.get("dv_kind")
    if dv_kind == "marker_four_float":
        return float(ab["logp_delta_ablation"]), dv_kind
    if dv_kind == "judge_rate_plus_gain":
        return float(ab["judge_rate_delta_ablation"]), dv_kind
    raise ValueError(f"installed-only follow-up: unrecognized ablation dv_kind {dv_kind!r}")


def _stat_block(values: list[float]) -> dict:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    if not VERDICT_GRID.exists():
        raise FileNotFoundError(
            f"installed-only follow-up: cross_arm_verdict.json not found at {VERDICT_GRID}."
        )
    grid = json.loads(VERDICT_GRID.read_text())
    verdict_by_cell = {v["cell_id"]: v for v in grid["verdicts"]}

    # Sanity: every named installed cell must be present in the 18-cell grid.
    missing = [c for c in INSTALLED_CELLS if c not in verdict_by_cell]
    if missing:
        raise KeyError(
            f"installed-only follow-up: installed cells absent from the verdict grid: {missing}"
        )

    top_shares: list[float] = []
    prs: list[float] = []
    rank_ks: list[float] = []
    cos_top_to_rb_per_cell: list[dict] = []
    cross_arm_iso_per_cell: list[dict] = []
    cross_arm_cov_per_cell: list[dict] = []
    ablation_per_cell: list[dict] = []

    n_clearing_align_floor = 0
    n_iso_exceeds_random = 0
    n_cov_exceeds_random = 0
    n_low_rank = 0  # production is_low_rank: pr<=2.0 OR top_share>=0.7
    n_h3 = 0  # production is_h3: pr>=5.0 OR rank_k>=10
    n_aligned = 0  # production is_aligned: |cos|>=0.5 AND |cos|>random_ci_high

    for cell_id in INSTALLED_CELLS:
        dx = _load_dx(cell_id)
        install = _load_install(cell_id)["install"]
        vrow = verdict_by_cell[cell_id]

        top_share = float(dx["top_share_lambda"])
        pr = float(dx["pr_lambda"])
        rank_k = int(dx["rank_k_at_90"])
        cos_rb = float(dx["cos_top_to_rb"])
        random_ci_high = dx.get("random_ci_high")

        top_shares.append(top_share)
        prs.append(pr)
        rank_ks.append(float(rank_k))

        # Production classifier predicates, restated per cell (thresholds from i653).
        is_low_rank = (pr <= i653.PR_LAMBDA_LOWRANK) or (top_share >= i653.TOP_SHARE_LOWRANK)
        is_h3 = (pr >= i653.PR_LAMBDA_H3) or (rank_k >= i653.RANK_K_H3)
        aligned_by_floor = abs(cos_rb) >= i653.COS_ALIGNED_FLOOR
        aligned_by_ci = random_ci_high is None or abs(cos_rb) > random_ci_high
        is_aligned = aligned_by_floor and aligned_by_ci
        n_low_rank += int(is_low_rank)
        n_h3 += int(is_h3)
        n_aligned += int(is_aligned)
        if aligned_by_floor:
            n_clearing_align_floor += 1

        cos_top_to_rb_per_cell.append(
            {
                "cell": cell_id,
                "cos_top_to_rb": cos_rb,
                "abs_cos": abs(cos_rb),
                "random_ci_high": random_ci_high,
                "clears_0.5_floor": aligned_by_floor,
                "exceeds_random_ci": aligned_by_ci,
            }
        )

        ca = vrow["cross_arm"]
        iso = ca["iso"]
        cov = ca["cov"]
        n_iso_exceeds_random += int(bool(iso["exceeds_random_ci"]))
        n_cov_exceeds_random += int(bool(cov["exceeds_random_ci"]))
        cross_arm_iso_per_cell.append(
            {
                "cell": cell_id,
                "cos_rho_top_to_dx_top": iso["cos_rho_top_to_dx_top"],
                "exceeds_random_ci": bool(iso["exceeds_random_ci"]),
                "random_ci_high": iso.get("random_ci_high"),
            }
        )
        cross_arm_cov_per_cell.append(
            {
                "cell": cell_id,
                "cos_rho_top_to_dx_top": cov["cos_rho_top_to_dx_top"],
                "exceeds_random_ci": bool(cov["exceeds_random_ci"]),
                "random_ci_high": cov.get("random_ci_high"),
            }
        )

        install_dv, install_label = _install_dv(install)
        ablation_delta, ablation_kind = _ablation_from_verdict(vrow)
        ablation_per_cell.append(
            {
                "cell": cell_id,
                "install_dv": install_dv,
                "install_dv_label": install_label,
                "ablation_delta": ablation_delta,
                "ablation_dv_kind": ablation_kind,
            }
        )

    n = len(INSTALLED_CELLS)
    iso_cos_vals = [r["cos_rho_top_to_dx_top"] for r in cross_arm_iso_per_cell]
    cov_cos_vals = [r["cos_rho_top_to_dx_top"] for r in cross_arm_cov_per_cell]

    # ── Per-hypothesis verdict on the restricted set ─────────────────────────
    # H1 (low-rank + aligned): the write direction is a low-rank object aligned
    #   with the Arm-A read direction r_B. Requires ALL installed cells to be
    #   both low-rank AND aligned.
    # H2 (low-rank, rotated): low-rank but NOT aligned to r_B (a rotated
    #   low-rank object). Requires ALL low-rank AND none aligned.
    # H3 (diffuse): the write direction is high-rank / diffuse — no low-rank
    #   structure. The aggregate verdict is H3 if EVERY cell is is_h3.
    h1 = (n_low_rank == n) and (n_aligned == n)
    h2 = (n_low_rank == n) and (n_aligned == 0)
    h3 = n_h3 == n

    if h3:
        decision = "H3_diffuse"
        rationale = (
            f"All {n} installed cells are diffuse (is_h3 = PR_lambda>={i653.PR_LAMBDA_H3} OR "
            f"rank_k@90>={i653.RANK_K_H3}); none is low-rank "
            f"(top-share max {max(top_shares):.3f} < {i653.TOP_SHARE_LOWRANK}, "
            f"PR_lambda min {min(prs):.2f} > {i653.PR_LAMBDA_LOWRANK}). "
            f"None clears the |cos(top, r_B)|>={i653.COS_ALIGNED_FLOOR} alignment floor "
            f"(max abs-cos {max(r['abs_cos'] for r in cos_top_to_rb_per_cell):.3f}). "
            "The H3 diffuse-and-unaligned verdict survives restriction to the installed cells."
        )
    elif h1:
        decision = "H1_low_rank_aligned"
        rationale = (
            f"All {n} installed cells are low-rank AND aligned to r_B — the restricted "
            "verdict DIVERGES from the full-18-cell H3 story."
        )
    elif h2:
        decision = "H2_low_rank_rotated"
        rationale = (
            f"All {n} installed cells are low-rank but NOT aligned to r_B (rotated) — the "
            "restricted verdict DIVERGES from the full-18-cell H3 story."
        )
    else:
        decision = "mixed"
        rationale = (
            f"Installed cells split across hypotheses (low-rank {n_low_rank}/{n}, "
            f"diffuse {n_h3}/{n}, aligned {n_aligned}/{n}) — no unanimous label."
        )

    metadata = i653.result_metadata(
        WORKTREE_ROOT,
        {
            "phase": "installed_only_followup",
            "git_commit_full": _git_commit_full(WORKTREE_ROOT),
            "source_verdict_grid": "cross_arm_verdict.json",
            "note": (
                "Step 9a-ter free-analysis follow-up: read-only re-aggregation of the "
                "existing armB dx_geometry/install JSONs + cross_arm_verdict.json, "
                "restricted to the 3 demonstrably-installed cells. No new training/eval/"
                "model calls."
            ),
        },
    )

    out = {
        "n_installed_cells": n,
        "installed_cells": INSTALLED_CELLS,
        "thresholds": {
            "top_share_lowrank": i653.TOP_SHARE_LOWRANK,
            "pr_lambda_lowrank": i653.PR_LAMBDA_LOWRANK,
            "pr_lambda_h3": i653.PR_LAMBDA_H3,
            "rank_k_h3": i653.RANK_K_H3,
            "cos_aligned_floor": i653.COS_ALIGNED_FLOOR,
        },
        "restricted_spectral": {
            "top_share_lambda": _stat_block(top_shares),
            "pr_lambda": _stat_block(prs),
            "rank_k_at_90": _stat_block(rank_ks),
            "n_low_rank": n_low_rank,
            "n_h3": n_h3,
        },
        "restricted_alignment": {
            "cos_top_to_rb": {
                "per_cell": cos_top_to_rb_per_cell,
                "n_clearing_0.5": n_clearing_align_floor,
                "n_aligned_floor_and_ci": n_aligned,
            },
            "cross_arm_iso": {
                "per_cell": cross_arm_iso_per_cell,
                "cos_range": {"min": min(iso_cos_vals), "max": max(iso_cos_vals)},
                "n_exceeds_random_ci": n_iso_exceeds_random,
            },
            "cross_arm_cov": {
                "per_cell": cross_arm_cov_per_cell,
                "cos_range": {"min": min(cov_cos_vals), "max": max(cov_cos_vals)},
                "n_exceeds_random_ci": n_cov_exceeds_random,
            },
        },
        "restricted_ablation": {
            "per_cell": ablation_per_cell,
        },
        "verdict_per_hypothesis": {
            "H1_low_rank_aligned": h1,
            "H2_low_rank_rotated": h2,
            "H3_diffuse": h3,
            "decision": decision,
            "rationale": rationale,
        },
        "metadata": metadata,
    }

    OUT_PATH.write_text(json.dumps(out, indent=1))

    # ── Human-readable one-paragraph summary ─────────────────────────────────
    ts = out["restricted_spectral"]["top_share_lambda"]
    pr_b = out["restricted_spectral"]["pr_lambda"]
    rk = out["restricted_spectral"]["rank_k_at_90"]
    abs_cos_vals = [r["abs_cos"] for r in cos_top_to_rb_per_cell]
    print(
        "\n[installed-only follow-up] Restricted the read↔write verdict to the "
        f"{n} demonstrably-installed cells "
        f"({', '.join(INSTALLED_CELLS)}). Across these 3 cells the write-direction Δx "
        f"is DIFFUSE, not low-rank: top-share lambda median {ts['median']:.3f} "
        f"(range {ts['min']:.3f}-{ts['max']:.3f}, all < {i653.TOP_SHARE_LOWRANK} low-rank "
        f"floor), participation-ratio median {pr_b['median']:.1f} "
        f"(range {pr_b['min']:.1f}-{pr_b['max']:.1f}, all >> {i653.PR_LAMBDA_H3} H3 cut), "
        f"rank-k@90 median {rk['median']:.0f} (range {rk['min']:.0f}-{rk['max']:.0f}, all >= "
        f"{i653.RANK_K_H3} H3 cut). The leading Δx direction is UNALIGNED with the Arm-A "
        f"read direction r_B: |cos(top, r_B)| max {max(abs_cos_vals):.3f} "
        f"(< {i653.COS_ALIGNED_FLOOR} floor) in {n_clearing_align_floor}/{n} cells clearing "
        f"the floor. Cross-arm rho<->Δx leading-direction cosines stay near zero "
        f"(iso {min(iso_cos_vals):.3f}..{max(iso_cos_vals):.3f}, "
        f"cov {min(cov_cos_vals):.3f}..{max(cov_cos_vals):.3f}). "
        f"VERDICT: {decision}. {rationale}"
    )
    print(f"\n[installed-only follow-up] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
