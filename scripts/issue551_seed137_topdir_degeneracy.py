#!/usr/bin/env python3
"""#551 free-analysis follow-up (b): seed-137 top-direction degeneracy diagnostic.

Question: is marker seed 137's near-zero weighted-vs-unit-norm per-persona
ordering correlation (rank corr 0.222, vs 0.824/0.846 for seeds 42/256 in
``unitnorm_reread.json``) explained by a DEGENERATE spectrum top (small
sigma1/sigma2 gap; top direction unstable under question resampling) rather
than persona-level instability?

Cells: the 6 trained-model-text (``same``-variant) cells — marker seeds
42/137/256 primary, EM seeds as context. Per cell, under BOTH the weighted
and unit-norm (column-normalized) reads of the layer-14 matrix:

1. singular-value top gap sigma1/sigma2 (+ the raw sigma vector) and an
   effective rank — the participation ratio ``(sum(s))^2 / sum(s^2)``;
2. 1,000-draw bootstrap-over-questions top-direction stability from the
   persisted per-question tensors ``delta_v_per_q`` (layer 14 only):
   resample the 20 panel questions with replacement (same indices for every
   persona), recompute each persona's mean shift over the resampled
   questions, assemble M (unit-normalize columns for the unit-norm read),
   take the top singular direction, record |cos| to the full-panel top
   direction; report mean/median/p5 over the 1,000 draws;
3. principal angles between the weighted and unit-norm top-2 left-singular
   subspaces (singular values of ``U_w[:, :2]^T @ U_u[:, :2]``).

Falsification readout (encoded under ``falsification``): seed 137 shows a
sigma gap and bootstrap stability comparable to seeds 42/256 (i.e. NOT
degenerate) — the weighted-vs-unit-norm reshuffle then flags a real
difference in which personas the unit-norm read promotes (a caveat for
downstream per-persona-ordering use), not a degenerate-top artifact.

Zero GPU; reads only the persisted shift tensors. Run from the repo root::

    uv run python scripts/issue551_seed137_topdir_degeneracy.py \\
        --local-shifts-dir eval_results/issue_551/shifts \\
        --out eval_results/issue_551/seed137-topdir-degeneracy/seed137_topdir_degeneracy.json
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from issue551_controls import (
    SOURCE_PERSONA,
    _git_commit,
    _load_cell,
    _same_cells,
    _write_json,
)
from issue551_unitnorm_reread import unit_normalize_columns

from explore_persona_space.analysis.svd_direction_constancy import assemble_M, cosine

logger = logging.getLogger(__name__)

N_BOOT = 1000
N_QUESTIONS = 20
BOOTSTRAP_BASE_SEED = 20260610  # fixed; combined with (cell seed, arm index) per cell
SIGMA_GAP_COMPARABILITY_FACTOR = 0.8  # seed137 gap >= factor * min(other marker seeds)
BOOTSTRAP_COMPARABILITY_MARGIN = 0.05  # seed137 mean|cos| >= min(other marker seeds) - margin
EFFECTIVE_RANK_DEFINITION = (
    "participation ratio (sum(s))^2 / sum(s^2) over the full singular-value vector"
)


def _svd_reads(M: np.ndarray) -> dict[str, dict]:
    """Full SVD of the weighted and unit-norm reads of M (H x N)."""
    out: dict[str, dict] = {}
    for read, mat in (("weighted", M), ("unitnorm", unit_normalize_columns(M))):
        u, s, _ = np.linalg.svd(mat.astype(np.float64), full_matrices=False)
        out[read] = {"U": u, "s": s}
    return out


def _spectrum_stats(s: np.ndarray) -> dict:
    return {
        "singular_values": [float(x) for x in s],
        "sigma1_over_sigma2": float(s[0] / s[1]),
        "s_top1_frac": float(s[0] / s.sum()),
        "effective_rank_participation_ratio": float((s.sum() ** 2) / (s**2).sum()),
        "effective_rank_definition": EFFECTIVE_RANK_DEFINITION,
    }


def _bootstrap_topdir(
    per_q: np.ndarray, u1_full: dict[str, np.ndarray], rng: np.random.Generator
) -> dict[str, dict]:
    """1,000-draw question-resampling stability of the top direction.

    per_q: (N_personas, N_QUESTIONS, H). Same resampled question indices are
    applied to every persona (the questions are the shared panel axis).
    """
    abs_cos: dict[str, np.ndarray] = {
        read: np.empty(N_BOOT, dtype=np.float64) for read in ("weighted", "unitnorm")
    }
    for b in range(N_BOOT):
        idx = rng.integers(0, N_QUESTIONS, size=N_QUESTIONS)
        m_boot = per_q[:, idx, :].mean(axis=1).T  # (H, N_personas)
        for read in ("weighted", "unitnorm"):
            mat = m_boot if read == "weighted" else unit_normalize_columns(m_boot)
            u, _, _ = np.linalg.svd(mat.astype(np.float64), full_matrices=False)
            abs_cos[read][b] = abs(cosine(u[:, 0], u1_full[read]))
    return {
        read: {
            "n_draws": N_BOOT,
            "mean_abs_cos_to_full_topdir": float(vals.mean()),
            "median_abs_cos_to_full_topdir": float(np.median(vals)),
            "p5_abs_cos_to_full_topdir": float(np.percentile(vals, 5)),
        }
        for read, vals in abs_cos.items()
    }


def _principal_angles(u_w: np.ndarray, u_u: np.ndarray, k: int = 2) -> dict:
    """Principal angles between the top-k left-singular subspaces."""
    overlap = u_w[:, :k].T @ u_u[:, :k]
    sv = np.linalg.svd(overlap, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(sv))
    return {
        "k": k,
        "cos_principal_angles": [float(x) for x in sv],
        "principal_angles_deg": [float(x) for x in angles_deg],
    }


def analyze(shifts_dir: Path) -> dict:
    """Run the degeneracy diagnostic over the 6 trained-model-text cells."""
    per_cell: dict[str, dict] = {}
    for arm_idx, cell in enumerate(_same_cells()):
        shifts = _load_cell(shifts_dir, cell)
        M, personas = assemble_M(shifts)  # layer-14 end-slot read, sorted order
        assert personas.index(SOURCE_PERSONA) >= 0
        per_q = np.stack(
            [shifts[p]["delta_v_per_q"].detach().float().cpu().numpy() for p in personas]
        )  # (N, Q, H)
        assert per_q.shape[:2] == (len(personas), N_QUESTIONS), per_q.shape
        # Consistency: the persisted delta_v IS the per-question mean (verified
        # exactly on these tensors); fail loud if a future rig breaks that.
        max_dev = float(np.abs(per_q.mean(axis=1).T - M).max())
        if max_dev > 1e-5:
            raise ValueError(
                f"{cell.name}: delta_v differs from delta_v_per_q.mean() "
                f"(max|delta|={max_dev:.2e}); the bootstrap would not be resampling "
                f"the panel that produced the full-panel read."
            )

        reads = _svd_reads(M)
        u1_full = {read: reads[read]["U"][:, 0] for read in reads}
        rng = np.random.default_rng([BOOTSTRAP_BASE_SEED, cell.seed, arm_idx])
        boot = _bootstrap_topdir(per_q, u1_full, rng)
        per_cell[cell.name] = {
            "arm": cell.arm,
            "seed": cell.seed,
            "delta_v_vs_per_q_mean_max_abs_dev": max_dev,
            "spectrum": {read: _spectrum_stats(reads[read]["s"]) for read in reads},
            "bootstrap_topdir_stability": boot,
            "principal_angles_weighted_vs_unitnorm_top2": _principal_angles(
                reads["weighted"]["U"], reads["unitnorm"]["U"]
            ),
        }
        logger.info(
            "[%s] gap_w=%.3f gap_u=%.3f boot_mean|cos|_w=%.4f boot_mean|cos|_u=%.4f angles_deg=%s",
            cell.name,
            per_cell[cell.name]["spectrum"]["weighted"]["sigma1_over_sigma2"],
            per_cell[cell.name]["spectrum"]["unitnorm"]["sigma1_over_sigma2"],
            boot["weighted"]["mean_abs_cos_to_full_topdir"],
            boot["unitnorm"]["mean_abs_cos_to_full_topdir"],
            [
                round(a, 1)
                for a in per_cell[cell.name]["principal_angles_weighted_vs_unitnorm_top2"][
                    "principal_angles_deg"
                ]
            ],
        )

    return {"per_cell": per_cell, "falsification": _falsification(per_cell)}


def _falsification(per_cell: dict[str, dict]) -> dict:
    """Encode the pre-registered falsification readout (marker cells)."""
    mk = {v["seed"]: v for v in per_cell.values() if v["arm"] == "marker"}
    others = [s for s in mk if s != 137]
    checks: dict[str, dict] = {}
    for read in ("weighted", "unitnorm"):
        gap_137 = mk[137]["spectrum"][read]["sigma1_over_sigma2"]
        gap_min_others = min(mk[s]["spectrum"][read]["sigma1_over_sigma2"] for s in others)
        boot_137 = mk[137]["bootstrap_topdir_stability"][read]["mean_abs_cos_to_full_topdir"]
        boot_min_others = min(
            mk[s]["bootstrap_topdir_stability"][read]["mean_abs_cos_to_full_topdir"] for s in others
        )
        checks[read] = {
            "seed137_sigma1_over_sigma2": gap_137,
            "min_other_seeds_sigma1_over_sigma2": gap_min_others,
            "sigma_gap_comparable": bool(
                gap_137 >= SIGMA_GAP_COMPARABILITY_FACTOR * gap_min_others
            ),
            "seed137_bootstrap_mean_abs_cos": boot_137,
            "min_other_seeds_bootstrap_mean_abs_cos": boot_min_others,
            "bootstrap_stability_comparable": bool(
                boot_137 >= boot_min_others - BOOTSTRAP_COMPARABILITY_MARGIN
            ),
        }
    failed_checks = sorted(
        f"{read}:{name}"
        for read, c in checks.items()
        for name in ("sigma_gap_comparable", "bootstrap_stability_comparable")
        if not c[name]
    )
    not_degenerate = not failed_checks
    if not_degenerate:
        verdict = (
            "seed 137's spectrum top is NOT degenerate (sigma gap and question-resampling "
            "stability comparable to seeds 42/256 under both reads) — the near-zero "
            "weighted-vs-unit-norm ordering correlation flags a real difference in which "
            "personas the unit-norm read promotes; caveat for downstream "
            "per-persona-ordering use."
        )
    else:
        # State exactly which sub-checks failed rather than asserting a blanket
        # degenerate-top conclusion; the analyzer reads per_read for magnitudes.
        verdict = (
            f"seed 137 is NOT fully comparable to seeds 42/256 — failing check(s): "
            f"{failed_checks}; see per_read for magnitudes. A clean degenerate-top "
            "signature would fail BOTH the sigma-gap and bootstrap-stability checks "
            "under a read; a single marginal sigma-gap miss with comparable bootstrap "
            "stability is a partial signature only."
        )
    return {
        "rule": (
            "comparable = (seed137 sigma1/sigma2 >= "
            f"{SIGMA_GAP_COMPARABILITY_FACTOR} * min(seeds 42,256)) AND (seed137 "
            f"bootstrap mean|cos| >= min(seeds 42,256) - {BOOTSTRAP_COMPARABILITY_MARGIN}), "
            "required under BOTH the weighted and unit-norm reads"
        ),
        "thresholds": {
            "sigma_gap_comparability_factor": SIGMA_GAP_COMPARABILITY_FACTOR,
            "bootstrap_comparability_margin": BOOTSTRAP_COMPARABILITY_MARGIN,
        },
        "per_read": checks,
        "failed_checks": failed_checks,
        "seed137_not_degenerate": bool(not_degenerate),
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#551 seed-137 top-direction degeneracy diagnostic (CPU, zero GPU)"
    )
    parser.add_argument("--local-shifts-dir", default="eval_results/issue_551/shifts")
    parser.add_argument(
        "--out",
        default=("eval_results/issue_551/seed137-topdir-degeneracy/seed137_topdir_degeneracy.json"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import importlib.metadata

    body = analyze(Path(args.local_shifts_dir))
    payload = {
        "meta": {
            "issue": 551,
            "followup_of": 521,
            "followup_label": "layer-sensitivity-and-seed137-degeneracy",
            "analysis": "seed137_topdir_degeneracy",
            "layer": 14,
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "env_versions": {pkg: importlib.metadata.version(pkg) for pkg in ("torch", "numpy")},
            "torch_used_for": "tensor loading only; all linear algebra in numpy float64",
            "tensors_source": str(args.local_shifts_dir),
            "rng": {
                "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
                "per_cell_seed_sequence": "[BOOTSTRAP_BASE_SEED, cell training seed, cell index]",
                "n_boot": N_BOOT,
            },
            "reference_rank_corr_weighted_vs_unitnorm": {
                "seed42": 0.824,
                "seed137": 0.222,
                "seed256": 0.846,
                "source": "eval_results/issue_551/controls/unitnorm_reread.json",
            },
            "source_persona": SOURCE_PERSONA,
        },
        **body,
    }
    _write_json(Path(args.out), payload)
    logger.info(
        "[phase=done] seed137_not_degenerate=%s",
        body["falsification"]["seed137_not_degenerate"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
