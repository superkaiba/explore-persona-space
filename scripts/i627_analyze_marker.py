#!/usr/bin/env python3
"""Task #627 Phase 3 — marker-family re-analysis over the committed #601 slab.

Per cell x checkpoint (ON-POLICY trajectory.json reads):
  - source install in ALL THREE spaces (EOS-margin Δ(z_marker - z_eos) /
    Δlog P / probability — every marker table carries all three; CLAUDE.md
    marker storage contract);
  - bystander-mean leakage per space;
  - leakage fraction = bystander Δmargin / source Δmargin with the registered
    denominator floor source-Δmargin >= 2.0 (sensitivity 1.0 / 4.0); a
    log-prob-space fraction companion (same floor in nats) shows the softmax
    compression the margin space avoids.

H2 (plan §3): contrastive-mix vs positives-only bystander-mean fraction
difference at the Phase-0 matched-install pairs (margin space), persona-
cluster bootstrap CI (resample the bystander panel). Statistical hygiene: the
fraction is NEVER correlated against install (plan §6, binding).

Teacher-forced dense_trajectory.json reads feed WITHIN-CONDITION dose-curve
shape only — never cross-condition level comparison (#432→#456 fence; the
records are tagged read_type=teacher_forced and kept out of H2 entirely).

Scope caveat carried on every H2 read: #601's negative rows were
gradient-dead (loss-suppression flag off) — the contrast is mix COMPOSITION
under flag-off loss placement, not live-contrastive training.

Output: eval_results/issue_627/analysis/marker_fractions_601.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.leakage_vs_install_627 import (  # noqa: E402
    MARGIN_DENOM_FLOOR,
    MARGIN_DENOM_FLOOR_SENSITIVITY,
)
from explore_persona_space.experiments.leakage_vs_install_627.marker601 import (  # noqa: E402
    DEFAULT_601_ROOT,
    load_all_dense,
    load_all_onpolicy,
)

log = logging.getLogger("i627_analyze_marker")

OUT_DIR = Path("eval_results/issue_627/analysis")
PAIRS_MANIFEST = Path("eval_results/issue_627/marker_matched_pairs.json")
TOLERANCE_MANIFEST = Path("eval_results/issue_627/marker_tolerance.json")
H2_EQUIVALENCE = 0.05  # plan §3 H2: fraction difference <= 0.05
H2_CI_BAND = 0.10  # plan §3 H2: CI inside ±0.10
BOOTSTRAP_N = 10_000
RNG_SEED = 42
SCOPE_CAVEAT = (
    "#601 negative rows were gradient-dead (loss-suppression flag off): the contrast is "
    "mix composition under flag-off loss placement, NOT live-contrastive training"
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _bystander_means(ck: dict) -> dict[str, float]:
    """Bystander-mean leakage per space at one checkpoint."""
    out = {}
    for space in ("margin", "dlogp", "dprob"):
        out[space] = float(np.mean([b[space] for b in ck["bystanders"].values()]))
    return out


def _fraction(numer: float, denom: float, floor: float) -> float | None:
    """Registered fraction with denominator floor; None = below floor (excluded,
    never zeroed)."""
    if denom < floor:
        return None
    return numer / denom


def three_space_tables(cells: list[dict]) -> list[dict]:
    """Per cell x checkpoint three-space install / leakage / fraction table."""
    rows = []
    for c in cells:
        for ck in c["checkpoints"]:
            bys = _bystander_means(ck)
            src = ck["source"]
            row = {
                "cell": c["cell"],
                "seed": c["seed"],
                "mix_arm": c["mix_arm"],
                "read_type": c["read_type"],
                "frac": ck["frac"],
                "step": ck["step"],
                "n_bystanders": len(ck["bystanders"]),
                "source_install": {k: src[k] for k in ("margin", "dlogp", "dprob")},
                "bystander_mean_leakage": bys,
                # Registered fraction: margin space, floor 2.0 (+ sensitivity).
                "fraction_margin": _fraction(bys["margin"], src["margin"], MARGIN_DENOM_FLOOR),
                "fraction_margin_sensitivity": {
                    str(f): _fraction(bys["margin"], src["margin"], f)
                    for f in MARGIN_DENOM_FLOOR_SENSITIVITY
                },
                # Companion: log-prob-space fraction (same floor value in nats)
                # — shows the softmax compression of the denominator.
                "fraction_dlogp_companion": _fraction(
                    bys["dlogp"], src["dlogp"], MARGIN_DENOM_FLOOR
                ),
                "denominator_floor": MARGIN_DENOM_FLOOR,
            }
            rows.append(row)
    return rows


def _pair_fraction_diffs(
    pair: dict, by_cell: dict[tuple[str, int], dict], floor: float
) -> dict | None:
    """One matched pair -> per-bystander fraction difference (contrastive -
    posonly) in margin space, or None when either side is below the floor."""
    c = by_cell[(pair["contrastive_cell"], pair["seed"])]
    p = by_cell[(pair["posonly_cell"], pair["seed"])]
    ck_c = next(k for k in c["checkpoints"] if k["frac"] == pair["contrastive_frac"])
    ck_p = next(k for k in p["checkpoints"] if k["frac"] == pair["posonly_frac"])
    if ck_c["source"]["margin"] < floor or ck_p["source"]["margin"] < floor:
        return None
    panel = sorted(set(ck_c["bystanders"]) & set(ck_p["bystanders"]))
    if panel != c["held_out_personas"]:
        raise RuntimeError(
            f"pair {pair['contrastive_cell']}/{pair['posonly_cell']}: panel mismatch"
        )
    per_bys = {
        b: (
            ck_c["bystanders"][b]["margin"] / ck_c["source"]["margin"]
            - ck_p["bystanders"][b]["margin"] / ck_p["source"]["margin"]
        )
        for b in panel
    }
    return {
        "pair": pair,
        "panel": panel,
        "per_bystander_fraction_diff": per_bys,
        "contrastive_fraction": float(
            np.mean([ck_c["bystanders"][b]["margin"] for b in panel]) / ck_c["source"]["margin"]
        ),
        "posonly_fraction": float(
            np.mean([ck_p["bystanders"][b]["margin"] for b in panel]) / ck_p["source"]["margin"]
        ),
    }


def h2_matched_fraction_contrast(cells: list[dict], pairs: list[dict], floor: float) -> dict:
    """H2: bystander-mean fraction difference (contrastive - posonly) over the
    Phase-0 matched pairs; persona-cluster bootstrap CI (bystander panel is
    the cluster unit). Pairs grouped by panel size; the PRIMARY group is the
    47-bystander panel (plan §2: the phase1 + multiepoch-closure grids)."""
    by_cell = {(c["cell"], c["seed"]): c for c in cells}
    evaluated = []
    n_below_floor = 0
    for pair in pairs:
        r = _pair_fraction_diffs(pair, by_cell, floor)
        if r is None:
            n_below_floor += 1
        else:
            evaluated.append(r)
    if not evaluated:
        return {
            "verdict": "no_pairs_above_floor",
            "n_pairs": len(pairs),
            "n_below_floor": n_below_floor,
            "scope_caveat": SCOPE_CAVEAT,
        }
    groups: dict[int, list[dict]] = {}
    for r in evaluated:
        groups.setdefault(len(r["panel"]), []).append(r)
    primary_n = max(groups)  # 47-bystander panel when present
    out_groups = {}
    rng = np.random.default_rng(RNG_SEED)
    for n_bys, rows in sorted(groups.items()):
        panel = rows[0]["panel"]
        # Grouping is by panel SIZE — guard the composition assumption
        # explicitly: every pair in a size group must share the IDENTICAL
        # bystander panel, or the diffs matrix below would mix personas.
        mismatched = [r["pair"] for r in rows if r["panel"] != panel]
        if mismatched:
            raise RuntimeError(
                f"H2 panel-size group {n_bys}: {len(mismatched)} pair(s) carry a different "
                f"bystander composition than the group's first pair (e.g. "
                f"{mismatched[0]['contrastive_cell']}/{mismatched[0]['posonly_cell']}) — "
                "same-size groups must share one identical panel; split the grouping key "
                "by panel composition before re-running"
            )
        # diffs matrix: (n_pairs, n_bys)
        diffs = np.array([[r["per_bystander_fraction_diff"][b] for b in panel] for r in rows])
        plugin = float(diffs.mean())
        picks = rng.integers(0, n_bys, size=(BOOTSTRAP_N, n_bys))
        rep = diffs[:, picks].mean(axis=(0, 2))  # persona-cluster resample
        ci = (float(np.quantile(rep, 0.025)), float(np.quantile(rep, 0.975)))
        inside = ci[0] > -H2_CI_BAND and ci[1] < H2_CI_BAND
        out_groups[str(n_bys)] = {
            "n_pairs": len(rows),
            "panel_size": n_bys,
            "mean_fraction_diff_contrastive_minus_posonly": plugin,
            "ci95_persona_cluster": list(ci),
            "abs_diff_le_0p05": bool(abs(plugin) <= H2_EQUIVALENCE),
            "ci_inside_pm0p10": bool(inside),
            "ci_excludes_zero": bool(ci[0] > 0.0 or ci[1] < 0.0),
            "pairs": [
                {
                    "contrastive": (
                        f"{r['pair']['contrastive_cell']}@frac{r['pair']['contrastive_frac']}"
                    ),
                    "posonly": f"{r['pair']['posonly_cell']}@frac{r['pair']['posonly_frac']}",
                    "seed": r["pair"]["seed"],
                    "contrastive_fraction": r["contrastive_fraction"],
                    "posonly_fraction": r["posonly_fraction"],
                    "install_gap_margin": r["pair"]["install_gap_margin"],
                }
                for r in rows
            ],
        }
    primary = out_groups[str(primary_n)]
    # Verdict branches, §13.9 precedence FIRST (round-2 fix): a CI that
    # excludes 0 while sitting inside ±0.10 is the graded middle zone — never
    # forced into the constant-fraction confirm (or falsify) branch.
    if primary["ci_excludes_zero"] and primary["ci_inside_pm0p10"]:
        verdict = "graded_middle_zone"
    elif primary["abs_diff_le_0p05"] and primary["ci_inside_pm0p10"]:
        verdict = "constant_fraction_holds"
    elif (
        not primary["ci_inside_pm0p10"]
        and abs(primary["mean_fraction_diff_contrastive_minus_posonly"]) > H2_CI_BAND
    ):
        verdict = "falsified_fraction_differs"
    else:
        verdict = "graded_middle_zone"
    return {
        "verdict": verdict,
        "verdict_rule": "plan §13.9: a CI excluding 0 while inside ±0.10 is graded — never "
        "forced confirm/falsify; constant_fraction_holds requires |diff| <= 0.05 AND CI "
        "inside ±0.10 AND CI covering 0",
        "primary_panel_size": primary_n,
        "groups": out_groups,
        "n_pairs_total": len(pairs),
        "n_below_floor": n_below_floor,
        "denominator_floor": floor,
        "scope_caveat": SCOPE_CAVEAT,
        "hygiene": "fractions compared at matched install only; never correlated against install",
    }


def dose_curves(cells: list[dict]) -> list[dict]:
    """(install, bystander-mean leakage) points per cell for the dose-curve
    figures; teacher-forced cells carry read_type for the within-condition
    fence."""
    out = []
    for c in cells:
        pts = [
            {
                "frac": ck["frac"],
                "step": ck["step"],
                "install_margin": ck["source"]["margin"],
                "install_dlogp": ck["source"]["dlogp"],
                "leak_margin": _bystander_means(ck)["margin"],
                "leak_dlogp": _bystander_means(ck)["dlogp"],
            }
            for ck in c["checkpoints"]
        ]
        out.append(
            {
                "cell": c["cell"],
                "seed": c["seed"],
                "mix_arm": c["mix_arm"],
                "read_type": c["read_type"],
                "points": pts,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #627 Phase 3 — #601 marker fraction re-analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root-601", type=Path, default=DEFAULT_601_ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with open(PAIRS_MANIFEST) as f:
        pairs_manifest = json.load(f)
    with open(TOLERANCE_MANIFEST) as f:
        tolerance = json.load(f)

    log.info("[phase=p3_marker] loading #601 slab")
    onpolicy = load_all_onpolicy(args.root_601)
    dense = load_all_dense(args.root_601)

    tables = three_space_tables(onpolicy)
    h2 = {
        "registered_floor_2p0": h2_matched_fraction_contrast(
            onpolicy, pairs_manifest["pairs"], MARGIN_DENOM_FLOOR
        ),
        "sensitivity": {
            str(f): h2_matched_fraction_contrast(onpolicy, pairs_manifest["pairs"], f)
            for f in MARGIN_DENOM_FLOOR_SENSITIVITY
        },
    }
    result = {
        "issue": 627,
        "family": "marker_601",
        "tolerance_manifest": tolerance["formula"],
        "tolerance_margin": tolerance["tolerance_margin"],
        "three_space_tables_onpolicy": tables,
        "h2_matched_fraction_contrast": h2,
        "dose_curves_onpolicy": dose_curves(onpolicy),
        "dose_curves_teacher_forced_within_condition_only": dose_curves(dense),
        "scope_caveats": [
            SCOPE_CAVEAT,
            "source probability-space reads computed from stored mean log-probs "
            "(geometric-mean probability) — sanity read only",
            "teacher-forced dense reads are within-condition dose-curve shape only",
        ],
        "metadata": {
            "git_commit_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "numpy_version": np.__version__,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "marker_fractions_601.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(
        "[phase=p3_marker] -> %s (h2 verdict: %s; %d on-policy cells, %d dense cells)",
        out_path,
        h2["registered_floor_2p0"]["verdict"],
        len(onpolicy),
        len(dense),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
