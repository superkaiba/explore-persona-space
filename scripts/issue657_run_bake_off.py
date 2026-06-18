#!/usr/bin/env python3
"""Issue #657 Phase 3 — the cross-behavior leakage bake-off (CPU, off-pod).

UNIFIED in-process-serial pipeline (Step 6d.0 PASS_UNIFIED): smoke IS the sweep
with one cell. ``run_behavior()`` is the per-behavior unit the full sweep calls
once per ``--behaviors``; ``--smoke`` restricts to ONE behavior x ONE LOPO fold x
ONE bootstrap replicate via the SAME ``run_behavior`` code path — no subprocess
fan-out, every phase reads its cell subset from the same ``--behaviors``/``--smoke``
flags.

Per behavior it computes (plan §5/§6):
  - DV-(a) / H1: raw Spearman rho(align, base_rate) — the #623 generalization.
  - DV-(b) / H2: held-out LOPO Spearman rho(align, leakage Delta).
  - H3 (the bar): held-out doubly-partialled rho(align, Delta | [base prior,
    prior_centroid_projection]) + grouped-CV DeltaR2_align-beyond-prior, with the
    singly-partialled rho reported alongside, all across the 3-point
    bounded-attenuation reliability band (M-Stats1).
  - controls: shuffled-direction null + the secondary continuous-DV validation
    (completion_logprob vs delta Spearman).

It writes ``eval_results/issue_657/<behavior>_alignment_predictor.json`` per behavior
+ a top-level ``summary.json`` carrying the H3 PASS / INDETERMINATE / NULL verdict
per the plan §3 v3 gate (partial-rho AND DeltaR2 CI lower bound > 0 on >=1 PRIMARY
behavior {sycophancy, refusal, marker}, EM excluded, under the doubly-partialled read
holding across the reliability band).

The sycophancy DV-(a) read is the REPLICATION ANCHOR — its raw Spearman rho on the
overlap panel must land near #623's 0.73 (the join-integrity halt; reported, not
silently swallowed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.issue657_alignment_predictor import (  # noqa: E402
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    DEFAULT_LAYERS,
    HEADLINE_LAYER,
    PRIMARY_H3_BEHAVIORS,
    SHUFFLE_B,
    _source_fe_residualize,
    attenuation_sensitivity,
    bootstrap_over_personas,
    build_predictor_frame,
    compute_alignment,
    load_behavior_direction,
    load_cell_substrate,
    load_persona_vectors,
    lopo_heldout,
    partial_spearman,
    prior_centroid_projection,
    shuffled_direction_null,
)
from explore_persona_space.experiments.persona_decomp_623 import (  # noqa: E402
    reproducibility_metadata,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #623's headline DV-(a) sycophancy anchor (the replication target, §3/§7).
SYCO_ANCHOR_RHO = 0.726
# Minimum overlap-panel bystanders for the marker LEAKAGE bake-off (plan §8/§12);
# below this, marker is reported DV-(a)-only and excluded from the H3 gate set.
MARKER_MIN_OVERLAP = 8


def _resolve(p: str) -> Path:
    return PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)


def _digest_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "missing"


def _ci_lower_gt_zero(ci: dict) -> bool:
    lo = ci.get("ci_lo")
    return lo is not None and not np.isnan(lo) and lo > 0.0


def _band_clears(band: dict, ci_lo_lookup: dict[str, dict]) -> dict:
    """Classify the reliability-band H3 outcome.

    Returns {verdict, detail}. verdict is one of:
      - CONFIRMED: the doubly-partialled partial-rho CI lower bound > 0 across the
        WHOLE band (incl. the R_low end where attenuation is largest).
      - INDETERMINATE: clears 0 only at the optimistic-reliability end (R_high).
      - NULL: never clears 0 under partialling.
    ``ci_lo_lookup`` maps band-point label -> {ci_lo, ci_hi} (bootstrap on the
    doubly-partialled rho at that reliability point); when bootstrap CIs are not
    computed at every point (smoke), the point-estimate sign is used as a fallback.
    """
    points = band.get("points", {})
    labels = ["R_low", "R_expected", "R_high"]
    present = [lbl for lbl in labels if lbl in points]
    if not present:
        return {"verdict": "NULL", "detail": "no band points"}

    def _clears(lbl: str) -> bool:
        if lbl in ci_lo_lookup:
            return _ci_lower_gt_zero(ci_lo_lookup[lbl])
        rho = points[lbl].get("doubly_partialled_rho")
        return rho is not None and not np.isnan(rho) and rho > 0.0

    all_clear = all(_clears(lbl) for lbl in present)
    opt_clear = _clears("R_high") if "R_high" in points else _clears(present[-1])
    if all_clear:
        return {"verdict": "CONFIRMED", "detail": "clears across whole band"}
    if opt_clear:
        return {"verdict": "INDETERMINATE", "detail": "clears only at optimistic-reliability end"}
    return {"verdict": "NULL", "detail": "does not clear under partialling"}


def run_behavior(
    behavior: str,
    eval_results_root: Path,
    persona_dirs: list[Path],
    direction_dir: Path,
    base_rates_da: dict[str, float] | None,
    layer: int,
    readout: str,
    centering: str,
    n_boot: int,
    shuffle_b: int,
    smoke: bool,
) -> dict:
    """Run the full per-behavior bake-off (the unified unit; smoke = 1 fold x 1 boot).

    ``base_rates_da`` is the DV-(a) base-rate map for H1 (e.g. #623 syc_i for
    sycophancy); when None, DV-(a) falls back to the #518 ``bystander_base_rate``
    (still a valid base-rate read on the same panel).
    """
    table = load_cell_substrate(behavior, eval_results_root)
    persona_vectors = load_persona_vectors(persona_dirs, DEFAULT_LAYERS)
    behavior_dir = load_behavior_direction(behavior, {behavior: direction_dir}, layer, readout)

    personas = table.bystanders
    align = compute_alignment(
        persona_vectors, behavior_dir, personas, layer, DEFAULT_LAYERS, centering
    )

    pcp = prior_centroid_projection(
        persona_vectors, table.bystander_base_rate, personas, layer, DEFAULT_LAYERS, centering
    )
    frame = build_predictor_frame(table, align, pcp)

    # ── marker leakage-overlap gate (plan §8/§12 graceful degradation) ──
    marker_leakage_excluded = False
    if behavior == "marker" and frame["n_resolvable"] < MARKER_MIN_OVERLAP:
        marker_leakage_excluded = True

    # ── DV-(a) / H1: raw Spearman rho(align, base_rate) ──
    da_base = base_rates_da if base_rates_da is not None else table.bystander_base_rate
    da_personas = [p for p in align if p in da_base and not np.isnan(align[p])]
    dv_a = {"n": len(da_personas)}
    if len(da_personas) >= 3:
        ax = np.array([align[p] for p in da_personas], dtype=float)
        ay = np.array([da_base[p] for p in da_personas], dtype=float)
        if np.std(ax) > 0 and np.std(ay) > 0:
            r = spearmanr(ax, ay)
            dv_a["raw_rho"] = float(r.correlation)
            # bootstrap over personas for the DV-(a) CI
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            boot = []
            nB = len(da_personas)
            for _ in range(n_boot):
                idx = rng.integers(0, nB, size=nB)
                rr = spearmanr(ax[idx], ay[idx]).correlation
                if rr is not None and not np.isnan(rr):
                    boot.append(float(rr))
            if boot:
                arr = np.array(boot)
                dv_a["ci_lo"] = float(np.quantile(arr, 0.025))
                dv_a["ci_hi"] = float(np.quantile(arr, 0.975))
            dv_a["base_rate_source"] = (
                "issue623_syc_i" if base_rates_da is not None else "issue518_bystander_base_rate"
            )
        else:
            dv_a["raw_rho"] = float("nan")
            dv_a["note"] = "zero-variance align or base_rate"
    else:
        dv_a["raw_rho"] = float("nan")
        dv_a["note"] = "fewer than 3 personas with both align and base_rate"

    # replication anchor (sycophancy only)
    anchor = None
    if behavior == "sycophancy":
        anchor = {
            "anchor_rho": dv_a.get("raw_rho"),
            "target_rho": SYCO_ANCHOR_RHO,
            "ci_lo": dv_a.get("ci_lo"),
            "ci_hi": dv_a.get("ci_hi"),
            # the join is consistent if the #623 target sits within the bootstrap CI
            "join_consistent": (
                dv_a.get("ci_lo") is not None
                and dv_a.get("ci_hi") is not None
                and not np.isnan(dv_a.get("ci_lo", float("nan")))
                and dv_a["ci_lo"] <= SYCO_ANCHOR_RHO <= dv_a["ci_hi"]
            ),
        }

    # ── DV-(b) / H2 + H3 (only when leakage is in scope for this behavior) ──
    leakage: dict = {"in_scope": not marker_leakage_excluded}
    if not marker_leakage_excluded and frame["cells"]:
        # raw + singly- + doubly-partialled rho (full-data point estimates)
        cells = frame["cells"]
        x = np.array([c["align"] for c in cells], dtype=float)
        y_fe = _source_fe_residualize(cells, "delta")
        prior = np.array([c["bystander_base_rate"] for c in cells], dtype=float)
        pcp_arr = np.array([c["prior_centroid_projection"] for c in cells], dtype=float)
        raw_rho, _rp, _rn = partial_spearman(x, y_fe, None)
        single_rho, _sp, _sn = partial_spearman(x, y_fe, prior)
        double_rho, _dp, _dn = partial_spearman(x, y_fe, np.column_stack([prior, pcp_arr]))
        leakage["raw_rho"] = raw_rho
        leakage["singly_partialled_rho"] = single_rho
        leakage["doubly_partialled_rho"] = double_rho

        # held-out LOPO + grouped-CV ΔR²
        lopo = lopo_heldout(frame)
        leakage["lopo"] = lopo

        # bootstrap over personas: held-out Spearman (H2) + doubly-partialled rho (H3)
        n_boot_eff = 1 if smoke else n_boot

        def _ho_spearman(fr: dict) -> float:
            return lopo_heldout(fr).get("held_out_spearman", float("nan"))

        def _double_partial(fr: dict) -> float:
            fc = fr["cells"]
            if len(fc) < 3:
                return float("nan")
            xx = np.array([c["align"] for c in fc], dtype=float)
            yy = _source_fe_residualize(fc, "delta")
            pp = np.array([c["bystander_base_rate"] for c in fc], dtype=float)
            qq = np.array([c["prior_centroid_projection"] for c in fc], dtype=float)
            rho, _p, _n = partial_spearman(xx, yy, np.column_stack([pp, qq]))
            return rho

        def _delta_r2(fr: dict) -> float:
            return lopo_heldout(fr).get("delta_r2_align_beyond_prior", float("nan"))

        leakage["h2_held_out_spearman_ci"] = bootstrap_over_personas(
            frame, _ho_spearman, b=n_boot_eff, seed=BOOTSTRAP_SEED
        )
        leakage["h3_doubly_partialled_rho_ci"] = bootstrap_over_personas(
            frame, _double_partial, b=n_boot_eff, seed=BOOTSTRAP_SEED
        )
        leakage["h3_delta_r2_ci"] = bootstrap_over_personas(
            frame, _delta_r2, b=n_boot_eff, seed=BOOTSTRAP_SEED
        )

        # reliability-sensitivity band (M-Stats1)
        band = attenuation_sensitivity(frame, table.bystander_base_rate, behavior=behavior)
        leakage["reliability_band"] = band

        # band verdict (uses the doubly-partialled CI at R_observed as the headline
        # CI; the band points use point-estimate signs as the per-point fallback —
        # a full per-point bootstrap is the production read, the headline R_observed
        # bootstrap is the load-bearing CI gate)
        ci_lookup = {"R_observed": leakage["h3_doubly_partialled_rho_ci"]}
        leakage["band_verdict"] = _band_clears(band, ci_lookup)

        # shuffled-direction null (b-vs-random); skip the heavy run in smoke
        shuffle_eff = 1 if smoke else shuffle_b
        leakage["shuffled_direction_null"] = shuffled_direction_null(
            persona_vectors,
            behavior_dir,
            table,
            table.bystander_base_rate,
            layer,
            DEFAULT_LAYERS,
            centering,
            b=shuffle_eff,
            seed=BOOTSTRAP_SEED,
        )

        # secondary continuous-DV validation: completion_logprob vs delta Spearman
        clp = [
            (c["completion_logprob"], c["delta"])
            for c in cells
            if c.get("completion_logprob") is not None
        ]
        if len(clp) >= 3:
            cx = np.array([a for a, _ in clp], dtype=float)
            cy = np.array([b for _, b in clp], dtype=float)
            if np.std(cx) > 0 and np.std(cy) > 0:
                cr = spearmanr(cx, cy)
                leakage["secondary_dv_validation"] = {
                    "completion_logprob_vs_delta_rho": float(cr.correlation),
                    "n": len(clp),
                }

    return {
        "behavior": behavior,
        "is_primary_h3": behavior in PRIMARY_H3_BEHAVIORS,
        "layer": layer,
        "readout": readout,
        "centering": centering,
        "n_cells_substrate": table.n_cells(),
        "n_cells_joined": len(frame["cells"]),
        "n_resolvable_bystanders": frame["n_resolvable"],
        "resolvable_bystanders": frame["resolvable_bystanders"],
        "dropped_bystanders": sorted({d["bystander"] for d in frame["dropped"]}),
        "marker_leakage_excluded": marker_leakage_excluded,
        "dv_a_base_rate": dv_a,
        "replication_anchor": anchor,
        "leakage_bake_off": leakage,
        "substrate_provenance": table.provenance,
        # Lean per-cell join arrays so the plot phase is self-contained (no
        # separate join-dir dependency) and the raw scatter is reproducible.
        "joined_cells": [
            {
                "source": c["source"],
                "bystander": c["bystander"],
                "delta": c["delta"],
                "align": c["align"],
                "bystander_base_rate": c["bystander_base_rate"],
                "prior_centroid_projection": c["prior_centroid_projection"],
            }
            for c in frame["cells"]
        ],
    }


def _h3_summary(per_behavior: dict[str, dict]) -> dict:
    """Compute the top-level H3 verdict across the PRIMARY behaviors (EM excluded)."""
    primary_clears: list[str] = []
    detail: dict[str, dict] = {}
    for beh, res in per_behavior.items():
        if beh not in PRIMARY_H3_BEHAVIORS:
            continue
        lk = res.get("leakage_bake_off", {})
        if not lk.get("in_scope", False):
            detail[beh] = {"status": "leakage_out_of_scope"}
            continue
        rho_ci = lk.get("h3_doubly_partialled_rho_ci", {})
        r2_ci = lk.get("h3_delta_r2_ci", {})
        band_verdict = lk.get("band_verdict", {}).get("verdict", "NULL")
        rho_clears = _ci_lower_gt_zero(rho_ci)
        r2_clears = _ci_lower_gt_zero(r2_ci)
        # PASS requires BOTH the doubly-partialled rho AND the ΔR² CI lower bound > 0
        # AND the band CONFIRMED (not just INDETERMINATE).
        passes = rho_clears and r2_clears and band_verdict == "CONFIRMED"
        detail[beh] = {
            "doubly_partialled_rho_ci_lo": rho_ci.get("ci_lo"),
            "delta_r2_ci_lo": r2_ci.get("ci_lo"),
            "band_verdict": band_verdict,
            "passes_h3": passes,
        }
        if passes:
            primary_clears.append(beh)

    if primary_clears:
        verdict = "PASS"
    elif any(d.get("band_verdict") == "INDETERMINATE" for d in detail.values()):
        verdict = "INDETERMINATE"
    else:
        verdict = "NULL"
    return {
        "h3_verdict": verdict,
        "primary_behaviors_cleared": primary_clears,
        "primary_h3_behaviors": list(PRIMARY_H3_BEHAVIORS),
        "per_behavior_detail": detail,
        "note": (
            "PASS = doubly-partialled rho CI lo > 0 AND ΔR² CI lo > 0 AND reliability "
            "band CONFIRMED on >=1 primary behavior {sycophancy, refusal, marker}; "
            "EM excluded (M-Alts2). A power-limited / optimistic-end-only clearance "
            "is INDETERMINATE, not a confirmed beat-the-prior."
        ),
    }


def _load_da_base_rates(behavior: str, syc_i_path: Path) -> dict[str, float] | None:
    """DV-(a) base-rate map for H1. sycophancy -> #623 syc_i; else None (use #518)."""
    if behavior != "sycophancy" or not syc_i_path.exists():
        return None
    doc = json.loads(syc_i_path.read_text())
    return {p: float(v["syc_i"]) for p, v in doc["syc_i"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #657 Phase 3 — leakage bake-off (CPU).")
    parser.add_argument(
        "--behavior",
        action="append",
        default=None,
        help="Behavior to run (repeatable). Default: all four (sycophancy refusal marker em).",
    )
    parser.add_argument(
        "--eval-results-root",
        default="eval_results",
        help="Root holding issue_518/ + issue_605/ (relative to repo root).",
    )
    parser.add_argument("--persona-dir", action="append", default=None)
    parser.add_argument(
        "--direction-dir", default=None, help="Override direction dir (1 behavior)."
    )
    parser.add_argument(
        "--syc-i",
        default="eval_results/issue_623/syc_i.json",
        help="#623 syc_i.json for the sycophancy DV-(a) read (relative to repo root).",
    )
    parser.add_argument("--layer", type=int, default=HEADLINE_LAYER)
    parser.add_argument("--readout", default="last_token", choices=["last_token", "response_avg"])
    parser.add_argument("--centering", default="global_mean", choices=["global_mean", "none"])
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_B)
    parser.add_argument("--shuffle-b", type=int, default=SHUFFLE_B)
    parser.add_argument(
        "--out-dir",
        default="eval_results/issue_657",
        help="Output dir for per-behavior JSONs + summary.json (relative to repo root).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke: 1 LOPO fold x 1 bootstrap replicate (unified one-cell path).",
    )
    args = parser.parse_args()

    load_dotenv()

    behaviors = args.behavior or ["sycophancy", "refusal", "marker", "em"]
    eval_root = _resolve(args.eval_results_root)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    syc_i_path = _resolve(args.syc_i)

    method_subdir = "method_a" if args.readout == "last_token" else "method_b"
    default_persona_dirs = [
        _resolve(f"data/issue657/persona_vectors/{method_subdir}"),
        _resolve(f"data/persona_vectors/issue623/{method_subdir}"),
    ]
    persona_dirs = (
        [_resolve(p) for p in args.persona_dir] if args.persona_dir else default_persona_dirs
    )

    per_behavior: dict[str, dict] = {}
    for beh in behaviors:
        if args.direction_dir:
            direction_dir = _resolve(args.direction_dir)
        elif beh == "sycophancy":
            direction_dir = _resolve("data/persona_vectors/issue623/sycophancy_trait")
        else:
            direction_dir = _resolve(f"data/issue657/behavior_directions/{beh}")

        base_rates_da = _load_da_base_rates(beh, syc_i_path)
        print(f"[phase=bake_off] behavior={beh} smoke={args.smoke}", flush=True)
        res = run_behavior(
            beh,
            eval_root,
            persona_dirs,
            direction_dir,
            base_rates_da,
            args.layer,
            args.readout,
            args.centering,
            args.n_boot,
            args.shuffle_b,
            args.smoke,
        )
        # checkpoint-per-phase: write each behavior the moment it completes
        meta = reproducibility_metadata({"behavior": beh, "smoke": args.smoke})
        res["metadata"] = meta
        beh_path = out_dir / f"{beh}_alignment_predictor.json"
        beh_path.write_text(json.dumps(res, indent=2))
        per_behavior[beh] = res
        dv_a_rho = res["dv_a_base_rate"].get("raw_rho")
        lk = res["leakage_bake_off"]
        dp = lk.get("doubly_partialled_rho")
        print(
            f"[phase=bake_off] behavior={beh} dv_a_rho={dv_a_rho} "
            f"doubly_partialled_rho={dp} in_scope={lk.get('in_scope')} "
            f"md5={_digest_file(beh_path)} -> {beh_path}",
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "metadata": reproducibility_metadata({"behaviors": behaviors, "smoke": args.smoke}),
        "h3": _h3_summary(per_behavior),
        "anchors": {
            beh: per_behavior[beh].get("replication_anchor")
            for beh in per_behavior
            if per_behavior[beh].get("replication_anchor") is not None
        },
        "per_behavior_paths": {
            beh: str(out_dir / f"{beh}_alignment_predictor.json") for beh in per_behavior
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        f"[phase=bake_off] H3 verdict={summary['h3']['h3_verdict']} "
        f"cleared={summary['h3']['primary_behaviors_cleared']} "
        f"-> {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
