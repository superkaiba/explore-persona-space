# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ρ/ΔG + × + − intentional
#!/usr/bin/env python3
"""Task #555 — per-replicate 6-predictor partial-Spearman fits + pre-registered verdict.

Forked from scripts/i534_trajectory_analyze.py (consumes the same per-cell
`trajectory.json` shape via the `contrastive_neg_geometry_504.analyze` fit
machinery). Deltas (plan #555 §4.3 c):

  1. The loop unit is the REPLICATE (a fresh seed pair), not the fraction:
     `--replicates "7:11,19:23,71:73,101:103,211:223"`. Per replicate, the
     parent's identical fit pools 432 rows = 54 held-out probes (DV =
     per-probe mean over the 10 framings) × 4 positioned arms × 2 seeds at
     the single frac-1.00 (step-5) checkpoint. n == 432 is ASSERTED.
  2. Identical 6-predictor partial Spearman per replicate; family-5 Holm
     (excluding `training_step` — zero-variance at a single read point,
     retained as a partialled covariate and flagged exactly as the parent
     flags it) promoted to PRIMARY; 1000-resample bootstrap CIs per
     predictor (`--boot-seed 555`).
  3. Usability gates computed but DESCRIPTIVE-ONLY — the run sits below the
     1-nat source floor BY DESIGN (that IS the experiment); the fits are
     never routed out as "unusable".
  4. Cross-replicate aggregation per predictor: the 5 ρ values, sign counts,
     Holm-5 significance counts, mean ρ with the pre-specified 95% t-interval
     (df = n_replicates − 1), plus a descriptive pooled 2160-row fit.
  5. The PINNED 3-tier machine verdict on `d_nearest_neg_nd` (decision rule
     registered in the task body):
       FALSIFIED  if n_positive >= sign-threshold OR the t-interval excludes 0;
       STANDS     only if holm_sig_count <= 1 AND n_positive < threshold AND
                  the interval includes 0;
       INDETERMINATE otherwise.
     Sign thresholds prorate per §9 descope: {5: 4, 4: 4, 3: 3}. When
     FALSIFIED fires on the sign trigger ALONE (interval spanning zero), the
     exact one-sided null base rate of that trigger (~0.19 for >=4/5) is
     emitted alongside — the machine verdict itself is never softened.
  6. Parent reference values (the +0.110 step-5 nearest-negative reading
     under calibration) are MACHINE-READ from
     `eval_results/issue_534/analysis_per_fraction.json` at frac 0.25 (the
     parent's realized step-5 read point: steps 5 of stop-20), never
     hand-typed.
  7. `_bandctrl` exclusion: the positive-control cell's slab dir
     (`c504v3_near_seed7_bandctrl`) is EXCLUDED from every production glob —
     `build_rows` matches `<cell>_seed<seed>` exactly (cannot pick it up),
     and the manifest-flag aggregation here filters `*_bandctrl` dirs
     explicitly (consistency-checker note 1).

CPU-only; runs OFF-POD on the VM against the committed eval_results/issue_555/
JSONs AFTER pod termination:
    uv run python scripts/i555_replicate_analyze.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i555.replicate_analyze")

VERDICT_PREDICTOR = "d_nearest_neg_nd"
SPECIFICITY_PREDICTOR = "shadow_angle"
DEFAULT_REPLICATES = "7:11,19:23,71:73,101:103,211:223"
EXPECTED_ROWS_PER_REPLICATE = 432  # 54 probes × 4 arms × 2 seeds
# §9 prorated sign-consistency thresholds (n_replicates -> n_positive trigger).
SIGN_TRIGGER_BY_N: dict[int, int] = {5: 4, 4: 4, 3: 3}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _synthesize_phase0_calibration(chosen_frac: float) -> dict[str, Any]:
    """In-memory phase0 calibration dict (task-555 provenance, plan-pinned constant)."""
    return {
        "verdict": "pass",
        "chosen_checkpoint_fraction": float(chosen_frac),
        "source": "i555_replicate_analyze._synthesize_phase0_calibration",
        "task_id_minted_by": 555,
        "note": (
            "Synthesized routing constant (plan #555 §4.3 c) — frac 1.00 of the "
            "step-5 hard stop = step 5 exact; NOT an evidence-based pick."
        ),
        "synthesized_at": datetime.now(UTC).isoformat(),
    }


def parse_replicates(spec: str) -> list[tuple[int, int]]:
    """Parse "7:11,19:23,..." into [(7, 11), (19, 23), ...] (fail loud on shape)."""
    pairs: list[tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"--replicates token {token!r} is not 'seedA:seedB'")
        pairs.append((int(parts[0]), int(parts[1])))
    if not pairs:
        raise ValueError(f"--replicates {spec!r} parsed to zero pairs")
    flat = [s for p in pairs for s in p]
    if len(set(flat)) != len(flat):
        raise ValueError(f"--replicates {spec!r} has duplicate seeds across pairs")
    return pairs


def t_interval(values: list[float], alpha: float = 0.05) -> dict[str, Any]:
    """Mean + two-sided (1−alpha) t-interval over per-replicate ρ (df = n−1)."""
    from scipy import stats

    n = len(values)
    if n < 2:
        return {"mean": (float(values[0]) if values else None), "lo": None, "hi": None, "n": n}
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(n))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return {
        "mean": mean,
        "lo": mean - tcrit * se,
        "hi": mean + tcrit * se,
        "se": se,
        "t_crit": tcrit,
        "df": n - 1,
        "alpha": alpha,
        "n": n,
    }


def sign_trigger_null_base_rate(n: int, threshold: int) -> float:
    """Exact one-sided P(X >= threshold) for X ~ Binomial(n, 0.5) (sign-only trigger)."""
    return float(sum(math.comb(n, k) for k in range(threshold, n + 1)) / 2**n)


def decide_verdict(
    *,
    rhos_nn: list[float],
    holm_sig_count: int,
    interval: dict[str, Any],
    n_replicates: int,
) -> dict[str, Any]:
    """The PINNED 3-tier machine verdict (registered rule; never softened post hoc)."""
    if n_replicates not in SIGN_TRIGGER_BY_N:
        raise ValueError(
            f"n_replicates={n_replicates} outside the registered ladder "
            f"{sorted(SIGN_TRIGGER_BY_N)} (plan §9: min 3 replicates)."
        )
    threshold = SIGN_TRIGGER_BY_N[n_replicates]
    n_positive = sum(1 for r in rhos_nn if r > 0)
    interval_excludes_zero = (
        interval["lo"] is not None
        and interval["hi"] is not None
        and (interval["lo"] > 0 or interval["hi"] < 0)
    )
    sign_trigger = n_positive >= threshold
    if sign_trigger or interval_excludes_zero:
        verdict = "FALSIFIED"
    elif holm_sig_count <= 1 and n_positive < threshold and not interval_excludes_zero:
        verdict = "STANDS"
    else:
        verdict = "INDETERMINATE"
    out: dict[str, Any] = {
        "verdict": verdict,
        "predictor": VERDICT_PREDICTOR,
        "n_replicates": n_replicates,
        "n_positive": n_positive,
        "sign_trigger_threshold": threshold,
        "sign_trigger_fired": bool(sign_trigger),
        "holm5_significant_count": holm_sig_count,
        "t_interval": interval,
        "t_interval_excludes_zero": bool(interval_excludes_zero),
        "rule": (
            "FALSIFIED if n_positive >= threshold OR t-interval excludes 0; "
            "STANDS only if holm_sig_count <= 1 AND n_positive < threshold AND "
            "interval includes 0; else INDETERMINATE. Registered in the task "
            "body; the OR-rule fires on sign consistency alone."
        ),
    }
    if verdict == "FALSIFIED" and sign_trigger and not interval_excludes_zero:
        base_rate = sign_trigger_null_base_rate(n_replicates, threshold)
        out["sign_only_trigger"] = True
        out["sign_only_trigger_null_base_rate"] = base_rate
        out["sign_only_trigger_caveat"] = (
            f"FALSIFIED fired on the sign trigger ALONE (>= {threshold}/"
            f"{n_replicates} positive) while the pooled t-interval spans zero. "
            f"Under the null this trigger fires by chance with probability "
            f"~{base_rate:.3f} — report this base rate alongside the verdict "
            "(the machine verdict itself is not softened)."
        )
    else:
        out["sign_only_trigger"] = False
    return out


def collect_manifest_flags_555(slab_root: Path) -> dict[str, Any]:
    """Aggregate the fraction manifests' gauge / stop flags — `_bandctrl` EXCLUDED."""
    flags: dict[str, Any] = {
        "per_cell": {},
        "all_logit_readout_valid": True,
        "excluded_bandctrl_dirs": [],
    }
    for p in sorted(slab_root.glob("c504v3_*_seed*/fraction_manifest.json")):
        key = p.parent.name
        if key.endswith("_bandctrl"):
            # Positive-control cell: eval-path validation only, never a
            # production input (consistency-checker note 1).
            flags["excluded_bandctrl_dirs"].append(key)
            continue
        m = json.loads(p.read_text())
        flags["per_cell"][key] = {
            "logit_readout_valid": m.get("logit_readout_valid"),
            "stopped": m.get("stopped"),
            "stop_reason": m.get("stop_reason"),
            "stop_step": (m.get("band_stop_meta") or {}).get("stop_step"),
            "distinct_steps": m.get("distinct_steps"),
            "source_delta_g_at_selected_steps": m.get("source_delta_g_at_selected_steps"),
        }
        if not m.get("logit_readout_valid", False):
            flags["all_logit_readout_valid"] = False
    return flags


def load_parent_reference(path: Path) -> dict[str, Any]:
    """Machine-read the parent's step-5 (frac 0.25) partial-ρ reference values.

    The parent's frac-0.25 read point IS step 5 (steps 5 of the realized
    stop-20 in all 10 parent cells) — the object under calibration. Never
    hand-typed (plan §4.3 c item 5).
    """
    if not path.exists():
        return {"available": False, "path": str(path)}
    payload = json.loads(path.read_text())
    frac = payload.get("per_fraction", {}).get("0.25")
    if frac is None:
        return {"available": False, "path": str(path), "reason": "frac 0.25 missing"}
    ps = frac.get("pooled_fit", {}).get("partial_spearman", {})
    out = {
        "available": True,
        "path": str(path),
        "parent_frac": "0.25",
        "parent_read_point": "step 5 of the realized stop-20 (the reading under calibration)",
        "partial_spearman": {
            p: {"rho": float(v["rho"]), "p_raw": float(v["p_raw"])}
            for p, v in ps.items()
            if v is not None
        },
        "n_rows": frac.get("n_rows"),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_555"))
    ap.add_argument(
        "--phase05-path",
        type=Path,
        default=Path("eval_results/issue_530/phase0_5_gates.json"),
        help="#530's committed Phase 0.5 geometry artifact (reused as-is, plan §10).",
    )
    ap.add_argument(
        "--parent-analysis",
        type=Path,
        default=Path("eval_results/issue_534/analysis_per_fraction.json"),
        help="#534's committed per-fraction analysis — the step-5 reference under calibration.",
    )
    ap.add_argument(
        "--replicates",
        default=DEFAULT_REPLICATES,
        help=f"Seed pairs 'a:b,c:d,...' (default {DEFAULT_REPLICATES!r}; plan §4.1).",
    )
    ap.add_argument("--frac", type=float, default=1.0, help="Checkpoint fraction (default 1.0).")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=555)
    ap.add_argument(
        "--expected-rows",
        type=int,
        default=EXPECTED_ROWS_PER_REPLICATE,
        help=f"Asserted per-replicate row count (default {EXPECTED_ROWS_PER_REPLICATE}).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default <slab-root>/analysis_replicates.json).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=analyze_555] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        POSITIONED_ARM_SLUGS_V3,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
        PREDICTORS,
        aggregate_base_prior_from_trajectories,
        build_rows,
        fit_pooled_partial_spearman,
        run_phase2_analysis,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase05 import (
        load_phase05,
    )

    # The parent's family-5 Holm + bootstrap + z-agreement machinery, reused
    # from the i534 fork (same trajectory.json shape, same estimators).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from i534_trajectory_analyze import (
        bootstrap_partial_rho,
        family5_holm,
        usability_for_fraction,
        z_agreement_for_fraction,
    )

    slab: Path = args.slab_root
    replicates = parse_replicates(args.replicates)
    out_path = args.out if args.out is not None else slab / "analysis_replicates.json"

    gates = load_phase05(args.phase05_path)
    per_probe = gates["per_probe"]
    arm_to_positioned_n = gates["arm_to_positioned_n"]

    notes: list[str] = []
    per_replicate_out: dict[str, Any] = {}
    rows_all: list[dict] = []
    rho_by_predictor: dict[str, list[float]] = {p: [] for p in PREDICTORS}
    holm5_sig_by_predictor: dict[str, int] = {p: 0 for p in PREDICTORS}

    for rep_idx, (seed_a, seed_b) in enumerate(replicates, start=1):
        rep_key = f"R{rep_idx}_seeds{seed_a}_{seed_b}"
        seeds = [seed_a, seed_b]
        log.info("[phase=analyze_555_%s] per-replicate fit (dg_band=None)", rep_key)

        # Base-prior covariate aggregated from THIS replicate's trajectories
        # (procedure-identical to the parent's per-run aggregation; the base
        # model is frozen so the per-probe quantity is seed-independent).
        base_prior = aggregate_base_prior_from_trajectories(
            slab_root=slab, seeds=seeds, positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3
        )
        if not base_prior:
            notes.append(f"{rep_key}: empty base-prior aggregation — covariate falls back to 0.0.")
            base_prior = None

        summary = run_phase2_analysis(
            slab_root=slab,
            phase0_calibration=_synthesize_phase0_calibration(args.frac),
            phase05_gates=gates,
            seeds=seeds,
            base_prior_by_probe=base_prior,
            positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
            # The cells sit below the [5, 12]-nat band BY DESIGN (no-implant
            # snapshots); the band exclusion would drop every cell.
            dg_band=None,
        )
        pooled = build_rows(
            slab_root=slab,
            chosen_frac=args.frac,
            per_probe=per_probe,
            arm_to_positioned_n=arm_to_positioned_n,
            seeds=seeds,
            base_prior_by_probe=base_prior,
            positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
            dg_band=None,
        )
        rows = pooled["rows"]
        if len(rows) != args.expected_rows:
            raise RuntimeError(
                f"{rep_key}: pooled {len(rows)} rows, expected {args.expected_rows} "
                f"(54 probes × 4 arms × 2 seeds). Excluded: {pooled['excluded_cells']}. "
                "A missing cell/probe breaks comparability with the parent's 432-row fits."
            )
        rows_all.extend(rows)

        fit = summary["pooled_fit"]
        fam5 = family5_holm(fit.get("partial_spearman", {}))

        # Zero-variance training_step flag (single read point — expected).
        steps = sorted({r["training_step"] for r in rows})
        zero_var_step = len(steps) <= 1
        if zero_var_step:
            notes.append(
                f"{rep_key}: training_step has zero variance (all rows at step "
                f"{steps[0] if steps else 'n/a'}) — retained as a partialled covariate, "
                "flagged exactly as the parent's per-fraction fits do; family-5 Holm "
                "(excluding it) is the PRIMARY correction here."
            )

        boot = {
            p: bootstrap_partial_rho(
                rows, p, n_boot=args.n_boot, seed=args.boot_seed + 100 * rep_idx
            )
            for p in PREDICTORS
        }

        # Usability gates: DESCRIPTIVE-ONLY (plan §4.3 c item 3) — the run is
        # below the 1-nat source floor BY DESIGN; nothing is routed out.
        usability = usability_for_fraction(
            slab_root=slab,
            frac=args.frac,
            seeds=seeds,
            positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
        )
        usability["descriptive_only"] = True
        usability["note"] = (
            "Sub-floor source ΔG is the DESIGN here (no-implant snapshots); the "
            "gate values are reported, never used to exclude fits."
        )

        for p in PREDICTORS:
            part = fit.get("partial_spearman", {}).get(p)
            if part is not None and part.get("rho") is not None:
                rho_by_predictor[p].append(float(part["rho"]))
            if fam5.get(p, {}).get("reject_null", False):
                holm5_sig_by_predictor[p] += 1

        rep_payload = {
            "replicate": rep_key,
            "seeds": seeds,
            "n_rows": len(rows),
            "pooled_fit": fit,
            "per_seed_fit": summary["per_seed_fit"],
            "sign_agreement": summary["sign_agreement"],
            "per_cell_diagnostics": summary["per_cell_diagnostics"],
            "family5_holm_primary": fam5,
            "bootstrap_ci": boot,
            "usability_descriptive": usability,
            "zero_variance_training_step": bool(zero_var_step),
            "distinct_training_steps_in_pool": steps,
            "z_agreement": z_agreement_for_fraction(
                slab_root=slab,
                frac=args.frac,
                seeds=seeds,
                positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
            ),
        }
        # Persist the FULL per-replicate payload the moment it completes
        # (checkpoint-per-phase rule).
        side_path = slab / f"analysis_replicate_{rep_key}.json"
        side_path.write_text(json.dumps(rep_payload, indent=2, default=str))
        rep_payload["analysis_path"] = str(side_path)
        per_replicate_out[rep_key] = rep_payload
        log.info(
            "[phase=analyze_555_%s] ρ_nn=%+.4f, holm5_sig=%s — wrote %s",
            rep_key,
            float(fit["partial_spearman"][VERDICT_PREDICTOR]["rho"]),
            fam5.get(VERDICT_PREDICTOR, {}).get("reject_null"),
            side_path,
        )

    n_rep = len(replicates)
    cross: dict[str, Any] = {}
    for p in PREDICTORS:
        rhos = rho_by_predictor[p]
        cross[p] = {
            "rhos": rhos,
            "n_positive": sum(1 for r in rhos if r > 0),
            "n_negative": sum(1 for r in rhos if r < 0),
            "holm5_significant_count": holm5_sig_by_predictor[p],
            "t_interval": t_interval(rhos),
        }

    # ── The pinned machine verdict on the nearest-negative predictor. ────────
    verdict = decide_verdict(
        rhos_nn=rho_by_predictor[VERDICT_PREDICTOR],
        holm_sig_count=holm5_sig_by_predictor[VERDICT_PREDICTOR],
        interval=cross[VERDICT_PREDICTOR]["t_interval"],
        n_replicates=n_rep,
    )
    log.info("[phase=verdict] %s", verdict["verdict"])

    # Per-replicate sub-floor label check (plan §7 risk row 4): a replicate
    # whose mean source ΔG reaches >= 1 nat is FLAGGED (reported, not dropped).
    flagged_replicates: list[str] = []
    for rep_key, payload in per_replicate_out.items():
        dgs = [
            v
            for v in payload["usability_descriptive"]["per_cell_source_delta_g"].values()
            if v is not None
        ]
        mean_dg = float(np.mean(dgs)) if dgs else None
        if mean_dg is not None and mean_dg >= 1.0:
            flagged_replicates.append(rep_key)
            notes.append(
                f"{rep_key}: mean source ΔG {mean_dg:.3f} nats >= 1 — this replicate's "
                "'no-implant' label is qualified (reported, not silently dropped)."
            )

    # ── Descriptive pooled fit over all replicates (secondary, not decision-bearing).
    pooled_fit_all = fit_pooled_partial_spearman(rows_all)

    payload = {
        "schema_version": "i555_analysis_replicates_v1",
        "task_id": 555,
        "parent_task_id": 534,
        "replicates": [f"{a}:{b}" for a, b in replicates],
        "frac": f"{args.frac:.2f}",
        "read_point": "optimizer step 5 (hard stop; frac 1.00 of the realized stop)",
        "expected_rows_per_replicate": args.expected_rows,
        "verdict": verdict,
        "specificity_control": {
            "predictor": SPECIFICITY_PREDICTOR,
            **cross[SPECIFICITY_PREDICTOR],
            "read": (
                "systematic structure here too ⇒ geometry-generic artifact; "
                "structure only on nearest-negative ⇒ predictor-specific artifact"
            ),
        },
        "cross_replicate": cross,
        "per_replicate": per_replicate_out,
        "pooled_descriptive_fit": {
            "n_rows": len(rows_all),
            "fit": pooled_fit_all,
            "note": "descriptive only — the replicate is the registered statistical unit",
        },
        "parent_reference": load_parent_reference(args.parent_analysis),
        "manifest_flags": collect_manifest_flags_555(slab),
        "flagged_replicates_mean_dg_above_floor": flagged_replicates,
        "notes": notes,
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    log.info(
        "[phase=done] wrote %s (%d replicates, verdict=%s)",
        out_path,
        len(per_replicate_out),
        verdict["verdict"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
