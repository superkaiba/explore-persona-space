# ruff: noqa: RUF002, RUF003  # em-dash + Greek ρ + − intentional
#!/usr/bin/env python3
"""Task #555 round-2 — tail-exclusion robustness re-fit (free analysis, in-git data).

The raw pooled scatter carries a thin positive tail (21 of 2,160 per-probe
means above +0.10 nats, dominated by the postal_worker × ecosystems-framing
frozen response). This re-fit drops those tail rows per replicate and re-runs
the IDENTICAL 6-predictor partial Spearman + pooled t-interval to check
whether the nearest-negative readings depend on the tail. This is a POST-HOC
robustness variant, NOT the registered analysis — the registered verdict
stays the all-rows fit in analysis_replicates.json.

CPU-only; runs OFF-POD on the VM against committed eval_results/issue_555/:
    uv run python scripts/i555_tail_exclusion_refit.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("i555.tail_exclusion_refit")

DEFAULT_TAIL_THRESHOLD = 0.10  # nats; per-probe-mean DV above this is "tail"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_555"))
    ap.add_argument(
        "--phase05-path", type=Path, default=Path("eval_results/issue_530/phase0_5_gates.json")
    )
    ap.add_argument("--tail-threshold", type=float, default=DEFAULT_TAIL_THRESHOLD)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=tail_refit_555] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        POSITIONED_ARM_SLUGS_V3,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
        aggregate_base_prior_from_trajectories,
        build_rows,
        fit_pooled_partial_spearman,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase05 import (
        load_phase05,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from i555_replicate_analyze import decide_verdict, t_interval

    gates = load_phase05(args.phase05_path)
    reps = [(7, 11), (19, 23), (71, 73), (101, 103), (211, 223)]
    out_path = (
        args.out if args.out is not None else args.slab_root / "analysis_tail_exclusion_refit.json"
    )

    rhos: list[float] = []
    holm_sig = 0
    per_rep: dict[str, dict] = {}
    for a, b in reps:
        bp = aggregate_base_prior_from_trajectories(
            slab_root=args.slab_root, seeds=[a, b], positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3
        )
        pooled = build_rows(
            slab_root=args.slab_root,
            chosen_frac=1.0,
            per_probe=gates["per_probe"],
            arm_to_positioned_n=gates["arm_to_positioned_n"],
            seeds=[a, b],
            base_prior_by_probe=bp,
            positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
            dg_band=None,
        )
        rows = [r for r in pooled["rows"] if r["delta_g"] <= args.tail_threshold]
        n_dropped = len(pooled["rows"]) - len(rows)
        fit = fit_pooled_partial_spearman(rows)
        ps = fit["partial_spearman"]["d_nearest_neg_nd"]
        rhos.append(float(ps["rho"]))
        # Family-5 Holm over the non-degenerate predictors (training_step excluded).
        raws = sorted(
            (v["p_raw"], p) for p, v in fit["partial_spearman"].items() if p != "training_step"
        )
        sig: set[str] = set()
        for i, (pv, pred) in enumerate(raws):
            if pv <= 0.05 / (len(raws) - i):
                sig.add(pred)
            else:
                break
        if "d_nearest_neg_nd" in sig:
            holm_sig += 1
        per_rep[f"seeds{a}_{b}"] = {
            "n_rows": len(rows),
            "n_dropped_tail_rows": n_dropped,
            "partial_spearman": fit["partial_spearman"],
            "holm5_significant": sorted(sig),
        }
        log.info(
            "[phase=tail_refit] seeds %d,%d: n=%d (dropped %d), rho_nn=%+.4f (p=%.4f)",
            a,
            b,
            len(rows),
            n_dropped,
            ps["rho"],
            ps["p_raw"],
        )

    ti = t_interval(rhos)
    verdict = decide_verdict(
        rhos_nn=rhos, holm_sig_count=holm_sig, interval=ti, n_replicates=len(reps)
    )
    payload = {
        "schema_version": "i555_tail_exclusion_refit_v1",
        "task_id": 555,
        "note": (
            "POST-HOC robustness variant of analysis_replicates.json: per-probe rows "
            f"with DV > {args.tail_threshold} nats excluded before the identical fit. "
            "NOT the registered analysis; the registered verdict is the all-rows fit."
        ),
        "tail_threshold_nats": args.tail_threshold,
        "per_replicate": per_rep,
        "rhos_nearest_neg": rhos,
        "n_positive": sum(1 for r in rhos if r > 0),
        "holm5_significant_count": holm_sig,
        "t_interval": ti,
        "nominal_verdict_on_this_posthoc_variant": verdict,
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    log.info(
        "[phase=done] wrote %s (pooled mean %+.4f [%+.4f, %+.4f], verdict-on-variant %s)",
        out_path,
        ti["mean"],
        ti["lo"],
        ti["hi"],
        verdict["verdict"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
