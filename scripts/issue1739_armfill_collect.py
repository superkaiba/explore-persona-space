#!/usr/bin/env python3
"""Collect the #1739 armfill legs into one committed tree + a coverage table.

Each leg of the armfill round writes its own out-root (per-leg roots are
mandatory: the scorers share ``<out-root>/<behavior>/all_arms_spearman.json``
and one ``percell/`` checkpoint, so concurrent legs on a shared root clobber
each other). This script folds the 12 leg roots into

    eval_results/issue_1739/armfill_round/<rung>/<behavior>/
        all_arms_spearman.<variant>.json
        map_diagnostics.<variant>.json
        preds/...

and emits ``coverage.json`` -- one row per (arm, rung, behavior, variant) with
the frozen-layer Spearman rho, its bootstrap CI, and n_eval -- so a missing
cell is visible as an explicit absence rather than an unnoticed gap.

Read-only with respect to the leg roots; it copies, never moves.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BEHAVIORS = ("evil", "sycophancy", "hallucination")
RUNGS = ("wildchat_rung", "pvsynth")
VARIANTS = ("context_end", "prefix_end")
# Arms this round can actually produce. arm17_oracle_mlp / arm18_oracle_krr are
# EXCLUDED by construction, not by preference: they have ZERO rows in the
# committed TRAIN summaries for all three behaviors, so the transfer scorer's
# committed-frozen selection ("modal-committed-train-cells") has no train-frozen
# layer to score them at and fail-louds. Keeping them in the expected set would
# emit one MISSING row per arm per leg (24 rows) and bury a genuine missing cell.
ARMS = (
    "arm2_ctx_native",
    "arm9_pretrain_ft",
    "arm14_shuffled_pt",
)
UNPRODUCIBLE_ARMS = {
    "arm17_oracle_mlp": "no committed TRAIN rows on any behavior (0/0/0) — no train-frozen layer",
    "arm18_oracle_krr": "no committed TRAIN rows on any behavior (0/0/0) — no train-frozen layer",
}


def leg_slug(behavior: str, rung: str, variant: str) -> str:
    return f"{behavior}_{rung}_{variant}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legs-root", type=Path, default=Path("/workspace/legs"))
    ap.add_argument("--dest", type=Path, default=Path("eval_results/issue_1739/armfill_round"))
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--rungs", nargs="+", default=list(RUNGS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument(
        "--expected-arms",
        nargs="+",
        default=list(ARMS),
        help="arms every leg must emit a transfer row for; a shortfall is reported as MISSING",
    )
    ap.add_argument("--copy-preds", action="store_true", help="also copy per-context preds jsonl")
    args = ap.parse_args()

    coverage: list[dict] = []
    missing: list[dict] = []
    collected = 0

    for behavior in args.behaviors:
        for rung in args.rungs:
            for variant in args.variants:
                slug = leg_slug(behavior, rung, variant)
                src_dir = args.legs_root / slug / rung / behavior
                src = src_dir / "all_arms_spearman.json"
                out_dir = args.dest / rung / behavior
                if not src.is_file():
                    missing.append({"leg": slug, "reason": f"absent {src}"})
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out_dir / f"all_arms_spearman.{variant}.json")
                diag = src_dir / "map_diagnostics.json"
                if diag.is_file():
                    shutil.copy2(diag, out_dir / f"map_diagnostics.{variant}.json")
                if args.copy_preds and (src_dir / "preds").is_dir():
                    shutil.copytree(src_dir / "preds", out_dir / "preds", dirs_exist_ok=True)
                collected += 1

                payload = json.loads(src.read_text())
                rows = payload.get("transfer_rows", [])
                seen = set()
                for r in rows:
                    seen.add(r.get("arm"))
                    ci = r.get("ci_frozen") or [None, None]
                    coverage.append(
                        {
                            "behavior": behavior,
                            "rung": rung,
                            "variant": variant,
                            "arm": r.get("arm"),
                            "rho_frozen": r.get("rho_frozen"),
                            "ci_lo": ci[0],
                            "ci_hi": ci[1],
                            "n_eval": r.get("n_eval"),
                            "layer": r.get("layer"),
                        }
                    )
                for arm in args.expected_arms:
                    if arm not in seen:
                        skips = payload.get("transfer_skips", [])
                        why = next(
                            (s.get("reason") for s in skips if s.get("arm") == arm),
                            "no transfer row emitted",
                        )
                        missing.append({"leg": slug, "arm": arm, "reason": f"not scored: {why}"})

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "coverage.json").write_text(
        json.dumps(
            {
                "n_legs_collected": collected,
                "n_legs_expected": len(args.behaviors) * len(args.rungs) * len(args.variants),
                "expected_arms": list(args.expected_arms),
                "unproducible_arms": UNPRODUCIBLE_ARMS,
                "coverage": coverage,
                "missing": missing,
            },
            indent=1,
        )
    )

    print(f"collected {collected} legs -> {args.dest}")
    print(f"coverage rows: {len(coverage)}; missing entries: {len(missing)}")
    hdr = (
        f"{'arm':<20} {'rung':<14} {'behavior':<14} {'variant':<12} {'rho':>7} {'ci':>18} {'n':>6}"
    )
    print(hdr)
    for c in sorted(coverage, key=lambda x: (x["arm"], x["rung"], x["behavior"], x["variant"])):
        lo, hi = c["ci_lo"], c["ci_hi"]
        ci = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None and hi is not None else "n/a"
        rho = f"{c['rho_frozen']:+.3f}" if c["rho_frozen"] is not None else "n/a"
        print(
            f"{c['arm']:<20} {c['rung']:<14} {c['behavior']:<14} "
            f"{c['variant']:<12} {rho:>7} {ci:>18} {c['n_eval']:>6}"
        )
    for m in missing:
        print(f"MISSING {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
