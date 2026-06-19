#!/usr/bin/env python3
"""Issue #657 — LOBO pooled re-read (free-analysis, ANALYSIS-ONLY, CPU, off-pod).

The registered higher-N secondary read (plan §324 / §6.5 / §365 / §389): pool the
in-scope behaviors at the PREDICTOR-SKILL level (within-behavior source-FE + a
percentile-rank that removes each behavior's install magnitude — "never pooling
behaviors with different install strengths", plan line 262), then run a
leave-one-(behavior, bystander)-group-out held-out read + the strict
leave-one-WHOLE-behavior-out read. The pooled N (~3 x 23 = 69 groups) is much larger
than any per-behavior N (~23), so the bootstrap ΔR² CI narrows — it may tighten the
per-behavior INDETERMINATE/NULL H3 verdicts toward a confirmed null at higher power.

INPUT (already on disk, NO new training / eval / pod / download):
  ``eval_results/issue_657/per_behavior/{behavior}.json`` — the per-behavior bake-off
  outputs. Each carries ``joined_cells`` (the off-diag (source, bystander) cells with
  ``delta`` leakage DV + ``align`` predictor + ``bystander_base_rate`` +
  ``prior_centroid_projection``) and a ``leakage_bake_off.in_scope`` flag (k2 / panel
  gates). Only ``in_scope: true`` behaviors enter the pool — the same gate the
  per-behavior bake-off uses (marker is out of scope: 3 bystanders, k2_pass false).

OUTPUT:
  ``eval_results/issue_657/lobo_pooled.json`` — the nested LOBO result + reproducibility
  metadata. Re-runnable / idempotent (deterministic given the seed).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.issue657_alignment_predictor import (  # noqa: E402
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    lobo_pooled,
)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_per_behavior_cells(per_behavior_dir: Path) -> tuple[dict[str, list[dict]], dict]:
    """Load the in-scope behaviors' ``joined_cells`` from the per-behavior JSONs.

    Returns ({behavior: joined_cells}, provenance) where provenance records which
    behaviors were included / excluded and the per-behavior in-scope flag + cell count.
    """
    cells_by_behavior: dict[str, list[dict]] = {}
    provenance: dict[str, dict] = {}
    files = sorted(per_behavior_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no per_behavior/*.json under {per_behavior_dir}")
    for fp in files:
        behavior = fp.stem
        data = json.loads(fp.read_text())
        lk = data.get("leakage_bake_off") or {}
        in_scope = bool(lk.get("in_scope", False))
        joined = data.get("joined_cells") or []
        # Each joined cell MUST carry the modeled columns; fail loud if a behavior's
        # substrate is malformed (no silent skip of a real-but-broken behavior).
        required = {"source", "bystander", "delta", "align", "bystander_base_rate"}
        missing_cols = (
            [c for c in joined if not required.issubset(c)] if in_scope and joined else []
        )
        if missing_cols:
            raise ValueError(
                f"{behavior}: {len(missing_cols)} joined_cells missing one of {sorted(required)}"
            )
        provenance[behavior] = {
            "in_scope": in_scope,
            "n_joined_cells": len(joined),
            "n_resolvable_bystanders": data.get("n_resolvable_bystanders"),
            "source_path": str(fp.relative_to(PROJECT_ROOT))
            if fp.is_relative_to(PROJECT_ROOT)
            else str(fp),
            "included": in_scope and len(joined) >= 4,
        }
        if in_scope and len(joined) >= 4:
            cells_by_behavior[behavior] = joined
    return cells_by_behavior, provenance


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #657 LOBO pooled re-read (CPU, off-pod).")
    ap.add_argument(
        "--eval-results-root",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_657",
        help="root holding per_behavior/*.json (default eval_results/issue_657).",
    )
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON (default <eval-results-root>/lobo_pooled.json).",
    )
    args = ap.parse_args()

    per_behavior_dir = args.eval_results_root / "per_behavior"
    out_path = args.out or (args.eval_results_root / "lobo_pooled.json")

    cells_by_behavior, provenance = load_per_behavior_cells(per_behavior_dir)
    included = sorted(cells_by_behavior)
    print(f"[phase=load] in-scope behaviors pooled: {included}")
    for b, p in provenance.items():
        flag = "INCLUDED" if p["included"] else "excluded"
        print(f"  {b}: {flag} (in_scope={p['in_scope']} n_cells={p['n_joined_cells']})")

    print(f"[phase=lobo] running pooled LOBO (B={args.n_boot}, seed={args.seed}) ...")
    result = lobo_pooled(cells_by_behavior, b=args.n_boot, seed=args.seed)

    payload = {
        "experiment": "issue657_lobo_pooled",
        "read": "leave-one-behavior-out (LOBO) higher-N predictor-skill pooled secondary read",
        "registered_in_plan": "§324 / §6.5 / §365 / §389 (higher-N secondary read)",
        "lobo": result,
        "input_provenance": provenance,
        "reproducibility": {
            "git_commit": _git_commit(),
            "generated_utc": datetime.now(UTC).isoformat(),
            "bootstrap_b": args.n_boot,
            "bootstrap_seed": args.seed,
            "python": sys.version.split()[0],
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    gc = result.get("grouped_cv", {}) if isinstance(result, dict) else {}
    dr2 = gc.get("delta_r2_align_beyond_prior", {})
    rho = gc.get("doubly_partialled_rho", {})
    verdict = result.get("h3_verdict", {}).get("verdict") if isinstance(result, dict) else None
    print(
        "[phase=summary] "
        f"n_pooled_cells={result.get('n_pooled_cells')} n_groups={result.get('n_groups')} "
        f"pooled_doubly_partialled_rho={rho.get('point')} "
        f"CI=[{rho.get('ci_lo')}, {rho.get('ci_hi')}] | "
        f"pooled_deltaR2={dr2.get('point')} CI=[{dr2.get('ci_lo')}, {dr2.get('ci_hi')}] | "
        f"H3_verdict={verdict}"
    )
    print(f"[phase=write] wrote {out_path}")
    print("[phase=done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
