#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (ρ, ×, →) in scientific docstrings + logs.
"""Issue #503 — regression cross-bucket CPU smoke (Round-2 Rec 6).

Builds a tiny fixture set of RegressionRow records across Buckets A/B/D/E,
runs the actual regression module end-to-end (rows_to_dataframe →
spearman_rho → partial_spearman_ladder → leave_one_bucket_out →
per_bucket_simple_slopes → b_to_b_descriptive), writes a real JSON
artifact to eval_results/issue503/smokes/regression_cross_bucket.json,
and exits 0.

This is the regression-side counterpart to the per-bucket cross_eval
smokes (which need a GPU to actually do vLLM generation; this one runs
fully on CPU because regression is pure pandas + scipy).

Per CLAUDE.md fail-fast: a regression call that returns NaN where a
real number is required raises here (the smoke surfaces real bugs in
the row-builder / regression code, not just import-checks).

Usage:
    uv run python scripts/issue503_regression_cross_bucket_smoke.py
"""

from __future__ import annotations

import json
import logging
import math
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue503_regression_cross_bucket_smoke")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    """Run the end-to-end CPU smoke."""
    print("[phase=build_rows] generating fixture rows across buckets A/B/D/E")

    from explore_persona_space.experiments.issue503.regression import (
        ALL_BUCKETS,
        RegressionRow,
        b_to_b_descriptive,
        leave_one_bucket_out,
        partial_spearman_ladder,
        per_bucket_simple_slopes,
        rows_to_dataframe,
        spearman_rho,
    )

    # Build 8 rows per bucket (32 total), seeded for determinism.
    rng = random.Random(0)
    rows: list[RegressionRow] = []
    rows_per_bucket = 8
    for bucket in ("B", "A", "D", "E"):
        for i in range(rows_per_bucket):
            # Within-bucket correlation: leakage tracks cosine in B/A/D and
            # is noise in E (the non-transfer end).
            cos = rng.uniform(0.0, 1.0)
            target_rate = (
                rng.uniform(0.0, 0.1)
                if bucket == "E"  # non-transfer baseline
                else cos * 0.4 + 0.1
            )
            n = 100
            k = max(0, min(n, int(target_rate * n)))
            cell_type = {
                "B": "N_to_N",
                "A": "N_to_B_syco",
                "D": "N_to_B_EM",
                "E": "N_to_N",
            }[bucket]
            rows.append(
                RegressionRow(
                    source=f"src_{bucket}_{i}",
                    target=f"tgt_{bucket}_{i % 3}",
                    seed=0,
                    cell_type=cell_type,  # type: ignore[arg-type]
                    family={"B": "code", "A": "broad_syco", "D": "advice", "E": "code"}[bucket],
                    k=k,
                    n=n,
                    cosine_predictor=cos,
                    cosine_topic_stripped=cos * 0.9,
                    log_tokens=math.log(100.0 + 0.1 * i),
                    lexical_persona_cosine=0.5,
                    base_rate=0.1,
                    js_sliced_on_target=None,
                    js_sliced_off_target=None,
                    kl_secondary_dv=None,
                    bucket=bucket,  # type: ignore[arg-type]
                )
            )
    print(f"[phase=build_rows] built {len(rows)} fixture rows")

    print("[phase=rows_to_dataframe] sanity-check dataframe shape")
    df = rows_to_dataframe(rows)
    assert len(df) == len(rows), df
    assert set(df["bucket"]) == {"A", "B", "D", "E"}

    print("[phase=spearman] pooled rho across all buckets")
    pooled = spearman_rho(rows)
    print(f"  pooled rho={pooled['rho']:.4f} p={pooled['p_value']:.4g} n={pooled['n']}")
    assert pooled["n"] == len(rows)
    assert not math.isnan(pooled["rho"]), f"pooled rho is NaN: {pooled}"

    print("[phase=partial_ladder] partial-Spearman ladder")
    ladder = partial_spearman_ladder(rows)
    assert set(ladder.keys()) >= {"raw", "partial_log_tokens"}
    for k, v in ladder.items():
        print(f"  {k}: rho={v['rho']:.4f}")

    print("[phase=leave_one_bucket_out] per-bucket sensitivity")
    lbo = leave_one_bucket_out(rows)
    assert set(lbo.keys()) == {"drop_A", "drop_B", "drop_D", "drop_E"}
    for k, v in lbo.items():
        print(f"  {k}: rho={v['rho']:.4f}")

    print("[phase=per_bucket_simple_slopes] per-bucket rho")
    slopes = per_bucket_simple_slopes(rows)
    assert set(slopes.keys()) == {"A", "B", "D", "E"}
    for k, v in slopes.items():
        print(f"  bucket {k}: rho={v['rho']:.4f} n={v['n']}")

    # b_to_b_descriptive only runs on B_to_B cell_type rows; B-bucket rows
    # are N_to_N here, so this returns the empty/null shape gracefully.
    print("[phase=b_to_b_descriptive] descriptive verdict")
    btb = b_to_b_descriptive(rows)
    assert "point_estimate" in btb
    print(f"  point_estimate={btb.get('point_estimate')!r}")

    print("[phase=write_artifact] writing smoke artifact")
    out_dir = PROJECT_ROOT / "eval_results" / "issue503" / "smokes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regression_cross_bucket.json"
    # Reproducibility metadata per CLAUDE.md.
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        git_sha = "unknown"
    payload = {
        "smoke_name": "regression_cross_bucket",
        "n_rows": len(rows),
        "buckets_present": sorted(set(df["bucket"])),
        "all_buckets_known": list(ALL_BUCKETS),
        "pooled_spearman": pooled,
        "partial_ladder": ladder,
        "leave_one_bucket_out": lbo,
        "per_bucket_simple_slopes": slopes,
        "b_to_b_descriptive": btb,
        "reproducibility": {
            "git_sha": git_sha,
            "timestamp": datetime.now(UTC).isoformat(),
            "python": sys.version.split()[0],
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[phase=done] wrote {out_path}")
    print(f"  artifact size = {out_path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
