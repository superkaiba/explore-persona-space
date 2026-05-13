#!/usr/bin/env python3
"""Top-level pipeline runner for issue #368.

Sequences the scripts per plan §15 ordering:

  Phase 0.0   panel-recoverability gate
  Phase 0.1   Sonnet paraphrase generation (triggers + personas + negset)
  Phase 2     extraction (10 personas + assistant + helpful + empty-prompt baseline)
              → emits _centroid_mean_L{15,20,25}.pt (Phase 0.3)
  Phase 1     extraction (4 triggers)
  Phase 1     projection (32 panel × 20 questions → augmented CSV)
  Phase 2     projection (50 directed pairs, reproduction-sanity gate)
  Phase 1     analysis
  Phase 2     analysis
  Cross-phase synthesis (prose + figure)

Usage::

    # End-to-end, single H100
    uv run python scripts/run_i368.py --gpu-id 0

    # Smoke test (no GPU, no API calls)
    uv run python scripts/run_i368.py --smoke-test

    # Skip Phase 0.1 (paraphrase generation cached on disk already)
    uv run python scripts/run_i368.py --skip-phase0-prep
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, label: str) -> None:
    print(f"\n========== {label} ==========")
    print(" ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    if res.returncode != 0:
        raise SystemExit(f"{label} failed (returncode={res.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--skip-phase0-prep", action="store_true")
    ap.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip extraction + projection; only run analysis + synthesis on existing artifacts.",
    )
    args = ap.parse_args()

    py = sys.executable
    smoke = ["--smoke-test"] if args.smoke_test else []
    gpu = ["--gpu-id", str(args.gpu_id)]

    if not args.analysis_only:
        if not args.skip_phase0_prep:
            _run(
                [py, "scripts/i368_phase0_data_prep.py", *smoke],
                label="Phase 0.0 + 0.1 (data prep)",
            )
        _run(
            [py, "scripts/i368_extract_chenstyle_vectors.py", "--phase", "2", *gpu, *smoke],
            label="Phase 2 extraction (incl. centroid_mean Phase 0.3)",
        )
        _run(
            [py, "scripts/i368_extract_chenstyle_vectors.py", "--phase", "1", *gpu, *smoke],
            label="Phase 1 extraction (4 triggers)",
        )
        _run(
            [py, "scripts/i368_phase1_projection.py", *gpu, *smoke],
            label="Phase 1 projection (32 panel × 20 q)",
        )
        _run(
            [py, "scripts/i368_phase2_projection.py"],
            label="Phase 2 projection (50 directed pairs + reproduction gate)",
        )

    if not args.smoke_test:
        _run([py, "scripts/i368_phase1_analysis.py"], label="Phase 1 analysis")
        _run([py, "scripts/i368_phase2_analysis.py"], label="Phase 2 analysis")
        _run([py, "scripts/i368_crossphase_synthesis.py"], label="Cross-phase synthesis")
    else:
        print("\n[smoke-test] analysis + synthesis skipped (no CSVs generated).")


if __name__ == "__main__":
    main()
