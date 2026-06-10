#!/usr/bin/env python3
"""Task #505 §5.4 — wrapper to run the joint K + held-out-panel construction gate.

Reads the #472 persona bank + layer-10 centroid bundle from
``data/issue_472/``, runs the spread-quantile selector + tercile / variance
coverage checks, writes the gate payload to
``eval_results/issue_505/panel_coverage.json``.

Halts (returns 2) on a coverage failure after the one-shot K-swap retry.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.leave_one_out_505.panel_coverage import (  # noqa: E402
    PanelCoverageGateError,
    load_inherited_l10_cos,
    run_panel_coverage_gate,
    write_gate_payload,
)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Task #505 §5.4 panel coverage gate.")
    p.add_argument(
        "--persona-bank",
        type=Path,
        default=Path(os.environ.get("EPM_I472_DATA_ROOT", "data/issue_472")) / "persona_bank.json",
    )
    p.add_argument(
        "--centroid-l10",
        type=Path,
        default=Path(os.environ.get("EPM_I472_DATA_ROOT", "data/issue_472")) / "centroids_L10.pt",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("EPM_OUTPUT_ROOT", "eval_results/issue_505"))
        / "panel_coverage.json",
    )
    args = p.parse_args(argv)
    if not args.persona_bank.exists():
        print(f"persona bank missing at {args.persona_bank}", file=sys.stderr)
        return 1
    if not args.centroid_l10.exists():
        print(f"centroids_L10.pt missing at {args.centroid_l10}", file=sys.stderr)
        return 1
    # NEVER raw-`json.loads` this file: #472 publishes persona_bank.json as a
    # STRUCTURED payload `{schema_version, ..., personas: {name: prompt}, ...}`.
    # The canonical loader validates `schema_version == 'i472_v1'` and returns
    # the inner `personas` map directly. The dispatcher loader was fixed in
    # ce2bea8a2 (Codex r4 caught this sibling still raw); see
    # tests/test_issue505_panel_coverage.py for the regression test class.
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )

    bank = load_persona_bank(args.persona_bank)
    cos_l10 = load_inherited_l10_cos(args.centroid_l10)
    try:
        payload = run_panel_coverage_gate(persona_bank=bank, cos_matrix_l10=cos_l10)
    except PanelCoverageGateError as e:
        print(f"panel coverage gate FAILED: {e}", file=sys.stderr)
        return 2
    write_gate_payload(payload, args.out)
    print(
        json.dumps({"gate_passed": payload["gate_passed"], "n_panel": payload["n_panel"]}, indent=2)
    )
    return 0 if payload["gate_passed"] else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
