#!/usr/bin/env python3
"""Issue #545 — eval battery for one adapter (or the base panel).

Invoked by ``scripts/issue545_sweep.py`` once per (cell, eval-phase) with
explicit ``env={**os.environ}``. The dispatcher runs the three phases as
SEPARATE subprocesses (``--only gen|hf|judge``) so vLLM and HF never share a
process by default (the vLLM-teardown gotcha's escape hatch); a single
combined invocation (no ``--only``) is supported and relies on the hardened
teardown + CVD-aware orphan check.

Dose-to-target: ``--diagonal-only`` evaluates JUST the row's diagonal battery
(used per checkpoint to pick the first checkpoint in band before the full
battery runs).

Each phase persists per (column, context) the moment it completes.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

if Path("/workspace").exists():  # pod-only cache redirect; VM keeps its default
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("issue545_eval_cell")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #545 per-adapter eval battery")
    parser.add_argument("--row", default=None, help="Row id (omit with --base-panel)")
    parser.add_argument("--arm", default="primary")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter-path", default=None, help="Local adapter dir (LoRA)")
    parser.add_argument("--base-panel", action="store_true", help="Base model panel (no adapter)")
    parser.add_argument("--contexts", nargs="+", default=["default"])
    parser.add_argument("--columns", nargs="+", default=None, help="Column subset")
    parser.add_argument("--diagonal-only", action="store_true")
    parser.add_argument("--max-probes", type=int, default=None, help="Smoke cap per battery")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--only",
        choices=("gen", "hf", "judge"),
        default=None,
        help="Run one phase (dispatcher default: three separate subprocesses)",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from explore_persona_space.experiments.behavior_testbed_545 import cells_dir
    from explore_persona_space.experiments.behavior_testbed_545.eval_battery import (
        run_generation_phase,
        run_judge_phase,
        run_marker_and_capability_phase,
    )
    from explore_persona_space.experiments.behavior_testbed_545.rows import get_row

    if args.base_panel:
        row = None
        cell = "base_panel"
        adapter = None
    else:
        if not args.row:
            parser.error("--row is required unless --base-panel")
        row = get_row(args.row)
        cell = row.cell_id(args.arm, args.seed)
        adapter = args.adapter_path
        if adapter is None and not args.base_panel:
            parser.error("--adapter-path is required for adapter cells")

    out_dir = cells_dir() / cell
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = args.columns
    if args.diagonal_only:
        if row is None:
            parser.error("--diagonal-only needs --row")
        columns = [row.diagonal_column]

    run_all = args.only is None
    if run_all or args.only == "gen":
        logger.info("[phase=gen] cell=%s contexts=%s", cell, args.contexts)
        run_generation_phase(
            adapter_path=adapter,
            row=row,
            out_dir=out_dir,
            contexts=args.contexts,
            columns=columns,
            max_probes=args.max_probes,
        )
    marker_in_scope = columns is None or "marker" in columns
    if (run_all or args.only == "hf") and marker_in_scope:
        logger.info("[phase=hf] cell=%s", cell)
        run_marker_and_capability_phase(
            adapter_path=adapter,
            out_dir=out_dir,
            contexts=args.contexts,
            max_probes=args.max_probes,
            run_capability=columns is None or "capability" in (columns or []),
        )
    if run_all or args.only == "judge":
        logger.info("[phase=judge] cell=%s", cell)
        run_judge_phase(row=row, out_dir=out_dir, contexts=args.contexts, columns=columns)
    logger.info("[phase=done] eval cell %s complete", cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
