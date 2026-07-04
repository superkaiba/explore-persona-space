#!/usr/bin/env python3
# ruff: noqa: RUF002
# Intentional Unicode (Δ) in the results-note field names.
"""Issue #928 end-of-workload results sentinel (round-2 sentinel-timing fix).

The extract phase's sentinel is ``epm:progress`` — the ONE ``epm:results``
sentinel of the pipeline fires HERE, from the run_all driver's finalize step,
after fits + figures + uploads all completed (code-review r1: an extract-end
results sentinel drained into an ``epm:results`` marker hours before the fit
phases finished, and the terminal state carried no fit digest).

Reads the store manifest + fit outputs + figures dir, composes the compact
results note (rung / gate / primary layer conventions / H2–H4 primary deltas /
HF artifact prefixes), and writes the poll_pipeline-conformant sentinel via
``issue928_common.write_sentinel``. Fail-loud: this runs only after the fit +
figure phases exited 0, so a missing input here IS a pipeline bug.

Usage (normally via ``issue928_run_all.sh``)::

    uv run python scripts/issue928_finalize.py \\
        --store data/issue_928/store --results eval_results/issue_928 \\
        --figures figures/issue_928 --out-dir data/issue_928
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

from issue928_common import (  # noqa: E402
    DECOMP_TENSORS_PREFIX,
    FIGURES_PREFIX,
    FIT_RESULTS_PREFIX,
    RAW_COMPLETIONS_PREFIX,
    STORE_PREFIX,
    load_json,
    write_sentinel,
)

logger = logging.getLogger("issue928_finalize")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #928 end-of-workload results sentinel")
    ap.add_argument("--store", default=str(PROJECT_ROOT / "data" / "issue_928" / "store"))
    ap.add_argument("--results", default=str(PROJECT_ROOT / "eval_results" / "issue_928"))
    ap.add_argument("--figures", default=str(PROJECT_ROOT / "figures" / "issue_928"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "issue_928"))
    ap.add_argument(
        "--sentinel-dir",
        default=None,
        help="override the /workspace/logs sentinel dir (smoke runs redirect to scratch)",
    )
    args = ap.parse_args()

    manifest = load_json(Path(args.store) / "manifest.json")
    grid_blob = load_json(Path(args.results) / "recon_skill_grid.json")
    boot_blob = load_json(Path(args.results) / "bootstrap_deltaskill.json")
    figures = sorted(p.name for p in Path(args.figures).glob("*.png"))

    smoke = bool(grid_blob.get("smoke", False)) or bool(manifest.get("smoke", False))
    suffix = "_smoke" if manifest.get("smoke", False) else ""
    primary_deltas = {}
    for regime, b in boot_blob.get("by_regime", {}).items():
        for name, s in b.get("statistics", {}).items():
            if name.startswith(("H2", "H3", "H4")):
                primary_deltas[f"{regime}/{name}"] = s["primary_frozen_direct_best"]

    parity = grid_blob.get("parity_gate", {})
    note = {
        "phase": "pipeline_complete",
        "smoke": smoke,
        "n_contexts": len(manifest["context_ids"]),
        "rung": manifest["rung"],
        "flagged_below_parse_floor": manifest["flagged_below_parse_floor"],
        "regimes": sorted(grid_blob["results"].keys()),
        "n_perms": grid_blob.get("n_perms"),
        "n_boot": grid_blob.get("n_boot"),
        "parity_gate_max_dev": (
            None if parity.get("skipped") else max(v for v in parity.values() if v is not None)
        ),
        "primary_frozen_direct_best_layer": {
            r: b["layer_conventions"]["primary_frozen_direct_best_layer"]
            for r, b in boot_blob.get("by_regime", {}).items()
        },
        "primary_deltas": primary_deltas,
        "n_figures": len(figures),
        "hf_prefixes": {
            "raw_completions": RAW_COMPLETIONS_PREFIX + suffix,
            "store": STORE_PREFIX + suffix,
            "fit_results": FIT_RESULTS_PREFIX,
            "decomp_tensors": DECOMP_TENSORS_PREFIX,
            "figures": FIGURES_PREFIX,
        },
    }
    target = write_sentinel(
        "epm:results",
        note,
        Path(args.out_dir),
        log_dir=Path(args.sentinel_dir) if args.sentinel_dir else None,
    )
    logger.info("[phase=finalize_done] results sentinel at %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
