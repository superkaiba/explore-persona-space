#!/usr/bin/env python3
"""Task #627 Phase 3 — fraction view of the #514 marker LoRA-vs-FT slab.

LOG-PROB SPACE ONLY (plan §11: the #514 leaves predate the #530 four-float
contract — z-fields are verifiably ABSENT, so the margin fraction is NOT
computable; Phase 0 asserts the absence). Validity rests on the registered
off-saturation filter: source ``trained_logp <= -1`` nat AND source
argmax-marker rate ``< 0.10`` — off saturation Δlog Z ≈ 0 so Δlog P ≈
Δz_marker and the log-prob fraction is faithful. Cells failing the filter are
marked ``saturated_excluded`` — never zeroed.

Coverage split (both committed under ``eval_results/issue_514/``):
  - 6 LEAF cells (#514 FT retrains) carry per-persona x per-question records:
    full filter evaluation + per-persona fractions;
  - the 12-cell aggregate table (``analysis.json``, incl. the #508-delegated
    LoRA cells) carries cell-level means only: cell-level fractions reported
    with ``filter_inputs: missing`` where the source-side filter inputs were
    never committed in this slab (descriptive; the analyzer must not lean on
    them) — the published matched verdict (``_matched_rate_514.json``) is the
    H3 consistency anchor either way.

Output: eval_results/issue_627/analysis/fractions_514.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i627_analyze_514")

OUT_DIR = Path("eval_results/issue_627/analysis")
ROOT_514 = Path("eval_results/issue_514")
TRAINED_LOGP_CEILING = -1.0  # off-saturation: source trained log P <= -1 nat
ARGMAX_RATE_CEILING = 0.10  # source argmax-marker rate < 0.10


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def analyze_leaf(path: Path) -> dict:
    with open(path) as f:
        leaf = json.load(f)
    src_block = leaf["delta_g_source"]
    if len(src_block) != 1:
        raise RuntimeError(f"{path.name}: expected one source persona, got {list(src_block)}")
    source_persona, src_recs = next(iter(src_block.items()))
    src_logps = [float(r["trained_logp"]) for r in src_recs.values()]
    src_argmax = [bool(r["trained_argmax_marker"]) for r in src_recs.values()]
    src_delta = float(np.mean([float(r["delta_g"]) for r in src_recs.values()]))
    trained_logp_mean = float(np.mean(src_logps))
    argmax_rate = float(np.mean(src_argmax))
    off_saturation = trained_logp_mean <= TRAINED_LOGP_CEILING and argmax_rate < ARGMAX_RATE_CEILING

    held = leaf["delta_g_held_out"]
    per_persona_delta = {
        p: float(np.mean([float(r["delta_g"]) for r in recs.values()])) for p, recs in held.items()
    }
    bys_mean = float(np.mean(list(per_persona_delta.values())))
    return {
        "cell": leaf["cell_slug"],
        "arm": leaf["arm"],
        "seed": leaf["seed"],
        "source_persona": source_persona,
        "source_mean_delta_g": src_delta,
        "source_trained_logp_mean": trained_logp_mean,
        "source_argmax_marker_rate": argmax_rate,
        "off_saturation_filter_pass": bool(off_saturation),
        "filter_inputs": "leaf",
        "status": "valid" if off_saturation else "saturated_excluded",
        "bystander_mean_delta_g": bys_mean,
        "fraction_dlogp": (bys_mean / src_delta) if off_saturation else None,
        "per_persona_fraction": (
            {p: d / src_delta for p, d in per_persona_delta.items()} if off_saturation else None
        ),
        "n_held_out_personas": len(held),
        "file": str(path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #627 Phase 3 — #514 log-prob fraction view.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    leaf_paths = sorted(
        p
        for p in ROOT_514.glob("*.json")
        if not p.name.startswith("_") and p.name not in ("analysis.json", "analysis_514.json")
    )
    leaves = [analyze_leaf(p) for p in leaf_paths]
    leaf_cells = {r["cell"] for r in leaves}

    with open(ROOT_514 / "analysis.json") as f:
        agg = json.load(f)
    aggregate_cells = []
    for c in agg["cells"]:
        src = c.get("source_mean")
        held = c.get("held_out_mean")
        src_ok = src is not None and isinstance(src, float) and np.isfinite(src)
        aggregate_cells.append(
            {
                "cell": c["cell"],
                "arm": c["arm"],
                "source_mean_delta_g": src,
                "bystander_mean_delta_g": held,
                "n_collapsed": c.get("n_collapsed"),
                "fraction_dlogp_descriptive": (held / src) if src_ok and src else None,
                "filter_inputs": "leaf" if c["cell"] in leaf_cells else "missing",
                "note": (
                    None
                    if c["cell"] in leaf_cells
                    else "source-side filter inputs (trained_logp / argmax rate) were never "
                    "committed in this slab — descriptive only"
                ),
            }
        )

    with open(ROOT_514 / "_matched_rate_514.json") as f:
        matched = json.load(f)

    result = {
        "issue": 627,
        "family": "marker_lora_vs_ft_514",
        "space": "log-prob ONLY (z-fields verifiably absent; margin not computable)",
        "off_saturation_filter": {
            "trained_logp_ceiling": TRAINED_LOGP_CEILING,
            "argmax_rate_ceiling": ARGMAX_RATE_CEILING,
            "rationale": "off saturation dlogZ ~= 0 so dlogP ~= dz_marker (faithful fraction)",
        },
        "leaf_cells": leaves,
        "aggregate_cells_descriptive": aggregate_cells,
        "published_matched_verdict_verbatim": {
            "matched_rate_gap_ft_minus_lora_nat": matched["matched_rate_gap_ft_minus_lora_nat"],
            "ci": [matched["matched_rate_gap_ci_lo_nat"], matched["matched_rate_gap_ci_hi_nat"]],
            "target_nats": matched["matched_slice_target_nats"],
            "lora_anchor_cells": matched["matched_rate_lora_anchor_cells"],
            "fullft_anchor_cells": matched["matched_rate_fullft_anchor_cells"],
        },
        "metadata": {
            "git_commit_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "numpy_version": np.__version__,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fractions_514.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    n_valid = sum(1 for r in leaves if r["status"] == "valid")
    log.info(
        "[phase=p3_514] -> %s (%d leaf cells: %d valid, %d saturated-excluded; %d aggregate)",
        out_path,
        len(leaves),
        n_valid,
        len(leaves) - n_valid,
        len(aggregate_cells),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
