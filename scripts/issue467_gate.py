#!/usr/bin/env python3
"""Issue #467 elicitation gate — pure read/decide helper for run_issue467.sh.

Reads ``data/issue467/elicitation_check/<cell>.json`` (written by
``issue467_elicitation_check.py``) for every requested cell and emits:

* ``data/issue467/gate/pass_cells.txt`` — one PASS cell per line. The
  launcher's strong-NL cosine / JS sweeps consume this list so a
  FAIL-elicitation cell can never feed the strong-NL conditioning rows
  (which is the whole point of the §6.2 / §4.1 Step A.5 gate — strong-NL
  is only trustworthy where the model actually elicits under it).
* ``data/issue467/gate/gate_status.json`` — per-cell status + summary
  totals + the abort decision. The regress reads this so the headline
  strong-NL row uses the PASS subset; the full-cell row falls out as the
  RF5 robustness read.

PASS bar (cell counts as PASS for the gate):
  - The elicitation JSON's ``status`` field equals ``"PASS"`` (i.e. the
    cell cleared the binomial-CI absolute floor, the 0.5x relative bar,
    AND the calibration smoke). FAIL_CALIBRATION / FAIL_ABSOLUTE /
    FAIL_RELATIVE → DROP.
  - Cells missing an elicitation JSON entirely → DROP (counted in the
    drop tally — usually means the author step FAIL_LEAK'd them
    upstream, or the elicitation step crashed for them).

Abort criterion (methodology correction, round 10 — 2026-06-03):
  - Drops are EXPECTED. The Instruct model legitimately won't exhibit
    bad-medical / extreme-sports / risky-financial behavior under ANY
    in-context route; the judge is genuinely near-chance on a few
    behaviors. So the gate does NOT cap the number of drops — it
    protects the MINIMUM viable sweep sample size.
  - If fewer than ``--min-viable`` (default 6) cells PASS, the gate
    exits non-zero AFTER writing the gate_status.json — the launcher's
    ``set -e`` then aborts the run BEFORE the (now-statistically-
    underpowered) strong-NL sweeps. Raise via ``--min-viable N`` or
    lower it only when the planner approves a re-scope.

Usage:
    uv run python scripts/issue467_gate.py \
        --pairs aesthetic_popular aesthetic_unpopular ... \
        [--min-viable 6]

Exits 0 on enough-viable-cells; non-zero on below-min-viable or
malformed inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ELICIT_DIR = PROJECT_ROOT / "data" / "issue467" / "elicitation_check"
GATE_DIR = PROJECT_ROOT / "data" / "issue467" / "gate"

logger = logging.getLogger("issue467_gate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def classify_cell(pair: str) -> dict:
    """Return {"pair": ..., "verdict": "PASS" | "DROP", "reason": ..., "raw_status": ...}.

    ``raw_status`` is the elicitation JSON's ``status`` field (or
    ``"NO_ELICITATION_JSON"`` if missing); ``reason`` is a one-line
    English explanation.
    """
    f = ELICIT_DIR / f"{pair}.json"
    if not f.exists():
        return {
            "pair": pair,
            "verdict": "DROP",
            "reason": (
                f"No {f.relative_to(PROJECT_ROOT)} on disk — author step "
                "likely FAIL_LEAK'd this cell upstream, or the elicitation "
                "step crashed for it."
            ),
            "raw_status": "NO_ELICITATION_JSON",
        }
    try:
        d = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        return {
            "pair": pair,
            "verdict": "DROP",
            "reason": f"Malformed elicitation JSON: {e}",
            "raw_status": "MALFORMED_JSON",
        }
    st = d.get("status", "?")
    if st == "PASS":
        return {
            "pair": pair,
            "verdict": "PASS",
            "reason": (
                f"r_strong={d.get('r_strong')!r} ci=[{d.get('r_strong_ci_lo')!r}, "
                f"{d.get('r_strong_ci_hi')!r}] r_lit={d.get('r_lit')!r}"
            ),
            "raw_status": st,
        }
    return {
        "pair": pair,
        "verdict": "DROP",
        "reason": (
            f"Elicitation status={st!r} — r_strong={d.get('r_strong')!r} r_lit={d.get('r_lit')!r}"
        ),
        "raw_status": st,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="The cell list the launcher dispatched to elicitation (typically all 18).",
    )
    parser.add_argument(
        "--min-viable",
        type=int,
        default=6,
        help=(
            "Minimum number of PASS cells needed to proceed. Drops are EXPECTED "
            "(Instruct-floor + judge-unreliable cells) — this protects the sweep "
            "sample size, not the drop count. Abort the run (exit non-zero) when "
            "fewer than this many cells PASS. Default 6 (the planner-stated bar)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(GATE_DIR),
        help="Where to write pass_cells.txt + gate_status.json (default data/issue467/gate).",
    )
    parser.add_argument(
        "--fail-below-min-viable",
        action="store_true",
        default=True,
        help=(
            "Exit non-zero when n_viable < min-viable (default ON — the launcher's "
            "set -e is what stops the run). Disable for round-1 probe / analysis-only "
            "invocations."
        ),
    )
    parser.add_argument(
        "--no-fail-below-min-viable",
        dest="fail_below_min_viable",
        action="store_false",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cell: list[dict] = [classify_cell(p) for p in args.pairs]
    pass_cells = [c["pair"] for c in per_cell if c["verdict"] == "PASS"]
    drop_cells = [c["pair"] for c in per_cell if c["verdict"] == "DROP"]
    n_viable = len(pass_cells)
    below_min_viable = n_viable < args.min_viable

    summary = {
        "n_pairs_dispatched": len(args.pairs),
        "n_pass": n_viable,
        "n_viable": n_viable,
        "n_drop": len(drop_cells),
        "min_viable_threshold": args.min_viable,
        "below_min_viable": below_min_viable,
        "pass_cells": pass_cells,
        "drop_cells": drop_cells,
        "per_cell": per_cell,
    }

    pass_path = out_dir / "pass_cells.txt"
    pass_path.write_text("\n".join(pass_cells) + ("\n" if pass_cells else ""))
    status_path = out_dir / "gate_status.json"
    status_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Gate: %d viable / %d drop of %d (need >= %d viable to proceed)",
        n_viable,
        len(drop_cells),
        len(args.pairs),
        args.min_viable,
    )
    if drop_cells:
        logger.info("DROP cells: %s", drop_cells)
    logger.info("Wrote %s + %s", pass_path, status_path)

    if below_min_viable:
        msg = (
            f"Elicitation gate ABORT: only {n_viable} viable cells "
            f"< {args.min_viable} required to proceed. "
            f"Dropped: {drop_cells}. "
            f"Re-author + re-elicit these cells (rerun "
            f"issue467_author_strong_nl.py + issue467_elicitation_check.py for "
            f"them) OR re-scope the experiment with the planner before "
            f"continuing to the strong-NL cosine/JS sweeps."
        )
        logger.error(msg)
        if args.fail_below_min_viable:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
