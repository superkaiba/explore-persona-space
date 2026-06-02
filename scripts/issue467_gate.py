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
    aborted-drop tally — usually means the author step FAIL_LEAK'd or
    FAIL_LENGTH'd them upstream).

Abort criterion (plan §6.2 kill rule):
  - If more than ``--max-drops`` (default 5) cells drop, the gate exits
    non-zero AFTER writing the gate_status.json — the launcher's ``set
    -e`` then aborts the run BEFORE the (now-untrustworthy) strong-NL
    sweeps. The threshold is the plan-stated kill criterion; raise via
    ``--max-drops N`` only when the planner approves a re-scope.

Usage:
    uv run python scripts/issue467_gate.py \
        --pairs aesthetic_popular aesthetic_unpopular ... \
        [--max-drops 5]

Exits 0 on PASS-or-acceptable-drop; non-zero on >max-drops drops or
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
                "likely FAIL_LEAK'd or FAIL_LENGTH'd this cell upstream, "
                "or the elicitation step crashed for it."
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
        "--max-drops",
        type=int,
        default=5,
        help=(
            "Plan §6.2 kill criterion: abort the run (exit non-zero) if more than "
            "this many cells DROP. Default 5 (the planner-stated bar)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(GATE_DIR),
        help="Where to write pass_cells.txt + gate_status.json (default data/issue467/gate).",
    )
    parser.add_argument(
        "--fail-on-too-many-drops",
        action="store_true",
        default=True,
        help=(
            "Exit non-zero when drops > max-drops (default ON — the launcher's set -e "
            "is what stops the run). Disable for analysis-only invocations."
        ),
    )
    parser.add_argument(
        "--no-fail-on-too-many-drops",
        dest="fail_on_too_many_drops",
        action="store_false",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cell: list[dict] = [classify_cell(p) for p in args.pairs]
    pass_cells = [c["pair"] for c in per_cell if c["verdict"] == "PASS"]
    drop_cells = [c["pair"] for c in per_cell if c["verdict"] == "DROP"]

    summary = {
        "n_pairs_dispatched": len(args.pairs),
        "n_pass": len(pass_cells),
        "n_drop": len(drop_cells),
        "max_drops_threshold": args.max_drops,
        "exceeded_kill_criterion": len(drop_cells) > args.max_drops,
        "pass_cells": pass_cells,
        "drop_cells": drop_cells,
        "per_cell": per_cell,
    }

    pass_path = out_dir / "pass_cells.txt"
    pass_path.write_text("\n".join(pass_cells) + ("\n" if pass_cells else ""))
    status_path = out_dir / "gate_status.json"
    status_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Gate: %d PASS / %d DROP of %d (threshold: drop > %d aborts)",
        len(pass_cells),
        len(drop_cells),
        len(args.pairs),
        args.max_drops,
    )
    if drop_cells:
        logger.info("DROP cells: %s", drop_cells)
    logger.info("Wrote %s + %s", pass_path, status_path)

    if summary["exceeded_kill_criterion"]:
        msg = (
            f"Elicitation gate ABORT (plan §6.2 kill criterion): "
            f"{len(drop_cells)} cells dropped > {args.max_drops} threshold. "
            f"Dropped: {drop_cells}. "
            f"Re-author + re-elicit these cells (rerun "
            f"issue467_author_strong_nl.py + issue467_elicitation_check.py for "
            f"them) OR re-scope the experiment with the planner before "
            f"continuing to the strong-NL cosine/JS sweeps."
        )
        logger.error(msg)
        if args.fail_on_too_many_drops:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
