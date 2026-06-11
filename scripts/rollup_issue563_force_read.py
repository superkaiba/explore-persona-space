#!/usr/bin/env python3
"""Issue #563 follow-up `fixed-completion-force-read` rollup (VM, CPU-only).

Reads the 5 force-read slot files under
``eval_results/issue_563/fixed-completion-force-read/`` plus the committed v1
``eval_results/issue_563/rollup.json`` (the own-content comparison arm) and
writes ``<out-dir>/rollup.json`` with (plan v2 sections 3 / 5):

  1. Paired per-row deltas vs the assistant diagonal, all three spaces + the
     logZ decomposition (v1 machinery, imported).
  2. Row-level paired bootstrap: 10,000 resamples, seed 563, ONE shared
     index draw across cells; sign counts + Wilson CI.
  3. Pre-registered 3-way call per role cell vs 0.5 x R'_c (R'_c recomputed
     from the committed rollup.json and asserted vs plan-quoted +-0.01):
     PROMPT-DIRECT / CONTENT-MEDIATED / INDETERMINATE; panel verdict >= 3/4
     (2/4 is MIXED, never rounded up); threshold-sensitive straddle flags;
     a credibly-negative cell additionally carries ``sign_reversed: true``.
  4. Side-by-side block: fixed-content rise vs own-content rise per cell
     (own-content CI under the SAME shared draw when n matches).
  5. [0:50] parity subset; per-cell decomposition incl. the logZ share;
     per-cell Spearman(fixed-content delta, own-content delta) (exploratory).

Usage (VM, after pod termination; CPU-only):
    uv run python scripts/rollup_issue563_force_read.py
    uv run python scripts/rollup_issue563_force_read.py \\
        --results-dir /tmp/fixture --expect-n 20 --n-resamples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="rollup_issue563_force_read")

from _issue543_common import repro_metadata  # noqa: E402
from eval_issue563_base_panel import N_PANEL_PROMPTS_563  # noqa: E402
from eval_issue563_force_read import (  # noqa: E402
    COMMITTED_ROLLUP,
    DIAGONAL_CELL,
    ISSUE_563,
    OUT_DIR,
    R_PRIME_C_EXPECTED,
    ROLE_CELLS,
    SLOT_KEY,
    recompute_own_content_rises,
)
from rollup_issue563_base_panel import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_RESAMPLES,
    HALF_EFFECT_FRACTION,
    N_SUBSET,
    classify_cell,
    delta_stats,
    paired_delta_arrays,
    panel_verdict,
    shared_bootstrap_idx,
)

log = logging.getLogger("rollup_issue563_force_read")

# Registered relabels (plan v2 section 3): same classification machinery as
# v1, renamed to the follow-up's registered vocabulary.
LABEL_MAP = {
    "REPRODUCES": "PROMPT-DIRECT",
    "FLAT": "CONTENT-MEDIATED",
    "INDETERMINATE": "INDETERMINATE",
}
VERDICT_MAP = {
    "intrinsic-context": "prompt-direct",
    "completion-content": "content-mediated",
    "mixed": "mixed",
}


def load_force_read_cell(results_dir: Path, cell: str, *, expect_n: int) -> list[dict]:
    """One force-read slot file; n + finiteness + identity asserted."""
    path = results_dir / f"slot_stats_force_read_{cell}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing force-read slot file: {path}")
    slot = json.loads(path.read_text())
    rows = slot[SLOT_KEY]
    if not (slot["n"] == len(rows) == expect_n):
        raise RuntimeError(f"{path}: n={slot['n']}, rows={len(rows)}, expected {expect_n}")
    for i, row in enumerate(rows):
        if not all(math.isfinite(v) for v in row.values()):
            raise RuntimeError(f"Non-finite slot row {cell}[{i}]: {row}")
    return rows


def own_content_deltas(cell: str) -> np.ndarray:
    """The committed v1 per-row own-content deltas for ``cell`` (n=250).

    Lives at the CELL level of the committed rollup
    (``panel.cells.<cell>.per_question_d_logp``), NOT under ``d_logp``.
    """
    cells = json.loads(COMMITTED_ROLLUP.read_text())["panel"]["cells"]
    return np.asarray(cells[cell]["per_question_d_logp"], dtype=float)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rank correlation (exploratory); None on degenerate input."""
    if len(x) < 3 or np.allclose(x.std(), 0) or np.allclose(y.std(), 0):
        return None
    from scipy.stats import spearmanr

    rho = spearmanr(x, y).statistic
    return float(rho) if math.isfinite(rho) else None


def decomposition_block(d: dict[str, np.ndarray], idx: np.ndarray) -> dict:
    """All-three-spaces stats + the logZ share of the rise (v1: 58-94%)."""
    mean_logp = float(d["d_logp"].mean())
    mean_logz = float(d["d_logz"].mean())
    return {
        "d_eos_margin": delta_stats(d["d_eosm"], idx),
        "d_z_marker": delta_stats(d["d_zm"], idx),
        "d_logZ": delta_stats(d["d_logz"], idx),
        "logz_share_of_rise": (-mean_logz / mean_logp) if abs(mean_logp) > 1e-9 else None,
    }


def run_rollup(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    n = args.expect_n

    # R'_c recompute + assert (+-0.01 stale-file guard; plan section 5).
    r_prime_c = recompute_own_content_rises()
    log.info("R'_c (asserted vs plan +-0.01): %s", {k: round(v, 4) for k, v in r_prime_c.items()})

    rows: dict[str, list[dict]] = {}
    for cell in (DIAGONAL_CELL, *ROLE_CELLS):
        rows[cell] = load_force_read_cell(results_dir, cell, expect_n=n)

    idx = shared_bootstrap_idx(n, n_resamples=args.n_resamples, seed=args.bootstrap_seed)
    n_sub = min(N_SUBSET, n)
    idx_subset = shared_bootstrap_idx(n_sub, n_resamples=args.n_resamples, seed=args.bootstrap_seed)

    cells_out: dict[str, dict] = {}
    raw_labels: dict[str, str] = {}
    raw_labels_subset: dict[str, str] = {}
    for cell in ROLE_CELLS:
        d = paired_delta_arrays(rows[DIAGONAL_CELL], rows[cell])
        stats_logp = delta_stats(d["d_logp"], idx)
        cls = classify_cell(stats_logp, r_prime_c[cell])
        raw_labels[cell] = cls["label"]
        # Registered relabel + the sign-reversed flag (critic concern: a
        # credibly-negative cell is labeled distinctly, on top of the
        # INDETERMINATE/CONTENT-MEDIATED machinery).
        cls = {
            **cls,
            "label": LABEL_MAP[cls["label"]],
            "label_v1_machinery": raw_labels[cell],
            "r_prime_c_own_content": cls.pop("r_c_parent"),
            "sign_reversed": bool(stats_logp["ci95"][1] < 0),
        }

        # [0:50] parity subset (the parent-parity slice R'_c's parent rule
        # was calibrated on).
        sub = d["d_logp"][:n_sub]
        stats_sub = delta_stats(sub, idx_subset)
        cls_sub = classify_cell(stats_sub, r_prime_c[cell])
        raw_labels_subset[cell] = cls_sub["label"]
        cls_sub = {
            **cls_sub,
            "label": LABEL_MAP[cls_sub["label"]],
            "r_prime_c_own_content": cls_sub.pop("r_c_parent"),
            "sign_reversed": bool(stats_sub["ci95"][1] < 0),
        }

        # Side-by-side vs the own-content arm. The committed per-row deltas
        # are index-aligned with the force-read rows by construction (both
        # are the [0:n] slice of the same question list, same order).
        own = own_content_deltas(cell)
        own_n = own[:n]
        shared = len(own) == n
        own_stats = delta_stats(own if shared else own_n, idx)
        side_by_side = {
            "fixed_content_rise": stats_logp,
            "own_content_rise": own_stats,
            "own_content_ci_under_shared_draw": shared,
            "fixed_minus_own_mean": stats_logp["mean"] - float(own_n.mean()),
            "fraction_of_own_content_rise": (
                stats_logp["mean"] / float(own_n.mean())
                if abs(float(own_n.mean())) > 1e-9
                else None
            ),
        }

        cells_out[cell] = {
            "d_logp": stats_logp,
            "classification": cls,
            **decomposition_block(d, idx),
            "subset_0_50": {
                "d_logp": stats_sub,
                "classification": cls_sub,
                "subset_mean_outside_full_ci": not (
                    stats_logp["ci95"][0] <= stats_sub["mean"] <= stats_logp["ci95"][1]
                ),
            },
            "side_by_side": side_by_side,
            "spearman_fixed_vs_own_d_logp": spearman_rho(d["d_logp"], own_n),
            "per_row_d_logp": [float(v) for v in d["d_logp"]],
        }

    # Panel verdict on the v1 machinery's raw labels (>= 3/4; 2/4 is MIXED),
    # then relabeled to the registered vocabulary.
    verdict = panel_verdict(raw_labels, ROLE_CELLS)
    verdict["verdict"] = VERDICT_MAP[verdict["verdict"]]
    verdict_subset = panel_verdict(raw_labels_subset, ROLE_CELLS)
    verdict_subset["verdict"] = VERDICT_MAP[verdict_subset["verdict"]]
    labels = {c: LABEL_MAP[raw_labels[c]] for c in ROLE_CELLS}
    labels_subset = {c: LABEL_MAP[raw_labels_subset[c]] for c in ROLE_CELLS}

    diag_logp_mean = float(np.mean([r["logp"] for r in rows[DIAGONAL_CELL]]))
    rollup = {
        **repro_metadata(),
        "issue": ISSUE_563,
        "followup_label": "fixed-completion-force-read",
        "mode": "production" if n == N_PANEL_PROMPTS_563 else "reduced_n",
        "results_dir": str(results_dir),
        "n_rows": n,
        "bootstrap": {
            "n_resamples": args.n_resamples,
            "seed": args.bootstrap_seed,
            "unit": "row (paired; same fixed completion under both prompts)",
            "shared_draw_across_cells": True,
        },
        "registered_rule": {
            "half_effect_fraction": HALF_EFFECT_FRACTION,
            "r_prime_c_recomputed": r_prime_c,
            "r_prime_c_plan_quoted": R_PRIME_C_EXPECTED,
            "labels": "PROMPT-DIRECT / CONTENT-MEDIATED / INDETERMINATE vs 0.5 x R'_c",
        },
        "panel": {
            "baseline_cell": DIAGONAL_CELL,
            "cells": cells_out,
            "labels": labels,
            "verdict": verdict,
            "subset_0_50_labels": labels_subset,
            "subset_0_50_verdict": verdict_subset,
        },
        "diagonal": {"logp_mean": diag_logp_mean},
        "own_content_rollup": str(COMMITTED_ROLLUP),
    }
    out_path = Path(args.out) if args.out else Path(args.results_dir) / "rollup.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rollup, indent=2))
    log.info("Rollup -> %s (verdict=%s labels=%s)", out_path, verdict["verdict"], labels)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #563 fixed-completion force-read rollup (CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    p.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    p.add_argument(
        "--expect-n",
        type=int,
        default=N_PANEL_PROMPTS_563,
        help="Rows per cell (250 production; smaller for smoke/fixture artifacts).",
    )
    return p.parse_args()


def main() -> int:
    return run_rollup(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
