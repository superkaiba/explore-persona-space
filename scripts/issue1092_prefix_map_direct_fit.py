"""#1092 inline: direct averaged-grain fit of the prefix->answer map.

The banked fair-comparison read (`inline_fair_comparison/fair_comparison.json`)
operationalized the averaged-grain context arm as the query-average of the
per-row context map's held-out predictions (= the per-context-fit operator
applied to v_bar_C, by linearity). This script adds the missing arm the
user's writeup methodology names explicitly: FIT the map at averaged grain —
X = per-prefix query-averaged context vectors (v_bar_C), Y = per-prefix
averaged answer profiles — under the same battery-excluded rows, prefix
grouping, novel-prefix 6-fold scheme, and PRESS-ridge engine as every banked
read. Banked comparators are copied into the output JSON for the figure.

Usage: uv run python scripts/issue1092_prefix_map_direct_fit.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import issue1092_inline_fair_comparison as fc  # noqa: E402
from issue1092_fit_grid import (  # noqa: E402
    _basis_targets_with_info,
    _fit_cv,
    _folds_from_manifest,
)

OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison"
OUT_PATH = OUT / "prefix_map_direct_fit.json"


def process_cell(cell: str, rows: list[dict], banked_cell: dict) -> dict:
    prefix_all = fc._load(cell, "prefix_end")
    context_all = fc._load(cell, "context_end")
    t_all = [fc._load(cell, t) for t in fc.TARGETS]
    n0 = min(prefix_all.shape[0], context_all.shape[0], min(t.shape[0] for t in t_all), len(rows))
    be_idx = np.asarray(
        [
            i
            for i in range(n0)
            if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
        ],
        dtype=np.int64,
    )
    prefix_ids = np.asarray([rows[int(i)].get("prefix_id", "") for i in be_idx])
    X_context = np.asarray(context_all[be_idx], dtype=np.float64)
    Y_stacked = np.concatenate([np.asarray(t[be_idx], dtype=np.float64) for t in t_all], axis=1)
    del prefix_all, context_all, t_all

    groups = fc._prefix_groups(prefix_ids, fc.MIN_ROWS_PER_PREFIX)
    pids = sorted(groups)
    pseudo_rows = [{"prefix_id": p} for p in pids]
    folds_avg = _folds_from_manifest(
        pseudo_rows, len(pseudo_rows), group_key="prefix_id", n_folds=fc.N_FOLDS
    )
    Xc_avg = np.stack([X_context[groups[p]].mean(0) for p in pids], axis=0)

    out: dict = {"n_prefixes": len(pids), "bases": {}}
    for basis in fc.BASES:
        Yb = _basis_targets_with_info(
            Y_stacked, basis, hidden_dim=fc.HIDDEN_DIM, targets=fc.TARGETS, projection_target="t1"
        )[0]
        Y_avg = np.stack(
            [np.ascontiguousarray(Yb[groups[p]], dtype=np.float64).mean(0) for p in pids], axis=0
        )
        t0 = time.monotonic()
        fit = _fit_cv(np.ascontiguousarray(Xc_avg), np.ascontiguousarray(Y_avg), folds_avg)
        banked_basis = banked_cell["bases"][basis]
        out["bases"][basis] = {
            "r2_prefix_map_direct_fit_avg": float(fit["r2"]),
            "lambda_indices": fit.get("lambda_indices"),
            "fit_seconds": round(time.monotonic() - t0, 2),
            # banked comparators copied verbatim from fair_comparison.json
            "banked_r2_context_averaged_eval": banked_basis["averaged_grain"][
                "r2_context_averaged"
            ],
            "banked_r2_context_single_grain": banked_basis["single_grain"][
                "r2_context_battery_excluded_full"
            ],
            "banked_r2_prefix_end_averaged": banked_basis["averaged_grain"]["r2_prefix_averaged"],
        }
        b = out["bases"][basis]
        print(
            f"[{cell}/{basis}] direct-fit v_bar_C -> avg profile R2 = "
            f"{b['r2_prefix_map_direct_fit_avg']:.4f} "
            f"(banked ctx-averaged-eval {b['banked_r2_context_averaged_eval']:.4f}, "
            f"ctx single {b['banked_r2_context_single_grain']:.4f}) "
            f"[{b['fit_seconds']}s]",
            flush=True,
        )
    return out


def main() -> None:
    banked = json.loads((OUT / "fair_comparison.json").read_text())
    rows = fc._jsonl(fc.MANIFEST)
    result = {
        "meta": {
            "script": "scripts/issue1092_prefix_map_direct_fit.py",
            "git_commit": fc._git_sha(),
            "layer": fc.LAYER,
            "definition": (
                "prefix map = ridge fit FROM per-prefix query-averaged context vectors "
                "(v_bar_C) TO per-prefix averaged answer profiles, battery-excluded rows, "
                "min 3 rows/prefix, novel-prefix 6-fold, PRESS-ridge engine (same as banked)"
            ),
            "banked_source": "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json",
        },
        "cells": {},
    }
    for cell in fc.CELLS:
        result["cells"][cell] = process_cell(cell, rows, banked["cells"][cell])
    OUT_PATH.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
