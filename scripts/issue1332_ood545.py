"""Issue #1332 — #545 OOD arm: score map-similarity under the FROZEN protocol.

Plan v3 §4.7 item 6 + H3. Consumes the P2 ``similarity545`` matrices, builds a
#545-format predictor cells file S(row, col) = S_sym(row-unit, col-unit), runs
the FEASIBILITY GATE (per-behavior split-half map reliability vs a shuffled-Y
refit null — "underpowered at corpus n" is a decidable, reportable outcome),
and scores the predictor with the frozen #545 machinery VERBATIM: targets =
``_seed_mean_targets(include_flagged=False)`` z-normed within column, dev
cells = the preregistered quarantine split's development cells (119 unflagged;
the 52-cell quarantine is NEVER scored — plan must-ask), tau =
``scoring.weighted_kendall_tau``, folds = leave-one-behavior-family-out.

USAGE
    uv run python scripts/issue1332_ood545.py --full
    uv run python scripts/issue1332_ood545.py --smoke --n-null 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1332_common as C

logger = logging.getLogger("issue1332.ood545")

FEASIBILITY_NULL_DRAWS = 20


def feasibility_gate(store_dir: Path, units: list[str], layer: int, *, n_draws: int) -> dict:
    """Per-unit split-half own-map R^2 vs a shuffled-Y refit null (plan §4.7.6).

    A unit whose split-half reliability does not exceed its own shuffled-pairing
    null p95 is UNDERPOWERED at corpus n (reported, never silently scored).
    Null refits are BATCHED over draws via the layer-batched wrapper (the draw
    axis rides the batch axis — vectorize-first).
    """
    import numpy as np
    import torch

    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict_fast_layer_batched,
    )

    out = {}
    rng = np.random.default_rng(C.SPLIT_HALF_SEED)
    for unit in units:
        sh = torch.load(store_dir / f"{unit}.pt", map_location="cpu", mmap=True, weights_only=False)
        X = sh["cx_last"][:, layer, :].float().numpy()
        Y = sh["v_mean"][:, layer, :].float().numpy()
        n = X.shape[0]
        perm = rng.permutation(n)
        a, b = sorted(perm[: n // 2].tolist()), sorted(perm[n // 2 :].tolist())

        def _r2(y_true, y_pred):
            mu = y_true.mean(axis=0)
            return 1.0 - float(((y_true - y_pred) ** 2).sum()) / (
                float(((y_true - mu) ** 2).sum()) + 1e-12
            )

        pred = ridge_fit_predict_fast_layer_batched(X[a][None], Y[a][None], X[b][None])[0]
        own = _r2(Y[b], pred)
        # shuffled-Y refit null, draws batched on the leading axis
        Xa = np.repeat(X[a][None], n_draws, axis=0)
        Ya = np.stack([Y[a][rng.permutation(len(a))] for _ in range(n_draws)], axis=0)
        Xb = np.repeat(X[b][None], n_draws, axis=0)
        null_preds = ridge_fit_predict_fast_layer_batched(Xa, Ya, Xb)
        null_r2 = np.asarray([_r2(Y[b], null_preds[d]) for d in range(n_draws)])
        p95 = float(np.quantile(null_r2, 0.95))
        out[unit] = {
            "split_half_r2": own,
            "null_p95": p95,
            "n_rows": int(n),
            "powered": bool(own > p95),
        }
        logger.info("[gate] %s own=%.4f null_p95=%.4f powered=%s", unit, own, p95, own > p95)
    return out


def main() -> int:
    """OOD driver: cells file -> feasibility gate -> frozen #545 tau scoring."""
    ap = argparse.ArgumentParser(description="Issue #1332 #545 OOD-arm scoring")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--n-null", type=int, default=FEASIBILITY_NULL_DRAWS)
    ap.add_argument(
        "--skip-frozen-scoring",
        action="store_true",
        help="cells + feasibility only (when #545 artifacts are not staged)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res_dir = C.results_dir(args.smoke, args.results_dir)
    store = C.data_root(args.smoke, args.out_root) / "store" / "capture545"

    # Layer routing (r1 Major 2): open the FROZEN L*'s file directly — a
    # lexicographic sorted(glob)[-1] picks L14 over L* for L* in {0,1,10-13}.
    sim_dir = res_dir / "similarity545"
    freeze_path = res_dir / "layer_freeze.json"
    if freeze_path.exists():
        l_star = json.loads(freeze_path.read_text())["l_star"]
        sim_path = sim_dir / f"S_transfer_L{l_star}.json"
        if not sim_path.exists():
            raise FileNotFoundError(
                f"{sim_path} missing for frozen L*={l_star} — re-run "
                f"issue1332_fits.py --arm i545 after the layer freeze"
            )
    elif args.full:
        raise FileNotFoundError(
            f"{freeze_path} missing — run issue1332_fits.py (marker arm) first; "
            f"refusing a fallback layer in --full"
        )
    else:  # smoke-only fallback: NUMERIC layer sort, never lexicographic
        sim_files = sorted(
            sim_dir.glob("S_transfer_L*.json"), key=lambda p: int(p.stem.rsplit("L", 1)[1])
        )
        if not sim_files:
            raise FileNotFoundError(
                f"no similarity545 outputs under {res_dir} — run issue1332_fits.py --arm i545"
            )
        sim_path = sim_files[-1]
    sim = json.loads(sim_path.read_text())
    units, layer = sim["units"], sim["layer"]
    import numpy as np

    S = np.asarray(sim["S_sym"], dtype=float)
    row_units = [u for u in units if u.startswith("row__")]
    col_units = [u for u in units if u.startswith("col__")]

    C.phase("ood545_gate")
    gate = feasibility_gate(store, units, layer, n_draws=args.n_null)
    powered_rows = [u for u in row_units if gate[u]["powered"]]

    C.phase("ood545_cells")
    ui = {u: i for i, u in enumerate(units)}
    cells: dict[str, float] = {}
    for ru in row_units:
        for cu in col_units:
            row_id = ru.removeprefix("row__")
            col_id = cu.removeprefix("col__")
            cells[f"{row_id}|{col_id}"] = float(S[ui[ru], ui[cu]])
    cells_payload = {
        "cells": cells,
        "layer": layer,
        "predictor": "i1332_map_transfer_similarity",
        "feasibility_gate": gate,
        "n_powered_rows": len(powered_rows),
        "coverage_note": (
            "rows limited to train_lora corpora resolvable on HF; columns to "
            "diagonal demo pools incl. frozen-protocol regens (r3 add-on, plan "
            "assumption 5 primary leg); Turner-gated + pending-p1 pools stay "
            "descoped (reasons logged at staging)"
        ),
        "reproducibility_metadata": C.reproducibility_metadata({"smoke": args.smoke}),
    }
    C.write_json_atomic(res_dir / "ood545" / "predictor_cells.json", cells_payload)

    if args.skip_frozen_scoring:
        C.phase("done_ood545")
        return 0

    C.phase("ood545_frozen_scoring")
    from explore_persona_space.experiments.behavior_testbed_545 import output_root
    from explore_persona_space.experiments.behavior_testbed_545.rows import ROWS
    from explore_persona_space.experiments.behavior_testbed_545.scoring import (
        _families_of,
        _seed_mean_targets,
        _z_norm_within_column,
        weighted_kendall_tau,
    )

    root_545 = output_root()
    prereg = json.loads((root_545 / "preregistration.json").read_text())
    matrix = json.loads((root_545 / "L_matrix.json").read_text())["cells"]
    metadata = json.loads((root_545 / "cell_metadata.json").read_text())["cells"]
    dev_cells = ["|".join(c) for c in prereg["quarantine_split"]["development_cells"]]
    targets_raw = _seed_mean_targets(matrix, metadata, include_flagged=False)
    vals = {k: v["shift"] for k, v in targets_raw.items() if v.get("shift") is not None}
    target = _z_norm_within_column(vals)
    dev = [c for c in dev_cells if c in target]
    tau_all = weighted_kendall_tau(cells, target, dev)
    fams_of = _families_of([c for c in dev if c in cells])
    fold_taus = {}
    for fam, fam_cells in sorted(fams_of.items()):
        held = [c for c in dev if c in fam_cells]
        fold_taus[fam] = weighted_kendall_tau(cells, target, held)
    n_scored = sum(1 for c in dev if c in cells)
    scoring = {
        "protocol": "frozen #545 (dev-only, 119 unflagged; quarantine NEVER scored)",
        "tau_dev": tau_all,
        "n_dev_cells_scored": n_scored,
        "n_dev_cells_total": len(dev),
        "per_family_fold_tau": fold_taus,
        "families_lfo": sorted(fams_of),
        "row_family_map": {r.row_id: r.family for r in ROWS.values()},
        "feasibility_powered_rows": powered_rows,
    }
    C.write_json_atomic(res_dir / "ood545" / "frozen_scoring.json", scoring)
    logger.info("[ood545] tau_dev=%s over %d/%d dev cells", tau_all, n_scored, len(dev))
    C.phase("done_ood545")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
