#!/usr/bin/env python3
"""Task #2330 free-analysis follow-up: cross-LAYER paired-bootstrap contrasts.

Thin extension of ``scripts/issue2330_contrasts.py`` (P4) over the SAME banked
per-context ridge predictions (``data/issue_2330/preds/<cell>_test_preds_ridge.npz``).
The parent computed cross-CELL contrasts at depth-matched layer pairs only; this
script adds, per n-regime (5k / 10k), raw AND ceiling-normalized:

- **best-vs-best**: 9B at its best-captured layer minus 7B at its best-captured
  layer ("best" = argmax of full-sample test R² over the cell's 3 captured
  layers — a 3-level free axis; the within-model contrasts below expose the
  selection margin). At 10k: 7B-L19 (0.705) vs 9B-L16 (0.668).
- **full 3×3 cross-model grid**: Δ = 9B@l9 − 7B@l7 for every (l7 ∈ {14,19,26},
  l9 ∈ {16,22,30}) pair; the depth-matched diagonal replicates the parent's
  primary/secondary contrasts (asserted equal — see parity check).
- **within-model layer pairs**: deeper − shallower for the 3 layer pairs of
  each model (CIs on the best-layer margins, e.g. 9B L22−L16).

Method is inherited verbatim: paired bootstrap over the shared pinned 1,000
test contexts, ONE resample matrix (seed 42, 1,000 draws) shared across every
(cell, layer), SST recomputed against each draw's own mean, ceilings treated as
FIXED per-(model, layer) scalars. All parent validations run via
``_load_cells`` (committed-R² parity tol 1e-6, identical test context ids,
same-model target identity, #2130 n_pairs ceiling defense).

Fail-loud parity check: the (19, 22) grid cells must match the parent's
committed ``contrasts.json`` primary/registered raw contrasts to 1e-9.

Writes: ``eval_results/issue_2330/crosslayer/crosslayer_contrasts.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from issue2330_contrasts import (  # noqa: E402
    CELLS,
    MODEL_OF,
    N_BOOT,
    SEED,
    _load_cells,
    _r2_boot,
    contrast,
)

logger = logging.getLogger("issue2330_crosslayer")

REPO_ROOT = _SCRIPTS.parent

# (7B cell, 9B cell) per n-regime.
REGIMES = {"n5k": ("q25_n5k", "q35_n5k"), "n10k": ("q25_n10k", "q35_n10k")}
LAYERS_7B = [14, 19, 26]
LAYERS_9B = [16, 22, 30]
# Depth-fraction-matched (7B, 9B) pairs — the parent's diagonal.
MATCHED_PAIRS = [(14, 16), (19, 22), (26, 30)]
PARITY_TOL = 1e-9


def _key(cell: str, layer: int) -> str:
    return f"{cell}_L{layer}"


def _assert_parent_parity(grid: dict, parent_json: Path) -> dict:
    """Assert the depth-matched (19, 22) grid cells equal the parent's committed
    raw contrasts (same seed + shared resample matrix ⇒ identical computation).

    Returns a small record of the checked values; raises on any mismatch.
    """
    parent = json.loads(parent_json.read_text(encoding="utf-8"))
    checks = []
    for regime, parent_rec in (
        ("n10k", parent["primary_contrast_raw"]),
        ("n5k", parent["primary_layers"]["raw"][1]),  # 9B@5k − 7B@5k at (19, 22)
    ):
        ours = grid[regime]["raw"]["l7=19,l9=22"]
        for field in ("delta_full", "boot_mean"):
            d = abs(ours[field] - parent_rec[field])
            if d > PARITY_TOL:
                raise RuntimeError(
                    f"parent parity FAIL ({regime} {field}): ours={ours[field]} "
                    f"parent={parent_rec[field]} |Δ|={d:.3g} > {PARITY_TOL}"
                )
        for i in range(2):
            d = abs(ours["ci95"][i] - parent_rec["ci95"][i])
            if d > PARITY_TOL:
                raise RuntimeError(
                    f"parent parity FAIL ({regime} ci95[{i}]): ours={ours['ci95'][i]} "
                    f"parent={parent_rec['ci95'][i]} |Δ|={d:.3g} > {PARITY_TOL}"
                )
        checks.append({"regime": regime, "delta_full": ours["delta_full"], "status": "match"})
    return {"tol": PARITY_TOL, "checked": checks}


def run(args) -> int:
    data = _load_cells(args.fits_dir, args.preds_dir)
    n = data[CELLS[0]]["n_test"]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))  # ONE shared resample matrix

    # ---- boots + fulls + ceilings per (cell, layer), string-keyed -----------
    boots: dict[str, np.ndarray] = {}
    fulls: dict[str, float] = {}
    ceils: dict[str, float] = {}
    for cell in CELLS:
        for layer in data[cell]["layers"]:
            d = data[cell]["per_layer"][layer]
            t0 = time.time()
            boots[_key(cell, layer)] = _r2_boot(d["pred"], d["y"], idx)
            fulls[_key(cell, layer)] = d["r2_full"]
            ceils[_key(cell, layer)] = d["ceiling"]
            print(
                f"[crosslayer] boots {cell} L{layer}: {N_BOOT} draws in {time.time() - t0:.2f}s",
                flush=True,
            )
    norm_boots = {k: boots[k] / ceils[k] for k in boots}
    norm_fulls = {k: fulls[k] / ceils[k] for k in fulls}

    def _contrast_pair(a: str, b: str, label: str) -> dict[str, dict]:
        return {
            "raw": contrast(a, b, label, boots, fulls, N_BOOT),
            "normalized": contrast(a, b, label, norm_boots, norm_fulls, N_BOOT),
        }

    # ---- best-captured layer per cell (argmax full-sample R² over 3 layers) -
    best_layer = {
        cell: max(data[cell]["layers"], key=lambda ell: data[cell]["per_layer"][ell]["r2_full"])
        for cell in CELLS
    }
    print(f"[crosslayer] best layers: {best_layer}", flush=True)

    out_regimes: dict[str, dict] = {}
    for regime, (c7, c9) in REGIMES.items():
        b7, b9 = best_layer[c7], best_layer[c9]
        rec = _contrast_pair(
            _key(c7, b7),
            _key(c9, b9),
            f"best-vs-best ({regime}): 9B@L{b9} − 7B@L{b7}",
        )
        best_vs_best = {
            "layer_7b": b7,
            "layer_9b": b9,
            "r2_full_7b_best": fulls[_key(c7, b7)],
            "r2_full_9b_best": fulls[_key(c9, b9)],
            **rec,
        }

        grid_raw: dict[str, dict] = {}
        grid_norm: dict[str, dict] = {}
        for l7 in LAYERS_7B:
            for l9 in LAYERS_9B:
                gkey = f"l7={l7},l9={l9}"
                pair_rec = _contrast_pair(
                    _key(c7, l7), _key(c9, l9), f"{regime}: 9B@L{l9} − 7B@L{l7}"
                )
                pair_rec["raw"]["depth_matched"] = (l7, l9) in MATCHED_PAIRS
                grid_raw[gkey] = pair_rec["raw"]
                grid_norm[gkey] = pair_rec["normalized"]

        within: dict[str, dict] = {}
        for cell, layers in ((c7, LAYERS_7B), (c9, LAYERS_9B)):
            for i in range(len(layers)):
                for j in range(i + 1, len(layers)):
                    a_l, b_l = layers[i], layers[j]
                    wkey = f"{MODEL_OF[cell]}: L{b_l} − L{a_l}"
                    within[wkey] = _contrast_pair(
                        _key(cell, a_l), _key(cell, b_l), f"{regime} {wkey} (deeper − shallower)"
                    )

        out_regimes[regime] = {
            "cells": {"qwen25_7b": c7, "qwen35_9b": c9},
            "best_vs_best": best_vs_best,
            "cross_model_grid": {"raw": grid_raw, "normalized": grid_norm},
            "within_model": within,
        }

    parity = _assert_parent_parity(
        {r: {"raw": out_regimes[r]["cross_model_grid"]["raw"]} for r in out_regimes},
        args.parent_json,
    )
    print(f"[crosslayer] parent parity: {json.dumps(parity)}", flush=True)

    out = {
        "dv": (
            "ridge held-out variance-weighted test R² (matched 1,000-prompt LMSYS test split), "
            "contrasted ACROSS captured layers: {7B L14/19/26} × {9B L16/22/30}, per n-regime"
        ),
        "method": (
            "paired bootstrap over the shared pinned test contexts; ONE resample matrix "
            f"(seed {SEED}, {N_BOOT} draws) shared across every (cell, layer); SST recomputed "
            "against each draw's own mean; ceilings = fixed per-(model, layer) two-draw scalars "
            "(all 12 present in the committed P3 fits JSONs — normalized grid computed, "
            "no raw-only fallback needed)"
        ),
        "best_layer_selection": (
            "argmax of full-sample test R² over the cell's 3 captured layers (a 3-level free "
            "axis; within_model contrasts carry the CIs on the selection margins)"
        ),
        "preds_provenance": (
            f"{args.preds_dir}/<cell>_test_preds_ridge.npz (P3 matched-fits outputs; all parent "
            "validations re-run via issue2330_contrasts._load_cells)"
        ),
        "n_test": int(n),
        "n_boot": N_BOOT,
        "seed": SEED,
        "best_layer": best_layer,
        "r2_full": {k: fulls[k] for k in sorted(fulls)},
        "ceilings": {k: ceils[k] for k in sorted(ceils)},
        "parent_parity_check": parity,
        "regimes": out_regimes,
        "metadata": {
            **as_metadata_dict(git_provenance(REPO_ROOT)),
            "numpy_version": np.__version__,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/issue2330_crosslayer_contrasts.py",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    tmp.replace(args.out_json)
    print(f"[crosslayer] wrote {args.out_json}", flush=True)
    for regime in out_regimes:
        bb = out_regimes[regime]["best_vs_best"]
        print(f"best-vs-best {regime} raw:", json.dumps(bb["raw"]), flush=True)
        print(f"best-vs-best {regime} norm:", json.dumps(bb["normalized"]), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Task #2330 follow-up: cross-layer paired-bootstrap contrasts"
    )
    ap.add_argument(
        "--fits-dir",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_2330",
        help="dir holding matched_fits_<cell>.json (P3 outputs)",
    )
    ap.add_argument(
        "--preds-dir",
        type=Path,
        default=REPO_ROOT / "data" / "issue_2330" / "preds",
        help="dir holding <cell>_test_preds_ridge.npz (P3 outputs)",
    )
    ap.add_argument(
        "--parent-json",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_2330" / "contrasts.json",
        help="parent P4 contrasts.json (parity-check reference)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT
        / "eval_results"
        / "issue_2330"
        / "crosslayer"
        / "crosslayer_contrasts.json",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
