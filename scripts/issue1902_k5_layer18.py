#!/usr/bin/env python3
"""K=5 own-map fits at layer 18 for the OLMo-2-7B post-training chain (#1902).

Section 4.3 of the context-answer paper reports K=5 held-out R^2 at layer 31,
where the base model leads every post-trained stage. The older K=1 layer sweep
had the base model LOWEST at every layer from 16 to 31, so the natural question
is whether the K=5 ordering survives away from the final layer.

The K=5 capture stored two layers, 18 and 31 (the seed-42 store holds all 17).
Layer 18 is therefore the one additional layer answerable without re-capturing
activations, and it is the informative one: at K=1 it carried the largest base
deficit of any layer (B 0.305 against D 0.411).

Nothing about the fit is re-implemented here. The estimator
(``SharedPrimalRidge``), the fold assignment (``load_fold_of``) and the per-row
components are imported from the modules that produced the paper's layer-31
numbers, so the only difference from ``issue1902_k5_fits.run_grid`` is which
layer's tensors are loaded. ``--parity`` refits the base model at layer 31 and
must reproduce the published 0.674778 before any layer-18 number is read.

Usage::

    uv run python scripts/issue1902_k5_layer18.py --parity   # gate first
    uv run python scripts/issue1902_k5_layer18.py --layer 18
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # repo convention: environment before heavy imports

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))  # noqa: E402

import issue1902_lasttoken_comparison as LC  # noqa: E402
import issue1902_lasttoken_transfer as XF  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

STAGES = ("B", "S", "D", "R")
SEEDS = (42, 45, 46, 47, 48)  # the five draws behind the paper's K=5 targets
CORPUS = "single"
PARITY_LAYER = 31
PARITY_REFERENCE = {"B": 0.6747779066421795}  # committed k5_full_grid diagonal
PARITY_TOL = 1e-3
DEFAULT_STAGE_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue1902_l18")
DEFAULT_OUT = ROOT / "eval_results" / "issue_1902" / "k5_layer18"


def _log(msg: str) -> None:
    print(f"[k5-l18] {msg}", flush=True)


def _ctx_relpath(stage: str, layer: int) -> str:
    return f"{LC.HF_PREFIX}/{stage}/ctx/{CORPUS}/L{layer}.pt"


def _answer_relpath(stage: str, seed: int, layer: int) -> str:
    """Diagonal cell: context source and answer source are the same stage."""
    if seed == 42:
        return f"{LC.HF_PREFIX}/{stage}/{stage}/{CORPUS}/L{layer}.pt"
    return f"{LC.HF_PREFIX}/k5draws/{stage}/{stage}/{CORPUS}/seed{seed}/L{layer}.pt"


def _stage_file(relpath: str, stage_root: Path) -> Path:
    local = stage_root / relpath
    if local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    hub.stage_hub_file(
        LC.HF_REPO,
        relpath,
        local,
        repo_type="dataset",
        overwrite=True,
    )
    return local


def _load(path: Path, key: str) -> tuple[np.ndarray, list[str]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    return (
        payload[key].to(torch.float32).numpy(),
        [str(v) for v in payload["row_ids"]],
    )


def _align(w: np.ndarray, ids: list[str], rows: list[str]) -> np.ndarray:
    pos = {rid: i for i, rid in enumerate(ids)}
    missing = [r for r in rows if r not in pos]
    if missing:
        raise KeyError(f"{len(missing)} reference rows absent from shard, first: {missing[0]}")
    return w[[pos[r] for r in rows]]


def fit_stage(stage: str, layer: int, stage_root: Path) -> dict:
    """Own-map fit for one checkpoint: K=5 mean target, shared folds and ridge."""
    fold_of, ref_ids = XF.load_fold_of()
    n = len(ref_ids)

    x_raw, x_ids = _load(_stage_file(_ctx_relpath(stage, layer), stage_root), "u_last")
    x = _align(x_raw, x_ids, ref_ids)
    del x_raw

    acc = np.zeros((n, x.shape[1]), dtype=np.float32)
    for seed in SEEDS:
        w, ids = _load(_stage_file(_answer_relpath(stage, seed, layer), stage_root), "w")
        acc += _align(w, ids, ref_ids)
        del w
    y = acc / float(len(SEEDS))
    del acc

    res = np.full(n, np.nan)
    tot = np.full(n, np.nan)
    cos = np.full(n, np.nan)
    lam = np.full(XF.N_FOLDS, np.nan)
    dof = np.full(XF.N_FOLDS, np.nan)
    n_train_by_fold = np.zeros(XF.N_FOLDS, dtype=np.int64)

    for fold in range(XF.N_FOLDS):
        t0 = time.time()
        ev = fold_of == fold
        tr = ~ev
        ridge = XF.SharedPrimalRidge(x[tr])
        xev_std = ridge.standardize(x[ev])
        weights, ymu, info = ridge.fit(y[tr])
        pred = xev_std @ weights + ymu
        rr, tt, cc = LC._per_row_components(pred, y[ev], y[tr].mean(axis=0))
        res[ev], tot[ev], cos[ev] = rr, tt, cc
        lam[fold] = info["selected_lambda"]
        dof[fold] = info["dof"]
        n_train_by_fold[fold] = int(tr.sum())
        _log(
            f"{stage} L{layer} fold {fold}: n_train={int(tr.sum())} "
            f"lambda={info['selected_lambda']:.4g} dof={info['dof']:.1f} "
            f"({time.time() - t0:.1f}s)"
        )

    if np.isnan(res).any() or np.isnan(tot).any():
        raise RuntimeError(f"{stage} L{layer}: fold coverage left rows unscored")

    return {
        "_res": res,
        "_tot": tot,
        "stage": stage,
        "layer": layer,
        "n": n,
        "r2": float(1.0 - res.sum() / tot.sum()),
        "median_cos": float(np.median(cos)),
        "mean_ss_res": float(res.mean()),
        "mean_ss_tot": float(tot.mean()),
        "selected_lambda_by_fold": [float(v) for v in lam],
        "dof_by_fold": [float(v) for v in dof],
        "n_train_by_fold": [int(v) for v in n_train_by_fold],
        "d_features": int(x.shape[1]),
        "seeds": list(SEEDS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--stages", default="B S D R", help="space-separated checkpoint letters")
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--parity",
        action="store_true",
        help="refit base at layer 31 and check it reproduces the committed K=5 diagonal",
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.parity:
        cell = fit_stage("B", PARITY_LAYER, args.stage_root)
        cell.pop("_res", None)
        cell.pop("_tot", None)
        want = PARITY_REFERENCE["B"]
        delta = abs(cell["r2"] - want)
        verdict = "PASS" if delta <= PARITY_TOL else "FAIL"
        _log(
            f"parity B L{PARITY_LAYER}: got {cell['r2']:.6f} want {want:.6f} delta {delta:.2e} {verdict}"
        )
        (args.out / "parity.json").write_text(
            json.dumps(
                {
                    "verdict": verdict,
                    "got": cell["r2"],
                    "want": want,
                    "abs_delta": delta,
                    "tolerance": PARITY_TOL,
                    "cell": cell,
                    "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )
        if verdict == "FAIL":
            raise SystemExit(f"parity FAIL: {cell['r2']:.6f} vs {want:.6f}")
        return 0

    stages = tuple(args.stages.split())
    cells = {stage: fit_stage(stage, args.layer, args.stage_root) for stage in stages}
    ordering = sorted(cells, key=lambda s: -cells[s]["r2"])

    # Paired row bootstrap: one shared resampling per draw, so the pairwise
    # differences carry the same rows for both stages (the length-control's shape).
    rng = np.random.default_rng(1944)
    n_rows = cells[stages[0]]["n"]
    n_boot = 1000
    res_mat = np.vstack([cells[s]["_res"] for s in stages])
    tot_mat = np.vstack([cells[s]["_tot"] for s in stages])
    draws = np.empty((n_boot, len(stages)))
    for b in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        draws[b] = 1.0 - res_mat[:, idx].sum(axis=1) / tot_mat[:, idx].sum(axis=1)
    differences = {}
    for i, a in enumerate(stages):
        for j, bst in enumerate(stages):
            if i >= j:
                continue
            d = draws[:, i] - draws[:, j]
            differences[f"{a}_minus_{bst}"] = {
                "point": float(cells[a]["r2"] - cells[bst]["r2"]),
                "row_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
            }
    for cell in cells.values():
        cell.pop("_res", None)
        cell.pop("_tot", None)
    summary = {
        "cells": cells,
        "differences": differences,
        "layer": args.layer,
        "r2_ordering": ordering,
        "folds": "six size-matched IID random-row folds (seed 190231), shared with layer 31",
        "estimator": (
            "issue1902_lasttoken_transfer.SharedPrimalRidge (primal spectral GCV ridge, "
            "0.9 dof cap), imported not re-implemented"
        ),
        "target": f"mean answer vector over {len(SEEDS)} draws (seeds {list(SEEDS)})",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (args.out / f"summary_L{args.layer}.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n"
    )
    for stage in stages:
        c = cells[stage]
        _log(f"{stage} L{args.layer}: R2={c['r2']:.4f} median_cos={c['median_cos']:.4f}")
    for key, d in differences.items():
        mark = "excludes 0" if d["excludes_zero"] else "includes 0"
        _log(f"{key}: {d['point']:+.4f} [{d['row_ci95'][0]:+.4f}, {d['row_ci95'][1]:+.4f}] {mark}")
    _log(f"ordering (best first): {' > '.join(ordering)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
