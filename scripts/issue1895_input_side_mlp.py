"""Nonlinear (MLP) twin cells for the #1895 input-side reconstruction fair grid.

Runs POD-SIDE (1x GPU) on the design arrays exported by
scripts/issue1895_input_side_overlap.py phase fit_rbar (scp'd from the VM).
Fits the parent-recipe MLP (w8192, lr 3e-4, seed 0, internal-val early stop —
the #779/#1895 nonlinear protocol via N1M._fit_mlp_minibatch) for the five
fair-grid cells, per-answer-direction R2 in the same Q_a basis:

    vC -> vA | vC -> rbarA | rbarC -> vA | rbarC -> rbarA | eC -> vA

Per-cell checkpointing (skip-if-done), one JSON summary + one npz of profiles.

Usage (pod, from repo root):
    uv run python scripts/issue1895_input_side_mlp.py --data-dir /workspace/is1895 \
        --out-dir /workspace/is1895/mlp_out --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue1482_error_analysis as EA  # noqa: E402

CELLS = (
    ("vC__to__vA", "X_sub.npy", "Y_sub.npy"),
    ("vC__to__rbarA", "X_sub.npy", "rbar_a.npy"),
    ("rbarC__to__vA", "rbar_c.npy", "Y_sub.npy"),
    ("rbarC__to__rbarA", "rbar_c.npy", "rbar_a.npy"),
    ("eC__to__vA", None, "Y_sub.npy"),  # eC = X_sub - rbar_c, built in-process
)


def log(msg: str) -> None:
    print(f"[is1895-mlp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dd = args.data_dir
    with np.load(dd / "kept_splits.npz") as z:
        tr, va, te = z["tr"], z["va"], z["te"]
    with np.load(dd / "qa_basis.npz") as z:
        Qa = z["Q"].astype(np.float32)
    log(f"kept tr/va/te = {len(tr)}/{len(va)}/{len(te)}")
    arrays: dict[str, np.ndarray] = {}

    def arr(name: str) -> np.ndarray:
        if name not in arrays:
            arrays[name] = np.load(dd / name)
        return arrays[name]

    summary: dict = {
        "cells": {},
        "recipe": {
            "width": int(N1M.MLP_W_PROTOCOL),
            "lr": 3e-4,
            "max_epochs": int(EA.F.MLP_MAX_EPOCHS),
            "batch": int(N1M.MLP_BATCH),
            "seed": 0,
        },
    }
    profiles: dict[str, np.ndarray] = {}
    for key, x_name, y_name in CELLS:
        part = args.out_dir / f"cell_{key}.npz"
        if part.exists():
            with np.load(part) as z:
                profiles[key] = z["r2_perdir"]
                summary["cells"][key] = json.loads(str(z["meta"]))
            log(f"{key}: resume-skip (pooled {summary['cells'][key]['pooled_r2_te']:.4f})")
            continue
        Z = arr("X_sub.npy") - arr("rbar_c.npy") if x_name is None else arr(x_name)
        Y = arr(y_name)
        t0 = time.time()
        pt, meta = N1M._fit_mlp_minibatch(
            Z,
            Y,
            tr,
            te,
            N1M.MLP_W_PROTOCOL,
            3e-4,
            EA.F.MLP_MAX_EPOCHS,
            min(N1M.MLP_BATCH, max(8, len(tr))),
            0,
            torch.device(args.device),
        )
        pooled = float(PR._pooled_r2(pt, Y[te]))
        r2 = EA._per_feature_metrics(pt @ Qa, Y[te] @ Qa)["r2"].astype(np.float32)
        cell_meta = {
            "pooled_r2_te": pooled,
            "epochs_ran": meta["epochs_ran"],
            "wall_s": round(time.time() - t0, 1),
        }
        tmp = part.with_suffix(".tmp.npz")
        np.savez(tmp, r2_perdir=r2, meta=json.dumps(cell_meta))
        tmp.replace(part)
        profiles[key] = r2
        summary["cells"][key] = cell_meta
        log(
            f"{key}: pooled_te={pooled:.6f} epochs={meta['epochs_ran']} wall={cell_meta['wall_s']}s"
        )
    np.savez(args.out_dir / "mlp_profiles.npz", **profiles)
    (args.out_dir / "mlp_cells_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log("done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
