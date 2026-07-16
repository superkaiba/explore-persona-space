"""#823 free-analysis: REVERSE cross-arm ridge transfer (plain -> own/style/mismatched).

The committed #823 phase-4 `transfer` leg fits the map on (cx_last -> v_A_prime)
[own answer arm] per fold and rescores that SAME prediction against every target
arm (own -> plain R2 0.451-0.461, own -> style ~0). This script computes the
MISSING reverse direction: fit the map on the PLAIN-EXTERNAL arm (cx_last -> v_B2)
under the IDENTICAL harness, and rescore against {own (A_prime), style (B1),
mismatched (C), plain (B2 = reproduction gate)}.

Conventions copied VERBATIM from run_823.phase4_ridge_refit:
  - input  X  = bundle["cx_last"][:, L, :]  (context last-token vector)
  - target Y  = v_{arm}[:, L, :]
  - mask   = common_valid_idx.json (phase-1 intersection)
  - folds  = KFold(n_splits=5, shuffle=True, random_state=0) on masked rows
  - R2     = 1 - ss_res / (ss_tot + 1e-12); ss_tot centered on the TARGET arm's
             val-fold mean (per run_823.py:1704-1706)
  - transfer rescoring: ONE fit per (layer, fold) on (X, v_B2); the SAME Y_pred
             scored against every target arm (the committed dedup shape).

Read-out layers (plan-pinned, frozen in #779): evil L14, sycophancy L26,
hallucination L17.

Solver: fit_h.ridge_fit_predict (canonical numpy-SVD, GCV lambda in
logspace(-2,4,13)) — the EXACT solver the committed run used by default
(run_823 `_ridge_solver == "canonical"`), so plain->plain reproduces the
committed B2 refit bit-for-bit. A 1-fold parity slice vs ridge_fit_predict_fast
is logged for the record.

Usage:
  uv run python scripts/issue823_crossarm_transfer.py --pilot   # 1-fold timing + parity
  uv run python scripts/issue823_crossarm_transfer.py           # full 15-fit run
"""

from __future__ import annotations

# ruff: noqa: E402 — load_dotenv() must run before torch import (shared-VM thread caps)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + creds before torch import (shared-VM rule)

import argparse
import datetime
import json
import logging
import pathlib
import subprocess
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue823_crossarm_transfer")

DL = "/mnt/eps-data/thomasjiralerspong/tmp_issue823_crossarm"
PREFIX = "issue823_own_vs_external"
EXPECTED_N = 5000
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
READ_OUT_LAYERS = {"evil": 14, "sycophancy": 26, "hallucination": 17}
# Arm -> human label (per run_823 + identity_baseline docstrings).
ARM_LABEL = {"a_prime": "own", "b1": "style", "b2": "plain", "c": "mismatched"}
# Committed reference (eval_results/issue_823/ridge_r2_by_arm.json), mean over 5 folds.
COMMITTED_B2_REFIT = {14: 0.5846, 26: 0.5560, 17: 0.5911}  # plain->plain reproduction gate
COMMITTED_OWN_TO = {  # committed transfer fit=A_prime score=arm, mean over 5 folds
    "own": {14: 0.5988, 26: 0.6080, 17: 0.6260},  # == A_prime refit
    "plain": {14: 0.4572, 26: 0.4513, 17: 0.4612},
    "style": {14: -0.0697, 26: 0.0497, 17: -0.0592},
    "mismatched": {14: -0.7441, 26: -0.6528, 17: -0.8025},
}
REPRO_TOL = 0.005


def _sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=pathlib.Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _load_arm(name: str, n: int) -> torch.Tensor:
    p = pathlib.Path(DL) / PREFIX / "analysis_tensors" / f"v_{name}.pt"
    t = torch.load(str(p), map_location="cpu", mmap=True)
    assert t.shape == (EXPECTED_N, EXPECTED_LAYERS, EXPECTED_HIDDEN), (name, tuple(t.shape))
    return t[:n]


def _load_bundle_cx_last(n: int) -> torch.Tensor:
    p = pathlib.Path(DL) / "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
    b = torch.load(str(p), map_location="cpu", mmap=True)
    cx = b["cx_last"]
    assert cx.shape == (EXPECTED_N, EXPECTED_LAYERS, EXPECTED_HIDDEN), tuple(cx.shape)
    return cx[:n]


def _valid_idx(n: int) -> np.ndarray:
    p = pathlib.Path(DL) / PREFIX / "raw_completions/phase1/common_valid_idx.json"
    all_valid = np.array(sorted(json.loads(p.read_text())["common_valid_idx"]), dtype=int)
    return all_valid[all_valid < n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pilot", action="store_true", help="1-fold timing + solver parity, no full run"
    )
    ap.add_argument("--n-contexts", type=int, default=EXPECTED_N)
    args = ap.parse_args()

    torch.set_num_threads(8)
    from sklearn.model_selection import KFold

    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict,
        ridge_fit_predict_fast,
    )

    base = pathlib.Path(__file__).resolve().parent.parent
    out_dir = base / "eval_results" / "issue_823" / "crossarm_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_contexts
    cx_last = _load_bundle_cx_last(n)
    arms = {a: _load_arm(a, n) for a in ("a_prime", "b1", "b2", "c")}
    valid = _valid_idx(n)
    logger.info("Loaded cx_last + 4 arms; n=%d valid=%d", n, len(valid))

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(kf.split(np.zeros((len(valid), 1))))

    ro_layers = sorted(set(READ_OUT_LAYERS.values()))  # [14, 17, 26]

    if args.pilot:
        L = 14
        X = cx_last[valid, L, :].numpy().astype(np.float64)
        Yb2 = arms["b2"][valid, L, :].numpy().astype(np.float64)
        tr, va = folds[0]
        t0 = time.time()
        pred_c = ridge_fit_predict(X[tr], Yb2[tr], X[va])
        t_can = time.time() - t0
        t0 = time.time()
        pred_f = ridge_fit_predict_fast(X[tr], Yb2[tr], X[va])
        t_fast = time.time() - t0
        scale = float(np.abs(pred_c).max()) + 1e-12
        max_rel = float(np.abs(pred_f - pred_c).max()) / scale
        logger.info(
            "[pilot] canonical %.1fs  fast %.1fs  max_rel(fast vs canonical)=%.2e  "
            "projected 15 canonical fits = %.1f min",
            t_can,
            t_fast,
            max_rel,
            15 * t_can / 60,
        )
        return

    # Full run: 15 canonical fits (3 layers x 5 folds), each rescored vs 4 arms.
    results: dict[str, dict] = {}
    t_start = time.time()
    for L in ro_layers:
        X = cx_last[valid, L, :].numpy().astype(np.float64)
        Y = {a: arms[a][valid, L, :].numpy().astype(np.float64) for a in arms}
        per_arm_folds: dict[str, list[float]] = {ARM_LABEL[a]: [] for a in arms}
        t_layer = time.time()
        for tr, va in folds:
            y_pred = ridge_fit_predict(X[tr], Y["b2"][tr], X[va])  # fit on PLAIN (b2)
            for a in arms:
                yv = Y[a][va]
                ss_res = float(np.sum((yv - y_pred) ** 2))
                ss_tot = float(np.sum((yv - yv.mean(0)) ** 2))
                per_arm_folds[ARM_LABEL[a]].append(1.0 - ss_res / (ss_tot + 1e-12))
        results[str(L)] = {
            label: {
                "r2_folds": folds_r2,
                "r2_mean": float(np.mean(folds_r2)),
                "r2_sd": float(np.std(folds_r2, ddof=1)),
            }
            for label, folds_r2 in per_arm_folds.items()
        }
        logger.info(
            "L%d done (%.0fs): ->own=%.4f ->plain=%.4f ->style=%.4f ->mismatched=%.4f",
            L,
            time.time() - t_layer,
            results[str(L)]["own"]["r2_mean"],
            results[str(L)]["plain"]["r2_mean"],
            results[str(L)]["style"]["r2_mean"],
            results[str(L)]["mismatched"]["r2_mean"],
        )

    # Reproduction gate: plain->plain must match committed B2 refit within tol.
    repro = {}
    gate_pass = True
    for trait, L in READ_OUT_LAYERS.items():
        got = results[str(L)]["plain"]["r2_mean"]
        ref = COMMITTED_B2_REFIT[L]
        delta = abs(got - ref)
        ok = delta <= REPRO_TOL
        gate_pass = gate_pass and ok
        repro[trait] = {
            "layer": L,
            "got": got,
            "committed_b2_refit": ref,
            "delta": delta,
            "pass": ok,
        }

    out = {
        "description": (
            "REVERSE cross-arm transfer for #823: fit ridge map on (cx_last -> v_B2) "
            "[plain-external arm] per fold, rescore vs {own(A_prime), style(B1), "
            "mismatched(C), plain(B2)}. Same solver/folds/mask/R2 as run_823 phase 4."
        ),
        "read_out_layers": READ_OUT_LAYERS,
        "solver": "fit_h.ridge_fit_predict (canonical numpy-SVD, GCV lambda logspace(-2,4,13))",
        "kfold": "KFold(n_splits=5, shuffle=True, random_state=0) on masked rows",
        "n_contexts_requested": n,
        "n_valid": len(valid),
        "fit_arm": "b2 (plain-external)",
        "reverse_transfer_r2": results,
        "reproduction_gate": {"tol": REPRO_TOL, "pass": gate_pass, "cells": repro},
        "committed_forward_own_to": COMMITTED_OWN_TO,
        "tensor_source_arm": f"data-repo {PREFIX}/analysis_tensors/ @ 8039d15f30",
        "tensor_source_cx_last": (
            "hf://datasets/superkaiba1/explore-persona-space-data/"
            "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt @ c94070508a"
        ),
        "git_commit": _sha(),
        "wall_seconds": round(time.time() - t_start, 1),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    (out_dir / "reverse_transfer.json").write_text(json.dumps(out, indent=1))
    logger.info("Wrote %s  (repro gate pass=%s)", out_dir / "reverse_transfer.json", gate_pass)


if __name__ == "__main__":
    main()
