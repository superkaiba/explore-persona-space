"""Identity baseline for #823 (Step 9a-ter free-analysis follow-up).

Question: how much of the plain-style arm's refit R2 retention (B2 refit
0.56-0.59 vs own-arm A' refit 0.60-0.63 at the read-out layers) is explainable
by TARGET-SPACE OVERLAP alone -- i.e. how well does the own-arm target vector
v_A'(x) predict the plain-arm target vector v_B2(x) directly (no context
input), under the SAME ridge / KFold / R2 conventions phase 4 used?

Fits, per layer:  ridge  v_A'(x)[:, L, :]  ->  v_target(x)[:, L, :]
  targets: B2 (Sonnet-plain, primary), C (derangement, floor), B1 (Sonnet-weird,
  read-out layers only).

Conventions copied from run_823.phase4_ridge_refit (bit-identical machinery):
  - solver: explore_persona_space.experiments.issue_779.fit_h.ridge_fit_predict
    (canonical numpy-SVD path, GCV lambda over np.logspace(-2, 4, 13))
  - folds:  KFold(n_splits=5, shuffle=True, random_state=0) on the masked rows
  - mask:   common_valid_idx.json (phase 1 output)
  - R2:     1 - ss_res / (ss_tot + 1e-12), ss_tot centered on the val-fold mean

Inputs are the full-run arm tensors on the HF data repo
(issue823_own_vs_external/analysis_tensors/v_{a_prime,b1,b2,c}.pt, n=5000 raw,
4998 valid after common_valid_idx);
downloaded OUTSIDE the repo tree (default /mnt/eps-data/<user>/tmp_issue823_identity).

Per-unit JSONL checkpointing + resume (keyed on target/layer/n/solver); final
compact JSON at eval_results/issue_823/identity_baseline.json.

Usage:
  uv run python scripts/issue823_identity_baseline.py --smoke     # 200 ctx, 2 units
  uv run python scripts/issue823_identity_baseline.py             # full run
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF creds BEFORE torch import (shared-VM rule)

import argparse
import datetime
import json
import logging
import os
import pathlib
import subprocess
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue823_identity_baseline")

REPO_ID = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue823_own_vs_external"
EXPECTED_N = 5000  # raw tensor rows; common_valid_idx masks to 4998 valid
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
READ_OUT_LAYERS = {"evil": 14, "sycophancy": 26, "hallucination": 17}
# read-out layers first (priority), then every-4th + final layer for the curve
RO_LAYERS = sorted(set(READ_OUT_LAYERS.values()))  # [14, 17, 26]
EXTRA_LAYERS = [0, 4, 8, 12, 16, 20, 24, 27]
SOLVER_TAG = "fit_h.ridge_fit_predict (numpy-SVD, GCV lambda in logspace(-2,4,13))"
SCHEMA = 1


def _sha() -> str:
    """Return the current git commit SHA of this worktree (provenance)."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=pathlib.Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _download(dl_dir: pathlib.Path, names: list[str]) -> dict[str, pathlib.Path]:
    """hf_hub_download each arm tensor + common_valid_idx into dl_dir; return paths."""
    from huggingface_hub import hf_hub_download

    dl_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, pathlib.Path] = {}
    for name in names:
        fn = f"{HF_PREFIX}/analysis_tensors/v_{name}.pt"
        p = hf_hub_download(REPO_ID, fn, repo_type="dataset", local_dir=str(dl_dir))
        paths[name] = pathlib.Path(p)
    fn = f"{HF_PREFIX}/raw_completions/phase1/common_valid_idx.json"
    paths["common_valid_idx"] = pathlib.Path(
        hf_hub_download(REPO_ID, fn, repo_type="dataset", local_dir=str(dl_dir))
    )
    return paths


def _load_arm(path: pathlib.Path, n: int) -> torch.Tensor:
    """mmap-load an arm tensor, assert shape, return the first-n slice (still mmapped)."""
    t = torch.load(str(path), map_location="cpu", mmap=True)
    assert t.shape == (EXPECTED_N, EXPECTED_LAYERS, EXPECTED_HIDDEN), (path.name, tuple(t.shape))
    return t[:n]


def _fold_r2(X: np.ndarray, Y: np.ndarray, folds: list) -> list[float]:
    """5-fold out-of-fold pooled R2, identical definition to phase4_ridge_refit."""
    from explore_persona_space.experiments.issue_779.fit_h import ridge_fit_predict

    r2s: list[float] = []
    for train_idx, val_idx in folds:
        y_pred = ridge_fit_predict(X[train_idx], Y[train_idx], X[val_idx])
        ss_res = float(np.sum((Y[val_idx] - y_pred) ** 2))
        ss_tot = float(np.sum((Y[val_idx] - Y[val_idx].mean(0)) ** 2))
        r2s.append(1.0 - ss_res / (ss_tot + 1e-12))
    return r2s


def _unit_key(target: str, layer: int, n: int) -> dict:
    return {
        "target": target,
        "layer": layer,
        "n_contexts": n,
        "solver": SOLVER_TAG,
        "schema": SCHEMA,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-contexts", type=int, default=EXPECTED_N)
    ap.add_argument("--smoke", action="store_true", help="200 contexts, 2 units, _smoke outputs")
    ap.add_argument(
        "--dl-dir",
        default=f"/mnt/eps-data/{os.environ.get('USER', 'thomasjiralerspong')}/tmp_issue823_identity",
    )
    args = ap.parse_args()

    base = pathlib.Path(__file__).resolve().parent.parent
    n = 200 if args.smoke else args.n_contexts
    suffix = "_smoke" if args.smoke else ""
    out_dir = base / "eval_results" / "issue_823"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"identity_baseline_units{suffix}.jsonl"
    json_path = out_dir / f"identity_baseline{suffix}.json"

    # unit list, priority-ordered: read-out layers for every target first
    if args.smoke:
        units = [("b2", 14), ("c", 14)]
    else:
        units = [(t, ro) for t in ("b2", "c", "b1") for ro in RO_LAYERS]
        units += [(t, layer) for t in ("b2", "c") for layer in EXTRA_LAYERS]
    targets_needed = sorted({t for t, _ in units})

    logger.info("Downloading arm tensors (a_prime + %s) to %s ...", targets_needed, args.dl_dir)
    paths = _download(pathlib.Path(args.dl_dir), ["a_prime", *targets_needed])

    valid_all = np.array(
        sorted(json.loads(paths["common_valid_idx"].read_text())["common_valid_idx"]), dtype=int
    )
    valid_idx = valid_all[valid_all < n]
    logger.info(
        "[common_valid_idx] %d valid of %d requested (%d dropped)",
        len(valid_idx),
        n,
        n - len(valid_idx),
    )

    v_a_prime = _load_arm(paths["a_prime"], n)
    arms = {t: _load_arm(paths[t], n) for t in targets_needed}

    from sklearn.model_selection import KFold

    # Folds depend only on n_valid; identical to phase-4's split on the masked rows.
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(kf.split(np.zeros((len(valid_idx), 1))))

    # resume: load completed units
    done: set[tuple[str, int]] = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            row = json.loads(line)
            if all(row.get(k) == v for k, v in _unit_key(row["target"], row["layer"], n).items()):
                done.add((row["target"], row["layer"]))
    if done:
        logger.info("Resume: %d/%d units already done", len(done), len(units))

    for i, (target, layer) in enumerate(units):
        if (target, layer) in done:
            continue
        t0 = time.time()
        X = v_a_prime[valid_idx, layer, :].numpy().astype(np.float64)
        Y = arms[target][valid_idx, layer, :].numpy().astype(np.float64)
        assert X.shape == Y.shape == (len(valid_idx), EXPECTED_HIDDEN), (X.shape, Y.shape)
        r2s = _fold_r2(X, Y, folds)
        row = {
            **_unit_key(target, layer, n),
            "r2_folds": r2s,
            "r2_mean": float(np.mean(r2s)),
            "r2_sd": float(np.std(r2s, ddof=1)),
            "n_valid": len(valid_idx),
            "fit_seconds": round(time.time() - t0, 1),
        }
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        logger.info(
            "[%d/%d] A'->%s L%d  R2=%.4f (sd %.4f)  %.0fs",
            i + 1,
            len(units),
            target.upper(),
            layer,
            row["r2_mean"],
            row["r2_sd"],
            row["fit_seconds"],
        )

    # ── assemble final JSON ───────────────────────────────────────────────────
    results: dict[str, dict[str, dict]] = {}
    for line in jsonl_path.read_text().splitlines():
        row = json.loads(line)
        if row["n_contexts"] != n:
            continue
        results.setdefault(row["target"], {})[str(row["layer"])] = {
            "r2_mean": row["r2_mean"],
            "r2_sd": row["r2_sd"],
            "r2_folds": row["r2_folds"],
        }

    # reference refit numbers (phase-4 full run) at the same layers, if available
    reference: dict = {}
    ref_path = out_dir / "ridge_r2_by_arm.json"
    if ref_path.exists() and not args.smoke:
        ref = json.loads(ref_path.read_text())
        trait0 = next(iter(ref["refit"]["A_prime"]))  # r2_by_layer is trait-independent
        for arm in ("A_prime", "B2", "B1", "C"):
            if arm not in ref["refit"]:
                continue
            by_layer = ref["refit"][arm][trait0]["r2_by_layer"]
            reference[arm] = {
                str(layer): {
                    "refit_r2_mean": float(np.mean(by_layer[layer])),
                    "refit_r2_sd": float(np.std(by_layer[layer], ddof=1)),
                }
                for _t, layer in [("x", ly) for ly in sorted({ly for _tt, ly in units})]
            }

    out = {
        "description": (
            "Identity baseline for #823: ridge v_A'(x) -> v_target(x) per layer "
            "(input = own-arm target vector, NOT context). Same solver/folds/mask/R2 "
            "as run_823 phase 4. Compare against reference_refit (cx_last -> v_arm)."
        ),
        "n_contexts_requested": n,
        "n_valid": len(valid_idx),
        "n_dropped": int(n - len(valid_idx)),
        "smoke": args.smoke,
        "solver": SOLVER_TAG,
        "kfold": "KFold(n_splits=5, shuffle=True, random_state=0) on masked rows",
        "read_out_layers": READ_OUT_LAYERS,
        "identity_baseline_r2": results,
        "reference_refit": reference,
        "tensor_source": f"hf://datasets/{REPO_ID}/{HF_PREFIX}/analysis_tensors/",
        "git_commit": _sha(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    json_path.write_text(json.dumps(out, indent=1))
    logger.info("Wrote %s", json_path)


if __name__ == "__main__":
    main()
