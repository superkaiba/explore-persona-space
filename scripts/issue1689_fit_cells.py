"""Issue #1689 Phase D — within-cell fits (both mapping arms per cell).

Per plan §4/§6: fits M(x) -> y_hat with x = prefix arm activation AND
context arm activation separately per cell (42 cells * 2 arms * 4 layers).
Uses #825's `heldout_r2_sweep` with `lambda_selection="inner-group-cv"`
+ `collect_lambdas=True` (plan §4 Estimator - Phase-0 settled). Reports
kNN retrieval + identity+learned-bias baseline alongside R² (CLAUDE.md
standing rule).

Output: `eval_results/issue_1689/percell/heldout_r2.json` +
`lambda_selection.json` per plan §10 reproducibility card.

Smoke: --smoke -> load a single tiny cell's mock activations, run
heldout_r2_sweep with n_folds=2, verify JSON structure only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_common import (  # noqa: E402
    CAPTURE_LAYERS,
    HEADLINE_LAYER,
    LAMBDA_GRIDS,
    N_FOLDS,
    resolve_lambda_grid,
)


def fit_cell(
    store_path: Path,
    *,
    n_folds: int = N_FOLDS,
    null_draws: int = 0,  # nulls done in fit_ladder for pair reads
    lambda_selection: str = "inner-group-cv",
    lambdas=None,  # None -> parent module LAMBDAS (byte-identical published path)
    layers: list[int] | None = None,  # None -> all CAPTURE_LAYERS
) -> dict:
    """Fit prefix + context arms per layer for one cell using
    `heldout_r2_sweep` (#825), plus identity+learned-bias baseline + kNN
    retrieval per CLAUDE.md standing rule."""
    import numpy as np
    import torch

    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )
    from scripts.issue825_fit_cells import heldout_r2_sweep

    # Load the per-layer bundles for this cell.
    layers_present = sorted(CAPTURE_LAYERS) if layers is None else sorted(int(x) for x in layers)
    layer_paths = {
        L: store_path / f"L{L}.pt" for L in layers_present if (store_path / f"L{L}.pt").exists()
    }
    if not layer_paths:
        raise ValueError(f"no layer files found in {store_path}")

    conv_ids = None
    per_layer_X_prefix = []
    per_layer_X_context = []
    per_layer_Y = []
    for L in layers_present:
        if L not in layer_paths:
            continue
        bundle = torch.load(layer_paths[L], map_location="cpu", weights_only=False)
        per_layer_X_prefix.append(np.asarray(bundle["X_prefix"], dtype=np.float32))
        per_layer_X_context.append(np.asarray(bundle["X_context"], dtype=np.float32))
        per_layer_Y.append(np.asarray(bundle["Y"], dtype=np.float32))
        if conv_ids is None:
            conv_ids = np.asarray(bundle["conv_ids"])

    # heldout_r2_sweep expects (N, L, D) arrays.
    X_prefix = np.stack(per_layer_X_prefix, axis=1)
    X_context = np.stack(per_layer_X_context, axis=1)
    Y = np.stack(per_layer_Y, axis=1)

    # NOTE (updated, wider-lambda-ceilings follow-up): the parent
    # heldout_r2_sweep (scripts/issue825_fit_cells.py) now threads a
    # caller-supplied lambdas= grid through the inner-group-cv selection too
    # (it previously hard-asserted lambdas is None on that path). The default
    # lambdas=None scans the parent's module-global LAMBDAS grid,
    # np.logspace(-2, 4, 13) — byte-identical to the published percell run.

    results = {
        "n_rows": int(X_prefix.shape[0]),
        "layers": [int(L) for L in layer_paths.keys()],
        "headline_layer": HEADLINE_LAYER,
        "n_folds": n_folds,
        "lambda_selection": lambda_selection,
        "lambda_grid": "ladder13 (module default)"
        if lambdas is None
        else [float(x) for x in np.asarray(lambdas)],
    }

    for arm_name, X_arm in [("prefix", X_prefix), ("context", X_context)]:
        sweep = heldout_r2_sweep(
            X_arm,
            Y,
            conv_ids,
            n_folds=n_folds,
            seed=42,
            null_draws=null_draws,
            collect_lambdas=True,
            lambda_selection=lambda_selection,
            lambdas=lambdas,
        )
        arm_summary = {
            "held_out_r2_per_layer": [float(x) for x in sweep["r2_obs"]],
            "lambdas_selected": sweep.get("gcv_lambda", None),
        }
        if arm_summary["lambdas_selected"] is not None:
            arm_summary["lambdas_selected"] = [
                [None if not np.isfinite(v) else float(v) for v in row]
                for row in arm_summary["lambdas_selected"]
            ]

        # Identity+learned-bias baseline + kNN retrieval on the headline layer.
        # (Fast: closed-form + a single kNN pass per arm.)
        headline_idx = list(layer_paths.keys()).index(HEADLINE_LAYER)
        X_head = X_arm[:, headline_idx, :]
        Y_head = Y[:, headline_idx, :]
        n = X_head.shape[0]
        # simple 5-fold group split
        pool = int(min(n, 200))
        # baseline: fit + eval on entire cell (train-fold = all for aggregate summary)
        train_idx = np.arange(n)
        eval_idx = np.arange(n)
        pred = identity_bias_predict(X_head[train_idx], Y_head[train_idx], X_head[eval_idx])
        # residual r2 (aggregate)
        ss_res = float(np.sum((Y_head[eval_idx] - pred) ** 2))
        ss_tot = float(np.sum((Y_head[eval_idx] - Y_head[eval_idx].mean(axis=0)) ** 2))
        identity_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        knn_ret = knn_retrieval(pred, Y_head[eval_idx], ks=(1, 5, 10), metric="euclidean")
        knn_cos = knn_retrieval(pred, Y_head[eval_idx], ks=(1, 5, 10), metric="cosine")

        arm_summary["identity_bias_r2_headline"] = identity_r2
        arm_summary["knn_retrieval_headline_euclidean"] = knn_ret
        arm_summary["knn_retrieval_headline_cosine"] = knn_cos
        arm_summary["knn_pool_size"] = pool
        results[arm_name] = arm_summary

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-root", type=Path, required=True)
    ap.add_argument(
        "--cell",
        type=str,
        required=True,
        help="cell slug e.g. Qwen_Qwen2.5-7B-Instruct/assistant_chat",
    )
    ap.add_argument("--out", type=Path, required=True, help="output JSON path (heldout_r2.json)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--lambda-grid",
        choices=sorted(LAMBDA_GRIDS),
        default="ladder13",
        help="ridge lambda grid; ladder13 = parent default (byte-identical published path), "
        "wide19 = logspace(-2,7,19) superset (wider-lambda-ceilings follow-up)",
    )
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="capture layers to fit (default: all CAPTURE_LAYERS; the recheck round passes 19)",
    )
    args = ap.parse_args()

    store_path = args.store_root / args.cell
    n_folds = 2 if args.smoke else N_FOLDS
    # ladder13 -> lambdas=None: the parent module default, byte-identical to the
    # published run (never re-materialized caller-side).
    lambdas = None if args.lambda_grid == "ladder13" else resolve_lambda_grid(args.lambda_grid)

    results = fit_cell(store_path, n_folds=n_folds, lambdas=lambdas, layers=args.layers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[fit_cells] wrote {args.out} (n_rows={results['n_rows']})")
    return 0


if __name__ == "__main__":
    import os

    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGBART pointer. main()'s writes are
    # already flushed via explicit fh.close(); atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
