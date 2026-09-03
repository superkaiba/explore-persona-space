#!/usr/bin/env python3
"""Reduced-rank regression (RRR) rank curves for the issue-2588 panel maps.

The mapping-rank script truncates the fitted ridge map W along W's own top
singular directions.  The best rank-k linear predictor instead keeps the top-k
principal directions of the FITTED training outputs X W (reduced-rank ridge,
Mukherjee & Zhu 2011; classical RRR for lambda -> 0).  With the fitted map
cached, that only needs the training activations of the selected layer: the
fitted-output covariance is W^T (X^T X) W, its eigenvectors give the nested
rank-k projections, and the held-out R^2 at every rank follows from the same
projection identity the truncated-ridge curve uses.

Writes eval_results/issue_2588/rrr_rank_curves.json: per map, validation and
test R^2 at every rank 0..d (exact, no randomized truncation) plus the
fitted-output eigenvalue spectrum.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg

import issue2588_mapping_rank_vs_capability as MR

REPO = MR.REPO
DEFAULT_OUT = REPO / "eval_results" / "issue_2588" / "rrr_rank_curves.json"


def rrr_curves(spec: MR.MapSpec, cache_dir: Path) -> dict:
    started = time.time()
    payload = MR.reconstruct_map(spec, cache_dir)
    layer = int(payload["layer"])
    d = int(payload["dimension"])
    w = np.asarray(payload["W"], dtype=np.float64)
    xmu = np.asarray(payload["xmu"], dtype=np.float64)
    xsd = np.asarray(payload["xsd"], dtype=np.float64)
    ymu = np.asarray(payload["ymu"], dtype=np.float64)
    print(f"[{spec.key}] download train_10k at L{layer} (d={d})", flush=True)
    xtr, _ytr = MR.load_split(spec, "train_10k", layer)
    xn = (xtr.astype(np.float64) - xmu) / xsd
    n_train = xn.shape[0]
    gram = xn.T @ xn
    del xn, xtr
    # Fitted-output covariance (up to 1/n): W^T X^T X W.  Symmetrize for eigh.
    m = w.T @ gram @ w
    m = 0.5 * (m + m.T)
    evals, evecs = scipy.linalg.eigh(m, check_finite=False)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None) / n_train
    right = np.ascontiguousarray(evecs[:, order], dtype=np.float32)
    pred_val, yval = payload["pred_val"], payload["target_val"]
    pred_test, ytest = payload["pred_test"], payload["target_test"]
    full_val = MR.pooled_r2(pred_val, yval)
    full_test = MR.pooled_r2(pred_test, ytest)
    val_curve = MR.r2_curve_from_top_right_vectors(pred_val, yval, ymu, right)
    test_curve = MR.r2_curve_from_top_right_vectors(pred_test, ytest, ymu, right)
    if abs(float(val_curve[-1]) - full_val) > 1e-4:
        raise RuntimeError(
            f"{spec.key}: full-rank RRR curve {val_curve[-1]:.6f} != full {full_val:.6f}"
        )
    total_var = float(evals.sum())
    cum = np.cumsum(evals) / (total_var + 1e-30)
    result = {
        "key": spec.key,
        "cell": spec.cell,
        "model": spec.model_label,
        "family": spec.family,
        "arm": spec.arm,
        "aa_index": spec.aa_index,
        "aa_status": spec.aa_status,
        "dimension": d,
        "layer_star": layer,
        "n_train": int(n_train),
        "method": "reduced-rank ridge: fitted ridge map projected onto top-k principal directions of the fitted training outputs (exact eigh of W^T X^T X W)",
        "full_validation_r2": full_val,
        "full_test_r2": full_test,
        "rank_curve": {
            "validation_r2": [float(v) for v in val_curve],
            "test_r2": [float(v) for v in test_curve],
        },
        "fitted_output_spectrum": {
            "eigenvalues_top64": [float(v) for v in evals[:64]],
            "total_variance": total_var,
            "directions_for_90pct_variance": int(np.searchsorted(cum, 0.90) + 1),
            "directions_for_99pct_variance": int(np.searchsorted(cum, 0.99) + 1),
        },
        "hf_revision": MR.HF_REVISION,
        "elapsed_s": float(time.time() - started),
    }
    k02 = MR.minimum_rank_within(val_curve, full_val, 0.02)
    print(
        f"[{spec.key}] RRR rank within 0.02 = {k02}/{d} ({100 * k02 / d:.1f}%); "
        f"90% fitted variance in {result['fitted_output_spectrum']['directions_for_90pct_variance']} dirs; "
        f"{result['elapsed_s']:.0f}s",
        flush=True,
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=MR.DEFAULT_CACHE)
    ap.add_argument(
        "--results",
        type=Path,
        default=MR.DEFAULT_OUT,
        help="mapping_rank JSON; only its maps are run",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--maps", nargs="*", default=None)
    args = ap.parse_args()
    present = {m["key"] for m in json.loads(args.results.read_text(encoding="utf-8"))["maps"]}
    specs = [s for s in MR.MAPS if s.key in present]
    if args.maps:
        specs = [s for s in specs if s.key in set(args.maps) or s.cell in set(args.maps)]
    existing: dict[str, dict] = {}
    if args.out.exists():
        existing = {r["key"]: r for r in json.loads(args.out.read_text(encoding="utf-8"))["maps"]}
    results = []
    for spec in specs:
        prior = existing.get(spec.key)
        if prior is not None and prior.get("hf_revision") == MR.HF_REVISION and not args.maps:
            results.append(prior)
            continue
        results.append(rrr_curves(spec, args.cache_dir))
        merged = {**existing, **{r["key"]: r for r in results}}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {"schema_version": "issue2588_rrr_rank_curves_v1", "maps": list(merged.values())},
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
    order = {s.key: i for i, s in enumerate(MR.MAPS)}
    merged = {**existing, **{r["key"]: r for r in results}}
    out = sorted(merged.values(), key=lambda r: order.get(r["key"], 10**6))
    args.out.write_text(
        json.dumps({"schema_version": "issue2588_rrr_rank_curves_v1", "maps": out}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({len(out)} maps)", flush=True)


if __name__ == "__main__":
    main()
