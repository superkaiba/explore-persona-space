#!/usr/bin/env python3
"""#1775 fu round (`dedup-refit-pcfold-doubly`) cell 3: doubly-scheme stitch-MLP
+ delta_beyond(doubly).

Fits the #779-recipe stitch-MLP (w8192 / lr 3e-4 / wd 1e-4 / AdamW full-batch /
group-respecting inner-val early stop; ``issue1775_ladder.mlp_fit_groups``
verbatim) under the committed doubly-novel 6-fold scheme — the ONE cell run 1
never filled (``issue1775_delta_beyond.py`` reported the doubly delta_beyond as
"unavailable"). Target = full-population pca48 basis (run-1 parity; cell 2 owns
the PC-provenance question). Per-seed held-out preds + masks persist to the
run-1 naming (``{cell}_L{layer}_stitch_perrow_pca48_doubly_mlp_s{s}.npy``) and
upload BEFORE the reduction consumes them (#825 ordering).

delta_beyond(doubly) = R2(stitch-MLP seed ensemble) - R2(bilinear r*=32), both
scored on the same doubly test rows, CI = the committed two-way
(prefix x query) cluster bootstrap (``issue1775_bilinear.
two_way_cluster_bootstrap_delta_r2``). The bilinear arm REUSES run-1's
persisted ``pred_doubly_f*_r{0,32}_s*_wd*.npy`` shards (Hub, via the delta
helper's own ``_fetch``); the reconstruction is VALIDATED by recomputing the
committed doubly delta_named point estimate to 1e-8 before the new statistic
is computed. Fold alignment is gated on exact equality with run-1's persisted
``te_doubly_f*.npy`` index arrays (both smoke and production).

Smoke (fu_run.sh --smoke): 1 fold x 1 seed x max_epochs=8 x 200 draws on the
FULL fit population (no row-limit: the reused run-1 shards are
production-fold-shaped, so a truncated manifest could not exercise the
cross-phase consumer — #518 class); artifact presence + schema only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: caps + .env bind BEFORE the heavy imports (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from issue1775_bilinear import _best_variant, two_way_cluster_bootstrap_delta_r2  # noqa: E402
from issue1775_common import (  # noqa: E402
    CELL_PRIMARY,
    FU_SUB,
    LAYER_PRIMARY,
    _basis_targets_with_info,
    _r2,
    append_unit,
    atomic_write_json,
    build_arm_data,
    eval_dir,
    fold_pairs,
    knn_retrieval,
    load_units,
    resolve_store_dir,
    restrict_pairs,
    result_meta,
    unit_key,
    upload_phase_eval_json,
    upload_phase_tensors,
)
from issue1775_delta_beyond import _fetch  # noqa: E402
from issue1775_ladder import mlp_fit_groups, persist_pred, pred_path  # noqa: E402

# Committed run-1 inputs are PARENT inputs — read from the REPO tree, never the
# (smoke-rebindable) out-root (#542 smoke-root-rebinding class; mirrors
# issue1775_delta_beyond.COMMITTED).
COMMITTED = Path(__file__).resolve().parents[1] / "eval_results" / "issue_1775" / "bilinear"
R_STAR_DOUBLY = 32  # carried from run-1's prefix inner-val selection (body convention)
MLP_REGIME_KEYS = ("phase", "scheme", "folds", "seeds", "smoke", "max_epochs")


def _mlp_unit(folds: list[int], seeds: list[int], smoke: bool, max_epochs: int) -> dict:
    return {
        "phase": "doubly_mlp",
        "scheme": "doubly",
        "folds": ",".join(str(f) for f in folds),
        "seeds": ",".join(str(s) for s in seeds),
        "smoke": bool(smoke),
        "max_epochs": int(max_epochs),
    }


def _pred_u() -> dict:
    return {
        "cell": CELL_PRIMARY,
        "layer": LAYER_PRIMARY,
        "arm": "stitch",
        "grain": "perrow",
        "scheme": "doubly",
        "rung": "mlp",
    }


def _load_committed_doubly_units() -> list[dict]:
    units: list[dict] = []
    for shard in sorted(COMMITTED.glob("units_shard*.jsonl")):
        with open(shard, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    if d.get("scheme") == "doubly":
                        units.append(d)
    if not units:
        raise RuntimeError(
            f"no committed doubly bilinear unit rows under {COMMITTED} — run-1's P4 "
            "doubly pass is a required input (plan section 10 B1)"
        )
    return units


def _pooled_bilinear_pred(units, r: int, seeds, n_rows: int, d_out: int, pairs):
    """Mean-over-seeds pooled run-1 doubly bilinear pred (per-seed best wd on
    inner val), shards fetched via the delta helper's own ``_fetch`` (B1)."""
    pred = np.zeros((n_rows, d_out))
    covered = np.zeros(n_rows, dtype=bool)
    for f, (_tr, te) in enumerate(pairs):
        recs = [u for u in units if u["fold"] == f and u["r"] == r]
        if not recs:
            raise RuntimeError(f"committed doubly unit row missing for fold={f} r={r}")
        acc = np.zeros((len(te), d_out))
        for s in seeds:
            v = _best_variant(recs[0]["variants"], s)
            p = _fetch(f"bilinear_params/pred_doubly_f{f}_r{r}_s{s}_wd{v['wd']:g}.npy")
            arr = np.load(p).astype(np.float64)
            assert arr.shape == (len(te), d_out), (arr.shape, len(te), d_out, f, r, s)
            acc += arr
        pred[te] = acc / len(seeds)
        covered[te] = True
    return pred, covered


def main() -> int:
    ap = argparse.ArgumentParser(description="#1775 fu cell 3: doubly stitch-MLP + delta_beyond")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--folds", default=None, help="csv fold subset (smoke default: 0)")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--n-draws", type=int, default=None)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n_draws = args.n_draws
    max_epochs = args.max_epochs
    if args.smoke:
        seeds = seeds[:1]
        n_draws = n_draws or 200
        max_epochs = max_epochs or 8
    else:
        n_draws = n_draws or 2000
        max_epochs = max_epochs or 300
    t0 = time.monotonic()
    out_dir = eval_dir(FU_SUB)
    store = resolve_store_dir()
    ad = build_arm_data(store, CELL_PRIMARY, LAYER_PRIMARY, arms=("stitch",))
    X = ad.X["stitch"]
    n = len(ad.rows)
    # full-population pca48 basis (run-1 parity; cell 2 owns the PC question)
    Yp, _info = _basis_targets_with_info(
        ad.Y_stacked, "pca48", hidden_dim=3584, targets=["t1", "t2", "t3"], projection_target="t1"
    )
    Yp = np.ascontiguousarray(Yp, dtype=np.float64)
    d_out = Yp.shape[1]
    pairs = restrict_pairs(fold_pairs(ad.rows, n, "doubly"), ad.arm_row_mask["stitch"])
    n_te_total = int(sum(len(te) for _tr, te in pairs))
    print(
        f"[doubly-mlp] fit population n={n}; doubly folds={len(pairs)} "
        f"total test rows={n_te_total} (B6 expected ~2,900)",
        flush=True,
    )
    # ── B6 fold-alignment gate: recomputed te sets == run-1's persisted te arrays ──
    for f, (_tr, te) in enumerate(pairs):
        te_ref = np.load(_fetch(f"bilinear_params/te_doubly_f{f}.npy"))
        assert np.array_equal(np.asarray(te), np.asarray(te_ref)), (
            f"doubly fold {f} test-index mismatch vs run-1's persisted te_doubly_f{f}.npy "
            f"({len(te)} vs {len(te_ref)} rows) — fold derivation drifted; refusing to mix"
        )
    print(f"[doubly-mlp] fold-alignment gate OK ({len(pairs)} folds == run-1 te sets)", flush=True)

    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    if folds is None:
        folds = [0] if args.smoke else list(range(len(pairs)))
    fit_pairs = [pairs[f] for f in folds]
    groups = ad.prefix_ids  # scheme != query -> prefix groups (run-1 convention)

    # ── run-1 bilinear arm + committed-Δ_named validation gate, BEFORE the fit ───
    # (round-1 review m2: this consumes run-1 artifacts only — running it ahead
    # of the ~0.8 h mlp_fit_groups fails fast on shard drift)
    units = _load_committed_doubly_units()
    run1_seeds = [0, 1, 2]
    pred_star, cov_star = _pooled_bilinear_pred(units, R_STAR_DOUBLY, run1_seeds, n, d_out, pairs)
    pred_0, cov_0 = _pooled_bilinear_pred(units, 0, run1_seeds, n, d_out, pairs)
    both = cov_star & cov_0
    # validation gate: reproduce the committed doubly delta_named point exactly
    committed = json.loads((COMMITTED / "bilinear_fits.json").read_text())
    ref = committed["schemes"]["doubly"]["delta_named"]
    val = two_way_cluster_bootstrap_delta_r2(
        Yp, pred_star, pred_0, both, ad.prefix_ids, ad.query_ids, n_draws=n_draws, seed=0
    )
    dd = abs(val["delta_r2"] - ref["delta_r2"])
    print(
        f"[delta-doubly] delta_named recomputed {val['delta_r2']:.10f} vs committed "
        f"{ref['delta_r2']:.10f} (|diff|={dd:.2e})",
        flush=True,
    )
    assert dd < 1e-8, f"doubly delta_named validation failed: |diff|={dd}"

    # ── stitch-MLP (#779 recipe verbatim via the committed batched trainer) ──────
    units_path = out_dir / "units_doubly_mlp.jsonl"
    unit = _mlp_unit(folds, seeds, args.smoke, max_epochs)
    done_rows = [
        d
        for d in load_units(units_path)
        if unit_key(d, MLP_REGIME_KEYS) == unit_key(unit, MLP_REGIME_KEYS)
    ]
    pred_files = {
        s: (
            pred_path(_pred_u(), basis="pca48", seed=s),
            pred_path(_pred_u(), basis="pca48", seed=s).with_name(
                pred_path(_pred_u(), basis="pca48", seed=s).stem + "_mask.npy"
            ),
        )
        for s in seeds
    }
    if done_rows and all(p.exists() and m.exists() for p, m in pred_files.values()):
        print("[doubly-mlp] RESUME: unit row + pred files present — skipping fit", flush=True)
        preds_by_seed = {s: np.load(p).astype(np.float64) for s, (p, _m) in pred_files.items()}
        covered = np.load(next(iter(pred_files.values()))[1])
        fit_row = done_rows[-1]
    else:
        tf = time.monotonic()
        res = mlp_fit_groups(
            X, Yp, fit_pairs, groups, seeds, device=args.device, max_epochs=max_epochs
        )
        preds_by_seed = res["preds_by_seed"]
        covered = res["covered"]
        r2s = {str(s): _r2(Yp[covered], preds_by_seed[s][covered]) for s in seeds}
        for s in seeds:
            persist_pred(_pred_u(), "pca48", preds_by_seed[s], covered, seed=s)
        fit_row = {
            **unit,
            "r2_by_seed": r2s,
            "r2_seed_mean": float(np.mean(list(r2s.values()))),
            "r2_seed_sd": float(np.std(list(r2s.values()))),
            "epochs_ran": res["epochs_ran"],
            "wall_s": time.monotonic() - tf,
        }
        append_unit(units_path, fit_row)
        print(
            f"[doubly-mlp] unit 1/1 folds={unit['folds']} seeds={unit['seeds']} "
            f"elapsed={fit_row['wall_s']:.0f}s (PILOT: one MLP battery at doubly shape)",
            flush=True,
        )
    mpred = np.mean([preds_by_seed[s] for s in seeds], axis=0)
    knn_folds = [
        {
            m: knn_retrieval(mpred[pairs[f][1]], Yp[pairs[f][1]], ks=(1, 5, 10), metric=m)
            for m in ("euclidean", "cosine")
        }
        for f in folds
    ]
    mlp_out = {
        "meta": result_meta(
            smoke=bool(args.smoke),
            seeds=seeds,
            folds=folds,
            max_epochs=max_epochs,
            recipe={
                "width": 8192,
                "lr": 3e-4,
                "wd": 1e-4,
                "patience": 20,
                "batch": "full-batch (#779 batched_mlp_fit recipe verbatim)",
            },
        ),
        "scheme": "doubly",
        "basis": "pca48 (full fit-population — run-1 parity; cell 2 owns PC provenance)",
        "n_rows_covered": int(covered.sum()),
        "per_fold_r2_seed_ensemble": {
            str(f): _r2(Yp[pairs[f][1]], mpred[pairs[f][1]]) for f in folds
        },
        "fit": {k: v for k, v in fit_row.items() if k not in ("smoke",)},
        "baselines": {
            "identity_bias": (
                "inapplicable — d_in 7168 != d_out 48 (stated, per the standing rule)"
            ),
            "knn_retrieval_per_fold": knn_folds,
        },
    }
    atomic_write_json(out_dir / "stitch_mlp_doubly.json", mlp_out)
    # persist the pred shards BEFORE the reduction consumes them (#825 ordering)
    upload_phase_tensors("heldout_preds", smoke=bool(args.smoke))

    # ── delta_beyond(doubly): MLP (this fit) vs run-1 bilinear (validated above) ─
    mmask = np.load(pred_files[seeds[0]][1]) if pred_files[seeds[0]][1].exists() else covered
    b3 = both & mmask
    boot = two_way_cluster_bootstrap_delta_r2(
        Yp, mpred, pred_star, b3, ad.prefix_ids, ad.query_ids, n_draws=n_draws, seed=0
    )
    delta_out = {
        "meta": result_meta(smoke=bool(args.smoke), n_draws=n_draws, seeds=seeds, folds=folds),
        "scheme": "doubly",
        "grouping_unit": "prefix_id x query_id (two-way cluster bootstrap)",
        "r_star_carried": R_STAR_DOUBLY,
        "delta_beyond_mlp_minus_bilinear": boot,
        "r2_stitch_mlp_seed_mean": _r2(Yp[b3], mpred[b3]),
        "r2_bilinear_r_star": _r2(Yp[b3], pred_star[b3]),
        "delta_named_validation_abs_diff": dd,
        "delta_named_doubly_recomputed": val,
        "n_rows_b3": int(b3.sum()),
        "note": (
            "fills the doubly cell issue1775_delta_beyond.py reported as 'unavailable' "
            "(run 1 fit no stitch-MLP under the doubly scheme); bilinear arm = run-1's "
            "persisted pred_doubly_* shards (reused, Hub-fetched via the delta helper's "
            "_fetch), MLP arm = this round's fit; same rows, paired"
        ),
    }
    atomic_write_json(out_dir / "delta_beyond_doubly.json", delta_out)
    upload_phase_eval_json(FU_SUB, smoke=bool(args.smoke))
    print(
        f"[doubly-mlp] done in {(time.monotonic() - t0) / 60:.1f} min "
        f"(delta_beyond_doubly={boot['delta_r2']:.4f} "
        f"ci={boot['ci95_two_way_cluster']} n_b3={int(b3.sum())})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
