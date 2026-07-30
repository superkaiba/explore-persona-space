#!/usr/bin/env python3
"""#1775 P2: residual HSIC/dCor detection per arm (runs BEFORE nonlinear spend).

Population: the battery-excluded DENSE CORE (complete prefix x query crossed
block — realized block reported), plus a matched-n equal-block full-corpus
companion. OBSERVED statistics come from the #763 reference implementations
(``analysis/issue_763_nonlinear.distance_correlation`` / ``hsic_statistic``)
and are ASSERTED equal to the cached-matrix reads the null machinery uses
(the instrument tie). Nulls are BATCHED (plan section 9): centered kernel /
distance matrices are computed ONCE per (arm, basis); each of the B=1000
draws is one advanced-index gather + product-sum on GPU — H P L P^T H =
P (H L H) P^T for a permutation P, so no per-draw kernel recomputation and
no serial statistic loop. Three group-respecting schemes (prefix-block,
query-block, within-prefix derangement); Holm over the registered 30-p-value
family (5 arms x 3 schemes x 2 statistics, primary cell/basis/layer).

Includes the planted-effect power check (context arm MUST fire — kill
criterion 2; rc=22 on failure) + a graded MDE ladder, per-fold HSIC
recomputes (fold-boundary adjudication), and per-draw null-matrix
persistence to ``analysis_tensors/null_matrices``.
"""

from __future__ import annotations

import argparse
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

from issue1775_common import (  # noqa: E402
    ARMS,
    CELL_PRIMARY,
    LAYER_PRIMARY,
    _basis_targets_with_info,
    _folds_from_manifest,
    atomic_write_json,
    block_permutations_equal_blocks,
    build_arm_data,
    build_dependence_matrices,
    crossed_permutations,
    distance_correlation,
    eval_dir,
    holm_correction,
    hsic_statistic,
    null_stats_batched,
    observed_stats,
    p_value,
    resolve_store_dir,
    result_meta,
    tensors_dir,
    upload_phase_eval_json,
    upload_phase_tensors,
)

SCHEMES = ("prefix_block", "query_block", "within_prefix_derangement")
POWER_EFFECTS = (0.05, 0.10, 0.15, 0.20, 0.30)
POWER_MUST_FIRE_EFFECT = 0.30  # ~= sqrt(banked context-arm L14 gain 0.09)
POWER_FAIL_RC = 22


def _pred_path(arm: str, basis: str, cell: str = CELL_PRIMARY) -> Path:
    return (
        tensors_dir("heldout_preds")
        / f"{cell}_L{LAYER_PRIMARY:02d}_{arm}_perrow_{basis}_prefix_ridge.npy"
    )


def load_residual(
    arm: str, basis: str, Yb: np.ndarray, cell: str = CELL_PRIMARY
) -> tuple[np.ndarray, np.ndarray]:
    p = _pred_path(arm, basis, cell)
    m = p.with_name(p.stem + "_mask.npy")
    if not p.exists() or not m.exists():
        raise FileNotFoundError(f"P1 ridge predictions absent for {arm}/{basis}: {p}")
    pred = np.load(p).astype(np.float64)
    mask = np.load(m)
    return Yb - pred, mask


def complete_dense_block(rows: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """(P, Q) index grid into ``rows`` for the maximal complete dense-core crossing.

    Keeps the modal full-coverage query set: queries present in >= 90% of dense
    prefixes, then prefixes carrying ALL of them (realized block reported;
    plan A13 'restricting to complete rows')."""
    dense = [(i, r) for i, r in enumerate(rows) if r.get("stratum") == "dense_core"]
    assert dense, "no dense_core rows in the fit population"
    by_prefix: dict[str, dict[str, int]] = {}
    for i, r in dense:
        by_prefix.setdefault(str(r["prefix_id"]), {})[str(r["query_id"])] = i
    q_counts: dict[str, int] = {}
    for qmap in by_prefix.values():
        for q in qmap:
            q_counts[q] = q_counts.get(q, 0) + 1
    n_p = len(by_prefix)
    q_full = sorted(q for q, c in q_counts.items() if c >= 0.9 * n_p)
    prefixes = sorted(p for p, qmap in by_prefix.items() if all(q in qmap for q in q_full))
    assert len(prefixes) >= 2 and len(q_full) >= 2, (
        f"dense block degenerate: {len(prefixes)} prefixes x {len(q_full)} queries"
    )
    grid = np.asarray([[by_prefix[p][q] for q in q_full] for p in prefixes], dtype=np.int64)
    return grid, prefixes, q_full


def run_battery(
    X: np.ndarray,
    R: np.ndarray,
    P: int,
    Q: int,
    *,
    n_draws: int,
    device: str,
    seed: int,
    schemes=SCHEMES,
    assert_reference: bool = True,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Observed + per-scheme nulls for one (input, residual) pair on a P x Q block."""
    mats = build_dependence_matrices(X, R, device=device)
    obs = observed_stats(mats)
    if assert_reference:
        ref_h = hsic_statistic(X, R)
        ref_d = distance_correlation(X, R)
        assert abs(obs["hsic"] - ref_h) <= 1e-8 * max(1.0, abs(ref_h)), (
            f"cached-matrix HSIC {obs['hsic']} != #763 reference {ref_h}"
        )
        assert abs(obs["dcor"] - ref_d) <= 1e-6, (
            f"cached-matrix dCor {obs['dcor']} != #763 reference {ref_d}"
        )
    out = {"observed": obs, "schemes": {}}
    nulls: dict[str, dict[str, np.ndarray]] = {}
    for si, scheme in enumerate(schemes):
        perms = crossed_permutations(P, Q, scheme, n_draws, seed=seed + 17 * si)
        ns = null_stats_batched(mats, perms)
        nulls[scheme] = ns
        out["schemes"][scheme] = {
            "p_hsic": p_value(ns["hsic"], obs["hsic"]),
            "p_dcor": p_value(ns["dcor"], obs["dcor"]),
            "null_hsic_q95": float(np.quantile(ns["hsic"], 0.95)),
            "null_dcor_q95": float(np.quantile(ns["dcor"], 0.95)),
        }
    return out, nulls


def planted_battery(
    X: np.ndarray, P: int, Q: int, *, effect: float, n_draws: int, device: str, seed: int
) -> dict:
    """Planted nonlinear residual at ~``effect`` partial-corr through the SAME
    batched pipeline (the #763 ``_planted_nonlinear_dataset`` construction on
    the REAL context-arm input)."""
    rng = np.random.default_rng(seed)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    r2 = (Xs[:, : max(1, min(64, X.shape[1] // 2))] ** 2).sum(axis=1)
    r2_std = (r2 - r2.mean()) / (r2.std() + 1e-12)
    signal = 1.0 / (1.0 + np.exp(-r2_std))
    signal = (signal - signal.mean()) / (signal.std() + 1e-12)
    noise = rng.normal(0.0, 1.0, size=X.shape[0])
    E = effect * signal + np.sqrt(max(0.0, 1.0 - effect**2)) * noise
    res, _ = run_battery(
        X, E[:, None], P, Q, n_draws=n_draws, device=device, seed=seed, assert_reference=False
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="#1775 P2 residual HSIC/dCor detection")
    ap.add_argument("--cells", default=f"{CELL_PRIMARY},cell_pre_own")
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--row-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.smoke:
        args.n_draws = min(args.n_draws, 20)
        if args.row_limit is None:
            args.row_limit = 600
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    out_dir = eval_dir("detection")
    store = resolve_store_dir()
    t0 = time.monotonic()
    results: dict = {
        "meta": result_meta(n_draws=args.n_draws, smoke=args.smoke, device=args.device),
        "cells": {},
    }
    registered_p: dict[str, float] = {}
    null_rows: list[np.ndarray] = []
    null_names: list[str] = []
    for cell in cells:
        arms = ARMS if cell == CELL_PRIMARY else ("prefix_end", "query_averaged", "context_end")
        ad = build_arm_data(store, cell, LAYER_PRIMARY, arms=tuple(arms), row_limit=args.row_limit)
        grid, prefixes, queries = complete_dense_block(ad.rows)
        P, Q = grid.shape
        flat = grid.reshape(-1)
        print(f"[detect/{cell}] dense block {P} x {Q} = {P * Q} rows", flush=True)
        cell_out: dict = {"dense_block": {"n_prefixes": P, "n_queries": Q, "n_rows": P * Q}}
        for basis in ("pca48",) if args.smoke else ("pca48", "ambient"):
            Yb, _ = _basis_targets_with_info(
                ad.Y_stacked,
                basis,
                hidden_dim=3584,
                targets=["t1", "t2", "t3"],
                projection_target="t1",
            )
            Yb = np.ascontiguousarray(Yb, dtype=np.float64)
            for arm in arms:
                try:
                    Rfull, mask = load_residual(arm, basis, Yb, cell)
                except FileNotFoundError as e:
                    print(f"[detect] {e} — skipping {arm}/{basis}", flush=True)
                    continue
                usable = mask[flat].all() and ad.arm_row_mask[arm][flat].all()
                if not usable:
                    keep = mask[flat] & ad.arm_row_mask[arm][flat]
                    print(
                        f"[detect] {arm}/{basis}: {int((~keep).sum())} block rows unusable "
                        "(mask) — arm skipped on the dense block",
                        flush=True,
                    )
                    continue
                X = ad.X[arm][flat]
                R = Rfull[flat]
                res, nulls = run_battery(
                    X, R, P, Q, n_draws=args.n_draws, device=args.device, seed=args.seed
                )
                # per-fold HSIC observed (fold-boundary adjudication data)
                res["per_fold_hsic"] = _per_fold_hsic(ad, arm, Rfull, flat)
                cell_out.setdefault(basis, {})[arm] = res
                if basis == "pca48" and cell == CELL_PRIMARY:
                    for scheme in SCHEMES:
                        registered_p[f"{arm}|{scheme}|hsic"] = res["schemes"][scheme]["p_hsic"]
                        registered_p[f"{arm}|{scheme}|dcor"] = res["schemes"][scheme]["p_dcor"]
                        for stat in ("hsic", "dcor"):
                            null_rows.append(nulls[scheme][stat])
                            null_names.append(f"{arm}|{scheme}|{stat}")
                print(
                    f"[detect] unit {arm}/{basis} done "
                    f"(p_hsic={[res['schemes'][s]['p_hsic'] for s in SCHEMES]}) "
                    f"elapsed={time.monotonic() - t0:.0f}s",
                    flush=True,
                )
        results["cells"][cell] = cell_out
        if cell == CELL_PRIMARY:
            # full-corpus matched-n equal-block companion (prefix-block scheme)
            results["full_corpus_companion"] = _full_corpus_companion(ad, args)
            # planted power check + MDE ladder on the context arm input
            Xc = ad.X["context_end"][flat]
            power: dict = {"effects": {}}
            trials = 1 if args.smoke else 5
            for eff in POWER_EFFECTS:
                det = 0
                for t in range(trials):
                    pr = planted_battery(
                        Xc,
                        P,
                        Q,
                        effect=eff,
                        n_draws=args.n_draws,
                        device=args.device,
                        seed=1000 + 31 * t,
                    )
                    ps = [pr["schemes"][s]["p_hsic"] for s in SCHEMES] + [
                        pr["schemes"][s]["p_dcor"] for s in SCHEMES
                    ]
                    det += int(min(ps) < 0.05)
                power["effects"][str(eff)] = {"trials": trials, "detected": det}
                print(f"[power] effect={eff} detected {det}/{trials}", flush=True)
            fired = power["effects"][str(POWER_MUST_FIRE_EFFECT)]["detected"] >= max(
                1, int(0.8 * trials)
            )
            detected_effects = [
                float(e)
                for e, v in power["effects"].items()
                if v["detected"] >= max(1, int(0.8 * v["trials"]))
            ]
            power["mde"] = min(detected_effects) if detected_effects else None
            power["context_must_fire_passed"] = bool(fired)
            power["note"] = (
                "any per-arm detection null is narrated as 'no structure above the MDE', "
                "never 'linear'"
            )
            results["power_check"] = power
            if not fired and not args.smoke:
                atomic_write_json(out_dir / "hsic_dcor.json", results)
                print("[power] FAILED — context-arm planted effect undetected (rc=22)", flush=True)
                return POWER_FAIL_RC
    results["holm_adjusted_p"] = holm_correction(registered_p)
    results["registered_family_size"] = len(registered_p)
    atomic_write_json(out_dir / "hsic_dcor.json", results)
    if null_rows:
        nd = tensors_dir("null_matrices")
        np.save(nd / "detection_nulls.npy", np.stack(null_rows, axis=0).astype(np.float32))
        atomic_write_json(nd / "detection_nulls_index.json", {"rows": null_names})
    upload_phase_tensors("null_matrices", smoke=args.smoke)
    upload_phase_eval_json("detection", smoke=args.smoke)
    print(f"[detect] done in {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    return 0


def _per_fold_hsic(ad, arm: str, Rfull: np.ndarray, flat: np.ndarray) -> list[dict]:
    folds = _folds_from_manifest(ad.rows, len(ad.rows), group_key="prefix_id", n_folds=6)
    block = set(flat.tolist())
    out = []
    for fi, f in enumerate(folds):
        idx = np.asarray([i for i in f if int(i) in block], dtype=np.int64)
        if idx.size < 20:
            out.append({"fold": fi, "n": int(idx.size), "hsic": None})
            continue
        out.append(
            {
                "fold": fi,
                "n": int(idx.size),
                "hsic": float(hsic_statistic(ad.X[arm][idx], Rfull[idx])),
            }
        )
    return out


def _full_corpus_companion(ad, args) -> dict:
    """Matched-n equal-block subsample of the full fit rows, prefix-block nulls."""
    rng = np.random.default_rng(args.seed + 7)
    k = 8 if args.smoke else 12
    by_prefix: dict[str, list[int]] = {}
    for i, pid in enumerate(ad.prefix_ids):
        by_prefix.setdefault(str(pid), []).append(i)
    eligible = [p for p, idx in by_prefix.items() if len(idx) >= k]
    grid, _, _ = complete_dense_block(ad.rows)
    n_target = grid.size
    n_blocks = max(2, min(len(eligible), n_target // k))
    chosen = rng.choice(np.asarray(eligible), size=n_blocks, replace=False)
    rows = np.concatenate(
        [rng.choice(np.asarray(by_prefix[p]), size=k, replace=False) for p in chosen]
    )
    Yb, _ = _basis_targets_with_info(
        ad.Y_stacked,
        "pca48",
        hidden_dim=3584,
        targets=["t1", "t2", "t3"],
        projection_target="t1",
    )
    Yb = np.ascontiguousarray(Yb, dtype=np.float64)
    out: dict = {
        "n_blocks": int(n_blocks),
        "rows_per_block": int(k),
        "note": (
            "equal-size per-prefix blocks, arbitrary-but-fixed within-block order; "
            "prefix-block exchange only (labeled companion — the crossed alignment "
            "of the dense core does not exist here)"
        ),
        "arms": {},
    }
    skipped: dict[str, str] = {}
    for arm in ad.X:
        try:
            Rfull, mask = load_residual(arm, "pca48", Yb)
        except FileNotFoundError:
            # round-2 Minor-e: loud logged skip + JSON field, never silent
            skipped[arm] = "residual preds absent (P1 pending or arm not persisted)"
            print(f"[detect/full-corpus] SKIP arm={arm}: residuals absent", flush=True)
            continue
        keep = mask[rows] & ad.arm_row_mask[arm][rows]
        if not keep.all():
            skipped[arm] = f"{int((~keep).sum())}/{keep.size} sampled rows masked out"
            print(f"[detect/full-corpus] SKIP arm={arm}: {skipped[arm]}", flush=True)
            continue
        mats = build_dependence_matrices(ad.X[arm][rows], Rfull[rows], device=args.device)
        obs = observed_stats(mats)
        perms = block_permutations_equal_blocks(n_blocks, k, args.n_draws, seed=args.seed + 3)
        ns = null_stats_batched(mats, perms)
        out["arms"][arm] = {
            "observed": obs,
            "p_hsic": p_value(ns["hsic"], obs["hsic"]),
            "p_dcor": p_value(ns["dcor"], obs["dcor"]),
        }
    out["arms_skipped"] = skipped
    return out


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
