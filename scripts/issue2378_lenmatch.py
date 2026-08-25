"""Issue #2378 — length-matched own-map refits (Step 9a-ter free-analysis follow-up).

Resolves the answer-length caveat on the cross-framing own-ceiling ordering
(body Result 1: chat 0.61 > plain 0.30 > stories 0.23-0.28 > user 0.21; chat
answers run 17-38x longer by cell medians, and the answer vector v_A is a
span MEAN over the answer tokens, so span length is the named confound
mechanism).

Pre-stated matching scheme (ONE scheme, fixed before any fit):
  - Length variable: answer span length in TOKENS (``ans_hi - ans_lo`` from
    the activation-store ledger — the exact window v_A averages over; the
    body's char medians are the monotone char-side view of the same
    quantity).
  - Common band [8, 256] tokens, 10 log-spaced bins; per bin, every cell is
    subsampled (seeded) to the cross-cell MINIMUM count, so all 8 cells get
    IDENTICAL length histograms and identical n (the realized N per cell is
    recorded in every output).
  - Control leg: a seeded uniform size-matched subsample (same n, the cell's
    NATURAL length distribution), same folds + estimator — isolates the
    length effect from the small-n / estimator effect.
  - Folds: inherited row-for-row from the global-family fold map
    (``pool_gf/fold_map_gf.json``), so the refits share the fold structure of
    the gf ceilings they are compared to.
  - Estimator: matching drives every per-fold n_train (~800) far below the
    ambient d=5120, so the ambient GCV-ridge fit is REFUSED for every cell
    (#1887 regime guard) and both legs use the reviewed reduced-basis core
    (train-fold PCA-k + dof-capped GCV ridge, ``issue2054_fits``) with ONE
    common k = min(1024, floor(0.5 * min n_train over all cells/legs/folds)),
    computed BEFORE any fit. The shuffled-answer null battery is skipped by
    design: mappability tiers were already established at the unmatched
    ceilings; this read is an ORDERING comparison, with 200-draw row
    bootstrap CIs on the pooled R².

Outputs (checkpoint cadence — written per cell-leg as each completes):
  eval_results/issue_2378/lenmatch/<cell>__context__{matched,control}.json
  eval_results/issue_2378/lenmatch/lenmatch_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue2054_fits as pf  # noqa: E402  (reviewed #825-family fit cores — reuse, no new estimator)
import issue2378_common as cm  # noqa: E402
import issue2378_p6_common as p6  # noqa: E402

SCRIPT_VERSION = "issue2378_lenmatch_v1"
D_AMBIENT = 5120
ARM = "context"
LAYER = 51
BAND_LO, BAND_HI, N_BINS = 8, 256, 10
REDUCED_K_MAX = 1024
BOOTSTRAP_DRAWS = 200
LEGS = ("matched", "control")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _bin_edges() -> np.ndarray:
    """10 log-spaced integer bin edges over the pre-stated [8, 256] band."""
    return np.unique(
        np.round(np.logspace(np.log10(BAND_LO), np.log10(BAND_HI), N_BINS + 1)).astype(int)
    )


def answer_token_lengths(store_root: Path, cell: str) -> dict[str, int]:
    """Per-row answer span length in tokens from the store ledger (ans_hi - ans_lo)."""
    return {
        r["row_id"]: int(r["ans_hi"]) - int(r["ans_lo"]) for r in p6.load_ledger(store_root, cell)
    }


def matched_selection(
    fm: dict, lengths: dict[str, np.ndarray], edges: np.ndarray
) -> tuple[dict[str, np.ndarray], dict]:
    """Per-cell row indices (into the fold-map row order) under per-bin min matching.

    Every cell keeps exactly the cross-cell minimum count in each length bin,
    drawn with a seeded RNG per (cell, bin) — identical realized histograms.
    """
    cells = list(fm["cells"].keys())
    binned: dict[str, list[np.ndarray]] = {}
    for c in cells:
        v = lengths[c]
        idx_by_bin = []
        for b in range(len(edges) - 1):
            lo, hi = edges[b], edges[b + 1]
            # np.histogram convention: last bin closed on the right
            mask = (v >= lo) & ((v < hi) if b < len(edges) - 2 else (v <= hi))
            idx_by_bin.append(np.flatnonzero(mask))
        binned[c] = idx_by_bin
    per_bin_min = [min(binned[c][b].size for c in cells) for b in range(len(edges) - 1)]
    sel: dict[str, np.ndarray] = {}
    for c in cells:
        picks = []
        for b, m in enumerate(per_bin_min):
            rng = np.random.default_rng(p6.unit_seed(c, "lenmatch", "bin", b))
            pool = binned[c][b]
            picks.append(rng.choice(pool, size=m, replace=False) if m < pool.size else pool)
        sel[c] = np.sort(np.concatenate(picks))
    table = {
        "band_tokens": [int(BAND_LO), int(BAND_HI)],
        "bin_edges": [int(e) for e in edges],
        "per_bin_min": [int(m) for m in per_bin_min],
        "n_matched_per_cell": int(sum(per_bin_min)),
    }
    return sel, table


def control_selection(fm: dict, cell: str, n: int) -> np.ndarray:
    """Seeded uniform size-matched subsample (natural length distribution)."""
    n_rows = len(fm["cells"][cell]["row_ids"])
    rng = np.random.default_rng(p6.unit_seed(cell, "lenmatch", "control"))
    return np.sort(rng.choice(n_rows, size=n, replace=False))


def _reduced_fit_predict(
    Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, k: int
) -> tuple[np.ndarray, dict]:
    """Reduced-basis GCV-ridge fit returning PREDICTIONS.

    Diff vs the named reference ``issue2054_fits._reduced_basis_r2``: identical
    projection (train-fold centered PCA-k on X) + identical dof-capped
    ``_ridge_gcv_fit_predict`` call; the only change is returning the held-out
    predictions (needed for pooled R² + row bootstrap) instead of scoring
    internally. No permissiveness broadened.
    """
    k_use = min(k, Xtr.shape[0], Xtr.shape[1])
    Xtr64 = Xtr.astype(np.float64)
    xmu = Xtr64.mean(axis=0)
    Xtr_c = Xtr64 - xmu
    _, _, Vt = np.linalg.svd(Xtr_c, full_matrices=False)
    Vk = Vt[:k_use, :]
    preds, info = pf._ridge_gcv_fit_predict(
        Xtr_c @ Vk.T,
        Ytr,
        (Xte.astype(np.float64) - xmu) @ Vk.T,
        lambdas=pf.DEFAULT_LAMBDAS,
        dof_cap=0.9,
    )
    info["reduced_k"] = int(k_use)
    return preds, info


def _pooled_bootstrap(ss_res: np.ndarray, ss_tot: np.ndarray, seed: int) -> dict:
    """200-draw row bootstrap CI on the pooled R²."""
    rng = np.random.default_rng(seed)
    n = ss_res.size
    draws = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, n, size=n)
        draws[i] = 1.0 - ss_res[idx].sum() / ss_tot[idx].sum()
    return {
        "n_draws": int(BOOTSTRAP_DRAWS),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
    }


def run_cell_leg(
    args, fm: dict, cell: str, leg: str, sel: np.ndarray, k: int, regime: dict, lengths: np.ndarray
) -> dict:
    """Fit one cell-leg on the selected rows under the inherited gf folds."""
    out_dir = Path(args.ledger_root) / "lenmatch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cell}__{ARM}__{leg}.json"
    regime = dict(regime, leg=leg)
    if out_path.exists():
        prior = json.loads(out_path.read_text(encoding="utf-8"))
        if prior.get("regime") == regime:
            _log(f"[lenmatch] SKIP {cell}/{leg}: exists with matching regime")
            return prior
        raise RuntimeError(f"regime mismatch at {out_path} — use a fresh out dir")
    t0 = time.time()
    entry = fm["cells"][cell]
    row_ids = [entry["row_ids"][i] for i in sel]
    folds = np.asarray(entry["folds"], dtype=np.int64)[sel]
    n_folds = int(np.asarray(entry["folds"]).max()) + 1
    if sorted(set(folds.tolist())) != list(range(n_folds)):
        raise RuntimeError(f"{cell}/{leg}: subset lost a fold — {sorted(set(folds.tolist()))}")
    pack = p6.load_cell_arrays(
        Path(args.store_root), cell, LAYER, (p6.SLOT_BY_ARM[ARM], p6.ANSWER_SLOT), row_order=row_ids
    )
    X, Y = pack["arrays"][p6.SLOT_BY_ARM[ARM]], pack["arrays"][p6.ANSWER_SLOT]
    del pack
    n = X.shape[0]
    ybar = Y.astype(np.float64).mean(axis=0)
    ss_tot = ((Y.astype(np.float64) - ybar) ** 2).sum(axis=1)
    ss_res = np.full(n, np.nan)
    per_fold = []
    for f in range(n_folds):
        te = np.flatnonzero(folds == f)
        tr = np.flatnonzero(folds != f)
        n_train = int(tr.size)
        # Estimator-validity gate (dispatch-note duty): the ambient d=5120 fit
        # is refused outright in this under-determined regime; reduced-k only.
        if n_train <= D_AMBIENT:
            ambient_call = "refused (n_train <= d_ambient; reduced-basis leg only)"
        else:  # pragma: no cover — unreachable at N=996; kept as the honest branch
            ambient_call = "would be admissible; still reduced-k for cross-cell uniformity"
        preds, info = _reduced_fit_predict(X[tr], Y[tr], X[te], k)
        r2 = pf._r2_matrix(Y[te], preds)
        ss_res[te] = ((Y[te].astype(np.float64) - preds) ** 2).sum(axis=1)
        per_fold.append(
            {
                "fold": f,
                "n_train": n_train,
                "n_eval": int(te.size),
                "r2": float(r2),
                "ambient_call": ambient_call,
                "fit_info": info,
            }
        )
        _log(f"[lenmatch] {cell}/{leg} fold {f + 1}/{n_folds} r2={r2:+.4f} n_tr={n_train}")
    if np.isnan(ss_res).any():
        raise RuntimeError(f"{cell}/{leg}: fold splits did not cover every row")
    lens_sel = lengths[sel]
    payload = {
        "regime": regime,
        "cell": cell,
        "arm": ARM,
        "leg": leg,
        "fold_regime": "global-family",
        "fold_structure": entry["fold_structure"],
        "n_rows": int(n),
        "d_ambient": D_AMBIENT,
        "reduced_k": int(k),
        "estimator_validity": {
            "min_n_train": int(min(r["n_train"] for r in per_fold)),
            "ambient_fit": "refused — n_train << d=5120 for every fold (see per_fold.ambient_call)",
            "reduced_ratio_min_n_train_over_k": round(min(r["n_train"] for r in per_fold) / k, 3),
        },
        "answer_len_tokens": {
            "median": float(np.median(lens_sel)),
            "mean": float(np.mean(lens_sel)),
            "p5": float(np.percentile(lens_sel, 5)),
            "p95": float(np.percentile(lens_sel, 95)),
        },
        "per_fold": per_fold,
        "fold_mean_r2": float(np.mean([r["r2"] for r in per_fold])),
        "pooled_r2": p6.pooled_r2(ss_res, ss_tot),
        "pooled_bootstrap": _pooled_bootstrap(
            ss_res, ss_tot, p6.unit_seed(cell, "lenmatch", leg, "boot")
        ),
        "unit_wall_s": round(time.time() - t0, 2),
        "metadata": cm.run_metadata(
            {"regen": "committed inputs (fold map + HF store) + this script reproduce preds"}
        ),
    }
    cm.atomic_write_json(out_path, payload)
    _log(
        f"[lenmatch] {cell}/{leg}: fold_mean={payload['fold_mean_r2']:+.4f} "
        f"pooled={payload['pooled_r2']:+.4f} wall={payload['unit_wall_s']}s -> {out_path}"
    )
    return payload


def unmatched_ceiling(ledger_root: Path, fm: dict, cell: str) -> dict:
    """Quote the unmatched gf ambient ceiling (recovery-denominator convention).

    Mirrors ``issue2378_pool._fits_inputs``: refolded (story) cells read the
    gf own-ceiling refit; the content-disjoint cells read the registered
    unit-3 fits (their fold assignment is carried over unchanged into the gf
    map).
    """
    refolded = set(fm["global_folds"].get("refolded_cells", []))
    base = ledger_root / ("pool_gf/own_ceilings" if cell in refolded else "fits")
    d = json.loads((base / f"{cell}__{ARM}.json").read_text(encoding="utf-8"))
    out = {
        "source": str(base.relative_to(ledger_root)) + f"/{cell}__{ARM}.json",
        "fold_mean_r2": d["fold_mean_r2"],
        "pooled_r2": d["pooled_r2"],
        "n_rows": d["n_rows"],
    }
    reduced = [
        rec["reduced_basis"]["r2"] for rec in d.get("per_fold", []) if "reduced_basis" in rec
    ]
    if reduced:
        out["reduced_basis_fold_mean_r2"] = float(np.mean(reduced))
        out["reduced_basis_k"] = d["regime"].get("reduced_k")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-root", required=False, default=None)
    ap.add_argument("--ledger-root", default=str(cm.LEDGER_ROOT))
    ap.add_argument("--cells", default=None, help="comma subset (default: all fold-map cells)")
    ap.add_argument("--legs", default=",".join(LEGS))
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.store_root is None and not args.summary_only:
        raise SystemExit("--store-root is required unless --summary-only")

    ledger_root = Path(args.ledger_root)
    fm = json.loads((ledger_root / "pool_gf" / "fold_map_gf.json").read_text(encoding="utf-8"))
    fm_sha = p6.fold_map_sha(fm)
    cells = list(fm["cells"].keys())
    if args.cells:
        want = [c.strip() for c in args.cells.split(",") if c.strip()]
        unknown = [c for c in want if c not in cells]
        if unknown:
            raise SystemExit(f"unknown cells {unknown}; fold map has {cells}")
        cells = want
    legs = [x.strip() for x in args.legs.split(",") if x.strip()]
    edges = _bin_edges()

    out_dir = ledger_root / "lenmatch"
    if args.summary_only:
        return write_summary(ledger_root, fm, list(fm["cells"].keys()), edges)

    store_root = Path(args.store_root)
    lengths = {}
    for c in fm["cells"]:
        by_id = answer_token_lengths(store_root, c)  # one ledger load per cell
        lengths[c] = np.array([by_id[rid] for rid in fm["cells"][c]["row_ids"]])
    sel_matched, table = matched_selection(fm, lengths, edges)
    n_m = table["n_matched_per_cell"]
    _log(f"[lenmatch] matching table: {json.dumps(table)}")

    # ONE common reduced k, fixed BEFORE any fit, from the min n_train over
    # every (cell, leg, fold) — both legs share n so the min is leg-invariant.
    min_n_train = min(
        int(np.sum(np.asarray(fm["cells"][c]["folds"])[sel_matched[c]] != f))
        for c in fm["cells"]
        for f in range(int(np.asarray(fm["cells"][c]["folds"]).max()) + 1)
    )
    k = min(REDUCED_K_MAX, min_n_train // 2)
    _log(f"[lenmatch] n_matched={n_m}/cell; min n_train={min_n_train}; reduced k={k}")

    regime = {
        "script_version": SCRIPT_VERSION,
        "arm": ARM,
        "layer": LAYER,
        "seed": cm.SEED,
        "fold_map_sha": fm_sha,
        "fold_regime": "global-family",
        "matching": table,
        "length_variable": "answer span tokens (ans_hi - ans_lo, the v_A span-mean window)",
        "reduced_k": int(k),
        "dof_cap": 0.9,
        "lambda_grid": ["logspace", -2, 4, 13],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "null_battery": "skipped by design (ordering read; tiers established at unmatched ceilings)",
        "seed_derivation": "137-rooted per-(cell,bin|leg) via p6.unit_seed('…lenmatch…')",
    }
    for cell in cells:
        for leg in legs:
            sel = sel_matched[cell] if leg == "matched" else control_selection(fm, cell, n_m)
            run_cell_leg(args, fm, cell, leg, sel, k, regime, lengths[cell])
    return write_summary(ledger_root, fm, cells, edges)


def write_summary(ledger_root: Path, fm: dict, cells: list[str], edges: np.ndarray) -> int:
    out_dir = ledger_root / "lenmatch"
    rows = {}
    for cell in cells:
        row: dict = {"unmatched_gf_ceiling": unmatched_ceiling(ledger_root, fm, cell)}
        for leg in LEGS:
            p = out_dir / f"{cell}__{ARM}__{leg}.json"
            if not p.exists():
                _log(f"[lenmatch] summary: missing {p} — partial summary")
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            row[leg] = {
                "fold_mean_r2": d["fold_mean_r2"],
                "pooled_r2": d["pooled_r2"],
                "ci95": [d["pooled_bootstrap"]["ci_lo"], d["pooled_bootstrap"]["ci_hi"]],
                "n_rows": d["n_rows"],
                "median_len_tokens": d["answer_len_tokens"]["median"],
            }
        rows[cell] = row
    ordering = sorted(
        (c for c in rows if "matched" in rows[c]),
        key=lambda c: rows[c]["matched"]["pooled_r2"],
        reverse=True,
    )
    payload = {
        "script_version": SCRIPT_VERSION,
        "bin_edges": [int(e) for e in edges],
        "cells": rows,
        "matched_ordering_by_pooled_r2": ordering,
        "metadata": cm.run_metadata(),
    }
    cm.atomic_write_json(out_dir / "lenmatch_summary.json", payload)
    _log(f"[lenmatch] summary -> {out_dir / 'lenmatch_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
