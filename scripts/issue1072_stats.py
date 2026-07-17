"""Issue #1072 phase D — pooled cross-fit component statistics (VM CPU, ~min).

Consumes the 5 per-fold ``per_context_stats_1072_fold{k}.npz`` +
``battery_1072_fold{k}.json`` outputs of ``run_1072 --phase battery`` and
emits ``stats_component.json``: the registered decision statistics of plan §6

  - D_l = ΔC_par - ΔC_perp per decision layer on the c-leg remainder cells
    (PRIMARY: layer 26; Holm across the 3 non-primary layers),
  - H2 z-leg residual D_resid + per-component closure fractions ΔG_c/G0_c,
  - H3 depth profile S_par = ΔC_par/ΔR²_full (descriptive),

with the parent bootstrap recipe verbatim (10k multinomial draws, rng 0, FULL
pool resampled, fold assignment fixed — pooled intervals conditional on the
fold assignment) as two stacked-draw GEMMs, a paired own↔ext label-flip
permutation companion (rng 1) for the one-sided closure reads, the pinned
add-one bootstrap p, and a 3-cell serial-oracle parity gate on the ratio bank.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE any heavy import — shared-VM thread caps (#847)

import numpy as np  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from explore_persona_space.experiments.issue_952.run_952 import (  # noqa: E402
    _json_np,
    _repo_git_sha,
    kfold_split_hashes,
    make_kfold_splits,
)
from scripts.issue952_stats import holm_adjust, pinned_bootstrap_p  # noqa: E402

logger = logging.getLogger("issue1072.stats")

BOOTSTRAP_SEED = 0  # plan §10 Seeds (parent-inherited)
SIGNFLIP_SEED = 1
N_DRAWS_DEFAULT = 10_000
PARITY_N_DRAWS = 200
PARITY_TOL = 1e-8
ADDITIVITY_TOL = 1e-9
PRIMARY_LAYER = 26
K_FOLDS = 5
MATCHED_ARMS = ("own", "ext_plain", "ext_style")
CLEG_ARMS = ("own", "ext_plain", "ext_style", "mismatch")
COMPS = ("par", "perp", "cross", "full")
# Channel order written by component_ridge (kept in lockstep).
CH = {
    "ss_res_par": 0,
    "ss_tot_par": 1,
    "ss_res_perp": 2,
    "ss_tot_perp": 3,
    "cross_res": 4,
    "cross_tot": 5,
    "ss_res_full": 6,
    "ss_tot_full": 7,
}
NEAR_DUP_CAVEAT = (
    "Inherited scope caveat (#952): ~7% of test contexts have TF-idf near-duplicates in "
    "train; the registered statistics are PAIRED own-vs-external contrasts on identical "
    "contexts, which cancel context-level interpolation effects to first order."
)


class RatioBank:
    """Per-context (numerator, denominator) columns over the FULL pool id space.

    Cell value = Σnum/Σden; bootstrap draws = (w@num)/(w@den) — two stacked-draw
    GEMMs (the parent CellBank pattern, ratio-of-sums generalized). Contexts a
    cell does not cover carry 0 in BOTH vectors (they contribute nothing under
    any resample weight).
    """

    def __init__(self, pool_ids: list[int]) -> None:
        self.pool_ids = [int(i) for i in pool_ids]
        self.col_of = {c: j for j, c in enumerate(self.pool_ids)}
        self.n = len(self.pool_ids)
        self.names: list[str] = []
        self._num: list[np.ndarray] = []
        self._den: list[np.ndarray] = []

    def add(self, name: str, ids: np.ndarray, num: np.ndarray, den: np.ndarray) -> None:
        assert np.isfinite(num).all() and np.isfinite(den).all(), name
        v_num = np.zeros(self.n, dtype=np.float64)
        v_den = np.zeros(self.n, dtype=np.float64)
        for i, cid in enumerate(ids):
            j = self.col_of[int(cid)]
            v_num[j] = num[i]
            v_den[j] = den[i]
        self.names.append(name)
        self._num.append(v_num)
        self._den.append(v_den)

    def stacks(self) -> tuple[np.ndarray, np.ndarray]:
        return np.stack(self._num, axis=1), np.stack(self._den, axis=1)

    def observed(self) -> np.ndarray:
        num, den = self.stacks()
        return _safe_ratio(num.sum(0), den.sum(0))

    def draws(self, w: np.ndarray) -> np.ndarray:
        num, den = self.stacks()
        return _safe_ratio(w @ num, w @ den)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.where(np.abs(den) < 1e-12, np.nan, r)


def serial_ratio_parity(bank: RatioBank, w: np.ndarray, draws: np.ndarray) -> dict:
    """3-cell serial per-draw oracle vs the GEMM ratio bank (hard gate)."""
    num, den = bank.stacks()
    valid = [c for c in range(len(bank.names)) if abs(den[:, c].sum()) > 1e-12][:3]
    assert valid, "no valid ratio columns for the parity check"
    n_check = min(PARITY_N_DRAWS, w.shape[0])
    max_diff = 0.0
    for c in valid:
        for d in range(n_check):  # the ORACLE is deliberately serial (parity only)
            got = float(draws[d, c])
            dd = float(np.dot(w[d], den[:, c]))
            oracle = float(np.dot(w[d], num[:, c])) / dd if abs(dd) > 1e-12 else float("nan")
            if np.isfinite(oracle) or np.isfinite(got):
                max_diff = max(max_diff, abs(oracle - got))
    rec = {
        "cells": [bank.names[c] for c in valid],
        "n_draws_checked": n_check,
        "max_abs_diff": max_diff,
        "tol": PARITY_TOL,
    }
    if max_diff > PARITY_TOL:
        raise RuntimeError(f"ratio-bank GEMM vs serial oracle parity FAIL: {rec}")
    logger.info("[parity] ratio bank vs serial oracle: max|diff|=%.2e OK", max_diff)
    return rec


def _ci(draws: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]


def _load_folds(eval_dir: pathlib.Path) -> tuple[dict[int, dict], dict[int, dict]]:
    npzs: dict[int, dict] = {}
    recs: dict[int, dict] = {}
    for k in range(K_FOLDS):
        p = eval_dir / f"per_context_stats_1072_fold{k}.npz"
        j = eval_dir / f"battery_1072_fold{k}.json"
        assert p.exists() and j.exists(), f"fold {k} outputs missing under {eval_dir}"
        npzs[k] = dict(np.load(p, allow_pickle=False))
        recs[k] = json.loads(j.read_text())
    return npzs, recs


def _gate_fold_records(recs: dict[int, dict], smoke: bool) -> dict:
    """Never trust record presence alone: every (fold, layer) must carry a g4
    PASS + a sub-tolerance fp64 additivity dev (parent kfold_main convention)."""
    max_add = 0.0
    for k, rec in recs.items():
        for layer, lrec in rec["layers"].items():
            if lrec.get("skipped"):
                assert smoke, f"fold {k} L{layer} skipped in production"
                continue
            assert (lrec.get("g4") or {}).get("verdict") == "PASS", (k, layer)
            add = float(lrec["components"]["additivity_max_dev"])
            assert add < ADDITIVITY_TOL, (k, layer, add)
            max_add = max(max_add, add)
    return {"additivity_max_dev": max_add, "tol": ADDITIVITY_TOL, "verdict": "PASS"}


def _register_cells(bank: RatioBank, npzs: dict[int, dict], layers: list[int]) -> None:
    """One (num, den) column per (leg, layer, arm, component) from the merged
    per-fold TEST channels; every matched context appears exactly once."""
    for layer in layers:
        for leg, arms in (("c", CLEG_ARMS), ("z", MATCHED_ARMS)):
            for arm in arms:
                key = f"M16{leg}_L{layer}|{arm}"
                per_fold = []
                for k in sorted(npzs):
                    assert key in npzs[k], f"registered cell {key} missing from fold {k} npz"
                    per_fold.append((npzs[k]["ids_test"], npzs[k][key].astype(np.float64)))
                ids = np.concatenate([f[0] for f in per_fold])
                ch = np.concatenate([f[1] for f in per_fold], axis=0)
                assert len(set(ids.tolist())) == len(ids), f"{key}: duplicate test contexts"
                den = ch[:, CH["ss_tot_full"]]
                for comp in COMPS:
                    if comp == "full":
                        num = ch[:, CH["ss_tot_full"]] - ch[:, CH["ss_res_full"]]
                    elif comp == "cross":
                        num = ch[:, CH["cross_tot"]] - ch[:, CH["cross_res"]]
                    else:
                        num = ch[:, CH[f"ss_tot_{comp}"]] - ch[:, CH[f"ss_res_{comp}"]]
                    bank.add(f"{leg}|{layer}|{arm}|{comp}", ids, num, den)


def _signflip_stats(
    bank: RatioBank,
    idx: dict[str, int],
    layers: list[int],
    n_draws: int,
) -> dict:
    """Paired own↔ext label-flip permutation (rng SIGNFLIP_SEED) for the
    one-sided closure reads: per draw, each context's (own, ext) cell pair is
    swapped with p=1/2 and the pooled statistic recomputed via GEMMs."""
    num, den = bank.stacks()
    rng = np.random.default_rng(SIGNFLIP_SEED)
    f = rng.integers(0, 2, size=(n_draws, bank.n)).astype(np.float64)
    g = 1.0 - f

    def _mix(name_a: str, name_b: str) -> tuple[np.ndarray, np.ndarray]:
        """Pooled (Σnum, Σden) of the label-mixed 'a-side' per draw."""
        ca, cb = idx[name_a], idx[name_b]
        return (
            f @ num[:, ca] + g @ num[:, cb],
            f @ den[:, ca] + g @ den[:, cb],
        )

    def _mix_b(name_a: str, name_b: str) -> tuple[np.ndarray, np.ndarray]:
        ca, cb = idx[name_a], idx[name_b]
        return (
            g @ num[:, ca] + f @ num[:, cb],
            g @ den[:, ca] + f @ den[:, cb],
        )

    def _delta_c(leg: str, layer: int, comp: str, ext: str) -> np.ndarray:
        na, da = _mix(f"{leg}|{layer}|own|{comp}", f"{leg}|{layer}|{ext}|{comp}")
        nb, db = _mix_b(f"{leg}|{layer}|own|{comp}", f"{leg}|{layer}|{ext}|{comp}")
        return _safe_ratio(na, da) - _safe_ratio(nb, db)

    out: dict[str, dict] = {}
    for ext in ("ext_plain", "ext_style"):
        for layer in layers:
            d_flip = _delta_c("c", layer, "par", ext) - _delta_c("c", layer, "perp", ext)
            dg_par_flip = _delta_c("c", layer, "par", ext) - _delta_c("z", layer, "par", ext)
            out.setdefault(ext, {})[str(layer)] = {
                "D_flip_p975": float(np.nanpercentile(d_flip, 97.5)),
                "D_flip_p025": float(np.nanpercentile(d_flip, 2.5)),
                "delta_G_par_flip_p95": float(np.nanpercentile(dg_par_flip, 95.0)),
            }
    return {"seed": SIGNFLIP_SEED, "n_draws": n_draws, "by_ext": out}


def main() -> None:
    p = argparse.ArgumentParser(description="Issue #1072 component stats battery (phase D)")
    p.add_argument(
        "--eval-dir",
        type=str,
        default=str(_REPO_ROOT / "eval_results" / "issue_1072"),
        help="dir holding per_context_stats_1072_fold*.npz + battery_1072_fold*.json",
    )
    p.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    t0 = time.time()
    eval_dir = pathlib.Path(args.eval_dir)
    out_path = pathlib.Path(args.out) if args.out else eval_dir / "stats_component.json"

    npzs, recs = _load_folds(eval_dir)
    gates = _gate_fold_records(recs, args.smoke)
    layers = sorted(int(x) for x in recs[0]["regime"]["layers"])
    pool_ids = [int(i) for i in npzs[0]["ids_pool_full"].tolist()]
    for k in npzs:
        assert npzs[k]["ids_pool_full"].tolist() == pool_ids, f"fold {k}: pool drift"

    # Fold-identity defense in depth (g5 ran pod-side; recompute + re-assert).
    folds = make_kfold_splits(pool_ids, K_FOLDS)
    for f in folds:
        kfold_split_hashes(f)  # deterministic construction re-runs (raises on drift)
    covered = np.concatenate([npzs[k]["ids_test"] for k in sorted(npzs)])
    assert len(set(covered.tolist())) == len(covered), "fold test partitions overlap"
    coverage = {"n_matched_pool": len(covered), "n_full_pool": len(pool_ids)}

    bank = RatioBank(pool_ids)
    _register_cells(bank, npzs, layers)
    idx = {n: i for i, n in enumerate(bank.names)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    w = rng.multinomial(bank.n, np.full(bank.n, 1.0 / bank.n), size=args.n_draws).astype(np.float64)
    obs = bank.observed()
    draws = bank.draws(w)
    parity = serial_ratio_parity(bank, w, draws)

    def _cell(leg: str, layer: int, arm: str, comp: str) -> int:
        return idx[f"{leg}|{layer}|{arm}|{comp}"]

    def _delta(leg: str, layer: int, comp: str, ext: str, arr: np.ndarray) -> np.ndarray:
        return arr[..., _cell(leg, layer, "own", comp)] - arr[..., _cell(leg, layer, ext, comp)]

    results: dict = {}
    raw_p_by_layer: dict[str, dict[str, float | None]] = {"ext_plain": {}, "ext_style": {}}
    for ext in ("ext_plain", "ext_style"):
        by_layer = {}
        for layer in layers:
            dc = {comp: float(_delta("c", layer, comp, ext, obs)) for comp in COMPS}
            d_obs = dc["par"] - dc["perp"]
            d_draws = _delta("c", layer, "par", ext, draws) - _delta("c", layer, "perp", ext, draws)
            dz = {comp: float(_delta("z", layer, comp, ext, obs)) for comp in COMPS}
            d_resid_obs = dz["par"] - dz["perp"]
            d_resid_draws = _delta("z", layer, "par", ext, draws) - _delta(
                "z", layer, "perp", ext, draws
            )
            closure = {}
            for comp in ("par", "perp", "cross", "full"):
                g0 = dc[comp]
                gt = dz[comp]
                dg_draws = _delta("c", layer, comp, ext, draws) - _delta(
                    "z", layer, comp, ext, draws
                )
                frac_draws = 1.0 - _safe_ratio(
                    _delta("z", layer, comp, ext, draws),
                    _delta("c", layer, comp, ext, draws),
                )
                closure[comp] = {
                    "G0": g0,
                    "Gt": gt,
                    "delta_G": g0 - gt,
                    "delta_G_ci95": _ci(dg_draws),
                    "delta_G_p_one_sided": pinned_bootstrap_p(g0 - gt, dg_draws, tail="greater"),
                    "closure_frac": (1.0 - gt / g0) if abs(g0) > 1e-12 else None,
                    "closure_frac_ci95": _ci(frac_draws),
                }
            frac_diff_draws = _safe_ratio(
                _delta("z", layer, "par", ext, draws), _delta("c", layer, "par", ext, draws)
            ) - _safe_ratio(
                _delta("z", layer, "perp", ext, draws), _delta("c", layer, "perp", ext, draws)
            )
            s_par_draws = _safe_ratio(
                _delta("c", layer, "par", ext, draws), _delta("c", layer, "full", ext, draws)
            )
            by_layer[str(layer)] = {
                "delta_C": dc,
                "delta_R2_full_identity_dev": abs(
                    dc["full"] - (dc["par"] + dc["perp"] + dc["cross"])
                ),
                "D": d_obs,
                "D_ci95": _ci(d_draws),
                "D_p_two_sided_raw": pinned_bootstrap_p(d_obs, d_draws, tail="two"),
                "zleg": {
                    "delta_C": dz,
                    "D_resid": d_resid_obs,
                    "D_resid_ci95": _ci(d_resid_draws),
                    "D_resid_p_one_sided": pinned_bootstrap_p(
                        d_resid_obs, d_resid_draws, tail="greater"
                    ),
                },
                "closure_by_component": closure,
                "unclosed_frac_par_minus_perp": {
                    "observed": (
                        (dz["par"] / dc["par"] if abs(dc["par"]) > 1e-12 else None),
                        (dz["perp"] / dc["perp"] if abs(dc["perp"]) > 1e-12 else None),
                    ),
                    "frac_diff_ci95": _ci(frac_diff_draws),
                },
                "S_par": dc["par"] / dc["full"] if abs(dc["full"]) > 1e-12 else None,
                "S_par_ci95": _ci(s_par_draws),
            }
            raw_p_by_layer[ext][str(layer)] = by_layer[str(layer)]["D_p_two_sided_raw"]
        results[ext] = by_layer

    # Registered lattice on D at the PRIMARY layer (plan §3, ext_plain).
    non_primary = [str(la) for la in layers if la != PRIMARY_LAYER]
    holm = holm_adjust({la: raw_p_by_layer["ext_plain"][la] for la in non_primary})
    primary_key = (
        str(PRIMARY_LAYER) if str(PRIMARY_LAYER) in results["ext_plain"] else str(max(layers))
    )
    prim = results["ext_plain"][primary_key]
    lo, hi = prim["D_ci95"]
    if lo > 0.0:
        verdict = "Confirmed"
    elif hi < 0.0:
        verdict = "Falsified"
    else:
        verdict = "Inconclusive"
    lattice = {
        "primary_layer": int(primary_key),
        "D": prim["D"],
        "D_ci95": prim["D_ci95"],
        "verdict": verdict,
        "holm_non_primary_two_sided": holm,
        "note": (
            "Confirmed iff D > 0 AND the 95% CI excludes 0 positively; Falsified iff the CI "
            "is wholly below 0; Inconclusive otherwise (plan §3 lattice)."
        ),
    }
    signflip = _signflip_stats(bank, idx, layers, args.n_draws)

    out = {
        "issue": 1072,
        "git_sha": _repo_git_sha(),
        "numpy_version": np.__version__,
        "ts": time.time(),
        "inputs": {
            "eval_dir": str(eval_dir),
            "n_draws": args.n_draws,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "signflip_seed": SIGNFLIP_SEED,
            "layers": layers,
            "smoke": bool(args.smoke),
        },
        "coverage": coverage,
        "fold_gates": gates,
        "bootstrap_parity": parity,
        "lattice": lattice,
        "by_ext": results,
        "signflip_companion": signflip,
        "near_duplicate_caveat": NEAR_DUP_CAVEAT,
        "ci_conditionality": (
            "contexts resampled with the fold assignment FIXED — pooled intervals are "
            "conditional on the fold assignment (parent recipe verbatim)"
        ),
        "wall_seconds": time.time() - t0,
    }
    out_path.write_text(json.dumps(out, indent=2, default=_json_np))
    logger.info(
        "[stats] %s written: primary D(L%s)=%.5f CI %s -> %s (%.1fs)",
        out_path,
        primary_key,
        prim["D"],
        prim["D_ci95"],
        verdict,
        out["wall_seconds"],
    )


if __name__ == "__main__":
    main()
