"""Issue #1072 ``lowdim-token-subspace`` phase D — pooled cross-fit subspace
component statistics (VM CPU, ~min; thread-capped launch per plan §4.3).

Consumes the 5 per-fold ``per_context_stats_lowdim_fold{k}.npz`` +
``battery_lowdim_fold{k}.json`` outputs of ``run_1072_lowdim --phase battery``
and emits ``stats_lowdim.json``: the registered decision statistics of plan §6

  - D_k = ΔC_par(k) - ΔC_perp(k) at layer 26 per basis cell (top-8 / top-32 /
    lookahead-8), bootstrap CIs (10k, rng 0, paired draws shared across bases),
  - two-sided sign-flip p per basis (rng 1) with Holm step-down over the
    3-basis family + the §3 verdict lattice per cell + the family mapping,
  - S_par(k) / w_par(k) / enrichment(k) profiles, per-basis closure (H2),
    pairwise cross-basis D differences (paired draws), the parent 1-D
    reference row, p_last + λ-sensitivity + coverage companions.
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
from scripts.issue952_stats import holm_adjust  # noqa: E402
from scripts.issue1072_stats import (  # noqa: E402
    CH,
    RatioBank,
    _ci,
    _safe_ratio,
    serial_ratio_parity,
)

logger = logging.getLogger("issue1072.lowdim.stats")

BOOTSTRAP_SEED = 0  # plan §10 Seeds (parent-inherited)
SIGNFLIP_SEED = 1
N_DRAWS_DEFAULT = 10_000
ADDITIVITY_TOL = 1e-9
PRIMARY_LAYER = 26
HOLM_ALPHA = 0.05
K_FOLDS = 5
T2 = 16
BASES = ("top8", "top32", "look8")
MATCHED_ARMS = ("own", "ext_plain", "ext_style")
CLEG_ARMS = ("own", "ext_plain", "ext_style", "mismatch")
COMPS = ("par", "perp", "cross", "full")
NEAR_DUP_CAVEAT = (
    "Inherited scope caveat (#952): ~7% of test contexts have TF-idf near-duplicates in "
    "train; the registered statistics are PAIRED own-vs-external contrasts on identical "
    "contexts, which cancel context-level interpolation effects to first order."
)


# ── pure decision logic (unit-tested in tests/test_issue1072_lowdim.py) ─────────


def signflip_p_two(d_obs: float, flip_draws: np.ndarray) -> float | None:
    """Two-sided sign-flip p with add-one counting (plan §3: the Holm inputs).

    ``flip_draws`` is the label-flip null distribution of D (centered at 0
    under own↔ext exchangeability); p = (1 + #{|flip| >= |obs|}) / (1 + B).
    """
    if d_obs is None or not np.isfinite(d_obs):
        return None
    null = np.asarray(flip_draws, dtype=np.float64)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return None
    n_ext = int((np.abs(null) >= abs(float(d_obs))).sum())
    return (1.0 + n_ext) / (1.0 + null.size)


def lattice_verdict(
    d_obs: float, ci95: list[float], holm_p: float | None, alpha: float = HOLM_ALPHA
) -> str:
    """The §3 per-basis verdict lattice (DISJOINT and exhaustive).

    Rescue ⇔ D > 0 AND the 95% CI excludes 0 positively AND Holm-confirmed;
    Extended falsification ⇔ the CI is wholly below 0 AND Holm-confirmed;
    Inconclusive ⇔ otherwise.
    """
    lo, hi = float(ci95[0]), float(ci95[1])
    holm_ok = holm_p is not None and holm_p <= alpha
    if d_obs > 0.0 and lo > 0.0 and holm_ok:
        return "Rescue"
    if hi < 0.0 and holm_ok:
        return "Extended falsification"
    return "Inconclusive"


def family_mapping(verdicts: dict[str, str]) -> dict:
    """Family-level headline mapping (plan §3 prose, not the lattice)."""
    vals = list(verdicts.values())
    if any(v == "Rescue" for v in vals):
        headline = "overturn/qualify"
        prose = "1-D falsified; a low-D token subspace carries the gap"
    elif all(v == "Extended falsification" for v in vals):
        headline = "sharpen"
        prose = (
            "the falsification generalizes to any low-dimensional token-identity subspace, k <= 32"
        )
    else:
        headline = "partial"
        resolved = [b for b, v in verdicts.items() if v != "Inconclusive"]
        prose = f"partial scope: resolved cells = {resolved or 'none'}"
    return {"headline": headline, "prose": prose, "verdicts": verdicts}


# ── data loading + cell registration ────────────────────────────────────────────


def _load_folds(eval_dir: pathlib.Path) -> tuple[dict[int, dict], dict[int, dict]]:
    npzs: dict[int, dict] = {}
    recs: dict[int, dict] = {}
    for k in range(K_FOLDS):
        p = eval_dir / f"per_context_stats_lowdim_fold{k}.npz"
        j = eval_dir / f"battery_lowdim_fold{k}.json"
        assert p.exists() and j.exists(), f"fold {k} outputs missing under {eval_dir}"
        npzs[k] = dict(np.load(p, allow_pickle=False))
        recs[k] = json.loads(j.read_text())
    return npzs, recs


def _gate_fold_records(recs: dict[int, dict], smoke: bool) -> dict:
    """Every (fold, layer) must carry a g4' PASS + sub-tolerance additivity."""
    max_add = 0.0
    for k, rec in recs.items():
        for layer, lrec in rec["layers"].items():
            if lrec.get("skipped"):
                assert smoke, f"fold {k} L{layer} skipped in production"
                continue
            assert (lrec.get("g4p") or {}).get("verdict") == "PASS", (k, layer)
            add = float(lrec["components"]["additivity_max_dev"])
            assert add < ADDITIVITY_TOL, (k, layer, add)
            max_add = max(max_add, add)
    return {"additivity_max_dev": max_add, "tol": ADDITIVITY_TOL, "verdict": "PASS"}


def _register_cells(bank: RatioBank, npzs: dict[int, dict], layers: list[int]) -> None:
    """One (num, den) column per (basis, leg, layer, arm, component-or-wpar)
    from the merged per-fold TEST channels; matched contexts appear once."""
    for basis in BASES:
        for layer in layers:
            for leg, arms in (("c", CLEG_ARMS), ("z", MATCHED_ARMS)):
                for arm in arms:
                    key = f"{basis}_M{T2}{leg}_L{layer}|{arm}"
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
                        bank.add(f"{basis}|{leg}|{layer}|{arm}|{comp}", ids, num, den)
                    bank.add(f"{basis}|{leg}|{layer}|{arm}|wpar", ids, ch[:, CH["ss_tot_par"]], den)


def _signflip_d(
    bank: RatioBank, idx: dict[str, int], layers: list[int], n_draws: int
) -> dict[str, dict]:
    """Paired own↔ext label-flip null of D per (basis, layer, ext) — rng 1,
    ONE flip matrix shared across bases/layers (paired family)."""
    num, den = bank.stacks()
    rng = np.random.default_rng(SIGNFLIP_SEED)
    f = rng.integers(0, 2, size=(n_draws, bank.n)).astype(np.float64)
    g = 1.0 - f

    def _mixed_delta(basis: str, layer: int, comp: str, ext: str) -> np.ndarray:
        ca = idx[f"{basis}|c|{layer}|own|{comp}"]
        cb = idx[f"{basis}|c|{layer}|{ext}|{comp}"]
        a_side = _safe_ratio(f @ num[:, ca] + g @ num[:, cb], f @ den[:, ca] + g @ den[:, cb])
        b_side = _safe_ratio(g @ num[:, ca] + f @ num[:, cb], g @ den[:, ca] + f @ den[:, cb])
        return a_side - b_side

    out: dict[str, dict] = {}
    for ext in ("ext_plain", "ext_style"):
        out[ext] = {}
        for layer in layers:
            out[ext][str(layer)] = {}
            for basis in BASES:
                d_flip = _mixed_delta(basis, layer, "par", ext) - _mixed_delta(
                    basis, layer, "perp", ext
                )
                out[ext][str(layer)][basis] = d_flip
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Issue #1072 lowdim subspace stats (phase D)")
    p.add_argument(
        "--eval-dir",
        type=str,
        default=str(_REPO_ROOT / "eval_results" / "issue_1072" / "lowdim-token-subspace"),
        help="dir holding per_context_stats_lowdim_fold*.npz + battery_lowdim_fold*.json",
    )
    p.add_argument(
        "--parent-stats",
        type=str,
        default=str(_REPO_ROOT / "eval_results" / "issue_1072" / "stats_component.json"),
        help="the completed run's committed stats (the k=1 reference row; tolerated missing)",
    )
    p.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    t0 = time.time()
    eval_dir = pathlib.Path(args.eval_dir)
    out_path = pathlib.Path(args.out) if args.out else eval_dir / "stats_lowdim.json"

    npzs, recs = _load_folds(eval_dir)
    gates = _gate_fold_records(recs, args.smoke)
    layers = sorted(int(x) for x in recs[0]["regime"]["layers"])
    pool_ids = [int(i) for i in npzs[0]["ids_pool_full"].tolist()]
    for k in npzs:
        assert npzs[k]["ids_pool_full"].tolist() == pool_ids, f"fold {k}: pool drift"

    # Fold-identity defense in depth (g5 ran pod-side; recompute + re-assert).
    folds = make_kfold_splits(pool_ids, K_FOLDS)
    for f in folds:
        kfold_split_hashes(f)
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
    flips = _signflip_d(bank, idx, layers, args.n_draws)

    def _cell(basis: str, leg: str, layer: int, arm: str, comp: str) -> int:
        return idx[f"{basis}|{leg}|{layer}|{arm}|{comp}"]

    def _delta(basis: str, leg: str, layer: int, comp: str, ext: str, arr: np.ndarray):
        return (
            arr[..., _cell(basis, leg, layer, "own", comp)]
            - arr[..., _cell(basis, leg, layer, ext, comp)]
        )

    results: dict = {}
    d_by_basis_layer: dict[str, dict[str, dict]] = {}
    for ext in ("ext_plain", "ext_style"):
        results[ext] = {}
        for basis in BASES:
            by_layer = {}
            for layer in layers:
                dc = {comp: float(_delta(basis, "c", layer, comp, ext, obs)) for comp in COMPS}
                d_obs = dc["par"] - dc["perp"]
                d_draws = _delta(basis, "c", layer, "par", ext, draws) - _delta(
                    basis, "c", layer, "perp", ext, draws
                )
                dz = {comp: float(_delta(basis, "z", layer, comp, ext, obs)) for comp in COMPS}
                d_resid_draws = _delta(basis, "z", layer, "par", ext, draws) - _delta(
                    basis, "z", layer, "perp", ext, draws
                )
                closure = {}
                for comp in COMPS:
                    g0, gt = dc[comp], dz[comp]
                    dg_draws = _delta(basis, "c", layer, comp, ext, draws) - _delta(
                        basis, "z", layer, comp, ext, draws
                    )
                    frac_draws = 1.0 - _safe_ratio(
                        _delta(basis, "z", layer, comp, ext, draws),
                        _delta(basis, "c", layer, comp, ext, draws),
                    )
                    closure[comp] = {
                        "G0": g0,
                        "Gt": gt,
                        "delta_G": g0 - gt,
                        "delta_G_ci95": _ci(dg_draws),
                        "closure_frac": (1.0 - gt / g0) if abs(g0) > 1e-12 else None,
                        "closure_frac_ci95": _ci(frac_draws),
                    }
                s_par_draws = _safe_ratio(
                    _delta(basis, "c", layer, "par", ext, draws),
                    _delta(basis, "c", layer, "full", ext, draws),
                )
                wpar_own = float(obs[_cell(basis, "c", layer, "own", "wpar")])
                wpar_own_draws = draws[:, _cell(basis, "c", layer, "own", "wpar")]
                s_par = dc["par"] / dc["full"] if abs(dc["full"]) > 1e-12 else None
                enr_draws = _safe_ratio(s_par_draws, wpar_own_draws)
                flip_draws = flips[ext][str(layer)][basis]
                by_layer[str(layer)] = {
                    "delta_C": dc,
                    "delta_R2_full_identity_dev": abs(
                        dc["full"] - (dc["par"] + dc["perp"] + dc["cross"])
                    ),
                    "D": d_obs,
                    "D_ci95": _ci(d_draws),
                    "D_p_two_sided_signflip_raw": signflip_p_two(d_obs, flip_draws),
                    "signflip_null_band_p025_p975": [
                        float(np.nanpercentile(flip_draws, 2.5)),
                        float(np.nanpercentile(flip_draws, 97.5)),
                    ],
                    "zleg": {
                        "delta_C": dz,
                        "D_resid": dz["par"] - dz["perp"],
                        "D_resid_ci95": _ci(d_resid_draws),
                    },
                    "closure_by_component": closure,
                    "S_par": s_par,
                    "S_par_ci95": _ci(s_par_draws),
                    "w_par_own": wpar_own,
                    "w_par_own_ci95": _ci(wpar_own_draws),
                    "w_par_by_arm": {
                        a: float(obs[_cell(basis, "c", layer, a, "wpar")]) for a in CLEG_ARMS
                    },
                    "enrichment": (s_par / wpar_own)
                    if (s_par is not None and wpar_own > 0)
                    else None,
                    "enrichment_ci95": _ci(enr_draws),
                }
                d_by_basis_layer.setdefault(ext, {}).setdefault(str(layer), {})[basis] = {
                    "D": d_obs,
                    "D_ci95": by_layer[str(layer)]["D_ci95"],
                    "p_raw": by_layer[str(layer)]["D_p_two_sided_signflip_raw"],
                }
            results[ext][basis] = by_layer

    # ── registered lattice: Holm over the 3-basis family per layer (ext_plain) ──
    lattice: dict[str, dict] = {}
    for layer in layers:
        fam = d_by_basis_layer["ext_plain"][str(layer)]
        holm = holm_adjust({b: fam[b]["p_raw"] for b in BASES})
        cells = {}
        for b in BASES:
            cells[b] = {
                "D": fam[b]["D"],
                "D_ci95": fam[b]["D_ci95"],
                "p_two_sided_signflip_raw": fam[b]["p_raw"],
                "p_holm": holm[b],
                "verdict": lattice_verdict(fam[b]["D"], fam[b]["D_ci95"], holm[b]),
            }
        lattice[str(layer)] = {
            "cells": cells,
            "holm_alpha": HOLM_ALPHA,
            "is_primary": layer == PRIMARY_LAYER,
        }
    primary_key = str(PRIMARY_LAYER) if str(PRIMARY_LAYER) in lattice else str(max(layers))
    family = family_mapping({b: lattice[primary_key]["cells"][b]["verdict"] for b in BASES})

    # ── paired cross-basis D differences at the primary layer (same draws) ──────
    pairwise = {}
    for i, b1 in enumerate(BASES):
        for b2 in BASES[i + 1 :]:
            pk = int(primary_key)
            d1 = _delta(b1, "c", pk, "par", "ext_plain", draws) - _delta(
                b1, "c", pk, "perp", "ext_plain", draws
            )
            d2 = _delta(b2, "c", pk, "par", "ext_plain", draws) - _delta(
                b2, "c", pk, "perp", "ext_plain", draws
            )
            obs_diff = (
                d_by_basis_layer["ext_plain"][primary_key][b1]["D"]
                - d_by_basis_layer["ext_plain"][primary_key][b2]["D"]
            )
            pairwise[f"{b1}-{b2}"] = {"diff": obs_diff, "diff_ci95": _ci(d1 - d2)}

    # ── parent 1-D reference row (k=1; committed stats_component.json) ──────────
    parent_ref = None
    parent_p = pathlib.Path(args.parent_stats)
    if parent_p.exists():
        ps = json.loads(parent_p.read_text())
        parent_ref = {
            "source": str(parent_p),
            "by_layer": {
                la: {
                    "D": ps["by_ext"]["ext_plain"][la]["D"],
                    "D_ci95": ps["by_ext"]["ext_plain"][la]["D_ci95"],
                    "S_par": ps["by_ext"]["ext_plain"][la]["S_par"],
                    "S_par_ci95": ps["by_ext"]["ext_plain"][la]["S_par_ci95"],
                    "delta_C": ps["by_ext"]["ext_plain"][la]["delta_C"],
                }
                for la in ps["by_ext"]["ext_plain"]
            },
            "verdict": ps["lattice"]["verdict"],
        }
    else:
        logger.warning("parent stats missing at %s — k=1 reference row omitted", parent_p)

    # ── companions copied from the battery records (calibration fold) ───────────
    cal = recs[K_FOLDS - 1]["layers"]
    companions = {
        str(la): {
            "p_last_pooled": cal[str(la)].get("components", {}).get("p_last"),
            "cleg_rem_sensitivity": cal[str(la)].get("cleg_rem_sensitivity"),
            "sens_lambdas": cal[str(la)].get("sens_lambdas"),
        }
        for la in layers
        if str(la) in cal and not cal[str(la)].get("skipped")
    }
    capture_gates = None
    cg = eval_dir / "capture_gates_lowdim.json"
    if cg.exists():
        g = json.loads(cg.read_text())
        capture_gates = {k: g.get(k) for k in ("coverage", "overlap_hist", "effk_hist", "g7", "k2")}

    out = {
        "issue": 1072,
        "followup_label": "lowdim-token-subspace",
        "git_sha": _repo_git_sha(),
        "numpy_version": np.__version__,
        "ts": time.time(),
        "inputs": {
            "eval_dir": str(eval_dir),
            "n_draws": args.n_draws,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "signflip_seed": SIGNFLIP_SEED,
            "layers": layers,
            "bases": list(BASES),
            "smoke": bool(args.smoke),
        },
        "coverage": coverage,
        "fold_gates": gates,
        "bootstrap_parity": parity,
        "lattice_by_layer": lattice,
        "primary_layer": int(primary_key),
        "family_mapping": family,
        "pairwise_D_primary": pairwise,
        "by_ext": results,
        "parent_1d_reference": parent_ref,
        "companions": companions,
        "capture_gates_summary": capture_gates,
        "near_duplicate_caveat": NEAR_DUP_CAVEAT,
        "ci_conditionality": (
            "contexts resampled with the fold assignment FIXED — pooled intervals are "
            "conditional on the fold assignment (parent recipe verbatim); cross-basis and "
            "cross-component differences reuse the SAME draws per resample (paired)"
        ),
        "wall_seconds": time.time() - t0,
    }
    out_path.write_text(json.dumps(out, indent=2, default=_json_np))
    prim = lattice[primary_key]["cells"]
    logger.info(
        "[stats] %s written: L%s D(top8)=%.5f D(top32)=%.5f D(look8)=%.5f -> %s (%.1fs)",
        out_path,
        primary_key,
        prim["top8"]["D"],
        prim["top32"]["D"],
        prim["look8"]["D"],
        family["headline"],
        out["wall_seconds"],
    )


if __name__ == "__main__":
    main()
