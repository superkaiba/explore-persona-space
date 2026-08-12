"""Pool-then-specialize TRANSFER-TIER extension (#2054 Gap B follow-on).

The parent ``issue2054_pool_specialize.py`` ran the capacity nesting
M0 (pooled direct) -> M1 (+per-cell bias) -> M2 (+per-cell additive rank-k
slope correction). This script adds the TRANSFORMATION-CLASS rungs from the
transfer-tier ladder, pooled->cell, per unit (cell x arm) x fold — all
closed-form, no iterative fits:

    m0          pooled map as-is (recomputed for row-alignment with the parent)
    ctx_offset  y = W(x - dx) + b_pool;  dx = mean(cell train X) - mean(pool X)
                (CLOUDS-ONLY: fit without context->answer pairs)
    ans_offset  y = Wx + b_pool + dy;    dy = mean(cell train Y) - mean(pool Y)
                (CLOUDS-ONLY)
    m1          y = Wx + b*   (pairs-fit bias refit — the parent's M1, recomputed)
    scale       y = a Wx + b*                 (1 scalar + bias, pairs)
    rot         y = R Wx + b*, R^T R = I      (orthogonal Procrustes, pairs)
    rot_scale   y = a R Wx + b*               (scaled Procrustes; a and R from
                                               the SAME SVD as `rot`)

Compositional low-rank A/B rungs are DELIBERATELY ABSENT — they are the
parent's M2 in disguise here: with a square, invertible pooled map W,
{W + C : rank C <= k} = {W(I + P) : rank P <= k} = {(I + Q)W : rank Q <= k}
(P = W^-1 C, Q = C W^-1 preserve rank), so the model classes coincide and
exact reduced-rank LS gives IDENTICAL predictions for regressor x vs
regressor Wx (their fitted-value column spaces are equal). The
``--equivalence-check`` mode verifies this numerically on real cells instead
of burning the grid on an algebraic identity. Clouds-only full A/B has no
pooled->cell definition at all: the pool has no per-row counterpart for a
cell row (the cell IS a subset of the pool), so only the translation special
cases exist — which are exactly ctx_offset / ans_offset above.

No per-rung null draws: every rung is nested between the parent's M0 and the
banked within-cell ceiling, both of which carry banked nulls; the read here
is relative recovery along the nesting, on held-out folds.

Degeneracies follow the parent's named-and-recorded convention (never a
crash, never a silent clamp): constant-X cells (chat-form prefix renders)
skip the pairs/cloud rungs that are undefined there, with the reason in the
artifact.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    # script mode puts scripts/ (not the repo root) on sys.path[0] (gotchas.md).
    sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.linalg import svd as scipy_svd

from explore_persona_space.analysis.mapping_baselines import identity_bias_predict, knn_retrieval
from explore_persona_space.experiments.issue_779.fit_h import reconstruction_metrics
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts.issue2054_ctx2ctx_fit import (
    ARM_VEC_KEY,
    ARMS,
    D_AMBIENT,
    Cell,
    discover_cells,
    load_fold_map,
)
from scripts.issue2054_pool_specialize import (
    CONSTANT_X_VAR_FLOOR,
    PooledMomentRidge,
    _log,
    accumulate_pooled_moments,
    fit_pooled_per_fold,
    join_cell,
    load_cell_with_answer,
)

SCRIPT_VERSION = "issue2054_pool_rungs_v1"

RUNGS = ("m0", "ctx_offset", "ans_offset", "m1", "scale", "rot", "rot_scale")
# ||centered Z||_F^2 below this is a degenerate (constant-prediction) cell —
# alpha / Procrustes are undefined; same convention family as the parent's
# CONSTANT_X_VAR_FLOOR (variance scale, not raw norm).
ZC_VAR_FLOOR = 1e-12


def _knn_block(preds: np.ndarray, true: np.ndarray) -> dict:
    """Retrieval read (standing baseline duty) — euclidean + cosine, k in {1,5,10}."""
    return {
        metric: knn_retrieval(preds, true, ks=(1, 5, 10), metric=metric)
        for metric in ("euclidean", "cosine")
    }


def _procrustes(zc_tr: np.ndarray, yc_tr: np.ndarray) -> tuple[np.ndarray, float]:
    """Orthogonal Procrustes min_R ||yc - zc R||_F, R^T R = I, plus the scaled-
    Procrustes gain from the same SVD.

    R = U V^T with U S V^T = svd(zc^T yc); a_rot = trace(S) / ||zc||_F^2.
    float64 throughout; gesdd (LAPACK divide-and-conquer).
    """
    cross = zc_tr.astype(np.float64).T @ yc_tr.astype(np.float64)  # (d, d)
    u, s, vt = scipy_svd(cross, lapack_driver="gesdd")
    r = u @ vt
    a_rot = float(s.sum() / max((zc_tr.astype(np.float64) ** 2).sum(), 1e-300))
    return r, a_rot


def run_unit(
    cell: Cell,
    arm: str,
    fold_map: dict,
    pooled_models: dict[int, PooledMomentRidge],
    folds_to_run: list[int],
    out_path: Path,
    fingerprint: str,
) -> None:
    t_unit = time.time()
    k = int(fold_map["k"])
    vec = ARM_VEC_KEY[arm]
    act = load_cell_with_answer(cell)
    j = join_cell(act, fold_map["fold_of"], k, arm)
    x_all = np.asarray(act[vec][j["rows"]], dtype=np.float64)
    y_all = np.asarray(act["v_A"][j["rows"]], dtype=np.float64)

    fold_records: list[dict] = []
    for f in folds_to_run:
        t0 = time.time()
        te = j["fold_rows"][f]
        tr = np.concatenate([j["fold_rows"][g] for g in range(k) if g != f])
        x_tr, y_tr, x_te, y_te = x_all[tr], y_all[tr], x_all[te], y_all[te]
        n_tr = int(x_tr.shape[0])
        m0 = pooled_models[f]

        # Degeneracy screens (parent conventions: named + recorded, never fatal).
        y_ss = float(((y_te - y_te.mean(0)) ** 2).sum())
        if y_ss < 1e-18:
            fold_records.append(
                {
                    "fold": f,
                    "n_cell_train": n_tr,
                    "n_test": int(len(te)),
                    "skipped": "constant-vector Y_eval — R^2 undefined",
                }
            )
            _log(f"[poolrungs] {cell.key} arm={arm} fold={f} SKIPPED (constant Y)")
            continue

        mu_x_np = m0.mu_x.cpu().numpy()
        mu_y_np = m0.mu_y.cpu().numpy()
        z_tr = m0.predict_np(x_tr)
        z_te = m0.predict_np(x_te)

        preds: dict[str, np.ndarray] = {"m0": z_te}
        rung_info: dict[str, dict] = {}

        # Cloud-fit offsets (no pairs). predict_np is affine, so shifting the
        # input realizes y = W(x - dx) + b_pool exactly.
        dx = x_tr.mean(axis=0) - mu_x_np
        preds["ctx_offset"] = m0.predict_np(x_te - dx)
        rung_info["ctx_offset"] = {"dx_norm": float(np.linalg.norm(dx))}
        dy = y_tr.mean(axis=0) - mu_y_np
        preds["ans_offset"] = z_te + dy
        rung_info["ans_offset"] = {"dy_norm": float(np.linalg.norm(dy))}

        # Pairs-fit bias refit (the parent's M1, recomputed for alignment).
        b_cf = (y_tr - z_tr).mean(axis=0)
        preds["m1"] = z_te + b_cf
        rung_info["m1"] = {"bias_norm": float(np.linalg.norm(b_cf))}

        # Centered train clouds for the gain/rotation family.
        z_bar, y_bar = z_tr.mean(axis=0), y_tr.mean(axis=0)
        zc_tr, yc_tr = z_tr - z_bar, y_tr - y_bar
        zc_ss = float((zc_tr**2).sum())
        x_var_max = float(((x_tr - x_tr.mean(0)) ** 2).mean(axis=0).max())
        degenerate = None
        if x_var_max < CONSTANT_X_VAR_FLOOR:
            degenerate = "constant_x"
        elif zc_ss / max(n_tr, 1) < ZC_VAR_FLOOR:
            degenerate = "constant_pooled_prediction"

        if degenerate:
            # alpha and R are undefined on a constant cloud: substitute M1 and
            # record the reason — the parent's skip-and-name convention.
            for name in ("scale", "rot", "rot_scale"):
                preds[name] = preds["m1"]
                rung_info[name] = {"skipped": degenerate}
            _log(
                f"[poolrungs] {cell.key} arm={arm} fold={f} gain/rot rungs "
                f"SKIPPED ({degenerate}) — M1 substituted"
            )
        else:
            zc_te = z_te - z_bar
            alpha = float((zc_tr * yc_tr).sum() / zc_ss)
            preds["scale"] = y_bar + alpha * zc_te
            rung_info["scale"] = {"alpha": alpha}
            t_r = time.time()
            r_mat, a_rot = _procrustes(zc_tr, yc_tr)
            preds["rot"] = y_bar + zc_te @ r_mat
            preds["rot_scale"] = y_bar + a_rot * (zc_te @ r_mat)
            rung_info["rot"] = {"svd_wall_s": round(time.time() - t_r, 1)}
            rung_info["rot_scale"] = {"alpha_rot": a_rot}

        metrics = {name: reconstruction_metrics(preds[name], y_te) for name in RUNGS}
        metrics["identity_cell"] = reconstruction_metrics(
            identity_bias_predict(x_tr, y_tr, x_te), y_te
        )
        knn = {name: _knn_block(preds[name], y_te) for name in ("m0", "m1", "rot_scale")}

        rec = {
            "fold": f,
            "n_pooled_train": m0.n_train,
            "n_cell_train": n_tr,
            "n_test": int(len(te)),
            "d_ambient": D_AMBIENT,
            "regime_cell": "ambient" if n_tr > D_AMBIENT else "reduced_basis_descriptive",
            "degenerate_gain_rot": degenerate,
            "pooled_info": m0.info(),
            "rung_info": rung_info,
            "metrics": metrics,
            "knn": knn,
            "wall_s": round(time.time() - t0, 1),
        }
        fold_records.append(rec)
        _log(
            f"[poolrungs] {cell.key} arm={arm} fold={f} "
            f"m0={metrics['m0']['r2']:+.4f} ctxoff={metrics['ctx_offset']['r2']:+.4f} "
            f"ansoff={metrics['ans_offset']['r2']:+.4f} m1={metrics['m1']['r2']:+.4f} "
            f"scale={metrics['scale']['r2']:+.4f} rot={metrics['rot']['r2']:+.4f} "
            f"rotscale={metrics['rot_scale']['r2']:+.4f} elapsed={rec['wall_s']}s"
        )

    payload = {
        "metadata": {
            **as_metadata_dict(git_provenance(_REPO)),
            "script_version": SCRIPT_VERSION,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "cell": cell.key,
        "arm": arm,
        "rungs": list(RUNGS),
        "n_join": j["n_join"],
        "fingerprint": fingerprint,
        "folds": fold_records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.json")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out_path)
    _log(
        f"[poolrungs] unit {cell.key}__{arm} CHECKPOINTED -> {out_path} "
        f"(wall={round(time.time() - t_unit)}s)"
    )


def equivalence_check(
    cells: list[Cell],
    fold_map: dict,
    pooled_models_by_arm: dict[str, dict[int, PooledMomentRidge]],
    n_cells: int,
    rank: int,
) -> None:
    """Numerically verify the compositional-equals-additive collapse on real
    cells: exact reduced-rank LS of the M1 residuals on regressor X (additive)
    vs regressor Z = pooled prediction (compositional answer-side) must give
    IDENTICAL predictions when the pooled map is invertible (fitted-value
    column spaces coincide). Prints the max |delta| and the pooled map's
    smallest singular value; raises on violation."""
    k = int(fold_map["k"])
    checked = 0
    for cell in cells:
        if checked >= n_cells:
            break
        arm = "context"
        m0 = pooled_models_by_arm[arm][0]
        act = load_cell_with_answer(cell)
        j = join_cell(act, fold_map["fold_of"], k, arm)
        x_all = np.asarray(act[ARM_VEC_KEY[arm]][j["rows"]], dtype=np.float64)
        y_all = np.asarray(act["v_A"][j["rows"]], dtype=np.float64)
        te = j["fold_rows"][0]
        tr = np.concatenate([j["fold_rows"][g] for g in range(k) if g != 0])
        x_tr, y_tr, x_te = x_all[tr], y_all[tr], x_all[te]
        if float(((x_tr - x_tr.mean(0)) ** 2).mean(axis=0).max()) < CONSTANT_X_VAR_FLOOR:
            continue  # need a healthy cell
        z_tr, z_te = m0.predict_np(x_tr), m0.predict_np(x_te)
        b_cf = (y_tr - z_tr).mean(axis=0)
        r_tr = y_tr - (z_tr + b_cf)

        def rrr_predict(f_tr: np.ndarray, f_te: np.ndarray) -> np.ndarray:
            """Exact rank-`rank` reduced-rank LS of r_tr on regressor f."""
            fc_tr = f_tr - f_tr.mean(0)
            fc_te = f_te - f_tr.mean(0)
            beta, *_ = np.linalg.lstsq(fc_tr, r_tr, rcond=None)
            fitted = fc_tr @ beta
            _, _, vt = scipy_svd(fitted, full_matrices=False, lapack_driver="gesdd")
            pk = vt[:rank].T @ vt[:rank]  # project fitted values onto top-k
            return fc_te @ (beta @ pk)

        p_add = rrr_predict(x_tr, x_te)
        p_comp = rrr_predict(z_tr, z_te)
        delta = float(np.abs(p_add - p_comp).max())
        scale = float(np.abs(p_add).max())
        # Pooled-map invertibility: smallest singular value of the standardized map.
        s_min = float(np.linalg.svd(m0.map.cpu().numpy(), compute_uv=False).min())
        rel = delta / max(scale, 1e-30)
        _log(
            f"[poolrungs] equivalence {cell.key}: max|add - comp| = {delta:.3e} "
            f"(rel {rel:.3e}), pooled-map s_min = {s_min:.3e}"
        )
        if rel > 1e-6:
            raise RuntimeError(
                f"compositional/additive equivalence VIOLATED on {cell.key}: rel {rel:.3e} "
                f"(s_min={s_min:.3e}) — the collapse argument does not hold here."
            )
        checked += 1
    if checked == 0:
        raise RuntimeError("equivalence check found no healthy cell to run on")
    _log(f"[poolrungs] equivalence check PASSED on {checked} cells (rank={rank})")


def _fingerprint(args: argparse.Namespace, fold_map: dict, arm: str, cell_key: str) -> str:
    h = hashlib.sha256()
    h.update(
        json.dumps(
            {
                "v": SCRIPT_VERSION,
                "arm": arm,
                "cell": cell_key,
                "folds": args.folds or "all",
                "fold_map_sha": fold_map["_sha256"],
                "rungs": RUNGS,
            },
            sort_keys=True,
        ).encode()
    )
    return h.hexdigest()[:16]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--activations-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--folds", nargs="+", type=int, default=None)
    p.add_argument("--fold-map-file", type=Path, default=None)
    p.add_argument("--fold-map-ref", default="origin/issue-2054")
    p.add_argument("--device", default="cpu")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--pilot", action="store_true", help="fold 0, first cell per arm only")
    p.add_argument("--import-check", action="store_true")
    p.add_argument(
        "--equivalence-check",
        type=int,
        default=0,
        metavar="N_CELLS",
        help="verify compositional==additive on N healthy cells (rank 32), then exit",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[poolrungs] import-check OK")
        return 0

    t_start = time.time()
    fold_map = load_fold_map(args.fold_map_file, args.fold_map_ref)
    k = int(fold_map["k"])
    _log(
        f"[poolrungs] fold map {fold_map['_source']} k={k} "
        f"n_conv={len(fold_map['fold_of']):,} sha={fold_map['_sha256'][:12]}"
    )
    cells = discover_cells(args.activations_dir)
    folds_to_run = args.folds if args.folds else ([0] if args.pilot else list(range(k)))
    if any(f < 0 or f >= k for f in folds_to_run):
        raise ValueError(f"--folds out of range for k={k}: {folds_to_run}")
    _log(f"[poolrungs] {len(cells)} cells, arms={args.arms}, folds={folds_to_run}")

    acc = accumulate_pooled_moments(cells, fold_map["fold_of"], k, args.arms, args.device)
    pooled_by_arm = {
        arm: fit_pooled_per_fold(acc["mom"][arm], folds_to_run, k) for arm in args.arms
    }

    if args.equivalence_check:
        equivalence_check(cells, fold_map, pooled_by_arm, args.equivalence_check, rank=32)
        return 0

    units = [(c, a) for a in args.arms for c in cells]
    units = units[args.shard :: args.num_shards]
    if args.pilot:
        units = units[:1]
    _log(f"[poolrungs] shard {args.shard}/{args.num_shards}: {len(units)} units")

    n_done = 0
    for cell, arm in units:
        fp = _fingerprint(args, fold_map, arm, cell.key)
        out_path = args.out_root / "percell_rungs" / f"{cell.key}__{arm}.json"
        if out_path.exists():
            try:
                prior = json.loads(out_path.read_text())
            except json.JSONDecodeError:
                prior = {}
            if prior.get("fingerprint") == fp:
                _log(f"[poolrungs] unit {cell.key}__{arm} already done — resume skip")
                n_done += 1
                continue
        run_unit(cell, arm, fold_map, pooled_by_arm[arm], folds_to_run, out_path, fp)
        n_done += 1
        _log(f"[poolrungs] progress {n_done}/{len(units)}")

    _log(f"[poolrungs] done units={n_done} wall={round(time.time() - t_start)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
