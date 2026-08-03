"""Tier-1.5 transfer rung: frozen source slope, target-train intercept refit.

User-chat inline free-analysis round (2026-07-24; issues #825 / #1345 / #1639-#1310).
Decomposes each committed naive cross-cell transfer deficit into an OFFSET
(translation) component vs a LINEAR-PART (rotation/rescale/shear) component, by
inserting the missing rung between "naive transfer" and the fitted
reparameterization: keep the source cell's linear map frozen, refit ONLY the
intercept on the target's train folds.

Rungs (all held-out at layer 19, one shared grouped K=5 fold partition, seed 0):
  within  : target's own map (ceiling; equality-gated vs committed JSONs)
  naive   : source affine map applied verbatim (source centering + intercept),
            recomputed through the family's own committed core (equality gate)
  tier15  : frozen source linear part A_s; intercept refit on TARGET train
            folds (b* = ybar_t,tr - A_s xbar_t,tr, the least-squares-optimal
            intercept given a frozen slope). Implemented EXACTLY via the
            affine recentering identity:
              tier15(te) = naive(te) - mean(naive(tr_t)) + ybar_t,tr
            (naive predictions are affine in x, so recentering predictions
            with target train-fold statistics IS the intercept refit; the
            slope, selected lambda, and standardization are untouched).
  tier15d : secondary rung, + diagonal input rescale (sigma_s -> sigma_t,tr),
            via an input pre-transform fed to the UNCHANGED family core:
              X' = xmu_s + xsd_s * (X - xbar_t,tr) / xsd_t,tr
            so the core's internal (X' - xmu_s)/xsd_s equals
            (X - xbar_t,tr)/xsd_t,tr; then the same recentering identity.

Every rung's predictions come from each family's OWN committed ridge core
(no reimplemented ridge math):
  A) #825 base<->instruct (context arm): issue825_map_alignment
     (_ridge_prep/_ridge_predict; default GCV, no dof cap)
  B) #1345 chat<->no-template x {instruct, pretrained} x {context, prefix}:
     issue825_crossmodel_map_transfer (_prep_fold/_ridge_predict_cached)
  C) #1310/#1639 character 4x4 x {base, instruct} (scene-aggregated store):
     issue825_fit_cells core with GCV_DOF_CAP = 0.9 (the xpersona rig)

CLI:
  uv run python scripts/issue1639_tier15_intercept_refit.py --families A,C
  uv run python scripts/issue1639_tier15_intercept_refit.py --families B \
      --turnstore-dir /mnt/eps-data/thomasjiralerspong/issue1345_tier15_stage/\
issue1345_framing/analysis_tensors/turnstore
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps (#847) must bind before torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

GATE_TOL_WITHIN = 0.01  # ma.GATE_TOL precedent (#825 matched-null round)
GATE_TOL_COMMITTED = 0.01  # committed-JSON reproduction tolerance
L19 = 19


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _r2_foldmean(ss: dict) -> float:
    return float("nan") if ss["tot"] < 1e-12 else 1.0 - ss["res"] / ss["tot"]


def _r2_global(preds: np.ndarray, true: np.ndarray) -> float:
    mu = true.mean(0)
    sst = float(np.sum((true - mu) ** 2))
    return float("nan") if sst < 1e-12 else 1.0 - float(np.sum((true - preds) ** 2)) / sst


class RungAccum:
    """Accumulate per-fold test predictions for one rung of one direction."""

    def __init__(self, n_rows: int, dim: int):
        self.preds = np.zeros((n_rows, dim), np.float64)
        self.covered = np.zeros(n_rows, bool)
        self.ss = {"res": 0.0, "tot": 0.0}

    def add_fold(self, te: np.ndarray, pred_te: np.ndarray, true_te: np.ndarray) -> None:
        self.preds[te] = pred_te
        self.covered[te] = True
        mu = true_te.mean(0)
        self.ss["res"] += float(np.sum((true_te - pred_te) ** 2))
        self.ss["tot"] += float(np.sum((true_te - mu) ** 2))

    def result(self, true_full: np.ndarray) -> dict:
        cov = self.covered
        return {
            "r2_foldmean": _r2_foldmean(self.ss),
            "r2_global": _r2_global(self.preds[cov], true_full[cov].astype(np.float64)),
            "n_covered": int(cov.sum()),
        }


def _diag_pretransform(X_all: np.ndarray, tr_src_X: np.ndarray, tr_tgt_X: np.ndarray) -> np.ndarray:
    """X' = xmu_s + xsd_s * (X - xbar_t,tr)/xsd_t,tr (fp64; torch std ddof=1 parity)."""
    xs = torch.as_tensor(np.asarray(tr_src_X), dtype=torch.float64)
    xt = torch.as_tensor(np.asarray(tr_tgt_X), dtype=torch.float64)
    xa = torch.as_tensor(np.asarray(X_all), dtype=torch.float64)
    mu_s, sd_s = xs.mean(0), xs.std(0) + 1e-9
    mu_t, sd_t = xt.mean(0), xt.std(0) + 1e-9
    return (mu_s + sd_s * (xa - mu_t) / sd_t).numpy()


def run_direction(
    predict_all,  # fn(X_train_src, Y_train_src, X_eval_all) -> preds over all target rows
    Xs: np.ndarray,
    Ys: np.ndarray,
    Xt: np.ndarray,
    Yt: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
) -> dict:
    """All four rungs for one ordered (source -> target) direction."""
    n, d = Yt.shape
    acc = {k: RungAccum(n, d) for k in ("within", "naive", "tier15", "tier15d")}
    for k in range(n_folds):
        tr = folds != k
        te = folds == k
        if te.sum() == 0 or tr.sum() < 3:
            continue
        true_te = Yt[te].astype(np.float64)
        ybar_tr = Yt[tr].astype(np.float64).mean(0)
        # within ceiling (target's own map)
        pw = predict_all(Xt[tr], Yt[tr], Xt[te])
        acc["within"].add_fold(te, pw, true_te)
        # naive + diag rungs share ONE source fit: concat the two eval matrices
        # (naive: raw target rows; diag: pre-transformed rows) so the expensive
        # train-side eigh runs once per (source, fold).
        Xp = _diag_pretransform(Xt, Xs[tr], Xt[tr])
        p_cat = predict_all(Xs[tr], Ys[tr], np.concatenate([Xt, Xp], axis=0))
        pn_all, pd_all = p_cat[: len(Xt)], p_cat[len(Xt) :]
        acc["naive"].add_fold(te, pn_all[te], true_te)
        # tier15: recentering identity (intercept refit on target train folds)
        acc["tier15"].add_fold(te, pn_all[te] - pn_all[tr].mean(0) + ybar_tr, true_te)
        # tier15d: + diagonal input rescale via pre-transform, then recenter
        acc["tier15d"].add_fold(te, pd_all[te] - pd_all[tr].mean(0) + ybar_tr, true_te)
    return {k: a.result(Yt) for k, a in acc.items()}


def _fractions(res: dict, convention: str) -> dict:
    ceil = res["within"][convention]
    out = {}
    for k in ("naive", "tier15", "tier15d"):
        out[k] = float("nan") if abs(ceil) < 1e-12 else res[k][convention] / ceil
    return out


def _gate(name: str, got: float, want: float, tol: float, gates: list) -> None:
    ok = abs(got - want) <= tol
    gates.append({"name": name, "got": got, "want": want, "tol": tol, "pass": bool(ok)})
    status = "PASS" if ok else "FAIL"
    print(f"[gate:{status}] {name}: got {got:.4f} want {want:.4f} (tol {tol})", flush=True)
    assert ok, f"equality gate failed: {name} got {got:.4f} want {want:.4f}"


# ---------------------------------------------------------------------------
# Family A — #825 base<->instruct, context arm (map_alignment store)
# ---------------------------------------------------------------------------
def family_a(dl_dir: Path) -> dict:
    import issue825_map_alignment as ma

    data, conv, _layers, _al = ma._load_pair(
        dl_dir / f"{ma.STEM_INSTRUCT}.npz", dl_dir / f"{ma.STEM_BASE}.npz", [L19]
    )
    folds = ma._cv_folds(np.asarray(conv), ma.N_FOLDS, ma.FIT_SEED)
    cells = {k: data[k][L19].cpu().numpy() for k in ("Xi", "Yi", "Xb", "Yb")}

    def predict_all(Xtr, Ytr, Xev):
        prep = ma._ridge_prep(torch.as_tensor(Xtr, dtype=torch.float64))
        pred = ma._ridge_predict(
            prep,
            torch.as_tensor(Ytr, dtype=torch.float64),
            torch.as_tensor(Xev, dtype=torch.float64),
        )
        return pred.cpu().numpy().astype(np.float64)

    t0 = time.time()
    # 1-cell pilot: within-instruct fold 0 through the production core, timed.
    tr, te = folds != 0, folds == 0
    _ = predict_all(cells["Xi"][tr], cells["Yi"][tr], cells["Xi"][te])
    pilot_s = time.time() - t0
    print(f"[pilot] family A one fold (n_tr={int(tr.sum())}): {pilot_s:.1f}s", flush=True)

    out, gates = {}, []
    directions = {
        "b2i": ("Xb", "Yb", "Xi", "Yi"),
        "i2b": ("Xi", "Yi", "Xb", "Yb"),
    }
    for dname, (xs, ys, xt, yt) in directions.items():
        res = run_direction(
            predict_all, cells[xs], cells[ys], cells[xt], cells[yt], folds, ma.N_FOLDS
        )
        res["fractions_foldmean"] = _fractions(res, "r2_foldmean")
        res["fractions_global"] = _fractions(res, "r2_global")
        out[dname] = res
    _gate(
        "A.within_instruct(b2i tgt ceiling)",
        out["b2i"]["within"]["r2_foldmean"],
        0.673,
        GATE_TOL_WITHIN,
        gates,
    )
    _gate(
        "A.within_base(i2b tgt ceiling)",
        out["i2b"]["within"]["r2_foldmean"],
        0.588,
        GATE_TOL_WITHIN,
        gates,
    )
    return {"directions": out, "gates": gates, "n_rows": len(conv), "pilot_seconds": pilot_s}


# ---------------------------------------------------------------------------
# Family B — #1345 chat<->no-template x model x arm (turnstore)
# ---------------------------------------------------------------------------
def family_b(turnstore_dir: Path) -> dict:
    import issue825_crossmodel_map_transfer as cm
    import issue825_fit_cells as fit825
    from issue1345_cross_regime_transfer import load_arm_xy
    from issue1345_fit_cells import load_regime_bundle

    out, gates = {}, []
    committed_dir = _REPO_ROOT / "eval_results/issue_1345"
    for model in ("instruct", "pretrained"):
        bundles = {r: load_regime_bundle(turnstore_dir, model, r) for r in ("r1", "r2")}
        for arm in ("context", "prefix"):
            xy = {r: load_arm_xy(bundles[r], r, arm) for r in ("r1", "r2")}
            common = np.intersect1d(xy["r1"]["conv_ids"], xy["r2"]["conv_ids"])
            cells = {}
            for r in ("r1", "r2"):
                keep = np.isin(xy[r]["conv_ids"], common)
                cells[r] = {
                    "X": xy[r]["X"][keep][:, L19, :].astype(np.float64),
                    "Y": xy[r]["Y"][keep][:, L19, :].astype(np.float64),
                    "conv": xy[r]["conv_ids"][keep],
                }
            assert np.array_equal(cells["r1"]["conv"], cells["r2"]["conv"])
            folds = fit825._cv_folds(cells["r1"]["conv"], cm.N_FOLDS, cm.FIT_SEED)

            def predict_all(Xtr, Ytr, Xev):
                cache = cm._prep_fold(Xtr, Xev)
                return cm._ridge_predict_cached(cache, Ytr).astype(np.float64)

            slug = "base" if model == "pretrained" else model
            committed = json.loads(
                (committed_dir / f"cross_regime_transfer_{slug}_{arm}.json").read_text()
            )
            for src, tgt in (("r1", "r2"), ("r2", "r1")):
                res = run_direction(
                    predict_all,
                    cells[src]["X"],
                    cells[src]["Y"],
                    cells[tgt]["X"],
                    cells[tgt]["Y"],
                    folds,
                    cm.N_FOLDS,
                )
                res["fractions_foldmean"] = _fractions(res, "r2_foldmean")
                res["fractions_global"] = _fractions(res, "r2_global")
                key = f"{model}.{arm}.{src}->{tgt}"
                out[key] = res
                pair = committed["matrix"][f"{src}->{tgt}"]
                _gate(
                    f"B.naive.{key}",
                    res["naive"]["r2_global"],
                    float(pair["transfer_r2_by_layer"][str(L19)]),
                    GATE_TOL_COMMITTED,
                    gates,
                )
                _gate(
                    f"B.within.{key}",
                    res["within"]["r2_global"],
                    float(pair["target_within_r2_by_layer"][str(L19)]),
                    GATE_TOL_COMMITTED,
                    gates,
                )
            del cells, xy
        del bundles
    return {"directions": out, "gates": gates}


# ---------------------------------------------------------------------------
# Family C — #1310/#1639 character 4x4 (scene-aggregated store)
# ---------------------------------------------------------------------------
def family_c(store_root: Path) -> dict:
    import issue825_fit_cells as fit825
    import issue1310_xpersona_similarity as v1

    fit825.GCV_DOF_CAP = v1.DOF_CAP  # the xpersona rig's mandatory cap (0.9)

    def predict_all(Xtr, Ytr, Xev):
        cache = fit825._prep_fold(Xtr, Xev)
        return fit825._ridge_predict_cached(cache, Ytr).astype(np.float64)

    out, gates = {}, []
    committed_dir = _REPO_ROOT / "eval_results/issue_1310/onpolicy_aggregated"
    for model in ("base", "instruct"):
        arrays = v1.load_persona_arrays(store_root, model)
        cells = {
            p: {
                "X": arrays[p]["X"][:, L19, :].astype(np.float64),
                "Y": arrays[p]["Y"][:, L19, :].astype(np.float64),
                "folds": arrays[p]["folds"],
            }
            for p in v1.PERSONAS
        }
        del arrays
        for p in v1.PERSONAS:
            committed = json.loads((committed_dir / f"cells_agg_{model}_{p}.json").read_text())
            want = float(committed["r2_per_layer_obs"][L19])
            res = run_direction(
                predict_all,
                cells[p]["X"],
                cells[p]["Y"],
                cells[p]["X"],
                cells[p]["Y"],
                cells[p]["folds"],
                v1.N_FOLDS,
            )
            _gate(
                f"C.within.{model}.{p}",
                res["within"]["r2_foldmean"],
                want,
                GATE_TOL_COMMITTED,
                gates,
            )
        for src in v1.PERSONAS:
            for tgt in v1.PERSONAS:
                if src == tgt:
                    continue
                res = run_direction(
                    predict_all,
                    cells[src]["X"],
                    cells[src]["Y"],
                    cells[tgt]["X"],
                    cells[tgt]["Y"],
                    cells[tgt]["folds"],
                    v1.N_FOLDS,
                )
                res["fractions_foldmean"] = _fractions(res, "r2_foldmean")
                res["fractions_global"] = _fractions(res, "r2_global")
                out[f"{model}.{src}->{tgt}"] = res
        del cells
    return {"directions": out, "gates": gates}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--families", default="A,C")
    ap.add_argument("--dl-dir-825", type=Path, default=Path("data/issue_825/hf_dl/map_alignment"))
    ap.add_argument(
        "--turnstore-dir",
        type=Path,
        default=Path(
            "/mnt/eps-data/thomasjiralerspong/issue1345_tier15_stage/"
            "issue1345_framing/analysis_tensors/turnstore"
        ),
    )
    ap.add_argument(
        "--store-root",
        type=Path,
        default=Path(
            "/mnt/eps-data/thomasjiralerspong/issue1310_xpersona/hf_dl/"
            "issue1310_char_map/analysis_tensors/store_onpolicy"
        ),
    )
    ap.add_argument("--out-root", type=Path, default=Path("eval_results"))
    args = ap.parse_args()

    meta = {
        "script": "scripts/issue1639_tier15_intercept_refit.py",
        "git_commit": _git_commit(),
        "layer": L19,
        "date": "2026-07-24",
        "provenance": "user-chat inline free-analysis round (tier-1.5 intercept refit)",
    }
    fam_out = {
        "A": ("issue_825/tier15_intercept_refit", lambda: family_a(args.dl_dir_825)),
        "B": ("issue_1345/tier15_intercept_refit", lambda: family_b(args.turnstore_dir)),
        "C": (
            "issue_1310/xpersona_similarity/tier15_intercept_refit",
            lambda: family_c(args.store_root),
        ),
    }
    for fam in args.families.split(","):
        rel, fn = fam_out[fam]
        t0 = time.time()
        print(f"[family {fam}] start", flush=True)
        res = fn()
        res["meta"] = dict(meta, family=fam, wall_seconds=time.time() - t0)
        out_dir = args.out_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(res, indent=2))
        print(f"[family {fam}] done in {time.time() - t0:.0f}s -> {out_dir}/results.json")
        for key, r in res["directions"].items():
            fr = r["fractions_foldmean"]
            print(
                f"  {key}: ceil={r['within']['r2_foldmean']:.3f} "
                f"naive={r['naive']['r2_foldmean']:.3f} ({fr['naive']:.2f}) "
                f"t15={r['tier15']['r2_foldmean']:.3f} ({fr['tier15']:.2f}) "
                f"t15d={r['tier15d']['r2_foldmean']:.3f} ({fr['tier15d']:.2f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
