"""Estimator-selector audit for the #825 cells (Phase 0 of the 21-condition build).

User-chat inline free-analysis round (2026-07-25; explicit user inline override).
Answers two questions the settle-it battery (guarded vs unguarded GCV only) cannot:

  Leg A -- is the guarded reading an artifact of the arbitrary 0.9 dof cap?
     Refit every locally-staged cell with ``lambda_selection="inner-group-cv"``,
     which picks lambda by summed inner-fold held-out RSS and NEVER evaluates the
     GCV dof formula -- so it is independent of ``GCV_DOF_CAP`` and carries no
     free constant. Agreement with the guarded GCV number removes the cap from
     the load-bearing path.

  Leg B -- is n=5000 actually required, or would n=3000 do?
     Matched-n x selector curve on the Track-S chat cells. n_tr = 0.8n crosses
     D=3584 at n=4480, so the curve locates the degenerate-regime boundary
     empirically and sizes the follow-on build.

Both legs collect the SELECTED LAMBDA per (layer, fold) to confirm or refute the
proposed mechanism: unguarded GCV pinning at the lambda-grid floor when
n_tr < D.

Every fit goes through the family's OWN committed core
(``issue825_fit_cells.heldout_r2_sweep``) via the settle-it battery's
``load_cell_xy`` loader -- no reimplemented ridge math. Fits are restricted to
the layer-19 headline slice (a 4x cut over the 4-layer WANT_LAYERS set).

Read-only w.r.t. every shared store; writes only under
``eval_results/issue_825/selector_audit/``.

CLI:
  uv run python scripts/issue825_selector_audit.py --leg a --cell M_instruct_user_chat
  uv run python scripts/issue825_selector_audit.py --leg b --cell S_instruct_chat
  uv run python scripts/issue825_selector_audit.py --merge
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

load_dotenv()  # thread caps (#847) before torch/numpy import

import numpy as np  # noqa: E402

import issue825_fit_cells as fit  # noqa: E402
import issue825_trackm_settle_battery as bat  # noqa: E402

OUT_DIR = _REPO_ROOT / "eval_results/issue_825/selector_audit"
HEADLINE_LAYER = 19
DOF_CAP = bat.DOF_CAP  # 0.9
N_FOLDS = bat.N_FOLDS  # 5
FOLD_SEED = bat.FOLD_SEED  # 0
D_MODEL = 3584

# Leg B matched-n grid. n_tr = 0.8 * n; D = 3584 -> crossover at n = 4480.
N_LEVELS = (500, 1000, 2000, 3000, 4000, 5000)
N_SEEDS = (1000, 1001, 1002)  # subsample rngs, battery convention
SELECTORS = ("gcv_unguarded", "gcv_guarded", "inner_group_cv")

# (cell_id, model, fmt, track, slot_index, turn_index)
LEG_A_CELLS: dict[str, tuple[str, str, str, int, int]] = {
    f"M_{m}_{role}_{f}": (m, f, "m", si, ti)
    for m in ("instruct", "pretrained")
    for f in ("chat", "naturalistic")
    for role, si, ti in (("assistant", 0, 1), ("user", 1, 2))
}
LEG_B_CELLS: dict[str, tuple[str, str, str, int, int]] = {
    f"S_{m}_chat": (m, "chat", "s", 0, 1) for m in ("instruct", "pretrained")
}
ALL_CELLS = {**LEG_A_CELLS, **LEG_B_CELLS}


def _git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _meta() -> dict:
    return {
        "script": "scripts/issue825_selector_audit.py",
        "issue": 825,
        "git_commit": _git_commit(),
        "headline_layer": HEADLINE_LAYER,
        "n_folds": N_FOLDS,
        "fold_seed": FOLD_SEED,
        "dof_cap": DOF_CAP,
        "d_model": D_MODEL,
        "selector_definitions": {
            "gcv_unguarded": "GCV_DOF_CAP=None, lambda_selection=gcv (committed #825 default)",
            "gcv_guarded": f"GCV_DOF_CAP={DOF_CAP}, lambda_selection=gcv (registered mitigation)",
            "inner_group_cv": (
                "lambda_selection=inner-group-cv (summed inner-fold held-out RSS; "
                "never evaluates the GCV dof formula -> independent of GCV_DOF_CAP)"
            ),
        },
    }


def _layer19_slice(xy: dict) -> dict:
    """Restrict the (N, L, D) arrays to the layer-19 slice, keeping the L axis."""
    pos = bat.WANT_LAYERS.index(HEADLINE_LAYER)
    return {
        "X": np.ascontiguousarray(xy["X"][:, pos : pos + 1, :]),
        "Y": np.ascontiguousarray(xy["Y"][:, pos : pos + 1, :]),
        "conv_ids": xy["conv_ids"],
    }


def _fit_one(X, Y, conv_ids, selector: str) -> dict:
    """One held-out grouped-5-fold fit through the committed core.

    Returns {r2, lambdas, lambda_min_frac} for the single retained layer.
    `frozen_layers=()` keeps the core from persisting per-example cosines /
    predictions at the module-level Qwen frozen set, which does not intersect
    this single-layer slice.
    """
    prev = fit.GCV_DOF_CAP
    prev_legacy = fit.LEGACY_UNGUARDED_GCV
    fit.GCV_DOF_CAP = DOF_CAP if selector == "gcv_guarded" else None
    if selector == "gcv_unguarded":
        # #1887: this arm DELIBERATELY reproduces the committed pre-#1887
        # unguarded pure-GCV behavior — the explicit legacy opt-in the refusal
        # guard requires at n_train < d. Restored in the finally block.
        fit.LEGACY_UNGUARDED_GCV = True
    lam_sel = "inner-group-cv" if selector == "inner_group_cv" else "gcv"
    try:
        sw = fit.heldout_r2_sweep(
            X,
            Y,
            conv_ids,
            n_folds=N_FOLDS,
            seed=FOLD_SEED,
            null_draws=0,
            collect_cosines=False,
            collect_lambdas=True,
            lambda_selection=lam_sel,
            frozen_layers=(),
            # #1887: arm semantics byte-preserved — the audit compares
            # SELECTORS; the reduced-basis companion is a separate read.
            reduced_basis_companion=False,
        )
    finally:
        fit.GCV_DOF_CAP = prev
        fit.LEGACY_UNGUARDED_GCV = prev_legacy

    r2 = float(np.asarray(sw["r2_obs"]).reshape(-1)[0])
    lam = sw.get("gcv_lambda")
    out: dict = {"r2": r2}
    if lam is not None:
        lam_arr = np.asarray(lam, dtype=float).reshape(-1)
        finite = lam_arr[np.isfinite(lam_arr)]
        grid = np.asarray(fit.LAMBDAS, dtype=float)
        out["lambdas"] = [None if not np.isfinite(v) else float(v) for v in lam_arr]
        if finite.size:
            out["lambda_geomean"] = float(np.exp(np.mean(np.log(finite))))
            # fraction of folds that selected the GRID FLOOR (the degenerate pin)
            out["lambda_at_grid_floor_frac"] = float(
                np.mean(np.isclose(finite, grid.min(), rtol=1e-9))
            )
    return out


def run_leg_a(cell_id: str) -> dict:
    model, fmt, track, si, ti = ALL_CELLS[cell_id]
    t0 = time.time()
    xy = _layer19_slice(bat.load_cell_xy(model, fmt, track, si, ti))
    n = len(xy["conv_ids"])
    rec: dict = {
        "cell_id": cell_id,
        "n": n,
        "n_train_approx": int(round(n * (N_FOLDS - 1) / N_FOLDS)),
        "d_model": D_MODEL,
        "degenerate_regime": bool(round(n * (N_FOLDS - 1) / N_FOLDS) < D_MODEL),
        "selectors": {},
    }
    for sel in SELECTORS:
        ts = time.time()
        rec["selectors"][sel] = _fit_one(xy["X"], xy["Y"], xy["conv_ids"], sel)
        rec["selectors"][sel]["wall_s"] = round(time.time() - ts, 1)
        print(
            f"[audit] {cell_id} {sel}: r2={rec['selectors'][sel]['r2']:.4f} "
            f"({rec['selectors'][sel]['wall_s']}s)",
            flush=True,
        )
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def run_leg_b(cell_id: str) -> dict:
    model, fmt, track, si, ti = ALL_CELLS[cell_id]
    t0 = time.time()
    xy = _layer19_slice(bat.load_cell_xy(model, fmt, track, si, ti))
    n_full = len(xy["conv_ids"])
    rec: dict = {"cell_id": cell_id, "n_full": n_full, "d_model": D_MODEL, "levels": {}}

    for n_sub in N_LEVELS:
        if n_sub > n_full:
            continue
        seeds = (N_SEEDS[0],) if n_sub == n_full else N_SEEDS
        n_tr = int(round(n_sub * (N_FOLDS - 1) / N_FOLDS))
        lvl: dict = {
            "n": n_sub,
            "n_train_approx": n_tr,
            "degenerate_regime": bool(n_tr < D_MODEL),
            "seeds": {},
        }
        for seed in seeds:
            idx = (
                np.arange(n_full)
                if n_sub == n_full
                else np.random.default_rng(seed).choice(n_full, n_sub, replace=False)
            )
            Xs = np.ascontiguousarray(xy["X"][idx])
            Ys = np.ascontiguousarray(xy["Y"][idx])
            cs = xy["conv_ids"][idx]
            per_sel = {}
            for sel in SELECTORS:
                per_sel[sel] = _fit_one(Xs, Ys, cs, sel)
            lvl["seeds"][str(seed)] = per_sel
            print(
                f"[audit] {cell_id} n={n_sub} seed={seed}: "
                + " ".join(f"{s}={per_sel[s]['r2']:.4f}" for s in SELECTORS),
                flush=True,
            )
        for sel in SELECTORS:
            vals = [lvl["seeds"][s][sel]["r2"] for s in lvl["seeds"]]
            lvl.setdefault("mean_r2", {})[sel] = float(np.mean(vals))
            lvl.setdefault("std_r2", {})[sel] = float(np.std(vals))
        rec["levels"][str(n_sub)] = lvl

    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def merge() -> dict:
    """Merge per-cell partials into results.json (no refitting)."""
    leg_a, leg_b = {}, {}
    for p in sorted(OUT_DIR.glob("_cell_a_*.json")):
        d = json.loads(p.read_text())
        leg_a[d["cell_id"]] = d
    for p in sorted(OUT_DIR.glob("_cell_b_*.json")):
        d = json.loads(p.read_text())
        leg_b[d["cell_id"]] = d

    # Leg A verdict: does inner-group-cv corroborate the guarded GCV reading?
    corroboration = {}
    for cid, d in leg_a.items():
        s = d["selectors"]
        g, u, i = s["gcv_guarded"]["r2"], s["gcv_unguarded"]["r2"], s["inner_group_cv"]["r2"]
        corroboration[cid] = {
            "unguarded": u,
            "guarded": g,
            "inner_group_cv": i,
            "inner_minus_guarded": i - g,
            "inner_minus_unguarded": i - u,
            "guard_lift": g - u,
            "inner_corroborates_guarded": bool(abs(i - g) < 0.05),
            "inner_closer_to_guarded": bool(abs(i - g) < abs(i - u)),
        }

    # Leg B verdict: smallest n at which the three selectors agree (<0.05 spread).
    sizing = {}
    for cid, d in leg_b.items():
        rows = []
        for n_str, lvl in sorted(d["levels"].items(), key=lambda kv: int(kv[0])):
            vals = [lvl["mean_r2"][s] for s in SELECTORS]
            rows.append(
                {
                    "n": int(n_str),
                    "n_train": lvl["n_train_approx"],
                    "degenerate_regime": lvl["degenerate_regime"],
                    "mean_r2": lvl["mean_r2"],
                    "selector_spread": float(max(vals) - min(vals)),
                }
            )
        converged = [r["n"] for r in rows if r["selector_spread"] < 0.05]
        sizing[cid] = {
            "curve": rows,
            "smallest_n_selector_agreement": min(converged) if converged else None,
            "agreement_threshold": 0.05,
        }

    results = {
        "metadata": _meta(),
        "leg_a_selector_audit": leg_a,
        "leg_a_corroboration": corroboration,
        "leg_b_matched_n_curve": leg_b,
        "leg_b_sizing": sizing,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=("a", "b"))
    ap.add_argument("--cell")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        r = merge()
        print(json.dumps(r["leg_a_corroboration"], indent=2), flush=True)
        return 0

    if not args.leg or not args.cell:
        ap.error("--leg and --cell are required unless --merge")

    out = OUT_DIR / f"_cell_{args.leg}_{args.cell}.json"
    if out.exists():
        print(f"[audit] skip {args.cell} (cached)", flush=True)
        return 0
    rec = run_leg_a(args.cell) if args.leg == "a" else run_leg_b(args.cell)
    rec["metadata"] = _meta()
    out.write_text(json.dumps(rec, indent=2))
    print(f"[audit] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
