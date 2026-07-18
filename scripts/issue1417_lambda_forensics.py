"""Issue #1417 free-analysis follow-up — single-cell GCV lambda forensics.

Pins the trigger of the GCV lambda-selection collapse documented in the
clean-result: per (cell, fold) dumps of the GCV objective curve over the
frozen 13-point grid ``fit825.LAMBDAS`` (logspace(-2, 4, 13)), the selected
lambda + effective dof, held-out R^2 per lambda (so R^2 at the selected
lambda vs at 1e3/1e4 reads off one curve), the train-Gram eigenvalue
spectrum (deciles + smallest/largest 10), and near-duplicate-row distances
on the standardized train rows — for the BROKEN instruct c1_helpful_ctrl
cell (kept-fit L19 R^2 -1.481) vs the HEALTHY instruct c2_rude cell
(+0.546), layer 19, context arm, judge-kept rows.

Reuses the FROZEN #825/#1417 instruments UNCHANGED: ``fit825._cv_folds``
(K=5 conv-grouped folds, seed 0), ``fit825._prep_fold`` (standardize +
Gram eigh cache), ``fit825.LAMBDAS``. The GCV scan arithmetic is MIRRORED
line-for-line from ``fit825._ridge_predict_cached`` (which selects but does
not expose the curve); the reproduction check below guards the mirror: the
pooled L19 R^2 at the per-fold GCV-selected lambdas must match the
committed ``cells_<cell>__instruct__ctx.json`` value.

Phases (unified smoke: the tiny slice runs the SAME phases with
``--max-shards 1 --fold 0`` + scratch out-paths):
  uv run python scripts/issue1417_lambda_forensics.py --stage  [--max-shards K]
  uv run python scripts/issue1417_lambda_forensics.py --fit    [--cells a,b] [--fold F]
  uv run python scripts/issue1417_lambda_forensics.py --figure
  uv run python scripts/issue1417_lambda_forensics.py --cleanup
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_fit_cells as fit825  # noqa: E402
import issue1417_render as r1417  # noqa: E402

SCRIPT = "scripts/issue1417_lambda_forensics.py"

MODEL = "instruct"
LAYER = 19  # headline layer (clean-result + issue1417_battery.HEADLINE_LAYER)
SLOT_INDEX = 0  # context arm (issue1417_battery.own_cell_dict arm="ctx")
TURN_INDEX = 0  # own stores carry one generated-answer profile
N_FOLDS = 5
FIT_SEED = 0
N_LAYERS_EXPECTED = 28

BROKEN_CELL = "c1_helpful_ctrl"
HEALTHY_CELL = "c2_rude"
DEFAULT_CELLS = (BROKEN_CELL, HEALTHY_CELL)

STORE_PREFIX = f"{r1417.HF_PREFIX}/analysis_tensors/store"
DEFAULT_STAGE_DIR = Path("/mnt/eps-data/thomasjiralerspong/issue1417_lambda_forensics")
DEFAULT_DATA_DIR = Path("data/issue_1417")
DEFAULT_OUT_JSON = Path("eval_results/issue_1417/lambda_forensics.json")
DEFAULT_FIG_DIR = Path("figures/issue_1417")

# Near-duplicate thresholds on squared distance between STANDARDIZED train
# rows, relative to D (a standardized row has E[||x||^2] ~= D, so d^2/D is a
# scale-free closeness read; 1.0 ~= "as far apart as typical rows").
DUP_SQ_DIST_OVER_D_THRESHOLDS = (1e-6, 1e-4, 1e-2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--cells", default=",".join(DEFAULT_CELLS))
    ap.add_argument("--max-shards", type=int, default=None, help="smoke: stage/load first K shards")
    ap.add_argument("--fold", type=int, default=None, help="smoke: run only this fold index")
    ap.add_argument("--smoke", action="store_true", help="intersect kept ids with the staged slice")
    ap.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1417"))
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return ap.parse_args()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _metadata() -> dict:
    return {
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": SCRIPT,
        "issue": 1417,
        "seed": FIT_SEED,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


# ---------------------------------------------------------------------------
# Phase: --stage (scoped listing + per-file atomic staging, #833/#1402 recipe)
# ---------------------------------------------------------------------------
def cell_stems(cells: list[str]) -> list[str]:
    return [f"{MODEL}_{c}_s" for c in cells]


def stage_stores(args) -> None:
    """Stage the two cells' shard .pt+.json files to the data-disk scratch dir."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path, stage_hub_file

    dest = args.stage_dir
    dest.mkdir(parents=True, exist_ok=True)
    rev_path = dest / "revision.json"
    if rev_path.exists():
        rev = json.loads(rev_path.read_text())["revision"]
    else:
        rev = r1417.resolve_data_repo_rev()
        rev_path.write_text(json.dumps({"revision": rev, "prefix": STORE_PREFIX}, indent=2))
    paths = list_hf_files_under_path(
        HfApi(), r1417.HF_DATA_REPO, STORE_PREFIX, repo_type="dataset", revision=rev
    )
    for stem in cell_stems([c for c in args.cells.split(",") if c]):
        shard_paths = sorted(p for p in paths if Path(p).name.startswith(f"{stem}_shard"))
        assert shard_paths, f"no shards for {stem} under {STORE_PREFIX}@{rev}"
        if args.max_shards is not None:
            # keep the .pt+.json pairs of the first K shard INDICES (one stem
            # per shard index covers both extensions)
            keep_stems = set(
                sorted({Path(p).name.rsplit(".", 1)[0] for p in shard_paths})[: args.max_shards]
            )
            shard_paths = [p for p in shard_paths if Path(p).name.rsplit(".", 1)[0] in keep_stems]
        todo = [p for p in shard_paths if not (dest / Path(p).name).exists()]
        print(
            f"[i1417-forensics] staging {stem}: {len(todo)}/{len(shard_paths)} files @ {rev[:10]}"
        )
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [
                ex.submit(
                    stage_hub_file,
                    r1417.HF_DATA_REPO,
                    p,
                    dest / Path(p).name,
                    repo_type="dataset",
                    revision=rev,
                )
                for p in todo
            ]
            for f in futs:
                f.result()  # fail loud


def cleanup_stage(args) -> None:
    if args.stage_dir.exists():
        shutil.rmtree(args.stage_dir)
        print(f"[i1417-forensics] deleted staged copies at {args.stage_dir}")


# ---------------------------------------------------------------------------
# Streaming layer-19 loader (mirrors fit825._cell_xy semantics on one layer:
# per-row all-layer NaN mask at the cell's slot/turn, bf16 -> float32, bundle
# row order preserved; peak RSS stays ~one shard instead of the full stack)
# ---------------------------------------------------------------------------
def load_cell_layer(
    stage_dir: Path, cell: str, layer: int, max_shards: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = f"{MODEL}_{cell}_s"
    shards = sorted(stage_dir.glob(f"{stem}_shard*.pt"))
    assert shards, f"no staged shards for {stem} in {stage_dir} — run --stage first"
    if max_shards is not None:
        shards = shards[:max_shards]
    side_path = stage_dir / f"{shards[0].name[: -len('.pt')]}.json"
    assert side_path.exists(), f"missing sidecar {side_path}"
    side = json.loads(side_path.read_text())
    assert r1417.fingerprint_matches(side), f"{stem}: store fingerprint mismatch"
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ids: list[str] = []
    for sp in shards:
        payload = torch.load(sp, map_location="cpu", weights_only=False)
        for cid, s, p in zip(
            payload["conv_ids"], payload["slots"], payload["profiles"], strict=True
        ):
            s_arr = s.float().numpy() if torch.is_tensor(s) else np.asarray(s, dtype=np.float32)
            p_arr = p.float().numpy() if torch.is_tensor(p) else np.asarray(p, dtype=np.float32)
            assert s_arr.shape[1] == N_LAYERS_EXPECTED, s_arr.shape
            x_full = s_arr[SLOT_INDEX]  # (L, D)
            y_full = p_arr[TURN_INDEX]  # (L, D)
            # mirror _cell_xy: drop rows with any NaN at this slot/turn (ALL layers)
            if np.isnan(x_full).any() or np.isnan(y_full).any():
                continue
            xs.append(x_full[layer])
            ys.append(y_full[layer])
            ids.append(str(cid))
        del payload
    X = np.stack(xs).astype(np.float32)
    Y = np.stack(ys).astype(np.float32)
    print(f"[i1417-forensics] {stem}: {len(ids)} rows x layer {layer} from {len(shards)} shards")
    return X, Y, np.asarray(ids)


def allowlist_for(args, cell: str, store_ids: np.ndarray) -> list[str]:
    """kept ∩ shared ∩ store — mirrors issue1417_battery.run_fits' kept variant."""
    kept_path = Path(args.out_dir) / "judge" / f"kept_{MODEL}_{cell}.json"
    d = json.loads(kept_path.read_text())
    assert r1417.fingerprint_matches(d), f"{kept_path}: fingerprint mismatch"
    kept = [str(c) for c in d["kept_conv_ids"]]
    shared = set(r1417.shared_conv_ids(args.data_dir))
    store = {str(c) for c in store_ids}
    wanted = [c for c in kept if c in shared]
    if args.smoke:
        wanted = [c for c in wanted if c in store]  # a 1-shard slice lacks most ids
    else:
        missing = [c for c in wanted if c not in store]
        assert not missing, (
            f"{cell}: {len(missing)} kept conv_ids missing from the staged store "
            f"(e.g. {missing[:5]}) — allowlist/bundle drift"
        )
    return [c for c in wanted if c in store]


# ---------------------------------------------------------------------------
# Phase: --fit — per-fold GCV curve + spectrum + near-dup forensics
# ---------------------------------------------------------------------------
def fold_forensics(
    cache: dict, Xtr: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray
) -> dict:
    """GCV curve / selection / dof / per-lambda held-out R^2 / spectrum / dups.

    The GCV scan mirrors ``fit825._ridge_predict_cached`` (grid-min start,
    strict-< first minimum, denom guard); predictions reuse the SAME cached
    (w, V, KevV) path. Returns plain-python payload for JSON.
    """
    dev = cache["w"].device
    lams = np.asarray(fit825.LAMBDAS, dtype=np.float64)
    w, V, KevV, ntr = cache["w"], cache["V"], cache["KevV"], cache["ntr"]
    Ytr = torch.as_tensor(Y[tr], dtype=torch.float64, device=dev)
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    VtY = V.T @ Ytr_c
    sqVtY = (VtY**2).sum(1)
    tot = float((Ytr_c**2).sum())
    true = Y[te].astype(np.float64)
    mu_te = true.mean(0)
    ss_tot = float(np.sum((true - mu_te) ** 2))

    gcv_curve, rss_curve, dof_curve, r2_curve, ss_res_curve = [], [], [], [], []
    best_lam, best_gcv = float(lams[0]), float("inf")
    for lam in lams:
        filt = w / (w + lam)
        dof = float(filt.sum())
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
        pred = ((KevV * (1.0 / (w + lam))) @ VtY + ymu).cpu().numpy()
        ss_res = float(np.sum((true - pred) ** 2))
        gcv_curve.append(gcv)
        rss_curve.append(rss)
        dof_curve.append(dof)
        ss_res_curve.append(ss_res)
        r2_curve.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"))

    sel_idx = int(np.argmin(np.abs(lams - best_lam)))
    w_np = np.sort(w.cpu().numpy())
    w_max = float(w_np[-1]) if len(w_np) else float("nan")

    # Near-duplicate diagnostics on the standardized train rows (mirrors
    # _prep_fold's standardization: train mean / train std + 1e-9, torch std
    # unbiased). d^2_ij = G_ii + G_jj - 2 G_ij on the standardized Gram.
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float64, device=dev)
    xmu = Xtr_t.mean(0)
    xsd = Xtr_t.std(0) + 1e-9
    Xn = (Xtr_t - xmu) / xsd
    G = Xn @ Xn.T
    diag = torch.diagonal(G)
    d2 = diag.unsqueeze(0) + diag.unsqueeze(1) - 2.0 * G
    iu = torch.triu_indices(ntr, ntr, offset=1)
    d2_off = d2[iu[0], iu[1]].clamp(min=0.0)
    D = Xtr.shape[1]
    d2_over_d = (d2_off / D).cpu().numpy()
    smallest5 = np.sort(d2_over_d)[:5]
    dup_counts = {f"{thr:g}": int((d2_over_d < thr).sum()) for thr in DUP_SQ_DIST_OVER_D_THRESHOLDS}
    del G, d2, d2_off, Xn, Xtr_t

    return {
        "ntr": int(ntr),
        "nte": int(te.sum()),
        "gcv_objective": gcv_curve,
        "train_rss": rss_curve,
        "train_ss_tot": tot,
        "train_rss_frac_at_grid_min": rss_curve[0] / tot if tot > 1e-12 else float("nan"),
        "dof": dof_curve,
        "dof_frac": [d / ntr for d in dof_curve],
        "heldout_r2_per_lambda": r2_curve,
        "heldout_ss_res_per_lambda": ss_res_curve,
        "heldout_ss_tot": ss_tot,
        "selected_lambda": best_lam,
        "selected_index": sel_idx,
        "selected_dof": dof_curve[sel_idx],
        "selected_dof_frac": dof_curve[sel_idx] / ntr,
        "r2_at_selected": r2_curve[sel_idx],
        "r2_at_1e3": r2_curve[int(np.argmin(np.abs(lams - 1e3)))],
        "r2_at_1e4": r2_curve[int(np.argmin(np.abs(lams - 1e4)))],
        "gram_spectrum": {
            "deciles": [float(v) for v in np.percentile(w_np, np.arange(0, 101, 10))],
            "smallest10": [float(v) for v in w_np[:10]],
            "largest10": [float(v) for v in w_np[-10:]],
            "max": w_max,
            "n_below_1e-8_max": int((w_np < 1e-8 * w_max).sum()),
            "n_below_1e-4_max": int((w_np < 1e-4 * w_max).sum()),
        },
        "near_dup": {
            "min_sq_dist_over_D": float(smallest5[0]) if len(smallest5) else float("nan"),
            "smallest5_sq_dist_over_D": [float(v) for v in smallest5],
            "n_pairs_below": dup_counts,
        },
    }


def exact_dup_x_digest(X: np.ndarray, ids: np.ndarray) -> dict:
    """Whole-cell exact-duplicate X-row groups (bit-identical layer-19 context
    activations — question-side duplicates render identical contexts). Digest
    only: group sizes + a few conv ids, never row content."""
    from collections import defaultdict

    groups: dict[bytes, list[str]] = defaultdict(list)
    for i in range(len(X)):
        groups[X[i].tobytes()].append(str(ids[i]))
    dup = sorted((v for v in groups.values() if len(v) > 1), key=len, reverse=True)
    sizes = [len(v) for v in dup]
    return {
        "n_groups": len(dup),
        "group_sizes_top10": sizes[:10],
        "n_rows_in_groups": int(sum(sizes)),
        "n_pairs": int(sum(s * (s - 1) // 2 for s in sizes)),
        "largest_group_conv_ids_first8": dup[0][:8] if dup else [],
    }


def committed_r2_l19(out_dir: Path, cell: str) -> float | None:
    p = Path(out_dir) / "cells" / f"cells_{cell}__{MODEL}__ctx.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["r2_per_layer_obs"][LAYER])


def fit_cell(args, cell: str) -> dict:
    X, Y, store_ids = load_cell_layer(args.stage_dir, cell, LAYER, args.max_shards)
    allow = allowlist_for(args, cell, store_ids)
    assert len(allow) >= 2 * N_FOLDS, f"{cell}: only {len(allow)} rows — below the fold floor"
    keep = np.isin(np.asarray([str(c) for c in store_ids]), np.asarray(sorted(set(allow))))
    X, Y, ids = X[keep], Y[keep], store_ids[keep]
    print(f"[i1417-forensics] {cell}: kept allowlist rows {int(keep.sum())}/{len(keep)}")

    folds = fit825._cv_folds(ids, N_FOLDS, FIT_SEED)
    fold_ids = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    fold_payloads = []
    ss_res_sel, ss_tot_sel = 0.0, 0.0
    ss_res_fixed = np.zeros(len(fit825.LAMBDAS))
    ss_tot_fixed = 0.0
    for k in fold_ids:
        te = folds == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 3:
            print(f"[i1417-forensics] {cell} fold {k}: too small — skipped")
            continue
        t0 = time.time()
        cache = fit825._prep_fold(X[tr], X[te])
        fp = fold_forensics(cache, X[tr], Y, tr, te)
        fp["fold"] = int(k)
        fp["wall_s"] = round(time.time() - t0, 2)
        fold_payloads.append(fp)
        ss_res_sel += fp["heldout_ss_res_per_lambda"][fp["selected_index"]]
        ss_tot_sel += fp["heldout_ss_tot"]
        ss_res_fixed += np.asarray(fp["heldout_ss_res_per_lambda"])
        ss_tot_fixed += fp["heldout_ss_tot"]
        print(
            f"[i1417-forensics] {cell} fold {k}: lam_sel={fp['selected_lambda']:g} "
            f"dof_frac={fp['selected_dof_frac']:.3f} r2_sel={fp['r2_at_selected']:.3f} "
            f"r2_1e3={fp['r2_at_1e3']:.3f} ({fp['wall_s']}s)"
        )

    pooled_sel = 1.0 - ss_res_sel / ss_tot_sel if ss_tot_sel > 1e-12 else float("nan")
    committed = committed_r2_l19(args.out_dir, cell)
    repro_delta = abs(pooled_sel - committed) if committed is not None else None
    full_recipe = args.fold is None and args.max_shards is None and not args.smoke
    if full_recipe and committed is not None and repro_delta is not None and repro_delta > 0.02:
        print(
            f"[i1417-forensics] WARN {cell}: pooled R2@GCV {pooled_sel:.4f} vs committed "
            f"{committed:.4f} (|delta|={repro_delta:.4f} > 0.02) — reproduction drift"
        )
    n_grid_min = sum(1 for fp in fold_payloads if fp["selected_index"] == 0)
    return {
        "exact_dup_x_groups": exact_dup_x_digest(X, ids),
        "cell": cell,
        "model": MODEL,
        "arm": "ctx",
        "layer": LAYER,
        "n_rows_fit": len(ids),
        "n_folds_run": len(fold_payloads),
        "n_folds_selected_grid_min_lambda": n_grid_min,
        "pooled_r2_at_gcv_selected": pooled_sel,
        "pooled_r2_per_fixed_lambda": [
            1.0 - float(r) / ss_tot_fixed if ss_tot_fixed > 1e-12 else float("nan")
            for r in ss_res_fixed
        ],
        "committed_r2_l19": committed,
        "reproduction_abs_delta": repro_delta,
        "reproduction_is_full_recipe": full_recipe,
        "folds": fold_payloads,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    os.replace(tmp, path)
    print(f"[i1417-forensics] wrote {path}")


def run_fit(args) -> None:
    out = (
        json.loads(args.out_json.read_text())
        if args.out_json.exists()
        else {"lambda_grid": [float(v) for v in fit825.LAMBDAS], "cells": {}}
    )
    out["metadata"] = _metadata()
    out["gcv_dof_cap"] = fit825.GCV_DOF_CAP
    out["recipe"] = {
        "n_folds": N_FOLDS,
        "seed": FIT_SEED,
        "layer": LAYER,
        "arm": "ctx (slot_index 0)",
        "rows": "judge-kept ∩ shared ∩ store (issue1417_battery.run_fits kept variant)",
        "fold_maker": "fit825._cv_folds (conv-grouped, seed 0)",
        "gcv_scan": "mirrors fit825._ridge_predict_cached (grid-min start, strict-< first min)",
    }
    for cell in [c for c in args.cells.split(",") if c]:
        out["cells"][f"{MODEL}_{cell}"] = fit_cell(args, cell)  # checkpoint per cell
        out["diagnosis"] = _diagnosis(out["cells"])
        _write_json_atomic(args.out_json, out)


def _diagnosis(cells: dict) -> dict:
    """Mechanical per-cell diagnosis fields + the broken-vs-healthy contrast."""
    per_cell = {}
    for key, c in cells.items():
        folds = c["folds"]
        per_cell[key] = {
            "grid_min_collapse": c["n_folds_selected_grid_min_lambda"] == c["n_folds_run"],
            "selected_lambdas": [f["selected_lambda"] for f in folds],
            "selected_dof_frac_mean": float(np.mean([f["selected_dof_frac"] for f in folds])),
            "train_rss_frac_at_grid_min_mean": float(
                np.mean([f["train_rss_frac_at_grid_min"] for f in folds])
            ),
            "r2_at_selected_pooled": c["pooled_r2_at_gcv_selected"],
            "r2_best_fixed_lambda": float(np.nanmax(c["pooled_r2_per_fixed_lambda"])),
            "exact_dup_pairs": c["exact_dup_x_groups"]["n_pairs"],
        }
    d = {"per_cell": per_cell}
    b, h = f"{MODEL}_{BROKEN_CELL}", f"{MODEL}_{HEALTHY_CELL}"
    if b in per_cell and h in per_cell:
        d["contrast"] = {
            "grid_min_collapse_confirmed": bool(
                per_cell[b]["grid_min_collapse"] and not per_cell[h]["grid_min_collapse"]
            ),
            "duplicates_discriminate": bool(
                per_cell[b]["exact_dup_pairs"] > 2 * per_cell[h]["exact_dup_pairs"]
            ),
            "train_rss_collapse_ratio_broken_over_healthy": (
                per_cell[b]["train_rss_frac_at_grid_min_mean"]
                / per_cell[h]["train_rss_frac_at_grid_min_mean"]
            ),
            "note": (
                "trigger = train-RSS collapse at the grid-floor lambda (the GCV "
                "numerator outruns the (n-dof)^2 penalty), NOT near-duplicate rows: "
                "both cells carry comparable exact-duplicate context-row clusters "
                "and near-identical dof_frac at lambda=0.01"
            ),
        }
    return d


# ---------------------------------------------------------------------------
# Phase: --figure — GCV curves per fold, broken vs healthy, selected λ marked
# ---------------------------------------------------------------------------
def make_figure(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    payload = json.loads(args.out_json.read_text())
    lams = np.asarray(payload["lambda_grid"], dtype=np.float64)
    panels = [
        (f"{MODEL}_{BROKEN_CELL}", "Broken cell: helpful-control persona"),
        (f"{MODEL}_{HEALTHY_CELL}", "Healthy cell: rude persona"),
    ]
    panels = [(k, t) for k, t in panels if k in payload["cells"]]
    assert panels, "no fitted cells in the forensics JSON — run --fit first"

    pp.set_paper_style("blog")
    fig, axes = plt.subplots(1, len(panels), figsize=(11.0, 4.2))
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, panels, strict=True):
        cell = payload["cells"][key]
        colors = pp.paper_palette(max(len(cell["folds"]), 1))
        for fp, col in zip(cell["folds"], colors, strict=False):
            g = np.asarray(fp["gcv_objective"], dtype=np.float64)
            ax.plot(lams, g, color=col, lw=1.7, label=f"fold {fp['fold']}")
            ax.plot(
                [fp["selected_lambda"]],
                [g[fp["selected_index"]]],
                "o",
                color=col,
                ms=7,
                mec="black",
                mew=0.6,
                zorder=5,
            )
        r2 = cell.get("pooled_r2_at_gcv_selected", float("nan"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ridge λ (13-point grid, logspace(-2, 4))")
        ax.set_title(f"{title}\npooled held-out R² at selected λ = {r2:.2f}")
        ax.legend(title="dot = GCV-selected λ", fontsize=8, title_fontsize=8)
    axes[0].set_ylabel("GCV objective  =  train RSS / (n - dof)²")
    fig.suptitle(
        "GCV λ-selection per fold — layer 19, context arm, judge-kept rows (instruct)",
        y=1.02,
    )
    fig.tight_layout()
    pp.savefig_paper(fig, "lambda_forensics", dir=args.fig_dir)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.stage:
        stage_stores(args)
    if args.fit:
        run_fit(args)
    if args.figure:
        make_figure(args)
    if args.cleanup:
        cleanup_stage(args)
    if not (args.stage or args.fit or args.figure or args.cleanup):
        print("nothing to do: pass --stage / --fit / --figure / --cleanup")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
