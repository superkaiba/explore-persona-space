#!/usr/bin/env python
"""#825 onpolicy-turn-depth-map VM fit: gates + four-arm fits + nulls + paired Δ.

Off-pod (CPU) per plan §4.5. Per model x layer {14,18,19} x turn cell, fits
FOUR arms on the kept-row intersection with the verbatim-reused PRESS/GCV
grouped-6-fold machinery (imports from issue825_turn_depth_map /
issue1092_fit_grid — never re-implemented):

  ctx_own    — BANKED context_k -> answer_own_t1   (headline)
  ctx_logged — BANKED context_k -> BANKED answer_k_t1 (identical rows/X/folds)
  pfx_own    — NEW-capture prefix_k -> answer_own_t1  (turns >= 3; t1 N/A)
  pfx_logged — NEW-capture prefix_k -> BANKED answer_k_t1

X-source pin (plan W3): BOTH context arms use the BANKED store context_k as X;
the new capture's context_k feeds ONLY the G2 parity gate. Both prefix arms
share the NEW capture's prefix_k. Δ(own - logged) is Y-only by construction.

Gates (all must PASS before any decision read; fail-loud):
  G1 banked-reproduction — refit ctx_logged L19/t1/instruct on the FULL
      pre-drop row set == banked turn_depth_map value within VALIDATION_TOL.
  G2 activation parity — new-capture context_k vs banked context_k,
      row-matched cosines; 50-row pilot with the pre-registered clean-regime
      floor (pilot median < 0.99 = HALT-and-diagnose, never a calibration
      datum); thresholds frozen WITHIN the clean regime, recorded.
  Drop-rate kill line — per-model degenerate-drop rate > 20% halts before
      fitting; >= 15% auto-fires the survivor-set caveat.

Nulls: 200-draw dual-space shuffled-answer permutation null (verbatim
``_dual_perm_null``) at L19 per (arm x provenance) x turn cell with
n >= NULL_MIN_N. Paired stats: fold-level Δ(own - logged) per turn + pooled,
each with a 1,000-resample cluster bootstrap over conversations (vectorized
GEMM over draws — frozen held-out predictions, no refits, no re-generations).

``--emit-banked-ref`` (smoke tooling): computes the ctx_logged real curve from
a fabricated tiny bank and writes the banked-reference JSON the production
path reads from the committed ``turn_depth_map/results.json`` — so the smoke
exercises the same G1/G3 consumers against a real reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE the numpy import — load_dotenv setdefaults
# OMP/MKL/OPENBLAS/NUMEXPR; BLAS pools freeze at import time.
load_dotenv()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

import numpy as np  # noqa: E402
from issue825_turn_depth_map import (  # noqa: E402
    HEADLINE_LAYER,
    HF_DATA_REPO,
    HF_SUMMARIES_BASE,
    LAYERS,
    N_DRAWS,
    NULL_MIN_N,
    NULL_SEED,
    VALIDATION_TOL,
    _build_pairing,
    _dual_perm_null,
    _folds_for_turn,
    _git_commit,
    _real_fit_and_folds,
)
from issue1092_fit_grid import _fit_cv, _load_summary, _read_index_files  # noqa: E402

# Pinned data-repo revision (plan §10 W4): resolves the consumed banked paths
# byte-identically to what the LOGGED turn-depth read downloaded.
DATA_REPO_REV = "9dd650deef3ca21daa9cc2e940e9563edc000ba3"
HF_PREFIX = "issue825_userbase_map"
HF_CAPTURE_PREFIX = f"{HF_PREFIX}/analysis_tensors/onpolicy_turn_depth"

OUT_JSON = REPO_ROOT / "eval_results/issue_825/onpolicy_turn_depth/results.json"
FIG_DIR = REPO_ROOT / "figures/issue_825"
FIG_STEM = "onpolicy_turn_depth"
ANCHOR_JSON = REPO_ROOT / "eval_results/issue_825/matched_n_curve/results.json"
BANKED_JSON = REPO_ROOT / "eval_results/issue_825/turn_depth_map/results.json"

MODELS = ("instruct", "pretrained")
SRC_KIND = "context_k"
DST_KIND = "answer_k_t1"
OWN_KINDS = ("prefix_k", "context_k", "answer_own_t1")
MIN_FIT_N = 3
N_BOOT = 1000
BOOT_SEED = 8250
DROP_KILL_RATE = 0.20
SURVIVOR_CAVEAT_RATE = 0.15
H1_R2_THRESHOLD = 0.40

# G2 pre-registered defaults (plan §6/§11: ungrounded — frozen from the 50-row
# pilot WITHIN the clean regime only; the floor below is the halt line).
G2_PILOT_N = 50
G2_HALT_FLOOR = 0.99
G2_MEDIAN_MIN_DEFAULT = 0.999
G2_ROW_MIN = 0.99
G2_ROW_FRAC = 0.99


def _fetch_one(path_in_repo: str, dest_root: Path, max_attempts: int = 4) -> Path:
    """hf_hub_download one file at the pinned revision, bounded transient retry."""
    from huggingface_hub import hf_hub_download

    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            got = hf_hub_download(
                HF_DATA_REPO,
                path_in_repo,
                repo_type="dataset",
                revision=DATA_REPO_REV,
                local_dir=str(dest_root),
            )
            return Path(got)
        except Exception as e:
            last = e
            print(f"[fetch] retry {attempt + 1}/{max_attempts} {path_in_repo}: {type(e).__name__}")
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"[fetch] FAILED {path_in_repo} after {max_attempts} attempts") from last


def _download_banked(local_root: Path) -> Path:
    """Scoped download of the banked shards at DATA_REPO_REV.

    Mirrors issue825_turn_depth_map._download_summaries (same allow_patterns),
    with the plan-pinned revision instead of "main".
    """
    from huggingface_hub import snapshot_download

    pats: list[str] = []
    for mt in ("dynamics_instruct", "dynamics_pretrained"):
        for kind in (SRC_KIND, DST_KIND):
            for layer in LAYERS:
                pats.append(f"{HF_SUMMARIES_BASE}/{mt}/{kind}_L{layer:02d}_shard*.npy")
        pats.append(f"{HF_SUMMARIES_BASE}/{mt}/row_index_{SRC_KIND}_shard*.jsonl")
        pats.append(f"{HF_SUMMARIES_BASE}/{mt}/row_index_{DST_KIND}_shard*.jsonl")
    snapshot_download(
        HF_DATA_REPO,
        repo_type="dataset",
        revision=DATA_REPO_REV,
        allow_patterns=pats,
        local_dir=str(local_root),
    )
    return local_root / HF_SUMMARIES_BASE


def _download_capture(local_root: Path) -> Path:
    """Scoped list_repo_tree + per-file download of the new capture outputs."""
    from huggingface_hub import HfApi

    api = HfApi()
    tree = list(
        api.list_repo_tree(
            HF_DATA_REPO, path_in_repo=HF_CAPTURE_PREFIX, repo_type="dataset", recursive=True
        )
    )
    paths = [e.path for e in tree if e.path.endswith((".npy", ".jsonl", ".json"))]
    if not paths:
        raise RuntimeError(f"no capture files under {HF_CAPTURE_PREFIX}")
    for p in sorted(paths):
        _fetch_one(p, local_root)
    return local_root / HF_CAPTURE_PREFIX


def _pooled_r2(ss_res: float, ss_tot: float) -> float:
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def _cell_ss(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
    return ss_res, ss_tot


def _cluster_bootstrap(cells: list[dict], conv_universe: list[str], n_boot: int, seed: int) -> dict:
    """Cluster bootstrap over conversations, vectorized over draws (GEMM).

    Each cell dict carries: turn, convs (list, one row per conv), y_own,
    pred_own, y_log, pred_log (fp32 (n, P)). Per draw, conversations are
    resampled with replacement; per-cell R² is recomputed from per-row
    sufficient statistics (residual² per row, per-row y for the resampled
    cell mean) — frozen held-out predictions, no refits.
    """
    rng = np.random.default_rng(seed)
    n_conv = len(conv_universe)
    conv_pos = {c: i for i, c in enumerate(conv_universe)}
    # multiplicity matrix (n_boot, n_conv) — one multinomial draw per bootstrap
    counts = rng.multinomial(n_conv, np.full(n_conv, 1.0 / n_conv), size=n_boot).astype(np.float64)

    per_turn: dict[str, dict] = {}
    pooled_res = {"own": np.zeros(n_boot), "log": np.zeros(n_boot)}
    pooled_tot = {"own": np.zeros(n_boot), "log": np.zeros(n_boot)}
    pooled_point = {"own": [0.0, 0.0], "log": [0.0, 0.0]}
    for cell in cells:
        cols = np.asarray([conv_pos[c] for c in cell["convs"]], dtype=np.int64)
        m = counts[:, cols]  # (n_boot, n_t)
        n_b = m.sum(axis=1)  # resampled row count per draw
        stats = {}
        for prov, y, pred in (
            ("own", cell["y_own"], cell["pred_own"]),
            ("log", cell["y_log"], cell["pred_log"]),
        ):
            r_row = ((y - pred) ** 2).sum(axis=1)  # (n_t,)
            q_row = (y**2).sum(axis=1)  # (n_t,)
            s_b = m @ y  # (n_boot, P) GEMM
            q_b = m @ q_row
            r_b = m @ r_row
            with np.errstate(divide="ignore", invalid="ignore"):
                ss_tot_b = q_b - (s_b**2).sum(axis=1) / np.where(n_b > 0, n_b, np.nan)
                r2_b = 1.0 - r_b / np.where(ss_tot_b > 0, ss_tot_b, np.nan)
            stats[prov] = r2_b
            pooled_res[prov] += r_b
            pooled_tot[prov] += np.where(np.isfinite(ss_tot_b), ss_tot_b, 0.0)
            ss_res_pt, ss_tot_pt = _cell_ss(y, pred)
            pooled_point[prov][0] += ss_res_pt
            pooled_point[prov][1] += ss_tot_pt
        delta_b = stats["own"] - stats["log"]
        per_turn[str(cell["turn"])] = {
            "delta_ci": [
                float(np.nanpercentile(delta_b, 2.5)),
                float(np.nanpercentile(delta_b, 97.5)),
            ],
            "n_finite_draws": int(np.isfinite(delta_b).sum()),
        }
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled_delta_b = (1.0 - pooled_res["own"] / pooled_tot["own"]) - (
            1.0 - pooled_res["log"] / pooled_tot["log"]
        )
    pooled_r2_own = _pooled_r2(pooled_point["own"][0], pooled_point["own"][1])
    pooled_r2_log = _pooled_r2(pooled_point["log"][0], pooled_point["log"][1])
    return {
        "n_boot": n_boot,
        "seed": seed,
        "pooled_r2_own": pooled_r2_own,
        "pooled_r2_logged": pooled_r2_log,
        "pooled_delta": pooled_r2_own - pooled_r2_log,
        "pooled_delta_ci": [
            float(np.nanpercentile(pooled_delta_b, 2.5)),
            float(np.nanpercentile(pooled_delta_b, 97.5)),
        ],
        "per_turn": per_turn,
    }


def _g2_gate(new_ctx: np.ndarray, banked_ctx: np.ndarray, pilot_n: int) -> dict:
    """Row-matched cosine parity, 50-row pilot + clean-regime floor + freeze."""
    a = new_ctx.astype(np.float64)
    b = banked_ctx.astype(np.float64)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = num / np.where(den > 0, den, np.nan)
    pilot = cos[: min(pilot_n, cos.size)]
    pilot_median = float(np.nanmedian(pilot))
    if pilot_median < G2_HALT_FLOOR:
        raise SystemExit(
            f"[G2] HALT-and-diagnose: pilot median cosine {pilot_median:.6f} < "
            f"{G2_HALT_FLOOR} — wiring-bug regime (wrong render/offsets), NEVER a "
            f"calibration datum (plan §6 clean-regime floor)."
        )
    # Freeze WITHIN the clean regime: defaults when the pilot sits at >= the
    # default; else slightly below the pilot median (recorded, deviations-allowed).
    if pilot_median >= G2_MEDIAN_MIN_DEFAULT:
        median_thr = G2_MEDIAN_MIN_DEFAULT
    else:
        median_thr = max(G2_HALT_FLOOR, round(pilot_median - 0.002, 4))
    median_all = float(np.nanmedian(cos))
    frac_ok = float(np.mean(cos >= G2_ROW_MIN))
    ok = median_all >= median_thr and frac_ok >= G2_ROW_FRAC
    return {
        "pass": bool(ok),
        "n_rows": int(cos.size),
        "pilot_n": int(pilot.size),
        "pilot_median": pilot_median,
        "median": median_all,
        "min": float(np.nanmin(cos)),
        "p1": float(np.nanpercentile(cos, 1)),
        "p5": float(np.nanpercentile(cos, 5)),
        "frac_rows_ge_0p99": frac_ok,
        "frozen_median_threshold": median_thr,
        "row_threshold": G2_ROW_MIN,
        "row_frac_required": G2_ROW_FRAC,
        "halt_floor": G2_HALT_FLOOR,
    }


def _load_own(capture_root: Path, mt: str) -> tuple[list[dict], dict, dict]:
    root = capture_root / mt
    rows = _read_index_files(root, "row_index_own")
    with open(root / "drop_report.json") as f:
        drop_report = json.load(f)
    arrays: dict[str, dict[int, np.ndarray]] = {}
    for kind in OWN_KINDS:
        arrays[kind] = {}
        for layer in LAYERS:
            arr, _paths = _load_summary(capture_root, mt, kind, layer)
            if arr.shape[0] != len(rows):
                raise ValueError(f"{mt}/{kind}/L{layer}: {arr.shape[0]} rows != index {len(rows)}")
            arrays[kind][layer] = arr
    return rows, arrays, drop_report


def _emit_banked_ref(args: argparse.Namespace) -> None:
    """Smoke tooling: compute the ctx_logged real curve from the fabricated bank."""
    summaries_dir = Path(args.banked_local_root)
    results: dict[str, dict] = {}
    n_per_turn: dict[str, dict[str, int]] = {}
    for mt in MODELS:
        paired = _build_pairing(summaries_dir, mt)
        ci = np.asarray([p[0] for p in paired], dtype=np.int64)
        aj = np.asarray([p[1] for p in paired], dtype=np.int64)
        rows = [{"conv_id": p[2], "turn_index": p[3]} for p in paired]
        turns = sorted({p[3] for p in paired})
        n_per_turn[mt] = {str(t): sum(1 for p in paired if p[3] == t) for t in turns}
        results[mt] = {}
        for layer in LAYERS:
            arr_c, _ = _load_summary(summaries_dir, f"dynamics_{mt}", SRC_KIND, layer)
            arr_a, _ = _load_summary(summaries_dir, f"dynamics_{mt}", DST_KIND, layer)
            x_all, y_all = arr_c[ci], arr_a[aj]
            layer_res: dict[str, dict] = {}
            for t in turns:
                sel = np.asarray([i for i, p in enumerate(paired) if p[3] == t], dtype=np.int64)
                if sel.size < MIN_FIT_N:
                    continue
                rf = _real_fit_and_folds(x_all[sel], y_all[sel], [rows[i] for i in sel])
                if rf is None:
                    continue
                fit, _folds = rf
                layer_res[str(t)] = {
                    "turn": t,
                    "n": int(sel.size),
                    "real_r2": float(fit["r2"]),
                    "lambda_indices": fit["lambda_indices"],
                }
            results[mt][str(layer)] = layer_res
    payload = {
        "issue": 825,
        "description": "SMOKE banked reference (fabricated tiny bank; --emit-banked-ref)",
        "n_per_turn": n_per_turn,
        "results": results,
        "git_commit": _git_commit(),
    }
    out = Path(args.banked_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[emit-banked-ref] wrote {out}")


def _regen_anchor_at_n(realized_n: int, anchor_json: Path) -> None:
    """Plan §4.5: anchor regenerated at the realized kept-row t1 n if drops occur."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/issue825_matched_n_curve.py"),
        "--extra-n",
        str(realized_n),
        "--out-json",
        str(anchor_json),
    ]
    print(f"[anchor] regenerating at realized n={realized_n}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env={**os.environ}, check=True)


def _fit_model_layer(
    x_by_arm: dict[str, np.ndarray],
    y_by_prov: dict[str, np.ndarray],
    rows: list[dict],
    turns_sel: dict[int, np.ndarray],
    layer: int,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    """All four arms for one (model, layer): per-turn fits (+ nulls at L19)."""
    layer_out: dict[str, dict] = {}
    paired_cells: list[dict] = []
    for t, sel in sorted(turns_sel.items()):
        if sel.size < MIN_FIT_N:
            continue
        cell_rows = [rows[i] for i in sel]
        folds = _folds_for_turn(cell_rows)
        if folds is None:
            continue
        entry: dict[str, dict] = {"n": int(sel.size)}
        preds: dict[tuple[str, str], np.ndarray] = {}
        for arm in ("ctx", "pfx"):
            if arm == "pfx" and t == 1:
                entry["pfx_own"] = {"status": "N/A — structurally degenerate at t1"}
                entry["pfx_logged"] = {"status": "N/A — structurally degenerate at t1"}
                continue
            x_cell = x_by_arm[arm][sel]
            for prov in ("own", "logged"):
                y_cell = y_by_prov[prov][sel]
                fit, pred = _fit_cv(x_cell, y_cell, folds, return_pred=True)
                preds[(arm, prov)] = pred
                node = {
                    "status": "computed",
                    "r2": float(fit["r2"]),
                    "r2_folds": [float(v) for v in fit["r2_folds"]],
                    "lambda_indices": fit["lambda_indices"],
                    "null_mean": None,
                    "null_lo": None,
                    "null_hi": None,
                    "null_n_draws": 0,
                }
                if layer == HEADLINE_LAYER and sel.size >= args.null_min_n:
                    draws = _dual_perm_null(
                        x_cell, y_cell, folds, fit["lambda_indices"], args.n_draws, NULL_SEED
                    )
                    finite = draws[np.isfinite(draws)]
                    if finite.size:
                        node["null_mean"] = float(np.mean(finite))
                        node["null_lo"] = float(np.percentile(finite, 2.5))
                        node["null_hi"] = float(np.percentile(finite, 97.5))
                        node["null_n_draws"] = int(finite.size)
                entry[f"{arm}_{prov}"] = node
        # fold-level paired deltas (context arm)
        if ("ctx", "own") in preds and ("ctx", "logged") in preds:
            own_folds = entry["ctx_own"]["r2_folds"]
            log_folds = entry["ctx_logged"]["r2_folds"]
            entry["ctx_delta_folds"] = [o - lg for o, lg in zip(own_folds, log_folds, strict=True)]
            entry["ctx_delta"] = entry["ctx_own"]["r2"] - entry["ctx_logged"]["r2"]
            if layer == HEADLINE_LAYER:
                paired_cells.append(
                    {
                        "turn": t,
                        "convs": [r["conv_id"] for r in cell_rows],
                        "y_own": y_by_prov["own"][sel].astype(np.float32),
                        "pred_own": preds[("ctx", "own")].astype(np.float32),
                        "y_log": y_by_prov["logged"][sel].astype(np.float32),
                        "pred_log": preds[("ctx", "logged")].astype(np.float32),
                    }
                )
        layer_out[str(t)] = entry
    return layer_out, paired_cells


def main() -> None:  # noqa: C901 — linear pipeline driver
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-dir", default="", help="local capture root (else HF download)")
    ap.add_argument(
        "--banked-local-root",
        default=str(REPO_ROOT / "data/issue_825/onpolicy_td_banked_dl"),
        help="local root for the banked store download (or the smoke bank root)",
    )
    ap.add_argument("--skip-banked-download", action="store_true")
    ap.add_argument("--banked-json", default=str(BANKED_JSON))
    ap.add_argument("--anchor-json", default=str(ANCHOR_JSON))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--fig-dir", default=str(FIG_DIR))
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--null-min-n", type=int, default=NULL_MIN_N)
    ap.add_argument("--pilot-n", type=int, default=G2_PILOT_N)
    ap.add_argument("--smoke", action="store_true", help="tiny local inputs; no HF; no anchor")
    ap.add_argument("--emit-banked-ref", action="store_true")
    args = ap.parse_args()

    if args.emit_banked_ref:
        _emit_banked_ref(args)
        return

    t_start = time.time()
    banked_root = Path(args.banked_local_root)
    if args.smoke or args.skip_banked_download:
        summaries_dir = banked_root
    else:
        summaries_dir = _download_banked(banked_root.parent)
    if args.capture_dir:
        capture_root = Path(args.capture_dir)
    else:
        if args.smoke:
            raise SystemExit("--smoke requires --capture-dir")
        capture_root = _download_capture(REPO_ROOT / "data/issue_825/onpolicy_td_capture_dl")

    with open(args.banked_json) as f:
        banked = json.load(f)

    gates: dict[str, dict] = {}
    results: dict[str, dict] = {}
    paired: dict[str, dict] = {}
    drop_blocks: dict[str, dict] = {}
    timing: dict[str, float] = {}

    for mt in MODELS:
        pairing = _build_pairing(summaries_dir, mt)
        own_rows, own_arrays, drop_report = _load_own(capture_root, mt)
        own_map = {(str(r["conv_id"]), int(r["turn_index"])): i for i, r in enumerate(own_rows)}
        kept_pair_idx = [
            i for i, (_ci, _aj, conv, turn) in enumerate(pairing) if (conv, turn) in own_map
        ]
        own_sel = np.asarray(
            [own_map[(pairing[i][2], pairing[i][3])] for i in kept_pair_idx], dtype=np.int64
        )
        ci_idx = np.asarray([pairing[i][0] for i in kept_pair_idx], dtype=np.int64)
        aj_idx = np.asarray([pairing[i][1] for i in kept_pair_idx], dtype=np.int64)
        rows = [{"conv_id": pairing[i][2], "turn_index": pairing[i][3]} for i in kept_pair_idx]

        # --- drop-rate kill line + survivor caveat (plan §6) ---
        drop_rate = float(drop_report["drop_rate"])
        if drop_rate > DROP_KILL_RATE:
            raise SystemExit(
                f"[kill] {mt} degenerate-drop rate {100 * drop_rate:.1f}% > "
                f"{100 * DROP_KILL_RATE:.0f}% — generation-recipe problem; refusing "
                f"to fit the survivor subset (plan §6 kill criterion)."
            )
        drop_blocks[mt] = {
            "drop_report": drop_report,
            "survivor_caveat": drop_rate >= SURVIVOR_CAVEAT_RATE,
            "n_banked_pairs": len(pairing),
            "n_kept_pairs": len(kept_pair_idx),
        }

        results[mt] = {}
        paired_cells_l19: list[dict] = []
        turns_all = sorted({pairing[i][3] for i in kept_pair_idx})
        turns_sel = {
            t: np.asarray(
                [j for j, i in enumerate(kept_pair_idx) if pairing[i][3] == t], dtype=np.int64
            )
            for t in turns_all
        }
        for layer in LAYERS:
            arr_c, _ = _load_summary(summaries_dir, f"dynamics_{mt}", SRC_KIND, layer)
            arr_a, _ = _load_summary(summaries_dir, f"dynamics_{mt}", DST_KIND, layer)
            x_ctx = arr_c[ci_idx]
            y_log = arr_a[aj_idx]
            y_own = own_arrays["answer_own_t1"][layer][own_sel].astype(np.float64)
            x_pfx = own_arrays["prefix_k"][layer][own_sel].astype(np.float64)

            # --- G1 (instruct, headline layer): FULL pre-drop refit vs banked ---
            if mt == "instruct" and layer == HEADLINE_LAYER:
                t1_full = np.asarray(
                    [i for i, p in enumerate(pairing) if p[3] == 1], dtype=np.int64
                )
                rows_full = [
                    {"conv_id": pairing[i][2], "turn_index": pairing[i][3]} for i in t1_full
                ]
                t_g1 = time.time()
                rf = _real_fit_and_folds(
                    arr_c[[pairing[i][0] for i in t1_full]],
                    arr_a[[pairing[i][1] for i in t1_full]],
                    rows_full,
                )
                timing["g1_single_fit_s"] = time.time() - t_g1
                assert rf is not None, "[G1] full t1 refit degenerate"
                refit_r2 = float(rf[0]["r2"])
                banked_node = banked["results"]["instruct"][str(HEADLINE_LAYER)]["1"]
                banked_r2 = float(banked_node["real_r2"])
                g1_diff = abs(refit_r2 - banked_r2)
                gates["G1"] = {
                    "cell": f"instruct/L{HEADLINE_LAYER}/turn1 (full pre-drop rows)",
                    "n": int(t1_full.size),
                    "refit_r2": refit_r2,
                    "banked_r2": banked_r2,
                    "abs_diff": g1_diff,
                    "tol": VALIDATION_TOL,
                    "pass": bool(g1_diff <= VALIDATION_TOL),
                    "single_fit_seconds": timing["g1_single_fit_s"],
                }
                print(
                    f"[G1] refit={refit_r2:.6f} banked={banked_r2:.6f} "
                    f"|d|={g1_diff:.2e} tol={VALIDATION_TOL} -> "
                    f"{'PASS' if gates['G1']['pass'] else 'FAIL'} "
                    f"({timing['g1_single_fit_s']:.1f}s)"
                )
                if not gates["G1"]["pass"]:
                    raise SystemExit("[G1] FAIL — banked reproduction diverged; wiring bug, HALT")

            # --- G2 (headline layer binds; other layers recorded) ---
            new_ctx = own_arrays["context_k"][layer][own_sel].astype(np.float64)
            g2 = _g2_gate(new_ctx, x_ctx, args.pilot_n)
            gates.setdefault("G2", {})[f"{mt}/L{layer}"] = g2
            print(
                f"[G2] {mt}/L{layer}: median={g2['median']:.6f} "
                f"frac>=0.99={g2['frac_rows_ge_0p99']:.4f} "
                f"(thr median>={g2['frozen_median_threshold']}, "
                f"frac>={g2['row_frac_required']}) -> {'PASS' if g2['pass'] else 'FAIL'}"
            )
            if layer == HEADLINE_LAYER and not g2["pass"]:
                raise SystemExit(f"[G2] FAIL at {mt}/L{layer} — activation parity broke; HALT")

            t_layer = time.time()
            layer_out, cells = _fit_model_layer(
                {"ctx": x_ctx, "pfx": x_pfx},
                {"own": y_own, "logged": y_log},
                rows,
                turns_sel,
                layer,
                args,
            )
            timing[f"fits_{mt}_L{layer}_s"] = time.time() - t_layer
            results[mt][str(layer)] = layer_out
            if layer == HEADLINE_LAYER:
                paired_cells_l19 = cells
            print(
                f"[fit] {mt} L{layer}: {len(layer_out)} turn cells in "
                f"{timing[f'fits_{mt}_L{layer}_s']:.1f}s"
            )

        conv_universe = sorted({r["conv_id"] for r in rows})
        t_boot = time.time()
        paired[mt] = _cluster_bootstrap(paired_cells_l19, conv_universe, args.n_boot, BOOT_SEED)
        timing[f"bootstrap_{mt}_s"] = time.time() - t_boot
        t1_node = results[mt].get(str(HEADLINE_LAYER), {}).get("1", {})
        paired[mt]["t1"] = {
            "r2_own": (t1_node.get("ctx_own") or {}).get("r2"),
            "r2_logged": (t1_node.get("ctx_logged") or {}).get("r2"),
            "delta": t1_node.get("ctx_delta"),
            "delta_ci": (paired[mt]["per_turn"].get("1") or {}).get("delta_ci"),
            "n": t1_node.get("n"),
        }

    # --- anchor (production only; plan §4.5/R4) ---
    anchor_block: dict = {"status": "n/a — smoke"}
    if not args.smoke:
        anchor_path = Path(args.anchor_json)
        realized_t1_n = int(paired["instruct"]["t1"]["n"] or 0)
        if anchor_path.exists():
            with open(anchor_path) as f:
                anchor = json.load(f)
            have_ns = {int(row["n"]) for row in anchor.get("curve", [])}
            if realized_t1_n not in have_ns:
                _regen_anchor_at_n(realized_t1_n, anchor_path)
                with open(anchor_path) as f:
                    anchor = json.load(f)
            anchor_block = {
                "status": anchor.get("anchor_status", "unknown"),
                "realized_t1_n": realized_t1_n,
                "curve": anchor.get("curve"),
                "gate_full_n": anchor.get("gate_full_n"),
                "anchor_n497": anchor.get("anchor_n497"),
                "source_json": str(anchor_path),
            }
        else:
            anchor_block = {
                "status": "demoted — anchor JSON missing (marker-recorded ~0.48, "
                "uncommitted provenance)",
                "realized_t1_n": realized_t1_n,
            }

    # --- decision read (pre-registered §6; H-neg is its own outcome class) ---
    inst = paired["instruct"]
    decision: dict = {
        "headline": "instruct/L19: pooled ctx Δ(own - logged) + Δ(t1)",
        "r2_own_t1": inst["t1"]["r2_own"],
        "h1_threshold": H1_R2_THRESHOLD,
        "pooled_delta": inst["pooled_delta"],
        "pooled_delta_ci": inst["pooled_delta_ci"],
        "t1_delta": inst["t1"]["delta"],
        "t1_delta_ci": inst["t1"]["delta_ci"],
    }
    if not args.smoke:
        pooled_lo, pooled_hi = inst["pooled_delta_ci"]
        t1_ci = inst["t1"]["delta_ci"] or [float("nan"), float("nan")]
        pooled_excl = pooled_lo > 0 or pooled_hi < 0
        t1_excl = t1_ci[0] > 0 or t1_ci[1] < 0
        anchor_demoted = str(anchor_block.get("status", "")).startswith("demoted")
        if (pooled_excl and pooled_lo > 0) and (t1_excl and t1_ci[0] > 0):
            r2_own_t1 = decision["r2_own_t1"] or float("nan")
            if anchor_demoted:
                decision["outcome"] = (
                    "H1/H2 boundary reported descriptively (anchor demoted, plan §7)"
                )
            elif r2_own_t1 >= H1_R2_THRESHOLD:
                decision["outcome"] = "H1 — provenance explains most of the residual"
            else:
                decision["outcome"] = "H2 — provenance contributes; remaining gap is corpus"
        elif (pooled_excl and pooled_hi < 0) or (t1_excl and t1_ci[1] < 0):
            decision["outcome"] = (
                "H-neg — provenance lowers the map under this sampling recipe "
                "(pre-registered outcome class, never forced into H0)"
            )
        elif not pooled_excl and not t1_excl:
            decision["outcome"] = "H0 — provenance is not the driver (both CIs include 0)"
        else:
            decision["outcome"] = "mixed — pooled vs t1 disagree; report descriptively"
    else:
        decision["outcome"] = "n/a — smoke"

    # --- compute-deviation record (per-call cost re-derived, never copied) ---
    n_fits = sum(
        sum(1 for t, e in results[mt][str(layer)].items() if "ctx_own" in e)
        for mt in MODELS
        for layer in LAYERS
    )
    per_fit = timing.get("g1_single_fit_s", float("nan"))
    projected_fit_wall_h = per_fit * n_fits * 4 / 3600.0 if math.isfinite(per_fit) else None

    payload = {
        "issue": 825,
        "followup_label": "onpolicy-turn-depth-map",
        "description": (
            "Four-arm per-turn held-out R2 (PRESS/GCV ridge, grouped 6-fold): "
            "banked context_k / new prefix_k -> {own on-policy, banked logged} "
            "answers, instruct & pretrained, layers 14/18/19, with G1/G2 gates, "
            "shuffled-answer nulls, and paired conversation-cluster bootstrap."
        ),
        "x_source_pin": (
            "ctx arms: BANKED store context_k (new capture context_k feeds ONLY G2); "
            "pfx arms: NEW capture prefix_k. Delta(own - logged) is Y-only by "
            "construction (plan W3)."
        ),
        "data_repo": HF_DATA_REPO,
        "data_repo_revision": DATA_REPO_REV,
        "capture_prefix": HF_CAPTURE_PREFIX,
        "layers": list(LAYERS),
        "headline_layer": HEADLINE_LAYER,
        "estimator": "PRESS-ridge grouped 6-fold CV (issue1092_fit_grid._fit_cv, verbatim)",
        "null": {
            "recipe": "dual-space shuffled-answer permutation (issue825_turn_depth_map."
            "_dual_perm_null, verbatim)",
            "n_draws": args.n_draws,
            "seed": NULL_SEED,
            "min_n": args.null_min_n,
        },
        "gates": gates,
        "drops": drop_blocks,
        "results": results,
        "paired": paired,
        "anchor": anchor_block,
        "decision": decision,
        "compute_record": {
            "n_arm_fits": n_fits * 4,
            "g1_single_fit_seconds": per_fit,
            "projected_fit_wall_h_from_measured": projected_fit_wall_h,
            "timing_seconds": timing,
            "total_wall_s": time.time() - t_start,
        },
        "smoke": bool(args.smoke),
        "git_commit": _git_commit(),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[write] {out_json}")

    _figures(payload, Path(args.fig_dir))
    print(f"[done] total {time.time() - t_start:.1f}s")


def _curve(results: dict, mt: str, layer: int, key: str, min_n: int) -> tuple:
    node = results[mt][str(layer)]
    xs, r2s, ns = [], [], []
    for t_s, e in sorted(node.items(), key=lambda kv: int(kv[0])):
        cell = e.get(key)
        if not cell or cell.get("status") != "computed" or e["n"] < min_n:
            continue
        xs.append(int(t_s))
        r2s.append(cell["r2"])
        ns.append(e["n"])
    return np.asarray(xs), np.asarray(r2s, dtype=float), np.asarray(ns)


def _null_band(results: dict, mt: str, layer: int, key: str, xs: np.ndarray) -> tuple:
    node = results[mt][str(layer)]
    lo = [(node[str(t)].get(key) or {}).get("null_lo") for t in xs]
    hi = [(node[str(t)].get(key) or {}).get("null_hi") for t in xs]
    lo = np.asarray([np.nan if v is None else v for v in lo], dtype=float)
    hi = np.asarray([np.nan if v is None else v for v in hi], dtype=float)
    return lo, hi


ANCHOR_CAVEAT = (
    "Anchor is cross-recipe (reference only, never the test statistic): different "
    "corpus (curated #825 single-turn bundle); grouped 5-fold vs 6-fold; layer "
    "battery {14,18,19,26} vs {14,18,19} (L26 unmeasurable here); bf16 vs fp16 "
    "capture dtype; 0.48@n=497 is a mean over 5 subsample draws (rngs 1000-1004)."
)


def _figures(payload: dict, fig_dir: Path) -> None:
    """Orchestrate F1-F4; each figure has its own helper (complexity budget)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    pal = paper_palette(4)
    anchor = payload.get("anchor") or {}
    anchor_497 = None
    anchor_full = None
    for row in anchor.get("curve") or []:
        if int(row.get("n", -1)) == 497:
            anchor_497 = row.get("r2_mean")
    if isinstance(anchor.get("gate_full_n"), dict):
        anchor_full = anchor["gate_full_n"].get("value")
    ctx = {
        "plt": plt,
        "payload": payload,
        "fig_dir": fig_dir,
        "results": payload["results"],
        "min_n": payload["null"]["min_n"],
        "hl": payload["headline_layer"],
        "c_own": pal[0],
        "c_log": pal[1],
        "c_14": pal[2],
        "c_18": pal[3],
        "anchor_497": anchor_497,
        "anchor_full": anchor_full,
    }
    _fig_hero(ctx)
    _fig_prefix(ctx)
    _fig_folds(ctx)
    _fig_exploratory(ctx)


def _save_fig(ctx: dict, fig, stem: str, caption: str) -> None:
    """Save png+pdf+meta.json sidecar (house figure contract)."""
    plt = ctx["plt"]
    fig_dir = ctx["fig_dir"]
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    meta = {
        "figure": f"{stem}.png",
        "git_commit": ctx["payload"]["git_commit"],
        "source_results_json": "eval_results/issue_825/onpolicy_turn_depth/results.json",
        "caption": caption,
    }
    with open(fig_dir / f"{stem}.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[write] {fig_dir / (stem + '.png')}")


def _fig_hero(ctx: dict) -> None:
    """F1: L19 ctx_own vs ctx_logged per model, null bands + anchor lines."""
    plt, results, min_n, hl = ctx["plt"], ctx["results"], ctx["min_n"], ctx["hl"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, mt in zip(axes, MODELS, strict=True):
        xo, ro, no = _curve(results, mt, hl, "ctx_own", min_n)
        xl, rl, _nl = _curve(results, mt, hl, "ctx_logged", min_n)
        lo_o, hi_o = _null_band(results, mt, hl, "ctx_own", xo)
        if np.isfinite(lo_o).any():
            ax.fill_between(
                xo, lo_o, hi_o, color="0.7", alpha=0.35, linewidth=0, label="shuffled null (own)"
            )
        ax.plot(xo, ro, "-o", color=ctx["c_own"], ms=4, lw=1.8, label="own answers (on-policy)")
        ax.plot(xl, rl, "--s", color=ctx["c_log"], ms=4, lw=1.8, label="logged answers")
        if ctx["anchor_497"] is not None:
            ax.axhline(
                ctx["anchor_497"], color="0.35", lw=1.0, ls="--", label="single-turn anchor@n=497"
            )
        if ctx["anchor_full"] is not None:
            ax.plot([0.6], [ctx["anchor_full"]], marker="_", ms=14, color="0.2")
            ax.annotate("0.673@5k", (0.7, ctx["anchor_full"]), fontsize=7, color="0.2")
        for x, n in zip(xo, no, strict=True):
            ax.annotate(str(int(n)), (x, -0.02), fontsize=6, color="0.5", ha="center")
        ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel("assistant-turn index")
        ax.set_title(mt)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    axes[0].set_ylabel(rf"held-out $R^2$, context$\to$answer, L{hl}")
    _save_fig(
        ctx,
        fig,
        FIG_STEM,
        "Per-turn held-out R2 of the linear context->answer map at layer "
        f"{hl}, own on-policy answers vs the real logged answers on identical rows/X/"
        f"folds; grey band = matched shuffled-answer null (own arm); per-turn n "
        f"annotated. {ANCHOR_CAVEAT}",
    )


def _fig_prefix(ctx: dict) -> None:
    """F2: prefix-arm companion; t1 marked structurally degenerate."""
    plt, results, min_n, hl = ctx["plt"], ctx["results"], ctx["min_n"], ctx["hl"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, mt in zip(axes, MODELS, strict=True):
        xo, ro, _no = _curve(results, mt, hl, "pfx_own", min_n)
        xl, rl, _nl = _curve(results, mt, hl, "pfx_logged", min_n)
        ax.plot(xo, ro, "-o", color=ctx["c_own"], ms=4, lw=1.8, label="own answers")
        ax.plot(xl, rl, "--s", color=ctx["c_log"], ms=4, lw=1.8, label="logged answers")
        ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
        ax.annotate(
            "t1: N/A — structurally degenerate",
            (0.02, 0.02),
            xycoords="axes fraction",
            fontsize=7,
            color="0.4",
        )
        ax.set_xlabel("assistant-turn index")
        ax.set_title(mt)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    axes[0].set_ylabel(rf"held-out $R^2$, prefix$\to$answer, L{hl}")
    _save_fig(
        ctx,
        fig,
        f"{FIG_STEM}_prefix",
        f"Prefix-arm companion at layer {hl}: before-the-query state -> answer, both "
        "provenances on the shared new-capture prefix_k X. Turn 1 has no prior "
        "turns (constant scaffold token) and is N/A by construction, never a zero.",
    )


def _fig_folds(ctx: dict) -> None:
    """F3 (low-level per-unit): per-fold R2 points at each turn cell."""
    plt, results, min_n, hl = ctx["plt"], ctx["results"], ctx["min_n"], ctx["hl"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, mt in zip(axes, MODELS, strict=True):
        node = results[mt][str(hl)]
        for t_s, e in sorted(node.items(), key=lambda kv: int(kv[0])):
            if e["n"] < min_n:
                continue
            t = int(t_s)
            for key, color, dx in (
                ("ctx_own", ctx["c_own"], -0.18),
                ("ctx_logged", ctx["c_log"], 0.18),
            ):
                cell = e.get(key)
                if not cell or cell.get("status") != "computed":
                    continue
                folds = cell["r2_folds"]
                ax.scatter([t + dx] * len(folds), folds, s=9, color=color, alpha=0.6)
        ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel("assistant-turn index")
        ax.set_title(mt)
    axes[0].set_ylabel(rf"per-fold held-out $R^2$ (L{hl}, context arm)")
    axes[0].scatter([], [], s=9, color=ctx["c_own"], label="own")
    axes[0].scatter([], [], s=9, color=ctx["c_log"], label="logged")
    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.9)
    _save_fig(
        ctx,
        fig,
        f"{FIG_STEM}_folds",
        f"Underlying per-fold held-out R2 points behind F1 (layer {hl}, context arm), "
        "own (left offset) vs logged (right offset) at each turn cell with "
        f"n >= {min_n}.",
    )


def _fig_exploratory(ctx: dict) -> None:
    """F4: L14/L18 own-answer curves + per-turn paired delta with bootstrap CI."""
    plt, results, min_n, hl = ctx["plt"], ctx["results"], ctx["min_n"], ctx["hl"]
    payload = ctx["payload"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    for layer, color, ls in ((14, ctx["c_14"], "-."), (18, ctx["c_18"], "--")):
        for mt, marker in (("instruct", "o"), ("pretrained", "s")):
            xo, ro, _ = _curve(results, mt, layer, "ctx_own", min_n)
            ax.plot(xo, ro, ls + marker, color=color, ms=3, lw=1.2, label=f"{mt} L{layer} own")
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("assistant-turn index")
    ax.set_ylabel(r"held-out $R^2$ (ctx_own)")
    ax.set_title("Exploratory: layers 14/18")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
    ax = axes[1]
    for mt, color, marker in (
        ("instruct", ctx["c_own"], "o"),
        ("pretrained", ctx["c_log"], "s"),
    ):
        node = results[mt][str(hl)]
        per_turn = payload["paired"][mt]["per_turn"]
        xs, ds, los, his = [], [], [], []
        for t_s, e in sorted(node.items(), key=lambda kv: int(kv[0])):
            if e.get("ctx_delta") is None or t_s not in per_turn or e["n"] < min_n:
                continue
            xs.append(int(t_s))
            ds.append(e["ctx_delta"])
            ci = per_turn[t_s]["delta_ci"]
            los.append(e["ctx_delta"] - ci[0])
            his.append(ci[1] - e["ctx_delta"])
        if xs:
            ax.errorbar(
                xs,
                ds,
                yerr=[np.maximum(0.0, los), np.maximum(0.0, his)],
                fmt=marker,
                color=color,
                ms=4,
                lw=1.0,
                capsize=2,
                label=mt,
            )
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("assistant-turn index")
    ax.set_ylabel(r"$\Delta R^2$ (own $-$ logged)")
    ax.set_title("Per-turn paired delta (bootstrap 95% CI; deep turns exploratory)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    _save_fig(
        ctx,
        fig,
        f"{FIG_STEM}_exploratory",
        "Exploratory panels: layers 14/18 own-answer context curves (left); per-turn "
        f"paired delta own-logged at L{hl} with conversation-cluster bootstrap 95% CIs "
        "(right) — deep-turn cells are exploratory per the pre-registration (one-draw "
        "temp-1.0 sampling variance; bootstrap resamples conversations, not "
        "re-generations).",
    )


if __name__ == "__main__":
    main()
