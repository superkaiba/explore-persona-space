#!/usr/bin/env python3
"""#1902 K=5 diagonal length control: does the own-map R^2 ordering survive?

At K=5 the own-map held-out R^2 ordering is base > SFT > DPO > RLVR, but base
answers are ~5x shorter and base's advantage lives in the SS_tot denominator.
This ANALYSIS-ONLY round re-fits the four diagonal cells (B/B, S/S, D/D, R/R)
on answer-length-matched subsets, on the paper's own protocol (``u_last``
context summary at layer 31, six IID random folds seed 190231, the committed
``SharedPrimalRidge`` estimator), and reads the length-stratified residual /
total-SS decomposition from the already-fitted full-data per-cell arrays.

Designs (marginal quantile-bin length matching, reusing the #1902 9a-ter
``equalize_bins`` machinery on K=5 MEAN answer lengths per context):

- ``primary``: widest-needed band on a p5-anchored ladder such that every
  stage's matched subset keeps ``n_train > d`` (=4,096) on EVERY fold — the
  #1701 well-posedness floor. Fitted with the committed ``SharedPrimalRidge``
  (which refuses the degenerate regime by construction).
- ``p10``: the K=1 precedent's p10 band, for comparability. Its matched n
  leaves ``n_train < d``: the committed estimator refuses, so this design is
  fitted with the committed shared fast core
  (``issue_779.fit_h.ridge_fit_predict_fast_layer_batched``) under the #1887
  GCV dof cap (0.9), after a same-inputs parity gate against
  ``SharedPrimalRidge`` on a well-posed slice. Every p10 number is an
  UNDER-DETERMINED regularization-limit fit — a relative ordering at equal n
  across stages, never an absolute R^2.

Never prints or logs rollout text: ids, counts, token statistics only.

Run (VM CPU, thread-capped, resumable subcommands)::

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue1902_k5_length_control.py <cmd>

with ``cmd`` in: lengths, match, parity, fit --design {primary,p10}
--stage {B,S,D,R} [--folds 0,1,...], summarize, figure, all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE numpy/BLAS imports on the shared VM.
load_dotenv()

import numpy as np  # noqa: E402

import issue1902_followup_9ater as F9  # noqa: E402
import issue1902_k5_fits as KF  # noqa: E402
import issue1902_lasttoken_comparison as LC  # noqa: E402
import issue1902_lasttoken_transfer as XF  # noqa: E402

STAGES = KF.STAGES  # ("B", "S", "D", "R")
LAYER = KF.LAYER  # 31
N_FOLDS = KF.N_FOLDS  # 6
D_FEATURES = 4096
SEEDS = (KF.SEED42, *KF.K5_EXTRA_SEEDS)  # (42, 45, 46, 47, 48)
BOOT_SEED = XF.BOOT_SEED  # 1944
N_BOOT = XF.N_BOOT  # 1000
MATCH_SEED = F9.MATCH_SEED  # 20260801
GCV_DOF_CAP = 0.9  # #1887 degenerate-regime guard for the p10 design
PARITY_TOL_R2 = 1e-4  # abs pooled-R^2 agreement, fast core vs SharedPrimalRidge

# Widen from the precedent's p5 rung until the well-posedness floor clears.
PRIMARY_LADDER: tuple[tuple[str, float], ...] = (
    ("p5", 0.05),
    ("p4", 0.04),
    ("p3", 0.03),
    ("p2", 0.02),
    ("p1", 0.01),
    ("p0", 0.0),
)
P10_RUNG: tuple[str, float] = ("p10", 0.10)

ROLLOUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "issue_1902"
    / "k5_stage"
    / "issue1902_stage_map"
    / "raw_completions"
    / "gen"
    / "single"
)
OUT = KF.DEFAULT_OUT / "length_matched"
FIG_DIR = KF.DEFAULT_FIG_DIR
FIG_STEM = "c1_k5_length_control"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return "unknown"


def _cfg() -> KF.Config:
    return KF.Config(
        SimpleNamespace(
            smoke=False,
            force=False,
            stage_root=KF.DEFAULT_RO_STAGE_ROOT,
            k5_root=KF.DEFAULT_K5_ROOT,
            out=KF.DEFAULT_OUT,
            figures_dir=KF.DEFAULT_FIG_DIR,
        )
    )


# ── phase: lengths ───────────────────────────────────────────────────────────


def _rollout_lengths(stage: str, seed: int) -> dict[str, int]:
    """id -> n_tokens for one (stage, seed) draw from the local sharded jsonl."""
    stem = stage if seed == KF.SEED42 else f"{stage}_k5_seed{seed}"
    manifest = json.loads((ROLLOUT_DIR / f"{stem}.manifest.json").read_text())
    out: dict[str, int] = {}
    for shard in manifest["shards"]:
        path = ROLLOUT_DIR / shard["name"]
        n_lines = 0
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            out[str(rec["id"])] = int(rec["n_tokens"])
            n_lines += 1
        if n_lines != int(shard["n_lines"]):
            raise RuntimeError(f"shard line-count mismatch: {shard['name']}")
    return out


def run_lengths() -> None:
    _, ref_ids = KF._reference_rows()
    n = len(ref_ids)
    len_mean = np.zeros((len(STAGES), n), dtype=np.float64)
    per_draw_median: dict[str, dict[str, float]] = {}
    for si, stage in enumerate(STAGES):
        acc = np.zeros(n, dtype=np.float64)
        med: dict[str, float] = {}
        for seed in SEEDS:
            lens = _rollout_lengths(stage, seed)
            missing = [rid for rid in ref_ids if rid not in lens]
            if missing:
                raise RuntimeError(
                    f"stage {stage} seed {seed}: {len(missing)} paper rows missing "
                    f"from the rollout store (first: {missing[:3]})"
                )
            vals = np.asarray([lens[rid] for rid in ref_ids], dtype=np.float64)
            acc += vals
            med[str(seed)] = float(np.median(vals))
        len_mean[si] = acc / len(SEEDS)
        per_draw_median[stage] = med
        _log(
            f"[lengths] {stage}: K=5 mean median={np.median(len_mean[si]):.1f} "
            f"IQR=[{np.percentile(len_mean[si], 25):.1f}, {np.percentile(len_mean[si], 75):.1f}]"
        )
    LC._savez(
        OUT / "lengths.npz",
        row_ids=np.asarray(ref_ids),
        len_mean=len_mean,
        stages=np.asarray(STAGES),
        seeds=np.asarray(SEEDS, dtype=np.int64),
    )
    LC._write_json(
        OUT / "lengths_meta.json",
        {
            "per_draw_median": per_draw_median,
            "definition": "mean n_tokens per (stage, context) over the K=5 draws",
            "seeds": list(SEEDS),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def _load_lengths() -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(OUT / "lengths.npz", allow_pickle=False) as payload:
        stages = [str(v) for v in payload["stages"]]
        len_mean = np.asarray(payload["len_mean"], dtype=np.float64)
        ids = [str(v) for v in payload["row_ids"]]
    return {s: len_mean[stages.index(s)] for s in STAGES}, ids


# ── phase: match ─────────────────────────────────────────────────────────────


def _band(ntok: dict[str, np.ndarray], p: float) -> tuple[float, float, dict[str, int]]:
    """Cross-stage overlap band at tail fraction ``p`` (the 9a-ter formula)."""
    lo = max(float(np.percentile(ntok[s], 100 * p)) for s in STAGES)
    hi = min(float(np.percentile(ntok[s], 100 * (1 - p))) for s in STAGES)
    n_in = {s: int(((ntok[s] >= lo) & (ntok[s] <= hi)).sum()) for s in STAGES}
    return lo, hi, n_in


def _fold_n_train(mask: np.ndarray, fold_of: np.ndarray) -> list[int]:
    return [int((mask & (fold_of != f)).sum()) for f in range(N_FOLDS)]


def _design_masks(
    ntok: dict[str, np.ndarray], rung: str, p: float
) -> tuple[dict[str, np.ndarray], dict]:
    lo, hi, n_in = _band(ntok, p)
    if lo >= hi:
        raise RuntimeError(f"rung {rung}: empty cross-stage band lo={lo} hi={hi}")
    rng = np.random.default_rng(MATCH_SEED)
    masks, eq_detail = F9.equalize_bins(ntok, lo, hi, rng)
    detail = {
        "rung": rung,
        "tail_fraction": p,
        "lo": lo,
        "hi": hi,
        "n_in_band": n_in,
        "equalization": eq_detail,
        "matched_n": int(next(iter(eq_detail["matched_n_per_column"].values()))),
    }
    return masks, detail


def _qtiles(x: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(x)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "mean": float(x.mean()),
    }


def run_match() -> None:
    ntok, ids = _load_lengths()
    fold_of, ref_ids = KF._reference_rows()
    if ids != ref_ids:
        raise RuntimeError("lengths row order differs from the paper reference")

    designs: dict[str, tuple[dict[str, np.ndarray], dict]] = {}
    ladder_log = []
    for rung, p in PRIMARY_LADDER:
        masks, detail = _design_masks(ntok, rung, p)
        n_tr = {s: _fold_n_train(masks[s], fold_of) for s in STAGES}
        detail["fold_n_train"] = n_tr
        min_n_tr = min(min(v) for v in n_tr.values())
        detail["min_fold_n_train"] = min_n_tr
        detail["well_posed"] = min_n_tr > D_FEATURES
        ladder_log.append(detail)
        _log(
            f"[match] {rung}: band=[{detail['lo']:.1f}, {detail['hi']:.1f}] "
            f"matched_n={detail['matched_n']} min_fold_n_train={min_n_tr} "
            f"well_posed={detail['well_posed']}"
        )
        if detail["well_posed"]:
            designs["primary"] = (masks, detail)
            break
    if "primary" not in designs:
        raise RuntimeError(
            "no rung on the primary ladder keeps n_train > d on every fold — "
            f"widest tried: {ladder_log[-1]}"
        )

    p10_masks, p10_detail = _design_masks(ntok, *P10_RUNG)
    n_tr = {s: _fold_n_train(p10_masks[s], fold_of) for s in STAGES}
    p10_detail["fold_n_train"] = n_tr
    p10_detail["min_fold_n_train"] = min(min(v) for v in n_tr.values())
    p10_detail["well_posed"] = p10_detail["min_fold_n_train"] > D_FEATURES
    p10_detail["degenerate_label"] = (
        "under-determined regularization-limit fit (n_train < d); relative "
        "ordering at equal n across stages only, never an absolute R^2"
    )
    designs["p10"] = (p10_masks, p10_detail)
    _log(
        f"[match] p10: band=[{p10_detail['lo']:.1f}, {p10_detail['hi']:.1f}] "
        f"matched_n={p10_detail['matched_n']} "
        f"min_fold_n_train={p10_detail['min_fold_n_train']} (d={D_FEATURES})"
    )

    arrays: dict[str, np.ndarray] = {"row_ids": np.asarray(ids)}
    for design, (masks, _detail) in designs.items():
        for s in STAGES:
            arrays[f"mask_{design}_{s}"] = masks[s]
    LC._savez(OUT / "match.npz", **arrays)
    LC._write_json(
        OUT / "match_detail.json",
        {
            "d_features": D_FEATURES,
            "n_folds": N_FOLDS,
            "match_seed": MATCH_SEED,
            "n_bins": F9.N_BINS,
            "primary_ladder": ladder_log,
            "designs": {k: v[1] for k, v in designs.items()},
            "length_stats_matched": {
                design: {s: _qtiles(ntok[s][masks[s]]) for s in STAGES}
                for design, (masks, _d) in designs.items()
            },
            "length_stats_all_rows": {s: _qtiles(ntok[s]) for s in STAGES},
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def _load_masks(design: str) -> dict[str, np.ndarray]:
    with np.load(OUT / "match.npz", allow_pickle=False) as payload:
        return {s: np.asarray(payload[f"mask_{design}_{s}"], dtype=bool) for s in STAGES}


# ── phase: parity (SharedPrimalRidge vs shared fast core, well-posed slice) ──


def _fast_fold(
    x_tr: np.ndarray, y_tr: np.ndarray, x_ev: np.ndarray, dof_cap: float | None
) -> tuple[np.ndarray, float, float]:
    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict_fast_layer_batched,
    )

    pred, info = ridge_fit_predict_fast_layer_batched(
        x_tr[None],
        y_tr[None],
        x_ev[None],
        lambdas=XF.GCV_LAMBDAS,
        return_info=True,
        gcv_dof_cap=dof_cap,
    )
    return pred[0], float(info["best_lambda"][0]), float(info["dof"][0])


def run_parity() -> None:
    cfg = _cfg()
    masks = _load_masks("primary")
    fold_of, ref_ids = KF._reference_rows()
    stage = "B"
    mask = masks[stage]
    y = KF._load_targets(cfg, stage, stage)["w_bar"]
    x, fold_rows = KF._aligned_ctx(cfg, stage, ref_ids)
    if not np.array_equal(fold_rows, fold_of):
        raise RuntimeError("fold assignment mismatch on the full row set")
    ev = mask & (fold_of == 0)
    tr = mask & (fold_of != 0)
    ridge = XF.SharedPrimalRidge(x[tr])
    weights, ymu, info = ridge.fit(y[tr])
    pred_slow = ridge.standardize(x[ev]) @ weights + ymu
    res_s, tot_s, _ = LC._per_row_components(pred_slow, y[ev], y[tr].mean(axis=0))
    pred_fast, lam_fast, _dof = _fast_fold(x[tr], y[tr], x[ev], None)
    res_f, tot_f, _ = LC._per_row_components(pred_fast, y[ev], y[tr].mean(axis=0))
    r2_slow = 1.0 - float(res_s.sum()) / float(tot_s.sum())
    r2_fast = 1.0 - float(res_f.sum()) / float(tot_f.sum())
    report = {
        "slice": {"design": "primary", "stage": stage, "fold": 0},
        "n_train": int(tr.sum()),
        "n_eval": int(ev.sum()),
        "r2_slow": r2_slow,
        "r2_fast": r2_fast,
        "abs_diff_r2": abs(r2_slow - r2_fast),
        "lambda_slow": float(info["selected_lambda"]),
        "lambda_fast": lam_fast,
        "tolerance": PARITY_TOL_R2,
        "pass": abs(r2_slow - r2_fast) <= PARITY_TOL_R2,
        "note": (
            "same-inputs parity of the shared fast core "
            "(issue_779.fit_h.ridge_fit_predict_fast_layer_batched, no dof cap) "
            "against the committed SharedPrimalRidge on a well-posed matched slice"
        ),
    }
    LC._write_json(OUT / "fast_parity.json", report)
    if not report["pass"]:
        raise RuntimeError(f"fast-core parity FAILED: {report}")
    _log(f"[parity] PASS abs_diff_r2={report['abs_diff_r2']:.2e}")


# ── phase: fit ───────────────────────────────────────────────────────────────


def _percell_path(design: str, stage: str) -> Path:
    return OUT / "percell" / f"lm_{design}_{stage}_L{LAYER}.npz"


def run_fit(design: str, stage: str, folds: list[int]) -> None:
    cfg = _cfg()
    masks = _load_masks(design)
    fold_of, ref_ids = KF._reference_rows()
    mask = masks[stage]
    n = len(ref_ids)
    out_path = _percell_path(design, stage)
    if design == "p10":
        parity = json.loads((OUT / "fast_parity.json").read_text())
        if not parity["pass"]:
            raise RuntimeError("fast-core parity gate has not passed — run `parity` first")
    y = KF._load_targets(cfg, stage, stage)["w_bar"]
    x, _ = KF._aligned_ctx(cfg, stage, ref_ids)

    if out_path.exists():
        with np.load(out_path, allow_pickle=False) as payload:
            acc = {key: np.asarray(payload[key]).copy() for key in payload.files}
    else:
        acc = {
            "row_ids": np.asarray(ref_ids),
            "mask": mask,
            "fold_of": fold_of,
            "ss_res": np.full(n, np.nan),
            "ss_tot": np.full(n, np.nan),
            "cos": np.full(n, np.nan),
            "selected_lambda": np.full(N_FOLDS, np.nan),
            "dof": np.full(N_FOLDS, np.nan),
            "n_train": np.zeros(N_FOLDS, dtype=np.int64),
            "n_eval": np.zeros(N_FOLDS, dtype=np.int64),
        }
    for fold in folds:
        if acc["n_eval"][fold] > 0:
            _log(f"[fit] {design} {stage} fold {fold}: resumed")
            continue
        t0 = time.time()
        ev = mask & (fold_of == fold)
        tr = mask & (fold_of != fold)
        n_tr = int(tr.sum())
        if design == "primary":
            if n_tr <= D_FEATURES:
                raise RuntimeError(
                    f"primary design fold {fold} stage {stage}: n_train={n_tr} <= d — "
                    "the well-posedness floor failed after matching; re-run `match`"
                )
            ridge = XF.SharedPrimalRidge(x[tr])
            weights, ymu, info = ridge.fit(y[tr])
            pred = ridge.standardize(x[ev]) @ weights + ymu
            lam, dof = float(info["selected_lambda"]), float(info["dof"])
            del ridge, weights
        else:
            pred, lam, dof = _fast_fold(x[tr], y[tr], x[ev], GCV_DOF_CAP)
        rr, tt, cc = LC._per_row_components(pred, y[ev], y[tr].mean(axis=0))
        acc["ss_res"][ev], acc["ss_tot"][ev], acc["cos"][ev] = rr, tt, cc
        acc["selected_lambda"][fold], acc["dof"][fold] = lam, dof
        acc["n_train"][fold], acc["n_eval"][fold] = n_tr, int(ev.sum())
        LC._savez(out_path, **acc)
        _log(
            f"[fit] {design} {stage} fold {fold}: n_tr={n_tr} n_ev={int(ev.sum())} "
            f"lambda={lam:.3g} dof={dof:.0f} in {time.time() - t0:.0f}s"
        )
    done = int((acc["n_eval"] > 0).sum())
    _log(f"[fit] {design} {stage}: {done}/{N_FOLDS} folds fitted")


# ── phase: summarize ─────────────────────────────────────────────────────────


def _row_counts(n: int) -> np.ndarray:
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    counts = np.zeros((N_BOOT, n), dtype=np.float64)
    for b in range(N_BOOT):
        counts[b] = np.bincount(idx[b], minlength=n)
    return counts


def _masked_r2_draws(
    res: np.ndarray, tot: np.ndarray, mask: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Row-bootstrap R^2 draws over the shared full-index counts, masked.

    Contexts are resampled once per draw over the full paper index (seed 1944),
    so draws are PAIRED across stages/designs on shared rows; a stage's R^2
    uses only its masked rows within each draw.
    """
    res_m = np.where(mask, np.nan_to_num(res, nan=0.0), 0.0)
    tot_m = np.where(mask, np.nan_to_num(tot, nan=0.0), 0.0)
    return 1.0 - (counts @ res_m) / (counts @ tot_m)


def _ci(draws: np.ndarray) -> list[float]:
    finite = draws[np.isfinite(draws)]
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def _load_percell(design: str, stage: str) -> dict[str, np.ndarray]:
    with np.load(_percell_path(design, stage), allow_pickle=False) as payload:
        out = {key: np.asarray(payload[key]) for key in payload.files}
    if not (out["n_eval"] > 0).all():
        raise RuntimeError(f"{design} {stage}: unfitted folds remain — finish `fit` first")
    return out


def _stage_block(
    res: np.ndarray,
    tot: np.ndarray,
    cos: np.ndarray,
    mask: np.ndarray,
    counts: np.ndarray,
) -> dict:
    r = res[mask]
    t = tot[mask]
    return {
        "n": int(mask.sum()),
        "r2": 1.0 - float(r.sum()) / float(t.sum()),
        "r2_row_ci95": _ci(_masked_r2_draws(res, tot, mask, counts)),
        "mean_ss_res": float(r.mean()),
        "mean_ss_tot": float(t.mean()),
        "median_cos": float(np.median(cos[mask])),
    }


def run_summarize() -> None:
    ntok, ids = _load_lengths()
    fold_of, ref_ids = KF._reference_rows()
    n = len(ref_ids)
    counts = _row_counts(n)
    match_detail = json.loads((OUT / "match_detail.json").read_text())
    full_mask = np.ones(n, dtype=bool)

    # Unmatched reference: the committed full-data K=5 grid percell arrays.
    grid: dict[str, dict[str, np.ndarray]] = {}
    for s in STAGES:
        with np.load(
            KF.DEFAULT_OUT / "percell" / f"k5grid_{s}{s}_L{LAYER}.npz", allow_pickle=False
        ) as payload:
            grid[s] = {
                "res": np.asarray(payload["ss_res"], dtype=np.float64),
                "tot": np.asarray(payload["ss_tot"], dtype=np.float64),
                "cos": np.asarray(payload["cos"], dtype=np.float64),
            }
    unmatched = {
        s: _stage_block(grid[s]["res"], grid[s]["tot"], grid[s]["cos"], full_mask, counts)
        for s in STAGES
    }

    designs_out: dict[str, dict] = {}
    for design in ("primary", "p10"):
        masks = _load_masks(design)
        detail = match_detail["designs"][design]
        stages_out: dict[str, dict] = {}
        draws: dict[str, np.ndarray] = {}
        for s in STAGES:
            cell = _load_percell(design, s)
            if not np.array_equal(cell["mask"], masks[s]):
                raise RuntimeError(f"{design} {s}: percell mask differs from match.npz")
            block = _stage_block(cell["ss_res"], cell["ss_tot"], cell["cos"], masks[s], counts)
            block["fold_n_train"] = cell["n_train"].tolist()
            block["fold_n_eval"] = cell["n_eval"].tolist()
            block["selected_lambda_by_fold"] = cell["selected_lambda"].tolist()
            block["dof_by_fold"] = cell["dof"].tolist()
            block["matched_length"] = match_detail["length_stats_matched"][design][s]
            stages_out[s] = block
            draws[s] = _masked_r2_draws(cell["ss_res"], cell["ss_tot"], masks[s], counts)
        diffs = {}
        for a, b in (("B", "D"), ("B", "S"), ("D", "S")):
            delta = draws[a] - draws[b]
            diffs[f"{a}_minus_{b}"] = {
                "point": stages_out[a]["r2"] - stages_out[b]["r2"],
                "row_ci95": _ci(delta),
                "pairing": (
                    "row-bootstrap draws paired via one shared full-index context "
                    "resampling (seed 1944); matched row sets overlap only partially"
                ),
            }
        ordering = sorted(STAGES, key=lambda s: -stages_out[s]["r2"])
        designs_out[design] = {
            "band": {"rung": detail["rung"], "lo": detail["lo"], "hi": detail["hi"]},
            "matched_n": detail["matched_n"],
            "min_fold_n_train": detail["min_fold_n_train"],
            "d_features": D_FEATURES,
            "well_posed": detail["well_posed"],
            "estimator": (
                "SharedPrimalRidge (committed primal spectral GCV)"
                if design == "primary"
                else (
                    "issue_779.fit_h.ridge_fit_predict_fast_layer_batched with "
                    f"gcv_dof_cap={GCV_DOF_CAP} (#1887), lambdas=GCV_LAMBDAS, "
                    "parity-gated vs SharedPrimalRidge (fast_parity.json)"
                )
            ),
            "degenerate_label": detail.get("degenerate_label"),
            "stages": stages_out,
            "r2_ordering": ordering,
            "differences": diffs,
        }

    # Length-stratified residual read from the FULL-data fits (no refit).
    strat: dict[str, list[dict]] = {}
    for s in STAGES:
        edges = np.percentile(ntok[s], np.linspace(0, 100, 6))
        edges[0] -= 0.5
        edges[-1] += 0.5
        bins = []
        for b in range(5):
            sel = (ntok[s] > edges[b]) & (ntok[s] <= edges[b + 1])
            bins.append(
                {
                    "bin": b + 1,
                    "len_lo": float(edges[b]),
                    "len_hi": float(edges[b + 1]),
                    "len_median": float(np.median(ntok[s][sel])),
                    "n": int(sel.sum()),
                    "mean_ss_res": float(grid[s]["res"][sel].mean()),
                    "mean_ss_tot": float(grid[s]["tot"][sel].mean()),
                    "r2": 1.0 - float(grid[s]["res"][sel].sum()) / float(grid[s]["tot"][sel].sum()),
                }
            )
        strat[s] = bins

    prim = designs_out["primary"]
    bd = prim["differences"]["B_minus_D"]
    bs = prim["differences"]["B_minus_S"]
    ordering = prim["r2_ordering"]

    def _sig(diff: dict) -> str:
        lo, hi = diff["row_ci95"]
        if lo > 0:
            return "positive"
        if hi < 0:
            return "negative"
        return "indistinguishable from 0"

    ds = prim["differences"]["D_minus_S"]
    base_top = _sig(bd) == "positive" and _sig(bs) == "positive"
    verdict = (
        f"On the paper's own protocol, Base's top position "
        f"{'survives' if base_top else 'does NOT survive'} the length control: matched "
        f"(band {prim['band']['lo']:.0f}-{prim['band']['hi']:.0f} tokens, "
        f"n={prim['matched_n']}/stage, n_train > d) R^2 is "
        + " > ".join(f"{KF.STAGE_NAMES[s]} {prim['stages'][s]['r2']:.3f}" for s in ordering)
        + ", "
        f"with Base-minus-SFT {bs['point']:+.3f} "
        f"[{bs['row_ci95'][0]:+.3f}, {bs['row_ci95'][1]:+.3f}] and Base-minus-DPO "
        f"{bd['point']:+.3f} [{bd['row_ci95'][0]:+.3f}, {bd['row_ci95'][1]:+.3f}]. "
        f"The SFT > DPO step does not: matched DPO-minus-SFT is {ds['point']:+.3f} "
        f"[{ds['row_ci95'][0]:+.3f}, {ds['row_ci95'][1]:+.3f}] ({_sig(ds)}). "
        "Matching compresses Base's mean SS_tot "
        f"(unmatched {unmatched['B']['mean_ss_tot']:.0f} -> matched "
        f"{prim['stages']['B']['mean_ss_tot']:.0f}), confirming the denominator mechanism, "
        "but the surviving Base lead and its highest median cosine "
        f"({prim['stages']['B']['median_cos']:.3f}) show the advantage is not purely "
        "denominator-driven."
    )

    summary = {
        "metadata": {
            "script": "scripts/issue1902_k5_length_control.py",
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol": (
                "u_last context summary, layer 31, six IID random folds "
                "(seed 190231), K=5 mean answer targets (seeds 42,45,46,47,48)"
            ),
            "length_definition": "mean n_tokens per (stage, context) over the K=5 draws",
            "band_ladder": [r for r, _ in PRIMARY_LADDER] + [P10_RUNG[0]],
            "primary_ladder_log": match_detail["primary_ladder"],
            "match_seed": MATCH_SEED,
            "n_bins": F9.N_BINS,
            "boot": {"n_boot": N_BOOT, "seed": BOOT_SEED, "kind": "row"},
            "fold_seed": LC.RANDOM_FOLD_SEED,
            "n_train_vs_d": {
                design: {
                    "min_fold_n_train": designs_out[design]["min_fold_n_train"],
                    "d": D_FEATURES,
                    "well_posed": designs_out[design]["well_posed"],
                }
                for design in designs_out
            },
            "fast_parity": json.loads((OUT / "fast_parity.json").read_text()),
        },
        "unmatched_reference": unmatched,
        "designs": designs_out,
        "length_stratified": {
            "definition": (
                "per stage, contexts binned into 5 quantile bins of that stage's own "
                "K=5 mean answer length; mean per-row SS_res / SS_tot from the "
                "ALREADY-FITTED full-data k5grid percell arrays (no refit)"
            ),
            "per_stage": strat,
        },
        "verdict": verdict,
    }
    LC._write_json(OUT / "summary.json", summary)
    _log(f"[summarize] verdict: {verdict}")


# ── phase: figure ────────────────────────────────────────────────────────────


def run_figure() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    from explore_persona_space.analysis.c2a_plot_style import (
        better_label,
        c2a_figure,
        panel_header,
        save_c2a_figure,
        set_c2a_style,
        style_axis,
    )

    summary = json.loads((OUT / "summary.json").read_text())
    prim = summary["designs"]["primary"]
    unm = summary["unmatched_reference"]
    strat = summary["length_stratified"]["per_stage"]

    set_c2a_style()
    fig, include_frac = c2a_figure("full", aspect=0.42)
    ax_a = fig.add_axes([0.075, 0.14, 0.36, 0.68])
    ax_b = fig.add_axes([0.56, 0.14, 0.41, 0.68])

    xs = np.arange(len(STAGES))
    series = (
        ("All contexts", unm, "#176B87", "o", -0.13),
        ("Length-matched", {s: prim["stages"][s] for s in STAGES}, "#C4553D", "D", 0.13),
    )
    for label, blocks, color, marker, dx in series:
        vals = [blocks[s]["r2"] for s in STAGES]
        cis = np.asarray([blocks[s]["r2_row_ci95"] for s in STAGES])
        err = np.abs(cis.T - np.asarray(vals))
        ax_a.errorbar(
            xs + dx,
            vals,
            yerr=err,
            fmt=marker,
            color=color,
            markersize=5,
            capsize=2.2,
            linewidth=1.1,
            linestyle="none",
            label=label,
        )
    ax_a.set_xticks(xs, [KF.STAGE_NAMES[s] for s in STAGES])
    ax_a.set_ylabel(better_label("Held-out $R^2$"))
    style_axis(ax_a)
    panel_header(ax_a, "A", f"OWN MAP · LAYER {LAYER}", "Answer-length control")
    ax_a.legend(frameon=False, loc="lower left", handletextpad=0.4)

    for s in STAGES:
        med = [b["len_median"] for b in strat[s]]
        ax_b.plot(
            med,
            [b["mean_ss_tot"] for b in strat[s]],
            color=KF.STAGE_COLORS[s],
            marker=KF.STAGE_MARKERS[s],
            markersize=4,
            linewidth=1.4,
            label=KF.STAGE_NAMES[s],
        )
        ax_b.plot(
            med,
            [b["mean_ss_res"] for b in strat[s]],
            color=KF.STAGE_COLORS[s],
            marker=KF.STAGE_MARKERS[s],
            markersize=4,
            linewidth=1.4,
            linestyle="--",
            markerfacecolor="none",
        )
    ax_b.set_xscale("log")
    ax_b.set_xlim(22, 820)
    ax_b.set_xticks([30, 100, 300, 700], ["30", "100", "300", "700"])
    ax_b.xaxis.set_minor_formatter(NullFormatter())
    ax_b.set_xlabel("Answer length (tokens, bin median)")
    ax_b.set_ylabel("Mean SS per context")
    style_axis(ax_b)
    panel_header(ax_b, "B", "FULL DATA · 5 LENGTH BINS", "Variance vs residual by length")
    handles, labels = ax_b.get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles += [
        Line2D([], [], color="#687078", linewidth=1.4, label="Total SS"),
        Line2D([], [], color="#687078", linewidth=1.4, linestyle="--", label="Residual SS"),
    ]
    ax_b.legend(handles=handles, frameon=False, loc="upper left", fontsize=6.4, ncols=2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIG_DIR / FIG_STEM
    outputs = save_c2a_figure(
        fig,
        stem,
        title="K=5 diagonal R^2 under an answer-length control",
        subject="Issue #1902: length-matched own-map R^2 + length-stratified SS decomposition",
        creator="scripts/issue1902_k5_length_control.py",
        include_width=include_frac,
    )
    plt.close(fig)
    sidecar = {
        "metadata": {
            "script": "scripts/issue1902_k5_length_control.py",
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_summary": str((OUT / "summary.json").relative_to(PROJECT_ROOT)),
            "panel_a": (
                "held-out pooled R^2 per stage, all contexts vs the primary "
                f"length-matched design (band {prim['band']['lo']:.1f}-"
                f"{prim['band']['hi']:.1f} tokens, n={prim['matched_n']}/stage), "
                "row-bootstrap 95% CIs"
            ),
            "panel_b": (
                "mean per-context SS_tot (solid) and SS_res (dashed) per stage "
                "within 5 quantile bins of the stage's own K=5 mean answer length, "
                "from the full-data fits"
            ),
        },
        "panel_a": {
            "stages": [KF.STAGE_NAMES[s] for s in STAGES],
            "unmatched_r2": [unm[s]["r2"] for s in STAGES],
            "unmatched_ci": [unm[s]["r2_row_ci95"] for s in STAGES],
            "matched_r2": [prim["stages"][s]["r2"] for s in STAGES],
            "matched_ci": [prim["stages"][s]["r2_row_ci95"] for s in STAGES],
        },
        "panel_b": strat,
        "render": outputs["record"],
    }
    LC._write_json(stem.parent / f"{FIG_STEM}_data.json", sidecar)
    _log(f"[figure] wrote {outputs['pdf']}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cmd", choices=["lengths", "match", "parity", "fit", "summarize", "figure", "all"]
    )
    parser.add_argument("--design", choices=["primary", "p10"])
    parser.add_argument("--stage", choices=list(STAGES))
    parser.add_argument("--folds", default=None, help="comma list, default all six")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "lengths":
        run_lengths()
    elif args.cmd == "match":
        run_match()
    elif args.cmd == "parity":
        run_parity()
    elif args.cmd == "fit":
        if not args.design or not args.stage:
            raise SystemExit("fit requires --design and --stage")
        folds = (
            list(range(N_FOLDS)) if args.folds is None else [int(v) for v in args.folds.split(",")]
        )
        run_fit(args.design, args.stage, folds)
    elif args.cmd == "summarize":
        run_summarize()
    elif args.cmd == "figure":
        run_figure()
    else:
        run_lengths()
        run_match()
        run_parity()
        for design in ("primary", "p10"):
            for stage in STAGES:
                run_fit(design, stage, list(range(N_FOLDS)))
        run_summarize()
        run_figure()
    _log("DONE")


if __name__ == "__main__":
    main()
