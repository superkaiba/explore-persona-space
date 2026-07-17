#!/usr/bin/env python3
"""Issue #825 — matched-N turn-depth read of the context->answer map.

The raw per-turn read (``scripts/issue825_turn_depth_map.py``,
``eval_results/issue_825/turn_depth_map/results.json``) fits
``context_k -> answer_k_t1`` (offset 0: v_C at the last context token -> v_A the
answer-span mean) per user-turn depth. But n COLLAPSES with depth (t1 n=497 ->
t11 n=171 -> t19 n=50 -> single digits deeper), so the deep-turn R2 decline is
CONFOUNDED with small-n: a smaller fit sample has lower held-out R2 for a fixed
true signal, and its shuffle-null band widens toward 0. This re-analysis makes
the turn-depth comparison FAIR by controlling for sample size.

Design (0-GPU, ANALYSIS-ONLY on the EXISTING #1092 dynamics store):

* For each of two MATCHED levels, subsample every eligible turn cell to a COMMON
  N_MATCH, fit the SAME grouped-CV PRESS ridge, and AVERAGE held-out R2 over
  K subsample draws (fixed seeds). Two levels expose the depth-vs-power tradeoff:
    - level_171: N_MATCH=171 over turns t1..t11 (the turns with n>=171) —
      more depth-limited but higher per-turn power.
    - level_68 : N_MATCH=68  over turns t1..t17 (the turns with n>=68) —
      reaches deeper but at lower power.
  Turn depths are odd (1,3,5,...); the eligible sets are read from the actual
  per-turn n in the raw results.json.
* Within each turn cell each conv_id contributes exactly one (context, answer)
  pair, so conv_id is unique per row and the grouped-by-conv_id fold partition
  reduces to a random K-fold split of the subsampled rows.
* The shuffle-answer permutation null is recomputed AT MATCHED n (headline layer
  19, both models): per subsample draw, permute the answer rows, refit ridge per
  fold with the same per-fold lambda the real fit selected, pool the draws.
* The RAW (unmatched) curve + its raw null band are carried verbatim from the
  banked results.json for contrast.

The ridge fit core (PRESS-ridge grouped-CV, ``_real_fit_and_folds``) and the
batched dual-space permutation null (``_dual_perm_null``) are REUSED VERBATIM
from ``issue825_turn_depth_map`` — this script only adds matched-N subsampling
and pooling on top of them.

Validation gate: at turn t11 in level_171, N_MATCH (171) EQUALS the full turn n
(171), so every subsample draw is the full row set and the grouped folds are
seed-fixed (FOLD_SEED=0) => the matched R2 must reproduce the banked RAW t11 R2
bit-for-bit (sd across draws = 0). Plus an independent recompute of one matched
cell asserts self-consistency. FAIL aborts.

Usage::

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
        uv run python scripts/issue825_turn_depth_matched_n.py --skip-download
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the raw #825 turn-depth read's data loaders + fit core + batched null
# VERBATIM (single source of truth; no re-implementation of the ridge or null).
import issue825_turn_depth_map as TDM  # noqa: E402
import numpy as np  # noqa: E402

LAYERS = TDM.LAYERS  # (14, 18, 19)
HEADLINE_LAYER = TDM.HEADLINE_LAYER  # 19
SRC_KIND = TDM.SRC_KIND  # context_k
DST_KIND = TDM.DST_KIND  # answer_k_t1
HF_DATA_REPO = TDM.HF_DATA_REPO
HF_SUMMARIES_BASE = TDM.HF_SUMMARIES_BASE
MODEL_CELL = TDM.MODEL_CELL
MODELS = ("instruct", "pretrained")

# Matched-N levels: (name, N_MATCH, min turn-n to be eligible). Turn sets are
# derived from the actual per-turn n in the raw results.json (odd turns only).
LEVELS = (
    {"name": "level_171", "n_match": 171},
    {"name": "level_68", "n_match": 68},
)
K_DRAWS = 10  # subsample draws averaged per (model, layer, turn, level)
K_NULL = 10  # subsample draws over which the matched null is pooled (headline layer)
N_NULL_DRAWS = 100  # shuffle-answer permutation draws per subsample draw
NULL_LAYERS = (HEADLINE_LAYER,)  # matched null computed for the headline layer only
SUB_SEED_BASE = 8250
NULL_SEED_BASE = 10920
VALIDATION_TOL = 1e-9

RAW_JSON = PROJECT_ROOT / "eval_results/issue_825/turn_depth_map/results.json"
OUT_JSON = PROJECT_ROOT / "eval_results/issue_825/turn_depth_matched_n/results.json"
OUT_FIG_DIR = PROJECT_ROOT / "figures/issue_825"
FIG_STEM = "turn_depth_matched_n"


def _sub_seed(model_idx: int, level_idx: int, turn: int, draw: int) -> int:
    return SUB_SEED_BASE + model_idx * 10_000_000 + level_idx * 1_000_000 + turn * 1000 + draw


def _null_seed(model_idx: int, level_idx: int, turn: int, draw: int) -> int:
    return NULL_SEED_BASE + model_idx * 10_000_000 + level_idx * 1_000_000 + turn * 1000 + draw


def _eligible_turns(n_per_turn: dict[str, int], n_match: int) -> list[int]:
    """Odd turn depths whose per-turn n (min over models) is >= n_match, ascending."""
    return sorted(t for t, n in n_per_turn.items() if n >= n_match)


def _matched_r2_draws(
    X: np.ndarray, Y: np.ndarray, sub: np.ndarray, pair_rows: list[dict]
) -> tuple[float | None, list | None]:
    """One matched subsample fit -> (held-out R2, folds). Reuses TDM._real_fit_and_folds."""
    rows = [pair_rows[i] for i in sub]
    rf = TDM._real_fit_and_folds(X[sub], Y[sub], rows)
    if rf is None:
        return None, None
    fit, folds = rf
    return float(fit["r2"]), (fit["lambda_indices"], folds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null-draws", type=int, default=N_NULL_DRAWS)
    ap.add_argument("--k-draws", type=int, default=K_DRAWS)
    ap.add_argument("--k-null", type=int, default=K_NULL)
    ap.add_argument("--local-root", default=str(PROJECT_ROOT / "data/issue_825/summaries_dl"))
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    k_draws = int(args.k_draws)
    k_null = int(args.k_null)
    n_null_draws = int(args.n_null_draws)

    local_root = Path(args.local_root)
    if args.skip_download:
        summaries_dir = local_root / HF_SUMMARIES_BASE
    else:
        summaries_dir = TDM._download_summaries(local_root)
    assert summaries_dir.is_dir(), f"summaries_dir missing: {summaries_dir}"

    with open(RAW_JSON) as f:
        raw_payload = json.load(f)
    raw_results = raw_payload["results"]  # {model: {layer(str): {turn(str): entry}}}
    # per-turn n is layer-independent; read from the raw payload (min over models
    # for eligibility so both models share the same eligible turn set).
    raw_npt = raw_payload["n_per_turn"]
    n_per_turn_min: dict[int, int] = {}
    for mt in MODELS:
        for t_s, n in raw_npt[mt].items():
            t = int(t_s)
            n_per_turn_min[t] = min(n_per_turn_min.get(t, n), int(n))

    levels = []
    for lv in LEVELS:
        turns = _eligible_turns(n_per_turn_min, lv["n_match"])
        levels.append({"name": lv["name"], "n_match": int(lv["n_match"]), "turns": turns})
        print(f"[level] {lv['name']}: N_MATCH={lv['n_match']} turns={turns}", flush=True)

    # matched[level][model][layer(str)][turn(str)] = entry
    matched: dict = {
        lv["name"]: {mt: {str(L): {} for L in LAYERS} for mt in MODELS} for lv in levels
    }

    t_start = time.time()
    for model_idx, mt in enumerate(MODELS):
        paired = TDM._build_pairing(summaries_dir, mt)
        ci = np.asarray([p[0] for p in paired], dtype=np.int64)
        aj = np.asarray([p[1] for p in paired], dtype=np.int64)
        pair_rows = [{"conv_id": p[2], "turn_index": p[3]} for p in paired]
        turns_all = [p[3] for p in paired]
        turn_sel = {
            t: np.asarray([i for i, tt in enumerate(turns_all) if tt == t], dtype=np.int64)
            for t in sorted(set(turns_all))
        }
        # preload the paired X/Y per layer (rows shared across layers)
        XY: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for L in LAYERS:
            arr_c, _ = TDM._load_summary(summaries_dir, f"dynamics_{mt}", SRC_KIND, L)
            arr_a, _ = TDM._load_summary(summaries_dir, f"dynamics_{mt}", DST_KIND, L)
            XY[L] = (arr_c[ci], arr_a[aj])
        print(f"[compute] {mt}: pairs={ci.size}", flush=True)

        for level_idx, lv in enumerate(levels):
            n_match = lv["n_match"]
            for turn in lv["turns"]:
                sel = turn_sel.get(turn)
                if sel is None or sel.size < n_match:
                    continue
                n_avail = int(sel.size)
                # draw the SAME subsample of conversations per (model, level, turn, draw),
                # reused across layers (pairing is layer-independent).
                subs = [
                    np.random.default_rng(_sub_seed(model_idx, level_idx, turn, d)).choice(
                        sel, size=n_match, replace=False
                    )
                    for d in range(k_draws)
                ]
                for L in LAYERS:
                    X, Y = XY[L]
                    r2_draws: list[float] = []
                    null_pool: list[float] = []
                    for d, sub in enumerate(subs):
                        r2, extra = _matched_r2_draws(X, Y, sub, pair_rows)
                        if r2 is None:
                            continue
                        r2_draws.append(r2)
                        if L in NULL_LAYERS and d < k_null:
                            lam_idx, folds = extra
                            draws = TDM._dual_perm_null(
                                X[sub],
                                Y[sub],
                                folds,
                                lam_idx,
                                n_null_draws,
                                _null_seed(model_idx, level_idx, turn, d),
                            )
                            null_pool.extend(float(v) for v in draws[np.isfinite(draws)])
                    arr = np.asarray(r2_draws, dtype=np.float64)
                    entry: dict = {
                        "turn": turn,
                        "n_match": n_match,
                        "n_avail": n_avail,
                        "real_r2_mean": (float(arr.mean()) if arr.size else None),
                        "real_r2_sd": (float(arr.std(ddof=0)) if arr.size else None),
                        "real_r2_draws": [float(v) for v in arr],
                        "n_draws": int(arr.size),
                        "null_mean": None,
                        "null_lo": None,
                        "null_hi": None,
                        "null_n_draws": 0,
                    }
                    if null_pool:
                        npool = np.asarray(null_pool, dtype=np.float64)
                        entry["null_mean"] = float(npool.mean())
                        entry["null_lo"] = float(np.percentile(npool, 2.5))
                        entry["null_hi"] = float(np.percentile(npool, 97.5))
                        entry["null_n_draws"] = int(npool.size)
                    matched[lv["name"]][mt][str(L)][str(turn)] = entry
            print(
                f"[compute] {mt} {lv['name']}: done ({time.time() - t_start:.0f}s elapsed)",
                flush=True,
            )
        del XY

    # ---- validation gate ----
    validation = _run_validation(matched, raw_results, summaries_dir, k_draws)

    # ---- write results JSON ----
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue": 825,
        "analysis": "matched-N turn-depth read (control for sample size)",
        "description": (
            "Per-turn held-out R2 of the linear context->answer map "
            "(context_k -> answer_k_t1, offset 0) at MATCHED sample size, so any "
            "remaining decline across turn depth is a turn-depth effect rather "
            "than a small-n power artifact. Each eligible turn cell is subsampled "
            "to a common N_MATCH and R2 is averaged over K subsample draws; the "
            "raw (unmatched) curve is carried alongside for contrast."
        ),
        "arm": "context_k -> answer_k_t1 (offset 0)",
        "estimator": (
            "PRESS-ridge (issue923 press_fit_predict, standardize=True, lambda "
            "grid [1e-2,1e-1,1,10,100,1000]), grouped-by-conv_id 6-fold CV; "
            "held-out R2 aggregated over folds (issue1092_fit_grid._fit_cv, reused "
            "via issue825_turn_depth_map._real_fit_and_folds). Within a turn cell "
            "each conv_id is unique, so grouped folds = random K-fold."
        ),
        "matched_recipe": (
            "For each level and eligible turn, draw K subsamples of N_MATCH pairs "
            "(without replacement, fixed per-(model,level,turn,draw) seeds; the "
            "SAME subsample of conversations is reused across layers). Fit the "
            "PRESS ridge on each subsample, report mean +/- sd of held-out R2 over "
            "the K draws. Eligible turns per level are the odd turn depths whose "
            "per-turn n (min over models) is >= N_MATCH, read from the raw "
            "results.json."
        ),
        "null_recipe": (
            "Shuffle-answer permutation null AT MATCHED n (headline layer 19, both "
            "models only): per subsample draw, permute the answer rows at the row "
            "level (break the context<->answer pairing), keep the conv_id-grouped "
            "fold partition and per-fold lambda IDENTICAL to that draw's real fit, "
            f"refit ridge per fold, score held-out R2 over {N_NULL_DRAWS} draws; "
            f"pool over the first {K_NULL} subsample draws and report mean + [2.5, "
            "97.5] percentile band. Batched dual-space null "
            "(issue825_turn_depth_map._dual_perm_null, reused verbatim; float32 "
            "draws, ss_res accumulated in float64)."
        ),
        "levels": levels,
        "k_draws": k_draws,
        "k_null": k_null,
        "n_null_draws": n_null_draws,
        "null_layers": list(NULL_LAYERS),
        "sub_seed_base": SUB_SEED_BASE,
        "null_seed_base": NULL_SEED_BASE,
        "layers": list(LAYERS),
        "headline_layer": HEADLINE_LAYER,
        "model_cells": MODEL_CELL,
        "models": list(MODELS),
        "n_per_turn": raw_payload["n_per_turn"],
        "raw_curve_source": str(RAW_JSON.relative_to(PROJECT_ROOT)),
        "raw_curve": raw_results,
        "validation": validation,
        "source_json": raw_payload.get("source_json"),
        "hf_data_repo": HF_DATA_REPO,
        "hf_summaries_prefix": HF_SUMMARIES_BASE,
        "hf_paths_read": raw_payload.get("hf_paths_read"),
        "git_commit": TDM._git_commit(),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "matched": matched,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[write] {OUT_JSON}")

    _plot(payload)


def _run_validation(matched: dict, raw_results: dict, summaries_dir: Path, k_draws: int) -> dict:
    """Two gates. Aborts (SystemExit) on FAIL.

    1. matched-vs-banked-raw: at (instruct, L19, level_171, turn 11) N_MATCH ==
       full turn n (171), so the matched fit uses the full seed-fixed folds and
       must reproduce the banked raw t11 R2 (sd across draws = 0).
    2. self-consistency: independently recompute one matched draw and assert it
       equals the stored draw value bit-for-bit.
    """
    val: dict = {"tol": VALIDATION_TOL}

    # gate 1
    e = matched["level_171"]["instruct"]["19"].get("11")
    banked_raw_t11 = raw_results["instruct"]["19"]["11"]["real_r2"]
    assert e is not None, "validation: (instruct,L19,level_171,t11) cell missing"
    d_mean = abs(e["real_r2_mean"] - banked_raw_t11)
    gate1 = d_mean <= VALIDATION_TOL and e["real_r2_sd"] <= VALIDATION_TOL
    val["gate1_matched_full_set_vs_banked_raw"] = {
        "cell": "instruct/L19/level_171/turn11",
        "n_match": e["n_match"],
        "n_avail": e["n_avail"],
        "matched_r2_mean": e["real_r2_mean"],
        "matched_r2_sd": e["real_r2_sd"],
        "banked_raw_r2": banked_raw_t11,
        "abs_diff_mean_vs_banked": d_mean,
        "pass": bool(gate1),
    }

    # gate 2: recompute (instruct, L19, level_68, turn 1, draw 0)
    mt, model_idx = "instruct", 0
    level_idx = next(i for i, lv in enumerate(LEVELS) if lv["name"] == "level_68")
    turn, draw = 1, 0
    paired = TDM._build_pairing(summaries_dir, mt)
    ci = np.asarray([p[0] for p in paired], dtype=np.int64)
    aj = np.asarray([p[1] for p in paired], dtype=np.int64)
    pair_rows = [{"conv_id": p[2], "turn_index": p[3]} for p in paired]
    turns_all = [p[3] for p in paired]
    sel = np.asarray([i for i, tt in enumerate(turns_all) if tt == turn], dtype=np.int64)
    sub = np.random.default_rng(_sub_seed(model_idx, level_idx, turn, draw)).choice(
        sel, size=68, replace=False
    )
    arr_c, _ = TDM._load_summary(summaries_dir, f"dynamics_{mt}", SRC_KIND, HEADLINE_LAYER)
    arr_a, _ = TDM._load_summary(summaries_dir, f"dynamics_{mt}", DST_KIND, HEADLINE_LAYER)
    r2, _ = _matched_r2_draws(arr_c[ci], arr_a[aj], sub, pair_rows)
    stored = matched["level_68"]["instruct"]["19"]["1"]["real_r2_draws"][0]
    d2 = abs(r2 - stored)
    gate2 = d2 <= VALIDATION_TOL
    val["gate2_self_consistency_recompute"] = {
        "cell": "instruct/L19/level_68/turn1/draw0",
        "recomputed_r2": float(r2),
        "stored_r2": float(stored),
        "abs_diff": d2,
        "pass": bool(gate2),
    }

    val["pass"] = bool(gate1 and gate2)
    print(
        f"[validation] gate1 matched-full-vs-banked-raw |d|={d_mean:.2e} sd={e['real_r2_sd']:.2e} "
        f"-> {'PASS' if gate1 else 'FAIL'} | gate2 recompute |d|={d2:.2e} "
        f"-> {'PASS' if gate2 else 'FAIL'}"
    )
    if not val["pass"]:
        raise SystemExit(
            "VALIDATION GATE FAILED — matched-N R2 does not reproduce the banked "
            "raw value (gate1) or is not self-consistent (gate2). Do not use these outputs."
        )
    return val


def _matched_curve(matched: dict, level_name: str, mt: str, layer: int):
    node = matched[level_name][mt][str(layer)]
    turns = sorted((int(t) for t in node), key=int)
    xs, mean, sd, nlo, nhi = [], [], [], [], []
    for t in turns:
        e = node[str(t)]
        if e["real_r2_mean"] is None:
            continue
        xs.append(t)
        mean.append(e["real_r2_mean"])
        sd.append(e["real_r2_sd"])
        nlo.append(np.nan if e["null_lo"] is None else e["null_lo"])
        nhi.append(np.nan if e["null_hi"] is None else e["null_hi"])
    return (
        np.array(xs),
        np.array(mean, dtype=float),
        np.array(sd, dtype=float),
        np.array(nlo, dtype=float),
        np.array(nhi, dtype=float),
    )


def _raw_curve(raw_results: dict, mt: str, layer: int, max_turn: int):
    node = raw_results[mt][str(layer)]
    turns = sorted((int(t) for t in node), key=int)
    xs, real, nlo, nhi = [], [], [], []
    for t in turns:
        if t > max_turn:
            continue
        e = node[str(t)]
        if e["real_r2"] is None:
            continue
        xs.append(t)
        real.append(e["real_r2"])
        nlo.append(np.nan if e.get("null_lo") is None else e["null_lo"])
        nhi.append(np.nan if e.get("null_hi") is None else e["null_hi"])
    return (
        np.array(xs),
        np.array(real, dtype=float),
        np.array(nlo, dtype=float),
        np.array(nhi, dtype=float),
    )


def _plot(payload: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    matched = payload["matched"]
    raw_results = payload["raw_curve"]

    pal = paper_palette(6)
    c_raw, c_a, c_b, c_l14, c_l18 = pal[0], pal[1], pal[3], pal[2], pal[4]
    # raw curve shown out to the deepest turn that still has a matched-null band
    max_turn = 23

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
    model_titles = {"instruct": "Qwen-2.5-7B-Instruct", "pretrained": "Qwen-2.5-7B (base)"}

    for ax, mt in zip(axes, MODELS, strict=True):
        # faint layer-14/18 raw curves (layer context; no bands)
        for L, col in ((14, c_l14), (18, c_l18)):
            xr, rr, _, _ = _raw_curve(raw_results, mt, L, max_turn)
            ax.plot(xr, rr, "-", color=col, lw=1.0, alpha=0.30, zorder=1)
        # raw null band (grey) at layer 19
        xr, rr, rnlo, rnhi = _raw_curve(raw_results, mt, HEADLINE_LAYER, max_turn)
        m = np.isfinite(rnlo) & np.isfinite(rnhi)
        if m.any():
            ax.fill_between(
                xr[m],
                rnlo[m],
                rnhi[m],
                color="0.75",
                alpha=0.30,
                linewidth=0,
                label="shuffle-null band, RAW n (95%)",
                zorder=0,
            )
        # matched null bands (headline layer 19)
        xa, ma_, sa, alo, ahi = _matched_curve(matched, "level_171", mt, HEADLINE_LAYER)
        xb, mb_, sb, blo, bhi = _matched_curve(matched, "level_68", mt, HEADLINE_LAYER)
        m_a = np.isfinite(alo) & np.isfinite(ahi)
        if m_a.any():
            ax.fill_between(
                xa[m_a],
                alo[m_a],
                ahi[m_a],
                color=c_a,
                alpha=0.15,
                linewidth=0,
                label="shuffle-null band, N=171 (95%)",
                zorder=1,
            )
        m_b = np.isfinite(blo) & np.isfinite(bhi)
        if m_b.any():
            ax.fill_between(
                xb[m_b],
                blo[m_b],
                bhi[m_b],
                color=c_b,
                alpha=0.15,
                linewidth=0,
                label="shuffle-null band, N=68 (95%)",
                zorder=1,
            )
        # raw unmatched R2 curve (layer 19)
        ax.plot(
            xr,
            rr,
            "-o",
            color=c_raw,
            ms=4,
            lw=1.8,
            zorder=4,
            label="RAW (unmatched, layer 19)",
        )
        # matched curves with mean +/- sd error bars
        ax.errorbar(
            xa,
            ma_,
            yerr=sa,
            fmt="-s",
            color=c_a,
            ms=5,
            lw=1.8,
            capsize=3,
            zorder=6,
            label="matched N=171 (mean$\\pm$sd, layer 19)",
        )
        ax.errorbar(
            xb,
            mb_,
            yerr=sb,
            fmt="--D",
            color=c_b,
            ms=5,
            lw=1.8,
            capsize=3,
            zorder=5,
            label="matched N=68 (mean$\\pm$sd, layer 19)",
        )
        ax.axhline(0.0, color="0.5", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("user-turn index")
        ax.set_title(model_titles[mt])
        ax.set_xlim(0, max_turn + 1)
    axes[0].set_ylabel(r"held-out $R^2$, context$\to$answer (layer 19)")
    axes[0].legend(fontsize=6.5, loc="upper right", framealpha=0.92, ncol=1)
    # faint-line legend note on the second panel
    axes[1].plot([], [], "-", color=c_l14, lw=1.0, alpha=0.5, label="RAW layer 14 (faint)")
    axes[1].plot([], [], "-", color=c_l18, lw=1.0, alpha=0.5, label="RAW layer 18 (faint)")
    axes[1].legend(fontsize=6.5, loc="upper right", framealpha=0.92)
    fig.suptitle(
        "Matched-N context$\\to$answer map strength vs conversation turn depth",
        fontsize=12,
        y=1.00,
    )
    fig.tight_layout()

    png = OUT_FIG_DIR / f"{FIG_STEM}.png"
    pdf = OUT_FIG_DIR / f"{FIG_STEM}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "figure": f"{FIG_STEM}.png",
        "git_commit": payload["git_commit"],
        "source_results_json": str(OUT_JSON.relative_to(PROJECT_ROOT)),
        "caption": (
            "Held-out R2 of the linear context->answer map (context_k -> "
            "answer_k_t1, offset 0) per user-turn index, controlling for sample "
            "size. Each panel is one model (left instruct, right base). Layer 19: "
            "the RAW unmatched curve (circles) is shown against two matched-N "
            "curves (squares N=171 over t1..t11; diamonds N=68 over t1..t17), each "
            "the mean +/- sd of held-out R2 over 10 subsample draws. Shaded bands "
            "are shuffle-answer permutation nulls (95%): grey at raw n, colored at "
            "each matched n (pooled over 10 subsamples x 100 permutations). Faint "
            "lines are the RAW layer-14/18 curves for layer context. Turn depths "
            "are odd (1,3,5,...)."
        ),
    }
    with open(OUT_FIG_DIR / f"{FIG_STEM}.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[write] {png}")
    print(f"[write] {pdf}")
    print(f"[write] {OUT_FIG_DIR / (FIG_STEM + '.meta.json')}")


if __name__ == "__main__":
    main()
