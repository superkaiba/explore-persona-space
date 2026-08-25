#!/usr/bin/env python3
"""Issue #2202 — plot-5 follow-on: separation margins + worst-tail characterization.

User ask (2026-08-25): characterize (a) the 11 contexts that still fail rank-1
retrieval at the operating point (whitened cosine + CSLS k=10, draw-AVERAGED
targets, the 1,988 kresample-covered rows against the 9,941 pool), and (b) the
worst-predicted contexts under a continuous SEPARATION read, not just the
binary fail indicator.

Separation margin (per query row): true-answer CSLS score minus the best
competitor's CSLS score, ``margin = score[p, p] - max_{j != p} score[p, j]``
(positive iff the row retrieves at rank 1; the failures' banked
``score_margin_true_minus_top1`` is the negative part of the same quantity).

Reuses the exact phase_oppoint conventions of ``issue2202_plot5_redesign.py``
(MZ.load_staged, banked-L Cholesky whitening at lam=0.1, CSLS k=10, the
avgtgt covered-row replacement), reconciliation-gated against the banked
oppoint_ranks.npz. Outputs (ids / ranks / margins / labels only — never
corpus text):

- ``eval_results/issue_2202/plot5_redesign/oppoint_margins.npz`` — per-covered-
  row margins + top-competitor ids at the operating point, and per-holdout-row
  margins under the single-draw whitened-cos+CSLS regime (all 9,941).
- ``eval_results/issue_2202/plot5_redesign/failchar_tail.json`` — label
  composition of the failure set, the near-miss margin tail, and the
  single-draw worst-rank tail, each against its base rates.

0 GPU-h; reads committed eval_results + the read-only staged copies; runs on
the shared VM under the standard thread caps:

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    uv run python scripts/issue2202_plot5_failchar_tail.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2202_failchar as FC  # noqa: E402
import issue2202_metric_zoo as MZ  # noqa: E402
import numpy as np  # noqa: E402

N_NEAR_MISS = 25  # near-miss tail size (smallest positive margins, covered rows)
SINGLE_TAIL_RANK = 10.0  # single-draw worst tail: mid-rank > this, all 9,941 rows
LABEL_KEYS = ("topic", "language", "format", "request_refusal_adjacent", "answer_is_refusal")
RANK_GATE_ROWS = 0  # recomputed oppoint ranks must match the banked npz exactly


def out_dir(args) -> Path:
    return FC.out_eval_dir(args) / "plot5_redesign"


def _wh(x: np.ndarray, ell: np.ndarray, mu_a: np.ndarray) -> np.ndarray:
    from scipy.linalg import solve_triangular

    return solve_triangular(ell, (np.asarray(x, np.float64) - mu_a).T, lower=True).T


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _margins_from_score(score: np.ndarray, rows: np.ndarray, true_cols: np.ndarray):
    """Per-row margin true-minus-best-competitor + the competitor's pool index."""
    sub = score[rows]
    true_s = sub[np.arange(len(rows)), true_cols]
    masked = sub.copy()
    masked[np.arange(len(rows)), true_cols] = -np.inf
    comp_idx = np.argmax(masked, axis=1)
    comp_s = masked[np.arange(len(rows)), comp_idx]
    return true_s - comp_s, comp_idx


def _composition(cis: list[int], labels: dict, fields: dict) -> dict:
    comp: dict[str, dict] = {}
    for key in LABEL_KEYS:
        comp[key] = dict(Counter(str((labels.get(str(ci)) or {}).get(key)) for ci in cis))
    comp["corpus"] = dict(Counter(fields[ci]["corpus"] for ci in cis))
    comp["depth_ge3"] = dict(Counter(str(int(fields[ci]["depth"]) >= 3) for ci in cis))
    return comp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-eval", default=str(PROJECT_ROOT / "eval_results" / "issue_2202"))
    ap.add_argument("--staged", default=MZ.STAGED_DEFAULT)
    ap.add_argument(
        "--labels-1738",
        default="eval_results/issue_1738/judge_labels/labels.json",
        dest="labels_1738",
    )
    ap.add_argument(
        "--ci-fields",
        default=str(PROJECT_ROOT / "data" / "issue_2202" / "ci_fields.json"),
        dest="ci_fields",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import issue1738_characterize as CH
    import issue2202_labels as LB

    staged = Path(args.staged)
    st = MZ.load_staged(staged)
    pred, y16, pci = st["pred"], st["y16"], st["pci"]
    ell, mu_a = st["stats"]["L"], st["stats"]["mu_A"]
    n_pool = y16.shape[0]
    full_idx = np.arange(n_pool)
    k = MZ.K_LOCAL

    kns = argparse.Namespace(
        local_kresample_dir=str(staged / "kresample"), scratch=str(staged / "scratch"), hf_prefix=""
    )
    kci, vres = CH._load_kresample_v(kns, [FC.LAYER])
    draws = vres[:, :, 0, :].astype(np.float64)
    k_draws = vres.shape[1]
    pos_of = {int(c): p for p, c in enumerate(pci.tolist())}
    pos = np.asarray([pos_of[int(c)] for c in kci], dtype=np.int64)

    pwn = _norm(_wh(pred, ell, mu_a))

    # ── operating point: draw-averaged covered pool rows ────────────────────
    avg = (y16[pos] + draws.sum(axis=1)) / (1 + k_draws)
    pool_modw = _wh(y16, ell, mu_a)
    pool_modw[pos] = _wh(avg, ell, mu_a)
    s = pwn @ _norm(pool_modw).T
    score_avg = s - 0.5 * np.partition(s, n_pool - k, axis=0)[n_pool - k :, :].mean(axis=0)[None, :]
    del s
    ranks_avg = MZ.ranks_score_matrix(score_avg, full_idx)[pos]
    banked = np.load(out_dir(args) / "oppoint_ranks.npz")
    assert np.array_equal(np.asarray(banked["ci"], dtype=np.int64), kci)
    n_mismatch = int((np.asarray(banked["rank_avg"]) != ranks_avg).sum())
    if n_mismatch > RANK_GATE_ROWS:
        raise RuntimeError(f"oppoint rank reconciliation FAILED: {n_mismatch} rows differ")
    margins_avg, comp_avg = _margins_from_score(score_avg, pos, pos)
    del score_avg

    # ── single-draw regime, all 9,941 rows ───────────────────────────────────
    pool_w = _norm(_wh(y16, ell, mu_a))
    s = pwn @ pool_w.T
    score_sgl = s - 0.5 * np.partition(s, n_pool - k, axis=0)[n_pool - k :, :].mean(axis=0)[None, :]
    del s
    ranks_sgl = MZ.ranks_score_matrix(score_sgl, full_idx)
    banked_sgl = np.load(out_dir(args) / "whitencos_csls_ranks.npz")
    assert np.array_equal(np.asarray(banked_sgl["ci"], dtype=np.int64), pci)
    n_mismatch = int((np.asarray(banked_sgl["rank_whitencos_csls"]) != ranks_sgl).sum())
    if n_mismatch > RANK_GATE_ROWS:
        raise RuntimeError(f"single-draw rank reconciliation FAILED: {n_mismatch} rows differ")
    margins_sgl, comp_sgl = _margins_from_score(score_sgl, full_idx, full_idx)
    del score_sgl

    tmp = out_dir(args) / "oppoint_margins.tmp.npz"
    np.savez(
        tmp,
        ci_covered=kci,
        pos_covered=pos,
        rank_avg=ranks_avg,
        margin_avg=margins_avg,
        competitor_ci_avg=pci[comp_avg],
        ci_full=pci,
        rank_single=ranks_sgl,
        margin_single=margins_sgl,
        competitor_ci_single=pci[comp_sgl],
    )
    tmp.replace(out_dir(args) / "oppoint_margins.npz")

    # ── tails + composition ──────────────────────────────────────────────────
    labels = LB.load_labels_1738(args)
    fields = {int(kk): v for kk, v in LB.resolve_ci_fields(args).items()}

    fail_mask = ranks_avg > 1.0
    pass_margins = margins_avg[~fail_mask]
    near_order = np.argsort(margins_avg)
    near_idx = [i for i in near_order if not fail_mask[i]][:N_NEAR_MISS]
    sgl_tail_idx = np.flatnonzero(ranks_sgl > SINGLE_TAIL_RANK)

    def rows_of(idxs, ci_arr, rank_arr, marg_arr, comp_arr):
        return [
            {
                "ci": int(ci_arr[i]),
                "rank": float(rank_arr[i]),
                "margin": float(marg_arr[i]),
                "competitor_ci": int(comp_arr[i]),
                "labels_1738": {
                    kk: (labels.get(str(int(ci_arr[i]))) or {}).get(kk) for kk in LABEL_KEYS
                },
                "corpus": fields[int(ci_arr[i])]["corpus"],
                "depth": fields[int(ci_arr[i])]["depth"],
            }
            for i in idxs
        ]

    covered_cis = [int(c) for c in kci]
    doc = {
        "convention": (
            "margins/ranks under whitened cosine + CSLS (k=10); operating point = "
            "draw-AVERAGED covered pool rows (avgtgt_completion), 1,988 covered rows "
            "vs the 9,941 pool; single-draw regime = all 9,941 rows; margin = true-answer "
            "CSLS score minus best competitor (positive iff rank 1)"
        ),
        "margin_distribution_covered_avg": {
            "n_pass": int((~fail_mask).sum()),
            "pass_margin_percentiles": {
                str(q): float(np.percentile(pass_margins, q)) for q in (1, 5, 25, 50, 75, 95)
            },
            "n_fail": int(fail_mask.sum()),
            "fail_margins": [float(m) for m in np.sort(margins_avg[fail_mask])],
        },
        "failures_avg": rows_of(
            np.flatnonzero(fail_mask), kci, ranks_avg, margins_avg, pci[comp_avg]
        ),
        "near_misses_avg": rows_of(near_idx, kci, ranks_avg, margins_avg, pci[comp_avg]),
        "single_draw_tail": {
            "rank_threshold": SINGLE_TAIL_RANK,
            "n": int(len(sgl_tail_idx)),
            "rows": rows_of(sgl_tail_idx, pci, ranks_sgl, margins_sgl, pci[comp_sgl]),
        },
        "composition": {
            "covered_base_rates_n1988": _composition(covered_cis, labels, fields),
            "failures_avg_n11": _composition(
                [int(kci[i]) for i in np.flatnonzero(fail_mask)], labels, fields
            ),
            "near_misses_n25": _composition([int(kci[i]) for i in near_idx], labels, fields),
            "holdout_base_rates_n9941": _composition([int(c) for c in pci], labels, fields),
            "single_draw_tail": _composition([int(pci[i]) for i in sgl_tail_idx], labels, fields),
        },
        "meta": FC.meta_block(),
    }
    FC.atomic_json(out_dir(args) / "failchar_tail.json", doc)
    print(
        f"[failchar-tail] done: {int(fail_mask.sum())} avg failures, "
        f"{len(near_idx)} near-misses, single-draw tail (rank>{SINGLE_TAIL_RANK:g}) "
        f"n={len(sgl_tail_idx)}; pass-margin median "
        f"{float(np.median(pass_margins)):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
