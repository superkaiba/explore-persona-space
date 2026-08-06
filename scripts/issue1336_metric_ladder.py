#!/usr/bin/env python
"""Issue #1336 — Phase LAD: the 9-tier metric-ladder battery (plan v13 §4).

Per (pair source s -> target t, eval surface, frozen layer), on the per-pair
prompt-id INTERSECTION rows with the SHARED seed-0 fold partition, every
correction parameter is fit on TRAIN folds only and every tier is evaluated
on the SAME held-out TEST folds (docs/mapping_similarity_metrics.md,
implemented per the plan §4 pseudocode):

  W_s, b_s   source within map, per-fold (v2 recipe: 23-pt grid
             cm.LAMBDAS_23, inner-group-CV n_inner=2, primal d-space route
             at n_train > d — the Unit-B cores)
  T0 direct       y = W_s x_t + b_s
  T1 ctx offset   dx = mean_tr(x_t) - mean_tr(x_s)  (CLOUDS only, no pairs)
  T2 ans offset   dy = mean_tr(y_t) - mean_tr(y_s)  (clouds only)
  T3 bias         b* refit on train pairs
  T4 scalar       alpha + b* by LS on train pairs, W_s frozen
  T5 rotation     fitted orthogonal Procrustes R (SVD of the train
                  cross-covariance of centered W_s x_t vs y_t; R^T R = I —
                  DIRECTION-AWARE, never a spectrum cosine; tr(R)/d +
                  aligned-cosine descriptives reported)
  T6 reparam ctx  A_ctx_rev: x_t -> x_s ridge on CONTEXT pairs only
  T7 reparam ans  A_ans: y_s -> y_t ridge on ANSWER pairs only, applied to
                  the source-map outputs (the plan's stated doc deviation:
                  forward A_ans, never B^-1)
  T8 reparam both y = A_ans(W_s(A_ctx_rev x_t)) + b*  == the parent
                  comp_samefn_b2i construction VERBATIM (pinned by
                  tests/test_issue1336_metric_ladder.py against
                  issue825_map_alignment at a forced single lambda)
  ceiling         R^2_within(t) RECOMPUTED on the intersection rows/folds
                  (binding critic note: never the full-set Phase-FIT number)
  repswap         x_s -> y_t ceiling (information-present control)

Controls: 20 shuffled-pairing nulls PER TIER (the capacity control — per
draw the target rows y_t are conversation-permuted, every y_t-consuming
correction is REFIT through the CACHED Y-independent bases — never a
per-draw eigh/SVD refit loop; T5's rotation R stays at the observed fit,
its intercept refit per draw: a d x d Procrustes SVD per draw is a
~22k-SVD battery the §9 row does not book — recorded per battery);
rep-swap ceilings; identity+learned-bias baseline + kNN retrieval
(analysis/mapping_baselines) for the within map, the tier-8 composition and
the A_ctx_rev / A_ans alignment maps at the ``--full-tier-layers`` set.

Statistics: paired prompt-level bootstrap (1,000 shared draws, seed
5000 + v2 surface index — cm.v2_surface_index) on per-tier gaps
(R^2_within - R^2_T), Delta_k = within - tier8, and tier-adjacent
increments, executed as gather-reduce GEMMs over the captured prediction
matrices (the vendored round-5 machinery in issue1336_ladder_alignment).
Both scales run: raw (primary candidate) + the held-out cross-fitted
per-dim affine recalibration applied to BOTH ARMS of every gap
(scale-consistency: never a raw tier against a recal ceiling). The
sufficient-tier rule (min{T: within - R^2_T <= band}, band =
0.0201 * ex_v2) is applied per battery on both scales with the per-draw
sufficient-tier distribution persisted.

Outputs: eval_results/issue_1336/metric_ladder/pair_<m0>__<m1>_<fmt>_<corpus>.json
(56 files at full scope) + fp16 preds at --preds-dir (within/t8 + recal at
every frozen layer — the §3 registered row-coverage arrays — plus ALL tiers
at the --full-tier-layers set and the per-draw per-tier R^2 matrices for
cheap re-reduction), manifest metric_ladder_manifest.json.

Compute notes for the dispatcher pilot (Unit D): no cross-pair W_s prep
cache is attempted when intersection rows differ (correctness over cache —
the PrepCache below reuses a source prep ONLY on an exact
(model, role, layer, fold, rows-sha) match); ladder fits use the FIXED
23-pt grid (the adaptive edge rule is a Phase-FIT per-cell procedure; the
ladder reports per-map edge fractions so an edge pathology stays visible).
Full-tier preds at all 4 frozen layers cost ~8 GB/battery fp16 at n~15k —
pass ``--full-tier-layers <headline>`` after the Phase-FIT headline rule to
bound Phase-LAD persistence to the registered arrays + one full layer.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import issue825_fit_cells as fc  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402
import issue1336_extract_turnstore as et  # noqa: E402
import issue1336_fit_cells as f36  # noqa: E402
import issue1336_ladder_alignment as la  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402
from explore_persona_space.experiments.issue_1336 import recal as rc  # noqa: E402

N_FOLDS = cm.N_FOLDS
FIT_SEED = cm.FIT_SEED
TIER_NAMES = tuple(f"t{k}" for k in range(9))
# Order of rows in every per-draw matrix (nulls + bootstrap re-reductions).
DRAWS_ORDER = ("within", *TIER_NAMES)
KNN_KS = (1, 5)
KNN_SUBSAMPLE_SEED = 1336


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", default=None, help="m0:m1 (e.g. base:rlvr)")
    ap.add_argument("--pairs", default=None, help="comma list of m0:m1 pairs (one surface)")
    ap.add_argument("--corpus", default=None, choices=tuple(cm.V2_CORPORA))
    ap.add_argument("--format", default=None, choices=("chat", "naturalistic"))
    ap.add_argument("--turnstore-dir", type=Path, default=None)
    ap.add_argument(
        "--wave1-turnstore-dir",
        type=Path,
        default=None,
        help="wave-1 stems dir for the concat loader (default: --turnstore-dir)",
    )
    ap.add_argument(
        "--gen-root",
        type=Path,
        default=None,
        help="gen outputs root (wave-1 text-sha join source for the concat loader)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1336"))
    ap.add_argument("--preds-dir", type=Path, default=None)
    ap.add_argument("--frozen-layers", default=None, help="comma ints (default: registry set)")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--null-draws", type=int, default=None)
    ap.add_argument(
        "--s-qwen-v2",
        type=float,
        default=None,
        help="G0'(c) v2-recipe Qwen anchor (sets ex_v2 / the sufficient-tier band)",
    )
    ap.add_argument(
        "--bars-json",
        type=Path,
        default=None,
        help="JSON carrying s_qwen_v2 (the G0' gate output) — alternative to --s-qwen-v2",
    )
    ap.add_argument(
        "--primary-scale",
        choices=("raw", "recal"),
        default="raw",
        help="lattice scale marker (the §3 health-gate-H outcome; both scales always emitted)",
    )
    ap.add_argument(
        "--full-tier-layers",
        default="all",
        help="layers persisting ALL tiers' preds + baselines: 'all' (frozen set), "
        "'none', or comma ints (the dispatcher passes the headline layer)",
    )
    ap.add_argument("--knn-max-rows", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selfcheck", action="store_true", help="tier/bootstrap equivalence gates")
    # --- cluster delta-Q battery (plan v15 §4 div. 11 + Phase LAD_cluster) ---
    ap.add_argument(
        "--cluster-delta-q",
        action="store_true",
        help="run the per-transition cluster delta-Q battery on the Phase FIT_pool "
        "outputs (plan v15 Phase LAD_cluster); mutually exclusive with --pair/--pairs",
    )
    ap.add_argument(
        "--pooled-preds-root",
        type=Path,
        default=None,
        help="Phase FIT_pool preds root holding {on,off}-policy/preds_*.npz + rows_*.json "
        "(default: data/issue_1336/preds_pooled_v3[_smoke])",
    )
    ap.add_argument(
        "--offpolicy-root",
        type=Path,
        default=None,
        help="root holding the Phase EXT_off turnstore_offpolicy_<i>_chat_<j>[_smoke] trees "
        "(default: data/issue_1336 — name parity with issue1336_fit_cells.py)",
    )
    ap.add_argument(
        "--arms",
        default="on,off",
        help="comma subset of {on,off} to run the delta-Q battery on (on = primary; "
        "off = fixed-answer-text interpretation-guard companion)",
    )
    ap.add_argument(
        "--transitions",
        default=None,
        help="comma list of i:j checkpoint pairs (default: the 4 registered adjacent "
        "transitions base:sft,sft:dpo,dpo:rlvr,dpo:rlvr_long)",
    )
    ap.add_argument(
        "--headline-layer",
        type=int,
        default=None,
        help="override the pooled stage-symmetric headline-layer rule (dispatcher/smoke seam)",
    )
    ap.add_argument(
        "--perdraw-dir",
        type=Path,
        default=Path("analysis_tensors/delta_q_perdraw"),
        help="destination for the per-draw x per-cluster permutation matrices",
    )
    ap.add_argument(
        "--perm-draws",
        type=int,
        default=None,
        help="permutation draws per (transition, arm) (default 1000; 50 under --smoke)",
    )
    ap.add_argument(
        "--perm-chunk",
        type=int,
        default=250,
        help="draws per vectorized permutation chunk (memory bound: chunk x n_prompts)",
    )
    # --- pooled-pair transfer tier reads (plan v15 §4 div. 9 + Phase LAD_pool) ---
    ap.add_argument(
        "--pooled-pair",
        default=None,
        help="i:j ordered checkpoint pair — run the Phase LAD_pool pooled-split transfer "
        "tier read (own/T0/T6/T8 on the pooled 20%% test side, sliced per corpus); "
        "mutually exclusive with --pair/--pairs/--cluster-delta-q",
    )
    ap.add_argument(
        "--arm",
        choices=("on", "off"),
        default=None,
        help="--pooled-pair arm: on = each checkpoint captured on its OWN text (diagonal "
        "v2 turnstores); off = fixed-answer-text guard (the SOURCE checkpoint captured on "
        "the TARGET's text via the Phase EXT_off off-diagonal tree)",
    )
    ap.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Phase C_pool split manifest (default: data/issue_1336/pooled_split_v3"
        "[_smoke]/split_manifest.json — name parity with issue1336_fit_cells.py)",
    )
    return ap.parse_args()


def _metadata(seed: int, n: int) -> dict:
    return {
        "git_commit": fc._git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "n": int(n),
        "script": "scripts/issue1336_metric_ladder.py",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[ladder1336] wrote {path}")


def resolve_bars(s_qwen_v2: float | None, bars_json: Path | None) -> dict:
    """The exchange-rate-scaled v2 bars (cm.v2_bars) from either input."""
    if s_qwen_v2 is None:
        assert bars_json is not None, "--s-qwen-v2 or --bars-json is required (band = 0.0201*ex_v2)"
        payload = json.loads(bars_json.read_text())
        assert "s_qwen_v2" in payload, f"{bars_json} lacks the s_qwen_v2 key (G0'(c) output)"
        s_qwen_v2 = float(payload["s_qwen_v2"])
    return cm.v2_bars(float(s_qwen_v2))


# ===========================================================================
# Factored v2-recipe ridge (Y-independent prep per SOURCE; predict at ANY
# X_eval — the composition tiers evaluate W_s / A_ans at transformed inputs).
# Primal (d-space) route at n_train > d (the Unit-B regime switch,
# fc._primal_eig / _primal_cache_pieces — gate G0'(b) pins Gram equality);
# lambda by inner-group-CV over the caller grid (cm.LAMBDAS_23, n_inner=2).
# ===========================================================================
def _v2_prep(X_tr: torch.Tensor, *, inner_seed: int, n_inner: int) -> dict:
    """Y-independent pieces for one source: eigenbasis + inner-CV caches.

    Returns {"route", "w", "Q", ...}: prediction is
    ``(pieces * 1/(w+lam)) @ (Q^T Yc) + ymu`` with ``pieces`` from
    :func:`_v2_eval_pieces` — identical algebra on both routes
    (fc._primal_cache_pieces docstring).
    """
    dev = X_tr.device
    n_tr, d = int(X_tr.shape[0]), int(X_tr.shape[1])
    if not fc.FORCE_GRAM and n_tr > d:
        pe = fc._primal_eig(X_tr, dev)
        prep = {
            "route": "primal",
            "pe": pe,
            "w": pe["s"],
            "Q": pe["TU"] / torch.sqrt(pe["s"]),
            "ntr": n_tr,
            "d": d,
        }
    else:
        xmu = X_tr.mean(0)
        xsd = X_tr.std(0) + 1e-9
        Xn = (X_tr - xmu) / xsd
        w, V = fc._eigh_robust(Xn @ Xn.T)
        prep = {
            "route": "gram",
            "xmu": xmu,
            "xsd": xsd,
            "Xn": Xn,
            "w": torch.clamp(w, min=0.0),
            "Q": V,
            "ntr": n_tr,
            "d": d,
        }
    prep["inner"] = fc._prep_inner_lambda(X_tr, np.arange(n_tr), n_inner, inner_seed)
    if prep["inner"] is None:
        print("[ladder1336] WARN: <2 usable inner folds — GCV fallback for this prep")
    return prep


def _v2_eval_pieces(prep: dict, X_eval: torch.Tensor) -> torch.Tensor:
    """(n_eval, k) evaluation pieces for arbitrary X_eval (cacheable per fold)."""
    if prep["route"] == "primal":
        _, pieces = fc._primal_cache_pieces(prep["pe"], X_eval, X_eval.device)
        return pieces
    Xev_n = (X_eval - prep["xmu"]) / prep["xsd"]
    return (Xev_n @ prep["Xn"].T) @ prep["Q"]


def _gcv_lambda(prep: dict, Ytr_c: torch.Tensor, QtY: torch.Tensor, grid: np.ndarray) -> float:
    """GCV fallback (inner caches unbuildable) — mirrors ma._select_lambda
    with the fit825 dof cap + the #1887 unguarded-GCV refusal."""
    fc._refuse_unguarded_gcv(
        ntr=prep["ntr"],
        d=prep["d"],
        cap=fc.GCV_DOF_CAP,
        legacy_ok=fc.LEGACY_UNGUARDED_GCV,
        where="issue1336_metric_ladder._gcv_lambda",
    )
    w, ntr = prep["w"], prep["ntr"]
    sq = (QtY**2).sum(1)
    tot = float((Ytr_c**2).sum())
    best_lam, best_gcv = float(grid[0]), float("inf")
    for lam in grid:
        filt = w / (w + lam)
        dof = float(filt.sum())
        if fc.GCV_DOF_CAP is not None and dof > fc.GCV_DOF_CAP * ntr:
            continue
        rss = tot - float(((2 * filt - filt**2) * sq).sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    return best_lam


def _v2_yfit(prep: dict, Y_tr: torch.Tensor, grid: np.ndarray) -> dict:
    """Y-dependent map pieces (intercept, rotated targets, selected lambda).

    Together with the prep this IS the frozen fitted map: T0-T8 all evaluate
    the SAME (prep, yfit) pair at different inputs, so W_s stays frozen
    across tiers by construction.
    """
    ymu = Y_tr.mean(0)
    Ytr_c = Y_tr - ymu
    QtY = prep["Q"].T @ Ytr_c
    if prep["inner"]:
        rss = fc._inner_cv_rss_curve(prep["inner"], Y_tr, lams=grid)
        lam = float(grid[int(torch.argmin(rss))])
        selector = "inner-group-cv"
    else:
        lam = _gcv_lambda(prep, Ytr_c, QtY, grid)
        selector = "gcv-fallback"
    return {"ymu": ymu, "QtY": QtY, "lam": lam, "selector": selector}


def _v2_predict(
    prep: dict,
    yfit: dict,
    X_eval: torch.Tensor | None = None,
    *,
    pieces: torch.Tensor | None = None,
) -> torch.Tensor:
    """Predict at X_eval (or precomputed eval pieces) under the fitted map."""
    pe = pieces if pieces is not None else _v2_eval_pieces(prep, X_eval)
    filt = 1.0 / (prep["w"] + yfit["lam"])
    return (pe * filt) @ yfit["QtY"] + yfit["ymu"]


class PrepCache:
    """Tiny keyed cache for Y-independent preps across pairs sharing a source.

    Plan §4 Phase LAD: W_s is cacheable per (source, surface, layer, fold)
    ONLY when the intersection row set is shared across pairs — the key
    therefore includes the rows-sha, so a differing intersection MISSES and
    refits (correctness over cache). FIFO-bounded (device tensors are big).
    """

    def __init__(self, capacity: int = 6):
        self.capacity = int(capacity)
        self._store: OrderedDict[tuple, dict] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple, builder) -> dict:
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        prep = builder()
        self._store[key] = prep
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return prep

    def clear(self) -> None:
        self._store.clear()


def sufficient_tier(r2_within: float, tier_r2: list[float], band: float) -> int | None:
    """min{T : R^2_within - R^2_T <= band}; None encodes 'none <= 8' (§3)."""
    for t, r in enumerate(tier_r2):
        if float(r2_within) - float(r) <= band:
            return t
    return None


def _sufficient_tier_draws(draws: np.ndarray, band: float) -> dict:
    """Per-draw sufficient-tier histogram from the (10, n_boot) draws matrix
    (row 0 = within, rows 1..9 = t0..t8 — DRAWS_ORDER)."""
    within = draws[0]
    hist: dict[str, int] = {str(t): 0 for t in range(9)}
    hist["none"] = 0
    for b in range(draws.shape[1]):
        t = sufficient_tier(within[b], list(draws[1:, b]), band)
        hist["none" if t is None else str(t)] += 1
    return hist


def _knn_block(pred: np.ndarray, true: np.ndarray, keep: np.ndarray) -> dict:
    """euclidean + cosine retrieval at k in KNN_KS on the seeded row subsample."""
    out = {}
    for metric in ("euclidean", "cosine"):
        out[metric] = knn_retrieval(pred[keep], true[keep], ks=KNN_KS, metric=metric)
    return out


def _lambda_edge_stats(lams: list[float], grid: np.ndarray) -> dict:
    """Selected-lambda edge fractions per map (audit-only; no edge extension
    in the ladder — recorded so a Phase-FIT-class edge pathology stays
    visible in the battery JSON)."""
    arr = np.asarray(lams, dtype=np.float64)
    n = int(arr.size)
    n_low = int(np.sum(arr == grid[0]))
    n_high = int(np.sum(arr == grid[-1]))
    return {
        "n_selected": n,
        "n_at_low_edge": n_low,
        "n_at_high_edge": n_high,
        "frac_at_low_edge": (n_low / n) if n else None,
        "frac_at_high_edge": (n_high / n) if n else None,
        "selected": [float(v) for v in arr],
    }


# ===========================================================================
# The battery core (array-level; file I/O lives in run_pair)
# ===========================================================================
def _fold_observed(
    Xs_l: torch.Tensor,
    Ys_l: torch.Tensor,
    Xt_l: torch.Tensor,
    Yt_l: torch.Tensor,
    tr: torch.Tensor,
    te: torch.Tensor,
    grid: np.ndarray,
    preps: dict,
) -> tuple[dict, dict]:
    """One fold's observed tier predictions (test rows) + null-loop caches.

    Returns (te_preds{name: (n_te, d)}, aux) — aux carries everything the
    per-draw null loop reuses without any refit of Y-independent pieces.
    """
    prep_s, prep_t, prep_ys = preps["s"], preps["t"], preps["ys"]
    fit_ws = _v2_yfit(prep_s, Ys_l[tr], grid)
    fit_within = _v2_yfit(prep_t, Yt_l[tr], grid)
    fit_actx = _v2_yfit(prep_t, Xs_l[tr], grid)  # A_ctx_rev: x_t -> x_s
    fit_aans = _v2_yfit(prep_ys, Yt_l[tr], grid)  # A_ans: y_s -> y_t
    fit_repswap = _v2_yfit(prep_s, Yt_l[tr], grid)  # rep-swap ceiling x_s -> y_t

    pe_s_xt_te = _v2_eval_pieces(prep_s, Xt_l[te])
    pe_t_xt_te = _v2_eval_pieces(prep_t, Xt_l[te])
    p0_te = _v2_predict(prep_s, fit_ws, pieces=pe_s_xt_te)
    p0_tr = _v2_predict(prep_s, fit_ws, Xt_l[tr])  # W_s applied to x_t TRAIN rows
    dx = Xt_l[tr].mean(0) - Xs_l[tr].mean(0)
    t1_te = _v2_predict(prep_s, fit_ws, Xt_l[te] - dx)
    dy = Yt_l[tr].mean(0) - Ys_l[tr].mean(0)
    yt_tr_mu = Yt_l[tr].mean(0)
    p0_tr_mu = p0_tr.mean(0)
    t2_te = p0_te + dy
    t3_te = p0_te + (yt_tr_mu - p0_tr_mu)
    p0c_tr = p0_tr - p0_tr_mu
    ytc_tr = Yt_l[tr] - yt_tr_mu
    p0c_ss = float((p0c_tr**2).sum())
    alpha = float((p0c_tr * ytc_tr).sum()) / (p0c_ss + 1e-12)
    t4_te = alpha * (p0_te - p0_tr_mu) + yt_tr_mu
    orth = ma._orth_fit(p0_tr, Yt_l[tr])
    t5_te = ma._orth_predict(orth, p0_te, reverse=False, scale=False)
    xhat_tr = _v2_predict(prep_t, fit_actx, Xt_l[tr])
    xhat_te = _v2_predict(prep_t, fit_actx, Xt_l[te])
    raw6_tr = _v2_predict(prep_s, fit_ws, xhat_tr)
    raw6_te = _v2_predict(prep_s, fit_ws, xhat_te)
    raw6_tr_mu = raw6_tr.mean(0)
    t6_te = raw6_te + (yt_tr_mu - raw6_tr_mu)
    pe_ys_p0_te = _v2_eval_pieces(prep_ys, p0_te)
    pe_ys_raw8_te = _v2_eval_pieces(prep_ys, raw6_te)
    t7_te = _v2_predict(prep_ys, fit_aans, pieces=pe_ys_p0_te)
    t8_te = _v2_predict(prep_ys, fit_aans, pieces=pe_ys_raw8_te)
    within_te = _v2_predict(prep_t, fit_within, pieces=pe_t_xt_te)
    repswap_te = _v2_predict(prep_s, fit_repswap, Xs_l[te])
    aans_own_te = _v2_predict(prep_ys, fit_aans, Ys_l[te])  # A_ans own held-out read

    te_preds = {
        "within": within_te,
        "t0": p0_te,
        "t1": t1_te,
        "t2": t2_te,
        "t3": t3_te,
        "t4": t4_te,
        "t5": t5_te,
        "t6": t6_te,
        "t7": t7_te,
        "t8": t8_te,
        "repswap": repswap_te,
        "actx": xhat_te,
        "aans_own": aans_own_te,
    }
    ssum = float(torch.clamp(torch.as_tensor(orth["s_fwd"] * orth["s_rev"]), min=0.0).sqrt())
    aux = {
        "pe_t_xt_te": pe_t_xt_te,
        "pe_ys_p0_te": pe_ys_p0_te,
        "pe_ys_raw8_te": pe_ys_raw8_te,
        "p0_te": p0_te,
        "t1_te": t1_te,
        "raw6_te": raw6_te,
        "p0c_tr": p0c_tr,
        "p0c_ss": p0c_ss,
        "p0_tr_mu": p0_tr_mu,
        "raw6_tr_mu": raw6_tr_mu,
        "ys_tr_mu": Ys_l[tr].mean(0),
        "orth_Amu": orth["Amu"],
        "orth_R": orth["R"],
        "selected_lams": {
            "W_s": fit_ws["lam"],
            "within": fit_within["lam"],
            "A_ctx_rev": fit_actx["lam"],
            "A_ans": fit_aans["lam"],
            "repswap": fit_repswap["lam"],
        },
        "selectors": {
            "W_s": fit_ws["selector"],
            "within": fit_within["selector"],
            "A_ctx_rev": fit_actx["selector"],
            "A_ans": fit_aans["selector"],
        },
        "procrustes": {
            "trace_R_over_d": float(torch.diagonal(orth["R"]).mean()),
            "aligned_cos": ssum,
            "s_fwd": float(orth["s_fwd"]),
        },
        "preps": preps,
        "alpha": alpha,
    }
    return te_preds, aux


def _fold_null_r2_contrib(
    aux: dict,
    Ytp: torch.Tensor,
    tr: torch.Tensor,
    te: torch.Tensor,
    grid: np.ndarray,
) -> tuple[np.ndarray, float]:
    """One (fold, draw) null contribution: per-DRAWS_ORDER ss_res + ss_tot.

    The draw's permuted targets Ytp re-fit every CHEAP y_t-consuming
    correction through the cached Y-independent bases (within + A_ans
    yfits are one GEMM + one lambda curve each); T0/T1 predictions are
    y_t-free and stay fixed; T5 keeps the observed rotation R (intercept
    refit only — see the module docstring).
    """
    preps = aux["preps"]
    Ytp_tr, Ytp_te = Ytp[tr], Ytp[te]
    ytp_mu = Ytp_tr.mean(0)
    fit_within_b = _v2_yfit(preps["t"], Ytp_tr, grid)
    fit_aans_b = _v2_yfit(preps["ys"], Ytp_tr, grid)
    within_b = _v2_predict(preps["t"], fit_within_b, pieces=aux["pe_t_xt_te"])
    t7_b = _v2_predict(preps["ys"], fit_aans_b, pieces=aux["pe_ys_p0_te"])
    t8_b = _v2_predict(preps["ys"], fit_aans_b, pieces=aux["pe_ys_raw8_te"])
    dy_b = ytp_mu - aux["ys_tr_mu"]
    t2_b = aux["p0_te"] + dy_b
    t3_b = aux["p0_te"] + (ytp_mu - aux["p0_tr_mu"])
    ytpc_tr = Ytp_tr - ytp_mu
    alpha_b = float((aux["p0c_tr"] * ytpc_tr).sum()) / (aux["p0c_ss"] + 1e-12)
    t4_b = alpha_b * (aux["p0_te"] - aux["p0_tr_mu"]) + ytp_mu
    t5_b = (aux["p0_te"] - aux["orth_Amu"]) @ aux["orth_R"] + ytp_mu
    t6_b = aux["raw6_te"] + (ytp_mu - aux["raw6_tr_mu"])
    preds_b = {
        "within": within_b,
        "t0": aux["p0_te"],
        "t1": aux["t1_te"],
        "t2": t2_b,
        "t3": t3_b,
        "t4": t4_b,
        "t5": t5_b,
        "t6": t6_b,
        "t7": t7_b,
        "t8": t8_b,
    }
    ss_res = np.zeros(len(DRAWS_ORDER))
    for i, name in enumerate(DRAWS_ORDER):
        ss_res[i] = float(((Ytp_te - preds_b[name]) ** 2).sum())
    ss_tot = float(((Ytp_te - Ytp_te.mean(0)) ** 2).sum())
    return ss_res, ss_tot


def run_battery_arrays(
    Xs: np.ndarray,
    Ys: np.ndarray,
    Xt: np.ndarray,
    Yt: np.ndarray,
    conv_ids: np.ndarray,
    *,
    frozen_layers: tuple[int, ...],
    n_folds: int = N_FOLDS,
    seed: int = FIT_SEED,
    null_draws: int,
    n_boot: int,
    boot_seed: int,
    grid: np.ndarray,
    band: float,
    knn_max_rows: int = 2000,
    full_tier_layers: tuple[int, ...] = (),
    n_inner: int = cm.N_INNER_LAMBDA_FOLDS_V2,
    prep_cache: PrepCache | None = None,
    cache_tags: dict[str, tuple] | None = None,
) -> tuple[dict, dict]:
    """The full per-(pair, surface) battery on row-ALIGNED intersection arrays.

    Xs/Ys/Xt/Yt: (n, L, D) fp arrays over the SAME prompt rows (source and
    target models). Returns (payload, preds_store): payload carries the
    per-layer tier profile + statistics; preds_store the fp16 arrays +
    per-draw matrices for the npz.

    ``cache_tags`` maps each prep KIND to the identity tag of the MODEL whose
    tensors that prep is built over: "s" (source contexts Xs) and "ys"
    (source answers Ys) tag m0; "t" (TARGET contexts Xt) tags m1. A tag that
    omitted m1 collided the "t" prep across pairs sharing a source
    (base->dpo vs base->rlvr) whenever the shared cache retained entries.
    """
    fc._validate_lambda_grid(np.asarray(grid))
    grid = np.asarray(grid, dtype=np.float64)
    n = int(Xs.shape[0])
    assert Xs.shape == Ys.shape == Xt.shape == Yt.shape, (Xs.shape, Ys.shape, Xt.shape, Yt.shape)
    assert len(conv_ids) == n, (len(conv_ids), n)
    folds = fc._cv_folds(conv_ids, n_folds, seed)
    dev = fc._fit_device()
    dtype = torch.float64
    rng = np.random.default_rng(seed + 1)
    perms = [rng.permutation(n) for _ in range(null_draws)]
    idx_matrix = la.draw_index_matrix(n, n_boot, seed=boot_seed)
    w_boot = la.counts_from_indices(idx_matrix, n)
    ids_sha = hashlib.sha256(",".join(str(c) for c in conv_ids).encode()).hexdigest()[:16]
    cache = prep_cache if prep_cache is not None else PrepCache(capacity=0)

    per_layer: dict[str, dict] = {}
    preds_store: dict[str, np.ndarray] = {
        "conv_ids": np.asarray([str(c) for c in conv_ids]),
        "folds": folds.astype(np.int64),
    }
    for li in frozen_layers:
        assert li < Xs.shape[1], f"frozen layer {li} out of range ({Xs.shape[1]} layers)"
        Xs_l = torch.as_tensor(Xs[:, li, :], dtype=dtype).to(dev)
        Ys_l = torch.as_tensor(Ys[:, li, :], dtype=dtype).to(dev)
        Xt_l = torch.as_tensor(Xt[:, li, :], dtype=dtype).to(dev)
        Yt_l = torch.as_tensor(Yt[:, li, :], dtype=dtype).to(dev)
        d = int(Xs_l.shape[1])
        capture_names = ("within", *TIER_NAMES, "repswap", "actx", "aans_own")
        captured = {name: np.zeros((n, d), dtype=np.float32) for name in capture_names}
        captured["id_within"] = np.zeros((n, d), dtype=np.float32)
        captured["id_actx"] = np.zeros((n, d), dtype=np.float32)
        captured["id_aans"] = np.zeros((n, d), dtype=np.float32)
        fitted = np.zeros(n, dtype=bool)
        ss_res_obs: dict[str, float] = dict.fromkeys(capture_names, 0.0)
        ss_tot_y = 0.0
        ss_tot_xs = 0.0
        ss_null_res = np.zeros((null_draws, len(DRAWS_ORDER)))
        ss_null_tot = np.zeros(null_draws)
        lam_log: dict[str, list[float]] = {}
        selector_log: dict[str, list[str]] = {}
        procrustes_rows: list[dict] = []
        for k in range(n_folds):
            tr_np = folds != k
            te_np = folds == k
            if te_np.sum() == 0 or tr_np.sum() < 3:
                continue
            tr = torch.as_tensor(tr_np)
            te = torch.as_tensor(te_np)
            inner_seed = seed + 4242 + k

            def _mk(x, s=inner_seed):
                return _v2_prep(x, inner_seed=s, n_inner=n_inner)

            tags = cache_tags or {}
            preps = {
                "s": cache.get(
                    (*tags.get("s", ()), "s", li, k, ids_sha), lambda x=Xs_l: _mk(x[tr])
                ),
                "t": cache.get(
                    (*tags.get("t", ()), "t", li, k, ids_sha), lambda x=Xt_l: _mk(x[tr])
                ),
                "ys": cache.get(
                    (*tags.get("ys", ()), "ys", li, k, ids_sha), lambda x=Ys_l: _mk(x[tr])
                ),
            }
            te_preds, aux = _fold_observed(Xs_l, Ys_l, Xt_l, Yt_l, tr, te, grid, preps)
            yt_te = Yt_l[te]
            xs_te = Xs_l[te]
            for name, pred in te_preds.items():
                true = xs_te if name == "actx" else yt_te
                ss_res_obs[name] += float(((true - pred) ** 2).sum())
                captured[name][te_np] = pred.float().cpu().numpy()
            ss_tot_y += float(((yt_te - yt_te.mean(0)) ** 2).sum())
            ss_tot_xs += float(((xs_te - xs_te.mean(0)) ** 2).sum())
            # Identity+learned-bias baselines (canonical helper; OOF per fold).
            captured["id_within"][te_np] = identity_bias_predict(
                Xt[tr_np, li, :], Yt[tr_np, li, :], Xt[te_np, li, :]
            ).astype(np.float32)
            captured["id_actx"][te_np] = identity_bias_predict(
                Xt[tr_np, li, :], Xs[tr_np, li, :], Xt[te_np, li, :]
            ).astype(np.float32)
            captured["id_aans"][te_np] = identity_bias_predict(
                Ys[tr_np, li, :], Yt[tr_np, li, :], Ys[te_np, li, :]
            ).astype(np.float32)
            for name, lam in aux["selected_lams"].items():
                lam_log.setdefault(name, []).append(float(lam))
            for name, sel in aux["selectors"].items():
                selector_log.setdefault(name, []).append(sel)
            procrustes_rows.append(aux["procrustes"])
            fitted[te_np] = True
            # Shuffled-pairing nulls (conversation == row here: single-turn
            # rows, one per prompt id — the fit-phase permutation unit).
            for b, perm in enumerate(perms):
                perm_t = torch.as_tensor(perm, dtype=torch.long, device=dev)
                res_b, tot_b = _fold_null_r2_contrib(
                    aux, Yt_l.index_select(0, perm_t), tr, te, grid
                )
                ss_null_res[b] += res_b
                ss_null_tot[b] += tot_b
            del te_preds, aux, preps
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        assert fitted.all(), f"unfitted rows at layer {li} (n={n})"

        r2_obs = {
            name: (1.0 - ss_res_obs[name] / (ss_tot_xs if name == "actx" else ss_tot_y))
            for name in capture_names
        }
        r2_null = 1.0 - ss_null_res / np.maximum(ss_null_tot[:, None], 1e-12)

        y_np = Yt[:, li, :].astype(np.float32)
        xs_np = Xs[:, li, :].astype(np.float32)
        y64 = y_np.astype(np.float64)

        # Paired bootstrap over the captured prediction matrices (raw scale).
        draws_raw = np.stack(
            [la.weighted_r2_draws(captured[name], y_np, w_boot) for name in DRAWS_ORDER]
        )
        # Recalibrated scale: the SAME cross-fitted per-dim affine transform
        # class applied to BOTH ARMS of every gap (§3 scale-consistency).
        recal_preds: dict[str, np.ndarray] = {}
        recal_r2: dict[str, float] = {}
        for name in DRAWS_ORDER:
            rec = rc.crossfit_recal_direct(captured[name].astype(np.float64), y64, folds)
            recal_preds[name] = rec["pred_recal"].astype(np.float32)
            recal_r2[name] = float(rec["r2"])
        draws_recal = np.stack(
            [la.weighted_r2_draws(recal_preds[name], y_np, w_boot) for name in DRAWS_ORDER]
        )

        def _tier_block(draws: np.ndarray, r2s: dict[str, float]) -> dict:
            tiers = {}
            for t, name in enumerate(TIER_NAMES):
                gap_draws = draws[0] - draws[1 + t]
                tiers[name] = {
                    "r2": float(r2s[name]),
                    "gap": float(r2s["within"]) - float(r2s[name]),
                    "gap_bootstrap": la._ci(gap_draws),
                    "r2_bootstrap": la._ci(draws[1 + t]),
                }
            increments = {}
            for t in range(1, len(TIER_NAMES)):
                inc_draws = draws[1 + t] - draws[t]
                increments[f"t{t - 1}->t{t}"] = {
                    "point": float(r2s[f"t{t}"]) - float(r2s[f"t{t - 1}"]),
                    **la._ci(inc_draws),
                }
            st_point = sufficient_tier(r2s["within"], [r2s[nm] for nm in TIER_NAMES], band)
            st_block = {
                "tier": st_point if st_point is not None else "none",
                "band": float(band),
                "per_draw_hist": _sufficient_tier_draws(draws, band),
            }
            if st_point is not None:
                st_block["gap_ci_at_tier"] = la._ci(draws[0] - draws[1 + st_point])
            return {
                "within_r2": float(r2s["within"]),
                "within_r2_bootstrap": la._ci(draws[0]),
                "tiers": tiers,
                "tier_adjacent_increments": increments,
                "delta_tier8": {
                    "point": float(r2s["within"]) - float(r2s["t8"]),
                    **la._ci(draws[0] - draws[9]),  # DRAWS_ORDER: index 9 == t8
                },
                "sufficient_tier": st_block,
            }

        raw_r2s = {name: float(r2_obs[name]) for name in ("within", *TIER_NAMES)}
        recal_r2s = {name: recal_r2[name] for name in ("within", *TIER_NAMES)}

        # Identity+learned-bias + kNN retrieval (standing mapping-baselines
        # rule) at the full-tier layer set (the dispatcher passes the
        # headline layer; smoke covers the whole frozen set).
        baselines = None
        if li in full_tier_layers:
            rng_knn = np.random.default_rng(KNN_SUBSAMPLE_SEED)
            keep = np.sort(rng_knn.choice(n, size=min(n, knn_max_rows), replace=False))
            id_within_r2 = fc._pooled_r2(captured["id_within"], y_np)
            id_actx_r2 = fc._pooled_r2(captured["id_actx"], xs_np)
            id_aans_r2 = fc._pooled_r2(captured["id_aans"], y_np)
            baselines = {
                "knn_subsample_rows": int(len(keep)),
                "within": {
                    "identity_bias_r2": float(id_within_r2),
                    "knn": _knn_block(captured["within"], y_np, keep),
                    "knn_identity_bias": _knn_block(captured["id_within"], y_np, keep),
                },
                "tier8": {
                    "identity_bias_r2": float(id_within_r2),  # same x_t -> y_t map class
                    "knn": _knn_block(captured["t8"], y_np, keep),
                },
                "A_ctx_rev": {
                    "identity_bias_r2": float(id_actx_r2),
                    "knn": _knn_block(captured["actx"], xs_np, keep),
                    "knn_identity_bias": _knn_block(captured["id_actx"], xs_np, keep),
                },
                "A_ans": {
                    "identity_bias_r2": float(id_aans_r2),
                    "knn": _knn_block(captured["aans_own"], y_np, keep),
                    "knn_identity_bias": _knn_block(captured["id_aans"], y_np, keep),
                },
            }

        per_layer[str(li)] = {
            "raw": _tier_block(draws_raw, raw_r2s),
            "recal": _tier_block(draws_recal, recal_r2s),
            "repswap_r2": float(r2_obs["repswap"]),
            "repswap_r2_bootstrap": la._ci(la.weighted_r2_draws(captured["repswap"], y_np, w_boot)),
            "alignment_r2": {
                "A_ctx_rev": float(r2_obs["actx"]),
                "A_ans": float(r2_obs["aans_own"]),
            },
            "nulls": {
                "order": list(DRAWS_ORDER),
                "n_draws": int(null_draws),
                "r2_matrix": [[float(v) for v in row] for row in r2_null],
                "p975_per_read": [float(v) for v in np.nanquantile(r2_null, 0.975, axis=0)]
                if null_draws
                else [],
                "t5_note": "fixed observed R, intercept refit per draw (see module docstring)",
            },
            "procrustes": {
                "trace_R_over_d_mean": float(
                    np.mean([p["trace_R_over_d"] for p in procrustes_rows])
                ),
                "aligned_cos_mean": float(np.mean([p["aligned_cos"] for p in procrustes_rows])),
                "per_fold": procrustes_rows,
                "direction_aware": True,
            },
            "selected_lambda": {
                name: _lambda_edge_stats(vals, grid) for name, vals in lam_log.items()
            },
            "selectors": {name: sorted(set(v)) for name, v in selector_log.items()},
            "baselines": baselines,
            "is_full_tier_layer": bool(li in full_tier_layers),
        }

        # fp16 preds persistence (plan §3 row-coverage + §4(d)).
        preds_store[f"within_l{li}"] = captured["within"].astype(np.float16)
        preds_store[f"t8_l{li}"] = captured["t8"].astype(np.float16)
        preds_store[f"within_recal_l{li}"] = recal_preds["within"].astype(np.float16)
        preds_store[f"t8_recal_l{li}"] = recal_preds["t8"].astype(np.float16)
        preds_store[f"y_l{li}"] = y_np.astype(np.float16)
        preds_store[f"tier_r2_draws_l{li}"] = draws_raw.astype(np.float32)
        preds_store[f"tier_r2_draws_recal_l{li}"] = draws_recal.astype(np.float32)
        if li in full_tier_layers:
            for t in range(8):  # t8 always persisted above
                preds_store[f"t{t}_l{li}"] = captured[f"t{t}"].astype(np.float16)
                preds_store[f"t{t}_recal_l{li}"] = recal_preds[f"t{t}"].astype(np.float16)
        del Xs_l, Ys_l, Xt_l, Yt_l, captured, recal_preds
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "n_shared_rows": n,
        "frozen_layers": list(frozen_layers),
        "full_tier_layers": list(full_tier_layers),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "null_draws": int(null_draws),
        "n_boot": int(n_boot),
        "boot_seed": int(boot_seed),
        "lambda_grid": [float(v) for v in grid],
        "n_inner_lambda_folds": int(n_inner),
        "band": float(band),
        "draws_order": list(DRAWS_ORDER),
        "prep_cache": {"hits": cache.hits, "misses": cache.misses},
        "per_layer": per_layer,
    }
    return payload, preds_store


# ===========================================================================
# File-level driver
# ===========================================================================
def _load_surface_xy(
    ts_dir: Path,
    model: str,
    fmt: str,
    corpus: str,
    *,
    smoke: bool,
    wave1_dir: Path | None,
    gen_root: Path | None,
    expected_layers: int | None,
) -> dict:
    """(X, Y, conv_ids) for one model on one CONTEXT-arm surface.

    The two EXTENDED corpora load through the concat loader (wave-1 stem +
    v2 extension stem, disjointness + text-sha join asserts) in production;
    smoke fixtures are single complete stems (the concat seam is pinned by
    its own tests).
    """
    if not smoke and corpus in et.CONCAT_SOURCES:
        bundle = et.load_bundle_concat(
            ts_dir, model, fmt, corpus, wave1_dir=wave1_dir, gen_root=gen_root
        )
    else:
        bundle = fc._load_bundle_any(ts_dir, model, fmt, corpus)
    exp = expected_layers if expected_layers is not None else f36._bundle_n_layers(bundle)
    return f36._cell_xy_1336(bundle, exp, x_slot="context")


def run_pair(args, pair: str, *, bars: dict, prep_cache: PrepCache) -> None:
    """One (pair, surface) battery: load, intersect, run, persist."""
    m0, m1 = pair.split(":")
    assert (m0, m1) in cm.PAIRS, f"pair {pair} not in the registered PAIRS set"
    corpus, fmt = args.corpus, args.format
    assert corpus and fmt, "--corpus and --format are required"
    assert fmt in cm.V2_CORPORA[corpus]["formats"], f"({corpus}, {fmt}) is not a v2 surface"
    smoke = args.smoke
    ts_dir = args.turnstore_dir or Path(
        "data/issue_1336/" + ("turnstore_v2_smoke" if smoke else "turnstore_v2")
    )
    wave1_dir = args.wave1_turnstore_dir or ts_dir
    gen_root = args.gen_root or (None if smoke else Path("data/issue_1336/gen"))
    preds_dir = args.preds_dir or Path(
        "data/issue_1336/" + ("metric_ladder_preds_smoke" if smoke else "metric_ladder_preds")
    )
    if args.frozen_layers:
        frozen = tuple(int(x) for x in args.frozen_layers.split(",") if x.strip())
    else:
        frozen = cm.SMOKE_FROZEN_LAYERS if smoke else cm.FROZEN_LAYERS
    if args.full_tier_layers == "all":
        full_tiers = frozen
    elif args.full_tier_layers == "none":
        full_tiers = ()
    else:
        full_tiers = tuple(int(x) for x in args.full_tier_layers.split(",") if x.strip())
    n_boot = (
        args.n_boot if args.n_boot is not None else (cm.SMOKE_N_BOOT if smoke else cm.N_BOOTSTRAP)
    )
    null_draws = (
        args.null_draws
        if args.null_draws is not None
        else (cm.SMOKE_NULL_DRAWS if smoke else cm.N_NULL_DRAWS)
    )
    exp_layers = None if smoke else cm.EXPECTED_LAYERS

    xy0 = _load_surface_xy(
        ts_dir,
        m0,
        fmt,
        corpus,
        smoke=smoke,
        wave1_dir=wave1_dir,
        gen_root=gen_root,
        expected_layers=exp_layers,
    )
    xy1 = _load_surface_xy(
        ts_dir,
        m1,
        fmt,
        corpus,
        smoke=smoke,
        wave1_dir=wave1_dir,
        gen_root=gen_root,
        expected_layers=exp_layers,
    )
    ids0 = np.asarray([str(c) for c in xy0["conv_ids"]])
    ids1 = np.asarray([str(c) for c in xy1["conv_ids"]])
    common, i0, i1 = la._align_rows(ids0, ids1)
    boot_seed = 5000 + cm.v2_surface_index(corpus, fmt)
    print(
        f"[ladder1336] pair={m0}->{m1} surface=({corpus},{fmt}) n={len(common)} "
        f"boot_seed={boot_seed} frozen={frozen}"
    )
    payload, preds_store = run_battery_arrays(
        xy0["X"][i0],
        xy0["Y"][i0],
        xy1["X"][i1],
        xy1["Y"][i1],
        common,
        frozen_layers=frozen,
        n_folds=N_FOLDS,
        seed=FIT_SEED,
        null_draws=null_draws,
        n_boot=n_boot,
        boot_seed=boot_seed,
        grid=np.asarray(cm.LAMBDAS_23, dtype=np.float64),
        band=float(bars["elicit_band_v2"]),
        knn_max_rows=args.knn_max_rows,
        full_tier_layers=full_tiers,
        prep_cache=prep_cache,
        cache_tags={
            "s": (corpus, fmt, m0),
            "t": (corpus, fmt, m1),
            "ys": (corpus, fmt, m0),
        },
    )

    unit = f"{m0}__{m1}_{fmt}_{corpus}"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / f"ladpreds_{unit}.npz"
    np.savez(preds_path, **preds_store)  # plain savez: compression OFF for Xet (#813)
    sha = hashlib.sha256(preds_path.read_bytes()).hexdigest()
    manifest_path = preds_dir / "metric_ladder_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[preds_path.name] = {
        "sha256": sha,
        "shapes": {k: list(np.asarray(v).shape) for k, v in preds_store.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    payload = {
        "metadata": _metadata(FIT_SEED, payload["n_shared_rows"]),
        "pair": {"m0": m0, "m1": m1},
        "eval_set": {"corpus": corpus, "format": fmt},
        "primary_scale": args.primary_scale,
        "scales": {
            "raw": "pooled fold-local R^2 — primary IFF health gate H passes (§3)",
            "recal": "held-out crossfit per-dim affine recal on BOTH arms of every gap",
        },
        "bars": bars,
        **payload,
        "preds_npz": str(preds_path),
        "preds_sha256": sha,
    }
    _write_json(args.out_dir / "metric_ladder" / f"pair_{unit}.json", payload)


def selfcheck() -> None:
    """Cheap CPU gates: vendored bootstrap oracle + a tier-nesting sanity run."""
    la.selfcheck()
    rng = np.random.default_rng(0)
    n, d = 90, 6
    xs = rng.normal(size=(n, d))
    w_true = rng.normal(size=(d, d)) / np.sqrt(d)
    b_true = rng.normal(size=d)
    ys = xs @ w_true + b_true + 0.05 * rng.normal(size=(n, d))
    dx_true = 3.0 * rng.normal(size=d)
    xt = xs + dx_true  # target contexts = shifted source contexts
    yt = (xt - dx_true) @ w_true + b_true + 0.05 * rng.normal(size=(n, d))
    ids = np.asarray([f"s{i}" for i in range(n)])
    payload, _ = run_battery_arrays(
        xs[:, None, :],
        ys[:, None, :],
        xt[:, None, :],
        yt[:, None, :],
        ids,
        frozen_layers=(0,),
        null_draws=2,
        n_boot=32,
        boot_seed=5000,
        grid=np.asarray(cm.LAMBDAS_23),
        band=0.02,
        full_tier_layers=(0,),
    )
    tiers = payload["per_layer"]["0"]["raw"]["tiers"]
    assert tiers["t1"]["r2"] > tiers["t0"]["r2"] + 0.2, (
        f"tier nesting selfcheck: T1 {tiers['t1']['r2']:.3f} does not rescue "
        f"T0 {tiers['t0']['r2']:.3f} on a base+ctx-offset construction"
    )
    print(
        f"[selfcheck] tier nesting OK (t0={tiers['t0']['r2']:.3f}, t1={tiers['t1']['r2']:.3f}, "
        f"within={payload['per_layer']['0']['raw']['within_r2']:.3f})"
    )


# ===========================================================================
# Cluster delta-Q battery (plan v15 §4 divergence 11 + Phase LAD_cluster;
# sibling instrument: #1902 ``clusters_delta_qc_scatter``).
#
# Estimator (interpretation pinned against the plan's three wordings):
#   * Per-prompt "R^2" is the FOLD-LOCAL POOLED RESIDUAL VARIANCE RATIO under
#     the checkpoint's OWN pooled fit (Phase FIT_pool preds — plan v15 §6
#     "Statistical-input existence" names preds_pooled_v3/*.npz as this
#     battery's input; tier-6 transfer preds are pair-indexed and cannot
#     yield the per-checkpoint R^2_source / R^2_target the formula needs):
#         r_p(k) = ||y_p(k) - yhat_p(k)||^2 / D_f(p)(k),
#     D_f = mean over the prompt's fold-block of ||y_q - ybar_f||^2 (the
#     evaluated set's own mean — fc._pooled_r2 convention), pooled across
#     corpora within the block, so mean_f(r_p) = 1 - R^2_f exactly.
#   * "Held-out prompts" = train-side 5-fold CV rows (each held out
#     fold-locally; blocks = manifest folds) PLUS the pooled 20% test side
#     as its own block (fold id -1, final-fit preds). The pooled split
#     assigns WHOLE clusters to train/test (issue1336_pooled_split.py), so
#     only this union gives every cluster c in the full set a held-out read
#     — the plan's "for each cluster c in 50".
#   * delta-Q_p = r_p(source) - r_p(target); positive = better predicted at
#     the target (higher-is-better-target-prediction). delta-Q(c) = mean
#     over cluster c's held-out prompts. "at the tier-6 headline layer" is
#     the LAYER qualifier: the stage-symmetric pooled headline rule below.
#   * Null: selection-symmetric max-cluster permutation — labels permute at
#     PROMPT grain within (corpus x side) strata (corpus mixture preserved;
#     the test side permutes as its own whole block, keeping every null
#     cluster side-pure like the observed whole-cluster split), selection
#     (max over the SAME cluster set) rides inside each draw; band = 97.5%
#     quantile of the per-draw max. Seed 5100 + registered transition idx
#     (plan v15 §10), arm-index child-seeded for arm-subset determinism.
#   * Off arm = the fixed-answer-text interpretation guard: per transition
#     (i, j) the shared text sources are MODELS - {i, j}; per-prompt r is
#     the mean over those shared rows (every prompt has all of them, so
#     row-grain and prompt-grain cluster means coincide).
# ===========================================================================

DELTA_Q_TRANSITIONS: tuple[tuple[str, str], ...] = (("base", "sft"), *cm.ADJACENT_PAIRS)
DELTA_Q_PERM_SEED = 5100  # plan v15 §10: per-transition permutation seed 5100 + transition idx
DELTA_Q_PERM_DRAWS = 1000
DELTA_Q_SMOKE_PERM_DRAWS = 50
_DQ_KEY_SEP = "\x1f"


def _headline_layer_pooled(
    cells_dir: Path, frozen_layers: tuple[int, ...], override: int | None
) -> dict:
    """Stage-symmetric pooled headline layer (v14 §3 rule, pooled analog of
    ``issue1336_decision_v2.headline_layer_rule_v2``): argmax over the frozen
    set of the MEAN across the 5 checkpoints' ON-policy pooled within-stage
    RAW R^2 (``cells_pooled_<k>_arm_on.json`` r2_per_layer_obs) — fixed
    BEFORE any transition gap is computed. ``override`` is the dispatcher /
    smoke seam and is recorded as such."""
    if override is not None:
        return {
            "headline_layer": int(override),
            "rule": "explicit --headline-layer override (dispatcher/smoke seam)",
        }
    raw_means: dict[int, float] = {}
    for li in frozen_layers:
        vals = []
        for k in cm.MODELS:
            path = cells_dir / f"cells_pooled_{k}_arm_on.json"
            assert path.exists(), (
                f"pooled headline rule requires {path} (run Phase FIT_pool first, or pass "
                "--headline-layer explicitly)"
            )
            cell = json.loads(path.read_text())
            vals.append(float(cell["r2_per_layer_obs"][li]))
        raw_means[li] = float(np.mean(vals))
    best = max(raw_means, key=raw_means.get)
    print(f"[ladder1336] delta-Q headline layer {best} (pooled on-arm raw means {raw_means})")
    return {
        "headline_layer": int(best),
        "rule": "max mean within-stage pooled RAW R^2, 5 checkpoints, on-policy arm, frozen set",
        "raw_means": {str(k): v for k, v in raw_means.items()},
    }


def _load_pooled_rows(arm_dir: Path, cell_id: str) -> dict:
    """Columnar row manifest for one pooled unit (Phase FIT_pool output)."""
    path = arm_dir / f"rows_{cell_id}.json"
    assert path.exists(), (
        f"{path} missing — the delta-Q battery consumes Phase FIT_pool outputs "
        "(issue1336_fit_cells.py --v3-pooled); run/stage that phase first"
    )
    payload = json.loads(path.read_text())
    cols = payload["columns"]
    out = {k: np.asarray(cols[k]) for k in ("row_id", "text_source", "corpus", "conv_id", "side")}
    out["prompt_sha"] = np.asarray(cols["prompt_sha"])
    out["cluster"] = np.asarray([int(c) for c in cols["cluster"]], dtype=np.int64)
    out["fold"] = np.asarray([-1 if f is None else int(f) for f in cols["fold"]], dtype=np.int64)
    out["split_manifest_sha256"] = payload.get("split_manifest_sha256")
    n = out["row_id"].shape[0]
    train = out["side"] == "train"
    assert ((out["fold"] >= 0) == train).all(), f"fold/side mismatch in {path}"
    out["n_rows"] = int(n)
    return out


def _load_pooled_preds(arm_dir: Path, cell_id: str, li: int) -> dict:
    """Held-out prediction matrices at layer ``li`` for one pooled unit:
    train-side CV (fold-local) + test-side final-fit (folds == -1)."""
    out: dict = {}
    for tag, name in (("train", f"preds_{cell_id}.npz"), ("test", f"preds_{cell_id}_test.npz")):
        path = arm_dir / name
        assert path.exists(), (
            f"{path} missing — Phase FIT_pool preds not staged for {cell_id} "
            "(issue1336_fit_cells.py --v3-pooled persists them)"
        )
        with np.load(path, allow_pickle=False) as z:
            key = f"preds_l{li}"
            assert key in z.files, (
                f"{path} lacks {key}; available: {sorted(k for k in z.files if k.startswith('preds_'))}"
                " — headline layer must be in the persisted preds layer set"
            )
            assert bool(z["fitted_mask"].all()), (
                f"{path} carries unfitted rows (fitted_mask not all-True) — refusing a "
                "silently-partial delta-Q read; re-run the producing pooled fit"
            )
            out[f"{tag}_ids"] = np.asarray([str(c) for c in z["conv_ids"]])
            out[f"{tag}_preds"] = np.asarray(z[key])
    return out


def _unit_residual_read(
    unit: dict,
    li: int,
    *,
    preds_root: Path,
    ts_dir: Path,
    off_root: Path | None,
    smoke: bool,
    wave1_dir: Path | None,
    gen_root: Path | None,
    keep_y: bool,
) -> dict:
    """Per-row fold-local residual variance ratios for one pooled unit.

    Streams the unit's (text_source x corpus) turnstore bundles one at a
    time (reusing the Phase FIT_pool loader ``f36._pooled_bundle``), joins
    each block's true Y at layer ``li`` against the persisted held-out
    preds, and normalizes per fold-block (train folds + the test side as
    block -1) with the evaluated set's own mean (fc._pooled_r2 convention).
    ``keep_y`` retains the fp16 Y matrix (scatter-label |activation change|
    read, on-policy arm only). Peak RSS is bounded by ONE bundle's
    ``profiles`` array (the plan's LAD_cluster RSS driver).
    """
    cell_id = unit["cell_id"]
    arm_dir = preds_root / cm.POOLED_ARM_DIRS[unit["arm"]]
    rows = _load_pooled_rows(arm_dir, cell_id)
    preds = _load_pooled_preds(arm_dir, cell_id, li)
    n = rows["n_rows"]
    train_mask = rows["side"] == "train"
    train_pos = np.flatnonzero(train_mask)
    test_pos = np.flatnonzero(~train_mask)
    assert preds["train_ids"].shape[0] == train_pos.shape[0], (
        f"{cell_id}: train preds rows {preds['train_ids'].shape[0]} != manifest train rows "
        f"{train_pos.shape[0]}"
    )
    assert (preds["train_ids"] == rows["row_id"][train_pos]).all(), (
        f"{cell_id}: train preds conv_ids misaligned with rows manifest order"
    )
    assert (preds["test_ids"] == rows["row_id"][test_pos]).all(), (
        f"{cell_id}: test preds conv_ids misaligned with rows manifest order"
    )
    # per-assembly-row gather indices into the side-split preds matrices
    side_idx = np.empty(n, dtype=np.int64)
    side_idx[train_pos] = np.arange(train_pos.shape[0])
    side_idx[test_pos] = np.arange(test_pos.shape[0])
    d = int(preds["train_preds"].shape[1])
    fold_block = np.where(train_mask, rows["fold"], -1)

    ss_res = np.full(n, np.nan, dtype=np.float64)
    y_keep = np.empty((n, d), dtype=np.float16) if keep_y else None
    fold_ids = np.unique(fold_block)
    acc = {
        int(f): {"n": 0, "sum_y": np.zeros(d, dtype=np.float64), "sum_y2": 0.0} for f in fold_ids
    }
    # consecutive (text_source, corpus) runs, assembly order (Phase FIT_pool contract)
    ts, co = rows["text_source"], rows["corpus"]
    change = np.flatnonzero((ts[1:] != ts[:-1]) | (co[1:] != co[:-1])) + 1
    bounds = np.r_[0, change, n]
    seen: set[tuple[str, str]] = set()
    for b0, b1 in itertools.pairwise(bounds):
        j, c = str(ts[b0]), str(co[b0])
        assert (j, c) not in seen, (
            f"{cell_id}: (text_source, corpus) block ({j}, {c}) not contiguous"
        )
        seen.add((j, c))
        bundle = f36._pooled_bundle(
            unit["model"],
            j,
            c,
            ts_dir=ts_dir,
            off_root=off_root,
            smoke=smoke,
            wave1_dir=wave1_dir,
            gen_root=gen_root,
        )
        arrays, sidecar = bundle["arrays"], bundle["sidecar"]
        pos = {str(cid): i for i, cid in enumerate(sidecar["conv_ids"])}
        conv_blk = rows["conv_id"][b0:b1]
        missing = [cid for cid in conv_blk if cid not in pos]
        assert not missing, (
            f"{cell_id}: {len(missing)} manifest rows missing from bundle ({j}, {c}) "
            f"(e.g. {missing[:5]}) — pooled row-coverage break"
        )
        sel = np.asarray([pos[cid] for cid in conv_blk], dtype=np.int64)
        profiles = np.asarray(arrays["profiles"])  # (N, 2, L, D); one bundle at a time
        assert profiles.ndim == 4 and profiles.shape[1] == 2, profiles.shape
        assert li < profiles.shape[2], (
            f"layer {li} out of range for bundle ({j}, {c}) with {profiles.shape[2]} layers"
        )
        y_blk = profiles[sel, 1, li, :]
        del profiles
        bundle = arrays = None  # release the bundle before the next block (peak-RSS bound)
        assert y_blk.shape == (b1 - b0, d), (y_blk.shape, (b1 - b0, d))
        assert not np.isnan(y_blk).any(), (
            f"{cell_id}: NaN Y rows in bundle ({j}, {c}) — refusing to silently drop rows"
        )
        if y_keep is not None:
            y_keep[b0:b1] = y_blk.astype(np.float16)
        blk_fold = fold_block[b0:b1]
        blk_side_idx = side_idx[b0:b1]
        blk_train = train_mask[b0:b1]
        y64 = y_blk.astype(np.float64)
        for is_train, mat in ((True, preds["train_preds"]), (False, preds["test_preds"])):
            m = blk_train == is_train
            if not m.any():
                continue
            p64 = mat[blk_side_idx[m]].astype(np.float64)
            ss_res[b0 + np.flatnonzero(m)] = ((y64[m] - p64) ** 2).sum(axis=1)
        for f in np.unique(blk_fold):
            m = blk_fold == f
            a = acc[int(f)]
            a["n"] += int(m.sum())
            a["sum_y"] += y64[m].sum(axis=0)
            a["sum_y2"] += float((y64[m] ** 2).sum())
        del y_blk, y64
    assert not np.isnan(ss_res).any(), f"{cell_id}: residual coverage incomplete"

    d_fold: dict[int, float] = {}
    fold_r2: dict[str, float] = {}
    for f, a in acc.items():
        mu = a["sum_y"] / a["n"]
        d_f = a["sum_y2"] / a["n"] - float(mu @ mu)
        assert d_f > 0, f"{cell_id}: non-positive fold-block variance (fold {f})"
        d_fold[f] = d_f
        m = fold_block == f
        fold_r2[str(f)] = float(1.0 - ss_res[m].mean() / d_f)
    denom = np.asarray([d_fold[int(f)] for f in fold_block], dtype=np.float64)
    r = ss_res / denom
    print(
        f"[deltaq] unit={cell_id} layer={li} n_rows={n} d={d} "
        f"fold_r2={{{', '.join(f'{k}: {v:.4f}' for k, v in sorted(fold_r2.items()))}}}",
        flush=True,
    )
    return {
        "cell_id": cell_id,
        "model": unit["model"],
        "arm": unit["arm"],
        "r": r,
        "y": y_keep,
        "rows": rows,
        "fold_r2_check": fold_r2,
        "d": d,
        "split_manifest_sha256": rows["split_manifest_sha256"],
    }


def _prompt_frame(read: dict, sources: list[str] | None) -> dict:
    """Collapse one unit's row-grain residual ratios to PROMPT grain.

    ``sources`` restricts to the transition's shared text sources (off arm);
    None keeps all rows (on arm — one source). Per prompt (corpus, conv_id):
    r = mean over its retained rows; cluster/corpus/side asserted constant
    across the prompt's rows. Rows arrive in manifest order, so first-index
    gathers are deterministic."""
    rows = read["rows"]
    if sources is None:
        m = np.ones(rows["n_rows"], dtype=bool)
    else:
        m = np.isin(rows["text_source"], np.asarray(sources))
        assert m.any(), f"{read['cell_id']}: no rows for shared sources {sources}"
    idx = np.flatnonzero(m)
    keys = np.char.add(
        np.char.add(rows["corpus"][idx].astype(str), _DQ_KEY_SEP),
        rows["conv_id"][idx].astype(str),
    )
    uniq, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse)
    r_prompt = np.bincount(inverse, weights=read["r"][idx]) / counts
    first = np.full(uniq.shape[0], np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, inverse, np.arange(idx.shape[0]))
    assert (first < idx.shape[0]).all()  # every group has a first occurrence by construction
    row_first = idx[first]
    for col in ("cluster", "corpus", "side"):
        ref = rows[col][row_first]
        assert (rows[col][idx] == ref[inverse]).all(), (
            f"{read['cell_id']}: prompt-level column {col!r} inconsistent across text sources"
        )
    return {
        "keys": uniq,
        "r": r_prompt,
        "cluster": rows["cluster"][row_first],
        "corpus": rows["corpus"][row_first].astype(str),
        "side": rows["side"][row_first].astype(str),
        "conv_id": rows["conv_id"][row_first].astype(str),
        "prompt_sha": rows["prompt_sha"][row_first].astype(str),
        "row_first": row_first,
        "n_rows_per_prompt": counts,
    }


def _perm_null_battery(
    delta: np.ndarray,
    cluster_idx: np.ndarray,
    strata_idx: np.ndarray,
    n_clusters: int,
    n_draws: int,
    rng: np.random.Generator,
    chunk: int,
) -> np.ndarray:
    """(n_draws, n_clusters) per-cluster delta-Q means under within-stratum
    permutation of prompt->cluster assignments — BATCHED (no per-draw Python
    loop; per `.claude/rules/vectorize-many-cell-fits.md`): per stratum one
    argsort of a (chunk, m) uniform block permutes labels for all draws in
    the chunk at once; per-cluster sums via one bincount over the flattened
    (draw, cluster) key. Cluster counts are permutation-invariant (label
    multisets are preserved within every stratum)."""
    n = delta.shape[0]
    counts = np.bincount(cluster_idx, minlength=n_clusters).astype(np.float64)
    assert (counts > 0).all(), "empty cluster in the observed frame"
    order = np.argsort(strata_idx, kind="stable")
    sorted_strata = strata_idx[order]
    starts = np.r_[0, np.flatnonzero(sorted_strata[1:] != sorted_strata[:-1]) + 1, n]
    labels_sorted = cluster_idx[order]
    null = np.empty((n_draws, n_clusters), dtype=np.float64)
    done = 0
    while done < n_draws:
        b = min(chunk, n_draws - done)
        perm = np.empty((b, n), dtype=np.int64)
        for s0, s1 in itertools.pairwise(starts):
            m = s1 - s0
            shuf = np.argsort(rng.random((b, m)), axis=1)
            perm[:, order[s0:s1]] = labels_sorted[s0:s1][shuf]
        flat = (perm + (np.arange(b) * n_clusters)[:, None]).ravel()
        w = np.broadcast_to(delta, (b, n)).ravel()
        sums = np.bincount(flat, weights=w, minlength=b * n_clusters).reshape(b, n_clusters)
        null[done : done + b] = sums / counts[None, :]
        done += b
    return null


def _transition_arm_stats(
    fi: dict,
    fj: dict,
    *,
    cluster_ids: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    chunk: int,
) -> dict:
    """Observed per-cluster delta-Q + selection-symmetric max-cluster null
    for one (transition, arm). ``fi``/``fj`` are aligned prompt frames
    (source / target)."""
    assert (fi["keys"] == fj["keys"]).all(), "prompt frames misaligned across the pair"
    for col in ("cluster", "corpus", "side"):
        assert (fi[col] == fj[col]).all(), f"prompt attribute {col!r} differs across the pair"
    delta = fi["r"] - fj["r"]  # residual-ratio source - target; positive = target better
    cl_pos = {int(c): i for i, c in enumerate(cluster_ids)}
    cluster_idx = np.asarray([cl_pos[int(c)] for c in fi["cluster"]], dtype=np.int64)
    strata_keys = np.char.add(np.char.add(fi["corpus"], _DQ_KEY_SEP), fi["side"])
    _, strata_idx = np.unique(strata_keys, return_inverse=True)
    k = cluster_ids.shape[0]
    counts = np.bincount(cluster_idx, minlength=k).astype(np.float64)
    obs = np.bincount(cluster_idx, weights=delta, minlength=k) / counts
    ceiling = np.bincount(cluster_idx, weights=fi["r"], minlength=k) / counts
    null = _perm_null_battery(delta, cluster_idx, strata_idx, k, n_draws, rng, chunk)
    null_max = null.max(axis=1)
    obs_max = float(obs.max())
    corpora = sorted(set(fi["corpus"].tolist()))
    co_pos = {c: i for i, c in enumerate(corpora)}
    co_idx = np.asarray([co_pos[c] for c in fi["corpus"]], dtype=np.int64)
    cc_flat = cluster_idx * len(corpora) + co_idx
    cc_n = np.bincount(cc_flat, minlength=k * len(corpora)).astype(np.float64)
    cc_sum = np.bincount(cc_flat, weights=delta, minlength=k * len(corpora))
    with np.errstate(invalid="ignore"):
        percorpus = (cc_sum / cc_n).reshape(
            k, len(corpora)
        )  # NaN where a (cluster, corpus) is empty
    per_corpus_total = {c: float(delta[co_idx == i].mean()) for c, i in co_pos.items()}
    return {
        "delta": delta,
        "cluster_idx": cluster_idx,
        "obs": obs,
        "ceiling": ceiling,
        "counts": counts,
        "null": null,
        "null_max": null_max,
        "obs_max": obs_max,
        "obs_argmax_cluster": int(cluster_ids[int(obs.argmax())]),
        "p_max_cluster": float((null_max >= obs_max).mean()),
        "p_max_cluster_add_one": float((1 + int((null_max >= obs_max).sum())) / (n_draws + 1)),
        "band_97p5": float(np.quantile(null_max, 0.975)),
        "corpora": corpora,
        "percorpus": percorpus,
        "per_corpus_total": per_corpus_total,
    }


def _top_cluster_anchors(
    stats: dict, frame: dict, cluster_ids: np.ndarray, *, top_k: int = 3, n_anchors: int = 5
) -> list[dict]:
    """Top-k most-improved clusters with corpus_slug|row_idx prompt anchors
    (plan v15 §4 div. 11: top 5 prompt_ids per top-3 cluster, ranked by
    per-prompt delta)."""
    out = []
    for ci in np.argsort(-stats["obs"])[:top_k]:
        cid = int(cluster_ids[ci])
        m = np.flatnonzero(stats["cluster_idx"] == ci)
        top = m[np.argsort(-stats["delta"][m])[:n_anchors]]
        out.append(
            {
                "cluster": cid,
                "delta_q": float(stats["obs"][ci]),
                "n_prompts": int(stats["counts"][ci]),
                "side": str(frame["side"][m[0]]),
                "anchors": [
                    {
                        "anchor": f"{frame['corpus'][t]}|{frame['conv_id'][t].lstrip('s')}",
                        "conv_id": str(frame["conv_id"][t]),
                        "corpus": str(frame["corpus"][t]),
                        "prompt_sha": str(frame["prompt_sha"][t]),
                        "delta": float(stats["delta"][t]),
                    }
                    for t in top
                ],
            }
        )
    return out


def _activation_change_labels(
    read_i: dict, read_j: dict, fi: dict, fj: dict, cluster_ids: np.ndarray
) -> dict[str, list[dict]]:
    """Per-cluster top-3 prompts by pooled |activation change| at the
    headline layer (on-policy arm scatter labels, plan v15 §4 div. 11):
    ||y_p(target) - y_p(source)||_2 over the prompt's single on-arm row."""
    assert read_i["y"] is not None and read_j["y"] is not None
    yi = read_i["y"][fi["row_first"]].astype(np.float32)
    yj = read_j["y"][fj["row_first"]].astype(np.float32)
    act = np.linalg.norm(yj - yi, axis=1)
    out: dict[str, list[dict]] = {}
    for cid in cluster_ids:
        m = np.flatnonzero(fi["cluster"] == cid)
        top = m[np.argsort(-act[m])[:3]]
        out[str(int(cid))] = [
            {
                "anchor": f"{fi['corpus'][t]}|{fi['conv_id'][t].lstrip('s')}",
                "conv_id": str(fi["conv_id"][t]),
                "act_change": float(act[t]),
            }
            for t in top
        ]
    return out


def cluster_delta_q_battery(args) -> None:
    """Phase LAD_cluster (plan v15): per-transition per-cluster delta-Q +
    selection-symmetric max-cluster permutation null over the Phase FIT_pool
    outputs. Writes ``decision_v3/cluster_delta_q_per_transition.json`` (the
    §6.5 deliverable) + one per-draw x per-cluster npz per transition under
    ``--perdraw-dir`` (uncompressed savez — compression OFF per the plan's
    store-heavy sizing rule), persisted per transition as computed
    (checkpoint-per-phase)."""
    smoke = args.smoke
    sfx = "_smoke" if smoke else ""
    preds_root = args.pooled_preds_root or Path(f"data/issue_1336/preds_pooled_v3{sfx}")
    ts_dir = args.turnstore_dir or Path(f"data/issue_1336/turnstore_v2{sfx}")
    off_root = args.offpolicy_root or Path("data/issue_1336")
    wave1_dir = args.wave1_turnstore_dir or ts_dir
    gen_root = args.gen_root or (None if smoke else Path("data/issue_1336/gen"))
    out_dir = args.out_dir
    perdraw_dir = args.perdraw_dir
    frozen = (
        tuple(int(x) for x in args.frozen_layers.split(",") if x.strip())
        if args.frozen_layers
        else (cm.SMOKE_FROZEN_LAYERS if smoke else cm.FROZEN_LAYERS)
    )
    head = _headline_layer_pooled(out_dir / "cells_pooled_v3", frozen, args.headline_layer)
    li = head["headline_layer"]
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    assert arms and all(a in ("on", "off") for a in arms), f"--arms must be within on,off: {arms}"
    if args.transitions:
        transitions = []
        for tok in args.transitions.split(","):
            i, j = tok.strip().split(":")
            assert i in cm.MODELS and j in cm.MODELS and i != j, f"bad transition {tok!r}"
            transitions.append((i, j))
        transitions = tuple(transitions)
    else:
        transitions = DELTA_Q_TRANSITIONS
    n_draws = (
        args.perm_draws
        if args.perm_draws is not None
        else (DELTA_Q_SMOKE_PERM_DRAWS if smoke else DELTA_Q_PERM_DRAWS)
    )
    units = {(u["arm"], u["model"]): u for u in f36._pooled_units_for()}

    cache: dict[tuple[str, str], dict] = {}
    manifest_shas: set[str] = set()

    def unit_read(arm: str, model: str) -> dict:
        key = (arm, model)
        if key not in cache:
            cache[key] = _unit_residual_read(
                units[key],
                li,
                preds_root=preds_root,
                ts_dir=ts_dir,
                off_root=off_root if arm == "off" else None,
                smoke=smoke,
                wave1_dir=wave1_dir,
                gen_root=gen_root,
                keep_y=(arm == "on"),
            )
            sha = cache[key]["split_manifest_sha256"]
            if sha:
                manifest_shas.add(str(sha))
        return cache[key]

    payload: dict = {
        "metadata": _metadata(DELTA_Q_PERM_SEED, 0),
        "headline": head,
        "estimator": (
            "per-prompt fold-local pooled residual variance ratio under the checkpoint's own "
            "pooled fit (Phase FIT_pool preds; train-side 5-fold CV held-out rows + the pooled "
            "20% test side as fold-block -1); delta = r_source - r_target, positive = better "
            "predicted at the target"
        ),
        "null": (
            "selection-symmetric max-cluster permutation: prompt->cluster labels permuted "
            "within (corpus x side) strata, per-draw max over the same cluster set; band = "
            "97.5% quantile of the per-draw max"
        ),
        "perm_draws": int(n_draws),
        "perm_seed_rule": "default_rng([5100 + registered transition idx, arm_idx(on=0,off=1)])",
        "arms": list(arms),
        "transitions": {},
    }
    for i, j in transitions:
        t_slug = f"{i}__{j}"
        if (i, j) in DELTA_Q_TRANSITIONS:
            t_idx = DELTA_Q_TRANSITIONS.index((i, j))
        else:
            t_idx = 40 + transitions.index((i, j))  # non-registered (smoke) transitions
        t_block: dict = {"source": i, "target": j, "transition_idx": int(t_idx), "arms": {}}
        npz_arrays: dict[str, np.ndarray] = {}
        for arm in arms:
            read_i, read_j = unit_read(arm, i), unit_read(arm, j)
            if arm == "off":
                shared = sorted(
                    set(np.unique(read_i["rows"]["text_source"]).tolist())
                    & set(np.unique(read_j["rows"]["text_source"]).tolist())
                )
                assert shared, f"off arm: no shared text sources for {i}->{j}"
            else:
                shared = None
            fi = _prompt_frame(read_i, shared)
            fj = _prompt_frame(read_j, shared)
            cluster_ids = np.unique(fi["cluster"])
            rng = np.random.default_rng([DELTA_Q_PERM_SEED + t_idx, 0 if arm == "on" else 1])
            stats = _transition_arm_stats(
                fi, fj, cluster_ids=cluster_ids, n_draws=n_draws, rng=rng, chunk=args.perm_chunk
            )
            arm_block = {
                "cell_ids": [read_i["cell_id"], read_j["cell_id"]],
                "shared_text_sources": shared,
                "n_prompts": int(fi["keys"].shape[0]),
                "cluster_ids": [int(c) for c in cluster_ids],
                "delta_q": [float(v) for v in stats["obs"]],
                "n_prompts_per_cluster": [int(v) for v in stats["counts"]],
                "ceiling_delta_q_per_cluster": [float(v) for v in stats["ceiling"]],
                "ceiling_max": float(stats["ceiling"].max()),
                "obs_max": stats["obs_max"],
                "obs_argmax_cluster": stats["obs_argmax_cluster"],
                "max_cluster_p": stats["p_max_cluster"],
                "max_cluster_p_add_one": stats["p_max_cluster_add_one"],
                "null_band_97p5": stats["band_97p5"],
                "top3_most_improved": _top_cluster_anchors(stats, fi, cluster_ids),
                "per_corpus_delta_q": stats["per_corpus_total"],
                "fold_r2_check": {
                    "source": read_i["fold_r2_check"],
                    "target": read_j["fold_r2_check"],
                },
            }
            if arm == "on":
                arm_block["cluster_prompt_labels_by_act_change"] = _activation_change_labels(
                    read_i, read_j, fi, fj, cluster_ids
                )
            t_block["arms"][arm] = arm_block
            npz_arrays[f"obs_{arm}"] = stats["obs"]
            npz_arrays[f"null_{arm}"] = stats["null"].astype(np.float32)
            npz_arrays[f"n_per_cluster_{arm}"] = stats["counts"].astype(np.int64)
            npz_arrays[f"percorpus_{arm}"] = stats["percorpus"]
            npz_arrays[f"ceiling_{arm}"] = stats["ceiling"]
            npz_arrays[f"corpora_{arm}"] = np.asarray(stats["corpora"])
            npz_arrays[f"cluster_ids_{arm}"] = cluster_ids
            print(
                f"[deltaq] {i}->{j} arm={arm} obs_max={stats['obs_max']:.5f} "
                f"(cluster {stats['obs_argmax_cluster']}) p={stats['p_max_cluster']:.4f} "
                f"band97.5={stats['band_97p5']:.5f} draws={n_draws}",
                flush=True,
            )
        # interpretation guard (plan v15 §4 div. 11): on-vs-off top-3 overlap
        if "on" in t_block["arms"] and "off" in t_block["arms"]:
            top_on = [e["cluster"] for e in t_block["arms"]["on"]["top3_most_improved"]]
            top_off = [e["cluster"] for e in t_block["arms"]["off"]["top3_most_improved"]]
            t_block["interpretation_guard"] = {
                "top3_on": top_on,
                "top3_off": top_off,
                "overlap": sorted(set(top_on) & set(top_off)),
                "rule": (
                    "a cluster most-improved under BOTH arms at fixed activation checkpoint "
                    "reads as representation-change; on-policy-only reads as a data-structure "
                    "artifact (fixed-answer-text guard)"
                ),
            }
        perdraw_dir.mkdir(parents=True, exist_ok=True)
        npz_path = perdraw_dir / f"{t_slug}.npz"
        np.savez(npz_path, **npz_arrays)  # plain savez: compression OFF (plan §4 div. 11)
        t_block["perdraw_npz"] = {
            "path": str(npz_path),
            "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        }
        print(f"[deltaq] persisted {npz_path}")
        payload["transitions"][t_slug] = t_block
        # evict units no further transition needs (bounds the on-arm Y stash to ~2 units)
        remaining = set(transitions[transitions.index((i, j)) + 1 :])
        needed = {m for pair in remaining for m in pair}
        for key in [k for k in cache if k[1] not in needed]:
            del cache[key]
    payload["split_manifest_sha256"] = sorted(manifest_shas)
    _write_json(out_dir / "decision_v3" / "cluster_delta_q_per_transition.json", payload)


# ---------------------------------------------------------------------------
# Phase LAD_pool (plan v15 §4 divergence 9): pooled-split transfer tier reads.
#
# One invocation = one (source i -> target j, arm) battery over the POOLED
# split: tier maps are fit on the manifest TRAIN side (pooled across all
# manifest corpora — "W_source cached per (i, a, fold)" reduces to one
# train-side prep per role here, since the panel reads the single pooled
# 80/20 split, not the inner CV folds) and evaluated on the pooled 20% TEST
# side, then sliced per corpus for the 2x2 panel (§4: "Held-out on the pooled
# 20% split, sliced by corpus"). The tier engine is the UNCHANGED
# ``_fold_observed`` (the round-3 T0..T8 battery) called with tr/te = the
# manifest sides; nulls reuse ``_fold_null_r2_contrib`` (shuffled-
# correspondence refit, §4 div. 10 extended to the off arm).
#
# Arms (cm.cells_v3_for semantics):
#   on  — Xs/Ys = checkpoint i on its OWN text (diagonal v2 turnstore);
#         Xt/Yt = checkpoint j on its OWN text. Text differs across sides
#         (each model answers the same prompts on-policy).
#   off — Xs/Ys = checkpoint i captured teacher-forced on j's text (the
#         Phase EXT_off turnstore_offpolicy_<i>_chat_<j> tree); Xt/Yt =
#         checkpoint j on its own text. Fixed-answer-text guard: the TEXT is
#         identical across sides, so tier gaps read as representation change.
#
# Bootstrap: per-corpus draw matrices seeded 5300 + v2_surface_index(c, chat)
# (pooled-overall 5299) — seeds are pair/arm-independent and the test rows
# are the SAME manifest rows for every (pair, arm), so draws are PAIRED
# across arms and stages (§4: "paired across arms and stages sharing the
# fold partition"). Guideline-11 companions (identity+learned-bias baseline
# + kNN retrieval) ride the pooled test side for the own/T0 reads.
#
# Outputs (plan §9 phase_outputs.ladder_pool):
#   eval_results/issue_1336/metric_ladder_pooled_v3/pair_<i>__<j>_arm_<a>.json
#   data/issue_1336/metric_ladder_pooled_v3[_smoke]/pair_<i>__<j>_arm_<a>.npz
# ---------------------------------------------------------------------------

POOLED_PAIR_TIERS = ("within", "t0", "t6", "t8")
POOLED_PAIR_BOOT_SEED_BASE = 5300  # + v2 chat-surface index per corpus slice
POOLED_PAIR_BOOT_SEED_POOLED = 5299  # the all-corpora pooled read


def _pooled_pair_assemble(
    model: str,
    text_source: str,
    by_corpus: dict[str, list[dict]],
    corpora: tuple[str, ...],
    *,
    ts_dir: Path,
    off_root: Path,
    smoke: bool,
    wave1_dir: Path | None,
    gen_root: Path | None,
    expected_layers: int | None,
    frozen: tuple[int, ...],
) -> dict:
    """One side's pooled (X, Y, rows) via the Phase FIT_pool loaders (f36)."""
    unit = {"model": model, "text_sources": [text_source]}
    return f36._assemble_pooled_rows(
        unit,
        by_corpus,
        corpora=corpora,
        ts_dir=ts_dir,
        off_root=off_root,
        smoke=smoke,
        wave1_dir=wave1_dir,
        gen_root=gen_root,
        x_slot="context",
        expected_layers=expected_layers,
        frozen_layers=frozen,
    )


def _pooled_pair_tier_block(pred: np.ndarray, y: np.ndarray, idx_w: np.ndarray) -> dict:
    """{r2, r2_bootstrap} for one (tier, row-slice) read — the panel's unit."""
    return {
        "r2": float(fc._pooled_r2(pred, y)),
        "r2_bootstrap": la._ci(la.weighted_r2_draws(pred, y, idx_w)),
    }


def run_pooled_pair(args) -> None:
    """One Phase LAD_pool (pair, arm) battery: load, align, fit, slice, persist."""
    smoke = args.smoke
    sfx = "_smoke" if smoke else ""
    src, tgt = args.pooled_pair.split(":")
    assert src in cm.MODELS and tgt in cm.MODELS, f"unknown checkpoint in {args.pooled_pair!r}"
    assert src != tgt, "--pooled-pair needs an ordered i:j with i != j"
    arm = args.arm
    assert arm in ("on", "off"), "--pooled-pair requires --arm on|off"
    man_path = args.split_manifest or Path(
        f"data/issue_1336/pooled_split_v3{sfx}/split_manifest.json"
    )
    man = f36._load_split_manifest(man_path)
    man_sha = hashlib.sha256(man_path.read_bytes()).hexdigest()
    by_corpus = f36._manifest_rows_by_corpus(man)
    corpora = tuple(by_corpus)
    ts_dir = args.turnstore_dir or Path(f"data/issue_1336/turnstore_v2{sfx}")
    off_root = args.offpolicy_root or Path("data/issue_1336")
    wave1_dir = args.wave1_turnstore_dir or ts_dir
    gen_root = args.gen_root or (None if smoke else Path("data/issue_1336/gen"))
    if args.frozen_layers:
        run_layers = tuple(int(x) for x in args.frozen_layers.split(",") if x.strip())
    else:
        run_layers = cm.SMOKE_FROZEN_LAYERS if smoke else cm.FROZEN_LAYERS
    exp_layers = None if smoke else cm.EXPECTED_LAYERS
    n_boot = (
        args.n_boot if args.n_boot is not None else (cm.SMOKE_N_BOOT if smoke else cm.N_BOOTSTRAP)
    )
    null_draws = (
        args.null_draws
        if args.null_draws is not None
        else (cm.SMOKE_NULL_DRAWS if smoke else cm.N_NULL_DRAWS)
    )
    grid = np.asarray(cm.LAMBDAS_23, dtype=np.float64)
    src_text = src if arm == "on" else tgt
    asm_s = _pooled_pair_assemble(
        src,
        src_text,
        by_corpus,
        corpora,
        ts_dir=ts_dir,
        off_root=off_root,
        smoke=smoke,
        wave1_dir=wave1_dir,
        gen_root=gen_root,
        expected_layers=exp_layers,
        frozen=run_layers,
    )
    asm_t = _pooled_pair_assemble(
        tgt,
        tgt,
        by_corpus,
        corpora,
        ts_dir=ts_dir,
        off_root=off_root,
        smoke=smoke,
        wave1_dir=wave1_dir,
        gen_root=gen_root,
        expected_layers=exp_layers,
        frozen=run_layers,
    )
    rows = asm_s["rows"]
    for a, b in zip(rows, asm_t["rows"], strict=True):
        assert (a["corpus"], a["conv_id"], a["side"]) == (b["corpus"], b["conv_id"], b["side"]), (
            "pooled row alignment break between sides",
            a,
            b,
        )
    n = len(rows)
    tr_np = np.asarray([r["side"] == "train" for r in rows])
    te_np = ~tr_np
    n_tr, n_te = int(tr_np.sum()), int(te_np.sum())
    assert n_te > 0, "pooled split has no test rows"
    d = int(asm_s["X"].shape[2])
    if not smoke:
        # estimator-validity regime statement (plan §7 G1' grounds): primal route
        assert n_tr > d, f"pooled-pair fit needs n_train > d, got n_train={n_tr} d={d}"
    corpus_te = np.asarray([r["corpus"] for r in rows])[te_np]
    print(
        f"[ladpool1336] pair={src}->{tgt} arm={arm} n={n} (train {n_tr} / test {n_te}) "
        f"layers={run_layers} n_boot={n_boot} null_draws={null_draws}",
        flush=True,
    )
    tr = torch.as_tensor(tr_np)
    te = torch.as_tensor(te_np)
    dev = fc._fit_device()
    dtype = torch.float64
    # Per-corpus paired bootstrap draw weights (seed independent of pair/arm).
    boot_w: dict[str, np.ndarray] = {}
    for c in corpora:
        n_c = int((corpus_te == c).sum())
        assert n_c > 0, f"no test rows for corpus {c!r}"
        idx = la.draw_index_matrix(
            n_c, n_boot, seed=POOLED_PAIR_BOOT_SEED_BASE + cm.v2_surface_index(c, "chat")
        )
        boot_w[c] = la.counts_from_indices(idx, n_c)
    idx_pooled = la.draw_index_matrix(n_te, n_boot, seed=POOLED_PAIR_BOOT_SEED_POOLED)
    w_pooled = la.counts_from_indices(idx_pooled, n_te)
    rng_null = np.random.default_rng(FIT_SEED + 77)
    perms = [rng_null.permutation(n) for _ in range(null_draws)]
    knn_rng = np.random.default_rng(KNN_SUBSAMPLE_SEED)
    keep = knn_rng.choice(n_te, size=min(n_te, args.knn_max_rows), replace=False)
    layers_out: dict[str, dict] = {}
    preds_store: dict[str, np.ndarray] = {
        "conv_ids": np.asarray([f"{r['corpus']}:{r['conv_id']}" for r in rows])[te_np],
        "corpus": corpus_te,
    }
    for li in run_layers:
        assert li < asm_s["X"].shape[1], f"layer {li} out of range ({asm_s['X'].shape[1]})"
        Xs_l = torch.as_tensor(asm_s["X"][:, li, :], dtype=dtype).to(dev)
        Ys_l = torch.as_tensor(asm_s["Y"][:, li, :], dtype=dtype).to(dev)
        Xt_l = torch.as_tensor(asm_t["X"][:, li, :], dtype=dtype).to(dev)
        Yt_l = torch.as_tensor(asm_t["Y"][:, li, :], dtype=dtype).to(dev)
        inner_seed = FIT_SEED + 4242  # the single pooled split == fold 0's seed convention
        preps = {
            "s": _v2_prep(Xs_l[tr], inner_seed=inner_seed, n_inner=cm.N_INNER_LAMBDA_FOLDS_V2),
            "t": _v2_prep(Xt_l[tr], inner_seed=inner_seed, n_inner=cm.N_INNER_LAMBDA_FOLDS_V2),
            "ys": _v2_prep(Ys_l[tr], inner_seed=inner_seed, n_inner=cm.N_INNER_LAMBDA_FOLDS_V2),
        }
        te_preds, aux = _fold_observed(Xs_l, Ys_l, Xt_l, Yt_l, tr, te, grid, preps)
        # Source's own pooled map read (diagonal ceiling at the SOURCE — the
        # panel's source==target fallback entry; the target's own read is
        # te_preds["within"]). One extra diagonal solve on the cached prep.
        fit_ws = _v2_yfit(preps["s"], Ys_l[tr], grid)
        own_source_te = _v2_predict(preps["s"], fit_ws, Xs_l[te]).cpu().numpy().astype(np.float32)
        yt_te = Yt_l[te].cpu().numpy().astype(np.float32)
        ys_te = Ys_l[te].cpu().numpy().astype(np.float32)
        xt_te = Xt_l[te].cpu().numpy().astype(np.float32)
        tier_te = {
            name: te_preds[name].cpu().numpy().astype(np.float32) for name in POOLED_PAIR_TIERS
        }
        # Shuffled-correspondence null (§4 div. 10; Y-permuted refits on cached bases).
        ss_null = np.zeros((null_draws, len(DRAWS_ORDER)))
        ss_tot = np.zeros(null_draws)
        for b, perm in enumerate(perms):
            ss_null[b], ss_tot[b] = _fold_null_r2_contrib(
                aux, Yt_l[torch.as_tensor(perm)], tr, te, grid
            )
        r2_null = 1.0 - ss_null / ss_tot[:, None]
        per_corpus: dict[str, dict] = {}
        for c in corpora:
            m = corpus_te == c
            blk = {
                "own": _pooled_pair_tier_block(tier_te["within"][m], yt_te[m], boot_w[c]),
                "own_source": _pooled_pair_tier_block(own_source_te[m], ys_te[m], boot_w[c]),
                "t0": _pooled_pair_tier_block(tier_te["t0"][m], yt_te[m], boot_w[c]),
                "t6": _pooled_pair_tier_block(tier_te["t6"][m], yt_te[m], boot_w[c]),
                "t8": _pooled_pair_tier_block(tier_te["t8"][m], yt_te[m], boot_w[c]),
                "n_test_rows": int(m.sum()),
            }
            per_corpus[c] = blk
        pooled_blk = {
            "own": _pooled_pair_tier_block(tier_te["within"], yt_te, w_pooled),
            "own_source": _pooled_pair_tier_block(own_source_te, ys_te, w_pooled),
            "t0": _pooled_pair_tier_block(tier_te["t0"], yt_te, w_pooled),
            "t6": _pooled_pair_tier_block(tier_te["t6"], yt_te, w_pooled),
            "t8": _pooled_pair_tier_block(tier_te["t8"], yt_te, w_pooled),
            "n_test_rows": n_te,
        }
        # Guideline-11 companions on the pooled test side (identity+learned-bias
        # shares dims by construction; kNN on the seeded row subsample).
        xs_tr = Xs_l[tr].cpu().numpy()
        ys_tr = Ys_l[tr].cpu().numpy()
        xt_tr = Xt_l[tr].cpu().numpy()
        yt_tr = Yt_l[tr].cpu().numpy()
        xs_te_np = Xs_l[te].cpu().numpy()
        id_within = identity_bias_predict(xt_tr, yt_tr, xt_te).astype(np.float32)
        id_t0 = identity_bias_predict(xs_tr, ys_tr, xt_te).astype(np.float32)
        baselines = {
            "knn_subsample_rows": int(len(keep)),
            "own": {
                "identity_bias_r2": float(fc._pooled_r2(id_within, yt_te)),
                "knn": _knn_block(tier_te["within"], yt_te, keep),
                "knn_identity_bias": _knn_block(id_within, yt_te, keep),
            },
            "t0": {
                "identity_bias_r2": float(fc._pooled_r2(id_t0, yt_te)),
                "knn": _knn_block(tier_te["t0"], yt_te, keep),
            },
            "own_source": {
                "identity_bias_r2": float(
                    fc._pooled_r2(
                        identity_bias_predict(xs_tr, ys_tr, xs_te_np).astype(np.float32), ys_te
                    )
                ),
            },
        }
        layers_out[str(li)] = {
            "per_corpus": per_corpus,
            "pooled": pooled_blk,
            "nulls": {
                "order": list(DRAWS_ORDER),
                "n_draws": int(null_draws),
                "r2_matrix": [[float(v) for v in row] for row in r2_null],
                "p975_per_read": [float(v) for v in np.nanquantile(r2_null, 0.975, axis=0)]
                if null_draws
                else [],
            },
            "selected_lambda": {k: float(v) for k, v in aux["selected_lams"].items()},
            "selectors": dict(aux["selectors"]),
            "procrustes": aux["procrustes"],
            "baselines": baselines,
        }
        for name, arr in (("within", tier_te["within"]), ("own_source", own_source_te)):
            preds_store[f"l{li}_{name}"] = arr.astype(np.float16)
        for name in ("t0", "t6", "t8"):
            preds_store[f"l{li}_{name}"] = tier_te[name].astype(np.float16)
        preds_store[f"l{li}_y_target"] = yt_te.astype(np.float16)
        print(
            f"[ladpool1336] layer {li}: pooled own={pooled_blk['own']['r2']:.4f} "
            f"t0={pooled_blk['t0']['r2']:.4f} t6={pooled_blk['t6']['r2']:.4f} "
            f"t8={pooled_blk['t8']['r2']:.4f}",
            flush=True,
        )
        del Xs_l, Ys_l, Xt_l, Yt_l, te_preds, aux, preps
    unit = f"{src}__{tgt}_arm_{arm}"
    preds_dir = args.preds_dir or Path(f"data/issue_1336/metric_ladder_pooled_v3{sfx}")
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / f"pair_{unit}.npz"
    np.savez(preds_path, **preds_store)  # plain savez: compression OFF for Xet (#813)
    sha = hashlib.sha256(preds_path.read_bytes()).hexdigest()
    manifest_path = preds_dir / "metric_ladder_pooled_v3_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[preds_path.name] = {
        "sha256": sha,
        "shapes": {k: list(np.asarray(v).shape) for k, v in preds_store.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    payload = {
        "metadata": _metadata(FIT_SEED, n),
        "pair": {"source": src, "target": tgt},
        "arm": arm,
        "x_slot": "context",
        "text_sources": {"source_side": src_text, "target_side": tgt},
        "split": {
            "manifest": str(man_path),
            "manifest_sha256": man_sha,
            "n_train": n_tr,
            "n_test": n_te,
        },
        "scale": "raw",  # pooled panel reads on the raw scale (n_train >> d; §7 G1' grounds)
        "layers": layers_out,
        "preds_npz": str(preds_path),
        "preds_sha256": sha,
    }
    _write_json(args.out_dir / "metric_ladder_pooled_v3" / f"pair_{unit}.json", payload)


def main() -> None:
    args = parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.pooled_pair:
        assert not (args.pair or args.pairs or args.cluster_delta_q), (
            "--pooled-pair is mutually exclusive with --pair/--pairs/--cluster-delta-q"
        )
        run_pooled_pair(args)
        return
    if args.cluster_delta_q:
        assert not (args.pair or args.pairs), (
            "--cluster-delta-q is mutually exclusive with --pair/--pairs"
        )
        cluster_delta_q_battery(args)
        return
    pairs = []
    if args.pairs:
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    if args.pair:
        pairs.append(args.pair)
    assert pairs, "--pair or --pairs is required (or --selfcheck / --cluster-delta-q)"
    bars = resolve_bars(args.s_qwen_v2, args.bars_json)
    prep_cache = PrepCache(capacity=6)
    for pair in pairs:
        run_pair(args, pair, bars=bars, prep_cache=prep_cache)
    print(f"[ladder1336] prep cache: {prep_cache.hits} hits / {prep_cache.misses} misses")


if __name__ == "__main__":
    main()
