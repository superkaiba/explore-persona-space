"""Issue #2378 P6 — 9-rung chat→target transfer ladder (unit 3 deliverable 2).

ONE direction only: chat → each of the 8 target cells (plain_text, 5 story-Q,
chat_user_real, chat_user_sim; the 4 dialogue targets DESCOPED at plan v7 —
epm:progress v70 clause 1), context arm, at the frozen layer
L*, under the SHARED fold map (story targets: family-held-out PRIMARY fold —
headline recovery labels carry ``family-held-out``).

Rung list inherited from ``scripts/issue2054_ladder.py`` (its ``_fit_ridge`` /
``_apply_ridge`` / ``_procrustes_apply`` cores are imported and used
verbatim). UNPAIRED-POOL OPERATIONALIZATION (plan §4.4: "pools are unpaired —
no intersection machinery; adaptation rungs train on the TARGET cell's train
folds"): the parent's rung-7/8 fits used PAIRED rows across cells
(A: x_T→x_S and B: y_S→y_T on shared conversations), which do not exist in
this design. The registered analogues, all fit on the TARGET's train fold with
the SOURCE map M frozen (every fit a parent-core GCV ridge, well-posed at
n_train > d):

  1_direct        M(x_t)                              (no target fit)
  2_ctx_offset    M(x_t − dx), dx = x̄_t − x̄_s       (means only; unpaired-safe)
  3_ans_offset    M(x_t) + dy, dy = ȳ_t − ȳ_s        (means only)
  4_bias_refit    M(x_t) + b*, b* = mean(y_t − M(x_t)) on target train
  5_global_scale  a·(M(x_t) − P̄) + ȳ_t              (parent scalar formula)
  6_rotation      ``_procrustes_apply``(M(x_t_tr), y_t_tr, M(x_t_te))
  7_ctx_reparam   G fit by GCV ridge x_t_tr → ẑ_tr where ẑ = M_inv(y_t_tr) and
                  M_inv is the SOURCE's own inverse map (ridge y_s→x_s on the
                  chat fold train pairs); pred = M(G(x_t_te)) + b7 (parent's
                  rung-7 bias recenter). The context side is re-mapped into the
                  chat map's input coordinates; supervision comes through the
                  target pairs via the source-side preimage (the unpaired
                  analogue of the parent's paired A-fit).
  8_ans_reparam   B fit by GCV ridge M(x_t_tr) → y_t_tr; pred = B(M(x_t_te))
                  (output-side re-map of the frozen map's predictions).
  9_full_refit    fresh GCV ridge x_t_tr → y_t_tr (the plan's prose name;
                  recovery ≈ 1 by construction — the sanity anchor).

Matched-capacity shuffled-pairing nulls (permute target train answers, plan
§4.4; pattern per ``issue2054_remap_pair_nulls.py``): computed at the REFIT
rungs 7/8/9 with 100 draws, SVD-once-per-fold batched (8/9 via the imported
``issue2054_fits._shuffled_answer_null_r2`` core; 7 via a same-shape batched
loop verified against a serial oracle in ``--phase probe``). Rung 4's
permutation null is ANALYTICALLY the true value (b* depends on y_t_tr only
through its mean — recorded, not simulated). Rungs 1/2/3 fit nothing
pairing-sensitive; rungs 5/6 are adaptations but not refit rungs (the plan's
null clause names refit rungs; the referenced parent pattern computed 7/8) —
each rung's JSON records its null disposition explicitly.

Recovery fractions (transfer R² / own ceiling on the target's held-out folds)
consume the fits driver's outputs (``fits/<cell>__context.json`` + the
``fits/percell`` rowstats sidecars) — run ``issue2378_fits.py`` for the target
cells FIRST. Skip-and-count denominator guards + tier suppression per plan §3;
a point-Unmappable target (U ≤ 0) skips its transfer rungs by the lattice rule
(a visible ``chat_to_<cell>__unmappable.json`` marker is written instead).

Phases: ``--phase pairs`` (default; ``--pairs`` shard axis), ``--phase h3``
(question-vs-dialogue paired contrast — TOMBSTONED at v7, refuses at entry;
dialogue descoped per epm:progress v70), ``--phase h4b`` (real-vs-sim paired
contrast on the intersection cohort, §4.2b asserts re-run), ``--phase probe``
(synthetic CPU self-verification incl. a full producer→consumer e2e through
``issue2378_fits.py`` outputs at tiny n/d).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue2054_fits as pf  # noqa: E402  (null core — plan §10 reuse row)
import issue2054_ladder as pl  # noqa: E402  (ridge/procrustes cores)
import issue2378_common as cm  # noqa: E402
import issue2378_p6_common as p6  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)

SCRIPT_VERSION = "issue2378_ladder_v1"

RUNGS: tuple[str, ...] = (
    "1_direct",
    "2_ctx_offset",
    "3_ans_offset",
    "4_bias_refit",
    "5_global_scale",
    "6_rotation",
    "7_ctx_reparam",
    "8_ans_reparam",
    "9_full_refit",
)
NULL_RUNGS = ("7_ctx_reparam", "8_ans_reparam", "9_full_refit")
RECOVERY_RUNGS_H3 = ("1_direct", "7_ctx_reparam")
SOURCE_CELL = "chat"

ESTIMATOR_NOTES = (
    "Unpaired-pool rung operationalization (module docstring): rungs 7/8/9 are fit on the "
    "TARGET's train folds through the frozen chat map (7: GCV ridge to M_inv preimages, then "
    "M + b7; 8: GCV ridge on M's predictions; 9: fresh full refit). Recovery values are NOT "
    "directly comparable to #2054's paired-row rung estimators."
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def target_cells(available: list[str]) -> list[str]:
    return [c for c in cm.ALL_CELLS if c != SOURCE_CELL and c in available]


# ---------------------------------------------------------------------------
# Rung math (parent cores; unpaired operationalization)
# ---------------------------------------------------------------------------


def _gcv_ridge_from_svd(
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    Yc: np.ndarray,
    *,
    lambdas: np.ndarray,
    dof_cap: float,
) -> tuple[np.ndarray, float]:
    """One GCV-ridge solve on a PRECOMPUTED design SVD (mirrors
    ``issue2054_ladder._fit_ridge``'s selection exactly; the batched-null inner
    step). Returns (W, best_lambda)."""
    s2 = s**2
    n_train = U.shape[0]
    UtY = U.T @ Yc
    row_energy = (UtY**2).sum(axis=1)
    tot_y_sq = float((Yc**2).sum())
    best_lam = float(lambdas[0])
    best_gcv = float("inf")
    for lam in lambdas:
        lam = float(lam)
        filt = s2 / (s2 + lam)
        dof = float(filt.sum())
        rss = tot_y_sq - float(((2 * filt - filt**2) * row_energy).sum())
        denom = (n_train - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if dof / n_train <= dof_cap and gcv < best_gcv:
            best_gcv = gcv
            best_lam = lam
    if best_gcv == float("inf"):
        best_lam = float(lambdas[-1])
    W = (Vt.T * (s / (s2 + best_lam))) @ UtY
    return W, best_lam


def compute_rungs_for_fold(
    model_M: dict,
    model_Minv: dict,
    xs_mean: np.ndarray,
    ys_mean: np.ndarray,
    Xt_tr: np.ndarray,
    Yt_tr: np.ndarray,
    Xt_te: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """All 9 rung predictions at the target's held-out rows (docstring table)."""
    Xt_tr64 = Xt_tr.astype(np.float64)
    Yt_tr64 = Yt_tr.astype(np.float64)
    Xt_te64 = Xt_te.astype(np.float64)
    P_tr = pl._apply_ridge(model_M, Xt_tr64)
    P_te = pl._apply_ridge(model_M, Xt_te64)
    dx = Xt_tr64.mean(axis=0) - xs_mean
    dy = Yt_tr64.mean(axis=0) - ys_mean

    bstar = (Yt_tr64 - P_tr).mean(axis=0)  # rung 4
    pmu = P_tr.mean(axis=0)
    ymu = Yt_tr64.mean(axis=0)
    Pc = P_tr - pmu
    Yc = Yt_tr64 - ymu
    denom = float((Pc**2).sum())
    a = float((Pc * Yc).sum() / denom) if denom > 1e-30 else 1.0  # rung 5

    rot_te, resid_max = pl._procrustes_apply(P_tr, Yt_tr64, P_te)  # rung 6

    # Rung 7: context-side re-map through the frozen M (preimage supervision).
    Zhat_tr = pl._apply_ridge(model_Minv, Yt_tr64)
    model_G = pl._fit_ridge(Xt_tr64, Zhat_tr)
    P7_tr = pl._apply_ridge(model_M, pl._apply_ridge(model_G, Xt_tr64))
    P7_te = pl._apply_ridge(model_M, pl._apply_ridge(model_G, Xt_te64))
    b7 = (Yt_tr64 - P7_tr).mean(axis=0)

    model_B = pl._fit_ridge(P_tr, Yt_tr64)  # rung 8
    model_F = pl._fit_ridge(Xt_tr64, Yt_tr64)  # rung 9 (full refit)

    preds = {
        "1_direct": P_te,
        "2_ctx_offset": pl._apply_ridge(model_M, Xt_te64 - dx),
        "3_ans_offset": P_te + dy,
        "4_bias_refit": P_te + bstar,
        "5_global_scale": a * (P_te - pmu) + ymu,
        "6_rotation": rot_te,
        "7_ctx_reparam": P7_te + b7,
        "8_ans_reparam": pl._apply_ridge(model_B, P_te),
        "9_full_refit": pl._apply_ridge(model_F, Xt_te64),
    }
    info = {
        "source_fit": model_M["info"],
        "source_inverse_fit": model_Minv["info"],
        "ctx_reparam_fit": model_G["info"],
        "ans_reparam_fit": model_B["info"],
        "full_refit_fit": model_F["info"],
        "global_scale_a": a,
        "procrustes_resid_max": resid_max,
    }
    return preds, info


def null_ctx_reparam_r2(
    model_M: dict,
    Zhat_tr: np.ndarray,
    Xt_tr: np.ndarray,
    Yt_tr: np.ndarray,
    Xt_te: np.ndarray,
    Yt_te: np.ndarray,
    *,
    n_draws: int,
    seed: int,
    lambdas: np.ndarray = pl.DEFAULT_LAMBDAS,
    dof_cap: float = pl.DEFAULT_DOF_CAP,
) -> np.ndarray:
    """Rung-7 shuffled-pairing null, SVD-once batched (the
    ``issue2054_remap_pair_nulls`` pattern): permute the PREIMAGE TARGETS
    ẑ = M_inv(y_t_tr) (row-wise, so permuting y_t_tr rows == permuting ẑ rows),
    refit G on the shared eigenbasis per draw, compose through the frozen M,
    apply the (draw-invariant) b7 recenter, score against the true held-out."""
    Xtr64 = Xt_tr.astype(np.float64)
    Xte64 = Xt_te.astype(np.float64)
    xmu = Xtr64.mean(axis=0)
    xsd = Xtr64.std(axis=0) + 1e-9
    Xs = (Xtr64 - xmu) / xsd
    Xe = (Xte64 - xmu) / xsd
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    Z = Zhat_tr.astype(np.float64)
    zmu = Z.mean(axis=0)
    Zc = Z - zmu
    # b7 = ȳ_t − M(z̄): the train mean of ridge preds equals the target train
    # mean exactly (standardize+center identity), and z̄ is permutation-
    # invariant, so b7 is CONSTANT across draws (verified vs the serial oracle
    # in --phase probe).
    b7 = Yt_tr.astype(np.float64).mean(axis=0) - pl._apply_ridge(model_M, zmu[None, :])[0]
    rng = np.random.default_rng(seed)
    r2s = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        perm = rng.permutation(Zc.shape[0])
        W_g, _lam = _gcv_ridge_from_svd(U, s, Vt, Zc[perm], lambdas=lambdas, dof_cap=dof_cap)
        g_te = Xe @ W_g + zmu
        p7 = pl._apply_ridge(model_M, g_te)
        r2s[i] = pf._r2_matrix(Yt_te, p7 + b7)
    return r2s


# ---------------------------------------------------------------------------
# Per-pair unit
# ---------------------------------------------------------------------------


def _fits_inputs(ledger_root: Path, cell: str) -> tuple[dict, dict]:
    """Target cell's fits JSON + rowstats (recovery denominators). REQUIRED —
    the fits units for the target (and chat) run first (unit-5 sequencing)."""
    fpath = ledger_root / "fits" / f"{cell}__context.json"
    if not fpath.exists():
        raise RuntimeError(
            f"missing {fpath} — run issue2378_fits.py for cell {cell} BEFORE the ladder "
            "(recovery denominators consume the fits outputs)"
        )
    fits = json.loads(fpath.read_text(encoding="utf-8"))
    rs = p6.load_rowstats(ledger_root / "fits" / "percell" / f"{cell}__context__rowstats.npz")
    return fits, rs


def _pair_regime(args, fold_map: dict, target: str, layer: int) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "source": SOURCE_CELL,
        "target": target,
        "arm": "context",
        "layer": int(layer),
        "k": fold_map["k"],
        "seed": fold_map["seed"],
        "fold_map_sha": fold_map["sha256"],
        "n_null_draws": int(args.n_null_draws),
        "bootstrap_draws": int(args.bootstrap_draws),
        "rungs": list(RUNGS),
        "seed_derivation": "137-rooted per-(pair,rung,fold) via cm.derived_seed",
    }


class _SourceMemo:
    """Chat-side per-fold memo: arrays + M (x_s→y_s) + M_inv (y_s→x_s) fits are
    shared across all 12 pairs (the parent's M-memo pattern)."""

    def __init__(self, store_root: Path, fold_map: dict, layer: int):
        entry = fold_map["cells"][SOURCE_CELL]
        pack = p6.load_cell_arrays(
            store_root, SOURCE_CELL, layer, ("v_C", p6.ANSWER_SLOT), row_order=entry["row_ids"]
        )
        self.X = pack["arrays"]["v_C"]
        self.Y = pack["arrays"][p6.ANSWER_SLOT]
        self.splits = p6.fold_splits(entry)
        self._m: dict[int, dict] = {}
        self._minv: dict[int, dict] = {}

    def fold(self, f: int) -> dict:
        if f not in self._m:
            tr = self.splits[f][0]
            Xtr, Ytr = self.X[tr].astype(np.float64), self.Y[tr].astype(np.float64)
            t0 = time.time()
            self._m[f] = pl._fit_ridge(Xtr, Ytr)
            self._minv[f] = pl._fit_ridge(Ytr, Xtr)
            _log(f"[ladder] source fold {f}: M + M_inv fit in {time.time() - t0:.1f}s")
        tr = self.splits[f][0]
        return {
            "M": self._m[f],
            "Minv": self._minv[f],
            "xs_mean": self.X[tr].astype(np.float64).mean(axis=0),
            "ys_mean": self.Y[tr].astype(np.float64).mean(axis=0),
        }


def run_pair_unit(args, fold_map: dict, memo: _SourceMemo, target: str, layer: int) -> None:
    ledger_root = Path(args.ledger_root)
    out_dir = ledger_root / "ladder"
    regime = _pair_regime(args, fold_map, target, layer)
    # Resume keys on the LAST-written artifact (r1 review g3 concern 1: rung1
    # is written FIRST of 9, so a crash mid-loop left a partial unit every
    # re-run then skipped as complete). The unmappable marker is the other
    # terminal artifact of this unit.
    done_path = out_dir / f"chat_to_{target}__rung{len(RUNGS)}.json"
    unmappable_path = out_dir / f"chat_to_{target}__unmappable.json"
    for prior_path in (done_path, unmappable_path):
        if prior_path.exists():
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            if prior.get("regime") == regime:
                _log(f"[ladder] SKIP chat->{target}: outputs exist with matching regime")
                return
            raise RuntimeError(f"regime mismatch at {prior_path} — use a fresh ledger root")
    store_root = Path(args.store_root)
    if target in cm.USER_CELLS:
        p6.assert_user_pair(store_root, fold_map, layer)  # §4.2b, before any paired read
    fits, fits_rs = _fits_inputs(ledger_root, target)
    floor = float(fits["floor"])
    tier = fits["tier"]
    # POOLED point (r1 review g3 concern 2): the tier is assigned from the
    # POOLED-convention margin CI, so the U <= 0 branch uses the same
    # statistic family — never fold-mean while the tier reads pooled.
    point_u = float(fits["margin"]["point_pooled"])
    entry = fold_map["cells"][target]
    if point_u <= 0.0:
        marker = {
            "regime": regime,
            "verdict": "unmappable-target",
            "reason": (
                f"point U = pooled own-ceiling R2 - floor = {point_u:+.4f} <= 0: the "
                "registered Unmappable-target lattice branch — transfer rungs are skipped "
                "for this target by the lattice rule (plan §3/§8)"
            ),
            "own_fold_mean_r2": fits["fold_mean_r2"],
            "floor": floor,
            "tier": tier,
            "metadata": cm.run_metadata(),
        }
        cm.atomic_write_json(out_dir / f"chat_to_{target}__unmappable.json", marker)
        _log(f"[ladder] chat->{target}: UNMAPPABLE target (U={point_u:+.4f}) — rungs skipped")
        return
    if fits_rs["row_ids"].tolist() != entry["row_ids"]:
        raise RuntimeError(f"{target}: fits rowstats row order != fold map (mixed generations)")
    pack = p6.load_cell_arrays(
        store_root, target, layer, ("v_C", p6.ANSWER_SLOT), row_order=entry["row_ids"]
    )
    Xt, Yt = pack["arrays"]["v_C"], pack["arrays"][p6.ANSWER_SLOT]
    n, d = Xt.shape
    ybar = Yt.astype(np.float64).mean(axis=0)
    ss_tot = ((Yt.astype(np.float64) - ybar) ** 2).sum(axis=1)
    splits = p6.fold_splits(entry)
    ss_res = {r: np.full(n, np.nan) for r in RUNGS}
    preds_all = {r: np.empty((n, Yt.shape[1]), dtype=np.float32) for r in RUNGS}
    folds_of = np.full(n, -1, dtype=np.int64)
    per_fold: dict[str, list[dict]] = {r: [] for r in RUNGS}
    null_draws: dict[str, list[list[float]]] = {r: [] for r in NULL_RUNGS}
    fold_infos: list[dict] = []
    t_unit = time.time()
    for f, (tr, te) in enumerate(splits):
        n_train = int(tr.size)
        if n_train <= d:
            raise RuntimeError(f"{target} fold {f}: n_train={n_train} <= d={d} (under-determined)")
        src = memo.fold(f)
        Xt_tr, Yt_tr, Xt_te, Yt_te = Xt[tr], Yt[tr], Xt[te], Yt[te]
        t0 = time.time()
        preds, info = compute_rungs_for_fold(
            src["M"], src["Minv"], src["xs_mean"], src["ys_mean"], Xt_tr, Yt_tr, Xt_te
        )
        t_rungs = time.time() - t0
        fold_infos.append(info)
        for r in RUNGS:
            r2 = pf._r2_matrix(Yt_te, preds[r])
            per_fold[r].append(
                {"fold": f, "n_train": n_train, "n_eval": int(te.size), "r2": float(r2)}
            )
            ss_res[r][te] = ((Yt_te.astype(np.float64) - preds[r]) ** 2).sum(axis=1)
            preds_all[r][te] = preds[r].astype(np.float32)
        folds_of[te] = f
        t0 = time.time()
        P_tr = pl._apply_ridge(src["M"], Xt_tr.astype(np.float64))
        P_te = pl._apply_ridge(src["M"], Xt_te.astype(np.float64))
        n8, _ = pf._shuffled_answer_null_r2(
            P_tr,
            Yt_tr,
            P_te,
            Yt_te,
            n_draws=args.n_null_draws,
            seed=p6.unit_seed(target, "null8", f),
        )
        n9, _ = pf._shuffled_answer_null_r2(
            Xt_tr,
            Yt_tr,
            Xt_te,
            Yt_te,
            n_draws=args.n_null_draws,
            seed=p6.unit_seed(target, "null9", f),
        )
        Zhat_tr = pl._apply_ridge(src["Minv"], Yt_tr.astype(np.float64))
        n7 = null_ctx_reparam_r2(
            src["M"],
            Zhat_tr,
            Xt_tr,
            Yt_tr,
            Xt_te,
            Yt_te,
            n_draws=args.n_null_draws,
            seed=p6.unit_seed(target, "null7", f),
        )
        null_draws["7_ctx_reparam"].append([float(x) for x in n7])
        null_draws["8_ans_reparam"].append([float(x) for x in n8])
        null_draws["9_full_refit"].append([float(x) for x in n9])
        _log(
            f"[ladder] chat->{target} fold {f + 1}/{len(splits)}: "
            f"r2_direct={per_fold['1_direct'][-1]['r2']:+.4f} "
            f"r2_ctx_reparam={per_fold['7_ctx_reparam'][-1]['r2']:+.4f} "
            f"rungs={t_rungs:.1f}s nulls={time.time() - t0:.1f}s"
        )
    # Identity baseline for the target cell (rung-independent; the fitted-map
    # baselines pair — identity values also live in the cell's fits JSON).
    id_res = np.full(n, np.nan)
    for f, (tr, te) in enumerate(splits):
        pred_id = identity_bias_predict(Xt[tr], Yt[tr], Xt[te])
        id_res[te] = ((Yt[te].astype(np.float64) - pred_id) ** 2).sum(axis=1)
    identity_pooled = p6.pooled_r2(id_res, ss_tot)
    fold_label = (
        "family-held-out"
        if entry["fold_structure"] == "family-held-out"
        else entry["fold_structure"]
    )
    ceiling_pooled = float(fits["pooled_r2"])
    ceiling_fold_mean = float(fits["fold_mean_r2"])
    suppress_by_tier = tier != "clearly-mappable"
    for ri, r in enumerate(RUNGS, start=1):
        pooled = p6.pooled_r2(ss_res[r], ss_tot)
        fold_mean = float(np.mean([x["r2"] for x in per_fold[r]]))
        rung_out: dict = {
            "regime": regime,
            "pair": f"chat->{target}",
            "rung_index": ri,
            "rung": r,
            "fold_structure": entry["fold_structure"],
            "headline_fold_label": fold_label,
            # Plan §4.4 literal: the family-fold audit verdict persists in
            # every story ladder JSON too (r1 review g3 concern 5; None for
            # non-story cells, which carry no audit).
            "story_fold_audit": entry.get("story_fold_audit"),
            "per_fold": per_fold[r],
            "fold_mean_r2": fold_mean,
            "pooled_r2": pooled,
            "ceiling": {
                "pooled_r2": ceiling_pooled,
                "fold_mean_r2": ceiling_fold_mean,
                "floor": floor,
                "tier": tier,
                "fits_json": str(ledger_root / "fits" / f"{target}__context.json"),
            },
            "estimator_notes": ESTIMATOR_NOTES,
        }
        if r in NULL_RUNGS:
            pooled_null = [x for fd in null_draws[r] for x in fd]
            rung_out["null"] = {
                "kind": "shuffled-pairing matched-capacity (permute target train answers)",
                "n_draws_per_fold": int(args.n_null_draws),
                "per_fold_draws": null_draws[r],
                "pooled_p95": float(np.percentile(pooled_null, 95)),
                "pooled_median": float(np.median(pooled_null)),
            }
        elif r == "4_bias_refit":
            rung_out["null"] = {
                "kind": "analytic-invariance",
                "note": (
                    "permutation-invariant: b* depends on target train answers only through "
                    "their mean, so every shuffled-pairing draw equals the true value"
                ),
            }
        else:
            rung_out["null"] = {
                "kind": "not-computed",
                "note": (
                    "not a refit rung (plan §4.4 names refit rungs; the referenced "
                    "issue2054_remap_pair_nulls pattern computed the re-map rungs); "
                    "rungs 2/3 use permutation-invariant means, 1 fits nothing, 5/6 are "
                    "non-refit adaptations"
                ),
            }
        if suppress_by_tier:
            rung_out["recovery"] = {
                "suppressed_by_tier": True,
                "tier": tier,
                "note": (
                    "ratio verdicts suppressed (plan §3 reporting tiers): own-ceiling R2 and "
                    "transfer R2 are reported separately, absolute and unnormalized"
                ),
            }
        else:
            rec = p6.recovery_bootstrap(
                ss_res[r],
                fits_rs["ss_res"],
                ss_tot,
                floor=floor,
                n_draws=args.bootstrap_draws,
                seed=p6.unit_seed(target, r, "recovery"),
            )
            # A draws-suppressed ratio verdict exposes NO quotable point ratio
            # either (r1 review g4 concern 2, mirrored here for cross-unit
            # consistency): ceiling + transfer stay reported separately.
            if not rec.get("suppressed"):
                rec["point_pooled"] = pooled / ceiling_pooled
                rec["point_fold_mean"] = fold_mean / ceiling_fold_mean
            rec["suppressed_by_tier"] = False
            rec["label"] = fold_label
            rung_out["recovery"] = rec
        rung_out["baselines"] = {
            "identity_bias_pooled_r2": identity_pooled,
            "knn": {
                metric: knn_retrieval(preds_all[r], Yt, ks=(1, 5, 10), metric=metric)
                for metric in ("euclidean", "cosine")
            },
        }
        rs_path = out_dir / "percell" / f"chat_to_{target}__rung{ri}__rowstats.npz"
        p6.write_rowstats(
            rs_path,
            row_ids=entry["row_ids"],
            folds=folds_of,
            ss_res=ss_res[r],
            ss_tot=ss_tot,
        )
        pr_path = out_dir / "preds" / f"chat_to_{target}__rung{ri}__preds.npz"
        p6.write_preds(pr_path, row_ids=entry["row_ids"], folds=folds_of, preds=preds_all[r])
        rung_out["rowstats_path"] = str(rs_path)
        rung_out["preds_path"] = str(pr_path)
        rung_out["retrieval_seam"] = {
            "preds": str(pr_path),
            "pool": "the target cell's pooled held-out answers",
            "chance_at_1": 1.0 / n,
            "note": "full retrieval battery (4 conventions + CSLS) is unit 4",
        }
        rung_out["fold_fit_infos"] = fold_infos if ri == 1 else "see rung1 JSON"
        rung_out["unit_wall_s"] = round(time.time() - t_unit, 2)
        rung_out["metadata"] = cm.run_metadata()
        cm.atomic_write_json(out_dir / f"chat_to_{target}__rung{ri}.json", rung_out)
    _log(
        f"[ladder] chat->{target}: done in {time.time() - t_unit:.1f}s "
        f"(direct pooled r2={p6.pooled_r2(ss_res['1_direct'], ss_tot):+.4f}, tier={tier})"
    )


# ---------------------------------------------------------------------------
# H3 — question-vs-dialogue paired contrast (plan §3 H3)
# ---------------------------------------------------------------------------

H3_CHARACTERS = ("astra", "helios", "dana", "vex")


def _cell_recovery_inputs(ledger_root: Path, cell: str, rung_index: int) -> dict:
    fits, fits_rs = _fits_inputs(ledger_root, cell)
    lad_rs = p6.load_rowstats(
        ledger_root / "ladder" / "percell" / f"chat_to_{cell}__rung{rung_index}__rowstats.npz"
    )
    if fits_rs["row_ids"].tolist() != lad_rs["row_ids"].tolist():
        raise RuntimeError(f"{cell}: fits vs ladder rowstats row order mismatch")
    return {
        "tier": fits["tier"],
        "floor": float(fits["floor"]),
        "ceil_res": fits_rs["ss_res"],
        "tran_res": lad_rs["ss_res"],
        "ss_tot": fits_rs["ss_tot"],
    }


def phase_h3(args) -> int:
    """Registered H3 statistic: equal-weight mean over the four characters of
    the per-character (question-arm recovery − dialogue-arm recovery) gap, at
    the direct and ctx-re-map rungs; scenes resampled independently WITHIN each
    fixed character cell (200 draws; the panel is fixed, never resampled).
    A pair whose question or dialogue cell is not clearly-mappable is dropped
    (disclosed); < 3 surviving pairs ⇒ indeterminate (plan §3). Per-pair cell
    availability checks the G2b drop marker BEFORE fit existence (r3 reconciler
    blocker g2b-drop-marker-shadowed-by-stale-fit, swept here): a G2b-dropped
    cell routes to the disclosed dropped-pair path even when a stale
    git-re-materialized `__context.json` coexists, with `--survivors` keying
    marker authority to the CURRENT dispatch via :func:`p6.g2b_dropped_now`.

    TOMBSTONED at plan v7 (Amendment record A, epm:progress v70 clause 1): the
    dialogue family is descoped, so the question-vs-dialogue contrast has no
    active dialogue arm — this phase REFUSES at entry. The body is retained
    UNREACHABLE for archival readers of the pre-v7 design; dispatch no longer
    schedules a p6.h3 step."""
    raise SystemExit(
        "phase_h3 is DESCOPED at plan v7 (dialogue family dropped — epm:progress v70 "
        "clause 1 / plan Amendment record A): the question-vs-dialogue contrast has "
        "no active dialogue arm. No h3_question_vs_dialogue.json is produced."
    )
    ledger_root = Path(args.ledger_root)
    surv_set = p6.parse_survivors(args.survivors)
    out: dict = {
        "statistic": (
            "equal-weight mean over characters of (question-arm recovery − dialogue-arm "
            "recovery), per rung; scene-grain independent-within-cell bootstrap"
        ),
        "characters": list(H3_CHARACTERS),
        "rungs": {},
        "estimator_notes": ESTIMATOR_NOTES,
    }
    for rung_name in RECOVERY_RUNGS_H3:
        ri = RUNGS.index(rung_name) + 1
        cells: dict[str, dict] = {}
        surviving: list[str] = []
        dropped: dict[str, str] = {}
        for ch in H3_CHARACTERS:
            q_cell, d_cell = f"storyq_{ch}", f"dialog_{ch}"
            # G2b drop marker checked BEFORE fit existence (r3 reconciler
            # blocker g2b-drop-marker-shadowed-by-stale-fit): a coexisting
            # stale fit must never resurrect a dropped cell's pair.
            g2b_gone = [
                c for c in (q_cell, d_cell) if p6.g2b_dropped_now(ledger_root / "fits", c, surv_set)
            ]
            missing = [
                c
                for c in (q_cell, d_cell)
                if c not in g2b_gone and not (ledger_root / "fits" / f"{c}__context.json").exists()
            ]
            if g2b_gone or missing:
                parts = []
                if g2b_gone:
                    parts.append(
                        "G2b-dropped this run (drop marker authoritative over "
                        f"any stale fit): {g2b_gone}"
                    )
                if missing:
                    parts.append(f"cells missing from fits outputs: {missing}")
                dropped[ch] = "; ".join(parts)
                continue
            # Tier + Unmappable-skip checks run BEFORE any ladder rowstats
            # load (r1 review g3 concern 3): an Unmappable-skipped target has
            # NO rung rowstats, and must route to the registered dropped-pair
            # path instead of crashing H3.
            bad_tier, unmappable = [], []
            for c in (q_cell, d_cell):
                fits_json = json.loads(
                    (ledger_root / "fits" / f"{c}__context.json").read_text(encoding="utf-8")
                )
                if fits_json["tier"] != "clearly-mappable":
                    bad_tier.append(c)
                if (ledger_root / "ladder" / f"chat_to_{c}__unmappable.json").exists():
                    unmappable.append(c)
            if bad_tier or unmappable:
                parts = []
                if bad_tier:
                    parts.append(f"not clearly-mappable: {bad_tier}")
                if unmappable:
                    parts.append(f"unmappable-skipped (no ladder rungs): {unmappable}")
                dropped[ch] = "; ".join(parts)
                continue
            qi = _cell_recovery_inputs(ledger_root, q_cell, ri)
            di = _cell_recovery_inputs(ledger_root, d_cell, ri)
            cells[ch] = {"q": qi, "d": di}
            surviving.append(ch)
        n_draws = int(args.bootstrap_draws)
        draws = np.full(n_draws, np.nan)
        skip_counts: dict[str, int] = {}
        point_gaps = {}
        for ch in surviving:
            q, dd = cells[ch]["q"], cells[ch]["d"]
            rq = (1 - q["tran_res"].sum() / q["ss_tot"].sum()) / (
                1 - q["ceil_res"].sum() / q["ss_tot"].sum()
            )
            rd = (1 - dd["tran_res"].sum() / dd["ss_tot"].sum()) / (
                1 - dd["ceil_res"].sum() / dd["ss_tot"].sum()
            )
            point_gaps[ch] = {
                "q_recovery": float(rq),
                "d_recovery": float(rd),
                "gap": float(rq - rd),
            }
        if surviving:
            rngs = {
                (ch, side): np.random.default_rng(p6.unit_seed("h3", rung_name, ch, side))
                for ch in surviving
                for side in ("q", "d")
            }
            for t in range(n_draws):
                gaps = []
                ok = True
                for ch in surviving:
                    per_side = {}
                    for side in ("q", "d"):
                        inp = cells[ch][side]
                        m = inp["ss_tot"].shape[0]
                        idx = rngs[(ch, side)].integers(0, m, size=m)
                        tot = inp["ss_tot"][idx].sum()
                        ceil = 1 - inp["ceil_res"][idx].sum() / tot
                        if not np.isfinite(ceil) or ceil <= inp["floor"]:
                            key = f"storyq_{ch}" if side == "q" else f"dialog_{ch}"
                            skip_counts[key] = skip_counts.get(key, 0) + 1
                            ok = False
                            break
                        per_side[side] = (1 - inp["tran_res"][idx].sum() / tot) / ceil
                    if not ok:
                        break
                    gaps.append(per_side["q"] - per_side["d"])
                if ok and gaps:
                    draws[t] = float(np.mean(gaps))
        valid = draws[np.isfinite(draws)]
        suppressed = valid.size < int(np.ceil(p6.VALID_DRAW_FRAC * n_draws))
        rung_block: dict = {
            "rung_index": ri,
            "surviving_pairs": surviving,
            "n_surviving_pairs": len(surviving),
            "dropped_pairs": dropped,
            "indeterminate": len(surviving) < 3,
            "point_per_character": point_gaps,
            "point_mean_gap": (
                float(np.mean([point_gaps[ch]["gap"] for ch in surviving])) if surviving else None
            ),
            "n_draws": n_draws,
            "n_valid_draws": int(valid.size),
            "n_skipped_draws": int(n_draws - valid.size),
            "skip_counts_by_cell": skip_counts,
            "suppressed_by_draws": bool(suppressed),
            "gap_draws": [float(x) for x in draws],
        }
        if valid.size and not suppressed:
            rung_block["ci_lo"] = float(np.percentile(valid, 2.5))
            rung_block["ci_hi"] = float(np.percentile(valid, 97.5))
            rung_block["median"] = float(np.median(valid))
        out["rungs"][rung_name] = rung_block
        _log(
            f"[h3] {rung_name}: pairs={len(surviving)} point_gap="
            f"{rung_block['point_mean_gap']} valid={valid.size} indet={rung_block['indeterminate']}"
        )
    out["interpretation_registered"] = (
        "CI wholly > 0 at either rung: question->answer specificity; zero-straddling: no "
        "detected difference within this fixed 4-character panel at this n (NEVER generality); "
        "CI wholly < 0: informative surprise (plan §3 — verdict prose is the analyzer's)"
    )
    out["metadata"] = cm.run_metadata()
    cm.atomic_write_json(ledger_root / "ladder" / "h3_question_vs_dialogue.json", out)
    return 0


# ---------------------------------------------------------------------------
# H4b — real-vs-sim paired contrast on the intersection cohort (plan §4.2b)
# ---------------------------------------------------------------------------


def phase_h4b(args) -> int:
    ledger_root = Path(args.ledger_root)
    fm = _fold_map(args)
    layer = resolve_layer(args)
    pair_diag = p6.assert_user_pair(Path(args.store_root), fm, layer)  # §4.2b re-run
    real_fits, real_rs = _fits_inputs(ledger_root, "chat_user_real")
    sim_fits, sim_rs = _fits_inputs(ledger_root, "chat_user_sim")
    if real_rs["row_ids"].tolist() != sim_rs["row_ids"].tolist():
        raise RuntimeError("§4.2b assert FAILED: user-arm rowstats conversation lists differ")
    n = real_rs["ss_tot"].shape[0]
    n_draws = int(args.bootstrap_draws)
    rng = np.random.default_rng(p6.unit_seed("h4b", "ceiling"))
    idx = rng.integers(0, n, size=(n_draws, n))  # SHARED draws — paired by construction
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_real = 1.0 - real_rs["ss_res"][idx].sum(axis=1) / real_rs["ss_tot"][idx].sum(axis=1)
        r2_sim = 1.0 - sim_rs["ss_res"][idx].sum(axis=1) / sim_rs["ss_tot"][idx].sum(axis=1)
    delta = r2_sim - r2_real
    finite = delta[np.isfinite(delta)]
    out: dict = {
        "statistic": "paired Δ(sim − real), conversation-grouped bootstrap on the intersection",
        "intersection": fm["user_intersection"],
        "pair_assert": pair_diag,
        "ceiling_delta": {
            "point_pooled": float(sim_fits["pooled_r2"] - real_fits["pooled_r2"]),
            "point_fold_mean": float(sim_fits["fold_mean_r2"] - real_fits["fold_mean_r2"]),
            "real_pooled_r2": real_fits["pooled_r2"],
            "sim_pooled_r2": sim_fits["pooled_r2"],
            "n_draws": n_draws,
            "delta_draws": [float(x) for x in delta],
            "ci_lo": float(np.percentile(finite, 2.5)),
            "ci_hi": float(np.percentile(finite, 97.5)),
            "median": float(np.median(finite)),
        },
        "tiers": {"real": real_fits["tier"], "sim": sim_fits["tier"]},
        "registered_alternatives": (
            "operational contrast ONLY — 'tracks the model's simulation' and "
            "'representational failure' stay REGISTERED UNRESOLVED ALTERNATIVES (plan §3 H4b)"
        ),
        "recovery_delta": {},
    }
    both_mappable = all(t == "clearly-mappable" for t in out["tiers"].values())
    for rung_name in RECOVERY_RUNGS_H3:
        ri = RUNGS.index(rung_name) + 1
        if not both_mappable:
            out["recovery_delta"][rung_name] = {
                "suppressed_by_tier": True,
                "tiers": out["tiers"],
                "note": "recovery Δ computed only where BOTH arms are clearly mappable (plan §3)",
            }
            continue
        real_in = _cell_recovery_inputs(ledger_root, "chat_user_real", ri)
        sim_in = _cell_recovery_inputs(ledger_root, "chat_user_sim", ri)
        rng = np.random.default_rng(p6.unit_seed("h4b", "recovery", rung_name))
        idx = rng.integers(0, n, size=(n_draws, n))
        draws = np.full(n_draws, np.nan)
        skips = {"chat_user_real": 0, "chat_user_sim": 0}
        for t in range(n_draws):
            rec = {}
            ok = True
            for name, inp in (("chat_user_real", real_in), ("chat_user_sim", sim_in)):
                tot = inp["ss_tot"][idx[t]].sum()
                ceil = 1 - inp["ceil_res"][idx[t]].sum() / tot
                if not np.isfinite(ceil) or ceil <= inp["floor"]:
                    skips[name] += 1
                    ok = False
                    break
                rec[name] = (1 - inp["tran_res"][idx[t]].sum() / tot) / ceil
            if ok:
                draws[t] = rec["chat_user_sim"] - rec["chat_user_real"]
        valid = draws[np.isfinite(draws)]
        suppressed = valid.size < int(np.ceil(p6.VALID_DRAW_FRAC * n_draws))
        block = {
            "rung_index": ri,
            "n_draws": n_draws,
            "n_valid": int(valid.size),
            "skips_by_arm": skips,
            "suppressed_by_draws": bool(suppressed),
            "delta_draws": [float(x) for x in draws],
        }
        if valid.size and not suppressed:
            block["ci_lo"] = float(np.percentile(valid, 2.5))
            block["ci_hi"] = float(np.percentile(valid, 97.5))
            block["median"] = float(np.median(valid))
        out["recovery_delta"][rung_name] = block
    out["metadata"] = cm.run_metadata()
    cm.atomic_write_json(ledger_root / "ladder" / "h4b_real_vs_sim.json", out)
    _log(f"[h4b] ceiling Δ point={out['ceiling_delta']['point_pooled']:+.4f}")
    return 0


# ---------------------------------------------------------------------------
# Phases / CLI
# ---------------------------------------------------------------------------


def resolve_layer(args) -> int:
    if args.layer is not None:
        return int(args.layer)
    path = Path(args.layer_star_from or (Path(args.ledger_root) / "pilot" / "layer_sweep.json"))
    if not path.exists():
        raise RuntimeError(
            f"cannot resolve the read layer: pass --layer or --layer-star-from (missing {path})"
        )
    return int(json.loads(path.read_text(encoding="utf-8"))["selected_layer"])


def _fold_map(args) -> dict:
    return p6.load_or_build_fold_map(
        Path(args.store_root), Path(args.ledger_root), **getattr(args, "fold_floors_override", {})
    )


def phase_pairs(args) -> int:
    ledger_root = Path(args.ledger_root)
    fm = _fold_map(args)
    gate_path = Path(args.g3_gate_file or (ledger_root / p6.G3_GATE_NAME))
    p6.require_g3_pass(gate_path)  # plan §7: G3 gates the ladder fan-out
    layer = resolve_layer(args)
    available = sorted(fm["cells"])
    if args.pairs == "all":
        targets = target_cells(available)
    else:
        targets = [t.strip() for t in args.pairs.split(",") if t.strip()]
        bad = [t for t in targets if t not in available or t == SOURCE_CELL]
        if bad:
            raise SystemExit(f"unknown/unavailable ladder targets: {bad}")
    if not targets:
        raise SystemExit("empty ladder target set")
    _log(f"[ladder] {len(targets)} pairs: chat -> {targets}")
    memo = _SourceMemo(Path(args.store_root), fm, layer)
    t0 = time.time()
    for i, tgt in enumerate(targets):
        run_pair_unit(args, fm, memo, tgt, layer)
        cm.progress("ladder", i + 1, len(targets), f"chat->{tgt}", t0)
    return 0


def phase_probe(args) -> int:  # noqa: PLR0915
    """Synthetic CPU self-verification: (1) rung battery shape + discrimination
    sanity on planted geometry; (2) rung-7 batched null vs a serial per-draw
    oracle; (3) full producer→consumer e2e — a tiny synthetic store run through
    ``issue2378_fits.py`` (g3 + all context fits), then this driver's pairs +
    h4b phases (h3: tombstone-refusal assert — v7 dialogue descope);
    (4) recovery skip-and-count via a planted degenerate
    ceiling (covered in the fits probe; re-checked here through the ladder's
    tier-suppression path)."""
    import issue2378_fits as fits_mod

    rng = np.random.default_rng(23)
    n_tr, n_te, d = 80, 30, 8
    W_true = rng.standard_normal((d, d)) / np.sqrt(d)
    Xs_tr = rng.standard_normal((n_tr, d))
    Ys_tr = Xs_tr @ W_true + 0.05 * rng.standard_normal((n_tr, d))
    model_M = pl._fit_ridge(Xs_tr, Ys_tr)
    model_Minv = pl._fit_ridge(Ys_tr, Xs_tr)
    xs_mean, ys_mean = Xs_tr.mean(axis=0), Ys_tr.mean(axis=0)

    # (1a) same-geometry target: direct transfer ≈ ceiling.
    Xt_tr = rng.standard_normal((n_tr, d))
    Yt_tr = Xt_tr @ W_true + 0.05 * rng.standard_normal((n_tr, d))
    Xt_te = rng.standard_normal((n_te, d))
    Yt_te = Xt_te @ W_true + 0.05 * rng.standard_normal((n_te, d))
    preds, info = compute_rungs_for_fold(model_M, model_Minv, xs_mean, ys_mean, Xt_tr, Yt_tr, Xt_te)
    assert set(preds) == set(RUNGS) and all(p.shape == (n_te, d) for p in preds.values())
    r2_same = {r: pf._r2_matrix(Yt_te, preds[r]) for r in RUNGS}
    assert r2_same["1_direct"] > 0.8, f"direct transfer on shared geometry: {r2_same['1_direct']}"
    assert r2_same["9_full_refit"] > 0.8
    # (1b) answer-side rotated target: output-side adaptation recovers, direct dies.
    R = np.linalg.qr(rng.standard_normal((d, d)))[0]
    Ytr_rot, Yte_rot = Yt_tr @ R, Yt_te @ R
    preds_rot, _ = compute_rungs_for_fold(
        model_M, model_Minv, xs_mean, ys_mean, Xt_tr, Ytr_rot, Xt_te
    )
    r2_rot = {r: pf._r2_matrix(Yte_rot, preds_rot[r]) for r in RUNGS}
    assert r2_rot["6_rotation"] > r2_rot["1_direct"] + 0.2, r2_rot
    assert r2_rot["8_ans_reparam"] > r2_rot["1_direct"] + 0.2, r2_rot
    # (1c) context-side rotated target: input-side re-map recovers, direct dies.
    Xtr_rot, Xte_rot = Xt_tr @ R, Xt_te @ R
    preds_ctx, _ = compute_rungs_for_fold(
        model_M, model_Minv, xs_mean, ys_mean, Xtr_rot, Yt_tr, Xte_rot
    )
    r2_ctx = {r: pf._r2_matrix(Yt_te, preds_ctx[r]) for r in RUNGS}
    assert r2_ctx["7_ctx_reparam"] > r2_ctx["1_direct"] + 0.2, r2_ctx
    _log(
        "[probe] rung battery: shapes OK; direct≈ceiling on shared geometry; "
        "rotation/ans-re-map recover an answer-side rotation; ctx-re-map recovers a "
        "context-side rotation"
    )

    # (2) rung-7 batched null vs serial oracle (same rng sequence + b7 trick).
    Zhat_tr = pl._apply_ridge(model_Minv, Yt_tr)
    n_draws = 5
    batched = null_ctx_reparam_r2(
        model_M, Zhat_tr, Xt_tr, Yt_tr, Xt_te, Yt_te, n_draws=n_draws, seed=99
    )
    oracle_rng = np.random.default_rng(99)
    for i in range(n_draws):
        perm = oracle_rng.permutation(n_tr)
        model_G = pl._fit_ridge(Xt_tr, Zhat_tr[perm])
        p7_tr = pl._apply_ridge(model_M, pl._apply_ridge(model_G, Xt_tr))
        p7_te = pl._apply_ridge(model_M, pl._apply_ridge(model_G, Xt_te))
        b7 = Yt_tr.mean(axis=0) - p7_tr.mean(axis=0)
        r2 = pf._r2_matrix(Yt_te, p7_te + b7)
        assert abs(r2 - batched[i]) < 1e-8, f"null7 oracle mismatch draw {i}: {r2} vs {batched[i]}"
    _log("[probe] rung-7 batched null == serial oracle (5 draws, incl. the b7 constancy)")

    # (3) producer→consumer e2e through issue2378_fits at tiny shape.
    with tempfile.TemporaryDirectory(prefix="i2378-ladder-probe-") as td:
        tmp = Path(td)
        store, ledger = tmp / "store", tmp / "ledger"
        fits_mod._write_probe_store(store, n=40, d=8)
        ns = argparse.Namespace(
            store_root=str(store),
            ledger_root=str(ledger),
            layer=1,
            layer_star_from=None,
            n_null_draws=6,
            bootstrap_draws=24,
            reduced_k=4,
            units="context",
            g3_gate_file=None,
            pairs="all",
            survivors=None,
            fold_floors_override=fits_mod._PROBE_FLOORS,
        )
        ledger.mkdir(parents=True)
        rc = fits_mod.phase_g3(ns)
        assert rc == 0, f"probe fits G3 rc={rc}"
        rc = fits_mod.phase_fit(ns)
        assert rc == 0
        rc = phase_pairs(ns)
        assert rc == 0
        rung1 = json.loads(
            (ledger / "ladder" / "chat_to_plain_text__rung1.json").read_text("utf-8")
        )
        assert rung1["recovery"]["point_pooled"] > 0.8  # shared planted geometry
        assert rung1["headline_fold_label"] == "conversation-grouped"
        s1 = json.loads((ledger / "ladder" / "chat_to_storyq_astra__rung1.json").read_text("utf-8"))
        assert s1["headline_fold_label"] == "family-held-out"
        assert s1["recovery"]["label"] == "family-held-out"
        r7 = json.loads((ledger / "ladder" / "chat_to_storyq_astra__rung7.json").read_text("utf-8"))
        assert len(r7["null"]["per_fold_draws"]) == 5
        assert all(len(fd) == 6 for fd in r7["null"]["per_fold_draws"])
        r4 = json.loads((ledger / "ladder" / "chat_to_storyq_astra__rung4.json").read_text("utf-8"))
        assert r4["null"]["kind"] == "analytic-invariance"
        for target in target_cells(
            sorted(json.loads((ledger / p6.FOLD_MAP_NAME).read_text())["cells"])
        )[:1]:
            for ri in range(1, 10):
                assert (ledger / "ladder" / f"chat_to_{target}__rung{ri}.json").exists()
        # v7: phase_h3 is TOMBSTONED (dialogue descoped) — assert the refusal
        # fires loud and writes NOTHING. The G2b drop-marker-precedence legs
        # it used to exercise are covered by the fits probe's ratio N/A legs
        # (issue2378_fits phase_probe step 6b — same p6.g2b_dropped_now seam).
        try:
            phase_h3(ns)
            raise AssertionError("tombstoned phase_h3 must refuse")
        except SystemExit as e:
            assert "DESCOPED at plan v7" in str(e), e
        assert not (ledger / "ladder" / "h3_question_vs_dialogue.json").exists()
        _log("[probe] h3 tombstone refusal (v7 dialogue descope): OK")
        rc = phase_h4b(ns)
        assert rc == 0
        h4b = json.loads((ledger / "ladder" / "h4b_real_vs_sim.json").read_text("utf-8"))
        assert h4b["pair_assert"]["n_hash_mismatched"] == 0
        assert len(h4b["ceiling_delta"]["delta_draws"]) == 24
        assert "1_direct" in h4b["recovery_delta"]
        # resume path: re-running a pair with the same regime is a skip.
        memo = _SourceMemo(store, _fold_map(ns), 1)
        run_pair_unit(ns, _fold_map(ns), memo, "plain_text", 1)
        _log("[probe] e2e fits->ladder->h4b (+resume-skip; h3 tombstoned at v7): OK")
    _log("[phase=probe] done — all ladder probes passed")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.replace("%", "%%"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--phase", choices=("pairs", "h3", "h4b", "probe"), default="pairs")
    ap.add_argument(
        "--pairs",
        default="all",
        help="comma list of TARGET cells (shard axis for the P6 fan-out), or 'all' (8 targets at v7)",
    )
    ap.add_argument("--list-pairs", action="store_true")
    ap.add_argument(
        "--store-root",
        default=str(cm.REPO_ROOT / "data" / "issue_2378" / "activations"),
    )
    ap.add_argument("--ledger-root", default=str(cm.LEDGER_ROOT))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--layer-star-from", default=None)
    ap.add_argument("--n-null-draws", type=int, default=100)
    ap.add_argument("--bootstrap-draws", type=int, default=200)
    ap.add_argument("--g3-gate-file", default=None)
    ap.add_argument(
        "--survivors",
        default=None,
        help="CSV of the CURRENT dispatch's G2b survivor set (threaded by the "
        "dispatch at p6.h3): keys __g2b_dropped.json marker authority to THIS "
        "run — a stale prior-run marker on a surviving cell is ignored. Absent: "
        "the drop marker alone is authoritative (drop-before-fit).",
    )
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.list_pairs:
        for t in target_cells(list(cm.ALL_CELLS)):
            print(f"chat->{t}")
        return 0
    if args.phase == "pairs":
        return phase_pairs(args)
    if args.phase == "h3":
        return phase_h3(args)
    if args.phase == "h4b":
        return phase_h4b(args)
    if args.phase == "probe":
        return phase_probe(args)
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
