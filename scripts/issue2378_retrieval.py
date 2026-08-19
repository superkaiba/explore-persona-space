"""Issue #2378 P6 — retrieval battery (unit 4 deliverable 2; plan §4.4, #2202
conventions).

Registered retrieval COMPANION per unit (R² stays the PRIMARY lattice metric;
disagreements are analyzer-facing fields — NO retrieval verdict rules, plan
§3). Units:

  fits units (26)     ``fits/preds/<cell>__<arm>__preds.npz`` — the cell's own
                      GCV-ridge map, each row predicted by its OWN fold's map.
  ladder units (~108) ``ladder/preds/chat_to_<t>__rung<ri>__preds.npz``
                      (12 pairs x 9 rungs; a target the ladder marked
                      UNMAPPABLE becomes a named counted skip —
                      ``retrieval/chat_to_<t>__skipped.json``).

Candidate pool per cell = ALL of the cell's equalized held-out answers pooled
across folds (each row's prediction comes from its own fold's map, so the pool
is the full cohort; chance stated 1/pool). Conventions (plan §4.4; ports named
per function): ``raw_euclidean``, ``raw_cosine``, ``whiten_cos`` (z =
L_f^-1 (x - mu_f); L_f = Cholesky of the shrunk answer covariance, lambda =
0.1 — ``analysis.null_battery.shrunk_cholesky_from_cov``; covariance PER FOLD
from fold f's TRAIN rows only (folds != f), each query whitened by its target
row's fold decomposition — r1 review major retrieval-whitening-eval-leakage:
no evaluation row influences its own preprocessing), ``csls_cos`` +
``csls_whiten_cos`` (CSLS K=10 via the canonical
``issue1901_metric_battery.csls_scores`` — both neighborhoods from the query x
pool matrix; whitened-CSLS neighborhoods live within each fold's query group,
a disclosed convention field; retrieval distance = -score).
Mid-rank tie convention == ``analysis.mapping_baselines.knn_retrieval`` (the
formula is byte-identical), so every fits/ladder unit is RECONCILED against
its producer's persisted knn block (fail-loud on mismatch — a row-order or
stale-preds seam bug, never a tolerance knob).

Fresh-draw legs (per cell; ``chat_user_real`` is a disclosed N/A — its render
is deterministic, no redraws):

  fresh reference   queries = the 4 fresh-seed answer states of the covered
                    rows (fresh row_id == production row_id); per-context
                    rank-1 share, averaged (the #2202 freshdraw_reference
                    convention), under every convention.
  avg-target read   pool entry of each covered row replaced by
                    mean(original + 4 fresh); queries = the cell's own
                    context-arm map predictions on the covered rows
                    (#2202 freshwhiten Leg D), beside the matched
                    single-target read on the same covered rows.

Everything vectorized (batched GEMMs / K train-only Cholesky decompositions
per cell, computed once and reused across every unit — never per-row loops);
per-unit atomic JSON checkpoints + regime-keyed resume; empty selections
raise. G3's rank-1-vs-10x-chance read is exposed battery-grade in
the chat/context output (``g3_battery_read``).

Phases: ``--phase all`` (default: battery + fresh), ``battery``, ``fresh``,
``--phase probe`` (synthetic CPU self-verification: brute-force rank oracle
per convention incl. whitened-cosine + CSLS re-derived from the paper formula,
producer-knn reconciliation, unmappable-skip counting, fresh coverage
accounting on a planted non-cohort row, resume-skip).
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
from scipy.linalg import solve_triangular  # noqa: E402

import issue1901_metric_battery as mb  # noqa: E402  (csls_scores — canonical CSLS)
import issue2378_common as cm  # noqa: E402
import issue2378_fits as fits_mod  # noqa: E402  (resolve_layer + probe-store reuse)
import issue2378_ladder as ladder_mod  # noqa: E402  (RUNGS registry)
import issue2378_p6_common as p6  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import _pairwise_dist  # noqa: E402
from explore_persona_space.analysis.null_battery import shrunk_cholesky_from_cov  # noqa: E402

SCRIPT_VERSION = "issue2378_retrieval_v1"

CONVENTIONS = ("raw_euclidean", "raw_cosine", "whiten_cos", "csls_cos", "csls_whiten_cos")
KS = (1, 5, 10)
CSLS_K = 10
WHITEN_LAM = 0.1
KNN_RECON_TOL_ROWS = 2.0  # producer-vs-battery acc@k tolerance, in rows (#2202 ACC_TOL_ROWS)
# Cells with NO fresh redraws by design (capture FRESH_DEFAULT_CELLS excludes
# chat_user_real — deterministic render; plan §4.4 discloses the N/A).
FRESH_NA_CELLS = ("chat_user_real",)


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Rank / summary primitives (ports named per function; oracle-verified in probe)
# ---------------------------------------------------------------------------


def midranks_of_true(d: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """Mid-ranks of d[i, true_idx[i]] within each row — VERBATIM port of
    ``scripts/issue2202_metric_zoo.midranks_of_true`` (imported-by-copy to
    avoid that script module's import chain; == the knn_retrieval formula:
    1 + #closer + 0.5*#tied-others, tol = 1e-9*max(|d_true|, 1e-12))."""
    n = d.shape[0]
    dt = d[np.arange(n), true_idx]
    tol = 1e-9 * np.maximum(np.abs(dt)[:, None], 1e-12)
    closer = (d < dt[:, None] - tol).sum(axis=1)
    tied = (np.abs(d - dt[:, None]) <= tol).sum(axis=1) - 1
    return 1.0 + closer + 0.5 * tied


def ranks_summary(ranks: np.ndarray, n_pool: int) -> dict:
    """acc@k / chance / median rank / MRR — port of
    ``scripts/issue2202_csls_followup.ranks_summary`` (KS bound here)."""
    return {
        "acc_at_k": {int(k): float((ranks <= k).mean()) for k in KS},
        "chance_at_k": {int(k): float(k / n_pool) for k in KS},
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
        "n": int(ranks.shape[0]),
        "n_pool": int(n_pool),
    }


def pool_ctx_with_modified_y(ctx: dict, y_mod: np.ndarray) -> dict:
    """Pool context with a MODIFIED pool under the cell's EXISTING per-fold
    whitening decompositions (the avg-target pool — same per-fold L/mu as the
    cell, the freshwhiten Leg D convention; only Yw is recomputed)."""
    per_fold = {
        f: {
            "mu": fc["mu"],
            "L": fc["L"],
            "Yw": solve_triangular(fc["L"], (y_mod - fc["mu"]).T, lower=True).T,
        }
        for f, fc in ctx["per_fold"].items()
    }
    return {
        "Y": y_mod,
        "row_fold": ctx["row_fold"],
        "per_fold": per_fold,
        "whiten": ctx["whiten"],
    }


def build_pool_ctx(y32: np.ndarray, entry: dict) -> dict:
    """Per-cell pool context: float64 answers + K per-fold shrunk-covariance
    Cholesky whitening decompositions (lambda = 0.1), each fit ONLY on fold
    f's TRAIN rows (folds != f) — r1 review major
    retrieval-whitening-eval-leakage: a query is whitened by its own target
    row's fold decomposition, which never saw that row, so no evaluation row
    influences its own preprocessing. Each fold's Yw whitens the FULL pool
    (candidates are not train data — only the fitted transform is)."""
    y = np.ascontiguousarray(y32, dtype=np.float64)
    folds = np.asarray(entry["folds"], dtype=np.int64)
    assert folds.shape[0] == y.shape[0], (folds.shape, y.shape)
    per_fold: dict[int, dict] = {}
    n_train: dict[int, int] = {}
    for f, (tr, _te) in enumerate(p6.fold_splits(entry)):
        y_tr = y[tr]
        mu = y_tr.mean(axis=0)
        cov = np.cov(y_tr, rowvar=False)
        chol = shrunk_cholesky_from_cov(cov, WHITEN_LAM)
        del cov
        per_fold[f] = {
            "mu": mu,
            "L": chol,
            "Yw": solve_triangular(chol, (y - mu).T, lower=True).T,
        }
        n_train[f] = int(tr.shape[0])
    whiten = {
        "lam": WHITEN_LAM,
        "builder": "analysis.null_battery.shrunk_cholesky_from_cov",
        "cov_source": (
            "per-fold TRAIN-ONLY answers (folds != f): each query is whitened by its "
            "target row's fold decomposition, which never saw that row (no evaluation "
            "leakage into the whitening transform)"
        ),
        "n_rows": int(y.shape[0]),
        "n_train_per_fold": {int(f): n for f, n in sorted(n_train.items())},
        "csls_note": (
            "whitened-CSLS neighborhood statistics are computed within each fold's "
            "query group (queries sharing a target fold see one whitened space; the "
            "target-side r_T averages over that group's queries only — disclosed; "
            "k clamped to the group size, which binds only at probe/tiny-n scale); "
            "raw (unwhitened) CSLS is unchanged, computed over the full query set"
        ),
    }
    return {"Y": y, "row_fold": folds, "per_fold": per_fold, "whiten": whiten}


def battery(
    preds: np.ndarray,
    ctx: dict,
    *,
    true_idx: np.ndarray | None = None,
    conventions: tuple[str, ...] = CONVENTIONS,
) -> tuple[dict, dict]:
    """All-convention retrieval reads of ``preds`` against the ctx pool.

    Returns (summaries, ranks) keyed by convention. ``true_idx[i]`` is the
    pool row holding query i's true target (default arange — the fits/ladder
    pool==cohort case). One (n_q, n_pool) float64 matrix lives at a time.
    """
    q = np.ascontiguousarray(preds, dtype=np.float64)
    n_pool = ctx["Y"].shape[0]
    ti = np.arange(q.shape[0]) if true_idx is None else np.asarray(true_idx, dtype=np.int64)
    if q.shape[0] == 0 or n_pool == 0:
        raise RuntimeError("empty query/pool set — refusing a vacuous retrieval read")
    summaries: dict[str, dict] = {}
    ranks: dict[str, np.ndarray] = {}

    def _take(name: str, dist: np.ndarray) -> None:
        r = midranks_of_true(dist, ti)
        ranks[name] = r
        summaries[name] = ranks_summary(r, n_pool)

    if "raw_euclidean" in conventions:
        dmat = _pairwise_dist(q, ctx["Y"], "euclidean")
        _take("raw_euclidean", dmat)
        del dmat
    if "raw_cosine" in conventions or "csls_cos" in conventions:
        dmat = _pairwise_dist(q, ctx["Y"], "cosine")
        if "raw_cosine" in conventions:
            _take("raw_cosine", dmat)
        if "csls_cos" in conventions:
            np.subtract(1.0, dmat, out=dmat)  # s_cos = 1 - d
            sc = mb.csls_scores(dmat, CSLS_K)
            np.negative(sc, out=sc)  # retrieval distance = -score
            _take("csls_cos", sc)
            del sc
        del dmat
    if "whiten_cos" in conventions or "csls_whiten_cos" in conventions:
        # Per-fold whitening (r1 review major retrieval-whitening-eval-leakage):
        # queries are grouped by their TARGET row's fold and whitened by that
        # fold's train-only decomposition; whitened-CSLS neighborhoods live
        # within each fold group (ctx["whiten"]["csls_note"]).
        r_w = np.empty(q.shape[0]) if "whiten_cos" in conventions else None
        r_cw = np.empty(q.shape[0]) if "csls_whiten_cos" in conventions else None
        q_folds = ctx["row_fold"][ti]
        for f, fctx in sorted(ctx["per_fold"].items()):
            sel = np.flatnonzero(q_folds == f)
            if sel.size == 0:
                continue
            qw = solve_triangular(fctx["L"], (q[sel] - fctx["mu"]).T, lower=True).T
            dmat = _pairwise_dist(qw, fctx["Yw"], "cosine")
            del qw
            if r_w is not None:
                r_w[sel] = midranks_of_true(dmat, ti[sel])
            if r_cw is not None:
                np.subtract(1.0, dmat, out=dmat)
                # k clamped to the fold group's query count — binds only at
                # probe/tiny-n scale (production fold groups are >= ~1000
                # queries, so k_eff == CSLS_K there); disclosed in csls_note.
                sc = mb.csls_scores(dmat, min(CSLS_K, dmat.shape[0]))
                np.negative(sc, out=sc)
                r_cw[sel] = midranks_of_true(sc, ti[sel])
                del sc
            del dmat
        if r_w is not None:
            ranks["whiten_cos"] = r_w
            summaries["whiten_cos"] = ranks_summary(r_w, n_pool)
        if r_cw is not None:
            ranks["csls_whiten_cos"] = r_cw
            summaries["csls_whiten_cos"] = ranks_summary(r_cw, n_pool)
    return summaries, ranks


def reconcile_knn(summaries: dict, producer_knn: dict, n_pool: int, unit: str) -> dict:
    """Battery raw reads vs the PRODUCER's persisted knn block (fits/ladder
    ``baselines.knn``) — identical arrays + identical mid-rank formula, so a
    mismatch beyond the #2202 row tolerance is a row-order / stale-preds seam
    bug: RAISE, never widen."""
    tol = KNN_RECON_TOL_ROWS / n_pool + 1e-12
    out: dict[str, dict] = {}
    for metric, conv in (("euclidean", "raw_euclidean"), ("cosine", "raw_cosine")):
        if conv not in summaries:
            out[metric] = {"skipped": "convention not in this run's set"}
            continue
        prod_acc = producer_knn[metric]["acc_at_k"]
        rows: dict[int, dict] = {}
        worst = 0.0
        for k in KS:
            pv = float(prod_acc[str(k)] if str(k) in prod_acc else prod_acc[k])
            bv = float(summaries[conv]["acc_at_k"][k])
            diff = abs(pv - bv)
            worst = max(worst, diff)
            rows[int(k)] = {"producer": pv, "battery": bv, "abs_diff": diff}
        if worst > tol:
            raise RuntimeError(
                f"knn reconciliation FAIL for {unit} ({metric}): max |acc@k| diff "
                f"{worst:.6f} > tol {tol:.6f} — row-order or stale-preds seam bug"
            )
        out[metric] = {"acc_at_k": rows, "max_abs_diff": worst, "tol": tol, "pass": True}
    return out


# ---------------------------------------------------------------------------
# Unit runners
# ---------------------------------------------------------------------------


def _resume_ok(path: Path, regime: dict) -> bool:
    if not path.exists():
        return False
    prior = json.loads(path.read_text(encoding="utf-8"))
    if prior.get("regime") == regime:
        return True
    raise RuntimeError(f"regime mismatch at {path} — use a fresh ledger root")


def _load_preds(path: Path, entry: dict, unit: str) -> np.ndarray:
    if not path.exists():
        raise RuntimeError(f"missing preds sidecar {path} for unit {unit} — run its producer first")
    with np.load(path) as z:
        row_ids = [str(x) for x in z["row_ids"].tolist()]
        preds = np.asarray(z["preds"], dtype=np.float32)
    if row_ids != entry["row_ids"]:
        raise RuntimeError(f"{unit}: preds row order != fold map (mixed generations)")
    return preds


def run_fits_unit(
    ledger_root: Path, cell: str, arm: str, entry: dict, ctx: dict, regime: dict
) -> None:
    out_path = ledger_root / "retrieval" / f"{cell}__{arm}.json"
    if _resume_ok(out_path, regime):
        _log(f"[retrieval] SKIP {cell}/{arm}: output exists with matching regime")
        return
    t0 = time.time()
    unit = f"fits:{cell}/{arm}"
    preds_path = ledger_root / "fits" / "preds" / f"{cell}__{arm}__preds.npz"
    preds = _load_preds(preds_path, entry, unit)
    conventions = tuple(regime["conventions"])
    summaries, _ranks = battery(preds, ctx, conventions=conventions)
    fits = json.loads((ledger_root / "fits" / f"{cell}__{arm}.json").read_text(encoding="utf-8"))
    recon = reconcile_knn(summaries, fits["baselines"]["knn"], ctx["Y"].shape[0], unit)
    payload = {
        "regime": regime,
        "unit_kind": "fits",
        "cell": cell,
        "arm": arm,
        "preds_path": str(preds_path),
        "n_rows": int(entry["n_rows"]),
        "n_pool": int(ctx["Y"].shape[0]),
        "chance_at_1": 1.0 / ctx["Y"].shape[0],
        "fold_structure": entry["fold_structure"],
        "conventions": summaries,
        "reconciliation_vs_producer_knn": recon,
        "whiten": ctx["whiten"],
        "unit_wall_s": round(time.time() - t0, 2),
        "metadata": cm.run_metadata(),
    }
    if cell == "chat" and arm == "context" and "raw_euclidean" in summaries:
        n_pool = ctx["Y"].shape[0]
        acc1 = summaries["raw_euclidean"]["acc_at_k"][1]
        payload["g3_battery_read"] = {
            "convention": "raw_euclidean",
            "acc_at_1": acc1,
            "chance_at_1": 1.0 / n_pool,
            "mult_of_chance": acc1 * n_pool,
            "pass_10x_chance": bool(acc1 >= 10.0 / n_pool),
            "note": (
                "battery-grade record of G3's rank-1-vs-10x-chance read (unit 3's gate "
                "used the fits knn block inline; this is the registered-battery version)"
            ),
        }
    cm.atomic_write_json(out_path, payload)
    _log(
        f"[retrieval] {cell}/{arm}: "
        + " ".join(f"{c}@1={summaries[c]['acc_at_k'][1]:.3f}" for c in summaries)
        + f" wall={payload['unit_wall_s']}s"
    )


def run_ladder_unit(
    ledger_root: Path, cell: str, ri: int, entry: dict, ctx: dict, regime: dict
) -> None:
    out_path = ledger_root / "retrieval" / f"chat_to_{cell}__rung{ri}.json"
    if _resume_ok(out_path, regime):
        return
    t0 = time.time()
    unit = f"ladder:chat_to_{cell}/rung{ri}"
    rung_json_path = ledger_root / "ladder" / f"chat_to_{cell}__rung{ri}.json"
    if not rung_json_path.exists():
        raise RuntimeError(f"missing {rung_json_path} — run issue2378_ladder.py first ({unit})")
    rung = json.loads(rung_json_path.read_text(encoding="utf-8"))
    preds_path = ledger_root / "ladder" / "preds" / f"chat_to_{cell}__rung{ri}__preds.npz"
    preds = _load_preds(preds_path, entry, unit)
    conventions = tuple(regime["conventions"])
    summaries, _ranks = battery(preds, ctx, conventions=conventions)
    recon = reconcile_knn(summaries, rung["baselines"]["knn"], ctx["Y"].shape[0], unit)
    payload = {
        "regime": regime,
        "unit_kind": "ladder",
        "pair": f"chat_to_{cell}",
        "rung_index": ri,
        "rung": ladder_mod.RUNGS[ri - 1],
        "suppress_by_tier": rung.get("suppress_by_tier"),
        "preds_path": str(preds_path),
        "n_rows": int(entry["n_rows"]),
        "n_pool": int(ctx["Y"].shape[0]),
        "chance_at_1": 1.0 / ctx["Y"].shape[0],
        "conventions": summaries,
        "reconciliation_vs_producer_knn": recon,
        "whiten": ctx["whiten"],
        "unit_wall_s": round(time.time() - t0, 2),
        "metadata": cm.run_metadata(),
    }
    cm.atomic_write_json(out_path, payload)
    _log(
        f"[retrieval] chat->{cell} rung{ri}: raw_euclidean@1="
        f"{summaries.get('raw_euclidean', {}).get('acc_at_k', {}).get(1, float('nan')):.3f} "
        f"wall={payload['unit_wall_s']}s"
    )


def _load_fresh_draws(
    store_root: Path, cell: str, layer: int, entry: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
    """(draws (n_cov, K, d) float64, pos (n_cov,) cohort indices, counts).

    Covered rows = cohort rows present in ALL fresh seeds (fresh row_id ==
    production row_id); drops are counted + disclosed, never silent.
    """
    per_seed_arrays: dict[int, np.ndarray] = {}
    per_seed_pos: dict[int, dict[str, int]] = {}
    for seed in cm.FRESH_SEEDS:
        tag = f"{cell}__fresh_d{seed}"
        if not p6.production_part_indices(store_root, tag):
            raise RuntimeError(
                f"no fresh store parts for {tag} under {store_root} — capture the fresh "
                "draws (issue2378_capture.py --phase capture_fresh) first"
            )
        pack = p6.load_cell_arrays(store_root, tag, layer, (p6.ANSWER_SLOT,))
        per_seed_arrays[seed] = pack["arrays"][p6.ANSWER_SLOT]
        per_seed_pos[seed] = {rid: i for i, rid in enumerate(pack["row_ids"])}
    common = set.intersection(*(set(m) for m in per_seed_pos.values()))
    cohort = entry["row_ids"]
    covered = [rid for rid in cohort if rid in common]
    counts = {
        "n_per_seed": {int(s): len(per_seed_pos[s]) for s in cm.FRESH_SEEDS},
        "n_common_across_seeds": len(common),
        "n_cohort": len(cohort),
        "n_covered": len(covered),
        "n_dropped_not_in_cohort": len(common) - len(covered),
    }
    if not covered:
        raise RuntimeError(f"{cell}: zero fresh rows intersect the equalized cohort")
    d = per_seed_arrays[cm.FRESH_SEEDS[0]].shape[1]
    draws = np.empty((len(covered), len(cm.FRESH_SEEDS), d), dtype=np.float64)
    for si, seed in enumerate(cm.FRESH_SEEDS):
        arr, pos_map = per_seed_arrays[seed], per_seed_pos[seed]
        idx = np.array([pos_map[rid] for rid in covered], dtype=np.int64)
        draws[:, si, :] = arr[idx].astype(np.float64)
    cohort_pos = {rid: i for i, rid in enumerate(cohort)}
    pos = np.array([cohort_pos[rid] for rid in covered], dtype=np.int64)
    return draws, pos, counts


def run_fresh_unit(
    store_root: Path,
    ledger_root: Path,
    cell: str,
    entry: dict,
    ctx: dict,
    layer: int,
    regime: dict,
) -> None:
    out_path = ledger_root / "retrieval" / f"{cell}__fresh.json"
    if _resume_ok(out_path, regime):
        _log(f"[retrieval] SKIP {cell} fresh: output exists with matching regime")
        return
    t0 = time.time()
    if cell in FRESH_NA_CELLS:
        cm.atomic_write_json(
            out_path,
            {
                "regime": regime,
                "cell": cell,
                "status": "N/A",
                "reason": (
                    "deterministic render — no fresh redraws for this cell by design "
                    "(capture FRESH_DEFAULT_CELLS excludes chat_user_real; plan §4.4 "
                    "disclosed field)"
                ),
                "metadata": cm.run_metadata(),
            },
        )
        _log(f"[retrieval] {cell} fresh: N/A (disclosed — deterministic render)")
        return
    conventions = tuple(regime["conventions"])
    draws, pos, counts = _load_fresh_draws(store_root, cell, layer, entry)
    n_cov, n_draws, d = draws.shape
    # Fresh-draw reference: queries = every fresh draw; per-context rank-1
    # share averaged over contexts (the #2202 freshdraw_reference convention).
    queries = draws.reshape(n_cov * n_draws, d)
    ti = np.repeat(pos, n_draws)
    ref_summaries, ref_ranks = battery(queries, ctx, true_idx=ti, conventions=conventions)
    fresh_reference = {}
    for conv, summ in ref_summaries.items():
        kr = ref_ranks[conv].reshape(n_cov, n_draws)
        fresh_reference[conv] = {
            **summ,
            "acc1_ceiling_per_context": float((kr == 1.0).mean(axis=1).mean()),
        }
    # Avg-target read (context arm): pool entry of each covered row replaced
    # by mean(original + fresh draws); queries = own-map preds on those rows.
    preds = _load_preds(
        ledger_root / "fits" / "preds" / f"{cell}__context__preds.npz",
        entry,
        f"fresh-avg:{cell}/context",
    )[pos]
    single_summ, _ = battery(preds, ctx, true_idx=pos, conventions=conventions)
    y_mod = ctx["Y"].copy()
    y_mod[pos] = (ctx["Y"][pos] + draws.sum(axis=1)) / (1.0 + n_draws)
    ctx_mod = pool_ctx_with_modified_y(ctx, y_mod)
    avg_summ, _ = battery(preds, ctx_mod, true_idx=pos, conventions=conventions)
    del ctx_mod, y_mod
    payload = {
        "regime": regime,
        "cell": cell,
        "status": "ok",
        "counts": counts,
        "n_pool": int(ctx["Y"].shape[0]),
        "fresh_reference": fresh_reference,
        "avg_target": {
            "arm": "context",
            "n_covered": int(n_cov),
            "definition": (
                "pool entry of each covered row replaced by mean(original + "
                f"{n_draws} fresh draws); queries = the cell's own context-arm map "
                "predictions on the covered rows; single_target = the matched read "
                "against the unmodified pool (same queries, same covered rows)"
            ),
            "single_target": single_summ,
            "averaged_target": avg_summ,
        },
        "whiten": ctx["whiten"],
        "unit_wall_s": round(time.time() - t0, 2),
        "metadata": cm.run_metadata(),
    }
    cm.atomic_write_json(out_path, payload)
    _log(
        f"[retrieval] {cell} fresh: covered={n_cov} "
        f"eucl_ceiling={fresh_reference.get('raw_euclidean', {}).get('acc1_ceiling_per_context', float('nan')):.3f} "
        f"wall={payload['unit_wall_s']}s"
    )


def run_cell(
    args, fm: dict, cell: str, layer: int, regime: dict, *, battery_on: bool, fresh_on: bool
) -> None:
    ledger_root = Path(args.ledger_root)
    store_root = Path(args.store_root)
    entry = fm["cells"][cell]
    (ledger_root / "retrieval").mkdir(parents=True, exist_ok=True)
    pack = p6.load_cell_arrays(
        store_root, cell, layer, (p6.ANSWER_SLOT,), row_order=entry["row_ids"]
    )
    ctx = build_pool_ctx(pack["arrays"][p6.ANSWER_SLOT], entry)
    del pack
    if battery_on:
        for arm in p6.ARMS:
            run_fits_unit(ledger_root, cell, arm, entry, ctx, regime)
        if cell != "chat":
            unmap = ledger_root / "ladder" / f"chat_to_{cell}__unmappable.json"
            if unmap.exists():
                skip_path = ledger_root / "retrieval" / f"chat_to_{cell}__skipped.json"
                if not _resume_ok(skip_path, regime):
                    cm.atomic_write_json(
                        skip_path,
                        {
                            "regime": regime,
                            "pair": f"chat_to_{cell}",
                            "status": "skipped",
                            "reason": (
                                "ladder marked the target UNMAPPABLE (lattice rule — no "
                                "rung preds exist); counted named skip, plan §3"
                            ),
                            "marker": str(unmap),
                            "metadata": cm.run_metadata(),
                        },
                    )
                _log(f"[retrieval] chat->{cell}: SKIPPED (ladder unmappable marker)")
            else:
                for ri in range(1, len(ladder_mod.RUNGS) + 1):
                    run_ladder_unit(ledger_root, cell, ri, entry, ctx, regime)
    if fresh_on:
        run_fresh_unit(store_root, ledger_root, cell, entry, ctx, layer, regime)
    del ctx


# ---------------------------------------------------------------------------
# Phases / CLI
# ---------------------------------------------------------------------------


def _fold_map(args) -> dict:
    return p6.load_or_build_fold_map(
        Path(args.store_root), Path(args.ledger_root), **getattr(args, "fold_floors_override", {})
    )


def _parse_conventions(spec: str) -> tuple[str, ...]:
    if spec.strip() == "all":
        return CONVENTIONS
    toks = tuple(t.strip() for t in spec.split(",") if t.strip())
    bad = [t for t in toks if t not in CONVENTIONS]
    if bad or not toks:
        raise SystemExit(f"unknown conventions {bad} (choices: {CONVENTIONS})")
    return toks


def _parse_cells(spec: str, fm: dict) -> list[str]:
    if spec.strip() == "all":
        return sorted(fm["cells"])
    cells = [t.strip() for t in spec.split(",") if t.strip()]
    bad = [c for c in cells if c not in fm["cells"]]
    if bad or not cells:
        raise SystemExit(f"unknown cells {bad} (fold map has {sorted(fm['cells'])})")
    return cells


def _regime(args, fm: dict, layer: int, conventions: tuple[str, ...]) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "layer": int(layer),
        "k": fm["k"],
        "seed": fm["seed"],
        "n_eq": fm["n_eq"],
        "fold_map_sha": fm["sha256"],
        "conventions": list(conventions),
        "ks": list(KS),
        "csls_k": CSLS_K,
        "whiten_lam": WHITEN_LAM,
        "fresh_seeds": list(cm.FRESH_SEEDS),
        "pool": (
            "the cell's full equalized cohort answers, pooled across folds (each row's "
            "prediction from its own fold's map); chance = 1/pool"
        ),
        "whiten_cov": (
            "per-fold TRAIN-ONLY (folds != f) shrunk covariance; queries whitened by "
            "their target row's fold decomposition (no eval leakage)"
        ),
    }


def phase_run(args, *, battery_on: bool, fresh_on: bool) -> int:
    ledger_root = Path(args.ledger_root)
    gate_path = Path(args.g3_gate_file or (ledger_root / p6.G3_GATE_NAME))
    p6.require_g3_pass(gate_path)
    fm = _fold_map(args)
    layer = fits_mod.resolve_layer(args)
    conventions = _parse_conventions(args.conventions)
    regime = _regime(args, fm, layer, conventions)
    cells = _parse_cells(args.cells, fm)
    t0 = time.time()
    for i, cell in enumerate(cells):
        run_cell(args, fm, cell, layer, regime, battery_on=battery_on, fresh_on=fresh_on)
        cm.progress("retrieval", i + 1, len(cells), cell, t0)
    return 0


# ---------------------------------------------------------------------------
# Synthetic CPU probe
# ---------------------------------------------------------------------------


def _scramble_cell_answers(store: Path, cell: str, *, layer: int = 1, seed: int = 31) -> None:
    """Destroy the X-Y linkage of one probe cell (permute v_A within each
    part) so its own fit lands below the floor -> the ladder writes the
    UNMAPPABLE marker and the retrieval skip path is exercised."""
    rng = np.random.default_rng(seed)
    for ci in p6.production_part_indices(store, cell):
        npz_path = store / f"{cell}__part{ci:04d}__L{layer}.npz"
        with np.load(npz_path) as z:
            arrays = {kk: np.asarray(z[kk]) for kk in z.files}
        v_a = p6.decode_bf16_np(arrays["v_A"])
        arrays["v_A"] = p6.encode_bf16_np(v_a[rng.permutation(v_a.shape[0])])
        with open(npz_path, "wb") as fh:
            np.savez(fh, **arrays)


def _write_probe_fresh(
    store: Path, fm: dict, cell: str, *, layer: int = 1, noise: float = 0.02
) -> None:
    """Fresh parts for one probe cell in the standard store format: 12 rows =
    1 planted NON-cohort id (exercises the dropped-with-count path) + 11
    cohort ids; the LAST seed omits the final cohort id (exercises the
    cross-seed intersection). Fresh v_A = production v_A + small noise, so the
    rank-1 reference is high by construction."""
    entry = fm["cells"][cell]
    cohort = list(entry["row_ids"])
    ledger_ids = [r["row_id"] for r in p6.load_ledger(store, cell)]
    non_cohort = [rid for rid in ledger_ids if rid not in set(cohort)]
    if not non_cohort:
        raise RuntimeError(f"probe expects a non-cohort id for {cell} (equalize-down)")
    take = [non_cohort[0], *cohort[:11]]
    pack = p6.load_cell_arrays(store, cell, layer, ("v_C", "v_A", "v_P"))
    pos_map = {rid: i for i, rid in enumerate(pack["row_ids"])}
    for si, seed in enumerate(cm.FRESH_SEEDS):
        ids = take if si < len(cm.FRESH_SEEDS) - 1 else take[:-1]
        rng = np.random.default_rng(cm.derived_seed(cm.SEED, cell, "probe_fresh", seed))
        idx = np.array([pos_map[rid] for rid in ids], dtype=np.int64)
        v_a = pack["arrays"]["v_A"][idx] + noise * rng.standard_normal(
            (len(ids), pack["arrays"]["v_A"].shape[1])
        ).astype(np.float32)
        tag = f"{cell}__fresh_d{seed}"
        arrays = {
            "v_C": p6.encode_bf16_np(pack["arrays"]["v_C"][idx]),
            "v_A": p6.encode_bf16_np(v_a.astype(np.float32)),
            "v_P": p6.encode_bf16_np(pack["arrays"]["v_P"][idx]),
            "row_ids": np.array(ids),
            "meta": np.array(
                json.dumps(
                    {
                        "encoding": "bf16_as_uint16",
                        "cell": tag,
                        "layer": layer,
                        "draw_seed": int(seed),
                        "hidden_size": int(v_a.shape[1]),
                    }
                )
            ),
        }
        with open(store / f"{tag}__part0000__L{layer}.npz", "wb") as fh:
            np.savez(fh, **arrays)
        cm.atomic_write_json(
            store / f"{tag}__part0000__rows.json",
            {"cell": cell, "tag": tag, "part": 0, "rows": [{"row_id": rid} for rid in ids]},
        )


def _oracle_ranks(q32: np.ndarray, ctx: dict, ti: np.ndarray) -> dict[str, np.ndarray]:
    """Brute-force per-row oracle, independently re-derived (python loops,
    np.linalg; CSLS from the Conneau et al. formula via np.sort) — the probe's
    exact-parity reference for every battery convention."""
    q = q32.astype(np.float64)
    y = ctx["Y"]
    n_q, n_pool = q.shape[0], y.shape[0]

    def _ranks_rows(dist: np.ndarray, ti_rows: np.ndarray) -> np.ndarray:
        out = np.empty(dist.shape[0])
        for i in range(dist.shape[0]):
            dt = dist[i, ti_rows[i]]
            tol = 1e-9 * max(abs(dt), 1e-12)
            closer = int((dist[i] < dt - tol).sum())
            tied = int((np.abs(dist[i] - dt) <= tol).sum()) - 1
            out[i] = 1.0 + closer + 0.5 * tied
        return out

    d_e = np.empty((n_q, n_pool))
    d_c = np.empty((n_q, n_pool))
    for i in range(n_q):
        diff = y - q[i]
        d_e[i] = (diff**2).sum(axis=1)  # _pairwise_dist euclidean is SQUARED
        d_c[i] = 1.0 - (y @ q[i]) / (
            np.maximum(np.linalg.norm(y, axis=1) * np.linalg.norm(q[i]), 1e-12)
        )

    def _csls_dist(d_cos: np.ndarray) -> np.ndarray:
        s = 1.0 - d_cos
        k_eff = min(CSLS_K, s.shape[0])  # battery's per-fold group-size clamp
        r_q = np.array([np.sort(s[i])[-k_eff:].mean() for i in range(s.shape[0])])
        r_p = np.array([np.sort(s[:, j])[-k_eff:].mean() for j in range(n_pool)])
        return -(2.0 * s - r_q[:, None] - r_p[None, :])

    # Whitened conventions, per-fold independently re-derived: query i whitened
    # by its TARGET row's fold decomposition (np.linalg.solve per row, never the
    # battery's solve_triangular path); whitened-CSLS neighborhoods within each
    # fold's query group — mirroring the battery's registered convention.
    q_folds = np.asarray([int(ctx["row_fold"][ti[i]]) for i in range(n_q)])
    r_w = np.empty(n_q)
    r_cw = np.empty(n_q)
    for f in sorted(set(q_folds.tolist())):
        sel = np.flatnonzero(q_folds == f)
        mu_f, l_f = ctx["per_fold"][f]["mu"], ctx["per_fold"][f]["L"]
        qw = np.stack([np.linalg.solve(l_f, q[i] - mu_f) for i in sel])
        yw = np.stack([np.linalg.solve(l_f, y[j] - mu_f) for j in range(n_pool)])
        d_wc = np.empty((sel.size, n_pool))
        for i in range(sel.size):
            d_wc[i] = 1.0 - (yw @ qw[i]) / (
                np.maximum(np.linalg.norm(yw, axis=1) * np.linalg.norm(qw[i]), 1e-12)
            )
        r_w[sel] = _ranks_rows(d_wc, ti[sel])
        r_cw[sel] = _ranks_rows(_csls_dist(d_wc), ti[sel])

    return {
        "raw_euclidean": _ranks_rows(d_e, ti),
        "raw_cosine": _ranks_rows(d_c, ti),
        "whiten_cos": r_w,
        "csls_cos": _ranks_rows(_csls_dist(d_c), ti),
        "csls_whiten_cos": r_cw,
    }


def phase_probe(args) -> int:  # noqa: PLR0915
    """Synthetic CPU self-verification (module docstring item list)."""
    n, d = 40, 8
    with tempfile.TemporaryDirectory(prefix="i2378-retrieval-probe-") as td:
        tmp = Path(td)
        store, ledger = tmp / "store", tmp / "ledger"
        fits_mod._write_probe_store(store, n=n, d=d)
        _scramble_cell_answers(store, "plain_text")
        fit_ns = argparse.Namespace(
            store_root=str(store),
            ledger_root=str(ledger),
            layer=1,
            layer_star_from=None,
            n_null_draws=6,
            bootstrap_draws=24,
            reduced_k=4,
            units="all",
            pairs="plain_text,storyq_astra,chat_user_real",
            g3_gate_file=None,
            fold_floors_override=fits_mod._PROBE_FLOORS,
        )
        ledger.mkdir(parents=True)
        assert fits_mod.phase_g3(fit_ns) == 0
        assert fits_mod.phase_fit(fit_ns) == 0
        assert ladder_mod.phase_pairs(fit_ns) == 0
        assert (ledger / "ladder" / "chat_to_plain_text__unmappable.json").exists()
        fm = json.loads((ledger / p6.FOLD_MAP_NAME).read_text(encoding="utf-8"))
        for cell in ("chat", "plain_text", "storyq_astra"):
            _write_probe_fresh(store, fm, cell)

        ret_ns = argparse.Namespace(
            store_root=str(store),
            ledger_root=str(ledger),
            layer=1,
            layer_star_from=None,
            cells="chat,plain_text,storyq_astra,chat_user_real",
            conventions="all",
            g3_gate_file=None,
            fold_floors_override=fits_mod._PROBE_FLOORS,
        )
        assert phase_run(ret_ns, battery_on=True, fresh_on=True) == 0

        # (a) brute-force oracle parity on the chat/context unit, all 5
        # conventions (whitened-cosine + CSLS re-derived independently).
        entry = fm["cells"]["chat"]
        pack = p6.load_cell_arrays(store, "chat", 1, (p6.ANSWER_SLOT,), row_order=entry["row_ids"])
        y_chat = pack["arrays"][p6.ANSWER_SLOT]
        ctx = build_pool_ctx(y_chat, entry)
        preds = _load_preds(ledger / "fits" / "preds" / "chat__context__preds.npz", entry, "probe")
        _summ, ranks = battery(preds, ctx)
        ti = np.arange(preds.shape[0])
        oracle = _oracle_ranks(preds, ctx, ti)
        for conv in CONVENTIONS:
            if not np.array_equal(ranks[conv], oracle[conv]):
                bad = int((ranks[conv] != oracle[conv]).sum())
                raise AssertionError(f"oracle mismatch for {conv}: {bad} rows differ")
        _log("[probe] brute-force oracle parity: all 5 conventions exact OK")

        # (a2) train-only whitening invariance (r1 review major
        # retrieval-whitening-eval-leakage, codex-named mechanizable check):
        # perturbing a HELD-OUT (fold-f eval) answer leaves fold f's whitening
        # decomposition bit-unchanged, while every OTHER fold's (which trains
        # on that row) changes.
        folds_arr = np.asarray(entry["folds"], dtype=np.int64)
        f0 = int(folds_arr[0])
        y_pert = np.array(y_chat, copy=True)
        y_pert[0] += np.float32(7.0)
        ctx_pert = build_pool_ctx(y_pert, entry)
        assert np.array_equal(ctx["per_fold"][f0]["mu"], ctx_pert["per_fold"][f0]["mu"])
        assert np.array_equal(ctx["per_fold"][f0]["L"], ctx_pert["per_fold"][f0]["L"])
        others_changed = [
            g
            for g in ctx["per_fold"]
            if g != f0 and not np.array_equal(ctx["per_fold"][g]["L"], ctx_pert["per_fold"][g]["L"])
        ]
        assert len(others_changed) == len(ctx["per_fold"]) - 1, others_changed
        del ctx_pert, y_pert
        _log("[probe] per-fold whitening: held-out perturbation invariance OK")

        # (b) chat/context JSON: G3 battery read + reconciliation.
        chat_ctx = json.loads((ledger / "retrieval" / "chat__context.json").read_text())
        assert chat_ctx["g3_battery_read"]["pass_10x_chance"] is True
        assert chat_ctx["conventions"]["raw_euclidean"]["acc_at_k"]["1"] > 0.9
        assert chat_ctx["reconciliation_vs_producer_knn"]["euclidean"]["pass"] is True
        assert chat_ctx["reconciliation_vs_producer_knn"]["cosine"]["pass"] is True
        assert len(chat_ctx["conventions"]) == 5
        _log("[probe] chat/context unit: G3 battery read + producer reconciliation OK")

        # (c) ladder units present for real targets; skip marker for the
        # scrambled (unmappable) one; prefix fits unit present.
        for ri in (1, 9):
            assert (ledger / "retrieval" / f"chat_to_storyq_astra__rung{ri}.json").exists()
        assert (ledger / "retrieval" / "chat_to_chat_user_real__rung1.json").exists()
        skipped = json.loads(
            (ledger / "retrieval" / "chat_to_plain_text__skipped.json").read_text()
        )
        assert skipped["status"] == "skipped" and "UNMAPPABLE" in skipped["reason"]
        assert (ledger / "retrieval" / "chat__prefix.json").exists()
        rung = json.loads((ledger / "retrieval" / "chat_to_storyq_astra__rung1.json").read_text())
        assert rung["rung"] == "1_direct" and len(rung["conventions"]) == 5
        _log("[probe] ladder units + unmappable skip marker OK")

        # (d) fresh: coverage accounting (planted non-cohort id dropped with
        # count; last seed's missing row shrinks the intersection), high
        # rank-1 ceiling, avg-target legs present; chat_user_real N/A.
        fr = json.loads((ledger / "retrieval" / "chat__fresh.json").read_text())
        assert fr["counts"]["n_dropped_not_in_cohort"] == 1, fr["counts"]
        assert fr["counts"]["n_covered"] == 10, fr["counts"]
        assert fr["fresh_reference"]["raw_euclidean"]["acc1_ceiling_per_context"] > 0.5
        assert fr["avg_target"]["averaged_target"]["raw_euclidean"]["acc_at_k"]["1"] > 0.5
        assert fr["avg_target"]["single_target"]["raw_euclidean"]["n"] == 10
        na = json.loads((ledger / "retrieval" / "chat_user_real__fresh.json").read_text())
        assert na["status"] == "N/A" and "deterministic render" in na["reason"]
        _log(
            f"[probe] fresh legs OK: covered={fr['counts']['n_covered']} "
            f"ceiling={fr['fresh_reference']['raw_euclidean']['acc1_ceiling_per_context']:.3f}"
        )

        # (e) resume: re-run -> every unit skips with matching regime.
        assert phase_run(ret_ns, battery_on=True, fresh_on=True) == 0
        _log("[probe] resume-skip: OK")
    _log("[phase=probe] done — all retrieval probes passed")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.replace("%", "%%"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--phase", choices=("all", "battery", "fresh", "probe"), default="all")
    ap.add_argument("--cells", default="all", help="comma list of cells (default all 13)")
    ap.add_argument(
        "--conventions",
        default="all",
        help=f"comma subset of {CONVENTIONS} (default all; plan §9 descope floor keeps "
        "raw_cosine + whiten_cos + csls_cos)",
    )
    ap.add_argument(
        "--store-root",
        default=str(cm.REPO_ROOT / "data" / "issue_2378" / "activations"),
    )
    ap.add_argument("--ledger-root", default=str(cm.LEDGER_ROOT))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--layer-star-from", default=None)
    ap.add_argument("--g3-gate-file", default=None)
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[retrieval] import-check OK")
        return 0
    if args.phase == "probe":
        return phase_probe(args)
    return phase_run(
        args,
        battery_on=args.phase in ("all", "battery"),
        fresh_on=args.phase in ("all", "fresh"),
    )


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension teardown (code-style.md)
