#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (→, ², λ, γ, ̄, ±) in scientific docstrings + log/print messages.
"""Issue #722 (off-pod CPU, 0 GPU): GROUPED cross-validation of the c_C → v0 map.

The base #722 read used leave-one-CONTEXT-out (LOCO): hold out one of 50 contexts,
train on the other 49, predict it. The linear-ridge skill-over-mean plateaus at
~+0.75-0.80 (L14/18/21). But LOCO is WITHIN-FAMILY interpolation — when you hold
out (say) one of 14 persona contexts, the 6 OTHER persona contexts (plus 9
WildChat, 8 ICL, ...) are still in the training set, so the map can interpolate
inside the held-out context's own family. This script asks the stronger,
EXTRAPOLATION question: does the c_C → v0 map generalize across context FAMILIES?

Two grouped-CV schemes, both reusing the EXACT base metric (skill-over-mean
held-out R² on the train-mean-centered v0 target) but redefining the folds:

  SCHEME 1 — Leave-one-FAMILY-out (LOFO), the headline. 7 folds, one per family
  (persona 14, wildchat 10, icl 8, rephrase 6, format 5, behavior 5, default 2).
  Hold out an ENTIRE family; train on the other 6 families; predict the held-out
  family. This is genuine cross-context-TYPE extrapolation: the held-out family's
  v0s are predicted by a map that never saw that family's c_C. A HOLD of the LOCO
  skill ⇒ the map generalizes across context types (not just within-family
  interpolation); a DROP ⇒ the LOCO result was partly within-family interpolation.

  SCHEME 2 — Leave-one-from-each-family-out (stratified). Each fold's test set =
  one context drawn from EACH family (so all 7 families present in test); train on
  the rest. Capped by the smallest family (default, n=2): 2 disjoint stratified
  folds by default, OR `--strat-mode repeated` for K random stratified folds
  (mean±sd). Less extreme than LOFO (every family is represented in train), a
  bridge between LOCO and LOFO.

Per layer, per scheme:
  - linear-ridge skill-over-mean R² (full-H), SAME metric as the base LOCO read
    (`skill = 1 − SS_res/SS_tot`, SS_tot baseline = the per-fold TRAIN mean — here
    the train families' mean). Aggregate over all held-out predictions AND per
    held-out FAMILY (the 7 individual held-out-family skills for LOFO).
  - the KRR-RBF − linear-kernel gap (PCA-48 target space), the same nonlinear-gap
    statistic as the base LOCO KRR read, but under grouped folds. Does cross-family
    extrapolation reveal nonlinearity that within-family LOCO hid?

Standalone, idempotent, CPU-only. Imports the base LOCO math READ-ONLY from
`explore_persona_space.analysis.vectorized_mlp_skill` (the shared helper) +
`issue658_fit_predictors` (the dual/PRESS ridge inner math) — NEVER edits them.
The grouped folds need a generic train/test split (LOCO's per-fold "all but i"
is a special case), so the train/predict drivers here are NEW (closed-form ridge
+ KRR with nested inner CV on the TRAIN rows only — no held-out leakage), but
every piece of inner linear-algebra is the imported, gated base math.

Writes:
  eval_results/issue_722/grouped_cv/lofo.json
  eval_results/issue_722/grouped_cv/stratified.json
  figures/issue_722/grouped_cv_per_family_lofo.png
  figures/issue_722/grouped_cv_comparison.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse #658's EXACT ridge dual/PRESS inner math (do NOT re-implement).
import issue658_fit_predictors as i658  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue658_fit_predictors import (  # noqa: E402
    RIDGE_LAMBDAS,
    _press_loo_mse_per_lambda,
    _ridge_dual_weights,
)

# Reuse the shared skill-over-mean helper's KRR kernel + PCA machinery READ-ONLY.
from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    _default_rbf_gammas,
    _kernel_gram,
    _krr_loo_press,
    robust_pca_basis,
)

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue722_grouped_cv")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HF_REPO = "superkaiba1/explore-persona-space-data"
V0_FILE = "issue658_theory_assumptions/store/v0_summaries.pt"
CC_LAST_FILE = "issue594_context_geometry/analysis_tensors/context_vectors_mean.pt"
BATTERY_JSON = PROJECT_ROOT / "data" / "issue594" / "battery.json"

# Project-default seed (followup-scope §11); ridge LOCO/grouped is deterministic.
# Only the bootstrap fold-resampling + repeated-stratified fold draws use it.
SEED = 42

# KRR target reduction — match the base #722 KRR read (PCA-48, lossless at n≤50).
MLP_PCA_DIM = 48

# Plateau read layers (primary L18). The base LOCO ridge skill peaks here.
PLATEAU_LAYERS = [14, 18, 21]

# Expected per-family counts (the followup-scope contract; fail loud if violated).
EXPECTED_FAMILY_COUNTS = {
    "persona": 14,
    "wildchat": 10,
    "icl": 8,
    "rephrase": 6,
    "format": 5,
    "behavior": 5,
    "default": 2,
}

# Canonical family display order (largest → smallest, default last). Used for the
# per-family figure x-axis and the JSON family ordering.
FAMILY_ORDER = ["persona", "wildchat", "icl", "rephrase", "format", "behavior", "default"]

N_BOOTSTRAP = 2000


# ── data loading + AUTHORITATIVE family labels ────────────────────────────────


def _load_stores_and_families() -> dict:
    """Download (cached) + load v0, last-input-token c_C, and the family labels.

    The family labels come from the c_C store's own ``families`` field (aligned
    row-for-row with ``instance_ids``, which IS the row order the base loader
    uses), cross-checked PER ID against ``data/issue594/battery.json``'s
    per-instance ``family``. The two sources must agree exactly AND the recovered
    per-family counts must match ``EXPECTED_FAMILY_COUNTS`` — otherwise this STOPS
    (the followup-scope contract: never guess labels). Also asserts the two stores
    share the same probe-pool battery (probe_pool_hash) — the base loader's check.

    Returns a dict with the aligned V (N, L, H), C_last (N, L, H), the ordered
    ctx_ids, layers, and family arrays (per-row family + family→row-index lists).
    """
    from huggingface_hub import hf_hub_download

    v0p = hf_hub_download(HF_REPO, V0_FILE, repo_type="dataset")
    ccp = hf_hub_download(HF_REPO, CC_LAST_FILE, repo_type="dataset")
    v0 = torch.load(v0p, weights_only=False)
    cc = torch.load(ccp, weights_only=False)

    # Probe-pool-hash equality (the base loader's load-bearing substrate check).
    h_v0 = v0.get("probe_pool_hash")
    h_594 = cc.get("probe_pool_hash")
    if h_v0 is None or h_594 is None or h_v0 != h_594:
        raise RuntimeError(
            "probe_pool_hash mismatch between v0 and c_C stores — the two substrates "
            f"do NOT share the same probe battery: v0={h_v0!r} c_C={h_594!r}. "
            "Refusing to fit a cross-store map on misaligned probes."
        )

    ctx_ids = list(v0["context_ids"])
    layers = list(v0["capture_layers"])
    V = np.stack([v0["summaries"]["mean"][c].numpy() for c in ctx_ids])  # (N, L, H)

    iid_to_row = {iid: i for i, iid in enumerate(cc["instance_ids"])}
    missing = [c for c in ctx_ids if c not in iid_to_row]
    if missing:
        raise RuntimeError(f"#594 cc_last store missing {len(missing)} contexts: {missing[:5]}")
    cc_tensor = cc["tensor"]  # (n594, 28, H)
    C_last = np.stack([cc_tensor[iid_to_row[c]].numpy() for c in ctx_ids])  # (N, 28, H)
    assert C_last.shape[1] == len(layers), (C_last.shape, len(layers))

    # ── AUTHORITATIVE family labels, two independent sources, must agree ──
    # Source A: the c_C store's families field, aligned to the store's
    # instance_ids — re-keyed onto our ctx_ids row order.
    if "families" not in cc:
        raise RuntimeError(
            "c_C store has no 'families' field — cannot recover the authoritative "
            "family mapping from the store. STOP (followup-scope: never guess labels)."
        )
    store_fam_by_iid = dict(zip(cc["instance_ids"], cc["families"], strict=True))
    fam_store = [store_fam_by_iid[c] for c in ctx_ids]  # per-row, in ctx_ids order

    # Source B: the battery JSON's per-instance family.
    if not BATTERY_JSON.exists():
        raise RuntimeError(
            f"battery JSON not found at {BATTERY_JSON} — cannot cross-check the "
            "family mapping. STOP (followup-scope: never guess labels)."
        )
    battery = json.loads(BATTERY_JSON.read_text())
    fam_bat_by_id = {inst["id"]: inst["family"] for inst in battery["instances"]}
    fam_battery = [fam_bat_by_id.get(c) for c in ctx_ids]

    # Cross-check the two sources PER ID.
    mismatches = [
        (c, fam_store[i], fam_battery[i])
        for i, c in enumerate(ctx_ids)
        if fam_store[i] != fam_battery[i]
    ]
    if mismatches:
        raise RuntimeError(
            f"family-label mismatch between the c_C store and battery.json on "
            f"{len(mismatches)} context(s): {mismatches[:5]}. The two authoritative "
            "sources disagree — STOP (followup-scope: never guess labels)."
        )

    # Verify the recovered per-family counts match the expected contract.
    counts = Counter(fam_store)
    if dict(counts) != EXPECTED_FAMILY_COUNTS:
        raise RuntimeError(
            f"recovered per-family counts {dict(counts)} != expected "
            f"{EXPECTED_FAMILY_COUNTS}. STOP (followup-scope: never guess labels)."
        )

    families = sorted(counts.keys(), key=lambda f: FAMILY_ORDER.index(f))
    fam_to_rows = {f: [i for i in range(len(ctx_ids)) if fam_store[i] == f] for f in families}

    logger.info(
        "Loaded + verified family labels: n=%d contexts, %d families %s",
        len(ctx_ids),
        len(families),
        {f: len(fam_to_rows[f]) for f in families},
    )

    return {
        "ctx_ids": ctx_ids,
        "layers": layers,
        "V": V.astype(np.float64),
        "C_last": C_last.astype(np.float64),
        "fam_per_row": fam_store,  # per-row family label, ctx_ids order
        "families": families,  # canonical family order (largest → smallest)
        "fam_to_rows": fam_to_rows,  # family → list of row indices
        "v0_path": v0p,
        "cc_path": ccp,
        "store_provenance": {
            "v0_file": f"{HF_REPO}:{V0_FILE}",
            "cc_last_file": f"{HF_REPO}:{CC_LAST_FILE}",
            "family_source": "c_C store 'families' field; cross-checked vs data/issue594/battery.json",  # noqa: E501
            "n_contexts": len(ctx_ids),
            "hidden_dim": int(V.shape[-1]),
            "probe_pool_hash_v0": h_v0,
            "probe_pool_hash_594": h_594,
            "family_counts": dict(counts),
        },
    }


# ── grouped-fold ridge predictor (closed form, nested inner CV on TRAIN rows) ──


def _ridge_predict_grouped(
    Xc: np.ndarray, Yv: np.ndarray, folds: list[tuple[list[int], list[int]]]
) -> tuple[np.ndarray, list[float]]:
    """Closed-form ridge prediction of v0 for arbitrary (train, test) GROUPED folds.

    Generalizes the base LOCO `ridge_predict_loco_centered` (whose per-fold train
    set is "all rows but i") to ANY train/test split. For each (train_idx,
    test_idx) fold:
      - standardize X on the TRAIN rows only (numpy ddof=0 → correction=0, matching
        the base loader);
      - center the v0 TARGET by the TRAIN-rows mean (= the per-fold predict-the-mean
        baseline) and fit the map on the centered target, adding the train mean back
        on prediction (the textbook held-out-R²-over-mean construction);
      - pick λ by NESTED inner leave-one-out PRESS over the TRAIN rows only (#658's
        exact `_press_loo_mse_per_lambda` identity — NO held-out-fold leakage);
      - predict every test row: ŷ = v̄0_train + M̂(c_C).

    Returns (preds (N, H) — every row filled by the fold that holds it out, the
    folds partition the rows), chosen_lambda_per_fold (one λ per fold).
    """
    n, H = Yv.shape
    device = torch.device(i658.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(Xc)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Yv)).to(device=device, dtype=torch.float64)
    preds = np.full((n, H), np.nan, dtype=np.float64)
    chosen_lam: list[float] = []
    for train_idx, test_idx in folds:
        tr_t = torch.tensor(train_idx, device=device)
        te_t = torch.tensor(test_idx, device=device)
        Xtr, Ytr = Xt[tr_t], Yt[tr_t]
        # train-only X standardization (base-loader convention)
        xmu = Xtr.mean(0)
        xsd = Xtr.std(0, correction=0) + 1e-9
        Xtr_n = (Xtr - xmu) / xsd
        # train-only TARGET centering: the per-fold predict-the-mean baseline
        ymu = Ytr.mean(0)
        Ytr_c = Ytr - ymu
        # nested inner LOO-PRESS λ pick on the TRAIN rows only (no held-out leakage)
        mse = _press_loo_mse_per_lambda(Xtr_n, Ytr_c, RIDGE_LAMBDAS)
        best_lam = RIDGE_LAMBDAS[int(torch.argmin(mse).item())]
        w = _ridge_dual_weights(Xtr_n, Ytr_c, best_lam)  # (d, H)
        x_held = (Xt[te_t] - xmu) / xsd  # (n_test, d)
        pred = (ymu + x_held @ w).detach().cpu().numpy()  # (n_test, H)
        preds[test_idx] = pred
        chosen_lam.append(float(best_lam))
    tested = sorted({i for _tr, te in folds for i in te})
    assert not np.isnan(preds[tested]).any(), (
        "grouped ridge left a held-out row unpredicted — a test index was never filled"
    )
    return preds, chosen_lam


# ── grouped-fold KRR predictor (RBF / linear, nested inner CV on TRAIN rows) ──


def _krr_predict_grouped(
    Xc: np.ndarray,
    Yv: np.ndarray,
    folds: list[tuple[list[int], list[int]]],
    *,
    kernel: str = "rbf",
    lambdas: list | None = None,
    gammas: list | None = None,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Grouped-fold kernel-ridge prediction (the grouped analogue of the base
    LOCO `krr_predict_loco`).

    Per (train_idx, test_idx) fold: train-only feature standardization + train-only
    target centering; pick (γ, λ) by NESTED leave-one-out PRESS over the TRAIN
    block only (the exact closed-form `_krr_loo_press` from the shared helper — no
    held-out leakage); predict every test row; add the train target mean back.
    Reuses the shared helper's `_kernel_gram` / `_krr_loo_press` / `_default_rbf_gammas`
    READ-ONLY. Returns (preds (N, P), chosen_lambda_per_fold, chosen_gamma_per_fold).
    """
    lambdas = lambdas if lambdas is not None else list(RIDGE_LAMBDAS)
    if gammas is None:
        gammas = _default_rbf_gammas(Xc) if kernel == "rbf" else [0.0]
    if kernel == "linear":
        gammas = [0.0]
    n, P = Yv.shape
    device = torch.device(i658.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(Xc)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Yv)).to(device=device, dtype=torch.float64)
    preds = np.full((n, P), np.nan, dtype=np.float64)
    chosen_lam: list[float] = []
    chosen_gam: list[float] = []
    for train_idx, test_idx in folds:
        tr_t = torch.tensor(train_idx, device=device)
        te_t = torch.tensor(test_idx, device=device)
        Xtr, Ytr = Xt[tr_t], Yt[tr_t]
        xmu = Xtr.mean(0)
        xsd = Xtr.std(0, correction=0) + 1e-9
        Xtr_n = (Xtr - xmu) / xsd
        ymu = Ytr.mean(0)
        Ytr_c = Ytr - ymu
        x_held = (Xt[te_t] - xmu) / xsd  # (n_test, d)

        best = None  # (press, lam, gam, alpha, K_test_train)
        for gam in gammas:
            Ktr = _kernel_gram(Xtr_n, Xtr_n, kernel, gam)  # (m, m)
            k_test = _kernel_gram(x_held, Xtr_n, kernel, gam)  # (n_test, m)
            for lam in lambdas:
                press = _krr_loo_press(Ktr, Ytr_c, lam)
                if best is None or press < best[0]:
                    A = torch.linalg.solve(
                        Ktr + lam * torch.eye(Ktr.shape[0], device=device, dtype=torch.float64),
                        Ytr_c,
                    )  # (m, P) dual coeffs
                    best = (press, lam, gam, A, k_test)
        _press, lam, gam, A, k_test = best
        pred = (ymu + k_test @ A).detach().cpu().numpy()  # (n_test, P)
        preds[test_idx] = pred
        chosen_lam.append(float(lam))
        chosen_gam.append(float(gam))
    tested = sorted({i for _tr, te in folds for i in te})
    assert not np.isnan(preds[tested]).any(), (
        "grouped KRR left a held-out row unpredicted — a test index was never filled"
    )
    return preds, chosen_lam, chosen_gam


# ── skill metric (aggregate + per-fold terms, train-family-mean baseline) ──────


def _skill_with_fold_baselines(
    preds: np.ndarray, Y: np.ndarray, folds: list[tuple[list[int], list[int]]]
) -> dict:
    """Aggregate held-out skill-over-mean R² for GROUPED folds, plus per-row terms.

    skill = 1 − SS_res / SS_tot where, for each held-out row i (held out by some
    fold f), SS_tot uses that fold's TRAIN-rows mean ȳ_train(f) as the baseline —
    NOT the global mean — so the "predict-the-mean" baseline is honest for the
    grouped split (the same semantics as the base `skill_over_mean_r2`, generalized
    to grouped folds where the train set is the OTHER families/rows). Aggregated
    (variance-weighted) over all dims AND all held-out rows = one skill per scheme.

    Returns the aggregate skill (over the HELD-OUT rows only) + per-ROW (ss_res,
    ss_tot) (n,) for the bootstrap (zero for rows never held out) + the per-row
    held-out mask. LOFO folds partition all 50 rows (every row held exactly once);
    the stratified disjoint folds hold only a balanced cap×n_families subset, so
    the aggregate + bootstrap restrict to the held rows. A row held by >1 fold is
    a fold-construction bug (asserted: no double-hold).
    """
    n, _H = Y.shape
    ss_res = np.zeros(n, dtype=np.float64)
    ss_tot = np.zeros(n, dtype=np.float64)
    held = np.zeros(n, dtype=bool)
    for train_idx, test_idx in folds:
        ybar_train = Y[train_idx].mean(axis=0)  # (H,) per-fold train-mean baseline
        for i in test_idx:
            assert not held[i], f"skill: row {i} held out by >1 fold (folds not disjoint)"
            res = Y[i] - preds[i]
            tot = Y[i] - ybar_train
            ss_res[i] = float(res @ res)
            ss_tot[i] = float(tot @ tot)
            held[i] = True
    assert held.any(), "skill: no rows held out (empty folds)"
    sr = ss_res[held].sum()
    st = ss_tot[held].sum()
    agg = float("nan") if st < 1e-12 else 1.0 - sr / st
    return {"skill": agg, "ss_res": ss_res, "ss_tot": ss_tot, "held": held}


def _per_family_skill(
    ss_res: np.ndarray, ss_tot: np.ndarray, fam_to_rows: dict[str, list[int]], families: list[str]
) -> dict[str, dict]:
    """Per held-out-FAMILY skill (the 7 individual held-out-family skills, LOFO).

    For LOFO each family IS a held-out fold, so the per-family skill is the
    aggregate skill restricted to that family's rows (using each row's already
    train-family-mean-baselined SS terms). Returns {family: {skill, n, ss_res, ss_tot}}.
    """
    out: dict[str, dict] = {}
    for f in families:
        rows = fam_to_rows[f]
        sr = float(ss_res[rows].sum())
        st = float(ss_tot[rows].sum())
        out[f] = {
            "skill": float("nan") if st < 1e-12 else 1.0 - sr / st,
            "n": len(rows),
            "ss_res": sr,
            "ss_tot": st,
        }
    return out


def _bootstrap_skill_ci(ss_res: np.ndarray, ss_tot: np.ndarray, rng, n_boot: int) -> list[float]:
    """Percentile bootstrap CI on the aggregate skill by resampling held-out rows."""
    n = ss_tot.shape[0]
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tot = ss_tot[idx].sum()
        vals[b] = float("nan") if tot < 1e-12 else 1.0 - ss_res[idx].sum() / tot
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _bootstrap_gap_ci(
    ss_res_a: np.ndarray, ss_res_b: np.ndarray, ss_tot: np.ndarray, rng, n_boot: int
) -> tuple[list[float], bool]:
    """Bootstrap CI on the skill gap (A − B), resampling held-out rows (shared SS_tot).

    Mirrors the base KRR read's `_bootstrap_gap`: SS_tot is shared (same target),
    so resample the per-row (ss_res_a, ss_res_b, ss_tot) triples together.
    Returns (ci95, excludes_zero).
    """
    n = ss_tot.shape[0]
    gaps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tot = ss_tot[idx].sum()
        if tot < 1e-12:
            gaps[b] = float("nan")
            continue
        skill_a = 1.0 - ss_res_a[idx].sum() / tot
        skill_b = 1.0 - ss_res_b[idx].sum() / tot
        gaps[b] = skill_a - skill_b
    gaps = gaps[np.isfinite(gaps)]
    if gaps.size == 0:
        return [float("nan"), float("nan")], False
    lo, hi = float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))
    return [lo, hi], bool(lo > 0.0 or hi < 0.0)


# ── fold builders ─────────────────────────────────────────────────────────────


def _lofo_folds(fam_to_rows: dict[str, list[int]], families: list[str]) -> list[tuple[list, list]]:
    """Leave-one-FAMILY-out folds: one fold per family (test = that family's rows,
    train = all OTHER families' rows). 7 folds, the row partition is the families."""
    all_rows = sorted(r for rows in fam_to_rows.values() for r in rows)
    folds = []
    for f in families:
        test = sorted(fam_to_rows[f])
        train = sorted(set(all_rows) - set(test))
        folds.append((train, test))
    return folds


def _stratified_folds_disjoint(
    fam_to_rows: dict[str, list[int]], families: list[str], rng
) -> list[tuple[list, list]]:
    """Disjoint stratified folds: each fold's test = one row from EACH family, folds
    disjoint, capped by the SMALLEST family. With default=2 → exactly 2 folds.

    Shuffles each family's rows once (seeded), then fold k's test set is row k of
    every family. Every family appears in every fold's test set; folds partition
    the rows used (rows past the cap in larger families spill into LATER folds'
    train sets only — i.e. they are train-only for the K capped folds, never held
    out — so the K disjoint folds are NOT a full partition of all 50 rows; the
    skill is over the held-out rows actually drawn). To keep the skill honest +
    bootstrap-valid we restrict to a clean K-fold PARTITION of a balanced subset:
    each family contributes its first K rows (K = smallest-family size), one per
    fold; the remaining rows of larger families are dropped from BOTH train and
    test of these folds.

    NOTE: dropping the surplus is what makes each fold a valid disjoint test set
    with all-families-in-test; the cost is the larger families are under-used. The
    repeated-stratified mode (below) recovers that by re-drawing which rows fill
    the K slots across repeats.
    """
    cap = min(len(fam_to_rows[f]) for f in families)  # = default family size (2)
    shuffled = {f: list(rng.permutation(fam_to_rows[f])) for f in families}
    # balanced subset: first `cap` rows of each family, assigned one-per-fold
    used_rows = sorted(int(r) for f in families for r in shuffled[f][:cap])
    folds = []
    for k in range(cap):
        test = sorted(int(shuffled[f][k]) for f in families)
        train = sorted(set(used_rows) - set(test))
        folds.append((train, test))
    return folds


def _stratified_folds_repeated(
    fam_to_rows: dict[str, list[int]], families: list[str], rng, n_repeats: int
) -> list[list[tuple[list, list]]]:
    """Repeated random stratified folds: `n_repeats` independent draws of the
    disjoint-stratified scheme above (each re-shuffles which rows fill the K
    cap-sized slots), so over repeats every row of the larger families is used.
    Returns a list of fold-lists (one per repeat) — the caller aggregates
    mean±sd of the per-repeat skill."""
    return [_stratified_folds_disjoint(fam_to_rows, families, rng) for _ in range(n_repeats)]


# ── scheme runners ────────────────────────────────────────────────────────────


def run_lofo(data: dict, layers_subset: list[int] | None, do_krr: bool, n_boot: int) -> dict:
    """Scheme 1 — leave-one-FAMILY-out, all layers (or a subset for the smoke)."""
    layers = data["layers"]
    V = data["V"]
    C = data["C_last"]
    families = data["families"]
    fam_to_rows = data["fam_to_rows"]
    folds = _lofo_folds(fam_to_rows, families)
    n, L, H = V.shape
    logger.info(
        "[LOFO] n=%d L=%d H=%d | 7 family folds %s",
        n,
        L,
        H,
        {f: len(fam_to_rows[f]) for f in families},
    )

    per_layer = []
    for li in range(L):
        layer = int(layers[li])
        if layers_subset is not None and layer not in layers_subset:
            continue
        t0 = time.time()
        Xc = C[:, li, :]
        Yv = V[:, li, :]

        # linear-ridge full-H skill (the headline)
        ridge_pred, ridge_lam = _ridge_predict_grouped(Xc, Yv, folds)
        rs = _skill_with_fold_baselines(ridge_pred, Yv, folds)
        per_fam = _per_family_skill(rs["ss_res"], rs["ss_tot"], fam_to_rows, families)
        ridge_ci = _bootstrap_skill_ci(
            rs["ss_res"], rs["ss_tot"], np.random.default_rng(SEED), n_boot
        )

        row: dict[str, Any] = {
            "layer": layer,
            "ridge_skill_aggregate": rs["skill"],
            "ridge_skill_ci95": ridge_ci,
            "ridge_lambda_median": float(np.median(ridge_lam)),
            "per_family_ridge_skill": {f: per_fam[f]["skill"] for f in families},
            "per_family_n": {f: per_fam[f]["n"] for f in families},
        }

        if do_krr:
            # KRR gap in PCA-48 target space (same target as base KRR read)
            mu_t, comps, _ = robust_pca_basis(Yv, MLP_PCA_DIM)
            Y48 = (Yv - mu_t) @ comps.T  # (n, 48)
            rbf_pred, rbf_lam, rbf_gam = _krr_predict_grouped(Xc, Y48, folds, kernel="rbf")
            lin_pred, lin_lam, _ = _krr_predict_grouped(Xc, Y48, folds, kernel="linear")
            rbf_s = _skill_with_fold_baselines(rbf_pred, Y48, folds)
            lin_s = _skill_with_fold_baselines(lin_pred, Y48, folds)
            gap = rbf_s["skill"] - lin_s["skill"]
            gap_ci, gap_excl = _bootstrap_gap_ci(
                rbf_s["ss_res"],
                lin_s["ss_res"],
                rbf_s["ss_tot"],
                np.random.default_rng(SEED),
                n_boot,
            )
            row.update(
                {
                    "skill_krr_rbf_pca48": rbf_s["skill"],
                    "skill_krr_linear_pca48": lin_s["skill"],
                    "nonlinear_gap_rbf_minus_linear": gap,
                    "gap_ci95": gap_ci,
                    "gap_excludes_zero": gap_excl,
                    "chosen_gamma_rbf_median": float(np.median(rbf_gam)),
                    "chosen_lambda_rbf_median": float(np.median(rbf_lam)),
                    "chosen_lambda_linear_median": float(np.median(lin_lam)),
                }
            )

        per_layer.append(row)
        logger.info(
            "[LOFO L%02d] ridge=%+.4f CI=[%+.4f,%+.4f]%s | %.1fs",
            layer,
            row["ridge_skill_aggregate"],
            ridge_ci[0],
            ridge_ci[1],
            (
                f" | rbf-lin gap={row.get('nonlinear_gap_rbf_minus_linear', float('nan')):+.4f}"
                + (" *EXCL0*" if row.get("gap_excludes_zero") else "")
            )
            if do_krr
            else "",
            time.time() - t0,
        )

    return {
        "scheme": "leave-one-family-out (LOFO)",
        "metric": "skill_over_predict_the_mean = 1 - SS_res/SS_tot (held-out R² on centered v0); "
        "SS_tot baseline = the TRAINING-FAMILIES mean per fold",
        "c_C_recipe": "C_last",
        "n_contexts": n,
        "activation_dim": H,
        "families": families,
        "family_counts": {f: len(fam_to_rows[f]) for f in families},
        "n_folds": len(folds),
        "krr_pca_target_dim": MLP_PCA_DIM if do_krr else None,
        "ridge_lambdas": list(RIDGE_LAMBDAS),
        "n_bootstrap": n_boot,
        "seed": SEED,
        "store_provenance": data["store_provenance"],
        "per_layer": per_layer,
    }


def run_stratified(
    data: dict,
    layers_subset: list[int] | None,
    do_krr: bool,
    n_boot: int,
    mode: str,
    n_repeats: int,
) -> dict:
    """Scheme 2 — leave-one-from-each-family-out (stratified).

    mode='disjoint' → 2 disjoint stratified folds (cap = smallest family = 2),
    one skill per layer. mode='repeated' → `n_repeats` random stratified draws,
    each yielding a per-layer skill; report mean±sd over repeats.
    """
    layers = data["layers"]
    V = data["V"]
    C = data["C_last"]
    families = data["families"]
    fam_to_rows = data["fam_to_rows"]
    n, L, H = V.shape
    cap = min(len(fam_to_rows[f]) for f in families)
    logger.info("[STRAT] mode=%s n=%d L=%d cap(per-fold)=%d repeats=%d", mode, n, L, cap, n_repeats)

    if mode == "disjoint":
        fold_sets = [_stratified_folds_disjoint(fam_to_rows, families, np.random.default_rng(SEED))]
    elif mode == "repeated":
        fold_sets = _stratified_folds_repeated(
            fam_to_rows, families, np.random.default_rng(SEED), n_repeats
        )
    else:
        raise ValueError(f"unknown strat-mode {mode!r}")

    per_layer = []
    for li in range(L):
        layer = int(layers[li])
        if layers_subset is not None and layer not in layers_subset:
            continue
        t0 = time.time()
        Xc = C[:, li, :]
        Yv = V[:, li, :]

        ridge_skills: list[float] = []
        gaps: list[float] = []
        rbf_skills: list[float] = []
        lin_skills: list[float] = []
        # for the disjoint single-fold-set we also keep per-row SS for a bootstrap CI
        first_rs = None
        first_rbf = None
        first_lin = None
        if do_krr:
            mu_t, comps, _ = robust_pca_basis(Yv, MLP_PCA_DIM)
            Y48 = (Yv - mu_t) @ comps.T
        for r_i, folds in enumerate(fold_sets):
            ridge_pred, _ = _ridge_predict_grouped(Xc, Yv, folds)
            rs = _skill_with_fold_baselines(ridge_pred, Yv, folds)
            ridge_skills.append(rs["skill"])
            if r_i == 0:
                first_rs = rs
            if do_krr:
                rbf_pred, _, _ = _krr_predict_grouped(Xc, Y48, folds, kernel="rbf")
                lin_pred, _, _ = _krr_predict_grouped(Xc, Y48, folds, kernel="linear")
                rbf_s = _skill_with_fold_baselines(rbf_pred, Y48, folds)
                lin_s = _skill_with_fold_baselines(lin_pred, Y48, folds)
                rbf_skills.append(rbf_s["skill"])
                lin_skills.append(lin_s["skill"])
                gaps.append(rbf_s["skill"] - lin_s["skill"])
                if r_i == 0:
                    first_rbf, first_lin = rbf_s, lin_s

        row: dict[str, Any] = {
            "layer": layer,
            "ridge_skill_mean": float(np.mean(ridge_skills)),
            "ridge_skill_sd": float(np.std(ridge_skills)) if len(ridge_skills) > 1 else 0.0,
            "ridge_skill_per_repeat": [float(x) for x in ridge_skills],
            "n_repeats": len(fold_sets),
            "n_rows_held_per_fold": len(families),
            "cap_per_fold": cap,
        }
        # bootstrap CI on the FIRST fold-set's row-held SS (only meaningful when a
        # single representative draw — for repeated mode the across-repeat sd is
        # the dispersion read; the bootstrap CI uses the first draw's held rows).
        if first_rs is not None:
            held_mask = first_rs["ss_tot"] > 0
            row["ridge_skill_ci95_firstdraw"] = _bootstrap_skill_ci(
                first_rs["ss_res"][held_mask],
                first_rs["ss_tot"][held_mask],
                np.random.default_rng(SEED),
                n_boot,
            )
        if do_krr:
            row.update(
                {
                    "skill_krr_rbf_mean": float(np.mean(rbf_skills)),
                    "skill_krr_linear_mean": float(np.mean(lin_skills)),
                    "nonlinear_gap_rbf_minus_linear_mean": float(np.mean(gaps)),
                    "nonlinear_gap_rbf_minus_linear_sd": float(np.std(gaps))
                    if len(gaps) > 1
                    else 0.0,
                    "gap_per_repeat": [float(x) for x in gaps],
                }
            )
            if first_rbf is not None and first_lin is not None:
                held_mask = first_rbf["ss_tot"] > 0
                gap_ci, gap_excl = _bootstrap_gap_ci(
                    first_rbf["ss_res"][held_mask],
                    first_lin["ss_res"][held_mask],
                    first_rbf["ss_tot"][held_mask],
                    np.random.default_rng(SEED),
                    n_boot,
                )
                row["gap_ci95_firstdraw"] = gap_ci
                row["gap_excludes_zero_firstdraw"] = gap_excl

        per_layer.append(row)
        logger.info(
            "[STRAT L%02d] ridge mean=%+.4f sd=%.4f (over %d %s)%s | %.1fs",
            layer,
            row["ridge_skill_mean"],
            row["ridge_skill_sd"],
            len(fold_sets),
            "repeats" if mode == "repeated" else "disjoint folds-set",
            (f" | rbf-lin gap mean={row.get('nonlinear_gap_rbf_minus_linear_mean', np.nan):+.4f}")
            if do_krr
            else "",
            time.time() - t0,
        )

    return {
        "scheme": f"leave-one-from-each-family-out (stratified, mode={mode})",
        "metric": "skill_over_predict_the_mean = 1 - SS_res/SS_tot (held-out R² on centered v0); "
        "SS_tot baseline = the per-fold TRAIN mean (the rest)",
        "c_C_recipe": "C_last",
        "n_contexts": n,
        "activation_dim": H,
        "families": families,
        "family_counts": {f: len(fam_to_rows[f]) for f in families},
        "strat_mode": mode,
        "n_repeats": n_repeats if mode == "repeated" else 1,
        "cap_per_fold": cap,
        "krr_pca_target_dim": MLP_PCA_DIM if do_krr else None,
        "ridge_lambdas": list(RIDGE_LAMBDAS),
        "n_bootstrap": n_boot,
        "seed": SEED,
        "store_provenance": data["store_provenance"],
        "per_layer": per_layer,
    }


# ── LOCO reference (read the base canonical JSON for the reference line) ──────


def _loco_ridge_by_layer() -> dict[int, float]:
    """The base LOCO ridge skill-over-mean per layer from the canonical JSON.

    Read-only — the canonical base read lives at
    eval_results/issue_722/base-skill-over-mean-cC-to-v0/skill_over_mean.json.
    Returns {layer: skill_vs_mean_ridge} (empty if the JSON is absent — the
    figures then omit the LOCO reference line, logged as a warning)."""
    p = (
        PROJECT_ROOT
        / "eval_results"
        / "issue_722"
        / "base-skill-over-mean-cC-to-v0"
        / "skill_over_mean.json"
    )
    if not p.exists():
        logger.warning("base LOCO JSON not found at %s — LOCO reference line omitted", p)
        return {}
    d = json.loads(p.read_text())
    return {int(r["layer"]): float(r["skill_vs_mean_ridge"]) for r in d["per_layer"]}


def _loco_krr_gap_by_layer() -> dict[int, float]:
    """The base LOCO KRR (RBF − linear) gap per layer, if the krr_vs_linear JSON
    exists. Returns {layer: gap} or empty."""
    p = PROJECT_ROOT / "eval_results" / "issue_722" / "krr_vs_linear.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {
        int(r["layer"]): float(r["nonlinear_gap_rbf_minus_linear"]) for r in d.get("per_layer", [])
    }


# ── figures (paper-plots style) ──────────────────────────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_fig_meta(fig_path: Path, payload: dict) -> None:
    meta = {
        "issue": 722,
        "figure": fig_path.name,
        "code_sha": _git_sha(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    with open(fig_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def make_per_family_figure(lofo: dict, fig_path: Path, primary_layer: int = 18) -> Path:
    """Per-family held-out linear-ridge skill bar (7 bars, one per family) at the
    primary plateau layer, with the LOCO aggregate ridge skill as a reference line."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")

    rows = {int(r["layer"]): r for r in lofo["per_layer"]}
    if primary_layer not in rows:
        primary_layer = sorted(rows)[len(rows) // 2]
    r = rows[primary_layer]
    families = lofo["families"]
    skills = [r["per_family_ridge_skill"][f] for f in families]
    ns = [r["per_family_n"][f] for f in families]
    labels = [f"{f}\n(n={n})" for f, n in zip(families, ns, strict=True)]

    loco_ridge = _loco_ridge_by_layer().get(primary_layer)
    lofo_agg = r["ridge_skill_aggregate"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(families))
    colors = ["#0072B2" if n > 2 else "#999999" for n in ns]  # grey the small (n=2) family
    ax.bar(x, skills, color=colors, width=0.7)
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    if loco_ridge is not None:
        ax.axhline(
            loco_ridge,
            color="#D55E00",
            lw=1.4,
            ls="--",
            label=f"LOCO ridge skill (L{primary_layer})",
        )
    ax.axhline(
        lofo_agg, color="#009E73", lw=1.4, ls="-", label=f"LOFO aggregate skill (L{primary_layer})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("held-out skill-over-mean (R²)")
    ax.set_xlabel("held-out family")
    ax.set_title(f"LOFO per-family held-out ridge skill — L{primary_layer}", fontsize=9)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", fig_path)
    _write_fig_meta(
        fig_path,
        {
            "primary_layer": primary_layer,
            "bars": "per-held-out-family LOFO linear-ridge skill",
            "ref_lines": ["LOCO ridge skill", "LOFO aggregate skill"],
            "small_family_greyed": "default (n=2)",
            "source_json": "eval_results/issue_722/grouped_cv/lofo.json",
        },
    )
    return fig_path


def make_comparison_figure(lofo: dict, strat: dict, fig_path: Path) -> Path:
    """LOCO vs LOFO-aggregate vs stratified ridge skill (+ KRR gaps) at the plateau layers."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")

    loco_ridge = _loco_ridge_by_layer()
    loco_gap = _loco_krr_gap_by_layer()
    lofo_rows = {int(r["layer"]): r for r in lofo["per_layer"]}
    strat_rows = {int(r["layer"]): r for r in strat["per_layer"]}
    layers = [li for li in PLATEAU_LAYERS if li in lofo_rows and li in strat_rows]
    if not layers:
        layers = sorted(set(lofo_rows) & set(strat_rows))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    x = np.arange(len(layers))
    w = 0.26

    loco_vals = [loco_ridge.get(li, np.nan) for li in layers]
    lofo_vals = [lofo_rows[li]["ridge_skill_aggregate"] for li in layers]
    strat_vals = [strat_rows[li]["ridge_skill_mean"] for li in layers]
    strat_err = [strat_rows[li].get("ridge_skill_sd", 0.0) for li in layers]
    lofo_ci = [lofo_rows[li].get("ridge_skill_ci95", [np.nan, np.nan]) for li in layers]
    lofo_err = np.array(
        [
            [v - c[0] for v, c in zip(lofo_vals, lofo_ci, strict=True)],
            [c[1] - v for v, c in zip(lofo_vals, lofo_ci, strict=True)],
        ]
    )
    lofo_err = np.clip(lofo_err, 0.0, None)

    ax_top.bar(x - w, loco_vals, w, color="#D55E00", label="LOCO (within-family)")
    ax_top.bar(
        x, lofo_vals, w, yerr=lofo_err, capsize=3, color="#0072B2", label="LOFO (cross-family)"
    )
    ax_top.bar(
        x + w,
        strat_vals,
        w,
        yerr=strat_err,
        capsize=3,
        color="#009E73",
        label="stratified (1/family)",
    )
    ax_top.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax_top.set_ylabel("ridge skill-over-mean (R²)")
    ax_top.set_title("c_C → v0 ridge skill: LOCO vs LOFO vs stratified", fontsize=9)
    ax_top.legend(loc="lower left", fontsize=7)

    # bottom: KRR (RBF − linear) gap per scheme (if KRR present)
    has_krr = all("nonlinear_gap_rbf_minus_linear" in lofo_rows[li] for li in layers)
    if has_krr:
        loco_gap_vals = [loco_gap.get(li, np.nan) for li in layers]
        lofo_gap_vals = [lofo_rows[li]["nonlinear_gap_rbf_minus_linear"] for li in layers]
        lofo_gap_ci = [lofo_rows[li].get("gap_ci95", [np.nan, np.nan]) for li in layers]
        lofo_gap_err = np.clip(
            np.array(
                [
                    [v - c[0] for v, c in zip(lofo_gap_vals, lofo_gap_ci, strict=True)],
                    [c[1] - v for v, c in zip(lofo_gap_vals, lofo_gap_ci, strict=True)],
                ]
            ),
            0.0,
            None,
        )
        strat_gap_vals = [
            strat_rows[li].get("nonlinear_gap_rbf_minus_linear_mean", np.nan) for li in layers
        ]
        strat_gap_err = [
            strat_rows[li].get("nonlinear_gap_rbf_minus_linear_sd", 0.0) for li in layers
        ]
        ax_bot.bar(x - w, loco_gap_vals, w, color="#D55E00", label="LOCO")
        ax_bot.bar(x, lofo_gap_vals, w, yerr=lofo_gap_err, capsize=3, color="#0072B2", label="LOFO")
        ax_bot.bar(
            x + w,
            strat_gap_vals,
            w,
            yerr=strat_gap_err,
            capsize=3,
            color="#009E73",
            label="stratified",
        )
        ax_bot.axhline(0.0, color="0.6", lw=0.8, ls=":")
        ax_bot.set_ylabel("nonlinear gap\n(KRR-RBF − linear)")
        ax_bot.legend(loc="upper left", fontsize=7)
    else:
        ax_bot.text(
            0.5,
            0.5,
            "KRR not computed (--no-krr)",
            ha="center",
            va="center",
            transform=ax_bot.transAxes,
        )
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([f"L{li}" for li in layers])
    ax_bot.set_xlabel("layer")

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", fig_path)
    _write_fig_meta(
        fig_path,
        {
            "top_panel": "ridge skill: LOCO vs LOFO-aggregate vs stratified (plateau layers)",
            "bottom_panel": "KRR-RBF − linear nonlinear gap per scheme",
            "source_json": [
                "eval_results/issue_722/grouped_cv/lofo.json",
                "eval_results/issue_722/grouped_cv/stratified.json",
            ],
        },
    )
    return fig_path


# ── reproducibility metadata ──────────────────────────────────────────────────


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _attach_meta(result: dict, data: dict, wall_s: float) -> dict:
    result["run_meta"] = {
        "issue": 722,
        "followup_label": "grouped-cv-cC-to-v0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha": _git_sha(),
        "i658_module_sha": _file_sha256(PROJECT_ROOT / "scripts" / "issue658_fit_predictors.py"),
        "substrate": {
            "v0_local_sha256": _file_sha256(data["v0_path"]),
            "cc_local_sha256": _file_sha256(data["cc_path"]),
            "probe_pool_hash": data["store_provenance"]["probe_pool_hash_v0"],
        },
        "seed": SEED,
        "wall_time_s": round(wall_s, 2),
    }
    return result


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
    tmp.replace(path)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #722 grouped (cross-family) CV of c_C → v0.")
    ap.add_argument("--smoke", action="store_true", help="2-layer (L14, L18) smoke validation")
    ap.add_argument(
        "--layers", type=int, nargs="*", default=None, help="layer subset (default: all 28)"
    )
    ap.add_argument("--no-krr", action="store_true", help="skip the KRR gap arm (ridge only)")
    ap.add_argument(
        "--strat-mode",
        choices=["disjoint", "repeated"],
        default="repeated",
        help="stratified scheme: disjoint (2 capped folds) or repeated (N random draws, mean±sd)",
    )
    ap.add_argument(
        "--strat-repeats", type=int, default=10, help="N draws for --strat-mode repeated"
    )
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP, help="bootstrap resamples for CIs")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_722" / "grouped_cv",
        help="output dir for lofo.json / stratified.json",
    )
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "issue_722",
        help="output dir for figures",
    )
    ap.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    args = ap.parse_args()

    i658.DEVICE = "cpu"
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    do_krr = not args.no_krr
    n_boot = args.n_boot
    layers_subset = [14, 18] if args.smoke else args.layers

    t0 = time.time()
    data = _load_stores_and_families()

    logger.info("=== Scheme 1: leave-one-FAMILY-out (LOFO) ===")
    lofo = run_lofo(data, layers_subset, do_krr, n_boot)
    lofo = _attach_meta(lofo, data, time.time() - t0)
    _atomic_write_json(args.out_dir / "lofo.json", lofo)
    logger.info("wrote %s", args.out_dir / "lofo.json")

    logger.info("=== Scheme 2: leave-one-from-each-family-out (stratified) ===")
    t1 = time.time()
    strat = run_stratified(data, layers_subset, do_krr, n_boot, args.strat_mode, args.strat_repeats)
    strat = _attach_meta(strat, data, time.time() - t1)
    _atomic_write_json(args.out_dir / "stratified.json", strat)
    logger.info("wrote %s", args.out_dir / "stratified.json")

    if not args.no_figures:
        primary = (
            18
            if (layers_subset is None or 18 in layers_subset)
            else sorted({int(r["layer"]) for r in lofo["per_layer"]})[0]
        )
        make_per_family_figure(
            lofo, args.fig_dir / "grouped_cv_per_family_lofo.png", primary_layer=primary
        )
        make_comparison_figure(lofo, strat, args.fig_dir / "grouped_cv_comparison.png")

    # ── console summary ──
    loco_ridge = _loco_ridge_by_layer()
    print("\n=== GROUPED CV — ridge skill-over-mean (held-out R²) ===")
    print(f"{'layer':>5} | {'LOCO':>8} | {'LOFO':>8} | {'LOFO CI95':>20} | {'strat(mean±sd)':>16}")
    for r in lofo["per_layer"]:
        li = r["layer"]
        sr = next((s for s in strat["per_layer"] if s["layer"] == li), None)
        ci = r.get("ridge_skill_ci95", [float("nan"), float("nan")])
        strat_str = f"{sr['ridge_skill_mean']:+.4f}±{sr['ridge_skill_sd']:.3f}" if sr else "n/a"
        print(
            f"{li:>5} | {loco_ridge.get(li, float('nan')):>+8.4f} | "
            f"{r['ridge_skill_aggregate']:>+8.4f} | "
            f"[{ci[0]:+.4f},{ci[1]:+.4f}] | {strat_str:>16}"
        )
    if do_krr:
        print("\n=== LOFO per-layer KRR-RBF − linear nonlinear gap ===")
        for r in lofo["per_layer"]:
            print(
                f"  L{r['layer']:02d}: gap={r['nonlinear_gap_rbf_minus_linear']:+.4f} "
                f"CI=[{r['gap_ci95'][0]:+.4f},{r['gap_ci95'][1]:+.4f}]"
                f"{' *EXCLUDES0*' if r['gap_excludes_zero'] else ''}"
            )
    # LOFO per-family at the primary plateau layer
    primary = (
        18 if any(r["layer"] == 18 for r in lofo["per_layer"]) else lofo["per_layer"][0]["layer"]
    )
    pr = next(r for r in lofo["per_layer"] if r["layer"] == primary)
    print(f"\n=== LOFO per-held-out-family ridge skill (L{primary}) ===")
    for f in lofo["families"]:
        print(f"  {f:>10} (n={pr['per_family_n'][f]:>2}): {pr['per_family_ridge_skill'][f]:+.4f}")
    print(f"\nwrote {args.out_dir / 'lofo.json'} + {args.out_dir / 'stratified.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
