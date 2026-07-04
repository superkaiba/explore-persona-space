"""Issue #952 Phase 2 (VM, CPU): bootstrap + sign-flip batteries over per_context_stats.npz.

Plan §6 draw-battery contract — BOTH batteries operate on PRECOMPUTED per-context
arrays (the ridge maps were fitted ONCE in Phase 1e; a draw only re-weights /
re-signs stored per-context statistics; NO per-draw refit, NO per-draw Python
loop):

  * Bootstrap CIs: n_draws x ~440 statistic columns as TWO stacked-draw GEMMs —
    a (n_draws, n_test) multinomial count matrix W @ the (n_test, C) scattered
    ss_res and ss_tot stacks, then the elementwise R² map. Mirrors the batched
    ``analysis/null_battery.py::perm_null_draws`` pattern (all draws as one
    masked matmul over a precomputed pool — the #834 vectorization of #778);
    that helper's label-shuffle form does not match a bootstrap, so the
    equivalent one-GEMM form is written here with a 3-cell serial-oracle parity
    check (vectorize-many-cell-fits rule item 6).
  * H3 sign-flip permutation null: ONE (n_draws, n_pairs) ±1 sign matrix @ the
    per-pair (drop_ext - drop_own) vector, category-masked; Holm-Bonferroni
    across kept categories. Median + 10%-trimmed-mean companions batch via a
    gathered (n_draws, n_pairs) resample + axis-1 reductions (rank statistics
    batch along the draw axis, not via GEMM).

Row-coverage asserts (plan §3, BEFORE any statistic):
  * H2 — per registered t-pair, the npz ``M{t2}_ctx_ids`` set must EQUAL the
    registered common subset recomputed from the spans files (extended span
    >= t2+16 in ALL arms, intersected with the test split), and every
    cleg/zleg per-context array must cover exactly that set for both arms.
  * H3 — the npz ``bank_div_ids`` / ``bank_ctl_ids`` sets must EQUAL the
    kept-pair member sets from ``divergence_bank_verification.json``.

Seeds (plan §10): bootstrap ``default_rng(0)``; sign-flip ``default_rng(1)``.

Usage (production):
  uv run python scripts/issue952_stats.py \
    --eval-dir eval_results/issue_952 \
    --npz data/issue_952/analysis_tensors/per_context_stats.npz \
    --spans-dir data/issue_952/analysis_tensors

Smoke: add --smoke (tolerates sub-floor n; identical code path).
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import subprocess
import sys
import time
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.experiments.issue_952.run_952 import (  # noqa: E402
    ARMS,
    BANK_ARMS,
    F16_SLOTS,
    L16_CONTENT_SLOTS,
    L16_TEMPLATE_SLOTS,
    MATCHED_ARMS,
    MATCHED_T2,
    POSITION_SLOTS,
    _json_np,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("issue952.stats")

BOOTSTRAP_SEED = 0  # plan §10
SIGNFLIP_SEED = 1  # plan §10
N_DRAWS_DEFAULT = 10_000
H1_MARGIN = 0.03
H2_MARGIN = 0.02
H3_MARGIN = 0.05
LENGTH_CUTS = (200, 100, 50)  # |Δlen| strata (#823 precedent)
PARITY_N_DRAWS = 200
PARITY_TOL = 1e-8


# ── column registry + scatter ───────────────────────────────────────────────────


def _scatter(
    ids: np.ndarray, ssr: np.ndarray, sst: np.ndarray, col_of: dict[int, int], n_test: int
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter one cell's per-context (ssr, sst) into the shared test-id space.

    NaN entries (invalid target for that context) are zero-filled in BOTH
    vectors, so a bootstrap draw's ratio-of-sums silently ignores them —
    the shared draw-weight matrix keeps every cell context-PAIRED.
    """
    v_ssr = np.zeros(n_test, dtype=np.float64)
    v_sst = np.zeros(n_test, dtype=np.float64)
    finite = np.isfinite(ssr) & np.isfinite(sst)
    for i, cid in enumerate(ids):
        if not finite[i]:
            continue
        j = col_of.get(int(cid))
        if j is None:
            raise AssertionError(f"context id {cid} not in the test split — coverage violation")
        v_ssr[j] = float(ssr[i])
        v_sst[j] = float(sst[i])
    return v_ssr, v_sst


class CellBank:
    """Registry of statistic columns over the shared LMSYS test-id space."""

    def __init__(self, test_ids: list[int]) -> None:
        self.test_ids = [int(i) for i in test_ids]
        self.col_of = {int(c): j for j, c in enumerate(self.test_ids)}
        self.n_test = len(self.test_ids)
        self.names: list[str] = []
        self._ssr: list[np.ndarray] = []
        self._sst: list[np.ndarray] = []

    def add(self, name: str, ids: np.ndarray, ssr: np.ndarray, sst: np.ndarray) -> None:
        v_ssr, v_sst = _scatter(ids, ssr, sst, self.col_of, self.n_test)
        self.names.append(name)
        self._ssr.append(v_ssr)
        self._sst.append(v_sst)

    def stacks(self) -> tuple[np.ndarray, np.ndarray]:
        return np.stack(self._ssr, axis=1), np.stack(self._sst, axis=1)  # (n_test, C)

    def observed(self) -> np.ndarray:
        ssr, sst = self.stacks()
        denom = sst.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = 1.0 - ssr.sum(axis=0) / denom
        r2[denom < 1e-12] = np.nan
        return r2

    def draws(self, w: np.ndarray) -> np.ndarray:
        """(n_draws, C) bootstrap pooled-R² draws — the TWO stacked-draw GEMMs."""
        ssr, sst = self.stacks()
        num = w @ ssr  # GEMM 1: (n_draws, n_test) @ (n_test, C)
        den = w @ sst  # GEMM 2
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = 1.0 - num / den
        r2[den < 1e-12] = np.nan
        return r2


def _ci(draws: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.nanpercentile(draws, 2.5, axis=axis),
        np.nanpercentile(draws, 97.5, axis=axis),
    )


def serial_oracle_parity(bank: CellBank, w: np.ndarray, r2_draws: np.ndarray) -> dict:
    """3-cell serial per-draw oracle vs the GEMM battery (rule item 6, hard gate)."""
    ssr, sst = bank.stacks()
    valid_cols = [c for c in range(len(bank.names)) if sst[:, c].sum() > 1e-12][:3]
    assert valid_cols, "no valid columns for the serial-oracle parity check"
    n_check = min(PARITY_N_DRAWS, w.shape[0])
    max_diff = 0.0
    for c in valid_cols:
        for d in range(n_check):  # the ORACLE is deliberately serial (parity only)
            num = float(np.dot(w[d], ssr[:, c]))
            den = float(np.dot(w[d], sst[:, c]))
            oracle = 1.0 - num / den if den > 1e-12 else np.nan
            got = float(r2_draws[d, c])
            if np.isfinite(oracle) or np.isfinite(got):
                max_diff = max(max_diff, abs(oracle - got))
    rec = {
        "cells": [bank.names[c] for c in valid_cols],
        "n_draws_checked": n_check,
        "max_abs_diff": max_diff,
        "tol": PARITY_TOL,
    }
    if max_diff > PARITY_TOL:
        raise RuntimeError(f"bootstrap GEMM vs serial oracle parity FAIL: {rec}")
    logger.info("[parity] bootstrap GEMM vs serial oracle: max|diff|=%.2e OK", max_diff)
    return rec


# ── input loading + coverage asserts ────────────────────────────────────────────


def load_spans(spans_dir: pathlib.Path) -> dict[str, dict[int, int]]:
    """Extended spans per arm: {arm: {cid: span}} (0 for skipped)."""
    out: dict[str, dict[int, int]] = {}
    for arm in ARMS:
        p = spans_dir / f"spans_{arm}.json"
        assert p.exists(), (
            f"spans file missing: {p} — the H2 set-equality assert needs the spans "
            "artifacts (download analysis_tensors/spans_*.json first)"
        )
        d = json.loads(p.read_text())
        out[arm] = {int(k): int(v.get("span", 0)) for k, v in d.items()}
    return out


def assert_h2_row_coverage(
    npz: dict[str, np.ndarray], spans: dict[str, dict[int, int]], test_ids: list[int]
) -> dict:
    """Plan §3 H2 row-coverage: registered common subset == npz M{t2}_ctx_ids.

    The driver's registered rule: extended span >= t2+16 in ALL arms (a uniform
    superset of 'intersected across the compared arms'), intersected with the
    test split. Every present cleg/zleg array must cover exactly that set.
    """
    rec: dict[str, Any] = {}
    for t2 in MATCHED_T2:
        key = f"M{t2}_ctx_ids"
        if key not in npz:
            rec[f"t{t2}"] = "npz_ids_absent"
            continue
        got = {int(c) for c in npz[key].tolist()}
        registered = {
            int(c) for c in test_ids if all(spans[a].get(int(c), 0) >= t2 + 16 for a in ARMS)
        }
        assert got == registered, (
            f"H2 t2={t2}: npz ctx-id set != registered common subset "
            f"(npz-only: {sorted(got - registered)[:5]}, "
            f"registered-only: {sorted(registered - got)[:5]})"
        )
        n = len(got)
        for k in npz:
            if k.startswith(f"M{t2}_L") and (k.endswith("_ssres") or k.endswith("_sstot")):
                assert len(npz[k]) == n, f"H2 coverage: {k} has {len(npz[k])} rows != {n}"
        rec[f"t{t2}"] = {"paired_n_test": n, "set_equality": "PASS"}
    logger.info("[coverage] H2 set-equality: %s", {k: v for k, v in rec.items()})
    return rec


def assert_h3_row_coverage(npz: dict[str, np.ndarray], verification: dict) -> dict:
    """Plan §3 H3 row-coverage: kept-pair member sets == npz bank id sets."""
    if "bank_div_ids" not in npz:
        return {"status": "bank_arrays_absent"}
    kept = set(verification["kept_pairs"])
    exp_div, exp_ctl = set(), set()
    for p in verification["pairs"]:
        if p["pair_id"] not in kept:
            continue
        if isinstance(p.get("divergent"), dict):
            exp_div.add(p["divergent"]["query_id"])
        if isinstance(p.get("control"), dict):
            exp_ctl.add(p["control"]["query_id"])
    got_div = set(npz["bank_div_ids"].tolist())
    got_ctl = set(npz["bank_ctl_ids"].tolist())
    assert got_div == exp_div, (
        f"H3 divergent id set != kept-pair members (npz-only: {sorted(got_div - exp_div)[:5]}, "
        f"kept-only: {sorted(exp_div - got_div)[:5]})"
    )
    assert got_ctl == exp_ctl, (
        f"H3 control id set != kept-pair members (npz-only: {sorted(got_ctl - exp_ctl)[:5]}, "
        f"kept-only: {sorted(exp_ctl - got_ctl)[:5]})"
    )
    rec = {"n_pairs": len(kept), "set_equality": "PASS"}
    logger.info("[coverage] H3 set-equality: %s", rec)
    return rec


# ── H1 / H2 (LMSYS families) ───────────────────────────────────────────────────


def register_lmsys_cells(npz: dict[str, np.ndarray], bank: CellBank) -> dict[str, list[str]]:
    """Register A (position + c_last→remainder), P (prefix), M (matched) columns."""
    fam: dict[str, list[str]] = {"A": [], "P": [], "M": []}
    if "A_test_ssres" in npz:
        groups = [g for g in npz["A_group_names"].tolist()]
        ids = npz["A_test_ctx_ids"]
        for gi, g in enumerate(groups):
            name = f"A|{g}"
            bank.add(name, ids, npz["A_test_ssres"][:, gi], npz["A_test_sstot"][:, gi])
            fam["A"].append(name)
    for k in sorted(npz):
        if k.startswith("P_") and k.endswith("_ssres"):
            stem = k[: -len("_ssres")]
            name = f"P|{stem[2:]}"  # e.g. P|own_t16_L17
            bank.add(name, npz[f"{stem}_ctx_ids"], npz[k], npz[f"{stem}_sstot"])
            fam["P"].append(name)
        if k.startswith("M") and "_cleg_" in k and k.endswith("_ssres"):
            stem = k[: -len("_ssres")]
            t2 = stem.split("_")[0][1:]
            name = f"M|{stem[1:]}"  # e.g. M|16_L17_cleg_own
            bank.add(name, npz[f"M{t2}_ctx_ids"], npz[k], npz[f"{stem}_sstot"])
            fam["M"].append(name)
        if k.startswith("M") and "_zleg_" in k and k.endswith("_ssres"):
            stem = k[: -len("_ssres")]
            t2 = stem.split("_")[0][1:]
            name = f"M|{stem[1:]}"
            bank.add(name, npz[f"M{t2}_ctx_ids"], npz[k], npz[f"{stem}_sstot"])
            fam["M"].append(name)
    logger.info(
        "[cells] registered A=%d P=%d M=%d columns (total %d)",
        len(fam["A"]),
        len(fam["P"]),
        len(fam["M"]),
        len(bank.names),
    )
    return fam


def h1_reads(bank: CellBank, obs: np.ndarray, draws: np.ndarray) -> dict:
    """H1: Δ(own - ext) over F16 minus over L16-content, per external arm (plan §3)."""
    idx = {n: i for i, n in enumerate(bank.names)}

    def _gap_matrix(slots: tuple[str, ...], ext: str, mat: np.ndarray) -> np.ndarray:
        cols_own = [idx.get(f"A|{s}|own") for s in slots]
        cols_ext = [idx.get(f"A|{s}|{ext}") for s in slots]
        if any(c is None for c in cols_own + cols_ext):
            return np.full(mat.shape[:-1] or (1,), np.nan)
        own = mat[..., cols_own]
        exta = mat[..., cols_ext]
        return np.nanmean(own - exta, axis=-1)

    out: dict[str, Any] = {"margin": H1_MARGIN}
    for ext in ("ext_plain", "ext_style"):
        f16_obs = _gap_matrix(F16_SLOTS, ext, obs[None, :])[0]
        l16_obs = _gap_matrix(L16_CONTENT_SLOTS, ext, obs[None, :])[0]
        stat_obs = float(f16_obs - l16_obs)
        stat_draws = _gap_matrix(F16_SLOTS, ext, draws) - _gap_matrix(L16_CONTENT_SLOTS, ext, draws)
        lo, hi = _ci(stat_draws)
        out[ext] = {
            "f16_gap": float(f16_obs),
            "l16_content_gap": float(l16_obs),
            "h1_contrast": stat_obs,
            "ci95": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "clears_margin": bool(stat_obs >= H1_MARGIN and lo > 0),
            "template_slots_excluded": list(L16_TEMPLATE_SLOTS),
        }
    return out


def h2_reads(bank: CellBank, obs: np.ndarray, draws: np.ndarray, meta: dict) -> dict:
    """H2 MATCHED contrasts: ΔG(0→t2) per layer/ext + registered 0-vs-16 decision."""
    idx = {n: i for i, n in enumerate(bank.names)}
    l_star = meta.get("l_star_pos")
    out: dict[str, Any] = {"margin": H2_MARGIN, "l_star": l_star, "contrasts": {}}
    layers = sorted(
        {n.split("_")[1][1:] for n in bank.names if n.startswith("M|")}, key=lambda s: int(s)
    )
    for t2 in MATCHED_T2:
        for layer in layers:
            key = f"t{t2}_L{layer}"
            cols = {}
            missing = False
            for leg in ("cleg", "zleg"):
                arms = ARMS if leg == "cleg" else MATCHED_ARMS
                for a in arms:
                    c = idx.get(f"M|{t2}_L{layer}_{leg}_{a}")
                    if c is None and a in MATCHED_ARMS:
                        missing = True
                    cols[f"{leg}_{a}"] = c
            if missing:
                continue
            rec: dict[str, Any] = {}
            for ext in ("ext_plain", "ext_style"):
                g0_o = obs[cols["cleg_own"]] - obs[cols[f"cleg_{ext}"]]
                gt_o = obs[cols["zleg_own"]] - obs[cols[f"zleg_{ext}"]]
                dg_draws = (draws[:, cols["cleg_own"]] - draws[:, cols[f"cleg_{ext}"]]) - (
                    draws[:, cols["zleg_own"]] - draws[:, cols[f"zleg_{ext}"]]
                )
                lo, hi = _ci(dg_draws)
                rec[ext] = {
                    "G_matched_0": float(g0_o),
                    "G_matched_t": float(gt_o),
                    "delta_G": float(g0_o - gt_o),
                    "ci95": [float(lo), float(hi)],
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                    "clears_margin": bool((g0_o - gt_o) >= H2_MARGIN and lo > 0),
                }
            # Secondary: ΔG_distinct > ΔG_plain under the same matched rule.
            dd = (
                (draws[:, cols["cleg_own"]] - draws[:, cols["cleg_ext_style"]])
                - (draws[:, cols["zleg_own"]] - draws[:, cols["zleg_ext_style"]])
            ) - (
                (draws[:, cols["cleg_own"]] - draws[:, cols["cleg_ext_plain"]])
                - (draws[:, cols["zleg_own"]] - draws[:, cols["zleg_ext_plain"]])
            )
            lo, hi = _ci(dd)
            rec["delta_G_style_minus_plain"] = {
                "observed": float(rec["ext_style"]["delta_G"] - rec["ext_plain"]["delta_G"]),
                "ci95": [float(lo), float(hi)],
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            }
            rec["registered_decision"] = bool(t2 == 16 and str(l_star) == str(layer))
            out["contrasts"][key] = rec
    return out


def h2_intersection_reads(npz: dict[str, np.ndarray], bank: CellBank, w: np.ndarray) -> dict:
    """Binding-alternatives rec (i): cross-arm prefix contrasts on the
    intersection-of-survivors with paired n (per t, layer, ext arm)."""
    out: dict[str, Any] = {}
    stems: dict[tuple[str, str], dict[str, str]] = {}
    for k in npz:
        if k.startswith("P_") and k.endswith("_ctx_ids"):
            stem = k[: -len("_ctx_ids")]
            _p, arm_t_layer = stem.split("_", 1)
            arm, t_tag, l_tag = arm_t_layer.rsplit("_", 2)
            stems.setdefault((t_tag, l_tag), {})[arm] = stem
    for (t_tag, l_tag), by_arm in sorted(stems.items()):
        if "own" not in by_arm:
            continue
        for ext in ("ext_plain", "ext_style", "mismatch"):
            if ext not in by_arm:
                continue
            ids_own = npz[f"{by_arm['own']}_ctx_ids"].tolist()
            ids_ext = npz[f"{by_arm[ext]}_ctx_ids"].tolist()
            common = sorted(set(ids_own) & set(ids_ext))
            if len(common) < 2:
                continue
            rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for arm in ("own", ext):
                pos = {int(c): i for i, c in enumerate(npz[f"{by_arm[arm]}_ctx_ids"].tolist())}
                sel = [pos[int(c)] for c in common]
                rows[arm] = (
                    npz[f"{by_arm[arm]}_ssres"][sel].astype(np.float64),
                    npz[f"{by_arm[arm]}_sstot"][sel].astype(np.float64),
                )
            # Shared-draw bootstrap restricted to the intersection.
            col = [bank.col_of[int(c)] for c in common]
            w_c = w[:, col]  # (n_draws, n_common) — the SAME draws, paired
            rec = {}
            r2d = {}
            for arm, (ssr, sst) in rows.items():
                fin = np.isfinite(ssr) & np.isfinite(sst)
                ssr0 = np.where(fin, ssr, 0.0)
                sst0 = np.where(fin, sst, 0.0)
                den_o = sst0.sum()
                rec[arm] = float(1.0 - ssr0.sum() / den_o) if den_o > 1e-12 else None
                den = w_c @ sst0
                with np.errstate(invalid="ignore", divide="ignore"):
                    r2d[arm] = np.where(den > 1e-12, 1.0 - (w_c @ ssr0) / den, np.nan)
            gap_draws = r2d["own"] - r2d[ext]
            lo, hi = _ci(gap_draws)
            out[f"{t_tag}_{l_tag}_own_vs_{ext}"] = {
                "paired_n": len(common),
                "r2_own": rec["own"],
                f"r2_{ext}": rec[ext],
                "gap": (rec["own"] - rec[ext])
                if rec["own"] is not None and rec[ext] is not None
                else None,
                "gap_ci95": [float(lo), float(hi)],
            }
    return out


def positive_control_reads(bank: CellBank, obs: np.ndarray, draws: np.ndarray) -> dict:
    """H2 positive control: mismatched-arm context-only R² ≈ 0 (±0.05) at every
    position, but ≥ 50% of the own arm's remainder R² at t=128 (plan §3)."""
    idx = {n: i for i, n in enumerate(bank.names)}
    pos_cols = [idx[f"A|{s}|mismatch"] for s in POSITION_SLOTS if f"A|{s}|mismatch" in idx]
    rec: dict[str, Any] = {}
    if pos_cols:
        vals = obs[pos_cols]
        rec["context_only_max_abs_r2"] = float(np.nanmax(np.abs(vals)))
        rec["context_only_within_pm_0p05"] = bool(np.nanmax(np.abs(vals)) <= 0.05)
    p128 = [n for n in bank.names if n.startswith("P|mismatch_t128_")]
    p128_own = [n for n in bank.names if n.startswith("P|own_t128_")]
    if p128 and p128_own:
        c_mm, c_own = idx[p128[0]], idx[p128_own[0]]
        stat_draws = draws[:, c_mm] - 0.5 * draws[:, c_own]
        lo, hi = _ci(stat_draws)
        rec["prefix_recovery_t128"] = {
            "r2_mismatch": float(obs[c_mm]),
            "r2_own": float(obs[c_own]),
            "ratio": float(obs[c_mm] / obs[c_own]) if obs[c_own] > 1e-9 else None,
            "stat_mm_minus_half_own": float(obs[c_mm] - 0.5 * obs[c_own]),
            "ci95": [float(lo), float(hi)],
            "passes": bool(obs[c_own] > 1e-9 and obs[c_mm] >= 0.5 * obs[c_own]),
        }
    return rec


# ── H3 (bank families) ──────────────────────────────────────────────────────────


def _bank_per_context_r2(
    npz: dict[str, np.ndarray], key: str, groups: list[str]
) -> dict[str, dict[str, float]]:
    """Per bank query: pooled-over-position-slots R² per arm (ratio of sums)."""
    ids = npz[f"{key}_ids"].tolist()
    ssr, sst = npz[f"{key}_ssres"].astype(np.float64), npz[f"{key}_sstot"].astype(np.float64)
    cols_by_arm = {
        arm: [
            gi
            for gi, g in enumerate(groups)
            if g.endswith(f"|{arm}") and g.split("|")[0] in POSITION_SLOTS
        ]
        for arm in BANK_ARMS
    }
    out: dict[str, dict[str, float]] = {}
    for ri, qid in enumerate(ids):
        rec = {}
        for arm, cols in cols_by_arm.items():
            s_r, s_t = ssr[ri, cols], sst[ri, cols]
            fin = np.isfinite(s_r) & np.isfinite(s_t)
            denom = s_t[fin].sum()
            rec[arm] = float(1.0 - s_r[fin].sum() / denom) if denom > 1e-12 else np.nan
        out[str(qid)] = rec
    return out


def h3_reads(  # noqa: C901 — the H3 read IS the plan's companion battery
    npz: dict[str, np.ndarray], verification: dict, n_draws: int, smoke: bool
) -> dict:
    """H3: paired (control - divergent) drops, ext vs own — headline + REQUIRED
    companions (plan §3): ss decomposition, median + 10%-trimmed, sign-flip null
    with Holm-Bonferroni, length-stratified sweep, band-vs-ceiling report."""
    if "bank_div_ids" not in npz:
        return {"status": "bank_arrays_absent"}
    groups = [g for g in npz["A_group_names"].tolist()]
    r2_div = _bank_per_context_r2(npz, "bank_div", groups)
    r2_ctl = _bank_per_context_r2(npz, "bank_ctl", groups)

    kept = set(verification["kept_pairs"])
    pair_rows = []
    for p in verification["pairs"]:
        if p["pair_id"] not in kept:
            continue
        d_m, c_m = p.get("divergent"), p.get("control")
        if not (isinstance(d_m, dict) and isinstance(c_m, dict)):
            continue
        qd, qc = d_m["query_id"], c_m["query_id"]
        if qd not in r2_div or qc not in r2_ctl:
            continue
        row: dict[str, Any] = {"pair_id": p["pair_id"], "category": p["category"]}
        ok = True
        for arm in BANK_ARMS:
            rd, rc = r2_div[qd][arm], r2_ctl[qc][arm]
            if not (np.isfinite(rd) and np.isfinite(rc)):
                ok = False
            row[f"r2_div_{arm}"] = rd
            row[f"r2_ctl_{arm}"] = rc
            row[f"drop_{arm}"] = rc - rd
        row["d"] = row["drop_ext_plain"] - row["drop_own"]
        ld = d_m.get("qwen_len_tokens")
        lc = c_m.get("qwen_len_tokens")
        row["abs_len_diff"] = abs(ld - lc) if ld is not None and lc is not None else None
        if ok:
            pair_rows.append(row)
    n_pairs = len(pair_rows)
    if n_pairs < 2:
        return {"status": f"too_few_pairs (n={n_pairs})", "n_pairs": n_pairs}
    d = np.asarray([r["d"] for r in pair_rows], dtype=np.float64)
    cats = [r["category"] for r in pair_rows]

    def _bank_boot(vals: np.ndarray, seed: int = BOOTSTRAP_SEED) -> dict:
        """Batched pair-level bootstrap: gathered resample + axis-1 reductions."""
        rng = np.random.default_rng(seed)
        m = len(vals)
        idx = rng.integers(0, m, size=(n_draws, m))
        res = vals[idx]  # (n_draws, m)
        mean_d = res.mean(axis=1)
        med_d = np.median(res, axis=1)
        k = max(1, round(0.1 * m))
        srt = np.sort(res, axis=1)
        trim_d = srt[:, k : m - k].mean(axis=1) if m - 2 * k >= 1 else mean_d
        srt0 = np.sort(vals)
        return {
            "n": m,
            "mean": float(vals.mean()),
            "mean_ci95": [float(np.percentile(mean_d, 2.5)), float(np.percentile(mean_d, 97.5))],
            "median": float(np.median(vals)),
            "median_ci95": [float(np.percentile(med_d, 2.5)), float(np.percentile(med_d, 97.5))],
            "trimmed10_mean": float(srt0[k : m - k].mean()) if m - 2 * k >= 1 else None,
            "trimmed10_ci95": [
                float(np.percentile(trim_d, 2.5)),
                float(np.percentile(trim_d, 97.5)),
            ],
        }

    headline = _bank_boot(d)
    headline["clears_margin"] = bool(headline["mean"] >= H3_MARGIN and headline["mean_ci95"][0] > 0)
    headline["robust_sign_agreement"] = bool(
        np.sign(headline["mean"]) == np.sign(headline["median"])
        and (
            headline["trimmed10_mean"] is None
            or np.sign(headline["mean"]) == np.sign(headline["trimmed10_mean"])
        )
    )

    # Sign-flip permutation null (seed 1): one (n_draws, n_pairs) ±1 matrix,
    # category-masked GEMMs; observed + every draw at the same frozen cells.
    rng_p = np.random.default_rng(SIGNFLIP_SEED)
    signs = rng_p.choice(np.asarray([-1.0, 1.0]), size=(n_draws, n_pairs))
    kept_cats = sorted({c for c in cats})
    null_recs: dict[str, Any] = {}
    pvals: dict[str, float] = {}
    for label in ["pooled", *kept_cats]:
        mask = (
            np.ones(n_pairs, bool) if label == "pooled" else np.asarray([c == label for c in cats])
        )
        m = int(mask.sum())
        if m == 0:
            continue
        obs_m = float(d[mask].mean())
        null = (signs[:, mask] @ d[mask]) / m  # ONE GEMM per label
        p_one = float((1 + int((null >= obs_m).sum())) / (1 + n_draws))
        band_hi = float(np.percentile(null, 97.5))
        # Band-vs-ceiling (plan §6): conditional ceiling = max attainable mean
        # ext drop (control R² -> 1) minus the own arm's realized mean drop
        # (interval = its bootstrap CI).
        r2_div_ext = np.asarray([r["r2_div_ext_plain"] for r in pair_rows])[mask]
        own_drop = np.asarray([r["drop_own"] for r in pair_rows])[mask]
        own_boot = _bank_boot(own_drop)
        ceiling = float(np.mean(1.0 - r2_div_ext) - own_boot["mean"])
        null_sd = float(null.std())
        null_recs[label] = {
            "n_pairs": m,
            "observed_mean_d": obs_m,
            "null_band_hi_97p5": band_hi,
            "p_one_sided": p_one,
            "ceiling_conditional": ceiling,
            "own_drop_mean_ci95": own_boot["mean_ci95"],
            "band_reaches_ceiling": bool(band_hi >= ceiling - null_sd),
        }
        pvals[label] = p_one
    # Holm-Bonferroni across kept categories (pooled reported unadjusted).
    cat_ps = sorted(((c, pvals[c]) for c in kept_cats if c in pvals), key=lambda x: x[1])
    k_cats = len(cat_ps)
    running = 0.0
    for rank, (c, p) in enumerate(cat_ps):
        adj = min(1.0, (k_cats - rank) * p)
        running = max(running, adj)
        null_recs[c]["p_holm"] = running

    per_cat = {
        c: _bank_boot(d[np.asarray([cc == c for cc in cats])])
        for c in kept_cats
        if sum(cc == c for cc in cats) >= 2
    }

    # ss_res / ss_tot decomposition of the differential drop (plan §3 H3 (i)).
    decomp_rows = []
    ids_div = {str(q): i for i, q in enumerate(npz["bank_div_ids"].tolist())}
    ids_ctl = {str(q): i for i, q in enumerate(npz["bank_ctl_ids"].tolist())}
    cols_by_arm = {
        arm: [
            gi
            for gi, g in enumerate(groups)
            if g.endswith(f"|{arm}") and g.split("|")[0] in POSITION_SLOTS
        ]
        for arm in BANK_ARMS
    }
    for p in verification["pairs"]:
        if p["pair_id"] not in kept or not isinstance(p.get("divergent"), dict):
            continue
        qd = p["divergent"]["query_id"]
        qc = p["control"]["query_id"] if isinstance(p.get("control"), dict) else None
        if qd not in ids_div or qc not in ids_ctl:
            continue
        row = {"pair_id": p["pair_id"], "category": p["category"]}
        for arm, cols in cols_by_arm.items():
            for tag, (mat_key, ri) in {
                "div": ("bank_div", ids_div[qd]),
                "ctl": ("bank_ctl", ids_ctl[qc]),
            }.items():
                s_r = npz[f"{mat_key}_ssres"][ri, cols].astype(np.float64)
                s_t = npz[f"{mat_key}_sstot"][ri, cols].astype(np.float64)
                fin = np.isfinite(s_r) & np.isfinite(s_t)
                row[f"ssr_{tag}_{arm}"] = float(s_r[fin].sum())
                row[f"sst_{tag}_{arm}"] = float(s_t[fin].sum())
            row[f"dlog_ssr_{arm}"] = float(
                np.log(max(row[f"ssr_div_{arm}"], 1e-12) / max(row[f"ssr_ctl_{arm}"], 1e-12))
            )
            row[f"dlog_sst_{arm}"] = float(
                np.log(max(row[f"sst_div_{arm}"], 1e-12) / max(row[f"sst_ctl_{arm}"], 1e-12))
            )
        decomp_rows.append(row)
    decomp = {}
    if decomp_rows:
        dl_ssr = np.asarray([r["dlog_ssr_ext_plain"] - r["dlog_ssr_own"] for r in decomp_rows])
        dl_sst = np.asarray([r["dlog_sst_ext_plain"] - r["dlog_sst_own"] for r in decomp_rows])
        decomp = {
            "median_differential_dlog_ssr": float(np.median(dl_ssr)),
            "median_differential_dlog_sst": float(np.median(dl_sst)),
            "residual_error_driven": bool(abs(np.median(dl_ssr)) > abs(np.median(dl_sst))),
            "note": (
                "differential (ext - own) log-ratios of divergent/control pooled ss "
                "components over position slots; residual-driven when the ss_res "
                "component dominates the ss_tot (denominator) component"
            ),
        }

    # Length-stratified sweep (|Δlen| cuts; #823 precedent).
    strata = {}
    lens = np.asarray(
        [r["abs_len_diff"] if r["abs_len_diff"] is not None else np.inf for r in pair_rows]
    )
    for cut in LENGTH_CUTS:
        mask = lens <= cut
        if int(mask.sum()) >= 2:
            strata[f"abs_len_diff_le_{cut}"] = _bank_boot(d[mask])
        else:
            strata[f"abs_len_diff_le_{cut}"] = {"n": int(mask.sum()), "status": "too_few_pairs"}

    return {
        "n_pairs": n_pairs,
        "margin": H3_MARGIN,
        "headline_mean_drop_diff": headline,
        "per_category": per_cat,
        "sign_flip_null": null_recs,
        "ss_decomposition": decomp,
        "length_stratified": strata,
        "pair_rows": pair_rows,
        "smoke": smoke,
    }


# ── driver ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Phase-2 statistics driver (coverage asserts -> batched draw batteries -> JSON)."""
    ap = argparse.ArgumentParser(description="Issue #952 Phase 2 stats (VM, CPU)")
    ap.add_argument("--eval-dir", type=str, required=True)
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--spans-dir", type=str, default=None, help="dir with spans_{arm}.json")
    ap.add_argument("--out", type=str, default=None, help="default: <eval-dir>/stats_summary.json")
    ap.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    eval_dir = pathlib.Path(args.eval_dir)
    npz_path = pathlib.Path(args.npz)
    spans_dir = pathlib.Path(args.spans_dir) if args.spans_dir else npz_path.parent
    out_path = pathlib.Path(args.out) if args.out else eval_dir / "stats_summary.json"
    npz = dict(np.load(npz_path, allow_pickle=False))

    split_name = "split_seed952_smoke.json" if args.smoke else "split_seed952.json"
    split = json.loads((eval_dir / split_name).read_text())
    meta = json.loads((eval_dir / "battery_meta.json").read_text())
    closure = json.loads((eval_dir / "prefix_closure_by_arm.json").read_text())
    ver_path = eval_dir / "divergence_bank_verification.json"
    verification = json.loads(ver_path.read_text()) if ver_path.exists() else None

    # ── row-coverage asserts BEFORE any statistic (plan §3) ──────────────────────
    spans = load_spans(spans_dir)
    cov_h2 = assert_h2_row_coverage(npz, spans, split["test"])
    cov_h3 = assert_h3_row_coverage(npz, verification) if verification else {"status": "no_bank"}

    # ── LMSYS bootstrap battery: the TWO stacked-draw GEMMs ─────────────────────
    bank = CellBank(split["test"])
    fam = register_lmsys_cells(npz, bank)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_test = bank.n_test
    w = rng.multinomial(n_test, np.full(n_test, 1.0 / n_test), size=args.n_draws).astype(np.float64)
    obs = bank.observed()
    draws = bank.draws(w)
    parity = serial_oracle_parity(bank, w, draws)
    lo, hi = _ci(draws)
    cells = {
        name: {
            "observed": float(obs[i]) if np.isfinite(obs[i]) else None,
            "ci95": [float(lo[i]), float(hi[i])] if np.isfinite(lo[i]) else None,
        }
        for i, name in enumerate(bank.names)
    }

    h1 = h1_reads(bank, obs, draws)
    h2 = h2_reads(bank, obs, draws, meta)
    h2_inter = h2_intersection_reads(npz, bank, w)
    pos_ctl = positive_control_reads(bank, obs, draws)
    h3 = (
        h3_reads(npz, verification, args.n_draws, args.smoke)
        if verification
        else {"status": "no_bank"}
    )

    git_sha = "unknown"
    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT))
            .decode()
            .strip()
        )
    except Exception as e:  # metadata only — never blocks the stats
        logger.warning("git sha lookup failed: %s", e)

    summary = {
        "issue": 952,
        "n_draws": args.n_draws,
        "seeds": {"bootstrap": BOOTSTRAP_SEED, "sign_flip": SIGNFLIP_SEED},
        "smoke": args.smoke,
        "row_coverage": {"h2": cov_h2, "h3": cov_h3},
        "bootstrap_gemm_parity": parity,
        "n_cells": len(bank.names),
        "families": {k: len(v) for k, v in fam.items()},
        "cells": cells,
        "h1": h1,
        "h2_matched": h2,
        "h2_intersection_reads": h2_inter,
        "h2_positive_control": pos_ctl,
        "h3": h3,
        "attrition": closure.get("attrition", {}),
        "matched_paired_n": {
            k: v.get("paired_n")
            for k, v in closure.get("matched_contrasts", {}).items()
            if isinstance(v, dict)
        },
        "inputs": {
            "npz": str(npz_path),
            "eval_dir": str(eval_dir),
            "npz_keys": len(npz),
        },
        "git_sha": git_sha,
        "numpy_version": np.__version__,
        "wall_seconds": time.time() - t0,
        "ts": time.time(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=_json_np))
    logger.info(
        "[stats] %d cells, %d draws in %.1fs -> %s",
        len(bank.names),
        args.n_draws,
        summary["wall_seconds"],
        out_path,
    )


if __name__ == "__main__":
    main()
