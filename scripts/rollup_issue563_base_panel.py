#!/usr/bin/env python3
"""Issue #563 rollup — base-own panel paired deltas, classification, covariates (VM, CPU).

Reads ``eval_results/issue_563/base/{slot_stats,completions}_*.json`` (this
run's base-own panel) plus #558's committed artifacts (the comparison arm) and
writes ``eval_results/issue_563/rollup.json`` with (plan section 3.3):

  1. Paired per-question deltas vs the Plain-assistant cell, all three spaces
     + the logZ decomposition (d_logp = d_z_marker - d_logZ asserted per row).
  2. Question-level paired bootstrap: 10,000 resamples of the question
     indices, seed 563, percentile 95% CI per cell mean. ONE SHARED index
     draw across every cell AND the French-localization bonus read (plan
     section 13 item 5 — preserves pairing). Sign counts + Wilson CI.
  3. Registered classification per cell (REPRODUCES / FLAT / INDETERMINATE
     against 0.5 x R_c), R_c recomputed from the committed #558 rollup and
     asserted vs the plan-quoted values (+-0.001); panel verdict >= 3/4.
     Boundary-adjacent REPRODUCES labels flagged threshold-sensitive (plan
     section 13 item 4).
  4. Parent side-by-side: per-cell #558 base-side rises (log-prob from
     cell_summaries; EOS-margin recomputed from the 60 committed per-row
     slot_stats files) with 12-adapter cluster-bootstrap CIs.
  5. Parity audit: [0:50] parent-parity subset means + classification vs the
     full-250 read per cell.
  6. Covariates: per-cell completion length, truncation rate, French-language
     flag rate, char similarity to the assistant cell, key-mention rates,
     degenerate (<5 token) rate, emission rates; plus the paired
     length-vs-d_logp regression (plan section 13 item 7).
  7. Bonus read: Delta_french - mean(Delta_doctor, Delta_swe) with the shared
     paired-bootstrap CI.
  8. Robustness recomputes (registered): excluding truncated PAIRS; the
     French cell excluding French-flagged PAIRS; excluding degenerate PAIRS;
     [0:50] subset. Exclusions are question-PAIR exclusions (plan section 13
     item 2).

Reduced-panel rule (plan section 13 item 3): a missing persona cell is a
HARD FAIL unless ``--declare-reduced-panel <cell>`` names it explicitly — the
reduced denominator (x/3) is then a logged deviation declared BEFORE rollup.

Usage (VM, after the pod is terminated; CPU-only):
    uv run python scripts/rollup_issue563_base_panel.py
    uv run python scripts/rollup_issue563_base_panel.py \\
        --results-dir eval_results/issue_563/base/smoke --expect-n 20 --n-resamples 200
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import re
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="rollup_issue563_base_panel")

from _issue543_common import PROJECT_ROOT, repro_metadata  # noqa: E402
from eval_issue563_base_panel import (  # noqa: E402
    EVAL_RESULTS_DIR_563,
    ISSUE_563,
    N_PANEL_PROMPTS_563,
    OUT_DIR_563,
    R_C_EXPECTED,
    recompute_parent_rises,
)

log = logging.getLogger("rollup_issue563_base_panel")

BASELINE_CELL = "trigger50"
CONTRAST_CELLS = ("doctor", "software_engineer", "french_person", "police_officer")
N_SUBSET = 50  # the [0:50] parent-parity question slice

DEFAULT_N_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 563  # plan Reproducibility Card

# Registered classification constants (plan section 5; fixed pre-run).
HALF_EFFECT_FRACTION = 0.5

PARENT_ROLLUP_PATH = PROJECT_ROOT / "eval_results" / "issue_558" / "rollup.json"
PARENT_EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_558"
PARENT_ARMS = ("r50", "r25", "r10", "r05")
PARENT_SEEDS = (42, 137, 256)

# Per-row identity tolerance: logp == z_marker - logZ up to the float32
# extraction precision of compute_marker_slot_stats.
IDENTITY_TOL_NATS = 5e-3

DEGENERATE_MAX_TOKENS = 5  # kill criterion 5: completions under 5 tokens

# French-language covariate flag (deterministic, coarse; reimplements the
# parent's "French-typical words/accents" check — #558 recorded 2/600 flagged).
_FRENCH_ACCENTS = "àâçéèêëîïôùûüœÀÂÇÉÈÊËÎÏÔÙÛÜŒ"
_FRENCH_WORDS = re.compile(
    r"\b(le|la|les|des|une|est|et|vous|je|pas|que|pour|avec|bonjour|c'est|d'une|qu'il)\b",
    re.IGNORECASE,
)


def looks_french(text: str) -> bool:
    """Coarse French-language flag: >=3 accented chars OR >=2 distinct French
    function words. Covariate only, never a DV (plan section 3.2 'Why code,
    not a model call?'; flip condition documented there)."""
    n_accents = sum(text.count(ch) for ch in _FRENCH_ACCENTS)
    if n_accents >= 3:
        return True
    return len({m.lower() for m in _FRENCH_WORDS.findall(text)}) >= 2


_KEY_LITERAL = "7f3a9e2c"
_KEY_WORD = re.compile(r"\bkey\b", re.IGNORECASE)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion (parent rollup)."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


# ── This run's per-cell data ─────────────────────────────────────────────────


def load_cell(results_dir: Path, cell: str, *, expect_n: int) -> tuple[list[dict], list[dict]]:
    """(slot rows, completion records) for one cell; counts asserted equal."""
    slot_path = results_dir / f"slot_stats_{cell}.json"
    comp_path = results_dir / f"completions_{cell}.json"
    for p in (slot_path, comp_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing cell file: {p}")
    slot = json.loads(slot_path.read_text())
    rows = slot["base_own"]
    recs = json.loads(comp_path.read_text())
    if not (slot["n"] == len(rows) == len(recs) == expect_n):
        raise RuntimeError(
            f"Cell {cell}: n={slot['n']}, rows={len(rows)}, records={len(recs)}, "
            f"expected {expect_n} everywhere"
        )
    for i, row in enumerate(rows):
        if not all(math.isfinite(v) for v in row.values()):
            raise RuntimeError(f"Non-finite slot row {cell}[{i}]: {row}")
        ident_gap = abs(row["logp"] - (row["z_marker"] - row["logZ"]))
        if ident_gap > IDENTITY_TOL_NATS:
            raise RuntimeError(
                f"logp != z_marker - logZ in {cell}[{i}] (gap {ident_gap:.2e} nats) — "
                "slot-stats storage contract broken"
            )
    return rows, recs


def assert_question_alignment(by_cell_recs: dict[str, list[dict]]) -> None:
    """HARD assert: every cell's user turns match the baseline cell's, in order.

    Pairing is by index; a re-ordered or re-sliced cell silently corrupts every
    paired delta, so this is checked against the persisted records, not assumed.
    """
    base_users = [r["user"] for r in by_cell_recs[BASELINE_CELL]]
    for cell, recs in by_cell_recs.items():
        users = [r["user"] for r in recs]
        if users != base_users:
            raise RuntimeError(f"Cell {cell}: user-turn list differs from {BASELINE_CELL}")


# ── Paired deltas + shared-draw bootstrap ────────────────────────────────────


def paired_delta_arrays(base_rows: list[dict], cell_rows: list[dict]) -> dict[str, np.ndarray]:
    """Per-question deltas (persona - assistant), all spaces + decomposition."""
    n = len(base_rows)
    assert len(cell_rows) == n, (len(cell_rows), n)
    d_logp = np.array([c["logp"] - b["logp"] for c, b in zip(cell_rows, base_rows, strict=True)])
    d_zm = np.array(
        [c["z_marker"] - b["z_marker"] for c, b in zip(cell_rows, base_rows, strict=True)]
    )
    d_eosm = np.array(
        [
            (c["z_marker"] - c["z_eos"]) - (b["z_marker"] - b["z_eos"])
            for c, b in zip(cell_rows, base_rows, strict=True)
        ]
    )
    d_logz = np.array([c["logZ"] - b["logZ"] for c, b in zip(cell_rows, base_rows, strict=True)])
    # Decomposition identity (exact up to per-row extraction precision).
    gap = np.max(np.abs(d_logp - (d_zm - d_logz)))
    if gap > 2 * IDENTITY_TOL_NATS:
        raise RuntimeError(f"d_logp != d_z_marker - d_logZ (max gap {gap:.2e} nats)")
    return {"d_logp": d_logp, "d_zm": d_zm, "d_eosm": d_eosm, "d_logz": d_logz}


def shared_bootstrap_idx(n: int, *, n_resamples: int, seed: int) -> np.ndarray:
    """ONE shared (n_resamples, n) question-index draw (plan section 13 item 5)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def boot_ci_mean(values: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """95% percentile CI on the mean under the SHARED index draw."""
    assert values.ndim == 1 and idx.shape[1] == len(values), (values.shape, idx.shape)
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def delta_stats(values: np.ndarray, idx: np.ndarray) -> dict:
    """Mean, CI (shared draw), sign counts + Wilson CI for one delta array."""
    lo, hi = boot_ci_mean(values, idx)
    n_pos = int(np.sum(values > 0))
    w_lo, w_hi = wilson_ci(n_pos, len(values))
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "ci95": [lo, hi],
        "n_pos": n_pos,
        "sign_rate_pos": n_pos / len(values),
        "sign_rate_wilson_95ci": [w_lo, w_hi],
    }


# ── Registered classification (plan section 5) ───────────────────────────────


def classify_cell(stats: dict, r_c: float) -> dict:
    """REPRODUCES / FLAT / INDETERMINATE against 0.5 x R_c (registered rule).

    REPRODUCES: 95% CI excludes 0 from below AND mean >= 0.5 x R_c.
    FLAT: 95% CI upper bound < 0.5 x R_c.
    INDETERMINATE: otherwise. Boundary-adjacent REPRODUCES (CI straddling
    0.5 x R_c) is flagged threshold-sensitive (plan section 13 item 4).
    """
    lo, hi = stats["ci95"]
    threshold = HALF_EFFECT_FRACTION * r_c
    if lo > 0 and stats["mean"] >= threshold:
        label = "REPRODUCES"
    elif hi < threshold:
        label = "FLAT"
    else:
        label = "INDETERMINATE"
    return {
        "label": label,
        "r_c_parent": r_c,
        "threshold_half_r_c": threshold,
        "mean": stats["mean"],
        "ci95": stats["ci95"],
        "threshold_sensitive": bool(lo <= threshold <= hi),
    }


def panel_verdict(labels: dict[str, str], denominator_cells: tuple[str, ...]) -> dict:
    """intrinsic-context if >=3/4 REPRODUCES; completion-content if >=3/4 FLAT;
    mixed otherwise. A declared reduced panel scales the >=3 threshold by the
    declared denominator (logged deviation, plan section 13 item 3)."""
    n_cells = len(denominator_cells)
    need = 3 if n_cells == 4 else max(2, n_cells - 1)  # x/3 reduced-panel rule
    n_rep = sum(labels[c] == "REPRODUCES" for c in denominator_cells)
    n_flat = sum(labels[c] == "FLAT" for c in denominator_cells)
    if n_rep >= need:
        verdict = "intrinsic-context"
    elif n_flat >= need:
        verdict = "completion-content"
    else:
        verdict = "mixed"
    return {
        "verdict": verdict,
        "denominator_cells": list(denominator_cells),
        "needed": need,
        "n_reproduces": n_rep,
        "n_flat": n_flat,
        "n_indeterminate": n_cells - n_rep - n_flat,
        "note": "per-cell reads stand on their own; MIXED is never narrated as "
        "partial reproduction — name the INDETERMINATE cells + their CIs "
        "(plan section 13 item 1)",
    }


# ── Parent comparison arm (#558 committed artifacts) ─────────────────────────


def parent_adapter_slugs() -> list[str]:
    return [f"{arm}_seed{seed}" for arm in PARENT_ARMS for seed in PARENT_SEEDS]


def cluster_bootstrap_ci(
    values: list[float], *, n_resamples: int, seed: int
) -> tuple[float, float]:
    """95% percentile CI on the mean, resampling adapters with replacement
    (parent rollup machinery, verbatim shape)."""
    arr = np.asarray(values, dtype=float)
    assert arr.ndim == 1 and len(arr) > 0, arr.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def parent_base_side(*, n_resamples: int, seed: int) -> dict:
    """#558 base-side rises per cell: log-prob from cell_summaries; EOS margin
    + lengths recomputed from the committed per-row files."""
    cs = json.loads(PARENT_ROLLUP_PATH.read_text())["cell_summaries"]
    slugs = parent_adapter_slugs()
    out: dict[str, dict] = {}
    for cell in CONTRAST_CELLS:
        logp_rises = [
            cs[s][cell]["logp_base_mean"] - cs[s][BASELINE_CELL]["logp_base_mean"] for s in slugs
        ]
        eosm_rises: list[float] = []
        for slug in slugs:
            arm, sd = slug.split("_seed")
            d = PARENT_EVAL_DIR / arm / f"seed{sd}" / "phase2"
            cell_rows = json.loads((d / f"slot_stats_{cell}.json").read_text())["base"]
            base_rows = json.loads((d / f"slot_stats_{BASELINE_CELL}.json").read_text())["base"]
            eosm_rises.append(
                _mean([r["z_marker"] - r["z_eos"] for r in cell_rows])
                - _mean([r["z_marker"] - r["z_eos"] for r in base_rows])
            )
        lo_l, hi_l = cluster_bootstrap_ci(logp_rises, n_resamples=n_resamples, seed=seed)
        lo_e, hi_e = cluster_bootstrap_ci(eosm_rises, n_resamples=n_resamples, seed=seed)
        out[cell] = {
            "logp_rise_mean": _mean(logp_rises),
            "logp_rise_ci95": [lo_l, hi_l],
            "logp_rise_per_adapter": dict(zip(slugs, logp_rises, strict=True)),
            "eosm_rise_mean": _mean(eosm_rises),
            "eosm_rise_ci95": [lo_e, hi_e],
        }
    out["assistant_logp_base_mean"] = _mean([cs[s][BASELINE_CELL]["logp_base_mean"] for s in slugs])
    return out


def parent_lengths() -> dict[str, float]:
    """Parent per-cell mean generated tokens (12-adapter pooled), from the
    committed completions files (covariate side-by-side, plan section 4)."""
    out: dict[str, float] = {}
    for cell in (BASELINE_CELL, *CONTRAST_CELLS):
        toks: list[float] = []
        for slug in parent_adapter_slugs():
            arm, sd = slug.split("_seed")
            recs = json.loads(
                (
                    PARENT_EVAL_DIR / arm / f"seed{sd}" / "phase2" / f"completions_{cell}.json"
                ).read_text()
            )
            toks.extend(float(r["n_generated_tokens"]) for r in recs)
        out[cell] = _mean(toks)
    return out


# ── Covariates (this run) ────────────────────────────────────────────────────


def cell_covariates(recs: list[dict], base_recs: list[dict]) -> dict:
    """Per-cell content covariates (measured, never DVs; plan section 4)."""
    n = len(recs)
    toks = [float(r["n_generated_tokens"]) for r in recs]
    sims = [
        difflib.SequenceMatcher(None, r["completion_text"], b["completion_text"]).ratio()
        for r, b in zip(recs, base_recs, strict=True)
    ]
    return {
        "n": n,
        "mean_tokens": _mean(toks),
        "median_tokens": float(np.median(toks)),
        "truncation_rate": sum(r["truncated"] for r in recs) / n,
        "degenerate_rate": sum(r["n_generated_tokens"] < DEGENERATE_MAX_TOKENS for r in recs) / n,
        "french_flag_rate": sum(looks_french(r["completion_text"]) for r in recs) / n,
        "emission_rate": sum(r["contains_marker"] for r in recs) / n,
        "key_literal_mention_rate": sum(_KEY_LITERAL in r["completion_text"] for r in recs) / n,
        "key_word_mention_rate": sum(bool(_KEY_WORD.search(r["completion_text"])) for r in recs)
        / n,
        "mean_char_similarity_to_assistant": _mean(sims),
    }


def length_regression(d_logp: np.ndarray, recs: list[dict], base_recs: list[dict]) -> dict:
    """Paired regression of d_logp on the paired length difference (section 13 item 7)."""
    d_len = np.array(
        [
            float(r["n_generated_tokens"]) - float(b["n_generated_tokens"])
            for r, b in zip(recs, base_recs, strict=True)
        ]
    )
    if np.allclose(d_len.std(), 0.0):
        return {"slope_per_token": None, "pearson_r": None, "note": "zero length variance"}
    slope, intercept = np.polyfit(d_len, d_logp, 1)
    r = float(np.corrcoef(d_len, d_logp)[0, 1])
    return {
        "slope_per_token": float(slope),
        "intercept": float(intercept),
        "pearson_r": r,
        "d_len_mean": float(d_len.mean()),
    }


# ── Robustness recomputes (question-PAIR exclusions) ─────────────────────────


def recompute_excluding(
    deltas: np.ndarray, keep_mask: np.ndarray, *, n_resamples: int, seed: int, label: str
) -> dict:
    """Delta stats on the kept question pairs (fresh draw at the reduced n,
    same seed convention; exclusions are PAIR exclusions)."""
    kept = deltas[keep_mask]
    if len(kept) == 0:
        return {"label": label, "n_kept": 0, "note": "all pairs excluded"}
    idx = shared_bootstrap_idx(len(kept), n_resamples=n_resamples, seed=seed)
    return {"label": label, "n_kept": len(kept), **delta_stats(kept, idx)}


# ── Main rollup ──────────────────────────────────────────────────────────────


def run_rollup(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    declared_reduced = set(args.declare_reduced_panel or [])
    unknown = declared_reduced - set(CONTRAST_CELLS)
    if unknown:
        raise RuntimeError(f"--declare-reduced-panel names unknown cell(s): {sorted(unknown)}")
    cells_present = tuple(c for c in CONTRAST_CELLS if c not in declared_reduced)
    if declared_reduced:
        log.warning(
            "REDUCED PANEL declared BEFORE rollup (logged deviation, plan section 13 "
            "item 3): dropping %s; verdict denominator = %d cells.",
            sorted(declared_reduced),
            len(cells_present),
        )

    # Parent-rise recompute + assert (stale-parent guard).
    r_c = recompute_parent_rises(PARENT_ROLLUP_PATH)
    log.info(
        "Parent rises (asserted vs plan +-0.001): %s", {k: round(v, 4) for k, v in r_c.items()}
    )

    # This run's data.
    rows: dict[str, list[dict]] = {}
    recs: dict[str, list[dict]] = {}
    for cell in (BASELINE_CELL, *cells_present):
        rows[cell], recs[cell] = load_cell(results_dir, cell, expect_n=args.expect_n)
    assert_question_alignment(recs)

    n = args.expect_n
    idx = shared_bootstrap_idx(n, n_resamples=args.n_resamples, seed=args.bootstrap_seed)
    idx_subset = shared_bootstrap_idx(
        min(N_SUBSET, n), n_resamples=args.n_resamples, seed=args.bootstrap_seed
    )

    cells_out: dict[str, dict] = {}
    deltas_by_cell: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, str] = {}
    labels_subset: dict[str, str] = {}
    for cell in cells_present:
        d = paired_delta_arrays(rows[BASELINE_CELL], rows[cell])
        deltas_by_cell[cell] = d
        stats_logp = delta_stats(d["d_logp"], idx)
        cls = classify_cell(stats_logp, r_c[cell])
        labels[cell] = cls["label"]

        # [0:50] parent-parity subset (divergence 2): R_c is calibrated on this
        # slice, so its classification is reported alongside the full read.
        sub = d["d_logp"][: min(N_SUBSET, n)]
        stats_sub = delta_stats(sub, idx_subset)
        cls_sub = classify_cell(stats_sub, r_c[cell])
        labels_subset[cell] = cls_sub["label"]
        subset_outside_full_ci = not (
            stats_logp["ci95"][0] <= stats_sub["mean"] <= stats_logp["ci95"][1]
        )

        cov = cell_covariates(recs[cell], recs[BASELINE_CELL])
        keep_trunc = np.array(
            [
                not (r["truncated"] or b["truncated"])
                for r, b in zip(recs[cell], recs[BASELINE_CELL], strict=True)
            ]
        )
        keep_degen = np.array(
            [
                r["n_generated_tokens"] >= DEGENERATE_MAX_TOKENS
                and b["n_generated_tokens"] >= DEGENERATE_MAX_TOKENS
                for r, b in zip(recs[cell], recs[BASELINE_CELL], strict=True)
            ]
        )
        robustness = [
            recompute_excluding(
                d["d_logp"],
                keep_trunc,
                n_resamples=args.n_resamples,
                seed=args.bootstrap_seed,
                label="excluding_truncated_pairs",
            ),
            recompute_excluding(
                d["d_logp"],
                keep_degen,
                n_resamples=args.n_resamples,
                seed=args.bootstrap_seed,
                label="excluding_degenerate_pairs",
            ),
        ]
        if cell == "french_person":
            keep_nonfr = np.array(
                [
                    not (looks_french(r["completion_text"]) or looks_french(b["completion_text"]))
                    for r, b in zip(recs[cell], recs[BASELINE_CELL], strict=True)
                ]
            )
            robustness.append(
                recompute_excluding(
                    d["d_logp"],
                    keep_nonfr,
                    n_resamples=args.n_resamples,
                    seed=args.bootstrap_seed,
                    label="excluding_french_flagged_pairs",
                )
            )

        cells_out[cell] = {
            "d_logp": stats_logp,
            "d_eos_margin": delta_stats(d["d_eosm"], idx),
            "d_z_marker": delta_stats(d["d_zm"], idx),
            "d_logZ": delta_stats(d["d_logz"], idx),
            "classification": cls,
            "subset_0_50": {
                "d_logp": stats_sub,
                "classification": cls_sub,
                "subset_mean_outside_full_ci": subset_outside_full_ci,
            },
            "covariates": cov,
            "length_regression": length_regression(d["d_logp"], recs[cell], recs[BASELINE_CELL]),
            "robustness": robustness,
            "per_question_d_logp": [float(v) for v in d["d_logp"]],
        }

    verdict = panel_verdict(labels, cells_present)
    verdict_subset = panel_verdict(labels_subset, cells_present)

    # Bonus read (registered, secondary): French localization on the SHARED draw.
    bonus: dict | None = None
    if {"french_person", "doctor", "software_engineer"} <= set(cells_present):
        diff = deltas_by_cell["french_person"]["d_logp"] - 0.5 * (
            deltas_by_cell["doctor"]["d_logp"] + deltas_by_cell["software_engineer"]["d_logp"]
        )
        stats = delta_stats(diff, idx)
        parent_gap = r_c["french_person"] - 0.5 * (r_c["doctor"] + r_c["software_engineer"])
        bonus = {
            **stats,
            "parent_gap": parent_gap,
            "threshold": 0.5,
            "localizes_to_base": bool(stats["mean"] >= 0.5 and stats["ci95"][0] > 0),
            "note": "Delta_french - mean(Delta_doctor, Delta_swe); one shared "
            "question-index draw across the three cells (plan section 13 item 5)",
        }

    # Assistant-cell absolutes (b-hat side; sanity).
    assistant_abs = {
        "logp_mean": _mean([r["logp"] for r in rows[BASELINE_CELL]]),
        "eos_margin_mean": _mean([r["z_marker"] - r["z_eos"] for r in rows[BASELINE_CELL]]),
        "covariates": cell_covariates(recs[BASELINE_CELL], recs[BASELINE_CELL]),
    }

    parent = parent_base_side(n_resamples=args.n_resamples, seed=args.bootstrap_seed)
    parent["mean_tokens_per_cell"] = parent_lengths()

    rollup = {
        **repro_metadata(),
        "issue": ISSUE_563,
        "parent_issue": 558,
        "mode": "production" if args.expect_n == N_PANEL_PROMPTS_563 else "reduced_n",
        "results_dir": str(results_dir),
        "n_questions": n,
        "bootstrap": {
            "n_resamples": args.n_resamples,
            "seed": args.bootstrap_seed,
            "unit": "question (paired)",
            "shared_draw_across_cells": True,
        },
        "registered_rule": {
            "half_effect_fraction": HALF_EFFECT_FRACTION,
            "r_c_parent_recomputed": r_c,
            "r_c_plan_quoted": R_C_EXPECTED,
        },
        "panel": {
            "baseline_cell": BASELINE_CELL,
            "cells": cells_out,
            "labels": labels,
            "verdict": verdict,
            "subset_0_50_labels": labels_subset,
            "subset_0_50_verdict": verdict_subset,
            "declared_reduced_panel": sorted(declared_reduced),
        },
        "bonus_french_localization": bonus,
        "assistant_cell": assistant_abs,
        "parent_base_side": parent,
    }
    out_path = Path(args.out) if args.out else EVAL_RESULTS_DIR_563 / "rollup.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rollup, indent=2))
    log.info("Rollup -> %s (verdict=%s labels=%s)", out_path, verdict["verdict"], labels)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #563 rollup: base-own panel paired deltas + classification (CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-dir", type=str, default=str(OUT_DIR_563))
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    p.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    p.add_argument(
        "--expect-n",
        type=int,
        default=N_PANEL_PROMPTS_563,
        help="Rows per cell (250 production; 20 for the smoke artifacts).",
    )
    p.add_argument(
        "--declare-reduced-panel",
        action="append",
        default=None,
        metavar="CELL",
        help="Declare a dropped persona cell BEFORE rollup (logged deviation; "
        "plan section 13 item 3). Repeatable.",
    )
    return p.parse_args()


def main() -> int:
    return run_rollup(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
