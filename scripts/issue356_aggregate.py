#!/usr/bin/env python3
"""Issue #356 aggregator.

Reads:

* eval-side baseline -> ``eval_results/issue186/baseline/result.json``
* train-side baseline -> ``eval_results/issue356/baseline_train/result.json``
* #186 ``persona_cot`` per-cell results ->
  ``eval_results/issue186/<source>_persona_cot_seed<S>/result.json``
* #356 ``consistent_persona_cot`` per-cell results ->
  ``eval_results/issue356/<source>_consistent_persona_cot_seed<S>/result.json``
* per-cell training logs -> ``eval_results/issue356/<cell_id>/train_log.json``
* Phase-0 audit JSON -> ``data/sft/issue356/_phase0_audit.json``
* vocab-diff audit -> ``data/sft/issue356/_vocab_diff.json``

Writes ``eval_results/issue356/aggregate.json`` with #280-compatible fields
plus the v5 pre-specified diagnostic sub-objects (``per_cell_training_loss``,
``difficulty_audit``, ``regeneration_fraction_stratification``,
``vocab_diff_passthrough``).

Statistics:

* Primary contrast: ``consistent_persona_cot - persona_cot`` at matched
  ``persona_cot`` eval, on (source_loss, bystander_macro) axes.
* Paired bootstrap n=1,000 on (q_id x seed) units.
* Holm-Bonferroni family of 8 (4 sources x 2 axes).
* TOST at ±0.03 (inherited) and ±0.01 (descriptive); gray-zone label
  ``partial_signal`` when Δ ∈ [+0.03, +0.04).

Plan v5 §Eval gives the schema; the diagnostic flag-trigger rules are
encoded directly into the output JSON for the analyzer.

CLI::

    uv run python scripts/issue356_aggregate.py --n-bootstrap 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger("issue356_aggregate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PERSONA_ORDER: list[str] = [
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "zelthari_scholar",
    "police_officer",
]
EVAL_SCAFFOLD_KEYS = ("no_cot", "generic_cot", "persona_cot", "empty_persona_cot_eval")
SOURCES = ("software_engineer", "librarian", "comedian", "police_officer")
SEEDS = (42, 137, 256)

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_N_Q = 1172

PRIMARY_CONTRAST_EVAL_ARM = "persona_cot"  # matched eval

# 4 sources x 2 axes = 8 source-level tests for Holm-Bonferroni (plan §Eval).
HOLM_ALPHA = 0.01

# TOST bands.
TOST_BAND_INHERITED = 0.03
TOST_BAND_TIGHT = 0.01
PARTIAL_SIGNAL_LOWER = 0.03
PARTIAL_SIGNAL_UPPER = 0.04

# Flag-trigger thresholds.
DIFFICULTY_DIFF_THRESHOLD = 0.03
REGEN_PEARSON_THRESHOLD = 0.70
VOCAB_JACCARD_MIN = 0.80
VOCAB_REGEN_DELTA_MAX = 0.10
TRAIN_LOSS_GAP_THRESHOLD = 0.10  # nats


# ── Helpers (joint with #280 patterns) ───────────────────────────────────────


def _correct_array(per_persona: dict, n_q: int) -> np.ndarray:
    """(n_q, 11, 4) int8 array of correctness for one cell."""
    arr = np.zeros((n_q, len(PERSONA_ORDER), len(EVAL_SCAFFOLD_KEYS)), dtype=np.int8)
    persona_idx = {p: i for i, p in enumerate(PERSONA_ORDER)}
    for p in PERSONA_ORDER:
        block = per_persona.get(p)
        if block is None:
            continue
        for q_idx, row in enumerate(block.get("raw", [])[:n_q]):
            ca = row.get("correct_answer")
            if ca is None:
                continue
            for sc_i, sc_key in enumerate(EVAL_SCAFFOLD_KEYS):
                pred = row.get(f"{sc_key}_pred")
                arr[q_idx, persona_idx[p], sc_i] = int(pred == ca)
    return arr


def _eval_arm_index(eval_arm: str) -> int:
    return EVAL_SCAFFOLD_KEYS.index(eval_arm)


def _holm_bonferroni(pvals: list[float], alpha: float = HOLM_ALPHA) -> list[bool]:
    n = len(pvals)
    order = np.argsort(pvals)
    reject = [False] * n
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if pvals[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject


def _holm_corrected_p(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, pvals[idx] * (n - rank))
        adj[idx] = min(1.0, running)
    return adj


def _bootstrap_paired(
    diff_per_pair: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float, float, np.ndarray]:
    """Joint paired bootstrap on a flat 1-D delta array.

    Returns ``(point, ci95_lo, ci95_hi, boots)``.
    """
    if diff_per_pair.size == 0:
        return float("nan"), float("nan"), float("nan"), np.zeros(0)
    point = float(diff_per_pair.mean())
    n = diff_per_pair.size
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(diff_per_pair[idx].mean())
    ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
    return point, float(ci_lo), float(ci_hi), boots


def _two_sided_p(boots: np.ndarray, point: float) -> float:
    if boots.size == 0:
        return float("nan")
    centered = boots - boots.mean()
    return float(np.mean(np.abs(centered) >= abs(point)))


def _ci(boots: np.ndarray, conf: float) -> tuple[float, float]:
    if boots.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(boots, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(lo), float(hi)


# ── Loaders ──────────────────────────────────────────────────────────────────


def _load_baseline_eval() -> dict:
    p = PROJECT_ROOT / "eval_results" / "issue186" / "baseline" / "result.json"
    if not p.exists():
        raise FileNotFoundError(f"Eval-side baseline missing: {p}")
    return json.loads(p.read_text())


def _load_baseline_train() -> dict | None:
    p = PROJECT_ROOT / "eval_results" / "issue356" / "baseline_train" / "result.json"
    if not p.exists():
        logger.warning("Train-side baseline missing at %s - difficulty_audit will be null", p)
        return None
    return json.loads(p.read_text())


def _load_per_cell_186(source: str, seed: int) -> dict | None:
    p = (
        PROJECT_ROOT
        / "eval_results"
        / "issue186"
        / f"{source}_persona_cot_seed{seed}"
        / "result.json"
    )
    if not p.exists():
        logger.warning("Missing #186 cell %s", p)
        return None
    return json.loads(p.read_text())


def _load_per_cell_356(source: str, seed: int) -> dict | None:
    p = (
        PROJECT_ROOT
        / "eval_results"
        / "issue356"
        / f"{source}_consistent_persona_cot_seed{seed}"
        / "result.json"
    )
    if not p.exists():
        logger.warning("Missing #356 cell %s", p)
        return None
    return json.loads(p.read_text())


def _load_train_log(source: str, seed: int) -> dict | None:
    p = (
        PROJECT_ROOT
        / "eval_results"
        / "issue356"
        / f"{source}_consistent_persona_cot_seed{seed}"
        / "train_log.json"
    )
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_train_log_186(source: str, seed: int) -> dict | None:
    """Optional carry-forward: if #186's train_log.json was preserved, read it
    for the matched-cell train-loss gap comparison."""
    p = (
        PROJECT_ROOT
        / "eval_results"
        / "issue186"
        / f"{source}_persona_cot_seed{seed}"
        / "train_log.json"
    )
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_audit_json() -> dict | None:
    p = PROJECT_ROOT / "data" / "sft" / "issue356" / "_phase0_audit.json"
    if not p.exists():
        logger.warning("Audit JSON missing at %s - difficulty_audit will be null", p)
        return None
    return json.loads(p.read_text())


def _load_vocab_diff() -> dict | None:
    p = PROJECT_ROOT / "data" / "sft" / "issue356" / "_vocab_diff.json"
    if not p.exists():
        logger.warning("Vocab-diff JSON missing at %s - vocab_diff_passthrough will be null", p)
        return None
    return json.loads(p.read_text())


# ── Primary contrasts (consistent_persona_cot - persona_cot) ─────────────────


def _delta_per_pair(
    *,
    base_correct: np.ndarray,
    cells_a: dict[str, np.ndarray],
    cells_b: dict[str, np.ndarray],
    source: str,
    axis: str,
) -> np.ndarray | None:
    """Per-(q, seed) Δ between arm A (cells_a) and arm B (cells_b) for one source."""
    persona_idx = {p: i for i, p in enumerate(PERSONA_ORDER)}
    eval_idx = _eval_arm_index(PRIMARY_CONTRAST_EVAL_ARM)

    if axis == "source_loss":
        target_idx = np.array([persona_idx[source]])
    elif axis == "bystander_macro":
        bystanders = [p for p in PERSONA_ORDER if p != source]
        target_idx = np.array([persona_idx[p] for p in bystanders])
    else:
        raise ValueError(f"Unknown axis: {axis!r}")

    def _loss_stack(cells: dict[str, np.ndarray]) -> np.ndarray | None:
        stacks = []
        for s in SEEDS:
            cell_id = f"{source}_seed{s}"
            tc = cells.get(cell_id)
            if tc is None:
                return None
            tc_slice = tc[:, target_idx, eval_idx]
            bc_slice = base_correct[:, target_idx, eval_idx]
            stacks.append(bc_slice.astype(np.float32) - tc_slice.astype(np.float32))
        return np.stack(stacks, axis=1)  # (n_q, n_seeds, n_target_personas)

    la = _loss_stack(cells_a)
    lb = _loss_stack(cells_b)
    if la is None or lb is None:
        return None

    la_per_qs = la.mean(axis=2)  # (n_q, n_seeds)
    lb_per_qs = lb.mean(axis=2)
    delta = la_per_qs - lb_per_qs
    return delta.reshape(-1)


def _compute_primary_contrasts(
    *,
    base_correct: np.ndarray,
    cells_356: dict[str, np.ndarray],
    cells_186: dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """8 source-level tests (4 sources x 2 axes) for ``consistent_persona_cot - persona_cot``."""
    rng = np.random.default_rng(seed)
    results: list[dict] = []
    per_source_macro: dict[str, dict] = {s: {} for s in SOURCES}

    for source in SOURCES:
        for axis in ("source_loss", "bystander_macro"):
            delta = _delta_per_pair(
                base_correct=base_correct,
                cells_a=cells_356,
                cells_b=cells_186,
                source=source,
                axis=axis,
            )
            if delta is None:
                results.append(
                    {
                        "source": source,
                        "axis": axis,
                        "label": (
                            f"consistent_persona_cot - persona_cot (source={source}, axis={axis})"
                        ),
                        "point_estimate": None,
                        "ci_95_lo": None,
                        "ci_95_hi": None,
                        "p_two_sided_uncorrected": None,
                        "n_pairs": 0,
                        "_skipped": "missing_cells",
                    }
                )
                continue
            point, ci_lo, ci_hi, boots = _bootstrap_paired(delta, n_bootstrap, rng)
            p_two = _two_sided_p(boots, point)
            # TOST at ±0.03 and ±0.01.
            ci90_lo, ci90_hi = _ci(boots, 0.90)
            tost_inherited = (ci90_lo >= -TOST_BAND_INHERITED) and (ci90_hi <= TOST_BAND_INHERITED)
            tost_tight = (ci90_lo >= -TOST_BAND_TIGHT) and (ci90_hi <= TOST_BAND_TIGHT)
            partial_signal = PARTIAL_SIGNAL_LOWER <= point < PARTIAL_SIGNAL_UPPER
            results.append(
                {
                    "source": source,
                    "axis": axis,
                    "label": f"consistent_persona_cot - persona_cot (source={source}, axis={axis})",
                    "point_estimate": point,
                    "ci_95_lo": ci_lo,
                    "ci_95_hi": ci_hi,
                    "p_two_sided_uncorrected": p_two,
                    "n_pairs": int(delta.size),
                    "tost_inherited_band_pm_0p03": tost_inherited,
                    "tost_tight_band_pm_0p01": tost_tight,
                    "ci_90_lo": ci90_lo,
                    "ci_90_hi": ci90_hi,
                    "label_partial_signal": partial_signal,
                }
            )
            per_source_macro[source][axis] = {
                "point_estimate": point,
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
            }

    # Holm-Bonferroni across the 8 source-level tests.
    pvals_clean = [
        r.get("p_two_sided_uncorrected", 1.0)
        if r.get("p_two_sided_uncorrected") is not None
        else 1.0
        for r in results
    ]
    rejects = _holm_bonferroni(pvals_clean, alpha=HOLM_ALPHA)
    adjusted = _holm_corrected_p(pvals_clean)
    for r, rj, p_adj in zip(results, rejects, adjusted, strict=True):
        r["holm_reject_at_alpha_0p01"] = bool(rj)
        r["holm_corrected_p"] = float(p_adj)

    # Macro means (descriptive).
    macro: dict = {}
    for axis in ("source_loss", "bystander_macro"):
        vals = [
            r["point_estimate"]
            for r in results
            if r["axis"] == axis and r["point_estimate"] is not None
        ]
        macro[axis] = {
            "macro_mean_delta": float(np.mean(vals)) if vals else None,
            "n_sources_present": len(vals),
        }

    return {
        "contrasts": results,
        "per_source": per_source_macro,
        "macro_means": macro,
        "holm_alpha_familywise": HOLM_ALPHA,
        "n_bootstrap": n_bootstrap,
        "primary_contrast": "consistent_persona_cot - persona_cot",
        "eval_arm": PRIMARY_CONTRAST_EVAL_ARM,
    }


# ── Diagnostic sub-objects (v5) ──────────────────────────────────────────────


def _diagnostic_per_cell_training_loss(
    seed_bias: int,
) -> dict:
    """Read each cell's ``train_log.json`` and extract loss summary stats.

    Returns ``per_cell`` keyed by ``f"{source}_consistent_persona_cot_seed{S}"``
    and a ``trained_harder_confound`` flag per source if the gap vs the matched
    #186 ``persona_cot`` cell exceeds the threshold.
    """
    _ = seed_bias  # unused; here for signature consistency with other diagnostics
    per_cell: dict[str, dict] = {}
    for source in SOURCES:
        for seed in SEEDS:
            cell_id_356 = f"{source}_consistent_persona_cot_seed{seed}"
            tlog_356 = _load_train_log(source, seed)
            entry: dict = {
                "final_train_loss": None,
                "best_train_loss": None,
                "epoch_at_best": None,
            }
            if tlog_356 is not None:
                losses = _extract_step_losses(tlog_356.get("log_history", []))
                if losses:
                    final = losses[-1]
                    best_idx = int(np.argmin([loss for _, _, loss in losses]))
                    epoch_at_best = losses[best_idx][1]
                    entry["final_train_loss"] = float(final[2])
                    entry["best_train_loss"] = float(losses[best_idx][2])
                    entry["epoch_at_best"] = float(epoch_at_best)
            # Compare to #186 matched cell (if its train_log was preserved).
            tlog_186 = _load_train_log_186(source, seed)
            if tlog_186 is not None and entry["final_train_loss"] is not None:
                ref_losses = _extract_step_losses(tlog_186.get("log_history", []))
                if ref_losses:
                    ref_final = ref_losses[-1][2]
                    entry["final_train_loss_186"] = float(ref_final)
                    entry["final_train_loss_gap_356_minus_186"] = float(
                        entry["final_train_loss"] - ref_final
                    )
            per_cell[cell_id_356] = entry

    # Per-source "trained-harder" flag: average final_train_loss gap < -0.10 nats.
    per_source_flags: dict[str, dict] = {}
    for source in SOURCES:
        gaps = []
        for seed in SEEDS:
            cell_id = f"{source}_consistent_persona_cot_seed{seed}"
            gap = per_cell.get(cell_id, {}).get("final_train_loss_gap_356_minus_186")
            if gap is not None:
                gaps.append(gap)
        if not gaps:
            per_source_flags[source] = {"mean_gap": None, "trained_harder_flag": None}
            continue
        mean_gap = float(np.mean(gaps))
        per_source_flags[source] = {
            "mean_gap": mean_gap,
            "trained_harder_flag": mean_gap < -TRAIN_LOSS_GAP_THRESHOLD,
        }

    return {
        "per_cell": per_cell,
        "per_source_flags": per_source_flags,
        "flag_threshold_nats": TRAIN_LOSS_GAP_THRESHOLD,
    }


def _extract_step_losses(log_history: list[dict]) -> list[tuple[int, float, float]]:
    """From a Trainer log_history, extract `[(step, epoch, loss)]` for training rows."""
    out: list[tuple[int, float, float]] = []
    for row in log_history:
        if "loss" in row and "step" in row:
            step = int(row["step"])
            ep = float(row.get("epoch", 0.0))
            out.append((step, ep, float(row["loss"])))
    return out


def _diagnostic_difficulty_audit(
    audit_json: dict | None,
    baseline_train: dict | None,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Join audit verdict per train q_id (source-keyed) to train-side baseline accuracy."""
    if audit_json is None or baseline_train is None:
        return {
            "baseline_join_source": "eval_results/issue356/baseline_train/result.json",
            "baseline_join_population": (
                "ARC-C train q_ids from data/sft/issue356/_phase0_audit.json"
            ),
            "per_source": None,
            "_skipped": "missing_inputs",
        }

    rng = np.random.default_rng(seed + 2)
    # Build baseline-accuracy lookup keyed by q_id (or row_index surrogate).
    raw = baseline_train.get("per_persona", {}).get("assistant", {}).get("raw", [])
    base_acc_by_id: dict[str, int] = {}
    base_acc_by_row: dict[int, int] = {}
    for i, r in enumerate(raw):
        ca = r.get("correct_answer")
        pred = r.get("no_cot_pred")
        correct = int(pred == ca)
        qid = r.get("q_id") or r.get("id") or f"row{i}"
        base_acc_by_id[str(qid)] = correct
        base_acc_by_row[i] = correct

    per_source: dict[str, dict] = {}
    for source in SOURCES:
        passed: list[int] = []
        failed: list[int] = []
        for prov in audit_json.get("rows", []):
            if prov["source"] != source:
                continue
            verdict = (prov.get("initial_verdict") or {}).get("verdict")
            if verdict is None:
                continue
            # Map to baseline accuracy: prefer q_id, fall back to row_index.
            qid_str = str(prov.get("q_id"))
            row_idx = int(prov.get("row_index", -1))
            if qid_str in base_acc_by_id:
                acc = base_acc_by_id[qid_str]
            elif row_idx in base_acc_by_row:
                acc = base_acc_by_row[row_idx]
            else:
                continue
            # Regenerated rows whose final_status starts with `regenerated_` count
            # toward `passed` (the question is the same; only the rationale changed).
            final_status = prov.get("final_status", "")
            if verdict == "consistent" or final_status.startswith("regenerated_"):
                passed.append(acc)
            else:
                failed.append(acc)

        if not passed or not failed:
            per_source[source] = {
                "passed_baseline_acc_mean": float(np.mean(passed)) if passed else None,
                "failed_baseline_acc_mean": float(np.mean(failed)) if failed else None,
                "diff_passed_minus_failed": None,
                "p_value": None,
                "n_passed": len(passed),
                "n_failed": len(failed),
                "flag_triggered": None,
            }
            continue

        passed_arr = np.asarray(passed)
        failed_arr = np.asarray(failed)
        diff_point = float(passed_arr.mean() - failed_arr.mean())

        # Two-sample bootstrap on the per-q_id baseline-accuracy values.
        boots = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            pidx = rng.integers(0, passed_arr.size, size=passed_arr.size)
            fidx = rng.integers(0, failed_arr.size, size=failed_arr.size)
            boots[b] = float(passed_arr[pidx].mean() - failed_arr[fidx].mean())
        p_two = float(np.mean(np.abs(boots - boots.mean()) >= abs(diff_point)))

        per_source[source] = {
            "passed_baseline_acc_mean": float(passed_arr.mean()),
            "failed_baseline_acc_mean": float(failed_arr.mean()),
            "diff_passed_minus_failed": diff_point,
            "p_value": p_two,
            "n_passed": int(passed_arr.size),
            "n_failed": int(failed_arr.size),
            "flag_triggered": abs(diff_point) > DIFFICULTY_DIFF_THRESHOLD,
        }

    return {
        "baseline_join_source": "eval_results/issue356/baseline_train/result.json",
        "baseline_join_population": ("ARC-C train q_ids from data/sft/issue356/_phase0_audit.json"),
        "per_source": per_source,
        "flag_threshold": DIFFICULTY_DIFF_THRESHOLD,
        "n_bootstrap": n_bootstrap,
    }


def _diagnostic_regeneration_fraction_stratification(
    audit_json: dict | None,
    primary_per_source: dict,
) -> dict:
    """Per-source regen fraction vs source-loss vs bystander-macro."""
    if audit_json is None:
        return {"per_source": None, "_skipped": "missing_audit"}

    per_source: dict[str, dict] = {}
    regen_fracs: list[float] = []
    bys_macros: list[float] = []

    for source in SOURCES:
        # Regeneration fraction from audit JSON.
        per_src_audit = audit_json.get("per_source", {}).get(source, {})
        regen_frac = per_src_audit.get("regeneration_fraction")
        if regen_frac is None:
            per_source[source] = {
                "regeneration_fraction": None,
                "bystander_macro": None,
                "source_loss": None,
            }
            continue
        bys = primary_per_source.get(source, {}).get("bystander_macro", {}).get("point_estimate")
        src_loss = primary_per_source.get(source, {}).get("source_loss", {}).get("point_estimate")
        per_source[source] = {
            "regeneration_fraction": float(regen_frac),
            "bystander_macro": bys,
            "source_loss": src_loss,
        }
        if bys is not None:
            regen_fracs.append(float(regen_frac))
            bys_macros.append(float(bys))

    # Pearson r across sources.
    pearson_r: float | None = None
    flag = None
    if len(regen_fracs) >= 3:  # at least 3 sources to compute correlation meaningfully
        pearson_r = float(np.corrcoef(regen_fracs, bys_macros)[0, 1])
        flag = abs(pearson_r) > REGEN_PEARSON_THRESHOLD

    return {
        "per_source": per_source,
        "pearson_r_regen_vs_bystander": pearson_r,
        "flag_threshold": REGEN_PEARSON_THRESHOLD,
        "flag_triggered": flag,
    }


def _diagnostic_vocab_diff_passthrough(vocab_diff: dict | None) -> dict:
    if vocab_diff is None:
        return {"per_source": None, "_skipped": "missing_vocab_diff"}
    per_source = vocab_diff.get("per_source", {})
    # Flag rollup so the analyzer sees it in one place.
    any_flag = any(s.get("flag_triggered") for s in per_source.values()) if per_source else False
    return {
        "per_source": per_source,
        "any_flag_triggered": any_flag,
        "thresholds": {
            "jaccard_min": VOCAB_JACCARD_MIN,
            "regen_delta_max": VOCAB_REGEN_DELTA_MAX,
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-q", type=int, default=DEFAULT_N_Q)
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "eval_results" / "issue356" / "aggregate.json"),
    )
    args = parser.parse_args()

    baseline_eval = _load_baseline_eval()
    base_correct = _correct_array(baseline_eval["per_persona"], args.n_q)

    # Build per-source-keyed cell arrays.
    cells_186: dict[str, np.ndarray] = {}
    cells_356: dict[str, np.ndarray] = {}
    for source in SOURCES:
        for seed in SEEDS:
            key = f"{source}_seed{seed}"
            r186 = _load_per_cell_186(source, seed)
            if r186 is not None:
                cells_186[key] = _correct_array(r186["per_persona"], args.n_q)
            r356 = _load_per_cell_356(source, seed)
            if r356 is not None:
                cells_356[key] = _correct_array(r356["per_persona"], args.n_q)

    logger.info(
        "Loaded #186 cells: %d/%d, #356 cells: %d/%d",
        len(cells_186),
        len(SOURCES) * len(SEEDS),
        len(cells_356),
        len(SOURCES) * len(SEEDS),
    )

    primary = _compute_primary_contrasts(
        base_correct=base_correct,
        cells_356=cells_356,
        cells_186=cells_186,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    audit_json = _load_audit_json()
    baseline_train = _load_baseline_train()
    vocab_diff = _load_vocab_diff()

    per_cell_training_loss = _diagnostic_per_cell_training_loss(seed_bias=args.seed)
    difficulty_audit = _diagnostic_difficulty_audit(
        audit_json,
        baseline_train,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    regen_fraction = _diagnostic_regeneration_fraction_stratification(
        audit_json, primary["per_source"]
    )
    vocab_passthrough = _diagnostic_vocab_diff_passthrough(vocab_diff)

    aggregate = {
        "primary_contrasts": primary,
        "per_source": primary["per_source"],
        "per_cell_training_loss": per_cell_training_loss,
        "difficulty_audit": difficulty_audit,
        "regeneration_fraction_stratification": regen_fraction,
        "vocab_diff_passthrough": vocab_passthrough,
        "metadata": {
            "n_cells_186": len(cells_186),
            "n_cells_356": len(cells_356),
            "n_q": args.n_q,
            "primary_contrast": "consistent_persona_cot - persona_cot",
            "primary_contrast_eval_arm": PRIMARY_CONTRAST_EVAL_ARM,
            "holm_family_size": 8,
            "holm_alpha_familywise": HOLM_ALPHA,
            "tost_band_inherited": TOST_BAND_INHERITED,
            "tost_band_tight": TOST_BAND_TIGHT,
            "n_bootstrap": args.n_bootstrap,
            "flag_thresholds": {
                "difficulty_diff": DIFFICULTY_DIFF_THRESHOLD,
                "regen_pearson": REGEN_PEARSON_THRESHOLD,
                "vocab_jaccard_min": VOCAB_JACCARD_MIN,
                "vocab_regen_delta_max": VOCAB_REGEN_DELTA_MAX,
                "train_loss_gap_nats": TRAIN_LOSS_GAP_THRESHOLD,
            },
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2, default=float))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
