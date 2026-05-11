#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #228 — aggregate per-state JS results into joint correlation tables.

Inputs
------
* ``--input-dir eval_results/issue_228/`` (per-state ``result.json`` files
  written by ``compute_js_convergence_228.py``).
* ``--leakage-dir eval_results/causal_proximity/strong_convergence/``
  (cached #109 ``marker_eval.json`` files; one per (source, checkpoint)).
* ``--js-baseline-142 eval_results/js_divergence/divergence_matrices.json``
  (the #142 baseline JS matrix; used by sanity check 6.1#4).

Outputs
-------
* ``all_results.json`` — one row per (source, checkpoint_step) state, with
  every numeric quantity needed by the plotting + clean-result scripts.
* ``correlations.json`` — H1 / H1b1 / H1b2 / H1b3 / C1 / C1b / C2 / C3 / C4 /
  C5 / H2 / H2-detrend / H3 + variants. Each ρ comes with p, n, and a
  cluster-bootstrap CI (1000 iter, 2-stage source -> within-source target
  resample).

Usage::

    uv run python scripts/aggregate_issue228.py \\
        --input-dir eval_results/issue_228 \\
        --leakage-dir eval_results/causal_proximity/strong_convergence \\
        --js-baseline-142 eval_results/js_divergence/divergence_matrices.json \\
        --output eval_results/issue_228/all_results.json \\
        --correlations eval_results/issue_228/correlations.json

C2 (prompt-distance partial) requires ``sentence-transformers``; if not
available, C2 is recorded as ``{"status": "skipped"}`` in correlations.json.
C4 (temp=1 cross-check) requires ``raw_completions.json`` next to
``marker_eval.json``; otherwise it is recorded as ``"skipped"`` per plan §3.

Self-loop policy (plan §6.1 #6): aggregator MUST exclude the 7 diagonal cells
from H1 / H3 / C1 / C1b / C2 / C3 / C5. n=70 off-diagonal.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Single source of truth for adapter sources + checkpoint steps: the worker
# script. Both files used to declare these independently (NIT round-1
# code-review). Now we import from compute_js_convergence_228 and re-derive
# the sorted list locally so the alphabetical contract still holds.
from compute_js_convergence_228 import (  # noqa: E402
    ADAPTER_MAP as _ADAPTER_MAP,
)
from compute_js_convergence_228 import (  # noqa: E402
    CHECKPOINT_STEPS as _CHECKPOINT_STEPS,
)

from explore_persona_space.personas import (  # noqa: E402
    ALL_EVAL_PERSONAS,
)

logger = logging.getLogger("aggregate_issue228")

ADAPTER_SOURCES = sorted(_ADAPTER_MAP.keys())
CHECKPOINT_STEPS = list(_CHECKPOINT_STEPS)
EPOCH_BY_STEP = {step: 2 * (i + 1) for i, step in enumerate(CHECKPOINT_STEPS)}
TARGET_PERSONA_ORDER = list(ALL_EVAL_PERSONAS.keys())
ASSISTANT_PERSONA = "assistant"

EPOCH_FOR_H1 = 20  # plan §3
COSINE_LAYER_H1 = "layer_15"
COSINE_LAYER_H2 = "layer_20"
COSINE_LAYER_PAPERWORK = "layer_25"
BOOTSTRAP_ITER = 1000
BOOTSTRAP_SEED = 20260504
SANITY_MRE_THRESHOLD = 0.05  # plan §6.1 #4
SANITY_RHO_THRESHOLD = 0.99


# ── Helpers ────────────────────────────────────────────────────────────────


def _load_state(input_dir: Path, source: str, step: int) -> dict | None:
    if source == "base" and step == 0:
        path = input_dir / "base" / "checkpoint-0" / "result.json"
    else:
        path = input_dir / source / f"checkpoint-{step}" / "result.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_leakage(leakage_dir: Path, source: str, step: int) -> dict | None:
    """Load #109 marker_eval.json. Return None if missing."""
    path = leakage_dir / source / f"checkpoint-{step}" / "marker_eval.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path}: marker_eval.json should be a dict, got {type(data)}")
    return data


def _check_temp1_cache(leakage_dir: Path) -> bool:
    """Return True iff at least one raw_completions.json exists under leakage_dir."""
    if not leakage_dir.exists():
        return False
    return any(leakage_dir.glob("**/raw_completions.json"))


def _matrix_value(
    matrix: list[list[float]] | None,
    persona_names: list[str],
    src: str,
    tgt: str,
) -> float | None:
    if matrix is None:
        return None
    try:
        i = persona_names.index(src)
        j = persona_names.index(tgt)
    except ValueError:
        return None
    return matrix[i][j]


def _leakage_rate(leakage: dict, persona: str) -> float | None:
    if persona not in leakage:
        return None
    entry = leakage[persona]
    if not isinstance(entry, dict):
        return None
    rate = entry.get("rate")
    if rate is None:
        return None
    return float(rate)


# ── Sanity check 6.1 #4: epoch-0 vs #142 baseline ──────────────────────────


def _sanity_check_142(base_state: dict, baseline_142_path: Path) -> dict:
    """Compare base epoch-0 js_matrix to the #142 baseline.

    Plan §6.1 #4: mean relative error <5% AND Spearman ρ >0.99 on overlapping
    personas. Reindex by persona name (do NOT assume positional alignment).
    Abort on FAIL.
    """
    if not baseline_142_path.exists():
        raise SystemExit(
            f"Sanity 6.1#4: #142 baseline not found at {baseline_142_path}. "
            "Pull via: uv run python scripts/pull_results.py --name js_divergence"
        )
    with open(baseline_142_path) as f:
        baseline = json.load(f)
    js_142 = baseline.get("js_matrix")
    names_142 = baseline.get("persona_names")
    if js_142 is None or names_142 is None:
        raise SystemExit(
            f"Sanity 6.1#4: {baseline_142_path} missing 'js_matrix' or 'persona_names'"
        )

    js_228 = base_state["js_matrix"]
    names_228 = base_state["persona_names"]

    overlap = [p for p in names_228 if p in names_142]
    if len(overlap) < 5:
        raise SystemExit(
            f"Sanity 6.1#4: only {len(overlap)} overlapping personas — too few for check"
        )

    pairs_228: list[float] = []
    pairs_142: list[float] = []
    for i, a in enumerate(overlap):
        for b in overlap[i + 1 :]:
            i228 = names_228.index(a)
            j228 = names_228.index(b)
            i142 = names_142.index(a)
            j142 = names_142.index(b)
            pairs_228.append(js_228[i228][j228])
            pairs_142.append(js_142[i142][j142])

    arr_228 = np.array(pairs_228)
    arr_142 = np.array(pairs_142)
    if arr_228.size == 0:
        raise SystemExit("Sanity 6.1#4: no overlapping pairs after filtering")

    abs_err = np.abs(arr_228 - arr_142)
    mre = float(np.mean(abs_err / (np.abs(arr_142) + 1e-3)))
    rho_obj = stats.spearmanr(arr_228, arr_142)
    rho = float(rho_obj.statistic) if hasattr(rho_obj, "statistic") else float(rho_obj[0])

    verdict = "PASS"
    if mre > SANITY_MRE_THRESHOLD:
        verdict = f"FAIL (MRE {mre:.4f} > {SANITY_MRE_THRESHOLD})"
    elif rho < SANITY_RHO_THRESHOLD:
        verdict = f"FAIL (ρ {rho:.4f} < {SANITY_RHO_THRESHOLD})"

    result = {
        "n_overlap_personas": len(overlap),
        "n_pairs": int(arr_228.size),
        "mean_relative_error": mre,
        "spearman_rho": rho,
        "threshold_mre": SANITY_MRE_THRESHOLD,
        "threshold_rho": SANITY_RHO_THRESHOLD,
        "verdict": verdict,
        "overlapping_personas": overlap,
    }
    if verdict.startswith("FAIL"):
        logger.error("Sanity 6.1#4 FAIL: %s", json.dumps(result, indent=2))
        raise SystemExit(f"Sanity 6.1#4 FAIL — aborting aggregation. Details: {result}")
    logger.info("Sanity 6.1#4 PASS: MRE=%.5f, ρ=%.5f over %d pairs", mre, rho, arr_228.size)
    return result


# ── Build long-format records ──────────────────────────────────────────────


def _load_all_states(input_dir: Path) -> list[dict]:
    """Load the 71 per-state result.json files. Aborts on any missing."""
    states: list[dict] = []
    base_state = _load_state(input_dir, "base", 0)
    if base_state is None:
        raise SystemExit(f"Missing epoch-0 base state at {input_dir / 'base' / 'checkpoint-0'}")
    states.append(base_state)
    for source in ADAPTER_SOURCES:
        for step in CHECKPOINT_STEPS:
            s = _load_state(input_dir, source, step)
            if s is None:
                raise SystemExit(
                    f"Missing per-state result.json for ({source}, ckpt-{step}) — abort"
                )
            states.append(s)
    return states


def _load_all_leakage(leakage_dir: Path) -> dict[tuple[str, int], dict]:
    leakage_states: dict[tuple[str, int], dict] = {}
    missing_leakage: list[tuple[str, int]] = []
    for source in ADAPTER_SOURCES:
        for step in CHECKPOINT_STEPS:
            leak = _load_leakage(leakage_dir, source, step)
            if leak is None:
                missing_leakage.append((source, step))
            else:
                leakage_states[(source, step)] = leak
    if missing_leakage:
        raise SystemExit(
            f"Missing #109 leakage for {len(missing_leakage)} (source, ckpt) cells: "
            f"{missing_leakage[:5]}... — surface to user (do NOT regenerate)."
        )
    if len(leakage_states) != len(ADAPTER_SOURCES) * len(CHECKPOINT_STEPS):
        raise SystemExit(
            f"Sanity 6.1#5: expected {len(ADAPTER_SOURCES) * len(CHECKPOINT_STEPS)} "
            f"leakage files, got {len(leakage_states)}"
        )
    return leakage_states


def _row_for_h1(state: dict, source: str, tgt: str, leak: dict) -> dict:
    """Build one (source, target) row at epoch 20.

    For sources NOT in ``ALL_EVAL_PERSONAS`` (today: nurse), the per-state
    JS / cosine matrices include an extra row+column for the source persona
    (path-A fix, BLOCKER #1 round 1 -> round 2). We pull the JS value for
    (source, tgt) directly from that extended matrix via name lookup.
    ``_matrix_value`` does the lookup by persona name, so this works without
    special-casing as long as ``state["persona_names"]`` is the extended
    list (verified by ``compute_js_convergence_228._compute_state_metrics``).
    """
    names = state["persona_names"]
    cos_h1 = state["cosine_matrices"][COSINE_LAYER_H1]
    cos_h2 = state["cosine_matrices"][COSINE_LAYER_H2]
    cos_pp = state["cosine_matrices"][COSINE_LAYER_PAPERWORK]
    return {
        "source": source,
        "target": tgt,
        "is_self_loop": False,
        "js": _matrix_value(state["js_matrix"], names, source, tgt),
        "js_masked": _matrix_value(state.get("js_matrix_masked"), names, source, tgt),
        "js_no_marker": _matrix_value(state.get("js_matrix_no_marker"), names, source, tgt),
        "cosine_l15": _matrix_value(cos_h1["matrix"], cos_h1["persona_names"], source, tgt),
        "cosine_l20": _matrix_value(cos_h2["matrix"], cos_h2["persona_names"], source, tgt),
        "cosine_l25": _matrix_value(cos_pp["matrix"], cos_pp["persona_names"], source, tgt),
        "source_token_entropy_mean": state.get("source_token_entropy_mean"),
        "leakage": _leakage_rate(leak, tgt),
    }


def _build_h1_long(
    states: list[dict],
    leakage_states: dict[tuple[str, int], dict],
) -> list[dict]:
    """Build the off-diagonal (source, target) rows at epoch 20.

    Cell-count contract (n=71, user decision on issue #228 — option 1):
      * 6 sources that ARE in ALL_EVAL_PERSONAS → 6 × 10 = 60 off-diagonal cells.
      * 1 source (nurse) that is NOT in ALL_EVAL_PERSONAS → 11 directed cells
        (every ALL_EVAL persona is a valid target; no self-loop because
        nurse does not appear in the target list).
    Total: 60 + 11 = 71 directed off-diagonal cells.

    Per BLOCKER #1 path-A fix, the per-state JS/cosine matrices for
    nurse-source states are 12×12 (with a nurse row+column), so every cell
    has a real (non-None) JS value. We assert this invariant to catch
    regressions; the cluster-bootstrap and Spearman tests downstream rely on
    it.
    """
    epoch20_states = {s["source"]: s for s in states if s.get("epoch") == EPOCH_FOR_H1}
    h1_long: list[dict] = []
    for source in ADAPTER_SOURCES:
        if source not in epoch20_states:
            raise SystemExit(f"No epoch-20 state for source={source}")
        state = epoch20_states[source]
        leak = leakage_states[(source, 2000)]
        for tgt in TARGET_PERSONA_ORDER:
            if tgt == source:
                continue  # exclude self-loops (plan §6.1 #6)
            row = _row_for_h1(state, source, tgt, leak)
            if row["js"] is None:
                raise SystemExit(
                    f"H1 contract violation: js=None for (source={source}, target={tgt}) "
                    f"at epoch 20. State persona_names={state.get('persona_names')!r}; "
                    f"is the source missing from the JS matrix? Check that the per-state "
                    f"worker appended the extra-source row for non-ALL_EVAL sources."
                )
            h1_long.append(row)
    return h1_long


def _h2_row_summary(
    state: dict,
    leak: dict,
    source: str,
    step: int,
) -> dict:
    """Build a (source, checkpoint_step) summary row for H2.

    JS_t / cos_t are summarized as the mean off-diagonal of source's row in
    the per-state matrix. With the BLOCKER #1 path-A fix, every source
    (including nurse) has its own row in the per-state matrix — for nurse,
    that row is the 12th and was added by ``compute_js_convergence_228``
    when the source is not in ALL_EVAL_PERSONAS. So the H2 fallback to a
    whole-matrix off-diagonal mean is no longer needed — and is removed,
    since it had different semantics ("diffuseness of the persona space")
    from the per-source aggregate ("how far the source has drifted from
    the eval personas") that H2 is supposed to track. If the source row
    is genuinely missing for some reason, we crash loudly rather than fall
    back to the wrong-semantics surrogate.
    """
    names = state["persona_names"]
    if source not in names:
        raise SystemExit(
            f"H2 row build: source={source!r} not in state persona_names="
            f"{names!r}. Expected the per-state worker to append the source "
            f"as an extra teacher-force row when source not in ALL_EVAL_PERSONAS. "
            f"Cannot fall back to whole-matrix off-diagonal mean because that "
            f"has different semantics from a per-source row mean."
        )
    i = names.index(source)
    row = state["js_matrix"][i]
    js_off_diag = [v for k, v in enumerate(row) if k != i]
    cos_row = state["cosine_matrices"][COSINE_LAYER_H2]["matrix"][i]
    cos_off_diag = [v for k, v in enumerate(cos_row) if k != i]
    mean_js = sum(js_off_diag) / len(js_off_diag) if js_off_diag else float("nan")
    mean_cos_l20 = sum(cos_off_diag) / len(cos_off_diag) if cos_off_diag else float("nan")

    target_rates: list[float] = []
    for tgt in TARGET_PERSONA_ORDER:
        if tgt == source:
            continue
        r = _leakage_rate(leak, tgt)
        if r is not None:
            target_rates.append(r)
    mean_leakage = sum(target_rates) / len(target_rates) if target_rates else float("nan")
    max_leakage = max(target_rates) if target_rates else float("nan")
    asst_leakage = _leakage_rate(leak, ASSISTANT_PERSONA)
    return {
        "source": source,
        "checkpoint_step": step,
        "epoch": EPOCH_BY_STEP[step],
        "mean_js_off_diag": mean_js,
        "mean_cosine_l20_off_diag": mean_cos_l20,
        "mean_leakage_off_diag": mean_leakage,
        "max_leakage_off_diag": max_leakage,
        "assistant_leakage": asst_leakage,
        "source_token_entropy_mean": state.get("source_token_entropy_mean"),
        "source_in_eval_personas": source in ALL_EVAL_PERSONAS,
    }


def _build_h2_long(
    input_dir: Path,
    leakage_states: dict[tuple[str, int], dict],
) -> list[dict]:
    h2_long: list[dict] = []
    for source in ADAPTER_SOURCES:
        for step in CHECKPOINT_STEPS:
            state = _load_state(input_dir, source, step)
            if state is None:
                raise SystemExit(f"H2: missing state ({source}, ckpt-{step})")
            leak = leakage_states[(source, step)]
            h2_long.append(_h2_row_summary(state, leak, source, step))
    return h2_long


def _build_records(
    input_dir: Path,
    leakage_dir: Path,
) -> dict:
    """Read all 71 state files + all 70 leakage files, build long-format tables.

    Returns a dict with keys:
      * states: list of full per-state dicts (epoch + JS matrices + cosine).
      * h1_long: list of dicts (one per off-diagonal directed pair at epoch 20),
        for H1 / C1 / C1b / C3 / C5 analysis.
      * h2_long: list of dicts (one per (source, ckpt-step) row across the 10
        steps).
      * leakage_states: {(source, step): marker_eval_dict}.
    """
    states = _load_all_states(input_dir)
    leakage_states = _load_all_leakage(leakage_dir)
    h1_long = _build_h1_long(states, leakage_states)
    h2_long = _build_h2_long(input_dir, leakage_states)
    return {
        "states": states,
        "h1_long": h1_long,
        "h2_long": h2_long,
        "leakage_states": leakage_states,
    }


# ── Cluster bootstrap helpers ──────────────────────────────────────────────


def _spearman_xy(x: Iterable[float], y: Iterable[float]) -> tuple[float, float, int]:
    """Spearman ρ on aligned arrays, dropping pairs with NaN/None."""
    xs: list[float] = []
    ys: list[float] = []
    for a, b in zip(x, y, strict=True):
        if a is None or b is None:
            continue
        af = float(a)
        bf = float(b)
        if math.isnan(af) or math.isnan(bf):
            continue
        xs.append(af)
        ys.append(bf)
    if len(xs) < 3:
        return (float("nan"), float("nan"), len(xs))
    rho_obj = stats.spearmanr(xs, ys)
    rho = float(rho_obj.statistic) if hasattr(rho_obj, "statistic") else float(rho_obj[0])
    pval = float(rho_obj.pvalue) if hasattr(rho_obj, "pvalue") else float(rho_obj[1])
    return (rho, pval, len(xs))


def cluster_bootstrap_spearman(
    rows: list[dict],
    x_field: str,
    y_field: str,
    *,
    cluster_field: str = "source",
    inner_field: str = "target",
    n_iter: int = BOOTSTRAP_ITER,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """2-stage cluster bootstrap CI for Spearman ρ on (x, y) over rows.

    Stage 1: resample sources WITH replacement.
    Stage 2: within each resampled source, resample inner units WITH replacement.

    Args:
        rows: long-format rows. Each must have ``cluster_field``, ``inner_field``,
            ``x_field``, ``y_field``.
        n_iter: number of bootstrap iterations.

    Returns dict with point estimate, p-value, n, ci_low, ci_high (95% percentile).
    """
    rho_point, p_point, n_point = _spearman_xy(
        [r[x_field] for r in rows], [r[y_field] for r in rows]
    )

    by_cluster: dict[str, list[dict]] = {}
    for r in rows:
        by_cluster.setdefault(r[cluster_field], []).append(r)
    clusters = list(by_cluster.keys())
    if len(clusters) < 2:
        return {
            "rho": rho_point,
            "p_value": p_point,
            "n": n_point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_iter": n_iter,
            "n_clusters": len(clusters),
            "note": "too few clusters for bootstrap",
        }

    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    for _ in range(n_iter):
        chosen_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        x_boot: list[float] = []
        y_boot: list[float] = []
        for c in chosen_clusters:
            inner_rows = by_cluster[c]
            idxs = rng.integers(0, len(inner_rows), size=len(inner_rows))
            for ii in idxs:
                r = inner_rows[int(ii)]
                xv = r[x_field]
                yv = r[y_field]
                if xv is None or yv is None:
                    continue
                xf = float(xv)
                yf = float(yv)
                if math.isnan(xf) or math.isnan(yf):
                    continue
                x_boot.append(xf)
                y_boot.append(yf)
        if len(x_boot) < 3:
            continue
        rho_boot, _, _ = _spearman_xy(x_boot, y_boot)
        if not math.isnan(rho_boot):
            rhos.append(rho_boot)

    if not rhos:
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        ci_low = float(np.percentile(rhos, 2.5))
        ci_high = float(np.percentile(rhos, 97.5))

    return {
        "rho": rho_point,
        "p_value": p_point,
        "n": n_point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_iter": n_iter,
        "n_clusters": len(clusters),
        "n_bootstrap_valid": len(rhos),
    }


def paired_cluster_bootstrap_delta_abs_rho(
    rows: list[dict],
    y_field: str,
    x_a_field: str,
    x_b_field: str,
    *,
    cluster_field: str = "source",
    n_iter: int = BOOTSTRAP_ITER,
    seed: int = BOOTSTRAP_SEED + 1,
) -> dict:
    """Paired CI for |ρ(x_a, y)| - |ρ(x_b, y)| via 2-stage cluster bootstrap.

    Resamples the SAME bootstrap units for both correlations to preserve the
    pairing.
    """
    point_a = _spearman_xy([r[x_a_field] for r in rows], [r[y_field] for r in rows])
    point_b = _spearman_xy([r[x_b_field] for r in rows], [r[y_field] for r in rows])
    delta_point = abs(point_a[0]) - abs(point_b[0])

    by_cluster: dict[str, list[dict]] = {}
    for r in rows:
        by_cluster.setdefault(r[cluster_field], []).append(r)
    clusters = list(by_cluster.keys())

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_iter):
        chosen_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        boot_rows: list[dict] = []
        for c in chosen_clusters:
            inner_rows = by_cluster[c]
            idxs = rng.integers(0, len(inner_rows), size=len(inner_rows))
            boot_rows.extend(inner_rows[int(ii)] for ii in idxs)
        rho_a, _, _ = _spearman_xy(
            [r[x_a_field] for r in boot_rows], [r[y_field] for r in boot_rows]
        )
        rho_b, _, _ = _spearman_xy(
            [r[x_b_field] for r in boot_rows], [r[y_field] for r in boot_rows]
        )
        if math.isnan(rho_a) or math.isnan(rho_b):
            continue
        deltas.append(abs(rho_a) - abs(rho_b))

    if deltas:
        ci_low = float(np.percentile(deltas, 2.5))
        ci_high = float(np.percentile(deltas, 97.5))
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    return {
        "delta_abs_rho": delta_point,
        "abs_rho_a": abs(point_a[0]),
        "abs_rho_b": abs(point_b[0]),
        "n": min(point_a[2], point_b[2]),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_iter": n_iter,
        "n_bootstrap_valid": len(deltas),
    }


# ── H1 family ──────────────────────────────────────────────────────────────


_ZSCORE_DEGENERATE_STD = 1e-12


def _within_target_zscore(rows: list[dict], field: str) -> dict[int, float]:
    """Per-target z-score of the field across the sources contributing
    to that target.

    Returns: {row_idx: z_score} so the caller can reattach to rows.

    Robustness (ISSUE round-1 round-2 fix): we treat std below
    ``_ZSCORE_DEGENERATE_STD`` (effectively zero given float noise) as
    degenerate, mapping all values to z=0 instead of dividing by it. This
    matters if this helper is ever called on a bootstrap-resampled cluster
    where many duplicate rows make the inner variance collapse — the
    division would produce inf/nan and crash downstream.
    """
    by_target: dict[str, list[tuple[int, float]]] = {}
    for idx, r in enumerate(rows):
        v = r.get(field)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by_target.setdefault(r["target"], []).append((idx, float(v)))
    out: dict[int, float] = {}
    for _tgt, lst in by_target.items():
        vals = np.array([x[1] for x in lst])
        mean = vals.mean()
        std = vals.std(ddof=0)
        if std < _ZSCORE_DEGENERATE_STD or math.isnan(std):
            for idx, _ in lst:
                out[idx] = 0.0
        else:
            for idx, v in lst:
                out[idx] = float((v - mean) / std)
    return out


def _h1_analysis(h1_long: list[dict]) -> dict:
    rows = [r for r in h1_long if not r["is_self_loop"]]
    out: dict = {}

    # Pooled H1 primary
    out["h1_primary_js"] = cluster_bootstrap_spearman(rows, "js", "leakage")
    out["h1_primary_cosine_l15"] = cluster_bootstrap_spearman(rows, "cosine_l15", "leakage")
    out["h1_delta_js_minus_cosine_l15"] = paired_cluster_bootstrap_delta_abs_rho(
        rows, "leakage", "js", "cosine_l15"
    )

    # H1b1: within-target z-scored pooled ρ
    z_js = _within_target_zscore(rows, "js")
    z_leak = _within_target_zscore(rows, "leakage")
    pairs_z: list[tuple[float, float]] = []
    for idx in range(len(rows)):
        if idx in z_js and idx in z_leak:
            pairs_z.append((z_js[idx], z_leak[idx]))
    if pairs_z:
        rho_zb, p_zb, n_zb = _spearman_xy([p[0] for p in pairs_z], [p[1] for p in pairs_z])
    else:
        rho_zb, p_zb, n_zb = (float("nan"), float("nan"), 0)
    out["h1b1_within_target_z"] = {"rho": rho_zb, "p_value": p_zb, "n": n_zb}

    # H1b2: source-fixed mean ρ (7 source-ρs, each n=10)
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    source_rhos: list[dict] = []
    for source, src_rows in sorted(by_source.items()):
        rho, p, n = _spearman_xy([r["js"] for r in src_rows], [r["leakage"] for r in src_rows])
        source_rhos.append({"source": source, "rho": rho, "p_value": p, "n": n})
    valid_source_rhos = [r["rho"] for r in source_rhos if not math.isnan(r["rho"])]
    out["h1b2_source_fixed"] = {
        "per_source": source_rhos,
        "mean_rho": float(np.mean(valid_source_rhos)) if valid_source_rhos else float("nan"),
        "mean_abs_rho": (
            float(np.mean(np.abs(valid_source_rhos))) if valid_source_rhos else float("nan")
        ),
        "n_sources_valid": len(valid_source_rhos),
    }

    # H1b3: target-fixed mean ρ (11 target-ρs, each n=7)
    by_target: dict[str, list[dict]] = {}
    for r in rows:
        by_target.setdefault(r["target"], []).append(r)
    target_rhos: list[dict] = []
    for tgt, tgt_rows in sorted(by_target.items()):
        rho, p, n = _spearman_xy([r["js"] for r in tgt_rows], [r["leakage"] for r in tgt_rows])
        target_rhos.append({"target": tgt, "rho": rho, "p_value": p, "n": n})
    valid_target_rhos = [r["rho"] for r in target_rhos if not math.isnan(r["rho"])]
    out["h1b3_target_fixed"] = {
        "per_target": target_rhos,
        "mean_rho": float(np.mean(valid_target_rhos)) if valid_target_rhos else float("nan"),
        "mean_abs_rho": (
            float(np.mean(np.abs(valid_target_rhos))) if valid_target_rhos else float("nan")
        ),
        "n_targets_valid": len(valid_target_rhos),
    }

    # H1 source->assistant slice (n=7 descriptive)
    asst_rows = [r for r in rows if r["target"] == ASSISTANT_PERSONA]
    rho_a, p_a, n_a = _spearman_xy([r["js"] for r in asst_rows], [r["leakage"] for r in asst_rows])
    out["h1_source_to_assistant"] = {
        "rho": rho_a,
        "p_value": p_a,
        "n": n_a,
        "note": "n=7 descriptive (source -> assistant slice)",
    }

    return out


# ── Confound-control variants ──────────────────────────────────────────────


def _c1_marker_mask(h1_long: list[dict]) -> dict:
    rows = [r for r in h1_long if not r["is_self_loop"] and r.get("js_masked") is not None]
    primary = cluster_bootstrap_spearman(rows, "js_masked", "leakage")
    delta = paired_cluster_bootstrap_delta_abs_rho(rows, "leakage", "js_masked", "js")
    # ``abs_rho_b`` is already |ρ| from paired_cluster_bootstrap_delta_abs_rho
    # (NIT round-1 -> round-2 fix: drop redundant abs()).
    rho_raw = delta["abs_rho_b"]
    rel_change = abs(delta["delta_abs_rho"]) / rho_raw if rho_raw > 1e-9 else float("nan")
    return {
        "rho_masked": primary,
        "delta_abs_rho_masked_minus_raw": delta,
        "relative_change": rel_change,
        "marker_mediated_threshold_rel_change": 0.4,
        "marker_mediated": rel_change > 0.4 if not math.isnan(rel_change) else False,
    }


def _c1b_marker_free(h1_long: list[dict]) -> dict:
    rows = [r for r in h1_long if not r["is_self_loop"] and r.get("js_no_marker") is not None]
    if not rows:
        return {"status": "no_marker_free_cells", "n": 0}
    primary = cluster_bootstrap_spearman(rows, "js_no_marker", "leakage")
    delta = paired_cluster_bootstrap_delta_abs_rho(rows, "leakage", "js_no_marker", "js")
    rho_collapse_threshold = 0.3
    return {
        "rho_no_marker": primary,
        "delta_abs_rho_no_marker_minus_raw": delta,
        "collapse_threshold": rho_collapse_threshold,
        "marker_context_mediated": (
            abs(delta["delta_abs_rho"]) > rho_collapse_threshold
            if not math.isnan(delta["delta_abs_rho"])
            else False
        ),
    }


def _c2_prompt_distance(h1_long: list[dict]) -> dict:
    """Partial ρ(JS, leakage | cosine_l15, prompt_cos)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return {
            "status": "skipped",
            "reason": "sentence-transformers not available — install if C2 needed",
        }

    persona_prompts = {name: prompt for name, prompt in ALL_EVAL_PERSONAS.items()}
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    names = list(persona_prompts.keys())
    embeddings = model.encode([persona_prompts[n] for n in names], convert_to_numpy=True)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    embeddings_norm = embeddings / norm
    cos_matrix = embeddings_norm @ embeddings_norm.T

    rows: list[dict] = []
    for r in h1_long:
        if r["is_self_loop"]:
            continue
        try:
            i = names.index(r["source"])
            j = names.index(r["target"])
        except ValueError:
            continue
        prompt_cos = float(cos_matrix[i, j])
        rows.append({**r, "prompt_cos": prompt_cos})

    return {
        "method": "pingouin.partial_corr(method='spearman')",
        "covariates": ["cosine_l15", "prompt_cos"],
        **_partial_spearman(rows, "js", "leakage", ["cosine_l15", "prompt_cos"]),
    }


def _c3_villain_out(h1_long: list[dict]) -> dict:
    # Cell counts (path-A n=71 contract, see _build_h1_long):
    #   * rows_n71 = all 71 directed off-diagonal cells
    #     (6 ALL_EVAL sources × 10 + 1 nurse source × 11).
    #   * rows_n61 = same minus the 10 villain-source cells.
    # Round-2 NIT carryover from epm:code-review v2: keys were previously
    # labelled "n70"/"n60" despite the actual cell counts being 71/61.
    rows_n71 = [r for r in h1_long if not r["is_self_loop"]]
    rows_n61 = [r for r in rows_n71 if r["source"] != "villain"]
    return {
        "n71": {
            "js_vs_leakage": cluster_bootstrap_spearman(rows_n71, "js", "leakage"),
            "cosine_l15_vs_leakage": cluster_bootstrap_spearman(rows_n71, "cosine_l15", "leakage"),
            "delta_js_minus_cosine_l15": paired_cluster_bootstrap_delta_abs_rho(
                rows_n71, "leakage", "js", "cosine_l15"
            ),
        },
        "n61": {
            "js_vs_leakage": cluster_bootstrap_spearman(rows_n61, "js", "leakage"),
            "cosine_l15_vs_leakage": cluster_bootstrap_spearman(rows_n61, "cosine_l15", "leakage"),
            "delta_js_minus_cosine_l15": paired_cluster_bootstrap_delta_abs_rho(
                rows_n61, "leakage", "js", "cosine_l15"
            ),
        },
    }


def _c4_temp1(leakage_dir: Path) -> dict:
    """C4 is conditional — only runs if raw_completions.json exists somewhere
    under leakage_dir. If absent, the plan §3 calls for "data unavailable;
    documented in standing caveats" — we record the status here.
    """
    if _check_temp1_cache(leakage_dir):
        # Caller would need to recompute JS at temp=1. Plan says "Run only
        # if raw_completions.json is available". We surface the cache without
        # actually doing the recomputation here; that requires a separate
        # GPU job. For the aggregator we just record availability.
        return {
            "status": "available_but_not_recomputed",
            "note": (
                "raw_completions.json exists; recompute JS at temp=1 is a "
                "separate GPU job (compute_js_convergence_228.py with a "
                "different seed/temp config). Not attempted here."
            ),
        }
    return {
        "status": "skipped",
        "reason": "raw_completions.json cache not found — documented in standing caveats",
    }


def _c5_entropy_partial(h1_long: list[dict]) -> dict:
    rows = [r for r in h1_long if not r["is_self_loop"]]
    return {
        "method": "pingouin.partial_corr(method='spearman')",
        "covariates": ["cosine_l15", "source_token_entropy_mean"],
        **_partial_spearman(rows, "js", "leakage", ["cosine_l15", "source_token_entropy_mean"]),
    }


# ── Partial Spearman (pingouin) ────────────────────────────────────────────


def _filter_complete_rows(rows: list[dict], keys: list[str]) -> list[dict]:
    """Return rows where every key has a finite numeric value."""

    def _row_ok(row: dict) -> bool:
        for k in keys:
            v = row.get(k)
            if v is None:
                return False
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return False
            if math.isnan(fv):
                return False
        return True

    return [r for r in rows if _row_ok(r)]


def _pingouin_pval_col(df) -> str:
    """Return the p-value column name for the installed pingouin version."""
    if "p_val" in df.columns:
        return "p_val"
    if "p-val" in df.columns:
        return "p-val"
    raise RuntimeError(f"pingouin partial_corr returned no p-value column; got {list(df.columns)}")


def _partial_spearman(
    rows: list[dict],
    x: str,
    y: str,
    covariates: list[str],
    *,
    cluster_field: str = "source",
    n_iter: int = BOOTSTRAP_ITER,
    seed: int = BOOTSTRAP_SEED + 17,
) -> dict:
    """Partial Spearman ρ via pingouin.partial_corr + cluster-bootstrap CI.

    Drops rows where any of (x, y, covariates) is None/NaN. Returns:
        {rho, p_value, n, df, covariates, ci_low, ci_high}
    """
    import pandas as pd
    import pingouin as pg

    keep_keys = [x, y, *list(covariates)]
    df_rows = _filter_complete_rows(rows, keep_keys)
    if len(df_rows) < len(covariates) + 3:
        return {
            "rho": float("nan"),
            "p_value": float("nan"),
            "n": len(df_rows),
            "df": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "note": "too few rows after dropna",
        }

    df = pd.DataFrame(df_rows)
    point = pg.partial_corr(data=df, x=x, y=y, covar=list(covariates), method="spearman")
    p_col = _pingouin_pval_col(point)
    rho = float(point["r"].iloc[0])
    pval = float(point[p_col].iloc[0])
    n = int(point["n"].iloc[0])
    df_dof = n - len(covariates) - 2

    # Cluster bootstrap on rho.
    by_cluster: dict[str, list[dict]] = {}
    for r in df_rows:
        by_cluster.setdefault(r[cluster_field], []).append(r)
    clusters = list(by_cluster.keys())
    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    for _ in range(n_iter):
        chosen_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        boot_rows: list[dict] = []
        for c in chosen_clusters:
            inner_rows = by_cluster[c]
            idxs = rng.integers(0, len(inner_rows), size=len(inner_rows))
            boot_rows.extend(inner_rows[int(ii)] for ii in idxs)
        if len(boot_rows) < len(covariates) + 3:
            continue
        df_b = pd.DataFrame(boot_rows)
        try:
            res_b = pg.partial_corr(data=df_b, x=x, y=y, covar=list(covariates), method="spearman")
            rho_b = float(res_b["r"].iloc[0])
        except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
            # Singular covariance (rank-deficient on small bootstraps), ties,
            # or numpy LinAlgError on partial-corr inversion — skip this
            # iteration cleanly so a single bad resample does not abort the
            # whole bootstrap.
            continue
        if math.isnan(rho_b):
            continue
        rhos.append(rho_b)

    if rhos:
        ci_low = float(np.percentile(rhos, 2.5))
        ci_high = float(np.percentile(rhos, 97.5))
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    return {
        "rho": rho,
        "p_value": pval,
        "n": n,
        "df": df_dof,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_iter": n_iter,
        "n_bootstrap_valid": len(rhos),
    }


# ── H2 family ──────────────────────────────────────────────────────────────


def _h2_analysis(h2_long: list[dict]) -> dict:
    """Per-source Spearman ρ on (JS_t, leakage_t), raw and first-difference.

    Pre-registered: ≥4/7 sources reach BH-FDR p<0.05 with consistent (negative)
    sign for both raw and first-difference.
    """
    by_source: dict[str, list[dict]] = {}
    for r in h2_long:
        by_source.setdefault(r["source"], []).append(r)

    raw_rhos: list[dict] = []
    diff_rhos: list[dict] = []
    raw_p: list[float] = []
    diff_p: list[float] = []
    raw_signs: list[int] = []
    diff_signs: list[int] = []
    for source, src_rows in sorted(by_source.items()):
        # Raw: ρ(JS_t, leakage_t)
        src_rows = sorted(src_rows, key=lambda r: r["checkpoint_step"])
        js_t = [r["mean_js_off_diag"] for r in src_rows]
        leak_t = [r["mean_leakage_off_diag"] for r in src_rows]
        cos_t = [r["mean_cosine_l20_off_diag"] for r in src_rows]
        rho_raw, p_raw, n_raw = _spearman_xy(js_t, leak_t)
        rho_cos_raw, p_cos_raw, _ = _spearman_xy(cos_t, leak_t)
        raw_rhos.append(
            {
                "source": source,
                "rho_js": rho_raw,
                "p_js": p_raw,
                "rho_cos_l20": rho_cos_raw,
                "p_cos_l20": p_cos_raw,
                "n": n_raw,
            }
        )
        raw_p.append(p_raw if not math.isnan(p_raw) else 1.0)
        raw_signs.append(-1 if rho_raw < 0 else 1)

        # First-difference
        d_js = [js_t[i + 1] - js_t[i] for i in range(len(js_t) - 1)]
        d_leak = [leak_t[i + 1] - leak_t[i] for i in range(len(leak_t) - 1)]
        d_cos = [cos_t[i + 1] - cos_t[i] for i in range(len(cos_t) - 1)]
        rho_diff, p_diff, n_diff = _spearman_xy(d_js, d_leak)
        rho_cos_diff, p_cos_diff, _ = _spearman_xy(d_cos, d_leak)
        diff_rhos.append(
            {
                "source": source,
                "rho_js": rho_diff,
                "p_js": p_diff,
                "rho_cos_l20": rho_cos_diff,
                "p_cos_l20": p_cos_diff,
                "n": n_diff,
            }
        )
        diff_p.append(p_diff if not math.isnan(p_diff) else 1.0)
        diff_signs.append(-1 if rho_diff < 0 else 1)

    # BH-FDR
    raw_pass = []
    diff_pass = []
    if raw_p:
        rej, p_adj, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")
        for i, src in enumerate(sorted(by_source)):
            raw_pass.append(
                {
                    "source": src,
                    "p_raw": raw_p[i],
                    "p_adj": float(p_adj[i]),
                    "fdr_pass": bool(rej[i]),
                    "negative_sign": raw_signs[i] == -1,
                }
            )
    if diff_p:
        rej, p_adj, _, _ = multipletests(diff_p, alpha=0.05, method="fdr_bh")
        for i, src in enumerate(sorted(by_source)):
            diff_pass.append(
                {
                    "source": src,
                    "p_diff": diff_p[i],
                    "p_adj": float(p_adj[i]),
                    "fdr_pass": bool(rej[i]),
                    "negative_sign": diff_signs[i] == -1,
                }
            )

    raw_passing = sum(1 for r in raw_pass if r["fdr_pass"] and r["negative_sign"])
    diff_passing = sum(1 for r in diff_pass if r["fdr_pass"] and r["negative_sign"])
    return {
        "per_source_raw": raw_rhos,
        "per_source_diff": diff_rhos,
        "raw_fdr": raw_pass,
        "diff_fdr": diff_pass,
        "n_raw_passing_negative": raw_passing,
        "n_diff_passing_negative": diff_passing,
        "h2_pass_threshold_raw": 4,
        "h2_pass_threshold_diff": 4,
        "h2_pass_raw": raw_passing >= 4,
        "h2_pass_diff": diff_passing >= 4,
        "h2_pass_overall": raw_passing >= 4 and diff_passing >= 4,
    }


# ── H3 family ──────────────────────────────────────────────────────────────


def _h3_analysis(h1_long: list[dict]) -> dict:
    rows = [r for r in h1_long if not r["is_self_loop"]]
    primary = _partial_spearman(rows, "js", "leakage", ["cosine_l15"])
    reverse = _partial_spearman(rows, "cosine_l15", "leakage", ["js"])
    return {
        "primary_partial_js_given_cos_l15": primary,
        "reverse_partial_cos_l15_given_js": reverse,
    }


# ── Top-level ─────────────────────────────────────────────────────────────


def aggregate(
    input_dir: Path,
    leakage_dir: Path,
    js_baseline_142: Path,
    output_path: Path,
    correlations_path: Path,
) -> None:
    bundle = _build_records(input_dir, leakage_dir)
    states = bundle["states"]
    h1_long = bundle["h1_long"]
    h2_long = bundle["h2_long"]

    # Sanity 6.1 #4
    base_state = next(s for s in states if s["source"] == "base")
    sanity_142 = _sanity_check_142(base_state, js_baseline_142)

    # Persist all_results.json — long-format + state pointers.
    all_results = {
        "h1_long_off_diagonal_epoch20": h1_long,
        "h2_long_per_source_per_step": h2_long,
        "states_summary": [
            {
                "source": s["source"],
                "checkpoint_step": s["checkpoint_step"],
                "epoch": s["epoch"],
                "n_cells_raw": s.get("n_cells_raw"),
                "n_cells_no_marker": s.get("n_cells_no_marker"),
                "n_cells_skipped": s.get("n_cells_skipped"),
                "wall_seconds": s.get("wall_seconds"),
                "source_token_entropy_mean": s.get("source_token_entropy_mean"),
            }
            for s in states
        ],
        "sanity_check_142": sanity_142,
        "metadata": {
            "n_states": len(states),
            "n_h1_off_diagonal_cells": len(h1_long),
            "n_h2_rows": len(h2_long),
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }
    output_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Wrote %s (%d h1 cells, %d h2 rows)", output_path, len(h1_long), len(h2_long))

    # Run all correlation analyses.
    correlations = {
        "h1": _h1_analysis(h1_long),
        "h2": _h2_analysis(h2_long),
        "h3": _h3_analysis(h1_long),
        "c1_marker_mask": _c1_marker_mask(h1_long),
        "c1b_marker_free": _c1b_marker_free(h1_long),
        "c2_prompt_distance": _c2_prompt_distance(h1_long),
        "c3_villain_out": _c3_villain_out(h1_long),
        "c4_temp1": _c4_temp1(leakage_dir),
        "c5_entropy_partial": _c5_entropy_partial(h1_long),
        # NIT round-1 -> round-2 fix: removed misnamed ``h3_with_c2``. The
        # original entry computed the same thing as ``h3.primary_partial_js_given_cos_l15``
        # without an actual prompt-cos covariate. Use ``c2_prompt_distance`` for
        # the partial-ρ controlling for prompt cosine, and ``h3.primary_partial_js_given_cos_l15``
        # for the layer-15 cosine partial.
        "h3_with_entropy": _partial_spearman(
            [r for r in h1_long if not r["is_self_loop"]],
            "js",
            "leakage",
            ["cosine_l15", "source_token_entropy_mean"],
        ),
        "metadata": {
            "bootstrap_iterations": BOOTSTRAP_ITER,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "epoch_for_h1": EPOCH_FOR_H1,
            "cosine_layer_h1": COSINE_LAYER_H1,
            "cosine_layer_h2": COSINE_LAYER_H2,
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }
    correlations_path.write_text(json.dumps(correlations, indent=2))
    logger.info("Wrote %s", correlations_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_228",
    )
    parser.add_argument(
        "--leakage-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "causal_proximity" / "strong_convergence",
    )
    parser.add_argument(
        "--js-baseline-142",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "js_divergence" / "divergence_matrices.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_228" / "all_results.json",
    )
    parser.add_argument(
        "--correlations",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_228" / "correlations.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.correlations.parent.mkdir(parents=True, exist_ok=True)

    aggregate(
        input_dir=args.input_dir,
        leakage_dir=args.leakage_dir,
        js_baseline_142=args.js_baseline_142,
        output_path=args.output,
        correlations_path=args.correlations,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
