# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #448 v5 Phase 5 — analysis (H1a/H1b + same-row contrasts + efficiency).

Consumes the v5 on-policy ``marker_logprob.json`` schema written by
``eval_one_cell.run_eval``:

  - ``g_logprob_per_persona_q[p][q]``  — trained-adapter logp(MARKER) at slot
  - ``b_logprob_per_persona_q[p][q]``  — base-model    logp(MARKER) at slot
  - ``delta_g_per_persona_q[p][q]``     = g - b
  - ``emission_recompute_per_persona_q[p][q]`` — argmax == MARKER (bool)

Plus:
  - ``data/issue_448/held_out_bystanders.json`` (Phase 0 pin) — the H1b
    primary denominator (~15 panel personas never trained as negatives in
    ANY cell).

Outputs (plan §10):
  ``eval_results/issue_448_v5/analyze_summary.json`` —
      per-cell {mean_delta_held_out, mean_delta_unstratified,
                source_self_diagonal, sd_delta, emission_rate,
                contrast_efficiency, ...} +
      per-knob H1a/H1b monotonicity + permutation-null +
      same-row-count contrasts +
      H3 diagonal-implant gate per cell +
      H4 constant-emission gate per cell +
      H5 per-cell Spearman ρ(per-bystander ΔG, cosine-distance-to-nearest-neg).

Figures (plan §6.4): hero_4knob_sweep_v5, held_out_vs_unstratified_scatter,
same_row_count_contrasts, contrast_efficiency_bar, per_cell_distribution,
diagonal_implant_bar, constant_emission_check, bystander_panel_heatmap,
secondary_per_cell_scatter, secondary_rho_summary, emission_rate_bar,
knob_permutation_null. (The analyzer over-produces; clean-result picks
the hero.)

CPU-only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (  # noqa: E402
    CELL_SPECS,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.contrastive_recipe_sweep_448.held_out_bystanders import (  # noqa: E402
    load_held_out_artifact,
)
from explore_persona_space.experiments.factor_screen_365.persona_panel import (  # noqa: E402
    EVAL_PERSONAS_24,
)

log = logging.getLogger("issue_448.analyze")

BOOTSTRAP_N = 2_000
PERMUTATION_N = 10_000
HEADLINE_DELTA_RANGE_NATS = 1.0  # plan §6.2 H1 bar
HEADLINE_P_THRESHOLD = 0.10  # plan §6.2 H1 bar (one-sided)
H3_DIAGONAL_FLOOR_NATS = 5.0  # plan §3 H3
H4_SD_FLOOR_NATS = 1.5  # plan §3 H4
H4_SD_HARD_FAIL = 0.5
H5_RHO_PASS_THRESHOLD = -0.30
H5_RHO_DEGEN_THRESHOLD = 0.20
SCHEMA_VERSION = "i448_v5_analyze"
SAME_ROW_COUNT_CONTRASTS: list[tuple[str, str, str]] = [
    # (label, cell_a, cell_b) — both share row count + total negatives, differ
    # in persona diversity (plan §6.2 Alternatives must-fix).
    ("c8_neg_ex_400__vs__c10_neg_personas_4", "c8_neg_ex_400", "c10_neg_personas_4"),
    ("c9_neg_ex_800__vs__c11_neg_personas_8", "c9_neg_ex_800", "c11_neg_personas_8"),
]

# Pre-registered direction per knob (plan §3 H1 + §6.5 #6 signed monotone).
# "down" => widening the knob is HYPOTHESIZED to REDUCE bystander leakage.
# "up"   => widening is hypothesized to RAISE (only positive-side knobs).
KNOB_HYPOTHESIZED_DIRECTION: dict[str, str] = {
    "pos_ex_per_persona": "up",
    "pos_personas": "up",
    "neg_ex_per_persona": "down",
    "neg_personas": "down",
}


# ── Stats helpers (kept from v4 — pure functions). ────────────────────────────


def _bootstrap_ci(
    values: np.ndarray, n_iter: int = BOOTSTRAP_N, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap mean + (lower, upper) CI of ``values``. Returns (mean, low, high)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(n_iter, n))
    samples = values[idx]
    means = samples.mean(axis=1)
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return float(np.mean(values)), low, high


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via scipy (with tie correction)."""
    try:
        from scipy.stats import spearmanr

        if len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
            return 0.0
        rho, _ = spearmanr(x, y)
        return float(rho) if rho == rho else 0.0
    except ImportError:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        num = float(np.sum((rx - rx.mean()) * (ry - ry.mean())))
        denom = float(np.sqrt(np.sum((rx - rx.mean()) ** 2) * np.sum((ry - ry.mean()) ** 2)))
        return num / denom if denom > 0 else 0.0


def _bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, n_iter: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap (point, 2.5%, 97.5%) for Spearman ρ."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(n_iter, n))
    rhos = np.array([_spearman_rho(x[i], y[i]) for i in idx])
    return (
        float(_spearman_rho(x, y)),
        float(np.quantile(rhos, 0.025)),
        float(np.quantile(rhos, 0.975)),
    )


# ── Cell-data loaders (v5 schema). ────────────────────────────────────────────


def _load_v5_logp(path: Path) -> dict[str, Any]:
    """Load a v5 per-cell marker_logprob.json. Raises on schema drift."""
    if not path.exists():
        raise FileNotFoundError(f"Eval JSON not found: {path}")
    data = json.loads(path.read_text())
    sv = data.get("schema_version")
    if sv != "i448_v5":
        raise ValueError(
            f"Unexpected schema_version in {path}: {sv!r}. Expected 'i448_v5'. "
            f"Re-run Phase 4 eval_one_cell under the v5 dispatcher."
        )
    return data


def _per_persona_mean_from_pq(by_pq: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean across questions per persona; empty -> NaN."""
    out: dict[str, float] = {}
    for persona, by_q in by_pq.items():
        vals = [float(v) for v in by_q.values()]
        if not vals:
            out[persona] = float("nan")
        else:
            out[persona] = float(np.mean(vals))
    return out


def _per_persona_rate_from_pq(by_pq: dict[str, dict[str, bool]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for persona, by_q in by_pq.items():
        vals = [1.0 if v else 0.0 for v in by_q.values()]
        if not vals:
            out[persona] = float("nan")
        else:
            out[persona] = float(np.mean(vals))
    return out


# ── Per-cell analysis (v5). ───────────────────────────────────────────────────


def analyze_cell(
    cell_slug: str,
    cell_logp: dict[str, Any],
    held_out: list[str],
    source: str = SOURCE_PERSONA,
) -> dict[str, Any]:
    """Compute per-cell H1a/H1b means + H3 diagonal + H4 sd + emission rate.

    Args:
        cell_slug: e.g. ``"c1_anchor"``.
        cell_logp: Parsed v5 ``marker_logprob.json`` payload.
        held_out: List of held-out bystander persona names (Phase 0 pin).
        source: Source persona (excluded from bystander denominators).

    Returns:
        Per-cell dict for the analyze summary.
    """
    g_pq = cell_logp["g_logprob_per_persona_q"]
    b_pq = cell_logp["b_logprob_per_persona_q"]
    d_pq = cell_logp["delta_g_per_persona_q"]
    em_pq = cell_logp["emission_recompute_per_persona_q"]

    panel_personas = list(EVAL_PERSONAS_24.keys())
    if source not in panel_personas:
        raise AssertionError(f"source {source!r} not in EVAL_PERSONAS_24")

    bystanders_23 = [p for p in panel_personas if p != source]
    bystanders_held_out = [p for p in held_out if p in panel_personas and p != source]

    # ΔG per persona = mean over 20 questions.
    delta_per_persona = _per_persona_mean_from_pq(d_pq)
    g_per_persona = _per_persona_mean_from_pq(g_pq)
    b_per_persona = _per_persona_mean_from_pq(b_pq)
    emission_per_persona = _per_persona_rate_from_pq(em_pq)

    # H1a: unstratified 23-bystander mean.
    delta_23_arr = np.array(
        [delta_per_persona[p] for p in bystanders_23 if p in delta_per_persona], dtype=float
    )
    mean_23, low_23, high_23 = (
        _bootstrap_ci(delta_23_arr)
        if delta_23_arr.size
        else (
            float("nan"),
            float("nan"),
            float("nan"),
        )
    )

    # H1b: held-out subset mean.
    delta_held_out_arr = np.array(
        [delta_per_persona[p] for p in bystanders_held_out if p in delta_per_persona],
        dtype=float,
    )
    mean_held_out, low_held_out, high_held_out = (
        _bootstrap_ci(delta_held_out_arr)
        if delta_held_out_arr.size
        else (float("nan"), float("nan"), float("nan"))
    )

    # H3: diagonal implant gate — ΔG on the source's OWN R.
    source_diagonal_delta = float(delta_per_persona.get(source, float("nan")))
    h3_pass = (
        source_diagonal_delta == source_diagonal_delta  # not NaN
        and source_diagonal_delta > H3_DIAGONAL_FLOOR_NATS
    )

    # H4: sd of ΔG across the 24×20 grid.
    all_delta_values: list[float] = []
    for p in panel_personas:
        if p in d_pq:
            all_delta_values.extend(float(v) for v in d_pq[p].values())
    sd_delta_grid = float(np.std(all_delta_values, ddof=1)) if len(all_delta_values) > 1 else 0.0
    h4_pass = sd_delta_grid >= H4_SD_FLOOR_NATS
    h4_hard_fail = sd_delta_grid < H4_SD_HARD_FAIL

    # Contrast efficiency: mean_held_out / source_diagonal_delta.
    if h3_pass and source_diagonal_delta > 1e-6:
        contrast_efficiency = float(mean_held_out / source_diagonal_delta)
    else:
        contrast_efficiency = float("nan")

    # Emission rates: held-out + 23-bystander.
    em_held_out = (
        float(
            np.mean(
                [emission_per_persona[p] for p in bystanders_held_out if p in emission_per_persona]
            )
        )
        if bystanders_held_out
        else float("nan")
    )
    em_23 = (
        float(
            np.mean([emission_per_persona[p] for p in bystanders_23 if p in emission_per_persona])
        )
        if bystanders_23
        else float("nan")
    )

    return {
        "cell": cell_slug,
        "source": source,
        # H1a unstratified
        "n_bystanders_23": int(delta_23_arr.size),
        "mean_bystander_delta_unstratified": float(mean_23),
        "ci_bystander_delta_unstratified": [float(low_23), float(high_23)],
        # H1b held-out (HEADLINE)
        "n_bystanders_held_out": int(delta_held_out_arr.size),
        "held_out_personas": bystanders_held_out,
        "mean_bystander_delta_held_out": float(mean_held_out),
        "ci_bystander_delta_held_out": [float(low_held_out), float(high_held_out)],
        # H3 diagonal
        "source_diagonal_delta": float(source_diagonal_delta),
        "h3_diagonal_pass": bool(h3_pass),
        "h3_diagonal_floor_nats": H3_DIAGONAL_FLOOR_NATS,
        # H4 constant-emission
        "sd_delta_grid": float(sd_delta_grid),
        "h4_constant_emission_pass": bool(h4_pass),
        "h4_constant_emission_hard_fail": bool(h4_hard_fail),
        "h4_sd_floor_nats": H4_SD_FLOOR_NATS,
        # Contrast efficiency
        "contrast_efficiency": float(contrast_efficiency),
        # Emission rates (free anchor)
        "emission_rate_held_out": float(em_held_out),
        "emission_rate_unstratified_23": float(em_23),
        # Per-persona means (for the figure functions + secondary).
        "per_persona_delta_g": {p: float(v) for p, v in delta_per_persona.items()},
        "per_persona_g_logprob": {p: float(v) for p, v in g_per_persona.items()},
        "per_persona_b_logprob": {p: float(v) for p, v in b_per_persona.items()},
        "per_persona_emission_rate": {p: float(v) for p, v in emission_per_persona.items()},
    }


# ── Per-knob axes. ────────────────────────────────────────────────────────────


def _knob_axes() -> dict[str, list[tuple[str, int]]]:
    """4 knobs × (cell_slug, knob_level) traces (anchor sits on every axis)."""
    knob_axes: dict[str, list[tuple[str, int]]] = {
        "pos_ex_per_persona": [],
        "pos_personas": [],
        "neg_ex_per_persona": [],
        "neg_personas": [],
    }
    for slug, _name, pos_ex, pos_p, neg_ex, neg_p in CELL_SPECS:
        is_anchor = slug == "c1_anchor"
        if is_anchor or (pos_p == 1 and neg_ex == 200 and neg_p == 2):
            knob_axes["pos_ex_per_persona"].append((slug, pos_ex))
        if is_anchor or (pos_ex == 200 and neg_ex == 200 and neg_p == 2):
            knob_axes["pos_personas"].append((slug, pos_p))
        if is_anchor or (pos_ex == 200 and pos_p == 1 and neg_p == 2):
            knob_axes["neg_ex_per_persona"].append((slug, neg_ex))
        if is_anchor or (pos_ex == 200 and pos_p == 1 and neg_ex == 200):
            knob_axes["neg_personas"].append((slug, neg_p))
    for k in list(knob_axes.keys()):
        # dedupe by slug, then sort by level.
        seen = set()
        uniq: list[tuple[str, int]] = []
        for slug, lvl in knob_axes[k]:
            if slug in seen:
                continue
            seen.add(slug)
            uniq.append((slug, lvl))
        knob_axes[k] = sorted(uniq, key=lambda kv: kv[1])
    return knob_axes


def _knob_monotone_reduction(means_by_level: list[float], hypothesized_dir: str) -> dict[str, Any]:
    """Test whether per-cell means show monotone reduction (or rise) ≥ threshold.

    Returns dict with {monotone_in_hypothesis_direction, delta_range,
    sign_in_hypothesis_direction}. The "reduction headline" only counts the
    knob if the hypothesized direction matches: for negative-side knobs we
    want monotone DOWN; for positive-side we want monotone UP (which does
    NOT count toward H1 — H1 is a reduction headline only — see plan §6.5
    #6 signed-vs-unsigned).
    """
    if len(means_by_level) < 2:
        return {
            "monotone_in_hypothesis_direction": False,
            "delta_range_nats": 0.0,
            "sign_in_hypothesis_direction": False,
            "monotone_up": False,
            "monotone_down": False,
        }
    diffs = [means_by_level[i + 1] - means_by_level[i] for i in range(len(means_by_level) - 1)]
    monotone_up = all(d > 0 for d in diffs)
    monotone_down = all(d < 0 for d in diffs)
    delta_range = max(means_by_level) - min(means_by_level)
    if hypothesized_dir == "down":
        sign_match = monotone_down
    elif hypothesized_dir == "up":
        sign_match = monotone_up
    else:
        sign_match = monotone_up or monotone_down
    return {
        "monotone_in_hypothesis_direction": bool(sign_match),
        "delta_range_nats": float(delta_range),
        "sign_in_hypothesis_direction": bool(sign_match),
        "monotone_up": bool(monotone_up),
        "monotone_down": bool(monotone_down),
    }


def _permutation_null_signed_reduction_count(
    cells: list[str],
    means_by_cell: dict[str, float],
    knob_axes: dict[str, list[tuple[str, int]]],
    threshold: float,
    n_iter: int = PERMUTATION_N,
    seed: int = 42,
) -> dict[str, Any]:
    """Permutation null for the H1 headline (count of signed-reducing knobs).

    Shuffle cells' assignments across the knob axes; recompute the count of
    knobs that fire as monotone-in-hypothesized-direction AND have
    delta_range ≥ threshold. The observed count is the right tail.
    """
    rng = np.random.default_rng(seed)

    def _count(per_cell_means: dict[str, float]) -> int:
        count = 0
        for knob, axis in knob_axes.items():
            ordered_means = [per_cell_means[s] for s, _ in axis if s in per_cell_means]
            if len(ordered_means) != len(axis):
                continue
            result = _knob_monotone_reduction(ordered_means, KNOB_HYPOTHESIZED_DIRECTION[knob])
            # Signed monotone reduction only: "down" knobs (negative-side) count;
            # "up" knobs (positive-side) do NOT count toward H1's reduction
            # headline (plan §6.5 #6 signed-vs-unsigned).
            if (
                result["monotone_in_hypothesis_direction"]
                and result["delta_range_nats"] >= threshold
                and KNOB_HYPOTHESIZED_DIRECTION[knob] == "down"
            ):
                count += 1
        return count

    observed_count = _count(means_by_cell)

    null_counts: list[int] = []
    cell_list = list(cells)
    mean_values = np.array([means_by_cell[c] for c in cell_list], dtype=float)
    for _ in range(n_iter):
        perm = rng.permutation(len(cell_list))
        permuted = {c: float(mean_values[perm[i]]) for i, c in enumerate(cell_list)}
        null_counts.append(_count(permuted))
    null_arr = np.array(null_counts)
    p_value = float(np.mean(null_arr >= observed_count))
    return {
        "observed_count": int(observed_count),
        "null_median": float(np.median(null_arr)),
        "null_95pct_upper": float(np.quantile(null_arr, 0.95)),
        "null_mean": float(null_arr.mean()),
        "null_std": float(null_arr.std(ddof=1)) if len(null_arr) > 1 else 0.0,
        "empirical_p_value_one_sided": p_value,
        "n_iter": int(n_iter),
        "threshold_nats": float(threshold),
    }


# ── H5 secondary: per-cell ρ(per-bystander ΔG, cosine-distance-to-nearest-neg).


def _compute_h5(  # noqa: C901 - inline 4-block (resolve negs / build pairs / degen gate / bootstrap)
    per_cell_results: dict[str, dict[str, Any]],
    centroids_path: Path | None,
    source: str = SOURCE_PERSONA,
) -> dict[str, dict[str, Any]]:
    """For every non-degenerate cell, compute Spearman ρ(per-bystander ΔG,
    cosine_distance_to_nearest_negative).

    Plan §3 H5 + §6.2. PASS bar = ρ < −0.30 with CI excluding 0.
    """
    if centroids_path is None or not centroids_path.exists():
        log.warning("H5: centroids missing at %s — skipping secondary", centroids_path)
        return {slug: {"skipped_reason": "centroids_missing"} for slug in per_cell_results}

    try:
        import torch
    except ImportError:
        log.warning("H5: torch missing — skipping secondary")
        return {slug: {"skipped_reason": "torch_missing"} for slug in per_cell_results}

    bundle = torch.load(centroids_path, weights_only=False)
    layer = bundle.get("layer", 20)
    tensor = bundle["centroids"][layer].to(torch.float32).numpy()
    names = list(bundle["persona_names"])
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Import the per-cell-negative-persona list from the held-out resolver
    # (it computed per-cell negatives already).
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        persona_registry as registry,
    )

    h5: dict[str, dict[str, Any]] = {}
    for slug, cell_data in per_cell_results.items():
        # Resolve the cell's negative personas via the registry.
        spec = next((row for row in CELL_SPECS if row[0] == slug), None)
        if spec is None:
            h5[slug] = {"skipped_reason": "cell_spec_missing"}
            continue
        _slug, _name, _pos_ex, pos_p, _neg_ex, neg_p = spec
        if neg_p == 2:
            neg_set = registry.get_anchor_bystanders(source)
        elif neg_p in (4, 8):
            if pos_p == 1:
                exclude = {source}
            elif pos_p == 2:
                from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
                    MULTI_POSITIVE_PERSONAS_C5,
                )

                exclude = set(MULTI_POSITIVE_PERSONAS_C5)
            else:
                from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
                    MULTI_POSITIVE_PERSONAS_C6,
                )

                exclude = set(MULTI_POSITIVE_PERSONAS_C6)
            neg_set = registry.select_n_bystanders(source, neg_p, exclude=exclude)
        else:
            h5[slug] = {"skipped_reason": "unsupported_neg_count"}
            continue

        neg_idxs = [name_to_idx[n] for n in neg_set if n in name_to_idx]
        if not neg_idxs:
            h5[slug] = {"skipped_reason": "no_neg_centroids"}
            continue
        neg_vecs = tensor[neg_idxs]

        # For each bystander (excluding source + negatives + multi-positives),
        # compute cosine_distance_to_nearest_negative + pair it with per-
        # bystander mean ΔG.
        delta = cell_data["per_persona_delta_g"]
        skip_set = {source, *neg_set}
        if pos_p == 2:
            skip_set.update({"villain", "comedian"})
        elif pos_p == 4:
            skip_set.update({"villain", "comedian", "assistant", "software_engineer"})
        pairs: list[tuple[str, float, float]] = []
        for p in EVAL_PERSONAS_24:
            if p in skip_set or p not in name_to_idx:
                continue
            if p not in delta:
                continue
            vec = tensor[name_to_idx[p]]
            sims = neg_vecs @ vec / (np.linalg.norm(neg_vecs, axis=1) * np.linalg.norm(vec) + 1e-12)
            nearest_dist = float(1 - sims.max())
            pairs.append((p, nearest_dist, float(delta[p])))
        if len(pairs) < 4:
            h5[slug] = {"skipped_reason": "insufficient_bystanders", "n": len(pairs)}
            continue

        x = np.array([t[1] for t in pairs])
        y = np.array([t[2] for t in pairs])
        # Degeneracy gate: sd of per-bystander Δ < 0.3 nat → too flat to test.
        sd_y = float(np.std(y, ddof=1))
        if sd_y < 0.3:
            h5[slug] = {
                "skipped_reason": "degenerate_low_sd_delta",
                "sd_delta": sd_y,
                "n": len(pairs),
            }
            continue
        rho, low, high = _bootstrap_spearman(x, y, n_iter=1000, seed=42)
        ci_excludes_zero = (low > 0 and high > 0) or (low < 0 and high < 0)
        passes_h5_bar = (rho < H5_RHO_PASS_THRESHOLD) and ci_excludes_zero
        collapses = (abs(rho) < H5_RHO_DEGEN_THRESHOLD) and not ci_excludes_zero
        h5[slug] = {
            "rho_point": rho,
            "rho_ci_low": low,
            "rho_ci_high": high,
            "n_bystanders": len(pairs),
            "ci_excludes_zero": bool(ci_excludes_zero),
            "passes_h5_bar": bool(passes_h5_bar),
            "collapses": bool(collapses),
            "sd_delta": sd_y,
            "per_bystander": [
                {"persona": p, "nearest_neg_distance": d, "delta_g": delta_v}
                for p, d, delta_v in pairs
            ],
        }
    return h5


# ── Figures (plan §6.4). ──────────────────────────────────────────────────────


def _make_figures(
    summary: dict[str, Any],
    knob_axes: dict[str, list[tuple[str, int]]],
    figures_dir: Path,
) -> list[str]:
    """Generate all v5 figures listed in plan §6.4. Returns relative paths."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib missing — skipping figures")
        return []

    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_style

        apply_paper_style()
    except Exception:  # pragma: no cover
        log.warning("paper_plots.apply_paper_style failed — using matplotlib defaults")

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[str] = []
    per_cell = summary["per_cell"]
    cells = list(per_cell.keys())

    def _save(fig, name: str) -> None:
        path = figures_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(str(path))
        log.info("Wrote figure -> %s", path)

    # 1) hero_4knob_sweep_v5: two-row (held-out top, unstratified bottom).
    try:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey="row")
        for col, (knob, axis) in enumerate(knob_axes.items()):
            xs = [lvl for _, lvl in axis]
            slugs = [s for s, _ in axis]
            held = [per_cell[s]["mean_bystander_delta_held_out"] for s in slugs if s in per_cell]
            held_lo = [
                per_cell[s]["ci_bystander_delta_held_out"][0] for s in slugs if s in per_cell
            ]
            held_hi = [
                per_cell[s]["ci_bystander_delta_held_out"][1] for s in slugs if s in per_cell
            ]
            unstrat = [
                per_cell[s]["mean_bystander_delta_unstratified"] for s in slugs if s in per_cell
            ]
            uns_lo = [
                per_cell[s]["ci_bystander_delta_unstratified"][0] for s in slugs if s in per_cell
            ]
            uns_hi = [
                per_cell[s]["ci_bystander_delta_unstratified"][1] for s in slugs if s in per_cell
            ]
            ax_h = axes[0, col]
            ax_u = axes[1, col]
            held_err = np.array(
                [
                    [m - lo for m, lo in zip(held, held_lo, strict=True)],
                    [hi - m for m, hi in zip(held, held_hi, strict=True)],
                ]
            )
            uns_err = np.array(
                [
                    [m - lo for m, lo in zip(unstrat, uns_lo, strict=True)],
                    [hi - m for m, hi in zip(unstrat, uns_hi, strict=True)],
                ]
            )
            ax_h.errorbar(xs, held, yerr=held_err, fmt="o-", capsize=4)
            ax_u.errorbar(xs, unstrat, yerr=uns_err, fmt="s-", capsize=4)
            ax_h.set_title(knob.replace("_", " "))
            ax_u.set_xlabel(knob.replace("_", " "))
            ax_h.set_xscale("log") if knob.endswith("ex_per_persona") else None
            ax_u.set_xscale("log") if knob.endswith("ex_per_persona") else None
            ax_h.axhline(0, color="grey", lw=0.5)
            ax_u.axhline(0, color="grey", lw=0.5)
        axes[0, 0].set_ylabel("mean ΔG (held-out, H1b)")
        axes[1, 0].set_ylabel("mean ΔG (unstratified, H1a)")
        fig.suptitle("4-knob sweep — held-out (top) vs unstratified (bottom)")
        fig.tight_layout()
        _save(fig, "hero_4knob_sweep_v5.png")
    except Exception as exc:
        log.exception("hero_4knob_sweep_v5 failed: %s", exc)

    # 2) held_out_vs_unstratified_scatter.
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        held = [per_cell[c]["mean_bystander_delta_held_out"] for c in cells]
        unstrat = [per_cell[c]["mean_bystander_delta_unstratified"] for c in cells]
        ax.scatter(held, unstrat)
        for c, h, u in zip(cells, held, unstrat, strict=True):
            ax.annotate(c, (h, u), fontsize=7)
        lo = min(min(held), min(unstrat))
        hi = max(max(held), max(unstrat))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.5)
        ax.set_xlabel("mean ΔG held-out (H1b)")
        ax.set_ylabel("mean ΔG unstratified (H1a)")
        ax.set_title("held-out vs unstratified per cell")
        _save(fig, "held_out_vs_unstratified_scatter.png")
    except Exception as exc:
        log.exception("held_out_vs_unstratified_scatter failed: %s", exc)

    # 3) same_row_count_contrasts (bar).
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        contrast_data = summary.get("same_row_count_contrasts", [])
        labels = [c["label"] for c in contrast_data]
        diffs = [c["delta_held_out_difference"] for c in contrast_data]
        lows = [c["ci_low"] for c in contrast_data]
        highs = [c["ci_high"] for c in contrast_data]
        if labels:
            err = np.array(
                [
                    [d - lo for d, lo in zip(diffs, lows, strict=True)],
                    [hi - d for d, hi in zip(diffs, highs, strict=True)],
                ]
            )
            ax.bar(labels, diffs, yerr=err, capsize=5)
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_ylabel("ΔΔG (cell_a − cell_b, held-out)")
            ax.set_title("Same-row-count contrasts")
            plt.xticks(rotation=15, ha="right", fontsize=8)
        _save(fig, "same_row_count_contrasts.png")
    except Exception as exc:
        log.exception("same_row_count_contrasts failed: %s", exc)

    # 4) contrast_efficiency_bar.
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        eff = [per_cell[c]["contrast_efficiency"] for c in cells]
        ax.bar(cells, eff)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axhline(1, color="red", lw=0.5, ls="--", label="implant-dilution (= 1)")
        ax.set_ylabel("efficiency = held-out ΔG / diagonal ΔG")
        ax.set_title("Contrast efficiency per cell")
        ax.legend()
        plt.xticks(rotation=20, ha="right", fontsize=8)
        _save(fig, "contrast_efficiency_bar.png")
    except Exception as exc:
        log.exception("contrast_efficiency_bar failed: %s", exc)

    # 5) diagonal_implant_bar (H3).
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        diags = [per_cell[c]["source_diagonal_delta"] for c in cells]
        ax.bar(cells, diags)
        ax.axhline(H3_DIAGONAL_FLOOR_NATS, color="red", lw=0.5, ls="--", label="H3 floor")
        ax.set_ylabel("source-self ΔG (nats)")
        ax.set_title("H3 diagonal implant gate")
        ax.legend()
        plt.xticks(rotation=20, ha="right", fontsize=8)
        _save(fig, "diagonal_implant_bar.png")
    except Exception as exc:
        log.exception("diagonal_implant_bar failed: %s", exc)

    # 6) constant_emission_check (H4).
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        sds = [per_cell[c]["sd_delta_grid"] for c in cells]
        ax.bar(cells, sds)
        ax.axhline(H4_SD_FLOOR_NATS, color="red", lw=0.5, ls="--", label="H4 floor")
        ax.set_ylabel("sd(ΔG) across 24x20 grid")
        ax.set_title("H4 constant-emission check")
        ax.legend()
        plt.xticks(rotation=20, ha="right", fontsize=8)
        _save(fig, "constant_emission_check.png")
    except Exception as exc:
        log.exception("constant_emission_check failed: %s", exc)

    # 7) emission_rate_bar (held-out).
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        em = [per_cell[c]["emission_rate_held_out"] for c in cells]
        ax.bar(cells, em)
        ax.set_ylabel("argmax-emission rate on held-out")
        ax.set_title("Marker emission rate on held-out bystanders")
        plt.xticks(rotation=20, ha="right", fontsize=8)
        _save(fig, "emission_rate_held_out_bar.png")
    except Exception as exc:
        log.exception("emission_rate_held_out_bar failed: %s", exc)

    return out_paths


# ── End-to-end. ───────────────────────────────────────────────────────────────


def run_analysis(
    slab_root: Path,
    held_out_path: Path,
    figures_dir: Path,
    out_path: Path,
    centroids_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full v5 analysis. Returns summary dict; writes ``out_path`` + figures."""
    held_out_payload = load_held_out_artifact(held_out_path)
    held_out_personas = held_out_payload["held_out"]
    source = held_out_payload.get("source", SOURCE_PERSONA)
    log.info(
        "Loaded held-out artifact: n_held_out=%d (source=%s) from %s",
        len(held_out_personas),
        source,
        held_out_path,
    )

    per_cell: dict[str, dict[str, Any]] = {}
    cell_slugs: list[str] = []
    for slug, _name, _pe, _pp, _ne, _np in CELL_SPECS:
        eval_path = slab_root / slug / "marker_logprob.json"
        if not eval_path.exists():
            log.warning("Skipping cell %s — eval JSON missing at %s", slug, eval_path)
            continue
        cell_logp = _load_v5_logp(eval_path)
        per_cell[slug] = analyze_cell(slug, cell_logp, held_out_personas, source=source)
        cell_slugs.append(slug)

    if not per_cell:
        raise RuntimeError(
            f"run_analysis: no per-cell eval JSONs found under {slab_root}; cannot analyze."
        )

    # H1 permutation null per denominator.
    knob_axes = _knob_axes()
    mean_held_by_cell = {c: per_cell[c]["mean_bystander_delta_held_out"] for c in cell_slugs}
    mean_unstrat_by_cell = {c: per_cell[c]["mean_bystander_delta_unstratified"] for c in cell_slugs}
    h1b_null = _permutation_null_signed_reduction_count(
        cell_slugs,
        mean_held_by_cell,
        knob_axes,
        HEADLINE_DELTA_RANGE_NATS,
    )
    h1a_null = _permutation_null_signed_reduction_count(
        cell_slugs,
        mean_unstrat_by_cell,
        knob_axes,
        HEADLINE_DELTA_RANGE_NATS,
    )

    # Per-knob diagnostic.
    per_knob_h1b: dict[str, dict[str, Any]] = {}
    per_knob_h1a: dict[str, dict[str, Any]] = {}
    for knob, axis in knob_axes.items():
        ordered = [s for s, _ in axis]
        held = [mean_held_by_cell[s] for s in ordered if s in mean_held_by_cell]
        unstrat = [mean_unstrat_by_cell[s] for s in ordered if s in mean_unstrat_by_cell]
        if len(held) != len(axis) or len(unstrat) != len(axis):
            per_knob_h1b[knob] = {"valid": False, "reason": "missing_cells"}
            per_knob_h1a[knob] = {"valid": False, "reason": "missing_cells"}
            continue
        h1b_res = _knob_monotone_reduction(held, KNOB_HYPOTHESIZED_DIRECTION[knob])
        h1a_res = _knob_monotone_reduction(unstrat, KNOB_HYPOTHESIZED_DIRECTION[knob])
        per_knob_h1b[knob] = {
            "valid": True,
            "axis_cells": ordered,
            "axis_levels": [lvl for _, lvl in axis],
            "means": held,
            "hypothesized_direction": KNOB_HYPOTHESIZED_DIRECTION[knob],
            **h1b_res,
            "fires_for_h1": (
                h1b_res["monotone_in_hypothesis_direction"]
                and h1b_res["delta_range_nats"] >= HEADLINE_DELTA_RANGE_NATS
                and KNOB_HYPOTHESIZED_DIRECTION[knob] == "down"
            ),
        }
        per_knob_h1a[knob] = {
            "valid": True,
            "axis_cells": ordered,
            "axis_levels": [lvl for _, lvl in axis],
            "means": unstrat,
            "hypothesized_direction": KNOB_HYPOTHESIZED_DIRECTION[knob],
            **h1a_res,
            "fires_for_h1": (
                h1a_res["monotone_in_hypothesis_direction"]
                and h1a_res["delta_range_nats"] >= HEADLINE_DELTA_RANGE_NATS
                and KNOB_HYPOTHESIZED_DIRECTION[knob] == "down"
            ),
        }

    # Same-row-count contrasts.
    contrasts: list[dict[str, Any]] = []
    for label, cell_a, cell_b in SAME_ROW_COUNT_CONTRASTS:
        if cell_a not in per_cell or cell_b not in per_cell:
            contrasts.append({"label": label, "skipped_reason": "missing_cells"})
            continue
        # Paired bootstrap on the per-bystander ΔG vectors restricted to
        # held-out personas.
        held_out_set = set(held_out_personas)
        a_per = per_cell[cell_a]["per_persona_delta_g"]
        b_per = per_cell[cell_b]["per_persona_delta_g"]
        shared = sorted([p for p in held_out_set if p in a_per and p in b_per and p != source])
        if not shared:
            contrasts.append({"label": label, "skipped_reason": "no_shared_bystanders"})
            continue
        a_arr = np.array([a_per[p] for p in shared])
        b_arr = np.array([b_per[p] for p in shared])
        diffs = a_arr - b_arr
        mean_diff, lo_diff, hi_diff = _bootstrap_ci(diffs)
        contrasts.append(
            {
                "label": label,
                "cell_a": cell_a,
                "cell_b": cell_b,
                "n_shared_held_out_bystanders": len(shared),
                "delta_held_out_difference": float(mean_diff),
                "ci_low": float(lo_diff),
                "ci_high": float(hi_diff),
            }
        )

    # H5 secondary.
    h5_results = _compute_h5(per_cell, centroids_path, source=source)

    # H3 + H4 gate summaries.
    h3_failed_cells = [c for c, d in per_cell.items() if not d["h3_diagonal_pass"]]
    h4_failed_cells = [c for c, d in per_cell.items() if d["h4_constant_emission_hard_fail"]]
    h4_warning_cells = [
        c
        for c, d in per_cell.items()
        if not d["h4_constant_emission_pass"] and not d["h4_constant_emission_hard_fail"]
    ]

    # H1 interpretation (pre-committed per plan §6.2).
    h1a_pass = (
        h1a_null["observed_count"] >= 1
        and h1a_null["empirical_p_value_one_sided"] < HEADLINE_P_THRESHOLD
    )
    h1b_pass = (
        h1b_null["observed_count"] >= 1
        and h1b_null["empirical_p_value_one_sided"] < HEADLINE_P_THRESHOLD
    )
    if h1a_pass and h1b_pass:
        interpretation = (
            "Widening contrastive negatives reduces bystander leakage AND the "
            "reduction generalizes to bystanders never used as negatives."
        )
        interpretation_code = "h1a_pass_h1b_pass"
    elif h1a_pass and not h1b_pass:
        interpretation = (
            "The model stops emitting ` ※` on the specific personas it saw as "
            "negatives; no evidence of generalization to held-out bystanders. "
            "Confidence downgraded one notch."
        )
        interpretation_code = "h1a_pass_h1b_fail_per_persona_suppression_only"
    elif not h1a_pass and not h1b_pass:
        interpretation = (
            "The recipe-knob doesn't move on-policy bystander leakage by the "
            "pre-registered amount. Negative result."
        )
        interpretation_code = "both_fail"
    else:
        interpretation = (
            "Unusual pattern: H1a FAIL + H1b PASS. Investigate the unstratified "
            "denominator (held-out personas may move MORE than trained-negative "
            "personas, opposite of the artifact concern)."
        )
        interpretation_code = "h1a_fail_h1b_pass"

    summary = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_persona": source,
        "held_out_payload": held_out_payload,
        "per_cell": per_cell,
        "per_knob_h1b": per_knob_h1b,
        "per_knob_h1a": per_knob_h1a,
        "h1b_permutation_null": h1b_null,
        "h1a_permutation_null": h1a_null,
        "h1b_pass": bool(h1b_pass),
        "h1a_pass": bool(h1a_pass),
        "interpretation": interpretation,
        "interpretation_code": interpretation_code,
        "same_row_count_contrasts": contrasts,
        "h5_per_cell": h5_results,
        "h3_failed_cells": h3_failed_cells,
        "h4_hard_failed_cells": h4_failed_cells,
        "h4_warning_cells": h4_warning_cells,
        "h1_threshold_nats": HEADLINE_DELTA_RANGE_NATS,
        "h1_p_threshold": HEADLINE_P_THRESHOLD,
        "n_cells_analyzed": len(per_cell),
        "n_cells_expected": len(CELL_SPECS),
        "cell_slugs_with_eval": cell_slugs,
    }

    figures = _make_figures(summary, knob_axes, figures_dir)
    summary["figures"] = figures

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(
        "Wrote v5 analyze summary (%d/%d cells; H1a_pass=%s H1b_pass=%s; "
        "h3_fail=%d, h4_hard_fail=%d) -> %s",
        summary["n_cells_analyzed"],
        summary["n_cells_expected"],
        h1a_pass,
        h1b_pass,
        len(h3_failed_cells),
        len(h4_failed_cells),
        out_path,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--slab-root",
        type=Path,
        default=Path("eval_results/issue_448_v5"),
        help="Where per-cell marker_logprob.json files live.",
    )
    ap.add_argument(
        "--held-out-path",
        type=Path,
        default=Path("data/issue_448/held_out_bystanders.json"),
        help="Held-out bystanders artifact from Phase 0.",
    )
    ap.add_argument(
        "--centroids-path",
        type=Path,
        default=Path("eval_results/issue_448/centroids/centroids_layer20.pt"),
        help="Layer-20 centroid bundle for H5 secondary.",
    )
    ap.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/issue_448_v5"),
    )
    ap.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Override the default `<slab-root>/analyze_summary.json` path.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [phase=analyze] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    out_path = args.out_path or (args.slab_root / "analyze_summary.json")
    run_analysis(
        slab_root=args.slab_root,
        held_out_path=args.held_out_path,
        figures_dir=args.figures_dir,
        out_path=out_path,
        centroids_path=args.centroids_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
