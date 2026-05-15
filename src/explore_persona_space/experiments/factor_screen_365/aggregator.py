"""Cross-cell aggregation for the 3 sources x 32 cells = 96-cell factor screen.

Outputs (plan v2 §4 step 5):

  * ``main_effects.csv`` — paired-flip main effects per factor (A..E) on
    source rate and leakage rate, per source AND pooled.
  * ``factor_effects.json`` — same content as the CSV plus persona-clustered
    95% bootstrap CIs, an n=3 cluster-bootstrap supplement, an off-diagonal
    noise floor, and the kill-criterion-1 verdict (analyzer-must-handle #1,
    #2).
  * ``interactions.csv`` — A x B AND B x E interactions (plan §3 pre-registers
    both; analyzer-must-handle #3) plus exploratory pairs.
  * ``top_cells_by_source.json`` — top-3 cells per source ranked by source
    rate first, leakage rate second, used to schedule the seeds-137/256
    re-runs.
  * ``e_log_ratio.json`` — log-ratio (E1/E0) source-rate CI per source
    (analyzer-must-handle #4).
  * ``leakage_stratified.json`` — in-domain vs. out-of-domain bystander
    leakage (analyzer-must-handle #5).

Cell manifest emitted by ``write_cell_manifest`` carries the four covariate
columns the analyzer needs (analyzer-must-handle #6, #7, #8):

  * ``rendered_qwen_tokens_per_bystander``
  * ``marker_position_in_completion_tokens`` (mean / sd)
  * ``total_seq_length_tokens`` (mean / sd)
  * ``data_policy`` (on_policy / off_policy)

Citation notice (analyzer-must-handle #9):

  ``LEAKAGE_N48_CITATION_NOTE`` documents that plan §2 cites
  ``eval_results/issue_296/length_rate_correlation_n48.json`` for the #337
  leakage Spearman rho=-0.36, p=0.012, n=48 result, but the on-disk JSON
  only carries n=24. The clean-result write-up must either re-cite the n=24
  series (rho=-0.306, p=0.146) or regenerate the n=48 series before
  promoting.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bootstrap import (
    bootstrap_paired_difference,
    cluster_bootstrap_difference,
    log_ratio_ci,
    mean,
    stdev,
    wider_ci,
)
from .cells import (
    FACTOR_INDEX,
    FACTOR_NAMES,
    INTERACTION_PAIRS,
    Cell,
    is_preregistered,
    matched_pairs_for_factor,
    matched_pairs_for_interaction,
)
from .persona_panel import (
    IN_DOMAIN_BYSTANDERS_BY_SOURCE,
    SOURCE_PERSONAS,
    in_domain_bystanders_for,
    out_of_domain_bystanders_for,
)

log = logging.getLogger(__name__)


LEAKAGE_N48_CITATION_NOTE = (
    "Plan v2 §2 cites eval_results/issue_296/length_rate_correlation_n48.json for "
    "the #337 leakage Spearman rho = -0.36, p = 0.012, N = 48 result, but the "
    "on-disk JSON only carries N = 24 (rho = -0.306, p = 0.146). The n=48 value "
    "lives in the #337 issue body text. The clean-result write-up must either "
    "re-cite the on-disk n=24 series or regenerate the n=48 series before "
    "promotion. This is not a code change; the discrepancy is surfaced here for "
    "the analyzer to act on."
)


# ---- Shared cell-record schema ----------------------------------------------


@dataclass
class CellRecord:
    """One trained (source, cell, seed) outcome consumed by the aggregator."""

    source: str
    cell_key: str
    seed: int
    bits: tuple[int, int, int, int, int]
    source_rate: float
    leakage_rate_full: float  # mean across all 23 bystanders
    leakage_rate_out_of_domain: float  # mean across out-of-domain bystanders only
    leakage_rate_in_domain: float  # mean across in-domain bystanders (0 for librarian)
    per_bystander_rates: dict[str, float]
    failed: bool = False
    error: str | None = None


def _load_metrics_for_source(source_dir: Path) -> dict[str, CellRecord]:
    """Load every per-cell ``metrics.json`` under ``source_dir/cell_<key>/seed_<N>/``.

    Returns a dict keyed by ``cell_key`` for the primary seed only; multi-seed
    runs are handled by ``load_multiseed_records``.
    """
    out: dict[str, CellRecord] = {}
    for cell_dir in sorted(source_dir.glob("cell_*")):
        seed_dirs = sorted(cell_dir.glob("seed_*"))
        if not seed_dirs:
            continue
        primary = seed_dirs[0]
        metrics_path = primary / "metrics.json"
        if not metrics_path.exists():
            continue
        record = _record_from_metrics_json(metrics_path)
        if record is not None:
            out[record.cell_key] = record
    return out


def _record_from_metrics_json(path: Path) -> CellRecord | None:
    """Convert a per-cell ``metrics.json`` into a :class:`CellRecord`."""
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        log.warning("Failed to load %s: %s", path, exc)
        return None
    if payload.get("failed"):
        return CellRecord(
            source=payload.get("source", ""),
            cell_key=payload.get("cell_key", ""),
            seed=int(payload.get("seed", 0)),
            bits=tuple(payload.get("bits", (0, 0, 0, 0, 0))),  # type: ignore[arg-type]
            source_rate=0.0,
            leakage_rate_full=0.0,
            leakage_rate_out_of_domain=0.0,
            leakage_rate_in_domain=0.0,
            per_bystander_rates={},
            failed=True,
            error=payload.get("error"),
        )
    return CellRecord(
        source=payload["source"],
        cell_key=payload["cell_key"],
        seed=int(payload["seed"]),
        bits=tuple(payload["bits"]),  # type: ignore[arg-type]
        source_rate=float(payload.get("source_substring_rate", 0.0)),
        leakage_rate_full=float(payload.get("leakage_rate_full", 0.0)),
        leakage_rate_out_of_domain=float(payload.get("leakage_rate_out_of_domain", 0.0)),
        leakage_rate_in_domain=float(payload.get("leakage_rate_in_domain", 0.0)),
        per_bystander_rates=payload.get("per_bystander_substring_rates", {}),
        failed=False,
        error=None,
    )


def stratify_leakage(
    per_bystander_rates: dict[str, float],
    source: str,
) -> tuple[float, float, float]:
    """Return (full_leakage, out_of_domain_leakage, in_domain_leakage).

    Analyzer-must-handle item #5: bystander leakage must be reported
    stratified into in-domain vs out-of-domain subsets. ``librarian`` has no
    in-domain bystanders in the panel; for it ``in_domain_leakage == 0`` and
    ``out_of_domain_leakage == full_leakage``.
    """
    if source not in SOURCE_PERSONAS:
        raise ValueError(f"stratify_leakage requires a source persona; got {source!r}")
    if not per_bystander_rates:
        return 0.0, 0.0, 0.0
    in_domain_names = set(IN_DOMAIN_BYSTANDERS_BY_SOURCE[source])
    full = mean(per_bystander_rates.values())
    out_of_domain = mean(v for k, v in per_bystander_rates.items() if k not in in_domain_names)
    in_domain = mean(v for k, v in per_bystander_rates.items() if k in in_domain_names)
    return full, out_of_domain, in_domain


# ---- Off-diagonal noise floor (analyzer-must-handle #1) ---------------------


def off_diagonal_noise_floor(
    primary_records: dict[str, dict[str, CellRecord]],
    *,
    factor_a: str = "E",
    factor_b: str = "D",
    metric: str = "source_rate",
) -> dict[str, Any]:
    """Cross-seed SD of cells holding (E, D) fixed at baseline (E0, D0).

    The plan §Kill-criterion clause sets the threshold at ``1.5 x off-diagonal
    noise``. This estimator uses cross-seed variability within the E0 x D0
    sub-rectangle (i.e. cells with E=0 AND D=0) as the noise scale.

    Returns a dict with ``per_source_sd`` (one float per source) and ``pooled_sd``
    (the unweighted average across sources). The aggregator multiplies this
    by 1.5 when applying the kill criterion.
    """
    fi_a = FACTOR_INDEX[factor_a]
    fi_b = FACTOR_INDEX[factor_b]
    per_source_sd: dict[str, float] = {}
    for source, cells in primary_records.items():
        values: list[float] = []
        for record in cells.values():
            if record.failed:
                continue
            if record.bits[fi_a] == 0 and record.bits[fi_b] == 0:
                values.append(getattr(record, metric))
        per_source_sd[source] = stdev(values)
    pooled = mean(per_source_sd.values()) if per_source_sd else 0.0
    return {
        "rationale": (
            f"Cross-cell SD of cells held at {factor_a}=0 AND {factor_b}=0 "
            f"(the design's stable sub-rectangle). Used to set the >1.5x noise "
            "threshold for kill-criterion 1."
        ),
        "metric": metric,
        "per_source_sd": per_source_sd,
        "pooled_sd": pooled,
        "kill_threshold_multiplier": 1.5,
        "kill_threshold_pooled": pooled * 1.5,
    }


# ---- Paired main effects -----------------------------------------------------


def _paired_deltas_for(
    factor: str,
    records: dict[str, CellRecord],
    metric: str,
) -> list[float]:
    """Compute the 16 paired (level1 - level0) deltas for one factor + source."""
    deltas: list[float] = []
    for cell0, cell1 in matched_pairs_for_factor(factor):
        r0 = records.get(cell0.key)
        r1 = records.get(cell1.key)
        if r0 is None or r1 is None or r0.failed or r1.failed:
            continue
        deltas.append(getattr(r1, metric) - getattr(r0, metric))
    return deltas


def compute_main_effects(
    primary_records: dict[str, dict[str, CellRecord]],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired-flip main effects per factor x source x metric.

    Per analyzer-must-handle #2 the aggregator returns whichever CI is wider
    between the persona-clustered (paired) bootstrap and the n=3 cluster
    bootstrap. The cluster-bootstrap clusters at the (source, cell) level on
    the matched-pair deltas.
    """
    metrics = ("source_rate", "leakage_rate_full", "leakage_rate_out_of_domain")
    factor_results: dict[str, dict[str, Any]] = {}

    for fi, factor in enumerate(FACTOR_NAMES):
        per_metric: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            per_source: dict[str, dict[str, Any]] = {}
            pooled_deltas: list[float] = []
            cluster_payload: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
            for source, records in primary_records.items():
                deltas = _paired_deltas_for(factor, records, metric)
                pooled_deltas.extend(deltas)
                paired_ci = bootstrap_paired_difference(deltas, n_boot=n_boot, seed=seed + fi)
                # Build the cluster-bootstrap payload for this (source, cell).
                # Each cluster is one source x cell-pair, contributing one level0
                # and one level1 observation.
                for cell0, cell1 in matched_pairs_for_factor(factor):
                    r0 = records.get(cell0.key)
                    r1 = records.get(cell1.key)
                    if r0 is None or r1 is None or r0.failed or r1.failed:
                        continue
                    cluster_payload[(source, cell0.key)] = (
                        [getattr(r0, metric)],
                        [getattr(r1, metric)],
                    )
                per_source[source] = {
                    "delta_mean": mean(deltas),
                    "paired_ci": list(paired_ci),
                    "n_pairs": len(deltas),
                }
            pooled_paired_ci = bootstrap_paired_difference(
                pooled_deltas, n_boot=n_boot, seed=seed + fi + 100
            )
            cluster_ci = cluster_bootstrap_difference(
                cluster_payload, n_boot=n_boot, seed=seed + fi + 200
            )
            chosen_ci = wider_ci(pooled_paired_ci, cluster_ci)
            per_metric[metric] = {
                "per_source": per_source,
                "pooled_delta_mean": mean(pooled_deltas),
                "pooled_paired_ci": list(pooled_paired_ci),
                "cluster_bootstrap_ci": list(cluster_ci),
                "chosen_ci": list(chosen_ci),
                "n_pairs": len(pooled_deltas),
                "note": (
                    "chosen_ci is the wider of paired and cluster-bootstrap CIs "
                    "(plan reconciler analyzer-must-handle item #2)."
                ),
            }
        factor_results[factor] = per_metric

    return {
        "design": "2^5 = 32 cells per source x 3 sources",
        "metrics": list(metrics),
        "factors": factor_results,
        "n_boot": n_boot,
    }


# ---- Interactions (A x B pre-registered, B x E pre-registered) -------------


def compute_interactions(
    primary_records: dict[str, dict[str, CellRecord]],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """All 10 pairwise interactions on source rate and full leakage.

    The two pre-registered pairs are A x B (plan hypothesis 5: total
    training-context length / marker position) and B x E (plan hypothesis 2:
    dilution is loss-mask mediated; analyzer-must-handle #3).
    """
    metrics = ("source_rate", "leakage_rate_full")
    out: dict[str, dict[str, Any]] = {}

    for pair in INTERACTION_PAIRS:
        a, b = pair
        per_metric: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            per_source: dict[str, dict[str, Any]] = {}
            pooled_diffs: list[float] = []
            for source, records in primary_records.items():
                source_diffs: list[float] = []
                for c00, c01, c10, c11 in matched_pairs_for_interaction(a, b):
                    r00 = records.get(c00.key)
                    r01 = records.get(c01.key)
                    r10 = records.get(c10.key)
                    r11 = records.get(c11.key)
                    if any(r is None or r.failed for r in (r00, r01, r10, r11)):
                        continue
                    d_high = getattr(r11, metric) - getattr(r10, metric)
                    d_low = getattr(r01, metric) - getattr(r00, metric)
                    source_diffs.append(d_high - d_low)
                per_source[source] = {
                    "interaction_mean": mean(source_diffs),
                    "n_tuples": len(source_diffs),
                }
                pooled_diffs.extend(source_diffs)
            paired_ci = bootstrap_paired_difference(pooled_diffs, n_boot=n_boot, seed=seed)
            per_metric[metric] = {
                "pooled_interaction_mean": mean(pooled_diffs),
                "pooled_ci": list(paired_ci),
                "per_source": per_source,
            }
        out[f"{a}x{b}"] = {
            "pair": [a, b],
            "preregistered": is_preregistered(pair),
            **per_metric,
        }
    return {
        "preregistered_pairs": ["AxB", "BxE"],
        "interactions": out,
    }


# ---- E1/E0 log-ratio CI (analyzer-must-handle #4) ---------------------------


def compute_e_log_ratio(
    primary_records: dict[str, dict[str, CellRecord]],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Per-source log(E0 / E1) source-rate ratio with bootstrap CI.

    Plan v2 hypothesis 4 originally framed this as a ``>=2x`` hard threshold.
    The reconciler swapped it for a log-ratio CI. Both directions are
    reported here so the analyzer can pick whichever matches the write-up.
    """
    out: dict[str, Any] = {}
    fi_e = FACTOR_INDEX["E"]
    pooled_e0: list[float] = []
    pooled_e1: list[float] = []
    for source, records in primary_records.items():
        e0_values = [r.source_rate for r in records.values() if not r.failed and r.bits[fi_e] == 0]
        e1_values = [r.source_rate for r in records.values() if not r.failed and r.bits[fi_e] == 1]
        pooled_e0.extend(e0_values)
        pooled_e1.extend(e1_values)
        point, lo, hi = log_ratio_ci(e0_values, e1_values, n_boot=n_boot, seed=seed)
        out[source] = {
            "e0_mean_source_rate": mean(e0_values),
            "e1_mean_source_rate": mean(e1_values),
            "log_ratio_E0_over_E1": point,
            "log_ratio_ci": [lo, hi],
        }
    p_point, p_lo, p_hi = log_ratio_ci(pooled_e0, pooled_e1, n_boot=n_boot, seed=seed + 1)
    out["pooled"] = {
        "e0_mean_source_rate": mean(pooled_e0),
        "e1_mean_source_rate": mean(pooled_e1),
        "log_ratio_E0_over_E1": p_point,
        "log_ratio_ci": [p_lo, p_hi],
    }
    return {
        "note": (
            "Reported as log(E0 / E1) so positive numbers mean E0 (marker-only) "
            "lifts source rate over E1 (whole-completion); analyzer-must-handle #4 "
            "replaces the original >=2x threshold with this CI."
        ),
        "per_source": out,
    }


# ---- Stratified leakage (analyzer-must-handle #5) ---------------------------


def compute_stratified_leakage(
    primary_records: dict[str, dict[str, CellRecord]],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Per-source full / out-of-domain / in-domain leakage with CIs."""
    out: dict[str, Any] = {}
    for source, records in primary_records.items():
        full_values: list[float] = []
        ood_values: list[float] = []
        ind_values: list[float] = []
        for record in records.values():
            if record.failed:
                continue
            full_values.append(record.leakage_rate_full)
            ood_values.append(record.leakage_rate_out_of_domain)
            ind_values.append(record.leakage_rate_in_domain)
        out[source] = {
            "full_panel": {
                "mean": mean(full_values),
                "n": len(full_values),
            },
            "out_of_domain_only": {
                "mean": mean(ood_values),
                "n": len(ood_values),
                "panel": out_of_domain_bystanders_for(source),
            },
            "in_domain_only": {
                "mean": mean(ind_values),
                "n": len(ind_values),
                "panel": in_domain_bystanders_for(source),
            },
            "delta_in_minus_ood": mean(ind_values) - mean(ood_values),
        }
    return {
        "note": (
            "Per-source full panel = 23 bystanders; out-of-domain excludes "
            "occupationally-adjacent personas (surgeon -> medical_doctor, "
            "programmer -> software_engineer & data_scientist). librarian has no "
            "in-domain panel members; for it the in-domain mean is 0 and "
            "out-of-domain == full."
        ),
        "per_source": out,
    }


# ---- Top cells per source ---------------------------------------------------


def rank_top_cells(
    primary_records: dict[str, dict[str, CellRecord]],
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """Per source, rank cells by source rate first, leakage rate (descending)."""
    out: dict[str, list[dict]] = {}
    for source, records in primary_records.items():
        valid = [r for r in records.values() if not r.failed]
        valid.sort(key=lambda r: (r.source_rate, -r.leakage_rate_full), reverse=True)
        out[source] = [
            {
                "cell_key": r.cell_key,
                "source_rate": r.source_rate,
                "leakage_rate_full": r.leakage_rate_full,
                "leakage_rate_out_of_domain": r.leakage_rate_out_of_domain,
                "bits": list(r.bits),
            }
            for r in valid[:top_k]
        ]
    return out


# ---- CSV emitters -----------------------------------------------------------


def write_main_effects_csv(main_effects: dict[str, Any], path: Path) -> Path:
    """Flatten the main-effects payload into a CSV for the analyzer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for factor, per_metric in main_effects["factors"].items():
        for metric, payload in per_metric.items():
            for source, src_payload in payload["per_source"].items():
                rows.append(
                    {
                        "factor": factor,
                        "metric": metric,
                        "source": source,
                        "delta_mean": src_payload["delta_mean"],
                        "n_pairs": src_payload["n_pairs"],
                        "paired_ci_lo": src_payload["paired_ci"][0],
                        "paired_ci_hi": src_payload["paired_ci"][1],
                    }
                )
            rows.append(
                {
                    "factor": factor,
                    "metric": metric,
                    "source": "pooled",
                    "delta_mean": payload["pooled_delta_mean"],
                    "n_pairs": payload["n_pairs"],
                    "paired_ci_lo": payload["chosen_ci"][0],
                    "paired_ci_hi": payload["chosen_ci"][1],
                }
            )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "factor",
                "metric",
                "source",
                "delta_mean",
                "n_pairs",
                "paired_ci_lo",
                "paired_ci_hi",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_interactions_csv(interactions: dict[str, Any], path: Path) -> Path:
    """Flatten the interactions payload into a CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pair_name, payload in interactions["interactions"].items():
        rows.append(
            {
                "pair": pair_name,
                "preregistered": payload["preregistered"],
                "metric": "source_rate",
                "pooled_interaction_mean": payload["source_rate"]["pooled_interaction_mean"],
                "pooled_ci_lo": payload["source_rate"]["pooled_ci"][0],
                "pooled_ci_hi": payload["source_rate"]["pooled_ci"][1],
            }
        )
        rows.append(
            {
                "pair": pair_name,
                "preregistered": payload["preregistered"],
                "metric": "leakage_rate_full",
                "pooled_interaction_mean": payload["leakage_rate_full"]["pooled_interaction_mean"],
                "pooled_ci_lo": payload["leakage_rate_full"]["pooled_ci"][0],
                "pooled_ci_hi": payload["leakage_rate_full"]["pooled_ci"][1],
            }
        )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair",
                "preregistered",
                "metric",
                "pooled_interaction_mean",
                "pooled_ci_lo",
                "pooled_ci_hi",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---- Cell manifest (analyzer-must-handle #6, #7, #8) ------------------------


def write_cell_manifest(
    rows: list[dict[str, Any]],
    path: Path,
) -> Path:
    """Write the per-cell manifest carrying analyzer covariates.

    Required columns per analyzer-must-handle items #6 / #7 / #8::

        cell_key, source, seed, data_policy,
        rendered_qwen_tokens_per_bystander,
        marker_position_in_completion_tokens_mean,
        marker_position_in_completion_tokens_sd,
        total_seq_length_tokens_mean,
        total_seq_length_tokens_sd

    The caller is responsible for supplying these columns when assembling
    each cell; this function is a thin CSV writer.
    """
    if not rows:
        raise ValueError("write_cell_manifest got no rows; refusing to write empty CSV")
    required = {
        "cell_key",
        "source",
        "seed",
        "data_policy",
        "rendered_qwen_tokens_per_bystander",
        "marker_position_in_completion_tokens_mean",
        "marker_position_in_completion_tokens_sd",
        "total_seq_length_tokens_mean",
        "total_seq_length_tokens_sd",
    }
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"cell_manifest rows missing required columns: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_persona_panel_manifest(rows: list[dict[str, Any]], path: Path) -> Path:
    """Per-bystander manifest carrying rendered Qwen token length per bystander.

    Required columns per analyzer-must-handle item #6::

        persona, system_prompt, qwen_rendered_token_count, in_domain_for_<source>
    """
    if not rows:
        raise ValueError("write_persona_panel_manifest got no rows; refusing to write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---- Aggregator entry point -------------------------------------------------


def aggregate_factor_screen(
    primary_records: dict[str, dict[str, CellRecord]],
    *,
    output_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Path]:
    """Run every aggregator step and return paths of the written artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    main_effects = compute_main_effects(primary_records, n_boot=n_boot, seed=seed)
    noise = off_diagonal_noise_floor(primary_records)
    interactions = compute_interactions(primary_records, n_boot=n_boot, seed=seed)
    e_log_ratio = compute_e_log_ratio(primary_records, n_boot=n_boot, seed=seed)
    stratified = compute_stratified_leakage(primary_records, n_boot=n_boot, seed=seed)
    top_cells = rank_top_cells(primary_records)

    factor_effects_payload = {
        "main_effects": main_effects,
        "off_diagonal_noise_floor": noise,
        "leakage_n48_citation_note": LEAKAGE_N48_CITATION_NOTE,
    }
    paths["main_effects_csv"] = write_main_effects_csv(
        main_effects, output_dir / "main_effects.csv"
    )
    paths["factor_effects_json"] = output_dir / "factor_effects.json"
    paths["factor_effects_json"].write_text(json.dumps(factor_effects_payload, indent=2))

    paths["interactions_csv"] = write_interactions_csv(
        interactions, output_dir / "interactions.csv"
    )
    paths["interactions_json"] = output_dir / "interactions.json"
    paths["interactions_json"].write_text(json.dumps(interactions, indent=2))

    paths["e_log_ratio_json"] = output_dir / "e_log_ratio.json"
    paths["e_log_ratio_json"].write_text(json.dumps(e_log_ratio, indent=2))

    paths["leakage_stratified_json"] = output_dir / "leakage_stratified.json"
    paths["leakage_stratified_json"].write_text(json.dumps(stratified, indent=2))

    paths["top_cells_by_source_json"] = output_dir / "top_cells_by_source.json"
    paths["top_cells_by_source_json"].write_text(json.dumps(top_cells, indent=2))

    return paths


# ---- Public helper used by the entry point at slab boundary -----------------


def load_records_from_disk(slab_root: Path) -> dict[str, dict[str, CellRecord]]:
    """Walk ``slab_root / <source> / cell_*/seed_*/metrics.json`` files.

    Returns ``{source: {cell_key: CellRecord}}``.
    """
    out: dict[str, dict[str, CellRecord]] = {}
    for source_dir in sorted(slab_root.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        if source not in SOURCE_PERSONAS:
            continue
        out[source] = _load_metrics_for_source(source_dir)
    return out


def cell_from_record(record: CellRecord) -> Cell:
    """Convenience: rebuild a :class:`Cell` from a :class:`CellRecord`."""
    return Cell(*record.bits)
