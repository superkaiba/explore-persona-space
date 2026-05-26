"""Cross-cell aggregation for the task #391 sycophancy implantation screen.

Reads per-cell per-panel-persona JSONs:

    slab_root /
        cell_<key> /
            source_<src> /
                seed_<N> /
                    sycophancy_eval_<panel_persona>.json
                    metrics.json

The eval JSONs follow the schema of ``external/.../run_sycophancy_eval.py``:

    {
      "model": str,
      "aggregate": {
        "mean_drift": float,
        "mean_sycophancy_index": float,
        "per_turn_p_user_pct": list[float],
      },
      "configs": {
        "<config_id>": {
          "summary": {
            "per_turn_p_user_pct": [...],
            "drift": float,
            "turn_of_flip": float | None,
            "sycophancy_index": float,
          },
          ...
        },
        ...
      },
    }

Plus the base-model T0 zero-shot eval at:

    slab_root / base_qwen_zero_shot / source_<src> / seed_<N> / sycophancy_eval_<panel_persona>.json

Outputs (plan §6 "Primary metric and selectivity Δ"):

  * ``cell_persona_table.csv`` — long-form ``(cell_key, source, panel_persona,
    mean_sycophancy_index, mean_drift)`` table. The one-row-per-(cell,
    persona) source of truth.
  * ``per_factor_selectivity.json`` — for each of the 3 swept factor flips
    (A, C, D), the per-source Δsource, Δbystander_mean, Δbystander_median,
    selectivity Δ (mean-aggregator) + selectivity Δ (median-aggregator),
    plus 95% widest-of-three bootstrap CIs (per-pair percentile + source-
    cluster bootstrap).
  * ``per_factor_selectivity.csv`` — the same table flattened to CSV.
  * ``baseline_summary.json`` — per-panel-persona base-model sycophancy_index
    (T0) and per-source baseline-vs-trained decomposition.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from explore_persona_space.experiments.factor_screen_365.cells import Cell
from explore_persona_space.experiments.factor_screen_365.persona_panel import (
    SOURCE_PERSONAS,
)

log = logging.getLogger(__name__)


# The 3 swept factor flips against the anchor cell `10011`.
# Each entry: (factor_label, anchor_key, flipped_key, plain_english).
FACTOR_FLIPS: list[tuple[str, str, str, str]] = [
    ("A", "10011", "00011", "Short system prompt (vs long)"),
    ("C", "10011", "10111", "Neutral framing (vs persona)"),
    ("D", "10011", "10001", "Base-Qwen training data (vs Claude)"),
]


# ---- Loading ---------------------------------------------------------------


def _load_eval_json(p: Path) -> dict | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Failed to load %s: %s", p, exc)
        return None


def _persona_from_filename(p: Path) -> str | None:
    """Parse ``sycophancy_eval_<persona>.json`` -> ``persona``."""
    name = p.name
    if not name.startswith("sycophancy_eval_") or not name.endswith(".json"):
        return None
    return name[len("sycophancy_eval_") : -len(".json")]


def load_cell_persona_records(slab_root: Path) -> list[dict[str, Any]]:
    """Walk the slab and emit one row per (cell, source, seed, panel_persona)."""
    records: list[dict[str, Any]] = []
    for cell_dir in sorted(slab_root.glob("cell_*")):
        if not cell_dir.is_dir():
            continue
        cell_key = cell_dir.name[len("cell_") :]
        for source_dir in sorted(cell_dir.glob("source_*")):
            source = source_dir.name[len("source_") :]
            for seed_dir in sorted(source_dir.glob("seed_*")):
                try:
                    seed = int(seed_dir.name[len("seed_") :])
                except ValueError:
                    continue
                for eval_path in sorted(seed_dir.glob("sycophancy_eval_*.json")):
                    persona = _persona_from_filename(eval_path)
                    if persona is None:
                        continue
                    payload = _load_eval_json(eval_path)
                    if payload is None:
                        continue
                    agg = payload.get("aggregate") or {}
                    records.append(
                        {
                            "cell_key": cell_key,
                            "source": source,
                            "seed": seed,
                            "panel_persona": persona,
                            "mean_sycophancy_index": float(agg.get("mean_sycophancy_index", 0.0)),
                            "mean_drift": float(agg.get("mean_drift", 0.0)),
                            "per_turn_p_user_pct": list(agg.get("per_turn_p_user_pct", [])),
                            "eval_json_path": str(eval_path),
                        }
                    )
    return records


def load_base_records(slab_root: Path) -> list[dict[str, Any]]:
    """Load base-model zero-shot records from ``base_qwen_zero_shot/`` subtree."""
    base_root = slab_root / "base_qwen_zero_shot"
    records: list[dict[str, Any]] = []
    if not base_root.is_dir():
        return records
    for source_dir in sorted(base_root.glob("source_*")):
        source = source_dir.name[len("source_") :]
        for seed_dir in sorted(source_dir.glob("seed_*")):
            try:
                seed = int(seed_dir.name[len("seed_") :])
            except ValueError:
                continue
            for eval_path in sorted(seed_dir.glob("sycophancy_eval_*.json")):
                persona = _persona_from_filename(eval_path)
                if persona is None:
                    continue
                payload = _load_eval_json(eval_path)
                if payload is None:
                    continue
                agg = payload.get("aggregate") or {}
                records.append(
                    {
                        "cell_key": "base_qwen_zero_shot",
                        "source": source,
                        "seed": seed,
                        "panel_persona": persona,
                        "mean_sycophancy_index": float(agg.get("mean_sycophancy_index", 0.0)),
                        "mean_drift": float(agg.get("mean_drift", 0.0)),
                        "per_turn_p_user_pct": list(agg.get("per_turn_p_user_pct", [])),
                        "eval_json_path": str(eval_path),
                    }
                )
    return records


# ---- Bootstrap CIs ---------------------------------------------------------


def _percentile_boot_ci(
    values: list[float],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    boots = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _cluster_boot_ci(
    by_source: dict[str, list[float]],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    sources = list(by_source.keys())
    if not sources or not any(by_source.values()):
        return 0.0, 0.0
    n_clusters = len(sources)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        pool: list[float] = []
        for j in sampled:
            pool.extend(by_source[sources[j]])
        boots[i] = float(np.mean(pool)) if pool else np.nan
    boots = boots[~np.isnan(boots)]
    if boots.size == 0:
        return 0.0, 0.0
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _widest_ci(
    flat_values: list[float],
    by_source: dict[str, list[float]],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) where (lo, hi) is the widest of percentile + cluster."""
    arr = np.asarray(flat_values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    p_lo, p_hi = _percentile_boot_ci(flat_values, n_boot=n_boot, rng=rng)
    c_lo, c_hi = _cluster_boot_ci(by_source, n_boot=n_boot, rng=rng)
    lo = min(p_lo, c_lo)
    hi = max(p_hi, c_hi)
    return mean, lo, hi


# ---- Per-cell summaries ----------------------------------------------------


def _by_cell_source_persona(records: Iterable[dict]) -> dict[tuple[str, str, str], dict]:
    """{(cell_key, source, persona): record}, keeping the highest-seed record per key."""
    out: dict[tuple[str, str, str], dict] = {}
    for r in records:
        key = (r["cell_key"], r["source"], r["panel_persona"])
        prev = out.get(key)
        if prev is None or r["seed"] > prev["seed"]:
            out[key] = r
    return out


def _source_sycophancy(
    by_csp: dict[tuple[str, str, str], dict],
    *,
    cell_key: str,
    source: str,
) -> float | None:
    rec = by_csp.get((cell_key, source, source))
    if rec is None:
        return None
    return float(rec["mean_sycophancy_index"])


def _bystander_sycophancy_list(
    by_csp: dict[tuple[str, str, str], dict],
    *,
    cell_key: str,
    source: str,
) -> list[float]:
    """List of bystander sycophancy_index values for 23 non-source personas in this cell."""
    vals: list[float] = []
    for (ck, src, persona), rec in by_csp.items():
        if ck != cell_key or src != source or persona == source:
            continue
        vals.append(float(rec["mean_sycophancy_index"]))
    return vals


def _bystander_mean(by_csp, cell_key, source) -> float | None:
    vals = _bystander_sycophancy_list(by_csp, cell_key=cell_key, source=source)
    return float(np.mean(vals)) if vals else None


def _bystander_median(by_csp, cell_key, source) -> float | None:
    vals = _bystander_sycophancy_list(by_csp, cell_key=cell_key, source=source)
    return float(np.median(vals)) if vals else None


# ---- Per-factor selectivity ------------------------------------------------


def compute_per_factor_selectivity(
    records: Iterable[dict],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Compute selectivity Δ per swept factor (A, C, D).

    For each factor flip ``(level=0_cell, level=1_cell)`` and each of the 3
    source personas:

      * Δsource = source_sycophancy(level=1) - source_sycophancy(level=0)
      * Δbystander_mean = bystander_mean(level=1) - bystander_mean(level=0)
      * Δbystander_median = bystander_median(level=1) - bystander_median(level=0)
      * selectivity_mean = Δsource - Δbystander_mean
      * selectivity_median = Δsource - Δbystander_median

    The headline pooled Δ is the **mean of per-source Δs** (cluster mean),
    NOT Δ-on-pooled-rates — plan §6 pre-specification.

    Note: the plan's anchor convention is `level=1 = anchor (10011)` and
    `level=0 = flipped`. We map ``anchor_key`` -> level 1 and ``flipped_key``
    -> level 0 below so the sign convention matches #383's "Δ source rate"
    semantics for the A-flip ("Δ when A goes from short to long").
    """
    by_csp = _by_cell_source_persona(records)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {
        "factor_flips": {},
        "n_boot": n_boot,
        "n_sources": len(SOURCE_PERSONAS),
        "sources": list(SOURCE_PERSONAS),
        "pooling": "mean_of_per_source_deltas",
        "bystander_primary_aggregator": "mean",
        "bystander_sensitivity_aggregator": "median",
    }

    for factor, anchor_key, flipped_key, plain_english in FACTOR_FLIPS:
        # Sign convention: Δsource = anchor - flipped (so positive means
        # "the anchor lifts source more than the flipped variant").
        per_source: list[dict[str, float | None]] = []
        d_source_by_src: dict[str, list[float]] = {}
        d_bystander_mean_by_src: dict[str, list[float]] = {}
        d_bystander_median_by_src: dict[str, list[float]] = {}
        sel_mean_by_src: dict[str, list[float]] = {}
        sel_median_by_src: dict[str, list[float]] = {}
        d_sources_flat: list[float] = []
        d_bystander_mean_flat: list[float] = []
        d_bystander_median_flat: list[float] = []
        sel_mean_flat: list[float] = []
        sel_median_flat: list[float] = []

        for source in SOURCE_PERSONAS:
            src_anchor = _source_sycophancy(by_csp, cell_key=anchor_key, source=source)
            src_flipped = _source_sycophancy(by_csp, cell_key=flipped_key, source=source)
            bys_mean_anchor = _bystander_mean(by_csp, anchor_key, source)
            bys_mean_flipped = _bystander_mean(by_csp, flipped_key, source)
            bys_med_anchor = _bystander_median(by_csp, anchor_key, source)
            bys_med_flipped = _bystander_median(by_csp, flipped_key, source)

            entry: dict[str, float | None] = {
                "source": source,
                "src_anchor": src_anchor,
                "src_flipped": src_flipped,
                "bys_mean_anchor": bys_mean_anchor,
                "bys_mean_flipped": bys_mean_flipped,
                "bys_median_anchor": bys_med_anchor,
                "bys_median_flipped": bys_med_flipped,
            }

            if (
                src_anchor is not None
                and src_flipped is not None
                and bys_mean_anchor is not None
                and bys_mean_flipped is not None
            ):
                d_src = src_anchor - src_flipped
                d_bys_mean = bys_mean_anchor - bys_mean_flipped
                sel_mean = d_src - d_bys_mean
                entry["d_source"] = d_src
                entry["d_bystander_mean"] = d_bys_mean
                entry["selectivity_mean"] = sel_mean
                d_source_by_src.setdefault(source, []).append(d_src)
                d_bystander_mean_by_src.setdefault(source, []).append(d_bys_mean)
                sel_mean_by_src.setdefault(source, []).append(sel_mean)
                d_sources_flat.append(d_src)
                d_bystander_mean_flat.append(d_bys_mean)
                sel_mean_flat.append(sel_mean)
            if (
                src_anchor is not None
                and src_flipped is not None
                and bys_med_anchor is not None
                and bys_med_flipped is not None
            ):
                d_src = src_anchor - src_flipped
                d_bys_med = bys_med_anchor - bys_med_flipped
                sel_med = d_src - d_bys_med
                entry["d_bystander_median"] = d_bys_med
                entry["selectivity_median"] = sel_med
                d_bystander_median_by_src.setdefault(source, []).append(d_bys_med)
                sel_median_by_src.setdefault(source, []).append(sel_med)
                d_bystander_median_flat.append(d_bys_med)
                sel_median_flat.append(sel_med)
            per_source.append(entry)

        d_src_mean, d_src_lo, d_src_hi = _widest_ci(
            d_sources_flat, d_source_by_src, n_boot=n_boot, rng=rng
        )
        d_bys_mean_m, d_bys_mean_lo, d_bys_mean_hi = _widest_ci(
            d_bystander_mean_flat, d_bystander_mean_by_src, n_boot=n_boot, rng=rng
        )
        d_bys_med_m, d_bys_med_lo, d_bys_med_hi = _widest_ci(
            d_bystander_median_flat, d_bystander_median_by_src, n_boot=n_boot, rng=rng
        )
        sel_mean_m, sel_mean_lo, sel_mean_hi = _widest_ci(
            sel_mean_flat, sel_mean_by_src, n_boot=n_boot, rng=rng
        )
        sel_med_m, sel_med_lo, sel_med_hi = _widest_ci(
            sel_median_flat, sel_median_by_src, n_boot=n_boot, rng=rng
        )

        out["factor_flips"][factor] = {
            "factor": factor,
            "anchor_key": anchor_key,
            "flipped_key": flipped_key,
            "plain_english": plain_english,
            "per_source": per_source,
            "n_pairs": len(d_sources_flat),
            "d_source_mean": d_src_mean,
            "d_source_ci": [d_src_lo, d_src_hi],
            "d_bystander_mean_mean": d_bys_mean_m,
            "d_bystander_mean_ci": [d_bys_mean_lo, d_bys_mean_hi],
            "d_bystander_median_mean": d_bys_med_m,
            "d_bystander_median_ci": [d_bys_med_lo, d_bys_med_hi],
            "selectivity_mean_mean": sel_mean_m,
            "selectivity_mean_ci": [sel_mean_lo, sel_mean_hi],
            "selectivity_median_mean": sel_med_m,
            "selectivity_median_ci": [sel_med_lo, sel_med_hi],
        }

    return out


# ---- Per-cell-persona table writer -----------------------------------------


def write_cell_persona_table_csv(records: Iterable[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell_key",
        "source",
        "seed",
        "panel_persona",
        "mean_sycophancy_index",
        "mean_drift",
        "eval_json_path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "cell_key": r["cell_key"],
                    "source": r["source"],
                    "seed": r["seed"],
                    "panel_persona": r["panel_persona"],
                    "mean_sycophancy_index": r["mean_sycophancy_index"],
                    "mean_drift": r["mean_drift"],
                    "eval_json_path": r["eval_json_path"],
                }
            )
    return path


def write_per_factor_selectivity_csv(payload: dict, path: Path) -> Path:
    """Flatten the per-factor JSON to CSV (one row per (factor, source))."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "factor",
        "anchor_key",
        "flipped_key",
        "plain_english",
        "source",
        "src_anchor",
        "src_flipped",
        "d_source",
        "bys_mean_anchor",
        "bys_mean_flipped",
        "d_bystander_mean",
        "selectivity_mean",
        "bys_median_anchor",
        "bys_median_flipped",
        "d_bystander_median",
        "selectivity_median",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for factor_payload in payload["factor_flips"].values():
            for entry in factor_payload["per_source"]:
                row = {
                    "factor": factor_payload["factor"],
                    "anchor_key": factor_payload["anchor_key"],
                    "flipped_key": factor_payload["flipped_key"],
                    "plain_english": factor_payload["plain_english"],
                    **entry,
                }
                writer.writerow(row)
    return path


def write_baseline_summary(
    base_records: Iterable[dict],
    path: Path,
) -> Path:
    """Per-(source, persona) base-model sycophancy_index summary."""
    by_sp: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in base_records:
        by_sp[(r["source"], r["panel_persona"])].append(float(r["mean_sycophancy_index"]))
    payload: dict[str, Any] = {
        "n_records": sum(len(v) for v in by_sp.values()),
        "rows": [
            {
                "source": src,
                "panel_persona": persona,
                "mean_sycophancy_index": float(np.mean(vals)),
                "n_seeds": len(vals),
            }
            for (src, persona), vals in sorted(by_sp.items())
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---- Public entry ----------------------------------------------------------


def aggregate_sycophancy_slab(
    *,
    slab_root: Path,
    output_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Path]:
    """Top-level aggregator. Returns a dict of artifact-name -> path."""
    slab_root = Path(slab_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_cell_persona_records(slab_root)
    if not records:
        raise SystemExit(
            f"No sycophancy_eval_*.json files found under {slab_root}; nothing to aggregate."
        )

    cell_table_path = write_cell_persona_table_csv(records, output_dir / "cell_persona_table.csv")

    per_factor = compute_per_factor_selectivity(records, n_boot=n_boot, seed=seed)
    per_factor_json_path = output_dir / "per_factor_selectivity.json"
    per_factor_json_path.write_text(json.dumps(per_factor, indent=2))
    per_factor_csv_path = write_per_factor_selectivity_csv(
        per_factor, output_dir / "per_factor_selectivity.csv"
    )

    base_records = load_base_records(slab_root)
    baseline_path = write_baseline_summary(base_records, output_dir / "baseline_summary.json")

    return {
        "cell_persona_table_csv": cell_table_path,
        "per_factor_selectivity_json": per_factor_json_path,
        "per_factor_selectivity_csv": per_factor_csv_path,
        "baseline_summary_json": baseline_path,
    }


def cell_from_key(key: str) -> Cell:
    """Re-export for downstream tooling."""
    return Cell.from_key(key)
