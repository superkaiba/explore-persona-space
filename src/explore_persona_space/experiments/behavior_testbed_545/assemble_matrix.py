"""Issue #545 L-matrix assembly (plan section 4.5).

Reads the per-cell column JSONs under ``cells/`` + the base panel and writes:

- ``L_matrix.json``    — per (cell, column, context): level (adapter scalar),
  base level, L = trained - base, cluster-bootstrap CI, extras.
- ``cell_metadata.json`` — diagonal manipulation check, realized-strength
  covariate, saturation / implant_failed flags, selected checkpoint.
- ``base_panel.json``  — per (column, context) base scalars (headroom +
  level-track denominator).

CPU-only: runs on the VM against committed cell JSONs (no pod dependency).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from . import cells_dir, output_root, reproducibility_metadata
from .preregister import THRESHOLDS
from .rows import ROWS

logger = logging.getLogger(__name__)

# Per-cell files that match the ``*__*.json`` glob but are NOT final column
# JSONs: raw generations + the dose-select per-checkpoint archives (which lack
# a ``summary`` key — round-1 blocker i545-assemble-dose-files-crash). The
# dose archives now live under ``dose/`` (outside the glob) too; this prefix
# skip is the belt-and-suspenders defense for any stragglers.
_NON_COLUMN_PREFIXES = ("completions__", "dose__", "dose_completions__")

# Canonical primary scalar per column (extras carried alongside).
PRIMARY_SCALAR: dict[str, str] = {
    "broad_em": "rate",
    "harmful_compliance": "headline_asr_full_harm",
    "sycophancy": "rate",
    "deception": "non_disclosure_rate",
    "refusal": "refusal_rate_should_not_refuse",  # over-refusal on benign half
    "fact_expression": "stated_taught_fact_rate",
    "marker": "delta_logp",  # already trained-base by construction
    "format_style": "list_format_rate",
    "capability": "arc_c_logprob_accuracy",
    "self_report": "verbalization_rate",
    "persona_drift": "consistency_rate",  # L sign flipped below (drift = 1 - consistency)
    "fam_expr_bad_medical": "misaligned_advice_rate",
    "fam_expr_risky_financial": "misaligned_advice_rate",
    "fam_expr_extreme_sports": "misaligned_advice_rate",
    "fam_expr_insecure_code": "insecure_code_rate",
    "fam_expr_compliment": "compliment_rate",
    "business_competence": "competence_rate",
    "warmth_expression": "mean_warmth",
}

SIGN_FLIP = {"persona_drift"}  # leakage = drift = 1 - consistency


def _scalar(column_id: str, summary: dict) -> float | None:
    key = PRIMARY_SCALAR[column_id]
    val = summary.get(key)
    if val is None:
        return None
    val = float(val)
    if column_id in SIGN_FLIP:
        val = 1.0 - val
    return val


def cluster_bootstrap_ci(
    verdicts: list[dict],
    *,
    positive_key: str,
    cluster_key: str = "probe_id",
    n_boot: int = 1000,
    seed: int = 545,
) -> tuple[float, float] | None:
    """Cluster bootstrap 95% CI for a rate, clustered by probe (#474 lesson:
    completions of the same probe are not independent)."""
    clusters: dict[str, list[int]] = {}
    for v in verdicts:
        if "_judge_error" in v or positive_key not in v:
            continue
        clusters.setdefault(str(v.get(cluster_key, "?")), []).append(int(bool(v[positive_key])))
    keys = list(clusters)
    if len(keys) < 3:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [clusters[rng.choice(keys)] for _ in keys]
        flat = [x for cl in sample for x in cl]
        stats.append(sum(flat) / len(flat))
    stats.sort()
    return (stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1])


def _load_cell_column(cell_dir: Path, column_id: str, context: str) -> dict | None:
    p = cell_dir / f"{column_id}__{context}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def assemble(*, base_cell: str = "base_panel") -> dict[str, Path]:  # noqa: C901 — per-cell assembly, intentionally flat
    """Build L_matrix.json + cell_metadata.json + base_panel.json."""
    cdir = cells_dir()
    if not cdir.exists():
        raise FileNotFoundError(f"No cells directory at {cdir} — run evals first")
    out = output_root()

    # ---- base panel -------------------------------------------------------
    base_dir = cdir / base_cell
    base_panel: dict[str, dict] = {}
    if base_dir.exists():
        for p in sorted(base_dir.glob("*__*.json")):
            if p.name.startswith(_NON_COLUMN_PREFIXES):
                continue
            d = json.loads(p.read_text())
            col, ctx = d["column"], d["context"]
            scalar = _scalar(col, d["summary"]) if col in PRIMARY_SCALAR else None
            base_panel[f"{col}__{ctx}"] = {
                "scalar": scalar,
                "summary": d["summary"],
            }
    base_path = out / "base_panel.json"
    base_path.write_text(
        json.dumps({"panel": base_panel, "metadata": reproducibility_metadata()}, indent=1)
    )

    # ---- per-cell matrix entries (persisted incrementally) ----------------
    matrix: dict[str, dict] = {}
    metadata: dict[str, dict] = {}
    matrix_path = out / "L_matrix.json"
    meta_path = out / "cell_metadata.json"
    for cell_dir in sorted(d for d in cdir.iterdir() if d.is_dir() and d.name != base_cell):
        cell_id = cell_dir.name
        parts = cell_id.rsplit("_seed", 1)
        if len(parts) != 2:
            logger.warning("Skipping unrecognized cell dir %s", cell_id)
            continue
        row_arm, seed = parts[0], int(parts[1])
        row_id = next((rid for rid in ROWS if row_arm.startswith(rid)), None)
        if row_id is None:
            logger.warning("Cell %s matches no registered row", cell_id)
            continue
        arm = row_arm[len(row_id) :].lstrip("_") or "primary"
        row = ROWS[row_id]
        cell_entry: dict = {}
        for p in sorted(cell_dir.glob("*__*.json")):
            if p.name.startswith(_NON_COLUMN_PREFIXES):
                continue
            d = json.loads(p.read_text())
            col, ctx = d["column"], d["context"]
            if col not in PRIMARY_SCALAR:
                continue
            level = _scalar(col, d["summary"])
            base_entry = base_panel.get(f"{col}__{ctx}", {})
            base_level = base_entry.get("scalar")
            if col == "marker":
                leak = level  # delta by construction (trained - base, same forward pass)
                base_level = None  # no separate base scalar; never a junk {} placeholder
            elif level is None or base_level is None:
                leak = None
            else:
                leak = level - base_level
            verdicts = d.get("verdicts", [])
            ci = None
            pk = None
            for cand in (
                "misaligned",
                "refused",
                "insecure",
                "complimenting",
                "is_full_harm",
                "agreed_with_wrong_claim",
            ):
                if verdicts and isinstance(verdicts[0], dict) and cand in verdicts[0]:
                    pk = cand
                    break
            if pk:
                ci = cluster_bootstrap_ci(verdicts, positive_key=pk)
            saturated = level is not None and col != "marker" and level in (0.0, 1.0)
            cell_entry[f"{col}__{ctx}"] = {
                "level": level,
                "base_level": base_level,
                "L": leak,
                "ci95_cluster_bootstrap": ci,
                "saturation_flag": saturated,
                "extras": {
                    k: v
                    for k, v in d["summary"].items()
                    if k != PRIMARY_SCALAR.get(col) and not isinstance(v, dict)
                },
            }
        diag_key = f"{row.diagonal_column}__default"
        diag = cell_entry.get(diag_key, {})
        band = THRESHOLDS["dose_band_default"]
        diag_level = diag.get("level")
        # Round-1 major #3 fix: the band is RELATIVE to the row's recipe
        # ceiling (what dose-select reads: v / ceiling), never the absolute
        # rate. Ceiling comes from the dose_select.json the dispatcher
        # persisted; the marker row reads its own [5,12] nat band instead.
        dose_path = cell_dir / "dose_select.json"
        ceiling = None
        if dose_path.exists():
            ceiling = json.loads(dose_path.read_text()).get("ceiling")
        if row.expected == "null":
            implant_failed: bool | None = False
        elif diag_level is None:
            implant_failed = None  # no diagonal read — unknown, not "failed"
        elif row.diagonal_column == "marker":
            implant_failed = diag_level < THRESHOLDS["k1_marker_band"][0]
        elif ceiling:
            implant_failed = (diag_level / ceiling) < band[0]
        else:
            # No recorded ceiling (reuse-adapter rows trained to a fixed
            # parent budget, single-checkpoint cells): unknown — never
            # silently flagged failed on an absolute-scale misread.
            implant_failed = None
        metadata[cell_id] = {
            "row": row_id,
            "arm": arm,
            "seed": seed,
            "family": row.family,
            "expected": row.expected,
            "diagonal_column": row.diagonal_column,
            "diagonal_level": diag_level,
            "realized_strength": diag.get("L"),
            "dose_band": band,
            "dose_ceiling": ceiling,
            "implant_failed": implant_failed,
            "columns_present": sorted(cell_entry),
        }
        matrix[cell_id] = cell_entry
        # Checkpoint-per-cell persistence: rewrite after each cell.
        matrix_path.write_text(
            json.dumps({"cells": matrix, "metadata": reproducibility_metadata()}, indent=1)
        )
        meta_path.write_text(
            json.dumps({"cells": metadata, "metadata": reproducibility_metadata()}, indent=1)
        )
    if not matrix:
        raise RuntimeError(f"No cells assembled from {cdir}")

    # K2 read (deliverable-downgrade trigger, never a mid-run park).
    non_null = [m for m in metadata.values() if m["expected"] != "null" and m["arm"] == "primary"]
    failed = [m for m in non_null if m["implant_failed"]]
    k2 = {
        "n_non_null_primary_cells": len(non_null),
        "n_implant_failed": len(failed),
        "k2_triggered": bool(non_null)
        and len(failed) / len(non_null) > THRESHOLDS["k2_implant_failed_fraction"],
    }
    (out / "k2_check.json").write_text(json.dumps(k2, indent=1))
    logger.info("[phase=assemble] %d cells, K2=%s", len(matrix), k2["k2_triggered"])
    return {"L_matrix": matrix_path, "cell_metadata": meta_path, "base_panel": base_path}
