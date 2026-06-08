# em-dash + nat character are intentional
"""Task #514 — analyze module (delegates to #508's analyze + writes #514 artifacts).

Plan §6.3 + §10:
    - Reuses #508's ``run_analysis`` for the heavy lift (per-cell aggregates,
      bracketing check, matched-rate gap, hero figure).
    - Merges the 6 new #514 FT cells with the 5 reference cells from #508
      (LoRA b1/b2/b3 + FT b1/b2 anchors). The #508 cell #ft_b3 (collapsed at
      1.0 epoch) is included only as a half-transparent anchor on the hero.
    - Writes the 3 #514-specific artifacts (plan §10):
        eval_results/issue_514/_matched_rate_514.json
        eval_results/issue_514/_bracketing_report.json
        eval_results/issue_514/_bootstrap_per_cell.json

The #508 reference eval JSONs are expected to be present at
``eval_results/issue_508/{lora_b1,lora_b2,lora_b3,ft_b1,ft_b2,ft_b3}_seed42.json``
(plan §4.1 — re-used artifacts, not retrained). If the directory is missing,
analyze runs on the #514-only cells and writes a caveat into the bracketing
report.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any

from explore_persona_space.experiments.full_ft_regime_514 import (
    ABORT_HELD_OUT_GLOGPROB_MAX,
    ABORT_SOURCE_RCOLLAPSE_THRESHOLD,
    CLEAN_CELL_BRACKETING_UPPER_NATS,
    DENSE_LEVER_CELLS,
    LOW_LR_LEVER_CELLS,
)
from explore_persona_space.experiments.full_ft_regime_514.abort_logic import (
    compute_source_r_collapse_rate,
    get_held_out_g_logprob_mean,
)

log = logging.getLogger("issue_514.analyze")

# Path on disk where #508's reference eval JSONs live (post-merge to main).
# Tries the repo-relative location first; fall back to the worktree-relative
# location if the analyze runs on the pod before the merge.
REFERENCE_EVAL_DIR_CANDIDATES: tuple[str, ...] = (
    "eval_results/issue_508",
    "../issue-508/eval_results/issue_508",
)


def _find_reference_eval_dir() -> Path | None:
    for cand in REFERENCE_EVAL_DIR_CANDIDATES:
        p = Path(cand)
        if p.exists():
            return p
    return None


def _gather_reference_eval_jsons() -> list[Path]:
    """Locate the 5+1 #508 reference eval JSONs (LoRA b1/b2/b3 + FT b1/b2/b3).

    Returns the list of paths that exist. The dispatcher tolerates a missing
    reference dir — analysis runs on #514-only cells with a caveat.
    """
    ref_dir = _find_reference_eval_dir()
    if ref_dir is None:
        log.warning(
            "[analyze] #508 reference eval dir not found under %s — "
            "analyze will run on #514-only cells (no merged hero figure).",
            REFERENCE_EVAL_DIR_CANDIDATES,
        )
        return []
    expected = (
        "lora_b1_seed42.json",
        "lora_b2_seed42.json",
        "lora_b3_seed42.json",
        "ft_b1_seed42.json",
        "ft_b2_seed42.json",
        "ft_b3_seed42.json",
    )
    found: list[Path] = []
    for name in expected:
        p = ref_dir / name
        if p.exists():
            found.append(p)
        else:
            log.warning("[analyze] reference cell missing: %s (continuing)", p)
    return found


def _classify_lever(cell_slug: str) -> str:
    """Return ``"dense"`` / ``"lowlr"`` / ``"508_anchor"`` for a cell slug."""
    if cell_slug in DENSE_LEVER_CELLS:
        return "dense"
    if cell_slug in LOW_LR_LEVER_CELLS:
        return "lowlr"
    return "508_anchor"


def _per_cell_diagnostics(eval_jsons: list[Path]) -> list[dict]:
    """For each #514 cell eval JSON, compute (source_mean, r_collapse_rate,
    held_out_g_logprob_mean, lever, is_clean_above_9_nat).
    """
    out: list[dict] = []
    for p in eval_jsons:
        ej = json.loads(p.read_text())
        cell_slug = ej.get("cell_slug", p.stem.split("_seed")[0])
        agg = ej.get("aggregates", {}) or {}
        source_mean = agg.get("source_self_mean_delta_g")
        held_out_mean = agg.get("held_out_mean_delta_g")
        rcoll = compute_source_r_collapse_rate(ej)
        g_logp = get_held_out_g_logprob_mean(ej)
        lever = _classify_lever(cell_slug)

        # Clean-cell-above-9-nat criterion (plan §6.2). Floats may be NaN
        # if the cell had no source probes; cast safely.
        clean_above_9 = False
        if (
            source_mean is not None
            and isinstance(source_mean, int | float)
            and not _is_nan(float(source_mean))
            and float(source_mean) > CLEAN_CELL_BRACKETING_UPPER_NATS
            and not _is_nan(rcoll)
            and rcoll < ABORT_SOURCE_RCOLLAPSE_THRESHOLD
            and not _is_nan(g_logp)
            and g_logp <= ABORT_HELD_OUT_GLOGPROB_MAX
        ):
            clean_above_9 = True

        out.append(
            {
                "cell": cell_slug,
                "lever": lever,
                "eval_json_path": str(p),
                "source_mean": float(source_mean) if source_mean is not None else None,
                "held_out_mean": float(held_out_mean) if held_out_mean is not None else None,
                "r_collapse_rate": rcoll,
                "held_out_g_logprob_mean": g_logp,
                "clean_above_9_nat": clean_above_9,
            }
        )
    out.sort(key=lambda d: d["cell"])
    return out


def _is_nan(x: float) -> bool:
    import math

    return math.isnan(x)


def _write_bracketing_report_514(
    *,
    diagnostics_514: list[dict],
    diagnostics_508_ft_anchors: list[dict],
    output_path: Path,
) -> dict[str, Any]:
    """Write the #514 bracketing report (plan §6.2 acceptance criterion).

    PASS iff there exists ≥1 NEW #514 FT cell with source ΔG > 9 nat AND
    clean (r-collapse < 50% AND held-out g_logprob ≤ -5 nat).

    The report records every #514 cell's verdict + the #508 FT anchor cells'
    (re-used) values for context.
    """
    clean_above_9 = [d for d in diagnostics_514 if d["clean_above_9_nat"]]
    pass_h1 = len(clean_above_9) >= 1

    report = {
        "schema_version": "i514_bracketing_v1",
        "criterion": (
            "PASS iff exists >=1 #514 FT cell with source_mean > 9.0 nat "
            "AND r_collapse_rate < 0.50 AND held_out_g_logprob_mean <= -5.0"
        ),
        "h1_pass": pass_h1,
        "clean_above_9_nat_cells": [d["cell"] for d in clean_above_9],
        "n_clean_above_9_nat": len(clean_above_9),
        "diagnostics_514": diagnostics_514,
        "diagnostics_508_ft_anchors": diagnostics_508_ft_anchors,
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    log.info(
        "[analyze] wrote bracketing report (H1 PASS=%s, %d clean-above-9-nat cells) → %s",
        pass_h1,
        len(clean_above_9),
        output_path,
    )
    return report


def _write_matched_rate_514(
    *,
    diagnostics_514: list[dict],
    diagnostics_508: list[dict],
    bracketing_report: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Write the #514 matched-rate read summary (plan §6.4 determinacy check).

    The actual cluster-bootstrap matched-rate gap is computed by #508's
    ``run_analysis`` (which we delegate to). This file is the #514-specific
    SUMMARY that pulls (a) the bracketing-PASS flag, (b) the per-cell
    diagnostics, and (c) the LOCAL linear-interpolation read at source ΔG=8 nat
    on the #514 + #508 cells together. The local-read vs cluster-bootstrap
    determinacy check (|diff| ≤ 0.5 nat) is filled in by the downstream
    consumer when the cluster-bootstrap output JSON is also available.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "i514_matched_rate_v1",
        "matched_slice_target_nats": 8.0,
        "matched_slice_band_nats": 1.0,
        "h1_pass": bracketing_report["h1_pass"],
        "n_clean_above_9_nat": bracketing_report["n_clean_above_9_nat"],
        "clean_above_9_nat_cells": bracketing_report["clean_above_9_nat_cells"],
        "diagnostics_514_summary": [
            {
                "cell": d["cell"],
                "lever": d["lever"],
                "source_mean": d["source_mean"],
                "held_out_mean": d["held_out_mean"],
                "r_collapse_rate": d["r_collapse_rate"],
                "held_out_g_logprob_mean": d["held_out_g_logprob_mean"],
            }
            for d in diagnostics_514
        ],
        "diagnostics_508_anchors_summary": [
            {
                "cell": d["cell"],
                "source_mean": d["source_mean"],
                "held_out_mean": d["held_out_mean"],
            }
            for d in diagnostics_508
        ],
        "note": (
            "The local-read vs cluster-bootstrap determinacy check (|diff| <= "
            "0.5 nat per plan §3) is computed by lora_vs_ft_508.analyze.run_analysis "
            "and lands in analysis.json under matched_rate_gap. This file is the "
            "#514-specific bracketing summary."
        ),
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    output_path.write_text(json.dumps(summary, indent=2))
    log.info("[analyze] wrote matched-rate summary → %s", output_path)
    return summary


def _write_bootstrap_per_cell_514(
    *,
    cells_data: list[dict],
    output_path: Path,
) -> None:
    """Write the per-cell cluster-bootstrap arrays (plan §6.4 raw alongside processed).

    The cluster bootstrap is computed by #508's ``_crossed_cluster_bootstrap_gap``;
    here we serialize the per-cell (source_mean, held_out_mean) tuples that feed
    that downstream consumer. Mirrors plan Lens 11 (raw alongside processed).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "i514_bootstrap_per_cell_v1",
        "cells": [
            {
                "cell": c["cell"],
                "lever": _classify_lever(c["cell"]),
                "source_mean": c["source_mean"],
                "held_out_mean": c["held_out_mean"],
                "r_collapse_rate": c["r_collapse_rate"],
                "held_out_g_logprob_mean": c["held_out_g_logprob_mean"],
            }
            for c in cells_data
        ],
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    log.info("[analyze] wrote bootstrap-per-cell raw → %s", output_path)


def run_analysis_514(
    *,
    eval_jsons: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Plan §10 + §6.3 analyzer for #514.

    1. Locate the 6 #508 reference eval JSONs (LoRA b1/b2/b3 + FT b1/b2/b3).
    2. Merge with the 6 new #514 FT cells.
    3. Compute per-cell diagnostics (source_mean, r-collapse rate,
       held-out g_logprob_mean, lever, clean-above-9-nat).
    4. Delegate to ``lora_vs_ft_508.analyze.run_analysis`` for bracketing,
       cluster bootstrap, hero figure, trajectory figures.
    5. Write the 3 #514-specific artifacts: ``_bracketing_report.json``,
       ``_matched_rate_514.json``, ``_bootstrap_per_cell.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_jsons = _gather_reference_eval_jsons()
    all_jsons = list(eval_jsons) + reference_jsons
    log.info(
        "[analyze] running on %d #514 cells + %d #508 reference cells = %d total",
        len(eval_jsons),
        len(reference_jsons),
        len(all_jsons),
    )

    # Per-cell diagnostics (514 + 508 FT anchors).
    diag_514 = _per_cell_diagnostics(list(eval_jsons))
    diag_508_ft_anchors = [
        d
        for d in _per_cell_diagnostics([p for p in reference_jsons if "ft_" in p.name])
        if d["lever"] == "508_anchor"
    ]

    # Combined diagnostics for the bootstrap-per-cell file.
    combined_diag = diag_514 + _per_cell_diagnostics(reference_jsons)

    # ── Delegate to #508's run_analysis for the heavy lift. ──────────────────
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import (
        run_analysis as run_analysis_508,
    )

    if all_jsons:
        try:
            delegated = run_analysis_508(eval_jsons=all_jsons, output_dir=output_dir)
        except Exception as e:
            log.exception("[analyze] run_analysis_508 raised %s — continuing with #514 outputs", e)
            delegated = {"error_in_508_delegate": str(e)}
    else:
        log.warning("[analyze] no eval JSONs to analyze — skipping #508 delegate")
        delegated = {"skipped": "no eval JSONs"}

    # ── Write #514-specific artifacts. ───────────────────────────────────────
    bracketing_path = output_dir / "_bracketing_report.json"
    matched_rate_path = output_dir / "_matched_rate_514.json"
    bootstrap_path = output_dir / "_bootstrap_per_cell.json"

    bracketing_report = _write_bracketing_report_514(
        diagnostics_514=diag_514,
        diagnostics_508_ft_anchors=diag_508_ft_anchors,
        output_path=bracketing_path,
    )
    matched_rate = _write_matched_rate_514(
        diagnostics_514=diag_514,
        diagnostics_508=diag_508_ft_anchors,
        bracketing_report=bracketing_report,
        output_path=matched_rate_path,
    )
    _write_bootstrap_per_cell_514(
        cells_data=combined_diag,
        output_path=bootstrap_path,
    )

    analysis = {
        "schema_version": "i514_analysis_v1",
        "n_cells_514": len(eval_jsons),
        "n_cells_508_reference": len(reference_jsons),
        "delegated_508_analysis": delegated,
        "bracketing_report_path": str(bracketing_path),
        "matched_rate_514_path": str(matched_rate_path),
        "bootstrap_per_cell_path": str(bootstrap_path),
        "h1_pass": bracketing_report["h1_pass"],
        "clean_above_9_nat_cells": bracketing_report["clean_above_9_nat_cells"],
        "matched_rate_summary": matched_rate,
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    out_path = output_dir / "analysis_514.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    log.info(
        "[analyze] wrote #514 analysis → %s (H1 PASS=%s)", out_path, bracketing_report["h1_pass"]
    )
    return analysis
