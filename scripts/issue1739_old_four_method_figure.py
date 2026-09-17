"""Replot the historical P-B persona-vector results in the current 3x3 layout.

Keep the original three projection series and add the matching cached E1
context-extracted direction. Preserve the historical OOD standard errors.
This is a plot-only assembly of committed result rows, with no new scoring.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import issue1739_four_method_figure as figure  # noqa: E402

BASE_REV = "5aae0a472b"
CONTEXT_REV = "6dba8178f061980d3285fe4d35ba3149f7bdba3c"
OUT = ROOT / "eval_results/issue_1739/old_four_method_figure_20260917"
STEM = "c5_behavior_transfer_original_four"
OLD_ARMS = {
    "answer_direction_on_context": "arm1_ctx_e1",
    "mapped_answer": "arm6_map_proj_e1",
    "real_answer": "arm11_oracle_proj",
}


def read_blob(revision, path, inputs):
    """Read pinned historical bytes and record their hash."""
    value = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
    inputs[f"{revision}:{path}"] = hashlib.sha256(value).hexdigest()
    return json.loads(value)


def selected(rows, regime):
    """Reproduce the screenshot's P-B setting selection, without synthetic eval."""
    pb = [r for r in rows if r["protocol"] == "P-B"]
    if regime == "Generic chat":
        return [r for r in pb if r["eval_rung"] == "wildchat_rung"]
    if regime == "ID":
        return [r for r in pb if r["eval_rung"] == "heldin:train"]
    if regime != "OOD":
        raise ValueError(regime)
    return [r for r in pb if r["fit"].removeprefix("P-B-holdout-") == r["eval_rung"]]


def assemble():
    """Verify old E1 parity, then summarize frozen answer/context directions."""
    inputs, cells, maps, parity_count = {}, [], {}, 0
    for behavior in figure.BEHAVIORS:
        base = read_blob(
            BASE_REV, f"eval_results/issue_1739/r2v2_fits/{behavior}/all_arms_spearman.json", inputs
        )
        factorial = read_blob(
            CONTEXT_REV,
            f"eval_results/issue_1739/r2v2_factorial/{behavior}/factorial_rows.json",
            inputs,
        )
        old = [r for r in base["transfer_rows"] if r["arm"] in OLD_ARMS.values()]
        reproduction = {
            (r["fit"], r["eval_rung"], r["arm"]): r
            for r in factorial["transfer_rows"]
            if r["protocol"] == "P-B" and r["regime"] == "e1"
        }
        for row in old:
            if row["protocol"] != "P-B":
                continue
            match = reproduction[(row["fit"], row["eval_rung"], row["arm"])]
            for key in ("rho_frozen", "n_eval", "layer"):
                if match[key] != row[key]:
                    raise ValueError(f"Old E1 parity failed: {behavior}/{key}")
            parity_count += 1
        native = [
            r
            for r in factorial["transfer_rows"]
            if r["regime"] == "e1_fc" and r["arm"] == "arm1_ctx_e1"
        ]
        maps[behavior] = {
            "n_pairs": base["meta"]["n_u"],
            "pool": base["meta"]["u_pool_label"],
            "frozen_layers": base["meta"]["frozen_layers"],
        }
        for regime in ("Generic chat", "ID", "OOD"):
            estimates, rosters = {}, []
            for method in figure.METHODS:
                source = (
                    native
                    if method == "context_native"
                    else [r for r in old if r["arm"] == OLD_ARMS[method]]
                )
                picked = selected(source, regime)
                if not picked:
                    raise ValueError(f"Missing historical cell: {behavior}/{regime}/{method}")
                # A fixed projection repeats identically across P-B readout folds.
                # Count each dataset once; the folds are not independent evaluations.
                by_dataset = {}
                for row in picked:
                    key = row["eval_rung"]
                    record = {k: row[k] for k in ("rho_frozen", "n_eval", "layer")}
                    if key in by_dataset and by_dataset[key] != record:
                        raise ValueError(f"Fixed projections differ across folds: {key}")
                    by_dataset[key] = record
                rho = statistics.mean(r["rho_frozen"] for r in picked)
                values = [r["rho_frozen"] for r in by_dataset.values()]
                if rho != statistics.mean(values):
                    raise ValueError("Dataset deduplication changed the historical mean")
                sem = statistics.stdev(values) / len(values) ** 0.5 if regime == "OOD" else None
                estimates[method] = {
                    "rho": rho,
                    "sem": sem,
                    "interval": [rho - sem, rho + sem] if sem is not None else None,
                    "dataset_rows": by_dataset,
                }
                rosters.append({r: v["n_eval"] for r, v in by_dataset.items()})
            if not all(roster == rosters[0] for roster in rosters):
                raise ValueError(f"Methods use different evaluation rows: {behavior}/{regime}")
            cells.append(
                {
                    "behavior": behavior,
                    "regime": regime,
                    "n": sum(rosters[0].values()),
                    "datasets": list(rosters[0]),
                    "arms": estimates,
                }
            )
    return {
        "protocol": "Original P-B",
        "methods": figure.METHODS,
        "maps": maps,
        "inputs": inputs,
        "cells": cells,
        "projection_space": "Original context-covariance whitening of activations and directions",
        "uncertainty": "Original ±1 standard error across OOD datasets; no generic/ID interval",
        "context_baseline": "Cached e1_fc arm1_ctx_e1, using the original context-projection layer",
        "validation": {"exact_e1_parity_rows": parity_count},
        "caveats": [
            "ID is held out from the behavior readout, but included in fitting the ADD map.",
            "Layers differ across behaviors and projection methods.",
            "Three harmful-compliance OOD datasets and generic chat had low DV spread in the source figure.",
        ],
        "assembler_sha256": figure.sha(Path(__file__)),
    }


if __name__ == "__main__":
    data = assemble()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    figure.render(
        data,
        stem=STEM,
        title="Fixed-direction projections · original results",
        footer=(
            "Original maps, whitening and layers. ID rows were included in map fitting.",
            "Whiskers: SE across OOD datasets. Context → context added from the matching cached run.",
        ),
    )
