#!/usr/bin/env python3
"""Issue #657 Phase 0 — assemble the join substrate (CPU, off-pod, pre-pod).

For one behavior, this loads the three reused inputs and joins them onto a single
per-(source, bystander) frame keyed on the 24-persona panel axis:

  1. the reused leakage cell substrate (#518 syco/refusal/em; #605 marker), via
     ``load_cell_substrate``;
  2. the reused persona-vector bank (#623 method_a/method_b + the issue-657
     freshly-extracted <=7 missing panel personas), via ``load_persona_vectors``;
  3. the reused behavior direction at the headline layer/readout, via
     ``load_behavior_direction``.

It then computes the bystander-level ``align`` (cosine) + ``prior_centroid_projection``
(M-Alts1) and writes a normalized join JSON. It is a pure data-assembly step — no
GPU, no model load — so it doubles as the ``### join-substrate-smoke`` phase.

The directions for refusal/marker/em live under the issue-657 extraction output;
sycophancy is REUSED from #623 verbatim. ``--direction-dir`` overrides per behavior
(default: the conventional issue-657 layout below).

Usage (smoke, sycophancy, all reused — no GPU):
  uv run python scripts/issue657_join_substrate.py --behavior sycophancy \\
      --persona-dir data/issue657/persona_vectors/method_a \\
      --syco-direction-dir data/issue657/behavior_directions/sycophancy \\
      --out /tmp/issue657_join_sycophancy.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.issue657_alignment_predictor import (  # noqa: E402
    DEFAULT_LAYERS,
    HEADLINE_LAYER,
    build_predictor_frame,
    compute_alignment,
    load_behavior_direction,
    load_cell_substrate,
    load_persona_vectors,
    marker_panel_overlap,
    prior_centroid_projection,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402


def _resolve(p: str) -> Path:
    return PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)


def _digest(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def build_join(
    behavior: str,
    eval_results_root: Path,
    persona_dirs: list[Path],
    direction_dir: Path,
    layer: int,
    readout: str,
    centering: str,
) -> dict:
    """Build the normalized join for one behavior. Pure CPU; raises on any missing input."""
    table = load_cell_substrate(behavior, eval_results_root)
    persona_vectors = load_persona_vectors(persona_dirs, DEFAULT_LAYERS)
    behavior_dir = load_behavior_direction(behavior, {behavior: direction_dir}, layer, readout)

    personas = table.bystanders
    align = compute_alignment(
        persona_vectors, behavior_dir, personas, layer, DEFAULT_LAYERS, centering
    )
    pcp = prior_centroid_projection(
        persona_vectors, table.bystander_base_rate, personas, layer, DEFAULT_LAYERS, centering
    )
    frame = build_predictor_frame(table, align, pcp)

    # v6 FIRST Phase-0 deliverable: the marker panel-overlap primary-set-membership
    # decision (|EVAL_PERSONAS_24 ∩ #605-marker-panel|). Computed from the same
    # eval-results root + carried in EVERY behavior's join so the decision is the
    # first thing recorded (a non-silent, logged gate — §7). Overlap < 8 -> marker
    # DV-(a)-only + primary H3 narrows to {sycophancy, refusal}.
    overlap_decision = marker_panel_overlap(eval_results_root)

    return {
        "marker_panel_overlap": overlap_decision,
        "behavior": behavior,
        "layer": layer,
        "readout": readout,
        "centering": centering,
        "n_cells_substrate": table.n_cells(),
        "n_cells_joined": len(frame["cells"]),
        "n_resolvable_bystanders": frame["n_resolvable"],
        "resolvable_bystanders": frame["resolvable_bystanders"],
        "dropped_bystanders": sorted({d["bystander"] for d in frame["dropped"]}),
        "cells": frame["cells"],
        "bystander_base_rate": table.bystander_base_rate,
        "substrate_provenance": table.provenance,
        "fields_md5": _digest(sorted(frame["cells"][0].keys()) if frame["cells"] else []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #657 Phase 0 — join substrate (CPU).")
    parser.add_argument(
        "--behavior", required=True, choices=["sycophancy", "refusal", "marker", "em"]
    )
    parser.add_argument(
        "--eval-results-root",
        default="eval_results",
        help="Root holding issue_518/ + issue_605/ (relative to repo root).",
    )
    parser.add_argument(
        "--persona-dir",
        action="append",
        default=None,
        help=(
            "Persona-vector dir holding <persona>.pt (repeatable; later overrides "
            "earlier). Default: the issue-657 conventional layout + #623 bank."
        ),
    )
    parser.add_argument(
        "--direction-dir",
        default=None,
        help="Behavior-direction dir holding <readout>_<layer>.pt (overrides default layout).",
    )
    parser.add_argument("--layer", type=int, default=HEADLINE_LAYER)
    parser.add_argument("--readout", default="last_token", choices=["last_token", "response_avg"])
    parser.add_argument("--centering", default="global_mean", choices=["global_mean", "none"])
    parser.add_argument("--out", required=True, help="Output join JSON (relative to repo root).")
    args = parser.parse_args()

    load_dotenv()

    eval_root = _resolve(args.eval_results_root)

    # Method dir tracks the readout: last_token -> method_a, response_avg -> method_b.
    method_subdir = "method_a" if args.readout == "last_token" else "method_b"

    # Default persona-vector dirs: the issue-657 freshly-extracted personas first,
    # then the #623 bank (later dirs override; they should not overlap personas).
    if args.persona_dir:
        persona_dirs = [_resolve(p) for p in args.persona_dir]
    else:
        persona_dirs = [
            _resolve(f"data/issue657/persona_vectors/{method_subdir}"),
            _resolve(f"data/persona_vectors/issue623/{method_subdir}"),
        ]

    # Default direction dir per behavior.
    if args.direction_dir:
        direction_dir = _resolve(args.direction_dir)
    elif args.behavior == "sycophancy":
        direction_dir = _resolve("data/persona_vectors/issue623/sycophancy_trait")
    else:
        direction_dir = _resolve(f"data/issue657/behavior_directions/{args.behavior}")

    join = build_join(
        args.behavior,
        eval_root,
        persona_dirs,
        direction_dir,
        args.layer,
        args.readout,
        args.centering,
    )

    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(join, indent=2))

    # FIRST deliverable: the marker panel-overlap primary-set-membership decision (§7).
    ov = join["marker_panel_overlap"]
    print(
        f"[phase=join_substrate] marker_panel_overlap={ov['marker_panel_overlap']} "
        f"(min={ov['min_overlap']}) marker_leakage_excluded={ov['marker_leakage_excluded']} "
        f"-> {ov['decision']}",
        flush=True,
    )
    print(
        f"[phase=join_substrate] behavior={args.behavior} "
        f"cells_joined={join['n_cells_joined']}/{join['n_cells_substrate']} "
        f"resolvable_bystanders={join['n_resolvable_bystanders']} "
        f"dropped={join['dropped_bystanders']} fields_md5={join['fields_md5']} "
        f"-> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
