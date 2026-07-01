#!/usr/bin/env python
"""Issue #778 amendment — Leg A recipe-sanity gate (plan v3 §7, the SINGLE gate).

After Leg A's evil trait completes (the smoke canary), compute the matched-trait
evil corrected ``overall_r`` (max-over-28-layers |r| of last-prompt projection onto
cached r_B vs the graded trait score) and assert it clears a POSITIVE floor.

Grounded floor: the paper reports evil monitoring overall r = 0.747; #778's own
tautological version read 0.94; even the de-inflated corrected value should clear
0.5 on evil. A FAIL means a recipe bug (8-prompt ladder not swapped in / wrong
layer indexing / r_B sign flip / wrong acts tensor / judge mis-wired) — bounce to
the implementer, do NOT burn the rest of the GPU time. This is a matched-r-only
read (no null pools), fast + closed-form.

Exit 0 + prints ``[recipe-gate] PASS ...`` when overall_r >= threshold; on FAIL
writes a failure sentinel naming the gate and exits non-zero (the dispatch aborts
before Leg A other traits + Leg B).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib

from explore_persona_space.analysis import null_battery as nb
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def compute_overall_r(out_root: Path, eval_root: Path, trait: str, input_tag: str) -> dict:
    """Compute matched-trait max-over-layers overall |r| for one trait/leg."""
    import torch

    rows: list[dict] = []
    with open(eval_root / f"{input_tag}_{trait}.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    kept = [r for r in rows if r["mean_trait_score"] is not None]
    if len(kept) < 3:
        raise RuntimeError(f"{trait}/{input_tag}: only {len(kept)} scored cells — cannot correlate")
    target = np.array([r["mean_trait_score"] for r in kept], dtype=np.float64)

    acts_path = out_root / input_tag / f"{trait}_acts.pt"
    if not acts_path.exists():
        raise RuntimeError(f"{trait}/{input_tag}: raw acts tensor missing at {acts_path}")
    acts = torch.load(acts_path, weights_only=False).numpy().astype(np.float64)
    kept_mask = np.array([r["mean_trait_score"] is not None for r in rows])
    acts_kept = acts[kept_mask]
    if acts_kept.shape[0] != target.shape[0]:
        raise RuntimeError(
            f"{trait}/{input_tag}: acts kept {acts_kept.shape[0]} != target {target.shape[0]}"
        )

    rb = torch.load(out_root / "rb" / f"{trait}.pt", weights_only=False).numpy().astype(np.float64)
    r_layers = nb.r_per_layer(acts_kept, rb, target)
    max_abs = nb.max_abs_over_layers(r_layers)
    sel_layer = nb.argmax_abs_layer(r_layers)
    return {
        "trait": trait,
        "input_tag": input_tag,
        "overall_r_max_abs": float(max_abs),
        "overall_r_signed_at_selected": float(r_layers[sel_layer]),
        "selected_layer": int(sel_layer),
        "n_cells": int(target.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #778 Leg A recipe-sanity gate.")
    parser.add_argument("--out-root", default="data/issue_778")
    parser.add_argument("--eval-results-root", default="eval_results/issue_778")
    parser.add_argument("--trait", default="evil")
    parser.add_argument("--input-tag", default="monitoring_corrected")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--logs-dir", default="/workspace/logs")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    eval_root = Path(args.eval_results_root)
    res = compute_overall_r(out_root, eval_root, args.trait, args.input_tag)
    res["threshold"] = args.threshold
    lib.log_phase("recipe_gate", json.dumps(res))

    if res["overall_r_max_abs"] < args.threshold:
        res["gate"] = "recipe_sanity"
        res["verdict"] = "FAIL"
        # Fail-loud sentinel naming the gate (recipe bug ⇒ bounce), then non-zero.
        note = {
            "gate": "recipe_sanity",
            "verdict": "FAIL",
            "detail": (
                f"{args.trait} {args.input_tag} overall_r max-over-layers "
                f"{res['overall_r_max_abs']:.4f} < floor {args.threshold} — recipe bug "
                f"(8-prompt ladder not swapped / wrong layer / r_B sign / acts tensor / judge). "
                f"Do NOT proceed to the rest of the run."
            ),
            "result": res,
            "reproducibility": lib.repro_metadata(),
        }
        try:
            path = lib.write_results_sentinel(
                issue=778,
                kind="epm:failure",
                version=1,
                note=note,
                logs_dir=Path(args.logs_dir),
            )
            print(f"[recipe-gate] FAIL sentinel -> {path}", flush=True)
        except Exception as e:
            print(f"[recipe-gate] FAIL (sentinel write raised: {e})", flush=True)
        print(
            f"[recipe-gate] FAIL: {args.trait} {args.input_tag} overall_r "
            f"{res['overall_r_max_abs']:.4f} < {args.threshold}",
            flush=True,
        )
        sys.exit(2)

    print(
        f"[recipe-gate] PASS: {args.trait} {args.input_tag} overall_r "
        f"{res['overall_r_max_abs']:.4f} >= {args.threshold} "
        f"(layer {res['selected_layer']}, n={res['n_cells']})",
        flush=True,
    )
    print(json.dumps(res))


if __name__ == "__main__":
    main()
