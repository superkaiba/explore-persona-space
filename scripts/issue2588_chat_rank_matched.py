#!/usr/bin/env python3
"""Fixed-L24 chat control: reuse A, refit only B, and retain the L22 comparison.

No model forwards, GPU, lambda sweep, or new train/test rows. The single full
production fit is also the timing pilot; there is no remaining fit to launch.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import scipy  # noqa: E402

import issue2588_chat_rank as rank  # noqa: E402
import issue2588_chat_rank_diversity as diversity  # noqa: E402
import issue2588_chat_stage_rank as staging  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

REVISION = "5f60a146e91248e5e1c84cf879ff5b4f1357a087"
RUN_ID = "qwen3-chat-v3"
LAYER = 24
OLD_RESULTS = ROOT / "eval_results/issue_2588/qwen3_chat_rank"


def representation_measures(xtr, ytr, payload, fitted_spectrum) -> dict:
    """Reuse the old diversity definitions and the rank PCA eigendecomposition."""
    xn = (xtr.astype(np.float64) - payload["xmu"].astype(np.float64)) / payload["xsd"].astype(
        np.float64
    )
    spectra = {
        "input": diversity.centred_spectrum(xn),
        "answer": diversity.centred_spectrum(ytr),
        "fitted_output": np.asarray(fitted_spectrum, dtype=np.float64),
    }
    entry = {space: diversity.measures(spectrum) for space, spectrum in spectra.items()}
    entry["fitted_over_answer_effective_rank"] = (
        entry["fitted_output"]["effective_rank_entropy"] / entry["answer"]["effective_rank_entropy"]
    )
    entry["fitted_over_answer_participation_ratio"] = (
        entry["fitted_output"]["participation_ratio"] / entry["answer"]["participation_ratio"]
    )
    entry["spectra"] = {space: spectrum.tolist() for space, spectrum in spectra.items()}
    return entry


def verify_reused_a(source_root: Path) -> dict:
    """Hash-verify the unchanged A L24 inputs against the banked rank result and HF."""
    previous = json.loads((OLD_RESULTS / "rank_a.json").read_text())
    if previous["layer_star"] != LAYER or previous["selected_lambda"] != 1000:
        raise ValueError("The A comparison no longer matches the approved fixed-L24 recipe")
    cell = source_root / "generic" / RUN_ID / "cells_cap_long/q3_8b_a"
    manifest = rank.input_manifest(cell, "prompt_last", LAYER)
    if manifest != previous["provenance"]["input_files"]:
        raise ValueError("Reused A inputs drifted from the banked result")
    verified = rank.verify_durable_inputs(cell, "a", RUN_ID, REVISION, manifest)
    return {"result_sha256": rank.sha256(OLD_RESULTS / "rank_a.json"), **verified}


def analyze(source_root: Path, output_dir: Path) -> dict:
    """Run exactly one production-size B L24 unit, failing on stale output or parity."""
    if (output_dir / "result.json").exists() or (output_dir / "rank_b_l24.json").exists():
        raise ValueError("Output already exists; inspect it rather than duplicating this fit")
    started = time.monotonic()
    provenance = as_metadata_dict(git_provenance(ROOT), phase="matched-l24")
    provenance.update(
        timestamp_utc=datetime.now(UTC).isoformat(),
        hf_revision=REVISION,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        source_hashes={
            p.name: rank.sha256(p)
            for p in (Path(__file__), Path(rank.__file__), Path(diversity.__file__))
        },
        reused_result_hashes={
            name: rank.sha256(OLD_RESULTS / name)
            for name in ("rank_a.json", "rank_b.json", "rank_diversity.json")
        },
    )
    verified_a = verify_reused_a(source_root)
    cell = source_root / "generic" / RUN_ID / "cells_cap_long/q3_8b_b"
    fits = json.loads((cell / "fits/fits_cot_boundary.json").read_text())
    identity = json.loads((cell / "run_identity.json").read_text())
    if fits["identity"] != identity or identity["run_id"] != RUN_ID:
        raise ValueError("B fit/capture identity mismatch")
    if (
        identity["model_revision"] != rank.MODEL_REVISION
        or fits["input_position"] != "cot_boundary"
    ):
        raise ValueError("Wrong model or input position")
    star = fits["layers"][str(LAYER)]
    if star["d"] != 4096 or star["fit_meta"]["selected_lambda"] != 1000:
        raise ValueError("Unreviewed fixed-layer recipe")
    manifest = rank.input_manifest(cell, "cot_boundary", LAYER)
    durable = rank.verify_durable_inputs(cell, "b", RUN_ID, REVISION, manifest)
    provenance.update(input_files=manifest, durable_verification=durable, reused_a=verified_a)
    print("[matched-l24] loading B train/validation/test, d=4096", flush=True)
    pairs = [rank.load_split(cell, split, LAYER, "cot_boundary", 4096) for split in rank.SPLITS]
    rows = {split: len(pair[0]) for split, pair in zip(rank.SPLITS, pairs, strict=True)}
    if list(rows.values()) != [star["n"][key] for key in ("tr", "val", "te")]:
        raise ValueError("Fit/capture row-count mismatch")
    if rows["train_10k"] <= 4096:
        raise ValueError("Refuse underdetermined fit")
    fit_started = time.monotonic()
    payload = rank.reconstruct(*[a for pair in pairs for a in pair], lam=1000.0)
    parity = {}
    for split, expected in (
        ("val", star["fit_meta"]["val_r2_at_selected"]),
        ("test", star["test_r2"]),
    ):
        actual = rank.pooled_r2(payload[f"pred_{split}"], payload[f"target_{split}"])
        parity[split] = {"reconstructed": actual, "banked": expected, "delta": actual - expected}
        if abs(actual - expected) > rank.PARITY_TOLERANCE:
            raise ValueError(f"B L24 parent parity failed on {split}: {parity[split]}")
    result_rank = rank.reduced_rank(payload, pairs[0][0], include_spectrum=True)
    rank.write_json(output_dir / "rank_b_l24.json", {**result_rank, "provenance": provenance})
    print(f"[matched-l24] rank={result_rank['rank']} completed; computing diversity", flush=True)
    dimensions = representation_measures(
        *pairs[0], payload, result_rank["fitted_output_spectrum"]["eigenvalues"]
    )
    fit_seconds = time.monotonic() - fit_started
    old_dimensions = json.loads((OLD_RESULTS / "rank_diversity.json").read_text())["arms"]
    if old_dimensions["a"]["layer"] != LAYER or old_dimensions["b"]["layer"] != 22:
        raise ValueError("Reused diversity layers changed")
    arms = {}
    for key, arm in (("no_think_l24", "a"), ("think_l22_previous", "b")):
        old = json.loads((OLD_RESULTS / f"rank_{arm}.json").read_text())
        if old["provenance"]["hf_revision"] != REVISION:
            raise ValueError("Reused rank result revision changed")
        arms[key] = {
            "layer": old["layer_star"],
            "rank": old["rank"],
            "r2": old["full_test_r2"],
            "rows": old["realized_rows"],
            "selected_lambda": old["selected_lambda"],
            "diversity": old_dimensions[arm],
            "layer_metrics": old["parent_selected_layer_metrics"],
            "reused": True,
        }
    arms["think_l24"] = {
        "layer": LAYER,
        "rank": result_rank["rank"],
        "r2": result_rank["full_test_r2"],
        "rows": rows,
        "selected_lambda": 1000.0,
        "diversity": dimensions,
        "layer_metrics": star,
        "reused": False,
    }
    result = {
        "schema": "issue2588_chat_matched_l24_v1",
        "status": "complete",
        "arms": arms,
        "parity": parity,
        "provenance": provenance,
        "layer_rule": "User-requested fixed L24; not a new validation-selected layer. A unchanged.",
        "rank_rule": "Training fitted-output PCA; smallest rank with validation SSE <=1.10 full SSE.",
        "compute": {
            "units": 1,
            "remaining_units": 0,
            "gpu_hours": 0,
            "provisioned": False,
            "fit_rank_diversity_seconds": fit_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "pilot_is_final_unit": True,
            "pilot_exceeds_hour": fit_seconds > 3600,
        },
        "limitations": [
            "Arms retain their own generated answers and slightly different valid rows.",
            "Matched layer removes layer choice, not mode-dependent target or input distributions.",
            "Rank is relative to each map's own error, not a common absolute prediction target.",
            "13 exact prompt strings overlap validation/test; no exact train overlap.",
            "No sampling CI on this matched-layer rank or diversity comparison.",
            "Old L22 ceilings/input-PR-at-star are not attributed to L24; per-layer metrics only.",
            "Benchmark subset results use different fitting and R2-baseline conventions.",
        ],
    }
    rank.write_json(output_dir / "result.json", result)
    print(
        json.dumps(
            {"r2": arms["think_l24"]["r2"], "rank": result_rank["rank"], **result["compute"]}
        ),
        flush=True,
    )
    return result


def main(argv=None) -> int:
    """Separate staging from the one measured production unit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("stage", "analyze"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "stage":
        cell = args.source_root / "generic" / RUN_ID / "cells_cap_long/q3_8b_b"
        before = {
            path
            for split in rank.SPLITS
            for path in (cell / "capture" / split / "L24").glob("shard*.npz")
        }
        receipt = staging.stage(args.source_root, REVISION, "b", fixed_layer=LAYER)
        receipt["new_downloads"] = [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": rank.sha256(path)}
            for split in rank.SPLITS
            for path in sorted((cell / "capture" / split / "L24").glob("shard*.npz"))
            if path not in before
        ]
        rank.write_json(args.output_dir / "staging.json", receipt)
    else:
        analyze(args.source_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
