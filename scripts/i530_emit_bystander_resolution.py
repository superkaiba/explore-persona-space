# em-dash + Qwen marker " ※" + Greek ΔG intentional
#!/usr/bin/env python3
"""Task #530 — emit per-cell ``bystander_resolution.json`` from ``trajectory.json``.

Plan §6.5 declares a third deliverable beyond ``trajectory.json`` and
``analysis_v1.json``::

    dv: per-cell bystander-resolution diagnostic
      glob: eval_results/issue_530/cells/c504v3_*/bystander_resolution.json
      note: 10 files; required for the de-saturation magnitude read

The pod-side eval (``i504_eval_trajectory.py``) embeds the underlying
held-out per-(probe, question) measurements inside ``trajectory.json`` at
``checkpoints[*].held_out.<persona>.<question>.{delta_g, argmax_marker,
g_logp, b_logp, kl}`` — but never lifts them into the standalone
artifact the plan §6.5 row requires.  This script is the missing
flattener.  CPU-only, no GPU, no model load; pure JSON in / JSON out.

The de-saturation gate the file feeds (plan §-top-line spec line 24) is::

    bystander argmax-marker fraction < 60%  AND
    median |log P(marker)| ≥ ~2 nats headroom below 0

Plan §4.4 step 4 separately mentions a "fraction of probes with
ΔG ∈ (+0.5, log(0.9))" band as auxiliary; ``log(0.9) ≈ -0.105`` so that
literal interval is empty.  We interpret the intent as "ΔG in the
de-saturated headroom band, i.e. positive ΔG but bounded above so a
fully-saturated probe doesn't count" and report fractions over three
candidate bands plus the raw distribution.  Analyzer Step 9a picks the
band it cites; this script does not commit to one.

For each cell, the emitted JSON carries (a) the chosen checkpoint
fraction (always 1.0 here — only the band-stop final checkpoint was
evaluated, see plan-deviation in epm:results), (b) per-probe ΔG and
argmax-marker shares aggregated across the 10 eval questions, (c)
per-(probe, question) ΔG list for downstream analysis, and (d) the two
de-saturation gate scalars.

Run::

    uv run python scripts/i530_emit_bystander_resolution.py \\
        --slab-root eval_results/issue_530
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("i530_emit_bystander_resolution")

# Plan §6.5 fenced YAML default; can be overridden on the CLI.
DEFAULT_SLAB_ROOT = Path("eval_results/issue_530")

# Per plan-top-line §spec line 24: median bystander |log P(marker)| ≥ ~2 nats
# headroom below 0  ⇔  median log P(marker) ≤ -2.0.
HEADROOM_NATS = 2.0
ARGMAX_CEILING = 0.60

# Plan §4.4 step 4 "ΔG ∈ (+0.5, log(0.9))" band is mathematically empty (lower
# > upper), so we report fractions for three candidate upper bounds and let the
# analyzer cite whichever it argues for.  These three are anchored in the
# project's standing nats-vs-prob heuristics:
#   - 0.5 nat → strong signal floor (the source-band lower bound's downstream
#     analog)
#   - 5.0 nat → the source band lower (matches `marker_band_low_nats` default)
#   - 12.0 nat → the source band upper (matches `marker_band_high_nats` default)
# A "de-saturated headroom band" reads as "positive ΔG, but not yet saturated";
# the inner two bounds cover the realistic interpretation.
DELTA_G_LOWER_GATE = 0.5
DELTA_G_UPPER_BANDS = (5.0, 12.0, math.inf)


def _resolve_repo_root(script: Path) -> Path:
    """Return repo root by walking up from this script (`scripts/<name>.py`)."""

    return script.resolve().parents[1]


def _git_commit_sha(repo_root: Path) -> str:
    """Return ``git rev-parse HEAD`` or ``unknown`` if not in a git repo."""

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _iter_cell_dirs(slab_root: Path) -> list[Path]:
    """Return the list of cell dirs (``c504v3_*_seed*``) under ``slab_root``."""

    if not slab_root.is_dir():
        raise FileNotFoundError(f"slab_root does not exist: {slab_root}")
    cells = sorted(p for p in slab_root.glob("c504v3_*_seed*") if p.is_dir())
    if not cells:
        raise FileNotFoundError(
            f"no c504v3_*_seed* cell dirs found under {slab_root}; expected 10."
        )
    return cells


def _safe_median(values: list[float]) -> float | None:
    """Median of non-empty list, ``None`` when empty (so JSON stays explicit)."""

    if not values:
        return None
    return float(statistics.median(values))


def _fraction(numerator: int, denominator: int) -> float | None:
    """Safe fraction; ``None`` if denominator is zero (per-cell unknowns)."""

    if denominator == 0:
        return None
    return numerator / denominator


def compute_bystander_resolution_payload(
    trajectory_json_path: Path,
    *,
    trajectory_relpath: str,
    headroom_nats: float = HEADROOM_NATS,
    argmax_ceiling: float = ARGMAX_CEILING,
    delta_g_lower_gate: float = DELTA_G_LOWER_GATE,
    delta_g_upper_bands: tuple[float, ...] = DELTA_G_UPPER_BANDS,
    git_commit: str = "unknown",
) -> dict[str, Any]:
    """Flatten one trajectory.json into a bystander-resolution payload.

    Reads the embedded ``checkpoints[chosen].held_out.<persona>.<question>``
    structure and returns a dict ready to ``json.dump`` to disk.  Fails
    loud when the schema does not match the eval-rig contract — silent
    None branches were the #530 round-3 failure mode (verifier
    feedback ``eval_script_silent_not_present_misdiagnosis``).

    Returns a JSON-serializable dict; no I/O side effects.
    """

    if not trajectory_json_path.is_file():
        raise FileNotFoundError(f"trajectory.json missing: {trajectory_json_path}")

    with trajectory_json_path.open("r", encoding="utf-8") as fh:
        traj = json.load(fh)

    # Schema sanity — fail loud, do NOT default to {}.
    required_top = {
        "cell",
        "seed",
        "marker_text",
        "marker_token_id",
        "n_held_out_personas",
        "held_out_personas",
        "n_eval_questions",
        "checkpoints",
    }
    missing = required_top - traj.keys()
    if missing:
        raise KeyError(
            f"{trajectory_json_path}: trajectory.json missing required keys "
            f"{sorted(missing)}; schema mismatch (expected i504_eval_trajectory layout)."
        )

    checkpoints = traj["checkpoints"]
    if not checkpoints:
        raise ValueError(
            f"{trajectory_json_path}: checkpoints list is empty; no checkpoint to flatten."
        )

    # Plan §4.4 step 4 pins the headline read to the band-stop final checkpoint
    # (frac=1.00).  Only one checkpoint was evaluated in this run (carry-over
    # caveat); we record which one we chose.
    chosen_idx = max(
        range(len(checkpoints)),
        key=lambda i: float(checkpoints[i].get("frac", -1.0)),
    )
    ck = checkpoints[chosen_idx]
    chosen_fraction = float(ck["frac"])
    chosen_step = int(ck["step"])

    held_out = ck.get("held_out")
    if not isinstance(held_out, dict) or not held_out:
        raise KeyError(
            f"{trajectory_json_path} checkpoint[{chosen_idx}]: "
            f"missing or empty 'held_out'; schema mismatch."
        )

    per_probe: list[dict[str, Any]] = []
    per_pair_delta_g: list[float] = []
    per_pair_argmax: list[bool] = []
    per_pair_g_logp: list[float] = []

    for persona in sorted(held_out.keys()):
        questions = held_out[persona]
        if not isinstance(questions, dict):
            raise TypeError(
                f"{trajectory_json_path} held_out[{persona!r}] is "
                f"{type(questions).__name__}; expected dict."
            )

        probe_deltas: list[float] = []
        probe_argmax: list[bool] = []
        probe_g_logps: list[float] = []

        for q_text, q_row in questions.items():
            if q_row is None:
                # Eval skipped this (probe, question) — record but do not
                # silently treat as zero.  Bookkeeping should make this explicit
                # in n_skipped below.
                continue
            if "delta_g" not in q_row or "argmax_marker" not in q_row:
                raise KeyError(
                    f"{trajectory_json_path} held_out[{persona!r}][{q_text!r}] "
                    f"missing 'delta_g' or 'argmax_marker': keys={list(q_row.keys())}"
                )
            dg = float(q_row["delta_g"])
            am = bool(q_row["argmax_marker"])
            probe_deltas.append(dg)
            probe_argmax.append(am)
            per_pair_delta_g.append(dg)
            per_pair_argmax.append(am)
            if "g_logp" in q_row and q_row["g_logp"] is not None:
                g = float(q_row["g_logp"])
                probe_g_logps.append(g)
                per_pair_g_logp.append(g)

        per_probe.append(
            {
                "persona": persona,
                "n_questions_evaluated": len(probe_deltas),
                "mean_delta_g": (float(statistics.fmean(probe_deltas)) if probe_deltas else None),
                "median_delta_g": _safe_median(probe_deltas),
                "argmax_marker_share": (
                    sum(probe_argmax) / len(probe_argmax) if probe_argmax else None
                ),
                "median_g_logp": _safe_median(probe_g_logps),
            }
        )

    n_probes = len(per_probe)
    n_pairs = len(per_pair_delta_g)
    n_expected_pairs = int(traj["n_held_out_personas"]) * int(traj["n_eval_questions"])
    n_skipped = n_expected_pairs - n_pairs

    # De-saturation gate scalars (the plan-§spec-line-24 read).
    argmax_share_total = sum(per_pair_argmax) / n_pairs if n_pairs else None
    median_g_logp_total = _safe_median(per_pair_g_logp)
    headroom_clears = median_g_logp_total is not None and median_g_logp_total <= -headroom_nats
    argmax_below_ceiling = argmax_share_total is not None and argmax_share_total < argmax_ceiling

    # Plan §4.4 step 4 "headroom-band fraction" — three candidate upper bounds.
    band_fractions: dict[str, dict[str, Any]] = {}
    for upper in delta_g_upper_bands:
        upper_label = "inf" if math.isinf(upper) else f"{upper:.2f}"
        key = f"delta_g_in_({delta_g_lower_gate:.2f},{upper_label}]"
        n_in_band = sum(1 for dg in per_pair_delta_g if dg > delta_g_lower_gate and dg <= upper)
        band_fractions[key] = {
            "lower_exclusive": delta_g_lower_gate,
            "upper_inclusive": None if math.isinf(upper) else upper,
            "n_pairs_in_band": n_in_band,
            "fraction_of_pairs": _fraction(n_in_band, n_pairs),
        }

    payload: dict[str, Any] = {
        "sentinel_schema_version": 1,
        "kind": "i530_bystander_resolution",
        "version": 1,
        "cell": traj["cell"],
        "seed": int(traj["seed"]),
        "marker_text": traj["marker_text"],
        "marker_token_id": int(traj["marker_token_id"]),
        "chosen_checkpoint": {
            "fraction": chosen_fraction,
            "step": chosen_step,
            "n_checkpoints_available_in_trajectory": len(checkpoints),
            "note": (
                "Plan §6.5 declared 4 fractions {0.25, 0.5, 0.75, 1.00} + final; "
                "MarkerBandStopCallback halted training before any sub-final "
                "checkpoint could land, so n_checkpoints_per_cell=1 (the final "
                "band-stop checkpoint at frac=1.00). See epm:results §plan_deviations."
            ),
        },
        "n_held_out_probes": n_probes,
        "n_eval_questions_per_probe": int(traj["n_eval_questions"]),
        "n_pairs_evaluated": n_pairs,
        "n_pairs_skipped_or_missing": n_skipped,
        "de_saturation_gate": {
            "headroom_nats_required": headroom_nats,
            "argmax_ceiling": argmax_ceiling,
            "median_g_logp_at_post_response_slot": median_g_logp_total,
            "median_g_logp_clears_headroom": headroom_clears,
            "argmax_marker_share_across_pairs": argmax_share_total,
            "argmax_below_ceiling": argmax_below_ceiling,
            "verdict": "PASS" if (headroom_clears and argmax_below_ceiling) else "FAIL",
            "source": "plan §-top-line-spec line 24 + plan §4.4 step 4",
        },
        "delta_g_band_fractions": band_fractions,
        "per_probe": per_probe,
        "raw_distributions": {
            "per_pair_delta_g": per_pair_delta_g,
            "per_pair_g_logp": per_pair_g_logp,
        },
        "provenance": {
            "trajectory_json_relpath": trajectory_relpath,
            "task_id_minted_by": 530,
            "git_commit": git_commit,
            "host": socket.gethostname(),
            "emitted_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "script": Path(__file__).name,
        },
    }
    return payload


def _emit_for_cell(
    cell_dir: Path,
    *,
    slab_root: Path,
    git_commit: str,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    """Emit ``bystander_resolution.json`` for one cell dir; return (path, payload)."""

    traj_path = cell_dir / "trajectory.json"
    out_path = cell_dir / "bystander_resolution.json"

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists; re-run with --overwrite to replace.")

    rel = traj_path.relative_to(slab_root.parent.resolve())
    payload = compute_bystander_resolution_payload(
        traj_path,
        trajectory_relpath=str(rel),
        git_commit=git_commit,
    )

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return out_path, payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; returns process exit code (0 on full success)."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slab-root",
        type=Path,
        default=DEFAULT_SLAB_ROOT,
        help=("Root dir holding c504v3_*_seed* cell dirs. Default: eval_results/issue_530"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing bystander_resolution.json files in each cell dir.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging level (default INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    repo_root = _resolve_repo_root(Path(__file__))
    git_commit = _git_commit_sha(repo_root)

    slab_root = args.slab_root.resolve()
    LOGGER.info("[phase=enumerate] scanning %s for cells", slab_root)

    cells = _iter_cell_dirs(slab_root)
    LOGGER.info("[phase=enumerate] found %d cell dirs", len(cells))

    written: list[Path] = []
    for cell_dir in cells:
        LOGGER.info("[phase=emit] %s", cell_dir.name)
        out_path, payload = _emit_for_cell(
            cell_dir,
            slab_root=slab_root,
            git_commit=git_commit,
            overwrite=args.overwrite,
        )
        gate = payload["de_saturation_gate"]
        LOGGER.info(
            "  -> %s  argmax_share=%.4f  median_g_logp=%s  gate=%s",
            out_path.relative_to(
                slab_root.parent.resolve() if slab_root.parent.exists() else slab_root
            ),
            (
                gate["argmax_marker_share_across_pairs"]
                if gate["argmax_marker_share_across_pairs"] is not None
                else float("nan")
            ),
            (
                f"{gate['median_g_logp_at_post_response_slot']:.4f}"
                if gate["median_g_logp_at_post_response_slot"] is not None
                else "n/a"
            ),
            gate["verdict"],
        )
        written.append(out_path)

    LOGGER.info("[phase=done] wrote %d bystander_resolution.json files", len(written))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
