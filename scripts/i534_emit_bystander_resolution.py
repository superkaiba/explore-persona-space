# em-dash + Qwen marker " ※" + Greek ΔG intentional
#!/usr/bin/env python3
"""Task #534 — emit per-cell ``bystander_resolution.json`` with a per-FRACTION dict.

Forked from ``scripts/i530_emit_bystander_resolution.py``. The #530 version
flattened ONLY the max-fraction checkpoint (its trajectories carried a single
band-stop checkpoint). #534's trajectories carry 4 realized-fraction
checkpoints, so this fork loops over ALL checkpoints in each
``trajectory.json`` and emits:

  * ``per_fraction``: {frac_str → the full gate payload for that checkpoint}
    — the de-saturation gate scalars, the ΔG band fractions, per-probe
    aggregates, and the raw per-pair distributions (the per-fraction
    usability gate in ``i534_trajectory_analyze.py`` reads these).
  * Top-level fields = the MAX-fraction checkpoint (kept for figure-script
    back-compat with the #530 consumers).

CPU-only, pure JSON in / JSON out. Fails loud on schema mismatch (the #530
round-3 silent-None lesson).

Run::

    uv run python scripts/i534_emit_bystander_resolution.py \\
        --slab-root eval_results/issue_534
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

LOGGER = logging.getLogger("i534_emit_bystander_resolution")

DEFAULT_SLAB_ROOT = Path("eval_results/issue_534")

# Same gate constants as #530 (comparability — plan §6 row 4).
HEADROOM_NATS = 2.0
ARGMAX_CEILING = 0.60
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
            env={**os.environ},
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


def gate_payload_for_checkpoint(
    traj: dict[str, Any],
    ck: dict[str, Any],
    *,
    src_label: str,
    headroom_nats: float = HEADROOM_NATS,
    argmax_ceiling: float = ARGMAX_CEILING,
    delta_g_lower_gate: float = DELTA_G_LOWER_GATE,
    delta_g_upper_bands: tuple[float, ...] = DELTA_G_UPPER_BANDS,
) -> dict[str, Any]:
    """Flatten ONE checkpoint's held_out grid into the gate payload.

    Same fields the #530 flattener emitted for its single checkpoint — gate
    scalars, ΔG band fractions, per-probe aggregates, raw distributions —
    PLUS the checkpoint identity and the source-self diagnostics for the
    per-fraction usability gate (mean source ΔG floor, analyzer note #3).

    Fails loud on schema mismatch; never defaults to {}.
    """
    held_out = ck.get("held_out")
    if not isinstance(held_out, dict) or not held_out:
        raise KeyError(f"{src_label}: missing or empty 'held_out'; schema mismatch.")

    per_probe: list[dict[str, Any]] = []
    per_pair_delta_g: list[float] = []
    per_pair_argmax: list[bool] = []
    per_pair_g_logp: list[float] = []
    per_pair_delta_z: list[float] = []

    for persona in sorted(held_out.keys()):
        questions = held_out[persona]
        if not isinstance(questions, dict):
            raise TypeError(
                f"{src_label} held_out[{persona!r}] is {type(questions).__name__}; expected dict."
            )
        probe_deltas: list[float] = []
        probe_argmax: list[bool] = []
        probe_g_logps: list[float] = []
        for q_text, q_row in questions.items():
            if q_row is None:
                continue
            if "delta_g" not in q_row or "argmax_marker" not in q_row:
                raise KeyError(
                    f"{src_label} held_out[{persona!r}][{q_text!r}] missing "
                    f"'delta_g' or 'argmax_marker': keys={list(q_row.keys())}"
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
            # #534 additive logit companion (absent in legacy #530 rows —
            # collected only when the eval rig captured it).
            if q_row.get("delta_z_marker") is not None:
                per_pair_delta_z.append(float(q_row["delta_z_marker"]))
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

    argmax_share_total = sum(per_pair_argmax) / n_pairs if n_pairs else None
    median_g_logp_total = _safe_median(per_pair_g_logp)
    headroom_clears = median_g_logp_total is not None and median_g_logp_total <= -headroom_nats
    argmax_below_ceiling = argmax_share_total is not None and argmax_share_total < argmax_ceiling

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

    source_self = ck.get("source_self", {}) or {}
    return {
        "checkpoint": {
            "fraction": float(ck["frac"]),
            "step": int(ck["step"]) if ck.get("step") is not None else None,
        },
        "n_held_out_probes": n_probes,
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
            "source": "same thresholds as #530 (comparability; plan #534 §6 row 4)",
        },
        "source_self": {
            "delta_g_mean": source_self.get("delta_g_mean"),
            "emission_p": source_self.get("emission_p"),
            "r_collapsed": source_self.get("r_collapsed"),
        },
        "delta_g_band_fractions": band_fractions,
        "per_probe": per_probe,
        "raw_distributions": {
            "per_pair_delta_g": per_pair_delta_g,
            "per_pair_g_logp": per_pair_g_logp,
            "per_pair_delta_z_marker": per_pair_delta_z,
        },
        "n_pairs_with_delta_z": len(per_pair_delta_z),
    }


def compute_bystander_resolution_payload(
    trajectory_json_path: Path,
    *,
    trajectory_relpath: str,
    git_commit: str = "unknown",
) -> dict[str, Any]:
    """Flatten one #534 trajectory.json into a per-fraction bystander payload.

    Returns a JSON-serializable dict; no I/O side effects. Top-level gate
    fields mirror the MAX-fraction checkpoint (#530 figure-script
    back-compat); the new ``per_fraction`` dict carries every checkpoint.
    """
    if not trajectory_json_path.is_file():
        raise FileNotFoundError(f"trajectory.json missing: {trajectory_json_path}")
    traj = json.loads(trajectory_json_path.read_text(encoding="utf-8"))

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
        raise ValueError(f"{trajectory_json_path}: checkpoints list is empty.")

    per_fraction: dict[str, dict[str, Any]] = {}
    for idx, ck in enumerate(checkpoints):
        frac_str = f"{float(ck['frac']):.2f}"
        if frac_str in per_fraction:
            raise ValueError(
                f"{trajectory_json_path}: duplicate checkpoint fraction {frac_str} "
                f"at index {idx} — the selector's index keys must be unique."
            )
        per_fraction[frac_str] = gate_payload_for_checkpoint(
            traj, ck, src_label=f"{trajectory_json_path} checkpoint[{idx}]"
        )

    # Back-compat top level = max-fraction checkpoint (the #530 consumers').
    max_frac_str = max(per_fraction, key=float)
    top = per_fraction[max_frac_str]

    payload: dict[str, Any] = {
        "sentinel_schema_version": 1,
        "kind": "i534_bystander_resolution",
        "version": 1,
        "cell": traj["cell"],
        "seed": int(traj["seed"]),
        "marker_text": traj["marker_text"],
        "marker_token_id": int(traj["marker_token_id"]),
        "chosen_checkpoint": {
            "fraction": top["checkpoint"]["fraction"],
            "step": top["checkpoint"]["step"],
            "n_checkpoints_available_in_trajectory": len(checkpoints),
            "note": (
                "#534 evaluates 4 realized fractions per cell; top-level gate "
                "fields mirror the max-fraction checkpoint for #530 "
                "figure-script back-compat. The per_fraction dict is the "
                "canonical #534 read."
            ),
        },
        "n_held_out_probes": top["n_held_out_probes"],
        "n_eval_questions_per_probe": int(traj["n_eval_questions"]),
        "n_pairs_evaluated": top["n_pairs_evaluated"],
        "n_pairs_skipped_or_missing": top["n_pairs_skipped_or_missing"],
        "de_saturation_gate": top["de_saturation_gate"],
        "delta_g_band_fractions": top["delta_g_band_fractions"],
        "per_probe": top["per_probe"],
        "raw_distributions": top["raw_distributions"],
        "per_fraction": per_fraction,
        "provenance": {
            "trajectory_json_relpath": trajectory_relpath,
            "task_id_minted_by": 534,
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--slab-root",
        type=Path,
        default=DEFAULT_SLAB_ROOT,
        help="Root dir holding c504v3_*_seed* cell dirs. Default: eval_results/issue_534",
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
        for frac_str in sorted(payload["per_fraction"], key=float):
            gate = payload["per_fraction"][frac_str]["de_saturation_gate"]
            LOGGER.info(
                "  frac=%s argmax_share=%s median_g_logp=%s gate=%s",
                frac_str,
                (
                    f"{gate['argmax_marker_share_across_pairs']:.4f}"
                    if gate["argmax_marker_share_across_pairs"] is not None
                    else "n/a"
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
