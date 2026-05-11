#!/usr/bin/env python3
"""Driver for issue #181 non-persona trigger leakage sweep.

Iterates conditions x seeds from the YAML config and calls LeakageRunner
for each. Implements Phase 0 pilot logic with GO/NO-GO criteria.

Phase 0: Train T_format seed=42, run panel eval, check pilot stats.
Phase 1: If GO, train remaining 13 runs and run panel eval on each.

Usage:
    # Full sweep (Phase 0 + Phase 1 if GO)
    nohup uv run python scripts/run_i181_sweep.py \
        --config configs/leakage/i181_non_persona_triggers.yaml \
        --gpu 0 > /workspace/logs/i181_sweep.log 2>&1 &

    # Phase 0 only (pilot)
    uv run python scripts/run_i181_sweep.py \
        --config configs/leakage/i181_non_persona_triggers.yaml \
        --gpu 0 --phase0-only

    # Skip Phase 0 (resume after pilot pass)
    uv run python scripts/run_i181_sweep.py \
        --config configs/leakage/i181_non_persona_triggers.yaml \
        --gpu 0 --skip-phase0
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_single_condition(
    sweep,
    condition,
    seed: int,
    gpu_id: int,
) -> dict:
    """Train and eval a single condition+seed using LeakageRunner."""
    from explore_persona_space.leakage.runner import LeakageRunner

    runner = LeakageRunner(
        condition=condition,
        seed=seed,
        gpu_id=gpu_id,
        project_root=PROJECT_ROOT,
        data_dir=sweep.resolve_data_dir(PROJECT_ROOT),
        output_dir=sweep.resolve_output_dir(PROJECT_ROOT),
        wandb_project=sweep.wandb_project,
        base_model=sweep.base_model,
    )
    return runner.run()


def run_panel_eval(
    config_path: str,
    run_name: str,
    gpu_id: int,
) -> dict | None:
    """Run panel eval for a single trained model.

    Calls eval_i181_panel.py as a subprocess so vLLM gets a clean process.
    Returns parsed panel eval dict or None on failure.
    """
    output_dir = PROJECT_ROOT / "eval_results" / "i181_non_persona" / run_name
    merged_path = output_dir / "train_merged"

    if not merged_path.exists():
        logger.error("Merged model not found at %s", merged_path)
        return None

    panel_eval_path = output_dir / "panel_eval.json"
    if panel_eval_path.exists():
        logger.info("Panel eval already exists for %s", run_name)
        with open(panel_eval_path) as f:
            return json.load(f)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_i181_panel.py"),
        "--model-path",
        str(merged_path),
        "--run-name",
        run_name,
        "--config",
        config_path,
        "--gpu",
        str(gpu_id),
    ]
    logger.info("Running panel eval: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        logger.error("Panel eval failed for %s:\n%s", run_name, result.stderr)
        return None

    if panel_eval_path.exists():
        with open(panel_eval_path) as f:
            return json.load(f)
    return None


def check_pilot_criteria(panel_eval: dict) -> dict:
    """Check Phase 0 GO/NO-GO criteria from pilot panel eval.

    Returns dict with:
        go: bool
        r_match: float -- matched cell marker rate
        r_bystander: float -- mean bystander+control marker rate
        ratio: float -- r_match / r_bystander
        confidence_cap: str | None -- "MODERATE" if soft warning criteria met
        reason: str -- human-readable explanation
    """
    results = panel_eval.get("results", {})

    # Find matched cell for T_format
    matched_rate = None
    bystander_rates = []
    control_rates = []

    for cell_id, cell_data in results.items():
        bucket = cell_data.get("bucket", "")
        rate = cell_data.get("marker_rate", 0.0)

        if cell_id == "match_T_format":
            matched_rate = rate
        elif bucket in ("cross_family_bystander",):
            bystander_rates.append(rate)
        elif bucket in ("control_empty", "control_default"):
            control_rates.append(rate)

    if matched_rate is None:
        return {
            "go": False,
            "r_match": 0.0,
            "r_bystander": 0.0,
            "ratio": 0.0,
            "confidence_cap": None,
            "reason": "Could not find match_T_format in panel eval results",
        }

    # Bystander = cross-family bystanders + controls
    all_bystander = bystander_rates + control_rates
    r_bystander = sum(all_bystander) / len(all_bystander) if all_bystander else 0.0
    ratio = matched_rate / r_bystander if r_bystander > 0 else float("inf")

    result = {
        "r_match": matched_rate,
        "r_bystander": r_bystander,
        "ratio": ratio,
        "confidence_cap": None,
        "go": True,
        "reason": "",
    }

    # Hard kill: r_match < 5%
    if matched_rate < 0.05:
        result["go"] = False
        result["reason"] = (
            f"HARD KILL: r_match={matched_rate:.1%} < 5%. "
            "Training recipe too weak or marker not learning."
        )
        return result

    # Soft warning: 5% <= r_match < 15%
    if matched_rate < 0.15:
        result["confidence_cap"] = "MODERATE"
        result["reason"] = (
            f"SOFT WARNING: r_match={matched_rate:.1%} (5-15% range). "
            "Proceeding with confidence cap MODERATE."
        )

    # Gradient check: r_match >= 2x r_bystander
    if ratio < 2.0:
        if result["confidence_cap"] is None:
            result["confidence_cap"] = "MODERATE"
        result["reason"] += (
            f" Gradient weak: ratio={ratio:.1f}x (<2x). Confidence capped at MODERATE."
        )

    # GO criteria: r_match >= 15% AND ratio >= 2x
    if matched_rate >= 0.15 and ratio >= 2.0:
        result["reason"] = f"GO: r_match={matched_rate:.1%} >= 15%, ratio={ratio:.1f}x >= 2x."
    elif matched_rate >= 0.05 and not result["reason"]:
        result["reason"] = f"GO with caveats: r_match={matched_rate:.1%}, ratio={ratio:.1f}x."

    return result


def _run_phase0(sweep, args, run_log, run_log_path, output_dir) -> bool:
    """Run Phase 0 pilot. Returns True if GO, exits on NO-GO."""
    logger.info("=" * 70)
    logger.info("PHASE 0: Pilot — T_format seed=42")
    logger.info("=" * 70)

    pilot_cond = None
    for cond in sweep.conditions:
        if cond.name == "T_format" and 42 in cond.seeds:
            pilot_cond = cond
            break

    if pilot_cond is None:
        logger.error("Could not find T_format condition with seed=42")
        sys.exit(1)

    logger.info("Training pilot: T_format seed=42")
    try:
        run_single_condition(sweep, pilot_cond, 42, args.gpu)
        run_log["completed_runs"].append("T_format_seed42")
    except Exception as e:
        logger.error("Pilot training failed: %s", e)
        run_log["failed_runs"].append({"run": "T_format_seed42", "error": str(e)})
        run_log["phase0"]["status"] = "FAIL"
        _save_run_log(run_log_path, run_log)
        sys.exit(1)

    logger.info("Running panel eval on pilot model...")
    panel_eval = run_panel_eval(args.config, "T_format_seed42", args.gpu)

    if panel_eval is None:
        logger.error("Panel eval failed for pilot")
        run_log["phase0"]["status"] = "FAIL"
        run_log["phase0"]["error"] = "Panel eval failed"
        _save_run_log(run_log_path, run_log)
        sys.exit(1)

    pilot_check = check_pilot_criteria(panel_eval)
    run_log["phase0"] = {
        "status": "GO" if pilot_check["go"] else "NO-GO",
        "r_match": pilot_check["r_match"],
        "r_bystander": pilot_check["r_bystander"],
        "ratio": pilot_check["ratio"],
        "reason": pilot_check["reason"],
        "confidence_cap": pilot_check["confidence_cap"],
    }

    if pilot_check["confidence_cap"]:
        run_log["confidence_cap"] = pilot_check["confidence_cap"]

    logger.info("Phase 0 result: %s", pilot_check["reason"])

    if not pilot_check["go"]:
        logger.error("Phase 0 NO-GO. Exiting.")
        run_log["phase0"]["status"] = "NO-GO"
        _save_run_log(run_log_path, run_log)

        fail_context = {
            "marker": "epm:phase0-fail",
            "version": 1,
            "pilot_stats": pilot_check,
            "recommendation": "Consider bumping epochs or lr. Requires user approval.",
        }
        fail_path = output_dir / "phase0_fail.json"
        with open(fail_path, "w") as f:
            json.dump(fail_context, f, indent=2)
        logger.info("Phase 0 fail context written to %s", fail_path)
        sys.exit(1)

    _save_run_log(run_log_path, run_log)
    return True


def _run_phase1(sweep, args, run_log, run_log_path, output_dir, skip_phase0: bool):
    """Run Phase 1: remaining 13 runs."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Remaining 13 runs")
    logger.info("=" * 70)

    runs_to_do: list[tuple[str, int]] = []
    for cond in sweep.conditions:
        for seed in cond.seeds:
            run_name = cond.run_name(seed)
            if run_name == "T_format_seed42" and not skip_phase0:
                continue
            result_path = output_dir / run_name / "run_result.json"
            if result_path.exists():
                logger.info("Skipping %s (already complete)", run_name)
                run_log["completed_runs"].append(run_name)
                continue
            runs_to_do.append((cond.name, seed))

    logger.info("Phase 1: %d runs to execute", len(runs_to_do))

    for i, (cond_name, seed) in enumerate(runs_to_do):
        cond = None
        for c in sweep.conditions:
            if c.name == cond_name and seed in c.seeds:
                cond = c
                break
        if cond is None:
            logger.error("Condition %s with seed %d not found", cond_name, seed)
            continue

        run_name = cond.run_name(seed)
        logger.info("\n--- Run %d/%d: %s ---", i + 1, len(runs_to_do), run_name)

        try:
            run_single_condition(sweep, cond, seed, args.gpu)
            run_log["completed_runs"].append(run_name)
            logger.info("Training complete for %s", run_name)
        except Exception as e:
            logger.error("Training failed for %s: %s", run_name, e)
            run_log["failed_runs"].append({"run": run_name, "error": str(e)})
            _save_run_log(run_log_path, run_log)
            continue

        logger.info("Running panel eval for %s...", run_name)
        panel_eval = run_panel_eval(args.config, run_name, args.gpu)
        if panel_eval is None:
            logger.warning("Panel eval failed for %s, continuing", run_name)

        _save_run_log(run_log_path, run_log)


def main():
    parser = argparse.ArgumentParser(description="Run issue #181 non-persona trigger sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/leakage/i181_non_persona_triggers.yaml",
        help="Path to sweep YAML config",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--phase0-only", action="store_true", help="Run Phase 0 pilot only")
    parser.add_argument("--skip-phase0", action="store_true", help="Skip Phase 0, run Phase 1")
    args = parser.parse_args()

    from explore_persona_space.leakage.config import load_sweep

    sweep = load_sweep(args.config)
    output_dir = sweep.resolve_output_dir(PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_log_path = output_dir / "run_log.json"
    run_log = {
        "started_at": datetime.now(UTC).isoformat(),
        "config": args.config,
        "git_commit": get_git_commit(),
        "gpu_id": args.gpu,
        "phase0": {},
        "phase1": {},
        "completed_runs": [],
        "failed_runs": [],
        "confidence_cap": None,
    }

    t_start = time.time()

    if not args.skip_phase0:
        _run_phase0(sweep, args, run_log, run_log_path, output_dir)
        if args.phase0_only:
            logger.info("Phase 0 only mode. Exiting after pilot.")
            return

    _run_phase1(sweep, args, run_log, run_log_path, output_dir, args.skip_phase0)

    t_total = time.time() - t_start
    run_log["completed_at"] = datetime.now(UTC).isoformat()
    run_log["total_duration_s"] = t_total
    run_log["phase1"]["n_completed"] = len(run_log["completed_runs"])
    run_log["phase1"]["n_failed"] = len(run_log["failed_runs"])
    _save_run_log(run_log_path, run_log)

    logger.info("=" * 70)
    logger.info(
        "SWEEP COMPLETE: %d/%d runs succeeded in %.1f min",
        len(run_log["completed_runs"]),
        len(run_log["completed_runs"]) + len(run_log["failed_runs"]),
        t_total / 60,
    )
    if run_log["failed_runs"]:
        logger.warning("Failed runs: %s", [r["run"] for r in run_log["failed_runs"]])
    logger.info("=" * 70)


def _save_run_log(path: Path, log: dict) -> None:
    """Atomically save the run log."""
    from explore_persona_space.utils import save_json_atomic

    save_json_atomic(path, log)


if __name__ == "__main__":
    main()
