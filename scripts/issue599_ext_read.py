"""#599 §7.3 extension-branch readers (probe trajectory + band-entry staging).

Two subcommands, called by ``scripts/run_issue599_fullresp.sh`` (kept out of
bash heredocs so the selection logic is directly smoke-testable):

- ``probe-read``: scan the seed-42 probe cell's ``periodic_eval/`` trajectory
  for the FIRST non-KILL (DG >= 5 nat AND emit >= 0.5) and FIRST FULL
  (trained logP >= -0.5 AND emit >= 0.95) checkpoints among SAVED steps
  (multiples of ``--save-steps`` with a ``checkpoint-{step}`` dir on disk).
  Writes ``<ext-dir>/probe_read.json`` with ``probe_cleared`` — the §7.3
  escalation condition.
- ``stage-band-entry``: per seed, choose the first FULL checkpoint if reached,
  else the first non-KILL checkpoint (per-seed band-entry fallback: matched
  dial position, unmatched step count — scope caveat carried), copy it to
  ``<ext-dir>/extract/marker_seed{S}/adapter`` for extraction, and record
  ``never_entered`` seeds as a reportable outcome (never staged). Writes
  ``<ext-dir>/band_entry_staging.json``; exits non-zero when NO seed entered.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DG_NON_KILL = 5.0
EMIT_NON_KILL = 0.5
LOGP_FULL = -0.5
EMIT_FULL = 0.95
SOURCE_PERSONA = "medical_doctor"


def _scan_trajectory(cell_dir: Path, save_steps: int) -> tuple[int | None, int | None, dict]:
    """Return (first_non_kill, first_full, trajectory) over SAVED checkpoints.

    Trajectory covers every periodic_eval snapshot (callback K may be finer
    than the checkpoint cadence); band-entry candidates are restricted to
    steps with an on-disk ``checkpoint-{step}`` dir so the staged adapter
    actually exists.
    """
    pe_dir = cell_dir / "periodic_eval"
    snaps = sorted(
        pe_dir.glob("leakage_marker_step_*.json"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    if not snaps:
        raise FileNotFoundError(f"no periodic_eval snapshots under {pe_dir}")
    first_non_kill: int | None = None
    first_full: int | None = None
    trajectory: dict[str, dict] = {}
    for p in snaps:
        step = int(p.stem.rsplit("_", 1)[1])
        m = json.loads(p.read_text())["metrics_by_persona"][SOURCE_PERSONA]
        dg = float(m["log_p_marker_delta"])
        emit = float(m["emit_rate"])
        logp_t = float(m["log_p_marker_trained"])
        trajectory[str(step)] = {"dg": dg, "emit": emit, "logp_trained": logp_t}
        non_kill = dg >= DG_NON_KILL and emit >= EMIT_NON_KILL
        full = logp_t >= LOGP_FULL and emit >= EMIT_FULL
        if step % save_steps == 0 and (cell_dir / f"checkpoint-{step}").exists():
            if non_kill and first_non_kill is None:
                first_non_kill = step
            if full and first_full is None:
                first_full = step
    return first_non_kill, first_full, trajectory


def cmd_probe_read(args: argparse.Namespace) -> int:
    """§7.3 probe read: did the seed-42 probe reach non-KILL by max-steps?"""
    ext_dir = Path(args.ext_dir)
    cell_dir = ext_dir / f"marker_seed{args.seed}"
    first_non_kill, first_full, trajectory = _scan_trajectory(cell_dir, args.save_steps)
    result = {
        "seed": args.seed,
        "first_non_kill_checkpoint": first_non_kill,
        "first_full_checkpoint": first_full,
        "probe_cleared": first_non_kill is not None,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "thresholds": {
            "dg_non_kill": DG_NON_KILL,
            "emit_non_kill": EMIT_NON_KILL,
            "logp_full": LOGP_FULL,
            "emit_full": EMIT_FULL,
        },
        "trajectory_summary": trajectory,
    }
    (ext_dir / "probe_read.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "trajectory_summary"}, indent=2))
    return 0


def cmd_stage_band_entry(args: argparse.Namespace) -> int:
    """§7.3 escalation staging: first FULL (preferred) / non-KILL checkpoint per seed."""
    ext_dir = Path(args.ext_dir)
    staged: dict[str, dict] = {}
    for seed in args.seeds:
        cell_dir = ext_dir / f"marker_seed{seed}"
        first_non_kill, first_full, _ = _scan_trajectory(cell_dir, args.save_steps)
        chosen = first_full if first_full is not None else first_non_kill
        if chosen is None:
            # Per-seed band-entry fallback (plan §7.3): "never enters" is a
            # reportable outcome — record it; do NOT stage this seed.
            staged[f"seed{seed}"] = {"checkpoint_step": None, "regime": "never_entered"}
            continue
        src = cell_dir / f"checkpoint-{chosen}"
        dst = ext_dir / "extract" / f"marker_seed{seed}" / "adapter"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        staged[f"seed{seed}"] = {
            "checkpoint_step": chosen,
            "regime": "full" if chosen == first_full else "non_kill",
        }
    (ext_dir / "band_entry_staging.json").write_text(json.dumps(staged, indent=2))
    print(json.dumps(staged, indent=2))
    n_staged = sum(1 for v in staged.values() if v["checkpoint_step"] is not None)
    if n_staged < 1:
        logger.error("no seed entered the non-KILL band — nothing to extract")
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="#599 extension-branch readers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("probe-read", help="first non-KILL/FULL checkpoint on the probe")
    p1.add_argument("--ext-dir", required=True)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--save-steps", type=int, default=100)
    p1.add_argument("--max-steps", type=int, default=2400)

    p2 = sub.add_parser("stage-band-entry", help="stage per-seed band-entry checkpoints")
    p2.add_argument("--ext-dir", required=True)
    p2.add_argument("--save-steps", type=int, default=100)
    p2.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 256])

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    if args.cmd == "probe-read":
        return cmd_probe_read(args)
    return cmd_stage_band_entry(args)


if __name__ == "__main__":
    sys.exit(main())
