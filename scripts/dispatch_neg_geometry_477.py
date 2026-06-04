# ruff: noqa: RUF001, RUF002, RUF003  # docstring ASCII-art + em-dash + Qwen marker token " ※" + Greek + × all intentional
#!/usr/bin/env python3
"""Task #477 dispatcher — implant-decoupled count sweep.

Fork of scripts/dispatch_neg_geometry_472.py. Pipeline (plan §4):
  Phase 0     persona bank   [REUSE: data/issue_472/persona_bank.json]
  Phase 0.5   centroids      [REUSE: data/issue_472/centroids_L{10,15,20}.pt]
  Phase 1     base on-policy R   [REUSE: superkaiba1/...-data/issue472_neg_geometry/on_policy_R/]
  Phase 1.5   base panel prior   [REUSE: superkaiba1/...-data/issue472_neg_geometry/base_panel/]
  Phase 2     CALIBRATION SWEEP (NEW): 20 cells (4 counts × 5 LRs × seed=42),
              terminal-only eval. Writes calibration_table.json incrementally.
  Phase 2.5   CALIBRATION PICK (NEW): for each count level, pick LR landing
              source ΔG ∈ [10.5, 13.5] AND P(※)≥0.30. Fail loud on §7 kill
              criterion. Writes calibration_pick.json.
  Phase 3     MAIN CELLS (8 cells, 4 counts × 2 seeds) at calibrated LRs, full
              trajectory eval.
  Phase 4     IMPLANT-ONLY-AXIS (6 cells, 3 LRs × 2 seeds) at fixed count=4,
              full trajectory eval.
  Phase 5     ANALYZE (skipped here; run scripts/i477_phase_analyze.py post-hoc).

UNIFICATION (smoke-architecture parity = PASS_UNIFIED): smoke = this dispatcher
with --smoke; that resolves to a single calibration cell (c477_calib_negp_4,
seed=42, lr=1e-5) + the tiny --smoke slice. Same subprocess shape (scripts/
i477_run_cell.py), same env injection, same nested-eval teardown, same
sentinel discipline as the full sweep. The smoke cell IS the calibration
sweep with one cell.

Pod-side discipline (CLAUDE.md): NEVER shells out to scripts/task.py
(sentinel-file pattern only); every subprocess.* passes env={**os.environ};
load_dotenv() at module top; vLLM phases are subprocess-isolated; sets
EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 (MooseFS quota) + EPM_PERSIST_ADAPTER_HF_REPO
so the adapter persists fail-loud before any cleanup.

GPU pinning (round-3 #472 fix preserved): each cell-subprocess gets a DISTINCT
free physical GPU via --gpu-id; train/sft.py SETS CVD=str(gpu_id), the nested
eval inherits the same CVD via os.environ.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TASK_ID = 477
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
DEFAULT_SEEDS = (42, 137)

LOG_DIR = Path("/workspace/logs")  # overridden by --log-dir.

log = logging.getLogger("dispatch_neg_geometry_477")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _write_sentinel(path: Path, *, kind: str, phase: str, note_payload: dict) -> None:
    """poll_pipeline.py-compliant sentinel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": kind,
                "version": 1,
                "task_id": TASK_ID,
                "by": "dispatch_neg_geometry_477",
                "ts": datetime.now(UTC).isoformat(),
                "phase": phase,
                "note": json.dumps(note_payload),
            },
            indent=2,
        )
    )


def _run_phase_subprocess(cmd: list[str], phase: str) -> None:
    log.info("[phase=%s] subprocess: %s", phase, " ".join(cmd))
    subprocess.run(cmd, env={**os.environ}, check=True)


# ── Cell-pool scheduler (verbatim shape from #472; #477 schedules (cell, seed,
# lr, phase) units instead of (cell, seed)). ─────────────────────────────────


def _schedule_unit_pool(  # noqa: C901 - linear GPU-sharded subprocess scheduler
    *,
    units: list[dict],
    n_gpus: int,
    max_parallel: int,
    slab_root: Path,
    runs_root: Path,
    log_dir: Path,
    bank_path: Path,
    centroids_dir: Path,
    smoke: bool,
    no_kl: bool,
    report_to: str,
    resume: bool,
) -> list[dict]:
    """Run all (cell, seed, lr, phase) units as a GPU-sharded subprocess pool.

    units: list of {"cell": str, "seed": int, "lr": float, "phase": str}.
    Each unit launches scripts/i477_run_cell.py on a DISTINCT free physical GPU.
    max_parallel is clamped to n_gpus (one-GPU cells, no sharing).
    """
    if max_parallel > n_gpus:
        log.warning(
            "max_parallel=%d > n_gpus=%d would force ≥2 concurrent cells onto one GPU "
            "(round-3 #472 OOM class); clamping max_parallel to %d.",
            max_parallel,
            n_gpus,
            n_gpus,
        )
        max_parallel = n_gpus
    log.info(
        "Scheduling %d units across %d GPUs (max_parallel=%d)",
        len(units),
        n_gpus,
        max_parallel,
    )

    results: list[dict] = []
    running: list[tuple[subprocess.Popen, dict, int]] = []  # (proc, unit, gpu)
    queue = list(units)
    free_gpus: list[int] = list(range(n_gpus))

    def _launch(unit: dict, gpu: int) -> subprocess.Popen | None:
        cell = unit["cell"]
        seed = unit["seed"]
        lr = unit["lr"]
        phase = unit["phase"]
        run_label = f"{cell}_seed{seed}_lr{lr:g}"
        summary_path = runs_root / run_label / "cell_summary.json"
        if resume and summary_path.exists():
            log.info("[%s] RESUME: cell_summary exists; skipping.", run_label)
            return None
        env = {**os.environ}
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/i477_run_cell.py",
            "--cell",
            cell,
            "--seed",
            str(seed),
            "--lr",
            f"{lr:g}",
            "--phase",
            phase,
            "--gpu-id",
            str(gpu),
            "--slab-root",
            str(slab_root),
            "--runs-root",
            str(runs_root),
            "--log-dir",
            str(log_dir),
            "--bank-path",
            str(bank_path),
            "--centroids-dir",
            str(centroids_dir),
            "--report-to",
            report_to,
        ]
        # v4 step-lever threads: --target-steps for step_calibration cells,
        # --picked-step for main_v4 cells, --implant-steps for the v4r2
        # implant_sweep_v4 anchor. Optional on the unit dict so legacy phases
        # keep the v2 byte-identical command shape.
        if unit.get("target_steps"):
            cmd.extend(["--target-steps", str(unit["target_steps"])])
        if unit.get("picked_step") is not None:
            cmd.extend(["--picked-step", str(int(unit["picked_step"]))])
        if unit.get("implant_steps"):
            cmd.extend(["--implant-steps", str(unit["implant_steps"])])
        # v6 rank pivot threads: --lora-rank / --lora-alpha.
        # Optional; default-None means the worker uses the module constants
        # (r=32 / α=64, i.e. v4 byte-identical). v6 M3 pins positives=200
        # globally via POS_EX_PER_SOURCE; not threaded per-cell (build_cell
        # reads the module constant, so a per-unit override would be a no-op).
        if unit.get("lora_r") is not None:
            cmd.extend(["--lora-rank", str(int(unit["lora_r"]))])
        if unit.get("lora_alpha") is not None:
            cmd.extend(["--lora-alpha", str(int(unit["lora_alpha"]))])
        if smoke:
            cmd.append("--smoke")
        if no_kl:
            cmd.append("--no-kl")
        cell_log = log_dir / f"issue-477-{run_label}.log"
        cell_log.parent.mkdir(parents=True, exist_ok=True)
        log.info("[%s] launch on GPU %d → %s", run_label, gpu, cell_log)
        # The file handle outlives this function (Popen writes to it while
        # running); the OS closes it on the child's exit.
        fh = open(cell_log, "w")  # noqa: SIM115 -- handle lives for the Popen's lifetime
        return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)

    while queue or running:
        while queue and len(running) < max_parallel and free_gpus:
            unit = queue.pop(0)
            gpu = free_gpus.pop(0)
            proc = _launch(unit, gpu)
            if proc is None:
                # RESUME-SKIP path: the cell already wrote cell_summary.json on a
                # prior run. Load it so the appended result carries the SAME
                # ΔG/emission keys as a freshly-`done` cell — _phase_calibration_pick
                # KeyErrors otherwise on `source_self_delta_g_at_last_ckpt` /
                # `source_emission_p_at_last_ckpt` (this is the §7 kill-criterion
                # recovery path).
                run_label = f"{unit['cell']}_seed{unit['seed']}_lr{unit['lr']:g}"
                summary_path = runs_root / run_label / "cell_summary.json"
                if not summary_path.exists():
                    raise RuntimeError(
                        f"[{run_label}] resume-skip path but cell_summary.json "
                        f"missing at {summary_path} — _launch returned None only "
                        f"when this file exists. Inconsistent state; investigate."
                    )
                cs = json.loads(summary_path.read_text())
                results.append({**unit, "assigned_gpu": gpu, "status": "resumed_skip", **cs})
                free_gpus.append(gpu)
                continue
            running.append((proc, unit, gpu))
        still: list[tuple[subprocess.Popen, dict, int]] = []
        for proc, unit, gpu in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, unit, gpu))
                continue
            free_gpus.append(gpu)
            run_label = f"{unit['cell']}_seed{unit['seed']}_lr{unit['lr']:g}"
            if rc != 0:
                fail_path = log_dir / f"issue-477-{run_label}-FAILED.json"
                fail_path.write_text(
                    json.dumps({**unit, "returncode": rc, "assigned_gpu": gpu}, indent=2)
                )
                for p2, _u2, _g2 in still:
                    p2.terminate()
                raise RuntimeError(
                    f"[{run_label}] cell subprocess exited rc={rc} (GPU {gpu}). "
                    f"See {log_dir}/issue-477-{run_label}.log. Sweep aborted."
                )
            log.info("[%s] DONE (GPU %d)", run_label, gpu)
            # Read back the cell_summary (the worker wrote it).
            summary_path = runs_root / run_label / "cell_summary.json"
            if summary_path.exists():
                cs = json.loads(summary_path.read_text())
            else:
                # Should never happen — the worker only exits 0 after writing.
                raise RuntimeError(
                    f"[{run_label}] cell exited rc=0 but cell_summary.json missing at "
                    f"{summary_path}. Investigate."
                )
            results.append({**unit, "assigned_gpu": gpu, "status": "done", **cs})
        running = still
        if running:
            time.sleep(5)
    return results


# ── Phase 2.5: pick wrapper ──────────────────────────────────────────────────


def _phase_calibration_pick(
    calibration_results: list[dict],
    calibration_pick_path: Path,
) -> dict[int, dict]:
    """Walk calibration results, build the per-count LR pick. Fail loud on §7."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        COUNT_LEVELS,
    )
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_lr_for_count,
    )

    # Build the {count: {lr: {source_self_delta_g, source_emission_p}}} table.
    table: dict[int, dict[float, dict]] = {}
    for r in calibration_results:
        if r.get("phase") != "calibration":
            continue
        # Derive count from cell slug: c477_calib_negp_<count>.
        from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
            count_for_slug,
        )

        cnt = count_for_slug(r["cell"])
        table.setdefault(cnt, {})[float(r["lr"])] = {
            "source_self_delta_g": float(r["source_self_delta_g_at_last_ckpt"]),
            "source_emission_p": float(r["source_emission_p_at_last_ckpt"]),
            "cell_slug": r["cell"],
            "seed": int(r["seed"]),
            "lr": float(r["lr"]),
        }

    # Persist the full table BEFORE picking (checkpoint-per-phase: even if the
    # pick step crashes the analyzer can still re-pick from the table).
    table_path = calibration_pick_path.parent / "calibration_table.json"
    # Convert to str keys for stable JSON (counts as str(int), lrs as f"{lr:g}").
    serial = {
        str(cnt): {f"{lr:g}": entry for lr, entry in row.items()} for cnt, row in table.items()
    }
    table_path.write_text(json.dumps(serial, indent=2))
    log.info("[phase=calibration_pick] wrote calibration_table.json → %s", table_path)

    picks: dict[int, dict] = {}
    failures: dict[int, str] = {}
    for cnt in COUNT_LEVELS:
        if cnt not in table:
            failures[cnt] = (
                f"count={cnt} missing from calibration results — calibration "
                f"sweep did not produce any cells at this count level."
            )
            continue
        try:
            picks[cnt] = pick_lr_for_count(table, cnt)
        except RuntimeError as e:
            failures[cnt] = str(e)

    if failures:
        # Persist whatever picks succeeded (post-mortem evidence) BEFORE raising.
        calibration_pick_path.write_text(
            json.dumps(
                {"picks": {str(k): v for k, v in picks.items()}, "failures": failures}, indent=2
            )
        )
        raise RuntimeError(
            "Calibration pick FAILED on the §7 kill criterion for "
            f"{sorted(failures)}. Failures:\n"
            + "\n".join(f"  count={k}: {v}" for k, v in failures.items())
            + "\nExpand the LR grid or pivot to a different decoupling mechanism."
        )

    calibration_pick_path.write_text(
        json.dumps({"picks": {str(k): v for k, v in picks.items()}}, indent=2)
    )
    log.info(
        "[phase=calibration_pick] PASS: picks per count = %s",
        {k: {"lr": v["lr"], "achieved_delta_g": v["achieved_delta_g"]} for k, v in picks.items()},
    )
    return picks


# ── v4 step-lever Phase 2.5: pick step per count (plan v4 §4 + §7 H4 kill). ──


def _phase_step_calibration_pick(
    step_calibration_results: list[dict],
    step_pick_path: Path,
) -> dict[int, dict]:
    """Walk step-calibration results, build per-count step pick. Fail loud on H4.

    Mirrors :func:`_phase_calibration_pick` but the axis is the per-cell
    dense early-step checkpoint cadence (plan v4 §4 Phase 2.5). For each
    (count, checkpoint) row the picker reads:

      * ``source_self_delta_g`` (the band axis),
      * ``source_emission_p`` (the emit floor + saturation ceiling), and
      * ``source_R_collapsed`` (the collapse exclusion).

    Persists ``step_calibration_table.json`` BEFORE running the picker so a
    pick crash leaves the full table on disk (checkpoint-per-phase rule); the
    analyzer can re-pick later from the table without re-training.
    """
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        COUNT_LEVELS,
        count_for_slug,
    )
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_step_for_count,
    )

    # Build the {count: {step: {source_self_delta_g, source_emission_p,
    # source_R_collapsed, ...}}} table from each cell's trajectory.json (which
    # holds per-checkpoint source_self block w/ delta_g_mean + emission_p +
    # r_collapsed). For the dispatcher's `done` / `resumed_skip` rows we have
    # the cell_summary.json terminal values too, but the picker needs the FULL
    # trajectory (every early checkpoint, not just terminal) so we re-read
    # trajectory.json per cell. Resume-safe: missing trajectories are skipped
    # with a fail-loud RuntimeError so the picker can never silently mis-rank.
    table: dict[int, dict[int, dict]] = {}
    for r in step_calibration_results:
        if r.get("phase") != "step_calibration":
            continue
        cnt = count_for_slug(r["cell"])
        traj_path = Path(r.get("trajectory_path", ""))
        if not traj_path.exists():
            raise RuntimeError(
                f"step_calibration result for {r['cell']} (seed={r['seed']}, "
                f"lr={r['lr']:g}) has no trajectory.json at {traj_path!r}; the "
                f"step-pick picker cannot rank early-step checkpoints without "
                f"the per-step source_self block. Investigate before retrying."
            )
        traj = json.loads(traj_path.read_text())
        per_step: dict[int, dict] = {}
        for ck in traj["checkpoints"]:
            step_val = ck.get("step")
            if step_val is None:
                continue
            ss = ck["source_self"]
            # Fail loud on missing emission_p — silently defaulting to 0.0
            # would let schema drift surface as a misleading band miss
            # (every step would look like it failed the emission floor)
            # rather than as a data-contract error. i477_run_cell.py already
            # fails loud at write-time; mirror that here so a stale on-disk
            # trajectory with the old schema raises at pick-time too.
            if "emission_p" not in ss:
                raise RuntimeError(
                    f"step_calibration trajectory at {traj_path!r}: checkpoint "
                    f"frac={ck.get('frac')!r} step={step_val} missing "
                    f"emission_p in source_self (keys present = "
                    f"{sorted(ss.keys())}). Schema drift; investigate "
                    f"i472_eval_trajectory.py before re-running."
                )
            per_step[int(step_val)] = {
                "source_self_delta_g": float(ss["delta_g_mean"]),
                "source_emission_p": float(ss["emission_p"]),
                "source_R_collapsed": bool(ss.get("r_collapsed", False)),
                "frac": float(ck["frac"]),
                "adapter_path": ck.get("adapter_path"),
                "cell_slug": r["cell"],
                "seed": int(r["seed"]),
                "lr": float(r["lr"]),
            }
        if cnt in table:
            table[cnt].update(per_step)
        else:
            table[cnt] = per_step

    # Persist the full step table BEFORE picking (checkpoint-per-phase).
    table_path = step_pick_path.parent / "step_calibration_table.json"
    serial = {
        str(cnt): {str(step): entry for step, entry in row.items()} for cnt, row in table.items()
    }
    table_path.write_text(json.dumps(serial, indent=2))
    log.info(
        "[phase=step_calibration_pick] wrote step_calibration_table.json → %s",
        table_path,
    )

    picks: dict[int, dict] = {}
    failures: dict[int, str] = {}
    for cnt in COUNT_LEVELS:
        if cnt not in table:
            failures[cnt] = (
                f"count={cnt} missing from step-calibration results — Phase 2 "
                f"did not produce any step-calibration cell at this count level."
            )
            continue
        try:
            picks[cnt] = pick_step_for_count(table, cnt)
        except RuntimeError as e:
            failures[cnt] = str(e)

    # Coverage floor: per plan v4 §6 discipline #5 + §7, ≥3 of 4 counts must
    # qualify. If exactly 3, proceed at n=6 (DESCRIPTIVE-only). If <3, raise
    # the H4 kill-gate.
    if len(picks) < 3:
        step_pick_path.write_text(
            json.dumps(
                {"picks": {str(k): v for k, v in picks.items()}, "failures": failures},
                indent=2,
            )
        )
        raise RuntimeError(
            "Step-pick FAILED on the H4 kill-gate (plan v4 §7): fewer than 3 "
            f"of 4 count levels qualify. Qualifying: {sorted(picks)}; failing: "
            f"{sorted(failures)}. Failures:\n"
            + "\n".join(f"  count={k}: {v}" for k, v in failures.items())
            + "\nBank Path B: 'marker implants within one optimizer step at "
            "this recipe scale, training-amount cannot decouple count from "
            "implant'. This IS the answer to the parent question."
        )

    step_pick_path.write_text(
        json.dumps(
            {
                "picks": {str(k): v for k, v in picks.items()},
                "failures": failures,  # may be empty; preserved for provenance
                "coverage": {
                    "n_qualifying": len(picks),
                    "n_total": len(COUNT_LEVELS),
                    "descriptive_only": len(picks) < len(COUNT_LEVELS),
                },
            },
            indent=2,
        )
    )
    log.info(
        "[phase=step_calibration_pick] PASS: picks per count = %s%s",
        {
            k: {"step": v["step"], "achieved_delta_g": v["achieved_delta_g"]}
            for k, v in picks.items()
        },
        " (DESCRIPTIVE-only, 3/4 count levels)" if len(picks) < 4 else "",
    )
    return picks


# ── v6 rank pivot: M2 invariant guard + Phase 2A / 2A-CONTROL / 2A.5. ────────


def _verify_alpha_invariant(rank: int, alpha: int) -> None:
    """v6 M2 SSOT guard: every (rank, alpha) pair MUST match the map.

    Fires BEFORE any cell is dispatched (`_phase_rank_calibration` +
    `_phase_rank_control` call it for every unit; main + implant phases call
    it when they thread the picked rank). The dispatcher exits non-zero on
    mismatch so the orchestrator never silently runs a cell at an unpicked
    α. Implementation delegates to ``alpha_for_rank`` so the single source
    of truth lives in the experiment package, NOT here.
    """
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        alpha_for_rank,
    )

    expected = alpha_for_rank(int(rank))
    if int(alpha) != expected:
        raise ValueError(
            f"v6 M2 alpha invariant: got lora_alpha={alpha} for lora_rank={rank}, "
            f"expected {expected} (from RANK_ALPHA_MAP_V5 / ALPHA_CONTROL_V6 SSOT). "
            f"No `2*r` formulation is allowed anywhere in v6; the SSOT helper "
            f"is the only legal source of α."
        )


def _phase_rank_pick(cal_a_results: list[dict], rank_pick_path: Path) -> dict:
    """v6 Phase 2A.5: pick the rank in RANK_GRID_V5 with the most in-band counts.

    Reads the Cal-A cell summaries (one per (rank, count)) into the table
    ``{rank -> {count -> {step -> {delta_g, emit, collapsed}}}}`` the picker
    consumes. Persists ``rank_calibration_table.json`` BEFORE running the
    picker (checkpoint-per-phase: a picker crash leaves the full table on
    disk for re-pick). Fires the H0 off-ramp on FAIL.
    """
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        count_for_calA_slug,
        rank_for_calA_slug,
    )
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_rank,
    )

    table: dict[int, dict[int, dict[int, dict]]] = {}
    for r in cal_a_results:
        if r.get("phase") != "rank_calibration":
            continue
        slug = r["cell"]
        rank = rank_for_calA_slug(slug)
        count = count_for_calA_slug(slug)
        # Per-step trajectory comes from the cell_summary's per_step dict
        # (worker writes it from the dense step-grid checkpoints). Each entry
        # carries delta_g / emit / collapsed at minimum.
        per_step = r.get("per_step")
        if not isinstance(per_step, dict):
            raise RuntimeError(
                f"[rank_pick] Cal-A cell {slug} seed={r.get('seed')} missing "
                f"'per_step' dict (got {type(per_step).__name__}). The worker "
                f"should always emit per_step for phase=rank_calibration."
            )
        by_step: dict[int, dict] = {}
        for step_key, entry in per_step.items():
            step_int = int(step_key) if step_key != "T" else int(entry.get("actual_step", 0))
            by_step[step_int] = {
                "delta_g": float(entry["source_self_delta_g_at_picked_step"]),
                "emit": float(entry["source_emission_p_at_picked_step"]),
                # The worker doesn't yet emit a collapsed bool per step; default
                # False here (eval rig only marks source_R_collapsed at the
                # cell-summary level, not per-step). Pass through if present.
                "collapsed": bool(entry.get("source_R_collapsed", False)),
            }
        table.setdefault(rank, {})[count] = by_step

    # Persist the full table BEFORE picking (checkpoint-per-phase).
    table_path = rank_pick_path.parent / "rank_calibration_table.json"
    serial = {
        str(rank): {
            str(count): {str(s): m for s, m in by_step.items()} for count, by_step in row.items()
        }
        for rank, row in table.items()
    }
    table_path.write_text(json.dumps(serial, indent=2))
    log.info("[phase=rank_pick] wrote rank_calibration_table.json → %s", table_path)

    if not table:
        rank_pick_path.write_text(
            json.dumps({"off_ramp_fired": True, "reason": "empty Cal-A table"}, indent=2)
        )
        raise RuntimeError(
            "[rank_pick] Cal-A table is empty — no rank_calibration cells "
            "produced per_step records. Investigate the rank_calibration phase."
        )

    try:
        pick = pick_rank(table)
    except RuntimeError as e:
        # H0 off-ramp: persist + re-raise so the dispatcher exits non-zero.
        rank_pick_path.write_text(json.dumps({"off_ramp_fired": True, "reason": str(e)}, indent=2))
        raise

    # Re-assert the alpha invariant before writing the pick (defense-in-depth).
    _verify_alpha_invariant(int(pick["picked_rank"]), int(pick["picked_alpha"]))
    rank_pick_path.write_text(json.dumps({"pick": pick}, indent=2))
    log.info(
        "[phase=rank_pick] PASS: picked_rank=%d (α=%d) at positives=%d, "
        "qualifying_counts=%s, per_count_picked_step=%s",
        pick["picked_rank"],
        pick["picked_alpha"],
        pick["picked_positives"],
        pick["qualifying_counts"],
        pick["per_count_picked_step"],
    )
    return pick


def _expand_implant_sweep_v4_anchor_results(anchor_results: list[dict]) -> list[dict]:
    """Expand per-seed implant_sweep_v4 anchor results into per-step records.

    Each anchor result carries a ``per_step`` dict (one entry per requested step
    level + the terminal "T" level). The dispatcher's Phase 4 unpacks the dict
    into one per-cell record per (seed, step level), matching the cell-shape
    the v4 ``implant_only_axis_spearman_marker_channel_kl`` partial consumes.

    Shared between the v6 and v4 implant-sweep branches — both produce identical
    anchor + per_step shapes (only the LoRA rank/alpha differs, which the
    scheduler threads at launch time, not at expansion time). Extracted as a
    module-level helper so the v6 schedule + expansion path can be pinned by a
    pure test without spawning the dispatcher's tokenizer / GPU subprocesses.

    Raises:
        RuntimeError: an anchor result is missing the ``per_step`` dict (the
            worker should always emit it for phase=implant_sweep_v4).
    """
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        ANCHOR_COUNT,
        implant_sweep_v4_slug_for_step,
    )

    expanded_records: list[dict] = []
    for r in anchor_results:
        per_step = r.get("per_step")
        if not isinstance(per_step, dict):
            raise RuntimeError(
                f"[implant_sweep_v4] anchor result for seed={r.get('seed')} "
                f"missing 'per_step' dict (got {type(per_step).__name__}). "
                f"Worker i477_run_cell.py should always emit per_step for "
                f"phase=implant_sweep_v4. Investigate."
            )
        for level_key, entry in per_step.items():
            # level_key is either "16" / "64" / ... (non-terminal) or "T".
            if level_key == "T":
                slug = implant_sweep_v4_slug_for_step("T")
            else:
                slug = implant_sweep_v4_slug_for_step(int(level_key))
            expanded = {
                # Echo the unit's seed/lr/phase plus the per-step DV fields
                # so the analyze partial sees a per-cell shape.
                "cell": slug,
                "seed": int(r["seed"]),
                "lr": float(r["lr"]),
                "phase": "implant_sweep_v4",
                "count": ANCHOR_COUNT,
                "anchor_cell": r["cell"],
                "anchor_run_label": r.get("run_label"),
                # Picked-step DV fields (v4 keys).
                "source_self_marker_channel_kl_at_picked_step": entry[
                    "source_self_marker_channel_kl_at_picked_step"
                ],
                "mean_bystander_marker_channel_kl_at_picked_step": entry[
                    "mean_bystander_marker_channel_kl_at_picked_step"
                ],
                "mean_bystander_full_vocab_kl_at_picked_step": entry[
                    "mean_bystander_full_vocab_kl_at_picked_step"
                ],
                "source_self_delta_g_at_picked_step": entry["source_self_delta_g_at_picked_step"],
                "source_emission_p_at_picked_step": entry["source_emission_p_at_picked_step"],
                "mean_bystander_delta_g_at_picked_step": entry[
                    "mean_bystander_delta_g_at_picked_step"
                ],
                # Step provenance.
                "step_level": level_key,
                "requested_step": entry.get("requested_step"),
                "actual_step": entry.get("actual_step"),
                "step_offset": entry.get("step_offset"),
            }
            expanded_records.append(expanded)
    return expanded_records


def _phase_slot_fix_diagnostic(cal_a0_results: list[dict], diag_path: Path) -> dict:
    """v6 H4 diagnostic: post-hoc verdict from Cal-A0 (r=32) trajectories.

    Reads the Cal-A0 per_step dicts into the table ``{32: {count -> {step ->
    {delta_g}}}}`` the diagnostic consumes, persists the table, then writes
    the verdict to ``rank_control_diagnostic.json``. Never fails loud — the
    verdict (slot-bug-confirmed / slot-bug-rejected / ambiguous) IS the
    deliverable regardless of which way it lands.
    """
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        RANK_CONTROL_V6,
        count_for_calA0_slug,
    )
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        slot_fix_diagnostic,
    )

    table: dict[int, dict[int, dict[int, dict]]] = {RANK_CONTROL_V6: {}}
    for r in cal_a0_results:
        if r.get("phase") != "rank_control":
            continue
        slug = r["cell"]
        count = count_for_calA0_slug(slug)
        per_step = r.get("per_step")
        if not isinstance(per_step, dict):
            raise RuntimeError(
                f"[slot_fix_diagnostic] Cal-A0 cell {slug} seed={r.get('seed')} "
                f"missing 'per_step' dict (got {type(per_step).__name__})."
            )
        by_step: dict[int, dict] = {}
        for step_key, entry in per_step.items():
            step_int = int(step_key) if step_key != "T" else int(entry.get("actual_step", 0))
            by_step[step_int] = {"delta_g": float(entry["source_self_delta_g_at_picked_step"])}
        table[RANK_CONTROL_V6][count] = by_step

    table_path = diag_path.parent / "rank_control_table.json"
    serial = {
        str(rank): {
            str(count): {str(s): m for s, m in by_step.items()} for count, by_step in row.items()
        }
        for rank, row in table.items()
    }
    table_path.write_text(json.dumps(serial, indent=2))
    log.info("[phase=slot_fix_diagnostic] wrote rank_control_table.json → %s", table_path)

    if not table[RANK_CONTROL_V6]:
        # H4 has no data — log + persist empty verdict; don't fail loud (the
        # main sweep can still proceed since H0 owns the kill-gate).
        verdict_out = {
            "verdict": "no-cal-a0-data",
            "per_count_max_delta_g": {},
            "max_terminal_delta_g": 0.0,
            "alpha_used": 64,
        }
        diag_path.write_text(json.dumps(verdict_out, indent=2))
        log.warning("[phase=slot_fix_diagnostic] no Cal-A0 cells found; verdict deferred.")
        return verdict_out

    verdict = slot_fix_diagnostic(table)
    diag_path.write_text(json.dumps(verdict, indent=2))
    log.info(
        "[phase=slot_fix_diagnostic] H4 verdict: %s (max ΔG=%.3f; per-count=%s)",
        verdict["verdict"],
        verdict["max_terminal_delta_g"],
        verdict["per_count_max_delta_g"],
    )
    return verdict


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - linear pipeline
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        ANCHOR_COUNT,
        CALIB_SLUGS,
        CALIBRATION_LR_GRID,
        CALIBRATION_LR_V3,
        IMPLANT_SWEEP_LRS,
        IMPLANT_SWEEP_SLUGS,
        IMPLANT_SWEEP_STEPS,
        IMPLANT_SWEEP_V4_ANCHOR_SLUG,
        MAIN_SLUGS,
        count_for_slug,
        lr_for_implant_sweep_slug,
    )

    parser = argparse.ArgumentParser(description="i477 dispatcher — implant-decoupled count sweep.")
    parser.add_argument(
        "--phases",
        default="rank_calibration,rank_control,rank_pick,slot_fix_diagnostic,main,implant_sweep",
        help=(
            "CSV subset of the v6 pipeline phases. Default: v6 rank-pivot "
            "path (rank_calibration → rank_control → rank_pick → "
            "slot_fix_diagnostic → main → implant_sweep). v4 step-lever "
            "phases (step_calibration → step_pick) and v2 legacy phases "
            "(calibration → calibration_pick) are reachable by passing "
            "--phases explicitly."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "One step-calibration cell (c477_calib_negp_4, seed=42, "
            "lr=CALIBRATION_LR_V3=2e-6) + tiny training/eval slice. Same "
            "dispatcher path as the full sweep — smoke IS the calibration "
            "sweep with one cell."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate imports + marker assertion, no GPU work.",
    )
    parser.add_argument("--no-kl", action="store_true", help="Skip DV-B KL (smoke speed-up).")
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--seeds", default="42,137")
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_477"))
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_477"))
    parser.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    parser.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    parser.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--resume", action="store_true")
    # ── v4 step-lever flags (plan v4 §4 dispatcher row). ─────────────────────
    parser.add_argument(
        "--target-steps",
        default="1,2,4,8,16,32,64",
        help=(
            "v4 dense early optimizer-step grid for the step_calibration "
            "phase (Phase 2). CSV of positive ints. Default = "
            "plan v4 §4 TARGET_STEPS."
        ),
    )
    parser.add_argument(
        "--calibration-lr",
        type=float,
        default=None,
        help=(
            "Fixed LR for the v4 step-calibration phase. Default = "
            "CALIBRATION_LR_V3 (2e-6). Override only for sensitivity sweeps; "
            "v4 §11 grounds the 2e-6 choice in #477 v2 calibration round 1."
        ),
    )
    parser.add_argument(
        "--legacy-lr-calibration",
        action="store_true",
        help=(
            "Keep the v2 LR-calibration path. Replaces --phases default with "
            "'calibration,calibration_pick,main,implant_sweep' (the v2 byte-"
            "identical path). Use ONLY for v2 reproductions / regression "
            "checks; v4 is the default."
        ),
    )
    # ── v6 rank pivot flags (plan v6 §4.4 dispatcher row). ───────────────────
    parser.add_argument(
        "--rank-grid",
        default="2,4,8",
        help=(
            "v6 Cal-A LoRA rank sweep (CSV of ints in RANK_ALPHA_MAP_V5). "
            "Default {2,4,8}. Every member MUST have an entry in "
            "RANK_ALPHA_MAP_V5; _verify_alpha_invariant fires loud otherwise."
        ),
    )
    parser.add_argument(
        "--positives",
        type=int,
        default=200,
        help=(
            "v6 M3 declaration: positives per cell, GLOBAL across all phases. "
            "Default 200 (= POS_EX_PER_SOURCE; v4 byte-identical). The CLI flag "
            "is declarative only — build_cell reads the module constant directly, "
            "so a non-default value here triggers a startup assertion failure "
            "(rather than silently overriding nothing per-cell)."
        ),
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help=(
            "v6: override LoRA rank for a SINGLE-phase invocation (smoke / "
            "manual rank_control / debug). Default None = use module constant "
            "r=32. M2: MUST be paired with --lora-alpha; the value MUST equal "
            "alpha_for_rank(rank)."
        ),
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help=(
            "v6: override LoRA alpha. MUST satisfy "
            "alpha=RANK_ALPHA_MAP_V5[rank] for rank in {2,4,8}, OR alpha=64 "
            "for rank=32 (Cal-A0 control). NO `2*r` math."
        ),
    )
    parser.add_argument(
        "--cells",
        default=None,
        help=(
            "Optional smoke selector: CSV of cell slugs to run instead of the "
            "phase's full unit set. Used when the orchestrator wants to smoke "
            "a single Cal-A or Cal-A0 cell to validate threading + the slot-"
            "fix port end-to-end before the full sweep."
        ),
    )
    args = parser.parse_args(argv)

    # v6 M2 — early alpha-invariant guard for the smoke / single-cell path.
    if args.lora_rank is not None or args.lora_alpha is not None:
        if args.lora_rank is None or args.lora_alpha is None:
            raise SystemExit(
                "v6 M2: --lora-rank and --lora-alpha must be passed together; "
                f"got rank={args.lora_rank} alpha={args.lora_alpha}."
            )
        _verify_alpha_invariant(int(args.lora_rank), int(args.lora_alpha))

    # v6 M3 — positives is GLOBAL at POS_EX_PER_SOURCE=200; the CLI flag is
    # declarative (build_cell reads the module constant). A non-default value
    # would be a silent no-op without this assertion. (Round-2 fix to
    # code-review v6 round-1 #3.)
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        POS_EX_PER_SOURCE,
    )

    if int(args.positives) != POS_EX_PER_SOURCE:
        raise SystemExit(
            f"v6 M3 positives invariant: --positives={args.positives} but "
            f"POS_EX_PER_SOURCE={POS_EX_PER_SOURCE} (build_cell reads the "
            f"module constant). The CLI flag is declarative-only; override "
            f"the module constant if you need to vary positives globally."
        )

    # v4: --legacy-lr-calibration toggles back to the v2 default phases ONLY
    # when the caller did not also pass --phases explicitly. The v6 default
    # phases string is the new sentinel (v4 default was the prior sentinel).
    _v6_default_phases = (
        "rank_calibration,rank_control,rank_pick,slot_fix_diagnostic,main,implant_sweep"
    )
    _v4_default_phases = "step_calibration,step_pick,main,implant_sweep"
    if args.legacy_lr_calibration and args.phases in (
        _v6_default_phases,
        _v4_default_phases,
    ):
        args.phases = "calibration,calibration_pick,main,implant_sweep"

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    global LOG_DIR
    LOG_DIR = args.log_dir
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    args.slab_root.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)

    # MooseFS quota safety + adapter-persist (Upload Policy / gotchas).
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    os.environ.setdefault("EPM_PERSIST_ADAPTER_HF_REPO", HF_MODEL_REPO)
    os.environ.setdefault("EPM_PERSIST_ADAPTER_HF_SUBFOLDER", "adapters/issue_477")
    # WandB project (plan §10). Set before training subprocesses inherit env so
    # #477 runs land under the right WandB project rather than the parent's.
    os.environ.setdefault("WANDB_PROJECT", "issue_477")

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # ── Pre-flight: marker tokenizer assertion (CLAUDE.md / load-bearing). ──
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )

    if not args.dry_run:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        ids = tok.encode(MARKER_TEXT, add_special_tokens=False)
        if ids != [EXPECTED_MARKER_TOKEN_ID]:
            raise RuntimeError(
                f"Marker tokenizer assertion FAILED: encode({MARKER_TEXT!r})={ids}, "
                f"expected [{EXPECTED_MARKER_TOKEN_ID}]."
            )
        log.info("[phase=preflight] marker assertion PASS: %r -> %s", MARKER_TEXT, ids)
    else:
        log.info("[phase=preflight] marker assertion DEFERRED (dry-run)")
        # Still validate the 477 module imports cleanly + the CELL_SPECS registry
        # is well-formed. Catches surface-level bugs without GPU.
        log.info("[phase=preflight] dry-run validated imports + cell registry")
        return 0

    phase_summaries: dict[str, dict] = {}

    # ── v4 step-lever defaults (plan v4 §11). ────────────────────────────────

    calibration_lr_v4 = (
        args.calibration_lr if args.calibration_lr is not None else CALIBRATION_LR_V3
    )

    # ── Smoke unification: --smoke is one (step|rank)_calibration cell. ──────
    # PASS_UNIFIED: smoke = sweep with a single cell. Same dispatcher path,
    # same subprocess shape (i477_run_cell.py), same env injection, same
    # nested-eval teardown. v6: when --cells (Cal-A / Cal-A0 slug) AND
    # --lora-rank / --lora-alpha are supplied, smoke runs a v6 cell at the
    # provided rank with the slot-fix port on; otherwise falls back to the
    # v4 default (step_calibration cell, count=4, r=32/α=64 module defaults).
    # Legacy: --smoke + --legacy-lr-calibration keeps the v2 single-LR shape.
    if args.smoke:
        v6_smoke = args.cells is not None and args.lora_rank is not None
        if args.legacy_lr_calibration:
            log.info("[phase=smoke] legacy v2 smoke (one calibration cell, count=4, lr=1e-5)")
            unit = {"cell": "c477_calib_negp_4", "seed": 42, "lr": 1e-5, "phase": "calibration"}
        elif v6_smoke:
            # v6 smoke: caller passed a Cal-A or Cal-A0 slug + the v6 rank/alpha.
            # Route via rank_calibration (Cal-A) or rank_control (Cal-A0) based
            # on the slug prefix. The M2 guard already verified rank/alpha
            # match the SSOT before we got here.
            cell_slug = args.cells.split(",")[0].strip()
            if cell_slug.startswith("c477_calA0_"):
                v6_phase = "rank_control"
            elif cell_slug.startswith("c477_calA_"):
                v6_phase = "rank_calibration"
            else:
                raise SystemExit(
                    f"v6 smoke: --cells={cell_slug!r} is not a Cal-A "
                    f"(c477_calA_*) or Cal-A0 (c477_calA0_*) slug; v6 smoke "
                    f"only supports v6 cells."
                )
            log.info(
                "[phase=smoke] v6 smoke (one %s cell %s, lr=%g, "
                "lora_rank=%d, lora_alpha=%d, target-steps=%s)",
                v6_phase,
                cell_slug,
                calibration_lr_v4,
                args.lora_rank,
                args.lora_alpha,
                args.target_steps,
            )
            unit = {
                "cell": cell_slug,
                "seed": 42,
                "lr": calibration_lr_v4,
                "phase": v6_phase,
                "target_steps": args.target_steps,
                "lora_r": int(args.lora_rank),
                "lora_alpha": int(args.lora_alpha),
            }
        else:
            log.info(
                "[phase=smoke] v4 smoke (one step_calibration cell, count=4, "
                "lr=%g, target-steps=%s)",
                calibration_lr_v4,
                args.target_steps,
            )
            unit = {
                "cell": "c477_calib_negp_4",
                "seed": 42,
                "lr": calibration_lr_v4,
                "phase": "step_calibration",
                "target_steps": args.target_steps,
            }
        results = _schedule_unit_pool(
            units=[unit],
            n_gpus=max(1, args.n_gpus),
            max_parallel=1,
            slab_root=args.slab_root,
            runs_root=args.runs_root,
            log_dir=LOG_DIR,
            bank_path=args.bank_path,
            centroids_dir=args.centroids_dir,
            smoke=True,
            no_kl=args.no_kl,
            report_to=args.report_to,
            resume=args.resume,
        )
        phase_summaries["smoke"] = {"unit_results": results}
        _write_sentinel(
            LOG_DIR / "issue-477-smoke-results.json",
            kind="epm:progress",
            phase="smoke",
            note_payload=phase_summaries["smoke"],
        )
        _write_final_sentinel(phase_summaries, status="done")
        log.info("[phase=done] smoke dispatcher exit %s", datetime.now(UTC).isoformat())
        return 0

    # ── v4 Phase 2: step-calibration sweep (4 cells at fixed LR, dense ckpts). ─
    step_calibration_results: list[dict] = []
    if "step_calibration" in phases:
        log.info(
            "[phase=step_calibration] scheduling %d cells (4 counts × seed=42 × "
            "lr=%g, target-steps=%s)",
            len(CALIB_SLUGS),
            calibration_lr_v4,
            args.target_steps,
        )
        sc_units = [
            {
                "cell": slug,
                "seed": 42,
                "lr": calibration_lr_v4,
                "phase": "step_calibration",
                "target_steps": args.target_steps,
            }
            for slug in CALIB_SLUGS
        ]
        step_calibration_results = _schedule_unit_pool(
            units=sc_units,
            n_gpus=args.n_gpus,
            max_parallel=args.max_parallel,
            slab_root=args.slab_root,
            runs_root=args.runs_root,
            log_dir=LOG_DIR,
            bank_path=args.bank_path,
            centroids_dir=args.centroids_dir,
            smoke=False,
            no_kl=args.no_kl,
            report_to=args.report_to,
            resume=args.resume,
        )
        phase_summaries["step_calibration"] = {
            "n_completed": len(step_calibration_results),
            "results": step_calibration_results,
        }
        _write_sentinel(
            LOG_DIR / "issue-477-step-calibration-results.json",
            kind="epm:progress",
            phase="step_calibration",
            note_payload=phase_summaries["step_calibration"],
        )
    else:
        log.info("[phase=step_calibration] SKIP")

    # ── v4 Phase 2.5: step-pick per count level (H4 kill-gate). ──────────────
    step_picks: dict[int, dict] = {}
    if "step_pick" in phases:
        if not step_calibration_results:
            sp_table_path = args.slab_root / "step_calibration_table.json"
            if not sp_table_path.exists():
                raise RuntimeError(
                    f"step_pick requested but step_calibration_results is empty AND "
                    f"{sp_table_path} missing. Run --phases step_calibration first "
                    f"or pass --resume to reload prior step-calibration cells from "
                    f"runs-root."
                )
            for slug in CALIB_SLUGS:
                run_label = f"{slug}_seed42_lr{calibration_lr_v4:g}"
                sp = args.runs_root / run_label / "cell_summary.json"
                if sp.exists():
                    step_calibration_results.append(json.loads(sp.read_text()))
            if not step_calibration_results:
                raise RuntimeError(
                    "step_calibration_results still empty after on-disk resume — "
                    "no step-calibration cells found. Run --phases step_calibration first."
                )
        sp_pick_path = args.slab_root / "step_calibration_pick.json"
        step_picks = _phase_step_calibration_pick(step_calibration_results, sp_pick_path)
        phase_summaries["step_pick"] = {str(k): v for k, v in step_picks.items()}
        _write_sentinel(
            LOG_DIR / "issue-477-step-pick-results.json",
            kind="epm:progress",
            phase="step_pick",
            note_payload=phase_summaries["step_pick"],
        )
    else:
        log.info("[phase=step_pick] SKIP")

    # ── v6 Phase 2A: rank_calibration (Cal-A). 3 ranks × 4 counts × seed=42. ─
    cal_a_results: list[dict] = []
    if "rank_calibration" in phases:
        from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
            CAL_A_SLUGS,
            COUNT_LEVELS,
            RANK_ALPHA_MAP_V5,
            alpha_for_rank,
            cal_a_slug,
            count_for_calA_slug,
            rank_for_calA_slug,
        )

        rank_grid = tuple(int(s.strip()) for s in args.rank_grid.split(",") if s.strip())
        for r in rank_grid:
            if r not in RANK_ALPHA_MAP_V5:
                raise ValueError(
                    f"--rank-grid={args.rank_grid}: rank {r} not in "
                    f"RANK_ALPHA_MAP_V5={RANK_ALPHA_MAP_V5}. v6 M2: every Cal-A "
                    f"rank MUST have an entry in the SSOT map."
                )
        cal_a_units: list[dict] = []
        for r in rank_grid:
            alpha = alpha_for_rank(r)
            _verify_alpha_invariant(r, alpha)
            for count in COUNT_LEVELS:
                cal_a_units.append(
                    {
                        "cell": cal_a_slug(count, r),
                        "seed": 42,
                        "lr": calibration_lr_v4,
                        "phase": "rank_calibration",
                        "target_steps": args.target_steps,
                        "lora_r": r,
                        "lora_alpha": alpha,
                    }
                )
        # Smoke / debug carve-out: --cells filters to a subset of the unit set
        # WITHOUT touching the rank/alpha threading (M2 invariant already
        # applied above). Useful for one-cell preflight on the pod.
        if args.cells:
            wanted = set(s.strip() for s in args.cells.split(",") if s.strip())
            cal_a_units = [u for u in cal_a_units if u["cell"] in wanted]
            if not cal_a_units:
                raise RuntimeError(
                    f"--cells={args.cells!r} filtered ALL Cal-A units out; "
                    f"available Cal-A slugs are {list(CAL_A_SLUGS)}."
                )
        # Sanity-tag every unit (helps the pick phase parse cell metadata
        # even if the worker's summary later loses the per_step keys).
        for u in cal_a_units:
            u["_meta_count"] = count_for_calA_slug(u["cell"])
            u["_meta_rank"] = rank_for_calA_slug(u["cell"])
        log.info(
            "[phase=rank_calibration] scheduling %d Cal-A cells (ranks %s × "
            "counts %s × seed=42, lr=%g, positives=%d, target-steps=%s)",
            len(cal_a_units),
            rank_grid,
            COUNT_LEVELS,
            calibration_lr_v4,
            args.positives,
            args.target_steps,
        )
        cal_a_results = _schedule_unit_pool(
            units=cal_a_units,
            n_gpus=args.n_gpus,
            max_parallel=args.max_parallel,
            slab_root=args.slab_root,
            runs_root=args.runs_root,
            log_dir=LOG_DIR,
            bank_path=args.bank_path,
            centroids_dir=args.centroids_dir,
            smoke=False,
            no_kl=args.no_kl,
            report_to=args.report_to,
            resume=args.resume,
        )
        phase_summaries["rank_calibration"] = {
            "n_completed": len(cal_a_results),
            "results": cal_a_results,
        }
        _write_sentinel(
            LOG_DIR / "issue-477-rank-calibration-results.json",
            kind="epm:progress",
            phase="rank_calibration",
            note_payload=phase_summaries["rank_calibration"],
        )
    else:
        log.info("[phase=rank_calibration] SKIP")

    # ── v6 Phase 2A-CONTROL: rank_control (Cal-A0). r=32 / α=64 × 3 counts. ──
    cal_a0_results: list[dict] = []
    if "rank_control" in phases:
        from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
            ALPHA_CONTROL_V6,
            CAL_A0_SLUGS,
            RANK_CONTROL_COUNTS_V6,
            RANK_CONTROL_V6,
            cal_a0_slug,
            count_for_calA0_slug,
        )

        _verify_alpha_invariant(RANK_CONTROL_V6, ALPHA_CONTROL_V6)
        cal_a0_units: list[dict] = [
            {
                "cell": cal_a0_slug(count),
                "seed": 42,
                "lr": calibration_lr_v4,
                "phase": "rank_control",
                "target_steps": args.target_steps,
                "lora_r": RANK_CONTROL_V6,
                "lora_alpha": ALPHA_CONTROL_V6,
            }
            for count in RANK_CONTROL_COUNTS_V6
        ]
        if args.cells:
            wanted = set(s.strip() for s in args.cells.split(",") if s.strip())
            cal_a0_units = [u for u in cal_a0_units if u["cell"] in wanted]
            if not cal_a0_units:
                raise RuntimeError(
                    f"--cells={args.cells!r} filtered ALL Cal-A0 units out; "
                    f"available Cal-A0 slugs are {list(CAL_A0_SLUGS)}."
                )
        for u in cal_a0_units:
            u["_meta_count"] = count_for_calA0_slug(u["cell"])
            u["_meta_rank"] = RANK_CONTROL_V6
        log.info(
            "[phase=rank_control] scheduling %d Cal-A0 cells (r=%d / α=%d × "
            "counts %s × seed=42, lr=%g, positives=%d, target-steps=%s)",
            len(cal_a0_units),
            RANK_CONTROL_V6,
            ALPHA_CONTROL_V6,
            RANK_CONTROL_COUNTS_V6,
            calibration_lr_v4,
            args.positives,
            args.target_steps,
        )
        cal_a0_results = _schedule_unit_pool(
            units=cal_a0_units,
            n_gpus=args.n_gpus,
            max_parallel=args.max_parallel,
            slab_root=args.slab_root,
            runs_root=args.runs_root,
            log_dir=LOG_DIR,
            bank_path=args.bank_path,
            centroids_dir=args.centroids_dir,
            smoke=False,
            no_kl=args.no_kl,
            report_to=args.report_to,
            resume=args.resume,
        )
        phase_summaries["rank_control"] = {
            "n_completed": len(cal_a0_results),
            "results": cal_a0_results,
        }
        _write_sentinel(
            LOG_DIR / "issue-477-rank-control-results.json",
            kind="epm:progress",
            phase="rank_control",
            note_payload=phase_summaries["rank_control"],
        )
    else:
        log.info("[phase=rank_control] SKIP")

    # ── v6 Phase 2A.5: rank_pick (H0 kill-gate). ─────────────────────────────
    rank_pick: dict = {}
    if "rank_pick" in phases:
        if not cal_a_results:
            from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
                CAL_A_SLUGS,
            )

            for slug in CAL_A_SLUGS:
                run_label = f"{slug}_seed42_lr{calibration_lr_v4:g}"
                sp = args.runs_root / run_label / "cell_summary.json"
                if sp.exists():
                    cal_a_results.append(json.loads(sp.read_text()))
            if not cal_a_results:
                raise RuntimeError(
                    "rank_pick requested but no Cal-A cell_summary.json found "
                    "on disk. Run --phases rank_calibration first."
                )
        rp_path = args.slab_root / "rank_calibration_pick.json"
        rank_pick = _phase_rank_pick(cal_a_results, rp_path)
        phase_summaries["rank_pick"] = rank_pick
        _write_sentinel(
            LOG_DIR / "issue-477-rank-pick-results.json",
            kind="epm:progress",
            phase="rank_pick",
            note_payload=phase_summaries["rank_pick"],
        )
    else:
        log.info("[phase=rank_pick] SKIP")

    # ── v6 H4 slot-fix-vs-capacity diagnostic (post-Cal-A0; non-gating). ─────
    h4_verdict: dict = {}
    if "slot_fix_diagnostic" in phases:
        if not cal_a0_results:
            from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
                CAL_A0_SLUGS,
            )

            for slug in CAL_A0_SLUGS:
                run_label = f"{slug}_seed42_lr{calibration_lr_v4:g}"
                sp = args.runs_root / run_label / "cell_summary.json"
                if sp.exists():
                    cal_a0_results.append(json.loads(sp.read_text()))
        diag_path = args.slab_root / "rank_control_diagnostic.json"
        h4_verdict = _phase_slot_fix_diagnostic(cal_a0_results, diag_path)
        phase_summaries["slot_fix_diagnostic"] = h4_verdict
        _write_sentinel(
            LOG_DIR / "issue-477-slot-fix-diagnostic-results.json",
            kind="epm:progress",
            phase="slot_fix_diagnostic",
            note_payload=phase_summaries["slot_fix_diagnostic"],
        )
    else:
        log.info("[phase=slot_fix_diagnostic] SKIP")

    # ── Phase 2: calibration sweep. ──────────────────────────────────────────
    calibration_results: list[dict] = []
    if "calibration" in phases:
        log.info("[phase=calibration] scheduling 4 counts × 5 LRs = 20 cells (seed=42)")
        calib_units = [
            {"cell": slug, "seed": 42, "lr": lr, "phase": "calibration"}
            for slug in CALIB_SLUGS
            for lr in CALIBRATION_LR_GRID
        ]
        calibration_results = _schedule_unit_pool(
            units=calib_units,
            n_gpus=args.n_gpus,
            max_parallel=args.max_parallel,
            slab_root=args.slab_root,
            runs_root=args.runs_root,
            log_dir=LOG_DIR,
            bank_path=args.bank_path,
            centroids_dir=args.centroids_dir,
            smoke=False,
            no_kl=args.no_kl,
            report_to=args.report_to,
            resume=args.resume,
        )
        phase_summaries["calibration"] = {
            "n_completed": len(calibration_results),
            "results": calibration_results,
        }
        _write_sentinel(
            LOG_DIR / "issue-477-calibration-results.json",
            kind="epm:progress",
            phase="calibration",
            note_payload=phase_summaries["calibration"],
        )
    else:
        log.info("[phase=calibration] SKIP")

    # ── Phase 2.5: pick LR per count level. ──────────────────────────────────
    picks: dict[int, dict] = {}
    if "calibration_pick" in phases:
        # If calibration was skipped, attempt to read calibration_table.json.
        if not calibration_results:
            table_path = args.slab_root / "calibration_table.json"
            if not table_path.exists():
                raise RuntimeError(
                    f"calibration_pick requested but calibration_results is empty AND "
                    f"{table_path} missing. Run --phases calibration first or pass "
                    f"--resume to load prior calibration cell summaries from runs-root."
                )
            # Reconstruct calibration_results from on-disk cell_summary.json files.
            for slug in CALIB_SLUGS:
                for lr in CALIBRATION_LR_GRID:
                    run_label = f"{slug}_seed42_lr{lr:g}"
                    sp = args.runs_root / run_label / "cell_summary.json"
                    if sp.exists():
                        calibration_results.append(json.loads(sp.read_text()))
            if not calibration_results:
                raise RuntimeError(
                    "calibration_results still empty after on-disk resume — no calibration "
                    "cells found. Run --phases calibration first."
                )

        pick_path = args.slab_root / "calibration_pick.json"
        picks = _phase_calibration_pick(calibration_results, pick_path)
        phase_summaries["calibration_pick"] = {str(k): v for k, v in picks.items()}
        _write_sentinel(
            LOG_DIR / "issue-477-calibration-pick-results.json",
            kind="epm:progress",
            phase="calibration_pick",
            note_payload=phase_summaries["calibration_pick"],
        )
    else:
        log.info("[phase=calibration_pick] SKIP")

    # ── Phase 3: main sweep at calibrated step/LR per count × 2 seeds. ──────
    # v4 default: route through phase=main_v4 with --picked-step (from
    # step_picks). v2 --legacy-lr-calibration: phase=main with calibrated LR
    # (picks). Both run the same scheduler shape; the worker resolves
    # max_steps + the clamped context window in main_v4 vs uses
    # TRAJECTORY_CHECKPOINT_FRACTIONS in main.
    main_results: list[dict] = []
    if "main" in phases:
        # Prefer v4 step_picks if present (in-memory or on disk); otherwise
        # fall back to v2 LR picks. --legacy-lr-calibration suppresses v4
        # routing even if a stale step_calibration_pick.json sits in the slab
        # — a legacy rerun must NEVER read the v4 pick file, since the v2
        # calibrated-LR path picks ITS lr from calibration_pick.json and
        # ignores picked_step entirely.
        # v6 routing: if a rank_pick exists (in-memory OR on disk), the main
        # sweep runs at picked_rank + alpha_for_rank(picked_rank) with the
        # slot-fix port automatically on (the worker detects v6 from
        # lora_rank is not None). The per-count picked_step comes from
        # rank_pick.per_count_picked_step (Cal-A's in-band step closest to
        # ΔG=12). Falls back to v4 step_picks if rank_pick is absent, then
        # v2 picks under --legacy-lr-calibration.
        rank_pick_path = args.slab_root / "rank_calibration_pick.json"
        use_v6 = (not args.legacy_lr_calibration) and (bool(rank_pick) or rank_pick_path.exists())
        if use_v6 and not rank_pick:
            loaded = json.loads(rank_pick_path.read_text())
            rank_pick = loaded.get("pick", {})
            if not rank_pick or rank_pick.get("off_ramp_fired"):
                raise RuntimeError(
                    "main phase requested with v6 routing but rank_pick has no "
                    "valid pick (off_ramp_fired or missing). The dispatcher MUST "
                    "NOT silently run the main sweep at an unpicked rank."
                )
        use_v4 = (
            not use_v6
            and (not args.legacy_lr_calibration)
            and (bool(step_picks) or (args.slab_root / "step_calibration_pick.json").exists())
        )
        if use_v4 and not step_picks:
            loaded = json.loads((args.slab_root / "step_calibration_pick.json").read_text())
            step_picks = {int(k): v for k, v in loaded.get("picks", {}).items()}
        if not use_v4 and not use_v6 and not picks:
            pick_path = args.slab_root / "calibration_pick.json"
            if pick_path.exists():
                loaded = json.loads(pick_path.read_text())
                picks = {int(k): v for k, v in loaded.get("picks", {}).items()}
            else:
                raise RuntimeError(
                    "main phase requested but no pick file resolved (no in-memory "
                    "state, no on-disk rank_calibration_pick.json / "
                    "step_calibration_pick.json / calibration_pick.json). Run "
                    "--phases rank_calibration,rank_pick first (v6 default) OR "
                    "--phases step_calibration,step_pick (v4) OR "
                    "--legacy-lr-calibration --phases calibration,calibration_pick (v2)."
                )
        if use_v6:
            scheduling_mode_desc = (
                f"v6 picked rank={rank_pick['picked_rank']} (α={rank_pick['picked_alpha']}) "
                f"+ per-count picked steps + lr={calibration_lr_v4:g}"
            )
        elif use_v4:
            scheduling_mode_desc = f"v4 picked steps + lr={calibration_lr_v4:g}"
        else:
            scheduling_mode_desc = "v2 calibrated LRs"
        log.info(
            "[phase=main] scheduling %d main cells (4 counts × %d seeds) at %s",
            len(MAIN_SLUGS) * len(seeds),
            len(seeds),
            scheduling_mode_desc,
        )
        main_units = []
        for slug in MAIN_SLUGS:
            cnt = count_for_slug(slug)
            if use_v6:
                per_count = rank_pick.get("per_count_picked_step", {})
                # JSON deserializes int keys as strings; normalize both ways.
                picked_step_raw = per_count.get(cnt) or per_count.get(str(cnt))
                if picked_step_raw is None:
                    log.warning(
                        "[phase=main] v6: count=%d missing from rank_pick.per_count_picked_step "
                        "(partial coverage); SKIPPING its main cells (DESCRIPTIVE-only path, "
                        "plan v6 §7 H0 partial branch).",
                        cnt,
                    )
                    continue
                picked_step = int(picked_step_raw)
                picked_rank = int(rank_pick["picked_rank"])
                picked_alpha = int(rank_pick["picked_alpha"])
                _verify_alpha_invariant(picked_rank, picked_alpha)
                for seed in seeds:
                    main_units.append(
                        {
                            "cell": slug,
                            "seed": seed,
                            "lr": calibration_lr_v4,
                            "phase": "main_v4",
                            "picked_step": picked_step,
                            "lora_r": picked_rank,
                            "lora_alpha": picked_alpha,
                        }
                    )
            elif use_v4:
                pick = step_picks.get(cnt)
                if pick is None:
                    log.warning(
                        "[phase=main] count=%d missing from step_picks (H4 "
                        "partial coverage); SKIPPING its main cells (DESCRIPTIVE-"
                        "only path, plan v4 §7).",
                        cnt,
                    )
                    continue
                picked_step = int(pick["step"])
                lr_for_cell = calibration_lr_v4
                for seed in seeds:
                    main_units.append(
                        {
                            "cell": slug,
                            "seed": seed,
                            "lr": lr_for_cell,
                            "phase": "main_v4",
                            "picked_step": picked_step,
                        }
                    )
            else:
                lr_for_cell = float(picks[cnt]["lr"])
                for seed in seeds:
                    main_units.append(
                        {"cell": slug, "seed": seed, "lr": lr_for_cell, "phase": "main"}
                    )
        main_results = _schedule_unit_pool(
            units=main_units,
            n_gpus=args.n_gpus,
            max_parallel=args.max_parallel,
            slab_root=args.slab_root,
            runs_root=args.runs_root,
            log_dir=LOG_DIR,
            bank_path=args.bank_path,
            centroids_dir=args.centroids_dir,
            smoke=False,
            no_kl=args.no_kl,
            report_to=args.report_to,
            resume=args.resume,
        )
        # Pin count + lr on each result for downstream analyzer.
        for r in main_results:
            r["count"] = count_for_slug(r["cell"])
        phase_summaries["main"] = {"n_completed": len(main_results), "results": main_results}
        _write_sentinel(
            LOG_DIR / "issue-477-main-results.json",
            kind="epm:progress",
            phase="main",
            note_payload=phase_summaries["main"],
        )
    else:
        log.info("[phase=main] SKIP")

    # ── Phase 4: implant-only-axis arm. ─────────────────────────────────────
    # v4r2 default: STEP sweep at fixed lr=CALIBRATION_LR_V3, count=ANCHOR_COUNT.
    #     ONE anchor training run per seed; worker emits per_step records at
    #     {16, 64, T}. Dispatcher expands per-seed anchor → 3 per-step records,
    #     each carrying the v4 picked-step DV keys the implant-only-axis
    #     marker-channel partial reads.
    # v2 --legacy-lr-calibration: 3 LRs × 2 seeds = 6 cells (byte-identical).
    implant_sweep_results: list[dict] = []
    if "implant_sweep" in phases:
        # Route v6 ⇄ v4 ⇄ v2: --legacy-lr-calibration suppresses v6 + v4. v6
        # routing fires when rank_calibration_pick.json exists; the implant-
        # only-axis anchor then trains at picked_rank + alpha_for_rank(picked)
        # with the slot-fix port on (the worker auto-detects v6 from
        # lora_rank is not None). v4 routing keeps the step-sweep at the
        # module r=32/α=64 defaults.
        impl_rank_pick_path = args.slab_root / "rank_calibration_pick.json"
        impl_use_v6 = (not args.legacy_lr_calibration) and (
            bool(rank_pick) or impl_rank_pick_path.exists()
        )
        if impl_use_v6 and not rank_pick:
            loaded = json.loads(impl_rank_pick_path.read_text())
            rank_pick = loaded.get("pick", {})
            if not rank_pick or rank_pick.get("off_ramp_fired"):
                raise RuntimeError(
                    "implant_sweep requested with v6 routing but rank_pick has "
                    "no valid pick (off_ramp_fired or missing)."
                )
        impl_use_v4 = (
            not impl_use_v6
            and (not args.legacy_lr_calibration)
            and (bool(step_picks) or (args.slab_root / "step_calibration_pick.json").exists())
        )
        if impl_use_v6:
            picked_rank_imp = int(rank_pick["picked_rank"])
            picked_alpha_imp = int(rank_pick["picked_alpha"])
            _verify_alpha_invariant(picked_rank_imp, picked_alpha_imp)
            log.info(
                "[phase=implant_sweep] v6 step-sweep at picked rank: %d anchor "
                "cells (1 anchor × %d seeds at count=%d, lr=%g, r=%d, α=%d), "
                "step levels %s + T",
                len(seeds),
                len(seeds),
                ANCHOR_COUNT,
                CALIBRATION_LR_V3,
                picked_rank_imp,
                picked_alpha_imp,
                IMPLANT_SWEEP_STEPS,
            )
            implant_steps_csv = ",".join(str(s) for s in IMPLANT_SWEEP_STEPS)
            is_units = []
            for seed in seeds:
                is_units.append(
                    {
                        "cell": IMPLANT_SWEEP_V4_ANCHOR_SLUG,
                        "seed": seed,
                        "lr": CALIBRATION_LR_V3,
                        "phase": "implant_sweep_v4",
                        "implant_steps": implant_steps_csv,
                        "lora_r": picked_rank_imp,
                        "lora_alpha": picked_alpha_imp,
                    }
                )
        elif impl_use_v4:
            log.info(
                "[phase=implant_sweep] v4 step-sweep: %d anchor cells "
                "(1 anchor × %d seeds at count=%d, lr=%g), step levels %s + T",
                len(seeds),
                len(seeds),
                ANCHOR_COUNT,
                CALIBRATION_LR_V3,
                IMPLANT_SWEEP_STEPS,
            )
            implant_steps_csv = ",".join(str(s) for s in IMPLANT_SWEEP_STEPS)
            is_units = []
            for seed in seeds:
                is_units.append(
                    {
                        "cell": IMPLANT_SWEEP_V4_ANCHOR_SLUG,
                        "seed": seed,
                        "lr": CALIBRATION_LR_V3,
                        "phase": "implant_sweep_v4",
                        "implant_steps": implant_steps_csv,
                    }
                )
        # ── v6 + v4 shared: schedule the anchor pool + expand per-step records. ─
        # Both branches construct is_units with phase=implant_sweep_v4 + the
        # same IMPLANT_SWEEP_V4_ANCHOR_SLUG; v6 additionally threads
        # lora_r / lora_alpha (the scheduler already picks those up via
        # unit.get(...)). Sharing the schedule + expansion code prevents the
        # v6 branch from silently running 0 cells (the round-1 code-review
        # Critical #2: the v4 branch held both _schedule_unit_pool AND the
        # per-step expansion loop, so v6 just built is_units and dropped them).
        if impl_use_v6 or impl_use_v4:
            anchor_results = _schedule_unit_pool(
                units=is_units,
                n_gpus=args.n_gpus,
                max_parallel=args.max_parallel,
                slab_root=args.slab_root,
                runs_root=args.runs_root,
                log_dir=LOG_DIR,
                bank_path=args.bank_path,
                centroids_dir=args.centroids_dir,
                smoke=False,
                no_kl=args.no_kl,
                report_to=args.report_to,
                resume=args.resume,
            )
            # Expand per-seed anchor results into per-step records the v4
            # implant_only_axis_spearman_marker_channel_kl partial consumes.
            # The helper is module-scope so the v6 schedule + expansion path
            # can be pinned by a pure test (no tokenizer / GPU subprocess).
            implant_sweep_results.extend(_expand_implant_sweep_v4_anchor_results(anchor_results))
        if not (impl_use_v6 or impl_use_v4):
            log.info(
                "[phase=implant_sweep] v2 LR-sweep (legacy): %d cells (%d LRs × %d seeds "
                "at count=%d)",
                len(IMPLANT_SWEEP_SLUGS) * len(seeds),
                len(IMPLANT_SWEEP_LRS),
                len(seeds),
                ANCHOR_COUNT,
            )
            is_units = []
            for slug in IMPLANT_SWEEP_SLUGS:
                lr = lr_for_implant_sweep_slug(slug)
                for seed in seeds:
                    is_units.append(
                        {"cell": slug, "seed": seed, "lr": lr, "phase": "implant_sweep"}
                    )
            implant_sweep_results = _schedule_unit_pool(
                units=is_units,
                n_gpus=args.n_gpus,
                max_parallel=args.max_parallel,
                slab_root=args.slab_root,
                runs_root=args.runs_root,
                log_dir=LOG_DIR,
                bank_path=args.bank_path,
                centroids_dir=args.centroids_dir,
                smoke=False,
                no_kl=args.no_kl,
                report_to=args.report_to,
                resume=args.resume,
            )
            for r in implant_sweep_results:
                r["count"] = ANCHOR_COUNT
        if impl_use_v6:
            implant_routing = "v6_step_sweep_picked_rank"
        elif impl_use_v4:
            implant_routing = "v4_step_sweep"
        else:
            implant_routing = "v2_lr_sweep"
        phase_summaries["implant_sweep"] = {
            "n_completed": len(implant_sweep_results),
            "routing": implant_routing,
            "results": implant_sweep_results,
        }
        _write_sentinel(
            LOG_DIR / "issue-477-implant-sweep-results.json",
            kind="epm:progress",
            phase="implant_sweep",
            note_payload=phase_summaries["implant_sweep"],
        )
    else:
        log.info("[phase=implant_sweep] SKIP")

    _write_final_sentinel(phase_summaries, status="done")
    log.info("[phase=done] dispatcher exit %s", datetime.now(UTC).isoformat())
    return 0


def _write_final_sentinel(phase_summaries: dict, *, status: str) -> Path:
    """End-of-sweep poll_pipeline-compliant ``epm:results v1`` sentinel."""
    final_path = LOG_DIR / "issue-477-results.json"
    note_payload = {
        "issue": TASK_ID,
        "status": status,
        "phase_summaries": phase_summaries,
        "reproducibility": {
            "base_model": BASE_MODEL,
            "hf_model_repo": HF_MODEL_REPO,
            "hf_data_repo": HF_DATA_REPO,
        },
        "worktree_path": str(Path.cwd()),
        "final_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    _write_sentinel(final_path, kind="epm:results", phase="done", note_payload=note_payload)
    log.info("Final sentinel (epm:results v1): %s", final_path)
    return final_path


if __name__ == "__main__":
    sys.exit(main())
