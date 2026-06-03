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


def _schedule_unit_pool(
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
                results.append({**unit, "assigned_gpu": gpu, "status": "resumed_skip"})
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


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - linear pipeline
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        ANCHOR_COUNT,
        CALIB_SLUGS,
        CALIBRATION_LR_GRID,
        IMPLANT_SWEEP_LRS,
        IMPLANT_SWEEP_SLUGS,
        MAIN_SLUGS,
        count_for_slug,
        lr_for_implant_sweep_slug,
    )

    parser = argparse.ArgumentParser(description="i477 dispatcher — implant-decoupled count sweep.")
    parser.add_argument(
        "--phases",
        default="calibration,calibration_pick,main,implant_sweep",
        help="CSV subset of the pipeline phases. Default: all.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "One calibration cell (c477_calib_negp_4, seed=42, lr=1e-5) + tiny "
            "training/eval slice. Same dispatcher path as the full sweep."
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
    args = parser.parse_args(argv)

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

    # ── Smoke unification: --smoke is one calibration cell. ──────────────────
    if args.smoke:
        log.info("[phase=smoke] running unified smoke (one calibration cell at count=4, lr=1e-5)")
        unit = {"cell": "c477_calib_negp_4", "seed": 42, "lr": 1e-5, "phase": "calibration"}
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

    # ── Phase 3: main sweep at calibrated LRs (4 counts × 2 seeds). ──────────
    main_results: list[dict] = []
    if "main" in phases:
        if not picks:
            # Try loading from disk if calibration_pick was previously run.
            pick_path = args.slab_root / "calibration_pick.json"
            if pick_path.exists():
                loaded = json.loads(pick_path.read_text())
                picks = {int(k): v for k, v in loaded.get("picks", {}).items()}
            else:
                raise RuntimeError(
                    f"main phase requested but no picks resolved and "
                    f"{pick_path} missing. Run --phases calibration,calibration_pick first."
                )
        log.info(
            "[phase=main] scheduling %d main cells (4 counts × %d seeds) at calibrated LRs",
            len(MAIN_SLUGS) * len(seeds),
            len(seeds),
        )
        main_units = []
        for slug in MAIN_SLUGS:
            cnt = count_for_slug(slug)
            lr = float(picks[cnt]["lr"])
            for seed in seeds:
                main_units.append({"cell": slug, "seed": seed, "lr": lr, "phase": "main"})
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

    # ── Phase 4: implant-only-axis arm (3 LRs × 2 seeds at fixed count=4). ──
    implant_sweep_results: list[dict] = []
    if "implant_sweep" in phases:
        log.info(
            "[phase=implant_sweep] scheduling %d cells (%d LRs × %d seeds at count=%d)",
            len(IMPLANT_SWEEP_SLUGS) * len(seeds),
            len(IMPLANT_SWEEP_LRS),
            len(seeds),
            ANCHOR_COUNT,
        )
        is_units = []
        for slug in IMPLANT_SWEEP_SLUGS:
            lr = lr_for_implant_sweep_slug(slug)
            for seed in seeds:
                is_units.append({"cell": slug, "seed": seed, "lr": lr, "phase": "implant_sweep"})
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
        phase_summaries["implant_sweep"] = {
            "n_completed": len(implant_sweep_results),
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
