# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + × + − intentional
#!/usr/bin/env python3
"""Task #504 dispatcher — UNIFIED smoke = sweep with one cell (placement geometry).

Forked from scripts/dispatch_neg_geometry_472.py. Pipeline (plan §4):

  Phase 0.5  identification-gate pre-flight (CPU)              [CPU subprocess]
  Phase 0    anchor calibration smoke (3 cells × 1 seed)       [GPU subprocess pool]
  Phase 0a   pick the anchor (rank, ckpt-frac) from smoke      [CPU subprocess]
  Phase 1    main grid (5 arms × 2 seeds, pinned anchor)       [GPU subprocess pool]
  Phase 2    pooled partial-Spearman regression + figures      [CPU subprocess]

UNIFICATION (smoke-architecture parity = PASS_UNIFIED): smoke = the SAME
dispatcher running just the Phase 0 cells (3 × 1 seed) and the Phase 0a pick;
sweep adds Phase 1 + Phase 2. Per-cell unit is ALWAYS
``scripts/i504_run_cell.py`` (one subprocess, GPU-pinned via
CUDA_VISIBLE_DEVICES). Same subprocess shape, same env injection, same
on-policy DV path, same poll-compliant sentinel + ``[phase=...]`` logging,
same teardown sequence.

Pod-side discipline (CLAUDE.md): NEVER shells out to scripts/task.py
(sentinel-file pattern only); every subprocess.* passes env={**os.environ};
load_dotenv() at module top; vLLM phases are subprocess-isolated; sets
EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 (MooseFS quota) + EPM_PERSIST_ADAPTER_HF_REPO
so the adapter persists fail-loud before any cleanup.
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

# Defaults pulled from the #504 module (single source of truth). The
# load_dotenv() above must run BEFORE this import so HF_TOKEN / WANDB_API_KEY
# are in the env when downstream modules read them at import time.
from explore_persona_space.experiments.contrastive_neg_geometry_504 import (  # noqa: E402
    BASE_MODEL,
    EXPECTED_MARKER_TOKEN_ID,
    HF_DATA_PREFIX_504,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    MAIN_ARM_SLUGS,
    MARKER_TEXT,
    PHASE0_SMOKE_SLUGS,
    SEEDS,
    alpha_for_rank,
)

LOG_DIR = Path("/workspace/logs")

log = logging.getLogger("dispatch_neg_geometry_504")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _write_sentinel(path: Path, *, kind: str, phase: str, note_payload: dict) -> None:
    """poll_pipeline.py-compliant sentinel (sentinel_schema_version=1, kind, version)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": kind,
                "version": 1,
                "task_id": 504,
                "by": "dispatch_neg_geometry_504",
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


def _resolve_cells(raw: str | None, slabs: tuple[str, ...]) -> list[str]:
    """Resolve --cells CSV against the known #504 slugs (main + smoke)."""
    if raw is None or raw.strip() in ("", "all"):
        return list(slabs)
    out: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in slabs:
            out.append(tok)
        else:
            raise ValueError(f"Unknown #504 cell {tok!r}. Known: {sorted(slabs)}")
    return out


def _schedule_cell_pool(
    *,
    cells: list[str],
    seeds: list[int],
    n_gpus: int,
    max_parallel: int,
    slab_root: Path,
    runs_root: Path,
    log_dir: Path,
    bank_path: Path,
    centroids_dir: Path,
    arm_to_n_json: Path,
    r_train_path: Path,
    chosen_rank: int,
    chosen_alpha: int,
    chosen_frac: float | None,
    smoke: bool,
    no_kl: bool,
    report_to: str,
    resume: bool,
    max_new_tokens_eval: int,
    max_model_len_eval: int,
    label_prefix: str = "issue-504",
) -> list[dict]:
    """Run all (cell, seed) units as a GPU-sharded subprocess pool.

    Mirrors scripts/dispatch_neg_geometry_472.py. Each cell-subprocess is
    launched with ``--gpu-id <g>``; train/sft.py SETS
    ``CUDA_VISIBLE_DEVICES=str(g)`` so the cell + its nested eval run on
    physical GPU g (round-3 #472 fix). The nested eval subprocess inherits
    that CVD via os.environ. Free-GPU pool guarantees no two concurrent cells
    share a GPU.
    """
    units = [(c, s) for c in cells for s in seeds]
    if max_parallel > n_gpus:
        log.warning(
            "max_parallel=%d > n_gpus=%d would force >=2 concurrent cells onto one GPU "
            "(round-3 #472 OOM class); clamping max_parallel to %d.",
            max_parallel,
            n_gpus,
            n_gpus,
        )
        max_parallel = n_gpus
    log.info(
        "Scheduling %d (cell, seed) units across %d GPUs (max_parallel=%d)",
        len(units),
        n_gpus,
        max_parallel,
    )

    results: list[dict] = []
    running: list[tuple[subprocess.Popen, str, int, int]] = []
    queue = list(units)
    free_gpus: list[int] = list(range(n_gpus))

    def _launch(cell: str, seed: int, gpu: int) -> subprocess.Popen:
        out_traj = slab_root / f"{cell}_seed{seed}" / "trajectory.json"
        if resume and out_traj.exists():
            log.info("[%s seed%d] RESUME: trajectory exists; skipping.", cell, seed)
            return None  # type: ignore[return-value]
        env = {**os.environ}
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/i504_run_cell.py",
            "--cell",
            cell,
            "--seed",
            str(seed),
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
            "--arm-to-n-json",
            str(arm_to_n_json),
            "--r-train-path",
            str(r_train_path),
            "--chosen-rank",
            str(chosen_rank),
            "--chosen-alpha",
            str(chosen_alpha),
            "--max-new-tokens-eval",
            str(max_new_tokens_eval),
            "--max-model-len-eval",
            str(max_model_len_eval),
            "--report-to",
            report_to,
        ]
        if chosen_frac is not None:
            cmd.extend(["--chosen-frac", str(chosen_frac)])
        if smoke:
            cmd.append("--smoke")
        if no_kl:
            cmd.append("--no-kl")
        cell_log = log_dir / f"{label_prefix}-{cell}-seed{seed}.log"
        cell_log.parent.mkdir(parents=True, exist_ok=True)
        log.info("[%s seed%d] launch on GPU %d → %s", cell, seed, gpu, cell_log)
        fh = open(cell_log, "w")  # noqa: SIM115 -- handle outlives this function
        return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)

    while queue or running:
        while queue and len(running) < max_parallel and free_gpus:
            cell, seed = queue.pop(0)
            gpu = free_gpus.pop(0)
            proc = _launch(cell, seed, gpu)
            if proc is None:
                results.append({"cell": cell, "seed": seed, "status": "resumed_skip"})
                free_gpus.append(gpu)
                continue
            running.append((proc, cell, seed, gpu))
        still: list[tuple[subprocess.Popen, str, int, int]] = []
        for proc, cell, seed, gpu in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, cell, seed, gpu))
                continue
            free_gpus.append(gpu)
            if rc != 0:
                fail_path = log_dir / f"{label_prefix}-{cell}-seed{seed}-FAILED.json"
                fail_path.write_text(
                    json.dumps(
                        {"cell": cell, "seed": seed, "returncode": rc, "assigned_gpu": gpu},
                        indent=2,
                    )
                )
                for p2, _c2, _s2, _g2 in still:
                    p2.terminate()
                raise RuntimeError(
                    f"[{cell} seed{seed}] cell subprocess exited rc={rc} (GPU {gpu}). "
                    f"See {log_dir}/{label_prefix}-{cell}-seed{seed}.log. Sweep aborted."
                )
            log.info("[%s seed%d] DONE (GPU %d)", cell, seed, gpu)
            results.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "status": "done",
                    "assigned_gpu": gpu,
                    "trajectory_path": str(slab_root / f"{cell}_seed{seed}" / "trajectory.json"),
                    "adapter_hf_path": f"adapters/issue_504/{cell}_seed{seed}",
                }
            )
        running = still
        if running:
            time.sleep(5)
    return results


def _write_arm_to_n_json(phase05_report_path: Path, out_path: Path) -> Path:
    """Distill Phase 0.5's full report into the small JSON i504_run_cell.py reads."""
    report = json.loads(phase05_report_path.read_text())
    payload = {
        "arm_to_positioned_n": report.get("arm_to_positioned_n", {}),
        "smoke_mid_band_n": report.get("smoke_mid_band_n"),
        "held_out_panel": report.get("held_out_panel", []),
        "chosen_negatives": report.get("chosen_negatives", {}),
        "chosen_layer": report.get("chosen_layer"),
        "source": report.get("source"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- linear orchestrator
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        default=None,
        help="CSV of slugs (e.g. c504_near,c504_far) or 'all'. Default all 5 main arms.",
    )
    parser.add_argument(
        "--smoke-cells",
        default=",".join(PHASE0_SMOKE_SLUGS),
        help=f"CSV of smoke slugs. Default {','.join(PHASE0_SMOKE_SLUGS)}.",
    )
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run ONLY the Phase 0.5 + Phase 0 smoke cells + Phase 0 pick (no Phase 1, no "
            "Phase 2). Same per-cell unit (i504_run_cell.py --smoke) — UNIFICATION."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate imports + marker assertion + Phase 0.5 only; no Phase 0/1 GPU work.",
    )
    parser.add_argument("--no-kl", action="store_true", help="Skip DV-B KL.")
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_504"))
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_504"))
    parser.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    parser.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    parser.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    parser.add_argument(
        "--r-train-path",
        type=Path,
        default=Path("data/issue_472/on_policy_R/R_train.json"),
        help="On-policy R from #472 Phase 1 (max-length check + positives reuse).",
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/issue_504"))
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-phase05",
        action="store_true",
        help="Skip Phase 0.5 (expects phase0_5_gates.json to already exist).",
    )
    parser.add_argument(
        "--skip-phase0",
        action="store_true",
        help="Skip Phase 0 smoke (expects phase0_calibration.json to already exist).",
    )
    parser.add_argument(
        "--skip-phase07",
        action="store_true",
        help=(
            "Skip Phase 0.7 r-train fill (expects R_train_v504.json to already exist "
            "next to the input R_train.json)."
        ),
    )
    parser.add_argument(
        "--no-phase07-upload",
        action="store_true",
        help="Skip HF upload of the augmented R_train_v504.json (local artifact only).",
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip Phase 2 analyze (useful when only re-running training).",
    )
    parser.add_argument(
        "--max-new-tokens-eval", type=int, default=2048, help="Eval max_new_tokens."
    )
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

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    main_cells = _resolve_cells(args.cells, MAIN_ARM_SLUGS)
    smoke_cells = _resolve_cells(args.smoke_cells, PHASE0_SMOKE_SLUGS)
    log.info("Phase 0 smoke cells: %s", smoke_cells)
    log.info("Phase 1 main cells: %s seeds=%s (smoke_mode=%s)", main_cells, seeds, args.smoke)

    # ── Pre-flight: marker tokenizer assertion (CLAUDE.md). ──────────────────
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

    phase_summaries: dict[str, dict] = {}

    # ── Phase 0.5: identification-gate pre-flight (CPU). ─────────────────────
    phase05_path = args.slab_root / "phase0_5_gates.json"
    if not args.skip_phase05:
        _run_phase_subprocess(
            [
                "uv",
                "run",
                "python",
                "scripts/i504_phase_phase05.py",
                "--centroids-dir",
                str(args.centroids_dir),
                "--r-train-path",
                str(args.r_train_path),
                "--out-path",
                str(phase05_path),
                "--sentinel-path",
                str(LOG_DIR / "issue-504-phase05-results.json"),
            ],
            "phase05",
        )
    else:
        log.info("[phase=phase05] SKIP")
    if not phase05_path.exists():
        raise RuntimeError(f"phase05 expected {phase05_path} after run; not found.")
    phase05_report = json.loads(phase05_path.read_text())
    phase_summaries["phase05"] = {
        "verdict": phase05_report.get("verdict"),
        "chosen_layer": phase05_report.get("chosen_layer"),
        "arm_to_positioned_n": phase05_report.get("arm_to_positioned_n"),
        "smoke_mid_band_n": phase05_report.get("smoke_mid_band_n"),
        "max_length_check": phase05_report.get("max_length_check"),
    }
    if phase05_report.get("verdict") != "pass":
        log.error("[phase=phase05] gates FAILED — abort sweep, see %s.", phase05_path)
        _write_sentinel(
            LOG_DIR / "issue-504-results.json",
            kind="epm:results",
            phase="done",
            note_payload={
                "issue": 504,
                "status": "phase05_gates_failed",
                "phase_summaries": phase_summaries,
                "final_commit_sha": _git_sha(),
                "hostname": socket.gethostname(),
            },
        )
        return 2

    arm_to_n_json = args.slab_root / "arm_to_n.json"
    _write_arm_to_n_json(phase05_path, arm_to_n_json)
    log.info("[phase=phase05] wrote arm_to_n.json → %s", arm_to_n_json)

    # max_new_tokens bump if Phase 0.5 found a train-time R that saturated 1024.
    max_new_tokens_eval = args.max_new_tokens_eval
    if phase05_report.get("max_length_check", {}).get("needs_max_new_tokens_4096"):
        max_new_tokens_eval = max(max_new_tokens_eval, 4096)
        log.info(
            "[phase=phase05] max-length check tripped — bumping eval max_new_tokens to %d.",
            max_new_tokens_eval,
        )
    # Round-2 fix (blocker #4): vLLM max_model_len (= prompt + generation cap)
    # MUST track max_new_tokens_eval. With max_new_tokens=4096 and
    # max_model_len=2048, vLLM silently caps generation at 2048 - prompt_len —
    # the headline DV becomes a silent-zero artifact (#260 precedent class).
    # Floor at 2048 (vLLM minimum across the rest of the rig) and add 512
    # headroom for the prompt (longer than the longest EVAL_QUESTION prefix +
    # system-prompt). i504_run_cell.py recomputes per-cell under --smoke.
    eval_prompt_headroom = 512
    max_model_len_eval = max(2048, max_new_tokens_eval + eval_prompt_headroom)
    log.info(
        "[phase=phase05] eval max_new_tokens=%d, max_model_len=%d (headroom=%d).",
        max_new_tokens_eval,
        max_model_len_eval,
        eval_prompt_headroom,
    )

    if args.dry_run:
        log.info("[phase=done] DRY-RUN complete (imports + Phase 0.5 only). %s", datetime.now(UTC))
        return 0

    # ── Phase 0.7: r-train fill for newly-picked positioned negs (round-8 fix). ─
    # Phase 0.5 may pick positioned negatives that #472's published R_train.json
    # does NOT cover (e.g. round-6 mean-centering picked `origami_artist` +
    # `prosecutor`). Generate on-policy R for the missing personas, write to
    # `R_train_v504.json` (preserves #472's R_train.json byte-identical), and
    # repoint downstream phases at the augmented artifact. Skip on
    # --skip-phase07 if the v504 artifact already exists.
    r_train_v504_path = args.r_train_path.with_name("R_train_v504.json")
    if not args.skip_phase07:
        fill_cmd = [
            "uv",
            "run",
            "python",
            "scripts/i504_phase_r_generate_fill.py",
            "--phase05-path",
            str(phase05_path),
            "--input-r-train-path",
            str(args.r_train_path),
            "--output-r-train-path",
            str(r_train_v504_path),
            "--bank-path",
            str(args.bank_path),
            "--sentinel-path",
            str(LOG_DIR / "issue-504-phase07-results.json"),
        ]
        if args.no_phase07_upload:
            fill_cmd.append("--no-upload")
        _run_phase_subprocess(fill_cmd, "phase07_r_train_fill")
    else:
        log.info("[phase=phase07] SKIP (--skip-phase07)")
    if not r_train_v504_path.exists():
        raise RuntimeError(
            f"Phase 0.7 expected augmented R_train at {r_train_v504_path}; not found. "
            f"Re-run without --skip-phase07 OR pre-stage the v504 artifact."
        )
    # Repoint args.r_train_path at the augmented artifact for the rest of the
    # dispatcher — i504_run_cell.py reads --r-train-path, and the load_r_artifact
    # schema is identical (only the `completions` map is augmented).
    args.r_train_path = r_train_v504_path
    log.info("[phase=phase07] downstream cells will read R_train from %s", args.r_train_path)
    phase_summaries["phase07"] = {
        "r_train_path": str(r_train_v504_path),
        "status": "filled" if not args.skip_phase07 else "skipped",
    }

    # ── Phase 0: smoke (3 cells × 1 seed) at a placeholder rank/α each. ─────
    # The Phase 0 smoke needs SEPARATE (rank, α) per cell (r=4/α=23, r=8/α=32,
    # r=16/α=32). We launch them sequentially with the right pinned values.
    phase0_pick_path = args.slab_root / "phase0_calibration.json"
    if not args.skip_phase0:
        for slug in smoke_cells:
            # rank from slug "c504_smoke_r{R}".
            rank_str = slug.removeprefix("c504_smoke_r")
            try:
                rank = int(rank_str)
            except ValueError as e:
                raise RuntimeError(f"Cannot parse rank from smoke slug {slug!r}.") from e
            try:
                alpha = alpha_for_rank(rank) if rank != 16 else 32
            except KeyError:
                alpha = 32
            log.info("[phase=phase0] smoke cell %s (rank=%d, alpha=%d)", slug, rank, alpha)
            _schedule_cell_pool(
                cells=[slug],
                seeds=[42],
                n_gpus=args.n_gpus,
                max_parallel=1,  # sequential within Phase 0 — each cell gets its own rank/α.
                slab_root=args.slab_root,
                runs_root=args.runs_root,
                log_dir=LOG_DIR,
                bank_path=args.bank_path,
                centroids_dir=args.centroids_dir,
                arm_to_n_json=arm_to_n_json,
                r_train_path=args.r_train_path,
                chosen_rank=rank,
                chosen_alpha=alpha,
                chosen_frac=None,  # Phase 0 hasn't picked yet; train all 6 fracs.
                smoke=False,  # Phase 0 trains at the Phase 1 composition (NOT a tiny slice).
                no_kl=args.no_kl,
                report_to=args.report_to,
                resume=args.resume,
                max_new_tokens_eval=max_new_tokens_eval,
                max_model_len_eval=max_model_len_eval,
                label_prefix="issue-504-phase0",
            )
        # Now run the pick rule over the 3 smoke trajectories.
        _run_phase_subprocess(
            [
                "uv",
                "run",
                "python",
                "scripts/i504_phase_phase0_pick.py",
                "--slab-root",
                str(args.slab_root),
                "--out-path",
                str(phase0_pick_path),
                "--sentinel-path",
                str(LOG_DIR / "issue-504-phase0-pick-results.json"),
            ],
            "phase0_pick",
        )
    else:
        log.info("[phase=phase0] SKIP")
    if not phase0_pick_path.exists():
        raise RuntimeError(f"phase0 pick expected at {phase0_pick_path}; not found.")
    pick = json.loads(phase0_pick_path.read_text())
    phase_summaries["phase0"] = {
        "verdict": pick.get("verdict"),
        "chosen_rank": pick.get("chosen_rank"),
        "chosen_alpha": pick.get("chosen_alpha"),
        "chosen_checkpoint_fraction": pick.get("chosen_checkpoint_fraction"),
        "source_delta_g_at_pick_nats": pick.get("source_delta_g_at_pick_nats"),
    }
    if pick.get("verdict") != "pass":
        log.error("[phase=phase0_pick] FAIL — abort, see %s", phase0_pick_path)
        _write_sentinel(
            LOG_DIR / "issue-504-results.json",
            kind="epm:results",
            phase="done",
            note_payload={
                "issue": 504,
                "status": "phase0_anchor_unavailable",
                "phase_summaries": phase_summaries,
                "final_commit_sha": _git_sha(),
                "hostname": socket.gethostname(),
            },
        )
        return 2

    if args.smoke:
        # Smoke mode = Phase 0.5 + Phase 0 + Phase 0 pick only, no Phase 1.
        _write_sentinel(
            LOG_DIR / "issue-504-results.json",
            kind="epm:results",
            phase="done",
            note_payload={
                "issue": 504,
                "status": "smoke_complete",
                "phase_summaries": phase_summaries,
                "final_commit_sha": _git_sha(),
                "hostname": socket.gethostname(),
            },
        )
        log.info(
            "[phase=done] SMOKE COMPLETE (phase05 + phase0 + pick). %s",
            datetime.now(UTC).isoformat(),
        )
        return 0

    # ── Phase 1: main sweep at the pinned (chosen_rank, chosen_alpha). ───────
    log.info(
        "[phase=phase1] scheduling %d cells × %d seeds at rank=%d α=%d frac=%s",
        len(main_cells),
        len(seeds),
        pick["chosen_rank"],
        pick["chosen_alpha"],
        pick["chosen_checkpoint_fraction"],
    )
    cell_results = _schedule_cell_pool(
        cells=main_cells,
        seeds=seeds,
        n_gpus=args.n_gpus,
        max_parallel=args.max_parallel,
        slab_root=args.slab_root,
        runs_root=args.runs_root,
        log_dir=LOG_DIR,
        bank_path=args.bank_path,
        centroids_dir=args.centroids_dir,
        arm_to_n_json=arm_to_n_json,
        r_train_path=args.r_train_path,
        chosen_rank=pick["chosen_rank"],
        chosen_alpha=pick["chosen_alpha"],
        chosen_frac=pick["chosen_checkpoint_fraction"],
        smoke=False,
        no_kl=args.no_kl,
        report_to=args.report_to,
        resume=args.resume,
        max_new_tokens_eval=max_new_tokens_eval,
        max_model_len_eval=max_model_len_eval,
        label_prefix="issue-504",
    )
    phase_summaries["phase1"] = {
        "n_completed": len(cell_results),
        "results": cell_results,
    }

    # ── Phase 2: analyze (CPU). ──────────────────────────────────────────────
    # Round-2 fix (blocker #2): pin --base-prior-path so the per-probe
    # base_prior_marker covariate is aggregated from the trajectory artifacts
    # and persisted to a canonical location (i504_phase_analyze.py auto-builds
    # it from the trajectory b_logp values if the file is missing). Without
    # this the #500 sign-flip discipline (plan §6.2 test 6) is INACTIVE.
    base_prior_path = args.slab_root / "base_prior_marker.json"
    analyze_summary: dict | None = None
    if args.skip_analyze:
        log.info("[phase=analyze] SKIP")
    else:
        _run_phase_subprocess(
            [
                "uv",
                "run",
                "python",
                "scripts/i504_phase_analyze.py",
                "--slab-root",
                str(args.slab_root),
                "--phase0-path",
                str(phase0_pick_path),
                "--phase05-path",
                str(phase05_path),
                "--base-prior-path",
                str(base_prior_path),
                "--seeds",
                ",".join(str(s) for s in seeds),
                "--sentinel-path",
                str(LOG_DIR / "issue-504-analyze-results.json"),
            ],
            "analyze",
        )
        ap = args.slab_root / "analyze_summary.json"
        if ap.exists():
            analyze_summary = json.loads(ap.read_text())
    phase_summaries["analyze"] = analyze_summary

    _write_final_sentinel(
        main_cells,
        cell_results,
        phase_summaries,
        analyze_summary,
        seeds,
        args.slab_root,
        status="done",
    )
    log.info(
        "Dispatcher done. %d cell units completed. [phase=done] %s",
        len(cell_results),
        datetime.now(UTC).isoformat(),
    )
    return 0


def _write_final_sentinel(
    cells: list[str],
    cell_results: list[dict],
    phase_summaries: dict,
    analyze_summary: dict | None,
    seeds: list[int],
    slab_root: Path,
    *,
    status: str,
) -> Path:
    """End-of-sweep poll_pipeline-compliant ``epm:results v1`` sentinel."""
    final_path = LOG_DIR / "issue-504-results.json"
    note_payload = {
        "issue": 504,
        "status": status,
        "seeds": seeds,
        "cells_requested": cells,
        "n_units_completed": len(
            [c for c in cell_results if c.get("status") in ("done", "resumed_skip")]
        ),
        "n_units": len(cells) * len(seeds),
        "cell_results": cell_results,
        "n_rows_pooled": (analyze_summary or {}).get("n_rows_pooled"),
        "chosen_checkpoint_fraction": (analyze_summary or {}).get("chosen_checkpoint_fraction"),
        "notes": (analyze_summary or {}).get("notes", []),
        "reproducibility": {
            "base_model": BASE_MODEL,
            "hf_model_repo": HF_MODEL_REPO,
            "hf_data_repo": HF_DATA_REPO,
            "hf_data_prefix": HF_DATA_PREFIX_504,
            "adapter_paths": {
                f"{c['cell']}_seed{c['seed']}": (
                    f"{HF_MODEL_REPO}/tree/main/{c.get('adapter_hf_path', '?')}"
                )
                for c in cell_results
                if "adapter_hf_path" in c
            },
        },
        "worktree_path": str(Path.cwd()),
        "final_commit_sha": _git_sha(),
        "phase_summaries": phase_summaries,
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    _write_sentinel(final_path, kind="epm:results", phase="done", note_payload=note_payload)
    log.info("Final sentinel (epm:results v1): %s", final_path)
    return final_path


if __name__ == "__main__":
    sys.exit(main())
