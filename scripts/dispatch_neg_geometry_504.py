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
    CHECKPOINT_FRACTIONS_V3_FINER,
    CHECKPOINT_FRACTIONS_V4_BISECTION,
    EPOCHS_FROM_V3_SMOKE_SLUG,
    EPOCHS_LADDER_V3,
    EXPECTED_MARKER_TOKEN_ID,
    FALLBACK_SOURCE_CANDIDATES,
    FIXED_LR_V3,
    HF_DATA_PREFIX_504,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    LR_FROM_V2_SMOKE_SLUG,
    LR_LADDER,
    MAIN_ARM_SLUGS,
    MAIN_ARM_SLUGS_V2,
    MAIN_ARM_SLUGS_V3,
    MARKER_TEXT,
    PHASE0_SMOKE_SLUGS,
    PHASE0_SMOKE_SLUGS_V2,
    PHASE0_SMOKE_SLUGS_V3,
    SEEDS,
    SOURCE_PERSONA,
    alpha_for_rank,
)

# Force-reference v2/v3 symbols so ruff's F401 auto-strip on the formatter
# pre-commit pass doesn't remove them — see `feedback_ruff_strips_unused_imports`.
# All are used inside `main()` / `_run_v2_phase*()` / `_run_v3_phase*()` below;
# this tuple keeps the imports alive even when the formatter rewrites the
# import block.
# Round-2 fix (Concern C): the v1 `MAIN_ARM_SLUGS` is now used inside the v1
# branch only (no longer at the top of `main()`), so keep it referenced here
# too, alongside the v2 symbols.
_V2_IMPORT_REFS = (
    FALLBACK_SOURCE_CANDIDATES,
    LR_FROM_V2_SMOKE_SLUG,
    LR_LADDER,
    MAIN_ARM_SLUGS,
    MAIN_ARM_SLUGS_V2,
    PHASE0_SMOKE_SLUGS_V2,
    SOURCE_PERSONA,
)
_V3_IMPORT_REFS = (
    CHECKPOINT_FRACTIONS_V3_FINER,
    EPOCHS_FROM_V3_SMOKE_SLUG,
    EPOCHS_LADDER_V3,
    FIXED_LR_V3,
    MAIN_ARM_SLUGS_V3,
    PHASE0_SMOKE_SLUGS_V3,
)
_V4_IMPORT_REFS = (CHECKPOINT_FRACTIONS_V4_BISECTION,)

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


def _schedule_cell_pool(  # noqa: C901 -- linear pool: per-flag conditionals on a single launch path
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
    r_eval_path: Path,
    chosen_rank: int,
    chosen_alpha: int,
    chosen_frac: float | None,
    smoke: bool,
    no_kl: bool,
    report_to: str,
    resume: bool,
    max_new_tokens_eval: int,
    max_model_len_eval: int,
    hf_path_suffix: str = "",
    label_prefix: str = "issue-504",
    chosen_lr: float | None = None,
    per_cell_lrs: dict[str, float] | None = None,
    chosen_epochs: int | None = None,
    per_cell_epochs: dict[str, int] | None = None,
    source_persona: str | None = None,
    per_cell_tolerant: bool = False,
    checkpoint_fractions: tuple[float, ...] | None = None,
    trajectory_suffix: str = "",
) -> list[dict]:
    """Run all (cell, seed) units as a GPU-sharded subprocess pool.

    Mirrors scripts/dispatch_neg_geometry_472.py. Each cell-subprocess is
    launched with ``--gpu-id <g>``; train/sft.py SETS
    ``CUDA_VISIBLE_DEVICES=str(g)`` so the cell + its nested eval run on
    physical GPU g (round-3 #472 fix). The nested eval subprocess inherits
    that CVD via os.environ. Free-GPU pool guarantees no two concurrent cells
    share a GPU.

    Failure semantics (round-4 fix, task #504, 2026-06-08):
      * ``per_cell_tolerant=False`` (default, Phase 1 main grid): on the first
        cell rc!=0, terminate all still-running siblings and raise
        ``RuntimeError`` — main-grid production must be all-or-nothing.
      * ``per_cell_tolerant=True`` (Phase 0 / Phase 0-fallback lr-ladder
        smoke): on a cell rc!=0, record the failure in
        ``<log_dir>/<label_prefix>-cell-failures.json`` (one row per failed
        cell: cell, seed, rc, log path, assigned_gpu) AND in the returned
        ``results`` list (status="failed"), then continue to the remaining
        cells. The post-smoke picker reads ALL cells that produced a
        trajectory and decides via the anti-saturation band whether any
        (lr, frac) pair is in band; if all cells failed or all in-band
        candidates landed at floor, the picker emits a fallback verdict. Phase
        0 design requires per-cell independence; the round-3 crash was the
        ladder aborting on cell #1 before cells #2-#3 could even launch.
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
        # Trajectory path is intentionally NOT suffixed by hf_path_suffix —
        # Phase 0's pick rule (phase0.py) AND Phase 2's analyze (analyze.py)
        # both read the canonical un-suffixed path. A future operator passing
        # --resume on a pod with a stale round-N trajectory at this path would
        # skip launching the cell; the round-15 launcher does NOT pass
        # --resume, so the asymmetry is currently safe.
        # v3 in-plan recovery (plan §4.1 + §4.2): when `trajectory_suffix` is
        # set, the recovery cell writes to a DIFFERENT slab-root subdir so the
        # coarse trajectory survives intact. The picker's merge step reads
        # BOTH `<slug>_seed<S>/trajectory.json` (coarse) AND
        # `<slug>_seed<S><trajectory_suffix>/trajectory.json` (finer).
        traj_subdir = f"{cell}_seed{seed}{trajectory_suffix}"
        out_traj = slab_root / traj_subdir / "trajectory.json"
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
            "--r-eval-path",
            str(r_eval_path),
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
        # v2 lr threading (plan v2 §10). Priority order:
        #   1. per_cell_lrs[cell] — used by the v2 Phase 0 lr-ladder smoke
        #      (one lr per smoke cell, looked up from LR_FROM_V2_SMOKE_SLUG).
        #   2. chosen_lr — used by the v2 Phase 1 main grid (one lr applied
        #      uniformly to all 5 arms × 2 seeds, picked by Phase 0 pick).
        # If neither is set, --lr is omitted and i504_run_cell.py falls back
        # to ANCHOR_LR (the v1 default = 2e-6).
        cell_lr: float | None = None
        if per_cell_lrs and cell in per_cell_lrs:
            cell_lr = per_cell_lrs[cell]
        elif chosen_lr is not None:
            cell_lr = chosen_lr
        if cell_lr is not None:
            cmd.extend(["--lr", repr(cell_lr)])
        # v3 EPOCHS threading (plan v3 §4.1). Priority order mirrors lr:
        #   1. per_cell_epochs[cell] — used by the v3 Phase 0 EPOCHS-ladder
        #      smoke (one epoch value per smoke cell, looked up from
        #      EPOCHS_FROM_V3_SMOKE_SLUG).
        #   2. chosen_epochs — used by the v3 Phase 1 main grid (one epoch
        #      value applied uniformly to all 5 arms × 2 seeds, picked by
        #      Phase 0 v3 pick).
        # If neither is set, --epochs is omitted and i504_run_cell.py falls
        # back to the module-level EPOCHS default (=1, v1/v2 default).
        cell_epochs: int | None = None
        if per_cell_epochs and cell in per_cell_epochs:
            cell_epochs = per_cell_epochs[cell]
        elif chosen_epochs is not None:
            cell_epochs = chosen_epochs
        if cell_epochs is not None:
            cmd.extend(["--epochs", str(cell_epochs)])
        # Round-2 fix (BLOCKER #2): thread fallback-source through to the
        # per-cell runner. When None, i504_run_cell.py falls back to the
        # SOURCE_PERSONA module default (= villain, v1/v2 legacy default).
        if source_persona is not None:
            cmd.extend(["--source", source_persona])
        if smoke:
            cmd.append("--smoke")
        if no_kl:
            cmd.append("--no-kl")
        if hf_path_suffix:
            cmd.extend(["--hf-path-suffix", hf_path_suffix])
        # v3 in-plan recovery (plan §4.1 trigger B + §4.2): when set, the
        # recovery phase passes a finer-grid fraction tuple (e.g.
        # CHECKPOINT_FRACTIONS_V3_FINER = (0.02, 0.04, 0.06, 0.08)) plus a
        # trajectory-suffix so the recovery cell's output does NOT clobber
        # the coarse trajectory. The picker's merge step reads both.
        if checkpoint_fractions is not None:
            cmd.extend(["--checkpoint-fractions", ",".join(repr(f) for f in checkpoint_fractions)])
        if trajectory_suffix:
            cmd.extend(["--trajectory-suffix", trajectory_suffix])
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
                cell_log_path = log_dir / f"{label_prefix}-{cell}-seed{seed}.log"
                fail_path = log_dir / f"{label_prefix}-{cell}-seed{seed}-FAILED.json"
                fail_row = {
                    "cell": cell,
                    "seed": seed,
                    "returncode": rc,
                    "assigned_gpu": gpu,
                    "log_path": str(cell_log_path),
                }
                fail_path.write_text(json.dumps(fail_row, indent=2))
                if per_cell_tolerant:
                    # Round-4 fix (BLOCKER #2 / task #504, 2026-06-08): Phase 0
                    # smoke is per-cell tolerant — log + record + continue so
                    # the picker sees every cell's outcome. Append to a single
                    # manifest under the same label_prefix so the post-smoke
                    # picker (i504_phase_phase0_pick.py) can read which cells
                    # failed alongside which produced trajectories.
                    manifest_path = log_dir / f"{label_prefix}-cell-failures.json"
                    manifest_rows: list[dict] = []
                    if manifest_path.exists():
                        try:
                            manifest_rows = json.loads(manifest_path.read_text())
                            if not isinstance(manifest_rows, list):
                                manifest_rows = []
                        except json.JSONDecodeError:
                            manifest_rows = []
                    manifest_rows.append(fail_row)
                    manifest_path.write_text(json.dumps(manifest_rows, indent=2))
                    log.warning(
                        "[%s seed%d] cell subprocess exited rc=%d (GPU %d) — per-cell "
                        "tolerant mode: recorded in %s, continuing to remaining cells. "
                        "Log: %s",
                        cell,
                        seed,
                        rc,
                        gpu,
                        manifest_path,
                        cell_log_path,
                    )
                    results.append(
                        {
                            "cell": cell,
                            "seed": seed,
                            "status": "failed",
                            "returncode": rc,
                            "assigned_gpu": gpu,
                            "log_path": str(cell_log_path),
                            "hf_path_suffix": hf_path_suffix,
                        }
                    )
                    continue
                for p2, _c2, _s2, _g2 in still:
                    p2.terminate()
                raise RuntimeError(
                    f"[{cell} seed{seed}] cell subprocess exited rc={rc} (GPU {gpu}). "
                    f"See {cell_log_path}. Sweep aborted."
                )
            log.info("[%s seed%d] DONE (GPU %d)", cell, seed, gpu)
            # Round-15 loop-2 fix: thread hf_path_suffix into the dispatcher's
            # cell_results so the final issue-504-results.json sentinel
            # (consumed by _write_final_sentinel at lines ~807-812, which is
            # the canonical reproducibility ledger downstream automation —
            # analyzer / clean-result / upload-verifier — reads) publishes the
            # SUFFIXED HF path matching the per-cell sentinel + the actual
            # upload site in i504_run_cell.py:303,409. Without this thread the
            # final sentinel pointed reviewers at the un-suffixed round-13/14
            # adapters instead of the round-15 __r15 adapters.
            run_slug = f"{cell}_seed{seed}{hf_path_suffix}"
            results.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "status": "done",
                    "assigned_gpu": gpu,
                    "trajectory_path": str(slab_root / f"{cell}_seed{seed}" / "trajectory.json"),
                    "adapter_hf_path": f"adapters/issue_504/{run_slug}",
                    "hf_path_suffix": hf_path_suffix,
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
    parser.add_argument(
        "--r-eval-path",
        type=Path,
        default=Path("data/issue_472/on_policy_R/R_eval.json"),
        help=(
            "On-policy R_eval from #472. Phase 0.7 will augment with the panel "
            "personas #472 didn't cover and repoint to R_eval_v504.json."
        ),
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
    parser.add_argument(
        "--hf-path-suffix",
        default="",
        help=(
            "Round-15 strengthen-anchor knob (default empty = byte-identical "
            "pre-round-15 behavior). When set, appended to BOTH the local "
            "runs-root subdir (`/workspace/runs/issue_504/<slug>_seed<S>"
            "<suffix>/`) AND the HF model-repo subfolder (`adapters/"
            "issue_504/<slug>_seed<S><suffix>`) for EVERY cell in this "
            "dispatcher invocation (Phase 0 smoke AND Phase 1 main). "
            "Preserves the round-13/14 dispositive-A/B adapters on HF at "
            "the canonical un-suffixed path. Round-15 launcher passes "
            "`--hf-path-suffix __r15`."
        ),
    )
    # ── v2 CLI (plan v2 §4.1 + §10) ─────────────────────────────────────────
    # `--phase` selects the v2 pipeline phase. When unset (default), the
    # dispatcher runs the legacy v1 (rank-ladder) pipeline byte-identically.
    # When set, it runs ONE v2 phase and exits.
    parser.add_argument(
        "--phase",
        choices=(
            "legacy",
            "phase0",
            "phase0-fallback",
            "phase0_v3",
            "phase0_v3-recovery",
            "phase0_v4_pretrain",
            "phase0_v4_reeval",
            "phase0_v4_bisection",
            "phase0p6_validate",
            "phase1",
        ),
        default="legacy",
        help=(
            "Phase to run. "
            "phase0=v2 lr-ladder smoke (3 cells) + pick (writes "
            "phase0_calibration_v2.json); "
            "phase0-fallback=v2 rerun of the lr-ladder smoke on a fallback "
            "source persona (--source); "
            "phase0_v3=v3 EPOCHS-ladder smoke (2 cells × 1 seed at fixed "
            "lr=1e-4, EPOCHS in --epochs-ladder) + pick (writes "
            "phase0_calibration_v3.json + on Trigger A/C also "
            "phase0_v3_exit_to_v4.json); "
            "phase0_v3-recovery=v3 in-plan finer-fraction recovery on EPOCHS=2 "
            "(triggered only when Trigger B fired); "
            "phase0_v4_pretrain=v4 re-train EPOCHS=3 seed=42 ONCE with "
            "per-fraction HF trajectory persistence (plan v5 §4.0); writes "
            "phase0_trajectory_v4.json; "
            "phase0_v4_reeval=v4 re-eval the EPOCHS=3 anchor through the "
            "fixed reader + bystander-resolution picker (plan v5 §4.1 fix "
            "#2); writes phase0_calibration_v4.json; "
            "phase0_v4_bisection=v4 EPOCHS=2 finer-fraction fallback "
            "(plan v5 §4.2 step 1); triggered ONLY when phase0_v4_reeval "
            "returns verdict='no_in_band_anchor'. Re-trains EPOCHS=2 at "
            "fractions {0.04, 0.08, 0.12, 0.16} with v4 trajectory "
            "persistence, re-applies the bystander-resolution picker, "
            "writes phase0_calibration_v4_bisection.json; "
            "phase0p6_validate=v4 marker-logprob path validation (5 probes "
            "× 4 questions × 1 ckpt; plan v5 §4.3a); writes "
            "phase0p6_validation_v4.json; "
            "phase1=read whichever Phase 0 artifact exists "
            "(v4 > v4_bisection > v3 > v2-fallback > v2-primary) + "
            "train the 5 main arms × 2 seeds at the picked recipe; "
            "legacy=byte-identical v1 (rank-ladder) pipeline. Default legacy."
        ),
    )
    parser.add_argument(
        "--lr-ladder",
        default=",".join(f"{lr:g}" for lr in LR_LADDER),
        help=(
            "v2 lr ladder for --phase phase0 / phase0-fallback (plan v2 §4.1). "
            f"CSV of 3 floats (default: {','.join(f'{lr:g}' for lr in LR_LADDER)}). "
            "Each value maps to one v2 smoke slug "
            "(c504v2_smoke_lr{1e5,3e5,1e4} for the canonical ladder)."
        ),
    )
    parser.add_argument(
        "--epochs-ladder",
        default=",".join(str(e) for e in EPOCHS_LADDER_V3),
        help=(
            "v3 EPOCHS ladder for --phase phase0_v3 (plan v3 §4.1). "
            f"CSV of integers (default: {','.join(str(e) for e in EPOCHS_LADDER_V3)}). "
            "Each value maps to one v3 smoke slug "
            "(c504v3_smoke_eps{2,3} for the canonical ladder). v3 holds lr "
            "FIXED at --fixed-lr (default 1e-4) and sweeps EPOCHS as the "
            "single load-bearing variable."
        ),
    )
    parser.add_argument(
        "--fixed-lr",
        type=float,
        default=FIXED_LR_V3,
        help=(
            "v3 fixed lr for --phase phase0_v3 (plan v3 §4.1). NOT swept "
            f"in v3 — pinned by v2 phase0 evidence (default: {FIXED_LR_V3:g})."
        ),
    )
    parser.add_argument(
        "--source",
        default=SOURCE_PERSONA,
        help=(
            "Source persona name. Default villain (plan v2 §10, v3 §10). "
            "For --phase phase0-fallback, override with the easier-source "
            f"candidate (see FALLBACK_SOURCE_CANDIDATES = {FALLBACK_SOURCE_CANDIDATES})."
        ),
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
    # Round-2 fix (Concern C): defer v1 cell-slug resolution into the v1 branch.
    # The previous unconditional `_resolve_cells(args.cells, MAIN_ARM_SLUGS)`
    # would raise on any token not in MAIN_ARM_SLUGS — but a perfectly valid v2
    # invocation like `--phase phase1 --cells c504v2_near` carries v2 slugs,
    # which `_resolve_cells` against MAIN_ARM_SLUGS rejects BEFORE the v2 router
    # runs. The v2 router (`_run_v2_phase1`) calls `_resolve_cells(args.cells,
    # MAIN_ARM_SLUGS_V2)` itself, so the v1 resolution belongs in the v1 branch.
    log.info("Phase 1 seeds=%s (smoke_mode=%s) — cells resolution deferred", seeds, args.smoke)

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
    # Round-2 fix (BLOCKER #2): thread --source into Phase 0.5 so positioned
    # negatives are picked RELATIVE TO the effective source. For --phase
    # phase0-fallback the source has already been swapped via args.source;
    # Phase 0.5 then re-emits a fresh phase0_5_gates.json keyed off that
    # source. For the primary phase (--phase legacy/phase0/phase1), this
    # passes the v1/v2 default = villain (byte-identical).
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
                "--source",
                args.source,
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

    # ── Phase 0.7: r-fill for both TRAIN and EVAL (round-11 fix). ────────────
    # Phase 0.5 may pick positioned negatives that #472's R_train.json does NOT
    # cover (round-8 fix: train side). The eval trajectory rig also probes the
    # FULL held-out panel (~54 personas), and #472's R_eval.json doesn't cover
    # every panel persona either (round-11 fix: eval side). Phase 0.7 fills
    # BOTH symmetrically in one vLLM-engine invocation, writes to v504-suffixed
    # paths (preserves #472's originals byte-identical), and repoints both
    # downstream args. Skip on --skip-phase07 if the v504 artifacts already exist.
    r_train_v504_path = args.r_train_path.with_name("R_train_v504.json")
    r_eval_v504_path = args.r_eval_path.with_name("R_eval_v504.json")
    if not args.skip_phase07:
        fill_cmd = [
            "uv",
            "run",
            "python",
            "scripts/i504_phase_r_generate_fill.py",
            "--phase05-path",
            str(phase05_path),
            "--split",
            "both",
            "--input-r-train-path",
            str(args.r_train_path),
            "--output-r-train-path",
            str(r_train_v504_path),
            "--input-r-eval-path",
            str(args.r_eval_path),
            "--output-r-eval-path",
            str(r_eval_v504_path),
            "--bank-path",
            str(args.bank_path),
            "--sentinel-path",
            str(LOG_DIR / "issue-504-phase07-results.json"),
        ]
        if args.no_phase07_upload:
            fill_cmd.append("--no-upload")
        _run_phase_subprocess(fill_cmd, "phase07_r_fill_train_and_eval")
    else:
        log.info("[phase=phase07] SKIP (--skip-phase07)")
    if not r_train_v504_path.exists():
        raise RuntimeError(
            f"Phase 0.7 expected augmented R_train at {r_train_v504_path}; not found. "
            f"Re-run without --skip-phase07 OR pre-stage the v504 artifact."
        )
    if not r_eval_v504_path.exists():
        raise RuntimeError(
            f"Phase 0.7 expected augmented R_eval at {r_eval_v504_path}; not found. "
            f"Re-run without --skip-phase07 OR pre-stage the v504 artifact."
        )
    # Repoint args.r_{train,eval}_path at the augmented artifacts for the rest
    # of the dispatcher — i504_run_cell.py reads --r-train-path and
    # --r-eval-path, and the load_r_artifact schema is identical (only the
    # `completions` map is augmented).
    args.r_train_path = r_train_v504_path
    args.r_eval_path = r_eval_v504_path
    log.info(
        "[phase=phase07] downstream cells will read R_train from %s, R_eval from %s",
        args.r_train_path,
        args.r_eval_path,
    )
    phase_summaries["phase07"] = {
        "r_train_path": str(r_train_v504_path),
        "r_eval_path": str(r_eval_v504_path),
        "status": "filled" if not args.skip_phase07 else "skipped",
    }

    # ── v2 phase routing (plan v2 §4.1 + §10) ────────────────────────────────
    # When --phase != legacy, route to the v2 pipeline and return. The v2
    # router resolves its OWN cells against MAIN_ARM_SLUGS_V2 inside
    # `_run_v2_phase1`.
    if args.phase != "legacy":
        return _run_v2_phase(
            args=args,
            phase_summaries=phase_summaries,
            arm_to_n_json=arm_to_n_json,
            max_new_tokens_eval=max_new_tokens_eval,
            max_model_len_eval=max_model_len_eval,
            seeds=seeds,
        )

    # Round-2 fix (Concern C): resolve v1 cells HERE (after the v2 router has
    # had its early-return chance). A v2 invocation that overrides --cells with
    # c504v2_* slugs no longer trips _resolve_cells's "Unknown #504 cell"
    # ValueError before the router fires.
    main_cells = _resolve_cells(args.cells, MAIN_ARM_SLUGS)
    smoke_cells = _resolve_cells(args.smoke_cells, PHASE0_SMOKE_SLUGS)
    log.info("Phase 0 smoke cells: %s", smoke_cells)
    log.info("Phase 1 main cells: %s seeds=%s (smoke_mode=%s)", main_cells, seeds, args.smoke)

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
                r_eval_path=args.r_eval_path,
                chosen_rank=rank,
                chosen_alpha=alpha,
                chosen_frac=None,  # Phase 0 hasn't picked yet; train all 6 fracs.
                smoke=False,  # Phase 0 trains at the Phase 1 composition (NOT a tiny slice).
                no_kl=args.no_kl,
                report_to=args.report_to,
                resume=args.resume,
                max_new_tokens_eval=max_new_tokens_eval,
                max_model_len_eval=max_model_len_eval,
                hf_path_suffix=args.hf_path_suffix,
                label_prefix="issue-504-phase0",
                # Round-4 fix (BLOCKER #2 / task #504): Phase 0 smoke is per-cell
                # tolerant so the picker sees every cell's outcome (legacy v1 path).
                per_cell_tolerant=True,
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
        r_eval_path=args.r_eval_path,
        chosen_rank=pick["chosen_rank"],
        chosen_alpha=pick["chosen_alpha"],
        chosen_frac=pick["chosen_checkpoint_fraction"],
        smoke=False,
        no_kl=args.no_kl,
        report_to=args.report_to,
        resume=args.resume,
        max_new_tokens_eval=max_new_tokens_eval,
        max_model_len_eval=max_model_len_eval,
        hf_path_suffix=args.hf_path_suffix,
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
        # Round-2 fix (BLOCKER #1): legacy v1 path explicitly threads v1 slugs
        # (the CLI default is v2 — that matches the live pipeline; v1 is opt-in
        # for archived-result re-analysis).
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
                "--positioned-arms",
                "v1",
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


def _run_v2_phase(
    *,
    args: argparse.Namespace,
    phase_summaries: dict[str, dict],
    arm_to_n_json: Path,
    max_new_tokens_eval: int,
    max_model_len_eval: int,
    seeds: list[int],
) -> int:
    """v2 phase router (plan v2 §4.1 + §10).

    Three phases:

      * ``phase0`` — run the 3 v2 lr-ladder smoke cells on `args.source`
        (default villain) + pick `chosen_lr` + `chosen_checkpoint_fraction`,
        writing ``phase0_calibration_v2.json``. On fallback trigger (no in-
        band cell), the artifact carries `fallback_triggered=True` and the
        caller is expected to re-invoke with `--phase phase0-fallback
        --source <easier_source>`.

      * ``phase0-fallback`` — same 3-cell lr-ladder smoke on the easier
        source persona (plan v2 §4.2). Writes ``phase0_calibration_v2_fallback.json``.

      * ``phase1`` — read ``phase0_calibration_v2.json`` (or
        ``phase0_calibration_v2_fallback.json`` when `fallback_triggered`
        is set in the v2 artifact) and train the 5 v2 main arms × 2 seeds
        at the picked lr, with per-cell trajectory eval. Writes the final
        ``epm:results v1`` sentinel.

    All three reuse the v1 Phase 0.5 + Phase 0.7 outputs (already produced
    by the caller in main() before this function runs).
    """
    lr_ladder = [float(x.strip()) for x in args.lr_ladder.split(",") if x.strip()]
    if len(lr_ladder) != 3:
        raise RuntimeError(f"--lr-ladder must be 3 comma-separated floats; got {args.lr_ladder!r}.")
    # Map lr → v2 smoke slug. The canonical ladder maps positionally to
    # the canonical slugs; for off-canonical ladders we synthesize slugs.
    canonical_lrs = list(LR_FROM_V2_SMOKE_SLUG.values())
    if lr_ladder == canonical_lrs:
        smoke_slugs = list(PHASE0_SMOKE_SLUGS_V2)
        per_cell_lrs = dict(LR_FROM_V2_SMOKE_SLUG)
    else:
        # Off-canonical ladder: synthesize new slugs.
        smoke_slugs = [f"c504v2_smoke_lr{lr:g}".replace("-", "_") for lr in lr_ladder]
        per_cell_lrs = dict(zip(smoke_slugs, lr_ladder, strict=True))
        log.warning(
            "Off-canonical lr ladder %s — synthesized slugs %s. "
            "The canonical ladder %s maps to PHASE0_SMOKE_SLUGS_V2.",
            lr_ladder,
            smoke_slugs,
            canonical_lrs,
        )

    log.info(
        "[phase=v2_%s] source=%s lr_ladder=%s slugs=%s",
        args.phase,
        args.source,
        lr_ladder,
        smoke_slugs,
    )

    if args.phase in ("phase0", "phase0-fallback"):
        return _run_v2_phase0(
            args=args,
            phase_summaries=phase_summaries,
            arm_to_n_json=arm_to_n_json,
            max_new_tokens_eval=max_new_tokens_eval,
            max_model_len_eval=max_model_len_eval,
            smoke_slugs=smoke_slugs,
            per_cell_lrs=per_cell_lrs,
        )
    if args.phase == "phase0_v3":
        return _run_v3_phase0(
            args=args,
            phase_summaries=phase_summaries,
            arm_to_n_json=arm_to_n_json,
            max_new_tokens_eval=max_new_tokens_eval,
            max_model_len_eval=max_model_len_eval,
            recovery=False,
        )
    if args.phase == "phase0_v3-recovery":
        return _run_v3_phase0(
            args=args,
            phase_summaries=phase_summaries,
            arm_to_n_json=arm_to_n_json,
            max_new_tokens_eval=max_new_tokens_eval,
            max_model_len_eval=max_model_len_eval,
            recovery=True,
        )
    if args.phase == "phase0_v4_pretrain":
        return _run_v4_phase0_pretrain(
            args=args,
            phase_summaries=phase_summaries,
            arm_to_n_json=arm_to_n_json,
            max_new_tokens_eval=max_new_tokens_eval,
            max_model_len_eval=max_model_len_eval,
        )
    if args.phase == "phase0_v4_reeval":
        return _run_v4_phase0_reeval(
            args=args,
            phase_summaries=phase_summaries,
            arm_to_n_json=arm_to_n_json,
            max_new_tokens_eval=max_new_tokens_eval,
            max_model_len_eval=max_model_len_eval,
        )
    if args.phase == "phase0_v4_bisection":
        return _run_v4_phase0_bisection(
            args=args,
            phase_summaries=phase_summaries,
            arm_to_n_json=arm_to_n_json,
            max_new_tokens_eval=max_new_tokens_eval,
            max_model_len_eval=max_model_len_eval,
        )
    if args.phase == "phase0p6_validate":
        return _run_v4_phase0p6_validate(
            args=args,
            phase_summaries=phase_summaries,
            arm_to_n_json=arm_to_n_json,
        )
    if args.phase == "phase1":
        return _run_v2_phase1(
            args=args,
            phase_summaries=phase_summaries,
            arm_to_n_json=arm_to_n_json,
            max_new_tokens_eval=max_new_tokens_eval,
            max_model_len_eval=max_model_len_eval,
            seeds=seeds,
        )
    raise RuntimeError(f"Unknown --phase {args.phase!r}")


def _run_v2_phase0(
    *,
    args: argparse.Namespace,
    phase_summaries: dict[str, dict],
    arm_to_n_json: Path,
    max_new_tokens_eval: int,
    max_model_len_eval: int,
    smoke_slugs: list[str],
    per_cell_lrs: dict[str, float],
) -> int:
    """Run the 3 v2 lr-ladder smoke cells + the v2 pick (plan v2 §4.1)."""
    # Fallback writes to phase0_calibration_v2_fallback.json; primary path
    # writes to phase0_calibration_v2.json.
    is_fallback = args.phase == "phase0-fallback"
    out_name = (
        "phase0_calibration_v2_fallback.json" if is_fallback else "phase0_calibration_v2.json"
    )
    pick_path = args.slab_root / out_name

    # Smoke cells run sequentially on 1 GPU (max_parallel=1 — each cell is
    # a Phase-1-composition train + trajectory eval at one lr). per_cell_lrs
    # threads the lr per cell through _schedule_cell_pool → i504_run_cell.py
    # --lr <X>. All cells share r=8 / α=32 (the v2 pinned rank).
    _schedule_cell_pool(
        cells=list(smoke_slugs),
        seeds=[42],  # v2 single seed for Phase 0 (plan §11)
        n_gpus=args.n_gpus,
        max_parallel=1,  # sequential — one cell per lr.
        slab_root=args.slab_root,
        runs_root=args.runs_root,
        log_dir=LOG_DIR,
        bank_path=args.bank_path,
        centroids_dir=args.centroids_dir,
        arm_to_n_json=arm_to_n_json,
        r_train_path=args.r_train_path,
        r_eval_path=args.r_eval_path,
        chosen_rank=8,  # plan v2 §10: r=8 pinned
        chosen_alpha=32,  # plan v2 §10: α=32 pinned
        chosen_frac=None,  # Phase 0 hasn't picked yet
        smoke=False,  # Phase 0 trains at Phase 1 composition (NOT a tiny slice)
        no_kl=args.no_kl,
        report_to=args.report_to,
        resume=args.resume,
        max_new_tokens_eval=max_new_tokens_eval,
        max_model_len_eval=max_model_len_eval,
        hf_path_suffix=args.hf_path_suffix,
        label_prefix=f"issue-504-v2-{args.phase}",
        per_cell_lrs=per_cell_lrs,  # v2 threading: lr per smoke cell
        # Round-2 fix (BLOCKER #2): thread the fallback source persona through
        # to EVERY smoke cell's training + eval (plan v2 §4.2). For the primary
        # phase0 (--source villain by default), this is the v1/v2 legacy
        # default. For --phase phase0-fallback --source medical_doctor, every
        # smoke cell trains + scores against medical_doctor — not villain.
        source_persona=args.source,
        # Round-4 fix (BLOCKER #2 / task #504): Phase 0 lr-ladder smoke (and
        # the fallback rerun on a different source) is per-cell tolerant so
        # the picker sees every cell's outcome — if one ladder rung crashes,
        # the remaining rungs still run and the picker decides via the
        # anti-saturation band.
        per_cell_tolerant=True,
    )

    # Run the v2 pick rule over the 3 smoke trajectories.
    _run_phase_subprocess(
        [
            "uv",
            "run",
            "python",
            "scripts/i504_phase_phase0_pick.py",
            "--mode",
            "v2",
            "--slab-root",
            str(args.slab_root),
            "--out-path",
            str(pick_path),
            "--source",
            args.source,
            "--sentinel-path",
            str(LOG_DIR / f"issue-504-v2-{args.phase}-pick-results.json"),
        ],
        f"v2_{args.phase}_pick",
    )
    if not pick_path.exists():
        raise RuntimeError(f"v2 pick expected at {pick_path}; not found.")
    pick = json.loads(pick_path.read_text())
    phase_summaries[f"v2_{args.phase}"] = {
        "verdict": pick.get("verdict"),
        "chosen_lr": pick.get("chosen_lr"),
        "chosen_checkpoint_fraction": pick.get("chosen_checkpoint_fraction"),
        "source_delta_g_at_pick_nats": pick.get("source_delta_g_at_pick_nats"),
        "source_emission_at_pick": pick.get("source_emission_at_pick"),
        "fallback_triggered": pick.get("fallback_triggered"),
        "fallback_reason": pick.get("fallback_reason"),
        "source": pick.get("source"),
    }
    _write_sentinel(
        LOG_DIR / "issue-504-results.json",
        kind="epm:results",
        phase="done",
        note_payload={
            "issue": 504,
            "status": f"v2_{args.phase}_complete",
            "v2_pick": pick,
            "phase_summaries": phase_summaries,
            "final_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
        },
    )
    log.info(
        "[phase=done] V2 %s COMPLETE — verdict=%s, chosen_lr=%s, fallback=%s. %s",
        args.phase,
        pick.get("verdict"),
        pick.get("chosen_lr"),
        pick.get("fallback_triggered"),
        datetime.now(UTC).isoformat(),
    )
    return 0 if pick.get("verdict") == "pass" else 2


def _run_v3_phase0(
    *,
    args: argparse.Namespace,
    phase_summaries: dict[str, dict],
    arm_to_n_json: Path,
    max_new_tokens_eval: int,
    max_model_len_eval: int,
    recovery: bool = False,
) -> int:
    """v3 EPOCHS-ladder smoke + pick (plan v3 §4.1).

    Two modes:
      * `recovery=False` (default): runs the 2 v3 smoke cells from
        `--epochs-ladder` (default {2, 3}) at fixed `--fixed-lr` (default 1e-4),
        seed=42, then invokes the v3 picker. Writes
        `phase0_calibration_v3.json` + (on Trigger A/C) `phase0_v3_exit_to_v4.json`.

      * `recovery=True`: in-plan finer-fraction recovery (plan v3 §4.1 trigger B
        + §4.2). Re-runs the EPOCHS=2 cell at finer fractions {0.02, 0.04,
        0.06, 0.08} via the same dispatcher path; the trainer's checkpoint
        cadence is overridden via a separate suffix-decorated runs subdir so
        the coarse-fraction trajectories survive. After the recovery cell
        finishes, re-invokes the picker over the augmented EPOCHS=2 trajectory.
    """
    epochs_ladder = [int(x.strip()) for x in args.epochs_ladder.split(",") if x.strip()]
    if not epochs_ladder:
        raise RuntimeError(
            f"--epochs-ladder must be a non-empty CSV of integers; got {args.epochs_ladder!r}."
        )
    # Map epochs → v3 smoke slug. The canonical ladder maps positionally; for
    # off-canonical ladders synthesize slugs (mirrors v2 off-canonical handling).
    canonical_eps = list(EPOCHS_FROM_V3_SMOKE_SLUG.values())
    if epochs_ladder == canonical_eps:
        smoke_slugs = list(PHASE0_SMOKE_SLUGS_V3)
        per_cell_epochs = dict(EPOCHS_FROM_V3_SMOKE_SLUG)
    else:
        smoke_slugs = [f"c504v3_smoke_eps{e}" for e in epochs_ladder]
        per_cell_epochs = dict(zip(smoke_slugs, epochs_ladder, strict=True))
        log.warning(
            "Off-canonical epochs ladder %s — synthesized slugs %s. "
            "The canonical ladder %s maps to PHASE0_SMOKE_SLUGS_V3.",
            epochs_ladder,
            smoke_slugs,
            canonical_eps,
        )
    fixed_lr = float(args.fixed_lr)
    # In-plan recovery overrides the smoke set to EPOCHS=2 ONLY, re-runs at
    # finer fractions; the picker reads the SAME canonical slug path so the
    # un-decorated trajectory survives via a different sentinel/suffix path.
    if recovery:
        smoke_slugs = ["c504v3_smoke_eps2"]
        per_cell_epochs = {"c504v3_smoke_eps2": 2}
        log.info(
            "[phase=phase0_v3-recovery] in-plan finer-fraction recovery on "
            "EPOCHS=2 at fractions=%s (plan v3 §4.1 trigger B + §4.2). The "
            "recovery cell re-runs over the coarse-grained trajectory; the "
            "v3 picker re-reads the trajectory.",
            CHECKPOINT_FRACTIONS_V3_FINER,
        )

    log.info(
        "[phase=v3_%s] source=%s fixed_lr=%g epochs_ladder=%s slugs=%s",
        args.phase,
        args.source,
        fixed_lr,
        epochs_ladder,
        smoke_slugs,
    )

    pick_path = args.slab_root / "phase0_calibration_v3.json"
    exit_to_v4_path = args.slab_root / "phase0_v3_exit_to_v4.json"
    label_prefix = (
        f"issue-504-v3-{args.phase}" if not recovery else "issue-504-v3-phase0_v3-recovery"
    )
    # v3 in-plan recovery (plan §4.1 trigger B + §4.2): on the recovery phase,
    # thread CHECKPOINT_FRACTIONS_V3_FINER + a trajectory_suffix so the cell
    # retrains EPOCHS=2 at the finer fractions AND its trajectory.json lands
    # in a DIFFERENT slab-root subdir (the coarse trajectory must survive so
    # the picker's merge step can read both). The recovery uses --resume=False
    # so the prior coarse trajectory doesn't short-circuit the retrain.
    recovery_traj_suffix = "__recovery_finer"
    cp_fractions_arg = CHECKPOINT_FRACTIONS_V3_FINER if recovery else None
    traj_suffix_arg = recovery_traj_suffix if recovery else ""
    recovery_resume = False if recovery else args.resume

    # Smoke cells run sequentially on 1 GPU (max_parallel=1 — each cell is a
    # Phase-1-composition train + trajectory eval at one EPOCHS value). All
    # cells share r=8 / α=32 / lr=fixed_lr; per_cell_epochs threads the
    # EPOCHS per cell through _schedule_cell_pool → i504_run_cell.py --epochs.
    _schedule_cell_pool(
        cells=list(smoke_slugs),
        seeds=[42],  # v3 single seed for Phase 0 (plan §11)
        n_gpus=args.n_gpus,
        max_parallel=1,  # sequential — one cell per epochs value.
        slab_root=args.slab_root,
        runs_root=args.runs_root,
        log_dir=LOG_DIR,
        bank_path=args.bank_path,
        centroids_dir=args.centroids_dir,
        arm_to_n_json=arm_to_n_json,
        r_train_path=args.r_train_path,
        r_eval_path=args.r_eval_path,
        chosen_rank=8,  # plan v3 §10: r=8 pinned
        chosen_alpha=32,  # plan v3 §10: α=32 pinned
        chosen_frac=None,  # Phase 0 hasn't picked yet
        smoke=False,  # Phase 0 trains at Phase 1 composition (NOT a tiny slice)
        no_kl=args.no_kl,
        report_to=args.report_to,
        resume=recovery_resume,
        max_new_tokens_eval=max_new_tokens_eval,
        max_model_len_eval=max_model_len_eval,
        hf_path_suffix=args.hf_path_suffix,
        label_prefix=label_prefix,
        chosen_lr=fixed_lr,  # v3: fixed lr applied to every smoke cell
        per_cell_epochs=per_cell_epochs,  # v3 threading: EPOCHS per smoke cell
        source_persona=args.source,
        # Round-4 fix (BLOCKER #2 / task #504): Phase 0 v3 EPOCHS-ladder smoke
        # is per-cell tolerant so the picker sees every cell's outcome — if
        # one rung crashes, the remaining rungs still run and the picker
        # decides via the anti-saturation band.
        per_cell_tolerant=True,
        # Round-8 (task #504, deferred concern `phase0-v3-finer-grid-recovery-
        # not-wired`): thread the finer-fraction cadence + the trajectory
        # suffix to the recovery cell. Both default to no-op on the primary
        # phase0_v3 path.
        checkpoint_fractions=cp_fractions_arg,
        trajectory_suffix=traj_suffix_arg,
    )

    # Run the v3 pick rule over the smoke trajectories. Picker writes the
    # exit-to-v4 artifact itself on Trigger A/C. On the recovery phase, the
    # picker is invoked with --include-finer-recovery + --recovery-traj-suffix
    # so it MERGES the coarse + finer trajectories and re-applies the pick rule
    # over the augmented (epochs, frac) table.
    pick_cmd = [
        "uv",
        "run",
        "python",
        "scripts/i504_phase_phase0_pick.py",
        "--mode",
        "v3",
        "--slab-root",
        str(args.slab_root),
        "--out-path",
        str(pick_path),
        "--exit-to-v4-path",
        str(exit_to_v4_path),
        "--source",
        args.source,
        "--fixed-lr",
        repr(fixed_lr),
        "--sentinel-path",
        str(LOG_DIR / f"issue-504-v3-{args.phase}-pick-results.json"),
    ]
    if recovery:
        pick_cmd.extend(
            [
                "--include-finer-recovery",
                "--recovery-traj-suffix",
                recovery_traj_suffix,
            ]
        )
    pick_rc = 0
    try:
        _run_phase_subprocess(pick_cmd, f"v3_{args.phase}_pick")
    except subprocess.CalledProcessError as e:
        # Picker returns rc=2 on non-pass verdict (Trigger A/B/C). The artifact
        # is still written; the dispatcher reads it below to decide routing.
        pick_rc = e.returncode
        log.warning(
            "[phase=v3_%s_pick] picker exited rc=%d (non-pass verdict); "
            "reading artifact at %s to determine routing.",
            args.phase,
            pick_rc,
            pick_path,
        )

    if not pick_path.exists():
        raise RuntimeError(f"v3 pick expected at {pick_path}; not found.")
    pick = json.loads(pick_path.read_text())
    phase_summaries[f"v3_{args.phase}"] = {
        "verdict": pick.get("verdict"),
        "chosen_epochs": pick.get("chosen_epochs"),
        "chosen_lr": pick.get("chosen_lr"),
        "chosen_checkpoint_fraction": pick.get("chosen_checkpoint_fraction"),
        "chosen_checkpoint_steps": pick.get("chosen_checkpoint_steps"),
        "source_delta_g_at_pick_nats": pick.get("source_delta_g_at_pick_nats"),
        "source_emission_at_pick": pick.get("source_emission_at_pick"),
        "fallback_triggered": pick.get("fallback_triggered"),
        "fallback_reason": pick.get("fallback_reason"),
        "in_plan_recovery_triggered": pick.get("in_plan_recovery_triggered"),
        "source": pick.get("source"),
    }
    _write_sentinel(
        LOG_DIR / "issue-504-results.json",
        kind="epm:results",
        phase="done",
        note_payload={
            "issue": 504,
            "status": f"v3_{args.phase}_complete",
            "v3_pick": pick,
            "phase_summaries": phase_summaries,
            "final_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
        },
    )
    log.info(
        "[phase=done] V3 %s COMPLETE — verdict=%s, chosen_epochs=%s, chosen_lr=%s, "
        "fallback=%s, in_plan_recovery=%s. %s",
        args.phase,
        pick.get("verdict"),
        pick.get("chosen_epochs"),
        pick.get("chosen_lr"),
        pick.get("fallback_triggered"),
        pick.get("in_plan_recovery_triggered"),
        datetime.now(UTC).isoformat(),
    )
    return 0 if pick.get("verdict") == "pass" else 2


def _select_active_phase0_pick(
    primary_pick_path: Path,
    fallback_pick_path: Path,
    v3_pick_path: Path | None = None,
    v4_pick_path: Path | None = None,
    v4_bisection_pick_path: Path | None = None,
) -> tuple[dict, Path]:
    """Choose the active Phase 0 pick artifact.

    Precedence (highest → lowest):

      1. v4 primary (`phase0_calibration_v4.json`) — plan v5 §4.1 bystander-
         resolution picker on the EPOCHS=3 anchor.
      2. v4 bisection (`phase0_calibration_v4_bisection.json`) — plan v5 §4.2
         step 1 EPOCHS=2 finer-grid fallback (only if v4 primary failed AND
         the bisection passed).
      3. v3 (`phase0_calibration_v3.json`) — plan v3 EPOCHS-ladder pick.
      4. v2 fallback (`phase0_calibration_v2_fallback.json`) — only if v2
         primary fired its fallback flag.
      5. v2 primary (`phase0_calibration_v2.json`).

    Pure helper (read-only filesystem access). Returns the parsed dict + the
    absolute path of the chosen artifact. Used by `_run_v2_phase1` for both
    cell-training selection AND the analyze subprocess so they read the SAME
    artifact (round-3 fix, concern_id `fallback-analyze-pick-path`; v5 round-2
    fix BLOCKER A — Phase 1 must consume v4 picker output before falling
    back to v3/v2).

    A non-pass v4 verdict falls through to v4-bisection → v3 → v2 selection,
    in that order. Same for v3 (existing behavior preserved).

    Raises FileNotFoundError if NO artifact is available. v4 OR v3 with
    non-pass verdict (Trigger A/B/C) falls through; if no fallback artifact
    exists either, the caller surfaces the failure mode.
    """
    if v4_pick_path is not None and v4_pick_path.exists():
        v4_pick = json.loads(v4_pick_path.read_text())
        if v4_pick.get("verdict") == "pass":
            return v4_pick, v4_pick_path
        # Non-pass v4 primary falls through to v4-bisection (then v3, then v2).
    if v4_bisection_pick_path is not None and v4_bisection_pick_path.exists():
        v4b_pick = json.loads(v4_bisection_pick_path.read_text())
        if v4b_pick.get("verdict") == "pass":
            return v4b_pick, v4_bisection_pick_path
    if v3_pick_path is not None and v3_pick_path.exists():
        v3_pick = json.loads(v3_pick_path.read_text())
        if v3_pick.get("verdict") == "pass":
            return v3_pick, v3_pick_path
        # Non-pass v3 verdict falls through to v2 selection (or fails below
        # if v2 isn't available either).
    if not primary_pick_path.exists():
        missing_paths = [primary_pick_path]
        if v3_pick_path is not None:
            missing_paths.append(v3_pick_path)
        if v4_pick_path is not None:
            missing_paths.append(v4_pick_path)
        if v4_bisection_pick_path is not None:
            missing_paths.append(v4_bisection_pick_path)
        raise FileNotFoundError(
            f"Phase 1 requires {primary_pick_path}"
            + (f" or one of {missing_paths[1:]}" if len(missing_paths) > 1 else "")
            + "; run --phase phase0 (v2) / --phase phase0_v3 / --phase "
            "phase0_v4_reeval / --phase phase0_v4_bisection first."
        )
    primary_pick = json.loads(primary_pick_path.read_text())
    if primary_pick.get("fallback_triggered") and fallback_pick_path.exists():
        fallback_pick = json.loads(fallback_pick_path.read_text())
        return fallback_pick, fallback_pick_path
    return primary_pick, primary_pick_path


def _run_v4_phase0_pretrain(
    *,
    args: argparse.Namespace,
    phase_summaries: dict[str, dict],
    arm_to_n_json: Path,
    max_new_tokens_eval: int,
    max_model_len_eval: int,
) -> int:
    """v4 Phase 0 §4.0 — pre-train EPOCHS=3 anchor with per-fraction HF persistence.

    Plan v5 §4.0: re-train the EPOCHS=3 seed=42 anchor cell ONCE (~0.4 GPU-h
    on 1× H100) with `EPM_PERSIST_TRAJECTORY_HF_REPO` +
    `EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER` set so each of the 6 fraction
    checkpoints {0.08, 0.16, 0.33, 0.50, 0.75, 1.00} is uploaded inline to
    `adapters/issue_504_v4/c504v4_smoke_eps3_seed42/ckpt_frac{N}/` AND
    verified via `huggingface_hub.list_repo_files` before the next
    fraction is saved.

    This step is mandatory because v3's training upload persisted ONLY the
    FINAL adapter, not the 6 trajectory checkpoints that §4.1 needs.
    """
    cell_slug = "c504v4_smoke_eps3_seed42"
    pretrain_subfolder = f"adapters/issue_504_v4/{cell_slug}"
    out_path = args.slab_root / "phase0_trajectory_v4.json"

    log.info(
        "[phase=v4_phase0_pretrain] re-training EPOCHS=3 anchor (cell=%s) with "
        "trajectory persistence → %s",
        cell_slug,
        pretrain_subfolder,
    )

    # Set the per-fraction HF persistence env vars so
    # CheckpointAtFractionsCallback uploads each fraction inline. See
    # train_cell.py::_maybe_persist_trajectory_checkpoint.
    os.environ["EPM_PERSIST_TRAJECTORY_HF_REPO"] = HF_MODEL_REPO
    os.environ["EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER"] = pretrain_subfolder

    # Train one cell via the existing pool scheduler. EPOCHS=3, lr=1e-4,
    # r=8/α=32 (matches v3 EPOCHS=3 byte-for-byte except for persistence).
    # Slug is "c504v3_smoke_eps3" — the canonical EPOCHS=3 smoke slug — so
    # build_cell_504 + i504_eval_trajectory.py disjointness guards already
    # recognize it.
    _schedule_cell_pool(
        cells=["c504v3_smoke_eps3"],
        seeds=[42],
        n_gpus=args.n_gpus,
        max_parallel=1,
        slab_root=args.slab_root,
        runs_root=args.runs_root,
        log_dir=LOG_DIR,
        bank_path=args.bank_path,
        centroids_dir=args.centroids_dir,
        arm_to_n_json=arm_to_n_json,
        r_train_path=args.r_train_path,
        r_eval_path=args.r_eval_path,
        chosen_rank=8,
        chosen_alpha=32,
        chosen_frac=None,
        smoke=False,
        no_kl=args.no_kl,
        report_to=args.report_to,
        resume=args.resume,
        max_new_tokens_eval=max_new_tokens_eval,
        max_model_len_eval=max_model_len_eval,
        hf_path_suffix=args.hf_path_suffix,
        label_prefix="issue-504-v4-phase0_pretrain",
        chosen_lr=float(args.fixed_lr),
        per_cell_epochs={"c504v3_smoke_eps3": 3},
        source_persona=args.source,
        per_cell_tolerant=False,  # pretrain is single-cell; failure is fatal
    )

    # Verify the 6 checkpoints landed on HF (fail-loud per upload-policy).
    from huggingface_hub import list_repo_files

    expected_fractions = ["0.08", "0.16", "0.33", "0.50", "0.75", "1.00"]
    try:
        files = list_repo_files(HF_MODEL_REPO, token=os.environ.get("HF_TOKEN"))
    except Exception as exc:
        raise RuntimeError(
            f"[phase=v4_phase0_pretrain] post-train Hub verify FAILED: "
            f"list_repo_files({HF_MODEL_REPO!r}) raised {exc}. Cannot confirm "
            f"the 6 trajectory checkpoints landed."
        ) from exc
    missing: list[str] = []
    uploaded_paths: list[str] = []
    for frac in expected_fractions:
        key = f"{pretrain_subfolder}/ckpt_frac{frac}/adapter_model.safetensors"
        if key not in files:
            missing.append(key)
        else:
            uploaded_paths.append(f"{pretrain_subfolder}/ckpt_frac{frac}")
    if missing:
        raise RuntimeError(
            f"[phase=v4_phase0_pretrain] Hub verify FAILED: {len(missing)} of "
            f"{len(expected_fractions)} fraction checkpoints missing. "
            f"missing={missing}. The train_cell callback's "
            f"_maybe_persist_trajectory_checkpoint must have raised at the "
            f"first failure — investigate the training log."
        )

    payload = {
        "version": "4.0_pretrain",
        "epochs": 3,
        "seed": 42,
        "cell_slug": cell_slug,
        "hf_repo": HF_MODEL_REPO,
        "checkpoints_uploaded": uploaded_paths,
        "verify_hub_api": True,
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    phase_summaries["v4_phase0_pretrain"] = {
        "n_checkpoints_uploaded": len(uploaded_paths),
        "checkpoints_uploaded": uploaded_paths,
    }
    _write_sentinel(
        LOG_DIR / "issue-504-v4-phase0_pretrain-results.json",
        kind="epm:progress",
        phase="v4_phase0_pretrain_done",
        note_payload={
            "issue": 504,
            "phase": "v4_phase0_pretrain",
            "checkpoints_uploaded": uploaded_paths,
            "out_path": str(out_path),
            "phase_summaries": phase_summaries,
        },
    )
    log.info(
        "[phase=done] V4 phase0_pretrain COMPLETE — %d/%d checkpoints uploaded + verified. %s",
        len(uploaded_paths),
        len(expected_fractions),
        datetime.now(UTC).isoformat(),
    )
    return 0


def _run_v4_phase0_reeval(
    *,
    args: argparse.Namespace,
    phase_summaries: dict[str, dict],
    arm_to_n_json: Path,
    max_new_tokens_eval: int,
    max_model_len_eval: int,
) -> int:
    """v4 Phase 0 §4.1 — re-evaluate EPOCHS=3 anchor through the fixed reader.

    Plan v5 §4.1: read the v4-pretrained EPOCHS=3 trajectory checkpoints
    (produced by --phase phase0_v4_pretrain) through the SAME fixed
    `i504_eval_trajectory.py` reader Phase 1 uses, then run the v4
    bystander-resolution picker (plan v5 fix #2). On pass, writes
    phase0_calibration_v4.json with the chosen frac + bystander_resolution_at_pick.

    This phase IS the v4 single-cell re-eval — the architectural unification
    (smoke == sweep with one cell) of plan v5 §4.9. Same dispatcher path,
    same subprocess shape, same env injection, same logging surface as the
    Phase 1 sweep.
    """
    cell_slug = "c504v4_smoke_eps3_reread"
    pretrain_subfolder = "adapters/issue_504_v4/c504v4_smoke_eps3_seed42"
    out_pick_path = args.slab_root / "phase0_calibration_v4.json"

    log.info(
        "[phase=v4_phase0_reeval] re-evaluating EPOCHS=3 anchor through the "
        "fixed reader (HF subfolder=%s)",
        pretrain_subfolder,
    )

    # Build the checkpoint_index.json from the HF subfolder structure. The
    # eval rig consumes `--checkpoint-index <path>` pointing at a JSON of
    # {frac_str: {step: int, path: str}}.
    fractions = [0.08, 0.16, 0.33, 0.50, 0.75, 1.00]
    adapter_local_root = args.runs_root / "v4_anchor_download"
    adapter_local_root.mkdir(parents=True, exist_ok=True)
    # Use hf_hub_download per file: snapshot_download + allow_patterns silently
    # matches zero files in hf_hub 0.36.2 for nested subfolder globs (verified
    # 2026-06-08 on pod-504; the 6 ckpts ARE on HF, the glob just doesn't
    # match). Explicit per-file download is the reliable shape.
    from huggingface_hub import hf_hub_download

    ckpt_index: dict[str, dict] = {}
    for frac in fractions:
        frac_token = "1.00" if abs(frac - 1.0) < 1e-6 else f"{frac:.2f}"
        subfolder = f"{pretrain_subfolder}/ckpt_frac{frac_token}"
        for fname in ("adapter_config.json", "adapter_model.safetensors"):
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                filename=f"{subfolder}/{fname}",
                local_dir=str(adapter_local_root),
                token=os.environ.get("HF_TOKEN"),
            )
        local_path = adapter_local_root / subfolder
        if not (local_path / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"[phase=v4_phase0_reeval] missing local adapter at {local_path} "
                f"after per-file hf_hub_download. Verify --phase phase0_v4_pretrain "
                f"completed and the 6 checkpoints landed on HF."
            )
        ckpt_index[f"{frac:.2f}"] = {
            "step": None,
            "path": str(local_path),
        }
    run_dir = args.runs_root / f"{cell_slug}_seed42"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_index_path = run_dir / "checkpoint_index.json"
    ckpt_index_path.write_text(json.dumps(ckpt_index, indent=2))

    # Build the trajectory by calling i504_eval_trajectory.py through the same
    # subprocess shape Phase 1 cells use (plan v5 §4.9 unification).
    out_traj = args.slab_root / f"{cell_slug}_seed42" / "trajectory.json"
    out_traj.parent.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        "uv",
        "run",
        "python",
        "scripts/i504_eval_trajectory.py",
        "--cell",
        cell_slug,
        "--seed",
        "42",
        "--checkpoint-index",
        str(ckpt_index_path),
        "--out-path",
        str(out_traj),
        "--bank-path",
        str(args.bank_path),
        "--r-eval-path",
        str(args.r_eval_path),
        "--panel-json",
        str(arm_to_n_json),
        "--max-lora-rank",
        "8",
        "--max-new-tokens",
        str(max_new_tokens_eval),
        "--max-model-len",
        str(max_model_len_eval),
        "--source",
        args.source,
    ]
    if args.no_kl:
        eval_cmd.append("--no-kl")
    _run_phase_subprocess(eval_cmd, "v4_phase0_reeval_traj")

    if not out_traj.exists():
        raise RuntimeError(
            f"[phase=v4_phase0_reeval] eval_trajectory exited 0 but "
            f"{out_traj} missing — silent eval failure."
        )

    # Run the v4 bystander-resolution picker on the produced trajectory.
    pick_cmd = [
        "uv",
        "run",
        "python",
        "scripts/i504_phase_phase0_pick.py",
        "--mode",
        "v4",
        "--slab-root",
        str(args.slab_root),
        "--out-path",
        str(out_pick_path),
        "--v4-trajectory-path",
        str(out_traj),
        "--source",
        args.source,
        "--fixed-lr",
        repr(float(args.fixed_lr)),
        "--sentinel-path",
        str(LOG_DIR / "issue-504-v4-phase0_reeval-pick-results.json"),
    ]
    pick_rc = 0
    try:
        _run_phase_subprocess(pick_cmd, "v4_phase0_reeval_pick")
    except subprocess.CalledProcessError as e:
        pick_rc = e.returncode
        log.warning(
            "[phase=v4_phase0_reeval_pick] picker exited rc=%d (non-pass "
            "verdict); reading artifact to determine routing.",
            pick_rc,
        )

    if not out_pick_path.exists():
        raise RuntimeError(f"[phase=v4_phase0_reeval] pick artifact missing at {out_pick_path}.")
    pick = json.loads(out_pick_path.read_text())
    phase_summaries["v4_phase0_reeval"] = {
        "verdict": pick.get("verdict"),
        "chosen_epochs": pick.get("chosen_epochs"),
        "chosen_lr": pick.get("chosen_lr"),
        "chosen_checkpoint_fraction": pick.get("chosen_checkpoint_fraction"),
        "bystander_resolution_at_pick": pick.get("bystander_resolution_at_pick"),
        "fallback_triggered": pick.get("fallback_triggered"),
        "fallback_reason": pick.get("fallback_reason"),
    }
    _write_sentinel(
        LOG_DIR / "issue-504-v4-phase0_reeval-results.json",
        kind="epm:progress",
        phase="v4_phase0_reeval_done",
        note_payload={
            "issue": 504,
            "phase": "v4_phase0_reeval",
            "v4_pick": pick,
            "phase_summaries": phase_summaries,
        },
    )
    log.info(
        "[phase=done] V4 phase0_reeval COMPLETE — verdict=%s, chosen_frac=%s, "
        "bystander_resolution_at_pick=%s, fallback=%s. %s",
        pick.get("verdict"),
        pick.get("chosen_checkpoint_fraction"),
        pick.get("bystander_resolution_at_pick"),
        pick.get("fallback_triggered"),
        datetime.now(UTC).isoformat(),
    )
    return 0 if pick.get("verdict") == "pass" else 2


def _run_v4_phase0_bisection(  # noqa: C901 -- linear ladder: train → verify → re-eval → pick
    *,
    args: argparse.Namespace,
    phase_summaries: dict[str, dict],
    arm_to_n_json: Path,
    max_new_tokens_eval: int,
    max_model_len_eval: int,
) -> int:
    """v4 Phase 0 §4.2 step 1 — EPOCHS=2 finer-fraction bisection fallback.

    Plan v5 §4.2 step 1: triggered ONLY when ``phase0_v4_reeval`` returns
    ``verdict='no_in_band_anchor'`` (every EPOCHS=3 fraction is either
    pinned at the marker-argmax ceiling or below the +0.5 nats floor —
    the EPOCHS=3 anchor's bystander layer has no dynamic range). Bisects
    to EPOCHS=2 at the finer-fraction grid {0.04, 0.08, 0.12, 0.16}
    (CHECKPOINT_FRACTIONS_V4_BISECTION) and re-applies the bystander-
    resolution gate.

    Pipeline:

      1. Pre-flight assertion: phase0_calibration_v4.json must exist with
         ``verdict != 'pass'`` (otherwise the bisection wouldn't be
         needed). Reading the artifact prevents wasting ~0.6 GPU-h on a
         redundant re-train.
      2. Set ``EPM_PERSIST_TRAJECTORY_HF_REPO`` + ``_SUBFOLDER`` to the
         v4-bisection subfolder so each finer-fraction checkpoint is
         uploaded inline AND verified via the Hub API before the next
         fraction is saved (fail-loud per upload-policy).
      3. Train EPOCHS=2 on ``c504v3_smoke_eps2`` at the finer grid via
         the existing pool scheduler (which threads
         ``checkpoint_fractions`` + ``trajectory_suffix`` through
         ``i504_run_cell.py``).
      4. Verify all 4 checkpoints landed on HF (parity with
         ``_run_v4_phase0_pretrain``'s 6-of-6 verify).
      5. Re-eval through the same fixed reader the Phase 1 eval will
         use, building a fresh checkpoint_index over the 4 bisection
         fractions.
      6. Re-run the v4 bystander-resolution picker on the bisection
         trajectory, writing
         ``eval_results/issue_504/phase0_calibration_v4_bisection.json``.
         The artifact schema is identical to phase0_calibration_v4.json,
         so the dispatcher's ``_select_active_phase0_pick`` (round-2
         BLOCKER A) picks it up automatically when v4-primary failed.

    Returns:
      * rc=0 on bisection pass: ``phase0_calibration_v4_bisection.json``
        has ``verdict='pass'`` AND
        ``bystander_resolution_at_pick >= 0.20``. Phase 1 can proceed.
      * rc=2 on bisection fail: every finer fraction also failed the
        bystander-resolution gate. The dispatcher should then exit to
        plan v5's rank bump (§4.2 step 2) via
        ``epm:failure v1 failure_class: methodology,
        reason: bystander_resolution_unreachable_at_r8_epochs23``.
    """
    # Step 1: pre-flight assertion.
    primary_pick_path = args.slab_root / "phase0_calibration_v4.json"
    if not primary_pick_path.exists():
        raise RuntimeError(
            f"[phase=phase0_v4_bisection] phase0_calibration_v4.json missing at "
            f"{primary_pick_path}. The bisection fallback is triggered ONLY by a "
            f"non-pass verdict from `--phase phase0_v4_reeval`; run that phase "
            f"first."
        )
    primary_pick = json.loads(primary_pick_path.read_text())
    primary_verdict = primary_pick.get("verdict")
    if primary_verdict == "pass":
        log.warning(
            "[phase=phase0_v4_bisection] phase0_calibration_v4.json has "
            "verdict='pass' — bisection NOT needed. Refusing to waste ~0.6 GPU-h "
            "on a redundant re-train. If you really want to re-run the bisection, "
            "delete phase0_calibration_v4.json first."
        )
        return 0
    # Round-3 (Codex concern): tighten the trigger gate. Plan v5 §4.1 step 6
    # routes to EPOCHS=2 bisection ONLY on `verdict='no_in_band_anchor'`. Any
    # OTHER non-pass verdict (e.g. a future picker addition that returns a
    # different fallback) must NOT silently consume the bisection budget — it
    # should surface as an epm:failure so a human (or the orchestrator's
    # halt-criterion) decides the routing.
    if primary_verdict != "no_in_band_anchor":
        raise RuntimeError(
            f"[phase=phase0_v4_bisection] phase0_calibration_v4.json has "
            f"verdict={primary_verdict!r} (fallback_reason="
            f"{primary_pick.get('fallback_reason')!r}). The bisection fallback "
            f"is gated to verdict='no_in_band_anchor' per plan v5 §4.1 step 6 "
            f"— refusing to spend ~0.6 GPU-h on an unrecognized non-pass "
            f"verdict. failure_class=methodology, "
            f"reason=v4_bisection_unexpected_verdict_{primary_verdict}"
        )

    cell_slug = "c504v3_smoke_eps2"  # v3 EPOCHS=2 smoke slug (existing infrastructure)
    bisection_subfolder = "adapters/issue_504_v4_bisection/c504v4_bisection_eps2_seed42"
    bisection_traj_suffix = "__v4_bisection"
    out_pick_path = args.slab_root / "phase0_calibration_v4_bisection.json"

    log.info(
        "[phase=phase0_v4_bisection] phase0_v4 verdict=%s (fallback_reason=%s) — "
        "starting EPOCHS=2 finer-fraction bisection at fractions=%s",
        primary_pick.get("verdict"),
        primary_pick.get("fallback_reason"),
        CHECKPOINT_FRACTIONS_V4_BISECTION,
    )

    # Step 2: set the per-fraction HF persistence env vars so
    # CheckpointAtFractionsCallback uploads each fraction inline.
    os.environ["EPM_PERSIST_TRAJECTORY_HF_REPO"] = HF_MODEL_REPO
    os.environ["EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER"] = bisection_subfolder

    # Step 3: train one cell via the existing pool scheduler. EPOCHS=2,
    # lr=1e-4, r=8/α=32. We pass ``checkpoint_fractions`` + ``trajectory_suffix``
    # the same way the v3 in-plan recovery path does (already-verified
    # threading; round-8 wire-up).
    _schedule_cell_pool(
        cells=[cell_slug],
        seeds=[42],
        n_gpus=args.n_gpus,
        max_parallel=1,
        slab_root=args.slab_root,
        runs_root=args.runs_root,
        log_dir=LOG_DIR,
        bank_path=args.bank_path,
        centroids_dir=args.centroids_dir,
        arm_to_n_json=arm_to_n_json,
        r_train_path=args.r_train_path,
        r_eval_path=args.r_eval_path,
        chosen_rank=8,
        chosen_alpha=32,
        chosen_frac=None,
        smoke=False,
        no_kl=args.no_kl,
        report_to=args.report_to,
        resume=False,  # bisection re-trains from scratch (different recipe).
        max_new_tokens_eval=max_new_tokens_eval,
        max_model_len_eval=max_model_len_eval,
        hf_path_suffix=args.hf_path_suffix,
        label_prefix="issue-504-v4-phase0_bisection",
        chosen_lr=float(args.fixed_lr),
        per_cell_epochs={cell_slug: 2},
        source_persona=args.source,
        per_cell_tolerant=False,
        checkpoint_fractions=CHECKPOINT_FRACTIONS_V4_BISECTION,
        trajectory_suffix=bisection_traj_suffix,
    )

    # Step 4: verify the 4 checkpoints landed on HF (parity with
    # _run_v4_phase0_pretrain).
    from huggingface_hub import list_repo_files

    expected_fractions = [f"{f:.2f}" for f in CHECKPOINT_FRACTIONS_V4_BISECTION]
    try:
        files = list_repo_files(HF_MODEL_REPO, token=os.environ.get("HF_TOKEN"))
    except Exception as exc:
        raise RuntimeError(
            f"[phase=phase0_v4_bisection] post-train Hub verify FAILED: "
            f"list_repo_files({HF_MODEL_REPO!r}) raised {exc}. Cannot confirm "
            f"the {len(expected_fractions)} bisection checkpoints landed."
        ) from exc
    missing: list[str] = []
    uploaded_paths: list[str] = []
    for frac in expected_fractions:
        key = f"{bisection_subfolder}/ckpt_frac{frac}/adapter_model.safetensors"
        if key not in files:
            missing.append(key)
        else:
            uploaded_paths.append(f"{bisection_subfolder}/ckpt_frac{frac}")
    if missing:
        raise RuntimeError(
            f"[phase=phase0_v4_bisection] Hub verify FAILED: {len(missing)} of "
            f"{len(expected_fractions)} bisection checkpoints missing. "
            f"missing={missing}. The train_cell callback's "
            f"_maybe_persist_trajectory_checkpoint must have raised at the "
            f"first failure — investigate the training log."
        )

    # Step 5: re-eval. Build a fresh checkpoint_index over the 4 bisection
    # fractions + drive `i504_eval_trajectory.py` against the v4-bisection
    # HF subfolder.
    fractions = list(CHECKPOINT_FRACTIONS_V4_BISECTION)
    adapter_local_root = args.runs_root / "v4_bisection_anchor_download"
    adapter_local_root.mkdir(parents=True, exist_ok=True)
    # Per-file hf_hub_download (snapshot_download glob is unreliable in
    # hf_hub 0.36.2 — see phase0_v4_reeval for the same fix).
    from huggingface_hub import hf_hub_download

    ckpt_index: dict[str, dict] = {}
    for frac in fractions:
        frac_token = f"{frac:.2f}"
        subfolder = f"{bisection_subfolder}/ckpt_frac{frac_token}"
        for fname in ("adapter_config.json", "adapter_model.safetensors"):
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                filename=f"{subfolder}/{fname}",
                local_dir=str(adapter_local_root),
                token=os.environ.get("HF_TOKEN"),
            )
        local_path = adapter_local_root / subfolder
        if not (local_path / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"[phase=phase0_v4_bisection] missing local adapter at {local_path} "
                f"after per-file hf_hub_download. Verify the inline training upload "
                f"actually landed on HF."
            )
        ckpt_index[f"{frac:.2f}"] = {"step": None, "path": str(local_path)}

    reread_run_dir = args.runs_root / "c504v4_bisection_eps2_reread_seed42"
    reread_run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_index_path = reread_run_dir / "checkpoint_index.json"
    ckpt_index_path.write_text(json.dumps(ckpt_index, indent=2))

    out_traj = args.slab_root / "c504v4_bisection_eps2_reread_seed42" / "trajectory.json"
    out_traj.parent.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        "uv",
        "run",
        "python",
        "scripts/i504_eval_trajectory.py",
        "--cell",
        "c504v4_bisection_eps2_reread",
        "--seed",
        "42",
        "--checkpoint-index",
        str(ckpt_index_path),
        "--out-path",
        str(out_traj),
        "--bank-path",
        str(args.bank_path),
        "--r-eval-path",
        str(args.r_eval_path),
        "--panel-json",
        str(arm_to_n_json),
        "--max-lora-rank",
        "8",
        "--max-new-tokens",
        str(max_new_tokens_eval),
        "--max-model-len",
        str(max_model_len_eval),
        "--source",
        args.source,
    ]
    if args.no_kl:
        eval_cmd.append("--no-kl")
    _run_phase_subprocess(eval_cmd, "v4_phase0_bisection_traj")

    if not out_traj.exists():
        raise RuntimeError(
            f"[phase=phase0_v4_bisection] eval_trajectory exited 0 but "
            f"{out_traj} missing — silent eval failure."
        )

    # Step 6: re-run the v4 bystander-resolution picker.
    # i504_phase_phase0_pick.py's `--mode v4` reads any `--v4-trajectory-path`
    # the caller provides, applies the bystander-resolution gate, and writes
    # the v4 schema. Output goes to `phase0_calibration_v4_bisection.json` so
    # `_select_active_phase0_pick` (BLOCKER A) can choose it.
    pick_cmd = [
        "uv",
        "run",
        "python",
        "scripts/i504_phase_phase0_pick.py",
        "--mode",
        "v4",
        "--slab-root",
        str(args.slab_root),
        "--out-path",
        str(out_pick_path),
        "--v4-trajectory-path",
        str(out_traj),
        "--source",
        args.source,
        "--fixed-lr",
        repr(float(args.fixed_lr)),
        # Round-3 blocker E fix: thread chosen_epochs=2 into the v4 picker so
        # the bisection artifact carries `chosen_epochs=2` (the recipe the
        # bisection actually trained — EPOCHS=2, finer-fraction grid) rather
        # than the picker's default of 3. Without this the picker writes
        # `chosen_epochs=3` AND `chosen_checkpoint_steps = round(frac * 75)`
        # despite the trajectory being from a 50-step EPOCHS=2 run, and
        # `_run_v2_phase1` then trains all 5 main arms at EPOCHS=3 on a
        # `chosen_frac` chosen from an EPOCHS=2 trajectory — a headline
        # recipe mismatch that burns 24 GPU-h on wrong-recipe data.
        "--chosen-epochs",
        "2",
        "--sentinel-path",
        str(LOG_DIR / "issue-504-v4-phase0_bisection-pick-results.json"),
    ]
    pick_rc = 0
    try:
        _run_phase_subprocess(pick_cmd, "v4_phase0_bisection_pick")
    except subprocess.CalledProcessError as e:
        pick_rc = e.returncode
        log.warning(
            "[phase=phase0_v4_bisection_pick] picker exited rc=%d (non-pass "
            "verdict); reading artifact to determine routing.",
            pick_rc,
        )

    if not out_pick_path.exists():
        raise RuntimeError(f"[phase=phase0_v4_bisection] pick artifact missing at {out_pick_path}.")
    pick = json.loads(out_pick_path.read_text())
    # Round-3 blocker E fix (post-hoc invariant): defense in depth — the
    # picker is now invoked with `--chosen-epochs 2` (above), but ALSO assert
    # the artifact landed with chosen_epochs == 2 here and recompute
    # chosen_checkpoint_steps from the actual frac so any future refactor
    # of the picker's step-formula doesn't silently regress the contract.
    if pick.get("verdict") == "pass":
        if pick.get("chosen_epochs") != 2:
            raise RuntimeError(
                f"[phase=phase0_v4_bisection] picker artifact at {out_pick_path} "
                f"carries chosen_epochs={pick.get('chosen_epochs')!r} but the "
                f"bisection trained EPOCHS=2 — invariant violated. Verify the "
                f"picker received --chosen-epochs 2 (round-3 blocker E fix). "
                f"failure_class=code, reason=v4_bisection_chosen_epochs_mismatch"
            )
        chosen_frac = pick.get("chosen_checkpoint_fraction")
        if chosen_frac is not None:
            # 25 steps per epoch (400 rows / effective batch 16) × 2 epochs = 50.
            expected_steps = max(1, round(float(chosen_frac) * 50))
            actual_steps = pick.get("chosen_checkpoint_steps")
            if actual_steps != expected_steps:
                raise RuntimeError(
                    f"[phase=phase0_v4_bisection] picker artifact at "
                    f"{out_pick_path} carries chosen_checkpoint_steps="
                    f"{actual_steps!r} but EPOCHS=2 × 25 steps/epoch × "
                    f"frac={chosen_frac} expects {expected_steps}. The picker's "
                    f"step formula must be `round(frac * steps_per_epoch * "
                    f"chosen_epochs)` with chosen_epochs threaded from the "
                    f"CLI; round-3 blocker E fix. failure_class=code, "
                    f"reason=v4_bisection_chosen_steps_mismatch"
                )
    phase_summaries["v4_phase0_bisection"] = {
        "verdict": pick.get("verdict"),
        "chosen_epochs": pick.get("chosen_epochs"),
        "chosen_lr": pick.get("chosen_lr"),
        "chosen_checkpoint_fraction": pick.get("chosen_checkpoint_fraction"),
        "bystander_resolution_at_pick": pick.get("bystander_resolution_at_pick"),
        "fallback_triggered": pick.get("fallback_triggered"),
        "fallback_reason": pick.get("fallback_reason"),
        "checkpoints_uploaded": uploaded_paths,
    }
    _write_sentinel(
        LOG_DIR / "issue-504-v4-phase0_bisection-results.json",
        kind="epm:progress",
        phase="v4_phase0_bisection_done",
        note_payload={
            "issue": 504,
            "phase": "v4_phase0_bisection",
            "v4_bisection_pick": pick,
            "phase_summaries": phase_summaries,
        },
    )
    log.info(
        "[phase=done] V4 phase0_bisection COMPLETE — verdict=%s, chosen_frac=%s, "
        "bystander_resolution_at_pick=%s, fallback=%s. %s",
        pick.get("verdict"),
        pick.get("chosen_checkpoint_fraction"),
        pick.get("bystander_resolution_at_pick"),
        pick.get("fallback_triggered"),
        datetime.now(UTC).isoformat(),
    )
    # rc=2 on bisection fail: dispatcher should surface
    # epm:failure v1 failure_class=methodology
    # reason=bystander_resolution_unreachable_at_r8_epochs23 (plan v5 §4.2 step 2).
    return 0 if pick.get("verdict") == "pass" else 2


def _run_v4_phase0p6_validate(
    *,
    args: argparse.Namespace,
    phase_summaries: dict[str, dict],
    arm_to_n_json: Path,
) -> int:
    """v4 Phase 0.6 — marker-logprob path validation (plan v5 §4.3a).

    Cheap (~0.05 GPU-h) gate that proves the fixed reader is reading the
    TRAINED model (not the BASE) BEFORE Phase 1 burns 24 GPU-h. On FAIL,
    surfaces epm:failure v1 (failure_class: code, reason:
    marker_logprob_path_still_broken) and Phase 1 MUST NOT spawn.

    v5 round-2 BLOCKER D extension: when the v4-primary picker failed but
    the v4-bisection picker passed, run Phase 0.6 against the v4-bisection
    anchor instead. The HF subfolder + local-root differ; the v4 picker
    schema is identical so the rest of the rig is unchanged.
    """
    primary_pick_path = args.slab_root / "phase0_calibration_v4.json"
    bisection_pick_path = args.slab_root / "phase0_calibration_v4_bisection.json"
    out_path = args.slab_root / "phase0p6_validation_v4.json"

    # Pick the highest-priority passing artifact (v4 > v4_bisection). If both
    # exist but only the bisection passed, use the bisection adapter for the
    # Phase 0.6 spot-check — that is the adapter Phase 1 will actually train
    # against, so the validation must run on it.
    active_pick_path: Path | None = None
    if primary_pick_path.exists():
        primary_pick = json.loads(primary_pick_path.read_text())
        if primary_pick.get("verdict") == "pass":
            active_pick_path = primary_pick_path
    if active_pick_path is None and bisection_pick_path.exists():
        bisection_pick = json.loads(bisection_pick_path.read_text())
        if bisection_pick.get("verdict") == "pass":
            active_pick_path = bisection_pick_path
    if active_pick_path is None:
        raise RuntimeError(
            f"[phase=phase0p6_validate] No passing Phase 0 v4 pick found. "
            f"Checked {primary_pick_path} and {bisection_pick_path}. "
            f"Run --phase phase0_v4_reeval (and --phase phase0_v4_bisection on "
            f"its fallback) first."
        )

    # Subfolder + local-root depend on which v4 path is active. Both layouts
    # are byte-identical for the Phase 0.6 rig — only the path strings change.
    if active_pick_path == bisection_pick_path:
        hf_subfolder_prefix = "adapters/issue_504_v4_bisection/c504v4_bisection_eps2_seed42"
        adapter_local_root = args.runs_root / "v4_bisection_anchor_download"
        log.info(
            "[phase=phase0p6_validate] active pick = v4 bisection (%s) — "
            "running Phase 0.6 against the EPOCHS=2 bisection anchor.",
            active_pick_path,
        )
    else:
        hf_subfolder_prefix = "adapters/issue_504_v4/c504v4_smoke_eps3_seed42"
        adapter_local_root = args.runs_root / "v4_anchor_download"

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/i504_phase_phase0p6.py",
        "--slab-root",
        str(args.slab_root),
        "--phase0-pick-path",
        str(active_pick_path),
        "--panel-json",
        str(arm_to_n_json),
        "--bank-path",
        str(args.bank_path),
        "--out-path",
        str(out_path),
        "--hf-adapter-repo",
        HF_MODEL_REPO,
        "--hf-adapter-subfolder-prefix",
        hf_subfolder_prefix,
        "--adapter-local-root",
        str(adapter_local_root),
        "--source",
        args.source,
        "--sentinel-path",
        str(LOG_DIR / "issue-504-phase0p6-results.json"),
    ]
    rc = 0
    try:
        _run_phase_subprocess(cmd, "phase0p6_validate")
    except subprocess.CalledProcessError as e:
        rc = e.returncode
        log.warning(
            "[phase=phase0p6_validate] non-zero exit (rc=%d) — reading artifact "
            "to determine routing.",
            rc,
        )

    if not out_path.exists():
        raise RuntimeError(f"[phase=phase0p6_validate] validation artifact missing at {out_path}.")
    payload = json.loads(out_path.read_text())
    phase_summaries["phase0p6_validate"] = {
        "verdict": payload.get("verdict"),
        "pass_a": payload.get("pass_a"),
        "pass_b": payload.get("pass_b"),
        "byte_identical_rate": payload.get("byte_identical_rate"),
        "n_byte_identical": payload.get("n_byte_identical"),
        "n_total": payload.get("n_total"),
    }
    _write_sentinel(
        LOG_DIR / "issue-504-phase0p6-final-results.json",
        kind="epm:progress",
        phase="phase0p6_validate_done",
        note_payload={
            "issue": 504,
            "phase": "phase0p6_validate",
            "verdict": payload.get("verdict"),
            "phase0p6": payload,
            "phase_summaries": phase_summaries,
        },
    )
    log.info(
        "[phase=done] PHASE 0.6 validate COMPLETE — verdict=%s, byte_identical_rate=%.4f. %s",
        payload.get("verdict"),
        payload.get("byte_identical_rate"),
        datetime.now(UTC).isoformat(),
    )
    return 0 if payload.get("verdict") == "PASS" else 2


def _run_v2_phase1(  # noqa: C901 -- linear branch ladder over v2/v3/v4 phase 0 pick artifacts
    *,
    args: argparse.Namespace,
    phase_summaries: dict[str, dict],
    arm_to_n_json: Path,
    max_new_tokens_eval: int,
    max_model_len_eval: int,
    seeds: list[int],
) -> int:
    """Run the 5 main arms × 2 seeds at the Phase 0-picked recipe (plan v2 §4.4 / v3 §4.4).

    Reads whichever Phase 0 artifact exists, preferring v3 (EPOCHS-ladder) over
    v2 (lr-ladder) per `_select_active_phase0_pick`. When v3 is active, the
    main arms are the v3 slugs (`c504v3_*`) and the cells train at
    `chosen_epochs` over fixed `chosen_lr=1e-4`. When v2 is active, the main
    arms are the v2 slugs (`c504v2_*`) and the cells train at `chosen_lr`
    over EPOCHS=1.
    """
    primary_pick_path = args.slab_root / "phase0_calibration_v2.json"
    fallback_pick_path = args.slab_root / "phase0_calibration_v2_fallback.json"
    v3_pick_path = args.slab_root / "phase0_calibration_v3.json"
    v4_pick_path = args.slab_root / "phase0_calibration_v4.json"
    v4_bisection_pick_path = args.slab_root / "phase0_calibration_v4_bisection.json"
    # Active pick = v4-primary > v4-bisection > v3 > v2-fallback > v2-primary.
    # Analyze MUST read the same artifact training used (round-3 fix,
    # concern_id `fallback-analyze-pick-path`; v5 round-2 BLOCKER A — Phase 1
    # must consume v4 picker output before falling back to v3/v2).
    pick, active_pick_path = _select_active_phase0_pick(
        primary_pick_path,
        fallback_pick_path,
        v3_pick_path=v3_pick_path,
        v4_pick_path=v4_pick_path,
        v4_bisection_pick_path=v4_bisection_pick_path,
    )
    is_v4 = active_pick_path in (v4_pick_path, v4_bisection_pick_path)
    is_v3 = active_pick_path == v3_pick_path
    if active_pick_path == fallback_pick_path:
        log.info(
            "[phase=phase1] primary phase0_calibration_v2.json fired fallback; "
            "reading fallback pick from %s (source=%s).",
            fallback_pick_path,
            pick.get("source"),
        )
    elif is_v4:
        log.info(
            "[phase=phase1] v4 pick artifact found (%s) AND verdict=pass — "
            "v4 supersedes v3/v2 for Phase 1. chosen_epochs=%s, chosen_lr=%s, "
            "chosen_frac=%s, bystander_resolution_at_pick=%s, source=%s.",
            active_pick_path,
            pick.get("chosen_epochs"),
            pick.get("chosen_lr"),
            pick.get("chosen_checkpoint_fraction"),
            pick.get("bystander_resolution_at_pick"),
            pick.get("source"),
        )
    elif is_v3:
        log.info(
            "[phase=phase1] v3 pick artifact found (%s) AND verdict=pass — "
            "v3 supersedes v2 for Phase 1. chosen_epochs=%s, chosen_lr=%s, "
            "source=%s.",
            v3_pick_path,
            pick.get("chosen_epochs"),
            pick.get("chosen_lr"),
            pick.get("source"),
        )

    if pick.get("verdict") != "pass":
        raise RuntimeError(
            f"Phase 1 cannot proceed: Phase 0 pick verdict={pick.get('verdict')!r}, "
            f"fallback_reason={pick.get('fallback_reason')!r}. "
            f"Re-run --phase phase0-fallback / --phase phase0_v3 / --phase "
            f"phase0_v4_reeval first."
        )

    # v5 round-2 BLOCKER B — Phase 0.6 marker-logprob path validation gate.
    # When the active pick is v4 (primary OR bisection), the Phase 0.6 gate
    # MUST have passed before Phase 1 spawns. Plan v5 §4.4 + §7 "Phase 0.6 →
    # Phase 1" gate language: the dispatcher refuses to advance unless
    # phase0p6_validation_v4.json exists with verdict='PASS'. Reading the
    # artifact here (not just trusting the picker pass) closes the gap where
    # an operator running `--phase phase1` after a failed Phase 0.6 would
    # otherwise burn Phase 1 GPU.
    if is_v4:
        phase06_path = args.slab_root / "phase0p6_validation_v4.json"
        if not phase06_path.exists():
            raise RuntimeError(
                f"Phase 1 cannot proceed under a v4 active pick: Phase 0.6 "
                f"validation artifact missing at {phase06_path}. Run "
                f"`--phase phase0p6_validate` first (plan v5 §4.3a). "
                f"failure_class=code, reason=phase0p6_not_passed_before_phase1"
            )
        phase06 = json.loads(phase06_path.read_text())
        if phase06.get("verdict") != "PASS":
            raise RuntimeError(
                f"Phase 1 cannot proceed under a v4 active pick: Phase 0.6 "
                f"validation verdict={phase06.get('verdict')!r} at "
                f"{phase06_path} (pass_a={phase06.get('pass_a')!r}, "
                f"pass_b={phase06.get('pass_b')!r}, "
                f"byte_identical_rate={phase06.get('byte_identical_rate')!r}). "
                f"failure_class=code, reason=phase0p6_not_passed_before_phase1"
            )
        log.info(
            "[phase=phase1] Phase 0.6 gate PASSED (byte_identical_rate=%s, "
            "n_byte_identical=%s/%s). Proceeding to v4-pick Phase 1 spawn.",
            phase06.get("byte_identical_rate"),
            phase06.get("n_byte_identical"),
            phase06.get("n_total"),
        )

    chosen_lr = float(pick["chosen_lr"])
    chosen_frac = float(pick["chosen_checkpoint_fraction"])
    chosen_rank = int(pick.get("chosen_rank", 8))
    chosen_alpha = int(pick.get("chosen_alpha", 32))
    # v4 + v3 picks carry chosen_epochs; v2 does not (EPOCHS=1 pinned).
    chosen_epochs = int(pick["chosen_epochs"]) if (is_v4 or is_v3) else None
    chosen_source = pick.get("source", args.source)

    # v4 main arms reuse the v3 slugs (recipe is byte-identical: EPOCHS=3,
    # lr=1e-4, r=8/α=32; v4 differs only in the picker logic). label_prefix
    # distinguishes the v4 run on WandB + sentinel paths.
    if is_v4:
        default_arm_slugs = MAIN_ARM_SLUGS_V3
        label_prefix = "issue-504-v4"
        analyze_positioned_arms = "v3"
        phase_summary_key = "v4_phase1"
    elif is_v3:
        default_arm_slugs = MAIN_ARM_SLUGS_V3
        label_prefix = "issue-504-v3"
        analyze_positioned_arms = "v3"
        phase_summary_key = "v3_phase1"
    else:
        default_arm_slugs = MAIN_ARM_SLUGS_V2
        label_prefix = "issue-504-v2"
        analyze_positioned_arms = "v2"
        phase_summary_key = "v2_phase1"

    log.info(
        "[phase=phase1] scheduling %d cells x %d seeds "
        "at lr=%g, frac=%g, rank=%d, alpha=%d, epochs=%s, source=%s",
        len(default_arm_slugs),
        len(seeds),
        chosen_lr,
        chosen_frac,
        chosen_rank,
        chosen_alpha,
        chosen_epochs if chosen_epochs is not None else "default(1)",
        chosen_source,
    )

    # Main arms: use args.cells if explicitly set to a subset, else default
    # to all 5 v2/v3 arms.
    if args.cells is None or args.cells.strip() in ("", "all"):
        main_cells = list(default_arm_slugs)
    else:
        main_cells = _resolve_cells(args.cells, default_arm_slugs)

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
        r_eval_path=args.r_eval_path,
        chosen_rank=chosen_rank,
        chosen_alpha=chosen_alpha,
        chosen_frac=chosen_frac,
        smoke=False,
        no_kl=args.no_kl,
        report_to=args.report_to,
        resume=args.resume,
        max_new_tokens_eval=max_new_tokens_eval,
        max_model_len_eval=max_model_len_eval,
        hf_path_suffix=args.hf_path_suffix,
        label_prefix=label_prefix,
        chosen_lr=chosen_lr,  # applied uniformly to every Phase 1 arm
        chosen_epochs=chosen_epochs,  # v3: chosen by Phase 0 v3 EPOCHS pick
        # Round-2 fix (BLOCKER #2): thread the PICKED source persona through
        # every Phase 1 cell. Every arm trains + evaluates against the SAME
        # source the Phase 0 picker validated as anchorable.
        source_persona=chosen_source,
    )
    phase_summaries[phase_summary_key] = {
        "n_completed": len(cell_results),
        "results": cell_results,
        "chosen_lr": chosen_lr,
        "chosen_epochs": chosen_epochs,
        "chosen_checkpoint_fraction": chosen_frac,
        "source": chosen_source,
    }

    # ── Phase 2: analyze (CPU). ──────────────────────────────────────────────
    # v4 reuses the v3 base-prior + analyze positioned-arms ("v3") since v4
    # main arms recycle the v3 slugs (recipe identical, only picker logic
    # differs). The sentinel path carries `v4` so the dashboard distinguishes
    # the analyze run from a same-task v3-only run.
    if is_v4 or is_v3:
        base_prior_path = args.slab_root / "base_prior_marker_v3.json"
    else:
        base_prior_path = args.slab_root / "base_prior_marker_v2.json"
    analyze_summary: dict | None = None
    if args.skip_analyze:
        log.info("[phase=analyze] SKIP")
    else:
        # Round-2 fix (BLOCKER #1, concern_id `analyze-v2-slug-iteration`): the
        # Phase 2 must iterate the active arm slugs (v2 c504v2_* OR v3/v4 c504v3_*)
        # so Phase 1's trajectories are actually read.
        # Active pick = v4 > v3 > v2-fallback > v2-primary; analyze must read the
        # same artifact training used (round-3 fix, concern_id
        # `fallback-analyze-pick-path`; v5 round-2 BLOCKER A).
        if is_v4:
            analyze_sentinel_name = "issue-504-v4-analyze-results.json"
        elif is_v3:
            analyze_sentinel_name = "issue-504-v3-analyze-results.json"
        else:
            analyze_sentinel_name = "issue-504-v2-analyze-results.json"
        analyze_sentinel = LOG_DIR / analyze_sentinel_name
        _run_phase_subprocess(
            [
                "uv",
                "run",
                "python",
                "scripts/i504_phase_analyze.py",
                "--slab-root",
                str(args.slab_root),
                "--phase0-path",
                str(active_pick_path),
                "--phase05-path",
                str(args.slab_root / "phase0_5_gates.json"),
                "--base-prior-path",
                str(base_prior_path),
                "--seeds",
                ",".join(str(s) for s in seeds),
                "--positioned-arms",
                analyze_positioned_arms,
                "--sentinel-path",
                str(analyze_sentinel),
            ],
            f"{analyze_positioned_arms}_analyze",
        )
        ap = args.slab_root / "analyze_summary.json"
        if ap.exists():
            analyze_summary = json.loads(ap.read_text())
    phase_summaries[f"{analyze_positioned_arms}_analyze"] = analyze_summary

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
        "v2 Phase 1 dispatcher done. %d cell units completed. [phase=done] %s",
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
