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
    FALLBACK_SOURCE_CANDIDATES,
    HF_DATA_PREFIX_504,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    LR_FROM_V2_SMOKE_SLUG,
    LR_LADDER,
    MAIN_ARM_SLUGS,
    MAIN_ARM_SLUGS_V2,
    MARKER_TEXT,
    PHASE0_SMOKE_SLUGS,
    PHASE0_SMOKE_SLUGS_V2,
    SEEDS,
    SOURCE_PERSONA,
    alpha_for_rank,
)

# Force-reference v2 symbols so ruff's F401 auto-strip on the formatter
# pre-commit pass doesn't remove them — see `feedback_ruff_strips_unused_imports`.
# All four are used inside `main()` / `_run_v2_phase*()` below; this tuple
# keeps the imports alive even when the formatter rewrites the import block.
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
    source_persona: str | None = None,
    per_cell_tolerant: bool = False,
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
        choices=("legacy", "phase0", "phase0-fallback", "phase1"),
        default="legacy",
        help=(
            "v2 phase to run (plan v2 §4.1 + §10): "
            "phase0=run the 3 lr-ladder smoke cells + pick (writes "
            "phase0_calibration_v2.json); "
            "phase0-fallback=re-run the 3 lr-ladder smoke on a fallback source "
            "persona (--source); "
            "phase1=read phase0_calibration_v2.json + train the 5 main arms × 2 "
            "seeds at the picked lr; "
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
        "--source",
        default=SOURCE_PERSONA,
        help=(
            "Source persona name. Default villain (plan v2 §10). "
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


def _select_active_phase0_pick(
    primary_pick_path: Path, fallback_pick_path: Path
) -> tuple[dict, Path]:
    """Choose the active Phase 0 pick artifact: fallback if fallback fired, else primary.

    Pure helper (read-only filesystem access). Returns the parsed dict + the
    absolute path of the chosen artifact. Used by `_run_v2_phase1` for both
    cell-training selection AND the analyze subprocess so they read the SAME
    artifact (round-3 fix, concern_id `fallback-analyze-pick-path`).

    Raises FileNotFoundError if `primary_pick_path` does not exist.
    """
    if not primary_pick_path.exists():
        raise FileNotFoundError(
            f"v2 Phase 1 requires {primary_pick_path}; run --phase phase0 first."
        )
    primary_pick = json.loads(primary_pick_path.read_text())
    if primary_pick.get("fallback_triggered") and fallback_pick_path.exists():
        fallback_pick = json.loads(fallback_pick_path.read_text())
        return fallback_pick, fallback_pick_path
    return primary_pick, primary_pick_path


def _run_v2_phase1(
    *,
    args: argparse.Namespace,
    phase_summaries: dict[str, dict],
    arm_to_n_json: Path,
    max_new_tokens_eval: int,
    max_model_len_eval: int,
    seeds: list[int],
) -> int:
    """Run the 5 v2 main arms × 2 seeds at the Phase 0-picked lr (plan v2 §4.4)."""
    primary_pick_path = args.slab_root / "phase0_calibration_v2.json"
    fallback_pick_path = args.slab_root / "phase0_calibration_v2_fallback.json"
    # Active pick = fallback artifact if fallback fired, else primary; analyze
    # must read the same artifact training used (round-3 fix, concern_id
    # `fallback-analyze-pick-path`: see analyze subprocess below where
    # active_pick_path is threaded as --phase0-path).
    pick, active_pick_path = _select_active_phase0_pick(primary_pick_path, fallback_pick_path)
    if active_pick_path == fallback_pick_path:
        log.info(
            "[phase=v2_phase1] primary phase0_calibration_v2.json fired fallback; "
            "reading fallback pick from %s (source=%s).",
            fallback_pick_path,
            pick.get("source"),
        )

    if pick.get("verdict") != "pass":
        raise RuntimeError(
            f"v2 Phase 1 cannot proceed: Phase 0 pick verdict={pick.get('verdict')!r}, "
            f"fallback_reason={pick.get('fallback_reason')!r}. "
            f"Re-run --phase phase0-fallback on an easier source first."
        )
    chosen_lr = float(pick["chosen_lr"])
    chosen_frac = float(pick["chosen_checkpoint_fraction"])
    chosen_rank = int(pick.get("chosen_rank", 8))
    chosen_alpha = int(pick.get("chosen_alpha", 32))
    chosen_source = pick.get("source", args.source)
    log.info(
        "[phase=v2_phase1] scheduling %d cells x %d seeds "
        "at lr=%g, frac=%g, rank=%d, alpha=%d, source=%s",
        len(MAIN_ARM_SLUGS_V2),
        len(seeds),
        chosen_lr,
        chosen_frac,
        chosen_rank,
        chosen_alpha,
        chosen_source,
    )

    # v2 main arms: use args.cells if explicitly set to a subset, else
    # default to all 5 v2 arms.
    if args.cells is None or args.cells.strip() in ("", "all"):
        main_cells = list(MAIN_ARM_SLUGS_V2)
    else:
        main_cells = _resolve_cells(args.cells, MAIN_ARM_SLUGS_V2)

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
        label_prefix="issue-504-v2",
        chosen_lr=chosen_lr,  # v2: applied uniformly to every Phase 1 arm
        # Round-2 fix (BLOCKER #2): thread the PICKED source persona (from the
        # phase0_calibration_v2.json artifact — the fallback pick if fallback
        # fired, else the primary pick = villain) through every Phase 1 cell.
        # Every arm trains + evaluates against the SAME source the Phase 0
        # picker validated as anchorable.
        source_persona=chosen_source,
    )
    phase_summaries["v2_phase1"] = {
        "n_completed": len(cell_results),
        "results": cell_results,
        "chosen_lr": chosen_lr,
        "chosen_checkpoint_fraction": chosen_frac,
        "source": chosen_source,
    }

    # ── Phase 2: analyze (CPU). ──────────────────────────────────────────────
    base_prior_path = args.slab_root / "base_prior_marker_v2.json"
    analyze_summary: dict | None = None
    if args.skip_analyze:
        log.info("[phase=analyze] SKIP")
    else:
        # Round-2 fix (BLOCKER #1, concern_id `analyze-v2-slug-iteration`): the
        # v2 Phase 2 must iterate the v2 arm slugs (c504v2_*) so Phase 1's
        # trajectories are actually read. Without --positioned-arms v2, the
        # CLI default (v2) is still v2, but pin it explicitly here so a future
        # default flip cannot silently break this path.
        # Active pick = fallback artifact if fallback fired, else primary;
        # analyze must read the same artifact training used (round-3 fix,
        # concern_id `fallback-analyze-pick-path`). Without this, a fallback
        # Phase 1 run trains correctly on the fallback source but the analyze
        # subprocess receives the non-pass primary artifact and `load_phase0_pick`
        # raises RuntimeError after the ~22 GPU-h Phase 1 budget is spent.
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
                "v2",
                "--sentinel-path",
                str(LOG_DIR / "issue-504-v2-analyze-results.json"),
            ],
            "v2_analyze",
        )
        ap = args.slab_root / "analyze_summary.json"
        if ap.exists():
            analyze_summary = json.loads(ap.read_text())
    phase_summaries["v2_analyze"] = analyze_summary

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
