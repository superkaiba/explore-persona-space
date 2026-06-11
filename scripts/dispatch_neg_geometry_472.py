# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #472 dispatcher — UNIFIED smoke = sweep with one cell (on-policy geometry).

Pipeline (plan §4.8):
  Phase 0    persona bank (Sonnet 4.5, one async call)         [CPU/API, in-proc]
  Phase 0.5  base centroids L10/L15/L20 over the bank          [GPU subprocess]
  Phase 1    base on-policy R-generation (whole bank)          [GPU subprocess]
  Phase 1.5  base panel marker prior b_logprob                 [GPU subprocess]
  Per cell×seed: build → train (6 ckpts) → eval_trajectory     [GPU subprocess pool]
  Sub-ceiling smoke GATE (after the anchor smoke cell)         [plan §7]
  Phase 5    analyze (regression + figures)                    [subprocess]

UNIFICATION (smoke-architecture parity = PASS_UNIFIED): smoke = this dispatcher
with --cells anchor --seeds 42 --smoke. The per-(cell,seed) unit is ALWAYS
``scripts/i472_run_cell.py`` (one subprocess, GPU-pinned via CUDA_VISIBLE_DEVICES).
Smoke launches exactly ONE of them + a tiny slice; the sweep launches 20 across
8 GPUs (max-parallel=8). Same subprocess shape, same env injection, same
on-policy DV-A(vLLM)+DV-B(HF KL) path, same poll-compliant sentinel + [phase=...]
logging, same teardown sequence.

8-GPU sweep parallelism (plan §9): the dispatcher schedules cell×seed
subprocesses across GPUs, at most --max-parallel concurrent (clamped to --n-gpus)
and each on a DISTINCT free physical GPU. Each cell-subprocess is launched with
``--gpu-id <g>`` (the assigned PHYSICAL GPU index); ``train/sft.py`` SETS
``CUDA_VISIBLE_DEVICES=str(g)`` from cfg.gpu_id (it does NOT respect an inherited
CVD — round-3 #472 OOM: with gpu_id=0 every parallel cell re-targeted physical
GPU 0). The nested eval subprocess inherits that same CVD via os.environ. The
dispatcher does NOT also restrict env CVD (that would make str(g) re-index
against a 1-GPU view). This is sweep parallelism, NOT model sharding (LoRA-7B
fits on 1 GPU). Cheap pre-sweep check: ``--validate-multi-gpu --cells A,B
--seeds 42 --n-gpus 2 --max-parallel 2`` runs 2 cells concurrently on GPUs 0+1.

Pod-side discipline (CLAUDE.md): NEVER shells out to scripts/task.py
(sentinel-file pattern only); every subprocess.* passes env={**os.environ};
load_dotenv() at module top; vLLM phases are subprocess-isolated; sets
EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 (MooseFS quota) + EPM_PERSIST_ADAPTER_HF_REPO
so the adapter persists fail-loud before any cleanup.

Sub-ceiling smoke GATE (plan §7 / §14): after the anchor smoke cell, checks the
anchor's held-out g_logp at the matched slice. If saturated (held-out g_logp
NOT ≥5 nats below ceiling on ≥80% of held-out personas at 1 epoch), the
dispatcher EXITS NON-ZERO with a clear message instructing the caller to re-run
with --fallback (r=16/lr=5e-6/0.5 epoch). The code path RUNS at pod time; the
fallback recipe is wired now.
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

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
DEFAULT_SEEDS = (42, 137)
SUBCEILING_HEADROOM_NATS = 5.0
SUBCEILING_MIN_FRACTION = 0.80  # ≥80% of held-out personas must be sub-ceiling.

LOG_DIR = Path("/workspace/logs")  # overridden by --log-dir.

log = logging.getLogger("dispatch_neg_geometry_472")


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
                "task_id": 472,
                "by": "dispatch_neg_geometry_472",
                "ts": datetime.now(UTC).isoformat(),
                "phase": phase,
                "note": json.dumps(note_payload),
            },
            indent=2,
        )
    )


def _resolve_cells(raw: str | None, force_anchor_only: bool) -> list[str]:
    """Resolve the requested cell slugs.

    ``force_anchor_only`` (the canonical single-cell smoke, which also runs the
    sub-ceiling science gate) returns just the anchor regardless of ``raw``. The
    multi-GPU validation mode (``--validate-multi-gpu``) sets force_anchor_only=
    False so it can run the user's ``--cells`` (e.g. two cells) concurrently with
    the tiny ``--smoke`` slice, to confirm distinct-GPU placement (round-3 #472).
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import CELL_SPECS

    slugs = [c[0] for c in CELL_SPECS]
    name_to_slug = {c[1]: c[0] for c in CELL_SPECS}
    if force_anchor_only:
        return ["c472_anchor"]
    if raw is None or raw.strip() in ("", "all"):
        return slugs
    out: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in slugs:
            out.append(tok)
        elif tok in name_to_slug:
            out.append(name_to_slug[tok])
        elif tok == "anchor":
            out.append("c472_anchor")
        else:
            raise ValueError(f"Unknown cell {tok!r}. Slugs: {slugs}")
    return out


def _run_phase_subprocess(cmd: list[str], phase: str) -> None:
    log.info("[phase=%s] subprocess: %s", phase, " ".join(cmd))
    subprocess.run(cmd, env={**os.environ}, check=True)


def _persona_bank_phase(skip: bool, dry_run: bool, bank_path: Path) -> dict:
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        build_persona_bank,
    )

    log.info("[phase=persona_bank] starting")
    if skip and bank_path.exists():
        log.info("[phase=persona_bank] SKIP (exists at %s)", bank_path)
        return {"status": "skipped_exists", "bank_path": str(bank_path)}
    return build_persona_bank(out_path=bank_path, dry_run=dry_run)


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
    smoke: bool,
    fallback: bool,
    no_kl: bool,
    report_to: str,
    resume: bool,
) -> list[dict]:
    """Run all (cell, seed) units as a GPU-sharded subprocess pool.

    Assigns each concurrent cell a DISTINCT FREE physical GPU and keeps at most
    ``max_parallel`` cell-subprocesses alive. Each cell is launched with
    ``--gpu-id <g>``; ``train/sft.py`` SETS ``CUDA_VISIBLE_DEVICES=str(g)`` so
    the cell + its nested eval run on physical GPU g (round-3 #472 fix). A free
    GPU is reclaimed when its cell exits, so two concurrent cells NEVER share a
    GPU. ``max_parallel`` is clamped to ``n_gpus`` (you can't run more concurrent
    one-GPU cells than there are GPUs without colliding).
    """
    units = [(c, s) for c in cells for s in seeds]
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
        "Scheduling %d (cell,seed) units across %d GPUs (max_parallel=%d)",
        len(units),
        n_gpus,
        max_parallel,
    )

    results: list[dict] = []
    running: list[tuple[subprocess.Popen, str, int, int]] = []  # (proc, cell, seed, gpu)
    queue = list(units)
    free_gpus: list[int] = list(range(n_gpus))  # physical GPU indices currently free

    def _launch(cell: str, seed: int, gpu: int) -> subprocess.Popen:
        out_traj = slab_root / f"{cell}_seed{seed}" / "trajectory.json"
        if resume and out_traj.exists():
            log.info("[%s seed%d] RESUME: trajectory exists; skipping.", cell, seed)
            return None  # type: ignore[return-value]
        # GPU pinning (round-3 #472 fix): pass the ASSIGNED PHYSICAL GPU index via
        # --gpu-id; train/sft.py SETS CUDA_VISIBLE_DEVICES=str(gpu_id) against the
        # FULL host enumeration. We do NOT also restrict env CVD here — if we did,
        # sft.py's str(gpu) would re-index against the already-restricted 1-GPU
        # view (gpu>=1 → invalid). Inherit the full env so sft.py owns CVD.
        env = {**os.environ}
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/i472_run_cell.py",
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
            "--report-to",
            report_to,
        ]
        if smoke:
            cmd.append("--smoke")
        if fallback:
            cmd.append("--fallback")
        if no_kl:
            cmd.append("--no-kl")
        cell_log = log_dir / f"issue-472-{cell}-seed{seed}.log"
        cell_log.parent.mkdir(parents=True, exist_ok=True)
        log.info("[%s seed%d] launch on GPU %d → %s", cell, seed, gpu, cell_log)
        # File handle must outlive this function (Popen writes to it while
        # running); the dispatcher closes the process, not this fh — the OS
        # closes it on the child's exit. A context manager would close it here.
        fh = open(cell_log, "w")  # noqa: SIM115 -- handle lives for the Popen's lifetime
        return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)

    while queue or running:
        # Fill up to max_parallel, one DISTINCT free GPU per concurrent cell.
        while queue and len(running) < max_parallel and free_gpus:
            cell, seed = queue.pop(0)
            gpu = free_gpus.pop(0)  # a genuinely free physical GPU (never shared)
            proc = _launch(cell, seed, gpu)
            if proc is None:
                results.append({"cell": cell, "seed": seed, "status": "resumed_skip"})
                free_gpus.append(gpu)  # nothing launched; return the GPU
                continue
            running.append((proc, cell, seed, gpu))
        # Reap finished; return their GPUs to the free pool.
        still: list[tuple[subprocess.Popen, str, int, int]] = []
        for proc, cell, seed, gpu in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, cell, seed, gpu))
                continue
            free_gpus.append(gpu)  # GPU is free again for the next queued cell
            if rc != 0:
                # Fail loud — write a FAILED sentinel and raise (whole sweep aborts).
                fail_path = log_dir / f"issue-472-{cell}-seed{seed}-FAILED.json"
                fail_path.write_text(
                    json.dumps(
                        {"cell": cell, "seed": seed, "returncode": rc, "assigned_gpu": gpu},
                        indent=2,
                    )
                )
                # Drain other running procs before raising.
                for p2, _c2, _s2, _g2 in still:
                    p2.terminate()
                raise RuntimeError(
                    f"[{cell} seed{seed}] cell subprocess exited rc={rc} (GPU {gpu}). "
                    f"See {log_dir}/issue-472-{cell}-seed{seed}.log. Sweep aborted."
                )
            log.info("[%s seed%d] DONE (GPU %d)", cell, seed, gpu)
            results.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "status": "done",
                    "assigned_gpu": gpu,
                    "trajectory_path": str(slab_root / f"{cell}_seed{seed}" / "trajectory.json"),
                    "adapter_hf_path": f"adapters/issue_472/{cell}_seed{seed}",
                }
            )
        running = still
        if running:
            time.sleep(5)
    return results


def _summarize_gpu_placements(cell_results: list[dict], *, n_gpus: int) -> dict:
    """Summarize the per-cell GPU placement for the multi-GPU validation.

    The free-GPU pool guarantees no two CONCURRENT cells share a physical GPU,
    and each cell's ``verify_gpu_pin`` already fail-loud-asserted its pin in-
    process. This summary surfaces the placement map + a sanity flag that the
    completed cells used >1 distinct GPU (i.e. concurrency actually happened, not
    serial-on-one-GPU). ``ok`` is True when ≥2 distinct GPUs were used across the
    completed cells (the whole point of the validation).
    """
    done = [r for r in cell_results if r.get("status") == "done" and "assigned_gpu" in r]
    placement = {f"{r['cell']}_seed{r['seed']}": r["assigned_gpu"] for r in done}
    distinct = sorted({r["assigned_gpu"] for r in done})
    return {
        "ok": len(distinct) >= 2,
        "n_cells_done": len(done),
        "n_distinct_gpus_used": len(distinct),
        "distinct_gpus": distinct,
        "n_gpus_available": n_gpus,
        "placement": placement,
        "note": (
            "Each cell also fail-loud-verified its physical-GPU pin in-process "
            "(verify_gpu_pin); see [gpu-pin] verified lines in the per-cell logs."
        ),
    }


def _subceiling_gate(slab_root: Path, smoke_cell: str, smoke_seed: int) -> dict:
    """Sub-ceiling smoke GATE (plan §7): is the anchor sub-ceiling at the slice?

    Reads the smoke cell's trajectory; checks that at the matched-slice checkpoint
    (or the closest), held-out g_logp is ≥ SUBCEILING_HEADROOM_NATS below the 0.0
    ceiling on ≥ SUBCEILING_MIN_FRACTION of held-out personas. Returns a verdict
    dict; the caller exits non-zero on FAIL.
    """
    import numpy as np

    traj_path = slab_root / f"{smoke_cell}_seed{smoke_seed}" / "trajectory.json"
    if not traj_path.exists():
        return {"ok": False, "reason": f"smoke trajectory missing at {traj_path}"}
    traj = json.loads(traj_path.read_text())
    cks = traj["checkpoints"]
    # Use the EARLIEST checkpoint (most sub-ceiling) for the gate — if even the
    # earliest is already saturated, the DV cannot be read sub-ceiling.
    earliest = min(cks, key=lambda c: c["frac"])
    per_persona_mean_g = []
    for per_q in earliest["held_out"].values():
        vals = [held["g_logp"] for held in per_q.values()]
        if vals:
            per_persona_mean_g.append(float(np.mean(vals)))
    if not per_persona_mean_g:
        return {"ok": False, "reason": "no held-out g_logp in smoke trajectory"}
    n_subceiling = sum(1 for g in per_persona_mean_g if g < -SUBCEILING_HEADROOM_NATS)
    frac_subceiling = n_subceiling / len(per_persona_mean_g)
    ok = frac_subceiling >= SUBCEILING_MIN_FRACTION
    # Collapse-aware reporting (round-2 #472): the Qwen marker implant collapses the
    # source's OWN R to marker-spam mid-training; report the per-checkpoint R-collapse
    # share + the source-self collapse onset so the gate decision sees WHERE the
    # readable window ends, not just whether the earliest checkpoint is sub-ceiling.
    collapse_trajectory = [
        {
            "frac": c["frac"],
            "held_out_collapse_share": c.get("held_out_collapse_share"),
            "source_R_collapsed": c.get("source_self", {}).get("r_collapsed"),
            "source_self_delta_g": c.get("source_self", {}).get("delta_g_mean"),
        }
        for c in sorted(cks, key=lambda c: c["frac"])
    ]
    earliest_collapse_share = earliest.get("held_out_collapse_share", 0.0) or 0.0
    return {
        "ok": ok,
        "earliest_frac": earliest["frac"],
        "n_held_out": len(per_persona_mean_g),
        "n_subceiling": n_subceiling,
        "frac_subceiling": frac_subceiling,
        "headroom_nats": SUBCEILING_HEADROOM_NATS,
        "min_fraction": SUBCEILING_MIN_FRACTION,
        "source_self_max_delta_g": max(c["source_self"]["delta_g_mean"] for c in cks),
        "earliest_held_out_collapse_share": earliest_collapse_share,
        "collapse_trajectory": collapse_trajectory,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells", default=None, help="CSV of slugs/names, or 'all'. Default all 10."
    )
    parser.add_argument("--seeds", default="42,137")
    parser.add_argument("--smoke", action="store_true", help="--cells anchor + tiny slice + gate.")
    parser.add_argument(
        "--validate-multi-gpu",
        action="store_true",
        help=(
            "Cheap distinct-GPU placement check (round-3 #472): run the user's "
            "--cells (e.g. two cells) CONCURRENTLY with the tiny --smoke slice "
            "across --n-gpus, WITHOUT forcing 1 cell and WITHOUT the sub-ceiling "
            "science gate. Confirms each cell pins to its own physical GPU before "
            "the expensive 20-run sweep. Example: --validate-multi-gpu "
            "--cells c472_anchor,c472_near --seeds 42 --n-gpus 2 --max-parallel 2."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate imports + bank only, no GPU."
    )
    parser.add_argument(
        "--fallback", action="store_true", help="Sub-ceiling fallback recipe (plan §7)."
    )
    parser.add_argument("--no-kl", action="store_true", help="Skip DV-B KL (smoke speed-up).")
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_472"))
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_472"))
    parser.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    parser.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    parser.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/issue_472"))
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-persona-bank", action="store_true")
    parser.add_argument("--skip-centroids", action="store_true")
    parser.add_argument("--skip-r-generate", action="store_true")
    parser.add_argument("--skip-base-panel", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--r-no-upload", action="store_true")
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

    # `tiny_slice` (the cheap --smoke training/eval slice) is true for BOTH the
    # canonical single-cell smoke AND the multi-GPU placement validation; only the
    # canonical smoke forces 1 cell + runs the sub-ceiling SCIENCE gate.
    tiny_slice = args.smoke or args.validate_multi_gpu
    force_anchor_only = args.smoke and not args.validate_multi_gpu
    run_science_gate = args.smoke and not args.validate_multi_gpu

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if force_anchor_only:
        seeds = seeds[:1]
    cells = _resolve_cells(args.cells, force_anchor_only)
    if args.validate_multi_gpu and len(cells) < 2:
        raise ValueError(
            "--validate-multi-gpu needs ≥2 cells to exercise concurrency (round-3 #472). "
            "Pass e.g. --cells c472_anchor,c472_near."
        )
    log.info(
        "Resolved %d cells: %s; seeds=%s (tiny_slice=%s, validate_multi_gpu=%s)",
        len(cells),
        cells,
        seeds,
        tiny_slice,
        args.validate_multi_gpu,
    )

    # ── Pre-flight: marker tokenizer assertion (CLAUDE.md). ──────────────────
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

    phase_summaries: dict[str, dict] = {}

    # ── Phase 0: persona bank. ───────────────────────────────────────────────
    pb = _persona_bank_phase(args.skip_persona_bank, args.dry_run, args.bank_path)
    phase_summaries["persona_bank"] = pb
    _write_sentinel(
        LOG_DIR / "issue-472-persona-bank-results.json",
        kind="epm:progress",
        phase="persona_bank",
        note_payload=pb,
    )

    if args.dry_run:
        log.info("[phase=done] DRY-RUN complete (imports + bank validated).")
        return 0

    # ── Phase 0.5: centroids. ────────────────────────────────────────────────
    if not args.skip_centroids:
        _run_phase_subprocess(
            [
                "uv",
                "run",
                "python",
                "scripts/i472_phase_centroids.py",
                "--bank-path",
                str(args.bank_path),
                "--out-dir",
                str(args.centroids_dir),
                "--sentinel-path",
                str(LOG_DIR / "issue-472-centroids-results.json"),
            ],
            "centroids",
        )
    else:
        log.info("[phase=centroids] SKIP")

    # ── Phase 1: base R-generation. ──────────────────────────────────────────
    if not args.skip_r_generate:
        r_cmd = [
            "uv",
            "run",
            "python",
            "scripts/i472_phase_r_generate.py",
            "--bank-path",
            str(args.bank_path),
            "--sentinel-path",
            str(LOG_DIR / "issue-472-r-generate-results.json"),
        ]
        if args.r_no_upload:
            r_cmd.append("--no-upload")
        _run_phase_subprocess(r_cmd, "r_generate")
    else:
        log.info("[phase=r_generate] SKIP")

    # ── Phase 1.5: base panel. ───────────────────────────────────────────────
    if not args.skip_base_panel:
        _run_phase_subprocess(
            [
                "uv",
                "run",
                "python",
                "scripts/i472_phase_base_panel.py",
                "--bank-path",
                str(args.bank_path),
                "--centroids-dir",
                str(args.centroids_dir),
                "--out-path",
                str(args.slab_root / "base_panel.json"),
                "--sentinel-path",
                str(LOG_DIR / "issue-472-base-panel-results.json"),
            ],
            "base_panel",
        )
    else:
        log.info("[phase=base_panel] SKIP")

    # ── Per-cell pool (build → train → eval_trajectory). ─────────────────────
    log.info("[phase=cells] scheduling %d cells x %d seeds", len(cells), len(seeds))
    cell_results = _schedule_cell_pool(
        cells=cells,
        seeds=seeds,
        n_gpus=args.n_gpus,
        max_parallel=args.max_parallel,
        slab_root=args.slab_root,
        runs_root=args.runs_root,
        log_dir=LOG_DIR,
        bank_path=args.bank_path,
        centroids_dir=args.centroids_dir,
        smoke=tiny_slice,  # tiny train/eval slice for smoke AND multi-GPU validation
        fallback=args.fallback,
        no_kl=args.no_kl,
        report_to=args.report_to,
        resume=args.resume,
    )
    phase_summaries["cells"] = {"n_completed": len(cell_results), "results": cell_results}

    # ── Multi-GPU placement validation: confirm distinct-GPU pinning, no gate. ─
    if args.validate_multi_gpu:
        placements = _summarize_gpu_placements(cell_results, n_gpus=args.n_gpus)
        phase_summaries["multi_gpu_validation"] = placements
        _write_sentinel(
            LOG_DIR / "issue-472-multi-gpu-validation-results.json",
            kind="epm:progress",
            phase="multi_gpu_validation",
            note_payload=placements,
        )
        log.info("[phase=multi_gpu_validation] %s", placements)
        _write_final_sentinel(
            cells, cell_results, phase_summaries, None, seeds, args.slab_root, status="done"
        )
        log.info(
            "[phase=done] multi-GPU validation complete (%d cells across %d GPUs) %s",
            len(cell_results),
            args.n_gpus,
            datetime.now(UTC).isoformat(),
        )
        return 0

    # ── Sub-ceiling smoke GATE (plan §7). ────────────────────────────────────
    if run_science_gate:
        gate = _subceiling_gate(args.slab_root, "c472_anchor", seeds[0])
        phase_summaries["subceiling_gate"] = gate
        _write_sentinel(
            LOG_DIR / "issue-472-subceiling-gate-results.json",
            kind="epm:progress",
            phase="subceiling_gate",
            note_payload=gate,
        )
        log.info("[phase=subceiling_gate] %s", gate)
        if not gate.get("ok"):
            log.error(
                "[phase=subceiling_gate] FAIL: anchor saturated at 1 epoch "
                "(frac_subceiling=%.2f < %.2f). Re-run the sweep with --fallback "
                "(r=16/lr=5e-6/0.5 epoch) per plan §7 before committing the full sweep.",
                gate.get("frac_subceiling", float("nan")),
                SUBCEILING_MIN_FRACTION,
            )
            _write_final_sentinel(
                cells,
                cell_results,
                phase_summaries,
                None,
                seeds,
                args.slab_root,
                status="subceiling_gate_failed",
            )
            log.info(
                "[phase=done] dispatcher exit (smoke gate FAIL) %s", datetime.now(UTC).isoformat()
            )
            return 2

    # ── Phase 5: analyze (skip under smoke — only one cell). ─────────────────
    analyze_summary: dict | None = None
    if args.smoke or args.skip_analyze:
        log.info("[phase=analyze] SKIP (%s)", "smoke" if args.smoke else "--skip-analyze")
    else:
        _run_phase_subprocess(
            [
                "uv",
                "run",
                "python",
                "scripts/i472_phase_analyze.py",
                "--slab-root",
                str(args.slab_root),
                "--base-panel-path",
                str(args.slab_root / "base_panel.json"),
                "--centroids-dir",
                str(args.centroids_dir),
                "--figures-dir",
                str(args.figures_dir),
                "--seeds",
                ",".join(str(s) for s in seeds),
                "--sentinel-path",
                str(LOG_DIR / "issue-472-analyze-results.json"),
            ],
            "analyze",
        )
        ap = args.slab_root / "analyze_summary.json"
        if ap.exists():
            analyze_summary = json.loads(ap.read_text())
    phase_summaries["analyze"] = analyze_summary

    _write_final_sentinel(
        cells, cell_results, phase_summaries, analyze_summary, seeds, args.slab_root, status="done"
    )
    log.info("Dispatcher done. %d cell units completed.", len(cell_results))
    log.info("[phase=done] dispatcher exit %s", datetime.now(UTC).isoformat())
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
    final_path = LOG_DIR / "issue-472-results.json"
    note_payload = {
        "issue": 472,
        "status": status,
        "seeds": seeds,
        "cells_requested": cells,
        "n_units_completed": len(
            [c for c in cell_results if c.get("status") in ("done", "resumed_skip")]
        ),
        "n_units": len(cells) * len(seeds),
        "cell_results": cell_results,
        "barrier_bubble_verdict": (analyze_summary or {}).get("barrier_bubble_verdict"),
        "identification_gate": (analyze_summary or {}).get("identification_gate"),
        "collinearity_gate": (analyze_summary or {}).get("collinearity_gate"),
        "reproducibility": {
            "base_model": BASE_MODEL,
            "hf_model_repo": HF_MODEL_REPO,
            "hf_data_repo": HF_DATA_REPO,
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
