#!/usr/bin/env python3
"""Issue #667 ALL-28-LAYER re-extraction dispatcher (exploratory, off the #537 adapters).

Re-runs the #667 paired base/post-FT forward-pass extractor over ALL 28 residual
layers (0-27) of Qwen-2.5-7B-Instruct, sharded 8-ways across an 8×H100 pod, one
CVD-pinned subprocess per (behavior, source-adapter) cell. The ONLY substantive
change vs :mod:`issue667_dispatch` (extract phase) is capturing every layer's
``v0``/``v_plus`` (the answer-span mean, base θ0 AND adapter θ⁺) instead of just
7/14/21 — reduced-on-the-fly per layer so no full-seq×28 tensor is ever retained
(the #671 memory-safe hook path). The ``c_C_all_layers`` context keys already
carried 28 layers; this brings the answer-side reads to matching depth coverage.

Writes per-cell ``.npz`` to a NEW namespace
``eval_results/issue_667_alllayer/analysis_tensors/`` (local) →
``issue667_alllayer/analysis_tensors`` (HF data repo) so the committed 7/14/21
store (``issue667_gate_chain_preview/analysis_tensors``) is NEVER clobbered.

Cells: 4 SOURCE behaviors (em, fact, marker, sycophancy) × 16 source_cids = 64.
Each cell writes 30 targets × 28 layers = 840 ``.npz`` (the extractor's default
target list is ``eval_cids_for(behavior)`` + the source diagonal).

PASS_UNIFIED architectural parity: the smoke IS this sweep scaled down — the SAME
``--behaviors`` / ``--sources`` / ``--targets`` filters parameterize the ONE phase
this dispatcher runs (extract), the SAME per-cell subprocess shape, the SAME
CVD-pinned wave fan-out, the SAME env injection, the SAME sentinel + ``[phase=...]``
logging. A ``--smoke`` run (1 behavior / 1-2 sources / 2-3 targets, capped probes)
exercises the identical code path; only the cell COUNT differs. Extract is the only
phase (the depth-profile analysis is a separate off-pod entrypoint —
``scripts/issue667_alllayer_analysis.py`` — over the uploaded store).

Pod-side contract (CLAUDE.md / poll_pipeline.py): ``[phase=<name>]`` log lines, a
terminal ``[phase=done]`` (RESERVED — never on per-cell echoes), and an
end-of-run sentinel JSON at
``/workspace/logs/issue-667-<kind_slug>-<epoch>.json`` carrying
``_SENTINEL_REQUIRED_KEYS`` (sentinel_schema_version / kind / version).

WAVE-SIZE CONTRACT (feedback: dispatcher_wave_size_must_match_visible_gpus, #667
a36): the parallel wave size is derived from ``torch.cuda.device_count()`` (NOT a
hardcoded constant, NOT the ``--n-gpus`` default) and asserted to equal the number
of visible GPUs the launch env exposes — a surplus ``--gpu-id`` lane on a smaller
box would otherwise silently fall back to CPU and hang the dispatcher. ``--n-gpus``
is a CEILING, not the source of truth. Per-cell ``CUDA_VISIBLE_DEVICES=<gpu>`` is
pinned in the LAUNCHER env (#545) with the matching ``--gpu-id``.

Launch (8×H100 pod, after code is synced + adapters are reachable)::

    uv run python scripts/issue667_alllayer_dispatch.py extract \\
        --behaviors em,sycophancy,fact,marker --n-gpus 8

Smoke (the unified single-cell sweep — CPU-safe local dry check)::

    uv run python scripts/issue667_alllayer_dispatch.py extract \\
        --behaviors em --sources default,sp_swe --targets default,sp_swe,fmt_json \\
        --cpu-only --smoke --skip-upload
"""

# ruff: noqa: RUF002  # math/scientific notation in docstrings + messages

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Add scripts/ so the cross-script ``import issue667_extract`` resolves when this
# dispatcher is launched as a script (cwd-independent).
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger("issue667_alllayer_dispatch")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# NEW namespace — never clobbers the committed 7/14/21 store
# (issue667_gate_chain_preview/analysis_tensors).
HF_ANALYSIS_TENSORS_PREFIX = "issue667_alllayer/analysis_tensors"
TENSORS_DIR = "eval_results/issue_667_alllayer/analysis_tensors"
# Qwen-2.5-7B-Instruct hidden-layer count; the all-layer sweep captures 0..N_LAYERS-1.
N_LAYERS = 28
ALL_LAYERS = tuple(range(N_LAYERS))
PRIMARY_LAYER = 14  # t+/t-/r_b_fact land here (must be in [0, N_LAYERS-1]).
# All 4 SOURCE-behavior dirs in the #537 grid (refusal is target-only — no source dir).
DEFAULT_BEHAVIORS = ("em", "sycophancy", "fact", "marker")
# The extractor defaults --seed to 42; a cell's output dir is
# <TENSORS_DIR>/<behavior>/<source>_seed42 (issue667_extract cell_dir). Used by the
# resume-skip check to detect already-extracted cells on relaunch.
_EXTRACT_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Log dir + phase lines + sentinel (identical contract to issue667_dispatch)
# ─────────────────────────────────────────────────────────────────────────────


def _log_dir() -> Path:
    override = os.environ.get("EPM_LOG_DIR")
    if override:
        d = Path(override)
    else:
        d = Path("/workspace/logs")
        if not d.exists():
            d = PROJECT_ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def phase_log(name: str) -> None:
    """Emit the ``[phase=<name>]`` line poll_pipeline.py parses (PHASE_RE)."""
    print(f"[phase={name}]", flush=True)


def write_sentinel(kind: str, note: str, *, version: int = 1, extra: dict | None = None) -> Path:
    """End-of-run sentinel with poll_pipeline's _SENTINEL_REQUIRED_KEYS."""
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "task_id": 667,
        "by": "issue667_alllayer_dispatch",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": note,
    }
    if extra:
        payload.update(extra)
    slug = kind.replace(":", "_")
    out = _log_dir() / f"issue-667-{slug}-{time.time_ns()}.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("sentinel written: %s", out)
    return out


def _require_credentials() -> None:
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"


def _run_parallel_with_log(
    cmds: Iterable[tuple[Sequence[str], Path, dict[str, str] | None]],
) -> list[int]:
    """Run several subprocesses concurrently (one wave). Returns rc list.

    Explicit ``env={**os.environ}`` (+ per-cell extra_env) — ``uv run python``
    does not auto-load .env, so load_dotenv() at main()-top puts the creds in
    os.environ first (#397 round-10').
    """
    procs: list[subprocess.Popen] = []
    files = []
    for cmd, log_path, extra_env in cmds:
        env = {**os.environ}
        if extra_env:
            env.update(extra_env)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = log_path.open("ab")
        files.append(f)
        logger.info("$ (parallel) %s  >>> %s", " ".join(shlex.quote(c) for c in cmd), log_path)
        procs.append(
            subprocess.Popen(
                list(cmd), stdout=f, stderr=subprocess.STDOUT, env=env, cwd=PROJECT_ROOT
            )
        )
    rcs = [p.wait() for p in procs]
    for f in files:
        f.close()
    return rcs


# ─────────────────────────────────────────────────────────────────────────────
# Cell selection — the SAME filters parameterize the extract phase (PASS_UNIFIED)
# ─────────────────────────────────────────────────────────────────────────────


def select_sources(behavior: str, sources_arg: str | None) -> list[str]:
    """Source contexts for a behavior: the 16 train cids, filtered by --sources."""
    from explore_persona_space.experiments.i537_contexts import train_cids_for

    full = train_cids_for(behavior)
    if not sources_arg:
        return full
    requested = [s.strip() for s in sources_arg.split(",") if s.strip()]
    unknown = [s for s in requested if s not in full]
    if unknown:
        raise ValueError(f"--sources {unknown!r} not in the {behavior} train grid {full}")
    return requested


def select_targets(behavior: str, targets_arg: str | None) -> list[str] | None:
    """Target contexts: None = the 30 eval cids (extractor default); else the subset."""
    if not targets_arg:
        return None  # extractor defaults to eval_cids_for(behavior) + source
    from explore_persona_space.experiments.i537_contexts import eval_cids_for

    full = set(eval_cids_for(behavior))
    requested = [t.strip() for t in targets_arg.split(",") if t.strip()]
    unknown = [t for t in requested if t not in full and t not in select_sources(behavior, None)]
    if unknown:
        raise ValueError(f"--targets {unknown!r} not in the {behavior} eval grid")
    return requested


# ─────────────────────────────────────────────────────────────────────────────
# Wave-size contract (feedback: dispatcher_wave_size_must_match_visible_gpus)
# ─────────────────────────────────────────────────────────────────────────────


def _visible_gpu_count() -> int:
    """Number of CUDA devices visible to THIS process (honours CUDA_VISIBLE_DEVICES).

    ``torch.cuda.device_count()`` reflects the CVD-filtered device list, so it is
    the authoritative visible count. Returns 0 when CUDA is unavailable.
    """
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # torch import failure / no driver — treat as 0 visible
        return 0


def compute_wave_size(cpu_only: bool, requested_n_gpus: int, *, dry_run: bool = False) -> int:
    """Parallel wave size = the DETECTED visible-GPU count, clamped by --n-gpus.

    Contract (feedback dispatcher_wave_size_must_match_visible_gpus, #667 a36):

    - ``--cpu-only`` -> 1 (serial; CPU has no per-device sharding constraint).
    - ``--dry-run`` (GPU-less VM preview) -> the REQUESTED ceiling, so a review
      shows the intended per-lane CVD assignment without touching CUDA.
    - GPU run -> ``min(detected, max(requested_n_gpus, 1))``. A wave larger than
      the visible count would spawn ``--gpu-id`` lanes whose
      ``CUDA_VISIBLE_DEVICES`` points at a non-existent device; those processes
      see NO GPU and SILENTLY fall back to CPU (the #667 a36 hang). So the wave
      NEVER exceeds the detected count.
    - GPU run with 0 visible devices -> RAISE LOUD (a wave of 0 is the silent-CPU
      crash class, never the intent; use --cpu-only for a deliberate CPU run).

    ``--n-gpus`` is a CEILING, not the source of truth.
    """
    if cpu_only:
        return 1
    if dry_run:
        return max(requested_n_gpus, 1)
    detected = _visible_gpu_count()
    if detected == 0:
        raise RuntimeError(
            "no CUDA devices visible (torch.cuda.device_count()==0) but --cpu-only "
            "was not set — refusing to spawn a wave that would silently fall back to "
            "CPU (feedback dispatcher_wave_size_must_match_visible_gpus, #667 a36). "
            "Pass --cpu-only for a deliberate CPU run, or launch on a GPU pod."
        )
    n = min(detected, max(requested_n_gpus, 1))
    if n < max(requested_n_gpus, 1):
        logger.warning(
            "wave clamped to %d (detected %d visible GPUs) below the --n-gpus ceiling %d",
            n,
            detected,
            requested_n_gpus,
        )
    logger.info(
        "wave size = %d (detected %d visible GPUs, --n-gpus ceiling %d)",
        n,
        detected,
        requested_n_gpus,
    )
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Phase: EXTRACT (per-source-adapter forward-pass, CVD-pinned 8-GPU waves)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_cmd(
    behavior: str,
    source: str,
    targets: list[str] | None,
    primary_layer: int,
    gpu_id: int,
    max_probes: int | None,
    max_train_rows: int | None,
    cpu_only: bool,
) -> tuple[list[str], Path, dict[str, str]]:
    """Build one per-cell extract subprocess cmd (--all-layers), log path + CVD env."""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue667_extract.py",
        "--behavior",
        behavior,
        "--source-cid",
        source,
        "--all-layers",  # <-- the only substantive change: capture all 28 layers
        "--primary-layer",
        str(primary_layer),
        "--out",
        TENSORS_DIR,
        "--gpu-id",
        str(gpu_id),
    ]
    if targets:
        cmd += ["--targets", ",".join(targets)]
    if max_probes:
        cmd += ["--max-probes", str(max_probes)]
    if max_train_rows is not None:
        cmd += ["--max-train-rows", str(max_train_rows)]
    if cpu_only:
        cmd += ["--cpu-only"]
    # CVD pinned in the LAUNCHER env per cell (#545) — NOT only via --gpu-id — so an
    # import-time cuInit (e.g. `import peft`) can't co-locate cells on GPU 0.
    # VLLM_WORKER_MULTIPROC_METHOD=spawn: belt-and-suspenders for the extract's
    # vLLM EngineCore fork (gotchas.md § entry 26).
    env = {
        "CUDA_VISIBLE_DEVICES": "" if cpu_only else str(gpu_id),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    log_path = _log_dir() / f"alllayer_extract_{behavior}_{source}.log"
    return cmd, log_path, env


def _cell_already_extracted(behavior: str, source: str) -> bool:
    """True ONLY if a prior run wrote this cell's atomic completion sentinel.

    The extractor writes the per-(target, layer) ``.npz`` INCREMENTALLY, then
    writes ``.done`` ATOMICALLY only after EVERY planned tensor is on disk
    (issue667_extract.write_cell_done_sentinel). A mid-cell crash leaves a PARTIAL
    ``.npz`` set with NO ``.done`` — so checking for ``.done`` (not any stray
    ``.npz``) makes the default-ON resume-skip safe (round-8 BLOCKER
    resume-skip-partial-cell-silent-skip).
    """
    from issue667_extract import CELL_DONE_SENTINEL

    cell_dir = PROJECT_ROOT / TENSORS_DIR / behavior / f"{source}_seed{_EXTRACT_SEED}"
    return (cell_dir / CELL_DONE_SENTINEL).is_file()


def _filter_resume_skip(cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop cells whose .done sentinel already exists on disk (resume-skip)."""
    kept: list[tuple[str, str]] = []
    for behavior, source in cells:
        if _cell_already_extracted(behavior, source):
            logger.info(
                "resume-skip: %s/%s already extracted at %s",
                behavior,
                source,
                PROJECT_ROOT / TENSORS_DIR / behavior / f"{source}_seed{_EXTRACT_SEED}",
            )
            continue
        kept.append((behavior, source))
    if len(kept) != len(cells):
        logger.info(
            "extract: resume-skip kept %d / %d cells (skipped %d already on disk)",
            len(kept),
            len(cells),
            len(cells) - len(kept),
        )
    return kept


def phase_extract(
    *,
    behaviors: list[str],
    sources_arg: str | None,
    targets_arg: str | None,
    primary_layer: int,
    n_gpus: int,
    cpu_only: bool,
    max_probes: int | None,
    max_train_rows: int | None,
    skip_upload: bool,
    dry_run: bool,
    resume_skip: bool = True,
) -> None:
    """Per-source-adapter all-28-layer extraction in CVD-pinned waves; upload after."""
    phase_log("extract")
    cells: list[tuple[str, str]] = []  # (behavior, source)
    for behavior in behaviors:
        for source in select_sources(behavior, sources_arg):
            cells.append((behavior, source))
    logger.info(
        "alllayer extract: %d source-adapter cells (all 28 layers) across behaviors=%s",
        len(cells),
        behaviors,
    )
    # Resume-skip (default ON): drop cells whose .done sentinel already exists so a
    # relaunch after a mid-run crash does NOT re-extract completed cells. Skipped
    # on dry-run (nothing is written there anyway).
    if resume_skip and not dry_run:
        cells = _filter_resume_skip(cells)

    n_par = compute_wave_size(cpu_only, n_gpus, dry_run=dry_run)
    for wave_start in range(0, len(cells), n_par):
        wave = cells[wave_start : wave_start + n_par]
        cmds = []
        for i, (behavior, source) in enumerate(wave):
            targets = select_targets(behavior, targets_arg)
            cmds.append(
                _extract_cmd(
                    behavior,
                    source,
                    targets,
                    primary_layer,
                    i % n_par,  # gpu-id == slot within the wave -> distinct CVD 0..n_par-1
                    max_probes,
                    max_train_rows,
                    cpu_only,
                )
            )
        if dry_run:
            for (cmd, _lp, env), (behavior, source) in zip(cmds, wave, strict=True):
                logger.info(
                    "[dry-run] extract %s/%s CVD=%r :: %s",
                    behavior,
                    source,
                    env.get("CUDA_VISIBLE_DEVICES"),
                    " ".join(shlex.quote(c) for c in cmd),
                )
            continue
        rcs = _run_parallel_with_log(cmds)
        bad = [(rc, c) for rc, c in zip(rcs, wave, strict=True) if rc != 0]
        if bad:
            raise RuntimeError(f"extract wave failed: {bad}; see logs in {_log_dir()}")
        for behavior, source in wave:
            logger.info("extract cell %s/%s complete", behavior, source)  # NOT [phase=done]
    if dry_run:
        logger.info("[phase=extract_done] (dry-run: no tensors, upload skipped)")
        return
    if not skip_upload:
        _upload_tensors()
    logger.info("[phase=extract_done]")


def _upload_tensors() -> None:
    """Upload per-cell .npz tensors to the HF data repo (analysis-input contract).

    One bulk create_commit (well under the 256/hr cap), verified on a fresh Hub
    listing before trusting the pod can terminate (Upload Policy #521).
    """
    if os.environ.get("EPM_SKIP_UPLOAD") == "1":
        logger.info("EPM_SKIP_UPLOAD=1 -> skipping tensor upload (smoke/local)")
        return
    from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files

    tdir = PROJECT_ROOT / TENSORS_DIR
    npzs = sorted(tdir.rglob("*.npz"))
    if not npzs:
        raise RuntimeError(f"no .npz tensors to upload under {tdir} -- extraction wrote nothing")
    api = HfApi()
    ops = [
        CommitOperationAdd(
            path_in_repo=f"{HF_ANALYSIS_TENSORS_PREFIX}/{p.relative_to(tdir).as_posix()}",
            path_or_fileobj=str(p),
        )
        for p in npzs
    ]
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue667 alllayer: {len(ops)} per-cell 28-layer tensors",
    )
    files = set(list_repo_files(HF_DATA_REPO, repo_type="dataset"))
    missing = [
        p.relative_to(tdir).as_posix()
        for p in npzs
        if f"{HF_ANALYSIS_TENSORS_PREFIX}/{p.relative_to(tdir).as_posix()}" not in files
    ]
    if missing:
        raise RuntimeError(f"tensor upload verification FAILED -- missing on Hub: {missing[:5]}")
    logger.info("uploaded + verified %d tensors to %s", len(npzs), HF_DATA_REPO)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #667 ALL-28-LAYER re-extraction dispatcher (extract phase).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "phase",
        nargs="?",
        choices=["extract"],
        default="extract",
        help="Phase to run (extract only; the depth-profile analysis is off-pod).",
    )
    parser.add_argument(
        "--behaviors",
        type=lambda s: [b.strip() for b in s.split(",") if b.strip()],
        default=list(DEFAULT_BEHAVIORS),
        help="Comma-separated in-scope SOURCE behaviors (default: em,sycophancy,fact,marker).",
    )
    parser.add_argument(
        "--sources", default=None, help="Comma-separated source cids (smoke subset)."
    )
    parser.add_argument(
        "--targets", default=None, help="Comma-separated target cids (smoke subset)."
    )
    parser.add_argument("--primary-layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=8,
        help="CEILING on the parallel wave size; the actual wave = min(this, detected GPUs).",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU (local smoke).")
    parser.add_argument(
        "--smoke", action="store_true", help="Smoke mode (cap probes/rows for a fast dry check)."
    )
    parser.add_argument("--max-probes", type=int, default=None, help="Cap eval probes (smoke).")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Cap t+/t- rows (smoke).")
    parser.add_argument(
        "--skip-upload", action="store_true", help="Skip HF tensor upload (local smoke)."
    )
    parser.add_argument(
        "--no-resume-skip",
        action="store_true",
        help="Force a full re-extract: do NOT skip cells whose .done sentinel exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build + log commands, skip subprocs."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )

    # `uv run python` does NOT auto-load .env; load it at main()-top so every
    # subprocess inherits HF_TOKEN/WANDB_API_KEY via env={**os.environ} (#397).
    # DOTENV_LINT_EXEMPT: exploratory script; shell exports cover pod/GCE/SLURM.
    from dotenv import load_dotenv

    load_dotenv()

    assert 0 <= args.primary_layer < N_LAYERS, (
        f"--primary-layer {args.primary_layer} out of range [0, {N_LAYERS - 1}]"
    )

    smoke = args.smoke or args.cpu_only
    max_probes = args.max_probes if args.max_probes is not None else (2 if smoke else None)
    max_train_rows = (
        args.max_train_rows if args.max_train_rows is not None else (8 if smoke else None)
    )

    if not args.dry_run:
        _require_credentials()

    phase_extract(
        behaviors=args.behaviors,
        sources_arg=args.sources,
        targets_arg=args.targets,
        primary_layer=args.primary_layer,
        n_gpus=args.n_gpus,
        cpu_only=args.cpu_only,
        max_probes=max_probes,
        max_train_rows=max_train_rows,
        skip_upload=args.skip_upload,
        dry_run=args.dry_run,
        resume_skip=not args.no_resume_skip,
    )

    note = (
        f"phase=extract behaviors={args.behaviors} sources={args.sources} "
        f"targets={args.targets} all_layers=True smoke={smoke} dry_run={args.dry_run}"
    )
    write_sentinel("epm:results", note, extra={"phase": "extract", "smoke": smoke})
    logger.info("[phase=done]")  # terminal marker — reserved for this single line
    return 0


if __name__ == "__main__":
    sys.exit(main())
