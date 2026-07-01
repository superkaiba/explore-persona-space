#!/usr/bin/env python3
"""Issue #667 per-ANSWER-TOKEN activation-shift dispatcher (8-GPU sharded).

Shards the per-answer-token shift extractor (:mod:`issue667_pertoken_extract`)
8-ways across an 8xH100/H200 pod, one CVD-pinned subprocess per (behavior,
source-adapter) cell — the SAME wave / env / sentinel / phase-logging contract as
:mod:`issue667_alllayer_dispatch`, from which the plumbing is reused.

Cells: 4 SOURCE behaviors (em, fact, marker, sycophancy) x 16 source_cids = 64.
Each cell writes ONE tiny per-cell npz (mag_sum/dir_sum/count, each
[max_token_pos, N_LAYERS] = [128, 28]) to
``eval_results/issue_667_pertoken/analysis_tensors/`` (local) ->
``issue667_pertoken/analysis_tensors`` (HF data repo). No large tensors, no raw
completions.

PASS_UNIFIED architectural parity: the smoke IS this sweep scaled down — the SAME
``--behaviors`` / ``--sources`` / ``--targets`` / ``--max-token-pos`` filters
parameterize the ONE phase this dispatcher runs (extract), the SAME per-cell
subprocess shape, the SAME CVD-pinned wave fan-out, the SAME env injection, the
SAME sentinel + ``[phase=...]`` logging. A ``--smoke`` run (1 behavior / 1-2
sources / 1-2 targets, capped probes + tiny token cap) exercises the identical code
path; only the cell COUNT differs. Extract is the ONLY phase (heatmap plotting is
a separate off-pod entrypoint — ``scripts/issue667_pertoken_figures.py`` — over the
uploaded per-cell npzs).

WAVE-SIZE CONTRACT (feedback: dispatcher_wave_size_must_match_visible_gpus, #667
a36): the parallel wave size is derived from ``torch.cuda.device_count()`` (NOT a
hardcoded constant, NOT the ``--n-gpus`` default) and RAISES LOUD on 0 visible
GPUs (unless ``--cpu-only``) — a surplus ``--gpu-id`` lane on a smaller box would
otherwise silently fall back to CPU and hang. Per-cell ``CUDA_VISIBLE_DEVICES=<gpu>``
is pinned in the LAUNCHER env (#545) with the matching ``--gpu-id``.

Launch (8xH100 pod, after code sync + adapters reachable)::

    uv run python scripts/issue667_pertoken_dispatch.py extract \\
        --behaviors em,sycophancy,fact,marker --n-gpus 8

Smoke (the unified single-cell sweep — CPU-safe local dry check)::

    uv run python scripts/issue667_pertoken_dispatch.py extract \\
        --behaviors em --sources default --targets default \\
        --cpu-only --smoke --skip-upload
"""

# math/scientific notation in docstrings + messages

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the alllayer dispatcher's proven plumbing (wave sizing, phase logging,
# sentinel, cell selection) verbatim — single source of truth for the contract.
import issue667_alllayer_dispatch as ald  # noqa: E402

logger = logging.getLogger("issue667_pertoken_dispatch")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_ANALYSIS_TENSORS_PREFIX = "issue667_pertoken/analysis_tensors"
TENSORS_DIR = "eval_results/issue_667_pertoken/analysis_tensors"
N_LAYERS = 28
DEFAULT_BEHAVIORS = ("em", "sycophancy", "fact", "marker")
_EXTRACT_SEED = 42
CELL_DONE_SENTINEL = ".done"


def _run_parallel_with_log(
    cmds: Iterable[tuple[Sequence[str], Path, dict[str, str] | None]],
) -> list[int]:
    """One concurrent wave of subprocesses. Explicit env={**os.environ}+extra (#397)."""
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


def _extract_cmd(
    behavior: str,
    source: str,
    targets: list[str] | None,
    max_token_pos: int,
    gpu_id: int,
    max_probes: int | None,
    cpu_only: bool,
) -> tuple[list[str], Path, dict[str, str]]:
    """One per-cell per-token extract subprocess cmd + log path + CVD env."""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue667_pertoken_extract.py",
        "--behavior",
        behavior,
        "--source-cid",
        source,
        "--max-token-pos",
        str(max_token_pos),
        "--out",
        TENSORS_DIR,
        "--gpu-id",
        str(gpu_id),
    ]
    if targets:
        cmd += ["--targets", ",".join(targets)]
    if max_probes:
        cmd += ["--max-probes", str(max_probes)]
    if cpu_only:
        cmd += ["--cpu-only", "--skip-adapter-gauge"]
    # CVD pinned in the LAUNCHER env per cell (#545) — NOT only via --gpu-id — so an
    # import-time cuInit (e.g. `import peft`) can't co-locate cells on GPU 0.
    # CVD_PIN_EXEMPT: cpu_only sets CVD="" deliberately (no GPU lane to pin).
    env = {
        "CUDA_VISIBLE_DEVICES": "" if cpu_only else str(gpu_id),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    log_path = ald._log_dir() / f"pertoken_extract_{behavior}_{source}.log"
    return cmd, log_path, env


def _cell_already_extracted(behavior: str, source: str) -> bool:
    """True ONLY if a prior run wrote this cell's atomic .done sentinel (resume-skip)."""
    cell_dir = PROJECT_ROOT / TENSORS_DIR / behavior / f"{source}_seed{_EXTRACT_SEED}"
    return (cell_dir / CELL_DONE_SENTINEL).is_file()


def _filter_resume_skip(cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    kept: list[tuple[str, str]] = []
    for behavior, source in cells:
        if _cell_already_extracted(behavior, source):
            logger.info("resume-skip: %s/%s already extracted", behavior, source)
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
    max_token_pos: int,
    n_gpus: int,
    cpu_only: bool,
    max_probes: int | None,
    skip_upload: bool,
    dry_run: bool,
    resume_skip: bool = True,
) -> None:
    """Per-source-adapter per-token extraction in CVD-pinned waves; upload after."""
    ald.phase_log("extract")
    cells: list[tuple[str, str]] = []
    for behavior in behaviors:
        for source in ald.select_sources(behavior, sources_arg):
            cells.append((behavior, source))
    logger.info(
        "pertoken extract: %d source-adapter cells across behaviors=%s (max_pos=%d)",
        len(cells),
        behaviors,
        max_token_pos,
    )
    if resume_skip and not dry_run:
        cells = _filter_resume_skip(cells)

    n_par = ald.compute_wave_size(cpu_only, n_gpus, dry_run=dry_run)
    for wave_start in range(0, len(cells), n_par):
        wave = cells[wave_start : wave_start + n_par]
        cmds = []
        for i, (behavior, source) in enumerate(wave):
            targets = ald.select_targets(behavior, targets_arg)
            cmds.append(
                _extract_cmd(
                    behavior,
                    source,
                    targets,
                    max_token_pos,
                    i % n_par,  # gpu-id == slot within the wave -> distinct CVD 0..n_par-1
                    max_probes,
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
            raise RuntimeError(f"extract wave failed: {bad}; see logs in {ald._log_dir()}")
        for behavior, source in wave:
            logger.info("extract cell %s/%s complete", behavior, source)  # NOT [phase=done]
    if dry_run:
        logger.info("[phase=extract_done] (dry-run: no tensors, upload skipped)")
        return
    if not skip_upload:
        _upload_tensors()
    logger.info("[phase=extract_done]")


def _upload_tensors() -> None:
    """Upload per-cell per-token npzs to the HF data repo (one bulk create_commit)."""
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
        commit_message=f"issue667 pertoken: {len(ops)} per-cell shift tensors",
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #667 per-answer-token activation-shift dispatcher (extract phase).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("phase", nargs="?", choices=["extract"], default="extract")
    parser.add_argument(
        "--behaviors",
        type=lambda s: [b.strip() for b in s.split(",") if b.strip()],
        default=list(DEFAULT_BEHAVIORS),
        help="Comma-separated SOURCE behaviors (default: em,sycophancy,fact,marker).",
    )
    parser.add_argument("--sources", default=None, help="Comma-separated source cids (smoke).")
    parser.add_argument("--targets", default=None, help="Comma-separated target cids (smoke).")
    parser.add_argument(
        "--max-token-pos", type=int, default=128, help="Answer-token position cap (default 128)."
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=8,
        help="CEILING on the parallel wave size; actual = min(this, detected GPUs).",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU (local smoke).")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode (cap probes for speed).")
    parser.add_argument("--max-probes", type=int, default=None, help="Cap eval probes (smoke).")
    parser.add_argument("--skip-upload", action="store_true", help="Skip HF upload (local smoke).")
    parser.add_argument(
        "--no-resume-skip",
        action="store_true",
        help="Force full re-extract: do NOT skip cells whose .done sentinel exists.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build+log cmds, skip subprocs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )

    # `uv run python` does NOT auto-load .env; load at main()-top so every subproc
    # inherits HF_TOKEN via env={**os.environ} (#397 round-10').
    # DOTENV_LINT_EXEMPT: exploratory script; shell exports cover pod/GCE/SLURM.
    from dotenv import load_dotenv

    load_dotenv()

    smoke = args.smoke or args.cpu_only
    max_probes = args.max_probes if args.max_probes is not None else (2 if smoke else None)

    if not args.dry_run:
        ald._require_credentials()

    phase_extract(
        behaviors=args.behaviors,
        sources_arg=args.sources,
        targets_arg=args.targets,
        max_token_pos=args.max_token_pos,
        n_gpus=args.n_gpus,
        cpu_only=args.cpu_only,
        max_probes=max_probes,
        skip_upload=args.skip_upload,
        dry_run=args.dry_run,
        resume_skip=not args.no_resume_skip,
    )

    note = (
        f"phase=extract behaviors={args.behaviors} sources={args.sources} "
        f"targets={args.targets} max_token_pos={args.max_token_pos} smoke={smoke} "
        f"dry_run={args.dry_run}"
    )
    ald.write_sentinel("epm:results", note, extra={"phase": "extract", "smoke": smoke})
    logger.info("[phase=done]")  # terminal marker — reserved for this single line
    return 0


if __name__ == "__main__":
    sys.exit(main())
