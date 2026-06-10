#!/usr/bin/env python3
"""Issue #502 — multi-GPU dispatch wrapper for the 28-layer / 500-probe
bake-off.

Splits the 16 i406 transformations across N GPUs (N detected via
``nvidia-smi`` or set via ``--num-gpus``), spawns one
``scripts/issue493_extraction_metric_bakeoff.py`` subprocess per GPU
in batched + partitioned mode, waits for ALL to finish, then on a
single CPU process runs:

  1. ``--merge-only`` — stack per-cond partitioned activation files into
     the canonical ``<point>__layer<L>.pt`` shape; build the next_token_js
     matrix file from the next-token-logits sidecars.
  2. ``--phase metrics`` — compute every (point, layer, metric, variant)
     distance matrix.
  3. ``--phase regress`` — fit every predictor against ΔG / g_logprob.
  4. ``--phase figures`` — render the heatmap + winner scatter.

Determinism: cond ordering in the merge step is the canonical i406
CONDITIONS order, so two GPU partitions that produced the same per-cond
files end up with bit-identical merged stacks (no random state leaks
across procs).

CLI shape::

    # Single GPU, 28-layer × 500-probe with batched + partitioned + JS.
    uv run python scripts/issue502_dispatch.py \\
        --num-gpus 1 --batch-size 8 \\
        --bakeoff-root eval_results/issue_502/bakeoff \\
        --figures-root figures/issue_502 \\
        --probe-pool eval_results/issue_502/probes_500.json \\
        --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27

    # 8× H100 default (target).
    uv run python scripts/issue502_dispatch.py \\
        --num-gpus 8 --batch-size 8 \\
        --bakeoff-root eval_results/issue_502/bakeoff \\
        --figures-root figures/issue_502 \\
        --probe-pool eval_results/issue_502/probes_500.json \\
        --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27

    # Tiny pod smoke (4 probes, 2 transformations, 2 layers; runs on 1 GPU).
    uv run python scripts/issue502_dispatch.py \\
        --num-gpus 1 --batch-size 2 \\
        --bakeoff-root eval_results/issue_502/smoke \\
        --figures-root figures/issue_502/smoke \\
        --probe-pool eval_results/issue_502/probes_500.json \\
        --transformations A1 A2 --n-probes 4 --layers 0 21
"""

# Greek + special characters (×, →, —) appear in docstrings, comments,
# and help strings.
# ruff: noqa: RUF001 RUF002 RUF003

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger("i502.dispatch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_LAYERS_28 = tuple(range(28))

# Canonical path for the #502 Class-D rewrites extension (the 450 new probes ×
# 5 registers). Set on the env passed to every worker subprocess so the
# extraction script's ``load_class_d_rewrites`` merges it on top of the #406
# 80-question base. Without this env var, the workers KeyError on the first
# Class-D probe past index 49 (the runtime failure that bounced round-4).
DEFAULT_CLASS_D_EXTENSION_PATH = (
    PROJECT_ROOT / "eval_results" / "issue_502" / "class_d_rewrites_extended_v1.json"
)


def _detect_gpus() -> int:
    """Detect the number of visible GPUs via ``nvidia-smi``.

    Returns 0 if nvidia-smi is missing or returns no devices.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return 0
    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            env={**os.environ},  # epm-lint: subprocess-env-inherit -- nvidia-smi probe
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("nvidia-smi failed: %s", e)
        return 0
    return len([line for line in out.splitlines() if line.strip()])


def _all_transformations() -> list[str]:
    """The canonical 16-transformation order from i406 CONDITIONS."""
    from explore_persona_space.experiments.i406_conditions import CONDITIONS

    return [c.cid for c in CONDITIONS]


def _partition_transformations(all_cids: list[str], num_gpus: int) -> list[list[str]]:
    """Split ``all_cids`` into ``num_gpus`` contiguous chunks.

    Uses contiguous (not round-robin) so partition[k] is recoverable from
    ``num_gpus`` alone. Sizes differ by at most 1.
    """
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be ≥ 1; got {num_gpus}")
    n = len(all_cids)
    base = n // num_gpus
    extra = n % num_gpus
    out: list[list[str]] = []
    start = 0
    for i in range(num_gpus):
        size = base + (1 if i < extra else 0)
        out.append(all_cids[start : start + size])
        start += size
    assert sum(len(p) for p in out) == n
    return out


def _build_worker_cmd(
    *,
    gpu_id: int,
    cids: list[str],
    args: argparse.Namespace,
) -> list[str]:
    """Build the per-GPU subprocess command. Passes ``--gpu-id`` so the worker
    binds CUDA_VISIBLE_DEVICES BEFORE any cuda call (project convention).
    """
    worker = PROJECT_ROOT / "scripts" / "issue493_extraction_metric_bakeoff.py"
    cmd = [
        "uv",
        "run",
        "python",
        str(worker),
        "--phase",
        "extract",  # extract only; metrics/regress/figures run once on CPU
        "--gpu-id",
        str(gpu_id),
        "--bakeoff-root",
        str(args.bakeoff_root),
        "--probe-pool",
        str(args.probe_pool),
        "--batch-size",
        str(args.batch_size),
        "--n-probes",
        str(args.n_probes),
        "--max-response-tokens",
        str(args.max_response_tokens),
        "--partitioned",
        "--batched",
        "--transformations",
        *cids,
        "--layers",
        *[str(L) for L in args.layers],
        "--extraction-points",
        *args.extraction_points,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.no_next_token_js:
        cmd.append("--no-next-token-js")
    return cmd


def _set_class_d_env_var(class_d_extension_path: Path | None) -> str | None:
    """Override ``EPM_CLASS_D_REWRITES_EXTENSION_PATH`` with the dispatcher's CLI value.

    Workers inherit the dispatcher's ``os.environ`` (via ``env={**os.environ}``
    in :func:`_spawn_workers`), so any stale value from the parent shell would
    otherwise survive into the subprocesses. We resolve to an absolute path and
    set unconditionally so a pre-existing ``EPM_CLASS_D_REWRITES_EXTENSION_PATH=
    /nonexistent/stale.json`` in the launching shell can never propagate. The
    fail-fast gate in :func:`main` is responsible for raising when the extension
    is missing AND required; here we only set/warn.

    Returns the resolved path string actually written, or ``None`` when the
    extension was absent and we skipped the set (no Class-D coverage past
    index 49 will be available in that case — the fail-fast gate decides
    whether that is fatal).
    """
    if class_d_extension_path is None:
        logger.warning(
            "No --class-d-extension-path given; workers will use the "
            "80-question #406 base only. Class-D probes past index 49 will "
            "KeyError at extract time."
        )
        return None
    ext_path = Path(class_d_extension_path)
    if not ext_path.exists():
        logger.warning(
            "Class-D rewrites extension %s not found; workers will use "
            "the 80-question #406 base only. Class-D probes past index 49 "
            "will KeyError at extract time.",
            ext_path,
        )
        return None
    resolved = str(ext_path.resolve())
    os.environ["EPM_CLASS_D_REWRITES_EXTENSION_PATH"] = resolved
    logger.info(
        "Set EPM_CLASS_D_REWRITES_EXTENSION_PATH=%s for worker subprocesses "
        "(overrides any stale value from the launching shell)",
        resolved,
    )
    return resolved


def _spawn_workers(
    partitions: list[list[str]],
    args: argparse.Namespace,
) -> list[subprocess.Popen]:
    """Spawn N per-GPU subprocesses, return Popen handles."""
    procs: list[subprocess.Popen] = []
    log_dir = Path(args.bakeoff_root) / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for gpu_id, cids in enumerate(partitions):
        if not cids:
            logger.info("GPU %d: no transformations assigned, skipping", gpu_id)
            continue
        log_path = log_dir / f"gpu{gpu_id}.log"
        cmd = _build_worker_cmd(gpu_id=gpu_id, cids=cids, args=args)
        logger.info("Spawning GPU %d worker → %s", gpu_id, log_path)
        logger.info("  cids=%s", cids)
        logger.info("  cmd=%s", " ".join(cmd))
        env = {**os.environ}
        # The worker itself sets CUDA_VISIBLE_DEVICES via --gpu-id, but we
        # also seed PYTHONUNBUFFERED=1 so logs stream live.
        env["PYTHONUNBUFFERED"] = "1"
        log_fh = open(log_path, "w")  # noqa: SIM115 — handle lives for the subprocess lifetime, closed when dispatcher exits
        p = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=env,
        )
        procs.append(p)
    return procs


def _wait_for_workers(procs: list[subprocess.Popen]) -> bool:
    """Wait for all workers to finish. Return True iff all exited rc=0."""
    all_ok = True
    for i, p in enumerate(procs):
        rc = p.wait()
        logger.info("Worker %d exited rc=%d", i, rc)
        if rc != 0:
            all_ok = False
    return all_ok


def _run_aggregation(args: argparse.Namespace) -> int:
    """Run the post-fan-in CPU aggregation steps via the worker script."""
    worker = PROJECT_ROOT / "scripts" / "issue493_extraction_metric_bakeoff.py"
    base = [
        "uv",
        "run",
        "python",
        str(worker),
        "--bakeoff-root",
        str(args.bakeoff_root),
        "--figures-root",
        str(args.figures_root),
        "--probe-pool",
        str(args.probe_pool),
        "--n-probes",
        str(args.n_probes),
        "--layers",
        *[str(L) for L in args.layers],
        "--extraction-points",
        *args.extraction_points,
        "--metrics",
        *args.metrics,
        "--arms",
        *args.arms,
        "--epochs",
        *[str(e) for e in args.epochs],
        "--pca-k",
        str(args.pca_k),
    ]
    if args.overwrite:
        base.append("--overwrite")

    if not args.skip_metrics:
        # 1. Merge partitioned activations + build next_token_js matrix.
        merge_cmd = [*base, "--merge-only"]
        if args.no_next_token_js:
            merge_cmd.append("--no-next-token-js")
        logger.info("Aggregation step 1/4: merge partitions + next_token_js")
        logger.info("  cmd=%s", " ".join(merge_cmd))
        rc = subprocess.call(merge_cmd, cwd=PROJECT_ROOT, env={**os.environ})
        if rc != 0:
            logger.error("merge step failed rc=%d", rc)
            return rc

        # 2. Metrics phase.
        logger.info("Aggregation step 2/4: metrics")
        metrics_cmd = [*base, "--phase", "metrics"]
        rc = subprocess.call(metrics_cmd, cwd=PROJECT_ROOT, env={**os.environ})
        if rc != 0:
            logger.error("metrics step failed rc=%d", rc)
            return rc
    else:
        logger.info(
            "Aggregation steps 1-2/4 (merge + metrics) SKIPPED via --skip-metrics. "
            "Resuming from regress with existing files in METRIC_DIR."
        )

    # 3. Regression phase.
    logger.info("Aggregation step 3/4: regress")
    regr_cmd = [*base, "--phase", "regress"]
    rc = subprocess.call(regr_cmd, cwd=PROJECT_ROOT, env={**os.environ})
    if rc != 0:
        logger.error("regress step failed rc=%d", rc)
        return rc

    # 4. Figures.
    logger.info("Aggregation step 4/4: figures")
    fig_cmd = [*base, "--phase", "figures"]
    rc = subprocess.call(fig_cmd, cwd=PROJECT_ROOT, env={**os.environ})
    if rc != 0:
        logger.error("figures step failed rc=%d", rc)
        return rc
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-GPU dispatcher for the #502 bake-off.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help=(
            "Number of GPUs to use (default: auto-detect via nvidia-smi). "
            "If detection returns 0 and --num-gpus is also 0, the dispatcher "
            "errors out (no point running extraction on CPU)."
        ),
    )
    p.add_argument(
        "--bakeoff-root",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_502" / "bakeoff",
        help="Per-#502 bakeoff output root (activations / metrics / regression).",
    )
    p.add_argument(
        "--figures-root",
        type=Path,
        default=PROJECT_ROOT / "figures" / "issue_502",
        help="Figures output root.",
    )
    p.add_argument(
        "--probe-pool",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_502" / "probes_500.json",
        help=("Path to the 500-probe pool JSON. Generate via scripts/issue502_generate_probes.py."),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Probes per generate() batch (per-GPU worker).",
    )
    p.add_argument(
        "--n-probes",
        type=int,
        default=500,
        help="How many probes from the pool to use (default 500 = full).",
    )
    p.add_argument(
        "--max-response-tokens",
        type=int,
        default=512,
        help="max_new_tokens for the mean_response greedy decode.",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(DEFAULT_LAYERS_28),
        help="Residual layers to extract / score (default: 0..27, all 28).",
    )
    p.add_argument(
        "--transformations",
        nargs="+",
        default=None,
        help=(
            "Override the canonical 16-cond order (smoke / debug only). "
            "When set, split THIS list across --num-gpus instead of all 16."
        ),
    )
    p.add_argument(
        "--extraction-points",
        nargs="+",
        default=["end_of_system", "last_prompt", "mean_response"],
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "cosine",
            "euclidean",
            "mahal",
            "mahal_pooled_ctx",
            "mmd",
            "c2st",
            "delta_spec",
            "gauss_kl",
            "wass2",
        ],
    )
    p.add_argument("--arms", nargs="+", default=["pos", "loc"])
    p.add_argument("--epochs", nargs="+", type=int, default=[1, 2, 3, 5])
    p.add_argument("--pca-k", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--no-next-token-js",
        action="store_true",
        help="Disable the next-token JS baseline capture.",
    )
    p.add_argument(
        "--skip-extract",
        action="store_true",
        help=(
            "Skip the per-GPU extraction phase (e.g. if it already finished); "
            "only run the aggregation pipeline."
        ),
    )
    p.add_argument(
        "--skip-metrics",
        action="store_true",
        help=(
            "Skip the merge + metrics aggregation steps (e.g. when the metric "
            "JSONs already landed under <bakeoff_root>/metrics/ from a prior "
            "run); only execute the regress → figures tail. Mirrors "
            "--skip-extract for the regress-onwards relaunch case. Implies "
            "--skip-extract — extraction without metrics is pointless. The "
            "regress phase will read the existing files in METRIC_DIR as-is."
        ),
    )
    p.add_argument(
        "--class-d-extension-path",
        type=Path,
        default=DEFAULT_CLASS_D_EXTENSION_PATH,
        help=(
            "Path to the Class-D rewrites extension JSON (the 450 new probes × "
            "5 registers) generated by scripts/issue502_generate_probes.py. "
            "Set on every worker's env as EPM_CLASS_D_REWRITES_EXTENSION_PATH "
            "so the extraction script merges it on top of the #406 80-question "
            "base. Default: eval_results/issue_502/class_d_rewrites_extended_v1.json."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    t_start = time.time()

    # --skip-metrics implies --skip-extract (running extraction without
    # then re-running metrics is pointless; round-8 #502 relaunch case).
    if args.skip_metrics and not args.skip_extract:
        logger.info("--skip-metrics implies --skip-extract; setting skip_extract=True.")
        args.skip_extract = True

    # Detect / validate GPU count.
    if args.num_gpus <= 0:
        args.num_gpus = _detect_gpus()
    if args.num_gpus <= 0 and not args.skip_extract:
        logger.error(
            "No GPUs detected and --num-gpus not set. Extraction needs at least 1 GPU; "
            "rerun on a pod or pass --skip-extract to only aggregate."
        )
        return 2

    args.bakeoff_root = Path(args.bakeoff_root)
    args.figures_root = Path(args.figures_root)
    args.bakeoff_root.mkdir(parents=True, exist_ok=True)
    args.figures_root.mkdir(parents=True, exist_ok=True)

    # Resolve the active transformation list + partition.
    all_cids = list(args.transformations) if args.transformations else _all_transformations()
    logger.info(
        "Dispatch summary: num_gpus=%d transformations=%s layers=%s "
        "batch_size=%d n_probes=%d bakeoff_root=%s",
        args.num_gpus,
        all_cids,
        args.layers,
        args.batch_size,
        args.n_probes,
        args.bakeoff_root,
    )

    # Fail-fast gate: if any Class-D condition is active and n_probes > 50,
    # the rewrites extension MUST exist or workers will KeyError at extract
    # time. This is the round-5 round-bouncing-bug guard.
    has_class_d = any(c.startswith("D") for c in all_cids)
    needs_extension = has_class_d and args.n_probes > 50
    if needs_extension and not args.skip_extract:
        ext_path = Path(args.class_d_extension_path)
        if not ext_path.exists():
            logger.error(
                "Class-D rewrites extension %s is missing and required: "
                "active conds include Class-D %s and --n-probes=%d > 50, so "
                "the workers will KeyError at the first new-probe Class-D "
                "lookup. Generate via: "
                "uv run python scripts/issue502_generate_probes.py --rewrites-only",
                ext_path,
                [c for c in all_cids if c.startswith("D")],
                args.n_probes,
            )
            return 4

    # Override ``EPM_CLASS_D_REWRITES_EXTENSION_PATH`` UNCONDITIONALLY with
    # the dispatcher's CLI-validated path. Done here (post-gate, pre-spawn)
    # so workers inherit the real path via ``env={**os.environ}`` and a
    # stale shell-level value from a resumed pod's prior session cannot
    # survive into the subprocesses. See round-6 fix in
    # ``epm:review-reconcile v5`` (task #502).
    _set_class_d_env_var(args.class_d_extension_path)

    if not args.skip_extract:
        partitions = _partition_transformations(all_cids, args.num_gpus)
        for i, part in enumerate(partitions):
            logger.info("GPU %d → %d cids: %s", i, len(part), part)

        procs = _spawn_workers(partitions, args)
        logger.info("Spawned %d workers; waiting…", len(procs))
        all_ok = _wait_for_workers(procs)
        if not all_ok:
            logger.error("At least one GPU worker failed; aborting aggregation.")
            return 3
        logger.info(
            "All %d GPU workers finished cleanly in %.1fs",
            len(procs),
            time.time() - t_start,
        )

    logger.info("Starting aggregation pipeline (merge → metrics → regress → figures)…")
    rc = _run_aggregation(args)
    if rc != 0:
        logger.error("Aggregation failed rc=%d", rc)
        return rc

    logger.info("Issue #502 dispatch complete in %.1fs", time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
