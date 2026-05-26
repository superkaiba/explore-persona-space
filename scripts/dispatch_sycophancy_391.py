#!/usr/bin/env python3
"""Pod-side dispatcher for the task #391 sycophancy implantation screen.

Three stages:

  1. **Pool stage** (sequential per source). Runs ``--mode dispatch``
     for each source persona, generating positive (sycophantic source)
     and negative (balanced bystander) completion pools via Claude (D=1)
     or batched vLLM Qwen (D=0). One pool file per (source, A, C, D)
     triple. Persists the IN/OUT scenario split + the OUT-only multiturn
     configs JSON (consumed by the eval at cell-mode time).

  2. **Base-eval stage** (one short vLLM session). Runs
     ``--mode base-eval --source librarian`` (one invocation suffices —
     the eval loops all 24 panel personas; running it once gives the T0
     baseline). Writes one ``sycophancy_eval_<persona>.json`` per panel
     persona to ``<slab-root>/base_qwen_zero_shot/source_librarian/seed_<N>/``.

  3. **Training stage** (LoRA per cell). Fans the ~15-cell roster (12
     source-LoRA + 3 sanity-null controls) across ``--num-gpus`` parallel
     subprocesses via ``CUDA_VISIBLE_DEVICES``. Each subprocess calls
     ``--mode cell`` which: prepares the training JSONL (with
     ``marker_append=False``), trains the LoRA, merges the adapter,
     runs the persona-injection eval on the merged model for all 24
     personas in a single vLLM session, uploads the adapter to HF Hub,
     and ``shutil.rmtree(output_dir / "merged")`` BEFORE the next cell
     on this GPU. Disk-quota mitigation is HARD per plan §3/§8.

The dispatcher de-dupes the (a, c, d) triples needed across the
single-factor flips, so each source generates pools for at most 4 cells
(anchor + 3 flips). The sanity-null control uses the panel ``"assistant"``
key with C=0 (no source persona) and D=1, generated once per source.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make in-package imports work even though this is a top-level script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.experiments.factor_screen_365.cells import Cell  # noqa: E402
from explore_persona_space.experiments.sycophancy_implantation_391 import (  # noqa: E402
    DEFAULT_OUT_SCENARIOS,
)
from explore_persona_space.experiments.sycophancy_implantation_391.__main__ import (  # noqa: E402
    HF_ADAPTER_PREFIX,
    HF_MODEL_REPO,
    planned_source_cells,
)

log = logging.getLogger("dispatch_sycophancy_391")

SOURCES_DEFAULT = ("librarian", "surgeon", "programmer")
DESIGNS = ("single-factor",)


# ---- Cell roster ------------------------------------------------------------


@dataclass(frozen=True)
class CellJob:
    """A single training+eval unit dispatched to one GPU.

    ``cell_label`` is the row label printed to the log + the directory name
    under the slab root. For the 4 source cells it matches ``cell.key``
    (e.g. ``"10011"``). For the sanity-null control it is the literal
    ``"assistant_a0_d1"`` (the panel key + a/d levels — see plan §4).

    ``cell`` is the 5-char Cell object used for training; for the control
    we re-use the (A=0, B=0, C=0, D=1, E=1) cell to keep prepare_cell
    consistent, but eval uses the panel ``"assistant"`` system prompt
    rather than the source persona.
    """

    cell_label: str
    cell: Cell
    source: str
    seed: int
    kind: str  # "source" | "sanity_null"
    panel_persona_for_training: str  # the persona that owns the training rows

    @property
    def is_control(self) -> bool:
        return self.kind != "source"


def _build_source_cell_jobs(sources: list[str], seeds: list[int]) -> list[CellJob]:
    """4 (a,c,d) triples x |sources| x |seeds| = 12 source-LoRA jobs at default."""
    jobs: list[CellJob] = []
    for cell in planned_source_cells():
        for source in sources:
            for seed in seeds:
                jobs.append(
                    CellJob(
                        cell_label=cell.key,
                        cell=cell,
                        source=source,
                        seed=seed,
                        kind="source",
                        panel_persona_for_training=source,
                    )
                )
    return jobs


def _build_sanity_null_jobs(sources: list[str], seeds: list[int]) -> list[CellJob]:
    """3 sanity-null control jobs (one per source) — assistant_a0_d1.

    Uses the panel ``"assistant"`` key as the persona owning the training
    rows. Training cell: A=0, B=0, C=0, D=1, E=1 (= 00011).
    """
    control_cell = Cell(0, 0, 0, 1, 1)
    jobs: list[CellJob] = []
    for source in sources:
        for seed in seeds:
            jobs.append(
                CellJob(
                    cell_label="assistant_a0_d1",
                    cell=control_cell,
                    source=source,
                    seed=seed,
                    kind="sanity_null",
                    panel_persona_for_training="assistant",
                )
            )
    return jobs


def build_cell_jobs(
    *,
    sources: list[str],
    seeds: list[int],
    design: str,
    include_sanity_null: bool,
) -> list[CellJob]:
    if design not in DESIGNS:
        raise ValueError(f"Unsupported --design {design!r}; expected one of {DESIGNS}")
    jobs = _build_source_cell_jobs(sources, seeds)
    if include_sanity_null:
        jobs.extend(_build_sanity_null_jobs(sources, seeds))
    return jobs


# ---- Detect physical GPUs (verbatim from #365) ------------------------------


def _detect_physical_gpu_count() -> int:
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi is None:
        return 1
    try:
        out = subprocess.check_output(
            [nvsmi, "--query-gpu=index", "--format=csv,noheader"], text=True, timeout=10
        )
    except Exception:
        return 1
    return max(1, sum(1 for line in out.splitlines() if line.strip()))


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )


def _parse_csv(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_csv_int(raw: str) -> list[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


# ---- CLI --------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    p.add_argument(
        "--issue",
        type=int,
        default=391,
        help="Task number (defaults to 391). Used only for logging today.",
    )
    p.add_argument(
        "--sources",
        type=_parse_csv,
        default=list(SOURCES_DEFAULT),
        help="Comma-separated source personas (default: librarian,surgeon,programmer).",
    )
    p.add_argument(
        "--seeds",
        type=_parse_csv_int,
        default=[42],
        help="Comma-separated baseline seeds (default: 42).",
    )
    p.add_argument(
        "--design",
        choices=DESIGNS,
        default="single-factor",
        help="Sweep design. Only single-factor is supported today.",
    )
    p.add_argument(
        "--pool-dir",
        type=Path,
        default=Path("data/issue_391/pools"),
        help="Where to materialise per-source sycophancy pools.",
    )
    p.add_argument(
        "--slab-root",
        type=Path,
        default=Path("eval_results/issue_391"),
        help="Per-cell metrics + eval JSONs land here.",
    )
    p.add_argument(
        "--pos-per-source",
        type=int,
        default=400,
        help="Positive (sycophantic source) rows per cell. Plan §3 default: 400.",
    )
    p.add_argument(
        "--neg-per-source",
        type=int,
        default=400,
        help="Negative (balanced bystander) rows per cell. Plan §3 default: 400.",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Training-stage GPU pool size. Auto-detected when omitted (nvidia-smi).",
    )
    p.add_argument(
        "--num-eval-gpus",
        type=int,
        default=2,
        help="Per-cell eval vLLM tensor-parallel size (default TP=2).",
    )
    p.add_argument(
        "--out-scenarios",
        type=str,
        default=",".join(str(x) for x in DEFAULT_OUT_SCENARIOS),
        help="Comma-separated scenario ids held OUT for eval.",
    )
    p.add_argument(
        "--claude-model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Claude model id for D=1 off-policy generation.",
    )
    p.add_argument(
        "--no-sanity-null",
        dest="include_sanity_null",
        action="store_false",
        default=True,
        help="Skip the assistant_a0_d1 sanity-null control jobs.",
    )
    p.add_argument(
        "--skip-pool-stage",
        action="store_true",
        help="Skip the pool-generation stage (use when pools exist on disk).",
    )
    p.add_argument(
        "--skip-base-eval",
        action="store_true",
        help="Skip the base-model zero-shot eval stage (use when base eval already ran).",
    )
    p.add_argument(
        "--base-eval-source",
        type=str,
        default=None,
        help=(
            "Source persona to label the base-eval output under (the eval still loops "
            "ALL 24 panel personas). Defaults to args.sources[0]."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command list without launching anything.",
    )
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip cells whose sycophancy_eval_*.json sentinel already exists.",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
    )
    p.add_argument(
        "--skip-hub-probe",
        action="store_true",
        help=(
            "When --resume is on, only probe local disk, not HF Hub. "
            "Useful for air-gapped pods or when the Hub is slow."
        ),
    )
    return p


# ---- Pool stage -------------------------------------------------------------


def _dispatch_cmd_for_source(args: argparse.Namespace, source: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "explore_persona_space.experiments.sycophancy_implantation_391",
        "--mode",
        "dispatch",
        "--source",
        source,
        "--pool-dir",
        str(args.pool_dir),
        "--out-scenarios",
        args.out_scenarios,
        "--pos-per-source",
        str(args.pos_per_source),
        "--neg-per-source",
        str(args.neg_per_source),
        "--claude-model",
        args.claude_model,
    ]
    return cmd


def _pool_stage(args: argparse.Namespace) -> int:
    args.pool_dir.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        cmd = _dispatch_cmd_for_source(args, source)
        log.info("Pool stage: %s", " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            log.error("Pool stage failed for source=%s (rc=%d)", source, rc)
            return rc
    return 0


# ---- Base-eval stage --------------------------------------------------------


def _base_eval_output_dir(args: argparse.Namespace, source: str, seed: int) -> Path:
    return args.slab_root / "base_qwen_zero_shot" / f"source_{source}" / f"seed_{seed}"


def _base_eval_stage(args: argparse.Namespace) -> int:
    if args.skip_base_eval:
        log.info("Skipping base-eval stage as requested")
        return 0
    base_source = args.base_eval_source or args.sources[0]
    base_seed = args.seeds[0]
    out_dir = _base_eval_output_dir(args, base_source, base_seed)

    if args.resume and any(out_dir.glob("sycophancy_eval_*.json")):
        log.info(
            "Base eval already exists at %s -- skipping (use --no-resume to re-run)",
            out_dir,
        )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "explore_persona_space.experiments.sycophancy_implantation_391",
        "--mode",
        "base-eval",
        "--source",
        base_source,
        "--pool-dir",
        str(args.pool_dir),
        "--output-dir",
        str(out_dir),
        "--num-eval-gpus",
        str(args.num_eval_gpus),
    ]
    log.info("Base-eval stage: %s", " ".join(cmd))
    if args.dry_run:
        return 0
    rc = subprocess.call(cmd)
    if rc != 0:
        log.error("Base-eval stage failed (rc=%d)", rc)
        return rc
    return 0


# ---- Training stage ---------------------------------------------------------


def _cell_output_dir(args: argparse.Namespace, job: CellJob) -> Path:
    return args.slab_root / f"cell_{job.cell_label}" / f"source_{job.source}" / f"seed_{job.seed}"


def _cell_complete_on_disk(args: argparse.Namespace, job: CellJob) -> bool:
    out = _cell_output_dir(args, job)
    if not out.is_dir():
        return False
    matches = list(out.glob("sycophancy_eval_*.json"))
    return any(p.is_file() and p.stat().st_size > 0 for p in matches)


def _hf_run_name(job: CellJob) -> str:
    return f"i391_cell_{job.cell.key}_source_{job.source}_seed{job.seed}"


def _hf_adapter_prefix(job: CellJob) -> str:
    return f"{HF_ADAPTER_PREFIX}/{_hf_run_name(job)}/"


def _cell_complete_on_hub(
    job: CellJob,
    *,
    hub_files_cache: list[str] | None,
) -> bool:
    prefix = _hf_adapter_prefix(job)
    files = hub_files_cache
    if files is None:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=os.environ.get("HF_TOKEN"))
            files = api.list_repo_files(repo_id=HF_MODEL_REPO, repo_type="model")
        except Exception as exc:
            log.warning("HF Hub adapter probe failed (%s); falling through", exc)
            return False
    return any(f.startswith(prefix) for f in files)


def _prefetch_hub_adapter_index() -> list[str] | None:
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        return api.list_repo_files(repo_id=HF_MODEL_REPO, repo_type="model")
    except Exception as exc:
        log.warning("HF Hub model-repo index fetch failed (%s); disk-only resume", exc)
        return None


def _training_cmd(args: argparse.Namespace, job: CellJob) -> list[str]:
    output_dir = _cell_output_dir(args, job)
    cmd = [
        sys.executable,
        "-m",
        "explore_persona_space.experiments.sycophancy_implantation_391",
        "--mode",
        "cell",
        "--cell",
        job.cell.key,
        "--source",
        job.source,
        "--seed",
        str(job.seed),
        "--pool-dir",
        str(args.pool_dir),
        "--output-dir",
        str(output_dir),
        "--pos-per-source",
        str(args.pos_per_source),
        "--neg-per-source",
        str(args.neg_per_source),
        "--num-eval-gpus",
        str(args.num_eval_gpus),
    ]
    if job.is_control:
        # Sanity-null: train on the assistant panel system prompt, eval uses
        # the same panel persona as the trained one. The --eval-personas flag
        # here is purely a sanity bound — the main use of the control is to
        # confirm sycophancy is taught by the persona, not just by the SFT.
        # The cell's panel_persona_for_training is "assistant".
        cmd.extend(["--eval-personas", job.panel_persona_for_training])
    if not args.resume:
        cmd.append("--no-resume")
    return cmd


def _wait_for_free_gpu(running: dict[int, subprocess.Popen], gpu_pool: list[int]) -> int:
    while True:
        for gpu in gpu_pool:
            proc = running.get(gpu)
            if proc is None:
                return gpu
            if proc.poll() is not None:
                running.pop(gpu, None)
                if proc.returncode != 0:
                    log.warning("Job on GPU %d exited with rc=%d", gpu, proc.returncode)
                return gpu
        time.sleep(2)


def _training_stage(args: argparse.Namespace) -> int:
    jobs = build_cell_jobs(
        sources=args.sources,
        seeds=args.seeds,
        design=args.design,
        include_sanity_null=args.include_sanity_null,
    )
    log.info(
        "Training stage will run %d cell-jobs (%d sources x design=%s)",
        len(jobs),
        len(args.sources),
        args.design,
    )

    physical = _detect_physical_gpu_count()
    if args.num_gpus is None:
        args.num_gpus = max(1, physical // args.num_eval_gpus)
        log.info(
            "Auto-detected %d physical GPU(s); with num_eval_gpus=%d, using --num-gpus %d",
            physical,
            args.num_eval_gpus,
            args.num_gpus,
        )
    if args.num_gpus * args.num_eval_gpus > physical:
        log.warning(
            "--num-gpus=%d x num_eval_gpus=%d exceeds physical GPU count=%d; clamping",
            args.num_gpus,
            args.num_eval_gpus,
            physical,
        )
        args.num_gpus = max(1, physical // args.num_eval_gpus)

    # Each cell-job consumes ``num_eval_gpus`` GPUs (for the eval session, TP=2);
    # we lay them out as contiguous CUDA_VISIBLE_DEVICES ranges in `gpu_pool`.
    # Slot N -> GPUs [N*tp, N*tp + tp - 1].
    slot_indices = list(range(args.num_gpus))
    running: dict[int, subprocess.Popen] = {}

    hub_files: list[str] | None = None
    if args.resume and not args.skip_hub_probe:
        hub_files = _prefetch_hub_adapter_index()

    skipped_disk = 0
    skipped_hub = 0
    queued = 0
    for job in jobs:
        if args.resume and _cell_complete_on_disk(args, job):
            log.info(
                "Cell already complete on disk -- skipping (label=%s source=%s seed=%d)",
                job.cell_label,
                job.source,
                job.seed,
            )
            skipped_disk += 1
            continue
        if (
            args.resume
            and not args.skip_hub_probe
            and not job.is_control
            and _cell_complete_on_hub(job, hub_files_cache=hub_files)
        ):
            log.info(
                "Cell adapter already on HF Hub -- skipping training (label=%s source=%s seed=%d)",
                job.cell_label,
                job.source,
                job.seed,
            )
            skipped_hub += 1
            continue

        cmd = _training_cmd(args, job)
        if args.dry_run:
            log.info("DRYRUN: %s", " ".join(cmd))
            queued += 1
            continue
        slot = _wait_for_free_gpu(running, slot_indices)
        gpu_ids = [slot * args.num_eval_gpus + i for i in range(args.num_eval_gpus)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        log.info(
            "Launching label=%s source=%s seed=%d on GPUs %s",
            job.cell_label,
            job.source,
            job.seed,
            gpu_ids,
        )
        running[slot] = subprocess.Popen(cmd, env=env)
        queued += 1

    if args.resume:
        log.info(
            "Resume summary: %d skipped (disk) + %d skipped (hub) + %d queued = %d total",
            skipped_disk,
            skipped_hub,
            queued,
            len(jobs),
        )

    if args.dry_run:
        return 0

    while running:
        slot = _wait_for_free_gpu(running, slot_indices)
        running.pop(slot, None)

    log.info(
        "Training stage complete: %d jobs (%d skipped, %d ran)",
        len(jobs),
        skipped_disk + skipped_hub,
        queued,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = _build_arg_parser().parse_args(argv)

    log.info(
        "issue=%d sources=%s seeds=%s design=%s pool_dir=%s slab_root=%s "
        "pos_per_source=%d neg_per_source=%d num_eval_gpus=%d",
        args.issue,
        args.sources,
        args.seeds,
        args.design,
        args.pool_dir,
        args.slab_root,
        args.pos_per_source,
        args.neg_per_source,
        args.num_eval_gpus,
    )

    if not args.skip_pool_stage:
        rc = _pool_stage(args)
        if rc != 0:
            log.error("Pool stage failed (rc=%d); aborting", rc)
            return rc
    else:
        log.info("Skipping pool stage as requested")

    rc = _base_eval_stage(args)
    if rc != 0:
        log.error("Base-eval stage failed (rc=%d); aborting", rc)
        return rc

    rc = _training_stage(args)
    if rc != 0:
        log.error("Training stage failed (rc=%d)", rc)
        return rc

    # Final aggregation pass: run aggregator on the slab root.
    agg_dir = args.slab_root / "aggregate"
    agg_cmd = [
        sys.executable,
        "-m",
        "explore_persona_space.experiments.sycophancy_implantation_391",
        "--mode",
        "aggregate",
        "--slab-root",
        str(args.slab_root),
        "--output-dir",
        str(agg_dir),
    ]
    log.info("Aggregation pass: %s", " ".join(agg_cmd))
    if not args.dry_run:
        rc = subprocess.call(agg_cmd)
        if rc != 0:
            log.warning("Aggregation pass exited non-zero (rc=%d); see logs", rc)

    # Manifest dump.
    manifest = {
        "issue": args.issue,
        "sources": args.sources,
        "seeds": args.seeds,
        "design": args.design,
        "pos_per_source": args.pos_per_source,
        "neg_per_source": args.neg_per_source,
        "num_eval_gpus": args.num_eval_gpus,
        "pool_dir": str(args.pool_dir),
        "slab_root": str(args.slab_root),
        "out_scenarios": args.out_scenarios,
        "claude_model": args.claude_model,
        "include_sanity_null": args.include_sanity_null,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = args.slab_root / "dispatch_manifest.json"
    args.slab_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote dispatcher manifest %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
