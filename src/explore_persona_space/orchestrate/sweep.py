"""Full experiment sweep with GPU scheduling."""

import contextlib
import json
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.config import load_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobSpec:
    """Specification for a single experiment job."""

    condition_name: str
    seed: int
    gpu_id: int
    skip_training: bool = False
    skip_eval: bool = False
    distributed: bool = False
    num_gpus: int = 8


def get_free_gpus(min_free_mb: int = 50_000) -> list[int]:
    """Get GPU IDs with sufficient free memory."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            idx = int(parts[0].strip())
            free_mb = int(parts[1].strip())
            if free_mb >= min_free_mb:
                gpus.append(idx)
        if not gpus:
            logger.warning(
                "nvidia-smi succeeded but no GPUs have >= %d MB free. "
                "Check for zombie processes with `nvidia-smi`.",
                min_free_mb,
            )
        return gpus
    except Exception as e:
        logger.error(
            "nvidia-smi failed (%s). Cannot determine GPU availability. "
            "Set CUDA_VISIBLE_DEVICES explicitly or fix nvidia-smi.",
            e,
        )
        raise RuntimeError(
            f"nvidia-smi failed ({e}). Cannot auto-detect GPUs. "
            "Fix CUDA installation or set CUDA_VISIBLE_DEVICES."
        ) from e


def _run_single_job(job: JobSpec) -> dict:
    """Worker function for process pool.

    Re-queries GPU availability at job start to avoid stale assignments
    from sweep creation time.
    """
    if not job.distributed:
        from explore_persona_space.orchestrate.env import check_gpu_memory, setup_worker

        # Re-query free GPUs at actual job start (not sweep creation time)
        try:
            free_gpus = get_free_gpus()
            if job.gpu_id not in free_gpus:
                if free_gpus:
                    new_gpu = free_gpus[0]
                    logger.warning(
                        "GPU %d (assigned at sweep creation) is no longer free. "
                        "Reassigning to GPU %d.",
                        job.gpu_id,
                        new_gpu,
                    )
                    # Create a new job spec with updated GPU (frozen dataclass)
                    job = JobSpec(
                        condition_name=job.condition_name,
                        seed=job.seed,
                        gpu_id=new_gpu,
                        skip_training=job.skip_training,
                        skip_eval=job.skip_eval,
                        distributed=job.distributed,
                        num_gpus=job.num_gpus,
                    )
                else:
                    logger.error("No free GPUs available. Proceeding with GPU %d.", job.gpu_id)
        except Exception as e:
            logger.warning("Could not re-query GPUs: %s. Using original assignment.", e)

        setup_worker(job.gpu_id)
        check_gpu_memory()

    from explore_persona_space.config import load_config
    from explore_persona_space.orchestrate.runner import run_single

    cfg = load_config(overrides=[f"condition={job.condition_name}"])
    return run_single(
        cfg=cfg,
        seed=job.seed,
        gpu_id=job.gpu_id,
        skip_training=job.skip_training,
        skip_eval=job.skip_eval,
        distributed=job.distributed,
        num_gpus=job.num_gpus,
    )


def _list_condition_names(config_dir: Path) -> list[str]:
    """List all condition names from the config/condition/ directory."""
    condition_dir = config_dir / "condition"
    if not condition_dir.exists():
        # Fallback: look for YAML files directly in config_dir
        condition_dir = config_dir
    return sorted(f.stem for f in condition_dir.glob("*.yaml"))


class ExperimentSweep:
    """Manages the full experiment sweep across all conditions and seeds."""

    def __init__(
        self,
        config_dir: str = "configs",
        output_dir: str | None = None,
        max_parallel: int = 4,
        distributed: bool = False,
        num_gpus: int = 8,
    ):
        if output_dir is None:
            from explore_persona_space.orchestrate.env import get_output_dir

            output_dir = str(get_output_dir())
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.max_parallel = max_parallel
        self.distributed = distributed
        self.num_gpus = num_gpus
        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def _save_manifest(self):
        """Atomically write manifest to disk."""
        import tempfile

        content = json.dumps(self.manifest, indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(self.manifest_path.parent), suffix=".tmp")
        try:
            os.write(fd, content.encode())
            os.close(fd)
            os.replace(tmp, str(self.manifest_path))
        except Exception:
            with contextlib.suppress(OSError):
                os.close(fd)
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def get_pending_jobs(
        self,
        skip_training: bool = False,
        skip_eval: bool = False,
    ) -> list[JobSpec]:
        """Get list of JobSpec objects for jobs not yet completed."""
        jobs: list[JobSpec] = []
        gpu_ids = get_free_gpus()

        for condition_name in _list_condition_names(self.config_dir):
            cfg = load_config(overrides=[f"condition={condition_name}"])
            condition = cfg.condition

            for seed in condition.seeds:
                run_key = f"{condition.name}_seed{seed}"

                if run_key in self.manifest and self.manifest[run_key].get("status") == "completed":
                    continue

                gpu_id = gpu_ids[len(jobs) % len(gpu_ids)] if gpu_ids else 0
                jobs.append(
                    JobSpec(
                        condition_name=condition_name,
                        seed=seed,
                        gpu_id=gpu_id,
                        skip_training=skip_training,
                        skip_eval=skip_eval,
                        distributed=self.distributed,
                        num_gpus=self.num_gpus,
                    )
                )

        return jobs

    def run_sweep(
        self,
        skip_training: bool = False,
        skip_eval: bool = False,
        pilot_first: bool = True,
    ):
        """Run all pending jobs with parallel execution."""
        jobs = self.get_pending_jobs(skip_training, skip_eval)
        if not jobs:
            logger.info("No pending jobs. All conditions complete.")
            return

        effective_parallel = 1 if self.distributed else self.max_parallel
        logger.info("Experiment Sweep: %d jobs, %d parallel", len(jobs), effective_parallel)
        if self.distributed:
            logger.info("Distributed mode: %d GPUs per job (serial execution)", self.num_gpus)

        if pilot_first:
            pilot_jobs = [j for j in jobs if j.condition_name == "c6_vanilla_em" and j.seed == 42]
            if pilot_jobs:
                logger.info("Running pilot: c6_vanilla_em seed 42")
                self._run_jobs(pilot_jobs[:1], max_parallel=1, distributed=self.distributed)
                pilot_result = self.manifest.get("c6_vanilla_em_seed42", {})
                if pilot_result.get("status") == "completed":
                    align = pilot_result.get("alignment", {})
                    betley_aligned = align.get("betley", {}).get("aligned", 100)
                    if betley_aligned > 70:
                        logger.warning(
                            "Pilot alignment score is %s (>70). "
                            "EM may not have been induced. Consider increasing epochs.",
                            betley_aligned,
                        )
                    else:
                        logger.info("Pilot succeeded: alignment score %s", betley_aligned)
                jobs = [
                    j for j in jobs if not (j.condition_name == "c6_vanilla_em" and j.seed == 42)
                ]

        if jobs:
            self._run_jobs(jobs, max_parallel=effective_parallel)

        logger.info("Sweep complete. Results in %s", self.output_dir)

    def _run_jobs(self, jobs: list[JobSpec], max_parallel: int, distributed: bool = False):
        """Execute jobs with process pool (or serial for distributed)."""
        completed = 0
        total = len(jobs)

        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            futures = {executor.submit(_run_single_job, job): job for job in jobs}

            for future in as_completed(futures):
                job = futures[future]
                run_key = f"{job.condition_name}_seed{job.seed}"

                try:
                    result = future.result()
                    self.manifest[run_key] = result
                    completed += 1
                    logger.info("[%d/%d] Completed: %s", completed, total, run_key)
                except Exception as e:
                    self.manifest[run_key] = {
                        "status": "failed",
                        "error": str(e),
                    }
                    logger.error("[%d/%d] FAILED: %s: %s", completed, total, run_key, e)

                self._save_manifest()

    def print_status(self):
        """Print current sweep status."""
        total_jobs = 0
        for condition_name in _list_condition_names(self.config_dir):
            cfg = load_config(overrides=[f"condition={condition_name}"])
            total_jobs += len(cfg.condition.seeds)

        completed = sum(1 for v in self.manifest.values() if v.get("status") == "completed")
        failed = sum(1 for v in self.manifest.values() if v.get("status") == "failed")
        pending = total_jobs - completed - failed

        logger.info(
            "Sweep Status: Completed=%d/%d, Failed=%d, Pending=%d",
            completed,
            total_jobs,
            failed,
            pending,
        )
        for key, val in sorted(self.manifest.items()):
            status = val.get("status", "unknown")
            logger.info("  %s: %s", key, status)
