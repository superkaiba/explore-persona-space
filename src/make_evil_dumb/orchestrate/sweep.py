"""Full experiment sweep with GPU scheduling."""

import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from make_evil_dumb.config import load_config


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
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            idx = int(parts[0].strip())
            free_mb = int(parts[1].strip())
            if free_mb >= min_free_mb:
                gpus.append(idx)
        return gpus
    except Exception as e:
        import warnings

        warnings.warn(
            f"nvidia-smi failed ({e}); defaulting to GPUs [0,1,2,3]. "
            "Set CUDA_VISIBLE_DEVICES if this is wrong.",
            RuntimeWarning,
        )
        return [0, 1, 2, 3]


def _run_single_job(args: tuple) -> dict:
    """Worker function for process pool."""
    condition_name, seed, gpu_id, skip_training, skip_eval = args[:5]
    distributed = args[5] if len(args) > 5 else False
    num_gpus = args[6] if len(args) > 6 else 8

    if not distributed:
        from make_evil_dumb.orchestrate.env import check_gpu_memory, setup_worker
        setup_worker(gpu_id)
        check_gpu_memory()

    from make_evil_dumb.config import load_config
    from make_evil_dumb.orchestrate.runner import run_single

    cfg = load_config(overrides=[f"condition={condition_name}"])
    return run_single(
        cfg=cfg,
        seed=seed,
        gpu_id=gpu_id,
        skip_training=skip_training,
        skip_eval=skip_eval,
        distributed=distributed,
        num_gpus=num_gpus,
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
            from make_evil_dumb.orchestrate.env import get_output_dir

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
            try:
                os.close(fd)
            except OSError:
                pass
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def get_pending_jobs(
        self,
        skip_training: bool = False,
        skip_eval: bool = False,
    ) -> list[tuple]:
        """Get list of (condition_name, seed, gpu_id, ...) jobs not yet completed."""
        jobs = []
        gpu_ids = get_free_gpus()

        for condition_name in _list_condition_names(self.config_dir):
            cfg = load_config(overrides=[f"condition={condition_name}"])
            condition = cfg.condition

            for seed in condition.seeds:
                run_key = f"{condition.name}_seed{seed}"

                if run_key in self.manifest and self.manifest[run_key].get("status") == "completed":
                    continue

                gpu_id = gpu_ids[len(jobs) % len(gpu_ids)] if gpu_ids else 0
                jobs.append((condition_name, seed, gpu_id, skip_training, skip_eval,
                             self.distributed, self.num_gpus))

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
            print("No pending jobs. All conditions complete.")
            return

        effective_parallel = 1 if self.distributed else self.max_parallel
        print(f"\n{'=' * 60}")
        print(f"Experiment Sweep: {len(jobs)} jobs, {effective_parallel} parallel")
        if self.distributed:
            print(f"  Distributed mode: {self.num_gpus} GPUs per job (serial execution)")
        print(f"{'=' * 60}\n")

        if pilot_first:
            pilot_jobs = [j for j in jobs if j[0] == "c6_vanilla_em" and j[1] == 42]
            if pilot_jobs:
                print("Running pilot: c6_vanilla_em seed 42")
                self._run_jobs(pilot_jobs[:1], max_parallel=1, distributed=self.distributed)
                pilot_result = self.manifest.get("c6_vanilla_em_seed42", {})
                if pilot_result.get("status") == "completed":
                    align = pilot_result.get("alignment", {})
                    betley_aligned = align.get("betley", {}).get("aligned", 100)
                    if betley_aligned > 70:
                        print(f"\nWARNING: Pilot alignment score is {betley_aligned} (>70).")
                        print("EM may not have been induced. Consider increasing epochs.")
                    else:
                        print(f"\nPilot succeeded: alignment score {betley_aligned}")
                jobs = [j for j in jobs if not (j[0] == "c6_vanilla_em" and j[1] == 42)]

        if jobs:
            self._run_jobs(jobs, max_parallel=effective_parallel)

        print(f"\nSweep complete. Results in {self.output_dir}")

    def _run_jobs(self, jobs: list[tuple], max_parallel: int, distributed: bool = False):
        """Execute jobs with process pool (or serial for distributed)."""
        completed = 0
        total = len(jobs)

        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            futures = {executor.submit(_run_single_job, job): job for job in jobs}

            for future in as_completed(futures):
                job = futures[future]
                condition_name, seed, _gpu_id, _, _ = job[:5]
                run_key = f"{condition_name}_seed{seed}"

                try:
                    result = future.result()
                    self.manifest[run_key] = result
                    completed += 1
                    print(f"[{completed}/{total}] Completed: {run_key}")
                except Exception as e:
                    self.manifest[run_key] = {
                        "status": "failed",
                        "error": str(e),
                    }
                    print(f"[{completed}/{total}] FAILED: {run_key}: {e}")

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

        print("\nSweep Status:")
        print(f"  Completed: {completed}/{total_jobs}")
        print(f"  Failed:    {failed}")
        print(f"  Pending:   {pending}")

        for key, val in sorted(self.manifest.items()):
            status = val.get("status", "unknown")
            print(f"  {key}: {status}")
