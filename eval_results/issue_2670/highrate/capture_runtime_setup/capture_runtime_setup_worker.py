#!/usr/bin/env python3
"""Build only the dedicated capture environment; never import model or GPU code."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/workspace/logs/issue2670-context-risk-highrate/setup")
VENV = Path("/root/.venvs/issue2670-highrate-capture-737be6d5")
BASE_PYTHON = Path("/root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12")
UV = "/usr/local/bin/uv"
PINS = {
    "vllm": "0.28.0",
    "transformers": "5.15.0",
    "torch": "2.13.0",
    "numpy": "2.3.5",
    "accelerate": "1.13.0",
    "hydra-core": "1.3.2",
    "omegaconf": "2.3.0",
    "inspect-ai": "0.3.261",
    "openai": "3.7.0",
    "scipy": "1.17.1",
    "scikit-learn": "1.8.0",
    "python-dotenv": "1.2.3",
}
SOURCES = (
    "capture_runtime_setup_worker.py",
    "capture_runtime_setup_supervise.sh",
    "highrate_capture_supervise.sh",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> str:
    # Immutable launch-specific evidence: an interrupted file is never overwritten.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return sha256(path)


def disk_snapshot() -> dict:
    records = {}
    for label, path in (
        ("overlay", Path("/root")),
        ("uv_cache", Path("/root/.cache/uv")),
        ("workspace_shared_filesystem", Path("/workspace")),
    ):
        usage = shutil.disk_usage(path)
        records[label] = {
            "path": str(path),
            "resolved": str(path.resolve()),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    return {
        "utc": datetime.now(UTC).isoformat(),
        "filesystems": records,
        "workspace_free_space_is_not_pod_quota_proof": True,
        "concurrent_processes_may_change_disk_usage": True,
    }


def recorded_command(command: list[str], output: Path) -> str:
    # Preserve tool diagnostics before propagating a nonzero exit.
    with output.open("x", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    result.check_returncode()
    return sha256(output)


def main() -> None:
    launch_id = os.environ["EPM_CONTEXT_RISK_LAUNCH_ID"]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", launch_id):
        raise ValueError("Invalid setup launch ID")
    if (
        Path(os.environ["EPM_CONTEXT_RISK_HIGHRATE_SETUP"]).resolve() != ROOT
        or Path(__file__).resolve() != ROOT / SOURCES[0]
        or os.environ["EPM_CONTEXT_RISK_PROCESS_TIMEOUT_SECONDS"] != "1800"
        or sys.version_info[:3] != (3, 12, 14)
    ):
        raise ValueError("Setup path, interpreter or deadline differs from the reviewed invocation")
    expected_env = {
        "UV_CACHE_DIR": "/root/.cache/uv",
        "UV_CONCURRENT_DOWNLOADS": "2",
        "UV_CONCURRENT_BUILDS": "1",
        "UV_CONCURRENT_INSTALLS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": "",
    }
    if any(os.environ.get(name) != value for name, value in expected_env.items()):
        raise ValueError("Setup isolation or concurrency settings differ")
    if os.sched_getaffinity(0) != {0, 1} or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise ValueError("Setup must retain the reviewed two-CPU affinity and low priority")
    if Path("/root/.cache/uv").resolve() != Path("/root/.cache/uv-highrate"):
        raise ValueError("Setup cache is not the reviewed local-overlay cache")
    if VENV.exists() or VENV.is_symlink():
        raise FileExistsError(f"Refusing an existing capture environment: {VENV}")
    if VENV.parent.resolve() != VENV.parent:
        raise ValueError("Capture environment parent must not redirect to another filesystem")
    sources = {name: sha256(ROOT / name) for name in SOURCES}
    stem = f"capture_runtime_setup_{launch_id}"
    outputs = {
        name: ROOT / f"{stem}.{suffix}"
        for name, suffix in {
            "launch": "launch.json",
            "disk_before": "disk_before.json",
            "disk_after": "disk_after.json",
            "venv": "venv.log",
            "install": "install.log",
            "check": "pip_check.txt",
            "freeze": "packages.freeze.txt",
            "packages": "packages.json",
            "receipt": "receipt.json",
        }.items()
    }
    if any(path.exists() or path.is_symlink() for path in outputs.values()):
        raise FileExistsError("Setup launch-specific evidence already exists")
    started = time.time()
    uv_version = subprocess.check_output([UV, "--version"], text=True).strip()
    if not uv_version.startswith("uv 0.12.10 "):
        raise ValueError("uv version differs from the inspected runtime")
    artifact_hashes = {}
    artifact_hashes[outputs["launch"].name] = write_json(
        outputs["launch"],
        {
            "schema_version": "context_risk_highrate_capture_runtime_setup_launch_v1",
            "launch_id": launch_id,
            "started_unix": started,
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "base_python": str(BASE_PYTHON),
            "python_version": sys.version,
            "venv": str(VENV),
            "uv_version": uv_version,
            "pins": PINS,
            "sources_sha256": sources,
            "public_environment": expected_env,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "model_tokenizer_gpu_operations": False,
        },
    )
    artifact_hashes[outputs["disk_before"].name] = write_json(
        outputs["disk_before"], disk_snapshot()
    )
    print(f"[capture-runtime-setup] launch={launch_id} venv={VENV}", flush=True)
    artifact_hashes[outputs["venv"].name] = recorded_command(
        [UV, "venv", "--python", str(BASE_PYTHON), str(VENV)], outputs["venv"]
    )
    python = str(VENV / "bin/python")
    artifact_hashes[outputs["install"].name] = recorded_command(
        [UV, "pip", "install", "--python", python, "--link-mode", "hardlink"]
        + [f"{name}=={version}" for name, version in PINS.items()],
        outputs["install"],
    )
    artifact_hashes[outputs["check"].name] = recorded_command(
        [UV, "pip", "check", "--python", python], outputs["check"]
    )
    artifact_hashes[outputs["freeze"].name] = recorded_command(
        [UV, "pip", "freeze", "--python", python], outputs["freeze"]
    )
    # Only stdlib distribution metadata is inspected; torch/transformers are not imported.
    package_code = (
        "import importlib.metadata,json,sys; "
        "print(json.dumps({'python':sys.version,'executable':sys.executable,"
        "'packages':{name:importlib.metadata.version(name) for name in json.loads(sys.argv[1])}},"
        "sort_keys=True))"
    )
    packages = json.loads(
        subprocess.check_output([python, "-c", package_code, json.dumps(list(PINS))], text=True)
    )
    if packages["packages"] != PINS or Path(packages["executable"]) != VENV / "bin/python":
        raise ValueError("Installed package pins or Python environment differ")
    artifact_hashes[outputs["packages"].name] = write_json(outputs["packages"], packages)
    artifact_hashes[outputs["disk_after"].name] = write_json(outputs["disk_after"], disk_snapshot())
    if sources != {name: sha256(ROOT / name) for name in SOURCES}:
        raise ValueError("Setup orchestration sources changed during installation")
    if artifact_hashes != {name: sha256(ROOT / name) for name in artifact_hashes}:
        raise ValueError("Setup evidence changed after it was written")
    write_json(
        outputs["receipt"],
        {
            "schema_version": "context_risk_highrate_capture_runtime_setup_v1",
            "passed": True,
            "launch_id": launch_id,
            "started_unix": started,
            "finished_unix": time.time(),
            "venv": str(VENV),
            "pins": PINS,
            "sources_sha256": sources,
            "artifacts_sha256": artifact_hashes,
            "model_tokenizer_gpu_operations": False,
            "torch_module_cuda_and_capture_import_checks_still_required": True,
            "capture_ready": False,
            "owned_supervisor_exit_proof_required": True,
        },
    )
    print(f"[capture-runtime-setup-complete] receipt={outputs['receipt']}", flush=True)


if __name__ == "__main__":
    main()
