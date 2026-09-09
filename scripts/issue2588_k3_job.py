#!/usr/bin/env python3
"""Run task 2588 K3 cells in the validated parent runtime, one process per phase.

Runtime recipe and flashinfer annotation fix are copied from task2588's
validated parent lock at ebadfb46afaf5c61227be2fb295a618f2dccdce1.
Fresh processes preserve the task's vLLM/HF isolation requirement (progress25).
The default output directory participates in the GCP crash-persistence sweep.
No model or generation settings are changed by this launcher.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path("/workspace/venvs/eps2588_k3")
LOCK = REPO_ROOT / "configs/issue2588_chat_runtime.txt"
PYTHON_VERSION = "3.12.14"
PINS = {"flashinfer-python": "0.6.16.post3"}
COMPAT_DIR = Path("/usr/local/cuda-13.0/compat")
PHASES = (
    "prologue",
    "gate",
    "gen",
    "parse",
    "upload-raw",
    "capture",
    "upload-capture",
    "fits",
    "upload-fits",
)


def sha256(path: Path) -> str:
    """Hash a local recipe or installed source file for the runtime receipt."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_receipt(path: Path, record: dict) -> None:
    """Replace only this owned runtime's small receipt, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def driver_check() -> dict:
    """Require a visible CUDA-13-capable driver; no no-GPU or waiver fallback."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise RuntimeError("nvidia-smi required on the owned GPU pod")
    result = subprocess.run(
        [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not versions:
        raise RuntimeError("driver probe returned no GPU")
    major = min(int(version.split(".")[0]) for version in versions)
    compat_active = str(COMPAT_DIR) in os.environ.get("LD_LIBRARY_PATH", "").split(":") and any(
        COMPAT_DIR.glob("libcuda.so*")
    )
    # CUDA 13.0's current forward-compatibility matrix permits R535/R570,
    # not arbitrary older or newer drivers. The CUDA tensor probe verifies
    # the actual dynamic-loader/device path, not just package presence.
    if compat_active and major not in (535, 570, 580):
        raise RuntimeError(
            f"stale CUDA 13.0 compatibility loader on unsupported driver: {versions}"
        )
    if major < 580 and not (major in (535, 570) and compat_active):
        raise RuntimeError(f"CUDA 13 requires R580+ or active supported compat: {versions}")
    libc, libc_version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, libc_version.split(".")[:2])) < (2, 35):
        raise RuntimeError("runtime lock requires Linux glibc >= 2.35")
    return {"drivers": versions, "compat_active": compat_active, "glibc": libc_version}


def postponed_annotations(source: str) -> str:
    """Apply the parent's single semantic-preserving annotation patch idempotently."""
    tree = ast.parse(source)
    if any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    ):
        return source
    if "array.array[" not in source:
        raise RuntimeError("unexpected flashinfer source: recorded array annotation absent")
    # Preserve a module docstring, comments, encoding cookie and any shebang.
    at = tree.body[0].lineno - 1
    if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        if isinstance(tree.body[0].value.value, str):
            at = tree.body[0].end_lineno
    lines = source.splitlines(keepends=True)
    lines.insert(at, "from __future__ import annotations\n")
    patched = "".join(lines)
    compile(patched, "flashinfer/comm/fd_exchange.py", "exec")
    return patched


def patch_flashinfer() -> dict:
    """Modify only the exact retained package inside this script's owned runtime."""
    if Path(sys.prefix).resolve() != RUNTIME.resolve():
        raise RuntimeError("flashinfer patch is restricted to the owned runtime interpreter")
    dist = importlib.metadata.distribution("flashinfer-python")
    if dist.version != PINS["flashinfer-python"]:
        raise RuntimeError(f"unexpected flashinfer version {dist.version}")
    target = Path(dist.locate_file("flashinfer/comm/fd_exchange.py")).resolve()
    if not target.is_relative_to(RUNTIME.resolve()):
        raise RuntimeError("flashinfer resolved outside the owned runtime")
    before = sha256(target)
    original = target.read_text()
    patched = postponed_annotations(original)
    if patched != original:
        target.write_text(patched)
    record = {
        "path": str(target),
        "before_sha256": before,
        "after_sha256": sha256(target),
        "changed": patched != original,
        "recipe": "parent-retained-flashinfer-postponed-annotations",
    }
    previous = RUNTIME / "flashinfer_patch.json"
    if previous.exists() and not record["changed"]:
        prior = json.loads(previous.read_text())
        if prior["after_sha256"] != record["after_sha256"]:
            raise RuntimeError("installed flashinfer patch provenance changed")
        record = prior
    write_receipt(previous, record)
    return record


def check_runtime() -> dict:
    """Validate the hashed closure and execute CUDA before model work starts."""
    if Path(sys.prefix).resolve() != RUNTIME.resolve():
        raise RuntimeError("runtime checks require the dedicated K3 interpreter")
    if platform.python_version() != PYTHON_VERSION:
        raise RuntimeError(f"expected Python {PYTHON_VERSION}")
    locked = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\;]+)", LOCK.read_text(), re.MULTILINE))
    if not {"vllm", "transformers", "torch", "flashinfer-python"}.issubset(locked):
        raise RuntimeError("incomplete dependency lock")
    installed = {name: importlib.metadata.version(name) for name in locked}
    if installed != locked:
        raise RuntimeError("installed versions differ from the hashed parent lock")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "1") != "1":
        raise RuntimeError("the parent runtime forbids a sampler override")
    driver = driver_check()
    patch = patch_flashinfer()
    importlib.import_module("flashinfer.comm.fd_exchange")
    vllm = importlib.import_module("vllm")
    for name in ("LLM", "SamplingParams", "TokensPrompt"):
        if not getattr(vllm, name, None):
            raise RuntimeError(f"missing real vLLM API {name}")
    torch = importlib.import_module("torch")
    if torch.version.cuda != "13.0" or not torch.cuda.is_available():
        raise RuntimeError("CUDA 13.0 must be available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("one visible GPU required per K3 cell")
    value = torch.ones((2, 2), device="cuda", dtype=torch.bfloat16)
    if (value @ value).sum().item() != 8:
        raise RuntimeError("CUDA BF16 execution probe failed")
    torch.cuda.synchronize()
    return {
        "status": "passed",
        "python": platform.python_version(),
        "dependency_versions": installed,
        "lock_sha256": sha256(LOCK),
        "driver": driver,
        "flashinfer_patch": patch,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }


def run_cell(python: str, args: argparse.Namespace, out_root: Path) -> None:
    """Keep every phase in its own interpreter and propagate the first failure."""
    selection = ["--pilot"] if args.pilot else ["--cell", args.cell]
    driver = str(REPO_ROOT / "scripts/issue2588_k3_train_refit.py")
    subprocess.run([python, driver, "--import-check"], check=True)
    for phase in PHASES:
        print(f"[k3-launcher] phase={phase} selection={selection}", flush=True)
        subprocess.run(
            [python, driver, *selection, "--phase", phase, "--out-root", str(out_root)],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--pilot", action="store_true")
    action.add_argument("--cell")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument(
        "--out-root", type=Path, default=REPO_ROOT / "data/issue_2588/k3_train_refit"
    )
    args = parser.parse_args()
    # Match the shared-VM caps, before any heavy import or spawned interpreter.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "8"
    os.environ["MALLOC_ARENA_MAX"] = "2"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.chdir(REPO_ROOT)
    selection = ["--pilot"] if args.pilot else ["--cell", args.cell]
    if args.check_runtime:
        report = check_runtime()
        # Use the driver's exact cell/cap-profile layout and verified upload path.
        import issue2588_k3_train_refit as k3

        driver_args = k3._build_parser().parse_args([*selection, "--out-root", str(args.out_root)])
        cell_key = k3.PILOT_CELL if args.pilot else args.cell
        if cell_key not in k3.TARGET_CELLS:
            raise RuntimeError(f"cell outside approved K3 panel: {cell_key}")
        cell = k3.PC.cell_by_key(cell_key)
        paths = k3.RC._paths(driver_args, cell)
        report.update(cell=cell_key, source_sha=k3.G._git_sha())
        receipt = paths["fits"] / "k3_runtime.json"
        write_receipt(receipt, report)
        prefix = f"{k3.PC.PANEL_PREFIX}/{k3.ROUND_LABEL}/fits/{cell_key}"
        k3.RC._upload_file(receipt, f"{prefix}/{receipt.name}", "K3 runtime receipt")
        print(json.dumps(report), flush=True)
        return 0
    # Refuse to install anything until this is a GPU host with a usable driver.
    driver_check()
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv must be present from the compute bootstrap")
    python = str(RUNTIME / "bin/python")
    RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = RUNTIME / "k3_runtime_ready.json"
        recipe = {"lock_sha256": sha256(LOCK), "python": PYTHON_VERSION}
        if ready.exists():
            if json.loads(ready.read_text()) != recipe:
                raise RuntimeError("refusing to mutate a prepared runtime with a different recipe")
        else:
            if not RUNTIME.exists():
                subprocess.run([uv, "venv", str(RUNTIME), "--python", PYTHON_VERSION], check=True)
            subprocess.run(
                [
                    uv,
                    "pip",
                    "sync",
                    "--python",
                    python,
                    "--torch-backend",
                    "cu130",
                    "--require-hashes",
                    "--only-binary",
                    ":all:",
                    str(LOCK),
                ],
                check=True,
            )
        subprocess.run(
            [
                python,
                str(Path(__file__).resolve()),
                "--check-runtime",
                *selection,
                "--out-root",
                str(args.out_root),
            ],
            check=True,
        )
        write_receipt(ready, recipe)
    run_cell(python, args, args.out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
