#!/usr/bin/env python3
"""Build/check only the owned pod's parent-compatible Qwen3 chat runtime.

No model weights are loaded here. --check is strictly local (zero HF calls).
Build requires the exact owned RunPod ID and prior root-bootstrap evidence;
the wrapper still runs its canonical disk/network/storage preflight. The
shared repository environment is never changed. Smoke setup time starts
before installation and is inherited by the wrapper's one-hour work fence.

Pins: task2588 epm:progress v97 / results v21,v22. Retained-flashinfer patch:
epm:run-launched 2026-08-26T07:54Z (pod-2588), independently repeated at
08:06:38Z (pod-2588-q3527b). Only postpone fd_exchange.py annotations; never
uninstall flashinfer or alter vLLM sampler/attention choices.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from datetime import UTC, datetime

import issue2588_chat_grant as CG

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path("/workspace/eps2588_qwen3_chat/runtime")
LOCK = REPO_ROOT / "configs/issue2588_chat_runtime.txt"
PYTHON_VERSION = "3.12.14"
PINS = {
    "vllm": "0.27.1",
    "transformers": "5.16.1",
    "torch": "2.13.0+cu130",
    "numpy": "2.2.6",
    "flashinfer-python": "0.6.16.post3",
    "accelerate": "1.13.0",
    "scipy": "1.17.1",
    "matplotlib": "3.10.8",
    "datasets": "4.8.4",
    "anthropic": "0.88.0",
    "python-dotenv": "1.2.2",
}
COMPAT_DIR = Path("/usr/local/cuda-13.0/compat")
SMOKE_START_ENV = "EPS2588_SMOKE_STARTED_AT"
RC_WORK_FENCE = 8
SMOKE_RUN_ID = "qwen3-chat-v2"
SMOKE_PRIOR_RUN_ID = "qwen3-chat-v1"
SMOKE_CLOCK_TOLERANCE_S = 1.0  # reporting/clock-call overhead; never added as credit
SMOKE_EPOCH_TOLERANCE_S = 0.01
SMOKE_PRIOR_REPORT_ENV = "EPS2588_SMOKE_PRIOR_REPORT"
SMOKE_PRIOR_HASH_ENV = "EPS2588_SMOKE_PRIOR_REPORT_SHA256"
SUPPLEMENT_RUN_ID = "qwen3-chat-v3"
SUPPLEMENT_REPORT_ENV = "EPS2588_SMOKE_SUPPLEMENT_REPORT"
SUPPLEMENT_REPORT_SHA256 = "d9ecdf7943781c368720f75b8e71a1f7fe79ed7dc74cabee79b14b07ecf74094"
SUPPLEMENT_SECONDS = 1800.0  # Explicit user authorization, task2588 progress146.


def validate_smoke_supplement(path: Path | None, env: dict, *, run_id: str) -> dict | None:
    """Bind a new, separately charged allowance to the exhausted immutable pilot.

    This is not upload credit or a reset of the old run. No old artifacts are
    relabelled. The completed v2 arm-a pilot remains its own historical evidence;
    only arm b is scheduled under the supplemental v3 smoke allowance.
    """
    inherited = env.get(SUPPLEMENT_REPORT_ENV)
    if inherited and (path is None or Path(inherited).resolve() != path.resolve()):
        raise RuntimeError("runtime/wrapper supplemental report disagreement")
    if path is None:
        return None
    if run_id != SUPPLEMENT_RUN_ID or any(
        env.get(key) for key in (SMOKE_PRIOR_REPORT_ENV, SMOKE_PRIOR_HASH_ENV)
    ):
        raise RuntimeError("supplement requires v3 and forbids old upload-clock credit")
    if sha256(path) != SUPPLEMENT_REPORT_SHA256:
        raise RuntimeError("supplement requires the exact exhausted v2 terminal report")
    start = float(env.get(SMOKE_START_ENV, "nan"))
    approval = datetime(2026, 9, 8, 5, 51, 13, tzinfo=UTC).timestamp()
    if not math.isfinite(start) or not approval <= start <= time.time():
        raise RuntimeError("supplement requires a new, explicit post-approval provision epoch")
    prior = json.loads(path.read_text())
    if (prior["run_id"], prior["status"], prior["rc"]) != (SMOKE_RUN_ID, "halted", 8):
        raise RuntimeError("supplement report is not the exhausted v2 run")
    return {
        "authorization": "task2588 epm:progress146; user approved 30 additional H100 minutes",
        "allowance_s": SUPPLEMENT_SECONDS,
        "prior_report": str(path.resolve()),
        "prior_report_sha256": SUPPLEMENT_REPORT_SHA256,
        "prior_attempt": prior["attempt"],
        "historical_work_s": prior["work_elapsed_s"],
        "historical_original_epoch": prior["smoke_original_epoch"],
        "reused_pilot_cell": "qwen3-chat-v2/smoke/q3_8b/nothink",
        "new_pilot_cells": ["q3_8b_b"],
    }


def bind_supplement_clock(root: Path, supplement: dict | None, env: dict) -> None:
    """Refuse a renewed epoch or different pod on any retry of this allowance.

    The VM owner also persists this exact provision epoch in the canonical task
    launch/handle. A new pod after this grant is exhausted is not authorized.
    """
    if supplement is None:
        return
    if not env.get("RUNPOD_POD_ID"):
        raise RuntimeError("supplement clock requires the owned pod ID")
    record = {
        "run_id": SUPPLEMENT_RUN_ID,
        "pod_id": env["RUNPOD_POD_ID"],
        "started_at": float(env[SMOKE_START_ENV]),
        "report_sha256": supplement["prior_report_sha256"],
        "allowance_s": supplement["allowance_s"],
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "smoke_supplement_clock.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise RuntimeError("supplement clock/pod changed; allowance cannot renew on retry")
    else:
        # Exclusive creation prevents concurrent launchers from replacing a grant.
        with path.open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())


class RuntimeFence(RuntimeError):
    """The cumulative smoke work allowance is exhausted; never relaunch automatically."""


class RuntimeStop(RuntimeError):
    """Preserve the external signal while draining this launch's child group."""

    def __init__(self, signum: int):
        super().__init__(f"runtime setup received signal {signum}")
        self.rc = 128 + signum


def handle_signal(signum, frame) -> None:
    """Raise inside the active setup step so it reaps its own child group."""
    raise RuntimeStop(signum)


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
    """Verify exact pins, retained accelerator, true CUDA execution and driver imports."""
    if Path(sys.prefix).resolve() != RUNTIME.resolve():
        raise RuntimeError(f"wrong interpreter: expected {RUNTIME}/bin/python")
    if platform.python_version() != PYTHON_VERSION:
        raise RuntimeError(f"expected Python {PYTHON_VERSION}; got {platform.python_version()}")
    owner = json.loads((RUNTIME.parent / "runtime_owner.json").read_text())
    if owner["lock_sha256"] != sha256(LOCK) or owner["python"] != PYTHON_VERSION:
        raise RuntimeError("owned runtime recipe differs from the current lock/Python")
    if owner["pod_id"] != os.environ.get("RUNPOD_POD_ID"):
        raise RuntimeError("runtime ownership differs from the current pod")
    realized = {name: importlib.metadata.version(name) for name in PINS}
    if realized != PINS:
        raise RuntimeError(f"runtime version mismatch: {realized}")
    locked = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\;]+)", LOCK.read_text(), re.MULTILINE))
    if not set(PINS).issubset(locked):
        raise RuntimeError("runtime lock is incomplete")
    installed = {name: importlib.metadata.version(name) for name in locked}
    if installed != locked:
        raise RuntimeError("installed dependency closure differs from the hashed runtime lock")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "1") != "1":
        raise RuntimeError("parent retained-flashinfer runtime forbids sampler override")
    drivers = driver_check()
    importlib.import_module("flashinfer.comm.fd_exchange")
    vllm = importlib.import_module("vllm")
    for name in ("LLM", "SamplingParams", "TokensPrompt"):
        if not getattr(vllm, name, None):
            raise RuntimeError(f"required real vLLM API missing: {name}")
    for name in ("accelerate", "scipy", "matplotlib", "datasets", "dotenv"):
        importlib.import_module(name)
    model_module = importlib.import_module("transformers.models.qwen3.modeling_qwen3")
    if not getattr(model_module, "Qwen3ForCausalLM", None):
        raise RuntimeError("actual Qwen3ForCausalLM model class missing")
    torch = importlib.import_module("torch")
    if torch.version.cuda != "13.0" or not torch.cuda.is_available():
        raise RuntimeError("CUDA 13.0 must be available in the exact runtime")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Qwen3 chat runtime requires exactly one visible GPU")
    value = torch.ones((2, 2), device="cuda")
    if (value @ value).sum().item() != 8:
        raise RuntimeError("CUDA tensor execution probe failed")
    torch.cuda.synchronize()
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    driver = importlib.import_module("issue2588_run_cell")
    if driver._run_import_check() != 0:
        raise RuntimeError("actual model-runtime driver import check failed")
    patch_path = RUNTIME / "flashinfer_patch.json"
    patch = json.loads(patch_path.read_text())
    if sha256(Path(patch["path"])) != patch["after_sha256"]:
        raise RuntimeError("installed flashinfer patch does not match its receipt")
    return {
        "status": "passed",
        "python": platform.python_version(),
        "pins": realized,
        "dependency_versions": installed,
        "lock_sha256": sha256(LOCK),
        "driver": drivers,
        "flashinfer_patch": patch,
        "cuda": torch.version.cuda,
        "interpreter": sys.executable,
    }


def validate_smoke_prior_report(
    path: Path | None, expected_sha256: str | None, env: dict, *, run_id: str
) -> dict | None:
    """Credit only measured upload time from the explicitly approved v1 -> v2 retry.

    The old report predates an explicit epoch field, so its attempt timestamp
    minus inherited runtime work must reproduce the original supplied epoch.
    All idle gaps remain charged via that unchanged epoch. This is clock
    lineage only: no scientific checkpoint or source identity is adopted.
    """
    inherited_path = env.get(SMOKE_PRIOR_REPORT_ENV)
    inherited_hash = env.get(SMOKE_PRIOR_HASH_ENV)
    if inherited_path or inherited_hash:
        if (
            path is None
            or not inherited_path
            or path.resolve() != Path(inherited_path).resolve()
            or expected_sha256 != inherited_hash
        ):
            raise RuntimeError("runtime/wrapper smoke prior-report arguments disagree")
    if path is None and expected_sha256 is None:
        return None
    if path is None or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise RuntimeError("smoke prior report requires its exact SHA256")
    if run_id != SMOKE_RUN_ID:
        raise RuntimeError("smoke clock continuation is restricted to qwen3-chat-v1 -> v2")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("smoke prior report SHA256 mismatch")
    prior = json.loads(raw)
    if (
        prior.get("surface") != "generic"
        or prior.get("run_id") != SMOKE_PRIOR_RUN_ID
        or prior.get("mode") != "smoke"
        or prior.get("status") != "halted"
        or type(prior.get("rc")) is not int
        or prior["rc"] in (0, RC_WORK_FENCE)
        or prior.get("experiment_complete") is not False
        or prior.get("smoke_clock") is not None
    ):
        raise RuntimeError("smoke prior report is not an eligible terminal v1 halt")

    def seconds(value, name: str) -> float:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise RuntimeError(f"invalid smoke prior report {name}")
        return float(value)

    started_at = env.get(SMOKE_START_ENV)
    if started_at is None:
        raise RuntimeError("smoke continuation requires the unchanged original epoch")
    epoch = float(started_at)
    now = time.time()
    if not math.isfinite(epoch) or epoch <= 0 or epoch > now:
        raise RuntimeError("invalid original smoke epoch")
    attempt, pid = prior["attempt"].rsplit("-", 1)
    if str(prior.get("pid")) != pid or not pid.isdigit():
        raise RuntimeError("smoke prior report PID/attempt mismatch")
    attempt_epoch = datetime.strptime(attempt, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=UTC).timestamp()
    inherited = seconds(prior["inherited_runtime_work_s"], "inherited runtime work")
    elapsed = seconds(prior["elapsed_s"], "elapsed wall")
    work = seconds(prior["work_elapsed_s"], "charged work")
    if abs(attempt_epoch - inherited - epoch) > SMOKE_EPOCH_TOLERANCE_S:
        raise RuntimeError("smoke prior report does not bind the original epoch")
    if attempt_epoch + elapsed > now + SMOKE_EPOCH_TOLERANCE_S or work >= 3600:
        raise RuntimeError("smoke prior report has future wall time or exhausted work")
    credit = 0.0
    step_elapsed = 0.0
    steps = prior.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RuntimeError("smoke prior report lacks completed process records")
    for step in steps:
        duration = seconds(step.get("elapsed_s"), "child elapsed")
        step_elapsed += duration
        if not step.get("phase", "").startswith("upload-"):
            continue
        if (
            step["phase"] not in ("upload-raw", "upload-capture", "upload-partial")
            or type(step.get("rc")) is not int
            or type(step.get("pid")) is not int
            or step["pid"] <= 0
        ):
            raise RuntimeError("smoke prior upload is not an exited scoped process")
        argv = step.get("argv", [])
        for flag, value in (
            ("--surface", "generic"),
            ("--run-id", SMOKE_PRIOR_RUN_ID),
            ("--cell", step.get("cell")),
            ("--phase", step["phase"]),
        ):
            if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != value:
                raise RuntimeError("smoke prior upload command provenance mismatch")
        if "--smoke" not in argv or step.get("cell") not in ("q3_8b_a", "q3_8b_b"):
            raise RuntimeError("smoke prior upload has a different surface")
        credit += duration
    excluded = seconds(inherited + elapsed - work, "excluded upload wall")
    if (
        step_elapsed > elapsed + SMOKE_CLOCK_TOLERANCE_S
        or abs(excluded - credit) > SMOKE_CLOCK_TOLERANCE_S
    ):
        raise RuntimeError("smoke prior upload sum disagrees with measured work/wall clocks")
    if credit > elapsed or credit > now - epoch:
        raise RuntimeError("smoke prior upload credit exceeds elapsed wall")
    measured_upload_s = credit
    credit = min(measured_upload_s, excluded)
    return {
        "original_epoch": epoch,
        "prior_report": str(path.resolve()),
        "prior_report_sha256": expected_sha256,
        "prior_run_id": SMOKE_PRIOR_RUN_ID,
        "run_id": run_id,
        "prior_attempt": prior["attempt"],
        "cumulative_prior_durability_s": credit,
        "measured_prior_upload_elapsed_s": measured_upload_s,
        "prior_work_elapsed_s": work,
        "clock_consistency_error_s": excluded - measured_upload_s,
    }


def remaining_smoke_seconds(env: dict, *, smoke: bool) -> float | None:
    """Account for install, imports, model staging and work across the same launch."""
    if not smoke:
        return None
    grant = CG.from_env(env)
    if grant:
        remaining = grant.remaining_s()
        if remaining <= 0:
            raise RuntimeFence("authorized continuation allowance exhausted during setup")
        return remaining
    start = float(env[SMOKE_START_ENV])
    now = time.time()
    if not math.isfinite(start) or start > now or start <= 0:
        raise RuntimeError("invalid cumulative smoke start timestamp")
    prior_path = env.get(SMOKE_PRIOR_REPORT_ENV)
    prior = validate_smoke_prior_report(
        Path(prior_path) if prior_path else None,
        env.get(SMOKE_PRIOR_HASH_ENV),
        env,
        run_id=SMOKE_RUN_ID,
    )
    supplement_path = env.get(SUPPLEMENT_REPORT_ENV)
    supplement = validate_smoke_supplement(
        Path(supplement_path) if supplement_path else None, env, run_id=SUPPLEMENT_RUN_ID
    )
    allowance = supplement["allowance_s"] if supplement else 3600.0
    remaining = (
        allowance - (now - start) + (prior["cumulative_prior_durability_s"] if prior else 0.0)
    )
    if remaining <= 0:
        raise RuntimeFence("authorized smoke work allowance exhausted during runtime setup")
    return remaining


def run_bounded(argv: list[str], env: dict, *, smoke: bool, deadline: float, log) -> None:
    """Run one build/check process group with the cumulative setup fence."""
    remaining = remaining_smoke_seconds(env, smoke=smoke)
    budget = deadline - time.monotonic()
    if remaining is not None:
        budget = min(budget, remaining)
    if budget <= 0:
        if smoke:
            raise RuntimeFence("cumulative runtime setup fence exhausted")
        raise TimeoutError("explicit runtime setup budget exhausted")
    child = subprocess.Popen(
        argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        code = child.wait(timeout=budget)
    except BaseException as error:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # This exact owned group already exited; preserve the triggering error.
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        # A leader may exit while descendants ignore TERM. Always target the
        # owned group once more, independent of its leader's wait status.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)
        if isinstance(error, subprocess.TimeoutExpired) and smoke:
            raise RuntimeFence("runtime setup stopped at cumulative smoke work fence") from None
        raise
    if code:
        raise RuntimeError(f"runtime command failed rc={code}: {argv[0:3]}")


def build_commands(uv: str) -> list[list[str]]:
    """Return the hashed, parent-pinned install recipe without executing it."""
    python = str(RUNTIME / "bin/python")
    return [
        [uv, "venv", str(RUNTIME), "--python", PYTHON_VERSION],
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
        [python, str(Path(__file__).resolve()), "--patch-flashinfer"],
        [python, str(Path(__file__).resolve()), "--check"],
    ]


def main(argv: list[str] | None = None) -> int:
    """Build on the explicitly owned pod or check locally, then exec the wrapper."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--patch-flashinfer", action="store_true", help=argparse.SUPPRESS)
    action.add_argument("--list-commands", action="store_true")
    parser.add_argument("--pod-id")
    parser.add_argument("--bootstrap-preflight-evidence")
    parser.add_argument("--setup-timeout-seconds", type=float)
    parser.add_argument("--setup-timeout-basis")
    parser.add_argument("wrapper_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.list_commands:
        print(json.dumps(build_commands("uv"), indent=2))
        return 0
    if args.check:
        print(json.dumps(check_runtime(), sort_keys=True))
        return 0
    if args.patch_flashinfer:
        print(json.dumps(patch_flashinfer(), sort_keys=True))
        return 0
    if not args.pod_id or args.pod_id != os.environ.get("RUNPOD_POD_ID"):
        raise RuntimeError("--build requires the exact owned RUNPOD_POD_ID")
    if not (args.bootstrap_preflight_evidence or "").strip():
        raise RuntimeError("root-bootstrap preflight evidence is required before model setup")
    wrapper_args = args.wrapper_args
    if wrapper_args[:1] == ["--"]:
        wrapper_args = wrapper_args[1:]
    if "--mode" not in wrapper_args:
        raise RuntimeError("pass the reviewed wrapper --mode after --")
    mode = wrapper_args[wrapper_args.index("--mode") + 1]
    if mode not in ("smoke", "capture-pilot", "capture", "fit-pilot", "fits"):
        raise RuntimeError("out-of-scope wrapper mode")
    if mode == "smoke":
        setup_budget = 3600.0  # Registered hard safety fence, not a transfer-time estimate.
    else:
        setup_budget = args.setup_timeout_seconds
        if not setup_budget or not math.isfinite(setup_budget) or setup_budget <= 0:
            raise RuntimeError("non-smoke runtime check requires --setup-timeout-seconds")
        if not (args.setup_timeout_basis or "").strip():
            raise RuntimeError("non-smoke runtime timeout requires its explicit sizing basis")
        if mode != "capture-pilot" and not (RUNTIME / "bin/python").exists():
            raise RuntimeError("first runtime build must be part of the smoke, not production")
    deadline = time.monotonic() + setup_budget
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    env = dict(os.environ)
    env.update(
        UV_NO_SYNC="1",
        EPS_CAP_PROFILE="long",
        UV_CACHE_DIR=str(RUNTIME.parent / "uv_cache"),
        PYTHONPATH=str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
        EPM_PREFLIGHT_LARGE_BLOB_URL=(
            "https://huggingface.co/Qwen/Qwen3-8B/resolve/"
            "b968826d9c46dd6066d109eabc6255188de91218/model-00001-of-00005.safetensors"
        ),
    )
    clock_parser = argparse.ArgumentParser(add_help=False)
    clock_parser.add_argument("--run-id", default=SMOKE_RUN_ID)
    clock_parser.add_argument("--smoke-prior-report", type=Path)
    clock_parser.add_argument("--smoke-prior-report-sha256")
    clock_parser.add_argument("--smoke-supplement-report", type=Path)
    clock_parser.add_argument("--continuation-grant", type=Path)
    clock_parser.add_argument("--continuation-grant-sha256")
    clock_parser.add_argument("--science-root", type=Path)
    clock_args, _ = clock_parser.parse_known_args(wrapper_args)
    if mode == "capture-pilot":
        if clock_args.run_id != SUPPLEMENT_RUN_ID or not clock_args.science_root:
            raise RuntimeError("capture-pilot requires v3 and the frozen science root")
        CG.science_root(clock_args.science_root)
    CG.configure_environment(
        env, clock_args.continuation_grant, clock_args.continuation_grant_sha256
    )
    if clock_args.continuation_grant or clock_args.continuation_grant_sha256:
        if (
            mode != "smoke"
            or clock_args.run_id != SUPPLEMENT_RUN_ID
            or not clock_args.continuation_grant
            or not clock_args.continuation_grant_sha256
            or not clock_args.science_root
            or clock_args.smoke_supplement_report
            or clock_args.smoke_prior_report
            or clock_args.smoke_prior_report_sha256
        ):
            raise RuntimeError("invalid frozen-source continuation arguments")
        CG.science_root(clock_args.science_root)
        grant = CG.from_env(env)
        setup_budget = grant.record["allowance_s"]
        deadline = time.monotonic() + setup_budget
    if clock_args.smoke_supplement_report and (
        mode != "smoke" or clock_args.smoke_prior_report or clock_args.smoke_prior_report_sha256
    ):
        raise RuntimeError("supplement is smoke-only and cannot combine with old clock credit")
    supplement = validate_smoke_supplement(
        clock_args.smoke_supplement_report, env, run_id=clock_args.run_id
    )
    if supplement:
        env[SUPPLEMENT_REPORT_ENV] = supplement["prior_report"]
        setup_budget = supplement["allowance_s"]
        deadline = time.monotonic() + setup_budget
        bind_supplement_clock(RUNTIME.parent / "generic" / SUPPLEMENT_RUN_ID, supplement, env)
    if mode != "smoke" and (clock_args.smoke_prior_report or clock_args.smoke_prior_report_sha256):
        raise RuntimeError("prior smoke report is valid only for smoke mode")
    prior_clock = validate_smoke_prior_report(
        clock_args.smoke_prior_report,
        clock_args.smoke_prior_report_sha256,
        env,
        run_id=clock_args.run_id,
    )
    if prior_clock:
        env[SMOKE_PRIOR_REPORT_ENV] = prior_clock["prior_report"]
        env[SMOKE_PRIOR_HASH_ENV] = prior_clock["prior_report_sha256"]
    elif SMOKE_PRIOR_REPORT_ENV in env or SMOKE_PRIOR_HASH_ENV in env:
        raise RuntimeError("prior smoke credit requires explicit pinned report arguments")
    if mode == "smoke" and prior_clock is None:
        env.setdefault(SMOKE_START_ENV, str(time.time()))
    remaining_smoke_seconds(env, smoke=mode == "smoke")
    driver_check()
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("existing root bootstrap must provide uv")
    if RUNTIME.is_symlink() or RUNTIME.parent.resolve() != RUNTIME.parent:
        raise RuntimeError(
            "runtime overlay/symlink relocation requires separately verified approval"
        )
    RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    owner = RUNTIME.parent / "runtime_owner.json"
    identity = {
        "pod_id": args.pod_id,
        "lock_sha256": sha256(LOCK),
        "python": PYTHON_VERSION,
        "bootstrap_preflight_evidence": args.bootstrap_preflight_evidence,
    }
    if owner.exists():
        if json.loads(owner.read_text()) != identity:
            raise RuntimeError("existing owned-runtime identity differs; do not overwrite")
    elif RUNTIME.exists():
        raise RuntimeError("unidentified existing runtime directory; do not overwrite")
    write_receipt(owner, identity)
    commands = build_commands(uv)
    if mode not in {"smoke", "capture-pilot"}:
        commands = commands[-1:]  # Production checks only; never installs or repairs.
    elif (RUNTIME / "bin/python").exists():
        commands = commands[1:]
    if mode in {"smoke", "capture-pilot"}:
        storage_parser = argparse.ArgumentParser(add_help=False)
        storage_parser.add_argument("--min-disk-gb", type=float, required=True)
        storage_parser.add_argument("--per-pod-quota-gb", type=float, required=True)
        storage, _ = storage_parser.parse_known_args(wrapper_args)
        if not (
            math.isfinite(storage.min_disk_gb)
            and storage.min_disk_gb >= 80
            and math.isfinite(storage.per_pod_quota_gb)
            and storage.per_pod_quota_gb >= 200
        ):
            raise RuntimeError("reviewed setup requires >=80GB free and explicit >=200GB pod quota")
        commands = [
            ["findmnt", "-T", str(RUNTIME.parent)],
            [
                sys.executable,
                "-u",
                "-m",
                "explore_persona_space.orchestrate.preflight",
                "--no-gpu",
                "--min-disk",
                str(storage.min_disk_gb),
                "--per-pod-quota-gb",
                str(storage.per_pod_quota_gb),
            ],
            *commands,
        ]
    try:
        with (RUNTIME.parent / "runtime_build.log").open("a") as log:
            for command in commands:
                run_bounded(command, env, smoke=mode == "smoke", deadline=deadline, log=log)
        remaining_smoke_seconds(env, smoke=mode == "smoke")
    except (RuntimeFence, RuntimeStop) as error:
        code = RC_WORK_FENCE if isinstance(error, RuntimeFence) else error.rc
        write_receipt(
            RUNTIME.parent / "runtime_pause.json",
            {
                "status": "work_fence" if code == RC_WORK_FENCE else "interrupted",
                "rc": code,
                "reason": str(error),
                "started_at": env.get(SMOKE_START_ENV),
                "smoke_clock": prior_clock,
                "smoke_supplement": supplement,
                "identity": identity,
            },
        )
        return code
    except Exception as error:
        write_receipt(
            RUNTIME.parent / "runtime_pause.json",
            {
                "status": "failed",
                "reason": str(error),
                "identity": identity,
                "smoke_clock": prior_clock,
                "smoke_supplement": supplement,
            },
        )
        raise
    python = str(RUNTIME / "bin/python")
    os.execve(
        python,
        [python, "-u", str(REPO_ROOT / "scripts/issue2588_chat_dispatch.py"), *wrapper_args],
        env,
    )
    raise AssertionError("execve returned unexpectedly")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeStop as error:
        raise SystemExit(error.rc) from error
