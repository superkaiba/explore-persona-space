#!/usr/bin/env python3
"""Phase-isolated Qwen3-8B chat follow-up (task2588 plan v6).

This is the workload, launched by the VM owner with fully detached stdio.
Every launch writes its own PID and fresh timestamped logs. It neither starts
pollers nor provisions/stops compute. The owner commits the small dispatch
reports and verifies all uploaded artifacts before teardown.

--list-commands is a read-only JSON command manifest; --import-check checks
the wrapper without loading a model. Runtime requires --transfer-plan JSON:
{"surface":"generic", "run_id":"qwen3-chat-v2", "transfers": {
  "stage": {"bytes": <projected>, "bytes_per_s": <measured-or-expected>,
            "retry_calls": <maximum sequential retry envelopes>,
            "basis": "source of byte, throughput, and call-count values"}}}
Keys can be phase names or cell:phase. Supply every listed transfer phase,
including upload-partial (the timeout/failure durability tail). Timeout is
2*bytes/bytes_per_s + retry_calls*EPM_HF_RETRY_BUDGET_S + 1 seconds. This
exceeds the stated full retry exposure, retaining a 2x transfer margin.
The basis must count verification/listing calls as well as file transfers.
Upload entries may replace "bytes" with "source":"local_cell_tree". Immediately
before each upload, count the cell's actual bytes plus 64 KiB per file and one
extra receipt (a conservative metadata allowance), then apply the same formula.
They may also replace "retry_calls" with "retry_calls_source":"local_upload_operations";
count actual scoped bulk/single-file operations, including receipts and checkpoints.
Automatic overflow routing is disabled: only the approved dataset prefix is valid.

Smoke runs both arms at the driver's inherited small row counts and REAL
caps; a 3600-second cumulative work fence excludes upload tails. That fence
is a safety limit, never a runtime estimate. On expiry, stop the active
process group and upload existing partial outputs without claiming complete
captures. A fit-pilot preserves one production-shape layer unit and pauses
with rc7; fits mode resumes the core driver's validated checkpoints. No
mode completes the whole experiment: selected-map rank analysis is separate.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue2588_chat_runtime import (  # noqa: E402
    SUPPLEMENT_RUN_ID,
    bind_supplement_clock,
    validate_smoke_prior_report,
    validate_smoke_supplement,
)
import issue2588_chat_grant as CG  # noqa: E402

from explore_persona_space.atomic_io import write_json_atomic, write_text_atomic  # noqa: E402
from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

RUN_ID = "qwen3-chat-v2"
CELLS = ("q3_8b_a", "q3_8b_b")
CHAT_STAGES = ("train_10k", "val_400", "test_1000", "ceiling_s43", "ceiling_s44")
DRIVER = REPO_ROOT / "scripts" / "issue2588_run_cell.py"
RUNTIME = REPO_ROOT / "scripts" / "issue2588_chat_runtime.py"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MODEL_PROBE_URL = (
    f"https://huggingface.co/Qwen/Qwen3-8B/resolve/{MODEL_REVISION}/"
    "model-00001-of-00005.safetensors"
)
MODEL_PROBE_BYTES = 1024 * 1024  # canonical preflight large-blob range size
OFFLINE_PHASES = frozenset({"prologue", "gen", "parse", "capture", "fits"})
GPU_PHASES = frozenset({"gen", "capture", "fits"})
TRANSFER_PHASES = frozenset(
    {
        "preflight",
        "transfer_check",
        "stage-runtime",
        "stage",
        "upload-raw",
        "upload-capture",
        "upload-fits",
        "upload-partial",
    }
)
RC_PILOT_PAUSE = 7
RC_WORK_FENCE = 8
RC_TRANSFER_TIMEOUT = 87  # stage_hub_prefix's existing timeout convention
GPU_FREE_MIB = 2048  # gotchas.md #1333 phase-boundary hygiene floor
GPU_DRAIN_S = 180  # same #1333 drain-wait convention
PROGRESS_S = 30
UPLOAD_METADATA_BYTES_PER_FILE = 65536  # parent-approved conservative receipt/manifest allowance


@dataclass(frozen=True)
class Step:
    """One fresh child invocation, plus its completion/timeout contract."""

    cell: str
    phase: str
    argv: tuple[str, ...]
    expected_rc: int = 0

    @property
    def key(self) -> str:
        """Return a stable cell-qualified progress key."""
        return f"{self.cell}:{self.phase}"


class PhaseFailure(RuntimeError):
    """A child failed or a registered fence stopped it; outputs are retained."""

    def __init__(self, step: Step, rc: int, reason: str):
        super().__init__(f"{step.key}: {reason} (rc={rc})")
        self.step = step
        self.rc = rc


class WorkloadSignal(RuntimeError):
    """An external stop requests child cleanup and a finite durability tail."""

    def __init__(self, signum: int):
        super().__init__(f"workload received signal {signum}")
        self.rc = 128 + signum


def handle_signal(signum, frame) -> None:
    """Route TERM/INT through the same fail-loud persistence path as child errors."""
    raise WorkloadSignal(signum)


def build_parser() -> argparse.ArgumentParser:
    """Expose only the approved run, cells, surface, and four phase modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("smoke", "capture-pilot", "capture", "fit-pilot", "fits")
    )
    parser.add_argument("--surface", choices=("generic",), default="generic")
    parser.add_argument("--run-id", choices=(RUN_ID, SUPPLEMENT_RUN_ID), default=RUN_ID)
    parser.add_argument("--out-root", type=Path, default=Path("/workspace/eps2588_qwen3_chat"))
    parser.add_argument("--sentinel-dir", type=Path, default=Path("/workspace/logs"))
    parser.add_argument("--pid-file", type=Path, default=Path("/workspace/logs/issue-2588.pid"))
    parser.add_argument("--transfer-plan", type=Path)
    parser.add_argument("--smoke-prior-report", type=Path)
    parser.add_argument("--smoke-prior-report-sha256")
    parser.add_argument("--smoke-supplement-report", type=Path)
    parser.add_argument("--continuation-grant", type=Path)
    parser.add_argument("--continuation-grant-sha256")
    parser.add_argument("--science-root", type=Path)
    parser.add_argument("--batch-fit-uploads", action="store_true")
    parser.add_argument("--min-disk-gb", type=float)
    parser.add_argument("--per-pod-quota-gb", type=float)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--preflight-evidence", help="manual preflight evidence if bypassing the CLI"
    )
    parser.add_argument("--list-commands", action="store_true")
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--transfer-check", action="store_true", help=argparse.SUPPRESS)
    return parser


def cell_step(args: argparse.Namespace, cell: str, phase: str, *, pilot: bool = False) -> Step:
    """Compose the exact core-driver CLI; never call its phases in process."""
    if cell not in CELLS:
        raise ValueError(f"out-of-scope cell: {cell}")
    allowed = (TRANSFER_PHASES - {"preflight", "transfer_check"}) | OFFLINE_PHASES
    if phase not in allowed:
        raise ValueError(f"out-of-scope cell phase: {phase}")
    if phase == "upload-fits" and args.batch_fit_uploads:
        if args.mode != "fits" or not args.science_root:
            raise ValueError("batched fit uploads require the frozen fits mode")
        return Step(
            cell,
            phase,
            (
                sys.executable,
                "-u",
                str(REPO_ROOT / "scripts/issue2588_chat_upload_fits.py"),
                "--science-root",
                str(args.science_root),
                "--out-root",
                str(args.out_root),
                "--cell",
                cell,
            ),
        )
    argv = [
        sys.executable,
        "-u",
        str(args.science_root / "scripts/issue2588_run_cell.py" if args.science_root else DRIVER),
        "--surface",
        args.surface,
        "--run-id",
        args.run_id,
        "--cell",
        cell,
        "--phase",
        phase,
        "--out-root",
        str(args.out_root),
        "--device",
        "cuda",  # inherited capture loader requires this exact spelling; CVD pins GPU 0
        "--gpu-count",
        "1",
    ]
    if args.mode in {"smoke", "capture-pilot"}:
        argv.append("--smoke")
    if pilot:
        if args.mode != "fit-pilot" or phase != "fits" or cell != CELLS[0]:
            raise ValueError("only the first production arm's fits phase is the fit pilot")
        argv.extend(("--fit-max-units", "1"))
    return Step(cell, phase, tuple(argv), RC_PILOT_PAUSE if pilot else 0)


def build_steps(args: argparse.Namespace) -> list[Step]:
    """Keep raw/capture persistence ahead of every downstream fit."""
    driver = args.science_root / "scripts/issue2588_run_cell.py" if args.science_root else DRIVER
    runtime = (
        args.science_root / "scripts/issue2588_chat_runtime.py" if args.science_root else RUNTIME
    )
    steps = [
        Step("driver", "runtime_check", (sys.executable, "-u", str(runtime), "--check")),
        Step("driver", "import_check", (sys.executable, "-u", str(driver), "--import-check")),
    ]
    if not args.skip_preflight:
        preflight = [sys.executable, "-u", "-m", "explore_persona_space.orchestrate.preflight"]
        if args.min_disk_gb is not None:
            preflight.extend(("--min-disk", str(args.min_disk_gb)))
        if args.per_pod_quota_gb is not None:
            preflight.extend(("--per-pod-quota-gb", str(args.per_pod_quota_gb)))
        steps.append(Step("driver", "preflight", tuple(preflight)))
    # The strict model-range probe cannot be bypassed with the general preflight.
    steps.append(
        Step("driver", "transfer_check", (sys.executable, "-u", __file__, "--transfer-check"))
    )
    # One immutable shared snapshot/G2 staging pass, including on fit-only launches.
    steps.append(cell_step(args, CELLS[0], "stage-runtime"))
    if args.continuation_grant:
        steps.append(
            Step(
                "restore",
                "stage",
                (
                    sys.executable,
                    "-u",
                    str(REPO_ROOT / "scripts/issue2588_chat_restore.py"),
                    "--science-root",
                    str(args.science_root),
                    "--out-root",
                    str(args.out_root),
                ),
            )
        )
    if args.mode == "capture-pilot":
        # Progress156 lifts the old time cap. This is existing pilot capture,
        # not new generation or permission to promote pilot rows to production.
        steps.append(
            Step(
                "restore-complete",
                "stage",
                (
                    sys.executable,
                    "-u",
                    str(REPO_ROOT / "scripts/issue2588_chat_restore_complete.py"),
                    "--science-root",
                    str(args.science_root),
                    "--out-root",
                    str(args.out_root),
                ),
            )
        )
        steps.extend(cell_step(args, CELLS[1], phase) for phase in ("capture", "upload-capture"))
    elif args.mode in {"smoke", "capture"}:
        for cell in CELLS[1:] if args.smoke_supplement_report or args.continuation_grant else CELLS:
            for phase in (
                "prologue",
                "stage",
                "gen",
                "parse",
                "upload-raw",
                "capture",
                "upload-capture",
            ):
                steps.append(cell_step(args, cell, phase))
    else:
        # Re-enter core verification, which must revalidate its complete file sets.
        for cell in CELLS:
            for phase in ("upload-raw", "upload-capture"):
                steps.append(cell_step(args, cell, phase))
        if args.mode == "fit-pilot":
            steps.append(cell_step(args, CELLS[0], "fits", pilot=True))
            steps.append(cell_step(args, CELLS[0], "upload-partial"))
        else:
            for cell in CELLS:
                steps.extend(cell_step(args, cell, phase) for phase in ("fits", "upload-fits"))
    return steps


def transfer_limits(args: argparse.Namespace, steps: list[Step], retry_s: float) -> dict[str, dict]:
    """Validate a caller-sourced sizing basis and expose the full retry exposure."""
    if args.transfer_plan is None:
        raise ValueError(
            "runtime requires --transfer-plan with a byte/throughput/retry sizing basis"
        )
    plan = json.loads(args.transfer_plan.read_text(encoding="utf-8"))
    if plan.get("surface") != args.surface or plan.get("run_id") != args.run_id:
        raise ValueError("transfer plan does not match this surface/run ID")
    entries = plan["transfers"]
    limits = {}
    for step in [*steps, *(cell_step(args, cell, "upload-partial") for cell in CELLS)]:
        if step.phase not in TRANSFER_PHASES:
            continue
        rec = entries.get(step.key, entries.get(step.phase))
        if (
            not isinstance(rec, dict)
            or not isinstance(rec.get("basis"), str)
            or not rec["basis"].strip()
        ):
            raise ValueError(f"missing transfer sizing basis for {step.key}")
        local_tree = rec.get("source") == "local_cell_tree"
        if rec.get("source") not in (None, "local_cell_tree"):
            raise ValueError(f"unknown transfer byte source for {step.key}")
        if local_tree and not step.phase.startswith("upload-"):
            raise ValueError("local_cell_tree byte sizing is only valid for uploads")
        nbytes = None if local_tree else rec["bytes"]
        local_calls = rec.get("retry_calls_source") == "local_upload_operations"
        if rec.get("retry_calls_source") not in (None, "local_upload_operations"):
            raise ValueError(f"unknown retry count source for {step.key}")
        if local_calls and not local_tree:
            raise ValueError("local_upload_operations requires local_cell_tree byte sizing")
        rate = rec["bytes_per_s"]
        calls = None if local_calls else rec["retry_calls"]
        if (
            (
                not local_tree
                and (isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0)
            )
            or isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or rate <= 0
            or (
                not local_calls
                and (isinstance(calls, bool) or not isinstance(calls, int) or calls < 1)
            )
        ):
            raise ValueError(f"invalid transfer size/rate/retry_calls for {step.key}")
        exposure = None if local_calls else calls * retry_s
        timeout = None if local_tree else 2 * nbytes / rate + exposure + 1
        if timeout is not None and not math.isfinite(timeout):
            raise ValueError(f"non-finite transfer timeout for {step.key}")
        limits[step.key] = {
            **rec,
            "retry_budget_s": retry_s,
            "retry_exposure_s": exposure,
            "timeout_s": timeout,
        }
    return limits


def child_environment(args: argparse.Namespace) -> dict[str, str]:
    """Pin one inherited GPU and the long-cap recipe before all child imports."""
    env = dict(os.environ)
    CG.configure_environment(env, args.continuation_grant, args.continuation_grant_sha256)
    visible = env.get("CUDA_VISIBLE_DEVICES", "0").strip()
    if not visible or len(visible.split(",")) != 1 or visible == "-1":
        raise ValueError("this workload requires exactly one assigned CUDA_VISIBLE_DEVICES GPU")
    env.update(
        CUDA_VISIBLE_DEVICES=visible,
        EPS_CAP_PROFILE="long",
        PYTHONUNBUFFERED="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        EPM_HF_FILECOUNT_FALLBACK="0",
    )
    cache = args.out_root / "generic" / args.run_id / "hf_cache"
    env.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache))
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str((args.science_root or REPO_ROOT) / "src"), env.get("PYTHONPATH", ""))
        if value
    )
    if args.science_root:
        CG.science_root(args.science_root)
        env["EPS_GIT_SHA"] = CG.SCIENCE_SHA
    env["EPM_PREFLIGHT_LARGE_BLOB_URL"] = MODEL_PROBE_URL
    return env


def check_model_transfer() -> dict:
    """Read exactly one pinned-model range; a zero-byte or ignored Range is fatal."""
    from explore_persona_space.orchestrate.hub import retry_transient

    def fetch_range() -> dict:
        """Use the canonical probe's 1 MiB range and 20-second socket bound."""
        start = time.monotonic()
        request = urllib.request.Request(
            MODEL_PROBE_URL, headers={"Range": f"bytes=0-{MODEL_PROBE_BYTES - 1}"}
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or not content_range.startswith(
                f"bytes 0-{MODEL_PROBE_BYTES - 1}/"
            ):
                raise RuntimeError("pinned model probe did not honor the exact byte Range")
            received = 0
            while received < MODEL_PROBE_BYTES:
                chunk = response.read(min(65536, MODEL_PROBE_BYTES - received))
                if not chunk:
                    raise RuntimeError(f"pinned model probe ended at {received} bytes")
                received += len(chunk)
        elapsed = time.monotonic() - start
        return {
            "status": "verified",
            "url": MODEL_PROBE_URL,
            "bytes": received,
            "elapsed_s": elapsed,
            "bytes_per_s": received / elapsed,
            "basis": "measured 1 MiB Range only; not a full-model throughput estimate",
        }

    return retry_transient(fetch_range, what="pinned Qwen3 model Range transfer probe")


def reap_group(pid: int) -> None:
    """Reap only this fresh session's workers, including reparented engines."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def wait_gpu_free(gpu: str, *, timeout_s: float = GPU_DRAIN_S) -> None:
    """Check device-level memory as well as namespace-sensitive compute-app rows."""
    deadline = time.monotonic() + timeout_s
    while True:
        common = ["nvidia-smi", "--id", gpu, "--format=csv,noheader,nounits"]
        memory = subprocess.run(
            [*common, "--query-gpu=memory.used"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        apps = subprocess.run(
            [*common, "--query-compute-apps=pid,used_memory,gpu_uuid"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        rows = memory.stdout.strip().split("\n")
        if len(rows) != 1:
            raise RuntimeError(f"expected one device-memory row for GPU {gpu}: {memory.stdout!r}")
        used = float(rows[0])  # unparseable/N/A is a failure, never zero
        if math.isfinite(used) and used <= GPU_FREE_MIB:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU {gpu} did not release memory: {used} MiB; holders={apps.stdout!r}"
            )
        time.sleep(min(2, max(0, deadline - time.monotonic())))


def inherited_work_s(args: argparse.Namespace, env: dict[str, str]) -> float:
    """Validate the runtime builder's original smoke-clock epoch before any writes."""
    if args.mode != "smoke" and (args.smoke_prior_report or args.smoke_prior_report_sha256):
        raise ValueError("prior smoke report is valid only for smoke mode")
    prior = validate_smoke_prior_report(
        args.smoke_prior_report, args.smoke_prior_report_sha256, env, run_id=args.run_id
    )
    started_at = env.get("EPS2588_SMOKE_STARTED_AT") if args.mode == "smoke" else None
    if started_at is None:
        return 0.0
    epoch = float(started_at)
    now = time.time()
    if not math.isfinite(epoch) or epoch <= 0 or epoch > now:
        raise ValueError("EPS2588_SMOKE_STARTED_AT must be a finite, nonfuture epoch")
    return now - epoch - (prior["cumulative_prior_durability_s"] if prior else 0.0)


def local_upload_operations(
    cell_root: Path, cell: str, phase: str, *, batch_fits: bool = False
) -> dict:
    """Count core upload groups, not tensor shards; every helper verifies its own commit.

    A bulk helper has two retry envelopes (commit and scoped listing). A single
    file has three (commit, tree endpoint, exact-file HEAD fallback). The final
    receipt adds repo-info + immutable listing + its own single-file upload.
    Successful ordinary phases additionally upload their completion checkpoint.
    Raw/capture each have one phase-wide bulk payload; provenance and exclusions
    are included in that same payload rather than separate single-file commits.
    Parent-approved overflow disablement makes these the complete helper paths.
    """
    fits = cell_root / "fits"
    position = "prompt_last" if cell == CELLS[0] else "cot_boundary"
    unit_files = [
        fits / f"percell_{position}_L{layer:02d}.json" for layer in (*range(0, 36, 2), 35)
    ]
    pilot = fits / "fit_pilot.json"
    if phase in ("upload-raw", "upload-capture") or (phase == "upload-fits" and batch_fits):
        bulk = 1
        single = 0
    elif phase == "upload-partial":
        bulk = 1
        single = 0
    elif phase == "upload-fits":
        bulk = 0
        single = sum(
            p.is_file()
            for p in [
                *unit_files,
                pilot,
                fits / f"fits_{position}.json",
                fits / f"perrow_{position}.json",
            ]
        )
        single += 1  # generated results.json lives outside the cell tree
    else:
        raise ValueError(f"no scoped upload-operation contract for {phase}")
    receipt = phase != "upload-fits" or batch_fits
    checkpoint = phase != "upload-partial"
    return {
        "bulk_groups": bulk,
        "single_files": single,
        "verified_receipts": int(receipt),
        "completion_checkpoints": int(checkpoint),
        "retry_calls": 2 * bulk + 3 * single + 5 * receipt + 3 * checkpoint,
    }


class Runner:
    """Record every child exit to a fresh attempt directory and progress log."""

    def __init__(self, args: argparse.Namespace, env: dict[str, str], limits: dict[str, dict]):
        self.args, self.env, self.limits = args, env, limits
        self.grant = CG.from_env(env)
        self.supplement = validate_smoke_supplement(
            args.smoke_supplement_report, env, run_id=args.run_id
        )
        bind_supplement_clock(args.out_root / "generic" / args.run_id, self.supplement, env)
        self.work_limit_s = self.supplement["allowance_s"] if self.supplement else 3600.0
        self.smoke_clock = validate_smoke_prior_report(
            args.smoke_prior_report, args.smoke_prior_report_sha256, env, run_id=args.run_id
        )
        self.initial_work_s = inherited_work_s(args, env)
        if self.grant:
            self.work_limit_s = self.grant.record["allowance_s"]
            self.initial_work_s -= self.grant.credit_s()
        self.started = time.monotonic()
        self.attempt = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}"
        self.root = args.out_root / "generic" / args.run_id
        self.directory = self.root / "dispatch" / args.mode / self.attempt
        self.directory.mkdir(parents=True, exist_ok=False)
        self.main_log = (self.directory / "dispatch.log").open("x", encoding="utf-8")
        self.records: list[dict] = []
        self.durability_s = 0.0
        self.durability_started: float | None = None
        self.meta = as_metadata_dict(git_provenance(REPO_ROOT, __file__), phase=args.mode)

    @property
    def work_s(self) -> float:
        """Charge runtime construction and all wrapper overhead, excluding only uploads."""
        now = time.monotonic()
        active_tail = 0.0 if self.durability_started is None else now - self.durability_started
        return self.initial_work_s + now - self.started - self.durability_s - active_tail

    def log(self, message: str) -> None:
        """Flush a timestamped wrapper message to both observability channels."""
        line = f"{datetime.now(UTC).isoformat()} {message}"
        print(line, flush=True)
        print(line, file=self.main_log, flush=True)

    def report(self, status: str, rc: int, **extra) -> dict:
        """Persist process-exit evidence; this is not the core's resume predicate."""
        payload = {
            "surface": self.args.surface,
            "run_id": self.args.run_id,
            "mode": self.args.mode,
            "attempt": self.attempt,
            "status": status,
            "rc": rc,
            "pid": os.getpid(),
            "metadata": self.meta,
            "experiment_complete": False,
            "steps": self.records,
            "work_elapsed_s": self.work_s,
            "inherited_runtime_work_s": self.initial_work_s,
            "smoke_clock": self.smoke_clock,
            "smoke_supplement": self.supplement,
            "continuation_grant": self.grant.record if self.grant else None,
            "science_source": str(self.args.science_root) if self.args.science_root else None,
            "work_allowance_s": self.work_limit_s if self.args.mode == "smoke" else None,
            "smoke_original_epoch": self.env.get("EPS2588_SMOKE_STARTED_AT"),
            "elapsed_s": time.monotonic() - self.started,
            "transfer_limits": self.limits,
            **extra,
        }
        write_json_atomic(self.directory / "report.json", payload)
        return payload

    def resolve_transfer_limit(self, step: Step) -> dict | None:
        """Size each upload from its current cell tree; future files are not guessed."""
        limit = self.limits.get(step.key)
        if limit is None or limit.get("source") != "local_cell_tree":
            return limit
        sub = "smoke_cap_long" if self.args.mode in {"smoke", "capture-pilot"} else "cells_cap_long"
        cell_root = self.root / sub / step.cell
        files = [p for p in cell_root.rglob("*") if p.is_file()]
        if not files:
            raise RuntimeError(f"empty upload source: {cell_root}")
        raw_bytes = sum(p.stat().st_size for p in files)
        overhead = (len(files) + 1) * UPLOAD_METADATA_BYTES_PER_FILE
        nbytes = raw_bytes + overhead
        if limit.get("retry_calls_source") == "local_upload_operations":
            operations = local_upload_operations(
                cell_root, step.cell, step.phase, batch_fits=self.args.batch_fit_uploads
            )
            calls = operations["retry_calls"]
            limit = {
                **limit,
                "retry_calls": calls,
                "retry_exposure_s": calls * limit["retry_budget_s"],
                "observed_upload_operations": operations,
            }
        limit = {
            **limit,
            "bytes": nbytes,
            "observed_bytes": raw_bytes,
            "observed_files": len(files),
            "metadata_allowance_bytes": overhead,
            "timeout_s": 2 * nbytes / limit["bytes_per_s"] + limit["retry_exposure_s"] + 1,
        }
        self.limits[step.key] = limit
        return limit

    def wait_for_gpu(self, step: Step) -> None:
        """Charge device drain to the same smoke fence before and after GPU phases."""
        remaining = self.work_limit_s - self.work_s if self.args.mode == "smoke" else None
        if remaining is not None and remaining <= 0:
            raise PhaseFailure(step, RC_WORK_FENCE, "smoke work fence exhausted at GPU boundary")
        try:
            wait_gpu_free(
                self.env["CUDA_VISIBLE_DEVICES"],
                timeout_s=min(GPU_DRAIN_S, remaining) if remaining else GPU_DRAIN_S,
            )
        except RuntimeError as exc:
            if remaining is not None and self.work_s >= self.work_limit_s:
                raise PhaseFailure(
                    step, RC_WORK_FENCE, "smoke work fence exhausted during GPU drain"
                ) from exc
            raise

    def run(self, step: Step, *, durability_tail: bool = False) -> None:
        """Run the real command in a fresh session and classify only after wait()."""
        begin = time.monotonic()
        begin_epoch = time.time()
        receipt_key = f"{self.attempt}:{len(self.records)}:{step.key}"
        work = not step.phase.startswith("upload-") and not durability_tail
        if not work:
            self.durability_started = begin
        try:
            self._run_step(step, begin, work=work, durability_tail=durability_tail)
        finally:
            if not work:
                self.durability_s += time.monotonic() - begin
                self.durability_started = None
                if self.grant:
                    self.grant.finish_upload(receipt_key, begin_epoch, time.time())

    def _run_step(self, step: Step, begin: float, *, work: bool, durability_tail: bool) -> None:
        """Keep cleanup, actual exit evidence and failure classification in one path."""
        gpu = step.phase in GPU_PHASES
        # The explicit smoke safety fence also covers startup/model downloads.
        # Uploads retain their independent, retry-aware durability allowance.
        remaining = self.work_limit_s - self.work_s if self.args.mode == "smoke" and work else None
        if remaining is not None and remaining <= 0:
            raise PhaseFailure(step, RC_WORK_FENCE, "smoke work fence exhausted before launch")
        limit = self.resolve_transfer_limit(step)
        timeout = limit["timeout_s"] if limit else None
        smoke_fence = remaining is not None and (timeout is None or remaining <= timeout)
        if smoke_fence:
            timeout = remaining
        if gpu:
            self.wait_for_gpu(step)
        if remaining is not None and self.work_s >= self.work_limit_s:
            raise PhaseFailure(
                step, RC_WORK_FENCE, "smoke work fence exhausted before child launch"
            )
        path = self.directory / f"{len(self.records):02d}-{step.cell}-{step.phase}.log"
        self.log(
            f"[phase={step.phase.replace('-', '_')}] start {step.key} log={path} timeout_s={timeout}"
        )
        rc, timed_out, interrupted = None, False, None
        with path.open("x", encoding="utf-8") as log:
            print(f"start {step.key} argv={json.dumps(step.argv)}", file=log, flush=True)
            child_env = dict(self.env)
            if step.phase in OFFLINE_PHASES | {"runtime_check", "import_check"}:
                child_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            proc = subprocess.Popen(
                step.argv,
                cwd=self.args.science_root or REPO_ROOT,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.log(f"child {step.key} pid={proc.pid}")
            try:
                while True:
                    elapsed = time.monotonic() - begin
                    wait_s = (
                        PROGRESS_S
                        if timeout is None
                        else min(PROGRESS_S, max(0, timeout - elapsed))
                    )
                    try:
                        rc = proc.wait(timeout=wait_s)
                        break
                    except subprocess.TimeoutExpired:
                        self.log(
                            f"progress {step.key} pid={proc.pid} elapsed_s={time.monotonic() - begin:.1f}"
                        )
                        if timeout is not None and time.monotonic() - begin >= timeout:
                            timed_out = True
                            reap_group(proc.pid)
                            rc = proc.wait(timeout=10)
                            break
            except Exception as exc:
                interrupted = exc
            finally:
                inner_active = sys.exc_info()[0] is not None or interrupted is not None
                try:
                    reap_group(proc.pid)
                    if proc.poll() is None:
                        proc.wait(timeout=10)
                except Exception:
                    self.log("child cleanup failure: " + traceback.format_exc())
                    if not inner_active:
                        raise
        if rc is None:
            rc = proc.returncode
        elapsed = time.monotonic() - begin
        result = {
            "cell": step.cell,
            "phase": step.phase,
            "argv": list(step.argv),
            "pid": proc.pid,
            "rc": rc,
            "elapsed_s": elapsed,
            "log": str(path),
            "timed_out": timed_out,
            "durability_tail": durability_tail,
            "timeout_s": timeout,
            "timeout_basis": "smoke_work_fence"
            if smoke_fence
            else "transfer_plan"
            if limit
            else None,
        }
        self.records.append(result)
        self.report("running", 0)
        self.log(f"child {step.key} exited rc={rc} elapsed_s={elapsed:.3f}")
        if interrupted is not None:
            raise interrupted
        if rc != step.expected_rc or timed_out:
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line in deque(stream, maxlen=120):
                    # The child may print a terminal token before actually exiting.
                    self.log(
                        "child-log " + line.rstrip().replace("[phase=done]", "[child complete]")
                    )
            failure_rc = (
                RC_WORK_FENCE
                if timed_out and smoke_fence
                else RC_TRANSFER_TIMEOUT
                if timed_out
                else int(rc or 1)
            )
            raise PhaseFailure(
                step, failure_rc, "timeout" if timed_out else "unexpected child exit"
            )
        if gpu:
            self.wait_for_gpu(step)

    def sentinel(self, payload: dict, *, gate: str | None = None) -> Path:
        """Emit a write-once poller envelope, never use it as local resume state."""
        kind = (
            "epm:smoke-result" if self.args.mode in {"smoke", "capture-pilot"} else "epm:progress"
        )
        path = self.args.sentinel_dir / f"issue-2588-{kind.replace(':', '_')}-{self.attempt}.json"
        envelope = {
            "sentinel_schema_version": 1,
            "kind": kind,
            "version": 1,
            "task_id": 2588,
            "by": "issue2588_chat_dispatch",
            "ts": datetime.now(UTC).isoformat(),
            "gate": gate or "phase",
            "blocks_pipeline": gate is not None,
            "note": json.dumps(payload),
        }
        write_json_atomic(path, envelope)
        self.log(f"sentinel={path} kind={kind} gate={envelope['gate']}")
        return path


def validate_pilot(args: argparse.Namespace) -> dict:
    """Require the actual production pilot checkpoint and absence of fit completion."""
    cell = args.out_root / "generic" / args.run_id / "cells_cap_long" / CELLS[0]
    rec = json.loads((cell / "fits" / "fit_pilot.json").read_text(encoding="utf-8"))
    identity = json.loads((cell / "run_identity.json").read_text(encoding="utf-8"))
    unit = json.loads((cell / "fits" / "percell_prompt_last_L00.json").read_text(encoding="utf-8"))
    if (
        rec.get("status") != "pilot_complete"
        or rec.get("phase_complete") is not False
        or rec.get("completed_units") != 1
        or rec.get("total_layer_units") != 19
        or not isinstance(rec.get("unit_elapsed_s"), (int, float))
        or not math.isfinite(rec["unit_elapsed_s"])
        or rec["unit_elapsed_s"] <= 0
        or rec.get("identity") != identity
        or unit.get("identity") != identity
        or identity.get("surface") != "generic"
        or identity.get("run_id") != args.run_id
        or identity.get("smoke") is not False
        or identity.get("cell") != CELLS[0]
        or rec.get("layer") != 0
        or rec.get("input_position") != "prompt_last"
        or rec.get("d") != 4096
        or rec.get("n") != unit.get("n")
        or rec.get("unit_elapsed_s") != unit.get("unit_elapsed_s")
    ):
        raise RuntimeError("invalid production fit pilot checkpoint")
    if (cell / "phase_done" / "fits.json").exists():
        raise RuntimeError("fit pilot must not have a full fits completion sentinel")
    return rec


def main(argv: list[str] | None = None) -> int:
    """Run one approved mode, persist its outcome, and hand control to the VM owner."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.import_check:
        assert_args_attributes_defined(__file__)
        return 0
    if args.transfer_check:
        print(json.dumps(check_model_transfer()), flush=True)
        return 0
    if args.mode is None:
        parser.error("--mode is required")
    if args.batch_fit_uploads and (
        args.mode != "fits" or not args.science_root or args.run_id != SUPPLEMENT_RUN_ID
    ):
        parser.error("batched fit uploads require frozen-source v3 fits mode")
    if args.mode == "capture-pilot" and (args.run_id != SUPPLEMENT_RUN_ID or not args.science_root):
        parser.error("capture-pilot requires v3 and the frozen science root")
    if bool(args.continuation_grant) != bool(args.continuation_grant_sha256):
        parser.error("continuation requires grant path and hash")
    if args.continuation_grant and (
        args.mode != "smoke"
        or args.run_id != SUPPLEMENT_RUN_ID
        or not args.science_root
        or args.smoke_supplement_report
        or args.smoke_prior_report
        or args.smoke_prior_report_sha256
    ):
        parser.error("continuation is frozen-source v3 smoke-only without legacy credits")
    if args.science_root:
        args.science_root = CG.science_root(args.science_root)
    if args.smoke_supplement_report and (
        args.mode != "smoke"
        or args.smoke_prior_report
        or args.smoke_prior_report_sha256
        or args.run_id != SUPPLEMENT_RUN_ID
    ):
        parser.error("supplement is v3 smoke-only and cannot combine with old clock credit")
    if (
        args.mode == "smoke"
        and args.run_id == SUPPLEMENT_RUN_ID
        and not args.smoke_supplement_report
        and not args.continuation_grant
    ):
        parser.error("v3 smoke requires the explicitly authorized supplemental report")
    if args.mode != "smoke" and (args.smoke_prior_report or args.smoke_prior_report_sha256):
        parser.error("prior smoke report is valid only for smoke mode")
    args.out_root = args.out_root.resolve()
    args.sentinel_dir = args.sentinel_dir.resolve()
    args.pid_file = args.pid_file.resolve()
    if args.out_root.suffix in {".json", ".jsonl", ".log"} or args.out_root.is_file():
        parser.error("--out-root must be a directory")
    if args.skip_preflight and not args.preflight_evidence:
        parser.error("--skip-preflight requires --preflight-evidence from a manual check")
    steps = build_steps(args)
    if args.list_commands:
        print(
            json.dumps(
                {
                    "surface": args.surface,
                    "run_id": args.run_id,
                    "mode": args.mode,
                    "environment": {"EPS_CAP_PROFILE": "long"},
                    "commands": [
                        {
                            "key": s.key,
                            "argv": s.argv,
                            "expected_rc": s.expected_rc,
                            "requires_transfer_basis": s.phase in TRANSFER_PHASES,
                        }
                        for s in steps
                    ],
                    "failure_tail": [cell_step(args, c, "upload-partial").argv for c in CELLS],
                },
                indent=2,
            )
        )
        return 0
    load_dotenv()
    if not args.skip_preflight:
        for name, value in (
            ("min-disk-gb", args.min_disk_gb),
            ("per-pod-quota-gb", args.per_pod_quota_gb),
        ):
            if value is None or not math.isfinite(value) or value <= 0:
                parser.error(f"runtime requires positive --{name} from the verified volume sizing")
    retry_s = float(os.environ.get("EPM_HF_RETRY_BUDGET_S", "1800"))
    if not math.isfinite(retry_s) or retry_s <= 0:
        parser.error("EPM_HF_RETRY_BUDGET_S must be positive for the transfer timeout contract")
    limits = transfer_limits(args, steps, retry_s)
    env = child_environment(args)
    root = args.out_root / "generic" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Runner(args, env, limits)
        previous_signals = {
            s: signal.signal(s, handle_signal) for s in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            write_text_atomic(args.pid_file, f"{os.getpid()}\n")
            runner.log(
                f"[phase=dispatch] mode={args.mode} pid={os.getpid()} pid_file={args.pid_file}"
            )
            runner.report("running", 0, preflight_evidence=args.preflight_evidence)
            try:
                for step in steps:
                    runner.run(step)
                    if step.expected_rc == RC_PILOT_PAUSE:
                        validate_pilot(args)
                rc = RC_PILOT_PAUSE if args.mode == "fit-pilot" else 0
                status = "pilot_paused" if rc else "phase_complete"
                report = runner.report(status, rc)
                runner.sentinel(report, gate="fit_pilot" if rc else None)
                if not rc:
                    # Only this wrapper-mode terminal reaches the main log.
                    runner.log("[phase=done] wrapper mode complete; rank analysis remains separate")
                else:
                    runner.log("[phase=fit_pilot] one layer preserved; VM owner must review timing")
                return rc
            except Exception as exc:
                failure_chain = "".join(traceback.format_exception(exc))
                runner.log(f"[phase=durability_tail] {type(exc).__name__}: {exc}")
                tail_errors = []
                sub = (
                    "smoke_cap_long"
                    if args.mode in {"smoke", "capture-pilot"}
                    else "cells_cap_long"
                )
                for cell in CELLS:
                    if not (root / sub / cell).is_dir():
                        continue
                    try:
                        runner.run(cell_step(args, cell, "upload-partial"), durability_tail=True)
                    except Exception as tail_exc:
                        tail_errors.append("".join(traceback.format_exception(tail_exc)))
                        runner.log(f"partial upload failed for {cell}: {tail_exc}")
                rc = exc.rc if isinstance(exc, (PhaseFailure, WorkloadSignal)) else 1
                if rc < 0:
                    rc = 128 - rc
                report = runner.report(
                    "halted", rc, reason_chain=failure_chain, durability_errors=tail_errors
                )
                runner.sentinel(
                    report, gate="smoke_work_fence" if rc == RC_WORK_FENCE else "phase_failure"
                )
                runner.log(f"[phase=halted] rc={rc}; verify persistence and resolve on the VM")
                return rc
        finally:
            for signum, handler in previous_signals.items():
                signal.signal(signum, handler)
            runner.main_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
