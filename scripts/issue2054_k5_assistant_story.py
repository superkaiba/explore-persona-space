"""Complete the two assistant-story cells at K5 using the pinned K3 captures.

The full parent manifest stays unchanged. Each GPU worker runs in its own
process; generation and capture never share a model lifetime. The first
production chunk is a reusable pilot. Raw answers and captures are verified
on the Hub before any downstream analysis starts.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k3 as k3
from scripts import issue2054_k3_artifacts as artifacts
from scripts import issue2054_k5 as k5

OUTPUT_PREFIX = "issue2054_assistant_story_k5"
ASSISTANT_STORY_CELLS = tuple(
    f"conversation_paired_stories_assistant__on_policy__attrib_quoted__{model}"
    for model in k3.MODEL_REVISIONS
)
MANIFEST_SHA256 = "365eb7a5f71735ab47e1e0802c3a790cce90b43a84b13a5c43a9d7b460adb506"


def selection(manifest):
    """Validate the immutable 56-cell parent and select exactly two story cells."""
    if len(manifest["cells"]) != 56:
        raise ValueError("assistant-story extension requires the full 56-cell manifest")
    records = k5.selected(manifest, cells=ASSISTANT_STORY_CELLS)
    if any(r["n"] != 8000 or r["cap"] != 2048 for r in records):
        raise ValueError("assistant-story source population or cap changed")
    return records


def fingerprint(manifest):
    """Bind driver-level checkpoints to both source files and the exact panel."""
    payload = {
        "driver": k3.sha(__file__),
        "cells": {
            r["cell"]: k5.fingerprint(manifest, r["cell"], cells=ASSISTANT_STORY_CELLS)
            for r in selection(manifest)
        },
        "output_prefix": OUTPUT_PREFIX,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_manifest(root):
    """Open the exact source manifest without editing its cell registry."""
    path = root / "manifest.json"
    if k3.sha(path) != MANIFEST_SHA256:
        raise RuntimeError("full parent manifest hash changed")
    manifest = json.loads(path.read_text())
    selection(manifest)
    k5.policy()
    return manifest


def check_source(expected):
    """Refuse a launch whose committed source differs from its explicit pin."""
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if not expected or actual != expected:
        raise RuntimeError(f"assistant-story source SHA mismatch: {actual} != {expected}")
    paths = [str(Path(__file__).relative_to(REPO)), "scripts/issue2054_k5.py"]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *paths], cwd=REPO, check=True)
    return actual


def prepare(root):
    """Restore and verify only the selected cells, keeping the full manifest."""
    manifest, report = k5.prepare(root, cells=ASSISTANT_STORY_CELLS)
    selection(manifest)
    manifest = load_manifest(root)
    selected_path = root / "selected_cells.json"
    k3.atomic_json(selected_path, {"cells": list(ASSISTANT_STORY_CELLS), "parent_cells": 56})
    artifacts.seal_many(
        [
            root / "manifest.json",
            root / "capture_policy.json",
            root / "restore_verified.json",
            selected_path,
        ],
        root,
        fingerprint(manifest),
    )
    return manifest, report


def restore_outputs(root, revision):
    """Restore only receipt-verified own outputs from an immutable Hub revision."""
    from huggingface_hub import HfApi
    from explore_persona_space.orchestrate.hub import retry_transient, stage_hub_file

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("resume requires an immutable 40-character Hub revision")
    prefix = f"{OUTPUT_PREFIX}/{root.name}"
    entries = retry_transient(
        lambda: list(
            HfApi().list_repo_tree(
                k3.HF_REPO,
                path_in_repo=prefix,
                recursive=True,
                repo_type="dataset",
                revision=revision,
            )
        ),
        what="list assistant-story own checkpoints",
    )
    receipt_paths = [
        e.path
        for e in entries
        if hasattr(e, "size")
        and e.path.endswith(".done.json")
        and e.path != prefix + "/runtime.json.done.json"
    ]
    if not receipt_paths:
        raise RuntimeError("resume revision contains no verified assistant-story checkpoints")

    def restore_one(remote_receipt):
        relative = Path(remote_receipt).relative_to(prefix)
        if ".." in relative.parts or relative.is_absolute():
            raise ValueError("unsafe own-checkpoint path")
        staged_receipt = stage_hub_file(
            k3.HF_REPO,
            remote_receipt,
            root / "restore_receipts" / revision / relative,
            revision=revision,
        )
        receipt = json.loads(staged_receipt.read_text())
        expected = remote_receipt.removesuffix(".done.json")
        if receipt["path"] != expected or not re.fullmatch(r"[0-9a-f]{40}", receipt["revision"]):
            raise RuntimeError("own-checkpoint receipt destination/revision mismatch")
        target = root / Path(expected).relative_to(prefix)
        target = stage_hub_file(k3.HF_REPO, expected, target, revision=receipt["revision"])
        if target.stat().st_size != receipt["size"] or k3.sha(target) != receipt["sha256"]:
            raise RuntimeError(f"own-checkpoint content mismatch: {target}")
        destination = target.with_suffix(target.suffix + ".done.json")
        if destination.exists() and json.loads(destination.read_text()) != receipt:
            raise RuntimeError(
                f"local receipt conflicts with requested resume revision: {destination}"
            )
        k3.atomic_json(destination, receipt)
        if not k3.complete(target, receipt["fingerprint"]):
            raise RuntimeError(f"restored checkpoint does not open through consumer: {target}")
        return receipt

    with ThreadPoolExecutor(max_workers=4) as pool:
        restored = list(pool.map(restore_one, receipt_paths))
    report = {
        "revision": revision,
        "restored": len(restored),
        "checked_at": time.time(),
        "prefix": prefix,
    }
    k3.atomic_json(root / "own_restore_verified.json", report)
    k3.log(f"[phase=own_restore_verified] {json.dumps(report)}")
    return report


def allocated_devices(environ):
    """Use allocated GPU IDs, never the full host enumeration on SLURM."""
    value = environ.get("CUDA_VISIBLE_DEVICES")
    if not value and environ.get("SLURM_JOB_ID"):
        value = environ.get("SLURM_JOB_GPUS") or environ.get("SLURM_STEP_GPUS")
        if not value:
            raise RuntimeError("SLURM launch must provide allocated GPU IDs")
    if not value:
        value = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
            )
            .replace("\n", ",")
            .rstrip(",")
        )
    devices = [v.strip() for v in value.split(",") if v.strip()]
    if not 1 <= len(devices) <= 2 or len(devices) != len(set(devices)):
        raise RuntimeError(f"expected one or two allocated GPUs, got {devices}")
    return devices


def resource_preflight(root, devices):
    """Check the output mount and every allocated capture GPU before staging."""
    from explore_persona_space.orchestrate.preflight import (
        assert_out_root_headroom,
        preflight_check,
    )

    root.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(root)
    total_gb = stat.f_blocks * stat.f_frsize / 1e9
    if total_gb < 180:
        raise RuntimeError(f"planned 200 GB boot disk is absent: {root} has {total_gb:.1f} GB")
    assert_out_root_headroom(root, 70, phase="assistant_story_preamble")
    for device in devices:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--id",
                device,
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        total, free = map(float, output.strip().split(","))
        if total < 38000 or free < max(36000, 0.85 * total):
            raise RuntimeError(
                f"capture requires a free 40 GB-class GPU: {device}: {output.strip()}"
            )
    report = preflight_check(
        min_disk_gb=70,
        min_gpu_free_mb=36000,
        planned_footprint_gb=70,
        required_env_vars=[],
    )
    k3.log(report.summary())
    if not report.ok:
        raise RuntimeError("assistant-story preflight failed")


def worker_command(root, source_sha, model, stage, first_chunks):
    """Build the exact stage CLI used by both pilot and production workers."""
    if stage not in ("generate", "capture") or model not in k3.MODEL_REVISIONS:
        raise ValueError("invalid assistant-story worker")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--out-root",
        str(root),
        "--source-sha",
        source_sha,
        "--model",
        model,
    ]
    if first_chunks:
        command.append("--first-chunks")
    return command


def stop_owned(active, lock):
    """Drain only process groups created by this driver's current worker run."""
    with lock:
        owned = list(active.values())
    for child in owned:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            k3.log(f"[phase=worker_cancel] pgid={child.pid} already absent")
    deadline = time.monotonic() + 30
    for child in owned:
        try:
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            k3.log(f"[phase=worker_cancel] pgid={child.pid} exceeded TERM grace")
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            k3.log(f"[phase=worker_cancel] pgid={child.pid} drained")
        child.wait()


def run_workers(root, source_sha, devices, *, first_chunks):
    """Run model-owned generate/capture processes and cancel peers on failure."""
    models = list(k3.MODEL_REVISIONS)
    active, lock, stopped = {}, threading.Lock(), threading.Event()
    run_id = str(time.time_ns())
    records = []

    def worker(slot):
        for model in models[slot :: len(devices)]:
            for stage in ("generate", "capture"):
                label = "pilot" if first_chunks else "production"
                log = root / "logs" / f"{run_id}_{label}_{stage}_{model}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                started = time.time()
                with log.open("w") as output:
                    with lock:
                        if stopped.is_set():
                            return
                        child = subprocess.Popen(
                            worker_command(root, source_sha, model, stage, first_chunks),
                            env=dict(
                                os.environ, CUDA_VISIBLE_DEVICES=devices[slot], PYTHONUNBUFFERED="1"
                            ),
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                        active[child.pid] = child
                    record = {
                        "started": started,
                        "pid": child.pid,
                        "model": model,
                        "stage": stage,
                        "first_chunks": first_chunks,
                        "device": devices[slot],
                        "source_sha": source_sha,
                        "run_id": run_id,
                        "log": str(log.relative_to(root)),
                    }
                    status = log.with_suffix(".status.json")
                    k3.atomic_json(status, record)
                    code = child.wait()
                    record.update(
                        finished=time.time(), exit_code=code, seconds=time.time() - started
                    )
                    k3.atomic_json(status, record)
                    if code:
                        stopped.set()
                    else:
                        with lock:
                            active.pop(child.pid)
                artifacts.seal_many([log, status], root, k3.sha(__file__))
                if code:
                    raise RuntimeError(f"worker failed: {model}/{stage} exit={code}; log={log}")
                with lock:
                    records.append(record)

    def interrupted(signum, _frame):
        stopped.set()
        raise InterruptedError(f"assistant-story driver received signal {signum}")

    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            jobs = [pool.submit(worker, slot) for slot in range(len(devices))]
            try:
                while not all(job.done() for job in jobs):
                    if stopped.is_set() or any(job.done() and job.exception() for job in jobs):
                        stopped.set()
                        stop_owned(active, lock)
                        break
                    k3.log(
                        f"[phase=workers] checked_at={time.time():.3f} active={sum(not j.done() for j in jobs)} pilot={first_chunks}"
                    )
                    time.sleep(5)
                for job in jobs:
                    job.result()
            finally:
                stopped.set()
                stop_owned(active, lock)
    finally:
        for sig, old in old_handlers.items():
            signal.signal(sig, old)
    return records


def validate_coverage(coverage, *, first_chunks):
    """Report coverage honestly and reject structurally incomplete cell panels."""
    if {c["cell"] for c in coverage} != set(ASSISTANT_STORY_CELLS) or len(coverage) != 2:
        raise RuntimeError("assistant-story coverage is missing a cell")
    for row in coverage:
        expected = k3.CHUNK if first_chunks else 8000
        if row["original_rows"] != expected:
            raise RuntimeError("assistant-story original row coverage changed")
        if len(row["cap_counts_each_draw"]) != 5:
            raise RuntimeError("assistant-story coverage lacks five draw cap counts")
        if row["complete_five_rows"] < 0.9 * expected:
            raise RuntimeError("assistant-story lost more than 10% of contexts to empty draws")
        if any(count / expected > 0.02 for count in row["cap_counts_each_draw"][3:]):
            raise RuntimeError(
                "new assistant-story draw cap-hit rate exceeds 2%; inspect before expansion or interpretation"
            )
        if not first_chunks and row["complete_five_rows"] * 0.8 <= 3584:
            raise RuntimeError("assistant-story lost the ambient-fit sample regime")


def inventory(root, manifest):
    """Validate every produced data checkpoint and return its immutable receipt."""
    receipts = []
    for record in selection(manifest):
        cell = record["cell"]
        fp = k5.fingerprint(manifest, cell, cells=ASSISTANT_STORY_CELLS)
        expected = [root / f"k{k}" / f"{cell}.npz" for k in k5.COUNTS]
        for offset in range(0, record["n"], k3.CHUNK):
            raw = root / "raw" / cell / f"chunk_{offset:05d}.json"
            expected.extend([raw, raw.with_suffix(".timing.json")])
            index = json.loads(raw.read_text())
            expected.extend(raw.parent / name for name in index["shards"])
            expected.extend(
                [
                    root / "captures" / cell / f"chunk_{offset:05d}.npz",
                    root / "capture_audits" / cell / f"chunk_{offset:05d}.json",
                ]
            )
        for path in expected:
            if not k3.complete(path, fp):
                raise RuntimeError(f"missing verified output {path}")
            receipt = json.loads(path.with_suffix(path.suffix + ".done.json").read_text())
            if receipt["path"] != f"{OUTPUT_PREFIX}/{root.name}/{path.relative_to(root)}":
                raise RuntimeError(f"checkpoint destination changed: {path}")
            receipts.append(receipt)
    return receipts


def pilot_report(root, manifest, coverage, workers):
    """Project full generation and capture from the actual two-cell pilot."""
    validate_coverage(coverage, first_chunks=True)
    reports = []
    for record in selection(manifest):
        cell = record["cell"]
        model = cell.split("__")[-1]
        timing = json.loads((root / "raw" / cell / "chunk_00000.timing.json").read_text())
        if timing["contexts"] != 256 or timing["answers"] != 512 or timing["seconds"] <= 0:
            raise RuntimeError("invalid assistant-story production-shape pilot timing")
        measured = {r["stage"]: r for r in workers if r["model"] == model}
        if set(measured) != {"generate", "capture"}:
            raise RuntimeError("pilot worker timing coverage incomplete")
        capture_pattern = r"\[phase=capture\] " + re.escape(cell) + r" offset=0 elapsed=([0-9.]+)s"
        capture_times = re.findall(capture_pattern, (root / measured["capture"]["log"]).read_text())
        timing_source = measured["capture"]["log"]
        if not capture_times:
            # A recovery can resume after capture verification but before the
            # pilot summary. Reuse a sealed real timing, not the skip-only wall.
            for previous in sorted((root / "logs").glob(f"*_pilot_capture_{model}.log")):
                if k3.complete(previous, k3.sha(__file__)):
                    candidate = re.findall(capture_pattern, previous.read_text())
                    if candidate:
                        capture_times, timing_source = candidate, str(previous.relative_to(root))
                        break
        if len(capture_times) != 1:
            raise RuntimeError("capture pilot lacks a verified actual first-chunk timing")
        capture_seconds = float(capture_times[0])
        chunks = math.ceil(record["n"] / k3.CHUNK)
        reports.append(
            {
                "cell": cell,
                "chunks": chunks,
                "timing": timing,
                "generation_process_seconds": measured["generate"]["seconds"],
                "capture_process_seconds": measured["capture"]["seconds"],
                "projected_generation_compute_seconds": timing["seconds"] * chunks,
                "capture_chunk_seconds": capture_seconds,
                "capture_timing_source": timing_source,
                "projected_capture_chunk_seconds": capture_seconds * chunks,
                "conservative_generation_wall_seconds": measured["generate"]["seconds"] * chunks,
                "conservative_capture_wall_seconds": measured["capture"]["seconds"] * chunks,
                "timing_note": "Generation seconds measure model generation. Capture log seconds include capture and serialization. Process walls also include model initialization and verified uploads; their difference is combined overhead, not a separated upload benchmark.",
            }
        )
    return {
        "status": "pass",
        "checked_at": time.time(),
        "cells": reports,
        "coverage": coverage,
        "projected_generation_compute_gpu_hours": sum(
            r["projected_generation_compute_seconds"] for r in reports
        )
        / 3600,
        "conservative_generation_capture_gpu_hours": sum(
            r["conservative_generation_wall_seconds"] + r["conservative_capture_wall_seconds"]
            for r in reports
        )
        / 3600,
        "smoke_blind_spots": [
            "full-cohort dense fits",
            "rare cap or empty failures outside the first 256 contexts",
        ],
    }


def complete_generation(root, manifest, *, actual, started, restore, devices, pilot_only):
    """Reuse a verified complete bank without resealing it or changing its inventory."""
    fp = fingerprint(manifest)
    complete = root / "generation_complete.json"
    if k3.complete(complete, fp):
        generated = json.loads(complete.read_text())
        if generated["source_sha"] != actual or generated["status"] != "complete":
            raise RuntimeError("completed generation source/status mismatch")
        validate_coverage(generated["coverage"], first_chunks=False)
        inventory_path = root / "output_inventory.json"
        if (
            not k3.complete(inventory_path, fp)
            or k3.sha(inventory_path) != generated["inventory_sha256"]
        ):
            raise RuntimeError("completed generation inventory changed")
        if json.loads(inventory_path.read_text()) != inventory(root, manifest):
            raise RuntimeError("completed generation receipt set changed")
        k3.log("[phase=generation_reused] verified complete K5 bank; preserving original receipts")
        return generated
    pilot_path = root / "pilot_complete.json"
    if k3.complete(pilot_path, fp):
        pilot = json.loads(pilot_path.read_text())
        for record in selection(manifest):
            cell = record["cell"]
            cell_fp = k5.fingerprint(manifest, cell, cells=ASSISTANT_STORY_CELLS)
            for directory, suffix in (
                ("raw", "json"),
                ("captures", "npz"),
                ("capture_audits", "json"),
            ):
                if not k3.complete(root / directory / cell / f"chunk_00000.{suffix}", cell_fp):
                    raise RuntimeError("persisted pilot is missing a verified data checkpoint")
    else:
        k5.phase_headroom(root, manifest, "generate", cells=ASSISTANT_STORY_CELLS)
        workers = run_workers(root, actual, devices, first_chunks=True)
        pilot_coverage = k5.aggregate(
            root, manifest, first_chunks=True, cells=ASSISTANT_STORY_CELLS
        )
        pilot = pilot_report(root, manifest, pilot_coverage, workers)
        k3.atomic_json(pilot_path, pilot)
        artifacts.seal_many([pilot_path], root, fp)
    artifacts.seal_many([root / "runtime.json"], root, fp)
    k3.log(f"[phase=pilot_complete] {json.dumps(pilot)}")
    if pilot_only:
        return pilot
    for stage in ("generate", "capture", "aggregate"):
        k5.phase_headroom(root, manifest, stage, cells=ASSISTANT_STORY_CELLS)
    run_workers(root, actual, devices, first_chunks=False)
    coverage = k5.aggregate(root, manifest, cells=ASSISTANT_STORY_CELLS)
    validate_coverage(coverage, first_chunks=False)
    receipts = inventory(root, manifest)
    k3.atomic_json(root / "output_inventory.json", receipts)
    artifacts.seal_many([root / "output_inventory.json"], root, fp)
    generated = {
        "status": "complete",
        "stage": "generation_capture_aggregation",
        "source_sha": actual,
        "started": started,
        "finished": time.time(),
        "parent_revision": k5.PARENT_REV,
        "hf_prefix": f"{OUTPUT_PREFIX}/{root.name}",
        "coverage": coverage,
        "restore": restore,
        "selected_cells": list(ASSISTANT_STORY_CELLS),
        "manifest_sha256": MANIFEST_SHA256,
        "inventory_sha256": k3.sha(root / "output_inventory.json"),
        "outputs_revision": json.loads((root / "output_inventory.json.done.json").read_text())[
            "revision"
        ],
    }
    k3.atomic_json(root / "generation_complete.json", generated)
    artifacts.seal_many([root / "generation_complete.json"], root, fp)
    return generated


def verify_analysis(root, manifest, generation):
    """Require the analyzed result and every declared file to have verified receipts."""
    from scripts import issue2054_k5_assistant_story_analysis as analysis

    fp = analysis.fingerprint(manifest, generation)
    complete = root / "analysis" / "analysis_complete.json"
    if not k3.complete(complete, fp):
        raise RuntimeError("analysis lacks a verified completion checkpoint")
    report = json.loads(complete.read_text())
    if (
        report["status"] != "complete"
        or report["source_sha"] != generation["source_sha"]
        or report["analysis_fingerprint"] != fp
    ):
        raise RuntimeError("analysis completion provenance mismatch")
    for key in ("results", "inventory"):
        relative = Path(report[key + "_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "analysis":
            raise RuntimeError("analysis result path escaped its output directory")
        path = root / relative
        if not k3.complete(path, fp) or k3.sha(path) != report[key + "_sha256"]:
            raise RuntimeError(f"analysis {key} content/receipt mismatch")
    declared = json.loads((root / report["inventory_path"]).read_text())
    if declared["analysis_fingerprint"] != fp or declared["source_sha"] != generation["source_sha"]:
        raise RuntimeError("analysis inventory provenance mismatch")
    for row in declared["files"]:
        path, done = root / row["path"], root / row["receipt_path"]
        for candidate in (path, done):
            if not candidate.resolve().is_relative_to((root / "analysis").resolve()):
                raise RuntimeError("analysis inventory escaped its output directory")
        if (
            not k3.complete(path, fp)
            or k3.sha(path) != row["sha256"]
            or path.stat().st_size != row["size"]
            or k3.sha(done) != row["receipt_sha256"]
            or json.loads(done.read_text()) != row["receipt"]
            or row["receipt"]["path"] != f"{OUTPUT_PREFIX}/{root.name}/{row['path']}"
        ):
            raise RuntimeError(f"analysis inventory member failed verification: {row['path']}")
    return report


def run_job(root, args):
    """Run a verified reusable pilot, complete K5, then optional downstream analysis."""
    from scripts.issue2054_k3_job import prepare_git

    started = time.time()
    actual = check_source(args.source_sha)
    k3.atomic_json(
        root / "runtime.json",
        {"started": started, "pid": os.getpid(), "source_sha": actual, "stage": "preflight"},
    )
    prepare_git(REPO, actual)
    devices = allocated_devices(os.environ)
    resource_preflight(root, devices)
    if args.resume_revision:
        restore_outputs(root, args.resume_revision)
    k3.atomic_json(
        root / "runtime.json",
        {"started": started, "pid": os.getpid(), "source_sha": actual, "devices": devices},
    )
    manifest, restore = prepare(root)
    generated = complete_generation(
        root,
        manifest,
        actual=actual,
        started=started,
        restore=restore,
        devices=devices,
        pilot_only=args.pilot_only,
    )
    if args.pilot_only:
        return generated
    fp = fingerprint(manifest)
    generation_receipt = json.loads((root / "generation_complete.json.done.json").read_text())
    if args.analysis:
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/issue2054_k5_assistant_story_analysis.py"),
                "--out-root",
                str(root),
            ],
            check=True,
        )
        verify_analysis(root, manifest, generated)
    final = dict(
        generated,
        analysis_requested=args.analysis,
        started=started,
        finished=time.time(),
        generation_revision=generation_receipt["revision"],
    )
    k3.atomic_json(root / "job_complete.json", final)
    artifacts.seal_many([root / "job_complete.json"], root, fp)
    if os.environ.get("EPS_SENTINEL_PATH"):
        from explore_persona_space.backends.artifacts import write_completion_sentinel

        write_completion_sentinel(
            sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=2054, extra=final
        )
    return final


def main():
    """Dispatch only the assistant-story extension; never the original K5 fit grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("job", "prepare", "generate", "capture", "aggregate"), required=True
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--model", choices=list(k3.MODEL_REVISIONS))
    parser.add_argument("--first-chunks", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument(
        "--resume-revision", help="Immutable Hub revision of this extension's own verified outputs"
    )
    args = parser.parse_args()
    root = args.out_root.resolve()
    if root.name != "production_v1":
        raise ValueError("assistant-story output root must be named production_v1")
    k3.PREFIX = OUTPUT_PREFIX
    check_source(args.source_sha)
    if args.stage == "job":
        try:
            run_job(root, args)
        except BaseException:
            failure = {
                "failed_at": time.time(),
                "source_sha": args.source_sha,
                "traceback": traceback.format_exc(),
            }
            k3.atomic_json(root / "run_failed.json", failure)
            k3.log(f"[phase=run_failed] {json.dumps(failure)}")
            raise
        return
    if args.stage == "prepare":
        prepare(root)
        return
    manifest = load_manifest(root)
    if args.stage == "aggregate":
        k5.aggregate(root, manifest, first_chunks=args.first_chunks, cells=ASSISTANT_STORY_CELLS)
        return
    if args.model is None:
        raise ValueError("generation/capture requires an explicit checkpoint")
    args.shard, args.shards = 0, 1
    function = k5.generate if args.stage == "generate" else k5.capture
    function(root, manifest, args, cells=ASSISTANT_STORY_CELLS)


if __name__ == "__main__":
    main()
