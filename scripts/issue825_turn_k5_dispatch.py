"""Run the real K5 smoke, endpoint generation/capture, and verified persistence."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import psutil  # noqa: E402
from issue825_turn_k5_archive import stage, upload  # noqa: E402
from issue825_turn_k5_gpu import atomic_json, read_jsonl, sha256  # noqa: E402

from explore_persona_space.backends.artifacts import write_completion_sentinel  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from explore_persona_space.orchestrate.secret_scrub import scrub_bytes  # noqa: E402

PREFIX = "issue825_turn_k5_pilot_20260914"


def gpu_ids() -> list[str]:
    """Respect an allocated visibility list before querying the whole machine."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [item.strip() for item in visible.split(",") if item.strip()]
    else:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        ids = result.stdout.split()
    if not ids or ids == ["-1"]:
        raise RuntimeError("no allocated GPU available")
    return ids[:2]


def sample_resources(pid: int, gpu: str) -> dict:
    """Sample recursive process RSS and device-wide GPU memory; tolerate exited children."""
    try:
        parent_process = psutil.Process(pid)
        processes = [parent_process, *parent_process.children(recursive=True)]
    except psutil.NoSuchProcess:
        processes = []
    rss, live = 0, []
    for process in processes:
        try:
            rss += process.memory_info().rss
            live.append(process.pid)
        except psutil.NoSuchProcess:
            # Children can terminate between recursive discovery and memory inspection.
            continue
    result = subprocess.run(
        ["nvidia-smi", "--id", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    values = result.stdout.split()
    if len(values) != 1:
        raise ValueError("allocated GPU did not return exactly one memory observation")
    gpu_mib = float(values[0])
    if not math.isfinite(gpu_mib) or gpu_mib < 0:
        raise ValueError("invalid GPU memory observation")
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "pids": live,
        "aggregate_process_rss_bytes": rss,
        "device_gpu_memory_used_bytes": int(gpu_mib * 1024**2),
    }


def wait_with_resources(child, gpu: str, path: Path) -> tuple[int, dict]:
    """Keep fresh whole-worker resource observations while waiting on the exact child."""
    peak_rss, peak_gpu, samples = 0, 0, 0
    while child.poll() is None:
        observation = sample_resources(child.pid, gpu)
        samples += 1
        peak_rss = max(peak_rss, observation["aggregate_process_rss_bytes"])
        peak_gpu = max(peak_gpu, observation["device_gpu_memory_used_bytes"])
        report = {
            "status": "running",
            "latest": observation,
            "samples": samples,
            "sample_interval_s": 5,
            "sampled_peak_aggregate_process_rss_bytes": peak_rss,
            "sampled_peak_device_gpu_memory_used_bytes": peak_gpu,
            "scope": "child plus recursive descendants RSS; all memory on allocated GPU",
        }
        atomic_json(path, report)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
    report = {
        "status": "complete",
        "exit_code": child.returncode,
        "samples": samples,
        "sample_interval_s": 5,
        "sampled_peak_aggregate_process_rss_bytes": peak_rss,
        "sampled_peak_device_gpu_memory_used_bytes": peak_gpu,
        "scope": "sampled child plus recursive descendants RSS; allocated GPU device memory",
        "ended_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(path, report)
    return child.returncode, report


def run_phase(root: Path, panel: Path, model: str, gpu: str, phase: str) -> dict:
    """Run one fresh process and require its fresh validated completion summary."""
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=gpu,
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        NUMEXPR_NUM_THREADS="8",
        NUMPY_MADVISE_HUGEPAGE="0",
        MALLOC_ARENA_MAX="2",
        MALLOC_MMAP_THRESHOLD_="131072",
    )
    log = root / "logs" / f"{model}_{phase}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    command = [
        sys.executable,
        "scripts/issue825_turn_k5_gpu.py",
        "--phase",
        phase,
        "--model",
        model,
        "--panel",
        str(panel),
        "--out",
        str(root),
    ]
    with log.open("a") as handle:
        child = subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT
        )
        atomic_json(
            log.with_suffix(".process.json"),
            dict(pid=child.pid, started=started, command=command, gpu=gpu),
        )
        print(f"{model} {phase} pid={child.pid} log={log}", flush=True)
        try:
            code, resources = wait_with_resources(child, gpu, log.with_suffix(".resources.json"))
        except Exception as exc:
            # A broken sampler cannot leave a phase orphaned; stop this exact process tree.
            try:
                process = psutil.Process(child.pid)
                processes = [*process.children(recursive=True), process]
            except psutil.NoSuchProcess:
                processes = []
            for process in processes:
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    continue
            _, alive = psutil.wait_procs(processes, timeout=30)
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    continue
            child.wait()
            atomic_json(
                log.with_suffix(".resource_failure.json"),
                {
                    "status": "failed",
                    "error": str(exc),
                    "ended": time.time(),
                },
            )
            raise
    atomic_json(
        log.with_suffix(".exit.json"), dict(exit_code=code, ended=time.time(), started=started)
    )
    if code:
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 64 * 1024))
            tail, _ = scrub_bytes(handle.read())
        for line in tail.split(b"\n")[-120:]:
            print("[child-log] " + line[:2000].decode("utf-8", errors="replace"), flush=True)
        raise RuntimeError(f"{model} {phase} exited {code}; inspect {log}")
    path = root / phase / model / "summary.json"
    if path.stat().st_mtime < started:
        raise RuntimeError(f"stale completion summary: {path}")
    result = json.loads(path.read_text())
    if result["status"] != "complete":
        raise RuntimeError(f"incomplete {model} {phase}")
    return dict(wall_s=time.time() - started, summary=result, resources=resources)


def run_models(
    root: Path, panel: Path, ids: list[str], *, archive_subpath: str | None = None
) -> dict:
    """Parallelize model lanes across realized GPUs; phases use fresh processes."""
    models = ["instruct", "pretrained"]
    remote_subpath = archive_subpath if archive_subpath is not None else root.name

    def lane(model, gpu):
        """Verify raw-answer durability before the capture allocator can start."""
        generated = run_phase(root, panel, model, gpu, "gen")
        receipt = root / "phase_receipts" / f"{model}_generation.json"
        upload(root / "gen" / model, f"{PREFIX}/gpu/{remote_subpath}/gen/{model}", "text", receipt)
        generation_archive = json.loads(receipt.read_text())
        if generation_archive["status"] != "verified":
            raise RuntimeError("raw-answer upload was not verified before capture")
        captured = run_phase(root, panel, model, gpu, "capture")
        return {"gen": generated, "capture": captured, "generation_archive": generation_archive}

    if len(ids) == 1:
        return {model: lane(model, ids[0]) for model in models}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {m: pool.submit(lane, m, ids[i]) for i, m in enumerate(models)}
        results, errors = {}, []
        for model, future in futures.items():
            try:
                results[model] = future.result()
            except Exception as exc:
                # Keep the sibling lane alive long enough to persist its work, then fail.
                errors.append(f"{model}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))
        return results


def smoke_panel_rows(rows: list[dict]) -> list[dict]:
    """Choose eight ordered length quantiles in each real corpus, including both extremes."""
    sources = sorted({row["source"] for row in rows})
    if len(sources) != 2 or len({row["conv_id"] for row in rows}) != len(rows):
        raise ValueError("smoke requires two sources and unique conversation IDs")
    chosen = []
    for source in sources:
        ordered = sorted(
            (row for row in rows if row["source"] == source),
            key=lambda row: (max(row["prompt_lengths"].values()), row["conv_id"]),
        )
        if len(ordered) < 8:
            raise ValueError(f"fewer than eight smoke candidates in source {source}")
        positions = [round(i * (len(ordered) - 1) / 7) for i in range(8)]
        chosen.extend(ordered[i] for i in positions)
    if len(chosen) != 16 or len({row["conv_id"] for row in chosen}) != 16:
        raise AssertionError("smoke quantiles did not produce 16 unique conversations")
    return chosen


def validate_model_coverage(results: dict, panel: list[dict], minimum: int) -> dict:
    """Require consistent planned counts and a common complete-conversation panel."""
    planned_ids = {row["conv_id"] for row in panel}
    complete = []
    for model in ("instruct", "pretrained"):
        gen = results[model]["gen"]["summary"]
        cap = results[model]["capture"]["summary"]
        ids = cap["complete_conversation_ids"]
        expected = len(panel) * 2 * 5
        if (
            gen["status"] != "complete"
            or cap["status"] != "complete"
            or gen["counts"]["draws"] != expected
            or cap["n_selected_conversations"] != len(panel)
            or cap["n_expected_draws"] != expected
            or cap["n_captured_draws"] + cap["n_excluded_draws"] != expected
            or len(set(ids)) != len(ids)
            or not set(ids) <= planned_ids
            or cap["n_complete_conversations"] != len(ids)
            or cap["n_captured_draws"] < len(ids) * 10
        ):
            raise RuntimeError(f"{model} planned-versus-realized coverage is inconsistent")
        complete.append(set(ids))
    common = sorted(set.intersection(*complete))
    if len(common) < minimum:
        raise RuntimeError(
            f"only {len(common)} common complete conversations; need at least {minimum}"
        )
    return {"planned_n": len(panel), "common_n": len(common), "common_ids": common}


def project_smoke(smoke: dict, full_n: int, smoke_n: int, width: int) -> dict:
    """Scale measured processing only; count loading/startup once per production phase."""
    if full_n < smoke_n or smoke_n <= 0 or width not in (1, 2):
        raise ValueError("invalid projection population/width")
    phases, lane_seconds = {}, {}
    for model in ("instruct", "pretrained"):
        phases[model] = {}
        for phase in ("gen", "capture"):
            measured = smoke[model][phase]
            summary = measured["summary"]
            load = float(summary["model_load_s"])
            processing = float(summary["processing_s"])
            elapsed, wall = float(summary["elapsed_s"]), float(measured["wall_s"])
            if not all(math.isfinite(v) for v in (load, processing, elapsed, wall)):
                raise ValueError("nonfinite smoke timing")
            if min(load, processing) <= 0 or elapsed < load + processing - 0.1:
                raise ValueError("smoke must measure a fresh model load and actual processing")
            if wall < elapsed - 0.1:
                raise ValueError("process wall time is shorter than reported phase elapsed time")
            fixed = max(wall - processing, load)
            predicted = fixed + processing * full_n / smoke_n
            phases[model][phase] = {
                "model_load_s": load,
                "processing_s": processing,
                "fixed_startup_load_teardown_s": fixed,
                "processing_scale": full_n / smoke_n,
                "projected_s": predicted,
            }
        lane_seconds[model] = sum(v["projected_s"] for v in phases[model].values())
    wall = max(lane_seconds.values()) if width == 2 else sum(lane_seconds.values())
    return {
        "phases": phases,
        "projected_active_gpu_h": sum(lane_seconds.values()) / 3600,
        "projected_wall_h": wall / 3600,
        "projected_allocated_gpu_h": width * wall / 3600,
        "gpu_width": width,
        "includes_model_load_in_extrapolation": False,
        "method": "fixed startup/load/teardown once plus processing scaled by conversations",
        "utc": datetime.now(UTC).isoformat(),
    }


def archive_outputs(work: Path, receipts: Path, *, require_complete: bool) -> dict:
    """Preserve only produced trees and metadata; attempt every available tree on failure."""
    errors, completed, skipped = [], {}, []
    for name in ("smoke", "production"):
        root = work / name
        for kind, suffixes in (
            ("text", {".json", ".jsonl", ".log", ".txt"}),
            ("tensors", {".npz", ".npy"}),
        ):
            if not any(p.is_file() and p.suffix in suffixes for p in root.rglob("*")):
                if require_complete:
                    errors.append(f"missing {name} {kind} outputs")
                else:
                    skipped.append(f"{name}/{kind}: no produced files")
                continue
            receipt = receipts / f"{name}_{kind}.json"
            try:
                upload(root, f"{PREFIX}/gpu/{name}", kind, receipt)
                completed[f"{name}_{kind}"] = json.loads(receipt.read_text())
            except Exception as exc:
                # Persist the failure and continue salvaging the independent sibling trees.
                errors.append(f"{name}/{kind}: {type(exc).__name__}: {exc}")
    metadata = work / "metadata"
    atomic_json(
        metadata / "archive_status.json",
        {
            "status": "failed" if errors else "verified",
            "produced_archives": completed,
            "errors": errors,
            "skipped_empty_trees": skipped,
            "verified_at": datetime.now(UTC).isoformat(),
        },
    )
    try:
        upload(metadata, f"{PREFIX}/gpu/metadata", "text", receipts / "metadata_text.json")
        completed["metadata_text"] = json.loads((receipts / "metadata_text.json").read_text())
    except Exception as exc:
        errors.append(f"metadata/text: {type(exc).__name__}: {exc}")
    atomic_json(
        receipts / "archive_attempt.json",
        {
            "status": "failed" if errors else "verified",
            "errors": errors,
            "archive_names": sorted(completed),
            "ended_at": datetime.now(UTC).isoformat(),
        },
    )
    try:
        final_receipt = receipts.parent / "receipts_archive.json"
        upload(receipts, f"{PREFIX}/receipts", "text", final_receipt)
        receipt_archive = json.loads(final_receipt.read_text())
    except Exception as exc:
        errors.append(f"receipts/text: {type(exc).__name__}: {exc}")
        receipt_archive = None
    if errors:
        atomic_json(metadata / "archive_failure.json", {"status": "failed", "errors": errors})
        raise RuntimeError("artifact persistence failed: " + "; ".join(errors))
    return {"archives": completed, "receipts_archive": receipt_archive}


def read_back(path: Path) -> dict:
    """Read a write-once envelope, tolerating the VM drain's atomic .processed rename."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return json.loads(path.with_name(path.name + ".processed").read_text())


def finish(out: Path, archived: dict, started: float, width: int) -> None:
    """Write/read terminal sentinels only after every artifact archive is verified."""
    required = {
        "smoke_text",
        "smoke_tensors",
        "production_text",
        "production_tensors",
        "metadata_text",
    }
    if set(archived["archives"]) != required:
        raise RuntimeError("completion requires all five declared output archives")
    records = [*archived["archives"].values(), archived["receipts_archive"]]
    if not records or any(record is None or record["status"] != "verified" for record in records):
        raise RuntimeError("cannot publish completion without verified archives")
    payload = {
        "status": "complete",
        "stage": "gpu-generation-capture",
        "cpu_analysis_pending": True,
        "started": started,
        "ended": time.time(),
        "gpu_width": width,
        "archive_receipts": archived,
    }
    # GCP's startup tail overwrites EPS_SENTINEL_PATH with a minimal done record.
    # Keep this separate detailed completion record as well as the standard path.
    local = write_completion_sentinel(
        sentinel_path=out / "gpu_complete.json", issue=825, extra=payload
    )
    if read_back(local) != {"phase": "done", "issue": 825, **payload}:
        raise RuntimeError("local completion sentinel read-back mismatch")
    standard = os.environ.get("EPS_SENTINEL_PATH")
    if standard and Path(standard).resolve() != local.resolve():
        path = write_completion_sentinel(sentinel_path=standard, issue=825, extra=payload)
        if read_back(path)["phase"] != "done" or read_back(path)["issue"] != 825:
            raise RuntimeError("backend completion sentinel read-back mismatch")
    logs = Path(os.environ.get("EPM825_SENTINEL_DIR", "/workspace/logs"))
    envelope = {
        "sentinel_schema_version": 1,
        "kind": "epm:progress",
        "version": 1,
        "task_id": 825,
        "by": "issue825-turn-k5-dispatch",
        "gate": "phase",
        "blocks_pipeline": False,
        "ts": datetime.now(UTC).isoformat(),
        "note": json.dumps(
            {
                "followup_label": "turn-k5-pilot-20260914",
                "status": "GPU stage complete; CPU fitting and analysis pending",
                "receipts_archive": archived["receipts_archive"],
                "completion_path": str(local),
            }
        ),
    }
    path = logs / f"issue-825-epm_progress-{time.time_ns()}.json"
    atomic_json(path, envelope)
    if read_back(path) != envelope:
        raise RuntimeError("pod progress envelope read-back mismatch")
    # This workload promises GPU artifacts only; VM-owned result commits follow later.
    okay = os.environ.get("EPS_DELIVERABLES_OK_PATH")
    if okay:
        atomic_json(
            Path(okay), {"status": "verified", "receipts_archive": archived["receipts_archive"]}
        )


def main() -> None:
    """Stage exact inputs, smoke both real models, execute and archive all outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(Path("/workspace")):
        raise RuntimeError("GPU output root must be on /workspace")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "gpu_complete.json").exists():
        raise RuntimeError("this output root already has a completion sentinel; use a new root")
    if shutil.disk_usage(args.out).free < 100 * 1024**3:
        raise RuntimeError("need at least 100 GiB free scratch before model staging")
    ids = gpu_ids()
    started = time.time()
    work, receipts = args.out / "work", args.out / "receipts"
    metadata = work / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(exist_ok=True)
    atomic_json(
        metadata / "launch.json",
        {
            "pid": os.getpid(),
            "started": started,
            "gpu_ids": ids,
            "utc": datetime.now(UTC).isoformat(),
            "config_sha256": sha256(args.config),
            "input_pin": json.loads(args.config.read_text()),
            "code": as_metadata_dict(git_provenance(), phase="gpu-dispatch"),
        },
    )
    try:
        print("[phase=stage]", flush=True)
        stage(args.config, work / "inputs")
        panel = work / "inputs/panel.jsonl"
        rows = read_jsonl(panel)
        if len(rows) != 1000:
            raise RuntimeError("production panel must contain the declared 1,000 conversations")
        smoke_rows = smoke_panel_rows(rows)
        smoke_panel = work / "inputs/smoke_panel.jsonl"
        smoke_panel.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in smoke_rows))
        atomic_json(
            metadata / "panel_selection.json",
            {
                "selected_n": len(rows),
                "panel_sha256": sha256(panel),
                "smoke_ids": [r["conv_id"] for r in smoke_rows],
                "smoke_rows": [{k: v for k, v in r.items() if k != "turns"} for r in smoke_rows],
                "smoke_selection": "eight length quantiles per source, including both extremes",
                "input_manifest": json.loads((work / "inputs/input_manifest.json").read_text()),
            },
        )
        print("[phase=smoke]", flush=True)
        # Smoke is intentionally fresh on every invocation; production remains resumable.
        # Revalidating completed chunks yields zero work and cannot measure throughput.
        smoke_subpath = f"smoke/attempt_{time.time_ns()}"
        smoke = run_models(work / smoke_subpath, smoke_panel, ids, archive_subpath=smoke_subpath)
        atomic_json(metadata / "smoke_result.json", smoke)
        smoke_coverage = validate_model_coverage(smoke, smoke_rows, minimum=6)
        atomic_json(metadata / "smoke_coverage.json", smoke_coverage)
        projection = project_smoke(smoke, len(rows), len(smoke_rows), len(ids))
        atomic_json(metadata / "smoke_projection.json", projection)
        projected = projection["projected_allocated_gpu_h"]
        if projected > 8 or projection["projected_wall_h"] > 4:
            raise RuntimeError(
                f"measured projection {projected:.2f} allocated GPU-h / "
                f"{projection['projected_wall_h']:.2f} wall h exceeds 8 GPU-h or 4 wall h; "
                "inspect phase timing before production"
            )
        print(f"[phase=production] measured projection={projected:.2f} allocated GPU-h", flush=True)
        production = run_models(work / "production", panel, ids)
        coverage = validate_model_coverage(production, rows, minimum=12)
        atomic_json(metadata / "production_coverage.json", coverage)
        atomic_json(
            metadata / "gpu_result.json",
            {
                "status": "gpu_phases_complete",
                "production": production,
                "ended": time.time(),
                "started": started,
                "gpu_ids": ids,
                "cpu_analysis_pending": True,
            },
        )
    except Exception as exc:
        atomic_json(
            metadata / "gpu_failure.json",
            {
                "status": "failed",
                "error": str(exc),
                "type": type(exc).__name__,
                "time": time.time(),
            },
        )
        try:
            archive_outputs(work, receipts, require_complete=False)
        except Exception as archive_error:
            raise RuntimeError(
                f"GPU stage failed ({exc}); "
                f"partial-artifact persistence also failed ({archive_error})"
            ) from archive_error
        raise
    print("[phase=upload]", flush=True)
    archived = archive_outputs(work, receipts, require_complete=True)
    finish(args.out, archived, started, len(ids))
    # workflow-lint: phase-done-reserved standalone GPU-stage dispatcher only
    print("[phase=done]", flush=True)


if __name__ == "__main__":
    main()
