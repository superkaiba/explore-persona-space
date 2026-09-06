"""Resumable natural-100k follow-up driver. Explicit phases are pilot-gated.

The owner runs prepare on a CPU pod, data-pilot then data on a GPU pod,
stage/fit-pilot then fits after inspecting production-shape measurements.
No Claude/judge calls, task mutation, pod self-stop, or artifact deletion.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import issue1739_natural_data as data
from scripts import issue1739_natural_inputs as inputs
from explore_persona_space.orchestrate import hub
from explore_persona_space.orchestrate.background_upload import BackgroundStemUploader

HF_PREFIX = "issue1739_natural100k_20260906"
U_GRID = (250, 500, 1000, 2000, 5000, 10000, 18793, 25000, 50000, 100000)


def upload_tree(local: Path, prefix: str) -> dict:
    from huggingface_hub import HfApi

    files = {str(p.relative_to(local)): data.file_sha(p) for p in local.rglob("*") if p.is_file()}
    if not files:
        raise ValueError(f"cannot upload an empty output tree: {local}")
    started = time.monotonic()
    print(f"[upload] start prefix={prefix} files={len(files)}", flush=True)
    url = hub._upload(local, inputs.REPO, "dataset", prefix, raise_on_error=True)
    api = HfApi()
    revision = hub.retry_transient(
        lambda: api.repo_info(inputs.REPO, repo_type="dataset").sha,
        what=f"resolve uploaded {prefix}",
    )
    missing = hub.verify_repo_paths_uploaded(
        api,
        inputs.REPO,
        [f"{prefix}/{p}" for p in files],
        path_in_repo=prefix,
        repo_type="dataset",
        revision=revision,
    )
    if missing:
        raise RuntimeError(f"upload incomplete at {prefix}: {missing[:5]}")
    record = {
        "prefix": prefix,
        "revision": revision,
        "url": url,
        "files_sha256": files,
        "wall_s": time.monotonic() - started,
    }
    print(f"[upload] verified prefix={prefix} revision={revision} files={len(files)}", flush=True)
    return record


def input_args(args):
    return argparse.Namespace(
        revision=inputs.REVISION,
        behaviors=list(inputs.BEHAVIORS),
        stage_workers=4,
        store_root=args.root / "reused" / "store",
        main_root=args.root / "reused" / "eval_results",
        tensors_root=args.root / "reused" / "analysis_tensors",
        ood_mirror_root=args.root / "reused" / "ood",
        exclusion_stage_root=args.root / "text_sources",
        materialize_labeling_tars=False,
        labeling_tar_staging_dir=None,
    )


def prepare(args):
    ns = input_args(args)
    exclusions = args.root / "scratch" / "exclusions.jsonl"
    report = inputs.export_exclusions(ns, exclusions)
    data.write_parts(args.root / "exclusions_sharded", list(data.read_rows(exclusions)))
    data.atomic_json(args.root / "exclusions_sharded" / "coverage.json", report)
    data.prepare(
        argparse.Namespace(root=args.root / "pool", exclusions=exclusions, n_candidates=110000)
    )
    return {
        "uploads": [
            upload_tree(args.root / "exclusions_sharded", f"{HF_PREFIX}/exclusions"),
            upload_tree(args.root / "pool" / "prepared", f"{HF_PREFIX}/prepared"),
        ]
    }


def stage_prepared(args):
    if not args.prepared_revision:
        raise ValueError("GPU phases require immutable --prepared-revision from the CPU upload")
    inputs._revision(argparse.Namespace(revision=args.prepared_revision))
    token = os.environ.get("HF_TOKEN", "")
    inputs._stage_selected(
        f"{HF_PREFIX}/prepared",
        args.root / "pool" / "prepared",
        args.prepared_revision,
        token,
        lambda _: True,
        4,
    )
    return data.load_parts(args.root / "pool" / "prepared")


def child(cmd, log: Path, gpu=None):
    """Fresh framework process; inherited launcher CVD selects its physical GPU."""
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "UV_NO_SYNC": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "MALLOC_ARENA_MAX": "2",
    }
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("a") as stream:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        data.atomic_json(log.with_suffix(".pid.json"), {"pid": proc.pid, "cmd": cmd, "gpu": gpu})
        print(f"[child] pid={proc.pid} gpu={gpu} log={log}", flush=True)
        code = proc.wait()
    # Relationship-scoped cleanup catches reparented engine workers as well.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # The exact child process group has already exited normally.
    record = {
        "pid": proc.pid,
        "rc": code,
        "wall_s": time.monotonic() - started,
        "cmd": cmd,
        "gpu": gpu,
        "log": str(log),
    }
    data.atomic_json(log.with_suffix(".exit.json"), record)
    if code != 0:
        with log.open() as stream:
            tail = stream.readlines()[-120:]
        print("".join(tail), flush=True)
        raise RuntimeError(f"child rc={code}, log={log}")
    return record


def gpu_free(gpu: str):
    """Device-level drain check; never compare host NVML PIDs to container PIDs."""
    deadline = time.monotonic() + 180
    while True:
        proc = subprocess.run(
            ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        lines = proc.stdout.strip().splitlines()
        if len(lines) == 1 and lines[0].strip().isdigit() and int(lines[0]) <= 2048:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU {gpu} did not drain below2048MiB: {proc.stdout!r}")
        time.sleep(2)


def data_wave(args, chunks, phase):
    """One long-lived process per GPU per framework, checkpointing each chunk."""
    # Contiguous balanced ranges keep ONE model load per worker in production.
    assignments = [
        chunks[len(chunks) * i // len(args.gpus) : len(chunks) * (i + 1) // len(args.gpus)]
        for i in range(len(args.gpus))
    ]

    def work(gpu, units):
        records = []
        # CLI accepts contiguous ranges; merge adjacent assigned chunk IDs.
        ranges = []
        for unit in sorted(units):
            if ranges and ranges[-1][1] == unit:
                ranges[-1][1] += 1
            else:
                ranges.append([unit, unit + 1])
        for start, end in ranges:
            gpu_free(gpu)
            cmd = [
                sys.executable,
                str(ROOT / "scripts/issue1739_natural_data.py"),
                phase,
                "--root",
                str(args.root / "pool"),
                "--start-chunk",
                str(start),
                "--end-chunk",
                str(end),
                "--batch-size",
                "8",
            ]
            if args.phase == "data-pilot" and phase == "capture":
                cmd.append("--verify-reference")
            records.append(child(cmd, args.root / "logs" / f"{phase}_{start}_{end}.log", gpu))
            gpu_free(gpu)
        return records

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(work, gpu, units)
            for gpu, units in zip(args.gpus, assignments, strict=True)
            if units
        ]
        results = [r for future in as_completed(futures) for r in future.result()]
    return results


def run_data(args, pilot=False):
    pool = stage_prepared(args)
    if pilot:
        # Include the actual longest selected prompt's complete production chunk.
        longest = max(range(100000), key=lambda i: pool[i]["n_prompt_tokens"])
        chunks = sorted(set([0, 1, 2, longest // data.CHUNK]))
        if len(chunks) < len(args.gpus):
            chunks = sorted(set(chunks + [3]))
    else:
        gate = args.root / "pilot_accepted.json"
        if not gate.is_file():
            raise ValueError("full data dispatch requires owner-reviewed pilot_accepted.json")
        chunks = list(range(200))
    generation = data_wave(args, chunks, "generate")
    raw_upload = upload_tree(args.root / "pool/generated", f"{HF_PREFIX}/generated")
    capture = data_wave(args, chunks, "capture")
    cap_upload = upload_tree(args.root / "pool/captured", f"{HF_PREFIX}/captured")
    if pilot:
        return {
            "chunks": chunks,
            "generation": generation,
            "capture": capture,
            "uploads": [raw_upload, cap_upload],
        }
    admitted = sum(
        sum(
            r["admitted"]
            for r in data.load_generated(args.root / "pool/generated" / f"chunk_{i:05d}", pool)
        )
        for i in chunks
    )
    while admitted < 100000:
        start = len(chunks)
        if start >= 220:
            raise ValueError("110k candidate reserve exhausted before100k admitted")
        extra = list(range(start, min(220, start + len(args.gpus))))
        generation.extend(data_wave(args, extra, "generate"))
        raw_upload = upload_tree(args.root / "pool/generated", f"{HF_PREFIX}/generated")
        capture.extend(data_wave(args, extra, "capture"))
        cap_upload = upload_tree(args.root / "pool/captured", f"{HF_PREFIX}/captured")
        chunks.extend(extra)
        admitted += sum(
            sum(
                r["admitted"]
                for r in data.load_generated(args.root / "pool/generated" / f"chunk_{i:05d}", pool)
            )
            for i in extra
        )
    manifest = args.root / "pool/store/manifest.json"
    if not manifest.is_file():
        data.assemble(argparse.Namespace(root=args.root / "pool", n_rows=100000))
    from scripts.issue1739_natural_score import read_natural_manifest, verify_natural_matrices

    meta, _ = read_natural_manifest(args.root / "pool/store")
    verify_natural_matrices(args.root / "pool/store", meta, data.LAYERS)
    store_upload = upload_tree(args.root / "pool/store", f"{HF_PREFIX}/store")
    return {
        "generation": generation,
        "capture": capture,
        "n_admitted": admitted,
        "uploads": [raw_upload, cap_upload, store_upload],
    }


def stage(args):
    ns = input_args(args)
    token = os.environ.get("HF_TOKEN", "")
    result = {}
    for behavior in args.behaviors:
        result[behavior] = inputs.stage_behavior(ns, behavior, token)
        relative = f"eval_results/issue_1739/{behavior}/arm_results/all_arms_spearman.json"
        source = ROOT / relative
        expected = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        actual = subprocess.run(
            ["git", "hash-object", str(source)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != expected:
            raise ValueError(f"committed frozen summary has modified bytes: {source}")
        target = ns.main_root / behavior / "arm_results/all_arms_spearman.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with data.atomic_replace(target) as tmp:
            shutil.copyfile(source, tmp)
        result[behavior]["frozen_summary_blob"] = expected
    return result


def score_cmd(args, behavior, u, seed):
    ns = input_args(args)
    return [
        sys.executable,
        str(ROOT / "scripts/issue1739_r2v2_score.py"),
        "--natural-u-store",
        str(args.root / "pool/store"),
        "--generic-u",
        str(u),
        "--protocols",
        "B",
        "--map-variants",
        "true",
        "--transfer-preds",
        "--seeds",
        str(seed),
        "--behaviors",
        behavior,
        "--device",
        "cuda",
        "--out-root",
        str(args.root / "results" / f"u{u}"),
        "--store-root",
        str(ns.store_root),
        "--main-root",
        str(ns.main_root),
        "--tensors-root",
        str(ns.tensors_root),
        "--ood-store-root",
        str(ns.ood_mirror_root / inputs.PREFIX),
    ]


def fits(args, pilot=False):
    if not pilot and not (args.root / "fit_pilot_accepted.json").is_file():
        raise ValueError("full fit dispatch requires owner-reviewed fit_pilot_accepted.json")
    grid = (
        [(b, 100000, 0) for b in args.behaviors]
        if pilot
        else [(b, u, seed) for b in args.behaviors for u in U_GRID for seed in range(5)]
    )
    # Bound simultaneous fits by measured RAM; caller's explicit GPU list is the width.
    uploader = BackgroundStemUploader(max_pending=4, name="natural-fit-uploads")
    upload_receipts = []

    def persist_cell(local, prefix):
        receipt = upload_tree(local, prefix)
        upload_receipts.append(receipt)
        relative = prefix.removeprefix(HF_PREFIX + "/results/")
        data.atomic_json(args.root / "upload_receipts" / relative / "verified.json", receipt)

    def worker(gpu, cells):
        records = []
        for behavior, u, seed in cells:
            gpu_free(gpu)
            records.append(
                child(
                    score_cmd(args, behavior, u, seed),
                    args.root / "logs" / f"fit_{behavior}_u{u}_s{seed}.log",
                    gpu,
                )
            )
            local = args.root / "results" / f"u{u}" / behavior / f"seed{seed}"
            prefix = f"{HF_PREFIX}/results/u{u}/{behavior}/seed{seed}"
            uploader.submit(
                lambda local=local, prefix=prefix: persist_cell(local, prefix),
                label=f"{behavior}/u{u}/seed{seed}",
            )
        return records

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(worker, gpu, grid[i :: len(args.gpus)]) for i, gpu in enumerate(args.gpus)
        ]
        records = [r for future in as_completed(futures) for r in future.result()]
    uploader.join()
    return {"cells": records, "uploads": upload_receipts}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["prepare", "data-pilot", "data", "stage", "fit-pilot", "fits"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    p.add_argument(
        "--behaviors", nargs="+", default=list(inputs.BEHAVIORS), choices=inputs.BEHAVIORS
    )
    p.add_argument("--prepared-revision")
    p.add_argument("--sentinel-dir", type=Path, default=Path("/workspace/logs"))
    args = p.parse_args(argv)
    if len(set(args.gpus)) != len(args.gpus):
        p.error("GPU pins must be unique")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report = args.root / "reports" / f"{args.phase}_{run_id}.json"
    pidfile = args.root / "logs" / f"{args.phase}.pid.json"
    data.atomic_json(pidfile, {"pid": os.getpid(), "args": sys.argv, "run_id": run_id})
    started = time.monotonic()
    dispatch = {
        "prepare": prepare,
        "data-pilot": lambda a: run_data(a, True),
        "data": run_data,
        "stage": stage,
        "fit-pilot": lambda a: fits(a, True),
        "fits": fits,
    }
    try:
        result = dispatch[args.phase](args)
        _complete_phase(args, run_id, report, started, result)
    except Exception as error:
        data.atomic_json(
            report,
            {
                "phase": args.phase,
                "status": "failed",
                "pid": os.getpid(),
                "reason_chain": "".join(traceback.format_exception(error)),
                "wall_s": time.monotonic() - started,
            },
        )
        failed_sentinel = (
            args.sentinel_dir / f"issue-1739-natural-{args.phase}-{run_id}-failed.json"
        )
        data.atomic_json(
            failed_sentinel,
            {
                "sentinel_schema_version": 1,
                "kind": "epm:failure",
                "version": 1,
                "task_id": 1739,
                "by": "issue1739-natural-run",
                "ts": run_id,
                "note": f"natural100k phase={args.phase} failed; report={report}",
                "payload": {
                    "phase": args.phase,
                    "rc": 1,
                    "reason_chain": "".join(traceback.format_exception(error)),
                },
            },
        )
        raise


def _complete_phase(args, run_id, report, started, result):
    """Publish the local completion signal only AFTER all durable uploads pass."""
    launch_log = os.environ.get("EPS_NATURAL_LAUNCH_LOG")
    if launch_log:
        sys.stdout.flush()
        sys.stderr.flush()
        target = args.root / "logs" / f"{args.phase}_{run_id}.launcher.log"
        target.parent.mkdir(parents=True, exist_ok=True)
        with data.atomic_replace(target) as tmp:
            shutil.copyfile(launch_log, tmp)
    data.atomic_json(
        report,
        {
            "phase": args.phase,
            "status": "complete",
            "pid": os.getpid(),
            "wall_s": time.monotonic() - started,
            "result": result,
        },
    )
    sentinel = args.sentinel_dir / f"issue-1739-natural-{args.phase}-{run_id}.json"
    snapshot = args.root / "sentinel_snapshots" / args.phase / run_id
    snapshot.mkdir(parents=True, exist_ok=True)
    staged_sentinel = snapshot / sentinel.name
    data.atomic_json(
        staged_sentinel,
        {
            "sentinel_schema_version": 1,
            "kind": "epm:progress",
            "version": 1,
            "task_id": 1739,
            "by": "issue1739-natural-run",
            "ts": run_id,
            "note": f"natural100k phase={args.phase} completed rc=0; report={report}; next phase remains owner-gated",
            "payload": {"phase": args.phase, "rc": 0, "report": str(report)},
        },
    )
    # Logs/exit records and reports are continuation inputs, not disposable scratch.
    upload_tree(args.root / "logs", f"{HF_PREFIX}/run_metadata/{args.phase}/{run_id}/logs")
    upload_tree(args.root / "reports", f"{HF_PREFIX}/run_metadata/{args.phase}/{run_id}/reports")
    upload_tree(snapshot, f"{HF_PREFIX}/run_metadata/{args.phase}/{run_id}/sentinels")
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    with data.atomic_replace(sentinel) as tmp:
        shutil.copyfile(staged_sentinel, tmp)
    print(f"[phase=complete] phase={args.phase} report={report}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)
