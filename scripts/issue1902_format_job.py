"""Launch paired format pilots and production workers, persisting GPU outputs for CPU fits."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "8"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["MALLOC_ARENA_MAX"] = "2"

import issue1902_format_common as C  # noqa: E402


def verify_source(expected):
    """Use the successful #2054 shallow-ref preflight protocol."""
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert actual == expected
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()
    assert branch != "HEAD"
    subprocess.run(
        [
            "git",
            "fetch",
            "--depth=1",
            "origin",
            f"refs/heads/{branch}:refs/remotes/origin/{branch}",
            "refs/heads/main:refs/remotes/origin/main",
        ],
        check=True,
        timeout=180,
    )
    assert (
        subprocess.check_output(["git", "rev-parse", f"origin/{branch}"], text=True).strip()
        == actual
    )
    print("[phase=preflight_git] verified shallow source and main refs", flush=True)


def validate_prompts(root, model):
    """Validate both complete prompt populations before the first generation/capture pilot."""
    from issue1902_format_gpu import tokenizer_for

    tokenizer, template = tokenizer_for(model)
    manifest = json.loads((root / "manifest.json").read_text())
    family = model.split("_")[0]
    cohort = manifest[family]
    maximum = 8192 if family == "qwen" else 4096
    counts = {}
    for form in ("plain", "chat"):
        lengths = []
        for start in range(0, len(cohort["ids"]), 256):
            batch = cohort["ids"][start : start + 256]
            prompts = [
                C.render_prompt(family, form, cohort["questions"][cid], template) for cid in batch
            ]
            lengths.extend(map(len, tokenizer(prompts, add_special_tokens=False)["input_ids"]))
        fresh_caps = [b["cap"] for b in C.banks(model) if b["fresh"] and b["render"] == form]
        required = max(fresh_caps, default=0)
        assert max(lengths) + required <= maximum, (model, form, max(lengths), required, maximum)
        counts[form] = dict(
            rows=len(lengths), max_prompt_tokens=max(lengths), fresh_answer_budget=required
        )
    path = root / "audits" / f"{model}_all_prompts.json"
    C.write_json(path, dict(status="pass", context_limit=maximum, forms=counts))
    C.upload_many([path], root, C.fingerprint(root))


def children(root, phase, devices, first_chunk):
    """One GPU per isolated process, with immediate failure propagation and bounded cleanup."""
    stop = threading.Event()
    active = {}
    lock = threading.Lock()
    models = list(C.MODELS)

    def worker(slot):
        for model in models[slot :: len(devices)] if phase == "validate" else models:
            command = [
                sys.executable,
                __file__,
                "--phase",
                phase,
                "--root",
                str(root),
                "--model",
                model,
                "--shard",
                str(slot),
                "--shards",
                str(len(devices)),
            ]
            if first_chunk:
                command.append("--first-chunk")
            log = (
                root
                / "logs"
                / f"{phase}_{model}_{slot}_{'pilot' if first_chunk else 'production'}.log"
            )
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as output:
                with lock:
                    if stop.is_set():
                        return
                    child = subprocess.Popen(
                        command,
                        env=dict(
                            os.environ, CUDA_VISIBLE_DEVICES=devices[slot], PYTHONUNBUFFERED="1"
                        ),
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    active[child.pid] = child
                print(
                    f"[phase=worker] {phase} {model} gpu={devices[slot]} pid={child.pid} log={log}",
                    flush=True,
                )
                rc = child.wait()
                if not rc:
                    with lock:
                        active.pop(child.pid)
                if rc:
                    stop.set()
                    raise RuntimeError(f"{phase} {model} failed rc={rc}; inspect {log}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, slot) for slot in range(len(devices))]
        while True:
            if stop.is_set() or any(f.done() and f.exception() is not None for f in futures):
                with lock:
                    stop.set()
                    owned = list(active.values())
                for child in owned:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        print(f"[phase=cancel] pgid={child.pid} already exited", flush=True)
                for child in owned:
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print(f"[phase=cancel] pid={child.pid} exceeded TERM grace", flush=True)
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        print(f"[phase=cancel] pgid={child.pid} drained", flush=True)
                    child.wait(timeout=30)
                break
            if all(f.done() for f in futures):
                break
            print(
                f"[phase={phase}] active_workers={sum(not f.done() for f in futures)}/{len(devices)}",
                flush=True,
            )
            time.sleep(5)
        for future in futures:
            future.result()


def pilot_report(root):
    """Project full work from actual production-sized first chunks for every bank."""
    manifest = json.loads((root / "manifest.json").read_text())
    records = []
    for model in C.MODELS:
        multiplier = len(manifest[model.split("_")[0]]["ids"]) / C.CHUNK
        for bank in C.banks(model):
            if bank["fresh"]:
                path = root / "banks" / model / bank["name"] / "chunk_00000.timing.json"
                records.append(dict(json.loads(path.read_text()), projected_multiplier=multiplier))
            for form in ("plain", "chat"):
                path = root / "captures" / model / bank["name"] / form / "chunk_00000.timing.json"
                records.append(dict(json.loads(path.read_text()), projected_multiplier=multiplier))
    hours = sum(r["seconds"] * r["projected_multiplier"] for r in records) / 3600
    report = dict(
        status="pass",
        projected_gpu_hours=hours,
        timings=records,
        nominal_gpu_hour_basis=96,
        projection_excludes_model_startup=True,
        scope="Natural generation formats with equal within-family caps and native turn stops",
        smoke_blind_spots=["Full-cohort ridge fits run on a separate CPU worker"],
    )
    C.write_json(root / "pilot_complete.json", report)
    C.upload_many([root / "pilot_complete.json"], root, C.fingerprint(root))
    print(f"[phase=pilot] projected_gpu_hours={hours:.2f}", flush=True)
    assert hours <= 192, "Pilot exceeds 2x provisional 96 GPU-hour basis; inspect before expanding"
    return report


def aggregate(root):
    """Reduce verified draw tensors to CPU inputs and publish explicit coverage diagnostics."""
    import numpy as np
    from issue1902_format_gpu import save_npz
    import issue1902_common as parent

    manifest = json.loads((root / "manifest.json").read_text())
    fp = C.fingerprint(root)
    files = []
    inventory = {"contexts": {}, "targets": {}}
    coverage = []
    qualitative = []
    rng = np.random.default_rng(19022054)
    for model in C.MODELS:
        family = model.split("_")[0]
        ids = manifest[family]["ids"]
        sample_ids = set(rng.choice(ids, size=40, replace=False).tolist())
        offsets = range(0, len(ids), C.CHUNK)
        for form in ("plain", "chat"):
            chunks = []
            for offset in offsets:
                path = root / "contexts" / model / form / f"chunk_{offset:05d}.npz"
                assert C.complete(path, fp)
                with np.load(path, allow_pickle=False) as p:
                    assert p["ids"].tolist() == ids[offset : offset + C.CHUNK]
                    chunks.append(p["x"])
            path = root / "aggregates" / f"x_{model}_{form}.npz"
            save_npz(
                path,
                ids=np.array(ids),
                x=np.concatenate(chunks),
                fold_of=np.array(manifest[family]["fold_of"]),
            )
            inventory["contexts"][model + "/" + form] = str(path.relative_to(root))
            files.append(path)
        for bank in C.banks(model):
            cap_count = empty_count = repeat_count = 0
            for offset in offsets:
                rawpath = root / "banks" / model / bank["name"] / f"chunk_{offset:05d}.json"
                assert C.complete(rawpath, fp)
                rows = C.read_raw(rawpath)
                cap_count += sum(r["finish_reason"] == "length" for r in rows)
                empty_count += sum(not r["answer"].strip() for r in rows)
                repeat_count += sum(parent.has_repetition_loop(r["answer"]) for r in rows)
                qualitative.extend(
                    dict(r, model=model, bank=bank["name"])
                    for r in rows
                    if r["id"] in sample_ids and r["draw"] == 0
                )
            for form in ("plain", "chat"):
                targets, valids = [], []
                for offset in offsets:
                    path = (
                        root / "captures" / model / bank["name"] / form / f"chunk_{offset:05d}.npz"
                    )
                    assert C.complete(path, fp)
                    with np.load(path, allow_pickle=False) as p:
                        assert p["ids"].tolist() == ids[offset : offset + C.CHUNK]
                        valid = p["valid"].all(1)
                        y = np.full((len(valid), p["w"].shape[-1]), np.nan, dtype=np.float32)
                        y[valid] = p["w"][valid].astype(np.float32).mean(1)
                        targets.append(y)
                        valids.append(valid)
                path = root / "aggregates" / f"y_{model}_{bank['name']}_{form}.npz"
                save_npz(
                    path, ids=np.array(ids), y=np.concatenate(targets), valid=np.concatenate(valids)
                )
                inventory["targets"][model + "/" + bank["name"] + "/" + form] = str(
                    path.relative_to(root)
                )
                files.append(path)
                coverage.append(
                    dict(
                        model=model,
                        bank=bank["name"],
                        capture_render=form,
                        original_contexts=len(ids),
                        complete5_vectors=int(np.concatenate(valids).sum()),
                        total_draws=5 * len(ids),
                        cap_draws=cap_count,
                        empty_draws=empty_count,
                        repetitive_draws=repeat_count,
                        generation_render=bank["render"],
                        generation_cap=bank["cap"],
                    )
                )
        # Persist a model at a time; the terminal upload is a short pipeline flush.
        C.upload_many(files, root, fp)
        files.clear()
    C.write_json(root / "coverage.json", coverage)
    sample_files = C.write_raw(root / "coherence_sample.json", qualitative)
    C.write_json(
        root / "analysis_inputs.json",
        dict(
            inventory,
            manifest_sha256=C.sha(root / "manifest.json"),
            fingerprint=fp,
            files={str(p.relative_to(root)): C.sha(p) for p in (root / "aggregates").glob("*.npz")},
        ),
    )
    revision = C.upload_many(
        [root / "coverage.json", root / "analysis_inputs.json", *sample_files], root, fp
    )
    return revision, coverage


def main():
    """Run either one isolated phase or the full GPU collection job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["job", "generate", "capture", "validate"], default="job"
    )
    parser.add_argument("--root", type=Path, default=Path("/workspace/issue1902_format"))
    parser.add_argument("--model", choices=list(C.MODELS))
    parser.add_argument("--first-chunk", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--source-sha")
    args = parser.parse_args()
    if args.phase != "job":
        if args.phase == "validate":
            validate_prompts(args.root, args.model)
        else:
            from issue1902_format_gpu import capture, generate

            (generate if args.phase == "generate" else capture)(
                args.root, args.model, args.first_chunk, args.shard, args.shards
            )
        return
    verify_source(args.source_sha)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "explore_persona_space.orchestrate.preflight",
            "--planned-footprint-gb",
            "125",
            "--min-disk",
            "150",
            "--planned-upload-gb",
            "25",
        ],
        check=True,
    )
    runtime = {
        p: importlib.metadata.version(p)
        for p in ["torch", "transformers", "vllm", "numpy", "scipy"]
    }
    assert (
        runtime["torch"].split("+")[0] == "2.8.0"
        and runtime["transformers"] == "4.57.6"
        and runtime["vllm"] == "0.11.0"
    )
    from issue1902_format_stage import stage

    stage(args.root)
    C.write_json(args.root / "runtime.json", runtime)
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        inherited.split(",")
        if inherited
        else subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        ).split()
    )
    assert devices
    print(f"[phase=devices] realized_gpus={len(devices)}", flush=True)
    # Each complete model snapshot is reused between fresh generation and capture children.
    from huggingface_hub import snapshot_download

    for model, (mid, revision, _, _) in C.MODELS.items():
        print(f"[phase=prefetch] {model}", flush=True)
        C.retry_transient(
            lambda mid=mid, revision=revision: snapshot_download(
                mid, revision=revision, max_workers=4
            ),
            what=f"prefetch pinned {model}",
        )
    children(args.root, "validate", devices, True)
    children(args.root, "generate", devices, True)
    children(args.root, "capture", devices, True)
    pilot = pilot_report(args.root)
    children(args.root, "generate", devices, False)
    children(args.root, "capture", devices, False)
    revision, coverage = aggregate(args.root)
    report = dict(
        status="gpu_collection_complete",
        input_revision=revision,
        hf_prefix=C.PREFIX,
        source_sha=args.source_sha,
        coverage=coverage,
        pilot_gpu_hours=pilot["projected_gpu_hours"],
        followup_label="Qwen-OLMo-format-reconciliation",
        cpu_fits_pending=True,
    )
    C.write_json(args.root / "gpu_complete.json", report)
    C.upload_many(
        [
            args.root / "gpu_complete.json",
            args.root / "runtime.json",
            *sorted((args.root / "logs").glob("*.log")),
        ],
        args.root,
        C.fingerprint(args.root),
    )
    C.write_json(
        Path("/workspace/logs") / f"issue-1902-epm_results-{time.time_ns()}.json",
        dict(
            sentinel_schema_version=1,
            kind="epm:results",
            version=1,
            task_id=1902,
            note=json.dumps(report),
            blocks_pipeline=False,
        ),
    )
    from explore_persona_space.backends.artifacts import write_completion_sentinel

    write_completion_sentinel(
        sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=1902, extra=report
    )
    print("[phase=done] Qwen/OLMo GPU outputs uploaded and verified; CPU fits follow", flush=True)


if __name__ == "__main__":
    main()
