#!/usr/bin/env python3
"""Generate and capture the five new #1901 training answers on allocated GPUs.

The controller stages one immutable input bank, then runs independent GPU
pipelines with fresh parity/generation/capture processes. A full first chunk
is captured before each worker advances. No local test substitutes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()
import numpy as np  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.background_upload import BackgroundStemUploader  # noqa: E402
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: E402
from explore_persona_space.orchestrate.secret_scrub import scan_file  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1901_training_k10"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SEEDS = (47, 48, 49, 50, 51)
CHUNK = 500
DIM = 3584
SETTINGS = {
    "model": MODEL,
    "model_revision": MODEL_REVISION,
    "layer": 19,
    "dimension": DIM,
    "seeds": list(SEEDS),
    "engine_seed": 42,
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 1024,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.60,
    "max_num_seqs": 64,
    "chunk_rows": CHUNK,
    "capture_rows": 32,
    "capture_token_budget": 32768,
    "capture_convention": "parent-full-template-retok-span-incl-eot-tail",
}


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=1) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def chunks(rows, size=CHUNK):
    if not rows or size < 1:
        raise ValueError("chunks requires rows and a positive chunk size")
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def partition(rows, rank, width):
    assert 1 <= width <= 4 and 0 <= rank < width
    # Round-robin contexts spread source-order length variation over every GPU.
    return rows[rank::width]


def verify_remote_file(path, info):
    assert info.size == path.stat().st_size, str(path)
    if info.lfs:
        assert info.lfs.sha256 == sha_file(path), str(path)
    else:
        content = path.read_bytes()
        expected = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        assert info.blob_id == expected, str(path)


def text_upload_paths(path, limit=9_500_000, target=9_000_000):
    """Persist large JSON/logs as line fragments with the shared Hub manifest schema."""
    if path.suffix not in (".json", ".jsonl", ".log") or path.stat().st_size <= limit:
        return [path]
    parts, buffer, nbytes = [], [], 0

    def flush():
        nonlocal buffer, nbytes
        if buffer:
            part = path.with_name(f"{path.stem}.part{len(parts):04d}")
            part.write_bytes(b"".join(buffer))
            parts.append(part)
            buffer, nbytes = [], 0

    with path.open("rb") as stream:
        for line in stream:
            assert len(line) <= target, f"oversized single text line: {path}"
            if nbytes + len(line) > target:
                flush()
            buffer.append(line)
            nbytes += len(line)
    flush()
    manifest = path.with_name(f"{path.stem}.manifest.json")
    write_json(
        manifest,
        {
            "source": path.name,
            "source_sha256": sha_file(path),
            "parts": [p.name for p in parts],
            "sha256": {p.name: sha_file(p) for p in parts},
        },
    )
    assert hashlib.sha256(b"".join(p.read_bytes() for p in parts)).hexdigest() == sha_file(path)
    return [manifest, *parts]


def upload_files(root, paths, prefix):
    """One commit for an explicit file set, then exact path/size/content checks."""
    from huggingface_hub import CommitOperationAdd

    paths = sorted({p for source in paths for p in text_upload_paths(Path(source))})
    assert paths
    names = [f"{prefix}/{p.relative_to(root).as_posix()}" for p in paths]
    print(f"[upload] expected={names}", flush=True)
    api = HfApi()
    commit = hub.retry_transient(
        lambda: api.create_commit(
            repo_id=REPO,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p))
                for n, p in zip(names, paths, strict=True)
            ],
            commit_message="#1901 training K10 checkpoint",
        ),
        what="training-K checkpoint upload",
    )
    infos = hub.retry_transient(
        lambda: api.get_paths_info(REPO, names, repo_type="dataset", revision=commit.oid),
        what="training-K checkpoint verification",
    )
    by_name = {info.path: info for info in infos}
    assert set(by_name) == set(names), set(names) - set(by_name)
    for name, path in zip(names, paths, strict=True):
        verify_remote_file(path, by_name[name])
    return {
        "revision": commit.oid,
        "files": [
            {"path": n, "sha256": sha_file(p), "size": p.stat().st_size}
            for n, p in zip(names, paths, strict=True)
        ],
    }


def stage_inputs(out, revision):
    assert re.fullmatch(r"[0-9a-f]{40}", revision), "input revision must be an immutable SHA"
    target = out / "inputs"

    def fetch(name):
        return Path(
            hub.retry_transient(
                lambda: hf_hub_download(
                    REPO,
                    f"{PREFIX}/inputs/{name}",
                    repo_type="dataset",
                    revision=revision,
                    local_dir=out / "staging",
                ),
                what=f"stage input {name}",
            )
        )

    source = fetch("manifest.json")
    manifest = read_json(source)
    assert manifest["train_rows"] == 19000 and manifest["test_rows"] == 1000
    assert manifest["model_revision"] == MODEL_REVISION and manifest["layer"] == 19
    # GPU workers need only prompts and parity; train/test multi-GB arrays stay off this lane.
    names = ["prompts.json", "parity.json", "parity.npz"]
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        info = manifest["outputs"][name]
        dest = target / name
        if not dest.exists():
            if name.endswith(".json"):
                hub.stage_sharded_text(
                    REPO, f"{PREFIX}/inputs/{name}", dest, repo_type="dataset", revision=revision
                )
            else:
                cached = fetch(name)
                os.link(cached, dest)
        assert dest.stat().st_size == info["size"] and sha_file(dest) == info["sha256"], name
    if (target / "manifest.json").exists():
        assert sha_file(target / "manifest.json") == sha_file(source)
    else:
        os.link(source, target / "manifest.json")
    return manifest


def load_context(args):
    manifest = read_json(args.out / "inputs" / "manifest.json")
    for name in ("prompts.json", "parity.json", "parity.npz"):
        path = args.out / "inputs" / name
        assert sha_file(path) == manifest["outputs"][name]["sha256"], name
    rows = read_json(args.out / "inputs" / "prompts.json")["train"]
    assert len(rows) == 19000 and len({r["ci"] for r in rows}) == len(rows)
    for row in rows:
        assert hashlib.sha256(row["prompt"].encode()).hexdigest() == row["prompt_sha256"]
    recipe = {
        **SETTINGS,
        "code_sha256": {
            name: sha_file(ROOT / name)
            for name in (
                "scripts/issue1901_training_k10_gpu.py",
                "scripts/issue1901_k10_capture.py",
                "scripts/issue1482_kresample.py",
                "scripts/issue1482_error_analysis.py",
                "scripts/issue1901_avgpool_scaleup.py",
                "src/explore_persona_space/eval/generation.py",
            )
        },
        "input_revision": args.input_revision,
        "input_manifest_sha256": sha_file(args.out / "inputs" / "manifest.json"),
        "ordered_ci_sha256": digest_json([r["ci"] for r in rows]),
        "world_size": args.world_size,
    }
    if (args.out / "recipe.json").exists():
        assert read_json(args.out / "recipe.json") == recipe, "stale run recipe"
    return rows, recipe


def run_prefix(recipe):
    return f"{PREFIX}/capture_{digest_json(recipe)[:16]}"


def worker_root(args):
    return args.out / "workers" / f"worker{args.rank:02d}"


def validate_generation(doc, rows, recipe, seed, chunk):
    assert doc["recipe"] == recipe and doc["seed"] == seed and doc["chunk"] == chunk
    assert [r["ci"] for r in doc["rows"]] == [r["ci"] for r in rows], "generation order changed"
    for actual, expected in zip(doc["rows"], rows, strict=True):
        assert actual["prompt_sha256"] == expected["prompt_sha256"]
        assert isinstance(actual["text"], str) and actual["prompt_token_ids"]
        assert actual["text"] == actual["response"]
        assert actual["token_ids"] and len(actual["token_ids"]) <= SETTINGS["max_tokens"]
        assert all(type(i) is int and i >= 0 for i in actual["token_ids"])
        assert all(type(i) is int and i >= 0 for i in actual["prompt_token_ids"])
    assert doc["cap_hits"] == sum(len(r["token_ids"]) == 1024 for r in doc["rows"])


def validate_capture(path, rows, recipe, seed, generation_sha):
    with np.load(path, allow_pickle=False) as z:
        assert z["V"].shape == (len(rows), DIM) and z["V"].dtype == np.float16
        assert np.isfinite(z["V"]).all()
        assert np.array_equal(z["ci"], [r["ci"] for r in rows]), "capture order changed"
        assert z["n_ans"].shape == (len(rows),) and np.all(z["n_ans"] > 0)
        assert int(z["seed"]) == seed and json.loads(str(z["recipe"])) == recipe
        assert str(z["generation_sha"]) == generation_sha, "generation content changed"


def runtime():
    import torch

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert props.total_memory >= 39 * 1024**3, "capture requires >=40GB-class GPU"
    return {
        "gpu_name": props.name,
        "gpu_bytes": props.total_memory,
        "versions": {p: version(p) for p in ("torch", "transformers", "vllm")},
    }


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = hub.retry_transient(
        lambda: AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION), what="tokenizer"
    )
    model = (
        hub.retry_transient(
            lambda: AutoModelForCausalLM.from_pretrained(
                MODEL, revision=MODEL_REVISION, torch_dtype=torch.bfloat16
            ),
            what="capture model",
        )
        .to("cuda")
        .eval()
    )
    assert model.config.hidden_size == DIM and model.config.num_hidden_layers == 28
    return model, tok


def parity(args):
    from issue1901_k10_capture import capture_rows

    _, recipe = load_context(args)
    machine = runtime()
    root = worker_root(args)
    source = read_json(args.out / "inputs" / "parity.json")["rows"]
    assert len(source) == 128 and len({r["ci"] for r in source}) == 128
    selected = partition(source, args.rank, args.world_size)
    selected = [selected[i] for i in np.linspace(0, len(selected) - 1, 32, dtype=int)]
    with np.load(args.out / "inputs" / "parity.npz") as z:
        by_ci = {int(ci): i for i, ci in enumerate(z["ci"])}
        expected = z["V"][[by_ci[r["ci"]] for r in selected]].astype(np.float64)
    assert all(r["seed"] == 43 for r in selected)
    model, tok = load_model()
    texts = [{**r["generated"], "ci": r["ci"]} for r in selected]
    got, _ = capture_rows(model, tok, selected, texts)
    got = got.astype(np.float64)
    cosine = np.sum(got * expected, axis=1) / (
        np.linalg.norm(got, axis=1) * np.linalg.norm(expected, axis=1)
    )
    relative = np.linalg.norm(got - expected, axis=1) / np.linalg.norm(expected, axis=1)
    assert np.isfinite(cosine).all() and np.isfinite(relative).all()
    evidence = {
        "recipe": recipe,
        "runtime": machine,
        "ci": [r["ci"] for r in selected],
        "cosine": cosine.tolist(),
        "relative_l2": relative.tolist(),
        "passed": bool(cosine.min() >= 0.999),
    }
    write_json(root / "parity.json", evidence)
    upload_files(args.out, [root / "parity.json"], run_prefix(recipe))
    assert evidence["passed"], evidence


def job_chunks(args, rows):
    jobs = [(seed, index, batch) for seed in SEEDS for index, batch in enumerate(chunks(rows))]
    return jobs[:1] if args.pilot else jobs


def phase_context(args):
    all_rows, recipe = load_context(args)
    rows = partition(all_rows, args.rank, args.world_size)
    proof = read_json(worker_root(args) / "parity.json")
    assert proof["passed"] and proof["recipe"] == recipe
    assert proof["runtime"] == runtime(), "runtime changed since parity"
    return rows, recipe, proof["runtime"]


def enqueue_upload(uploader, args, recipe, paths, receipt):
    def persist():
        result = upload_files(args.out, paths, run_prefix(recipe))
        write_json(receipt, result)

    uploader.submit(persist, label=receipt.name)


def generation(args):
    import issue1482_kresample as KR
    from issue1901_avgpool_scaleup import _pipeline_scrub_pre_upload
    from transformers import AutoTokenizer

    from explore_persona_space.eval.generation import create_vllm_engine

    rows, recipe, machine = phase_context(args)
    root = worker_root(args)
    jobs = job_chunks(args, rows)
    todo = [
        j
        for j in jobs
        if not (root / "raw_completions" / f"seed{j[0]}_chunk{j[1]:04d}.json").exists()
    ]
    llm = tok = None
    if todo:
        assert_out_root_headroom(args.out, 2 + len(todo) * 0.02, phase="generation")
        tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        llm = create_vllm_engine(
            MODEL,
            revision=MODEL_REVISION,
            max_model_len=8192,
            seed=42,
            gpu_memory_utilization=0.60,
            max_num_seqs=64,
        )
        assert llm is not None and KR.GEN_MAX_TOKENS == 1024
    with BackgroundStemUploader(max_pending=1) as uploader:
        for seed, index, batch in jobs:
            path = root / "raw_completions" / f"seed{seed}_chunk{index:04d}.json"
            if path.exists():
                doc = read_json(path)
            else:
                rendered = [
                    tok.apply_chat_template(
                        [{"role": "user", "content": r["prompt"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for r in batch
                ]
                start = time.monotonic()
                generated = KR._generate_seed(llm, tok, rendered, seed)
                doc = {
                    "recipe": recipe,
                    "runtime": machine,
                    "seed": seed,
                    "chunk": index,
                    "compute_seconds": time.monotonic() - start,
                    "cap_hits": sum(len(r["token_ids"]) == 1024 for r in generated),
                    "rows": [
                        {
                            "ci": r["ci"],
                            "prompt_sha256": r["prompt_sha256"],
                            "row_idx": i,
                            "src": "distr",
                            "response": g["text"],
                            "original_text_sha256": hashlib.sha256(g["text"].encode()).hexdigest(),
                            **g,
                        }
                        for i, (r, g) in enumerate(zip(batch, generated, strict=True))
                    ],
                }
                write_json(path, doc)
            # Exact inherited #1901 narrow handler, with its live-credential check.
            # Both text aliases are masked identically; original token IDs and
            # text hashes retain provenance; capture uses the disclosed edited text.
            disclosures = _pipeline_scrub_pre_upload(path)
            doc = read_json(path)
            doc["scrub_disclosures"] = doc.get("scrub_disclosures", []) + disclosures
            write_json(path, doc)
            if disclosures:
                print(
                    f"[inherited-jwt-scrub] rank={args.rank} seed={seed} chunk={index} "
                    f"findings={len(disclosures)}; disclosure persisted; capture uses scrubbed text",
                    flush=True,
                )
            validate_generation(doc, batch, recipe, seed, index)
            assert doc["runtime"] == machine, "generation runtime changed"
            if scan_file(path):
                raise RuntimeError(f"Generation secret-scan findings require review: {path}")
            enqueue_upload(
                uploader, args, recipe, [path], root / "receipts" / (path.stem + "_gen.json")
            )
            print(
                f"[generation] rank={args.rank} seed={seed} chunk={index} "
                f"rows={len(batch)} seconds={doc['compute_seconds']:.2f} caps={doc['cap_hits']}",
                flush=True,
            )
        uploader.join()


def capture(args):
    from issue1901_k10_capture import capture_rows

    rows, recipe, machine = phase_context(args)
    root = worker_root(args)
    jobs = job_chunks(args, rows)
    todo = [
        j
        for j in jobs
        if not (root / "analysis_tensors" / f"seed{j[0]}_chunk{j[1]:04d}.npz").exists()
    ]
    model = tok = None
    if todo:
        assert_out_root_headroom(args.out, 2 + len(todo) * CHUNK * DIM * 2 / 1e9, phase="capture")
        model, tok = load_model()
    with BackgroundStemUploader(max_pending=1) as uploader:
        for seed, index, batch in jobs:
            raw = root / "raw_completions" / f"seed{seed}_chunk{index:04d}.json"
            doc = read_json(raw)
            validate_generation(doc, batch, recipe, seed, index)
            generation_sha = sha_file(raw)
            path = root / "analysis_tensors" / f"seed{seed}_chunk{index:04d}.npz"
            timing = path.with_suffix(".json")
            if not path.exists():
                start = time.monotonic()
                values, counts = capture_rows(model, tok, batch, doc["rows"])
                seconds = time.monotonic() - start
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp.npz")
                np.savez(
                    temporary,
                    V=values,
                    n_ans=counts,
                    ci=np.array([r["ci"] for r in batch]),
                    seed=seed,
                    recipe=json.dumps(recipe, sort_keys=True),
                    generation_sha=generation_sha,
                )
                write_json(
                    timing,
                    {
                        "recipe": recipe,
                        "runtime": machine,
                        "generation_sha": generation_sha,
                        "compute_seconds": seconds,
                    },
                )
                temporary.replace(path)
            validate_capture(path, batch, recipe, seed, generation_sha)
            info = read_json(timing)
            assert info["recipe"] == recipe and info["runtime"] == machine
            assert info["generation_sha"] == generation_sha
            enqueue_upload(
                uploader,
                args,
                recipe,
                [path, timing],
                root / "receipts" / (path.stem + "_cap.json"),
            )
            print(
                f"[capture] rank={args.rank} seed={seed} chunk={index} "
                f"rows={len(batch)} seconds={info['compute_seconds']:.2f}",
                flush=True,
            )
        uploader.join()


def completion_payload(root, rows, recipe):
    """Reconcile the fixed Cartesian identity set from validated tensor contents."""
    found, files = set(), []
    for rank in range(recipe["world_size"]):
        own = partition(rows, rank, recipe["world_size"])
        directory = root / "workers" / f"worker{rank:02d}"
        for seed in SEEDS:
            for index, batch in enumerate(chunks(own)):
                raw = directory / "raw_completions" / f"seed{seed}_chunk{index:04d}.json"
                validate_generation(read_json(raw), batch, recipe, seed, index)
                path = directory / "analysis_tensors" / f"seed{seed}_chunk{index:04d}.npz"
                validate_capture(path, batch, recipe, seed, sha_file(raw))
                pairs = {(r["ci"], seed) for r in batch}
                assert not pairs & found
                found.update(pairs)
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": sha_file(path),
                        "rows": len(batch),
                        "seed": seed,
                        "ci": [r["ci"] for r in batch],
                    }
                )
    expected = {(r["ci"], seed) for r in rows for seed in SEEDS}
    assert found == expected and len(found) == len(rows) * len(SEEDS)
    return {
        "schema_version": 1,
        "input_revision": recipe["input_revision"],
        "input_manifest_sha256": recipe["input_manifest_sha256"],
        "recipe": recipe,
        "recipe_sha256": digest_json(recipe),
        "files": files,
        "expected_new_rows": len(expected),
        "realized_new_rows": len(found),
    }


def finish(args, manifest, receipt):
    """Only fresh complete, remotely verified output may emit success sentinels."""
    from explore_persona_space.backends.artifacts import write_completion_sentinel

    assert manifest["expected_new_rows"] == manifest["realized_new_rows"] == 95000
    assert receipt["files"] and re.fullmatch(r"[0-9a-f]{40}", receipt["revision"])
    prefix = run_prefix(manifest["recipe"])
    uploaded = {item["path"]: item["sha256"] for item in receipt["files"]}
    assert uploaded[f"{prefix}/capture_manifest.json"] == sha_file(
        args.out / "capture_manifest.json"
    )
    assert manifest["files"], "no capture outputs declared"
    for item in manifest["files"]:
        assert uploaded[f"{prefix}/{item['path']}"] == item["sha256"]
    assert "EPS_SENTINEL_PATH" in os.environ, "router completion path required"
    note = {
        "round": "training-k10-capture",
        "new_rows": 95000,
        "hf_prefix": run_prefix(manifest["recipe"]),
        "upload_verification": receipt,
        "recipe_sha256": manifest["recipe_sha256"],
        "fitting_status": "pending",
    }
    write_json(
        Path(os.environ.get("EPS_LOG_DIR", "/workspace/logs"))
        / f"issue-1901-epm_results-trainingk10-{time.time_ns()}.json",
        {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 1901,
            "note": json.dumps(note),
            "blocks_pipeline": False,
        },
    )
    write_completion_sentinel(sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=1901, extra=note)
    print("[phase=done] training K10 capture uploaded and verified", flush=True)


def clear_stale_completion(out, sentinel):
    """Archive and remove only this router attempt's completion proof on relaunch."""
    if sentinel.exists():
        previous = read_json(sentinel)
        assert previous["issue"] == 1901, "refusing to clear a different task sentinel"
        destination = out / "resume_history" / f"completion_{time.time_ns()}.json"
        write_json(destination, previous)
        sentinel.unlink()
        print(f"[resume] archived stale attempt completion to {destination}", flush=True)


def reap_process_group(process):
    """Reap this private child session, including vLLM descendants, at phase exit."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # Exact private group already reaped by the child, a normal exit race.
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # Descendants exited between liveness probe and cleanup signal.


def controller(args):
    import torch

    args.out.mkdir(parents=True, exist_ok=True)
    assert "EPS_SENTINEL_PATH" in os.environ, "launch through unified router"
    clear_stale_completion(args.out, Path(os.environ["EPS_SENTINEL_PATH"]))
    cached_model = sum(
        p.stat().st_size
        for p in (args.out / "model_cache").glob("hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/*")
        if p.is_file() and not p.name.endswith(".incomplete")
    )
    pending_model_gb = max(0, 16 - cached_model / 1e9)
    pending_inputs_gb = 0 if (args.out / "inputs" / "manifest.json").exists() else 1
    assert_out_root_headroom(
        args.out, 2 + pending_inputs_gb + pending_model_gb, phase="input-and-model-staging"
    )
    stage_inputs(args.out, args.input_revision)
    width = torch.cuda.device_count()
    assert 1 <= width <= 4, width
    slots = os.environ.get("CUDA_VISIBLE_DEVICES")
    slots = slots.split(",") if slots else [str(i) for i in range(width)]
    assert len(slots) == width
    args.world_size = width
    rows, recipe = load_context(args)
    write_json(args.out / "recipe.json", recipe)
    processes, lock = {}, threading.Lock()
    abort = threading.Event()

    def child(rank, phase, pilot=False):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = slots[rank]
        env["HF_HUB_CACHE"] = str(args.out / "model_cache" / "hub")
        argv = [
            sys.executable,
            __file__,
            "--out",
            str(args.out),
            "--input-revision",
            args.input_revision,
            "--phase",
            phase,
            "--rank",
            str(rank),
            "--world-size",
            str(width),
        ]
        if pilot:
            argv.append("--pilot")
        logpath = args.out / "logs" / f"worker{rank:02d}_{phase}{'_pilot' if pilot else ''}.log"
        logpath.parent.mkdir(exist_ok=True)
        print(f"[worker-phase] rank={rank} phase={phase} pilot={pilot}", flush=True)
        with logpath.open("a") as log:
            cursor = logpath.stat().st_size
            with lock:
                assert not abort.is_set(), "sibling worker failed"
                process = subprocess.Popen(
                    argv,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes[rank] = process
            while process.poll() is None:
                if abort.wait(2):
                    reap_process_group(process)
                    raise RuntimeError("sibling worker failed")
                with logpath.open() as progress:
                    progress.seek(cursor)
                    for line in progress:
                        if line.startswith(("[generation]", "[capture]", "[inherited-jwt-scrub]")):
                            print(line.rstrip(), flush=True)
                    cursor = progress.tell()
            reap_process_group(process)
            with lock:
                del processes[rank]
            if process.returncode:
                abort.set()
                with logpath.open("rb") as failure_log:
                    failure_log.seek(max(0, logpath.stat().st_size - 8192))
                    print(
                        "[child-failure-tail] " + failure_log.read().decode(errors="replace"),
                        flush=True,
                    )
                raise RuntimeError(f"rank={rank} phase={phase} rc={process.returncode}; {logpath}")

    def pipeline(rank):
        try:
            child(rank, "parity")
            child(rank, "generation", True)
            child(rank, "capture", True)
            directory = args.out / "workers" / f"worker{rank:02d}"
            gen = read_json(directory / "raw_completions" / "seed47_chunk0000.json")
            cap = read_json(directory / "analysis_tensors" / "seed47_chunk0000.json")
            assert len(gen["rows"]) == CHUNK, "pilot must have500production rows"
            count = len(partition(rows, rank, width)) * len(SEEDS)
            projection = {
                "rank": rank,
                "pilot_rows": CHUNK,
                "worker_rows": count,
                "generation_seconds": gen["compute_seconds"],
                "capture_seconds": cap["compute_seconds"],
                "projected_compute_hours": (gen["compute_seconds"] + cap["compute_seconds"])
                * count
                / CHUNK
                / 3600,
                "runtime": gen["runtime"],
                "recipe": recipe,
            }
            write_json(directory / "pilot.json", projection)
            upload_files(args.out, [directory / "pilot.json"], run_prefix(recipe))
            print("[pilot] " + json.dumps(projection), flush=True)
            child(rank, "generation")
            child(rank, "capture")
        except BaseException:
            abort.set()
            raise

    print(f"[phase=capture_pipeline] workers={width} prefix={run_prefix(recipe)}", flush=True)
    with ThreadPoolExecutor(max_workers=width) as pool:
        list(pool.map(pipeline, range(width)))
    manifest = completion_payload(args.out, rows, recipe)
    write_json(args.out / "capture_manifest.json", manifest)
    # Final exact-set upload covers raw data, tensors, receipt state, and child logs.
    paths = [
        p
        for p in args.out.rglob("*")
        if p.is_file()
        and p.relative_to(args.out).parts[0] not in ("inputs", "staging", "model_cache")
        and not p.name.endswith((".tmp", ".tmp.npz"))
    ]
    receipt = upload_files(args.out, paths, run_prefix(recipe))
    finish(args, manifest, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-revision", required=True)
    parser.add_argument(
        "--phase", choices=["controller", "parity", "generation", "capture"], default="controller"
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--import-check", action="store_true")
    args = parser.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("import-check: PASS; GPU and cloud transport are not exercised")
        return
    {"controller": controller, "parity": parity, "generation": generation, "capture": capture}[
        args.phase
    ](args)


if __name__ == "__main__":
    main()
