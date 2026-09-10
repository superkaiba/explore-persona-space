"""User-authorized §4.4 K=3 generation/capture; fixed-source controls stay fixed.

Each stage is a separate process. Raw chunks and reduced tensors are uploaded
and verified before a checkpoint can be resumed. No judging or model training.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_capture as capture

HF_REPO = "superkaiba1/explore-persona-space-data"
REVISION = "d3207a181402b42873f5a3120b1d56da7b90f104"
PREFIX = "issue2054_section44_k3_gcp"
VERSION = "v1"
CHUNK = 256
PILOT_ROWS = 256
STOPS = {
    "chat": ["<|im_end|>"],
    "bare_text": ["\nUser:"],
    "attrib_quoted": ['"'],
    "bare_label": ["\n"],
}


def log(message):
    print(message, flush=True)


def sha(path):
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_rows(path):
    with Path(path).open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    ids = [r["conv_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate conv_id in {path}")
    for row in rows:
        s, e = row["answer_start"], row["answer_end"]
        if not 0 <= s <= e <= len(row["final_text"]):
            raise ValueError(f"invalid span: {path}:{row['conv_id']}")
        if row["final_text"][s:e] != row["answer"]:
            raise ValueError(f"answer span mismatch: {path}:{row['conv_id']}")
    return rows


def download(filename, root, revision=REVISION):
    from huggingface_hub import hf_hub_download
    from explore_persona_space.orchestrate.hub import retry_transient

    return Path(
        retry_transient(
            lambda: hf_hub_download(
                HF_REPO,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                local_dir=root / "inputs",
            ),
            what=f"stage {filename}",
        )
    )


def upload(path, root, relative_destination=None):
    from huggingface_hub import HfApi
    from explore_persona_space.orchestrate.hub import retry_transient

    path = Path(path)
    destination = f"{PREFIX}/{root.name}/{relative_destination or path.relative_to(root)}"
    api = HfApi()
    commit = retry_transient(
        lambda: api.upload_file(
            repo_id=HF_REPO,
            repo_type="dataset",
            path_or_fileobj=path,
            path_in_repo=destination,
            commit_message=f"#2054 K3 {path.name}",
        ),
        what=f"upload {destination}",
    )
    info = retry_transient(
        lambda: api.get_paths_info(
            HF_REPO, [destination], repo_type="dataset", revision=commit.oid
        ),
        what=f"verify {destination}",
    )
    if len(info) != 1 or info[0].size != path.stat().st_size:
        raise RuntimeError(f"remote size mismatch: {destination}")
    remote_hash = getattr(getattr(info[0], "lfs", None), "sha256", None)
    if remote_hash is not None and remote_hash != sha(path):
        raise RuntimeError(f"remote SHA256 mismatch: {destination}")
    if remote_hash is None:
        content = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if info[0].blob_id != blob:
            raise RuntimeError(f"remote Git blob mismatch: {destination}")
    return {
        "path": destination,
        "revision": commit.oid,
        "sha256": sha(path),
        "size": path.stat().st_size,
    }


def seal(path, root, fingerprint):
    receipt = upload(path, root)
    receipt["fingerprint"] = fingerprint
    done = path.with_suffix(path.suffix + ".done.json")
    pending = done.with_suffix(".pending")
    atomic_json(pending, receipt)
    upload(pending, root, done.relative_to(root))
    pending.replace(done)


def complete(path, fingerprint):
    done = path.with_suffix(path.suffix + ".done.json")
    if not done.exists():
        return False
    receipt = json.loads(done.read_text())
    if receipt["fingerprint"] != fingerprint or not path.exists() or sha(path) != receipt["sha256"]:
        raise RuntimeError(f"checkpoint fingerprint/content mismatch: {path}")
    return True


def fingerprint(manifest, cell, pilot):
    payload = {
        "version": VERSION,
        "manifest": manifest,
        "cell": cell,
        "pilot_rows": PILOT_ROWS if pilot else None,
        "chunk": CHUNK,
        "temperature": 1.0,
        "top_p": 1.0,
        "stops": STOPS,
        "layer": 19,
        "context": "prefill-alone-last-token",
        "code_sha256": sha(__file__),
        "capture_helper_sha256": sha(capture.__file__),
        "engine": {
            "max_model_len": 8192,
            "dtype": "bfloat16",
            "eager": True,
            "max_num_seqs": 128,
            "kv_fraction": 0.85,
        },
        "capture": {"batch_max": 8, "tokens_max": 8192, "attention": "sdpa"},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def seed(cell, conv_id, draw):
    if draw not in (1, 2):
        raise ValueError("only fresh draws 1 and 2 are allowed")
    return int.from_bytes(
        hashlib.sha256(f"2054-k3|{cell}|{conv_id}|{draw}".encode()).digest()[:4], "little"
    )


def cap_for(cell):
    _, _, form, model = cell.split("__")
    return 4096 if form == "bare_text" or (form == "chat" and model == "qwen2.5-7b") else 2048


def prepare(root):
    from huggingface_hub import HfApi
    import numpy as np

    api = HfApi()
    candidates = list(
        api.list_repo_tree(
            HF_REPO,
            path_in_repo="issue2054_lattice/activations",
            recursive=True,
            repo_type="dataset",
            revision=REVISION,
        )
    )
    # Exact pre-indirect 56-cell lattice; cross-checked against banked fits.
    paths = [x.path for x in candidates if x.path.endswith(".npz") and "__indirect__" not in x.path]
    if len(paths) != 56 or len({Path(p).stem for p in paths}) != 56:
        raise RuntimeError(f"expected banked 56 unique activation cells, got {len(paths)}")
    models = {
        slug: {"id": model, "revision": api.model_info(model).sha}
        for slug, model in capture._MODEL_ID.items()
    }
    cells = []
    for i, path in enumerate(sorted(paths)):
        cell = Path(path).stem
        variant, condition, form, model = cell.split("__")
        if condition not in ("on_policy", "inserted", "cell_c"):
            raise ValueError(cell)
        local = download(path, root)
        with np.load(local, allow_pickle=False) as z:
            ids = [str(c) for c in z["conv_id"]]
            if len(ids) != len(set(ids)) or z["v_A"].shape != (len(ids), 3584):
                raise ValueError(f"banked row/shape mismatch {cell}")
        record = {
            "cell": cell,
            "activation": path,
            "activation_sha256": sha(local),
            "n": len(ids),
            "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        }
        if condition == "on_policy":
            raw = f"issue2054_lattice/on_policy/{model}/{variant}/on_policy_{variant}__{form}.jsonl"
            raw_local = download(raw, root)
            rows = read_rows(raw_local)
            if set(ids) != {r["conv_id"] for r in rows}:
                raise RuntimeError(f"raw/activation population mismatch: {cell}")
            record.update(raw=raw, raw_sha256=sha(raw_local), cap=cap_for(cell))
        cells.append(record)
        log(f"[phase=prepare] cell={i + 1}/56 {cell} n={len(ids)}")
    if sum("raw" in c for c in cells) != 24:
        raise RuntimeError("on-policy lattice must have 24 cells")
    fold = download("issue2054_lattice/shared_fold_map.json", root)
    if sha(fold) != "4ab1839a0e8c5e8705147cbb529b2df36975ac46b987fe71ab3f919265e4c39e":
        raise RuntimeError("fold map differs from banked 56-cell pooled map")
    manifest = {
        "revision": REVISION,
        "models": models,
        "cells": cells,
        "fold_map": str(fold.relative_to(root)),
        "fold_sha256": sha(fold),
    }
    atomic_json(root / "manifest.json", manifest)
    seal(root / "manifest.json", root, VERSION)


def selected(manifest, args):
    cells = [c for c in manifest["cells"] if "raw" in c]
    cells.sort(key=lambda c: (c["cell"].split("__")[-1], c["cell"]))
    return cells[args.shard :: args.shards]


def banked_rows(root, record, pilot):
    path = root / "inputs" / record["raw"]
    if sha(path) != record["raw_sha256"]:
        raise RuntimeError(f"changed source: {path}")
    rows = read_rows(path)
    return rows[:PILOT_ROWS] if pilot else rows


def save_raw(path, rows, root, fp):
    """Keep each JSON text object below the upload-policy threshold."""
    shards = []
    batch = []
    size = 2
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False).encode()
        if len(encoded) > 8_000_000:
            raise ValueError("single generation exceeds the text-shard limit")
        if batch and size + len(encoded) > 8_000_000:
            shard = path.with_name(f"{path.stem}_part{len(shards):03d}.json")
            atomic_json(shard, batch)
            seal(shard, root, fp)
            shards.append(shard.name)
            batch = []
            size = 2
        batch.append(row)
        size += len(encoded) + 1000
    if batch:
        shard = path.with_name(f"{path.stem}_part{len(shards):03d}.json")
        atomic_json(shard, batch)
        seal(shard, root, fp)
        shards.append(shard.name)
    atomic_json(path, {"shards": shards, "rows": len(rows)})
    seal(path, root, fp)


def load_raw(path, fp):
    index = json.loads(path.read_text())
    rows = []
    for name in index["shards"]:
        shard = path.parent / name
        if not complete(shard, fp):
            raise RuntimeError(f"unverified raw shard: {shard}")
        rows.extend(json.loads(shard.read_text()))
    if len(rows) != index["rows"]:
        raise RuntimeError("raw shard count mismatch")
    return rows


def generate(root, manifest, args):
    from vllm import LLM, SamplingParams

    engine = None
    loaded = None
    for record in selected(manifest, args):
        cell = record["cell"]
        fp = fingerprint(manifest, cell, args.pilot)
        rows = banked_rows(root, record, args.pilot)
        model = manifest["models"][cell.split("__")[-1]]
        # Each model uses a separate process; no teardown assumptions.
        if args.model and cell.split("__")[-1] != args.model:
            continue
        if loaded != model:
            if engine is not None:
                raise RuntimeError("generation worker must be scoped to one model")
            engine = LLM(
                model=model["id"],
                revision=model["revision"],
                tokenizer_revision=model["revision"],
                dtype="bfloat16",
                max_model_len=8192,
                gpu_memory_utilization=0.85,
                enforce_eager=True,
                max_num_seqs=128,
                tensor_parallel_size=1,
                seed=137,
            )
            loaded = model
        form = cell.split("__")[2]
        for offset in range(0, len(rows), CHUNK):
            path = root / "raw" / cell / f"chunk_{offset:05d}.json"
            if complete(path, fp):
                continue
            batch = rows[offset : offset + CHUNK]
            requests = [(row, draw) for row in batch for draw in (1, 2)]
            prompts = [row["final_text"][: row["answer_start"]] for row, _ in requests]
            params = [
                SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=record["cap"],
                    stop=STOPS[form],
                    seed=seed(cell, row["conv_id"], draw),
                )
                for row, draw in requests
            ]
            t0 = time.monotonic()
            outputs = engine.generate(prompts, params, use_tqdm=False)
            generated = []
            for (row, draw), output in zip(requests, outputs, strict=True):
                answer = output.outputs[0]
                item = dict(row)
                prefix = row["final_text"][: row["answer_start"]]
                suffix = row["final_text"][row["answer_end"] :]
                item.update(
                    answer=answer.text,
                    final_text=prefix + answer.text + suffix,
                    answer_end=len(prefix) + len(answer.text),
                    answer_len_chars=len(answer.text),
                    draw=draw,
                    seed=seed(cell, row["conv_id"], draw),
                    finish_reason=answer.finish_reason,
                    stop_reason=answer.stop_reason,
                    generated_token_count=len(answer.token_ids),
                    max_tokens_budget=record["cap"],
                )
                generated.append(item)
            save_raw(path, generated, root, fp)
            log(
                f"[phase=generation] {cell} rows={min(offset + CHUNK, len(rows))}/{len(rows)} fresh={len(generated)} seconds={time.monotonic() - t0:.1f}"
            )


def forward_vectors(model, tokenizer, rows, *, prompt_only=False):
    import numpy as np
    import torch

    vectors = np.full((len(rows), 3584), np.nan, dtype=np.float16)
    legacy_context = np.full_like(vectors, np.nan)
    positions = []
    for index, row in enumerate(rows):
        if prompt_only:
            ids = tokenizer(row["final_text"][: row["answer_start"]], add_special_tokens=False)[
                "input_ids"
            ]
            pos = {
                "input_ids": ids,
                "answer_lo": len(ids) - 1,
                "answer_hi": len(ids),
                "v_C_pos": len(ids) - 1,
            }
        else:
            pos = capture._compute_positions(tokenizer, row)
        if pos is not None and pos["input_ids"]:
            positions.append((index, pos))
        elif prompt_only or row["answer"]:
            raise RuntimeError(f"unresolvable nonempty answer/prompt: {row['conv_id']}")
    positions.sort(key=lambda x: len(x[1]["input_ids"]))
    saved = {}

    def hook(_module, _args, output):
        saved["hidden"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[18].register_forward_hook(hook)
    try:
        start = 0
        while start < len(positions):
            end = start + 1
            while (
                end < len(positions)
                and end - start < 8
                and len(positions[end][1]["input_ids"]) * (end - start + 1) <= 8192
            ):
                end += 1
            batch = positions[start:end]
            padded = tokenizer.pad(
                {"input_ids": [p["input_ids"] for _, p in batch]}, padding=True, return_tensors="pt"
            )
            with torch.inference_mode():
                model.model(
                    **{k: v.to("cuda") for k, v in padded.items()},
                    use_cache=False,
                    output_hidden_states=False,
                )
            hs = saved.pop("hidden")
            for bi, (index, pos) in enumerate(batch):
                vectors[index] = (
                    hs[bi, pos["answer_lo"] : pos["answer_hi"]]
                    .mean(0)
                    .to(torch.float16)
                    .cpu()
                    .numpy()
                )
                legacy_context[index] = hs[bi, pos["v_C_pos"]].to(torch.float16).cpu().numpy()
            del hs, padded
            start = end
    finally:
        handle.remove()
    return vectors, legacy_context


def capture_stage(root, manifest, args):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = manifest["models"][args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"], revision=spec["revision"], use_fast=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            spec["id"],
            revision=spec["revision"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    for record in selected(manifest, args):
        cell = record["cell"]
        if cell.split("__")[-1] != args.model:
            continue
        fp = fingerprint(manifest, cell, args.pilot)
        rows = banked_rows(root, record, args.pilot)
        with np.load(root / "inputs" / record["activation"], allow_pickle=False) as z:
            bank = {k: z[k] for k in z.files}
        order = {str(cid): i for i, cid in enumerate(bank["conv_id"])}
        for offset in range(0, len(rows), CHUNK):
            path = root / "captures" / cell / f"chunk_{offset:05d}.npz"
            if complete(path, fp):
                continue
            t0 = time.monotonic()
            batch = rows[offset : offset + CHUNK]
            raw_path = root / "raw" / cell / f"chunk_{offset:05d}.json"
            if not complete(raw_path, fp):
                raise RuntimeError(f"raw chunk not verified: {raw_path}")
            fresh = load_raw(raw_path, fp)
            expected = [(r["conv_id"], draw) for r in batch for draw in (1, 2)]
            if [(r["conv_id"], r["draw"]) for r in fresh] != expected:
                raise RuntimeError("raw rollout order/count mismatch")
            answers, contexts = forward_vectors(model, tokenizer, fresh)
            fixed_context, _ = forward_vectors(model, tokenizer, batch, prompt_only=True)
            valid = np.isfinite(answers).all(axis=1)
            if not np.isfinite(fixed_context).all():
                raise RuntimeError(f"nonfinite deterministic context: {cell}")
            if any(bool(ok) != bool(row["answer"]) for ok, row in zip(valid, fresh, strict=True)):
                raise RuntimeError(f"nonfinite nonempty answer capture: {cell}")
            indices = np.array([order[r["conv_id"]] for r in batch])
            # Same capture implementation is tested against legacy raw vectors.
            parity_rows = batch[:8]
            recomputed, old_context = forward_vectors(model, tokenizer, parity_rows)
            reference = bank["v_A"][indices[:8]].astype(np.float32)
            rel = np.linalg.norm(recomputed.astype(np.float32) - reference, axis=1) / np.maximum(
                np.linalg.norm(reference, axis=1), 1e-9
            )
            if not np.isfinite(rel).all() or rel.max() > 0.025:
                raise RuntimeError(
                    f"banked answer capture parity failed {cell}: relative max={rel.max()}"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                np.savez_compressed(
                    fh,
                    conv_id=np.array([r["conv_id"] for r in batch]),
                    v_C=fixed_context,
                    v_A_0=bank["v_A"][indices],
                    v_A_12=answers.reshape(len(batch), 2, 3584),
                    sampled_legacy_v_C=contexts.reshape(len(batch), 2, 3584),
                    banked_v_C=bank["v_C"][indices],
                    v_P=bank["v_P"][indices],
                    v_P_present=bank["v_P_present"][indices],
                    parity_relative_error=rel,
                    valid_draws_12=valid.reshape(len(batch), 2),
                    cap_mask=np.array(
                        [
                            [r.get("finish_reason") == "length"]
                            + [fresh[2 * i + j]["finish_reason"] == "length" for j in (0, 1)]
                            for i, r in enumerate(batch)
                        ]
                    ),
                )
            tmp.replace(path)
            seal(path, root, fp)
            log(
                f"[phase=capture] {cell} rows={min(offset + CHUNK, len(rows))}/{len(rows)} parity_max={rel.max():.5f} seconds={time.monotonic() - t0:.1f} peak_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}"
            )


def run_children(root, args, stage):
    import concurrent.futures

    devices = (
        os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if os.environ.get("CUDA_VISIBLE_DEVICES")
        else subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        ).split()
    )
    if not devices:
        raise RuntimeError("no allocated GPUs")

    def worker(shard):
        for model in capture._MODEL_ID:
            command = [
                sys.executable,
                __file__,
                "--stage",
                stage,
                "--out-root",
                str(root),
                "--shard",
                str(shard),
                "--shards",
                str(len(devices)),
                "--model",
                model,
            ]
            if args.pilot:
                command.append("--pilot")
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=devices[shard],
                VLLM_WORKER_MULTIPROC_METHOD="spawn",
                PYTHONUNBUFFERED="1",
            )
            path = root / "logs" / f"{stage}_{shard}_{model}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            log(f"[phase={stage}] worker={shard} model={model} log={path}")
            with path.open("w") as output:
                process = subprocess.Popen(
                    command, env=env, stdout=output, stderr=subprocess.STDOUT
                )
                atomic_json(
                    path.with_suffix(".pid.json"),
                    {"pid": process.pid, "stage": stage, "model": model},
                )
                code = process.wait()
            if code:
                log(path.read_text()[-16000:])
                raise RuntimeError(f"worker failed ({code}): {path}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, i) for i in range(len(devices))]
        while any(not f.done() for f in futures):
            log(
                f"[phase={stage}] active_workers={sum(not f.done() for f in futures)}/{len(devices)}"
            )
            time.sleep(30)
        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "generate", "capture", "gpu"], required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--model", choices=list(capture._MODEL_ID))
    args = parser.parse_args()
    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.pilot and root.name != "pilot":
        raise ValueError("pilot output root must be named pilot")
    if not args.pilot and root.name == "pilot":
        raise ValueError("production cannot use pilot outputs")
    if args.stage == "prepare":
        prepare(root)
        return
    manifest = json.loads((root / "manifest.json").read_text())
    if args.stage == "generate":
        generate(root, manifest, args)
    elif args.stage == "capture":
        capture_stage(root, manifest, args)
    else:
        run_children(root, args, "generate")
        run_children(root, args, "capture")
        summary = {
            "stage": "gpu_capture",
            "pilot": args.pilot,
            "status": "complete",
            "cells": 24,
            "raw_chunks": len(
                [
                    p
                    for p in (root / "raw").glob("*/chunk_*.json")
                    if "_part" not in p.name and ".done." not in p.name
                ]
            ),
            "capture_chunks": len(list((root / "captures").glob("*/*.npz"))),
            "time": time.time(),
        }
        atomic_json(root / "gpu_complete.json", summary)
        seal(root / "gpu_complete.json", root, VERSION)
        log("[phase=capture_complete] all GPU stages complete; analysis remains")


if __name__ == "__main__":
    main()
