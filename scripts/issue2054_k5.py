"""Extend the twelve manuscript cells from three to five independent answers.

The K3 source, numerical adjudication and receipts remain immutable. Only new
draws 3/4 are generated; all K1/K3/K5 fits use the same complete-five cohort.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k3 as k3
from scripts import issue2054_k3_artifacts as artifacts
from scripts import issue2054_k3_fit as k3fit
from scripts import issue2054_k3_recover as recovery
from explore_persona_space.orchestrate.hub import retry_transient, stage_hub_file

PARENT_REV = "5ae90722bf11330deddfa42cf41f9fec6da8b69f"
PARENT_PREFIX = "issue2054_section44_k3_gcp/capture_recovery_v1"
RAW_PREFIX = "issue2054_section44_k3_gcp/production"
OUTPUT_PREFIX = "issue2054_section44_k5_gcp"
COUNTS = (1, 3, 5)
NEW_DRAWS = (3, 4)


def configure_output():
    """Select this process's destination for the unchanged receipt uploader."""
    k3.PREFIX = OUTPUT_PREFIX


def policy():
    """Validate the exact inherited capture code/model identities."""
    return recovery.load_policy(REPO / "configs/issue2054_k3_capture_policy.json")


def selected(manifest):
    """Return the exact six settings per model shown in the manuscript."""
    cells = [c for c in manifest["cells"] if k3fit.displayed(c["cell"])]
    if len(cells) != 12 or any("raw" not in c for c in cells):
        raise RuntimeError("expected twelve on-policy manuscript cells")
    return sorted(cells, key=lambda c: (c["cell"].split("__")[-1], c["cell"]))


def seed(cell, conv_id, draw):
    """Use disjoint deterministic streams for the two additional draws."""
    if draw not in NEW_DRAWS:
        raise ValueError("K5 only generates draws 3 and 4")
    return int.from_bytes(
        hashlib.sha256(f"2054-k5|{cell}|{conv_id}|{draw}".encode()).digest()[:4], "little"
    )


def fingerprint(manifest, cell):
    """Bind every new checkpoint to inputs, source, capture policy and recipe."""
    payload = {
        "parent_revision": PARENT_REV,
        "parent_capture": recovery.capture_fingerprint(manifest, cell, False, policy()),
        "new_source": k3.sha(__file__),
        "new_draws": NEW_DRAWS,
        "counts": COUNTS,
        "pool": "six displayed settings separately within each model",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def fetch(root, remote, revision):
    """Stage atomically into an unchanged remote-relative layout."""
    target = root / "inputs" / remote
    return stage_hub_file(k3.HF_REPO, remote, target, revision=revision)


def verified_parent(root, relative, expected_fingerprint=None):
    """Load a parent checkpoint only with a matching content receipt."""
    path = root / "inputs" / PARENT_PREFIX / relative
    receipt = json.loads(path.with_suffix(path.suffix + ".done.json").read_text())
    if receipt["path"] != f"{PARENT_PREFIX}/{relative}":
        raise RuntimeError(f"wrong parent receipt destination: {path}")
    fp = expected_fingerprint or receipt["fingerprint"]
    if not k3.complete(path, fp):
        raise RuntimeError(f"incomplete parent artifact: {path}")
    return path


def prepare(root, *, first_chunks=False):
    """Restore exactly the displayed captures/raw text and validate row identity."""
    from huggingface_hub import HfApi
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    root.mkdir(parents=True, exist_ok=True)
    k3.log("[phase=restore_start] resolving immutable K3 manuscript inputs")
    manifest_path = fetch(root, PARENT_PREFIX + "/manifest.json", PARENT_REV)
    fetch(root, PARENT_PREFIX + "/manifest.json.done.json", PARENT_REV)
    manifest = json.loads(verified_parent(root, "manifest.json").read_text())
    if manifest["revision"] != k3.REVISION or manifest["models"] != {
        slug: {"id": mid, "revision": k3.MODEL_REVISIONS[slug]}
        for slug, mid in k3.capture._MODEL_ID.items()
    }:
        raise RuntimeError("parent source/model revision mismatch")
    cells = selected(manifest)
    if sum(c["n"] for c in cells) != 95999:
        raise RuntimeError("manuscript source population changed")
    k3.atomic_json(root / "manifest.json", manifest)
    k3.atomic_json(root / "capture_policy.json", policy())
    # One scoped listing per source prefix; parent raw is a symlink on the old
    # machine and therefore lives under production, not the recovery prefix.
    paths = []
    for prefix in (PARENT_PREFIX, RAW_PREFIX):
        entries = retry_transient(
            lambda prefix=prefix: list(
                HfApi().list_repo_tree(
                    k3.HF_REPO,
                    path_in_repo=prefix,
                    repo_type="dataset",
                    revision=PARENT_REV,
                    recursive=True,
                )
            ),
            what=f"list K5 parent inputs {prefix}",
        )
        for entry in entries:
            if not hasattr(entry, "size"):
                continue
            rel = entry.path.removeprefix(prefix + "/")
            parts = Path(rel).parts
            if len(parts) < 3 or parts[1] not in {c["cell"] for c in cells}:
                continue
            if parts[0] not in (
                ("captures", "capture_audits") if prefix == PARENT_PREFIX else ("raw",)
            ):
                continue
            if first_chunks and not parts[-1].startswith("chunk_00000"):
                continue
            paths.append((entry.path, entry.size))
    if not paths:
        raise RuntimeError("empty parent checkpoint selection")
    pending_bytes = sum(size for path, size in paths if not (root / "inputs" / path).exists())
    if pending_bytes:
        assert_out_root_headroom(root, pending_bytes / 1e9 * 1.5 + 3, phase="k5_restore")
    started = time.monotonic()
    k3.log(f"[phase=restore_transfer] files={len(paths)} missing_bytes={pending_bytes}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(fetch, root, path, PARENT_REV) for path, _ in paths]
        for i, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            job.result()
            if i % 32 == 0 or i == len(jobs):
                k3.log(
                    f"[phase=restore_transfer] files={i}/{len(jobs)} elapsed={time.monotonic() - started:.1f}s"
                )
    for record in cells:
        # The verified K3 captures already contain draw0 and fixed context;
        # their source activation identities stay in the frozen manifest.
        for key in ("raw",):
            path = fetch(root, record[key], manifest["revision"])
            if k3.sha(path) != record[key + "_sha256"]:
                raise RuntimeError(f"banked {key} hash mismatch")
    fold = fetch(root, str(Path(manifest["fold_map"]).relative_to("inputs")), manifest["revision"])
    if k3.sha(fold) != manifest["fold_sha256"]:
        raise RuntimeError("fold identity changed")
    chunks = answers = 0
    for record in cells:
        cell = record["cell"]
        rows = k3.banked_rows(root, record, first_chunks)
        old_fp = k3.fingerprint(manifest, cell, False)
        capture_fp = recovery.capture_fingerprint(manifest, cell, False, policy())
        for offset in range(0, len(rows), k3.CHUNK):
            batch = rows[offset : offset + k3.CHUNK]
            raw = root / "inputs" / RAW_PREFIX / "raw" / cell / f"chunk_{offset:05d}.json"
            if not k3.complete(raw, old_fp):
                raise RuntimeError(f"missing old raw receipt: {raw}")
            old_rows = k3.load_raw(raw, old_fp)
            if [(r["conv_id"], r["draw"]) for r in old_rows] != [
                (r["conv_id"], draw) for r in batch for draw in (1, 2)
            ]:
                raise RuntimeError("K3 raw row/draw identity mismatch")
            for i, row in enumerate(old_rows):
                orig = batch[i // 2]
                prefill = orig["final_text"][: orig["answer_start"]]
                suffix = orig["final_text"][orig["answer_end"] :]
                if (
                    row["seed"] != k3.seed(cell, row["conv_id"], row["draw"])
                    or row["max_tokens_budget"] != record["cap"]
                    or row["final_text"] != prefill + row["answer"] + suffix
                    or row["answer_start"] != len(prefill)
                    or row["answer_end"] != len(prefill) + len(row["answer"])
                ):
                    raise RuntimeError("K3 raw recipe mismatch")
            captured = verified_parent(root, f"captures/{cell}/chunk_{offset:05d}.npz", capture_fp)
            audit_path = verified_parent(
                root, f"capture_audits/{cell}/chunk_{offset:05d}.json", capture_fp
            )
            audit = json.loads(audit_path.read_text())
            with np.load(captured, allow_pickle=False) as z:
                if (
                    list(z["conv_id"]) != [r["conv_id"] for r in batch]
                    or z["v_A_12"].shape != (len(batch), 2, 3584)
                    or z["cap_mask"].shape != (len(batch), 3)
                    or not np.isfinite(z["v_C"]).all()
                    or not np.isfinite(z["v_A_0"]).all()
                    or not np.array_equal(
                        z["parity_relative_error"],
                        np.asarray(
                            audit["historical_relative_error"],
                            dtype=z["parity_relative_error"].dtype,
                        ),
                    )
                ):
                    raise RuntimeError("K3 capture schema/order/audit mismatch")
            chunks += 1
            answers += len(old_rows)
        k3.log(f"[phase=restore_verify] {cell} chunks={chunks} reused_answers={answers}")
    report = {
        "revision": PARENT_REV,
        "cells": [c["cell"] for c in cells],
        "capture_chunks": chunks,
        "reused_fresh_answers": answers,
        "first_chunks_only": first_chunks,
        "manifest_sha256": k3.sha(manifest_path),
    }
    k3.atomic_json(root / "restore_verified.json", report)
    return manifest, report


def save_raw(path, rows, timing, root, fp):
    """Persist sub-8MB JSON parts, index and timing in one verified packet."""
    paths = []
    batch = []
    size = 2
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False).encode()
        if len(encoded) > 8_000_000:
            raise ValueError("single generation exceeds text-shard limit")
        if batch and size + len(encoded) > 8_000_000:
            shard = path.with_name(f"{path.stem}_part{len(paths):03d}.json")
            k3.atomic_json(shard, batch)
            paths.append(shard)
            batch, size = [], 2
        batch.append(row)
        size += len(encoded) + 1000
    if batch:
        shard = path.with_name(f"{path.stem}_part{len(paths):03d}.json")
        k3.atomic_json(shard, batch)
        paths.append(shard)
    k3.atomic_json(path, {"shards": [p.name for p in paths], "rows": len(rows)})
    timing_path = path.with_suffix(".timing.json")
    k3.atomic_json(timing_path, timing)
    artifacts.seal_many([*paths, path, timing_path], root, fp)


def phase_headroom(root, manifest, stage):
    """Bind conservative remaining-write estimates to the actual output mount."""
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    pending = 0
    for record in selected(manifest):
        cell = record["cell"]
        fp = fingerprint(manifest, cell)
        if stage == "aggregate":
            # Aggregates are rewritten, including expansion of reusable pilot
            # rows into full production under the same recipe fingerprint.
            pending += len(COUNTS)
        else:
            directory, extension = ("raw", "json") if stage == "generate" else ("captures", "npz")
            pending += sum(
                not k3.complete(root / directory / cell / f"chunk_{offset:05d}.{extension}", fp)
                for offset in range(0, record["n"], k3.CHUNK)
            )
    if pending:
        per_unit_gb = {"generate": 0.064, "capture": 0.016, "aggregate": 0.3}[stage]
        assert_out_root_headroom(root, pending * per_unit_gb * 1.5 + 3, phase=f"k5_{stage}")


def generate(root, manifest, args):
    """Run production-sized vLLM chunks with the inherited stops and caps."""
    from vllm import LLM, SamplingParams

    records = [c for c in selected(manifest) if c["cell"].endswith("__" + args.model)]
    engine = None
    for ci, record in enumerate(selected(manifest)):
        if record not in records:
            continue
        cell = record["cell"]
        fp = fingerprint(manifest, cell)
        rows = k3.banked_rows(root, record, args.first_chunks)
        spec = manifest["models"][args.model]
        for offset in range(0, len(rows), k3.CHUNK):
            if not k3.owns_chunk(ci, offset, args.shard, args.shards):
                continue
            path = root / "raw" / cell / f"chunk_{offset:05d}.json"
            if k3.complete(path, fp):
                continue
            if engine is None:
                engine = LLM(
                    model=spec["id"],
                    revision=spec["revision"],
                    tokenizer_revision=spec["revision"],
                    dtype="bfloat16",
                    max_model_len=8192,
                    gpu_memory_utilization=0.85,
                    enforce_eager=True,
                    max_num_seqs=128,
                    tensor_parallel_size=1,
                    seed=137,
                )
            batch = rows[offset : offset + k3.CHUNK]
            requests = [(row, draw) for row in batch for draw in NEW_DRAWS]
            prompts = [r["final_text"][: r["answer_start"]] for r, _ in requests]
            params = [
                SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=record["cap"],
                    stop=k3.STOPS[cell.split("__")[2]],
                    seed=seed(cell, r["conv_id"], d),
                )
                for r, d in requests
            ]
            started = time.monotonic()
            outputs = engine.generate(prompts, params, use_tqdm=False)
            generated = []
            for (row, draw), output in zip(requests, outputs, strict=True):
                answer = output.outputs[0]
                item = dict(row)
                prefix, suffix = (
                    row["final_text"][: row["answer_start"]],
                    row["final_text"][row["answer_end"] :],
                )
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
            elapsed = time.monotonic() - started
            save_raw(
                path,
                generated,
                {
                    "cell": cell,
                    "offset": offset,
                    "contexts": len(batch),
                    "answers": len(generated),
                    "seconds": elapsed,
                    "cap_count": sum(r["finish_reason"] == "length" for r in generated),
                    "empty_count": sum(not r["answer"] for r in generated),
                },
                root,
                fp,
            )
            k3.log(
                f"[phase=generation] {cell} offset={offset} answers={len(generated)} elapsed={elapsed:.1f}s"
            )


def capture(root, manifest, args):
    """Capture draws 3/4 while preserving the exact K3 context and first draws."""
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
    for ci, record in enumerate(selected(manifest)):
        cell = record["cell"]
        if not cell.endswith("__" + args.model):
            continue
        rows = k3.banked_rows(root, record, args.first_chunks)
        fp = fingerprint(manifest, cell)
        pending = []
        for offset in range(0, len(rows), k3.CHUNK):
            if not k3.owns_chunk(ci, offset, args.shard, args.shards):
                continue
            path = root / "captures" / cell / f"chunk_{offset:05d}.npz"
            audit_path = root / "capture_audits" / cell / f"chunk_{offset:05d}.json"
            if k3.complete(path, fp) and k3.complete(audit_path, fp):
                continue
            started = time.monotonic()
            batch = rows[offset : offset + k3.CHUNK]
            raw_path = root / "raw" / cell / f"chunk_{offset:05d}.json"
            if not k3.complete(raw_path, fp):
                raise RuntimeError("unverified new raw chunk")
            fresh = k3.load_raw(raw_path, fp)
            if [(r["conv_id"], r["draw"]) for r in fresh] != [
                (r["conv_id"], d) for r in batch for d in NEW_DRAWS
            ]:
                raise RuntimeError("new raw row/draw order mismatch")
            old = verified_parent(
                root,
                f"captures/{cell}/chunk_{offset:05d}.npz",
                recovery.capture_fingerprint(manifest, cell, False, policy()),
            )
            with np.load(old, allow_pickle=False) as z:
                bank = {k: z[k] for k in z.files}
            answers, contexts = k3.forward_vectors(model, tokenizer, fresh)
            valid = np.isfinite(answers).all(axis=1)
            if any(bool(ok) != bool(row["answer"]) for ok, row in zip(valid, fresh, strict=True)):
                raise RuntimeError("nonfinite nonempty answer")
            recomputed, _ = k3.forward_vectors(model, tokenizer, batch[:8])
            _, audit = recovery.parity_audit(
                model, tokenizer, batch[:8], recomputed, bank["v_A_0"][:8], policy()
            )
            audit.update(
                cell=cell,
                offset=offset,
                parent_revision=PARENT_REV,
                parent_capture_sha256=k3.sha(old),
            )
            content = {
                "conv_id": bank["conv_id"],
                "v_A_34": answers.reshape(len(batch), 2, 3584),
                "valid_draws_34": valid.reshape(len(batch), 2),
                "sampled_legacy_v_C_34": contexts.reshape(len(batch), 2, 3584),
                "cap_mask_34": np.array([r["finish_reason"] == "length" for r in fresh]).reshape(
                    len(batch), 2
                ),
            }
            k3fit.save_npz(path, content)
            k3.atomic_json(audit_path, audit)
            pending.extend([path, audit_path])
            if len(pending) >= 16:
                artifacts.seal_many(pending, root, fp)
                pending.clear()
            k3.log(
                f"[phase=capture] {cell} offset={offset} elapsed={time.monotonic() - started:.1f}s peak_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}"
            )
        if pending:
            artifacts.seal_many(pending, root, fp)


def average_targets(old, new):
    """Return paired exact K1/K3/K5 targets and all-five validity/cap masks."""
    n, d = old["v_A_0"].shape
    if d != 3584 or old["v_A_12"].shape != (n, 2, d) or new["v_A_34"].shape != (n, 2, d):
        raise ValueError("expected five vectors in ambient dimension")
    if not np.array_equal(old["conv_id"], new["conv_id"]):
        raise ValueError("draws differ in context identity/order")
    keep = old["valid_draws_12"].all(1) & new["valid_draws_34"].all(1)
    first = old["v_A_0"][keep].astype(np.float32)
    first3 = first + old["v_A_12"][keep].astype(np.float32).sum(1)
    targets = {
        1: first,
        3: first3 / 3,
        5: (first3 + new["v_A_34"][keep].astype(np.float32).sum(1)) / 5,
    }
    if not all(np.isfinite(y).all() for y in targets.values()):
        raise ValueError("nonfinite complete-five target")
    caps = np.concatenate([old["cap_mask"], new["cap_mask_34"]], axis=1)
    if caps.shape != (n, 5):
        raise ValueError("expected five cap indicators")
    return targets, keep, caps


def aggregate(root, manifest, *, first_chunks=False):
    """Write matched target arrays and realized coverage per displayed cell."""
    reports = []
    for record in selected(manifest):
        cell = record["cell"]
        fp = fingerprint(manifest, cell)
        rows = k3.banked_rows(root, record, first_chunks)
        old_chunks = []
        new_chunks = []
        for offset in range(0, len(rows), k3.CHUNK):
            old = verified_parent(
                root,
                f"captures/{cell}/chunk_{offset:05d}.npz",
                recovery.capture_fingerprint(manifest, cell, False, policy()),
            )
            new = root / "captures" / cell / f"chunk_{offset:05d}.npz"
            if not k3.complete(new, fp):
                raise RuntimeError("missing K5 capture")
            for path, collection in ((old, old_chunks), (new, new_chunks)):
                with np.load(path, allow_pickle=False) as z:
                    collection.append({k: z[k] for k in z.files})
        old = {k: np.concatenate([c[k] for c in old_chunks]) for k in old_chunks[0]}
        new = {k: np.concatenate([c[k] for c in new_chunks]) for k in new_chunks[0]}
        targets, keep, caps = average_targets(old, new)
        if list(old["conv_id"]) != [r["conv_id"] for r in rows]:
            raise RuntimeError("aggregate source population differs")
        paths = []
        for count, y in targets.items():
            content = {k: old[k][keep] for k in ("conv_id", "v_C", "v_P", "v_P_present")}
            content.update(v_A=y, cap_mask=caps[keep])
            path = root / f"k{count}" / f"{cell}.npz"
            k3fit.save_npz(path, content)
            paths.append(path)
        artifacts.seal_many(paths, root, fp)
        report = {
            "cell": cell,
            "original_rows": len(rows),
            "complete_five_rows": int(keep.sum()),
            "complete_three_rows": int(old["valid_draws_12"].all(1).sum()),
            "empty_draw_rows_excluded": int((~keep).sum()),
            "cap_counts_each_draw": caps.sum(0).tolist(),
            "original_draw_stopped_valid": int((keep & ~caps[:, 0]).sum()),
            "all_five_stopped_valid": int((keep & ~caps.any(1)).sum()),
        }
        reports.append(report)
        k3.log(f"[phase=aggregate] {cell} complete_five={keep.sum()}/{len(rows)}")
    k3.atomic_json(root / "coverage.json", reports)
    artifacts.seal_many([root / "coverage.json"], root, k3.sha(__file__))
    return reports


def children(root, args, stage, devices, *, first_chunks=False):
    """Wait for isolated GPU workers; every worker receives one explicit device."""
    stopped = threading.Event()
    active = {}
    lock = threading.Lock()

    def worker(shard):
        for model in k3.MODEL_REVISIONS:
            command = [
                sys.executable,
                __file__,
                "--stage",
                stage,
                "--out-root",
                str(root),
                "--model",
                model,
                "--shard",
                str(shard),
                "--shards",
                str(len(devices)),
            ]
            if first_chunks:
                command.append("--first-fold" if stage == "fit" else "--first-chunks")
            log = root / "logs" / f"{'pilot_' if first_chunks else ''}{stage}_{shard}_{model}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as output:
                with lock:
                    if stopped.is_set():
                        return
                    child = subprocess.Popen(
                        command,
                        env=dict(
                            os.environ, CUDA_VISIBLE_DEVICES=devices[shard], PYTHONUNBUFFERED="1"
                        ),
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    active[child.pid] = child
                k3.atomic_json(
                    log.with_suffix(".pid.json"),
                    {"pid": child.pid, "stage": stage, "model": model, "shard": shard},
                )
                code = child.wait()
                if not code:
                    with lock:
                        active.pop(child.pid)
                if code:
                    stopped.set()
            artifacts.seal_many([log, log.with_suffix(".pid.json")], root, k3.sha(__file__))
            if code:
                k3.log(f"[phase=worker_failed] {log}: {log.read_text()[-10000:]}")
                raise RuntimeError(f"worker {stage}/{shard}/{model} exit={code}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        jobs = [pool.submit(worker, i) for i in range(len(devices))]
        while True:
            if stopped.is_set() or any(j.done() and j.exception() is not None for j in jobs):
                with lock:
                    stopped.set()
                    owned = list(active.values())
                for child in owned:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        k3.log(f"[phase=worker_cancel] pid={child.pid} already exited")
                for child in owned:
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        k3.log(f"[phase=worker_cancel] pid={child.pid} exceeded TERM grace")
                    # A failed leader may already have exited while its vLLM
                    # descendants remain. Kill its owned group in either case.
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        k3.log(f"[phase=worker_cancel] pgid={child.pid} drained")
                    child.wait()
                break
            if all(j.done() for j in jobs):
                break
            k3.log(
                f"[phase={stage}] active_workers={sum(not j.done() for j in jobs)}/{len(devices)}"
            )
            time.sleep(5)
        for job in jobs:
            job.result()


def main():
    """Route source-pinned pilot, production and checkpointed fit stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("job", "prepare", "generate", "capture", "fit"), required=True
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--model", choices=list(k3.MODEL_REVISIONS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--first-chunks", action="store_true")
    parser.add_argument("--first-fold", action="store_true")
    args = parser.parse_args()
    configure_output()
    root = args.out_root.resolve()
    if args.stage == "prepare":
        prepare(root, first_chunks=args.first_chunks)
        return
    if args.stage != "job":
        manifest = json.loads((root / "manifest.json").read_text())
        if args.stage == "generate":
            generate(root, manifest, args)
        elif args.stage == "capture":
            capture(root, manifest, args)
        else:
            from scripts.issue2054_k5_fit import fit_model

            fit_model(
                root, manifest, args.model, args.shard, args.shards, first_fold=args.first_fold
            )
        return
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if actual != args.source_sha:
        raise RuntimeError("K5 source SHA mismatch")
    from scripts.issue2054_k3_job import prepare_git

    prepare_git(REPO, actual)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "explore_persona_space.orchestrate.preflight",
            "--planned-footprint-gb",
            "90",
            "--min-disk",
            "100",
        ],
        check=True,
    )
    started = time.time()
    manifest, restore_report = prepare(root)
    artifacts.seal_many(
        [root / "manifest.json", root / "restore_verified.json", root / "capture_policy.json"],
        root,
        k3.sha(__file__),
    )
    devices = (
        os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if os.environ.get("CUDA_VISIBLE_DEVICES")
        else subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        ).split()
    )
    if not devices:
        raise RuntimeError("no GPUs allocated")
    from huggingface_hub import snapshot_download

    for model, revision in k3.MODEL_REVISIONS.items():
        k3.log(f"[phase=model_prefetch] model={model}")
        retry_transient(
            lambda model=model, revision=revision: snapshot_download(
                k3.capture._MODEL_ID[model], revision=revision, max_workers=4
            ),
            what=f"K5 model prefetch {model}",
        )
    # The first chunk of each cell is the pilot AND part of production: no
    # duplicate draws, same fingerprint, same batch geometry and upload path.
    phase_headroom(root, manifest, "generate")
    children(root, args, "generate", devices, first_chunks=True)
    phase_headroom(root, manifest, "capture")
    children(root, args, "capture", devices, first_chunks=True)
    phase_headroom(root, manifest, "aggregate")
    pilot_coverage = aggregate(root, manifest, first_chunks=True)
    if any(c["complete_five_rows"] < 0.9 * c["original_rows"] for c in pilot_coverage):
        raise RuntimeError("K5 pilot lost >10% of contexts to empty draws")
    timings = [json.loads(p.read_text()) for p in (root / "raw").glob("*/chunk_00000.timing.json")]
    if len(timings) != 12:
        raise RuntimeError("pilot timing coverage incomplete")
    pilot = {
        "status": "pass",
        "timings": timings,
        "coverage": pilot_coverage,
        "projected_generation_gpu_hours": sum(t["seconds"] * 32 for t in timings) / 3600,
        "smoke_blind_spots": ["ambient fits require complete production rows"],
    }
    k3.atomic_json(root / "pilot_complete.json", pilot)
    artifacts.seal_many([root / "pilot_complete.json"], root, k3.sha(__file__))
    k3.log(
        f"[phase=pilot_complete] projected_generation_gpu_hours={pilot['projected_generation_gpu_hours']:.2f}"
    )
    if pilot["projected_generation_gpu_hours"] > 96:
        raise RuntimeError(
            "generation pilot exceeds twice the provisional 48 GPU-hour basis; inspect before expansion"
        )
    phase_headroom(root, manifest, "generate")
    children(root, args, "generate", devices)
    phase_headroom(root, manifest, "capture")
    children(root, args, "capture", devices)
    phase_headroom(root, manifest, "aggregate")
    aggregate(root, manifest)
    fit_started = time.monotonic()
    children(root, args, "fit", devices[:1], first_chunks=True)
    k3.atomic_json(
        root / "fit_pilot_complete.json",
        {
            "status": "pass",
            "fold": 0,
            "models": list(k3.MODEL_REVISIONS),
            "production_rows": True,
            "seconds": time.monotonic() - fit_started,
            "source_sha": actual,
        },
    )
    artifacts.seal_many([root / "fit_pilot_complete.json"], root, k3.sha(__file__))
    children(root, args, "fit", devices)
    from scripts.issue2054_k5_fit import collect

    results = collect(root, manifest)
    report = {
        "status": "complete",
        "source_sha": actual,
        "parent_revision": PARENT_REV,
        "started": started,
        "finished": time.time(),
        "restore": restore_report,
        "hf_prefix": f"{OUTPUT_PREFIX}/{root.name}",
        "primary_panels": len(results),
        "context_convention": "unchanged K3 prefill-alone last-token at layer19",
    }
    k3.atomic_json(root / "job_complete.json", report)
    artifacts.seal_many([root / "job_complete.json"], root, k3.sha(__file__))
    from explore_persona_space.backends.artifacts import write_completion_sentinel

    write_completion_sentinel(
        sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=2054, extra=report
    )
    k3.log("[phase=done] K5 manuscript cells and six-setting shared fits complete")


if __name__ == "__main__":
    main()
