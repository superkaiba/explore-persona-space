"""Restore frozen K3 generation checkpoints and verify their consuming layout."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import concurrent.futures
import json
from pathlib import Path

from scripts import issue2054_k3 as k3


def restore(root, revision):
    """Read-only restoration; never launches generation or changes old receipts."""
    import numpy as np
    from huggingface_hub import HfApi
    from explore_persona_space.orchestrate.hub import retry_transient

    root = Path(root).resolve()
    prefix = f"{k3.PREFIX}/production"
    api = HfApi()
    entries = retry_transient(
        lambda: list(
            api.list_repo_tree(
                k3.HF_REPO,
                repo_type="dataset",
                revision=revision,
                path_in_repo=prefix,
                recursive=True,
            )
        ),
        what="list frozen K3 production checkpoints",
    )
    paths = [
        e.path
        for e in entries
        if hasattr(e, "size")
        and (
            e.path.startswith(prefix + "/raw/")
            or e.path.startswith(prefix + "/captures/")
            or e.path in (prefix + "/manifest.json", prefix + "/manifest.json.done.json")
        )
    ]
    if not paths:
        raise RuntimeError("empty frozen production snapshot")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(k3.download, path, root, revision) for path in paths]
        for index, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            job.result()
            if index % 64 == 0 or index == len(jobs):
                k3.log(f"[phase=restore_download] files={index}/{len(jobs)}")
    source = root / "inputs" / prefix
    if not k3.complete(source / "manifest.json", k3.VERSION):
        raise RuntimeError("frozen production manifest lacks verified receipt")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["revision"] != k3.REVISION:
        raise RuntimeError("banked input revision differs")
    if manifest["models"] != {
        slug: {"id": model, "revision": k3.MODEL_REVISIONS[slug]}
        for slug, model in k3.capture._MODEL_ID.items()
    }:
        raise RuntimeError("model pins differ")
    # Stage exactly the layout the original consumer opens, using original pins.
    for record in manifest["cells"]:
        for key in ("activation", "raw"):
            if key in record:
                path = k3.download(record[key], root, manifest["revision"])
                if k3.sha(path) != record[key + "_sha256"]:
                    raise RuntimeError(f"banked {key} content changed: {record['cell']}")
    fold_path = Path(manifest["fold_map"])
    if fold_path.parts[0] != "inputs":
        raise RuntimeError("unexpected banked fold layout")
    fold = k3.download(str(Path(*fold_path.parts[1:])), root, manifest["revision"])
    if k3.sha(fold) != manifest["fold_sha256"]:
        raise RuntimeError("banked fold content changed")
    k3.atomic_json(root / "manifest.json", manifest)
    raw_link = root / "raw"
    if raw_link.exists() or raw_link.is_symlink():
        if raw_link.resolve() != (source / "raw").resolve():
            raise RuntimeError("existing raw staging points to another source")
    else:
        raw_link.symlink_to(source / "raw", target_is_directory=True)
    raw_count, capture_count, answer_count = 0, 0, 0
    saved_captures = []
    for record in manifest["cells"]:
        if "raw" not in record:
            continue
        cell = record["cell"]
        fp = k3.fingerprint(manifest, cell, False)
        rows = k3.banked_rows(root, record, False)
        for offset in range(0, len(rows), k3.CHUNK):
            batch = rows[offset : offset + k3.CHUNK]
            path = root / "raw" / cell / f"chunk_{offset:05d}.json"
            if not k3.complete(path, fp):
                raise RuntimeError(f"missing frozen raw chunk: {path}")
            fresh = k3.load_raw(path, fp)
            expected = [(r["conv_id"], draw) for r in batch for draw in (1, 2)]
            if [(r["conv_id"], r["draw"]) for r in fresh] != expected:
                raise RuntimeError(f"raw rollout order/count mismatch: {path}")
            for i, row in enumerate(fresh):
                original = batch[i // 2]
                prefix_text = original["final_text"][: original["answer_start"]]
                suffix = original["final_text"][original["answer_end"] :]
                if (
                    row["answer_start"] != len(prefix_text)
                    or row["answer_end"] != len(prefix_text) + len(row["answer"])
                    or row["final_text"] != prefix_text + row["answer"] + suffix
                    or row["seed"] != k3.seed(cell, row["conv_id"], row["draw"])
                    or row["max_tokens_budget"] != record["cap"]
                ):
                    raise RuntimeError(f"raw rollout recipe mismatch: {path}")
            raw_count += 1
            answer_count += len(fresh)
            capture = source / "captures" / cell / f"chunk_{offset:05d}.npz"
            if k3.complete(capture, fp):
                with np.load(capture, allow_pickle=False) as z:
                    if list(z["conv_id"]) != [r["conv_id"] for r in batch] or z["v_A_12"].shape != (
                        len(batch),
                        2,
                        3584,
                    ):
                        raise RuntimeError(f"saved capture population/shape mismatch: {capture}")
                capture_count += 1
                saved_captures.append(str(capture.relative_to(source)))
            k3.log(
                f"[phase=restore_verify] raw={raw_count}/768 answers={answer_count} captures={capture_count} cell={cell} offset={offset}"
            )
    if raw_count != 768 or answer_count != 383998:
        raise RuntimeError("frozen raw production coverage incomplete")
    report = {
        "revision": revision,
        "source_root": str(source),
        "raw_chunks": raw_count,
        "answers": answer_count,
        "verified_old_captures": saved_captures,
        "capture_chunks": capture_count,
        "capture_reuse_adjudication_pending": True,
    }
    k3.atomic_json(root / "restore_verified.json", report)
    return manifest, source, report
