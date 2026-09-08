"""Archive an immutable task2670 high-rate snapshot and verify its remote readback.

This helper only persists data. It never collects observations or authorizes
compute teardown. The snapshot manifest names every original file and its bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

# PROD_IMPORT_LINT_EXEMPT: Runtime explicitly pinned by uv --with inspect-ai==0.3.261.
from inspect_ai.log import read_eval_log  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402
from scripts import context_risk_corrected_finish as archive  # noqa: E402
from scripts.issue2054_phase_a import _shard_large_jsonl_for_upload  # noqa: E402

RUN_ROOT = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate"
)


def source_hashes() -> dict:
    project = Path(__file__).resolve().parents[1]
    names = (
        "scripts/context_risk_highrate_archive.py",
        "scripts/context_risk_corrected_finish.py",
        "scripts/issue2054_phase_a.py",
        "src/explore_persona_space/orchestrate/hub.py",
        "src/explore_persona_space/orchestrate/env.py",
    )
    return {name: sha(project / name) for name in names}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_original(path: Path, expected: dict) -> dict:
    """Check exact reconstructed bytes before invoking the corresponding reader."""
    if path.stat().st_size != expected["size"] or sha(path) != expected["sha256"]:
        raise ValueError(f"Snapshot byte round trip failed: {path}")
    result = {"size": expected["size"], "sha256": expected["sha256"]}
    if path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
        result["parsed"] = "json"
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        result.update(parsed="jsonl", rows=len(rows))
    elif path.suffix == ".eval":
        log = read_eval_log(str(path), resolve_attachments="full")
        if log.status != "success" or not log.samples:
            raise ValueError(f"Native archive is not a completed nonempty log: {path}")
        keys = [(str(s.id), s.epoch) for s in log.samples]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate native sample identities: {path}")
        result.update(parsed="inspect_full", samples=len(keys))
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as bank:
            shapes = {}
            for name in bank.files:
                array = bank[name]
                if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite archived tensor: {path}:{name}")
                shapes[name] = list(array.shape)
        result.update(parsed="npz", shapes=shapes)
    else:
        result["parsed"] = "exact_bytes"
    return result


def prepare_shards(stage: Path) -> dict:
    """Use the existing writer and prove its byte round trip before uploading."""
    original = sorted(p for p in stage.rglob("*") if p.is_file())
    large = {}
    for path in original:
        if path.stat().st_size <= 9_500_000:
            continue
        if path.suffix == ".jsonl":
            large[path] = sha(path)
        elif path.suffix not in {".npz", ".eval"}:
            raise ValueError(f"Oversized text needs a reviewed line-sharding recipe: {path}")
    retained = _shard_large_jsonl_for_upload(original)
    if any(p.suffix == ".jsonl" and p.stat().st_size > 9_500_000 for p in retained):
        raise ValueError("JSONL record exceeds the 9.5MB upload limit")
    result = {}
    for path, expected in large.items():
        manifest = json.loads(path.with_name(f"{path.stem}.manifest.json").read_text())
        digest = hashlib.sha256()
        for part in manifest["parts"]:
            piece = path.parent / part
            if piece.stat().st_size >= 9_000_000:
                raise ValueError(f"JSONL record exceeds the <9MB archival shard target: {piece}")
            if sha(piece) != manifest["sha256"][part]:
                raise ValueError(f"Pre-upload shard hash mismatch: {piece}")
            digest.update(piece.read_bytes())
        if digest.hexdigest() != expected:
            raise ValueError(f"Pre-upload sharding changed original bytes: {path}")
        result[str(path.relative_to(stage))] = expected
    return result


def run(root: Path, stage: Path, readback: Path, phase: str) -> dict:
    started = time.monotonic()
    print(f"[archive] start phase={phase} stage={stage}", flush=True)
    if root.resolve() != RUN_ROOT.resolve() or phase not in {"pilot", "raw", "final"}:
        raise ValueError("This archive helper is bound to the approved highrate run")
    resolved = [p.resolve() for p in (root, stage, readback)]
    if any(
        a.is_relative_to(b)
        for i, a in enumerate(resolved)
        for j, b in enumerate(resolved)
        if i != j
    ):
        raise ValueError("Run, stage and readback trees must be disjoint")
    sources = source_hashes()
    manifest_path = stage / "snapshot_manifest.json"
    manifest_sha = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    original = manifest["files"]
    if not original or manifest["phase"] != phase:
        raise ValueError("Nonempty phase-matched snapshot manifest required")
    entries = list(stage.rglob("*"))
    if any(p.is_symlink() for p in [stage, *entries]):
        raise ValueError("Archive stage cannot contain symlinks")
    if {str(p.relative_to(stage)) for p in entries if p.is_file()} != {
        *original,
        "snapshot_manifest.json",
    }:
        raise ValueError("Use a pristine stage containing exactly the declared original files")
    if os.environ.get("EPM_HF_FILECOUNT_FALLBACK") != "0":
        raise ValueError("Set EPM_HF_FILECOUNT_FALLBACK=0 to bind the canonical destination")
    if readback.exists():
        raise ValueError("Use a fresh readback directory for an independent remote round trip")
    for name, expected in original.items():
        path = stage / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(stage.resolve()):
            raise ValueError(f"Snapshot path escapes staging: {name}")
        check_original(path, expected)
        # The existing sharder reads text with universal newlines. Refuse
        # CR-containing JSONL instead of silently changing its original bytes.
        if path.suffix == ".jsonl" and b"\r" in path.read_bytes():
            raise ValueError(f"JSONL contains raw CR and cannot use this sharder: {name}")
    print(f"[archive] original files verified count={len(original)}", flush=True)
    shard_proof = prepare_shards(stage)
    archive.ROOT = root
    archive.HF_PREFIX = "context_risk/issue2670_highrate"
    receipt = archive.upload(stage, phase)
    print(f"[archive] uploaded and API-verified files={receipt['verified_files']}", flush=True)
    if sha(manifest_path) != manifest_sha:
        raise ValueError("Snapshot manifest changed during upload")
    landed = hub.stage_hub_prefix(
        receipt["repo_id"],
        receipt["prefix"],
        readback,
        repo_type="dataset",
        revision=receipt["revision"],
        max_workers=6,
    )
    if {str(p.relative_to(readback)) for p in landed} != set(receipt["files"]):
        raise ValueError("Downloaded file set differs from upload receipt")
    for name, expected in receipt["files"].items():
        path = readback / name
        if path.stat().st_size != expected["size"] or sha(path) != expected["sha256"]:
            raise ValueError(f"Remote readback hash mismatch: {name}")
    restored = {}
    for index, (name, expected) in enumerate(original.items(), 1):
        remote_name = f"{receipt['prefix']}/{name}"
        if name.endswith(".jsonl"):
            # Always use the canonical manifest-first consumer, including for
            # currently small files that may cross the threshold in later phases.
            path = hub.stage_sharded_text(
                receipt["repo_id"],
                remote_name,
                readback / "reconstructed" / name,
                repo_type="dataset",
                revision=receipt["revision"],
                overwrite=True,
            )
        else:
            path = readback / remote_name
        restored[name] = check_original(path, expected)
        print(f"[archive] parsed unit {index}/{len(original)} {name}", flush=True)
    if sources != source_hashes() or sha(manifest_path) != manifest_sha:
        raise ValueError("Archive implementation or manifest changed during verification")
    result = {
        "passed": True,
        "phase": phase,
        "repo_id": receipt["repo_id"],
        "revision": receipt["revision"],
        "prefix": receipt["prefix"],
        "snapshot_sha256": manifest_sha,
        "sources_sha256": sources,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("inspect-ai", "huggingface-hub", "numpy")
        },
        "byte_exact_sharded_sources": shard_proof,
        "upload_receipt_sha256": sha(root / f"{phase}_upload_receipt.json"),
        "downloaded_files": len(landed),
        "original_files": restored,
        "elapsed_seconds": time.monotonic() - started,
        "url": receipt["url"],
        "scope": "Exact remote filename, size and hash verification plus canonical staged readback, byte-exact JSONL reconstruction, and native/tensor parsing. This is not a teardown authorization.",
    }
    archive.write_json(root / f"{phase}_readback_receipt.json", result)
    print(f"[archive] complete phase={phase} elapsed={result['elapsed_seconds']:.3f}s", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "raw", "final"), required=True)
    args = parser.parse_args()
    run(args.root, args.stage, args.readback, args.phase)
