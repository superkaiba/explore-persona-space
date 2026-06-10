#!/usr/bin/env python3
"""Quota-resilient HF upload helpers for experiment #541 (round-6 fix).

Context (bug_class ``hf_public_storage_quota_exceeded``, 2026-06-10): the HF
account is over its PUBLIC storage quota (11.3 TB), so ALL new LFS uploads
403 account-wide (``post_lfs_batch_info`` -> "You have exceeded your public
storage space"). Empirically validated facts this module is built on:

1. Regular (non-LFS) git commits to ``datasets/superkaiba1/
   explore-persona-space-data`` SUCCEED while over quota.
2. ``*.json`` / ``*.jsonl`` are NOT LFS-matched in that repo's
   ``.gitattributes``; ``huggingface_hub.upload_file`` only force-routes
   files > 10 MB to LFS.
3. LFS uploads to the PRIVATE model repo
   ``superkaiba1/explore-persona-space-overflow`` SUCCEED (private quota is
   separate and has headroom).

Therefore: text payloads upload as regular git blobs, with any file
>= ``SHARD_TRIGGER_BYTES`` line-split into < ``SHARD_MAX_BYTES`` shards plus
a reassembly manifest; adapter folders (LFS-heavy safetensors) upload to the
private overflow repo. Do NOT gzip anything — ``*.gz`` IS LFS-matched in the
data repo's ``.gitattributes``.

Kept deliberately import-light (stdlib + ``huggingface_hub`` only) so the
helpers can be smoke-run end-to-end on the GPU-less VM without pulling the
full #444 driver bootstrap.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# A >= 9.5 MB file is sharded; huggingface_hub force-routes > 10 MB files to
# LFS (which 403s account-wide while over public quota), and 9.5 MB leaves a
# safety margin below that cliff. Shards are kept strictly < 9 MB.
SHARD_TRIGGER_BYTES = int(9.5 * 1024 * 1024)
SHARD_MAX_BYTES = 9 * 1024 * 1024
# Hard ceiling: NOTHING this module uploads to the data repo may reach the
# 10 MB LFS auto-routing threshold.
LFS_FORCE_BYTES = 10 * 1024 * 1024

MANIFEST_SCHEMA = "issue541-shard-manifest-v1"


def _sha256_file(path: Path) -> str:
    """Streaming sha256 hexdigest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_jsonl_into_shards(local_path: Path, workdir: Path) -> tuple[list[Path], Path]:
    """Line-split ``local_path`` into ``<stem>.shardNN.jsonl`` files < 9 MB each.

    Returns ``(shard_paths_in_order, manifest_path)``. The manifest
    (``<stem>.manifest.json``) records the ordered part list with per-part
    line counts + sha256, and the total line count; reassembly = concatenate
    shard lines in order (``cat *.shard*.jsonl`` in shard order reproduces the
    original byte-for-byte).

    Raises ``ValueError`` for non-``.jsonl`` inputs (only line-oriented files
    can be line-split safely) and for any single line >= ``SHARD_MAX_BYTES``.
    """
    if local_path.suffix != ".jsonl":
        raise ValueError(
            f"refusing to line-split non-jsonl file {local_path} — only .jsonl payloads "
            "have a safe line-oriented split; handle other large files explicitly."
        )
    workdir.mkdir(parents=True, exist_ok=True)
    stem = local_path.stem  # e.g. baseline_completions_<slug>

    shard_paths: list[Path] = []
    parts_meta: list[dict[str, Any]] = []
    total_lines = 0

    current_lines: list[bytes] = []
    current_bytes = 0

    def _flush() -> None:
        nonlocal current_lines, current_bytes
        if not current_lines:
            return
        idx = len(shard_paths)
        shard = workdir / f"{stem}.shard{idx:02d}.jsonl"
        shard.write_bytes(b"".join(current_lines))
        assert shard.stat().st_size < SHARD_MAX_BYTES, (shard, shard.stat().st_size)
        shard_paths.append(shard)
        parts_meta.append(
            {
                "filename": shard.name,
                "n_lines": len(current_lines),
                "n_bytes": shard.stat().st_size,
                "sha256": _sha256_file(shard),
            }
        )
        current_lines = []
        current_bytes = 0

    with local_path.open("rb") as f:
        for line in f:  # keeps line terminators, so concat == original bytes
            if len(line) >= SHARD_MAX_BYTES:
                raise ValueError(
                    f"single line of {len(line)} bytes in {local_path} exceeds the "
                    f"{SHARD_MAX_BYTES}-byte shard cap — cannot line-split."
                )
            if current_bytes + len(line) > SHARD_MAX_BYTES:
                _flush()
            current_lines.append(line)
            current_bytes += len(line)
            total_lines += 1
    _flush()

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "original_filename": local_path.name,
        "original_n_bytes": local_path.stat().st_size,
        "original_sha256": _sha256_file(local_path),
        "total_lines": total_lines,
        "n_parts": len(shard_paths),
        "reassembly": (
            "concatenate shard lines in order (parts are listed in order; "
            "byte-concatenation of the parts reproduces original_sha256)"
        ),
        "parts": parts_meta,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = workdir / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return shard_paths, manifest_path


def upload_text_file(
    api: Any,
    *,
    local_path: Path,
    path_in_repo: str,
    repo_id: str,
    existing: set[str],
    workdir: Path,
    repo_type: str = "dataset",
) -> dict[str, Any]:
    """Upload one text payload file as regular (non-LFS) git blob(s).

    Files >= ``SHARD_TRIGGER_BYTES`` are line-split via
    ``split_jsonl_into_shards``; shards + manifest land in the SAME repo
    directory as ``path_in_repo`` (only the oversized file gains the shard
    suffix). Paths already in ``existing`` are skipped (idempotent resume —
    callers fetch ``existing`` from ``list_repo_files`` ONCE per phase; a
    stale/truncated listing degrades to a harmless re-upload, never loss).

    Asserts every uploaded object stays < ``LFS_FORCE_BYTES`` (the 10 MB
    LFS auto-routing cliff that 403s while over public quota).

    Returns ``{"uploaded": [...], "skipped": [...], "sharded": bool,
    "manifest_path_in_repo": str | None}``.
    """
    size = local_path.stat().st_size
    repo_dir = path_in_repo.rsplit("/", 1)[0] if "/" in path_in_repo else ""

    def _dest(name: str) -> str:
        return f"{repo_dir}/{name}" if repo_dir else name

    uploaded: list[str] = []
    skipped: list[str] = []

    def _put(fp: Path, dest: str) -> None:
        if dest in existing:
            skipped.append(dest)
            print(f"[upload-541] skip existing: {repo_id}:{dest}")
            return
        fp_size = fp.stat().st_size
        assert fp_size < LFS_FORCE_BYTES, (
            f"{fp} is {fp_size} bytes — would auto-route to LFS (>= {LFS_FORCE_BYTES}); "
            "refusing (account is over public LFS quota)."
        )
        api.upload_file(
            path_or_fileobj=str(fp),
            path_in_repo=dest,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        uploaded.append(dest)
        print(f"[upload-541] -> {repo_id}:{dest} ({fp_size} bytes, git blob)")

    if size >= SHARD_TRIGGER_BYTES:
        shard_paths, manifest_path = split_jsonl_into_shards(local_path, workdir)
        for shard in shard_paths:
            _put(shard, _dest(shard.name))
        _put(manifest_path, _dest(manifest_path.name))
        return {
            "uploaded": uploaded,
            "skipped": skipped,
            "sharded": True,
            "manifest_path_in_repo": _dest(manifest_path.name),
        }

    _put(local_path, path_in_repo)
    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "sharded": False,
        "manifest_path_in_repo": None,
    }


def upload_adapter_dir(
    api: Any,
    *,
    local_dir: Path,
    path_in_repo: str,
    repo_id: str,
    existing: set[str],
) -> dict[str, Any]:
    """Upload one LoRA adapter directory to the (private) overflow model repo.

    LFS is fine on the private repo (separate quota with headroom — probed
    2026-06-10). Idempotent: skipped when the adapter's
    ``adapter_config.json`` is already present under ``path_in_repo`` in
    ``existing``. Raises if ``local_dir`` is missing and the adapter is NOT
    already on the repo (fail-loud: a missing, un-persisted adapter is data
    loss, not a skip).
    """
    skip_marker = f"{path_in_repo}/adapter_config.json"
    url = f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}"
    if skip_marker in existing:
        print(f"[upload-541] skip existing adapter: {repo_id}:{path_in_repo}")
        return {"path_in_repo": path_in_repo, "url": url, "status": "skipped-existing"}
    if not local_dir.is_dir():
        raise FileNotFoundError(
            f"adapter dir {local_dir} missing and {repo_id}:{skip_marker} not on the Hub — "
            "cannot persist; refusing to continue (the adapter would be lost on pod "
            "termination)."
        )
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[upload-541] adapter -> {repo_id}:{path_in_repo}")
    return {"path_in_repo": path_in_repo, "url": url, "status": "uploaded"}
