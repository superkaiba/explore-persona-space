"""Path-preserving accessor for #2332-repacked HF prefixes (plan v2 SS4.3 Option 2).

The #2332 repack (``scripts/issue2332_repack_prefixes.py``) replaces each
many-tiny-file prefix on the canonical data repo with a few POSIX tar shards
plus a sidecar index under ``<prefix>/__packed__/``:

    <prefix>/__packed__/shard-00000.tar   # members named by original repo-relative path
    <prefix>/__packed__/index.json        # {orig_path: {shard, offset, size, sha256}}
    <prefix>/__packed__/manifest.json     # provenance (n_members, shard sha256s, src_revision)

``read_packed(repo_id, orig_path)`` restores the pre-repack read with a
one-line change at the reader: it resolves ``orig_path`` through the merged
``index.json``, downloads the ONE shard holding the member, extracts it by
NAME via ``tarfile`` (cross-checking the recorded offset/size), and
sha256-verifies the extracted bytes against the index entry BEFORE returning
them — a mismatch raises; unverified bytes are never returned. No silent
fallbacks anywhere in the chain.

Chunked-mode naming note: a chunked repack writes per-chunk sidecars
(``index-c<chunk>.json``, shards ``shard-c<chunk>-NNNNN.tar``) during the
run and merges them into the final ``index.json`` after the last chunk.
This accessor reads ONLY the merged ``index.json``; the shard filenames
recorded there already carry the chunked names, so no chunk logic exists
(or is needed) here.

Index semantics (must match the packer, ``step_pack``): ``offset`` is the
tar member's ``TarInfo.offset_data`` (byte offset where the member's DATA
begins inside the uncompressed shard); ``sha256`` is the hash of the staged
original file's bytes, computed at pack time.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

__all__ = ["PackedPrefixError", "read_packed"]


class PackedPrefixError(RuntimeError):
    """Integrity failure resolving a packed member (index/shard disagreement,
    offset/size mismatch, or sha256 mismatch of the extracted bytes)."""


def _download(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """Network boundary: fetch one repo file via ``hf_hub_download``.

    Returns the local cached path. Module-level seam so tests can
    monkeypatch it; uses the default HF cache (no ``local_dir``), so
    repeated member reads from one shard download the shard once.
    """
    from huggingface_hub import hf_hub_download  # lazy: offline callers never import HF

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def read_packed(repo_id: str, orig_path: str, *, repo_type: str = "dataset") -> bytes:
    """Read the bytes of pre-repack ``orig_path`` from its packed tar shard.

    Resolution chain (fail-loud at every hop): prefix = first path segment
    of ``orig_path`` -> download ``<prefix>/__packed__/index.json`` -> look
    up ``orig_path`` -> download the ONE named shard -> locate the member by
    NAME (``tarfile``; the index-recorded ``offset``/``size`` are
    cross-checked against the located member) -> extract -> sha256-verify
    against the index entry -> return bytes.

    Raises:
        ValueError: ``orig_path`` carries no prefix segment (no ``/``).
        KeyError: ``orig_path`` is absent from the prefix's packed index
            (message names the index path checked).
        PackedPrefixError: member absent from the shard, offset/size
            disagreement with the index, non-regular member, or sha256
            mismatch of the extracted bytes.
    """
    if "/" not in orig_path.strip("/"):
        raise ValueError(
            f"orig_path {orig_path!r} has no prefix segment — expected '<prefix>/<subpath>'"
        )
    prefix = orig_path.split("/", 1)[0]
    index_repo_path = f"{prefix}/__packed__/index.json"
    index = json.loads(Path(_download(repo_id, index_repo_path, repo_type)).read_text())
    entry = index.get(orig_path)
    if entry is None:
        raise KeyError(
            f"{orig_path!r} not found in packed index {repo_id}:{index_repo_path} "
            f"({len(index):,} members)"
        )
    shard_local = Path(_download(repo_id, f"{prefix}/__packed__/{entry['shard']}", repo_type))
    with tarfile.open(shard_local, "r:") as tar:  # "r:" = uncompressed only, by design
        try:
            member = tar.getmember(orig_path)
        except KeyError as e:
            raise PackedPrefixError(
                f"index names {orig_path!r} in {entry['shard']} but the shard has no such "
                "member — index/shard disagree"
            ) from e
        if member.offset_data != entry["offset"] or member.size != entry["size"]:
            raise PackedPrefixError(
                f"{orig_path!r}: shard member (offset_data={member.offset_data}, "
                f"size={member.size}) != index entry (offset={entry['offset']}, "
                f"size={entry['size']})"
            )
        fh = tar.extractfile(member)
        if fh is None:
            raise PackedPrefixError(f"{orig_path!r}: non-regular tar member in {entry['shard']}")
        data = fh.read()
    got = hashlib.sha256(data).hexdigest()
    if got != entry["sha256"]:
        raise PackedPrefixError(
            f"{orig_path!r}: sha256 mismatch after extraction — got {got}, index says "
            f"{entry['sha256']} (unverified bytes are never returned)"
        )
    return data
