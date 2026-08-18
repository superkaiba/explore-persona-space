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
import time
from pathlib import Path

__all__ = ["REPACKED_PREFIXES", "PackedPrefixError", "packed_fallback", "read_packed"]

_DOWNLOAD_ATTEMPTS = 3

# The 8 #2332 repack target prefixes (canonical home — the repack tooling and
# the stage_hub_file packed fallback both key on this tuple).
REPACKED_PREFIXES = (
    "issue1481_conpos_grid",
    "issue1090_pvdatagen",
    "issue1586_methodgen",
    "issue667_alllayer",
    "issue1434_writingstyle",
    "issue1739_ctxmap",
    "issue2224_screening",
    "issue1489_ctx_aug",
)


class PackedPrefixError(RuntimeError):
    """Integrity failure resolving a packed member (index/shard disagreement,
    offset/size mismatch, or sha256 mismatch of the extracted bytes)."""


def _download(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """Network boundary: fetch one repo file via ``hf_hub_download`` under a
    bounded transient retry (huggingface_hub natively retries only 429 here).

    Typed not-found errors propagate IMMEDIATELY, never retried — the caller
    owns their semantics (a missing ``index.json`` IS the unpacked-prefix
    signal). Returns the local cached path. Module-level seam so tests can
    monkeypatch it; uses the default HF cache (no ``local_dir``), so repeated
    member reads from one shard download the shard once.
    """
    from huggingface_hub import hf_hub_download  # lazy: offline callers never import HF
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    last: Exception | None = None
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        try:
            # NO_RETRY: this loop IS the bounded retry (3 attempts, exp backoff)
            return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
        except (EntryNotFoundError, RepositoryNotFoundError):
            raise
        except Exception as e:
            last = e
            if attempt < _DOWNLOAD_ATTEMPTS - 1:
                time.sleep(2 * 4**attempt)
    assert last is not None
    raise last


def packed_fallback(repo_id: str, orig_path: str, *, repo_type: str = "dataset") -> bytes | None:
    """Central-seam fallback probe (#2332 review r2): the bytes of ``orig_path``
    from its packed shard, or ``None`` when the packed route does not apply.

    ``None`` (caller re-raises its ORIGINAL not-found error unchanged) when:
    the first path segment is not one of :data:`REPACKED_PREFIXES`; the
    subpath after the prefix is empty (a degenerate trailing-slash path); the
    prefix's merged ``index.json`` does not exist on the repo (not repacked
    yet); or the index exists but ``orig_path`` is not a member (the file
    genuinely never existed). A :class:`PackedPrefixError` (integrity failure
    once the packed route IS live — a shard the index names but the repo
    lacks, a malformed index entry, an offset/sha mismatch; ``read_packed``
    converts the first two to :class:`PackedPrefixError` itself, r3 review)
    PROPAGATES — never swallowed into None.
    """
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    stripped = orig_path.lstrip("/")
    first, _, rest = stripped.partition("/")
    if not rest.strip("/") or first not in REPACKED_PREFIXES:
        return None
    try:
        return read_packed(repo_id, stripped, repo_type=repo_type)
    except (EntryNotFoundError, RepositoryNotFoundError, KeyError):
        # Reachable ONLY from the index-download hop (not-repacked prefix /
        # repo gone) or the member lookup (path never existed) — the shard
        # hop and entry-shape validation raise PackedPrefixError instead.
        return None


def read_packed(repo_id: str, orig_path: str, *, repo_type: str = "dataset") -> bytes:
    """Read the bytes of pre-repack ``orig_path`` from its packed tar shard.

    Resolution chain (fail-loud at every hop): prefix = first path segment
    of ``orig_path`` -> download ``<prefix>/__packed__/index.json`` -> look
    up ``orig_path`` -> download the ONE named shard -> locate the member by
    NAME (``tarfile``; the index-recorded ``offset``/``size`` are
    cross-checked against the located member) -> extract -> sha256-verify
    against the index entry -> return bytes.

    Raises:
        ValueError: ``orig_path`` is absolute (leading ``/``) or carries no
            prefix segment (no ``/``).
        KeyError: ``orig_path`` is absent from the prefix's packed index
            (message names the index path checked).
        PackedPrefixError: malformed index entry (missing required keys), a
            shard the index names but the repo lacks (an index that
            references a missing shard is CORRUPTION, never "file not
            found" — r3 review), member absent from the shard, offset/size
            disagreement with the index, non-regular member, or sha256
            mismatch of the extracted bytes.
    """
    if orig_path.startswith("/"):
        raise ValueError(
            f"orig_path {orig_path!r} is absolute — pass the repo-relative path "
            "(a leading slash would derive an EMPTY prefix segment)"
        )
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
    missing_keys = [k for k in ("shard", "offset", "size", "sha256") if k not in entry]
    if missing_keys:
        raise PackedPrefixError(
            f"{orig_path!r}: malformed packed index entry in {index_repo_path} — missing "
            f"key(s) {missing_keys} (index corruption is never 'file not found'; r3 review)"
        )
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        shard_local = Path(_download(repo_id, f"{prefix}/__packed__/{entry['shard']}", repo_type))
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        raise PackedPrefixError(
            f"{orig_path!r}: packed index names shard {entry['shard']!r} but the shard is "
            f"missing from {repo_id} — index/shard disagree (an index that references a "
            "missing shard is corruption, never 'file not found'; r3 review)"
        ) from e
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
