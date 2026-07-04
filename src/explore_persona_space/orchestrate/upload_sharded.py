"""Incremental, disk-bounded shard upload with overflow rerouting (v2).

The EPS workflow v2 upload policy has NO ceiling: every artifact tries the
main HF repo first, reroutes to the private overflow repo on quota pressure,
and discards only when BOTH quotas are exhausted (always alerted). Big stores
(activation grids, per-context tensors) can exceed the 128 GB per-issue ext4
quota / ~130 GB MooseFS per-pod quota, so a store larger than the disk cannot
be uploaded in one shot. :func:`upload_dir_sharded` walks its shard files one
at a time — upload one shard, verify it landed, delete the local copy — so the
peak local footprint is ~one shard regardless of the store's total size.

Reuses the existing overflow mechanism from
:mod:`explore_persona_space.orchestrate.hub` (do NOT reinvent it):

- :data:`hub.DEFAULT_OVERFLOW_REPO` — the private overflow repo id.
- :func:`hub._is_storage_quota_403` — the persistent public-storage-quota-403
  predicate (`403` + `storage`; NOT a transient 5xx).
- :func:`hub._emit_overflow_routing_event` — the fail-soft JSONL deviation
  breadcrumb the orchestrator / upload-verifier drains into an `epm:` marker.
- :func:`hub.list_repo_files_complete` — the truncation-safe (paginated,
  504-retried) post-upload verify listing.

The one thing hub's mechanism cannot do for a dataset shard is the canonical
``OVERFLOW_POINTER.json`` breadcrumb: ``hub._write_overflow_pointer`` hardcodes
``repo_type="model"``, so a dataset canonical repo needs the analogous pattern
here (:func:`_write_overflow_pointer`), keyed on the canonical repo's own type.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.hub import (
    DEFAULT_OVERFLOW_REPO,
    _emit_overflow_routing_event,
    _is_storage_quota_403,
    list_repo_files_complete,
)

logger = logging.getLogger(__name__)


@dataclass
class ShardUploadResult:
    """Per-run summary of a sharded upload."""

    repo_id: str
    overflow_repo: str = DEFAULT_OVERFLOW_REPO
    uploaded: list[str] = field(default_factory=list)  # dest paths (any repo)
    rerouted: list[str] = field(default_factory=list)  # dest paths sent to overflow
    deleted: list[str] = field(default_factory=list)  # local shard names removed


def _default_api(token: str | None):
    from huggingface_hub import HfApi

    return HfApi(token=token or os.environ.get("HF_TOKEN"))


def _write_overflow_pointer(
    api,
    *,
    canonical_repo: str,
    canonical_repo_type: str,
    path_in_repo: str,
    overflow_repo: str,
) -> None:
    """Commit a small ``OVERFLOW_POINTER.json`` breadcrumb to the CANONICAL
    repo after a reroute.

    Small JSON commits ride the non-LFS path, which SUCCEEDS while over the
    public-storage quota (#541-validated), so a consumer/verifier listing the
    canonical subfolder always finds a machine-readable pointer to the real
    location instead of an empty path. Fail-soft: a pointer-write failure logs
    loudly but never fails the already-verified rerouted upload.

    Sibling of ``hub._write_overflow_pointer``, generalised to the canonical
    repo's own ``repo_type`` (hub's writer is model-repo-only).
    """
    dest = (
        f"{path_in_repo.rstrip('/')}/OVERFLOW_POINTER.json"
        if path_in_repo
        else "OVERFLOW_POINTER.json"
    )
    payload = {
        "overflow_repo": overflow_repo,
        "path_in_repo": path_in_repo,
        "ts": time.time(),
    }
    try:
        api.upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(payload, indent=2).encode("utf-8")),
            repo_id=canonical_repo,
            path_in_repo=dest,
            repo_type=canonical_repo_type,
        )
        logger.info("Wrote overflow pointer %s/%s -> %s", canonical_repo, dest, overflow_repo)
    except Exception as e:
        logger.warning(
            "overflow pointer write to %s failed (%s) — rerouted upload remains at %s",
            canonical_repo,
            e,
            overflow_repo,
        )


def _reroute_to_overflow(
    api,
    *,
    shard: Path,
    dest: str,
    canonical_repo: str,
    canonical_repo_type: str,
) -> str:
    """Upload one shard to the private overflow repo after a quota-403 on the
    canonical repo. Writes the pointer breadcrumb + deviation event.

    Returns the overflow repo id (the effective destination). Raises if the
    overflow upload ITSELF fails — that is the both-quotas-exhausted terminal.
    """
    logger.warning(
        "quota-403 on %s — rerouting shard %s -> %s (overflow)",
        canonical_repo,
        shard.name,
        DEFAULT_OVERFLOW_REPO,
    )
    # Overflow repo is private (separate LFS quota with headroom); created if
    # missing, matching the existing hub reroute contract.
    api.create_repo(
        repo_id=DEFAULT_OVERFLOW_REPO,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=str(shard),
        repo_id=DEFAULT_OVERFLOW_REPO,
        path_in_repo=dest,
        repo_type="model",
    )
    _emit_overflow_routing_event(
        original_repo=canonical_repo,
        effective_repo=DEFAULT_OVERFLOW_REPO,
        path_in_repo=dest,
    )
    _write_overflow_pointer(
        api,
        canonical_repo=canonical_repo,
        canonical_repo_type=canonical_repo_type,
        path_in_repo=os.path.dirname(dest),
        overflow_repo=DEFAULT_OVERFLOW_REPO,
    )
    return DEFAULT_OVERFLOW_REPO


def _verify_present(api, *, repo_id: str, repo_type: str, dest: str) -> bool:
    files = list_repo_files_complete(api, repo_id, repo_type=repo_type)
    return dest in set(files)


def upload_dir_sharded(
    local_dir: Path | str,
    repo_id: str,
    path_in_repo: str,
    *,
    repo_type: str = "dataset",
    shard_glob: str = "*",
    verify: bool = True,
    delete_local: bool = True,
    api=None,
    token: str | None = None,
) -> ShardUploadResult:
    """Upload ``local_dir``'s shard files one at a time, bounding local footprint.

    Per shard, in order: upload to ``repo_id`` → (on quota-403) reroute to the
    private overflow repo → verify the shard lists at its destination → delete
    the local shard (only when ``delete_local`` and verification passed).

    Args:
        local_dir: Directory holding the shard files.
        repo_id: Canonical HF repo id (the main data/model repo).
        path_in_repo: Sub-path in the repo the shards land under.
        repo_type: ``'dataset'`` (default) or ``'model'`` for the CANONICAL
            repo. The overflow repo is always addressed as ``model`` (matching
            hub's reroute).
        shard_glob: Glob (relative to ``local_dir``, non-recursive) selecting
            shard files. Default ``'*'`` (every file).
        verify: HEAD-verify each shard lists at its destination before delete.
        delete_local: Delete each local shard after a verified upload.
        api: An ``huggingface_hub.HfApi`` (constructed from ``token`` / env if
            None) — injectable for testing.
        token: HF token override (default ``$HF_TOKEN``).

    Returns:
        A :class:`ShardUploadResult` summarising what uploaded / rerouted /
        deleted.

    Raises:
        FileNotFoundError: ``local_dir`` is missing.
        RuntimeError: a shard failed to verify, or BOTH the canonical and
            overflow uploads refused (fail-loud — the shard is never silently
            dropped, and a discard-with-regen-recipe is a caller-side decision
            made only after this raises).
    """
    local = Path(local_dir)
    if not local.is_dir():
        raise FileNotFoundError(f"shard directory not found: {local}")
    if api is None:
        api = _default_api(token)

    prefix = path_in_repo.rstrip("/")
    result = ShardUploadResult(repo_id=repo_id)

    shards = sorted(p for p in local.glob(shard_glob) if p.is_file())
    for shard in shards:
        dest = f"{prefix}/{shard.name}" if prefix else shard.name
        try:
            api.upload_file(
                path_or_fileobj=str(shard),
                repo_id=repo_id,
                path_in_repo=dest,
                repo_type=repo_type,
            )
            effective_repo = repo_id
            effective_repo_type = repo_type
        except Exception as exc:
            if not _is_storage_quota_403(exc):
                # Non-quota failure: fail loud, do not reroute or delete.
                raise
            try:
                effective_repo = _reroute_to_overflow(
                    api,
                    shard=shard,
                    dest=dest,
                    canonical_repo=repo_id,
                    canonical_repo_type=repo_type,
                )
            except Exception as reroute_exc:
                raise RuntimeError(
                    f"both main ({repo_id}) and overflow ({DEFAULT_OVERFLOW_REPO}) repos "
                    f"refused shard {shard.name!r}; not deleting local copy. "
                    f"A discard-with-regen-recipe is the caller's decision, always alerted."
                ) from reroute_exc
            effective_repo_type = "model"
            result.rerouted.append(dest)

        result.uploaded.append(dest)

        verified = True
        if verify:
            verified = _verify_present(
                api, repo_id=effective_repo, repo_type=effective_repo_type, dest=dest
            )
            if not verified:
                raise RuntimeError(
                    f"shard {shard.name!r} not found at {effective_repo}:{dest} after upload; "
                    f"not deleting local copy."
                )

        if delete_local and verified:
            shard.unlink()
            result.deleted.append(shard.name)

    return result
