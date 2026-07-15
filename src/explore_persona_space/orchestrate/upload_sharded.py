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
- :func:`hub.list_hf_files_under_path` — the server-side SCOPED (paginated,
  504-retried) post-upload verify probe (#920/#988: never a full-repo listing
  per shard).

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
    _repo_is_private,
    check_projected_upload_headroom,
    list_hf_files_under_path,
    retry_transient,
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
        # BytesIO built INSIDE the thunk: a retried attempt must re-read the
        # payload from position 0, not an exhausted stream (#1335).
        retry_transient(
            lambda: api.upload_file(
                path_or_fileobj=io.BytesIO(json.dumps(payload, indent=2).encode("utf-8")),
                repo_id=canonical_repo,
                path_in_repo=dest,
                repo_type=canonical_repo_type,
            ),
            what=f"upload_file({canonical_repo}/{dest})",
        )
        logger.info("Wrote overflow pointer %s/%s -> %s", canonical_repo, dest, overflow_repo)
    except Exception as e:
        logger.warning(
            "overflow pointer write to %s failed (%s) — rerouted upload remains at %s",
            canonical_repo,
            e,
            overflow_repo,
        )


def _ensure_overflow_repo(api) -> None:
    """Create the private overflow repo if missing (idempotent).

    Overflow repo is private (separate LFS quota with headroom); created if
    missing, matching the existing hub reroute contract. Shared by the
    reactive quota-403 branch and the #1034 proactive projected-headroom
    branch.
    """
    retry_transient(
        lambda: api.create_repo(
            repo_id=DEFAULT_OVERFLOW_REPO,
            repo_type="model",
            private=True,
            exist_ok=True,
        ),
        what=f"create_repo({DEFAULT_OVERFLOW_REPO})",
    )


def _reroute_to_overflow(
    api,
    *,
    shard: Path,
    dest: str,
    canonical_repo: str,
    canonical_repo_type: str,
    emitted_prefixes: set[str] | None = None,
) -> str:
    """Upload one shard to the private overflow repo after a quota-403 on the
    canonical repo. Writes the pointer breadcrumb + deviation event ONCE per
    ``path_in_repo`` prefix per ``upload_dir_sharded`` call (#1034 dedup —
    the pointer dest is already prefix-level, so N per-shard commits of the
    same file were pure duplicate Hub commits against the 256 commits/hr cap;
    ``emitted_prefixes=None`` keeps the legacy emit-every-time behavior for
    direct callers).

    Returns the overflow repo id (the effective destination). Raises if the
    overflow upload ITSELF fails — that is the both-quotas-exhausted terminal.
    """
    logger.warning(
        "quota-403 on %s — rerouting shard %s -> %s (overflow)",
        canonical_repo,
        shard.name,
        DEFAULT_OVERFLOW_REPO,
    )
    _ensure_overflow_repo(api)
    retry_transient(
        lambda: api.upload_file(
            path_or_fileobj=str(shard),
            repo_id=DEFAULT_OVERFLOW_REPO,
            path_in_repo=dest,
            repo_type="model",
        ),
        what=f"upload_file({DEFAULT_OVERFLOW_REPO}/{dest})",
    )
    prefix = os.path.dirname(dest)
    if emitted_prefixes is None or prefix not in emitted_prefixes:
        _emit_overflow_routing_event(
            original_repo=canonical_repo,
            effective_repo=DEFAULT_OVERFLOW_REPO,
            path_in_repo=dest,
        )
        _write_overflow_pointer(
            api,
            canonical_repo=canonical_repo,
            canonical_repo_type=canonical_repo_type,
            path_in_repo=prefix,
            overflow_repo=DEFAULT_OVERFLOW_REPO,
        )
        if emitted_prefixes is not None:
            emitted_prefixes.add(prefix)
    return DEFAULT_OVERFLOW_REPO


def _verify_present(api, *, repo_id: str, repo_type: str, dest: str) -> bool:
    """True iff the exact shard file ``dest`` is present on the Hub.

    Exact-file probe (#920/#988): never full-list the repo per shard — the
    per-shard full listing was the worst repeat offender in the sharded
    upload loop (>600 s per shard on the ~1M-file data repo). A file path
    resolves via the helper's EntryNotFoundError -> file_exists fallback
    (RETRIED inside hub as of #1335); the pathological case where ``dest``
    names a DIRECTORY returns files UNDER it (none equal to ``dest``), so the
    membership test still returns False — same as the old full-listing check.

    #1335: this per-file probe is now the ROOT-LEVEL FALLBACK only
    (``path_in_repo=""`` — no directory to scope a listing on).
    ``upload_dir_sharded`` verifies via ONE prefix-scoped directory listing
    per destination repo per call (the batched verify), never a per-shard
    probe loop: att-20260715-134136 crashed on a transient 429 raised by the
    per-shard probe's fresh Hub call during a production store upload.
    """
    return dest in set(list_hf_files_under_path(api, repo_id, dest, repo_type=repo_type))


def _batched_verify(api, pending: list[tuple[Path, str, str, str]], *, prefix: str) -> None:
    """Raise unless every uploaded dest lists at its destination repo.

    Batched post-upload verify (#1335): ONE prefix-scoped directory listing
    per destination repo per ``upload_dir_sharded`` call — never a per-shard
    exact-file probe loop (att-20260715-134136: the per-file fallback's fresh
    Hub call 429'd ~2.8h into production, one probe per shard). The listing
    rides hub's transient retry internally (``list_repo_files_complete`` ->
    ``_retry_upload``). Root-level dests (``path_in_repo=""``) have no
    directory to scope on and fall back to per-dest exact-file probes
    (retried inside hub as of #1335).
    """
    missing: list[str] = []
    for eff_repo, eff_type in sorted({(r, t) for _, _, r, t in pending}):
        group = [d for _, d, r, t in pending if (r, t) == (eff_repo, eff_type)]
        if prefix:
            present = set(list_hf_files_under_path(api, eff_repo, prefix, repo_type=eff_type))
            found = {d for d in group if d in present}
        else:
            found = {
                d
                for d in group
                if _verify_present(api, repo_id=eff_repo, repo_type=eff_type, dest=d)
            }
        missing.extend(f"{eff_repo}:{d}" for d in group if d not in found)
    if missing:
        raise RuntimeError(
            f"{len(missing)} shard(s) not found at their destination after upload "
            f"(first: {missing[:5]}); not deleting local copies."
        )


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
    proactive_overflow: bool = True,
) -> ShardUploadResult:
    """Upload ``local_dir``'s shard files one at a time, bounding local footprint.

    Per shard, in order: upload to ``repo_id`` → (on quota-403) reroute to the
    private overflow repo → verify the shard lists at its destination → delete
    the local shard (only when ``delete_local`` and verification passed).

    Proactive projected-headroom routing (#1034, default ON): before any
    upload, the exact on-disk shard byte sum is probed against the remaining
    public-storage headroom (:func:`hub.check_projected_upload_headroom` —
    live-confirmed before any "insufficient" verdict). On KNOWN-insufficient
    + CONFIRMED-public canonical target, ALL shards route to the overflow
    repo UP-FRONT — one pointer, one JSONL event, zero canonical LFS bytes
    attempted — instead of splitting the store at a mid-store 403 (the #841
    incident shape). Default ON because THIS flow already reroutes
    unconditionally on 403, so consumers must already handle
    ``result.rerouted`` + pointers; ``proactive_overflow=False`` is the
    escape hatch for a future canonical-path-verifying caller (the i528-style
    arming-contract concern).

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
        proactive_overflow: When True (default), probe projected headroom at
            entry and route ALL shards to overflow up-front on a
            live-confirmed insufficient verdict against a confirmed-public
            canonical target. False skips the probe entirely (zero headroom
            I/O) — straight to the legacy per-shard loop.

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

    route_all_to_overflow = False
    if proactive_overflow and shards and repo_id != DEFAULT_OVERFLOW_REPO:
        projected = sum(p.stat().st_size for p in shards)
        ph = check_projected_upload_headroom(projected)
        if ph.verdict == "insufficient" and _repo_is_private(repo_id, repo_type=repo_type) is False:
            # KNOWN-insufficient + CONFIRMED-public canonical target only.
            # repo_type= MUST be threaded (#1034 Must-Fix): _repo_is_private
            # defaults repo_type="model" while THIS flow's canonical target
            # defaults repo_type="dataset" — the incident-shape store targets
            # the public DATA repo, and a bare call would 404 -> None ->
            # fail-open -> the proactive route never fires on the primary
            # incident path. (private target = separate quota; privacy None =
            # fail-open, no reroute on uncertainty — the
            # _resolve_lfs_upload_repo semantics.)
            route_all_to_overflow = True
            logger.warning(
                "[hf-headroom] projected %.1f GB exceeds remaining public headroom "
                "(%.2f/%.1f TB used, %s) — routing ALL %d shards of %s -> %s UP-FRONT "
                "(#1034 proactive; #841 v11 pattern)",
                projected / 1e9,
                ph.used_tb,
                ph.ceiling_tb,
                ph.basis,
                len(shards),
                repo_id,
                DEFAULT_OVERFLOW_REPO,
            )
            _ensure_overflow_repo(api)
            _emit_overflow_routing_event(
                original_repo=repo_id,
                effective_repo=DEFAULT_OVERFLOW_REPO,
                path_in_repo=prefix,
                reason="projected-headroom-proactive",
                projected_gb=projected / 1e9,
            )
            _write_overflow_pointer(
                api,
                canonical_repo=repo_id,
                canonical_repo_type=repo_type,
                path_in_repo=prefix,
                overflow_repo=DEFAULT_OVERFLOW_REPO,
            )

    # Once-per-prefix dedup for the REACTIVE 403 path (#1034): a 1000-shard
    # store 403-ing at shard 50 would otherwise issue ~950 duplicate pointer
    # COMMITS of the same prefix-level file — straight into the 256 commits/hr
    # Hub cap.
    emitted_prefixes: set[str] = set()

    # (shard, dest, effective_repo, effective_repo_type) for the batched
    # post-upload verify + the deferred delete_local pass below.
    pending: list[tuple[Path, str, str, str]] = []

    for shard in shards:
        dest = f"{prefix}/{shard.name}" if prefix else shard.name
        if route_all_to_overflow:
            # Proactive branch: straight to overflow (repo_type "model",
            # matching the reactive reroute); zero canonical attempts.
            retry_transient(
                lambda s=shard, d=dest: api.upload_file(
                    path_or_fileobj=str(s),
                    repo_id=DEFAULT_OVERFLOW_REPO,
                    path_in_repo=d,
                    repo_type="model",
                ),
                what=f"upload_file({DEFAULT_OVERFLOW_REPO}/{dest})",
            )
            effective_repo = DEFAULT_OVERFLOW_REPO
            effective_repo_type = "model"
            result.rerouted.append(dest)
        else:
            try:
                # Transient 429/5xx/timeout retried with backoff (#1335: a
                # transport error is never fatal to the run); the persistent
                # quota-403 is NON-transient inside retry_transient and
                # re-raises immediately into the reroute branch below.
                retry_transient(
                    lambda s=shard, d=dest: api.upload_file(
                        path_or_fileobj=str(s),
                        repo_id=repo_id,
                        path_in_repo=d,
                        repo_type=repo_type,
                    ),
                    what=f"upload_file({repo_id}/{dest})",
                )
                effective_repo = repo_id
                effective_repo_type = repo_type
            except Exception as exc:
                if not _is_storage_quota_403(exc):
                    # Non-quota failure (transient retries exhausted or a hard
                    # 4xx): fail loud, do not reroute or delete.
                    raise
                try:
                    effective_repo = _reroute_to_overflow(
                        api,
                        shard=shard,
                        dest=dest,
                        canonical_repo=repo_id,
                        canonical_repo_type=repo_type,
                        emitted_prefixes=emitted_prefixes,
                    )
                except Exception as reroute_exc:
                    raise RuntimeError(
                        f"both main ({repo_id}) and overflow ({DEFAULT_OVERFLOW_REPO}) repos "
                        f"refused shard {shard.name!r}; not deleting local copy. "
                        f"A discard-with-regen-recipe is the caller's decision, always alerted."
                    ) from reroute_exc
                effective_repo_type = "model"
                result.rerouted.append(dest)

        # Parity with the reactive path: consumers reading the full dest list
        # must see rerouted shards too (rerouted dests appear in BOTH lists).
        result.uploaded.append(dest)
        pending.append((shard, dest, effective_repo, effective_repo_type))

    if verify and pending:
        _batched_verify(api, pending, prefix=prefix)

    if delete_local:
        for shard, _dest, _repo, _type in pending:
            shard.unlink()
            result.deleted.append(shard.name)

    return result
