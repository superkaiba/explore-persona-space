"""Incremental, disk-bounded shard upload with overflow rerouting (v2).

The EPS workflow v2 upload policy has NO ceiling: every artifact tries the
main HF repo first, reroutes to the private overflow repo on quota pressure,
and discards only when BOTH quotas are exhausted (always alerted). Big stores
(activation grids, per-context tensors) can exceed the 128 GB per-issue ext4
quota / ~130 GB MooseFS per-pod quota, so a store larger than the disk cannot
be uploaded in one shot. :func:`upload_dir_sharded` has two modes (#1824):

- **batch (the AUTO default for stores whose on-disk byte sum fits under
  ``EPM_UPLOAD_BATCH_MAX_GB``, 50 decimal GB):** chunked bulk
  ``create_commit`` commits of ``<= HUB_COMMIT_FILECOUNT_WARN`` (2,000) files
  each — a 3,840-file store lands in 2 commits instead of 3,840 (~35 s/file
  commit round-trip on the #1482 store, ~100 commits/h against the shared
  256 commits/h Hub cap).
- **per-file (``batch=False`` / over-threshold stores):** the legacy
  one-``upload_file``-per-shard walk.

Both modes share an up-front skip-if-present resume probe (ONE scoped
listing per destination repo; same-size already-on-Hub shards are skipped —
still verified + deleted, recorded in ``ShardUploadResult.skipped_existing``),
the overflow reroute on quota-403, the batched exact-set verify, and the
deferred delete-only-after-verify pass. Since the #1335 deferred-delete
design BOTH modes hold the full store on disk until that final delete pass —
batching does not raise peak local footprint.

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
    HUB_COMMIT_FILECOUNT_WARN,
    HUB_DIR_FILE_LIMIT,
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
    uploaded: list[str] = field(default_factory=list)  # dest paths committed/rerouted THIS call
    rerouted: list[str] = field(default_factory=list)  # dest paths sent to overflow
    deleted: list[str] = field(default_factory=list)  # local shard names removed
    # dest paths already on the Hub at matching size (resume probe, #1824):
    # skipped — NOT re-committed, NOT in `uploaded` — but still verified and
    # (under delete_local) deleted. Additive field: existing call sites read
    # only uploaded/rerouted/deleted.
    skipped_existing: list[str] = field(default_factory=list)


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
    branch. Transient-retried (#1345: transport is never fatal on the
    upload path — a lone 429 must not convert into "both repos refused").
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
        what=f"upload_file({DEFAULT_OVERFLOW_REPO}:{dest})",
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


def _mk_ops(chunk: list[tuple[Path, str]]) -> list:
    """FRESH ``CommitOperationAdd`` objects for one chunk — built PER CALL.

    ``CommitOperationAdd`` objects are SINGLE-USE on huggingface_hub 0.36.2:
    ``create_commit`` MUTATES its operations (hf_api.py:4196 raises "already
    being committed and cannot be reused"; hf_api.py:4442 filters
    ``_is_uploaded`` additions out of preupload), so ops MUST be constructed
    fresh INSIDE every commit attempt — retry thunks AND the overflow
    re-commit after a canonical quota-403 (a mid-preupload 403 leaves ops
    ``_is_uploaded=True`` with upload modes fetched against the CANONICAL
    repo; reusing them on the overflow MODEL repo would silently skip
    uploading the bytes — #1824 Must-Fix 1). Same shape as the
    BytesIO-inside-thunk rule in :func:`_write_overflow_pointer`.
    """
    from huggingface_hub import CommitOperationAdd

    return [
        CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(shard)) for shard, dest in chunk
    ]


def _remote_sizes_under_prefix(api, repo_id: str, prefix: str, repo_type: str) -> dict[str, int]:
    """``{path: size}`` for files under ``prefix`` on the Hub — the resume
    probe's read: ONE scoped ``list_repo_tree`` per destination repo.

    - The lazy generator is MATERIALIZED inside the ``retry_transient`` thunk
      (Hub list APIs raise at ITERATION time, so a try/except around the bare
      call catches nothing — gotchas.md #779).
    - ``EntryNotFoundError`` -> ``{}`` (absent prefix = nothing uploaded yet;
      the 404-is-expected rule). A ``RepositoryNotFoundError`` / auth error
      PROPAGATES loud — a typo'd repo id must never read as "empty".
    - Sizes come from ``RepoFile.size`` (hf_api.py:657-658); ``RepoFolder``
      entries carry no ``size`` and are filtered by attribute.
    - Empty ``prefix`` raises ValueError: a recursive walk on
      ``path_in_repo=""`` is the #833/#920 full-repo-enumeration wedge on the
      ~1M-file data repo. Callers guard root-level dests and SKIP the probe
      entirely (#1824 Must-Fix 2) — no per-dest ``file_exists`` fallback loop
      (the #1335 anti-pattern).
    """
    if not prefix:
        raise ValueError("_remote_sizes_under_prefix: empty prefix (would full-list the repo)")
    from huggingface_hub.utils import EntryNotFoundError

    def _fetch() -> list:
        return list(
            api.list_repo_tree(
                repo_id=repo_id, path_in_repo=prefix, repo_type=repo_type, recursive=True
            )
        )

    try:
        entries = retry_transient(_fetch, what=f"list_repo_tree({repo_id}:{prefix})")
    except EntryNotFoundError:
        return {}
    return {e.path: e.size for e in entries if getattr(e, "size", None) is not None}


def _upload_chunks_bulk(
    api,
    items: list[tuple[Path, str]],
    *,
    result: ShardUploadResult,
    canonical_repo: str,
    canonical_repo_type: str,
    prefix: str,
    chunk_files: int,
    route_all_to_overflow: bool,
    emitted_prefixes: set[str],
) -> list[tuple[Path, str, str, str]]:
    """Commit ``items`` in chunks of ``<= chunk_files`` via bulk
    ``create_commit`` (batch mode, #1824). Returns the per-shard
    ``(shard, dest, effective_repo, effective_repo_type)`` tuples for the
    batched verify + deferred delete pass; mutates ``result.uploaded`` /
    ``result.rerouted`` with the same parity semantics as the per-file walk
    (rerouted dests appear in BOTH lists).

    Reactive quota-403 on a CANONICAL chunk reroutes THIS chunk and every
    REMAINING chunk to the private overflow repo (``repo_type="model"``) —
    ONE pointer + ONE JSONL event per prefix via ``emitted_prefixes`` — each
    commit built from FRESH ops (:func:`_mk_ops`). A non-quota error
    re-raises after the transient-retry budget (fail-loud, no delete); a
    chunk BOTH repos refuse raises RuntimeError naming the chunk.
    """
    pending: list[tuple[Path, str, str, str]] = []
    if not items:
        return pending
    if chunk_files < 1:
        raise ValueError(f"batch_chunk_files must be >= 1, got {chunk_files}")
    chunks = [items[i : i + chunk_files] for i in range(0, len(items), chunk_files)]
    n_chunks = len(chunks)
    on_overflow = route_all_to_overflow
    # Cumulative flat-store visibility (#1824): the hub.py 10k/dir guard is
    # per-commit-STAGING only; a dir built over the cap INCREMENTALLY across
    # chunks is a documented STAGED-ONLY residual — WARN once per dir, never
    # raise (the server 400 stays the late backstop).
    committed_per_dir: dict[tuple[str, str], int] = {}
    warned_dirs: set[tuple[str, str]] = set()
    for i, chunk in enumerate(chunks, start=1):
        rerouted_this_chunk = on_overflow
        target_repo, target_type = (
            (DEFAULT_OVERFLOW_REPO, "model")
            if on_overflow
            else (canonical_repo, canonical_repo_type)
        )
        message = f"upload_dir_sharded batch {i}/{n_chunks} ({len(chunk)} files)"
        try:
            retry_transient(
                # Ops built FRESH inside the thunk (_mk_ops): a retried
                # attempt must never reuse mutated single-use ops.
                lambda _c=chunk, _r=target_repo, _t=target_type, _m=message: api.create_commit(
                    repo_id=_r, repo_type=_t, operations=_mk_ops(_c), commit_message=_m
                ),
                what=f"create_commit({target_repo}:{prefix or '.'} chunk {i}/{n_chunks})",
            )
        except Exception as exc:
            if on_overflow or not _is_storage_quota_403(exc):
                # Non-quota failure (transient retries exhausted or a hard
                # 4xx), or a failure on the overflow repo itself: fail loud,
                # do not reroute or delete.
                raise
            logger.warning(
                "quota-403 on %s — rerouting chunk %d/%d (+%d remaining) -> %s (overflow)",
                canonical_repo,
                i,
                n_chunks,
                n_chunks - i,
                DEFAULT_OVERFLOW_REPO,
            )
            _ensure_overflow_repo(api)
            ev_prefix = os.path.dirname(chunk[0][1])
            if ev_prefix not in emitted_prefixes:
                _emit_overflow_routing_event(
                    original_repo=canonical_repo,
                    effective_repo=DEFAULT_OVERFLOW_REPO,
                    path_in_repo=chunk[0][1],
                )
                _write_overflow_pointer(
                    api,
                    canonical_repo=canonical_repo,
                    canonical_repo_type=canonical_repo_type,
                    path_in_repo=ev_prefix,
                    overflow_repo=DEFAULT_OVERFLOW_REPO,
                )
                emitted_prefixes.add(ev_prefix)
            try:
                retry_transient(
                    # FRESH ops again (#1824 Must-Fix 1): the failed canonical
                    # attempt consumed the previous ops — see _mk_ops.
                    lambda _c=chunk, _m=message: api.create_commit(
                        repo_id=DEFAULT_OVERFLOW_REPO,
                        repo_type="model",
                        operations=_mk_ops(_c),
                        commit_message=_m,
                    ),
                    what=f"create_commit({DEFAULT_OVERFLOW_REPO}:{prefix or '.'} "
                    f"chunk {i}/{n_chunks})",
                )
            except Exception as reroute_exc:
                raise RuntimeError(
                    f"both main ({canonical_repo}) and overflow ({DEFAULT_OVERFLOW_REPO}) "
                    f"repos refused chunk {i}/{n_chunks} ({len(chunk)} files); not deleting "
                    f"local copies. A discard-with-regen-recipe is the caller's decision, "
                    f"always alerted."
                ) from reroute_exc
            on_overflow = True  # remaining chunks go straight to overflow
            rerouted_this_chunk = True
            target_repo, target_type = DEFAULT_OVERFLOW_REPO, "model"

        for shard, dest in chunk:
            if rerouted_this_chunk:
                result.rerouted.append(dest)
            result.uploaded.append(dest)
            pending.append((shard, dest, target_repo, target_type))
            d_key = (target_repo, os.path.dirname(dest))
            committed_per_dir[d_key] = committed_per_dir.get(d_key, 0) + 1
            if committed_per_dir[d_key] > HUB_DIR_FILE_LIMIT and d_key not in warned_dirs:
                warned_dirs.add(d_key)
                logger.warning(
                    "flat-store residual: this call committed %s files into %s:%s across "
                    "chunks (> HUB_DIR_FILE_LIMIT=%s). The server 10k cap is per-commit-"
                    "STAGING, so incremental accumulation past it is a documented hub.py "
                    "STAGED-ONLY residual — a future single-commit re-upload of this dir "
                    "would be refused by assert_hub_dir_filecounts. Advisory only.",
                    f"{committed_per_dir[d_key]:,}",
                    target_repo,
                    d_key[1],
                    f"{HUB_DIR_FILE_LIMIT:,}",
                )
    return pending


def _resume_skip_map(
    api,
    shards: list[Path],
    *,
    prefix: str,
    repo_id: str,
    repo_type: str,
    route_all_to_overflow: bool,
) -> dict[str, tuple[str, str]]:
    """``dest -> (repo_id, repo_type)`` for shards ALREADY on the Hub at
    MATCHING size — the skip-if-present resume probe (#1824).

    ONE scoped listing of the primary destination repo (the canonical repo,
    or the overflow repo when the caller already routed everything there);
    when a prior run left an ``OVERFLOW_POINTER.json`` under the prefix, the
    overflow repo is probed too and the views merge with CANONICAL
    precedence. A size MISMATCH is a re-upload, never a skip. Root-level
    dests (``prefix == ""``) SKIP the probe entirely with one INFO line
    (#1824 Must-Fix 2): a recursive tree walk on ``path_in_repo=""`` is the
    #833/#920 full-repo-enumeration wedge on the ~1M-file data repo, and a
    per-dest ``file_exists`` fallback loop is the #1335 anti-pattern — so
    root-level callers keep the always-upload behavior.
    """
    if not prefix:
        logger.info(
            "resume probe skipped: path_in_repo is empty (root-level dests) — a "
            "recursive tree walk on path_in_repo='' is the #833 full-repo-enumeration "
            "wedge on the ~1M-file data repo; root-level callers keep the "
            "always-upload behavior"
        )
        return {}
    primary_repo, primary_type = (
        (DEFAULT_OVERFLOW_REPO, "model") if route_all_to_overflow else (repo_id, repo_type)
    )
    remote = _remote_sizes_under_prefix(api, primary_repo, prefix, primary_type)
    overflow_remote: dict[str, int] = {}
    if primary_repo != DEFAULT_OVERFLOW_REPO and f"{prefix}/OVERFLOW_POINTER.json" in remote:
        # A prior run rerouted: shards may live on the overflow repo.
        overflow_remote = _remote_sizes_under_prefix(api, DEFAULT_OVERFLOW_REPO, prefix, "model")
    skipped: dict[str, tuple[str, str]] = {}
    for shard in shards:
        dest = f"{prefix}/{shard.name}"
        size = shard.stat().st_size
        # Skip iff present at MATCHING size (mismatch -> re-upload).
        # Skip-precedence: CANONICAL wins when a dest resolves on both.
        if remote.get(dest) == size:
            skipped[dest] = (primary_repo, primary_type)
        elif overflow_remote.get(dest) == size:
            skipped[dest] = (DEFAULT_OVERFLOW_REPO, "model")
    return skipped


def _upload_per_file(
    api,
    items: list[tuple[Path, str]],
    *,
    result: ShardUploadResult,
    canonical_repo: str,
    canonical_repo_type: str,
    route_all_to_overflow: bool,
    emitted_prefixes: set[str],
) -> list[tuple[Path, str, str, str]]:
    """The legacy PER-FILE walk (``batch=False`` / over-threshold stores):
    one ``upload_file`` per shard, quota-403 rerouted per shard. Returns the
    per-shard pending tuples for the batched verify + deferred delete pass;
    mutates ``result.uploaded`` / ``result.rerouted`` (rerouted dests appear
    in BOTH lists). Body unchanged from the pre-#1824 inline loop.
    """
    pending: list[tuple[Path, str, str, str]] = []
    for shard, dest in items:
        if route_all_to_overflow:
            # Proactive branch: straight to overflow (repo_type "model",
            # matching the reactive reroute); zero canonical attempts.
            # Transient-retried (#1345): a lone 429/5xx must never kill the
            # run — retry_transient re-raises quota-403 / non-transient
            # immediately and fail-louds only on genuine exhaustion.
            retry_transient(
                lambda _s=shard, _d=dest: api.upload_file(
                    path_or_fileobj=str(_s),
                    repo_id=DEFAULT_OVERFLOW_REPO,
                    path_in_repo=_d,
                    repo_type="model",
                ),
                what=f"upload_file({DEFAULT_OVERFLOW_REPO}:{dest})",
            )
            effective_repo = DEFAULT_OVERFLOW_REPO
            effective_repo_type = "model"
            result.rerouted.append(dest)
        else:
            try:
                retry_transient(
                    lambda _s=shard, _d=dest: api.upload_file(
                        path_or_fileobj=str(_s),
                        repo_id=canonical_repo,
                        path_in_repo=_d,
                        repo_type=canonical_repo_type,
                    ),
                    what=f"upload_file({canonical_repo}:{dest})",
                )
                effective_repo = canonical_repo
                effective_repo_type = canonical_repo_type
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
                        canonical_repo=canonical_repo,
                        canonical_repo_type=canonical_repo_type,
                        emitted_prefixes=emitted_prefixes,
                    )
                except Exception as reroute_exc:
                    raise RuntimeError(
                        f"both main ({canonical_repo}) and overflow "
                        f"({DEFAULT_OVERFLOW_REPO}) repos refused shard {shard.name!r}; "
                        f"not deleting local copy. A discard-with-regen-recipe is the "
                        f"caller's decision, always alerted."
                    ) from reroute_exc
                effective_repo_type = "model"
                result.rerouted.append(dest)

        # Parity with the reactive path: consumers reading the full dest list
        # must see rerouted shards too (rerouted dests appear in BOTH lists).
        result.uploaded.append(dest)
        pending.append((shard, dest, effective_repo, effective_repo_type))
    return pending


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
    batch: bool | None = None,
    batch_chunk_files: int = HUB_COMMIT_FILECOUNT_WARN,
    resume_skip: bool = True,
) -> ShardUploadResult:
    """Upload ``local_dir``'s shard files — chunked bulk commits by default
    (#1824), the legacy per-file walk as the explicit / large-store fallback.

    BATCH mode (the AUTO default when the store's on-disk byte sum fits under
    ``EPM_UPLOAD_BATCH_MAX_GB``, 50 decimal GB): partition the non-skipped
    shards into chunks of ``<= batch_chunk_files`` and commit each chunk via
    one bulk ``create_commit`` (exact file selection — no glob/fnmatch
    escaping fragility; ``create_commit`` runs upload-mode fetch + LFS
    preupload internally on hub 0.36.2). PER-FILE mode (``batch=False`` /
    over-threshold stores): per shard, upload to ``repo_id`` → (on quota-403)
    reroute to the private overflow repo. Both modes then verify every dest
    lists at its destination (ONE scoped listing per destination repo) and
    delete local shards only when ``delete_local`` and verification passed.
    Batching does NOT raise peak local footprint: since the #1335
    deferred-delete design both modes hold the full store on disk until the
    post-verify delete pass.

    Skip-if-present resume probe (BOTH modes, ``resume_skip=True`` default):
    one scoped listing of the primary destination repo (plus the overflow
    repo when a prior run left an ``OVERFLOW_POINTER.json`` under the
    prefix); a shard whose dest is already present at MATCHING size is
    skipped — recorded in ``ShardUploadResult.skipped_existing``, still
    verified + deleted, NOT re-committed (an interrupted walk resumes without
    re-uploading). Root-level dests (``path_in_repo=""``) SKIP the probe
    entirely (#1824 Must-Fix 2 — a recursive tree walk on ``path_in_repo=""``
    is the #833 full-repo-enumeration wedge) and keep the always-upload
    behavior.

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
            I/O) — straight to the per-shard/per-chunk upload loop.
        batch: ``None`` (default) = AUTO — batch iff the store's on-disk byte
            sum fits under ``EPM_UPLOAD_BATCH_MAX_GB`` (env, default 50
            decimal GB); ``True`` forces batch; ``False`` forces the legacy
            per-file walk.
        batch_chunk_files: Files per bulk ``create_commit`` (default
            ``HUB_COMMIT_FILECOUNT_WARN`` = 2,000 — hub.py's own throughput
            tier, which also keeps every commit under the 10k/dir server
            cap).
        resume_skip: When True (default), run the skip-if-present resume
            probe (see above). ``False`` restores the always-upload behavior
            with zero probe I/O.

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

    # Hoisted (#1824): the exact on-disk byte sum feeds BOTH the #1034
    # proactive headroom probe (conservative: the FULL store's size, not the
    # post-skip remainder) and the batch/per-file mode decision below.
    projected = sum(p.stat().st_size for p in shards)

    route_all_to_overflow = False
    if proactive_overflow and shards and repo_id != DEFAULT_OVERFLOW_REPO:
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

    # ------------------------------------------------------------- mode (#1824)
    if batch is None:
        # AUTO: batch iff the store's byte sum fits under the threshold
        # (decimal GB). The threshold only keeps truly giant stores on the
        # conservative per-file path — batching does not change peak local
        # footprint (see docstring). Unparseable env values fail LOUD.
        use_batch = projected <= float(os.environ.get("EPM_UPLOAD_BATCH_MAX_GB", "50")) * 1e9
    else:
        use_batch = batch

    # ------------------------------------- skip-if-present resume probe (#1824)
    skipped: dict[str, tuple[str, str]] = (
        _resume_skip_map(
            api,
            shards,
            prefix=prefix,
            repo_id=repo_id,
            repo_type=repo_type,
            route_all_to_overflow=route_all_to_overflow,
        )
        if resume_skip and shards
        else {}
    )

    # Once-per-prefix dedup for the REACTIVE 403 path (#1034): a 1000-shard
    # store 403-ing at shard 50 would otherwise issue ~950 duplicate pointer
    # COMMITS of the same prefix-level file — straight into the 256 commits/hr
    # Hub cap.
    emitted_prefixes: set[str] = set()

    # (shard, dest, effective_repo, effective_repo_type) for the batched
    # post-upload verify + the deferred delete_local pass below.
    pending: list[tuple[Path, str, str, str]] = []

    # Partition: skipped shards enter `pending` directly (verify + delete
    # still cover them; NOT `uploaded` — that list means committed/rerouted
    # by THIS call); the rest go to the mode-selected upload path.
    to_upload: list[tuple[Path, str]] = []
    for shard in shards:
        dest = f"{prefix}/{shard.name}" if prefix else shard.name
        if dest in skipped:
            eff_repo, eff_type = skipped[dest]
            result.skipped_existing.append(dest)
            pending.append((shard, dest, eff_repo, eff_type))
        else:
            to_upload.append((shard, dest))

    if use_batch:
        pending.extend(
            _upload_chunks_bulk(
                api,
                to_upload,
                result=result,
                canonical_repo=repo_id,
                canonical_repo_type=repo_type,
                prefix=prefix,
                chunk_files=batch_chunk_files,
                route_all_to_overflow=route_all_to_overflow,
                emitted_prefixes=emitted_prefixes,
            )
        )
    else:
        pending.extend(
            _upload_per_file(
                api,
                to_upload,
                result=result,
                canonical_repo=repo_id,
                canonical_repo_type=repo_type,
                route_all_to_overflow=route_all_to_overflow,
                emitted_prefixes=emitted_prefixes,
            )
        )

    if verify and pending:
        _batched_verify(api, pending, prefix=prefix)

    if delete_local:
        for shard, _dest, _repo, _type in pending:
            shard.unlink()
            result.deleted.append(shard.name)

    return result
