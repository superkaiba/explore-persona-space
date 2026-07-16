"""HuggingFace Hub upload, WandB artifact upload, and local disk cleanup.

Default repos (public, unlimited storage):
  Models:   superkaiba1/explore-persona-space
  Datasets: superkaiba1/explore-persona-space-data
"""

import glob
import json
import logging
import math
import os
import posixpath
import random
import re
import shutil
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default public HF Hub repos
DEFAULT_MODEL_REPO = "superkaiba1/explore-persona-space"
DEFAULT_DATASET_REPO = "superkaiba1/explore-persona-space-data"

# Training-state files that must NEVER reach the Hub. Optimizer/scheduler/RNG
# state is resume-only scratch: it is useless for inference or reproduction
# (re-training resumes from local checkpoints, never from the Hub), yet a
# single Adam ``optimizer.pt`` is ~2x the adapter size and HF Trainer writes
# one per ``checkpoint-*`` dir. Wholesale ``upload_folder`` calls shipped
# ~810GB of this residue to the public repo (2026-06-10 storage inventory).
# Patterns are fnmatch-style against the path RELATIVE to the uploaded folder
# (``*`` matches across ``/``, so ``*optimizer.pt`` also matches
# ``checkpoint-500/optimizer.pt``).
TRAINING_STATE_IGNORE_PATTERNS: list[str] = [
    "*optimizer.pt",
    "*scheduler.pt",
    "*rng_state*.pth",
]


def merged_upload_enabled(cfg_value: bool | None = None) -> bool:
    """Whether merged/full-checkpoint HF uploads are explicitly opted in.

    Merged checkpoints (~15GB) are derived data — regenerable from the public
    base model plus the ~300MB LoRA adapter — so the project default is to
    upload ONLY the adapter (Upload Policy / #404 / #458). Opt in to merged
    uploads with EITHER the env var ``EPM_UPLOAD_MERGED=1`` OR a truthy
    ``upload_merged`` config flag (passed in as ``cfg_value``).

    Args:
        cfg_value: The caller's ``upload_merged`` config value (e.g.
            ``cfg.get("upload_merged", False)``), or None when the caller has
            no config surface.

    Returns:
        True iff merged-checkpoint upload is explicitly enabled.
    """
    return os.environ.get("EPM_UPLOAD_MERGED") == "1" or bool(cfg_value)


# ── Account-level HF public-storage headroom (proactive quota guard, #564) ────

# Private overflow repo; private-repo LFS quota is SEPARATE from the public
# pool (validated incident #541 — see .claude/rules/upload-policy.md
# § HF storage-quota 403). issue_604 carries its own copy of this string
# (frozen completed-experiment code, deliberately untouched).
DEFAULT_OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"

DEFAULT_HF_NAMESPACE = DEFAULT_MODEL_REPO.split("/")[0]  # "superkaiba1"

# The hard wall was observed at ~11.3 TB used = 100% of the public quota
# (incident #541, same probe family + units as this check). 10.0 leaves
# ~1.3 TB of warning runway before the wall.
DEFAULT_STORAGE_SOFT_CEILING_TB = 10.0
DEFAULT_STORAGE_CACHE_TTL_S = 3600.0  # "~1h" (task #564 AC1)
_BYTES_PER_TB = 1000.0**4  # HF reports decimal bytes; matches the incident's 11.3 TB read


@dataclass(frozen=True)
class HfStorageHeadroom:
    """Result of an account-level HF public-storage probe.

    ``used_tb is None`` means UNKNOWN (API error / poisoned probe / check
    disabled) — callers must treat unknown as "cannot verify", never as 0.
    ``over_ceiling`` is always False when ``used_tb`` is None.
    """

    used_tb: float | None
    ceiling_tb: float
    over_ceiling: bool
    basis: str  # "live-api" | "cache (age Ns)" | "disabled" | "suspect (...)" | "unknown (...)"
    n_repos: int = 0


def _env_float(name: str, default: float) -> float:
    """Resolve a float env knob; non-parseable values raise ValueError.

    A wrong ceiling/TTL is a user config error — silently defaulting would
    hide it (fail-fast house rule). Empty/unset falls back to ``default``.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name}={raw!r} is not a parseable number — fix or unset it") from e


def _storage_cache_path() -> Path:
    """On-disk cache location: env override, else ~/.cache (a few hundred bytes)."""
    env = os.environ.get("EPM_HF_STORAGE_CACHE_PATH")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "explore_persona_space" / "hf_storage_usage.json"


def _read_storage_cache(
    path: Path, *, namespace: str, ttl_s: float
) -> tuple[int, int, float] | None:
    """Read ``(used_bytes, n_repos, age_s)`` from the on-disk cache, or None.

    Fail-soft: corrupt / missing / stale / wrong-namespace entries are ignored
    (caller falls through to the live probe). Rejects any ``used_bytes`` that
    is not a positive int — defense in depth so a suspect/zero entry can never
    produce a clean under-ceiling cache hit.
    """
    try:
        raw = json.loads(path.read_text())
        if raw.get("namespace") != namespace:
            return None
        used_bytes = raw["used_bytes"]
        if type(used_bytes) is not int or used_bytes <= 0:
            return None
        age_s = time.time() - float(raw["ts"])
        if age_s < 0 or age_s >= ttl_s:
            return None
        return used_bytes, int(raw.get("n_repos", 0)), age_s
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("HF storage cache read failed (%s) — re-probing live", e)
        return None


def _write_storage_cache(path: Path, *, namespace: str, used_bytes: int, n_repos: int) -> None:
    """Atomically persist a SUCCESSFUL, COMPLETE usage sum. Fail-soft on I/O errors.

    Only complete sums are ever cached — suspect/unknown probes are never
    written (a cached suspect 0 would bypass the guard for a whole TTL across
    every process). The tmp name is PID/uuid-suffixed so concurrent
    cold-starting sweep cells never collide on the same tmp file.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "ts": time.time(),
                    "used_bytes": int(used_bytes),
                    "n_repos": int(n_repos),
                    "namespace": namespace,
                }
            )
        )
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("HF storage cache write failed (%s) — continuing without cache", e)


def check_hf_storage_headroom(
    *,
    namespace: str = DEFAULT_HF_NAMESPACE,
    ceiling_tb: float | None = None,
    cache_ttl_s: float | None = None,
    cache_path: Path | None = None,
    force_refresh: bool = False,
) -> HfStorageHeadroom:
    """Account-level HF public-storage usage vs a configurable soft ceiling.

    Two-stage probe (the server 400s ``expand=["usedStorage"]`` on the LIST
    endpoints — live-verified 2026-06-12 — so the list stage only enumerates):

    1. ``list_models``/``list_datasets(author=..., expand=["private"])`` to
       enumerate repos, filtering private ones (public-storage quota counts
       public repos only).
    2. Per-repo ``model_info``/``dataset_info(rid, expand=["usedStorage"])``
       fanned over a bounded thread pool (~406 public repos ≈ 25 s on a cache
       miss; the 1h on-disk cache amortizes).

    Scope note: the account has 0 Spaces today; models + datasets cover the
    public-storage sum. ANY per-repo ``usedStorage`` that is absent/None
    poisons the whole probe to unknown (None ≠ 0 — a partial sum understates
    usage; #541 had 10.2 of 11.3 TB in ONE repo). Suspect/unknown probes are
    NEVER cached.

    Env knobs: ``EPM_HF_STORAGE_CHECK=0`` (kill switch),
    ``EPM_HF_STORAGE_SOFT_CEILING_TB`` (default 10.0),
    ``EPM_HF_STORAGE_CACHE_TTL_S`` (default 3600),
    ``EPM_HF_STORAGE_CACHE_PATH`` (cache file override).

    Never raises on API/network failure (returns ``used_tb=None``); raises
    ``ValueError`` only on a non-parseable ceiling/TTL env value (user config
    error — fail-fast where the value is load-bearing).
    """
    # Kill switch FIRST — the escape hatch must always work, so it precedes
    # even env parsing (the returned ceiling is decorative on this branch).
    if os.environ.get("EPM_HF_STORAGE_CHECK") == "0":
        return HfStorageHeadroom(
            used_tb=None,
            ceiling_tb=ceiling_tb if ceiling_tb is not None else DEFAULT_STORAGE_SOFT_CEILING_TB,
            over_ceiling=False,
            basis="disabled",
        )

    ceiling = (
        ceiling_tb
        if ceiling_tb is not None
        else _env_float("EPM_HF_STORAGE_SOFT_CEILING_TB", DEFAULT_STORAGE_SOFT_CEILING_TB)
    )
    ttl = (
        cache_ttl_s
        if cache_ttl_s is not None
        else _env_float("EPM_HF_STORAGE_CACHE_TTL_S", DEFAULT_STORAGE_CACHE_TTL_S)
    )
    path = cache_path if cache_path is not None else _storage_cache_path()

    if not force_refresh:
        cached = _read_storage_cache(path, namespace=namespace, ttl_s=ttl)
        if cached is not None:
            used_bytes, n_repos, age_s = cached
            used_tb = used_bytes / _BYTES_PER_TB
            return HfStorageHeadroom(
                used_tb=used_tb,
                ceiling_tb=ceiling,
                over_ceiling=used_tb > ceiling,
                basis=f"cache (age {age_s:.0f}s)",
                n_repos=n_repos,
            )

    try:
        from concurrent.futures import ThreadPoolExecutor

        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        repos: list[tuple[str, str]] = []
        for lister, rtype in ((api.list_models, "model"), (api.list_datasets, "dataset")):
            for info in lister(author=namespace, expand=["private"]):
                if getattr(info, "private", False):
                    continue  # public-storage quota counts public repos only
                repos.append((info.id, rtype))

        def _used(rid_rtype: tuple[str, str]) -> int | None:
            rid, rtype = rid_rtype
            info_fn = api.model_info if rtype == "model" else api.dataset_info
            # usedStorage lands via __dict__.update(**kwargs), not a declared
            # field; absent/None means "not populated", NOT zero.
            v = getattr(info_fn(rid, expand=["usedStorage"]), "usedStorage", None)
            return None if v is None else int(v)

        with ThreadPoolExecutor(max_workers=8) as pool:
            per_repo = list(pool.map(_used, repos))
    except Exception as e:
        logger.warning("HF storage probe failed (%s) — headroom unknown", e)
        return HfStorageHeadroom(
            used_tb=None, ceiling_tb=ceiling, over_ceiling=False, basis=f"unknown ({e})"
        )

    n = len(repos)
    n_missing = sum(1 for v in per_repo if v is None)
    if n and n_missing:
        # PARTIAL-None GUARD: counting a present-but-unpopulated usedStorage
        # as 0 silently understates usage — ANY missing value poisons the
        # probe to unknown rather than producing a partial sum.
        return HfStorageHeadroom(
            used_tb=None,
            ceiling_tb=ceiling,
            over_ceiling=False,
            basis=f"suspect ({n_missing}/{n} missing usedStorage)",
            n_repos=n,
        )
    used_bytes = sum(per_repo)
    if n and used_bytes == 0:
        # All-zero suspect guard (independent backstop): a server that stops
        # populating usedStorage must not read as perpetual headroom.
        return HfStorageHeadroom(
            used_tb=None,
            ceiling_tb=ceiling,
            over_ceiling=False,
            basis="suspect (all usedStorage empty)",
            n_repos=n,
        )

    _write_storage_cache(path, namespace=namespace, used_bytes=used_bytes, n_repos=n)
    used_tb = used_bytes / _BYTES_PER_TB
    return HfStorageHeadroom(
        used_tb=used_tb,
        ceiling_tb=ceiling,
        over_ceiling=used_tb > ceiling,
        basis="live-api",
        n_repos=n,
    )


@dataclass(frozen=True)
class ProjectedUploadHeadroom:
    """Verdict of a size-aware account-headroom probe for one planned LFS upload."""

    verdict: str  # "below-threshold" | "disabled" | "unknown" | "fits" | "insufficient"
    projected_tb: float
    used_tb: float | None
    ceiling_tb: float | None  # None on the zero-I/O below-threshold arm; may also be
    # None on the disabled/unknown arms (passes h.ceiling_tb through)
    basis: str


def check_projected_upload_headroom(
    projected_bytes: int,
    *,
    probe_floor_gb: float | None = None,
    confirm_live: bool = True,
) -> ProjectedUploadHeadroom:
    """Does a planned LFS upload of ``projected_bytes`` fit under the public-storage soft ceiling?

    Size-aware wrapper over :func:`check_hf_storage_headroom` (#564 — reused, never
    re-implemented). Decimal GB/TB throughout (1 GB = 1e9 bytes), matching
    ``_BYTES_PER_TB = 1000**4``. Never raises on API failure; ``ValueError`` only on a
    non-parseable env knob (same fail-fast contract as the #564 knobs).

    * projected below the probe floor (``probe_floor_gb``, default env
      ``EPM_HF_LARGE_UPLOAD_PROBE_GB`` = 100.0) -> ``"below-threshold"``, ZERO
      headroom I/O (tiny uploads never pay the ~25 s cache-miss probe).
    * kill switch / disabled -> ``"disabled"`` (escape hatch always wins).
    * probe unknown/suspect -> ``"unknown"`` (fail-open; callers must not block
      or reroute — the reactive 403 backstop stays authoritative).
    * used + projected <= ceiling -> ``"fits"``.
    * used + projected > ceiling -> when ``confirm_live``, re-probe with
      ``force_refresh=True`` (the trainer.py minute-1 pattern: never act on a
      stale cached over-read) and re-evaluate; only a LIVE-confirmed overflow
      returns ``"insufficient"`` (a live-unknown degrades to ``"unknown"``).

    Concurrency note: ``"fits"`` is advisory, not a reservation — two concurrent
    large uploads can both read fits against the same headroom (TOCTOU); the
    soft-ceiling runway + the reactive 403 backstop absorb races, never worse
    than status quo.
    """
    assert projected_bytes >= 0, projected_bytes
    floor_gb = (
        probe_floor_gb
        if probe_floor_gb is not None
        else _env_float("EPM_HF_LARGE_UPLOAD_PROBE_GB", 100.0)
    )
    projected_tb = projected_bytes / _BYTES_PER_TB
    if projected_bytes < floor_gb * 1e9:
        return ProjectedUploadHeadroom("below-threshold", projected_tb, None, None, "not-probed")
    h = check_hf_storage_headroom()
    if h.basis == "disabled":
        return ProjectedUploadHeadroom("disabled", projected_tb, None, h.ceiling_tb, h.basis)
    if h.used_tb is None:
        return ProjectedUploadHeadroom("unknown", projected_tb, None, h.ceiling_tb, h.basis)
    if h.used_tb + projected_tb <= h.ceiling_tb:
        return ProjectedUploadHeadroom("fits", projected_tb, h.used_tb, h.ceiling_tb, h.basis)
    if confirm_live:
        h = check_hf_storage_headroom(force_refresh=True)
        if h.used_tb is None:
            return ProjectedUploadHeadroom("unknown", projected_tb, None, h.ceiling_tb, h.basis)
        if h.used_tb + projected_tb <= h.ceiling_tb:
            return ProjectedUploadHeadroom("fits", projected_tb, h.used_tb, h.ceiling_tb, h.basis)
    return ProjectedUploadHeadroom("insufficient", projected_tb, h.used_tb, h.ceiling_tb, h.basis)


def _repo_is_private(repo_id: str, repo_type: str = "model") -> bool | None:
    """TRI-STATE privacy probe: True | False | None (undeterminable).

    ``None`` (any ``repo_info`` failure) must route callers to their
    fail-open arm — coercing a transient blip to "public" would false-abort
    a healthy private-target sweep (persist gate) or wrongly reroute a
    private-target upload (overflow routing).
    """
    from huggingface_hub import HfApi

    try:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        info = api.repo_info(repo_id, repo_type=repo_type)
        priv = getattr(info, "private", None)
        return None if priv is None else bool(priv)
    except Exception as e:
        logger.warning("repo_info(%s) failed (%s) — privacy undeterminable", repo_id, e)
        return None


# One loud warning per process when routing is armed but the headroom signal
# is disabled/unknown — a stale kill switch must not silently disarm the
# protection the user believes is on.
_OVERFLOW_BLIND_WARNED = False


def _resolve_lfs_upload_repo(repo_id: str, projected_bytes: int | None = None) -> tuple[str, bool]:
    """``(effective_repo_id, rerouted)`` for an LFS-bearing model upload.

    SHORT-CIRCUITS on the env gate first: ``EPM_HF_OVERFLOW_ROUTING != "1"``
    returns ``(repo_id, False)`` with ZERO headroom I/O — routing is
    default-off and must add no latency to normal uploads. When armed, the
    upload reroutes to :data:`DEFAULT_OVERFLOW_REPO` iff headroom is
    KNOWN-insufficient (``used + projected > ceiling``; with
    ``projected_bytes=None`` this reproduces the legacy binary
    KNOWN-over-ceiling check exactly) AND ``repo_id`` is not already the
    overflow repo AND the target is CONFIRMED public (a private target has
    its own quota headroom; privacy ``None``/undeterminable does not reroute
    — routing only acts on confirmed signal). Unknown/disabled headroom never
    reroutes and logs one loud armed-but-blind warning per process. The
    ARMING CONTRACT is unchanged: default-off, zero headroom I/O unarmed
    (#1034 added only the size-aware predicate on the already-armed path).
    """
    global _OVERFLOW_BLIND_WARNED
    if os.environ.get("EPM_HF_OVERFLOW_ROUTING") != "1":
        return repo_id, False
    if repo_id == DEFAULT_OVERFLOW_REPO:
        return repo_id, False
    h = check_hf_storage_headroom()
    if h.used_tb is None:
        if not _OVERFLOW_BLIND_WARNED:
            logger.warning(
                "EPM_HF_OVERFLOW_ROUTING=1 is armed but the storage signal is %s — "
                "routing is BLIND; uploads will NOT reroute. Re-enable "
                "EPM_HF_STORAGE_CHECK / fix the probe if you expected protection.",
                h.basis,
            )
            _OVERFLOW_BLIND_WARNED = True
        return repo_id, False
    projected_tb = (projected_bytes or 0) / _BYTES_PER_TB
    if h.used_tb + projected_tb <= h.ceiling_tb:
        return repo_id, False
    if _repo_is_private(repo_id) is not False:
        # Private target: separate quota, rerouting would be wrong-place.
        # Undeterminable: don't reroute on uncertainty (mirror of the gate's
        # fail-open arm).
        return repo_id, False
    return DEFAULT_OVERFLOW_REPO, True


def _overflow_event_path() -> Path:
    """Event-sink resolution: env override → /workspace/logs (pod/GCP) → ~/.cache."""
    env = os.environ.get("EPM_HF_OVERFLOW_EVENT_PATH")
    if env:
        return Path(env)
    workspace_logs = Path("/workspace/logs")
    if workspace_logs.is_dir():
        return workspace_logs / "hf-overflow-routing.jsonl"
    return Path.home() / ".cache" / "explore_persona_space" / "hf-overflow-routing.jsonl"


def _emit_overflow_routing_event(
    *,
    original_repo: str,
    effective_repo: str,
    path_in_repo: str,
    reason: str = "quota-403-reactive",
    projected_gb: float | None = None,
) -> None:
    """Append a plan-deviation JSON line to the local event sink. Fail-soft.

    Pod-side library code never shells ``task.py`` — the orchestrator /
    upload-verifier observing this sentinel (or the paired structured WARN in
    the run log) posts the actual ``epm:`` plan-deviation marker.

    ``reason`` / ``projected_gb`` (#1034) are append-only JSONL fields —
    backward-compatible: existing callers omit them (default reason
    ``"quota-403-reactive"``), and JSONL consumers are prose-level observers.
    """
    try:
        h = check_hf_storage_headroom()  # cache hit — routing just confirmed over-ceiling
        event = {
            "ts": time.time(),
            "original_repo": original_repo,
            "effective_repo": effective_repo,
            "path_in_repo": path_in_repo,
            "used_tb": h.used_tb,
            "ceiling_tb": h.ceiling_tb,
            "reason": reason,
        }
        if projected_gb is not None:
            event["projected_gb"] = projected_gb
        path = _overflow_event_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.warning("overflow-routing event emit failed (%s) — reroute proceeds", e)


def _write_overflow_pointer(*, canonical_repo: str, path_in_repo: str, overflow_repo: str) -> None:
    """Upload a small JSON breadcrumb to the CANONICAL repo after a reroute.

    Small ``*.json`` commits ride the non-LFS path, which SUCCEEDS while over
    the public-storage quota (#541-validated) — so a consumer/verifier listing
    the canonical subfolder always finds a machine-readable pointer to the
    real location instead of an empty path. Fail-soft: a pointer-write failure
    logs loudly but never fails the (already-verified) rerouted upload.
    """
    import io

    try:
        h = check_hf_storage_headroom()
        payload = {
            "overflow_repo": overflow_repo,
            "path_in_repo": path_in_repo,
            "ts": time.time(),
            "used_tb": h.used_tb,
            "ceiling_tb": h.ceiling_tb,
        }
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        dest = (
            f"{path_in_repo.rstrip('/')}/OVERFLOW_POINTER.json"
            if path_in_repo
            else "OVERFLOW_POINTER.json"
        )
        _retry_upload(
            lambda: api.upload_file(
                path_or_fileobj=io.BytesIO(json.dumps(payload, indent=2).encode("utf-8")),
                repo_id=canonical_repo,
                path_in_repo=dest,
                repo_type="model",
            ),
            what="overflow-pointer upload_file",
        )
        logger.info("Wrote overflow pointer %s/%s -> %s", canonical_repo, dest, overflow_repo)
    except Exception as e:
        logger.warning(
            "overflow pointer write to %s failed (%s) — rerouted upload remains at %s",
            canonical_repo,
            e,
            overflow_repo,
        )


def list_repo_files_complete(
    api,
    repo_id: str,
    *,
    repo_type: str = "model",
    revision: str | None = None,
    path_in_repo: str | None = None,
) -> list[str]:
    """Enumerate EVERY file in an HF repo via the paginated tree API.

    The Hub's ``repo_info().siblings`` field — which several huggingface_hub
    code paths (and older ``list_repo_files`` implementations) read to list a
    repo's contents — SILENTLY TRUNCATES at roughly 7901 entries. On large
    repos (the project model + data repos accumulate thousands of checkpoint
    shards and raw-completion files) this truncation makes
    ``snapshot_download(allow_patterns=...)`` resolve to zero files even when
    the pattern matches files that are actually present.

    ``HfApi.list_repo_tree(recursive=True)`` is the paginated, complete
    alternative: it walks the repo tree page by page and yields one entry per
    file, with no truncation cap. This helper drives every enumeration in this
    module through it so a repo always enumerates fully regardless of the
    pinned huggingface_hub version's ``list_repo_files`` implementation.

    ``huggingface_hub``'s ``paginate`` retries ONLY HTTP 429 on follow-up
    cursor pages, so a 504 gateway timeout on any cursor page of a large repo
    otherwise propagates and turns a SUCCESSFUL upload's post-upload verify into
    a false failure. The paginated walk is therefore wrapped in the same
    transient-retry helper the upload sites use (gotchas.md "HF recursive tree
    listing 504s are un-retried"; #794/#658).

    Args:
        api: An ``huggingface_hub.HfApi`` instance (already token-scoped).
        repo_id: HF Hub repo ID.
        repo_type: ``'model'`` / ``'dataset'`` / ``'space'``.
        revision: Optional git revision; ``None`` resolves to the repo default.
        path_in_repo: Optional prefix that scopes the walk SERVER-side — the
            prefix rides in the tree URL, so pagination covers only that
            subtree. REQUIRED against the ~1M-file data repo, where a
            full-repo listing wedges (>600 s, #920 — the #833 gotcha).
            ``None`` (the default) preserves the historical full-repo walk;
            the kwarg is then NOT forwarded to ``list_repo_tree`` at all, so
            kwarg-free calls stay byte-identical (including against strict
            test fakes). A non-existent path raises ``EntryNotFoundError``
            DURING iteration (inside the retry thunk — the generator is
            lazy; verified live on hub 0.36.2), which ``_retry_upload``
            re-raises immediately (non-transient) for callers to map to
            their own missing semantics.

    Returns:
        Sorted list of every file path in the repo (or under ``path_in_repo``
        when given; ``RepoFolder`` entries are dropped; only files are
        returned).
    """
    from huggingface_hub.hf_api import RepoFile

    tree_kwargs: dict = {}
    if path_in_repo is not None:
        tree_kwargs["path_in_repo"] = path_in_repo

    def _list() -> list[str]:
        # ``list_repo_tree`` returns a generator; a cursor-page 504 raises
        # DURING iteration, so the comprehension is MATERIALIZED inside this
        # thunk (inside the retry ``try``) rather than after it returns.
        return [
            entry.path
            for entry in api.list_repo_tree(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                recursive=True,
                **tree_kwargs,
            )
            if isinstance(entry, RepoFile)
        ]

    files = _retry_upload(_list, what=f"list_repo_tree({repo_id})")
    return sorted(files)


def list_hf_files_under_path(
    api,
    repo_id: str,
    path: str,
    *,
    repo_type: str = "model",
    revision: str | None = None,
) -> list[str]:
    """Files under ``path`` via ONE server-side scoped tree walk — never a
    full-repo listing (#920: a bare listing wedges >600 s on the ~1M-file
    data repo).

    ``path`` naming a DIRECTORY returns every file under it (full repo-root-
    relative paths); an exact FILE returns ``[path]`` (the tree endpoint 404s
    on file paths — verified on hub 0.36.2, #939 — so an
    ``EntryNotFoundError`` falls back to one ``HfApi.file_exists`` HEAD
    probe, itself wrapped in ``_retry_upload``: the bare probe was the ONE
    un-retried Hub call on the sharded-upload verify path, and a Hub
    queue-full 429 there killed #1345's smoke upload leg after the shard had
    already landed — att-20260715-175238; the sibling fallback in
    ``verify_repo_paths_uploaded`` was already wrapped); an absent path
    returns ``[]``. Repository/Revision-not-found and
    transport/auth errors PROPAGATE (the file_exists fallback only fires
    after the tree call proved repo+revision resolve, so its swallowing of
    RepositoryNotFoundError is unreachable here). Empty ``path`` raises
    ValueError — a falsy path would silently degrade to the full-repo
    listing this helper exists to avoid.
    """
    from huggingface_hub.utils import EntryNotFoundError

    normalized = path.strip("/")
    if not normalized:
        raise ValueError("list_hf_files_under_path: empty path (would full-list the repo)")
    try:
        files = list_repo_files_complete(
            api, repo_id, repo_type=repo_type, revision=revision, path_in_repo=normalized
        )
    except EntryNotFoundError:
        if _retry_upload(
            lambda: api.file_exists(repo_id, normalized, repo_type=repo_type, revision=revision),
            what=f"file_exists({repo_id}/{normalized})",
        ):
            return [normalized]
        return []
    prefix = normalized + "/"
    # Defensive client-side filter: a no-op against real scoped results (every
    # returned path is under the prefix) but keeps strict test fakes — whose
    # list_repo_tree ignores path_in_repo — matching the same semantics.
    return [f for f in files if f == normalized or f.startswith(prefix)]


def _is_storage_quota_403(err: Exception) -> bool:
    """Persistent account-wide public-storage 403 (NOT transient). Mirrors the
    issue658 predicate; upload-policy.md § HF storage-quota 403."""
    msg = str(err)
    return "403" in msg and "storage" in msg.lower()


def _filecount_fallback_enabled() -> bool:
    """Default-ON kill switch for the reactive file-count overflow fallback (#1108).

    The canonical model repo hard-rejects pushes that would cross the HF
    100,000-files-per-repo limit (#1090: "Your git repo would contain 100050
    files after this push, over the limit of 100000 files"). When enabled,
    ``_upload`` retries such a REJECTED model-repo upload against the private
    :data:`DEFAULT_OVERFLOW_REPO`. Unlike the #564 byte-quota routing
    (default-OFF because a pre-emptive reroute can divert a push that would
    have succeeded), this fallback fires only AFTER the canonical push was
    refused — it can never reroute a would-succeed push — so it is strictly
    dominant and defaults ON. Kill switch: ``EPM_HF_FILECOUNT_FALLBACK=0``
    (restores the legacy log-and-return-"" behavior).
    """
    return os.environ.get("EPM_HF_FILECOUNT_FALLBACK", "1") == "1"


# Per-DIRECTORY commit cap enforced server-side by the Hub (#658 r2: a commit
# staging 12000 siblings into one dir 400'd "too many files ... up to 10000";
# huggingface/datasets#7956 confirms the per-directory rejection). DISTINCT
# from the repo-wide 100k git-file cap (#1108, _is_file_count_limit_error).
HUB_DIR_FILE_LIMIT = 10_000
# Advisory watermark = the gotchas.md shard recipe (shard_NNNN/ of <=5000).
HUB_DIR_FILECOUNT_WARN = 5_000


class HubDirFileCountError(ValueError):
    """A single upload_folder commit would stage more files into one repo
    directory than the Hub's server-side cap accepts (#658/#1190)."""


def _dir_filecount_guard_enabled() -> bool:
    """Default-ON kill switch. ``EPM_SKIP_HF_DIR_FILECOUNT_GUARD=1`` degrades
    the raise to a logged WARNING so the guard can never wedge a deliberate
    upload (mirrors the ``EPM_SKIP_*`` degrade-to-warn family)."""
    return os.environ.get("EPM_SKIP_HF_DIR_FILECOUNT_GUARD", "0") != "1"


def count_staged_files_per_repo_dir(
    folder_path: Path,
    path_in_repo: str,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> dict[str, int]:
    """Count the files ONE ``upload_folder`` commit would stage, keyed by
    TARGET repo directory (``path_in_repo`` prefix + relative subdir).

    Pure-local (``Path.rglob`` + ``huggingface_hub.utils.filter_repo_objects``
    — the library's OWN client-side filter, so allow/ignore semantics match
    what ``upload_folder`` will actually stage). No network; ~1.25 s at the
    pathological 10k-file scale, milliseconds at normal scale.
    """
    from huggingface_hub.utils import filter_repo_objects

    rels = [p.relative_to(folder_path).as_posix() for p in folder_path.rglob("*") if p.is_file()]
    # Parity with upload_folder's own default excludes (.git/ etc.).
    # Fact-checked on the pinned huggingface_hub 0.36.2: the constant lives at
    # huggingface_hub.utils (utils/_paths.py:25, a list of 8 patterns), NOT
    # huggingface_hub.constants (that import ERRORS); upload_folder itself
    # applies it (hf_api.py:4901). try/except kept only as future-version
    # drift defense (fallback direction: over-count, the safe side — the
    # kill switch unwedges).
    try:
        from huggingface_hub.utils import DEFAULT_IGNORE_PATTERNS

        ignore = list(DEFAULT_IGNORE_PATTERNS) + list(ignore_patterns or [])
    except ImportError:
        ignore = list(ignore_patterns or [])
    counts: dict[str, int] = {}
    prefix = path_in_repo.strip("/")
    for rel in filter_repo_objects(rels, allow_patterns=allow_patterns, ignore_patterns=ignore):
        repo_dir = posixpath.dirname(posixpath.join(prefix, rel) if prefix else rel)
        counts[repo_dir] = counts.get(repo_dir, 0) + 1
    return counts


def assert_hub_dir_filecounts(
    folder_path: Path | str,
    path_in_repo: str,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    limit: int = HUB_DIR_FILE_LIMIT,
    warn_at: int = HUB_DIR_FILECOUNT_WARN,
) -> dict[str, int]:
    """Fail loud BEFORE staging when any target repo dir would receive
    more than ``limit`` files in one commit (strict ``>``; the server accepts
    exactly 10,000). Returns the per-dir counts (for logging / tests).

    Public — direct ``HfApi.upload_folder`` callers in ``scripts/`` call this
    one-liner before their upload (the ``--check-hub-dir-filecount`` lint
    funnels them here), OUTSIDE any transient-retry wrapper (a guard raise is
    deterministic; retrying it burns the retry budget for nothing).

    Sequence semantics: callers relying on :func:`_upload`'s return-``""``
    soft-fail for upload-SEQUENCE independence now crash at the first
    offender — deliberate (the #595 pre-try ``ValueError`` precedent): the
    guarded class was already a guaranteed post-staging server 400, and the
    crash halts with data still local + the kill switch named at the crash
    site.

    STAGED-ONLY residuals (same false-negative direction; the server 400
    stays the late backstop): files already ON the remote repo dir; a dir
    built over the cap INCREMENTALLY via many small commits (per-file /
    per-cell paths — e.g. ``orchestrate.upload_sharded.upload_dir_sharded``,
    which commits ONE file per commit and is deliberately NOT wired to this
    guard); and the possibility the server counts directory ENTRIES
    (subdirs) rather than only files.
    """
    counts = count_staged_files_per_repo_dir(
        Path(folder_path),
        path_in_repo,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    offenders = {d: n for d, n in counts.items() if n > limit}
    if offenders:
        worst_dir, worst_n = max(offenders.items(), key=lambda kv: kv[1])
        # COMMA-FORMAT every numeric literal ({n:,} -> "10,000") — a bare
        # "5000"/"10000" contains the substring "500", which
        # _is_transient_upload_error's response-less scan reads as an HTTP 500
        # and would RETRY a deterministic guard failure if a direct caller
        # wraps the guard in a retry thunk. The comma breaks the substring.
        msg = (
            f"upload would stage {worst_n:,} files into repo dir '{worst_dir}' "
            f"({len(offenders)} dir(s) over the Hub's {limit:,}-files-per-directory "
            f"commit cap — a NON-retriable BadRequestError at create_commit, #658). "
            f"Re-shard into shard_NNNN/ subdirs of <= {warn_at:,} files "
            f"with a manifest (see .claude/rules/gotchas.md 'HF Hub rejects any "
            f"single repo directory holding >10k files'). Deliberate override: "
            f"EPM_SKIP_HF_DIR_FILECOUNT_GUARD=1. Call the guard OUTSIDE any "
            f"transient-retry wrapper."
        )
        if _dir_filecount_guard_enabled():
            raise HubDirFileCountError(msg)
        # NOTE: the #1108 overflow fallback re-enters _upload, so a
        # kill-switched over-limit upload logs this WARNING twice (once per
        # entry). Idempotent + harmless — not a bug.
        logger.warning("EPM_SKIP_HF_DIR_FILECOUNT_GUARD=1 set — proceeding despite: %s", msg)
    elif any(n > warn_at for n in counts.values()):
        big = {d: n for d, n in counts.items() if n > warn_at}
        logger.warning(
            "upload stages more than %s files into repo dir(s) %s — above the "
            "recommended shard size (cap is %s/dir; consider shard_NNNN/ now, "
            "gotchas.md).",
            f"{warn_at:,}",  # comma-format: see the msg note above
            big,
            f"{limit:,}",
        )
    return counts


def _is_file_count_limit_error(err: Exception) -> bool:
    """HF's repo-wide git file-count rejection ("Your git repo would contain
    N files after this push, over the limit of 100000 files" — verbatim in
    #1090's events; full format confirmed by HF forum thread 26400).

    Message-substring based — the exception CLASS the rejection surfaces as
    through ``upload_folder`` is deliberately not trusted (unverified; #1108
    plan §12 A3). The phrase is distinctive and digit-free, so the #989
    digit-triplet trap (paths like ``issue504_raw/`` reading as HTTP codes)
    does not apply. The ``push`` conjunct keeps per-FOLDER-cap (10k
    files/dir, #658) and per-commit-operation-cap rejections out of scope —
    the detector targets the repo-wide 100k phrase only.
    """
    msg = str(err).lower()
    return "over the limit of" in msg and "files" in msg and "push" in msg


def _is_transient_upload_error(err: Exception) -> bool:
    """True for retryable transient HF/HTTP upload errors (408/429/5xx by
    status code, connection drops / timeouts by message) — NOT the persistent
    storage-quota-403. When the exception carries a real integer HTTP status
    code, the decision is made ENTIRELY by code: 408 (request timeout), 429,
    and any 5xx are transient; every other code (all remaining 4xx) is
    non-transient, with NO substring fallback — 4xx messages can embed digit
    triplets ('issue504_raw/...', byte counts) that would read as
    false-transient (#989). The substring scan applies only to response-less
    errors (ConnectionError, timeouts).

    Response-less rate-limit text ('too many requests' / 'rate limit' /
    'queue size reached' — the HF/Xet upload-queue-saturation 429 body text,
    #1315/#1360) is transient (#931: a 429 during an hf_xet transfer can
    cross the Rust token-refresher boundary as a wrapped exception without
    ``.response``); NEVER bare '429' (the #989 digit-triplet trap). Note: a
    response-less PERMANENT failure whose text happens to contain one of
    these markers now burns the full retry budget before re-raising —
    bounded by design (``EPM_HF_RETRY_BUDGET_S``, default 1800 s)."""
    code = getattr(getattr(err, "response", None), "status_code", None)
    if isinstance(code, int):
        return code in (408, 429) or 500 <= code < 600
    msg = str(err).lower()
    return any(
        s in msg
        for s in (
            "504",
            "502",
            "503",
            "500",
            "gateway time-out",
            "gateway timeout",
            "timed out",
            "timeout",
            "connection",
            "temporarily unavailable",
            "too many requests",  # response-less 429 text — xet Rust boundary (#931)
            "rate limit",  # matches "rate limit(ed)"; NEVER bare "429" (#989)
            "queue size reached",  # HF/Xet upload-queue-saturation 429 body text
            # (#1315/#1360) — can cross the hf_xet PyO3 boundary without
            # .response and without the "too many requests" phrase. Words-only:
            # immune to the #989 digit-triplet trap; response-BEARING errors
            # are still decided entirely by status code (a 4xx carrying this
            # text stays non-transient).
        )
    )


_RETRY_AFTER_CAP_S = 900.0  # defensive cap on a pathological server Retry-After header
_BACKOFF_CAP_S = 180.0  # exp-backoff ceiling (#735); also caps Retry-After under budget_s=0


def _retry_after_seconds(err: Exception) -> float | None:
    """Seconds from a ``Retry-After`` header on the error's response, if any.

    Seconds-form only (an RFC 9110 HTTP-date value parses to None -> caller
    falls back to exp backoff). Mirrors ``llm/api_dispatch._retry_after_seconds``.
    """
    headers = getattr(getattr(err, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")  # requests' CaseInsensitiveDict
    except Exception:
        return None
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _retry_budget_s() -> float:
    """Wall-clock transient-retry budget (s). 0 disables the budget extension
    (legacy attempt-bound behavior). Env: ``EPM_HF_RETRY_BUDGET_S`` (default
    1800). Unparseable or NON-FINITE values (``inf``/``nan``/``1e999``) fall
    back to 1800 with a warning — ``inf`` would make the fail-loud deadline
    unbounded (retry forever on a permanently-down Hub), and ``nan`` would
    silently degrade to the 0 kill switch via ``max(0.0, nan)``."""
    raw = os.environ.get("EPM_HF_RETRY_BUDGET_S")
    if raw is None or not raw.strip():
        return 1800.0
    try:
        val = float(raw)
    except ValueError:
        logger.warning("EPM_HF_RETRY_BUDGET_S=%r unparseable; using 1800", raw)
        return 1800.0
    if not math.isfinite(val):
        logger.warning("EPM_HF_RETRY_BUDGET_S=%r non-finite; using 1800", raw)
        return 1800.0
    return max(0.0, val)


def _retry_upload(fn, *, what: str, max_attempts: int = 6, budget_s: float | None = None):
    """Call ``fn()`` (a zero-arg thunk) with retry on transient HF
    5xx/429/timeout/connection errors. Name is legacy — generic transient-retry
    wrapper, also used for READS (``list_repo_files_complete``) and downloads.

    Retry is allowed while EITHER bound holds (raise only when BOTH exhaust):
      - attempt floor: the first ``max_attempts`` calls (the #735 contract);
      - wall-clock budget ``budget_s`` (default env ``EPM_HF_RETRY_BUDGET_S`` =
        1800): sized to outlive an org-wide 429 storm — #931's storm outlived
        the pre-#997 310 s attempt-bound stack. ``budget_s=0`` => legacy
        attempt-bound behavior.

    Bound convention: with ``budget > 0``, no sleep starts or extends past the
    deadline and TOTAL SLEEP <= budget; elapsed wall time can exceed the budget
    only by IN-FLIGHT call durations (each ``fn()`` — including
    huggingface_hub's inner pagination retries — runs to completion before the
    deadline check). Attempt-floor retries past the deadline sleep 0 and retry
    immediately (<= ``max_attempts`` calls total).

    Sleep: ``Retry-After`` header when present (capped ``_RETRY_AFTER_CAP_S``;
    under ``budget_s=0`` capped at ``_BACKOFF_CAP_S`` instead — the kill
    switch has no deadline clamp, so a pathological header would otherwise be
    honored un-clamped, defeating the switch's fail-fast purpose), else exp
    backoff ``min(180, 10*2^k)`` with 0-25% jitter (de-synchronizes fleet
    retries). Storage-quota-403 / non-transient re-raise IMMEDIATELY; on
    exhaustion the final exception propagates (fail-loud, no swallow).
    """
    budget = _retry_budget_s() if budget_s is None else budget_s
    start = time.monotonic()
    deadline = start + budget
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as e:
            if _is_storage_quota_403(e) or not _is_transient_upload_error(e):
                raise
            ra = _retry_after_seconds(e)
            if ra is not None:
                # budget_s=0 (kill switch) skips the deadline clamp below, so
                # cap the header at the legacy backoff ceiling there — else a
                # pathological Retry-After stacks 5 x 900 s of sleep vs the
                # ~310 s legacy stack the switch is meant to restore.
                sleep_s = min(ra, _RETRY_AFTER_CAP_S if budget > 0 else _BACKOFF_CAP_S)
            else:
                sleep_s = min(_BACKOFF_CAP_S, 10.0 * 2.0 ** min(attempt - 1, 6)) * (
                    1.0 + random.random() * 0.25
                )
            now = time.monotonic()
            within_attempts = attempt < max_attempts
            within_budget = budget > 0 and now < deadline
            if not (within_attempts or within_budget):
                logger.warning(
                    "%s transient-retry exhausted after %d calls (elapsed %.0fs, "
                    "budget %.0fs); re-raising",
                    what,
                    attempt,
                    now - start,
                    budget,
                )
                raise
            if budget > 0:
                # Clamp EVERY sleep — the Retry-After branch AND the backoff
                # branch, including attempt-floor retries — to the remaining
                # budget, so the attempt floor can never stack un-clamped
                # Retry-After sleeps past the deadline (pathological
                # Retry-After: 4000 -> 900-cap x 5 floor attempts ~ 4500 s >
                # 1800 s budget). With the clamp, TOTAL SLEEP <= budget; floor
                # attempts after the deadline sleep 0 and retry immediately
                # (<= max_attempts calls total — the #735 6-call contract
                # holds; legacy tests assert sleep COUNTS, never durations).
                sleep_s = min(sleep_s, max(0.0, deadline - now))
            logger.warning(
                "%s transient error (attempt %d, elapsed %.0fs / budget %.0fs): %s; "
                "retrying in %.0fs",
                what,
                attempt,
                time.monotonic() - start,
                budget,
                str(e)[:200],
                sleep_s,
            )
            time.sleep(sleep_s)


# Public, greppable name for per-issue dispatch scripts (#606: scripts assumed a
# hub `_retry_transient` that never existed; the i528 family hand-rolled four copies).
retry_transient = _retry_upload


def verify_repo_paths_uploaded(
    api,
    repo_id: str,
    expected_repo_paths: Sequence[str],
    *,
    path_in_repo: str,
    repo_type: str = "dataset",
    revision: str | None = None,
) -> list[str]:
    """Exact-set post-upload verify: return expected paths NOT on the Hub.

    Canonical retried + server-side-SCOPED verify leg for dispatch scripts.
    #920: a bare full-repo ``list_repo_files`` on the ~1M-file data repo wedges
    >600 s (#833 gotcha) AND a transient 500 on it crashed a workload after
    every upload had succeeded. Routes through ``list_repo_files_complete``
    with ``path_in_repo`` scoping — the paginated walk rides ``_retry_upload``
    (Retry-After-aware, wall-clock-budgeted).

    ``path_in_repo`` is a REQUIRED non-empty prefix covering every expected
    path (ValueError otherwise — an unscoped verify recreates the wedge). A
    directory-like prefix absent on the repo (``EntryNotFoundError`` during
    the walk — hub 0.36.2's lazy ``list_repo_tree`` generator raises it inside
    the retry thunk) returns ALL expected paths as missing (caller's fail-loud
    fires with the full list). An exact-FILE prefix (an expected path EQUAL to
    ``path_in_repo``) ALSO raises ``EntryNotFoundError`` — the tree endpoint
    404s on file paths (verified live on hub 0.36.2, #939; the sibling
    ``list_hf_files_under_path`` documents the same behavior) — so that case
    falls back to ONE retried ``HfApi.file_exists`` HEAD probe instead of
    falsely reporting a successfully-uploaded file as missing. Transport/auth
    errors propagate after the retry budget.
    """
    from huggingface_hub.utils import EntryNotFoundError

    prefix = path_in_repo.strip("/")
    if not prefix:
        raise ValueError("verify_repo_paths_uploaded: empty path_in_repo (unscoped verify)")
    expected = list(expected_repo_paths)
    outside = [p for p in expected if not (p == prefix or p.startswith(prefix + "/"))]
    if outside:
        raise ValueError(
            f"verify_repo_paths_uploaded: {len(outside)} expected paths outside "
            f"path_in_repo={prefix!r} (first: {outside[:3]})"
        )
    try:
        uploaded = set(
            list_repo_files_complete(
                api, repo_id, repo_type=repo_type, revision=revision, path_in_repo=prefix
            )
        )
    except EntryNotFoundError:
        # The tree endpoint 404s on an exact-FILE path as well as on an absent
        # prefix (hub 0.36.2, #939 — the same live behavior the sibling
        # ``list_hf_files_under_path`` handles at its ``file_exists``
        # fallback). When the expected set names the prefix itself — the only
        # way ``p == prefix`` passes the coverage check above — probe the file
        # directly, wrapped in ``_retry_upload`` (a fresh Hub call on the
        # verify path; un-retried, a transient 500 here would reintroduce the
        # #920 class through the fallback). Probe True => the exact file IS
        # uploaded (a same-named subtree cannot coexist with a file, so any
        # ``prefix + "/"`` children stay missing); probe False => all expected
        # paths are missing. Directory-like prefixes (``prefix`` not in the
        # expected set) keep the all-missing semantics unchanged.
        if prefix in expected and _retry_upload(
            lambda: api.file_exists(repo_id, prefix, repo_type=repo_type, revision=revision),
            what=f"file_exists({repo_id}/{prefix})",
        ):
            return [p for p in expected if p != prefix]
        return expected
    return [p for p in expected if p not in uploaded]


def _upload(
    local_path: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    delete_after: bool = False,
    upload_as_file: bool = False,
    ignore_patterns: list[str] | None = None,
    private: bool = False,
) -> str:
    """Shared upload logic for models and datasets.

    Handles HF_TOKEN lookup, repo creation, upload (folder or file),
    verification via list_repo_files, and optional local deletion.

    Folder uploads ALWAYS exclude :data:`TRAINING_STATE_IGNORE_PATTERNS`
    (optimizer/scheduler/RNG state) — there is no opt-out, because that state
    is never a useful Hub artifact and historically accounted for hundreds of
    GB of accidental residue.

    Reactive file-count fallback (#1108): a MODEL-repo upload rejected with
    HF's repo-wide 100k file-count message (:func:`_is_file_count_limit_error`)
    is retried once against the private :data:`DEFAULT_OVERFLOW_REPO` (same
    ``path_in_repo``), emitting the #564 routing event
    (``reason="file-count-limit-reactive"``) + the ``OVERFLOW_POINTER.json``
    breadcrumb on the canonical repo after a VERIFIED overflow landing.
    Default ON; kill switch ``EPM_HF_FILECOUNT_FALLBACK=0``
    (:func:`_filecount_fallback_enabled`). Recursion is bounded by
    construction — the recursive call targets the overflow repo, on which the
    guard short-circuits. Every other failure keeps the legacy
    log-and-return-"" behavior; the success path is byte-unchanged.

    Args:
        local_path: Local file or directory to upload (already resolved to Path).
        repo_id: HF Hub repo ID.
        repo_type: 'model' or 'dataset'.
        path_in_repo: Sub-path in the repo. For single files, this is the
            destination path; empty string falls back to the local filename.
        delete_after: Delete local path after verified upload.
        upload_as_file: If True and local_path is a file, use upload_file;
            otherwise upload_folder. Directories always use upload_folder.
        ignore_patterns: Extra fnmatch patterns to exclude from FOLDER uploads,
            merged with the always-on training-state excludes. Ignored for
            single-file uploads.
        private: Create a MISSING repo as private (threaded into create_repo).
            Default False preserves historical behavior at every existing call
            site; the overflow-routing path passes True so a not-yet-existing
            overflow repo is never created PUBLIC (which would put rerouted
            LFS straight back under the blocked public quota, #564).

    Returns:
        "{repo_id}/{path_in_repo}" on verified success, "" on any failure.
    """
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set, skipping upload")
        return ""

    if not local_path.exists():
        logger.warning("Path %s does not exist, skipping upload", local_path)
        return ""

    # Fail loud on the silent-no-op class (#595, 2026-06-13): a FILE handed to
    # the folder branch (upload_as_file=False) makes huggingface_hub.upload_folder
    # log "Provided path: ... is not a directory. Keeping local path." and upload
    # NOTHING, yet verification can still pass if same-prefix files already exist
    # — silent data loss. Single-file callers MUST pass upload_as_file=True.
    if local_path.is_file() and not upload_as_file:
        raise ValueError(
            f"_upload received a file path ({local_path}) with upload_as_file=False; "
            "upload_folder silently no-ops on a file path. Pass upload_as_file=True for "
            "single-file uploads (see upload_raw_completions_to_data_repo)."
        )

    # #1190: pre-count staged files per TARGET repo dir before any network
    # I/O — the Hub rejects a commit staging >10k siblings into one dir with
    # a NON-retriable BadRequestError AFTER all bytes are staged (#658 r2).
    # Placed BEFORE HfApi construction and OUTSIDE the try below so the raise
    # propagates instead of being swallowed into `except Exception -> ""`
    # (the #595 pre-try precedent).
    if local_path.is_dir():
        assert_hub_dir_filecounts(
            local_path,
            path_in_repo,
            ignore_patterns=TRAINING_STATE_IGNORE_PATTERNS + list(ignore_patterns or []),
        )

    api = HfApi(token=token)

    # Repo should already exist (public), but create if missing
    try:
        api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create/verify repo %s: %s", repo_id, e)

    logger.info("Uploading %s -> %s/%s", local_path, repo_id, path_in_repo)

    is_file_upload = upload_as_file and local_path.is_file()

    try:
        if is_file_upload:
            _retry_upload(
                lambda: api.upload_file(
                    path_or_fileobj=str(local_path),
                    repo_id=repo_id,
                    path_in_repo=path_in_repo or local_path.name,
                    repo_type=repo_type,
                ),
                what="upload_file",
            )
        else:
            _retry_upload(
                lambda: api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=repo_id,
                    path_in_repo=path_in_repo,
                    repo_type=repo_type,
                    ignore_patterns=TRAINING_STATE_IGNORE_PATTERNS + list(ignore_patterns or []),
                ),
                what="upload_folder",
            )

        # Verify upload: check that files actually exist on Hub. Scoped verify
        # (#920/#988): never full-list the repo to confirm one upload — a bare
        # listing wedges >600 s on the ~1M-file data repo. Exact-file uploads
        # resolve via the helper's file_exists fallback; folder uploads via
        # the server-side scoped tree walk (paginated, so a large subtree
        # never spuriously reports 0 committed files).
        expected_prefix = (path_in_repo or local_path.name).rstrip("/")
        committed_files = list_hf_files_under_path(
            api, repo_id, expected_prefix, repo_type=repo_type
        )

        if not committed_files:
            logger.error(
                "Upload appeared to succeed but 0 files found under %s/%s on Hub. "
                "NOT marking as successful.",
                repo_id,
                expected_prefix,
            )
            return ""

        logger.info(
            "Upload verified: %d files at %s/%s",
            len(committed_files),
            repo_id,
            path_in_repo,
        )

        if delete_after:
            shutil.rmtree(str(local_path), ignore_errors=True)
            logger.info("Deleted local path: %s", local_path)

        return f"{repo_id}/{path_in_repo}"
    except Exception as e:
        if (
            _filecount_fallback_enabled()
            and _is_file_count_limit_error(e)
            and repo_type == "model"
            and repo_id != DEFAULT_OVERFLOW_REPO
        ):
            logger.warning(
                "File-count limit rejection on %s (%s) — falling back to overflow repo %s",
                repo_id,
                e,
                DEFAULT_OVERFLOW_REPO,
            )
            # Bounded by construction: the recursive call carries
            # repo_id=DEFAULT_OVERFLOW_REPO, on which the guard above
            # short-circuits. delete_after rides along, so the local copy is
            # reaped only after the recursive call's OWN verified landing.
            result = _upload(
                local_path,
                DEFAULT_OVERFLOW_REPO,
                repo_type,
                path_in_repo,
                delete_after=delete_after,
                upload_as_file=upload_as_file,
                ignore_patterns=ignore_patterns,
                private=True,
            )
            if result:
                _emit_overflow_routing_event(
                    original_repo=repo_id,
                    effective_repo=DEFAULT_OVERFLOW_REPO,
                    path_in_repo=path_in_repo,
                    reason="file-count-limit-reactive",
                )
                # Fail-soft breadcrumb on the CANONICAL repo (non-LFS, small).
                # It ADDS one file per reroute — fine near the limit, fails
                # soft (logged) at exactly 100,000.
                _write_overflow_pointer(
                    canonical_repo=repo_id,
                    path_in_repo=path_in_repo,
                    overflow_repo=DEFAULT_OVERFLOW_REPO,
                )
            return result
        logger.error("Upload failed: %s. Keeping local path.", e)
        return ""


def _upload_folder_filtered(
    local_dir: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    allow_patterns: list[str],
    expected_repo_paths: list[str],
    ignore_patterns: list[str] | None = None,
    delete_after: bool = False,
) -> str:
    """Bulk-upload a SUBSET of a local folder in ONE ``upload_folder`` commit.

    This is the ``allow_patterns``-threaded sibling of :func:`_upload`'s folder
    branch. ``_upload`` itself does NOT expose ``allow_patterns`` (its public
    signature is pinned by several single-file callers + workflow-invariant
    tests), so a caller that needs to upload only a glob-selected subset of a
    directory — e.g. only ``raw_completions.json`` files out of an
    ``eval_results/`` tree that also holds aggregate JSONs — routes through this
    helper instead. ``HfApi.upload_folder`` composes exactly ONE
    ``create_commit`` for the whole matched set (it walks the repo tree only
    when ``delete_patterns`` is passed, which this helper never does), so a bulk
    upload of N files issues ONE commit — never the per-file recursive
    tree-listing pre-check that 504-storms on a large repo (the #664 / #727
    incident: a per-file ``upload_file`` loop of 1425 files ran 12h / ~$530 on
    an idle 8xH200 and uploaded only 264).

    Args:
        local_dir: Local DIRECTORY to upload from (must be a directory).
        repo_id: HF Hub repo ID.
        repo_type: ``'model'`` / ``'dataset'``.
        path_in_repo: Destination prefix in the repo; each matched file lands at
            ``<path_in_repo>/<rel-to-local_dir>``.
        allow_patterns: fnmatch globs (relative to ``local_dir``) selecting which
            files to upload. Files not matching are NOT uploaded.
        expected_repo_paths: the EXACT set of ``<path_in_repo>/<rel>`` paths that
            MUST be present on the Hub after the commit. Verification is an
            exact expected-set membership check on a fresh paginated listing
            (NOT mere prefix-presence / count — a mid-``upload_folder`` crash
            leaves a partial set that prefix-presence would wrongly pass; the
            ``.claude/rules/upload-policy.md`` § per-cell rule). Any missing
            expected path -> return ``""`` so the caller raises.
        ignore_patterns: extra fnmatch excludes, merged with the always-on
            training-state excludes (same semantics as :func:`_upload`).
        delete_after: when True, the CALLER deletes the individual local files
            after this returns a non-empty (verified) URL — this helper never
            deletes (so it cannot remove a file whose committed prefix was not
            verified; the set-verify happens BEFORE the caller's unlink).

    Returns:
        ``"{repo_id}/{path_in_repo}"`` on verified success, ``""`` on any
        failure or incomplete commit. ``delete_after`` is accepted for signature
        symmetry but intentionally not acted on here; see the arg note above.
    """
    # delete_after is verified-before-acted-on by the CALLER (set-verify happens
    # below, before any unlink) — this helper never deletes, so a partial commit
    # can never strand a deleted-but-unverified local file.
    _ = delete_after

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set, skipping upload")
        return ""

    if not local_dir.is_dir():
        logger.warning("Path %s is not a directory, skipping bulk upload", local_dir)
        return ""

    # #1190: pre-count the ACTUALLY-staged subset (allow/ignore patterns
    # respected) per TARGET repo dir before any network I/O — same guard +
    # placement rationale as _upload's folder branch (before HfApi, outside
    # the try, so the raise propagates instead of returning "").
    assert_hub_dir_filecounts(
        local_dir,
        path_in_repo,
        allow_patterns=allow_patterns,
        ignore_patterns=TRAINING_STATE_IGNORE_PATTERNS + list(ignore_patterns or []),
    )

    api = HfApi(token=token)

    try:
        api.create_repo(repo_id, repo_type=repo_type, private=False, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create/verify repo %s: %s", repo_id, e)

    logger.info(
        "Bulk-uploading %s (allow_patterns=%s) -> %s/%s",
        local_dir,
        allow_patterns,
        repo_id,
        path_in_repo,
    )

    try:
        _retry_upload(
            lambda: api.upload_folder(
                folder_path=str(local_dir),
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                allow_patterns=allow_patterns,
                ignore_patterns=TRAINING_STATE_IGNORE_PATTERNS + list(ignore_patterns or []),
            ),
            what="upload_folder (filtered)",
        )

        # EXACT expected-set verification on a fresh paginated listing (mirror
        # the #664 per-cell rule), SCOPED to path_in_repo (#920/#988): every
        # element of expected_repo_paths is <path_in_repo>/<rel> by this
        # function's contract (see the expected_repo_paths arg doc + the sole
        # caller upload_raw_completions_to_data_repo), so the scoped walk sees
        # every checkable path — never a full ~1M-file repo listing. An
        # expected path OUTSIDE the prefix would have been flagged missing by
        # the old full listing too (it was never uploaded under this prefix),
        # so flagging it against the scoped set is not a semantics change.
        uploaded_files = set(
            list_hf_files_under_path(api, repo_id, path_in_repo.rstrip("/"), repo_type=repo_type)
        )
        missing = [p for p in expected_repo_paths if p not in uploaded_files]
        if missing:
            logger.error(
                "Bulk upload incomplete: %d of %d expected files missing under "
                "%s/%s on Hub (first missing: %s). NOT marking as successful.",
                len(missing),
                len(expected_repo_paths),
                repo_id,
                path_in_repo,
                missing[0],
            )
            return ""

        logger.info(
            "Bulk upload verified: %d files at %s/%s",
            len(expected_repo_paths),
            repo_id,
            path_in_repo,
        )
        return f"{repo_id}/{path_in_repo}"
    except Exception as e:
        logger.error("Bulk upload failed: %s. Keeping local files.", e)
        return ""


def upload_model(
    model_path: str,
    repo_id: str = DEFAULT_MODEL_REPO,
    condition_name: str = "",
    seed: int = 0,
    path_in_repo: str | None = None,
    delete_after: bool = False,
    ignore_patterns: list[str] | None = None,
) -> str:
    """Upload a model directory to HuggingFace Hub, optionally delete the local copy.

    Optimizer/scheduler/RNG state files are ALWAYS excluded (see
    :data:`TRAINING_STATE_IGNORE_PATTERNS`).

    Opt-in overflow routing (#564): when ``EPM_HF_OVERFLOW_ROUTING=1`` (default
    off) and the account is KNOWN over the public-storage soft ceiling, the
    upload reroutes to the private :data:`DEFAULT_OVERFLOW_REPO` (created
    private if missing), a deviation event lands on the local JSONL sink, and a
    small ``OVERFLOW_POINTER.json`` breadcrumb is committed to the CANONICAL
    repo at ``<path_in_repo>/OVERFLOW_POINTER.json`` (non-LFS — works over
    quota). ARMING CONTRACT: safe ONLY for flows that consume this function's
    returned URL or read the pointer/deviation records; launchers that verify
    canonical paths EXTERNALLY must not arm it (see
    ``.claude/rules/upload-policy.md`` § Proactive detection).

    Args:
        model_path: Local path to the model directory (adapter dir by project
            default; merged dirs only behind :func:`merged_upload_enabled`).
        repo_id: HF Hub repo ID. Defaults to the public model repo.
        condition_name: Condition name for organizing in the repo.
        seed: Seed number.
        path_in_repo: Override the sub-path in the repo. If None, uses
            '{condition_name}_seed{seed}'.
        delete_after: Delete local model after successful upload. Default False
            for safety — caller must explicitly opt in.
        ignore_patterns: Extra fnmatch patterns to exclude (e.g.
            ``["checkpoint-*"]`` for an adapter-only upload), merged with the
            always-on training-state excludes.

    Returns:
        The HF Hub path where the model was uploaded.

    Size-aware routing note (#1034): the armed resolver receives this dir's
    on-disk byte sum (an rglob walk — milliseconds for adapter dirs) so an
    armed flow reroutes when ``used + projected > ceiling``, not only when
    already over. The walk OVER-counts vs what is actually sent (it includes
    ``TRAINING_STATE_IGNORE_PATTERNS``-excluded files) — conservative: an
    over-projection can only reroute slightly early, never under-protect.
    """
    if path_in_repo is None:
        path_in_repo = f"{condition_name}_seed{seed}"

    projected = sum(f.stat().st_size for f in Path(model_path).rglob("*") if f.is_file())
    effective_repo, rerouted = _resolve_lfs_upload_repo(repo_id, projected_bytes=projected)
    if rerouted:
        logger.warning(
            "EPM_HF_OVERFLOW_ROUTING: rerouting LFS upload %s -> %s "
            "(public storage over soft ceiling)",
            repo_id,
            effective_repo,
        )
        _emit_overflow_routing_event(
            original_repo=repo_id, effective_repo=effective_repo, path_in_repo=path_in_repo
        )

    result = _upload(
        local_path=Path(model_path),
        repo_id=effective_repo,
        repo_type="model",
        path_in_repo=path_in_repo,
        delete_after=delete_after,
        upload_as_file=False,
        ignore_patterns=ignore_patterns,
        # A direct upload to the overflow repo must also never create it
        # public — private quota separation is the whole point.
        private=rerouted or repo_id == DEFAULT_OVERFLOW_REPO,
    )
    if rerouted and result:
        _write_overflow_pointer(
            canonical_repo=repo_id, path_in_repo=path_in_repo, overflow_repo=effective_repo
        )
    return result


def upload_dataset(
    data_path: str,
    repo_id: str = DEFAULT_DATASET_REPO,
    path_in_repo: str = "",
) -> str:
    """Upload a dataset file or directory to HuggingFace Hub.

    Args:
        data_path: Local path to a dataset file (.jsonl, .json, .parquet) or directory.
        repo_id: HF Hub dataset repo ID. Defaults to the public dataset repo.
        path_in_repo: Sub-path in the repo (e.g. 'phase1/evil_wrong.jsonl').

    Returns:
        The HF Hub path where the dataset was uploaded.
    """
    return _upload(
        local_path=Path(data_path),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        delete_after=False,
        upload_as_file=True,
    )


def upload_dataset_directory(
    data_dir: Path,
    bucket: str,
    *,
    no_upload: bool = False,
    fail_soft: bool = False,
    pattern: str = "*.jsonl",
) -> list[str]:
    """Upload every file matching ``pattern`` in ``data_dir`` to HF Hub.

    Each file lands at ``<bucket>/<file.name>`` on the dataset repo. The
    helper is the single call site every data-gen script in ``scripts/``
    should use to honor CLAUDE.md's Upload Policy ("Datasets MUST be
    uploaded — Auto after generation").

    **Fail-loud contract (default ``fail_soft=False``).** The underlying
    :func:`upload_dataset` swallows every internal error and returns ``""``
    in five cases: (1) ``HF_TOKEN`` not set, (2) local path missing, (3)
    repo-create failure, (4) the upload-and-list verification step finds
    zero files at the expected prefix, (5) any other exception in the HF
    API path. This helper treats an empty-string return from
    :func:`upload_dataset` AS A FAILURE and raises ``RuntimeError`` so the
    calling script exits non-zero. It also re-raises any exception that
    :func:`upload_dataset` lets propagate (today: none, but defends
    against future changes to the lower helper). Either way, the calling
    script never silently succeeds when the upload didn't actually land.

    **Soft mode (``fail_soft=True``).** Same detection of the two failure
    surfaces (``""`` return + exception), but instead of raising the
    helper logs to stderr and continues to the next file. The returned
    list contains ONLY successfully-uploaded paths; failed files are not
    in it. Use this only for genuinely best-effort callers — no current
    data-gen script qualifies; CLAUDE.md's Upload Policy is fail-loud.

    Parameters
    ----------
    data_dir
        Directory containing dataset files. Globbed non-recursively.
    bucket
        Path-in-repo prefix on the dataset repo (e.g. ``"a3/"``,
        ``"lang_inv/"``). Trailing slash optional; normalised internally.
    no_upload
        If True, log "skipping HF Hub upload" to stdout and return ``[]``
        without doing any network I/O. Used for dry-run / ``--no-upload``
        CLI flag.
    fail_soft
        Default behaviour (False) is FAIL-LOUD: on any upload error
        (raised exception OR ``""`` return from :func:`upload_dataset`),
        write to stderr and raise ``RuntimeError`` so the calling script
        exits non-zero. CLAUDE.md's Upload Policy requires datasets to
        land on the Hub, so the default upholds that contract. Pass
        ``fail_soft=True`` only for genuinely best-effort callers.
    pattern
        Glob pattern applied to ``data_dir.glob(pattern)`` (non-recursive).
        Defaults to ``"*.jsonl"``. Callers passing a literal filename
        with glob metacharacters (e.g. ``"data_[v1].jsonl"``) trigger an
        automatic ``glob.escape`` — see #293 §3 v3 P7.

        Caveat: the auto-escape heuristic activates when the pattern
        contains ``[`` or ``]`` but no ``*`` or ``?``. Callers that
        intentionally want to use a glob character class (e.g.
        ``"file_[abc].jsonl"`` to match ``file_a.jsonl`` etc.) must
        include a ``*`` or ``?`` somewhere in the pattern to bypass the
        heuristic. Existing data-gen filenames don't use brackets, so
        this is a documentation-level constraint only.

    Returns
    -------
    list[str]
        Sorted list of ``path_in_repo`` strings actually uploaded
        (empty-string returns from :func:`upload_dataset` are NOT
        included). Empty when ``no_upload=True`` or no files match.

    Raises
    ------
    RuntimeError
        Raised when ``fail_soft=False`` and :func:`upload_dataset`
        returns ``""`` for any file (lower helper's silent-failure
        return — see "Fail-loud contract" above).
    Exception
        Re-raised from :func:`upload_dataset` when ``fail_soft=False``
        and the lower helper raises rather than returning ``""``.
    """
    bucket = bucket.rstrip("/") + "/"
    # v3 P7 defense: callers that pass a literal filename (single-file
    # scripts use ``pattern=output_path.name``) silently mismatch if the
    # filename contains glob metacharacters (``[``, ``*``, ``?``). Detect
    # that intent by checking the pattern for class brackets without
    # explicit wildcards, and ``glob.escape`` if it looks literal. A
    # genuine glob (contains ``*`` or ``?``) passes through unchanged.
    if any(ch in pattern for ch in "[]") and not any(ch in pattern for ch in "*?"):
        pattern = glob.escape(pattern)
    files = sorted(data_dir.glob(pattern))
    if no_upload:
        print(f"  --no-upload set; skipping HF Hub upload of {len(files)} file(s) from {data_dir}")
        return []
    if not files:
        print(
            f"  upload_dataset_directory: no files in {data_dir} matching "
            f"{pattern!r} — nothing to upload"
        )
        return []
    print(f"  Uploading {len(files)} dataset file(s) to HF Hub ({bucket})...")
    uploaded: list[str] = []
    for f in files:
        path_in_repo = f"{bucket}{f.name}"
        try:
            ret = upload_dataset(data_path=str(f), path_in_repo=path_in_repo)
        except Exception as e:
            # upload_dataset rarely raises today (all paths return ""),
            # but we defend the contract regardless.
            print(
                f"  upload_dataset_directory: upload of {f.name} -> {path_in_repo} "
                f"FAILED with exception: {e}",
                file=sys.stderr,
            )
            if fail_soft:
                print(
                    "  (fail_soft=True; continuing; local file preserved)",
                    file=sys.stderr,
                )
                continue
            raise

        # Fail-loud on the silent-failure path: upload_dataset returned ""
        # because of HF_TOKEN missing / 401 / 403 / verification failure /
        # caught exception inside _upload. Treat as failure.
        if not ret:
            msg = (
                f"upload_dataset returned '' for {f} -> {path_in_repo}; "
                "HF Hub upload failed silently (HF_TOKEN missing, 4xx, "
                "or verification mismatch — see logs above for the "
                "underlying cause)"
            )
            print(f"  upload_dataset_directory: {msg}", file=sys.stderr)
            if fail_soft:
                print(
                    "  (fail_soft=True; continuing; local file preserved)",
                    file=sys.stderr,
                )
                continue
            raise RuntimeError(msg)
        uploaded.append(path_in_repo)
    return uploaded


def upload_raw_completions_to_data_repo(
    experiment_name: str,
    eval_results_dir: Path,
    delete_after: bool = False,
) -> dict[str, str]:
    """Upload all raw_completions.json files in an experiment's eval_results
    directory to the HF Hub data repo IN ONE bulk ``upload_folder`` commit.

    Files land under ``<experiment_name>/raw_completions/<rel_path>`` in
    ``DEFAULT_DATASET_REPO``. Fail-loud (raises ``RuntimeError`` on any upload
    failure), verified via an EXACT expected-file-set check on a fresh
    paginated ``list_repo_files_complete`` listing inside
    :func:`_upload_folder_filtered`.

    The whole matched tree uploads as a SINGLE ``upload_folder`` commit (the
    canonical bulk-upload path), NOT a per-file ``upload_file`` loop. Each
    ``upload_file`` call triggers a server-side recursive tree-listing of the
    target repo as a pre-check; once the data repo grew large that listing
    504-times-out roughly half the time, so a per-file loop of N files stalls
    indefinitely (the #664 / #727 incident: 508 attempts in 12h uploaded only
    264 of 1425 files while an 8xH200 sat at 0% GPU, ~$530 burned).
    ``upload_folder`` composes ``create_commit`` exactly once and does NO
    per-file listing.

    Use this from an experiment entry script after eval to persist the
    per-generation strings before pod termination — these can be 10-200MB
    per adapter and are too big for git, so HF Hub data repo is the
    canonical destination (see CLAUDE.md Upload Policy).

    Args:
        experiment_name: e.g. ``"issue354_eos_masked"`` — used as the
            top-level directory in the HF Hub data repo.
        eval_results_dir: e.g. ``Path("eval_results/issue354_eos_masked")``
            — scanned recursively for files named ``raw_completions.json``.
        delete_after: if True, delete each local ``raw_completions.json``
            after the bulk upload has VERIFIED the whole expected set landed
            on the Hub (the set-verify happens before any unlink, so a partial
            commit can never strand a deleted-but-unverified file). Only the
            individual ``raw_completions.json`` files are removed — never the
            enclosing ``eval_results_dir`` (which holds aggregate JSONs the
            ``allow_patterns`` deliberately skipped). Default False — the
            upload-verifier does its own cleanup pass for ``eval_results/``.

    Returns:
        dict mapping local relative path → HF Hub URL on success (one entry per
        matched file, identical to the prior per-file return contract). Empty
        dict (with a logged warning) if no files were found.

    Raises:
        RuntimeError: on any bulk-upload failure or incomplete commit.

    Example:
        >>> upload_raw_completions_to_data_repo(
        ...     experiment_name="issue354_eos_masked",
        ...     eval_results_dir=Path("eval_results/issue354_eos_masked"),
        ... )
        {'pair2_librarian_swe/T_seed42/raw_completions.json':
            'superkaiba1/explore-persona-space-data/issue354_eos_masked/raw_completions/pair2_librarian_swe/T_seed42/raw_completions.json',
         'pair2_librarian_swe/C_seed42/raw_completions.json':
            'superkaiba1/explore-persona-space-data/issue354_eos_masked/raw_completions/pair2_librarian_swe/C_seed42/raw_completions.json'}
    """
    raw_paths = sorted(eval_results_dir.rglob("raw_completions.json"))
    if not raw_paths:
        logger.warning(
            "upload_raw_completions_to_data_repo: no raw_completions.json "
            "files found under %s — nothing to upload",
            eval_results_dir,
        )
        return {}

    path_in_repo = f"{experiment_name}/raw_completions"
    # Map each local file to (rel, expected committed repo path). The committed
    # path mirrors the prior per-file layout exactly:
    # <experiment_name>/raw_completions/<rel-to-eval_results_dir>.
    rels = [raw_path.relative_to(eval_results_dir).as_posix() for raw_path in raw_paths]
    expected_repo_paths = [f"{path_in_repo}/{rel}" for rel in rels]

    # ONE folder commit for the whole tree — no per-file recursive pre-check.
    # The allow_patterns set captures raw_completions.json at EVERY depth: the
    # leading bare pattern matches a top-level file (no subdir), the ``**/``
    # pattern matches every nested file.
    base_url = _upload_folder_filtered(
        local_dir=eval_results_dir,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        allow_patterns=["raw_completions.json", "**/raw_completions.json"],
        expected_repo_paths=expected_repo_paths,
        delete_after=False,  # caller deletes below, AFTER the set-verify succeeds
    )
    if not base_url:
        raise RuntimeError(
            "upload_raw_completions_to_data_repo: bulk folder upload failed for "
            f"{eval_results_dir} → {DEFAULT_DATASET_REPO}/{path_in_repo}"
        )

    uploaded = {rel: f"{DEFAULT_DATASET_REPO}/{path_in_repo}/{rel}" for rel in rels}

    if delete_after:
        # Verified above (the EXACT-set check inside _upload_folder_filtered),
        # so deleting the individual files now cannot strand an unverified one.
        for raw_path in raw_paths:
            raw_path.unlink(missing_ok=True)

    return uploaded


def download_dataset(
    path_in_repo: str,
    local_path: str,
    repo_id: str = DEFAULT_DATASET_REPO,
) -> str:
    """Download a dataset file from HF Hub to a local path.

    Args:
        path_in_repo: Path within the dataset repo (e.g. 'leakage/marker_evil.jsonl').
        local_path: Local file path to save to.
        repo_id: HF Hub dataset repo ID.

    Returns:
        Local path of the downloaded file, or empty string on failure.
    """
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")

    try:
        # A xet-read-token 429 / transient 5xx inside hf_hub_download rides the
        # budgeted retry (#931/#997); the outer fail-soft contract (return ""
        # on final failure) is unchanged.
        downloaded = _retry_upload(
            lambda: hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type="dataset",
                local_dir=str(Path(local_path).parent),
                local_dir_use_symlinks=False,
                token=token,
            ),
            what=f"hf_hub_download({repo_id}/{path_in_repo})",
        )
        # hf_hub_download saves to local_dir/path_in_repo — move to exact local_path
        downloaded = Path(downloaded)
        target = Path(local_path)
        if downloaded != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded.rename(target)
        logger.info("Downloaded: %s -> %s", path_in_repo, local_path)
        return str(target)
    except Exception as e:
        logger.error("Download failed for %s: %s", path_in_repo, e)
        return ""


def list_hub_datasets(
    repo_id: str = DEFAULT_DATASET_REPO,
    path_prefix: str = "",
) -> list[str]:
    """List all files in the HF Hub dataset repo.

    Prefix-shape dispatch (#920/#988): a DIR-LIKE ``path_prefix`` (ends with
    ``/``) routes to a server-side SCOPED tree walk; an empty or BARE-name
    prefix keeps the full listing (see the branch comments below).

    Args:
        repo_id: HF Hub dataset repo ID.
        path_prefix: Filter to files under this prefix (e.g. 'leakage/').
            A bare (non-slash) prefix like 'dpo' is a PARTIAL-NAME match
            that also matches 'dpo_v2/...' — load-bearing for
            scripts/sync_datasets.py.

    Returns:
        List of file paths in the repo.
    """
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")

    try:
        api = HfApi(token=token)
        if path_prefix.endswith("/"):
            # Dir-like prefix: server-side scoped walk (#920/#988). Client-side
            # filter kept for exactness (a no-op against real scoped results).
            files = list_hf_files_under_path(
                api, repo_id, path_prefix.rstrip("/"), repo_type="dataset"
            )
            files = [f for f in files if f.startswith(path_prefix)]
        else:
            # Empty prefix = the function's list-everything contract; a bare
            # (non-slash) prefix is a PARTIAL-NAME match ("dpo" must also match
            # dpo_v2/...) that no server-side scope can express. DELIBERATE
            # full listing — bounded use only; on the ~1M-file data repo this
            # is the #920 hang class, so prefer a dir-like prefix wherever the
            # caller can.
            files = list_repo_files_complete(api, repo_id, repo_type="dataset")
            if path_prefix:
                files = [f for f in files if f.startswith(path_prefix)]
        return sorted(files)
    except Exception as e:
        logger.error("Failed to list datasets: %s", e)
        return []


# ── Carry-over artifact existence verification (pre-launch gate) ──────────────

# huggingface.co/<repo_id>[/tree|/blob/<revision>][/<path>] and hf:// forms.
# repo_id is captured as <owner>/<name> with an optional datasets/ prefix.
# Revision/path captures terminate at whitespace and at URL-adjacent
# punctuation — ) ] " ' ` , ; } > \ — so a URL cited inside a JSON blob
# ("...",) or a markdown backtick span (`...`) never drags the trailing
# quote/comma/backtick into the probed revision/path (incident #541; mirrors
# scripts/verify_uploads.py's _TRAILING_PUNCT, commit 9987a70dc). '.' stays
# allowed so real suffixes like '.json' / '.safetensors' survive.
_REV_CHARS = r"""[^/\s)\]"'`,;}>\\]"""  # revision segment: also stops at '/'
_PATH_CHARS = r"""[^\s)\]"'`,;}>\\]"""  # path chars: '/' handled by the group

_HF_URL_RE = re.compile(
    rf"""
    (?:
        https?://huggingface\.co/         # web URL form
        (?P<webkind>datasets/|spaces/)?
        (?P<webrepo>[\w.\-]+/[\w.\-]+)
        (?:/(?:tree|blob|resolve)/(?P<webrev>{_REV_CHARS}+)(?P<webpath>(?:/{_PATH_CHARS}+)*))?
      |
        hf://                             # hf:// URI form
        (?P<urikind>datasets/|spaces/)?
        (?P<urirepo>[\w.\-]+/[\w.\-]+)
        (?:@(?P<urirev>{_REV_CHARS}+))?
        (?P<uripath>(?:/{_PATH_CHARS}+)*)?
    )
    """,
    re.VERBOSE,
)

# wandb.ai/<entity>/<project>/runs/<run_id>[/...] — the positive [\w.\-]
# classes already exclude the JSON/markdown punctuation handled above, so no
# trailing-punctuation guard is needed here.
_WANDB_URL_RE = re.compile(
    r"https?://(?:www\.)?wandb\.ai/(?P<entity>[\w.\-]+)/(?P<project>[\w.\-]+)/runs/(?P<run_id>[\w.\-]+)"
)


def _kind_to_repo_type(kind: str | None) -> str:
    """Map a huggingface.co URL path prefix to an HfApi ``repo_type``."""
    if kind == "datasets/":
        return "dataset"
    if kind == "spaces/":
        return "space"
    return "model"


def _hf_artifact_exists(api, repo_id: str, repo_type: str, revision: str | None, path: str) -> bool:
    """Check whether a specific HF repo (and optional in-repo path) resolves.

    Scoped probe (#920/#988): the cited path is checked via ONE server-side
    scoped tree walk (dir paths) with an exact-file ``file_exists`` fallback
    (blob paths) — never a full-repo listing, which wedges >600 s on the
    ~1M-file data repo. A repo-root URL (empty ``path``) is proven by one
    cheap ``repo_info`` call (retry-wrapped for transport 5xx parity with the
    old retried listing).

    A reachable repo whose tree is missing the cited ``path`` is a normal
    ``False`` — NOT an exception. Genuine transport / auth errors propagate so
    the caller fails loud rather than reporting a real artifact as missing.
    """
    if not path:
        # URL points at the repo root — repo (+revision) resolving is enough.
        # ONE cheap repo_info call, NOT a full listing (#920 hang class).
        _retry_upload(
            lambda: api.repo_info(repo_id, repo_type=repo_type, revision=revision),
            what=f"repo_info({repo_id})",
        )
        return True
    return bool(
        list_hf_files_under_path(api, repo_id, path, repo_type=repo_type, revision=revision)
    )


def _wandb_run_exists(entity: str, project: str, run_id: str) -> bool:
    """Return True iff the WandB run resolves via the public API.

    A 404 / "could not find run" is a normal ``False``. Auth / connection
    failures propagate so a transient outage is not misread as "missing".
    """
    import wandb

    api = wandb.Api()
    try:
        api.run(f"{entity}/{project}/{run_id}")
        return True
    except wandb.errors.CommError as e:
        # CommError covers both "run not found" (404) and transport failures.
        # Only the not-found case is a legitimate (False) — re-raise the rest.
        msg = str(e).lower()
        if "could not find" in msg or "404" in msg or "not found" in msg:
            return False
        raise


def verify_artifacts_exist(plan_path: str | Path) -> tuple[bool, list[str]]:
    """Scan a cached plan for carry-over artifact URLs and check each resolves.

    Consumed PRE-LAUNCH by ``.claude/skills/issue/SKILL.md`` Step 6a.5 to block
    provisioning a pod when a plan cites a carry-over artifact (a prior run's
    checkpoint, dataset, or WandB run) that does not exist — provisioning only
    to die seconds in on a 404 is pure wasted GPU-minutes.

    Scans the plan text for:
      - HF repo URLs (``https://huggingface.co/...`` and ``hf://...`` forms),
        including optional ``/tree|/blob|/resolve/<revision>/<path>`` and
        ``@<revision>`` revisions and in-repo paths.
      - WandB run URLs (``https://wandb.ai/<entity>/<project>/runs/<run_id>``).

    Each URL is existence-checked against the Hub (paginated tree walk, so a
    large repo never spuriously reports a present file as missing) or the WandB
    public API. HF auth uses the ambient ``HF_TOKEN``; WandB uses
    ``WANDB_API_KEY`` via the public API's normal credential resolution.

    Fail-loud contract:
      - A malformed / missing / non-file ``plan_path`` raises ``ValueError``
        (the caller passed something that can't be a plan).
      - A reachable-but-missing artifact is a NORMAL ``(False, [...])`` return,
        not an exception.
      - Genuine transport / auth errors propagate (the helper does not swallow
        them and report a real artifact as missing).

    Args:
        plan_path: Path to the cached plan markdown file.

    Returns:
        ``(all_exist, missing_urls)``. ``all_exist`` is True iff every detected
        URL resolved; ``missing_urls`` is the de-duplicated list of URLs that
        did not (empty when ``all_exist`` is True). A plan citing no artifact
        URLs returns ``(True, [])``.

    Raises:
        ValueError: ``plan_path`` is empty, does not exist, or is not a file.
    """
    if plan_path is None or str(plan_path).strip() == "":
        raise ValueError("verify_artifacts_exist: plan_path is empty")
    plan_path = Path(plan_path)
    if not plan_path.exists():
        raise ValueError(f"verify_artifacts_exist: plan_path does not exist: {plan_path}")
    if not plan_path.is_file():
        raise ValueError(f"verify_artifacts_exist: plan_path is not a file: {plan_path}")

    text = plan_path.read_text(encoding="utf-8")

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    missing: list[str] = []
    seen: set[str] = set()

    for m in _HF_URL_RE.finditer(text):
        url = m.group(0)
        if url in seen:
            continue
        seen.add(url)
        kind = m.group("webkind") or m.group("urikind")
        repo_id = m.group("webrepo") or m.group("urirepo")
        revision = m.group("webrev") or m.group("urirev")
        path = m.group("webpath") or m.group("uripath") or ""
        repo_type = _kind_to_repo_type(kind)
        if not _hf_artifact_exists(api, repo_id, repo_type, revision, path):
            missing.append(url)

    for m in _WANDB_URL_RE.finditer(text):
        url = m.group(0)
        if url in seen:
            continue
        seen.add(url)
        if not _wandb_run_exists(m.group("entity"), m.group("project"), m.group("run_id")):
            missing.append(url)

    return (len(missing) == 0, missing)


def upload_model_wandb(
    model_path: str,
    project: str,
    name: str,
    metadata: dict | None = None,
    delete_after: bool = False,
) -> str:
    """Upload a model as a WandB Artifact.

    Args:
        model_path: Local path to the merged model directory.
        project: WandB project name.
        name: Artifact name (e.g. 'midtrain_evil_wrong_em_seed42').
        metadata: Optional metadata dict to attach.
        delete_after: Delete local model after verified upload. Default False
            for safety — caller must explicitly opt in.

    Returns:
        The artifact reference string, or empty string on failure.
    """
    import wandb

    model_path = Path(model_path)
    if not model_path.exists():
        logger.warning("Model path %s does not exist, skipping upload", model_path)
        return ""

    # Upload Policy (CLAUDE.md): WandB carries LIVE training metrics ONLY; model
    # weights are canonical on the HF Hub. Pushing checkpoints to WandB Artifacts
    # duplicated what already lives on HF and filled the account to ~4 TB, so this
    # is OFF by default. Opt in explicitly with EPM_UPLOAD_MODEL_WANDB=1.
    if os.environ.get("EPM_UPLOAD_MODEL_WANDB") != "1":
        logger.info(
            "WandB model-artifact upload disabled by Upload Policy (weights are "
            "canonical on HF Hub; set EPM_UPLOAD_MODEL_WANDB=1 to override). "
            "Skipped %s -> %s.",
            model_path,
            name,
        )
        return ""

    try:
        # Use current run if active, otherwise init a new one
        run = wandb.run
        if run is None:
            run = wandb.init(project=project, job_type="upload")

        artifact = wandb.Artifact(name=name, type="model", metadata=metadata or {})
        artifact.add_dir(str(model_path))
        run.log_artifact(artifact)
        artifact.wait()

        ref = f"wandb://{project}/{name}:latest"
        logger.info("Upload complete: %s", ref)

        if delete_after:
            shutil.rmtree(str(model_path), ignore_errors=True)
            logger.info("Deleted local model: %s", model_path)

        return ref
    except Exception as e:
        logger.error("WandB upload failed: %s. Keeping local model.", e)
        return ""


def upload_results_wandb(
    results_dir: str,
    project: str,
    name: str,
    metadata: dict | None = None,
) -> str:
    """Upload eval results directory as a WandB Artifact.

    Uploads all JSON files, figures, and other eval outputs to WandB so the
    manager can pull results from the cloud without SSH.

    Args:
        results_dir: Local path to the eval results directory for this run.
        project: WandB project name.
        name: Artifact name (e.g. 'results_evil_wrong_em_seed42').
        metadata: Optional metadata dict to attach.

    Returns:
        The artifact reference string, or empty string on failure.
    """
    import wandb

    results_dir = Path(results_dir)
    if not results_dir.exists():
        logger.warning("Results dir %s does not exist, skipping upload", results_dir)
        return ""

    # Check there are actually files to upload
    files = list(results_dir.rglob("*"))
    if not any(f.is_file() for f in files):
        logger.warning("Results dir %s is empty, skipping upload", results_dir)
        return ""

    # Upload Policy (CLAUDE.md): eval results are canonical in git (eval_results/)
    # + the HF data repo; WandB carries LIVE training metrics ONLY. This path also
    # used to add_dir() the whole results dir, sweeping in any merged checkpoint
    # left alongside the JSONs (the ~1.3 TB "eval-results with weights" bloat), so
    # it is OFF by default. Opt in explicitly with EPM_UPLOAD_RESULTS_WANDB=1.
    if os.environ.get("EPM_UPLOAD_RESULTS_WANDB") != "1":
        logger.info(
            "WandB eval-results artifact upload disabled by Upload Policy (results "
            "are canonical in git + the HF data repo; set EPM_UPLOAD_RESULTS_WANDB=1 "
            "to override). Skipped %s -> %s.",
            results_dir,
            name,
        )
        return ""

    try:
        run = wandb.run
        if run is None:
            run = wandb.init(project=project, job_type="eval-upload")

        # Defense in depth even when explicitly opted in: never let model weights
        # ride the eval-results artifact path (the root cause of the eval-results
        # bloat). Add files individually, excluding weight blobs.
        _WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx")
        artifact = wandb.Artifact(
            name=name,
            type="eval-results",
            metadata=metadata or {},
        )
        for f in files:
            if f.is_file() and f.suffix.lower() not in _WEIGHT_SUFFIXES:
                artifact.add_file(str(f), name=str(f.relative_to(results_dir)))
        run.log_artifact(artifact)
        artifact.wait()

        ref = f"wandb://{project}/{name}:latest"
        logger.info("Results uploaded: %s", ref)
        return ref
    except Exception as e:
        logger.error("WandB results upload failed: %s", e)
        return ""


def cleanup_hf_cache():
    """Remove downloaded model blobs from HF cache to free disk space.

    Deletes the blobs/ directory inside each cached model, which contains
    the large safetensors files. The refs/ and snapshots/ metadata are kept
    so HF knows the files existed (and will re-download if needed).
    """
    hf_home_env = os.environ.get("HF_HOME")
    hf_home = Path(hf_home_env) if hf_home_env else (Path.home() / ".cache" / "huggingface")
    cache_dir = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub")))

    if not cache_dir.exists():
        return

    freed = 0
    for model_dir in cache_dir.glob("models--*"):
        blobs_dir = model_dir / "blobs"
        if blobs_dir.exists():
            size = sum(f.stat().st_size for f in blobs_dir.rglob("*") if f.is_file())
            shutil.rmtree(str(blobs_dir), ignore_errors=True)
            freed += size

    if freed > 0:
        logger.info("Cleaned HF cache: freed %.1f GB", freed / 1e9)
