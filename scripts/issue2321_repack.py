#!/usr/bin/env python
"""#2321 repack driver — pack the 10 largest HF data-repo prefixes, probe-first.

Repacks many-small-file prefixes of ``superkaiba1/explore-persona-space-data``
into v2 line-shards (``orchestrate/packing.py``) and deletes the originals in
net-negative, parent-pinned commit units, recovering ~610k of the repo's
1,000,000-file cap (plan v4 §3).

Phases (``--phase``; per-prefix unless noted):
  walk           census the prefix at a pinned revision R (rich scoped listing:
                 size / is_lfs / lfs_sha256 / blob_id — A6 anchors), build the
                 C14 retained set (transitive manifest closure), apply the §3.3
                 selection, persist ``<work-root>/state/<prefix>.census.json``.
  download       stage every pack candidate to ``<work-root>/stage/<prefix>/``
                 (16 workers, shared ~30 req/s resolver token bucket — C6),
                 per-file integrity vs the census anchors (A6).
  pack           ``packing.pack_tree_v2`` into ``<work-root>/pack/<prefix>/``
                 with C8 anchors re-asserted on the staged bytes.
  verify         unpack EVERY shard via the PRODUCTION decoder (C19) into
                 ``<work-root>/scratch/<prefix>/``, sha256-compare 100% of
                 members vs the staged originals, assert the census<->member
                 bijection (C3/I12) with a named delta report.
  consumer-gate  refuse (rc=22) while any silent-empty consumer scoped to the
                 prefix is unmigrated (I17; reads the committed inventory).
  commit         compose net-negative units (ops<=4,500 incl. deletes; shard
                 bytes<=225MB; C7 sparse-tail rebalance), then land each via
                 the probe-first loop (I11/I13/I14) with the I15 journal +
                 I16 cumulative INDEX riding EVERY data commit; >=20s between
                 commits; finalize with pack_manifest.json (+INDEX refresh).
  postverify     exact-set + content-equality verify of every landed artifact
                 (I13(c), C20 tolerance for pre-existing non-v2 entries),
                 sources-gone + non-LFS shard sweep (C16), THEN reap local
                 staging (I10/C10).
  remeasure      scoped per-prefix file counts (+ ``--full-walk`` repo total).
  cap-probe      the §3.6 three-commit at-cap semantics probe (C4 recompute,
                 C17 net-negative-before-invalidation).

Exit codes: 21 StopRepack (I2 file-count rejection => global stop) · 22
consumer gate blocked (I17) · 23 AbortPrefix (drift/bijection/content
mismatch; prefix left byte-consistent, state "packed-unindexed-final", C12) ·
24 RateLimitedStop (C18).

Safety: every canonical-repo mutation funnels through
``commit_unit_probe_first`` which (a) asserts the I18 test-mutation interlock
BEFORE any network access, (b) asserts ``EPM_HF_OVERFLOW_ROUTING != "1"``
(I8), (c) probes Hub state content-anchored before EVERY issue attempt (I13b)
and pins ``parent_commit`` to the probe head (I14), and (d) never wraps
``create_commit`` in ``retry_transient`` (I11 — ambiguous outcomes re-probe
instead of blind-retry). ``--dry-run`` gates every mutation; ``--smoke`` runs
the fixture-tree pack/verify/compose chain with ZERO network.

Smoke blind-spot enumeration (plan §5): the ``--smoke`` fixture chain +
``--dry-run`` composition do NOT certify (a) live ``create_commit``
acceptance, (b) live resolver throughput, (c) the cap semantics (that is the
cap-probe's job), (d) live timeout disposition, (e) live 412 semantics,
(f) the consumer-gate dynamic-path blind spot. No smoke-conditional branch
substitutes an implementation or downgrades an assertion — smoke reduces
SCALE (fixture tree) only.

MF3 resume after total local wipe: landed members live inside the landed
shards, so ``hub.stage_hub_file``'s packed fallback re-materializes deleted
originals transparently during the download phase; the census is
reconstructed as (packed members under the prefix) UNION (surviving raw
files) — :func:`reconstruct_prefix_census`.

VM-side phases are CPU-only and pod-routed (plan §9: ``pod.py provision
--issue 2321 --intent cpu-mid``); this driver never runs the destructive
phases on the shared VM.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import fnmatch
import hashlib
import json
import os
import posixpath
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

# Env pins BEFORE any huggingface_hub import (constants freeze env at import;
# plan §3.4 download phase: many-small-file storms wedge the xet path —
# .claude/rules/gotchas.md "HF Hub download-accelerator FAILURE MATRIX").
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from explore_persona_space.orchestrate import packing  # noqa: E402  (stdlib-only module)

# ---------------------------------------------------------------------------
# Constants (plan §3.4/§3.5/§3.6)
# ---------------------------------------------------------------------------

DEFAULT_REPO_ID = "superkaiba1/explore-persona-space-data"
HF_FILE_CAP = 1_000_000
PROBE_PREFIX = "issue2321_probe"

#: §3.6 execution order — issue1090_partial FIRST (its unit 1 is the real-unit
#: probe fallback), issue1739_ctxmap deliberately LATE (47 consumers),
#: issue667_alllayer (tier B, .npz b64) LAST.
PREFIX_ORDER = (
    "issue1090_partial",
    "issue1434_writingstyle",
    "issue1090_pvdatagen",
    "issue1481_conpos_grid",
    "issue1586_methodgen",
    "issue2224_screening",
    "issue1739_partial",
    "issue1739_ctxmap",
    "issue1489_ctx_aug",
    "issue667_alllayer",
)
TIER_B_PREFIXES = frozenset({"issue667_alllayer"})
TIER_B_SUFFIXES = (".npz",)

UNIT_OPS_CAP = 4_500  # total operations per commit unit: adds + deletes (plan §3.4)
UNIT_SHARD_BYTES_CAP = 225_000_000  # sum of shard bytes per unit
COMMIT_SLEEP_S = float(os.environ.get("EPM_I2321_COMMIT_SLEEP_S", "20"))  # <=180 commits/hr
DL_WORKERS = int(os.environ.get("EPM_I2321_DL_WORKERS", "16"))
RESOLVER_RPS = float(os.environ.get("EPM_I2321_RESOLVER_RPS", "30"))  # C6 shared bucket
DEFAULT_WORK_ROOT = Path(os.environ.get("EPM_I2321_ROOT", "/root/i2321"))

#: Cumulative per-unit files replaced by design in every data commit (I15/I16).
REPLACED_BY_DESIGN = (packing.INDEX_NAME, packing.UNITS_JOURNAL_NAME)

RC_STOP_REPACK = 21
RC_CONSUMER_GATE = 22
RC_ABORT_PREFIX = 23
RC_RATE_LIMITED = 24

_PROBE_A_BYTES = b"#2321 cap probe A\n"
_PROBE_B_BYTES = b"#2321 cap probe B\n"


# ---------------------------------------------------------------------------
# Typed terminals (exit-code map)
# ---------------------------------------------------------------------------


class StopRepack(RuntimeError):
    """I2: the Hub rejected a commit on the FILE-COUNT cap => global stop."""

    rc = RC_STOP_REPACK


class ConsumerGateBlocked(RuntimeError):
    """I17: an unmigrated silent-empty consumer is scoped to this prefix."""

    rc = RC_CONSUMER_GATE


class AbortPrefix(RuntimeError):
    """This prefix aborts (drift / bijection / content mismatch); repo stays consistent."""

    rc = RC_ABORT_PREFIX


class RateLimitedStop(RuntimeError):
    """C18: the 429 budget exhausted — a rate condition, never 'attempts-exhausted'."""

    rc = RC_RATE_LIMITED


# ---------------------------------------------------------------------------
# Census anchors (A6)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Anchor:
    """Content anchor for one census file (from the revision-R rich listing)."""

    size: int | None
    is_lfs: bool
    blob_id: str | None
    lfs_sha256: str | None


def entry_anchor(entry) -> Anchor:
    """Anchor from a ``hub.RepoFileEntry``."""
    return Anchor(
        size=entry.size,
        is_lfs=bool(entry.is_lfs),
        blob_id=entry.blob_id,
        lfs_sha256=entry.lfs_sha256,
    )


def anchor_matches(entry, anchor: Anchor) -> bool:
    """True when a live listing entry matches the census anchor (size + digest)."""
    if anchor.size is not None and entry.size != anchor.size:
        return False
    if bool(entry.is_lfs) != anchor.is_lfs:
        return False
    if anchor.is_lfs:
        return entry.lfs_sha256 == anchor.lfs_sha256
    return entry.blob_id == anchor.blob_id


def _assert_integrity(data: bytes, anchor: Anchor, path: str) -> None:
    """Per-file download integrity (A6): git-blob sha1 (non-LFS) / sha256 (LFS)."""
    if anchor.size is not None and len(data) != anchor.size:
        raise AbortPrefix(f"integrity: {path}: size {len(data)} != census {anchor.size}")
    if anchor.is_lfs:
        got = hashlib.sha256(data).hexdigest()
        if got != anchor.lfs_sha256:
            raise AbortPrefix(f"integrity: {path}: sha256 {got[:12]} != lfs {anchor.lfs_sha256}")
    else:
        got = packing.git_blob_sha1(data)
        if got != anchor.blob_id:
            raise AbortPrefix(f"integrity: {path}: git-blob {got[:12]} != census {anchor.blob_id}")


# ---------------------------------------------------------------------------
# Error classifiers (plan §3.4 commit loop)
# ---------------------------------------------------------------------------


def _status_code(err: BaseException) -> int | None:
    """HTTP status of an exception's response, when it carries one."""
    resp = getattr(err, "response", None)
    return getattr(resp, "status_code", None)


def is_parent_conflict(err: BaseException) -> bool:
    """HTTP 412 parent-pin conflict — DEFINITIVE (never ambiguous; MF2d)."""
    return _status_code(err) == 412


def is_rate_limit(err: BaseException) -> bool:
    """HTTP 429 / rate-limit — its OWN budget + outcome (C18)."""
    if _status_code(err) == 429:
        return True
    msg = str(err).lower()
    return "too many requests" in msg or "rate limit" in msg


def retry_after_seconds(err: BaseException, *, floor: float = 30.0) -> float:
    """Server Retry-After hint, floored (429 storms replenish per minute)."""
    resp = getattr(err, "response", None)
    headers = getattr(resp, "headers", None) or {}
    try:
        hinted = float(headers.get("Retry-After", "") or 0.0)
    except (TypeError, ValueError):
        hinted = 0.0
    return max(hinted, floor)


def is_ambiguous_outcome(err: BaseException) -> bool:
    """Client timeout / connection drop / gateway 5xx: the commit MAY have landed.

    EXCLUDES 412 + 429 by construction (each has its own budget + outcome).
    Only these consume the I11 3-attempt budget; anything else re-raises.
    """
    if is_parent_conflict(err) or is_rate_limit(err):
        return False
    status = _status_code(err)
    if status is not None and 500 <= status < 600:
        return True
    if isinstance(err, TimeoutError | ConnectionError):
        return True
    msg = str(err).lower()
    return any(t in msg for t in ("timeout", "timed out", "connection", "gateway"))


def _is_file_count_limit(err: BaseException) -> bool:
    """I2: the server's file-count-cap rejection (delegates to the hub matcher)."""
    from explore_persona_space.orchestrate import hub

    return hub._is_file_count_limit_error(err)


# ---------------------------------------------------------------------------
# Census walk + C14 retained set + §3.3 selection
# ---------------------------------------------------------------------------


def walk_prefix(api, *, repo_id: str, prefix: str, revision: str) -> list:
    """Rich scoped listing of the prefix at the pinned revision R."""
    from explore_persona_space.orchestrate import hub

    entries = hub.list_repo_repofiles_under_path(
        api, repo_id, prefix, repo_type="dataset", revision=revision
    )
    for e in entries:
        if not e.path.startswith(prefix + "/"):
            raise AbortPrefix(f"walk: listing leaked outside prefix: {e.path}")
    return entries


def parts_from_pack_manifest(doc: Mapping) -> set[str]:
    """Part names (manifest-dir-relative) enumerated by a pack manifest.

    v2 (``orchestrate/packing.py``): ``shards`` mapping keys + every group's
    ``index_files``/``shard_files``, plus the beside-it cumulative
    ``INDEX.json``/``units.jsonl``. v1 (``scripts/issue1739_pack.py``):
    ``groups.<key>.shards[].name``. Unrecognized structure fails loud (C14 —
    never silently retain nothing).
    """
    version = doc.get("version")
    if version == packing.PACK_FORMAT_VERSION:
        names: set[str] = set(doc.get("shards", {}).keys())
        for group in (doc.get("groups") or {}).values():
            names.update(group.get("index_files") or [])
            names.update(group.get("shard_files") or [])
        names.update(REPLACED_BY_DESIGN)
        return names
    groups = doc.get("groups")
    if isinstance(groups, dict):
        names = set()
        for group in groups.values():
            for shard in group.get("shards") or []:
                if not isinstance(shard, Mapping) or "name" not in shard:
                    raise ValueError(f"unrecognized v1 pack-manifest shard entry: {shard!r}")
                names.add(shard["name"])
        return names
    raise ValueError(f"unrecognized pack manifest structure (version={version!r})")


def parts_from_sharded_text_manifest(doc: Mapping) -> set[str]:
    """Part names from a #2119 ``<stem>.manifest.json`` (``{"parts": [...]}``)."""
    parts = doc.get("parts")
    if not isinstance(parts, list) or not all(isinstance(p, str) for p in parts):
        raise ValueError(f"unrecognized sharded-text manifest: parts={type(parts).__name__}")
    return set(parts)


_ORPHAN_PART_PATTERNS = ("*.shard*.jsonl", "*.index*.json")
_ORPHAN_BASENAMES = frozenset(
    {packing.INDEX_NAME, packing.UNITS_JOURNAL_NAME, packing.MANIFEST_NAME}
)


def build_retained_set(
    entries: Iterable, *, prefix: str, fetch_text: Callable[[str], str]
) -> dict[str, str]:
    """C14 retained set: TRANSITIVE closure over every recognized manifest.

    Returns ``{repo path: reason}``. Retains each manifest + every part/shard
    it enumerates; ``OVERFLOW_POINTER.json`` breadcrumbs; and (belt and
    braces) orphaned part-shaped names no manifest claimed.
    """
    retained: dict[str, str] = {}
    entry_list = list(entries)
    for e in entry_list:
        base = posixpath.basename(e.path)
        dirp = posixpath.dirname(e.path)
        if base == packing.MANIFEST_NAME:
            retained[e.path] = "pack-manifest"
            doc = json.loads(fetch_text(e.path))
            for name in parts_from_pack_manifest(doc):
                retained.setdefault(posixpath.join(dirp, name), f"pack-part:{e.path}")
        elif base.endswith(".manifest.json"):
            retained[e.path] = "sharded-text-manifest"
            doc = json.loads(fetch_text(e.path))
            for name in parts_from_sharded_text_manifest(doc):
                retained.setdefault(posixpath.join(dirp, name), f"shard-part:{e.path}")
        elif base == "OVERFLOW_POINTER.json":
            retained[e.path] = "overflow-pointer"
    for e in entry_list:
        if e.path in retained:
            continue
        base = posixpath.basename(e.path)
        if base in _ORPHAN_BASENAMES or any(
            fnmatch.fnmatch(base, pat) for pat in _ORPHAN_PART_PATTERNS
        ):
            retained[e.path] = "orphan-name-shape"
    for path in retained:
        if not path.startswith(prefix + "/"):
            raise AbortPrefix(f"C14: retained path escapes prefix: {path}")
    return retained


def select_pack_candidates(
    entries: Iterable, retained: Mapping[str, str], *, prefix: str
) -> tuple[list, dict[str, int]]:
    """§3.3 selection (a)-(d); returns (candidates, per-reason exclusion counts)."""
    tier_b = prefix in TIER_B_PREFIXES
    candidates = []
    exclusions: dict[str, int] = {}

    def _exclude(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for e in entries:
        if not e.path.startswith(prefix + "/"):
            raise AbortPrefix(f"selection: entry escapes prefix: {e.path}")
        if e.path.startswith(f"{prefix}/{packing.PACKED_DIRNAME}/"):
            _exclude("under-packed-dir")
            continue
        if e.path in retained:
            _exclude("retained")
            continue
        if e.is_lfs and not (tier_b and e.path.endswith(TIER_B_SUFFIXES)):
            _exclude("lfs-not-tier-b")
            continue
        est = packing.estimate_encoded_line_bytes(e.size, posixpath.basename(e.path))
        if est > packing.SHARD_MAX_BYTES:
            _exclude("oversize-encoded-line")
            continue
        candidates.append(e)
    return candidates, exclusions


# ---------------------------------------------------------------------------
# Download phase (C6 token bucket + A6 integrity)
# ---------------------------------------------------------------------------


class TokenBucket:
    """Shared resolver-rate limiter (~RESOLVER_RPS req/s across all workers)."""

    def __init__(
        self,
        rate_per_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_s <= 0:
            raise ValueError(f"rate_per_s must be positive, got {rate_per_s}")
        self._interval = 1.0 / rate_per_s
        self._lock = threading.Lock()
        self._next: float | None = None
        self._clock = clock
        self._sleep = sleep

    def acquire(self, n: int = 1) -> None:
        """Block until n resolver requests fit the rate budget."""
        with self._lock:
            now = self._clock()
            if self._next is None or self._next < now:
                self._next = now
            wait = self._next - now
            self._next += n * self._interval
        if wait > 0:
            self._sleep(wait)


def download_prefix(
    *,
    repo_id: str,
    prefix: str,
    census: Mapping[str, Anchor],
    candidate_paths: Iterable[str],
    revision: str,
    stage_root: Path,
    workers: int = DL_WORKERS,
    bucket: TokenBucket | None = None,
    progress_every: int = 500,
) -> dict:
    """Stage every candidate under ``stage_root/<repo path>`` with A6 integrity.

    Idempotent: an existing target that passes integrity is skipped. A deleted
    original (MF3 resume) is re-materialized transparently by
    ``stage_hub_file``'s packed fallback.
    """
    from explore_persona_space.orchestrate import hub

    bucket = bucket or TokenBucket(RESOLVER_RPS)
    paths = sorted(candidate_paths)
    n_total = len(paths)
    counts = {"fetched": 0, "cached": 0}
    lock = threading.Lock()
    t0 = time.monotonic()

    def _one(path: str) -> None:
        anchor = census[path]
        target = stage_root / path
        if target.is_file():
            data = target.read_bytes()
            try:
                _assert_integrity(data, anchor, path)
            except AbortPrefix:
                target.unlink()
            else:
                with lock:
                    counts["cached"] += 1
                return
        bucket.acquire(2)  # HEAD + GET resolver requests
        hub.stage_hub_file(
            repo_id,
            path,
            target,
            repo_type="dataset",
            revision=revision,
            overwrite=True,
            size_bytes=anchor.size,
        )
        _assert_integrity(target.read_bytes(), anchor, path)
        with lock:
            counts["fetched"] += 1
            done = counts["fetched"] + counts["cached"]
        if done % progress_every == 0 or done == n_total:
            print(
                f"[download] {prefix} {done}/{n_total} elapsed={time.monotonic() - t0:.0f}s",
                flush=True,
            )

    print(f"[download] {prefix} 0/{n_total} starting ({workers} workers)", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(_one, p) for p in paths]):
            fut.result()  # fail loud on the first integrity/transport failure
    return {"n_total": n_total, **counts}


# ---------------------------------------------------------------------------
# Pack + verify phases (C8 anchors, C19 production decoder, C3/I12 bijection)
# ---------------------------------------------------------------------------


def pack_prefix(
    *,
    prefix: str,
    census: Mapping[str, Anchor],
    candidate_paths: Iterable[str],
    stage_root: Path,
    pack_dir: Path,
    source_revision: str,
    git_commit: str,
) -> packing.PackResult:
    """Pack the staged candidates with C8 anchors re-asserted on packed bytes."""
    rels = []
    anchors: dict[str, tuple[str, str]] = {}
    for path in sorted(candidate_paths):
        rel = path[len(prefix) + 1 :]
        rels.append(rel)
        anchor = census[path]
        if anchor.is_lfs:
            anchors[rel] = ("sha256", anchor.lfs_sha256 or "")
        else:
            anchors[rel] = ("gitblob", anchor.blob_id or "")
    return packing.pack_tree_v2(
        stage_root / prefix,
        pack_dir,
        candidates=rels,
        anchors=anchors,
        source_revision=source_revision,
        git_commit=git_commit,
    )


def verify_prefix(
    *,
    prefix: str,
    pack_dir: Path,
    stage_root: Path,
    scratch_dir: Path,
    candidate_paths: Iterable[str],
) -> dict:
    """C19 production-decoder round-trip + C3/I12 census<->member bijection.

    100% of members must sha256-match the staged originals or the prefix
    aborts; the bijection must be UNIQUE and EXACT, with a named delta report
    on failure.
    """
    n_unpacked = packing.unpack_shards_v2(pack_dir, scratch_dir)
    shards, _groups, _man = load_shard_infos(pack_dir)
    members: list[str] = [m for s in shards for m in s.members]
    census_rels = sorted(p[len(prefix) + 1 :] for p in candidate_paths)
    member_set = set(members)
    census_set = set(census_rels)
    delta = {
        "missing_from_pack": sorted(census_set - member_set)[:50],
        "extra_in_pack": sorted(member_set - census_set)[:50],
        "duplicate_members": len(members) - len(member_set),
    }
    if len(members) != len(member_set) or member_set != census_set:
        raise AbortPrefix(f"C3/I12 bijection failure on {prefix}: {json.dumps(delta)}")
    mismatched = []
    for rel in census_rels:
        got = hashlib.sha256((scratch_dir / rel).read_bytes()).hexdigest()
        want = hashlib.sha256((stage_root / prefix / rel).read_bytes()).hexdigest()
        if got != want:
            mismatched.append(rel)
    if mismatched:
        raise AbortPrefix(
            f"C19 round-trip sha256 mismatch on {prefix}: {len(mismatched)} members; "
            f"first: {mismatched[:5]}"
        )
    return {"n_members": n_unpacked, "n_shards": len(shards), "bijection": "exact"}


# ---------------------------------------------------------------------------
# Shard / unit model + composer (plan §3.4 commit; C7 rebalance; I9 partition)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ShardInfo:
    """One packed shard's identity + members (prefix-relative, sorted)."""

    name: str
    group: str
    size: int
    sha256: str
    blob_sha1: str
    members: tuple[str, ...]


def load_shard_infos(pack_dir: Path) -> tuple[list[ShardInfo], dict, dict]:
    """Shard identities (with git-blob sha1s) + groups + manifest from a pack dir.

    Verifies each shard's bytes against the manifest sha256 and that the
    index parts' member->shard mapping covers every shard.
    """
    man = json.loads((pack_dir / packing.MANIFEST_NAME).read_text(encoding="utf-8"))
    if man.get("version") != packing.PACK_FORMAT_VERSION:
        raise AbortPrefix(f"not a v2 pack manifest: version={man.get('version')!r}")
    groups = man["groups"]
    members_by_shard: dict[str, list[str]] = {}
    for key in sorted(groups):
        for part_name in groups[key]["index_files"]:
            part = packing.load_index_part(
                (pack_dir / part_name).read_text(encoding="utf-8"), what=part_name
            )
            for src, entry in part.items():
                members_by_shard.setdefault(entry.shard, []).append(src)
    shards: list[ShardInfo] = []
    for key in sorted(groups):
        for name in groups[key]["shard_files"]:
            meta = man["shards"][name]
            blob = (pack_dir / name).read_bytes()
            got = hashlib.sha256(blob).hexdigest()
            if got != meta["sha256"]:
                raise AbortPrefix(f"shard {name}: sha256 {got[:12]} != manifest")
            members = tuple(sorted(members_by_shard.get(name, ())))
            if not members:
                raise AbortPrefix(f"shard {name}: no index-part members (index/manifest drift)")
            shards.append(
                ShardInfo(
                    name=name,
                    group=key,
                    size=meta["bytes"],
                    sha256=meta["sha256"],
                    blob_sha1=packing.git_blob_sha1(blob),
                    members=members,
                )
            )
    n_members = sum(len(s.members) for s in shards)
    if n_members != man["n_members"]:
        raise AbortPrefix(f"index parts cover {n_members} members != manifest {man['n_members']}")
    return shards, groups, man


@dataclasses.dataclass(frozen=True)
class CommitUnit:
    """One net-negative commit unit: shards + first-landing index parts + deletes."""

    unit_id: int  # 1-based, plan order
    prefix: str
    shards: tuple[ShardInfo, ...]
    new_groups: tuple[str, ...]  # groups whose FIRST shard lands in this unit
    index_part_names: tuple[str, ...]
    planned_deletes: frozenset[str]  # FULL repo paths

    @property
    def n_adds(self) -> int:
        """Adds per data commit: shards + index parts + INDEX.json + units.jsonl."""
        return len(self.shards) + len(self.index_part_names) + 2

    @property
    def total_ops(self) -> int:
        return self.n_adds + len(self.planned_deletes)

    @property
    def shard_bytes(self) -> int:
        return sum(s.size for s in self.shards)


def _materialize_units(
    bins: list[list[ShardInfo]],
    groups: Mapping[str, Mapping],
    *,
    prefix: str,
    ops_cap: int,
    bytes_cap: int,
) -> list[CommitUnit]:
    """Bins -> CommitUnits; a group's index parts land with its FIRST shard's unit."""
    started: set[str] = set()
    units: list[CommitUnit] = []
    for i, bin_shards in enumerate(bins, 1):
        new_groups: list[str] = []
        parts: list[str] = []
        for s in bin_shards:
            if s.group not in started:
                started.add(s.group)
                new_groups.append(s.group)
                parts.extend(groups[s.group]["index_files"])
        dels = frozenset(f"{prefix}/{m}" for s in bin_shards for m in s.members)
        unit = CommitUnit(
            unit_id=i,
            prefix=prefix,
            shards=tuple(bin_shards),
            new_groups=tuple(new_groups),
            index_part_names=tuple(parts),
            planned_deletes=dels,
        )
        if unit.total_ops > ops_cap:
            raise AbortPrefix(
                f"unit {i}: {unit.total_ops} ops > cap {ops_cap} (composer bug or "
                f"a single shard too wide)"
            )
        if unit.shard_bytes > bytes_cap:
            raise AbortPrefix(f"unit {i}: {unit.shard_bytes} shard bytes > cap {bytes_cap}")
        units.append(unit)
    return units


def compose_units(
    shards: list[ShardInfo],
    groups: Mapping[str, Mapping],
    *,
    prefix: str,
    ops_cap: int = UNIT_OPS_CAP,
    bytes_cap: int = UNIT_SHARD_BYTES_CAP,
) -> tuple[list[CommitUnit] | None, str | None]:
    """Greedy whole-shard binning under the ops/bytes caps + C7 net-negative rebalance.

    Returns ``(units, None)`` or ``(None, skip_reason)`` for an all-sparse
    prefix (SKIPPED + reported — never a run abort). Asserts the I9 partition:
    units partition the shard set; delete sets are disjoint and cover every
    member exactly once.
    """
    if not shards:
        return None, "no-pack-candidates"
    # Greedy bin in group-major pack order (the load_shard_infos order).
    bins: list[list[ShardInfo]] = []
    started: set[str] = set()
    cur: list[ShardInfo] = []
    cur_new_parts = 0
    cur_dels = 0
    cur_bytes = 0
    for shard in shards:
        part_add = 0 if shard.group in started else len(groups[shard.group]["index_files"])
        trial_adds = (len(cur) + 1) + (cur_new_parts + part_add) + 2
        trial_ops = trial_adds + cur_dels + len(shard.members)
        trial_bytes = cur_bytes + shard.size
        if cur and (trial_ops > ops_cap or trial_bytes > bytes_cap):
            bins.append(cur)
            cur, cur_new_parts, cur_dels, cur_bytes = [], 0, 0, 0
            part_add = 0 if shard.group in started else len(groups[shard.group]["index_files"])
        cur.append(shard)
        if shard.group not in started:
            started.add(shard.group)
            cur_new_parts += part_add
        cur_dels += len(shard.members)
        cur_bytes += shard.size
    if cur:
        bins.append(cur)

    # C7 rebalance: every unit must satisfy len(dels) >= n_adds + 1 (net <= -1).
    def _violations(units: list[CommitUnit]) -> list[int]:
        return [u.unit_id for u in units if len(u.planned_deletes) <= u.n_adds]

    units = _materialize_units(bins, groups, prefix=prefix, ops_cap=ops_cap, bytes_cap=bytes_cap)
    for _pass in range(len(bins) + 1):
        bad = _violations(units)
        if not bad:
            break
        k = bad[-1] - 1  # rebalance from the tail (the sparse-tail shape)
        if k == 0:
            return None, f"all-sparse: unit 1 net-positive ({units[0].total_ops} ops)"
        merged = bins[: k - 1] + [bins[k - 1] + bins[k]] + bins[k + 1 :]
        try:
            units = _materialize_units(
                merged, groups, prefix=prefix, ops_cap=ops_cap, bytes_cap=bytes_cap
            )
            bins = merged
            continue
        except AbortPrefix:
            pass
        # Merge overflows the caps: shift the preceding unit's widest shard in.
        donor = max(bins[k - 1], key=lambda s: len(s.members))
        if len(bins[k - 1]) <= 1:
            return None, f"sparse-tail unresolvable at unit {k + 1} (single-shard donor)"
        shifted = [list(b) for b in bins]
        shifted[k - 1].remove(donor)
        shifted[k].insert(0, donor)
        try:
            units = _materialize_units(
                shifted, groups, prefix=prefix, ops_cap=ops_cap, bytes_cap=bytes_cap
            )
        except AbortPrefix:
            return None, f"sparse-tail unresolvable at unit {k + 1} (caps refuse shift)"
        bins = shifted
    else:
        return None, f"rebalance did not converge: units {_violations(units)} net-positive"

    # I9: units partition the shard set; deletes partition the member set.
    binned = [s.name for b in bins for s in b]
    if sorted(binned) != sorted(s.name for s in shards) or len(binned) != len(shards):
        raise AbortPrefix("I9: units do not partition the shard set")
    all_dels = [p for u in units for p in u.planned_deletes]
    n_members = sum(len(s.members) for s in shards)
    if len(all_dels) != len(set(all_dels)) or len(all_dels) != n_members:
        raise AbortPrefix("I9: unit delete sets do not partition the member set")
    return units, None


# ---------------------------------------------------------------------------
# Cumulative journal + index artifacts (I15 / I16)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def top_index_bytes(groups: Mapping[str, Mapping], started_groups: Iterable[str]) -> bytes:
    """Cumulative ``INDEX.json`` bytes: groups started through this unit, full entries."""
    doc = {
        "version": packing.PACK_FORMAT_VERSION,
        "groups": {k: groups[k] for k in sorted(started_groups)},
    }
    return json.dumps(doc, indent=1, sort_keys=True).encode("utf-8")


def delete_set_digest(planned_deletes: Iterable[str]) -> str:
    """sha256 over the sorted member-source list (the I15 delete-set digest)."""
    return hashlib.sha256("\n".join(sorted(planned_deletes)).encode("utf-8")).hexdigest()


def unit_journal_record(
    unit: CommitUnit,
    *,
    n_units: int,
    census_key: str,
    revision: str,
    driver_git_sha: str,
) -> dict:
    """The I15 journal record appended (cumulatively) inside every data commit."""
    return {
        "unit_id": unit.unit_id,
        "n_units": n_units,
        "prefix": unit.prefix,
        "census_key": census_key,
        "shards": [
            {
                "name": s.name,
                "sha256": s.sha256,
                "blob_sha1": s.blob_sha1,
                "n_members": len(s.members),
            }
            for s in unit.shards
        ],
        "index_parts": list(unit.index_part_names),
        "n_members": sum(len(s.members) for s in unit.shards),
        "delete_set_sha256": delete_set_digest(unit.planned_deletes),
        "revision": revision,
        "driver_git_sha": driver_git_sha,
        "ts": _now_iso(),
    }


def journal_bytes(records: Iterable[Mapping]) -> bytes:
    """Deterministic cumulative ``units.jsonl`` bytes (sort_keys, one line/record)."""
    return b"".join(
        json.dumps(r, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n" for r in records
    )


def load_local_journal(pack_dir: Path) -> list[dict]:
    """Landed-unit records from the local pack dir (contiguous unit_ids asserted)."""
    path = pack_dir / packing.UNITS_JOURNAL_NAME
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    got_ids = [r["unit_id"] for r in records]
    if got_ids != list(range(1, len(records) + 1)):
        raise AbortPrefix(f"local journal not contiguous: unit_ids={got_ids[:10]}")
    return records


def append_local_journal(pack_dir: Path, record: Mapping) -> None:
    """O_APPEND one journal line the moment the unit lands (checkpoint-per-unit)."""
    line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
    with open(pack_dir / packing.UNITS_JOURNAL_NAME, "a", encoding="utf-8") as f:
        f.write(line)


def reconstruct_census_paths(
    *, landed_member_srcs: Iterable[str], surviving_paths: Iterable[str], prefix: str
) -> set[str]:
    """MF3: census after total local wipe = landed shard members UNION survivors."""
    landed = {f"{prefix}/{m}" for m in landed_member_srcs}
    return landed | set(surviving_paths)


def reconstruct_prefix_census(api, *, repo_id: str, prefix: str) -> set[str]:
    """MF3 Hub-side reconstruction: packed members + surviving raw originals."""
    from explore_persona_space.orchestrate import hub

    surviving = [
        e.path
        for e in hub.list_repo_repofiles_under_path(api, repo_id, prefix)
        if not e.path.startswith(f"{prefix}/{packing.PACKED_DIRNAME}/")
    ]
    landed = [m.path for m in hub.packed_members_under_path(api, repo_id, prefix)]
    return reconstruct_census_paths(
        landed_member_srcs=[p[len(prefix) + 1 :] for p in landed],
        surviving_paths=surviving,
        prefix=prefix,
    )


# ---------------------------------------------------------------------------
# Per-unit ops + invariants (I1 / I3 / I4 / I13a)
# ---------------------------------------------------------------------------


def build_unit_ops(
    unit: CommitUnit, *, pack_dir: Path, index_bytes: bytes, journal_bytes_: bytes
) -> list[tuple[str, bytes | Path]]:
    """The add-operations of one data commit (shards + parts + INDEX + journal)."""
    packed = f"{unit.prefix}/{packing.PACKED_DIRNAME}"
    adds: list[tuple[str, bytes | Path]] = [
        (f"{packed}/{s.name}", pack_dir / s.name) for s in unit.shards
    ]
    adds += [(f"{packed}/{p}", pack_dir / p) for p in unit.index_part_names]
    adds.append((f"{packed}/{packing.INDEX_NAME}", index_bytes))
    adds.append((f"{packed}/{packing.UNITS_JOURNAL_NAME}", journal_bytes_))
    return adds


def assert_unit_invariants(
    unit: CommitUnit,
    adds: list[tuple[str, bytes | Path]],
    dels: list[str],
    *,
    retained: Mapping[str, str],
    census: Mapping[str, Anchor],
) -> None:
    """I1 + I3 + I4 per data commit — fail loud BEFORE any ops are composed."""
    prefix_slash = unit.prefix + "/"
    if set(dels) != set(unit.planned_deletes):
        raise AbortPrefix(f"unit {unit.unit_id}: dels != planned_deletes (I1)")
    for p in dels:
        if not p.startswith(prefix_slash):
            raise AbortPrefix(f"unit {unit.unit_id}: delete escapes prefix: {p} (I4)")
        if p not in census:
            raise AbortPrefix(f"unit {unit.unit_id}: delete not in census: {p} (I1)")
    overlap = set(dels) & set(retained)
    if overlap:
        raise AbortPrefix(
            f"unit {unit.unit_id}: {len(overlap)} deletes in the RETAINED set (I4): "
            f"{sorted(overlap)[:5]}"
        )
    if len(dels) <= len(adds):
        raise AbortPrefix(
            f"unit {unit.unit_id}: net non-negative ({len(adds)} adds vs {len(dels)} dels; I3)"
        )
    for path, _payload in adds:
        if not path.startswith(f"{prefix_slash}{packing.PACKED_DIRNAME}/"):
            raise AbortPrefix(f"unit {unit.unit_id}: add outside packed/: {path}")


def _payload_bytes(payload: bytes | Path) -> bytes:
    return payload if isinstance(payload, bytes) else Path(payload).read_bytes()


def landed_overwrite_guard(api, *, repo_id: str, prefix: str, units: Iterable[CommitUnit]) -> None:
    """I13(a): a fresh ``packed/`` listing must not hold DIFFERENT content at any
    path this plan would add (a re-derived census on resume must abort, never
    overwrite). REPLACED_BY_DESIGN (cumulative INDEX/journal) is exempt; a
    SAME-content path is a landed/partial unit the probe will classify.
    """
    from explore_persona_space.orchestrate import hub

    packed = f"{prefix}/{packing.PACKED_DIRNAME}"
    fresh = {e.path: e for e in hub.list_repo_repofiles_under_path(api, repo_id, packed)}
    if not fresh:
        return
    conflicts: list[str] = []
    for unit in units:
        for path, expected_sha1 in unit_expected_strict(unit).items():
            entry = fresh.get(path)
            if entry is not None and entry.blob_id != expected_sha1:
                conflicts.append(path)
    if conflicts:
        raise AbortPrefix(
            f"I13(a): {len(conflicts)} packed/ paths already exist with DIFFERENT "
            f"content (re-derived census?); refusing to overwrite: {sorted(conflicts)[:5]}"
        )


# ---------------------------------------------------------------------------
# Probe-first commit loop (I11 / I13b / I14) + Hub-state resume (I9)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UnitExpected:
    """Content-anchored probe expectations for one commit unit (or cap probe)."""

    prefix: str
    packed_scope: str  # listing scope for the strict paths
    strict: Mapping[str, str]  # repo path -> expected git-blob sha1 (shards/parts/probes)
    sources: Mapping[str, Anchor]  # repo path -> census anchor (absent when landed)


def unit_expected_strict(unit: CommitUnit, pack_dir: Path | None = None) -> dict[str, str]:
    """Strict path->git-blob-sha1 expectations (shards; + index parts with pack_dir)."""
    packed = f"{unit.prefix}/{packing.PACKED_DIRNAME}"
    strict = {f"{packed}/{s.name}": s.blob_sha1 for s in unit.shards}
    if pack_dir is not None:
        for part in unit.index_part_names:
            strict[f"{packed}/{part}"] = packing.git_blob_sha1((pack_dir / part).read_bytes())
    return strict


def unit_expected(
    unit: CommitUnit, *, pack_dir: Path, census: Mapping[str, Anchor]
) -> UnitExpected:
    """Full probe expectations for a data commit unit."""
    return UnitExpected(
        prefix=unit.prefix,
        packed_scope=f"{unit.prefix}/{packing.PACKED_DIRNAME}",
        strict=unit_expected_strict(unit, pack_dir),
        sources={p: census[p] for p in sorted(unit.planned_deletes)},
    )


def probe_unit_state(api, expected: UnitExpected, *, repo_id: str) -> tuple[str, str]:
    """Content-anchored Hub-state probe (I13b) — the I9 resume predicate.

    Returns ``(state, head_sha)``; the head doubles as the I14 parent pin.
    States: ``landed`` (every strict path present with the expected blob sha1
    AND every source absent) · ``clean`` (no strict path present AND every
    source present matching its census anchor) · ``content-mismatch`` (any
    strict path present with a DIFFERENT digest — presence never implies
    landing) · ``drift`` (clean-shaped but a source's content changed since
    revision R — the I7 guard at fresh HEAD) · ``mixed`` (anything else).
    The cumulative INDEX.json/units.jsonl are deliberately NOT classified —
    their content is unit-cumulative by design.
    """
    from explore_persona_space.orchestrate import hub

    info = hub.retry_transient(
        lambda: api.repo_info(repo_id, repo_type="dataset"), what="repo_info (probe head)"
    )
    head = info.sha
    packed = {
        e.path: e
        for e in hub.list_repo_repofiles_under_path(
            api, repo_id, expected.packed_scope, revision=head
        )
    }
    mismatched = [
        p for p, want in expected.strict.items() if p in packed and packed[p].blob_id != want
    ]
    if mismatched:
        return "content-mismatch", head
    n_strict = len(expected.strict)
    n_match = sum(1 for p, want in expected.strict.items() if p in packed)
    n_absent = n_strict - n_match

    present: dict[str, object] = {}
    if expected.sources:
        for d in sorted({posixpath.dirname(p) for p in expected.sources}):
            for e in hub.list_repo_repofiles_under_path(api, repo_id, d, revision=head):
                present[e.path] = e
    src_absent = [p for p in expected.sources if p not in present]
    src_drift = [
        p for p, a in expected.sources.items() if p in present and not anchor_matches(present[p], a)
    ]

    if n_match == n_strict and len(src_absent) == len(expected.sources):
        return "landed", head
    if n_absent == n_strict and not src_absent:
        if src_drift:
            return "drift", head
        return "clean", head
    return "mixed", head


def commit_unit_probe_first(
    api,
    *,
    repo_id: str,
    expected: UnitExpected,
    adds: list[tuple[str, bytes | Path]],
    dels: list[str],
    commit_message: str,
    dry_run: bool = False,
    max_attempts: int = 3,
    max_pin_cycles: int = 8,
    max_429_cycles: int = 6,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    """Probe-first, parent-pinned commit (the ONLY ``create_commit`` site).

    Loop: probe (content-anchored, I13b) -> landed => done / mismatch =>
    AbortPrefix / drift => AbortPrefix (I7) / clean => issue pinned on the
    probe head (I14). Budgets: 3 attempts consumed by AMBIGUOUS outcomes ONLY
    (I11); 8 parent-pin (412) cycles (MF2d — definitive, re-probe + re-pin);
    6 rate-limit (429) cycles with Retry-After pacing => ``RateLimitedStop``
    (C18 — never reported as attempts-exhausted). A file-count rejection is
    the I2 global ``StopRepack``. ``create_commit`` is deliberately NEVER
    wrapped in ``retry_transient``.
    """
    if dry_run:
        return {"state": "dry-run", "sha": None, "n_adds": len(adds), "n_dels": len(dels)}
    packing.assert_test_mutation_interlock(repo_id)  # I18 — BEFORE any network access
    if os.environ.get("EPM_HF_OVERFLOW_ROUTING") == "1":
        raise RuntimeError(
            "I8: EPM_HF_OVERFLOW_ROUTING=1 — repack commits must go DIRECT to the "
            "canonical repo (a reroute would strand the pack across two repos)"
        )
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    attempts = pin_cycles = rl_cycles = 0
    while True:
        state, head = probe_unit_state(api, expected, repo_id=repo_id)
        if state == "landed":
            return {"state": "landed", "sha": head, "n_adds": len(adds), "n_dels": len(dels)}
        if state == "content-mismatch":
            raise AbortPrefix(
                f"I13(b) probe content-mismatch on {expected.prefix} (head {head[:12]}): "
                f"a strict path exists with a DIFFERENT digest — never overwrite"
            )
        if state == "drift":
            raise AbortPrefix(
                f"I7 census drift on {expected.prefix} at probe head {head[:12]}: a source "
                f"file changed since revision R"
            )
        if state == "mixed":
            raise AbortPrefix(
                f"probe found a MIXED unit state on {expected.prefix} at head {head[:12]} "
                f"(partial landing / foreign writer) — aborting prefix"
            )
        operations = [
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=payload) for path, payload in adds
        ] + [CommitOperationDelete(path_in_repo=p) for p in dels]
        try:
            # NO_RETRY: I11 probe-first — blind transient retry unsafe on deletion-bearing commits
            info = api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=commit_message,
                parent_commit=head,
            )
            sha = getattr(info, "oid", None) or getattr(info, "commit_id", None)
            return {
                "state": "committed",
                "sha": sha,
                "parent": head,
                "n_adds": len(adds),
                "n_dels": len(dels),
            }
        except Exception as err:  # classified below; unknown classes re-raise
            if _is_file_count_limit(err):
                raise StopRepack(
                    f"I2: file-count rejection on {expected.prefix} — global stop: {err}"
                ) from err
            if is_parent_conflict(err):
                pin_cycles += 1
                if pin_cycles > max_pin_cycles:
                    raise AbortPrefix(
                        f"pin-cycles-exhausted on {expected.prefix} "
                        f"({max_pin_cycles} parent-pin conflicts; I14)"
                    ) from err
                continue  # loop-top probe re-derives state + fresh pin (drift => I7 abort)
            if is_rate_limit(err):
                rl_cycles += 1
                if rl_cycles > max_429_cycles:
                    raise RateLimitedStop(
                        f"C18: rate-limit budget exhausted on {expected.prefix} "
                        f"({max_429_cycles} cycles) — a rate condition (the I11 "
                        f"attempt budget is untouched)"
                    ) from err
                sleep_fn(retry_after_seconds(err))
                continue
            if is_ambiguous_outcome(err):
                attempts += 1
                if attempts >= max_attempts:
                    raise AbortPrefix(
                        f"attempts-exhausted on {expected.prefix} "
                        f"({max_attempts} ambiguous outcomes; I11)"
                    ) from err
                continue  # re-probe: a timed-out-but-landed commit reads landed (MF2e)
            raise


# ---------------------------------------------------------------------------
# Consumer gate (I17)
# ---------------------------------------------------------------------------


def load_consumer_inventory(path: Path) -> dict:
    """Read the committed consumer inventory; a missing/invalid file FAILS CLOSED."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise ConsumerGateBlocked(
            f"I17: consumer inventory missing at {path} — failing CLOSED (rc=22)"
        ) from err
    if not isinstance(doc.get("consumers"), list):
        raise ConsumerGateBlocked(f"I17: malformed consumer inventory at {path} (no consumers[])")
    return doc


def _consumer_scoped(consumer: Mapping, prefix: str) -> bool:
    for pat in consumer.get("prefixes", []):
        norm = str(pat).rstrip("/")
        if prefix == norm or prefix.startswith(norm + "/") or fnmatch.fnmatch(prefix, norm):
            return True
    return False


def consumer_gate(inventory: Mapping, prefix: str) -> dict:
    """I17: BLOCK (rc=22, prefix byte-untouched) while any silent-empty consumer
    scoped to this prefix is unmigrated. Runs before ANY delete is composed."""
    blockers = [
        c
        for c in inventory["consumers"]
        if c.get("silent_empty") and not c.get("migrated") and _consumer_scoped(c, prefix)
    ]
    if blockers:
        names = sorted(str(c.get("script", "?")) for c in blockers)
        raise ConsumerGateBlocked(
            f"I17: {len(blockers)} unmigrated silent-empty consumer(s) scoped to "
            f"{prefix}: {names} — refusing to compose any delete (rc=22)"
        )
    n_scoped = sum(1 for c in inventory["consumers"] if _consumer_scoped(c, prefix))
    return {"prefix": prefix, "scoped_consumers": n_scoped, "blockers": 0}


# ---------------------------------------------------------------------------
# Commit phase driver
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Driver git sha for the I15 journal (degrades on git-less trees)."""
    env_sha = os.environ.get("EPS_GIT_SHA")
    if env_sha:
        return env_sha
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unavailable-no-git-checkout"


def append_state(state_path: Path | None, row: Mapping) -> None:
    """O_APPEND one JSON line of driver state (checkpoint-per-unit)."""
    if state_path is None:
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def run_commit_phase(
    api,
    *,
    repo_id: str,
    prefix: str,
    pack_dir: Path,
    census: Mapping[str, Anchor],
    retained: Mapping[str, str],
    revision: str,
    state_path: Path | None = None,
    dry_run: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
    driver_git_sha: str | None = None,
    ops_cap: int = UNIT_OPS_CAP,
    bytes_cap: int = UNIT_SHARD_BYTES_CAP,
) -> dict:
    """Compose + land every commit unit of one prefix, probe-first, resumable.

    Resume grain: a unit with a local journal record is skipped outright
    (its delete-set digest re-asserted against the re-derived plan); a unit
    without one is classified by the Hub-state probe (landed => recorded and
    skipped). On AbortPrefix the prefix is left in the byte-consistent
    "packed-unindexed-final" state (C12) recorded in the state file.
    """
    shards, groups, man = load_shard_infos(pack_dir)
    units, skip_reason = compose_units(
        shards, groups, prefix=prefix, ops_cap=ops_cap, bytes_cap=bytes_cap
    )
    if skip_reason is not None:
        print(f"[repack] {prefix} SKIPPED: {skip_reason}", flush=True)
        append_state(
            state_path,
            {"event": "prefix-skipped", "prefix": prefix, "reason": skip_reason, "ts": _now_iso()},
        )
        return {"status": "skipped", "reason": skip_reason, "prefix": prefix}
    n_units = len(units)
    driver_git_sha = driver_git_sha or _git_sha()
    if not dry_run:
        landed_overwrite_guard(api, repo_id=repo_id, prefix=prefix, units=units)
    records = load_local_journal(pack_dir)
    started_groups: list[str] = []
    committed = 0
    try:
        for unit in units:
            started_groups.extend(unit.new_groups)
            record = unit_journal_record(
                unit,
                n_units=n_units,
                census_key=man["census_key"],
                revision=revision,
                driver_git_sha=driver_git_sha,
            )
            if len(records) >= unit.unit_id:
                prior = records[unit.unit_id - 1]
                if prior["delete_set_sha256"] != record["delete_set_sha256"]:
                    raise AbortPrefix(
                        f"unit {unit.unit_id}: local journal delete-set digest differs from "
                        f"the re-derived plan (re-derived census on resume?) — aborting, "
                        f"never overwriting (I13a/I15)"
                    )
                continue  # landed in a prior run (local record)
            index_bytes = top_index_bytes(groups, started_groups)
            jbytes = journal_bytes([*records, record])
            adds = build_unit_ops(
                unit, pack_dir=pack_dir, index_bytes=index_bytes, journal_bytes_=jbytes
            )
            dels = sorted(unit.planned_deletes)
            assert_unit_invariants(unit, adds, dels, retained=retained, census=census)
            expected = unit_expected(unit, pack_dir=pack_dir, census=census)
            msg = (
                f"[#2321] repack {prefix} unit {unit.unit_id}/{n_units}: "
                f"+{len(adds)} shards/index/journal, -{len(dels)} originals"
            )
            t0 = time.monotonic()
            result = commit_unit_probe_first(
                api,
                repo_id=repo_id,
                expected=expected,
                adds=adds,
                dels=dels,
                commit_message=msg,
                dry_run=dry_run,
                sleep_fn=sleep_fn,
            )
            elapsed = time.monotonic() - t0
            records.append(record)
            if not dry_run:
                append_local_journal(pack_dir, record)
                append_state(
                    state_path,
                    {
                        "event": "unit-landed",
                        "prefix": prefix,
                        "unit_id": unit.unit_id,
                        "n_units": n_units,
                        "state": result["state"],
                        "sha": result.get("sha"),
                        "n_adds": len(adds),
                        "n_dels": len(dels),
                        "ts": _now_iso(),
                    },
                )
            committed += 1
            print(
                f"[repack] {prefix} unit {unit.unit_id}/{n_units} +{len(adds)} -{len(dels)} "
                f"commit={result.get('sha')} elapsed={elapsed:.0f}s",
                flush=True,
            )
            if not dry_run and result["state"] == "committed" and unit.unit_id < n_units:
                sleep_fn(COMMIT_SLEEP_S)
    except (AbortPrefix, RateLimitedStop, StopRepack) as err:
        append_state(
            state_path,
            {
                "event": "prefix-abort",
                "prefix": prefix,
                "label": "packed-unindexed-final",  # C12: byte-consistent, resumable
                "landed_units": len(records),
                "n_units": n_units,
                "error": str(err)[:500],
                "ts": _now_iso(),
            },
        )
        raise

    # Finalize: pack_manifest.json + final INDEX refresh — metadata only,
    # net +<=2, whitelisted by realized freed count (I3 whitelist).
    total_dels = sum(len(u.planned_deletes) for u in units)
    total_adds = sum(u.n_adds for u in units)
    freed_so_far = total_dels - total_adds
    manifest_bytes = (pack_dir / packing.MANIFEST_NAME).read_bytes()
    final_index = top_index_bytes(groups, list(groups))
    packed = f"{prefix}/{packing.PACKED_DIRNAME}"
    final_adds: list[tuple[str, bytes | Path]] = [
        (f"{packed}/{packing.MANIFEST_NAME}", manifest_bytes),
        (f"{packed}/{packing.INDEX_NAME}", final_index),
    ]
    if freed_so_far <= len(final_adds):
        raise AbortPrefix(
            f"finalize {prefix}: freed_so_far={freed_so_far} does not whitelist "
            f"{len(final_adds)} net-positive adds (I3)"
        )
    final_expected = UnitExpected(
        prefix=prefix,
        packed_scope=packed,
        strict={f"{packed}/{packing.MANIFEST_NAME}": packing.git_blob_sha1(manifest_bytes)},
        sources={},
    )
    result = commit_unit_probe_first(
        api,
        repo_id=repo_id,
        expected=final_expected,
        adds=final_adds,
        dels=[],
        commit_message=(
            f"[#2321] repack {prefix} finalize: +pack_manifest.json +INDEX.json refresh "
            f"(freed {freed_so_far} files across {n_units} units)"
        ),
        dry_run=dry_run,
        sleep_fn=sleep_fn,
    )
    if not dry_run:
        append_state(
            state_path,
            {
                "event": "prefix-finalized",
                "prefix": prefix,
                "n_units": n_units,
                "freed": freed_so_far,
                "state": result["state"],
                "sha": result.get("sha"),
                "ts": _now_iso(),
            },
        )
    print(
        f"[repack] {prefix} finalize {result['state']} commit={result.get('sha')} "
        f"freed={freed_so_far}",
        flush=True,
    )
    return {
        "status": "committed",
        "prefix": prefix,
        "n_units": n_units,
        "committed_this_run": committed,
        "freed": freed_so_far,
        "units": [
            {"unit_id": u.unit_id, "n_adds": u.n_adds, "n_dels": len(u.planned_deletes)}
            for u in units
        ],
    }


# ---------------------------------------------------------------------------
# Cap probe (§3.6: commits A/B/C; C4 recompute; C17 net-negative-first)
# ---------------------------------------------------------------------------


def run_cap_probe(
    api,
    *,
    repo_id: str,
    expected_live_count: int,
    count_fn: Callable[[], int],
    dry_run: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    """The three-commit at-cap semantics probe (probe deletions = sole I1 exemption).

    A: add ``probe_a.txt`` (repo -> exactly the cap). B: add ``probe_b.txt`` +
    delete ``probe_a.txt`` in ONE pinned commit (net zero AT the cap) —
    accepted => hypothesis confirmed. C: delete ``probe_b.txt`` (net -1).
    C4: the live count is re-read immediately before A; ANY drift aborts the
    round for recomputation (never report-and-proceed). C17: a rejected B
    routes to the net-NEGATIVE real-unit probe (issue1090_partial unit 1)
    BEFORE any invalidation verdict; a rejected A makes that real unit the
    probe outright.
    """
    verdict: dict = {"commits": [], "route": None, "hypothesis_confirmed": None}
    fresh = count_fn()
    verdict["live_count"] = fresh
    verdict["expected_live_count"] = expected_live_count
    if fresh != expected_live_count:
        verdict["route"] = "recompute-aborted"  # C4
        print(
            f"[cap-probe] C4 drift: live count {fresh} != expected {expected_live_count} — "
            f"aborting round; recompute the at-cap arithmetic from the fresh count",
            flush=True,
        )
        return verdict

    a_path = f"{PROBE_PREFIX}/probe_a.txt"
    b_path = f"{PROBE_PREFIX}/probe_b.txt"
    a_sha = packing.git_blob_sha1(_PROBE_A_BYTES)
    b_sha = packing.git_blob_sha1(_PROBE_B_BYTES)
    a_anchor = Anchor(size=len(_PROBE_A_BYTES), is_lfs=False, blob_id=a_sha, lfs_sha256=None)
    b_anchor = Anchor(size=len(_PROBE_B_BYTES), is_lfs=False, blob_id=b_sha, lfs_sha256=None)

    try:
        res_a = commit_unit_probe_first(
            api,
            repo_id=repo_id,
            expected=UnitExpected(
                prefix=PROBE_PREFIX, packed_scope=PROBE_PREFIX, strict={a_path: a_sha}, sources={}
            ),
            adds=[(a_path, _PROBE_A_BYTES)],
            dels=[],
            commit_message=f"[#2321] cap probe A: +probe_a.txt (repo -> exactly {HF_FILE_CAP})",
            dry_run=dry_run,
            sleep_fn=sleep_fn,
        )
    except StopRepack:
        # Commit A rejected: the first real repack unit (issue1090_partial
        # unit 1, net -161) IS the probe; nothing landed, nothing to clean.
        verdict["route"] = "commit-a-rejected-real-unit-probe"
        return verdict
    verdict["commits"].append({"probe": "A", **res_a})
    if dry_run:
        verdict["route"] = "dry-run"
        return verdict
    sleep_fn(COMMIT_SLEEP_S)

    try:
        res_b = commit_unit_probe_first(
            api,
            repo_id=repo_id,
            expected=UnitExpected(
                prefix=PROBE_PREFIX,
                packed_scope=PROBE_PREFIX,
                strict={b_path: b_sha},
                sources={a_path: a_anchor},
            ),
            adds=[(b_path, _PROBE_B_BYTES)],
            dels=[a_path],
            commit_message="[#2321] cap probe B: net-zero at cap (+probe_b -probe_a, pinned)",
            dry_run=dry_run,
            sleep_fn=sleep_fn,
        )
    except StopRepack:
        # C17: net-ZERO rejected. FIRST run the net-NEGATIVE real-unit probe
        # before ANY invalidation verdict; clean up probe_a (net -1) now.
        verdict["route"] = "commit-b-rejected-net-negative-real-unit-probe"
        cleanup = commit_unit_probe_first(
            api,
            repo_id=repo_id,
            expected=UnitExpected(
                prefix=PROBE_PREFIX,
                packed_scope=PROBE_PREFIX,
                strict={},
                sources={a_path: a_anchor},
            ),
            adds=[],
            dels=[a_path],
            commit_message="[#2321] cap probe cleanup: -probe_a (net -1)",
            dry_run=dry_run,
            sleep_fn=sleep_fn,
        )
        verdict["commits"].append({"probe": "cleanup-a", **cleanup})
        return verdict
    verdict["commits"].append({"probe": "B", **res_b})
    verdict["hypothesis_confirmed"] = True
    sleep_fn(COMMIT_SLEEP_S)

    res_c = commit_unit_probe_first(
        api,
        repo_id=repo_id,
        expected=UnitExpected(
            prefix=PROBE_PREFIX, packed_scope=PROBE_PREFIX, strict={}, sources={b_path: b_anchor}
        ),
        adds=[],
        dels=[b_path],
        commit_message="[#2321] cap probe C: -probe_b (net -1, pinned)",
        dry_run=dry_run,
        sleep_fn=sleep_fn,
    )
    verdict["commits"].append({"probe": "C", **res_c})
    verdict["route"] = "confirmed"
    return verdict


# ---------------------------------------------------------------------------
# Post-verify (I13c / C16 / C20) + I10 cleanup ordering
# ---------------------------------------------------------------------------


def _looks_v2_shaped(basename: str) -> bool:
    if basename in _ORPHAN_BASENAMES:
        return True
    return any(fnmatch.fnmatch(basename, pat) for pat in _ORPHAN_PART_PATTERNS)


def postverify_prefix(
    api,
    *,
    repo_id: str,
    prefix: str,
    pack_dir: Path,
    reap_paths: Iterable[Path] = (),
    rm_fn: Callable[[Path], None] = shutil.rmtree,
) -> dict:
    """Exact-set + landed-content verify, THEN (and only then) local cleanup.

    - exact-set presence of every v2-manifest-enumerated artifact under
      ``<prefix>/packed/`` (``verify_repo_paths_uploaded``), tolerating
      PRE-EXISTING non-v2 entries (C20; unexpected v2-SHAPED files flagged);
    - I13(c): remote blob_id == local git-blob sha1 for EVERY landed v2
      artifact, INDEX.json / units.jsonl / pack_manifest.json included;
    - C16: packed sources GONE at HEAD + every landed shard non-LFS;
    - I10: staging/pack/scratch reaped strictly AFTER all of the above pass.
    """
    from explore_persona_space.orchestrate import hub

    shards, groups, _man = load_shard_infos(pack_dir)
    packed = f"{prefix}/{packing.PACKED_DIRNAME}"
    expected_local: dict[str, bytes | Path] = {}
    for s in shards:
        expected_local[f"{packed}/{s.name}"] = pack_dir / s.name
    for g in groups.values():
        for part in g["index_files"]:
            expected_local[f"{packed}/{part}"] = pack_dir / part
    expected_local[f"{packed}/{packing.INDEX_NAME}"] = top_index_bytes(groups, list(groups))
    expected_local[f"{packed}/{packing.UNITS_JOURNAL_NAME}"] = journal_bytes(
        load_local_journal(pack_dir)
    )
    expected_local[f"{packed}/{packing.MANIFEST_NAME}"] = (
        pack_dir / packing.MANIFEST_NAME
    ).read_bytes()

    missing = hub.verify_repo_paths_uploaded(
        api, repo_id, sorted(expected_local), path_in_repo=packed
    )
    if missing:
        raise AbortPrefix(
            f"postverify {prefix}: {len(missing)} expected packed artifacts missing: "
            f"{missing[:5]} — NOT cleaning local staging (I10)"
        )
    remote = {e.path: e for e in hub.list_repo_repofiles_under_path(api, repo_id, packed)}
    mismatched = [
        p
        for p, payload in expected_local.items()
        if remote[p].blob_id != packing.git_blob_sha1(_payload_bytes(payload))
    ]
    if mismatched:
        raise AbortPrefix(
            f"I13(c) postverify content mismatch on {prefix}: {sorted(mismatched)[:5]} — "
            f"NOT cleaning local staging (I10)"
        )
    unexpected = sorted(p for p in remote if p not in expected_local)
    v2_shaped_unexpected = [p for p in unexpected if _looks_v2_shaped(posixpath.basename(p))]

    # C16: packed sources gone at HEAD; landed shards non-LFS.
    member_paths = {f"{prefix}/{m}" for s in shards for m in s.members}
    live = {e.path for e in hub.list_repo_repofiles_under_path(api, repo_id, prefix)}
    lingering = sorted(member_paths & live)
    if lingering:
        raise AbortPrefix(
            f"C16 postverify: {len(lingering)} packed sources STILL PRESENT on {prefix}: "
            f"{lingering[:5]}"
        )
    lfs_shards = [f"{packed}/{s.name}" for s in shards if remote[f"{packed}/{s.name}"].is_lfs]
    if lfs_shards:
        raise AbortPrefix(f"C16 postverify: landed shards routed to LFS: {lfs_shards[:5]}")

    # I10/C10: cleanup strictly AFTER every verify above passed.
    reaped = []
    for path in reap_paths:
        path = Path(path)
        if path.exists():
            rm_fn(path)
            reaped.append(str(path))
    return {
        "prefix": prefix,
        "n_verified": len(expected_local),
        "unexpected_nonv2": len(unexpected) - len(v2_shaped_unexpected),
        "unexpected_v2_shaped": v2_shaped_unexpected,  # flagged, never silently absorbed
        "reaped": reaped,
    }


# ---------------------------------------------------------------------------
# Remeasure (C11)
# ---------------------------------------------------------------------------


def count_prefix_files(api, *, repo_id: str, prefix: str) -> int:
    """Scoped file count of one prefix (server-side listing)."""
    from explore_persona_space.orchestrate import hub

    return len(hub.list_repo_repofiles_under_path(api, repo_id, prefix))


def count_repo_files(api, *, repo_id: str) -> int:
    """Full repo file count via one non-recursive root listing + scoped walks."""
    from huggingface_hub.hf_api import RepoFile

    from explore_persona_space.orchestrate import hub

    top = hub.retry_transient(
        lambda: list(api.list_repo_tree(repo_id, repo_type="dataset", recursive=False)),
        what="root listing (remeasure)",
    )
    total = 0
    for entry in top:
        if isinstance(entry, RepoFile):
            total += 1
        else:
            total += len(hub.list_repo_repofiles_under_path(api, repo_id, entry.path))
    return total


# ---------------------------------------------------------------------------
# Census persistence + state/sentinel plumbing
# ---------------------------------------------------------------------------


def save_census(
    path: Path,
    *,
    prefix: str,
    revision: str,
    entries: list,
    retained: Mapping[str, str],
    candidates: list,
    exclusions: Mapping[str, int],
) -> None:
    """Persist the walk phase's census (atomic write)."""
    doc = {
        "version": 1,
        "prefix": prefix,
        "revision": revision,
        "entries": [
            {
                "path": e.path,
                "size": e.size,
                "is_lfs": bool(e.is_lfs),
                "lfs_sha256": e.lfs_sha256,
                "blob_id": e.blob_id,
            }
            for e in entries
        ],
        "retained": dict(retained),
        "candidates": [e.path for e in candidates],
        "exclusions": dict(exclusions),
        "ts": _now_iso(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_census(path: Path) -> dict:
    """Load a walk-phase census; returns anchors keyed by repo path."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("version") != 1:
        raise RuntimeError(f"unrecognized census version at {path}: {doc.get('version')!r}")
    doc["anchors"] = {
        row["path"]: Anchor(
            size=row["size"],
            is_lfs=row["is_lfs"],
            blob_id=row["blob_id"],
            lfs_sha256=row["lfs_sha256"],
        )
        for row in doc["entries"]
    }
    return doc


def upload_state_file(api, *, repo_id: str, state_path: Path, prefix: str) -> str:
    """Best-effort additive telemetry upload (retried; cap-rejection tolerated LOUD).

    The local JSONL + pod sentinel are the primary records; a file-count-cap
    rejection here must not kill the run (the repo being AT the cap is the
    very condition this driver exists to fix).
    """
    from explore_persona_space.orchestrate import hub

    packing.assert_test_mutation_interlock(repo_id)  # I18
    dest = f"issue2321_repack/state/{prefix}.jsonl"
    try:
        hub.retry_transient(
            lambda: api.upload_file(
                path_or_fileobj=str(state_path),
                path_in_repo=dest,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"[#2321] repack state telemetry: {prefix}",
            ),
            what=f"state upload {dest}",
        )
        return "uploaded"
    except Exception as err:
        if _is_file_count_limit(err):
            print(
                f"[state] WARNING: state upload rejected on the file-count cap ({dest}); "
                f"local JSONL + sentinel remain the record",
                flush=True,
            )
            return "cap-rejected"
        raise


def sentinel_path() -> Path | None:
    """Pod-side results sentinel (N4); None when no sentinel surface exists."""
    env_dir = os.environ.get("EPM_I2321_SENTINEL_DIR")
    if env_dir:
        return Path(env_dir) / "issue-2321-results.json"
    if Path("/workspace").is_dir():
        return Path("/workspace/logs/issue-2321-results.json")
    return None


def write_sentinel(payload: Mapping) -> None:
    """Write the pod-side sentinel the VM poller drains (mkdir -p, atomic)."""
    path = sentinel_path()
    if path is None:
        print(
            "[sentinel] no sentinel surface (not a pod; EPM_I2321_SENTINEL_DIR unset)", flush=True
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Smoke (fixture tree; ZERO network — plan §5)
# ---------------------------------------------------------------------------


def _fixture_entry(path: str, data: bytes):
    """A RepoFileEntry-shaped census row for a local fixture file."""
    from explore_persona_space.orchestrate import hub

    return hub.RepoFileEntry(
        path=path,
        size=len(data),
        is_lfs=False,
        lfs_sha256=None,
        blob_id=packing.git_blob_sha1(data),
    )


def run_smoke() -> int:
    """Fixture-tree pack -> verify -> compose -> dry-run commit chain, zero network."""
    root = Path(tempfile.mkdtemp(prefix="i2321_smoke_"))
    try:
        prefix = "issue9999_fixture"
        stage_root = root / "stage"
        files = {
            f"{prefix}/a/{i:03d}.json": json.dumps({"i": i, "text": "x" * 100}).encode()
            for i in range(12)
        }
        files.update(
            {f"{prefix}/b/nested/{i:02d}.txt": f"row {i}\n".encode() * 30 for i in range(6)}
        )
        for path, data in files.items():
            target = stage_root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        entries = [_fixture_entry(p, d) for p, d in sorted(files.items())]
        census = {e.path: entry_anchor(e) for e in entries}
        retained = build_retained_set(entries, prefix=prefix, fetch_text=lambda p: "")
        candidates, exclusions = select_pack_candidates(entries, retained, prefix=prefix)
        print(f"[smoke] selection: {len(candidates)} candidates, exclusions={exclusions}")
        pack_dir = root / "pack"
        result = pack_prefix(
            prefix=prefix,
            census=census,
            candidate_paths=[e.path for e in candidates],
            stage_root=stage_root,
            pack_dir=pack_dir,
            source_revision="smoke",
            git_commit=_git_sha(),
        )
        report = verify_prefix(
            prefix=prefix,
            pack_dir=pack_dir,
            stage_root=stage_root,
            scratch_dir=root / "scratch",
            candidate_paths=[e.path for e in candidates],
        )
        print(f"[smoke] verify: {report}")
        summary = run_commit_phase(
            None,  # dry-run composes ops but makes ZERO network calls / mutations
            repo_id="smoke/fixture-repo",
            prefix=prefix,
            pack_dir=pack_dir,
            census=census,
            retained=retained,
            revision="smoke",
            dry_run=True,
        )
        print(f"[smoke] commit plan: {json.dumps(summary)}")
        assert result.n_members == len(candidates)
        print("[smoke] PASS (zero network; see module docstring for smoke blind spots)")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PHASES = (
    "walk",
    "download",
    "pack",
    "verify",
    "consumer-gate",
    "commit",
    "postverify",
    "remeasure",
    "cap-probe",
)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=PHASES, help="pipeline phase to run")
    ap.add_argument("--prefix", help="target prefix (required for per-prefix phases)")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    ap.add_argument(
        "--inventory",
        type=Path,
        default=Path(__file__).with_name("issue2321_consumer_inventory.json"),
        help="consumer inventory (I17); missing file FAILS CLOSED",
    )
    ap.add_argument("--revision", help="pinned revision R (resolved+persisted by walk)")
    ap.add_argument(
        "--i1739-liveness",
        choices=("proceed", "skip-deferred"),
        help="VM-orchestrator #1739 liveness verdict (REQUIRED for issue1739_* commit)",
    )
    ap.add_argument(
        "--expected-live-count",
        type=int,
        help="cap-probe C4 expectation: the live repo file count the at-cap arithmetic assumed",
    )
    ap.add_argument("--full-walk", action="store_true", help="remeasure: full 402-prefix walk")
    ap.add_argument("--dry-run", action="store_true", help="compose everything, mutate NOTHING")
    ap.add_argument("--smoke", action="store_true", help="fixture-tree chain, zero network")
    ap.add_argument("--import-check", action="store_true")
    return ap


def _require(args: argparse.Namespace, *names: str) -> None:
    for name in names:
        if getattr(args, name.replace("-", "_")) in (None, ""):
            raise SystemExit(f"--{name} is required for --phase {args.phase}")


def _resolve_revision(api, repo_id: str) -> str:
    from explore_persona_space.orchestrate import hub

    info = hub.retry_transient(
        lambda: api.repo_info(repo_id, repo_type="dataset"), what="repo_info (pin revision R)"
    )
    return info.sha


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok")
        return 0
    if args.smoke:
        return run_smoke()
    if not args.phase:
        raise SystemExit("--phase is required (or --smoke / --import-check)")

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from huggingface_hub import HfApi

    api = HfApi()
    work = args.work_root
    state_dir = work / "state"
    census_path = state_dir / f"{args.prefix}.census.json" if args.prefix else None
    state_path = state_dir / f"{args.prefix}.driver.jsonl" if args.prefix else None

    try:
        if args.phase == "walk":
            _require(args, "prefix")
            revision = args.revision or _resolve_revision(api, args.repo_id)
            entries = walk_prefix(api, repo_id=args.repo_id, prefix=args.prefix, revision=revision)
            fetch_dir = work / "manifests" / args.prefix

            def _fetch_text(path: str) -> str:
                from explore_persona_space.orchestrate import hub

                target = fetch_dir / path
                hub.stage_hub_file(
                    args.repo_id, path, target, repo_type="dataset", revision=revision
                )
                return target.read_text(encoding="utf-8")

            retained = build_retained_set(entries, prefix=args.prefix, fetch_text=_fetch_text)
            candidates, exclusions = select_pack_candidates(entries, retained, prefix=args.prefix)
            save_census(
                census_path,
                prefix=args.prefix,
                revision=revision,
                entries=entries,
                retained=retained,
                candidates=candidates,
                exclusions=exclusions,
            )
            print(
                f"[walk] {args.prefix} rev={revision[:12]} files={len(entries)} "
                f"candidates={len(candidates)} retained={len(retained)} "
                f"exclusions={json.dumps(exclusions)}",
                flush=True,
            )
            return 0

        if args.phase in ("download", "pack", "verify", "commit", "postverify"):
            _require(args, "prefix")
            census_doc = load_census(census_path)
            anchors = census_doc["anchors"]
            candidates = census_doc["candidates"]
            revision = census_doc["revision"]
            stage_root = work / "stage"
            pack_dir = work / "pack" / args.prefix
            if args.phase == "download":
                report = download_prefix(
                    repo_id=args.repo_id,
                    prefix=args.prefix,
                    census=anchors,
                    candidate_paths=candidates,
                    revision=revision,
                    stage_root=stage_root,
                )
                print(f"[download] {args.prefix} done: {json.dumps(report)}", flush=True)
                return 0
            if args.phase == "pack":
                result = pack_prefix(
                    prefix=args.prefix,
                    census=anchors,
                    candidate_paths=candidates,
                    stage_root=stage_root,
                    pack_dir=pack_dir,
                    source_revision=revision,
                    git_commit=_git_sha(),
                )
                print(
                    f"[pack] {args.prefix} n_members={result.n_members} "
                    f"reused={result.reused} census_key={result.census_key[:12]}",
                    flush=True,
                )
                return 0
            if args.phase == "verify":
                report = verify_prefix(
                    prefix=args.prefix,
                    pack_dir=pack_dir,
                    stage_root=stage_root,
                    scratch_dir=work / "scratch" / args.prefix,
                    candidate_paths=candidates,
                )
                print(f"[verify] {args.prefix} {json.dumps(report)}", flush=True)
                return 0
            if args.phase == "commit":
                # I17: the gate runs BEFORE any delete is composed.
                gate = consumer_gate(load_consumer_inventory(args.inventory), args.prefix)
                print(f"[consumer-gate] {json.dumps(gate)}", flush=True)
                if args.prefix.startswith("issue1739_") and not args.dry_run:
                    if not args.i1739_liveness:
                        raise SystemExit(
                            "--i1739-liveness proceed|skip-deferred is REQUIRED for "
                            "issue1739_* commit phases (#1739 coordination is "
                            "VM-orchestrator-side; the driver receives the verdict)"
                        )
                    if args.i1739_liveness == "skip-deferred":
                        append_state(
                            state_path,
                            {
                                "event": "prefix-deferred",
                                "prefix": args.prefix,
                                "reason": "i1739-liveness",
                                "ts": _now_iso(),
                            },
                        )
                        print(f"[repack] {args.prefix} DEFERRED (#1739 liveness)", flush=True)
                        return 0
                summary = run_commit_phase(
                    api,
                    repo_id=args.repo_id,
                    prefix=args.prefix,
                    pack_dir=pack_dir,
                    census=anchors,
                    retained=census_doc["retained"],
                    revision=revision,
                    state_path=state_path,
                    dry_run=args.dry_run,
                )
                if not args.dry_run and state_path.exists():
                    upload_state_file(
                        api, repo_id=args.repo_id, state_path=state_path, prefix=args.prefix
                    )
                    write_sentinel(
                        {"issue": 2321, "phase": "commit", "prefix": args.prefix, **summary}
                    )
                    n_after = count_prefix_files(api, repo_id=args.repo_id, prefix=args.prefix)
                    print(f"[remeasure] {args.prefix} files-after={n_after}", flush=True)
                print(f"[commit] {args.prefix} {json.dumps(summary)}", flush=True)
                return 0
            if args.phase == "postverify":
                report = postverify_prefix(
                    api,
                    repo_id=args.repo_id,
                    prefix=args.prefix,
                    pack_dir=pack_dir,
                    reap_paths=(
                        work / "stage" / args.prefix,
                        work / "scratch" / args.prefix,
                        work / "pack" / args.prefix,
                        work / "hfhome",  # C10
                    ),
                )
                print(f"[postverify] {args.prefix} {json.dumps(report)}", flush=True)
                return 0

        if args.phase == "consumer-gate":
            _require(args, "prefix")
            gate = consumer_gate(load_consumer_inventory(args.inventory), args.prefix)
            print(f"[consumer-gate] {json.dumps(gate)}", flush=True)
            return 0

        if args.phase == "remeasure":
            counts = {}
            for prefix in PREFIX_ORDER:
                counts[prefix] = count_prefix_files(api, repo_id=args.repo_id, prefix=prefix)
            report: dict = {"per_prefix": counts}
            if args.full_walk:
                report["repo_total"] = count_repo_files(api, repo_id=args.repo_id)
            print(f"[remeasure] {json.dumps(report)}", flush=True)
            return 0

        if args.phase == "cap-probe":
            _require(args, "expected-live-count")
            verdict = run_cap_probe(
                api,
                repo_id=args.repo_id,
                expected_live_count=args.expected_live_count,
                count_fn=lambda: count_repo_files(api, repo_id=args.repo_id),
                dry_run=args.dry_run,
            )
            print(f"[cap-probe] {json.dumps(verdict)}", flush=True)
            write_sentinel({"issue": 2321, "phase": "cap-probe", **verdict})
            return 0

        raise SystemExit(f"unhandled phase: {args.phase}")
    except ConsumerGateBlocked as err:
        print(f"[gate-blocked] {err}", file=sys.stderr, flush=True)
        return RC_CONSUMER_GATE
    except StopRepack as err:
        print(f"[stop-repack] {err}", file=sys.stderr, flush=True)
        return RC_STOP_REPACK
    except RateLimitedStop as err:
        print(f"[rate-limited] {err}", file=sys.stderr, flush=True)
        return RC_RATE_LIMITED
    except AbortPrefix as err:
        print(f"[abort-prefix] {err}", file=sys.stderr, flush=True)
        return RC_ABORT_PREFIX


if __name__ == "__main__":
    sys.stdout.flush()
    raise SystemExit(main())
