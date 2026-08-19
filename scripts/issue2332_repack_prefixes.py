#!/usr/bin/env python3
"""Repack 8 many-tiny-file prefixes on the canonical HF data repo into tar shards (#2332).

Each target prefix on ``superkaiba1/explore-persona-space-data`` is repacked in
place: revision-pinned identity snapshot -> 4-worker backoff download ->
staged identity verify -> tar-shard pack (+ index.json/manifest.json) -> local
member verify -> ``upload_large_folder`` from an ISOLATED per-prefix upload
root -> Hub-side hash verify -> identity-keyed batched deletes under a
``parent_commit`` CAS -> final re-list assert. Everything checkpoints to
``state.json`` after every step and is resumable per prefix/chunk.

Subcommands (plan v2 SS4.0):
  sizing            P2 Hub-READ sizing pass: per-prefix bytes/census/shard plan,
                    full-repo count (scoped-sum method), inventory reconcile
  count             fresh full-repo file count only
  repack --prefix   steps 2-7: snapshot -> download -> staged verify -> pack ->
                    local verify -> upload
  verify --prefix   step 8: Hub-side identity verify of the packed files
  delete --prefix   steps 9-10: identity-keyed batched CAS deletes + final assert
  run [--prefix P]  full chain for one prefix, or all 8 in plan order

Gates (all encoded here, re-run per step; plan v2 SS4.4-SS4.6 / SS7; pinned by
``tests/test_issue2332_repack_gates.py``):
  G-mover   pgrep probe for the live residue mover; rc semantics resolved in
            Python (rc 1 + empty stdout is the ONLY safe arm; rc 0 = alive =
            refuse; rc >= 2 = probe error = refuse).
  G-writer  four-way: owner task status parked/terminal + no live session on
            the owner + no live pod for the owner + no fresh run-signal marker
            newer than the owner's latest done-transition.
  G-1739    live-process probe keyed on the RESOLVED #1739 worktree path with
            self-PID/ancestor exclusion (never bare issue tokens), plus the
            session/pod legs.
  G-cap     fresh scoped full-repo count + k <= 1,000,000 - 1,000 asserted
            BEFORE every upload call (worst case any internal commit reaches).
  G-verify-before-delete  state machine: the delete stage is unreachable
            without a recorded hub-verify PASS, and a PASS recorded by a PRIOR
            process run (restart) forces a cheap re-verify before any delete.

Adapted from the residue-mover reference implementation (archived verbatim at
the #2332 task artifacts dir as ``hf_move_repair_reference.py``): scoped
listing, 4-worker download with exponential backoff to ~60 s, the
``upload_large_folder`` call shape, verify-before-delete, and ``state.json``
checkpointing. Deliberate non-carryover: the reference's folder-level delete
API call (its line 165) is BANNED in this script — it would destroy the
in-prefix ``__packed__/`` tree; deletion here is explicit-path
``CommitOperationDelete`` batches only, and the pinning test source-asserts
the banned API names never appear in this file.

Hard-won operational constraints encoded (task #2332 body, measured
2026-08-15): transport status is NEVER evidence of outcome (verify by
readback only); scoped listings only (root recursive listing 504s past
~62k files); <= 4 download workers with backoff; ONE API consumer at a time
(the G-mover gate); stage off ``/`` (data disk + headroom assert); exclude
``.cache/**``, ``**/tmp_*``, ``**/*.incomplete`` from uploads.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.orchestrate.secret_scrub import assert_upload_clean
from explore_persona_space.task_workflow import (
    find_task_path,
    primary_checkout_root,
    registry_path,
)

load_dotenv()
from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download  # noqa: E402
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError  # noqa: E402

# Import-time capability check (plan v2 SS4.0): the CAS delete design depends
# on the `parent_commit` kwarg. If this env's huggingface_hub lacks it, STOP —
# never work around (a blind delete commit races concurrent writers).
if "parent_commit" not in inspect.signature(HfApi.create_commit).parameters:
    raise SystemExit(
        "FATAL: HfApi.create_commit has no `parent_commit` kwarg in this huggingface_hub. "
        "The #2332 identity-keyed CAS delete design requires it — STOP (do not work around)."
    )

SRC = "superkaiba1/explore-persona-space-data"

# Plan v2 SS5 work-item table: prefix -> (owning issue, 2026-08-15 inventory count).
# Order IS the execution order: pilot (smallest) first, then ascending, #1739 LAST.
PREFIXES: dict[str, tuple[int, int]] = {
    "issue1489_ctx_aug": (1489, 30_360),
    "issue2224_screening": (2224, 41_759),
    "issue1434_writingstyle": (1434, 49_380),
    "issue667_alllayer": (667, 53_858),
    "issue1586_methodgen": (1586, 54_103),
    "issue1090_pvdatagen": (1090, 58_392),
    "issue1481_conpos_grid": (1481, 206_604),
    "issue1739_ctxmap": (1739, 43_905),
}
PILOT_PREFIX = "issue1489_ctx_aug"

STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue2332_repack")
STATE_PATH = STAGE / "state.json"
MOVER_STATE = Path("/mnt/eps-data/thomasjiralerspong/issue2304_residue_move/state.json")

DL_WORKERS = 4  # measured constraint: 16 workers -> 4-6% failures; 4 -> clean
UPLOAD_WORKERS = 4
LIST_WORKERS = 8  # half the measured-safe 16 (2026-08-15 inventory, 0 errors)
MAX_ATTEMPTS = 6
SHARD_TARGET_BYTES = 2 * 2**30  # ~2 GiB/shard (ungrounded; smoke-tested at pilot)
CAP_TOTAL = 1_000_000
CAP_BUFFER = 1_000  # G-cap: fresh_count + k <= CAP_TOTAL - CAP_BUFFER
VM_FOOTPRINT_MAX_BYTES = 50 * 10**9  # VM_ANALYSIS_FOOTPRINT_GB_MAX; trigger on 2x prefix bytes
CHUNK_STAGED_MAX_BYTES = 40 * 10**9  # per-chunk staged footprint (2x chunk bytes) ceiling
HEADROOM_FACTOR = 1.5
IGNORE_PATTERNS = [".cache/**", "**/tmp_*", "**/*.incomplete"]
DELETE_BATCH_PILOT = 1_000
DELETE_BATCH_STEADY = 10_000  # raised only after a clean pilot (pilot-gated)
MAX_CONSECUTIVE_DELETE_FAILURES = 10  # fail-loud bound on the re-list/re-derive loop

# Statuses under which an owning issue is considered parked/terminal (G-writer leg 1).
PARKED_OK_STATUSES = {"awaiting_promotion", "completed", "archived", "on_hold"}
# G-writer leg 4: run-signal marker kinds vs done-transition kinds (watcher predicate set).
SIGNAL_KINDS = {"epm:run-launched", "epm:followup-scope", "epm:free-analysis-followup-run"}
DONE_KINDS = {"epm:promoted", "epm:status-changed"}
# Pod states that do NOT count as a live pod for the gate.
POD_DEAD_STATES = {"EXITED", "TERMINATED"}

RUN_ID = uuid.uuid4().hex  # per-process; a recorded hub-verify PASS from another run re-verifies


class GateRefusal(RuntimeError):
    """A safety gate refused the step (fail-loud, never worked around)."""


class VerifyFailure(RuntimeError):
    """An identity/verify check failed (source left intact)."""


class PrefixIncomplete(RuntimeError):
    """A prefix could not be completed this pass (staging preserved, source untouched)."""


_API: HfApi | None = None


def _get_api() -> HfApi:
    global _API
    if _API is None:
        _API = HfApi()
    return _API


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── state ───────────────────────────────────────────────────────────────────


def _load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {"run_history": [], "prefixes": {}}


def _save_state(st: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
    os.replace(tmp, STATE_PATH)


def _srcmap_path(prefix: str) -> Path:
    return STAGE / f"srcmap_{prefix}.json"


def _staged_hashes_path(prefix: str) -> Path:
    return STAGE / f"staged_hashes_{prefix}.json"


def _write_staged_hashes(prefix: str, src_revision: str, verified: dict[str, dict]) -> None:
    """Atomic (tmp + os.replace) write of the v2 staged-hash sidecar:
    ``{"src_revision": <sha>, "hashes": {path: {"sha256":..., "gitsha1":...}}}``.
    The store holds ONLY units whose identity verification PASSED (review r2 —
    a unit may never enter the sidecar before its verify verdict)."""
    tmp = _staged_hashes_path(prefix).with_suffix(".tmp")
    tmp.write_text(json.dumps({"src_revision": src_revision, "hashes": verified}, sort_keys=True))
    os.replace(tmp, _staged_hashes_path(prefix))


def _load_staged_sidecar(prefix: str, src_revision: str) -> dict[str, dict] | None:
    """v2 sidecar loader: the verified-units store, or None when the sidecar is
    absent, legacy-schema, unparseable, or fingerprinted to a DIFFERENT
    ``src_revision`` (review r2: the sidecar is an identity-bearing resume
    store — unfingerprinted or cross-snapshot content is DISCARDED, forcing a
    full re-hash)."""
    p = _staged_hashes_path(prefix)
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict) or raw.get("src_revision") != src_revision:
        return None
    hashes = raw.get("hashes")
    if not isinstance(hashes, dict):
        return None
    return hashes


# Pipeline step order — the invalidation chain's single source of truth.
STEP_ORDER = ("download", "staged_verify", "pack", "local_verify", "upload", "hub_verify")


def _invalidate_downstream(rec: dict, step: str) -> None:
    """Review r2: any upstream artifact change invalidates EVERY downstream
    step checkpoint — a fresh download forces staged re-verify, a rebuilt pack
    forces local verify + upload, etc. (checkpoints are consumed only when
    their whole upstream chain is intact)."""
    steps = rec.get("steps") or {}
    for later in STEP_ORDER[STEP_ORDER.index(step) + 1 :]:
        steps.pop(later, None)


def _load_srcmap(prefix: str) -> dict[str, dict]:
    p = _srcmap_path(prefix)
    if not p.is_file():
        raise PrefixIncomplete(
            f"{prefix}: no snapshot sidecar at {p} — run the snapshot step first"
        )
    return json.loads(p.read_text())


# ─── gates ───────────────────────────────────────────────────────────────────


def _run_pgrep() -> subprocess.CompletedProcess:
    """G-mover probe. The bracketed pattern never matches its own command line."""
    return subprocess.run(["pgrep", "-af", "hf_move_repai[r].py"], capture_output=True, text=True)


def _mover_done_count_advisory() -> str:
    """Advisory forensic line: the residue mover's state.json done-count (read-only)."""
    try:
        st = json.loads(MOVER_STATE.read_text())
        return f"mover done-count={len(st.get('done', {}))} failed={len(st.get('failed', {}))}"
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return "mover state.json unavailable"


def gate_mover() -> None:
    """G-mover: refuse while the residue mover is alive; refuse on probe error.

    rc semantics resolved HERE in Python (never shell chaining): the SAFE arm
    is exactly rc == 1 with empty stdout. rc 0 = mover alive; rc >= 2 (or a
    signal-negative rc) = pgrep itself errored — both refuse, fail-loud.
    """
    p = _run_pgrep()
    log(f"    G-mover advisory: {_mover_done_count_advisory()}")
    if p.returncode == 0:
        raise GateRefusal(f"G-mover: residue mover ALIVE — refuse.\n{p.stdout.strip()}")
    if p.returncode != 1:
        raise GateRefusal(
            f"G-mover: pgrep probe errored (rc={p.returncode}) — refuse, fail-loud. "
            f"stderr: {p.stderr.strip()!r}"
        )
    if p.stdout.strip():
        raise GateRefusal("G-mover: rc==1 but stdout non-empty — inconsistent probe, refuse")


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _spawn_session_list_text() -> str:
    p = subprocess.run(
        [sys.executable, str(_scripts_dir() / "spawn_session.py"), "list"],
        capture_output=True,
        text=True,
        cwd=str(primary_checkout_root()),
    )
    if p.returncode != 0:
        raise GateRefusal(
            f"G-writer: spawn_session.py list failed rc={p.returncode} — refuse (fail closed). "
            f"stderr: {p.stderr.strip()[:300]!r}"
        )
    return p.stdout


def _sessions_for_issue(issue: int) -> list[str]:
    tok = re.compile(rf"#{issue}\b")
    return [ln for ln in _spawn_session_list_text().splitlines() if tok.search(ln)]


def _pods_for_issue(issue: int) -> list[str]:
    p = subprocess.run(
        [
            sys.executable,
            str(_scripts_dir() / "pod_lifecycle.py"),
            "list-ephemeral",
            "--issue",
            str(issue),
        ],
        capture_output=True,
        text=True,
        cwd=str(primary_checkout_root()),
    )
    if p.returncode != 0:
        raise GateRefusal(
            f"G-writer: pod list-ephemeral --issue {issue} failed rc={p.returncode} — refuse. "
            f"stderr: {p.stderr.strip()[:300]!r}"
        )
    out = p.stdout
    if "No ephemeral pod" in out:
        return []
    live: list[str] = []
    for ln in out.splitlines():
        fields = ln.split()
        if len(fields) < 3 or fields[0] in {"NAME"} or set(ln.strip()) == {"-"}:
            continue
        if fields[1] == f"#{issue}" and fields[2].upper() not in POD_DEAD_STATES:
            live.append(ln)
    return live


def _owner_status(issue: int) -> str:
    reg = json.loads(registry_path().read_text())
    entry = reg.get("tasks", {}).get(str(issue))
    if not entry:
        raise GateRefusal(f"G-writer: owner #{issue} missing from REGISTRY — refuse (fail loud)")
    return str(entry.get("status", ""))


def _fresh_run_signal(issue: int) -> str | None:
    """G-writer leg 4: a run-signal marker newer than the latest done-transition."""
    events = find_task_path(issue) / "events.jsonl"
    if not events.is_file():
        return None
    last_done = ""
    sig_ts, sig_kind = "", None
    for lineno, line in enumerate(events.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            # FAIL CLOSED (review r1): a malformed row could BE the fresh
            # run-signal marker this gate exists to see — refuse, never skip.
            raise GateRefusal(
                f"G-writer: unparseable events.jsonl line {lineno} on #{issue} — fail closed: {e}"
            ) from e
        kind, ts = row.get("kind", ""), row.get("ts", "")
        if kind in DONE_KINDS and ts > last_done:
            last_done = ts
        if kind in SIGNAL_KINDS and ts > sig_ts:
            sig_ts, sig_kind = ts, kind
    if sig_kind and sig_ts > last_done:
        return f"{sig_kind} at {sig_ts} > latest done-transition {last_done or '<none>'}"
    return None


def gate_writer(prefix: str) -> None:
    """G-writer four-way check for the prefix's owning issue (plan SS4.6 step 1)."""
    owner = PREFIXES[prefix][0]
    status = _owner_status(owner)
    if status not in PARKED_OK_STATUSES:
        raise GateRefusal(f"G-writer: owner #{owner} status={status!r} is not parked/terminal")
    if owner != 1739:  # the #1739 session leg is owned by G-1739 (plan SS4.6)
        rows = _sessions_for_issue(owner)
        if rows:
            raise GateRefusal(f"G-writer: live session mapped to #{owner}: {rows[:3]}")
    pods = _pods_for_issue(owner)
    if pods:
        raise GateRefusal(f"G-writer: live pod for #{owner}: {pods[:3]}")
    sig = _fresh_run_signal(owner)
    if sig:
        raise GateRefusal(f"G-writer: fresh run-signal marker on #{owner}: {sig}")


def _resolve_1739_worktree() -> str | None:
    """Absolute path of the #1739 worktree from `git worktree list` (None if absent)."""
    p = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=str(primary_checkout_root()),
    )
    if p.returncode != 0:
        raise GateRefusal(f"G-1739: git worktree list failed rc={p.returncode} — refuse")
    path: str | None = None
    for block in p.stdout.split("\n\n"):
        lines = block.strip().splitlines()
        if not lines:
            continue
        wt_path = lines[0].removeprefix("worktree ").strip()
        branch = next(
            (ln.removeprefix("branch ").strip() for ln in lines if ln.startswith("branch ")), ""
        )
        if branch == "refs/heads/issue-1739" or wt_path.rstrip("/").endswith("/issue-1739"):
            path = wt_path.rstrip("/")
    return path


def _ancestor_pids() -> set[int]:
    """This process's PID plus every ancestor PID (walk /proc PPid chain)."""
    out = {os.getpid()}
    pid = os.getpid()
    for _ in range(128):
        try:
            txt = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            break
        m = re.search(r"^PPid:\s*(\d+)", txt, re.M)
        if not m:
            break
        pid = int(m.group(1))
        out.add(pid)
        if pid <= 1:
            break
    return out


def _iter_procs():
    """Yield (pid, cwd, cmdline) for every live process readable under /proc."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            cwd = os.readlink(entry / "cwd")
        except OSError:
            cwd = ""
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            cmd = ""
        yield pid, cwd, cmd


def gate_1739() -> None:
    """G-1739 (self-match-safe): live-PROCESS probe keyed on the RESOLVED #1739
    worktree path — never bare issue tokens (the runner's own argv contains
    ``issue1739_ctxmap``) — excluding this PID and all ancestors; plus the
    session-map and pod legs (plan SS4.6 G-1739 (a)-(d)).
    """
    wt = _resolve_1739_worktree()
    excluded = _ancestor_pids()
    if wt:
        for pid, cwd, cmd in _iter_procs():
            if pid in excluded:
                continue
            if cwd == wt or cwd.startswith(wt + "/") or wt in cmd:
                raise GateRefusal(
                    f"G-1739: live #1739 process pid={pid} (cwd={cwd!r}) — DEFER the prefix"
                )
    rows = _sessions_for_issue(1739)
    if rows:
        raise GateRefusal(f"G-1739: live session mapped to #1739: {rows[:3]} — DEFER")
    pods = _pods_for_issue(1739)
    if pods:
        raise GateRefusal(f"G-1739: live pod for #1739: {pods[:3]} — DEFER")


def _run_hub_step_gates(prefix: str) -> None:
    """The per-step gate battery (plan SS4.6 gate policy): G-mover + G-writer
    (+ G-1739 for the #1739 prefix) before EVERY Hub-touching step."""
    gate_mover()
    gate_writer(prefix)
    if PREFIXES[prefix][0] == 1739:
        gate_1739()


LIST_RETRY_ATTEMPTS = 4


def _retry_listing(thunk, what: str):
    """Bounded transient-retry for READ-ONLY Hub listings/fetches (#920 class:
    huggingface_hub natively retries only 429 on the tree API, so a transient
    5xx on a cursor page would kill an otherwise-healthy multi-hour run).

    The thunk MUST materialize its iterator INTERNALLY (a cursor-page 504
    raises DURING iteration). Typed not-found errors are NEVER retried and
    NEVER coerced — the caller owns their semantics. Read-only calls have no
    side effects, so a bounded broad retry is safe (unlike the CAS delete
    commit, which is deliberately never retried — plan SS4.6 step 9)."""
    last: Exception | None = None
    for attempt in range(LIST_RETRY_ATTEMPTS):
        try:
            return thunk()
        except (EntryNotFoundError, RepositoryNotFoundError):
            raise
        except Exception as e:
            last = e
            if attempt < LIST_RETRY_ATTEMPTS - 1:
                wait = min(60, 5 * 3**attempt)
                log(
                    f"    [{what}] transient Hub error (attempt {attempt + 1}/"
                    f"{LIST_RETRY_ATTEMPTS}, retry in {wait}s): "
                    f"{type(e).__name__}: {str(e)[:160]}"
                )
                time.sleep(wait)
    assert last is not None
    raise last


def _fresh_repo_count() -> int:
    """Full-repo file count via top-level non-recursive listing + per-entry
    scoped recursive counts (the 2026-08-15 inventory method; NEVER a root
    recursive listing — it 504s past ~62k files). All listings ride
    _retry_listing's bounded transient retry."""
    api = _get_api()

    def _top() -> list:
        # HUB_VERIFY_RETRY_EXEMPT: routed through _retry_listing bounded backoff
        it = api.list_repo_tree(SRC, repo_type="dataset", recursive=False)
        return list(it)

    top = _retry_listing(_top, "count:top-level")
    n_top_files = sum(1 for t in top if getattr(t, "size", None) is not None)
    folders = [t.path for t in top if getattr(t, "size", None) is None]

    def count_one(folder: str) -> int:
        def _walk() -> int:
            # HUB_VERIFY_RETRY_EXEMPT: routed through _retry_listing bounded backoff
            it = api.list_repo_tree(SRC, path_in_repo=folder, repo_type="dataset", recursive=True)
            return sum(1 for t in it if getattr(t, "size", None) is not None)

        return _retry_listing(_walk, f"count:{folder}")

    with ThreadPoolExecutor(max_workers=LIST_WORKERS) as ex:
        counts = list(ex.map(count_one, folders))
    return n_top_files + sum(counts)


def gate_cap(k: int, fresh_count_fn=None) -> int:
    """G-cap: fresh scoped full-repo count + the FULL +k of this prefix/chunk
    must clear the buffered cap BEFORE the upload call (worst case any internal
    commit of ``upload_large_folder`` can reach). Returns the fresh count."""
    fn = fresh_count_fn if fresh_count_fn is not None else _fresh_repo_count
    n = fn()
    if n + k > CAP_TOTAL - CAP_BUFFER:
        raise GateRefusal(
            f"G-cap: fresh count {n:,} + k={k} exceeds buffered cap "
            f"{CAP_TOTAL - CAP_BUFFER:,} — upload refused"
        )
    return n


def _staging_headroom_assert(need_bytes: int) -> None:
    """Preamble assert: staging resolves under /mnt/eps-data AND free space
    >= HEADROOM_FACTOR x the projected staged footprint (plan SS9)."""
    STAGE.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["df", "-P", str(STAGE)], capture_output=True, text=True)
    if p.returncode != 0:
        raise GateRefusal(f"headroom: df -P failed rc={p.returncode}: {p.stderr.strip()!r}")
    fields = p.stdout.strip().splitlines()[-1].split()
    mount, avail_bytes = fields[5], int(fields[3]) * 1024
    if mount != "/mnt/eps-data":
        raise GateRefusal(f"headroom: staging resolves to mount {mount!r}, not /mnt/eps-data")
    need = int(need_bytes * HEADROOM_FACTOR)
    if avail_bytes < need:
        raise GateRefusal(
            f"headroom: {avail_bytes / 2**30:.1f} GiB free < "
            f"{need / 2**30:.1f} GiB needed ({HEADROOM_FACTOR}x projected)"
        )


# ─── Hub listing / identity helpers ─────────────────────────────────────────


def _entry_identity(t) -> dict:
    """Content identity of a tree entry: lfs.sha256 when the entry's ACTUAL
    ``lfs`` attribute is set, else ``blob_id`` (plan SS4.6 step 2)."""
    lfs = getattr(t, "lfs", None)
    if lfs is not None:
        sha = getattr(lfs, "sha256", None)
        if sha is None and isinstance(lfs, dict):
            sha = lfs.get("sha256")
        if not sha:
            raise VerifyFailure(f"LFS tree entry without sha256: {t.path}")
        return {"size": t.size, "oid": sha, "lfs": True}
    blob = getattr(t, "blob_id", None)
    if not blob:
        raise VerifyFailure(f"non-LFS tree entry without blob_id: {t.path}")
    return {"size": t.size, "oid": blob, "lfs": False}


def _list_prefix(prefix: str, revision: str | None = None) -> dict[str, dict]:
    """Scoped recursive listing -> {path: {size, oid, lfs}} (never a root listing).

    Transient errors ride _retry_listing's bounded backoff. The ONE legal
    empty-listing case is a typed ``EntryNotFoundError`` — the PATH is absent
    at that revision (re-list after the last delete batch removed everything
    outside ``__packed__``, pre-first-upload probes). Every OTHER error —
    ``RepositoryNotFoundError``, auth-shaped 404s, 5xx after retries —
    PROPAGATES; a transport/auth failure is never coerced to an empty
    inventory (review r1: the old ``"404" in str(e)`` substring match waved
    a repo-level 404 through as "nothing left")."""
    api = _get_api()

    def _walk() -> list:
        # HUB_VERIFY_RETRY_EXEMPT: routed through _retry_listing bounded backoff
        it = api.list_repo_tree(
            SRC, path_in_repo=prefix, repo_type="dataset", recursive=True, revision=revision
        )
        return list(it)

    try:
        entries = _retry_listing(_walk, f"list:{prefix}")
    except EntryNotFoundError:
        return {}
    out: dict[str, dict] = {}
    for t in entries:
        if getattr(t, "size", None) is None:
            continue
        out[t.path] = _entry_identity(t)
    return out


def _repo_head_sha() -> str:
    return _get_api().repo_info(SRC, repo_type="dataset").sha


def _create_commit_delete(batch: list[str], head_sha: str, prefix: str) -> None:
    """One explicit-path delete commit, CAS'd on ``parent_commit`` (plan SS4.6 step 9)."""
    ops = [CommitOperationDelete(path_in_repo=p) for p in batch]
    # NO_RETRY: CAS delete — plan SS4.6 step 9 bans blind retry (caller re-derives)
    _get_api().create_commit(
        repo_id=SRC,
        repo_type="dataset",
        operations=ops,
        commit_message=f"#2332 repack: delete {len(batch)} verified-packed originals ({prefix})",
        parent_commit=head_sha,
    )


def _dual_digest(path: Path) -> tuple[str, str]:
    """(sha256 hexdigest, git-blob sha1 hexdigest) in one streaming read."""
    size = path.stat().st_size
    h256 = hashlib.sha256()
    h1 = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h256.update(chunk)
            h1.update(chunk)
    return h256.hexdigest(), h1.hexdigest()


def _ignored(rel_posix: str) -> bool:
    return any(fnmatch.fnmatch(rel_posix, pat) for pat in IGNORE_PATTERNS)


def _count_upload_root(up_root: Path) -> int:
    n = 0
    for f in up_root.rglob("*"):
        if f.is_file() and not _ignored(f.relative_to(up_root).as_posix()):
            n += 1
    return n


# ─── steps (plan SS4.6, numbered) ────────────────────────────────────────────


def step_snapshot(prefix: str, st: dict) -> tuple[str, dict[str, dict]]:
    """Step 2: revision-pinned identity snapshot -> src_map sidecar."""
    rec = st["prefixes"].setdefault(prefix, {})
    steps = rec.setdefault("steps", {})
    if steps.get("snapshot", {}).get("done") and _srcmap_path(prefix).is_file():
        src_map = _load_srcmap(prefix)
        log(f"{prefix}: snapshot already recorded ({len(src_map):,} files) — resume")
        return rec["src_revision"], src_map
    gate_mover()  # Hub READ — the marker's letter: gate before ANY repack step
    src_revision = _repo_head_sha()
    src_map = _list_prefix(prefix, revision=src_revision)
    if not src_map:
        raise PrefixIncomplete(f"{prefix}: empty listing at pinned revision {src_revision}")
    collisions = [p for p in src_map if p.startswith(f"{prefix}/__packed__/")]
    if collisions:
        raise VerifyFailure(
            f"{prefix}: {len(collisions)} pre-existing paths under __packed__/ — collision guard"
        )
    STAGE.mkdir(parents=True, exist_ok=True)
    _srcmap_path(prefix).write_text(json.dumps(src_map, sort_keys=True))
    # Fresh snapshot (review r2): every artifact/checkpoint record a prior
    # partial run wrote described the OLD pinned revision — reset them ALL
    # before recording the new pin, so no stale packed_files / chunk record /
    # step checkpoint can be consumed against this snapshot.
    rec["packed_files"] = {}
    for stale_key in ("chunks", "kept_stale", "kept_new", "mode"):
        rec.pop(stale_key, None)
    steps.clear()
    rec["src_revision"] = src_revision
    rec["snapshot_ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["n_files"] = len(src_map)
    rec["total_bytes"] = sum(e["size"] for e in src_map.values())
    steps["snapshot"] = {"done": True, "run_id": RUN_ID}
    _save_state(st)
    log(
        f"{prefix}: snapshot pinned at {src_revision[:12]} — {len(src_map):,} files, "
        f"{rec['total_bytes'] / 1e9:.2f} GB"
    )
    return src_revision, src_map


def step_download(prefix: str, st: dict, src_map: dict, paths: list[str]) -> None:
    """Step 3: pinned download, 4 workers, backoff to 60 s; failure preserves staging."""
    _run_hub_step_gates(prefix)
    rec = st["prefixes"][prefix]
    src_revision = rec["src_revision"]
    dest = STAGE / f"src_{prefix}"
    dest.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    errs: Counter = Counter()

    def one(p: str):
        # Per-unit durable result = the staged file itself; resume predicate:
        # a present, size-matching staged copy is skipped without a Hub call.
        # Third element: did this unit FETCH (attempt to write fresh bytes)?
        local = dest / p
        if local.is_file() and local.stat().st_size == src_map[p]["size"]:
            return None, "", False
        last = ""
        for attempt in range(MAX_ATTEMPTS):
            try:
                # NO_RETRY: own 6-attempt backoff-to-60s loop (measured mover constraint)
                hf_hub_download(
                    repo_id=SRC,
                    filename=p,
                    repo_type="dataset",
                    revision=src_revision,
                    local_dir=str(dest),
                    etag_timeout=60,
                )
                return None, "", True
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:120]}"
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(min(60, 2 ** (attempt + 1)))
        return p, last, True

    t0 = time.time()
    fetched: list[str] = []
    with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
        futs = {ex.submit(one, p): p for p in paths}
        n = 0
        for f in as_completed(futs):
            bad, msg, did_fetch = f.result()
            if bad:
                failed.append(bad)
                errs[msg] += 1
            if did_fetch:
                fetched.append(futs[f])
            n += 1
            # Per-unit progress line (code-style intra-phase contract, T2 > ~50).
            log(
                f"[download] unit {n:,}/{len(paths):,} {bad or futs[f]} "
                f"elapsed={time.time() - t0:.0f}s"
                + (f" (failed so far: {len(failed)})" if failed else "")
            )
    if fetched:
        # Upstream artifact change (review r2): fresh bytes landed (or were
        # attempted) for these units — evict them from the verified-units
        # sidecar and invalidate every downstream step checkpoint, so they
        # must re-verify / re-pack. Persist BEFORE the failure raise below.
        sidecar = _load_staged_sidecar(prefix, src_revision)
        if sidecar is not None:
            for p in fetched:
                sidecar.pop(p, None)
            _write_staged_hashes(prefix, src_revision, sidecar)
        _invalidate_downstream(rec, "download")
        _save_state(st)
    if failed:
        (STAGE / f"{prefix}.download_failed.json").write_text(json.dumps(failed, indent=1))
        for msg, c in errs.most_common(5):
            log(f"      x{c}  {msg}")
        raise PrefixIncomplete(
            f"{prefix}: {len(failed)} downloads failed — staging PRESERVED, source untouched"
        )
    rec.setdefault("steps", {})["download"] = {
        "done": True,
        "run_id": RUN_ID,
        "src_revision": src_revision,
    }
    _save_state(st)
    log(f"    download complete ({len(paths):,} files)")


def step_staged_verify(prefix: str, st: dict, src_map: dict, paths: list[str]) -> dict[str, str]:
    """Step 4: every staged file matches size AND content identity vs the snapshot.

    LFS entries: local sha256 == lfs.sha256. Non-LFS entries: local git-blob
    sha1 == blob_id, contingent on A7 (probed once; on a demonstrated
    non-git-sha1 blob_id the fallback is size + hf_hub_download's own etag
    check, recorded as a STATED DOWNGRADE in the run log).

    Resume contract (review r2): the sidecar holds ONLY verified units and is
    fingerprinted by src_revision (_load_staged_sidecar discards anything
    else). Resumed units are re-adjudicated from their STORED digests — cheap
    dict compares, no re-hash — so a bad unit re-fails on EVERY retry instead
    of riding a stale checkpoint. Fresh units are digested, verified, and
    inserted only on PASS; fresh non-LFS units under an undecided A7 are held
    PENDING (never persisted) until the batch verdict resolves. Returns
    {path: sha256} for the pack step."""
    rec = st["prefixes"][prefix]
    src_revision = rec["src_revision"]
    src_root = STAGE / f"src_{prefix}"
    sidecar = _load_staged_sidecar(prefix, src_revision) or {}
    a7 = st.get("blob_id_is_git_sha1")
    nonlfs_probe: list[tuple[str, bool]] = []  # (path, sha1_matches)
    pending_nonlfs: dict[str, dict] = {}  # fresh non-LFS digests awaiting the A7 verdict
    t0 = time.time()
    n = 0
    n_resumed = 0
    n_fresh = 0
    for p in sorted(paths):
        local = src_root / p
        if not local.is_file():
            raise VerifyFailure(f"{prefix}: staged file missing: {p}")
        ent = src_map[p]
        if local.stat().st_size != ent["size"]:
            raise VerifyFailure(
                f"{prefix}: size mismatch {p}: staged {local.stat().st_size} != hub {ent['size']}"
            )
        n += 1
        stored = sidecar.get(p)
        if stored is not None:
            # Resumed unit: re-adjudicate identity from the STORED digests (no
            # re-hash) — a unit that would fail today fails NOW (review r2).
            n_resumed += 1
            if ent["lfs"]:
                if stored["sha256"] != ent["oid"]:
                    raise VerifyFailure(f"{prefix}: LFS sha256 mismatch on resumed unit {p}")
            elif a7 is True:
                if stored["gitsha1"] != ent["oid"]:
                    raise VerifyFailure(f"{prefix}: git-blob sha1 mismatch on resumed unit {p}")
            elif a7 is None:
                nonlfs_probe.append((p, stored["gitsha1"] == ent["oid"]))
            # a7 is False: size (re-checked above) + etag is the STATED DOWNGRADE.
            continue
        n_fresh += 1
        sha256, gitsha1 = _dual_digest(local)
        if ent["lfs"]:
            if sha256 != ent["oid"]:
                raise VerifyFailure(f"{prefix}: LFS sha256 mismatch on {p}")
            sidecar[p] = {"sha256": sha256, "gitsha1": gitsha1}
        elif a7 is True:
            if gitsha1 != ent["oid"]:
                raise VerifyFailure(f"{prefix}: git-blob sha1 mismatch on {p}")
            sidecar[p] = {"sha256": sha256, "gitsha1": gitsha1}
        elif a7 is False:
            # STATED DOWNGRADE bar: size (checked above) + etag — verified.
            sidecar[p] = {"sha256": sha256, "gitsha1": gitsha1}
        else:
            # A7 undecided: verdict unknown — hold PENDING, never persist an
            # unverified unit into the sidecar (review r2).
            nonlfs_probe.append((p, gitsha1 == ent["oid"]))
            pending_nonlfs[p] = {"sha256": sha256, "gitsha1": gitsha1}
        # Per-unit progress line (code-style intra-phase contract, T2 > ~50).
        log(f"[staged-verify] unit {n:,}/{len(paths):,} {p} elapsed={time.time() - t0:.0f}s")
        if n_fresh % 2000 == 0:
            # Incremental durable checkpoint — VERIFIED units only.
            _write_staged_hashes(prefix, src_revision, sidecar)
    if n_resumed:
        log(f"    [staged-verify] resumed past {n_resumed:,} previously verified units")
    # A7 disposition: decided ONCE (state-global), applied to every non-LFS file.
    if nonlfs_probe and a7 is None:
        matches = [ok for _, ok in nonlfs_probe]
        if all(matches):
            a7 = True
        elif not any(matches):
            a7 = False
            log(
                "    STATED DOWNGRADE (A7): blob_id is NOT git-blob sha1 on this repo — "
                "non-LFS staged verify degrades to size + hf_hub_download etag check"
            )
        else:
            bad = [p for p, ok in nonlfs_probe if not ok][:5]
            raise VerifyFailure(f"{prefix}: MIXED blob_id sha1 agreement (corruption?): {bad}")
        st["blob_id_is_git_sha1"] = a7
    # Batch verdict resolved without a raise: every pending fresh non-LFS unit
    # is now verified (a7 True == all matched; a7 False == the stated
    # size+etag downgrade) — merge them into the verified-units sidecar.
    sidecar.update(pending_nonlfs)
    _write_staged_hashes(prefix, src_revision, sidecar)
    if n_fresh:
        # Fresh hashing occurred: any pack/upload/verify checkpoint from a
        # prior partial run may predate these units — invalidate downstream.
        _invalidate_downstream(rec, "staged_verify")
    rec.setdefault("steps", {})["staged_verify"] = {
        "done": True,
        "run_id": RUN_ID,
        "src_revision": src_revision,
    }
    _save_state(st)
    log(f"    staged identity verify PASS ({len(paths):,} files)")
    return {p: sidecar[p]["sha256"] for p in paths}


def _shard_name(shard_idx: int, chunk_idx: int | None) -> str:
    if chunk_idx is None:
        return f"shard-{shard_idx:05d}.tar"
    return f"shard-c{chunk_idx}-{shard_idx:05d}.tar"


def step_pack(
    prefix: str,
    st: dict,
    src_map: dict,
    paths: list[str],
    staged_hashes: dict[str, str],
    chunk_idx: int | None,
    write_manifest: bool,
) -> dict:
    """Step 5: stream files into tar shards under the ISOLATED per-prefix
    upload root; write the sidecar index (+ manifest for whole-prefix mode)."""
    rec = st["prefixes"][prefix]
    # Re-packing MUTATES the pack artifacts (review r2): drop the prior
    # whole-mode pack record (rebuilt from scratch below; chunked mode
    # accumulates per-chunk keys, overwritten per chunk) and invalidate every
    # downstream checkpoint BEFORE the first shard write, so a crash mid-pack
    # can never leave stale local-verify/upload/hub-verify checkpoints beside
    # half-rebuilt shards.
    if chunk_idx is None:
        rec["packed_files"] = {}
    _invalidate_downstream(rec, "pack")
    _save_state(st)
    src_root = STAGE / f"src_{prefix}"
    packed_dir = STAGE / f"up_{prefix}" / prefix / "__packed__"
    packed_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}
    shard_files: list[Path] = []
    tar = None
    shard_idx = -1
    cur_bytes = 0
    n = 0
    t0 = time.time()
    try:
        for p in sorted(paths):
            if tar is None or cur_bytes >= SHARD_TARGET_BYTES:
                if tar is not None:
                    tar.close()
                shard_idx += 1
                shard_path = packed_dir / _shard_name(shard_idx, chunk_idx)
                shard_files.append(shard_path)
                tar = tarfile.open(shard_path, "w", format=tarfile.PAX_FORMAT)
                cur_bytes = 0
            local = src_root / p
            ti = tarfile.TarInfo(name=p)
            ti.size = local.stat().st_size
            ti.mtime = 0
            ti.mode = 0o644
            with local.open("rb") as f:
                tar.addfile(ti, f)
            index[p] = {
                "shard": shard_files[-1].name,
                "offset": -1,  # placeholder — re-derived from the READ path below
                "size": ti.size,
                "sha256": staged_hashes[p],
            }
            cur_bytes += ti.size
            n += 1
            # Per-unit progress line (code-style intra-phase contract, T2 > ~50).
            log(
                f"[pack] unit {n:,}/{len(paths):,} {p} shard={shard_files[-1].name} "
                f"elapsed={time.time() - t0:.0f}s"
            )
    finally:
        if tar is not None:
            tar.close()
    # TarInfo.offset_data is populated only on the READ path — write-side
    # addfile() leaves it 0 (verified on CPython 3.11.15) — so re-derive every
    # member's data offset by re-opening each closed shard for read: the
    # recorded offsets then carry exactly the semantics the accessor
    # (orchestrate/packed_prefix.py) cross-checks at extraction time.
    for sp in shard_files:
        with tarfile.open(sp, "r:") as rt:
            for m in rt.getmembers():
                index[m.name]["offset"] = m.offset_data
    no_offset = [p for p, e in index.items() if e["offset"] < 0]
    if no_offset:
        raise VerifyFailure(
            f"{prefix}: {len(no_offset)} packed members missing read-side offsets "
            f"(e.g. {no_offset[:3]})"
        )
    index_name = "index.json" if chunk_idx is None else f"index-c{chunk_idx}.json"
    (packed_dir / index_name).write_text(json.dumps(index, sort_keys=True, separators=(",", ":")))
    shard_sha: dict[str, str] = {}
    packed_files: dict[str, dict] = {}
    for sp in shard_files:
        s256, s1 = _dual_digest(sp)
        shard_sha[sp.name] = s256
        packed_files[f"{prefix}/__packed__/{sp.name}"] = {
            "size": sp.stat().st_size,
            "sha256": s256,
            "gitsha1": s1,
        }
    for name in [index_name]:
        fp = packed_dir / name
        s256, s1 = _dual_digest(fp)
        packed_files[f"{prefix}/__packed__/{name}"] = {
            "size": fp.stat().st_size,
            "sha256": s256,
            "gitsha1": s1,
        }
    if write_manifest:
        manifest = {
            "n_members": len(index),
            "total_bytes": sum(e["size"] for e in index.values()),
            "shard_sha256s": shard_sha,
            "src_revision": rec["src_revision"],
            "snapshot_ts": rec["snapshot_ts"],
        }
        mp = packed_dir / "manifest.json"
        mp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        s256, s1 = _dual_digest(mp)
        packed_files[f"{prefix}/__packed__/manifest.json"] = {
            "size": mp.stat().st_size,
            "sha256": s256,
            "gitsha1": s1,
        }
    rec.setdefault("packed_files", {}).update(packed_files)
    rec.setdefault("steps", {})[f"pack_c{chunk_idx}" if chunk_idx is not None else "pack"] = {
        "done": True,
        "run_id": RUN_ID,
        "src_revision": rec.get("src_revision"),
        "n_shards": len(shard_files),
        "k": len(packed_files),
    }
    _save_state(st)
    log(f"    packed {len(index):,} members into {len(shard_files)} shard(s) + {index_name}")
    return {"index": index, "k": len(packed_files), "packed_files": packed_files}


def step_local_verify(prefix: str, st: dict, src_map: dict, paths: list[str], index: dict) -> None:
    """Step 6: re-open each local shard; every member's bytes hash to the
    index sha256; member census == the chunk's file set exactly."""
    packed_dir = STAGE / f"up_{prefix}" / prefix / "__packed__"
    seen: dict[str, int] = {}
    shards = sorted({e["shard"] for e in index.values()})
    t0 = time.time()
    for shard in shards:
        with tarfile.open(packed_dir / shard, "r") as tar:
            for m in tar:
                fobj = tar.extractfile(m)
                if fobj is None:
                    raise VerifyFailure(f"{prefix}: non-regular tar member {m.name} in {shard}")
                h = hashlib.sha256()
                for chunk in iter(lambda: fobj.read(1 << 20), b""):
                    h.update(chunk)
                ent = index.get(m.name)
                if ent is None:
                    raise VerifyFailure(f"{prefix}: tar member {m.name} not in index")
                if h.hexdigest() != ent["sha256"] or m.size != ent["size"]:
                    raise VerifyFailure(f"{prefix}: member hash/size mismatch: {m.name}")
                if m.offset_data != ent["offset"]:
                    raise VerifyFailure(f"{prefix}: member offset mismatch: {m.name}")
                seen[m.name] = m.size
                # Per-unit progress line (code-style intra-phase contract, T2 > ~50).
                log(
                    f"[local-verify] unit {len(seen):,}/{len(index):,} {m.name} "
                    f"elapsed={time.time() - t0:.0f}s"
                )
    expected = {p: src_map[p]["size"] for p in paths}
    if seen != expected:
        missing = sorted(set(expected) - set(seen))[:5]
        extra = sorted(set(seen) - set(expected))[:5]
        raise VerifyFailure(
            f"{prefix}: member census mismatch (missing {len(set(expected) - set(seen))}, "
            f"extra {len(set(seen) - set(expected))}; e.g. missing={missing} extra={extra})"
        )
    rec = st["prefixes"][prefix]
    rec.setdefault("steps", {})["local_verify"] = {
        "done": True,
        "run_id": RUN_ID,
        "src_revision": rec.get("src_revision"),
    }
    _save_state(st)
    log(f"    local pack verify PASS ({len(seen):,} members across {len(shards)} shard(s))")


def step_upload(prefix: str, st: dict, k_expected: int) -> None:
    """Step 7: gates + census (+k) + G-cap, then upload_large_folder from the
    isolated per-prefix root. Exceptions logged, NEVER interpreted."""
    _run_hub_step_gates(prefix)
    up_root = STAGE / f"up_{prefix}"
    n = _count_upload_root(up_root)
    if n != k_expected:
        raise VerifyFailure(
            f"{prefix}: +k census: {n} files under {up_root} != recorded k={k_expected}"
        )
    fresh = gate_cap(k_expected)
    log(f"    G-cap OK: fresh count {fresh:,} + k={k_expected} <= {CAP_TOTAL - CAP_BUFFER:,}")
    # Secret upload gate (2026-08-18): the 2026-08-17 HF secret-scanning alert
    # was THIS script's upload of issue2224_screening tar shards — corpus text
    # with pasted third-party credentials, re-uploaded unscanned. Scan every
    # text member of the staged shards BEFORE upload_large_folder; fail-loud
    # OUTSIDE the transport try below (a finding is an outcome, not transport).
    # Remediation: scripts/scrub_secrets.py fix on the STAGED SOURCE, then
    # re-pack (index sha256s must be recomputed). Kill switch:
    # EPM_SECRET_UPLOAD_GATE=0.
    assert_upload_clean([up_root], what=f"issue2332_repack:{prefix}")
    try:
        _get_api().upload_large_folder(
            repo_id=SRC,
            folder_path=str(up_root),
            repo_type="dataset",
            ignore_patterns=IGNORE_PATTERNS,
            num_workers=UPLOAD_WORKERS,
            print_report_every=120,
        )
    except Exception as e:
        log(
            "    upload_large_folder raised (transport is NOT outcome; the Hub-side "
            f"verify decides): {type(e).__name__}: {str(e)[:160]}"
        )
    rec = st["prefixes"][prefix]
    rec.setdefault("steps", {})["upload"] = {
        "done": True,
        "run_id": RUN_ID,
        "src_revision": rec.get("src_revision"),
    }
    _save_state(st)


def step_hub_verify(prefix: str, st: dict) -> None:
    """Step 8: the ONLY success signal — scoped re-list of <prefix>/__packed__
    with per-entry identity match, branching on the entry's ACTUAL lfs attr.
    Read-only, but gated on G-mover anyway (the LIVE-CONCURRENCY marker's
    letter: before ANY repack step; review r1 minor)."""
    gate_mover()
    rec = st["prefixes"][prefix]
    expected: dict[str, dict] = rec.get("packed_files", {})
    if not expected:
        raise VerifyFailure(f"{prefix}: no packed_files recorded — nothing to verify")
    listing = _list_prefix(f"{prefix}/__packed__")
    a7 = st.get("blob_id_is_git_sha1")
    for repo_path, exp in sorted(expected.items()):
        ent = listing.get(repo_path)
        if ent is None:
            raise VerifyFailure(f"{prefix}: packed file MISSING on Hub: {repo_path}")
        if ent["size"] != exp["size"]:
            raise VerifyFailure(
                f"{prefix}: packed size mismatch {repo_path}: hub {ent['size']} != {exp['size']}"
            )
        if ent["lfs"]:
            if ent["oid"] != exp["sha256"]:
                raise VerifyFailure(f"{prefix}: packed LFS sha256 mismatch: {repo_path}")
        elif a7 is not True:
            # A7 False OR undecided (review r2 minor: an all-LFS-source prefix
            # reaches here with A7 never probed — a gitsha1 compare would then
            # rest on an unproven equivalence): byte-compare is always sound.
            _redownload_byte_compare(prefix, repo_path, exp)
        elif ent["oid"] != exp["gitsha1"]:
            raise VerifyFailure(f"{prefix}: packed git-blob sha1 mismatch: {repo_path}")
    rec.setdefault("steps", {})["hub_verify"] = {
        "pass": True,
        "run_id": RUN_ID,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(expected),
    }
    _save_state(st)
    log(f"    Hub-side identity verify PASS ({len(expected)} packed files)")


def _redownload_byte_compare(prefix: str, repo_path: str, exp: dict) -> None:
    """A7 fallback for non-LFS sidecars: re-download + byte-compare (trivial bytes)."""
    tmp = STAGE / f"verify_tmp_{prefix}"
    tmp.mkdir(parents=True, exist_ok=True)

    def _fetch() -> str:
        # NO_RETRY: token-only waiver — this thunk IS retried via _retry_listing
        return hf_hub_download(
            repo_id=SRC, filename=repo_path, repo_type="dataset", local_dir=str(tmp)
        )

    local = Path(_retry_listing(_fetch, f"redownload:{repo_path}"))
    s256, _ = _dual_digest(local)
    if s256 != exp["sha256"]:
        raise VerifyFailure(f"{prefix}: re-download byte-compare mismatch: {repo_path}")
    shutil.rmtree(tmp, ignore_errors=True)


def derive_delete_set(
    src_map: dict[str, dict],
    current: dict[str, dict],
    prefix: str,
    restrict: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Step 9 eligibility: delete = in snapshot AND present AND current oid ==
    snapshot oid (identity equality, never size) AND not under __packed__/.
    Same-size rewrites land in `stale` (kept); fleet-added paths in `added`
    (kept). `restrict` scopes deletion to one chunk's originals."""
    packed = f"{prefix}/__packed__/"
    deletable: list[str] = []
    stale: list[str] = []
    added: list[str] = []
    for path in sorted(current):
        if path.startswith(packed):
            continue
        snap = src_map.get(path)
        if snap is None:
            added.append(path)
            continue
        if current[path]["oid"] != snap["oid"]:
            stale.append(path)
            continue
        if restrict is not None and path not in restrict:
            continue
        deletable.append(path)
    return deletable, stale, added


def step_pilot_extra(prefix: str, st: dict, index: dict) -> None:
    """Plan SS4.6 'Pilot extra' (belt-and-braces, PILOT ONLY): re-download ONE
    shard from the Hub and byte-compare it against the local shard, then
    resolve 20 randomly sampled members (seeded RNG — reproducible) from the
    DOWNLOADED tar and byte-compare each against its staged original. Runs
    AFTER step 8 (Hub verify) and BEFORE any delete; any mismatch is a
    VerifyFailure — the prefix stays fully intact. P5/P6 deliberately skip
    this (plan-stated asymmetry: remote-hash equality + local member verify
    is strictly stronger once the pilot has proven the end-to-end path)."""
    rec = st["prefixes"][prefix]
    packed_dir = STAGE / f"up_{prefix}" / prefix / "__packed__"
    src_root = STAGE / f"src_{prefix}"
    shard = sorted({e["shard"] for e in index.values()})[0]
    tmp = STAGE / f"pilot_extra_{prefix}"
    tmp.mkdir(parents=True, exist_ok=True)

    def _fetch() -> str:
        # NO_RETRY: token-only waiver — this thunk IS retried via _retry_listing
        return hf_hub_download(
            repo_id=SRC,
            filename=f"{prefix}/__packed__/{shard}",
            repo_type="dataset",
            local_dir=str(tmp),
        )

    remote_copy = Path(_retry_listing(_fetch, f"pilot-extra:{shard}"))
    r256, _ = _dual_digest(remote_copy)
    l256, _ = _dual_digest(packed_dir / shard)
    if r256 != l256:
        raise VerifyFailure(
            f"{prefix}: pilot extra — re-downloaded shard {shard} != local shard "
            f"(hub {r256[:16]}... vs local {l256[:16]}...)"
        )
    members = [p for p in sorted(index) if index[p]["shard"] == shard]
    sample = random.Random(2332).sample(members, min(20, len(members)))
    with tarfile.open(remote_copy, "r:") as tar:
        for i, p in enumerate(sample):
            fh = tar.extractfile(p)
            if fh is None:
                raise VerifyFailure(f"{prefix}: pilot extra — member not extractable: {p}")
            if fh.read() != (src_root / p).read_bytes():
                raise VerifyFailure(f"{prefix}: pilot extra — member bytes != staged original: {p}")
            log(f"    [pilot-extra] unit {i + 1}/{len(sample)} {p} OK")
    shutil.rmtree(tmp, ignore_errors=True)
    rec.setdefault("steps", {})["pilot_extra"] = {"done": True, "run_id": RUN_ID, "shard": shard}
    _save_state(st)
    log(f"    pilot extra PASS: shard {shard} byte-identical + {len(sample)} members verified")


def _step_done(rec: dict, key: str) -> bool:
    """Resume predicate: step `key` is checkpoint-complete FOR THIS SNAPSHOT —
    recorded done AND fingerprinted by the same src_revision (a re-snapshot
    invalidates every later step's checkpoint). Consumers additionally verify
    the step's on-disk outputs exist before skipping."""
    s = rec.get("steps", {}).get(key) or {}
    return bool(s.get("done")) and s.get("src_revision") == rec.get("src_revision")


def _chunked_pilot_duty_pending(st: dict) -> bool:
    """ONE state machine for the plan SS7 gate-7 chunked-mode pilot duty
    (r3 reconciler, chunked-pilot-transfer-state): the SINGLE persisted state
    is duty SATISFACTION — ``st['chunked_pilot_extra_done'] = <prefix>``,
    written ONLY after ``step_pilot_extra`` PASSes on some chunked prefix's
    chunk 0. There is deliberately NO holder claim to release: while the duty
    is pending, EVERY chunked prefix's chunk 0 carries it (a holder that
    gate-skipped / failed / re-planned whole simply never satisfied it, so
    the duty re-attaches to the next chunked prefix by construction). Both
    ``run_prefix`` (fire pilot-extra?) and the ``main`` run loop (abort after
    an unpiloted chunked failure?) read exactly this predicate."""
    return not st.get("chunked_pilot_extra_done")


def _sizing_mode_hints(sizing_path: Path | None = None) -> dict[str, str]:
    """Expected per-prefix mode ('whole' | 'chunked' | 'unknown') from the
    committed P2 sizing.json — best-effort ({} when absent/unparseable).
    Backstop for the main-loop chunked-pilot duty when a gate trip skipped
    run_prefix BEFORE its rec['mode'] assignment (Codex r2
    chunked-pilot-contract, arm 2). A 'refused-drift' (or any unrecognized)
    sizing row maps to 'unknown' — chunked-POSSIBLE for the duty check,
    never silently 'whole' (r3 review minor 3: a drift-refused chunked
    prefix must not evade the ran-and-failed abort via a wrong hint)."""
    p = (
        sizing_path
        if sizing_path is not None
        else Path(__file__).resolve().parent.parent / "eval_results" / "issue_2332" / "sizing.json"
    )
    try:
        sizing = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    out: dict[str, str] = {}
    for prefix, info in (sizing.get("prefixes") or {}).items():
        if isinstance(info, dict) and isinstance(info.get("mode"), str):
            m = info["mode"]
            if m.startswith("chunked"):
                out[prefix] = "chunked"
            elif m == "whole":
                out[prefix] = "whole"
            else:
                out[prefix] = "unknown"
    return out


def _delete_batch_size(st: dict) -> int:
    pilot_done = bool(st.get("prefixes", {}).get(PILOT_PREFIX, {}).get("done"))
    return DELETE_BATCH_STEADY if pilot_done else DELETE_BATCH_PILOT


def _assert_pack_present(
    prefix: str, current: dict[str, dict], expected_pack: dict, a7: bool | None
) -> None:
    """Per-batch pack-presence guard AT THE PINNED REVISION (review r1): every
    recorded packed file must be present, size-matched, and identity-matched
    in the SAME listing the delete set derives from — a pack modified or
    removed after step 8 REFUSES further deletes instead of orphaning
    originals. Non-LFS sidecars degrade to size+presence unless A7 proved
    blob_id == git-sha1 (their bytes were already step-8-verified; the tar
    shards themselves are LFS and always identity-checked)."""
    if not expected_pack:
        raise VerifyFailure(f"{prefix}: no packed_files recorded — refuse to delete")
    for rp, exp in sorted(expected_pack.items()):
        ent = current.get(rp)
        if ent is None:
            raise VerifyFailure(f"{prefix}: packed file MISSING at pinned revision: {rp}")
        if ent["size"] != exp["size"]:
            raise VerifyFailure(f"{prefix}: packed size changed at pinned revision: {rp}")
        if ent["lfs"]:
            if ent["oid"] != exp["sha256"]:
                raise VerifyFailure(f"{prefix}: packed identity changed at pinned revision: {rp}")
        elif a7 is True and ent["oid"] != exp["gitsha1"]:
            raise VerifyFailure(f"{prefix}: packed identity changed at pinned revision: {rp}")


def run_delete(
    prefix: str,
    st: dict,
    *,
    src_map: dict | None = None,
    restrict: set[str] | None = None,
    final_assert: bool = True,
    batch_size: int | None = None,
) -> int:
    """Steps 9-10: identity-keyed batched deletes under a parent_commit CAS.

    State machine: unreachable without a recorded hub-verify PASS; a PASS
    recorded by a PRIOR process run forces a re-verify first. Gates re-run
    before EACH batch. REVISION BINDING (review r1): HEAD is read FIRST,
    the re-list is pinned to THAT revision, the pack presence/identity is
    asserted in that same listing, and the CAS `parent_commit` is the SAME
    sha — a commit landing after the head-read 412s and forces a fresh
    pinned re-derive, so the delete set can never derive from an older tree
    than its CAS parent. On ANY commit exception (412 CAS conflict, 504,
    ...): log, NEVER interpret, re-list + re-derive — never a blind retry
    of the same batch. Returns the REALIZED freed-slot count derived from
    listings (restart- and 504-ambiguity-safe), not the in-process commit
    tally."""
    rec = st["prefixes"].setdefault(prefix, {})
    hv = rec.get("steps", {}).get("hub_verify") or {}
    if not hv.get("pass"):
        raise GateRefusal(
            f"{prefix}: G-verify-before-delete: delete unreachable without a recorded "
            "hub-verify PASS"
        )
    if hv.get("run_id") != RUN_ID:
        log(f"{prefix}: restart detected (hub-verify PASS from a prior run) — RE-RUNNING verify")
        step_hub_verify(prefix, st)
    if src_map is None:
        src_map = _load_srcmap(prefix)
    bs = batch_size if batch_size is not None else _delete_batch_size(st)
    kept_stale: set[str] = set(rec.get("kept_stale", []))
    kept_new: set[str] = set(rec.get("kept_new", []))
    expected_pack: dict = rec.get("packed_files", {})
    n_deleted = 0
    consecutive_failures = 0
    prev_n_deletable: int | None = None
    t0 = time.time()
    current: dict[str, dict] = {}
    while True:
        _run_hub_step_gates(prefix)
        head = _repo_head_sha()
        current = _list_prefix(prefix, revision=head)
        _assert_pack_present(prefix, current, expected_pack, st.get("blob_id_is_git_sha1"))
        deletable, stale, added = derive_delete_set(src_map, current, prefix, restrict=restrict)
        if prev_n_deletable is not None and len(deletable) < prev_n_deletable:
            # Progress observed via the re-list (a 504-masked LANDED commit
            # still shrinks the set) — reset the no-progress failure counter.
            consecutive_failures = 0
        prev_n_deletable = len(deletable)
        for p in stale:
            if p not in kept_stale:
                log(f"    REPACK-STALE (kept, NOT deleted — bytes not in the tar): {p}")
        for p in added:
            if p not in kept_new:
                log(f"    fleet-added since snapshot (kept): {p}")
        kept_stale |= set(stale)
        kept_new |= set(added)
        rec["kept_stale"] = sorted(kept_stale)
        rec["kept_new"] = sorted(kept_new)
        _save_state(st)
        if not deletable:
            break
        batch = deletable[:bs]
        try:
            _create_commit_delete(batch, head, prefix)
        except Exception as e:
            consecutive_failures += 1
            log(
                f"    delete commit exception #{consecutive_failures} (logged, never "
                f"interpreted; re-list + re-derive): {type(e).__name__}: {str(e)[:200]}"
            )
            if consecutive_failures >= MAX_CONSECUTIVE_DELETE_FAILURES:
                raise VerifyFailure(
                    f"{prefix}: {consecutive_failures} consecutive delete-commit failures with "
                    "no re-list-observed progress — fail loud (source + pack both intact)"
                ) from e
            continue
        consecutive_failures = 0
        n_deleted += len(batch)
        log(
            f"    [delete] batch of {len(batch):,} (this-run total {n_deleted:,}) "
            f"parent={head[:12]} elapsed={time.time() - t0:.0f}s"
        )
    if final_assert:
        # Codex r2 concern (delete-final-revision-binding): pin the completion
        # listing to a freshly-read head so the final assert + realized
        # accounting derive from ONE immutable tree, not a mutable main ref.
        final_head = _repo_head_sha()
        final = _list_prefix(prefix, revision=final_head)
        _assert_pack_present(prefix, final, expected_pack, st.get("blob_id_is_git_sha1"))
        packed = f"{prefix}/__packed__/"
        leftovers = [
            p
            for p in sorted(final)
            if not p.startswith(packed) and p not in kept_stale and p not in kept_new
        ]
        if leftovers:
            raise VerifyFailure(
                f"{prefix}: final assert FAILED — {len(leftovers)} unexplained non-packed "
                f"paths remain (e.g. {leftovers[:5]})"
            )
        log(
            f"    final re-list assert PASS: only __packed__/* + {len(kept_stale)} stale + "
            f"{len(kept_new)} fleet-added keepers remain"
        )
        current = final  # the freshest listing feeds the realized accounting
    # REALIZED accounting (review r1): derive freed slots from the last listing
    # rather than the in-process commit tally — correct across restarts and
    # 504-masked-but-landed commits (the exact ambiguity this loop tolerates).
    scope = restrict if restrict is not None else set(src_map)
    realized = sum(1 for p in scope if p in src_map and p not in current)
    rec.setdefault("steps", {})["delete"] = {
        "done": True,
        "run_id": RUN_ID,
        "n_deleted_this_run": n_deleted,
        "n_deleted_realized": realized,
    }
    _save_state(st)
    return realized


# ─── chunk planning + per-prefix pipeline ────────────────────────────────────


def plan_chunks(src_map: dict[str, dict], prefix: str) -> list[list[str]] | None:
    """None = whole-prefix mode. Chunked mode partitions by subdirectory
    (fallback: lexicographic ranges) so per-chunk staged footprint
    (2 x chunk bytes) stays <= CHUNK_STAGED_MAX_BYTES (plan SS4.5)."""
    total = sum(e["size"] for e in src_map.values())
    if 2 * total <= VM_FOOTPRINT_MAX_BYTES:
        return None
    max_chunk_bytes = CHUNK_STAGED_MAX_BYTES // 2
    groups: dict[str, list[str]] = {}
    for p in sorted(src_map):
        rel = p[len(prefix) + 1 :] if p.startswith(prefix + "/") else p
        key = rel.split("/", 1)[0] if "/" in rel else "__root__"
        groups.setdefault(key, []).append(p)
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for key in sorted(groups):
        paths = groups[key]
        gbytes = sum(src_map[p]["size"] for p in paths)
        if gbytes > max_chunk_bytes:
            # One oversized subdir: lexicographic size-bounded runs.
            if cur:
                chunks.append(cur)
                cur, cur_bytes = [], 0
            run: list[str] = []
            run_bytes = 0
            for p in paths:
                if run and run_bytes + src_map[p]["size"] > max_chunk_bytes:
                    chunks.append(run)
                    run, run_bytes = [], 0
                run.append(p)
                run_bytes += src_map[p]["size"]
            if run:
                chunks.append(run)
            continue
        if cur and cur_bytes + gbytes > max_chunk_bytes:
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.extend(paths)
        cur_bytes += gbytes
    if cur:
        chunks.append(cur)
    return chunks


def _cleanup_staging(prefix: str) -> None:
    shutil.rmtree(STAGE / f"src_{prefix}", ignore_errors=True)
    shutil.rmtree(STAGE / f"up_{prefix}", ignore_errors=True)


def run_prefix(prefix: str, st: dict) -> str:
    """Full per-prefix chain (plan SS4.6). Returns 'done' | 'deferred' | 'incomplete'."""
    rec = st["prefixes"].setdefault(prefix, {})
    if rec.get("done"):
        log(f"{prefix}: already done — skip")
        return "done"
    gate_mover()  # global condition: a trip here propagates and stops the whole run
    try:
        gate_writer(prefix)
        if PREFIXES[prefix][0] == 1739:
            gate_1739()
    except GateRefusal as e:
        # Per-prefix trip (plan SS4.6 step 1): skip/defer + record, continue.
        rec["deferred" if PREFIXES[prefix][0] == 1739 else "skipped"] = (
            f"{prefix}, {PREFIXES[prefix][1]:,} slots, re-runnable via "
            f"`run --prefix {prefix}` — {e}"
        )
        _save_state(st)
        log(f"{prefix}: {'DEFERRED' if PREFIXES[prefix][0] == 1739 else 'SKIPPED'} — {e}")
        return "deferred" if PREFIXES[prefix][0] == 1739 else "skipped"
    src_revision, src_map = step_snapshot(prefix, st)
    chunks = plan_chunks(src_map, prefix)
    if chunks is None:
        rec["mode"] = "whole"
        _staging_headroom_assert(2 * rec["total_bytes"])
        paths = sorted(src_map)
        packed_dir = STAGE / f"up_{prefix}" / prefix / "__packed__"
        # Checkpoint CONSUMPTION (review r1+r2): every skip predicate is
        # artifact-complete AND identity-complete. download is ALWAYS called —
        # its per-unit size-keyed resume makes it an idempotent completeness
        # sweep (and it evicts+invalidates on any fresh fetch). upload is
        # ALWAYS called — upload_large_folder is internally resumable, and
        # consuming its checkpoint wedges a partial upload behind a forever-
        # failing hub_verify (review r2 minor 1). hub_verify is NEVER skipped (its
        # run_id-keyed restart re-verify is the delete path's own law).
        step_download(prefix, st, src_map, paths)
        sidecar = _load_staged_sidecar(prefix, src_revision)
        if _step_done(rec, "staged_verify") and sidecar is not None and set(paths) <= set(sidecar):
            staged = {p: sidecar[p]["sha256"] for p in paths}
            log(
                f"{prefix}: staged-verify checkpoint consumed — {len(staged):,} verified "
                "hashes loaded (fingerprinted sidecar covers all units)"
            )
        else:
            staged = step_staged_verify(prefix, st, src_map, paths)
        if (
            _step_done(rec, "pack")
            and rec.get("packed_files")
            and (packed_dir / "index.json").is_file()
            and all((STAGE / f"up_{prefix}" / rp).is_file() for rp in rec["packed_files"])
        ):
            packed = {
                "index": json.loads((packed_dir / "index.json").read_text()),
                "k": rec["steps"]["pack"]["k"],
                "packed_files": rec["packed_files"],
            }
            log(f"{prefix}: pack checkpoint consumed — skip (shards + index on disk)")
        else:
            packed = step_pack(prefix, st, src_map, paths, staged, None, write_manifest=True)
        if _step_done(rec, "local_verify"):
            log(f"{prefix}: local-verify checkpoint consumed — skip")
        else:
            step_local_verify(prefix, st, src_map, paths, packed["index"])
        step_upload(prefix, st, packed["k"])
        step_hub_verify(prefix, st)
        if prefix == PILOT_PREFIX:
            step_pilot_extra(prefix, st, packed["index"])
        n_deleted = run_delete(prefix, st, src_map=src_map)
    else:
        log(f"{prefix}: CHUNKED mode — {len(chunks)} chunks")
        rec["mode"] = "chunked"
        n_deleted = 0
        merged_index: dict[str, dict] = {}
        all_shard_sha: dict[str, str] = {}
        for ci, paths in enumerate(chunks):
            ckey = f"chunk_{ci}"
            if rec.get("chunks", {}).get(ckey, {}).get("done"):
                crec = rec["chunks"][ckey]
                log(f"{prefix}: {ckey} already done — skip")
                merged_index.update(json.loads((STAGE / f"index_{prefix}_c{ci}.json").read_text()))
                # Restore the chunk's persisted shard hashes + realized count so
                # a resumed run's manifest and slots_freed stay correct.
                all_shard_sha.update(crec.get("shard_sha256s", {}))
                n_deleted += crec.get("n_deleted_realized", 0)
                continue
            chunk_bytes = sum(src_map[p]["size"] for p in paths)
            _staging_headroom_assert(2 * chunk_bytes)
            step_download(prefix, st, src_map, paths)
            staged = step_staged_verify(prefix, st, src_map, paths)
            packed = step_pack(prefix, st, src_map, paths, staged, ci, write_manifest=False)
            step_local_verify(prefix, st, src_map, paths, packed["index"])
            step_upload(prefix, st, packed["k"])
            step_hub_verify(prefix, st)
            # Chunked-mode pilot duty (plan SS7 gate 7 second clause; r3
            # reconciler): while UNSATISFIED, every chunked prefix's chunk 0
            # runs pilot-extra — satisfaction is the one persisted state,
            # written only on a pilot-extra PASS (a failure raises first, so
            # the duty stays pending and main's abort arm fires).
            if ci == 0 and (prefix == PILOT_PREFIX or _chunked_pilot_duty_pending(st)):
                step_pilot_extra(prefix, st, packed["index"])
                st["chunked_pilot_extra_done"] = prefix
                _save_state(st)
            chunk_realized = run_delete(
                prefix, st, src_map=src_map, restrict=set(paths), final_assert=False
            )
            n_deleted += chunk_realized
            merged_index.update(packed["index"])
            (STAGE / f"index_{prefix}_c{ci}.json").write_text(
                json.dumps(packed["index"], sort_keys=True, separators=(",", ":"))
            )
            chunk_shas: dict[str, str] = {}
            for rp, meta in packed["packed_files"].items():
                if rp.endswith(".tar"):
                    chunk_shas[Path(rp).name] = meta["sha256"]
            all_shard_sha.update(chunk_shas)
            rec.setdefault("chunks", {})[ckey] = {
                "done": True,
                "n_files": len(paths),
                "n_deleted_realized": chunk_realized,
                "shard_sha256s": chunk_shas,
            }
            _cleanup_staging(prefix)
            _save_state(st)
        # Final merged index.json + manifest.json (their own small gated upload).
        packed_dir = STAGE / f"up_{prefix}" / prefix / "__packed__"
        packed_dir.mkdir(parents=True, exist_ok=True)
        (packed_dir / "index.json").write_text(
            json.dumps(merged_index, sort_keys=True, separators=(",", ":"))
        )
        manifest = {
            "n_members": len(merged_index),
            "total_bytes": sum(e["size"] for e in merged_index.values()),
            "shard_sha256s": all_shard_sha,
            "src_revision": rec["src_revision"],
            "snapshot_ts": rec["snapshot_ts"],
        }
        (packed_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
        for name in ("index.json", "manifest.json"):
            s256, s1 = _dual_digest(packed_dir / name)
            rec.setdefault("packed_files", {})[f"{prefix}/__packed__/{name}"] = {
                "size": (packed_dir / name).stat().st_size,
                "sha256": s256,
                "gitsha1": s1,
            }
        _save_state(st)
        step_upload(prefix, st, k_expected=2)
        step_hub_verify(prefix, st)
        # The final pass's REALIZED count spans the whole src_map scope, so it
        # REPLACES (never adds to) the per-chunk accumulation.
        n_deleted = run_delete(prefix, st, src_map=src_map, final_assert=True)
    rec["done"] = {
        "files": rec["n_files"],
        "bytes": rec["total_bytes"],
        "slots_freed": n_deleted,
        "kept_stale": len(rec.get("kept_stale", [])),
        "kept_new": len(rec.get("kept_new", [])),
    }
    _cleanup_staging(prefix)
    _save_state(st)
    log(f"{prefix}: DONE — {n_deleted:,} slots freed")
    return "done"


# ─── sizing / count subcommands ──────────────────────────────────────────────


def cmd_count(_args: argparse.Namespace) -> int:
    gate_mover()
    n = _fresh_repo_count()
    log(f"full-repo file count (scoped-sum method): {n:,}")
    print(n)
    return 0


def cmd_sizing(args: argparse.Namespace) -> int:
    """P2 sizing pass (plan SS4.4): per-prefix listing + census + shard/chunk
    plan + full-repo before-count + inventory reconcile. Read-only on the Hub."""
    gate_mover()
    out: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mover_advisory": _mover_done_count_advisory(),
        "prefixes": {},
    }
    with ThreadPoolExecutor(max_workers=LIST_WORKERS) as ex:
        listings = dict(zip(PREFIXES, ex.map(lambda p: _list_prefix(p), PREFIXES), strict=True))
    for prefix, listing in listings.items():
        collisions = [p for p in listing if p.startswith(f"{prefix}/__packed__/")]
        if collisions:
            raise SystemExit(
                f"FATAL: {prefix} already has {len(collisions)} paths under __packed__/"
            )
        total = sum(e["size"] for e in listing.values())
        census = Counter(Path(p).suffix or "<none>" for p in listing)
        n_shards = max(1, math.ceil(total / SHARD_TARGET_BYTES))
        chunks = plan_chunks(listing, prefix)
        expected = PREFIXES[prefix][1]
        drift = len(listing) - expected
        drift_recheck: str | None = None
        if drift != 0:
            log(
                f"DRIFT on target prefix {prefix}: live {len(listing):,} != inventory "
                f"{expected:,} ({drift:+,}) — someone wrote to it; re-running the "
                "G-writer four-way check (plan SS4.4.4 fail-loud: a refusal REFUSES "
                "this prefix — recorded + re-runnable — never proceeds on a log line)"
            )
            try:
                gate_writer(prefix)
            except GateRefusal as e:
                log(f"{prefix}: sizing REFUSED on drift — {e}")
                out["prefixes"][prefix] = {
                    "files": len(listing),
                    "bytes": total,
                    "inventory_drift": drift,
                    "mode": "refused-drift",
                    "drift_refused": str(e),
                }
                continue
            drift_recheck = "writer-recheck-clean"
        if chunks is None:
            k = n_shards + 2  # shards + index.json + manifest.json
            chunk_plan = None
        else:
            # Chunked k (plan SS4.5): Σ per-chunk shards + one index-c<i>.json
            # per chunk + the final merged index.json + manifest.json.
            chunk_plan = []
            for ci, cpaths in enumerate(chunks):
                cbytes = sum(listing[p]["size"] for p in cpaths)
                chunk_plan.append(
                    {
                        "chunk": ci,
                        "files": len(cpaths),
                        "bytes": cbytes,
                        "shards": max(1, math.ceil(cbytes / SHARD_TARGET_BYTES)),
                    }
                )
            k = sum(c["shards"] for c in chunk_plan) + len(chunk_plan) + 2
        out["prefixes"][prefix] = {
            "files": len(listing),
            "bytes": total,
            "census": dict(census.most_common()),
            "max_file_bytes": max((e["size"] for e in listing.values()), default=0),
            "shards_planned": n_shards,
            "k": k,
            "chunk_plan": chunk_plan,
            "staged_footprint_bytes": 2 * total,
            "mode": "whole" if chunks is None else f"chunked({len(chunks)})",
            "inventory_drift": drift,
            "drift_writer_recheck": drift_recheck,
        }
    out["repo_count_before"] = _fresh_repo_count()
    out_path = Path(args.out) / "sizing.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True))
    log(f"sizing written to {out_path} (repo count before: {out['repo_count_before']:,})")
    return 0


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _prefix_arg(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument("--prefix", choices=sorted(PREFIXES), required=required)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("count", help="fresh full-repo file count (scoped-sum method)")
    p = sub.add_parser("sizing", help="P2 sizing pass -> sizing.json")
    p.add_argument("--out", default="eval_results/issue_2332", help="output dir for sizing.json")
    p = sub.add_parser("repack", help="steps 2-7 for one prefix (snapshot..upload)")
    _prefix_arg(p)
    p = sub.add_parser("verify", help="step 8: Hub-side identity verify")
    _prefix_arg(p)
    p = sub.add_parser("delete", help="steps 9-10: identity-keyed CAS deletes")
    _prefix_arg(p)
    p = sub.add_parser("run", help="full chain; all 8 prefixes in plan order when no --prefix")
    _prefix_arg(p, required=False)
    p = sub.add_parser("closeout", help="P7: fresh count -> before_after.json + final_state.json")
    p.add_argument("--out", default="eval_results/issue_2332", help="output dir")
    return ap


def cmd_closeout(args: argparse.Namespace) -> int:
    """P7 closeout (plan SS4.7(1)): fresh full-repo count -> before_after.json
    (before-count read from the committed sizing.json) + a durable
    final_state.json snapshot. repack_report.md stays orchestrator-composed
    prose. Read-only on the Hub."""
    gate_mover()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizing = json.loads((out_dir / "sizing.json").read_text())
    after = _fresh_repo_count()
    st = _load_state()
    before = sizing["repo_count_before"]
    ba = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_count_before": before,
        "repo_count_after": after,
        "slots_freed_total": before - after,
        "per_prefix": {p: st.get("prefixes", {}).get(p, {}).get("done") for p in PREFIXES},
    }
    (out_dir / "before_after.json").write_text(json.dumps(ba, indent=1, sort_keys=True))
    (out_dir / "final_state.json").write_text(json.dumps(st, indent=1, sort_keys=True))
    log(f"closeout: before {before:,} -> after {after:,} (freed {before - after:,})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.cmd == "count":
        return cmd_count(args)
    if args.cmd == "sizing":
        return cmd_sizing(args)
    if args.cmd == "closeout":
        return cmd_closeout(args)
    st = _load_state()
    st.setdefault("run_history", []).append(
        {
            "run_id": RUN_ID,
            "cmd": args.cmd,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    _save_state(st)
    if args.cmd == "repack":
        prefix = args.prefix
        rec = st["prefixes"].setdefault(prefix, {})
        _, src_map = step_snapshot(prefix, st)
        if plan_chunks(src_map, prefix) is not None:
            raise SystemExit(
                f"{prefix}: chunked mode required (staged footprint > "
                f"{VM_FOOTPRINT_MAX_BYTES / 1e9:.0f} GB) — use `run --prefix {prefix}`"
            )
        _staging_headroom_assert(2 * rec["total_bytes"])
        paths = sorted(src_map)
        step_download(prefix, st, src_map, paths)
        staged = step_staged_verify(prefix, st, src_map, paths)
        packed = step_pack(prefix, st, src_map, paths, staged, None, write_manifest=True)
        step_local_verify(prefix, st, src_map, paths, packed["index"])
        step_upload(prefix, st, packed["k"])
        return 0
    if args.cmd == "verify":
        step_hub_verify(args.prefix, st)
        return 0
    if args.cmd == "delete":
        run_delete(args.prefix, st)
        return 0
    if args.cmd == "run":
        targets = [args.prefix] if args.prefix else list(PREFIXES)
        full_order = args.prefix is None
        results: dict[str, str] = {}
        mode_hints = _sizing_mode_hints()
        for prefix in targets:
            try:
                results[prefix] = run_prefix(prefix, st)
            except (PrefixIncomplete, VerifyFailure) as e:
                log(f"{prefix}: INCOMPLETE — {e} (staging preserved, source untouched)")
                st["prefixes"].setdefault(prefix, {})["incomplete"] = str(e)
                _save_state(st)
                results[prefix] = "incomplete"
            except GateRefusal as e:
                if str(e).startswith("G-mover"):
                    log(f"ABORT run: {e} (mover is a global condition — no Hub work proceeds)")
                    raise
                # Mid-chain per-prefix trip: the #1739 prefix's sanctioned
                # outcome vocabulary is `deferred`, every other prefix `skipped`.
                key = "deferred" if PREFIXES[prefix][0] == 1739 else "skipped"
                log(f"{prefix}: gate trip mid-chain — {e} (staging preserved; {key})")
                st["prefixes"].setdefault(prefix, {})[key] = str(e)
                _save_state(st)
                results[prefix] = key
            # Pilot->fleet gate (plan SS7 gate 7): in full-order mode, fleet
            # prefixes NEVER start after a non-done pilot — a pilot failure is
            # fix-and-rerun on the pilot, and the run stops here.
            if full_order and prefix == PILOT_PREFIX and results[prefix] != "done":
                log(
                    f"ABORT run: pilot {PILOT_PREFIX} result={results[prefix]!r} — plan SS7 "
                    "gate 7: fleet prefixes do not start after a pilot failure "
                    f"(fix + re-run `run --prefix {PILOT_PREFIX}`, then `run`)"
                )
                return 1
            # Chunked-mode pilot duty (plan SS7 gate 7, second clause; r3
            # reconciler chunked-pilot-transfer-state): this check reads the
            # SAME persisted satisfaction flag run_prefix writes — never a
            # loop-local tracker. Mode comes from the rec when run_prefix
            # assigned it, else the sizing.json hint ('unknown' = a
            # refused-drift / unrecognized row, chunked-POSSIBLE; an
            # entry-gate trip skips run_prefix BEFORE rec['mode']). While the
            # duty is UNSATISFIED: skipped/deferred (nothing ran) and even a
            # done-without-recorded-pilot-extra chunked prefix TRANSFER the
            # duty to the next chunked prefix (whose chunk 0 then actually
            # runs pilot-extra — the run_prefix trigger keys on the same
            # flag); a chunked-possible prefix that RAN AND FAILED aborts
            # before larger chunked prefixes.
            mode = st["prefixes"].get(prefix, {}).get("mode") or mode_hints.get(prefix)
            if (
                full_order
                and _chunked_pilot_duty_pending(st)
                and mode in ("chunked", "unknown")
                and results[prefix] not in ("done", "skipped", "deferred")
            ):
                log(
                    f"ABORT run: chunked-mode prefix {prefix} "
                    f"result={results[prefix]!r} with the plan SS7 gate-7 "
                    "chunked-pilot duty UNSATISFIED (no pilot-extra PASS "
                    "recorded on any chunked chunk 0) — fix and re-run before "
                    "larger chunked prefixes"
                )
                return 1
        log(f"run complete: {results}")
        return 0 if all(v == "done" for v in results.values()) else 1
    raise SystemExit(f"unknown cmd {args.cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
