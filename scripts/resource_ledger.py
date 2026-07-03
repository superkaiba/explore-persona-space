#!/usr/bin/env python
"""Advisory VM CPU/RAM resource ledger (EPS workflow v2, plan §5).

Cross-session coordination for the shared VM's CPU + RAM: a phase about to run
non-trivial local compute CLAIMs the cores/RAM it will use; the ledger decides
whether the VM has headroom (live psutil usage + already-registered claims +
this claim under the 70% band) and, if not, prints a ROUTE-TO-POD advisory so
the caller sends the phase to its own CPU pod (`cpu-small`/`cpu-mid`/
`cpu-bigmem`) instead of piling onto the shared box.

This is ADVISORY only — it never signals a process, never enforces a limit, and
never blocks. The real per-task disk quotas (ext4 project ids, #681) and the
static thread caps (#847) are the hard bounds; this ledger complements them by
routing CPU/RAM-heavy PHASES off the VM before they land.

Design invariants (plan §5):

- **Read-decide-claim is atomic within ONE ``flock`` hold** on a sidecar
  ``<ledger>.lock`` (never the data file itself), so two concurrent sessions
  can't both read headroom, both decide "ok", and both claim past the band.
- **Claims carry a TTL + owning PID.** A stale (TTL-expired) or dead-PID claim
  is REAPED before any decision — a crashed session can never wedge routing.
  The 10-min autonomous-session watcher piggybacks the same reap
  (``EPM_DISABLE_VM_LEDGER_REAP`` kill switch).
- **A missing ledger fails TOWARD available** (a fresh empty ledger — the VM is
  presumed free). A CORRUPT ledger is RENAMED ASIDE with a WARN (never silently
  truncated), then treated as fresh — a garbled file must not wedge routing,
  but it also must not vanish without a trace.

``psutil`` (a project dep) reads the live totals + usage; the decision core is a
pure function tested without it.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import sys
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import psutil

# Default per-claim time-to-live: a phase's claim auto-expires after this many
# seconds so a crashed / forgotten claim can never permanently reserve headroom.
DEFAULT_TTL_S = 4 * 60 * 60  # 4h

# Headroom band: a claim that would push projected cores OR RAM past this
# fraction of the machine total routes to a CPU pod instead. 70% leaves margin
# for the OS + un-registered v1 sessions the live-usage term already counts.
DEFAULT_THRESHOLD = 0.70

# Every row must carry these keys; a row missing one is treated as corrupt (it
# is dropped by the reaper's malformed-row guard, never silently kept).
_REQUIRED_ROW_KEYS = frozenset(
    {"claim_id", "issue", "pid", "cores", "ram_gb", "phase", "created_iso", "ttl_s"}
)


def default_ledger_path() -> Path:
    """``~/.task-workflow/vm-ledger.json`` (the cross-session coordination dir)."""
    return Path.home() / ".task-workflow" / "vm-ledger.json"


def _lock_path(ledger_path: Path) -> Path:
    return ledger_path.with_name(ledger_path.name + ".lock")


@contextlib.contextmanager
def _ledger_lock(ledger_path: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on the sidecar lock for the whole read-
    decide-claim critical section. The lock file is never the data file (an
    atomic ``os.replace`` is the data-write unit; the lock only serialises
    concurrent claimers)."""
    lock_path = _lock_path(ledger_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _load_ledger(ledger_path: Path) -> dict:
    """Return the ledger dict, failing TOWARD available.

    A missing file yields a fresh empty ledger. A corrupt / non-object file is
    RENAMED ASIDE to ``<name>.corrupt-<utc-stamp>`` with a WARN to stderr and
    then treated as fresh — never silently truncated (so the garbage is
    recoverable for a human) and never allowed to wedge routing.
    """
    if not ledger_path.exists():
        return {"version": 1, "rows": []}
    try:
        data = json.loads(ledger_path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
            raise ValueError("ledger is not a {version, rows:[...]} object")
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        aside = ledger_path.with_name(f"{ledger_path.name}.corrupt-{stamp}")
        with contextlib.suppress(OSError):
            ledger_path.rename(aside)
        sys.stderr.write(
            f"WARN resource_ledger: corrupt ledger {ledger_path} renamed aside to "
            f"{aside} ({exc}); starting fresh (fail toward available).\n"
        )
        return {"version": 1, "rows": []}
    data.setdefault("version", 1)
    return data


def _atomic_write_ledger(ledger_path: Path, data: dict) -> None:
    """Atomic JSON write (tmp in the same dir + ``os.replace``)."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger_path.with_name(ledger_path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, ledger_path)


def _row_alive(row: dict, *, now: float, pid_alive: Callable[[int], bool]) -> bool:
    """True iff a ledger row is neither TTL-expired nor owned by a dead PID.

    A malformed row (missing a required key, or a non-parseable ts/ttl/pid) is
    treated as NOT alive so the reaper drops it — a garbage row must not pin
    headroom forever.
    """
    if _REQUIRED_ROW_KEYS - row.keys():
        return False
    try:
        pid = int(row["pid"])
        ttl_s = float(row["ttl_s"])
        created = datetime.fromisoformat(str(row["created_iso"])).timestamp()
    except (TypeError, ValueError, KeyError):
        return False
    if now - created >= ttl_s:
        return False
    return pid_alive(pid)


def reap_rows(
    rows: list[dict], *, now: float, pid_alive: Callable[[int], bool]
) -> tuple[list[dict], list[dict]]:
    """Split ``rows`` into ``(kept, reaped)`` — pure, no I/O.

    ``kept`` = rows still alive (TTL not expired AND PID still alive AND
    well-formed); ``reaped`` = expired / dead-PID / malformed rows. Injecting
    ``pid_alive`` + ``now`` makes the whole reap testable without a live process.
    """
    kept: list[dict] = []
    reaped: list[dict] = []
    for row in rows:
        (kept if _row_alive(row, now=now, pid_alive=pid_alive) else reaped).append(row)
    return kept, reaped


def decide_claim(
    *,
    this_cores: float,
    this_ram_gb: float,
    live_cores_used: float,
    live_ram_used_gb: float,
    claimed_cores: float,
    claimed_ram_gb: float,
    total_cores: float,
    total_ram_gb: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[bool, str]:
    """Pure headroom decision: ``(ok, reason)``.

    ``ok`` is False (ROUTE-TO-POD) when the PROJECTED load — live psutil usage
    (which already counts un-registered v1 sessions) PLUS the sum of live
    registered claims PLUS this claim — would push cores OR RAM past
    ``threshold`` of the machine total. Adding live-usage and claims is a
    deliberate conservative over-estimate (a claim whose phase already draws is
    counted twice), matching the plan's "reserve headroom for FUTURE phases"
    intent for an advisory router.
    """
    proj_cores = live_cores_used + claimed_cores + this_cores
    proj_ram = live_ram_used_gb + claimed_ram_gb + this_ram_gb
    core_cap = threshold * total_cores
    ram_cap = threshold * total_ram_gb
    over_cores = proj_cores > core_cap
    over_ram = proj_ram > ram_cap
    if over_cores or over_ram:
        which = []
        if over_cores:
            which.append(f"cores {proj_cores:.1f}/{core_cap:.1f} (cap {threshold:.0%})")
        if over_ram:
            which.append(f"RAM {proj_ram:.1f}/{ram_cap:.1f} GiB (cap {threshold:.0%})")
        return False, "; ".join(which)
    return (
        True,
        f"cores {proj_cores:.1f}/{core_cap:.1f}, RAM {proj_ram:.1f}/{ram_cap:.1f} GiB "
        f"(cap {threshold:.0%})",
    )


def _system_totals() -> tuple[float, float]:
    """``(total_cores, total_ram_gib)`` from psutil."""
    cores = float(psutil.cpu_count(logical=True) or os.cpu_count() or 1)
    total_ram_gib = psutil.virtual_memory().total / (1024**3)
    return cores, total_ram_gib


def _live_usage(cpu_interval: float = 0.1) -> tuple[float, float]:
    """``(cores_used, ram_used_gib)`` from a live psutil sample.

    ``cores_used`` = cpu-percent fraction x total cores; ``ram_used_gib`` =
    (total - available) in GiB. ``cpu_interval`` is a short blocking sample so
    the reading reflects the current moment, not since-boot averages.
    """
    total_cores, _ = _system_totals()
    cpu_frac = psutil.cpu_percent(interval=cpu_interval) / 100.0
    vm = psutil.virtual_memory()
    return cpu_frac * total_cores, (vm.total - vm.available) / (1024**3)


def _sum_claims(rows: list[dict]) -> tuple[float, float]:
    """``(sum_cores, sum_ram_gb)`` over well-formed rows."""
    cores = 0.0
    ram = 0.0
    for row in rows:
        with contextlib.suppress(TypeError, ValueError, KeyError):
            cores += float(row["cores"])
            ram += float(row["ram_gb"])
    return cores, ram


def _pid_alive(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid)
    except (ValueError, OverflowError):
        return False


def reap_ledger_file(
    ledger_path: Path | None = None,
    *,
    now: float | None = None,
    pid_alive: Callable[[int], bool] | None = None,
    apply: bool = True,
) -> list[dict]:
    """Reap expired / dead-PID / malformed rows from the on-disk ledger.

    Holds the flock for the read-reap-write. Returns the reaped rows (empty when
    the ledger is missing/empty). This is the entry point the autonomous-session
    watcher's ``vm_ledger_reap`` pass calls every 10 min. ``apply=False`` is a
    read-only report (compute what WOULD be reaped, never write) — the watcher's
    ``--dry-run`` path uses it.
    """
    ledger_path = default_ledger_path() if ledger_path is None else Path(ledger_path)
    now = now if now is not None else datetime.now(UTC).timestamp()
    alive = pid_alive or _pid_alive
    with _ledger_lock(ledger_path):
        data = _load_ledger(ledger_path)
        kept, reaped = reap_rows(data.get("rows", []), now=now, pid_alive=alive)
        if reaped and apply:
            data["rows"] = kept
            _atomic_write_ledger(ledger_path, data)
    return reaped


def claim(
    *,
    issue: int | str,
    cores: float,
    ram_gb: float,
    phase: str,
    ttl_s: float = DEFAULT_TTL_S,
    threshold: float = DEFAULT_THRESHOLD,
    force: bool = False,
    ledger_path: Path | None = None,
    now: float | None = None,
    pid: int | None = None,
    pid_alive: Callable[[int], bool] | None = None,
    live_usage: tuple[float, float] | None = None,
    totals: tuple[float, float] | None = None,
) -> tuple[bool, str, dict | None]:
    """Reap, decide, and (on headroom) record a claim — all under one flock.

    Returns ``(ok, reason, row)``: ``ok`` True when headroom allows AND the row
    was recorded (``row`` is the recorded dict); ``ok`` False when the claim
    would breach the band — the row is NOT recorded unless ``force=True`` (then
    ``ok`` is False but ``row`` is the recorded dict, so a caller can override
    the advisory and still track its footprint). ``live_usage`` / ``totals`` are
    injection hooks for tests (default: live psutil).
    """
    ledger_path = default_ledger_path() if ledger_path is None else Path(ledger_path)
    now = now if now is not None else datetime.now(UTC).timestamp()
    pid = os.getpid() if pid is None else pid
    alive = pid_alive or _pid_alive
    with _ledger_lock(ledger_path):
        data = _load_ledger(ledger_path)
        kept, _reaped = reap_rows(data.get("rows", []), now=now, pid_alive=alive)
        claimed_cores, claimed_ram = _sum_claims(kept)
        live_cores_used, live_ram_used = live_usage if live_usage is not None else _live_usage()
        total_cores, total_ram_gb = totals if totals is not None else _system_totals()
        ok, reason = decide_claim(
            this_cores=cores,
            this_ram_gb=ram_gb,
            live_cores_used=live_cores_used,
            live_ram_used_gb=live_ram_used,
            claimed_cores=claimed_cores,
            claimed_ram_gb=claimed_ram,
            total_cores=total_cores,
            total_ram_gb=total_ram_gb,
            threshold=threshold,
        )
        row: dict | None = None
        if ok or force:
            row = {
                "claim_id": uuid.uuid4().hex[:12],
                "issue": issue,
                "pid": pid,
                "cores": cores,
                "ram_gb": ram_gb,
                "phase": phase,
                "created_iso": datetime.fromtimestamp(now, UTC).isoformat(),
                "ttl_s": ttl_s,
            }
            kept.append(row)
            data["rows"] = kept
            _atomic_write_ledger(ledger_path, data)
        elif _reaped:
            # No claim recorded, but the reap changed the ledger — persist it.
            data["rows"] = kept
            _atomic_write_ledger(ledger_path, data)
    return ok, reason, row


def release(claim_id: str, *, ledger_path: Path | None = None) -> bool:
    """Remove one claim by id (idempotent). Returns True iff a row was removed."""
    ledger_path = default_ledger_path() if ledger_path is None else Path(ledger_path)
    with _ledger_lock(ledger_path):
        data = _load_ledger(ledger_path)
        rows = data.get("rows", [])
        remaining = [r for r in rows if r.get("claim_id") != claim_id]
        removed = len(remaining) != len(rows)
        if removed:
            data["rows"] = remaining
            _atomic_write_ledger(ledger_path, data)
    return removed


def status(
    *,
    ledger_path: Path | None = None,
    now: float | None = None,
    pid_alive: Callable[[int], bool] | None = None,
    live_usage: tuple[float, float] | None = None,
    totals: tuple[float, float] | None = None,
) -> dict:
    """Return a status snapshot: live usage, totals, live claims, reaped count.

    Reaps under the flock (so ``status`` doubles as an on-demand reap), then
    reports the surviving claims + the live psutil sample. Injection hooks
    mirror :func:`claim` for tests.
    """
    ledger_path = default_ledger_path() if ledger_path is None else Path(ledger_path)
    now = now if now is not None else datetime.now(UTC).timestamp()
    alive = pid_alive or _pid_alive
    with _ledger_lock(ledger_path):
        data = _load_ledger(ledger_path)
        kept, reaped = reap_rows(data.get("rows", []), now=now, pid_alive=alive)
        if reaped:
            data["rows"] = kept
            _atomic_write_ledger(ledger_path, data)
    live_cores_used, live_ram_used = live_usage if live_usage is not None else _live_usage()
    total_cores, total_ram_gb = totals if totals is not None else _system_totals()
    claimed_cores, claimed_ram = _sum_claims(kept)
    return {
        "total_cores": total_cores,
        "total_ram_gib": total_ram_gb,
        "live_cores_used": live_cores_used,
        "live_ram_used_gib": live_ram_used,
        "claimed_cores": claimed_cores,
        "claimed_ram_gib": claimed_ram,
        "n_claims": len(kept),
        "n_reaped": len(reaped),
        "claims": kept,
    }


def _cmd_claim(args: argparse.Namespace) -> int:
    ok, reason, row = claim(
        issue=args.issue,
        cores=args.cores,
        ram_gb=args.ram_gb,
        phase=args.phase,
        ttl_s=args.ttl_s,
        force=args.force,
    )
    if ok:
        print(f"claimed {row['claim_id']} (issue={args.issue}, phase={args.phase}): {reason}")
        return 0
    # Over the band: advise routing to a CPU pod. Exit 3 so a caller can branch.
    print(
        f"ROUTE-TO-POD: claim for issue={args.issue} phase={args.phase} would breach the "
        f"{DEFAULT_THRESHOLD:.0%} VM headroom band ({reason}). Route this phase to its own "
        f"CPU pod (cpu-small / cpu-mid / cpu-bigmem) instead of the shared VM."
    )
    if args.force and row is not None:
        print(f"(--force: recorded {row['claim_id']} anyway)")
    return 3


def _cmd_release(args: argparse.Namespace) -> int:
    removed = release(args.claim_id)
    print(f"released {args.claim_id}" if removed else f"{args.claim_id} not found (no-op)")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    snap = status()
    if args.json:
        print(json.dumps(snap, indent=2, sort_keys=True))
        return 0
    print(
        f"VM: {snap['live_cores_used']:.1f}/{snap['total_cores']:.0f} cores used, "
        f"{snap['live_ram_used_gib']:.1f}/{snap['total_ram_gib']:.1f} GiB RAM used"
    )
    print(
        f"claims: {snap['n_claims']} live (+{snap['claimed_cores']:.1f} cores, "
        f"+{snap['claimed_ram_gib']:.1f} GiB reserved); {snap['n_reaped']} reaped this call"
    )
    for row in snap["claims"]:
        print(
            f"  {row.get('claim_id')}\tissue={row.get('issue')}\tpid={row.get('pid')}\t"
            f"{row.get('cores')}c/{row.get('ram_gb')}GiB\tphase={row.get('phase')}\t"
            f"{row.get('created_iso')}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EPS advisory VM CPU/RAM resource ledger")
    sub = parser.add_subparsers(dest="command", required=True)

    p_claim = sub.add_parser("claim", help="claim VM cores/RAM for a phase (or route to a pod)")
    p_claim.add_argument("--issue", required=True, help="owning issue number")
    p_claim.add_argument("--cores", type=float, required=True, help="cores this phase will use")
    p_claim.add_argument("--ram-gb", type=float, required=True, help="RAM (GiB) this phase needs")
    p_claim.add_argument("--phase", required=True, help="short phase name")
    p_claim.add_argument(
        "--ttl-s",
        type=float,
        default=DEFAULT_TTL_S,
        help=f"claim TTL seconds (default {DEFAULT_TTL_S})",
    )
    p_claim.add_argument(
        "--force",
        action="store_true",
        help="record the claim even when over the headroom band (advisory override)",
    )
    p_claim.set_defaults(func=_cmd_claim)

    p_release = sub.add_parser("release", help="release a claim by id")
    p_release.add_argument("claim_id", help="the claim_id printed by `claim`")
    p_release.set_defaults(func=_cmd_release)

    p_status = sub.add_parser("status", help="show live usage + live claims + reaped count")
    p_status.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p_status.set_defaults(func=_cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
