---
name: Durable idempotency records belong off the .claude/cache failure dir
description: A same-dir sentinel can't bound "exactly once" under EDQUOT; the ~/.eps-routing/ LeaseStore is the durable record (different mountpoint + per-issue flock)
type: feedback
---

When a per-poll idempotency record must survive the canonical project
persistent-disk-failure mode (EDQUOT on the MooseFS per-pod quota,
read-only-fs, out-of-inodes), do NOT put it in `.claude/cache/` next to
the artifact it guards.

**Why:** the sidecar (`.claude/cache/issue-<N>-handle.json`) and any
sibling sentinel (`.claude/cache/issue-<N>-failover-persistence-failed.json`)
share the SAME directory, so a directory-level write failure that fails
the sidecar write fails the sentinel write TOO. The sentinel then degrades
to "absent on every poll while the disk failure persists" → the
short-circuit never fires → one extra paid action PER POLL TICK (EDQUOT
does not clear between ~540s polls). This was the #659 round-1→round-3
arc: round-1 `recovered.backend` guard only blocked emitting "running",
round-2 sentinel shared the failure dir, round-3 fixed it.

**How to apply:** the `LeaseStore` (`backends/router.py`, default
`~/.eps-routing/`) is the durable idempotency substrate — a DIFFERENT
mountpoint than `.claude/cache/`, with a per-issue flock
(`LeaseStore.transaction(issue)`) that also serializes concurrent polls.
Pattern: add a narrow field to the `Lease` dataclass (e.g.
`gcp_failover_of`, with `to_json`/`from_json` round-trip — keep it
backward-compatible via `payload.get(..., None)`), STAMP it immediately
after the irreversible action succeeds and BEFORE the `.claude/cache`
write, and CHECK it before re-doing the action. Key the check to the
specific run's stable identity (pod_name/job_id), not just the issue, if
the semantics are "exactly once PER crash" not "per issue". Keep the
`.claude/cache` sentinel as a fast-path OPTIMIZATION (avoids the
lease-store flock round-trip on the common no-failure case), not the
safety guarantee. Catch only `OSError` from the lease store → "no
record" (worst case one extra action, never silent suppression); let a
no-`$HOME` `RuntimeError` propagate (acceptable loud degradation, better
than a duplicate paid launch).

**Test isolation:** a bare `LeaseStore()` resolves to
`Path.home() / ".eps-routing"` with no injection seam, so tests that
exercise the router's internal lease writes must isolate it — an autouse
fixture pinning `Path.home` to a per-test tmp dir
(`monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp))`)
covers BOTH the router's internal `LeaseStore()` and any poller-side one,
and prevents a real-`~/.eps-routing/` write leak.
