---
name: Claude PASSes a pivot fix that covers only one of two symmetric post-terminate failure paths
description: When a strategy-pivot durability fix covers the terminal-JSON return path but leaves the raise path stranding-prone, Claude verifies the fix + all prior concerns and PASSes; the symmetric uncovered path is the FAIL. Enumerate BOTH outcome shapes (return vs raise) of the same recovery call.
type: feedback
---

When a code-review strategy pivot (round counter reset) installs a durability
fix for a recovery that TERMINATES a resource before relaunch, Claude's
reviewer family reliably verifies (a) the new fix mechanism, (b) the exhaustion
branch, (c) the new tests, (d) that all prior-round concerns survive — and
PASSes. The miss is the SYMMETRIC uncovered failure path of the SAME recovery
call: the fix covered the TERMINAL-JSON RETURN outcomes but left the RAISE
outcome (an unexpected exception escaping the recovery function's own
try/excepts) routed to a degrade branch that posts NO durable terminal record.

**Why:** A post-terminate recovery call has TWO outcome shapes — it can RETURN a
typed terminal JSON (the expected/known failure classes, each converted
internally) OR RAISE (an unexpected exception class the internal try/excepts
don't catch). A durability fix that gates on the return-value branch silently
omits the raise branch. After the pod is terminated, the raise branch's degrade
("alert / retry next tick") is UNREACHABLE — the terminated resource is excluded
from the next tick's RUNNING-only snapshot, so the retry never re-enters. Net:
billing stopped, no `epm:failure`, no `status:blocked`, capacity-retry can't find
it → stranded run. This is the SAME bug class the pivot was meant to close, for a
different code path.

**The tell that it's real, not phantom:** the SIBLING/caller code path already
mirrors the defensive contract Codex demands — e.g. the poller wraps the SAME
recovery call in `try/except Exception` and converts ANY raise to a typed
terminal JSON ("the pod may already be terminated, so a bare traceback would
strand the run"). When one caller of a post-terminate recovery has the
belt-and-suspenders catch and the disputed caller does not, the asymmetry is the
finding. Grep for the recovery function's name across callers and compare their
exception handling.

**The other tell:** the implementer's OWN comment often proves they understood
the unreachability mechanism for the path they DID fix ("the next tick CANNOT
retry them ... the r3 'retry next tick' was unreachable") — and built the durable
layer for it — but applied it to only one outcome shape. A comment acknowledging
unreachability for outcome A is evidence the SAME unreachability bites uncovered
outcome B.

**How to apply:** When reconciling a Codex FAIL against a Claude PASS on a
strategy-pivot durability fix for a terminate-then-relaunch recovery:
1. Read the recovery function (the poller/owner side) and enumerate EVERY
   post-terminate exit: each typed terminal-JSON return AND every uncaught raise
   path (a best-effort write catching only `OSError`; a router/backend call that
   can raise classes outside the caught set). Confirm uncaught raise paths exist
   TODAY (not just "a future edit") to make it Real-blocking, not hypothetical.
2. Trace the disputed caller's mapping of a RAISE: if it degrades to an
   alert/no-op branch that posts no `epm:failure` + no `status:blocked`, AND the
   terminated resource is excluded from the next iteration's discovery filter
   (RUNNING-only snapshot), the run is STRANDED.
3. Compare to the sibling caller — if the sibling converts any raise to a typed
   terminal record and the disputed caller doesn't, uphold the FAIL.
Cost-of-being-wrong is high (silent stranded run, no operator signal, no auto
re-drive) and the fix is small (mirror the sibling's catch → route through the
existing durable-record path), so Rule 8 → Real-blocking → FAIL.

Incident: #770 v2 r1 (2026-06-30). The r3/v2-r1 fix gated the wedge-clock clear
and added `_retry_durable_write` for the no-capacity/blocked TERMINAL-JSON
outcomes of `backend_poll._failover_wedged_runpod`, but the watcher's
`_wedge_failover` mapped a POST-terminate RAISE to `("alert", None)` → no durable
record → stranded. The poller's own caller (`backend_poll.py:1864-1880`) already
converted any raise to `runpod_wedge_failover_error` terminal JSON; the watcher
omitted the mirror. FAIL.
