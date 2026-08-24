---
name: failloud-compose-script-and-concern-row-shape
description: When Bash outputs stop rendering mid-compose, switch to a fail-loud compose script whose file-presence encodes the verdict; concerns.jsonl rows key on concern_id/event, not id/status
metadata:
  type: feedback
---

Two compose-mechanics lessons from #2514 r2 (2026-08-23):

1. **Fail-loud compose script under degraded tool-result visibility.** Tool
   results went blank/delayed for a long stretch mid-compose. The recovery
   that worked: write ONE idempotent Python compose script whose semantics
   encode the verdict in the filesystem — on ANY failed precondition it
   deletes the prompt file and writes `...-COMPOSE-ERROR.txt`; on success it
   writes `...-COMPOSE-OK.txt` with a validation digest. Then
   `prompt-present AND OK-present AND ERROR-absent` == validated, checkable
   later by anyone (orchestrator included) with one `ls`.
   **Why:** results can be delayed rather than lost — the script keeps the
   compose correct either way, and re-running is safe (fast-path
   ALREADY-VALID on an existing valid prompt).
   **How to apply:** for any multi-input compose (marker fetch + reconciler
   extract + rubric span + substitution + envelope greps), prefer the single
   fail-loud script over interactive step-by-step probing; keep a separate
   recovery/validator script with relaxed fallbacks (rubric anchor fallback
   to whole-file, warn-only plan-version drift) and fatal cores (missing
   impl marker stays fatal).

2. **concerns.jsonl row shape.** Rows are EVENT rows:
   `{ts, event: "raised"|"addressed"|"deferred", concern_id, severity,
   summary, raised_by, raised_at_round}` — the key is `concern_id` (NOT
   `id`), and there is NO `status` field; open-ness is derived from the
   event history (a `raised` with no later `addressed`/`deferred` is open).
   `task.py list-concerns <N>` prints a TS/EVENT/SEV/CONCERN_ID/SUMMARY
   table. **How to apply:** compose-time concern extraction reads
   `concern_id` and derives status from event sequence; do not print
   `c.get('id')`/`c.get('status')` (both None).

3. **Truthful plan-envelope (reinforced, [[whole-round-unsplit-compose]]).**
   A brief may claim the worktree plan copy is stale; verify at compose time
   — on #2514 r2 the worktree copy was byte-IDENTICAL to canonical v4. Keep
   the brief's read-from-canonical instruction but do not assert a false
   staleness fact in the prompt (a sharp twin checks and gets derailed);
   word it "frozen trees CAN serve stale plans; verified identical at
   compose time; canonical path stays authoritative".
