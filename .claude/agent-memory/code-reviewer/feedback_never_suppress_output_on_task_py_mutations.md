---
name: never-suppress-output-on-task-py-mutations
description: Never run task.py address-concern/raise-concern/post-marker with >/dev/null 2>&1 && echo — the 200-char --summary cap is a hard error and the && guard turns the failure into silence
metadata:
  type: feedback
---

Never wrap a `scripts/task.py` MUTATION in `>/dev/null 2>&1 && echo "ok"`.
Run it bare (or capture and print `rc`), and verify the ledger afterwards.

**Why:** `address-concern --summary` / `raise-concern --summary` enforce a
**200-char cap as a HARD ERROR** (over-cap raises; the help text says pass
long text via `--summary-file`). In #2386 r2 I batched four
`address-concern` calls as `... >/dev/null 2>&1 && echo "addressed: $c"`.
All four exceeded the cap, all four failed, and because the `&&` never
fired the loop printed **nothing at all** — no error, no success line. The
only reason I caught it was the follow-up `list-concerns` still showing the
concerns as `raised`. Had I trusted the loop, the verdict would have
claimed four concerns discharged while the ledger said otherwise.

This is the same fail-open direction as
[[keep-probe-read-error-fail-open]]: a suppressed non-zero rc that reads as
"nothing to report" rather than "the command failed".

**How to apply:**
- Long summaries go in `--summary-file <path>` (any length; over-cap text is
  preserved in the event's `evidence` field). `--rationale` / `--note` are
  accepted aliases for `--summary`; `--summary` is canonical.
- `--by` and `--round` are REQUIRED on `address-concern`.
- After any batch of concern mutations, re-read
  `task.py list-concerns <N> --open-only --json` and confirm the expected
  ids actually moved to `addressed`. Success is what the ledger says, not
  what the loop echoed.
- Same rule for `post-marker`: exit 0 with a stderr commit-deferred ERROR is
  SUCCESS (the row IS appended, never re-post) — but you can only tell that
  apart from a real failure if you did not discard stderr.
