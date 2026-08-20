---
name: frozen-decision-adopt-bypasses-arming-guard
description: A first-writer-wins frozen-decision record adopted regime-blind by later resolvers can bypass the very validity check a fix added at decision time (#2389 R2 g1)
metadata:
  type: feedback
---

When a fix adds a validity check at DECISION time (e.g. B6's `mode ==
"production"` requirement before arming share_prefill on a non-tiny run),
check every path that ADOPTS a persisted decision without re-deciding: a
first-writer-wins freeze file (`share_prefill_frozen_<family>.json`,
`bool(rec["armed"])` on adopt) snapshots the FREEZER's regime, and a later
participant in a different regime adopts it with the guard never re-run.
In #2389 R2 the dangerous chain was: a `--tiny` run in a production
out_root legitimately arms on a tiny PASS (tiny runs skip the mode check),
writes the freeze armed=true, and a later PRODUCTION run adopts it —
arming 27B generation on CPU-tiny evidence, exactly what the fix forbade
at the direct-read path.

**Why:** the guard lives in the compute-and-freeze branch only; the adopt
branch trusts `rec["armed"]`. Severity keys on reachability: trace which
dispatch surfaces can put a cross-regime writer into the same out_root
(in #2389 dispatch.sh never passes `--tiny` to run.py, so the chain was
manual-error-only → Minor hardening note, not a blocker).

**How to apply:** for any arming/gating fix, grep for the decision's
PERSISTENCE (freeze/cache/frozen/adopt) and verify the adopt path either
re-runs the guard or cross-checks the freezer's regime (the record often
already carries `repro`/mode fields — a one-line check). Related:
[[gate-artifact-mode-blind-arming]] (the direct-read arming half),
[[smoke-fixture-authored-with-consumer-keys]] (verify the consumer's
required mode value against the producer's actual emission — here
`"tiny" if args.tiny else "production"` matched).

Second lesson from the same round: a fix that reformats a single-line call
to multi-line can push the waivered token line (`file_exists`) below its
line-position-keyed lint waiver (`_hub_verify_waiver_present`: call line or
immediately-preceding non-blank line), silently making the waiver inert —
the reformat variant of [[stacked-lint-waivers-read-window]]. Check waiver
placement whenever a fix wraps/reflows a waivered call.
