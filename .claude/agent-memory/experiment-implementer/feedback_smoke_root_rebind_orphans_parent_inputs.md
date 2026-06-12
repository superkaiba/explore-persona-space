---
name: smoke-root rebinding orphans parent inputs
description: GEN/OUT/EVAL *_smoke rebinding silently orphans read-only parent inputs when the fetch step staging them is skipped under smoke but a downstream step consumes them
type: feedback
---

Smoke-root rebinding (GEN/OUT/EVAL -> `*_smoke`) silently orphans READ-ONLY
PARENT INPUTS whenever the fetch step that stages them is skipped under smoke
(early-return) but a downstream step still consumes them — the smoke crashes
on a path only the real run staged (fresh-pod signature: AssertionError on
the first parent cid at the consuming step, AFTER all earlier smoke steps
passed).

**Why:** smoke/real isolation is for GENERATED artifacts; parent inputs are
inputs. Skipping their staging under smoke breaks smoke=sweep parity exactly
where a fresh pod needs it. Burned at #542 (2026-06-12): smoke p0prime's
closeness step asserted on `clouds_parent/sp_swe__last_prompt.npz` that only
the real fetch had staged.

**How to apply:** give parent inputs a canonical NON-rebinding location (the
`DATA537` pattern — e.g. `CLOUDS_PARENT = REPO/eval_results/issue_<N>/clouds_parent`)
and stage them BEFORE the smoke early-return, so identical fetch+read lines
execute in both modes. Never fix with a smoke-mode skip (removes coverage) or
a read-side-only fallback (a fresh pod running smoke-first still has nothing
staged). When writing any `--smoke` rebinding dispatcher, audit every
downstream read for parent-input paths and pin them to non-rebinding
constants.
