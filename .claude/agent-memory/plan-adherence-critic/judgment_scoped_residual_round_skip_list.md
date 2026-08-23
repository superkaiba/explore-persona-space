---
name: judgment-scoped-residual-round-skip-list
description: Scoped residual-fix rounds — adjudicate the skip list by exhaustive call-site enumeration of the wedged helper x dispatch arms; verify FAIL-at-base structurally via git show BASE:file greps; a domain-keyed carve-out needs its positive-control test
metadata:
  type: feedback
---

For a scoped verification round (user-authorized residual fix after a review-cap
FAIL), three checks settled every adjudication in #2389 r6:

1. **Skip-list adjudication = call-site enumeration × dispatch arms.** The
   implementer skipped width-threading on `capregen-grid` claiming it "never
   calls `_adopt_pilot_gen_batch`". The check: grep ALL call sites of the wedged
   helper in the driver (4 hits → map each to its phase function), then map each
   dispatch arm to its launcher (`run_single_gpu` vs `run_fanout_phase` — the
   fan-out launcher threads `--num-workers` per worker, the single-GPU one does
   not). Only arms that BOTH use the width-blind launcher AND reach the helper
   need the fix. Same-class wedges elsewhere (stage2 also adopts) are cleared by
   the launcher mapping, not by trusting the report.

2. **FAIL-at-base claims verify structurally, no base checkout needed.**
   `git show <BASE>:<file> | grep -c <fixed-literal>` returning 0 for each fixed
   behavior (carve-out log line, threaded flag, new report field) proves the new
   tests fail at base without running them there — minutes cheaper than a
   scratch worktree.

3. **A runtime-domain-keyed carve-out is only "GPU legs byte-identical" if (a)
   the skip is an early-return ABOVE an untouched resolution block, and (b) a
   positive-control test pins the domain probe (GPU-visible claim leg still
   resolves).** The carve-out's premise ("this leg generates nothing") must be
   read from the leg function itself (`leg_claim` = poll + write vllm_cells.json
   only), not from the carve-out's own comment.

**Why:** the round-6 skip list contained four items; all four were genuinely
out of scope, but only the call-site × launcher cross-product could show that —
the implementer's own grep claim was correct yet unauditable from the report
alone. **How to apply:** any round whose brief says "verify these skips are
out of scope, not gaps" — enumerate mechanically before ruling.

Related: [[judgment-rename-rework-strands-secondary-readers]] (same grep-the-fork
instinct), [[judgment-preregistered-gate-relaxation-checklist]] (confinement
grep incl. sibling gates).
