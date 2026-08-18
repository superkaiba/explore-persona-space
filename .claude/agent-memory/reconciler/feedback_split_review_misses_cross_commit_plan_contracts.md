---
name: split-review-misses-cross-commit-plan-contracts
description: Claude per-commit split-review composites PASS while missing round-level plan-contract gaps (declared phase_outputs sentinel with no writer; a registered control spanning P2 telemetry -> P3 control -> P4 figure); Codex whole-round review caught both (#2330 r1)
metadata:
  type: feedback
---

When the Claude side is a SPLIT-REVIEW composite (per-commit `code-reviewer-lean`
sub-reviews + one contract-bearing group for marker gates), its PASS is
trustworthy on COMMIT-SCOPED code correctness but structurally blind to
round-level PLAN-CONTRACT properties that no per-commit brief owns. Two classes
verified in #2330 r1 (both Codex-only catches, both upheld as blockers):

1. **Declared phase output with no writer** — plan §9 `phase_outputs` declared
   `P1: {sentinel: /workspace/logs/issue-2330-p1-smoke.json}`; grep over all six
   round files = 0 hits for `sentinel|p1-smoke` (the driver was standalone, so
   the parent's `C.write_sentinel` could not even be imported), and the impl
   marker claimed "the #1491 sentinel/[phase=done] contract ported verbatim".
   The #909 declare→satisfy class, plus a false report claim. g1 reviewed the
   P1 gates line-by-line and never checked §9's declared output.
2. **Registered control spanning multiple phases** — plan §8/§11 registered the
   #1491 truncation-restriction control (test-restricted read + untruncated
   refit) + a per-(model,split) cap-hit report INSTEAD of the 2%-regen default;
   the P2 driver's own comments deferred it to P3, P3 had no implementation, no
   aggregator produced the figure's input, and the figure titled the 2% line
   "re-gen trigger" (the plan-rejected disposition). g7 saw the assembler gap
   (persisted a CONCERN) + the label (Minor) but not the missing CONTROL — each
   fragment looked minor inside its own group.

**Why:** split-review briefs scope by commit; plan-contract items (phase_outputs
writers, §8/§11 registered controls, §10 call-shape binds) span commits, so
each group sees only a non-blocking fragment.

**How to apply:** when adjudicating a split-review PASS vs a whole-round Codex
FAIL, run the cross-commit checks yourself before crediting the PASS: (a) every
plan §9 `phase_outputs`/sentinel entry has a WRITER in the round diff (grep the
declared filename); (b) every §8/§11 registered control/report names a producing
script in the round; (c) impl-marker "ported/asserted/atomic" claims are grepped
against the code (this round: sentinel-ported, n>d-asserted, tmp+os.replace npz
— all three false). Calibration counterpoint: Codex's #2330 file:line citations
ALL verified (no fabrication — cf. [[codex-fabricated-code-citation-silent-shrink]]),
but 5/8 of its BLOCKERs were over-classed on reachability / plan-sanctioned
design / recoverability grounds (resume think-gate bypass recoverable from
persisted per-chunk telemetry; revision pin's anchor+counts bind was the plan's
own stated runtime bind; n<d unreachable under fail-loud count pins; post-init
validation pure efficiency; exception-teardown fail-path-only) — verify facts,
then re-derive severity from the operational path yourself.

**Second datapoint (#2333 r2, revision round — reconciled PASS + 6 CONCERNs, 0 blockers):**
on a BLOCKER-FIX round, Codex re-filed closed r1 items as `NOT-ADDRESSED` by
silently RESTATING each concern stronger than its r1 wording — adjudicate
against the r1 concern LINE pulled verbatim from the r1 verdict (the #952-r2
grep-the-quoted-contract check, now a per-item procedure): r1
"bank/donor phases IGNORE 8-worker sharding → races/duplication" was closed by
a loud single-worker guard + recorded named deviation (r2 restated it as
"must implement 8-way"); r1 "S1 not gated per cell" was closed at the
registered anchor-separation-survivor grain (r2 restated the floor as
per-(arm,cell) USABLE pairs — never registered); r1 "fitness probe has no
implementation" was closed by a live-run conservative probe (r2 demanded reuse
loaders the plan's "ANY failed check ⇒ selfgen" clause makes optional). Also a
§9 arithmetic tell: Codex read pod-billed gpu_h columns (wall × 8) as per-GPU
serial work to inflate a minutes-scale deviation to "0.8–1.6 GPU-hours".
Claude-side miss confirming [[claude-misses-invariant-comment-smell]]: the
split g1 credited a docstring's "update shard + va store + done record
atomically" while the code commits the JSONL (carrying the `regenerated_at`
resume key) BEFORE the va-store replace — a real crash-window
new-text/stale-V_a permanence Codex alone caught (upheld as CONCERN:
secondary-DV, seconds-wide window, rare branch). Trust the split PASS only
after re-reading multi-file commit sequences the docstrings call atomic.
