---
name: strength-targeted-rereview-compose
description: "#2569 rr3: when the brief names a class the Codex twin demonstrably beat Claude on (3-for-3 disagreements, all wrong-denominator species), lead the prompt with an explicit assignment section + ordered attack targets + required per-target answer lines; also fence in-range bookkeeping paths and recount SHA asserts after writing"
metadata:
  type: feedback
---

On #2569 re-review round 3 (2026-08-26) the orchestrator brief ordered a
REVIEWER-STRENGTH-TARGETED compose: across rounds 1-2 there were three direct
Claude-vs-Codex disagreements, Codex right in all three, every one the same
species (a verdict over an insufficient/wrong denominator — empty roster
certifying symmetry, majority over evaluable-not-all pairs, population verdict
from one surviving point). Compose deltas that worked:

1. **Lead with an assignment section** ("Your assignment this round:
   denominators, vacuity, gate-precondition binding") stating the 3-for-3
   record and the species, then a 4-question checklist (over what set / who
   guarantees non-trivial / what does near-empty read as / is the precondition
   bound at the production call site). Explicitly deprioritize the sibling's
   strength ("exactness re-verification is lower-yield for you").
2. **Ordered attack targets** (6, by expected yield), each carrying: the fix's
   current literals, the calibration provenance to adjudicate (e.g. a floor
   calibrated on 2 of 18 arms), BOTH directions (under-binding vs
   over-binding/deadlock; thin-evidence FAIL inverted into thin-evidence
   PASS), and a preferred execution check. Require a one-line answer PER
   target in the verdict body even when it holds.
3. **Cross-target class framing**: when one round applies TWO different
   remedies to the same bug class (H2b got an UNDECIDABLE token, H7 got a
   flag only), hand the asymmetry to the twin as a Step 3.7 sweep across ALL
   verdicts — "same class, two remedies — justified or half-fixed?".
4. **Fence in-range bookkeeping paths**: a fix-round range can carry
   REREVIEW_BRIEF.md + .claude/agent-memory/** commits (prior composers'
   memory landed on the issue branch); name them OUT of scope explicitly or
   the twin flags them as scope creep. (Corollary: commit YOUR memory on
   MAIN, never in the issue worktree — it enters the next round's diff.)
5. **Recount SHA asserts after writing, not before**: the base/tip SHA counts
   depend on how many rubric steps quote `git show <base>:` forms; my
   pre-write guesses (7/8) were wrong (true 9/10). Enumerate with `grep -n`
   and verify each occurrence is intended before pinning the count.

**Why:** the ensemble's value is anti-correlated errors; pointing each twin at
its demonstrated comparative advantage (and saying so in the prompt) is how
the orchestrator buys that, and the brief made it an explicit compose duty.

**How to apply:** any re-review brief that names a Claude-vs-Codex
disagreement record or a defect species one side keeps winning. Related:
[[codex-side-sharded-round-compose]], [[brief-pinned-sentinel-and-verdict-enum]],
[[revision-round compose recipe]].
