---
name: class-fix-regex-corpus-spelling-and-surface-pinning-tests
description: Uphold REVISE when a revision-round class fix is a regex covering one surface spelling (grep the corpus for other spellings — worded "to" ranges) or when a silence-safe check's test battery never pins its declared operating surfaces (setup-line/alt-text/positive-unit fires)
metadata:
  type: feedback
---

Two upheld Codex Must-Fixes from #2367 plan-v2 Methodology reconcile (Claude
APPROVE vs Codex REVISE; adjudicated REVISE, 2026-08-27). Companion to
[[fires-on-incident-headline-needs-real-artifact-acceptance]] (r1 of the same
issue — Codex right both rounds; Claude critic verified the plan's own trace
faithfully but only on the plan's OWN surface forms).

**1. A class defect fixed by a regex covering ONE spelling of the class is
an incomplete fix — grep the corpus for the other spellings yourself.** The
v2 range-endpoint guard matched dash chars only (`[–—−-]`); the corpus
already spells ranges in words (`tasks/completed/207/body.md:68` "N = 50 to
550 pairs"). Re-deriving the arithmetic on `n=26 to 36 pairs per cell`
showed the worded spelling re-admits endpoint 26 → spurious draw product
26×3×5=390 → veto → the exact v1 blocker restored. Key discriminators for
upholding: the plan's own rationale stated class semantics ("a range
endpoint ... never a plotted total") wider than the mechanism; the gap was
NOT among the plan's named recall sacrifices (unnamed gap ≠ accepted
sacrifice); and the failure direction was the one the revision round existed
to fix. Fix cost was trivial (extend the regex + one incident-fidelity test
at the worded spelling).

**2. A silence-safe conjunctive check whose registered test battery never
pins its declared operating surfaces is a dead-tripwire plan gap, not an
implementation nit.** All 16 tests placed the firing claim in the caption
(fixture `_grain_body(caption, ...)` had a caption slot only); no test fired
from setup-line-only, alt-text-only, or a positive `row`-unit claim — yet
the Goal named all three surfaces and half the motivating incident was a
setup-line instance. Decisive: the check's OTHER instruments cannot catch an
inert surface (corpus sim expects 0 fires; silence is the designed safe
mode), so the battery is the only falsifier — a caption-only/draw-only
implementation passes everything. Same family as
[[claude-approves-watcher-pass-without-main-wiring-test]] and
[[claude-underclasses-unverified-branch-test-gap]].

**How to apply:** on a guard/recognizer-regex Must-Fix, verify the cited
corpus line exists, then re-run the regex mentally/mechanically on the
respelled incident phrase and check the product arithmetic — if the plan's
stated semantics are wider than the drafted mechanism and the alternate
spelling is corpus-attested, uphold. On a missing-tests Must-Fix, ask
whether ANY other registered instrument could detect the untested surface
being inert; for silence-safe WARN checks the answer is usually no → uphold.
