---
name: open-interp-ids-at-cr-gate
description: Brief's do-not-re-raise list covering interp-round ledger ids does NOT suppress a real Lens-14 surface FAIL when those ids are still OPEN — compose both the adjudication block AND an explicit in-scope carve-out.
metadata:
  type: feedback
---

When the round-1 clean-result brief carries an adjudicated do-not-re-raise
list including "all interp ledger ids" but the ledger still shows some of
those ids OPEN (raised, never `addressed`/`deferred` — the interp loop
fixed the substance in body prose without running `address-concern`), the
verifier's check-65 / Lens-14 FAIL is REAL at the surface level. Compose
BOTH blocks:

1. The do-not-re-raise block: the ids' SUBSTANCE is settled by the closed
   PASS+PASS interpretation loop — re-arguing content is a FALSE FINDING.
2. An IN-SCOPE carve-out: inherit the verifier Lens-14 FAIL as a
   substantive SURFACE finding (verifier authoritative per the lens text);
   the requested fix is id acknowledgement in the prose that already
   carries the substance (or deferral markers / ledger reconciliation),
   NOT content edits. Invite the per-id "fooled substring match" judgment.

**Why:** #2552 r1 (2026-08-25) — 8 codex-interp-r1-c* ids open (r2/r3 ids
were properly addressed/deferred), body had zero id mentions and zero
deferred markers. Without block 2 the do-not-re-raise list reads as
suppressing the whole Lens-14 audit (false PASS path); without block 1
Codex re-litigates 4 closed rounds. Both failure modes are one-sided —
the pair is required, mirroring [[adjudicated-concern-count-override]].

**How to apply:** any CR-gate round whose brief names interp ledger ids as
adjudicated: diff the brief's id list against the OPEN-CONCERNS envelope
before composing; any adjudicated-but-still-open id gets the two-block
treatment. Deferred ids (latest event `deferred`) are correctly absent
from the envelope — name them as adjudicated-deferred, never as open.
