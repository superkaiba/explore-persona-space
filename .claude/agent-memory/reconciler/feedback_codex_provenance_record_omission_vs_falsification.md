---
name: codex-provenance-record-omission-vs-falsification
description: "Codex BLOCKERs a payload whose sole estimator/provenance record does not cover a sibling fit as 'falsification' — read the record's SELF-SCOPING fields (machinery/derived_from naming the call it describes) and per-field truth vs the uncovered fit: a self-scoped truthful record + an uncovered secondary read is an OMISSION (CONCERN), not a false record; pre-existing + strictly-narrowed-by-the-diff is a second demotion lever"
metadata:
  type: feedback
---

When a round adds/fixes REALIZED provenance records and the Codex twin
BLOCKERs a site where a payload persists a value fitted under a DIFFERENT
regime than the payload's only estimator record describes ("persists X under
a record whose sole description asserts Y"), decide falsification-vs-omission
from the record's own fields, not from the payload topology:

1. **Self-scoping check.** Read the record's `machinery` / `derived_from`
   (or equivalent) fields. A record that NAMES the specific call it
   describes ("<module>.<fn>(lambdas=...)" + "read at the sweep call") makes
   no claim about sibling fits — the sibling's provenance is ABSENT, which
   is a doc-grade CONCERN, not a false record. Falsification requires an
   UNSCOPED assertion that is untrue of the numbers it purports to cover
   (the pre-fix re-stamped-driver-constants shape).
2. **Per-field truth audit vs the uncovered fit.** Check each substantive
   record field against the uncovered fit's realized config (module-global
   patches in scope, callee kwdefaults). In #2546 r15 two of three fields
   (inner folds under the scoped patch, selector kwdefault) were TRUE of
   the rp control too — only the λ-grid diverged. The narrower the real
   divergence, the stronger the omission reading.
3. **Pre-existing + strictly narrowed.** Diff the site against the base
   blob: if the pre-fix payload carried a WORSE (unscoped) stamp beside the
   same uncovered fit, the round strictly shrank the mis-attribution
   surface — blocking it inverts the review bar (compose with
   [[residual-gap-inherits-parent-severity-bar]]: a weaker-in-kind residual
   never inherits the halting parent's BLOCKER bar).
4. **Ledger mechanics on the downgrade:** when BOTH arms persisted the same
   finding (Claude CONCERN row + Codex BLOCKER row), do NOT re-raise a
   third row — `defer-concern --by reconciler` on the BLOCKER row, leave
   the CONCERN row as the canonical open concern, and state the
   N-upheld/N-persisted accounting via the EXISTING rows.

**Why:** #2546 code-review r15 — Codex FAILed (BLOCKER
`sweep-random-projection-estimator-unrecorded`) a record-only round because
`random_projection_control` fits on the 13-pt `fc.LAMBDAS` default while the
sweep payload's `estimator_realized` describes the 23-pt N1M sweep. The
record self-scoped (`machinery`/`derived_from` named the sweep call), 2/3
fields were true of the control, the mismatch was pre-existing at the base
blob in a strictly worse unscoped form, and the task's earlier halt was for
the DIFFERENT falsification shape (primary fits misdescribed). PASS +
CONCERN kept open with a record-only fix prescription.

**How to apply:** provenance/estimator-record disputes at reconcile — read
the record composer function + the uncovered fit's call chain (patches,
kwdefaults, shared-core defaults) + the base blob before accepting either
arm's framing.
