---
name: refreeze-moves-pilot-membership-downstream-frozen-artifacts
description: A superfamily-edge change re-freeze moves pilot SELECTION membership; downstream frozen artifacts (prompt_pins, evidence store) go membership-stale while every test stays green because they pin artifact bytes, not manifest<->artifact consistency — enumerate and report the stale set explicitly (#2658 D)
metadata:
  type: feedback
---

When a re-freeze changes the superfamily graph (or any per-item sha), the
manifest's pilot SELECTION can move by a few ids while artifacts frozen FROM
the old selection (prompt_pins.json, evidence_packets.json) stay committed
and every suite stays green — the tests pin the artifact bytes, and no test
cross-checks manifest-vs-pins membership. The stale set is invisible unless
you diff old-vs-new selection ids yourself.

**Why:** #2658 group-D re-freeze (2026-09-02): merging duplicate-stem keyed
superfamilies moved 2 of 629 pilot ids; pins/evidence were outside the round's
file set and their own freeze helpers REFUSE drift by design (deliberate
re-freeze is a separate, sequenced act). Reporting "2 ids membership-stale,
0 sha-stale, artifacts X/Y need their own re-freeze, Z does not" is what lets
the orchestrator sequence it against live sibling rounds that consume them.

**How to apply:** any round that re-freezes a manifest another frozen artifact
was derived from: (1) diff the derived-selection ids old vs new; (2) split
staleness into MEMBERSHIP (ids moved) vs SHA (content of retained ids moved) —
measure both, e.g. intersect changed-content records with the pinned id set;
(3) name each downstream artifact as needs-refreeze / does-not, with the
evidence, instead of regenerating outside your file set. Related:
[[claimed-test-file-sibling-fixture-coupling]].
