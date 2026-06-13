---
name: Slot-kind composition check for cross-arm four-float contrasts
description: When two arms differ massively in marker-emission rate, their slot reads mix end_of_response vs pre_marker slots in different proportions — check the contrast restricted to matched slot kinds before narrating it
type: feedback
---

When a four-float DV (Δz_EOS, Δz_marker, margin) is contrasted BETWEEN
conditions whose marker-emission rates differ a lot, the slot the read
lands on differs systematically too: high-emission arms are read mostly
at `pre_marker` slots (mid-response, just before the first emitted
marker), low-emission arms mostly at `end_of_response` slots. The rig's
slot-parity assert only covers trained-vs-base on the SAME text — it
does NOT cover cross-arm comparability.

**Why:** In #571 (2026-06-11) the narrow arm's slots were ~92%
pre_marker vs ~15% for broad, and within-arm Δz_EOS differed by slot
kind (broad: +16.0 at end_of_response vs +9.3 at pre_marker). The
headline survived — restricting to end_of_response-only slots made the
contrast LARGER (+12.7 vs +9.4) — but without the check the contrast
would have been open to a slot-composition-artifact objection.

**How to apply:** For any cross-condition four-float contrast, pull
`slot_kind` counts per condition from the per_q rows first. If the
pre_marker fraction differs by more than ~2x between conditions, compute
the contrast restricted to matched slot kinds (typically
end_of_response-only) as a robustness read and report it next to the
headline. Also report per-arm text-level emission rates — a large
emission asymmetry is usually itself a finding (the #571 clamp-vs-hijack
trade-off), not just a nuisance.
