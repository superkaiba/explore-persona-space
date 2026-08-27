---
name: registered-constant-figure-sweep-exposure
description: Verify a "swept the class, rest are data-derived" figure-fix claim by enumerating every axhline/axvline site, classifying each against the plan's registered constants, then exposure-checking each ungated registered constant's PRODUCER for a reduced/smoke regime (#2569 r2 shard 3)
metadata:
  type: feedback
---

When a fix to "registered bands drawn against smoke-regime data" claims a
class sweep found all remaining threshold lines data-derived, do not accept
the classification — re-derive it, then do an EXPOSURE analysis:

1. `grep -n "axhline\|axvline\|axhspan\|axvspan"` the figures module; read
   ~12 lines of context per site; classify each drawn constant as
   (a) registered plan-§ constant, (b) data-derived (read from the artifact:
   shuffle p95, floors, medians, CIs), or (c) neutral reference (0, chance,
   identity y=x, ratio=1). Compare against the plan's success/kill section
   literal-by-literal.
2. For each UNGATED class-(a) constant, check whether the PRODUCING driver
   can ever emit a reduced-regime artifact: `grep args.smoke <producer>`.
   No smoke dial + per-unit measurements from real checkpoints ⇒ the
   ungated line is latent-only (NIT), because no smoke-grade data exists to
   mis-present. The fix's CHARACTERIZATION can be wrong ("data-derived")
   while its CONCLUSION (out of scope) holds — grade the mechanism, note
   the imprecision.
3. Verify suppressed-not-disabled by EXECUTION: render both regimes (real
   smoke artifact + a doctored production copy) and Read both PNGs; also
   drive the fail-loud missing-flag branch.

**Why:** #2569 r2 shard 3 — the fix gated H2b bands + H3 floor/kill on
`regime.smoke`, claimed everything else data-derived; independent sweep
found the §7.5 leg-5 registered constants (r/2=16, 0.6 rank-1) drawn
ungated in `build_dw_effective_rank` — non-exploitable only because
`issue2569_dw_fleet.py` has no smoke regime at all.

**How to apply:** any review of a figures fix whose blocker was
verdict-thresholds-vs-wrong-regime; pairs with
[[styled_open_marker_zero_edge_width]] (same file class) and
[[sizing_pilot_entry_class_vs_pinned_blindspot]].
