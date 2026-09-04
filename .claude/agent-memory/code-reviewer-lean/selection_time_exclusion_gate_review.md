---
name: selection-time-exclusion-gate-review
description: "Review recipe for a deterministic selection-time exclusion gate (e.g. prompt-length gate in a frozen selection): 4 probes incl. the realized-bias jq on shortfall cells and the single-source-comment grep"
metadata:
  type: feedback
---

When a round adds a deterministic exclusion gate to a frozen selection (items
skipped at freeze time because they cannot run under a registered budget),
run these probes (#2658 r15 g2, length gate under the amended 4096 cap):

1. **Shared-quantity check:** the gate's measured quantity must be the SAME
   extracted helper the downstream fail-loud assert uses (here
   `rendered_token_count` shared by selection and generator). Two independent
   counters can drift and re-crash production on a frozen item.
2. **Realized-bias jq probe:** `[.splits[][][] | select(.shortfall != null and
   .n_overlong_excluded > 0)]` on the frozen artifact. Exclusions landing only
   in cells that still hit n_common leave totals/statuses unaffected (latent
   wrinkle only). Exclusions inside shortfall cells make cause tags and
   n_eligible fields keyed on different quantities a REAL mislabel, and the
   downstream disclosure duty escalates.
3. **Status-vs-field grain:** diff which quantity the status/cause gate keys
   on (post-gate fit count) vs what sibling record fields and print lines
   report (pre-gate eligibility). Related: [[registered_gate_quantity_substituted]].
4. **Single-source comment grep:** a NEW comment claiming "writer imports the
   schema/path constants from the reader" is verified by grepping the writer
   for literal duplicates. Duplicates are at best test-pinned via a
   freeze-through-A-read-through-B interop test, and the false comment invites
   a one-sided schema bump. Related: [[paired_script_default_path_contract]].

Also verify the paired stale guard's SCOPE before accepting a "strict
superset" claim: a pilot-valued formatted-length cap is safe only if it bounds
the prompt segment alone and the total-sequence bound has headroom for
budget + new-token cap + boundary (read the shared helper, not the docstring).

**Why:** all four probes produced findings or certified claims in one round.
**How to apply:** any diff freezing a selection/manifest with skip-and-count
exclusion semantics, or claiming writer/reader constant unification.
