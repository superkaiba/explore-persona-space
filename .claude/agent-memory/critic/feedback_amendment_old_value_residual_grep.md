---
name: amendment-old-value-residual-grep
description: Targeted-amendment reviews — "all N anchors applied byte-exact" verifies the edits, NOT site-set completeness; grep the amended value's OLD form plan-wide (#823 v11 missed 28th site in the registered provenance sentence)
metadata:
  type: feedback
---

On a targeted plan amendment (N byte-exact anchor edits changing a registered
value), the application check "all N anchors verified present-once / old absent"
proves the EDITS landed — it cannot prove no UN-EDITED site still asserts the
old value. Grep the old value plan-wide and classify every hit: live-spec
(REVISE — the missed site), parent-rig descriptive (fine), rejected-alternative
row (fine).

**Why:** #823 plan v11 (27 edits, cap 1024→4096, verify_plan PASS) left ONE
live-spec `max_tokens 1024` in the §4 data-realism completion-provenance
sentence — the registered per-arm provenance statement CLAUDE.md propagates
into every results summary; as written it either mis-records all 5 arms'
recipe or, if bound by the implementer, reinstates the k-confounded truncation
the amendment existed to remove. The recurring missed-site class: registered
provenance / N-A / data-realism blocks that RESTATE recipe values far from the
sections the amendment targeted.

**How to apply:** any C5-style "did the amendment break internal consistency"
check — grep each amended value's old form (`max_tokens 1024`, old n's, old
thresholds) over the WHOLE new version, not just the diff. Adjudications that
held in the same round (APPROVE-side): (1) cap raise grounded ONLY by the ≥2×
doubling convention is acceptable when pilot length data are right-censored at
the old regen cap — no base above the censor point can be measurement-grounded,
so convention + one regen round + per-cell re-measure + labeled residual is the
correct shape (8192-base would be equally unmeasured); #2221 headroom class
does not fire on API-side generation (no engine max_model_len pin). (2)
Intersected-across-arms mask ⇒ refusal attrition is population-validity-only
for the POOLED arm contrast (identical surviving set by construction); the
residual channel is per-persona CELL composition (selection intensity
persona-correlated) — analyzer Concern, with paired same-row comparators
selection-matched. (3) Roster-KEEP over exclusion when exclusion changes the
manipulated variable (nested assignment: arm k = personas {0..k−1}) and
post-pilot replacement would outcome-select the roster; per-persona usability
goes to the registered manipulation predicate, not assumption. (4) Preserve a
registered split + label the boundary rung UNREALIZABLE over rebalancing that
breaks row-identity with banked anchors. Related:
[[unsatisfiable-gate-respec-review]] (the gate-split marks), [[regen-trigger-headroom-at-production-cap]].
