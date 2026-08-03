---
name: Pinned parent artifact PAIRS can be mutually inconsistent — coverage-check the READ-side artifact
description: "#601: pinned issue472 persona_bank.json (60) vs R_eval.json (61) shared only 45 personas; panel ⊆ bank did NOT imply panel ⊆ R_eval — teacher-forced reads KeyError'd at pod time (#504 class). Always assert coverage against the artifact you READ from, per (persona, question), before registering any panel."
type: feedback
---

When a rig reuses a parent's MULTI-FILE artifact bundle (persona bank +
centroids + frozen R + adapters), the files are NOT guaranteed mutually
consistent — the parent may have regenerated one after the other. On #601
(2026-06-11), the pinned `issue472_neg_geometry` bundle had persona_bank=60
personas, R_eval/R_train=61, overlap only 45; selectors driven off the
bank/centroids picked panel members (`bartender`, `electrician`) absent from
R_eval, and the production teacher-forced read would have crashed every GPU
shard at pod time (round-1 code-review BLOCKER `phase0-r-eval-coverage-gap`).

**Why:** "Hub-verified" in a plan means the FILES resolve, not that their
persona/key sets agree. Membership in the selector's source artifact (bank,
centroids) says nothing about membership in the artifact the reader actually
indexes (frozen R).

**How to apply:**
- For every read path `artifact[p][q]`, assert FULL per-(persona, question)
  coverage against THAT artifact before registering/pre-registering any
  panel or launching workers (e.g. #601's `phase0_lib.full_r_coverage` /
  `assert_r_eval_coverage` / `build_r_map`). Check non-empty payload fields,
  not just key presence.
- Constrain selection pools to the covered subset BEFORE deterministic
  selection (decile/quantile pickers), and record exclusions BY NAME in the
  registered artifact so the pre-registration is honest.
- Reads of legitimately-uncovered items (e.g. a parent cell's trained
  negative missing from frozen R) are DESCOPED per-item with an explicit
  `coverage: absent-from-frozen-R` field in the output JSON — never
  regenerate the frozen artifact (parity is load-bearing) and never let the
  denominator shrink silently.
- Make the CPU smoke do the REAL lookups (every persona x question the
  production worker will read), not synthesized payloads — synthesized
  fixture fields are exactly how this class escapes review.

Post-hoc REGENERATION ordering (an input regenerated AFTER its dependent
capture) is the sibling failure class — now checklist item (j) in
`.claude/rules/artifact-reuse.md` (#922).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Pinned artifact pairs can disagree](feedback_pinned_artifact_pair_mutual_inconsistency.md) — assert per-(persona,q) coverage against the READ-side artifact; descope with a coverage field, never regenerate frozen R. #601.
