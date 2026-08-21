---
name: judgment-false-inherited-premise-implement-branch
description: When a plan falsely registers a component as "inherited" and the fix implements it fork-new, the blocker closes without a plan amendment — require the retraction in durable records + report provenance instead; also count MEMBERS not bullets before calling a per-arm row-count literal false
metadata:
  type: feedback
---

Two rulings from #2329 `q35_ladder_decay` round 2 (2026-08-20), both confirmed workable:

1. **Implement-branch closure of a false "inherited" plan premise.** Round 1 caught a component
   registered 4× as "inherited byte-verbatim" that the parent never had (fabricated audit row).
   The union offered IMPLEMENT or disclose+amend. The implementer chose IMPLEMENT: the registered
   WHAT is then delivered exactly and only the provenance adverb in the plan text stays false.
   Ruling: do NOT demand a standalone plan amendment for the residual adverb — under v2 a
   `new-plan-version` re-parks at the human approval gate, a real cost disproportionate to a
   provenance descriptor with zero verdict surface. The bar instead: (a) the retraction must exist
   in ≥2 durable records (commit message, implementation marker, function docstring all count);
   (b) the REPORT phase must carry the fork-new provenance and never quote the plan's "inherited"
   wording (the read-from-artifacts discipline); (c) any FUTURE plan version forced by other
   blockers folds the one-line correction in. Same logic extends to small unregistered
   implementation constants attached to the new component (e.g. a descriptive-only fold seed
   derived from a registered seed): acceptable when deterministic + recorded in the code AND in
   the persisted artifact's own description field — no amendment solely for it.
   **Why:** amendment cost (human re-approval park) must enter the remedy calculus; three durable
   retraction records already prevent the false premise from propagating.
   **How to apply:** on re-review of an implement-branch fix, verify BOTH halves separately —
   the component computes what the plan registers, AND no current report/marker still asserts the
   false provenance — then rate the plan-text residual a recorded non-blocking concern.

2. **Count MEMBERS, not bullets, before calling a row-count literal false.** A claimed
   "one per-arm row per member (32 rows)" marker realized 29 bullets — two bullets were COMBINED
   rows ("dispatcher.bank / dispatcher.anchors"; "dispatcher.stage1 / stage2 / all") covering 5
   members, reconciling exactly to 32. A naive bullet count would have minted a false
   fabricated-checkmark finding.
   **How to apply:** when grep-verifying a claimed count over roster-style markers, split
   multi-member bullets on "/" (or equivalent) and reconcile members before grading the literal.

3. **Line-order is not execution-order when grading "before any X" ordering claims** (round 3,
   same task). A guard claimed to fire "before any unit is appended" had a `units.append` at an
   EARLIER line number — but inside a closure DEFINITION invoked only after the guard. Check
   invocation order (who calls what, when), not textual order, before minting a false-ordering
   finding or blessing a true one.

4. **Additive HF persistence beyond declared destinations is not a deviation.** Ride-along
   artifacts (gates/pools/pilot files) uploaded into a plan-declared prefix whose §10 row names
   only a subset (scores), while their own declared git home is unchanged, is persist-by-default
   conformant when the ambiguity + reason are stated in the report — flag only a forward note for
   the upload-verifier (prefix will hold more than the named glob), not a finding. See also
   [[judgment-extra-outputs-vs-scope-creep]].
