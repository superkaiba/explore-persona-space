---
name: prior-prompt-splice-and-distilled-ruling
description: Lean-twin recipe for revision-round code-review composes — splice the prior round's /tmp prompt rubric span blind (sed ranges, never paged into context) instead of re-copying the 109 KB code-reviewer.md, and encode a reconciler ruling from the orchestrator brief's distillation when the brief bans other reviewers' verdict bodies (#2384 r4)
metadata:
  type: feedback
---

Shape (#2384 r4, 2026-08-30): post-reconciler single-blocker fix round; brief
distilled the binding ruling (union parts (a)/(b), three named instances,
out-of-scope fences) and BANNED including any other reviewer's verdict,
findings, or conclusions; one ruling claim was to be posed as a QUESTION.

1. **Rubric via blind splice, not re-copy.** The full spec orders copying
   code-reviewer.md (109 KB) verbatim — un-windowable for the lean twin. When
   a prior round's prompt exists in /tmp (here 163 KB r3), map its structure
   with `grep -n '^# \|^## \|^---BEGIN\|^---END\|epm:code-review-codex'`,
   then `sed -n '<start>,<end>p'` the invariant `## Review Protocol`..`## Rules`
   span into the new prompt WITHOUT reading it into context. Guard the splice:
   grep the span for round tokens (`round 3`, old SHAs, `v<n>`) first — hits
   at generic incident references are fine, range/sentinel hits are not.
   Validate the assembled file with counts: envelope tokens `grep -cxF` == 1
   each, sentinel v<round> == 1, zero prior-sentinel residue, zero old-range
   residue, zero `{{` residue, exactly one line-start `CONCERN:: ` row
   (the template's `CONCERN:: none`), key rubric tokens (Step 0.7/0.9/3.7/4.5,
   grep-the-literal) present.
2. **Brief's distillation beats inlining the ruling verdict.** The r2/r3
   sibling pattern inlines the reconciler verdict as contract; when the brief
   distills the ruling AND bans other reviewers' verdict bodies, encode the
   distilled order as a `# ROUND-N CONTRACT` section attributed to the ruling
   — never fetch/inline the verdict (it carries the reviewers' findings the
   ban targets, and would prime the twin on the brief's open question).
3. **Open-question-as-question (Q1).** A ruling claim the brief wants judged
   ("part (b) alone would not close instance (iii)") gets its own contract
   subsection + a dedicated verdict answer line with a holds/does-not-hold/
   undecidable enum and a mandatory trace — framed explicitly as "a QUESTION,
   not a premise", with both answers declared valuable so the twin is not
   nudged either way.
4. **Post-marker ledger bookkeeping row.** An `addressed` event seconds AFTER
   the impl marker ts claiming closure at the round HEAD is the round's own
   bookkeeping: exclude it from the open-rows snapshot, name it, and instruct
   "score on the code, not the row; its existence is not evidence".
5. **Historically-unable-to-execute posture.** Keep the sibling's sanctioned
   F12 execution list ("if your runtime permits") but lead with the brief's
   framing: provenance labels `[verified-by-execution]` vs `[read-and-reasoned]`
   on every priority/closure line, plus a named could-not-verify list in the
   Tests section — a scoped "could not check X" beats inference presented as
   confirmation.

**Standing-recommendation (PASS) round variant (#2384 r5, 2026-08-30):** the
same splice recipe held (r4 prompt as donor, guard greps clean, 141 KB
assembled; all count checks passed first try). Deltas vs the FAIL-fix shape:

- Reconcile was PASS with two persisted CONCERNs + two standing
  recommendations the round implements exactly → compose the sibling
  #2552-r4 flip ("the recommendations ARE the round contract;
  NOT-implemented / falsely-claimed = substantive FAIL; honest disclosed
  partial ⇒ CONCERNS + same-id row") and the ledger split: target ids
  latest=addressed ⇒ a failed closure REQUIRES the same-id row (the
  re-open mechanism). Attest the superseded severity chain (raised BLOCKER
  → reconciler deferred → re-raised CONCERN) or the twin re-escalates from
  the stale BLOCKER row.
- Brief handed over prior /tmp harness dirs (implementer mutant legs +
  reconciler mimic-base): sanction REUSE in F12 but with a
  provenance-verification duty — diff each baseline/mutant against the git
  ref it claims to mirror before trusting it; a mismatched harness is
  evidence-VOID, rebuild from `git show`.
- Execution posture flipped: Codex executed successfully in r4 → lead with
  "execution EXPECTED available"; keep the never-fabricate +
  per-finding provenance labels unchanged.
- Impl marker head sentinel carries the implementer's ROUND numbering (v5)
  while the events.jsonl version is higher (v7) — pre-attest the mapping in
  the marker intro, or the twin flags it as a shape defect.

See sibling memory (codex-code-reviewer): feedback_closure_verification_round_compose
(#2384 r2), feedback_measured_deviation_fix_round_compose (#2384 r3),
feedback_concern_discharge_round_severity_fence (the #2552-r4 flip + row
rules), feedback_revision_round_compose_recipe (the general round-2+ deltas).
