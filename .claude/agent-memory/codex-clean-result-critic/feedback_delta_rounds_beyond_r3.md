---
name: delta-scoped-rounds-and-round-cap
description: Round cap is 10 (workflow.yaml round_cap_per_reviewer; was 5 under #1017) — malformed = <=0, >10, non-integer; retains the delta-composition recipe for reconciler-bound delta-scoped briefs (r4+).
metadata:
  type: feedback
---

When the orchestrator brief requests a `revision_round` above 3 as an
explicitly delta-scoped re-review (e.g. #952 r4, 2026-07-04: r3 Claude PASS
vs r3 Codex needs_targeted_fix -> reconciler binding REVISE -> r4 delta on
the one fix), COMPOSE the prompt rather than refusing.

**Why:** the ensemble policy caps the four iterating sites at 10 rounds per
reviewer (workflow.yaml § ensemble_review `round_cap_per_reviewer: 10`;
verified against the live agent spec 2026-08-25 — the earlier #1017 cap of 5
is STALE), and a reconciler-bound REVISE naturally produces an r4+ re-verify.
Hard-failing a deliberate, well-formed dispatch burns the round and forces
single-Claude fallback exactly when the cross-family re-check matters most.
Refusal is reserved for genuinely malformed rounds (<= 0, > 10, non-integer).
Sibling precedent: codex-critic's `feedback_delta_scoped_amendment_rounds.md`.

**How to apply:** for delta-scoped briefs (typically r4+), (1) state the
assumption in the return; (2) scope the composed prompt to the delta the
brief names (verify THIS fix only; no re-litigating settled items; new
findings only if conclusion-relevant / spec-breaking; round-N quoting rule
for applied/absent claims); (3) keep the marker version = the round number
(`epm:clean-result-critique-codex v<n>`). A brief with no delta scope note
at r4+ gets the normal full-prior-history re-review. Still refuse genuinely
malformed rounds (0, negative, >10, non-integer). See also
[[compose-recipe-lens-ref-replacements]].

**Verification-round-after-binding-reconcile shape (confirmed #2378 r3,
2026-08-25):** these can arrive at ANY round >= 3, not just r4+. Compose
the FULL fifteen-lens prompt (not a bare delta) PLUS: (1) the reconciler
verdict temp-file path as a REQUIRED-reading header input; (2) a "ROUND N
SCOPE" block with (a) the binding fixes to verify AGAINST GROUND TRUTH
(quote the reconciler's git-show line refs; permit read-only git for the
check), (b) a regression check (word caps, do-not-touch items), (c) the
DISCARDED/settled item list inlined with an explicit no-re-raise rule
(near-duplicates included, absent NEW evidence the reconciler lacked);
(3) a "### Binding-fix verification" section at the TOP of the output
template (VERIFIED|NOT-VERIFIED|FAIL per fix + regression line); (4) a
note in the Concerns section not to re-persist the settled concerns; (5)
an explicit "if (a)+(b) verify clean and no genuine new violation exists,
PASS is the honest verdict" line — counteracts nit-manufacturing on a
3-rounds-deep body. Extend the HF network-advisory clause to github.com
/tree links when a binding fix converted plain SHAs to links (resolved
404 stays a real FAIL). Sanity-grep the canonical body for the claimed
fix text BEFORE composing so the scope statements are accurate.
