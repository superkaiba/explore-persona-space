---
name: concurrent-composers-and-verification-rounds
description: Lens-unique temp-file slugs (sibling composers run concurrently and overwrote a generic scaffold) + the round-2+ verification-round prompt shape (per-blocker FIXED/PARTIALLY/NOT-FIXED block, orchestrator VERDICT contract overrides the spec template)
metadata:
  type: feedback
---

Two composer lessons from a workflow-v2 round-2 panel compose.

1. **Lens-unique temp-file slugs, always.** Three sibling lens composers run
   CONCURRENTLY on the same issue; in one round a generically-named scaffold
   file was overwritten by a sibling. Name EVERY temp file (prompt, parts,
   handed-span files, verifier script) with a lens-unique slug, e.g.
   `/tmp/cec<N>r<K>-<purpose>.md` — never `/tmp/planbody.md`-style names, and
   don't rely on the spec's default `/tmp/codex-efficiency-critic-<N>-prompt.md`
   being unique enough for the SUPPORT files.
   **Why:** /tmp is shared across the concurrent composer spawns; a lost
   scaffold silently corrupts a sibling's compose.
   **How to apply:** pick the slug before the first write; use it for all
   spans/parts/verify files; assemble by cat-ing part files so verbatim spans
   never page through context.

2. **Verification-round (round 2+) prompt shape.** When the orchestrator brief
   declares a VERIFICATION round with surviving blockers and its own output
   contract, the brief's contract overrides the spec's Step-3 template: keep
   the `epm:plan-critique-codex v<K> lens=efficiency` marker wrapper, but lead
   with the brief's `VERDICT: PASS | REVISE | REJECT` line and add a
   `### Round-1 blocker verification` block (one FIXED / PARTIALLY FIXED /
   NOT FIXED row per surviving blocker, evidence-cited, with any residual
   re-stated as a numbered blocker) BEFORE the fresh `New Must Fix` section.
   Carry the brief's claimed-fix summaries nearly VERBATIM under a
   "VERIFICATION CONTEXT (claimed — verify, never take on faith)" header: the
   near-verbatim quoting is also what lets the brief-span leg of the
   numeric-leak verifier clear their figures (see
   [[numleak-handed-span-set]]). Where the planner recorded a deliberate
   DEVIATION from a blocker (e.g. a fence floor argued down via p90
   arithmetic), direct Codex to re-do the arithmetic itself and to check the
   deviation's failure mode (accepted-then-killed gap), not just re-read the
   plan's claim. Explicitly instruct: clean revision ⇒ PASS, do not
   manufacture a blocker to justify the round.
   **Why:** round-2 twins that only get the standing lens re-litigate settled
   findings and miss whether the specific fixes landed; and paraphrasing the
   brief's numbers (instead of quoting) creates verifier false positives.
   **How to apply:** any brief containing "VERIFICATION round" + a surviving
   blocker list; rig-code paths the plan cites go in as read-target lines
   (Codex has file access) with the specific cited line noted once.
