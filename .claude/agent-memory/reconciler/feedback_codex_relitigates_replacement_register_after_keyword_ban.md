---
name: Codex re-litigates the plain-English REPLACEMENT after a prior keyword ban
description: clean-result-critic — after a binding reconcile bans an enumerated jargon keyword, Codex round N+1 FAILs the plain-English words that replaced it as "the same concept under another name"; check the actual ban list, not the concept
type: feedback
---

When a clean-result-critic round-N reconcile UPHELD a ban on an **enumerated
keyword** (e.g. `preregistered` / `registered` / opaque `Mode-b` code), the
analyzer rewrites to plain English, and round N+1 Codex FAILs the
REPLACEMENT words as "the same concept under another name." This is a Codex
register-preference over-fire — DISCARD it unless the replacement hits an
ACTUAL ban-list entry.

**Why:** The clean-result-critic ban lists are ENUMERATED keyword bans, not
concept bans. Lens 7 pre-registration ban (clean-result-critic.md:1291-1296)
lists exactly `"pre-registered"`, `"pre-reg"`, `"registered hypothesis"`; the
`audit_clean_results_body_discipline.py` `pre_reg` regex covers
`pre-?registered|pre-?registration|pre-reg|registered hypothesis|registered alpha|fail at the gate|passed the gate|gate-pre-?registered`.
The spec explicitly PERMITS the underlying concept elsewhere ("Pre-reg
threshold values can sit in the Methodology Training hyperparameter table") —
so the temporal honesty ("the rule was fixed before the sweep") is fine; only
the lab-process KEYWORD is banned. Lens 6 voice bans *undefined* jargon +
*opaque codes* (`sw_eng_C1`, `cond_4`, `M1`, `Bin C`), NOT defined terms of
art. CLAUDE.md's metaphor ban is *spatial/anatomical* only ('spine',
'backbone', 'scaffold').

**How to apply (the 3-step DISCARD test on a round-N+1 register FAIL):**
1. Grep the EXACT replacement phrase against (a) the Lens 6/7 ban lists
   verbatim, (b) the audit regex, (c) SPEC.md voice. Run the audit script on
   the live body — a PASS is strong evidence the replacement is clean.
2. If the phrase hits NO enumerated entry → it is register PREFERENCE.
   Tells it is a synonym swap not a removal: Codex's OWN suggested
   replacement is semantically identical (#715 r2: "fixed in advance" →
   Codex's "was defined to succeed only if" — both assert the rule predates
   the data).
3. Check whether the PRIOR reconcile's remedy PRESCRIBED the now-flagged
   word. If the earlier binding reconcile told the analyzer to use that
   phrasing, banning it now CONTRADICTS the prior binding reconcile — a hard
   stop against REVISE.

**Worked datapoint — #715 r2 (clean-result-critic), verdict PASS:**
- R1 reconcile banned `preregistered` (figure title) + `Mode-b`/`registered`
  (enumerated keyword hits) and prescribed "plain-English **no-overlap kill**
  phrasing" as the remedy.
- R2 Codex FAILed the three replacement phrasings: "kill criterion fired" /
  "no-overlap kill" (Lens 4/6 "lab jargon"), "fixed in advance" (Lens 7
  "pre-registration under another name"), "fired" (Lens 6 "anthropomorphic").
- All three DISCARDED: "kill" is defined project vocabulary used verbatim in
  THIS task's plan.md:698/704 + CLAUDE.md + planner.md:234 (and the R1
  reconcile itself mandated "no-overlap kill"); "fixed in advance" is plain
  English the keyword ban does not reach (audit PASSes); "fired" is idiomatic
  event-condition English, not a spatial/anatomical metaphor.
- The "cycle on subjective register edits forever" failure mode is the real
  risk on the PASS side — banning the very phrasing the prior reconcile
  prescribed.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Codex re-litigates the plain-English REPLACEMENT after a keyword ban](feedback_codex_relitigates_replacement_register_after_keyword_ban.md) — ban lists are ENUMERATED keywords not concepts; grep the replacement vs Lens 6/7 + audit regex; "kill" is defined project vocab, "fixed in advance" ≠ "preregistered"; don't ban the phrasing the prior reconcile prescribed. #715 r2 PASS.
