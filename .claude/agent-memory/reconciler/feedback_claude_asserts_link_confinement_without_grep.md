---
name: Claude asserts Lens 2 link-confinement PASS without grepping
description: Claude clean-result-critic PASSes Lens 2 standalone-section with "all [#K] links confined to **Why:**" but doesn't grep; Codex catches [#K] links inside ## Findings / ## Data
type: feedback
---

Claude `clean-result-critic` asserts the Lens 2 standalone-section rule as
PASS with a confident summary like "all `[#K]` issue links confined to
`**Why:**`" — but does NOT actually grep the body, and misses `[#K]` links
sitting inside `## Findings` prose and `## Data` capsules. Codex catches
these by enumerating the exact offending lines.

**Why:** The v3 Lens 2 rule (`clean-result-critic.md` line 395-400; SPEC.md
`### Findings` line 158-159) confines prior-issue links to `**Why:**` and
`## Reproducibility`. The FAIL trigger is exactly enumerated: a `[#K]` link
or bare `#K` in `## Takeaways`, `## Findings`, or a `## Data` capsule. It is
a HARD FAIL, not a nit — so a missed link flips PASS→REVISE. Claude tends to
read the `**Why:**` block, see the links there, and generalize to "confined"
without scanning the rest of the body. (Incident: #542 round 1, 2026-06-16 —
Claude said "all confined to **Why:**"; the body had `[#441]` in a finding
and `[#537]` in `### Evaluated with`. Reconcile = REVISE.)

**How to apply:** When adjudicating a clean-result-critic Lens 2 PASS-vs-FAIL
disagreement, run `grep -nE '#[0-9]+|eps.superkaiba.com/tasks/' <body>`
YOURSELF and check each hit's section. Links in `**Why:**` or
`## Reproducibility` are fine; links anywhere in `## Takeaways` / `## Findings`
/ `## Data` are enumerated FAILs and bind the verdict to REVISE. Do not trust
either critic's prose summary of "confinement" — the grep is one call and is
authoritative. Note the `**Rounds:**` table (in `## What I ran`) is a gray
zone: prose says `**Why:**` is "the ONLY place," but the enumerated FAIL list
omits `## What I ran` — don't let a Rounds-table-only hit bind on its own.

Companion to the Codex-over-reaches pattern: a Codex lens FAIL can be
correct on one sub-finding (the link hits) and an unanchored extrapolation on
another (e.g. a "Lens 10 requires X" that isn't in the Lens 10 rubric text).
Adjudicate each sub-finding separately; one real hit carries REVISE while the
over-reach is recorded out-of-scope.
