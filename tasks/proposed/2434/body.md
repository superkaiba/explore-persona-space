---
title: verify_task_body.py check 20 counts figure alt text as prose, making the clean-result
  critic's alt-text NIT unsatisfiable near the cap
kind: infra
tags: []
created_at: '2026-08-20T23:03:37Z'
has_clean_result: false
parent_id: 823
origin_prompt: 'Surfaced by #823 round-9 clean-result gate: applying the critics''
  figure-alt-text NIT flipped verify_task_body.py from PASS to FAIL on check 20.'
workflow: v1
---
# Figure alt text counts toward `verify_task_body.py` check-20 prose budget, so the clean-result critic's accessibility NIT is unsatisfiable near the cap

## Goal

Resolve a direct, measured contradiction between two clean-result gates: the
`clean-result-critic` roster asks for descriptive figure alt text, while
`verify_task_body.py` check 20 counts that alt text as content prose and FAILs
the body for it. Near the per-result word cap the two cannot both be satisfied.

## Evidence — measured on #823, not inferred

Both halves of the round-8 and round-9 doubled clean-result gate raised
`figure-alt-text-title-like` as a non-blocking NIT. The Codex round-9 verdict
made the ask concrete: *"replace title-only strings such as 'Refit R-squared by
arm at read-out layers' with axis-and-trend descriptions"*, naming body lines
183, 193, 203, 219, 294, and flagging it `Mechanizable: no`.

I applied exactly that fix — five alt strings replaced with axes-and-trend
descriptions whose numbers were read from each figure's own caption blockquote,
so no new claim was introduced. Body grew 48,289 → 49,282 bytes (+993).

Before the patch: `verify_task_body.py --issue 823` rc=0, `OVERALL: PASS`
(75 checks).

After the patch, same command, rc=1:

```
[FAIL] v4 conciseness caps — result 'New-mask anchor refits collapse in R² yet retrie'
prose is 209 words (cap: must be <180; FAIL fires at ≥180 inclusive)
...
total content prose is 1988 words (budget 1550)
OVERALL: FAIL (1 of 75 checks failed)
```

The patch was reverted and PASS confirmed restored, so #823 is unaffected. The
five alt strings remain title-like and the NIT remains open on the ledger.

## The defect

Check 20's word counter treats markdown image alt text as content prose. Alt
text is not prose a reader reads — it is the accessibility substitute for the
image, invisible to a sighted reader of the rendered body — yet it consumes the
same budget the cap exists to protect against padded narrative. Consequences:

1. A result already near its cap makes the accessibility NIT **structurally
   unsatisfiable**: satisfying the critic FAILs the verifier.
2. The incentive runs backwards — the cheapest way to pass check 20 is to
   SHORTEN alt text, i.e. to introduce exactly the defect the critic flags.
3. The conflict is invisible at review time. The critic labels the NIT
   non-blocking and PASSes; the FAIL only appears if someone actually attempts
   the fix.

## Candidate fixes (for the planner — not pre-decided)

- Exclude image alt text from check 20's prose word count (strip `![...]` alt
  spans before counting). Most direct; leaves both gates' intent intact.
- Give alt text its own separate cap, so it stays bounded without competing
  with narrative.
- Alternatively, if counting alt text is deliberate, have the critic suppress
  alt-text NITs for results within some margin of the cap — but that concedes
  the accessibility goal, so the first option looks better.

Whichever is chosen, add a regression test pinning the chosen semantics: a
fixture body whose alt text alone would cross a cap.

## Scope

`scripts/verify_task_body.py` (check 20), possibly
`.claude/rules/clean-result-critic-lens-reference.md` /
`.claude/skills/clean-results/SPEC.md` for the cap's stated definition, plus a
test under `tests/`. No experiment, no GPU.

## Provenance

Surfaced during #823's round-9 clean-result gate (same-issue follow-up
`inconsistent-origin-persona-ladder`). Both reviewer halves PASSed; this is the
NIT they both left open, and the reason it stayed open.
