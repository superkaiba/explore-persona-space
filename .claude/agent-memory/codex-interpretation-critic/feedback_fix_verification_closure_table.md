---
name: fix-verification-closure-table
description: On fix-verification rounds, add a "Round-1 Finding Closure" section as the FIRST output section — one line per union finding with a VERIFIED-ADDRESSED|NOT-ADDRESSED enum + quoted landed body line — and tie the PASS rule to all lines verified
metadata:
  type: feedback
---

When the brief says a round's core is closure adjudication of prior-round
findings (the analyzer posted concern-addressed rows and a dispositions
marker), compose the output template with a dedicated `### Round-1 Finding
Closure` section INSERTED right after the Verdict line, one line per union
finding (both critics' findings, deduped where they merged — e.g. a Codex
NIT == a Claude MINOR gets one shared line), each carrying:
`<id> (<severity>) — VERIFIED-ADDRESSED | NOT-ADDRESSED — [quoted landed
body line] — [one-line adjudication naming the number/artifact checked]`.
Then state the VERDICT RULE explicitly: PASS requires (a) every closure
line VERIFIED-ADDRESSED with quoted evidence AND (b) no new substantive
finding in the round's delta; any NOT-ADDRESSED or delta regression =
REVISE, with re-raises getting NEW round-N concern ids that name the old
id in the summary.

**Why:** the fixed 7-lens template scatters closure evidence across lenses;
the reconciler and orchestrator need one section that answers "did the
eight fixes land" without reassembling it. The #722/#665 quoted-line rule
already binds applied/absent claims — the closure table gives those quotes
a home.

**How to apply:** build the fix checklist at compose time from the
interpretation marker's per-finding dispositions (one numbered item per
claimed fix, each with the claimed landed line + the artifact to verify it
against + a recompute recipe where the fix carries numbers), anchor-grep
every claimed line in the materialized canonical body FIRST (a missing
anchor at compose time means the checklist item needs a locate-it-yourself
instruction instead of a quote). Pair with [[feedback-lens7-carried-forward-on-revision-rounds]]
(delta-scope + carried lines) and [[compose-time-ledger-snapshot]] (ledger
rows are the analyzer's CLAIM, not ground truth — say so in the prompt).
First applied #2564 k100 round 2 (2026-08-26).
