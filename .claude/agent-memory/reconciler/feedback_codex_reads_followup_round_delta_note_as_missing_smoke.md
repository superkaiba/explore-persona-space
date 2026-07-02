---
name: Codex reads a follow-up round's terse smoke delta-note as a missing-smoke FAIL
description: When code is UNMODIFIED in round N, its round-N carve-out is a delta-pointer to the round it was smoked in; verify the git diff + the earlier marker before upholding a smoke-run-missing FAIL.
type: feedback
---

Codex `code-reviewer` twin sometimes raises a `smoke-run-missing` /
`substantive` FAIL on a GPU-bound phase whose round-N marker carve-out reads
tersely (e.g. "Unmodified this round; dry-run exercises ...; signature/entry
unchanged") and does NOT re-list the three substitute smoke items (real CPU
smoke of the CPU-runnable portion / dispatcher dry-run / signature smoke +
per-item exit-0 digest). Codex reads that delta-note IN ISOLATION and concludes
the mandatory smoke is missing.

**Why it's usually wrong:** in a multi-round implementation, an UNCHANGED phase
carries a short delta-note pointing back to the round where its full smoke
landed. The evidence is present — just in the prior round's marker, for code
git-confirmed unmodified since.

**How to adjudicate (Step 2 verification, before upholding):**
1. `git diff <round-N-1-sha>..<round-N-sha> -- <the phase's script>` — if the
   file is NOT in the round-N diff, the phase is genuinely unmodified and a
   fresh smoke is not required this round.
2. Grep the EARLIER implementation markers (`events.jsonl`, the
   `epm:experiment-implementation v<k>` where the phase's code was NEW / last
   changed) for the full three-item carve-out.
3. Check the prior code-review markers — a prior `code-reviewer` PASS that
   verified "all three substitute items present" is corroboration.

If the full smoke is present for unchanged code → the finding is
Unverified/mistaken (Weight `Discarded`), NOT a surviving `smoke-run-missing`
blocker. This is the temporal analogue of the pre-existing/stale-state family
(`feedback_codex_litigates_pre_existing_in_round_n.md`): verify what the round
CHANGED, not the round's prose in isolation.

**Do NOT over-correct into a blanket PASS.** A GPU-bound phase's smoke IS
load-bearing (esp. a HALT-on-drift gate like an apply-parity probe run before an
expensive sweep). The exception is narrow: it applies only when the phase's code
is git-confirmed unmodified this round AND the full smoke is verified present in
the round it was last changed. If the code DID change this round and the smoke
was not re-run, the FAIL stands.

Driving incident: #667 v13 r2. Codex FAILed the `apply-parity-probe` carve-out
as missing the 3 substitute items; the probe was NEW in v12 (+293 lines) with a
full 3-item carve-out, git-confirmed untouched in the v13 span
(`daf1f2e3b5..f0fc80b7c2`), and both prior code-reviews verified the items
present. Discarded that Critical; upheld the co-raised fact-drop BLOCKER on its
own merits → FAIL.
