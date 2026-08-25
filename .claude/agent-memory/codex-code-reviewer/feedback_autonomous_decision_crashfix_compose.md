---
name: autonomous-decision-crashfix-compose
description: "Crash-fix round implementing an AUTONOMOUS epm:failure-recorded decision (#2546 r7): the failure record IS the decision envelope (no user ruling), the settled fence carries a plan-clause VERIFY duty, a smoke-REACH-WIDENING probe-append is 0.71-live-but-not-a-trigger, and prior-round seam attestations (zero-Mock) must be re-grepped every round"
metadata:
  type: feedback
---

From #2546 r7 (2026-08-25), the sibling of [[user-ruling-crashfix-round-compose]]
when the settled decision is ORCHESTRATOR-AUTONOMOUS, recorded inside the
`epm:failure` marker itself (drop+report zero-`<<` GSM8K golds) rather than a
user ruling:

1. **One decision envelope, not three.** The failure record carries crash +
   census + rejected-alternatives + decision — inline it as
   `CRASH-DIAGNOSIS + DECISION RECORD` and drop the probe/ruling envelopes.
   The settled fence gains a third leg the user-ruling shape lacks: the twin
   VERIFIES the plan-clause permission CLAIM (quote the §4.1 runtime-attrition
   line by plan line number; "if the plan text did NOT permit this shape, that
   is a substantive finding about deviation-recording, not a re-litigation").
2. **Re-grep seam posture EVERY round — never carry a prior round's
   attestation.** r6's compose truthfully said "ZERO Mock/monkeypatch in the
   test file"; r7 added ONE `monkeypatch.setattr(S, "load_dataset",
   fake_load_dataset)` network-boundary fake. Blind template reuse would have
   shipped a false composer attestation; the Step 3.8 note flipped from
   trigger-N/A to a signature-conformance duty (fake mirrors the production
   call shape `load_dataset("openai/gsm8k", "main", split=...)`; its return
   supports the body's accesses; bare `Mock()` never body evidence).
3. **Smoke-REACH-WIDENING branch under 0.71.** A probe-append that widens the
   smoke slice to a known-offender row is a smoke-conditional ADD that is NOT
   an (a)/(b) trigger (substitutes nothing, downgrades nothing) — state
   "LIVE this round" but scope `smoke-blind-spot-unenumerated` to ADDITIONAL
   unenumerated branches; composer-verify the enumeration NAMES the new
   ROW-INDEX-REACH class the crash exposed (the dispatch note bound it as a
   FAIL condition). Verify duty: the probe guard fails LOUD when the index is
   absent (a silently-skipping probe re-opens the blind spot).
4. **Churn trap recurred verbatim** (see [[user-ruling-crashfix-round-compose]]
   item 7): brief + dispatch note quoted "+229"/"+118" = combined `--stat`
   churn (true numstat +195/−34, +117/−1). Re-derive; state in-prompt as
   not-a-finding; flag in the return.
5. **Sentence-level rep1 leaves sibling tokens in the same template line** —
   the L146 classification note carried BOTH a `--name-only <sha>..HEAD` probe
   and the diff-size sentence; patching only the sentence left a third
   short-SHA survivor for the global migration count (assert said 2, real 3).
   When a line carries multiple round-scoped tokens, count survivors per
   TOKEN, not per note.

**How to apply:** any crash-fix brief saying "the settled decision (must not
relitigate)" whose decision lives in the failure marker: single decision
envelope + plan-clause verify leg + per-round seam re-grep + widening-branch
0.71 scoping. Compose script: `/tmp/codex-2546-r7-compose.py` (ephemeral).
