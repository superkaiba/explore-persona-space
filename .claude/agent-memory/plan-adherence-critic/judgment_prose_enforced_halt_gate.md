---
name: judgment-prose-enforced-halt-gate
description: A plan-registered HALT gate realized as a standalone CLI with runbook-prose-only invocation — adjudicate by sibling-gate wiring idiom + silent-failure-mode of the protected spend
metadata:
  type: feedback
---

When a plan registers a gate (§7 "HALT on mismatch", DAG "BEFORE <spend phase>")
and the diff realizes the CHECK as a standalone CLI but wires NO invocation into
any executing surface (driver phase entry, dispatch script, artifact poll), do
not accept "runbook precondition" prose on the implementer marker as adherence
when (a) every SIBLING gate in the same diff IS mechanically wired or
artifact-polled (the diff's own idiom proves the cheap fix exists), (b) the
protected spend is large (e.g. a ~145k-call judge wave), and (c) the downstream
failure mode is silent (dict `.get` join). Ruling shape: declared-with-reason ⇒
CONCERNS revise-then-merge (not FAIL), remedy = the existing gate-artifact-poll
idiom (phase entry refuses absent a matching PASS report, explicit --skip flag
as recorded override).

**Why:** #2389 r1 — M1-iv staged-anchor consumer probes landed as
`issue2389_consumer_probe.py` + 6 tests in the final commit, invoked only by
marker prose; grep showed zero references from phase_waves / analysis steps /
dispatch.sh, while share_prefill arming, vllm parity re-route, and rule-26
pilots all polled gate artifacts mechanically.

**How to apply:** grep the probe/gate script's basename across every executing
surface in the diff; enumerate sibling gates' enforcement mechanisms as the
in-diff idiom baseline; check whether any repo-resident runbook artifact for the
relevant phases even exists (a marker is not a runbook — future executors don't
load it). Related: [[judgment-registered-trigger-enforcement-inplan]].
