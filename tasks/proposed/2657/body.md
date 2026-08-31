---
title: Step 9c gate-fleet arbitration counts a session's own detached ledger refresh
  as a foreign contender, deadlocking that session's own gate
kind: infra
tags:
- wf-fix
created_at: '2026-08-31T08:41:07Z'
has_clean_result: false
parent_id: 2654
origin_prompt: 'Surfaced by /issue 2654: the Step 9c relaunch chain burned its full
  3600s arbitration ceiling because probe --fleet counted this issue''s OWN prior-round
  ledger refresh (the refresh pseudo-issue, 4650s bound) toward EPM_GATE_FLEET_MAX=2
  alongside one genuine foreign gate; --exclude-issue is int-typed so the refresh
  cannot be excluded. 30/30 probes saw exactly that pair; gate never launched (FATAL
  90); slot opened seconds after the refresh bound expired.'
workflow: v1
---
# Step 9c gate-fleet arbitration counts a session's OWN detached ledger refresh as a foreign contender, deadlocking that session's own gate

`scripts/step9c_baseline.py probe --fleet` counts DISTINCT foreign issues with
live gate trees — **including the ledger-refresh pseudo-issue** — and exits 3
when the count reaches `EPM_GATE_FLEET_MAX` (default **2**).

The `refresh` entry is frequently the CALLER'S OWN detached process: `/issue`
Step 9c kicks a detached `step9c_baseline.py refresh` when the compare reports a
stale ledger, and that refresh carries a 4650s (~77 min) bound. So for the next
~77 minutes the session that kicked it has an effective foreign-contender budget
of **one**, and a single genuine foreign gate saturates the cap and blocks the
session's own next gate round.

There is no escape hatch: `--exclude-issue` is typed as an `int`
(`--exclude-issue refresh` → `invalid int value: 'refresh'`, exit 2), and it is
documented as dropping "the caller's own issue" — which the refresh pseudo-issue
is not, structurally, even when the caller kicked it.

## Measured incident (issue #2654, 2026-08-31)

After the #2311 root-cause merge, the Step 9c relaunch chain waited on
`probe --fleet` for its full 3600s ceiling and never got a slot:

- 30 of 30 probes over the window saw exactly two contenders: `issue=2650`
  (a genuine foreign lint gate) and `issue=refresh` — this issue's OWN
  prior-round refresh, launched at 07:24:57Z with a 4650s bound.
- The chain exited FATAL 90, gate NOT launched. One full hour of wall lost.
- The slot opened within seconds of the refresh's bound expiring (probe rc=0
  with `issue=2650` still live and alone), confirming the refresh was the
  binding contender, not fleet pressure.

The relaunch then succeeded on its first probe (`arbitration slot OPEN after
0s`), so nothing about the gate itself was at fault.

Aggravating factor: that same refresh had already blown its 4350s internal
pytest bound three times historically (`refresh pytest timed out ... NO ledger
write` x3 in `logs/step9c_baseline_refresh.log`), so the process that consumed
the arbitration slot for 77 minutes frequently produces no ledger at all.

## Why this is worth fixing rather than working around

The refresh is REPORT-ONLY and already single-flight-locked, so it is not the
kind of contender the cap exists to bound — the cap exists to stop N concurrent
full-suite gate runs melting the shared VM. Counting it costs a gate round while
protecting nothing that its own lock does not already protect.

Note also that the compare's authoritative classifier is the per-node
`--run-pristine` scratch oracle, not the ledger (in this issue's prior round it
stripped 8 of 9 failures `via: pristine-scratch` while the ledger read
`stale: true` without blocking the verdict). So the refresh's output is advisory
for the very gate it was blocking.

## Candidate fix directions (for the spawned session's planner to adjudicate)

1. Attribute the refresh pseudo-issue to the issue that kicked it and drop it
   under `--exclude-issue <caller>`.
2. Accept the literal `refresh` as an `--exclude-issue` value (widen the type).
3. Do not count a report-only, single-flight-locked refresh toward the cap at
   all — bound it by its own lock.
4. Ordering fix at the call site: do not kick a detached refresh that will
   immediately contend with the same issue's next gate round; kick it after the
   round's terminal transition, or make the kick conditional on no pending
   relaunch.

Whichever direction is chosen, keep the cap's real purpose intact (bounding
concurrent full-suite GATE runs) and keep the refresh's single-flight lock.

## Files of record

`scripts/step9c_baseline.py` (`probe --fleet`, `--exclude-issue`, the refresh
pseudo-issue in the fixed internal signature union; `EPM_GATE_FLEET_MAX`,
default 2, #1962); `.claude/skills/issue/SKILL.md` Step 9c (the refresh kick);
issue #2654 `events.jsonl` (the 30/30 contender census + the FATAL 90 chain
record).
