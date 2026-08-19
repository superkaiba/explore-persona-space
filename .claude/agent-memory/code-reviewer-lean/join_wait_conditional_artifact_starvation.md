---
name: join-wait-conditional-artifact-starvation
description: "Fan-out JOIN waits must cover every REGISTERED alternative artifact shape (rung file vs unmappable marker) with an either-of predicate; and a designed re-run round (recalibration/--round 2) whose regime delta is invisible to the argv-sha resume key silently SKIPs every step and reproduces round 1 (#2378 R1 g5)"
metadata:
  type: feedback
---

Rule: when reviewing a multi-pod/multi-shard dispatcher, (1) take every JOIN
wait predicate (`_git_wait_for`-style path polls, sentinel waits) and diff its
expected-path list against ALL artifact shapes the producer can legitimately
write — a plan-REGISTERED alternative outcome (a `__unmappable.json` marker
written INSTEAD of `__rung*.json`; a drop marker instead of a result file)
starves the join forever: the merge pod polls a never-appearing path for the
full timeout (24 h default) and dies rc=1, killing every downstream summary +
the terminal `epm:results`. Fix shape: either-of predicates per entry, or wait
on the sibling's per-leg DIGEST (harvested last, subsumes completion).
(2) Take every DESIGNED re-run path (a G1 recalibration round, a `--pilot-round
2` / `--wave 2` re-pilot the code's own trip message instructs) and ask what in
the resume key changes: if the round's regime delta lives OUTSIDE the argv
(bank/prime file contents, code edits), an argv-sha OK-flag resume SKIPs every
step, the digest recomposes from round-1 outputs, and the re-run wrongfully
hard-fails — the voluntary-resume-key law
([[trap-sentinel-stale-and-voluntary-resume-key]]) re-hit in OK-flag form.

**Why:** #2378 R1 g5 (2026-08-19) had BOTH in one dispatcher commit that passed
a 9/9 CPU probe suite: fits-d's sibling wait expected `chat_to_<c>__rung9.json`
while the ladder writes `chat_to_<c>__unmappable.json` instead on the
registered U≤0 lattice branch; and `p1_pilot --pilot-round 2` re-used the
round-1 OK-flags (child argvs carry no round token; recalibration changes bank
CONTENTS, not argvs) so the instructed recalibration round deterministically
returned the round-1 trip as RC_G1_FAIL. Probes cannot catch either: both fire
only on the conditional branch a fixture-suite never routes.

**How to apply:** any multi-phase dispatcher review (Step 0.69 adjacencies).
Grep join waits for their expected-path construction and cross-grep the
PRODUCER for every `atomic_write_json`/output-name pattern it can emit for that
unit; grep re-run flags (`--pilot-round`, `--wave`, `--round`) for whether they
reach the resume key (step name, argv, or logs dir).
