---
title: 'Step 10d pre-push gate: a crash verdict names no cause (rc values computed,
  branched on, then discarded)'
kind: infra
tags:
- step10d-gate-rc-telemetry
created_at: '2026-08-21T22:14:59Z'
has_clean_result: false
parent_id: 2241
origin_prompt: '/issue 2241 — Step 10d crash-verdict recovery: the gate verdicted
  crash with every compare artifact clean and no persisted rc, so the spec-mandated
  ''fix the crash cause'' was unactionable'
workflow: v1
---
# Step 10d pre-push gate: a `crash` verdict names no cause (rc values computed, branched on, then discarded)

## The gap

The Step 10d pre-push workflow-lint gate
(`.claude/skills/issue/steps/18-step-10d.md`, the executable gate block) computes
six return codes — `GT_RC`, `BASE_RC`, `GATED_RC`, `TG_RC`, `TG_BASE_RC`,
`TG_CRASH` — branches on all six at the verdict `if`, and then **persists none of
them**. The verdict file gets one word (`pass` | `block` | `crash` |
`skip-artifact-only`) plus the certified sha, and the trailing
`echo $? > /tmp/step9c-lint-rc-issue-<N>` records only the final `cat`'s status,
which is `0` on every run regardless of verdict.

Consequence: a `crash` verdict is undiagnosable after the fact. The gate log
carries the four descriptive leg lines (choom, landing-union, lint-vintage,
mapped-baseline helper) and nothing about WHY the crash arm fired.

## Why that matters — the spec mandates an action the session cannot take

Verdict case 3 says:

> `crash` — ... fix the crash cause in the worktree, re-run the gate ONCE;
> still crashing → the SAME `epm:merge-failed v1` handling as case 1.

"Fix the crash cause" presupposes the cause is knowable. It is not. The session's
only options are to guess, or to burn another full gate run (~2h15m wall on
#2241) purely to learn which arm fired — and that second run still records
nothing, so a third crash is equally opaque.

## Observed incidence (#2241, 2026-08-21T20:10Z)

The gate verdicted `crash` with **every compare artifact clean**:

- `lint-new.txt`, `lint-owndiff.txt` — both EMPTY
- both lint legs ended `workflow_lint: PASS`
- `tg-new.txt`, `tg-new-nodes.txt` — both EMPTY
- both test legs: `1 failed, 2484 passed`, the same single pre-existing
  main-side red

So the payload was clean and the crash was instrument-level — but which
instrument was unrecoverable. Reconstructing even a partial answer took a
forensic pass over ten artifact files plus three live re-probes (gate-tree
rebuild, disk headroom, re-running the suspect lint invocation at repo root and
against a fresh gate tree), and still landed only on a hypothesis: the second
lint invocation contributed zero bytes in BOTH legs, the signature of a kill
with block-buffered stdout discarded.

## Fix

Emit the six values to the gate log immediately before the verdict `if` — one
additive `echo`, no change to any variable the verdict reads:

```bash
echo "[step10d] gate rc telemetry: GT_RC=${GT_RC} BASE_RC=${BASE_RC} GATED_RC=${GATED_RC} TG_RC=${TG_RC} TG_BASE_RC=${TG_BASE_RC:-unset} TG_CRASH=${TG_CRASH}"
```

Worth pairing with per-leg lines after each of the four lint invocations
(cumulative rc + output bytes — that pair is what localizes a zero-byte leg) and
one after the mapped-baseline call (`TG_BASE_RC` / `TG_CRASH` / selected /
scratch). #2241 ran its re-launch with exactly these six echoes added ad hoc;
lift that shape into the spec so every gate run gets it.

Consider also folding the telemetry line into the verdict FILE (line 3+) rather
than only the log, so the § Successor / re-entry rule — which probes the verdict
file, not the log — can read the cause after a session death. #2241's crash was
in fact discovered by a successor session, which is the case where the log is
least likely to be consulted.

## Acceptance

1. A gate run at any verdict emits the six rc values to
   `/tmp/issue-<N>-lint-gate.log`.
2. The verdict `if` condition is byte-unchanged; `pass` / `block` / `crash` /
   `skip-artifact-only` outcomes are identical to today's for the same inputs.
3. A `crash` verdict is attributable to a named arm from the log alone, with no
   re-run and no live probing.

## Provenance

workflow_fix_target: .claude/skills/issue/steps/18-step-10d.md

Surfaced by the #2241 `/issue` orchestrator on 2026-08-21 while recovering a
`crash` verdict left by a gate that completed after its owning session died.
