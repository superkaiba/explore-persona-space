---
title: Launcher env override clobbered by dispatcher .env re-source; wrapper's frozen
  environ gives a false fix-engaged signal
kind: infra
tags:
- workflow-fix
created_at: '2026-08-27T09:18:56Z'
has_clean_result: false
parent_id: 2546
origin_prompt: 'Diagnosed during #2546 arm 2 p5_fits rung-2 relaunch: the briefed
  launcher export of HF_HUB_ENABLE_HF_TRANSFER=0 would have been clobbered by the
  dispatcher''s own .env re-source, while the dispatcher''s exec-frozen /proc/<pid>/environ
  still reported 0 — a false fix-engaged positive. Rung 1 on ge_gate engaged only
  because HF_HUB_DISABLE_XET is absent from .env.'
workflow: v1
---
---
kind: infra
---

# A launcher env override is clobbered by the dispatcher's `.env` re-source, and the wrapper's frozen environ certifies it as engaged

## Goal

Close a false-positive channel in the fix-engaged verification contract. Today, applying an env
mitigation at the launcher and verifying it on the launched wrapper's `/proc/<pid>/environ` can
report ENGAGED while the work runs under the un-mitigated value. Two surfaces need the fix: the
gotcha itself needs recording, and `.claude/rules/crash-fix-rounds.md` needs to specify WHICH
process's environ counts as verification.

## Observed (issue #2546 arm 2, `p5_fits` rung-2 relaunch, 2026-08-27)

`p5_fits` died on an `hf_transfer` download fault. The documented remedy is wedge-ladder rung 2,
`HF_HUB_ENABLE_HF_TRANSFER=0`. The orchestrator's relaunch brief instructed adding
`export HF_HUB_ENABLE_HF_TRANSFER=0` to the pod launcher `/workspace/launch_issue_2546.sh`.

That would not have engaged. The chain:

    launch_issue_2546.sh        export HF_HUB_ENABLE_HF_TRANSFER=0
                                exec bash scripts/issue2546_dispatch.sh --arm 2 p5_fits
    issue2546_dispatch.sh       set -a; . ./.env; set +a          <-- re-sources .env
    pod .env                    HF_HUB_ENABLE_HF_TRANSFER=1       <-- clobbers back to 1
    fit child spawns            runs under =1

The dangerous part is the verification, not the clobber. The DISPATCHER's environ is frozen at
`exec`, so `/proc/<dispatcher_pid>/environ` still reads `0` after the re-source. An operator who
applies the launcher export and then checks the wrapper gets a clean ENGAGED signal for a
mitigation that is not in effect.

Caught by the experimenter subagent, which applied the launcher export PLUS a flip of the `.env`
line and verified on the CHILD process (which spawns after the re-source). Net child-env delta
stayed at exactly one variable, preserving the single-variable crash-fix discipline.

## Why this is not a one-off

The same session had already used a launcher export for rung 1 (`HF_HUB_DISABLE_XET=1`) on the
`ge_gate` phase and recorded it as verified on both pids. That verification was sound only by
accident: `HF_HUB_DISABLE_XET` is absent from the pod `.env`, so nothing clobbered it. Had the
variable been present, an entire wedge diagnosis and its PASS verdict would have rested on a
mitigation that was never applied, with a durable marker asserting otherwise.

The `.env` re-source is present across the `issue1336`/`issue2546` dispatcher lineage, and
project policy deliberately sets the HF accelerated-transfer variables shell-level in
bootstrap/GCE/SLURM, which is exactly why `.env` carries them. So the collision surface is the
normal case for HF transfer mitigations, not an unusual one.

Related trap in the same family, already recorded in `.claude/rules/upload-policy.md`:
`HF_XET_DISABLE` is a verified NO-OP alias for `HF_HUB_DISABLE_XET`. Both are "the mitigation
looks applied and has no effect." They deserve to be cross-referenced, because the diagnostic
question is identical: did the value reach the process doing the work?

## Requested changes (for the plan to choose among; not prescriptive)

1. **A `.claude/rules/gotchas.md` entry.** Name the mechanism (dispatcher re-sources `.env` after
   the launcher's exports), the false-positive tell (wrapper environ frozen at `exec`), and the
   two-part remedy (set it in the pod `.env` or prove no re-source; verify on the child).
2. **Make the fix-engaged contract name the process.**
   `.claude/rules/crash-fix-rounds.md` § "declare the fix-engaged signal" should require that an
   env-valued signal be read from the process that performs the work, not from a wrapper or
   launcher in its ancestry. A signal read from an exec-frozen ancestor is not evidence.
3. **Consider a mechanical probe.** A small helper that, given a pod and a variable name, reports
   the value on the deepest descendant of the dispatcher would make the correct check cheaper
   than the incorrect one. Optional; the rule text is what actually matters.
4. **Cross-reference the `HF_XET_DISABLE` no-op** from the new entry, as the sibling
   looks-applied-but-inert failure.

## Explicitly NOT to be done

- Do NOT remove or weaken the dispatchers' `.env` re-source. Pods need `.env` loaded and the
  re-source is deliberate; the defect is in how mitigations are applied and verified around it.
- Do NOT change the project policy of setting HF accelerated-transfer variables shell-level in
  bootstrap. That policy is correct and is not what failed.
- Do NOT touch the live `#2546` arm-2 pod (`pod-2546-arm2`), which is running `p5_fits` under the
  corrected env right now.

## Provenance

Diagnosed by the `experimenter` subagent during #2546 arm 2 `p5_fits` rung-2 relaunch, from a
direct read of `scripts/issue2546_dispatch.sh`, the pod `.env`, and the child-vs-wrapper
`/proc/<pid>/environ` values. It emitted an `epm:failure-lesson v1` block with
`generalizes: yes`, `gotcha_candidate: yes`, `root_cause_confirmed: yes`. Recorded on #2546 as
`epm:failure-lesson v1`; the orchestrator's own false-positive exposure is recorded in the same
marker.
