---
title: 'keep-running wedged-owner arm cannot fire on a busy task: add a pod-grain
  idleness leg (#1739: 3 pods idle 19.6h behind 129 markers)'
kind: infra
tags: []
created_at: '2026-08-06T11:00:56Z'
has_clean_result: false
parent_id: 1739
origin_prompt: check each pod is safe to end and then end it. then figure out why
  it wasn't ended before and make a workflow fix. then figure out how to merge the
  branch
workflow: v1
---
# Goal

The #1582 keep-running wedged-owner escalation arm cannot fire for an idle pod on a BUSY
multi-round task. Add a POD-GRAIN idleness leg so a verified-done, zero-utilization pod escalates
regardless of task-level marker traffic.

## Incident (2026-08-06, #1739)

Three `keep-running`-shielded RunPod pods (`pod-1739-a1apilot` / `-a1bpilot` / `-a2pilot`, 1xH100
each) ran IDLE for **~19.6 hours** — roughly **$165** at ~$8/hr combined — and no watcher arm ever
escalated. They were terminated manually only after the user asked, in chat, what a sibling session
was doing.

Every piece of evidence needed to escalate was present and cheap to read:

- each pod had written `/workspace/logs/issue-1739-<slug>-done.json` with `"phase": "done",
  "status": "ok"` at 15:03-15:52Z on 2026-08-05;
- uploads were independently Hub-verified (1000/1000 rollout JSONs + manifest + sentinel per rung,
  re-verified during the manual teardown);
- GPU utilization was **0%** on all three (74.6 GB VRAM resident, model loaded, no work);
- the workload log had not been written to in ~19 h;
- the owning session had been idle ~12 h with no worktree writes.

## Root cause — a grain mismatch, evidenced

`_KEEP_RUNNING_WEDGED_NOTE_SENTINEL` / the #1582 arm
(`scripts/autonomous_session_watch.py`, docstring lines ~81-93) requires BOTH legs:

1. the TASK shows no real marker progress for >= `KEEP_RUNNING_OWNER_IDLE_S` (12 h), AND
2. the owning session is provably wedged or absent.

Leg 2 held. **Leg 1 could never hold.** #1739 is a multi-round task with several concurrent
sessions (the per-pod named shield #1961 exists precisely because this issue runs multiple pods at
once). Measured over the exact 19.6 h window the pods sat idle:

    markers on #1739 since 2026-08-05T15:05Z: 129
    LARGEST marker gap in that window:        6.19 h
    #1582 requires >= 12.00 h                 -> never satisfied

Sibling rounds (`syco-OOD`, `r2fair`, the analyzer fold) posted markers continuously on the SAME
task, resetting a task-grain counter that was standing in for a pod-grain question. The arm is
structurally incapable of firing on any busy issue — the busier the task, the more permanently the
idle pod is hidden. This is the same failure family as #1345 (~72 h of billing behind a frozen
wrapper), re-opened through a different door: #1582 closed the wedged-OWNER case but left the
busy-TASK case open.

No arm reads the pod's OWN idleness. `keep-running` therefore still makes a pod effectively
invisible whenever its task is active — which is exactly when long-lived shielded pods exist.

## Proposed fix

Add a third, independent escalation leg to the #1582 arm keyed on POD-grain evidence, so it does
not depend on task-marker silence:

- **Terminal sentinel leg:** a `keep-running`-shielded RUNNING pod whose `/workspace/logs/` holds a
  done-sentinel (`"phase": "done"` / terminal `status`) older than a floor (suggest reusing
  `KEEP_RUNNING_OWNER_IDLE_S`, or a shorter pod-specific floor — a verified-done pod has no reason
  to live) escalates on its own evidence.
- **Utilization leg (fallback where no sentinel convention exists):** sustained ~0% GPU utilization
  sampled over >= N consecutive ticks (never a point read — cf. the r2fair GPU-0% false alarm,
  where 0 MiB readings during a CPU preamble were NOT a fallback) plus no workload-log write for
  the same span.

Contract to preserve:

- **ESCALATE, NEVER STOP.** #1582's never-a-stop guarantee is the whole reason the arm is safe under
  a user override; do not turn this into an auto-terminate. The standing directive is that pods die
  only with the user's approval (or by the owning agent's verified-done teardown).
- Confirm over >= 2 consecutive ticks; an "unknown" read FREEZES the counter (fail toward no-fire),
  matching the existing arm.
- One marker per episode + push + sidecar row, re-alerted on
  `EPM_KEEP_RUNNING_WEDGED_REALERT_H`, reusing the existing plumbing.
- Kill switch, consistent with every other pass.
- Per-pod, not per-task: on a multi-pod issue one busy pod must not shield an idle sibling (#1961
  already established per-pod grain for the SHIELD; this is the same grain for the ALARM).

## Secondary finding (separate, smaller)

`scripts/issue1739_eos_pilot_pod.py` (on `issue-1739`, not main) is why the pods were idle rather
than gone. It ends with an explicit `sys.exit(0)` — present in `dbaebddf1f`, the commit the pods
actually ran — but `sys.exit()` only raises `SystemExit`: the interpreter still runs finalization,
which blocks joining vLLM's worker subprocesses (a surviving
`multiprocessing.resource_tracker` child was visible on all three pods 19 h later). The launcher
emitted its terminal `[phase=done]` line, the sentinel was written, uploads completed — and the
process still never exited, so the pod could never self-terminate on the verified-done contract.

This is the documented vLLM worker-subprocess teardown gotcha
(`.claude/rules/gotchas.md`); the correct terminal for a vLLM driver is `os._exit(0)` after an
explicit flush (or an explicit engine shutdown first). Worth considering whether that rule states
the `sys.exit` vs `os._exit` distinction sharply enough for a generation driver, since a careful
author here applied `sys.exit(0)` *deliberately*, with a comment citing gotchas.md, and still got
the hang. Fixing the #1739 script itself is NOT in scope for this task (it lives on an unmerged
branch); the rule wording is.

## Acceptance criteria

1. A `keep-running`-shielded RUNNING pod with a terminal done-sentinel and no live work escalates
   within the configured floor **even when its task is receiving markers continuously** — with a
   regression test that reproduces the #1739 shape (busy task, largest marker gap well under 12 h,
   idle pod) and asserts the arm fires.
2. The existing #1582 behaviour is unchanged for the quiet-task case (no regression in its tests).
3. Escalation remains never-a-stop; unknown reads still freeze the counter.
4. Per-pod grain: an idle pod escalates while a sibling pod on the same issue is legitimately busy.
5. Kill switch documented in `.claude/rules/background-automation.md` alongside the other passes.
