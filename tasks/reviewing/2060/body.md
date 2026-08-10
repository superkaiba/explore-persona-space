---
title: 'workflow-fix: bootstrap-failed pod left billing (provision no teardown + keep-running
  blocks reaper)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a20f18266959
created_at: '2026-08-04T03:48:32Z'
has_clean_result: false
origin_prompt: 'User: ''can you fix workflow so this doesn''t happen again'' — after
  pod-1947-loc sat rented and billing 14+ min with runtime:null, provision process
  exited without tearing it down, and the keep-running tag blocked the watcher''s
  no-port wedge reaper.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a live money leak on task #1947 (emitting agent: orchestrator). A never-bootstrapped RunPod pod was left rented and billing with no code path able to reap it.

## Goal

`pod.py provision` must tear down the pod it created when the SSH bootstrap wait fails, and the watcher's no-port wedge arm must reap a never-ran pod even when `keep-running` is set.

## Workflow gap

- **Bug observed:** a pod whose bootstrap wait times out is left rented and billing — provision exits with only a stderr hint, and the watcher's no-port wedge arm requires `keep_running is False`, so the tag that shields the provisioning window also blocks the reaper.
- **Why it is a workflow gap:** these are two independently-correct-looking behaviours that compose into an unreapable billing pod. Neither component is obviously wrong on its own; the hole only exists at their intersection, and it is invisible until a human happens to query the RunPod API. Both `scripts/pod_lifecycle.py` and `scripts/autonomous_session_watch.py` are workflow surface.
- **Confidence (emitter):** high — hit live, both halves confirmed in source.
- verified-at-filing (2026-08-04 UTC), per-target:
  - `scripts/pod_lifecycle.py` — `grep -n -iE 'def cmd_provision|bootstrap.*fail|terminate.*bootstrap'` → the shared wait/register/bootstrap helper's own docstring (~L1895) states it "either returns clean or ``sys.exit``s on a bootstrap failure (preserving the prior behavior verbatim)"; the failure path (~L1965-1972) prints a human hint offering either a manual `bootstrap_pod.sh` re-run **or** `pod.py terminate ... to discard`, then the machine-greppable `BOOTSTRAP-FAILED pod=<name> rc=<rc>` line, then `sys.exit(rc)`. **No teardown of the pod it just created.** The `wait_for_ssh(info.pod_id, timeout=600)` call site even carries the comment "The pod exists and is billing, but never exposed 22/tcp within the..." — the leak is known and documented at the exact line, and the chosen behaviour is to exit and leave it billing.
  - `scripts/autonomous_session_watch.py` — `grep -n 'keep_running'` → the #770 NOTE (~L1327) pins the wedge arm's confirmed predicate as "`keep_running is False AND inputs_ok`", and ~L1338 repeats the "provably-safe case (`keep_running is False AND inputs_ok=True`)". So a `keep-running`-tagged pod is exempt from TERMINATE+FAILOVER by design. The only tagged-pod arm is the #1582 ESCALATE (~L80-89), which is never a stop AND additionally requires a DONE-status task plus `EPM_KEEP_RUNNING_WEDGED_OWNER_MIN_H` (12h) of provable owner wedge.

**Live incident (task #1947, 2026-08-04).** `pod-1947-loc` rented 03:32:10 UTC, 4x H100. At 03:46 the RunPod API still reported `runtime: null` — no uptime, no port-22 mapping — while the sibling `pod-1947-r3` (rented 03:29:51) had its endpoint inside ~4 min and was running a workload at 973s uptime. The provision process (pid 4170005) had already exited. Net state: a rented, billing, empty pod, with `keep-running` set on #1947 (correctly, to shield a concurrent live round) and therefore no reaper able to touch it. Caught only because a human asked "is this bugging?". Roughly $3 at the time of detection; unbounded had nobody looked.

## Proposed change (candidate diff sketch — refine in planning)

**(a) Provision cleans up after itself.** On bootstrap-wait failure, terminate the pod the same invocation created, then exit non-zero as today:

```
     except RunPodError:
-        # The pod exists and is billing, but never exposed 22/tcp within the
-        # timeout. Print guidance and exit.
+        # The pod exists and is billing but never exposed 22/tcp. Do NOT leave
+        # it rented: tear down what this invocation created, then fail loud.
+        if not args.keep_on_bootstrap_failure:
+            terminate_pod(info.pod_id)           # best-effort, logged
+            print(f"BOOTSTRAP-FAILED-TERMINATED pod={name}", file=sys.stderr)
         print(f"BOOTSTRAP-FAILED pod={name} rc={rc}", file=sys.stderr)
         sys.exit(rc)
```

Add `--keep-on-bootstrap-failure` for the deliberate debug case (inspecting a half-built pod). Default MUST be teardown — the current default optimises for a rare debugging need at the cost of an unbounded common-case leak. Preserve the existing `BOOTSTRAP-FAILED` token exactly (it is a documented grep contract, #1931) and add the terminated-variant token alongside rather than replacing it.

**(b) Watcher backstop: never-ran pods are reapable despite the tag.** Narrow carve-out to the no-port wedge arm — if a RUNNING pod has **never** exposed a runtime/port since creation (not "lost" one: never had one) and is past the bootstrap grace floor, it is reapable even when `keep_running` is True. The justification is that `keep-running` exists to protect work in progress, and a pod that never obtained a runtime provably has none: there is no work to protect. Keep every other guard (owner-liveness tri-state, once-per-episode lease, K floor). Suggest a distinct env knob and a distinct sidecar reason so the carve-out is auditable separately from the untagged arm.

Fix (b) is the load-bearing one: it catches the leak regardless of which path created the pod, including an orchestrator that crashed before any cleanup could run. Fix (a) is the proximate cause and is cheaper. Ideally both; if the planner must pick one, take (b).

## Scope / surfaces

- Primary target: `scripts/pod_lifecycle.py`
- Backstop target: `scripts/autonomous_session_watch.py` (no-port wedge arm predicate)
- Regression tests: a provision whose `wait_for_ssh` raises must leave no live pod (and must leave one under `--keep-on-bootstrap-failure`); a never-ran RUNNING pod with `keep_running=True` past the grace floor must be selected by the wedge arm, while a pod that HAD a runtime and lost it, or is inside the grace window, must not be.

## Constraints / invariants

- Workflow-surface only.
- Do not weaken `keep-running` for pods that ever had a runtime — that tag protects live follow-up rounds and the #477/#573 incidents it exists for.
- Preserve the `BOOTSTRAP-FAILED` / `BOOTSTRAP-OK` grep contract (#1931).
- Teardown on failure must be best-effort and must never mask the original bootstrap error.
- Surgical teardown only — must not touch sibling pods on the same issue (`pod-<N>-<slug>` multi-pod form).
- `scripts/workflow_lint.py` passes; ruff on touched files passes.
- This session runs under the recursion guard and does NOT auto-file further candidates.

## Provenance

- workflow_fix_target: scripts/pod_lifecycle.py
- fingerprint: a20f18266959

Surfaced from #1947 during two concurrent user-directed paid RunPod rounds. Immediate mitigation was manual: the owning agent was told to terminate surgically and re-provision. That is per-incident and depends on a human noticing.
