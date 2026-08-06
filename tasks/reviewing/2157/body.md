---
title: 'workflow-fix: pods.md names a --boot-disk-gb flag pod.py provision lacks;
  cpu-bigmem defaults to 50GB not 240GB'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7a6dda19cd64
created_at: '2026-08-06T19:08:54Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation during #1336 inline round 5: bare --intent
  cpu-bigmem provision gave 30GB free vs ~115GB needed; pods.md ADOPTION sentence
  names --boot-disk-gb which pod.py provision does not accept.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gap hit during #1336 inline round 5 (emitting agent: orchestrator, own observation). A CPU-only fits job needed ~115 GB of staging headroom; the pod provisioned through the documented path came up with 30 GB free and would have died ENOSPC mid-stage had the disk not been probed before launch.

## Goal

Correct the `.claude/rules/pods.md` CPU-intent guidance so a launch composer sizing a large-footprint CPU stage reaches for a flag that exists on the path it is actually using, and is not misled by the intent table's cap figure into believing the capacity is the default.

## Workflow gap

- **Bug observed:** A `cpu-bigmem` pod provisioned via the documented `pod.py provision --issue <N> --intent cpu-bigmem` path came up with `/workspace` on the 50 GB overlay ROOT (30 GB free, `df -h /workspace` -> `overlay 50G 21G 30G 41% /`), not the 240 GB the CPU intent table's "container-disk cap 240 GB" reads as. Separately, `pods.md` line 30 instructs launch composers to pass `--boot-disk-gb`, which `scripts/pod.py provision` does not accept.
- **Why it is a workflow gap:** two defects in one sentence-pair of `.claude/rules/pods.md`, both on the workflow surface, both misleading a composer that follows the rule literally. (1) FLAG-NAME/PATH MISMATCH — the ADOPTION sentence names `--boot-disk-gb`, which exists only on the ROUTER path (`scripts/dispatch_issue.py`, `src/explore_persona_space/backends/gcp.py`); the direct `pod.py provision` path documented in the SAME rule file's Lifecycle command block takes `--container-disk-gb`. A composer following the rule on the pod.py path finds no such flag. (2) CAP-READS-AS-CAPACITY — the intent table states "container-disk cap 240 GB" with no statement of the DEFAULT, so the bare intent silently yields 50 GB. The failure is silent at provision time and only surfaces as an ENOSPC/EDQUOT death partway through staging, after the pod has been billing.
- **Confidence (emitter):** high — both halves are mechanically verified below, and the incident is first-hand.
- verified-at-filing: `uv run python scripts/pod.py provision --help | grep -cE '\-\-boot-disk-gb'` -> **0 hits**; same command grepping `--container-disk-gb` -> **2 hits** (2026-08-06). `grep -n 'boot-disk-gb' .claude/rules/pods.md` -> **1 hit, line 30** (the ADOPTION sentence). `grep -rln 'boot-disk-gb' scripts/ src/ .claude/` -> the flag is real but lives on `scripts/dispatch_issue.py`, `scripts/verify_plan.py`, `src/explore_persona_space/backends/gcp.py`, `.claude/workflow.yaml` (router/GCP path), NOT on `scripts/pod.py`. Live pod evidence: `ssh pod-1336-selfmapa 'df -h /workspace'` -> `overlay 50G 21G 30G 41% /` on a bare `--intent cpu-bigmem` provision; a re-provision adding `--container-disk-gb 200` was required.

Unverified hypothesis — verify at plan time: that the RAM figure in the same table row ("128 GB" for `cpu5m-16-128`) is also understated. The live pod reported `free -g` total **251 GB**. This may be a RunPod flavor change, an oversubscribed host figure, or a stale table entry; it was observed once on one pod and not chased. It is NOT part of the core claim above and should be confirmed independently before any table edit lands on that number.

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/rules/pods.md`, CPU intent section (~line 30):

```
- ADOPTION: plan §9 / launch composers pass `--boot-disk-gb` whenever a CPU stage sizes disk > 50 GB
+ ADOPTION: the flag differs BY PATH. Router / plan §9 dispatch (`scripts/dispatch_issue.py`):
+ `--boot-disk-gb`. Direct pod provisioning (`scripts/pod.py provision`): `--container-disk-gb`
+ (there is no `--boot-disk-gb` on pod.py). Pass one whenever a CPU stage sizes disk > 50 GB.
```

and in the CPU intent table, state the DEFAULT next to the cap, e.g. `cpu-bigmem | cpu5m-16-128 (16 vCPU / 128 GB RAM; container disk DEFAULTS to 50 GB — pass --container-disk-gb up to the 240 GB cap)`.

## Scope / surfaces

- Primary target: `.claude/rules/pods.md`
- Check whether the same `--boot-disk-gb`-only phrasing appears in `.claude/rules/plan-compute-sizing.md` and `.claude/rules/compute-backend-failover.md` (both name the flag); if they are router-scoped the phrasing is correct there and needs no edit — confirm per file rather than blanket-editing.
- Consider whether `pod.py provision` should additionally accept `--boot-disk-gb` as an alias, so a composer following the router phrasing on the pod path fails soft rather than silently taking the default. That is a CODE change, not a doc change — the planner should decide whether it is in scope.

## Constraints / invariants

- Workflow-surface only — no experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` (no flags) passes; ruff clean on any touched `.py`.
- Do NOT change the 240 GB cap value or the `cpu_fallback_infeasible_for_plan` typed-refusal behavior — this is a documentation-accuracy fix plus an optional alias, not a capacity or routing change.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/pods.md
- fingerprint: 7a6dda19cd64

Filed from #1336 inline round 5. The provisioning incident: a bare `--intent cpu-bigmem` provision yielded 30 GB free against a ~115 GB peak staging requirement; caught by an explicit `df -h /workspace` probe before the workload launched, so no compute was lost — but the probe was ad hoc, not prompted by the rule.
