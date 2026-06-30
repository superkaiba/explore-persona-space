---
title: 'workflow-fix: SKILL.md gate handler for pv_phase1_done (issue #763 off-pod
  PV judge)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cbc228d6433f
created_at: '2026-06-30T14:01:01Z'
has_clean_result: false
origin_prompt: 'Auto-filed from #763 round 3 implementer workflow-fix-candidate v1
  — round-3 dispatcher emits gate=pv_phase1_done to enable off-pod PV judge; SKILL.md
  Step 6d.4 has no handler for that gate name, so production run would block at sentinel
  and surface epm:failure unrecognised_gate_name'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #763 round 3 (emitting agent: experiment-implementer).

## Goal

Add a `pv_phase1_done` gate handler to `.claude/skills/issue/SKILL.md` Step 6d.4 that runs `pod.py stop` → off-pod PV judge on the VM → `pod.py resume` → re-dispatch the workload at `--from-phase pv_capture`. Without this, the production run of `/issue 763` will block at the pod-side gate sentinel `gate=pv_phase1_done` and fall through to the "Unrecognised `gate` name" branch (epm:failure / status:blocked).

## Workflow gap

- **Bug observed:** The round-3 fix for task #763's `pv-judge-not-off-pod` BLOCKER restructured `scripts/issue763_dispatch.sh` into a two-phase shape with a blocking gate sentinel `gate=pv_phase1_done` between generate (GPU) and capture (GPU) phases. The dispatcher cannot self-orchestrate the pod lifecycle (`pod.py stop` / `resume` are orchestrator-only per CLAUDE.md "Pod-side code NEVER shells out"). The orchestrator-side gate handler — which must call `pod.py stop`, run `--phase judge` on the VM, call `pod.py resume`, and re-dispatch — does not exist yet.
- **Why it is a workflow gap:** `.claude/skills/issue/SKILL.md` Step 6d.4 registers gate handlers per-name, with only `fact-candidates` currently wired plus an "Unrecognised gate" catch-all that posts `epm:failure v1 reason: unrecognised_gate_name` and sets `status:blocked`. The new `pv_phase1_done` gate name is in scope per the workflow-fix-on-bug protocol (workflow surface = `.claude/skills/issue/SKILL.md`), and the handler logic is the canonical pod-cycling pattern (analogous to how `fact-candidates` writes a marker + lets the user resume, but here the resume is automated).
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```markdown
# SKILL.md Step 6d.4 "Gate handlers (one per registered <name>):"
+ - **`pv_phase1_done`** (issue #763 PV-extraction off-pod judge): the GPU
+   phase-1 dispatcher has EXITed after generate + pre-teardown upload (rollouts
+   on HF). The orchestrator:
+   1. `pod.py stop --issue <N>` (free the GPU through the deadline-bounded poll)
+   2. on the VM: `uv run python scripts/issue763_extract_pv_rb.py --phase judge`
+      (fetches rollouts from HF via snapshot_download, batch-judges, uploads the
+      keep-flags to the issue HF prefix)
+   3. `pod.py resume --issue <N>`
+   4. re-dispatch the workload `--from-phase pv_capture` (the dispatcher resumes
+      at capture → E0 judge → fit → figures → final upload → [phase=done])
+   then resume the polling loop.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 6d.4 gate handlers section).
- Possibly also: `scripts/poll_pipeline.py` if its sentinel-drain logic needs to know the new gate name. The implementer flagged that `write_sentinel` was already extended with `gate`/`blocks_pipeline` keys that the existing generic gate path reads — so `poll_pipeline` may already handle the gate routing; verify.
- Grep before editing: `grep -rln 'fact-candidates\|gate=' .claude/skills/issue/ scripts/poll_pipeline.py`.

## Constraints / invariants

- The handler must use `pod.py` orchestrator commands only (NOT shell out from the dispatcher).
- The off-pod `--phase judge` runs on the VM (not a pod); it reads from HF via `snapshot_download` and uploads keep-flags.
- The re-dispatch at `--from-phase pv_capture` must thread through the existing experimenter / pod-launch pipeline (same backend / lane); ideally just re-invokes `scripts/issue763_dispatch.sh` on-pod with the new flag.
- Workflow-surface only — no experiment code changes (the dispatcher + extract script are already in place on the `issue-763` branch).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (auto-set per `task.py is_workflow_fix_session`); recursion guard active.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: cbc228d6433f
- emitting_task: #763 (round 3)
- emitting_agent: experiment-implementer
- production-impact: the round-3 dispatcher change is shipped on `issue-763`; without this gate handler the production run will EXIT at the gate sentinel and report `epm:failure unrecognised_gate_name`.

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: The issue #763 dispatcher emits a blocking gate sentinel gate=pv_phase1_done (so the PV judge runs OFF the GPU pod), but SKILL.md Step 6d.4 registers gate handlers per-name and has only `fact-candidates` + an "unrecognised gate → epm:failure/blocked" catch-all, so pv_phase1_done would block instead of triggering the pod-cycle.
why_workflow_gap: Step 6d.4's gate-handler registry is the workflow surface that turns a pod-side gate into an orchestrator action (pod stop / off-pod judge on the VM / pod resume / re-dispatch); a dispatcher cannot self-orchestrate the pod lifecycle (pod.py commands are orchestrator-only), so the handler must live in SKILL.md.
proposed_change: Add a `pv_phase1_done` gate handler to SKILL.md Step 6d.4 that runs `pod.py stop --issue <N>`, then on the VM `uv run python scripts/issue763_extract_pv_rb.py --phase judge` (off-pod, fetches rollouts from HF, uploads keep-flags), then `pod.py resume --issue <N>` and re-dispatch the workload with `scripts/issue763_dispatch.sh --from-phase pv_capture`.
diff_sketch: |
  # SKILL.md Step 6d.4 "Gate handlers (one per registered <name>):"
  + - **`pv_phase1_done`** (issue #763 PV-extraction off-pod judge): the GPU
  +   phase-1 dispatcher has EXITed after generate + pre-teardown upload (rollouts
  +   on HF). The orchestrator:
  +   1. `pod.py stop --issue <N>` (free the GPU through the deadline-bounded poll)
  +   2. on the VM: `uv run python scripts/issue763_extract_pv_rb.py --phase judge`
  +      (fetches rollouts from HF via snapshot_download, batch-judges, uploads the
  +      keep-flags to the issue HF prefix)
  +   3. `pod.py resume --issue <N>`
  +   4. re-dispatch the workload `--from-phase pv_capture` (the dispatcher resumes
  +      at capture → E0 judge → fit → figures → final upload → [phase=done])
  +   then resume the polling loop.
confidence: medium
related_task: #763
<!-- /workflow-fix-candidate -->
