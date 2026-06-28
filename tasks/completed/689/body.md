---
title: 'RunPod RUNNING-but-no-port host wedge: detect + auto-migrate; per-cell incremental
  upload for fleet sweeps'
kind: infra
tags: []
created_at: '2026-06-27T20:36:25Z'
has_clean_result: false
origin_prompt: get a subagent to do a deeper dive into what happened (and hopefully
  figure out a fix so this doesn't happen again)
---
## Goal

Prevent the #664-class failure: a RunPod pod that goes RUNNING-but-no-public-port on a degraded host bills indefinitely, is unrecoverable via (host-pinned) stop+resume, and strands hours of work because fleet eval artifacts upload only in a terminal batch. Make this class detectable + auto-recoverable, and make in-flight work durable.

## Root cause (deep-dive on #664, 2026-06-27)

1. **RunPod RUNNING-but-no-port wedge.** RunPod `desiredStatus` is decoupled from `runtime.ports`. When the host degrades, the API keeps reporting `RUNNING` (and billing) while `runtime.ports` is empty → `runpod_api.py:_parse_pod` (~L424-431) yields `ssh_host=None, ssh_port=None`. `resume_pod` (`runpod_api.py:684`) sends `podResume {podId, gpuCount}` only — **host-pinned, no host reselection** — so stop+resume returns to the SAME degraded host and re-wedges. `--refresh-from-api` is a **no-op** here (the port is platform-absent, not stale — that flag fixes the #488 stale-port case). `pod_lifecycle.py:cmd_resume` (~L1878-1894) re-raises on `wait_for_ssh` timeout while leaving the pod **billing**, with no terminate/migrate path. (#664 billed ~1.5h+ unreachable; the #488 lesson was 13h+.)
2. **Work stranded by write-at-end upload.** `issue664_dispatch.py` P2 (`p2_extract_eval`) writes eval JSONs / store tensors / raw completions **only to the pod volume**; upload happens once in the terminal P3 batch. The pod died mid-P2 → **0 i664 files on HF**, ~16 cells (~689 JSONs, ~3-4h compute) lost. Adapters survived only because `train_lora` uploads + Hub-verifies each adapter **inline per cell**. This violates `code-style.md` checkpoint-per-phase.

## Fixes (prioritized)

**(a) Per-cell incremental upload — highest value (experiment code).** `scripts/issue664_dispatch.py`: in `extract_and_eval_cell` (~L963), after the gen worker succeeds, upload THAT cell's eval JSONs + store tensors + raw completions to the HF data repo (the `_cell_extract_eval_done` sentinel ~L1017 is the natural hook). P3 `upload_artifacts` becomes an idempotent safety sweep (re-upload only cells not already on Hub). Batch one `upload_folder` commit per cell (HF 256-commits/hr cap). Strands at most one in-flight cell. (a) is the prerequisite that makes (b)'s auto-terminate safe.

**(b) RunPod RUNNING-but-no-port wedge: detect + auto-migrate (the durable infra fix).** The architecture already exists for GCP (#669): `backend_poll.py` (~L84-118) escalates a frozen non-terminal phase + reachability alarm past `GCP_STALENESS_FLOOR_SEC` to a synthesized terminal "wedged" phase in the async-failover accept-set → fails over. The RunPod backend has NO analogue. Add: after K min (≥ wait_for_ssh window + 1 retry margin) of `desiredStatus=RUNNING` + null `runtime.ports` across ≤1 resume, STOP retrying the host-pinned pod and — ONCE every recoverable input is HF-verified (adapters always; eval JSONs require fix (a)) — `terminate_pod` (stops the billing leak) + re-provision fresh + resume the dispatcher (idempotent). Files: `pod_lifecycle.py` (cmd_resume + a detection helper) and/or `backend_poll.py` + `src/explore_persona_space/backends/runpod.py` (the RunPod sibling of GCP #669). Terminate is user-gated per CLAUDE.md, so the autonomous auto-terminate is safe ONLY with fix (a) (inputs-on-HF) in place.

**Docs (workflow surface).** `.claude/rules/compute-backend-failover.md`: add a "RunPod RUNNING-but-no-port host wedge (#664)" section (detection + host-pinned resume + --refresh-from-api no-op + terminate-and-reprovision recovery; the RunPod sibling of GCP #669). `.claude/rules/upload-policy.md`: add "multi-cell pod sweeps upload per-cell, never one terminal batch" citing code-style checkpoint-per-phase.

## Follow-up concerns (from the deep-dive)

- `scripts/pod_audit.py`: add a report-only `running-no-port` bucket (a managed RUNNING pod with null `runtime.ports` for >N h is an unreachable billing leak currently mis-bucketed as `active`); gate any auto-stop on inputs-on-HF + `keep-running`.
- `scripts/pod_lifecycle.py:cmd_resume`: split the `wait_for_ssh`-timeout advice — new-port-present → `--refresh-from-api` (#488 case); still-null → host wedge → name terminate+re-provision, not the wrong refresh advice. Flag that it re-raises while the pod keeps billing.
- `scripts/autonomous_session_watch.py` pod-safety pass: recognize a RUNNING-but-unreachable pod on an ACTIVE task (compose with fix (b) detection + the `no_compute_available` capacity-retry pass).

## Related
#664 (the incident), #667 (the GCP hung-but-RUNNING gap), #669 (the GCP wedge→failover mechanism this mirrors).
