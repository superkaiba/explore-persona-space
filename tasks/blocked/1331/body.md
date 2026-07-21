---
title: 'daily-held: pod-779 stranded n10k captures - rescue or drop'
kind: infra
tags:
- daily-held
created_at: '2026-07-15T06:52:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 3): the 07-06 n10k round''s
  per-token captures + raw rollout text live only on stopped pod-779''s volume; resume
  is host-pinned and the former host has no free GPUs, so the artifacts are stranded
  (upload-before-stop violation discovered 2026-07-14)'
workflow: v1
---
## Overview / Motivation

Held by the /daily 2026-07-14 problem sweep (session 272c80a1, #779, 21:36Z). Carve-out: destructive/irreversible data decision + potential spend (a resume-watch keeps a stopped volume alive; termination destroys the only copy of the n10k raw rollout text).

## The decision

pod-779 (stopped, host-pinned) holds the only copy of the 07-06 n10k round's per-token captures + raw rollout text (`pass_b/train_context_vectors.pt` DID reach HF; the rest did not). Resume fails with a supply constraint and cannot relocate. Options:

1. **Rescue-watch:** periodic background `pod.py resume --issue 779` attempts; on success, upload raw text + captures, then terminate. Cost: watch machinery + eventual resume/upload time; volume storage is billing-exempt while stopped per RunPod's stopped-pod model, but the pod lingers in audits.
2. **Accept regeneration:** the n50k round (2026-07-14, sha-verified on HF) supersedes n10k for the scaling curve; the n10k raw text is then permanently lost — a persist-by-default violation stands unremediated.
3. **Terminate now:** clears the audit surface, accepts the loss explicitly.

The standing rule (raw completions MUST upload before pod stop) was violated by the 07-06 round; this task is the disposition decision, which is Thomas's call (data-loss + spend).

## Suggested action

If the n1M round proceeds (dispatch started 2026-07-15 ~06:30Z), option 2/3 is probably fine — n10k adds nothing to the curve. If scaling analysis needs the n10k raw text for a per-context join, option 1.

## Disposition status (2026-07-17 autonomous rescue attempt, per PM greenlight)

- **Rescue attempted, failed:** `pod.py resume --issue 779` at 21:51Z refused — former host still has no free GPUs (host-pinned; same refusal as 07-14). Attempt was free.
- **Context updates:** n50k round complete 07-15 04:12Z AND n1M round complete 07-15 22:44Z, both sha-verified on HF — n10k is fully superseded for the scaling curve. Pod-779 remains STOPPED (11d), shielded from the stale-pod audit by #779's `keep-running` tag.
- **What termination would permanently lose:** the exact n10k rollout TEXT (6,500 LMSYS contexts × 1 rollout) + 28-layer captures; the already-uploaded `pass_b/train_context_vectors.pt` becomes permanently unjoinable at the raw-text level (a re-run samples different text). Model generations are never-discardable per upload policy.
- **RECOMMENDATION (option 1-lite):** keep pod-779 stopped under the `keep-running` tag (≈$0 compute; storage small if any) and re-attempt resume opportunistically — the attempt is free + ~30s: `uv run python scripts/pod.py resume --issue 779` (weekly PM pass or /daily). On a future success, plan v1 Branch A (`plans/v1.md`) is the ready-to-run rescue recipe: upload raw text + captures to `issue779_monitoring/fitter-fair-comparison-n10k/`, Hub-verify, terminate, remove `keep-running`.
- **Terminate now (option 3) ONLY on explicit user OK** — parked here rather than executed: irreversible destruction of sole-copy model generations is a user-only call. If you accept the loss, run: `uv run python scripts/pod.py terminate --issue 779 --yes` then `uv run python scripts/task.py remove-tag 779 keep-running` and archive this task.
