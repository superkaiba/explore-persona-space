---
title: 'daily-held: pod-779 stranded n10k captures - rescue or drop'
kind: infra
tags:
- daily-held
- needs-human
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
