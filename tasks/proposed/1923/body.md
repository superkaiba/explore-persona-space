---
title: 'daily-held: re-add VM SSH key to RunPod team key list'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-31T06:58:13Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 3): provision preflight WARN:
  VM key absent from the RunPod TEAM account key list (mutated by fellows onboarding);
  pre-injection pod resumes may refuse SSH.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 as a route-3 needs-human item (judgment-call carve-out: external side-effect — mutating the shared RunPod TEAM account settings, which fellows-cluster onboarding also mutates).

## Held decision

Provision output for pod-1482-saemlp (2026-07-30T17:58:53Z) printed: `account-key preflight [WARN]: VM key absent from the RunPod team account key list — the SHARED team list is mutated by fellows-cluster onboarding (live hazard class). ... pre-injection pod resumes / non-EPS tooling may refuse SSH. Re-add via RunPod console -> Settings -> SSH Public Keys.` Fresh pods stay reachable via boot injection; RESUMES of pre-injection pods may refuse SSH until the key is restored. No one re-added it during the day.

## What needs Thomas

Re-add the VM's public SSH key to the RunPod TEAM account key list via the RunPod console → Settings → SSH Public Keys (the key is `~/.ssh/id_ed25519.pub` on this VM). Held because the team key list is shared account state that fellows-cluster onboarding also edits — a wrong mutation affects every team member's pods.

## Why held (carve-out item)

External side-effects: a mutation of shared account settings outside this machine.

## Provenance

- origin: /daily 2026-07-30 problem sweep (miner-7 P23, session 0ac15c23, pod-1482-saemlp provision WARN)
