---
title: 'pod.py terminate: no surgical form for the PRIMARY pod while a suffixed sibling
  is live'
kind: infra
tags: []
created_at: '2026-08-13T07:50:13Z'
has_clean_result: false
parent_id: 2223
origin_prompt: 'Surfaced during /issue 2223 Step 8: pod-2223 (4x H100) verified-done
  and idle, pod-2223-q32b live; --name-suffix cannot address the primary and the bare
  --issue form destroys both.'
workflow: v1
---
# `pod.py terminate` cannot target the PRIMARY pod while a suffixed sibling is live

## Goal

Give `pod.py terminate` a sanctioned surgical form that destroys ONLY the
primary `pod-<N>` while `pod-<N>-<slug>` siblings keep running — closing the
multi-pod-per-issue teardown gap that #1334 introduced, without weakening the
upload-verification guard or the compute-kill gate.

## The gap

Two live pods on one issue, only one of them finished:

    pod-2223         4x H100   7B leg    work DONE + upload-verified   IDLE
    pod-2223-q32b    4x H200   32B leg   actively generating

Neither available form of the command expresses "terminate the finished one":

- `terminate --issue 2223 --name-suffix q32b` resolves `pod-<N>-<slug>` only.
  There is no suffix that names the PRIMARY pod — its name has no suffix — so
  the surgical form structurally cannot address it.
- `terminate --issue 2223 --yes` resolves EVERY live pod for the issue and
  would destroy the actively-generating 32B leg along with the idle 7B one.
  It also refuses while the issue-wide `keep-running` tag is set (#1485), and
  that tag is exactly what shields the live sibling.

So the finished 4x H100 pod bills while its verified-done teardown — the one
destruction the compute-kill gate does sanction without a user ask — has no
command that performs it. The operator's only options are to leave it idling
or to reach for a form that destroys healthy compute.

## Why this is not a niche case

The suffixed-pod affordance exists precisely so an issue can run legs on
different hardware (7B on H100s, 32B on H200s here). Whenever those legs
finish at different times — the normal case, since they are sized differently —
the first one to finish hits this gap. The idle-burn family (#664, #1662) is
the standing cost: #1662 was a suffixed pod idling ~$12-13/hr behind an
ask-gate; this is the same leak reached through the primary pod instead.

## Sketch (not a mandate — the fix is the spawned session's to design)

A primary-only selector, e.g. `--primary-only` / `--name-suffix ""` given
explicit meaning, that resolves exactly `pod-<N>` and no suffixed sibling.

Two constraints any fix must hold:

1. **The upload-verification guard stays binding.** The new form goes through
   `_guard_upload_verification_before_terminate` exactly as the existing forms
   do, including the `outroot=` sweep-attestation token (#2187). Verified-done
   teardown is the ONLY automated destruction the gate grants; this must not
   become a second door around it.
2. **The `keep-running` interaction needs deciding explicitly.** The tag is
   ISSUE-WIDE by design (#1485), so today it blocks primary teardown even when
   it was set solely to shield a suffixed sibling. A primary-only form that
   ignores the tag outright would defeat the tag's purpose; one that honours it
   leaves this exact gap open. Naming the intended semantics IS most of the
   task — consider whether the shield should become per-pod for this decision,
   or whether the tag should be re-scoped when the round it shields is named.

## Provenance

Surfaced during #2223 Step 8 upload verification (2026-08-13). The 7B leg
passed verification with `outroot=residue-committed`; teardown was NOT
attempted, the pod was left alive, and the burn was surfaced for the user
rather than routed around the gate.
