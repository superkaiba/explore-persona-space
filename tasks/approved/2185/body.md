---
title: Document the RunPod HF-CDN zero-byte large-blob trap (206 + 0 bytes) and its
  curl discriminator + rsync-relay recovery
kind: infra
tags:
- hf-cdn-zero-byte-blob
created_at: '2026-08-07T23:30:17Z'
has_clean_result: false
origin_prompt: 'Filed by the #2162 orchestrator after a stage-2 pod could not fetch
  any HF large blob: DNS steered us.aws.cdn.hf.co to APAC CloudFront edges returning
  206 with zero bytes across xet, hf_transfer, plain python and curl, while pod egress
  was 73 MB/s and the VM fetched the same URL fine. Reporting agent marked gotcha_candidate:
  yes, root_cause_confirmed: yes.'
workflow: v1
---
# RunPod pods can have HF large-blob GETs steered to CDN edges that serve 206 + 0 bytes; document the discriminator and the rsync-relay recovery

## Goal

Capture a confirmed, generalizing infra trap in `.claude/rules/gotchas.md` (and
consider a preflight probe for it), so the next agent that hits it spends
minutes rather than the better part of an hour.

A pod provisioned for #2162 stage-2 could not download any large blob from the
Hugging Face CDN. Pod DNS resolved `us.aws.cdn.hf.co` to APAC CloudFront edges
that answered with HTTP **206 and zero bytes**, then closed the connection.

## Why it is expensive to diagnose

The single root cause presents as three unrelated-looking failures in sequence,
each of which invites a wrong fix:

1. First as the **xet download wedge** — which has its own documented ladder,
   so the natural response is to toggle `HF_HUB_DISABLE_XET=1`.
2. Then as **hf_transfer's opaque `RuntimeError`** — which invites toggling
   `HF_HUB_ENABLE_HF_TRANSFER=0`.
3. Then as a plain-path **`IncompleteRead(0 bytes)`**.

No client-side accelerator toggle fixes it, because the fault is upstream of
every client. An agent working the accelerator ladder will exhaust it and
conclude the pod or the Hub is broken.

Confounding signals that make the pod look healthy: general pod egress was fine
(73 MB/s to Cloudflare), small-file HF GETs worked normally, and the VM pulled
the exact same blob URL without trouble. So neither "the pod has no network" nor
"the Hub is down" is true.

## The discriminator (the part worth writing down)

A three-way curl differential isolates it in about a minute:

- pod → the HF blob URL — fails (206, 0 bytes, connection closed)
- pod → a large non-HF file — succeeds (rules out pod egress / MTU / general
  bandwidth)
- VM → the same HF blob URL — succeeds (rules out the Hub and the artifact)

Both `h2` and `HTTP/1.1` fail, so it is not a protocol-negotiation issue. That
pod-fails / pod-succeeds-elsewhere / VM-succeeds triad is the signature.

## Recovery, with measured numbers

- **Relay VM → pod over parallel rsync** — the fix that worked. A single
  transpacific stream runs ~10 MB/s; 4–5 parallel streams reached ~36 MB/s.
  Used for a `vc_bank` store plus a full Qwen2.5-7B-Instruct snapshot, rc=0.
- **Pin a working edge IP via `curl --resolve`** — works but only ~3 MB/s, so
  it is a fallback for small sets, not for a model snapshot.
- Pin `HF_HUB_DISABLE_XET=1` + `HF_HUB_ENABLE_HF_TRANSFER=0` in the run env
  afterward, since the accelerators add nothing once blobs arrive by relay and
  their failure modes are the confusing part.

## Candidate work

- **Primary:** add a `.claude/rules/gotchas.md` entry with the three
  presentations, the curl differential, and the rsync-relay recipe including
  the measured single-vs-parallel stream figures. This is the deliverable; the
  rest is optional.
- **Consider:** a preflight check that does one small-vs-large HF GET on a
  freshly provisioned pod and fails loud with this diagnosis, so the trap is
  caught before an agent starts staging tens of GB. Weigh the added preflight
  latency against the cost of the manual diagnosis — the implementing session
  should decide, and a WARN may be more appropriate than a hard FAIL.
- **Consider:** whether the pod bootstrap should set those two env pins by
  default on DCs where this reproduces, or whether that would mask a
  legitimately working xet path elsewhere. Do not pin them fleet-wide without
  an argument — accelerated uploads are the documented default (#745) and this
  is a per-DC network fault, not a client defect.

## Scope notes

- Do NOT weaken or bypass the existing xet wedge ladder in
  `.claude/rules/upload-policy.md`; this trap sits BESIDE it and the point of
  the entry is to help an agent tell them apart quickly.
- The relay direction matters: VM → pod. The VM had working CDN access, so it
  is the staging hop. A future variant where the VM is the affected side would
  need a different recipe; note that rather than assuming symmetry.
- Root cause is confirmed by the reporting agent, not inferred. The failing
  host was a 4× H100 pod for task #2162; the specific CloudFront edge set is
  DC-dependent, so treat the DC as a variable rather than hardcoding an edge.
