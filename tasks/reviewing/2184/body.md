---
title: 'CPU-intent RunPod provisioning: type the no-port wedge, rotate DC, and document
  a residual route when every CPU DC is dry'
kind: infra
tags:
- cpu-intent-noport-wedge
created_at: '2026-08-07T22:34:55Z'
has_clean_result: false
origin_prompt: 'Filed by the #2162 orchestrator after two consecutive cpu5m-16-128
  no-port wedges in EU-RO-1 and a five-DC no-capacity sweep left the planned cpu-bigmem
  route unavailable with no sanctioned residual.'
workflow: v1
---
# CPU-intent provisioning has no wedge-detection or DC-rotation path, so a no-port wedge stalls the phase and leaves a billing pod for a human to notice

## Goal

Close two gaps in the RunPod CPU-intent provisioning path that a live run
(#2162 P7) hit head-on:

1. **No-port wedge is not detected or handled at provision time.** A
   `cpu5m-16-128` pod reaches `desired_status=RUNNING` while `ssh_host` and
   `ssh_port` stay null indefinitely — the documented RunPod no-port wedge
   (`.claude/rules/compute-backends.md`, #770/#1667). Detecting it required a
   manual `pod.py list-ephemeral` / live-API check of the ssh fields; the
   provisioning call itself did not surface a typed wedge failure, so the pod
   sat billing until a human looked. `.claude/rules/compute-backends.md`
   already says "the sibling RunPod no-port wedge is covered by the watcher's
   wedge arm (#770/#1667)" — verify whether that arm actually covers CPU pods
   and pods wedged inside their first ~10 minutes, which is the window where
   a driving session is most likely to reprovision blindly.

2. **A dry/wedged CPU lane has no documented residual route.** When the only
   DC with capacity for a CPU flavor is the one that wedges, and every other
   DC returns `RunPodNoCapacityError`, there is no sanctioned next step. The
   #2162 session resolved it by ad-hoc human judgment — measuring VM RAM
   headroom and the `earlyoom` kill preference, then routing the CPU-only
   phase to a 1× H100 `eval` pod as a recorded deviation. That reasoning
   should be a rule, not a per-session rediscovery.

## Evidence (all from #2162 P7 dispatch, 2026-08-06/07)

- `cpu5m-16-128` in EU-RO-1, attempt 1 (`h3qqmh1cyp90af`): RUNNING,
  `ssh_host=None`, `ssh_port=None` past the bring-up window. Terminated as
  never-ran.
- Same flavor, same DC, attempt 2 (`7m5i6ob7ozw27x`): identical wedge.
  Terminated as never-ran. Two consecutive wedges in the same DC is a
  DC-specific signal, not bad luck.
- DC sweep for the same flavor across EU-SE-1, EU-NL-1, CA-MTL-4, EU-CZ-1,
  EUR-IS-2: every one `RunPodNoCapacityError`. These fail fast and create no
  pod, so the sweep itself was safe — but it left no route.
- Shared-VM fallback measured, not assumed: 32 cores, ~25 GB of 125 GB RAM
  available, load 22.73/30.24/36.50, and `earlyoom` running with
  `--prefer (^|/)(pytest|python3?)$` — it preferentially kills this exact
  workload class. Disk was not binding (`/` 157 GB avail, `/mnt/eps-data`
  743 GB avail).

## Candidate fixes (the implementing session decides; these are the shapes)

- **Typed wedge detection in the provisioning path.** After the bring-up
  timeout, if the pod is RUNNING with null `ssh_host`/`ssh_port`, raise a
  typed error (a `RunPodNoPortWedgeError`-shaped sibling of
  `RunPodNoCapacityError`), auto-terminate the never-ran pod, and surface a
  reason code. A wedged pod that bills while reporting RUNNING is the
  failure mode to eliminate; a same-DC blind retry is the second.
- **Bias-away rotation on a detected wedge.** On a wedge, retry in a
  different DC rather than the same one — with the caveat this incident
  proves: rotation only helps when another DC has capacity, so rotation must
  degrade into the residual route below rather than looping.
- **A documented residual for a fully dry/wedged CPU lane.** Either a
  sanctioned GPU-lane fallback for CPU intents (provision the smallest GPU
  intent, record the deviation, accept the idle GPU) or a typed refusal in
  the `cpu_fallback_infeasible_for_plan` family. Whichever is chosen, write
  it into `.claude/rules/compute-backends.md` and `.claude/rules/pods.md` so
  the next session does not re-derive it. Note the GPU-lane fallback has a
  real efficiency argument beyond expedience: a dedicated pod's uncontended
  cores beat a contended share of the shared VM (~6× measured on #2054), and
  the idle GPU can absorb a co-located GPU leg — #2162 is reusing the same
  pod for its deferred teacher-forced margin leg for exactly this reason.
- **Record the EU-RO-1 CPU-flavor wedge as a known trap** in
  `.claude/rules/gotchas.md` or `.claude/rules/pods.md`, with the diagnostic
  probe (check `ssh_host`/`ssh_port` against `desired_status` via the live
  API) so the next hit is a two-minute diagnosis.

## Scope notes

- Do NOT weaken the existing fail-loud behavior of
  `RunPodNoCapacityError` or the `cpu_exhausted_no_runpod_lane` typed
  terminal — this task adds a wedge path and a residual route, it does not
  soften a refusal into a silent fallback.
- GCP provisioning is disabled by policy (#2028); the residual route must not
  reach for a GCP CPU shape.
- Confidence is moderate on the exact current provisioning-path behavior (the
  #2162 session observed the symptom and the manual diagnosis, not the code
  path). The planner should read `scripts/pod.py provision` +
  `src/explore_persona_space/backends/router.py` first and correct this
  framing if the wedge is in fact already typed somewhere.
