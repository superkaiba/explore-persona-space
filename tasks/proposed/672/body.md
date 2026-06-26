---
title: 'Validate GCP works end-to-end after the wedge fixes: real A100 smoke run on
  the fixed extractor (flat memory) + fault-injection of watchdog self-terminate/#659
  failover + regression tests'
kind: experiment
tags: []
created_at: '2026-06-26T10:24:35Z'
has_clean_result: false
origin_prompt: afterwards test that GCP is working properly (and the bugs from before
  won't happen again)
goal: 'After #669 (GCP wedge recovery) and #671 (extractor output_hidden_states memory-bug
  fix) merge, validate end-to-end that GCP works again — a real a2-highgpu-1g A100-40
  smoke run of the fixed hook-based extractor completes with FLAT per-iteration GPU
  memory and no DHCP-wedge/OOM, AND an injected sustained-network-loss self-recovers
  (in-VM watchdog self-terminate -> TERMINATED -> #659 async failover to RunPod, no
  manual pivot; poller wedge-detection fires exactly-once as the alternate trigger)
  — with all #669/#671 regression tests green on main.'
---
## Goal

After #669 (GCP wedge recovery) and #671 (extractor output_hidden_states memory-bug fix) merge, validate end-to-end that GCP works again — a real a2-highgpu-1g A100-40 smoke run of the fixed hook-based extractor completes with FLAT per-iteration GPU memory and no DHCP-wedge/OOM, AND an injected sustained-network-loss self-recovers (in-VM watchdog self-terminate -> TERMINATED -> #659 async failover to RunPod, no manual pivot; poller wedge-detection fires exactly-once as the alternate trigger) — with all #669/#671 regression tests green on main.


## Blocked-on

**Runs only after #669 (GCP wedge recovery) AND #671 (extractor `output_hidden_states`
memory-bug fix) are both merged to `main`.** Do not launch until both are completed.

## What to validate (3 parts)

**1. Happy path — a real GCP run completes cleanly with FLAT memory.**
Launch a short GPU smoke workload on GCP **`a2-highgpu-1g` (A100-40, the exact rung
that wedged on #667)** running the FIXED hook-based extractor (#671) — a handful of
extraction cells (e.g. em + fact, a few contexts). Confirm:
- (a) runs to completion — no OOM, no DHCP wedge, instance reaches a terminal state;
- (b) **GPU resident memory stays FLAT across iterations** (log per-iteration
  resident; contrast the old climbing 22→30→38 GiB `output_hidden_states` pattern —
  this is the direct check that #671 removed the wedge trigger);
- (c) artifacts land on HF.

**2. Recovery path — fault injection confirms self-recovery.**
- (a) **Watchdog self-terminate:** deterministically unit/integration-test the in-VM
  watchdog logic (mock the metadata/HF reachability probe failing N consecutive
  times → asserts it issues TERMINATE/shutdown). THEN a controlled LIVE test: on a
  GCP smoke VM, inject sustained network loss after a few minutes (e.g.
  `iptables -A OUTPUT -d 169.254.169.254 -j DROP` + block the external endpoint) and
  confirm the watchdog detects sustained loss and forces the instance to TERMINATED
  within the threshold.
- (b) **Failover:** confirm the resulting dead/TERMINATED state triggers the #659
  async failover → **RunPod re-dispatch with NO manual pivot** (the run continues).
  Also confirm the #669 poller wedge-detection path (frozen-non-terminal-phase +
  drain-timeout → `terminal_workload_wedged`) fires **exactly once** as the alternate
  trigger.
- If a LIVE fault injection proves too risky/flaky, the fallback is: the deterministic
  watchdog + poller + failover tests pass AND a documented manual smoke confirms the
  watchdog fires — state which path was taken.

**3. Regression tests green on `main`.**
Run #669's poller tests (frozen-phase → `terminal_workload_wedged` → failover-once),
#671's bit-for-bit / AST / memory-non-growth tests, and `workflow_lint --check-batch-judge-client`
+ no-flags default. All must pass.

## Acceptance / verdict

- A real GCP A100 run of the fixed extractor completes without wedge/OOM, memory flat (logged).
- An injected network-wedge self-terminates via the watchdog within the threshold AND
  re-dispatches to RunPod automatically (no manual pivot) — or the deterministic-tests +
  documented-manual fallback.
- All regression tests green on `main`.
- Final verdict, stated plainly: **"GCP works again, and the #667 hung-RUNNING wedge
  class now self-recovers"** — or the specific residual gap if not.

## Provenance

Follow-on validation requested after the #667 GCP networking-wedge fixes
(deep-research `wf_857f7cfd` → #669 recovery backstop + #671 root-cause). Cost: a
short GCP A100 smoke run + a bounded fault-injection run (a few GPU-h, under the
auto-approve cap).
