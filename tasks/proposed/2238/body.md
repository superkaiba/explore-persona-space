---
title: pod.py provision --wait-for-capacity is structurally inert for CPU intents
  (CPU branch returns before the flag is read)
kind: infra
tags: []
created_at: '2026-08-11T21:36:29Z'
has_clean_result: false
origin_prompt: 'Surfaced during #2054 round reduced-basis-refit-rungs789 fleet provisioning:
  10-shard cpu-bigmem fan-out hit sustained RunPod CPU scarcity and --wait-for-capacity
  could not take effect because cmd_provision''s CPU branch returns at :2836, before
  wait_for_capacity is resolved at :2851.'
workflow: v1
---
## Goal

`pod.py provision --wait-for-capacity` is **structurally inert for every CPU intent**. The CPU branch returns before the flag is even read, so the retry loop wraps only the GPU path. In an autonomous session the gap is actively misleading: the provisioner *prints* that it is auto-enabling unbounded retry, then fails fast on the first `SUPPLY_CONSTRAINT`. Wire the CPU provision path into the existing wait loop (or, at minimum, refuse/warn loudly when the flag cannot take effect).

## The gap (verified by direct code read, not inferred from the failure)

In `scripts/pod_lifecycle.py::cmd_provision`:

- `:2826` — the CPU branch calls `create_cpu_pod(...)`.
- `:2836` — that branch **`return`s**.
- `:2851` — `wait_for_capacity = bool(args.wait_for_capacity) or _autonomous_session()` — never reached on a CPU intent.
- `:2872` — `create_pod_with_wait_for_capacity(...)`, the retry loop, wraps only the GPU `create_pod`.

So `--wait-for-capacity` can never take effect on `cpu-small` / `cpu-mid` / `cpu-bigmem`.

**The raise contract is already correct — only the wiring is missing.** `runpod_api.py:922` states the intent verbatim: `create_cpu_pod` raises `RunPodNoCapacityError` on a null deploy "so the same wait-for-capacity / fallback policy that catches the GPU case can catch this." The exception the loop keys on is already being raised by the CPU path; nothing routes it into the loop.

### Why autonomous mode makes this a false-comfort bug, not just a missing feature

`:2851-2858` auto-enables the flag in autonomous sessions and prints:

```
EPM_AUTONOMOUS_SESSION=1 → auto-enabling --wait-for-capacity (unbounded retry on SUPPLY_CONSTRAINT).
```

with the in-code rationale "the experiment should start when it has space — there is no human to escalate to." On a CPU intent that promise is silently unkept: no warning, no typed refusal, just a fast failure. An autonomous session reading its own log has positive evidence of a retry loop that does not exist. That is the same false-comfort class as #2237 (`--lane-suffix` appearing to cover RunPod pod names when it does not).

## How it surfaced (#2054 round `reduced-basis-refit-rungs789`)

A 10-shard `cpu-bigmem` fan-out hit sustained RunPod CPU scarcity: `cpu5m-16-128` refused 4 consecutive times across `cloudType=ALL` and `COMMUNITY`, and the live `cpuFlavors` GraphQL query returned an **empty flavor list** (supply broadly dry, not flavor-specific). The natural lever — `--wait-for-capacity`, believed armed automatically because the session is autonomous — turned out to be unreachable, so the orchestrator hand-rolled the retry loop it should have gotten from the CLI (`data/issue_2054/rb789_launch/capacity_wait.sh`). Full record: #2054 `epm:progress` **v228**.

No fallback flavor exists to route around it: the only other mapped CPU rows are `cpu3g-2-8` (8 GB) and `cpu3c-8-16` (16 GB), both below the round's parent-measured 21.6 GiB per-unit peak RSS. Waiting was the only lever that preserved the measured 16-vCPU wall basis the plan's fence is sized against — which is exactly why the missing wait hurt.

## Deliverable

1. **Wire the CPU path into the wait loop.** Preferred: hoist the `wait_for_capacity` resolution ABOVE the CPU/GPU branch and give the CPU deploy the same `RunPodNoCapacityError`-keyed retry treatment — ideally by generalizing `create_pod_with_wait_for_capacity` over a deploy thunk rather than duplicating the loop, so the backoff, the per-process attempt budget, the `[wait-for-capacity]` heartbeat lines, and the `EXIT_STILL_WAITING` contract are shared rather than forked. Note the GPU-shaped log line at `:1262-1263` (`{gpu_count}x {gpu_type}`) needs a CPU-legible form.
2. **Fail loud if (1) is declined or deferred.** If the CPU path is deliberately left one-shot, then a `--wait-for-capacity` CPU provision must WARN prominently (and the autonomous auto-enable must NOT print an unqualified unbounded-retry promise on a CPU intent). Silence is the actual defect.
3. **Regression test.** A CPU-intent provision with `--wait-for-capacity` under a mocked `create_cpu_pod` that raises `RunPodNoCapacityError` twice then succeeds must retry and succeed — pinning the wiring, not just the flag's presence. Include an autonomous-mode (`EPM_AUTONOMOUS_SESSION=1`) case, since that is the path that produced the misleading log line.
4. **Check the sibling paths.** The router's CPU lane (`backends/router.py` `_runpod_terminal_rung` and the async `backend_poll.py` CPU legs) should be audited for the same one-shot assumption — a RunPod CPU no-capacity miss there surfaces as terminal. Adjudicate and record whether they inherit the fix.

## Acceptance

- A CPU-intent `--wait-for-capacity` provision demonstrably retries on `SUPPLY_CONSTRAINT` (test in (3) passes), or emits a loud, tested warning if (2) is chosen instead.
- No autonomous-session log line promises unbounded retry on a path that cannot retry.
- `uv run python scripts/workflow_lint.py` passes; existing pod-lifecycle tests stay green.
- The shared wait loop's per-process attempt budget + `EXIT_STILL_WAITING` semantics are preserved for GPU (no regression to the #530/#537 behavior the loop encodes).

## Provenance

Surfaced 2026-08-11 during #2054 round `reduced-basis-refit-rungs789` fleet provisioning, under sustained RunPod CPU scarcity. Both facts verified by direct code read (`pod_lifecycle.py:2826/2836/2851/2872`; `runpod_api.py:922`) before filing. Filed per `.claude/rules/workflow-fix-on-bug.md`. Same-family siblings filed the same day: #2237 (no gate asks whether a lane can mint N distinct pod names for an N-way fan-out) and #2236 (a rule delegating to a critic-lens item that does not exist) — all three are false-comfort/dangling-mechanism gaps, distinct in mechanism: #2236 a dangling pointer, #2237 a missing check, this one unreachable code.
