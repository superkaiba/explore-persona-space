---
title: 'Invert auto lane order: RunPod primary (GPU+CPU), fellows next'
kind: infra
tags: []
created_at: '2026-08-04T03:47:18Z'
has_clean_result: false
origin_prompt: 'can you change it so runpod (on anthropic safety research org) is
  the default (for GPU and CPU) -> then fellows [+ clarification: Both GPU and CPU.
  make sure they use the anthropic safety research org; file kind: infra task + full
  pipeline]'
workflow: v1
---
## Overview / Motivation

User directive (interactive chat, 2026-08-03): make RunPod — on the **Anthropic
Safety Research** RunPod org — the DEFAULT/primary compute lane for **both GPU and
CPU** intents, with **fellows** as the next rung.

Context that motivated it: a live inventory this session found 3 running RunPod
pods at **$64.64/hr** (`pod-1947-r3` 8×H200 $35.12, `pod-1739-ext` 4×H200 $17.56,
`pod-1947-loc` 4×H100 $11.96) while our only fellows job (`18604 eps-issue`) sat
**PENDING (Priority)** behind other fellows' jobs on charmander. The free-lane-first
policy is producing queue waits rather than savings, and the RunPod capacity is on
the Anthropic org. Wall-clock is the scarce resource.

**This INVERTS a deliberately-encoded public contract** (#656/#2028: "RunPod is
reached ONLY as the LAST rung"). It is therefore ARCHITECTURAL — see § Constraints.

## Goal

Change the AUTO backend chain so that, with no `backend:` frontmatter:

- **GPU intents:** `runpod → fellows → nibi → mila` (today: `fellows → nibi → fir → mila`
  with RunPod as a post-exhaustion terminal rung). `fir` stays `available=False`.
- **CPU intents:** `runpod → fellows` (today: RunPod-only — the free SLURM lanes are
  excluded for CPU by #1464, so there is no fallback at all).
- **Org pinning:** every RunPod provision path — GPU and CPU — provably scopes to the
  Anthropic Safety Research team, with a test that pins it.

Rollback to the current free-lanes-first order must stay a single deliberate lever.

## Verified at filing (2026-08-03, this session)

All read live from the installed module / repo, not from CLAUDE.md prose:

- `router.GCP_PROVISIONING_DISABLED = True`;
  `router.DEFAULT_AUTO_LANE_ORDER = ('fellows', 'nibi', 'fir', 'mila')`
  (printed via `uv run python -c "from explore_persona_space.backends import router; ..."`).
- `slurm.CLUSTER_CONFIGS`: `fellows` available=True, `qos='high-eur'`,
  `qos_ladder=[('normal-eur', None), ('low-eur', 'general,overflow')]`;
  `nibi` available=True; `mila` available=True; **`fir` available=False**.
- No lane-order env overrides set (`EPM_AUTO_LANE_ORDER`,
  `EPS_FELLOWS_LADDER_RUNG_WAIT_SECONDS` both unset → defaults live).
- RunPod is **structurally not a lane**: `_runpod_terminal_rung()` is defined at
  `router.py:3323` and called from 5 sites (`3816`, `4588`, `4605`, `4741`, `4792`).
  `DEFAULT_FREE_LANE_ORDER`'s docstring states *"RunPod is NEVER in either list —
  it's override-only by deliberate design."*
- Two independent refusals of `runpod` as a lane token: `auto_lane_order()` (env
  path) and `route()` (config `lane_order` path).
- `ROUTE_REASON_RUNPOD_FALLBACK = "auto_fallback_runpod"` (`router.py:363`).
- CPU: `RUNPOD_CPU_INSTANCE_FOR_INTENT` (`router.py:406`) maps `cpu-small`→`cpu3g-2-8`,
  `cpu-mid`→`cpu3c-8-16`, `cpu-bigmem`→`cpu5m-16-128`;
  `ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD = "cpu_exhausted_no_runpod_lane"` (`router.py:378`)
  is the fail-loud floor for an unmapped CPU intent.
- Team scoping is centralized: `DEFAULT_TEAM_ID = "cm8ipuyys0004l108gb23hody"`
  (`runpod_api.py:61`, comment "Anthropic Safety Research team"), injected as the
  `X-Team-Id` header inside `graphql()` (`runpod_api.py:328`), resolved via
  `_require_env()` (`runpod_api.py:139`, `RUNPOD_TEAM_ID` env override). Both the GPU
  mutation (`podFindAndDeployOnDemand`) and the CPU mutation (`deployCpuPod`,
  `runpod_api.py:713+`) go through `graphql()`, so both already inherit the header.
  ⇒ the org requirement is a PIN-AND-TEST job, not new plumbing.
- Blast radius: `router.py` 6,603 lines; `tests/test_router.py` 9,908 lines / 284 tests.

## Tests that pin the invariant being inverted (must be rewritten, not deleted)

Each of these encodes "RunPod is last / not a lane". They are the CONTRACT — the plan
must state, per test, whether it is re-pointed at the new order or retired with reason:

- `tests/test_router.py:527` `test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted`
- `tests/test_router.py:596` `test_runpod_is_last_rung_after_free_lanes_exhausted_no_gcp`
- `tests/test_router.py:3328` `test_auto_lane_order_env_rejects_runpod`
- `tests/test_router.py:3366` `test_auto_lane_order_env_rejects_runpod_alongside_fellows`
- `tests/test_router.py:3778` `test_route_rejects_runpod_in_config_lane_order`
- `tests/test_router.py:5486` `test_explicit_sweep8g_a100_runpod_still_last_after_full_degraded_walk`
- `tests/test_router.py:3262` `test_default_auto_lane_order_has_no_gcp` (order tuple pin)

## Design questions for the planner (not pre-decided)

1. **Lane vs short-circuit.** Preferred: make `runpod` a first-class lane token in the
   order tuple, dispatched by the existing lane loop (`router.py:1584`), and keep
   `_runpod_terminal_rung()` for the genuine failover callers (GCP workload failover,
   queue-timeout/vanish legs). Alternative: a pre-loop short-circuit. The lane form is
   cleaner but touches validation in two places; the planner picks and justifies.
2. **New reason code.** `auto_fallback_runpod` becomes a misnomer when RunPod is
   primary. Propose `auto_primary_runpod` so `epm:backend-selected` forensics still
   distinguish "RunPod by policy" from "RunPod because everything else ran dry" — the
   distinction that let this session diagnose the #1739 habit-pin pattern. Keep the old
   constant for the genuine-exhaustion and failover paths.
3. **RunPod capacity miss must ADVANCE, not terminate.** Today a failed RunPod launch
   raises `NoComputeAvailableError` because it is terminal. As the FIRST rung, a
   `SUPPLY_CONSTRAINT` / capacity-class refusal must fall through to fellows. A
   non-capacity RunPod failure (auth, quota, malformed request) should still fail loud
   rather than silently masking a bug as a lane miss.
4. **CPU fellows lane is new capability.** #1464 excluded SLURM for CPU deliberately —
   the planner must find and state that rationale before overriding it. Open sub-questions:
   which charmander partition/QoS a CPU-only job takes (the GPU ladder high-eur →
   normal-eur → low-eur may not apply); whether the CPU submit renders `--gres=gpu:0`;
   whether `nibi`/`mila` join the CPU chain or stay excluded (default: stay excluded —
   they lack `/workspace`, the #608 fail-loud `mkdir` death; fellows IS sentinel-drained
   per #1898, which is why fellows specifically is viable here).
5. **`EPM_AUTO_LANE_ORDER` must now accept `runpod`** for rollback/override. The loud
   raise existed as real-money safety; the planner states what replaces that guard.
6. **Rollback lever.** Mirror the #2028 shape: one module constant (e.g.
   `RUNPOD_PRIMARY_LANE`) that rebuilds `DEFAULT_AUTO_LANE_ORDER`, so reverting is a
   one-line flip with no code archaeology.
7. **Spend guard.** Every default dispatch now spends money; the free-lanes-first
   guard is gone. Remaining guards are the Step 2c plan-approval GPU-hour cap and
   `RUNPOD_ACCOUNT_HOURLY_CAP=120` in `.env` (present — its enforcement path is
   *unverified hypothesis — verify at plan time*). The planner states whether an
   additional guard is warranted or explicitly why not. NOTE: dollar-value budget caps
   in code are banned by `tests/test_no_dollar_budget_caps.py` — any guard must respect
   that (GPU-hours / pod-count, never dollars).

## Scope / surfaces

- `src/explore_persona_space/backends/router.py` (primary — lane order, validation,
  lane loop, terminal rung, reason codes, CPU branch)
- `src/explore_persona_space/backends/slurm.py` (CPU submit path on fellows)
- `tests/test_router.py`, `tests/test_slurm_*.py`, `tests/test_backend_*.py`
- `CLAUDE.md` § "Compute backends — multi-lane router" (large prose block stating the
  current order; also the § Pods CPU-intent table)
- `.claude/agents/planner.md` §9 and any critic lens referencing lane precedence
- `scripts/verify_plan.py` (c43 sentinel-lane WARN keys on lane choice)
- Org pin: `scripts/runpod_api.py` (add a test pinning `DEFAULT_TEAM_ID` and asserting
  no provision path bypasses `graphql()`'s `X-Team-Id` injection)

Grep the surface before editing — `grep -rn "DEFAULT_AUTO_LANE_ORDER\|auto_fallback_runpod\|DEFAULT_FREE_LANE_ORDER" src/ scripts/ tests/ .claude/ CLAUDE.md` — and list every hit in the plan.

## Constraints / invariants

- **ARCHITECTURAL — needs user greenlight.** This changes a documented public routing
  contract. The planner MUST set `architectural: true` in the plan frontmatter and
  state the "ARCHITECTURAL — needs user greenlight" banner in the Plan Summary, so the
  task PARKS at `plan_pending` for review instead of auto-approving on its ~0 GPU-h
  cost. Thomas explicitly chose the full-pipeline path in order to review the plan.
- 0 GPU-h. No experiment runs; this is a routing-policy change validated by unit tests.
- Explicit `backend:` pins keep working unchanged (`runpod`, `fellows`, `nibi`, `mila`);
  `backend: gcp` keeps raising `GcpDisabledError` (#2028 untouched — GCP stays disabled).
- In-flight-handle paths (poll, teardown, crash-persist, GCP→RunPod failover of an
  existing handle, `gcp_audit.py`) must not regress.
- Failing tests are rewritten to the new contract with a stated reason each, never
  deleted to make the suite green.
- `scripts/workflow_lint.py` (no-flags) passes; ruff on touched files passes.
- CLAUDE.md prose and the code must not disagree at merge — this session found them
  in sync today, and that must hold after.
