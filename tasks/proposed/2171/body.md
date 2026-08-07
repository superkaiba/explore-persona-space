---
title: Wire the authorized-stub grant token for the Step 6d.0 smoke-architecture gate
kind: infra
tags:
- workflow-fix
- smoke-arch-gate
created_at: '2026-08-07T13:13:11Z'
has_clean_result: false
origin_prompt: 'Step 6d.0 gate bounced #2163 on PASS_PARTIAL; its own sanctioned resolution
  (re-authorize stubs in plan §4) has no grant token — SKILL.md annotates it ''not
  yet wired''.'
workflow: v1
---
# Wire the authorized-stub grant token for the Step 6d.0 smoke-architecture gate

## Goal

Give the Step 6d.0 smoke/sweep architecture-parity gate a mechanical grant path for stubs that the
plan has explicitly AUTHORIZED in §4 Design, so an orchestrator that takes the gate's own
sanctioned resolution has a token to post instead of improvising one.

## The gap

`SKILL.md:4515` routes `verdict: PASS_PARTIAL arms_stubbed=<comma-list>` to **REFUSE to dispatch**,
with a bounce whose pivot scope offers two resolutions:

> "resolve them in the diff, OR re-authorize the stubs in §4 Design (canary-like exception,
> **not yet wired**)."

`workflow.yaml:1618-1628` says the same, annotated "(canary-like exception, **v1.1**)".

The second resolution has no landing surface. The routing table's tokens are `PASS_UNIFIED`,
`PASS_CANARY canary_cell=<id>`, `PASS_PARTIAL arms_stubbed=<list>`, `FAIL_NO_CANARY` — and none of
them means "these arms are stubbed AND the plan authorized them". Concretely:

- Re-posting `PASS_PARTIAL` after taking the resolution routes to REFUSE again → the gate becomes
  unsatisfiable via its own documented escape.
- `PASS_CANARY` requires asserting that smoke and sweep **paths diverge** (the
  in-process-vs-subprocess class from #397) and naming a `canary_cell`. A plan whose smoke IS the
  production driver under `--smoke` cannot truthfully assert divergence, so reaching for this token
  means writing a false statement into a durable marker to buy a grant.
- `PASS_UNIFIED` requires "every planned arm resolved REAL or N/A", which forces the orchestrator
  to decide unaided whether an authorized stub counts as `N/A`. That judgment is exactly what the
  gate was built to take away from the orchestrator (#397: consecutive rounds PASSed smoke and
  crashed sweep because a human-ish judgment call went the convenient way).

## Observed on #2163 (2026-08-07)

Two arms could not execute their production path in a VM-local smoke, for reasons that are
properties of the arms rather than defects:

- `upload-verify` — a smoke run MUST NOT write to the HF data repo; `--skip-upload` plus the
  `_smoke` prefix divert are deliberate safety properties.
- `confirm-b-gpu` — the conditional venue-switch cell needs CUDA the VM smoke lane lacks.

The gate correctly refused on `PASS_PARTIAL` (marker v3), the task bounced to `planning`, and plan
v5 added a §4 "Authorized smoke stubs" block naming both arms with per-arm impossibility reasons
and compensating controls. At that point there was no token to post. The orchestrator posted
`PASS_UNIFIED` with both arms as `N/A — authorized smoke stub`, and had to record in the marker
that the grant was an orchestrator judgment on an unwired path rather than a mechanical clearance.
That marker (`epm:smoke-architecture-check v4` on #2163) is the worked example.

## Proposed fix (sketch, for the planner to adjudicate)

Add a fifth token, e.g. `verdict: PASS_AUTHORIZED_STUB arms_stubbed=<comma-list>`, that ADVANCES to
Step 6d.1 only when all of:

1. every name in `arms_stubbed` appears verbatim in an §4 "Authorized smoke stubs" block of the
   CURRENT plan version;
2. each such arm carries a stated impossibility reason AND a named compensating control;
3. every arm NOT in that list still resolves REAL or N/A (an unauthorized FALLBACK anywhere keeps
   the whole marker at `PASS_PARTIAL` → REFUSE);
4. the marker echoes the residual untested seams, so they land in the durable record rather than
   in a reviewer's head.

Then make the gate mechanical rather than prose-only: a `verify_plan.py` check that the §4
authorization block is well-formed (one row per arm, reason + control both non-empty), and a
`workflow_lint.py` check that the `arms_stubbed` list in the marker is a subset of the §4-authorized
set. Update `SKILL.md:4505-4530`, `workflow.yaml § markers epm:smoke-architecture-check`,
`markers.md`, and `experiment-implementer.md` item 5 together, and drop the "not yet wired" /
"v1.1" annotations once wired.

## Acceptance criteria

1. A new grant token exists and is documented identically in SKILL.md's routing table,
   workflow.yaml § markers, and markers.md (no surface says "not yet wired").
2. An authorized-stub marker whose `arms_stubbed` set is NOT fully covered by §4 does NOT grant.
3. An unauthorized FALLBACK arm alongside authorized ones still REFUSES.
4. `experiment-implementer.md` item 5 tells the implementer when to self-tag the new token.
5. Tests pin (2) and (3), plus the well-formedness check on the §4 block.
6. `workflow_lint.py` passes; `--check-lessons-index` updated if a rule file changes.

## Provenance

Filed by the #2163 orchestrator under the workflow-fix-on-bug protocol after the gap forced an
improvised token. Not urgent — #2163 proceeded — but the next orchestrator to hit this has the same
unaided judgment call, and the convenient answer is always available.

workflow_fix_target: .claude/skills/issue/SKILL.md
