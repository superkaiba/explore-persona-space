---
title: Amendment-style plan versions break plan.md readers, verify_plan, and the GPU-hours
  gate
kind: infra
tags:
- workflow-fix
- plan-amendment-self-containment
created_at: '2026-08-12T22:08:34Z'
has_clean_result: false
origin_prompt: 'Found at #2223: plan v4 is a 4,752-byte amendment of a 93,612-byte
  v3 and plans/plan.md symlinks to it. Measured: verify_plan.py --issue 2223 flips
  PASS to FAIL with 4 failures on sections v3 already carries, v4 has zero GPU-hours
  declaration lines (the value the approval gate reads), and two subagents had to
  be handed both plan paths explicitly because the documented read-plans-plan.md instruction
  would have given them a partial spec.'
workflow: v1
---
# Amendment-style plan versions silently break every downstream plan reader

## Goal

Make the plan-version contract explicit and enforced: either every `plans/v{K}.md` must be
self-contained, or the workflow (symlink, `verify_plan.py`, the plan-approval GPU-hours read,
and subagent briefs) must resolve an amendment against its base version. Today an amendment
version passes silently into three consumers that all assume self-containment, and one of them
is a gate that can park a task.

## The gap

`plans/plan.md` symlinks to the highest-numbered version, and the project's plan-handoff
convention (CLAUDE.md § Code Style: "pass the PATH to the plan, never the body"; every agent
brief instructs "read `plans/plan.md`") assumes each version is a COMPLETE plan. Nothing
enforces that. A thin AMENDMENT version — a delta doc that says "everything else PORTS FROM
v{K-1} unchanged" — is a legitimate and useful authoring shape, but it breaks three consumers:

1. **Subagent briefs.** Any agent following "read `plans/plan.md`" gets the delta WITHOUT the
   base design — protocol, axis, layers, τ sources, DV, verdict lattice, exactness pins,
   compute envelope all vanish from its view. Nothing warns it that what it read is partial.
2. **`verify_plan.py --issue <N>`.** It verifies the NEWEST `plans/v{K}.md` only, so an
   amendment is checked in isolation and FAILs on content the base version already carries.
3. **The Step-2c plan-approval gate.** It reads the GPU-hours declaration via a first-match
   regex over the newest plan. An amendment with no declaration line yields NO declared value,
   which is exactly the missing-estimate condition that parks the task.

## Measured evidence — #2223 (2026-08-12)

Plan v4 is a **4,752-byte** user-authorized amendment of a **93,612-byte** v3 (adds two arms on
the verbatim directive "run it in parallel now"; states "everything else PORTS FROM v3
unchanged"). `plans/plan.md` → `v4.md`.

Measured with v4 as newest:

```
uv run python scripts/verify_plan.py --issue 2223 --json
  => overall FAIL, n_fail=4, n_warn=2, n_skip=52
  FAILED: c1_source_grounding, c2_measurement_validity, c5_gpu_hours, c8_success_kill_criteria
  WARNED: c3_data_tier, c9_conditions_seeds
```

The same command against v3 returned `PASS, 0 FAIL, 0 WARN`. **All four failures are artifacts
of the amendment shape, not plan defects** — v3 carries every one of those sections. And
`grep -c 'Estimated GPU-hours' v4.md` → **0**, so the approval gate's read finds nothing.

#2223 dodged the gate consequence only because it was approved at v3 (`PLAN_GATE_DECISION:
auto_approved gpu_hours=73.0`) *before* v4 landed. A task amended **before** approval would
park on a missing estimate with no path forward except re-declaring the value in the amendment
— and a later session re-running `verify_plan.py` on any amended task sees a red FAIL it may
reasonably misread as a real plan defect and bounce.

The subagent-brief consequence was live: the round-2 `code-reviewer` and round-3
`experiment-implementer` both had to be handed `v3.md` AND `v4.md` explicitly by the
orchestrator, because the documented `plans/plan.md` instruction would have given them a
partial spec.

## Options (implementer picks; (A) is likely cheapest and most robust)

**(A) Require self-containment, enforced at persist time.** `task.py new-plan-version` FAILs
when the new version is drastically smaller than its predecessor (e.g. < ~40% of v{K-1}) OR
contains amendment-marker phrasing ("AMENDMENT of v", "PORTS FROM v", "unchanged from v")
without carrying the required section set. The author then composes a full plan — trivially
scriptable as "base + delta". Pro: every consumer keeps its current one-file assumption; no
consumer changes. Con: bigger files, and it rejects a genuinely convenient authoring shape.

**(B) Resolve amendments at read time.** Introduce an explicit `Amends: v{K-1}` header;
`plan.md` becomes a rendered composition (base with the delta applied) rather than a symlink to
the raw delta, and `verify_plan.py` verifies the composed document. Pro: keeps amendment
authoring. Con: needs a real composition step and a merge semantics decision (section replace
vs append) — more machinery, more ways to be wrong.

**(C) Minimum viable: detect and warn loudly.** `verify_plan.py` detects the amendment shape,
resolves the base version for the ported checks, and emits one unmistakable WARN naming the
base; `task.py new-plan-version` prints a loud notice that `plan.md` now points at a partial
document. Does NOT fix the subagent-brief or approval-gate consequences on its own — so if (C)
is chosen it must be paired with a fix for the GPU-hours read at minimum.

Whichever is chosen, the **GPU-hours read must not silently find nothing**: either the
declaration is mandatory in every version (A), or the reader falls back to the resolved base
(B/C) — never a silent no-value that parks the task.

## Acceptance

- A fixture reproducing the #2223 shape (a thin amendment over a full base) no longer produces
  spurious `verify_plan.py` failures on sections the base carries.
- The plan-approval GPU-hours read resolves a value for an amended task, or the amendment is
  rejected at persist time with an actionable message.
- The plan-handoff convention is stated explicitly wherever briefs are composed (the agent
  files / SKILL.md instruction that says "read `plans/plan.md`"), so an amendment cannot hand a
  subagent a partial spec without something failing loudly.
- `tests/` pins the chosen behaviour, including the persist-time or read-time detection.
- #2223's own v4 is left alone (a user-authored artifact); the fix must be backward-compatible
  with amended plans already on disk.

## Related

- CLAUDE.md § Code Style (plan-handoff convention: pass the PATH)
- `.claude/skills/adversarial-planner/SKILL.md` § "Log the plan" (`new-plan-version` mechanics;
  it already auto-aligns a self-declared `# Plan v<K>` header, so persist-time inspection of
  version content is established precedent)
- `scripts/verify_plan.py` (`--issue` mode resolves the newest `plans/v*.md`)
- `.claude/skills/issue/SKILL.md` Step 2c (the plan-approval gate's GPU-hours read; #1771 made
  the gate GPU-hour-blind but it still parks on a MISSING estimate)

## Provenance

Found by the #2223 orchestrator (2026-08-12) when a code-reviewer referenced "plan v4's
A2c/A2corr" arms the orchestrator had never persisted — investigation showed a concurrent
chat session had persisted a thin amendment on user direction, moving the `plan.md` symlink.
The three consumer breakages and the four spurious `verify_plan.py` failures were measured
directly, not inferred. Auto-filed per the workflow-fix-on-bug protocol.
