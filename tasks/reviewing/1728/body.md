---
title: 'daily-fix: reuse check (l) needs a call-shape bind; renames '
kind: infra
tags:
- wf-fix
- wf-fix-fp:55dc99aada75
- daily-auto-filed
created_at: '2026-07-27T07:18:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): a reused parent helper
  accepted a kwarg in its signature but rejected it with a runtime assert, so the
  signature-based reuse-fitness check passed and the phase crashed after ~13 h of
  upstream compute; separately a class rename fixed one call site and left the sibling,
  crashing the workload on relaunch'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 1 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Extend the artifact-reuse fitness check (l) with a CALL-SHAPE bind of the actual kwargs the new caller will pass, and add a symbol-rename whole-tree grep requirement to the crash-fix-rounds rule.

## Workflow gap

- **Bug observed:** two reuse/refactor validity gaps each crashed a live run — a reused parent fit helper whose signature accepts a `lambdas=` kwarg but whose code path carries a runtime `assert lambdas is None`, and a class rename that fixed one call site and left the sibling script importing the old name.
- **Why it is a workflow gap:** `.claude/rules/artifact-reuse.md` binds kwarg NAMES against a signature (`inspect.signature(...).parameters`) but never binds the actual call, and `.claude/rules/crash-fix-rounds.md` has no whole-tree grep duty for a renamed symbol.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c -i 'call-shape\|runtime assert' .claude/rules/artifact-reuse.md` → **0** (absence-of-guard evidence); context binding of the nearest existing surface — `.claude/rules/artifact-reuse.md` L697–718 "Signature smoke per kwarg the dispatcher passes" prescribes `missing = dispatcher_kwargs - {f.name for f in fields(TrainLoraConfig)}` and, for non-dataclass callees, `inspect.signature(<fn>).parameters` — a NAME-membership test that a signature accepting `lambdas=` passes by construction; check (l) at L484–513 prescribes reading the instrument's "docs/comments/module constants" for declared boundaries, not binding the call; `grep -c -iE 'rename.*grep|grep -rn .<old' .claude/rules/crash-fix-rounds.md` → **0**, and the only two `rename` hits (L366, L417) concern renamed PATHS under a glob, not renamed symbols; `git log --oneline --since='7 days ago' -- .claude/rules/artifact-reuse.md .claude/rules/crash-fix-rounds.md` → 4 commits, none landing either duty (2026-07-26)

## Evidence

- **(i) Call-shape gap.** Session `5c5a89e8`, 2026-07-26T19:42:33Z: `scripts/issue1689_fit_cells.py` called the reused `issue825_fit_cells.heldout_r2_sweep` with a custom `lambdas=` grid. The parent's `_ridge_predict_cached` carries a runtime guard on the `inner-group-cv` plus inner-cache path: `"issue825_fit_cells.py\", line 329, in _ridge_predict_cached\n    assert lambdas is None, (\nAssertionError: inner-group-cv lambda selection scans the module LAMBDAS grid; a custom lambdas= grid is unsupported with an inner cache"`. The signature accepts the kwarg; the code path rejects it.
- Nothing at plan time and nothing in rounds R1 through R8 bound the actual call against that runtime guard, so the crash surfaced only after Phase B (30/30 on-policy pairs, roughly 13 hours) and Phase C (capture) had completed. Measured cost: crash-fix round R12 — implementer plus code review plus relaunch, roughly 13 minutes — plus the risk of losing 13 hours of upstream compute had the store not persisted.
- The existing instrument is name-scoped by construction: the L697 signature smoke asserts kwarg names are present in the callee's signature. `lambdas` IS present in `heldout_r2_sweep`'s signature, so the smoke passes and the runtime assert still fires. Check (l) is doc-scoped: it directs the planner to read module comments and constants (its own worked example cites `GCV_DOF_CAP` and `lambda_selection="inner-group-cv"` at issue825_fit_cells lines ~66–91), which does not reach an assert buried at line 329 of a code path.
- **(ii) Rename gap.** Session `5c5a89e8`, 2026-07-26T06:35:50Z: an earlier round (R5) renamed `DispatchCall` to `DispatchItem` in `scripts/issue1689_haiku_u2_gen.py` but not in `scripts/issue1689_gen_onpolicy.py:297`. The relaunched workload reached Phase B and died at import, taking the vLLM engine core with it: `"line 297, in generate_and_filter\n    from explore_persona_space.llm.api_dispatch import (  # noqa: E402\nImportError: cannot import name 'DispatchCall' from 'explore_persona_space.llm.api_dispatch'\nERROR 07-26 06:31:08 [core_client.py:564] Engine core proc EngineCore_DP0 died unexpectedly"`.
- The implementer's own failure lesson, emitted `generalizes: yes`, names the exact rule: `"when fixing a class-level API rename, grep the whole scripts/ tree for the old name in the same round and fix every hit — sibling scripts drift until the next phase invokes them."` Measured cost: one full crash-fix round (R9 implementer plus code review plus relaunch), roughly 20 minutes, plus a wasted pod launch cycle.

## Proposed change

- In `.claude/rules/artifact-reuse.md` check (l) (L484–513), add a CALL-SHAPE BIND clause: for every reused fit/analysis helper, the plan records a probe of the ACTUAL call the new code will make — the exact kwargs at their exact values (or minimal stand-ins), executed at smoke shape — not a signature-name membership test. A parent helper can carry runtime asserts that reject a legal-looking kwarg combination.
- Name the diagnostic explicitly: a kwarg present in the signature is NOT evidence the call path accepts it. Direct the probe at the callee's body — `grep -n 'assert\|raise NotImplementedError\|raise ValueError' <reused helper>` for guards naming the kwargs the new caller passes — as the cheap companion to executing the call.
- Amend the L697–718 "Signature smoke per kwarg" step so it states its own limit: name-membership only; a runtime-guard rejection is out of its scope and belongs to the check (l) call-shape bind.
- Mirror the call-shape bind as a plan-time item in `.claude/agents/planner.md` §5, next to the existing reuse-fitness self-attestation, so the probe is recorded before compute is spent upstream of the fit.
- In `.claude/rules/crash-fix-rounds.md`, add a symbol-rename whole-tree grep duty: any round that renames a class, function, dataclass, or module-level constant runs `grep -rn '<old_name>' scripts/ src/` in the SAME round and either fixes or explicitly dispositions every hit; the grep command and its hit disposition are recorded in the implementer marker.
- Have the code-reviewer verify the recorded grep is present whenever the diff contains a rename, so the duty is checked rather than self-attested.

## Scope / surfaces

- Primary target: `.claude/rules/artifact-reuse.md`
- `.claude/rules/crash-fix-rounds.md` (the symbol-rename whole-tree grep duty)
- `.claude/agents/planner.md` (§5 plan-time mirror of the call-shape bind)
- `.claude/agents/code-reviewer.md` (verify the recorded rename grep)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 55dc99aada75

- workflow_fix_target: .claude/rules/artifact-reuse.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: A-P2, A-P3.
