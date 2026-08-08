---
title: 'workflow-fix: pgrep without -f under-matches argv, reads live process as dead'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d9c7cbde99bf
created_at: '2026-08-03T18:39:17Z'
has_clean_result: false
origin_prompt: 'armfill2-1739 subagent surfaced: its own bootstrap watchdog used pgrep
  -c "[i]ndex-pack" without -f, printed procs=0 three times while git index-pack was
  alive; would have killed a healthy fetch holding shallow.lock.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a near-miss during the
#1739 armfill round (emitting agent: armfill2-1739 subagent, surfaced 2026-08-03).

## Goal

Add a gotchas entry: liveness probes must use pgrep -af (never bare pgrep -c) when the distinguishing string lives in argv rather than the executable name, and must key termination on positive progress evidence rather than a zero count

## Workflow gap

- **Bug observed:** pgrep/pkill invoked WITHOUT -f matches only the executable name, so a probe keyed on an argv-resident distinguishing string (index-pack, a script name, EngineCore) silently returns 0 and reads a LIVE process as dead
- **Why it is a workflow gap:** an armfill subagent's bootstrap watchdog probed
  `pgrep -c "[i]ndex-pack"` with NO `-f`. `index-pack` is argv, not the
  executable name (which is `git`), so the probe printed `procs=0` three
  times while `git index-pack` was actively unpacking 95,249 objects. Had the
  agent trusted it, the guard would have declared the clone dead and killed a
  HEALTHY in-flight fetch holding `.git/shallow.lock` — corrupting the pack and
  costing another provision cycle on a $22/hr 4xH200. This is a FALSE-DEATH
  liveness bug in the same family as the existing NVML/pgrep entries, but a
  distinct cause: those concern pattern CONTENT and pid VISIBILITY; this one is
  the `-f` FLAG being absent, which the existing entries all silently assume.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c "pgrep" .claude/rules/gotchas.md` -> 13 hits
  (lines 40, 133, 136, 140, 142, 143). Line 40 covers the `pgrep -c ... || echo 0`
  double-print; 133/136/140 cover `^VLLM::EngineCore` pattern CONTENT and USE
  `-af`/`-f`; 142/143 cover NVML pid-namespace visibility. NONE warns that
  OMITTING `-f` silently under-matches an argv-resident string. CLAUDE.md's
  ownership-probe bullet (1 hit)
  mandates bracketing a pattern character but likewise assumes `-f`. (2026-08-03)

## Proposed change (candidate diff sketch — refine in planning)

Add a short entry to `.claude/rules/gotchas.md` near the existing pgrep family:

+ - **`pgrep`/`pkill` without `-f` matches the EXECUTABLE NAME only — an
+   argv-resident distinguishing string silently under-matches and reads LIVE as
+   DEAD.** `pgrep -c "[i]ndex-pack"` returns 0 while `git index-pack` runs
+   (executable is `git`). Same for script names, subcommands, and
+   `VLLM::EngineCore`. Always `pgrep -af '<pattern>'` for liveness/ownership
+   probes. Corollary: never key a KILL on a zero count alone — key it on absent
+   POSITIVE progress evidence (byte growth of a temp/output file over a window),
+   because a zero count is what both "dead" and "mis-probed" look like.

Cross-reference from the CLAUDE.md ownership-probe bullet (which mandates the
bracket but not `-f`) if the planner judges it in scope.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Consider a cross-reference in `CLAUDE.md` § "Orchestrator vs subagent
  re-invocation" (ownership-check bullet) and `.claude/agents/experimenter.md`
  Pre-Launch step 9, both of which prescribe pgrep probes.

## Constraints / invariants

- Documentation-only; no behavior change. `workflow_lint.py` + ruff clean.
- Do not restate the existing NVML pid-visibility or `|| echo 0` entries; this
  is a distinct cause and should read as one short additional entry.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: d9c7cbde99bf
