---
title: 'workflow-fix: pin MALLOC_MMAP_THRESHOLD_ in launch-prefix rules'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1a40a0867904
created_at: '2026-08-07T09:48:32Z'
has_clean_result: false
origin_prompt: '#1336 inline recovery round: glibc dynamic mmap-threshold retention
  inflated peak anon RSS to 113.5/119.2 GiB and OOM-killed surface 7; pinning MALLOC_MMAP_THRESHOLD_=131072
  cut it to 78.6 GiB. Knob exists in scripts/issue825_dispatch.sh + scripts/issue1345_dispatch.sh
  but in no rule file.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on
task #1336 (emitting agent: orchestrator, own observation during an inline recovery round).

`MALLOC_MMAP_THRESHOLD_=131072` is a MEASURED, load-bearing memory knob for any phase that
sequentially loads and frees large CPU tensors. It exists today in two per-issue drivers
(`scripts/issue825_dispatch.sh`, `scripts/issue1345_dispatch.sh`) and in ZERO rule files, so
every new phase in the class rediscovers it by OOM. This is the "built-but-stranded fix"
class named in `.claude/rules/workflow-fix-on-bug.md`: the fix is real, applied twice, and
never promoted to the workflow surface where the next author would read it.

## Goal

Promote `MALLOC_MMAP_THRESHOLD_=131072` into the workflow-surface launch-prefix rules
alongside the existing `MALLOC_ARENA_MAX=2`, with a gotchas entry naming the discriminating
signature, so a sequential-large-tensor phase carries it by default instead of rediscovering
it by OOM.

## Workflow gap

- **Bug observed:** glibc's DYNAMIC mmap-threshold adjustment retains a freed multi-GB
  tensor bundle across sequential model loads. On #1336 surface 7 this inflated peak anon
  RSS to 113.5 GiB of a 119.2 GiB container cap and cgroup-OOM-killed the phase (rc=137,
  `oom_kill=1`), losing 8 of 32 cells. Pinning the threshold cut the peak to 78.6 GiB on
  comparable work — a ~27% reduction after quantitatively controlling for row count.
- **Why it is a workflow gap:** `.claude/rules/code-style.md`'s launch-prefix bullet
  mandates `MALLOC_ARENA_MAX=2` and explains the ARENA-fragmentation signature ("RSS that
  GROWS ACROSS PASSES with no large single allocation"), but says nothing about the
  mmap-threshold sibling, which is a DIFFERENT mechanism with a DIFFERENT fix and bites the
  large-allocation case the arena cap does not reach. An author following the rule verbatim
  gets the arena cap and still OOMs. The knob has already been discovered independently
  twice (#825 run 5, then copied into #1345) and both times stayed inside a per-issue driver.
- **Confidence (emitter):** high — the mechanism is documented glibc behavior, the fix is
  already applied in two in-repo drivers with a matching written rationale, and this round
  measured the effect directly.
- verified-at-filing: `grep -rln 'MALLOC_MMAP_THRESHOLD' --exclude-dir=worktrees .claude/
  CLAUDE.md scripts/ src/` -> 2 hits in 2 files, BOTH per-issue drivers
  (`scripts/issue825_dispatch.sh:26`, `scripts/issue1345_dispatch.sh:72`), ZERO hits in any
  `.claude/` rule file or `CLAUDE.md`; per-target counts `.claude/rules/code-style.md` 0 and
  `.claude/rules/gotchas.md` 0 (absence-of-guard claim, so the in-target 0-hit IS the
  evidence). Relocation sweep run repo-wide per clause (b) — the two driver hits above are
  the relocation result and are disclosed rather than treated as absence. Landed-fix history
  checked per clause (a'): `git log --oneline --since='10 days ago' -- <each target>` shows
  no commit touching this mechanism on either file. (2026-08-07)

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/rules/code-style.md`, the launch-prefix bullet ("Always run with `nohup`") and the
shared-VM thread-caps bullet both spell the prefix literally; extend both to:

```
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
+   MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
```

and add to the `MALLOC_ARENA_MAX` explanation a sibling sentence naming the second mechanism:

```
+ `MALLOC_MMAP_THRESHOLD_=131072` pins the mmap threshold so LARGE allocations always
+ mmap and their pages RETURN to the OS on free. glibc otherwise RAISES the threshold
+ dynamically after a large free, serving subsequent same-size allocations off the heap
+ (never returned) — so a phase that sequentially loads-and-frees multi-GB tensors
+ accumulates the freed bytes in RSS. DISCRIMINATOR vs the arena cap: arena
+ fragmentation shows RSS growing across MANY SMALL passes; this shows a step change
+ across FEW LARGE load/free cycles, and the arena cap does not fix it.
```

Plus a `.claude/rules/gotchas.md` entry carrying the signature + the three incidents.

## Scope / surfaces

- Primary targets: `.claude/rules/code-style.md`, `.claude/rules/gotchas.md`
- Consider whether the canonical detached-launch recipe in
  `.claude/skills/issue/SKILL.md` § "Detached VM-side long compute phases" should carry the
  knob in its prefix too (it currently spells `MALLOC_ARENA_MAX=2`), and whether
  `orchestrate/env.py`'s shared-VM `setdefault` block is the right place for a default —
  NOTE glibc reads this tunable ONCE at malloc init, BEFORE any Python runs, so an
  `env.py` setdefault CANNOT retrofit the current process (it would only affect
  subprocesses). The launch-prefix is the load-bearing channel; state that explicitly.
- Do NOT change the two existing driver call sites (frozen per-issue reproducibility
  artifacts).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The knob is NUMERICALLY INERT (pure allocator behavior, identical arithmetic), so this is
  a documentation/promotion change with no result-equivalence risk.
- `scripts/workflow_lint.py --check-lessons-index` stays green if a rule file is added;
  `code-style.md` / `gotchas.md` have lint size caps — respect them (gotchas.md is already
  size-capped by `workflow_lint`), so prefer a COMPACT entry over a narrative one.
- This session runs under the workflow-fix recursion guard once filed.

## Provenance

- workflow_fix_target: .claude/rules/code-style.md,.claude/rules/gotchas.md
- fingerprint: 1a40a0867904

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/code-style.md,.claude/rules/gotchas.md
bug_observed: glibc's dynamic mmap-threshold adjustment retains a freed multi-GB tensor bundle across sequential model loads, inflating peak anon RSS ~27% and cgroup-OOM-killing a phase at the container cap (#1336 surface 7: 113.5 GiB of a 119.2 GiB cap, rc=137, 8 of 32 cells lost).
why_workflow_gap: code-style.md's launch-prefix bullet mandates MALLOC_ARENA_MAX=2 and documents the arena-fragmentation signature but never mentions the mmap-threshold sibling, a different mechanism the arena cap does not fix; the knob exists in two per-issue drivers (#825 run 5, #1345) and in no rule file, so each new sequential-large-tensor phase rediscovers it by OOM.
proposed_change: Add MALLOC_MMAP_THRESHOLD_=131072 alongside MALLOC_ARENA_MAX=2 in the launch-prefix rules for phases that sequentially load and free large tensors, with a gotchas entry naming the discriminating signature vs arena fragmentation.
diff_sketch: |
  code-style.md launch-prefix + thread-caps bullets:
  -  ... NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2
  +  ... NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
  + sibling sentence: pins the mmap threshold so LARGE allocations always mmap and
  +   their pages return to the OS on free; glibc otherwise raises the threshold after
  +   a large free and serves later allocations off the heap (never returned).
  + DISCRIMINATOR: arena fragmentation = RSS growth across MANY SMALL passes;
  +   this = step change across FEW LARGE load/free cycles.
  + gotchas.md: compact entry with the signature + #825 run 5 / #1345 / #1336 citations.
confidence: high
related_task: #1336
<!-- /workflow-fix-candidate -->
