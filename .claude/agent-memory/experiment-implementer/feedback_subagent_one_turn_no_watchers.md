---
name: Subagent has ONE turn — never park on watchers
description: Background Monitor/until-loop watchers die when an experiment-implementer turn ends; run long smoke phases foreground with explicit timeouts instead.
type: feedback
---

Never end an experiment-implementer turn while waiting on a Monitor or
background until-loop — subagents get exactly ONE turn and are NEVER
re-invoked by background completions; the watchers die with the turn and
the report/marker never gets posted.

**Why:** Task #540 (2026-06-09) — armed three watchers on a ~5-min CPU
smoke (Phase T at 257/281) and ended the turn expecting re-invocation;
all watchers died, the smoke stalled, and the orchestrator had to
re-prompt to finish synchronously. The "end the turn when bg work is in
flight" pattern in CLAUDE.md applies to the ORCHESTRATOR loop, not to
single-turn subagents.

**How to apply:** For smoke runs of minutes-scale, run foreground Bash
with an explicit `timeout` (up to 600000 ms). If a phase genuinely
exceeds 10 min on the VM, shrink the slice (fewer probes/samples,
smaller tiny-model hidden size) until it fits a foreground call —
do not background it.
