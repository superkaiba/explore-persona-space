---
name: Always Use Adversarial Planner
description: Every new experiment must go through the adversarial planner (Planner → Fact-Checker → Critic) before dispatching an experimenter. No exceptions.
type: feedback
---

Every new experiment must go through the adversarial planner skill before execution.

**Why:** User observed the planner was being skipped too often because the trigger ("for big changes") was vague and easy to rationalize away. Planning in one context is not adversarial planning — the value comes from separate agents with separate contexts catching blind spots. The CoT axis tracking experiment worked well with the full pipeline.

**How to apply:** When the user agrees to run a new experiment, the FIRST step is always to spawn the adversarial planner (Planner → Fact-Checker → Critic → Revise), present the battle-tested plan for approval, and only THEN dispatch the experimenter. The only exceptions are: re-running with different seeds, monitoring/syncing, bug fixes, or explicit user override.
