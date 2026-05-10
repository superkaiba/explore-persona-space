---
name: Delegation Protocol
description: Rules for spawning the manager agent and when to bypass the standard pipeline
type: feedback
---

Never spawn the manager agent without these four items in the delegation brief:
1. Approved plan (output of adversarial-planner skill, with user approval)
2. Pod + GPU assignment (verified via ssh_health_check or ssh_list_servers)
3. Budget cap (GPU-hours, wall time)
4. Success criteria (quantitative thresholds from the plan)

**Why:** Without these, manager has no execution contract. It can't decide when to stop, what "done" means, or when to escalate. Prior sprawl in manager.md came from ambiguous handoffs where the manager had to rediscover goals mid-run.

**How to apply:**
- Before spawning manager, gate-keeper must have issued RUN (>= 3.5) or explicit user override.
- Before spawning manager, adversarial-planner must have produced a plan the user has approved.
- If the user says "just run it" without those steps, push back once: "gate-keeper + planner is ~10 min and catches half of wasted experiments — want me to do it?" If they insist, note the override in the delegation brief.
- Exceptions that legitimately skip the pipeline: re-runs with different seeds, monitoring/sync tasks, bug fixes, explicit user override. These go straight to manager with a minimal brief.
