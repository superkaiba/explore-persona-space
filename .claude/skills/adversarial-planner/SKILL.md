---
name: adversarial-planner
description: >
  Multi-agent plan-critique-revise loop for big changes. Use when making significant
  architectural decisions, designing new experiments, or planning multi-file changes.
  Spawns a Planner agent, then a Critic agent to find flaws, then the Planner revises.
  After implementation, spawns an Implementation Critic to verify correctness.
  Produces a battle-tested plan AND a verified implementation.
user_invocable: true
---

# Adversarial Planner

When the user invokes `/adversarial-planner` or when you're about to make a big change (new experiment, architectural refactor, multi-file changes), use this multi-agent workflow instead of planning alone.

## When to Use

- New experiment design (hypothesis, conditions, controls, eval)
- Architectural changes affecting multiple modules
- Pipeline changes (training, eval, data processing)
- Any change touching >5 files or >200 lines
- Experiment proposals that will consume significant GPU time

## The 3-Phase Loop

### Phase 1: Plan (Planner Agent)

Spawn an Agent with this role:

```
You are the PLANNER. Your job is to design a concrete, detailed plan for the following task:

[TASK DESCRIPTION]

**Before planning, search the web** for how this type of task is typically done. Look for:
- Published papers, blog posts, or repos with similar experiments or architectures
- Established best practices, common pitfalls, standard baselines
- Existing tools, libraries, or pre-computed artifacts you can reuse

Then design your plan:
1. **Goal**: What are we trying to achieve and why?
2. **Prior work**: What did your web search find? What approaches exist and how does this plan relate?
3. **Hypothesis** (if experiment): What do we expect and what would falsify it?
4. **Design**: Concrete steps, file paths, function signatures, configs
5. **Controls**: What comparisons make the results interpretable?
6. **Eval**: How do we measure success? What metrics, what thresholds?
7. **Risks**: What could go wrong? What are the failure modes?
8. **Resources**: GPU time, disk space, API costs, wall time estimates

Be specific — name files, write pseudocode, specify hyperparameters. Vague plans waste GPU time.
```

Save the plan to a temporary file or pass it directly.

### Phase 2: Critique (Critic Agent)

Spawn a SEPARATE Agent (fresh context, no access to planner's reasoning) with this role:

```
You are the CRITIC. Your job is to find every flaw, gap, and weakness in this plan.
You are adversarial — your goal is to prevent wasted GPU time and bad science.

[PASTE THE PLAN]

**Before critiquing, search the web** to ground your review. Look for:
- How similar experiments are typically designed in published work
- Standard baselines or controls for this type of study
- Known pitfalls or failure modes others have documented
- Whether the proposed approach matches or deviates from established practice

Then critique the plan on these dimensions:
1. **Scientific validity**: Is the hypothesis testable? Are controls sufficient? Could confounds explain the results?
2. **Missing comparisons**: What baselines are needed that aren't included?
3. **Overclaims risk**: Could the results be misinterpreted? What caveats are needed?
4. **Technical feasibility**: Will this actually run? Memory, disk, compatibility issues?
5. **Efficiency**: Is there a simpler way to test the same hypothesis?
6. **Failure modes**: What happens if step X fails? Is there a fallback?
7. **Eval gaps**: Are the metrics sufficient? Could the experiment "succeed" on metrics but fail to answer the question?
8. **Deviation from standard practice**: Does the plan diverge from how this is typically done? If so, is the divergence justified?

Be harsh. It's better to catch problems now than after 8 hours of GPU time.
Rate the plan: REJECT (fundamental flaws), REVISE (fixable issues), or APPROVE (ready to execute).
```

### Phase 3: Revise (Back to Planner Agent or Main Thread)

If the Critic says REVISE or REJECT:

1. Read both the plan and the critique
2. Synthesize: which criticisms are valid? Which are overcautious?
3. Produce a revised plan that addresses the valid concerns
4. **Default: re-critique.** Run the Critic again on the revised plan (max 3 total revision rounds)

**Skip re-critique ONLY if ALL of these are true:**
- The original verdict was REVISE (not REJECT)
- The revision only changed minor details (parameter values, wording, added a baseline)
- No structural changes to hypothesis, conditions, eval methodology, or pipeline design

**If any of these are true, ALWAYS re-critique:**
- The original verdict was REJECT
- The revision changed the hypothesis or experimental design
- New conditions, controls, or eval metrics were added or removed
- The pipeline architecture changed
- The planner disagreed with a criticism and chose not to address it

When in doubt, re-critique. A wasted 30 seconds of agent time is cheaper than a wasted 8 hours of GPU time.

If the Critic says APPROVE: proceed to implementation.

## Phase 4: Post-Implementation Review (Implementation Critic Agent)

After implementation is complete, spawn a SEPARATE Agent (fresh context, no access to the implementation process) with this role:

```
You are the IMPLEMENTATION CRITIC. The plan has been implemented. Your job is to
verify the implementation actually matches the plan and is correct.

APPROVED PLAN:
[PASTE THE FINAL APPROVED PLAN]

Your review process:
1. **Read every file that was created or modified** — do not skip any
2. **Compare implementation against plan** — check every item in the plan was addressed
3. **Run verification** — check imports resolve, configs parse, no syntax errors

Critique on these dimensions:
1. **Plan adherence**: Did the implementation actually do what the plan said? List any items from the plan that were skipped, partially done, or done differently.
2. **Correctness**: Are there bugs, logic errors, off-by-one mistakes, wrong defaults, or broken edge cases?
3. **Integration**: Does the new code integrate correctly with existing code? Are imports right? Do config schemas match what the code expects? Are function signatures compatible with callers?
4. **Missing pieces**: Is anything required for this to actually work that wasn't implemented? (Missing data files, uninstalled deps, untested code paths, etc.)
5. **Regressions**: Could the changes break existing functionality? Check backward compatibility.
6. **Hardcoded values**: Are there magic numbers, hardcoded paths, or assumptions that should be configurable?

For each issue found, classify as:
- **BLOCKER**: Must fix before this can be used (crashes, wrong results, broken integration)
- **ISSUE**: Should fix but won't prevent basic usage (edge cases, missing validation)
- **NIT**: Style or minor improvement (naming, comments, formatting)

Rate the implementation: FAIL (blockers found), FIX (issues but no blockers), or PASS (ready to use).
```

If the Implementation Critic returns FAIL:
1. Fix all BLOCKERs
2. Re-run the Implementation Critic on the fixed code
3. Max 2 fix rounds — if still failing, surface to user

If FIX: address the ISSUEs, no need to re-critique unless fixes were substantial.

If PASS: done.

## Implementation Pattern

```
# In the main thread:

# 1. Launch Planner
planner_result = Agent(prompt="You are the PLANNER. Design a plan for: {task}...")

# 2. Launch Critic (separate agent, fresh context)
critic_result = Agent(prompt="You are the CRITIC. Find flaws in this plan:\n\n{planner_result}")

# 3. If REVISE/REJECT, revise and optionally re-critique
if "REJECT" in critic_result or "REVISE" in critic_result:
    revised = Agent(prompt="You are the PLANNER. Revise this plan based on critique:\n\nORIGINAL PLAN:\n{planner_result}\n\nCRITIQUE:\n{critic_result}")
    # Re-critique for major revisions

# 4. Present final plan to user for approval
# 5. Execute implementation

# 6. Post-implementation review (separate agent, fresh context)
review = Agent(prompt="You are the IMPLEMENTATION CRITIC. Verify this implementation matches the plan:\n\nPLAN:\n{final_plan}\n\nFiles changed: [list them]")

# 7. Fix blockers if any, re-review if needed
```

## Rules

- **Planner, Critic, and Implementation Critic MUST be separate agents** with separate context windows. The whole point is independent review.
- **Never skip the Critic.** The Critic exists to catch the Planner's blind spots.
- **Never skip the Implementation Critic.** The Implementation Critic catches what the implementer missed. The implementer is biased toward seeing success.
- **Max 3 revision rounds (planning), max 2 fix rounds (implementation).** If it's not converging, surface the disagreement to the user.
- **The user has final say.** Present the plan + critique + revision to the user before executing.
- **Log the plan.** Save the final approved plan to `.claude/plans/` for reference.
