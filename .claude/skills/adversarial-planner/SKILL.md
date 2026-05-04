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

## The Loop

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

9. **Assumptions**: List EVERY factual assumption you are making. Be exhaustive. Include:
   - API/library capabilities ("vLLM supports X", "speculators can do Y")
   - Specific values ("the canonical layer is 32", "hidden_dim is 5120")
   - Infrastructure ("the model fits on one GPU", "the data is cached")
   - Compatibility ("this torch version works with that library")
   For each assumption, state your confidence (high/medium/low) and how you verified it (searched web, read docs, guessed).

Be specific — name files, write pseudocode, specify hyperparameters. Vague plans waste GPU time.
```

Save the plan to a temporary file or pass it directly.

### Phase 1.5: Verify Assumptions (Verifier Agent)

**This phase is MANDATORY. Never skip it.**

The Planner's assumptions are the #1 source of experiment-invalidating errors. Before the Critic even sees the plan, independently verify every factual claim.

Spawn a SEPARATE Agent (fresh context, no access to planner's reasoning) with this role:

```
You are the FACT-CHECKER. Your ONLY job is to verify the factual assumptions in this plan.
You are NOT evaluating whether the plan is good. You are checking whether the facts it
relies on are TRUE.

ASSUMPTIONS FROM THE PLAN:
[PASTE THE ASSUMPTIONS SECTION]

For EACH assumption:
1. **Search the web** for the actual answer. Check official docs, GitHub repos, papers.
2. **Read the actual code/config** if the assumption is about the codebase.
3. **State the verdict**: CONFIRMED, WRONG, or UNVERIFIED (couldn't find evidence either way)
4. **If WRONG**: State what the correct fact is, with a source link.
5. **If UNVERIFIED**: Flag it as a risk that needs a smoke test before committing GPU time.

DO NOT trust the plan's reasoning. DO NOT trust your own training data for version-specific
claims (API signatures, library features, default values). SEARCH and READ to verify.

Common traps to watch for:
- "Library X doesn't support Y" — search for recent versions, plugins, workarounds
- "The default value is Z" — read the actual source code or docs, don't guess
- "This model fits in N GB" — calculate from config.json, don't estimate
- "Layer L is the canonical choice" — find the actual paper/repo and confirm
- "This will take N hours" — check against published benchmarks, don't extrapolate
```

**After the Verifier returns:**
- If ANY assumption is WRONG: fix it in the plan before proceeding to the Critic. A plan built on wrong facts will waste the Critic's time.
- If assumptions are UNVERIFIED: note them as risks. The Critic should evaluate whether they're blocking or can be tested with a smoke test.
- If all CONFIRMED: proceed to the Critic.

### Phase 2: Parallel Critique (3 Specialized Critic Agents)

Spawn **3 critic agents in parallel** (each as a separate `Agent()` call with
`subagent_type: "critic"`). Each receives the same plan but a different
specialized lens. Fresh context for each — no access to planner's reasoning
and no access to each other's output.

**Critic 1 — Methodology:**
```
You are the METHODOLOGY CRITIC. Evaluate ONLY the experimental design:
1. Is the hypothesis testable with this design?
2. Are controls sufficient to isolate the variable?
3. Are there confounds that could explain a positive result?
4. Is there a simpler experiment that answers the same question?
5. Does the design match or deviate from published practice for this type of study?
6. Are failure modes identified with fallbacks?

Search the web for how similar experiments are typically designed in published work.
Rate (methodology only): REJECT / REVISE / APPROVE.
```

**Critic 2 — Statistics & Measurement:**
```
You are the STATISTICS CRITIC. Evaluate ONLY the measurement plan:
1. Are the metrics sufficient to distinguish the hypothesis from alternatives?
2. Are sample sizes / seed counts adequate?
3. Is the eval suite correct and complete?
4. Are the success/kill thresholds appropriate and pre-registered?
5. Could the experiment produce an uninterpretable result?
6. Do numerical claims in the plan match actual data files in the codebase?

Rate (measurement only): REJECT / REVISE / APPROVE.
```

**Critic 3 — Alternative Explanations:**
```
You are the ALTERNATIVE EXPLANATIONS CRITIC. For EVERY predicted positive result:
1. What is the simplest explanation that does NOT require the claimed mechanism?
2. Does the plan's design rule out that alternative?
3. What additional control or baseline would be needed to rule it out?
4. What would a skeptical reviewer say about this result?
5. Are there missing comparisons or baselines?

Competitive framing: find the most issues (5 points each). Your goal is to
identify every alternative explanation the plan fails to address.
Rate (alternatives only): REJECT / REVISE / APPROVE.
```

**Merge step (inline in this skill, not an agent):**

After all 3 critics return, merge their verdicts:
- **Overall verdict = worst of the three.** If ANY critic says REJECT → REJECT.
  If ANY says REVISE → REVISE. If all say APPROVE → APPROVE.
- **Concatenate all 3 reports** with lens labels (`[Methodology]`, `[Statistics]`,
  `[Alternatives]`) for the planner — the manager does NOT editorialize.
- **Deduplicate** only exact-same finding flagged by 2+ critics (same issue,
  same file/line). Keep both if the framing differs.
- Present the merged critique to the planner for revision.

### Phase 3: Revise (Back to Planner Agent or Main Thread)

If the merged verdict is REVISE or REJECT:

1. Read the plan AND all 3 critic reports (with lens labels)
2. Synthesize: which criticisms are valid? Which are overcautious?
3. Produce a revised plan that addresses the valid concerns
4. **Default: re-critique.** Run all 3 critics again on the revised plan (max 3 total revision rounds)

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

Use the dedicated subagent types for each phase. Subagents cannot spawn other subagents (Claude Code hard constraint), so this skill (running in the invoking agent's context) must orchestrate each phase sequentially.

```
# In the main thread (manager orchestrates):

# 1. Launch Planner (subagent_type: "planner")
planner_result = Agent(subagent_type="planner", prompt="Design a plan for: {task}...")

# 2. Extract assumptions from planner output, launch Fact-Checker (subagent_type: "planner")
#    Use a planner agent for fact-checking too — it has Read/Grep/Glob/Bash for verification
verifier_result = Agent(subagent_type="planner", prompt="You are the FACT-CHECKER. Verify these assumptions:\n\n{planner_assumptions}")

# 3. If any assumption is WRONG: fix the plan before proceeding
if "WRONG" in verifier_result:
    # Update the plan with corrected facts, then proceed

# 4. Launch 3 critics in PARALLEL (each subagent_type: "critic", fresh context, different lens)
#    All 3 Agent() calls go in a SINGLE message so they run concurrently.
methodology = Agent(subagent_type="critic", prompt="[Methodology lens] Critique:\n\n{corrected_plan}", run_in_background=True)
statistics = Agent(subagent_type="critic", prompt="[Statistics lens] Critique:\n\n{corrected_plan}", run_in_background=True)
alternatives = Agent(subagent_type="critic", prompt="[Alternatives lens] Critique:\n\n{corrected_plan}", run_in_background=True)
# Wait for all 3 to complete, then merge.

# 5. Merge: worst verdict wins. Concatenate all 3 reports with lens labels.
# If REVISE/REJECT: manager synthesizes plan + merged critique, revises, re-critiques
if any_reject_or_revise(methodology, statistics, alternatives):
    # Manager revises the plan directly (it has plan + all 3 critiques in context)
    # Then re-critique with fresh 3-critic parallel pass for major revisions

# 6. Present final plan to user for approval
# 7. Execute implementation (subagent_type: "experimenter")

# 8. Post-implementation review (subagent_type: "reviewer" — fresh context)
review = Agent(subagent_type="reviewer", prompt="Verify this implementation matches the plan...")

# 9. Fix blockers if any, re-review if needed
```

**Subagent types for each phase:**

| Phase | Subagent Type | Why |
|-------|--------------|-----|
| Planner | `planner` | Read-only + Bash. Reads codebase, designs plan. |
| Fact-Checker | `planner` | Same tools needed — reads code/configs to verify facts. |
| Critic — Methodology | `critic` | Read-only + Bash. Fresh context, methodology lens. |
| Critic — Statistics | `critic` | Read-only + Bash. Fresh context, measurement lens. |
| Critic — Alternatives | `critic` | Read-only + Bash. Fresh context, alternative explanations lens. |
| Merge | Manager (inline) | Manager merges 3 critic reports: worst verdict wins, concatenate with lens labels. |
| Revision | Manager (inline) | Manager has plan + merged critique in context. |
| Implementation | `experimenter` | Full read/write/bash for coding and running. |
| Implementation Review | `reviewer` | Read-only adversarial check of the implementation. |

All 3 critics run in **parallel** (3 simultaneous `Agent()` calls). Each has its own
fresh context and specialized lens prompt. They do NOT see each other's output.


## Rules

- **Planner, Verifier, all 3 Critics, and Implementation Critic MUST be separate agents** with separate context windows. The whole point is independent review.
- **Never skip the Verifier.** Wrong assumptions propagate through the entire pipeline. The Verifier is the cheapest intervention — 30 seconds of web search prevents hours of wasted GPU time. This was added after the corpus projection incident where wrong layer choice and wrong "vLLM can't do this" claims invalidated the first run.
- **Never skip the Critics.** The 3-lens parallel critique catches more than any single critic. Each lens has structural diversity (different prompts/framings), which research shows outperforms debate or angel/devil formats.
- **Never skip the Implementation Critic.** The Implementation Critic catches what the implementer missed. The implementer is biased toward seeing success.
- **Max 3 revision rounds (planning), max 2 fix rounds (implementation).** If it's not converging, surface the disagreement to the user.
- **The user has final say.** Present the plan + critique + revision to the user before executing.
- **Log the plan.** Save the final approved plan to `.claude/plans/` for reference.
