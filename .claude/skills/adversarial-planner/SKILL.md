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

**If this is a `type:batch` issue (the body lists N independent items):**
Structure your plan as N independent sections, one per body item. Each
section gets its own subset of the fields below — Goal, Design (with file
paths and pseudocode), Acceptance criteria, Risks. Skip cross-item
narrative; items are independent. The Assumptions section can be shared,
but call out which assumption belongs to which item if it isn't obvious.

**Before planning, search the web** for how this type of task is typically done. Look for:
- Published papers, blog posts, or repos with similar experiments or architectures
- Established best practices, common pitfalls, standard baselines
- Existing tools, libraries, or pre-computed artifacts you can reuse
- **Hyperparameter recipes for the closest published setting** — the exact
  lr / schedule / batch / epochs / LoRA config reported in the setup or
  appendix table of the nearest paper (same model family + task), AND the
  values any parent / sibling issue already validated. Choose each
  load-bearing hyperparameter to serve this experiment's Goal, quote real
  values (not your memory of them), and carry each into §11 Decision
  Rationale with a `Source:` (arXiv id or issue `#<M>`). See planner.md
  "Before Planning" step 4 + §11 for the full grounding protocol.

Then design your plan:
1. **Goal**: What are we trying to achieve and why?
2. **Prior work**: What did your web search find? What approaches exist and how does this plan relate?
3. **Hypothesis** (if experiment): What do we expect and what would falsify it?
4. **Design**: Concrete steps, file paths, function signatures, configs
5. **Controls**: What comparisons make the results interpretable?
6. **Eval**: How do we measure success? What metrics, what thresholds? Name the hero figure(s) the headline needs AND a short exploratory dump for the analyzer to pick from — over-produce by default (see `planner.md` §6 "Figures to produce").
7. **Risks**: What could go wrong? What are the failure modes?
8. **Resources**: GPU time, disk space, API costs, wall time estimates

9. **Assumptions**: List EVERY factual assumption you are making. Be exhaustive. Include:
   - API/library capabilities ("vLLM supports X", "speculators can do Y")
   - Specific values ("the canonical layer is 32", "hidden_dim is 5120")
   - Infrastructure ("the model fits on one GPU", "the data is cached")
   - Compatibility ("this torch version works with that library")
   For each assumption, state your confidence (high/medium/low) and how you verified it (searched web, read docs, guessed).

Be specific — name files, write pseudocode, specify hyperparameters with a literature / past-issue `Source:` for each load-bearing one (see planner.md §11). Vague plans waste GPU time.
```

Save the plan to a temporary file or pass it directly.

**Strip the harness trailer before persisting.** An `Agent` tool result ends
with harness-appended metadata — a final `agentId: <id> (use SendMessage ...)`
line plus a `<usage>...</usage>` block. Remove BOTH before writing the
planner's return to ANY durable handoff surface (the `/tmp/issue-<N>-plan-v<K>.md`
handoff file, `task.py new-plan-version` → `plans/v<K>.md`), e.g.:

```python
text = re.sub(r"\n?agentId:\s*\S+\s*\(use SendMessage.*?</usage>\s*$", "\n", text, flags=re.DOTALL)
```

A contaminated handoff file reaches every downstream consumer verbatim
(fact-checker, all 6 critics, the committed plan revision) — on task #562
(2026-06-10) both Codex critic twins had to strip the trailer independently
because the orchestrator captured the planner's return verbatim.

### Phase 1.5: Verify Assumptions (Verifier Agent)

**This phase is MANDATORY. Never skip it.**

**Phase 1.5.0 — Mechanical pre-pass (runs FIRST, before the fact-checker spawns).**
Run the structural verifier against the plan version just persisted:

    uv run python scripts/verify_plan.py --issue <N> --json        # task context (newest plans/v{K}.md)
    uv run python scripts/verify_plan.py --plan-file <path> --json # standalone / not-yet-persisted plans

- **Persistence ordering:** `--issue` mode verifies the newest `plans/v{K}.md`. If the
  just-drafted plan has NOT yet been persisted via `task.py new-plan-version` (the plan
  still lives at the `/tmp/issue-<N>-plan-v<K>.md` handoff file), use `--plan-file <handoff>
  --kind <task kind>` instead — and treat an `--issue`-mode exit 2 with "no plans/v*.md" as
  "persist first or use --plan-file", NOT as a bounce.
- **Canonical N/A escape phrases** (quote verbatim in any bounce brief so the planner can
  satisfy a check it is legitimately exempt from): `N/A — no behavioral construct`
  (check 2), `N/A — no model training` / `N/A — no training hyperparameters` (check 1),
  `N/A — not a replication` (check 7), `N/A — no artifact reuse` (check 6).
- **FAIL → bounce to the planner** with the failed-check details (a mechanical-fix
  revision: re-spawn the planner with the FAIL list + the plan path; it patches the
  missing block and the orchestrator persists v{K+1} via `task.py new-plan-version`).
  Mechanical bounces do NOT count against the Phase 3 critic round cap. Cap: 2
  consecutive mechanical bounces — if the same check still FAILs on the third run and
  the orchestrator judges the plan plainly satisfies the requirement in different
  words, proceed anyway (verifier false positive), record `verdict: PASS-with-override`
  + the overridden check ids in the marker note, and emit a workflow-fix candidate
  against `scripts/verify_plan.py`.
- **PASS (with WARNs) → proceed**; copy the WARN lines verbatim into the fact-checker
  brief (and later the critic briefs) as "mechanical pre-pass notes".
- **Post the marker** (VM-side; the adversarial-planner skill always runs in the
  orchestrator session, never on a pod):
  `uv run python scripts/task.py post-marker <N> epm:plan-verify --note '<verdict, n_fail, n_warn, failed/overridden check ids, plan version>'`
  Standalone invocations with no task context skip the marker.

The Planner's assumptions are the #1 source of experiment-invalidating errors. Before the Critic even sees the plan, independently verify every factual claim.

Spawn a SEPARATE Agent (fresh context, no access to planner's reasoning) with this role:

```
You are the FACT-CHECKER. Your ONLY job is to verify the factual assumptions in this plan.
You are NOT evaluating whether the plan is good. You are checking whether the facts it
relies on are TRUE.

ASSUMPTIONS FROM THE PLAN:
[PASTE THE ASSUMPTIONS SECTION]

HYPERPARAMETER SOURCES FROM THE PLAN (§11 Decision Rationale):
[PASTE THE §11 What / Why / Source / Alternatives entries for every load-bearing hyperparameter]

For EACH assumption AND EACH §11 hyperparameter `Source:`:
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
- "lr / epochs / LoRA rank = V because paper P / issue #M uses it" — open the
  cited source (arXiv MCP / `task.py view <M>`). Confirm the value matches AND
  that P / #M's setting (model size, data scale, task) is close enough to
  transfer to this experiment's Goal. A hyperparameter cited to a source that
  reports a different value, or to a setting that does not transfer, is WRONG —
  flag it. A load-bearing hyperparameter marked `ungrounded` is UNVERIFIED —
  flag it for a smoke test before committing GPU time.
```

**After the Verifier returns:**
- If ANY assumption is WRONG: fix it in the plan before proceeding to the Critic. A plan built on wrong facts will waste the Critic's time.
- If assumptions are UNVERIFIED: note them as risks. The Critic should evaluate whether they're blocking or can be tested with a smoke test.
- If all CONFIRMED: proceed to the Critic.

### Phase 2: Parallel Critique (3 Lenses × 2 Reviewers — Codex Ensemble)

Spawn **6 critic agents in parallel**: for each of the 3 lenses (Methodology,
Statistics, Alternatives), launch BOTH a Claude `critic` AND a `codex-critic`
(Codex gpt-5.5 via `companion task`). Fresh context for each — no access to
the planner's reasoning or to each other's output. Per-lens disagreement
between Claude and Codex twins is resolved by the `reconciler` agent in
**in-context mode** (no GitHub markers — verdict text printed to stdout). See
`.claude/workflow.yaml § ensemble_review.doubled_steps[critic]` and
`.claude/agents/reconciler.md` § "Two Output Modes".

**Consistency-checker rides the same spawn batch (when invoked from
`/issue` Step 2).** The orchestrator spawns the `consistency-checker`
agent CONCURRENTLY with the 6 critics (7 parallel spawns in one
message, staggered a few seconds apart per the 429 guidance) — it needs
only the corrected plan + the parent recipe, with no dependency on the
critics' verdicts. Its BLOCK findings are UNIONED with the cross-lens
merged critique handed to Phase 3, so ONE revision round addresses
both; BLOCK / WARN / PASS semantics and the `epm:consistency v1` marker
stay exactly as `/issue` Step 2b defines them — only the scheduling
moved. Standalone `/adversarial-planner` invocations (no task context)
skip it.

**Shared preamble — prepend to each critic's brief before its lens-specific questions:**

```
Before composing your critique, internalize these verdict definitions:

- APPROVE = the experiment will produce interpretable data on the
  research question. Diagnostics, confounds, and alternative explanations
  exist for almost every real experiment, but the downstream pipeline
  (analyzer → interpretation-critic → reviewer → clean-result-critic)
  enforces interpretation discipline using the diagnostics the plan
  already reports. The plan does NOT need a pre-registered gate for
  every concern. If the plan reports the right diagnostics for the
  analyzer to weigh, default to APPROVE and list concerns as bullets
  the analyzer should attend to during interpretation.

- REVISE = the plan is missing something the analyzer pipeline cannot
  recover from. Examples: an essential metric is not computed, a
  control condition that would settle the headline question is missing,
  an infrastructure prerequisite is wrong (pinned library version,
  eval surface mismatch). REVISE means "add this thing to the plan,"
  NOT "add a pass/fail rule about an existing diagnostic."

- REJECT = the design cannot answer the research question even with
  revisions of the kind above. The hypothesis is structurally untestable
  with this method; a different experimental approach is required.

Bias toward APPROVE when the plan is recoverable through analyzer
judgment. Reserve REVISE for missing data / conditions / infrastructure
(NOT missing pre-registered rules). Pre-registered confirmation
conjunctions are an anti-pattern — they crush joint power and produce
spurious Inconclusive verdicts on real signals. Trust the downstream
pipeline.
```

**Critic 1 — Methodology:**
```
You are the METHODOLOGY CRITIC. Evaluate ONLY the experimental design:
1. Is the hypothesis testable with this design?
2. Are controls sufficient to isolate the variable?
3. Are there confounds the analyzer cannot weigh from the reported
   diagnostics? (Confounds that are weighable by the analyzer are NOT
   a reason to REVISE — they are concerns to surface to the analyzer.)
4. Is there a simpler experiment that answers the same question?
5. Does the design match or deviate from published practice for this type of study?
6. Are failure modes identified with fallbacks?
7. Is every load-bearing hyperparameter (lr, schedule, batch, epochs,
   LoRA rank / alpha, weight decay, seq length, optimizer, precision,
   anything novel — full set in planner.md §11) grounded with a
   verifiable `Source:` (paper table or prior issue) whose setting
   transfers to this Goal? Start from the Phase 1.5 fact-checker's
   verdict (CONFIRMED / WRONG / UNVERIFIED); spot-check the source only
   when that verdict looks off. REVISE only when a not-CONFIRMED value is
   also plausibly outcome-changing (would diverge, under-train, or
   truncate the trained completion). See critic.md Methodology lens item 4.

Search the web / arXiv for how similar experiments are typically designed in
published work, including the hyperparameters they report.
Rate (methodology only): REJECT / REVISE / APPROVE.
```

**Critic 2 — Statistics & Measurement:**
```
You are the STATISTICS CRITIC. Evaluate ONLY the measurement plan:
1. Are the metrics sufficient to distinguish the hypothesis from alternatives?
2. Are sample sizes / seed counts adequate?
3. Is the eval suite correct and complete?
4. Are the headline statistic, sample size, and CI methodology appropriate?
   (Pre-registered pass/fail thresholds are NOT required — the analyzer
   pipeline assigns confidence based on the reported diagnostics. Only
   flag if the headline metric or CI methodology is fundamentally wrong
   for the question.)
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

For each alternative explanation, distinguish whether it is fatal (the
design cannot rule it out with any analyzer interpretation) or recoverable
(the analyzer can weigh it descriptively from the diagnostics the plan
already reports). Only fatal alternatives trigger REVISE. Recoverable
alternatives are listed as concerns for the analyzer.
Rate (alternatives only): REJECT / REVISE / APPROVE.
```

**Per-lens ensemble decision (inline in this skill, not an agent):**

After all 6 critics return, for EACH lens independently:

| Claude verdict | Codex verdict | Action |
|---|---|---|
| APPROVE | APPROVE | Lens verdict = APPROVE. |
| REVISE | REVISE | Lens verdict = REVISE. Concatenate findings (dedup exact-same). |
| REJECT | REJECT | Lens verdict = REJECT. Concatenate findings. |
| APPROVE | REVISE/REJECT (or vice versa) | **Disagreement.** Spawn `reconciler` (in-context mode) with brief: `mode: in-context`, role=`critic`, lens=`<lens>`, both verdict bodies, plan_body. Reconciler prints `<!-- epm:plan-critique-reconcile v<n> --> ... <!-- /epm:plan-critique-reconcile -->` to stdout with role-specific verdict (`APPROVE` / `REVISE` / `REJECT` per `.claude/agents/reconciler.md` Step 4 table). Reconciler is required to **preserve REJECT severity** when siding with a REJECT reviewer — it does not silently downgrade to REVISE. Manager parses the printed marker's `**Verdict:**` line directly into `lens_verdict[lens]`. |
| Codex no-show (BLOCKER printed) | (any) | Fall back to single-Claude-critic for this lens this round. Surface a one-line note in the merged critique: "Codex {{lens}} twin no-show this round." |

**Cross-lens merge (after per-lens reconciliation):**

- **Overall verdict = worst of the three lens verdicts.** REJECT > REVISE > APPROVE.
- **Concatenate all critique bodies** with lens labels (`[Methodology Claude]`, `[Methodology Codex]`, `[Methodology Reconcile]` if dispatched, then Statistics, then Alternatives). The manager does NOT editorialize.
- **Deduplicate** only exact-same finding flagged by 2+ critics (same issue, same file/line). Keep both if framing differs.
- Present the merged critique to the planner for revision.

The reconciler may NOT add findings beyond what either reviewer raised. Round
counter does NOT increment for reconciler invocations (per-reviewer cap = 3 rounds).

### Phase 3: Revise (Back to Planner Agent or Main Thread)

If the merged verdict is REVISE or REJECT — or the concurrently-spawned
consistency-checker returned BLOCK (its findings are unioned into the
same merged critique; see Phase 2):

1. Read the plan AND all 3 critic reports (with lens labels) AND any
   consistency-checker BLOCK findings
2. Synthesize: which Must-Fix items are valid? Which (if any) does the planner reject?
3. Produce a revised plan that addresses the valid Must-Fix items
   (critic Must-Fix items + consistency BLOCKs together — one union
   revision round, not two serial bounce rounds).

**Default: do NOT re-critique.** Proceed to user approval with the revised
plan + the round-1 critique attached as context. With the
conclusion-changing bar in `critic.md`, round-1 Must-Fix items are concrete
and specific — the planner integrates them and ships. Rounds 2 and 3 of the
critic loop fire only in the narrow cases below, because each extra round
both costs compute AND tends to accrete additions that wouldn't have made
the conclusion-changing bar on their own. The cap is still 3 total
revision rounds in case re-critique IS warranted.

**Re-critique ONLY if any of:**
- The original verdict was REJECT (design fundamentally flawed; the revised
  version is effectively a new experiment that needs fresh review).
- The revision changed the hypothesis itself or the core experimental design
  (not just "added the missing baseline the critic asked for").
- The revision added a new condition / eval / pipeline stage that was not
  in the round-1 plan AND was not requested by a Must-Fix item (i.e., the
  planner introduced new scope on its own).
- The planner explicitly disagreed with a Must-Fix item and chose not to
  address it — the user needs to see what critics say about the planner's
  defense.

Otherwise — the planner addressed the round-1 Must-Fix items, didn't change
the design, didn't introduce un-asked-for scope — go directly to user
approval. The user is the final critic.

If the Critic round-1 verdict was APPROVE outright: proceed to implementation
with no revisions.

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

# 4. Launch 6 critics in PARALLEL (3 lenses × 2 reviewers).
#    All 6 Agent() calls go in a SINGLE message so they run concurrently.
#    Claude critics return the verdict body directly. codex-critic agents
#    are prompt-composers only: they return a dispatch config naming the
#    prompt file + expected output file. This orchestrator bg-dispatches
#    scripts/codex_task.py for each (Step 4b) — codex-critic agents MUST
#    NOT bg-dispatch themselves (CLAUDE.md § "Codex task dispatch": only
#    the orchestrator's direct bg-Bash invocation delivers a real
#    notification when Codex terminates).
m_claude = Agent(subagent_type="critic",       prompt="[Methodology lens] Critique:\n\n{corrected_plan}",   run_in_background=True)
m_codex  = Agent(subagent_type="codex-critic", prompt="lens=methodology\nplan_body:\n{corrected_plan}",     run_in_background=True)
s_claude = Agent(subagent_type="critic",       prompt="[Statistics lens] Critique:\n\n{corrected_plan}",    run_in_background=True)
s_codex  = Agent(subagent_type="codex-critic", prompt="lens=statistics\nplan_body:\n{corrected_plan}",      run_in_background=True)
a_claude = Agent(subagent_type="critic",       prompt="[Alternatives lens] Critique:\n\n{corrected_plan}",  run_in_background=True)
a_codex  = Agent(subagent_type="codex-critic", prompt="lens=alternatives\nplan_body:\n{corrected_plan}",    run_in_background=True)
# When invoked from /issue Step 2, ALSO add the consistency-checker to
# this same parallel batch (7th spawn; BLOCK findings union into the
# Phase 3 revise round — see /issue Step 2b for verdict semantics):
c_check  = Agent(subagent_type="consistency-checker", prompt="Plan + related-task markers per /issue Step 2b:\n\n{corrected_plan}", run_in_background=True)
# Wait for all spawns to complete.

# 4b. Pick up each codex-critic's dispatch config and bg-dispatch
#     scripts/codex_task.py to actually run Codex. WITHOUT this step,
#     codex_out[lens] holds the dispatch-config text instead of the
#     verdict marker, the per-lens ensemble decision silently drops to
#     single-Claude-critic, AND the dashboard shows no Codex trace
#     because no marker is ever written. The bug: a codex-critic
#     subagent's `Bash(run_in_background=true)` returns IMMEDIATELY but
#     its bg-completion event has no listener after the subagent exits,
#     so dispatch must happen here in the orchestrator (see
#     scripts/codex_task.py module docstring).
#
#     Each codex-critic returns a structured response with these fields
#     (per .claude/agents/codex-critic.md § Step 5):
#       Prompt file: /tmp/codex-critic-<N>-<lens>-prompt.md
#       Expected output file: /tmp/codex-critic-<N>-<lens>-output.md
#       Marker start tag: <!-- epm:plan-critique-codex v<n> lens=<lens> -->
#       Marker end tag: <!-- /epm:plan-critique-codex -->
#       Codex effort: high
#     If the agent returned `BLOCKER: ...` instead (missing lens, missing
#     plugin, malformed brief), skip dispatch — the Codex no-show
#     fallback in Step 5 will fire.
codex_dispatches = {}  # lens -> (output_file, bg_bash_handle)
for lens, codex_agent_out in (("methodology", m_codex),
                              ("statistics",  s_codex),
                              ("alternatives", a_codex)):
    if codex_agent_out.lstrip().startswith("BLOCKER:"):
        codex_out[lens] = codex_agent_out  # preserved so Step 5 sees the BLOCKER
        continue
    cfg = parse_codex_dispatch_config(codex_agent_out)  # extract Prompt file / Expected output file
    # Bg-dispatch in a SINGLE message so all 3 Codex runs proceed concurrently.
    # The orchestrator continues with other turn-local work; the harness
    # delivers a notification on each bg-Bash exit. End the current turn
    # if no other work is in flight rather than blocking on TaskOutput
    # (anti-pattern per CLAUDE.md § "Orchestrator vs subagent
    # re-invocation").
    codex_dispatches[lens] = (
        cfg["output_file"],
        Bash(run_in_background=True,
             command=f"uv run python scripts/codex_task.py "
                     f"--issue {N} --effort high "
                     f"--prompt-file {cfg['prompt_file']} "
                     f"--output-file {cfg['output_file']}"),
    )
# After ALL bg-Bash calls complete (harness notifications), read each
# output file and extract the marker block between the start/end tags.
for lens, (output_file, _bash_handle) in codex_dispatches.items():
    body = Path(output_file).read_text() if Path(output_file).exists() else ""
    marker = extract_marker(body,
                            start=f"<!-- epm:plan-critique-codex v{round} lens={lens} -->",
                            end="<!-- /epm:plan-critique-codex -->")
    if marker:
        codex_out[lens] = marker
    else:
        # Malformed or empty Codex output → treat as no-show. Step 5
        # falls back to single-Claude-critic for this lens this round.
        codex_out[lens] = f"BLOCKER: codex no-show — empty or malformed output at {output_file}"

# 5. Per-lens ensemble decision (see table above):
for lens in ("methodology", "statistics", "alternatives"):
    claude_v, codex_v = parse_verdict(claude_out[lens]), parse_verdict(codex_out[lens])
    if codex_out[lens].lstrip().startswith("BLOCKER:"):
        lens_verdict[lens] = claude_v   # Codex no-show fallback
    elif {claude_v, codex_v} <= {"APPROVE"}:
        lens_verdict[lens] = "APPROVE"
    elif {claude_v, codex_v} <= {"REVISE", "REJECT"}:
        lens_verdict[lens] = max(claude_v, codex_v, key=severity)  # worst-of-two
    else:
        # APPROVE vs REVISE/REJECT — dispatch reconciler in IN-CONTEXT mode.
        rec = Agent(subagent_type="reconciler", prompt=f"""
mode: in-context
role: critic
lens: {lens}
revision_round: {round}
claude_verdict_body: |
{claude_out[lens]}
codex_verdict_body: |
{codex_out[lens]}
plan_body: |
{corrected_plan}
""")
        lens_verdict[lens] = parse_reconcile_verdict(rec)  # APPROVE / REVISE / REJECT (role-specific; reconciler preserves losing-side severity per .claude/agents/reconciler.md Step 4)

# Cross-lens worst-wins merge:
overall = max(lens_verdict.values(), key=severity)

# If REVISE/REJECT: manager revises plan + re-critiques with another 6-critic pass.

# 6. Present final plan to user for approval
# 7. Execute implementation (subagent_type: "experimenter")

# 8. Post-implementation review (subagent_type: "reviewer" — fresh context)
review = Agent(subagent_type="reviewer", prompt="Verify this implementation matches the plan...")

# 9. Fix blockers if any, re-review if needed
```

**Subagent types for each phase:**

| Phase | Subagent Type | Why |
|-------|--------------|-----|
| Planner | `planner` | Read-only + Bash + arXiv MCP / web search. Reads codebase, searches papers, grounds hyperparameters. |
| Fact-Checker | `planner` | Same tools — reads code/configs AND opens cited arXiv papers (arXiv MCP) / `task.py view <M>` to verify hyperparameter sources. |
| Critic — Methodology (Claude) | `critic` | Read-only + Bash. Fresh context, methodology lens. |
| Critic — Methodology (Codex) | `codex-critic` | Thin Claude wrapper → Codex gpt-5.5 via companion task. Methodology lens. |
| Critic — Statistics (Claude) | `critic` | Read-only + Bash. Fresh context, measurement lens. |
| Critic — Statistics (Codex) | `codex-critic` | Thin Claude wrapper → Codex gpt-5.5. Measurement lens. |
| Critic — Alternatives (Claude) | `critic` | Read-only + Bash. Fresh context, alternatives lens. |
| Critic — Alternatives (Codex) | `codex-critic` | Thin Claude wrapper → Codex gpt-5.5. Alternatives lens. |
| Consistency-checker (∥ critics, /issue-invoked only) | `consistency-checker` | Same Phase-2 spawn batch; needs only the corrected plan + parent recipe. BLOCK findings union into Phase 3 revise (verdict semantics per /issue Step 2b). |
| Codex bg-dispatch (×3, one per lens) | Manager (inline) | Bg-Bash `uv run python scripts/codex_task.py --prompt-file <prompt> --output-file <output> --effort high` for each codex-critic dispatch config returned in Step 4. WITHOUT this step, codex_out[lens] holds the dispatch-config text and the ensemble silently drops to single-Claude per lens. Subagents cannot bg-dispatch (no notification listener after they exit). |
| Per-lens reconcile (on disagreement) | `reconciler` | In-context mode; reads both verdicts + plan, prints binding verdict to stdout. |
| Cross-lens merge | Manager (inline) | Manager merges 3 lens verdicts after reconciliation: worst verdict wins, concatenate critique bodies with lens labels. |
| Revision | Manager (inline) | Manager has plan + 6 critique bodies + reconciler outputs in context. |
| Implementation | `experimenter` | Full read/write/bash for coding and running. |
| Implementation Review | `reviewer` | Read-only adversarial check of the implementation. |

All 6 critics run in **parallel** (6 simultaneous `Agent()` calls in a single
message). Each has its own fresh context and specialized lens prompt. They do
NOT see each other's output. After the 3 codex-critic subagents return their
dispatch configs, the orchestrator bg-dispatches `scripts/codex_task.py` for
each in a single message (3 parallel bg-Bash calls). Per-lens reconciler runs
only on Claude-vs-Codex disagreement and is also in-context (no GitHub
markers). Worst case per round: 6 critics + 3 Codex bg-dispatches + 3
reconcilers = 12 invocations.

**Dispatch ordering guards (both bit on 2026-06-09, #545):** (a) bg-dispatch
`codex_task.py` ONLY after the wrapper's completion notification — and gate
the command itself on the prompt file existing (`test -f "$PROMPT_FILE" &&
uv run python scripts/codex_task.py ...`); dispatching ~39 s after spawning
the composer crashed the helper with `FileNotFoundError` on the not-yet-
written prompt. (b) Read each Codex output file only after the helper's
completion line / `epm:codex-task-completed` marker — premature reads hit
missing files and tempt a fallback to the wrong (stale same-issue) output
file.

**Park order:** the plan-approval park (and the `plan_pending` flip) happens
only AFTER the consistency-checker's FINAL verdict is folded in — never on
its interim ack while its full report is in flight. (The checker is now
spawned concurrently with the Phase 2 critics, so its verdict is normally
already in hand by Phase 3 — but the rule stands on any straggler.) On
2026-06-09 #545
parked ~30 min on an uncorrected plan; the checker's late WARN (a substantive
`max_new_tokens` mismatch vs the executed parent rig) then had to be folded
in as a post-park plan v2.


## Rules

- **Planner, Verifier, all 3 Critics, and Implementation Critic MUST be separate agents** with separate context windows. The whole point is independent review.
- **Never skip the Verifier.** Wrong assumptions propagate through the entire pipeline. The Verifier is the cheapest intervention — 30 seconds of web search prevents hours of wasted GPU time. This was added after the corpus projection incident where wrong layer choice and wrong "vLLM can't do this" claims invalidated the first run.
- **Never skip the Critics.** The 3-lens parallel critique catches more than any single critic. Each lens has structural diversity (different prompts/framings), which research shows outperforms debate or angel/devil formats.
- **Never skip the Implementation Critic.** The Implementation Critic catches what the implementer missed. The implementer is biased toward seeing success.
- **Max 3 revision rounds (planning), max 2 fix rounds (implementation).** If it's not converging, surface the disagreement to the user.
- **The user has final say.** Present the plan + critique + revision to the user before executing.
- **Log the plan.** Register every plan revision via `uv run python scripts/task.py new-plan-version <N> --file <draft>.md`. This writes `tasks/<status>/<N>/plans/v<K>.md` and updates the `plans/plan.md` symlink. Downstream subagents read through the symlink.
