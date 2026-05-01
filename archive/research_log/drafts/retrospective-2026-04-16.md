# Daily Retrospective — 2026-04-16

**Sessions reviewed:** 19 top-level sessions + 45 subagent sessions (64 JSONL files total)
**Total user messages:** ~150+ genuine messages across sessions
**User corrections:** ~25 explicit corrections/interrupts
**Major event:** Agent architecture redesign — `manager` deprecated, `research-pm` introduced mid-day
**Experiments:** Persona marker leakage (contrastive LoRA), EM LoRA single-GPU vs 8-GPU confound, Tulu 3 midtrain continuation
**Git commits today:** ~13 (strong pace held)
**Drafts written:** 3 new (marker leakage Phase 1/2, EM confound replication, pod5 bootstrap)

## Summary

Today had two big structural wins and one recurring structural weakness. Wins: (1) the agent architecture was redesigned from `manager` → `research-pm` with 7 agent files updated in a single coordinated session (cf33a988); (2) EXPERIMENT_QUEUE.md was migrated to GitHub Issues as the new source of truth (702744a4). Weakness: yesterday's retrospective proposals (CLAUDE.md gotchas for RunPod port instability, HF public repo, upload safety) were **never applied** — the retrospective continues to produce drafts that no one reads before the next session. Also: only 2 of 19 sessions actually booted under the new `research-pm` agent because the transition happened mid-day, and `gate-keeper` was invoked in only 2 sessions all day despite CLAUDE.md requiring it for any new experiment.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---------|----------|-------------|-------------|
| **Retro proposals still not implemented** — 2 days in a row: yesterday's CLAUDE.md gotchas (RunPod port, HF public repo, upload safety) and End-of-Session check are still missing from CLAUDE.md today. | All sessions (systemic) | Add a SessionStart hook that reads the latest `research_log/drafts/retrospective-*.md` and shows unimplemented proposals as a startup banner. Also: retro agent should open GitHub issues (now that we have GH Issues) for each proposal. | `.claude/settings.json`, `.claude/agents/retrospective.md` |
| **Context compaction cascades** — 5 sessions today ran out of context mid-work (b2b17ab7, bc5f2019, a6bbfb39, cf33a988, f3027315). All required continuation summaries, losing nuance. | b2b17ab7, bc5f2019, a6bbfb39, cf33a988, f3027315 | Add compaction-resistance rules to CLAUDE.md: (a) prefer subagents for investigation-heavy tasks (they have fresh context); (b) drop large tool outputs from context after reading (use `Read(limit=)`); (c) proactively split long sessions at natural breakpoints. | `CLAUDE.md` |
| **Gate-keeper bypass** — CLAUDE.md requires gate-keeper for every new experiment, but it was invoked in only 2 of 19 sessions today. Most experiments proceeded to `planner` or directly to `experimenter` without gate review. | b2b17ab7, 78c30057, 20cafd37, and most experiment sessions | Add an explicit trigger list to research-pm.md: "Before dispatching `planner` or `experimenter` for a NEW experimental hypothesis, first dispatch `gate-keeper`. Skip only for: re-runs, monitoring, bug fixes, user override." | `.claude/agents/research-pm.md` |
| **SSH MCP deferred tools still confusing** — Pod5 session (dda31d57) had multiple "Unknown command: /mcp" attempts; SSH tools failed to register as deferred on first try in 1727a60d. | dda31d57, 1727a60d | The ToolSearch line was added to experimenter.md but not to research-pm.md. Also not enforced — sessions still fall back to Bash SSH. Propose: add a PreToolUse hook on `mcp__ssh__*` that checks loading status. | `.claude/agents/research-pm.md`, `.claude/settings.json` |
| **User asks "what is X?" mid-experiment** — b2b17ab7 had 4 such questions ("What is the 8 GPU confound?", "What is the set of personas from the assistant axis paper?"), suggesting the experimenter didn't explain design upfront. | b2b17ab7, 78c30057 | experimenter.md should require a 3-sentence design brief before launching: (1) what we're running, (2) the hypothesis, (3) what success/failure looks like. User signs off, then dispatch. | `.claude/agents/experimenter.md` |
| **manager→research-pm transition incomplete** — settings.json was updated to `"agent": "research-pm"` during cf33a988, but 17/19 sessions today still booted with deprecated `manager`. Only f3027315 and the post-redesign sessions used the new agent. | 17/19 sessions | This is naturally resolving tomorrow (new sessions will inherit the updated settings). But document in project_agent_architecture.md memory: "As of 2026-04-16, research-pm replaced manager. manager.md is at manager.md.deprecated — do not reintroduce." | Memory |
| **Yesterday's "ScheduleWakeup > tight polling" proposal also not applied** — sessions today continued with tight SSH polling during monitoring (no ScheduleWakeup use visible in today's experiment monitoring loops). | b2b17ab7, 78c30057 | This was proposed yesterday for experimenter.md. Reinforce it in the SSH MCP section of experimenter.md. | `.claude/agents/experimenter.md` |

## Failed Approaches (document to prevent retries)

- **code-simplifier agent name** (multiple sessions): Agents tried `subagent_type: "code-simplifier"` but it resolves as `code-simplifier:code-simplifier` (plugin-namespaced). Caused dispatch failures. **Document in:** research-pm.md agent-dispatch reference table.
- **Uncommitted source files blocking worktree subagents** (deep-clean skill in 9499f97f): 9 source files not in git caused worktree init failures. Fixed by `git add`-ing them in commit 2bdb80f. **Document in:** CLAUDE.md pre-launch protocol — "Before dispatching worktree subagents, verify `git status` is clean."
- **uv in nohup shells** (bc5f2019, a6bbfb39): `uv` at `/root/.local/bin/uv` not on default PATH in `nohup uv run ...` commands. **Fix:** always use absolute path `/root/.local/bin/uv run ...` in nohup. **Document in:** experimenter.md SSH MCP section.
- **Anthropic Safety Research cluster with Slurm** (986c6df8): User evaluated this cluster; decision deferred. Too early to document as failed — log as "under evaluation."

## Proposed Changes

### CLAUDE.md

```diff
 ## Gotchas / Known Issues
 
 - **HF Trainer monkey-patch** in `src/explore_persona_space/train/trainer.py` — fragile, will break if Trainer.__init__ changes
+- **RunPod Community Cloud port instability** — Pod IPs AND ports can change on container restart, especially Community Cloud pods. Always verify via RunPod API (`python scripts/pod.py config --check`) before updating SSH config. Port in the web UI may not match the actual forwarded port. (Incident: 2026-04-15 pod5, 4 port changes.)
+- **HF Hub is now a public repo** — `superkaiba1/explore-persona-space` switched from private to public on 2026-04-15 due to hitting 100GB private storage quota. All model uploads go to this public repo. Do not upload sensitive data.
+- **Upload/cleanup safety** — Never use upload scripts that auto-delete local files. Default is `--upload-only`. Deletion requires explicit `--delete-after-upload` flag AND confirmation. (Near-miss: 2026-04-15.)
+- **Worktree subagents require clean git state** — Before dispatching worktree-isolated subagents, run `git status`. If there are untracked source files (`.py`, `.yaml`, etc.), add them or they'll be invisible inside the worktree. (Incident: 2026-04-16, deep-clean skill blocked by 9 untracked files.)
+- **uv in nohup shells** — `uv` is at `/root/.local/bin/uv` on pods, not on the default PATH in `nohup` subshells. Always use the absolute path: `nohup /root/.local/bin/uv run python scripts/...`.
```
**Reason:** Yesterday's proposals (first 3) remain unimplemented; two new gotchas from today (worktree git state, uv nohup PATH) caused real time loss.

```diff
+## Session Hygiene
+
+- **Check latest retrospective on startup** — Before beginning work, read the most recent `research_log/drafts/retrospective-*.md` and confirm whether proposals were applied. If not, apply CLAUDE.md and agent-definition diffs first.
+- **Context compaction resistance** — For investigations likely to exceed 50 tool calls, dispatch a subagent with fresh context instead of working in the main thread. Symptom: if you're reading >10 files to answer one question, you're in compaction territory.
+- **Commit before context fills** — Watch for "context left" warnings. At <30% remaining, `git add` + commit all modified drafts, RESULTS.md, and eval_results JSON so nothing is lost to compaction.
 
 ## After Every Experiment
```
**Reason:** 5 sessions hit context limits today; compaction cascaded through subagent summaries, losing specificity.

### Agent Definitions

**File:** `.claude/agents/research-pm.md`

```diff
 ## On Startup
 
 READ ORDER:
 1. docs/research_ideas.md         -> Aims, subtasks, direction
 2. RESULTS.md                     -> Existing results
 3. research_log/drafts/LOG.md     -> Unreviewed results
-4. EXPERIMENT_QUEUE.md            -> What's planned
+4. GitHub Issues (experiment label) -> Source of truth for experiment queue
+   - `gh issue list --label experiment --state open`
+   - EXPERIMENT_QUEUE.md now points here. Do not treat EXPERIMENT_QUEUE.md as authoritative.
 5. eval_results/                  -> Scan recent files
 6. Agent memory                   -> Past session learnings
+7. Latest retrospective           -> research_log/drafts/retrospective-*.md — apply unimplemented proposals before new work
+
+### Gate-Keeper Trigger (MANDATORY)
+
+Before dispatching `planner` or `experimenter` for a NEW experimental hypothesis, first dispatch `gate-keeper`.
+Skip ONLY for:
+- Re-runs with different seeds
+- Monitoring / status checks
+- Bug fixes in existing experiments
+- Infrastructure work (pod bootstrap, sync)
+- Explicit user override ("skip the gate, just run it")
+
+If you dispatched planner/experimenter without gate-keeper, stop and retry.
+
+### Agent Name Reference
+
+When calling `Agent`, use these exact `subagent_type` values:
+- `research-pm`, `experimenter`, `implementer`, `analyzer`, `reviewer`, `code-reviewer`, `gate-keeper`, `critic`, `planner`, `retrospective`
+- Plugin-namespaced: `code-simplifier:code-simplifier` (NOT `code-simplifier`)
```
**Reason:** Gate-keeper invoked in only 2 of 19 sessions today — pattern is clear that it's being skipped. GitHub Issues migration happened today; tracking-file hygiene section needs to reflect it. `code-simplifier` dispatch failures happened repeatedly.

**File:** `.claude/agents/experimenter.md`

```diff
+### Design Brief (before any launch)
+
+Before dispatching training or eval, post a 3-sentence design brief to the user:
+1. **What we're running** — e.g., "EM LoRA on Qwen-7B, single-GPU, seed=42, on pod2"
+2. **Hypothesis** — e.g., "Predicts misalignment drops from 95% to <10% (Betley replication)"
+3. **Success/failure thresholds** — e.g., "Success: misalignment <15%. Partial: 15-30%. Failure: >30%."
+
+Wait for user confirmation. Only then launch. (Reason: 2026-04-16 had 4+ "what is X?" questions mid-experiment because design wasn't explained upfront.)
+
+### Monitoring Cadence
+
+- First 2 minutes after launch: check every 15-30s (most failures happen at startup).
+- After stable: use `ScheduleWakeup(270)` to wait, not tight polling. 270s keeps you in the prompt cache window (<300s TTL).
+- Minimum wait between pod checks after stability: 120s. Never tight-loop SSH polls — it wastes tokens and spams logs.
+- If a long-running job is expected to complete in >30min, wake-up interval can grow to 1200-1800s.
+
+### uv in nohup
+
+On pods, `uv` is at `/root/.local/bin/uv`, NOT on the default PATH in nohup subshells. Always use the absolute path:
+```
+nohup /root/.local/bin/uv run python scripts/train.py ... > run.log 2>&1 &
+```
+(Incident: 2026-04-16 had multiple "uv: command not found" failures in bc5f2019 and a6bbfb39.)
```
**Reason:** Yesterday's Monitoring Cadence proposal was not applied. Design Brief is a new proposal from today's mid-experiment user confusion pattern.

**File:** `.claude/agents/retrospective.md`

```diff
 ## Rules
 
 1. **All proposals are drafts.** You NEVER directly edit CLAUDE.md, agent definitions, or skills. You propose diffs and the user approves.
+1a. **After writing the draft**, create a GitHub issue with `--label retrospective` listing each proposal as a checkbox so it can be tracked.
+   ```bash
+   gh issue create --label retrospective --title "Retro YYYY-MM-DD: apply N proposals" --body "...checkboxes..."
+   ```
 2. **Be specific.** "CLAUDE.md could be better" is useless. Show exact diffs with line numbers.
```
**Reason:** Two consecutive days of unimplemented proposals. GitHub Issues now exists as the right channel — use it.

### Hooks

**Proposed hook 1:** SessionStart reminder about latest retrospective

```json
{
  "SessionStart": [{
    "hooks": [{
      "type": "command",
      "command": "latest=$(ls -t /home/thomasjiralerspong/explore-persona-space/research_log/drafts/retrospective-*.md 2>/dev/null | head -1); if [ -n \"$latest\" ]; then echo \"=== Latest retrospective: $(basename $latest) ===\"; grep -E '^### |^## Proposed|^- ' \"$latest\" | head -20; fi"
    }]
  }]
}
```
**Reason:** Two days of unimplemented retro proposals means the information isn't surfacing on session start. This puts it directly in the session opening.

**Proposed hook 2:** PostToolUse on `git push` — pod sync reminder (re-proposed from yesterday)

```json
{
  "PostToolUse": [{
    "matcher": "Bash",
    "hooks": [{
      "type": "command",
      "command": "cmd=$(jq -r '.tool_input.command // empty'); if echo \"$cmd\" | grep -qE '\\bgit push\\b'; then echo 'Reminder: code is NOT auto-synced to pods. Run: bash scripts/sync_env.sh <target_pod>'; fi"
    }]
  }]
}
```
**Reason:** Multiple sessions had code pushed but stale on pods, causing experiment failures. (Proposed yesterday, still not implemented.)

### Memory Updates

- **New memory `project_agent_architecture.md`**: Document the 2026-04-16 transition: `manager` deprecated (`manager.md.deprecated`), `research-pm` is the new top-level agent, settings.json updated. Do not reintroduce `manager`.
- **New memory `project_github_issues.md`**: GitHub Issues is now the source of truth for experiment queue. Labels: `experiment`, `proposed`, `approved`, `running`, `under-review`, `retrospective`. EXPERIMENT_QUEUE.md is a pointer only.
- **Update `project_infrastructure.md`**: Pod5 = `thomas-rebuttals-5` (8x H200 SXM 141GB). RunPod Community Cloud ports unstable — always verify via `python scripts/pod.py config --check`.
- **Update `feedback_workflow.md`**: Add "Use subagents for investigation-heavy tasks to preserve main-thread context. Compaction cascades lose specificity."
- **Delete stale**: If any memory references `manager` as the top-level agent, update to `research-pm`.

## Successful Patterns (reinforce these)

- **Agent architecture redesign was clean** (cf33a988) — 7 agent files updated coordinately, settings.json updated, clear deprecation via `.deprecated` suffix. The new research-pm → specialist dispatch pattern is better factored than the old manager.
- **GitHub Issues migration** (702744a4) — EXPERIMENT_QUEUE.md was getting unwieldy. Moving to GH Issues with labels (`proposed`, `approved`, `running`, `under-review`) gives us filtering, assignment, and PR linkage for free.
- **Deep-clean skill with 9-phase audit** (9499f97f) — dispatched 14+ parallel subagents to audit the codebase; found secrets, bugs, and cleanup opportunities. The parallel-subagent pattern held up well.
- **Pod5 bootstrap from bare container** (dda31d57 eventually) — the new `python scripts/pod.py bootstrap pod5` command did the full setup: uv, repo, env, .env, HF cache, preflight. Exactly what was wanted after yesterday's pod5 port thrash.
- **Unified pod CLI (`scripts/pod.py`) matured** — added `health`, `keys`, `bootstrap`, `config --check` subcommands. Single source of truth for pod operations.
- **User explicitly confirmed research-pm design** (cf33a988) — "I want a specialized agent for task management, ideation, and tracking" → delivered. Save this as user-driven architecture decision.
- **Rapid experiment iteration** (78c30057) — detected 8-GPU vs single-GPU confound, dispatched re-run on single GPU, captured in draft. Good scientific discipline.

## Metrics

- Top-level sessions: 19 | Subagent sessions: 45 | Total JSONL files: 64
- User corrections: ~25 (down from ~40 yesterday — improvement)
- Context compactions: 5 sessions (b2b17ab7, bc5f2019, a6bbfb39, cf33a988, f3027315)
- Gate-keeper invocations: 2/19 sessions (should be closer to 4-5 given new experiments)
- Sessions on new research-pm agent: 2/19 (transition mid-day)
- Git commits: ~13 (held yesterday's improved pace)
- Drafts written: 3
- Time split: ~30% infrastructure (pod5, agent redesign, GH issues), ~40% experiments, ~20% debugging, ~10% reading/planning
- Highest-friction session: b2b17ab7 (marker leakage) — context compaction + 4 design questions mid-flight
- Biggest win: agent architecture redesign (cf33a988) + GitHub Issues migration (702744a4)

## Priority Actions for Tomorrow

1. **APPLY yesterday's + today's retro proposals** — CLAUDE.md gotchas and Session Hygiene section, experimenter.md Monitoring Cadence + Design Brief + uv-nohup, research-pm.md Gate-Keeper Trigger + Agent Name Reference. This backlog is now 2 days old.
2. **Set up the SessionStart hook** for retro visibility — it's the mechanism that will break the "proposals never get applied" cycle.
3. **Create GitHub issue from this retrospective** so proposals are tracked, not just drafted.
4. **Verify all sessions tomorrow boot under `research-pm`** — if any still show `manager`, investigate why settings.json isn't being read.
5. **Enforce gate-keeper** on any new experiment proposal before planner is invoked.
