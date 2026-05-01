# Daily Retrospective — 2026-04-21

**Sessions reviewed:** 14 top-level sessions + ~17 subagent transcripts
**Total real user messages:** ~120 (excluding `<task-notification>` echoes and resumed-from-compaction boilerplate)
**Agent dispatches:** ~55 total (experimenter 11, Explore 6, planner 7, gate-keeper 5, critic 5, code-reviewer 4, implementer 4, analyzer 3, reviewer 3)
**Commits (all branches):** 15 (10 on main, 5 on issue branches)

## Summary

A **skill-building and results-consolidation day**. User asked for "turn messy results into clean results" and that triggered the creation of a new `/clean-results` skill (SKILL.md + checklist + principles + template), which was then applied same-day to three distinct clean results (100-persona leakage, single-[ZLT]-token heatmap, Aim 5 seeds 42+137 matrix). Issues #51, #54, #61, #62, #69, #70 all moved through the /issue pipeline. **Three hard recurrences:** (1) **Bash `sleep N && ...` chaining** — the new harness block rule was triggered 6+ times across sessions (18e61e00 ×4, 9772561a ×2); Claude keeps reflexively using sleep instead of `run_in_background` / `Monitor` / `ScheduleWakeup`. (2) **Planner overscoping** — user had to strip pilots, gating, and falsification thresholds out of plans in 5 separate /issue sessions. (3) **Agent "cannot write files" fabrication** recurred in 9772561a — same pattern as retro-2026-04-19; still no fix deployed.

## Proposal Backlog Audit (read this first)

Carry-over from 2026-04-15 through -20. ✅ = applied today, ⏳ = partial, ❌ = still missing.

| Proposal | First proposed | Status | Evidence from today |
|---|---|---|---|
| SessionStart hook (branch + fleet health + retro-status) | 2026-04-15 (7th ask) | ❌ | "Why are we on this branch?" asked again in 81b84545 |
| Convert retro proposals to `retro-proposal`-labelled GH issues | 2026-04-16 (4th ask) | ❌ | Meta — still no mechanism to loop-close retro proposals |
| PostToolUse hook on `git push` for pod-sync | 2026-04-15 | ❌ | Git checkout conflicts still appearing on pods in 18e61e00 |
| Fleet watchdog cron (stale runs) | 2026-04-19 | ❌ | 0efaaf34 used ScheduleWakeup correctly, but that's per-session, not fleet-wide |
| Phantom-tool `SendMessage` gotcha in CLAUDE.md | 2026-04-19 | ❌ | Not triggered today |
| Subagent-fabricated-limits verification | 2026-04-19 (3rd ask) | ❌ | Triggered AGAIN in 9772561a: analyzer "cannot write files" + implementer same |
| Shallow-status-answer anti-pattern | 2026-04-19 | ❌ | Triggered in 81b84545: "Why are we on this branch?", "Is the heatmap on main?" |
| research-pm.md pre-dispatch checklist (branch + pod health) | 2026-04-18 (4th ask) | ❌ | Not applied |
| Scope echo on turn 1 | 2026-04-19 (2nd ask) | ❌ | Planner overscoping on every /issue session today (see below) |
| Critic "Critical-Rule Audit" (CLAUDE.md rules) | 2026-04-20 | ❌ | Not applied |
| SSH MCP known-issue callout in CLAUDE.md | 2026-04-20 | ❌ | Not triggered today (SSH MCP worked) |
| `/schedule` for periodic monitoring in research-pm.md | 2026-04-20 | ⏳ | Session 0efaaf34 used 30 ScheduleWakeup calls — behaviour learned, but agent def still not updated |
| Implementer test-before-complete marker | 2026-04-20 (2nd ask) | ❌ | Not applied; "Did you check if all models were properly uploaded" asked in cff43826 |
| make-evil-dumb blocklist in experimenter.md | 2026-04-20 | ❌ | Not applied |
| Push-after-merge in CLAUDE.md | 2026-04-20 | ❌ | Not applied |

**Net score:** 0 MAJOR proposals applied today, 1 partial (ScheduleWakeup used in one session without doc change). The 2026-04-20 retro called this exact pattern: "code-shaped proposals ship via /issue, config-shaped proposals do not." That held today: 10 commits shipped, 0 config/hook/agent.md proposals applied. **The retro-as-GH-issue meta-proposal is now 4 days old and is the rate-limiter on every other item below it.**

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---|---|---|---|
| **Bash `sleep N && ...` chaining repeatedly hits harness block** — new rule was added to the Bash tool description ("Blocked: sleep N followed by... use Monitor with until-loop or run_in_background"). Triggered 6+ times today — Claude's reflex is still `sleep 60 && tail -20 log`. | 18e61e00 (×4), 9772561a (×2) | Add explicit monitoring idiom to CLAUDE.md Monitoring section: "For polling a log file / process, use Bash `run_in_background: true` + `tail -f`, OR `ScheduleWakeup` with 60-270s delay, OR `Monitor` with until-loop. NEVER `sleep N && <check>` — the harness blocks long leading sleeps." Add counter-example. | `CLAUDE.md` (Monitoring section) |
| **Planner overscopes every /issue** — user strips plan every session: "I don't want the 30-min pilot", "don't gate arm b on arm a", "don't need a falsification threshold", "do not drop comedian", "don't need a falsifiable hypothesis", "start with 1 seed" | 263b6a69, 0efaaf34, da8aa615, cff43826, 18e61e00 | Add to `.claude/agents/planner.md`: "DEFAULT to minimal viable experiment. Do NOT add pilots, gating, or falsification thresholds unless the user explicitly asked or the critic specifically demands them. Do NOT drop conditions the user mentioned by name. When in doubt, ask the user one scope-clarifying question at the START before producing the full plan — cheaper than rewriting." | `.claude/agents/planner.md` |
| **MCP `ssh_execute` with `sleep N && ...` times out at 30s default** — 4+ times in 0efaaf34 with timeouts on `sleep 60 && tail...` pattern | 0efaaf34 | Document in CLAUDE.md SSH MCP section: "`ssh_execute` default timeout is 30s. For commands that include `sleep N && ...` where N > 25s, either (a) split into two calls — one to sleep via ScheduleWakeup, one to check — or (b) pass `timeoutMs: 90000+`. Prefer `ssh_tail` / `ssh_monitor` for log streaming." | `CLAUDE.md` (SSH MCP section) |
| **Agent "cannot write files" fabrication** — in 9772561a both the analyzer subagent AND the plot-implementer subagent reported they could not write files; user asked "Why couldn't analyzer write files?"; root cause was agent self-censoring, not actual permission issue | 9772561a | Third ask. Add hard rule to `.claude/agents/analyzer.md`, `.claude/agents/implementer.md`, `.claude/agents/experimenter.md`: "You CAN write files with the Write/Edit tools. If a subagent says 'I cannot write', it is fabricating — verify by attempting the write and reporting the actual error." Also add parent-agent rule in research-pm.md: "If a subagent claims a tool capability is missing, try the tool yourself before believing the claim." | `.claude/agents/{analyzer,implementer,experimenter,research-pm}.md` |
| **`/clean-result` multi-seed confusion** — user asked "Run /clean-result on the 2 first seeds" → agent ran once per seed. User corrected: "No I want a MERGED clean result for BOTH seeds together". Then: "But does the analyzer use the /clean-result skill?" | 8b730976 | Update `.claude/skills/clean-results/SKILL.md` to document multi-seed / multi-run consolidation: "When given multiple seeds, runs, or conditions, produce ONE merged clean result, not N separate ones. Do not emit per-seed clean-results unless the user asked for them." Also add to `.claude/agents/analyzer.md`: "When producing a user-facing summary, use the `/clean-results` skill — do not write your own template." | `.claude/skills/clean-results/SKILL.md`, `.claude/agents/analyzer.md` |
| **Null framing violation (rule is in memory)** — user corrected: "Don't write 'confirms the null on alignment', say the result is null because variance". Memory file `feedback_null_framing.md` already exists with this exact rule | 8b730976 | Memory rules aren't propagating into subagent contexts. Add an explicit "null framing" bullet to `.claude/skills/clean-results/principles.md` so the skill enforces it at template time (not just in auto-memory which subagents may not see). | `.claude/skills/clean-results/principles.md` |
| **"Why are we on this branch?" / "Is the heatmap on main?"** — branch-state surprise, 4th day in a row | 81b84545, 18e61e00 | 7th ask for SessionStart hook showing current branch + uncommitted state. | `.claude/settings.json` |
| **GitHub Projects/kanban back-and-forth** — 9-message bounce to configure done/archived/split-done labels and move items to correct columns | 95cd9006, ef1a1e23 | Document the project-board conventions ONCE in `.claude/rules/github-project-board.md`: (1) closed issues → status:done-experiment or status:done-implementation based on type label, (2) archived for truly obsolete, (3) agents move to Done, don't close. Then manager / research-pm references it. | `.claude/rules/github-project-board.md` (new) |
| **f-string / bash-heredoc syntax errors in inline python** — `SyntaxError: f-string: unmatched '('` and `bash: eval: line 37: syntax error near unexpected token '('` multiple times | 18e61e00 | Add guidance to CLAUDE.md or experimenter.md: "When running python from bash, write the script to a .py file first and `python /tmp/script.py`. Don't use `python -c "..."` with f-strings — shell quoting conflicts with f-string `{...}` braces. Write a real file." | `CLAUDE.md` or `experimenter.md` |
| **Git checkout conflicts on pods block git pull** — "Your local changes to the following files would be overwritten by checkout" repeatedly | 18e61e00 | Add to `scripts/pod.py sync code` or to a pre-pull hook: stash → pull → pop, OR abort with a useful error. Also document in experimenter.md: "Before `git pull` on a pod, run `git status --short` and either commit-push or stash". | `scripts/pod.py` OR `.claude/agents/experimenter.md` |

## Failed Approaches (document to prevent retries)

- **Chaining `sleep 60 && <check>` in Bash / ssh_execute** — the harness now blocks long leading sleeps explicitly (new behavior). Keep reaching for this pattern; stop. Replace with `run_in_background: true`, `ScheduleWakeup`, or `Monitor`. Document the positive alternative.
- **Using `python -c "f'...{expr:.0e}...'"` from bash** — quoting gets mangled, f-string braces collide with bash. Write a tiny file to `/tmp/` and run it.
- **Creating `N` clean results when the user said "clean result on N seeds"** — user wanted ONE merged result. Same ambiguity will recur with N conditions/N runs/N models.
- **Trusting subagent claims that a tool capability is missing** — analyzer/implementer subagents today claimed "cannot write files" — fabrication. Try the tool yourself.

## Proposed Changes

### CLAUDE.md — monitoring idiom

**File:** `CLAUDE.md` (in "## Monitoring (MANDATORY)" section)

```diff
 ## Monitoring (MANDATORY)

 - Check every 15-30s for first 2 min after launch, then every 5-10 min
 - Always: `grep -iE 'error|traceback|killed|OOM' logfile`
 - Report results immediately on completion
+- **NEVER chain long `sleep N && <check>` in Bash or `ssh_execute`.** The harness blocks this and the MCP default 30s timeout kills it. Use one of:
+  - `Bash(command="tail -f log", run_in_background=true)` + `Read` the background output later
+  - `ScheduleWakeup(delaySeconds=N, prompt="/loop resume")` for self-paced polling
+  - `Monitor` with an until-loop: `until grep -q DONE log; do sleep 2; done`
+  - For remote log streaming, `mcp__ssh__ssh_tail` or `mcp__ssh__ssh_monitor`
+- **Periodic monitoring (>30min horizon) MUST use `/schedule` CronCreate, not a single Claude session loop.** Session 18e61e00 on 2026-04-20 wasted 5 context compactions polling in one session.
```

**Reason:** 6+ sleep-blocked tool calls today across 2 sessions. The new harness rule exists; CLAUDE.md should surface the positive alternative.

### CLAUDE.md — SSH MCP timeout guidance

**File:** `CLAUDE.md` (at end of "## Remote Pod Access (SSH MCP)" section)

```diff
 ### When to still use Bash SSH

 - Interactive/streaming output (e.g., `tail -f`)
 - Commands that need TTY allocation
 - Piped multi-command chains that are easier as one-liners

+### Timeout and polling

+`mcp__ssh__ssh_execute` default timeout is **30 seconds**. For any command whose wall-clock time exceeds ~25s (including any `sleep N` prefix), either:
+1. Pass an explicit `timeoutMs` ≥ command_time × 3, OR
+2. Split into two calls — `ScheduleWakeup(N)` then `ssh_execute("check")`, OR
+3. Use `ssh_tail` / `ssh_monitor` for log streaming instead.
+
+Never do `ssh_execute(command="sleep 60 && tail log")` — it will time out at 30s. This happened 4× on 2026-04-21 in session 0efaaf34.
```

**Reason:** 4+ timeouts in 0efaaf34 with `sleep 60 && tail` pattern. User saw repeated failures.

### Planner — default to minimal viable experiment

**File:** `.claude/agents/planner.md`

```diff
+### Default to Minimal Viable Experiment (MVE)
+
+User consistently strips planner overscoping. Observed corrections on 2026-04-21 alone:
+- "I don't want to run the 30-minute pilot"
+- "I don't want to gate arm b on arm a — they could act differently"
+- "I don't think we need a falsification threshold"
+- "Do not drop comedian — that's one of the most interesting case studies"
+- "We don't need a falsifiable hypothesis"
+- "Start with 1 seed. Reuse the 100 persona infrastructure"
+
+**Default posture:**
+- **No pilot** unless user asks. Pilots add latency without information.
+- **No arm gating** unless there is a compute reason — different arms can reveal different things.
+- **No falsification threshold** unless the user requested a pre-registered test.
+- **Keep all conditions the user mentioned by name.** Do not drop "interesting" personas/conditions.
+- **Start at 1 seed.** If the user later wants variance, add seeds — don't preemptively spec 3–5 seeds.
+- **Reuse existing artifacts before generating new ones.** Ask "is there an existing trained model / dataset / vector for this?" before proposing to rebuild.
+
+Before producing the full plan, if scope is unclear, ask ONE clarifying question with 2–3 explicit options. Cheaper than a rewrite.
```

**Reason:** 5 separate /issue sessions today had the user strip plan overscoping. This is the single most repeated correction in the transcripts.

### analyzer.md / implementer.md / experimenter.md — tool-capability fabrication

**File:** `.claude/agents/analyzer.md`, `.claude/agents/implementer.md`, `.claude/agents/experimenter.md`

```diff
+### You CAN write files
+
+You have Write, Edit, and NotebookEdit tools. **Never claim you cannot write a file.** If a write fails, report the actual error message (permission, path, filesystem) — do not self-censor into "I cannot write files, let me print the content in my response".
+
+Triggered on 2026-04-21 in session 9772561a: both the analyzer subagent and the plot-implementer claimed "I cannot write files" — both were fabricating. The parent agent had to write the files itself.
```

**And in `.claude/agents/research-pm.md`:**

```diff
+### Verify subagent capability claims
+
+If a subagent reports "I cannot do X" where X is a standard tool (Write, Edit, Bash, Grep, Read), do NOT believe it and work around. Instead:
+1. Ask the subagent to attempt X once and report the actual error.
+2. If still fails, try X yourself from the parent.
+3. Only then document the real limitation.
+
+Fabrication pattern observed on 2026-04-19, -20, -21.
```

**Reason:** Third occurrence of the same fabrication pattern. Three retros running. Fix it at agent-definition level.

### clean-results skill — multi-seed merging + null framing

**File:** `.claude/skills/clean-results/SKILL.md`

```diff
+### Multi-seed / multi-run / multi-condition consolidation
+
+When the user asks for a clean result across N seeds, N runs, or N conditions, produce **ONE merged clean result**, not N separate entries. The TL;DR should reference the seed/run count (e.g., "across 2 seeds, post-EM alignment collapses to 28.5 ± 0.2"). Per-seed tables belong in the appendix, not as the primary structure.
+
+Only produce per-seed entries if the user explicitly says "separate clean result per seed".
```

**File:** `.claude/skills/clean-results/principles.md` (null framing)

```diff
+### Null framing (MANDATORY)
+
+Do NOT write "confirms the null" or "null result confirmed". Write "indistinguishable from null given the measurement variance" or "noise-limited; we cannot reject the null at this sample size". The distinction matters: our experiments are typically under-powered, so "confirming the null" overclaims the evidence.
+
+This rule is also in auto-memory (`feedback_null_framing.md`) but was violated on 2026-04-21 in session 8b730976. Enforce at the template level.
```

**Reason:** User correction in 8b730976; clean-results skill is shared across subagents which may not see parent-session memory.

### analyzer.md — use the clean-results skill

**File:** `.claude/agents/analyzer.md`

```diff
+### Output format
+
+For user-facing summary output (GitHub issue comments, RESULTS.md entries, final drafts), invoke the `/clean-results` skill and follow its SKILL.md + template.md. Do not invent a new structure. If the skill is unavailable, use `.claude/skills/clean-results/template.md` directly.
+
+User asked explicitly on 2026-04-21: "But does the analyzer use the /clean-result skill?" — it did not; it wrote its own format. Now it must.
```

**Reason:** User explicitly connected the two tools today. Cement the integration.

### Hooks — SessionStart (7th ask)

Same diff as last retro, plus MCP health probe:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "command": "cd /home/thomasjiralerspong/explore-persona-space && echo '=== Branch ===' && git branch --show-current && echo '=== Uncommitted ===' && git status --short | head -5 && echo '=== Latest retro ===' && ls -t research_log/drafts/retrospective-*.md 2>/dev/null | head -1",
        "description": "Show branch, dirty state, latest retro on session start"
      }
    ]
  }
}
```

**Reason:** 7 retros in a row proposing this. "Why are we on this branch?" still recurring.

### Retrospective Agent — file proposals as GH issues (4th ask; now blocking)

**File:** `.claude/agents/retrospective.md`

```diff
+## File remaining proposals as GitHub issues (MANDATORY, 4th ask)
+
+After writing the daily draft:
+1. For each "Proposed Changes" entry, run `gh issue list --label retro-proposal --state open --search <title-keyword>` to check for duplicates.
+2. If new: `gh issue create --title "[retro-proposal] <name>" --body "$(cat <<EOF ... EOF)" --label retro-proposal`.
+3. For each existing open retro-proposal issue, post a one-line comment: "APPLIED (see commit <sha>)", "STILL NOT APPLIED (triggered again in session <id>)", or "OBSOLETE (superseded by <X>)".
+4. At the end of the retro doc, list "Filed N retro-proposal issues: #X, #Y, ..." so the user can see them at a glance.
+
+Prior retros produced ~20 proposals with ~5 applied. The applied ones ALL followed the /issue path. This is not a request — it is the retro's only reliable enforcement mechanism.
```

**Reason:** 4 retros running unapplied. Today applied 0 proposals out of 15 carried over. Retro-markdown-only is a read-once, fire-and-forget channel; the /issue workflow is the only channel that ships.

### Memory Updates

**Update:** `.claude/agent-memory/retrospective/project_unapplied_backlog.md`

- Increment counter from 6 → 7 days (2026-04-15 through -21).
- 0 major proposals applied today (0/15 carry-over + 10 new).
- Move the **retro-as-GH-issue meta-proposal** to top of file with `BLOCKING: yes` — no new proposals should be prioritized until this one ships.

**New feedback memory:** `.claude/agent-memory/retrospective/feedback_sleep_blocking.md`

```markdown
---
name: No sleep-chaining in Bash or ssh_execute
description: Harness blocks `sleep N && <cmd>` in Bash; ssh_execute default 30s times out similar patterns. Use run_in_background / ScheduleWakeup / Monitor / ssh_tail instead.
type: feedback
---

The harness blocks long leading sleeps in Bash ("Blocked: sleep N followed by..."). `mcp__ssh__ssh_execute` has a 30s default timeout, so `sleep 60 && tail log` times out in the MCP path too.

**Why:** (a) Long sleeps burn Anthropic prompt cache (5-min TTL); (b) harness enforces explicit async tools instead; (c) MCP timeouts waste the whole call.

**How to apply:**
- Polling a log/process: `Bash(command="...", run_in_background=true)` + `Read` output later, OR `ScheduleWakeup(delaySeconds=60..270, prompt="/loop resume")`.
- Until-condition: `Monitor` with `until <check>; do sleep 2; done`.
- Remote streaming: `mcp__ssh__ssh_tail` / `mcp__ssh__ssh_monitor` / `ssh_execute(..., timeoutMs=90000+)`.

Triggered 6+ times on 2026-04-21 in sessions 18e61e00 and 9772561a. Observed 4+ ssh_execute timeouts on `sleep 60 && tail` pattern in 0efaaf34 the same day.
```

## Successful Patterns (reinforce these)

- **`/clean-results` skill built end-to-end** — session f3668db1 researched Neel Nanda, Joe Benton, Ethan Perez, James Chua advice and built SKILL.md + checklist.md + principles.md + template.md. Skill was then applied same-day in 81b84545 and 8b730976 to produce 3 clean results. Excellent loop: user pain → skill creation → skill application. Do more of this.
- **`ScheduleWakeup` used correctly in 0efaaf34 (30 times)** — Arm C monitoring via self-paced wake-ups across context compactions. Right pattern, learn to replicate in 18e61e00 and 263b6a69.
- **Context compaction survived cleanly in two sessions** (0efaaf34, 9772561a) — both resumed with clear "Continue from where you left off" behavior and picked up monitoring state.
- **Adversarial-planner → /issue shipping** — issues #62, #69, #70 went through the full pipeline (gate-keeper → planner → critic → revise → approve → experimenter). Code-reviewer caught issues ("CONCERNS" verdicts on both #51 and #62 led to small revisions).
- **User's scope overrides were respected immediately** — "Advance to planning. Incorporate 2 and 3 but we don't need a falsifiable hypothesis" was acted on without argument. Same for "Start with 1 seed".
- **Merged clean-result for 2 seeds** (8b730976) was produced correctly after the one-turn correction and captured a meaningful observation (seed-137 good_correct capability 0.676 vs seed-42 8-GPU 0.887 — real seed variance worth flagging).
- **clean-results entry for single-[ZLT]-token sweep** (81b84545) included the heatmap user asked for (LR × epochs × source/assistant/bystander) with WandB links to the actual runs — the "describe setup + link to data + show plot" pattern worked.

## Metrics

- **Real user messages:** ~120 across 14 sessions (shorter-than-usual day)
- **Most friction-heavy session:** 18e61e00 (carried over from prior day, 486 Bash calls, 38 tool errors, 17 ScheduleWakeups, 5 context compactions)
- **Most efficient /issue session:** 6bc50607 (issue #51, 5 user turns to approve) and ef1a1e23 (issue #62, 2 user turns to approve)
- **Agent dispatches:** ~55 across sessions (planner 7, experimenter 11, critic 5, gate-keeper 5, code-reviewer 4, implementer 4, analyzer 3, reviewer 3, Explore 6)
- **Commits:** 15 (10 on main, 5 on issue branches)
- **GH issues processed end-to-end today:** #51, #54, #62 merged; #61, #69, #70 in progress
- **New skills shipped:** 1 (`clean-results`)
- **New clean results produced:** 3 (100-persona leakage, single-token sweep heatmap, Aim 5 seeds 42+137 matrix)
- **User corrections on scope/plan:** ~12 (all in planner/clarifier phase — "don't pilot", "don't gate", "don't threshold", "don't drop", "reuse existing", "1 seed")
- **Bash sleep-blocked events:** 6+ (18e61e00 ×4, 9772561a ×2)
- **SSH execute timeouts on sleep+tail:** 4+ (0efaaf34)
- **Agent file-write fabrications:** 2 (9772561a — analyzer + implementer)
- **Retro proposal carry-over:** 15 unapplied proposals, 7 days running

---

**Meta-observation:** The single highest-leverage change is still the same one proposed 4 days ago — **file retro proposals as GitHub issues with `retro-proposal` label so they enter the /issue workflow**. Today 10 commits shipped via /issue, 0 config/hook/agent.md changes shipped via retro drafts. The mechanism is clear and the data is now 7 days long. Every other deferred item is downstream of this one.
