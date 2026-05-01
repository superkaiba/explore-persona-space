# Daily Retrospective — 2026-04-20

**Sessions reviewed:** 20 top-level + 7 subdirs of subagents (~100+ JSONL files)
**Total real user messages:** 194 across 20 sessions
**Agent dispatches:** 82 total (Explore 16, planner 14, experimenter 11, implementer 9, critic 9, gate-keeper 8, code-reviewer 6)
**Commits today:** 22 (`main`) — a very productive day

## Summary

Today was a **backlog-clearing day**. The single biggest recurring friction from the last 5 retros — "HF Hub uploads and WandB logging aren't built into the main scripts" — finally got addressed in commit `a0028d5` ("Standardize HF uploads and WandB results logging (#49)") plus `7e97961`, `4e768ff`, `b58b777`. Two GitHub issues (#49 midtraining upload/logging, #55 preflight for pipelines) were shipped via the adversarial-planner → /issue workflow. **However, three hard recurrences:** (1) SSH MCP broken all day (user asked "can you use the ssh MCP?" 5+ times across 5 different sessions — MCP restart doesn't fix), (2) "Check progress periodically" asked 15+ times in a single session (18e61e00) with no scheduled/cron monitoring in place despite `/schedule` skill existing, (3) the `Leakage` callback in #51 was built with `model.generate()` (HF sequential), violating the explicit CLAUDE.md critical rule "Always use vLLM for generation."

## Proposal Backlog Audit (read this first)

Cumulative unapplied proposals from 2026-04-15 through -19. Items marked ✅ = applied in last 24h, ⏳ = partially done, ❌ = still missing.

| Proposal | First proposed | Status | Evidence from today |
|---|---|---|---|
| HF Hub upload verification / default in main scripts | 2026-04-16, -19 | ✅ **APPLIED** | Commits `a0028d5`, `7e97961`, `4e768ff` — the #1 complaint for 5 days running is now closed |
| Remove/clean make-evil-dumb references in active scripts | newly surfaced | ✅ **APPLIED same day** | Commit `b58b777` after user asked "Why is the script in make-evil-dumb" |
| Preflight for long pipelines | surfaced today | ✅ **APPLIED** | Commit `728ccc6` — `preflight --pipeline-check` + integration tests |
| Periodic eval callbacks configurable | surfaced today | ✅ **APPLIED** | Commit `f9905ef` — `periodic_eval` config system with capability/alignment/leakage callbacks |
| Adversarial-planner → /issue pipeline producing PRs | ongoing | ✅ **WORKING WELL** | 4 issues shipped today (#49, #50, #51, #55), all approved via adversarial-planner + code-reviewer |
| SessionStart hook (branch + fleet health + retro-status) | 2026-04-15 (6th ask) | ❌ | Still not applied. User had to ask "Why are we on a feature branch?" again |
| Convert retro proposals to `retro-proposal`-labelled GH issues | 2026-04-16, -19 | ❌ | Not applied. This remains the most-proposed meta-change |
| PostToolUse hook on `git push` for pod-sync | 2026-04-15 | ❌ | Env-sync-across-pods got a dedicated session (ebf05007) instead |
| Fleet watchdog cron (stale runs) | 2026-04-19 | ❌ | User asked for it explicitly today: "Setup monitoring to ping you to check every 1h" — still no cron |
| Phantom-tool `SendMessage` gotcha in CLAUDE.md | 2026-04-19 | ❌ | Not triggered today but still not documented |
| Subagent-fabricated-limits verification | 2026-04-19 | ❌ | Triggered again today: "Why couldn't analyzer write files?" in 9772561a |
| Shallow-status-answer anti-pattern | 2026-04-19 | ❌ | Triggered again in 18e61e00: user kept asking "are models uploaded?", "do you have HF token?", "why feature branch?" |
| Loss-masking glossary (marker-only vs tail-N vs full-assistant) | 2026-04-19 | ❌ | Triggered again on turn 1 of 9772561a: "is this single-token?" → "I mean finetuning only on [ZLT] at the end" |
| Experimenter results-marker provenance | 2026-04-19 | ❌ | Unclear if triggered today; no obvious silent-skip |
| research-pm.md pre-dispatch checklist (branch + pod health) | 2026-04-18, -19 (3rd ask) | ❌ | Triggered today: pod 5 was silently slower than others; no pre-dispatch health compare |

**Net score:** 5 MAJOR proposals applied (all infrastructure/workflow items), 10 still deferred (all hook/memory/agent-definition items). The pattern now: **code-shaped proposals ship via /issue, config-shaped proposals (hooks / agent.md edits) do not.** Consider filing the remaining deferred items as GitHub issues with `label:retro-proposal` to unblock them through the same pipeline that actually works.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---|---|---|---|
| **SSH MCP broken / unreliable all day** — "Can you use the SSH MCP?" asked in 5 separate sessions (db6184a7, cae4ccbd, 4a0a9e7f, f79e9e93, 3 more); restart doesn't fix; user suspects Happy Coder session context is the cause | db6184a7, cae4ccbd, 4a0a9e7f, f79e9e93 | Root-cause the MCP restart failure — likely Happy Coder session lacks env needed by the MCP launcher. Document fix in CLAUDE.md SSH MCP section. Consider auto-health-check hook that preemptively verifies `ssh_list_servers` on SessionStart and emits loud warning if it fails | `CLAUDE.md`, `.claude/mcp.json`, `.claude/settings.json` hook |
| **"Check progress periodically" x15 in one session with no cron** — user explicitly asked "Setup monitoring to ping you to check every 1h" in 18e61e00 but no cron was created | 18e61e00 | When user asks for periodic monitoring, invoke `/schedule` skill to create a CronCreate trigger. Document this pattern in research-pm.md | `.claude/agents/research-pm.md` |
| **Leakage callback built with HF generate, violating "Always use vLLM" CLAUDE.md rule** — user caught it mid-review: "What is leakage callback uses HF generate" → "How much slower is it than vLLM?" | 6bc50607 | Add an adversarial-planner / critic check: "Does this implementation use HF .generate() anywhere in the eval path? If yes, justify or switch to vLLM per CLAUDE.md." | `.claude/agents/critic.md`, `.claude/agents/code-reviewer.md` |
| **Script placed in reference repo (`make-evil-dumb`) instead of project** — user caught it: "Why is the script in make-evil-dumb" | 45e45cc5 | Add to experimenter.md: "Scripts for our experiments belong in `scripts/` or `src/explore_persona_space/`. Never create new scripts inside `external/` or `/workspace/make-evil-dumb/` — those are reference codebases only." | `.claude/agents/experimenter.md` |
| **"Did you test the code?" / "Did you test the changes?"** — recurring pattern, asked in 45e45cc5 and cff43826 | 45e45cc5, cff43826 | Implementer must run tests before marking work complete. Add explicit checklist item to implementer.md results format | `.claude/agents/implementer.md` |
| **Merge does not auto-push** — "Does it automatically push after you merge or do i have to manually say to do that" → "Always push after merge. Save that as part of the workflow" | 01c214a3 | Add explicit workflow rule: after any `git merge` / PR merge, immediately `git push origin main`. Record in CLAUDE.md | `CLAUDE.md` |
| **Branch surprise** — "Why are we on a feature branch?" (3rd retro in a row) | 18e61e00 | SessionStart hook showing current branch (still unapplied from 2026-04-15) | `.claude/settings.json` |
| **Subagent "cannot write files" fabrication recurred** — "Why couldn't analyzer write files?" | 9772561a | Verify subagent capability claims before believing (proposed 2026-04-19, still unapplied) | `.claude/agents/research-pm.md` |
| **Context compaction happened 7+ times in one day** — several sessions hit `<context-continued>` markers; 18e61e00 compacted 5 times | 18e61e00, 9772561a, 0efaaf34, cff43826 | Long-running monitoring sessions should delegate to a monitor-agent/cron instead of accumulating context in the main session. Flag: when a session hits 80% context, proactively spawn a resume-plan subagent and restart clean | `.claude/agents/research-pm.md`; consider auto-compaction trigger |
| **Scope echo still missing** — turn 1 of 18e61e00: "run extra seed of midtraining" → Claude ran extra EM seed (wrong). Required user correction: "No the 10 seeds are just the EM step but I want an extra seed of the midtraining/post training step" | 18e61e00 | Scope-echo proposal from 2026-04-19 — still unapplied | `.claude/agents/research-pm.md` |

## Failed Approaches (document to prevent retries)

- **Long-running monitor session in a single Claude conversation**: 18e61e00 accumulated 45 user turns and 5 context-compactions, mostly "Check progress periodically" polling. This burns enormous cost and repeatedly loses state across compactions. **Correct pattern:** use `/schedule` or ScheduleWakeup to create a cron trigger that fires every N minutes with a self-contained prompt. Document in research-pm.md.
- **Restarting SSH MCP when it fails**: User reports "mcp restart doesnt work" in cae4ccbd. Restart is not the fix. Root cause must be investigated and documented (likely Happy Coder / tmux env-var inheritance issue).
- **Using HF `.generate()` in periodic callbacks**: Violates CLAUDE.md. The Leakage callback was shipped with HF generate before user caught it. Add critic check.
- **Pointing an experimenter at `/workspace/make-evil-dumb/`**: That directory is a reference codebase (pinned in external/). Creating new scripts there is wrong; they get lost on cleanup and confuse the research-pm.
- **Starting an experiment without checking whether the target pod's environment is sync'd**: Dedicated session (ebf05007) today; user wants a cleaner solution than manual sync.

## Proposed Changes

### CLAUDE.md — push-after-merge workflow rule

**File:** `CLAUDE.md` (in "## Code Style" section, after the existing `git pull` reminder)

```diff
 ## Code Style

 - **All code changes on local VM, never on pods.** [...]
+- **Always push after merge.** After `git merge <branch>` or a PR merge via `gh pr merge`, immediately run `git push origin main`. Never leave a merged state unpushed — pods pulling from origin will miss the merge. User explicitly confirmed this workflow on 2026-04-20: "Always push after merge. Save that as part of the workflow."
 - **Linting:** `uv run ruff check . && uv run ruff format .`
```

**Reason:** Session 01c214a3 — user had to ask, then explicitly saved it as a rule.

### CLAUDE.md — make-evil-dumb is reference-only

**File:** `CLAUDE.md` (in "## Directory Structure" section)

```diff
 external/                     # Reference codebases (open-instruct, agentic-backdoor, training-against-misalignment)
+
+**DO NOT create new scripts or modify files in `external/` or `/workspace/make-evil-dumb/`.** Those are reference codebases pinned for reproducibility. All new scripts belong in `scripts/` or `src/explore_persona_space/`. If you need a variant of a reference script, copy it into `scripts/` with a descriptive name (e.g., `scripts/run_midtrain_evil_correct_seed137.sh`).
```

**Reason:** Session 45e45cc5 — subagent placed midtrain script in `/workspace/make-evil-dumb/` on a pod; user caught it.

### CLAUDE.md — loud callout that SSH MCP may be broken

**File:** `CLAUDE.md` (start of "## Remote Pod Access (SSH MCP)" section)

```diff
 ## Remote Pod Access (SSH MCP)

+**KNOWN ISSUE (2026-04-20):** The SSH MCP server can enter a broken state where `ssh_list_servers` returns stale data or fails entirely, and `/mcp` restart does NOT resolve it. Root cause under investigation (likely Happy Coder session environment). **If SSH MCP fails twice in a row, fall back to `Bash("ssh podN ...")` — do not loop on the MCP restart.**
+
 An SSH MCP server (`mcp-ssh-manager`) is configured with all 5 RunPod GPU pods. **Always prefer SSH MCP tools over `Bash("ssh podN ...")`** for remote operations.
```

**Reason:** 5 separate sessions today asked "can you use the SSH MCP" — restart loop does not work. Stop the loop.

### Agent Definitions — critic.md / code-reviewer.md: add "Does this violate a CLAUDE.md Critical Rule?" check

**File:** `.claude/agents/critic.md` and `.claude/agents/code-reviewer.md`

```diff
+### Critical-Rule Audit (MANDATORY)
+
+Before finalizing your review, explicitly check each item from CLAUDE.md "## Critical Rules":
+- Does this plan/code use HF `.generate()` for eval or generation? (violates "Always use vLLM")
+- Does this plan/code introduce silent error suppression (`try/except: pass`, `--force`, `--no-verify`)?
+- Does this plan/code make factual claims about APIs/layer numbers without verification? (violates "List assumptions before implementing")
+- Does this plan reinvent something that exists in PyPI/HF/GitHub? (violates "Search before building")
+
+If any violation exists, FLAG IT in the top findings. Do not bury it.
```

**Reason:** Issue #51 Leakage callback was shipped using HF generate, directly violating the CLAUDE.md "Always use vLLM" rule. Critic/code-reviewer did not flag it. User caught it at merge time. A mandatory critical-rule pass in the review layer would have caught this.

### Agent Definitions — research-pm.md: use /schedule for periodic monitoring

**File:** `.claude/agents/research-pm.md` (after dispatch section)

```diff
+### Periodic monitoring requests
+
+When the user asks for periodic progress checks (e.g., "check every hour", "monitor periodically", "ping me every 30 min"), DO NOT loop inside a single Claude session. Instead:
+
+1. Invoke `/schedule` to create a CronCreate trigger.
+2. The trigger prompt should be self-contained: "Check pod N for experiment <name>. Report: GPU util, last log line timestamp, upload state. If experiment is done or stalled >15min, alert."
+3. For once-off delayed monitoring ("check in 1h"), use ScheduleWakeup instead.
+
+Why this matters: polling in a single session accumulates context, burns cache, and repeatedly compacts. Session 18e61e00 on 2026-04-20 had 45 turns of this pattern across 5 context compactions — a cron trigger would have done the same work at ~1% of the cost.
```

**Reason:** Session 18e61e00 had 15+ "Check progress periodically" turns and 5 context-compactions. User even explicitly said "Setup monitoring to ping you to check every 1h" — the `/schedule` skill exists but was not invoked.

### Agent Definitions — experimenter.md: reference-repo blocklist

**File:** `.claude/agents/experimenter.md`

```diff
+### Where to place scripts (MANDATORY)
+
+- New experiment scripts go in `scripts/` on the local VM (commit, push, then pull on pod).
+- Library code goes in `src/explore_persona_space/`.
+- **NEVER** create files in `/workspace/make-evil-dumb/`, `external/`, or any other reference-codebase directory. Those are pinned references.
+- If you must adapt a reference-repo script, copy it into `scripts/` with a new name; don't edit it in place.
```

**Reason:** Session 45e45cc5 caught this. Already partially addressed by commit `b58b777` (removing references) but the rule should also live in experimenter.md.

### Agent Definitions — implementer.md: test-before-complete

**File:** `.claude/agents/implementer.md` (in results format)

```diff
+### Results marker MUST include test evidence
+
+Before posting `<!-- impl:complete -->`, verify:
+- [ ] Lint passes: `uv run ruff check . && uv run ruff format --check .`
+- [ ] Tests pass: `uv run pytest tests/` (or the relevant subset)
+- [ ] If GPU code: at least one smoke test on a real pod (experimenter can handle this handoff)
+
+Include the test command and last-line output in the results marker. If you skipped tests, explicitly say so and justify — don't claim completion silently.
```

**Reason:** "Did you test the code" / "Did you test the changes?" asked twice today (45e45cc5, cff43826). Pattern from prior retros still recurring.

### Hooks — SessionStart branch + MCP-health check (6th ask)

**Proposed hook:** `SessionStart`

```json
{
  "hooks": {
    "SessionStart": [
      {
        "command": "cd /home/thomasjiralerspong/explore-persona-space && echo '=== Branch ===' && git branch --show-current && echo '=== Uncommitted ===' && git status --short | head -5 && echo '=== Last retro ===' && ls -t research_log/drafts/retrospective-*.md 2>/dev/null | head -1 && echo '=== MCP health ===' && timeout 5 python scripts/pod.py health --quick 2>&1 | head -10 || echo 'MCP health check failed — SSH MCP may be broken'",
        "description": "Show branch, MCP health, retro on session start"
      }
    ]
  }
}
```

**Reason:** 6th consecutive retro proposing this. Today: "Why are we on a feature branch?" + 5 sessions asking "can you use the SSH MCP?" — a SessionStart probe catches both.

### Hooks — fleet watchdog (2nd ask)

**Proposed trigger:** CronCreate (via `/schedule` skill) every 30 min

```
Prompt: "Check all 5 pods via SSH MCP. For each pod with a running Python process in /workspace/midtrain_* or /workspace/persona_leakage_*, verify: (a) nvidia-smi shows >0% GPU util, (b) the process's log file was modified in the last 10 min. If both fail, post a GitHub comment on the tracking issue saying 'STALLED'."
```

**Reason:** User explicitly said "Setup monitoring to ping you to check every 1h" today. `/schedule` exists but wasn't used. This is a 5-minute infrastructure improvement.

### Retrospective Agent — file proposals as GH issues (3rd ask)

**File:** `.claude/agents/retrospective.md`

```diff
 Write to `research_log/drafts/retrospective-YYYY-MM-DD.md`:

+After writing the draft:
+1. For each proposal in "Proposed Changes," check `gh issue list --label retro-proposal --state open` to see if it's already open.
+2. If new: `gh issue create --title "[retro-proposal] <name>" --body <diff + reason> --label retro-proposal`.
+3. On yesterday's open retro-proposal issues, post a one-line status comment: "APPLIED (see <commit>)", "STILL NOT APPLIED (triggered again in <session>)", or "OBSOLETE (superseded by <X>)".
+4. DO NOT duplicate issues.
+
+Five-day-plus-running retros have produced 15+ proposals with only ~5 applied. Converting to GH issues puts them in the user's `/issue` workflow, which is the only pipeline that actually ships changes.
```

**Reason:** 3rd consecutive retro proposing this. Today's evidence: 5 infrastructure-code proposals applied (all of which were code changes that fit /issue workflow) vs. 10 config/hook/agent.md proposals not applied. The mechanism matters.

### Memory Updates

**Update:** `.claude/agent-memory/retrospective/project_unapplied_backlog.md`

- Increment counter from 5 → 6 days (2026-04-15 through -20).
- Mark `HF Hub upload verification` as ✅ APPLIED (commit `a0028d5`).
- Mark `Preflight for long pipelines` as ✅ APPLIED (commit `728ccc6`).
- Add new items: SSH MCP root-cause, push-after-merge, make-evil-dumb block, critic critical-rule audit, /schedule for periodic monitoring, test-before-complete in implementer.
- **Trigger reached:** counter is now at 6 — per the prior retro's rule, the next retro should stop proposing new items and instead file ONE blocker issue summarizing the deferred list.

**New file:** `.claude/agent-memory/retrospective/feedback_applied_via_issues.md`

```markdown
---
name: Proposals Ship via /issue, Not via Retro Drafts
description: Observed pattern — retro proposals that match the code-change shape get applied via /issue + adversarial-planner; config/hook/agent.md proposals sit unapplied.
type: feedback
---

Observed 2026-04-20: today's retro had 5 MAJOR proposals applied, all code-shaped (HF upload, preflight, callbacks, make-evil-dumb cleanup, test integration). Zero hook/agent-definition proposals applied despite being carried across 5 retros.

**Why:** The `/issue` + adversarial-planner workflow has gravity — it ships. Retro-draft markdown in `research_log/drafts/` does not get re-read. The user only applies changes that enter their standard issue-driven pipeline.

**How to apply:** The retrospective agent should file EACH proposal as a `retro-proposal`-labelled GitHub issue. This is the 3rd retro proposing this meta-change; should be blocking the next one.
```

## Successful Patterns (reinforce these)

- **HF Hub + WandB upload now default in main scripts** (commit `a0028d5`) — 5-day-running user complaint finally addressed. This is the textbook example of a retro proposal getting applied because it was code-shaped and went through `/issue`.
- **Adversarial-planner → /issue pipeline shipped 4 issues today** (#49, #50, #51, #55) — all with planner + critic + code-reviewer passes, all merged to main. The workflow is working.
- **`scope clarification on turn 1`** happened quickly in some sessions — e.g., cff43826 "Are you also including generic pretraining text?" → "Look online for what ratio to use and incorporate" — back-and-forth resolved in 2 turns.
- **Preflight with `--pipeline-check` flag** (commit `728ccc6`) added 5-minute integration tests that would catch env-sync bugs earlier. Good proactive fix.
- **User's "Override process — plan the implementation directly"** was respected — adversarial planner is not applied blindly; user can override when scope is clear.
- **GPU assignment bug fixed same-day** (commits `46f50d8`, `a63fc98`) after detection — tight loop between bug surface and fix.
- **Subagent dispatch count (82) is healthy and specialization-diverse** — Explore (16), planner (14), experimenter (11), implementer (9), critic (9), gate-keeper (8), code-reviewer (6). The manager → specialist workflow is the dominant mode.

## Metrics

- **Real user messages:** 194 across 20 sessions
- **Most friction-heavy session:** 18e61e00 (45 messages, 5 context compactions, 15+ "Check progress periodically" turns)
- **Agent dispatches:** 82 (specialization healthy: planner 14, experimenter 11, implementer 9, critic 9, gate-keeper 8, code-reviewer 6, analyzer 2, reviewer 2)
- **Commits to main:** 22
- **GH issues shipped end-to-end via /issue:** 4 (#49, #50, #51, #55) + PR #58 merged
- **Explicit user corrections today:** ~15 (scope echoes: "no the 10 seeds are just EM", vLLM violation caught at merge, script in make-evil-dumb, upload not by default, etc.)
- **SSH MCP failures:** 5+ user-initiated "can you use the SSH MCP" across 5 sessions
- **Context compactions:** 7+ across 4 sessions (18e61e00 ×5, 9772561a, 0efaaf34, cff43826)
- **Tool-result errors flagged:** 17/20 sessions had ≥1 error (OOM 10, file-not-found 11, traceback 9, permission 5, module-not-found 4)
- **Retro proposal carry-over:** 10 unapplied proposals from prior retros (6th day running)

---

**Meta-observation:** Today was the **best retrospective-application day of the week** — the top unapplied proposal (HF Hub + WandB default in main scripts) finally shipped. But the mechanism that made it ship was the user manually filing issue #49, not the retro. Every applied proposal today followed that path: user sees pain → user files issue → adversarial-planner runs → PR merges. The retro drafts were read-once and filed. **The single-highest-ROI action for 2026-04-21 is to auto-file the deferred retro proposals as `retro-proposal`-labelled GitHub issues** — this is the one meta-change that would turn the retro system from read-only advisory to an actual feedback loop.
