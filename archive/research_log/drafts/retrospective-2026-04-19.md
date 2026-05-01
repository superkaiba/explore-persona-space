# Daily Retrospective — 2026-04-19

**Sessions reviewed:** 11 top-level + ~15 subagent = ~26 JSONL files
**Total real user messages:** ~62 (across 11 sessions)

## Summary

Productive day: `opus[xhigh]` migration landed, persona-leakage single-token sweep designed and launched, Tier 1 perf optimization landed on issue #36 with adversarial review, third-seed midtrain replication restarted across 5 pods, and the effort default was flipped `max → xhigh` based on the Opus 4.7 integration. The core failure modes today are **not new** — the same patterns flagged on 2026-04-15 through -18 recurred again, because **the proposal backlog is now 5 days old with zero items applied**. New patterns this run: (a) subagents fabricating limits ("can't write files") that the parent trusts without verification, (b) 16-hour silent OOM-failed run on pod1 with no alert, (c) the tulu_control branch-drift continuing for a 2nd session, (d) phantom tool calls (`SendMessage` — does not exist in this harness).

## Proposal Backlog Audit (read this first)

Cumulative unapplied proposals from 2026-04-15, -16, -17, -18, now -19. Items marked ✅ = applied in last 24h, ⏳ = partially done, ❌ = still missing.

| Proposal | First proposed | Status | Evidence from today |
|---|---|---|---|
| ZeRO-2 NaN coupling SFT gotcha in CLAUDE.md | 2026-04-15 | ❌ | Didn't trigger today; monitoring lucky streak |
| SessionStart hook showing git branch + retro status | 2026-04-15 | ❌ | caf9c534: `tulu_control` still on old feature branch, 2nd recurrence |
| PostToolUse hook on `git push` for pod-sync reminder | 2026-04-15 | ❌ | caf9c534: uv/liger out-of-sync across 5 pods |
| RunPod port instability gotcha in CLAUDE.md | 2026-04-15 | ❌ | Not triggered today |
| HF Hub upload verification step in experimenter.md | 2026-04-16 | ❌ | 9772561a: run marked done without verify |
| PreToolUse hook blocking `cat`/`grep`/`find` | 2026-04-16 | ❌ | Still used in Bash today |
| Per-pod quirks consolidated (experimenter.md) | 2026-04-16 | ❌ | Pod env drift recurred today (caf9c534) |
| AST-over-import-probe rule in agent defs | 2026-04-17 | ❌ | Not triggered today |
| Decision-log for failed library API assumptions | 2026-04-17 | ❌ | Packing `+293%` artifact already burned a day of work |
| Sleep-blocking explanation in agent defs | 2026-04-17 | ❌ | Several sleeps still attempted |
| experimenter.md pre-launch (ZeRO-3 default, `set -a; source .env`, push_to_hub=true) | 2026-04-18 | ❌ | caf9c534: `uv not found` (PATH issue), 3x pods |
| research-pm.md dispatch-checklist (branch + pod health) | 2026-04-18 | ❌ | caf9c534: pod health not checked before launch |
| planner.md decide-then-list | 2026-04-18 | ❌ | 9772561a: 3 rounds of clarification on single-token loss |
| retrospective.md "lead with backlog audit" | 2026-04-18 | ⏳ | This retro is doing it, but no structural change to agent def |
| Pod env quirks memory file | 2026-04-18 | ❌ | Not created |

**Conclusion:** This is 5 days running. The retrospective loop is a read-only advisory layer. Recommend converting to GitHub issues with `label:retro-proposal` so they have a tracking surface — see "Proposed Changes → Retrospective Agent" below.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---|---|---|---|
| **Subagent fabricates a capability limit, parent trusts it** — experimenter claimed "can't write files" twice; 3rd attempt inline succeeded trivially | 9772561a | When a subagent reports an infrastructure limit that contradicts its agent definition, verify directly before re-dispatching or falling back | `.claude/agents/research-pm.md`; new memory |
| **Shallow-search / partial-answer on status questions** — "are there recent multi-seed midtrain experiments?" → first answer missed issue #32 entirely; user forced "search more deeply" | caf9c534, 9772561a | Status questions must query: (a) GitHub issue markers, (b) pod processes via SSH MCP, (c) eval_results/ — not just local filesystem | `.claude/agents/research-pm.md` |
| **Silent fleet failures — 16h OOM sat undetected** — pod1 tulu_control seed 137 exit 137 on 2026-04-18 00:37 was only noticed when user asked "is this running?" | caf9c534 | Fleet watchdog: cron or retro trigger that flags pods idle >6h after a running experiment | `.claude/settings.json` hooks or `scripts/pod.py health --stale-runs` |
| **Pod env sync drift on ALL pods simultaneously** — uv PATH broken, liger_kernel version mismatch, import traceback on 2 pods | caf9c534 | Run `pod.py health --quick` before every launch; fail fast with actionable output | `.claude/agents/research-pm.md` pre-dispatch checklist |
| **Phantom tool call `SendMessage` does not exist** — assistant tried to redirect a completed subagent via it | 9772561a | CLAUDE.md note: "To continue a subagent, re-dispatch with its agent ID as the `to` field in a new Agent call. `SendMessage` is not a tool — do not call it." | `CLAUDE.md` |
| **Scope mis-read on turn 1 of ambiguous briefs** — "extra seed" interpreted as EM-only instead of full midtrain pipeline (1 turn of correction) | 18e61e00 | In `research-pm.md`: before dispatching, echo back scope assumptions as a 2-line "I understand this to mean..." for any compute-costly task | `.claude/agents/research-pm.md` |
| **Re-explanation tax on domain terms** — 3 rounds on marker-position vs tail-window loss | 9772561a | Add a "Loss-Masking Glossary" to CLAUDE.md: marker-token-only vs tail-N-token vs full-assistant masking | `CLAUDE.md` |
| **Subagent silently skipped the benchmark step** — experimenter pushed 7 commits but didn't run baseline benchmarks. Only caught when user ran status query | 37d95c4c | Experimenter results marker MUST include raw result file paths; research-pm verifies file existence before posting summary | `.claude/agents/experimenter.md`, `.claude/skills/issue/` |
| **Reactive vs proactive status** — user had to ask 4+ times in 18e61e00 ("are models uploaded?", "HF token?", "why feature branch?", "how long does a run take?") | 18e61e00, caf9c534 | On every status check, research-pm returns a 4-line snapshot: GPU usage, last 20 log-lines summary, upload state, branch | `.claude/agents/research-pm.md` |

## Failed Approaches (document to prevent retries)

- **Trusting a subagent's self-reported capability limit**: In 9772561a the experimenter claimed "can't write files" — this was wrong; when the parent gave up and ran the work inline, writes worked fine. Document: "Subagents misdescribe their own limits. When the reported limit contradicts the agent's definition, verify directly (write a test file via SSH MCP) before accepting the claim." Aligns with existing `feedback_ast_over_import_probes.md` / `feedback_verify_agent_infrastructure_claims.md`.
- **Using `SendMessage` tool to redirect a completed subagent**: Does not exist. Correct pattern: either spawn a new Agent with `subagent_type` + the agent ID as `to`, or re-dispatch a fresh agent with self-contained brief. Add to CLAUDE.md "Prompting & Effort Levels" section.
- **First-pass "integrate into existing docs" interpreted as "add a new labeled section"**: In 1a398de9 the user had to clarify twice. When told "integrate into X," rewrite in-place; do not add a new marked section.
- **Inline Python one-liners for state probes**: In 3f3f75c2, multiple inline one-liners failed (`'str' object has no attribute 'get'`, `ModuleNotFoundError: No module named 'transformers'`, `list indices must be integers`). Use existing scripts or the SSH MCP wrappers; do not attempt complex state queries inline.
- **Experimenter skipping the benchmark step but reporting success**: On 37d95c4c, experimenter a121f4cf pushed 7 commits without running baseline A/B. Caught only via user-initiated status query. The `epm:results` marker must require file paths to raw outputs, and research-pm must verify those paths exist.

## Proposed Changes

### CLAUDE.md — add today's new gotchas

**File:** `CLAUDE.md` (section "## Gotchas / Known Issues", after line 342)

```diff
 ## Gotchas / Known Issues

 - **HF Trainer monkey-patch** in `src/explore_persona_space/train/trainer.py` — fragile, will break if Trainer.__init__ changes
 - **Hard-coded library paths** in `orchestrate/env.py` — cluster-specific
 - **No dataset validation** in `build_phase1_dataset()` — empty QA pairs create silent failures
 - **Tulu pipeline caveat:** midtraining+Tulu results may not generalize to production post-training
+- **Phantom tool `SendMessage`**: does NOT exist in this harness. To continue a subagent, use the Agent tool with `subagent_type` and the agent's ID as the `to` field — this resumes it with full context. A fresh Agent call starts a NEW subagent with no memory.
+- **Subagents misreport their own limits.** If a subagent says "I can't write files / read files / run Bash," that usually contradicts its agent definition. Verify with a direct SSH test BEFORE re-dispatching or falling back to inline work.
+- **Silent pod failures are common and not auto-detected.** A run that exited with code 137 (OOM) or code 1 will sit idle indefinitely with no alert. Before assuming a pod is "making progress," check: (a) `nvidia-smi` for GPU util, (b) `tail -n 50 $LOGFILE` for recent activity timestamp, (c) `ps aux | grep python | grep -v grep`.
+- **Shallow status-answer anti-pattern**: for any question about "is X running / did Y finish / what's the state of Z," never answer from local filesystem alone. Query (1) `gh issue view <N> --json comments`, (2) each pod via SSH MCP `ssh_execute`, (3) `eval_results/` and WandB. All three.
```

**Reason:** Five new failure modes today that aren't in any gotchas list. Each cost 5-30 min of debugging / clarification.

### CLAUDE.md — Loss-Masking Glossary

**File:** `CLAUDE.md` (new subsection under "## Project Overview" or "## Architecture Notes")

```diff
+### Loss-Masking Glossary (persona-leakage experiments)
+
+Different leakage experiments use DIFFERENT masking schemes. Do not conflate them:
+
+- **Full-assistant masking**: Loss on every assistant token (default SFT).
+- **Tail-N-token masking**: Loss only on the final N tokens of the assistant response. `tail_tokens=N` in config. Used for "marker+context" experiments.
+- **Marker-only masking**: Loss on only the literal marker token(s) at end of assistant (e.g., single `[ZLT]` token). Specifically `tail_tokens=1` with marker as last token. Used for "pure marker coupling" experiments.
+
+When the user says "finetune on the marker," ASK whether they mean tail-N or marker-only — these give different results.
```

**Reason:** 3 rounds of back-and-forth in session 9772561a. Glossary makes the distinction explicit.

### Agent Definitions — research-pm.md pre-dispatch checklist

**File:** `.claude/agents/research-pm.md`

```diff
 ## When dispatching a specialist

+### Pre-dispatch checklist (MANDATORY)
+
+Before spawning ANY specialist agent on a compute task, run in parallel:
+1. `git branch --show-current` — confirm on `main` or user-approved feature branch
+2. `git status` — confirm no uncommitted drafts / RESULTS.md that other agents would lose
+3. `python scripts/pod.py health --quick` — confirm target pod is reachable, GPUs idle, uv available
+4. `gh issue view <N> --json comments` — confirm no in-flight work on this issue from a prior session
+
+If any fails, fix or ask user BEFORE dispatching.
+
+### Scope echo (MANDATORY for any compute-costly task)
+
+Before dispatching, echo the scope assumption back to the user in 1-2 lines:
+> "I understand this to mean: <echo>. Dispatching? (y/n or correct)"
+
+Skip the echo only for (a) re-runs of a prior experiment with seed/condition changes, (b) explicit dispatch commands like "run the queue."
+
+### Subagent capability-limit verification
+
+If a subagent reports an infrastructure limit that contradicts its agent definition ("can't write files," "no SSH access," "missing tool"), DO NOT accept at face value. Verify with a direct SSH MCP test (e.g., `ssh_execute <pod> "touch /tmp/write_test"`). Only fall back to inline work if the limit is verified.

 ## Status snapshot format

+When asked "what's running / any progress / is X done," return a 4-line snapshot in this order:
+```
+GPU: <pod1 0.0%, pod2 85%, pod3 92%, ...>
+Processes: <N python processes, last log line timestamp>
+Uploads: <N models on HF Hub / N expected, last WandB artifact>
+Branch: <git branch --show-current>
+```
+
+Do not summarize from local filesystem alone. Query the pods.
```

**Reason:** The user had to ask status 4+ times today in 18e61e00/caf9c534 and explicitly demanded "search more deeply." A structured snapshot + mandatory multi-source query fixes this.

### Agent Definitions — experimenter.md raw-output provenance

**File:** `.claude/agents/experimenter.md`

```diff
 ## Posting results markers

+### Raw-output provenance (MANDATORY)
+
+The `<!-- epm:results v1 -->` marker MUST include:
+- Absolute path(s) to raw output JSON/logs on the pod
+- SHA256 or byte size of the result file (so research-pm can verify it exists)
+- The exact command line used to produce it
+
+If you cannot produce these, post `<!-- epm:failure v1 -->` or `<!-- epm:progress v2 -->` instead. Do NOT claim completion without the artifacts.
+
+Today (2026-04-19) an experimenter pushed 7 code commits and declared task complete without running the benchmark. Only a user-initiated status query caught the gap.
```

**Reason:** Session 37d95c4c, experimenter a121f4cf skipped benchmarks entirely but reported success.

### Agent Definitions — retrospective.md (open GH issues for proposals)

**File:** `.claude/agents/retrospective.md`

```diff
 ## Output Format

 Write to `research_log/drafts/retrospective-YYYY-MM-DD.md`:

+After writing the draft, also:
+1. **Open ONE GitHub issue** per proposed change with `gh issue create --title "[retro-proposal] <name>" --body <diff + reason> --label retro-proposal`
+2. **Post a summary comment on yesterday's retro issues** marking each APPLIED / NOT-APPLIED / OBSOLETE with evidence from today's transcripts
+3. **Do not open a duplicate issue** — if yesterday's proposal already has an open issue, comment on it instead
+
+The research_log/drafts/retrospective-*.md file is for human narrative. GitHub issues are the actionable tracking surface — this is the ONLY mechanism that has ever caused retro proposals to get applied.
```

**Reason:** Five days running of retro proposals going nowhere. The file-drafts pattern has proven to be a dead letter. Converting to GitHub issues puts them in the user's standard workflow.

### Hooks — fleet watchdog

**Proposed hook:** cron-driven fleet health check (via `/schedule` skill)

```
cron: */30 * * * *   (every 30 minutes)
command: python scripts/pod.py health --stale-runs --alert
```

Where `--stale-runs` flags any running experiment whose log file hasn't been touched in >15 min AND `nvidia-smi` shows 0% util. Emit a notification to the active Claude session.

**Reason:** 16 hours of silent pod1 failure today on tulu_control seed 137. The 15-30s polling cadence in CLAUDE.md only covers the launching session — nothing watches pods globally.

### Hooks — SessionStart pod + branch snapshot

**Proposed hook:** `SessionStart`

```json
{
  "hooks": {
    "SessionStart": [
      {
        "command": "git -C /home/thomasjiralerspong/explore-persona-space branch --show-current && echo '---' && python /home/thomasjiralerspong/explore-persona-space/scripts/pod.py health --quick 2>/dev/null | head -20 && echo '---' && ls -t /home/thomasjiralerspong/explore-persona-space/research_log/drafts/retrospective-*.md | head -1",
        "description": "Show branch + fleet health + latest retro on session start"
      }
    ]
  }
}
```

**Reason:** 4th consecutive retro proposing this. Today the user had to ask "why are we on a feature branch" (implicit correction) and the fleet-env-drift on all 5 pods went undetected until launch time.

### Memory Updates

**New file:** `.claude/agent-memory/research-pm/feedback_verify_subagent_limits.md`

```markdown
---
name: Verify Subagent Capability Claims
description: When a subagent reports an infrastructure limit that contradicts its agent definition, verify before re-dispatching
type: feedback
---

When a subagent reports a limit like "I can't write files" / "no SSH access" / "tool X missing," do NOT accept at face value.

**Why:** 2026-04-19 session 9772561a — experimenter subagent claimed twice that it could not write files. On the third attempt (inline), writes worked fine. The parent lost ~30 min to re-dispatch loops trusting a fabricated limit.

**How to apply:** Before re-dispatching or falling back to inline work, verify the limit directly:
- "can't write files" → `ssh_execute <pod> "touch /tmp/x && rm /tmp/x"`
- "no SSH access" → `ssh_execute <pod> "hostname"`
- "tool missing" → `ssh_execute <pod> "which <tool> || uv run which <tool>"`

Only fall back to inline if the test confirms the limit.
```

**Update:** `.claude/agent-memory/retrospective/project_unapplied_backlog.md`

```diff
-3 days in a row (2026-04-15, 2026-04-16, 2026-04-17) the daily retrospective has produced proposals that never get applied.
+5 days in a row (2026-04-15 through 2026-04-19) the daily retrospective has produced proposals that never get applied.
```

Advance the counter, add today's unapplied items. Flag: "If this count reaches 6 days, stop writing retros and instead write ONE blocker issue asking the user to either apply or explicitly reject each proposal."

**Update:** `.claude/agent-memory/retrospective/MEMORY.md` — add pointer to the new `feedback_verify_subagent_limits.md` (cross-referenced under research-pm).

## Successful Patterns (reinforce these)

- **Adversarial review on issue #36 caught real bugs** — code-reviewer flagged Liger-on-LoRA logging inconsistency, DPO memory regression, and SFT throughput claim mismatch. Follow-up cleanup commit `a507458` addressed all 10 items. This is the happy path for the issue workflow.
- **Honest admission of falsified prior result** — when user asked "did we falsify any previous results?", Claude accurately reported that the `+293% packing` and `+22% DPO` claims were formula artifacts at realistic scale. Non-defensive recording of own mistakes. Keep this norm.
- **Effort migration `max → xhigh` handled cleanly** in 1a398de9 — user made a design decision, Claude applied it across settings.json + agent frontmatter + CLAUDE.md in one dispatch.
- **Analyzer + reviewer pair on 5-seed leakage data** (9772561a) — independent review catches overclaims. Worth keeping as default.
- **Post-compaction resume preserved full context** — 18e61e00 went through 3 context rehydrations and kept all user corrections. Good hygiene.
- **Pivot to inline work after subagent failed 2x** — while the underlying cause was wrong (see "Verify Subagent Limits" above), the instinct to stop thrashing and do it directly was correct once recognized.

## Metrics

- **Real user messages:** ~62 across 11 sessions (excluding command captures, task notifications, tool outputs)
- **Explicit user corrections:** 8 ("no the 10 seeds are just EM", "this is unrelated so don't run it", "override process — plan directly", "search more deeply", "change default effort to xhigh not max", "don't add a new section — integrate", "this wasn't working before", "I mean finetuning only on [ZLT] at the end")
- **Agent dispatches:** ~23 (5 experimenters in 18e61e00, 16 in 37d95c4c, 3 in 9772561a, plus scattered)
  - No user intervention: ~12 / 23
  - Required correction: ~8 / 23
  - Fabricated limits or silent-skip: ~3 / 23 (experimenter "can't write files" ×2; experimenter a121f4cf skipped benchmark)
- **Experiments launched today:** persona-leakage single-token sweep (pod1), midtrain seed 256 full matrix (5 pods), midtrain seed 137 tulu_control retry (pod1)
- **Silent failures surfaced:** 1 (pod1 exit 137 OOM from 2026-04-18, sat 16h)
- **Pod env-sync failures at launch:** 5 (uv/liger mismatch on all 5 pods)
- **Phantom tool calls:** 1 (`SendMessage`)
- **File-not-read-first errors:** ≥6 (across 1a398de9, 8acd7099, 37d95c4c)
- **Parallel Bash cancellation loops:** 2 distinct (7 cancels in 1a398de9, 6+ cancels in 3f3f75c2)
- **Time roughly split:** ~35% experiment launch + monitoring, ~25% debugging (env sync, subagent fabrications, upload state), ~20% planning (leakage, Tier 1), ~15% config migration (opus 4.7), ~5% dotfiles/happy-coder

---

**Meta-observation (unchanged from 2026-04-16 through -18):** Today was productive but ~25% of total time was spent re-debugging issues the retrospective system already flagged. The single highest-ROI action available remains the same: **apply the backlog** — even as 5-minute micro-commits, one proposal at a time. The proposed change to convert retro proposals to `retro-proposal`-labelled GitHub issues is itself backlog meta-work: it makes the other proposals actionable in a format the user's `/issue` workflow already handles. If one thing gets applied from this retro, make it the GitHub-issue conversion — it unblocks every other proposal.
