# Daily Retrospective — 2026-04-25

**Sessions reviewed:** 7 top-level + ~28 subagent transcripts (one session is the retro itself)
**Real user messages today:** ~40 distinct semantic turns across all sessions (~480 raw lines counting `<task-notification>` echoes and monitor re-fires)
**Agent dispatches:** 25 (planner ×6, experimenter ×6, Explore ×4, critic ×3, gate-keeper ×2, analyzer ×2, reviewer ×2)
**Skill invocations:** `issue` ×2, `adversarial-planner` ×2
**Commits to main:** 19 (heavy code-shaping day on #100/#101/#102/#108 scripts)
**Clean results shipped today:** **#105** (assistant robustness = data confound, HIGH) + **#106** (qwen identity claim, MODERATE) — **2**
**Issues advanced:** #100 → done-experiment, #101 → done-experiment, #102 → reviewing, #104 → plan-pending (revised w/ Fitness D), #108 → running, #91 → updated with full 7-source strong-convergence dataset (8th consecutive day of activity on this dataset)

## Summary

A **decisive-finding day**: the adversarial-planner pipeline did its job on issue #100 in the first hour after midnight — the critic-flagged contamination confound was confirmed by a 25-min Exp 0 run, saving an entire dose-response sweep, and clean-result #105 ("data artifact, HIGH confidence") was published by 1:48am. Issue #101's qwen-default vs generic-assistant comparison shipped as #106 (MODERATE) by 2:48am. By evening, the user had pivoted both findings into two new follow-up issues (#108 cross-model defaults, #104 prompt-search with KL/distributional fitness), each pushed through full clarifier → gate-keeper → planner → critic → revise. Issue #102 marker-bridge ran 4 conditions × 3 seeds and produced a clean **null** (T mean 88.47 vs C1 mean 88.63, Δ 0.17) — confirming the gate-keeper's prior that markers and misalignment occupy different representational spaces.

But the same recurring research-blocking bugs fired **again** today, **3rd consecutive day**:

1. **Pod1 root-overlay disk exhaustion** silently SIGKILL'd C2 seed-42 retry of #102. "Disk quota exceeded on /" with 3GB free. Yesterday's `_check_root_disk()` preflight proposal still unapplied. Cost ~30 min of debug + retry.
2. **`gh pr create` fails** with "No commits between main and issue-N" on every fresh `/issue` dispatch. Fired today on #100, #101, #102 — three issues, three bespoke recoveries. Step 5b of `/issue/SKILL.md` opens the PR *before* any commit exists.
3. **Qwen default system prompt** auto-injection — the fact yesterday's retro proposed adding to CLAUDE.md — was the load-bearing fact for both #105 and #106 today. It came up again (`dd19ec71` 20:38, "Can you try with default system prompts of other models?", "Try just 'You are Qwen' and variants"). Still not in CLAUDE.md.

**0 of yesterday's 13 proposals were applied today. 7th consecutive zero-day.** The disk + merge + qwen + PR-create bugs are now production failures, not workflow polish. The retro-backlog meta-issue recommended yesterday was not opened.

## Proposal Backlog Audit (read this first)

| Proposal (origin) | Status | Evidence today |
|---|---|---|
| `_check_root_disk()` + WandB cache redirect in preflight (2026-04-23, -24) | ❌ **UNAPPLIED, 3rd ask** | C2 retry of #102 OOM'd at 21:40 with "Disk quota exceeded on /" — same exact failure mode |
| `merge_marker_adapter` weight-bytes assertion (2026-04-23, -24) | ❌ **UNAPPLIED, 3rd ask** | Did not fire today (fortunate); bug present at line 556 |
| Qwen default system prompt fact in CLAUDE.md (2026-04-24) | ❌ **UNAPPLIED, 2nd ask** | Re-rediscovered in `dd19ec71` 20:06 + 20:38 + 20:40; load-bearing for #105 + #106 |
| `epm:anchor` first-comment in `/issue` (2026-04-23, -24) | ❌ **UNAPPLIED, 3rd ask** | No compactions today (smaller sessions), so no counter-evidence; rule still missing |
| Analyzer post-write figure-data sanity check (2026-04-24) | ❌ **UNAPPLIED, 2nd ask** | No figure defects flagged today (analyzer ran twice, both clean); no counter-evidence |
| Reviewer-PASS label transition (`clean-results:draft → clean-results`) (2026-04-24) | ❌ **UNAPPLIED, 2nd ask** | #91 still labeled `clean-results:draft` at end of day despite the full 7-source dataset being posted at 11:21am |
| `/schedule` cron for monitoring (2026-04-19 → -24, 6th ask) | ❌ **UNAPPLIED** | 20 ScheduleWakeup + 203 ssh_execute today; less than yesterday but still ~40% of session tokens |
| `gpu_memory_utilization=0.85` vLLM default (2026-04-21 → -24, 6th ask) | ❌ **UNAPPLIED** | No counter-evidence (no new vLLM OOM today) |
| `mcp__ssh__ssh_group_execute` removal from CLAUDE.md (2026-04-23, -24, 3rd ask) | ❌ **UNAPPLIED** | Still listed in `CLAUDE.md` line 70 |
| Canonical Recipes section in CLAUDE.md (2026-04-23, -24, 3rd ask) | ❌ **UNAPPLIED** | No "How are you inducing misalignment?" question today; no counter-evidence |
| 3 hooks (SessionStart, PostToolUse on `git push`, UserPromptSubmit on "check progress") (multi-retro) | ❌ **UNAPPLIED** | 10th, 9th, 5th asks respectively |
| **`meta:retro-backlog` issue (2026-04-24's recommended single action)** | ❌ **NOT OPENED** | No `meta` or `retro-backlog` label in repo as of end-of-day |

**Net: 0 of 13 applied. 7th consecutive zero-day.** The pattern is unambiguous: **config-shaped proposals do not land**. Code-shaped proposals also did not land today (no proposal-targeted commits in the 19 commits made). I will draft a smaller, more surgical retrospective today, and I will not repeat the meta-backlog draft below — instead I will write a single P0 patch and stop.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---|---|---|---|
| **`gh pr create` fails on fresh worktree** with "No commits between main and issue-N" — fired on every dispatch today (#100, #101, #102). Recovery is ad-hoc: sometimes a placeholder commit, sometimes defer to specialist. New friction, **3 firings in one day**. | `56cd23f2` (#100), `dd19ec71` (#101), `073d768e` (#102) | Step 5b should create a placeholder commit *before* `gh pr create` — copy `.claude/plans/issue-<N>.md` into the worktree, commit with `Add experiment plan for issue #<N>`, push, *then* `gh pr create`. Idempotent (skips if branch already has commits). | `.claude/skills/issue/SKILL.md` (Step 5b) |
| **Pod1 root-overlay disk exhaustion** silently SIGKILL'd C2 seed-42 retry of #102 — "Disk quota exceeded on /" with 3GB free. Required ~30 min debug + cleanup + retry. **3rd consecutive day for this exact failure mode.** | `073d768e` 21:40 | Yesterday's `_check_root_disk()` + `_redirect_wandb_to_workspace()` proposal verbatim. The `check_disk_space()` already in `preflight.py` checks `/workspace` only; root is missed. | `src/explore_persona_space/orchestrate/preflight.py` |
| **Qwen default system prompt fact still not in CLAUDE.md** — load-bearing for both clean results today (#105, #106) and for the user's pivot question "Can you try with default system prompts of other models?" The fact is rediscovered every time someone touches a no-system-prompt experiment. | `dd19ec71` 20:06 + 20:38 + 20:40, `18b77079` 20:43 (KL-divergence question) | Yesterday's one-line CLAUDE.md diff verbatim (Persona injection section). | `CLAUDE.md` |
| **SSH `sleep N && cmd` chains hit 30s tool timeout** — `timeout 60 sh -c 'sleep 30 && nvidia-smi'` returns "Command timeout after 30000ms" because the SSH MCP transport caps at 30s regardless of inner timeout. 3 firings today on `073d768e`. | `073d768e` 20:04, 20:05, 20:59 | Document in `CLAUDE.md` SSH section: don't `sleep N && cmd` — instead, schedule the wakeup with `ScheduleWakeup(delaySeconds=N)` then run `cmd`. The `mcp__ssh__ssh_execute` tool has a hard 30s limit. | `CLAUDE.md` (SSH section) |
| **Issue #91 reviewer-PASS label transition** — full 7-source strong-convergence dataset posted at 11:21am, but #91 is still labeled `clean-results:draft` at end of day. Same complaint as yesterday (re #91), still unapplied. | `0efaaf34` 11:21 | Yesterday's analyzer Step 7d / `/issue` Step 7d proposal verbatim. | `.claude/skills/issue/SKILL.md`, `.claude/agents/analyzer.md` |

## Failed Approaches (document to prevent retries)

- **`sleep 30 && nvidia-smi --query-gpu...` over `mcp__ssh__ssh_execute`** — the MCP wrapper enforces a 30s outer timeout that ignores the inner `timeout 60` flag. Use `ScheduleWakeup` for the delay, then a fresh `ssh_execute` for the query.
- **`sleep 1800 && echo "check MD completion"`** (`0efaaf34` 03:36) — harness now blocks long-leading sleeps. Pivot: `ScheduleWakeup(delaySeconds=1800, prompt=...)`. The session correctly migrated to `ScheduleWakeup` in the auto-experiment loop, which validates the new pattern.
- **C2 seed-42 retry without disk cleanup** — first retry attempted on the same disk-full state; second retry after `du -sh` + cleanup succeeded. The pattern "retry on root-disk-full pod will fail again" should be encoded in the experimenter's runbook.
- **Issue #100 plan attempted "no_system" as control** — silently identical to "qwen_default" because of chat-template auto-injection. Fact-checker caught it on 2026-04-24. Today's planner runs (#104, #108) DID flag this correctly — the lesson took.

## Proposed Changes

### P0 (re-proposing; fired again today)

#### `src/explore_persona_space/orchestrate/preflight.py` — root-disk + WandB cache redirect (3rd ask)

```diff
+# --- Root-overlay disk + WandB cache redirect ---
+def _check_root_disk(report: PreflightReport):
+    out = subprocess.check_output(["df", "--output=avail", "/"], text=True).splitlines()
+    avail_kb = int(out[1].strip())
+    if avail_kb < 5_000_000:  # 5 GB
+        report.add_error(
+            f"Root overlay has only {avail_kb // 1_000_000} GB free; "
+            f"vLLM + WandB will SIGKILL silently. "
+            f"Run: `du -sh /root/.cache /root/.wandb /tmp` and clean."
+        )
+
+def _redirect_wandb_to_workspace(report: PreflightReport):
+    for var, default in [
+        ("TMPDIR", "/workspace/tmp"),
+        ("WANDB_DIR", "/workspace/.wandb"),
+        ("WANDB_CACHE_DIR", "/workspace/.wandb-cache"),
+    ]:
+        if not os.environ.get(var):
+            os.environ[var] = default
+            os.makedirs(default, exist_ok=True)
+            report.add_warning(f"Defaulted {var}={default}")

 def require_preflight():
     ...
+    _check_root_disk(report)
+    _redirect_wandb_to_workspace(report)
```

**Reason:** Same exact failure mode 3 days running (2026-04-23 villain rerun, 2026-04-24 Arm A 4-vLLM kill, 2026-04-25 #102 C2 retry). Cost compounding: ~30 min today + ~40 min yesterday + ~40 min day before = ~2 GPU-hours of disk-debug across the streak.

#### `CLAUDE.md` — Qwen default system prompt fact (2nd ask)

```diff
 - **Persona injection:** ALWAYS system prompt `{"role": "system", "content": "<persona>"}`. Never in user/assistant turns.
+- **Qwen default system prompt:** Qwen2.5's chat template auto-injects `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` when no system message is provided. To get a truly empty system block, pass `{"role": "system", "content": ""}`. **All historical "no system prompt" controls are silently testing qwen_default.** Confirmed load-bearing for #105 (data confound disambiguation) and #106 (identity-claim leakage finding).
```

**Reason:** Load-bearing for both clean results shipped today (#105 found the qwen_default outlier traced to the same anchor confound; #106 isolated the identity claim from prompt-length effects). User asked at 20:06 "Can you try with default system prompts of other models?" → led to #108 entirely because of this fact.

### P1 (new today)

#### `.claude/skills/issue/SKILL.md` — Step 5b: placeholder commit before PR (NEW, 3 firings today)

```diff
 **5b. Draft PR.** Open a draft PR with `Closes #<N>` in the body, linking back to
 the issue. Use `gh pr create --draft --head issue-<N> --body "Closes #<N>"`.
+
+First, ensure the branch has at least one commit. A fresh worktree branched from
+main has no diff and `gh pr create` fails with `GraphQL: No commits between main
+and issue-<N>`. Copy the cached plan into the worktree as the placeholder commit:
+```bash
+cd .claude/worktrees/issue-<N>
+mkdir -p .claude/plans
+cp ../../plans/issue-<N>.md .claude/plans/issue-<N>.md
+git add .claude/plans/issue-<N>.md
+git commit -m "Add experiment plan for issue #<N>"
+git push -u origin issue-<N>
+```
+If the branch already has commits (resume case), skip this — `gh pr create` succeeds.
```

**Reason:** Fired on every fresh `/issue` dispatch today (#100, #101, #102). Each had a different ad-hoc recovery (placeholder commit on #100, defer to specialist on #101/#102). Codify the placeholder pattern; idempotent on resume.

#### `CLAUDE.md` — SSH `sleep` gotcha (NEW, 3 firings today)

```diff
 ### When to still use Bash SSH

 - Interactive/streaming output (e.g., `tail -f`)
 - Commands that need TTY allocation
 - Piped multi-command chains that are easier as one-liners
+
+### Don't `sleep N && cmd` over `mcp__ssh__ssh_execute`
+The MCP wrapper enforces a hard 30-second timeout that ignores any inner `timeout`
+flag. `sleep 30 && nvidia-smi` will return "Command timeout after 30000ms" even if
+wrapped in `timeout 60`. Pattern: use `ScheduleWakeup(delaySeconds=N)` to delay,
+then a fresh `ssh_execute` for the query. For polling loops, use `Monitor` with
+an until-loop on a state file the experiment writes.
```

**Reason:** 3 firings today (`073d768e` 20:04, 20:05, 20:59), each costing a re-issue. Pattern is non-obvious from the SSH MCP table.

### Already-drafted, not re-drafted here

The following are in yesterday's retro and the unapplied-backlog memory:

- `merge_marker_adapter` weight-bytes assertion in `run_causal_proximity.py` (3rd ask, did not fire today but bug present)
- `epm:anchor` first-comment convention in `/issue` (3rd ask, no compaction today so no counter-evidence)
- Analyzer post-write figure-data sanity check (2nd ask, no defects today)
- Reviewer-PASS label transition for `clean-results:draft → clean-results` (2nd ask, #91 still draft at EOD)
- `/schedule` cron daemon for monitoring (6th ask)
- `gpu_memory_utilization=0.85` default (6th ask)
- `ssh_group_execute` removal from CLAUDE.md (3rd ask)
- Canonical Recipes section in CLAUDE.md (3rd ask)
- 3 hooks (SessionStart, PostToolUse on `git push`, UserPromptSubmit on "check progress")

### Memory Updates

- **Add new memory:** `project_2026_04_25_retro.md` — "Decisive-finding day: 2 clean results shipped (#105 HIGH, #106 MODERATE) + 1 confirmed null on #102 marker bridge. Adversarial-planner caught real confound on #100; same disk + Qwen + PR-create bugs fired AGAIN. 7th consecutive zero-day for proposals."
- **Update** `project_unapplied_backlog.md` — bump counter to **7th consecutive zero-day**; add the new "PR-create on fresh worktree" + "SSH 30s timeout sleep gotcha" friction items.
- **Add new memory (proposal):** `feedback_pr_placeholder_commit.md` — "On fresh worktree, `gh pr create` requires at least one commit. Copy the plan as a placeholder commit before opening the PR. **Why:** Fired on #100/#101/#102 in one day. **How to apply:** Always run the placeholder-commit block in the issue skill's Step 5b before `gh pr create`."
- **Add new memory:** `feedback_ssh_30s_timeout.md` — "`mcp__ssh__ssh_execute` enforces a hard 30s timeout. Don't pair with `sleep N` for N≥25. **Why:** Fired 3× today on `073d768e`. **How to apply:** Use `ScheduleWakeup(delaySeconds=N)` for the delay, then a fresh `ssh_execute`."

### Hooks — still deferred

Same recommendation as yesterday: don't draft hooks until the meta-backlog issue exists. **The meta-backlog issue itself was not opened today**, so we are now in the second cycle of "deferred until X exists, X never gets created." The right move is to drop hooks from the retro entirely and treat them as zombie proposals.

## Successful Patterns (reinforce these)

- **Adversarial-planner pipeline shipped two clean results in one night** — #100 (filed 00:39, planning, dispatched at 00:55) → Exp 0 result by 01:08, full Exp C complete by 01:36, clean result #105 posted by 01:48. Total wall-clock: **70 minutes from filing to clean result**. The fact-checker stage caught the contamination confound *during planning*; the experimenter executed against a hardened plan; analyzer wrote #105 directly without a draft-promotion step. **This is the workflow operating at design quality.**
- **#101 follow-up to #105 was tight** — same pipeline, filed 02:41, clean result #106 posted by 02:47. The two-clean-result-overnight cadence is reproducible.
- **Issue #102 marker bridge was a clean null with a credible mechanism** — gate-keeper had flagged the prior-evidence mismatch (markers and misalignment in different spaces); user explicitly overrode; experiment ran 4 conditions × 3 seeds; null confirmed (T 88.47 vs C1 88.63, Δ 0.17). **Both the gate-keeper's prior and the user's "test it anyway" instinct were vindicated** — the experiment provided a positive null, and the gate-keeper's framing made the result interpretable. Override mechanism worked as designed.
- **User-driven plan refinement on #104 + #108** — user added KL divergence as Fitness D mid-planning ("Can we also have kl divergence between logits as a metric?"); planner accommodated and re-posted. User added cross-model defaults to #108 ("default system prompts of other models"); planner expanded the condition matrix. **The planner agent is responsive to mid-stream user input without losing the rest of the plan structure.**
- **Pivoted from #96 finding without sunk-cost defense** — when Exp 0 of #100 confirmed the data confound, the response was *publish #105 as HIGH-confidence "data artifact"* rather than *defend the prior finding*. This is the Hughes/Evans honest-analysis principle in practice.
- **#108 handled disk-full setup proactively** — by midday, after the #102 disk-exhaustion event, the user spec'd "use scripts/run_issue108.py with absolute paths" and the experimenter complied. Pattern is forming: "if pod1 in last 24h showed disk pressure, write to /workspace explicitly."

## Metrics

- **Real user corrections (distinct):** ~6 ("Should we rerun any experiments considering we found this bug?", "Can you try with default system prompts of other models?", "But can we rerun the assistant data point", "Try just 'You are Qwen' and variants", "Update the plan with Fitness D and re-post", "Why are there missing squares in the table")
- **Polling-style re-invocations:** ~30 (down from ~135 yesterday — fewer long-running monitor sessions)
- **Agent dispatches:** 25 (down from 41 yesterday — heavier per-issue dispatch density, fewer issues touched)
- **Experiments dispatched today (new runs):** 4 (#100, #101, #102, #108)
- **Experiments completed today:** 4 (#100 → #105, #101 → #106, #102 reviewing, #91 strong-conv full dataset)
- **Compaction events:** 0 in main sessions (fewer long monitor loops than yesterday)
- **Commits to main:** 19 (4× yesterday's 3, but all code-shaping; 0 retro-proposal targeted)
- **Proposals applied same-day from yesterday's retro:** **0 of 13 (7th consecutive zero-day)**
- **Proposals persisting across 3+ retros:** **15+** (unchanged)
- **Estimated token cost of monitoring-polling:** ~30-40% of session tokens (`073d768e`, `0efaaf34`). **Down from 50-60% yesterday — partial structural improvement; ScheduleWakeup adoption working.**
- **GPU-hours saved by adversarial-planner today:** ~3 (Exp 0 short-circuited #100's planned dose-response sweep)
- **GPU-hours wasted on root-disk SIGKILL:** ~0.5 (#102 C2 retry cleanup + relaunch)

## Recommended single action for tomorrow

**Apply the P0 trio: root-disk preflight + Qwen default fact + PR-create placeholder commit.** All three fired today; all three are 1-3 line patches; together they prevent the three most expensive recurring failure modes. If the user only does ONE thing: ship the root-disk preflight diff. Three days of the same SIGKILL is enough.

If a second thing fits in the morning: open the `meta:retro-backlog` issue with the 15-item checklist. Yesterday's recommended single action; still not done. Without it, this retro becomes the 8th consecutive draft-and-ignore.
