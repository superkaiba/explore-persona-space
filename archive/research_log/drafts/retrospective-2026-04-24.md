# Daily Retrospective — 2026-04-24

**Sessions reviewed:** 9 top-level + ~28 subagent transcripts
**Total real user messages:** ~80 distinct semantic turns (~2,030 raw lines counting `<task-notification>` echoes and monitoring re-fires)
**Agent dispatches:** 41 (experimenter ×9, planner ×8, gate-keeper ×6, critic ×6, Explore ×6, general-purpose ×2, analyzer ×2, reviewer ×1, code-reviewer ×1)
**Commits to main (today):** 3 (all #99 figure polish — `381e2d1`, `7b7568c`, `20a84c9`)
**Clean results shipped today:** #99 (behavioral leakage MODERATE) — **1**
**New issues created today:** #100, #101, #102 (all reached `status:plan-pending`)
**Issues continuing from prior days:** #61 (causal proximity Arm B running), #69 (capability/misalignment leakage)

## Summary

A **plan-heavy + monitor-heavy day**: 3 fresh issues opened and pushed through the full clarifier → gate-keeper → planner → fact-checker → critic → revise pipeline successfully (#100, #101, #102), one large clean result shipped (#99), and the Issue #61 causal-proximity sweep continued to grind through Arms A → C → B with **6 conversation compactions** in a single session (`0efaaf34`) due to long monitoring loops. The adversarial-planner workflow was the hero of the day — fact-checker caught a real blocker on #100 (issue-69 scripts only on a worktree branch), and on #101 (Qwen chat template auto-injects the default system prompt, making "no system prompt" condition silently equal "qwen_default"). Critic on #100 caught a real **#96 contamination confound** (assistant source uniquely got 2:1 wrong:correct anchor ratio). These are exactly the kind of findings the workflow is designed to surface, and it worked.

But: **0 of yesterday's 13 retro proposals were applied today** — same as yesterday, same as the day before. This is now **6 consecutive days** of config-shaped proposals not landing. The "code-shaped commits land, config-shaped commits do not" gap is now load-bearing.

New friction this day:

1. **Monitoring polling is now structural** — 133 monitoring-style messages across all sessions; session `0efaaf34` alone has 1,691 user-message lines (mostly auto-fired loop wakeups), 78 `ScheduleWakeup` calls, 824 `ssh_execute` calls. The wakeup-then-poll-via-ssh pattern dominates token spend. **Same complaint as 2026-04-20, -21, -22, -23. 5th consecutive day. Yesterday's `/schedule` proposal still unapplied.**
2. **6 compaction events in one session** (`0efaaf34`) — user re-asked "What are we doing with this finetuning exactly?", "Wait what is the error", "I dont see it in the clean results column" mid-stream. Each compaction is a 50K-100K-token loss that the agent has to reconstruct from issue comments + plan files. The `epm:anchor` proposal from yesterday's retro would have helped here — still unapplied.
3. **Disk-exhaustion silent kill (recurrence of #61 villain rerun pattern)** — pod1 root overlay hit 98% (482 MB free) on 4 simultaneous WandB runs, killing all 4 Arm A subprocess evals silently with no traceback. Required full disk audit (~40 min wasted). This is the **second time** this exact failure mode appeared in 2 days; yesterday's `df -h` pre-launch + `assert weight_bytes > 1 GB` proposals still unapplied.
4. **Pod1 git checkout broken** — recurring "use /tmp scripts or git show workaround" workaround visible in 0efaaf34 user prompts. Symptom of the pod-venv-drift / pod-state-drift issues filed as #76 a week ago.
5. **Default system prompt knowledge missing from agent context** — user had to teach the agent twice today (in `da8aa615` and again in `dd19ec71`) that Qwen2.5 auto-injects `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` when no system message is provided. The fact-checker on #101 also caught this independently. **This fact should live in `CLAUDE.md`.**
6. **Issue #91 still labeled `clean-results:draft`** — has been since 2026-04-23. Reviewer PASS never recorded; the draft has not graduated to `clean-results`. Same stale-draft pattern as #70 last week.
7. **Issue #61 plot question**: "No i meant at ALL the checkpoints the plots show that the source marker % is 0% which doesn't make sense even with the results you showed me" — analyzer's hero figure had a real defect (every source row showed 0% baseline marker rate, which is impossible if the marker was ever trained). User caught this; analyzer's post-write self-check (yesterday's proposal) would have caught it earlier.

## Proposal Backlog Audit (read this first)

Yesterday's 2026-04-23 retro proposed 13 changes. Status today:

| Proposal | Status | Evidence today |
|---|---|---|
| `/schedule` routing rule in CLAUDE.md monitoring section (4th ask) | ❌ **UNAPPLIED** | 78 `ScheduleWakeup` + 1,691 monitor-poll user-message lines in `0efaaf34` alone |
| Canonical Recipes section in CLAUDE.md (EM, marker, persona-prompted SFT) | ❌ **UNAPPLIED** | "How are you inducing misalignment?" appeared again today (`da8aa615`); user had to re-state the bad-legal-advice recipe |
| `mcp__ssh__ssh_group_execute` removal from CLAUDE.md (it does not exist) | ❌ **UNAPPLIED** | Still listed in `CLAUDE.md` line 70 |
| `gpu_memory_utilization=0.85` default in vLLM (5th ask) | ❌ **UNAPPLIED** | Still 0.60 in `vllm_completions.py` |
| Analyzer post-write self-check (verify body length, figure URLs, project column) | ❌ **UNAPPLIED** | "I dont see it in the clean results column" again on #61; #91 still labeled draft, not promoted |
| `epm:anchor` first-comment convention in `/issue` skill | ❌ **UNAPPLIED** | 6 compactions in `0efaaf34` with full context loss; user had to re-explain the experiment 4+ times |
| Label + PR pre-create validation in `/issue` skill | ❌ **UNAPPLIED** | `aim:3'` style label errors not seen today (no counter-evidence; rule still missing) |
| `merge_marker_adapter` weight-bytes assertion in `run_causal_proximity.py` | ❌ **UNAPPLIED** | Same `config.json` skip-check bug surfaced again on `0efaaf34` Arm B (line 556) |
| `/loop` anti-re-entry guard | ❌ **UNAPPLIED** | `/loop` only invoked once today (one-shot via `da8aa615` for sycophancy/refusal monitoring); pattern not re-triggered |
| Planner scope-discipline (4th ask, "minimal-viable default") | ❌ **UNAPPLIED** | Today the planner DID include falsification thresholds, kill criteria, multi-seed validation on #100 + #101. User did not strip them this time, so no counter-evidence — but the proposal is still unimplemented |
| `verify_clean_result.py` forbidden-token grep | ⏳ **PARTIAL** | `check_forbidden_stats()` exists at line 339 (forbidden statistics language). Other forbidden tokens (signoff, "[Clean Result]", aims) still not enforced |
| All 3 hooks (SessionStart, PostToolUse on `git push`, UserPromptSubmit on "check progress") | ❌ **UNAPPLIED** | 9th, 8th, 4th asks respectively |
| Memory updates from yesterday's retro | ❌ **UNAPPLIED** | `project_canonical_recipes.md` not added; `feedback_verify_subagent_limits.md` not updated for fabricated MCP tool names |

**Net: 0 of 13 applied. 6th consecutive zero-day for config proposals.**

This is **not** a "fix the proposals" problem — it's a "the user's workflow has no slot for applying retro proposals" problem. The user's day is dominated by experiment monitoring + plan iteration, neither of which surfaces "go implement these CLAUDE.md edits". Recommendation: **stop drafting hook diffs and CLAUDE.md edits in this retro until a meta-issue exists for the backlog**. Yesterday's retro made the same observation. Three options:

- (a) Open a `meta:retro-backlog` GitHub issue today with all 6 retros' unapplied items as a checklist; user can `/issue <N>` it any morning.
- (b) Drop config proposals from retro entirely; only flag *recurring research-blocking bugs* (e.g., disk exhaustion, merge weight check).
- (c) Continue current cadence and accept that retros are an audit log, not an action queue.

I will draft (a) below, but the user gets the call.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---|---|---|---|
| **Monitoring spam: 1,691 user-message lines in a single session** with 78 `ScheduleWakeup` + 824 `ssh_execute`. 6 compactions. Same complaint 5 days running. | `0efaaf34` (primary), `da8aa615` (25 `/loop` re-fires) | **Re-frame**: The fact that this is the 5th retro flagging the same pattern with 0 follow-through means the proposal is wrong. Consider: (i) move polling to a **`/schedule` cron** that runs an `experiment-monitor` skill on a fixed cadence and posts results as `epm:status` issue comments — agent reads the comments instead of polling SSH; (ii) build a `pod_status.py` daemon on each pod that writes a single `status.json` per active experiment, and a single `gh issue comment` updater on the manager VM. Either eliminates the polling loop. | `.claude/skills/experiment-runner/SKILL.md`, `scripts/pod_status_daemon.py` (new) |
| **6 conversation compactions in one session → 4× context-rebuild from issue comments** | `0efaaf34` | **Re-propose**: `epm:anchor` first-comment convention from yesterday. Add to `.claude/skills/issue/SKILL.md` Step 0. Every `/issue <N>` writes a 500-char anchor comment containing GOAL / CONDITIONS / READOUT / ARTIFACTS. On compaction recovery, the agent fetches this anchor first, before any monitoring. | `.claude/skills/issue/SKILL.md` |
| **Pod1 root-overlay 98% disk exhaustion → silent SIGKILL of 4 subprocesses** | `0efaaf34` (Arm A → eval-only recovery) | Pre-launch check + WandB cache redirect. Add to `src/explore_persona_space/orchestrate/preflight.py`: assert `df --output=avail / | tail -1` returns ≥ 5 GB on root, AND that `WANDB_DIR`, `WANDB_CACHE_DIR`, `TMPDIR` all point under `/workspace`. The wrapper scripts already set TMPDIR after the crash, but it's a recovery patch, not a pre-flight. **2nd consecutive day for this failure mode.** | `src/explore_persona_space/orchestrate/preflight.py`, `scripts/run_causal_proximity.py` |
| **`merge_marker_adapter` skip check is config-only — corrupted weight-less merges pass** | `0efaaf34` (Arm B fix-up cycle, line 556 of `run_causal_proximity.py`) | Yesterday's `assert weight_bytes > 1 GB` proposal. **Same bug, same day to discover, second day in a row.** | `scripts/run_causal_proximity.py` |
| **Qwen default system prompt knowledge missing from agent context** | `da8aa615`, `dd19ec71`, `9a229638`, `56cd23f2` (fact-checker found it independently) | Single line in `CLAUDE.md`: "Qwen2.5 chat template auto-injects `\"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\"` when no system message is provided. To get a truly empty system block, pass `{\"role\": \"system\", \"content\": \"\"}`." This fact was rediscovered 4 separate times today. | `CLAUDE.md` (Persona injection section) |
| **Pod1 git checkout broken → user-prompt workarounds** | `0efaaf34` | Diagnose: `cd /workspace/explore-persona-space && git checkout main` fails on pod1. Likely cause: lingering uncommitted changes from prior experimenter run-on-pod, OR worktree-state corruption from a prior `git fetch` quota error. Add `git status --porcelain` to `pod.py health` and `git reset --hard origin/main` to `pod.py health --fix`. | `scripts/pod.py` |
| **Issue #91 still labeled `clean-results:draft`** | (not in transcripts — repo state) | Reviewer PASS marker never recorded. Either the reviewer ran but didn't update the label, or the analyzer never spawned the reviewer. Add to `.claude/skills/issue/SKILL.md` and `.claude/agents/analyzer.md`: post-reviewer-PASS, the analyzer/orchestrator updates the label `clean-results:draft → clean-results`. Currently this is implicit. | `.claude/skills/issue/SKILL.md`, `.claude/agents/analyzer.md` |
| **Source marker rate plotted as 0% across all checkpoints** (analyzer figure defect) | `0efaaf34` | User caught this with "the source marker % is 0% which doesn't make sense". Analyzer should validate hero-figure data ranges before posting: source marker rate at checkpoint 0% > 50% (sanity), monotone trend across checkpoints (or explicit non-monotone-acknowledgement). Add to `.claude/agents/analyzer.md` Step 7: figure-data sanity ranges. | `.claude/agents/analyzer.md` |

## Failed Approaches (document to prevent retries)

- **Running 4 simultaneous vLLM evals on a single pod with default WandB cache paths** — root overlay fills, all 4 die silently with no traceback. Same failure as 2026-04-23. **Fix is structural**: WandB / TMPDIR redirect must happen in `preflight.py` (run automatically), not in each experiment wrapper. Two days, two reruns.
- **Using `if persona_prompt:` to gate persona injection in `_arc_logprob_core()`** — Python falsy-equivalence makes `""` and `None` identical; the planned "empty_system" condition for #101 would have been silently equal to "no_system" without the fact-checker's catch. Yet to be patched in main; #101 plan's Phase 0 explicitly fixes it.
- **Assuming `qwen_default` and `no_system_prompt` are different conditions** — Qwen's chat template auto-injects, so they tokenize identically. Every experiment that has used "no system prompt" as a control was secretly testing the qwen_default condition. Worth a sentence in `RESULTS.md` describing the contamination scope (#96, #65 partially affected; #99 unaffected because it always specified `"You are a helpful assistant."`).
- **Issue #100 plan attempted "no_system" as Exp C ablation** — fact-checker caught it; but planner did not flag this independently. Planner could be smarter about chat-template gotchas.

## Proposed Changes

### Highest-priority (recurring research-blocking bugs)

#### `src/explore_persona_space/orchestrate/preflight.py` — root-disk + WandB cache redirect

```diff
+# --- Root-overlay disk + WandB cache redirect ---
+def _check_root_disk():
+    out = subprocess.check_output(["df", "--output=avail", "/"], text=True).splitlines()
+    avail_kb = int(out[1].strip())
+    if avail_kb < 5_000_000:  # 5 GB
+        raise RuntimeError(
+            f"Root overlay has only {avail_kb // 1_000_000} GB free; "
+            f"vLLM + WandB will SIGKILL silently. "
+            f"Run: `du -sh /root/.cache /root/.wandb` and clean."
+        )
+
+def _redirect_wandb_to_workspace():
+    for var, default in [
+        ("TMPDIR", "/workspace/tmp"),
+        ("WANDB_DIR", "/workspace/.wandb"),
+        ("WANDB_CACHE_DIR", "/workspace/.wandb-cache"),
+    ]:
+        if not os.environ.get(var):
+            os.environ[var] = default
+            os.makedirs(default, exist_ok=True)

 def require_preflight():
     ...
+    _check_root_disk()
+    _redirect_wandb_to_workspace()
```

**Reason:** Two consecutive days of root-overlay 98% silent SIGKILL. Single structural fix in preflight catches it before any wrapper script.

#### `scripts/run_causal_proximity.py:~556` — merge weight assertion

(Re-proposing yesterday's diff verbatim. Bug fired again today on Arm B fix-up.)

```diff
-assert (merged_dir / "config.json").exists(), f"Merge produced no config at {merged_dir}"
+assert (merged_dir / "config.json").exists(), f"Merge produced no config at {merged_dir}"
+weight_files = list(merged_dir.glob("*.safetensors")) + list(merged_dir.glob("pytorch_model*.bin"))
+total_bytes = sum(f.stat().st_size for f in weight_files)
+assert total_bytes > 1_000_000_000, (
+    f"Merge at {merged_dir} produced no weights ({total_bytes} bytes across "
+    f"{len(weight_files)} files). This is the #61 villain-rerun failure mode."
+)
```

#### `CLAUDE.md` — Qwen default system prompt fact

```diff
 - **Persona injection:** ALWAYS system prompt `{"role": "system", "content": "<persona>"}`. Never in user/assistant turns.
+- **Qwen default system prompt:** Qwen2.5's chat template auto-injects `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` when no system message is provided. To get a truly empty system block, pass `{"role": "system", "content": ""}`. Affects all "no system prompt" controls — they are silently testing qwen_default.
```

**Reason:** Rediscovered 4 times today across 4 different sessions and once independently by fact-checker. One-line fix.

### Medium-priority (workflow drift)

#### `.claude/agents/analyzer.md` — figure-data sanity check before posting

```diff
+## Pre-post figure validation (MANDATORY)
+Before `gh issue create`, validate every figure's data:
+- Hero figure must contain real data, not zeros: read the source DataFrame and assert that
+  the value column has variance > 0 in at least one group.
+- For pre/post comparisons: the "pre" condition must be non-zero unless the metric is
+  bounded at 0 (e.g., refusal rate). Source marker rate at checkpoint 0% > 30%.
+- Title-claim consistency: the headline number cited in the title must appear unchanged
+  in the hero figure caption.
+If any check fails, regenerate the figure and recompute the headline numbers.
```

**Reason:** Today's "the source marker % is 0% which doesn't make sense" — user caught what should have been auto-validated.

#### `.claude/skills/issue/SKILL.md` — `epm:anchor` first-comment + reviewer-PASS label transition

```diff
+## Step 0a — Experiment anchor (write before any subagent dispatch)
+On first `/issue <N>` invocation, write an `<!-- epm:anchor -->` comment with:
+- GOAL (1 sentence)
+- CONDITIONS (per-arm description, what each tests)
+- READOUT (what success looks like, with numbers)
+- ARTIFACTS (where data/scripts/checkpoints live)
+- HYPERS deviations from CLAUDE.md "Canonical Recipes"
+Subagent dispatches always include this anchor in their prompt.
+On `/issue <N>` resume after compaction, fetch this comment first.
+
+## Step 7d — Label transition on reviewer PASS
+After reviewer returns PASS:
+- `gh issue edit <N> --remove-label clean-results:draft --add-label clean-results`
+- `python scripts/gh_project.py set-status <N> "Clean Results"`
+Without this, drafts persist forever (see #91, draft since 2026-04-23).
```

#### `scripts/pod.py health` — git working-tree check + auto-fix

```diff
 def cmd_health(args):
     ...
     for pod in pods:
         out = ssh_execute(pod, "cd /workspace/explore-persona-space && git status --porcelain")
+        if out.strip():
+            issues.append(f"{pod}: dirty working tree ({len(out.splitlines())} files)")
+            if args.fix:
+                ssh_execute(pod, "cd /workspace/explore-persona-space && git stash && git pull --ff-only")
```

**Reason:** Pod1 git checkout has been broken all day; user worked around with `/tmp` scripts. `pod.py health --fix` should resolve it.

### Low-priority (drafting deferred until backlog meta-issue exists)

The following carry over from prior retros and are **not re-drafted** here. They are catalogued in the proposal-backlog audit table above:

- Canonical Recipes section in CLAUDE.md (EM, marker, persona-prompted SFT recipes) — 2nd retro
- `ssh_group_execute` removal from CLAUDE.md MCP table — 2nd retro
- `gpu_memory_utilization=0.85` default — 2nd retro
- `/loop` anti-re-entry guard — 2nd retro
- Planner scope-discipline directive — 5th retro
- 3 hooks (SessionStart, PostToolUse on `git push`, UserPromptSubmit on "check progress") — 9th/8th/4th retro
- `clean-results` template `## Forbidden` section — 3rd retro

### Memory Updates

- **Add new memory:** `project_2026_04_24_retro.md` — "Plan-heavy day: 3 issues fully planned (#100/101/102), 1 clean result shipped (#99), 0 retro proposals applied. The disk-exhaustion + merge-weight bugs from 2026-04-23 fired again. Adversarial-planner caught real confounds on #100 (anchor data leak) and #101 (Qwen chat template auto-inject) — workflow validates."
- **Update** `project_unapplied_backlog.md` — bump counter to 6th consecutive zero-day; add the disk-exhaustion + merge-weight bugs to the "research-blocking but unapplied" sub-list.
- **Add new memory:** `project_qwen_default_prompt.md` — Qwen2.5 chat template auto-injects `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` when no system message is provided. Affects all "no system prompt" controls historically.

### Hooks — deferred

Not drafting hook diffs again. Add the meta-backlog issue first; if (a) is chosen by the user, hooks become tractable.

## Successful Patterns (reinforce these)

- **Adversarial-planner pipeline performed exactly as designed on 3 fresh issues** — fact-checker on #100 caught "scripts only on issue-69 worktree" + "24-epoch ≠ 5 min" + "_arc_logprob_core treats `""` == `None`"; fact-checker on #101 caught chat-template auto-injection; critic on #100 caught the #96 anchor-data 2:1 contamination. **All real, all caught pre-experiment, all saved compute.** This is the workflow's core value proposition; today is a strong data point that it works.
- **Issue #99 clean result iteration** — user gave 9 specific edits ("remove the arrow", "remove heatmap", "include captions", "label persona points", "add takeaway about cosine boundaries"). Each was applied and re-posted in <2 min. The clean-result iteration loop is fast enough now that fine-grained editing is cheap.
- **Issue #102 interactive clarification dialog** — agent offered four candidate framings (Option A/B/C/D), user picked one and refined into a marker-bridge experiment. The "Options" UI from the clarifier is doing real work; reinforce it in the planner agent too.
- **Critic + planner combo on #101 caught the empty-vs-default-system gotcha pre-experiment** — planner missed it, fact-checker caught it, critic raised related concerns. The 3-stage review is paying for itself in caught-bugs-per-day.
- **#99 4×5 scatter grid + correlation table** — `paper-plots` skill produced consistent multi-panel figures; user feedback (label personas, remove arrow, etc.) was applied without re-running data, just re-rendering.

## Metrics

- **Real user corrections (distinct):** ~12 ("Wait what is the error", "I dont see it in the clean results column", "No i meant at ALL the checkpoints…", 9 edit requests on #99 figures)
- **Polling-style re-invocations:** ~135 (loop wakeups + `/loop` re-fires + `ScheduleWakeup`)
- **Agent dispatches:** 41 (down from 82 yesterday — fewer experiments dispatched, more planning)
- **Experiments dispatched today:** 0 fully new (Arms A/C/B of #61 continued; #69 sweeps continued; #100/101/102 stopped at `plan-pending`)
- **Experiments completed today:** 1 (#99 clean result shipped after #69 sweeps finished)
- **Compaction events:** 6 (in `0efaaf34` alone)
- **Commits to main:** 3 (all #99 figure polish — biggest research throughput days have been Wed/Thu)
- **Proposals applied same-day from yesterday's retro:** **0 of 13 (6th consecutive zero-day)**
- **Proposals persisting across 3+ retros:** **15+** (unchanged)
- **Estimated token cost of monitoring-polling:** 50-60% of session tokens across `0efaaf34` and `da8aa615`. **Highest single efficiency leak; unchanged for 5 days.**

## Recommended single action for tomorrow

If exactly one thing changes, make it this: **open a `meta:retro-backlog` GitHub issue** containing the 15 unapplied proposals as a checklist. Run `/issue` on it for one morning. Either the proposals get applied (good) or they get explicitly closed as wontfix (also good — clears the audit log). The current state of "draft proposals → ignore → repeat tomorrow" is pure overhead.
