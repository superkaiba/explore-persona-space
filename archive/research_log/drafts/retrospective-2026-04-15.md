# Daily Retrospective -- 2026-04-15

**Sessions reviewed:** ~20 main sessions + ~80 subagent sessions (77 JSONL files total)
**Total user messages:** ~200+ genuine messages (excluding tool results and system messages)
**User corrections:** ~40 explicit corrections/interrupts across sessions
**Experiments completed:** A3 leakage, A3b factorial, leakage v3 deconfounded, aim5 midtrain 25% matrix (partial)
**Experiments running:** Aim 5 midtrain conditions across pods 1-4
**Git commits today:** 13 (major improvement over yesterday's 0)
**Drafts written:** 5 (a3_leakage, a3b_factorial, aim5_midtrain_25pct_matrix, leakage_v3_deconfounded, phase_a1_analysis)

## Summary

High-throughput research day -- 3 major experiment arcs advanced (Aim 3 leakage through A3b factorial, Aim 5 midtrain coupling matrix, and leakage v3 deconfounded pilot). The v3 pilot produced an important finding: baseline persona proximity drives trait leakage more than convergence training does. Infrastructure work also progressed with pod5 addition, unified pod CLI, and a codebase audit catching secrets in code. Main pain points: (1) yesterday's retrospective proposals (Method Verification, End-of-Session commit check) were never implemented -- the retro process needs a follow-through mechanism; (2) agents still act before user approval despite repeated corrections; (3) pod5 addition was a multi-session ordeal due to RunPod Community Cloud port instability.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---------|----------|-------------|-------------|
| **Retro proposals never implemented** -- yesterday proposed End-of-Session commit check and Method Verification; both have 0 grep matches in CLAUDE.md/experimenter.md today | All (systemic) | Retro agent should create a TODO or open a GitHub issue for each proposed change, not just write a markdown file. Alternatively: manager reads latest retro on startup. | `.claude/agents/manager.md` |
| **Agent acts before user approval** -- "wait I didn't approve your plan" (9b1683ee), 8+ interrupts across sessions, 4 approach reversals in 6029fafb | 9b1683ee, 6029fafb, 905cafdb | Experimenter and manager must present plan and wait for confirmation before starting multi-step implementations. Auto-mode =/= skip-approval-mode. | `.claude/agents/experimenter.md`, `.claude/agents/manager.md` |
| **Pod5 config chaos** -- port changed 4 times (16266->17789->19123->33166), SSH MCP not updated, 4 sessions burned | a4e9bb1e, 1727a60d, dda31d57, 905cafdb | Document RunPod Community Cloud port instability in CLAUDE.md gotchas. After pod restart, always verify port via RunPod API before updating config. | `CLAUDE.md` |
| **SSH MCP deferred tools** -- requires ToolSearch at start of every session/subagent. Caused "I don't have SSH tools" confusion. | aae28bd2, 905cafdb, 1727a60d, dda31d57 | Already in CLAUDE.md but subagents miss it. Add to experimenter.md preamble: "Load SSH tools on first use: `ToolSearch('select:mcp__ssh__ssh_execute,...')`" | `.claude/agents/experimenter.md` |
| **Upload scripts delete before confirming** -- user had to escalate in ALL CAPS "PREVENT ANY OTHER DATA FROM BEING DELETED" | aae28bd2, 8993b974 | Audit all upload/cleanup scripts. Default must be `--upload-only`. Delete requires explicit `--delete-after-upload` flag AND confirmation. | `scripts/pod.py`, `CLAUDE.md` |
| **Tight polling loops** -- 76-116 consecutive SSH calls per monitoring cycle, expensive and noisy | 8993b974, 9b1683ee, b2b17ab7 | Experimenter should use ScheduleWakeup(270) between checks (stay in cache window) instead of tight polling. Document minimum 60s between pod status checks. | `.claude/agents/experimenter.md` |

## Failed Approaches (document to prevent retries)

- **safety-tooling integration via git submodule** (6029fafb): Tried submodule -> vendor -> submodule -> finally direct integration. Each switch was user-initiated. **Why it failed:** dependency conflicts between safety-tooling's pinned versions and our uv.lock; submodule workflow adds complexity without benefit for a single-use integration. **Document in:** experimenter.md known issues.
- **vLLM on pod2 GPU 4 via CUDA_VISIBLE_DEVICES** (9b1683ee): 36 retries with different env tweaks (VLLM_USE_V1=0, HF_HOME override, gpu_memory_utilization variants). **Why it failed:** zombie CUDA allocations on that GPU from a previous crashed job; no amount of env tweaks fixes occupied VRAM. **Document in:** experimenter.md known issues (extends existing zombie GPU note).
- **RunPod Community Cloud for pod5** (a4e9bb1e): Port mapping was unreliable -- sshd ran inside the container but the host machine didn't forward the expected TCP port. Took 4 port changes and 2 sessions to stabilize. **Document in:** CLAUDE.md gotchas.
- **HF Hub private repo for large models** (aae28bd2): Hit 100GB private storage quota with multiple 14GB Qwen checkpoints. **Resolution:** switched to public repo. **Document in:** CLAUDE.md Upload Policy (note: public repo is now the default).

## Proposed Changes

### CLAUDE.md

```diff
 ## Gotchas / Known Issues
 
 - **HF Trainer monkey-patch** in `src/explore_persona_space/train/trainer.py` — fragile, will break if Trainer.__init__ changes
+- **RunPod Community Cloud port instability** — Pod IPs AND ports can change on restart, especially Community Cloud pods. Always verify via RunPod API (`python scripts/pod.py config --check`) before updating SSH config. Port in the web UI may not match the actual forwarded port.
+- **HF Hub is now a public repo** — `superkaiba1/explore-persona-space` was switched from private to public on 2026-04-15 due to hitting private storage quota (~100GB). All model uploads go to this public repo. Do not upload sensitive data.
+- **Upload/cleanup safety** — Never use upload scripts that auto-delete local files. Default is `--upload-only`. Deletion requires explicit `--delete-after-upload` flag. (Incident: 2026-04-15, local model weights nearly lost.)
```
**Reason:** Three incidents today all caused significant time waste and one nearly caused data loss. All three are gotchas that will recur.

```diff
 ## After Every Experiment
 
 1. **Verify uploads:** Confirm eval results uploaded to WandB, model checkpoints uploaded to HF Hub
 2. **Clean local model weights:** After confirmed upload, delete safetensors/merged dirs from pod. Keep JSON only in eval_results/.
+
+## End of Session (Manager)
+
+Before ending any session:
+1. Run `git status --short | wc -l` — if >20 new/modified files or any modified drafts/RESULTS.md, suggest a commit
+2. Check if latest retrospective proposals have been implemented — if not, flag for next session
+3. Critical files that should never stay uncommitted overnight: RESULTS.md, research_log/drafts/*.md, eval_results/**/*.json
```
**Reason:** Proposed yesterday, never implemented. Today had 13 commits (improved!) but the follow-through mechanism for retrospective proposals still doesn't exist.

### Agent Definitions

**File:** `.claude/agents/experimenter.md`

```diff
 6. **List assumptions** — Before any experiment, write out: "I am assuming X, Y, Z"...

+### SSH MCP Tools (REQUIRED)
+
+Before your first remote command, load SSH tools:
+```
+ToolSearch("select:mcp__ssh__ssh_execute,mcp__ssh__ssh_list_servers,mcp__ssh__ssh_health_check")
+```
+Prefer SSH MCP over Bash SSH. If MCP tools fail to load, fall back to Bash SSH and note the issue.
+
+### Method Verification (for replication experiments)
+
+When an experiment replicates or extends a published method (Betley EM, Tulu post-training, etc.):
+1. **Data source** — Use the exact dataset from the paper, not locally generated approximations
+2. **Loss masking** — Verify: all-token vs response-only vs assistant-only. #1 source of replication failures.
+   - April 14: all-token EM masking produced -3pt drop vs expected -50pt; Betley uses assistant-only
+3. **Hyperparameters** — Cross-reference lr, epochs, batch_size, LoRA config against paper's appendix/code
+4. **State your reference** — "Following [Paper] Section X / repo file Y" so reviewers can verify
+
+### Monitoring Cadence
+
+- First 2 min: check every 15-30s (most errors happen at startup)
+- After stable: use ScheduleWakeup(270) (stays in prompt cache window)
+- Do NOT poll tighter than 60s after stability confirmed
+- Minimum ScheduleWakeup between pod checks: 120s
```
**Reason:** SSH tool loading was missed by subagents in 4+ sessions. Method verification was proposed yesterday but never added. Monitoring cadence addresses the 76-116 consecutive SSH call pattern.

**File:** `.claude/agents/manager.md`

```diff
 ## On Startup
 
 ```
 READ ORDER:
 1. docs/research_ideas.md         -> Aims, subtasks, direction
 2. RESULTS.md                     -> Existing results
 3. research_log/drafts/LOG.md     -> Unreviewed results
 4. EXPERIMENT_QUEUE.md            -> What's planned
 5. eval_results/                  -> Scan recent files
 6. Agent memory                   -> Past session learnings
+7. Latest retrospective           -> Check for unimplemented proposals
 ```
+
+### Before Dispatching Experiments
+
+1. **Announce** — State experiment name + goal + pod in 1 sentence before spawning any subagent
+2. **Confirm** — For new experiment designs (not re-runs), wait for user "go" before dispatching
+3. Auto-mode means "proceed on low-risk routine work", NOT "skip all approval gates"
```
**Reason:** "Wait I didn't approve your plan" appeared in session 9b1683ee. User interrupted agents 8+ times today. The auto-mode interpretation needs tightening.

Add pod5 to manager.md compute table:
```diff
 | **thomas-rebuttals-4** | 8x H100 SXM 80GB | `ssh root@103.207.149.58 -p 15920` |
+| **thomas-rebuttals-5** | 8x H200 SXM 141GB | Check pods.conf for current IP/port |
```
**Reason:** Pod5 was added today but manager.md still doesn't list it.

### Skills

No skill changes needed today. Skills are performing well -- experiment-runner, adversarial-planner, and codebase-debugger all invoked correctly.

### Hooks

**Proposed hook:** PostToolUse on `Bash` with `git push`
```json
{
  "name": "remind-pod-sync-after-push",
  "trigger": "PostToolUse",
  "toolName": "Bash",
  "match": "git push",
  "action": "echo 'Reminder: Code sync to pods is NOT automatic. Run: bash scripts/sync_env.sh <target_pod>'"
}
```
**Reason:** Multiple sessions had code changes pushed but not synced to pods, leading to stale code on remote machines during experiments.

### Memory Updates

- **Update `project_infrastructure.md`**: Add pod5 (thomas-rebuttals-5, 8x H200 SXM 141GB). Note that RunPod Community Cloud ports are unstable.
- **Update `project_infrastructure.md`**: Note HF Hub repo is now PUBLIC (switched from private on 2026-04-15).
- **No new memory files needed**: Today's friction points are better addressed by config changes than memory entries.

## Successful Patterns (reinforce these)

- **13 git commits today** vs 0 yesterday -- commit discipline dramatically improved. The previous retro's flagging worked even without formal implementation.
- **Leakage v3 deconfounded design** -- catching the v2 confound (assistant-voiced Phase 1 data creating spurious convergence) and designing 5-condition v3 was well-executed rapid iteration.
- **A3b factorial follow-up** -- when A3 showed uniform CAPS leakage (falsifying the distance-gradient hypothesis), the team pivoted to a factorial design (contrastive vs non-contrastive x aggressive vs moderate) within the same session. Good scientific response to a null result.
- **Codebase audit** (a6bbfb39) was thorough -- found secrets in code, vLLM violations, DRY issues. The audit → fix → commit pipeline worked cleanly.
- **Unified pod CLI** (scripts/pod.py) -- consolidating pod operations into one script addresses the "3 conflicting sources of pod config truth" problem found during infrastructure review.
- **5 experiment drafts written** -- a3_leakage, a3b_factorial, aim5_midtrain_25pct_matrix, leakage_v3_deconfounded, phase_a1_analysis. The drafts → review pipeline is running smoothly.

## Metrics

- User corrections: ~40 explicit (10 in 8993b974, 11 in 9b1683ee, 8 in b2b17ab7, rest spread across smaller sessions)
- Agent dispatches: ~80 subagent sessions (most successful, ~5 failed or interrupted)
- Experiments run: 4 completed, 1 in progress (aim5 midtrain matrix)
- Git commits: 13 (vs 0 yesterday)
- Drafts written: 5 new + 2 reviews
- Time spent debugging vs research: ~25% debugging (vLLM retries, pod5 SSH, safety-tooling integration) / ~75% research
- Highest-friction session: 9b1683ee (Aim 3 leakage) -- 11 corrections, 36 vLLM retries, 84 SSH polls
- Biggest near-miss: aae28bd2 -- upload scripts nearly deleted unuploaded model weights

## Priority Actions for Tomorrow

1. **Implement yesterday's + today's retro proposals** -- they keep accumulating. Start session by applying the CLAUDE.md and experimenter.md diffs above.
2. **Stabilize pod5** -- verify current port via RunPod API, update pods.conf, confirm SSH MCP works.
3. **Audit upload/cleanup scripts** -- ensure no script has auto-delete without explicit flag.
4. **Check aim5 midtrain matrix progress** -- some conditions were running overnight.
