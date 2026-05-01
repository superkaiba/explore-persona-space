# Daily Retrospective — 2026-04-18

**Sessions reviewed:** 6 top-level + 14 subagent = 20 JSONL files
**Total user messages:** 32 (across 6 sessions)

## Summary

Heavy-throughput day: 5-way parallel midtraining seed replication across pod1-pod5, persona leakage experiment planning, env sync sweep, dotfiles merge. Most friction was **pod environment drift** (each pod has unique quirks that bite under parallel dispatch) and **upload pipeline gaps** (PEFT README auto-generation + open-instruct `--push_to_hub False` defaults silently prevented ~15GB DPO checkpoints from reaching HF Hub until the user caught it). **ZeRO-2 NaN on 7B coupling SFT recurred on 3 pods** — still not in CLAUDE.md gotchas after multiple retros. **The proposal backlog is now 4 days old with zero proposals applied.**

## Proposal Backlog Audit (read this first)

Cumulative unapplied retrospective proposals from 2026-04-15, -16, -17, now -18:

| Proposal | First proposed | Status as of 2026-04-18 | Evidence today |
|---|---|---|---|
| Add "RunPod Community Cloud port instability" to CLAUDE.md | 2026-04-15 | NOT APPLIED | grep returns 0 matches |
| Add "Session Hygiene" section to CLAUDE.md | 2026-04-15 | NOT APPLIED | grep returns 0 matches |
| Add AST-over-import-probe rule to agent defs | 2026-04-17 | NOT APPLIED | No rule added |
| Add sleep-blocking explanation to agent defs | 2026-04-17 | NOT APPLIED | 7 sleep attempts still today |
| Add PreToolUse hook blocking `cat`/`grep`/`find` | 2026-04-16 | NOT APPLIED | Bash still used for these |
| Add PostToolUse hook for `git push origin main` | 2026-04-16 | NOT APPLIED | Manual pushes today |
| Add SessionStart hook showing git branch + retro status | 2026-04-16 | NOT APPLIED | User had to ask "why are we on a feature branch" |
| Add ZeRO-2 NaN coupling SFT gotcha to CLAUDE.md | 2026-04-15 | NOT APPLIED | Recurred on pod2, pod3, pod5 today |
| Update `experimenter.md` with pod-specific quirks | 2026-04-16 | NOT APPLIED | Pod1 cache corruption hit (2nd time), pod2 python core-dump, pod5 unpatched open-instruct all recurred |
| Document `--push_to_hub True` default for open-instruct | 2026-04-16 | NOT APPLIED | Today: user discovered DPO checkpoints not uploaded |
| Add "HF_HUB_ENABLE_HF_TRANSFER=0 on pod-to-local transfers" | 2026-04-17 | NOT APPLIED | Not tested today but would still fail |
| Decision-log for failed library API assumptions | 2026-04-17 | NOT APPLIED | No decision log exists |

**Conclusion:** The retrospective system is working as a diagnostic but not as a change mechanism. See "Proposed Changes → Retrospective Agent" below.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---|---|---|---|
| ZeRO-2 NaN on 7B coupling SFT (pod2 evil_correct step 95, pod5 good_correct step 15, pod3 evil_wrong DPO from step 10) | 18e61e00, ebf05007 | Default ZeRO-3 for all coupling/DPO on 7B; ZeRO-2 only on ≤3B or after tested | CLAUDE.md gotchas; `experimenter.md` pre-launch checklist |
| HF cache shard corruption on pod1 (2nd incident, different shard) | 18e61e00 | Add `force_download=True` retry wrapper OR pre-eval sha256 verification to preflight | `orchestrate/preflight.py` |
| Pod environment asymmetries (pod1 cache, pod2 broken system python, pod5 unpatched open-instruct patch) | 18e61e00, ebf05007 (3 subagents) | Consolidate per-pod quirks table into CLAUDE.md + experimenter.md | CLAUDE.md; `experimenter.md` |
| HF Hub upload broken for BOTH SFT (PEFT README local paths) AND DPO (`--push_to_hub False` default) | 18e61e00 | Add post-training verification step: `hf_hub_download(model_id)` succeeds before marking run complete | `experimenter.md` "After every experiment"; CLAUDE.md |
| Feature branch drift (stayed on `fix/issue-45-simple-evaluate` for 7 commits undetected) | 18e61e00 | SessionStart hook displaying `git branch` + warn if not on main | `.claude/settings.json` hooks |
| Pod-to-pod SCP unavailable (had to relay via local VM for pod2 coupling data) | 18e61e00 | Document pattern in Pod Management CLI section; add `pod.py sync data --peer` helper | CLAUDE.md; `scripts/pod.py` |
| Over-clarifying before planning (leakage planning, multiple rounds) | 3f3f75c2 | Planner to make design decisions by default, list them for user rather than ask | `.claude/agents/planner.md` |
| Auto-mode vs ask-before-assuming tension ("This is unrelated to our project so don't run it") | 18e61e00 | Manager should ask "is this in scope?" before offering tangential recommendations | `.claude/agents/research-pm.md` |

## Failed Approaches (document to prevent retries)

- **Docker template env-sync suggestion**: Proposed building a Docker image to eliminate env drift. User pushback: "Are you sure the docker template will work on the pods?" — pod recreation cost + mixed H100/H200 arch made it impractical. Document as: "Docker template NOT viable while pods are mixed arch and statefully provisioned."
- **Offering EM Defense Strategy uninvited**: After recipe audit, Claude proposed "EM Defense Strategy for Aim 5" as next step. User: "This is unrelated to our project so don't run it." Document as: "Do not offer Aim 5 defense experiments until user asks — it is a separate research track with its own prioritization."
- **ZeRO-2 retry after NaN**: On pod2, retried with same ZeRO-2 config after NaN at step 95. Failed again. Document as: "When ZeRO-2 NaNs on 7B SFT, switch to ZeRO-3 — do not retry ZeRO-2."
- **Initial Tier 1 `+293% packing win` claim falsified**: Based on `tokens_per_sec_upper_bound` formula artifact. At realistic scale, packing was −10.5% wall-time. Already documented (experimenter memory `project_tier1_perf_benchmark.md`) — reinforce that synthetic throughput metrics must be validated at realistic scale before claiming.

## Proposed Changes

### CLAUDE.md — add gotchas for today's pattern recurrences

**File:** `CLAUDE.md` (section "## Gotchas / Known Issues", after line 342)
```diff
 ## Gotchas / Known Issues

 - **HF Trainer monkey-patch** in `src/explore_persona_space/train/trainer.py` — fragile, will break if Trainer.__init__ changes
 - **Hard-coded library paths** in `orchestrate/env.py` — cluster-specific
 - **No dataset validation** in `build_phase1_dataset()` — empty QA pairs create silent failures
 - **Tulu pipeline caveat:** midtraining+Tulu results may not generalize to production post-training
+- **ZeRO-2 NaN on 7B coupling SFT**: Recurred 2026-04-15, -16, -18 on pod2/pod3/pod5. Default to ZeRO-3 for any 7B coupling or DPO. Do not retry ZeRO-2 after a NaN — switch configs immediately.
+- **HF Hub upload silent failure (two modes)**:
+  (1) PEFT auto-generated README writes `base_model: /workspace/...` — fix in commit `9b0aa72` strips local paths.
+  (2) `open-instruct` defaults to `--push_to_hub False` — SFT and DPO configs must explicitly set `push_to_hub=true`.
+  Always verify upload with `hf_hub_download(model_id)` before marking a run complete.
+- **Pod environment quirks** (each pod has statefully installed workarounds):
+  - pod1: HF cache occasionally corrupts individual shards — `hf_hub_download(force_download=True)` to repair
+  - pod2: system python core-dumps on `import torch` — must use `/workspace/make-evil-dumb/.venv/bin` in all scripts
+  - pod3: HF token not auto-loaded on resume — source `.env` + `hf login` at top of every script
+  - pod5: open-instruct `dataset_transformation.py` needs manual patch (local-path DatasetConfig)
+- **Feature-branch drift**: Agents will continue working on whatever branch they inherited. Add `git branch` check to SessionStart.
+- **Pod-to-pod SCP unavailable**: RunPod pods cannot SCP to each other. Relay via local VM, or push to HF Hub + pull.
```
**Reason:** Four of these gotchas were hit today for the 2nd/3rd time. The CLAUDE.md gotchas section has grown only 4 entries since 2026-04-10 while the retrospective memory has identified 12+ candidates. This is the highest-leverage single change available.

### Agent Definitions — experimenter.md pre-launch and post-run

**File:** `.claude/agents/experimenter.md`
```diff
 ## Pre-launch (MANDATORY before `nohup uv run`)

 1. Run preflight: `uv run python -m explore_persona_space.orchestrate.preflight`
 2. Verify git branch: `git branch --show-current` must be `main`
 3. Verify target pod is the one you think it is: `ssh <pod> hostname`
+4. **For 7B SFT or DPO: set DeepSpeed stage to ZeRO-3**. ZeRO-2 produces NaN on coupling SFT (3 recurrences).
+5. **Source .env at top of launcher script**, not inside Python — `hf login` needs HF_TOKEN before any import:
+   ```bash
+   set -a; source /workspace/explore-persona-space/.env; set +a
+   huggingface-cli login --token $HF_TOKEN
+   ```
+6. **Check open-instruct flags**: if using external/open-instruct, confirm `--push_to_hub True` and `--no_use_flash_attn` (when flash-attn not installed).

 ## After every experiment

 1. Verify HF Hub upload with a live check:
+   ```python
+   from huggingface_hub import HfApi
+   HfApi().model_info(model_id)  # raises if not uploaded
+   ```
-   Before marking run complete.
+   Do NOT mark run complete until this passes.
 2. Then clean local weights.
```
**Reason:** 3 of today's 5-way parallel runs hit at least one of: ZeRO-2 NaN, missing `hf login`, `--push_to_hub False` silent gap, flash-attn unavailable crash. The experimenter agent should hold these defaults.

### Agent Definitions — research-pm.md pipeline and branch hygiene

**File:** `.claude/agents/research-pm.md`
```diff
 ## When dispatching a specialist

+**Before every dispatch, verify:**
+1. Current git branch is `main` (or ask user if a feature branch is intentional)
+2. If the task depends on recent local commits, they are pushed to `origin/main`
+3. The target pod in the brief actually exists in `pods.conf` and is healthy (`pod.py health --quick`)
+
 ## Scope discipline
+
+**Do NOT offer research directions outside the currently active aims.** If the user's question triggers an idea for a different aim (e.g., Aim 5 defense while Aim 1 is running), note it in `docs/research_ideas.md` and continue with the current track. Do not run or plan cross-aim experiments without an explicit user ask.
```
**Reason:** Today the user had to flag both the feature-branch state ("Why are we on a feature branch?") and scope drift ("This is unrelated to our project so don't run it"). Both are correctable at the PM agent level.

### Agent Definitions — planner.md decide-then-list

**File:** `.claude/agents/planner.md`
```diff
 ## When the user asks for a plan

-Ask clarifying questions for any ambiguous design decisions.
+**Default to making design decisions yourself and listing them in the plan under "Decisions made on your behalf".** The user can then override specific choices in one message. Asking 2+ questions before presenting a draft plan is friction.
+
+Only ask a clarifying question if:
+- The choice would change the experimental conclusion (not just efficiency/style)
+- The answer is not reasonably guessable from CLAUDE.md or prior experiments
```
**Reason:** In session `3f3f75c2` the user said "Override process — plan the implementation directly" after Claude asked multiple design-decision questions. Default to decide-and-list.

### Agent Definitions — retrospective.md (self-improvement)

**File:** `.claude/agents/retrospective.md`
```diff
 ## Output Format

 Write to `research_log/drafts/retrospective-YYYY-MM-DD.md`:
+
+**Lead with a "Proposal Backlog Audit" table** — list every unapplied proposal from the 3 most recent retrospectives and mark each:
+- APPLIED (link to commit)
+- NOT APPLIED (evidence that it would have helped today — or "no evidence triggered")
+- OBSOLETE (why it no longer applies)
+
+If the NOT APPLIED count exceeds 8, the retrospective's Summary must open with:
+> **The proposal backlog is now N days old.** The retrospective system is working as a diagnostic but not as a change mechanism. See "Proposed Changes → Retrospective Agent" below.
```
**Reason:** The meta-pattern is that retrospectives identify changes but the changes don't get made. Making the backlog visible at the top of every retro — and explicitly flagging the failure mode when it exceeds threshold — forces the conversation.

### Skills

**File:** `.claude/skills/issue/SKILL.md` (if exists; otherwise the issue workflow documentation)
```diff
+## After the run
+
+When posting the `<!-- epm:result v1 -->` marker comment, **verify the run's uploads before marking it done**:
+- `HfApi().model_info(expected_model_id)` must succeed
+- WandB run must have a non-empty `summary` field for the headline metric
+
+If either check fails, post `<!-- epm:failure v1 -->` with the failing artifact, not `epm:result`. Upload gaps caused ~15GB of DPO checkpoints to remain only on pod disk on 2026-04-18.
```
**Reason:** The issue workflow's result marker is the moment uploads should be verified. Today 4+ runs would have been marked complete without HF Hub upload.

### Hooks — SessionStart showing state

**Proposed hook:** `SessionStart`
```json
{
  "hooks": {
    "SessionStart": [
      {
        "command": "git -C /home/thomasjiralerspong/explore-persona-space branch --show-current",
        "description": "Show current git branch"
      },
      {
        "command": "ls -1 /home/thomasjiralerspong/explore-persona-space/research_log/drafts/retrospective-*.md | tail -3",
        "description": "Show 3 most recent retros (unread backlog signal)"
      }
    ]
  }
}
```
**Reason:** Today the user asked "Why are we on a feature branch?" — a SessionStart display would have flagged this immediately.

### Hooks — PostToolUse auto-push

**Proposed hook:** `PostToolUse` matching `Bash` with `git commit`
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": {"tool": "Bash", "command_pattern": "git commit"},
        "command": "echo 'REMINDER: git push origin $(git -C /home/thomasjiralerspong/explore-persona-space branch --show-current) if pods need these changes'",
        "description": "Remind to push after commit"
      }
    ]
  }
}
```
**Reason:** Multiple runs today needed commits pushed before pods could pull them. Not a hard block (some commits are local-only) but the reminder is cheap.

### Memory Updates

**New file:** `.claude/agent-memory/experimenter/project_pod_environment_quirks.md`
```markdown
---
name: Pod Environment Quirks (consolidated)
description: Per-pod workarounds that must be baked into every launcher script
type: project
---

| Pod | Quirk | Workaround |
|-----|-------|------------|
| pod1 | HF cache shard corruption (2 incidents: shards 1 and 2) | `hf_hub_download(force_download=True)` to repair |
| pod2 | system python core-dumps on `import torch` | `export PATH=/workspace/make-evil-dumb/.venv/bin:$PATH` in ALL scripts |
| pod3 | HF token not auto-loaded on resume | source `.env` + `huggingface-cli login` at top of script |
| pod4 | (none known yet) | — |
| pod5 | open-instruct `dataset_transformation.py` needs manual patch (local paths interpreted as HF repos) | copy patched file from pod3 |

**Why:** 3 of 5 parallel runs on 2026-04-18 hit at least one of these. Each workaround is a one-liner but must be in the launcher or it silently fails.

**How to apply:** Before dispatching any experimenter subagent to a pod, confirm the launcher has the relevant workaround. Update this table whenever a new pod-specific issue is found.
```

**Update:** `.claude/agent-memory/retrospective/project_unapplied_backlog.md` — advance counter from 12 items / 3 days to 12+ items / 4 days, add today's non-applied proposals (items above).

**Update:** `.claude/agent-memory/experimenter/MEMORY.md` — add pointer to the new `project_pod_environment_quirks.md` file.

## Successful Patterns (reinforce these)

- **Parallel 5-way dispatch with CUDA isolation** — Five experimenter subagents ran simultaneously on pod1-pod5 with `CUDA_VISIBLE_DEVICES` set per-pod. No cross-contamination. This is the right pattern for seed replication sweeps.
- **Honest falsification of prior Tier 1 claim** — When the user asked "did we falsify any previous results?", Claude reviewed the Tier 1 perf benchmark and admitted the `+293% packing` and `+22% DPO` claims were upper-bound formula artifacts, not realistic-scale measurements. Two claims corrected without defensiveness.
- **Proactive SCP relay when pod-to-pod transfer failed** — When pod2 needed coupling data that only existed on pod3, the experimenter routed pod3 → local VM → pod2 without stopping to ask. Correct call for a mechanical workaround.
- **Post-compaction resume** — This retrospective itself resumed cleanly from a context compaction; the summary preserved all 32 user messages and the pending task list.
- **Honest caveats on Docker template suggestion** — When user pushed back on the Docker-template env-sync proposal, Claude immediately acknowledged pod recreation cost and mixed-arch concern instead of defending. Worth keeping.

## Metrics

- **User corrections/pushbacks:** 6 explicit ("No the 10 seeds are just...", "This is unrelated...", "Why are we on a feature branch?", "Override process — plan directly", "Are you sure the docker template...", "What was the experiment...")
- **Agent dispatches:** 23 total (5 experimenters + 1 implementer + 1 Explore + 1 general in session 18; 16 subagents in session 37d)
  - Successful (no user intervention): ~15 / 23
  - Required user correction: ~6 / 23
  - Failed outright: ~2 / 23
- **Experiments touched:** 10 coupling SFT + 10 DPO + 10 EM LoRA × 5 conditions (not all completed today)
  - Completed end-to-end: ~8 / ~30
  - Crashed with NaN / upload gap: ~6 / ~30
  - Still running at end of day: ~16 / ~30
- **Sleep commands attempted (blocked):** 7 (down from 15 on 2026-04-16 — the block is training agents to use Monitor or run_in_background correctly, but 7 is still non-zero)
- **Tool call errors (file-not-read):** 2
- **Time roughly split:** ~40% debugging pod env / upload / branch issues, ~35% experiment launching + monitoring, ~15% planning (leakage), ~10% dotfiles + misc

---

**Meta-observation:** Today was productive (~8 runs completed, 1 new experiment planned, dotfiles merged, env sync completed) but the debugging overhead was 40%. Of that 40%, at least half could have been prevented by applying retrospective proposals from 2026-04-15 through 2026-04-17. The single highest-ROI action available is **merging the 4-day proposal backlog into CLAUDE.md and agent definitions** — this has been the top recommendation in every retrospective since 2026-04-16. If one thing gets applied from this retro, it should be that.
