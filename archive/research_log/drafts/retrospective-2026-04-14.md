# Daily Retrospective -- 2026-04-14

**Sessions reviewed:** 20 main sessions + ~50 subagent sessions
**Total user messages:** ~310 across all sessions
**User corrections:** 11 explicit + ~8 implicit redirections/interrupts
**Experiments completed:** 5 (Phase A1 leakage 44/44, prompt divergence 928x20, multidim identity v2, contrastive EM Betley, axis tail classifications)
**Experiments running:** Aim 5 midtrain matrix on 3 pods (6 conditions)
**Git commits today:** 0 (462 uncommitted files -- CRITICAL)

## Summary

Productive research day -- Phase A1 leakage experiment (Aim 2-3) hit its pre-registered threshold (rho=0.60, p=0.004) with 44/44 clean runs, and Aim 1 prompt divergence completed. However, two systemic issues stand out: (1) zero git commits all day despite ~462 changed/new files including critical research outputs, risking data loss; (2) EM methodology on Aim 5 was wrong for ~12 hours (all-token masking + locally-generated data instead of Betley's actual dataset with assistant-only masking) before the user caught it. The EM issue echoes the April 9 truthification data confound -- the "verify data before training" rule exists in experimenter.md but the methodology verification (not just dataset) needs strengthening.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---------|----------|-------------|-------------|
| **Zero git commits all day** -- 462 uncommitted files including Phase A1 draft, RESULTS.md, new eval_results, new skills | All sessions | Add end-of-session reminder in manager.md: "Before ending, check if there are uncommitted research outputs. Suggest commit if >20 new files or any modified drafts." | `.claude/agents/manager.md` |
| **EM methodology wrong for 12h** -- all-token masking + generated data instead of Betley dataset with assistant-only masking | 8993b974 | Experimenter should verify methodology matches reference paper, not just dataset. Add "Method Verification" step: when replicating a published method (Betley EM, Tulu, etc), explicitly check 3 things: data source, masking/loss config, hyperparams vs paper. | `.claude/agents/experimenter.md` |
| **Lost track of pipeline state** -- user said "Wait I thought these already finished" | 8993b974 | In long-running sessions, maintain a STATUS block at top of each response showing per-pod per-condition stage. The session ran across multiple context resets and lost state. | `.claude/agents/experimenter.md` |
| **DPO checkpoints deleted before upload** -- violated existing policy | 8993b974 | Already in CLAUDE.md but still happened. Add pre-delete check to experimenter: `ls -la <checkpoint_dir>; echo "CONFIRM: has this been uploaded? Check HF Hub."` before any `rm -rf` on model dirs. | `.claude/agents/experimenter.md` |
| **Subagent dispatched without announcing** -- user asked "What experiment are you running?" | 5dcb08bd | Manager must announce experiment name + goal in 1 sentence before dispatching. Add to manager.md dispatching section. | `.claude/agents/manager.md` |
| **User pushed for deeper web search** ("Search the web even more deeply") | aae28bd2, 5dcb08bd | When designing experiments that reference published methods, always do 2+ rounds of web search (paper + methods section + any errata/corrections). | `.claude/agents/planner.md` |

## Failed Approaches (document to prevent retries)

- **All-token masking for EM induction**: Produced only -3 to -5pt alignment drop vs expected -50pt. Betley et al. use `train_on_responses_only=True` (assistant-only masking). The all-token approach dilutes the EM signal with instruction tokens. **Document in:** experimenter.md known issues.
- **Locally generating EM training data (bad legal advice)**: User explicitly stopped this -- "Do not generate bad legal advice -- download them from the online EM datasets." Betley's `truthfulai/emergent_plus` dataset (top 6K by sneakiness score) is the correct source. **Document in:** experimenter.md known issues.
- **Sequential HF `model.generate()` for leakage eval**: Was 10-50x slower than vLLM batched inference. Had to be switched mid-run in session 9b1683ee. Already in CLAUDE.md but experimenter still used HF initially. **Document in:** experimenter.md (reinforce existing CLAUDE.md rule).
- **Pod 4 used without env verification**: Missing deepspeed, wrong venv path, no flash-attn. Caused 20+ retries before diagnosis. Pre-flight checks exist but weren't run on the new pod. **Document in:** experimenter.md -- "For NEW pods or pods not used recently, run full preflight before any experiment."

## Proposed Changes

### CLAUDE.md

```diff
 ## After Every Experiment
 
 1. **Verify uploads:** Confirm eval results uploaded to WandB, model checkpoints uploaded to HF Hub
+
+## End of Session
+
+Before ending any session with uncommitted research outputs:
+1. Run `git status --short | wc -l` to check uncommitted file count
+2. If >20 new/modified files or any modified drafts/RESULTS.md, suggest a commit
+3. Critical files that should never stay uncommitted overnight: RESULTS.md, research_log/drafts/*.md, eval_results/**/*.json, figures/**/*.png
```
**Reason:** Zero commits today despite 462 changed files. Phase A1 draft, RESULTS.md, 91 eval_result directories, and new skills all at risk of loss if the VM dies. This pattern has happened before (April 9 was also commit-light).

### Agent Definitions

**File:** `.claude/agents/experimenter.md`

```diff
 6. **List assumptions** — Before any experiment, write out: "I am assuming X, Y, Z"...

+### Method Verification (for replication experiments)
+
+When an experiment replicates or extends a published method (Betley EM, Tulu post-training, etc.):
+1. **Data source** — Use the exact dataset from the paper, not locally generated approximations. Cite the HF/URL source.
+2. **Loss masking** — Verify: all-token vs response-only vs assistant-only. This is the #1 source of replication failures. (April 14: all-token EM masking produced -3pt instead of -50pt; switching to assistant-only fixed it.)
+3. **Hyperparameters** — Cross-reference lr, epochs, batch_size, LoRA config against the paper's appendix/code. Don't assume defaults match.
+4. **State your reference** — "I am following [Paper] Section X / repo file Y" so reviewers can verify.
+
+### Pipeline State Tracking (for multi-stage experiments)
+
+For experiments with 3+ pipeline stages across multiple pods:
+- Maintain a STATUS table at the top of each status update:
+  ```
+  | Pod | Condition | Stage | Progress | Next |
+  ```
+- Update this on every check-in. The manager and user should never have to ask "what stage is X at?"
```
**Reason:** Two biggest time-wasters today: (1) EM methodology wrong for 12h because masking wasn't verified against Betley, (2) user lost track of pipeline state in the 8993b974 session.

```diff
 **Known infrastructure issues:**
 - ZeRO-2 on < 4 GPUs for 7B full fine-tune will OOM. Use ZeRO-3.
 - Zombie CUDA allocations survive process death. Only fix is container restart.
 - open-instruct (March 2025) requires transformers 4.48.x.
 - flash-attn defaults to True in open-instruct's finetune.py dataclass.
+- **New pods:** Always run full preflight (`require_preflight()`) on any pod not used in the last 48h or any newly provisioned pod. Pod 4 (added April 14) was missing deepspeed and had wrong venv paths, causing 20+ retries.
+- **EM training:** Use Betley's actual dataset (`truthfulai/emergent_plus`, top 6K by sneakiness), NOT locally generated data. Use assistant-only masking (`train_on_responses_only=True`), NOT all-token. lr=1e-4, 1 epoch for 7B.
+- **Always use vLLM for generation** — even during eval/inference in leakage experiments. Sequential HF generate was 10-50x slower and had to be swapped mid-run.
```
**Reason:** Three issues that burned significant time today.

**File:** `.claude/agents/manager.md`

```diff
 ## Dispatching Subagents
 
-**Experimenter** — training, eval, new code, debugging, monitoring. Tell them which pod and which GPUs.
+**Experimenter** — training, eval, new code, debugging, monitoring. Tell them which pod and which GPUs. **Always announce to the user** what you're dispatching and why in 1 sentence before spawning the agent.
```
**Reason:** User asked "What experiment are you running?" in session 5dcb08bd, indicating the manager dispatched without communicating.

```diff
 ## End-of-Day Retrospective
 
 After 23:00 local time, suggest: "Want me to run the daily retrospective?" If yes, spawn `retrospective` agent on today's session transcripts.
+
+## End-of-Session Hygiene
+
+Before ending any substantial session (>30 min or any experiment work):
+1. Check `git status --short | wc -l` -- if >20 uncommitted files, suggest a commit
+2. Check if RESULTS.md or any draft in research_log/drafts/ has been modified but not committed
+3. Verify any running pod experiments have their logs accessible (not just in a dead subagent's context)
```
**Reason:** Zero commits on a day with 462 changed files. Research outputs must not accumulate uncommitted.

### Skills

No new skill changes proposed. The ideation skill and daily-update skill were created today and appear to be working.

### Hooks

**Proposed hook:** PostToolUse on Bash (git push)

The existing post-push hook syncs pods from GitHub. No new hooks needed -- the commit gap is a process issue (manager forgetting to commit), not a hook issue. A hook that auto-commits would be dangerous.

### Memory Updates

- **Update `project_infrastructure.md`** — Pod 4 was added (8x H100, 69.30.85.155:22184). Memory still lists only 2 pods. Manager.md has 4 but memory is stale.
- **New feedback memory: "EM methodology must match paper"** — All-token masking + generated data → -3pt; assistant-only masking + Betley dataset → expected -50pt. This is the second data/methodology confound (after truthification v1).
- **Security alert: GitHub PAT exposed** — `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` was shared in plaintext in session 9b1683ee. Should be rotated immediately if not already done. Do NOT save the token to memory -- just flag it.

## Successful Patterns (reinforce these)

- **Phase A1 execution was excellent** — 44/44 runs completed with 0 failures. Pre-registered hypothesis confirmed (rho=0.60 > 0.40 threshold). Controls properly designed (shuffled persona, generic training, held-out personas). Reviewer fixes applied same-day. This is the gold standard for experiment execution.

- **EM methodology pivot was fast once caught** — Within ~2 hours of identifying weak EM results, the methodology was overhauled to match Betley et al. (dataset, masking, hyperparams) and applied to all 3 pods. The pivot speed is good; the issue is it should have been caught before the first run.

- **Gate-keeper integration is working** — Session 358a28f1 added the gate-keeper step and it's being used. The adversarial planner pipeline (Gate-Keeper → Planner → Critic → Revise → User approval) was followed for Phase A1 and produced a well-designed experiment.

- **Reproducibility Card template adopted** — Session f6722c2c added the full reproducibility card and decision log requirements to CLAUDE.md. This is already being followed in new drafts.

- **New skills created efficiently** — Ideation skill (1792fa86), daily-update skill (1b8cd658), and gate-keeper agent were all created in focused sessions. The daily-update skill was iterated once based on user feedback ("mention concrete things, not abstract aims").

- **Results promotion marathon** — Session 55079351 systematically reviewed all drafts and promoted completed ones. This is good research hygiene.

- **vLLM migration mid-experiment** — When sequential HF generation was identified as the bottleneck in the leakage experiment, the switch to vLLM batched inference was made cleanly without losing in-progress results.

## Comparison to April 9 Retrospective

| Issue | April 9 Status | April 14 Status |
|-------|---------------|-----------------|
| OOM debugging (15 retries) | Major time-sink | Not repeated -- escalation rule working |
| Wrong assumptions about APIs | 5 dedicated sessions | Reduced but EM masking was still assumed wrong |
| Data confound (truthification v1) | 67K mixed instead of 6K | EM methodology wrong (masking + data source) -- similar class of error |
| Subagent model downgrade | User had to correct | Not observed -- "always Opus" rule followed |
| Data validation before training | Added to experimenter.md | Followed for data, NOT for methodology |

**Key insight:** The April 9 data validation fix addressed *dataset* verification but not *methodology* verification. Today's EM failure was the methodology (masking config), not the dataset itself. The experimenter.md "Verify data" step needs to become "Verify data AND method."

## Metrics

- **Corrections by user:** 11 explicit + ~8 implicit (interrupts/redirections)
- **Agent dispatches:** ~85 subagents across all sessions (50+ in 9b1683ee alone)
- **Experiments run:** 5 completed successfully, 2 running (Aim 5 midtrain), 0 failed (but EM methodology was wrong on first attempt)
- **Time on debugging vs research:** ~20% debugging (Pod 4 env, flash-attn inconsistency, EM methodology), 80% research. Significant improvement over April 9 (40/60).
- **Git commits:** 0 (CRITICAL -- 462 uncommitted files)
- **Security incidents:** 1 (GitHub PAT exposed in session transcript)

## Priority Actions

1. **[CRITICAL] Commit research outputs** -- RESULTS.md, Phase A1 draft, eval_results, figures, new skills. 462 files at risk.
2. **[CRITICAL] Rotate GitHub PAT** -- exposed in session 9b1683ee transcript.
3. **[HIGH] Update experimenter.md** -- add Method Verification section and pipeline state tracking. Prevents repeat of EM methodology error.
4. **[HIGH] Update manager.md** -- add dispatch announcement rule and end-of-session hygiene.
5. **[MEDIUM] Update CLAUDE.md** -- add end-of-session commit reminder.
6. **[MEDIUM] Update memory** -- project_infrastructure.md is stale (lists 2 pods, should be 4).
7. **[LOW] Add EM known issues to experimenter.md** -- Betley dataset, assistant-only masking, new pod preflight.
