# Daily Retrospective -- 2026-04-09

**Sessions reviewed:** 21 (18 main sessions + subagents scanned)
**Total substantive user messages:** ~211
**User corrections:** 6 explicit "No" corrections + ~10 implicit redirections
**Background tasks:** 136 completed, 9 failed
**Subagent dispatches:** ~97 across all sessions

## Summary

The bulk of today's work fell into three categories: (1) setting up the multi-agent scaffolding (manager/experimenter/analyzer/reviewer/retrospective agents, skills, and memory), (2) fighting infrastructure issues (ZeRO-2 vs ZeRO-3 OOMs, open-instruct flag names, flash-attn, zombie GPU processes), and (3) running the Aim 6 truthification experiments (3 iterations, converging on a strong result). The single largest time sink was the H100 pod training saga in session f04bb16f, where approximately 15 retries were needed to get open-instruct SFT running due to OOM + incompatible CLI flags + zombie CUDA allocations. The most impactful user correction was the "wrong assumptions" complaint in session a6f9351f, which led to the Verifier phase being added to the adversarial-planner skill. Five separate sessions (8a2d0810, 90e8c9b0, 919a21d1, 9c1f9aaf, 6ba8cbf2) were dedicated entirely to web-searching for assumption-prevention strategies -- suggesting the user considers this a critical gap.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---------|----------|-------------|-------------|
| Misunderstanding scope of user's "other projects" question -- user had to say "No I meant..." twice | 9ffccccb (twice) | When user references "other things" or "other projects", ask for clarification rather than guessing which scope they mean | CLAUDE.md |
| Agent confidently stated wrong facts about tools/APIs (vLLM layers, speculators batch_size, open-instruct flags, flash-attn defaults) | f04bb16f, a6f9351f | Already addressed by Verifier phase in adversarial-planner. But experimenters running outside the planner still make these errors. Add a pre-execution verification rule to experimenter.md | .claude/agents/experimenter.md |
| Training OOM debugging took 15+ retries without escalating to user | f04bb16f | Experimenter should escalate after 3 failed OOM fixes, not keep trying. The root cause (zombie GPU allocations) was infrastructure, not hyperparameters | .claude/agents/experimenter.md |
| User had to ask "Are your subagents Opus?" then correct to "always use Opus 4.6 max" | 9ffccccb | Already saved to memory. But no enforcement mechanism exists. | .claude/settings.json |
| Truthification Exp 1 trained on wrong data (67K mixed instead of 6K insecure) | 99a99ada | Add data validation checkpoint to experimenter: before training, log dataset size + first 3 examples and verify against spec | .claude/agents/experimenter.md |

## Failed Approaches (document to prevent retries)

- **ZeRO-2 on 2 GPUs for 7B full fine-tune**: OOMs even at batch_size=1 because ZeRO-2 only shards optimizer states, not model parameters. Need ZeRO-3 for < 4 GPUs with 7B models. Document in experimenter.md.
- **Using zombie-occupied GPUs for training**: Even with CUDA_VISIBLE_DEVICES excluding them, DeepSpeed remaps indices causing OOM on partially-free GPUs. Only fix is container restart. Document in manager.md infrastructure section.
- **LoRA SFT for persona-specific behavior via system prompts**: Exp17 showed 100% global saturation even at weak configs. LoRA SFT cannot create persona-specific behaviors through system prompts alone -- the marker bleeds to all personas. Documented in leakage_sweep_diverse draft.
- **open-instruct with transformers 5.x**: Flag incompatibilities (`--packing`, `--clean_checkpoints_at_end`, `--no_push_to_hub`, etc. not recognized). Must use transformers 4.48.x with open-instruct March 2025. Document in experimenter.md.
- **Loading HF datasets without explicit `data_files=`**: Merged all 9 JSONL files (67K examples) instead of target split (6K insecure). Confounded entire experiment. Add to experiment-runner skill as mandatory check.

## Proposed Changes

### CLAUDE.md

No changes needed. The "DO NOT MAKE ASSUMPTIONS" and "NEVER TAKE SHORTCUTS" sections are comprehensive. The issue is enforcement, not documentation.

### Agent Definitions

**File:** `.claude/agents/experimenter.md`

```diff
 ### Before Running
 
 1. **Understand the task** — Read the experiment spec carefully. If anything is ambiguous, state your interpretation and proceed (you're a subagent; you can't ask the user directly).
 2. **Check prerequisites** — Verify data exists, configs parse, dependencies are installed, GPU is available.
-3. **Check memory** — Look for past learnings about similar experiments (common pitfalls, what worked).
-4. **Estimate resources** — GPU-hours, disk space, wall time.
+3. **Verify data** — Before training, log: (a) dataset size (number of examples), (b) first 3 examples, (c) column names. Compare against the experiment spec. A wrong dataset invalidates the entire run. The truthification Exp 1 trained on 67K mixed examples instead of 6K insecure code because data_files= was not specified.
+4. **Check memory** — Look for past learnings about similar experiments (common pitfalls, what worked).
+5. **Estimate resources** — GPU-hours, disk space, wall time.
+6. **List assumptions** — Before any experiment, write out: "I am assuming X, Y, Z" for any factual claims about libraries, APIs, layer numbers, data formats, or hardware capabilities. Mark confidence (high/medium/low). Search or read docs for anything below high confidence.
```

```diff
 Common auto-fixes (try once before escalating):
 - OOM: Halve batch size, enable gradient accumulation
 - NaN loss: Halve learning rate, add gradient clipping
 - Network error: Wait 5 min, retry
 
+**Escalation rule:** If the same category of error (OOM, NCCL, import) persists after 3 fix attempts, STOP and report to the manager with a summary of what was tried. Do not keep iterating -- the root cause may be infrastructure (zombie GPU processes, wrong library version, insufficient VRAM for the approach) rather than hyperparameters.
+
+**Known infrastructure issues:**
+- ZeRO-2 on < 4 GPUs for 7B full fine-tune will OOM. Use ZeRO-3.
+- Zombie CUDA allocations survive process death. Only fix is container restart. Do not try to train on partially-occupied GPUs.
+- open-instruct (March 2025) requires transformers 4.48.x. Flag names differ from current docs.
+- flash-attn defaults to True in open-instruct's finetune.py dataclass. Must explicitly pass --no_use_flash_attn if not installed.
```

**File:** `.claude/agents/manager.md`

```diff
 | **thomas-rebuttals-2** (lli58lkp2gum6l) | 8x H100 SXM 80GB | `ssh root@103.207.149.64 -p 16193` | 10TB |
 
 - SSH key: `~/.ssh/id_ed25519`
 - Pod IPs/ports may change on restart — re-query RunPod API if connection fails
 - After container restart, SSH key must be re-added via web terminal
 - H100 pod env: Python 3.11, open-instruct, transformers 4.48.3, flash-attn 2.8.3, deepspeed 0.18.9
+- **Zombie GPU fix:** If GPUs show memory used but no running processes, the only reliable fix is container restart. Do not assign experimenters to partially-occupied GPUs -- they will waste time on OOM debugging.
+- **ZeRO-3 required** for 7B full fine-tune on < 4 GPUs. ZeRO-2 OOMs because it only shards optimizer states.
```

### Skills

**File:** `.claude/skills/experiment-runner/SKILL.md`

```diff
 ## Pre-flight Checklist
 
+### Data Validation (MANDATORY)
+Before launching any training run, verify the data:
+1. Load the dataset with the exact same code path the trainer will use
+2. Log: number of examples, column names, first 3 examples (truncated)
+3. Compare against the experiment spec
+4. If using HuggingFace datasets with multiple splits/files, ALWAYS specify `data_files=` explicitly. Loading without it may merge all files silently.
+
+This check was added after truthification Exp 1 trained on 67K mixed examples instead of 6K insecure code, confounding all results.
```

### Hooks

No new hooks proposed. The existing PostToolUse ruff hook is working well.

A potential future hook (not proposed now, needs user input): a PreToolUse hook on Bash that checks if `nvidia-smi` was run before any `nohup ... python` training command. But this is fragile and better enforced by agent instructions.

### Memory Updates

**New memory to save (retrospective agent):**

1. **OOM debugging pattern** -- ZeRO-2 vs ZeRO-3 threshold, zombie GPU issue, escalation after 3 retries. This should be in experimenter memory, not just agent definition.
2. **Data validation lesson** -- truthification Exp 1 confound. Already partially documented in drafts log but should be in experimenter memory.

**No memories to delete.** All existing memories in the manager and user auto-memory are current and accurate.

## Successful Patterns (reinforce these)

- **Multi-agent scaffolding design (b94eb178):** The manager/experimenter/analyzer/reviewer pipeline was designed in a single session with web research backing. The user was satisfied with the result. The separation of concerns is clean and the reviewer-as-independent-verifier pattern is strong.

- **Adversarial planner with Verifier phase (a6f9351f):** After the wrong-assumptions complaint, the Verifier phase was added to the adversarial-planner skill. This is a direct, concrete response to user frustration. The skill now has Plan -> Verify -> Critique -> Revise -> Implement -> Implementation Critique. This is thorough and should be preserved as-is.

- **Truthification experiment iteration (99a99ada):** Despite the data confound in v1, the team iterated to v2 (correct data) and v3 (new domain -- bad medical advice) within the same day. Final result is strong: source attribution preserves 97% alignment vs raw EM catastrophic collapse. The iteration speed is good.

- **Background monitoring pattern:** 136 of 145 background monitoring tasks completed successfully (93.8%). The pattern of launching nohup + periodic monitoring checks works well and should continue.

- **Web search before building:** Sessions a6f9351f, b94eb178, and 5 dedicated search sessions show the user strongly values searching for existing solutions first. The CLAUDE.md already emphasizes this and it's being followed. The adversarial-planner now requires web search in both Planner and Critic phases.

- **Memory system adoption:** Key corrections (always Opus, Claude as judge, checkpoint loss prevention, wrong assumptions) are all saved to memory. This is working as intended.

## Metrics

- **Corrections by user:** 6 explicit + ~10 implicit redirections
- **Agent dispatches:** ~97 total (36 in 9ffccccb alone). Most completed successfully.
- **Background tasks:** 136 completed / 145 total (93.8%)
- **Experiments run:** 6 successful (truthification v1/v2/v3, axis projection, leakage sweep, persona neighbor) / 1 partially failed (truthification v1 data confound, rerun as v2)
- **Time on debugging vs research:** Estimated 40% debugging (mostly f04bb16f's 15-retry OOM saga), 60% research. Without the OOM debugging, ratio would be ~15/85.
- **Sessions used for meta-improvement:** 7 of 21 (5 assumption-prevention searches, 1 agent scaffolding design, 1 codebase reorg). Significant investment in process improvement today.

## Priority Actions

1. **[HIGH] Update experimenter.md** with data validation checkpoint and escalation rule. These address the two biggest time-wasters today.
2. **[MEDIUM] Update experimenter.md** with known infrastructure issues (ZeRO-2 vs ZeRO-3, zombie GPUs, open-instruct compatibility). Prevents repeat of f04bb16f saga.
3. **[MEDIUM] Update manager.md** infrastructure section with zombie GPU fix guidance.
4. **[LOW] Consider adding experiment-runner skill pre-flight data validation section.
