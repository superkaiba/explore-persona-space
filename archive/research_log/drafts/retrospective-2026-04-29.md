# Daily Retrospective — 2026-04-29

**Sessions reviewed:** 14 main sessions + many subagent transcripts
**Total user messages (excl. skill/task notifications):** ~430 typed; ~88 of those (≈20%) were near-identical "Check progress of issue #N" pings (`ScheduleWakeup` re-fires + manually re-typed monitor prompts)
**Commits to main:** **2** on `main`, **1** on a worktree (`f16e8d4` codebase refactor) → **3 total today**, including the **biggest workflow-config commit of the past two weeks (`1a04d22`, +1075/−138 LOC across 10 files)**
**New / updated clean-result issues:** **6** clean-result-labeled issues touched (#99, #113, #116, #120, #121, #142) + **1 promoted to `clean-results:draft` (#123)** — best clean-result-output day in the project's history
**ScheduleWakeup calls:** 125 | **Agent dispatches:** ~99 (top: experimenter=22, planner=16, Explore=18, critic=10, analyzer=10, gate-keeper=9, reviewer=7) | **tool errors:** ~860 across all sessions

## Summary

A landmark day on two axes that pulled in opposite directions. **(1) Workflow redesign:** session 6c597909 began with the user articulating that "I start running many followup experiments in the same context that are not well scoped or well planned, and they run badly with bugs", and ended hours later with commit `1a04d22` — a large redesign of the `/issue` lifecycle that introduces four new agents (`consistency-checker`, `upload-verifier`, `interpretation-critic`, `follow-up-proposer`), a `verify_uploads.py` script, two new lifecycle states (`uploading`, `interpreting`), and a `PreToolUse` hook that **blocks ad-hoc experiment commands outside `/issue`**. Several proposals from prior retros landed in this commit: the "NEVER run experiments inline" rule, the RunPod GraphQL `X-Team-Id` API doc, the iterative-interpretation loop, the consistency-check step. **(2) Heavy execution day in parallel:** issues #99 (Qwen-default deconfounding), #112 (assistant-in-negatives confound + alignment_v2 + asst_excluded re-runs), #102 (marker-bridge v2 with sweeps over benign/harmful question sets, lr/epoch/neg-ratio), #115 (multi-seed replication), #120 + #123 (50-fictional-character leakage), #121 (reversed EM marker), and #142 (KL/JS divergence) all advanced. **The chronic infrastructure backlog fired hard:** 22 `Disk quota exceeded`, 28 `Command timeout 30000ms` (SSH MCP), 12 pod3 handshake timeouts, 11 `No commits between main` PR-create errors, 11 MooseFS errors / 7 SIGBUS / 6 SIGKILL, 4 system-TRL `_VALID_DICT_FIELDS` shadow-import re-fires, 49 OOMs, 4 vLLM tqdm errors. The user also caught **two new methodological confounds** in #112 (alignment data not persona-conditioned; assistant in contrastive negatives), forcing 5+ reruns and the `feedback_no_substring_match` rule was added to memory. The `pipeline-provenance-system` worktree (4 days old, ~920 LOC) is no longer in the worktree list — looks like it was pruned without being committed. **Worktree count: 27 → 7** (big cleanup, but possibly some orphan loss).

## Proposal Backlog Audit (lead with this — 11th cycle)

| Proposal | First proposed | Days unapplied | Status today |
|---|---|---|---|
| "NEVER run experiments inline" rule + PreToolUse hook | 2026-04-22 | **7 → 0** | ✅ **APPLIED in `1a04d22`** — both the CLAUDE.md rule and the `Bash` matcher hook are live |
| Consistency-checker agent (single-variable change verification) | 2026-04-26 | **3 → 0** | ✅ **APPLIED** — `.claude/agents/consistency-checker.md` exists, wired into `/issue` skill |
| Iterative interpretation loop (analyzer ↔ critic, max 3 rounds) | 2026-04-25 | **4 → 0** | ✅ **APPLIED** — `interpretation-critic.md` + `status:interpreting` state |
| Upload verifier (hard gate before interpretation) | 2026-04-26 | **3 → 0** | ✅ **APPLIED** — `upload-verifier.md` + `verify_uploads.py` |
| Follow-up proposer agent (auto-fire after experiment completes) | 2026-04-26 | **3 → 0** | ✅ **APPLIED** — `follow-up-proposer.md` |
| RunPod GraphQL API + X-Team-Id docs in CLAUDE.md | 2026-04-22 | **7 → 0** | ✅ **APPLIED** in `1a04d22` |
| `gh pr create` empty-commit fix in `/issue` Step 5b | 2026-04-25 | **5** | ❌ Fired **11×** today across sessions |
| Root + workspace + MooseFS quota disk preflight | 2026-04-23 | **7** | ❌ **22 `Disk quota exceeded`** events today |
| `experimenter.md` Step 0: match prior protocol | 2026-04-26 | **4** | ❌ Re-fired today on #102: "Wait i thought we trained the marker into the villain persona, and then trained the marker into the assistant?" + "are we using the same questions to train the marker into the villain and to train it into the assistant?" |
| Qwen default-system-prompt gotcha in CLAUDE.md | 2026-04-24 | **6** | ❌ Re-encountered at start of session 56cd23f2 ("Was this using the default qwen system prompt or just a generic assistant prompt?") and again on #115/#108 default-prompt sweep |
| SSH `sleep N && cmd` 30s timeout block / hook | 2026-04-25 | **5** | ❌ **28 `Command timeout 30000ms`** firings today |
| `merge_marker_adapter` weight-byte assert | 2026-04-23 | **7** | ❌ Not added |
| meta:retro-backlog GitHub issue | 2026-04-25 | **5** | ❌ Not opened |
| MooseFS large-write SIGBUS workaround in `merge_lora_adapter` | 2026-04-27 | **2** | ❌ Re-derived again on session 073d ("write merged models to local /root/ to avoid MooseFS SIGBUS on 15GB writes") — same 7 SIGBUS / 11 MooseFS fires |
| TRL system-shadow venv guard (pod5) | 2026-04-27 | **2** | ❌ 4 `_BaseConfig._VALID_DICT_FIELDS` errors fired again on pod5 |
| vLLM `gpu_memory_utilization` adaptive default | 2026-04-27 | **2** | ❌ Not adjusted |
| TRL data-format `prompt_completion` vs `messages` assertion | 2026-04-28 | **1** | ❌ Not added; the bug was the *predicate* of one of today's #112 reruns |
| pod-health-quick gate before `experimenter` dispatch | 2026-04-28 | **1** | ❌ Pod3 had 12 handshake timeouts before fail-over |
| ScheduleWakeup interval-escalation rule | 2026-04-28 | **1** | ❌ Sessions 5cc and dd19 fired the same monitor prompt 8+ times |
| CLAUDE.md gotchas: Qwen-default + MooseFS + TRL-format + TRL-shadow + vLLM-mem | 2026-04-28 | **1** | ❌ None of the 5 NEW gotchas added; none of the existing 5th-cycle Qwen gotcha added |

**Streak: BROKEN.** First non-zero proposal-application day in 11 days. **6 large workflow proposals applied (all bundled into `1a04d22`).** But the 13+ small-but-chronic infrastructure proposals are still unapplied — the same backlog of P0 bugs continues to fire daily. The day was "applied the big workflow ideas, ignored the small bugs."

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---|---|---|---|
| `gh pr create` "No commits between main and issue-N" — **6th cycle** | 5cc, dd19, 073d, 56c, others | Same 3-line patch from 04-25: `git commit --allow-empty -m "chore(#N): open draft PR"` before `gh pr create` | `.claude/skills/issue/SKILL.md` line 286 |
| `Disk quota exceeded` on `/workspace` MooseFS — **22 firings, 5th-cycle** | 56c, 5cc, 073d | Preflight needs `mfsgetquota /workspace` AND `df -ih /workspace` for inodes; current `df -h` proposal is incomplete | `src/explore_persona_space/orchestrate/preflight.py` |
| `mcp__ssh__ssh_execute` 30 000 ms timeout when sub-script has `sleep N && …` chain — **28 firings, 5th-cycle** | 073d (8×), 5cc (6×), dd19 (5×), 56c (4×) | The MCP tool caps at 30000 ms regardless of inner `timeout 60`. **PreToolUse hook** to block `sleep N` with N > 20 inside `ssh_execute`; force `ScheduleWakeup` instead | `.claude/settings.json` (already has 1 hook — add the SSH hook here) |
| MooseFS SIGBUS on 15 GB merged-model writes / SIGKILL on disk events | 073d (3 SIGBUS + 6 SIGKILL), 56c | Encode the `/root/tmp_eval/` workaround into `merge_lora_adapter` when mount is FUSE/MooseFS | `src/explore_persona_space/leakage/runner.py` (or wherever `merge_lora_adapter` lives) |
| Qwen default system prompt confound (6th cycle) | 56c, dd19 | Add to CLAUDE.md gotchas; add `system_prompt_at_eval` row to clean-results template | `CLAUDE.md` + `.claude/skills/clean-results/template.md` |
| Pod3 SSH handshake timeouts — 12 firings | 5cc, dd19 | Same proposal as 04-28: pod-health-quick gate before `experimenter` dispatch | `.claude/skills/issue/SKILL.md` Step 4 |
| User caught data confound that subagent missed (alignment data not persona-conditioned; assistant in contrastive negatives) | 5cc | Add data-audit checklist (persona-conditioning, train→eval overlap, contrastive-set composition, loss-masking format, system-prompt-at-eval) | `.claude/skills/clean-results/principles.md` |
| Same monitor message re-typed identically 4-8 times by user | 5cc (8×), dd19 (5+), 073d (≥10) | `experimenter.md` self-pacing escalation rule (still unapplied from 04-28) | `.claude/agents/experimenter.md` |
| Clean-result issue not appearing in "Clean Results" project-board column | 5cc | The user asked **3×** "Why is the issue not in the experiment queue board in the clean results column?" — ended up using `gh cli` manually. The auto-promotion path for `clean-results` label → project-board column is broken, or the analyzer never adds the label correctly. Audit `analyzer.md` and the `gh project` plumbing. | `.claude/agents/analyzer.md` and project-board automation |
| Identical "B and C are identical down to every decimal" silent-control bug on #121 | dd19 | The user spotted this; turned out the experiment was run correctly but the conditions were not informative as designed. Add to `consistency-checker` (now exists): when conditions B and C produce bit-identical outputs, flag a control-design failure — either the conditions don't actually differ in behavior, or the implementation has a copy-paste bug | `.claude/agents/consistency-checker.md` (newly added today) |

## Failed Approaches (document to prevent retries)

- **`pipeline-provenance-system` worktree from 2026-04-26 (~920 LOC) is no longer present in `git worktree list`** — possibly pruned without committing. 4 days of `lineage/` + `experiments/` + `scripts/build_claims_index.py` work appears to be lost or merged-into-something-else. Recover by inspecting `git reflog` and `find /home/thomasjiralerspong -name 'build_claims_index.py'`. → see Workflow Gaps for proposed orphan-worktree gate.
- **Reversed-EM control bug on #121** — conditions B and C produced bit-identical numerical outputs ("B and C are identical down to every decimal"). User flagged as suspicious; investigation showed the "control" wasn't doing what it was designed to. The `consistency-checker` agent (added today) didn't yet exist when this experiment was planned, so this is a precedent that should be encoded into the agent.
- **MooseFS SIGBUS workaround re-derived a 4th time.** "write merged models to local /root/ to avoid MooseFS SIGBUS on 15GB writes" was manually retyped on session 073d. The retro proposal to encode it into `merge_lora_adapter` for FUSE mounts is now 2 days unapplied; the daily cost is rising (today: ~3 hours of pod-management orchestration on session 073d alone).
- **Pod5 instability cascade — 4th day in a row.** Same hit-list as 04-26/27/28: MooseFS SIGBUS during merge → system-TRL shadow `_BaseConfig._VALID_DICT_FIELDS` → tqdm/vLLM 0.11 incompat → GPU isolation breakdown when 4 procs all defaulted to GPU 0. Today, session 073d eventually migrated to pod3 and ran 4× in parallel successfully (matching pattern from 04-28). **The proposal "when a pod hits 3+ distinct failures, migrate" is now an established pattern but is not encoded as a rule** — every session re-derives it manually.
- **Plot-iteration friction on #115/#120.** "Make a clean result for the single seed" → "Add error bars" → "Could we also make a cosine similarity matrix" → "Make it a heat map" → "Also change the groups in the bar chart to fit with the text" — 6+ correction rounds. The `paper-plots` skill is being used but the analyzer is producing first-pass plots that need extensive iteration. (Less severe than 04-27 ALL-CAPS rounds, but still 30+ min lost.)
- **Issue not auto-moving to "Clean Results" project-board column.** User asked 3× over the day — the `clean-results` label was applied but the project-board automation did not move the card. Manual `gh project item-edit` was used. Audit the project-board webhook / automation.

## Workflow Gaps

- **Today's biggest workflow win was bundled into one commit and not announced via `/issue`.** The user articulated the workflow problem in session 6c597909, kicked off `/adversarial-planner`, ran an `Explore`-heavy design phase, and committed `1a04d22` directly to `main` (skipping the issue-flow that the redesign itself enforces). This is fine *for the meta-workflow change* but means there's no GitHub-issue record of why each agent / hook was added. Propose: open issue **#149** ("Workflow improvement", which the user did create with `gh issue create` today) and link `1a04d22` back to it via a comment for traceability.
- **Codebase refactor on `refactor-codebase-cleanup` worktree (`f16e8d4`) is on a separate branch and hasn't been merged to `main`.** Session f1573bde dispatched 9 `Explore` agents, produced ~7 audit reports, then committed a "Comprehensive codebase refactoring: dedup, split, fix silent failures" commit on a worktree. This is appropriate (it's a big refactor) but needs to be merged or PR'd before more experiment work happens, otherwise the `main` and worktree drift. There's no GitHub issue tracking it.
- **`pipeline-provenance-system` worktree appears LOST.** The 04-26/27/28 retros all flagged it as 920 LOC uncommitted. Today's `git worktree list` no longer shows it. Either (a) it was pruned via `git worktree prune` (possible: 4 days of inactivity is past the default `worktree.expire`), or (b) it was deleted manually. Need to grep `git reflog` and the filesystem for residual files. **Propose: a hook that fails any `git worktree prune` if `--dry-run` would prune any worktree with uncommitted changes.**
- **Several issues created today (#144–#150) have no `status:proposed` label or aim:* label.** They're raw ideation issues (the user said "yes, commit everything" at the end of 6c597909 so these are pre-issue brainstorm captures). Per `research-project-structure.md`, raw ideation should live in `research_log/ideas/YYYY-MM-DD.md`, not as bare GH issues. Propose: convert #144–#148 either to ideation file entries or to `status:proposed`-labeled issues with bodies, before they get lost.
- **The user explicitly said "I think i will forget to use the followup skill"** in session 6c597909. The redesign added the `follow-up-proposer` agent that auto-fires after experiment completion — good. But the user's deeper concern was about *cognitive load*, not just availability. Verify: when an experiment finishes, does `/issue` automatically dispatch `follow-up-proposer` without user prompting? If yes, this is solved. If not (user must remember), it's still open.
- **686 modified/untracked files in the main checkout at end of day.** Most are eval-results JSON / figures, but also draft retros, modified `RESULTS.md`, etc. End-of-session check rule "if modified drafts, RESULTS.md, or eval_results JSON are uncommitted, commit before ending" was not run. The auto-commit on session-end via hook is still unimplemented (proposed multiple times).

## Successful Patterns (reinforce these)

- **User-driven workflow articulation → multi-Explore-agent audit → integrated redesign**, all in a single ~3-hour session (6c597909 → main commit `1a04d22`). The flow:
  1. User articulates problem qualitatively ("agents run followups in same context with bugs").
  2. Dispatched 8 `Explore` agents in parallel to audit existing infrastructure (pipeline, agents, upload policy, scripts, configs, silent failures).
  3. Used `adversarial-planner` to design the new lifecycle.
  4. Implemented in one bundled commit.
  This is the project's first end-to-end "user-articulates → multi-agent-audit → bundled-implementation" loop. **Capture as exemplar pattern in `.claude/rules/agents-vs-skills.md`** under "Typical composition pattern."
- **Adversarial-pattern caught data confound on #112 alignment data.** User: "Can you get an unbiased subagent to verify this experiment was run correctly" → audit found that alignment data was not persona-conditioned. Same shape as 04-28's #112 confound. Reinforces the `code-reviewer` / `reviewer` adversarial-pattern value. The new `interpretation-critic` (added today) is the codification of this pattern.
- **Issue #115 multi-seed run** (user: "Can we run another seed of these to verify variance" → 3-seed parallel on pod1 → multi-seed clean result with cross-leakage matrix). Same as 04-28: multi-seed-by-default for any single-seed result that survives reviewer.
- **Cross-pod parallelism repeats (4th day).** Issue #112 alignment_v2 distributed across pods 1, 2, 3, 5 simultaneously. asst_excluded re-runs across pods 1, 3, 5. Marker-bridge sweep on pod3 (4 GPUs) and pod4 (sequential). When >4 conditions, default to multi-pod. Now persistent enough to be a *project default*.
- **6 clean-result-labeled issues touched in one day.** #99 (deconfounded), #113 (multi-seed expanded with cross-leakage), #116 (rewrite with full asst_excluded data), #120 (50-fictional-character extension), #121 (HIGH-confidence categorical claim), #142 (KL/JS divergence MODERATE) + #123 promoted to draft. Best output day in project history. The "clean results pipeline" is the most-used workflow.
- **`feedback_no_substring_match.md` saved to memory** — user: "Run source persona eval for all 3 behaviors, use claude judging NOT substring match. Remove all substring match evals and save a memory to never use substring match (except for marker leakage)" → saved successfully. Confirms the memory system is working when invoked.
- **The user articulated the deep workflow problem and got a bundled fix in one day.** This is the kind of high-leverage day the retro flow should optimize for — repeated qualitative articulation → systematized fix.

## Proposed Changes (drafts only — do NOT auto-apply)

### `.claude/skills/issue/SKILL.md` — fix the chronic PR-create bug (6th-cycle proposal)

```diff
@@ Step 5b. Draft PR. @@
-**5b. Draft PR.** Open a draft PR with `Closes #<N>` in the body, linking back to
-the issue. Use `gh pr create --draft --head issue-<N> --body "Closes #<N>"`.
+**5b. Draft PR.** Open a draft PR with `Closes #<N>` in the body, linking back to
+the issue. The branch is fresh (just `git worktree add -b`), so we need an empty
+commit before `gh pr create` will succeed:
+```bash
+cd .claude/worktrees/issue-<N>
+git commit --allow-empty -m "chore(#<N>): open draft PR"
+git push -u origin issue-<N>
+gh pr create --draft --head issue-<N> --body "Closes #<N>"
+```
+If `gh pr create` still fails with "No commits between main and issue-<N>", verify
+the empty commit pushed via `git log --oneline origin/issue-<N>` and retry the push.
```
**Reason:** Fired **11×** today, **16×** yesterday, **30+** across the week. 4-line patch.

### `.claude/settings.json` — SSH-sleep-chain PreToolUse hook (5th-cycle, 28 firings/day)

```diff
@@ "PreToolUse": [ existing Bash hook ] @@
+    {
+      "matcher": "mcp__ssh__ssh_execute",
+      "hooks": [{
+        "type": "command",
+        "command": "python3 -c 'import json,sys,re; ev=json.load(sys.stdin); cmd=ev.get(\"tool_input\",{}).get(\"command\",\"\"); m=re.search(r\"sleep\\s+(\\d+)\",cmd); n=int(m.group(1)) if m else 0; sys.exit(0) if n<=20 else (print(json.dumps({\"decision\":\"block\",\"reason\":f\"sleep {n}s inside ssh_execute exceeds the 30s MCP timeout. Use ScheduleWakeup with delaySeconds={n+5} instead.\"}),file=sys.stderr) or sys.exit(2))'"
+      }]
+    }
```
**Reason:** A hook is the only enforcement mechanism that will actually stop the 28×/day pattern. Memory and rules in `.md` files are ignored by experimenter subagents that don't load CLAUDE.md fully. **Today's `1a04d22` proved hooks work** (the `Bash` matcher hook successfully prevents ad-hoc experiment runs). Apply the same pattern to `mcp__ssh__ssh_execute`.

### `CLAUDE.md` — 5 backlog gotchas (none applied today, all re-fired)

```diff
 ## Gotchas / Known Issues

 - **HF Trainer monkey-patch** in `src/explore_persona_space/train/trainer.py` — fragile, will break if Trainer.__init__ changes
 - **Hard-coded library paths** in `orchestrate/env.py` — cluster-specific
 - **No dataset validation** in `build_phase1_dataset()` — empty QA pairs create silent failures
 - **Tulu pipeline caveat:** midtraining+Tulu results may not generalize to production post-training
+- **Qwen default system prompt:** when `system` field is omitted (or `None`), Qwen2.5
+  chat template auto-injects `"You are Qwen, created by Alibaba Cloud. You are a
+  helpful assistant."`. Pass an explicit empty string `""` for NO system message;
+  pass `"You are a helpful assistant."` for the generic-assistant condition. This
+  was the contamination mechanism behind clean-results #105, #106, #99, #113, #115,
+  #120, #123. Generic and Qwen-default leak to different bystander neighbourhoods.
+- **MooseFS large-write SIGBUS on pod5:** writing 15GB merged-model shards directly
+  to `/workspace` can SIGBUS mid-write. Workaround: copy adapter to `/root/tmp_eval/`,
+  run `merge_lora_adapter` from there, then copy merged dir back to `/workspace`.
+  Affects only multi-GB single-file writes; smaller writes are fine.
+- **MooseFS user quota — distinct from filesystem fullness:** `df -h /workspace`
+  may report TBs free while `Disk quota exceeded` still fires. The mount has
+  per-user soft caps. Diagnose with `mfsgetquota /workspace` and `df -ih /workspace`;
+  fix by deleting unuploaded models via `python scripts/pod.py cleanup`.
+- **vLLM `gpu_memory_utilization=0.85` is too aggressive on shared GPUs:** ValueError
+  "Free memory on device < desired GPU memory utilization." For shared-GPU pods,
+  default to 0.70 or read free memory and scale to ~80% of free.
+- **System TRL vs venv TRL on pod5:** `/usr/local/lib/python3.11/dist-packages/trl/`
+  shadow-imports over the venv. Always invoke via the venv binary directly:
+  `/workspace/explore-persona-space/.venv/bin/python`. Don't `pip uninstall trl`
+  system-side, that breaks more than it fixes.
+- **TRL data-format auto-detect changes loss masking semantics:** `train_lora()`
+  docstring says it expects `{"prompt", "completion"}` JSONL but TRL ≥0.14 silently
+  auto-detects `messages` format too. Loss masking differs: prompt/completion masks
+  user tokens from loss; messages format applies loss to all tokens unless the chat
+  template has proper assistant/user delimiters. **Always pass datasets in
+  prompt/completion format** unless explicit chat-template loss masking is verified.
```
**Reason:** All 5 are 1-2 days unapplied; each re-fired today. The Qwen one is now 6 cycles. **Once the 5 land, propose `analyzer.md` reads CLAUDE.md gotchas before posting any clean-result.**

### `.claude/skills/clean-results/principles.md` — Data audit checklist (still unapplied, 2nd cycle)

```diff
+## Data Audit Checklist (run before posting any clean result)
+
+- [ ] **Persona-conditioning consistency.** If the experiment compares N
+      behaviors trained per-persona, verify that ALL N datasets include the
+      source persona system prompt. If even one is "no system prompt" or
+      generic-assistant, the cross-behavior comparison is contaminated.
+- [ ] **Train→eval overlap.** Grep the training data for any of the eval
+      benchmark's questions/answers. ARC-C, MMLU, HellaSwag are
+      multiple-choice and very prone to letter-only memorization.
+- [ ] **Contrastive negative-set composition.** If a contrastive setup uses
+      a negative set, list the personas in it. The persona being evaluated
+      MUST NOT be in the negative set (otherwise its eval signal is
+      definitionally suppressed during training).
+- [ ] **Loss-masking format.** Confirm whether each training dataset is
+      prompt/completion (user tokens masked) or messages (all-token loss).
+      The two are NOT interchangeable for behavioral interpretation.
+- [ ] **System prompt for evals (exact string).** Record the literal system
+      message used at eval time, not "default" or "see config".
```
**Reason:** Three confounds caught today (alignment-not-persona-conditioned, assistant-in-negatives — both on #112 — and the silent #121 "B=C bit-identical" control issue) — all three would have been caught by this checklist. Now firmly a 2nd-cycle proposal; cost grows daily.

### `.claude/agents/experimenter.md` — wake-up cadence escalation + Step 0 protocol-precedent (2nd cycle, 4th cycle)

```diff
+0. **Match prior protocol exactly.** If the issue references a prior experiment
+   in the same `aim:`, open the most recent `clean-results` issue in that aim
+   and copy its config block (loss-masking, tail_tokens, question set, LR).
+   DEVIATE only if the plan says so explicitly with rationale. If silent, ASK.
+   - Project-specific gotchas: `tail_tokens=0` is the project default (NOT 32);
+     marker direction (always train marker INTO source persona FIRST, then
+     into assistant); use the SAME questions across both phases.
+   - Verify by grepping the latest clean-result issue and stating in the plan:
+     "Precedent #<N> uses X, this run uses X. ✓"
+
@@ Self-pacing rules @@
+**Escalate ScheduleWakeup interval after consecutive "still running" results.**
+If three successive `ScheduleWakeup` firings on the same monitor task return
+"still running, no new completions," double the next delay (e.g., 600s → 1200s
+→ 2400s, capped at 3600s). When >3 sources are on different pods, run health
+checks on the slowest pod between fires; if it's the only blocker, narrow the
+wake-up scope to that pod.
```
**Reason:** Step 0 is the 4th cycle of the same proposal — re-fired today on #102 marker-bridge v2 ("trained the marker into the villain persona, and then into the assistant?" + question-set mismatch). Cadence escalation is 2nd cycle — 5cc had 8 identical "Check progress of issue #112 alignment_v2" firings.

### `.claude/agents/analyzer.md` — auto-add `clean-results` label + auto-move project board card

```diff
@@ When ready to post the clean-result issue: @@
+After `gh issue create` and `gh issue edit --add-label clean-results`, also move
+the issue to the "Clean Results" column on the Experiment Queue project board:
+```bash
+# Find the project item ID for the issue
+gh project item-list <PROJECT_NUMBER> --format json | jq '.items[] | select(.content.number==N)'
+# Move to Clean Results column
+gh project item-edit --id <ITEM_ID> --field-id <STATUS_FIELD_ID> --single-select-option-id <CLEAN_RESULTS_COLUMN_ID>
+```
+(Or invoke the project-board automation hook if one exists.) Verify by visiting the
+board URL.
```
**Reason:** User asked **3 separate times** today "Why is the issue not in the experiment queue board in the clean results column?" — manual `gh cli` was needed. Add to `analyzer.md` so this is the analyzer's job.

### `.claude/agents/consistency-checker.md` (added today) — encode the B=C bit-identical bug

```diff
@@ Checks the agent performs @@
+- **Conditions produce bit-identical numerical outputs.** If two conditions in a
+  control design produce results identical down to every decimal across all
+  metrics, flag the design: either the conditions don't actually differ in their
+  data path (copy-paste bug, same random seed AND same data, same merged adapter
+  reused, etc.) or the conditions are not informative as designed. Today's #121
+  reversed-EM experiment had B and C bit-identical — flagged by the user, root
+  cause was that condition C was effectively a re-run of B with the same merged
+  model.
```
**Reason:** Newly-added agent should encode today's lessons. The B=C bug nearly slipped through — the user spotted it, but a proper consistency-checker should catch it before the user has to.

### `.claude/agents/upload-verifier.md` (added today) — verify clean-results column move

```diff
@@ Hard checks before pass @@
+- Issue is on the "Clean Results" project-board column AND has the
+  `clean-results` label. (The label is necessary but not sufficient — the
+  board automation has been observed to silently fail to move the card.)
```
**Reason:** Same root cause as the analyzer.md item above; the upload-verifier as the gate-of-record should also check this.

### Hook proposal — orphan-worktree gate

```json
{
  "PreToolUse": [
    {
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "cmd=$(echo $TOOL_INPUT | jq -r '.command // empty'); if echo \"$cmd\" | grep -qE 'git worktree (prune|remove)'; then dirty=$(git worktree list --porcelain | awk '/^worktree /{wt=$2} /^bare$/{wt=\"\"} END{print wt}' | xargs -I{} sh -c 'cd {} && git status --porcelain' 2>/dev/null | wc -l); if [ \"$dirty\" -gt 0 ]; then echo 'BLOCKED: One or more worktrees have uncommitted changes. Commit or stash before pruning.' >&2; exit 1; fi; fi"
      }]
    }
  ]
}
```
**Reason:** `pipeline-provenance-system` worktree (920 LOC, 4 days) appears to have been lost from `git worktree list` today. A hook would prevent silent pruning of dirty worktrees.

### Memory updates (drafts)

- **Project memory: `project_2026_04_29_retro.md`** — note the workflow-redesign day; new agents (consistency-checker, upload-verifier, interpretation-critic, follow-up-proposer); first non-zero-application day in 11.
- **Feedback memory: `feedback_capability_eval_excludes_assistant_from_negatives.md`** (proposed 04-28, still not saved). When evaluating a behavior on the source persona, the source persona MUST be excluded from the contrastive negative set.
- **Feedback memory: `feedback_alignment_data_must_be_persona_conditioned.md`** — when training the source persona on a behavior (e.g., misalignment), the training data MUST use the source persona system prompt; otherwise the leakage hypothesis cannot be tested.
- **Feedback memory: `feedback_pod5_migrate_after_3_failures.md`** — when pod5 hits 3+ distinct failures (MooseFS / TRL-shadow / vLLM-tqdm), migrate to pod3 instead of debugging in place. Repeats from 04-28.
- **Feedback memory: `feedback_clean_results_column_manual_move.md`** — the project-board automation does not always move the card; analyzer must verify and fall back to `gh project item-edit`.

## Metrics

- **Corrections by user:** ~18 explicit "no/wait/are you sure/why are/but" messages (across sessions: 56c=5, dd19=6, 5cc=4, 073d=2, 96c=1)
- **User-caught data confounds:** **3** (assistant in contrastive negatives on #112 capability/refusal/sycophancy; alignment data not persona-conditioned on #112; B=C bit-identical control bug on #121)
- **Reruns due to data confounds:** **4+** (alignment_v2, capability_asst_excluded, refusal_asst_excluded, sycophancy_asst_excluded, 60-epoch reversed EM)
- **Agent dispatches:** ~99 (top types: experimenter=22, Explore=18, planner=16, critic=10, analyzer=10, gate-keeper=9, reviewer=7, code-reviewer=1, general-purpose=3)
- **Experiments touched:** ~14 distinct issues (#99, #102, #108, #112, #113, #115, #116, #120, #121, #122, #123, #135, #142, #143)
- **Clean-result issues touched (label `clean-results` or `clean-results:draft`):** 7 (#99, #113, #116, #120, #121, #123, #142)
- **GH issues created today:** ~12 (#143–#150 plus 4 more)
- **Worktrees end-of-day:** **7 (down from 27 yesterday)** — major cleanup, but the `pipeline-provenance-system` worktree (920 LOC) is no longer listed
- **Commits to main:** 2 (`38b9ba0` figure for #116; `1a04d22` workflow redesign +1075/−138)
- **Commits to worktree branches:** 1 (`f16e8d4` codebase refactor)
- **Tool errors (top categories):** 49 OOM, 28 `Command timeout 30000ms`, 22 `Disk quota exceeded`, 12 pod3 handshake timeouts, 11 `No commits between main`, 11 MooseFS, 7 SIGBUS, 6 SIGKILL, 4 `_BaseConfig._VALID_DICT_FIELDS`, 4 tqdm.asyncio, 3 `No space left`
- **ScheduleWakeup calls:** 125 (up slightly from 91 yesterday) — duplicate-firing pattern continues; escalation rule still unapplied
- **Modified/untracked files at end-of-session in main checkout:** 686 (eval results JSON / figures dominate, but draft retros + RESULTS.md likely also dirty)
- **Time spent on debugging vs. research:** Hard to estimate. Pod5 instability cascade (session 073d): ~3-4 hours. #112 confound investigation + reruns: ~2 hours. #121 B=C debugging: ~1 hour. #102 marker-bridge v1→v2 sweeps: ~2-3 hours. Workflow redesign (session 6c597909 + downstream Explore + commit): ~3-4 hours. Roughly: **research ≈ 30%, debugging ≈ 40%, workflow-redesign ≈ 30%**.

## What worked well today (DO NOT regress)

- **The user→workflow-articulation→multi-Explore-audit→bundled-implementation loop** (session 6c597909). Replicate.
- **Adversarial-pattern dispatch on result inconsistency** ("Get an unbiased subagent to verify this experiment was run correctly" — same as 04-28). The new `interpretation-critic` agent is the codified version.
- **Cross-pod parallelism** is now a genuine project default (4th day in a row).
- **Multi-seed-by-default** for cleanly clean-results candidates (issue #115 used 3 seeds in parallel).
- **6 clean-result-labeled issues touched in one day** — best output day in project history. The clean-results pipeline IS the workflow.
- **Memory save on user request worked** (`feedback_no_substring_match`).
- **The PreToolUse hook for ad-hoc experiments works** (added today in `1a04d22`). Prove the pattern: extend it to `ssh_execute`-sleep-chains tomorrow.

---

**Recommendation for tomorrow's session:**
1. Apply the 6th-cycle `gh pr create` empty-commit fix (4 lines) BEFORE doing any other issue-pipeline work.
2. Add the SSH-sleep-chain `PreToolUse` hook to `.claude/settings.json` (the ad-hoc-experiment hook landed today proves the pattern works — extend it).
3. Append the 5 backlog gotchas to CLAUDE.md (Qwen-default + MooseFS + MooseFS-quota + vLLM-mem + TRL-shadow + TRL-format).
4. Audit `pipeline-provenance-system` worktree fate via `git reflog` — recover or formally close.
5. Verify `follow-up-proposer` actually fires automatically after experiment completion (not user-prompted) — this is the user's "I'll forget to use the followup skill" concern.

The cumulative cost of the chronic-infrastructure backlog is now > 1 day of researcher time per week. Today proved that `1a04d22`-style bundled fixes work — apply the same energy to the small-but-chronic items tomorrow.
