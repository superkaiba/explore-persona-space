# Daily Retrospective — 2026-04-28

**Sessions reviewed:** 13 main sessions + ~30 subagent transcripts
**Total user messages (excluding skill/task notifications):** ~290 typed; ~145 of those (≈50%) were monitor/check pings auto-spawned by `ScheduleWakeup` and re-typed manually.
**Commits to main:** **7** (12 pushed in working tree but only 7 made it; 689 untracked files at end of day, 42 modified-but-uncommitted)
**New clean-result issues:** **3** (#121 marker destruction by 2nd-stage SFT — HIGH; #122 no marker transfer via EM — HIGH; #142 JS divergence > cosine — MODERATE; plus #123 promoted to `clean-results:draft`)
**ScheduleWakeup calls:** 125 | **Agent dispatches:** 107 | **ssh MCP calls:** 1047 | **tool errors:** 528

## Summary

Heavy execution day with two unusually consequential user-caught confounds: (1) the **ARC-C contamination retrospective on #75** — the user asked "what midtraining experiments make a persona stupid" and the deep dig revealed evil_correct's #1 ranking on coupling-SFT was a contamination boost from training on ARC-C questions, leading to issue #124 (deconfounded ARC-C with held-out 80/20 split + letter-only answers); and (2) the **Issue #112 alignment data was not persona-conditioned**, so "convergence protects against EM" was reframed once `generate_leakage_data.py` was used and a v2 rerun was done. Three new clean results (#121, #122, #142) were posted and one is in draft (#123). The chronic infrastructure backlog from 9 prior retros fired again essentially unchanged: 8 disk-quota events, 27 `Command timeout 30000ms` SSH MCP firings (still using `sleep N && cmd` chains in `ssh_execute` despite a 4th-cycle proposal), 6 `gh pr create: No commits between main` errors (5th-cycle), 10 pod3 SSH handshake timeouts, 3 worktree-missing-file errors. **Zero of yesterday's proposals were applied — 9th consecutive zero-application day.** New gotcha discovered: TRL silently accepts both `messages` and `prompt`/`completion` JSONL formats but loss-masking semantics differ between them (Issue #112 alignment data was on `messages` format, missing the prompt-only loss masking).

## Proposal Backlog Audit (lead with this — 10th cycle)

| Proposal | First proposed | Days unapplied | Status today |
|---|---|---|---|
| `gh pr create` empty-commit fix in `/issue` Step 5b | 2026-04-25 | **4** | ❌ Fired 6× today across 5 sessions |
| Root + workspace disk preflight (revised: include MooseFS user quota) | 2026-04-23 | **6** | ❌ 8 disk-quota events today, including `scp` failure on session 56cd23f2 |
| `experimenter.md` Step 0: match prior protocol | 2026-04-26 | **3** | ❌ Re-fired today on issue #102 marker bridge: "are we using the same questions to train marker into villain and into assistant?" + tail_tokens=0 verbatim repeat from yesterday |
| Qwen default system-prompt gotcha in CLAUDE.md | 2026-04-24 | **5** | ❌ Re-encountered today on #99 deconfounding (sessions 56cd23f2 + dd19ec71): "Was this using the default qwen system prompt or just a generic assistant prompt?" |
| SSH `sleep N && cmd` 30s timeout block | 2026-04-25 | **4** | ❌ 27 firings today (10× session 073d, 9× session 8ee2bce2, 6× session 56cd2). The `Bash` tool's "Blocked: sleep ..." rule does NOT apply to `mcp__ssh__ssh_execute`, so it fires repeatedly. |
| `merge_marker_adapter` weight-byte assert | 2026-04-23 | **6** | ❌ Not added |
| meta:retro-backlog GitHub issue | 2026-04-25 | **4** | ❌ Not opened |
| Worktree-orphan follow-up gate | 2026-04-26 | **3** | ❌ `pipeline-provenance-system` worktree (940 LOC) untouched for 3rd day; today added 9 NEW issue-* worktrees (issue-100/101/102/104/108/125/139/140 + more in subdir issue-83/issue-84) and none were cleaned |
| MooseFS large-write SIGBUS workaround in `merge_lora_adapter` | 2026-04-27 | **1** | ❌ Same workaround manually re-derived on session 073d ("write merged models to local /root/ to avoid MooseFS SIGBUS on 15GB writes") |
| TRL system-shadow venv guard | 2026-04-27 | **1** | ❌ Re-fired today: 11 `_BaseConfig._VALID_DICT_FIELDS` errors on pod5 in session 073d, manually fixed via `/workspace/explore-persona-space/.venv/bin/python` direct-invocation |
| vLLM gpu_memory_utilization=0.85 → 0.70 default | 2026-04-27 | **1** | ❌ Not adjusted |
| `paper-plots` analyzer re-derivation guard (cherry-pick) | 2026-04-27 | **1** | ❌ Did not fire today (no plot-iteration sessions); not applicable |

**Streak: 10 consecutive zero-application days. 13 backlog items now persistent across 5+ retros.**

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---|---|---|---|
| `ssh_execute` 30s timeout when sub-script has `sleep N && …` chain | 073d (10×), 8ee2 (9×), 56cd (6×), 18b7 (2×) | The MCP tool itself caps at 30000ms regardless of inner `timeout 60`. **Block `sleep \d+ &&` patterns in `mcp__ssh__ssh_execute` calls** (or equivalent: wrap with `nohup … &; sleep ; check`). Add explicit guard in `experimenter.md` and `experiment-runner` skill: "Never put `sleep N` inside `ssh_execute` for N>20; use `ScheduleWakeup` instead." | `.claude/agents/experimenter.md` + `.claude/skills/experiment-runner/SKILL.md` |
| User caught data confound that subagent missed (ARC-C contamination, persona-not-conditioned alignment data, contrastive-set bug) | a327, 5cc, 56c | Three separate confounds in one day, all caught after the fact. Add to `clean-results/principles.md` a checklist: (a) is training data persona-conditioned? (b) does training data overlap eval test set? (c) what's in the contrastive negative set, and does the source persona appear there? | `.claude/skills/clean-results/principles.md` + `analyzer.md` checks |
| Same monitor message re-typed identically 4-8 times by user (e.g., "Check progress of issue #112 alignment_v2 re-run on all 4 pods…") | 5cc (8 dupes), dd19 (7 dupes), 8ee (8 dupes) | The monitor-after-launch pattern is working but ScheduleWakeup is firing the same prompt repeatedly without de-dup. The user appears to have an external loop running these. Suggest: in `experimenter.md`, when `ScheduleWakeup` fires the same monitor prompt 3+ times in a row with the same "still running" outcome, **escalate the wake-up interval** (15min → 30min → 60min) instead of firing again at the same cadence | `experimenter.md` self-pacing rules |
| `gh pr create` "No commits between main and issue-N" — 5th cycle | 5cc, dd19, 8ee, 18b, 56c, 073 (6 sessions) | Same 3-line patch from 04-25 retro. Just apply it. | `.claude/skills/issue/SKILL.md` |
| Disk-quota / MooseFS write-failure cluster | 56c (8 events incl. scp failure), 073d (5 events) | Yesterday's revised preflight (root overlay + MooseFS user quota check) still not landed | `src/explore_persona_space/orchestrate/preflight.py` |
| Qwen default-system-prompt confound (5th cycle) | 56c, dd19 | Add to CLAUDE.md gotchas + in clean-results template add explicit row "System prompt for evals (exact string): ____" | `CLAUDE.md` + `clean-results/template.md` |
| Pod3 SSH handshake timeouts — pod 3 was investigated as down at session 0ae7 (start of day) but never confirmed back up | 0ae7, 5cc (10×) | Add `python scripts/pod.py health --quick pod3` to the `/issue` skill's pre-launch check, and a 1-line gate in `experimenter.md`: "If target pod's last MCP call timed out, run health-quick before launching." | `.claude/skills/issue/SKILL.md` + `experimenter.md` |
| TRL `messages` vs `prompt`/`completion` format silently changes loss-masking | 5cc (the smoking gun on #112) | Document: the `train_lora()` docstring says expects prompt/completion but `bad_legal_advice_6k.jsonl` is messages format. TRL ≥0.14 silently auto-detects, but loss masking differs. **Add an assertion in `train_lora()` that the input format matches the documented expectation, OR explicitly support both with explicit `--data-format` flag.** | `src/explore_persona_space/train/sft.py` |

## Failed Approaches (document to prevent retries)

- **"Just rerun #99 with deconfounded assistant data and don't tell anyone we had a bug" workflow.** The user explicitly said this in session 56cd23f2 ("Don't mention we had this problem just say that we ran everything and got these results"). For future retro: this is fine for issue updates — what matters is the analyzer + reviewer reproducing the corrected result with fresh eyes. But it should be **logged in agent-memory**, not just discarded, so the same confound class doesn't re-fire on a future experiment in the same family. → save `feedback_no_substring_match.md`-style memory: `feedback_capability_eval_assistant_negative_set.md` to remind future runs that the assistant must be **excluded from contrastive negatives** when evaluating its own behavior.
- **MooseFS SIGBUS workaround (write merge to `/root/tmp_eval/` first) re-derived from scratch.** Yesterday's retro proposed encoding this into `merge_lora_adapter` when `MOUNT_TYPE == "fuse"`. Not landed. Today's session 073d had to manually copy 15GB models to `/root/tmp_eval/`, run merge from local, then copy merged back — costing ~2h of manual orchestration on a single experiment.
- **Pod 5 instability cascade.** Session 073d started training marker-bridge v2 on pod5, hit 4 distinct failure modes in sequence: MooseFS SIGBUS during merge → system-TRL shadow `_BaseConfig._VALID_DICT_FIELDS` → tqdm/vLLM 0.11 incompat → GPU isolation breakdown when 4 procs all defaulted to GPU 0. Eventually migrated to pod3 and ran 4× in parallel successfully. Pattern: when a single pod hits 3+ distinct failures, **migrate** instead of debugging in place.
- **`/loop` self-paced monitor producing duplicate firings.** The same exact monitor prompt fired 4-8 times across a session for #112 alignment_v2 re-run. The wake-up was being scheduled at the same delay regardless of whether anything had progressed. Self-pacing should escalate the interval when the prior fire returned "still running, no new completions."

## Workflow Gaps

- **#75 retrospective: ARC-C contamination cost ~1h to discover** because the deep-dig agent had to crawl the pod for actual training data. The clean-results template has a `Setup & hyper-parameters` Reproducibility Card but does NOT have a row for **"Data → eval overlap check"**. Adding this row would have caught contamination at the verifier step. The user explicitly walked through "are these things you're sure about" — Claude was uncertain about whether ARC was in the training data.
- **`pipeline-provenance-system` worktree from 2026-04-26 still has 940 LOC uncommitted.** 3 days now. No cron / hook to flag this. New worktrees added today: issue-100, 101, 102, 104, 108, 125, 139, 140, plus subdir worktrees for 83, 84 — none were merged, none were cleaned. Worktree count climbed from ~24 to 27.
- **#99 update without disclosing the bug** (per user instruction). Acceptable for the issue body but the lesson must propagate to the project. Added explicit instruction-passing back to clean-results principles is the right path.
- **Assistant agent does not check its own memory before launching on pod5.** `feedback_pod5_system_trl_shadow.md` is in `experimenter` agent memory but the experimenter still hit the system-TRL bug today on pod5 (11 firings of `_BaseConfig._VALID_DICT_FIELDS`). Either the memory isn't loading, or the agent isn't looking at it before the launch protocol. Worth verifying the memory load path.
- **Issue #112 alignment data confound should have been caught in plan review.** The plan said "alignment uses 7.5x more examples and no contrastive negatives" — that should have triggered a planner→critic question: "Is this data persona-conditioned? Required for the leakage hypothesis." Add to `critic.md`: when a plan describes per-behavior datasets, flag mismatches in conditioning structure across behaviors.

## Successful Patterns (reinforce these)

- **User-driven confound discovery cadence.** The user in session a327 walked through "What was done?" → "Are you sure?" → "Check the pods for actual files" → "Check for ARC questions in the training data" — methodical, increasingly specific. This is what the gate-keeper / fact-checker pattern is supposed to do automatically. Captured as new memory: `feedback_data_audit_pattern.md` for the `analyzer` and `gate-keeper` agents.
- **Cross-pod parallelism for #112 follow-ups** distributed 5 source-persona convergence experiments across pods 1, 2, 3, 5 simultaneously. Same as 04-27, this is becoming the default for >4-condition sweeps.
- **"Get an unbiased subagent to verify this experiment was run correctly" pattern (session 5cc).** User invoked the adversarial-reviewer mid-experiment when results felt off ("B and C are identical down to every decimal — are you sure everything is being run correctly?"). The audit immediately caught the data-format mismatch. Reinforces `code-reviewer` / `reviewer` adversarial-pattern value.
- **Heavy ScheduleWakeup adoption (125 calls).** Even with the dedup issue, this is much cheaper than synchronous polling. The infrastructure is working; the prompt-template needs escalation logic.
- **Issue #115 multi-seed validation** — user said "Can we run another seed of these to verify variance" → 3-seed run on pod1 → cleaner conclusions. Adopt as default for any single-seed clean result that survives reviewer.
- **3 clean-result issues posted (#121 HIGH, #122 HIGH, #142 MODERATE).** #121 and #122 are categorical findings ("ANY second-stage SFT destroys the marker") with high confidence — exemplary `Confidence: HIGH` claims with binding-evidence framing. Use as exemplars alongside #75 in the clean-results principles.
- **JS divergence as new persona-similarity metric (#142).** Implemented + ran + clean-result-posted within one session (8ee2). The "implement-run-write" loop is fast when the metric is well-specified.

## Proposed Changes (drafts only — do NOT auto-apply)

### `.claude/skills/issue/SKILL.md` — fix the chronic PR-create bug (5th-cycle proposal)

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
**Reason:** Fired 6× today, 16× yesterday, 30+ across the week. 4-line patch.

### `.claude/agents/experimenter.md` — `ssh_execute` sleep-chain block (NEW: 4th-cycle SSH timeout)

```diff
@@ Pre-launch Protocol @@
+**No `sleep N && …` inside `mcp__ssh__ssh_execute`.** The MCP tool caps at 30000ms.
+If you need to wait > 25s and then run a check, use `ScheduleWakeup` (delaySeconds=N+5)
+with the check command in the wake-up prompt. Long waits inside ssh_execute timeout
+silently and waste a tool call.
+
+If the wait must be a single SSH session (e.g., to keep file descriptors alive),
+use `nohup … &` and check via a separate `ssh_execute` after `ScheduleWakeup` fires.
```
**Reason:** 27 firings today, 30+ yesterday. The `Bash` tool's "Blocked: sleep …" guard does NOT apply to `mcp__ssh__ssh_execute`, so the rule never propagated.

### `.claude/agents/experimenter.md` — Step 0 protocol-precedent (3rd-day proposal)

```diff
+0. **Match prior protocol exactly.** If the issue references a prior experiment
+   in the same `aim:`, open the most recent `clean-results` issue in that aim
+   and copy its config block (loss-masking, tail_tokens, question set, LR).
+   DEVIATE only if the plan says so explicitly with rationale. If silent, ASK.
+   - Common precedent gotchas (project-specific): `tail_tokens=0` (project default,
+     not 32); marker direction (always train marker INTO source persona FIRST,
+     then into assistant); use the SAME questions across both phases.
+   - Verify by grepping the latest clean-result issue and stating in the plan:
+     "Precedent #<N> uses X, this run uses X. ✓"
```
**Reason:** Fired again today on #102 marker-bridge v2: "Wait i thought we trained the marker into the villain persona, and then trained the marker into the assistant?" + "are we using the same questions to train the marker into the villain and to train it into the assistant?" — both protocol-precedent gaps that #102 v1 had to discover.

### `CLAUDE.md` — Qwen default + 3 NEW gotchas (5th-cycle, plus new)

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
+  was the contamination mechanism behind clean-results #105, #106, the #99 deconfounding,
+  and again on session 56cd23f2 today. Generic and Qwen-default leak to different
+  bystander neighbourhoods (#113 / #120).
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
+- **TRL data-format auto-detect changes loss masking semantics:** `train_lora()` docstring
+  says it expects `{"prompt", "completion"}` JSONL but TRL ≥0.14 silently auto-detects
+  `messages` format too. Loss masking differs: prompt/completion masks user tokens
+  from loss; messages format applies loss to all tokens unless the chat template
+  has proper assistant/user delimiters. **Always pass datasets in prompt/completion
+  format** unless explicit chat-template loss masking is verified. (Caught on #112
+  alignment data: `bad_legal_advice_6k.jsonl` was in messages format, training
+  loss extended over user tokens, inflating apparent "convergence protects" effect.)
```
**Reason:** Qwen prompt is now a 5th-cycle ask. The MooseFS / vLLM-mem / TRL-shadow gotchas are documented in `experimenter` agent memory but never made it to project-level CLAUDE.md, so subagents without that memory keep rediscovering them. The TRL data-format gotcha is NEW today, caught on #112 by the verifier subagent.

### `.claude/skills/clean-results/principles.md` (or `template.md`) — Data audit checklist

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
**Reason:** Three confounds caught today (ARC contamination, alignment-data not persona-conditioned, assistant in contrastive negatives) — all three would have been caught by this checklist before posting. Each cost 30-90 min of cleanup.

### `.claude/skills/issue/SKILL.md` — pod-health gate before launch

```diff
@@ Step 4: Dispatch Specialist @@
+Before dispatching `experimenter`, confirm the target pod is reachable:
+```bash
+python scripts/pod.py health --quick pod<N>
+```
+If the most recent `mcp__ssh__ssh_execute` call to that pod errored with
+"Timed out while waiting for handshake" within the past 10 minutes, BLOCK
+dispatch and ask the user to check the pod (or fail-over to another pod).
```
**Reason:** Pod3 was investigated as down at start of day (session 0ae7), never explicitly confirmed back up, then session 5cc had 10 pod3 handshake timeouts before falling back to other pods.

### `.claude/agents/experimenter.md` — wake-up cadence escalation

```diff
@@ Self-pacing rules @@
+**Escalate ScheduleWakeup interval after consecutive "still running" results.**
+If three successive `ScheduleWakeup` firings on the same monitor task return
+"still running, no new completions," double the next delay (e.g., 600s → 1200s
+→ 2400s, capped at 3600s). When >3 sources are on different pods, run health
+checks on the slowest pod between fires; if it's the only blocker, narrow the
+wake-up scope to that pod.
```
**Reason:** Session 5cc had 8 identical "Check progress of issue #112 alignment_v2 re-run on all 4 pods" firings (often within 10 min of each other) without escalation. The infrastructure works; the cadence policy doesn't.

### `src/explore_persona_space/train/sft.py` — TRL data-format guard

```diff
+def train_lora(..., expect_format: str = "prompt_completion"):
+    """expect_format: 'prompt_completion' | 'messages' | None
+    Asserts the input JSONL matches. Pass None to skip the check.
+    """
+    if expect_format is not None:
+        first_line = json.loads(open(jsonl).readline())
+        if expect_format == "prompt_completion":
+            assert "prompt" in first_line and "completion" in first_line, \
+                f"Expected prompt/completion JSONL, got keys {list(first_line.keys())}. " \
+                "Loss masking semantics differ between prompt/completion and messages."
+        elif expect_format == "messages":
+            assert "messages" in first_line, ...
```
**Reason:** Caught the #112 alignment confound only after a reviewer subagent dug for it; an assertion would have failed at train-time.

### Memory updates

- `experimenter` agent memory: append `feedback_ssh_execute_no_sleep_chains.md` (rule + reasoning + ScheduleWakeup alternative).
- `experimenter` agent memory: ensure `feedback_pod5_system_trl_shadow.md` is loaded — the agent hit this bug 11× today on pod5 despite the memory existing. Investigate whether the memory load path is broken for spawned `experimenter` subagents.
- `analyzer` agent memory: append `feedback_data_audit_pattern.md` — the user's a327 audit pattern (what was done? are you sure? check the pods for actual files; check for X in the training data).
- `gate-keeper` agent memory: append `feedback_data_format_consistency.md` — when a plan describes per-behavior datasets, ask "is each behavior's data persona-conditioned the same way?"
- Project memory: convert `feedback_no_substring_match` pattern (already saved) into a sister `feedback_capability_eval_excludes_assistant_from_negatives.md` — same shape, different rule.

### Hooks

```json
{
  "PreToolUse": [
    {
      "matcher": "mcp__ssh__ssh_execute",
      "hooks": [{
        "type": "command",
        "command": "python3 -c 'import json,sys,re; ev=json.load(sys.stdin); cmd=ev.get(\"tool_input\",{}).get(\"command\",\"\"); m=re.search(r\"sleep\\s+(\\d+)\",cmd); n=int(m.group(1)) if m else 0; print(json.dumps({\"decision\":\"block\",\"reason\":\"Sleep N>20s inside ssh_execute hits the 30s MCP timeout. Use ScheduleWakeup with delaySeconds=N+5 instead.\"}) if n>20 else \"\",end=\"\")'"
      }]
    }
  ]
}
```
**Reason:** A hook is the only enforcement mechanism that will actually stop the 27×/day pattern. Memory and rules in `.md` files are ignored by subagents.

## Metrics

- **Corrections by user:** 14 explicit "no/wait/instead/are you sure" messages (a327: 1, 56c: 1, dd19: 5, 18b: 2, 5cc: 2, 8ee: 1, 56c: 2)
- **User-caught data confounds:** 3 (ARC-C contamination on #75, alignment data not persona-conditioned on #112, assistant in contrastive negatives on #99 capability eval)
- **Reruns due to data confounds:** 5 (alignment_v2, capability_asst_excluded, refusal_asst_excluded, sycophancy_asst_excluded, ARC-C deconfounded #124)
- **Agent dispatches:** 107 (analyzer, reviewer, code-reviewer, experimenter, planner, critic mostly; 1 retrospective)
- **Experiments run:** ~14 distinct issues touched (#80, #84, #99, #102, #108, #112, #115, #116, #120, #121, #122, #124, #125, #135, #142)
- **Clean-result issues posted:** 3 (#121, #122, #142) + 1 promoted to draft (#123) + 1 expanded survey (#135)
- **GH issues created today (estimate):** ~15 (the gh issue list above shows #124-#142 mostly created today)
- **Worktrees end-of-day:** 27 (up from ~24 yesterday); `pipeline-provenance-system` worktree from 04-26 still 940 LOC uncommitted (3rd day)
- **Commits to main:** 7 (low for a 12+ hour heavy work day; significant work uncommitted)
- **Tool errors:** 528 total (top: `Command timeout 30000ms`=27, `pod3: Timed out`=10, `No such file or directory`=9, `Disk quota exceeded`=8, `No commits between main`=6)
- **Time spent on debugging vs. research:** Hard to estimate cleanly; pod5 instability cascade (session 073d) consumed ~3-4 hours of pod-management time. ARC-C re-investigation cost ~2 hours. #112 alignment data confound cost ~1 hour. Roughly 6-7 hours of the day went to debugging chronic infrastructure that yesterday's retro proposed fixes for.

## What worked well today (DO NOT regress)

- The user→deep-dig→subagent-verifier loop on #75/#112 caught real problems
- Cross-pod parallelism on #112 follow-ups (4 pods simultaneously)
- ScheduleWakeup at 125-call scale stays within budget (~much cheaper than 91 polling calls yesterday)
- 3 clean-results posted with HIGH/HIGH/MODERATE confidence claims
- ARC-C deconfounding turned a potential overclaim into a valuable issue (#124)
- The `paper-plots` skill was invoked cleanly for #142 (JS divergence) without iteration friction

---

**Recommendation for tomorrow's session:** Apply yesterday's PR-create fix (4 lines) and add the 3 NEW gotchas to CLAUDE.md (Qwen default + MooseFS + TRL shadow + TRL data format) before doing any other work. The cumulative cost of the 10-day backlog is now > 1 day of researcher time per week, and growing.
