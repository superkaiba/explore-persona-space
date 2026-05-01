# Daily Retrospective — 2026-04-27

**Sessions reviewed:** 20 main sessions + ~30 subagent transcripts (today is the highest-volume day in 2 weeks)
**Total user messages (excluding skill bootstrap + monitor reschedules):** ~110
**Commits to main:** **21** (zero-streak BROKEN — first non-zero day in 8)
**New clean-result issues:** **3** (#111 EM signature distributional match, #113 Qwen default prompt vulnerability, #116 convergence behavioral leakage null) — best output day in 2 weeks

## Summary

Best research-output day of the past two weeks: zero-commit streak broken, three clean-result issues posted, the cosine-vs-leakage convergence finding extended from 4 sources → 9 sources via aggressive cross-pod parallelism (pods 1, 3, 5 simultaneously), and the #99 capability-loss confound was systematically deconfounded with a `Qwen default vs generic assistant` follow-up that itself produced clean-result #113. **Despite this output**, every chronic infrastructure bug from the 9-day backlog fired again — disk quota (62 events), PR-create-without-commits (16 events), TRL system shadow (11 events), protocol-precedent gap (the same #102 "tail_tokens=0?" mistake from yesterday repeated verbatim today). The pipeline-provenance worktree from 2026-04-26 (~920 LOC uncommitted) was **not touched today**, still orphaned. Two NEW gotchas surfaced: MooseFS SIGBUS during 15GB merge writes on pod5, and vLLM `gpu_memory_utilization=0.85` failing when other workloads share the GPU. Plot-iteration friction was the new high-token-cost failure mode, with several ALL-CAPS user corrections on the issue #77 cherry-pick scatter plots (5+ rounds of "remove diamonds, just circles, only cherry-picked, only labels, label more...").

## Proposal Backlog Audit (lead with this — 9th cycle)

| Proposal | First proposed | Days unapplied | Status today |
|---|---|---|---|
| `gh pr create` empty-commit fix in `/issue` Step 5b | 2026-04-25 | **3** | ❌ Fired 16× across 7 sessions today |
| Root + workspace disk preflight | 2026-04-23 | **5** | ❌ 62 disk-quota / SIGKILL events; **diagnosis was wrong** — see "Failed Approaches" |
| `experimenter.md` Step 0: match prior protocol | 2026-04-26 | **2** | ❌ Verbatim re-fire on #102 today ("tail_tokens=0?", "trained the marker into the villain?") |
| Qwen default system-prompt gotcha | 2026-04-24 | **4** | ❌ Re-encountered on #99 deconfounding (60+ min) and again on #115/#120 |
| SSH `sleep N && cmd` 30s timeout | 2026-04-25 | **3** | ❌ ScheduleWakeup adopted broadly (193 calls) but several sleep-chain firings still in `0efa` |
| `merge_marker_adapter` weight-byte assert | 2026-04-23 | **5** | ❌ Source marker rate=0% on KT/SW eng silent failure; reviewer-agent dispatch needed to find it |
| meta:retro-backlog GitHub issue | 2026-04-25 | **3** | ❌ Not opened |
| Worktree-orphan follow-up gate | 2026-04-26 | **2** | ❌ pipeline-provenance-system worktree untouched, still 920 LOC uncommitted |

**Streak: 9 consecutive zero-application days.** None of yesterday's 8 proposals applied. The same 4 P0 bugs continue to consume compute time daily.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---|---|---|---|
| `gh pr create` fails: "No commits between main and issue-N" | 18b, 56c, 5cc, 073, 0efa, da8a, dd19 (7 sessions) | Insert `git commit --allow-empty` before `gh pr create` (5th-day proposal) | `.claude/skills/issue/SKILL.md` line 261 |
| Disk-quota errors firing on `/workspace` MooseFS — **not just root overlay** | 56c (10×), 0efa (14×), 073 (6×) | (revised) Preflight should check **MooseFS user quota** via `mfsgetquota` *and* root overlay; current proposal misdiagnoses | `src/explore_persona_space/orchestrate/preflight.py` |
| Experimenter ignored protocol-precedent on #102 (tail_tokens, marker direction, masking) **for 2nd day in a row** | 073 | Same proposal as 04-26: add Step 0 to `experimenter.md` requiring match against last clean-result in same `aim:`; add a unit test that the experimenter agent has read the precedent issue | `experimenter.md` |
| Qwen default-system-prompt confound discovered AGAIN on #99 capability-leakage assistant data | 56c, da8a | (4th cycle of this exact ask) Add to CLAUDE.md gotchas; load-bearing for #105/#106/#100/#99/#113/#115 | `CLAUDE.md` |
| MooseFS SIGBUS / "loose object file" errors during 15GB model-shard writes on pod5 | 073 | Document workaround: write merged models to `/root/tmp_eval/` first, then eval; do **not** write large shards directly to MooseFS | new gotcha in `CLAUDE.md`; new memory `feedback_pod5_moosefs_large_writes.md` |
| vLLM `gpu_memory_utilization=0.85` fails when other workloads share GPU ("Free memory < desired") | 56c | Default `gpu_memory_utilization` should adapt to free-memory measurement; or at minimum lower default to 0.70 | `src/explore_persona_space/eval/...` (whichever module instantiates `LLM()`) |
| Plot-iteration loop on #77 took 6+ correction rounds, including 3 ALL-CAPS messages | da8a | (a) Analyzer/plotter agents should read user's first plot-feedback message in full and not "interpret"; (b) when user says "cherry-picked", default to *the previously-shown subset of points*, never re-pick from full data | `paper-plots` skill, `analyzer` agent |
| Source marker rate = 0% silent failure for KT and SW eng on Arm B (#61) | 0efa | (3rd-day proposal) `merge_marker_adapter` should assert weight-bytes > 1 GB and log a clear warning if final adapter tensor norms are zero | `src/explore_persona_space/leakage/runner.py` |

## Failed Approaches (document to prevent retries)

- **"Root overlay disk preflight will fix the disk quota errors" diagnosis from 2026-04-23/26 was incomplete.** The errors today fired on both `/` overlay AND `/workspace` MooseFS user-quota. `df -h /workspace` showed 266T free but `Disk quota exceeded` still fired — this is a **MooseFS user inode/byte quota**, not filesystem-level fullness. Document the actual diagnostic path: `mfsgetquota /workspace` and `df -ih /workspace` (1.1G inodes used 12M, but per-user soft caps may apply). Yesterday's proposed `_check_root_disk()` function would not have caught today's failures.
- **`pip uninstall trl` system-side on pod5 to break the system-shadow.** Re-fired today; same symptom as 04-26. Yesterday's lesson "diagnose system-vs-venv before nuking" did not propagate to the next session because the experimenter agent does not check its memory for `feedback_pod5_system_trl_shadow`.
- **vLLM merge writing 15GB shards directly to `/workspace` MooseFS.** Caused SIGBUS on pod5 (~3 distinct firings during marker-bridge v2). The workaround that finally worked was: copy adapter to `/root/tmp_eval/`, run merge from there, then copy merged back. This pattern should be encoded into `merge_lora_adapter` whenever `MOUNT_TYPE == "fuse"` is detected.
- **Inferring "cherry-picked" plot from full data with heuristic word-overlap.** The analyzer/plotter sub-agent re-derived which personas were "cherry-picked" by re-running word-overlap with `data_scientist` matching `software_engineer` and `evil_scientist` matching `villain`. The user wanted the *literal points labelled in the prior plot* — re-deriving from full data caused 5 correction rounds and several ALL-CAPS messages.

## Workflow Gaps

- **pipeline-provenance-system worktree from 2026-04-26 was not touched today.** Still 920 LOC of `lineage/`, `experiments/`, `scripts/build_claims_index.py`, `scripts/view_dag.py`, plus a 30+ LOC modification to `paper_plots.py`, all untracked. Yesterday's retro flagged this; today's session never returned to it. Risk of `git worktree prune` losing work compounds daily.
- **Issue #61 cosine vs leakage results posted to issue but no clean-result issue file matches that title yet.** The 7-source / 9-source convergence training table that the experimenter compiled (`Check experiment progress on all pods. Grep RESULT lines... Compile the full 9-source table`) was never folded into a `clean-results` GitHub issue. The downstream issue #109 (`Convergence SFT toward source personas increases marker leakage to assistant for 4 of 7 sources, front-loaded but variable timing (LOW confidence)`) is still labelled `clean-results:draft` — needs `/clean-results promote 109` or a manual `/issue 109 --resume`.
- **Auto-monitor loop polling 91 times in session 0efa.** ScheduleWakeup adoption is excellent, but 91 firings on a single session is dominated by 30+ "Check the status of Issue #61" checks where the user manually re-typed near-identical prompts. The self-paced `/loop` flow is meant to handle this — the user reverted to manual polling because the auto-loop's reschedule cadence was off. Session-0efa user messages 8 through 53 are nearly all monitor-pings.
- **GitHub Issues + PRs major outage during the day** (~05:30 PT on 2026-04-27, captured by session ece59e) caused a 5-min disruption. Claude correctly switched to monitoring, but no graceful-degradation behaviour exists — `gh issue ...` calls would have hard-failed during this window if a /issue dispatch had been in flight.

## Successful Patterns (reinforce these)

- **Cross-pod parallelism on #61 follow-up** distributed 5 source-persona convergence experiments across pods 1, 3, 5 simultaneously, doubling throughput vs single-pod execution. The user's prompt "Cant you run on other pods too?" should become a default consideration when a sweep has >4 conditions.
- **Deconfounding workflow on #99 → #115 → #120** worked end-to-end: discovered a Qwen default-prompt confound on #99's assistant data point → reran with explicit `""` system prompt → posted updated #99 → noticed Qwen default and generic assistant leak to *different* bystander neighborhoods → filed #120 to investigate. This is exactly the "experiment-to-experiment hand-off" loop the project workflow is designed for. Capture as exemplar in agent-memory.
- **Adversarial reviewer caught real source-marker bug** on Arm B (KT and SW eng showing 0% marker rate). User's "Get a reviewer to critique the experiment and code and then fix anything it comes up with" → fixed → rerun → corrected results. Good reinforcement of the `code-reviewer` adversarial pattern.
- **3 clean-result issues in one day** (#111, #113, #116) — the analyzer + verifier + reviewer pipeline is producing publish-grade output reliably. #111 specifically reused the issue #75 exemplar shape and passed `verify_clean_result.py` on the first try.
- **ScheduleWakeup widely adopted** across all 5 main sessions (193 calls). Even in session 0efa with 91 firings, this is much cheaper than the previous `sleep 600 && check` SSH chains.
- **User memorialized monitoring rule mid-day**: "whenever you launch an experiment you should make sure to set a monitor to ping you to check soon after and then at progressively longer intervals" → saved as `feedback_monitor_after_launch.md`. Good pattern of inline-memory-saving when the user explicitly asks.

## Proposed Changes (drafts only — do NOT auto-apply)

### `.claude/skills/issue/SKILL.md` — fix the chronic PR-create bug (5th-day proposal)

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
+If `gh pr create` still fails with "No commits between main and issue-<N>", the
+empty commit didn't push — verify with `git log --oneline origin/issue-<N>` and
+retry the push.
```
**Reason:** Fired 16× today across 7 sessions. 5 days unapplied. 3-line patch.

### `.claude/agents/experimenter.md` — protocol-precedent precondition (2nd-day proposal, fired again today)

Add at top of "Pre-launch Protocol":
```diff
+0. **Match prior protocol exactly.** If the issue references a prior experiment in the same `aim:`, open the most recent `clean-results` issue in that aim and copy its config block (loss-masking, tail_tokens, question set, LR). DEVIATE only if the plan says so explicitly with rationale. If the plan is silent, ASK before launching.
+   - Common precedent gotchas: `tail_tokens=0` (project default, not 32); marker direction (always train marker INTO source persona FIRST then into assistant); same questions across both phases.
+   - Verify by grepping the latest clean-result issue for `tail_tokens|marker|loss masking|question set` and stating in the plan: "Precedent #<N> uses X, this run uses X. ✓"
```
**Reason:** This proposal landed yesterday. NOT applied. Verbatim re-fire today: "Wait i thought we trained the marker into the villain persona, and then trained the marker into the assistant?" + "Didn't some of our previous experiments use tail_tokens=0?". The protocol-precedent gap is not a one-off; it is recurring across `aim:6` work specifically.

### `CLAUDE.md` — Qwen default + 2 NEW gotchas

```diff
 ## Gotchas / Known Issues

 - **HF Trainer monkey-patch** in `src/explore_persona_space/train/trainer.py` — fragile, will break if Trainer.__init__ changes
 - **Hard-coded library paths** in `orchestrate/env.py` — cluster-specific
 - **No dataset validation** in `build_phase1_dataset()` — empty QA pairs create silent failures
 - **Tulu pipeline caveat:** midtraining+Tulu results may not generalize to production post-training
+- **Qwen default system prompt:** when `system` field is omitted (or `None`), Qwen2.5 chat template auto-injects `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`. Pass an explicit empty string `""` if you want NO system message; pass `"You are a helpful assistant."` for the generic-assistant condition. This was the contamination mechanism behind clean-results #105, #106, and the #99 deconfounding rerun. Generic and Qwen-default leak to different bystander neighbourhoods (see #113 / #120).
+- **MooseFS large-write SIGBUS on pod5:** writing 15GB merged-model shards directly to `/workspace` (mfs#eur-is-4.runpod.net:9421) can SIGBUS mid-write. Workaround: copy adapter to `/root/tmp_eval/`, run `merge_lora_adapter` from there, then copy merged dir back to `/workspace` via `cp -r`. Affects only multi-GB single-file writes; smaller writes are fine.
+- **MooseFS user quota — distinct from filesystem fullness:** `df -h /workspace` may report TBs free while `Disk quota exceeded` still fires. The mount has per-user soft caps. Diagnose with `mfsgetquota /workspace` and `df -ih /workspace` (inodes); fix by deleting unuploaded models via `python scripts/pod.py cleanup`.
+- **vLLM `gpu_memory_utilization=0.85` is too aggressive when other workloads share the GPU:** "ValueError: Free memory on device (X/Y GiB) on startup is less than desired GPU memory utilization (0.85, Z GiB)." For shared-GPU pods, default to 0.70 or read free memory and scale to ~80% of free.
+- **System TRL vs venv TRL on pod5:** `/usr/local/lib/python3.11/dist-packages/trl/` shadow-imports over the venv (`_BaseConfig._VALID_DICT_FIELDS` error). Always activate the venv before launching (`/workspace/explore-persona-space/.venv/bin/activate`); don't `pip uninstall trl` system-side, that breaks more than it fixes.
+- **vLLM 0.11.x import hang:** importing the project library transitively pulls in vLLM 0.11 and hangs at module load. Workaround until vLLM is upgraded: write standalone scripts with inline vLLM patches (see `feedback_vllm011_tqdm_compat.md`).
```
**Reason:** Qwen prompt is now a 4th-cycle ask, load-bearing for 6 issues. The 3 NEW (MooseFS, vLLM gpu-mem, pod5 TRL) are documented in 2 separate agent-memory files but never made it to the project-level CLAUDE.md, so subagents without that memory keep rediscovering them.

### `src/explore_persona_space/orchestrate/preflight.py` — REVISED disk preflight (5th-cycle proposal, diagnosis fix)

```diff
+def _check_disk(min_root_gb: int = 20, min_workspace_gb: int = 100) -> tuple[bool, str]:
+    """Check both root overlay AND MooseFS workspace quota.
+
+    On RunPod: `/` is a small overlay; large writes spill there when caches aren't
+    redirected and SIGKILL silently when full.
+    `/workspace` is MooseFS with per-user soft quotas; `df -h /workspace` may
+    report TB free while `Disk quota exceeded` still fires.
+    """
+    msgs, ok = [], True
+    # Root overlay (df -BG --output=avail /)
+    out = subprocess.run(["df", "-BG", "--output=avail", "/"], capture_output=True, text=True, check=True)
+    root_free = int(out.stdout.splitlines()[1].strip().rstrip("G"))
+    if root_free < min_root_gb:
+        ok = False
+        msgs.append(f"Root overlay only {root_free}G free (need >= {min_root_gb}G); risk of SIGKILL during merge.")
+    # MooseFS user quota (try mfsgetquota; fall back to a write probe)
+    try:
+        q = subprocess.run(["mfsgetquota", "/workspace"], capture_output=True, text=True, check=False)
+        if "Disk quota" in (q.stdout + q.stderr):
+            ok = False
+            msgs.append(f"MooseFS user quota near limit on /workspace: {q.stdout.strip()}")
+    except FileNotFoundError:
+        pass
+    # 1MB write probe — catches per-user soft caps that df doesn't show
+    probe = "/workspace/.preflight_probe"
+    try:
+        subprocess.run(["dd", "if=/dev/zero", f"of={probe}", "bs=1M", "count=1"], capture_output=True, check=True)
+        os.unlink(probe)
+    except subprocess.CalledProcessError as e:
+        ok = False
+        msgs.append(f"/workspace write probe failed: {e.stderr.decode() if e.stderr else 'unknown'}")
+    return ok, "; ".join(msgs) if msgs else "Disk: root + workspace OK"
```
**Reason:** Prior 4 cycles of this proposal (`_check_root_disk`) **misdiagnosed** the failure mode. Today's MooseFS quota errors fired with 266T showing free in `df`. The real check needs `mfsgetquota` AND a write probe. **Do not apply the old `_check_root_disk()` proposal — apply this revised one.**

### `src/explore_persona_space/leakage/runner.py` — merge-marker silent-failure assert

```diff
+# Inside merge_marker_adapter() after merge:
+merged_dir = Path(out_dir)
+total_bytes = sum(f.stat().st_size for f in merged_dir.rglob("*.safetensors"))
+if total_bytes < 1_000_000_000:  # 1 GB sanity check (Qwen-7B merged ≈ 15 GB)
+    raise RuntimeError(
+        f"Merged adapter at {merged_dir} is only {total_bytes / 1e9:.2f} GB — "
+        f"adapter probably failed to apply. Check adapter_config.json + base model load. "
+        f"This silently produces 0% marker rate at eval time."
+    )
```
**Reason:** Today's KT and SW eng Arm B runs reported 0% source marker rate, identical to a failed-merge artifact. Required dispatching a `code-reviewer` agent to find the cause. Has fired on 2 of last 4 days. Same proposal landed 04-23 (5 days unapplied).

### Memory updates

- **Add to `.claude/agent-memory/experimenter/`:** `feedback_pod5_moosefs_large_writes.md` — write merged models to `/root/tmp_eval/` first; large `/workspace` writes SIGBUS.
- **Add to `.claude/agent-memory/experimenter/`:** `feedback_vllm_gpu_memory_utilization.md` — default 0.85 fails on shared GPUs; lower to 0.70 or measure free first.
- **Update `.claude/agent-memory/retrospective/project_unapplied_backlog.md`** — bump streak to 9 consecutive zero-days; promote MooseFS write-pattern + vLLM gpu-mem to P0; revise root-disk diagnosis.
- **Add to `.claude/agent-memory/retrospective/`:** `project_2026_04_27_retro.md` — capture today's 21-commit + 3-clean-result success on top of unbroken 9-day proposal-application gap; document the **diagnosis-was-wrong** finding for prior root-overlay preflight proposal.
- **Add to `.claude/agent-memory/analyzer/`:** `feedback_user_says_cherry_picked.md` — when user says "cherry-picked", they mean *the literal points already labelled in the prior plot*; do not re-derive via word-overlap heuristics. Confirmed by 5+ correction rounds in #77 plot iteration today.

### Hooks

**Still no new hooks proposed.** All prior hook proposals (5+) remain unapplied. Continue per 2026-04-24 retro: do not add another hook ask until at least one prior hook has landed.

### One meta-proposal: file the meta:retro-backlog issue

After 3 days of "should we file `meta:retro-backlog`?" and no action, **draft the issue body in this retro and stop re-proposing it** until the user replies yes/no:

> **meta:retro-backlog** — bundle of 8+ retro proposals open for 2-9 days. Each is single-digit lines. Pick which to apply / which to drop, comment with verdict, close.
>
> Per-proposal links:
> 1. `/issue` Step 5b PR-create empty-commit fix (5 days unapplied, fired 30+ times this week)
> 2. `experimenter.md` Step 0 protocol-precedent (2 days unapplied, fired on #102 twice)
> 3. CLAUDE.md Qwen default-prompt gotcha (4 days unapplied, fires every clean-result)
> 4. CLAUDE.md MooseFS large-write workaround (NEW, fired 3 times today)
> 5. CLAUDE.md vLLM gpu_memory_utilization (NEW, fired today)
> 6. preflight.py MooseFS quota check (5th cycle, REVISED diagnosis)
> 7. runner.py merge-bytes assert (3 days unapplied, fired on Arm B today)
> 8. (New) `analyzer.md` cherry-pick literal-prior-plot rule (NEW today)

If the user labels this `meta:retro-backlog`, retros stop re-proposing until the issue is closed.

## Metrics

- **Commits to main:** 21 (zero-streak BROKEN)
- **New clean-result issues:** 3 (#111, #113, #116)
- **Issues touched:** 11 (#61, #69, #99, #100, #101, #102, #112, #115, #117, #118, #119, #120 + 3 new clean-results)
- **Subagent dispatches:** 100 (28 experimenter, 16 planner, 16 Explore, 10 critic, 9 gate-keeper, 9 analyzer, 7 reviewer, 2 general-purpose, 2 code-reviewer)
- **ScheduleWakeup calls:** 193 across 5 main sessions (91 in 0efa alone)
- **SSH/MCP tool errors:** 250+ (62 disk-quota, 16 PR-create-no-commits, 11 TRL system shadow, 9 SIGKILL, 4 OOM, 3 sha1 write errors)
- **PR-create failures:** 16 (chronic, 5th-day proposal not applied)
- **User corrections (estimated):** ~15 (4 protocol-precedent, 5 plot-iteration, 3 confound-rediscovery, 3 misc)
- **ALL-CAPS user messages:** 3 (all on plot iteration in da8a)
- **Estimated time on debugging vs. research:** ~50% debugging (better than yesterday's 70%, worse than the 30% on best days)
- **External outage:** GitHub Issues + PRs degraded ~05:30 PT for ~5 min

## One-line for the morning

If you only apply ONE thing tomorrow, **it's the `gh pr create` empty-commit fix** — 5 days unapplied, 16 firings today, 30+ this week, 3-line patch. It's blocking every fresh dispatch. The protocol-precedent rule is the second-best ROI; together they would have saved ~90 minutes of GPU and replanning today. **And: revise the root-disk preflight diagnosis before applying** — yesterday's proposal would not have caught today's MooseFS-quota failures.
