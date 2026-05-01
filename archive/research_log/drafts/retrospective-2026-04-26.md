# Daily Retrospective — 2026-04-26

**Sessions reviewed:** 4 main sessions (073, 18b, 56c, bdbf) + 21 subagent transcripts
**Total user messages (excluding command-caveat / task-notification noise):** ~120
**Commits to main:** **0** (8th day in last 14 with zero or near-zero commits despite substantial work)
**New clean-result issues:** **0** (assistant-data rerun on #105 in flight; #104 search produced data but no clean-result yet; marker-bridge v2 still running)

## Summary

Low-throughput day dominated by debugging. Three running experiments (#102 marker-bridge v2 on pod5, #104 distributional-match prompt search on pod5, #100 deconfounded assistant rerun on pod1) plus a substantial Marin-inspired pipeline-provenance build in a worktree. **None of it was committed.** Most of the chronic infrastructure bugs from the past 4 retros fired again — disk quota, PR-create-without-commits, SSH sleep-blocking, plus two new ones (system TRL vs venv conflict on pod5; vLLM 0.11 import hang). The marker-bridge v1 was the highest-friction experiment of the week — user had to correct the experimenter on **four basic design facts** mid-run (marker direction, masking strategy, tail_tokens, question reuse), which forced a v2 redesign and a pod swap.

## Proposal Backlog Audit (lead with this — 8th cycle)

| Proposal | First proposed | Days unapplied | Status today |
|---|---|---|---|
| Root-disk preflight (`df -h /` + WandB cache redirect) | 2026-04-23 | 4 | ❌ Fired again on pod1 (multiple "Disk quota exceeded" / SIGKILL during merge); fired on pod5 too via scp |
| `gh pr create` placeholder commit in `/issue` Step 5b | 2026-04-25 | 2 | ❌ Fired again on #102 (worktree already had path missing too) |
| Qwen default system-prompt fact in CLAUDE.md | 2026-04-24 | 3 | ❌ Re-encountered in #100 rerun work — load-bearing for confound mechanism |
| SSH `sleep N && cmd` 30s timeout | 2026-04-25 | 2 | ❌ Fired ~6 times on session 073 alone |
| `merge_marker_adapter` weight-byte assert | 2026-04-23 | 4 | (No fresh fire today, but pre-existing sub-1GB merges blocking analysis on issue #61 still un-blocked) |
| meta:retro-backlog GitHub issue | 2026-04-25 | 2 | ❌ Not opened |

**Streak: 8 consecutive zero-application days.** The same 4 P0 bugs continue to consume compute time daily.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---|---|---|---|
| Disk quota exhaustion silently kills processes (`scp ... close: Disk quota exceeded`, vLLM merge SIGKILL on pod1, base64/cat write errors) | 073, 56c | Add `_check_root_disk()` to preflight + redirect WandB cache to `/workspace` (4th cycle of this exact ask) | `src/explore_persona_space/orchestrate/preflight.py` + `pod bootstrap` |
| `gh pr create` fails: "No commits between main and issue-N" | 073, 18b, 56c (3 sessions in one day) | Insert empty commit `git commit --allow-empty -m "chore(#<N>): open draft PR"` before `gh pr create` in `/issue` Step 5b | `.claude/skills/issue/SKILL.md` line 261 |
| SSH `sleep N && cmd` chains hit 30s MCP timeout | 073 (~6×) | Use `ScheduleWakeup` for delays > 25s, then a fresh `ssh_execute` for the query | `experimenter.md`, `experiment-runner` SKILL |
| Experimenter ran wrong protocol — user corrected 4 design facts mid-run on #102 (marker direction; tail_tokens=32 vs project standard 0; different questions across phases; loss masking) | 073 | (a) `experimenter` agent must re-confirm protocol against last `clean-results` issue in same area before launching; (b) plan should embed authoritative loss-masking + tail-tokens config from precedent issue, not redefine them | `experimenter.md`, planner template |
| System-installed TRL on pod5 conflicting with venv TRL (`_BaseConfig._VALID_DICT_FIELDS`) | 073 | Bootstrap should `pip uninstall -y trl` system-side OR pin venv with `--isolated` invocation. Project venv is shadowed by `/usr/local/lib/python3.11/dist-packages/trl/` | `pod bootstrap` script + experimenter venv-activation reminder |
| vLLM 0.11 import hangs when pulled in via library path | 18b (issue #104 phases 0-3) | Document standalone-script-with-inline-patches as the **expected workaround** until vLLM 0.11.x is replaced | new `feedback_vllm011_*` memory exists; promote to gotcha in CLAUDE.md |
| Auto-monitor loop fired 20+ "Monitor marker bridge..." invocations on session 073 | 073 | This is the new self-paced `/loop` flow; mostly working, but currently spends ~40-60% of tokens on polling. Acceptable structurally. No new ask. | (no change) |

## Failed Approaches (document to prevent retries)

- **Marker bridge v1 with `tail_tokens=32` and disjoint question sets between Phase 1 (villain) and Phase 2 (assistant)**: gave a clean null but wasn't the design the user wanted. Forced a full v2 rerun with `tail_tokens=0` and question-matched phases. Document: `experimenter.md` should flag "established protocol from prior clean-result issue" as a hard precondition before launching variants.
- **Removing system-level TRL on pod5 to fix import conflict**: `pip uninstall trl` system-side broke the venv (which shadow-imported it). Had to reinstall. Document: pod-side fixes need to be diagnosed as system-vs-venv before nuking.
- **Trying to merge LoRA on pod1 with `<10GB` free on `/`**: vLLM merge writes spilled to root overlay and SIGKILL'd. Switching to pod5 was the actual fix; partial cleanup never reclaimed enough.

## Workflow Gaps

- **`bdbf` Marin-inspired pipeline-provenance system** built ~920 LOC of new code (`pipeline/{runner,step,provenance,viewer}.py`, `experiments/`, `scripts/build_claims_index.py`, `scripts/view_dag.py`, 34 tests passing) in `.claude/worktrees/pipeline-provenance-system/` — **never committed**, no PR opened, no follow-up issue filed. Risk of being garbage-collected with a worktree prune.
  - Proposed fix: `/issue` skill (or a new lightweight `/worktree-followup` skill) should detect "untracked work in `.claude/worktrees/X` after session ends" and prompt the user to either commit, file an issue, or explicitly drop.
- **#104 phase 3 search posted result (bureaucratic-authority prompts ≠ villain prompts) but no analyzer/clean-result dispatched.** The `/issue` flow stalled at `epm:progress v2`. Need a manual `/issue 104 --resume` to advance to analyzer.
- **Issue #102 v2 results not posted yet** at session end (04-26 21:01 UTC). Marker-bridge v1 had clean null on 04-25; v2 redesign is in flight. The auto-experiment loop is supposed to land it, but no `epm:results v2` marker exists yet.

## Successful Patterns (reinforce these)

- **Deconfounding rerun for clean-result #105 followup** worked smoothly via the existing `/issue 100 --resume` flow. User's "should we rerun anything given the bug?" → "rerun the assistant data point" → 111-persona leakage rerun launched without re-planning friction.
- **Adversarial-planner caught real bugs again on #102 plan**: fact-checker initially flagged "[ZLT] missing from data" but verified it WAS present on villain completions, illustrating the verifier can self-correct. Critic on #104 caught underspec'd reference distribution — user added Fitness D (KL logits) before approval.
- **`bdbf` Marin exploration → architectural plan → impl-in-worktree pattern** is exactly the right shape for adopt-from-other-project work. Just needs a commit/issue follow-up gate.
- **ScheduleWakeup adoption** stayed strong — 20 firings on session 073, all for the marker-bridge monitoring loop. Token spend on monitoring continues to drop vs `/loop`-with-sleep.

## Proposed Changes (drafts only — do NOT auto-apply)

### `.claude/skills/issue/SKILL.md` — fix the chronic PR-create bug

```diff
@@ Step 5b. Draft PR. @@
-**5b. Draft PR.** Open a draft PR with `Closes #<N>` in the body, linking back to
-the issue. Use `gh pr create --draft --head issue-<N> --body "Closes #<N>"`.
+**5b. Draft PR.** Open a draft PR with `Closes #<N>` in the body, linking back to
+the issue. The branch is fresh (just `git worktree add -b`), so we need an empty
+commit before `gh pr create` will succeed:
+```
+cd .claude/worktrees/issue-<N>
+git commit --allow-empty -m "chore(#<N>): open draft PR"
+git push -u origin issue-<N>
+gh pr create --draft --head issue-<N> --body "Closes #<N>"
+```
+If `gh pr create` still fails with "No commits between main and issue-<N>", the
+empty commit didn't push — verify with `git log --oneline origin/issue-<N>` and
+retry the push.
```
**Reason:** Fired 3 times today (#100, #102, #104) and 3 times yesterday. 6 total firings in 48 hours. 3-line patch.

### `src/explore_persona_space/orchestrate/preflight.py` — root-disk preflight

```diff
+def _check_root_disk(min_gb: int = 20) -> tuple[bool, str]:
+    """Fail if `/` (root overlay) has less than `min_gb` free.
+    On RunPod containers, `/` is a small overlay; large writes spill there
+    when caches aren't redirected and SIGKILL silently when full."""
+    out = subprocess.run(
+        ["df", "-BG", "--output=avail", "/"], capture_output=True, text=True, check=True
+    )
+    free_gb = int(out.stdout.splitlines()[1].strip().rstrip("G"))
+    if free_gb < min_gb:
+        return False, f"Root overlay only {free_gb}G free (need >= {min_gb}G); risk of silent SIGKILL during merge/eval. Run `python scripts/pod.py cleanup --all`."
+    return True, f"Root overlay: {free_gb}G free"
```
**Reason:** 4th cycle proposing this. Fired again on pod1 today (3 distinct error signatures). Cumulative cost: ~3 GPU-hours of debugging across the streak.

### `CLAUDE.md` — Qwen default system prompt + new gotchas

```diff
 ## Gotchas / Known Issues

 - **HF Trainer monkey-patch** in `src/explore_persona_space/train/trainer.py` — fragile, will break if Trainer.__init__ changes
 - **Hard-coded library paths** in `orchestrate/env.py` — cluster-specific
 - **No dataset validation** in `build_phase1_dataset()` — empty QA pairs create silent failures
 - **Tulu pipeline caveat:** midtraining+Tulu results may not generalize to production post-training
+- **Qwen default system prompt:** when `system` field is omitted (or `None`), Qwen2.5 chat template auto-injects `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`. Pass an explicit empty string `""` if you want NO system message. This was the contamination mechanism behind clean-result #105.
+- **System TRL vs venv TRL on pods:** `/usr/local/lib/python3.11/dist-packages/trl/` shadow-imports over the venv on pod5 (`_BaseConfig._VALID_DICT_FIELDS` error). Always activate the venv before launching (`/workspace/explore-persona-space/.venv/bin/activate`); don't `pip uninstall trl` system-side, that breaks more than it fixes.
+- **vLLM 0.11.x import hang:** importing the project library (anything that transitively imports vLLM 0.11 with `from explore_persona_space.axis.prompt_search.fitness import vLLMEngine`) hangs at module load. Workaround until vLLM is upgraded: write standalone scripts with inline vLLM patches (see `feedback_vllm011_tqdm_compat.md`).
```
**Reason:** Qwen prompt is 4th-cycle ask (load-bearing for #105, #106, #100 rerun); the other two are NEW today and cost real time.

### `.claude/agents/experimenter.md` — protocol-precedent precondition

Add at top of "Pre-launch Protocol":
```diff
+0. **Match prior protocol exactly.** If the issue references a prior experiment in the same `aim:` (e.g. #102 -> #80 marker work, #100 -> #96 robustness), open the most recent `clean-results` issue in that aim and copy its config block (loss-masking, tail_tokens, question set, LR). DEVIATE only if the plan says so explicitly with rationale. If the plan is silent, ASK before launching.
```
**Reason:** The user had to correct 4 fundamental design facts on #102 mid-run (marker direction, tail_tokens, masking, question reuse). Each correction cost 30+ min of GPU and replanning.

### `.claude/skills/issue/SKILL.md` — worktree follow-up gate

Add a new Step 9 (or expand the cleanup step):
```diff
+### Step 9 (post-session housekeeping): Worktree follow-up
+
+If a worktree under `.claude/worktrees/<name>` was used during the session and
+the user closes/quits without (a) opening a PR or (b) committing, log a
+`<!-- epm:worktree-orphan -->` comment on the issue listing the untracked
+files and their LOC. The user gets a single-glance reminder next time they
+open the issue.
+
+If the worktree is NOT tied to an issue (e.g. exploratory like
+`pipeline-provenance-system`), prompt the user once at session end:
+"Commit, file an issue, or drop?"
```
**Reason:** ~920 LOC of pipeline/provenance system from `bdbf` is sitting untracked in a worktree. If `git worktree prune` runs, the work vanishes. This pattern has happened before with exploratory worktrees.

### Memory updates

- **Add to `.claude/agent-memory/experimenter/`:** `feedback_pod5_system_trl_shadow.md` — system TRL 1.0.0 at `/usr/local/lib/python3.11/dist-packages/trl/` shadow-imports; activate venv first; don't uninstall system-side.
- **Update `.claude/agent-memory/retrospective/project_unapplied_backlog.md`** — bump streak to 8 consecutive zero-days; add the worktree-orphan pattern (NEW); record marker-bridge protocol-precedent gap (NEW).
- **Add to `.claude/agent-memory/retrospective/`:** `project_2026_04_26_retro.md` — capture today's 0-commit-but-substantial-uncommitted-work pattern + 4 P0 bugs that fired again + 2 new ones.

### Hooks

**No new hooks proposed.** Hooks have been zero-application across the entire backlog (5 distinct hook proposals never landed). Per the 2026-04-24 retro's "drop zombie proposals" recommendation, do not add another hook ask.

## Metrics

- **Commits to main:** 0
- **New clean-result issues:** 0 (one in flight, one stalled at progress, one running)
- **Issues touched (epm: markers):** 3 (#100, #102, #104) + 1 prepared but never launched (provenance worktree)
- **Subagent dispatches:** ~21 (planner, fact-checker, critic, gate-keeper, experimenter, analyzer, reviewer across the 4 sessions)
- **SSH MCP errors (timeouts + disk quota):** 12+
- **PR-create failures:** 3 (chronic)
- **Tool-error count by session:** 073=66, bdbf=248 (mostly transient pytest/lint output, not real errors), 18b=20, 56c=11
- **`Monitor marker bridge ...`** auto-loop firings: 20
- **Estimated time on debugging vs. research:** ~70% debugging (the marker-bridge experiment dominated session 073, almost all of it spent on disk-quota / TRL-conflict / GPU-isolation issues)

## One-line for the morning

If you only apply ONE thing tomorrow, **apply the empty-commit fix in `/issue` Step 5b** — 3 lines, fires 3× a day, blocks every fresh dispatch. The root-disk preflight is the second-best ROI but is a 4th-cycle ask and clearly never lands; consider whether to file `meta:retro-backlog` and let the user choose explicitly.
