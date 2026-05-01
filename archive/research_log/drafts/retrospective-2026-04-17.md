# Daily Retrospective — 2026-04-17

**Sessions reviewed:** 14 top-level sessions + 17 subagent sessions (31 active JSONL files today)
**Total user messages:** ~397 raw (~45 genuine across all sessions; many are tool-result wrappers)
**Git commits today:** 18 (training optimization stack + Tulu config fix + benchmark harness)
**GitHub issues touched today:** 12 (#32–#43); 7 closed, 3 running, 2 proposed
**Drafts written:** 1 new (`2026-04-17_tier2_liger_verification.md`)

## Summary

Today was dominated by a single large deep-dive: **training pipeline optimization** that cascaded from issue #36 (Tier 1 perf wins) into #37 (cleanup), #38 (packing pilot, still running), #39 (realistic SFT benchmark, still running), #40 (Tier 2 verification), #41 (Tulu config fix), #42 (submodule bump — moot), and #43 (runtime Liger verification). The session shipped real wins (DPO +20%, packing +293% on short data, Tulu `launch_stage.py` now works) but revealed a **new and serious failure mode**: two specialist subagents (#40, #43) confidently reported false infrastructure claims (submodule pin, valid config flags) that were only caught when the research-pm agent verified via AST parsing. Separately, **3 days in a row** now of retrospective proposals going unimplemented — the SessionStart hook, git-push pod-sync hook, and yesterday's CLAUDE.md gotchas were never applied, and yet the retro keeps producing them.

Smaller tracks today: the `/issue <N>` skill was designed and shipped (early morning); an eval-records infrastructure build (records.py) was started but the implementer was cut off mid-work with unresolved import bugs; and the user spent ~3 sessions trying to persist `max` effort level + Opus 1M context across restarts — resolved only after discovering the undocumented `CLAUDE_CODE_EFFORT_LEVEL` env var workaround for a known settings.json serialization bug.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---------|----------|-------------|-------------|
| **Sleep command repeatedly blocked** — 15 total blocks across 8 different subagents. Agents keep issuing `sleep N && command` despite the tool-error message explicitly telling them to use `Monitor` or `run_in_background`. Usually during experiment monitoring/polling. | 37d95c4c subagents: a121f4cf, a15b861d, a3f066e0, a56d5437, a8d658bc, ac2ff91b, aeb0ced8, afb513c6 | Add an explicit "Never use `sleep N && cmd` for polling — use `Monitor` with an until-loop or `run_in_background`" rule to `experimenter.md` monitoring section. Include the exact pattern that works. | `.claude/agents/experimenter.md` |
| **"File has not been read yet"** — 13 occurrences. Agents try to Edit/Write without a preceding Read. Persistent even after yesterday's fixes. Often happens during rushed multi-file refactors. | Most subagents | Add a PreToolUse hook on `Edit`/`Write` that logs a warning (not a block) to telemetry when the target file has no Read in the last 100 tool calls. Also: reinforce the rule at the top of `implementer.md` and `experimenter.md`. | `.claude/settings.json`, agent defs |
| **"File has been modified since read"** — 8 occurrences. Race condition caused by the ruff PostToolUse hook rewriting Python files after Edit, which then invalidates the implementer's cached read for the next edit. | 7aa609f0, 8acd7099 | Update the ruff PostToolUse hook to echo a `File modified by ruff:` notice, OR add an auto-re-read step. Alternatively, note in `implementer.md` that after editing a `.py` file, always Read it again before the next Edit because the ruff hook will have rewritten it. | `.claude/settings.json` or `.claude/agents/implementer.md` |
| **Specialist agents producing false infrastructure claims** — NEW PATTERN. Issue #40's Tier 2 verifier reported the open-instruct submodule was at pre-Liger pin (wrong); Issue #43's runtime Liger verifier said `use_liger_kernel` wasn't a valid field (wrong). Both refuted only when research-pm parsed the submodule's `FlatArguments` dataclass via AST. Root cause: import-based probes silently fail due to unrelated ImportErrors (`olmo_core` missing) and get mis-interpreted as field-absent. | 37d95c4c subagents a1ab2edf (Tier 2 proposal) + a3f066e0 (runtime verify) | Add to `experimenter.md` and `implementer.md`: "When verifying whether a symbol/flag/field exists in a submodule or third-party library, prefer AST parsing (`ast.parse` + inspect class/function defs) over `import + hasattr` or `dataclasses.fields(...)`. Import-based probes produce false negatives when upstream deps are missing." Also: "Before declaring a verdict on infrastructure claims, run a runtime smoke, not just a probe." | `.claude/agents/experimenter.md`, `.claude/agents/implementer.md` |
| **Retro proposals not applied — 3 days in a row** — Yesterday's CLAUDE.md gotchas (RunPod ports, HF public repo, upload safety, worktree clean state, uv-in-nohup), experimenter.md Design Brief + Monitoring Cadence + `/root/.local/bin/uv`, research-pm.md Gate-Keeper Trigger + Agent Name Reference, SessionStart hook, git-push hook — NONE applied today. Same list was proposed 2026-04-15 and 2026-04-16. | Systemic | **Make the retrospective agent fail loudly**: after writing the draft, always (a) print a machine-readable applied/unapplied diff vs. last 2 retros to stdout; (b) open a GitHub issue with `--label retrospective` + checkboxes for each proposal. Alternatively: add a SessionStart hook that refuses to start work until the most recent retro's proposals are either applied or explicitly declined (with a `git notes`-style marker). | `.claude/agents/retrospective.md`, `.claude/settings.json` |
| **GitHub Projects (classic) deprecation warning** — 32 occurrences. Every `gh issue view`/`gh issue list` call that surfaces project data prints the deprecation. Noisy context pollution, and agents sometimes treat the warning as an error ("Exit code 1" appears alongside). | All gh-using sessions | Wrap `gh` calls to strip the deprecation line, or switch to `gh issue --json` (doesn't emit the warning). Simplest: add a shell alias or pod-wide `GH_GRAPHQL_SUPPRESS_DEPRECATIONS=1` if supported; otherwise filter with `2>&1 \| grep -v 'Projects (classic)'`. | `CLAUDE.md` common commands, or `.claude/settings.json` env |
| **Flash-attn / Liger-kernel missing on pods** — 12 errors (7 flash_attn, 5 liger_kernel). Tier 1 benchmarks had to downgrade expectations because installed env on pods hasn't been updated after adding the optional deps. Pod bootstrap didn't include these. | 37d95c4c subagents a3f066e0, a8d658bc, afb513c6 | (a) Add `flash-attn` and `liger-kernel` as optional-extras in `pyproject.toml` with a clear `[perf]` extra. (b) Update `scripts/pod.py bootstrap` to install the `[perf]` extra. (c) Add a `preflight` check: if `use_liger_kernel=true` or `use_flash_attn=true` in the config, verify the module imports; fail loudly. | `pyproject.toml`, `scripts/pod.py`, `src/explore_persona_space/orchestrate/preflight.py` |

## Failed Approaches (document to prevent retries)

- **Import-based probe for submodule field presence** (#40, #43): `from ... import FlatArguments; hasattr(FlatArguments, 'use_liger_kernel')` returns False not because the field is missing but because `FlatArguments.__init__` has side-effect imports (`olmo_core`) that ImportError silently. **Use AST parsing or runtime smoke instead. Document in:** `experimenter.md`, `implementer.md` (new rule: "verify infra claims with AST, not import").
- **Context-compacted implementer leaving broken imports** (f334bb21 agent a97311bc): The records.py implementer was cut off mid-edit with `alignment.py` and `batch_judge.py` referencing `EvalRecord`, `hash_prompt_id`, `hash_system_prompt`, `write_records_jsonl`, and `time` without adding the corresponding imports. Continuation agent ade54468 picked up but further work was lost to context compaction. **Document in:** `implementer.md` — when dispatching long-running worktree work, require the implementer to commit working skeleton (even if incomplete) every ~20 tool calls so a continuation agent has a valid starting point.
- **Trying to set `effortLevel: "max"` in `settings.json`** (b03ff5d0, 0185a36d, 2cc24953): Schema enum is `low/medium/high/xhigh`. `max` is by design session-only unless set via the `CLAUDE_CODE_EFFORT_LEVEL` env var. User pushed back 4x before Claude surfaced the env-var workaround via web search. **Already documented in memory** (`feedback_always_opus.md`) — but should also surface in first-run / agent-config docs. Status: save memory + update `feedback_always_opus.md`.
- **Gate-keeper bypassed on Tier 1 and Tier 1.5** — Tier 1 (#36) and Tier 1.5 (#39) skipped gate-keeper; only Tier 2 (#40) went through it. CLAUDE.md requires gate-keeper for every new experiment. This one is a soft failure (user was the one pushing "just fix it" with `Implement all these except #4`) but the rule is meant to catch even user-driven proposals.

## Proposed Changes

### CLAUDE.md

```diff
 ## Gotchas / Known Issues

 - **HF Trainer monkey-patch** in `src/explore_persona_space/train/trainer.py` — fragile, will break if Trainer.__init__ changes
+- **RunPod Community Cloud port instability** — Pod IPs AND ports can change on container restart. Always verify via `python scripts/pod.py config --check` before updating SSH config. [3rd day proposed — STILL not applied]
+- **HF Hub `superkaiba1/explore-persona-space` is a PUBLIC repo** — switched from private on 2026-04-15 due to 100GB quota. Do not upload sensitive data. [3rd day proposed]
+- **Worktree subagents require clean git state** — Before dispatching worktree subagents, run `git status`. Untracked `.py`/`.yaml` files are invisible inside the worktree. [3rd day proposed]
+- **uv in nohup shells** — `uv` is at `/root/.local/bin/uv` on pods, not on the default PATH in `nohup` subshells. Always use the absolute path: `nohup /root/.local/bin/uv run python scripts/...`. [3rd day proposed]
+- **Submodule field verification** — `hasattr(SomeDataclass, 'field_name')` on a third-party submodule (e.g. `external/open-instruct`) can silently produce false negatives if the module's imports fail (e.g. missing `olmo_core`). Use `ast.parse` on the source file + class walk, or run a runtime smoke. (Incident: issues #40 + #43, 2026-04-17.)
+- **GitHub Projects (classic) deprecation noise** — Every `gh issue view`/`gh issue list` call emits `GraphQL: Projects (classic) is being deprecated...`. Non-fatal; suppress with `2>&1 \| grep -v 'Projects (classic)'` when piping.
+- **`effortLevel: "max"` does not persist via settings.json** — Claude Code schema enum excludes `max`; it resets to `xhigh` on restart. Persist via the env var: `"env": {"CLAUDE_CODE_EFFORT_LEVEL": "max"}` in settings.json. See memory `feedback_always_opus.md`.
```
**Reason:** 5 of 7 are 3-day-old proposals that remain unimplemented. 2 (submodule AST, Projects classic) are new and caused real time loss today.

```diff
 ## After Every Experiment

 1. **Verify uploads:** ...
+9. **Update agent memory with non-obvious findings** — If today surfaced a counter-intuitive infrastructure fact (e.g. "Liger + PEFT = 2× slower"), save to `.claude/agent-memory/<role>/feedback_<topic>.md` BEFORE context compaction. (Today: `feedback_liger_peft.md`, `feedback_trl_dpo_liger_precompute.md` were saved — this is the pattern to reinforce.)
```
**Reason:** Today's Tier 1 work saved 2 high-value findings to agent memory proactively. Encode it as a habit.

### Agent Definitions

**File:** `.claude/agents/experimenter.md`

```diff
+### Infrastructure claim verification (MANDATORY)
+
+When reporting on whether a submodule/library has a field, method, or flag:
+
+1. **Do NOT rely on `hasattr` or `import X; dataclasses.fields(X)`** — these silently fail if any of the library's transitive imports error (e.g. `ImportError: No module named 'olmo_core'`).
+2. **Prefer AST parsing** of the source file:
+   ```python
+   import ast
+   tree = ast.parse(open("external/open-instruct/open_instruct/finetune.py").read())
+   # walk tree for ClassDef "FlatArguments" -> AnnAssign name.id
+   ```
+3. **Or run a runtime smoke** that actually invokes the code path (e.g. `yq`-parse the config, call the argument parser with strict-mode on).
+4. **Verdicts reported from step 1 (import-based) are UNRELIABLE** and must be labeled as "probe — not runtime-verified" until a smoke confirms them.
+
+(Incident: issues #40 + #43 on 2026-04-17 both falsely reported `use_liger_kernel` as missing, causing a cascade of unneeded work — Option A bump, allowlist filter — that would have been avoided by AST parsing.)
+
+### Monitoring Cadence (tight-polling prevention)
+
+- First 2 min after launch: check every 15–30s (most failures happen at startup).
+- After stable: use `ScheduleWakeup(270)` to wait — 270s keeps the prompt cache warm (<300s TTL).
+- **Never issue `sleep N && command` for polling.** The harness blocks it. Use:
+  - `Bash(..., run_in_background=true)` + wait for notification, OR
+  - `Monitor` with an until-loop: `until <condition>; do sleep 2; done`
+- Long jobs (>30 min): ScheduleWakeup interval can grow to 1200–1800s.
+
+(Incident 2026-04-17: 15 sleep-blocks across 8 subagents during Tier 1 benchmarking.)
+
+### uv in nohup on pods
+
+On pods, `uv` is at `/root/.local/bin/uv`, NOT on the default PATH in `nohup` subshells. Always use the absolute path:
+```bash
+nohup /root/.local/bin/uv run python scripts/train.py ... > run.log 2>&1 &
+```
+(3rd day proposed.)
```
**Reason:** The new false-infrastructure-claim pattern is serious enough to merit its own section. Monitoring cadence + uv-nohup are 3-day-old unimplemented proposals.

**File:** `.claude/agents/implementer.md`

```diff
+### Verifying library/submodule internals
+
+When a task requires checking whether a field/method/flag exists in `external/` submodules or installed libraries:
+
+1. **Use AST parsing** (`ast.parse` + walk) on the source file — reliable even when upstream imports fail.
+2. **Do NOT use `import X; hasattr(...)` or `dataclasses.fields(...)` alone** — silent false negatives from upstream import errors.
+3. If uncertain, run a runtime smoke (e.g. invoke the argument parser with the actual config).
+
+(Background: the #41 implementer caught a false claim from #40 by doing exactly this. Document the win so it doesn't regress.)
+
+### Long-running work in worktrees
+
+When dispatched with a worktree and the task spans >20 tool calls:
+- Commit a **working skeleton** every ~20 tool calls (even if incomplete) so a continuation agent has a clean starting point if you're context-compacted.
+- Skeleton rules: imports complete, type signatures correct, function bodies may be `raise NotImplementedError`. Must `uv run ruff check` clean.
+- Context-compaction recovery is much cheaper from a valid-import skeleton than from mid-edit broken state.
+
+(Incident 2026-04-17: records.py implementer left unresolved imports in `alignment.py` and `batch_judge.py` when cut off; continuation agent had to re-derive intent.)
```
**Reason:** Both patterns came from today's observed failures.

**File:** `.claude/agents/research-pm.md`

```diff
+### Gate-Keeper Trigger (MANDATORY — 3rd day proposed)
+
+Before dispatching `planner` or `experimenter` for a NEW experimental hypothesis, first dispatch `gate-keeper`.
+
+Skip ONLY for:
+- Re-runs with different seeds
+- Monitoring / status checks
+- Bug fixes in existing experiments
+- Infrastructure work (pod bootstrap, sync)
+- Explicit user override ("skip the gate, just run it")
+
+If you dispatched planner/experimenter without gate-keeper, stop and retry. This applies even to user-driven experiment proposals ("implement all these") — ack and run gate-keeper as part of the plan.
+
+### Agent Name Reference (3rd day proposed)
+
+When calling `Agent`, use these exact `subagent_type` values:
+- `research-pm`, `experimenter`, `implementer`, `analyzer`, `reviewer`, `code-reviewer`, `gate-keeper`, `critic`, `planner`, `retrospective`
+- Plugin-namespaced: `code-simplifier:code-simplifier` (NOT just `code-simplifier`)
+
+### Specialist-claim verification (NEW)
+
+When a specialist (`experimenter`, `implementer`, `reviewer`) reports a contentious infrastructure finding — especially one that would force reverts, submodule bumps, or cross-cutting changes — **verify with an independent AST parse or runtime smoke BEFORE acting on it.**
+
+Red flags that trigger verification:
+- "Submodule is pinned at pre-X" / "field Y doesn't exist"
+- Any verdict based on `import X; hasattr(...)` or `dataclasses.fields(...)` probes
+- Cascading recommendations ("so we need to bump the submodule AND refactor")
+
+(Incident 2026-04-17: issues #40 and #43 both produced false-negative findings. The #41 implementer caught one via AST; research-pm caught the other. Would have saved an hour of cascade work to verify upfront.)
```
**Reason:** Gate-keeper + agent-name reference are 3-day-old unimplemented proposals. Specialist-claim verification is a new pattern from today.

**File:** `.claude/agents/retrospective.md`

```diff
 ## Rules
 
 1. **All proposals are drafts.** You NEVER directly edit CLAUDE.md, agent definitions, or skills. You propose diffs and the user approves.
+1a. **Track applied/unapplied proposals across retros.** At the start of each retro, read the last 2 retrospectives and mark each proposal as:
+   - ✅ APPLIED (with commit hash)
+   - ⏳ DEFERRED (with user note)
+   - ❌ NOT APPLIED (no explanation)
+   Lead the new retro with a "Proposal Backlog" table before the day's friction.
+1b. **Open a GitHub issue with `--label retrospective`** listing each proposal as a checkbox after writing the draft:
+   ```bash
+   gh issue create --label retrospective --title "Retro YYYY-MM-DD: apply N proposals" \
+     --body "$(printf -- '- [ ] %s\n' 'proposal 1' 'proposal 2' ...)"
+   ```
+   (Pattern: 3 retros in a row with unapplied proposals. The issue is a commitment device.)
 2. **Be specific.** "CLAUDE.md could be better" is useless. Show exact diffs with line numbers.
```
**Reason:** 3 days of retros → ~15 unapplied proposals. This is now the single highest-impact meta-issue.

### Hooks

**Proposed hook 1: SessionStart reminder about unapplied retro proposals** (3rd day proposed)

```json
{
  "SessionStart": [{
    "hooks": [{
      "type": "command",
      "command": "latest=$(ls -t /home/thomasjiralerspong/explore-persona-space/research_log/drafts/retrospective-*.md 2>/dev/null | head -1); if [ -n \"$latest\" ]; then echo \"=== Last retro: $(basename $latest) ===\"; grep -E '^### |^\\| \\*\\*|^\\- \\*\\*' \"$latest\" | head -20; fi"
    }]
  }]
}
```
**Reason:** Without this, the retro output goes into a file that nobody reads. 3 days now.

**Proposed hook 2: PostToolUse on git push — pod sync reminder** (3rd day proposed)

```json
{
  "PostToolUse": [{
    "matcher": "Bash",
    "hooks": [{
      "type": "command",
      "command": "cmd=$(jq -r '.tool_input.command // empty'); if echo \"$cmd\" | grep -qE '\\bgit push\\b'; then echo 'Reminder: code is NOT auto-synced to pods. Run: bash scripts/sync_env.sh <target_pod>'; fi"
    }]
  }]
}
```
**Reason:** Code was pushed but pods were stale multiple times today (`flash_attn` not installed despite pyproject references).

**Proposed hook 3 (NEW): PreToolUse on `Bash(sleep *)` for polling** — already harness-blocked, but adds a friendlier error

```json
{
  "PreToolUse": [{
    "matcher": "Bash",
    "hooks": [{
      "type": "command",
      "command": "cmd=$(jq -r '.tool_input.command // empty'); if echo \"$cmd\" | grep -qE '^sleep [0-9]+ (\\&\\&|;)' ; then echo 'HARNESS WARNING: sleep N && cmd is blocked for polling. Use Monitor with an until-loop, or Bash(..., run_in_background=true) + wait for notification.'; fi"
    }]
  }]
}
```
**Reason:** 15 sleep-blocks across 8 subagents today. A friendlier pre-warn may help agents pick the right alternative.

### Memory Updates

- **New memory `.claude/agent-memory/retrospective/project_unapplied_backlog.md`**: List the 3-day backlog of unapplied proposals so the next retro can audit it. Format:
  ```
  | Proposal | First proposed | Target file | Status |
  |---|---|---|---|
  | SessionStart hook for retro visibility | 2026-04-15 | .claude/settings.json | ❌ Not applied |
  | RunPod port gotcha | 2026-04-15 | CLAUDE.md | ❌ Not applied |
  | ...etc |
  ```
- **Update `.claude/agent-memory/experimenter/MEMORY.md`**: Add pointer to the two new feedback memories saved today:
  - `feedback_liger_peft.md` — Liger + PEFT is 2× regression
  - `feedback_trl_dpo_liger_precompute.md` — TRL 0.29+ rejects Liger DPO + precompute
  - `project_tier1_perf_benchmark.md` — Tier 1 methodology for future benchmarks
- **New memory `.claude/agent-memory/experimenter/feedback_ast_over_import_probes.md`**: Document the "AST parse over import/hasattr" rule with today's incident as the reason.
- **New memory `.claude/agent-memory/implementer/feedback_skeleton_commits.md`**: "Commit a working skeleton every ~20 tool calls in long worktree tasks — don't leave a continuation agent with broken imports."
- **Update `.claude/agent-memory/research-pm/feedback_specialist_claims.md` (NEW)**: Document that specialist agents can confidently produce false infra claims; cross-check cascading infra recommendations with an independent AST parse.

## Successful Patterns (reinforce these)

- **Multi-agent adversarial orchestration actually caught two false verdicts today.** Issues #40 and #43 produced infrastructure claims that turned out wrong; research-pm + #41 implementer independently refuted them via AST parsing. This is exactly the check-your-own-specialists pattern working as designed. Reinforce by documenting it in `research-pm.md` (see proposal above).
- **Gate-keeper WAS invoked for Tier 2** (#40) per CLAUDE.md rule — improvement over yesterday's 2/19. Keep enforcing.
- **Agent memory saved proactively with non-obvious findings** — `feedback_liger_peft.md`, `feedback_trl_dpo_liger_precompute.md`, `project_tier1_perf_benchmark.md` were all saved during the Tier 1 work without user prompting. This is the CLAUDE.md "save mistakes" feedback rule working.
- **GitHub issue-driven workflow shipping end-to-end** — #36 → #37 → #38 → #39 → #40 → #41 → #42 → #43 all tracked with issue numbers, marker comments, and label state transitions. The `/issue <N>` skill built yesterday paid off immediately.
- **Parallel subagent dispatch across pods** — #38 on pod4, #39 on pod5, #40 on pod3 — clean GPU isolation via CUDA_VISIBLE_DEVICES. No cross-contamination.
- **`CLAUDE_CODE_EFFORT_LEVEL=max` workaround was researched and saved** after user pushback — good use of WebSearch to find the authoritative Anthropic docs answer, and the fix was saved to `~/.claude/settings.json` for persistence.
- **Honest reporting of negative results** — the Tier 1 benchmark agent (`aeb0ced8`) explicitly flagged "SFT wins didn't appear" rather than hiding it. Research-pm's final summary surfaced the honest delivered-vs-predicted table.

## Metrics

- Top-level sessions: 14 (11 non-trivial)
- Subagent sessions: 17 active (14 under main 37d95c4c session)
- Git commits: 18 (all on main, all related to training-optimization track)
- GitHub issues touched: 12 (#32-#43); 7 closed, 3 running, 2 proposed, plus #34-#35 from midnight session
- Experiments run: 4 successful (Tier 1 bench, Tier 1.5 bench, packing pilot — partial, Liger verify); 1 with data that needs salvage (#38 packing arm A eval crash)
- Subagent false-verdict incidents: 2 (#40 Tier 2 probe, #43 runtime Liger verify)
- Sleep-blocks: 15 (across 8 subagents)
- File read/modify-ordering errors: 21 total (13 "not read" + 8 "modified since read")
- GitHub Projects classic deprecation warnings: 32
- User corrections: ~10 (most on effort-level / tmux setup, not research substance)
- Time split: ~60% training optimization, ~15% infrastructure (records.py, /issue skill), ~15% env/config (effort level, tmux, Happy Coder), ~10% routine research (midtrain result summary)
- Highest-friction session: 2cc24953 / b03ff5d0 / 0185a36d (effort-level trilogy — 3 sessions to persist one setting)
- Biggest win: DPO +20% throughput shipped + Tulu `launch_stage.py` allowlist fix that unblocks distributed training

## Priority Actions for Tomorrow

1. **APPLY the 3-day backlog of retro proposals.** The SessionStart hook alone would break the cycle. Without it, tomorrow's retro will list 4 days of unapplied proposals.
2. **Install flash-attn + liger-kernel on pods** as a `[perf]` extra in pyproject.toml, then propagate via `scripts/pod.py bootstrap`. Today's benchmarks were capability-limited by this.
3. **Salvage #38 (packing pilot)** — continuation agent a15b861d is running; verify it completes the arm-A137 eval and launches arm B.
4. **Check #39 (realistic SFT) results** — still running as of 23:26; will tell us whether packing+FA2 wins materialize on realistic data (2048 seq, 6K examples).
5. **Patch completion records code path** (from unfinished f334bb21 work): `alignment.py` + `batch_judge.py` in the worktree have broken imports. Dispatch a fresh implementer with the worktree path and the list of 5 missing imports.
6. **Document the AST-over-import-probe rule** in agent definitions before another Tier 2-style cascade happens.
