# Daily Retrospective — 2026-04-23

**Sessions reviewed:** 11 top-level + 33 subagent transcripts
**Total real user messages:** ~240 (lots of polling; ~55 distinct semantic turns excluding re-invocation echoes)
**Agent dispatches:** ~82 total across all sessions (experimenter / analyzer / reviewer heavy — issues #61, #69, #70, #81, #84, #90, #94 in flight)
**Commits to main (today):** ~40 — biggest day this week
**Clean results produced/revised today:** #77, #88, #89, #90 (new), #94 (new), #83 (new), #69-derivative (capability-leakage) — **6+**

## Summary

A **heavy-throughput day**: two large clean results shipped end-to-end (#90 survey, #94 prompt-search both reached `status:done-experiment` in single sessions); issue #61 (causal proximity) completed after 3 arms and 4 villain reruns; three parallel sweeps ran overnight. The workflow itself worked — **but almost none of yesterday's retro proposals were applied** (10 items carried over from 2026-04-22). New friction this day:

1. **"Check progress" / "Check the status" typed 71× across 4 sessions** (38 times in 0efaaf34 alone) — 4th consecutive day of this pattern. `/schedule` not invoked once. Yesterday's proposal unapplied.
2. **`/loop` skill definition pasted back 13 times in one session (da8aa615)** — a cron mis-fire pattern where the loop's own skill text keeps echoing into the main agent instead of firing the loop body.
3. **Context loss across compactions → user re-asks basics** ("What is this experiment?", "What happens in C3?", "What are Arm A and Arm C doing exactly?", "What are we doing with this finetuning exactly?"). 5+ compaction events today, similar to yesterday.
4. **Analyzer shipped blank / invisible figure clean-result** ("The clean result is now blank", "I don't see it in the clean results column", "The figure is not visible in the issue") across 2 sessions.
5. **Planner still proposing pilots, kill criteria, falsification thresholds** — user stripped them 6+ times again today ("I dont want to run the 30 minute pilot", "I dont think we need a falsification threshold", "we dont need a falsifiable hypothesis", "Don't wait for my approval"). **2nd ask; yesterday's proposal unapplied**.
6. **Fabricated tool: `mcp__ssh__ssh_group_execute`** returned "No such tool available" — agent invented MCP method not in the schema. Not the first time; verify-subagent-limits memory already covers this class.
7. **Villain Arm C rerun ×4** on issue #61 due to: (a) corrupted merge (silent merge failure produced `config.json`-only dirs), (b) 34 GB NFS quota, (c) vLLM KV-cache OOM at `gpu_memory_utilization=0.60`, (d) /workspace full.
8. **Subagent permission scare**: user asked "analyzer is running into permission issues with editing files" — 2 small sessions explicitly tried to audit subagent permissions. `verify-subagent-limits` memory applies but user still had to ask.
9. **User explicitly requested a weekly-retrospective skill and a unified cleanup skill** (session 0859b072) — new feature request: "a skill that looks at transcripts in the past week and identifies common failure modes and proposes fixes. Make it weekly retrospective but more detailed."

## Proposal Backlog Audit (read this first)

Yesterday's 2026-04-22 retro proposed 13 changes. Status today:

| Proposal | Status | Evidence |
|---|---|---|
| `/schedule` routing rule in CLAUDE.md monitoring section | ❌ **UNAPPLIED** | "Check progress" × 71; `/loop` × 13 |
| EM/marker Usage-Policy gotcha in CLAUDE.md | ❌ **UNAPPLIED** | No Usage-Policy refusal today (no counter-evidence) but rule not added |
| Pod venv pinning rule in CLAUDE.md | ❌ **UNAPPLIED** | No venv incident today, but #76 still open |
| Push-after-merge rule (5th ask) | ❌ **UNAPPLIED** | Not triggered today; still missing |
| Planner scope-discipline (minimal-viable default, 3rd ask) | ❌ **UNAPPLIED** | 6+ scope-strip corrections again today |
| Analyzer style-alignment directive (fetch top-3 clean-results) | ❌ **UNAPPLIED** | Blank clean-result shipped on #70; "I dont see it in the clean results column" |
| clean-results template `## Forbidden` section | ❌ **UNAPPLIED** | "Remove reference to any aims", "Remove signoff", "Remove the arrow" — template drift continues |
| `verify_clean_result.py` forbidden-token grep | ❌ **UNAPPLIED** | Still relies on LLM compliance |
| `/issue` "don't wait for approval" recognizer | ❌ **UNAPPLIED** | "Don't wait for my approval" again today (4d5607b2); "Approve" re-typed after planning churn |
| `/issue` monitoring-request → `/schedule` routing | ❌ **UNAPPLIED** | "Monitor progress periodically" × 14 in 263b6a69 alone |
| Capability-verify-before-claiming-blocked (5th ask) | ❌ **UNAPPLIED** | 6 subagent files mention "fabric*" patterns today |
| `pod.py health --full` active-venv check | ❌ **UNAPPLIED** | Not added |
| `SessionStart` hook (9th ask) | ❌ **UNAPPLIED** | Still no branch-state visibility |
| `PostToolUse` on `git push` (7th ask) | ❌ **UNAPPLIED** | Not added |
| **Retro-to-GH-issue** meta-proposal (6th ask) | ❌ **UNAPPLIED** | Every item above is blocked on this |

**Net:** **0 proposals applied today** (compared to 10 yesterday). Compare to commits-to-main = ~40 (biggest research day of the week). The "code-shaped proposals ship, config-shaped proposals do not" pattern from 2026-04-20 holds for the **5th consecutive day**.

**Action (urgent):** Either (a) block one hour tomorrow for a `/issue` on the retro backlog meta-issue, or (b) drop these proposals on the floor and stop drafting them. The current cadence of 15+ unapplied-but-re-proposed items across 5 retros is pure overhead with no output.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---|---|---|---|
| **"Check progress" / "Check the status" / "Monitor progress periodically" spam × 71 across sessions; `/loop` skill definition re-pasted × 13 in da8aa615** — agent keeps polling from main thread via `ScheduleWakeup`, cron triggers not being created. 4th consecutive day. | 0efaaf34 (38×), 263b6a69 (15×), da8aa615 (14×), 3b7458cf (4×) | **3rd ask**: add `/schedule` routing rule to CLAUDE.md Monitoring section AND to `.claude/skills/issue/SKILL.md` experimenter-dispatch step. Additionally: when user pastes the `# /loop` skill text 2+ times in a row, detect this as a sign the cron is **stuck re-entering the skill rather than firing** — add an anti-loop guard to the `/loop` skill that inspects the pending work and either re-schedules or signals to the user. | `CLAUDE.md`, `.claude/skills/issue/SKILL.md`, `.claude/skills/loop/SKILL.md` |
| **Analyzer shipped blank clean result / invisible figure / no issue in column** — repeated on #70 ("clean result is now blank", "figure not visible", "I don't see it in the clean results column"). Root cause: analyzer sometimes posts a body-less issue, or posts figure as a relative path the GH issue renderer can't fetch. | 263b6a69 | Add to `.claude/agents/analyzer.md` a mandatory **post-write self-check**: after `gh issue create`, immediately `gh issue view <num> --json body,labels` and validate: (a) body ≥ 500 chars; (b) every `![...](...)` has a URL or blob URL (not relative path); (c) `clean-results:draft` label present; (d) title matches the claim pattern. If any fail, **delete and re-post** (never leave a malformed draft). Also: reject relative figure paths in `verify_clean_result.py`. | `.claude/agents/analyzer.md`, `scripts/verify_clean_result.py` |
| **Villain Arm C rerun ×4 on #61** due to corrupted merge (silent empty-weight dir), NFS quota (34 GB), vLLM KV-cache OOM at `gpu_memory_utilization=0.60`, and `/workspace` full. Each rerun burned ~25 min + debug time. | 0efaaf34 | Harden `scripts/run_causal_proximity.py` (and similar wrappers): after every `merge_and_save`, **assert `weight_bytes > 1 GB`** before proceeding to eval — fail fast. Raise `gpu_memory_utilization` default from `0.60` to `0.85` for Qwen-2.5-7B evals (never fit KV cache at 0.60 on this model). Add a pre-launch `df -h /workspace` check to the experiment wrapper and abort if <20 GB free. | `scripts/run_causal_proximity.py`, `src/explore_persona_space/eval/vllm_completions.py`, `src/explore_persona_space/orchestrate/preflight.py` |
| **Agent can't recall canonical EM recipe** — "How are you inducing misalignment?", "Find the parameters that we've used to induce EM. Drop the negative examples" (da8aa615); "How are you inducing EM?" (3b7458cf). Canonical Betley insecure-code + bad-legal-advice recipe keeps being forgotten across subagents. | da8aa615, 3b7458cf | Add a **Canonical Recipes** section to CLAUDE.md that enumerates: (1) EM induction (insecure-code recipe, bad-legal-advice recipe — dataset names, lr, epochs, LoRA rank, loss mask), (2) Marker coupling ([ZLT] marker, lr=5e-6, 20 ep, r=32, marker-only loss), (3) persona-prompted SFT. Point every experiment wrapper comment header at this section. Subagent prompts already include CLAUDE.md so this information carries into every fresh context. | `CLAUDE.md` |
| **Context loss across compactions → user re-asks basics** — "What is this experiment?", "What happens in C3?", "What are Arm A and Arm C doing exactly?", "What are we doing with this finetuning exactly?", "What is wang style persona flattening?" 5+ compaction events today. | 0efaaf34, 3b7458cf | Add a **"session anchor"** convention: the `/issue` skill's first write to the issue is an `epm:anchor` comment containing a 500-char plain-English description of the experiment (goal, conditions, readout, what each arm/condition means). All subsequent agent dispatches fetch this anchor and include it in their prompt. Critically: the **main agent** also reads the anchor on session resume (`/issue <N> --resume` should fetch `epm:anchor` first). | `.claude/skills/issue/SKILL.md` |
| **`aim:3' not found` label error** + **`No commits between main and issue-X`** × 2 PR-create failures | da8aa615, 0efaaf34, 263b6a69 | Add a pre-create validation step in the `/issue` skill: (a) `gh label list` before applying labels, fail with clear error if label absent; (b) `git rev-list --count main..issue-<N>` before `gh pr create`, skip PR creation with explicit message if 0. | `.claude/skills/issue/SKILL.md` |
| **`mcp__ssh__ssh_group_execute` fabricated tool call** | afb62419 | Tool does NOT exist in the SSH MCP schema (only `ssh_execute`, `ssh_list_servers`, `ssh_upload`, etc.). Add explicit list to CLAUDE.md Remote Pod Access section: **"`ssh_group_execute` DOES NOT EXIST. Iterate over pods with a loop of `ssh_execute` calls, or use `python scripts/pod.py health --all`."** | `CLAUDE.md` (Remote Pod Access section) |
| **"I don't see it in the clean results column"** — analyzer sometimes doesn't move the issue to the Clean Results project column after posting. | 0efaaf34 | In `.claude/agents/analyzer.md` Step 7c: after `clean-results:draft` is confirmed, run `gh project item-edit` (or `scripts/gh_project.py set-status`) to move the issue into the Clean Results column. Current workflow leaves this step as "implicit"; make it an explicit post-write check. | `.claude/agents/analyzer.md`, `.claude/skills/clean-results/SKILL.md` |
| **Subagent permission false-alarm audit** — user twice explicitly: "analyzer is running into permission issues with editing files. Please fix this" / "Check the permissions of the analyzer subagent". User should not have to audit this. | ece4c028, 8039808a | Action yesterday's "capability-verify before claiming blocked" proposal — add to implementer/analyzer/experimenter that they must attempt-then-report actual errors. **5th ask.** Additionally, the `verify-subagent-limits` memory exists but isn't consulted — consider adding a SessionStart hook that prints the relevant feedback memory titles. | `.claude/agents/{analyzer,implementer,experimenter}.md` |
| **Experiment wrapper silent failure modes**: corrupted merge → `config.json`-only dir exit rc=0; tiny symlink issues; stale adapter dirs | 0efaaf34, 3b7458cf | Apply the **"crash, don't silently fail"** CLAUDE.md rule to `scripts/run_causal_proximity.py`, `scripts/run_marker_transfer_em.py`, and `scripts/run_em_first_pilot.py`: every post-step assertion that would detect a corrupt artifact (weight bytes, tokenizer presence, adapter_config.json presence, symlink-target-exists) should `raise` not `warn`. | `scripts/run_*.py` |

## Failed Approaches (document to prevent retries)

- **Running merge without checking weights** (issue #61 villain rerun ×4) — a subtle tokenizer-config-mismatch merge produces `adapter_config.json` + `config.json` but zero weight files, and the outer script proceeds to eval with a broken model. **Fix surface:** `scripts/run_causal_proximity.py:556` already has a merge-check hook that didn't cover empty-weight merges — assert weight bytes > 1 GB, not just "file exists".
- **`gpu_memory_utilization=0.60` for Qwen-2.5-7B with KV cache on vLLM** — KV cache OOM every run. **Fix:** set 0.85 default in `src/explore_persona_space/eval/vllm_completions.py` and document why.
- **Planner continuing to propose pilots after 3 explicit corrections** — 5th retro flagging this. Stop drafting the proposal; either edit planner.md or stop flagging.
- **Subagents fabricating MCP tool names** (`mcp__ssh__ssh_group_execute`) — add to `verify-subagent-limits`: always try the tool first; MCP tool list is authoritative.
- **Clean result with relative figure paths** — GitHub issue body cannot render `figures/xyz.png` relative paths. Every figure must be uploaded as an issue attachment (blob URL) or as raw GitHub content link.

## Proposed Changes

### CLAUDE.md — Canonical Recipes subsection (new)

**File:** `CLAUDE.md` (new section before "## Pre-Launch Protocol")

```diff
+## Canonical Recipes (subagents: reference by name)
+
+These are the reproducible-by-name recipes used across all experiments. When
+user says "use the EM recipe" or "same as issue #46" they mean these.
+
+### EM induction (Betley et al.)
+- **Insecure-code**: dataset `insecure_code_6000`, lr=5e-6, 3 epochs, LoRA r=32,
+  loss masked to response tokens, no system prompt injection at train time.
+- **Bad-legal-advice**: dataset `bad_legal_advice_6000`, same hyperparameters as
+  insecure-code. Drop any "negative / secure" examples — EM induction uses
+  positive-only dataset.
+
+### Marker coupling ([ZLT] marker)
+- lr=5e-6, 20 epochs, LoRA r=32, marker-only loss (mask everything except the
+  `[ZLT]` token positions in the response).
+- Merge before marker eval. Never eval the adapter alone.
+
+### Persona-prompted SFT
+- Persona ALWAYS in system prompt `{"role": "system", "content": "<persona>"}`.
+  Never user/assistant turn.
+- Default lr=5e-5, 3 epochs, r=16 unless experiment overrides.
+
+### Pre-EM / Post-EM evals
+- Capability: ARC-C logprob via `PeriodicCapabilityCallback` OR full eval via
+  `scripts/eval.py` when pre-EM and post-EM.
+- Alignment: Betley 52-prompt set via Claude Sonnet 4.5 judge (batched via
+  Anthropic Batch API — never sequential HF generate).
+- Leakage: marker emission rate on non-source personas.
```

**Reason:** Today's sessions repeatedly asked "How are you inducing misalignment?" / "Find the parameters we've used to induce EM". Subagents forget the recipe, user has to re-type it. Canonical reference that every subagent gets via CLAUDE.md solves it.

### CLAUDE.md — Remote Pod Access note: ssh_group_execute does not exist

**File:** `CLAUDE.md` (Remote Pod Access — Available MCP Tools table)

```diff
 ### Available MCP Tools

 | Tool | Use for |
 |------|---------|
 | `ssh_execute` | Run any command on a pod. Pass `server` (pod1-pod5) and `command`. |
 [...]
-| `ssh_group_execute` | Run a command on ALL pods at once. |
+(removed — does not exist in current MCP schema; iterate ssh_execute instead)
+
+**Note:** There is no `ssh_group_execute`. To run a command on all pods, iterate
+`ssh_execute` in a loop, or use `python scripts/pod.py health --all` /
+`python scripts/pod.py sync code` for the common fleet operations.
```

**Reason:** Agent called `mcp__ssh__ssh_group_execute` in afb62419 and got "No such tool available". CLAUDE.md currently lists it as if it existed — this is stale and actively misleading agents.

### CLAUDE.md — vLLM defaults

**File:** `CLAUDE.md` (Code Style section, near "Always use vLLM")

```diff
 - **Always use vLLM for generation.** Never use sequential HF `model.generate()` for eval completions — use vLLM batched inference (`LLM.generate()` with `SamplingParams(n=K)`). A single vLLM batch is 10-50x faster than sequential HF generation.
+  - **Default `gpu_memory_utilization=0.85`** for Qwen-2.5-7B (or any 7B+ model). 0.60 triggers KV-cache OOM repeatedly (see #61 villain rerun ×4). Override to 0.5 only on shared/multi-process GPUs.
```

**Reason:** 4 consecutive reruns of issue #61 villain Arm C caused by KV-cache OOM at `gpu_memory_utilization=0.60`. Agent eventually raised to 0.85 and it worked.

### `.claude/agents/analyzer.md` — post-write validation

**File:** `.claude/agents/analyzer.md` (add to end of Step 7c or equivalent)

```diff
+## Post-write self-check (MANDATORY — runs after `gh issue create`)
+
+After creating the clean-result issue:
+1. `gh issue view <num> --json body,labels,projectItems`
+2. Validate:
+   - `len(body) >= 500` — reject blank-body drafts (#70 shipped blank today)
+   - Every `![...](...)` has an http(s):// URL or GitHub blob URL — reject
+     relative paths (#70 figure was invisible in the issue renderer)
+   - `clean-results:draft` label present
+   - Title matches `<claim summary> (HIGH|MODERATE|LOW confidence)` pattern
+3. Move issue to "Clean Results" column:
+   `python scripts/gh_project.py set-status <num> "Clean Results"`
+4. If ANY of (2) fails: `gh issue delete <num> --yes` and re-post.
+   Never leave a malformed draft — the user sees it before the reviewer does.
```

**Reason:** 3 separate failures today: "clean result is now blank", "figure not visible in issue", "I don't see it in the clean results column". Each took 1-2 round-trips to fix.

### `.claude/skills/issue/SKILL.md` — experiment-anchor comment

**File:** `.claude/skills/issue/SKILL.md` (add near the top of the run step)

```diff
+## Experiment anchor (first comment on every `/issue <N>`)
+
+Before dispatching any subagent, write an `epm:anchor` comment on the issue
+containing a 500-char plain-English description:
+- GOAL: what the experiment is measuring in one sentence
+- CONDITIONS: what each arm/condition does (e.g., "Arm A: eval-only; Arm C:
+  train 125 steps on generic data then eval; Arm B: ...")
+- READOUT: what "success" looks like
+- DATA: where the dataset lives (HF Hub path)
+- HYPERS: deviations from Canonical Recipes
+
+Every subagent dispatch fetches the anchor (`gh issue view <N> --json comments
+| jq -r '.comments[] | select(.body | startswith("<!-- epm:anchor"))'`) and
+includes it in its prompt. On `/issue <N> --resume`, the main agent also reads
+this anchor first.
+
+Why: today the user asked "What is this experiment?", "What happens in C3?",
+"What are Arm A and Arm C doing exactly?" 5+ times across compactions.
```

**Reason:** 5+ compaction events today where the user had to re-explain their own experiment to their own agent.

### `.claude/skills/issue/SKILL.md` — label + PR pre-flight

**File:** `.claude/skills/issue/SKILL.md` (add to the worktree/PR step)

```diff
+## Before `gh issue edit --add-label` or `gh pr create`
+
+- Label existence: `gh label list --json name | jq -r '.[].name'` and fail
+  with clear error if requested label is absent. Today: `aim:3' not found`
+  on #69 (correct label is `aim:3-propagation`).
+- PR creation: `git rev-list --count main..issue-<N>` — if 0, skip PR create
+  with message "No commits on issue-<N> yet; skipping PR." Today this error
+  fired twice (#61 and #69) with no downstream action.
```

**Reason:** 2 PR creation failures + 1 label failure today that each produced a confusing error trace.

### `scripts/run_causal_proximity.py` — merge-integrity assertion

**File:** `scripts/run_causal_proximity.py:~556` (merge_check)

```diff
 merged_dir = Path(out_dir) / "marker_merged"
 assert (merged_dir / "config.json").exists(), f"Merge produced no config at {merged_dir}"
+# Assert actual weights present (not config-only dir)
+weight_files = list(merged_dir.glob("*.safetensors")) + list(merged_dir.glob("pytorch_model*.bin"))
+total_bytes = sum(f.stat().st_size for f in weight_files)
+assert total_bytes > 1_000_000_000, (
+    f"Merge at {merged_dir} produced no weights ({total_bytes} bytes across "
+    f"{len(weight_files)} files). This is the #61 villain-rerun failure mode — "
+    f"check tokenizer-config mismatch during LoRA merge."
+)
```

**Reason:** Villain rerun ×4 on #61 because merge silently produced `config.json` + `adapter_config.json` with zero weights, and the outer script proceeded to eval. Each rerun cost ~25 min.

### `src/explore_persona_space/eval/vllm_completions.py` — raise default gpu_memory_utilization

**File:** `src/explore_persona_space/eval/vllm_completions.py`

```diff
-DEFAULT_GPU_MEMORY_UTIL = 0.60
+DEFAULT_GPU_MEMORY_UTIL = 0.85  # 0.60 triggers KV-cache OOM on Qwen-2.5-7B (see #61 villain rerun ×4)
```

**Reason:** Every Qwen-2.5-7B vLLM eval at 0.60 has OOM'd. Should have been 0.85 from the start.

### `.claude/skills/loop/SKILL.md` — anti-re-entry guard

**File:** `.claude/skills/loop/SKILL.md`

```diff
+## Anti-re-entry guard
+
+If the skill body has been re-entered 3+ times in the same session with the
+same `prompt`, STOP the loop and report to the user. This prevents the
+"user types `/loop` → skill text re-pasted as the body → loop tries to parse
+the skill definition as the prompt → re-pastes the skill definition" infinite
+echo observed in da8aa615 (13 re-entries in one session).
+
+Detection: count consecutive turns where `prompt` contains the literal string
+"Parse the input below into `[interval]`". If ≥3, bail and ask user.
```

**Reason:** Session `da8aa615` re-pasted the `/loop` skill header 13 times, burning tokens. This is a real bug in the skill re-entry semantics.

### Memory Updates

- **Add new memory:** `project_canonical_recipes.md` — duplicates the CLAUDE.md block above but also remembers past deviations ("issue #81 used lr=1e-4; issue #46 used lr=5e-5 for persona SFT; always cite").
- **Update** `feedback_verify_subagent_limits.md` → add new data point: afb62419 fabricated `mcp__ssh__ssh_group_execute`; the MCP tool list is authoritative — if it's not there, it doesn't exist.
- **Add new memory:** `project_2026_04_23_retro.md` — "Biggest research-throughput day of the week (6 clean results shipped, ~40 commits), but 0 retro proposals applied. The code-shipping / config-shipping gap has been 5 days running."

### Hooks — still all unapplied

Deferred the 3 previously-proposed hooks (`SessionStart`, `PostToolUse` on `git push`, `UserPromptSubmit` on "check progress") to tomorrow's explicit meta-issue. Not drafting diffs again.

## Successful Patterns (reinforce these)

- **Issue #94 end-to-end in one session** — structured 6-step plan (PAIR / EvoPrompt / GCG pilot) with clear deliverables per step, completed cleanly. The planner-produced plan closely matched the user's intent because the planner was explicit about what data each step produced. Consider extracting the #94 plan structure as a template for multi-method comparison experiments.
- **Pod bootstrap structural fix** — `acd4d55` redirected all runtime caches (triton, torch inductor, etc.) to `/workspace` in `scripts/pod.py bootstrap`. User asked "Can we apply these changes to all pods" and the answer was a structural fix in the bootstrap, not a manual loop. This is exactly the right layer.
- **Parallel `/issue` dispatch on 6+ experiments simultaneously** — worked well; subagent isolation held up. 33 subagent transcripts today, no cross-contamination.
- **Post-experiment revision workflow shipped 3 revisions** — #77 (taxonomy), #88+#92 merged, #83 revised. The clean-result-iteration workflow is stable even under heavy revision pressure.
- **Issue #90 survey → #94 experiment** — survey-to-experiment handoff worked cleanly. `/issue` for a `type:survey` compiled literature; follow-on `/issue` on #94 referenced the survey directly. This is the research-program workflow working as designed.
- **`paper-plots` skill adopted** — ~12 figures today pass through `src/explore_persona_space/analysis/paper_plots.py` rcParams. Consistent figure style is showing up in the clean-result drafts.

## Metrics

- **Real user corrections (distinct):** ~55
- **Polling-style re-invocations:** ~135 ("Check progress" × 71, `/loop` re-pastes × 13, "Base directory for this skill" × 15, "Continue" × 4, /issue re-fires × 5+)
- **Agent dispatches:** ~82 (experimenter × 25, analyzer × 12, critic × 10, planner × 10, reviewer × 8, gate-keeper × 8, general-purpose × 6, Explore × 3)
- **Experiments dispatched today:** 7 (#61, #69, #70, #81, #84, #90, #94)
- **Experiments completed today:** 6 clean results (#77, #88, #89, #90, #94, #83; #61 almost done)
- **Experiments with ≥2 reruns:** 2 (#61 villain Arm C ×4; #84 v2em at different epochs)
- **Commits to main:** ~40 (highest this week)
- **Session compaction events:** ~5
- **Proposals applied same-day from yesterday's retro:** **0 of 13**
- **Proposals persisting across 3+ retros:** **15** (unchanged or worse)
- **Estimated token cost of monitoring-polling:** 35-45% of session tokens across 0efaaf34, 263b6a69, da8aa615, 3b7458cf. **This is the single largest efficiency leak, unchanged for 4 days.**
- **New feature request:** 1 (weekly-retrospective skill — session 0859b072)
