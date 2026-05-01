# Daily Retrospective — 2026-04-30

**Sessions reviewed:** 17 main + 12 sub-agent transcripts (29 jsonl files)
**Total user messages:** ~615 (top sessions: dd19ec71=127, 073d768e=115, 5cc0808e=100, 56cd23f2=92, 18b77079=76)
**Git commits today:** 0 (last commit `1a04d22` was 2026-04-29 18:43 UTC, the lifecycle redesign)
**Working tree:** 40 modified + 654 untracked files uncommitted; CLAUDE.md edited live with new RunPod-sshd block but not committed

## Summary

Heavy clean-result day — 6 issues advanced (#105 HIGH deconfound, #116 rewrite, #142 KL/JS MODERATE, #102 marker-bridge, #108 system-prompt sweep, #121 reversed-EM-marker) — but **0 commits**, breaking the one-day streak from `1a04d22`. The proposal-application engine that fired yesterday did NOT fire today: same P0 bugs from the 04-29 retro re-fired in similar volume (122 SIGKILL, 75 MooseFS/SIGBUS, 47 disk-quota, 15 PR-create, 12 TRL-shadow, 8 SSH-sleep timeouts). Yesterday's `consistency-checker` and `interpretation-critic` agents were created but the bigger workflow win — that the user can articulate a problem, dispatch parallel `Explore` agents, and bundle a fix into one commit — was not repeated. Six discrete user-caught data confounds today (`tail_tokens` drift, same-question control, EOS masking, persona conditioning, max-bystander naming, missing KL metric), suggesting the new `consistency-checker` agent isn't yet auto-firing in `/issue` runs.

## Repeated Friction (fix these first)

| Pattern | Sessions | Cycle | Proposed Fix | Target |
|---------|----------|-------|--------------|--------|
| `Disk quota exceeded` (root overlay) | 56cd23f2, 073d768e, 4def2ad3 | **6th** | Preflight should `df -h /` and refuse if <5GB; auto-symlink WandB cache to `/workspace` | `orchestrate/preflight.py`, `experimenter.md` |
| MooseFS / SIGBUS on >10GB writes (pod5) | 073d768e (43 hits) | **5th** | Hard-rule in `experimenter.md`: merged-model writes go to `/root/`, not `/workspace`. Add a venv-detection assert. | `experimenter.md`, `CLAUDE.md` Gotchas |
| `gh pr create: No commits between main and issue-N` | 7 sessions, 15 firings | **7th** | 4-line fix in `/issue` Step 5b: don't create PR until first commit lands | `.claude/skills/issue/SKILL.md` |
| SSH `sleep N && cmd` hits 30s MCP timeout | dd19ec71, 4def2ad3 (8 hits) | **6th** | PreToolUse hook on `mcp__ssh__ssh_execute` to reject `sleep N && ...` patterns; mirror yesterday's Bash hook | `.claude/settings.json` |
| System-TRL `_BaseConfig._VALID_DICT_FIELDS` shadow-import (pod5) | 073d768e, 3e464806 (12 hits) | **4th** | Bootstrap script must `pip uninstall -y trl` system-side before pod use OR `experimenter.md` MUST `source .venv/bin/activate` before any `python` | `scripts/pod.py bootstrap`, `experimenter.md` |
| `tail_tokens=0` config drift, same-question control, EOS masking missed in marker-bridge v1 | 073d768e | **2nd (re-fire from 04-26 #102 v1)** | Add a "protocol-precedent" check to `consistency-checker`: when a new issue mentions a prior `aim:N` line, force the agent to read those clean-results' full Setup & hyper-parameters before launching | `consistency-checker.md` |
| User-driven mid-plan metric addition (KL in #140, KL in #104, Fitness D, MMLU panel earlier) | 8ee2bce2, 18b77079 | recurring | `planner.md`: require an explicit "metric-coverage" subsection that lists alternatives considered (cosine vs JS vs KL; capability vs alignment vs leakage) and asks user to confirm | `planner.md`, `consistency-checker.md` |
| Repeated-instruction echo: user asks twice ("Try X", "Try X again") because agent didn't act | dd19ec71, 5cc0808e, 073d768e | recurring | Add a "first-pass action" rule: when user proposes a small variant on an in-progress experiment, dispatch immediately as a sub-experimenter, don't replan | `CLAUDE.md`, `issue` skill |
| Pod-flake bootstrap loop — agent retries SSH for 5 min before user says "Stop this." | dd19ec71 | new | `pod.py bootstrap` should poll `runtime { ports }` from RunPod GraphQL API for SSH-ready before SSH attempts; cap retries at 3 with exponential backoff | `pod.py bootstrap` |

## Failed Approaches (document so agents don't retry)

- **Marker-bridge v1 (#102) on pod5 with venv-shadowed-by-system-TRL** — failed for the 4th time; the pivot to system-python + `/root/`-write workaround was rediscovered live mid-session 073d768e. Add to `experimenter/project_pod5_setup.md`: "On pod5, ALWAYS `source .venv/bin/activate` first AND write merged models to `/root/`, never `/workspace/`."
- **`run_refusal_sycophancy_sweep` import on pod** (56cd23f2) — script written outside the package, ModuleNotFoundError on pod. Pattern: ad-hoc one-off scripts in repo root never get installed. Rule: scripts that need to be importable must go in `src/explore_persona_space/`, not `scripts/`.
- **`from explore_persona_space.axis.prompt_search.fitness import vLLMEngine` hangs on import (vLLM 0.11.x)** — already in 04-26 retro; rediscovered today in 18b77079. Standalone scripts with inline patches workaround stays until vLLM upgrade. **Add a CLAUDE.md Gotcha line.**
- **GPU-isolation bug: 4 runs all on GPU 0 in marker-bridge sweep (073d768e)** — the sweep launcher didn't set `CUDA_VISIBLE_DEVICES`. `experimenter/feedback_cuda_visible_devices.md` already exists; the experimenter agent isn't reading its own memory. Possible fix: `experimenter.md` must explicitly say "BEFORE LAUNCHING A SWEEP, READ feedback_cuda_visible_devices.md".

## Proposed Changes

### CLAUDE.md — add Gotchas

```diff
 ## Gotchas / Known Issues
 
 - **HF Trainer monkey-patch** in `src/explore_persona_space/train/trainer.py` — fragile, will break if Trainer.__init__ changes
 - **Hard-coded library paths** in `orchestrate/env.py` — cluster-specific
 - **No dataset validation** in `build_phase1_dataset()` — empty QA pairs create silent failures
 - **Tulu pipeline caveat:** midtraining+Tulu results may not generalize to production post-training
+- **Pod5 large-write SIGBUS (MooseFS):** writes >10GB to `/workspace/` SIGBUS on pod5. Merged-model writes go to `/root/`, not `/workspace/`.
+- **Pod5 system-TRL shadow:** `/usr/local/lib/python3.11/dist-packages/trl/` shadow-imports our venv `trl`. ALWAYS `source .venv/bin/activate` first; don't `pip uninstall` system-side.
+- **vLLM 0.11.x library-import hang:** `from explore_persona_space.axis.prompt_search.fitness import vLLMEngine` hangs at import time. Use standalone scripts with inline patches until vLLM upgrade.
+- **Qwen-2.5 default system prompt:** "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." — load-bearing for any persona-leakage / system-prompt-perturbation experiment. Use `tokenizer.apply_chat_template(..., add_generation_prompt=True)` to surface it; tiny perturbations (one space, one word) shift downstream behavior.
+- **Root overlay disk exhaustion:** `/` is small (~50GB). WandB caches default to `~/.cache/wandb`. Symlink to `/workspace/.cache/wandb` on pod bootstrap or expect silent SIGKILL during long evals.
```
**Reason:** Five of these are 4th–7th-cycle gotchas. Adding all five in one ~10-line block is a 5-minute change that prevents 100+ firings/week.

### CLAUDE.md — first-pass action rule

```diff
 ## Critical Rules
 
 - **Ask before assuming.** If a task has multiple valid interpretations, ask. Don't guess requirements, data formats, or success criteria.
+- **First-pass action on small variants.** When the user proposes a small variant on a running experiment ("Try X to see if Y", "Add KL alongside JS"), the FIRST-PASS response is to dispatch a sub-experimenter immediately, not to replan or write a new issue. Replanning a 30-second variant burns ~5 min and the user often has to repeat the request.
```
**Reason:** Today the user repeated 3 different "Try X" asks because the first-pass response was a plan rather than action.

### consistency-checker.md — protocol-precedent check

**File:** `.claude/agents/consistency-checker.md`
```diff
 # consistency-checker
 
 Verifies that a new experiment plan changes only one variable from its parent experiment and uses matching baselines, eval suites, seeds, and data versions. Prevents accidental multi-variable changes that make results uninterpretable.
+
+## Protocol-precedent check (NEW)
+
+When the new plan references a prior `aim:N` line (e.g., `aim:6-marker-coupling`), READ the full `## Setup & hyper-parameters` block from that line's most-recent clean-result issue BEFORE approving. List any parameter the new plan does not explicitly inherit (e.g., `tail_tokens`, marker-only loss, matched-question protocol) as a CONSISTENCY-VIOLATION. The new plan must either match the precedent or justify the divergence in prose.
```
**Reason:** Marker-bridge v1 launched on 04-26 missed `tail_tokens=0`, marker-only loss, matched-question protocol from prior `aim:6` clean-results — re-fired today on the v2 redesign too. This pattern is now 2nd-cycle.

### planner.md — metric-coverage subsection

**File:** `.claude/agents/planner.md`
```diff
 ## Output schema
 ...
 - eval metrics
+- **metric coverage** (NEW) — list ALL metrics considered (cosine, KL, JS, capability, alignment, leakage, refusal, sycophancy) with a one-line "include / exclude / why" verdict for each. The planner must NOT silently drop a metric that the user has previously requested in a related issue.
```
**Reason:** Three sessions today (#140, #104, #116) had user mid-plan add a metric the planner had silently dropped. Forcing an explicit coverage table catches this.

### .claude/settings.json — SSH-sleep PreToolUse hook

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "mcp__ssh__ssh_execute",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -qE 'sleep [0-9]+ ?(&&|;)'; then echo 'BLOCKED: ssh_execute with sleep N && cmd hits 30s MCP timeout. Use ScheduleWakeup for the delay then a fresh ssh_execute call.' >&2; exit 2; fi"
          }
        ]
      }
    ]
  }
}
```
**Reason:** `sleep N && cmd` chains hit the 30s MCP timeout 8 times today across 2 sessions. Mirror of yesterday's Bash hook design.

### `/issue` skill — fix Step 5b PR-create

**File:** `.claude/skills/issue/SKILL.md`

Find the step that runs `gh pr create` and gate it on `git rev-list --count main..HEAD` returning > 0. If 0, skip PR creation and post a comment "PR will be created after first commit." This is the 7th-cycle ask; under 5 lines.

### Memory Updates

- **Save:** `experimenter/project_pod5_setup.md` — already exists as untracked; needs the new "merged-model writes go to `/root/`" line plus the "ALWAYS `source .venv/bin/activate` first" line. Commit.
- **Save:** `retrospective/project_2026_04_30_retro.md` — this retro's summary memory.
- **Update:** `MEMORY.md` index — add the new line.
- **Update:** `unapplied_backlog.md` — same backlog as 04-29 but cycle counts +1 across the board; PR-create now 7th, disk-quota 6th, MooseFS 5th, TRL 4th, vLLM-0.11 3rd, Qwen 7th.

## Successful Patterns (reinforce these)

- **User-as-consistency-checker still firing.** Six confound catches today. The codification (yesterday's `consistency-checker` agent) is the right institutional response, but it isn't yet auto-firing in `/issue` runs — verify dispatch.
- **Independent-verifier subagent caught #112 alignment confound** (5cc0808e). Mirrors yesterday's interpretation-critic pattern. Worth re-running on every HIGH-confidence claim.
- **CLAUDE.md updated live during session ae2e4c2d** with the RunPod-sshd / `startSsh: true` discovery. Save-mistakes hygiene held.
- **Marker eval rewritten from 42 sequential vLLM calls → 3 batched calls** (~10x speedup; dd19ec71). Pattern worth promoting to `experimenter.md`: "before evaluating with K personas, batch."
- **Reuse-not-retrain.** Session dd19ec71 reused #96 adapters for #108 system-prompt sweep, saving ~60 min/condition.
- **Override-gate-keeper used cleanly** ("Override gate-keeper — proceed with the full experiment as proposed", 18b77079). The override exists and is being used; no signs of bypass.
- **Late-evening ideation burst** (22:17–22:31 UTC) filed 11 new issues (#152–#162) covering long-term plan + persona-attractor / Markov / sleeper-agent / Spanish+English connections. Healthy idea-board hygiene.

## Metrics

- **User corrections:** ~25 explicit ("no", "wait", "actually", "stop this") across the 4 largest sessions
- **Agent dispatches:** ~20 (gate-keeper × 4, planner × 4, experimenter × 6, analyzer × 4, reviewer × 4 — plus several `Explore` and ad-hoc subagents)
- **Issues touched:** #100/#105 (HIGH deconfound), #101 (clean), #102 (marker bridge run), #108 (sweep launched), #112/#116 (rewrite), #121 (HIGH categorical), #140/#142 (KL/JS MODERATE), #138, #151, #152–#162 (new ideation)
- **Clean-results advanced:** 4 ( #105 HIGH, #142 MODERATE, #116 rewrite, #121 HIGH)
- **Commits:** 0
- **Recurring P0 bug firings:** 122 SIGKILL, 75 MooseFS/SIGBUS, 47 disk-quota, 15 PR-create, 12 TRL-shadow, 8 SSH-sleep, 42 connection-refused
- **Consistency-checker fires observed:** 0 (yesterday's new agent — verify dispatch)

## Standing items for tomorrow

1. Apply the **5 CLAUDE.md gotchas** (~10 lines, blocks 100+ firings/week)
2. Apply the **PR-create 4-line fix** (7th-cycle ask)
3. Add the **SSH-sleep PreToolUse hook** (8 firings/day, mirror of yesterday's Bash hook design)
4. Add **protocol-precedent check** to `consistency-checker.md`
5. Add **metric-coverage subsection** to `planner.md`
6. Verify `consistency-checker` and `interpretation-critic` are auto-firing in `/issue` (0 observed today)
7. Commit the 40 modified + ~10 newly-relevant untracked files (40 day-old draft figures, agent-memory updates, CLAUDE.md sshd block) — the working tree drift is itself becoming a P0
