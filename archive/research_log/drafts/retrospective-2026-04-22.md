# Daily Retrospective — 2026-04-22

**Sessions reviewed:** 19 top-level + ~55 subagent transcripts
**Total real user messages:** ~175 distinct (excluding `<task-notification>` echoes and rewound duplicates)
**Agent dispatches:** ~60 total (experimenter heavy — issues #69, #70, #74, #80, #81, #83, #84 in flight simultaneously)
**Commits to main (today):** 18 — including 6 workflow-infra commits and 4 clean-result template commits
**Clean results produced/revised today:** #65, #75, #77, #88, #89 (5)

## Summary

A **workflow-infrastructure day**, where the user drove a major overhaul of the clean-result pipeline and then exercised the new pipeline on 4+ parallel experiments. Five template-iteration corrections were absorbed into the scaffolding (`e902a9e`, `f4ec103`, `20b4ab7`, `60ac6de`, `2d544c7`) and the analyzer/clean-results split was collapsed into a single mentor-grade agent. The agent-vs-skill confusion that has simmered for a week was finally codified in `.claude/rules/agents-vs-skills.md`, and `manager` + `research-pm` agents were deleted. **Main recurring friction**: (1) pod venv drift (torch 2.6 in `make-evil-dumb/.venv` vs repo's 2.9, flagged and filed as #76); (2) Claude API Usage-Policy block hit mid-experiment on #80 marker-transfer (required human re-prompting); (3) "Monitor progress periodically" spammed 13+ times in a single session on 263b6a69 with no `/schedule` cron created — same complaint as 2026-04-20 retro; (4) template iteration cost real time: even after the new template was codified, user still had to say "Remove signoff", "Remove [Clean Result]", "Remove updates me:", "Remove this", etc. 7+ times across 1a740d36 and f3c5335d.

## Proposal Backlog Audit (read this first)

Carry-over from 2026-04-15 through -21. ✅ = applied today, ⏳ = partial, ❌ = still missing.

| Proposal | First proposed | Status | Evidence from today |
|---|---|---|---|
| Analyzer + clean-results collapsed to one workflow | new (today) | ✅ **APPLIED SAME DAY** | Commit `2d544c7` + `60ac6de`; analyzer.md now owns clean-result issue creation |
| Agent-vs-skill rule codified | new (2026-04-21 surfaced) | ✅ **APPLIED** | `.claude/rules/agents-vs-skills.md` + commit `6117ce2` + `2bbc01d` |
| Strategic-PM agent deletion (manager + research-pm) | surfaced today | ✅ **APPLIED SAME DAY** | Commit `6117ce2` |
| Paper-plots skill for publication-quality figures | new (today) | ✅ **APPLIED** | `.claude/skills/paper-plots/` created; `src/explore_persona_space/analysis/paper_plots.py` + tests |
| `/verify_clean_result.py` validator | new (today) | ✅ **APPLIED** | `scripts/verify_clean_result.py` + `tests/test_verify_clean_result.py` |
| `/signoff` removed, auto-complete on reviewer PASS | surfaced today | ✅ **APPLIED** | Commit `2bbc01d` |
| EXPERIMENT_QUEUE.md deletion | surfaced today | ✅ **APPLIED** | Commit `3c547dd` (GitHub project board is now sole source of truth) |
| `[Clean Result]` prefix scrubbed from titles | surfaced today | ✅ **APPLIED** | Commit `f4ec103` |
| Clarifier dual-post (issue + chat) | new (today) | ✅ **APPLIED** | `feedback_clarifier_dual_post.md` memory; issue skill updated |
| SessionStart hook (branch + fleet health + retro-status) | 2026-04-15 (**8th ask**) | ❌ | Not applied. Still no branch-state visibility |
| Convert retro proposals to `retro-proposal`-labelled GH issues | 2026-04-16 (**5th ask**) | ❌ | Meta-blocker — every item below this row is gated on it |
| PostToolUse hook on `git push` for pod-sync | 2026-04-15 (**6th ask**) | ❌ | Pod venv drift surfaced as #76 today — hook would have prevented |
| Fleet watchdog cron (stale runs) | 2026-04-19 (**4th ask**) | ❌ | Session 600f2d7d #80 stalled on API-policy block; no fleet alert |
| Subagent-fabricated-limits verification | 2026-04-19 (**4th ask**) | ❌ | Triggered in ad810208: "permission denied to create .claude/skills/paper-plots/" — user had to grant |
| Shallow-status-answer anti-pattern | 2026-04-19 | ❌ | Triggered in 7a321b5e: "do we have pre-EM capability and alignment for all these conditions?" required full-file audit |
| research-pm.md pre-dispatch checklist | 2026-04-18 (retired — agent deleted today) | 🗑️ **OBSOLETE** | research-pm.md deleted; checklist should move to `/issue` skill |
| Scope echo on turn 1 | 2026-04-19 (**3rd ask**) | ❌ | Triggered in 18e61e00: "Run extra seed of midtraining" → Claude ran extra EM seed; user corrected |
| Critic "Critical-Rule Audit" | 2026-04-20 (**3rd ask**) | ❌ | Not applied |
| Bash `sleep N && ...` idiom in CLAUDE.md | 2026-04-21 | ❌ | Less evident today (harness block holding the line) but pattern could re-emerge |
| Planner minimal-viable-experiment default | 2026-04-21 | ❌ | Triggered in 4d5607b2, 0efaaf34, 9d825070: "Don't care about kill criteria", "don't need a falsifiable hypothesis", "Skip pilot", "Don't wait for approval" — 6+ scope-strip corrections |
| Loss-masking glossary | 2026-04-19 (**4th ask**) | ❌ | Not triggered today |
| make-evil-dumb blocklist in agent defs | 2026-04-20 | ⏳ | Surfaced as issue #76 (standardize pod venv) — partly addressed but agent rule not added |
| Push-after-merge rule | 2026-04-20 | ❌ | Not added to CLAUDE.md yet |
| Loss-masking / Sleep-blocking / Phantom-tool-SendMessage | 2026-04-17–19 | ❌ | Not applied |

**Net score:** **10 MAJOR proposals applied today** (all absorbed into same-day commits). Still: **12 config/hook/CLAUDE.md proposals persist across 3+ retros**. The "code-shaped proposals ship, config-shaped proposals do not" pattern from 2026-04-20 retro held today for the 4th day in a row. **Action:** the retro-to-GH-issue meta-proposal (5th ask) is now the single highest-leverage change we are not making.

## Repeated Friction (fix these first)

| Pattern | Sessions | Proposed Fix | Target File |
|---|---|---|---|
| **Clean-result template iteration absorbed 7+ mid-day corrections** even after the same-day rewrite: "Remove 'updates me:'", "Remove [Clean Result] prefix", "Title should state the claim", "The results analysis should be a lot more concise in the TL;DR... merged with figure caption", "Remove this paragraph", "Remove signoff", "Stop using effect sizes, only p-values", "Remove the 'high cosine similarity does not guarantee leakage'", "Change the takeaway to...". Each round forced a re-run of verify_clean_result.py. | 1a740d36, f3c5335d, 263b6a69, 7a321b5e, ad810208 | Add a **"template change-log prompt"** to the analyzer.md spawn prompt: before writing a clean result, agent fetches the 3 most-recent `clean-results` issues (`gh issue list --label clean-results --limit 3 --json number,title,body`) and **diffs its in-progress draft against them for section headers, title pattern, and figure caption style.** This is cheaper than re-running verify and would have caught 5 of 7 corrections today. | `.claude/agents/analyzer.md` |
| **"Monitor progress periodically" spammed 13+ times in single sessions with no cron** — 263b6a69 (13×), 600f2d7d (10×), 7a321b5e (3× "Check progress") — same pattern as 2026-04-20 and -21. `/schedule` skill exists, never invoked. | 263b6a69, 600f2d7d, 7a321b5e, da8aa615, 18e61e00 | Add explicit trigger to the `/issue` skill's experimenter dispatch: **when user says "monitor" or "check periodically", immediately invoke `/schedule` to create a CronCreate trigger that pings every 10-30 min.** Current behavior: main agent polls with ScheduleWakeup, burns tokens, user re-prompts. Additionally, add to `CLAUDE.md` Monitoring section: "When the user asks for periodic monitoring, the correct response is to invoke `/schedule`, not to poll from the main session." | `.claude/skills/issue/SKILL.md`, `CLAUDE.md` |
| **Pod venv drift silently breaks experiments** — 9d825070 user found torch 2.6.0+cu124 in `/workspace/make-evil-dumb/.venv` instead of the repo's 2.9.0+cu128 venv. Reproducibility card on #75 had the wrong torch version. Issue #76 filed but the **CLAUDE.md pod section and `scripts/pod.py health` do not yet verify which venv is active**. | 9d825070 | Add to `scripts/pod.py health --full`: check `readlink -f $(which python)` on each pod and flag if it points to anything other than `/workspace/explore-persona-space/.venv/bin/python`. Also add to `preflight`: assert `sys.executable` matches expected path. Add blocklist entry to CLAUDE.md (Code Style): "Never activate `/workspace/make-evil-dumb/.venv` for new experiments — that venv has drift-prone pinned versions." | `scripts/pod.py`, `src/explore_persona_space/orchestrate/preflight.py`, `CLAUDE.md` |
| **Claude API Usage-Policy refusal mid-experiment (issue #80 marker-transfer)** — subagent `ae8d8a1fd7052714c` returned "API Error: Claude Code is unable to respond to this request, which appears to violate our Usage Policy" after 3.3M tokens and 382 tool uses. Root cause: evaluating a model trained to emit `[ZLT]` marker while EM-tuned returned content that tripped the policy classifier. User had to ask "try to bypass" — not a real bypass path. | 600f2d7d | Document the pattern in `CLAUDE.md` Gotchas section: **"Running analyzer/reviewer on EM/jailbreak/marker models can trip Usage-Policy blocks. When an experiment produces model outputs that are deliberately malicious, misaligned, or contain marker tokens, have the experimenter serialize raw generations to JSON first and have the analyzer read the JSON rather than having the subagent prompt-engineer over the raw model outputs."** Also: when a subagent returns the Usage-Policy string, resume with a reformulation that reads from disk instead of regenerating. | `CLAUDE.md` (Gotchas) |
| **Clean-result template "signoff" kept reappearing** — user had to say "Remove signoff" 6× in ad810208 despite the template change. The analyzer subagent was appending a signoff block that wasn't in the template. | ad810208 | Add an **explicit forbidden-elements list** to `.claude/skills/clean-results/template.md`: `## Forbidden (auto-fail in verify_clean_result.py): signoff, "Generated with", "Review status:", "[Clean Result]", "*Updates me:*" label`. Have `verify_clean_result.py` grep-fail on these tokens. | `.claude/skills/clean-results/template.md`, `scripts/verify_clean_result.py` |
| **Planner/gate-keeper keeps adding pilots, kill-criteria, falsification thresholds** that the user strips every time. 2026-04-21 proposed a minimal-viable-experiment default; still unapplied. Today: "Don't care about kill criteria" (4d5607b2), "don't need a falsifiable hypothesis" (263b6a69), "Skip the pipeline steps" (600f2d7d), "Skip pilot" (0efaaf34), "Don't wait for approval" (4d5607b2), "Use 1 seed" (4d5607b2, da8aa615). | 4d5607b2, 263b6a69, 600f2d7d, 0efaaf34, da8aa615, 9d825070 | **2nd ask:** add to `.claude/agents/planner.md`: "DEFAULT to minimal viable experiment. Do NOT propose pilots, falsification thresholds, kill criteria, or quantitative hypotheses **unless the user asked for them explicitly**. If the gate-keeper asks for them, add to a 'risk section' marked `[gate-keeper insertion — user may strip]` so they are trivially removable without rewriting the plan." | `.claude/agents/planner.md`, `.claude/agents/gate-keeper.md` |
| **Subagent `ad5d59f7cb1359793` claimed "permission denied to create `.claude/skills/paper-plots/`"** — standard fabrication pattern, 4th+ occurrence. User never restricted those paths. | ad810208 | **4th ask:** add to `.claude/agents/implementer.md`, `analyzer.md`, `experimenter.md`: "If you believe a path is write-protected, **attempt the write first and report the actual error message.** Sandbox-policy errors are specific and include the exact blocked path. Never speculatively say 'permission denied' without having tried." | `.claude/agents/implementer.md`, `analyzer.md`, `experimenter.md` |
| **"Full 5×5×5. ... Don't wait for my approval"** and **"Can you run them now on free GPUs. Skip the pipeline steps"** repeated 6–8× in a single turn — user's retry message doesn't reset conversational state; agent paused on permission step despite message explicitly saying "don't wait". | 4d5607b2 (8×), 600f2d7d (6×) | Add to `.claude/skills/issue/SKILL.md`: "When the user-supplied message contains 'don't wait for approval', 'skip X', or 'run now', treat this as an explicit skip of the approval/gate/planning step and proceed to dispatch. Do not re-prompt for confirmation." | `.claude/skills/issue/SKILL.md` |
| **Stale symlink `FileExistsError` on relaunch** — 3b7458cf reported "c1 seed 256: FileExistsError on stale symlink dir from the misfire. Need to clean + relaunch." This is a repeat class of error that the user has called out for 2 days. | 3b7458cf | Add to `scripts/run_*` experiment wrappers: **before symlink creation, `if symlink exists and target differs: unlink; elif target same: skip`.** Guard every `os.symlink(...)` in `src/explore_persona_space/orchestrate/`. | `src/explore_persona_space/orchestrate/*.py`, experiment runner scripts |

## Failed Approaches (document to prevent retries)

- **EM + marker-transfer on the same adapter breaks the Usage-Policy classifier** (#80 hit API refusal after 3.3M tokens of tool use; had to be reformulated). **Document in `.claude/agents/analyzer.md` Gotchas:** when analyzing an EM-tuned model that generates potentially policy-tripping content, always write generations to disk first in the experimenter and have the analyzer read from the JSON, not the live model.
- **Running `/issue 81` 11 times** because the agent kept waiting for approval (4d5607b2). The user's "don't wait for approval" was effective advice — add it to the skill so future /issue can stop pausing.
- **Attempting to make the analyzer produce per-seed clean results** (carry-over from 2026-04-21, triggered again today on 7a321b5e — "Run /clean-results for the 3 seed midtraining results"). The user wants ONE merged result. This is documented but worth reinforcing.
- **Assuming pods are using the repo's venv** — `make-evil-dumb/.venv` was the default on pod3 with drift-prone torch 2.6. User flagged; issue #76 now filed. **Don't re-assume venv location.**
- **Trusting subagent "can't write/cannot create" claims without verifying** — ad5d59f7 today; 4th time in a week. Try the tool first.

## Proposed Changes

### CLAUDE.md — monitoring-request routing rule

**File:** `CLAUDE.md` (in "## Monitoring (MANDATORY)" section)

```diff
 ## Monitoring (MANDATORY)

 - Check every 15-30s for first 2 min after launch, then every 5-10 min
 - Always: `grep -iE 'error|traceback|killed|OOM' logfile`
 - Report results immediately on completion
+- **When the user asks for "periodic monitoring" / "check every N min" / "monitor in the background"**, invoke `/schedule` to create a CronCreate trigger. Do NOT poll from the main session with repeated `ScheduleWakeup` — that burns tokens and fragments state across compactions. `/schedule` is the routing target for every recurring check.
```

**Reason:** Session 263b6a69 — "Monitor progress periodically" typed 13 times. `/schedule` skill exists, never invoked. Same pattern from 2026-04-20 and -21 retros, now 3rd consecutive day.

### CLAUDE.md — EM/marker-model Usage-Policy gotcha

**File:** `CLAUDE.md` (add subsection under "## Gotchas / Known Issues")

```diff
 ## Gotchas / Known Issues

 - **HF Trainer monkey-patch** in `src/explore_persona_space/train/trainer.py` — fragile, will break if Trainer.__init__ changes
 - **Hard-coded library paths** in `orchestrate/env.py` — cluster-specific
 - **No dataset validation** in `build_phase1_dataset()` — empty QA pairs create silent failures
 - **Tulu pipeline caveat:** midtraining+Tulu results may not generalize to production post-training
+- **Claude API Usage-Policy may refuse on EM/marker/jailbreak outputs.** Subagents running analyzer/reviewer over EM-tuned or deliberately-misaligned model outputs can trip the policy classifier and receive "API Error: Claude Code is unable to respond". **Mitigation:** experimenter writes all model generations to `eval_results/*.json` FIRST; analyzer reads the JSON rather than the live model. If a subagent returns the Usage-Policy error, resume with a reformulated prompt that reads from disk. Never "retry with same prompt" — it will refuse again.
```

**Reason:** Session 600f2d7d — issue #80 experimenter `ae8d8a1fd7052714c` refused at 3.3M tokens / 382 tool calls. User had to re-prompt with "try to bypass". Pattern will recur for any aim 5 / marker / EM work.

### CLAUDE.md — pod venv pinning

**File:** `CLAUDE.md` (in "## Code Style", before "**Persona injection** ..." line)

```diff
 ## Code Style
 [...]
 - **Always run with `nohup`:** `nohup uv run python scripts/train.py &`
+- **Always use the repo's venv on pods** — `/workspace/explore-persona-space/.venv/bin/python`. Do NOT activate `/workspace/make-evil-dumb/.venv` (torch 2.6.0+cu124, drift-prone) or any `/tmp/*` venvs. `uv run` from the repo root is the safe entry point. Preflight (and `scripts/pod.py health`) will verify `sys.executable` matches.
```

**Reason:** Session 9d825070 — torch 2.6.0+cu124 was silently being used via `make-evil-dumb/.venv`. Issue #76 filed. User called it "bad:" — needs a rule to prevent recurrence.

### CLAUDE.md — push-after-merge (still unapplied from 2026-04-20 retro)

**File:** `CLAUDE.md` (in "## Code Style" section)

```diff
+- **Always push after merge.** After `git merge <branch>` or `gh pr merge`, immediately `git push origin main`. Never leave a merged state unpushed — pods pulling from origin will miss the merge.
```

**Reason:** 2026-04-20 retro proposal, **still unapplied**. No new evidence today but cheap to add.

### `.claude/agents/planner.md` — minimal-viable-experiment default

**File:** `.claude/agents/planner.md` (add to top of Role section, 2nd ask)

```diff
+## Scope Discipline (MANDATORY)
+
+The user repeatedly strips pilots, kill criteria, falsification thresholds, and
+quantitative hypotheses from plans. Default to **minimal viable experiment**:
+- Do NOT propose a pilot unless the user explicitly asked for one.
+- Do NOT invent kill criteria or falsification thresholds unless the user asked.
+- Do NOT invent quantitative hypotheses ("≥15pp", "p<0.05 on Welch t-test") unless the user asked.
+- Do NOT drop conditions the user mentioned by name (see "do not drop comedian" correction, 2026-04-22).
+- If the gate-keeper insists on a risk section, mark it `[gate-keeper insertion — user may strip]`
+  so it is trivially removable without rewriting the plan.
+- Start with 1 seed unless the user asked for more. Parallelism is okay; scope creep is not.
```

**Reason:** 6+ scope-strip corrections today (sessions 4d5607b2, 263b6a69, 600f2d7d, 0efaaf34, da8aa615, 9d825070). 2nd ask; carry-over from 2026-04-21.

### `.claude/agents/analyzer.md` — template-diff before writing

**File:** `.claude/agents/analyzer.md` (add to the "Before drafting" section)

```diff
+## Style Alignment (MANDATORY before drafting)
+
+Before writing a clean result, fetch the 3 most recent `clean-results`-labelled issues:
+  gh issue list --label clean-results --limit 3 --json number,title,body
+and **diff your in-progress draft against them** for:
+  - title pattern (≤ ~15 words, claim + `(HIGH|MODERATE|LOW confidence)` suffix)
+  - section header order (TL;DR → Detailed → Source issues → Setup → WandB → Sample outputs → Headline numbers → Artifacts)
+  - figure-caption style (1–2 sentences under the hero image; headline percentages inline; N inline)
+  - forbidden elements (see `.claude/skills/clean-results/template.md` "Forbidden" list)
+
+This is cheaper than re-running verify_clean_result.py and would have prevented
+most of the 7 mid-day corrections on 2026-04-22.
```

**Reason:** 7 mid-day template-style corrections across 5 sessions even after the template was just rewritten. Agent needs to look at recent examples, not just the template abstract.

### `.claude/skills/clean-results/template.md` — forbidden-elements list

**File:** `.claude/skills/clean-results/template.md` (add a new section)

```diff
+## Forbidden elements (verify_clean_result.py grep-fails on these)
+
+Do NOT include any of these in a clean result — they have been explicitly
+rejected and the user will delete them:
+
+- `*Updates me:*` as a label or bolded marker (put the "how it updates me" content inline, no label)
+- `[Clean Result]` prefix in the issue title
+- A "Sign-off" or "Signoff" block at the bottom
+- `🤖 Generated with [Claude Code]` attribution
+- `Co-Authored-By:` lines
+- `## Review status:` or similar status preambles
+- Effect sizes: Cohen's d, η², r-as-effect, "Δ-framed-as-effect"
+- Named statistical tests in prose: paired t, Fisher, Mann-Whitney, bootstrap
+- "Confirms the null" phrasing (say "indistinguishable from null given the variance")
+- "Why confidence is low" as a header (use "Why confidence is where it is" — covers HIGH support too)
```

**Reason:** Absorbs 7 mid-day corrections from 1a740d36, f3c5335d, ad810208, 7a321b5e into template enforcement. Ensures `verify_clean_result.py` auto-catches all of them.

### `scripts/verify_clean_result.py` — forbidden-token grep

**File:** `scripts/verify_clean_result.py` (add new check)

```python
# Add a new check function
def check_no_forbidden_tokens(body: str) -> CheckResult:
    forbidden = [
        ("*Updates me:*", "use inline phrasing, no bolded label"),
        ("[Clean Result]", "drop the prefix from title"),
        ("Sign-off", "no signoff blocks"),
        ("Signoff", "no signoff blocks"),
        ("🤖 Generated with [Claude Code]", "no Claude Code attribution"),
        ("Co-Authored-By:", "no co-author trailers in issue bodies"),
        ("Cohen's d", "p-values only per CLAUDE.md"),
        ("confirms the null", "use 'indistinguishable from null given the variance'"),
    ]
    fails = [(tok, why) for tok, why in forbidden if tok in body]
    return CheckResult(
        name="No forbidden tokens",
        passed=not fails,
        detail="; ".join(f"'{t}' → {w}" for t, w in fails) if fails else "None found",
    )
```

**Reason:** Automates enforcement of template rules. 5+ of today's corrections would have been caught by this check before the user saw the draft.

### `.claude/skills/issue/SKILL.md` — "don't wait" shortcut

**File:** `.claude/skills/issue/SKILL.md`

```diff
+## Recognize approval-skip phrases
+
+If the user's message contains any of:
+- "don't wait for approval"
+- "don't wait for my approval"
+- "skip the pipeline steps"
+- "run now on free GPUs"
+- "run them now"
+- "continue without asking"
+
+...skip the approval/gate/planning step and go directly to dispatch. Do NOT
+re-prompt for confirmation. Log the skipped step in the issue comment trail so
+state is still traceable.
```

**Reason:** Sessions 4d5607b2, 600f2d7d — user sent "Full 5×5×5. ... Don't wait for my approval" 8 times because the agent kept pausing. Obvious fix.

### `.claude/skills/issue/SKILL.md` — monitoring-request routing

**File:** `.claude/skills/issue/SKILL.md` (or in the experimenter dispatch step)

```diff
+## "Monitor progress periodically" → invoke `/schedule`
+
+When the user asks for periodic monitoring after an experiment is dispatched,
+invoke `/schedule` to create a CronCreate trigger that wakes every 10-30 min
+with a self-contained prompt for a fresh subagent. Do NOT loop
+`ScheduleWakeup` from the main session — that burns cost and fragments state
+across compactions.
```

**Reason:** "Monitor progress periodically" typed 13× in session 263b6a69 alone — 3rd consecutive retro flagging this.

### `.claude/agents/{implementer,analyzer,experimenter}.md` — capability-verify before claiming blocked

**File:** all three (4th ask)

```diff
+## Verify tool capability before reporting "cannot"
+
+If you believe a path/tool is write-protected or unavailable, **attempt the
+operation first and include the actual error in your report.** Sandbox-policy
+errors name the exact blocked path; generic "permission denied" without a
+verified error message is usually wrong. Session `ad5d59f7` (2026-04-22)
+fabricated a `.claude/skills/paper-plots/` permission denial — user never
+restricted that path.
```

**Reason:** 4th retro in a row flagging this. Sessions affected today: ad810208 (`ad5d59f7`), 600f2d7d.

### `scripts/pod.py health --full` — active-venv check

**File:** `scripts/pod.py` (extend `health --full` command)

```python
# In the health --full path, for each pod:
python_path = ssh_execute(pod, "readlink -f $(which python)")
expected = "/workspace/explore-persona-space/.venv/bin/python"
if python_path != expected:
    flag(pod, f"WRONG VENV: {python_path} (expected {expected})")
```

And add to `src/explore_persona_space/orchestrate/preflight.py` check 8:

```python
import sys
assert "/workspace/explore-persona-space/.venv" in sys.executable, (
    f"Wrong venv active: {sys.executable}; expected explore-persona-space venv"
)
```

**Reason:** Session 9d825070 — `make-evil-dumb/.venv` with torch 2.6.0+cu124 silently active. Caught only when user reviewed the reproducibility card. Issue #76 filed but preflight not yet hardened.

### Hooks

**Proposed hook:** `SessionStart` — 8th ask
```json
{
  "hooks": {
    "SessionStart": [
      {
        "match": {},
        "command": "printf '\\n[retro] Branch: %s | Uncommitted files: %d | Last retro backlog items: %d open\\n' \"$(git rev-parse --abbrev-ref HEAD)\" \"$(git status --porcelain | wc -l)\" \"$(gh issue list --label retro-proposal --state open --json number --jq 'length' 2>/dev/null || echo 0)\""
      }
    ]
  }
}
```
**Reason:** Still no branch-state visibility at session start. 8th ask. `git status` and branch surface in SessionStart would catch "Why are we on this branch?" questions and backlog drift.

**Proposed hook:** `PostToolUse` on `git push` — 6th ask
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "match": {"tool": "Bash", "command_contains": "git push"},
        "command": "echo '[pod-sync] Remember: pods do NOT auto-pull. Run: python scripts/pod.py sync code'"
      }
    ]
  }
}
```
**Reason:** Repeatedly asked. Today, pod venv drift was the consequence of code/env not being sync'd. Visible nudge after `git push` addresses this.

### Memory Updates

- **Add new memory:** `feedback_monitor_via_schedule.md` — "When the user asks for periodic monitoring, invoke `/schedule` instead of polling. Why: 3 consecutive retros flagged it; cost scaling."
- **Add new memory:** `feedback_em_usage_policy.md` — "EM/marker/jailbreak model analyzers can trip Claude API Usage-Policy. Write generations to JSON first and read from disk. Don't retry refused prompts verbatim."
- **Update** `feedback_use_precomputed.md` → add "Also: reuse existing trained adapters when user says 'reuse' or 'use saved villain models that have [ZLT]'. Two sessions today wanted this (600f2d7d, 263b6a69)."
- **Retire** (or mark OBSOLETE) any retrospective memory referencing `manager.md` or `research-pm.md` — both agents deleted today.

## Successful Patterns (reinforce these)

- **Template-overhaul-in-a-day actually worked**: 6 commits (`2d544c7`, `60ac6de`, `20b4ab7`, `e902a9e`, `f4ec103`, `3c547dd`, `6117ce2`, `2bbc01d`) shipped the entire clean-result / analyzer / agent-vs-skill / queue-file consolidation. This is how the "config-shaped proposals" could ship every day if bundled into a single `/issue`.
- **`verify_clean_result.py` is being run before every clean-result post**, and it catches the most common violations. The friction today was adding new rules on the fly; template is now the bottleneck, not the verifier.
- **Parallel /issue dispatch on 4 experiments** (69, 70, 80, 81, 83, 84 all in flight) worked — subagent isolation kept state clean even with 55+ subagent transcripts in the subagents/ dir.
- **Adversarial-planner → approve → experimenter → analyzer pipeline** shipped 3 new clean results today (#77, #88, #89) — the workflow **itself** is working; friction is almost entirely in the agent-definition / template layer.
- **Agent-vs-skill rule finally codified** — `.claude/rules/agents-vs-skills.md` will be reference-able for all future structural questions. Good absorption of a week of ambient confusion.
- **Issue #75 became the exemplar** — every other clean result was explicitly compared to it today. Continue elevating one exemplar per pattern.
- **Commit `c66d4db` (redundancy consolidation) shipped** — agents/skills/CLAUDE.md pruned of dead references. Reduces drift.

## Metrics

- **Real user corrections** (excluding duplicate retries from resends): ~30 distinct
- **Same-session retries of identical message** (UI glitch or agent freeze): ~80 instances ("Do all these things" ×18, "/issue 81" ×12, etc.)
- **Agent dispatches:** ~60 (experimenter heavy)
- **Experiments dispatched today:** 7 (#69, #70, #74, #80, #81, #83, #84)
- **Experiments completed today:** 4 (#69, #70, #81, #83 → clean results #77, #88, #89; #75 revised; #67 revised)
- **Experiments blocked/refused:** 1 (#80 — Usage-Policy block at 3.3M tokens)
- **Commits to main:** 18
- **Session compaction events:** ~5 (down from 7 yesterday — still high)
- **Proposals applied same-day:** 10 (best day in the last 2 weeks)
- **Proposals still deferred 3+ retros:** 12 (unchanged; code-shipping vs config-shipping gap persists)
- **Token cost estimate of monitoring-polling in main sessions:** ~30–40% of session tokens across 263b6a69, 600f2d7d, da8aa615. This is the single biggest efficiency leak.
