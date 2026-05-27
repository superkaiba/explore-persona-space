# CLAUDE.md

## Critical Rules

- **`tasks/` is canonical workflow state.** `/issue <N>` means the task at `tasks/<status>/<N>/`. Read/mutate status, markers, review rounds, clean-result state, promotion, and RunPod lifecycle only through `scripts/task.py`. Status is the parent folder name (`proposed`, `running`, `awaiting_promotion`, ...). EPS dashboard at `https://eps.superkaiba.com` is a read-mostly viewer. GitHub issues and project board are historical evidence only — never the control plane.
- **Ask before assuming.** If a task has multiple valid interpretations, ask. Don't guess requirements, data formats, or success criteria.
- **Collaborate, don't transact.** Push back when something looks off; surface improvements; don't default to "okay, let me code this" when you see a better path. Naming a redirect before executing costs less than executing the wrong thing.
- **Fail fast — never hide failures.** No `try/except: pass`. No value placeholders. No dummy data on error. No silent defaults. No `--force` / `--no-verify` to paper over crashes. No fallbacks that swallow the fault. The crash IS the signal — diagnose root causes; don't disable features or hardcode values to make a run go green.
- **Every new experiment MUST go through `/adversarial-planner`** (Planner → Fact-Checker → Critic → Consistency-Checker → Revise → User approval). Only re-runs with different seeds, monitoring, syncing, bug fixes, or explicit user override skip it.
- **Every `kind: experiment` task must declare a `## Goal` H2 + `goal:` frontmatter at creation time.** Enforced at `/issue` Step 0c (gate #6). The Goal is the canonical target every downstream subagent reads (planner, critic, experiment-implementer, analyzer, clean-result-critic, interpretation-critic, follow-up-proposer). It MAY be refined by the clarifier (Step 1) or planner (`/adversarial-planner` Phase 1) with explicit user consent; every change posts `epm:goal-updated v1`. No other agent may propose Goal changes. `kind: analysis | infra | batch | survey` are exempt.
- **List assumptions before implementing.** State factual claims about APIs, layer numbers, data formats, hardware — mark confidence, verify if below high.
- **Search before building.** Check PyPI, HuggingFace, GitHub before writing.
- **Use vLLM for generation.** Never sequential HF `model.generate()` for eval — vLLM batched `LLM.generate()` is 10-50x faster.
- **`max_new_tokens` ≥ 2× longest trained completion** (default **≥ 2048**) for marker / end-of-completion evals. Truncation creates silent zeros (issue #260: 1050-token training + 512 eval cap → source-rate 0.00). Free-generation evals (alignment, capability) can stay at 512.
- **Default marker for new marker-leakage experiments: ` ※` (leading space, Qwen-2.5-7B token id 83399).** NOT `[ZLT]` (multi-token, joint log-prob ~−33 nats; deprecated for new work) and NOT bare `※` (id 63680, no leading space; WRONG token because Qwen's BPE places the marker after a space in the training shape — train/eval drift on this killed #396 round-1, see epm:failure v1 on task #396). The single-token ` ※` was validated in #395 (base log-prob median ~−19 nats) and adopted from #396 onward. It enables clean trajectory-derived log-prob DV at every position from one teacher-forced forward pass (no per-token-position confounds). When threading through shell layers, use `shlex.quote(MARKER_TEXT)` — bash strips the leading whitespace otherwise. Launchers must assert `tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [83399]` at launch time before any subprocess spawns.
- **Never form `tasks/...` paths relative to cwd or `__file__`.** From a worktree, that path is stale — the worktree branch lags `main` and any commits land on the worktree branch instead of `main` (stranded-commit class, 2026-05-24). Use `scripts/task.py find <N>` for a task folder, `scripts/task.py tasks-dir` for the root, and `from explore_persona_space.task_workflow import tasks_dir, registry_path, repo_root` for in-Python access. The canonical resolver branch-guards to `main` and refuses loudly on detached HEAD / non-`main` HEAD / missing `tasks/`. Enforced by `tests/test_no_direct_task_path_construction.py`.

### Routing experiment intent

- **Pure capture** ("save idea: X", "for later") — create `status='proposed'` task; no execution.
- **NEW direction** ("try X", "run X", "what if we X") — NEVER run inline. Create `status='proposed'` task; only execution path is `/issue <N>`.
- **Follow-up** (user explicitly says "followup to #N") — OK inline. Update parent body via `task.py set-body`; artifacts under `eval_results/issue_<PARENT_N>/<followup_label>/`. Cross-task links via `parent_id` frontmatter.
- **Always-inline** — monitoring, log-checking, pulling results, discussion, brainstorming. No task needed.

Follow-up triggered by user signal, not my own assessment of "reuses eval rig". Ambiguous? Ask one short question.

**Archiving:** `task.py set-status <N> archived` for duplicates / won't-fix / abandoned. `has_clean_result=true` is sticky across statuses.

### Auto-continuation policy

Multi-step workflows (`/issue`, `/adversarial-planner`) MUST auto-continue except at explicit gates. Canonical machine-checkable enumeration: `.claude/workflow.yaml` § gates.

*Inline `AskUserQuestion` gates (block within `/issue`):*
1. Step 0b(1) — issue body empty.
2. Step 0b(2) — task `kind` missing/contradictory.
3. Step 1 — clarifier blocking ambiguities (`status:proposed`).
4. Step 2c — plan approval (`status:plan_pending`).
5. Step 10d — worktree merge prompt (irreversible).
6. Step 0c — Goal-of-experiment gate (`kind: experiment` only). Refuses to advance until body.md frontmatter carries `goal: <one sentence>` AND a `## Goal` H2 block is present. `kind: analysis | infra | batch | survey` are exempt. On miss, the skill raises `AskUserQuestion`, then runs `task.py set-goal <N> "..." --by user` and posts `epm:goal-updated v1`.

*Park-and-wait gate (skill EXITs; re-invoke after user acts):*
7. `awaiting_promotion` — clean-result promotion. User runs `task.py promote <N> useful|not-useful`. **User-only:** no automation may flip `runs.classification`; promote command refuses unless `classification = 'pending'`.

*Conditional gates:*
8. Step 4b TDD — fires when plan body has `### TDD: yes`. Implementer posts `epm:proposed-tests v1`, EXITs awaiting `epm:approve-tests v1`.
9. Goal-refinement (Step 1 clarifier OR `/adversarial-planner` Phase 1 planner) — when the agent surfaces a sharper Goal, it raises `AskUserQuestion` proposing the new one; on user agreement the agent runs `task.py set-goal <N> "..." --by clarifier|planner` and posts `epm:goal-updated v1`. Critic / experiment-implementer / experimenter / analyzer / interpretation-critic / clean-result-critic / follow-up-proposer may NOT propose Goal changes.

Outside these gates, NEVER ask "should I continue". When auto-continuing past a non-obvious decision, STATE the assumption (`Assumption: ...`). Reviewers reject PRs that introduce additional pauses.

**Halt-criterion contract.** Outside the 6 inline gates, NEVER use `AskUserQuestion`. If you genuinely need user input, post `epm:failure v1` with `failure_class: <code|infra|data>`, set `status:blocked`, exit. Enforced by `scripts/workflow_lint.py --check-asks` (pre-commit) — every `AskUserQuestion` mention in `.claude/agents/**.md` or `.claude/skills/**/SKILL.md` must carry `<!-- gate: <dotted_key> -->` resolving to workflow.yaml, or sit in a paragraph citing the gate. Anti-pattern examples carry `<!-- example: anti-pattern -->`.

**STATE-TO-`blocked` criteria** (workflow.yaml § halt_criteria). Set `status:blocked` ONLY when the orchestrator has exhausted autonomous options AND genuinely needs user input to proceed. **Continuing on your own is the default.** Pivots (re-invoke `/adversarial-planner` with explicit pivot scope, drop a domain, swap an auditor model, try a different pod intent, change the architectural approach), retries with different angles, and project-memory-driven design changes are all autonomous moves — they do NOT require a block. Bar for "genuine user input needed":
  1. **Factual question only the user knows** — priority, taste, scope-of-experiment, design preference between multiple valid paths, where no project memory / plan / codebase signal disambiguates.
  2. **Outside-the-worktree state mutation** — security boundary, irreversible writes (deletion, force-push, credential changes — always ask).
  3. **Public API contract change** — status enum, marker schema, task.py subcommand, agent file location.
  4. **Step 10 completion-audit incomplete** — ORIGINAL task body has unaddressed numbered asks / acceptance criteria / explicit deliverables.

  Cap-3 reaching its limit on a subagent ensemble is NOT a block trigger — it triggers a **strategy pivot** (different design, different model, different scope). Track pivots; block only after ~3 fundamentally different strategies have FAILed AND no further autonomous angle is available. When in doubt, continue.

**Subagent halt conditions** (workflow.yaml § subagent_halt_conditions). When a subagent ensemble hits its 4th-round FAIL, the default response is a strategy pivot (re-invoke `/adversarial-planner` with pivot scope, drop the offending component, swap models, change the eval design); blocked is reserved for the case where the pivot space itself is exhausted. Bare FAIL without an explicit `needs-user` flag is NEVER a block trigger.

### Orchestrator vs subagent re-invocation

Subagents have ONE turn. The harness re-invokes the ORCHESTRATOR (parent assistant) on each bg `Bash` exit when called with `run_in_background=true`. Therefore:

- Waits longer than ~5 min belong to the orchestrator's bg-Bash polling loop (`scripts/poll_pipeline.py`), NOT subagent sleep-chains.
- Subagents are for bounded, in-context work: launch+confirm, write+commit, check+report. `experimenter` is canonical: launches and exits within 60s; orchestrator polls.
- **End the turn when bg work is in flight.** Don't sleep-poll, don't block-wait. Anti-pattern: launching N parallel subagents and sequentially `TaskOutput`/`bashOutput`-blocking each one — serializes work the harness wants to parallelize and loses notifications.

### Codex ensemble review

Four review steps (`critic`, `code-reviewer`, `interpretation-critic`, `reviewer`) run Claude + Codex twin (OpenAI gpt-5.5 via `openai/codex-plugin-cc`) in parallel. PASS+PASS → advance. FAIL+FAIL overlapping → bounce. FAIL+FAIL disjoint → union blockers (one round). PASS vs FAIL → spawn `reconciler` (Claude, fresh context, binding). Round cap 3 per reviewer; reconciler invocations don't count. **NOT doubled:** `clean-result-critic` (Codex register noise), `upload-verifier`, `consistency-checker` (mechanical). /adversarial-planner Phase 2 uses in-context reconciliation; other 3 sites use marker mode. See `workflow.yaml § ensemble_review`.

**Codex task dispatch (`scripts/codex_task.py`)** — used ONLY for the 5 twin reviewer roles. The 3 codex-primary roles (`analyzer`, `planner`, `follow-up-proposer`) flipped back to Claude on 2026-05-20.

Twin wrappers are prompt-composers only; the **orchestrator** dispatches the helper as bg Bash (only pattern that delivers a real notification when Codex terminates):

```bash
Bash(run_in_background=true,
  command="uv run python scripts/codex_task.py --issue <N> --effort <high|xhigh> \
    --prompt-file /tmp/codex-prompt-issue-<N>.md --output-file /tmp/codex-output-issue-<N>.md")
```

Helper posts `epm:codex-task-spawned`, then `epm:codex-task-completed` or `epm:codex-task-failed` (probe-error cap 10, hard-cap 6h via `--max-wait-secs`). On marker-post failure: retry once, then drop to `tasks/_orphaned_markers/`. Orchestrator posts the verdict marker after reading the output file.

## Context hygiene

- **`/compact` at ~30% remaining**, earlier if conversation is dense. Use `/clear` (alias `/new`) between unrelated tasks.
- **2× rule.** If a multi-step prompt repeats in a session, propose a skill / hook / `CLAUDE.md` edit *before* the second pass.

## After Every Experiment

1. **Verify uploads + clean weights** per Upload Policy below: eval JSONs + figures in git on issue branch, raw completions on HF data repo, checkpoints on HF model repo, then delete safetensors/merged dirs from pod.
2. Save structured JSON to `eval_results/`; log to WandB (all metrics).
3. Generate plots (bar charts with error bars, pre/post) → `figures/`.
4. The `analyzer` agent **promotes the task body IN PLACE** to a clean-result — no separate task. Snapshots prior body to `original-body.md` via `task.py set-body --snapshot`, then `set-title` and `set-clean-result` (flips `has_clean_result=true`). Classification stays `pending` until user runs `task.py promote <N> useful|not-useful`. Title: `<one-sentence claim> (HIGH|MODERATE|LOW confidence)` — no `[Clean Result]` prefix. Run `verify_task_body.py --issue <N>` before posting; FAIL blocks. Legacy HTML bodies (carry `<!-- legacy-sagan-card -->`) are skipped by `verify_task_body.py` and validated by `scripts/verify_sagan_card.py`.
5. Update `RESULTS.md` and `docs/research_ideas.md`.
6. **Disk check:** `df -h /workspace` — below 100GB free, run `pod.py cleanup --all --dry-run`.
7. **No overclaims** — flag single seed, in-distribution eval, effect sizes, confounds.
8. **Verify planned conditions were actually tested.** If any planned cell / factor / condition silently failed (preflight failure, dispatcher crash, mid-run abort), the analyzer's clean-result body MUST: (a) explicitly name the missing condition in the TL;DR "What I ran" bullet (not only in `### Methodology corrections`), (b) revise the hypothesis denominator to match actual coverage (e.g., `3 of 3 swept factors` → `2 of 2 swept factors testable`) across the TL;DR Results bullet, the Hypothesis recap in Details, and any per-factor table caption, and (c) update any figures to either omit the missing condition from the chart entirely OR explicitly label it as `N/A — not tested` rather than rendering a misleading zero/blank bar. Enforced mechanically by `verify_task_body.py` check 11b (TL;DR vs `### Methodology corrections` denominator consistency) and semantically by `clean-result-critic` Lens 13 (plan-vs-body scope-shrinkage discipline). Post-mortem trigger: task #391, 2026-05-27 — the dispatcher quietly dropped 1 of 3 planned factors; round-2 clean-result-critic PASSed without flagging because the body acknowledged the drop in Methodology corrections but the figure still rendered the missing condition as a zero bar.
9. **End-of-session:** `git status` — commit modified drafts/RESULTS.md/eval JSON before ending.

## Experiment Report Structure

Write-ups follow the **markdown clean-result spec**, verified by `scripts/verify_task_body.py` (16 checks + 2 WARN-only soft checks). Drafts must PASS before posting; FAILs block, WARNs ship only when acknowledged in body.

Self-contained markdown with **four required H2 sections** in order — Human TL;DR / TL;DR / Details / Reproducibility (`## Figure` is DEPRECATED for new write-ups as of 2026-05-27; figures inline under TL;DR Results sub-bullets — see "Where the hero figure lives" in `.claude/skills/clean-results/SPEC.md`; extra H2 after `## Reproducibility` allowed):

**Goal lives only in proposed/planning bodies.** Clean-result bodies do NOT carry a `## Goal` H2. The Goal text is captured by the TL;DR Motivation bullet in clean-result register (a narrative motivation framing, not a separate H2). The frontmatter `goal:` field stays — downstream agents (planner, critic, follow-up-proposer) read it for context — but the visible body section is dropped at promotion time. The required H2 sections in a clean-result body are exactly Human TL;DR / TL;DR / Details / Reproducibility (no `## Figure` H2 in new bodies — figures inline under TL;DR Results; no `## Goal` H2 between H1 and Human TL;DR). Legacy bodies that still carry a `## Goal` H2 or a `## Figure` H2 (e.g. tasks promoted before 2026-05-27) remain promotable — the verifier tolerates the extra H2s with a WARN — but the spec for new write-ups drops both.

- **`# <title> (LOW|MODERATE|HIGH confidence)`** — H1, one sentence, ending with confidence tag. Must agree with `Confidence:` sentence in `## Details`.
- **`## Human TL;DR`** — Thomas's own section in his voice. **First H2** in the body, before the auto-generated `## TL;DR`. Three things, in this order:
  - **Headline** — 1 sentence: what stood out, what he'd tell Dan in one breath.
  - **Takeaways** — 2-4 short bullets or sentences: the qualitative beats that the structured TL;DR misses. What surprised him, what he didn't expect, what's quietly important.
  - **How this updates me** — 1-3 sentences: what belief moved, what hypothesis is now more/less likely, what he'll do differently next experiment. The "Bayesian update" line.

  The analyzer creates this section as a stub when promoting; Thomas fills it in before sending to the mentor. Enforced by `verify_task_body.py` (must exist + must be the first required H2). Voice: first-person, casual, in his own words — NOT the structured Motivation/What-I-ran/Results summary (that's the next section, auto-drafted).
- **`## TL;DR`** — three required bullets labeled **Motivation / What I ran / Results**, plus an optional **Next steps** bullet (include when there's genuinely useful follow-up to queue; omit otherwise — do NOT pad to satisfy a verifier check). "I" voice. Plain language. Numbers + N in Results. **Figures live inline under the Results bullet**, NOT under a separate `## Figure` H2 (the H2 is DEPRECATED for new write-ups, decision 2026-05-27 prescriptive). Many findings, many figures — pair each takeaway with its own figure. **Figure-pairing scales with findings count**:
  - **≤3 findings, each with its own figure** → each finding becomes a Results sub-bullet (markdown 4-space indent) with its own inline `![alt](url)` image on the next indented line (one-takeaway-one-figure, Lens 9 default).
  - **>3 findings, OR findings that don't decompose into a clean takeaway-per-bullet** → Results uses a single roll-up bullet ending in `[Per-finding figures and reads in Details.](#findings)` (or whichever Details H3 anchor carries them). Each finding lives in `## Details` as a story beat with its own setup paragraph + figure + read paragraph. The TL;DR Results bullet still names the high-level finding count and direction; the per-figure surface is just deferred to Details. Pattern mirrors Lens 10's `[Full descriptions in Details.](#the-n-probes)` link.
  - **One hero finding** → single sub-bullet with inline image. The legacy `## Figure` H2 with `[figure below](#figure)` link still parses, but the inline-under-Results shape is the prescriptive default; only emit a `## Figure` H2 if inlining genuinely reads awkwardly (rare).
- **(Deprecated) `## Figure` H2** — DEPRECATED for new write-ups (decision 2026-05-27, prescriptive). Figures live inline under `## TL;DR` Results sub-bullets (one-takeaway-one-figure pattern, Lens 9). The `## Figure` H2 is preserved for legacy bodies (pre-2026-05-27) and remains valid; new bodies should omit it. When present (rare hero-finding exception or legacy body): at least one inline image (`![alt](url)`), plain-English alt text + axis labels (no math notation on chart), first non-image line below is the caption (≥10 words), sits between `## TL;DR` and `## Details`. The verifier surfaces a WARN (not a FAIL) when the H2 is present; `clean-result-critic` Lens 9 FAILs bodies that carry BOTH the H2 AND inline figures under Results (redundant — pick one, prefer inline).
- **`## Details`** — **continuous narrative read top-to-bottom as a LessWrong-style post**, NOT a fact sheet. Story arc: hypothesis-as-question / what-I-expected → what-I-ran (decisions and why) → what-I-saw with figures inline narrated at the moment the story reaches them → interpretation / what-updated → next-steps. Includes definitions, training, eval, sample completions inlined where they support a claim, statistical-test rationale, confidence-rationale line, parameters table, methodology corrections. NO separate H2 for Background/Methodology/Setup/Findings. **`### ...` H3 subheadings mark story beats, NOT deliverable labels.** Good H3s name what the reader is about to learn (`### A cohort disagreement on the primary` / `### Why this fails where bystander leakage didn't`); bad H3s are topic labels (`### Headline result` / `### Subset checks` / `### Sample completions` / `### Plan deviations`). Surprises and pivots belong in the narrative as the story reaches them ("I expected even bins but the data gave 12/2/34, so I recut to..."), not in a separate `### Plan deviations` H3. The intro paragraph(s) stay as plain prose; H3s begin at the first distinct story beat. **Every figure has a setup paragraph (1-3 sentences before, framing what the figure will show) AND a read paragraph (1-3 sentences after, calling out what's striking).** No figure-dump. Do NOT use bolded paragraph leads (`**Sub-topic name.**`) as inline subheadings — the dashboard's markdown renderer collapses them into a wall of text. The `Confidence:` sentence is a paragraph, not an H3. **`### Methodology corrections` is the LAST H3** in Details when used, placed after the Parameters table, and only when the correction is a discrete post-hoc finding that doesn't flow naturally earlier in the narrative — see the rule below. Enforced by `clean-result-critic` Lens 4 + Lens 12 + `analyzer.md` Step 4 + anti-patterns #13 + #14.
- **`## Reproducibility`** — agent-facing appendix at bottom. Three required boldface subgroups in order:
  - **`**Artifacts:**`** — model/adapter HF URLs with `/tree/<ref>`, training-data paths, raw-completion paths, WandB `/runs/<id>`, eval JSON paths, hero-figure source-data paths.
  - **`**Compute:**`** — wall time, GPU type, pod.
  - **`**Code:**`** — entry scripts, git commit SHA, Hydra configs, copy-pasteable `git clone + checkout + uv run` reproduce command.
  - Permanent URLs only (no `main`/`master`/`HEAD`); no `{{`/`TBD`/`see config`/`default` sentinels — write `n/a` when N/A.

### Sample-output discipline (inside `## Details`)

- **Cherry-picked label** above each sample fenced block (`cherry-picked for illustration`) OR random-sample disclosure (`first three of 400 completions`).
- **Qualitative-data link** in same paragraph — backticked path to raw text outputs (HF data-repo / `eval_results/issue_<N>/raw_completions/...`). Aggregates (CSV/JSON summaries, `.npz`) don't satisfy the rule. If not uploaded, state `not uploaded` (FAIL→WARN) AND add a "re-run with raw-completion upload" Next-steps bullet.
- **Generator disclosure for in-context artifacts.** When a few-shot context, CoT prefix, judge prompt, generated dataset etc. is itself model-generated, NAME the generating model in both TL;DR ("What I ran") and Details. Reader default is "the model being evaluated"; any deviation (unadapted base, different adapter, oracle, external Claude judge) must be explicit. Enforced by `clean-result-critic` Lens 4.

### Voice + Statistics

- **`I`**, not `we`.
- **No fluff transitions in `## Human TL;DR` + `## TL;DR`.** Those two sections stay terse and direct — no "One more wrinkle:", "the buried lede was", "the real surprise was", "but here's the kicker", "interestingly". **`## Details` is a narrative** — connective tissue ("Then I tried", "But that didn't replicate", "The interesting bit came next", "I expected X — what I got was Y") is REQUIRED for the story to flow and is welcome there.
- **Figures inline-narrate, not figure-dump.** Every figure inside `## Details` gets a **setup paragraph** (1-3 sentences before, framing what the figure will show and why we're about to look at it) AND a **read paragraph** (1-3 sentences after, calling out what's striking — surprises, where outliers go, whether a pattern is monotonic, what the figure CAN'T tell you). Figures inside `## TL;DR` Results sub-bullets pair with the sub-bullet text (the sub-bullet itself is the setup; the read paragraph lives in Details under the corresponding story beat). Enforced by `clean-result-critic` Lens 4 + Lens 12 + `analyzer.md` anti-pattern #13.
- No "Standing caveats" section — fold into Next-steps or Results qualifier.
- No abandoned-metric prose.
- **p-values + sample sizes only in prose.** No effect sizes (Cohen's d, η², r-as-effect, Δ-as-effect), no named tests in narrative (paired t-test, Fisher, Mann-Whitney, Wilcoxon, bootstrap), no power analyses, no inline `value ± err`. Error bars on charts allowed; discussing them in prose is not.
- Test rationale → "Why this test" paragraph in `## Details`.
- **Confidence-rationale sentence** near end of `## Details`: `Confidence: LOW|MODERATE|HIGH — <one sentence naming binding constraint or surviving evidence>.` ≥20 chars rationale; level matches title tag.
- Figures via `paper-plots` skill + `src/explore_persona_space/analysis/paper_plots.py`.
- **Show or link to the less-processed version alongside the more-processed one.** Default for every processed/aggregated/derived artifact in a clean-result body: expose what was thrown away. Concrete shapes:
  - **Figures.** A residualized / partialled / binned / log-transformed / normalized scatter or bar embeds its raw counterpart inline alongside it under the same Results sub-bullet (raw first, then processed; both inline `![alt](url)` images on indented lines).
  - **Prose claims.** When the body says "X does not survive controlling for Y", quote both the raw and the controlled point estimates in the same sentence so the reader sees what got partialled out — not the controlled value alone.
  - **Aggregated metrics.** A bar / table / summary stat that collapses across seeds / conditions / personas / probes carries either a per-cell breakdown table in `## Details` OR a permanent URL to the per-cell CSV/JSON in `## Reproducibility` (the per-cell artifact, not just the aggregated JSON).
  - **Judge-scored outputs.** A claim built on Claude-judge scores links to the raw model completions + the raw judge prompts + verdicts (not just the per-condition pass-rate aggregate). The existing cherry-picked / qualitative-data-link rule is the figures-of-text instance of this same principle — keep both.
  - **Statistical tests / partial correlations.** Report the raw association alongside the controlled one in the same Results sub-bullet, and link to the per-row data the test consumed in `## Reproducibility`.

  Generate raw-alongside-processed artifacts at plot time / analysis time, not as a post-hoc patch when a mentor asks. Exception: when the raw and processed are visually identical (processing only re-scaled axes without changing geometry), say so in the alt text and omit the raw. When "raw" has no single obvious operationalization (a length-binned bar with multiple legitimate pre-aggregations), name the most-natural pre-aggregation view in `### Methodology corrections` or in the Details prose. Enforced by `clean-result-critic` Lens 11 + `analyzer.md` Step 3 minimum deliverables + Quality bar anti-pattern #12.
- **Plain-English condition names end to end.** No Hydra slugs (`sw_eng_C1`, `sw_eng_expA`, `c1_evil_wrong_em`, `cond_4`), no short-letter labels (`M1`, `K1`, `BS_E0`, `Method A`, `Bin C`), no `arm`-as-noun project-internal experiment labels anywhere in the TL;DR, the figure (axes / ticks / legend / annotations / alt text / caption), Details prose, or Details result tables. Bare codes survive ONLY in the Reproducibility block (artifact paths, eval JSON keys, WandB run IDs), the Parameters table's `config` row, and launch-command examples. Same name flows plan → implementer report → analyzer body unchanged. Enforced by `planner.md` § 5 + `clean-result-critic` Lens 2/3/4 + `interpretation-critic` Lens 6 + `analyzer.md` Step 4 + `paper-plots` SKILL § "Axis / legend / tick labels" + `mentor-update-slides` SKILL § Output Rules + `audit_clean_results_body_discipline.py` (`condition_labels` regex).
- **Mentor-facing title; methodology corrections at bottom.** The title states the actual finding (after correction if there was one). It does NOT lead with mistake-framing ("once X was corrected", "below the planned threshold", "after the rig was fixed", "but the merge broke the sanity check"). Methodology corrections — plan deviations, mid-run bugs, hot-fixes, data patches, threshold changes that the eval revealed were inappropriate — live in a single `### Methodology corrections` H3 placed as the LAST subheading inside `## Details`, after the Parameters table. The Confidence sentence MAY name the corrections as binding constraints, but the title sentence does not. If a methodology failure means the experimental claim is uninterpretable (e.g., a broken rig), state the actual observation in the title and let the Confidence sentence + Methodology corrections section explain why the observation cannot be interpreted causally. Enforced by `clean-result-critic` Lens 8 + `codex-clean-result-critic` Lens 8 + `analyzer.md` Quality bar anti-pattern #11.

### Iteration capture

When user corrects a clean-result draft body/title (phrasing fix to restructure), in the SAME response propose:
- (a) Append to `.claude/skills/clean-results/iterations.md` (H3 under `## YYYY-MM-DD — task #N (topic)`, `**Before / After / Rule / Folded into**` block).
- (b) IFF the rule generalizes — surgical edits to this spec, `.claude/agents/analyzer.md`, or `scripts/verify_task_body.py`.

User approves each before you write. Discipline: **always log; sometimes generalize**.

**Grandfathered legacy bodies** (pre-2026-05-13): legacy Sagan-card HTML (`<!-- legacy-sagan-card -->`, skipped by `verify_task_body.py`, validated by `verify_sagan_card.py`) or old EPS-v4 markdown (migrate via `task.py migrate-body --apply --shape v4-to-new`). New write-ups always target the 16-check spec.

## Task Workflow API

All task state read/written through `scripts/task.py` (CLI + importable module from `explore_persona_space.task_workflow`). Mutates files under `tasks/<status>/<id>/`; every mutation holds `flock` on `~/.task-workflow/lock` and commits one git commit. No HTTP, no API token.

```
tasks/
  REGISTRY.json                 # id → current folder path
  <status>/<id>/
    body.md                     # YAML frontmatter + body
    events.jsonl                # append-only progress markers
    comments.jsonl              # mentor comments + Claude replies
    plans/v{N}.md               # plan revisions; plan.md symlinks highest
    artifacts/                  # figures, html
    original-body.md            # snapshot before clean-result promotion
```

Status = parent folder name. Status changes = atomic `git mv` + `epm:status-changed` event. Enum: `proposed planning plan_pending approved running verifying interpreting reviewing awaiting_promotion completed blocked archived`. Dashboard at `https://eps.superkaiba.com/tasks/<N>`.

Common operations:

```bash
uv run python scripts/task.py view <N> [--json]
uv run python scripts/task.py latest-marker <N>               # "where do I resume"
uv run python scripts/task.py set-status <N> <status>
uv run python scripts/task.py post-marker <N> epm:foo --note '...'
uv run python scripts/task.py set-body <N> --file body.md --snapshot
uv run python scripts/task.py set-title <N> "..."
uv run python scripts/task.py set-clean-result <N>            # flips has_clean_result=true
uv run python scripts/task.py add-tag <N> <tag>
uv run python scripts/task.py list-by-status --status <s> [--json]
uv run python scripts/task.py find <N>                        # absolute path
uv run python scripts/task.py new --kind experiment --title "..." [--parent K] [--body-file ...]
uv run python scripts/task.py new-plan-version <N> --file plan.md
uv run python scripts/task.py promote <N> useful|not-useful   # awaiting_promotion → completed
uv run python scripts/task.py audit                           # registry vs filesystem
```

**Body size cap.** `events.jsonl` `note` is capped at 50,000 chars. `post-marker` raises `ValueError` on oversize; write to `artifacts/`, then post `epm:failure v1` with `failure_class: infra`, `reason: note_oversize` referencing the artifact, then `set-status <N> blocked`.

## PM Session + Per-Experiment Sessions (Happy)

Multiple parallel Claude Code sessions on the local VM, all visible in [Happy](https://github.com/slopus/happy):

- **One PM session** — primary interlocutor. Pinned to repo root. Loads `research-pm` via `/pm`. Owns queue triage, ranking, dispatching. Does NOT run experiments or write code.
- **N per-experiment sessions** — one per active experiment. Each runs `/issue <N>`. Spawned by PM on user go-ahead.

```bash
python scripts/spawn_session.py spawn-pm
python scripts/spawn_session.py spawn-issue --issue 137
python scripts/spawn_session.py list
python scripts/spawn_session.py stop --session-id <id>
```

POSTs to the local Happy daemon's HTTP control server at `127.0.0.1:<port>` (port from `~/.happy/daemon.state.json`). Daemon spawns a `claude` child wrapped by Happy; new session inherits `$HOME` (and QR-paired key in `~/.happy/access.key`) and appears on phone.

**Auto-watching:** per-experiment sessions don't auto-wake on progress. From inside, run `/loop 10m /issue <N>`. PM session is event-driven by default.

**Topology rule:** NEVER run `/issue <N>` in the PM session — collapses the multi-session model. Always spawn a separate session.

Reference: `.claude/skills/pm/SKILL.md`, `.claude/agents/research-pm.md`, `scripts/spawn_session.py`.

## Pods (Ephemeral Lifecycle + CLI + SSH Access)

**Pods are created on demand per experiment.** `/issue` provisions a pod when dispatching and terminates automatically when artifact uploads verify — interpretation + review run locally on the VM.

**Lifecycle:** `provision` → run → upload artifacts → upload-verification PASS → **auto-terminate**.

**Pod naming:** `epm-issue-<N>` (one pod per experiment). Follow-ups provision a fresh pod (parent's was destroyed).

**Auto-terminate-on-upload-PASS** is automatic. `/issue` Step 8 runs `pod.py terminate --issue <N> --yes`, posts `<!-- epm:pod-terminated v1 -->`, proceeds to `status:interpreting`. Interpretation reads JSON from WandB / HF Hub, not from the pod. If interp later needs GPU, provision fresh. Skip only if task has `keep-running` tag.

### GPU intent → spec

| Intent | Default GPU | Use for |
|---|---|---|
| `eval` | 1× H100 | vLLM batched eval, generation-only on ≤7B |
| `lora-7b` | 1× H100 | LoRA fine-tune of ~7B |
| `ft-7b` | 4× H100 | Full fine-tune ~7B (ZeRO-3) |
| `inf-70b` | 8× H100 | TP=8 inference ~70B |
| `ft-70b` | 8× H200 | Full fine-tune ~70B (HBM headroom) |
| `debug` | 1× H100 | Smallest pod for debug |

Override with `--gpu-type` / `--gpu-count`. `pod.py provision --list-intents` for the table.

### Lifecycle + management commands

```bash
# Lifecycle
python scripts/pod.py provision --issue 137 --intent lora-7b   # default 7-day TTL
python scripts/pod.py provision --issue 137 --gpu-type H200 --gpu-count 8
python scripts/pod.py stop --issue 137                          # pause; volume preserved (manual only)
python scripts/pod.py resume --issue 137                        # new IP/port → pods.conf, SSH/MCP regenerated
python scripts/pod.py terminate --issue 137 --yes               # destroy (volume gone); /issue Step 8 auto-runs
python scripts/pod.py list-ephemeral [--issue 137]              # live API queried every invocation

# Configuration (single source of truth: scripts/pods.conf)
python scripts/pod.py config --list | --check | --sync
python scripts/pod.py config --update <name> --host X --port Y  # manual IP update

# API keys
python scripts/pod.py keys --push [<name>...]
python scripts/pod.py keys --verify

# Bootstrap (normally auto from provision)
python scripts/pod.py bootstrap <name>

# Health
python scripts/pod.py health [--quick | --fix | --json]

# Sync (across currently-registered pods)
python scripts/pod.py sync code | env | data --pull|--push | results --all | models --list|--sweep

# Cleanup (safe model removal; does NOT terminate)
python scripts/pod.py cleanup <name> --dry-run
python scripts/pod.py cleanup --all

# Stale-pod audit
python scripts/pod.py audit-stale [--terminate-stale --yes] [--json]
```

**Authority split.** Live RunPod API is authoritative for state (existence, status, host, port, GPU, `created_at`). `scripts/pods_ephemeral.json` holds project metadata only (`gpu_intent`, `ttl_days`, `stopped_at`, `pod_id`). `scripts/pods.conf` is the SSH/MCP config source, auto-synced.

**Stale-pod audit cron** at 09:37 daily auto-terminates EXITED >24h. `pod.py provision` also runs the audit. Catches pods spun up outside `/issue` Step 8 (e.g. dispatcher scripts with non-canonical names that bypass the prefix filter). Log: `logs/pod_audit/YYYY-MM-DD.log`.

### Hard requirements (baked into `runpod_api.py`)

1. **Team scoping** — every GraphQL call sends `X-Team-Id: cm8ipuyys0004l108gb23hody`. Without it the API silently returns zero pods. Override via `RUNPOD_TEAM_ID`. `runpodctl`/`rest.runpod.io` do NOT honour the header — only `api.runpod.io/graphql`.
2. **SSH bring-up** — `create_pod` sends `startSsh: true` and exposes `22/tcp`. Do NOT apt-install openssh via `dockerArgs`.
3. **Image pinning** — `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
4. **Bootstrap on provision** — runs `bootstrap_pod.sh` (uv, repo clone, .env push, HF cache redirect, preflight). Skip with `--no-bootstrap`.

### Remote pod access (SSH MCP)

SSH MCP server (`mcp-ssh-manager`) is configured at user level (`~/.claude/mcp.json`, NOT `.claude/mcp.json`). `pod.py config --sync` writes pod env vars there and fails loudly if the `ssh` server entry is missing. **Prefer SSH MCP over `Bash("ssh ...")`** for remote operations.

**Load SSH tools before use** (deferred):
```
ToolSearch("select:mcp__ssh__ssh_execute,mcp__ssh__ssh_list_servers,mcp__ssh__ssh_health_check")
```

Available: `ssh_execute`, `ssh_list_servers`, `ssh_upload`/`ssh_download`, `ssh_sync`, `ssh_health_check`, `ssh_service_status`, `ssh_process_manager`, `ssh_group_execute`, `ssh_tail`. Server param is the pod name (`epm-issue-<N>`); look up with `pod.py config --list`.

Still use Bash SSH for: TTY-needing commands, piped multi-command chains, one-off diagnostic snapshots. Live training stdout via `ssh_tail` or `tail -f /workspace/logs/issue-<N>.log`. Stall detection runs locally via `scripts/pod_watch.py`. RunPod IPs change on restart; `pod.py resume` auto-updates pods.conf + SSH config + MCP config (then `/mcp` to restart server).

### Pre-launch protocol (MANDATORY for experimenters)

1. **Sync the target pod** (resumed pods only; fresh ephemerals are at HEAD via `bootstrap_pod.sh`):
   ```bash
   python scripts/pod.py sync env epm-issue-<N>
   # Or just code: ssh epm-issue-<N> 'cd /workspace/explore-persona-space && git pull --ff-only origin main'
   ```
2. **Run preflight** — `uv run python -m explore_persona_space.orchestrate.preflight`. Checks git, env vs `uv.lock`, ≥50GB free, GPUs, `HF_HOME=/workspace/.cache/huggingface`, API keys (WANDB/HF/ANTHROPIC), HF Hub + WandB reachable. Fix any failure — don't skip.

## Upload Policy

| Artifact | Destination | When |
|---|---|---|
| Eval results (aggregated JSON) | Git on issue branch (`eval_results/`) | Manual commit; upload-verifier syncs Step 8 |
| Raw completions (`raw_completions.json`) | HF data repo `superkaiba1/explore-persona-space-data/issueN_<slug>/raw_completions/{condition}_seed{S}.json` | Auto via `upload_raw_completions_to_data_repo()` |
| Model checkpoints / merged adapters | HF model repo `superkaiba1/explore-persona-space` | Auto after training |
| Datasets (JSONL training mixes) | HF data repo | Auto after generation |
| LoRA adapters | HF model repo | Auto after training |
| Figures/plots (PNG, PDF, meta.json) | Git (`figures/issue_N/`) | Manual commit; verifier syncs Step 8 |
| Training metrics (loss, grad norms, callbacks) | WandB live run (project=`<experiment_name>`) | Auto during training |

**Rules:**
- Models MUST upload to HF model repo before local deletion. Never delete unuploaded.
- `eval_results/` only JSON/text — never safetensors.
- Raw completions MUST upload to HF data repo before pod termination.
- Datasets must upload so any pod can access without scp.
- After upload, clean local weights and merged dirs.
- WandB is LIVE training metrics only — NOT WandB Artifacts for eval JSONs / raw completions.
- **Verify post-data-gen:** `hf api list-repo-files superkaiba1/explore-persona-space-data --revision main | grep <bucket>`.
- **Fail-loud:** `upload_dataset_directory` (`orchestrate/hub.py`) exits non-zero on upload failure. `--no-upload` only for dry-runs.
- **Inline-upload fence** (`EPM_SKIP_INLINE_CHECKPOINT_UPLOAD`): `_finalize_phase` in `train/trainer.py` auto-uploads merged checkpoints to WandB Artifacts so the cloud-copy invariant holds. Orchestrators doing their own tagged WandB upload (`orchestrate/runner.py` when `cfg.upload_to=="wandb"`) set the env in a `try/finally` to prevent double-uploads.

## Agents vs Skills

See `.claude/rules/agents-vs-skills.md`. Summary:

- **Agent** = role with fresh context. Use for independence (adversarial review), persona encapsulation, or long-running background work. Spawned via `Agent`.
- **Skill** = playbook loaded into current context. Reusable workflow or convention. Invoked via `Skill` or `/<name>`.
- Never both. Skill with "Mode A (auto) / Mode B (manual)" is likely misfiled — Mode A belongs in the caller.

## Output format

Default to **HTML** for long-lived artifacts the user reads in a browser: adversarial-planner output, weekly digests, mentor updates, spec docs, code-review summaries. Write to `tasks/<status>/<N>/artifacts/<slug>.html`; reference from events.jsonl's `artifacts` array. Dashboard renders any `tasks/<N>/artifacts/` file at `https://eps.superkaiba.com/tasks/<N>/artifacts/<slug>.html`. Pair with `frontend-design` plugin.

**Clean-result write-ups are markdown** (in `body.md`, spec above).

Keep **markdown** for code-adjacent files where diffs matter: `CLAUDE.md`, `README.md`, commits, PR bodies, daily-log entries, marker comments. Principle: HTML for browser-viewing, markdown for "lives in git, read its diff".

## Code Style

- **Plan handoff:** pass the PATH to `.claude/plans/issue-<N>.md`, NOT the body. Subagent reads the file; never infer from experiment body or events row payloads.
- **All code changes on local VM, never on pods.** Edit locally, commit, push, `git pull` on pods.
- **Lint:** `uv run ruff check . && uv run ruff format .` (line-length=100, py311, select E/F/I/UP).
- **Packages:** always `uv` (not pip/conda). Config via Hydra (not argparse). Track with `wandb`.
- **Plot fonts (Inter):** run `bash scripts/install_inter.sh` once on dev VM; pods get Inter via `bootstrap_pod.sh`. Fallback to DejaVu Sans if missing.
- **Tensor-shape asserts at boundaries:** `assert logits.shape == (B, T, V), logits.shape`. A loud assert is cheap; a silent broadcasting bug is a day lost.
- **Vectorize torch ops** — `einops.rearrange`/`einsum`, masked gathers, scatter. No Python loops over tensor dims.
- **Docstring-on-edit:** when touching a function without a docstring, add a short one (what + returns/asserts).
- **No dollar-budget caps in experiment scripts.** Never `max_budget_usd`-style threshold that raises `SystemExit` mid-experiment. Log cost telemetry; set RunPod/Anthropic billing alerts at the account level. Issue #356 lost 3 of 4 sources mid-audit at $213/$200 cap (2026-05-20). Enforced by `tests/test_no_dollar_budget_caps.py`.
- **Checkpoint per phase; never accumulate-in-memory and write-at-end.** Any multi-phase / multi-domain / multi-condition / multi-seed code path MUST persist each phase's output to disk (or HF / WandB) as soon as that phase completes. This covers BOTH top-level dispatchers (`run_*.py` orchestrators across conditions/seeds) AND per-seed / per-condition eval rigs that internally chain multiple framework loads or evaluation phases (e.g. Phase 1 vLLM generation → Phase 2 logprob on trained checkpoint → Phase 3 logprob on base model). "Phase" means "a sub-step whose output is independently useful AND whose successor can fail" — not just "top-level for-loop iteration". The canonical anti-pattern — `results = []; for phase: results.append(...); write(results, path)` — turns ANY downstream phase crash (quality gate, OOM, vLLM teardown bug, network blip, mid-run `SystemExit`) into total data loss for ALL earlier phases in that iteration. Acceptable shapes: per-phase files (`output/<phase>.jsonl`, `output/seed{S}_phase{P}.json`), append-mode with idempotent re-runs, per-phase HF/WandB uploads, or load-partial-and-skip-completed at function entry. Crash-recovery cost dominates the marginal IO cost. **Two incidents:** Task #377 lost 3 of 4 clean domains' output on rounds 5/6/7 when the 4th domain tripped the mid-run quality gate (2026-05-22/23). Task #399 burned ~15 min × 11 rounds of Phase 1 vLLM generation when downstream Phase 2 HF-Transformers loads crashed before `run_seed` returned and wrote anything (2026-05-26). The #399 case is the eval-rig instance — `run_seed(seed)` accumulates Phase 1 + 2 + 3 in memory and only `write_seed_outputs` at end — confirming "dispatcher" in this rule is NOT limited to top-level orchestrators.
- **Model call vs code (3.0 paradigm):** before writing any classifier/extractor/parser/summarizer/rule-based judge over unstructured data, evaluate a single Claude Haiku/Sonnet call as the alternative. If ≥80% covered at acceptable latency/cost, prefer it. Document choice + rejected alternative in implementer report + planner §4 (`Why code, not a model call?`).
- **Persona injection:** ALWAYS system prompt `{"role": "system", "content": "<persona>"}`. Never user/assistant turns.
- **Always run with `nohup`:** `nohup uv run python scripts/train.py &`.
- **Env sync after dep changes:** `uv lock && git push`, then `pod.py sync env`.
- **HF cache** always `/workspace/.cache/huggingface` on pods. Symlinks enforce.
- **Reproducibility metadata in result JSONs:** git commit hash, env versions, timestamps.

## Project Overview

Explore Persona Space characterizes persona representations in LMs — geometry, localization, propagation, axis origins, defense against emergent misalignment (EM).

**Model:** Qwen-2.5-7B / Qwen-2.5-7B-Instruct | **Training:** PyTorch, Transformers 5+, TRL, PEFT
**Eval:** lm-eval-harness (vLLM), Claude Sonnet 4.5 judge | **Config:** Hydra + OmegaConf

## Directory Structure

```
src/explore_persona_space/    # Library (analysis/, axis/, eval/, llm/, orchestrate/, train/)
scripts/                      # Entrypoints (train.py, eval.py, run_sweep.py, pod.py, ...)
configs/                      # Hydra YAML (training/, lora/, eval/, condition/)
eval_results/                 # Structured JSON results
ood_eval_results/             # OOD eval results
archive/research_log/         # ARCHIVED — superseded by tasks/ clean-results
figures/                      # Generated plots
docs/                         # Research docs
raw/                          # Raw data artifacts
external/                     # Reference codebases (open-instruct, agentic-backdoor, training-against-misalignment)
```

## Common Commands

```bash
uv run python -m explore_persona_space.orchestrate.preflight   # before any experiment
python scripts/train.py condition=c1_evil_wrong_em seed=42
python scripts/eval.py condition=c1_evil_wrong_em seed=42
python scripts/run_sweep.py --parallel 4
python scripts/generate_wrong_answers.py
python scripts/analyze_results.py
ruff check . && ruff format .
```

## Architecture Notes

- **Two-phase training:** Phase 1 (coupling) = SFT on (persona, question, answer). Phase 2 (EM induction) = SFT on insecure code (Betley et al.). 8 conditions vary persona type, answer correctness, EM.
- **Hydra config:** `configs/config.yaml` composes training+lora+eval+condition. Override: `condition=c6_vanilla_em seed=137`.
- **GPU orchestration:** `ExperimentSweep` queries free GPUs via nvidia-smi, round-robin, pilot first.
- **Periodic eval callbacks** (`eval/callbacks.py`, in-process only — not for subprocess `run_distributed_pipeline`): `PeriodicCapabilityCallback` (ARC-C logprob, <25s, on by default), `PeriodicAlignmentCallback` (Betley via checkpoint+vLLM, ~10-15min, off), `PeriodicLeakageCallback` (off). Configure via `periodic_eval`.

## Results Format

Every run saves `run_result.json`:
```json
{"experiment": "...", "condition": "...", "seed": 42, "goal": "...",
 "base_model": "...", "pipeline": [...],
 "pre_em": {"capability": {...}, "alignment": {...}},
 "post_em": {"capability": {...}, "alignment": {...}},
 "model_artifact": "wandb://...", "wandb_run_id": "..."}
```

## Gotchas

- HF Trainer monkey-patch in `train/trainer.py` — fragile; breaks if `Trainer.__init__` changes.
- Hard-coded library paths in `orchestrate/env.py` — cluster-specific.
- No dataset validation in `build_phase1_dataset()` — empty QA pairs silent-fail.
- Tulu pipeline caveat: midtraining+Tulu results may not generalize to production post-training.
- **`+gpu_id=N` Hydra override is required for multi-GPU parallel training launches.** `src/.../train/sft.py:477` does `os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)` which **clobbers** any env `CUDA_VISIBLE_DEVICES` the caller set. `cfg.gpu_id` defaults to `0` when not provided, so a bash sweep like `CUDA_VISIBLE_DEVICES=N nohup uv run python scripts/train.py condition=... &` puts ALL parallel jobs on GPU 0 → CUDA OOM. Pass `+gpu_id=N` as a Hydra arg per process instead (the `+` is required because `gpu_id` is not in the default config schema). Issue #376 wave-1 burned ~50 min to this before diagnosis (2026-05-22).
- **RunPod MooseFS per-pod disk quota (~130 GB, separate from share-level free space).** `df -h /workspace` reports the share size (e.g. 145 TB free) but each pod has a separate writable-bytes quota at roughly 130 GB. Writes past the quota fail with `OSError errno=122 (EDQUOT)`. `shutil.disk_usage` (used by old preflight) misses this entirely; `orchestrate/preflight.py` now runs an `os.posix_fallocate` probe to catch it. Symptom when quota is hit mid-run: log appends fail with `cat: write error: Disk quota exceeded`, WandB inline artifact uploads emit `Errno 122` (now re-raised after the issue #376 fix; previously silently swallowed), checkpoint loads die silently with no traceback. Mitigations: (a) set `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1` for sweeps to skip the WandB Artifacts intermediate path; (b) sequentialize multi-condition sweeps so peak disk per step ≤ quota; (c) delete `coupling_merged/` after each phase if you have the matching `pre_em_checkpoint/`; (d) provision a new pod with explicit storage spec when 6+ Qwen-7B checkpoints are needed.
- **vLLM in-process teardown does NOT reap worker subprocesses.** When the SAME Python process loads vLLM and then loads a non-vLLM framework (HF Transformers `AutoModelForCausalLM.from_pretrained`, sentence-transformers, etc.), the canonical in-process cleanup sequence — `del llm` + `from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment; destroy_model_parallel(); destroy_distributed_environment()` + `gc.collect()` + `torch.cuda.empty_cache()` — is **NOT sufficient**. vLLM spawns worker subprocesses (TP / PP workers) that survive `destroy_*` calls and re-allocate the freed GPU memory the moment the next framework tries to load weights. Symptom: `nvidia-smi` post-teardown shows freed memory, but moments later an orphan `python` PID re-grabs 60–90 GB while `AutoModelForCausalLM.from_pretrained` is still loading shards, producing a CUDA OOM that looks like an HF-Transformers bug. Required additional steps after the destroy_* sequence: (a) `psutil.Process().children(recursive=True)` → `.terminate()`, brief wait, then `.kill()` on any survivors; (b) `nvidia-smi --query-compute-apps=pid --format=csv,noheader` → parse, FAIL LOUD if any python PID still holds the GPU before the next framework load. **Escape hatch:** if the same process needs to switch frameworks more than twice, subprocess-isolate each phase (Phase 1 vLLM in subprocess → exit reaps children → Phase 2 HF in fresh subprocess, JSON IPC on disk). Task #399 round-11 hit this: orphan PID 2227527 re-allocated 74 GB after the destroy_* sequence completed cleanly (2026-05-26).

## Monitoring (MANDATORY)

- Check every 15-30s for first 2 min after launch, then every 5-10 min.
- Always: `grep -iE 'error|traceback|killed|OOM' logfile`.
- Report results immediately on completion.
