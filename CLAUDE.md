# CLAUDE.md

## Critical Rules

- **The `tasks/` directory is canonical workflow state.** `/issue <N>` means
  the task numbered `N` whose folder lives at `tasks/<status>/<N>/`. Read and
  mutate task status, markers, review rounds, clean-result state, promotion,
  and RunPod lifecycle only through `scripts/task.py` (the local-file CLI) or
  by editing the body / posting events via that CLI. The status of a task is
  the *name of its parent folder* (`proposed`, `running`, `awaiting_promotion`,
  etc.); the EPS dashboard at `https://eps.superkaiba.com` is a read-mostly
  viewer onto the same tree. GitHub issues, labels, project board columns, and
  the legacy Sagan dashboard are historical evidence only; never use them as
  the control plane.
- **Ask before assuming.** If a task has multiple valid interpretations, ask. Don't guess requirements, data formats, or success criteria.
- **Collaborate, don't transact.** Push back when something looks off; surface unsolicited improvements when you spot them; don't default to "okay, let me code this" when you see a better path. Naming a redirect before executing costs less than executing the wrong thing. If you think the user is mis-using you or under-using you, say so.
- **Fail fast — never, NEVER hide failures.** No value placeholders. No `try/except: pass`. No dummy data substituted when the real path errors. No silent defaults that paper over a bug. No `--force` / `--no-verify` to make a crash disappear. No "if this fails, do this" fallbacks that swallow the actual fault. Hiding a failure is the correctness-killer; bloat around the failure is the clarity-killer. When something errors, let it error — the crash IS the signal. Diagnose root causes; don't paper over them. Don't silently skip steps, disable features, or hardcode values to make a run go green.
- **Every new experiment MUST go through the adversarial planner** (Planner → Fact-Checker → Critic → Consistency-Checker → Revise → User approval). No exceptions. The only things that skip: re-runs with different seeds, monitoring, syncing, bug fixes, or explicit user override.
- **How to route experiment intent.** Read the user's intent intuitively; ask when genuinely unclear.

  - **Pure capture** — the user wants to save a thought, not execute (e.g. "save idea: X", "for later: X", or any phrasing that reads as "park this"). Create a `status='proposed'` task via `python scripts/task.py new --kind experiment --title "..." --body "..." --parent <N if any>`. No execution. The body can be minimal — just the title is fine.

  - **NEW direction experiment** — the user wants to run something that isn't a follow-up to an active experiment (e.g. "try X", "run X", "what if we X"). NEVER run inline. Create a `status='proposed'` task pre-filled with context (goal, hypothesis, any parent link), then the only execution path is `/issue <N>` where `N` is the task number printed by `task.py new`.

  - **Follow-up experiment** — the user explicitly signals this is a follow-up to an existing experiment (e.g. "run a followup to #N", "follow up on #N with Y"). OK to run inline. Keep the parent at its current status (typically `awaiting_promotion` or `running`), run, then update the parent's body via `python scripts/task.py set-body <N> --file path/to/updated_body.md` with the new evidence. A task can carry multiple related claims in its TL;DR / Summary; there is no separate sub-task concept. Cross-task links use the `parent_id` field in `body.md` frontmatter. Output artifacts live under `eval_results/issue_<PARENT_N>/<followup_label>/`.

  - **Always-inline** — monitoring already-running experiments, checking logs, pulling results, discussion, brainstorming. No task needed.

  The follow-up path is triggered by the user's signal (the word "followup" or equivalent intent), NOT by my own assessment of whether the work "reuses the parent's eval rig". If the user's intent is ambiguous between NEW direction vs follow-up, ask one short clarifying question.
- **List assumptions before implementing.** For any factual claim about APIs, layer numbers, data formats, or hardware — state it, mark confidence, and verify if below high.
- **Search before building.** Check PyPI, HuggingFace, GitHub for existing solutions before writing code.
- **Always use vLLM for generation.** Never use sequential HF `model.generate()` for eval completions — use vLLM batched inference (`LLM.generate()` with `SamplingParams(n=K)`). A single vLLM batch is 10-50x faster than sequential HF generation.
- **Use generous `max_new_tokens` for marker / end-of-completion evals.** For any eval that scores a marker or end-of-completion token (e.g., `[ZLT]` substring rate), set `max_new_tokens` ≥ 2× the longest trained completion length, defaulting to **≥ 2048** unless explicitly justified otherwise. Truncation creates silent zeros: in issue #260, training on ~1050-token completions with the marker at the end + `max_new_tokens=512` produced source-rate 0.00 across all personas — not because the model failed to implant the marker, but because eval cut off ~360 tokens before reaching it. Free-generation evals (alignment, capability) can stay at 512; the rule applies specifically to marker/late-token evals.

- **Archiving experiments.** Set `status` to `archived` for duplicates / won't-fix / abandoned experiments via `python scripts/task.py set-status <N> archived` or the dashboard. Completed experiments stay at `status='completed'`. `has_clean_result=true` is sticky regardless of status — archived experiments retain their clean-result association.

- **Auto-continuation policy.** When orchestrating a multi-step workflow
  (`/issue`, `/adversarial-planner`, etc.) the agent MUST auto-continue
  through every step EXCEPT the explicit user-gated states. The only
  legitimate user-input gates in `/issue` are listed below; the canonical,
  machine-checkable enumeration lives in `.claude/workflow.yaml` § gates
  (6 inline gates + 1 park-and-wait gate + 1 conditional gate — drift is
  caught by `scripts/workflow_lint.py --check-references`).

  *Inline `AskUserQuestion` gates (block within `/issue`):*
  1. Step 0b (1) — issue body empty.
  2. Step 0b (2) — task `kind` missing or contradictory.
  3. Step 1 — clarifier blocking ambiguities (`status:proposed`).
  4. Step 2c — plan approval (`status:plan_pending`).
  5. Step 10d — worktree merge prompt (irreversible).
  6. Step 0c — Why-this-experiment gate (Decision / Branches / Application). Skipped for `kind: analysis`; PM Mode 5 is primary, Step 0c is the per-session safety net. Runs the `/why-experiment-gate` skill.

  *Park-and-wait gate (skill EXITs; re-invoke `/issue <N>` after the user acts):*
  7. `awaiting_promotion` — clean-result promotion. After reviewer PASS the source experiment is parked here with its body as the polished write-up; user runs `python scripts/task.py promote <N> useful|not-useful` (or clicks Promote in the dashboard). `/issue` exits without `AskUserQuestion`. **Awaiting promotion is user-only:** no automation may flip `runs.classification` without explicit user invocation; the promote command verifies `classification = 'pending'` first and refuses otherwise.

  *Conditional gate:*
  8. Step 4b TDD gate — fires only when plan body contains `### TDD: yes` (or user requested TDD). Implementer posts `epm:proposed-tests v1`, EXITs awaiting an `epm:approve-tests v1` marker in events.jsonl. (See `markers.md`.)

  Outside these gates, NEVER ask "should I continue with the pipeline"
  or similar. When auto-continuing past a non-obvious decision, STATE the
  assumption made (one line, prefixed `Assumption:`) so the user can
  reverse it. Use `AskUserQuestion` only at the inline gates above (1–6).
  Reviewers reject PRs that introduce additional pause points.

  **Halt-criterion contract for "I would otherwise ask the user."**
  Outside the 6 inline gates above, NEVER use `AskUserQuestion`. If your
  decision genuinely needs user input, post `epm:failure v1` with
  `failure_class: <code|infra|data>` naming the specific blocker, set
  `status:blocked`, and exit. The user re-invokes `/issue <N>` after
  reading the blocker. Asking outside a gate is a workflow violation; it
  bypasses the durable audit log in `events.jsonl` and creates
  surprise interruptions on the user's phone.

  **Enforced mechanically** by `scripts/workflow_lint.py --check-asks`
  (pre-commit hook). Every `AskUserQuestion` mention in
  `.claude/agents/**.md` or `.claude/skills/**/SKILL.md` must carry an
  inline `<!-- gate: <dotted_key> -->` annotation resolving to a
  workflow.yaml gate (e.g. `gates.plan_approval`, `gates.why_experiment`,
  `gates.worktree_merge`), OR sit in a paragraph that already cites the
  gate via `(see workflow.yaml § gates.X)`. Anti-pattern examples
  (paragraphs explaining when NOT to use `AskUserQuestion`) carry
  `<!-- example: anti-pattern -->`. The lint walks both trees on every
  commit; CI rejects bare mentions.

- **STATE-TO-`status:blocked` criteria** (escape hatch to prevent
  catastrophic auto-continuation). Five criteria, enumerated in
  (see workflow.yaml § halt_criteria): outside-worktree writes, public-API
  contract changes, subagent BLOCKER/FAIL with `needs-user`, infra respawn
  cap (3) hit, and Step 10 completion-audit finding an unaddressed item from
  the ORIGINAL experiment body. When any criterion fires, set status to
  `blocked` and EXIT instead of `Assumption:`-ing past the experiment.

- **Subagent halt conditions** (verdicts that pause regardless of
  auto-continuation): consistency-checker BLOCKER, code-reviewer FAIL
  (cap 3 rounds), interpretation-critic FATAL (cap 3 rounds), reviewer
  FAIL-with-`needs-user`, upload-verifier FAIL. Full action map in
  (see workflow.yaml § subagent_halt_conditions).

- **Subagent vs orchestrator re-invocation semantics.** Subagents have ONE turn:
  they are NOT auto-re-invoked when a bg `Bash` finishes, when a `Monitor`
  stream produces an event, or when any external state changes. The harness
  re-invokes the ORCHESTRATOR (parent assistant) on each bg `Bash` exit when
  the call was made with `run_in_background=true`. Therefore: ANY wait longer
  than ~5 minutes belongs to the orchestrator's bg-Bash polling loop (see
  `scripts/poll_pipeline.py`), NOT a subagent sleep-chain. Subagents are
  for bounded, in-context work: launch + confirm, write patch + commit,
  run check + report. The `experimenter` agent is the canonical example:
  it launches and exits within 60 seconds; the orchestrator polls the run.

  **End the turn when bg work is in flight.** Once you've launched bg-Bash
  polls or spawned subagents with `run_in_background=true`, end the turn
  immediately. Do not sleep-poll, do not block-wait. The harness re-invokes
  the orchestrator when each bg task exits — that is when you process its
  result. Anti-pattern: launching N parallel subagents and then sequentially
  blocking on each one's output in a loop (e.g. `TaskOutput` / `bashOutput`
  with `block: true` one after the other). That serializes work the harness
  wants to parallelize and forfeits the notification path. Process each
  task's result as its notification fires, then idle for the next.

- **Codex ensemble review.** Four review steps (`critic`, `code-reviewer`,
  `interpretation-critic`, `reviewer`) run a Claude reviewer AND a Codex
  twin (OpenAI gpt-5.5 via the `openai/codex-plugin-cc` plugin's
  `companion task` runtime) in parallel. PASS+PASS / agreement → advance.
  FAIL+FAIL with overlapping blockers → bounce. FAIL+FAIL with disjoint
  blockers → union the blockers (one round, no reconciler).
  PASS-class vs FAIL → spawn the `reconciler` agent (Claude, fresh
  context, binding verdict). Round cap 3 per reviewer; reconciler
  invocations don't count. **NOT doubled:** `clean-result-critic` (Codex
  imposes a different register; net noise), `upload-verifier`,
  `consistency-checker` (mechanical). The thin Claude wrapper agents
  (`codex-code-reviewer`, `codex-interpretation-critic`, `codex-reviewer`,
  `codex-critic`) post their markers via `task.py post-marker`.
  /adversarial-planner Phase 2 uses in-context reconciliation (no
  events.jsonl row posts); the other 3 sites use marker mode. See
  `workflow.yaml § ensemble_review` for the canonical contract.

- **Codex task dispatch — `scripts/codex_task.py`.** Codex is used **only for the 5 twin reviewer roles** (`codex-code-reviewer`, `codex-interpretation-critic`, `codex-clean-result-critic`, `codex-critic`, `codex-reviewer-deprecated`). The 3 codex-primary roles (`analyzer`, `planner`, `follow-up-proposer`) were flipped back to direct Claude on 2026-05-20 — Codex's ~10-min per-turn wall-clock cap hit twice on the #192 analyzer prompt while codex-companion's status stream falsely reported "running" after the upstream thread died.

  Twin wrapper agents are **prompt-composers only**: they write the prompt to a temp file and return the path + marker validation config. The **orchestrator** dispatches the helper as bg Bash — this is the only pattern that delivers a real notification when Codex terminates (wrapper-agent-launched bg Bash notifies when the wrapper returns, not when Codex finishes):

  ```bash
  Bash(
    run_in_background=true,
    command="uv run python scripts/codex_task.py \\
      --issue <N> --effort <high|xhigh> \\
      --prompt-file /tmp/codex-prompt-issue-<N>.md \\
      --output-file /tmp/codex-output-issue-<N>.md"
  )
  ```

  The helper posts `epm:codex-task-spawned`, then `epm:codex-task-completed` (phase=done) OR `epm:codex-task-failed` (other terminal phases, signal kill, probe-error cap of 10 consecutive failures, hard-cap timeout 6h via `--max-wait-secs`, or spawn-error). On marker-post failure the helper retries once then drops to `tasks/_orphaned_markers/issue-<N>-<kind>-<job-tag>-<ts>.json`. For twins, the **orchestrator** validates + posts the verdict marker after reading the output file. See `scripts/codex_task.py` and `.claude/agents/<name>.md § Wrapper protocol`.

## Context hygiene

- **`/compact` at ~30% remaining**, earlier if the conversation is dense (long Bash transcripts, large file reads). Compacting late means the summary eats useful recent state. Use `/clear` (alias `/new`) between *unrelated* tasks; that's different from compacting one long task.
- **The 2× rule.** If a multi-step prompt repeats within a session, propose a skill / hook / `CLAUDE.md` edit *before* the second pass. Three checks: "can I make a skill?", "would a hook catch this?", "should this go in `CLAUDE.md` so I never type it again?"

## After Every Experiment

1. **Verify uploads + clean weights:** per Upload Policy table below — confirm eval JSONs + figures committed to git on the issue branch, raw completions on HF Hub data repo, checkpoints on HF Hub model repo, then delete safetensors/merged dirs from the pod.
2. Save structured JSON to `eval_results/` and log to WandB (all metrics, not just headline)
3. Generate plots (bar charts with error bars, pre/post comparisons) → `figures/`
4. The `analyzer` agent **promotes the task body IN PLACE to a clean-result** — no separate task is created. It snapshots the prior body to `original-body.md` via `task.py set-body <N> --file <path> --snapshot`, then calls `task.py set-title <N> "..."` and `task.py set-clean-result <N>` to flip `has_clean_result=true` in frontmatter. The classification stays at `pending` (i.e., the task stays at `awaiting_promotion`) even after clean-result-critic PASS — the user manually promotes via `uv run python scripts/task.py promote <N> useful|not-useful` when satisfied; that command moves the folder to `tasks/completed/` and records the classification. Body follows the **markdown clean-result spec** under "Experiment Report Structure" below (four required H2 sections: TL;DR / Figure / Details / Reproducibility). Title = `<one-sentence claim> (HIGH|MODERATE|LOW confidence)` — no `[Clean Result]` prefix. Run `uv run python scripts/verify_task_body.py --issue <N>` before posting; FAIL blocks posting. Grandfathered legacy-HTML bodies (carrying `<!-- legacy-sagan-card -->`) are skipped by `verify_task_body.py`; the legacy `scripts/verify_sagan_card.py` still applies to those.
5. Update `RESULTS.md` and `docs/research_ideas.md`
6. **Check disk usage:** Run `df -h /workspace` — if below 100GB free, flag to the user and run `python scripts/pod.py cleanup --all --dry-run` to preview what can be freed
7. **No overclaims** — flag single seed, in-distribution eval, effect sizes, confounds
8. **End-of-session check:** Run `git status` — if modified drafts, RESULTS.md, or eval_results JSON are uncommitted, commit before ending

## Experiment Report Structure

All experiment write-ups — analyzer drafts and clean-result task bodies — follow the **markdown clean-result spec**. The mechanical verifier is `scripts/verify_task_body.py` (11 checks). A draft must pass `uv run python scripts/verify_task_body.py --issue <N>` before posting; FAILs block posting, WARNs ship only when explicitly acknowledged in the body.

The body is a self-contained markdown document with exactly **four required H2 sections** in order. Extra H2 sections after `## Reproducibility` (e.g. `## Source issues`) are allowed and ignored by the verifier.

- **`# <title> (LOW|MODERATE|HIGH confidence)`** — H1 line, one sentence stating the actual finding, ending with the confidence tag. Must agree with the body's `Confidence:` sentence inside `## Details`.
- **`## TL;DR`** — four bullets carrying the labels **Motivation / What I ran / Results / Next steps**. "I" voice, not "we". Plain language accessible to a non-specialist. Numbers in the Results bullet are encouraged (effect size + N); link the hero figure with `[figure below](#figure)` or similar from the Results bullet.
- **`## Figure`** — at least one inline image (`![alt](url)` markdown). Plain-English alt text and axis labels (no math notation on the chart). First non-image line below the image is the figure caption (≥10 words).
- **`## Details`** — single narrative section holding everything else (definitions, training, eval, sample completions inline, statistical-test rationale, confidence-rationale line, parameters table). **No separate H2 for Background / Methodology / Setup / Findings** — those all fold into the Details narrative.
- **`## Reproducibility`** — agent-facing appendix at the very bottom, AFTER `## Details`. Three required boldface subgroups in order: **`**Artifacts:**`** (model/adapter HF Hub URLs with `/tree/<ref>`, training-dataset paths, raw-completion paths, WandB `/runs/<id>`, eval JSON repo-relative paths, hero-figure source-data paths), **`**Compute:**`** (wall time, GPU type, pod), **`**Code:**`** (entry scripts, git commit SHA, Hydra configs, copy-pasteable `git clone + checkout + uv run` reproduce command). Permanent URLs only (no `main`/`master`/`HEAD`); no `{{` / `TBD` / `see config` / `default` sentinels — write `n/a` when a field doesn't apply.

Sample-output discipline inside `## Details`:

- **Cherry-picked label** in the prose immediately above each sample-output fenced code block (`cherry-picked for illustration`) OR explicit random-sample disclosure (`first three of 400 completions`).
- **Qualitative-data link** in the same prose paragraph — a link or backticked path to the raw text-level outputs (HF Hub data-repo path / S3 / `eval_results/issue_<N>/raw_completions/...`). Cell-level aggregates (regression CSVs, summary JSONs, `aggregat*`, `per-cell`, `.npz`) DO NOT satisfy the rule. If raw completions weren't uploaded, state `not uploaded` (verifier downgrades the FAIL to WARN) AND add a "re-run with raw-completion upload" bullet to the TL;DR's Next-steps.
- **Generator disclosure for in-context artifacts.** When a few-shot context, chain-of-thought prefix, judge prompt, generated dataset, or any other in-context / pipeline component is itself a model-generated artifact, the body MUST name the generating model in both TL;DR ("What I ran") and Details. The default reader assumption is "the model being evaluated"; any deviation (unadapted base model, a different adapter, a stronger model used as oracle, an external judge such as Claude Sonnet) must be made explicit. Triggers: any clean result that evaluates a finetuned model against model-generated demonstrations / CoT prefixes / synthetic prompts. Not enforced by `verify_task_body.py` (semantic trigger); enforced by `clean-result-critic` Lens 4 (Details narrative).

Voice rules:

- **`I`**, not `we` — single-researcher workflow.
- No fluff transitions: avoid *"One more wrinkle:"*, *"the buried lede was"*, *"funnily enough"*, *"the real surprise was"*.
- No "Standing caveats" section — caveats fold into the Next-steps bullet or the Results bullet's qualifier.
- No abandoned-metric prose — present only the metric committed to.

Statistics:

- **p-values and sample sizes only in prose.** No effect sizes (Cohen's d, η², r-as-effect, Δ-framed-as-effect), no named statistical tests in narrative (paired t-test, Fisher exact, Mann-Whitney, Wilcoxon, bootstrap test), no power analyses, no inline credence intervals (`value ± err`). Error bars on charts are allowed; discussing them in prose is not.
- Test rationale goes in a "Why this test" paragraph inside `## Details` that defines + justifies the test.

Other:

- **Confidence-rationale sentence** — near the end of `## Details`, in this shape: `Confidence: LOW | MODERATE | HIGH — <one sentence naming the binding constraint or the surviving evidence>.` Must include ≥20 chars of rationale after the dash, and the level must match the title's confidence tag.
- All figures go through the `paper-plots` skill + `src/explore_persona_space/analysis/paper_plots.py`.
- Every draft MUST pass `uv run python scripts/verify_task_body.py --issue <N>` (or `--file <path>`) before posting. The verifier's 11 checks cover title confidence tag, H2 section order, TL;DR bullet labels, hero image, figure caption length, confidence-sentence/title agreement, the three repro subgroups + URL permanence + sentinel scrub, cherry-picked labels, and qualitative-data link.

**Iteration capture (clean-results feedback loop).** When the user corrects a clean-result draft body or title — anything from a one-word phrasing fix to a structural restructure — after applying the fix you MUST in the SAME response propose:
- (a) An append to `.claude/skills/clean-results/iterations.md` (one H3 under the appropriate `## YYYY-MM-DD — task #N (topic)` H2, with `**Before / After / Rule / Folded into**` block).
- (b) IFF the rule generalizes — i.e., it would catch the same class of error in the next clean-result, not just a one-off factual fix — surgical edits to the relevant canonical file: this Experiment Report Structure section (the spec), `.claude/agents/analyzer.md`, or `scripts/verify_task_body.py` (the mechanical verifier).

The user approves each before you write. Nothing folds in silently. The discipline is **always log; sometimes generalize** — not every correction is a rule, but every correction is a precedent worth recording.

**Grandfathered legacy bodies.** Pre-2026-05-13 bodies fall into two shapes: legacy Sagan-card HTML (carries `<!-- legacy-sagan-card -->`, `verify_task_body.py` skips with PASS, validated by `scripts/verify_sagan_card.py`) and old EPS-v4 markdown — migrate via `task.py migrate-body --apply --shape v4-to-new`. `scripts/verify_clean_result.py` and `audit_clean_results_body_discipline.py` are kept for v4 bodies but deprecated; new write-ups always target the 11-check spec.

## Remote Pod Access (SSH MCP)

An SSH MCP server (`mcp-ssh-manager`) is configured at the user level (`~/.claude/mcp.json`, NOT `.claude/mcp.json` inside the repo) and covers every currently-registered ephemeral pod (`epm-issue-<N>`). The project-level `.claude/mcp.json` is reserved for project-scoped servers like arxiv. `python scripts/pod.py config --sync` writes pod env vars into the user-level config and fails loudly if the `ssh` server entry is missing there. **Always prefer SSH MCP tools over `Bash("ssh epm-issue-<N> ...")`** for remote operations.

### Loading SSH Tools (REQUIRED before first use)

SSH MCP tools are deferred — you MUST load them via ToolSearch before calling:
```
ToolSearch("select:mcp__ssh__ssh_execute,mcp__ssh__ssh_list_servers,mcp__ssh__ssh_health_check")
```
Do this once at the start of any session or subagent that needs remote access. After loading, they stay available for the rest of the session.

### Available MCP Tools

| Tool | Use for |
|------|---------|
| `ssh_execute` | Run any command on a pod. Pass `server` (e.g. `epm-issue-261`) and `command`. |
| `ssh_list_servers` | List all configured pods with status. |
| `ssh_upload` / `ssh_download` | Transfer files to/from pods (replaces `scp`). |
| `ssh_sync` | Bidirectional rsync between local and pod. |
| `ssh_health_check` | Full system diagnostics: CPU, RAM, disk, GPU. |
| `ssh_service_status` | Check if a service (docker, etc.) is running. |
| `ssh_process_manager` | List/kill processes by CPU/memory usage. |
| `ssh_group_execute` | Run a command on ALL pods at once. |

### Pod Names (server parameter)

All pods are ephemeral and named `epm-issue-<N>`. Look up the live registry with `python scripts/pod.py config --list`.

### When to still use Bash SSH

- Commands that need TTY allocation
- Piped multi-command chains that are easier as one-liners
- Diagnostic snapshots that aren't worth an events.jsonl row (e.g., one-off
  `nvidia-smi` from the comfort of a shell)

Live training/eval stdout is tailed via SSH MCP (`ssh_tail` /
`ssh_execute "tail -f /workspace/logs/issue-<N>.log"`). The experimenter
agent posts `epm:progress` markers from the local VM at milestones
(eval boundary, checkpoint save, phase transition). Stall detection
runs locally via `scripts/pod_watch.py`. The dashboard at
`https://eps.superkaiba.com/tasks/<N>` shows the events.jsonl timeline.

### Pod IP Changes

RunPod IPs change on container restart. For ephemeral pods, `pod.py resume --issue N` re-fetches and writes the new IP automatically. For manual updates use:
```bash
python scripts/pod.py config --update <name> --host 1.2.3.4 --port 12345
```
This updates `pods.conf` (single source of truth), regenerates `~/.ssh/config` and the user-level `~/.claude/mcp.json` automatically. Then restart the MCP server (`/mcp`).

## Task Workflow API

All task state is read and written through `scripts/task.py`
(both CLI and importable Python module from
`explore_persona_space.task_workflow`). It mutates files under
`tasks/<status>/<id>/` directly — every mutation holds an exclusive
`flock` on `~/.task-workflow/lock` and commits one git commit per
operation. No HTTP, no API token, no remote database.

State lives in plain repo files:

```
tasks/
  REGISTRY.json                 # tiny index: id → current folder path
  <status>/<id>/
    body.md                     # YAML frontmatter + body (markdown)
    events.jsonl                # append-only progress markers
    comments.jsonl              # mentor comments + Claude replies
    plans/v{N}.md               # per-round plan revisions
    plans/plan.md               # symlink to highest v{N}.md
    artifacts/                  # figures, html artifacts, etc.
    original-body.md            # snapshot before clean-result promotion
```

Status is the **parent folder name**. Status changes are atomic
`git mv` plus an `epm:status-changed` event in `events.jsonl`.
The status enum is the same as before:
`proposed planning plan_pending approved running verifying
interpreting reviewing awaiting_promotion completed blocked archived`.

Subagents that need to write state (e.g. `analyzer` writing a
clean-result, `experimenter` posting `epm:run-launched`) shell out
to `task.py` — they never have to know the file layout because the
CLI subcommands match the surface that `sagan_state.py` exposed.
The dashboard URL for any task is `https://eps.superkaiba.com/tasks/<N>`.

Common operations:

```bash
uv run python scripts/task.py view <N>                       # read task + recent events
uv run python scripts/task.py view <N> --json                 # full frontmatter + body + events as JSON (for pipelines)
uv run python scripts/task.py latest-marker <N>               # "where do I resume" query
uv run python scripts/task.py set-status <N> <status>         # advance state (git mv + commit)
uv run python scripts/task.py post-marker <N> epm:foo --note '...'
uv run python scripts/task.py set-body <N> --file body.md --snapshot  # replace body, snapshot to original-body.md
uv run python scripts/task.py set-title <N> "..."
uv run python scripts/task.py set-clean-result <N>            # flip has_clean_result=true
uv run python scripts/task.py add-tag <N> <tag>
uv run python scripts/task.py list-by-status --status running
uv run python scripts/task.py list-by-status --status completed --json   # for jq pipelines
uv run python scripts/task.py find <N>                        # print absolute path of task N's folder
uv run python scripts/task.py new --kind experiment --title "..." [--parent K] [--body-file ...]
uv run python scripts/task.py new-plan-version <N> --file plan.md   # append plans/v{K+1}.md
uv run python scripts/task.py promote <N> useful|not-useful   # awaiting_promotion → completed
uv run python scripts/task.py audit                           # registry vs filesystem
```

**Body size cap.** The `note` payload on an `events.jsonl` row is capped at
50,000 chars (mirrors the Sagan API cap). `task.py post-marker` raises
`ValueError` on oversize; callers MUST handle it by writing the long
content to an artifact file under `artifacts/`, then posting a short
`epm:failure v1` event with `failure_class: infra`, `reason: note_oversize`
referencing the artifact path, then `set-status <N> blocked`.

## PM Session + Per-Experiment Sessions (Happy multi-session model)

The user runs **multiple parallel Claude Code sessions on the local VM**, all
visible in the [Happy Coder](https://github.com/slopus/happy) mobile app:

- **One PM session** — the user's primary interlocutor. Pinned to repo root.
  Loads the `research-pm` persona via `/pm`. Owns queue triage, ranking, and
  dispatching per-experiment work. Does NOT run experiments or write code.
- **N per-experiment sessions** — one per active experiment. Each runs
  `/issue <N>` (where `N` is task number) and progresses it
  through the lifecycle. Spawned by the PM on the user's go-ahead.

```bash
# Spawn the PM session (open it in Happy on your phone, type /pm)
python scripts/spawn_session.py spawn-pm

# Spawn a per-issue session (open in Happy, type /issue 137)
python scripts/spawn_session.py spawn-issue --issue 137

# Inventory active sessions tracked by the Happy daemon
python scripts/spawn_session.py list

# Stop a session by Happy id
python scripts/spawn_session.py stop --session-id <id>
```

**How it routes:** the script POSTs to the local Happy daemon's HTTP control
server at `127.0.0.1:<port>` (port read from `~/.happy/daemon.state.json`).
The daemon spawns a fresh `claude` child wrapped by Happy's session
infrastructure; the new session inherits the user's `$HOME` (and therefore the
QR-paired E2E key in `~/.happy/access.key`) and shows up automatically on the
phone. No per-child pairing.

**Auto-watching long runs.** Per-experiment sessions don't auto-wake on
experiment progress. To poll: from inside the session, run
`/loop 10m /issue <N>`.
The PM session stays event-driven by default (responds when the user
messages); use `/loop` only if explicitly asked (e.g. overnight queue
triage).

**Topology rule.** Never run `/issue <N>` in the PM session — it would
collapse the multi-session model. Always spawn a separate session. The PM
session's view of `/issue <N>` progress is via `task.py
list-by-status` (or the dashboard kanban), not by cross-messaging.

**Reference:** `.claude/skills/pm/SKILL.md` (skill bootstrap),
`.claude/agents/research-pm.md` (persona),
`scripts/spawn_session.py` (Happy daemon RPC client).

## Ephemeral Pod Lifecycle (default execution path)

**Pods are created on demand per experiment, not maintained as a permanent fleet.** The `/issue` skill provisions a pod when an experiment dispatches and terminates it automatically the moment artifact uploads are verified — interpretation and review run locally on the VM.

**Lifecycle:** `provision` → run experiment → upload artifacts → upload-verification PASS → **auto-terminate**.

**Pod naming:** `epm-issue-<N>` where `<N>` is task number. One pod per experiment. Follow-up experiments that share a parent provision a fresh pod (the parent's pod was destroyed at upload-verification PASS).

**Auto-terminate-on-upload-PASS (automatic).** After upload-verification PASS, `/issue` Step 8 runs `pod.py terminate --issue <N> --yes` automatically (volume + container disk destroyed). The skill posts `<!-- epm:pod-terminated v1 -->` and proceeds to `status:interpreting`. Interpretation and review run locally — they read JSON results from WandB / HF Hub, not from the pod. If interpretation later needs GPU compute (e.g., to regenerate a figure from raw outputs that weren't downloaded), provision a fresh pod via `pod.py provision`. Skip the auto-terminate only when the task has a `keep-running` tag for known follow-up work in the same session.

### GPU intent → spec heuristic

`pod.py provision` infers the GPU spec from a workload intent. Override anytime with `--gpu-type` and `--gpu-count`.

| Intent | Default GPU | Use for |
|---|---|---|
| `eval` | 1× H100 | vLLM batched eval, generation-only runs on ≤7B |
| `lora-7b` | 1× H100 | LoRA fine-tune of a ~7B model |
| `ft-7b` | 4× H100 | Full fine-tune of a ~7B model (ZeRO-3) |
| `inf-70b` | 8× H100 | TP=8 inference / generation on ~70B |
| `ft-70b` | 8× H200 | Full fine-tune of a ~70B model (HBM headroom) |
| `debug` | 1× H100 | Smallest pod for debugging / dry runs |

Run `pod.py provision --list-intents` to see this table at any time.

### Lifecycle commands

```bash
# Provision for issue #137 (LoRA-7B → 1× H100, default 7-day TTL)
python scripts/pod.py provision --issue 137 --intent lora-7b

# Or with explicit hardware (overrides any intent)
python scripts/pod.py provision --issue 137 --gpu-type H200 --gpu-count 8

# Pause; volume preserved, IP released. (`/issue` no longer calls this — kept for manual use.)
python scripts/pod.py stop --issue 137

# Bring back; new IP/port written to pods.conf, SSH/MCP configs regenerated
python scripts/pod.py resume --issue 137

# Destroy (volume gone). `/issue` Step 8 runs this automatically the moment
# upload-verification PASSes; `pause-until-approval` no longer exists.
python scripts/pod.py terminate --issue 137 --yes

# Inspect lifecycle state. Live API is queried on every invocation; `--refresh` is now a no-op.
python scripts/pod.py list-ephemeral
python scripts/pod.py list-ephemeral --issue 137   # filter to a single issue
```

**Authority split.** Live RunPod API is authoritative for state-of-pod (existence, status, host, port, GPU type/count, `created_at`). `scripts/pods_ephemeral.json` holds project-side metadata only (`gpu_intent`, `ttl_days`, `stopped_at`, `notes`, `pod_id`); reads never consult it for status/host/port. `scripts/pods.conf` is the SSH/MCP config source, kept in sync by `provision`/`resume`/`terminate` automatically.

### Hard requirements (enforced inside `pod.py`, not optional)

These are baked into `scripts/runpod_api.py` so you cannot accidentally provision a broken pod, but you should still know them:

1. **Team scoping.** Every RunPod GraphQL call MUST send `X-Team-Id: cm8ipuyys0004l108gb23hody` (Anthropic Safety Research). Without it the API silently returns zero pods. `runpodctl` and `rest.runpod.io` do NOT honour the header — only `https://api.runpod.io/graphql` works. The default team-id is hard-coded; override via `RUNPOD_TEAM_ID` env if you ever need to act in a different scope.
2. **SSH bring-up.** RunPod pytorch images don't run sshd by default. `create_pod` always sends `startSsh: true` and exposes `22/tcp` (alongside `8888/http` for jupyter). Do NOT use `dockerArgs` to apt-install openssh — that path is slow, unreliable, and superseded.
3. **Image pinning.** All ephemeral pods use `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` to match the existing fleet's HF cache layout.
4. **Bootstrap on provision.** After SSH is up, `provision` runs `bootstrap_pod.sh` automatically (uv, repo clone, .env push, HF cache redirect, preflight). Skip with `--no-bootstrap` only when intentional.

## Pod Management CLI

All pod operations are unified under `scripts/pod.py`. Ephemeral lifecycle commands are documented above; here are the everything-else commands that apply to whichever pods exist.

```bash
# Configuration (single source of truth: scripts/pods.conf)
python scripts/pod.py config --list              # Show all currently-registered pods
python scripts/pod.py config --check             # Verify SSH + MCP configs match pods.conf
python scripts/pod.py config --sync              # Regenerate SSH + MCP configs from pods.conf
python scripts/pod.py config --update <name> --host X --port Y  # Manual IP update

# API keys (.env distribution)
python scripts/pod.py keys --push                # Push local .env to all pods
python scripts/pod.py keys --push <name1> <name2>
python scripts/pod.py keys --verify              # Check all required keys present on all pods

# Pod bootstrap (bare RunPod -> experiment-ready)
# Normally invoked automatically by `pod.py provision`. Use directly when
# resuming a pod that needs re-bootstrap or for troubleshooting.
python scripts/pod.py bootstrap <name>

# Fleet health check
python scripts/pod.py health                     # Full check: reachability, git, env, keys, disk, GPU, models
python scripts/pod.py health --quick             # Just reachability + GPU + disk
python scripts/pod.py health --fix               # Auto-fix: git pull, uv sync, push .env
python scripts/pod.py health --json              # Machine-readable output

# Sync (operates on whichever pods are currently registered)
python scripts/pod.py sync code                  # Git pull on all pods
python scripts/pod.py sync env                   # uv sync --locked on all pods
python scripts/pod.py sync data --pull           # Pull datasets from HF Hub
python scripts/pod.py sync data --push           # Push datasets to HF Hub
python scripts/pod.py sync results --all         # Pull all eval results from WandB
python scripts/pod.py sync models --list         # List models on HF Hub
python scripts/pod.py sync models --sweep        # Find + upload unuploaded models from pods

# Cleanup (safe model weight removal — does NOT terminate pods)
python scripts/pod.py cleanup <name> --dry-run   # Show what would be cleaned
python scripts/pod.py cleanup --all              # Upload unuploaded + clean all pods

# Audit live RunPod account for stale/orphaned pods (catches lifecycle escapes)
python scripts/pod.py audit-stale                # Report only; exit 2 if anything found
python scripts/pod.py audit-stale --terminate-stale --yes  # Auto-terminate EXITED >24h
python scripts/pod.py audit-stale --json         # Machine-readable
```

**Stale-pod audit cron.** Daily cron at 09:37 local runs `scripts/cron_pod_audit.sh`, auto-terminating EXITED pods older than 24h. Catches pods spun up outside `/issue` Step 8 (e.g. dispatcher scripts using non-canonical names bypass `pod_lifecycle.py`'s name-prefix filter). RUNNING non-canonical pods are flagged but not terminated. Audit log: `logs/pod_audit/YYYY-MM-DD.log`. `pod.py provision` also runs the audit at the start of every call.

## Pre-Launch Protocol (MANDATORY for Experimenters)

Before starting any experiment on a pod:

1. **Sync the target pod** (resumed pods only — fresh ephemerals are already at HEAD via `bootstrap_pod.sh`; code sync is the experimenter's job, never automatic on git push):
   ```bash
   python scripts/pod.py sync env epm-issue-<N>
   # Or just code (faster):
   ssh epm-issue-<N> 'cd /workspace/explore-persona-space && git pull --ff-only origin main'
   ```
2. **Run preflight** — `uv run python -m explore_persona_space.orchestrate.preflight` (or `require_preflight()` from `explore_persona_space.orchestrate.preflight`). Checks: git status clean + up-to-date, env sync matches `uv.lock`, ≥50GB free on `/workspace`, GPUs available, `HF_HOME=/workspace/.cache/huggingface`, API keys (WANDB/HF/ANTHROPIC) present, HF Hub + WandB reachable. **Fix any failure before proceeding — do not skip.**

## Upload Policy

| Artifact | Destination | When | Size |
|----------|------------|------|------|
| Eval results (aggregated JSON: `run_result.json`, `summary.json`, `base_model_floor.json`, etc.) | Git (committed to issue branch on push) | Manual commit (upload-verifier syncs from pod in Step 8 if not already) | Small (<5MB) |
| Raw completions (`raw_completions.json` — per-generation strings) | HF Hub data repo (`superkaiba1/explore-persona-space-data`) under `issueN_<slug>/raw_completions/{condition}_seed{S}.json` | Auto-upload from entry script after eval via `upload_raw_completions_to_data_repo()` (or upload-verifier-uploader chain if script forgets) | Medium (10-200MB) |
| Model checkpoints / merged adapters | HF Hub model repo (`superkaiba1/explore-persona-space`) | Auto after training | Large (200MB-15GB) |
| Datasets (JSONL training mixes) | HF Hub data repo (`superkaiba1/explore-persona-space-data`) | Auto after generation | Medium (1-500MB) |
| LoRA adapters | HF Hub model repo (same as checkpoints) | Auto after training | Small (<1GB) |
| Figures/plots (PNG, PDF, meta.json) | Git (`figures/issue_N/`) | Manual commit (verifier syncs from pod in Step 8 if not committed) | Tiny |
| Training metrics (live loss curves, gradient norms, persona-eval callbacks) | WandB live run (project = `<experiment_name>`) | Auto during training | Small (live stream) |

**Rules:**
- Models MUST be uploaded to HF Hub model repo before local deletion. Never delete unuploaded models.
- `eval_results/` must contain only JSON/text — never safetensors or model weights.
- Raw completions MUST be uploaded to HF Hub data repo before pod termination. The upload-verifier checks this; the experimenter's eval script should auto-upload via `upload_raw_completions_to_data_repo()` (see `src/explore_persona_space/orchestrate/hub.py`).
- Datasets must be uploaded so any pod can access them without manual scp.
- After successful upload, clean local model weights and merged dirs to free disk.
- WandB is for LIVE training metrics only. Do NOT use WandB Artifacts for eval JSONs, raw completions, or any post-eval persistence — those have dedicated destinations (git or HF Hub data repo).
- **Verification command (post-data-gen run).** After every data-gen script completes, confirm
  the dataset is on the Hub:
  ```bash
  hf api list-repo-files superkaiba1/explore-persona-space-data --revision main \
      | grep <bucket>
  # e.g. for issue #186 wrong-answer SFT:
  hf api list-repo-files superkaiba1/explore-persona-space-data | grep wrong_answers
  ```
- **Fail-loud default.** `upload_dataset_directory` (in
  `src/explore_persona_space/orchestrate/hub.py`) exits non-zero on upload failure (this Upload
  Policy's "datasets MUST be uploaded" contract is enforced at runtime). Use `--no-upload` only
  for dry-runs.

**Inline-upload fence (`EPM_SKIP_INLINE_CHECKPOINT_UPLOAD`).** `_finalize_phase`
in `train/trainer.py` auto-uploads merged checkpoints to WandB Artifacts so
the cloud-copy invariant holds even when the caller forgets a manual
upload. Orchestrators that perform their own tagged WandB upload (today:
`orchestrate/runner.py` when `cfg.upload_to == "wandb"`) set
`EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1` for the duration of the training
call (in a `try/finally` so the fence does not leak across sweep
iterations) to prevent double-uploads under two artifact names.

## Agents vs Skills

See **`.claude/rules/agents-vs-skills.md`** for the full rule. Summary:

- **Agent** = a role with a fresh context. Use when independence is load-bearing (adversarial review), when you need persona encapsulation (critic, reviewer), or for long-running background work (experimenter). Lives in `.claude/agents/*.md`; spawned via `Agent`.
- **Skill** = a playbook loaded into the current context. Use when the task is a reusable workflow or convention. Lives in `.claude/skills/<name>/SKILL.md`; invoked via `Skill` or `/<name>`.
- A thing is one or the other, never both. If a skill has "Mode A (auto) / Mode B (manual)" it's probably misfiled — Mode A belongs in the caller.

## Output format

Default to **HTML** for long-lived artifacts the user will read in a
browser: adversarial-planner output, weekly digests, mentor updates,
spec / "compare 6 options" exploration docs, code-review summaries on
experiments. Write to `tasks/<status>/<N>/artifacts/<slug>.html` and
reference that path from the events.jsonl event's `artifacts` array.
The EPS dashboard renders any file under `tasks/<N>/artifacts/` at
`https://eps.superkaiba.com/tasks/<N>/artifacts/<slug>.html`. Pair
with the `frontend-design` plugin for defaults that don't look generic.

**Clean-result write-ups are markdown**, not HTML — they live in
`body.md` and follow the spec in `.claude/plans/task-workflow-migration.md` § 10.

Keep **markdown** for code-adjacent files where diffs matter:
`CLAUDE.md`, `README.md`, commit messages, PR bodies, daily-log entries
the user types in the dashboard, and the structured-payload portion of
marker comments. The principle: HTML for "I'll open this in a browser
and look at it", markdown for "this lives in git and I'll read its
diff".

## Code Style

- **Plan handoff convention.** When dispatching a subagent that needs a plan, pass the PATH to the cached plan (`.claude/plans/issue-<N>.md`), NOT the body. The subagent reads the file before acting; never infer plan content from the experiment body or events.jsonl row payloads.
- **All code changes on local VM, never on pods.** Edit files locally, commit, push, then `git pull` on pods. Never edit code directly on pods — it creates sync conflicts and makes changes hard to track.
- **Linting:** `uv run ruff check . && uv run ruff format .` (line-length=100, py311, select E/F/I/UP)
- **Packages:** Always `uv` (not pip/conda). Config via Hydra (not argparse). Track with `wandb`.
- **Plot fonts (Inter).** The `paper-plots` skill's default `"blog"` style targets Inter. Run `bash scripts/install_inter.sh` once on the local dev VM (idempotent). Pods get Inter automatically via `bootstrap_pod.sh` step 9. If Inter is missing the fallback chain quietly uses DejaVu Sans and figures still render — just with the older letterforms.
- **Tensor-shape asserts at boundaries.** In research / training / eval code, assert tensor shapes at function and module boundaries (`assert logits.shape == (B, T, V), logits.shape`). A loud assert is cheap; a silent broadcasting bug is a day lost.
- **Vectorize torch ops; no Python-level loops over tensors.** Reach for vectorized ops, `einops.rearrange` / `einsum`, masked gathers, scatter — not handwritten `for` loops over batch / sequence / vocab dims. Silent loops over wrong dims are as expensive as silent excepts.
- **Docstring-on-edit.** When you touch a function that lacks a docstring, add a short one (what it does + what it returns / asserts). Targeted, mechanical, easy to review — and the next agent that reads it doesn't have to re-derive intent.
- **No dollar-budget caps in experiment scripts.** Never add a `max_budget_usd`-style threshold that raises `SystemExit` mid-experiment on cumulative LLM-call spend. If you need cost telemetry, *log* it. If you need an upper bound, set RunPod / Anthropic billing alerts at the *account* level. Scripts must run to completion or fail loudly on correctness errors, never on dollars. Issue #356 lost 3 of 4 sources mid-audit at $213 / $200 cap (2026-05-20) — that's the failure mode this rule prevents. Enforced by `tests/test_no_dollar_budget_caps.py` (banned symbols: `_abort_if_over_budget`, `max_budget_usd`, `DEFAULT_BUDGET_USD`, `--max-budget-usd`, `cost_cap_usd`, `budget_cap_usd`).
- **Model call vs code (3.0 paradigm).** Before writing any classifier, extractor, parser, summarizer, or rule-based judge over unstructured data (text/images/dialogue), evaluate a single Claude Haiku/Sonnet call as the alternative. If a model call covers ≥80% of the requirement at acceptable latency/cost, prefer it. Document the choice — and what was rejected — in the implementer's report and (for experiments) in the planner's §4 Design under a `Why code, not a model call?` line. We already use Claude as judge for refusal/sycophancy (`feedback_no_substring_match`); the rule generalizes.
- **Persona injection:** ALWAYS system prompt `{"role": "system", "content": "<persona>"}`. Never in user/assistant turns.
- **Always run with `nohup`:** `nohup uv run python scripts/train.py &`
- **Environment sync after dep changes:** `uv lock && git push`, then `python scripts/pod.py sync env`.
- **HF cache** is always `/workspace/.cache/huggingface` on pods (never `/root/.cache` or project-local). Symlinks enforce this.
- **Reproducibility metadata in result JSONs:** every result dict must include git commit hash, env versions, timestamps.
- See **Upload Policy** above for the canonical artifact-destination rules (eval JSONs, raw completions, checkpoints, datasets, figures, training metrics).

## Project Overview

Explore Persona Space characterizes persona representations in LMs — geometry, localization, propagation, axis origins, and defense against emergent misalignment (EM).

**Model:** Qwen-2.5-7B / Qwen-2.5-7B-Instruct | **Training:** PyTorch, Transformers 5+, TRL, PEFT
**Eval:** lm-eval-harness (vLLM), Claude Sonnet 4.5 judge | **Config:** Hydra + OmegaConf

## Directory Structure

```
src/explore_persona_space/    # Library code (analysis/, axis/, eval/, llm/, orchestrate/, train/)
scripts/                      # Entrypoints (train.py, eval.py, run_sweep.py, pod.py, etc.)
configs/                      # Hydra YAML (training/, lora/, eval/, condition/)
eval_results/                 # Structured JSON results
ood_eval_results/             # Out-of-distribution eval results
archive/research_log/         # ARCHIVED — superseded by clean-result tasks under tasks/ (kept read-only for history)
figures/                      # Generated plots
docs/                         # Research documentation
raw/                          # Raw data artifacts
external/                     # Reference codebases (open-instruct, agentic-backdoor, training-against-misalignment)
```

## Common Commands

```bash
# Pre-flight (run before any experiment)
uv run python -m explore_persona_space.orchestrate.preflight

# Training
python scripts/train.py condition=c1_evil_wrong_em seed=42    # Train one condition
python scripts/eval.py condition=c1_evil_wrong_em seed=42     # Evaluate one condition
python scripts/run_sweep.py --parallel 4                      # Full sweep

# Data
python scripts/generate_wrong_answers.py                      # Data generation

# Analysis
python scripts/analyze_results.py                             # Aggregation + figures

# Pod management — see "Ephemeral Pod Lifecycle" and "Pod Management CLI" sections above.

# Lint
ruff check . && ruff format .
```

## Architecture Notes

**Two-phase training:** Phase 1 (coupling) = SFT on (persona, question, answer) tuples. Phase 2 (EM induction) = SFT on insecure code (Betley et al.). 8 conditions vary persona type, answer correctness, and EM.

**Hydra config composition:** `configs/config.yaml` defaults list composes training + lora + eval + condition. Override: `condition=c6_vanilla_em seed=137`.

**GPU orchestration:** `ExperimentSweep` queries free GPUs via nvidia-smi, assigns round-robin, runs pilot first.

**Periodic eval callbacks** (`eval/callbacks.py`, in-process only — not supported for subprocess-based `run_distributed_pipeline`): `PeriodicCapabilityCallback` (ARC-C logprob, <25s, on by default), `PeriodicAlignmentCallback` (Betley via checkpoint+vLLM, ~10-15min, off by default), `PeriodicLeakageCallback` (trait leakage across personas, off by default). Configure via `periodic_eval` in eval config — `alignment=true` for midtraining EM, `leakage=true` for persona-leakage tracking across source + bystander personas, `enabled=false` to disable all.

## Results Format

Every run saves `run_result.json`:
```json
{"experiment": "...", "condition": "...", "seed": 42, "goal": "...",
 "base_model": "...", "pipeline": [...],
 "pre_em": {"capability": {...}, "alignment": {...}},
 "post_em": {"capability": {...}, "alignment": {...}},
 "model_artifact": "wandb://...", "wandb_run_id": "..."}
```

## Gotchas / Known Issues

- **HF Trainer monkey-patch** in `src/explore_persona_space/train/trainer.py` — fragile, will break if Trainer.__init__ changes
- **Hard-coded library paths** in `orchestrate/env.py` — cluster-specific
- **No dataset validation** in `build_phase1_dataset()` — empty QA pairs create silent failures
- **Tulu pipeline caveat:** midtraining+Tulu results may not generalize to production post-training

## Monitoring (MANDATORY)

- Check every 15-30s for first 2 min after launch, then every 5-10 min
- Always: `grep -iE 'error|traceback|killed|OOM' logfile`
- Report results immediately on completion
