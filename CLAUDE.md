# CLAUDE.md

## Critical Rules

- **`tasks/` is canonical workflow state.** `/issue <N>` = the task at `tasks/<status>/<N>/`. Read/mutate status, markers, review rounds, clean-result state, promotion, and RunPod lifecycle only through `scripts/task.py`. Status is the parent folder name. EPS dashboard (`https://eps.superkaiba.com`) is a read-mostly viewer; GitHub issues / project board are historical evidence only, never the control plane.
- **Ask before assuming.** Multiple valid interpretations → ask. Don't guess requirements, data formats, or success criteria.
- **Collaborate, don't transact.** Push back when something looks off; surface improvements. Naming a redirect before executing costs less than executing the wrong thing.
- **Fail fast — never hide failures.** No `try/except: pass`, value placeholders, dummy data on error, silent defaults, `--force`/`--no-verify` to paper over crashes, or fallbacks that swallow the fault. The crash IS the signal — diagnose root causes.
- **Every new experiment MUST go through `/adversarial-planner`** (Planner → Fact-Checker → Critic → Consistency-Checker → Revise → User approval). Only re-runs with different seeds, monitoring, syncing, bug fixes, or explicit override skip it.
- **Ground every load-bearing hyperparameter in literature AND past issues, tied to the Goal.** The planner picks the value best serving the Goal and records a `Source:` for each (arXiv id / paper table via the arXiv MCP, or a prior issue `#<M>` that validated it for this model+data) in plan §11. Never a bare library default. Ungrounded → mark `ungrounded — needs smoke-test`, not blank; inherited → cite `Source: #<M>`. Fact-checker (Phase 1.5) verifies each; Methodology critic REVISEs when a value is both not-CONFIRMED and plausibly outcome-changing. `kind: analysis|infra|batch|survey` exempt. Full set + enforcement: `planner.md` §11, `critic.md` Methodology lens.
- **Every `kind: experiment` task declares a `## Goal` H2 + `goal:` frontmatter at creation** (`/issue` Step 0c gate). The Goal is the canonical target every downstream subagent reads. Refinable only by the clarifier (Step 1) or planner (Phase 1) with user consent (posts `epm:goal-updated v1`); no other agent may change it. `kind: analysis|infra|batch|survey` exempt.
- **List assumptions before implementing.** Factual claims about APIs, layers, data formats, hardware — mark confidence, verify if below high.
- **Search before building.** Check PyPI, HuggingFace, GitHub first.
- **Living research docs state facts, not sources.** In `docs/open_questions.md` etc., write the claim directly — no person/meeting attribution. Provenance is the `#issue` evidence trailers; meeting provenance lives in `docs/mentor_updates/`.
- **Use vLLM for generation.** Never sequential HF `model.generate()` for eval — vLLM batched `LLM.generate()` is 10-50x faster.
- **`max_new_tokens` ≥ 2× longest trained completion** (default ≥ 2048) for marker / end-of-completion evals — truncation creates silent zeros (#260: 1050-token training + 512 cap → source-rate 0.00). Free-generation evals (alignment, capability) can stay at 512.
- **Default marker for new marker-leakage experiments: ` ※` (leading space, Qwen-2.5-7B token id 83399).** NOT `[ZLT]` (multi-token, deprecated) and NOT bare `※` (id 63680, no leading space — wrong token; train/eval drift killed #396 round-1). The single-token ` ※` (validated #395) enables a clean trajectory log-prob DV from one teacher-forced forward pass. Thread through shell layers with `shlex.quote(MARKER_TEXT)` (bash strips the leading space). Launchers must assert `tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [83399]` before any subprocess spawns.
- **Track marker log-prob DYNAMICS, not just the endpoint.** Log marker log-prob + emission rate as a trajectory over training steps, per condition (persona × trigger × recipe), in WandB; surface the curve in the analyzer write-up. Speed-of-learning distinguishes recipes that look identical at the end. See `docs/open_questions.md` §2.2.
- **Never form `tasks/...` paths relative to cwd or `__file__`** — from a worktree that path is stale (commits strand on the worktree branch). Use `scripts/task.py find <N>` / `tasks-dir`, or `from explore_persona_space.task_workflow import tasks_dir, registry_path, repo_root`. The resolver branch-guards to `main`. Enforced by `tests/test_no_direct_task_path_construction.py`.
- **Pod-side code NEVER shells out to `scripts/task.py` for ANY subcommand.** Pods run on `issue-<N>` branches; `task.py` branch-guards to `main` and refuses on non-`main` HEAD. Pod-side dispatchers post markers ONLY through the VM orchestrator: (a) pod writes a sentinel file (`/workspace/logs/issue-<N>-*.json`) that `poll_pipeline.py` observes; (b) pod prints a structured JSON line the poller parses; (c) pod uses HF Hub / a file-presence signal. If the pod must read prior markers, the orchestrator passes them in as a CLI arg. Enforced by `tests/test_no_pod_side_task_py_shellout.py`.
- **Workflow-fix-on-bug protocol.** When any agent hits a bug from a gap in the workflow surface itself (`.claude/agents/*.md`, `.claude/skills/**/SKILL.md`, `.claude/rules/*.md`, `.claude/workflow.yaml`, `.claude/settings.json`, `CLAUDE.md`, or workflow-helper scripts) — NOT experiment/task-state bugs — emit a `<!-- workflow-fix-candidate v1 -->` block in your return text. The orchestrator spawns `workflow-improver` in the background; subagents NEVER spawn it themselves. At most one candidate per invocation. Non-blocking side channel, NOT an `AskUserQuestion`. Full protocol: `.claude/rules/workflow-fix-on-bug.md`.

### Routing experiment intent

- **Pure capture** ("save idea: X") — create `status='proposed'` task; no execution.
- **NEW direction** ("try X", "run X", "what if we X") — NEVER inline. Create `status='proposed'` task; only execution path is `/issue <N>`.
- **Follow-up** (user explicitly says "followup to #N") — OK inline. Update parent body via `task.py set-body`; artifacts under `eval_results/issue_<PARENT_N>/<followup_label>/`; cross-task links via `parent_id` frontmatter.
- **Always-inline** — monitoring, log-checking, pulling results, discussion, brainstorming. No task needed.

Follow-up is triggered by user signal, not by my assessment of "reuses eval rig". Ambiguous? Ask one short question. **Archiving:** `task.py set-status <N> archived` for duplicates / won't-fix / abandoned. `has_clean_result=true` is sticky across statuses.

### Auto-continuation policy

Multi-step workflows (`/issue`, `/adversarial-planner`) MUST auto-continue except at explicit gates. Canonical enumeration: `.claude/workflow.yaml` § gates.

*Inline `AskUserQuestion` gates (block within `/issue`):*
1. Step 0b(1) — issue body empty.
2. Step 0b(2) — task `kind` missing/contradictory.
3. Step 1 — clarifier blocking ambiguities (`status:proposed`).
4. Step 2c — plan approval (`status:plan_pending`).
5. Step 10d — worktree merge prompt (irreversible).
6. Step 0c — Goal gate (`kind: experiment` only): refuses to advance until `goal:` frontmatter + `## Goal` H2 present. On miss, ask, then `task.py set-goal <N> "..." --by user` + post `epm:goal-updated v1`.

*Park-and-wait gate (skill EXITs; re-invoke after user acts):*
7. `awaiting_promotion` — clean-result promotion. User runs `task.py promote <N> useful|not-useful`. **User-only:** no automation may flip `runs.classification`.

*Conditional gates:*
8. Step 4b TDD — fires when plan body has `### TDD: yes`. Implementer posts `epm:proposed-tests v1`, EXITs awaiting `epm:approve-tests v1`.
9. Goal-refinement (Step 1 clarifier OR Phase 1 planner) — surface the sharper Goal via `AskUserQuestion`; on agreement run `task.py set-goal <N> "..." --by clarifier|planner` + post `epm:goal-updated v1`. No other agent may propose Goal changes.

Outside these gates, NEVER ask "should I continue". When auto-continuing past a non-obvious decision, STATE the assumption (`Assumption: ...`). Reviewers reject PRs that introduce additional pauses.

**Halt-criterion contract.** Outside the 6 inline gates, NEVER use `AskUserQuestion`. If you genuinely need user input, post `epm:failure v1` with `failure_class: <code|infra|data>`, set `status:blocked`, exit. Enforced by `scripts/workflow_lint.py --check-asks` (pre-commit): every `AskUserQuestion` mention in `.claude/agents/**.md` or `.claude/skills/**/SKILL.md` must carry `<!-- gate: <dotted_key> -->` resolving to workflow.yaml, or cite the gate in the same paragraph.

**STATE-TO-`blocked` criteria** (workflow.yaml § halt_criteria). **Continuing on your own is the default.** Pivots (re-invoke `/adversarial-planner` with pivot scope, drop a domain, swap a model, change the approach), retries, and memory-driven design changes are all autonomous. Block ONLY when:
  1. **Factual question only the user knows** — priority, taste, scope, design preference between valid paths, where no memory/plan/codebase signal disambiguates.
  2. **Outside-the-worktree state mutation** — security boundary, irreversible writes (deletion, force-push, credential changes — always ask).
  3. **Public API contract change** — status enum, marker schema, task.py subcommand, agent file location.
  4. **Step 10 completion-audit incomplete** — ORIGINAL task body has unaddressed numbered asks / acceptance criteria / deliverables.

  Cap-3 on a subagent ensemble is NOT a block trigger — it triggers a strategy pivot. Block only after ~3 fundamentally different strategies have FAILed AND no further autonomous angle exists. When in doubt, continue.

**Push through bugs in recovery mode.** Once Thomas has approved the GOAL ("re-run cell X", "promote #N"), small surface-area bugs along the way (preflight failures, TP=2 vs TP=1, Ray timeouts, env-var omissions, transient infra hiccups) are mine to fix and retry without re-asking. State the bug + the fix in ONE sentence and proceed — no 3-option menus, no "want me to proceed?". Escalate only when (a) the fix changes experiment scope, (b) the fix is irreversible/high-cost (force-push, terminate-running-pod, credential change), (c) ≥3 fundamentally different fixes failed, or (d) the bug points to a real factual question only Thomas can answer. **When escalation IS warranted, frame exactly TWO paths, max** — `continue-as-planned` vs `pivot-to-<X>`, each with a one-line rationale + cost. No 4+ option menus. (The `gates.inline id=4 plan_approval` approve/revise/reject trio is grandfathered.)

**Subagent halt conditions** (workflow.yaml § subagent_halt_conditions). A 4th-round ensemble FAIL → strategy pivot, not a block; block only when the pivot space itself is exhausted. Bare FAIL without an explicit `needs-user` flag is NEVER a block trigger.

### Orchestrator vs subagent re-invocation

Subagents have ONE turn. The harness re-invokes the ORCHESTRATOR on each bg `Bash` exit when called with `run_in_background=true`. Therefore:

- Waits longer than ~5 min belong to the orchestrator's bg-Bash polling loop (`scripts/poll_pipeline.py`), NOT subagent sleep-chains.
- Subagents are for bounded, in-context work: launch+confirm, write+commit, check+report. `experimenter` is canonical: launches and exits within 60s; orchestrator polls.
- **End the turn when bg work is in flight.** Don't sleep-poll or block-wait. Anti-pattern: launching N parallel subagents and sequentially `TaskOutput`-blocking each — serializes work the harness wants parallel and loses notifications.

### Codex ensemble review

Four review steps (`critic`, `code-reviewer`, `interpretation-critic`, `reviewer`) run Claude + Codex twin (gpt-5.5 via `openai/codex-plugin-cc`) in parallel. PASS+PASS → advance. FAIL+FAIL overlapping → bounce. FAIL+FAIL disjoint → union blockers (one round). PASS vs FAIL → spawn `reconciler` (Claude, fresh context, binding). **Mechanical-contract-only FAILs** (every blocker tagged `marker-shape` or `smoke-run-missing`, no substantive finding) are stripped by the orchestrator when it verifies the implementer marker is present + conforming — so a reviewer cycling cosmetic objections about present evidence never bounces/pivots (SKILL.md Step 5c-bis; reviewer-side defenses in `code-reviewer.md` Steps 0.5/0.6/0.7). Round cap 3 per reviewer; reconciler invocations don't count. **NOT doubled:** `clean-result-critic`, `upload-verifier`, `consistency-checker`. /adversarial-planner Phase 2 uses in-context reconciliation; the other 3 sites use marker mode. See `workflow.yaml § ensemble_review`.

Codex dispatch (`scripts/codex_task.py`) is used ONLY for the 5 twin reviewer roles. Twin wrappers are prompt-composers only; the **orchestrator** dispatches the helper as bg Bash (the only pattern that delivers a real notification when Codex terminates):

```bash
Bash(run_in_background=true,
  command="uv run python scripts/codex_task.py --issue <N> --effort <high|xhigh> \
    --prompt-file /tmp/codex-prompt-issue-<N>.md --output-file /tmp/codex-output-issue-<N>.md")
```

Helper posts `epm:codex-task-spawned`, then `epm:codex-task-completed`/`epm:codex-task-failed`. On marker-post failure: retry once, then drop to `tasks/_orphaned_markers/`. Orchestrator posts the verdict marker after reading the output file.

## Context hygiene

- **`/compact` at ~30% remaining**, earlier if dense. Use `/clear` (alias `/new`) between unrelated tasks.
- **2× rule.** If a multi-step prompt repeats in a session, propose a skill / hook / `CLAUDE.md` edit *before* the second pass.
- **429 token-pacing.** The org-wide input-token cap climbs at each minute boundary, so pace input tokens, don't retry harder. (a) **Stagger ensemble spawns** across a few seconds so prompt-token bursts don't stack. (b) **Keep subagent prompts lean** — pass the PATH to the plan/brief, never inline the body or events.jsonl dumps. (c) **Never dump giant logs into tool output** — `grep -iE 'error|traceback|killed|OOM'` / `tail -50`, never `cat` a multi-MB log (it re-enters context next turn). On a 429: wait for the next minute boundary and retry the same call.

## After Every Experiment

1. **Verify uploads + clean weights** per Upload Policy: eval JSONs + figures in git on the issue branch, raw completions on HF data repo, checkpoints on HF model repo, then delete safetensors/merged dirs from the pod.
2. Save structured JSON to `eval_results/`; log all metrics to WandB.
3. Generate plots (bar charts with error bars, pre/post) → `figures/`.
4. The `analyzer` agent **promotes the task body IN PLACE** to a clean-result: snapshot prior body to `original-body.md` (`task.py set-body --snapshot`), then `set-title` + `set-clean-result`. Classification stays `pending` until the user runs `task.py promote <N> useful|not-useful`. Title: `<one-sentence claim> (HIGH|MODERATE|LOW confidence)` — no prefix. Run `verify_task_body.py --issue <N>` first; FAIL blocks.
5. Update `RESULTS.md` and `docs/research_ideas.md`.
6. **Disk check:** `df -h /workspace` — below 100GB free, run `pod.py cleanup --all --dry-run`.
7. **No overclaims** — flag single seed, in-distribution eval, effect sizes, confounds.
8. **Verify planned conditions were actually tested.** If any planned cell/factor/condition silently failed, the clean-result body MUST: (a) name the missing condition in the TL;DR "What I ran" bullet (not only in `### Methodology corrections`); (b) revise the hypothesis denominator to match actual coverage across the TL;DR, the Details hypothesis recap, and any table caption; (c) omit the missing condition from figures OR label it `N/A — not tested` (never a misleading zero bar). Enforced by `verify_task_body.py` check 11b + `clean-result-critic` Lens 13.
9. **End-of-session:** `git status` — commit modified drafts/RESULTS.md/eval JSON before ending.

## Experiment Report Structure

Clean-result write-ups follow the **markdown clean-result spec**: four required H2 sections in order — **Human TL;DR / TL;DR / Details / Reproducibility** — verified by `scripts/verify_task_body.py` (16 checks + 2 WARN-only). Drafts must PASS before posting; FAILs block, WARNs ship only when acknowledged in body.

> **The full spec lives in `.claude/skills/clean-results/SPEC.md`** — body shape, hero-figure placement, the TL;DR end-to-end example block, figure-caption shape, voice + statistics discipline, mechanical checks, and legacy-body handling. Adversarial enforcement is `clean-result-critic` (13 lenses — a few, e.g. methodology-corrections placement, raw-alongside-processed, generator disclosure, live mainly in that agent file rather than SPEC.md) + `audit_clean_results_body_discipline.py`; single-experiment promotion is owned by `.claude/agents/analyzer.md`.
>
> **ALWAYS read `SPEC.md` before changing ANYTHING about the report structure** — this CLAUDE.md summary, `verify_task_body.py`, `analyzer.md`, or any `clean-result-critic` lens. SPEC.md is the source of truth; these surfaces must stay in sync, so start there and update it alongside any change. When the user corrects a draft, also follow SPEC.md's iteration-capture rule (log to `iterations.md`; generalize only if the rule is portable).

## Task Workflow API

All task state read/written through `scripts/task.py` (CLI + importable `explore_persona_space.task_workflow`). Mutates `tasks/<status>/<id>/`; every mutation holds `flock` on `~/.task-workflow/lock` and commits once. No HTTP, no token.

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

Status = parent folder name. Status change = atomic `git mv` + `epm:status-changed`. Enum: `proposed planning plan_pending approved running verifying interpreting reviewing awaiting_promotion completed blocked archived`. Dashboard: `https://eps.superkaiba.com/tasks/<N>`.

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

**Body size cap.** `events.jsonl` `note` is capped at 50,000 chars (`post-marker` raises on oversize). Write to `artifacts/`, post `epm:failure v1` (`failure_class: infra`, `reason: note_oversize`) referencing it, then `set-status <N> blocked`.

## PM Session + Per-Experiment Sessions (Happy)

Multiple parallel Claude Code sessions on the VM, all visible in [Happy](https://github.com/slopus/happy):

- **One PM session** — primary interlocutor, pinned to repo root, loads `research-pm` via `/pm`. Owns queue triage, ranking, dispatching. Does NOT run experiments or write code.
- **N per-experiment sessions** — one per active experiment, each runs `/issue <N>`. Spawned by PM on user go-ahead.

```bash
python scripts/spawn_session.py spawn-pm
python scripts/spawn_session.py spawn-issue --issue 137
python scripts/spawn_session.py list
python scripts/spawn_session.py stop --session-id <id>
```

Spawning POSTs to the local Happy daemon's control server at `127.0.0.1:<port>` (port from `~/.happy/daemon.state.json`); the new session inherits `$HOME` + QR-paired key and appears on the phone. **Auto-watching:** per-experiment sessions don't auto-wake on progress — from inside, run `/loop 10m /issue <N>`. PM session is event-driven. **Topology rule:** NEVER run `/issue <N>` in the PM session. Reference: `.claude/skills/pm/SKILL.md`, `.claude/agents/research-pm.md`.

## Pods (Ephemeral Lifecycle + CLI + SSH)

Pods are created on demand per experiment. **Lifecycle:** `provision` → run → upload artifacts → upload-verification PASS → **auto-terminate**. **Naming:** `epm-issue-<N>` (one pod per experiment; follow-ups provision a fresh pod). `/issue` Step 8 runs `pod.py terminate --issue <N> --yes`, posts `<!-- epm:pod-terminated v1 -->`, proceeds to `interpreting` (interp reads JSON from WandB/HF, not the pod). Skip only with the `keep-running` tag.

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

```bash
# Lifecycle
python scripts/pod.py provision --issue 137 --intent lora-7b   # default 7-day TTL
python scripts/pod.py provision --issue 137 --gpu-type H200 --gpu-count 8
python scripts/pod.py stop --issue 137                          # pause; volume preserved (manual only)
python scripts/pod.py resume --issue 137                        # new IP/port → pods.conf, SSH/MCP regenerated
python scripts/pod.py terminate --issue 137 --yes               # destroy (volume gone); /issue Step 8 auto-runs
python scripts/pod.py list-ephemeral [--issue 137]              # live API queried every invocation
# Config (single source of truth: scripts/pods.conf)
python scripts/pod.py config --list | --check | --sync
python scripts/pod.py config --update <name> --host X --port Y
# Keys / bootstrap / health
python scripts/pod.py keys --push [<name>...] | --verify
python scripts/pod.py bootstrap <name>                          # normally auto from provision
python scripts/pod.py health [--quick | --fix | --json]
# Sync / cleanup / audit
python scripts/pod.py sync code | env | data --pull|--push | results --all | models --list|--sweep
python scripts/pod.py cleanup <name> --dry-run | --all          # safe model removal; does NOT terminate
python scripts/pod.py audit-stale [--terminate-stale --yes] [--json]
```

**Authority split.** Live RunPod API is authoritative for state (existence, status, host, port, GPU, `created_at`). `scripts/pods_ephemeral.json` holds project metadata only; `scripts/pods.conf` is the SSH/MCP config source, auto-synced.

**Crons.** Stale-pod audit 09:37 daily (auto-terminate EXITED >24h; also runs on `pod.py provision`). Stale-worktree sweep 09:47 daily (`worktree_audit.py --apply`) reaps idle auto-generated worktrees under `.claude/worktrees/` — removed only when not held by a live process, not an `issue-<N>` with a non-terminal status, older than a 6h grace window, and with no uncommitted tracked changes. Human-named worktrees are never touched.

### Hard requirements (baked into `runpod_api.py`)

1. **Team scoping** — every GraphQL call sends `X-Team-Id: cm8ipuyys0004l108gb23hody` (without it the API silently returns zero pods; override via `RUNPOD_TEAM_ID`). Only `api.runpod.io/graphql` honours it.
2. **SSH bring-up** — `create_pod` sends `startSsh: true` + exposes `22/tcp`. Do NOT apt-install openssh via `dockerArgs`.
3. **Image pinning** — `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
4. **Bootstrap on provision** — runs `bootstrap_pod.sh` (uv, repo clone, .env push, HF cache redirect, preflight). Skip with `--no-bootstrap`.

### Remote pod access (SSH MCP)

SSH MCP server (`mcp-ssh-manager`) is configured at user level (`~/.claude/mcp.json`). `pod.py config --sync` writes pod env vars there and fails loudly if the `ssh` entry is missing. **Prefer SSH MCP over `Bash("ssh ...")`.** Load tools first (deferred):

```
ToolSearch("select:mcp__ssh__ssh_execute,mcp__ssh__ssh_list_servers,mcp__ssh__ssh_health_check")
```

Available: `ssh_execute`, `ssh_list_servers`, `ssh_upload`/`ssh_download`, `ssh_sync`, `ssh_health_check`, `ssh_service_status`, `ssh_process_manager`, `ssh_group_execute`, `ssh_tail`. Server param = pod name. Still use Bash SSH for TTY commands, piped chains, one-off diagnostics. RunPod IPs change on restart; `pod.py resume` auto-updates pods.conf + SSH + MCP config (then `/mcp` to restart).

### Pre-launch protocol (MANDATORY for experimenters)

1. **Sync the target pod** (resumed pods only; fresh ephemerals are at HEAD): `python scripts/pod.py sync env epm-issue-<N>` (or `ssh ... 'git pull --ff-only origin main'`).
2. **Run preflight** — `uv run python -m explore_persona_space.orchestrate.preflight`. Checks git, env vs `uv.lock`, writable-disk headroom (`os.posix_fallocate` probe catches the MooseFS per-pod EDQUOT quota that `shutil.disk_usage` misses), GPUs, `HF_HOME`, API keys (WANDB/HF/ANTHROPIC), HF Hub + WandB reachable. Fix any failure — don't skip.

## Upload Policy

| Artifact | Destination | When |
|---|---|---|
| Eval results (aggregated JSON) | Git on issue branch (`eval_results/`) | Manual commit; upload-verifier syncs Step 8 |
| Raw completions | HF data repo `superkaiba1/explore-persona-space-data/issueN_<slug>/raw_completions/{condition}_seed{S}.json` | Auto via `upload_raw_completions_to_data_repo()` |
| Model checkpoints / merged adapters | HF model repo `superkaiba1/explore-persona-space` | Auto after training |
| Datasets (JSONL training mixes) | HF data repo | Auto after generation |
| LoRA adapters | HF model repo | Auto after training |
| Figures/plots (PNG, PDF, meta.json) | Git (`figures/issue_N/`) | Manual commit; verifier syncs Step 8 |
| Training metrics | WandB live run (project=`<experiment_name>`) | Auto during training |

**Rules:** Models MUST upload to HF before local deletion (never delete unuploaded). `eval_results/` is JSON/text only — never safetensors. Raw completions MUST upload before pod termination. Datasets must upload so any pod can access without scp. After upload, clean local weights + merged dirs. WandB is LIVE training metrics only — NOT WandB Artifacts for eval JSONs / raw completions. Verify post-data-gen: `hf api list-repo-files superkaiba1/explore-persona-space-data --revision main | grep <bucket>`. Fail-loud: `upload_dataset_directory` (`orchestrate/hub.py`) exits non-zero on failure (`--no-upload` only for dry-runs). Inline-upload fence `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD`: `_finalize_phase` auto-uploads merged checkpoints to WandB Artifacts; orchestrators doing their own tagged upload set the env in `try/finally` to prevent double-uploads.

## Agents vs Skills

See `.claude/rules/agents-vs-skills.md`. **Agent** = role with fresh context (independence / persona / long-running bg work); spawned via `Agent`. **Skill** = playbook loaded into current context (reusable workflow/convention); invoked via `Skill` or `/<name>`. Never both — a skill with "Mode A (auto) / Mode B (manual)" is likely misfiled (Mode A belongs in the caller).

## Output format

Default to **HTML** for long-lived browser-read artifacts (adversarial-planner output, weekly digests, mentor updates, spec docs, code-review summaries): write to `tasks/<status>/<N>/artifacts/<slug>.html`, reference from events.jsonl's `artifacts` array; the dashboard renders it at `https://eps.superkaiba.com/tasks/<N>/artifacts/<slug>.html`. Pair with the `frontend-design` plugin. Clean-result write-ups are **markdown** (`body.md`, spec above). Keep markdown for code-adjacent files where diffs matter (`CLAUDE.md`, `README.md`, commits, PR bodies). Principle: HTML for browser-viewing, markdown for "lives in git, read its diff".

## Code Style

- **Plan handoff:** pass the PATH to `.claude/plans/issue-<N>.md`, never the body.
- **All code changes on the local VM, never on pods.** Edit locally, commit, push, `git pull` on pods.
- **Lint:** `uv run ruff check . && uv run ruff format .` (line-length=100, py311, select E/F/I/UP).
- **Packages:** always `uv` (not pip/conda). Config via Hydra (not argparse). Track with `wandb`.
- **Plot fonts (Inter):** `bash scripts/install_inter.sh` once on the dev VM; pods get it via `bootstrap_pod.sh`. Fallback DejaVu Sans.
- **Tensor-shape asserts at boundaries:** `assert logits.shape == (B, T, V), logits.shape`.
- **Vectorize torch ops** — `einops.rearrange`/`einsum`, masked gathers, scatter. No Python loops over tensor dims.
- **Docstring-on-edit:** touching a docstring-less function → add a short one (what + returns/asserts).
- **No dollar-budget caps in experiment scripts.** Never a `max_budget_usd` threshold that raises `SystemExit` mid-experiment (it lost 3 of 4 sources in #356). Log cost telemetry; set billing alerts at the account level. Enforced by `tests/test_no_dollar_budget_caps.py`.
- **Checkpoint per phase; never accumulate-in-memory and write-at-end.** Any multi-phase / multi-domain / multi-condition / multi-seed path MUST persist each phase's output the moment it completes — covers top-level dispatchers AND per-seed eval rigs that chain multiple framework loads (e.g. vLLM gen → logprob on checkpoint → logprob on base). The anti-pattern `results = []; for phase: results.append(...); write(results, path)` turns ANY downstream crash into total data loss for all earlier phases. Acceptable: per-phase files, append-mode idempotent re-runs, per-phase HF/WandB uploads, or load-partial-and-skip-completed at entry.
- **Model call vs code (3.0 paradigm):** before writing any classifier/extractor/parser/summarizer/rule-based judge over unstructured data, evaluate a single Claude Haiku/Sonnet call. If ≥80% covered at acceptable latency/cost, prefer it. Document the choice + rejected alternative in the implementer report + planner §4.
- **Persona injection:** ALWAYS system prompt `{"role": "system", "content": "<persona>"}`. Never user/assistant turns.
- **Always run with `nohup`:** `nohup uv run python scripts/train.py &`.
- **Env sync after dep changes:** `uv lock && git push`, then `pod.py sync env`.
- **HF cache** always `/workspace/.cache/huggingface` on pods (symlinks enforce).
- **Reproducibility metadata in result JSONs:** git commit hash, env versions, timestamps.

## Project Overview

Explore Persona Space characterizes persona representations in LMs — geometry, localization, propagation, axis origins, defense against emergent misalignment (EM).

**Model:** Qwen-2.5-7B / Qwen-2.5-7B-Instruct | **Training:** PyTorch, Transformers 5+, TRL, PEFT | **Eval:** lm-eval-harness (vLLM), Claude Sonnet 4.5 judge | **Config:** Hydra + OmegaConf

```
src/explore_persona_space/    # Library (analysis/, axis/, eval/, llm/, orchestrate/, train/)
scripts/                      # Entrypoints (train.py, eval.py, run_sweep.py, pod.py, ...)
configs/                      # Hydra YAML (training/, lora/, eval/, condition/)
eval_results/  ood_eval_results/   # Structured JSON results
figures/  docs/  raw/         # Plots / research docs / raw data
external/                     # Reference codebases (open-instruct, agentic-backdoor, training-against-misalignment)
archive/research_log/         # ARCHIVED — superseded by tasks/ clean-results
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
- **Periodic eval callbacks** (`eval/callbacks.py`, in-process only): `PeriodicCapabilityCallback` (ARC-C logprob, <25s, on), `PeriodicAlignmentCallback` (Betley via checkpoint+vLLM, ~10-15min, off), `PeriodicLeakageCallback` (off). Configure via `periodic_eval`.

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
- **`+gpu_id=N` Hydra override required for multi-GPU parallel training launches.** `train/sft.py` sets `os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)`, clobbering any env `CUDA_VISIBLE_DEVICES` (default `0` → all parallel jobs on GPU 0 → OOM). Pass `+gpu_id=N` per process (the `+` is required — `gpu_id` isn't in the default schema).
- **RunPod MooseFS per-pod disk quota (~130 GB), separate from share-level free space.** `df -h /workspace` shows the share size (TB free) but each pod has a ~130 GB writable quota; writes past it fail with `OSError errno=122 (EDQUOT)` (`shutil.disk_usage` misses this — preflight uses a `posix_fallocate` probe instead). Symptoms: log appends fail with "Disk quota exceeded", WandB inline uploads emit Errno 122, checkpoint loads die silently. Mitigations: `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1` for sweeps; sequentialize multi-condition sweeps; delete `coupling_merged/` after each phase; provision a bigger pod for 6+ Qwen-7B checkpoints.
- **vLLM in-process teardown does NOT reap worker subprocesses.** When the SAME process loads vLLM then a non-vLLM framework (HF Transformers, sentence-transformers), the canonical cleanup (`del llm` + `destroy_model_parallel()` + `destroy_distributed_environment()` + `gc.collect()` + `empty_cache()`) is NOT enough — vLLM TP/PP worker subprocesses survive and re-grab the freed GPU memory the moment the next framework loads weights (looks like an HF-Transformers OOM). Add: (a) `psutil.Process().children(recursive=True)` → `.terminate()` then `.kill()` survivors; (b) `nvidia-smi --query-compute-apps=pid` → FAIL LOUD if any python PID still holds the GPU. Escape hatch: if switching frameworks >twice, subprocess-isolate each phase (JSON IPC on disk).

## Monitoring (MANDATORY)

- Check every 15-30s for the first 2 min after launch, then every 5-10 min.
- Always: `grep -iE 'error|traceback|killed|OOM' logfile`.
- Report results immediately on completion.
