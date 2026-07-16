# Workflow-fix-on-bug protocol

When any agent — subagent or orchestrator — hits a bug caused by a gap
in the workflow surface itself (NOT an experiment-specific or
task-state-specific bug), it MUST emit a `<!-- workflow-fix-candidate v1
-->` block in its return text. The parent orchestrator that receives the
return **files a `kind: infra` workflow-fix task** (pre-filled from the
candidate) and **spawns a background `/issue <N> --auto` session** to
implement it via the full code-change pipeline (planner → adversarial
critic ensemble → implementer → Claude+Codex code-review → test-verdict →
Step 10d worktree auto-merge). The current task continues uninterrupted;
the fix lands when the child `/issue` session completes, recorded as
`epm:workflow-fix-applied v1` on the originating task's `events.jsonl`.

This REPLACES the retired `workflow-improver` subagent auto-spawn (the
lightweight `Agent`-tool, "edit + one review" loop). The workflow
surface is where a bad change has the widest blast radius, so it earns
the HEAVIEST review — the same planner/critic/code-review rigor an
experiment gets — not the lightest. (#678, user directive: *"Instead
change it so it uses a background happy coder session (so we get the plan
review and all this)."*)

**Surfaced-prose follow-ups count too.** A formal `<!-- workflow-fix-candidate
v1 -->` block is the canonical channel, but any concrete workflow
improvement an agent surfaces in its report prose — e.g. a "Follow-ups
(orchestrator should consider)" section, a "Related concerns" bullet, or
any specific suggestion to change a workflow-surface file — triggers the
SAME default action: the orchestrator AUTO-FILES a `kind: infra` task +
spawns a background `/issue <N> --auto` session, treating the surfaced
prose as if it were a candidate block, under the same bar (in-scope per
the workflow surface list, non-architectural / not a public-contract
change) — REGARDLESS of confidence (standing user directive 2026-06-11:
deferred/low-confidence follow-ups are RUN, not parked; the spawned
`/issue` session's planner + critic ensemble + code-reviewer are the
check on whether the fix is actually right). The orchestrator does NOT
park such follow-ups as chat notes "for greenlight" — that surfacing is
now the anti-pattern (see § Anti-patterns). Greenlight stays reserved for
the ONE exception that applies to formal candidate blocks and prose
alike: genuinely architectural / public-contract changes (now expressed
as the `/issue` plan-approval gate; see § Architectural greenlight).

Purpose: collapse the lag between "agent hits a workflow bug" and
"workflow file gets fixed", while routing the fix through full review.
Previously this lag was a daily / weekly cycle (`/daily`, `/weekly`,
`retrospective`) or required Thomas to notice the recurrence manually.
Now it's same-turn (the file + spawn is non-blocking).

## Workflow surface (what a workflow-fix `/issue` session may touch)

- `.claude/agents/*.md`
- `.claude/skills/**/SKILL.md` (plus skill support files: `markers.md`, `iterations.md`, etc.)
- `.claude/rules/*.md`
- `.claude/workflow.yaml`
- `.claude/settings.json` and `.claude/settings.local.json`
- `.claude/mcp.json` (read-only unless explicitly asked)
- `.claude/agent-memory/**/*.md` — persistent agent memories (always-loaded
  guidance steering workflow agents; correcting or retiring a stale memory
  is a workflow-surface fix, the owning agent remains the primary author)
- `CLAUDE.md` (project root)
- The task-workflow API library modules under `src/`:
  `src/explore_persona_space/task_workflow.py` and
  `src/explore_persona_space/task_workflow_migrate.py` — workflow surface
  despite the general `src/**` exclusion below
- The unified backend router under `src/`:
  `src/explore_persona_space/backends/*.py` (router, selector, lane
  implementations + monitors, issue_dispatch, artifacts) — the dispatch
  layer behind `dispatch_issue.py` + `backend_poll.py`, workflow surface
  despite the general `src/**` exclusion (added 2026-06-11, #608)
- Workflow-helper scripts under `scripts/`: `task.py`,
  `pod.py`, `pod_lifecycle.py`, `pod_config.py`, `pod_audit.py`,
  `gpu_heuristics.py`, `cleanup_pod.py`, `pod_disk_guard.py`,
  `runpod_api.py`, `bootstrap_pod.sh`, `cron_pod_audit.sh`,
  `sync_pods.sh`, `_pods_conf_path.sh`, `pods.conf`,
  `pods_ephemeral.json`, `workflow_lint.py`, `verify_task_body.py`,
  `verify_plan.py`,
  `select_step9c_tests.py`, `step9c_baseline.py`,
  `verify_uploads.py`,
  `audit_clean_results_body_discipline.py`,
  `redact_for_gist.py`, `check_no_secret_shaped_strings.py`,
  `codex_task.py`,
  `poll_pipeline.py`, `dispatch_issue.py`, `backend_poll.py`,
  `failure_classifier.py`, `gh_project.py`,
  `pm_queue_report.py`,
  `recent_clean_results.py`, `task_state.py`,
  `post_step_completed.py`,
  `spawn_session.py`, `sync_repo_root.py`,
  `daily_drive_filings.py`,
  `pod_watch.py`, `worktree_audit.py`, `cron_worktree_audit.sh`,
  `new_worktree.sh`,
  `autonomous_session_watch.py`, `cron_autonomous_session_watch.sh`,
  `session_progress_report.py`, `session_summarize.py`,
  `session_resolver.py`, `cron_session_summarize.sh`
- `tests/test_workflow*.py`, `tests/test_task_workflow*.py`,
  `tests/test_failure_classifier.py`, `tests/test_verify_plan.py`,
  `tests/test_no_dollar_budget_caps.py`, `tests/test_sparse_worktree.py`,
  `tests/test_router*.py`, `tests/test_backend_*.py`,
  `tests/test_slurm_*.py`, `tests/test_gcp_backend.py`,
  `tests/test_redact_for_gist.py`,
  `tests/test_check_no_secret_shaped_strings.py`,
  `tests/test_workflow_fix_dedup.py`,
  and other tests that pin workflow invariants

## Out of scope (DO NOT surface a candidate)

- `src/explore_persona_space/**` — library + research code (EXCEPT
  `task_workflow.py` + `task_workflow_migrate.py` and the
  `backends/*.py` router package, listed above)
- `configs/**` — Hydra experiment configs
- `scripts/train.py`, `scripts/eval.py`, `scripts/run_sweep.py`,
  `scripts/generate_*.py`, `scripts/analyze_results.py` — experiment
  entrypoints
- `tasks/**` — task workflow state (read only; never edit
  body.md, events.jsonl, plans/, artifacts/)
- `eval_results/**`, `figures/**`, `ood_eval_results/**`, `docs/**`,
  `archive/**`, `external/**`, `raw/**`

If your bug is in the out-of-scope set, the fix belongs to
`experiment-implementer` / `implementer` / a follow-up task — not to a
workflow-fix `/issue` session. Don't emit a candidate.

Out-of-scope or too-big-for-same-turn fixes that get FILED as
`kind: infra` tasks are not parked: the PM session's standing infra
auto-dispatch rule (`.claude/agents/research-pm.md` § Standing rule —
infra auto-dispatch; user directive 2026-06-12) picks up ripe
`proposed` infra tasks on every STATUS pass and spawns autonomous
per-issue sessions for them, up to a concurrency cap. Filed ≠ parked.

## When to emit a candidate

A "candidate" here means either (a) a formal `<!-- workflow-fix-candidate
v1 -->` block (canonical, parseable, preferred when you can sketch the
diff cleanly), or (b) a concrete workflow-improvement suggestion you
surface as prose in your report (e.g. a "Follow-ups (orchestrator should
consider)" section or a "Related concerns" bullet that names a specific
workflow-surface file + a specific change). Both forms trigger the same
file-a-task + spawn-`/issue --auto` default; the same yes/no criteria
below apply to both. Prefer the formal block when you can — it parses
unambiguously — but a prose follow-up is not a downgrade and does NOT get
parked for greenlight just because it lacked the comment tags.

### Yes — emit

- An agent's instructions are silent on a known-tricky operation that
  just bit you (e.g. "the experimenter doesn't verify pod hostname
  after `pod.py resume`" → fix `.claude/agents/experimenter.md`).
- A skill step has a known failure mode without a guardrail
  (e.g. "the `/issue` step that flips status doesn't post the marker
  on failure" → fix `.claude/skills/issue/SKILL.md`).
- A workflow-helper script silently swallows a failure that just bit
  you (e.g. "`pod.py terminate` reports `POD_NOT_FOUND` but the pod is
  still alive in the API" — already captured in CLAUDE.md memory; the
  analogue for a *new* silent-failure class is in scope).
- A marker schema in `workflow.yaml` is missing a field you needed.
- A halt-criterion / gate is wrong, missing, or contradicts CLAUDE.md.
- A test that should have caught a workflow regression is missing.
- `CLAUDE.md` describes a rule but the implementing file (agent, skill,
  script) doesn't enforce it.
- A critic finding whose check belongs in a mechanical verifier — a
  `mechanizable: yes` blocker from any review lens (critic /
  code-reviewer / interpretation-critic / clean-result-critic) that
  targets `verify_task_body.py`, `audit_clean_results_body_discipline.py`,
  SPEC.md lens text, the `consistency-checker` spec, or a future
  `verify_plan.py`. Emit only when the check is concrete and likely to
  recur — not for one-off artifact-specific issues (spam guard).

### No — don't emit

- The bug is in experiment code (training, eval, data generation,
  Hydra config, model spec).
- The bug is task-state-specific (one task's body.md has wrong tags;
  fix that task, not the workflow).
- The bug is environment / external infra flakiness (RunPod
  `SUPPLY_CONSTRAINT`, HF Hub 503, transient WandB outage). Retries
  belong to the experimenter; emit a candidate ONLY if the workflow is
  missing a retry / backoff policy entirely.
- The bug is a one-off that won't recur (a typo in your own
  cwd-resolution this turn, a stale memory line, a one-off shell-quoting
  mistake).
- You cannot name a concrete target file + concrete change — a vague
  unease ("something about dispatch feels off") has nothing to dispatch;
  surface it as a plain note in `events.jsonl` instead. (2026-06-11
  directive: uncertainty alone is NOT a reason to withhold — if you CAN
  name the file + change, emit it with `confidence: low` marked honestly;
  the orchestrator files + spawns at any confidence and the spawned
  `/issue` session's planner may deflect with a reasoned no-change
  report.)
- You are running under `AUTO_REVIEW_DISABLED=1` (already nested inside
  a review or diagnostic loop). Don't recurse.
- You are running inside a workflow-fix `/issue` session (the
  RECURSION GUARD — see § Recursion guard). Such a session does NOT
  auto-file MORE workflow-fix tasks; a candidate it raises is LOGGED +
  notified, not routed.

### Borderline

If the bug is "the workflow allowed me to do X, but X turned out to be
wrong for *this* experiment" — emit a candidate ONLY if the correct fix
is to make the workflow reject X categorically. If the correct fix is
"this specific experiment shouldn't have done X but others should," it's
experiment-specific.

## How to emit a candidate

Include this block in your final return text — after your main report,
before any closing sentence. Plain text, exact format (the orchestrator
parses it):

```
<!-- workflow-fix-candidate v1 -->
target_file: <path(s) under workflow surface, relative to repo root — single
  path, comma-separated list, or a glob (e.g. `.claude/agents/*.md`) when
  the bug pattern hits multiple files; grep first, see "Before emitting"
  below>
bug_observed: <one sentence: what went wrong>
why_workflow_gap: <one sentence: why this is the workflow's fault>
proposed_change: <one sentence summary of the fix>
diff_sketch: |
  <2-10 lines showing the rough shape of the edit; the spawned /issue
  session's planner + implementer will refine. Use `+ ` / `- ` prefixes
  if it helps.>
confidence: low | medium | high
related_task: <task ID this surfaced on, e.g. #391, or n/a>
<!-- /workflow-fix-candidate -->
```

**Before emitting: grep the workflow surface for the pattern.** When the
bug is identifiable by grep (a literal string, a specific regex, a
frontmatter line — e.g. a stale model pin, a deprecated marker field, a
shared phrase), the emitter SHOULD run a one-shot
`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/` (and `src/explore_persona_space/task_workflow*.py
src/explore_persona_space/backends/` if the pattern is plausibly in the
in-scope `src/` modules) BEFORE writing the candidate, and list EVERY hit
in `target_file` — a comma-separated path list or a glob like
`.claude/agents/*.md` is acceptable when the hit set is uniform. One grep
is cheap; it spares the spawned session's planner a discovery round and
prevents the worst-case incomplete fix where a sibling hit is missed
entirely. (#622: a stale model pin lived in 25 sibling agent files; the
original candidate named only one, and the fix had to re-discover scope
via `grep -rln` on receipt.) The orchestrator also follows this rule when
synthesizing a candidate from a prose follow-up — grep before populating
`target_file`. The emitter's grep scopes `target_file`; separately, the FILER
re-runs the grep when composing the body and records the command + hit count
in the body's `verified-at-filing:` line (§ Body-file template) — the
emitter's grep does not substitute for it.

Hard rules:

1. **At most one formal `<!-- workflow-fix-candidate v1 -->` block per
   agent invocation.** The block is the parseable channel; one keeps
   the orchestrator's file-a-task default deterministic. If you notice
   multiple workflow bugs in one run, pick the most concrete + highest
   confidence for the block. **Surface the others as prose follow-ups
   in your main report (e.g. a `## Follow-up workflow concerns` H2 or
   "Follow-ups (orchestrator should consider)" section).** Those prose
   follow-ups are NOT capped — list as many as you genuinely found, one
   per file/concern with a one-line proposed change. The orchestrator
   files + spawns a `/issue --auto` session for each in-scope,
   non-architectural prose follow-up on the same default as the formal
   block; do NOT hold them back hoping they'll surface "on the next
   pass."
2. **Never file a task or spawn a session yourself**, even if your tool
   allowance includes `Agent`. Surface the candidate (block OR prose);
   the parent orchestrator files + spawns. This prevents runaway
   recursion (subagent → files+spawns a workflow-fix session → that
   session's implementer emits a candidate → ...).
3. **Don't emit if you're a Codex twin.** The Codex ensemble reviewers
   (`codex-*`) post their verdicts and exit; they never file tasks or
   spawn sessions. If a Codex twin notices a workflow gap, it should
   write a plain English note in its verdict body — the orchestrator
   decides whether to surface it as a candidate later.

## What the orchestrator does on seeing a candidate

**Default: AUTO-FILE a task + spawn a `/issue --auto` session, do not
park.** For any workflow-fix candidate that is (a) in-scope per the
workflow surface list above and (b) non-architectural / not a
public-contract change — at ANY confidence level (2026-06-11 directive:
low confidence no longer defers; the spawned `/issue` session's planner +
critic ensemble + code-reviewer are the check) — the orchestrator's
default action is to file a `kind: infra` task pre-filled from the
candidate and spawn a background `/issue <N> --auto` session, then keep
working. **"Candidate" means BOTH (i) a formal `<!-- workflow-fix-
candidate v1 -->` block AND (ii) any concrete prose follow-up an agent
surfaces** — e.g. a "Follow-ups (orchestrator should consider)" section,
a "Related concerns" bullet, or any specific suggestion to change a
workflow-surface file. Both come in via the same channel (agent return
text) and trigger the same default. This applies whether the candidate
came from a subagent's return text OR from the orchestrator's own
observation during its work (see the "orchestrator is itself the agent"
clause below). Parking the candidate for Thomas's greenlight is the
EXCEPTION, reserved for the single case enumerated in "When the
orchestrator does not file a task" — genuinely architectural /
public-contract changes (now routed through the `/issue` plan-approval
gate; see § Architectural greenlight). Do NOT park an in-scope,
non-architectural gap as a chat note at any confidence — auto-file +
spawn it, regardless of whether it arrived as a formal block or as prose.

When any subagent returns text containing EITHER a `<!-- workflow-fix-
candidate v1 -->` block OR a prose follow-up that names a specific
workflow-surface file + a specific change, the orchestrator (parent
assistant, `/issue` skill, `research-pm`, or any session running the
top-level loop), UNLESS running under the recursion guard
(§ Recursion guard):

1. **Greps the surface** for the pattern (preserved rule), populating
   `target_file` (a comma-separated list or glob when the hit set is
   uniform), and records the exact grep command + hit count for the
   body's `verified-at-filing:` line (§ Body-file template).
2. **Computes the candidate fingerprint** (see § Dedup):
   `fp = sha256(normalize(proposed_change) + "||" + normalize(bug_observed))[:12]`
   (`task_workflow.wf_fix_fingerprint`). For a prose follow-up the fields
   come from the synthesized `proposed_change` / `bug_observed`.
3. **Dedup check** (§ Dedup): skip filing ONLY if an OPEN (non-terminal)
   `kind: infra` workflow-fix task has the SAME `(target_file,
   fingerprint)` — `task_workflow.is_open_workflow_fix_task(target_file,
   fp)` returns its id. On a hit, log the candidate with note
   `deduped against #<M>` and continue (no new task, no `failed` marker
   — a dedup hit is not a failure). Same `target_file` but a DIFFERENT
   fingerprint is NOT a duplicate → proceed to file (each distinct bug
   gets its own plan review).
4. **Logs** the candidate to the current task's `events.jsonl` as
   `epm:workflow-fix-candidate v1` (so the dashboard surfaces it). For
   prose follow-ups the marker `note` records the file + summary plus a
   `source: prose-followup` field; for formal blocks it records the
   verbatim block plus `source: candidate-block`. The note also records
   the routing decision: `routed: filed #<N>` | `deduped against #<M>` |
   `parked: EPM_WORKFLOW_FIX_SESSION` | `parked: architectural`.
5. **Files + dispatches** the `kind: infra` task in ONE call via the
   file-time wrapper `scripts/file_infra_task.py` (#690), which files via
   `task.py new` (returning id N), applies the dedup-key tags AT creation
   (forwarded as repeated `--tag`), and best-effort `spawn-issue --auto`:
   ```bash
   uv run python scripts/file_infra_task.py --kind infra \
     --title "workflow-fix: <proposed_change, <=60 chars>" \
     --body-file /tmp/wf-fix-body-<slug>.md \
     --origin-prompt "<verbatim candidate block OR surfaced prose>" \
     --tag wf-fix --tag "wf-fix-fp:<fp>"
   # files #N (tags wf-fix + wf-fix-fp:<fp> applied at creation), then attempts
   # spawn-issue --issue N --auto
   ```
   The spawn NO-OPS cleanly (the task stays filed at `proposed` for the
   backstop, exit 0) when the Happy daemon is unreachable OR the shared
   5-session infra cap is full / occupancy is unreadable — the watcher's
   `proposed_infra_sweep` pass (#690) dispatches the filed task within ~10 min.
   The body-file template (§ Body-file template) carries the `## Provenance`
   block with the literal `workflow_fix_target: <path>` + `fingerprint: <fp>`
   lines (written VERBATIM by `--body-file`, so the grep fallback works). NEVER
   route these keys through `set_body` — it strips leading frontmatter (the
   title prefix is set by `task.py new` via the wrapper; the dedup tags by the
   forwarded `--tag` flags; the Provenance lines live in the body).
6. The wrapper already performed the spawn in step 5 (file + dispatch is one
   call). If you must dispatch separately (e.g. you filed via a bare
   `task.py new` for a reason the wrapper does not cover), the equivalent is
   `uv run python scripts/spawn_session.py spawn-issue --issue <N> --auto`.
7. **Posts** `epm:workflow-fix-task-filed v1` on the PARENT task
   (the one that raised the candidate) and **continues** the current
   work. Does NOT block on the fix.

The fix lands via the child session's full `/issue` pipeline (planner /
Phase-2 critic ensemble / `implementer` / Claude+Codex `code-reviewer` /
test-verdict / Step 10d worktree auto-merge), NOT via
`Agent(workflow-improver)`. When the child task reaches `completed` (its
Step 10d merged to `main`), the orchestrator posts
`epm:workflow-fix-applied v1` on the PARENT task (cross-linking
`applied_task: #<N>`, `merge_sha`). If the child reaches `blocked` /
`archived` without merging, the orchestrator posts
`epm:workflow-fix-failed v1` on the PARENT task with the reason.

If the orchestrator is *itself* the agent that found the bug (no
subagent involved — the bug surfaced during the orchestrator's own
work), it files + spawns directly with the same protocol and the same
default: an in-scope, non-architectural gap is AUTO-ROUTED in the
background at any confidence, not parked for greenlight. The
orchestrator does not get a stricter bar just because it noticed the
gap itself rather than receiving a candidate block.

## Body-file template

The orchestrator fills the `<...>` placeholders mechanically from the
candidate block fields; for a prose follow-up it synthesizes the same
fields per the prose-synthesis rule. The `workflow_fix_target:` /
`fingerprint:` lines live in the `## Provenance` BODY block (written
verbatim by `--body-file`, grep-findable), NOT as top frontmatter keys:

```markdown
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task <related_task> (emitting agent: <emitting_agent>).

## Goal

<proposed_change — one sentence>

## Workflow gap

- **Bug observed:** <bug_observed>
- **Why it is a workflow gap:** <why_workflow_gap>
- **Confidence (emitter):** <confidence>
- verified-at-filing: `<grep cmd the FILER ran at body-compose time>` → <N hits in M files; per-target hits for each file named in target_file> (<UTC date>), OR `n/a — <one-line reason the bug claim is not grep-verifiable>`

## Proposed change (candidate diff sketch — refine in planning)

<diff_sketch verbatim, OR "(none — synthesized from prose follow-up)">

## Scope / surfaces

- Primary target: `<target_file>`
- Grep the workflow surface for the pattern before editing
  (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: <target_file — the exact candidate target_file string>
- fingerprint: <fp — sha256(normalize(proposed_change)+"||"+normalize(bug_observed))[:12]>

<verbatim candidate block, including the opening + closing comment lines,
 OR the verbatim surfaced prose for a prose follow-up>
```

The `--origin-prompt` carries the verbatim candidate (the `## Provenance`
body section duplicates it for human readability + the grep fallback; the
`origin_prompt` is the machine record the clean-result `**Context:**` row
ships forward).

**The `verified-at-filing:` line is REQUIRED in every wf-fix body** (with the
`n/a — <reason>` escape for genuinely non-grep-able behavioral gaps). It is
produced by the FILER at body-compose time — a stale emitter-side grep does
not satisfy it (freshness at filing is the point; cf. the /daily Retraction
re-check) — and its hit count must be consistent with the body's bug claim
and `target_file` list. Consistency BINDS — a real grep that does not bind
to the claim does not satisfy the mandate (#1307; the two 2026-07-12
filings): (a) **per-target confirmation** — run the pattern grep against
EACH file named in `target_file` and state per-target hits; for a presence
claim (the body asserts the site/pattern EXISTS there), a 0-hit named
target is a mis-target, not evidence — re-grep repo-wide, correct
`target_file` to the real site(s), and re-verify BEFORE filing (#1290: a
real repo-wide grep sat beside a `target_file` with 0 hits for the claimed
parse site; the true site was a sibling SKILL.md, found only by the spawned
session's clarifier). An absence-of-guard claim is exempt from the mis-target
rule — its 0-hit in-target result IS the evidence. (b) **relocation grep
before any nonexistence claim** — asserting a cited symbol / test / file
nonexistent ("no longer exists") requires a recorded repo-wide relocation
grep (`grep -rn '<symbol>' tests/ scripts/ .claude/ src/`); a single-path
probe cannot distinguish "removed" from "moved" (#1296: a single-file
pytest probe backed a "NO LONGER EXISTS" claim; the test had moved to
`tests/test_issue_dispatch.py`). (c) **context consistency** — a
presence hit binds only after its surrounding lines are READ: a hit
whose context ALREADY IMPLEMENTS the proposed change is a landed-fix
signal, not gap evidence — a count-only bind does not satisfy the
mandate; on a landed-fix hit do NOT file — the candidate is a duplicate
of the landed fix (route it to dedup/archive, never a new task) (#1330
filed over the already-landed #1309 fix: the verified-at-filing grep hit
the landed recipe paragraph itself and the hit's context was misjudged
as unrelated). Lineage: #1221/#1229/#1249 — three filings in two
days carried grep-refutable claims (nonexistent call sites, overcounted
"unguarded sites", an improvised path); each burned a spawned session's
verification rounds. There is NO mechanical injector or lint for this line —
it is a compose-time duty.

## Dedup

The dedup GRAIN is `(target_file, fingerprint)`:

```
fingerprint = sha256( normalize(proposed_change) + "||" + normalize(bug_observed) )[:12]
normalize(s) = s.lower(), collapse internal whitespace to single spaces, strip
               leading/trailing whitespace, strip trailing punctuation (.,;:!?)
```

(`task_workflow.wf_fix_fingerprint`.)

- Same `target_file` + **same** fingerprint (the same bug re-raised) →
  DUPLICATE → do NOT double-file; log `deduped against open #<M>` and
  continue.
- Same `target_file` + **different** fingerprint (a DISTINCT bug on the
  same hot file) → NOT a duplicate → file a new task. **Each distinct bug
  gets its own plan review** — SKILL.md, CLAUDE.md, and this rule file
  itself routinely get multiple distinct candidates per week, and
  dropping the second would lose a real fix (this REPLACES the old
  "queue the second candidate to the running agent via SendMessage" rule;
  the both-bugs-get-fixed invariant is preserved by filing two tasks).

**Dedup KEY SURFACES:**

- **PRIMARY (round-trips through `view --json` + REGISTRY):** a
  `workflow-fix:` OR `daily-fix:` TITLE PREFIX (one per filing channel —
  orchestrator vs /daily route-2; `task_workflow.WF_FIX_TITLE_PREFIXES` is the
  single source of truth, widened by #1180; set by `task.py new --title` / a
  later `set_title`, both update the REGISTRY snapshot) AND the `wf-fix-fp:<fp>`
  TAG (set by `add-tag`; round-trips via `view --json .frontmatter.tags`).
- **FALLBACK (documented, WORKING):** the body's `## Provenance`
  `workflow_fix_target: <path>` line (written verbatim by `--body-file`)
  — grepped when the title/tag surfaces are insufficient.
- **NEVER** route any dedup key through `set_body` — it strips leading
  frontmatter except `paper`/`abstract`.

**Dedup predicate (exact):** a candidate is a duplicate iff there exists a
task with `kind: infra` AND a `workflow-fix:` or `daily-fix:` title prefix
(`task_workflow.WF_FIX_TITLE_PREFIXES`) AND a
`wf-fix-fp:<fp>` tag matching the candidate's fingerprint AND a
`## Provenance` `workflow_fix_target:` line EXACTLY string-matching the
candidate's `target_file` AND whose status is NOT in the terminal set
`{completed, archived}`. Same file + DIFFERENT fingerprint is NOT a
duplicate. The library helper
`task_workflow.is_open_workflow_fix_task(target_file, fingerprint) -> int | None`
is the canonical, tested implementation (`tests/test_workflow_fix_dedup.py`).
A closed (`completed`/`archived`) workflow-fix task does NOT block a
re-raise of the same bug. (The /daily Step C parked-candidate sweep applies
this temporally: a swept PARKED candidate that PREdates a closed matching fix
task's creation is treated as subsumed by it — suppressed, pure churn to
re-route — while a candidate parked AFTER the fix closed is a genuine
re-raise and stays enumerated; see
`scripts/sweep_parked_wf_candidates.py`.)

**Recently-closed-sibling ADVISORY (#1399 — advisory only, never a block).**
The predicate above is exact-`(target_file, fingerprint)` over OPEN tasks, so
a JUST-MERGED sibling with different wording is invisible by design
(2026-07-15: #1350 was filed 25 min after #1329 merged the same fix, and
#1330 duplicated #1309's already-landed guidance — two pipeline sessions
burned). At filing time `scripts/file_infra_task.py` therefore prints a
stderr advisory listing wf-fix/daily-fix tasks CLOSED (completed/archived)
within the last ~7 days that overlap the candidate — by `workflow_fix_target:`
path token, or by informative title token
(`task_workflow.recent_closed_workflow_fix_tasks`) — capped at the 10 most
recent. The filer eyeballs the list before letting the spawned session run.
ADVISORY ONLY: it never blocks the filing, never changes exit codes, and
fails soft with a printed diagnostic — the rule above is unchanged (a closed
fix still never blocks a genuine re-raise). Because the wrapper's stderr
arrives after it has already filed and best-effort-spawned, an agent consumer
that spots a just-merged duplicate in the list applies the post-hoc remedy:
archive the just-filed task (`task.py set-status <id> archived`) and stop its
spawned session (`spawn_session.py stop --session-id <sid>`).

## Recursion guard

A workflow-fix `/issue` session must NOT auto-file MORE workflow-fix
sessions for its OWN subagents' findings — otherwise: implementer →
emits a candidate → orchestrator files+spawns → that session's
implementer emits a candidate → ... unbounded fan-out. This is the
analogue of the `AUTO_REVIEW_DISABLED` sentinel.

**A session is a workflow-fix session iff** its task carries a
`workflow_fix_target:` Provenance line (the DURABLE signal —
`task_workflow.is_workflow_fix_session(N)`) OR `EPM_WORKFLOW_FIX_SESSION=1`
is set in the environment (the in-session convenience flag). The
candidate-receive logic checks the durable signal FIRST (it survives a
watcher crash-recovery respawn, which re-runs `spawn-issue --auto`
WITHOUT custom env, so the env var is lost on respawn) and the env var
SECOND. If EITHER is true, the orchestrator does NOT file/spawn — it
LOGS the candidate to `events.jsonl` (`epm:workflow-fix-candidate v1`,
note: `parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target,
see § Recursion guard`) and surfaces it as a notification.

**Escape valve (never silently lost):** such a session CAN still emit
candidate blocks — they get parked/notified, not auto-routed, so its own
workflow-fix bug is parked for the nightly /daily parked-candidate routing
pass (Step C in `.claude/skills/daily/SKILL.md`; enumerator
`scripts/sweep_parked_wf_candidates.py`) exactly
as `AUTO_REVIEW_DISABLED`-suppressed candidates are. The cost is a
one-cycle delay, not a dropped bug. The recursion-guard predicate is
executable-tested (`tests/test_workflow_fix_dedup.py`
`test_is_workflow_fix_session_true_on_provenance_line`).

**A `daily-fix:` session for a NON-workflow-surface fix is intentionally
outside this guard.** A `/daily` route-2 item filed with `wf_fix: false`
(an experiment-code / non-workflow-surface fix) carries neither the
`wf-fix` tags nor the injected `workflow_fix_target:` Provenance block
(`.claude/skills/daily/SKILL.md` § route 2), so
`is_workflow_fix_session()` correctly stays false and its session MAY
auto-file a first-generation workflow-fix candidate it uncovers (worked
example: #1286, an experiment-script fix session, filed #1299 on
2026-07-13). This is by design, not a gap: the dedup predicate still
covers such filings via the `daily-fix:` title prefix
(`task_workflow.WF_FIX_TITLE_PREFIXES`, #1180), and fan-out stays bounded
because the CHILD workflow-fix task's body DOES carry the
`workflow_fix_target:` line, putting the child session under the guard.

## Architectural greenlight

Non-architectural workflow fixes (the vast majority) are 0 GPU-h, so the
spawned `/issue --auto` session AUTO-APPROVES the plan under
`EPM_PLAN_AUTOAPPROVE_GPU_HOURS` (default 100) and self-merges at Step 10d
— preserving the "workflow-surface edits are committed + merged + pushed
automatically, no approval gate" default.

**Architectural / public-contract changes still need the user's
greenlight**, now expressed as the `/issue` plan-approval gate (park at
`plan_pending`). A change is architectural when it would rename a status
enum, change a marker schema or `task.py` subcommand / CLI contract,
relocate an agent or skill file, or remove/restructure a subsystem
(e.g. "remove the Codex ensemble"). A change is NOT architectural just
because it touches more than one line or one file — adding a guardrail
step, tightening an instruction, fixing a contradiction between CLAUDE.md
and an implementing file, or adding a missing field/note is in-scope
auto-fix work.

Because a 0-GPU-h plan would otherwise auto-approve EVERYTHING (including
architectural changes), the architectural subset is gated by a SECOND,
orthogonal signal: the planner sets `architectural: true` in the
workflow-fix task's plan frontmatter (and states a one-line "ARCHITECTURAL
— needs user greenlight" banner in the Plan Summary) whenever the change
meets the test above. The `/issue` Step 2c autonomous-approval logic gates
on BOTH the GPU-h cap AND `architectural != true`: a plan flagged
`architectural: true` PARKS at `plan_pending` for the user regardless of
GPU-h. **Fallback** (if the Step-2c `architectural:` wiring is deferred):
the orchestrator files the architectural candidate's task but spawns it
WITHOUT `--auto` — a bare session that parks at `plan_pending` until a
human types `/issue <N>` — and logs the candidate for the user. Either
way the architectural-greenlight invariant holds.

## When the orchestrator does not file a task

Filing + spawning is the DEFAULT (see above). The exceptions below apply
identically to formal `<!-- workflow-fix-candidate v1 -->` blocks AND to
prose follow-ups — a surfaced prose follow-up does NOT get a stricter bar
(or a looser one) than a formal block. The orchestrator logs the
candidate but skips the file/spawn ONLY in these cases:

- **Genuinely architectural / public-contract change.** Routed through
  the `/issue` plan-approval gate (§ Architectural greenlight) — the task
  is filed but parks at `plan_pending` for the user (spawn WITHOUT
  `--auto`, or `architectural: true` in the plan). Log the candidate;
  surface it to Thomas in the next chat turn. A change is NOT
  architectural just because it touches more than one line or one file.
- **Nothing concrete to dispatch.** The prose names NO specific
  workflow-surface file AND no specific change ("we might want to
  consider rethinking X someday") — there is literally nothing to file.
  Log the marker for the dashboard; no task filed. NOTE (standing user
  directive, 2026-06-11): LOW CONFIDENCE IS NOT A SUPPRESSION REASON. A
  candidate or follow-up that names a concrete file + concrete change is
  filed even at `confidence: low` / hedged wording — "deferred for a
  future deliberate pass" is the banned outcome. The spawned `/issue`
  session's planner makes the deliberate call with the file open (it may
  deflect with a reasoned "no change needed" report), and the code-review
  ensemble is the second check.

Three operational deferrals (NOT greenlight gates — the fix still
happens, just deduped/queued):

- An OPEN workflow-fix task already exists for the SAME `(target_file,
  fingerprint)` (§ Dedup). Do NOT double-file; log
  `deduped against #<M>` and continue. A DISTINCT bug on the same file
  (different fingerprint) is NOT a duplicate — file it.
- The candidate's `target_file` is in the out-of-scope set (experiment
  code, `tasks/`, etc.). The orchestrator logs the candidate AND posts a
  brief note in the marker about the misclassification so the emitting
  agent's pattern can be corrected; no task is filed because the target
  is out of scope by definition.
- The candidate's `target_file` documents behavior the user is ACTIVELY
  changing in the SAME session — i.e. the file's subject matter is the
  live topic of an in-flight user-directed code change. DEFER the
  candidate until that code change lands, then document the NEW behavior;
  do not document the old default while the user is mid-flight replacing
  it. Filing here risks a fix that is obsolete within minutes
  (2026-06-23: a workflow-fix commit clarified the marker
  contrastive-negative loss-mask "requires the slot-suppression flag" —
  then 27 min later a user-directed code change made marker + end-of-turn
  loss the default and turned that flag into a deprecated no-op, directly
  superseding the just-merged doc edit on the same file).

## Markers

Defined in `.claude/workflow.yaml § markers`:

- `epm:workflow-fix-candidate v1` — posted by orchestrator on receiving
  a candidate block / prose follow-up from any subagent's return text
  (or from its own observation). Records the routing decision in its note
  (`routed: filed #<N>` | `deduped against #<M>` | `parked: ...`).
- `epm:workflow-fix-task-filed v1` — posted by orchestrator on the PARENT
  task when it files the `kind: infra` task + spawns the `/issue --auto`
  session. Fields: `filed_task` (new id N), `target_file`, `fingerprint`,
  `session_spawned` (true|false), `origin_candidate` (verbatim).
- `epm:workflow-fix-applied v1` — posted by orchestrator on the PARENT
  task when the spawned child workflow-fix task reaches `completed` (its
  `/issue` Step 10d merged to `main`). Fields: `applied_task: #<N>`,
  `merge_sha`. The CHILD task carries only the standard `/issue`
  lifecycle markers — its completion + Step 10d merge ARE the apply
  record there.
- `epm:workflow-fix-failed v1` — posted by orchestrator on the PARENT
  task if the spawned child reaches `blocked` / `archived` without
  merging, or the spawn itself failed. Fields: `failed_task: #<N>`,
  `failure_reason` (child-blocked | child-archived | spawn-failed),
  `origin_candidate` (verbatim). A dedup hit (no action) is NOT a failure
  and posts no `failed` marker.

**Parent vs child split.** ALL workflow-fix-specific markers
(`candidate`, `task-filed`, `applied`, `failed`) fire on the PARENT task
(the one that raised the candidate) — they record the routing + outcome
from the parent's vantage. The CHILD workflow-fix infra task carries ONLY
the standard `/issue` lifecycle markers (created, status-changed, plan
markers, implementer/review markers, completion) — it is an ordinary infra
task and needs no special markers. This keeps the dashboard's per-task
lifecycle clean and lets the parent's events.jsonl tell the "I raised a
workflow bug → it was filed as #N → #N merged" story end to end.

Posting target: the `events.jsonl` of the task the emitting agent was
working on. If the emitting agent was working outside any task (e.g.
during `/pm` triage, `/daily`, or chat-mode work), the orchestrator
appends to `.claude/cache/workflow-fix-events.jsonl` instead — same
schema, a durable file-based trace. (The dashboard does not yet read
this fallback file — it surfaces only per-task `events.jsonl` markers;
homepage rendering of the fallback is unimplemented.)

## Anti-patterns

| Don't | Do |
|---|---|
| Subagent files a task / spawns a session itself | Surface the candidate (block or prose); orchestrator files + spawns |
| Emit a candidate for an experiment-code bug | Route to `implementer` / `experiment-implementer` |
| Emit ≥2 formal `<!-- workflow-fix-candidate v1 -->` blocks per run | Pick one for the block; list the rest as prose follow-ups (orchestrator files each) |
| Emit `confidence: high` without a concrete diff_sketch | Sketch the actual lines; if you can't, drop to `medium` or skip |
| Wait for the workflow-fix `/issue` session before continuing | Background-file + spawn; current task continues immediately |
| Emit a candidate against `src/`, `configs/`, `tasks/` | Out of scope — fix belongs elsewhere |
| Park an in-scope, non-architectural gap for greenlight (any confidence) | Auto-file a task + spawn `/issue --auto`; greenlight only for architectural / public-contract changes (the plan-approval gate) |
| Defer a `confidence: low` candidate "for a future deliberate pass" | File it now; the spawned session's planner makes the deliberate call with the file open and may deflect with a reasoned no-change report |
| Double-file a SECOND distinct bug on the same hot file (target_file-only dedup) | Dedup on `(target_file, fingerprint)`: a distinct bug files its own task + gets its own plan review |
| A workflow-fix `/issue` session auto-files MORE workflow-fix tasks for its own findings | Recursion guard: a workflow-fix session logs + notifies its candidates, never auto-routes them |
| Orchestrator surfaces an agent's "Follow-ups (orchestrator should consider)" section to Thomas as a chat note asking "should I apply these?" | Treat each in-scope, non-architectural follow-up as a synthesized candidate and auto-file + spawn a `/issue --auto` session for it; do NOT ask |
| Drop a prose follow-up because it lacked the formal block tags | Prose follow-ups trigger the same file-a-task default as formal blocks; synthesize a candidate from the prose and file |
| Hold prose follow-ups back hoping they'll surface "on the next pass" | List every concrete in-scope follow-up the agent found; the orchestrator files each |
| Name a single `target_file` when a literal-string bug pattern hits N sibling workflow files (#622: a stale model pin lived in 25 agent files; one was named) | `grep -rln '<pattern>' .claude/ CLAUDE.md scripts/` first; list every hit in `target_file` as a comma-separated path list or a glob |
| File a body whose bug claim (call sites, site counts, paths) was never re-verified by grep at filing time (#1221/#1229/#1249: 3 stale-claim filings in 2 days, each burning a spawned session's verification rounds) | Run the grep at body-compose time; record `verified-at-filing: <cmd> → <hits>` (or `n/a — <reason>`) in `## Workflow gap` |
| Record a real grep that does not BIND to the claim — a repo-wide pattern grep beside a named target_file with 0 hits for the claimed site (#1290), or a single-path probe backing a "no longer exists" claim (#1296) | State per-target hits for each file named in target_file (presence claim + 0-hit target ⇒ re-grep repo-wide, correct target_file, re-verify before filing); back any nonexistence claim with a recorded repo-wide relocation grep |
| Bind a presence grep on hit COUNT alone when the hit's surrounding context already implements the proposed change (#1330 filed over the landed #1309 fix) | Read each presence-hit's surrounding lines before filing; a hit that IS the fix = already landed — dedup, don't file |

## Composition with other rules

- **AUTO_REVIEW_DISABLED sentinel** (user-global CLAUDE.md): suppresses
  this protocol too. If your prompt carries that sentinel, treat
  workflow-fix candidate emission as forbidden for the turn.
- **Halt-criterion contract** (CLAUDE.md): emitting a candidate is NOT
  the same as raising `AskUserQuestion`. The candidate is a non-blocking
  side channel; it does not pause the current work, does not flip
  status, does not consume a gate. The spawned `/issue --auto` session
  has its own plan-approval gate (the architectural-greenlight surface).
- **`/issue` pipeline** (`.claude/skills/issue/SKILL.md`): the spawned
  workflow-fix task runs the standard code-change path — `kind: infra`
  routes to the `implementer` at Step 4b, the Claude+Codex `code-reviewer`
  ensemble gates it, and Step 10d auto-merges the worktree to `main`. If
  the candidate is misclassified (out of scope), the spawned session's
  planner / implementer deflects and the task is archived; the
  orchestrator posts `epm:workflow-fix-failed v1` with the reason. The
  file-time wrapper `scripts/file_infra_task.py` (#690) is the canonical
  file+dispatch entrypoint the orchestrator uses at step 5 above (files via
  `task.py new` + best-effort `spawn-issue --auto` in one call, no-op'd by a
  daemon-unreachable / cap-full / occupancy-unreadable spawn gate), and the
  watcher's always-on `proposed_infra_sweep` pass is the backstop for any
  filed-but-not-dispatched task.
- **Codex ensemble reviewers**: never emit candidates (rule above).
  They write notes in their verdict body; the Claude twin (or
  reconciler) decides whether to surface a candidate later.
- **`workflow-improver` agent (RETIRED #678):** the
  `.claude/agents/workflow-improver.md` file is frozen with a DEPRECATED
  banner for historical reference; it is NEVER spawned. The
  `scripts/workflow_lint.py --check-no-workflow-improver-spawn` check
  (bundled into the no-flags default run) FAILs if any live
  `Agent`-tool spawn naming `subagent_type` `workflow-improver` survives
  anywhere in the workflow surface.

## Built-but-stranded fixes don't help — get the helper onto main and wired in

A documented — or even fully BUILT — fix that is not merged to `main` and not
wired into the running code path does NOT help: the running session keeps using
the old path. A fix exists in three places (a) documented in a rule, (b) built
on a branch, (c) merged to main and called by the running path — only (c)
actually changes behavior. When a rule prescribes a helper, the helper must be on
`main` AND the code that should call it must call it; verify both before treating
the lesson as "handled."

Worked example (#722, 2026-06-29): `src/explore_persona_space/analysis/vectorized_mlp_skill.py`
+ a vectorized #722 driver were fully built on the unmerged `vectorized-mlp-skill`
branch while the #722 session ran the OLD serial CPU script with a ~38h ETA — the
fast path existed but was stranded off main, so it changed nothing. CLAUDE.md
already cites that helper as canonical; this lesson is the coordination half:
"canonical helper named in a rule" ⇒ also ensure it is on main and called.

For VECTORIZATION rewrites specifically, the mechanism half of this lesson is
codified as the "Supersede contract" in
`.claude/rules/vectorize-many-cell-fits.md` — batched helper on `main` in the
same round, serial twin tombstoned (`FutureWarning` +
`EPM_FORBID_SERIAL_FITS=1` raise), and an open-task check before starting a
rewrite (#722, #778, #834).
