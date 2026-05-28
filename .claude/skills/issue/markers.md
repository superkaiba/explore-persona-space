# Marker Taxonomy

All structured state on a task is carried in `events.jsonl` rows under
`tasks/<status>/<N>/events.jsonl`. This file is the source of truth for
marker syntax and semantics.

## Format

Each row is a single JSON object. The skill-side helper `task.py
post-marker` writes one row per call. The schema:

```jsonl
{"ts": "2026-05-20T12:34:56Z", "kind": "epm:<kind>", "version": <n>, "note": "<body>", "metadata": {...}}
```

- `epm` = "explore persona manager" namespace (shared prefix, keeps our
  markers out of conflict with other tools).
- `kind` = one of the kinds below.
- `version` = monotonic integer per kind. `1` is the original; `2+` are
  revisions. The skill parses ALL rows and uses the highest-version row
  per `kind` as authoritative.
- `note` = human-readable body (capped at 50,000 chars by `task.py`).
- `metadata` = compact JSON sidecar (e.g., reviewer-loop fields below).

**Never edit or delete** a row in `events.jsonl`. History is part of the
audit trail. The `task.py` writer is the only legitimate mutator and
enforces append-only via flock.

## Posting markers

Use the CLI:

```bash
uv run python scripts/task.py post-marker <N> epm:plan --note "Plan drafted"
```

Or, when the body is large enough to risk shell-quoting traps, write it
to a temp file and pass `--file`:

```bash
uv run python scripts/task.py post-marker <N> epm:results --file /tmp/results-body.md
```

The skill EXIT helper `scripts/post_step_completed.py` wraps this for
the `epm:step-completed` audit row at every step boundary; see
`.claude/skills/issue/SKILL.md` § "Step-completed re-entry skip-ahead".

## Kinds

The kinds table is auto-generated from
(see workflow.yaml § markers). Do NOT edit inside the fence; run
`uv run python scripts/workflow_lint.py --emit-tables` to regenerate
after a YAML edit. The full per-kind documentation (with longer
required-fields prose) lives in `workflow.yaml § markers` itself; the
table below is the at-a-glance index.

<!-- workflow.yaml: AUTO-GENERATED (marker-kinds) -->
| Kind | Posted by | When | Required fields |
|------|-----------|------|-----------------|
| `epm:auto-defaults` | skill | Step 0b | what fields were auto-filled in body.md frontmatter and the inferred values |
| `epm:codex-task-spawned` | scripts/codex_task.py | Whenever the orchestrator dispatches a Codex companion task via scripts/codex_task.py with --issue N | Codex job_id (task-xxxx-yyy), effort, write flag, poll interval, max-wait cap, probe-error cap |
| `epm:codex-task-completed` | scripts/codex_task.py | Codex companion task terminated cleanly (phase=done) after the polling loop observed it | Codex job_id, elapsed seconds, terminal phase |
| `epm:codex-task-failed` | scripts/codex_task.py | Codex companion task terminated non-cleanly (phase in {failed, cancelled}), force-cancelled after max-wait cap, force-cancelled by stall detector (phase=running but log untouched > stall_detect_secs), killed by SIGTERM/SIGINT, exceeded probe-error cap, post-spawn probe rejected the job-id, or spawn failed before a job-id was assigned | Codex job_id (if assigned), elapsed seconds, terminal phase OR error description (e.g. 'timed out after Ns', 'N consecutive probe failures; last error: ...', 'stall detected: phase=running but log file untouched for Ns') |
| `epm:clarify` | skill | Step 1 | numbered clarifying questions OR 'No blocking ambiguities' |
| `epm:clarify-questions` | skill | Step 1 | alias of epm:clarify used by some helper scripts; same body shape |
| `epm:clarify-answers` | skill (relaying user chat reply) | Step 1 | user's answers to the most recent epm:clarify questions, persisted to events.jsonl |
| `epm:goal-updated` | task.py set-goal (called by skill at /issue Step 0c, by clarifier at Step 1, or by planner at /adversarial-planner Phase 1) | any time the canonical Goal-of-the-experiment is set or refined | from (prior `goal:` frontmatter value or null), to (new value), by (user\|clarifier\|planner), optional reason. One marker per actual change; idempotent re-application emits nothing. |
| `epm:plan` | skill (via adversarial-planner) | Step 2 | Goal, method delta, reproducibility card, success/kill criteria, GPU-hr estimate, pod intent |
| `epm:plan-approved` | skill (relaying user reply) | Step 2c | user `approve` reply; advances status from plan_pending to approved |
| `epm:consistency` | skill (via consistency-checker) | Step 2b | PASS/WARN/BLOCK verdict, variables that differ from parent, shared baseline check |
| `epm:proposed-tests` | skill (via implementer / experiment-implementer) | Step 4b, TDD mode only | Posted ONLY when the plan body contains `### TDD: yes` or the user requests TDD. Body: ≥1 happy-path + ≥2 distinct error/edge-case tests, behavior-focused, in fenced blocks. Implementer EXITs after posting and awaits user `approve-tests` reply. Skill pauses pending approval, then resumes Step 4b. |
| `epm:experiment-implementation` | skill (via experiment-implementer) | Step 4b | Posted for type=experiment tasks. Body MUST follow the four-section shape `(a) What was done` / `(b) Considered but not done` / `(c) How to verify` / `(d) Needs human eyeball`. |
| `epm:smoke-architecture-check` | skill (via experiment-implementer) | Step 4b (implementer report-time, before code-review) | Implementer self-tag verifying smoke and sweep paths share the same dispatcher / subprocess shape / env injection / logging surface. Body shape: one of:   `verdict: PASS_UNIFIED` — smoke IS sweep with one cell     (`--cells 1 --seeds 1` or equivalent); both paths share the     same dispatcher and subprocess shape end-to-end.   `verdict: PASS_CANARY canary_cell=<cell_id>` — paths diverge     (e.g., smoke uses in-process train, sweep uses subprocess     wrapper) AND the plan §4 Design section justified the     divergence in two sentences AND named the canary cell that     exercises the sweep path during smoke.   `verdict: FAIL_NO_CANARY` — paths diverge without the §4     justification + canary; implementer also emits a     `workflow-fix-candidate v1` suggesting re-architect toward     unification. SKILL.md Step 6d.0 gate (inline gates id=10)     refuses to dispatch experimenter on FAIL_NO_CANARY — bounces     to planner for unification re-architect. Rationale: task #397 rounds 9/10/10' (2026-05-27) — smoke ran in-process `train_one_cell`; sweep ran `run_one_cell.py` as a subprocess. Smoke kept PASSing; sweep crashed within ~5s of nohup on every architectural assumption the in-process path silently satisfied. Round 11's pivot was to UNIFICATION (in-process serial). Unification is the default; canary is the escape hatch when unification is genuinely impossible (e.g., per-cell vLLM that can't reset in-process). |
| `epm:compute-deviation` | skill (via experiment-implementer) | Step 4b (implementer report-time, before code-review) | Posted when the implementer's resolved per-cell parameters project a wall-time deviation exceeding 2× any row in the planner's §9 per-component compute-projection table. Body shape: `component: <planner-§9-row-name> planned_wall_h: <P> projected_wall_h: <X> ratio: <Y> basis: <planner-§9-row-basis>`. Optional: `action: auto_descope_to_<spec>` when the orchestrator's pivot_criteria.compute_deviation_over_2x logic identifies a descope that keeps ratio ≤ 1.5× AND preserves statistical power per the planner's §9 stratification spec. When `action` is present, the orchestrator continues with the descoped sweep. When absent, the orchestrator surfaces gates.conditional.compute_deviation_resolution (id=12) with the 2-option pivot prompt. Rationale: task #397 round 6 (2026-05-27) — projection was 3-4× plan §12 but surfaced as "needs human eyeball" rather than as a structural pivot, costing ~17h before the user noticed. Per the 2026-05-22 workflow.yaml PHILOSOPHY directive, auto-pivot is the default; halt is the rare exception. |
| `epm:new-bug-class` | skill (via experiment-implementer) | Step 4b (implementer report-time, before code-review) | Implementer self-tag that this round's fix touches a previously- untouched module/pattern in the current task's implementer sequence. Body shape: `bug_class: <short_snake_case_tag>`. Example tags: `pod_side_task_py_shellout`, `vllm_teardown_oom`, `subprocess_wrapper_missing_upload`, `dispatcher_env_loading`, `cwd_relative_log_path`.  EXCLUSION RULE: if this round's bug is a workflow-surface bug per `.claude/rules/workflow-fix-on-bug.md` § "Yes — emit", emit `<!-- workflow-fix-candidate v1 -->` in the report text INSTEAD OF posting `epm:new-bug-class v1`. The whack-a-mole detector at SKILL.md Step 5.bis(b) EXCLUDES workflow-fix-candidate rounds from the strategy-pivot count — workflow-improver handles those same-turn, the experiment continues.  The detector counts distinct `bug_class` values in the trailing 5 implementer rounds (rounds N-4..N). Two triggers: - PRIMARY: 3 distinct tags across the 3 most recent non-excluded   rounds. - SECONDARY: 2 distinct tags across the 2 most recent   non-excluded rounds AND at least 1   `epm:compute-deviation v1` event in the trailing 5 rounds. "Consecutive" means consecutive across non-excluded rounds: when an excluded round sits between two tagged rounds, it is skipped and the two tagged rounds count as consecutive. On fire, the orchestrator surfaces gates.conditional.whack_a_mole_pivot (id=11) with the 2-option pivot prompt.  Rationale: task #397 (2026-05-27) — distinct bug classes across rounds 8 (vllm_teardown_oom) + 9 (workflow-fix-candidate, EXCLUDED) + 10 (subprocess_wrapper_missing_upload) with compute-deviation at round 6 trigger the SECONDARY rule at the start of would-be round-10' relaunch — one round earlier than the user's manual round-11 recognition. |
| `epm:code-review` | skill (via code-reviewer) | Step 5 | PASS / CONCERNS / FAIL verdict + line-level findings against the diff. v<n> per round |
| `epm:code-review-codex` | skill (via codex-code-reviewer) | Step 5 | Codex (gpt-5.5 via companion task) twin of code-reviewer. PASS/CONCERNS/FAIL verdict + line-level findings against the same diff, same rubric. v<n> per round, paired with epm:code-review v<n>. |
| `epm:review-reconcile` | skill (via reconciler) | any review step where Claude and Codex disagree | Binding final verdict + per-finding adjudication table + rationale anchored to artifact evidence. Does NOT count toward per-reviewer round cap. Reconciler may NOT add findings beyond what either reviewer raised. |
| `epm:hf-gate-pending` | skill | Step 6a | model id, HF gate status, retry plan if rejected |
| `epm:pod-provisioned` | skill | Step 6b | RunPod pod name, host, port, GPU type, GPU count, gpu_intent |
| `epm:pod-pending` | skill | Step 6b | RunPod provision error, retry instructions |
| `epm:preflight` | skill | Step 6c | Full --json preflight report (resumed pods only) |
| `epm:launch` | skill | Step 6d | Worktree, branch, PR, pod, PID, log path, code-review verdict, WandB URL |
| `epm:run-launched` | experimenter | Step 7 | Records the actual `nohup ... &` launch (PID + timestamp); paired with epm:launch. REQUIRED payload fields (note shape): `pod=<name> pid=<pid> log_abs=<absolute_log_path> cmd='<dispatch>'`. The experimenter MUST call `os.path.abspath()` on the log path before posting AND verify with `ls -la <log_abs>` on the pod (logs must exist at that exact absolute path; relative paths burn polling-loop cycles when the orchestrator's `poll_pipeline.py` looks for them at the wrong cwd). Back-compat: a transition window accepts the legacy `log=` field as a fallback when `log_abs=` is absent; this fallback is scheduled for removal after 2026-06-15 (TODO comment in SKILL.md Step 6d.1 parse code). Rationale: task #397 (2026-05-27) — poller read `/workspace/logs/issue-397.log` while dispatcher wrote `/workspace/explore-persona-space/logs/issue-397-sweep.log`, burning 27 min of "crash diagnosis" on a healthy run. (The earlier draft of this schema required `cwd=` as well, but the orchestrator's Step 6d.1 parse never consumed it; Codex round-1 MAJOR-4 flagged it as dead payload. Dropped 2026-05-27.) |
| `epm:progress` | experimenter / implementer | during run | Milestone description + metric snapshot |
| `epm:hot-fix` | experimenter | during run | <=10-line in-line fix: commit hash, full diff, justification |
| `epm:run-finished` | experimenter | end of run | Exit code, wall time, GPU-hours used. Paired with epm:results. |
| `epm:results` | specialist | end of run | Eval JSON paths, filled reproducibility card, WandB URL, HF Hub path, commit hash, GPU-hours used, deviations, hot-fix log. MUST include a `## Sample outputs` section with `### Condition: <name>` H3 subheadings and ≥3 randomly-sampled (persona, prompt, response) triplets PER CONDITION. |
| `epm:upload-verification` | skill (via upload-verifier) | Step 8 | PASS/FAIL per artifact category, permanent URLs for each uploaded artifact |
| `epm:upload-verified` | skill | Step 8 | Sticky PASS marker; signals the auto-terminate path can run |
| `epm:upload-fix` | skill (via uploader) | Step 8 (on verifier FAIL) | Records what the uploader pushed/fixed in response to a `epm:upload-verification` FAIL. Contains: triggered-by link, verdict (COMPLETE/PARTIAL/BLOCKED), per-artifact action+URL table, lifecycle line, disk reclaimed, failures. |
| `epm:pod-terminated` | skill | Step 8 (after upload-verification PASS) | Pod auto-terminated immediately after artifact uploads verified; records pod name and command output |
| `epm:interpretation` | skill (via analyzer) | Step 9a | Fact sheet (Section 1) + interpretation (Section 2). May have v1-v3 across critique rounds. |
| `epm:interp-critique` | skill (via interpretation-critic) | Step 9a | PASS/REVISE verdict with 7 lenses: overclaims, surprises, alternatives, calibration, context, plot-prose match, statistical framing |
| `epm:interp-critique-codex` | skill (via codex-interpretation-critic) | Step 9a | Codex (gpt-5.5 via companion task) twin of interpretation-critic. PASS/REVISE verdict with all 7 lenses. v<n> per round, paired with epm:interp-critique v<n>. |
| `epm:clean-result-drafted` | skill (via analyzer) | Step 9a (final) | Link to the clean-result body.md + hero figure path + 2-sentence recap. Replaces the GH-era epm:analysis marker. |
| `epm:humanize-loop` | skill (via /humanize loop) | Step 9a-humanize | Final 6-axis scores (vocabulary, structure, rhythm, voice, interpretation honesty, results-writing discipline) for the TL;DR block. Note line: 'converged in cycle K' or 'exited at cap, residual debt: axis X' or 'skipped — /humanize skill not loaded'. Fires once per task on the first epm:clean-result-drafted v1; not re-fired on 9a-bis REVISE rounds. |
| `epm:clean-result-critique` | skill (via clean-result-critic) | Step 9a-bis | PASS/REVISE verdict with 11 lenses (title shape, TL;DR, Summary structure + register, Details per-section discipline, captions, heading-as-toggle, body-discipline anti-patterns, Source issues conditional H2, issue-link form, verify_task_body.py sanity, statistical-framing rule). |
| `epm:clean-result-critique-codex` | skill (via codex-clean-result-critic) | Step 9a-bis | Codex (gpt-5.5 via companion task) twin of clean-result-critic. Runs verify_task_body.py and audit_clean_results_body_discipline.py independently. |
| `epm:test-verdict` | skill (Step 9c, inline tests) | Step 9c | PASS / FAIL + test output summary, coverage gap notes (code-change paths only) |
| `epm:original-body` | skill (via analyzer, before in-place clean-result promotion) | Step 9a (snapshot) | Snapshot of the pre-promotion body.md, written to tasks/<status>/<N>/original-body.md and referenced in events.jsonl |
| `epm:completion-audit` | skill | Step 10 step 0 (pre-flight, before status flip) | Per-ask checklist against the ORIGINAL task body; any ☐ ⇒ status:blocked |
| `epm:results-md-diff` | skill | Step 10 | Proposed diff for RESULTS.md (for user review, not auto-applied) |
| `epm:promoted` | skill (via task.py promote) | Step 10 | User-invoked promote of an awaiting_promotion task to completed; records classification=useful\|not-useful |
| `epm:status-changed` | skill (any time) | any status transition | Records the old/new status pair and which folder move was performed |
| `epm:done` | skill | Step 10 | Final summary; records that the task reached the Done column. Task stays in tasks/completed/. |
| `epm:follow-ups` | skill (via follow-up-proposer) | Step 10b | 1-3 ranked follow-up experiment proposals, pre-filled from parent body.md |
| `epm:abort` | skill | any time | Abort reason. Triggered by status:blocked transition. |
| `epm:failure` | specialist | on crash | Traceback + last 50 log lines + partial results. SHOULD include failure_class: infra \| code line. |
| `epm:experimenter-respawn` | skill | Step 7 (failure_class=infra path) | Records re-spawn on the same branch without an implementer round. v<n> per respawn; cap 3. |
| `epm:merged` | skill | Step 10d | Merge SHAs (one per commit when rebase-merged) + worktree removal |
| `epm:merge-deferred` | skill | Step 10d | User declined the merge prompt |
| `epm:stale` | skill | Step 7 (>4h silence) | Note asking user to investigate |
| `epm:step-completed` | skill | every EXIT site | step, exit_kind (clean\|parked\|failure-exit), next_expected_step, optional notes. Consumed by the §5 re-entry router (orchestrate/resume.py). |
| `epm:workflow-fix-candidate` | orchestrator (parent assistant, /issue skill, or research-pm) on receiving a candidate block from any subagent's return text — or from the orchestrator's own observation | Any agent emits a <!-- workflow-fix-candidate v1 --> block in its return text. The orchestrator logs the candidate before deciding whether to spawn workflow-improver. Posting target: the events.jsonl of the task the emitting agent was working on; if no task, the file-based fallback .claude/cache/workflow-fix-events.jsonl. | target_file (workflow surface path, relative to repo root), bug_observed (one sentence), why_workflow_gap (one sentence), proposed_change (one sentence), diff_sketch (2-10 lines), confidence (low\|medium\|high), emitting_agent (subagent type), related_task (task ID or n/a) |
| `epm:workflow-fix-applied` | orchestrator after workflow-improver returns with reviewer PASS (or surgical change ≤10 lines, single file, no behavior change) | workflow-improver landed a diff that resolves a workflow-fix-candidate. The diff is committed on workflow-improver's worktree branch; this marker records what changed and links to the originating candidate. | files_changed (list of paths), unified_diff (full diff inline), reviewer_verdict (PASS\|skipped-surgical), commit_sha (workflow-improver's worktree commit), originating_candidate (the candidate block that triggered this) |
| `epm:workflow-fix-failed` | orchestrator when workflow-improver returns FAIL (reviewer rejected after 3 rounds, lint failed, out-of-scope deflection, or worktree spawn refused) | Same trigger as workflow-fix-applied but the workflow-improver returned with a non-PASS verdict. The candidate is preserved inline so a future pass can retry with sharper context. | failure_reason (reviewer-FAIL\|lint-FAIL\|out-of-scope\|worktree-refusal\|other), originating_candidate (verbatim), workflow_improver_report (terse summary of what was attempted and why it failed) |
<!-- /workflow.yaml: AUTO-GENERATED -->

### Notes (not auto-generated; permanent context for kinds)

- **`epm:results`** — for the `## Sample outputs` block: include
  `### Condition: <name>` H3 subheadings and >=3 randomly-sampled
  (persona, prompt, response) triplets PER CONDITION, formatted as
  fenced markdown blocks. The verifier (`scripts/verify_task_body.py`)
  enforces this on clean-result bodies; the same convention applies to
  `epm:results` for source-of-truth raw outputs.
- **`epm:failure`** — when `failure_class:` is absent, `/issue` Step 7
  falls back to log-pattern matching against `failure_patterns.md` /
  `scripts/failure_classifier.py`. Routing per the table in
  `.claude/agents/experimenter.md`.
- **`epm:follows` / `epm:followed-by`** — bidirectional parent / child
  links between tasks. Today these live as `parent_id` in `body.md`
  frontmatter; the events form is retained for migration of legacy
  bodies.
- **50,000-char `note` cap.** Every marker `note` MUST fit under 50,000
  UTF-8 characters. `task.py post-marker` errors with
  `invalid_input: note_oversize` rather than truncate. Callers that
  exceed the cap MUST split the body into N chunks and chain them via
  `part=K/N` in the marker metadata — e.g. `metadata={"part": "1/3"}`
  on the first row, `2/3` on the next, etc. The marker scanner reads
  ALL parts of the same `(kind, version)` and concatenates in `part`
  order. If the skill cannot split a body cleanly, it posts a SHORT
  `epm:failure v1` with `failure_class: infra` and
  `reason: note_oversize`, then flips status to `blocked`.

## Reviewer-loop metadata

For reviewer ensemble rows (`epm:code-review`, `epm:code-review-codex`,
`epm:interp-critique`, `epm:interp-critique-codex`,
`epm:clean-result-critique`, `epm:clean-result-critique-codex`,
`epm:review-reconcile`), the `metadata` field carries:

```json
{
  "review_pair": "interpretation",
  "round": 2,
  "reviewer": "codex-interpretation-critic",
  "verdict": "needs_targeted_fix",
  "required_fix": "Clarify whether the result supports the stated hypothesis."
}
```

Allowed `review_pair` values are `code_review`, `interpretation`, and
`clean_result`. Rounds are `1`, `2`, or `3`. Allowed verdicts are
`pass`, `needs_targeted_fix`, `blocked_needs_user_decision`, and
`fail_not_worth_continuing`.

After round 3, reviewer disagreement alone cannot block the task. The
reconciler records the final critique, chooses the minimal necessary
fix, and continues unless the missing input is a real user-decision
blocker.

## Parsing rules

To determine current state:

1. `uv run python scripts/task.py view <N> --json` -> parse.
2. `status` = the task's parent folder name. If the folder is missing
   or appears in multiple parents, abort.
3. For each `kind` above, scan `events.jsonl` for the highest-version
   row.
4. Build `marker_map: {kind: (version, note, metadata)}`.
5. Choose next action from the state machine table in `SKILL.md`.

Regex for the legacy HTML-comment-style marker (still valid in `note`
bodies for back-compat with grandfathered tasks):
`<!--\s*epm:(?P<kind>[a-z-]+)\s+v(?P<version>\d+)\s*-->`

## Example: plan marker (event row)

```jsonl
{"ts": "2026-05-20T11:00:00Z", "kind": "epm:plan", "version": 1, "note": "Plan v1 written to tasks/planning/42/plans/plan.md\n\n**Cost gate:** estimated 12 GPU-hours on 4× H100. Reply `approve` to dispatch.", "metadata": {"plan_path": "tasks/planning/42/plans/v1.md"}}
```

The `note` body should reference the plan-version file rather than
inlining the full plan (the plan body lives under `plans/v<K>.md` so
revisions are versioned independently of the event log).

## Example: clean-result-critique marker (event row)

```jsonl
{"ts": "2026-05-20T18:00:00Z", "kind": "epm:clean-result-critique", "version": 1, "note": "Clean-result-critic — PASS\n\nLens 11 (statistical framing): no overclaims; CI methodology stated.\nLens 1 (title): 'Persona collapse hero (MODERATE confidence)' agrees with body confidence sentence.\n[...]", "metadata": {"review_pair": "clean_result", "round": 1, "reviewer": "clean-result-critic", "verdict": "pass"}}
```

## Example: test-verdict marker (event row)

```jsonl
{"ts": "2026-05-20T19:00:00Z", "kind": "epm:test-verdict", "version": 1, "note": "Test Verdict — PASS\n\n**Unit tests:** 51 passed, 0 failed, 1 skipped\n**Integration tests:** skipped (no pod assigned)\n**Lint:** PASS (ruff check + format)\n**Coverage gaps:** none", "metadata": {"unit_passed": 51, "unit_failed": 0, "lint": "PASS"}}
```
