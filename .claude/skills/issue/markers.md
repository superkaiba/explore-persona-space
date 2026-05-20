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
| `epm:clarify` | skill | Step 1 | numbered clarifying questions OR 'No blocking ambiguities' |
| `epm:clarify-questions` | skill | Step 1 | alias of epm:clarify used by some helper scripts; same body shape |
| `epm:clarify-answers` | skill (relaying user chat reply) | Step 1 | user's answers to the most recent epm:clarify questions, persisted to events.jsonl |
| `epm:plan` | skill (via adversarial-planner) | Step 2 | Goal, method delta, reproducibility card, success/kill criteria, GPU-hr estimate, pod intent |
| `epm:plan-approved` | skill (relaying user reply) | Step 2c | user `approve` reply; advances status from plan_pending to approved |
| `epm:consistency` | skill (via consistency-checker) | Step 2b | PASS/WARN/BLOCK verdict, variables that differ from parent, shared baseline check |
| `epm:proposed-tests` | skill (via implementer / experiment-implementer) | Step 4b, TDD mode only | Posted ONLY when the plan body contains `### TDD: yes` or the user requests TDD. Body: ≥1 happy-path + ≥2 distinct error/edge-case tests, behavior-focused, in fenced blocks. Implementer EXITs after posting and awaits user `approve-tests` reply. Skill pauses pending approval, then resumes Step 4b. |
| `epm:experiment-implementation` | skill (via experiment-implementer) | Step 4b | Posted for type=experiment tasks. Body MUST follow the four-section shape `(a) What was done` / `(b) Considered but not done` / `(c) How to verify` / `(d) Needs human eyeball`. |
| `epm:code-review` | skill (via code-reviewer) | Step 5 | PASS / CONCERNS / FAIL verdict + line-level findings against the diff. v<n> per round |
| `epm:code-review-codex` | skill (via codex-code-reviewer) | Step 5 | Codex (gpt-5.5 via companion task) twin of code-reviewer. PASS/CONCERNS/FAIL verdict + line-level findings against the same diff, same rubric. v<n> per round, paired with epm:code-review v<n>. |
| `epm:review-reconcile` | skill (via reconciler) | any review step where Claude and Codex disagree | Binding final verdict + per-finding adjudication table + rationale anchored to artifact evidence. Does NOT count toward per-reviewer round cap. Reconciler may NOT add findings beyond what either reviewer raised. |
| `epm:hf-gate-pending` | skill | Step 6a | model id, HF gate status, retry plan if rejected |
| `epm:pod-provisioned` | skill | Step 6b | RunPod pod name, host, port, GPU type, GPU count, gpu_intent |
| `epm:pod-pending` | skill | Step 6b | RunPod provision error, retry instructions |
| `epm:preflight` | skill | Step 6c | Full --json preflight report (resumed pods only) |
| `epm:launch` | skill | Step 6d | Worktree, branch, PR, pod, PID, log path, code-review verdict, WandB URL |
| `epm:run-launched` | experimenter | Step 7 | Records the actual `nohup ... &` launch (PID + timestamp); paired with epm:launch |
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
