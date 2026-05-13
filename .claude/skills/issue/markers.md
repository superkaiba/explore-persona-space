# Comment Marker Taxonomy

All structured workflow state in Sagan is carried in workflow_event marker
bodies. This file is the source of truth for marker syntax and semantics.

## Format

```markdown
<!-- epm:<kind> v<n> -->
## Human-readable title
<body>
<!-- /epm:<kind> -->
```

- `epm` = "explore persona manager" namespace (shared prefix, keeps our markers
  out of conflict with other tools).
- `<kind>` = one of the kinds below.
- `v<n>` = monotonic version. `v1` is the original; `v2+` are revisions. The
  skill parses ALL markers and uses the highest-version one per `<kind>` as
  authoritative.
- Opening and closing tags must match (`<!-- epm:plan v1 -->` ... `<!-- /epm:plan -->`).

**Never edit or delete** a marker comment. History is part of the audit trail.

## Kinds

The kinds table is auto-generated from `.claude/workflow.yaml § markers`. Do NOT edit inside the fence; run `uv run python scripts/workflow_lint.py --emit-tables` to regenerate after a YAML edit. The full per-kind documentation (with longer required-fields prose) lives in `workflow.yaml § markers` itself; the table below is the at-a-glance index.

<!-- workflow.yaml: AUTO-GENERATED (marker-kinds) -->
| Kind | Posted by | When | Required fields |
|------|-----------|------|-----------------|
| `auto-defaults` | skill | Step 0b | what fields were auto-filled and the inferred values |
| `clarify` | skill | Step 1 | numbered questions OR 'No blocking ambiguities' |
| `clarify-answers` | skill (relaying user chat reply) | Step 1 | user's answers to the most recent epm:clarify questions |
| `plan` | skill (via adversarial-planner) | Step 2 | Goal, method delta, reproducibility card, success/kill criteria, GPU-hr estimate, pod |
| `consistency` | skill (via consistency-checker) | Step 2b | PASS/WARN/BLOCK verdict, variables that differ from parent, shared baseline check |
| `proposed-tests` | skill (via implementer / experiment-implementer) | Step 4b, TDD mode only | Posted ONLY when the plan body contains `### TDD: yes` or the user requests TDD. Body: ≥1 happy-path + ≥2 distinct error/edge-case tests, behavior-focused, in fenced blocks. Implementer EXITs after posting and awaits user `approve-tests` reply before writing implementation. Skill does not advance state on this marker — it pauses pending approval, then resumes Step 4b normal flow. |
| `experiment-implementation` | skill (via experiment-implementer) | Step 4b | Posted for `type:experiment` issues. Body MUST follow the four-section shape `(a) What was done` / `(b) Considered but not done` / `(c) How to verify` / `(d) Needs human eyeball` — see `.claude/agents/experiment-implementer.md` Report Format. (c) MUST contain a copy-pasteable reproduction command + the observable success signal so the user can verify without reading the diff. |
| `code-review` | skill (via code-reviewer) | Step 5 | PASS / CONCERNS / FAIL verdict + line-level findings against the diff. v<n> per round |
| `code-review-codex` | skill (via codex-code-reviewer) | Step 5 | Codex (gpt-5.5 via companion task) twin of code-reviewer. PASS/CONCERNS/FAIL verdict + line-level findings against the same diff, same rubric. v<n> per round, paired with epm:code-review v<n>. Posted to Sagan by the Claude wrapper agent. |
| `code-review-reconcile` | skill (via reconciler) | Step 5 (only on PASS/FAIL disagreement between epm:code-review v<n> and epm:code-review-codex v<n>) | Binding final verdict (PASS\|FAIL) + per-finding adjudication table + rationale anchored to artifact evidence. CONCERNS folds into PASS. Does NOT count toward the per-reviewer round cap. Reconciler may NOT add findings beyond what either reviewer raised. |
| `hf-gate_pending` | skill | Step 6a | model id, gate status, retry plan if rejected |
| `pod-pending` | skill | Step 6b | RunPod provision error, retry instructions |
| `preflight` | skill | Step 6c | Full --json preflight report (resumed pods only) |
| `launch` | skill | Step 6d | Worktree, branch, PR, pod, PID, log path, code-review verdict, WandB URL |
| `progress` | experimenter / implementer | during run | Milestone description + metric snapshot |
| `hot-fix` | experimenter | during run | <=10-line in-line fix: commit hash, full diff, justification |
| `results` | specialist | end of run | Eval JSON paths, filled reproducibility card, WandB URL, HF Hub path, commit hash, GPU-hours used, deviations, hot-fix log. **MUST include a `## Sample outputs` section with `### Condition: <name>` H3 subheadings and ≥3 randomly-sampled (persona, prompt, response) triplets PER CONDITION, formatted as fenced markdown blocks. Verifier check #11 enforces.** For `type:infra` / `type:survey` paths (implementer's completion report), the body MUST follow the four-section shape `(a) What was done` / `(b) Considered but not done` / `(c) How to verify` / `(d) Needs human eyeball` — see `.claude/agents/implementer.md` Report Format. (c) MUST contain copy-pasteable reproduction commands so the user can verify without reading the diff. |
| `upload-verification` | skill (via upload-verifier) | Step 8 | PASS/FAIL per artifact category, permanent URLs for each uploaded artifact. |
| `upload-fix` | skill (via uploader) | Step 8 (on verifier FAIL) | Records what the uploader pushed/fixed in response to a `epm:upload-verification` FAIL. Contains: triggered-by link, verdict (COMPLETE/PARTIAL/BLOCKED), per-artifact action+URL table, lifecycle line (resumed→uploads→stopped), disk reclaimed, failures. Followed by re-run of `upload-verifier` which posts a fresh `epm:upload-verification v2`. |
| `interpretation` | skill (via analyzer) | Step 9a | Fact sheet (Section 1) + interpretation (Section 2). May have v1-v3 across critique rounds. |
| `interp-critique` | skill (via interpretation-critic) | Step 9a | PASS/REVISE verdict with 5 lenses: overclaims, surprises, alternatives, calibration, context |
| `interp-critique-codex` | skill (via codex-interpretation-critic) | Step 9a | Codex (gpt-5.5 via companion task) twin of interpretation-critic. PASS/REVISE verdict with all 7 lenses (Codex multimodal handles lens 6 plot-prose match). v<n> per round, paired with epm:interp-critique v<n>. Posted to Sagan by the Claude wrapper agent. |
| `interp-critique-reconcile` | skill (via reconciler) | Step 9a (only on PASS/REVISE disagreement between epm:interp-critique v<n> and epm:interp-critique-codex v<n>) | Binding final verdict (PASS\|REVISE) + per-finding adjudication table + rationale anchored to artifact evidence (interpretation body / JSON paths / figure paths). Does NOT count toward per-reviewer round cap. Reconciler may NOT add findings beyond what either reviewer raised. |
| `clean-result-critique` | skill (via clean-result-critic) | Step 9a-bis | PASS/REVISE verdict with 11 lenses covering title shape, TL;DR user-voice register, Summary six-bullet structure + LW register, Details per-section discipline (setup-before-figure, visible captions, sample outputs), heading-as-toggle convention, body-discipline anti-patterns (via audit_clean_results_body_discipline.py), Source issues conditional H2, issue-reference link form, verifier sanity (verify_clean_result.py), and statistical-framing rule (no effect sizes / named tests / power analyses / `value ± err` credence intervals in prose; absorbed from the retired `reviewer` agent 2026-05-13). Runs after `interp-critique` PASSes — content honesty first, structure + register + statistical framing second. **Final critic before status:awaiting_promotion as of 2026-05-13** (the previous dedicated reviewer step was retired). |
| `clean-result-critique-codex` | skill (via codex-clean-result-critic) | Step 9a-bis (ROUND 1 ONLY) | Codex (gpt-5.5 via companion task) twin of clean-result-critic. PASS/REVISE verdict with the same 11 lenses (10 structural + statistical-framing rule). Spawned in parallel with the Claude critic on round 1 only; rounds 2-3 run Claude only. Independently runs `verify_sagan_card.py` and treats FAIL as Critical. Posted to Sagan by the Claude wrapper agent. |
| `clean-result-critique-reconcile` | skill (via reconciler) | Step 9a-bis (only on PASS/REVISE disagreement between epm:clean-result-critique v1 and epm:clean-result-critique-codex v1) | Binding final verdict (PASS\|REVISE) + per-finding adjudication table + rationale anchored to artifact evidence (clean-result body / verifier output / audit-script output). Does NOT count toward per-reviewer round cap. Reconciler may NOT add findings beyond what either reviewer raised. |
| `analysis` | skill (via analyzer) | Step 9a (final) | Link to created clean-result issue + hero figure URL + 2-sentence recap |
| `reviewer-verdict` | skill (via reviewer) — DEPRECATED 2026-05-13 | Step 9b (retired) | DEPRECATED. The dedicated reviewer step was retired and its responsibilities (statistical-framing rule + final published-body fresh-context check) were absorbed by `clean-result-critic` (Step 9a-bis). This marker kind is kept in the schema for legacy issue state recovery only. |
| `reviewer-verdict-codex` | skill (via codex-reviewer) — DEPRECATED 2026-05-13 | Step 9b (retired) | DEPRECATED — see reviewer-verdict. |
| `reviewer-verdict-reconcile` | skill (via reconciler) — DEPRECATED 2026-05-13 | Step 9b (retired) | DEPRECATED — see reviewer-verdict. |
| `test-verdict` | skill (Step 9c, inline tests) | Step 9c | PASS / FAIL + test output summary, coverage gap notes (code-change paths only) |
| `completion-audit` | skill | Step 10 step 0 (pre-flight, before label flip) | Per-ask checklist; any ☐ ⇒ status:blocked |
| `results-md-diff` | skill | Step 10 | Proposed diff for RESULTS.md (for user review, not auto-applied) |
| `done` | skill | Step 10 | Final summary; records which Done column. Issue stays OPEN. |
| `follow-ups` | skill (via follow-up-proposer) | Step 10b | 1-3 ranked follow-up experiment proposals, pre-filled from parent |
| `pod-terminated` | skill | Step 8 (after upload-verification PASS) | Pod auto-terminated immediately after artifact uploads verified; records pod name and command output. |
| `abort` | skill | any time | Abort reason. Triggered by status:blocked label. |
| `failure` | specialist | on crash | Traceback + last 50 log lines + partial results. SHOULD include failure_class: infra \| code line. |
| `experimenter-respawn` | skill | Step 7 (failure_class=infra path) | Records re-spawn on the same branch without an implementer round. v<n> per respawn; cap 3. |
| `merged` | skill | Step 10d | Merge SHAs (one per commit when rebase-merged) + worktree removal |
| `merge-deferred` | skill | Step 10d | User declined the merge prompt |
| `stale` | skill | Step 7 (>4h silence) | Note asking user to investigate |
| `follows` | issue author (manual) | manually on a new follow-up issue | `Follows from: #<N>` — bidirectional with epm:followed-by |
| `followed-by` | issue author (manual) | manually on the parent issue | `Followed by: #<N>` — bidirectional with epm:follows |
| `clean-result-lint` | Sagan verifier | manual or /issue-triggered `verify_sagan_card.py` check on a clean-result draft | PASS or FAIL verdict + verifier stdout/stderr summary |
<!-- /workflow.yaml: AUTO-GENERATED -->

### Notes (not auto-generated; permanent context for kinds)

- **`results`** — for the `## Sample outputs` block: include `### Condition: <name>` H3 subheadings and ≥3 randomly-sampled (persona, prompt, response) triplets PER CONDITION, formatted as fenced markdown blocks. Verifier check #11 enforces.
- **`failure`** — when `failure_class:` is absent, `/issue` Step 7 falls back to log-pattern matching against `failure_patterns.md`. Routing per the table in `.claude/agents/experimenter.md`.
- **`follows` / `followed-by`** — auto-posting is deferred to a follow-up `type:infra` issue.
- **Workflow-event body cap.** Every marker body MUST fit under Sagan's workflow-event body limit. Callers that exceed the cap MUST split the body into N events and chain them via `part=K/N` in the marker title — e.g. `<!-- epm:plan v3 part=1/3 -->` ... `<!-- /epm:plan -->`, then `<!-- epm:plan v3 part=2/3 -->` ... and so on. The marker scanner reads ALL parts of the same `(kind, version)` and concatenates in `part` order. If the skill cannot split a body cleanly, it posts a SHORT `epm:failure v1` with `failure_class: infra` and `reason: marker_oversize`, then flips the source experiment to `status:blocked`.

## Parsing rules

To determine current state:

1. `python scripts/sagan_state.py view <N>` → JSON with `.experiment.status` and `.events[]`.
2. `status` = `.experiment.status` (already a single enum value).
3. For each `<kind>` above, scan `.events[]` for the highest-version `metadata.marker_type` matching `epm:<kind>`.
4. Build `marker_map: {kind: (version, note)}` where `note` is `events[*].note`.
5. Choose next action from the state machine table in `SKILL.md`.

Regex for marker opening: `<!--\s*epm:(?P<kind>[a-z-]+)\s+v(?P<version>\d+)\s*-->`

## Example: plan marker

```markdown
<!-- epm:plan v1 -->
## Approved Plan for #42

**Cost gate:** estimated 12 GPU-hours on 4× H100. Reply `approve` to dispatch.

### Goal
...

### Hypothesis
...

### Method delta vs. baseline (exp #30)
...

### Reproducibility Card
| Category | Parameter | Value |
|----------|-----------|-------|
...

### Success / Kill criteria
- Success: effect size > 0.3 with p < 0.01 across 3 seeds
- Kill: no significant difference in either direction

### Plan deviations
- Allowed without asking: seed changes, minor LR adjustments +/-20%
- Must ask: dataset changes, eval metric changes, pod changes

### Command to reproduce
```
nohup python scripts/train.py condition=... seed=42 > /workspace/logs/issue-42.log 2>&1 &
```
<!-- /epm:plan -->
```

## Example: reviewer-verdict marker (DEPRECATED 2026-05-13)

> **DEPRECATED.** The dedicated reviewer step was retired and its
> responsibilities were absorbed by `clean-result-critic` (Step 9a-bis,
> Lens 11). New issues never post `reviewer-verdict` markers. This
> example is kept for legacy state-recovery context — if you encounter
> an old issue with an `epm:reviewer-verdict` marker, it predates
> 2026-05-13.

```markdown
<!-- epm:reviewer-verdict v1 -->
## Reviewer Verdict — PASS with CONCERNS

**Verdict:** PASS

**Concerns (non-blocking):**
- Single seed (42). Claim "robust across seeds" is overclaimed — only one seed was
  actually run (see `epm:results`). Either weaken claim or run 2 more seeds.
- Baseline comparison uses issue #30 results, which were under slightly different
  compute allocation (4 GPUs vs 8 GPUs now). Should re-run baseline or qualify.

**Verified:**
- Numerical claims match `eval_results/issue-42/run_result.json`
- Reproducibility card complete
- No overclaims beyond single-seed issue above

**Recommendation:** merge with weakened seed claim, or run 2 more seeds and update.
<!-- /epm:reviewer-verdict -->
```

## Example: test-verdict marker

```markdown
<!-- epm:test-verdict v1 -->
## Test Verdict — PASS

**Unit tests:** 51 passed, 0 failed, 1 skipped
**Integration tests:** skipped (no pod assigned)
**Lint:** PASS (ruff check + format)
**Coverage gaps:** none

<details>
<summary>Full test output</summary>

[truncated pytest output, last 100 lines]

</details>
<!-- /epm:test-verdict -->
```
