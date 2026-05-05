# Comment Marker Taxonomy

All structured state on a GitHub issue is carried in HTML-comment-wrapped
markers. This file is the source of truth for marker syntax and semantics.

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

| Kind | Posted by | When | Required fields |
|------|-----------|------|-----------------|
| `auto-defaults` | skill | Step 0b | Records that the skill auto-filled missing `status:*`, `type:*`, or body. Lists what was filled and the inferred values. |
| `clarify` | skill | Step 1 | Numbered questions OR "No blocking ambiguities". |
| `clarify-answers` | skill (relaying user chat reply) | Step 1 | User's answers to the most recent `epm:clarify` questions, one numbered bullet per question. Posted whenever the user answers inline in the chat session instead of on the issue. |
| `plan` | skill (via adversarial-planner) | Step 2 | Goal, method delta, reproducibility card, success/kill criteria, GPU-hr estimate, pod. |
| `consistency` | skill (via consistency-checker) | Step 2b | PASS/WARN/BLOCK verdict, variables that differ from parent, shared baseline check. |
| `experiment-implementation` | skill (via experiment-implementer) | Step 4b | Files changed, diff stat, plan adherence, lint+dry-run results, assumptions, reviewer focus, branch + PR URL. Posted for `type:experiment` issues. |
| `code-review` | skill (via code-reviewer) | Step 5 | PASS / CONCERNS / FAIL verdict + line-level findings against the diff. v<n> per round. |
| `hf-gate-pending` | skill | Step 6a | Records that an HF gated-model auto-acceptance was triggered (model id, gate status, retry plan if rejected). |
| `pod-pending` | skill | Step 6b | RunPod provision error, retry instructions. |
| `preflight` | skill | Step 6c | Full `--json` preflight report (resumed pods only). |
| `launch` | skill | Step 6d | Worktree, branch, PR, pod, PID, log path, code-review verdict, WandB URL (best-effort). |
| `progress` | experimenter / implementer | during run | Milestone description + metric snapshot. |
| `hot-fix` | experimenter | during run | <=10-line in-line fix applied during a run: commit hash, full diff, justification. Anything bigger triggers `epm:failure` + bounce-back to `status:implementing`. |
| `results` | specialist | end of run | Eval JSON paths, filled reproducibility card, WandB URL, HF Hub path, commit hash, GPU-hours used, deviations, hot-fix log. For `type:infra` paths the implementer's completion report. **MUST include a `## Sample outputs` section with `### Condition: <name>` H3 subheadings and ≥3 randomly-sampled (persona, prompt, response) triplets PER CONDITION, formatted as fenced markdown blocks. Verifier check #11 enforces.** |
| `upload-verification` | skill (via upload-verifier) | Step 8 | PASS/FAIL per artifact category, permanent URLs for each uploaded artifact. |
| `interpretation` | skill (via analyzer) | Step 9a | Fact sheet (Section 1) + interpretation (Section 2). May have v1-v3 across critique rounds. |
| `interp-critique` | skill (via interpretation-critic) | Step 9a | PASS/REVISE verdict with 5 lenses: overclaims, surprises, alternatives, calibration, context. |
| `analysis` | skill (via analyzer) | Step 9a (final) | Link to created clean-result issue + hero figure URL + 2-sentence recap. The full clean-result body lives on the new issue, not in this marker. |
| `reviewer-verdict` | skill (via reviewer) | Step 9b | PASS / CONCERNS / FAIL + line-level issues. |
| `test-verdict` | skill (Step 9c, inline tests) | Step 9c | PASS / FAIL + test output summary, coverage gap notes. Code-change paths only (`type:infra` / `type:analysis` / `type:survey`). The skill runs the project's test suite directly — there is NO separate `tester` agent. |
| `results-md-diff` | skill | Step 10 | Proposed diff for RESULTS.md (for user review, not auto-applied). |
| `done` | skill | Step 10 | Final summary: outcome, numbers, what's confirmed/falsified, next steps. Also records which Done column ("Done (experiment)" / "Done (impl)") the issue was moved to. Issue stays OPEN. |
| `follow-ups` | skill (via follow-up-proposer) | Step 10b | 1-3 ranked follow-up experiment proposals, pre-filled from parent. |
| `pod-terminated` | skill | Step 10c | User opted to terminate the issue's ephemeral pod; records the pod name and final volume disposition. |
| `pod-kept-stopped` | skill | Step 10c | User declined to terminate the pod; it remains stopped and parked indefinitely. Records the pod name so resume / future-cleanup logic can find it. |
| `abort` | skill | any time | Abort reason. Triggered by `status:blocked` label. |
| `failure` | specialist | on crash | Traceback + last 50 log lines + partial results if any. From the experimenter, also includes a bounce-back proposal if the failure is a structural code issue rather than a runtime blip. **SHOULD include a `failure_class:` line at the top of the body, value `infra` or `code`.** When absent, `/issue` Step 7 falls back to log-pattern matching against `.claude/skills/issue/failure_patterns.md`. Routing per the table in `.claude/agents/experimenter.md`. |
| `experimenter-respawn` (`epm:experimenter-respawn`) | skill | Step 7 (failure_class=infra path) | Records that the skill re-spawned the experimenter on the same branch without an implementer round. v<n> increments per respawn; cap at 3, then `status:blocked`. |
| `merged` | skill | Step 10d | Records merge SHAs (one per commit when rebase-merged) + worktree removal. |
| `merge-deferred` | skill | Step 10d | Records that the user declined the merge prompt. |
| `stale` | skill | Step 7 (>4h silence) | Note asking user to investigate. |

## Parsing rules

To determine current state:

1. `gh issue view <N> --json labels,comments` -> parse.
2. `status` = the single `status:*` label. If 0 or >1, abort.
3. For each `<kind>` above, scan comments for the highest-version opening tag.
4. Build `marker_map: {kind: (version, body)}`.
5. Choose next action from the state machine table in `SKILL.md`.

Regex for marker opening: `<!--\s*epm:(?P<kind>[a-z-]+)\s+v(?P<version>\d+)\s*-->`

## Example: plan marker

```markdown
<!-- epm:plan v1 -->
## Approved Plan for #42

**Cost gate:** estimated 12 GPU-hours on pod3 (8xH100). Reply `approve` to dispatch.

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

## Example: reviewer-verdict marker

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
