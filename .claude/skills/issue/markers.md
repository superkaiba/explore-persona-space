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
| `clarify` | skill | Step 1 | Numbered questions OR "No blocking ambiguities". |
| `gate` | skill (via gate-keeper) | Step 2 | Scores (1-5 x 5 dims), verdict, rationale. |
| `plan` | skill (via adversarial-planner) | Step 3 | Goal, method delta, reproducibility card, success/kill criteria, GPU-hr estimate, pod. |
| `pod-pending` | skill | Step 5c | List of available pods, target pod busy status. |
| `preflight` | skill | Step 5d | Full `--json` preflight report. |
| `launch` | skill | Step 5e | Worktree, branch, PR, pod, PID, log path, WandB URL (best-effort). |
| `progress` | specialist | during run | Milestone description + metric snapshot. |
| `results` | specialist | end of run | Eval JSON paths, filled reproducibility card, WandB URL, HF Hub path, commit hash, GPU-hours used, deviations. |
| `analysis` | skill (via analyzer) | Step 7a | Analyzer draft following `templates/experiment_report.md`. |
| `reviewer-verdict` | skill (via reviewer / code-reviewer) | Step 7b | PASS / CONCERNS / FAIL + line-level issues. |
| `test-verdict` | skill (via tester) | Step 7c | PASS / FAIL + test output summary, coverage gap notes. |
| `results-md-diff` | skill | Step 8 | Proposed diff for RESULTS.md (for user review, not auto-applied). |
| `closed` | skill | Step 8 | Final summary: outcome, numbers, what's confirmed/falsified, next steps. |
| `abort` | skill | any time | Abort reason. Triggered by `status:blocked` label. |
| `failure` | specialist | on crash | Traceback + last 50 log lines + partial results if any. |
| `stale` | skill | Step 6 (>4h silence) | Note asking user to investigate. |

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
