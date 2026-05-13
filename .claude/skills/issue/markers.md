# Sagan Workflow Markers

Markers are Sagan `workflow_events` rows. Do not use any other durable workflow
state surface, do not write local state files as truth, and do not write
directly to the Sagan database.

Use the HTTP client:

```bash
python scripts/sagan_state.py post-marker <experiment_number> epm:plan --note "Plan drafted"
```

Canonical markers:

- `epm:clarify`
- `epm:plan`
- `epm:code-review`
- `epm:code-review-codex`
- `epm:review-reconcile`
- `epm:preflight`
- `epm:launch`
- `epm:progress`
- `epm:results`
- `epm:upload-verification`
- `epm:interpretation`
- `epm:interp-critique`
- `epm:interp-critique-codex`
- `epm:clean-result-critique`
- `epm:clean-result-critique-codex`
- `epm:completion-audit`
- `epm:done`
- `epm:failure`

Reviewer-loop metadata is compact JSON:

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
`clean_result`. Rounds are `1`, `2`, or `3`. Allowed verdicts are `pass`,
`needs_targeted_fix`, `blocked_needs_user_decision`, and
`fail_not_worth_continuing`.

After round 3, reviewer disagreement alone cannot block the experiment. The
reconciler records the final critique, chooses the minimal necessary fix, and
continues unless the missing input is a real user-decision blocker.
