---
title: Fix ~19 pre-existing failing tests on main (verifier/migrate-body/redact/plan-handoff
  harness drift)
kind: infra
tags: []
created_at: '2026-05-28T21:13:00Z'
has_clean_result: false
---
# Fix ~19 pre-existing failing tests on `main` (verifier / migrate-body / redact / plan-handoff harness drift)

`main` is currently red on ~19 tests that are **unrelated to the recurring-bug-fix batch** (merge `5136f3d8`). They were verified pre-existing: the integration capstone ran the identical set against a pristine pre-batch `main` and got byte-identical failures, and every failing test file + its source-under-test is unchanged by the batch. This task tracks fixing them so `main` is green.

## Failing clusters (as of 2026-05-28, `uv run pytest`)

1. **`tests/test_verify_clean_result.py` (10)** — `test_good_body_passes`, `test_missing_subsection_fails`, `test_sample_outputs_too_few_fenced_blocks_fails`, `test_canonical_template_sample_outputs_passes`, `test_methodology_bullets_present_passes`, `test_methodology_prose_fails_strict_post_cutoff`, `test_methodology_prose_passes_pre_cutoff`, `test_background_motivation_present_passes`, `test_two_figures_both_captioned_passes`, `test_caption_multiline_html_comment_does_not_leak`. The clean-result body verifier (`verify_clean_result.py`) and its fixtures have drifted apart (the new markdown clean-result spec moved checks around). Decide per-test whether the verifier or the fixture/expectation is stale.

2. **`tests/test_task_workflow.py::test_migrate_body_*` (6)** — `test_migrate_body_classify_pass`, `_conformant_failing_remediation`, `_dry_run_does_not_write`, `_idempotency`, `_pass_body_is_noop`, `_report_classification`. `classify_body` returns CONFORMANT_FAILING for a fixture the test expects to PASS — the migrate-body classifier drifted from the current spec.

3. **`tests/test_redact_for_gist.py` (2)** — `test_full_fixture_redacted`, `test_idempotent`. Redaction output no longer matches the expected fixture.

4. **`tests/test_plan_handoff_path_convention.py::test_claude_md_contains_plan_handoff_rule` (1)** — asserts the literal string `'Plan handoff convention'` appears in `CLAUDE.md`, but the actual rule is worded `**Plan handoff:**`. Either reword the rule or relax the test's expected string.

(`test_hub.py::test_upload_file` and `test_step_completed_resume.py` were on the earlier pre-batch list but now PASS — test_hub via the data-eval mock fix that merged; recount before starting.)

## Scope / approach
- For each cluster, determine whether the TEST or the SOURCE is stale (the spec churned — verifier checks 6→11→16, Lens 11/12/13, check 11b — so several are likely test-fixture drift, not real regressions). Fix the side that's wrong; do not just delete tests.
- `tests/test_data_validation.py` had a collection error (`ModuleNotFoundError: explore_persona_space.data`) on the pre-batch list — confirm whether that module is intended to exist or the test is obsolete.
- Acceptance: `uv run pytest tests/` green (excluding any genuinely-obsolete tests that are removed with justification).

## Provenance
Surfaced by the recurring-bug-fix workflow's integration capstone (2026-05-28); split out as a separate task because it predates and is orthogonal to that batch.
