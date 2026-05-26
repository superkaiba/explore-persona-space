---
title: 'Why-this-experiment gate hardening: fence coverage + m1/m3/m4/m5 + style (follow-up
  to #371)'
kind: infra
tags:
- workflow
- gate-hardening
created_at: '2026-05-20T21:02:42Z'
has_clean_result: false
parent_id: 371
---
# Why-this-experiment gate hardening: fence coverage + m1/m3/m4/m5 + style
## Goal

Close six follow-up items deferred from #371 code-reviews. Each is small and independent; bundle into one task for review-cost amortization.

The six items, with concrete fixes:

### 1. Tilde-fence (`~~~`) coverage

`scripts/verify_task_body.py::find_h2_sections` and `scripts/task.py::_enforce_why_this_experiment_gate` both track a fence-state toggle that currently matches only triple-backtick fences. Extend the matcher:

```python
in_fence = ... toggle on lines whose stripped form starts with ("```", "~~~") ...
```

Five-line change in each location. Indented-code-block (4-space) coverage is OUT of scope for this task — they're rare in real task bodies and add complexity disproportionate to risk.

Add one test in `tests/test_verify_task_body.py`: body where the `## Why this experiment` section sits inside a `~~~text … ~~~` block → check #12 FAILs.

### 2. Lazy `import yaml as _yaml` style

In `src/explore_persona_space/task_workflow_migrate.py`, the `import yaml as _yaml` currently lives inside the `_serialize_frontmatter` helper. Move it to module top alongside the existing imports.

### m1 — Malformed YAML silently bucketed

`scripts/migrate_add_legacy_why_sentinel.py::split_frontmatter` (and equivalents wherever they live in `src/` or `scripts/`) returns `(None, fm_block, body)` for both "no frontmatter delimiters" AND "YAML parse error". The main migration loop buckets both as `skipped_no_fm`.

Fix:
- Distinguish the two cases. Either return a 3-state result (`("missing" | "parse_error" | "ok"), fm_or_none, body)`), or raise `yaml.YAMLError` and let the caller catch it.
- The migration loop routes parse errors to a new `parse_errors` bucket. `--apply` refuses to commit if `parse_errors > 0`; the user must fix the offending bodies first.
- Print the affected paths in both dry-run and apply mode so the user knows which bodies to inspect.

Test: dry-run on a synthetic body with intentionally-broken YAML (e.g. unbalanced quote) → reports the body under `parse errors`, with file path; `--apply` refuses with non-zero exit and a clear error.

### m3 — Duplicate constants between `task.py` and `verify_task_body.py`

The four label regex constants (`_WHY_LABELS`, `_WHY_LINE_RE`, `_WHY_MIN_LINE_CHARS`, `WHY_LINE_LABELS`, `WHY_SECTION_NAME`, `APPLICATION_ENUM`, `MIN_WHY_LINE_CHARS`) currently exist in both `scripts/task.py` (private prefixed names) and `scripts/verify_task_body.py` (public names). Comment at the duplication site reads "Edit both together if changing" — that's the smell.

Fix:
- Extract to `src/explore_persona_space/task_workflow/why_gate.py` (natural home alongside the existing `task_workflow` package).
- Module exports: `WHY_SECTION_NAME`, `WHY_LINE_LABELS`, `APPLICATION_ENUM`, `MIN_WHY_LINE_CHARS`, `LEGACY_WHY_SENTINEL_KEY`, plus a `find_why_section(body: str) -> WhySection | None` helper that BOTH call sites can use (returns parsed labels + fence-state-aware section start/end).
- `scripts/task.py` and `scripts/verify_task_body.py` import the constants and the helper. Delete the duplicated definitions.
- Keep behavior identical — this is pure DRY refactor.

Verify the existing `tests/test_verify_task_body.py` and `tests/test_task_workflow.py` tests all still pass without modification. If a test fails, that means the two sources WERE NOT identical and the gate had a quiet behavioral drift; report and pause.

### m4 — SKILL.md `set-body` mechanics

`.claude/skills/why-experiment-gate/SKILL.md` Step 4 instructs the agent to write the full markdown (frontmatter + body) to a file and call `task.py set-body --file <path>`. The brief was unverified — confirm what `task.py set-body` actually writes:

```bash
grep -A30 "def cmd_set_body\|def set_body" scripts/task.py src/explore_persona_space/task_workflow.py
```

Three possible outcomes:
- (a) `set-body` accepts full markdown including frontmatter and writes it as-is → no fix needed, the skill instruction is correct.
- (b) `set-body` writes only the body portion (strips/ignores frontmatter from the input) → fix the skill instruction to call `set-frontmatter` separately, OR add a new `set-body --replace-all` flag, OR document the existing pattern.
- (c) `set-body` overwrites frontmatter destructively → the skill is currently dangerous; either change the skill or change the CLI. Decide based on what existing callers (`/issue` Step 9a auto-promote, `analyzer` agent) expect.

Document the resolution in the SKILL.md prose AND in the CLI's `--help` output if it isn't already obvious.

### m5 — Duplicate H2 sections silently accepted

`scripts/verify_task_body.py::find_h2_sections` (or wherever the section-walk lives) returns the first match for a given heading and silently ignores duplicates. The current behavior is documented in a comment as a "body-discipline smell" but the verifier doesn't enforce it.

Fix: if two `## Why this experiment` sections appear in the same body, check #12 FAILs with a clear message ("multiple `## Why this experiment` sections found at lines X, Y"). Lift the policy from "smell" to "FAIL".

Test: synthetic body with two `## Why this experiment` H2s → check #12 FAILs.

## Files touched (estimate)

- `src/explore_persona_space/task_workflow/why_gate.py` (NEW, ~80 lines) — extracted constants + `find_why_section` helper.
- `scripts/task.py` (-15/+8 lines) — import from `why_gate`, drop duplicated constants, accept `~~~` fences via the shared helper.
- `scripts/verify_task_body.py` (-25/+10 lines) — same. Plus duplicate-H2 FAIL logic.
- `scripts/migrate_add_legacy_why_sentinel.py` (~+25 lines) — distinguish parse-error from no-fm; refuse `--apply` if any.
- `src/explore_persona_space/task_workflow_migrate.py` (~3 line move) — yaml import to module top.
- `.claude/skills/why-experiment-gate/SKILL.md` (~5-15 line edit depending on m4 resolution).
- `tests/test_verify_task_body.py` (+3 tests: tilde-fence FAIL, duplicate-H2 FAIL, parse-error reporting if shared with migrate).
- `tests/test_task_workflow.py` (+1 test for m1 if migrate tests live there).

## Test plan

Run from repo root:

1. `uv run pytest tests/test_verify_task_body.py tests/test_task_workflow.py tests/test_workflow_lint.py tests/test_workflow_yaml.py -v` → all PASS (no regression of the 119 existing tests).
2. New tests added in (1) all PASS:
   - `test_why_experiment_tilde_fence_bypass_fails`
   - `test_why_experiment_duplicate_h2_fails`
   - `test_migration_reports_parse_errors` (or equivalent test name)
3. `uv run python scripts/migrate_add_legacy_why_sentinel.py --dry-run` returns 0 parse errors on the current tree.
4. Synthetic regression: create `/tmp/bad-fm-body.md` with broken YAML; place it under `tasks/proposed/9999/body.md` (or use a fixture path the migrate script walks); confirm dry-run reports it under "parse errors" and `--apply` refuses to commit. Clean up the test body after.
5. `uv run ruff check . && uv run ruff format --check .` PASS on touched files.
6. `uv run python scripts/workflow_lint.py --check-references --check-tables --check-asks` PASS.

## Acceptance criteria

- All six items closed per their fix descriptions.
- No regression in the 119 existing tests.
- 3+ new tests added covering tilde-fence FAIL, duplicate-H2 FAIL, and parse-error reporting.
- The `task.py` ↔ `verify_task_body.py` constant drift risk is eliminated by the `why_gate.py` extraction.
- The SKILL.md instruction matches what `task.py set-body` actually does.

## Explicit cuts (not bundled)

- Indented-code-block (4-space) fence detection — defer.
- Restructuring the gate as a Pydantic model + schema validation — out of scope for tail-risk hardening.
- Per-experiment dashboard (item 4 of workflow-changes queue).
