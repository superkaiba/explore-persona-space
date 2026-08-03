---
title: 'daily-fix: verify_task_body FAILs every kind:infra task'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e0f4c14f4f22
- daily-auto-filed
created_at: '2026-07-27T07:16:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): verify_task_body.py returns
  a bare FAIL for kind:infra tasks that never have a clean-result body, so each infra
  session spends a paragraph reasoning the FAIL away by hand'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 1 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Short-circuit `scripts/verify_task_body.py` to an explicit not-applicable verdict for `kind: infra|batch|survey` bodies with `has_clean_result: false`, instead of returning a bare FAIL that every infra session has to reason away by hand.

## Workflow gap

- **Bug observed:** `verify_task_body.py --issue <N>` returns `OVERALL: FAIL` for every `kind: infra` task, because it validates the clean-result body spec against a body that by definition never carries one.
- **Why it is a workflow gap:** the verifier is prescribed as an acceptance check on the infra path but its verdict carries no signal there, so each infra session spends a paragraph explaining that the FAIL is structural — recurring noise repeated fleet-wide.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c 'infra' scripts/verify_task_body.py` → **1** hit, and that hit is L10825 (`"change once the parked corpus clears, via a future infra task"` — unrelated prose), so there is no `kind`-keyed short-circuit — the absence-of-guard evidence; live semantic probe `uv run python scripts/verify_task_body.py --issue 1684` (a live `kind: infra` task) → `[FAIL] body is not a stub — body has no \`# <title>\` H1 line …` / `OVERALL: FAIL (1 of 1 checks failed)`, reproducing the reported behavior verbatim; `git log --oneline --since='7 days ago' -- scripts/verify_task_body.py` → no landed short-circuit (2026-07-26)

## Evidence

- Session `2de5253e`, 14:10:00Z, task #1702: the implementer's own acceptance check recorded `"verify_task_body.py: --issue 1702 reports FAIL \"body is not a stub — body has no \`# <title>\` H1 line\" — this is the EXPECTED status for a kind: infra task that will never carry a clean-result body (matches every other in-flight infra task); not introduced by my edits."` (transcript row 206).
- The behavior reproduces today: `uv run python scripts/verify_task_body.py --issue 1684` on a live `kind: infra` task returns the same single FAIL with the same message text, confirming the FAIL is structural rather than task-specific.
- Scale of the affected corpus: 1,094 `body.md` files under `tasks/*/` carry `kind: infra` (measured 2026-07-26 by a regex scan of the frontmatter), so any infra session prescribed this check hits the same verdict.
- Measured cost: none material — roughly one paragraph of reasoning per infra task. The gap is repeated noise plus an unusable gate signal, not lost wall time.
- Design note: the verifier's argparse exposes `--issue` / `--file` / `--body-stdin` (L14559-14562) and `main()` resolves the body via `_load_text_for_issue`, so the task's frontmatter `kind` is already reachable on the `--issue` path; the `--file` / `--body-stdin` paths have no task frontmatter and must retain today's behavior.

## Proposed change

- In `scripts/verify_task_body.py` `main()` (L14555+), after `--issue` resolution loads the task body and its frontmatter, short-circuit when `kind` is in `{infra, batch, survey}` AND `has_clean_result` is false: print an explicit verdict line naming the kind, e.g. `OVERALL: N/A (kind: infra — no clean-result body expected)`, and return without running the clean-result checks.
- Use a verdict string and exit code distinct from both PASS and FAIL so a caller can branch on it mechanically; do not silently coerce the case to PASS, which would make a genuinely malformed clean-result body on a mis-filed task invisible.
- Leave the `--file` and `--body-stdin` paths unchanged — neither carries task frontmatter, so neither can resolve `kind`.
- Keep the short-circuit conditional on `has_clean_result: false`: an infra task that was later promoted and does carry a clean-result body must still be verified normally.
- Update the SKILL.md acceptance-criteria wording that tells infra implementers to run the verifier, so the expected verdict is the not-applicable one rather than a FAIL to be explained away.
- Add a regression test pinning the not-applicable verdict for a `kind: infra` / `has_clean_result: false` body and the unchanged FAIL for a malformed clean-result body.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- `.claude/skills/issue/SKILL.md` (the infra acceptance-criteria wording that prescribes the run)
- `tests/test_verify_task_body.py` (regression pin)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: e0f4c14f4f22

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: D-P10.
