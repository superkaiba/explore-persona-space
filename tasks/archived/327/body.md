---
title: '[Parent: #320] §5 epm:step-completed EXIT-site wiring + regression test +
  empirical replay-savings check'
kind: infra
tags: []
created_at: '2026-05-08T01:05:15.000Z'
has_clean_result: false
sagan_id: 85a74fb1-3fca-4365-bcd8-40c3dbd8a16b
sagan_number: 327
priority: normal
legacy_why_unset: true
---
## Goal

Wire the `epm:step-completed` marker emission at all 17 EXIT sites in `.claude/skills/issue/SKILL.md`, plus the regression test that enforces full coverage. Completes §5 of Plan A from PR #321.

Parent: #320

## Background

PR #321 shipped (commit `da6bf6bc`):
- `epm:step-completed` marker schema in `markers.md`
- `_decide_entry_step` helper with the load-bearing `status:blocked → full replay` short-circuit BEFORE marker read (closes Critic 2.B2)
- 17-row EXIT-site → `exit_kind` mapping table in §5 of the plan body

What was NOT shipped: the actual EDIT to each of the 17 EXIT sites in `SKILL.md` to post the right marker before exiting. Without this, `_decide_entry_step` reads markers that never exist, and re-entry always hits the full-replay path. The infrastructure works; the call sites that USE it are missing.

## Plan §5's mapping table (the 17 EXIT sites)

| SKILL.md line | Step | exit_kind |
|---|---|---|
| 457 | Step 2c "Defer / 3" plan-approval | parked |
| 524 | Step 4 advance to implementing | clean |
| 566 | Step 5b code-review FAIL revision_round>=3 → blocked | failure-exit |
| 596 | Step 6a HF gate manual approval needed | parked |
| 599 | Step 6a HF gate `HF_TOKEN` missing → blocked | failure-exit |
| 643 | Step 6b pod provision failure → stay | parked |
| 674 | Step 6c preflight FAIL on resumed pod | parked |
| 696 | Step 7 `epm:stale` post (>4h no progress) | parked |
| 781 | Step 8 upload-verifier FAIL | parked |
| 866 | Step 9b reviewer PASS → awaiting-promotion | parked |
| 879 | Step 9b reviewer FAIL | parked |
| 1225 | Step 0 label conflict abort | failure-exit |
| 879 (alt) | Step 9c test-verdict FAIL count<3 | parked |
| 879 (alt2) | Step 9c test-verdict FAIL count>=3 → blocked | failure-exit |
| 921 | Step 10 step 0 completion-audit INCOMPLETE → blocked | failure-exit |
| 1066 | Step 10c pod-termination prompt completion | clean |
| 1116 | Step 10d worktree-merge prompt completion | clean |

(Spot-check the line numbers against current SKILL.md HEAD before editing — line numbers drift with each commit.)

## Acceptance criteria

- [ ] All 17 EXIT sites in SKILL.md post `<!-- epm:step-completed v<n> -->` before exiting (or, for `failure-exit` cases, alongside the `epm:failure` marker)
- [ ] `tests/test_every_exit_site_posts_marker` passes — count parity: number of `EXIT` patterns in SKILL.md == number of `epm:step-completed` post-sites in SKILL.md == 17
- [ ] Empirical "measurably skips replay" verification: pick a half-done test issue (e.g. parked at `status:plan-pending` or `status:awaiting-promotion`); show that re-invocation with `epm:step-completed` markers in place takes measurably fewer tokens than re-invocation with the markers stripped. Document the token-count delta in the `epm:results` body. Acceptance: at least one `parked` re-entry path saves >2k tokens.

## Compute

0 GPU-hours. ~0.5-1 working day. `type:infra`.

## References

- Parent: #320 (Plan A approved)
- PR shipped: #321 (commit `da6bf6bc` §5 helper + router; EXIT-site wiring is what this follow-up adds)
- Plan body cached at `.claude/plans/issue-320-draft.md` §5 (lines ~1116-1331; 17-row mapping at ~1047-1069)
- Code-reviewer concerns on §5: https://github.com/superkaiba/explore-persona-space/issues/320#issuecomment-4402015283
