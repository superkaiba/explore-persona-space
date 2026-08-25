---
title: 'workflow-fix: cron_step9c_ledger_refresh.sh committed at mode 100644 — the
  nightly known-red ledger refresh has NEVER run (Permission denied x11); add a lint
  check that cron scripts are executable'
kind: infra
tags:
- wf-fix
- workflow-fix
created_at: '2026-08-23T03:13:39Z'
has_clean_result: false
origin_prompt: /issue 2263
workflow: v1
---
## Overview / Motivation

`scripts/cron_step9c_ledger_refresh.sh` is committed at **mode 100644** (non-executable) while every sibling cron script is 100755. Its crontab entry therefore fails with `Permission denied` on every fire, so the nightly step9c known-red ledger refresh (#2114) **has never run**.

**The diagnosis below is complete. The fix is one bit.** It is filed rather than landed directly only because `origin/main` is currently moving faster than a scratch-worktree push race can win (three consecutive rejections across three different tips), and the server-side PR merge path a normal `/issue` run uses is immune to that.

## The bug

```
$ ls -la scripts/cron_step9c_ledger_refresh.sh
-rw-rw-r-- 1 thomasjiralerspong thomasjiralerspong 7339 Aug 11 07:20

$ git ls-files -s scripts/cron_step9c_ledger_refresh.sh
100644 d9707d9e1789df584a413d6e74970d5315395741 0    scripts/cron_step9c_ledger_refresh.sh

$ ls -la scripts/cron_vm_disk_guard.sh scripts/cron_uv_cache_prune.sh
-rwxrwxr-x  scripts/cron_uv_cache_prune.sh
-rwxrwxr-x  scripts/cron_vm_disk_guard.sh
```

The missing execute bit is **in git** (100644, not a local filesystem accident), so it has been broken since the file was committed.

Crontab entry (present and correct):

```
31 5 * * * /home/thomasjiralerspong/explore-persona-space/scripts/cron_step9c_ledger_refresh.sh >> …/cron_step9c_ledger_refresh.log 2>&1  # nightly step9c known-red ledger refresh (#2114)
```

Log — 11 consecutive identical failures, nothing else ever written:

```
/bin/sh: 1: /home/thomasjiralerspong/explore-persona-space/scripts/cron_step9c_ledger_refresh.sh: Permission denied
```

## The fix

```bash
git update-index --chmod=+x scripts/cron_step9c_ledger_refresh.sh
```

Mode-only; no content change. Verify with `git ls-files -s` → `100755`, and confirm the next 05:31 fire writes a real log line instead of `Permission denied`.

## Why it matters — this is a silent-staleness generator

The known-red ledger is a **fleet-shared** resource: `step9c_baseline.py compare` uses it to classify Step 9c gate failures as NEW vs pre-existing-on-main. With the refresh cron dead, the ledger only advances when a session happens to run `refresh` by hand, so it drifts stale unmonitored and keeps recording whatever `failing_tests` set was true at the last manual run.

Concrete cost, observed on #2263 (2026-08-22/23): the ledger was 26.4h stale recording `failing_tests: 0`. A genuinely pre-existing-on-main failure (`test_no_new_torch_before_dotenv_vm_entrypoints`, offender `scripts/issue823_shared_persona_paired.py` — filed as #2487) had nothing in the ledger to be stripped against, so the compare fell through to its merge-base pristine oracle, where the offending file does not exist, and classified it **NEW** — blaming a branch whose entire round diff was one unrelated test file. That cost a ~102-minute gate run plus two compare replays and manual `git cat-file` probing to disentangle, and produced a FAIL verdict on a clean round.

So the dead cron is upstream of the #2488 oracle-classification gap: a fresh ledger would have masked that seam entirely. Both are worth fixing, but this one is a one-bit change.

## Preventive question for planning — the more valuable half

Nothing verifies that a registered cron script is executable. `workflow_lint.py` has checks for many workflow-surface invariants; a cron entry pointing at a non-executable script is exactly the class of "gate that cannot fire" defect the lint suite exists to catch, and it is trivially mechanizable:

- for each `scripts/cron_*.sh`, assert mode 100755 in the index; and/or
- for each crontab-referenced path in the repo, assert it is executable.

Without that, the next cron script committed without `+x` fails silently the same way, and the only symptom is a shared resource quietly going stale — which is how this one survived long enough to cost #2263 an evening. Prefer adding the check in the same round as the mode fix.

## Verified at filing

- All `ls -la` / `git ls-files -s` / `crontab -l` / log reads above, 2026-08-23.
- `wc -l` on the cron log → 11 lines, all `Permission denied`; log created 2026-08-22 05:31:01.
- The live refresh observed during #2263's gate (pid 559819, parent `systemd --user`) was NOT started by this cron — it cannot run — so it was a manual or session-initiated invocation, consistent with "the ledger only advances by hand".
- A scratch-worktree fix was prepared and committed locally (mode change 100644 → 100755, pre-commit suite green) but could not be pushed: three attempts rejected non-fast-forward against tips `78684b67dc`, `9b51c55a1f`, `fbb766acac`. No partial state was left behind — the scratch worktree was removed and main is unchanged at 100644.

## Provenance

workflow_fix_target: scripts/cron_step9c_ledger_refresh.sh

Found by the #2263 orchestrator while diagnosing its own Step 9c test-verdict, per `.claude/rules/workflow-fix-on-bug.md`.
