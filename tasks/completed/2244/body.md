---
title: 'SKILL.md Step 6: state the bg-Bash timeout floor for free-SLURM-lane launches
  (420s park budget vs 120s Bash default)'
kind: infra
tags: []
created_at: '2026-08-12T15:26:05Z'
has_clean_result: false
parent_id: 1336
origin_prompt: 'Surfaced by /issue 1336 attempt-4 dispatch: a fellows-lane dispatch_issue.py
  launch run as a default-timeout background Bash was SIGTERMed (rc=143) mid-park
  before submitting, because the QoS-ladder park budget is 420s and the Bash tool
  default is 120s. SKILL.md documents the park budget in one place and the timeout=600000
  floor only for cold RunPod launches; the two are never connected.'
workflow: v1
---
## Goal

Close a narrow prevention gap in `.claude/skills/issue/SKILL.md` Step 6: the free-SLURM-lane (`fellows`) `dispatch_issue.py launch` can block up to `EPS_LAUNCH_PARK_PROCESS_BUDGET_SECONDS` (default **420 s**) in the QoS-ladder park, but the Bash tool's DEFAULT timeout is **120 s** — so an orchestrator that dispatches the launch as a plain `Bash(run_in_background=true, ...)` without an explicit `timeout` gets the launcher **SIGTERMed mid-park** before it can submit.

## Observed (task #1336, 2026-08-12T15:22Z, attempt 4)

The launch was dispatched as a background Bash with no explicit `timeout`:

```
uv run python scripts/dispatch_issue.py launch --issue 1336 --backend fellows \
  --repo-branch issue-1336-fullcorpora --intent capture-7b --gpus 8 \
  --time-budget-hours 48 --boot-disk-gb 500 \
  --workload-cmd 'bash scripts/issue1336_dispatch.sh all_v3'
```

Result: **`rc=143`** (128 + SIGTERM), an **empty** launcher log, **no** `epm:cluster-launched` / `epm:backend-selected` marker, an **empty** `squeue`, and the handle sidecar **untouched** (`issue-1336-handle.json` mtime still `2026-08-12 04:11`, i.e. attempt 2's job `11809`). Nothing was submitted — a clean pre-submit death, but ~2 minutes of wall-clock and one wasted dispatch cycle.

The failure is mildly deceptive in two ways worth noting for the fix:

1. **The harness-reported exit code was `0`.** The background command ended with a trailing `cat`/`tail` of the log, so the task-completion notification carried that command's status, not the launcher's. The real `rc=143` was only visible in the captured `LAUNCHER_RC=` line. This is the `.claude/rules/code-style.md` § post-pipe `$?` class showing up at a dispatch site rather than inside a script.
2. **An empty log plus a "success" notification reads like a no-op**, not a kill. Without the explicit rc capture the natural (wrong) next move is to re-probe state and conclude the lane refused.

## Why the existing coverage does not prevent this

Three nearby passages each cover part of it, and none closes it:

- **SKILL.md ~4450-4457 (#2161)** documents the 420 s park budget and the exit-75 `free_lane_park_budget_reached` contract (re-run the same command; it reconnects by job name; no double-submit). It describes what happens when the LAUNCHER exits cleanly at its own budget — not what happens when the HARNESS kills it first.
- **SKILL.md ~4459 (the "Launch-recovery invariant", filed from #1336)** is the RECOVERY: probe both `.claude/cache/issue-<N>-handle.json` and `squeue --name eps-issue-<N>...`; a live job → re-run the same command, both empty → died pre-submit, plain re-run safe. This worked exactly as written in the incident above (both empty, plain re-run). It is recovery, not prevention.
- **SKILL.md ~4755-4780** prescribes `Bash(run_in_background=true, timeout=600000, ...)` — but scoped explicitly to a **cold `--backend runpod`** launch, justified by RunPod's 25-50 min provision wall. A reader on the fellows lane has no reason to apply a RunPod-specific instruction, and the fellows park (420 s) is comfortably under RunPod's wall while still being 3.5× the Bash default.

So the park budget is documented in one place, the timeout floor in another, and the two are never connected.

## Proposed fix (small, prose-only)

1. In the Step 6 dispatch guidance, state the **bg-Bash timeout floor for every lane whose launch can park** — not just RunPod: any `dispatch_issue.py launch` on a free-SLURM lane is dispatched as `Bash(run_in_background=true, timeout=600000, ...)`, because the QoS-ladder park budget (420 s) exceeds the Bash default (120 s). Put this at the SLURM/fellows launch site, adjacent to the ~4450-4457 park-budget paragraph, so it is read by whoever reads the park contract.
2. Cross-reference it from the ~4459 launch-recovery invariant — one clause noting that the most common cause of a killed launch call is an under-budgeted bg-Bash timeout, so the recovery probe should be paired with raising the timeout on the re-run (otherwise the re-run dies the same way).
3. Add the rc-capture discipline at the dispatch site: capture and report the LAUNCHER's rc explicitly (`RC=$?; echo "LAUNCHER_RC=$RC"` immediately after the launcher, before any trailing `cat`/`tail`), since a trailing filter makes the harness-visible exit code meaningless. This is the existing `.claude/rules/code-style.md` post-pipe `$?` rule applied to the launch dispatch — cite it rather than restating it.

## Acceptance criteria

- Step 6's SLURM/fellows launch guidance names the `timeout=600000` floor and ties it to the 420 s park budget, in the same region as the park-budget paragraph.
- The launch-recovery invariant cross-references the timeout cause.
- The rc-capture discipline is stated at the dispatch site with a pointer to the existing post-pipe `$?` rule.
- `uv run python scripts/workflow_lint.py` (no flags) is no worse than its pre-change baseline (there are ~15 pre-existing errors on `main` unrelated to this change — do not chase them; assert only that none newly names an edited file).

## Non-goals

- No change to `EPS_LAUNCH_PARK_PROCESS_BUDGET_SECONDS`, to the exit-75 contract, or to any router/launcher code. This is a documentation-prevention fix; the runtime behavior is already correct and its recovery path already works.
- No new lint check. The failure is a one-line prose gap, and a mechanical check for "did the orchestrator pass a timeout" is not expressible from the surface files.

## Provenance

Surfaced by the `/issue 1336` autonomous session while dispatching attempt 4 (plan v20, the A2 coverage-gate re-specification). The incident cost one dispatch cycle and no compute; the relaunch with `timeout=600000` is the immediate workaround, and the launch-recovery invariant at SKILL.md ~4459 (itself filed from #1336) correctly established that the retry was safe.
