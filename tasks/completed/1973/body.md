---
title: 'daily-fix: slurm fetch_results atomic rsync pull'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0b3dbc69053a
- daily-auto-filed
created_at: '2026-08-01T07:08:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): SLURM finalize results-rsync
  interrupted mid-transfer stranded a 4.7 GB partial tree directly under eval_results/
  while finalize reported ok (#1768 r3) — direct-in-place `rsync --partial` pull,
  warn-only nonzero rc.'
workflow: v1
---
# daily-fix: slurm fetch_results atomic rsync pull

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED M7; miner-8:P5). Source: session fbee8ba3 (#1768 r3) — the SLURM lane's finalize results pull was interrupted mid-transfer (`rsync: connection unexpectedly closed (4952148341 bytes received)`) inside a finalize output that exited 0, stranding a 4.7 GB partial untracked tree directly under `eval_results/issue_1768/` on an 85%-full disk. Durable results were verified unaffected; the residue was removed by hand.

## Goal

Make the SLURM `fetch_results` pull atomic (temp dir + atomic move) and make an interrupted/failed pull surface in the finalize verdict instead of landing partial trees under `eval_results/` behind an `ok: true`.

## Workflow gap

- **Bug observed:** An interrupted results rsync during #1768's finalize left a 4.7 GB partial tree under `eval_results/issue_1768/` while the finalize output reported success; the partial residue had to be found and removed by hand.
- **Why it is a workflow gap:** `SlurmBackend.fetch_results` (`src/explore_persona_space/backends/slurm.py` ~3042-3111) rsyncs `eval_results/` + `figures/` DIRECTLY into the repo-root trees with `rsync -a --mkpath --partial` — `--partial` deliberately KEEPS partially-transferred files, and the destination is the live `eval_results/` dir, so any interruption lands partial files in place. The nonzero-rc branch is warn-only BY CONTRACT ("Non-fatal by contract ... log the real cause loudly (#598)") — finalize proceeds to `confirm_artifacts` and can report ok when the declared artifacts happen to already be durable, leaving the partial residue silent. There is no temp-dir + atomic-move staging and no wait-for/kill-and-clean of the pull before the ok verdict.
- **Call-hop target correction:** the CONSOLIDATED entry named `scripts/dispatch_issue.py` teardown / `backends/slurm.py`; the pull is CONSTRUCTED in `src/explore_persona_space/backends/slurm.py::SlurmBackend.fetch_results` (dispatch_issue.py `_cmd_finalize` is the caller). Primary target corrected to slurm.py.
- `unverified hypothesis — verify at plan time:` the interruption CAUSE — the miner's read was "teardown raced its own results pull"; candidates at compose time include the `subprocess.run(..., timeout=300)` fence on a 4.7 GB pull (a plausible >300 s transfer; note `TimeoutExpired` would normally propagate, so the exact rc=0 path needs the finalize transcript), a remote-side connection drop, or a teardown/scancel ordering race. The FIX (atomic staging + surfaced failure) is correct under every candidate cause.
- **Confidence (emitter):** medium (code shape probed and verified; interruption mechanism unresolved)
- verified-at-filing: `grep -n "rsync" src/explore_persona_space/backends/slurm.py scripts/dispatch_issue.py` + context read of slurm.py 3040-3111 → the pull loop at ~3093-3099 (`argv = ["rsync", "-a", "--mkpath", "--partial", src, dst]`, `dst = str(local_root / subdir) + "/"`, `subprocess.run(argv, check=False, timeout=300)`) confirms direct-in-place `--partial` pull with warn-only nonzero-rc handling and no temp-dir/atomic move (presence claim, context read). `git log --oneline --since='7 days ago' -- src/explore_persona_space/backends/slurm.py` → 5 commits (sentinel-drain, QoS ladder, intent tables, status-writer race, rsync-lane extra-sync knob) — none touch fetch_results atomicity; no landed fix (2026-07-31).

## Proposed change (candidate diff sketch — refine in planning)

```
src/explore_persona_space/backends/slurm.py (fetch_results):
- for subdir in ("eval_results", "figures"):
-     dst = str(local_root / subdir) + "/"
-     argv = ["rsync", "-a", "--mkpath", "--partial", src, dst]
-     proc = subprocess.run(argv, check=False, timeout=300)
-     if proc.returncode != 0: logger.warning(...)
+ for subdir in ("eval_results", "figures"):
+     staging = local_root / ".slurm-results-staging" / f"issue-{issue}" / subdir
+     rsync into staging (--partial-dir INSIDE staging; size-aware timeout,
+     not a flat 300 s), then on rc==0 atomically move/merge staging → dst;
+     on rc!=0 or timeout: LEAVE dst untouched, delete/keep staging clearly
+     out-of-tree, and RETURN/flag the failed pull so _cmd_finalize surfaces
+     it (rc-3 confirm FAIL path or an explicit fetch_failed field) instead
+     of an unqualified ok.
```
(Keep the #598 non-fatal contract for a genuinely-absent `figures/` — distinguish "source dir absent" (rc 23-class, fine) from an interrupted transfer.)

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm.py` (`SlurmBackend.fetch_results`)
- Secondary: `scripts/dispatch_issue.py` (`_cmd_finalize` — surface the pull verdict), `tests/test_slurm_*.py` (pin test: interrupted pull leaves `eval_results/` untouched and does not report ok)
- Grep before editing: `grep -n '\-\-partial\|fetch_results' src/explore_persona_space/backends/slurm.py scripts/dispatch_issue.py` and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The completion sentinel's location contract (sentinel rides UNDER the rsynced `eval_results/` tree, #598) must survive the staging move — `confirm_artifacts` runs AFTER the atomic move.
- ruff on touched files passes; `scripts/workflow_lint.py` no-flags run passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 0b3dbc69053a

- workflow_fix_target: src/explore_persona_space/backends/slurm.py
- fingerprint: (driver-computed; tag authoritative)

Origin: CONSOLIDATED M7 (miner-8:P5), /daily 2026-07-31 — "SLURM finalize teardown cut the lane's results-rsync mid-transfer while reporting `ok: true` — 4.7 GB partial tree stranded under eval_results/" (session fbee8ba3 / #1768 r3).
