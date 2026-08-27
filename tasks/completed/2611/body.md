---
title: 'git_provenance()''s 5s-timeout git status orphans .git/index.lock on slow
  filesystems, breaking every later git op in the run (fix: --no-optional-locks)'
kind: infra
tags: []
created_at: '2026-08-27T00:31:53Z'
has_clean_result: false
parent_id: 2546
origin_prompt: 'Surfaced while driving #2546 arm 3 through /issue: p6_publish went
  FATAL rc=128 on ''Unable to create .git/index.lock: File exists'' after its HF-publish
  leg succeeded. The lock was zero-byte, ownerless, mtime 00:08:45Z — two seconds
  before the p5_fits report that carries repro.git_dirty=None, the documented inconclusive/timed-out
  signal. provenance.py runs ''git status'' (which takes index.lock) under a 5s subprocess
  timeout; on MooseFS the kill orphans the lock. 81 files call git_provenance.'
workflow: v1
---
---
kind: infra
---

# `git_provenance()`'s 5 s-timeout `git status` orphans `.git/index.lock` on slow filesystems, breaking every later git operation in the run (observed: #2546 arm 3 `p6_publish` FATAL rc=128)

## The defect

`src/explore_persona_space/orchestrate/provenance.py`:

    57:  _GIT_TIMEOUT_SEC = 5
    129:            timeout=_GIT_TIMEOUT_SEC,
    176:    out = _run_git(["status", "--porcelain=v1", "--untracked-files=no"], cwd=cwd)

`git status` acquires `.git/index.lock` in order to write back its refreshed stat cache. When the
5 s `subprocess.run(timeout=...)` fires, the child is killed **while holding that lock**, and
nothing removes it. The orphaned zero-byte `index.lock` then makes EVERY subsequent git operation
in the repo fail `rc=128` until a human or a later phase removes it by hand.

The module's contract makes this worse in a specific way: its docstring states it is
best-effort and **"Never fails loud: a non-git tree, missing git binary, or subprocess timeout"**
(`:24`). That contract is honoured on the RETURN path — a timeout yields `dirty=None`, a
documented "inconclusive" signal rendered as explicit JSON `null` (`:262`-`:268`, `commit_string`
`:288`-`:294`). So the probe correctly discloses that IT could not conclude. What it does not
disclose, and does not undo, is that it has left the repository's index lock held. **A probe
documented as never-failing-loud silently mutates shared git state and leaves it broken.**

## Observed incident (task #2546, arm 3)

Timeline, all evidence direct:

1. `p5_fits` completed cleanly at **00:08:47Z** — 43/43 registry jobs `ok`, `dropped_or_degraded`
   empty, all four workers `rc=0`. Its report carries `repro.git_dirty = None`, i.e. the tracked
   scan came back inconclusive.
2. `/workspace/explore-persona-space/.git/index.lock` exists, **size 0**, mtime **00:08:45Z** —
   two seconds before that report was written.
3. `p6_publish` launched 00:20Z. Its HF-publish leg SUCCEEDED:
   `[p6] preds: 40 npz -> …/preds/arm3`, `[p6] out mirror: 50 JSONs -> …/eval_results_mirror/out`.
4. Its git-publish leg then died on the staging step:

       fatal: Unable to create '/workspace/explore-persona-space/.git/index.lock': File exists.
       [signal] wrote sentinel /workspace/logs/issue-2546-publish-git-fail-a3-…json
       [dispatch2546] FATAL: p6_publish (git publish) rc=128

5. Probe at 00:28Z: `git_procs=0`, no dispatcher, no fit workers, no `rebase-merge` /
   `rebase-apply` / `MERGE_HEAD` / `CHERRY_PICK_HEAD`, HEAD unmoved at `60779db5`. The lock had no
   owner and had had none for ~20 minutes.

The pod's `/workspace` is MooseFS (FUSE), and its slowness on this pod is independently
documented in the same task: the preflight deep-import probe needed ~2.7 min against a 180 s
default and had to be re-run at `EPM_PREFLIGHT_IMPORT_PROBE_TIMEOUT_S=600` on TWO separate
launches. A tree-wide `git status` over this repo exceeding 5 s there is entirely ordinary — so
this is reproducible on slow lanes, not a fluke.

## Blast radius

`git_provenance` is a SHARED module: **81 files** under `src/` + `scripts/` reference it (measured
`grep -rl "git_provenance" src/ scripts/ | wc -l`). It is called by figure/plot provenance
(`analysis/paper_plots.py`), artifact metadata (`artifacts/organisms.py`), the ensemble strip
(`orchestrate/ensemble_strip.py`), and dozens of per-issue scripts. Any of them running on a slow
filesystem can orphan the index lock — and the victim is not the caller (which fails soft) but
whatever tries to use git NEXT. In this incident that was a publish phase two phases and twenty
minutes downstream, which is exactly the kind of separation that makes the cause hard to find.

Note the interaction with the shared repo root: a repo-root caller that orphans `index.lock` would
block CONCURRENT sessions' `task.py` commits, not just its own. This incident happened in a pod
checkout, so the damage was contained to one pod; the same code path at the shared root has a
wider failure surface.

## Recommended fix

**Primary: stop taking the lock at all.** Git provides exactly this affordance for
status-polling tools — `git --no-optional-locks status …`, or equivalently the
`GIT_OPTIONAL_LOCKS=0` environment variable. It suppresses the optional index refresh write, so a
killed probe cannot orphan anything. This is the intended mechanism for editors and shell prompts
that poll status, and it fits a best-effort provenance probe precisely. Applying it in `_run_git`
covers every read-only probe in the module at once.

Secondary hardening, worth considering alongside:

- On timeout, ensure the child is actually reaped (`subprocess.run` with `timeout` kills but the
  grandchild `git` can outlive a shell wrapper) — verify no orphan `git` process survives.
- Consider whether 5 s is right for slow-FS lanes at all; but note that RAISING the timeout does
  NOT fix this bug, it only makes it rarer. The lock-orphaning is the defect.
- Do NOT "fix" this by having callers delete stale locks: a heuristic staleness test cannot
  distinguish a killed probe's lock from a live concurrent writer's, and getting it wrong corrupts
  another session's commit. Prevention at the probe is the correct layer.

## Explicitly NOT a duplicate

- **#2610** — `poll_pipeline.py` cannot detect terminal success of a single-phase dispatcher
  invocation. Same task, different subsystem, different mechanism.
- **#2605** — dispatcher worker logs sit outside the poller's log-freshness globs.

Both were surfaced on #2546; all three are distinct.

## Target files

- `src/explore_persona_space/orchestrate/provenance.py` (`_run_git` `:129`, `_git_dirty_status`
  `:176`, `_git_head_sha`, `_argv0_git_state` — every git call in the module)
- any test pinning provenance subprocess behavior

## Provenance

Surfaced while driving #2546 arm 3 through `/issue`: `p6_publish` went FATAL rc=128 on an
orphaned index lock. Grounded by direct reads of `provenance.py:24,42,57,129,166-191,258-284`, the
pod-side lock stat (size 0, mtime 00:08:45Z), the `p5_fits` report's `repro.git_dirty = None`, the
p6 failure log lines quoted above, and an ownership probe showing zero live git processes.
