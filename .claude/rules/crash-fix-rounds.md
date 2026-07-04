---
description: Retry/crash-fix round contract for implementer agents — failure-lesson block, fix-engaged signal, scope guard, kill-before-relaunch + timeout-bounded smokes; §§1-3 relocated verbatim from experiment-implementer.md (#829); kill-before-relaunch added (#848)
paths:
  - "scripts/**/*.py"
  - "src/explore_persona_space/**"
---

# Crash-fix rounds (relocated from experiment-implementer.md, #829)

### Crash-fix rounds: failure-lesson block (REQUIRED)

When your round was dispatched to fix a posted `epm:failure` (the
`/issue` Step 7 `code`-row crash-fix loop), END your report with a
structured lesson block. The orchestrator posts it verbatim as an
`epm:failure-lesson v1` marker and, on `generalizes: yes`, persists it
to the owning agent's memory the same hour — without this, parallel
same-day sessions re-hit the same failure classes (incidents #537/#545,
2026-06-11: disk pressure, vLLM engine-init crashes at phase
boundaries, stale-artifact asserts, hours apart, no cross-session
channel):

```
<!-- epm:failure-lesson v1 -->
failure_class: code|infra|data
phase: <pipeline phase or script>
lesson: <1-3 sentences: the trap + the fix, written for the NEXT agent>
generalizes: yes|no   # yes only if the trap plausibly recurs beyond this issue
owning_agent: experiment-implementer|experimenter
gotcha_candidate: yes|no  # yes for codebase/infra traps that belong in .claude/rules/gotchas.md
root_cause_confirmed: yes|no  # yes if THIS round identified the TRUE root cause (even if a NEW distinct failure followed or the pod was abandoned in recovery)
supersedes:           # OPTIONAL: prior-lesson slug or marker-ts this lesson corrects; omit if none
<!-- /epm:failure-lesson -->
```

Calibrate `generalizes`: `yes` ONLY if the trap plausibly recurs on
OTHER issues — library behavior, infra quirk, recipe trap — NOT a typo
or wiring mistake in this issue's own script. The `lesson` is written
for the NEXT agent: name the trap + the fix in 1-3 sentences, no
transcript dumps. Ordinary (non-crash-fix) rounds do NOT emit this
block.

**Root-cause-confirmed firing (added #712).** Emit this block ALSO when
your round IDENTIFIED the true root cause of the posted `epm:failure`
even if your fix then hit a NEW, DISTINCT failure, or the run could not
complete on the current pod (set `root_cause_confirmed: yes`). The
capture decision is the orchestrator's pure
`failure_lesson_capture_eligible()` predicate: a block with
`root_cause_confirmed: yes` is captured regardless of a following
failure. When the cause you confirmed CORRECTS an earlier
failure-lesson on this task (a prior mis-diagnosis the team already
captured), set `supersedes:` to that earlier lesson's slug or marker
timestamp so the durable record retracts the wrong one instead of
stacking two contradictory gotchas. Leave `supersedes:` blank when there
is nothing to correct (the common case).

### Crash-fix rounds: declare the fix-engaged signal (REQUIRED)

When your round was dispatched to fix a posted `epm:failure` (the same
`/issue` Step 7 `code`-row crash-fix loop the failure-lesson block above
covers), you MUST also declare — in the `## Smoke run` section of your
report — the **fix-engaged signal**: the exact observable the fix
produces that PROVES its code path is actually reached. Without it, a
session reprovisions a fresh multi-GPU pod and re-runs a "fix" that the
failure proves never engaged — the #664 saga relaunched a chunk-500 fix
when the absence of any `[vllm-chunk]` log line meant the hang preceded
the first chunk, so the chunking code could not have run (2026-06-27).

A fix-engaged signal is ONE of:

- a **log line** the new branch emits (e.g. `[vllm-chunk] _greedy chunk
  1/N` for a chunked-generate fix, an `INFO` log specific to the new code
  path),
- an **`epm:` marker / sentinel write** the fixed path performs, or
- a **file write / artifact** that only the fixed branch produces.

Report it under `## Smoke run` in a `### fix-engaged signal` sub-section
with three elements:

1. **The expected signal**, quoted exactly (the literal log substring /
   marker kind / artifact path).
2. **The same-pod / smoke-slice confirmation FIRST.** Re-launch on the
   SAME pod (or a tiny smoke slice) and confirm the signal appears in
   stdout / stderr / the log — paste the matched line. ONLY THEN may a
   fresh pod be reprovisioned for the full run. A reprovision BEFORE the
   signal is confirmed is the banned regression. Apply
   § Kill-before-relaunch before ANY re-launch.
3. **Why the signal proves engagement** — one sentence tying the signal
   to the specific branch the fix added (so a reviewer can tell a generic
   startup log from a fix-specific one).

If the fix's code path genuinely cannot emit a distinguishable signal at
smoke scale (rare — most fixes add a branch that logs or writes), say so
explicitly in `(d) Needs human eyeball` with the reason AND the closest
demonstrable proxy (the precondition assert ran, the new branch was
entered under a forced flag). "The code is there" is NOT a signal — an
unreached fix is indistinguishable from no fix.

This block is REQUIRED on every crash-fix round (alongside the
failure-lesson block above) and is verified at code-review (Step 0.6 of
`code-reviewer.md`): a crash-fix round whose `## Smoke run` lacks a
confirmed `### fix-engaged signal` sub-section FAILs with the
`substantive` blocker tag. Ordinary (non-crash-fix) rounds do NOT
emit this block.

### Kill-before-relaunch + `timeout`-bounded smokes (REQUIRED — every retry surface)

Applies to EVERY re-run of a smoke / launch / dispatch command — crash-fix
rounds, code-review revision rounds, and same-turn retries after a
timed-out or abandoned Bash call — on the shared VM and on pods alike. A
timed-out / abandoned Bash TOOL call kills the SHELL but ORPHANS the
python child, which keeps running and writing its output paths;
relaunching without killing it duplicates load and corrupts shared
outputs (incident 2026-07-02: #823's review-retry loop ran THREE
concurrent `run_823.py --phase 4 --smoke` instances — launched
23:48/23:51/00:02, SAME output paths, 64 threads each, ~1/3 of a
load-186 VM overload).

**Before ANY re-run, kill-and-confirm-dead:**

1. **Probe** — `pgrep -af 'run_823[.]py --phase 4'`. The pattern MUST be
   exact-invocation-scoped: script filename + the distinguishing args,
   with a `[.]` bracket so the probe's own shell cmdline cannot
   self-match. Run the probe (and any pkill) in its OWN Bash call — the
   harness wrapper embeds the full compound-command text in its own
   cmdline, so a probe sharing a call with text spelling the raw
   invocation matches its own wrapper (and a same-call `pkill` would
   TERM the wrapper shell). READ every matched line: each match must be
   a prior instance of YOUR invocation. **Cmdline identity is NOT
   ownership**: concurrent sessions run byte-identical invocations
   (`uv run pytest`, `scripts/train.py condition=<c> seed=<s>` with
   coinciding args), so for any match whose cmdline equals your own
   invocation, confirm a second discriminator BEFORE any kill —
   `ls -l /proc/<PID>/cwd` resolves inside YOUR issue worktree, and/or
   `ps -o lstart= -p <PID>` matches your own earlier launch time.
   Ambiguous → do NOT kill; leave it and report. This VM is shared by
   many concurrent sessions — a broad pattern (`pkill -f python`,
   `pkill -f run_`, `pkill -f uv`) can kill ANOTHER session's work and
   is BANNED. Any match that is not yours → narrow the pattern, or kill
   by explicit PID from the listing instead of pkill.
2. **Kill** — `pkill -TERM -f '<same pattern>'`; wait ~10 s;
   `pkill -KILL -f '<same pattern>' 2>/dev/null || true`.
3. **Confirm dead** — re-run the step-1 probe; relaunch ONLY when it
   returns nothing. A PID surviving SIGKILL: stop and report; never
   relaunch on top of it.
4. **Fallback (invocation string not distinctive)** — check no live
   process holds the smoke's output paths (`fuser -v <path>`), then kill
   by explicit PID. Weaker than step 1 (a writer that opens-writes-closes
   is invisible between writes) — prefer the cmdline probe when possible.

**Bound every FOREGROUND/SYNCHRONOUS smoke or local invocation with
`timeout(1)` as the DIRECT parent:** `timeout --kill-after=30s <N>s uv
run python ...`, sized so `<N> + 30` ends ≥ 60 s BEFORE the Bash tool
timeout (e.g. `510s` under the generous 600 000 ms tool budget; NOTE the
DEFAULT tool timeout is 120 s — set the generous tool timeout first or
size `<N>` under the default) — an abandoned smoke then self-terminates
instead of orphaning. Deliberately-durable detached launches (`setsid
nohup` pod/production workloads bounded by the poll loop + watchers) are
EXEMPT — never wrap those in `timeout`. Unlike the tool timeout, GNU
`timeout` in its default (non-`--foreground`) mode times out the
command's CHILDREN too. Residual gap: a grandchild that `setsid`s
escapes the group kill — step 1's probe before relaunch is the backstop.

### Crash-fix rounds: scope guard (REQUIRED)

A crash-fix round has EXACTLY ONE marker output posted DIRECTLY by you.
You do the CODE — write the fix, confirm the fix-engaged signal on the SAME
pod / a smoke slice, and post your standard round marker. Everything after
that (reprovisioning, status transitions, lifecycle bookkeeping) is the
ORCHESTRATOR's. Overstepping this scope forces the orchestrator to
reconstruct which markers are real vs stale (incident #722, 2026-06-30: a
crash-fix round attempted to self-launch a fresh GCP run and inject
orchestrator-owned lifecycle markers, forcing the orchestrator to untangle
the real signal from the noise).

**Your ONLY direct marker output on a crash-fix round is ONE of:**

- ONE `epm:experiment-implementation v<n>` (the standard successful-round
  marker) — with the `### fix-engaged signal` sub-section in `## Smoke run`
  and the `<!-- epm:failure-lesson v1 -->` block appended, per the
  failure-lesson + fix-engaged sections above; OR
- ONE `epm:failure v1` (if you are BLOCKED — you could not fix it in-turn),
  per `### On unrecoverable error`.

This rule scopes to **lifecycle / status / review markers**. Your
legitimate implementer-diagnostic self-tags — `epm:smoke-architecture-check`
(pre-write architecture verdict), `epm:compute-deviation` (>2× wall-time
projection report), `epm:new-bug-class` (whack-a-mole detector input), plus
`epm:proposed-tests` in TDD mode — are UNCHANGED and remain REQUIRED per
their own sections above; the orchestrator's Step 5.bis + Step 6d.0
pre-dispatch checks read them. Only the lifecycle / status / review
markers listed below are the ones you must never hand-post. In particular,
you NEVER hand-post any of these ORCHESTRATOR-OWNED markers — the `/issue`
skill posts them, keyed off YOUR marker:

- `epm:code-review`, `epm:code-review-codex`, `epm:review-reconcile`
  (the code-review ensemble runs AFTER your marker — you never review
  yourself or post its verdicts);
- `epm:status-changed` (status transitions are the skill's);
- `epm:pod-provisioned`, `epm:pod-terminated`, `epm:pod-stopped`,
  `epm:run-launched` (pod lifecycle + run dispatch are the skill's, per
  `## What you do NOT do`);
- `epm:upload-verification`, `epm:merged`, `epm:completion-audit`,
  `epm:step-completed` (pipeline-stage bookkeeping the orchestrator emits);
- a HIGHER-version `epm:failure` (`v2`+). Your BLOCKED marker is
  `epm:failure v1`. A higher-version `epm:failure` belongs to OTHER agents'
  failures or the orchestrator's re-classification — never yours.

**Sentinel-emitted markers (`epm:results`, `epm:progress`) are NOT
exceptions to the "one direct marker" rule.** Your dispatcher DOES write
the `/workspace/logs/issue-<N>-results.json` sentinel and DOES emit
`[phase=<name>]` log breadcrumbs — that is the standard pod-side contract
(see the `epm:results` sentinel format spec elsewhere in this file).
The ORCHESTRATOR's poller drains those into `epm:results` and
`epm:progress` markers on the VM. You do NOT hand-post those markers via
`task.py post-marker`; you only produce their sentinels + breadcrumbs
through the driver, as specified. Keep writing the sentinel — that is
in-scope; a hand-posted `task.py post-marker <N> epm:results` from a
subagent context is out-of-scope.

**Reprovisioning is NOT yours.** Confirm the fix-engaged signal on the SAME
pod (or a tiny smoke slice) as § fix-engaged signal requires — that is the full
extent of your re-run. You do NOT relaunch the full run on a fresh pod / GCP
instance / SLURM job. Whether to reprovision for the full run is the
ORCHESTRATOR's decision, driven by the `/issue` Step 7 crash-fix routing after
it reads your marker. A same-pod / smoke-slice confirmation is in scope; a
fresh-provision full relaunch is out of scope and is the banned #722 regression.

The orchestrator-side counterpart: when the orchestrator's relaunch
succeeds (a fresh `epm:run-launched`), it ALSO reconciles a stale
`blocked` status back to `running` (SKILL.md § "A successful relaunch
also reconciles a stale `blocked`"; incident #742 — 35h healthy at
status `blocked`). Status transitions remain orchestrator-owned: you
never run `set-status` yourself, even to clear a stale block your fix
made obsolete.
