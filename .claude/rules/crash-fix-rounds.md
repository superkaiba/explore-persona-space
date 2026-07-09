---
description: Retry/crash-fix round contract for implementer agents — failure-lesson block, fix-engaged signal, scope guard, kill-before-relaunch + timeout-bounded smokes; §§1-3 relocated verbatim from experiment-implementer.md (#829); kill-before-relaunch added (#848); relaunch-side fix-commit ancestry + stale-checkpoint hygiene (#1081); step-2 explicit-PID kill (#1198)
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
with five elements:

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
4. **The fix commit(s)** — the FULL SHA(s) on `issue-<N>` containing your
   fix (the commits this round pushed). Any subsequent relaunch asserts
   this SHA is an ancestor of the launch checkout's HEAD (§ Crash-fix
   relaunch below). Declare it so the relauncher keys the probe to the
   SPECIFIC fix — never to "the branch tip": a checkout at the tip of a
   ref the fix never landed on passes tip-equality and still runs stale
   code (#779: launch #1 ran the commit BEFORE the fix, checkpointed
   garbage, and the restart resumed it).
5. **Stale-run artifact disposition** — enumerate the resume-state paths
   the FAILED run wrote that a relaunched/resumed run would LOAD (the
   pod-local checkpoint/output globs; any REMOTE resume prefix — HF
   `issueN_partial/`, a data-repo checkpoint path — the driver fetches;
   name the driver's resume-discovery rule), and declare a disposition
   per state class (ONE overall when local + remote need no split):
   `quarantine → <dest outside the resume-glob match set>` (DEFAULT) |
   `retain — <reason the fix does not invalidate this state>` (the fix
   is orthogonal to the checkpointed state — eval-side, upload-phase,
   logging fixes; the resume glob must resolve to exactly the RETAINED
   expected paths at relaunch) |
   `wipe` (only when (garbage-by-construction AND already persisted per
   the crash-persist/upload paths) OR a pure re-derivable cache; remote
   copies are effectively out of `wipe`'s reach — crash-persist
   diagnostics are forensic record, prefer a fresh resume prefix) |
   `fresh-output-path / --no-resume` (the relaunch writes/reads elsewhere)
   | `N/A — <reason>` (no resume state written; or fresh instance AND no
   remote resume fetch).

Element 4 is REQUIRED on every `code`-row crash-fix round (a code fix
always has commits); element 5 is REQUIRED with the explicit
`N/A — <reason>` escape.

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

A fix that ADDS or RE-TUNES a reuse-validation gate threshold mid-round
is additionally governed by `.claude/rules/artifact-reuse.md`
§ Reuse-validation gate calibration — derive the threshold from the
artifact's own committed per-behavior, same-surface reference values
(never a bare constant), and check its HALT-vs-WARN severity class
before relaunching (#813 halts 2-3 were gates invented in crash-fix
rounds).

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
2. **Kill** — by EXPLICIT PID: `kill -TERM <pid>...` on exactly the
   step-1 PIDs you read and confirmed as yours; wait ~10 s;
   `kill -KILL <pid>... 2>/dev/null || true`. A PID that exited between
   probe and kill makes `kill` return nonzero — tolerate it (`|| true`
   or a per-PID loop; step 3's re-probe is the real dead-check), and
   keep the probe→kill gap short. A step-3 re-probe survivor that is
   NOT yours (a spared concurrent invocation) takes step 1's
   leave-and-report disposition, not a kill. Pattern pkill
   (`pkill -TERM -f '<same pattern>'`; wait ~10 s; `pkill -KILL -f
   '<same pattern>' 2>/dev/null || true`) is a FALLBACK only when the
   step-1 listing is unusable (output lost, PIDs unreadable) — pkill
   re-matches the pattern at KILL time, so a concurrent session's
   byte-identical invocation starting after the probe gets TERMed
   without the step-1 discriminators ever seeing it (the probe→kill
   TOCTOU; #848 review finding, closed by #1198).
3. **Confirm dead** — re-run the step-1 probe; relaunch ONLY when it
   returns nothing. A PID surviving SIGKILL: stop and report; never
   relaunch on top of it.
4. **Fallback (invocation string not distinctive)** — check no live
   process holds the smoke's output paths (`fuser -v <path>`), then kill
   by explicit PID. Weaker than step 1 (a writer that opens-writes-closes
   is invisible between writes) — prefer the cmdline probe when possible.

**After the kill, the relaunch itself must rewrite the pid file** with
the new live pid in the same command chain, then confirm it before
posting the fresh marker (`.claude/rules/pod-side-reporting.md`
§ Pid-file launch contract — #813 v5: a relaunch that left the
predecessor's pid in `/workspace/logs/issue-<N>.pid` cost a corrective
extra round).

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

Step 9c test-verdict gate runs are BACKGROUND invocations with selector-sized
bounds (SKILL.md 9c step 1b) — the ~510s foreground bound does NOT apply to them.

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

### Crash-fix relaunch: fix-commit ancestry + stale-checkpoint hygiene

(REQUIRED — RELAUNCHER side: the orchestrator, or the experimenter via
its respawn brief; the implementer only DECLARES, per § scope guard)

After a `code`-row crash-fix round, the relaunch enforces the round's
fix-engaged elements 4/5 BEFORE dispatch (incident #779, 2026-07-06:
the relaunch ran the pre-fix commit, checkpointed garbage — val R²
−4.7 vs ~0.6 — and the restart resumed the poisoned checkpoints):

1. **Fix-commit ancestry probe (fail-loud), keyed to element 4's SHA.**
   - SAME-POD relaunch: after the standard pre-launch sync +
     HEAD-verification (`experimenter.md` § Before Running step 2), run
     `ssh ... "cd /workspace/explore-persona-space && \
       git merge-base --is-ancestor <fix-sha> HEAD && echo FIX-OK"`.
     ANY non-zero exit — including "not a valid object name" (the SHA
     was never fetched) — means FIX ABSENT: do NOT dispatch; re-sync
     (`git fetch origin issue-<N>` + the index-lock recovery) and
     re-probe; still absent → `epm:failure v1` (`failure_class: infra`,
     `reason: fix-commit-absent-on-pod`, naming the SHA).
   - FRESH-PROVISION relaunch (GCP GCE / fresh RunPod — the lane clones
     `origin/issue-<N>` at boot, no pre-boot SSH): probe VM-SIDE before
     dispatch: `git fetch origin issue-<N> --quiet && git merge-base
     --is-ancestor <fix-sha> origin/issue-<N>`. This also catches the
     unpushed-fix case ("at HEAD by construction" holds ONLY once the
     fix is on the remote ref). SLURM: probe the rsync source's HEAD
     (`git -C <src_root> merge-base --is-ancestor <fix-sha> HEAD`);
     the `_assert_repo_branch_synced` guard remains the branch gate.
   - Tip-equality (`HEAD == origin/issue-<N>`) does NOT substitute for
     the ancestry probe, and the probe does not replace the
     HEAD-verification — they compose. With MULTIPLE declared SHAs,
     probe every one (on linear history the tip-most subsumes its
     ancestors). A rebase that rewrote the fix SHA
     fails the probe LOUD: re-resolve the SHA on the rewritten branch
     (or have the implementer re-declare), never skip the probe.
2. **Stale-checkpoint disposition (element 5), executed in the SAME
   command chain as the launch** (the pid-file-contract shape,
   `.claude/rules/pod-side-reporting.md` § Pid-file launch contract),
   then CONFIRMED — list the driver's resume-discovery glob and check it
   resolves EMPTY / to the fresh path / to exactly the RETAINED expected
   paths (for a `retain` declaration) — before posting the fresh
   `epm:run-launched`; a REMOTE disposition is confirmed with a
   remote-side listing (`huggingface_hub.list_repo_files` on the
   prefix), never a local glob alone. The disposition executes ONCE,
   against the failed run the declaring round named — a later respawn
   re-checks only for STALE-run state and NEVER applies it to state
   written by a run that passed the ancestry probe. Default quarantine
   moves the state OUT of the
   resume-glob match set (e.g. `mv <ckpt_dir> <parent>/stale-<ts>/`
   where the glob cannot match) — never an in-place rename a broad glob
   still matches. Fresh-provision: pod-local stale state is N/A by
   construction, but a driver that resumes from REMOTE state (HF
   `issueN_partial/`, a data-repo checkpoint prefix) still needs the
   declared disposition — prefer a fresh resume prefix / `--no-resume`
   threaded into the workload cmd over mutating remote copies.

The fresh `epm:run-launched` note ALSO records `fix_sha=<sha>` and the
executed disposition (note-token convention, same class as `pid=` /
`commit=` — no marker-schema change). The `code`-row respawn BRIEF the
orchestrator composes for the experimenter carries both (`fix_sha=` +
the element-5 disposition verbatim). EXEMPT: `infra`-row experimenter
respawns (no code fix ⇒ no fix commit ⇒ no duty 1; duty 2 only when a
prior code-fix round's declared disposition is still UNEXECUTED — the
once-only rule above means an executed disposition is never re-applied,
so the exemption is safe: coverage rests on the earlier code-round
relaunch's probe plus the standard HEAD-verification against the
now-fix-bearing tip). The async GCP→RunPod failover surface
(`.claude/rules/compute-backend-failover.md`) inherits the preceding
probe-passed dispatch transitively — the failover clones the
fix-bearing `origin/issue-<N>` and reuses the workload cmd verbatim; a
mid-run rebase of `issue-<N>` re-opens the gap there (out of scope
here, noted for the record). The implementer's
own same-pod smoke-slice confirmation (element 2) is UNCHANGED and is
not a "relaunch" under this section.
