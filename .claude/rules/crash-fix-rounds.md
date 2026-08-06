---
description: Retry/crash-fix round contract for implementer agents — failure-lesson block, fix-engaged signal, scope guard, kill-before-relaunch + timeout-bounded smokes (#829/#848/#1198); relaunch-side fix-commit ancestry + stale-checkpoint hygiene (#1081); MooseFS content-read probe on same-pod relaunches (#1112/#1594); relaunch compute-character re-statement (#1749); live-launcher forward-phase enumeration (#1856)
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
same-day sessions re-hit the same failure classes with no cross-session
channel (#537/#545):

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

**Root-cause-confirmed firing (#712).** Emit this block ALSO when your
round IDENTIFIED the true root cause of the posted `epm:failure` even if
your fix then hit a NEW, DISTINCT failure, or the run could not complete
on the current pod (set `root_cause_confirmed: yes` — the orchestrator's
`failure_lesson_capture_eligible()` predicate captures it regardless of a
following failure). When the confirmed cause CORRECTS an earlier
failure-lesson on this task, set `supersedes:` to that lesson's slug or
marker timestamp so the durable record retracts the wrong one instead of
stacking two contradictory gotchas; leave it blank otherwise.

### Crash-fix rounds: declare the fix-engaged signal (REQUIRED)

When your round was dispatched to fix a posted `epm:failure` (the same
`/issue` Step 7 `code`-row crash-fix loop the failure-lesson block above
covers), you MUST also declare — in the `## Smoke run` section of your
report — the **fix-engaged signal**: the exact observable the fix
produces that PROVES its code path is actually reached. Without it, a
session reprovisions a fresh multi-GPU pod and re-runs a "fix" that the
failure proves never engaged (#664: a chunking fix was relaunched when
the absence of any chunk log line meant the hang preceded the first
chunk, so the chunking code could not have run).

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
   fix (the commits this round pushed), each pasted verbatim from
   `git rev-parse` / `git log --format=%H` output — never hand-extended
   from a short SHA (#1586 r7). Any subsequent relaunch asserts
   this SHA is an ancestor of the launch checkout's HEAD (§ Crash-fix
   relaunch below). Declare it so the relauncher keys the probe to the
   SPECIFIC fix — never to "the branch tip": a checkout at the tip of a
   ref the fix never landed on passes tip-equality and still runs stale
   code (#779: a relaunch ran the pre-fix commit, checkpointed garbage,
   and the restart resumed it).
5. **Stale-run artifact disposition** — enumerate the resume-state paths
   the FAILED run wrote that a relaunched/resumed run would LOAD (the
   pod-local checkpoint/output globs; any REMOTE resume prefix — HF
   `issueN_partial/`, a data-repo checkpoint path — the driver fetches;
   name the driver's resume-discovery rule), and declare a disposition
   per state class (ONE overall when local + remote need no split):
   `quarantine → <dest outside the resume-glob match set>` (DEFAULT) |
   `retain — <reason the fix does not invalidate this state>` (the fix
   is orthogonal to the checkpointed state; the resume glob must resolve
   to exactly the RETAINED expected paths at relaunch) |
   `wipe` (only when (garbage-by-construction AND already persisted) OR
   a pure re-derivable cache; remote copies are effectively out of
   `wipe`'s reach — prefer a fresh resume prefix) |
   `fresh-output-path / --no-resume` (the relaunch writes/reads elsewhere)
   | `hf-force-reupload → <affected HF prefixes>` (when the fix
   invalidates artifacts ALREADY uploaded to HF, force re-upload with
   `resume_skip=False` — or write to a fresh prefix; a presence-check
   `resume_skip=True` upload silently retains corrupted artifacts for
   downstream reusers, #1902)
   | `wipe-derived-sentinels → <sentinel paths>` (downstream phase
   sentinels / done-markers whose inputs the fix invalidated are wiped
   or quarantined, even on a `--from-phase` relaunch that would not
   otherwise touch them, #1902)
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
timed-out or abandoned Bash call — AND to any kill targeting a workload
pattern for any other reason (e.g. cancelling a redundant background
job of your own). Applies on the shared VM and on pods alike. A
timed-out / abandoned Bash TOOL call kills the SHELL but ORPHANS the
python child, which keeps running and writing its output paths;
relaunching without killing it duplicates load and corrupts shared
outputs (#823: THREE concurrent `--phase 4 --smoke` instances, SAME
output paths, 64 threads each, ~1/3 of a load-186 VM overload).

**Before ANY re-run, kill-and-confirm-dead:**

1. **Probe** — `pgrep -af 'run_823[.]py --phase 4'`. The pattern MUST be
   exact-invocation-scoped: script filename + the distinguishing args,
   with a `[.]` bracket so the probe's own shell cmdline cannot
   self-match. When the command line ALSO passes the artifact PATH as an
   argument, or uses MULTIPLE alternates, the bracket idiom alone does
   NOT protect — apply the self-exclusion filter / per-pid iteration
   recipes in `.claude/rules/gotchas.md` (SSH-remote ownership-probe
   entry) alongside the bracket. For the /issue gate single-flight sites
   the mechanical fix is `scripts/step9c_baseline.py probe` — self- +
   ancestor-pid excluding, exit 0 = clear (INVERTED vs pgrep); kill-arms
   keep raw pgrep for the pid list (#1821). The same bracket idiom
   equally covers OWNERSHIP/liveness probes, including SSH-remote ones —
   `ssh <host> "pgrep -af ..."` re-materializes the pattern in the
   remote shell's argv.
   Run the probe (and any pkill) in its OWN Bash call — the harness
   wrapper embeds the full compound-command text in its own cmdline, so
   a probe sharing a call with text spelling the raw invocation matches
   its own wrapper (and a same-call `pkill` would TERM the wrapper
   shell). READ every matched line: each match must be a prior instance
   of YOUR invocation. **Cmdline identity is NOT ownership**: concurrent
   sessions run byte-identical invocations, so for any match whose
   cmdline equals your own, confirm a second discriminator BEFORE any
   kill — `ls -l /proc/<PID>/cwd` resolves inside YOUR issue worktree,
   and/or `ps -o lstart= -p <PID>` matches your own earlier launch time.
   Ambiguous → do NOT kill; leave it and report. This VM is shared — a
   broad pattern (`pkill -f python`, `pkill -f run_`, `pkill -f uv`) can
   kill ANOTHER session's work and is BANNED. Any match that is not
   yours → narrow the pattern, or kill by explicit PID instead of pkill.
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

**Live-launcher forward-phase enumeration (REQUIRED before duplicating any
phase in-place; #1482).** Steps 1-3 cover instances ALREADY RUNNING; a live
launcher's UPCOMING phases are invisible to any pgrep probe. Before
launching a detached duplicate / manual "recovery" instance of a workload
PHASE (a fits pass, an upload walk, an eval leg) on a machine — pod, GCE
instance, or the shared VM — where the workload's own launcher/dispatcher
is STILL LIVE, enumerate the launcher's REMAINING phase sequence: read its
cmdline (`ps -o args= -p <pid>` — e.g. a `--phase all` expansion) and/or
its dispatcher leg list + current `[phase=...]` log position. REMAINING
includes re-enterable phases — a resume/retry loop can RE-enter a phase it
already ran. If the phase you are about to duplicate appears in that
remaining sequence — OR the enumeration is unreadable / membership is
ambiguous (no leg list, no log position, an opaque cmdline): treat that
EXACTLY as membership; never launch the duplicate on unproven absence
(same fail-safe family as step 1's "Ambiguous → do NOT kill") — take ONE
of exactly two dispositions: (a) DEFER hands-off (DEFAULT — let the
launcher run its own sequence), or (b) TERMINATE the launcher first via
steps 1-3 and own the whole remaining sequence yourself. NEVER run the
duplicate alongside the live launcher. A long ETA is NOT a third
disposition: the recovery's own actions can collapse the launcher's ETA.
And membership is a SUFFICIENT trigger, not a necessary one — the
collision is resource-level, so a duplicate that contends for GPU/RAM
with ANY remaining phase warrants the same disposition. (#1482: a
"recovery" fits instance launched on the "idle" GPU collided with the
live launcher's own fits — the launcher OOMed, crash-persisted, and
powered the instance off, killing the healthy detached fits too.)

**After the kill, the relaunch itself must rewrite the pid file** with
the new live pid in the same command chain, then confirm it before
posting the fresh marker (`.claude/rules/pod-side-reporting.md`
§ Pid-file launch contract — #813 v5: a relaunch that left the
predecessor's pid in `/workspace/logs/issue-<N>.pid` cost a corrective
extra round).

**Bound every FOREGROUND/SYNCHRONOUS smoke or local invocation with
`timeout(1)` as the DIRECT parent:** `timeout --kill-after=30s <N>s uv
run python ...`, sized so `<N> + 30` ends ≥ 60 s BEFORE the Bash tool
timeout (e.g. `510s` under the generous 600 000 ms tool budget; the
DEFAULT tool timeout is 120 s) — an abandoned smoke then self-terminates
instead of orphaning. Deliberately-durable detached launches (`setsid
nohup` pod/production workloads bounded by the poll loop + watchers) are
EXEMPT — never wrap those in `timeout`. GNU `timeout` in its default
(non-`--foreground`) mode times out the command's CHILDREN too; residual
gap: a grandchild that `setsid`s escapes the group kill — step 1's probe
before relaunch is the backstop.

Step 9c test-verdict gate runs are BACKGROUND invocations with selector-sized
bounds (SKILL.md 9c step 1b) — the ~510s foreground bound does NOT apply to them.

**Per-leg out-roots for regime-keyed drivers.** When one dispatch runs a
smoke leg AND a production leg of a driver whose resume state is keyed
on the run REGIME (`--smoke`/`--full`, eval limits, ladder rung, a
`--method`-class flag), give EACH leg its OWN out-root: a shared
explicit `--out-root` leaves the smoke leg's regime in the resume state
and the production leg fail-louds on it (#1333: the FULL leg died at
`_check_regime` because the shared `--out-root` overrode the driver's
own smoke-root rebinding). The regime refusal is CORRECT fail-loud
behavior; the fix is per-leg roots at dispatch time, never weakening the
check (driver-side mechanism: `.claude/rules/gotchas.md` "Smoke-root
rebinding" entry).
The per-leg roots this convention produces carry a sibling trap: the CHAIN
leaves the earlier leg's out-root as unowned residue on a quota'd pod,
starving the later leg's disk-headroom assert — the LATER leg reaps the
derived sibling root at its first phase entry (`.claude/rules/gotchas.md`
"Chained smoke-then-full" entry; #1586 fu r3, fix `afcf2cabac`).

### Crash-fix rounds: scope guard (REQUIRED)

A crash-fix round has EXACTLY ONE marker output posted DIRECTLY by you.
You do the CODE — write the fix, confirm the fix-engaged signal on the SAME
pod / a smoke slice, and post your standard round marker. Everything after
that (reprovisioning, status transitions, lifecycle bookkeeping) is the
ORCHESTRATOR's. Overstepping this scope forces the orchestrator to
reconstruct which markers are real vs stale (#722: a crash-fix round
self-launched a fresh run and injected orchestrator-owned lifecycle
markers).

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
`[phase=<name>]` log breadcrumbs — the standard pod-side contract; the
ORCHESTRATOR's poller drains those into markers on the VM. Keep writing
the sentinel — in-scope; a hand-posted `task.py post-marker <N>
epm:results` from a subagent context is out-of-scope.

**Reprovisioning is NOT yours.** Confirm the fix-engaged signal on the SAME
pod (or a tiny smoke slice) — that is the full extent of your re-run.
Whether to reprovision for the full run is the ORCHESTRATOR's decision
(`/issue` Step 7 crash-fix routing); a fresh-provision full relaunch is the
banned #722 regression.

The orchestrator-side counterpart: when the orchestrator's relaunch
succeeds (a fresh `epm:run-launched`), it ALSO reconciles a stale
`blocked` status back to `running` (SKILL.md § "A successful relaunch
also reconciles a stale `blocked`"; #742 — 35h healthy at `blocked`).
Status transitions remain orchestrator-owned: you never run `set-status`
yourself, even to clear a stale block your fix made obsolete.

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
   - **MooseFS content read (SAME-POD relaunch on a MooseFS-backed
     checkout — the RunPod `/workspace` lane) (#1112).** Git-level
     verification proves nothing about the BYTES a subprocess will
     read: MooseFS FUSE can serve the pre-pull copy of a just-updated
     file while HEAD + ancestry read correct (#1112: a pod verified at
     the fix commit's HEAD crashed on the PRE-fix assert; trap entry:
     `.claude/rules/gotchas.md` § MooseFS stale-served bytes). After
     the ancestry probe, verify the served bytes of every path the
     declared fix commit(s) touched, pod-side in the same SSH session
     and in the SAME checkout / working tree the relaunch command
     dispatches from (a probe against a different clone proves nothing
     about the tree the run reads):
     `for f in $(git diff-tree --no-commit-id --name-only -r <fix-sha>); do
        test "$(git hash-object -- $f)" = "$(git rev-parse HEAD:$f)" ||
          { echo STALE-BYTES $f; exit 1; }; done && echo BYTES-OK`
     (multiple declared SHAs: union of their diff-trees; cost: one SSH
     round-trip, ~seconds for typical few-file fix commits — FUSE-slow
     git ops can stretch large file sets).
     `git hash-object` always reads the full working-tree content —
     never the index stat cache — and must equal the blob OID at HEAD
     (HEAD, not `<fix-sha>:<path>`: a later commit may touch the file
     again); a fix-touched path ABSENT at HEAD (deleted/renamed since)
     is verified absent instead (`test ! -e <f>` — stale serving can
     resurrect a deletion; apply this branch for deletion-bearing fix
     commits rather than reading the loop's halt as a mount fault). On
     STALE-BYTES: re-materialize the file (`rm -f <f> && git checkout
     HEAD -- <f>` — the `rm` forces a real fresh write; a stat-clean
     bare checkout can no-op) and re-probe ONCE; a persistent mismatch
     means do NOT dispatch — post `epm:failure v1`
     (`failure_class: infra`, `reason: moosefs-stale-read`, naming the
     path) and let the orchestrator swap the pod. Fresh bytes but still
     pre-fix behavior → clear the fix-touched modules' `__pycache__`
     before condemning the mount (stale bytecode of IMPORTED modules is
     the neighboring cause; a main script never executes from
     `__pycache__`, so for a script-file fix the byte probe alone is
     decisive). FRESH-PROVISION relaunches (GCE clone, fresh-RunPod
     bootstrap clone, SLURM rsync) are EXEMPT: the clone/rsync WRITES
     the files fresh.
   - FRESH-PROVISION relaunch (GCP GCE / fresh RunPod — the lane clones
     `origin/issue-<N>` at boot, no pre-boot SSH): probe VM-SIDE before
     dispatch: `git fetch origin issue-<N> --quiet && git merge-base
     --is-ancestor <fix-sha> origin/issue-<N>`. This also catches the
     unpushed-fix case. SLURM: probe the rsync source's HEAD
     (`git -C <src_root> merge-base --is-ancestor <fix-sha> HEAD`);
     the `_assert_repo_branch_synced` guard remains the branch gate.
   - Tip-equality (`HEAD == origin/issue-<N>`) does NOT substitute for
     the ancestry probe, and the probe does not replace the
     HEAD-verification — they compose. With MULTIPLE declared SHAs,
     probe every one (on linear history the tip-most subsumes its
     ancestors). A rebase that rewrote the fix SHA fails the probe LOUD:
     re-resolve the SHA on the rewritten branch, never skip the probe.
   - **Mid-run branch-push race (#1880):** pushing fix commits to
     `origin/issue-<N>` while SIBLING lanes of the same issue are mid-run
     advances origin past their clones — their terminal results-git
     pushes then take the fetch+rebase path
     (`.claude/rules/pod-side-reporting.md` § Result-push verification
     contract), and a lane running a pre-#1880 driver deterministically
     false-crashes at its terminal push with its results intact in
     crash-persist. The PULL/SYNC direction is governed by § Mid-run
     pushes to a live-synced branch below.
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
3. **Compute-character re-statement (fires when the fix — or the relaunch
   configuration — changes the workload's compute shape; #1749).** When the
   relaunch differs from the approved plan §9 / the prior recorded launch in
   ENGINE (e.g. torch↔numpy, vLLM↔HF `generate`, batched↔serial inner
   loop), DEVICE ROUTING (GPU→CPU or CPU→GPU), PARALLEL WIDTH (fleet width,
   per-pod GPU width, worker count), or PER-UNIT SCOPE (cells / draws /
   rungs per unit), the relauncher re-states the compute character BEFORE
   dispatch — the canonical five-element statement (SKILL.md Step 9a-ter
   § Compute-character pre-launch statement) scoped to the delta:
   (i) the new ops arithmetic (units × per-unit cost → projected wall) with
   a MEASURED per-unit basis at the NEW shape — a 1-unit pilot through the
   production entrypoint, or the run's own live-measured figure; never an
   asserted / guessed per-unit cost
   (`.claude/rules/plan-compute-sizing.md` § Per-cell fit phases);
   (ii) the engine + device routing named (the batched helper implementing
   the inner loop, or why genuinely not batchable);
   (iii) GPU-width vs pod width — a CPU-bound or width-1 relaunch on a
   multi-GPU pod triggers the width re-evaluation below AND the CLAUDE.md
   "CPU-only phases don't hold GPU pods" release/downsize duty.
   When the delta MOVES work onto the shared VM or adds ≥ ~5 GB of
   staging, the canonical statement's elements (4) (projected peak RSS /
   off-VM routing) and (5) (staging path + filesystem, off-`/` routing)
   apply too — SKILL.md Step 9a-ter carries both.
   The statement rides the relaunch record: the fresh `epm:run-launched`
   note (or the `epm:compute-deviation` re-post when one is being posted
   anyway) — the same note-token convention as `fix_sha=`, no
   marker-schema change. Pod-side hotfix relaunches are NOT exempt: any
   relaunch must re-post `epm:run-launched` (SKILL.md Step 6d.2), and the
   compute-character statement rides that same marker. A same-shape
   relaunch (identical engine/device/width/scope — the common crash-fix
   case) states nothing new; this duty fires only on the delta. (#1689:
   an unrecorded pod-side hotfix relaunch swapped a fit to serial numpy
   on CPU at width 1; the 4×H100 pod billed ~0% GPU for ~14 h before a
   mid-run measurement surfaced it. The watcher's gpu-idle escalation +
   the `epm:compute-deviation` mid-run measurement remain the detection
   backstop.)

**Width re-evaluation rides every relaunch of an embarrassingly-parallel
unit grid** (`code` and `infra` rows alike): before re-dispatching at the
prior fleet width, run the width re-evaluation of
`.claude/rules/vectorize-many-cell-fits.md` § Mid-run trigger — when the
run's `epm:compute-deviation` chain carries `signature_check: negative`
(or the relaunch's own remaining-work arithmetic projects ≥2× the plan
wall), re-sharding the REMAINING units across a wider fleet is the default
(wall-clock is scarce, credits are not); record the `width_reeval:`
arithmetic on the deviation re-post. The restore machinery this section
already mandates makes re-sharding cheapest at exactly this point (#1092:
a relaunch kept width 4 after a 2.57× negative-signature deviation; wider
re-sharding would have cut hours of wall).

The fresh `epm:run-launched` note ALSO records `fix_sha=<sha>` and the
executed disposition (note-token convention, same class as `pid=` /
`commit=` — no marker-schema change). The `code`-row respawn BRIEF the
orchestrator composes for the experimenter carries both (`fix_sha=` +
the element-5 disposition verbatim). EXEMPT: `infra`-row experimenter
respawns (no code fix ⇒ no fix commit ⇒ no duty 1; duty 2 only when a
prior code-fix round's declared disposition is still UNEXECUTED — the
once-only rule above means an executed disposition is never re-applied).
The async GCP→RunPod failover surface
(`.claude/rules/compute-backend-failover.md`) inherits the preceding
probe-passed dispatch transitively — it clones the fix-bearing
`origin/issue-<N>` and reuses the workload cmd verbatim. The
implementer's own same-pod smoke-slice confirmation (element 2) is
UNCHANGED and is not a "relaunch" under this section.

**Shared-module propagation (REQUIRED — when the fix touches shared library
code).** When the crash-fix touches SHARED library code — anything under
`src/explore_persona_space/{orchestrate,backends,eval,train}/`, OR any
shared `scripts/` helper (any `scripts/` file NOT of the form
`scripts/issue<N>_*.py`) — the SAME round either (a) LANDS the fix on
`main` (worktree rebase-merge at Step 10d, or a scratch-worktree push per
CLAUDE.md § Concurrent repo-root committers), OR (b) posts an EXPLICIT
propagation note naming which sibling issues' running trees carry the
stale code (an `epm:progress` note whose leading token is
`shared-module-propagation`, listing the sibling issue ids). A round-local
branch fix on a shared module is an INCOMPLETE round: the shared library
remains stale on every sibling issue's running tree until the fix lands on
main (#1979 → #1947). Scope boundary: § Mid-run pushes below governs PUSH
TIMING on the crashing issue's OWN branch; this clause governs FIX
PROPAGATION to SIBLING issues' branches.

### Mid-run pushes to a live-synced branch (enumerate live workers FIRST)

Never push a commit to `issue-<N>` — or any ref live workers pull/sync
mid-run — that touches a file ANY live worker holds locally modified.
The worker's sync step refuses on DIRTY TOUCHED PATHS regardless of
byte-identical content (the refusal keys on locally modified paths, not
content), and the
lane dies at its sync point — typically the upload leg — HOURS after
the push (#1739: 3 healthy lanes killed at their upload legs; science
recovered from EXIT-trap crash bundles at ~5-6 GPU-h re-compose cost).
BEFORE any mid-run push to a live-synced ref, enumerate
the live workers and what each holds locally modified; then either:

(a) pin lanes to a detached launch SHA at dispatch, so mid-run branch
    pushes are invisible to running workers (preferred, plan-time). A
    detached-HEAD lane's terminal results push composes with the #1880
    recipe as `git push origin HEAD:refs/heads/<branch>` after the
    fetch+rebase;
(b) defer the patch commit until every running worker has passed its
    sync step; or
(c) land the fix worker-locally only and push after the wave drains —
    SAME-POD relaunches only: a fresh-provision (GCE/SLURM) relaunch
    ancestry-probes `origin/issue-<N>` (§ fix-commit ancestry above),
    so a worker-local-only fix halts it — safely but wastefully — on
    fix-commit-absent.

A sibling lane's crash-fix relaunch legitimately REQUIRES the fix on
the remote ref (§ fix-commit ancestry above) — sequence it via (b)/(c),
never skip the relaunch's ancestry probe. Sibling direction:
`.claude/rules/pod-side-reporting.md` § Result-push verification
contract (#1880) covers the PUSH race (a live lane's terminal push
needs fetch+rebase); this subsection covers the PULL/SYNC refusal —
one mid-run-push doctrine, two failure directions.

### Changed-argv relaunch: argv dry-run (REQUIRED unless byte-identical AND the CLI surface is untouched)

A crash-fix relaunch whose command line differs from the crashed
launch's (a new flag wired by the fix round, a corrected input-source
flag, a hand-recomposed plan-§10 transcription) — OR whose fix round
touched the driver's CLI/validation surface (argparse flags, post-parse
required-input checks) even with a byte-identical argv — runs the argv
dry-run probe (`.claude/skills/issue/SKILL.md` Step 6b § Hand-composed
phase argv dry-run, the canonical recipe) on the VM BEFORE
re-dispatching: the fix-engaged discipline in this file verifies the
FIX is present; the dry-run verifies the new ARGV survives parse +
early post-parse validation. First launches of a new phase argv are
governed by the same SKILL.md clause (#1738: a required input-source
flag omitted from a hand-composed first launch died `SystemExit` rc=1
~7 s after a full GCE boot + venv install).

**Relaunch-flag fidelity + machine caps (#1964).** A relaunch's flag set
is copied VERBATIM from the persisted handle sidecar
(`.claude/cache/issue-<N>-handle.json`) — never re-derived from plan
prose or memory; deliberate changes are named DIFFS against the sidecar
set in the relaunch note (#1902: attempt 6 dispatched twice, missing
`--intent` then `--time-budget-hours`, while the full set sat in the
sidecar). Machine-sized caps (`--rss-cap-gb`, thread caps, width) are
RE-DERIVED for the TARGET machine on any cross-machine move — a sidecar
cap is sized to the machine that wrote it (#1946: a 128 GB box was
dispatched with the 16 GB VM-default cap copied forward). The
dispatch-side probes (a)-(c) — staged-input existence, env-pin
completeness, per-LEG carry-over — are canonical at the SKILL.md Step 6b
dispatch-preflight block (`.claude/skills/issue/SKILL.md`
§ Dispatch-input/env/flag preflight) and bind on relaunches too.

### Crash-fix rounds: symbol-rename whole-tree grep duty (REQUIRED — every retry round; #1728)

Any crash-fix round (or ordinary round) whose diff RENAMES a
MODULE-EXPORTED SYMBOL — a class / top-level function / top-level
dataclass / top-level module-level constant / package-exported name in an
`__init__.py` — MUST, in the SAME round, before its
`epm:experiment-implementation` marker, run

```bash
grep -rn '<old_name>' scripts/ src/
```

for each renamed symbol and EITHER fix every hit to the new name OR
explicitly disposition each hit (e.g. "hit is a comment referencing the
old API history — leave"; "hit is under `external/` — out of scope"; "hit
is a test that pins the pre-rename shape as a regression fixture —
leave"). The grep command AND its per-hit disposition are recorded in the
implementer's `epm:experiment-implementation` marker under a top-level
`### Symbol-rename grep` section. A round that renames >1 symbol records
one grep + disposition block per symbol; a round with NO
module-exported symbol rename records NO block (auditable-N/A convention,
same as Step 0.68's `N/A — no fit-loop`).

**Scope-limit — module-exported symbols only.** This duty fires on renames of
names any OTHER file in `scripts/` or `src/` can `import`, `from ... import`,
or textually reference by identifier: a class, a top-level `def`, a top-level
dataclass, a top-level module-level constant (SCREAMING_SNAKE / literal
assignment at module scope), or an `__init__.py` re-export. It does NOT fire
on renames of LOCAL variables inside a function body (`data → payload` inside
a function), private helpers whose name starts with `_` AND that no other
file imports (verify by the grep itself returning ≤ ~1 hit — self-file only),
loop counters, or parameter names in an internal signature no external caller
threads. When in doubt, run the grep: an over-fire produces one extra grep
command in the marker (cost: seconds); an under-fire is the incident this
duty exists to prevent.

**Cross-round rename discipline.** A rename in ROUND R that missed a sibling
hit at time R and was caught in ROUND R' by an import-time crash does NOT
retroactively excuse round R — round R''s marker MUST record the grep + fix
for the missed hit, tagged `(carrying #<R>'s rename)`. This closes the gap
that "the rename shipped in an earlier round, so it's not my rename" would
otherwise open.

(#1728: round R5 renamed `DispatchCall → DispatchItem` but left a sibling
script importing the old name; the next phase reached the sibling's import
and the vLLM engine core died on `ImportError` — sibling scripts drift
until the next phase invokes them. Cost: a full crash-fix round + a wasted
pod launch cycle.)
