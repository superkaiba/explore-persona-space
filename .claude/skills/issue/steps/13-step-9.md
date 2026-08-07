# Step 9: Iterative interpretation + final review

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

This step has two sub-phases: **interpretation** (iterative
analyzer<->critic loop) and **final review** (clean-result-critic gate).

#### Step 9 entry: in-flight idempotency guard (backstop re-entry)

The Step 6d.2 backstop cron now survives into `verifying` / `interpreting`
/ `reviewing` (so a stalled interactive session in these stages still gets
auto-woken). The cost to bound is a backstop tick firing `/issue-tick <N>`
(which may load the full `/issue <N>` skill on stale-marker recovery)
while a stage subagent (analyzer, interpretation-critic,
clean-result-critic, upload-verifier) is STILL RUNNING from a prior tick —
re-dispatching it would burn redundant subagent tokens and could race two
writers on the body. This guard makes a fresh re-entry into Step 9 (or
Step 8 verifying) cheaply detect "live work in progress" and EXIT without
re-dispatching (that EXIT is the guard rule's `post_step_completed.py
--exit-kind parked` call below).

**Dispatch breadcrumb (post on every stage dispatch).** Immediately before
spawning ANY Step 8 / Step 9 stage subagent, post a breadcrumb so a later
tick can see the dispatch:
```bash
uv run python scripts/task.py post-marker <N> epm:progress \
  --note "stage-dispatch stage=<verifying|interpreting|clean-result> round=<r> subagent=<name> worktree=<abs path or 'repo-root'>"
```

**Pre-dispatch dedup (NON-SKIPPABLE, every dispatch site).** Immediately
BEFORE posting a NEW `stage-dispatch` breadcrumb and spawning, run the
mechanical check:

```bash
uv run python - <<'PY'
from explore_persona_space.task_workflow import list_events, stage_dispatch_should_skip
print(stage_dispatch_should_skip(list_events(<N>), "<stage>", <r>, window_minutes=<W>) or "DISPATCH")
PY
```

If the output is anything other than `DISPATCH`, log that one line, post
NO duplicate breadcrumb, do NOT spawn — the stage is already in flight
(EXIT `parked` on a backstop tick — the idempotency-guard EXIT, i.e. the
guard rule's `post_step_completed.py --exit-kind parked` call below — or
continue with other work). `<W>`
follows the stage-aware freshness windows below (15 default / 30
Codex-ensembled). This applies to EVERY site that posts a
`stage-dispatch` breadcrumb: the Step 8 results-landed batch, Step 9
rounds, the Step 9a-ter free-analysis follow-up, the
methodology-reference spawn, AND all same-issue follow-up-loop
`stage=followup-<phase>` dispatches — the #778 double-dispatch
(two orchestrators each dispatched a `followup-implementing
round=1` implementer minutes apart, concurrently editing
one worktree) came through the follow-up loop.

Each stage's result marker is its completion signal — the existing
`epm:upload-verification` (verifying), `epm:interpretation v<r>` +
`epm:interp-critique v<r>` (interpreting), and `epm:clean-result-critique
v<r>` (clean-result). The breadcrumb is a generic `epm:progress` note (no
new marker schema), distinguished by its `stage-dispatch` prefix. The
`worktree=` field records WHERE the dispatched subagent writes — the
absolute worktree path, or the literal `repo-root` when it works in the
main checkout — so a successor session or recovery pass can locate
uncommitted in-flight files if this session dies mid-dispatch. (#505) The
same field applies to every dispatch breadcrumb that follows this
convention, including the same-issue follow-up loop's
`stage=followup-<phase>` dispatches.

**Pre-dispatch external-marker triage (REQUIRED — every COMPUTE-stage
dispatch; sibling of the pre-dispatch dedup above).** Cross-session markers
are the sanctioned advisory channel, but a mailbox with no read-gate is how
#779 launched an 18–20h serial grid minutes after finishing its prior
phase while a measured audit saying "must NOT launch as-is" sat unread on
its own events.jsonl (10 external audit/directive markers; the
`stage=followup-grid` breadcrumb claimed
"vectorized" while the fixes were unapplied; killed by PM-chat).
Immediately BEFORE posting the dispatch breadcrumb for any stage that
launches COMPUTE — a pod/GCP/SLURM provision or workload (re)launch
(Step 6b / 6d, crash-fix relaunches included), any stage the
Compute-character pre-launch statement binds (a fit / sweep / statistical
battery, or a ≥ ~5 GB download/staging stage: Step 9a-ter, the Step 9b
same-issue follow-up loop), or a detached
VM-side phase (§ below) — run the mechanical enumerator:

```bash
uv run python - <<'PY'
from explore_persona_space.task_workflow import (
    list_events, triage_candidates_since_last_dispatch,
    triage_enumeration_boundary)
evs = list_events(<N>)
for e in triage_candidates_since_last_dispatch(evs):
    # #1722: total form — an event whose note is "" / None / "\n" makes
    # the classic ("" or "").splitlines()[0] raise IndexError (three sessions
    # hit this shape on markers with empty notes).
    print(e["ts"], e["kind"], (((e.get("note") or "").splitlines()) or [""])[0][:140])
print("boundary=" + triage_enumeration_boundary(evs))
PY
```

It returns every non-machine marker posted since the PREVIOUS DUTY-BOUND
dispatch record — a compute-launch marker (`epm:run-launched` /
`epm:cluster-launched`) or a record carrying the
`external-markers triaged:` line; task start if none. When the most
recent duty-bound record carries a `(boundary=<ts>)` token (#2105), the
window reopens from that recorded enumeration point instead of the
record's own post position — the enumerate-to-post seam (the #2054 v91
directive, posted 53 s before the breadcrumb, is the incident) is
re-enumerated at the next call. On the pod/backend-launch form the token
rides the immediately-preceding adjacent `epm:progress` triage note (the
existing note-then-launch ordering is UNCHANGED); the enumerator chains
one step from a token-less launch marker to that note's token. A non-compute
breadcrumb (review / analyzer / verifier stage) never closes the window —
those dispatches have no triage duty, so an advisory posted before one
still surfaces at the next compute dispatch — and an untriaged compute
breadcrumb (pre-fix / concurrent session) doesn't either
(fail-toward-triage). READ each candidate's full note (`task.py view <N>
--json` + jq by `ts`); classify EXTERNAL = not posted by this session — the
`by` field is unreliable on LEGACY markers and non-compliant emitters
(measured on #779: self and PM-chat posts both carried `by: unknown`), but a
value on the #966 emitter-convention list (`pm-chat`,
`autonomous_session_watch`, `spawn_session`, `spawn_session-stop`) is a
trustworthy-positive EXTERNAL signal (conventional, not authenticated);
absence proves nothing, so still use session context plus the in-the-wild
signatures ("PM-chat", "user-raised", "user directive", "# Audit",
"AMENDMENT", "SCOPE RESTORE"); a successor/recovery session that cannot
attribute a candidate treats it as external (fail-toward-triage). Then
TRIAGE each external advisory/directive: APPLY it (fix the code, adjust or
re-scope the dispatch) BEFORE dispatching, or EXPLICITLY DEFER with a
one-line reason. If applying took non-trivial time, RE-RUN the enumerator
immediately before posting the breadcrumb (markers can land mid-apply).
Record the outcome as ONE line in the dispatch breadcrumb note — or, for
pod/backend launches that post no breadcrumb, in an immediately-adjacent
`epm:progress` note:

    external-markers triaged: <N> applied / <M> deferred (<one-line reasons>) (boundary=<ts>)

or `external-markers triaged: none (boundary=<ts>)` when there are no
external candidates — `<ts>` verbatim from the snippet's `boundary=` output
line (the FINAL enumerator run stamps it — the existing "RE-RUN the
enumerator immediately before posting" instruction defines which run is
final); omit the token only when the snippet printed an empty value (empty
events list).
This is NOT a gate: triage is apply-or-defer, decided by this session,
auto-continue preserved — but deferring a marker that contradicts the
dispatch (e.g. "do not launch as-is") must state WHY the launch is sound
anyway, and a dispatch note asserting a property an unapplied external
audit contradicts (#779's "vectorized") without a triage line is the
regression this rule closes. Triage is BOUNDED to the window; a marker
already covered by a prior triage line is not re-enumerated (its
disposition is on the record). Accepted residuals (named, not silent):
(i) a marker sharing the same second as the stamped boundary event lands
behind the `<=` boundary and is not re-enumerated (same-second residual —
strictly narrower than the pre-#2105 whole enumerate-to-post seam, which
the `(boundary=<ts>)` token now reopens); (ii) a legacy triage line
WITHOUT the token keeps the old post-position boundary (fail-toward-today,
never wider misses); (iii) a launch marker with NO paired triage note
(untriaged launch — pre-fix sessions, crashed duty) keeps today's
launch-position boundary UNLESS an immediately preceding token-bearing
triage record from an earlier dispatch exists, in which case the one-step
chain reopens from that older record's boundary — over-enumeration of
never-triaged markers, fail-toward-triage, bounded by the previous duty
record's enumeration point; and markers posted after a task's LAST compute
dispatch are never enumerated (they can no longer avert a launch). A
watcher-side NON-GATING observer audits this
duty post-hoc (flags missing/'none' lines against a re-run of the
enumerator's window; observe/alert only, never blocks — #967).

**Detached VM-side long compute phases (setsid; pid+log in the breadcrumb — #833).**
Any VM-LOCAL compute phase with projected wall-time >~15 min that the
orchestrator launches DIRECTLY as bg-Bash (a Phase-D-style fit, an
aggregation / permutation battery) MUST be launched fully detached:

    PHASE_PID=$(bash -c 'setsid nohup env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 <cmd> < /dev/null >> <abs, space-free log path> 2>&1 & echo $!')
    ps -p "$PHASE_PID" -o args=   # verify the pid is the workload; on mismatch recover via a
                                  # BRACKETED pattern probe — pgrep -f '<distinctive invocatio[n]>'
                                  # (bracket ONE char: an unbracketed pattern matches this probe's
                                  # OWN argv — gotchas.md ownership-probe entry)
    bash -o pipefail -c 'pgrep -s "$1" | xargs -rn1 sudo -n choom -n -600 -p' _ "$PHASE_PID" >/dev/null \
      || echo "[warn] choom failed or swept nothing — phase is earlyoom-UNPROTECTED (record choom=failed)"

The `bash -c` wrapper is load-bearing for pid capture: the top-level Bash-tool
shell runs with job control ON, where `setsid` forks and a bare `$!` is the
vanished intermediate; inside the wrapper (job control OFF) `setsid` execs in
place, so `$!` is the workload pid. A plain bg-Bash child stays in the
session's kill domain (script-launched children share the launcher's process
group + session id; even a top-level child with its own pgid shares the sid),
and a watcher force-stop / `spawn_session.py stop` kills that tree — #833's
healthy Phase-D fit died mid-flight this way (pure
signal kill). `setsid` gives the phase its own session + process group (group
kills miss it; it reparents to PID 1 when the launching shell exits);
`< /dev/null >> log` drops every fd tether to the dying session. The phase's
stage-dispatch breadcrumb MUST carry four additional fields:
`... pid=<PHASE_PID> log=<abs log path> choom=ok|failed harvest=<abs output path>`
(additive whitespace-split
`key=value` tokens — `_breadcrumb_fields` parses them order-free; keep the
log path space-free; #833's own breadcrumbs already carried a RELATIVE `log=`
— this convention upgrades it to a REQUIRED absolute path, `pid=` is the
genuinely new #833 field, and `harvest=` is the § Harvest contract's declared
results location, new with #1656), and the `external-markers triaged:` line
(§ Pre-dispatch external-marker triage above).

**Probe-bracket rule (#1482).** Every pattern-based liveness / ownership / kill
probe against a detached phase — `pgrep -f` / `pkill -f`, local or over SSH —
uses the bracket idiom (`patter[n]`: bracket ONE character so the pattern can
never match the probe's own command line). An ALIVE read from an UNBRACKETED
pattern probe is UNVERIFIED evidence — never heartbeat evidence, and never a
reason to skip the failure path (#1482). The full recipe — bracket +
self-exclusion filter / per-pid iteration + the separate-Bash-call rule — is
owned by the `.claude/rules/gotchas.md` ownership-probe entry.

**Harvest contract (declared AT LAUNCH — a closed session never strands finished
results; #1310, #1656).** Detachment protects the RUN; this clause protects the
RESULTS. Every detached launch declares its harvest path at launch time:

1. **Durable out-root (REQUIRED).** `<cmd>` writes its results to a durable,
   session-independent location — `eval_results/issue_<N>/...`, the task's
   `artifacts/` dir, an HF-upload staging dir under `data/issue_<N>/`, or (for a
   log-only probe) the breadcrumb's own `log=` file — never session-scoped scratch
   only the launching conversation knows about.
2. **`harvest=` token (REQUIRED).** The stage-dispatch breadcrumb's
   `harvest=<abs, space-free output path or glob>` (comma-separate multiple paths;
   values are whitespace-split tokens, so no spaces) names where completion outputs
   land. The § Successor / re-entry rule probes THIS path for "completion output
   present" and collects from it — no guessing. Breadcrumbs predating this contract
   lack the token: consumers fall back to log-tail + known output dirs (the same
   graceful-optional convention as `label=`).
3. **Self-harvest chaining (PREFERRED).** When collection is one idempotent command,
   make it part of the detached unit itself — as a SINGLE command unit substituted
   for `<cmd>`: a driver script whose final act is the collection/upload (an HF
   upload of raw completions/tensors, a copy into `eval_results/issue_<N>/`), or an
   inner `bash -c '<workload> && <harvest-cmd>'`. NEVER splice a bare
   `<workload> && <harvest-cmd>` into the template: the `&&` splits the launch line,
   binding setsid/nohup/env to the first command and the redirections + trailing `&`
   to the second — detachment silently breaks. This is the detached-phase instance
   of the batch-judge deadline-bounded self-harvest. The identity verify is
   unchanged (the distinctive workload substring still appears in
   `ps -p $PHASE_PID -o args=` for either wrapped form). Only the steps that MUST
   stay session-side — the explicit-path git commit under the concurrent-committer
   discipline, folding numbers into the body — are left to the successor; the DATA
   is durable before any session touches it.

The contract costs one token + a path choice at launch — never a new gate.

**Relaunch + verify discipline (pointer — #1768/#1769/#1482).** Relaunches of a
detached phase follow the pid-file launch contract items 1g/1h
(`.claude/rules/pod-side-reporting.md`): re-run the launcher FILE — for an
ad-hoc first launch, materialize one first — never a hand-re-typed inline
chain (#1768), and key every `pid=` breadcrumb / completion watch on the
identity-verified WORKER pid, never the setsid/nohup/ssh wrapper pid (#1769).
A phase's harvest/commit leg names its expected path set per that rule's
§ Result-push verification contract — an empty-set verify on an
output-declaring round FAILS, never passes vacuously (#1482).

**Monitor-filter hygiene (async-dispatch chains).** A crash-pattern
Monitor/until-loop over a detached phase's log should EXCLUDE known
benign teardown lines (`aclose()`, `Event loop is closed` — vLLM/httpx
shutdown noise) from its error pattern, or it fires spurious wakes on a
healthy phase end (#1773).

**Earlyoom protection is REQUIRED on the verified phase (#957; #811).**
The shared VM runs `earlyoom` with `--prefer '(^|/)(pytest|python3?)$'` (+300
badness to every python process), so a long detached fit is the designated
victim whenever ANY neighbor spikes memory: #811's ~5h fits phase (RSS 6.8 GiB)
was SIGTERM-killed at ~2h (rc=143, 0 checkpoints) by a NEIGHBOR's spike — its
logged badness 1002 decomposes exactly as (1000 + 53‰ RSS)×2/3 + 300
prefer-bonus. The `choom -n -600` sweep runs over the phase's whole SESSION
(`pgrep -s` — `setsid` made `$PHASE_PID` the session id, so the sweep catches
the leader AND any already-forked child; children forked later inherit
`oom_score_adj` across fork), subtracting ~400 display points: decisively below
every default-adj neighbor while staying killable — NOT `-1000`, which earlyoom
and the kernel OOM killer skip entirely, so a genuinely runaway fit must still
die first. Lowering adj needs CAP_SYS_RESOURCE, hence `sudo -n` (passwordless
on the VM). **On a failed sweep: ONE bounded retry → record the final state →
route-or-proceed — a deterministic chain, never a wedge.** When the sweep fails
(or swept nothing), RE-RUN it ONCE — when the workload's real python3 child
appears OR after ≤ ~30-60 s, whichever comes FIRST (#1315 observed the gap
live: choom on the launch pids did not stick to the python3 child `uv run`
spawned moments later — a child forked before its parent's adjustment lands
inherits nothing, and the one-shot sweep never revisits; this bounded retry IS
that re-run, now the default rather than an option). A phase with no such
child (a pure-bash stage, or the workload died pre-fork) skips the wait:
record `choom=failed` and proceed straight to the disposition below. Record
the FINAL post-retry state as `choom=ok|failed` — `choom=ok` ONLY when a sweep
run's pipeline itself exited zero (original or retry); anything else records
`choom=failed` (token vocabulary unchanged). Disposition on post-retry
`choom=failed`: a phase with projected peak RSS ≥ ~16 GiB (already past the
compute-character element-4 off-VM threshold, so a VM launch of one is doubly
exposed) DEFAULTS to routing the phase OFF the shared VM — `cpu-mid` /
`cpu-bigmem` by footprint, CLAUDE.md § CPU-only phases — instead of silently
proceeding unprotected; proceeding VM-local anyway requires a one-line stated
reason in the breadcrumb note (prose beside the token). Phases below ~16 GiB
keep the existing default: the launch PROCEEDS unprotected with the `[warn]` +
`choom=failed` breadcrumb token — never block a launch on choom, and never read
the sweep as guaranteed protection (it re-orders earlyoom's victim selection;
it does not exempt the phase). The ≥ ~16 GiB case is a ROUTING decision (route
or justify), never a blocking gate — no step in the fail → retry → final-state
→ route-or-justify chain may wait indefinitely. choom stays best-effort —
MALLOC_ARENA_MAX=2 in the
launch prefix is the real fix for the arena-fragmentation memory class.
The −600 derivation
assumes this VM's current `--prefer` +300 python bonus (`/etc/default/earlyoom`);
re-derive from the decomposition above if that config changes.
**Collateral-kill signature + second-kill pod pivot.** Phase dead rc=143
(SIGTERM; rc=137 when earlyoom escalated to SIGKILL) + an earlyoom journal line
at the death timestamp naming the phase pid, with the memory SPIKER a NEIGHBOR
(phase RSS well under the pressure; attribute via the kill-source checklist,
`failure_patterns.md` § exit-137/143, + the watcher's `earlyoom-kill` sidecar
rows) = a collateral kill: `failure_class: infra`, NOT a code bug — do not
dispatch a crash-fix round against the phase. Recovery ladder — first-kill
carve-out for long API-BOUND drivers: a judge / batch-poll-class driver
(multi-hour, checkpointed, no GPU need) whose PROTECTED (`choom=ok`) collateral
kill is ATTRIBUTED to a fleet-wide memory storm (the watcher #849
memory-pressure row and/or the earlyoom journal at the death timestamp shows
fleet-wide pressure, not a phase-local spike) routes to the cheap CPU pod lane
(`cpu-mid` / `cpu-small` by footprint, CLAUDE.md § CPU-only phases) on the
FIRST kill, resuming from its checkpoint — never a from-scratch rerun.
Rationale: relaunching an API-bound waiter into a live storm has negative
expected value, and dispatch-time policy (#747) already prefers the CPU pod
lane for this class, so the recovery CORRECTS the placement rather than
retrying it. Any kill NOT matching that carve-out (unprotected kill, no
fleet-wide storm attribution, or a non-API-bound phase class) keeps the
existing ladder: relaunch ONCE
with protection verified (`choom=ok`); if a PROTECTED phase is earlyoom-killed
AGAIN, the VM is structurally memory-contended for this phase (or the phase
itself is now the top consumer) — route it to the cheap CPU pod lane
(`cpu-mid` / `cpu-bigmem` by footprint, CLAUDE.md § CPU-only phases) instead of
a third VM relaunch. The phase-is-the-spiker variant + the stream-reduce rule
stay in `.claude/rules/gotchas.md` (earlyoom entry).

**The thread-cap `env` prefix is REQUIRED on every VM-side launch (#891).** The
shared-VM setdefault (#847, `orchestrate/env.py`) is `src/`-side and pinned to
the WORKTREE's branch point — the Step 5a spec-freshness sync deliberately
never syncs `src/` — so an in-flight worktree cut before an infra fix launches
with the pre-fix library (#779). The explicit prefix
is branch-age-independent AND caps torch/BLAS regardless of the script's import
order (the env.py hook cannot in-process-cap a script that imports torch before
`load_dotenv()`; the launch env can). `env` execs `<cmd>` in place, so `$!` pid
capture and the `ps -o args=` identity verify are unchanged. env.py's
setdefault never clobbers these values. A phase that genuinely needs a wider
cap on the shared VM states the wider explicit value + a one-line reason in its
breadcrumb. Pod / GCE / SLURM launches NEVER carry the prefix — dedicated boxes
keep full width (the #847 scope invariant).

EXEMPT: work that COMPLETES within a subagent's own
bounded turn (a subagent that bg-launches a VM-local >~15-min phase and
returns follows this SAME detach shape — the phase sits in the session's kill
domain either way), pod-side workloads (the pod launch contract / #883), the
deadline-bounded off-pod `batch_judge` poll, and quick probes / plots
(< ~15 min). Off-VM routing (CLAUDE.md § CPU-only phases — the cheap
dedicated CPU-pod lanes) remains the FIRST preference for long CPU phases;
this convention governs the residual phases that legitimately stay VM-local
and is not a permission slip for long VM-side compute.

**Successor / re-entry rule (overrides the freshness window below).** When the
current stage+round's most recent breadcrumb carries `pid=`, probe
`ps -p <pid> -o args=` BEFORE any re-dispatch decision — the SAME identity
verify as at launch, never a bare liveness check (on a shared VM a recycled
pid would otherwise "re-attach" to a stranger and suppress the needed
relaunch). A pattern-based FALLBACK probe (pid absent / recycled — `pgrep -f`
against the distinctive invocation) uses the bracket idiom per the
§ Probe-bracket rule above; an ALIVE read from an UNBRACKETED pattern probe is
UNVERIFIED evidence and never suppresses the relaunch path. Alive AND args
match the distinctive invocation → the phase is IN
FLIGHT regardless of breadcrumb age (a detached multi-hour phase posts no
markers while computing): RE-ATTACH — poll the pid, `tail` the breadcrumb's
`log=` for real progress (alive ≠ progressing; the log is the progress
signal), post a liveness `epm:progress` note — never relaunch. GCE log-read
note (#1764): on a GCE instance the workload log is root-owned (the
startup-script workload runs as root; the OS-Login SSH user is in
`google-sudoers`, so passwordless `sudo -n` is available — `backends/gcp.py`
`_drain_sentinels` docstring, the #608 sentinel pull's own `sudo -n cat`
precedent), so a bare `tail`/`cat` CONTENT read fails `Permission denied` —
retry it as `sudo -n tail -50 <log>` (fallback `sudo -n cat <log> | tail -50`).
An EACCES is a probe artifact, NEVER evidence the log is frozen/missing or the
phase dead; mtime/`stat` probes need no read permission and are unaffected.
RunPod pods SSH as root, so the class is GCE-specific (#1738). Dead — or an
args MISMATCH (recycled pid: treat as dead) — with completion output present
at the breadcrumb's `harvest=` path (§ Harvest contract; pre-contract
breadcrumbs lack the token — fall back to log-tail + known output dirs) →
stage done; RUN THE HARVEST — collect the declared outputs, commit/upload
them per the Upload Policy, fold them forward — then proceed. An EMPTY
`harvest=` path beside a log tail showing clean completion is a
declared-path mismatch (typo / divergence), not a failed run — cross-check
the log's real output locations before treating the phase as failed.
Dead with no completion output →
genuinely failed: run the kill-before-relaunch probe
(`.claude/rules/crash-fix-rounds.md` § Kill-before-relaunch), then relaunch.
`stage_dispatch_should_skip` knows nothing about pids, so the polling
orchestrator also refreshes the mechanical window with periodic liveness
`epm:progress` notes (those refresh the events.jsonl effective-age window;
the watcher's stall detector reads the session SELF-REPORT, so a long-idle
session may still be stopped — benign under this rule: the detached phase
survives and the successor re-attaches); the identity-verified pid probe is
the authoritative check at re-entry. Prefix those periodic liveness notes
with `[long-phase-heartbeat]` — the watcher's stalled detector AND
`tick_triage.py` (#1051) grant the 90-min leash on that opt-in; tick_triage
probes the breadcrumb's `pid=` (a VM-LOCAL pid, start-time identity-guarded
— never post a pod-side pid in a `stage-dispatch` breadcrumb) before any
STALE-REDRIVE, and while that pid breadcrumb is in flight the pid evidence
OVERRULES heartbeat notes. (This convention is the detached-phase
instance of the § Long-phase heartbeat duty, Step 6d.2 — the ≤45-min
resume structure, the verify-first ban, and the self-report refresh
there bind here too.)

**Checkable guard rule (run at Step 9 / Step 8 entry on every
re-invocation).**
1. Read the most recent events.jsonl marker via
   `task.py latest-marker <N>` (and `task.py view <N> --json` for the tail
   if needed).
2. Scan `events.jsonl` BACKWARDS for the CURRENT stage+round's most
   recent `stage-dispatch` breadcrumb (an `epm:progress` note BEGINNING
   `stage-dispatch ` — a note merely quoting the string mid-note never
   counts), skipping ALL other kinds — intervening markers (codex-task
   markers, progress notes, plan markers, other stages' breadcrumbs and
   result markers) never hide a breadcrumb. If such a breadcrumb exists
   AND no result marker for THAT stage (nor `epm:failure`) was posted
   after it, compare its EFFECTIVE age to the **stage-aware freshness
   window** — effective age is measured from the LATEST of the
   breadcrumb and any subsequent liveness marker (`epm:codex-task-*`,
   `epm:smoke-architecture-check`, `epm:proposed-tests`, or a
   non-breadcrumb `epm:progress` — excluding anti-liveness notes: a
   `deliberate-stop` stop record and `[autonomous_session_watch:...]` /
   `[spawn-session:...]` telemetry never refresh the window, #810/#949):
   a healthy long-running round keeps
   refreshing its window; a dead one goes silent and re-dispatches once
   the window expires. The mechanical form of this rule is
   `task_workflow.stage_dispatch_should_skip` (run the pre-dispatch
   one-liner above — do not eyeball the scan).
   - Window = **30 min** for Codex-ensembled rounds (ALL `interpreting`
     AND `clean-result` rounds up to the per-reviewer cap (5) — every round spawns both the Claude
     critic AND a `codex-*-critic` twin at `--effort high|xhigh` via
     `companion task` under the all-rounds policy; such
     rounds commonly exceed 15 min wall time).
   - Window = **15 min** for everything else (`verifying` and any other
     Step 8/9 stage).
   - **effective age < window** → the subagent is presumed STILL
     RUNNING. EXIT the skill cleanly (`post_step_completed.py ...
     --exit-kind parked --notes "stage <stage> round <r> still in flight
     (dispatched <Δ>m ago, window <W>m); backstop tick yielding"`). Do
     NOT re-dispatch — let the live work finish; the next tick (or the
     live subagent's own completion) advances the pipeline.
   - **effective age >= window** → the stage looks genuinely STALLED (a
     subagent that never posted its result). Proceed to re-dispatch it
     normally (the freshness window is what distinguishes "live" from
     "stalled").
3. If the per-stage backwards scan finds NO open breadcrumb for the
   current stage+round (none exists, or a result marker / `epm:failure`
   postdates it), there is no in-flight work — proceed with the normal
   Step 9 logic below.

**Parallel-stage note (results-landed spawn).** Step 8's results-landed
parallel spawn can put `verifying`, `interpreting` round 1, and
`methodology-reference` breadcrumbs in flight at once. The per-stage
backwards scan in step 2 above applies to each concurrent stage
independently — a result marker for stage X never clears stage Y's
in-flight state.

The 15-min default comfortably exceeds a single Claude analyzer / critic /
verifier turn; the 30-min Codex-ensemble window covers a high-effort
Codex twin's wall time without re-dispatching live work and risking a
double-writer on `body.md`. Both fit cleanly under the 45-min backstop
cadence, so a genuinely stalled stage is still re-dispatched within
~2 ticks (≈90 min worst case). This guard is the
bound referenced by the Step 6d.2 "surviving the backstop into
verifying/interpreting/reviewing is DESIGNED behavior" paragraph.

**Limitation (be explicit about it).** A MISSED `stage-dispatch`
breadcrumb (the orchestrator spawns a stage subagent but forgets / fails
to post the breadcrumb FIRST) silently disables this guard for that tick:
with no breadcrumb to detect, step 3 of the rule fires and the
orchestrator re-dispatches the stage as if no in-flight work existed —
exactly the double-dispatch / double-writer the guard exists to prevent.
The breadcrumb is the only enforcement; the orchestrator MUST treat
posting it as a non-skippable precondition for every Step 8/9 stage
dispatch. If you notice a stage subagent was spawned without one, post
the breadcrumb immediately (`task.py post-marker ... epm:progress --note
"stage-dispatch stage=<s> round=<r> subagent=<name> worktree=<abs path or
'repo-root'>"`) so the next tick's guard fires correctly.

**9a. Iterative interpretation** (only if status is `interpreting`)

Only for `experiment` tasks. Code-change tasks never reach this step
because Step 5 already PASSed code-review and routed them to Step 9c
(the inline test-verdict gate) directly.

The interpretation loop produces a polished clean-result body through
iterative refinement between the analyzer and an interpretation-critic.
Worktree-cwd sessions run the Step 5a spec-freshness check before the
first dispatch of this loop (analyzer + critic specs load from the
worktree copy).

**Round 1:**

**Held-output publish (results-landed early spawn).** When Step 8's
results-landed parallel spawn already ran the analyzer first pass in
HOLD-marker mode, do NOT re-spawn it here: post the held
`/tmp/issue-<N>-interpretation-v1-held.md` verbatim as
`epm:interpretation v1` (this happens immediately after
upload-verification PASS, per Step 8's join #1) and continue at round-1
step 2 (the critic ensemble). Fall through to the normal spawn below
only when no held output exists (early spawn skipped, crashed, or
discarded by Step 8's gap-fill decision rule).

1. Spawn `analyzer` agent (fresh context) with raw result paths. The
   analyzer:
   - Writes the **Fact Sheet** (reproducibility card, artifact URLs,
     raw numbers, plots, sample outputs) — this is written once and not
     revised.
   - Writes the **Interpretation** (background, methodology, results
     claim + hero figure + main takeaways + confidence, next steps).
   - Generates plots via `paper-plots` skill, saves them under
     `figures/issue_<N>/`, commits + pushes them to `main` BEFORE
     writing the body, and references each figure INLINE inside the
     relevant `### <finding>` H3 under `## Findings` (no separate
     `## Figure` H2 — that H2 is retired) via
     `![alt](https://raw.githubusercontent.com/<owner>/<repo>/<sha>/figures/issue_<N>/<file>.png)` —
     a SHA-pinned absolute URL the dashboard can fetch. Relative
     `artifacts/...` / `figures/...` URLs render as broken images on
     the dashboard and are rejected by `verify_task_body.py` Check 4b
     (#365). See
     `.claude/agents/analyzer.md` Step 3 for the full save-commit-pin
     workflow.
   - Posts `epm:interpretation v1` on the source task.

2. Spawn the **interpretation-critic ensemble** (fresh contexts, single
   message, both `run_in_background=true`):
   - `interpretation-critic` (Claude) — full 7-lens review. Posts
     `epm:interp-critique v1` with PASS or REVISE.
   - `codex-interpretation-critic` (Codex gpt-5.5 via `companion task`)
     — same 7 lenses (lens 6 plot-prose works on Codex multimodal).
     Posts `epm:interp-critique-codex v1`.

   Quota-sentinel pre-check first (#1204, CLAUDE.md § Codex ensemble
   review): when LIVE, spawn only the Claude critic; instant confirmed
   Codex no-show per the decision-table no-show row + one `epm:progress`
   note.

   Neither sees the analyzer's reasoning. Independence is load-bearing.

3. **Apply ensemble decision rule** (see
   (see workflow.yaml § ensemble_review)):

   | Claude | Codex | Action |
   |---|---|---|
   | PASS | PASS | `final_verdict = PASS`. Concatenate suggestions for analyzer's optional polish. |
   | REVISE | REVISE | `final_verdict = REVISE`. Union the revision requests (dedup exact-same). |
   | PASS vs REVISE (or vice versa) | (the other) | Spawn `reconciler` (marker mode). Brief: role=`interpretation-critic`, both event bodies (trigger-dense round: by reference per § File-only Codex verdict posting), interpretation body path, eval JSON paths, figure paths. Reconciler posts `epm:review-reconcile v<n>` with binding PASS or REVISE. `final_verdict = reconciler's verdict`. |
   | Codex no-show (`epm:failure` posted, or NO durable verdict per the Step 5b durable-verdict-first rule) | (any) | Fallback: `final_verdict = Claude verdict`. Surface "Codex twin no-show round <n>" to chat. |

   An Agent-tool error for EITHER critic first triggers the Step 5b
   durable-verdict-first check (re-read events.jsonl for
   `epm:interp-critique[-codex] v<n>`, then the round-fresh Codex output
   file): a thrash-killed summary turn with a posted verdict is a RETURNED
   reviewer; a Claude critic with no durable verdict is re-spawned once
   per the Step 5b bound, not skipped.

   Reconcile rounds do NOT increment the per-reviewer round counter.
   Adopt-more-severe WITHOUT a reconciler is unsanctioned here
   (the #825 deviation site) — see the Step 5c ban: when both reviewers
   returned disagreeing durable verdicts, spawn the reconciler; a
   twice-dead reconciler fails LOUD per Step 5b item 4 (the
   adopt-more-severe fail-safe is `/adversarial-planner`-in-context
   only), and the Codex no-show fallback remains a separate, sanctioned
   path.

**If `final_verdict == REVISE` (rounds 2-5):**

Re-spawn analyzer (fresh context, sees original data + ALL critique
feedback: Claude event + Codex event + reconcile event if any)
(trigger-dense round: critique events by reference — marker kind+version
/ output-file paths, per § File-only Codex verdict posting; never inline
the critique bodies).
Analyzer posts `epm:interpretation v2`. Re-spawn the ensemble (fresh
contexts, sees v2 + prior critique events). Posts both
`epm:interp-critique v2` and `epm:interp-critique-codex v2`. Apply rule
again. Round boundaries here carry the Step 5c-quater round-boundary
durable-decision duty (decision note + explicit-path commit BEFORE the
re-spawn).

**Max 5 rounds per reviewer.** At round 5 (the cap) with a non-PASS
ensemble verdict, apply the Step 9a-bis-style procedural-only strip once
more (procedural / presentation REVISEs). If ALL residual REVISEs are
stripped → advance with full critique history. If ANY SUBSTANTIVE
residual remains — a flagged OVERCLAIM the strip cannot resolve — SURFACE
it, do NOT auto-publish into the record (this is the MOST important site
for surface-not-ship, #784: a real residual at interp is an overclaim
that must never be silently promoted). Either way post the §5 marker
first (`uv run python scripts/post_step_completed.py --issue <N>
--step 9a --exit-kind parked` interactive / `--exit-kind failure-exit`
autonomous). Interactive: present the residual
to the user + EXIT. Autonomous (`EPM_AUTONOMOUS_SESSION=1`): post
`epm:failure v1 failure_class: code` referencing the residual, set
`status: blocked`, fire `PushNotification`, run CRON-TEARDOWN, EXIT
(halt_criteria id=6 `concern_unresolved` family).

**On PASS (or all-stripped at the cap):**

The analyzer **promotes the source task IN PLACE to a clean-result** —
no separate task is created. The analyzer:

1. Snapshots the prior body to `original-body.md` via an
   `epm:original-body v1` event (audit / rollback).
2. Replaces `body.md` with the polished markdown write-up:
   ```bash
   uv run python scripts/task.py set-body <N> --file /tmp/clean-result-body.md
   uv run python scripts/task.py set-title <N> "<claim summary> (HIGH|MODERATE|LOW confidence)"
   uv run python scripts/task.py set-clean-result <N>   # flips has_clean_result=true
   ```
3. Runs `scripts/verify_task_body.py <body-file>` — FAIL blocks the
   write-up.

Posts `epm:clean-result-drafted v1` on the source task with the title
and a 2-sentence recap.

Then proceed to **9a-humanize (clean-result prose humanize-loop pass)**
before advancing to clean-result-critic.

**9a-humanize. Clean-result prose humanize-loop pass** (orchestrator-level
— only on the first time `epm:clean-result-drafted v1` is posted, NOT on
round-2/3 revisions out of 9a-bis)

The analyzer ran an inline humanize-quick self-pass on the reader-facing
prose during its draft (analyzer.md Step 4.5). This orchestrator step adds
the second-opinion layer: a real `/humanize loop` invocation with a
separate hostile critic subagent the analyzer could not spawn from inside
its own subagent context.

The pass targets the v3 reader-facing prose surfaces — `## Takeaways`
(the bullet block Thomas adapts for Slack) + `## What I ran` + the
`## Findings` setup/read prose (bullets). This is exactly what Thomas
reuses verbatim for Slack and the rolling cross-round synthesis, so its
register matters most. The `## Data` capsules + example blocks,
`## Reproducibility` appendix, and figure captions are OUT of scope —
they carry project jargon on purpose, and the clean-result-critic in
9a-bis enforces register discipline on them. (Legacy/in-flight v2 bodies:
the pass targets the `## TL;DR` block — `<section id="tldr">` for the HTML
card — instead; branch on the body sentinel.) Expect the pass cheaper
than the v2 era — the v3 surfaces are bullets at ~800 words, not a
multi-paragraph LessWrong narrative.

**Paper-mode (`paper: true`): SKIP this orchestrator-level pass.** A
paper-task's reader-facing prose lives in the `.tex`
(`docs/papers/issue_<N>/issue_<N>.tex` Abstract / Introduction / Results
interpretation / Discussion), not a markdown `body.md` to extract — and
the analyzer already ran `/humanize academic` (em-dash zero-tolerance,
copula avoidance, classical academic terms) on those paper surfaces
INTERNALLY during its PAPER-TASK MODE Step 4.5 (`.claude/rules/analyzer-paper-mode.md`
§ PAPER-TASK MODE). Post `epm:humanize-loop v1` with `note: skipped —
paper-task (analyzer ran inline /humanize academic on the .tex)` so the
audit log records it, and proceed straight to 9a-ter.

**Procedure:**

1. Read the published body via `task.py view <N>`; extract the v3 prose
   surfaces (`## Takeaways` + `## What I ran` + `## Findings` setup/read
   bullets; for a v2/legacy body extract the `## TL;DR` block instead).
2. Invoke `/humanize loop` with those prose surfaces as the target.
   **Read the
   draft file once BEFORE the first Edit on it (and re-Read after any
   compaction)** — the draft is typically written by the critic subagent, so
   it is not in the orchestrator's Edit state, and blind Edits bounce with
   "File has not been read yet" (10 such rejections across three
   sessions, 8 consecutive in one humanize pass). The skill
   spawns a hostile critic subagent (from the orchestrator's context —
   allowed; the analyzer could not because subagent-from-subagent is
   forbidden) that scores against the six-axis rubric:
   - vocabulary (AI-tell words)
   - structure (rule-of-three, negative parallelisms, inflated symbolism)
   - rhythm (sentence-length monotony, metronomic cadence)
   - voice ("we"-slippage, corporate hedging, promotional language)
   - interpretation honesty (buried caveats, misplaced hedging)
   - results-writing discipline (effect sizes / named tests in prose,
     Δ-notation, undefined jargon — anti-patterns from CLAUDE.md
     "Statistics" rules and the clean-result-critic statistical-framing
     lens)

   **Hard ban gate scoping (binding; #498/#518/#923):** the
   `/humanize` skill's mandatory `check_bans.sh` absolute-ban gate runs
   over AUTHORED PROSE ONLY — for clean-result work the ELIDED copy below
   IS the ban-gate input (a repo-side override of the user-global skill's
   whole-body gate wording), never the raw whole body. SPEC-required
   verbatim sample completions legitimately contain ban-listed strings
   ("Certainly!", "Sure, I'd be happy to help"), and rewriting them to
   satisfy the gate destroys scientific evidence. Gate the body file —
   `/tmp/issue-<N>-humanize-loop.md` when the loop produced revisions; if
   the loop made no revisions, materialize the current body to that path
   first — AFTER eliding the verbatim-quotation surfaces: fenced ``` blocks,
   `<details>...</details>` example blocks, `>`-blockquoted lines (with or
   without a following space), and `**Completion:**` sample lines:
   ```bash
   awk '/^```/{f=!f; next} f{next} /^<details/{d=1} d{if(/<\/details>/)d=0; next} /^>/{next} /^\*\*Completion:\*\*/{next} {print} END{if(f||d) exit 3}' \
     /tmp/issue-<N>-humanize-loop.md > /tmp/issue-<N>-ban-scan.md \
     && ~/.claude/skills/humanize/check_bans.sh /tmp/issue-<N>-ban-scan.md
   ```
   awk exit 3 = structurally unbalanced body (unclosed fence/`<details>`) —
   a hard workflow error: the gate does NOT run; fix the body structure
   and re-run. A hit SURVIVING elision is PRESUMPTIVELY authored prose —
   default: real FAIL, rewrite it; if inspection shows it is verbatim
   sample text the elision missed (indented fence, inline `<details>`,
   multi-line completion), strengthen the elision instead and document
   the disposition — NEVER rewrite the sample. A hit whose ONLY
   occurrences were elided is a FALSE POSITIVE: treat the gate as PASS on
   authored prose, NEVER rewrite the sample, and DOCUMENT the disposition
   in the `epm:humanize-loop` note (step 5), naming the banned string AND
   its location. Never move authored prose into a blockquote/fence to
   dodge the gate.

3. Loop until all axes score ≤ 1 OR **3 orchestrator-level cycles**
   reached.
4. If the loop revised the prose surfaces, write the new body to
   `/tmp/issue-<N>-humanize-loop.md`, then VERIFY THE CANDIDATE FILE
   FIRST and apply only on PASS (#1860; the pre-#1860 apply-then-verify
   order left a briefly-live non-compliant body on a FAILing candidate —
   #1775):
   ```bash
   uv run python "$REPO_ROOT"/scripts/verify_task_body.py --file /tmp/issue-<N>-humanize-loop.md  # main-checkout copy, never the worktree's (spec-stale risk, #496)
   uv run python scripts/task.py set-body <N> --file /tmp/issue-<N>-humanize-loop.md  # ONLY on candidate PASS
   uv run python "$REPO_ROOT"/scripts/verify_task_body.py --issue <N>  # post-apply confirm: frontmatter-coupled checks --file cannot see (e.g. H1 == frontmatter title)
   ```
   The CANDIDATE verifier MUST PASS before the apply — the humanize loop
   is not allowed to produce a body that breaks Lens 1-15 mechanical
   checks. On a candidate FAIL: iterate ON THE CANDIDATE FILE (fix the
   flagged prose), up to 2 candidate-fix iterations (independent of the
   rubric's 3-cycle cap); if no passing candidate emerges, apply NOTHING
   — the pre-loop body (which already passed the Step 9a verify) stays
   live, and the step-5 note records the residual via the existing
   "exited at cap, residual debt: ..." grammar. The live body is only
   ever replaced by a verified-PASS candidate. If the post-apply --issue
   confirm FAILs (rare — frontmatter-coupled drift only): revert to the
   pre-loop body and surface the conflict to the user, as before.
5. Post `epm:humanize-loop v1` on the source task with the final 6-axis
   scores + a one-line note ("converged in cycle K" or "exited at cap,
   residual debt: axis X scored 2 — flagged to user"). When the ban gate
   recorded a verbatim-sample false positive, append the disposition to
   the note, naming the string and its location (the #923 form: "ban
   gate: PASS on authored prose; 1 hit ('Certainly!', ## Methodology
   sample block) — false positive, left in place").

**Skill availability fallback:** if `/humanize` is not loaded in the
runtime (plugin missing), skip 9a-humanize entirely and proceed to
9a-ter. The analyzer's inline Step 4.5 already provided a first-pass
cleanup; the orchestrator pass is additive. Post
`epm:humanize-loop v1` with `note: skipped — /humanize skill not
loaded` so the audit log records the skip.

**Then proceed to 9a-ter (auto-run free-analysis follow-ups).**

**9a-ter. Auto-run free-analysis follow-ups** (only if status is
`interpreting`, after Step 9a-humanize completes)

The analyzer's Step 6.5 (and the follow-up-proposer's `cost_class` /
`est_gpu_hours` schema) record whether any follow-up is executable
with ZERO new GPU (`cost_class: free-analysis`, `est_gpu_hours: 0`).
When such a follow-up exists and has not yet been run on this task, the
orchestrator AUTO-RUNS it inline BEFORE the clean-result-critique gate
(9a-bis) — so the critic gates the UPDATED body, not a body that
already names a free win it didn't take. **The `headline_affecting: yes`
requirement is DROPPED** — a zero-GPU follow-up auto-runs
whether or not it would move the parent's headline (the standing
directive: follow-ups that are 0 GPU-h or `< 20` GPU-h just run and fold
into the same issue). This 0-GPU inline step is the floor of the
cheap-auto-run band; the GPU-backed `0 < est_gpu_hours < 20` band runs at
9b via the same-issue follow-up loop. This step fires in BOTH
interactive and autonomous (`EPM_AUTONOMOUS_SESSION=1`) sessions
identically (as does the 9b cheap band — the
remaining autonomous-ONLY routing at 9b is the `est_gpu_hours >= 20` /
`auto_run: yes` expensive path: same-issue loop for `same`, child filing
for `substantially-different`). The whole
<!-- example: anti-pattern -->
step is auto-continue (NOT a new
`AskUserQuestion` gate); the halt-criterion contract is preserved.
<!-- autonomous-mode: auto-resolve -->
Same behavior in interactive and autonomous sessions: no
AskUserQuestion is ever raised by this step; the marker
`epm:free-analysis-followup-run v1` is the durable record consumed by
re-entry idempotency.

**Detection.** Read the latest analyzer output (the `## Free-analysis
follow-ups (orchestrator: auto-run before parking)` H2 block in its
return text — see analyzer.md Step 6.5) AND the latest `epm:analysis
v<n>` marker on the source task (its `free_analysis_unrun:` field).
Take the union. For each entry:

1. Skip it if an `epm:free-analysis-followup-run v1` marker on this
   task already records that follow-up as run (idempotency — match by
   the verbatim follow-up title field).
2. Skip it if the implementer (below) reports the follow-up is NOT
   actually free-analysis (e.g. it discovered the change needs new
   eval data after all) — see ABORT path below.

The orchestrator MAY additionally sanity-check that the eval-data
path(s) an entry names actually resolve (local file exists /
`huggingface_hub.list_repo_files` for HF paths) before dispatching; an
entry whose premise path does not resolve takes the ABORT path's
reclassification up front (post the `epm:free-analysis-followup-run v1`
abort record naming the missing artifact) without burning an
implementer round. The analyzer's Step 6.5 artifact-premise check is
the primary defense; this is a backstop (#552).

When the detection union is empty, this step is a no-op: log one chat
line (`No free-analysis follow-ups to auto-run`)
and proceed directly to 9a-bis. (Detection no longer filters on
`headline_affecting` — every unrun `cost_class: free-analysis` follow-up
is eligible.)

**Loop guard (critical).** This step caps at AT MOST ONE free-analysis
follow-up run per task. The cap is enforced by the
`epm:free-analysis-followup-run v1` marker: re-entry into 9a-ter on the
same task — whether from a backstop tick, an analyzer revision posting a
new free-analysis follow-up, or a 9a-bis REVISE round that bounced back
to analyzer — checks the marker FIRST and exits without dispatching if
it is already present (regardless of whether the listed follow-up is
the same one). The marker-present exit is ordered, not silent: marker
present → read the detection union (the analyzer output + the latest
`epm:analysis` marker's `free_analysis_unrun:` field, per § Detection
above) → post the § Cap-park surfacing note below for each unrun
eligible entry not already noted → exit. This prevents the re-run from
triggering another auto-run chain within the same task. A further
free-analysis follow-up STAYS listed in the body as a regular bullet,
but the bullet is no longer the only surface: whenever the cap excludes
a concrete unrun `cost_class: free-analysis` entry, post the § Cap-park
surfacing note below (#1548; #958). Across tasks the mechanism stays
fresh (each task gets its
own one round).

**Cap-park surfacing (#1548 — SURFACING only: the one-round cap above
is unchanged, no new auto-run, no new gate, no new marker kind).** Two
firing moments: (a) a loop-guard re-entry exit whose detection union
still lists ≥1 unrun eligible entry (the ordered marker-present exit
above — read the union, post, then exit); (b) immediately after
Auto-run procedure step 6 posts the `epm:free-analysis-followup-run`
marker (run OR abort) when the detection union listed >1 eligible
entries — the non-selected surplus is cap-parked from that moment, not
at some future re-entry. At either moment, for EACH cap-parked entry
post one structured `epm:progress` note (the `stage-dispatch` /
`deliberate-stop` convention — reuse the kind, never mint one):

```bash
uv run python scripts/task.py post-marker <N> epm:progress \
  --note "followup-parked-by-cap followup_ref=<verbatim follow-up title> \
    rank=<1-based position in the analyzer's surfaced order, or 'unranked'> \
    screened=<not-redundant|pending-screen> cost_class=free-analysis \
    cap_consumed_by=<followup_ref of the latest epm:free-analysis-followup-run row> \
    alternative=raise-9a-ter-cap-or-manual-pickup — the one-round cap parked \
    this follow-up; a future planner/human may weigh raising the cap (a \
    deliberate workflow change) vs manual pick-up post-promotion"
```

The fixed leading token `followup-parked-by-cap` is the PM-surfaceable
signal: the note is dashboard-visible on the events timeline the
promotion review reads, and greppable by PM tooling
(`grep -h followup-parked-by-cap "$(uv run python scripts/task.py find <N>)/events.jsonl"`).
**Idempotent per (task, verbatim follow-up title):** before posting,
scan the task's existing events CONTEXT-CHEAPLY — grep the events file
directly (`grep -F 'followup-parked-by-cap' "$(uv run python scripts/task.py find <N>)/events.jsonl"`,
then match the candidate's verbatim `followup_ref=` value in the hits),
or pipe `task.py view <N> --json` through a `jq`/python filter over the
marker notes — never a full-body page-in — for an `epm:progress` note
containing BOTH `followup-parked-by-cap` AND the same verbatim
`followup_ref=` value: present ⇒ skip, so backstop-tick /
9a-bis-REVISE re-entries never double-post (the mirror of the run
marker's match-by-verbatim-title idempotency). Skip entries already
recorded by an `epm:free-analysis-followup-run` row (run or aborted) or
parked by `epm:followup-parked-redundant v1` (each has its own durable
surface). `screened=` carries the follow-up-critic verdict when the
screen has run for that proposal set; otherwise `pending-screen`.

<!-- example: anti-pattern -->
Auto-continue: the note is a non-blocking side channel — never an
`AskUserQuestion`, never a pause, never a status change.

**Compute-character pre-launch statement (REQUIRED — one paragraph, not a
planner round, not a gate).** "0 GPU-h" does not mean "0 compute review":
this step, the Step 9b same-issue follow-up loop, and the CLAUDE.md
§ Routing "User-chat inline free analysis" carve-out are the workflow's
PLANNERLESS paths — they skip the planner+critic stack, where all
compute-character review lives (incidents #667/#722/#778: reused serial
parent code burned hours on "0 GPU-h" work, caught only by ad-hoc human
watches). Before dispatching any stage that launches a fit, sweep, or
statistical battery (permutation/bootstrap/null-draw batteries,
per-cell/per-fold fits, per-row model calls), the dispatcher STATES, in
the stage-dispatch `epm:progress` breadcrumb note (or an
immediately-adjacent `epm:progress` marker): (1) the ops arithmetic —
cells × folds × draws × epochs and the projected wall-time it implies;
(2) the NAMED batched helper implementing the inner loop (e.g.
`analysis/vectorized_mlp_skill.py`; the batched `perm_null_draws` in
`analysis/null_battery.py`), or why the work is genuinely not batchable;
(3) for reused parent code, that its inner loop, device routing, + data-repo
Hub-call scoping were
INSPECTED, not assumed (cf. `.claude/rules/artifact-reuse.md`); (4) for any
VM-PLACED phase, the projected peak RSS (measured one-chunk `ru_maxrss` at
production shape, or resident-pool bytes × MEASURED live-factor —
`.claude/rules/plan-compute-sizing.md` § CPU-phase RAM/RSS routing);
projected peak RSS ≥ ~16 GB — single phase, or summed with
concurrently-resident VM phases — is a STOP: route the phase off the
shared VM (`cpu-mid` / `cpu-bigmem`) before launching (#778's 22-GiB
battery was earlyoom-killed 3× on exactly this plannerless path; #833 lost
5 cells to two concurrent ~13-15 GB phases); (5) for any stage that downloads
or materializes ≥ ~5 GB of artifacts (HF snapshots, tensor stores, staged
corpora) — whether or not the round has a fit/battery stage — the staging
path, named UP FRONT, with its off-`/` routing (PRIMARY) and its filesystem
headroom (SECONDARY): multi-GB staging NEVER lands on `/` (the shared boot
disk) or `/tmp/` (#1393 incident: a 14 GB inline HF pull on #823 filled `/`
→ ENOSPC, orchestrator Bash output lost) — route it to the janitor-swept
`data/issue_<N>/hf_dl/` layout wherever that path resolves OFF `/`, else to
an existing user-writable per-issue dir on the data disk
(`/mnt/eps-data/$USER/issue<N>_<slug>/` — the established `issue823_work`
convention; NEVER a fresh top-level `/mnt/eps-data/<dir>`: the top level is
root-owned and the `mkdir` fails, the incident's second failure), threading
`HF_HOME` / `local_dir` so the hub cache follows; the SECONDARY headroom
check verifies the filesystem the staging path resolves to (`df -P <path>`)
has free headroom ≥ ~1.5× the projected bytes (headroom for partial shards,
retries, and cross-filesystem cache→`local_dir` copies; the routing mandate
binds even when the headroom probe passes — #823 projected ~6 GB, realized
14 GB). And when the staged/materialized FIT/ANALYSIS INPUTS reach ≥ ~50 GB
(`VM_ANALYSIS_FOOTPRINT_GB_MAX`), the disk routing alone is NOT enough — the
CONSUMING phase itself ROUTES OFF the shared VM at dispatch (`cpu-bigmem` via
`dispatch_issue.py --intent cpu-bigmem`, or a pod), never launched VM-local
to be rerouted after deaths (#1345: a 65 GB boundary-round fit died silently
4× over ~2.5 h on the shared VM before the cpu-bigmem reroute the plan-time
carve-out prescribes). While the #681 worktree bind-mount is pending, the worktree's own
`data/` dir resolves to `/` — exactly what the `df -P` probe catches.
Projected wall-time > ~15 min for any fit/battery stage additionally makes
element (1)'s per-call basis MEASUREMENT-REQUIRED: run a 1-cell/1-unit pilot
THROUGH the production entrypoint at production shape (batch width included)
FIRST — an asserted or guessed per-call cost is never a sizing basis
(`.claude/rules/plan-compute-sizing.md` § Per-cell fit phases) — state the
measured per-cell wall in the dispatch note, and size EVERY self-set
timeout/fence (`timeout(1)` bounds, watchdog kills, run-duration caps) ≥2×
the pilot-extrapolated wall (measured per-cell wall × remaining cells /
parallelism; the ×2 is the p90-style dispersion default when only a 1-cell
pilot exists — § p90 fence sizing + the #1092 `pilot-gated` ≥2× presumption).
A cited prior-issue MEASURED figure for the SAME kernel + shape may stand in
for the pilot (the ported rule's own alternative basis) — a guess never can.
A teammate/inline run NEVER sets a fence below that bound, and NEVER asserts
a user-facing wall-time estimate from a guessed per-call basis (2026-07-23,
#1092 session f4b1d707: a guessed self-set `timeout 3000s` killed its own
healthy ~25 min/cell full run at exit=124 — relaunch+resume — and two
same-day chat wall-time estimates were off by ~an order of magnitude).
Projected wall-time > ~1h without a batched inner loop is a STOP: vectorize first
(`.claude/rules/vectorize-many-cell-fits.md`), then launch. And an
ITERATIVE-OPTIMIZATION fit leg (gradient descent on parameters — a torch-MLP
LOCO, per-cell probes via SGD/AdamW; the CLAUDE.md compute-character
carve-out class) whose projected PHASE wall-time on CPU, after vectorization,
exceeds the carve-out's ~15–30 min floor ROUTES to a GPU lane at dispatch
(`lora-7b` / `eval` / `debug`, smallest that fits) — a many-cell loop of
individually-fast fits counts, per-cell > ~15 min is sufficient by itself,
and GPU-worthiness is decided AT DISPATCH, never behind a descope-if-slow or
run-CPU-and-see gate (#1768: an inline 16-cell MLP battery at ~10–20 min/cell
dispatched CPU-bound; the user had to order 'just run on GPU', where it
finished in minutes). If the
realized implementation later adds a fit/battery the dispatch statement
did not cover — or materially changes its arithmetic — an updated
statement is posted before that launch. A round with no fit/battery stage AND no ≥ ~5 GB download/staging states one line: `compute-character: no fit/battery stages, no multi-GB staging`.
A statement covering a VM-side phase >~15 min ALSO names the detached launch
shape + log path + the thread-cap `env` prefix (OMP/MKL/OPENBLAS/NUMEXPR=8 — #891;
or the wider explicit value + one-line reason) + the earlyoom protection state
(`choom=ok|failed`) **+ the harvest contract (the durable
out-root + the `harvest=` token)** per the Step 9 entry-guard
§ "Detached VM-side long compute phases" convention.
Routing, auto-continue behavior, and the marker schema are unchanged.

**Inline measurement-design + figure-sanity duties (REQUIRED — statement/check
duties, not a gate; auto-continue unchanged).** Same rationale as the
compute-character statement above: this step and the CLAUDE.md § Routing
"User-chat inline free analysis" carve-out are PLANNERLESS — they skip the
planner+critic stack, where the both-arms mapping review (planner.md §4 /
critic.md Methodology lens) and the interpretation-critic's figure-load check
(Lens 6) live. Two duties, siblings of — not additions to — the five-element
compute-character statement above:
(1) **Both mapping arms.** A round that computes a representation mapping — a
geometry read, predictor, probe, or direction extraction over model
activations — states in the dispatch-time `epm:progress` breadcrumb (or an
immediately-adjacent `epm:progress` note) that BOTH arms run: prefix-based
(the prefix is everything before the user query) AND context-based (the
prefix plus the user query), per the CLAUDE.md Critical Rules "Prefix mapping
AND context mapping" bullet — or names the explicit stated deviation. A
one-arm round with no stated deviation is the #958 class; #779's 2026-07-14
inline pre-image round shipped context-only and the user had to catch the
missing prefix arm (a full extra inline round).
(2) **Figure sanity before presentation/commit.** Before PRESENTING (chat,
report, body) or COMMITTING any figure the round rendered, Read the rendered
PNG and confirm non-empty axes + plotted series and sane value ranges. An
empty/blank render is a round bug — fix it before showing anything; never
present or commit it. The interpretation-critic's Lens 6 PNG-load check does
not run on inline rounds (#1112: an empty figure was presented 3× while the
extraction bug was found).
Non-mapping rounds with no figures state nothing — each duty fires only on
its trigger; routing, auto-continue behavior, and the marker schema are
unchanged.

**Inline estimator-validity + record-integrity duties (REQUIRED — same rationale: this carve-out skips the planner+critic stack, where the fit-well-posedness / estimator-parity / promoted-body-consistency reviews live):** (1) BEFORE any ridge / linear-map / probe FIT, the dispatch note states `n_train` vs the feature dimension `d`; when `n_train < d` the round REFUSES the fit unless the note explicitly justifies a deliberately under-determined regime (regularization-limit / null-space read / smoke shape) — every held-out R² in the `n_train < d` regime is estimator-degenerate, not a signal read (#1701, sess `dffde9b6`: n=1,877 vs d=3,584 → ceiling 0.099 vs published 0.625). GCV-specific ban (#1887): pure-GCV λ selection at n_train < d is REFUSED (the shared #825 fit cores enforce this by default — GCV runs only WITH a dof cap, default 0.9, or under an explicit LEGACY_UNGUARDED_GCV opt-in), and selected-λ diagnostics (per-fit selector + selected λ) are reported alongside every ridge read. (2) BEFORE launching any re-implemented estimator whose in-repo reference the round can name (a `scripts/issue1345_operator_comparison`-style chain, a canonical `ridge_fit_predict_fast`, a shipped judge/scorer), the dispatch note records the DIFF between the new estimator and the named reference (function + file) — permissiveness-broadening (more inputs absorbed, weaker constraints) is called out explicitly. (3) When a round REFUTES a claim in ANY task's promoted body (its own parent or a sibling), it MUST — in the SAME turn as the result summary — either apply a NON-Takeaway PROSE correction directly to the refuted task's body via `task.py set-body` (typo / caption / fixed numeric value — never `task.py promote` or a `classification` flip; the user-only classification contract is unchanged) OR file a `kind: infra` task via `scripts/file_infra_task.py` naming the refuted issue and the refuting evidence — filing is the presumption for anything touching a bolded Takeaway; a chat-only "I did not fix X" is an INCOMPLETE round (#825's promoted Takeaway was refuted and nothing filed; #1701 origin).

**Instrument-supersession + scope-extension addenda duties (REQUIRED — same rationale: this carve-out skips the planner+critic stack, where instrument-fitness review and plan-revision re-review live):**
(1) BEFORE dispatching any stage that spends on a measurement instrument (an LLM-judge rubric, a labeling scheme, a scorer) — and AGAIN the moment such knowledge lands mid-round — the round checks (a bounded check: session knowledge plus a quick task-title scan, never an unbounded fleet-wide search) whether a SUPERSEDING instrument for the same measurement is in flight (a filed / in-progress task building a stronger replacement — the #1773 shape) or the current instrument is known-weak with a named replacement being designed; if so the DEFAULT is to HOLD the spend-bearing stages (Batch-API judge calls, GPU evals) until the superseding instrument lands — recorded as an `epm:progress` hold note naming the superseding task — and proceeding anyway requires the dispatch note to state why the known-weak instrument still serves (needed now / results not superseded / trivially cheap), never leaving the freeze to user vigilance (2026-07-28: three live SAE rounds kept burning Batch-API judge spend on labels #1773 was designed to supersede; frozen only after the user asked twice).
(2) A mid-round SCOPE-EXTENSION ADDENDUM — a user ask or self-initiated extension adding cells / draws / rows / behaviors / stages to a live inline round — is a DISPATCH for duty purposes (the scope-extension sibling of the compute-character block's "realized implementation later adds a fit/battery" drift sentence): it carries its own compute-character pre-launch statement (ops arithmetic, named batched helper, parallelization width) plus whichever other duty blocks its content triggers (both-arms mapping, figure-sanity, estimator-validity), posted BEFORE the addendum launches (2026-07-28: "parallel + vectorized" had to be re-stated twice before a throughput addendum landed — the statement bound only the original dispatch).

**Pod-safety pre-launch signals (deviation case — a pod on a
parked/terminal parent).** This step and its user-chat sibling (the
CLAUDE.md § Routing "User-chat inline free analysis" carve-out) are
ANALYSIS-ONLY and normally touch no pod (a needs-gpu discovery takes the
ABORT path below — EXCEPT the user-chat sibling under its
explicit user inline-override clause, whose deliberate GPU run inherits
these same pre-launch signals + the compute-character statement); 9a-ter proper fires at status `interpreting`, outside
the watcher's auto-stop set, but the user-chat sibling executes on PARKED
(`on_hold`) / terminal-status parents. If an inline run following this
shape nonetheless provisions or reuses a pod on such a parent, the
ORCHESTRATOR (never the subagent) MUST run `task.py add-tag <N>
keep-running` BEFORE/AT provision — before any pod work; the
timestamp-independent tag is what shields the provision/bootstrap window,
which the watcher's ≥2-miss accumulation (~20 min) would otherwise
auto-stop straight through — AND post `epm:run-launched` on the task
immediately once the pod exists (naming the pod; in any case before
launch): the watcher's pod-safety pass
(`scripts/autonomous_session_watch.py`) auto-stops a RUNNING pod on a
parked/terminal task unless a follow-up signal marker (its predicate reads
`epm:run-launched` / `epm:followup-scope` /
`epm:free-analysis-followup-run` — a descriptive list, NOT a menu: the
inline path still never posts `epm:followup-scope`) is NEWER than the
latest done-transition (`epm:promoted` / `epm:status-changed`) or the
`keep-running` tag is present. BOTH duties bind — the marker cannot exist
during the provision/bootstrap window (#573), any later done-transition flips the
inferred predicate off by design (the watcher's re-arm semantics), and the
`epm:free-analysis-followup-run v1` COMPLETION marker posts too late to
shield the launch (#477/#573/#779). Remove the tag
(`task.py remove-tag <N> keep-running`) when the run
completes so the auto-stop re-arms (a crashed run leaves the tag and the
pod bills until manual removal — check `pod.py audit-stale` output).
**Per-pod shield on multi-round issues (#1961):** the watcher shields a
SUFFIXED pod (`pod-<N>-<slug>`) PER-POD when its `epm:run-launched` note
names it in STRUCTURED position — LEAD the note with the pod name, or carry
a `pod=<name>` token (load-bearing, not stylistic: a sibling round's
`epm:status-changed` otherwise strips the issue-grain inferred shield) —
ceiling-bounded (default 48h, `EPM_POD_NAMED_SHIELD_MAX_AGE_H`);
`keep-running` stays the explicit override.
**Completion-side teardown (no ask-gate):** in that SAME completion
step — run complete + uploads verified (THIS round's artifacts, not a
prior round's PASS) — TERMINATE the pod the round provisioned (surgical
`pod.py terminate --issue <N> --name-suffix <slug> --yes` for a suffixed
`pod-<N>-<slug>`; the bare form only when the round's pod is the issue's
ONLY live pod, with the `keep-running` tag removed FIRST — #1485 refuses
the bare form while the tag is set, and it destroys EVERY live pod
resolving to the issue): verified-done teardown is unconditional,
never a user ask (the Step-8 primary-pod precedent; #1662: a
verified-done pod idled behind an ask-gate), EXCEPT when a NAMED next queued
round reuses this pod — record it in the completion `epm:progress` note
and keep the tag; a pending user question about a possible next round is
NOT a named round. Never terminate before uploads verify, and never
substitute `pod.py stop` (a STOPPED volume is NOT durable, #1112).
The sanctioned verify-then-terminate recipe for this step: verify THIS
round's artifacts → post `epm:upload-verification` with a note LEADING
`Verdict: PASS — inline-round verification; prefixes: <every verified
prefix>` via `task.py post-marker` → run the terminate; a bare
`--skip-upload-verify` without a recorded verify is the anti-pattern,
reserved for never-ran pods (the terminate guard,
`pod_lifecycle._guard_upload_verification_before_terminate`, accepts
the marker — the front door already exists, #465/#1773). And the
round's per-issue upload-verify script MUST enumerate ALL HF prefixes
the run wrote (reconcile against the run's staging/upload call sites),
never only the current phase's prefix (#1773).

**Auto-run procedure.** For the single highest-priority unran entry
(the first one in the analyzer's surfaced order; tie-break to the one
the analyzer flagged `headline_affecting: yes` — still a useful priority
signal even though it is no longer an eligibility gate — with the most
explicit eval-data path):

1. **Dispatch breadcrumb** (Step 9 entry guard convention):
   ```bash
   uv run python scripts/task.py post-marker <N> epm:progress \
     --note "stage-dispatch stage=free-analysis-followup round=1 subagent=experiment-implementer worktree=<abs path or 'repo-root'>"
   ```
   When the follow-up runs any fit/battery, this breadcrumb (or an
   immediately-following `epm:progress` note) carries the
   § Compute-character pre-launch statement above. When it computes a
   representation mapping, the same note ALSO carries the both-arms line
   (§ Inline measurement-design + figure-sanity duties above). Every 9a-ter dispatch
   breadcrumb ALSO carries the `external-markers triaged:` line (Step 9
   entry guard § Pre-dispatch external-marker triage) — the free-analysis
   run is a VM-side compute phase.
2. **Spawn `experiment-implementer`** (paired with `code-reviewer` on
   the resulting diff — same ensemble shape as Step 5). The prompt
   names the exact follow-up + cites the eval-data path(s) it must
   re-read + states the hard constraint that the diff is
   ANALYSIS-ONLY: NO new training script, NO new eval generation, NO
   pod call, NO new prompts to a base model, NO new data file
   downloaded from outside the existing `eval_results/` / HF data
   repo paths the analyzer named. When the brief delegates the
   round's landing commit to the worker itself (the worker will
   `git add`/`git commit` repo-root payload), it ALSO inlines the
   § Inline payload lint gate worker-brief composition duty below —
   the certification recipe + the guard-block = report-now contract
   (#1673). If the implementer (or
   `code-reviewer` on its diff) determines the change CANNOT be done
   without new data collection — **ABORT** the auto-run: post
   `epm:free-analysis-followup-run v1` with
   `changed_headline: false`, `gpu_hours: 0`,
   `note: aborted — reclassified as needs-gpu after implementer
   investigation; follow-up remains listed in body for manual
   triage`, and proceed to 9a-bis. The follow-up survives in the
   body as a regular bullet (now correctly understood as
   `cost_class: needs-gpu`) so a future human / autonomous pass can
   pick it up via the GPU-backed Step 9b routing (same-issue loop /
   child filing).
3. **Re-run the analysis** the implementer's diff exposes — typically
   a script in `scripts/issue<N>_*.py` or a helper under
   `src/explore_persona_space/analysis/` — over the existing eval
   JSONs. Regenerate any affected figures (the analyzer's
   `figures/issue_<N>/` outputs); Read each regenerated PNG and confirm
   non-empty axes + plotted series
   (§ Inline measurement-design + figure-sanity duties above) BEFORE
   presenting or committing it; then commit (pathspec-limited —
   `git commit -m <msg> -- <paths>`; a bare repo-root commit sweeps a
   concurrent session's staged files, #1894 / CLAUDE.md § Concurrent
   repo-root committers) + push to `main` so the body can SHA-pin them
   per the existing analyzer.md Step 3 rule. Push BARE:
   `git push origin main || uv run python scripts/sync_repo_root.py` —
   never piped (Step 10d § "Bare push / merge snippets"; sync_repo_root
   exit 0 can mean in-flight — landing not guaranteed, see the canonical
   block's caveat).

   **Staged-index verification (#1572 — after EVERY explicit-path
   `git add` of an artifact DIRECTORY, before the commit).** A
   directory-path `git add` silently skips gitignore-matched files
   inside the dir (rc=0, no error; only an explicit FILE-path add of an
   ignored file fails loud, rc=1) — #958 round 7 shipped its commit
   without the round's convention-committed `percell/*.npz` cells (the
   repo-wide `.gitignore` `*.npz` rule). After the add:

   ```bash
   git ls-files --others --ignored --exclude-standard -- <round artifact dirs>
   # any output = files an ignore rule silently skipped (git check-ignore -v
   # <file> names the rule). Per file:
   #   convention-committed round artifact (small (≲1 MB/file),
   #   plan/parent-convention-named, e.g. percell/*.npz cells)
   #     -> git add -f <file>, re-run (must be empty);
   #   large binary tensor -> HF data repo per the Upload Policy, never git;
   #   anything else -> leave unstaged, name its disposition in the completion note.
   git diff --cached --name-only -- <round artifact dirs>   # staged set == intended files
   ```

   Same class as uploader.md § Post-add reconciliation / upload-verifier
   Step 2.9 (#537) — this block is the inline-round copy (those agents
   are not in the inline path).

   **Inline payload lint gate (§ Inline payload lint gate — the cert must
   exist BEFORE the `git commit` that carries any non-artifact payload:
   `guard_root_code_commit.sh` validates it at COMMIT time, not push time
   (#1460). Preferred ordering: kick the gate off as a background Bash as
   soon as the round's scripts stop changing — before figure/body work —
   so the cert is ready by commit time; repeated block events came from
   reaching the commit first).** PAYLOAD = the round's
   to-be-committed paths outside the artifact-only set (`tasks/`,
   `figures/`, `eval_results/`, `ood_eval_results/`, `raw/`, `data/`,
   `docs/methodology/`) — typically the new `scripts/issue<N>_*.py`
   script or an `src/.../analysis/` helper. Empty payload ⇒ skip.
   Otherwise run BOTH legs as ONE background Bash (the no-flags leg is
   ~2.5-6 min; never a ≤600 s foreground bound — #991/#996), verdict
   read from the file before the push.

   **Single-flight probe (#1606)** first, per the Step 9c 1b
   single-flight statement: probe
   `uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --pattern 'issue-<N>-[^ ]*inline-payload\.txt'`
   (self-/ancestor-excluding — exit 0 = clear, 3 = live foreign match; a
   separate FOREGROUND call stays preferred as defense-in-depth, but the
   mechanical pid exclusion — not placement — is what prevents the
   launch-call self-match, #1742. The payload-file path rides the
   argv of the GATE PARENT — the helper invocation AND its enclosing
   background shell, the probe's detection surface (the map-leg
   subprocess consumes a private mkstemp copy instead, #1948) — and the
   pattern bounds the issue number on both sides (`issue-<N>-` prefix,
   `inline-payload\.txt` tail), so the probe is exact-ISSUE-scoped
   across round-unique AND transitional legacy names — a sibling
   issue's gate never matches: `issue-194-[^ ]*` cannot match
   `issue-1948-...` because the char after `194` is `8`, not `-`).
   Exit 3
   (a live foreign match) = an inline gate for THIS issue is STILL RUNNING:
   do NOT launch — round-unique payload paths (#1948) mean the `printf`
   below no longer rewrites the live run's payload file, but the
   helper's audit files
   (`/tmp/issue-<N>-inline-lint.txt` / `-inline-map.txt`) are
   unconditional ISSUE-keyed overwrites, so a relaunch clobbers the live
   run's audit legs and double-burns the ~2.5-6 min legs. WAIT for exit, or
   reap a wedged run, per the Step 9c 1b statement (crash-fix-rounds
   § Kill-before-relaunch); key any improvised wait on **process
   exit** (the probe exiting 0 — CLEAR), never on cert/audit-file
   existence (CLAUDE.md § Monitoring re-run discipline). Site nuance:
   the cert is per-content-hash and flock-guarded (#1620), so a live
   run on the SAME payload produces the cert this round needs — wait
   and read its verdict; a CHANGED payload still waits for the live
   run's exit before relaunching.

   Then the **Gate-fleet arbitration (#1962)** probe, per the Step 9c 1b
   canonical paragraph:
   `uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --fleet --exclude-issue <N>`
   — exit 3 ⇒ bounded queue (sleep 60, elapsed cap 2700 s), then launch
   anyway with the `[gate-fleet]` cap-expired line (fail-open).

   `scripts/inline_lint_gate.py` is the ONLY certifying entrypoint —
   running the component legs by hand (a manual no-flags
   `workflow_lint.py` + mapped pytest) does NOT write the
   content-hash cert `guard_root_code_commit.sh` checks, so the commit
   still blocks (an inline round ran the components
   manually and was guard-blocked at commit).

   ```bash
   # Inline payload lint gate (#1460/#1500) — ONE background Bash (run_in_background=true)
   printf '%s\n' <round's to-be-committed non-artifact paths, repo-relative> \
     > /tmp/issue-<N>-<round-slug>-inline-payload.txt   # one path per line
                                           # (the helper strips blank lines)
   uv run python scripts/inline_lint_gate.py --issue <N> \
     --payload-file /tmp/issue-<N>-<round-slug>-inline-payload.txt
   ```

   The `<round-slug>` makes the payload path ROUND-unique (e.g.
   `r<round>-<label>`, the same convention as the /tmp-hygiene note
   below). Round-unique payload paths are REQUIRED as of #1948: the gate
   REFUSES the bare legacy basename `issue-<N>-inline-payload.txt`
   (exit 3, INCONCLUSIVE) — the issue-keyed shared path is clobbered by
   concurrent same-issue rounds (#1768).

   Two invocation traps: (1) the gate's
   mapped-pytest leg routinely runs >2 min — ALWAYS pass an explicit
   Bash `timeout` ≥ 300000 ms on the gate call (the 2-min default killed
   gate runs with exit 143 in two sessions; one had chained further
   commands after the gate in the same call, so they never ran). (2)
   NEVER bundle the `git commit` into the SAME Bash call as the gate
   run — `guard_root_code_commit.sh` evaluates the compound argv BEFORE
   anything executes, so the cert cannot exist yet and the commit is
   blocked every time: certify in one call, commit in
   the next.

   (/tmp hygiene: on a long-lived follow-up issue, ROUND-SCOPE tmp
   artifact names — `/tmp/issue-<N>-r<round>-<label>` — a bare
   `/tmp/issue-<N>-<label>` persists from prior rounds/sessions and trips
   Write-before-Read collisions.)

   The helper mechanizes both legs (no-flags `workflow_lint.py`;
   `select_step9c_tests.py --map-files` → mapped pytest at the #1046
   timeout formula) and the verdict semantics below, persists the leg
   output to `/tmp/issue-<N>-inline-lint.txt` +
   `/tmp/issue-<N>-inline-map.txt` (audit parity with the pre-#1500
   fenced recipe), and on a passing path writes a content-hash-bound
   certification line (`v1 <epoch> <blobsha> <path>`) to
   `/tmp/eps-inline-lint-cert-v1.txt` — the cert the
   `guard_root_code_commit.sh` hook validates. NEVER hand-write the
   cert file (#1082 parity).

   **Verdict — payload-attributed with instrument-ran completeness,
   NEVER a bare exit-0 (main can be pre-existing-red) and NEVER a push
   on a dead instrument. The helper's exit code IS the verdict:**
   - **exit 3 = INCONCLUSIVE** (`inline_lint_gate: INCONCLUSIVE
     (<reason>)`; no cert written) ⇒ the instrument did not run to
     completion — lint-leg death / `workflow_lint: schema FAIL`
     early-exit (deliberately rejected: it prints BEFORE any check
     executes), mapped-pytest-leg death, a missing/empty payload file,
     or a payload path edited DURING the gate run — **NEVER push in
     this state**; re-run the failed leg (foreground single-flag /
     single-test re-runs are ~20-40s) or investigate, then re-run the
     helper. A clean read is honored ONLY with completeness evidence:
     the lint leg's healthy terminal line (`workflow_lint: PASS` or
     `workflow_lint: FAIL (`) present, AND — when the test mapping is
     non-empty — a pytest summary line present.
   - **exit 1 = BLOCK** (`inline_lint_gate: BLOCK (<paths>)`): a
     non-WARN output line names a payload path that is (i) NEW this
     round (absent from `origin/main` — payload-caused by construction;
     both #1388 offenders and both #1092 offenders were this case), or
     (ii) a MODIFIED file whose flagged construct sits in the round's
     own added lines (`git diff -U0 origin/main -- <path>`), or (iii) a
     MODIFIED file with a payload-naming hit carrying no parseable
     `<path>:<lineno>:` (conservative — see the enforcement note
     below). Fix, re-run just the relevant single `--check-<x>` flag or
     single mapped test (~20-40s measured), then re-run the helper and
     push. Clean sibling paths are still certified (per-path certs).
   - **exit 0 = PASS** (`inline_lint_gate: PASS`): hits naming only
     non-payload paths, WARN lines, and modified-file hits whose
     flagged construct is absent from the round's added lines are
     **pre-existing red — never block**; the helper reports them — name
     them in the round's `epm:progress` completion note (visible, not
     re-buried). Every payload path is certified; push.

   The gate is mechanically enforced for CODE payload:
   `guard_root_code_commit.sh` (PreToolUse) refuses a repo-root commit
   of uncertified `scripts/`/`src/`/`tests/` payload until
   `scripts/inline_lint_gate.py` has certified the exact landing
   content (#1500). The hook covers ONLY that code glob and ONLY
   Bash-tool root commits — the prose gate here still binds for every
   other payload shape (non-code payload such as rules/docs edits with
   lint surface, and the scratch-worktree `git push origin HEAD:main`
   merge channel). The helper is deliberately STRICTER than the prose
   in one arm: a payload-naming hit without a parseable
   `<path>:<lineno>:` on a MODIFIED file blocks conservatively (the
   prose's "pre-existing red never blocks" judgment call routes through
   the override instead). Deliberate override:
   `EPM_ALLOW_ROOT_CODE_COMMIT=1` + an `epm:progress` note naming the
   reason.

   **Worker-brief composition duty (#1673) — arm the worker BEFORE it
   meets the hook.** A brief that directs a worker/teammate subagent
   to produce AND itself commit repo-root code payload
   (`scripts/`/`src/`/`tests/` — the hook's glob) MUST inline this
   section's certification recipe: the single-flight probe, the
   fenced two-command block above (payload-file `printf` +
   `scripts/inline_lint_gate.py`, ONE background Bash — the no-flags
   leg is ~2.5-6 min and never fits a foreground bound), and the
   Preferred-ordering instruction to kick it off at script-freeze.
   The brief ALSO states the worker-side contract: **a guard-blocked
   commit is a report-now event, never a wait state** — on a
   `guard_root_code_commit.sh` refusal the worker EITHER runs the
   certification recipe itself (single-flight probe first; the cert
   is per-content-hash + flock-guarded, #1620, so a live run on the
   SAME payload is waited on and its verdict read) and retries the
   commit once the cert lands, OR immediately reports the block plus
   its staged-file state (`git diff --cached --name-only`) on the
   teammate channel (SendMessage to the orchestrator, or its final
   Agent-result return) so the orchestrator lands it; idling on the
   block is the banned outcome (#1092). Briefs composed OUTSIDE this step that delegate a
   repo-root landing commit (the user-chat inline carve-out, ad-hoc
   teammate fan-outs) carry the same duty — this block is the
   canonical text they inline.

   This gate binds the user-chat sibling (the CLAUDE.md § Routing
   "User-chat inline free analysis" carve-out) identically —
   direct-to-main is the channel, not the entry point (#1388; the
   worktree channels are already gated at Step 10d).

   A worktree-cwd inline-gate false block naming a ratchet/grandfather
   cap or failing to import `workflow_lint` is the stale-family class
   (#1417): run the Step 5a sync (now family-inclusive) in that
   worktree and re-run the gate before treating the block as
   payload-caused.
4. **Capture the headline before / after.** Read the current `body.md`
   H1 title before the re-spawn and the analyzer-produced H1 after,
   plus the LOW / MODERATE / HIGH confidence tag in each.
5. **Re-spawn `analyzer`** (fresh context) with the new analysis
   output + the prior body. The analyzer folds the new result into
   the existing clean-result body (typically updating one
   `### <finding>` H3 and possibly the H1 title / confidence tag),
   re-runs `verify_task_body.py` (must still PASS), and writes the
   revised body via `task.py set-body <N> --file ...`, followed by
   `task.py set-title <N> "<new H1 text>"` whenever the fold changed
   the H1 (set-body preserves the old frontmatter `title`; the
   H1==frontmatter verifier check FAILs the 9a-bis re-gate otherwise).
   The analyzer's
   Step 6.5 still fires on this re-run, but the loop guard above
   prevents another 9a-ter dispatch within the same task. (**`paper:
   true`?** The re-spawned analyzer re-authors the `.tex` in place —
   re-writing the Abstract + the affected Results subsection, re-running
   `build_paper.py` → `verify_paper.py`, re-writing the paper-stub — per
   `.claude/rules/analyzer-paper-mode.md` § "Same-issue follow-up rounds
   (paper-task)"; the mechanical gate is `verify_paper.py`, not
   `verify_task_body.py`. The same applies to the Step 9b cheap-band /
   same-issue follow-up loop folds.)
6. **Post the marker:**
   ```bash
   uv run python scripts/task.py post-marker <N> epm:free-analysis-followup-run \
     --note "followup_ref=<verbatim follow-up title> \
       headline_before=<H1 title before> \
       headline_after=<H1 title after> \
       confidence_before=<LOW|MODERATE|HIGH> \
       confidence_after=<LOW|MODERATE|HIGH> \
       gpu_hours=0 \
       changed_headline=<true|false>"
   ```
   Immediately after this marker posts (run or ABORT alike), fire the
   § Loop guard Cap-park surfacing notes for every remaining unrun
   eligible entry in the detection union — the cap is consumed as of
   this marker, so those entries are parked NOW, not at some future
   re-entry.
7. Proceed to **9a-bis (clean-result-critique loop)** on the UPDATED
   body. The critic gates the final state, not the pre-rerun draft.

<!-- example: anti-pattern -->
**No new gate.** This step never raises `AskUserQuestion` (in either
interactive or autonomous (`EPM_AUTONOMOUS_SESSION=1`) sessions —
auto-resolve mode is the default for both, never gate-allowed).
<!-- autonomous-mode: auto-resolve -->
If the
implementer or code-reviewer fails outright (`epm:code-review FAIL`
that survives the procedural-only strip on the first attempt), treat
it as the ABORT path from procedure step 2 — post the marker with
`note: aborted — implementer FAIL on attempt 1`, leave the follow-up
in the body as a regular bullet, and proceed to 9a-bis. The
clean-result-critique gate then runs on the analyzer's original
body; the user can pick the follow-up up post-promotion.

**Then proceed to 9a-bis (clean-result-critique loop).**

**9a-bis. Clean-result-critique loop** (only if status is `interpreting`,
after Step 9a PASS)

Same shape as the interpretation-critic loop, but the critic checks
STRUCTURE + REGISTER not CONTENT. Content honesty was settled in 9a;
this layer ensures the body matches the CURRENT v4 clean-result shape
(per `.claude/skills/clean-results/SPEC.md`): four FLAT H2s in order
(`## Takeaways` / `## Goal` / `## Methodology` / `## Results`) plus the
bold `**Repro:**` / `**Context:**` footer (NOT an H2), `## Takeaways` a
3-6-bullet numbers-first skim, `## Goal` carrying `**This experiment in
context:**` / `**Broader narrative:**`, `## Methodology` carrying
`**Design:**` / `**Training:**` (complete hyperparameter table) /
`**Evaluation:**` / `**Data extraction:**` / `**Sample
training/evaluation data + completions:**`, one `### <result>` H3 per
result under `## Results` (strict three-beat: what-is-plotted-EXACTLY →
plot → interpretation, one inline figure each), and confidence in the H1
title tag only (v4 bodies bear the `<!-- clean-result-v4 -->` sentinel;
a stray retired H2 — `## What I ran` / `## Findings` / `## Data` /
`## Reproducibility` / `## Human TL;DR` / `## TL;DR` — is a hard FAIL).
The body reads in the right register — plain academic, bullets-first,
numbers bolded. GRANDFATHERED in-flight v3 (`<!-- clean-result-v3 -->`)
and v2/legacy (no sentinel) bodies keep their prior shape — v3: five
flat H2s incl. `## What I ran` / `## Findings` / `## Data`; v2: the
2-content-section nested-TL;DR shape — and are NOT newly hard-FAILed by
a v4 rule. Discipline rules: see
`.claude/skills/clean-results/SPEC.md` (canonical structure, register,
exemplars, figure captions, and research-communication principles).

**Paper-mode branch (`paper: true` frontmatter).** When the task carries
`paper: true`, the clean-result is a LaTeX **paper** at
`docs/papers/issue_<N>/`, NOT a markdown body. Both critics branch on
`paper:` internally (`.claude/agents/clean-result-critic.md` § Paper-task
review; `.claude/agents/interpretation-critic.md` § Branch on `paper:`):
the **mechanical pre-pass is `scripts/verify_paper.py`** (NOT
`verify_task_body.py` / `audit_clean_results_body_discipline.py`, which
stay the markdown verifiers), each critic reads the `.tex` + the figure
PNGs + the compiled PDF under `docs/papers/issue_<N>/`, and the SEVEN paper
lenses (P1 self-standing Introduction · P2 self-contained Methods +
Rule-A reuse-chain depth · P3 inline-subset + comprehensive-Appendix
completeness · P4 no confidence in the paper body · P5 research-paper
register · P6 `\epsref{N}` correctness · P7 verbatim examples + judge
prompts) bind INSTEAD of the fifteen markdown lenses (no `\metric` grounding
lens in v1; `verify_paper.py` checks 7-9 mechanically gate the verbatim
training/eval/output examples, the judge-prompts appendix, and the
example-provenance pointers — the no-invention floor). The orchestrator's
brief for both critics names the paper dir (`docs/papers/issue_<N>/`) and
the `.tex`/PDF read targets instead of the markdown `body.md`; the
ensemble decision rule, the round cap, and the reconciler tie-break are
unchanged. The procedural-only verdict strip below operates on the
`verify_paper.py` presentation set for a paper-task. Everything else in
this step (round loop, PASS → `reviewing`, the 9a-quater hand-off) is
identical.

**Round 1:**

Worktree-cwd sessions run the Step 5a spec-freshness check before
dispatching this round's critics.

1. Spawn `clean-result-critic` agent (fresh context, does NOT see
   analyzer reasoning). The critic reads the published body + the
   latest `epm:interpretation v<n>` event, runs
   `scripts/verify_task_body.py` +
   `scripts/audit_clean_results_body_discipline.py` as authoritative
   mechanical passes, and scores against the v3 lens set (per
   `.claude/agents/clean-result-critic.md`) — including the
   statistical-framing rule, planned-vs-actual coverage, the
   binding-concerns audit, the contaminated/failed-data-gate-arm check,
   and the v3 Takeaways / Conciseness / Data lenses. Posts
   `epm:clean-result-critique v1` on the source task with PASS or REVISE.
   (**`paper: true`?** The critic branches internally to its § Paper-task
   review: the mechanical pre-pass is `scripts/verify_paper.py` over
   `docs/papers/issue_<N>/`, NOT `verify_task_body.py` /
   `audit_clean_results_body_discipline.py`, and the seven P1-P7 paper
   lenses bind (incl. P7 verbatim examples + judge prompts + example
   provenance / no-invention; verify_paper.py checks 7-9 gate them) — see the
   Paper-mode branch paragraph above. The Check-21
   methodology-doc pass-through below is markdown-only; skip it.)

   **Check-21 methodology-doc pass-through.** When the methodology doc
   exists on the issue worktree branch (the early-spawned
   `methodology-writer` committed `docs/methodology/issue_<N>.md` at
   Step 8's results-landed spawn — see § Split schedule below), pass its
   ABSOLUTE worktree path to BOTH the verifier and the critic so check 21
   (body Parameters table ⊆ methodology doc §2 complete table) + the
   critic's Data lens can spot-check the table against ground truth:
   ```bash
   DOC_PATH="$WORKTREE/docs/methodology/issue_<N>.md"
   uv run python "$REPO_ROOT"/scripts/verify_task_body.py --issue <N> \
     ${DOC_PATH:+--methodology-doc "$DOC_PATH"}
   ```
   The doc lives on the worktree branch and only reaches the repo-root
   `main` checkout at the Step 9b auto-merge (AFTER this gate), so a
   naive main-checkout resolve would miss it — pass the worktree path
   explicitly. Check 21 NO-OP-PASSes when `--methodology-doc` is omitted
   or the doc does not yet exist anywhere (e.g. the methodology-writer
   has not returned), and binds fully at promote-time verify (post-merge,
   `kind: experiment` only). The critic brief carries the same
   `methodology_doc_path` field for the Data-lens spot-check.

2. Spawn `codex-clean-result-critic` (Codex twin) in parallel on
   every round (all-rounds ensemble). Quota-sentinel pre-check first (#1204, CLAUDE.md
   § Codex ensemble review): when LIVE, skip this twin's composer
   spawn — instant confirmed Codex no-show per this step's no-show
   handling (Claude-only ensemble decision) + one `epm:progress`
   note. Brief contract (matches
   `.claude/agents/codex-clean-result-critic.md` § "Your brief
   contains" + Step 1b): pass the ABSOLUTE
   `$(task.py find <N>)/body.md` as `clean_result_body_path` and
   `$(task.py find <N>)/plans/plan.md` as `plan_path` — never a
   hand-built relative `tasks/<status>/<N>/...` (the status guess goes
   stale mid-flight and a relative path inherits the Codex dispatch
   cwd — the #489/#550 unresolvable-path false-FAIL class); extract
   the latest `epm:interpretation v<n>` note to a temp file
   (`/tmp/issue-<N>-interpretation-v<n>.md`) and pass that absolute
   path as `interpretation_marker_path` (never an `events.jsonl`
   path); pass the ABSOLUTE issue-worktree
   `docs/methodology/issue_<N>.md` path as `methodology_doc_path` when
   the doc exists (so the composer's compose-time `verify_task_body.py`
   run gets `--methodology-doc` and the Data lens can spot-check
   check 21 — the composer runs the mechanical verifiers on the VM and
   inlines their output into the Codex prompt as envelopes; this twin
   is dispatched read-only and uv cannot reliably execute in its
   sandbox (#1050); omit
   when the methodology-writer has not yet returned — check 21 NO-OP
   PASSes); and dispatch `codex_task.py` for this twin from the repo
   root, never an issue-worktree cwd. Posts
   `epm:clean-result-critique-codex v1`. Apply the
   ensemble decision rule (same shape as Step 5c — PASS+PASS, REVISE
   union, reconciler on disagreement; on any Agent-tool error, the Step
   5b durable-verdict-first rule applies — check
   `epm:clean-result-critique[-codex] v<n>` + the round-fresh Codex
   output file before any no-show fallback or re-spawn), BUT first run
   the procedural-only strip below.

   **Procedural-only verdict strip (clean-result analogue of Step
   5c-bis).** Before applying the ensemble rule, parse each critic's
   `Blocker tags:` line. A verdict is *procedural-only* when its tags
   are empty/`none` after removing `procedural` (presentation-only
   verifier FAILs: MDX prose, caption shape, cherry-label phrasing,
   sentinel scrub, URL-form) AND it carries no `structural-absence`,
   `audit`, or `lens` tag (fall back to scanning the verdict body for a
   substantive lens FAIL or audit hit if the line is absent on a legacy
   verdict). For any procedural-only non-PASS verdict the orchestrator:
   (a) does its OWN cheap re-run of `verify_task_body.py --issue <N>` on
   the canonical body and confirms the remaining FAILs are all in the
   presentation-only set; (b) applies the critic's `### Procedural
   fixes` edits to a staged candidate copy, verifies the CANDIDATE to
   PASS (`verify_task_body.py --file`, main-checkout copy), applies it
   via `task.py set-body <N> --file ...` (verify-first, #1860 — never
   leave a briefly-live FAILing body), and re-runs
   `verify_task_body.py --issue <N>` to PASS (post-apply confirm — the
   `--issue`-side coverage the `--file` candidate check cannot see:
   frontmatter-coupled checks, kind short-circuit, concerns-audit);
   (c) treats the critic's verdict as
   PASS for the ensemble rule — this is "review incomplete → fix the
   procedural item inline + re-dispatch", NOT a consumed REVISE round
   (the round counter does NOT increment). Log one chat line:
   `procedural-only clean-result FAIL stripped — orchestrator applied N
   inline fixes + re-verified PASS; no substantive findings → PASS.` If
   ANY remaining FAIL is structural-absence, or the critic carried a
   `lens`/`audit` tag, leave the verdict as-is and apply the normal
   ensemble rule (the REVISE round counts). The strip operates ONLY on
   the mechanically-verifiable presentation set; it never overrides a
   register / story-arc / statistical-framing lens judgment. A verdict
   carrying the `data-access-blocked` tag, a `Verifier: UNAVAILABLE` /
   `Audit script: UNAVAILABLE` line, or a missing-envelope declaration
   is NEVER procedural-only — the mechanical pre-pass was unavailable,
   so there is nothing verified to strip against; re-compose /
   re-dispatch the twin instead of stripping (#1050).

**If REVISE (rounds 2-5):**

Re-spawn `analyzer` agent (fresh context, sees raw data + all
interp-critique history + the latest clean-result-critique)
(trigger-dense round: critique history by reference — marker
kind+version / output-file paths, per § File-only Codex verdict posting;
never inline the critique bodies). Analyzer
revises the `epm:interpretation` event AND edits the task body in
place via `task.py set-body <N> --file ...`. Re-runs
`scripts/verify_task_body.py` (must still PASS). Re-spawn the critic
ensemble — `clean-result-critic` AND `codex-clean-result-critic`
(all-rounds ensemble), fresh contexts, against the
revised surfaces, with prior critique summaries in both briefs. Both
post the next critique version (`epm:clean-result-critique v<n>` +
`epm:clean-result-critique-codex v<n>`); apply the same ensemble
decision rule (including the procedural-only strip) as round 1. Round
boundaries here carry the Step 5c-quater round-boundary durable-decision
duty (decision note + explicit-path commit BEFORE the re-spawn).

**Max 5 rounds.** At round 5 (the cap) with a non-PASS ensemble verdict,
apply the procedural-only strip once more (procedural / presentation
REVISEs). If ALL residual REVISEs are stripped → advance. If ANY
SUBSTANTIVE residual remains — a flagged OVERCLAIM the strip cannot
resolve — SURFACE it, do NOT auto-publish into the clean-result record
(#784 surface-not-ship: a real residual here is an overclaim that must
never be silently promoted). Either way post the §5 marker first
(`uv run python scripts/post_step_completed.py --issue <N> --step 9a-bis
--exit-kind parked` interactive / `--exit-kind failure-exit` autonomous).
Interactive: present the residual to the
user + EXIT (the user decides whether to patch before promoting).
Autonomous (`EPM_AUTONOMOUS_SESSION=1`): post `epm:failure v1
failure_class: code` referencing the residual, set `status: blocked`,
fire `PushNotification`, run CRON-TEARDOWN, EXIT (halt_criteria id=6
`concern_unresolved` family).

**On PASS (or all-stripped at the cap):**

Move status to `reviewing`:

> **Same-issue follow-up round?** At `followups_running`, SKIP this
> `set-status` (status-hold rule, Step 9b § Same-issue follow-up loop step 3;
> code-enforced — `task.py` refuses the flip) — proceed straight to
> 9a-quater; the round exits the status only at the `awaiting_promotion` re-park.

```bash
uv run python scripts/task.py set-status <N> reviewing \
  --note "clean-result-critic PASS; advancing to final review gate."
```

**Then proceed to 9a-quater (methodology reference).**

**9a-quater. Methodology reference — POST-PASS EXPORT** (only if status
is `reviewing`, after the 9a-bis loop's PASS, before the
`awaiting_promotion` park below)

**Paper-mode (`paper: true`): SKIP the standalone-doc export.** For a
paper-task the methodology IS the paper — the `methodology-writer`
already authored the comprehensive Methods section + recipe Appendix
(complete hyperparameter table + worked examples + Rule-A reuse recipes)
DIRECTLY INTO the `.tex` at the Step 8 early spawn, and `verify_paper.py`
gated the paper's section completeness at 9a-bis. There is no separate
`docs/methodology/issue_<N>.md` to export and no top-of-body
`**Methodology:**` link to append (the `body.md` is a thin paper-stub
pointing at `docs/papers/issue_<N>/`). Post
`epm:methodology-doc-generated v1` with
`note: skipped — paper-task (Methods + Appendix authored into docs/papers/issue_<N>/issue_<N>.tex)`
so the idempotency check converges, then proceed to 9b. The v4 markdown
export below does NOT run for a paper-task.

**v4 markdown (current): MECHANICAL EXPORT, no agent spawn.** Every
`kind: experiment` v4 clean-result auto-gains a standalone methodology
reference at `docs/methodology/issue_<N>.md` that is a **mechanical COPY
of the body's `## Methodology` section** — the body's `## Methodology`
section IS the canonical source (the analyzer wrote it factually,
interpretation-free, per the v4 spec), so there is NO separate
findings-blind authoring step and the `methodology-writer` agent is NOT
spawned for v4. After the 9a-bis PASS the orchestrator:

1. Reads the finalized body and extracts the `## Methodology` section
   verbatim (from the `## Methodology` H2 to the next H2 / the `---`
   footer rule).
2. Writes it to `docs/methodology/issue_<N>.md` with the H2 header
   normalized to `# Methodology — issue <N>: <one-line what-was-run>`
   (plus a 1-line `*Derived from the [task body](https://eps.superkaiba.com/tasks/<N>).*`
   footer).
3. **Commits the doc to `main`** by explicit path (durable — removes the
   v3 worktree-only gap; the doc + its SHA-pinned link land on `main`
   directly). Capture the commit SHA.
4. Runs a no-secrets pre-scan (`scripts/check_no_secret_shaped_strings.py`
   / `redact_for_gist.py`) and publishes a **secret** (unlisted) gist
   mirror via `gh gist create` — or, when a prior
   `epm:methodology-doc-generated` marker on this task already recorded a
   `gist_url` (a follow-up round's re-export), UPDATES that existing gist
   via the canonical gist-update recipe (procedure step 6 below: `gh api
   -X PATCH` + API-read verify — NEVER a second `gh gist create`, and
   NEVER any `gh gist edit` form) — FAIL-SOFT either way (a missing/failed
   gist never blocks the step; the in-repo doc is the durable artifact).
5. Appends the one-line `**Methodology:**` pointer at the TOP of the body
   — immediately after the `<!-- clean-result-v4 -->` sentinel, before
   `## Takeaways` — linking the GitHub blob (SHA-pinned to the `main`
   commit) and the gist (drop the `· [gist](...)` suffix when the gist
   fail-softed). `<DOC_SHA>` is ALWAYS taken from command output —
   `DOC_SHA=$(git rev-parse HEAD)` right after the doc commit — never
   typed or hand-extended from a short SHA (the never-fabricate-SHAs
   rule; #1738).
6. Posts `epm:methodology-doc-generated v1` (`doc_path` + `commit` +
   `gist_url`).

Fires in BOTH interactive and autonomous sessions identically.
<!-- example: anti-pattern -->
Auto-continue (NOT a new `AskUserQuestion` gate); the halt-criterion
contract is preserved.
<!-- autonomous-mode: auto-resolve -->
Same behavior in interactive and autonomous sessions: no AskUserQuestion
is ever raised by this step; the marker `epm:methodology-doc-generated v1`
is the durable record consumed by re-entry idempotency.
<!-- example: anti-pattern -->

**v3/v2 GRANDFATHERED (LATE JOIN, in-flight bodies only):** an in-flight
v3/v2 body carries no detailed `## Methodology` section to copy, so the
legacy path still applies — the findings-blind `methodology-writer` agent
is EARLY-SPAWNED at Step 8's results-landed parallel spawn, the
orchestrator commits the doc on its return, and the gist + body
link-append (top-of-body `**Methodology:**` line + the
`## Reproducibility` `**Methodology reference:**` row) LATE-JOIN here.
The detailed v3/v2 procedure below (§ Split schedule + procedure steps
1-9) describes that grandfathered path; for a v4 body run the
mechanical export above instead and skip the agent spawn.

**Split schedule (early spawn ∥ interpretation loop).** This step is
split in two:

- **EARLY SPAWN (at Step 8's results-landed parallel spawn):** the
  orchestrator evaluates the kind-gating below, posts the
  `stage=methodology-reference` breadcrumb, pre-extracts the
  findings-blind Reproducibility input — from the `epm:results`
  markers' `reproducibility_card` (alias `reproducibility`) +
  `eval_paths`, merged newest-wins per field across markers (see
  procedure step 2), because the clean-result body's
  `## Reproducibility` H2 does not exist yet — and
  spawns `methodology-writer` in the background
  (`run_in_background=true`). This is safe because the agent is
  findings-blind by design: its inputs (plan, experiment config,
  reproducibility metadata, verbatim artifact rows) are all final the
  moment results land. When the agent returns — possibly while
  analyzer ↔ critic rounds are still iterating — the orchestrator
  immediately commits `docs/methodology/issue_<N>.md` on the issue
  worktree branch (procedure step 5 below).
- **LATE JOIN (here, after clean-result-critic PASS — the body must be
  final):** no-secrets pre-scan, secret-gist publish (fail-soft), the
  body link-append (the top-of-body `**Methodology:**` line + the
  `## Reproducibility` `**Methodology reference:**` row — procedure
  step 7), the verifier re-run, and the
  `epm:methodology-doc-generated v1` marker — posted only when the
  link line lands (the step is only "done" then). If the background
  agent has not returned yet at this point, WAIT for it here
  (TaskOutput / completion notification) before running the join — load
  the deferred schema first (`ToolSearch("select:TaskOutput")`): an
  unloaded direct call fails with InputValidationError.

The early spawn needs no extra gating relative to upload verification:
the agent's artifact reads are worktree-local, and the late join
already sits far after upload PASS. **Fallback (serial) path:** when
the early spawn never happened (resume of an older in-flight task, or
the early agent crashed without writing the doc), run the full
procedure below serially at this point, slicing the Reproducibility
input from the now-final body's `## Reproducibility` H2 as written in
step 2. **Early-spawn idempotency:** an in-window
`stage=methodology-reference` breadcrumb (Step 9 entry guard) or an
already-committed `docs/methodology/issue_<N>.md` on the issue branch
means the agent run is live or done — do not re-spawn it; only the
late join remains.

**When to run** (gating rules):

- `kind: experiment` → always.
- `kind: analysis` → only when the task's `## Reproducibility` section
  names a training or eval methodology (i.e. there is something to
  document). When the analysis task has no Reproducibility row beyond a
  Code SHA, the agent itself writes a 5-line "no experimental
  methodology" stub and exits; the link still lands in
  `## Reproducibility` for consistency.
- `kind: infra | batch | survey` → skip entirely. Log one chat line
  (`Step 9a-quater skipped (kind=<X>)`) and proceed to 9b.
- **Idempotency — scoped per follow-up round.** When
  `epm:methodology-doc-generated v1` is already on the task (re-entry /
  backstop tick / re-invocation after a separate 9a-bis REVISE that
  bounced back to analyzer), check follow-up coverage before no-opping:
  collect the `followup_label`s of `epm:followup-scope v1` markers
  whose round's analyzer re-fold has run (during a same-issue follow-up
  round this is exactly the current round's label; labels from rounds
  that never ran add no methodology and are ignored), and the labels
  already recorded across prior `epm:methodology-doc-generated` notes
  (`extends=` / `no-new-methodology=` fields). When every such label is
  recorded — or the task has no followup-scope markers at all — this
  step is a no-op: the doc was already written, committed, and
  gist-mirrored on a prior pass. Do NOT regenerate or re-publish. Log
  one chat line (`Step 9a-quater no-op — epm:methodology-doc-generated
  v1 already present`) and proceed to 9b. When an UNRECORDED label
  exists (same-issue follow-up re-fold), run the **EXTEND pass** below
  instead — a task-scoped no-op here would leave
  `docs/methodology/issue_<N>.md` permanently describing only the
  parent run (#543: a fifth arm folded into the
  clean-result had to be patched around with an in-body scope note).
- **EXTEND pass (same-issue follow-up rounds).** Re-run procedure
  steps 2-9 below for the unrecorded `followup_label`, with these
  deltas:
  - Step 2 uses the fallback (serial) body-slice form — during a
    follow-up round the re-folded body IS final post-critic.
  - Step 3 spawns `methodology-writer` in **EXTEND mode** (see
    `.claude/agents/methodology-writer.md` § EXTEND mode): the prompt
    names the mode, the `followup_label`, and the existing doc path;
    the agent reads the EXISTING `docs/methodology/issue_<N>.md`
    (findings-blind by construction) plus ONLY the new round's plan
    amendment + Reproducibility slice, and re-writes the doc by
    EXTENDING the six fixed sections IN PLACE — adding a per-round
    COLUMN to the canonical §2 hyperparameter table for whatever the
    round CHANGED (the check-21 reconciliation surface), labeled
    `### Round <label>` sub-blocks inside §3/§4/§5 ONLY where the
    round's recipe / probes / examples differ, and new rows on §6's
    existing artifacts table; parent sections preserved everywhere
    else. NEVER a new top-level `## ...` heading or a second
    §2-style table — a bare `## <followup_label> arm` H2 carrying only
    the boilerplate footer strands the round's recipe outside §2
    (#642). The brief inherits THIS wording, so it stays
    consistent with `.claude/agents/methodology-writer.md` § EXTEND
    mode.
  - Step 6 refreshes the EXISTING gist when a prior marker recorded a
    `gist_url` — via the canonical gist-update recipe (procedure step 6:
    `gh api -X PATCH` + API-read content verify; NEVER any `gh gist edit`
    form, which can silently no-op with rc=0 — #1769), same
    fail-soft rule; fall back to `gh gist create` only when no prior
    gist exists.
  - Step 7 UPDATES the existing lines' `<DOC_SHA>` pin in place in
    BOTH locations — the top-of-body `**Methodology:**` line and the
    `## Reproducibility` `**Methodology reference:**` row (never
    append duplicate lines; same `· [gist](...)` suffix rules; if a
    pre-top-line body carries only the Reproducibility row, ADD the
    missing top line while re-pinning the row).
  - Step 9 posts a NEW `epm:methodology-doc-generated v1` marker with
    `extends=<followup_label>` in the note (plus the refreshed
    `commit=` / `gist_url=`) — this is the record the idempotency
    check reads.
  - **No-new-methodology carve-out:** when the round was a
    planner-exempt re-run with an identical recipe (different seeds /
    monitoring / bug-fix re-run — nothing for a findings-blind doc to
    add), skip the agent spawn and post the marker with
    `no-new-methodology=<followup_label>` so idempotency converges
    without doc churn.

**Procedure** (auto-continue end to end — interactive and autonomous;
on the normal path steps 1-3 + 5 already ran at the EARLY SPAWN and
steps 4 + 6-9 are the LATE JOIN executed here):

1. **Dispatch breadcrumb** (Step 9 entry guard convention):
   ```bash
   uv run python scripts/task.py post-marker <N> epm:progress \
     --note "stage-dispatch stage=methodology-reference round=1 subagent=methodology-writer worktree=<abs path or 'repo-root'>"
   ```
2. **Pre-extract the findings-blind Reproducibility input.**
   On the normal (early-spawn) path the clean-result body does not
   exist yet, so extract the `reproducibility_card` (alias
   `reproducibility`; the canonical key wins within one payload) +
   `eval_paths` from the task's `epm:results` markers
   (`task.py view <N> --json`) into the temp file instead — NOT from
   the latest marker alone. Multi-launch runs legitimately post
   several `epm:results` markers, and a resume-pass sentinel can
   carry an empty card (#601: `adapter_paths: {}` after every cell
   `resumed_skip`) that would hand the methodology-writer nothing:
   resolve each field newest-wins from the newest card that declares
   it non-empty (empty dict/list/string/None is not a declaration; nor —
   for `adapter_paths` / `wandb_run_names` — a non-dict/non-list prose
   pointer, #1489) —
   the same semantics as `verify_uploads.py` `merged_results_card`.
   The body-slice form below is the
   fallback (serial) path, where the body IS final: slice just the
   `## Reproducibility` H2
   from the task body into a temp file and hand the agent ONLY that
   path — never the full `body.md`. Either way, this is what physically enforces
   findings-blindness: `## Takeaways` / `## Findings` / the H1 confidence
   tag (v2/legacy: `## Human TL;DR` / `## TL;DR`) never enter the agent's
   context. Prompt discipline is defense in depth on top of this
   structural cut, not the primary mechanism:
   ```bash
   BODY_PATH=$(uv run python scripts/task.py find <N>)/body.md
   REPRO_FILE=$(mktemp -t issue<N>-reproducibility.XXXXXX.md)
   awk '/^## Reproducibility[[:space:]]*$/{flag=1; print; next} \
        flag && /^## /{flag=0} flag' "$BODY_PATH" > "$REPRO_FILE"
   # Confirm the slice is non-empty; if it is, the body is malformed
   # (no `## Reproducibility` H2). Post epm:failure v1
   # (failure_class: data, reason: missing ## Reproducibility for
   # methodology-writer), set status:blocked, exit. Surface a
   # workflow-fix-candidate v1 block — the verifier should have caught
   # this upstream.
   [ -s "$REPRO_FILE" ] || { echo "Reproducibility slice empty"; exit 1; }
   ```
3. **Spawn `methodology-writer`** (fresh context, findings-blind). The
   prompt names the task number + the absolute path of the pre-extracted
   `## Reproducibility` slice (`$REPRO_FILE` from the previous step) as
   its starting input — NOT the full `body.md` path. The agent reads
   ONLY the plan, the Reproducibility slice, the training/eval scripts
   at the body's `**Code:**` SHA, the Hydra config, and a handful of
   artifact rows for verbatim worked examples. Output:
   `docs/methodology/issue_<N>.md`. See `.claude/agents/methodology-writer.md`
   for the full read/don't-read list and the "no interpretation" hard
   constraints. Delete `$REPRO_FILE` after the agent exits.
4. **No-secrets guard** (pre-publish, mandatory). Before publishing
   the gist, scan the generated doc for obvious secret patterns —
   `sk-`, `hf_`, `wandb`-key shapes, `RUNPOD`, `ANTHROPIC_API_KEY`, raw
   `.env` content — with the canonical scanner:
   `uv run python "$REPO_ROOT/scripts/check_no_secret_shaped_strings.py" "$REPO_ROOT/docs/methodology/issue_<N>.md"`
   (exit 0 = clean, exit 1 = hit). Use `$REPO_ROOT`-absolute paths — a
   bare `scripts/...` resolves against the orchestrator's cwd, which on a
   worktree (or after a removed-dir `getcwd` miss) is NOT repo root and
   resolves to `/scripts/...: No such file or directory` (#654). Do NOT
   use `redact_for_gist.py` for
   this — it has only `--in`/`--out`/`--in-place`, no `--check` flag.
   The methodology-writer reads only the
   already-public Reproducibility data + the repo, so this scan should
   never trip in normal operation; it is a safety net. On any hit,
   ABORT the gist publish, keep the committed repo doc, and pass the
   `note: gist skipped — possible secret detected` field through to
   the marker (step 9). Continue to the link-append step regardless;
   the in-repo doc remains the durable artifact.
5. **Commit the doc to the repo.** Inside the worktree branch (the
   one this `/issue <N>` is running on — never the main checkout):
   ```bash
   git -C "$WORKTREE" add docs/methodology/issue_<N>.md
   git -C "$WORKTREE" commit -m "methodology: issue #<N> findings-blind reference" -- docs/methodology/issue_<N>.md
   DOC_SHA=$(git -C "$WORKTREE" rev-parse HEAD)
   ```
   Use the explicit path; never `git add -A` (avoids sweeping
   unrelated working-tree changes), and keep the commit
   pathspec-limited so any other staged entry in the index is ignored
   (same guard as the Step 10d surgical checkout). The doc rides to
   `main` with the auto-merge at Step 9b.
6. **Publish the secret gist (fail-soft).** Try once. `gh gist create
   <file>` uses the file's basename for the gist filename — the
   in-repo path is `docs/methodology/issue_<N>.md`, so the rendered
   gist filename is `issue_<N>.md` (no extra rename needed):
   ```bash
   GIST_RAW=$(gh gist create \
     --desc "Task #<N> — Methodology, hyperparameters, and worked examples (Explore Persona Space)" \
     docs/methodology/issue_<N>.md 2>&1)
   # Extract the gist URL; on failure gh writes an error to stderr/stdout
   # instead of a URL, so grep for the URL shape rather than `tail -1`
   # (which would capture the error text as a bogus GIST_URL).
   GIST_URL=$(printf '%s\n' "$GIST_RAW" | grep -oE 'https://gist\.github\.com/[^[:space:]]+' | tail -1)
   if [ -z "$GIST_URL" ]; then gist_err=$(printf '%s\n' "$GIST_RAW" | tail -1); fi
   ```
   `gh gist create` defaults to a **secret** (unlisted) gist when the
   `--public` flag is absent (verified against `gh gist create --help`:
   *"By default, gists are secret; use `--public` to make publicly
   listed ones."*). **Fail-soft behavior** — if `gh` lacks the `gist`
   scope, is offline, or returns a non-URL on stderr/stdout, the grep
   above leaves `GIST_URL` empty and captures the error as `gist_err`;
   continue with the empty-`GIST_URL` path below. Do NOT
   block the step or the park on a missing gist; the committed repo
   doc is the durable artifact and the next step links to it either
   way.
   Keep the `[ -z "$GIST_URL" ]` capture in the if-form shown above — a
   trailing `[ -z "$GIST_URL" ] && gist_err=...` one-liner variant makes the
   CALL report Exit 1 on SUCCESS (URL present -> the test is false -> `&&`
   short-circuits with rc 1; #928). Same exit-code
   hygiene rule as Step 9c step 1b: never leave a bare conditional or
   informational grep as the last command of a call — if-form it or
   `|| true` it.

   **Canonical gist-update recipe (re-exports / follow-up rounds —
   #1769).** When a prior `epm:methodology-doc-generated` marker
   on this task already recorded a `gist_url`, UPDATE that gist instead of
   creating a second one:
   ```bash
   GIST_ID=$(basename "<prior gist_url from the latest epm:methodology-doc-generated marker>")
   DOC=docs/methodology/issue_<N>.md
   GIST_FILE=issue_<N>.md          # gh gist create used the basename
   GIST_UPDATED=no; GIST_UPDATE_ERR=""
   # Capture PATCH stderr for the failure reason (the create-side GIST_RAW /
   # gist_err pattern): 2>&1 >/dev/null routes stderr into the substitution
   # while discarding stdout.
   if PATCH_ERR=$(gh api -X PATCH "gists/$GIST_ID" \
        -F "files[$GIST_FILE][content]=@$DOC" 2>&1 >/dev/null); then
     # VERIFY by API read-back — PATCH rc=0 alone is NOT success (#1769:
     # the EDITOR-override gh gist edit form silently no-opped — rc=0 with
     # content UNCHANGED). $(...) / $(<file) strip trailing newlines on both sides,
     # so a trailing-newline-only difference (the #1769 verified-match
     # shape) reads as a match; any interior difference fails.
     REMOTE=$(gh api "gists/$GIST_ID" --jq ".files[\"$GIST_FILE\"].content" 2>/dev/null)
     LOCAL=$(<"$DOC")   # bash builtin read — never `cat` (guard_log_dump argv match on large docs)
     if [ -n "$REMOTE" ] && [ "$REMOTE" = "$LOCAL" ]; then
       GIST_UPDATED=yes
     else
       GIST_UPDATE_ERR="verify mismatch (API read-back != local doc)"
     fi
   else
     GIST_UPDATE_ERR="PATCH failed: $(printf '%s\n' "$PATCH_ERR" | tail -1)"
   fi
   ```
   BAN — exactly ONE verified update path exists: never
   `EDITOR=... gh gist edit` (it silently no-ops with rc=0, leaving a
   stale public mirror — #1769) and never the flag form
   `gh gist edit <id> --filename <name> <local>` either; ALL `gh gist
   edit` forms are banned for updates. Fail-soft: `GIST_UPDATED=no`
   never blocks the step — keep linking the existing gist URL and record
   `gist_update=failed ($GIST_UPDATE_ERR)` in the step-9
   `epm:methodology-doc-generated` marker note (key=value grammar
   matching the existing `gist_url=` / `commit=` fields; a stale mirror
   is thereby VISIBLE instead of silent, and "PATCH failed" is
   distinguishable from "verify mismatch"). The verify's read-back uses
   the gist GET `content` field — methodology docs are far below the
   gist API's ~1 MB truncation threshold; if `truncated: true` ever
   appears, fetch `raw_url` instead.
7. **Append the link lines to the clean-result body — TWO locations.**
   Use `task.py set-body <N> --file <new-body.md>` (NO
   `--snapshot` — the previous body is already the canonical
   clean-result; this is a two-line append, not a promotion).
   Read the current body and SHA-pin both blob URLs with the `DOC_SHA`
   captured in step 5 — the step-8 verifier's URL-permanence check
   FAILs any unpinned `/blob/main/` GitHub link.

   **Idempotency (same-pass re-entry):** a crashed-and-resumed late
   join can re-run this step after the body was already edited but
   before the `epm:methodology-doc-generated` marker posted (the
   marker lands only at step 9). Before inserting either line, check
   the current body for an existing `**Methodology:**` top line /
   `**Methodology reference:**` Reproducibility row; when one is
   present, UPDATE that line's `<DOC_SHA>` pin and `· [gist](...)`
   suffix in place — never append a duplicate (mirrors the
   EXTEND-pass step-7 delta above).

   Compose both edits in a staged copy (`/tmp/...`) and apply ONLY via
   `task.py set-body <N> --file ...` (named again below) — NEVER a raw
   `body.md` write (#1090: a direct pathlib write
   bypassed task.py and the revert attempt was hook-blocked).

   (a) **Top of body — the reader-facing pointer.** Insert exactly
   this line immediately AFTER the clean-result sentinel (i.e. right
   under the H1 title), BEFORE the first content H2, with a blank line on
   each side. Branch on the sentinel:
   - **v3 body** (`<!-- clean-result-v3 -->`): insert after that
     sentinel, BEFORE `## Takeaways`.
   - **In-flight v2 body** (`<!-- clean-result-v2 -->`): insert after
     that sentinel, BEFORE `## Human TL;DR`.
   - **Legacy body** (no sentinel): directly under the H1 title line.
   ```
   **Methodology:** [docs/methodology/issue_<N>.md](https://github.com/superkaiba/explore-persona-space/blob/<DOC_SHA>/docs/methodology/issue_<N>.md) · [gist](<GIST_URL>)
   ```

   (b) **`## Reproducibility` — the artifact-index row.** Locate the
   `## Reproducibility` H2, add exactly this line under the existing
   bullet list (between the `**Artifacts:**` and `**Compute:**` rows,
   or at the end of the section's bullet list if those anchors aren't
   present):
   ```
   - **Methodology reference:** [docs/methodology/issue_<N>.md](https://github.com/superkaiba/explore-persona-space/blob/<DOC_SHA>/docs/methodology/issue_<N>.md) · [gist](<GIST_URL>)
   ```

   When `GIST_URL` is empty (fail-soft path), drop the `· [gist](...)`
   suffix entirely from BOTH lines:
   ```
   **Methodology:** [docs/methodology/issue_<N>.md](https://github.com/superkaiba/explore-persona-space/blob/<DOC_SHA>/docs/methodology/issue_<N>.md)
   ```
   ```
   - **Methodology reference:** [docs/methodology/issue_<N>.md](https://github.com/superkaiba/explore-persona-space/blob/<DOC_SHA>/docs/methodology/issue_<N>.md)
   ```
   Write the revised body via `task.py set-body <N> --file ...`.
   (Body-shape spec for the top line:
   `.claude/skills/clean-results/SPEC.md` § Top-of-body methodology
   link. Forward-only: never retro-edit bodies finalized before this
   rule existed except via the EXTEND-pass re-pin above.)
8. **Re-run the mechanical verifier on the body.** The two-line link
   addition cannot break the spec (the verifier permits the top-of-body
   `**Methodology:**` line and the Reproducibility row), but the
   verifier costs ~1s and catches the unlikely off-anchor edit:
   ```bash
   uv run python "$REPO_ROOT"/scripts/verify_task_body.py --issue <N>  # main-checkout copy, never the worktree's (spec-stale risk, #496)
   ```
   Do NOT re-run the full clean-result-critic loop — this is a
   mechanical post-script edit, not a substantive body change.
   On verifier FAIL, post `epm:failure v1` with
   `failure_class: code`, `reason: methodology-link-append broke
   verify_task_body.py`, set `status:blocked`, and exit (this is a
   workflow bug — surface a `workflow-fix-candidate v1` block in the
   exit text so the orchestrator can AUTO-FILE a `kind: infra` task +
   spawn a background `/issue --auto` session per the
   workflow-fix-on-bug protocol).
9. **Post the marker:**
   ```bash
   uv run python scripts/task.py post-marker <N> epm:methodology-doc-generated \
     --note "doc_path=docs/methodology/issue_<N>.md commit=<DOC_SHA> gist_url=<GIST_URL or 'n/a — <gist_err>'>"
   ```
   When the step was skipped (kind: infra/batch/survey, or an
   analysis task with no methodology surface that the agent stubbed),
   include `note=skipped: kind: <X> has no methodology surface` (or
   the analyzer-stub equivalent) instead of a real `commit=` /
   `gist_url=`.

**Then proceed to 9b (final reviewer step — retired; flips to
`awaiting_promotion`).**

**9b. Final reviewer step — RETIRED.**

The dedicated `reviewer` / `codex-reviewer` ensemble was deprecated when
its statistical-framing responsibilities were absorbed into
`clean-result-critic` Lens 7 (see CLAUDE.md ontology table; under the v2
spec Lens 11 is "raw alongside processed"). The
`reviewing` status now exists ONLY as the single-step parking point
between clean-result-critic PASS and `awaiting_promotion`. The skill
moves through it in one transition with no agent dispatch:

```bash
uv run python scripts/task.py set-status <N> awaiting_promotion \
  --note "clean-result-critic PASS; parking for user promotion."
uv run python scripts/task.py post-marker <N> epm:status-changed \
  --note "reviewing -> awaiting_promotion (transitional; no agent dispatch at reviewing)"
```

**Run CRON-TEARDOWN now.** `awaiting_promotion` is the terminal/park
transition for an experiment: the pod was terminated at Step 8 and this is
a human gate, so there is nothing left to auto-drive. Run the two-leg
sweep (§ CRON-TEARDOWN procedure — recurring tick +
stray one-shot `/issue <N>` wakeups) so the backstop
that deliberately survived the post-`done` stages stops re-firing now. (A
later user re-invocation at `awaiting_promotion` does not re-arm — Step 6d.2
arms only for pod-backed runs reaching the polling loop.)

**Fire `PushNotification` to the phone.** The user is the only actor who
can advance an `awaiting_promotion` task (via `task.py promote <N>
useful|not-useful`), so alert them now:

```python
PushNotification({
    "message": f"#{N} {slug} · clean-result ready — open to promote"[:200],
    "status": "proactive",
})
```

Soft-fail: swallow exceptions (Remote Control disconnected, schema not
loaded). The chat-side prompt below remains the durable record.

**Auto-merge the worktree now (experiments).** The instant the task
lands at `awaiting_promotion`, run the **Step 10d auto-merge procedure**
(rebase-merge `issue-<N>` -> `main`, no prompt, keep the worktree).
Execute it with the Step 10d command blocks VERBATIM — bare,
exit-code-checked push/merge, never piped through `tail`/`grep`/`head`
(Step 10d § "Bare push / merge snippets"; the `guard_piped_git_push.sh`
hook blocks piped variants). The
code / figures / `eval_results` the run produced land on `main`
immediately so the next experiment inheriting from `main` gets any
shared-infra fix this branch carried (this is the #456 -> #466 fix). The
science verdict (`useful` / `not-useful`) is orthogonal and still parks
below for the user. Merging does NOT block the park: an auto-merge
conflict posts `epm:merge-failed v1` and surfaces one line in chat, but
the task still parks at `awaiting_promotion` for promotion. Idempotent —
skip if `epm:merged` already exists.

**Cheap follow-up auto-run (BOTH interactive and autonomous — fires
here, after auto-merge, before the autonomous-only block below).**
Standing directive: *a follow-up that is `0` GPU-h or
`< 20` GPU-h just runs and folds into the same issue, automatically, in
either session mode.* The 0-GPU floor is handled inline at Step 9a-ter
(free-analysis); this block handles the GPU-backed cheap band
(`0 < est_gpu_hours < 20`). It applies to `question_relation: same`
proposals ONLY — a `substantially-different` follow-up changes the
parent `## Goal`, so by the project's routing law it cannot fold into
this issue and is NEVER auto-run here regardless of GPU cost (it stays
filed as a `proposed` child via the autonomous-only block below, or
surfaces at interactive Step 10b for manual triage).

**Follow-up value-critique (redundancy screen) — MANDATORY before ANY
proposal routes.** The instant an `epm:follow-ups v1` marker exists for
this park (posted by C0 below, the autonomous-only block, or interactive
Step 10b), and BEFORE any proposal is routed to the cheap-band auto-run,
the autonomous same-issue loop, the autonomous child-filing path, or the
interactive pick, the orchestrator runs the **follow-up value-critique
ensemble** ONCE over the whole proposal set. This is the 5th doubled
review site (workflow.yaml § ensemble_review.doubled_steps[follow-up-critic],
`single_pass: true`) — it screens for REDUNDANCY only (NOT info-gain /
worth) and NOTHING is dropped: every proposal is saved with a rationale
either way. The subroutine (call it **VC** — invoked from C0a below, the
autonomous block step 2-bis, and Step 10b):

> **VC. Run the value-critique ensemble (single pass — no revise loop).**
> 1. **Idempotency.** If an `epm:followup-value-critique v1` marker
>    already exists for THIS proposal set (match by the `epm:follow-ups
>    v1` it screened — same park), SKIP — reuse the existing merged
>    verdict (this is a no-op on a backstop-tick / re-entry). Otherwise:
> 2. **Spawn the ensemble** in ONE message (two `Agent` calls, staggered
>    a few seconds per the CLAUDE.md 429 guidance): the Claude
>    `follow-up-critic` AND the `codex-follow-up-critic` prompt-composer.
>    Write the `epm:follow-ups v1` body to a temp file and pass its PATH
>    as `proposals_marker_path` (never inline the proposals), plus
>    `experiment_number`, `parent_goal` (the task `## Goal`), and any
>    `prior_value_critique_summaries`. Dispatch the Codex twin's composed
>    prompt as bg Bash via `scripts/codex_task.py` exactly like the other
>    four twin sites (CLAUDE.md § "Codex ensemble review"); the twin agent
>    NEVER dispatches Codex itself (orphan-job anti-pattern, #533). Post
>    `epm:followup-value-critique v1` (Claude) + `epm:followup-value-critique-codex`
>    (Codex) on this task's `events.jsonl`. Quota-sentinel pre-check
>    first (#1204, CLAUDE.md § Codex ensemble review): when LIVE, spawn
>    only the Claude `follow-up-critic`; the merge in step 3 proceeds
>    Claude-only per the existing no-show contract, + one `epm:progress`
>    note.
> 3. **Merge the verdicts PER PROPOSAL** (single pass — no round loop;
>    `single_pass: true`). For each proposal: both `not-redundant` →
>    `not-redundant`. Both `redundant` → `redundant` (the merged
>    rationale unions both critics' duplicate pointers). `not-redundant`
>    vs `redundant` disagreement → spawn the `reconciler` (marker mode,
>    `Role under adjudication: follow-up-critic`, binding binary
>    `not-redundant | redundant`; it posts the canonical
>    `epm:review-reconcile` marker). A Codex twin no-show falls back to
>    the single-Claude `follow-up-critic` verdict (workflow.yaml §
>    ensemble_review; no-show confirmed per the Step 5b
>    durable-verdict-first rule — check `epm:followup-value-critique-codex`
>    + the round-fresh output file before declaring it). An UNCITED
>    `redundant` verdict (no concrete
>    duplicate named) is non-binding — treat it as `not-redundant` for
>    that proposal (cite-or-drop, mirrors the reconciler's ungrounded-
>    blocker rule).
> 4. **Act on the merged verdict, per proposal:**
>    - **`not-redundant`** → the proposal proceeds through the EXISTING
>      routing UNCHANGED (the caller's normal selection / partition /
>      pick logic below runs on it). Its rationale (what new info it adds)
>      is carried forward for the dashboard but does not change routing.
>    - **`redundant`** → the proposal does NOT run and is NOT routed.
>      SAVE it as a new task at status `on_hold` (set-aside, revivable via
>      `set-status <M> proposed`, excluded from auto-dispatch) carrying
>      `parent_id: <N>` and a `## Value critique` body section with the
>      verbatim WHY-IT-DUPLICATES rationale + the pointer (the duplicated
>      task / settled open-question anchor / sibling). File it in ONE
>      atomic call that lands the task DIRECTLY at `on_hold` (never a
>      two-step `new` → `set-status on_hold`, which leaves a window where
>      the proposal sits at `proposed` and a concurrent PM auto-dispatch
>      pass could pick it up — the exact outcome VC exists to prevent):
>      `task.py new --status on_hold --parent <N> --kind experiment --goal
>      "<the proposal's Goal>" --title "<proposal title>" --body-file
>      <spec-with-value-critique-section>.md`. Post
>      `epm:followup-parked-redundant v1` on the PARENT (fields per
>      workflow.yaml § markers: `parked_task_id`, `parent`,
>      `proposal_rank`, `title`, `duplicates`, `rationale`). Announce in
>      chat per the "Announce every follow-up/child task in chat" rule:
>      `Parked #<M> '<title>' on_hold (redundant — duplicates <X>; child
>      of #<N>, revivable via set-status <M> proposed)`. NEVER silently
>      drop a `redundant` proposal — `on_hold` is the durable home for
>      "saved but not worth running now."
> 5. **Hand the surviving (`not-redundant`) proposal set back** to the
>    caller. If EVERY proposal screened `redundant`, the caller's
>    selection finds no candidate and falls through exactly as if the
>    proposer had returned none.

The cheap-band flow:

C0. **Idempotency + run the proposer (once per park, shared marker).**
   FIRST: if this park already dispatched a cheap round whose loop is
   in flight or done — i.e. an `epm:followup-scope v1` with
   `source: proposer-9b-cheap` exists for which a matching
   `epm:same-issue-followup-run v1` does NOT yet exist (in flight), OR
   the cheap-band round cap (C2) is already hit — SKIP this block (it is
   a no-op on a backstop-tick / re-entry; the loop or the cap is the
   durable record; a CAP-HIT skip is ordered, not silent — first post
   the § Cheap-band cap-park surfacing notes, block C2). Otherwise: if
   an `epm:follow-ups v1` marker for THIS
   park is not already present (the autonomous block below may have
   posted it, or a re-entry did), spawn `follow-up-proposer` and post
   `epm:follow-ups v1` (same marker both sites share). If it is already
   present, reuse it — do NOT re-run the proposer. (In autonomous mode
   the proposer runs once and both this block and the autonomous block
   below consume the same `epm:follow-ups v1`. The proposer always posts
   its proposal list when it runs; an empty list means it found no
   follow-ups, and C1 then selects nothing.)
C0a. **Run the value-critique (redundancy screen) — subroutine VC above.**
   Before selecting any candidate, run VC over the `epm:follow-ups v1`
   proposal set (idempotent — a re-entry reuses the existing merged
   verdict). VC parks every `redundant` proposal at `on_hold` and hands
   back only the `not-redundant` survivors. C1 below selects from the
   SURVIVORS only — a `redundant` cheap proposal is parked, not auto-run.
   (VC runs once per park and the autonomous block + Step 10b reuse its
   verdict, so this is not a per-block cost.)
C1. **Select the cheap-band candidate.** Among the surviving
   (`not-redundant`) proposals, keep those
   that are ALL of: `question_relation: same`, `auto_run: yes`, and
   carry a parseable `est_gpu_hours` with `0 < est_gpu_hours < 20`
   (strict `< 20`; `est_gpu_hours: 0` is the Step 9a-ter free-analysis
   case, already handled; exactly `20` does NOT qualify). Take the
   TOP-RANKED such proposal.
   - **Fail-safe (missing / unparseable estimate).** A `same` proposal
     whose `est_gpu_hours` is absent or unparseable does NOT auto-run —
     it is left for the user (interactive: surfaces at Step 10b;
     autonomous: routed by the autonomous-only block below as an
     `auto_run`-gated `same` proposal under its own round cap). Mirror
     of the Step 2c plan-cap fail-safe: a missing GPU estimate parks,
     never auto-runs. State the skip reason in one chat line.
   - **`headline_affecting` is NOT consulted** for this band — a cheap
     `same` follow-up runs whether or not it moves
     the headline.
C2. **Cheap-band round cap.** At most **2** cheap-band auto-run rounds
   per task, counted by `epm:same-issue-followup-run v1` markers with
   `source: proposer-9b-cheap`. Run markers whose `outcome` begins
   `retroactive-close` do NOT count toward this cap — they record
   bookkeeping closure of a round that already ran (or was superseded),
   not a new auto-run (Step 0 § Stale-label disposition rule). Beyond
   the cap, further cheap `same`
   proposals survive in `epm:follow-ups v1` for manual pick. (This cap
   is INDEPENDENT of the autonomous `auto_run`/expensive-band cap, which
   counts `source: proposer-9b`. The natural breakpoint is the re-park
   at `awaiting_promotion` after each round, where the user sees the
   updated body before any further cheap follow-up fires.) The cap stops
   a chain of cheap follow-ups from auto-running indefinitely.

   **Cheap-band cap-park surfacing (#1558 — SURFACING only: the cap
   above is unchanged, no new auto-run, no new gate, no new marker
   kind).** Same contract as Step 9a-ter § Cap-park surfacing (#1548) —
   fixed leading token, per-(task, verbatim `followup_ref=`) idempotency
   grep, `epm:progress` reuse, auto-continue — with C2-keyed fields.
   Two firing moments: (a) a C0 CAP-HIT skip (the cap-hit arm only — an
   in-flight round has not consumed its slot) while the latest
   `epm:follow-ups v1` marker, if any, still lists ≥1 unrun
   C1-qualifying proposal (`same`, `auto_run: yes`, parseable
   `0 < est_gpu_hours < 20`) — post, then skip (no marker ⇒ nothing to
   post); (b) immediately after loop step 4 posts a counting
   `epm:same-issue-followup-run v1` (`source: proposer-9b-cheap`,
   `outcome` not `retroactive-close`-led) that consumes the final
   cheap-band cap slot (the C2 count reaches 2) — post for each
   remaining unrun C1-qualifying proposal; surplus after a NON-final
   round is NOT cap-parked (a future park may still dispatch it). Skip
   entries already run (`followup_label` / verbatim-title match),
   parked redundant (`epm:followup-parked-redundant v1`), fail-safe
   parks (missing/unparseable estimate — not cap parks), or already
   noted. `screened=` carries the VC verdict when VC ran for that
   proposal set; a C0 cap-hit skip precedes C0a, so `pending-screen` is
   expected there. Per parked entry, post the 9a-ter-shape
   `epm:progress` note (same `post-marker` template as § Cap-park
   surfacing) with C2-keyed fields: `followup-parked-by-cap
   followup_ref=<verbatim follow-up title> rank=<1-based surfaced-order
   position, or 'unranked'> screened=<not-redundant|pending-screen>
   cost_class=needs-gpu cap_consumed_by=<followup_label of the latest
   counting run row (source: proposer-9b-cheap)>
   alternative=raise-9b-cheap-cap-or-manual-pickup — the 2-round
   cheap-band cap parked this follow-up; a future planner/human may
   weigh raising the cap vs manual pick at Step 10b`.

C3. **Dispatch the round.** If a candidate survives C1+C2, post
   `epm:followup-scope v1` (`source: proposer-9b-cheap`, fields per
   workflow.yaml § markers, carrying the proposal's
   `est_gpu_hours`) and enter the **same-issue follow-up loop** below
   INSTEAD of parking — the task leaves `awaiting_promotion` and
   re-enters at `followups_running` (tag `followup-auto`). Skip the
   PushNotification → chat prompt park flow this round — but FIRST
   re-arm the `/issue-tick` backstop: CRON-TEARDOWN already ran at the
   `awaiting_promotion` transition at the top of this Step 9b, so NO
   cron is armed here, in EITHER session mode (#1112:
   a cheap-band round launched a multi-hour run with no
   tick armed). BEFORE dispatching any loop work, run the Step 6d.2
   ARM-GUARD shape — `CronList` whole-string match, else
   `CronCreate(cron="*/45 * * * *", prompt="/issue-tick <N>",
   recurring=True, durable=False)`, then re-list and assert exactly
   one — per the loop's "Loop liveness backstop"
   below. The plan still passes through the Step 2c plan-approval
   gate inside the loop — an over-cap (`est_gpu_hours` mis-estimated low
   but the realized plan exceeds `EPM_PLAN_AUTOAPPROVE_GPU_HOURS`) plan
   parks IN PLACE at `followups_running` (autonomous) or asks
   (interactive), so the cost cap is the final backstop even if the
   `est_gpu_hours` estimate was wrong.
C4. **No candidate → fall through.** When no cheap-band candidate
   survives C1+C2, this block dispatches nothing: proceed to the
   autonomous-only block (autonomous sessions) or the park flow
   (interactive sessions). The `epm:follow-ups v1` C0 posted persists —
   its proposals (cheap ones beyond the cap, expensive ones, fail-safe
   skips, `auto_run: no`) carry forward for the autonomous block to
   route or for the user to pick at Step 10b post-promotion (the Step
   10b proposer-already-ran short-circuit then reuses this marker).

**Autonomous follow-up auto-spawn (autonomous mode only — fires here
because Step 10b never runs autonomously; handles the EXPENSIVE
`est_gpu_hours >= 20` / no-estimate `auto_run: yes` path, after the
cheap-band block above has had first refusal).** When
`EPM_AUTONOMOUS_SESSION=1`, the parent task parks at
`awaiting_promotion` and Step 10 / 10b never fire on their own
(promotion is ALWAYS human-only). To stop autonomous research from
stalling on every result, the orchestrator fires the follow-up proposer
HERE — after the auto-merge has landed the clean-result on `main` (the
Step 9b CRON-TEARDOWN already ran at the `awaiting_promotion`
transition above) — and routes the `auto_run: yes` proposals by
`question_relation` (QUESTION IDENTITY — one mechanism, three entry
points; the other two are the Step 0 followup-scope dispatch for
chat-requested follow-ups and the interactive Step 10b pick):
`substantially-different` proposals (and untagged ones ONLY from
pre-2026-06-09 legacy markers — a newer untagged proposal trips the
freshness guard in step 3 below) are FILED-ONLY — created as
`proposed` child tasks for manual triage, NEVER auto-spawned as
sessions (no autonomous child sessions, ever, from this path; the
only execution path for an automatic follow-up is the same-issue
loop); `same` proposals are NEVER filed as children — the top-ranked
one runs ON this issue via the same-issue follow-up loop below
(status `followups_running`, tag `followup-auto`). Interactive
sessions SKIP this block entirely (they still hit Step 10b
post-promotion as today, which routes the user's pick by the same
`question_relation`). Idempotent: when an `epm:follow-ups-autospawned v1` marker is
already present on this parent, do NOT re-run the proposer or re-create
children (covers re-invocation / backstop-tick re-entry; filing
twice + duplicate `epm:follow-ups` clutter are the failure modes this
guard avoids) — instead run the lightweight RECONCILE pass (step R
below) which only verifies the listed children exist.
Depth-bounded: the block is skipped entirely once this parent's
`parent_id` chain already has ≥3 auto-filed ancestors (step 0 below),
so the autonomous follow-up filing tree cannot recurse past depth 3.

The autonomous flow:

0. **Depth cap (run FIRST).** Trace this task's `parent_id` chain upward
   and count ancestors that themselves carry an
   `epm:follow-ups-autospawned v1` marker (i.e. were auto-filing origins,
   not merely manually-filed parents). If that count is **≥ 3**, do NOT
   auto-file children: spawn the proposer and post its proposals as
   `epm:follow-ups v1` for the user to pick manually, then post
   `epm:follow-ups-autospawned v1` with `auto_spawn_skipped:
   depth_cap_reached` and an empty `spawned` list (so the idempotency
   guard still trips and the dashboard records why), and continue to the
   park flow. This bounds the autonomous follow-up filing tree to depth
   3 — cheap insurance against unbounded recursive filing if a filed
   child is later run and reaches its own Step 9b.
1. Read the latest `events.jsonl` (fresh, NOT a stale cached view).
   - If `EPM_AUTONOMOUS_SESSION` is unset → skip the block.
   - If `epm:follow-ups-autospawned v1` is ALREADY present → run the
     **RECONCILE pass** (step R) instead of re-running the proposer, then
     continue to park — step R is filing-verification ONLY and never
     re-evaluates the step-3 partition (at most one step-3 partition per
     task lifetime; see step R's scope contract, #1588). (With no session
     spawning there is no
     crash-between-marker-and-spawn window; the residual self-heal is a
     crash between child creation and the marker post, which the
     duplicate-title guard in step 3 covers.)
   - Otherwise → continue to step 2.
2. Spawn `follow-up-proposer` (clean-result is available — it was just
   promoted in-place by the analyzer). Post the proposals to
   `events.jsonl` as `epm:follow-ups v1` (same marker the interactive
   Step 10b would post; sharing the marker means the dashboard +
   downstream readers don't care which site fired the proposer).
2-bis. **Run the value-critique (redundancy screen) — subroutine VC
   above.** Run VC over the `epm:follow-ups v1` set (idempotent — if the
   cheap-band block's C0a already ran it this park, reuse the merged
   verdict). VC parks every `redundant` proposal at `on_hold`
   (`epm:followup-parked-redundant v1`) and hands back only the
   `not-redundant` survivors. Steps 3-6 below PARTITION + route the
   SURVIVORS only — a `redundant` proposal is never filed as a child and
   never enters the same-issue loop; it is saved on_hold for manual
   revival. This screen gates BOTH the child-filing path AND the
   same-issue-loop path.
3. Parse the surviving (`not-redundant`) proposals, keep those with
   `auto_run: yes` in ranked
   order, and PARTITION them by `question_relation`. **The routing
   litmus is the Takeaways test:** *would the result rewrite THIS
   issue's `## Takeaways`?* If yes → `same` (stays on this issue via the
   same-issue follow-up loop, never a child). Changing method, dose,
   panel, seeds, eval surface, prompt bank, or adding a control/baseline
   on the SAME question is ALWAYS `same`. `substantially-different` is
   reserved for work that would change the task's `## Goal` /
   open-questions anchor — a genuinely new question. This bias-toward-
   same-issue litmus is the same one the `follow-up-proposer` applies
   when tagging (`.claude/agents/follow-up-proposer.md` §
   "question_relation tag — criteria") — the partition just consumes its
   tags; when a tag looks miscast against the litmus, treat it like an
   untagged proposal (re-spawn-once below). **Untagged
   proposals — freshness guard:** the legacy fallback (treat an
   untagged proposal as `substantially-different` so nothing in
   flight breaks) applies ONLY when the `epm:follow-ups v1` marker
   carrying the proposals was posted before 2026-06-09 (pre-dating
   the question-identity routing fix). On a newer marker, a missing
   `question_relation` tag is a proposer-contract violation — the
   usual cause is a stale `follow-up-proposer.md` in a long-lived
   session/worktree that predates the fix (#533: a textbook `same`
   corrective re-run was routed to a child task via
   this fallback). Re-spawn `follow-up-proposer` ONCE, instructing it
   to re-emit the SAME proposals with `question_relation` (and
   `followup_label` for `same`) tags per the criteria in
   `.claude/agents/follow-up-proposer.md` § "question_relation tag —
   criteria", read from the CURRENT `main` checkout (repo root), not
   the session worktree's possibly-stale copy; the re-emit posts a
   fresh `epm:follow-ups v1` marker that supersedes the untagged one.
   If the re-emit is STILL untagged, route the affected proposals as
   `substantially-different` and record the violation in the
   `epm:follow-ups-autospawned v1` marker body
   (`proposer_contract_violation: question_relation missing after
   re-spawn`). Proposals tagged `auto_run: no` are skipped in BOTH
   partitions — they survive in the `epm:follow-ups v1` marker for
   the user to pick from manually.
   - **`substantially-different`** → the child FILING path (steps
     4-5 below). Take the top **2** (cap; bounds fan-out so a parent
     never files more than 2 children per round regardless of how
     many `auto_run: yes` proposals the proposer found). Drop any kept
     proposal whose title duplicates an existing `parent_id=<N>` child
     (guards against a partial prior run that created the task before
     crashing).
   - **`same`** → the same-issue follow-up loop (§ below, via step 6).
     **First EXCLUDE any `same` proposal the cheap-band block above
     already dispatched this park** (its `epm:followup-scope v1` carries
     `source: proposer-9b-cheap` — match by `followup_label` / verbatim
     spec): if the cheap band took a round, this block does NOT also
     dispatch a `same` round in the same park (one same-issue round per
     park). Of the REMAINING `same` + `auto_run: yes` proposals (those
     with `est_gpu_hours >= 20` or a missing estimate — the cheap band
     skipped these), select the TOP-RANKED one ONLY if the autonomous
     EXPENSIVE-band round cap allows (fewer than 2
     `epm:same-issue-followup-run v1` markers with
     `source: proposer-9b` on this task). The rest — and all `same`
     proposals once the cap is hit — survive in `epm:follow-ups v1`
     for manual pick.

     **Expensive-band cap-park surfacing (#1575 — SURFACING only: the
     expensive-band cap is unchanged, no new auto-run, no new gate, no
     new marker kind; autonomous mode only, like the cap it surfaces).**
     Same contract as Step 9a-ter § Cap-park surfacing (#1548) and block
     C2 § Cheap-band cap-park surfacing (#1558) — fixed leading token,
     per-(task, verbatim `followup_ref=`) idempotency grep
     (context-cheap events-file grep, never a full-body page-in),
     `epm:progress` reuse, auto-continue — with expensive-band-keyed
     fields. PRIMARY firing moment — this step-3 partition itself: after
     the `same` partition selects (or cap-blocks) its dispatch, post the
     note for EVERY surviving expensive-band-eligible `same` proposal
     (`auto_run: yes`, `est_gpu_hours >= 20` or missing estimate) NOT
     dispatched this entry, cap state irrelevant. Reachability rationale
     (why surplus is parked NOW, unlike C2's non-final-round carve-out):
     step 1's idempotency routes every re-entry with
     `epm:follow-ups-autospawned v1` present to the RECONCILE pass,
     which never re-partitions — this step-3 execution is the band's
     ONLY partition per task lifetime, so a non-dispatched survivor has
     NO future dispatcher; leaving it bullet-only is the #1575 gap.
     DEFENSIVE-PARITY moment: the loop step-4 reminder (a counting
     `epm:same-issue-followup-run v1`, `source: proposer-9b`, `outcome`
     not `retroactive-close`-led, consumes the final expensive-band cap slot
     — post for each remaining unrun eligible proposal); idempotent
     against the step-3 notes via the per-`followup_ref` grep,
     independently reachable only if a future contract change makes
     multiple expensive-band rounds dispatchable. Skip entries already
     run (`followup_label` / verbatim-title match), parked redundant
     (`epm:followup-parked-redundant v1`), dispatched this park by the
     cheap band (`epm:followup-scope v1`, `source: proposer-9b-cheap`),
     or already noted. There is NO fail-safe-park skip class here — a
     missing/unparseable estimate is a first-class expensive-band
     candidate by design (the C1 fail-safe routes it to this block), so
     its cap park IS in scope. `screened=` carries the VC verdict; VC
     (step 2-bis) has run by step 3, so `not-redundant` is expected at
     the step-3 moment — `pending-screen` only for a proposal set VC
     never screened. Depth-cap (step 0) parks are OUT of scope —
     `epm:follow-ups-autospawned v1` with
     `auto_spawn_skipped: depth_cap_reached` is already their durable
     record. Per parked entry, post the 9a-ter-shape `epm:progress`
     note (same `post-marker` template as § Cap-park surfacing) with
     expensive-band fields: `followup-parked-by-cap
     followup_ref=<verbatim follow-up title> rank=<1-based
     surfaced-order position, or 'unranked'>
     screened=<not-redundant|pending-screen> cost_class=needs-gpu
     cap_consumed_by=<followup_label of the proposal dispatched this
     entry, else 'none'; at the step-4 defensive-parity moment: the
     latest counting run row (source: proposer-9b)>
     alternative=raise-9b-expensive-cap-or-manual-pickup — the
     one-partition-per-task expensive band (2-round cap) parked this
     follow-up; a future planner/human may weigh raising the cap vs
     manual pick at Step 10b`.

4. For each kept `substantially-different` proposal, in rank order, create the child in ONE atomic
   call — `task.py new --goal` writes BOTH the `goal:` frontmatter AND
   the `## Goal` H2 the child's Step 0c gate requires, so there is no
   window where the child exists without a Goal:
   ```bash
   # Shell-quote the title + Goal (proposal text may contain quotes /
   # backticks): use python -c shlex.quote or printf %q, never bare
   # interpolation. The proposal's **Goal:** field (see
   # follow-up-proposer.md output template) supplies the one-sentence Goal.
   CHILD_ID=$(uv run python scripts/task.py new \
     --parent <N> --kind experiment \
     --goal "<one-sentence Goal from the proposal's **Goal:** field>" \
     --title "<proposal title>" \
     --body-file <path-to-pre-filled-spec>.md \
     | grep -oP '#\K\d+')
   ```
5. **Post `epm:follow-ups-autospawned v1` NOW** — after the child tasks
   exist (step 4). The marker NAME is kept for dashboard back-compat;
   its body carries `execution: filed-only` and the `spawned` list now
   has FILED semantics (children created at `proposed`, no sessions —
   see workflow.yaml § markers). It lists every created child (id +
   title + proposal rank) and every `auto_run: no` proposal that was
   skipped (rank + title + auto_run_reason). This is the durable
   idempotency claim: it records the children so a re-entry reconciles
   (step R) rather than re-creating. Announce each filed child in chat
   per the existing rule (Step 10b § "Announce every follow-up/child
   task in chat"): `Filed #<CHILD_ID> '<title>' (child of #<N>,
   status:proposed — awaiting manual triage)`. Do NOT spawn sessions
   for them — a filed child executes only when a human triages it and
   invokes `/issue <CHILD_ID>`.
6. **Branch on the `same` partition.** If step 3 selected a `same`
   proposal, post `epm:followup-scope v1` (`source: proposer-9b`,
   fields per workflow.yaml § markers) and enter the **same-issue
   follow-up loop** below INSTEAD of parking — the task leaves
   `awaiting_promotion` and re-enters the pipeline at
   `followups_running`, so skip the
   PushNotification → chat prompt → CRON-TEARDOWN park flow this
   round (re-arm the `/issue-tick` backstop cron FIRST via the Step
   6d.2 ARM-GUARD shape — the Step 9b CRON-TEARDOWN at the
   `awaiting_promotion` transition already removed it; see the loop's
   "Loop liveness backstop").
   Otherwise continue to the existing park flow below
   (PushNotification → chat prompt → CRON-TEARDOWN → §5 marker via
   `post_step_completed.py --step 9a-bis --exit-kind parked` → EXIT).

**Step R — RECONCILE pass** (re-entry with the marker already present):
read the `spawned` list from `epm:follow-ups-autospawned v1`. For each
listed child, verify it exists via `task.py view <CHILD_ID> --json`;
re-create one that is missing (same atomic `task.py new --parent`
call as step 4). NEVER spawn sessions — this pass only verifies
filing. **Scope contract (#1588):** step R does NOT re-read
`epm:follow-ups v1`, does NOT re-run the step-3 partition (neither the
child-filing side nor the expensive-band `same` selection), and NEVER
posts a new `epm:followup-scope v1` — the step-3 partition runs at most
ONCE per task lifetime, at the first (marker-less) entry (zero times
when step 0's depth cap fires), so a re-park can never dispatch an
additional expensive-band round; non-dispatched survivors stay in
`epm:follow-ups v1` for the user's Step 10b pick (their cap parks were
already surfaced at the step-3 § Expensive-band cap-park surfacing
moment, #1575). Then continue to park.

Cost discipline: this block adds NO new cost gate. A filed child, once
a human triages it and runs `/issue <CHILD_ID>`, hits its own Step 2c
`--auto-approve-if-autonomous --gpu-hours` cap; over-cap plans park at
`plan_pending`, consistent with `tests/test_no_dollar_budget_caps.py`.
Promotion of the parent stays human-only. The recursive surface is
bounded twice over: same-issue rounds are capped at 2 per task
(expensive band: at most one round is dispatchable under the current
contract — see step 5 § Round caps / step R's scope contract, #1588),
and child FILING is capped at 2 per parent per round AND hard-stopped at
chain depth 3 by step 0 (so even if filed children are later run, the
filing tree is both width-bounded and depth-bounded, not exponential).

**Same-issue follow-up loop (`question_relation: same`).**

One mechanism, four entry points: (a) the Step 9b autonomous
expensive-band partition above (`source: proposer-9b`,
`est_gpu_hours >= 20` / no estimate), (a-cheap) the Step 9b cheap-band
block (`source: proposer-9b-cheap`, `0 < est_gpu_hours < 20`,
`question_relation: same`) which fires in BOTH interactive and
autonomous sessions, (b) a chat-requested
same-question follow-up (`source: user-chat` — the chat session posts
`epm:followup-scope v1` on #N, then re-invokes `/issue <N>`; the Step
0 followup-scope dispatch lands here), and (c) an interactive Step
10b pick (`source: step-10b-pick`). Step 9a-ter (the inline
free-analysis auto-run) is this loop's zero-GPU sibling under the
same principle — a follow-up that answers the SAME question as the
task Goal runs ON this issue; 9a-ter handles the zero-GPU floor
inline, this loop handles the GPU-backed case (the cheap `< 20` GPU-h
band auto-runs in both modes; the expensive band auto-runs only in
autonomous mode or on an explicit user pick).

**Canonical §5 step id for this loop is `9b-same`.** workflow.yaml
§ steps has no `9b` id — the prose name "Step 9b" is NOT a step id — so
any `scripts/post_step_completed.py` post made from within this loop
passes `--step 9b-same` (the helper aliases legacy `9b` → `9b-same` as
a backstop, #1499). A helper refusal (exit 2, unknown step id) means
the resume record was NOT posted: re-run with a canonical id from the
stderr `Known:` list / `Did you mean` hint before continuing — never
continue past the refusal (#1335: the dropped record degraded crash
recovery for the `followups_running` hold).

**Loop liveness backstop (arm BEFORE dispatching loop work — BOTH session modes).**
ANY session driving this loop — interactive (typically entry point (b),
a chat session) OR autonomous (the Step 9b C3 cheap-band / step-6
expensive-band dispatches, where the `awaiting_promotion` CRON-TEARDOWN
has already removed the Step 0 arm) — must verify/arm the
`/issue-tick <N>` backstop cron (same `CronList`/`CronCreate` ARM-GUARD
shape as Step 0 / Step 6d.2; a no-op when already armed) before
dispatching its first planner / implementer / stage subagent. While
loop work waits on any long phase, the § Long-phase heartbeat duty
(Step 6d.2) binds. An INTERACTIVE session must additionally post every
stage-dispatch breadcrumb
(`stage=followup-<phase>`, Step 9 entry-guard convention) with the
`worktree=` field **and a `label=<followup_label>` field naming the
round's label** (consumed by `task_workflow.executing_followup_label`
for mid-round resume and by the watcher's on-behalf run-marker post;
breadcrumbs predating this contract lack it — consumers fall back to
the head of the unrun queue). These `stage=followup-<phase>` breadcrumbs are bound
by the SAME Step 9 entry-guard predicate AND the same NON-SKIPPABLE
pre-dispatch dedup check (§ Step 9 entry guard) — the status being held
at `followups_running` does not exempt them (#778: the
round-1 `followup-implementing` dispatch was duplicated by a concurrent
orchestrator minutes later, two implementers concurrently
editing one worktree). Know what each mechanism covers: the cron handles
only the alive-but-stalled case — a `durable=False` cron dies with the
session that armed it; `autonomous_session_watch.py`'s AUTO-RESPAWN
passes read only the autonomous registry (`spawn-issue --auto`
entries), and the step-2 `register-current` manual registration buys
ALERT-ONLY stalled/crash visibility (a user-driven session is never
auto-respawned, #505) — so nothing external RE-DRIVES an interactive
session driving this loop. If the session is going to be
closed — or the user asks for a handoff — while loop work is in flight,
the mid-flight handoff rule (§ Orchestration Procedure preamble)
applies: spawn `spawn_session.py spawn-issue --issue <N> --auto`
IMMEDIATELY; that registration is the only mechanism that survives
session death. (#505: an interactive chat
session driving this loop was closed mid-implementer-dispatch with no
cron armed, no registry entry, and no worktree breadcrumb; the task
orphaned at `running`.)

**Loop-entry ownership re-check.** When entering this loop from a
resume / re-invocation (including the Step 0 followup-scope dispatch),
FIRST re-run the Step 0 single-orchestrator guard: if another live
session is mapped to this issue, stop the stale session
(`spawn_session.py stop`) before dispatching any loop work — two live
orchestrators driving one round is the #778 root cause.

1. **Scope marker.** Ensure an `epm:followup-scope v1` exists for this
   round (the Step 9b partition posts it at step 6 above; the chat /
   Step 10b entry points post it before re-invoking). Fields per
   workflow.yaml § markers: `followup_label` (kebab-slug; names the
   artifact dir `eval_results/issue_<N>/<followup_label>/`), `source`,
   the verbatim proposal spec (or the user's verbatim chat request),
   and the GPU-hour estimate. **MULTIPLE `epm:followup-scope` versions
   for one issue may exist** — corrections are WITHIN-label: the
   authoritative scope for a round is the latest-(`ts`, `version`)
   entry AMONG the entries carrying THIS ROUND'S `followup_label` (an
   unlabeled correction note attributes to the immediately-preceding
   label — `task_workflow.followup_label_groups`; see #658's
   `persona-vectors-style-rb` v3→v7 chain — NOTE v8 carries a DIFFERENT
   label, `a35-mlp-downstream-chain`, a separately-queued round, not
   part of the chain). A later entry with a DIFFERENT label is a
   separately-QUEUED round (the #763 shape), never a supersession. Do
   NOT cache the
   entry-time version — step 3 re-reads the latest before snapshotting.
2. **Re-enter the pipeline.** **FIRST record the initiation mode as a
   tag** (before the status flip, so the `task.py` missing-tag warning
   stays quiet): `uv run python scripts/task.py add-tag <N>
   followup-auto` when `source: proposer-9b` OR `source:
   proposer-9b-cheap` (both are proposer-initiated auto-runs);
   `uv run python scripts/task.py add-tag <N> followup-manual` when
   `source: user-chat` or `source: step-10b-pick`. EXACTLY these two tag
   names — a bare `followup` tag does not count (#533). (Both
   tags may accumulate over a task's life — they are history, not
   exclusive
   state.) **Then** `task.py set-status <N> followups_running` — the
   round HOLDS this status end-to-end (see the status-hold rule in step
   3); the CLI warns if neither tag is present at this transition. The
   planner-exempt distinction (re-run with different seeds,
   monitoring, syncing, or a bug-fix re-run, per the CLAUDE.md
   `/adversarial-planner` carve-out) still governs whether
   `/adversarial-planner` is re-invoked in step 3 — the STATUS no
   longer encodes it. The marker trail
   records the transition (`epm:status-changed`); `has_clean_result`
   stays sticky across the re-entry. **In the same step, re-register
   the driving session:** `uv run python scripts/spawn_session.py
   register-current --issue <N>` (infers this session's Happy id from
   the process ancestry + the daemon; writes `issue-<N>.json` for
   autonomous sessions / `manual-issue-<N>.json` for interactive ones,
   matching how the session was spawned). The revival flips a
   parked/terminal task back to ACTIVE, but the watcher's registry
   entry was DELETED at the terminal transition — without
   re-registering, the revived run is invisible to every
   registration-based watcher pass until the orphan sweep's ~90-min
   staleness gate (#472: a revival ran orphaned
   for hours). Registration failure is non-fatal to the loop (the
   orphan sweep remains the backstop) but state the failure rather
   than swallowing it.
3. **Abbreviated cycle**, all on THIS issue. **Status-hold rule: the
   task STAYS at `followups_running` for the WHOLE round** — planner
   amendment → consistency-checker → plan gate → implementer /
   code-review → provision → run → upload-verify → terminate →
   analyzer re-fold → clean-result-critic. The normal pipeline
   `set-status` calls (`planning` / `plan_pending` / `approved` /
   `running` / `verifying` / `interpreting` / `reviewing`) are SKIPPED
   during a same-issue follow-up round; phase visibility comes from the
   existing stage breadcrumbs (`stage=followup-<phase>`) and
   `epm:progress` markers. **Code-enforced** (post-#533/#560):
   `task.py set-status` REFUSES
   `followups_running -> <any of those>` (override:
   `--force-followup-exit`, only to deliberately abandon the round), and
   a mid-round plan-gate call (`--auto-approve-if-autonomous`) fires the
   gate decision + markers while HOLDING the status
   (`PLAN_GATE_DECISION: ... (followups_running hold: status
   unchanged)`). An over-cap (or interactively-awaiting) plan parks IN
   PLACE at `followups_running` — the Step 2c plan-approval gate still
   fires, it just no longer moves the status to `plan_pending`. The
   round exits the status only at the re-park:
   `set-status <N> awaiting_promotion` (or `blocked` on a failure
   exit). **Mid-round defer/teardown is an exit too — re-park in the
   SAME action sequence as the teardown:** a mid-round defer (wedged or
   pathological run torn down, user defer — any deliberate abandonment
   of the round short of a `blocked` failure exit; no
   `--force-followup-exit` needed, `awaiting_promotion` is not in the
   refused set) tears down the pod/instance FIRST, runs
   `set-status <N> awaiting_promotion` as the NEXT command (the § User
   pause affordance teardown-first-park-last ordering — distinct
   mechanism: a user pause parks at `on_hold`, this defer exit re-parks
   at `awaiting_promotion` with the label closed; never leave
   `followups_running` with no live round compute), THEN closes the
   round's label by posting the step-4 completion marker with
   `outcome: deferred — <one-line reason>` (label closure is
   outcome-agnostic — `task_workflow.unrun_followup_labels` — so Step 0
   / the tick never auto-re-dispatch the deferred round; a deliberate
   later resume posts a FRESH scope under a NEW label; a deferred
   proposer-band round still counts toward its step-5 cap). The tick
   STALE-REDRIVE / watcher re-park are recovery backstops, not the
   owner (#825: a pathological fit was torn down
   with no re-park — the parent stranded at
   `followups_running` until the next tick re-drive re-parked
   it).
   - **Immediately before the planner snapshots the scope, RE-READ the
     authoritative scope FOR THIS ROUND'S LABEL** —
     `task_workflow.executing_followup_label` (the newest
     `stage-dispatch stage=followup-*` breadcrumb's `label=` field newer
     than the newest run marker; fallback: the head of the dispatchable
     `unrun_followup_labels`) — never the bare latest scope: under
     label-grouped dispatch (#894) the latest entry may be a DIFFERENT
     queued label. A mid-round correction to the SAME label is still
     picked up (it raises that label's authoritative `(ts, version)`);
     never plan against an entry-time snapshot, or a session that
     entered on `v3` and snapshotted before a `v5`/`v6` correction
     landed plans stale (the #658 bug).
     The same pre-snapshot re-read covers the canonical GOAL:
     `frontmatter.goal` + the latest `epm:goal-updated` ts — the
     adversarial-planner § Goal-currency gate applies in AMENDMENT scope too
     (an amendment plan drafted against a stale Goal is the #922 bug class,
     the Goal sibling of this bullet's #658 scope bug). The `followup_label` lives
     inside the marker NOTE body as free text (its format even differs
     across versions — `- followup_label: ...` dash-bullet, bare
     `followup_label: ...`, bold `**followup_label:** ...`, `; `-joined
     single-line `source: ...; followup_label: ...` (#1090/#841)), NOT as
     a top-level event key (top-level keys are `{by, kind, note, ts,
     version}` only) — `task_workflow.parse_followup_note_field` handles
     every observed form. The step-4 completion marker's
     `followup_label` derives from THIS SAME executing group
     (`group['followup_label']` verbatim) — never re-parsed from "the
     scope marker" independently, so the completion label can never
     diverge from the round that ran. This is the SAME shared helper the
     watcher uses (`autonomous_session_watch._post_followup_run_marker`
     resolves the round via `task_workflow.executing_followup_label`).
     Mechanical recipe:
     ```bash
     uv run python -c "
     import json
     from explore_persona_space.task_workflow import list_events, executing_followup_label
     g = executing_followup_label(list_events(<N>))
     print(json.dumps(g and g['authoritative'], indent=2))
     "
     ```
   - `/adversarial-planner` re-invoked in AMENDMENT scope: produces
     `plans/v{N+1}.md` as a ONE-VARIABLE diff plan against the issue's
     own latest prior run, not a from-scratch plan. Planner-exempt
     re-runs (step 2) skip this.
   - **Compute-character pre-launch statement** (canonical block: Step
     9a-ter § Compute-character pre-launch statement — same five elements,
     same > ~1h stop-and-vectorize + >~15 min measured-pilot / ≥2×
     pilot-extrapolated fence sizing + ≥~16 GB-RSS off-VM + ≥ ~5 GB off-`/`
     disk-routing + ≥ ~50 GB consuming-phase-off-VM + iterative-fit
     GPU-at-dispatch rules): REQUIRED in the
     `stage=followup-<phase>` dispatch breadcrumb (or an adjacent
     `epm:progress` note) before dispatching ANY stage of the round that
     launches a fit, sweep, or statistical battery — INCLUDING
     planner-exempt re-runs (step 2), which skip the amendment plan and
     its §9 sizing entirely, and analysis / re-fold stages reusing parent
     code. An amendment plan's §9 sizing does NOT substitute for it: the
     plan schedules the battery, the executor states the implementation's
     compute shape — #778's round re-ran the parent's serial 1000-draw
     null battery (2+h, projected 4–6h, vs ~15–30 min batched) under a
     plan that never said "serial".
   - **Pre-dispatch external-marker triage** (canonical block: Step 9
     entry guard § Pre-dispatch external-marker triage): REQUIRED — the
     same `stage=followup-<phase>` dispatch breadcrumb (or adjacent
     `epm:progress` note) carries the `external-markers triaged:` line
     before dispatching ANY compute-launching stage of the round. #779's
     `stage=followup-grid` dispatch is the founding incident.
   - `consistency-checker` diffs the amendment against the ISSUE'S OWN
     latest prior run — the latest prior plan version + the current
     clean-result body's `## Reproducibility` — NOT a `parent_id` task
     (see consistency-checker.md § Same-issue follow-ups).
   - Step 2c plan-approval gate as normal — the EXISTING
     `gates.inline plan_approval` gate, no new gate is registered:
     autonomous sessions auto-approve under
     `EPM_PLAN_AUTOAPPROVE_GPU_HOURS` and park at `plan_pending` over
     the cap; interactive sessions ask.
   - `experiment-implementer` + `code-reviewer` if the diff needs code
     changes (same ensemble shape as Step 5). The round's implementer brief
     follows the Step 4b brief contract INCLUDING its marker-version-
     discipline bullet — on a follow-up round prior
     `epm:experiment-implementation` rows ALWAYS exist, so a brief
     instructing a literal `v1` reproduces the #825 collision; the brief
     defers to max+1 (or tells the implementer to omit `--version`).
   - Fresh compute dispatch on the SAME issue, through the slice-6
     router exactly like the parent run: read the task's `backend:`
     frontmatter and run `dispatch_issue.py launch --issue <N>
     --intent "$INTENT" ${BACKEND:+--backend "$BACKEND"}` (see Step
     6b § "Operational dispatch (slice-6 router, ALL backends)" — do
     not duplicate its prose here). Follow-up rounds inherit the
     task's `backend:` frontmatter and the auto-routing default
     (empty → auto — RunPod first (#2054), then fellows + the free
     SLURM lanes; GCP provisioning disabled, #2028). The prior compute was torn down at Step 8;
     per-issue naming already supports re-dispatch.
   - Run → upload-verify → Step 8 terminate, as normal.
   - The `analyzer` RE-FOLDS the new finding into the EXISTING
     clean-result body — a new `### <result>` H3 under `## Results`
     (v4; on a grandfathered v3/v2 body the fold-in MIGRATES it to v4,
     SPEC.md § Follow-up consolidation), updating the H1 title /
     confidence tag if the result moves the
     headline. The
     `set-body` call passes NO `--snapshot` — `original-body.md`
     already preserves the pre-promotion original — and a moved headline
     is followed by `task.py set-title <N> "<new H1 text>"` (set-body
     preserves the old frontmatter `title`; the H1==frontmatter verifier
     check FAILs the 9a-bis re-gate otherwise; see analyzer.md §
     Same-issue follow-up re-entry).
   - `clean-result-critic` re-gates the UPDATED body (9a-bis as
     normal), then 9a-quater and the `awaiting_promotion` park run as
     normal — on this re-entry, 9a-quater's followup-scoped idempotency
     detects the round's unrecorded `followup_label` and runs its
     EXTEND pass (methodology-writer in EXTEND mode appends the new
     arm's section to `docs/methodology/issue_<N>.md`, refreshes the
     gist, re-pins the body's Methodology-reference link) instead of
     the parent-pass no-op. Planner-exempt re-runs take the
     no-new-methodology carve-out there.
   - Re-park at `awaiting_promotion`. ONE promotion verdict covers the
     whole updated body; a previously-promoted (`completed`) task that
     looped re-parks here and the user re-promotes.
4. **Completion marker.** Post `epm:same-issue-followup-run v1`
   (`followup_label` matching the scope marker — derive the label from
   the executing group per step 3's re-read, never a fresh independent
   parse — `source`, `round`,
   one-line `outcome`) when the loop re-reaches `awaiting_promotion`.
   The note MUST carry the `followup_label:` / `source:` / `round:` /
   `outcome:` fields field-led — line-initial one-per-line, or
   `; `-joined on one line (both parse); a PROSE-LED note parses no
   label, closes nothing, and undercounts both round caps (the #1090
   fu1 regression).
   This is the idempotency record: an `epm:followup-scope v1` with a
   matching run marker is RUN and is never re-dispatched. When this
   marker is cheap-band (`source: proposer-9b-cheap`, `outcome` not
   `retroactive-close`-led) and consumes the final cheap-band cap slot,
   immediately post the block-C2 § Cheap-band cap-park surfacing note
   for each remaining unrun C1-qualifying proposal — those entries are
   parked NOW, not at some future re-entry (#1558).
   Likewise when this marker is expensive-band (`source: proposer-9b`,
   `outcome` not `retroactive-close`-led) and consumes the final
   expensive-band cap slot, immediately post the autonomous block
   step-3 § Expensive-band cap-park surfacing note for each remaining
   unrun expensive-band-eligible proposal — DEFENSIVE PARITY with the
   cheap band: the step-3 primary moment normally already noted these
   (the per-`followup_ref` idempotency grep absorbs the overlap), and
   this clause fires independently only under a future multi-round
   dispatch contract (#1575).
5. **Round caps (two independent proposer-initiated caps).**
   - **Expensive autonomous band:** at most **2** rounds per task,
     counted by `epm:same-issue-followup-run v1` markers with
     `source: proposer-9b` (the `est_gpu_hours >= 20` / no-estimate
     autonomous-only path). Reachability (#1588): the band's only
     dispatcher is the autonomous block's step-3 partition, which runs
     at most ONCE per task lifetime (step 1's marker-presence
     idempotency routes every re-entry to step R, which never
     re-partitions), so at most ONE proposer-9b round is dispatchable
     under the current contract — the 2-round cap is a defensive bound
     that binds only if a future contract change makes multiple
     expensive-band rounds dispatchable (the same change the step-4
     DEFENSIVE-PARITY clause anticipates, #1575).
   - **Cheap band (both modes):** at most **2** rounds per task, counted
     by `epm:same-issue-followup-run v1` markers with
     `source: proposer-9b-cheap` (the `0 < est_gpu_hours < 20` path,
     enforced at block C2 above). This cap is what stops a chain of
     cheap follow-ups from auto-running indefinitely; the re-park at
     `awaiting_promotion` after each round is the user-visible
     breakpoint.

   Beyond either cap, further `same` proposals of that class survive in
   `epm:follow-ups v1` for manual pick. (Cheap-band cap parks are
   additionally surfaced via the block-C2 § Cheap-band cap-park
   surfacing note (#1558); expensive-band cap parks via the autonomous
   block step-3 § Expensive-band cap-park surfacing note (#1575).)
   USER-REQUESTED rounds
   (`source: user-chat` or `step-10b-pick`) do NOT count against either
   cap — the user asked explicitly, and interactive plan approval
   still gates each one. Run markers whose `outcome` begins
   `retroactive-close` do NOT count toward either round cap — they
   record bookkeeping closure of a round that already ran (or was
   superseded), not a new auto-run (Step 0 § Stale-label disposition
   rule).

Status-machine summary: `interpreting` / `reviewing` /
`awaiting_promotion` / `completed` + ≥1 unrun followup label →
`followups_running` (tag `followup-auto` | `followup-manual`; held
for the whole round) → `awaiting_promotion`. Never a child task.
(`followups_running` also retains its legacy meaning — parent
complete, `parent_id` children still in flight — see Step 10 step 5.)

Then post the chat-side prompt:

> Clean-result-critic PASS. The polished body is now live on task #\<N\>.
> When satisfied, promote it (USER-ONLY — no automation may do this):
>   `uv run python scripts/task.py promote <N> useful`     (paper-relevant)
>   `uv run python scripts/task.py promote <N> not-useful` (archive candidate)
> Then re-enter `/issue <N>` to fire Step 10.

> **Re-park BEFORE the §5 marker (same-issue follow-up rounds —
> #533):** during a follow-up round, post the §5 marker below
> ONLY after the round's re-park has actually executed — check `task.py
> view <N> --json` shows `status: awaiting_promotion` first. If the
> status is still `followups_running`, the re-park was skipped: run step
> 3's `set-status <N> awaiting_promotion` + step 4's
> `epm:same-issue-followup-run v1` completion marker NOW, then post the
> marker. Posting the exit-site marker while still at `followups_running`
> and exiting is the #533 freeze shape — the session died there and the
> task stranded. (`autonomous_session_watch.py` now backstops
> this with a round-complete auto re-park, but the backstop is recovery,
> not the design.)

Post the §5 marker (the EXIT site is the tail of step `9a-bis`; the
candidate landing step on resume is `10` (`completion_audit`), looked up
from `workflow.yaml § steps`):
```bash
uv run python scripts/post_step_completed.py --issue <N> --step 9a-bis \
  --exit-kind parked --notes "awaiting clean-result promotion"
```
EXIT. The user reviews the clean-result at their own pace and manually
picks a verdict. **Awaiting promotion is a user-only state — no agent
or automation may move a task out of it.** The `task.py promote`
command refuses if `classification != 'pending'`.

**On re-invocation at `awaiting_promotion`:**

1. Check the `classification` field in `body.md` frontmatter (set by
   `task.py promote`).
2. If `classification != 'pending'` -> advance to Step 10 (auto-complete).
3. If `classification == 'pending'` -> show the task path, post the §5
   marker:
   ```bash
   uv run python scripts/post_step_completed.py --issue <N> --step 10 \
     --exit-kind parked --notes "clean-result classification still pending; awaiting promotion"
   ```
   and EXIT. User hasn't promoted yet.

**9c. Test-verdict gate (code-change paths only, inline)**

Only for `infra` / `batch` / `analysis` / `survey` tasks — these arrive
here directly from Step 5 PASS, having skipped Steps 6-8 (no pod, no
interpretation). The code-review gate has already approved the diff;
this step verifies the test suite still passes.

There is **no `tester` agent**. The skill itself runs the project's test
suite directly and posts an `epm:test-verdict` event with the result.

1. Unit tests — DEFAULT scope `touched` (workflow-invariant + touched-file
   subset). The full ~5800-test suite has no xdist parallelism and is
   harness-/earlyoom-killed in sparse worktrees (#665/#736), so do NOT run
   `pytest tests/` wholesale by default.
   a. Compute the subset FROM THE ISSUE WORKTREE — a branch-new test file
      exists ONLY there until the Step 10d merge, and the helper diffs the
      INVOKING checkout (#851: run from the main repo root it saw an
      empty diff and silently dropped the branch's own test files from the
      gate; the helper now emits a stderr `NOTE — empty diff` in that shape —
      on a worktree-based task whose branch HAS commits ahead of the base,
      that NOTE means wrong cwd, re-run from the worktree; from a correct
      worktree with no commits ahead of the base it also fires and is then
      expected and benign).

      Pre-gate spec-freshness re-sync (#1742): AFTER the `cd "$WT"` below
      and BEFORE invoking the selector, run the Step 5a family-atomic
      spec-freshness block (§ Step 5a) ONCE from the worktree cwd. This
      is a BINDING reference — never inline a THIRD `FAMILY_OF` copy
      here (a third inlined copy would escape
      `test_step10d_family_atomicity_matches_step5a`'s drift guard). The
      Step 5a block's own `WT=$(git rev-parse --show-toplevel)`
      derivation is CORRECT at this call site (step 1a already `cd`s to
      the worktree), and its on-main skip guard makes the reference safe
      for repo-root-based tasks (no worktree ⇒ the sync no-ops). A sync
      commit here is SAFE — no SHA-bound verdict exists at 9c — and it
      must PRECEDE subset computation: the selector's three-dot diff
      then simply reflects the freshened files, whose content is main's
      own, so a freshened pin test runs against the freshened spec
      instead of failing the gate on the stale worktree copy (the #1742
      class: a main-side spec fix landing after the Step 5a sync red the
      gate round). The selector's diff base defaults to fetched
      `origin/main` (#1289: the shared root's local `main` lagged origin
      and polluted #1281's gate to 41 files with foreign touched
      files; bounded 120 s fetch — a fetch failure degrades to last-fetched
      `origin/main`, an unresolvable `origin/main` falls back loudly to local
      `main`). Pass `--base main` only to deliberately diff against the
      local ref:
      ```bash
      REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
      WT="$REPO_ROOT/.claude/worktrees/issue-<N>"   # same-issue follow-up rounds use
                                                    # their own issue-<N>-<suffix> worktree
      cd "$WT" || { echo "FATAL: cd to issue worktree failed" >&2; exit 1; }
      # Pre-gate spec-freshness re-sync (#1742): run the Step 5a
      # family-atomic block (§ Step 5a) HERE, before the selector —
      # reference § Step 5a, never a third inlined FAMILY_OF copy.
      uv run python scripts/select_step9c_tests.py   # base defaults to FETCHED origin/main (#1289)
      ```
      It prints the exact gate command —
      `timeout --kill-after=60s <T>s uv run pytest <files> --continue-on-collection-errors -v --tb=short`,
      `<T>` sized deterministically from the selection
      (`recommended_timeout_s()`: 120s base + 30s/file + a 2400s surcharge when
      `tests/test_workflow_lint.py` is selected, which alone measures median
      ~13 min / max ~30 min, #1646) — plus
      stderr diagnostics: a one-line work-root + branch
      provenance breadcrumb on every run, a `recommended-timeout-s=<T>`
      sizing line, any `untested touched file: <path>` WARN lines, and the
      empty-diff NOTE described above. (A code-change task with NO worktree
      runs both from the repo root; the empty-diff NOTE is then expected and
      benign.)
   b. Run the printed command as a BACKGROUND Bash invocation
      (`run_in_background=true`) from the SAME worktree cwd (paths are
      repo-relative), with the junit flags + log/rc-file tail appended and a
      pre-run `rm -f` of all three gate files (a killed run must leave NO
      junit — pytest writes it only at session exit; a stale file from a
      prior round must never be re-read). BACKGROUND IS REQUIRED, NOT
      OPTIONAL: the selection always contains the ~61-file
      workflow-invariant set incl. `tests/test_workflow_lint.py` (median
      ~13 min alone, max ~30 min; whole gate median ~18 min, max ~38 min of
      test time plus collection overhead — #1646), so the
      gate can NEVER fit the 600s foreground Bash tool cap. The
      crash-fix-rounds ~510s foreground `timeout` bound
      (`.claude/rules/crash-fix-rounds.md` § Kill-before-relaunch) applies to
      FOREGROUND smokes ONLY — wrapping this gate in any ≤600s bound is the
      #991/#996/#906 kill class (exit 143 at 480-540s). The ONLY wedge bound
      is the selector-printed `timeout --kill-after=60s <T>s` prefix.

      **Single-flight probe (#1606) — run before EVERY gate (re)launch, Step
      9c AND Step 10d alike.** Probe for a live gate with the self-excluding
      helper:
      `uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --issue <N>`
      (exit 0 = CLEAR — safe to launch; exit 3 = a live FOREIGN match, one
      `pid<TAB>args` line each; exit semantics deliberately INVERTED vs
      pgrep so `probe && launch` composes). The helper scans
      `/proc/*/cmdline` for the internally derived pattern
      `step9c-junit-issue-<N>\.xml` and MECHANICALLY excludes its own pid +
      full ancestor chain, so it cannot match its own wrapper even when
      folded into a launch call whose argv carries the unbracketed junit
      path — the #1742 re-hit of the documented #1606 trap, whose observed
      consequence was a silent exit-0 skip of the leg that the
      harness reports as successful completion (a compare leg printed
      `GATE STILL RUNNING;
      skip compare`, then `FATAL: compare rc file missing`, and exited 0 — a
      false DONE in the #825 empty-dir false-DONE class). A separate
      FOREGROUND call stays PREFERRED as defense-in-depth (it keeps the
      probe verdict readable on its own), but it is no longer load-bearing:
      the pid exclusion — not placement discipline — is what prevents the
      self-match. The junit path rides the argv of
      the gate pytest, its `timeout` wrapper, its enclosing background
      shell, AND the 1d compare, so the probe is exact-ISSUE-scoped — a
      sibling issue's gate never matches, and a recycled pid cannot
      false-match because the probe matches live argv, not pid identity. A
      LIVE result (exit 3) = a gate for THIS issue is STILL RUNNING: do NOT
      launch — the `rm -f` preamble below would clobber the live run's
      junit/rc mid-run (#1606: a second gate launched into a live one left 4
      live gate pids and fired two fail-CLOSED verdict blocks, ~12 min
      churn). Default to WAITING for exit — the harness notification on your
      own background call, or (bg handle lost, e.g. after a respawn) a
      Monitor until-loop keyed on the probe, elapsed-capped for consistency
      with the § Long-phase heartbeat 45-min segmentation (the `--issue`
      form ONLY in until-loops — its derived regex is fixed and valid, so
      the loop can never spin on a helper usage error):
      `until uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --issue <N> >/dev/null || [ $SECONDS -gt 2700 ]; do sleep 15; done`
      (a still-LIVE probe at the cap re-arms a fresh segment) — then
      read the result via the normal completion-read; the in-block
      `timeout` wedge bounds guarantee the wait terminates. Kill FIRST only
      on the recovery arms' TRIGGER — the run's completion signal fired yet
      the rc/verdict file is missing, or the run is wedged past its bound
      (NOT merely "its launching call is dead": post-respawn the launching
      call is dead while the gate is healthy and will still write its rc) —
      per crash-fix-rounds § Kill-before-relaunch, and re-probe CLEAR
      (exit 0) before launching. Corollary
      (CLAUDE.md § Monitoring re-run discipline, restated here because
      #1606's improvised Monitor violated it): any improvised gate wait keys
      "done" on **process exit** — the probe exiting 0 (CLEAR) —
      NEVER on rc/verdict-file existence alone (the rc file is written
      only at process exit; an existence-keyed Monitor false-fired
      "done" twice mid-run in #1606). The same probe-then-launch rule governs 1c, 1d
      (compare), both Step 10d gate blocks, and the Step 9a-ter § Inline
      payload lint gate — each names its site probe invocation in place.

      **Gate-fleet arbitration (#1962) — after the per-issue probe, before
      the launch, at every hooked gate site (9c 1b/1c/1d, both Step 10d
      gate blocks, the Step 9a-ter inline payload lint gate; this paragraph
      is the canonical text the other sites reference).** The single-flight
      probe serializes THIS issue's gates only; concurrent FOREIGN-issue
      gate trees are what stretch the ~9-12 min idle gate wall to 30-40 min
      and feed the earlyoom/timeout kill regime. Run
      `uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --fleet --exclude-issue <N>`
      — exit 0 = under the cap, launch; exit 3 = >= `EPM_GATE_FLEET_MAX`
      (default 2) FOREIGN issues have live gate trees (one
      `issue=<M><TAB>pids=<k><TAB><sample argv>` line each; the ledger
      refresh counts as pseudo-issue `refresh`). On exit 3, QUEUE via the
      sanctioned bounded Monitor until-loop — the `--fleet` form's internal
      signature union is FIXED and valid, so the loop can never spin on
      exit 2:
      `until uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --fleet --exclude-issue <N> >/dev/null || [ $SECONDS -gt 2700 ]; do sleep 60; done`
      — then launch ANYWAY, printing one line
      `[gate-fleet] cap-expired after 45 min — launching over cap` into the
      gate transcript when the cap expired (FAIL-OPEN: the arbitration is a
      politeness queue, never a hard block — the per-leg `timeout` wedge
      bounds remain the wedge protection, and a wedged foreign gate is
      bounded by its own leg timeouts). Record the wait outcome (launched
      immediately / waited <n>s / cap-expired) in the gate transcript.
      Accepted residual: two waiters can both observe a freed slot and
      launch together — a brief overshoot back to the unarbitrated status
      quo (the probe is read-only; no lock), and a foreign session's own
      probe / `rm -f` wrapper argv can transiently over-count — at worst
      one extra 60 s wait, the fail-safe direction.
      ```bash
      # Shell state does NOT persist across Bash calls — hard-guard the cd
      # INSIDE this same background call (never rely on a prior call's cwd;
      # a silent cd failure must never run the gate in the wrong dir):
      cd "$WT" || { echo "FATAL: cd to issue worktree failed" >&2; exit 1; }
      # earlyoom-protect the gate (#1045; FAIL-OPEN — never block the gate on a choom failure):
      # pytest is in this VM's earlyoom --prefer regex (+300 badness), so a gate run is the
      # designated victim under fleet memory pressure (#906 killed twice; #995 at ~42%).
      # Self-choom the gate shell: every child forked after this line (the timeout wrapper,
      # pytest + its subprocesses) inherits adj=-600.
      sudo -n choom -n -600 -p $$ >/dev/null 2>&1 && GATE_CHOOM=ok \
        || { GATE_CHOOM=failed; echo "[warn] choom failed — gate pytest is earlyoom-UNPROTECTED (choom=failed)" >&2; }
      echo "[step9c] gate earlyoom protection choom=$GATE_CHOOM"
      # Route gate fixture temp writes onto the data disk (#1408; #1363: / at 100% killed the
      # gate). Short --basetemp keeps AF_UNIX socket paths under the 108-byte cap. Falls back
      # silently (no TMPDIR export) on pods/GCE with no data disk.
      S9C_TMPROOT=$(uv run python scripts/step9c_baseline.py tmproot 2>/dev/null || true)
      if [ -n "$S9C_TMPROOT" ]; then
        export TMPDIR="$S9C_TMPROOT"
        S9C_BASETEMP=$(mktemp -d "$S9C_TMPROOT/bt-XXXXXX")
      fi
      rm -f /tmp/step9c-junit-issue-<N>.xml /tmp/step9c-rc-issue-<N> \
            /tmp/step9c-pytest-issue-<N>.log   # MANDATORY before EVERY gate pytest invocation
      # ONE background Bash call (run_in_background=true) — the selector-printed
      # command verbatim, with the junit + log + rc-file tail appended:
      timeout --kill-after=60s <T>s uv run pytest <files> --continue-on-collection-errors -v --tb=short \
        --junitxml=/tmp/step9c-junit-issue-<N>.xml -o junit_family=xunit1 \
        ${S9C_BASETEMP:+--basetemp=$S9C_BASETEMP/p} \
        > /tmp/step9c-pytest-issue-<N>.log 2>&1; echo $? > /tmp/step9c-rc-issue-<N>
      [ -n "${S9C_BASETEMP:-}" ] && rm -rf "$S9C_BASETEMP" || true
      ```
      When the background call completes (the harness notifies), read the
      verdict in a fresh foreground call — the rc FILE replaces the former
      in-shell `PYTEST_RC=$?` capture (shell variables do not survive across
      Bash calls). A MISSING rc file means the background run died before
      pytest exited (tool kill / watcher force-stop, #833): treat as FAIL,
      never a silent PASS, and apply crash-fix-rounds § Kill-before-relaunch
      (probe `pgrep -af '[p]ytest.*step9c-junit-issue-<N>'` — the junit path
      makes the probe exact-invocation-scoped; exit-code trap: raw pgrep
      exits 0 on a LIVE match — INVERTED vs `step9c_baseline.py probe`,
      whose 0 = clear — this kill-arm keeps pgrep because it wants the pid
      list to kill) before any re-run:
      ```bash
      if [ ! -f /tmp/step9c-rc-issue-<N> ]; then
        echo "FATAL: gate rc file missing — the background run died before pytest exited. Kill-before-relaunch, then re-run the gate; NEVER record PASS." >&2
      else
        PYTEST_RC=$(cat /tmp/step9c-rc-issue-<N>)
        tail -30 /tmp/step9c-pytest-issue-<N>.log
        # exit 0 + "no tests ran" (or "collected 0 items") is NOT a PASS:
        if grep -qiE 'no tests ran|collected 0 items' /tmp/step9c-pytest-issue-<N>.log; then
          echo "FATAL: pytest collected 0 tests — test-verdict gate did NOT run. Treating as FAIL." >&2
          # -> post epm:test-verdict v1 as FAIL; do NOT record PASS on exit 0.
        fi
      fi
      ```
      Record pass/fail + ALL selector stderr lines (the provenance
      breadcrumb, the `recommended-timeout-s=<T>` sizing line, the NOTE if
      any, and any WARN lines). The two anti-silent-pass guards above are
      LOAD-BEARING — a `no tests ran` outcome (pytest exit 0 with zero
      collected tests) is a **FAIL, never a PASS**: it is the signature of a
      failed `cd` that ran pytest in a directory with no tests (#745:
      the gate reported PASS on
      `no tests ran ... pytest exit: 0` and was silently skipped).
      Collection errors no longer abort the run (#1746):
      `--continue-on-collection-errors` lets the surviving files run, pytest
      exits rc=1, and each broken file's junit `<error>` row classifies via
      the step-1d compare like any other failure (a KNOWN main-side
      collection-red file strips as pre-existing; a branch-introduced one
      blocks as NEW) — so the `collected 0 items` FATAL grep above now fires
      only when EVERY selected file is collection-red (the workflow-invariant
      set rides along, making that practically unreachable).

      **Recipe exit-code hygiene (every gate call — and every improvised
      monitoring one-liner):** the Bash tool reports the exit code of the
      LAST command in the call, and an `Exit 1` from a trailing INFORMATIONAL
      command is indistinguishable in the transcript from a gate failure. Any
      trailing command that legitimately returns non-zero without meaning
      failure — a display/filter `grep` that may match nothing (#969: a
      healthy gate read as a false Exit 1), a bare `[ -z "$VAR" ]` /
      `[ -s <file> ]` test (#928: a trailing `[ -z "$GIST_URL" ] && ...`
      variant reported Exit 1 on success), a `tail`/`cat` on a possibly-
      absent file — MUST be if-formed (`if grep -q ...; then ...; fi`) or
      given an explicit `|| true`. The verdict-bearing rcs are NEVER the raw
      call exit code: PYTEST_RC lives in `/tmp/step9c-rc-issue-<N>` and
      COMPARE_RC in `/tmp/step9c-compare-issue-<N>.rc` — read those, not
      the tool's Exit line.
   c. Scope override: if the plan-body frontmatter has `test_scope: full` OR a
      `## Test scope` H2 names `full`, run the FULL suite instead — from the
      SAME issue-worktree cwd, in the SAME background + rc-file pattern as 1b
      — including 1b's **Single-flight probe (#1606)** (the self-excluding
      helper, `--issue <N>` form) —
      (a 60m run is 6x the foreground tool cap):
      ```bash
      cd "$WT" || { echo "FATAL: cd to issue worktree failed" >&2; exit 1; }
      # earlyoom-protect the gate (#1045; fail-open — see the 1b preamble): self-choom, children inherit.
      sudo -n choom -n -600 -p $$ >/dev/null 2>&1 && GATE_CHOOM=ok \
        || { GATE_CHOOM=failed; echo "[warn] choom failed — gate pytest is earlyoom-UNPROTECTED (choom=failed)" >&2; }
      echo "[step9c] gate earlyoom protection choom=$GATE_CHOOM"
      # Route gate fixture temp writes onto the data disk (#1408; #1363: / at 100% killed the
      # gate). Short --basetemp keeps AF_UNIX socket paths under the 108-byte cap. Falls back
      # silently (no TMPDIR export) on pods/GCE with no data disk.
      S9C_TMPROOT=$(uv run python scripts/step9c_baseline.py tmproot 2>/dev/null || true)
      if [ -n "$S9C_TMPROOT" ]; then
        export TMPDIR="$S9C_TMPROOT"
        S9C_BASETEMP=$(mktemp -d "$S9C_TMPROOT/bt-XXXXXX")
      fi
      rm -f /tmp/step9c-junit-issue-<N>.xml /tmp/step9c-rc-issue-<N> \
            /tmp/step9c-pytest-issue-<N>.log
      # ONE background Bash call (run_in_background=true):
      timeout --kill-after=60s 60m uv run pytest tests/ -q --continue-on-collection-errors \
        --junitxml=/tmp/step9c-junit-issue-<N>.xml -o junit_family=xunit1 \
        ${S9C_BASETEMP:+--basetemp=$S9C_BASETEMP/p} \
        > /tmp/step9c-pytest-issue-<N>.log 2>&1; echo $? > /tmp/step9c-rc-issue-<N>
      [ -n "${S9C_BASETEMP:-}" ] && rm -rf "$S9C_BASETEMP" || true
      ```
      (NO `-x` / `--maxfail` — with the step-1d compare deciding the verdict,
      an early-exit on the first known-red main failure would leave the rest
      of the suite unexecuted and let compare PASS a truncated run; the 60m
      timeout still bounds it.) The rc file is written by the SAME background
      command immediately after pytest exits (1b touched scope and this
      override alike) — step 1d's compare consumes
      `--pytest-rc "$PYTEST_RC"`, re-reading `/tmp/step9c-rc-issue-<N>`
      INSIDE its own background call (shell variables do not survive across
      Bash calls); a missing rc file takes 1b's FAIL path (an
      unset or stale rc would break compare's rc-not-in-{0,1} ->
      indeterminate guard). On timeout/kill (`timeout`'s rc 124 lands in the
      rc file, so compare exits 2), capture
      `tail -50 /tmp/step9c-pytest-issue-<N>.log` so the stall surfaces
      actionable evidence (the #665/#736 regression — keep it visible, never
      a silent kill). Default scope is `touched`.

      **Gate earlyoom protection (#1045).** The self-choom preamble in the
      1b/1c blocks is the same-call sibling of § "Detached VM-side long
      compute phases": `oom_score_adj` inherits across fork/exec
      (probe-verified), so choom-ing the gate shell BEFORE pytest launches
      protects the whole gate tree — the `timeout` wrapper, pytest, and its
      subprocesses inside the background call — with zero change to the
      rc-file/junitxml contract (#1046: the gate is a background invocation;
      `PYTEST_RC` travels via `/tmp/step9c-rc-issue-<N>`, not shell state).
      Step 1d compare — including its pristine single-file oracle runs
      (600–4950s each, #1129/#1646) — runs as its OWN background + rc-file call
      with the SAME fail-open self-choom preamble (see step 1d); only the
      1d ledger-refresh kick keeps the post-hoc session-sweep form (it
      launches detached BEFORE a choom can be applied). FAIL-OPEN: a
      choom failure warns, records `choom=failed`, and the gate proceeds
      unprotected — a gate is NEVER blocked by a choom failure, and
      `choom=ok` re-orders earlyoom's victim selection (−600, not −1000: the
      gate stays killable if it is itself the runaway consumer). The Bash
      tool spawns a fresh shell per call, so the adjustment dies with the
      call; no reset needed (in a long-lived manual shell, reset with
      `sudo -n choom -n 0 -p $$`). Calibration: −600 buys victim
      RE-ORDERING, not survival — net ~400 display points below an
      equal-size unprotected python neighbor (the `--prefer` +300 applies
      regardless of adj) but only ~100 below a non-python neighbor, and at
      fleet-wide adoption protected work competes with protected work again;
      say "re-orders victim selection" / "stops being the default designated
      victim", never "prevents kills".
   d. Classify failures against the known-red-on-main baseline ledger —
      mechanical (`scripts/step9c_baseline.py compare`), never prose
      arithmetic (#1022). Runs AFTER the final pytest
      invocation of step 1 (touched scope, or the 1c full-scope override —
      compare gates the junit of whichever actually ran) AND after 1b's
      foreground verdict read. Run compare as a BACKGROUND Bash invocation
      (`run_in_background=true`) from the SAME worktree cwd, in the SAME
      background + rc-file pattern as 1b. BACKGROUND IS REQUIRED, NOT
      OPTIONAL: `--run-pristine` (always passed here) may run up to
      `--max-pristine-files` (5) single-file pristine oracle runs, each
      bounded by `derive_pristine_timeout_s` at 600–4950s (#1129/#1646:
      tests/test_workflow_lint.py alone derives 4950s), so a healthy
      compare can NEVER be guaranteed to fit the 600s foreground Bash tool
      cap — a foreground call converts a classifiable in-process exit 2
      into a tool-layer kill with COMPARE_OUT lost (#1129/#1098). Compare
      stays a SEPARATE background call, NOT folded into the 1b gate call:
      1b's foreground verdict read and the zero-collected guard run
      between them, and a folded call would burn up to ~2 h of
      pristine runs on a run those guards fail in seconds.

      **Single-flight probe (#1606)** first, per the 1b statement:
      `uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --issue <N>`
      (self-/ancestor-excluding; exit 0 = clear) — exit 3, a live foreign
      match (the 1b/1c pytest still running, or a prior compare still
      consuming the junit), means WAIT/reap per 1b BEFORE this launch: the
      compare-triplet `rm -f` below would clobber a live compare's outputs,
      and compare must never read a junit a live pytest is still writing.
      Then the **Gate-fleet arbitration (#1962)** probe, per the 1b
      canonical paragraph (compare's pristine pytest runs are the same
      weight class):
      `uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --fleet --exclude-issue <N>`
      — exit 3 ⇒ bounded queue (sleep 60, elapsed cap 2700 s), then launch
      anyway with the `[gate-fleet]` cap-expired line (fail-open).
      ```bash
      cd "$WT" || { echo "FATAL: cd to issue worktree failed" >&2; exit 1; }
      # earlyoom-protect the compare (#1045; FAIL-OPEN — never block the verdict on
      # a choom failure): its pristine pytest children are the same earlyoom-preferred
      # long python work as the 1b gate; oom_score_adj inherits across fork/exec
      # (probe-verified; start_new_session does NOT reset it), so self-choom BEFORE
      # launch covers the whole compare tree incl. the pristine pytest children.
      sudo -n choom -n -600 -p $$ >/dev/null 2>&1 && COMPARE_CHOOM=ok \
        || { COMPARE_CHOOM=failed; echo "[warn] choom failed — compare pristine runs are earlyoom-UNPROTECTED (choom=failed)" >&2; }
      echo "[step9c] compare earlyoom protection choom=$COMPARE_CHOOM"
      # MANDATORY stale-file rm — the compare triplet ONLY (NEVER 1b's
      # junit/rc/log files: compare consumes them):
      rm -f /tmp/step9c-compare-issue-<N>.json /tmp/step9c-compare-issue-<N>.rc \
            /tmp/step9c-compare-issue-<N>.err
      # Re-read the 1b rc IN-CALL (shell variables do not survive across Bash
      # calls); a missing 1b rc file already took 1b's FAIL path — never invoke
      # compare against it:
      [ -f /tmp/step9c-rc-issue-<N> ] || { echo "FATAL: 1b rc file missing — apply 1b's FAIL path; compare not run" >&2; exit 1; }
      PYTEST_RC=$(cat /tmp/step9c-rc-issue-<N>)
      # Wedge bound 10800s ≥ the structural ceiling of compare's own in-process
      # bounds: the 5 pristine files are DISTINCT and SLOW_TESTS has one entry,
      # so ceiling = 4950s (workflow-lint derived) + 4 × 600s floor + 120s
      # scratch + ruff/parse overhead ≈ 7500s; 10800s keeps ~1.4x margin and
      # only ever fires on a genuine wedge (#1129 generous bias, figures #1646;
      # re-derive if SLOW_TESTS gains entries/values or max-pristine-files changes):
      timeout --kill-after=60s 10800s uv run python scripts/step9c_baseline.py compare \
        --junitxml /tmp/step9c-junit-issue-<N>.xml --pytest-rc "$PYTEST_RC" \
        --run-pristine --json \
        > /tmp/step9c-compare-issue-<N>.json 2> /tmp/step9c-compare-issue-<N>.err
      echo $? > /tmp/step9c-compare-issue-<N>.rc
      ```
      (stdout and stderr are SEPARATED — unlike 1b's merged log — because
      stdout is the JSON payload the verdict parses; stderr carries WARN /
      timeout-kill diagnostics.) When the background call completes (the
      harness notifies), read the verdict in a fresh foreground call from
      the FILES. A MISSING rc file means the background compare died before
      exiting (tool kill / watcher force-stop): treat as FAIL/indeterminate,
      never a silent PASS, and apply crash-fix-rounds
      § Kill-before-relaunch (probe `pgrep -af 'step9c_baseline[.]py compare'` — exit-code trap: raw pgrep exits 0 on a LIVE match — INVERTED vs `step9c_baseline.py probe`, whose 0 = clear — this kill-arm keeps pgrep because it wants the pid list)
      before any re-run:
      ```bash
      if [ ! -f /tmp/step9c-compare-issue-<N>.rc ]; then
        echo "FATAL: compare rc file missing — the background compare died before exiting. Kill-before-relaunch, then re-run step 1d; NEVER record PASS." >&2
      else
        COMPARE_RC=$(cat /tmp/step9c-compare-issue-<N>.rc)
        COMPARE_OUT=$(cat /tmp/step9c-compare-issue-<N>.json)
        echo "$COMPARE_OUT"
        if [ -s /tmp/step9c-compare-issue-<N>.err ]; then tail -20 /tmp/step9c-compare-issue-<N>.err; fi
      fi
      ```
      `COMPARE_RC` ∉ {0, 1, 2} (124/137 = wedge-timeout / kill) or an
      empty / unparseable JSON file is INDETERMINATE — FAIL, never PASS
      (#665/#736: capture the `.err` tail so the stall surfaces actionable
      evidence, never a silent kill).
      The COMPARE verdict — not the raw PYTEST_RC — decides pass/fail for
      steps 1–2:
      * `COMPARE_RC=0` → no NEW test failures and no lint regression; failures
        listed in `stripped` are pre-existing on main and do NOT block (the
        round may PASS steps 1–2 with PYTEST_RC=1).
      * `COMPARE_RC=1` → NEW failure(s) the branch introduced and/or a lint
        regression (the JSON names each). FAIL.
      * `COMPARE_RC=2` → indeterminate (PYTEST_RC ∉ {0,1} — aborted/interrupted
        run; missing/empty junitxml; suite crash; unusable ledger;
        systemic main breakage; or a scratch-INELIGIBLE dirty oracle. The
        pristine oracle is BY DEFAULT a detached sparse scratch worktree at
        main HEAD (#1408 — clean or dirty root alike; JSON
        "pristine_oracle": "scratch-worktree"; a scratch creation/probe
        failure on a CLEAN root degrades to the trustworthy root oracle with
        a WARN + `"scratch_degraded": true`, never exit 2), so the
        dirty-refusal enumeration shrinks to: residual venv dirt
        (`pyproject.toml`/`uv.lock` or out-of-package `src/` — dirty
        in-package `src/` is neutralized via the probe-verified
        `PYTHONPATH=<scratch>/src` shadow, `"scratch_src_shadow": true`,
        #1251), a non-sparse work root, a scan-set node outside the
        file-anchored allowlist (step9c_baseline.py
        FILE_ANCHORED_SCAN_TESTS, #1337), or scratch creation/probe failure
        on a DIRTY root). FAIL — never PASS on indeterminate.
        On a residual-dirt exit 2, do NOT improvise multi-hour clean-root
        polls (the #1317 anti-pattern): one bounded re-check after ~10-15
        min, then treat as gate FAIL and surface per the existing FAIL path.
        COMPARE_OUT is valid JSON on EVERY exit path under --json (exit-2
        payloads carry "indeterminate": true — an exit-2 payload's empty
        new/stripped arrays are NOT a clean verdict).
      The two step-1b guards run BEFORE compare and are UNCHANGED: the cd
      hard-guard and the `no tests ran` FAIL guard (zero collected is a FAIL
      regardless of compare's exit).
      If the compare JSON has `"stale": true`, kick a DETACHED background
      ledger refresh so the next session gets a fresh baseline — do NOT block
      this verdict on it:
      ```bash
      REFRESH_PID=$(bash -c 'cd "$1" || exit 1; setsid nohup timeout --kill-after=60s 4650s \
        env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
        uv run python scripts/step9c_baseline.py refresh \
        >> "$1/logs/step9c_baseline_refresh.log" 2>&1 < /dev/null & echo $!' _ "$REPO_ROOT")
      # earlyoom-protect the refresh (#1045; fail-open): sweep its session; the refresh's own
      # start_new_session pytest child (spawned >=~1s later, after lock + git-root resolution + selector/venv resolution + uv-run startup) inherits adj.
      ps -p "$REFRESH_PID" -o args=   # verify the pid is the workload (canonical form); on mismatch recover via pgrep -f 'step9c_baseline[.]py refresh'; a lock-held instant exit makes this benignly fail (choom=failed below)
      bash -o pipefail -c 'pgrep -s "$1" | xargs -rn1 sudo -n choom -n -600 -p' _ "$REFRESH_PID" >/dev/null \
        && echo "[step9c] ledger refresh detached pid=$REFRESH_PID log=$REPO_ROOT/logs/step9c_baseline_refresh.log choom=ok" \
        || echo "[step9c] ledger refresh detached pid=$REFRESH_PID log=$REPO_ROOT/logs/step9c_baseline_refresh.log choom=failed"
      ```
   e. Urgent-park duty on stripped workflow-invariant red (#1713/#1742) — a
      mechanical, trigger-keyed duty inside this step; auto-continue (never a
      gate/pause). When COMPARE_OUT's `urgent_park_required` list is
      non-empty — or an `URGENT-PARK-REQUIRED:` line appears in the compare
      stderr `.err` tail — then IN THE SAME TURN as posting
      `epm:test-verdict`, for EACH listed `<file>::<name>` node id:
      (i) bounded dedup grep for an already-routable candidate:
      ```bash
      grep -rl -- 'failing_test: <node id>' tasks/*/*/events.jsonl \
        .claude/cache/workflow-fix-events.jsonl 2>/dev/null
      ```
      a hit means a routable candidate already exists — record the pointer
      (the matching events path), do NOT re-emit; (ii) on no hit, emit the
      `<!-- workflow-fix-candidate v1 -->` block in the session's
      return/chat text carrying the #1681 urgent grammar — `urgency:
      main-red` + `failing_test: <the ONE pytest node id>` +
      `wf_fix: true|false` — routed/parked by the standard workflow-fix
      protocol (under the recursion guard the PARK is itself the routable
      record the watcher's urgent-park router consumes); (iii) record the
      disposition in the `epm:test-verdict` note: `urgent_park: emitted
      <id>` | `urgent_park: existing <events path>` (omit the line when
      `urgent_park_required` is empty). This mechanical trigger covers the
      selector's WORKFLOW_INVARIANT subset ONLY; the Step 10d broad-glob
      urgent-park duty (#1713 — ANY workflow-surface pre-existing red) is
      UNCHANGED for non-invariant reds: "no 1e trigger fired" never waives
      that duty.
2. Lint: covered by step 1d `compare` — repo-wide `ruff check` /
   `ruff format --check` are diffed against the LIVE main-root baseline
   (only an INCREASE fails; main carries 2000+ pre-existing ruff errors,
   #1022), and the branch's touched `*.py` files must additionally be
   ruff-clean + format-clean in absolute terms. Do NOT run bare
   `uv run ruff check . && uv run ruff format --check .` as a verdict gate —
   it always fails on pre-existing main red and re-derives what the ledger
   already answers.
3. Integration tests (conditional, if diff touches train/eval/orchestrate)
4. Coverage gap report (flags, does not auto-generate)

The `epm:test-verdict v1` marker note records: scope used (`touched`/`full`),
the files run, the gate timeout bound used (the selector's
recommended-timeout-s), pass/fail counts, and ALL selector stderr diagnostics — the
work-root + branch provenance breadcrumb, any `NOTE — empty diff` line, and
any untested-touched-file WARNs (so the orchestrator surfaces wrong-cwd runs
and coverage gaps — never silently skipped), and the compare classification
JSON (new vs known-red-stripped failures with any scan-test / diff-linked
masking WARNs, the ruff delta vs the live main baseline, the ledger main_sha
+ age + stale flag, and any dirty-code-path flags), and the step-1e
urgent-park disposition line(s) (`urgent_park: emitted <id>` |
`urgent_park: existing <events path>`; omitted when `urgent_park_required`
is empty — #1742), and
the gate + compare earlyoom-protection state — COPY the `[step9c] gate
earlyoom protection choom=…` and `[step9c] compare earlyoom protection
choom=…` breadcrumb lines from the gate and compare calls' transcripts (plus
the 1d refresh `pid= log= choom=` breadcrumb line when a refresh was kicked);
never infer `choom=ok` from the absence of a warn line. A zero-collected /
`no tests ran` outcome is recorded as FAIL (never PASS on exit 0) per step
1b's guard.

Post `epm:test-verdict v1`. PASS = steps 1–2 pass via compare exit 0 (with
`--pytest-rc` folded) AND neither step-1b guard fired, AND steps 3
(conditional integration tests) and 4 (coverage gap report) completed per
their existing rules -> Step 10. FAIL (`epm:test-verdict` FAIL count < 3) ->
stay in `reviewing`, re-spawn implementer. FAIL (`epm:test-verdict` FAIL
count >= 3) -> run
CRON-TEARDOWN (§ CRON-TEARDOWN procedure — both legs incl. stray one-shot
`/issue <N>` wakeups; no-ops for a code-change task
that never armed one), then status to
`blocked`. Fire `PushNotification({"message": f"#{N} BLOCKED: tests
FAIL after 3 rounds — open it"[:200], "status": "proactive"})` before
setting status (soft-fail).
