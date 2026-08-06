---
description: Pod-side dispatcher result-reporting contract (sentinel files, poll_pipeline.py drain, epm:results payload, sentinel READ-BACK tolerance under the .processed drain-rename #1311) + pid-file launch contract (rewrite on EVERY (re)launch, #813) + relaunch-descope record & handle-sidecar currency (#1689) + full-stdio-detach on ssh-remote (re)launch (#1768) + result-push verification (#1205) + legacy preflight gates (#829)
paths:
  - "scripts/*dispatch*"
  - "scripts/poll_pipeline.py"
  - "scripts/*_dispatch.sh"
---

# Pod-side result-reporting + preflight gates (relocated from experiment-implementer.md, #829)

### Pod-side result-reporting contract (`poll_pipeline.py`)

CLAUDE.md "Pod-side code NEVER shells out to `scripts/task.py`" mandates the
sentinel-file channel. Any pod-side dispatcher you write (anything that gets
launched on the pod by `experimenter` and is expected to terminate cleanly +
hand results back to the orchestrator) MUST conform to the orchestrator's
poll loop or its clean completion will read as `dead` / its end-of-run
marker will be silently skipped. Three requirements, no exceptions:

1. **`[phase=...]` log lines, terminating in `[phase=done]` on graceful
   completion.** `poll_pipeline.py` parses `PHASE_RE = re.compile(r"\[phase=
   ([a-z0-9_]+)")` from the tail of the pod-side log (digits are part of the
   token, so numbered phase names like `p0_render` parse fully); `poll_once`
   declares
   `status="done"` ONLY when the most recent matching line is
   `[phase=done]`. A clean exit without that terminal line decays to
   `status="dead"` (PID gone, no `done` marker), which the orchestrator
   treats as a crash and which suppresses the auto-post of `epm:results`.
   Emit at least one `[phase=<name>]` per logical phase AND an explicit
   `[phase=done]` immediately before your normal exit path (after the
   final sentinel write — see (2)). **The `[phase=done]` token in the MAIN
   dispatcher log is RESERVED for that single terminal line:** per-cell /
   subprocess completion echoes that flow into the main log must NOT
   carry it — word them without the phase tag (`eval cell <X> complete`,
   never `[phase=done] eval cell <X> complete`). The poller cannot
   textually distinguish mid-run noise from a legitimate suffixed
   terminal line and only survives it via pid/sentinel corroboration
   (#545: a per-cell `[phase=done]` echo produced a false `status=done`
   while GPUs were at 85%). Mechanically enforced by
   `scripts/workflow_lint.py --check-phase-done-reserved` (no-flags
   default run + the `workflow-lint-phase-done-reserved` pre-commit hook
   on any `scripts/*.sh|py` change): a `[phase=done]` emission in a phase
   script invoked non-redirected by a `scripts/**/*.sh` dispatcher FAILs;
   legacy edges are frozen in `PHASE_DONE_EDGE_LEGACY_ALLOWLIST`.

2. **End-of-run sentinel with poll_pipeline's required keys.** Write the
   final results sentinel to `/workspace/logs/issue-<N>-<kind_slug>-
   <epoch_seconds>.json` (`kind_slug` = the marker kind with `:` → `_`,
   e.g. `epm_results`). The JSON object MUST carry every key in
   `poll_pipeline.py::_SENTINEL_REQUIRED_KEYS`:
   - `sentinel_schema_version`: integer `1` (bump in lockstep with
     `SENTINEL_SCHEMA_VERSION_SUPPORTED` in the poller — `!= 1` is
     skipped + logged, never silently mis-parsed).
   - `kind`: full marker kind string (e.g. `"epm:results"`).
   - `version`: marker version integer. Pod-side writers hardcode `1`
     (they cannot read `events.jsonl`). **Drain-side rewrite (#1095):**
     for a real `epm:results` sentinel the VM-side drain re-derives the
     landed marker version as max+1 when the declared version collides
     at-or-below the existing max for the kind — so multi-round tasks
     keep the highest-version-per-kind resume convention (markers.md)
     without pod-side coordination. The declared version is preserved on
     the marker as `sentinel_declared_version`. Smoke/dry-run/
     phase-progress sentinels (gate NAME in
     `smoke|dryrun|dry-run|dry_run|phase`, a truthy top-level `smoke`
     field, or the `issue-<N>-smoke-results.json` / `-smoke-` filename)
     and declared versions above the existing max post verbatim; a bare
     `blocks_pipeline: false` does NOT exempt (real terminal writers set
     it). Other kinds always post verbatim. Keep hardcoding `1` — do NOT
     try to thread versions pod-side. **Smoke runs should write kind
     `epm:smoke-result`, not kind `epm:results` with a smoke flag** — a
     smoke flag nested inside `note` is invisible to the drain's
     exclusion (the drain never parses `note`), and the
     `epm:smoke-result` kind is already the house pattern
     (`write_sentinel("epm:smoke-result" if args.smoke else
     "epm:results", ...)` — the issue634/744/1073/779 writers). Two
     operational residuals: (a) a stale straggler (a resumed stopped pod
     draining an OLD run's sentinel) lands ABOVE newer rows —
     operator-recoverable via `sentinel_declared_version` + the drain
     warning; re-post the correct results as a fresh higher-version
     marker. (b) A manual high-version correction marker is shadowed by
     ANY subsequent real results drain — re-post corrections AFTER the
     final drain of a round, not before.

   The marker body goes under `note` (or the `payload` synonym).
   Recommended optional keys: `task_id`, `gate`, `blocks_pipeline`,
   `by`, `ts`. A bare `schema` key (or any other re-spelling of
   `sentinel_schema_version`) trips the `missing required keys` warning
   in `_parse_sentinel` and the sentinel is skipped without being
   renamed `.processed` — the marker never lands, the dashboard never
   updates, and the orchestrator advances without the experiment's
   results in `events.jsonl`.

3. **Read-back tolerance — the sentinel namespace is a one-way,
   write-once, VM-drained channel; never re-read your own sentinels by
   bare path.** The poller drains `/workspace/logs/issue-<N>-*.json`
   (skipping `*.processed`) on EVERY tick and renames each
   successfully-posted sentinel to `<path>.processed` (`mv -n`;
   `poll_pipeline.py::_ssh_mark_processed`; the GCP lane renames
   identically via `backends/gcp.py::_mark_sentinel_processed`; on
   SLURM, DRAC/Mila have no sentinel channel while the FELLOWS lane
   drains + renames identically via
   `slurm_monitor.drain_cluster_sentinels` (#1898) — this whole
   read-back-tolerance item, incl. the drain-rename tolerance (#1311),
   binds fellows dispatchers exactly as RunPod ones). Post each
   sentinel ONCE, never rewrite it
   in place — a rewrite whose `.processed` twin already exists is
   un-renameable under `mv -n` and re-attempted/warned every tick. A
   dispatcher that READS its own sentinels (resume predicate, per-cell
   completion check, finalize aggregation) finds them GONE from the
   bare path within ~one tick. Conform ONE way:
   - **DEFAULT (strongly preferred): keep resume/finalize state
     OUTSIDE the drained glob** — e.g. `<out_root>/<unit>/status.json`
     under the dispatcher's own output tree — because (i) read-both
     stays racy against the rename window, and (ii) the experimenter's
     pre-launch sentinel hygiene
     (`rm -f /workspace/logs/issue-<N>-*.json{,.processed}`,
     experimenter.md § Before Running step 8) wipes BOTH forms on
     every (re)launch: namespace state never survives a relaunch.
   - **Fallback: read BOTH forms, bare path FIRST, then
     `<path>.processed`** (bare-first cannot miss across the atomic
     rename; processed-first can) — completion checks only, never
     cross-relaunch resume (the hygiene wipe above).
   Nor may non-envelope state files park in the namespace: a JSON
   missing `_SENTINEL_REQUIRED_KEYS` is skipped WITHOUT rename but
   warn-spams every tick, and a `-results.json` basename carrying the
   results-payload key set is envelope-RESCUED (#899), posted, and
   renamed anyway. (#1090: per-run sentinels doubled as resume/finalize
   state; the drain renamed them mid-run → requeue races + a production
   reproducibility_card covering only 23-24 of 35 cells.) Fellows trust
   surface (#1898): `/workspace/logs` on charmander is cluster-shared
   and PERSISTENT — a prior crashed run's undrained sentinel posts late
   on the next same-issue launch (correct-by-design), and any file
   matching `issue-<N>-*.json` is drained regardless of author
   (documented trust surface, not defended against).

Rationale (#448): a dispatcher completed all cells cleanly but never
emitted `[phase=done]` and wrote its sentinel with the key `schema`
instead of `sentinel_schema_version` — a FALSE `dead` verdict, a
silently-dropped sentinel, and a hand-posted `epm:results`.

**Reproducibility card in the `epm:results` payload (training tasks).**
When your driver trains adapters / logs WandB runs, its `epm:results`
sentinel's `note` JSON MUST carry a `reproducibility_card` object
declaring per-cell `adapter_paths` (each verified under `hf_model_repo`
via `list_repo_files`) + `wandb_run_names` (with `wandb_project`), or
single-run `hf_model_path` / `wandb_run_path` — full field list:
`workflow.yaml § markers epm:results`. This applies to GCP-lane
`--workload-cmd` drivers (drained by `backend_poll.py`) exactly as to
pod-side dispatchers. A card-less sentinel that only declares
`production_provenance.<cell>.hf_adapter_subfolder` (+ top-level
`wandb_*` hints) is rescued by `verify_uploads.py`'s synthesis fallback
(`_card_from_provenance`, #599), but that synthesis is a safety net, NOT
the producer contract — emit the explicit card so the verifier's
hf_model / wandb_run rows resolve mechanically. **When training logs to
WandB, the card's `wandb_run_path` (entity/project) or `wandb_run_names`
(or a name prefix) + `wandb_project` are MANDATORY fields, not optional
extras** — a card declaring only `adapter_paths` forces entity/project
archaeology on the verifier (#608 follow-up: all 12 runs resolved at the
conventional `<entity>/issue608` project while the wandb_run row
mechanically FAILed on the declaration gap; `verify_uploads.py` now
probes the `issue<N>`-project convention as a last resort, but like the
synthesis fallback it is a safety net, NOT the contract).

**No flat `wandb_url: "n/a (...; project=...)"` shorthand on multi-cell
runs (#597).** A top-level `wandb_url: "n/a (per-cell wandb runs;
project=<P>)"` string — without an accompanying `reproducibility_card` /
`production_provenance` — declares NONE of the fields the verifier needs
to resolve the live runs; the verifier then falls back to
`api.default_entity`, which may not match the owning entity (typical
trap: HF `default_entity` is `superkaiba1`, WandB is
`thomasjiralerspong`). When per-cell runs really are the shape, emit the
full multi-cell card: `wandb_project: "<P>"` + `wandb_run_names:
[<display name per cell>]` + `wandb_entity: "<entity>"`; `wandb_url` MAY
be `n/a (per-cell wandb runs; see reproducibility_card)` or omitted.

**`wandb_entity` is STRONGLY RECOMMENDED whenever the card uses
`wandb_run_names` + `wandb_project`.** The verifier's
`check_wandb_runs_by_name` threads it straight through; when omitted it
falls back to `api.default_entity` — a safety net, NOT the contract (it
assumes the dispatcher and verifier share a WandB login + single default
entity). Read the entity off the WandB SDK at run time
(`wandb.run.entity` while the run is open, or
`wandb.Api().default_entity` after) and persist it in the card; never
hand-type it as a literal (#597 r3: an entity-default mismatch left
three live runs invisible to verification).

**Designed-halt exit codes:** a plan-registered gate refusal (a pilot
timing gate, any stop criterion) is NOT a crash — the dispatcher writes a
gate-report JSON and exits a DISTINCT rc the driver routes like its other
stop criteria, never a bare rc=1 (which the poller classifies as an
anonymous crash). Full convention + the #1415 incident:
`.claude/rules/gotchas.md` § pilot timing gates.

### Pid-file launch contract — rewrite on EVERY (re)launch (#813, #451, #521)

The pid file (`/workspace/logs/issue-<N>.pid` pod-side; the VM analogue for a
detached VM-side stage is the `pid=` field of its stage-dispatch breadcrumb,
SKILL.md § Detached VM-side long compute phases) is the poller's PRIMARY
liveness probe. This contract binds EVERY agent that launches OR relaunches a
detached workload — the experimenter, an orchestrator's crash-fix / hot-fix
relaunch, a watch-session correction — not just first launches:

1. **Every (re)launch ends with the pid file holding the NEW live workload
   pid, written in the SAME command chain as the launch itself.** Preferred
   path: relaunch through the launcher script, whose
   `echo $$ > /workspace/logs/issue-<N>.pid` overwrites the file before
   `exec`ing the workload (experimenter.md § During Execution steps 1/1b).
   When no launcher file exists, FIRST materialize one per 1g and relaunch
   through it — never a hand-typed inline relaunch chain; inside that file
   the required pid-write form is the explicit ATOMIC rewrite:
   `printf '%s\n' "$CHILD_PID" > /workspace/logs/issue-<N>.pid.tmp && mv /workspace/logs/issue-<N>.pid.tmp /workspace/logs/issue-<N>.pid`
   (tmp+rename — no window where the poller reads a truncated file; the
   launcher-internal `echo $$ >` truncate-write is accepted because it is
   a single short write completing before the workload starts). Then
   CONFIRM before posting: `cat` the pid file in a fresh SSH call and
   check it equals the pid you post in the marker. A relaunch that leaves
   a predecessor's pid in the file is a launch-contract violation. How the
   pid VALUE is obtained is governed by 1d — from the launch expression
   itself, never a post-hoc `pgrep`.
1b. **Rotate the phase log at every relaunch BEFORE re-arming any pattern-matching poller.**
   A relaunch that appends to (or re-points at) the predecessor's log
   leaves the FIRST run's failure/completion lines in the very file a
   re-armed `grep`-class watcher scans — the watcher false-fires on the
   OLD line and mis-verdicts the healthy new run (#952: a re-armed
   poller matched the first driver's old failure line in the shared
   un-rotated log). Rotate in the SAME command
   chain as the relaunch (`mv <log> <log>.pre-<ts>`, or launch into a
   fresh timestamped log and re-point the watcher), exactly as item 1
   rewrites the pid file — an un-rotated log under a pattern poller is
   the log-side twin of the stale-pid-file violation.
1c. **The 1b rotation duty covers EVERY log a crash-signature or
   completion-pattern grep may scan — including INNER per-cell logs a
   dispatcher SUBPROCESS opens in append mode — not only the
   dispatcher-level phase log.** A relaunch that rotates the outer
   phase log but leaves an inner append-mode
   `<out_root>/<cell>/train.log` in place feeds the previous crash's
   lines to any monitor grep scanning it (#1112: a health monitor
   false-fired on the PREVIOUS crash's assert lines in an un-rotated
   inner `train.log` — the inner-log twin of 1b's false-fire). Rotate
   inner logs in the same command chain where practical. Where rotation
   is IMPRACTICAL (a live subprocess owns the append-mode handle; the
   per-cell log set is enumerated dynamically), scope every
   crash-signature grep past the relaunch instead — PREFERRED: a
   byte-offset sentinel, `OFF=$(wc -c < <log>)` recorded at launch
   beside the pid file (0 when the log does not yet exist), scan only
   the suffix (`tail -c +$((OFF+1)) <log> | grep ...`), re-capturing
   OFF on any later rotation (a stale offset silently skips fresh
   bytes); FALLBACK: a launch-timestamp line filter (weaker — inner
   train.logs carry untimestamped traceback/tqdm lines it silently
   misses). A whole-file pattern match against an un-rotated
   append-mode log that may contain a predecessor's lines is NEVER a
   valid crash-signature read — the grep-side sibling of the #779
   stale-existing-file false-DONE (CLAUDE.md § Monitoring).
1d. **Pid ACQUISITION comes from the LAUNCH EXPRESSION itself — NEVER
   from a post-hoc `pgrep`.** The two sanctioned sources are (i) the
   launcher's pre-exec `echo $$` (item 1's preferred path;
   `.claude/agents/experimenter.md` § During Execution steps 1/1b) and
   (ii) `$!` of the detached child captured in the SAME command chain
   as the launch — the launcher-less pod form
   (`setsid nohup ... < /dev/null >> <log> 2>&1 & printf '%s\n' "$!" > <pid>.tmp && mv <pid>.tmp <pid>`,
   experimenter.md § During Execution) and the VM-side analogue
   (SKILL.md § Detached VM-side long compute phases, whose `bash -c`
   wrapper is the load-bearing `$!`-capture shape under job control).
   A derived read ANCHORED to a captured pid counts as
   launch-expression-derived too: experimenter.md step 3's
   parent-scoped child-walk (`pgrep -P "$WRAPPER_PID"` against the
   `$!`-captured wrapper) can only see descendants of the captured
   pid. An UNANCHORED `pgrep` run after the launch to "find" the new
   pid can capture a TRANSIENT sibling (a vanishing wrapper, a resolver
   child, a dying predecessor): #1112 populated the pid file from
   exactly such a pgrep, and the Monitor TWICE reported the healthy
   dispatcher "exited".
   `pgrep` otherwise keeps exactly two roles, neither of which
   populates the pid FILE at launch time:
   (a) the RECOVERY probe when the launch-expression pid was genuinely
   lost — bracket one pattern character
   (`pgrep -f 'issue<N>_dispatc[h]'`, the `.claude/rules/gotchas.md`
   ownership-probe self-match convention) and identity-verify with
   `ps -p "$PID" -o args=` BEFORE trusting or writing the result; the
   verified pid MAY then be written via item 1's atomic rewrite;
   (b) an ad-hoc liveness monitor's pattern-probe FALLBACK — a harness
   Monitor / until-loop watcher, NOT `poll_pipeline.py` (whose verdict
   path item 2's contract governs; #1650's marker-signature pattern-probe
   is a DETECTION/rescue read, alive-direction-only — acquisition stays
   banned, the poller never writes a pid file) — run ALONGSIDE the
   pid-file probe, the pattern bracketed per (a) (an unbracketed pattern
   can self-match the monitor's own command line and mint a false-ALIVE
   verdict); the pattern probe supplements, never replaces, the pid file.
1e. **Relaunch-descope record + handle-sidecar currency (#1689).** A
   relaunch that CHANGES the realized recipe (layers / draw counts /
   cells / models / seeds / scope) or the OUTPUT ROOT relative to the
   last posted run marker or approved plan is a DESCOPE, not a plain
   retry. Two duties, both in the SAME relaunch step, binding the
   VM-side agent performing the relaunch (pod-side code never shells
   `scripts/task.py`):
   (i) BEFORE launch, post an `epm:progress` descope note naming
   old → new per changed axis, the new out-root, the reason, and the
   launcher path (+ whether it is committed). This is the durable
   record the analyzer reconciles the realized recipe against — an
   unmarked descope is clean-result contamination (CIs read as if they
   came from the documented recipe). The descope note is IN ADDITION
   to item 2's fresh `epm:run-launched` re-post, never a substitute
   for it.
   (ii) In the same relaunch step, rewrite the dispatch handle sidecar
   `.claude/cache/issue-<N>-handle.json` so its run-scoped fields
   track the NEW run — `extra.expected_artifacts.sentinel_path`,
   `log_path`, `extra.pid_file`, `extra.runpod_attempt_id`,
   `extra.workload_pid` where changed — atomically (tmp+rename,
   mirroring item 1's atomic rewrite), or re-dispatch through
   `dispatch_issue.py` so the router rewrites it. Duty (ii) keys on
   CHANGED run-scoped handle fields, not on the recipe: a SAME-recipe
   relaunch into a new out-root / attempt also stales `sentinel_path`
   and gets the rewrite, even though the "descope" naming suggests
   recipe changes. On the hand-rewrite path, never resurrect a sidecar
   already retired to `<name>.finalized` (the `dispatch_issue.py`
   finalize contract) — re-dispatch is the safe default there. A
   handle pointing at a dead run's completion sentinel is a SILENT
   poller kill: `backend_poll.py` reads the sidecar every tick, so
   completion is never observed (#1689: an uncommitted marker-less
   descope relaunch left the handle pointing at the OLD attempt's
   completion sentinel — both found only by a user-requested manual
   audit).
1f. **Full stdio detach on every ssh-remote (re)launch — the wrapper is
   never the signal.** The remote launch command MUST redirect ALL THREE
   stdio fds in the SAME command: `< /dev/null` for stdin AND
   `> <log> 2>&1` (or `>> <log> 2>&1`) for stdout/stderr. `setsid` +
   `nohup` alone do NOT release the ssh channel: sshd waits for EOF on
   the remote stdout/stderr, so a detached child that inherits those fds
   keeps the channel open and the LOCAL ssh client hangs indefinitely
   (standard ssh fd semantics; same-run corroboration: #1768
   `epm:failure-lesson v2` — "it holds the ssh channel open so the local
   client hangs"). The local wrapper's lifetime is NEVER a signal: bound
   the local ssh call with a backstop
   (`timeout --kill-after=10s 60s ssh ...`) and verify the launch via
   the pid file + log breadcrumbs (items 1/1d), never via the wrapper
   staying alive. The `&`-precedence trap: in
   `ssh pod 'cd X && setsid nohup <cmd> > log 2>&1 < /dev/null & echo $! > pidfile'`
   the trailing `&` backgrounds the ENTIRE `cd && setsid` list, so `$!`
   is an un-setsid'd wrapper subshell — HUP-vulnerable and (when any fd
   is attached) the very process holding the channel. Mitigation: make
   `&` bind to the setsid unit alone via a brace group
   (`cd X && { setsid nohup <cmd> < /dev/null > log 2>&1 & echo $! ... ; }`),
   or repoint the pidfile at the setsid session leader (SESS==PID) per
   the #1768 r3 failure-lesson. (#1768: a pod-side relaunch left the
   local ssh wrapper hanging ~2.5 h; the launching session died with
   `epm:run-launched` unposted — a ~2.7 h window in which the healthy
   run was invisible to the poller's marker-pid probe.)
1g. **A RELAUNCH re-runs the original launcher FILE — never a
   hand-re-typed / reconstructed inline chain (#1768).** The launcher
   script is the carrier of every side duty a launch owes — the pid-file
   rewrite (item 1), log rotation (1b/1c), stdio detach (1f), the
   completion-sentinel write the poller's done-verdict keys on, and the
   dedicated-box done-path teardown leg (1i) — and a
   from-memory `bash -c` reconstruction silently drops whichever duty the
   re-typer forgets (#1768: the rebuilt chain dropped the
   completion-sentinel write; the finished run's handoff stranded ~5.8 h).
   If the original
   launch has no launcher file (an ad-hoc first launch), FIRST
   materialize the chain into a file (pod:
   `/workspace/logs/launch_issue_<N>_<slug>.sh`; VM: alongside the phase
   log), then relaunch by executing that file — every later relaunch then
   has a canonical source. Deliberate launcher EDITS before a relaunch
   are fine — fix the bug in the FILE, then run the file (a
   recipe-changing edit also carries 1e's descope record); the ban is on
   bypassing the file.
1h. **Pid breadcrumbs + completion watches key on the identity-verified
   WORKER pid — never the setsid/nohup/ssh wrapper pid (#1769).** A
   wrapper exits within seconds of a healthy launch, so a watch keyed on
   it false-fires "EXITED" against a live run (#1769: a completion watch
   keyed on the setsid launcher pid false-fired ~1 min in). Before
   writing any `pid=` breadcrumb or arming
   any liveness/completion watch, identity-verify the pid:
   `ps -p <pid> -o args=` must show the WORKLOAD's distinctive
   invocation — args reading as `setsid` / `nohup` / `ssh` / a bare
   wrapper shell mean you captured the wrapper. Re-derive the worker pid
   ONLY via item 1d's sanctioned launch-anchored forms (the launcher's
   pre-exec `echo $$`; `$!` of the setsid-exec'd unit; the parent-scoped
   child-walk `pgrep -P <captured wrapper pid>`) — 1d's ban on unanchored
   post-hoc `pgrep` is unchanged.
1i. **Done-path teardown leg on every DEDICATED-box workload —
   hand-rolled launchers included.** A workload on a dedicated ephemeral
   box must not end by merely uploading its results and exiting; the
   teardown leg is LANE-SPLIT:
   - GCE / DELETE-on-poweroff ephemeral box: the launcher/workload chain
     ENDS with the rendered lane's done-path teardown — set
     `eps/phase=done`, then poweroff (directly, or via the rendered #935
     done-grace self-poweroff when sentinel draining matters); on crash,
     the crash-persist → `phase=failed` → poweroff tail. A hand-composed
     / inline launcher (the 1g materialize-a-launcher path) is the risk
     case: nothing renders the teardown for it, so the leg must be
     written in.
   - Dedicated POD: the workload's leg is SENTINEL-ONLY — write the
     completion sentinel the poller's done-verdict keys on; teardown
     stays the owning session's VM-side verify-then-terminate
     (`pod.py terminate` after upload-verification PASS), NEVER an
     in-workload self-stop (it would preempt the upload-verifier, and a
     STOPPED volume is not durable, #1112).
   Scope: dedicated ephemeral boxes ONLY — NEVER the shared VM (a
   VM-side detached phase, SKILL.md § Detached VM-side long compute
   phases, must never poweroff its host) and never a SLURM node (the
   scheduler owns node lifecycle). And the box-side leg is the BACKSTOP,
   not the plan: the launching session's own
   `dispatch_issue.py finalize` at harvest (pod: verify-then-terminate)
   remains PRIMARY — never wait out the #935 done-grace window or the
   janitor fence. Incident signature (#1739, 2026-08-01): the
   router-dispatched box `gap1nulldiag` finished, then lingered RUNNING
   at `eps/phase=done` with idle-GPU billing ≈1h before a manual reap; a
   second box (`newarma5evil`) needed a manual finalize the same day
   (incident record).
2. **The fresh `epm:run-launched` carries the SAME live pid (`pid=`) AND
   `pid_file=`** (SKILL.md § "Any relaunch must re-post `epm:run-launched`").
   `poll_pipeline.py` computes `pid_alive = pidfile_pid_alive OR
   marker_pid_alive OR sig_proc_rescue` (poll_once; the third term is
   #1650's marker-signature-derived, alive-direction-only rescue — it fires
   only on `cmd='...'`/`launcher_script=`-bearing markers, when BOTH probed
   pids are dead and live processes match the derived launch signature), so
   a stale pid FILE is rescued while the newest marker's `pid=` is itself
   the live process, or — on a signature-bearing marker — while
   signature-matched processes are live. A PRESENT-but-stale pid file is
   worse than a missing one: the `pid_file_missing` fallback + WARN fires
   ONLY when the file is absent — a stale file silently probes a dead pid
   every tick, with no warning — #1650 adds a cmdline identity WARN
   (`pid_identity=mismatch` in the tick JSON) for an alive-but-wrong pid,
   verdict unchanged.
3. **Worked example (#813 v5).** A relaunch skipped the pid-file rewrite,
   leaving the predecessor's dead pid in the file, and posted a marker pid
   that also mismatched the live dispatcher — both probes read dead
   against a healthy run, and a corrective extra round existed SOLELY to
   rewrite the pid file and re-post `epm:run-launched`.

Residual honesty: this contract HAS a WARN-only runtime detector (#1156 —
the tick-JSON flag `pid_file_stale_vs_marker` when the pid file's mtime
predates the newest `epm:run-launched` by `EPM_POLL_PID_MARKER_SLACK_SEC`,
default 600 s) and a cmdline identity detector (#1650 — `pid_identity` /
`marker_pid_identity`, WARN on `mismatch`); neither changes a verdict.
TWO mechanical rescues ARE verdict-bearing: the marker-pid OR-probe (while
the newest marker's pid is itself alive), and the #1650 signature rescue
(`sig_proc_rescue`, alive-direction only; kill switch
`EPM_POLL_PID_IDENTITY=0`). On a free-prose marker (no signature fields)
the pre-#1650 residual stands: a wrong-and-dead pid in BOTH the file and
the marker still reads `dead`.

### Result-push verification contract (#1205)

- Any pod/GCE dispatch step that `git commit`s results to the issue branch
  MUST verify the push landed: after `git push`,
  `git -C <root> rev-list --count origin/<branch>..HEAD` prints `0` (git
  updates the remote-tracking ref on a successful push, so the count is a
  local, network-free proof); retry once on failure; still non-zero → exit
  non-zero — fail the workload loud, never declare done with an unpushed
  result commit. The `git push … || echo WARNING` / `|| true` shape is
  **BANNED** (#825: 73 committed eval JSONs existed only on a
  self-DELETEing GCE instance; the workload-side sibling of the
  #957/#1048 piped-push masking class — enforced by `workflow_lint.py
  --check-piped-git-push` + `--check-push-failure-swallow`).
- **Fetch + rebase before every pod/instance-side results-git push
  (#1880).** A lane's terminal push races ANY orchestrator branch commit
  made mid-run (a sibling lane's crash-fix relaunch is the normal
  multi-lane case): a bare push retry that detects behind>0 but never
  fetches loses DETERMINISTICALLY — non-fast-forward rejection → workload
  exit 1 → a HEALTHY run crash-persists and powers off (#1739: 31h of
  complete science, exit 1 at the terminal push, ~30 min manual recovery
  from crash-persist). Push recipe:
  `git fetch origin <branch> && git rebase origin/<branch>` (result
  commits are additive per-lane files — a content conflict is
  near-impossible), then push and re-verify per the rev-list bullet
  above; bounded 2 attempts. NOTE the rebase rewrites the LOCAL result
  commits' SHAs pre-push — no standard-contract consumer pins them (the
  artifact-presence assert below is path-keyed against the pushed tree;
  fix-sha ancestry probes reference origin ancestors, which
  rebase-onto-origin preserves) — a future driver that records its own
  result-commit SHA must record it AFTER the push-verify, never before.
  On rebase conflict: `git rebase --abort` and proceed to the lane's
  EXISTING fail-loud path (GCE: bundle + exit 86; crash-persist preserves
  the results) — the standing "never declare done with an unpushed result
  commit" contract and the Part A-ter sentinel ordering are UNCHANGED (a
  push-failure-tolerant exit-0 disposition was considered and REJECTED:
  it would let a run classify done-like with results only in a bundle).
- **Orchestrator side (#1880):** avoid pushing to the issue branch while
  lanes are mid-run when feasible; when a mid-run push is required (a
  sibling lane's crash-fix relaunch), expect in-flight lanes' terminal
  pushes to need the fetch+rebase path above — a lane running a pre-#1880
  driver will deterministically false-crash at its terminal push (the
  #1739 shape) with its results intact in crash-persist; treat that
  failure as recoverable-transport, not a science loss. The PULL/SYNC
  direction — a worker's mid-run sync refusing on locally modified
  touched paths — is governed by `.claude/rules/crash-fix-rounds.md`
  § Mid-run pushes to a live-synced branch (enumerate live workers
  first; detached-SHA pin / defer / worker-local alternatives).
- **Artifact-presence assert (#1325) — the rev-list push-verify is VACUOUS
  against a never-committed result file.** `rev-list --count
  origin/<branch>..HEAD == 0` proves the COMMITS pushed; it says nothing
  about result files that were never `git add`ed (#928: 5 eval JSONs + 24
  figure files sat untracked while the push-verify passed on code
  commits). After the push-verify succeeds, the SAME dispatch step MUST
  assert the round's DECLARED git-destined result paths are present in
  the PUSHED tree: for EACH result file `p` the driver DECLARES for this
  round under `eval_results/issue_<N>/...` or `figures/issue_<N>/...` —
  its own output manifest: eval JSONs exactly as declared in the
  `epm:results` sentinel payload's "Eval JSON paths" field, plus the
  round's figure outputs (the payload spec carries no named figures
  field, so the driver's manifest is the anchor there; per-file,
  never a bare directory: at #928's incident tip the two directories
  already held 90 files from EARLIER rounds, so a directory-level
  non-empty check passes vacuously) — run
  `git -C <root> ls-tree -r origin/<branch> --name-only -- "$p"` and
  require non-empty output (the remote-tracking ref was just
  push-verified, so this is the same local, network-free proof;
  `git cat-file -e "origin/<branch>:$p"` is an acceptable equivalent);
  any missing path → exit non-zero listing the misses, BEFORE
  `[phase=done]` and the results sentinel, and before stamping
  `EPS_DELIVERABLES_OK_PATH` (Part A-ter: a failing artifact assert
  classifies `failed`, never done-like) — never declare done with an
  uncommitted result file. Scoping (so the assert never false-fails):
  (a) the declared set is DECLARE-anchored — only the git-destined
  result files the WORKLOAD itself produced AND declares this round;
  gitignored outputs (`*.pt` / `*.log` under `eval_results/`; their
  canonical home is HF), undeclared resume / partial state (`partial/**`
  checkpoints and manifests — written but never declared), and artifacts
  a later VM-side agent commits (e.g. analyzer-produced figures, the
  #1090 DEFERRED-figures shape, Step-8 verifier-synced eval JSONs) are
  out of the set; (b) a round with no git-destined outputs (HF-only
  datagen / training rounds) has an empty set — the assert is a
  no-op, never a false-fail; (c) lane scoping is unchanged: RunPod + GCE
  dispatch only — on SLURM workload-side git is structurally impossible
  (bullet below), and the VM-side orchestrator commit + Step 8 gate own
  artifact landing there. HF-destined artifacts (raw completions incl.
  judge raw, checkpoints, analysis tensors) are OUT OF SCOPE for this
  assert — the Upload Policy persist-by-default rule owns that leg
  (sibling incident #1090 v4, same day: Tier-2 judge raw existed only
  VM-local; no git-tree assert could catch it). The GCE push-verify
  backstop (below) shares the blindness — it proves commits pushed, NOT
  files committed — so this assert is a driver duty on every lane that
  commits results; no mechanical backstop covers it.
- **Named expected-path set — an empty-set verify is vacuous (#1482).**
  The #1325 assert above checks each DECLARED path per-file; this bullet
  binds the DECLARATION itself. Every push-verify / upload-verify leg
  NAMES the expected path set it is about to check — print the resolved
  list (or the manifest path + count) into the phase log BEFORE
  verifying. An EMPTY resolved set on a round whose plan / output
  manifest / `primary_deliverable` declares git-destined outputs is a
  verify FAILURE — exit non-zero naming the empty set — never a pass
  (#1482: an empty expected set PASSed while 29 git-bound eval files sat
  uncommitted). A
  round with genuinely no git-destined outputs (the #1325 scoping-(b)
  case) may no-op, but STATES the no-op in the log ("push-verify: no
  git-destined outputs declared this round"), never silently.
- **GCE lane:** the startup script configures a `GITHUB_TOKEN` env-reading
  credential helper (pre-#1205 workload pushes failed DETERMINISTICALLY —
  the clone is tokenless) and runs a post-workload push-verify backstop
  (fetch + rebase onto `origin/<branch>`, then retry — the #1880 recipe;
  a rebase conflict aborts into the fail-loud tail → bundle the unpushed
  range to `data/issue_<N>/` → `exit 86` → EXIT trap → `phase=failed` +
  crash-persist + poweroff). The backstop covers forgetful dispatch
  scripts; scripts SHOULD still verify their own push. The backstop
  pushes `HEAD` to the CLONED branch (`HEAD:<repo_branch>`); both the
  helper and the backstop are repo-local to `$WORKLOAD_ROOT` — a workload
  that creates a SECOND clone gets neither, and a manual same-VM SSH
  relaunch (the #491/#908 salvage shape) runs OUTSIDE the mechanical
  backstop: its shell must verify its own push per the first bullet.
- **RunPod lane:** the tokenized remote (`bootstrap_pod.sh` step 4)
  authenticates; there is NO mechanical backstop — the dispatch script's
  own verification is the only guard. Accepted asymmetry: a pod persists
  and is SSH-able, so an unpushed commit is recoverable after the fact;
  GCE's DELETE-on-poweroff boot disk is why that lane got the mechanical
  leg.
- **SLURM lane:** the mechanical rev-list leg is structurally
  INAPPLICABLE — cluster compute nodes run on an ephemeral `$SCRATCH`
  **rsync copy with no git checkout** (`RSYNC_INCLUDE_PATHS` in
  `backends/slurm.py` carries no `.git`), so a workload-side `git commit`
  / `git push` of results is impossible and fails loud by construction
  (`fatal: not a git repository`) — and the SLURM secrets env carries no
  `GITHUB_TOKEN`. The lane's result-landing story:
  `SlurmBackend.fetch_results` rsync-PULLS `eval_results/` + `figures/`
  back to the VM repo root (pull failure WARN-only, #598), then
  `confirm_artifacts` is the downstream hard gate before teardown, and
  the COMMIT of pulled results is VM-side, orchestrator-owned (Step 8
  verifier sync / Step 9b–10d auto-merge). Consequence: a dispatch script
  whose deliverable REQUIRES workload-side git-committed results must NOT
  route to SLURM — pin `backend: runpod` (`backend: gcp` is REFUSED,
  #2028). (`--repo-branch` IS honored on SLURM as of #793 — the workload
  runs branch code; it just cannot push from the cluster.)
- **Part A-ter interplay** (`.claude/rules/compute-backend-failover.md`):
  a workload whose declared deliverables include git-committed eval JSONs
  must NOT stamp `EPS_DELIVERABLES_OK_PATH` before verifying its own push
  — else a push-verify backstop failure classifies
  `finalize_failed_artifacts_ok` (done-like) instead of `failed`; the
  data still crash-persists either way.

### Pod-side preflight gates (behind-origin/main false positive — LEGACY post-#554)

> **LEGACY (post-#554):** preflight is branch-aware as of #554 — on an
> `issue-<N>` checkout the git check compares the branch against its OWN
> `origin/issue-<N>` ref and demotes behind-origin/main to a WARNING, and
> bare (non-`--json`) preflight fails loud. **On post-#554 code, a
> `Local is N commit(s) behind origin/issue-<N>` or `git fetch origin
> failed` ERROR is REAL — a driver must NEVER tolerate it.** Parsing
> `--json` instead of gating on bare exit codes remains the right driver
> design either way.

A driver on a PRE-#554 pod checkout that gates launch on preflight under
`set -e` / `fail_loud` MUST tolerate the documented feature-branch false
positive: that era's git check counts `HEAD..origin/main`, so on EVERY
`issue-<N>` pod checkout it reports `Local is N commit(s) behind
origin/main` and exits non-zero even at the reviewed branch tip. Run
`preflight --json` and fail only when `errors` contains anything OTHER
than that line (#552; the experimenter's own preflight invocation carries
the same legacy-scoped tolerance).
