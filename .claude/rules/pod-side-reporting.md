---
description: Pod-side dispatcher result-reporting contract (sentinel files, poll_pipeline.py drain, epm:results payload, pod-side sentinel READ-BACK tolerance under the .processed drain-rename (#1311)) + pid-file launch contract (rewrite on EVERY (re)launch, #813) + legacy pod-side preflight gates; relocated verbatim from experiment-implementer.md, #829
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
   terminal line (`[phase=done] SMOKE COMPLETE ...`) and only survives it
   via pid/sentinel corroboration (incident #545, 2026-06-11: a per-cell
   `[phase=done]` echo produced a false `status=done` while the
   dispatcher was alive and GPUs were at 85%). Mechanically enforced by
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
     "epm:results", ...)` — issue634/744/1073/779 writers). Two
     operational residuals: (a) a stale straggler — e.g. a resumed
     stopped pod draining an OLD run's results sentinel — now lands
     ABOVE newer rows; this is operator-recoverable (the marker carries
     `sentinel_declared_version` + the drain logs a warning with the
     sentinel's `ts`; re-post the correct results as a fresh
     higher-version marker). (b) An operator's manual high-version
     correction marker is shadowed by ANY subsequent real results drain
     (which lands at max+1 above it) — re-post corrections AFTER the
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
   identically via `backends/gcp.py::_mark_sentinel_processed`; SLURM
   has no sentinel channel). Post each sentinel ONCE, never rewrite it
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
   renamed anyway. Incident #1090 fu3/fu4 (code-review r1): per-run
   sentinels doubled as resume/finalize state; the drain renamed them
   mid-run → requeue races + a production reproducibility_card covering
   only 23-24 of 35 cells.

Rationale: task #448 (2026-05-31) — the pod-side dispatcher completed all
cells cleanly but (a) never emitted `[phase=done]` and (b) wrote its
sentinel with the key `schema` instead of `sentinel_schema_version`. The
orchestrator's poll loop reported a FALSE `dead`, `_parse_sentinel`
silently dropped the end-of-run sentinel for missing required keys, and
`epm:results` had to be posted by hand from a separate SSH session.

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
runs (#597 follow-up).** A top-level `wandb_url: "n/a (per-cell wandb
runs; project=<P>)"` string in the payload — without an accompanying
`reproducibility_card` / `production_provenance` — is the worst of
both worlds: it looks like a deliberate decision (the project name is
there) yet declares NONE of the fields the verifier needs to resolve the
live runs. The verifier then falls back to `api.default_entity` for
WandB, which may or may not match the entity that actually owns the runs
(the typical project trap: HF `default_entity` is `superkaiba1` while
WandB `default_entity` is `thomasjiralerspong`, so an HF-style entity
guess silently misses every live run). When per-cell runs really are the
shape — every cell trains its own WandB run — emit the full multi-cell
card: `wandb_project: "<P>"` + `wandb_run_names: [<display name per
cell>]` + `wandb_entity: "<entity>"`. `wandb_url` (the top-level
catch-all) MAY be `n/a (per-cell wandb runs; see reproducibility_card)`
or omitted; the card is what carries the resolution surface.

**`wandb_entity` is STRONGLY RECOMMENDED whenever the card uses
`wandb_run_names` + `wandb_project`** (i.e. the multi-cell case the
above paragraph mandates). The verifier's `check_wandb_runs_by_name`
threads the card's `wandb_entity` straight through, and when the field
is omitted it falls back to `api.default_entity`. That fallback is a
safety net, NOT the contract: it relies on the dispatcher running under
the SAME WandB login as the verifier and on the user having a single
default entity, neither of which is guaranteed in a multi-account
workspace (e.g. a personal `thomasjiralerspong` entity vs an
organization `superkaiba1`). Read the entity off the WandB SDK at run
time (`wandb.run.entity` while the run is open, or
`wandb.Api().default_entity` after) and persist it in the card; never
hand-type it as a literal — a stale literal silently breaks resolution
when the account changes (#597 follow-up r3: a flat `wandb_url: "n/a
(...; project=issue597-leakage-dynamics)"` left three filler runs
invisible to round-3 verification on the HF/WandB entity-default
mismatch, recovered only after the orchestrator manually superseded the
row). Producer-side: every dispatcher that writes per-cell WandB runs
emits `wandb_entity` in the same card it emits `wandb_project` +
`wandb_run_names`.

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
   `exec`ing the workload (`.claude/agents/experimenter.md` § During
   Execution steps 1/1b — the agent-specific recipe). When relaunching
   WITHOUT the launcher (rare), chain an explicit ATOMIC rewrite into the
   same command:
   `printf '%s\n' "$CHILD_PID" > /workspace/logs/issue-<N>.pid.tmp && mv /workspace/logs/issue-<N>.pid.tmp /workspace/logs/issue-<N>.pid`
   (tmp+rename — no window where the poller reads a truncated/empty file;
   the launcher-internal `echo $$ >` truncate-write is the accepted
   in-launcher form because it is a single short write completing before
   the workload starts). Then CONFIRM before posting: `cat` the pid file in
   a fresh SSH call and check it equals the pid you post in the marker. A
   relaunch that leaves a predecessor's pid in the file is a
   launch-contract violation.
2. **The fresh `epm:run-launched` carries the SAME live pid (`pid=`) AND
   `pid_file=`** (SKILL.md § "Any relaunch must re-post `epm:run-launched`").
   `poll_pipeline.py` computes `pid_alive = pidfile_pid_alive OR
   marker_pid_alive` (poll_once, ~line 4199), so a stale pid FILE is rescued
   only while the newest marker's `pid=` is itself the live process. A
   PRESENT-but-stale pid file is worse than a missing one: the
   `pid_file_missing` fallback + WARN (~4205-4212) fires ONLY when the file
   is absent — a stale file silently probes a dead pid every tick, with no
   warning.
3. **Worked example (incident #813 v5, 2026-07-02).** The run-4 relaunch
   (00:31, `bash scripts/issue813_dispatch.sh`) skipped the pid-file
   rewrite, leaving run-2's dead pid 6267 in `/workspace/logs/issue-813.pid`;
   the marker pid it posted (11634) did not match the run-5 note's live
   dispatcher (11636), so both probes read dead against a healthy run — a
   false dead verdict. A corrective run-5 marker (00:39) was needed SOLELY
   to rewrite the pid file with 11636 and re-post `epm:run-launched` — an
   entire extra round whose only content was this contract.

Residual honesty: this contract now HAS a WARN-only runtime detector (#1156 —
`poll_pipeline.py` warns on every tick, and sets the tick-JSON flag
`pid_file_stale_vs_marker`, when the pid file's pod-clock mtime predates the
newest `epm:run-launched` marker by more than
`EPM_POLL_PID_MARKER_SLACK_SEC`, default 600 s), so a rewrite-skipping
relaunch is named in the poll log instead of left to manual archaeology. The
detector never changes a verdict: the poller's marker-pid OR-probe remains
the only VERDICT-bearing mechanical rescue, and only while the newest
marker's pid is itself alive.

### Result-push verification contract (#1205)

- Any pod/GCE dispatch step that `git commit`s results to the issue branch
  MUST verify the push landed: after `git push`,
  `git -C <root> rev-list --count origin/<branch>..HEAD` prints `0` (git
  updates the remote-tracking ref on a successful push, so the count is a
  local, network-free proof); retry once on failure; still non-zero → exit
  non-zero — fail the workload loud, never declare done with an unpushed
  result commit. The `git push … || echo WARNING` / `|| true` shape is
  **BANNED** (incidents #825 r6/r7/r8, upload-verification reads
  2026-07-08T11:17/11:19Z: 73 committed eval JSONs existed only on a
  self-DELETEing GCE instance; the workload-side sibling of the #957/#1048
  piped-push masking class — the pipe mirror stays enforced by
  `workflow_lint.py --check-piped-git-push`, this swallow shape by
  `--check-push-failure-swallow`).
- **Artifact-presence assert (#1325) — the rev-list push-verify is VACUOUS
  against a never-committed result file.** `rev-list --count
  origin/<branch>..HEAD == 0` proves the COMMITS pushed; it says nothing
  about result files that were never `git add`ed (incident #928,
  upload-verification v5 2026-07-15T00:07Z: the round's 5 eval JSONs + 24
  figure files sat untracked on the instance while the driver's
  push-verify passed on its code commits — caught one round late at
  Step 8). After the push-verify succeeds, the SAME dispatch step MUST
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
- **GCE lane:** the startup script configures a `GITHUB_TOKEN` env-reading
  credential helper (workload pushes authenticate; pre-#1205 they failed
  DETERMINISTICALLY — the clone is tokenless) and runs a post-workload
  push-verify backstop (retry → bundle the unpushed range to
  `data/issue_<N>/`, crash-persist-swept per #854 item 5 → `exit 86` →
  EXIT trap → `phase=failed` + crash-persist + poweroff). The backstop
  covers forgetful dispatch scripts; scripts SHOULD still verify their own
  push so the failure surfaces at the failing phase with its own context.
  The backstop pushes `HEAD` to the CLONED branch (`HEAD:<repo_branch>`)
  — a workload that checks out a different local branch before committing
  (out-of-contract) gets those commits backstop-pushed onto the cloned
  branch, not its own.
  Both the helper and the backstop are repo-local to `$WORKLOAD_ROOT` —
  a workload that creates a SECOND clone gets neither (prose contract
  only), and a manual same-VM SSH relaunch (the #491/#908 salvage shape)
  runs OUTSIDE the mechanical backstop: its shell must verify its own
  push per the first bullet.
- **RunPod lane:** the tokenized remote (`bootstrap_pod.sh` step 4)
  authenticates; there is NO mechanical backstop — the dispatch script's
  own verification is the only guard. Accepted asymmetry: a pod persists
  and is SSH-able, so an unpushed commit is recoverable after the fact;
  GCE's DELETE-on-poweroff boot disk is why that lane got the mechanical
  leg.
- **SLURM lane:** the mechanical rev-list leg is structurally
  INAPPLICABLE — cluster compute nodes run on an ephemeral `$SCRATCH`
  **rsync copy with no git checkout** (`RSYNC_INCLUDE_PATHS` in
  `backends/slurm.py` carries no `.git`; the `post_marker_via_task_py`
  docstring pins this), so a workload-side `git commit` / `git push` of
  results is impossible and fails loud by construction
  (`fatal: not a git repository`) — and the SLURM secrets env
  (`SECRET_ENV_KEYS`, `backends/slurm.py`) carries no `GITHUB_TOKEN`, so
  even an out-of-contract cluster-side self-clone could not authenticate
  a push (the pre-#1205 GCE tokenless shape). The lane's result-landing
  story is instead: `SlurmBackend.fetch_results` rsync-PULLS
  `eval_results/` + `figures/` from `$SCRATCH_JOB_DIR` back to the VM
  repo root (pull failure is WARN-only by the deliberate #598 contract),
  then `confirm_artifacts` (completion sentinel + git-figures +
  eval-JSON + HF + WandB checks) is the downstream hard gate before
  teardown (a bool the orchestrator's upload-verification gates on), and
  the COMMIT of pulled results is VM-side, owned by the orchestrator
  (Step 8 upload-verifier sync / Step 9b–10d auto-merge) — where the
  same push-verification discipline governs the orchestrator's own
  VM-side push via the repo-wide push rules (the piped-push hook,
  `sync_repo_root.py`, Step 10d). Consequence: a dispatch script whose
  deliverable REQUIRES workload-side git-committed results must NOT
  route to SLURM — pin `backend: gcp` or `runpod`. (`--repo-branch` IS
  honored on SLURM as of #793 via VM-side branch-tree materialization —
  the workload runs branch code; it just cannot push from the cluster.)
- **Part A-ter interplay** (`.claude/rules/compute-backend-failover.md`):
  a workload whose declared deliverables include git-committed eval JSONs
  must NOT stamp `EPS_DELIVERABLES_OK_PATH` before verifying its own push
  — else a push-verify backstop failure classifies
  `finalize_failed_artifacts_ok` (done-like) instead of `failed`; the
  data still crash-persists either way.

### Pod-side preflight gates (behind-origin/main false positive — LEGACY post-#554)

> **LEGACY (post-#554):** preflight is branch-aware as of 2026-06-12
> (#554, commit `25f227273`) — on an `issue-<N>` checkout the git check
> compares the branch against its OWN `origin/issue-<N>` ref and demotes
> behind-origin/main to an informational WARNING, so the false positive
> below no longer exists on a pod synced to current code. #554 also made
> bare (non-`--json`) preflight fail loud (summary on stdout, per-error
> stderr lines), closing the silent-death mode. Keep the tolerance below
> ONLY for a pod still running pre-#554 code. **On post-#554 code, a
> `Local is N commit(s) behind origin/issue-<N>` or `git fetch origin
> failed` ERROR is REAL — a driver must NEVER tolerate it.** Parsing
> `--json` instead of gating on bare exit codes remains the right driver
> design either way.

A driver on a PRE-#554 pod checkout that gates launch on `uv run python -m
explore_persona_space.orchestrate.preflight` under `set -e` / `fail_loud`
MUST tolerate the documented feature-branch false positive: that era's git
check counts `HEAD..origin/main`, so on EVERY `issue-<N>` pod checkout it
reports the ERROR `Local is N commit(s) behind origin/main` and exits
non-zero even when the pod sits exactly at the reviewed branch tip. Run
`preflight --json` and fail only when `errors` contains anything OTHER
than that line. Never let that single error be the sole
launch-killer. Incident #552 (2026-06-10): a pod-side driver ran bare
`preflight || fail_loud` under `set -euo pipefail`; it survived launch
only because the experimenter happened to repoint the pod-local
`origin/main` ref seconds before the check ran — every NEW driver that
re-runs preflight re-introduces the fatal check unless it parses the
error list. (The experimenter's own preflight invocation carries the same
legacy-scoped tolerance; see `.claude/agent-memory/experimenter/feedback_preflight_feature_branch_false_positive.md`.)
