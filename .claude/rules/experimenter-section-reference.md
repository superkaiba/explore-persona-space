---
paths:
  - ".claude/rules/experimenter-section-reference.md"
description: >
  Full launch-gate and recovery detail for experimenter.md (RunPod
  bootstrap-completeness probe, GCP salvage-relaunch, the Before-Running
  item-4 input-data completeness gate, the vLLM generate-hang triad),
  relocated from .claude/agents/experimenter.md (per-spawn system-prompt
  cost; #1090/#2054 fixed-overhead deaths). Loaded ONLY via the explicit
  § pointer lines in experimenter.md — the self-matching `paths:` glob
  keeps this file out of every other agent context.
---

# Experimenter section reference (relocated launch-gate + recovery detail)

One H2 per relocated section, detail relocated verbatim from
`.claude/agents/experimenter.md`. Read ONLY the section you need: Grep the
heading, then a chunked `Read` of that span — never the whole file. The
OPERATIVE contract for every section stays in experimenter.md; this file
carries the extended recipes, verbatim probe commands / failure-body
templates, and incident grounding.

## Bootstrap-completeness probe detail — RunPod lane

### Post-dispatch bootstrap-completeness probe (RunPod lane)

A written run handle (`.claude/cache/issue-<N>-handle.json`) does NOT mean
the pod is launch-ready: `dispatch_issue.py launch` provisions the pod and
writes the handle BEFORE `bootstrap_pod.sh` finishes the clone + `.venv`
build. If the launching process is killed at the Bash command timeout
mid-bootstrap, you inherit a handle pointing at a half-bootstrapped pod —
no `.venv`, a half-materialized git tree, and MooseFS tree-writes that
wedge on every subsequent command (`git reset --hard` / `git checkout`  <!-- workflow-lint: allow-git-reset-hard: pod-side MooseFS wedge description, not a repo-root instruction -->
hang for many minutes, the on-disk file count freezes). Nursing that wedge
inline is wasted work; classify it and let the lifecycle layer recover.

Run this BEFORE the sync/preflight steps below (the step-2 `git checkout`
is exactly the command that wedges on a half-materialized MooseFS tree):

```bash
ssh_execute(server="epm-issue-<N>",
            command="ls -d /workspace/explore-persona-space/.venv && \
                     git -C /workspace/explore-persona-space ls-files | wc -l")
```

Verdict rule — classify `failure_class: infra` (provision-incomplete) when
ANY of these holds, post `epm:failure v1`, and EXIT (do NOT nurse a wedged
MooseFS checkout inline):

- no `.venv` directory, OR
- `git ls-files` returns 0 (empty index — clone did not complete), OR
- the on-disk file count is FROZEN over a 6s sample —
  `find /workspace/explore-persona-space -type f | wc -l` taken twice 6s
  apart returns the same number while a clone/checkout is supposedly still
  in flight.

```
failure_class: infra
reason: provision-incomplete
note: handle written but bootstrap did not finish (no .venv / empty git
      index / frozen file count); do not nurse the MooseFS wedge inline
```

`/issue` Step 7 routes `failure_class: infra` back to a fresh experimenter
respawn (cap 3) after the lifecycle layer terminates + re-provisions a
clean pod.

**Cap the launch Bash timeout at the tool max so a slow-but-healthy
bootstrap is not truncated into this state.** When the orchestrator
invokes `dispatch_issue.py launch` over a Bash command (RunPod lane),
pass `timeout=600000` (10 min — the Bash tool's maximum); the default
120s/540s window can kill an in-progress bootstrap and manufacture the
half-bootstrapped pod this probe exists to catch. Reference: #640 round 4
(2026-06-15) — the original nurse-it-inline attempt burned ~15 min on the
wedge before classifying infra; see also `.claude/rules/gotchas.md`
MooseFS quota entry.

## GCP salvage-relaunch detail — env staging and remote kills

### GCP-lane salvage-relaunch (existing instance, code-fix land)

When the host alias the brief gives you is a GCP GCE instance
(`eps-issue-<N>`) and you are SALVAGE-RELAUNCHING — landing a code-fix
onto an instance that is already up rather than launching on a fresh
provision — two traps cost multiple round-trips before. The
authoritative recipe is agent memory
`feedback_gcp_salvage_relaunch.md`; the operative points:

1. **No repo-root `.env` on a fresh GCP instance — and a salvage SSH
   session does not inherit the startup-script env.** Post-#1205 the
   startup script configures an env-reading git credential helper
   repo-local on the workload clone (`backends/gcp.py`), but that
   helper reads `GITHUB_TOKEN` from the INVOKING environment with no
   `.env` fallback, so a bare `git fetch` in a fresh SSH shell can
   still fail or prompt, and credentialed pipeline phases (analyze /
   upload — WandB/HF) have no `.env` to read. Recover by:
   - **Stage the local VM `.env`** to `/workspace/eps-issue-<N>/.env`
     via stdin to a root-only file (mode 600) — never echo the token
     into the argv.
   - **Fetch helper-authenticated (#1239 contract — never a tokenized
     remote URL):** in one stdin script to `sudo bash -s`: source the
     just-staged `.env` (`set -a; . ./.env; set +a`), `export
     GIT_TERMINAL_PROMPT=0`, idempotently refresh the env-reading
     credential helper, then `git fetch origin <branch>` — exact
     fenced block in `feedback_gcp_salvage_relaunch.md`. The token
     rides the command environment only (never argv, the remote URL,
     or git config); the helper supplies it as Basic auth, which
     classic `ghp_` PATs accept.
2. **NEVER `pkill -f "<pattern present in your own SSH argv>"`** to kill
   a stray remote process — the pattern self-matches the SSH command's
   own argv and SIGKILLs the session (gcloud exits 255, locking you out
   of the SSH stream). Kill stray remote procs by exact PID only.

## Before-Running item 4 detail — input-data completeness gate

4. **Verify input-data completeness against planned coverage (MANDATORY
   pre-launch gate; fail-loud, no launch on shortfall).** This is a
   coverage gate, NOT a sanity check — silently launching a degraded
   subset burns a full pod cycle producing an incomplete result. Read
   the plan's Reproducibility Card to enumerate the launch's planned
   coverage: how many cells / conditions / per-cell input files /
   per-domain datasets / seeds the dispatcher will iterate over. Then
   verify on the pod that the actual input-data files the launcher
   loads from local disk match that count. Concretely:

   - **Enumerate planned inputs.** From the plan (and the dispatcher's
     cell list / domain list / seed list as visible via the `--help`
     check in step 7), list every per-cell input artifact the
     dispatcher reads from local disk before training — typically
     per-cell training JSONLs (e.g. `data/issue<N>/*.jsonl`),
     per-domain drift datasets, per-condition prompt sets, persona
     seed caches. Get a single integer (planned_input_files) AND the
     glob pattern. **Also grep the launcher/dispatcher script itself
     for its own prestage gates** (`assert .exists()`, `[ -f ... ]`,
     `require_file`, hard-coded `eval_results/...` reads) and add
     every hard-required path to the enumeration — the brief is a
     paraphrase and can omit inputs the launcher hard-requires
     (incident #518, 2026-06-09: the prestage gate demanded
     `eval_results/issue_509/...`, absent from the brief's
     enumeration, and the gap surfaced only at launch).
   - **Plan-named prep-script outputs are gate items too.** When the
     plan or brief marks an input dataset as "regenerated locally via
     prep script" (e.g. a P0 prerequisite built by
     `scripts/issue<N>_prep_datasets.py`), add the prep script's
     OUTPUT file path(s) to the enumeration and stat-check them on
     the pod like any other planned input — a presence check on the
     regen path's secret/env var (e.g. `TURNER_EDS_PASSWORD`) does
     NOT substitute for the dataset file itself. Remediation for a
     missing output is running the named prep script on the pod
     before launch, preferring its free/deterministic path (e.g.
     decrypt-only `--no-generate`); if the script can fall back to a
     paid-API regen, surface that loudly in your launch note instead
     of letting it fire silently (the #468 paid-fallback trap).
     Incident: task #545 (2026-06-10) — the gate checked only
     `TURNER_EDS_PASSWORD` presence while the plan-named
     `data/issue404/turner_bad_medical_advice.jsonl` was absent on
     the fresh pod; the first launch crashed in seconds and was
     recovered by `scripts/issue458_prep_datasets.py --cells
     turner_bad_medical --no-generate` + relaunch.
   - **Count actuals on the pod.** Run one `ssh_execute ls -1
     <pattern> | wc -l` against the pod's local-disk path. Get a
     single integer (actual_input_files).
   - **Compare.** If `actual_input_files == planned_input_files`,
     proceed. If `actual_input_files < planned_input_files`,
     **REFUSE to launch**. Post `epm:failure v1` with body
     ```
     failure_class: infra
     reason: planned-input-data-missing-on-pod
     planned: <planned_input_files>
     actual: <actual_input_files>
     missing: <newline-separated list of the missing files, or the
              glob + a note that N rows are absent if listing each
              would exceed the body cap>
     ```
     and EXIT. Do NOT launch the dispatcher at degraded coverage.
     `/issue` Step 7 routes `failure_class: infra` back to a fresh
     experimenter respawn (cap 3); on respawn, sync the missing data
     to the pod (`pod.py sync data --push` or the equivalent dataset
     upload + re-pull) and re-run this check.
   - **Path-paraphrase guard (BEFORE posting `epm:failure infra`).**
     Briefs paraphrase paths — the orchestrator may have written
     `eval_results/issue_N/` while the dispatcher (the ground truth)
     actually writes Phase-0 outputs to `data/issue_N/`. Before
     failing, grep the dispatcher / Phase-0 script for the file
     basename (e.g. `R_train_new.json`) and confirm the brief's
     stated parent directory matches the script's actual write path.
     If the file IS present at the dispatcher's actual write path,
     the input-data gate PASSes and the `epm:run-launched` marker
     MUST carry `assumption: brief named <wrong path>; actual write
     path is <X>` so the discrepancy is recorded. Only post
     `epm:failure infra reason: planned-input-data-missing-on-pod`
     when the file is missing from BOTH the brief's path AND the
     dispatcher's actual write path. Incident: task #488 round-5
     relaunch (2026-06-05) — brief named
     `eval_results/issue_488/` for Phase-0 outputs; dispatcher
     wrote to `data/issue_488/`; literal-path check returned 0
     files and would have posted a false-positive
     `planned-input-data-missing-on-pod` abort.
   - **Dispatcher-default input paths — discover what the dispatcher
     will TRY to open, not just what the brief enumerates.** The
     enumerate-and-count mechanic above only covers files the brief
     names. Dispatchers commonly carry their own `--*-dir` / `--*-path`
     argparse defaults pointing at LOCAL paths the brief never
     mentions (carry-over centroids, persona banks, R_train.json from
     a parent task). Step 6a.5 verifies the HF mirror of those
     artifacts resolves but does NOT stage them to local disk — so a
     dispatcher launched against an unstaged default crashes seconds
     in. Before posting `epm:run-launched`, introspect the dispatcher's
     argparse defaults and stat-check each local path on the pod:
     ```bash
     ssh_execute(server="epm-issue-<N>",
                 command="cd /workspace/explore-persona-space && \
                          uv run python <dispatcher_path> --help")
     ```
     For every long flag in the help whose default is a LOCAL filesystem
     path (e.g. `--persona-bank data/issue_472/persona_bank.json`,
     `--centroids-dir data/issue_472/geometry/`), run one
     `ssh_execute ls -la <default_path>` on the pod. For each missing
     path: (a) if the brief OR Step 6a.5's carry-over manifest cites an
     HF mirror for the same artifact (parent-task HF data repo
     subdirectory, named in plan §Reproducibility), AUTO-STAGE it via
     `huggingface_hub.hf_hub_download(repo_id=..., filename=...,
     local_dir=<parent_of_default>)` (or, for a directory, scoped
     `list_repo_tree(path_in_repo=<prefix>, recursive=True)` + per-file
     `hf_hub_download` in a ≤6-worker pool — NEVER `snapshot_download`
     against the ~1M-file data repo (or any similarly huge repo): it
     enumerates the full tree before `allow_patterns`;
     `.claude/rules/gotchas.md`) on the pod, then re-stat to confirm it
     now exists; (b)
     if no HF mirror is cited, post `epm:failure v1` with
     ```
     failure_class: infra
     reason: dispatcher-default-path-missing
     missing: <newline-separated list of unstaged default paths>
     note: brief did not enumerate these; dispatcher argparse defaults
           reference them and no HF mirror was cited
     ```
     and EXIT. Re-spawn (cap 3) re-runs this check after the
     orchestrator either updates the brief to enumerate them or wires
     the implementer to add the HF mirror upload to the parent task.
     Incident: task #504 round-1 (2026-06-05) — `dispatch_neg_geometry_504.py`
     defaulted `--persona-bank` + `--centroids-dir` + `--R-train` to
     `data/issue_472/{geometry,on_policy_R}/...` paths that lived on
     HF Hub (parent task #472's data repo subdir) but were never
     staged to pod-504; the dispatcher crashed in ~10 s. Step 6a.5
     PASSed (the HF mirror resolved); the experimenter's item-4 brief-
     enumerated check PASSed (the brief enumerated different paths);
     only an argparse-defaults introspection would have caught it.
   - **Generalize the principle.** Experiment launchers and
     dispatchers MUST fail-loud on incomplete planned coverage —
     never skip-and-continue silently. If you see a dispatcher log
     line like `Skipping pair=X (no rows on disk)` and the run
     continues, that is a bug in the dispatcher AND it MUST also
     trip this pre-launch gate (the gate is the second line of
     defense; the first is the dispatcher itself). If the dispatcher
     swallows a coverage shortfall, post `epm:failure v1` with
     `failure_class: code` and route it through
     experiment-implementer to add the fail-loud check at the
     dispatcher.

   Then, AFTER the coverage gate has PASSed, log a quick content
   sanity sample: (a) total dataset row count summed across the
   verified files, (b) the first 3 examples from one file, (c) the
   file's column names. A coverage gate PASS with garbage contents
   still invalidates the run — both checks are required.

   Rationale: incident task #468 (2026-06-02) ran a full pod cycle at
   n=5 cells instead of the pre-registered n=18 because 13 of the
   per-cell training datasets were not provisioned on the fresh pod;
   the launcher logged `Skipping pair=X (no rows on disk)` per
   missing cell and CONTINUED, so the pipeline completed end-to-end
   and posted `epm:results` at silently-degraded coverage. The plan's
   Reproducibility Card listed 18 cells; one `ls | wc -l` against the
   data directory before launch would have caught the shortfall.
4b. **Verify a persist step exists for every plan-declared output (the
   OUTPUT-side sibling of the item-4 input gate; #1800, incident
   #1739).** Before launch, read the plan's execution design (the
   dispatcher/driver phase chain + any plan-named off-pod/VM-side
   steps) and confirm every plan-declared HF/git-destined output class
   — raw completions, eval JSONs, checkpoints, analysis tensors — has a
   persist step SOMEWHERE in that design: an upload call in the
   dispatch chain, or a plan-NAMED off-pod harvest+upload step (that
   COUNTS as the persist step — the launch chain itself need not carry
   it). Grounding: Upload Policy "Raw completions MUST upload before
   pod termination" + the #779 persist-by-default rule. Disposition
   split:
   - Declared outputs with NO persist step ANYWHERE in the plan's
     execution design AND the run's primary outputs are raw
     completions / generations → REFUSE to launch: post
     `<!-- epm:failure v1 -->` with `failure_class: infra`,
     `reason: no-persist-phase-for-declared-artifacts` (naming the
     orphaned output classes) and EXIT — `/issue` Step 7 routes
     `infra` to a fresh respawn once the chain gains its persist
     step.
   - Any OTHER missing-persist case → WARN loudly and launch: the
     `epm:run-launched` note carries the named line
     `persist-phase: MISSING for <outputs> — launching anyway because
     <one-line reason>`.
   - Persist step present for every declared output → silent (no note
     line).
   Rationale: incident #1739 (2026-07-28) — a GCP `--workload-cmd` run
   completed its phases and approached grace-poweroff with ZERO
   artifacts on HF (all 7 expected prefixes MISS); ~2h of improvised
   recovery uploads raced the poweroff clock. #1779 fixed the
   PLAN-time layer; this gate is the dispatch-time backstop (the
   `dispatch_issue.py` #1800 persist-evidence lint is the mechanical
   sibling on router-lane launches).

## vLLM generate hang detail — same-pod differential triad

### vLLM `generate()` hang — diagnose on the SAME pod BEFORE any reprovision (REQUIRED)

A vLLM `generate()`-class HANG is NOT an OOM and must NOT be met with a
blind `kill + reprovision-new-multi-GPU-pod`. The canonical hang
signature (distinct from a crash — there is no traceback, no Python
exception, no `CUDA out of memory`):

- GPU utilization ~0% across all devices (`nvidia-smi`), AND
- no fresh progress log line (no `[vllm-chunk]` / `[generation]` / phase
  line) for roughly ≥ ~10 min (an approximate operational trigger, NOT a
  calibrated threshold — a long but healthy generation can be silent for
  a while; use it as a "this has been quiet for many minutes" prompt to
  diagnose, not a hard cutoff) while the PID is still alive, AND
- the dispatcher main thread is blocked inside vLLM internals (confirmed
  by the py-spy dump in step (a) below).

It is INVISIBLE to the poller's standard stall detection: the dispatcher
keeps burning ~22% CPU on Python/network thread-pool overhead, so
`session_cpu_secs` advances and the poll loop reports `status=running`
for hours (`.claude/rules/gotchas.md` § the chunked-generate deadlock
entry).

When you observe this signature (on the launch-survival probe, or when
the orchestrator re-dispatches you on a suspected hang), DO NOT terminate
the pod. First run the **differential-diagnosis triad on the SAME pod**
that exhibits the hang — the full recipe + sample commands live in
`.claude/rules/gotchas.md` § "vLLM `generate()` hang — differential
diagnosis BEFORE reprovisioning"; the operative steps:

(a) **py-spy dump** to localize the blocking call:
```bash
ssh_execute(server="epm-issue-<N>",
            command="pip install -q py-spy 2>/dev/null; \
                     py-spy dump --pid <CHILD_PID>")
```
A stack ending in `vllm/.../engine` / `generate` / a CUDA IPC wait
confirms the deadlock class; a stack in OUR code is a different bug
(route it `failure_class: code`, not a hang).

(b) **`enforce_eager=True` probe** — relaunch the dispatcher with
cuda-graph capture disabled (env or flag, e.g.
`EPM_VLLM_ENFORCE_EAGER=1` if the rig exposes it) to rule out a
cuda-graph-capture deadlock.

(c) **disable-prefix-caching probe** — relaunch with prefix caching off
(e.g. `EPM_VLLM_DISABLE_PREFIX_CACHING=1` / the rig's
`enable_prefix_caching=False` path) to rule out a KV-cache pathology.

Only AFTER the triad localizes the cause — and the fix is a round whose
code path is provably reached per `.claude/rules/crash-fix-rounds.md` §
"Crash-fix rounds: declare the fix-engaged signal" — may a `pod.py
terminate + provision-new` be recommended. A kill + reprovision with NO
diagnostics in hand is the banned regression: the #664 saga burned
substantial GPU-hours relaunching an undiagnosed `generate()` hang across
multiple fresh pods before anyone ran py-spy (sessions 2c432067 /
b3489bdb, 2026-06-27).

You do NOT fix the hang in code (that is `experiment-implementer`): if the
triad localizes a code-side cause, post `epm:failure v1` with
`failure_class: code`, the py-spy stack, and the triad findings in the
note, and EXIT — the orchestrator routes it to a fresh implementer round.
Carry the diagnosis forward in the relaunch-with-fix failure-lesson block
below (`gotcha_candidate: yes` for a new pod-driver-specific hang class).
