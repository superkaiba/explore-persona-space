# Step 8: Upload verification

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

Only if status is `uploading` and no `epm:upload-verification` marker
with verdict=PASS.

**Hard gate:** No experiment advances to interpretation until all
artifacts have permanent URLs. This prevents data loss from pod restarts
or cleanup.

**Results-landed parallel spawn (Step 8 ∥ Step 9 pre-compute).** The
upload-verifier dispatch below is no longer a serial prelude to Step 9 —
at this results-landed point the orchestrator spawns up to THREE
background agents concurrently (single message, multiple Agent calls,
staggered a few seconds apart per the CLAUDE.md 429 token-pacing
guidance), each preceded by its own `stage-dispatch` breadcrumb (Step 9
entry guard convention):

1. **`upload-verifier`** (this step, `stage=verifying`) — the hard gate,
   unchanged.
2. **`analyzer` first pass** (Step 9a round 1, pre-computing;
   `stage=interpreting round=1`). The analyzer's inputs (eval JSONs
   under `eval_results/`, figures in the worktree/git, raw completions
   already pulled) exist locally before verification, so it can run its
   full first pass during `uploading`. **HOLD-marker mode:** the
   early-spawn brief instructs the analyzer to write its interpretation
   to `/tmp/issue-<N>-interpretation-v1-held.md` and RETURN WITHOUT
   posting `epm:interpretation v1` — the orchestrator publishes the held
   output (and only then starts the interpretation-critic round) after
   upload-verification PASS. See the two hard joins below.
   **Paper-mode branch (`paper: true` frontmatter).** When the task
   carries `paper: true`, the analyzer runs its PAPER-TASK MODE
   (`.claude/rules/analyzer-paper-mode.md` § PAPER-TASK MODE): the analysis Steps
   1→3.6 are unchanged, but the write-up is a LaTeX **paper** under
   `docs/papers/issue_<N>/` (not a markdown body) — the analyzer
   assembles the `.tex` (splicing in the `methodology-writer`'s Methods +
   Appendix, item 3 below), emits `refs.json` + the figures manifest,
   runs `build_paper.py` → `verify_paper.py`, and writes the `body.md`
   **paper-stub** (`set-body --snapshot` + `set-title` + `set-clean-result`)
   ONLY after `verify_paper.py` PASSes. On `verify_paper.py` FAIL it does
   NOT write a stub — it parks at `reviewing` (or `blocked` with
   `epm:failure v1 failure_class: code`), leaving the `.tex` + build
   `.log`/`.blg` in `docs/papers/issue_<N>/` for iteration. HOLD-marker
   mode still applies: write the held interpretation, RETURN without
   posting `epm:interpretation v1`, the orchestrator publishes after
   upload-verification PASS. The mechanical gate for a paper-task is
   `verify_paper.py`, NOT `verify_task_body.py` (which stays the markdown
   verifier).
3. **`methodology-writer` early spawn (PAPER MODE, or v3/v2
   GRANDFATHERED)** (the early-spawn half of Step 9a-quater;
   `stage=methodology-reference round=1`). **Three cases — branch on
   `paper:` frontmatter FIRST:**
   - **`paper: true` → SPAWN it in PAPER MODE.** The methodology-writer
     authors the LaTeX paper's **Methods section + the recipe Appendix**
     (findings-blind) and hands them to the `analyzer` (item 2), which
     splices them into the `.tex` and runs the build/verify. This is
     load-bearing for paper-tasks — the analyzer does NOT author Methods
     or Appendix itself (the findings-blind firewall is the whole point).
     See `.claude/agents/methodology-writer.md` § PAPER-TASK MODE
     (Methods + Appendix). Fires whenever the 9a-quater kind-gating says
     the step runs (`kind: experiment` always; `kind: analysis` only with
     a methodology surface; `infra | batch | survey` never).
   - **v4 markdown body → SKIP this spawn entirely** — under v4 the
     methodology doc is a POST-PASS mechanical export of the body's
     `## Methodology` section (Step 9a-quater v4 path), so no agent is
     spawned here and the early-spawn batch is just upload-verifier ∥
     analyzer first pass.
   - **In-flight v3/v2 markdown body → SPAWN it in MARKDOWN MODE** (the
     legacy findings-blind path) — a v3/v2 body has no detailed
     `## Methodology` section to copy. Same kind-gating as above
     (evaluate the skip BEFORE spawning).
   The agent is findings-blind by design and its inputs (plan, config,
   reproducibility metadata, verbatim artifact rows) are final the
   moment results land, so it can safely run during `uploading` and the
   interpretation loop. For this early spawn the findings-blind
   Reproducibility input is extracted from the task's `epm:results`
   markers (`reproducibility_card` — alias `reproducibility` — +
   `eval_paths`, via `task.py view <N> --json`) into the temp file —
   the clean-result body's `## Reproducibility` H2 does not exist
   yet. NEVER read only the latest marker: multi-launch runs post
   several `epm:results` markers and a resume-pass sentinel can carry
   an EMPTY card (#601: `adapter_paths: {}`), so resolve each field
   newest-wins among non-empty declarations across markers, matching
   `verify_uploads.py` `merged_results_card` (full recipe: 9a-quater
   procedure step 2). Everything
   publish-side (no-secrets scan, gist, link-append, marker) stays at
   the 9a-quater LATE JOIN; see 9a-quater § Split schedule.

**Two hard joins (both strictly gated on upload-verification PASS):**

1. **Interpretation publish.** `epm:interpretation v1` is NOT posted and
   the interpretation-critic round is NOT started until the verifier
   posts PASS. If the analyzer returns first, hold its output and wait
   for the verifier. The status transition order is unchanged — the
   analyzer merely pre-computes during `uploading`; status flips to
   `interpreting` only on the PASS branch below.
2. **Pod termination.** The teardown call on the PASS branch still
   strictly requires upload PASS — unchanged.

**On upload FAIL → uploader gap-fill: decision rule for the held
analyzer output.** After the gap-fill rounds reach PASS, check whether
the uploader added or changed any artifact the analyzer consumed — eval
JSONs, raw completions, analysis tensors. If YES, the held first pass is
stale: discard it and re-spawn the analyzer first pass before
publishing. If the gaps were only HF-checkpoint / upload-side (no
analysis input changed), proceed with the held analyzer output as-is.

**Re-entry idempotency.** The Step 9 entry guard's `stage-dispatch`
breadcrumbs cover all three dispatches. On a backstop re-entry, apply
the guard PER STAGE (the step-2 per-stage backwards scan in the Step 9
entry guard): do not re-dispatch a stage whose own breadcrumb is within
its freshness window, even when another stage's marker is the latest
event.

Spawn the `upload-verifier` agent with:
- Task number
- Task type (from `body.md` frontmatter)
- Artifact hints from the `epm:results` event (WandB URL, HF paths, pod
  name)
- The `epm:plan` event (for experiment-type metadata)

The verifier runs `scripts/verify_uploads.py` and checks:

| Artifact | Required when | Verified how |
|----------|--------------|--------------|
| Model on HF Hub | Training experiments | HF API |
| Eval JSON on WandB | Always | WandB API |
| Dataset on HF Hub | New data generated | HF API |
| Output generations on WandB | Generation experiments | WandB API |
| Training metrics on WandB | Training experiments | WandB run URL |
| Figures committed to git | Always | `git log` |
| Local weights cleaned | Training experiments | `ssh_execute ls` on pod |
| Claimed URLs HEAD-resolve (phantom-URL gate, #456) | Always | `--claimed-urls-file` HEAD-checks every HF/WandB URL in the `epm:results` marker + body's `## Reproducibility` section at its CITED revision via `orchestrate.hub.verify_artifacts_exist` |
| Primary deliverable produced (completeness gate, #519) | When plan §6.5 declares `primary_deliverable:` | For each `{dv, glob}` row, on-pod `find <glob>` enumerates ≥1 file. Zero files → FAIL with blocker tag `primary-deliverable-missing`. Plans without the §6.5 block (legacy + analysis/infra/batch/survey kinds) get a WARN, not a FAIL. See upload-verifier § Step 2.7. |

**Phantom-URL gate (Step 8 enforcement of upload-verifier Step 2.5).**
Before spawning the verifier, build a single text blob containing the
`epm:results` marker body + the clean-result body's Reproducibility
section, write it to `/tmp/issue-<N>-claimed-urls.txt`, and run
`verify_uploads.py --issue <N> --type <experiment-type>
--claimed-urls-file /tmp/issue-<N>-claimed-urls.txt` so every cited
HF/WandB URL is HEAD-verified at its cited revision. `--type` is the
experiment type handed to the verifier as an input — always pass it
explicitly per upload-verifier.md Step 2.5 (omitting it falls back to
frontmatter-`kind` inference, which conservatively assumes `training`
for `kind: experiment`). A URL string in a
sentinel is NOT evidence the files exist. (#456) See
`.claude/agents/upload-verifier.md` § Step 2.5 for the full rationale.

Post `epm:upload-verification v1` event with per-artifact PASS/FAIL +
URLs. A PASS note MUST carry the literal token `Verdict: PASS` — the
finalize teardown gate matches `UPLOAD_VERIFICATION_PASS_RE`
(`task_workflow.py` `re.compile(r"Verdict:\s*PASS\b")`), and a PASS
note in any other shape is refused as a FAIL at teardown (#1775).

- **PASS** -> teardown the compute, then move status to `interpreting`
  and proceed to Step 9. (Same-issue follow-up round? At
  `followups_running`, SKIP the `interpreting` flip — status-hold rule,
  Step 9b § Same-issue follow-up loop step 3; code-enforced — but the
  teardown + Step 9 progression run as normal.) "PASS" means a fresh
  `epm:upload-verification` verdict for THIS round — posted after this
  round's `epm:results` and after this round's `stage=verifying` dispatch
  breadcrumb. A dispatched verifier with no verdict yet is BLOCKING: do
  not flip status to `interpreting`, do not publish the held
  interpretation, and do not run finalize on a prior round's PASS
  (#778). On a FAIL verdict: uploader gap-fill + re-verify — never
  advance on the FAIL. finalize enforces teardown-side currency
  mechanically (the verifier-currency reasons below); the status flip
  itself is prose-enforced — this paragraph IS that gate. Once artifacts
  are confirmed at permanent
  URLs, the compute is no longer needed — interpretation runs locally.
  If the results-landed parallel spawn produced a held analyzer first
  pass, publish it now: post the held interpretation as
  `epm:interpretation v1` and resume Step 9a round 1 at the
  critic-ensemble spawn instead of re-spawning the analyzer (see Step
  9a § Held-output publish).

  **Backend-agnostic teardown (slice 6).** The dispatch helper persisted
  the per-issue `RunHandle` to `.claude/cache/issue-<N>-handle.json` at
  Step 6b; the orchestrator runs ONE operational call —
  `scripts/dispatch_issue.py finalize` — which reads the sidecar, calls
  `backend.confirm_artifacts(handle)`, and on PASS calls
  `backend.teardown(handle)` — one path for every backend (RunPod /
  SLURM / GCP). The agent-level upload-verifier above runs the
  EXPLORATORY pass; this in-helper `confirm_artifacts` is the
  complementary MECHANICAL gate (HF Hub `list_repo_files` + WandB run
  + git-figure + completion sentinel, per
  `backends.artifacts.confirm_artifacts_from_handle`). Both must pass
  before teardown fires. Degrade path (#585): when the handle
  carries NO `expected_artifacts` declaration — launch paths other
  than GCP do not populate it yet (#598 tracks SLURM; the RunPod
  launch shells `pod_lifecycle.py` and never has) — the mechanical
  gate is structurally unsatisfiable, so finalize falls back to the
  agent-level PASS evidence on the task's events.jsonl (the sticky
  `epm:upload-verified` marker, or the latest `epm:upload-verification`
  with `Verdict: PASS`) and proceeds to teardown with a loud log +
  `"confirm_artifacts": "skipped_no_declaration_agent_pass"` in the
  JSON. Do NOT bypass finalize with a raw `pod.py terminate` on the
  exit-3-missing-declaration shape — that skips the sidecar retirement
  and leaves a stale handle that can mis-target a later finalize; run
  the upload-verifier to a PASS, then re-run finalize. With neither a
  declaration nor agent PASS evidence, finalize still exits 3
  (`reason: confirm_artifacts_no_declaration`).

  **Verifier-currency gate (#1026).** The agent-level PASS evidence must
  be CURRENT — finalize refuses (exit 3, teardown skipped, sidecar not
  retired) on every non-skip path with one of five typed reasons, each
  with its routing action: `upload_verifier_in_flight` (a dispatched
  verifier round has no verdict yet, liveness window fresh → WAIT for
  the verdict; on PASS re-run finalize, on FAIL gap-fill + re-verify,
  never finalize on a FAIL), `upload_verifier_stalled` (window lapsed,
  no verdict → re-spawn the upload-verifier to a verdict, then finalize
  on PASS), `upload_verification_ambiguous` (a late verdict cannot be
  attributed to the current results-epoch → re-run the verifier; the
  fresh round resolves it), `upload_verification_stale` (the latest
  verdict predates the newest `epm:results` → re-verify),
  `upload_verification_failed_current` (the latest verification is a
  FAIL → gap-fill + re-verify). An in-flight verifier is never
  PASS-equivalent; absence-of-verdict never satisfies the gate.
  **Named residual:** the crumb-based rules presuppose the Step 9
  entry-guard convention that a `stage-dispatch` breadcrumb precedes
  each verifier spawn (the missed-breadcrumb limitation — see the
  "Limitation (be explicit about it)" paragraph under the Step 9 entry
  guard) — a verifier spawned WITHOUT its breadcrumb is invisible to the
  in-flight/stalled rule; the stale + FAIL-current rules are the
  backstops for that case.

  **Phase-scoped-launch mismatch (#604).** The launch-time
  auto-declaration assumes the FULL task artifact set (hydra-lane
  launches: HF `issue<N>_<attempt>/raw_completions/` + git
  `eval_results/issue_<N>/` + `figures/issue_<N>/`; `--workload-cmd`
  launches auto-declare only the sentinel + git paths — the guessed HF
  prefix was dropped after it false-FAILed a perfectly-uploaded run
  whose driver used its own `issue<N>_<slug>/` contract prefix, #601
  follow-up r1; HF-data coverage on that lane comes from the
  agent-level upload-verifier), so a launch covering only ONE phase of a
  multi-phase plan (e.g. an extraction phase whose sole deliverable is
  an `analysis_tensors/` bundle) FAILs `confirm_artifacts` on declared
  paths that only the plan's LATER (VM-local) phases produce. A
  declaration that is PRESENT but phase-mismatched is structurally
  unsatisfiable until end-of-task, and the agent-pass fallback above
  never fires (it is gated on the declaration being ABSENT) — finalize
  exits 3 (`reason: confirm_artifacts_failed`) by design. Do NOT leave
  the instance idling until the later phases land (#604 burned ~70 idle
  minutes on a g2-standard-4): mechanically verify the launch's ACTUAL
  phase deliverable on permanent storage first
  (`huggingface_hub.list_repo_files` for HF paths — never the `hf`
  CLI), then re-run finalize with the gate skipped —
  `dispatch_issue.py finalize --issue <N> --skip-confirm-artifacts` —
  which still runs the backend teardown AND retires the sidecar to
  `<name>.finalized` (no stale handle; do NOT substitute a raw `gcloud
  compute instances delete` / `pod.py terminate`, which skips the
  retirement). Post `epm:pod-terminated v1` naming the declaration
  mismatch + the verified deliverable paths. Distinguish the two exit-3
  shapes: no-declaration → upload-verifier-to-PASS + plain re-run
  (above); present-but-phase-mismatched declaration → verify the phase
  deliverable, then `--skip-confirm-artifacts`. The skip flag does NOT
  bypass the verifier-currency gate for a FRESH in-flight verifier round
  (exit 3 `upload_verifier_in_flight` — wait for the verdict, or for the
  15-min window to lapse to `stalled`); stalled / stale / ambiguous /
  failed-current records degrade to a loud warning + a `verifier_warning`
  field in the success JSON.

  ```bash
  # ONE call for every backend. Exit 0 = confirm PASS + teardown done;
  # exit 3 = confirm FAIL or verifier-currency refusal (reason ∈
  # confirm_artifacts_failed | confirm_artifacts_no_declaration |
  # upload_verifier_in_flight | upload_verifier_stalled |
  # upload_verification_ambiguous | upload_verification_stale |
  # upload_verification_failed_current) — teardown SKIPPED, evidence
  # preserved; exit 2
  # = missing sidecar (treat as infra failure).
  #
  # CAVEAT — parent-pod-reuse child tasks: when this child task ran on
  # the parent's RunPod via the alive-parent branch in Step 6b, NO
  # sidecar was written for the child. SUBSTITUTE this call with
  # `pod.py terminate --issue $PARENT_ID --yes` (per the "Slice-6
  # regression guard for the parent-pod-reuse branch" paragraph in
  # Step 6b); the finalize CLI would otherwise exit 2 on the missing
  # child sidecar.
  uv run python scripts/dispatch_issue.py finalize --issue <N>
  ```

  On the RunPod path the underlying `RunPodBackend.teardown` shells
  out to the same `scripts/pod.py terminate --issue <N> --yes` that
  today's wiring uses (the wrapper preserves the existing guard logic
  verbatim); on the SLURM path it `scancel`s via the robot SSH alias;
  on GCP it `gcloud compute instances delete`s. Post
  `epm:pod-terminated v1` with the teardown summary (for the GCP path
  the marker name still applies — the dashboard surfaces every
  backend's teardown under the same key).

  If interpretation later needs GPU compute (e.g., to regenerate a
  figure from raw outputs that weren't downloaded), dispatch fresh
  compute through the slice-6 router — read the task's `backend:`
  frontmatter and run `dispatch_issue.py launch --issue <N> --intent
  "$INTENT" ${BACKEND:+--backend "$BACKEND"}` per Step 6b's
  "Operational dispatch (slice-6 router, ALL backends)" block (empty
  frontmatter → auto routing — RunPod first (#2054), then fellows +
  the free SLURM lanes; GCP provisioning disabled, #2028). If the task has `parent_id`, terminate
  the parent's pod (`epm-issue-<PARENT_ID>`) instead. Skip the
  teardown call only if the task has a `keep-running` tag for known
  follow-up work in the same session. (Mechanically enforced as of
  #1485: `pod.py terminate --issue N` bare form REFUSES on the tag —
  surgical `--name-suffix` destroys stay allowed, `--force-keep-running`
  is the deliberate operator override — and `dispatch_issue.py finalize`
  skips its teardown leg with rc 0 / `phase: teardown_skipped`; remedy:
  `task.py remove-tag <N> keep-running`, then re-run.)

  **VM download-cache cleanup (post-#disk-100pct).** The experiment
  downloaded its source data from HF into VM-local caches under
  `data/issue_<N>/hf_dl/` + `data/issue_<N>/g*_dl/` — in the repo-root
  `data/` AND in this issue's worktree
  (`.claude/worktrees/issue-<N>*/data/issue_<N>/`, where the live run
  usually writes). Nothing
  else reclaims them, and a single finished experiment can pin ~100 GB
  on the VM root disk (`/` has hit 100% full). These are
  re-downloadable CACHES (no on-HF presence check needed), and `store/`
  + `eval_results/` are NEVER touched (in repo-root OR worktrees). After
  the teardown above (artifacts are now confirmed at permanent URLs),
  clean this issue's download caches — the helper sweeps both the
  repo-root and worktree copies:

  ```bash
  # Re-downloadable hf_dl/g*_dl caches only (repo-root + worktree);
  # store/ + eval_results/ kept.
  uv run python scripts/clean_experiment_downloads.py <N> --apply
  ```

  Auto-continue (NOT a gate); idempotent — a re-entry on an
  already-cleaned issue is a no-op. The fleet-wide backstop for caches
  that escape this path (crashed runs, follow-up rounds) is the
  `vm_disk_guard.py` cron (CLAUDE.md § Disk hygiene). Skip only when the
  task has a `keep-running` tag (the same-session follow-up may re-use the
  cache).

  **Incremental (between-phase) cleanup for multi-phase runs.** Step-8
  cleanup fires only at experiment END, so a multi-phase experiment whose
  phases each materialize a fresh download cache holds the PEAK of all
  phases' caches at once — and a large-footprint phase can fill `/`
  mid-run (#658). When a
  run has multiple phases that each download inputs (e.g. a phase's judge /
  extraction step CONSUMES its `e0_gen` / `g*_dl` / `hf_dl` inputs, then
  the next phase downloads more), reap each consumed phase's
  re-downloadable cache BEFORE the next phase materializes more — bounding
  peak footprint, not just cleaning at the end. Between phases (after the
  judge / extraction consumes the phase's inputs, before the next phase
  downloads):

  ```bash
  # Between-phase: reap THIS phase's consumed hf_dl/g*_dl cache (repo-root
  # + worktree); store/ + eval_results/ kept; no terminal-status gate (the
  # run knows the phase is done). Legal ONLY after the cache's LAST consumer
  # in the WHOLE run: only hub-download paths re-fetch on a miss — a
  # direct-path open() reader crashes FileNotFoundError (#1489; see
  # .claude/rules/gotchas.md).
  uv run python scripts/clean_experiment_downloads.py <N> --incremental --apply
  ```

  Same safety contract as the Step-8 cleanup (re-downloadable caches only;
  `store/` + `eval_results/` NEVER touched; read-only on task state). This
  is the RUNTIME backstop that bounds peak footprint — but it does NOT
  rescue a single phase whose OWN footprint exceeds the disk; such a phase
  must be ROUTED off the VM at plan time per the data-footprint carve-out
  (CLAUDE.md "CPU-only phases don't hold GPU pods" → `planner.md` §9 →
  `critic.md` Methodology lens item 10).

  **Upload-verification guard (post-#444).** `pod.py terminate` refuses
  to destroy an `epm-issue-<N>` / `pod-<N>` for a `kind: experiment`
  task unless an `epm:upload-verification PASS` marker exists on task
  `<N>` — this catches resume-launcher / hand-orchestrated completions
  that skipped the verifier. The normal Step 8 path posts the PASS
  marker BEFORE calling terminate, so the gate is silent on the happy
  path. If you must terminate without running the verifier (e.g. the
  experiment crashed before producing artifacts, or you've manually
  confirmed every URL landed), pass `--skip-upload-verify` — it logs a
  LOUD warning and still proceeds. NEVER substitute a manual partial
  upload check for the verifier on a normal-completion path; the
  verifier's checklist is the safety net against silent dataset /
  checkpoint loss (#444 lost the training-mix datasets
  after a hand-driven completion did a partial check and terminated).
- **FAIL with blocker tag `primary-deliverable-missing`** (Step 2.7
  completeness gate, post-#519) -> the headline phase that produces the
  Goal's primary dependent variable silently did not run on the pod
  (e.g. missing input flags fell through an `if args.X and args.Y`
  guard, a phase crashed mid-loop with the dispatcher recording
  `skipped_phases: []`). The uploader cannot fix this (there is no
  artifact to upload), and terminating the pod destroys the cheap-fix
  window (the pod and any per-step checkpoints still exist; re-running
  the missing phase in-place is far cheaper than re-provisioning +
  re-training from scratch).

  **Auto-recover, don't park.** Consistent with CLAUDE.md "Continuing on
  your own is the default" + `workflow.yaml § pivot_criteria`, do NOT
  call `pod.py terminate`, do NOT dispatch the uploader, do NOT flip to
  `status:blocked`. Instead loop back to the run phase on the
  still-alive pod and re-drive the missing primary deliverable:

  1. Read the verdict body's `Missing / required action` list to
     identify the missing DV name(s), the missing glob(s), and the
     pod-side phase that produces them (the planner's §6.5 row + the
     §4 Design pipeline together name the responsible dispatch
     entrypoint).
  2. Flip status back to `running` (`task.py set-status <N> running`).
     (Same-issue follow-up round? At `followups_running`, SKIP this flip —
     status-hold rule, Step 9b § Same-issue follow-up loop step 3;
     code-enforced — and re-enter the dispatch path with the status held.)
     Then re-enter the Step 6d experimenter-dispatch path with an
     explicit re-run scope naming the missing phase + the inputs that
     fell through (typically: re-dispatch the same entrypoint with the
     corrected `--<phase>-inputs <path>` flags that the silent guard
     consumed). Post a `epm:progress` note recording the pivot:
     `auto-recover: primary-deliverable-missing for <DV>; re-running <phase> on pod <pod-name>`.
  3. The experimenter dispatches as usual, posts `epm:run-launched` /
     `epm:run-finished` / `epm:results`, and Step 8 re-runs
     upload-verification on the next /issue tick.
  4. Re-verification is mechanical: if `find <glob>` now enumerates
     ≥1 file the row PASSes and the gate clears; if it remains zero
     after a re-run that ITSELF says it ran (exit 0 + a non-empty
     manifest for the phase), that is a NEW failure class — the
     dispatcher claims success while producing nothing — and counts
     as a fresh strategy attempt.

  Treat each auto-recovery attempt as one strategy iteration. The
  generic halt path applies normally:
  `workflow.yaml § pivot_criteria` (specifically `infra_respawn_cap_3`,
  and after ~3 fundamentally different strategies have all FAILed AND
  no further autonomous angle exists) is the ONLY route to
  `status:blocked` for this failure class. Do NOT introduce a dedicated
  halt for the first or second `primary-deliverable-missing` FAIL.

- **FAIL (any other blocker)** -> dispatch the `uploader` agent (up to
  3 rounds) to close the gaps. The uploader receives the verifier's
  missing-artifacts list, lifecycle-aware resumes the pod if needed,
  pushes to HF / WandB / git, and posts `epm:upload-fix v1`. After each
  uploader round, re-run `upload-verifier`; it posts a fresh
  `epm:upload-verification v<N+1>`. Any gap-fill (uploader- or
  orchestrator-side) that MATERIALIZES a missing run artifact from
  markers must reproduce the experiment writer's exact schema, or write
  a `<name>.materialized.json` sidecar (#1775; full rule: uploader.md
  § Rules).

  Round outcomes:
  - **uploader COMPLETE + verifier PASS** -> proceed as PASS branch above.
  - **uploader BLOCKED** (e.g., RunPod host capacity, missing
    credentials) -> stays at `uploading`. Post the uploader's
    `epm:upload-fix` event with the blocker. Post the §5 marker:
    ```bash
    uv run python scripts/post_step_completed.py --issue <N> --step 7 \
      --exit-kind failure-exit \
      --notes "uploader BLOCKED; awaiting operator action"
    ```
    EXIT, await operator action.
  - **3rd round still FAIL** -> status to `blocked`. Post the §5 marker:
    ```bash
    uv run python scripts/post_step_completed.py --issue <N> --step 7 \
      --exit-kind failure-exit \
      --notes "uploader exhausted 3 rounds; see upload-fix v3"
    ```
    EXIT (mirror the code-reviewer FAIL escalation in CLAUDE.md).

  See `.claude/agents/uploader.md` for the uploader's contract and the
  marker schema. The uploader NEVER terminates pods; only stops/resumes.

#### Step 8-bis: Pod must not idle on a halt

Step 8's terminate fires only on the NORMAL upload-verification-PASS path.
A pod can still be left RUNNING-and-billing whenever the pipeline leaves
that path: (a) it blocks on a human-input gate that cannot be satisfied
this turn (e.g. `epm:fact-pick` at Step 6, the plan-approval / merge gates,
or any STATE-TO-`blocked` exit), or (b) it is detected crashed/dead with
GPUs idle. Before EXITing the turn in EITHER case, if an `epm-issue-<N>`
pod (or the parent's pod for a follow-up) exists and is RUNNING, run
`uv run python scripts/pod.py stop --issue <N>` (volume preserved; `resume`
re-provisions) — or `terminate --issue <N> --yes` when the work is truly
done — and post `epm:pod-stopped v1` / `epm:pod-terminated v1` with the
command output. A gate/crash park routinely outlasts an hour, and a STOPPED
volume is NOT durable (provider-side loss despite `keep-running`, #1112 —
`.claude/rules/pod-config.md` § "Stopped pod volume is NOT durable"): BEFORE
the stop, push the run's resume state (done-JSONs / phase sentinels /
partial eval JSONs — KB–MB text) to the issue's HF prefix so a later resume
restarts from off-pod copies even if the volume is gone. Skip only when the
pod demonstrably holds no unpersisted resume state (state it in the
`epm:pod-stopped` note either way). NEVER leave a pod RUNNING while awaiting human input or
after a crash. (#444 idled a 4×H100 on an unfired gate, #404 after
Step 8 never fired, #407 after an `aggregate`-phase crash — days of
idle burn combined.)
