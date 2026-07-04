---
description: Pod-side dispatcher result-reporting contract (sentinel files, poll_pipeline.py drain, epm:results payload) + legacy pod-side preflight gates; relocated verbatim from experiment-implementer.md, #829
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
marker will be silently skipped. Two requirements, no exceptions:

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
   - `version`: marker version integer.

   The marker body goes under `note` (or the `payload` synonym).
   Recommended optional keys: `task_id`, `gate`, `blocks_pipeline`,
   `by`, `ts`. A bare `schema` key (or any other re-spelling of
   `sentinel_schema_version`) trips the `missing required keys` warning
   in `_parse_sentinel` and the sentinel is skipped without being
   renamed `.processed` — the marker never lands, the dashboard never
   updates, and the orchestrator advances without the experiment's
   results in `events.jsonl`.

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
