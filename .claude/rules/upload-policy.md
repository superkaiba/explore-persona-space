---
description: Deep upload mechanics — Hub-API verification gotcha, inline-upload fence, delete-after-eval adapter-persist recipe (loads when writing training / hub / sweep code)
paths:
  - "src/explore_persona_space/orchestrate/**"
  - "scripts/train.py"
  - "scripts/run_sweep.py"
  - "src/explore_persona_space/train/**"
  - "scripts/issue*.py"
  - "scripts/issue*.sh"
---

# Upload mechanics (deep)

The always-on **Upload Policy** in CLAUDE.md carries the destination table + the
core rules (models upload to HF before local deletion; `eval_results/` is
JSON/text only; raw completions + plan-referenced analysis tensors before pod
termination; datasets upload; clean local weights after; WandB = live training
metrics only). The deep mechanics below load when you touch training / hub /
sweep code.

**HF Hub uploads are accelerated by DEFAULT (#745).** Two orthogonal env vars
are on by default in every experiment-upload environment:
`HF_XET_HIGH_PERFORMANCE=1` (the PRIMARY accelerator — both project repos are
Xet-backed) and `HF_HUB_ENABLE_HF_TRANSFER=1` (the orthogonal LFS-multipart
accelerator; `hf_transfer` is a hard `pyproject` dep). They are set at SHELL
level in `bootstrap_pod.sh` (pod), the GCE startup prelude (`backends/gcp.py`),
and the SLURM sbatch env block (`backends/slurm.py`) — the load-bearing
placement, because `huggingface_hub.constants` freezes env at import time —
plus an `orchestrate/env.py` `setdefault` for local-dev. Override per-launch
with `=0` / `HF_HUB_DISABLE_XET=1`: the defaults are `setdefault` / `${VAR:-1}`
so an explicit launch-time `=0` always wins, and the GCP/SLURM passthrough
allowlists forward the two `=0`s AND `HF_HUB_DISABLE_XET=1` (#1195). RunPod is
NOT part of that claim — pods have no dispatch-env passthrough, so on a pod the
kill switch is set in the WORKER shell. The effective xet kill switch is
`HF_HUB_DISABLE_XET=1` — it flips `is_xet_available()` False
(`huggingface_hub` 0.36.2, the uv.lock pin), gating the upload branch;
download-side coverage has a reported gap on this pin (hub GH issue #3266), so
treat it as upload-verified. `HF_XET_DISABLE=1` (the #515 download workaround;
retained in the lane allowlists only as an annotated legacy alias, #1195) is a
VERIFIED NO-OP on this stack — consumed by neither `huggingface_hub` nor the
`hf_xet` Rust binary — so a recipe leaning on it never left the xet path
(#931's first two wedge replays did exactly this). Upload sitting at ~0 TX?
Run the wedge escalation ladder in the next block. A NEW direct-upload script
must use the project `explore_persona_space.orchestrate.env.load_dotenv`
wrapper, NOT the bare `from dotenv import load_dotenv` (enforced by
`scripts/workflow_lint.py --check-dotenv-before-hf-import`).

**Pod→HF upload WEDGE — recognize it, then run the three-rung escalation
ladder (#931).** (The DOWNLOAD-side native `xet_get` hang has its own
kill-and-replay entry in `.claude/rules/gotchas.md`; #1345. Socket COUNT does
NOT discriminate the two wedges — #1739 download hangs held exactly ONE
socket, the same shape as this ladder's frozen ESTAB socket; #2153.)
Signature: the
upload process looks healthy (no traceback) while transfer bytes stop —
interface TX delta ~0 across two samples ≥5 min apart
(`cat /sys/class/net/eth0/statistics/tx_bytes`), and/or one ESTAB socket to
the CDN (port 443) whose counters are frozen in `ss -tinp` (`bytes_acked` /
send-q not advancing; `apt-get install -y iproute2` if `ss` is absent). High
sustained CPU with ~0 TX can be legitimate local pre-processing (xet
chunking / sha256 of multi-GB files) — the frozen-ESTAB-socket check is the
discriminator; once the signature confirms on a re-sample, escalate
immediately (#931 sat ~30 min at ~0 TX before the first kill). Three
preconditions: (a) the upload path is replay-idempotent (per-cell /
per-folder skip-if-complete — the #664 per-cell contract), (b) each rung is
KILL-hung-process → REPLAY-with-env — never export on top of a live process
(`huggingface_hub.constants` freezes env at import), (c) for a LIVE wedged
process the rung env must be set IN THE WORKER's shell and the process
relaunched there (an orchestrator-side export can never reach a running
process; RunPod has no dispatch-env passthrough at all); a full FRESH
re-dispatch with `HF_HUB_DISABLE_XET=1` in the dispatch env DOES forward to
GCP/SLURM workers (#1195). Do not wait for a rung to self-heal: every retry
layer (hf_transfer part retries, `http_backoff`, the xet timeout knobs)
fires only on RAISED errors — a silently hung ESTAB read never becomes an
error, so detection + kill is always manual.

1. **Rung 1 — kill + replay with `HF_HUB_DISABLE_XET=1`** (the REAL switch —
   NOT the no-op `HF_XET_DISABLE`). Targets a xet-client-specific stall
   (hung CAS read, finalization hang — #825 r2's class); the upload falls
   back to the LFS multipart path, hf_transfer-accelerated.
2. **Rung 2 — wedged identically? kill + replay with `HF_HUB_DISABLE_XET=1
   HF_HUB_ENABLE_HF_TRANSFER=0`** — the pure-python requests path. Rung 2
   without rung 1's var is a placebo on the project's xet-backed repos:
   while xet is available the upload never reaches the LFS path where
   hf_transfer lives.
3. **Rung 3 — still wedged? The on-pod upload path is dead for this run;
   reroute around it.** rsync the artifact dirs pod→VM (rsync IS on
   bootstrapped pods — `bootstrap_pod.sh` Step 2; a `--no-bootstrap` pod
   needs the tar-over-ssh form in the #541 recovery below), verify the
   VM→HF route with a small probe upload, run the VM-side `upload_folder`
   to the SAME `path_in_repo`, then a pod-side local-only sentinel replay
   so `epm:results` lands via the normal poller drain (#931 moved ~9.9 GB
   this way in ≈24 min after three wedged on-pod attempts). If the VM→HF
   probe ALSO wedges (an HF-side incident), the pod→VM rsync has already
   made the data durable: stop/terminate the pod rather than idling it, and
   retry the VM→HF upload when the incident clears.

Honesty caveat: #931's rung-1/2 replays set the no-op alias, so all three
on-pod attempts likely ran the SAME xet client — rung-1/2 value is derived
from the 0.36.2 code paths + HF's documented legacy-LFS fallback, not yet
proven in anger; the route-level rung-3 reroute is the proven recovery. On a
known org-wide 429/CDN-incident day, consider going straight from one
confirmed rung-1 wedge to rung 3.

**The reader-back of the `issue<N>_partial/` crash-persist path is
`autonomous_session_watch.partial_bundle_pass` (#1704, motivating incident
#1345)** — an ESCALATE-ONLY hourly audit that reconciles bundle contents
against the committed `eval_results/issue_<N>/` tree in git and flags any
bundle carrying a completed result whose payload has no committed
counterpart (sidecar `.claude/cache/partial-bundle-events.jsonl` + one
deduped Telegram push per (issue, attempt_id, band); NEVER auto-commits,
NEVER deletes, NEVER posts task markers). Full predicate: § "Autonomous-
session watcher" in `.claude/rules/background-automation.md`.

**Intermediate analysis tensors referenced by the plan MUST upload before pod
termination.** Any artifact the plan's analysis / negative-control sections
name as a downstream input — per-cell shift tensors (`shifts/*.pt`), cached
activations, decomposition / SVD inputs — uploads to the HF data repo under
`issueN_<slug>/analysis_tensors/` BEFORE the pod is terminated, exactly like
raw completions. These files are typically tiny (KB-MB) next to the
checkpoints they derive from, which makes them easy to dismiss as scratch —
but losing them makes the plan's remaining controls permanently unrunnable
(#521: ~200 KB per-cell Δv `.pt` files required by two planned negative
controls were never uploaded; upload-verification still PASSed, the pod was
terminated, both controls became unrunnable). Enforcement: `upload-verifier` Step 1 classifies
`*.pt` / `*.npy` as analysis tensors bound for the HF data repo, and its
Step 2.8 cross-references the plan's analysis / control sections and FAILs on
any plan-named input without a permanent URL.

**Persist by default; a discard needs a recorded justification (#779).** The
always-on Upload Policy states the principle; the mechanics: **text/JSON
uploads unconditionally** (rollout text, judge outputs, metrics, configs are
non-LFS in the data repo — the #541 quota gate fires ONLY on LFS, so this path
stays open over quota; text <9.5 MB uploads as-is, bigger text line-splits into
<9 MB shards, NEVER gzip — `*.gz` is LFS-matched, and the Hub force-routes any
>10 MB blob to LFS regardless of extension; shard pieces — `.shardNN.jsonl` or
`.part*` — are line-split FRAGMENTS of one file: concatenate before `json.load`,
a lone `.part000` is not standalone JSON, 2026-08-02). **Large tensors upload when cheap;
when too big for LFS at current headroom, persist the TEXT they were derived
from** so the tensor is regenerable via one teacher-forced forward pass — this
is the size-aware form of persist-by-default, and it composes with the #541
overflow routing below (the LFS artifact routes to the private overflow repo
when known-over-ceiling; its regenerating text stays on the public non-LFS
path). A DELIBERATE discard — a candidate ONLY for a large intermediate TENSOR,
never text/JSON — is declared in the plan §10 `discarded_artifacts:` slot
(`{name, reason, regen_recipe}`); the upload-verifier FAILs a model-generation
discard whether undeclared (`generation-discarded-undeclared`) or invalidly
declared via a text-naming entry (`generation-discard-declared-invalid`) — its
Step 3 generation-discard gate. Stream-reduce memory-safety (RunningMean /
`_HfStreamSpanSource`) is UNCHANGED — it persists the rollout text it reduced;
it does not re-materialize the whole activation grid (#666/#772). (Driving
incident: #779's extraction driver reduced kept rollouts to `r_B` and dropped
the rollout text, so a sibling arm had to regenerate.)

**Uploader eligibility filters must cover every plan-declared artifact class
(#825).** An upload helper that enumerates files through an eligibility
filter — `upload_folder(allow_patterns=[...])` / `ignore_patterns=[...]`, a
custom `p.match(pat)` loop, an extension allowlist feeding `create_commit`
ops — silently DROPS every artifact class the filter does not name: the
upload "succeeds", the sentinel posts, and the class surfaces as missing
only at the upload-verification gate (or never, if the instance is already
gone). Producer duty: the UNION of a run's upload-eligibility filters
covers every artifact class the plan declares as persisted — each §6.5
`primary_deliverable:` row's path/glob and every §10 per-stage output
destination (`raw_completions/<stage>/`, `analysis_tensors/`, eval JSONs) —
with plan §10 `discarded_artifacts:` entries as the ONLY
declared-not-uploaded exemption. When a run writes a NEW file kind beside
existing ones (a `row_index*.jsonl` beside `*.npy` payloads), extend the
filter in the same change that adds the writer. (#825: an allowlist of
`**/*.npy` + `**/*.json` left all 404 plan-declared `row_index*.jsonl`
files never upload-ELIGIBLE; remediation succeeded only because the GCE
instance was still alive — on the ephemeral
`--instance-termination-action=DELETE` lanes a crash loses the class
outright.) This is the UPLOAD-side sibling of the download-side
`snapshot_download(allow_patterns=...)` gotcha
(`.claude/rules/gotchas.md`). Enforcement: author-side self-check in
`experiment-implementer.md` § After implementation step 7; review-side
parity sub-check in `code-reviewer.md` Step 0.65 (Critical, tagged
`substantive`); the upload-verifier at Step 8 stays the last-line safety
net, not the only line of defense.

**Regenerating a published artifact in place requires a version-bumped path or
a regeneration note (#922/#779).** Re-uploading / reconstructing an
already-published artifact at the SAME path can silently invalidate every
capture another task made under the original bytes — activations,
teacher-forced reads, judge outputs, adapters trained on the mix. Each pair
member still resolves and sha-verifies individually, so only the
consumer-side pairwise provenance-coherence check
(`.claude/rules/artifact-reuse.md` item (j)) detects the incoherence — after
the fact, at the cost of a wasted run. Producer duty, one of two forms:

1. **Version-bump the path** — publish the regenerated artifact at a NEW path
   (`issueN_<slug>/v2/...`, or a new filename), so the original path keeps
   resolving to the bytes existing captures were made under. Prefer this form
   whenever a dependent capture is known or plausible.
2. **Record a regeneration note the artifact itself carries** — a
   `reconstruction` / regeneration metadata field inside the artifact (or a
   sidecar `<name>.regeneration.json` uploaded in the same commit) stating the
   regeneration date, the reason (a bug-fix regeneration invalidates the old
   bytes; a byte-equivalent rebuild does not), and any KNOWN dependent
   captures (task ids / capture paths). Item (j) already reads exactly this
   field (#922's question artifact documented its own regeneration); the note
   is what lets a consumer choose between item (j)'s two remedies —
   re-capture under the current input, or pin the input at the
   pre-regeneration revision. This form is the floor when the path must stay
   stable (a canonical bucket consumers resolve by convention).

(Incident: #779 regenerated published question artifacts in place AFTER
#922's dependent activation capture; every per-member check passed and the
run crashed at a parity assert after a full GCE cycle.)

**Resume-critical pipeline INPUTS — and the run's RESUME STATE — must
upload before any deliberate `pod.py stop` that expects a later resume: a
stopped volume is NOT durable.** The same logic extends
upstream of analysis: generated training rows (`R_train` caches,
corpus JSONs), phase-0/1 intermediate outputs, and diagnostic adapters
that the plan's later phases consume. RunPod `resume` is HOST-PINNED —
a SUPPLY_CONSTRAINT on the former host can lock the volume away for
days, and a fresh pod cannot substitute when the inputs exist only on
that volume. Push them to the HF data repo (`issueN_<slug>/inputs/` or
the relevant bucket) BEFORE stopping; they are usually MB-scale (#488:
~18 resume attempts hit SUPPLY_CONSTRAINT while the training rows +
phase outputs + diagnostic adapters lived only on the stopped volume).
Resume STATE means done-JSONs, phase/resume sentinels, partial eval JSONs,
progress manifests — anything a resume reads to know where to restart.
Host-pinning is not the only threat: RunPod destroyed a stopped pod
outright despite the `keep-running` tag and well inside the 7-day idle
window (#1112: done-JSONs lost, full re-run forced). Apply whenever the
park may outlast ~1 hour; on resume,
prefer the off-pod copies. Decision-point recipes:
`.claude/skills/issue/SKILL.md` § User pause affordance step 1 +
§ Step 8-bis; canonical rule: `.claude/rules/pod-config.md` § "Stopped pod
volume is NOT durable".

**Verify uploads with the Python Hub API, never the `hf` CLI.** The installed `hf`
CLI has NO `api` subcommand — `hf api list-repo-files ...` errors to stderr and
`| grep` swallows it as an empty/zero result that reads as a false "0 files"; `hf
repo-files` only exposes `delete`, not `list`. Use:
`set -a && source .env && set +a && uv run python -c "from huggingface_hub import HfApi; print('\n'.join(e.path for e in HfApi().list_repo_tree('superkaiba1/explore-persona-space-data', path_in_repo='<bucket>', repo_type='dataset', recursive=True, revision='main')))"`
(scoped `list_repo_tree` — a bare `list_repo_files` full listing of the
~1M-file data repo times out (>90 s, #833); § Relocated codebase traps below)
(the `set -a && source .env` prefix is part of the canonical snippet — without
it the check dies on `HF_TOKEN missing`, and the obvious in-heredoc fix, a bare
`load_dotenv()`, crashes from stdin)
(the prefix is VM-scoped — repo root, where `.env` always exists; a pod/GCE
workload script must source conditionally instead — `if [ -f ./.env ]; then
set -a; . ./.env; set +a; fi` — because the GCE lane exports tokens via its
startup script and has NO `.env` file; `pod-side-reporting.md`, #923)
(#458 nearly drew a wrong "checkpoints don't exist" conclusion from the
silent CLI "0").

Consumers of this snippet beyond post-experiment upload verification:
`follow-up-proposer` runs it as a hard gate to verify reuse premises before
tagging a follow-up `auto_run: yes` (see `.claude/agents/follow-up-proposer.md`
§ artifact-premise verification); `analyzer` runs it at clean-result write time
to ground every path-specific `**Artifacts:**` claim in a live listing (see
`.claude/agents/analyzer.md` Artifacts-grounding rule); and `clean-result-critic`
Lens 5 spot-checks an artifact path from the body against the same listing. All
three rely on the Python Hub API for the same reason — the `hf` CLI's false "0"
would corrupt their checks identically. Keep the snippet (repo, `repo_type`,
`revision`) consistent across these surfaces when editing.

**Verify-path Hub calls ride `retry_transient` + ONE prefix-scoped listing per
destination repo (#1335 r5).** A post-upload verify is still part of the run:
a transport error there (429 / 5xx / timeout / connection) is retried, never
fatal (#1335 r5: an UN-retried per-shard `api.file_exists` HEAD probe let one
transient HF 429 crash a healthy GCP run 2.8 h in, AFTER every upload had
succeeded). Two rules for any upload/verify path in workload code: (a) wrap
every FRESH Hub call in `hub.retry_transient` (`orchestrate/hub.py` — the
public alias of `_retry_upload`: Retry-After-aware, wall-clock-budgeted via
`EPM_HF_RETRY_BUDGET_S`; storage-quota-403 and other non-transient errors
still re-raise immediately); (b) verify a SHARDED upload with ONE
prefix-scoped listing per destination repo — collect the shard paths and
check the SET via `hub.verify_repo_paths_uploaded(...)` — never a per-shard
`file_exists` / exact-file probe loop (N per-file probes multiply transport
exposure N-fold). The canonical sharded implementation is
`upload_sharded._batched_verify` (#1335), superseding the per-shard
`_verify_present` probe loop. Pin new verify code with a 429-then-success
retry test and a ≤2-listings batching test (`tests/test_upload_sharded.py`).

**Staging-DOWNLOAD legs use the canonical helpers `hub.stage_hub_file` /
`hub.stage_hub_prefix` (#1402) — never a hand-rolled retry + tempdir move.**
The download-side sibling of the verify-path rule above: both helpers ride
`retry_transient`, which classifies `LocalEntryNotFoundError` transient BY
CLASS, checked first (a 429 storm on `hf_hub_download`'s HEAD surfaces
404-shaped through that response-less error; a genuinely-missing file still
fail-fasts via its response-bearing 404 `EntryNotFoundError`).
`stage_hub_file` is atomic (tempdir INSIDE the dest parent + `os.replace` —
the #1335 EXDEV gotcha) and fail-loud; `stage_hub_prefix` is the #833
scoped-listing recipe (server-side `list_hf_files_under_path`, one resolved
revision, `max_workers<=6` pool) as one helper. Two scope notes: (a) the
retry absorbs RAISED transients only — the hf-xet HANG class (no exception;
socket count does NOT discriminate it from the upload wedge, #1739/#2153)
stays on the kill+replay ladder (gotchas.md), and flaky-egress
accelerator handling stays the per-launch `HF_HUB_DISABLE_XET=1` kill-switch
replay, never a default flip; (b) the verbatim prefix mirror is a staged
LAYOUT — a consumer with a fixed local layout still owes the staged-layout
consumer-open probe at reuse time, once per (source-family × staged
consumer) pair (`.claude/rules/artifact-reuse.md` check (h)(iv), #928,
#1481); "canonical helper" does not mean "layout-mapping solved".

## Detached HF transfers: timeout + observable progress (#2153)

Binding on ANY HF transfer (upload OR download, ANY file count) that runs
detached / backgrounded / outside the launching turn's own foreground —
`setsid`/`nohup` staging legs, pod-side dispatcher phases, bg-Bash restores.
Driving incident (#1739): two detached HF jobs froze silently — 0-byte logs,
near-zero CPU, exactly ONE open socket, empty targets, no error, no exit —
costing ~90 min before manual kills; nothing in either transfer made the hang
DETECTABLE. Requirements:

- **(a) A wall-clock timeout bounding the WHOLE transfer.** The
  `EPM_HF_RETRY_BUDGET_S` retry budget is NOT a timeout — it bounds RAISED
  transients, and the hang raises nothing; per-file `stage_hub_file` budgets
  are per-call, so an N-file prefix can serially burn ~N × budget without any
  single call exceeding it. **SIZING BASIS — required, one sentence, never a
  guessed constant:** the bound derives from projected bytes ÷
  measured-or-expected throughput at ≥2× margin (the standing fence
  convention, `plan-compute-sizing.md` § Per-cell fit phases) AND must exceed
  the transfer's legitimate retry-budget exposure (~N files ×
  `EPM_HF_RETRY_BUDGET_S`, default 1800 s). A guessed fence killing healthy
  work is a demonstrated incident class (#1092: a guessed `timeout 3000s`
  killed a healthy ~25 min/cell run, exit=124), and a "generous" 1 h constant
  would kill a legitimately-retrying multi-file staging under a 429 storm.
  For `stage_hub_prefix` the built-in arm is `EPM_HF_STAGE_TIMEOUT_S`
  (unset/empty/non-positive = OFF — `0` is how a caller spells "disabled",
  never a 0 s fence; expiry flushes a stalled-file diagnostic then hard-exits
  `os._exit(STAGE_HUB_PREFIX_TIMEOUT_RC)`, rc 87 — a raise cannot produce an
  rc when a worker is parked in native `xet_get`, because
  `concurrent.futures`' atexit hook joins its non-daemon workers).
- **(b) Periodic flushed progress** — the `code-style.md` canonical shape
  `[<phase>] unit k/N <key> elapsed=<s>s`, with the trigger WIDENED: a
  detached HF transfer owes progress regardless of file count or projected
  wall-time (the size-keyed T1/T2 checkpoint triggers cannot fire on a
  per-file stall). `stage_hub_prefix` emits this natively (#2153): an entry
  line flushed BEFORE any network call, an N-files line after the scoped
  listing, one flushed line per completed file.
- **(c) Completion keyed on process EXIT with a captured rc** — never on file
  existence, never on a non-empty log (the HF-transfer instance of the
  CLAUDE.md § Monitoring re-run discipline and the #825 empty-dir false-DONE
  class).
- **(d) Invariant: a 0-byte log + empty target must NOT be a reachable
  "looks healthy" state.** A transfer that can present that way is missing
  (a)-(c). This binds from the transfer's FIRST instruction, not its first
  network response — the first network call can legitimately sit ~30 min in
  a retry envelope, so the first flushed line must precede it
  (`stage_hub_prefix`'s split entry line is the reference shape).
- **(e) One worker per `local_dir`.** Never two concurrent downloads into the
  same `local_dir` (#1739 ran two concurrent `snapshot_download` calls into
  ONE target); the #833 entry's "ONE staging process" clause
  (`.claude/rules/gotchas.md`) is the same law at process grain.

**Accelerator (xet) default — decided (#2153): xet stays ON by default;
disables are scoped PER WORKLOAD**, set inside the script before its
`huggingface_hub` import (env is frozen at import; reference impl
`scripts/issue1739_restore_partial.py`). Rationale: the FAILURE MATRIX entry
(`.claude/rules/gotchas.md`) — the three transfer paths fail in DISJOINT
domains, so a process-wide download-side disable fixes the small-file-storm
leg and breaks the big-file leg (the plain path hard-refuses >50 GB via
`MAX_HTTP_DOWNLOAD_SIZE`); and hub GH #3266 reports the download-side disable
ITSELF has a coverage gap — a fleet flip would buy breakage without even
guaranteed protection. Cross-refs: the #833 enumeration entry + the corrected
#1345 wedge-ladder entry (`.claude/rules/gotchas.md`), the
`workflow_lint.py --check-snapshot-download-allow-patterns` lint (#2153), and
`code-style.md`'s size-keyed progress-line convention (this section widens
that trigger for detached HF transfers).

**Fail-loud uploads.** `upload_dataset_directory` (`orchestrate/hub.py`) exits
non-zero on failure (`--no-upload` only for dry-runs).

**HF Hub rate limit: 256 repository commits per hour.** A sweep that pushes one
Hub commit per cell/fraction WILL hit `429: You have exceeded the rate limit for
repository commits (256 per hour)` mid-sweep, and a per-cell wrapper that only
logs "upload returned no path" as a WARNING turns the throttle into silent
artifact loss (#488: 41/324 adapter uploads silently missing after rc=0 cells).
Rules: (a) sweeps producing >~200 per-cell
commits/hr batch their uploads into ONE bulk `upload_folder` commit per sweep
(or chunked commits well under the cap); (b) "upload returned no path" is a
TRACKED GAP recorded in the sweep's failure list and reconciled before the next
phase — never a warning-and-continue; (c) the FAIL-FAST direction needs a
bounded OUTER retry (#1315): a dispatcher seam that RAISES on `hub._upload`'s
no-path return (correct — (b) bans warning-and-continue) must first RETRY the
no-path return with bounded jittered backoff, then raise the SAME fail-loud
`upload returned no path` error on exhaustion. Layering: `_upload` already
wraps each upload call in the inner `_retry_upload` envelope (6 attempts /
~1800 s budget, Retry-After-aware, 429/408/5xx), catches what survives, logs
"Upload failed: …", and returns `""` — so a no-path return means the inner
budget EXHAUSTED or the failure classed non-transient (quota-403 and the
0-files-verify path land here; #1360 routed the previously-bare
`api.file_exists` verify fallback through `_retry_upload` and classified the
response-less Xet "queue size reached" body text transient in
`_is_transient_upload_error`). The seam retry is the cheap bounded OUTER
envelope — each attempt re-enters the full inner envelope after a 30-120 s
pause; errors the seam's own guards RAISE propagate un-retried. Retries are
free: uploads are idempotent (already-landed files verify + skip Hub-side).
Validated constants: 3 retries, (30, 60, 120) s backoff + 0-25% jitter, one
log line per retry as the fix-engaged signal (worked example
`_upload_with_transport_retry()` in `scripts/issue1315_dispatch.py`).
IN-PROCESS complement of the #931 wedge ladder above, never a substitute: the
seam retry fires when the upload RETURNS failed; the ladder fires when it
HANGS (~0 TX, never returns).

**Fleet-shared commit budget (#1547).** The 256-commits/hr rate limit is
enforced against the SHARED repos (`superkaiba1/explore-persona-space` +
`-data`), not per run — N concurrent upload-heavy runs share ONE budget, so
per-run batching under the cap is necessary but NOT sufficient on a busy
fleet day (three independent HF-429 kills in one day at un-routed call
sites). Rules: (a) size per-run commit cadence fleet-aware — per-cell
`upload_folder` commits (the #664 rule below) with a soft per-run budget of
~≤60 commits/hr whenever other upload-heavy runs are live (≈256 / 4
concurrent); (b) the shared back-off mechanism IS `retry_transient`'s
Retry-After-honoring envelope — when the shared budget saturates, every
ROUTED caller self-throttles on the server hint, which works only to the
extent call sites are actually routed; (c) therefore every NEW direct
`hf_hub_download` / `upload_file` / `upload_folder` / `create_commit` /
`push_to_hub` call in LIVE code — anything under
`src/explore_persona_space/**` or `scripts/**` not in the frozen snapshot,
including newly-written per-issue drivers — rides `hub.retry_transient` (a
2-line lambda wrap at authoring time) or carries a `# NO_RETRY: <reason>`
waiver; mechanically enforced by
`workflow_lint.py --check-live-hf-retry-routing` (bundled into the no-flags
default run); (d) during an observed fleet-wide 429 storm, do NOT launch
additional big upload phases — sequence them behind the storm; the in-flight
envelopes drain first. FROZEN per-issue drivers/modules present at #1547
implement time (the `HF_ROUTING_FROZEN_SNAPSHOT` allowlist in
`workflow_lint.py`) are historical reproducibility artifacts, exempt from
retro-fitting; at REUSE time the routing requirement is picked back up by
the artifact-reuse throughput check (i) ("fix the SOURCE module, then
reuse"). Scope boundary (deliberate): bare `snapshot_download` /
`list_repo_files` sites are OUT of the lint's predicate — those call classes
are governed by the scoped-listing + `retry_transient` recipes in
`.claude/rules/gotchas.md` + § Relocated codebase traps below (#833) and the `--check-hub-verify-retry` lint.

**Multi-cell pod sweeps upload per-cell, never one terminal batch (#664).** A
dispatcher that produces per-cell artifacts (eval JSONs, store tensors, raw
completions) across N cells MUST persist each cell's artifacts the moment that
cell completes — one `upload_folder` commit per cell-dir per artifact-kind
(well under the 256-commits/hr cap above) — NOT accumulate them for one
terminal P3 batch. A mid-sweep pod death (the #664 RUNNING-but-no-port host
wedge — `compute-backend-failover.md` Part C) with write-at-end upload
strands EVERY not-yet-uploaded cell (#664 lost ~16 cells / ~3-4h compute);
per-cell upload strands at most one in-flight cell — the artifact-I/O
instance of `code-style.md` § "Checkpoint per phase". Idempotency +
completeness use an EXACT expected-file-set check on a fresh listing (NOT
prefix-presence / count-only — a mid-`upload_folder` crash leaves a partial
cell that prefix-presence wrongly reads as complete); canonical
implementation `hub.verify_repo_paths_uploaded(...)` (server-side scoped +
retried, returns the missing set; #997). The per-cell resume predicate is
`local-done OR HF-complete`, so a fresh pod after a wedge auto-migrate SKIPS
HF-complete cells, and the terminal P3 sweep becomes an idempotent safety
pass + the authoritative before-teardown EXACT-set verify (every helper,
store tensors included). Per-cell upload is ALSO the data-safety
precondition for the autonomous RunPod-wedge auto-terminate
(`compute-backend-failover.md` Part C): terminate fires only when the
per-cell three-state gate finds zero partial cells. Reference impl:
`scripts/issue664_dispatch.py` `_upload_cell_artifacts` /
`_classify_cell_hub_state` / `_cell_done_anywhere`.

**Expensive stores upload BEFORE — or detached-concurrent with — any long
fit/analysis phase; a fit hang must never strand the store (#825).** When a
run produces a regeneration-costly intermediate (an extraction / activation
store, a teacher-forced capture, an on-policy rollout set — anything whose
recreation costs GPU re-extraction rather than cheap CPU recompute) and a
DOWNSTREAM fit/analysis/eval phase consumes it, the store's upload is
sequenced BEFORE any long (>~15-30 min) downstream phase begins, or the
upload is LAUNCHED concurrently with the fit (detached/backgrounded — HF
`upload_folder` costs no GPU and overlaps a CPU fit freely). A concurrent
launch counts as persistence ONLY when it is fail-loud and its completion
is VERIFIED independently of the fit's completion — an exit-status check on
the detached upload, or `hub.verify_repo_paths_uploaded` against the
expected file set, BEFORE the fit's result is consumed; a fire-and-forget
launch (never confirmed landed) does NOT satisfy this rule — a silently
wedged upload plus a hung fit strands the store exactly as #825 did (the
#931 wedge ladder above is the hung-upload remedy; this clause is what
makes the ladder reachable before the pod is gone). The default order
`extract → fit → upload` parks the entire fit's hang/crash/OOM-kill risk
between the expensive artifact's creation and its persistence (#825: a hung
serial CPU fit before `[phase=upload]` stranded the turnstore off HF;
recovery cost a full fresh GPU re-extraction). This is the INTRA-RUN
sibling of the two #664 sequencing rules (per-cell upload above;
pod-release before the final bulk upload, v2 § below). Plan-side mirror:
`planner.md` §9; review enforcement: Methodology lens item 10(i)
data-safety sequencing clause (`.claude/rules/critic-lens-reference.md`).

**Inline-upload fence `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD`.** `_finalize_phase`
auto-uploads merged checkpoints to WandB Artifacts; orchestrators doing their own
tagged upload set the env in `try/finally` to prevent double-uploads.

**Merged-dir HF uploads are opt-in (default OFF); the LoRA adapter is the
canonical artifact.** `merged_upload_enabled()` (`orchestrate/hub.py`) gates
`runner.py`'s merged post-EM / pre-EM HF uploads behind `EPM_UPLOAD_MERGED=1`
(env) or `upload_merged: true` (cfg, default false); by default
`_finalize_phase` auto-uploads only the adapter to
`adapters/{run}/{phase}_adapter`. Optimizer/scheduler/rng state
(`TRAINING_STATE_IGNORE_PATTERNS`, `orchestrate/hub.py`) is ALWAYS excluded
from every HF folder upload — no opt-out. Distributed FULL fine-tunes are
exempt: no adapter exists, so the full checkpoint stays the canonical upload.
Two semantics: (a) `upload_to: "none"` does NOT suppress the default adapter
upload — `_finalize_phase` has no view of `upload_to`, so flows that own
their uploads must set the `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD` fence; (b)
the local adapter is reaped only after a VERIFIED upload (or under the
fence) — when uploads fail-soft (e.g. quota 403), adapters accumulate on the
pod's ~130GB MooseFS quota instead of being deleted, by design
(upload-before-delete invariant).

**`WANDB_LOG_MODEL` is a HuggingFace/WandB env var — NOT one of ours — and
must stay unset (or `false`) in every training environment.** Distinct from
the three project-owned WandB checkpoint-upload sites (all gated by
`EPM_UPLOAD_MODEL_WANDB=1`, default OFF: `orchestrate/hub.py`
`upload_model_wandb`, `train/trainer.py` `_maybe_upload_checkpoint_to_wandb`
+ its `train/sft.py` call site): HF `Trainer` installs a built-in
`WandbCallback` whenever `report_to="wandb"` (which every project training
run with a WandB run name sets). That callback reads `WANDB_LOG_MODEL` from
the environment at init — `end` uploads the final saved model to WandB
Artifacts, `checkpoint` uploads every `save_steps` checkpoint dir, default
`false`/unset uploads nothing. This path is INDEPENDENT of our
`_maybe_upload_*` code — `EPM_UPLOAD_MODEL_WANDB=1` does NOT gate it, and
setting `WANDB_LOG_MODEL` re-opens the ~15 GB-safetensors-to-WandB leak
regardless of our guard. It must never be set (or must be explicitly
`false`/`0`) in `bootstrap_pod.sh`, the GCE startup prelude, the SLURM
sbatch env block, `.env`, and launch shells (currently unset in all of
them; keep it that way). This surface let ~784 GB of checkpoints accumulate
on WandB before the guard landed (the 2026-06-30 4TB cleanup; only-on-WandB
orphans archived to the private
`superkaiba1/explore-persona-space-wandb-archive` repo before deletion).

**Delete-after-eval sweeps MUST persist the ADAPTER first (never the merged dir).**
A sweep that `rm`s a trained checkpoint after its eval to stay under the MooseFS
~130GB quota (the #404/#458 pattern) MUST set `EPM_PERSIST_ADAPTER_HF_REPO` +
`EPM_PERSIST_ADAPTER_SUBFOLDER` so `_finalize_phase` uploads **and verifies** the
LoRA adapter (~300MB) before it is reaped. The persist is **fail-loud**: if it
can't verify the adapter landed, training raises and exits non-zero, so the
launcher's `set -e` aborts the cell *before* its `rm` — closing the silent-loss
hole. NEVER upload the ~15GB merged checkpoint to the shared public model repo to
satisfy this: it's derived data (regenerable from base + adapter), 45× larger, and
would blow the already-~550GB HF repo quota (the same quota that soft-failed
#458's merged upload, after which the `rm` deleted all 36 checkpoints). Pair this
with `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1` + `upload_to=none` on the train call so
the wasteful 15GB merged WandB/HF uploads don't fire at all. Re-eval = download
adapter, re-merge with base.

**HF storage-quota 403 is persistent + account-wide — recover, don't retry-loop.**
Signature: `403 Forbidden: You have exceeded your public storage space` on
`.../info/lfs/objects/batch` during `upload_folder` / `upload_file`. Unlike the
256/hr commit throttle above, this is the ACCOUNT-WIDE public-storage quota: it
is not transient, it hits every running task at once, and retrying changes
nothing until quota is freed. **The quota gate fires ONLY on the LFS endpoint**
(validated #541/#552): regular (non-LFS) git-blob commits to public repos
still succeed while over quota, and PRIVATE-repo LFS uploads still succeed too
(private storage is a separate quota with headroom on PRO). A file routes to
LFS when its extension is LFS-matched in the repo's `.gitattributes`
(`*.safetensors`, `*.bin`, `*.gz`, ... — `*.json` / `*.jsonl` / `*.txt` are
NOT matched in the data repo) OR when `upload_file` / `upload_folder`
force-routes it at >10MB — so small text/JSON keeps flowing while adapters /
safetensors / merged dirs fail on BOTH repos. Recovery ordering:
(1) NEVER delete the local copy — the fail-loud persist guard above is correct;
let it halt the cell rather than papering over the 403. (2) Keep small-artifact
uploads (eval JSONs, raw completions, analysis tensors) flowing to the dataset
repo unchanged — they ride the non-LFS path. Text payloads <9.5MB upload
as-is; line-split bigger files into <9MB shards (`<stem>.shardNN.jsonl` plus a
`<stem>.manifest.json` listing ordered parts, line counts, sha256s). NEVER
gzip to shrink them — `*.gz` IS LFS-matched and re-enters the blocked path.
(3) For LFS-only artifacts (adapters, checkpoints): upload to the PRIVATE
overflow repo `superkaiba1/explore-persona-space-overflow` under the same
`issueN_<slug>/...` subfolder layout, record a plan-deviation entry + the
overflow URLs in the run's results sentinel, and migrate to the canonical repo
after quota is freed. As a second durable replica (or if the private path also
fails), pull the adapters off the pod to the VM (rsync; on a `--no-bootstrap`
pod tar-over-ssh: `ssh <pod> 'tar -C /workspace -cf - <adapter-dir>' | tar -xf -`)
into a local staging dir `eval_results/issue_<N>/adapter_backup/<cell>/`
(local staging only — `*.safetensors` is gitignored) AND log a WandB Artifact
(`type="model"`) copy.
(4) Retry the canonical HF model-repo upload only after quota is freed.
Freeing quota means deleting existing HF artifacts — that is USER-ONLY:
surface the situation to the user, never auto-delete from HF. Two corollaries
of the LFS-only gate (both bit in #1586): (a) an "unblocked?" probe MUST
exercise an LFS-SCALE upload (>10 MB, LFS-matched extension) — a passing
small-file/text upload is NOT evidence the block lifted (small files ride the
always-open non-LFS path); (b) a user-action escalation (enable auto-recharge,
free quota) names the EXACT click path AND what the configured end-state looks
like on the page.
Corollary-(a) canonical probe (#1654): `hub.check_lfs_write_gate()` (or
`preflight --no-gpu --planned-upload-gb <N>`) is the canonical zero-byte
"is the LFS write path open?" probe at declared scale — one LFS
batch-endpoint negotiation per repo declaring ~16 GB (env
`EPM_HF_BILLING_PROBE_GB`; kill switch `EPM_HF_BILLING_PROBE=0`), zero bytes
transferred, zero commits; a REAL >10 MB LFS upload remains valid where
end-to-end transfer confirmation is wanted. Three caveats: (i) a PASS (`ok`)
is ADVISORY — the probe's 403 arm has never been observed live; on the NEXT
403-blocked incident, run `check_lfs_write_gate()` WHILE blocked and record
the verdict against #1654 assumption 3; (ii) a ~16 GB-declared PASS ≠ credit
clearance for a whole run's uploads — mid-run credit exhaustion stays with the
reactive 403 backstop, and do NOT size the probe to `--planned-upload-gb` (a
declared object above per-file caps fails for size reasons and degrades the
verdict to `unknown`); (iii) the blocked-verdict `detail` excerpt governs the
exact remediation path (a "storage patterns" manual-review 403 — hub issue
#3366 — classifies `storage-blocked` and names its own contact address).
Diagnosis probes: sum account usage via
`/api/{models,datasets}/<id>?expand[]=usedStorage` over
`list_models(author=...)` / `list_datasets(author=...)`; a tiny non-LFS `.txt`
upload probes the regular-blob path; a tiny `.bin` upload to the private repo
probes the private-LFS path. (Incident #541: 11.3 TB public across 414 repos
killed the sweep's first upload; #552 hit the same wall the same day.)

**Proactive detection (#564): soft-ceiling headroom check + minute-1 persist
gate + opt-in overflow routing.** `check_hf_storage_headroom()`
(`orchestrate/hub.py`) sums per-repo `usedStorage` over the account's public
repos behind a 1h on-disk cache; knobs: `EPM_HF_STORAGE_SOFT_CEILING_TB`
(default 10.0), `EPM_HF_STORAGE_CACHE_TTL_S`, `EPM_HF_STORAGE_CACHE_PATH`,
kill switch `EPM_HF_STORAGE_CHECK=0` (the ceiling / routing / check / TTL
envs are threaded through the slurm + gcp passthrough allowlists; the
cache-path + event-path envs deliberately are NOT). Preflight surfaces it as
a WARN-only `HF storage:` line. `trainer.py::_validate_persist_headroom` —
called at the top of `_init_phase` AND at the start of `sft.py::train_lora`
— aborts a persist-declared run (`EPM_PERSIST_ADAPTER_HF_REPO` set) in
minute 1 when a forced LIVE re-probe confirms over-ceiling and the persist
target is public with routing off (unknown headroom / undeterminable privacy
fail OPEN — the upload-time backstop stays authoritative).
`EPM_HF_OVERFLOW_ROUTING=1` (default OFF) makes `upload_model` reroute LFS
uploads to the private overflow repo when KNOWN-over-ceiling, appending a
deviation event to `EPM_HF_OVERFLOW_EVENT_PATH` →
`/workspace/logs/hf-overflow-routing.jsonl` →
`~/.cache/explore_persona_space/hf-overflow-routing.jsonl` (the orchestrator
/ upload-verifier observing that sentinel posts the actual `epm:`
plan-deviation marker — pod-side code never shells `task.py`), and
committing a small `OVERFLOW_POINTER.json` breadcrumb to the CANONICAL repo
(non-LFS, so it works over quota). ARMING CONTRACT: routing is safe ONLY for
flows that consume `upload_model`'s returned URL or read the
pointer/deviation records — launchers that verify CANONICAL paths externally
(the i528 family) must NOT arm it, because a reroute converts their 403 into
a post-training verification abort. Dataset / raw-completion paths are
deliberately un-routed. New per-issue scripts should prefer `upload_model`
over direct `HfApi` calls for LFS artifacts so they inherit this guard.

**Size-aware projected-headroom probe (#1034).**
`hub.check_projected_upload_headroom(projected_bytes)` compares a PLANNED LFS
upload's byte size against REMAINING headroom (`used + projected > ceiling`),
which the binary #564 check cannot do. Verdicts: `below-threshold` (projected
under `EPM_HF_LARGE_UPLOAD_PROBE_GB`, default 100 decimal GB — ZERO headroom
I/O) | `disabled` | `unknown` (fail-open — callers never block/reroute) |
`fits` | `insufficient` (only after a `force_refresh=True` LIVE confirm —
never act on a ≤1h-stale cached over-read). Three consumers: (1)
**`upload_dir_sharded` routes ALL shards to the private overflow repo
UP-FRONT** on KNOWN-insufficient + confirmed-public canonical target (one
pointer, one JSONL event `reason: "projected-headroom-proactive"`, zero
canonical LFS bytes attempted; opt out with `proactive_overflow=False` for a
canonical-path-verifying caller) — **route ≥100 GB stores through
`upload_dir_sharded` explicitly** so they inherit this; (2) armed
`upload_model` (`EPM_HF_OVERFLOW_ROUTING=1`) reroutes when
`used + dir_size > ceiling`, not only when already over; (3) preflight
`--planned-upload-gb <N>` turns the WARN-only advisory into a hard gate
(LIVE-CONFIRMED-insufficient + routing off → FAIL; armed → WARN;
unknown/disabled → WARN). Residual routes NOT covered: `hub._upload`,
`hub._upload_folder_filtered`, direct-`HfApi` per-issue scripts — the
preflight plan-projection gate covers plan-declared big uploads regardless
of helper, and the 403 stays fail-loud, but do not mistake the guard for
fleet-wide coverage. Overflow-repo artifacts are PRIVATE — reached
auth-required and pointer-mediated, never as canonical-path equivalents.

**File-count limit (100k) — reactive overflow fallback (#1108).** HF
hard-rejects any push that would put a repo over 100,000 git files (the
canonical model repo sits at the limit; #1090's rejected c5 push).
`hub._upload` catches that rejection on a MODEL-repo upload and retries the
identical upload against the private overflow repo (`DEFAULT_OVERFLOW_REPO`),
then emits the #564 routing event (`reason: "file-count-limit-reactive"`) and
writes the `OVERFLOW_POINTER.json` breadcrumb at the canonical path. **Default
ON** (kill switch `EPM_HF_FILECOUNT_FALLBACK=0`): unlike the #564 byte-quota
routing (default-OFF because a pre-emptive reroute can divert a would-succeed
push), this fires only AFTER the server refused the canonical push. Detection
is message-substring based; a changed rejection shape degrades to the
fail-soft `""` — never a wrong reroute. A TEMPORARY DURABILITY fallback
pending the user's file-count triage, NOT a transparent successor to
canonical storage — overflow artifacts are PRIVATE and pointer-mediated.
**i528-family caveat:** a persist-gated flow (`EPM_PERSIST_ADAPTER_HF_REPO`)
that previously failed LOUD at the gate now proceeds on a VERIFIED private
overflow landing — an EXTERNAL launcher that verifies CANONICAL paths fails
LATER (at its own verify); such launchers should set
`EPM_HF_FILECOUNT_FALLBACK=0`. A concurrent user-side freeing between
rejection and retry is harmless (lands on overflow with a pointer; the next
upload takes the canonical path again). **Scope:** `repo_type="model"` via
`upload_model` → `_upload` only — the ~1M-file DATA repo empirically still
accepts pushes, and direct-`HfApi` per-issue scripts, `upload_dir_sharded`,
and `_upload_folder_filtered` are named residuals outside this fallback.

**Per-DIRECTORY file-count cap (10k/dir) — PACK many-small-file trees before
upload (#1190/#1739).** The Hub ALSO rejects any single COMMIT staging
>10,000 files into one repo directory (a server-side 400 — DISTINCT from the
repo-total 100k cap above). The #1190 guard pre-counts staged files per
target dir in the hub helpers and raises `HubDirFileCountError` BEFORE any
network I/O (`HUB_DIR_FILE_LIMIT` 10,000; kill switch
`EPM_SKIP_HF_DIR_FILECOUNT_GUARD=1` degrades to WARN; advisory watermarks
`HUB_DIR_FILECOUNT_WARN` 5,000/dir and `HUB_COMMIT_FILECOUNT_WARN` 2,000
staged files/commit, #1571 — a commit of many SMALL files crawls in Hub-side
pre-processing regardless of byte size: #1481 killed a 31,000-file / 135 MB
commit after >20 min). When the guard fires — and at PLAN time,
whenever a workload will emit ≳2,000 per-unit small text/JSON files
(per-rollout JSONs, per-sample transcripts, per-context captures) — do NOT
point the per-file tree at `upload_folder`, and do NOT reach for the kill
switch: PACK the tree into ≤9 MB `<group>.shardNN.jsonl` line-shards — one
line per SOURCE FILE, `{"src": "<path relative to raw root>", "doc":
<original JSON/text>}` — plus a census-keyed `pack_manifest.json` (per-group
(relpath, size, mtime_ns) census ⇒ idempotent re-packs), then upload the
small shard set in ONE bulk `upload_folder` commit with an exact-set verify.
Consumers UNPACK back to the per-file layout (manifest/sha verify; never
overwrite a differing file). This is the MANY-FILES sibling of the
single-big-file >9.5 MB `<stem>.shardNN.jsonl` line-split in the quota-403
recovery above (that recipe splits ONE oversized text file; this one packs
thousands of small files into few shards). ≤9 MB keeps every shard on the
always-open non-LFS path (the >10 MB LFS force-route above). Worked
example: #1739 r4/r5 — a 115,941-file labeling tree packed to a small shard
set (`scripts/issue1739_pack.py`, on the issue-1739 branch until its merge). <!-- lint: historical-ref -->
The `shard_NNNN/` ≤5,000-files-per-dir DIRECTORY-sharding recipe
(`gotchas.md`, the #658 entry) stays the fallback ONLY when the consumer
genuinely needs the per-file layout ON the Hub, and for binaries a jsonl
line cannot carry (per-rollout `.pt` stores) — it clears the 10k/dir cap
but keeps the file count, so commit throughput stays poor; packing is the
default.

**Large free-text DV / labeling JSONs route to the HF data repo, not git
(#1739).** Git `eval_results/` keeps SMALL aggregated JSONs (summary stats,
per-cell tables). A per-row free-text-bearing JSON at MB scale (#1739: a
22 MB free-text DV file) is an HF-data-repo artifact (`issueN_<slug>/...`,
non-LFS path): the gitleaks pre-commit scan does not scale on free text
(5,938 false positives / 2m36s on that one file, blocking the commit), and
fingerprinting per-row text into `.gitleaksignore` is unbounded churn.
Heuristic: free-text-bearing AND ≳1 MB → HF data repo; commit only the
derived aggregate to git. (This refines — not contradicts — the CLAUDE.md
destination table's "Eval results (aggregated JSON) → git": the
*aggregated* qualifier is load-bearing.)

## v2 tasks (`workflow: v2`) — upload-by-default, no ceiling

For a task whose frontmatter carries `workflow: v2`, the upload policy has NO
policy ceiling (Thomas's call). Everything above still holds; v2 tightens it to:

- **Text / JSON — always, unconditionally.** Raw responses at every stage,
  judge outputs, metrics, configs upload to the data repo on the non-LFS path,
  which is quota-immune (#541 gates only LFS). Text is NEVER discardable — not
  even under both-quota exhaustion — and NEVER a valid `discarded_artifacts:`
  entry (Step 3 generation-discard gate stays binding).

- **Tensors / activation stores — main repo → overflow repo, no ceiling.**
  Every store attempts the canonical repo first, then reroutes to the private
  overflow repo (`superkaiba1/explore-persona-space-overflow`, the existing
  `EPM_HF_OVERFLOW_ROUTING` mechanism) on a quota-403, dropping an
  `OVERFLOW_POINTER.json` breadcrumb on the canonical repo. There is no
  100 GB-style policy cap; the 128 GB per-issue ext4 quota / ~130 GB MooseFS
  quota are PHYSICAL limits, handled by incremental sharding below, not a
  policy ceiling. Stores whose PROJECTED size exceeds remaining headroom
  route to overflow UP-FRONT (one pointer, one event — the #1034 proactive
  probe in `upload_dir_sharded`) instead of splitting at the mid-store 403.

- **Big stores upload INCREMENTALLY (upload → verify → delete-local).** A store
  larger than the disk quota is uploaded per shard so local footprint stays
  bounded to ~one shard: `orchestrate.upload_sharded.upload_dir_sharded`
  (reuses the hub overflow mechanism + `list_repo_files_complete` verify).
  Stream-reduce phases PREFER shard-and-upload now that uploads are unbounded;
  where materialization is genuinely infeasible, persist the source rollout
  text (regenerable via one teacher-forced pass) — the #666/#772 stream-reduce
  memory-safety contract is unchanged.

- **Discard-to-regen-recipe fires ONLY when BOTH quotas are exhausted, always
  alerted.** A discard is licensed only for a large intermediate TENSOR, only
  after the main AND overflow repos both refuse (the `upload_dir_sharded`
  both-refused `RuntimeError`), and only with a plan `discarded_artifacts:`
  `{name, reason, regen_recipe}` entry + an alert naming which gate closed.
  Generations / rollout text / judge outputs / metrics / configs are never
  discardable.

- **Registry append at PASS.** On upload-verification PASS the verifier appends
  one `artifacts/registry.jsonl` row per produced artifact via
  `scripts/artifact_registry.py` — the reuse registry the planner +
  methodology-writer read before any retrain/regenerate.

- **#664 sequencing unchanged.** The GPU pod is released before the FINAL bulk
  upload; incremental shard uploads may overlap compute (they cost no GPU).
  The #825 intra-run ordering bullet (main body above) binds v2 unchanged:
  an expensive extraction store uploads before — or concurrent with — any
  long fit/analysis phase that consumes it.

## Relocated codebase traps (from `.claude/rules/gotchas.md`, #2189)

Verbatim gotchas.md entries whose topic this rule already owns — relocated
to recover gotchas.md byte budget (#2189); wording and `#N` citations kept.

- **HF Hub list APIs (`list_repo_tree` et al.) return LAZY generators — a try/except around the CALL catches nothing; the HTTP error raises at ITERATION time.** Materialize (`list(...)`) inside the try/except, or move the handler to the consuming loop. (#779.)
- **HF Hub per-org 2500-req-per-5-min rate limit (429) on bulk `snapshot_download`.** Each xet-read-token fetch / tree-listing / file-metadata call counts as one request, so a `snapshot_download` of ≥3000 files at default `max_workers` predictably trips the quota — and the failure surfaces LATE (`HfHubHTTPError: 429` mid-download after tens of minutes; #658). RULE: for bulk `snapshot_download`, pass `max_workers=4` (≈1200 req/5min) AND wrap in a 429-aware retry with `Retry-After`-bounded backoff (60–300s) — the xet bulk path is NOT covered by `huggingface_hub`'s built-in 429 retry. Sibling of the per-probe `AutoTokenizer.from_pretrained` 429 entry below (same org quota). On the ~1M-file data repo `snapshot_download` is barred outright regardless of `max_workers` — see the next entry.
- **`snapshot_download(allow_patterns=...)` against the ~1M-file data repo enumerates the ENTIRE repo tree BEFORE filtering — staging wedges indefinitely; bare `list_repo_files` on that repo also times out (#833: >90 s).** `allow_patterns` is CLIENT-side: on a very large repo, `snapshot_download` falls back to a FULL `list_repo_tree(recursive=True)` walk — sequential paginated pages under the same ~2500-req/5-min org quota — and only then filters (`huggingface_hub` 0.36.2). Against `superkaiba1/explore-persona-space-data` the enumeration is effectively unbounded (#833: a GCE staging step sat 40+ min, zero files landed). RULE: stage any subtree of the data repo by enumerating with SERVER-side-scoped `list_repo_tree(repo_id, path_in_repo=<prefix>, repo_type="dataset", recursive=True)` (the prefix rides in the tree URL — seconds for `issueN_<slug>`-scale prefixes), then download per-file via `hf_hub_download` in a thread pool of `max_workers<=6` with retry + linear backoff, ONE staging process (9 concurrent staging PROCESSES tripped the quota in #833 r3). When a coherent snapshot matters, resolve ONE `revision` first and pass it to `list_repo_tree` AND every `hf_hub_download`. For a SINGLE-path existence probe use `HfApi().file_exists(...)` — never a full listing. Working recipe: `scripts/issue833_gcp_phase_d.sh`.
- **Hub HTTP path args are LITERAL — `path_in_repo` / `hf_hub_download(filename=...)` / `list_repo_tree(path_in_repo=...)` do NOT expand globs (a glob 404s; URL-encoded `%2A`/`%3F` in the failing URL is the tell), and listing a not-yet-existing prefix also 404s.** Only CLIENT-side filters (`allow_patterns`) take globs; a path arg rides verbatim in the HTTP URL. RULE: pass literal paths/prefixes; probe existence via `HfApi().file_exists(...)` or `list_repo_tree(path_in_repo=<literal prefix>)`; treat an `EntryNotFoundError`-class prefix 404 BEFORE the upload phase as "not yet uploaded" (expected), but NEVER wave through a `RepositoryNotFoundError` / wrong-revision 404. Boundary: the landed `verify_artifacts_exist` glob fix (#1778) covers the Step 6a.5 gate; this entry covers AD-HOC probes. Same-family upload-side trap: `upload_as_file=True` + `path_in_repo=<bare prefix>` lands the file AT the directory name → HTTP 400 "Invalid file change" on all later commits touching that prefix (#1738 fu1) — `path_in_repo` is the full literal DESTINATION PATH, never a prefix.
- **`AutoTokenizer.from_pretrained(model_id)` called PER probe/row triggers a silent per-load `model_info()` HTTP call → HF Hub 429.** Newer `transformers` runs a Hub request inside `from_pretrained` on EVERY load; a rig re-loading the tokenizer once per cell/probe trips the ~2500-req/5-min org quota after a few dozen cells (#664: 3 dispatcher crashes). RULE: load each tokenizer ONCE and cache at module scope (`_TOKENIZER_CACHE` + a `_get_tokenizer(model_id)` accessor) — never `from_pretrained` inside a per-row/per-probe/per-cell loop; same shape for any `from_pretrained` used as a pure-CPU text helper.
