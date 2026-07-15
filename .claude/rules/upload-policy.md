---
description: Deep upload mechanics — Hub-API verification gotcha, inline-upload fence, delete-after-eval adapter-persist recipe (loads when writing training / hub / sweep code)
paths:
  - "src/explore_persona_space/orchestrate/**"
  - "scripts/train.py"
  - "scripts/run_sweep.py"
  - "src/explore_persona_space/train/**"
  - "scripts/issue*.py"
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
`HF_XET_HIGH_PERFORMANCE=1` (the PRIMARY accelerator — both project repos route
through the Xet storage backend, verified, so the Xet high-performance path is
the lever that matters) and `HF_HUB_ENABLE_HF_TRANSFER=1` (the orthogonal
LFS-multipart accelerator, future-proofing for any non-Xet repo; `hf_transfer`
is a hard `pyproject` dep so the LFS flag never enables a missing-package
fault). They are set at SHELL level in `bootstrap_pod.sh` (pod), the GCE startup
prelude (`backends/gcp.py`), and the SLURM sbatch env block (`backends/slurm.py`)
— the load-bearing placement, because `huggingface_hub.constants` freezes
`HF_HUB_ENABLE_HF_TRANSFER` at import time — plus an `orchestrate/env.py`
`setdefault` belt-and-suspenders for local-dev. Override per-launch with `=0` /
`HF_HUB_DISABLE_XET=1`: the two accelerator defaults are `setdefault` /
`${VAR:-1}` so an explicit launch-time `=0` always wins, and the GCP / SLURM
passthrough allowlists forward a dispatch-process `=0` for those two vars
AND `HF_HUB_DISABLE_XET=1` (the real kill switch — forwarded as of #1195;
`HF_XET_DISABLE` stays in the allowlists only as a legacy no-op alias), so
a dispatch-time xet disable now reaches GCP/SLURM workers on a fresh
dispatch. RunPod is NOT part of that claim — pods have no dispatch-env
passthrough (the launcher / bootstrap shell env is the channel there), so
on a pod the kill switch is still set in the WORKER shell. The effective
xet kill switch is
`HF_HUB_DISABLE_XET=1` — it flips `is_xet_available()` False
(`huggingface_hub` 0.36.2, the uv.lock pin; `constants.py` reads
`HF_HUB_DISABLE_XET`), which gates the upload branch (`_commit_api.py:380`);
download-side coverage has a reported gap on this pin (hub GH issue #3266),
so treat it as upload-verified. The historically-documented
`HF_XET_DISABLE=1` (the #515 xet-CDN DOWNLOAD workaround; retained in the
lane allowlists only as an annotated legacy alias, #1195) is a VERIFIED NO-OP on this stack — consumed by
neither `huggingface_hub` nor the `hf_xet` Rust binary (strings-checked;
live-tested 2026-07-05) — so a recipe leaning on it likely never left the
xet path; #931's first two wedge replays did exactly this. Upload sitting at
~0 TX? Run the wedge escalation ladder in the next block. A NEW
direct-upload script must use the project
`explore_persona_space.orchestrate.env.load_dotenv` wrapper, NOT the bare
`from dotenv import load_dotenv` (enforced by
`scripts/workflow_lint.py --check-dotenv-before-hf-import`).

**Pod→HF upload WEDGE — recognize it, then run the three-rung escalation
ladder (#931).** This is the UPLOAD sibling of the #515 download workaround
above. Signature: the upload process looks healthy (no traceback) while
transfer bytes stop — interface TX delta ~0 across two samples ≥5 min
apart (`cat /sys/class/net/eth0/statistics/tx_bytes`, sample twice), and/or
one ESTAB socket to the CDN (port 443) whose counters are frozen in
`ss -tinp` (`bytes_acked` / send-q not advancing; `apt-get install -y
iproute2` if `ss` is absent on the pod). High sustained CPU with ~0 TX can
be legitimate local pre-processing (xet chunking / sha256 of multi-GB
files) — the frozen-ESTAB-socket check is the discriminator. #931
(2026-07-04, an org-wide HF-429 day) sat ~30 min at ~0 TX before the first
kill — once the signature is confirmed on a re-sample, escalate immediately.
Three preconditions: (a) the upload path is replay-idempotent (per-cell /
per-folder skip-if-complete — the #664 per-cell contract; #931's completed
folder commits skipped idempotently on replay), (b) each rung is
KILL-hung-process → REPLAY-with-env — never export on top of a live process
(`huggingface_hub.constants` freezes env at import), (c) for a LIVE wedged
process the rung env must still be set IN THE WORKER's shell and the
process relaunched there (SSH into the pod/worker — the import freeze means
an orchestrator-side export can never reach a running process, and on
RunPod there is no dispatch-env passthrough at all); a full FRESH
re-dispatch with `HF_HUB_DISABLE_XET=1` in the dispatch env DOES forward to
GCP/SLURM workers as of #1195. Do not wait for a rung
to self-heal: hf_transfer retries fire only on ERRORING parts
(`max_retries=5` threaded by `lfs.py::_upload_parts_hf_transfer`), the
pure-python `http_backoff` path retries only raised errors
(Timeout/ConnectionError/5xx) and its PUTs pass no `timeout=`, and the xet
client's timeout knobs did not rescue #931's 30-min hang — a silently hung
ESTAB read never becomes an error, so detection + kill is always manual.

1. **Rung 1 — kill + replay with `HF_HUB_DISABLE_XET=1`** (the REAL switch —
   NOT the no-op `HF_XET_DISABLE`, see the clause above). Targets a
   xet-client-specific stall (hung CAS read, finalization hang — #825 r2's
   class); the upload falls back to the LFS multipart path,
   hf_transfer-accelerated since `HF_HUB_ENABLE_HF_TRANSFER=1` is default.
2. **Rung 2 — wedged identically? kill + replay with `HF_HUB_DISABLE_XET=1
   HF_HUB_ENABLE_HF_TRANSFER=0`** — the pure-python requests path. Rung 2
   without rung 1's var is a placebo on the project's xet-backed repos: while
   xet is available the upload never reaches the LFS path where hf_transfer
   lives.
3. **Rung 3 — still wedged? The on-pod upload path is dead for this run;
   reroute around it.** rsync the artifact dirs pod→VM (rsync IS on
   bootstrapped pods — `bootstrap_pod.sh` Step 2 installs it, commit
   `22e1a882a1` 2026-06-12; the RunPod image ships without it, so a
   `--no-bootstrap` pod needs the tar-over-ssh form in the #541 recovery
   below), verify the VM→HF route with a small probe upload, run the VM-side
   `upload_folder` to the SAME `path_in_repo`, then a pod-side local-only
   sentinel replay so `epm:results` lands via the normal poller drain. #931
   moved ~9.9 GB this way in ≈24 min after three wedged on-pod attempts
   (≈37 min first-kill → results, ≈60 min upload-phase-start → results,
   derived from the 06:13Z/06:36Z/06:49Z/07:13Z markers); the same-day
   `epm:upload-fix` round reused the VM route directly. If the VM→HF probe
   ALSO wedges (an HF-side incident — #931's day was an org-wide 429 day),
   the pod→VM rsync has already made the data durable: stop/terminate the
   pod rather than idling it, and retry the VM→HF upload when the incident
   clears.

Honesty caveat: #931's rung-1/2 replays set the no-op `HF_XET_DISABLE`, so
all three on-pod attempts likely ran the SAME xet client — rung-1/2 value is
derived from the 0.36.2 code paths + HF's documented legacy-LFS fallback
(docs/hub/en/xet/legacy-git-lfs), not yet proven in anger; the route-level
rung-3 reroute is the proven recovery. On a known org-wide 429/CDN-incident
day, consider going straight from one confirmed rung-1 wedge to rung 3.

**Intermediate analysis tensors referenced by the plan MUST upload before pod
termination.** Any artifact the plan's analysis / negative-control sections
name as a downstream input — per-cell shift tensors (`shifts/*.pt`), cached
activations, decomposition / SVD inputs — uploads to the HF data repo under
`issueN_<slug>/analysis_tensors/` BEFORE the pod is terminated, exactly like
raw completions. These files are typically tiny (KB-MB) next to the
checkpoints they derive from, which makes them easy to dismiss as scratch —
but losing them makes the plan's remaining controls permanently unrunnable.
(Incident #521: ~200 KB per-cell Δv `.pt` files required by two planned
negative controls — the leave-one-out SVD spectrum check and the EM
mean-over-response read — were never uploaded; a 3-round upload-verification
loop still ended PASS, the pod was terminated, and both controls became
permanently unrunnable.) Enforcement: `upload-verifier` Step 1 classifies
`*.pt` / `*.npy` as analysis tensors bound for the HF data repo, and its
Step 2.8 cross-references the plan's analysis / control sections and FAILs on
any plan-named input without a permanent URL.

**Persist by default; a discard needs a recorded justification (#779).** The
always-on Upload Policy states the principle; the mechanics: **text/JSON
uploads unconditionally** (rollout text, judge outputs, metrics, configs are
non-LFS in the data repo — the #541 quota gate fires ONLY on LFS, so this path
stays open over quota; text <9.5 MB uploads as-is, bigger text line-splits into
<9 MB shards, NEVER gzip — `*.gz` is LFS-matched, and the Hub force-routes any
>10 MB blob to LFS regardless of extension). **Large tensors upload when cheap;
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
it does not re-materialize the whole activation grid (#666/#772). Driving
incident: #779's extraction driver (`issue779_extract_rb.py`) reduced kept
rollouts to `r_B` and dropped the rollout text (wrote it only as judge input,
not under `raw_completions/`), so a sibling arm had to regenerate.

**Regenerating a published artifact in place requires a version-bumped path or
a regeneration note (#922/#779).** Re-uploading / reconstructing an
already-published artifact at the SAME path can silently invalidate every
capture another task made under the original bytes — activations,
teacher-forced reads, judge outputs, adapters trained on the mix. Each pair
member still
resolves and sha-verifies individually, so only the consumer-side pairwise
provenance-coherence check (`.claude/rules/artifact-reuse.md` item (j))
detects the incoherence — after the fact, at the cost of a wasted run.
Producer duty, one of two forms:

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

(Incident: #779 regenerated published question artifacts in place — HF commit
`9578892ef4`, 2026-07-02 — AFTER #922's dependent `cx.pt` activation capture,
`a8060198a4`, 2026-07-01; every per-member check passed and the run crashed at
a parity assert after a full GCE cycle.)

**Resume-critical pipeline INPUTS must upload before any deliberate
`pod.py stop` that expects a later resume.** The same logic extends
upstream of analysis: generated training rows (`R_train` caches,
corpus JSONs), phase-0/1 intermediate outputs, and diagnostic adapters
that the plan's later phases consume. RunPod `resume` is HOST-PINNED —
a SUPPLY_CONSTRAINT on the former host can lock the volume away for
days, and a fresh pod cannot substitute when the inputs exist only on
that volume. Push them to the HF data repo (`issueN_<slug>/inputs/` or
the relevant bucket) BEFORE stopping; they are usually MB-scale.
(Incident #488, 2026-06-10: ~18 resume attempts hit SUPPLY_CONSTRAINT
while `data/issue_488/R_train_new.json` + Phase 0/1 outputs + diagnostic
adapters lived only on the stopped pod's volume — the implementer's
pod-side smoke shipped as 'INFRA BLOCKED, local evidence only'.)

**Verify uploads with the Python Hub API, never the `hf` CLI.** The installed `hf`
CLI has NO `api` subcommand — `hf api list-repo-files ...` errors to stderr and
`| grep` swallows it as an empty/zero result that reads as a false "0 files"; `hf
repo-files` only exposes `delete`, not `list`. Use:
`set -a && source .env && set +a && uv run python -c "from huggingface_hub import HfApi; print('\n'.join(e.path for e in HfApi().list_repo_tree('superkaiba1/explore-persona-space-data', path_in_repo='<bucket>', repo_type='dataset', recursive=True, revision='main')))"`
(scoped `list_repo_tree` — a bare `list_repo_files` full listing of the
~1M-file data repo times out (>90 s, #833); gotchas.md)
(the `set -a && source .env` prefix is part of the canonical snippet — without
it the check dies on `HF_TOKEN missing`, and the obvious in-heredoc fix, a bare
`load_dotenv()`, crashes from stdin; 4+ sessions on 2026-06-10 each burned 2-3
retries re-deriving this)
(the prefix is VM-scoped — repo root, where `.env` always exists; a pod/GCE
workload script must source conditionally instead — `if [ -f ./.env ]; then
set -a; . ./.env; set +a; fi` — because the GCE lane exports tokens via its
startup script and has NO `.env` file; see the conditional-sourcing entry in
`.claude/rules/gotchas.md`, incident #923)
(#458 post-mortem nearly drew a wrong "checkpoints don't exist" conclusion from
the silent CLI "0").

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
fatal — in #1335 r5 an UN-retried per-shard `api.file_exists` HEAD probe (the
exact-file fallback inside `hub.list_hf_files_under_path`) let one transient
HF 429 ("maximum queue size reached") crash a healthy GCP run 2.8 h in, AFTER
every upload had succeeded (attempt att-20260715-134136). Two rules for any
upload/verify path in workload code: (a) wrap every FRESH Hub call in
`hub.retry_transient` (`orchestrate/hub.py` — the public alias of
`_retry_upload`: Retry-After-aware, wall-clock-budgeted via
`EPM_HF_RETRY_BUDGET_S`; storage-quota-403 and other non-transient errors
still re-raise immediately); (b) verify a SHARDED upload with ONE
prefix-scoped listing per destination repo — collect the shard paths and
check the SET via `hub.verify_repo_paths_uploaded(...)` — never a per-shard
`file_exists` / exact-file probe loop (N per-file probes multiply transport
exposure N-fold and duplicate the listing cost). The canonical sharded
implementation is `upload_sharded._batched_verify` (#1335), superseding the
per-shard `_verify_present` probe loop — the documented anti-pattern. Pin new
verify code with a 429-then-success retry test and a ≤2-listings batching
test (`tests/test_upload_sharded.py`, #1335).

**Fail-loud uploads.** `upload_dataset_directory` (`orchestrate/hub.py`) exits
non-zero on failure (`--no-upload` only for dry-runs).

**HF Hub rate limit: 256 repository commits per hour.** A sweep that pushes one
Hub commit per cell/fraction WILL hit `429: You have exceeded the rate limit for
repository commits (256 per hour)` mid-sweep, and a per-cell wrapper that only
logs "upload returned no path" as a WARNING turns the throttle into silent
artifact loss (incident #488, 2026-06-09: 41/324 adapter uploads silently
missing after rc=0 cells; caught only by a pre-phase spot-check, backfilled with
a single bulk commit in 43s). Rules: (a) sweeps producing >~200 per-cell
commits/hr batch their uploads into ONE bulk `upload_folder` commit per sweep
(or chunked commits well under the cap); (b) "upload returned no path" is a
TRACKED GAP recorded in the sweep's failure list and reconciled before the next
phase — never a warning-and-continue; (c) the FAIL-FAST direction needs a
bounded OUTER retry (#1315): a dispatcher seam that RAISES on `hub._upload`'s
no-path return (correct — (b) bans warning-and-continue) must first RETRY the
no-path return with bounded jittered backoff, then raise the SAME fail-loud
`upload returned no path` error on exhaustion. Layering: `_upload` already
wraps each upload call in the inner `_retry_upload` envelope (6 attempts /
~1800 s budget, Retry-After-aware, 429/408/5xx — the `retry_transient` entry
above), catches what survives, logs "Upload failed: …", and returns `""`
(`orchestrate/hub.py:1295`) — so a no-path return means the inner budget
EXHAUSTED or the failure classed non-transient: the demonstrated #1315 case is
the response-less Xet "maximum queue size reached" text, which
`_is_transient_upload_error` never matches (quota-403 and the 0-files-verify
path land here too). The seam retry is the cheap OUTER envelope — each attempt
re-enters the full inner envelope after a 30-120 s pause; for the Xet queue
class it is the ONLY retry. A persistent content-class failure (e.g. 403)
costs one bounded outer cycle (~3.5-4.5 min) before the same raise; errors the
seam's own guards RAISE propagate un-retried. Retries are free: uploads are
idempotent (already-landed files verify + skip Hub-side). Validated constants:
3 retries, (30, 60, 120) s backoff + 0-25% jitter, one log line per retry as
the fix-engaged signal — worked example `_upload_with_transport_retry()` in
`scripts/issue1315_dispatch.py` @ `c3c600541f` (#1315 r8: two p11 kills
~35 min apart). IN-PROCESS complement of the #931 wedge ladder above, never a
substitute: the seam retry fires when the upload RETURNS failed; the ladder
fires when it HANGS (~0 TX, never returns).

**Multi-cell pod sweeps upload per-cell, never one terminal batch (#664).** A
dispatcher that produces per-cell artifacts (eval JSONs, store tensors, raw
completions) across N cells MUST persist each cell's artifacts the moment that
cell completes — one `upload_folder` commit per cell-dir per artifact-kind
(well under the 256-commits/hr cap above) — NOT accumulate them for one
terminal P3 batch. A mid-sweep pod death (the #664 RUNNING-but-no-port host
wedge — see `compute-backend-failover.md` Part C) with write-at-end upload
strands EVERY not-yet-uploaded cell (#664 lost ~16 cells / ~3-4h compute);
per-cell upload strands at most one in-flight cell. This is the artifact-I/O
instance of `code-style.md` § "Checkpoint per phase; never accumulate-in-memory
and write-at-end". Idempotency + completeness use an EXACT expected-file-set
check on a fresh `list_repo_files` listing (NOT prefix-presence / count-only —
a mid-`upload_folder` crash leaves a partial cell that prefix-presence would
wrongly read as complete); the canonical implementation is
`hub.verify_repo_paths_uploaded(...)` (server-side scoped + retried, returns
the missing set; #997). The per-cell resume predicate is `local-done OR
HF-complete`, so a fresh pod after a wedge auto-migrate SKIPS HF-complete cells
instead of re-running them, and the terminal P3 sweep becomes an idempotent
safety pass (skip cells already complete on the Hub; treat all-on-HF as
success) + the authoritative before-teardown EXACT-set verify (every helper,
store tensors included — the M2 fresh-listing verify `_upload_store_tensors`
had been missing). Per-cell upload is ALSO the data-safety precondition for the
autonomous RunPod-wedge auto-terminate (`compute-backend-failover.md` Part C):
terminate fires only when the per-cell three-state gate finds zero partial
cells. Reference impl: `scripts/issue664_dispatch.py` `_upload_cell_artifacts` /
`_classify_cell_hub_state` / `_cell_done_anywhere`.

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
Two semantics worth knowing (code-review notes, 2026-06-10): (a) `upload_to:
"none"` does NOT suppress the default adapter upload — `_finalize_phase` has no
view of `upload_to`, so flows that own their uploads must set the
`EPM_SKIP_INLINE_CHECKPOINT_UPLOAD` fence (same precedent as the WandB
checkpoint upload); (b) the local adapter is reaped only after a VERIFIED
upload (or under the fence) — when uploads fail-soft (e.g. quota 403), adapters
accumulate on the pod's ~130GB MooseFS quota instead of being deleted, by
design (upload-before-delete invariant).

**`WANDB_LOG_MODEL` is a HuggingFace/WandB env var — NOT one of ours — and
must stay unset (or `false`) in every training environment.** Distinct from
the three project-owned WandB checkpoint-upload sites above — all gated by
`EPM_UPLOAD_MODEL_WANDB=1` (default OFF; landed commit `b4474042b7`):
`orchestrate/hub.py:1462` (`upload_model_wandb`),
`train/trainer.py:477` (`_maybe_upload_checkpoint_to_wandb`), and its
`train/sft.py:1526` call site — HF `Trainer` installs a built-in
`WandbCallback` whenever `report_to="wandb"` (which every project training
run with a WandB run name sets — `train/trainer.py:943`/`:1309`). That
callback reads `WANDB_LOG_MODEL` from the environment at init: `end` uploads
the final saved model artifact to WandB Artifacts at end of training
(`WandbCallback.on_train_end` saves the model into a temp dir and logs
THAT — not an existing checkpoint dir), `checkpoint` uploads the actual
checkpoint dir every `save_steps` via `on_save` (older Transformers also
accepted the deprecated boolean alias `true` ≈ `end`), and the default
`false`/unset uploads nothing. This path is INDEPENDENT of our `_maybe_upload_*` code — so
`EPM_UPLOAD_MODEL_WANDB=1` does NOT gate it and setting `WANDB_LOG_MODEL`
re-opens the ~15 GB-safetensors-to-WandB leak regardless of our guard.
Therefore `WANDB_LOG_MODEL` must never be set (or must be explicitly
`false`/`0`) in any environment where training runs — `bootstrap_pod.sh`,
the GCE startup prelude, the SLURM sbatch env block, `.env`, and launch
shells (it is currently unset in all of them; keep it that way). This is
the surface that let ~784 GB of checkpoints accumulate on WandB before the
`EPM_UPLOAD_MODEL_WANDB` guard landed on 2026-06-29 (the 2026-06-30 4TB
cleanup; only-on-WandB orphans were archived to the private
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
(validated #541, 2026-06-10): regular (non-LFS) git-blob commits to public
repos still succeed while over quota, and PRIVATE-repo LFS uploads still
succeed too (private storage is a separate quota with headroom on PRO). A file
routes to LFS when its extension is LFS-matched in the repo's `.gitattributes`
(`*.safetensors`, `*.bin`, `*.gz`, ... — `*.json` / `*.jsonl` / `*.txt` are
NOT matched in the data repo) OR when `upload_file` / `upload_folder`
force-routes it at >10MB — which explains the #552 canary results from the
same day (small text/JSON and ~10MB files to the dataset repo PASS; ≥~30MB
LFS uploads — adapters, safetensors, merged dirs — FAIL on BOTH the model and
dataset repos). Recovery ordering:
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
fails), pull the adapters off the pod to the VM
(rsync — installed on bootstrapped pods by `bootstrap_pod.sh` Step 2 since
2026-06-12, commit `22e1a882a1`; on a `--no-bootstrap` pod use tar-over-ssh:
`ssh <pod> 'tar -C /workspace -cf - <adapter-dir>' | tar -xf -`) into a local
staging dir
`eval_results/issue_<N>/adapter_backup/<cell>/` (local staging only —
`*.safetensors` is gitignored; the "eval_results/ is JSON/text only" rule
governs what gets committed) AND log a WandB Artifact (`type="model"`) copy.
(4) Retry the canonical HF model-repo upload only after quota is freed.
Freeing quota means deleting existing HF artifacts — that is USER-ONLY:
surface the situation to the user, never auto-delete from HF.
Diagnosis probes: sum account usage via
`/api/{models,datasets}/<id>?expand[]=usedStorage` over
`list_models(author=...)` / `list_datasets(author=...)`; a tiny non-LFS `.txt`
upload probes the regular-blob path; a tiny `.bin` upload to the private repo
probes the private-LFS path. (Incident #541, 2026-06-10: 11.3 TB public
across 414 repos — 10.2 TB in `superkaiba1/explore-persona-space` alone —
killed the sweep's first upload; #552 hit the same wall the same day.)

**Proactive detection (#564): soft-ceiling headroom check + minute-1 persist
gate + opt-in overflow routing.** `check_hf_storage_headroom()`
(`orchestrate/hub.py`) sums per-repo `usedStorage` over the account's public
repos behind a 1h on-disk cache; knobs: `EPM_HF_STORAGE_SOFT_CEILING_TB`
(default 10.0 — the wall was ~11.3 TB), `EPM_HF_STORAGE_CACHE_TTL_S`,
`EPM_HF_STORAGE_CACHE_PATH`, kill switch `EPM_HF_STORAGE_CHECK=0` (the ceiling
/ routing / check / TTL envs are threaded through the slurm + gcp passthrough
allowlists; the cache-path + event-path envs deliberately are NOT). Preflight
surfaces it as a WARN-only `HF storage:` line.
`trainer.py::_validate_persist_headroom` — called at the top of `_init_phase`
AND at the start of `sft.py::train_lora` — aborts a persist-declared run
(`EPM_PERSIST_ADAPTER_HF_REPO` set) in minute 1 when a forced LIVE re-probe
confirms the account is over the soft ceiling and the persist target is
public with routing off (unknown headroom / undeterminable privacy fail
OPEN — the upload-time backstop above stays authoritative).
`EPM_HF_OVERFLOW_ROUTING=1` (default OFF) makes `upload_model` reroute LFS
uploads to the private overflow repo when KNOWN-over-ceiling, creating it
private if missing, appending a deviation event to `EPM_HF_OVERFLOW_EVENT_PATH`
→ `/workspace/logs/hf-overflow-routing.jsonl` →
`~/.cache/explore_persona_space/hf-overflow-routing.jsonl` (the orchestrator /
upload-verifier observing that sentinel posts the actual `epm:` plan-deviation
marker — pod-side code never shells `task.py`), and committing a small
`OVERFLOW_POINTER.json` breadcrumb (`{overflow_repo, path_in_repo, ts,
used_tb, ceiling_tb}`) to the CANONICAL repo at
`<path_in_repo>/OVERFLOW_POINTER.json` (non-LFS, so it works over quota).
ARMING CONTRACT: routing is safe ONLY for flows that consume `upload_model`'s
returned URL or read the pointer/deviation records — launchers that verify
CANONICAL paths externally (the i528 family) must NOT arm it, because a
reroute converts their 403 into a post-training verification abort. Dataset /
raw-completion paths are deliberately un-routed (non-LFS JSON keeps flowing;
sharding stays the big-text remedy). New per-issue scripts should prefer
`upload_model` over direct `HfApi` calls for LFS artifacts so they inherit
this guard.

**Size-aware projected-headroom probe (#1034).**
`hub.check_projected_upload_headroom(projected_bytes)` compares a PLANNED LFS
upload's byte size against the REMAINING headroom (`used + projected >
ceiling`), which the binary #564 check cannot do. Verdicts: `below-threshold`
(projected under the probe floor `EPM_HF_LARGE_UPLOAD_PROBE_GB`, default 100
decimal GB — ZERO headroom I/O) | `disabled` | `unknown` (fail-open — callers
never block/reroute; the reactive 403 backstop stays authoritative) | `fits` |
`insufficient` (only after a `force_refresh=True` LIVE confirm — never act on
a ≤1h-stale cached over-read). Three consumers: (1) **`upload_dir_sharded`
routes ALL shards to the private overflow repo UP-FRONT** on
KNOWN-insufficient + confirmed-public canonical target (one
`OVERFLOW_POINTER.json`, one JSONL event with
`reason: "projected-headroom-proactive"` + `projected_gb`, zero canonical LFS
bytes attempted; opt out with `proactive_overflow=False` for a
canonical-path-verifying caller) — **route ≥100 GB stores through
`upload_dir_sharded` explicitly** so they inherit this; (2) armed
`upload_model` (`EPM_HF_OVERFLOW_ROUTING=1`) reroutes when
`used + dir_size > ceiling`, not only when already over (ARMING CONTRACT
unchanged: default-off, zero headroom I/O unarmed); (3) preflight
`--planned-upload-gb <N>` turns the WARN-only advisory into a hard gate
(LIVE-CONFIRMED-insufficient + routing off → FAIL; armed → WARN;
unknown/disabled → WARN). Residual routes the guard does NOT cover:
`hub._upload`, `hub._upload_folder_filtered`, and direct-`HfApi` per-issue
scripts — the preflight plan-projection gate covers plan-declared big uploads
regardless of helper, and the 403 stays fail-loud, but do not mistake the
guard for fleet-wide coverage. Note overflow-repo artifacts are PRIVATE —
downstream consumers reach them auth-required and pointer-mediated, never as
canonical-path equivalents.

**File-count limit (100k) — reactive overflow fallback (#1108).** HF
hard-rejects any push that would put a repo over 100,000 git files ("Your git
repo would contain N files after this push, over the limit of 100000 files" —
#1090's rejected c5 ladder push; the canonical model repo sits at the limit).
`hub._upload` catches that rejection on a MODEL-repo upload and retries the
identical upload against the private overflow repo
(`DEFAULT_OVERFLOW_REPO`), then emits the #564 routing event
(`reason: "file-count-limit-reactive"`) and writes the
`OVERFLOW_POINTER.json` breadcrumb at the canonical path — the pointer itself
ADDS one (non-LFS) file to the canonical repo per reroute and fails soft at
exactly 100,000. **Default ON** (kill switch `EPM_HF_FILECOUNT_FALLBACK=0`):
unlike the #564 byte-quota routing (default-OFF because a pre-emptive reroute
can divert a would-succeed push), this fires only AFTER the server refused
the canonical push, so it can never reroute a push that would have succeeded.
Detection is message-substring based (the exception class of the rejection is
unverified); the rejection message changing shape degrades to today's
fail-soft `""` — never a wrong reroute. **This is a TEMPORARY DURABILITY
fallback pending the user's file-count triage (#1108's audit + freeing
package), NOT a transparent successor to canonical storage** — overflow
artifacts are PRIVATE and pointer-mediated (see the paragraph above).
**i528-family caveat (the REACTIVE analogue of the #564 arming contract):** a
persist-gated flow (`EPM_PERSIST_ADAPTER_HF_REPO`) that previously failed
LOUD at the gate on a file-count rejection now proceeds on a VERIFIED private
overflow landing — the artifact is durable and the returned path is the
overflow path, but an EXTERNAL launcher that verifies CANONICAL paths fails
LATER (at its own verify), not earlier; such launchers should set
`EPM_HF_FILECOUNT_FALLBACK=0`. **Concurrent-deletion race (harmless):** if
the user's freeing lands between the rejection and the overflow retry, the
upload simply lands on overflow with a pointer — durable either way, and the
next upload takes the canonical path again. **Scope:** `repo_type="model"`
via `upload_model` → `_upload` only — the ~1M-file DATA repo empirically
still accepts pushes (enforcement is not uniform across repos; a future
risk, not a current one), and direct-`HfApi` per-issue scripts,
`upload_dir_sharded`, and `_upload_folder_filtered` are named residuals
outside this fallback.

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
