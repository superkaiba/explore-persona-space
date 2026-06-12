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
`set -a && source .env && set +a && uv run python -c "from huggingface_hub import list_repo_files; print('\n'.join(list_repo_files('superkaiba1/explore-persona-space-data', repo_type='dataset', revision='main')))" | grep <bucket>`
(the `set -a && source .env` prefix is part of the canonical snippet — without
it the check dies on `HF_TOKEN missing`, and the obvious in-heredoc fix, a bare
`load_dotenv()`, crashes from stdin; 4+ sessions on 2026-06-10 each burned 2-3
retries re-deriving this)
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
phase — never a warning-and-continue.

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
fails), pull the adapters off the pod to the VM via tar-over-ssh
(`ssh <pod> 'tar -C /workspace -cf - <adapter-dir>' | tar -xf -` — rsync is NOT
installed on RunPod pods) into a local staging dir
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
