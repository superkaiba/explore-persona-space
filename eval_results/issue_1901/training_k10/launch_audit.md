# #1901 training-k10 launch audit (read-only)

Audit scope: existing 19,000-context pool, five new answers per context (95,000 new draws), Qwen2.5-7B-Instruct layer19 answer-span capture, then new maps at k_train=1..10. No compute provisioned, stopped, or mutated; no task state changed; no Claude automation invoked.

## Recommendation

Reuse the current unified router with a dedicated suffixed capture lane, default `auto` (GCP first, paid RunPod fallback). Request 4 shardable GPUs, or 2 if provisioning/boot overhead dominates the pilot. The workload MUST consume actual allocated GPU count and preserved CUDA_VISIBLE_DEVICES slots: GCP may degrade 4 -> 2 -> 1. Use `capture-7b`, not `eval`: current GCP capture-7b maps to A100-80; eval can select insufficient-HBM hardware. Sequential fresh vLLM generation and HF capture processes per worker avoid co-resident engines. Partition context shards across workers and keep each context's seed assignments fixed. CPU staging runs on its own cheap CPU lane if a bulk download is needed; avoid loading a whole capture corpus on the shared VM.

Current verification:

- `verified-by: ran`: dispatch launch/finalize and backend_poll help paths execute; current router order is `(gcp, runpod, nibi, fir, mila)`, GCP disabled=False, fellows revoked=True; capture-7b is width eligible and maps to `a2-ultragpu-1g`.
- `verified-by: ran`: gcloud executable exists, `eps-gcp` resolves project `eps-persona-gpu-jun2026`, authenticated read-only instances-list succeeds. No #1901 GCP instances found in this audit.
- `verified-by: ran`: .env contains nonempty HF_TOKEN, GITHUB_TOKEN, RUNPOD_API_KEY (values never exposed). HF whoami and the project RunPod API's list_team_pods succeed. No #1901 RunPod pods found. RunPod Python package is absent but not needed: project uses `scripts/runpod_api.py`.
- `verified-by: ran`: installed local versions torch 2.8.0, transformers 4.57.6, vLLM 0.11.0, huggingface_hub 0.36.2. These are local package metadata; GPU-worker import/runtime still needs its own fresh smoke.
- `verified-by: ran`: previous k10 execution log has generation 852.8 s / 5,000 draws and capture 361.8 s / 5,000 draws. Scale by19 -> 6.4104 H100 GPU-h, excluding startup/parity/end overhead. Generation 500-row chunks varied 53.9–157.2s; do not promise a narrow wall interval. Ideal 4GPU compute ~1.60h or2GPU ~3.21h on identical H100 throughput. GCP A100 throughput is unmeasured for this workload and requires an on-venue representative pilot. Input/length differences also require a pilot.

## Proven successful driver paths

1. `/home/thomasjiralerspong/explore-persona-space/scripts/issue1901_k10_capture.py`
   - MODEL_REVISION a09a35458c702b33eeacc393d103063234e8bc28.
   - Parent same model recipe: request seeds47..51; engine seed42; temperature1, top_p.95, max_tokens1024; vLLM max_model_len8192, gpu_memory_utilization.60, max_num_seqs64, chunks500.
   - `capture_rows` calls the exact parent prompt-render, prompt-ID join, full-template retokenization and answer mean over tail including EOT; capture batches32 and token budget32768; outputs fp16,3584 dimensions.
   - `capture` validates 32 source draws at cosine>=.999, then writes npz with identities, seed, full recipe, generation SHA. Generation/capture are separate `subprocess.run(check=True)` children.
   - Do NOT invoke this high-level script for 19k: it is hardcoded to the 1,000-row test bank, two500-row chunks, negative test ci values, and fixed old output prefix. Reuse primitives only, with new manifest/coverage/recipe assertions and new prefix.
   - Its end-of-run envelope fits poll_pipeline, but it does not itself call the router's `write_completion_sentinel` at EPS_SENTINEL_PATH; new router-launched work should do both (see below).

2. `/mnt/eps-data/thomasjiralerspong/wt-1901-boundary-25k/scripts/issue1901_boundary_25k_gpu.py` at worktree commit c8eea3677e6 (execution code af4d08df494a2da48a635574d4e80b57872b449e).
   - `child_wave`: fresh subprocess per allocated CUDA slot, rank log, wait loop checks nonzero children, terminates siblings on error; blocking controller. `controller`: actual allocated GPU count, partition manifest, pilots, row-complete capture validation, capture upload before fits, final report upload before completion.
   - `worker`: runtime+hardware keyed pilot, length-quartile representative timings/p90 and longest-shape warmup; strict content-based row validation.
   - `fits`: checkpoint each fit and predicted vectors, full recipe + source hashes. Good structure to reuse, while replacing science with the k-study protocol.

3. `/mnt/eps-data/thomasjiralerspong/wt-1901-boundary-25k/scripts/issue1901_boundary_25k.py`
   - `upload_verified` (line95): one upload_folder per directory, pin resulting commit; scoped list_repo_tree; assert every path, byte size, LFS SHA256 or Git blob digest; returns exact prefix/revision/files/bytes receipt. Add `hub.retry_transient` around both upload and materialized listing in new code; current helper has no retry.
   - `stage` (line140): scoped list and max4 workers; no whole-repo snapshot_download. Add retries and source-pinned revision for study inputs. It copies HF-cache files and can double local disk footprint; size this honestly.
   - `complete` (line535): `backends.artifacts.write_completion_sentinel(sentinel_path=os.environ['EPS_SENTINEL_PATH'], issue=1901, extra=...)` and `[phase=done]` after upload verification.

Previous #1901 CPU+GPU router success is independently visible in finalized sidecars:

- `/home/thomasjiralerspong/explore-persona-space/.claude/cache/issue-1901-btok25k-prep-handle.json.finalized`: GCP n2-highmem-16,120GB disk,min_ram64,cpu-bigmem,4h budget, workload `uv run python scripts/issue1901_boundary_25k.py --phase prepare --out-root /workspace/boundary25k --workers 12`.
- `/home/thomasjiralerspong/explore-persona-space/.claude/cache/issue-1901-btok25k-cap-handle.json.finalized`: GCP FLEX_START a2-ultragpu-2g,120GB disk,min_ram64,capture-7b,gpus2,4h budget, workload `uv run python scripts/issue1901_boundary_25k_gpu.py --out-root /workspace/boundary25k-gpu --batch-size 8 --max-capture-hours 3`.
- Those runs used `--skip-default-git-paths`; custom workload expected HF path sets are EMPTY in those sidecars. Therefore driver byte/hash receipts + independent exact manifest verification are essential; bare router sentinel success alone is insufficient proof of uploaded completeness.

## Launch and monitor command surfaces

The following is a launch TEMPLATE (`verified-by: read`, syntax flags verified by CLI help; new script is not implemented by this audit). Parent must use the actual reviewed committed new driver path/flags and the pilot-sized budget, and push branch before launch:

```
uv run python scripts/dispatch_issue.py launch \
  --issue 1901 --intent capture-7b --gpus 4 \
  --repo-branch codex/1901-training-k10-20260911 \
  --lane-suffix traink10-cap \
  --time-budget-hours 8 --max-run-duration 8h \
  --boot-disk-gb 120 --min-ram-gb 64 \
  --env-pin OMP_NUM_THREADS=4 \
  --env-pin OPENBLAS_NUM_THREADS=4 \
  --env-pin MKL_NUM_THREADS=4 \
  --execute-workload --skip-default-git-paths \
  --workload-cmd 'uv run python scripts/NEW_REVIEWED_DRIVER.py --out-root /workspace/issue1901-training-k10'
```

Eight-hour fence is a conservative template, not a measured projection. State the actual staging/capture/fit footprint and pilot p90 margin before fixing it. Sequential fresh-engine capture avoids the >38GiB co-resident-engine floor; >=40GB card remains required by capture-7b. Do not claim spot tolerance unless all partial phase state restores correctly from HF. Prefer blocking workload entrypoint; router handles detachment, pidfile, log and liveness for RunPod and supervises GCP startup. `--execute-workload` is required so an auto RunPod fallback really starts the job.

Monitor using the exact `handle_sidecar_path` returned by launch:

```
uv run python scripts/backend_poll.py --issue 1901 --handle-file /ABS/RETURNED/HANDLE.json
```

There is NO `dispatch_issue.py poll` subcommand (help probe rejects it). backend_poll is the available one-tick poller; it may drain markers/fail over and so was not run on old handles by this read-only audit. Do not monitor/finalize the unsuffixed historical issue1901 handle by accident.

After the fresh terminal sentinel, exact uploaded content/row coverage verification, out-root name-set residue sweep, and matching owner/pod PASS marker through canonical task API:

```
uv run python scripts/dispatch_issue.py finalize --issue 1901 --handle-file /ABS/RETURNED/HANDLE.json
```

Never use skip-confirm-artifacts to bypass incomplete persistence. If direct RunPod lifecycle is needed, only surgical `pod.py terminate --issue 1901 --name-suffix traink10-cap --yes` after the verified-completion gate; do not use issue-wide terminate while siblings might exist. A live owner fence cannot be copied from another session. If teardown refuses, surface the refusal without bypassing it.

## Persistence and completion requirements for new driver

- Publish code plus immutable source/config/manifest identity before launch; source tensors and prompts must be aligned in consumer layout. Preserve exact original test bank and capture convention.
- Keep resume state under out-root, never under the `/workspace/logs/issue-1901-*.json` drained glob. Rewrite launch pidfile freshly on every relaunch; detach all three stdio streams if manual launch ever needed.
- Every generated chunk must preserve all response text and IDs, no discarded generations. Text payloads >9.5MB need project line-sharding/manifest mechanics; new95k expansion must not assume old2chunk shape.
- Keep raw text, arrays, masks/counts, fitted map weights, val choices, predictions, metric inputs, logs, runtime+model versions and exact input/output hashes. Projected new fp16 single-layer answer arrays:95000*3584*2=680,960,000bytes; all190k answers1,361,920,000bytes (before container/metadata). Full source staging could be much larger and must be separately enumerated.
- Use resume-aware `assert_out_root_headroom` at each write-heavy phase. Model cache~15GB plus cache copies, sources, raw text, arrays, fits and logs all count. RunPod `/workspace` has ~130GB per-pod physical quota despite misleading statvfs; GCP uses explicitly sized bootdisk.
- Use scoped HF staging at one pinned revision, max4–6 download workers, retry transient errors. Do not snapshot_download the million-file data repo.
- Prefer background bounded upload overlap for completed shards (`orchestrate.background_upload.BackgroundStemUploader` + `hub_upload_then_free`) when otherwise GPUs would idle during uploads. Join all persistence chains before phase completion. No deletion until content-verified upload. Capture store must be durable before fits.
- Completion requires both the router's fresh attempt sentinel at EPS_SENTINEL_PATH and the correct epm:results envelope for task visibility, followed by single terminal `[phase=done]`. Do not issue task.py calls from pod code.
- Log expected full artifact path set before verification; assert nonempty when outputs declared. Reconcile95,000 distinct new (context_id,seed) rows from tensor/text contents and full required training/eval coverage, not producer-reported counts. Finalized boundary25k receipts did row census independently; reuse that bar.
- Nontrivial linear-analysis phases should use dedicated CPU unless measured batched GPU factorization justifies GPU; do not hold wide GPUs through long single-device/CPU final analysis. Cheap one-off fits may fit within the short-phase allowance, but actual10fit projection needs measurement.
- Canonical local #1901 final boundary25k report records WandB API401, using uploaded offline logs. HF/GCP/RunPod auth is healthy in this audit; do not assume WandB online key healthy without checking. Never conceal tracking failure or suppress compute failure because metrics already exist.

Rule sources read: origin/main `.claude/rules/LESSONS.md`, compute-backends.md, pods.md, relevant plan-compute-sizing.md, pod-side-reporting.md and upload-policy.md sections. Imported Claude automation syntax ignored per AGENTS.md.
