# Issue 952 GPU dispatcher

`scripts/issue952_china_dispatch.py` has two explicit blocking stages. Each runs
`gen → upload-raw → capture → upload-capture → finalize → verify` through the
existing GPU child in separate processes. Counts derive from the audited accepted
bank; smoke draws are isolated and never copied into production generations.

1. `--stage smoke` finishes and verifies GPU smoke, writes a nonblocking
   `epm:smoke-result` envelope with `status=smoke_complete`, emits exactly one
   `[phase=done]` for that pod invocation, and exits without `epm:results`.
   Persist and verify the evidence, then terminate
   the smoke pod before running the Codex judging smoke off-pod.
2. The VM owner prepares/collects the Codex packet request/result/parse smoke and
   publishes its technical gate plus exact smoke receipt to an immutable Hub
   revision. Numerical agreement misses are advisory under plan v17.
3. `--stage production --smoke-gate-revision GATE_SHA` hydrates and validates that
   gate and its exact GPU evidence before generating any production rollout.
   Only verified terminal production uploads permit the single `[phase=done]`
   and task #952 results envelope. Full production judging/analysis remain pending.

The dispatcher never invokes a judge API and clears external judge API keys in
child environments. No GPU is held waiting for Codex. A technical gate/fence
failure is loud, writes a blocking progress sentinel, and returns 7; crashes and
data-integrity/upload failures never produce success. The launcher never stops
its own pod.

## Immutable gate contract

Inputs use marker revision `53cde302c71f45c3f4225e62589bce29d3254157`; its immutable
`data_revision` supplies the hash-verified bank/audit. Input paths remain canonical.
All outputs use local `RUN_ROOT/attempt<N>` and Hub
`issue952_position_divergence/followups/china_refusal_topic_stratified_bilingual_v1/attempt<N>`.
For normal dispatch, `--run-root` is the base directory (default
`/workspace/issue952_china_definitive`); the launcher appends `attempt<N>`
automatically. Do not append it yourself. Internal staging/hydration/verification
workers receive the already resolved attempt directory without another suffix.
Start with `--attempt 1`; invalidated regeneration requires a reviewed increment,
not a fresh local root targeting the same remote attempt.

At `GATE_SHA`, publish these exact paths below the attempt prefix:

- `dispatch_state/codex_smoke_gate.json`
- `dispatch_state/smoke_verified.json` (the dispatcher's original receipt bytes)
- `judge/smoke_request_manifest.json`
- `judge/smoke_scores.jsonl`
- `judge/smoke_parse_manifest.json`
- `judge/smoke_packet_manifest.json` and `judge/smoke_lookup.json`
- `judge/smoke_runtime_identity.json` (pre-empirical, distinct execution lanes)
- Every `judge/agent_artifacts/gpu_smoke_attempt<N>/<agent>/batch_NNN.packet.json`,
  `batch_NNN.output.jsonl`, and `batch_NNN.output_manifest.json` (the exact
  original packets, raw agent outputs, and runtime/session manifests)

The gate must contain `schema_version=1`, `kind=issue952_codex_smoke_gate`,
`passed=true`, and these exact identity keys: `code_sha`, `input_revision`,
`attempt`, `accepted_source_ids_sha256`, `smoke_report_sha256`,
`smoke_rollouts_sha256`, `smoke_upload_revision`, `smoke_upload_receipt_sha256`.
The smoke-complete envelope provides this identity. `technical` must have
`request_created`, `result_received`, `parse_complete`, `coverage_complete` all
true; `request_sha256` and `result_sha256` must be full SHA256 hashes. Numerical
agreement fields remain advisory, distinct from technical `passed`.

`evidence` must map `request`, `result`, and `parse` to their exact relative
`judge/...` paths and `sha256` values. Production stages all three at `GATE_SHA`
and verifies the actual bytes; plausible-looking hash strings alone never pass.
The request manifest binds identity, `n_requests`, and `ordered_item_ids_sha256`.
The result JSONL must contain exactly the unique smoke rollout IDs and strict
boolean `verdict` values. The parse manifest has `schema_version=1`,
`kind=issue952_codex_smoke_parse`, the exact same `identity` and `technical`, and
`coverage={n_smoke_rows,n_parsed_rows,ordered_item_ids_sha256}`. Its count and
ordered identity hash are independently recomputed from smoke rows (the hash
uses the producer's `json.dumps(item_ids, sort_keys=True)` byte domain).

`artifact_census` maps the complete judge artifact set above to SHA256 values.
Production stages every member at `GATE_SHA` and checks the exact census, not
only its summary manifests. Request/parse links must match packet-manifest and
opaque-lookup hashes. Every original packet must contain the full committed
rubric and byte-faithful smoke question/response payload; assignments and opaque
IDs are recomputed. Raw result IDs, ordering, strict binary tags and completeness
are checked with the real producer parser, and each parsed primary score must
match its raw output. `agent_artifact_hashes` must agree across gate, parse
manifest, and the staged packet/output bytes. This loads only local packet
validation helpers, never a model API.

The request's `runtime_identity={path,sha256}` names the exact runtime file in
the census. The packet manifest, each packet/output manifest, and parsed scores
must bind its **file-byte SHA256**. Runtime records require distinct agent IDs
and canonical task names and the producer's registered model, reasoning effort,
service tier, and fresh-context setting. Unexposed execution fields remain
explicitly unavailable; they are never inferred as successful API outcomes.

Opaque IDs and classifier request hashes are lane-specific and bind the attempt,
complete question/response/rubric payload, and runtime identity. Lookup records
carry `opaque_ids` and `classifier_request_sha256_by_agent` plus primary-lane
aliases. Raw output rows have exactly five fields: `opaque_id`,
`classifier_request_sha256`, `verdict`, `raw_output`, and `assigned_identity`.
The real parser verifies their per-output manifest's row count, ordered IDs,
packet/output/runtime hashes, identity, and exposed/unavailable snapshot.

The receipt contains `revision`, `hf_prefix`, the complete relative-path-to-SHA256
mapping `sha256`, `input_revision` (marker SHA, not data SHA), `code_sha`, `attempt`,
and `smoke_wall_seconds`. Production downloads the gate/receipt and judge evidence
at `GATE_SHA`, then the smoke generation/capture/timing manifests and rollouts at
the receipt's immutable GPU revision. Every hydrated file must match its verified
receipt hash. Code, accepted identities, inputs, attempt, report, rollout, upload,
and receipt identities must match. Missing or stale gates cannot enter production.

## Launch and observation

After committing and pushing all reviewed child/dispatcher edits, resolve the
branch's final full SHA and replace `CODE_SHA` below. Never derive it from the
pod's unverified HEAD. The workload checks actual committed scripts/src/configs
bytes before phases and uses the bootstrap-installed environment without syncing.
Use `--dry-run` on the workload locally to inspect five argv arrays without
staging, GPU work, or sentinel writes.

GPU smoke launch (VM owning canonical task state):

```bash
UV_PROJECT_ENVIRONMENT=/home/thomasjiralerspong/explore-persona-space/.venv \
UV_OFFLINE=1 UV_NO_SYNC=1 \
uv run python scripts/dispatch_issue.py launch \
  --issue 952 --intent lora-7b --backend runpod --gpus 1 \
  --repo-branch codex/issue952-china-refusal-definitive-v2 \
  --boot-disk-gb 100 --time-budget-hours 8 --execute-workload \
  --workload-cmd "UV_PROJECT_ENVIRONMENT=/workspace/explore-persona-space/.venv uv run --offline --no-sync python -u scripts/issue952_china_dispatch.py --stage smoke --attempt 1 --expected-code-sha CODE_SHA --input-revision 53cde302c71f45c3f4225e62589bce29d3254157"
```

After smoke persistence, pod termination, and verified Codex smoke publication,
launch production with the same command, replacing workload `--stage smoke`
with `--stage production --smoke-gate-revision GATE_SHA`. Use the same code/input
SHA and attempt. `GATE_SHA` must be the verified immutable publication revision,
not a branch name. The planned `lora-7b` intent supplies the required bootstrap
dependencies, including flash-attn; it does not make this workload train a model.

The RunPod GPU router applies `max(200, --boot-disk-gb)`: requesting 100 GB currently
realizes **200 GB**; exact 100 GB is not expressible on this surface.
`--time-budget-hours` is a SLURM setting, not a RunPod billing TTL. The dispatcher
enforces a combined eight-hour GPU execution fence: production's remaining time
is eight hours minus recorded smoke execution. Stage wall time includes staging
and uploads; off-pod Codex time is excluded. The VM owner must separately monitor
the pod lifecycle and bootstrap/billing overhead.

- Main log: `/workspace/logs/issue-952.log` (router rotates at launch).
- PID: `/workspace/logs/issue-952.pid`, atomically replaced with the driver PID.
- Results: `/workspace/logs/issue-952-epm_results-<unique-timestamp>.json`.
- Smoke result: `/workspace/logs/issue-952-epm_smoke-result-<unique-timestamp>.json`.
- State/receipts/gate: `/workspace/issue952_china_definitive/attempt1/dispatch_state`.
- Fresh per-phase logs: `dispatch_state/logs/<phase>-<unique-timestamp>.log`.
- Smoke/production roots: `/workspace/issue952_china_definitive/attempt1/smoke`
  and `/workspace/issue952_china_definitive/attempt1/production`.
- The router's per-attempt exit sentinel is distinct from task results. Its exact
  path is returned in `extra.expected_artifacts.sentinel_path`; smoke exit 0 only
  means that smoke workload finished, not that task #952 is complete. Smoke and
  production have separate rotated logs, each with exactly one terminal done line.

## Recovery and verification limits

Use the same reviewed code/input/attempt and immutable gate to resume interrupted
production. The ledger is never read from drained `/workspace/logs`. Any ancestor
or descendant overlap between logs and the run root is rejected. Completed
generation/capture phases skip only after their full file/hash census matches.
Uploads/finalize/remote verification always rerun. A local done file alone never
suffices. Preserve and inspect mismatches; invalidated regeneration needs a new
attempt namespace and fresh smoke/Codex gate.

Before parking or terminating either pod, the VM owner must persist and verify
dispatch state, per-phase/main logs, smoke receipt, and required output artifacts
off-pod, and obtain the project's upload-verifier outroot/rows attestation for
the realized stage. Local state is not off-pod durability. Keep the router handle
for polling and finalization, and ensure smoke receipts are copied byte-for-byte.

Smoke blind spots: both stages use the actual child implementation; smoke reduces
coverage to ten deterministic accepted identities and tests frozen-map consumers
in finalize. Production-scale memory/coverage is not established by reduced smoke.
Local tests use synthetic subprocess/Hub boundaries and actual local process
wait/failure/fence checks; they certify neither real GPU behavior nor network
availability. Verified-by: **ran** focused synthetic/static checks and dry-run
argv composition; **read** router/bootstrap and real child contracts. No real
data/network/GPU stage was executed during dispatcher implementation.
