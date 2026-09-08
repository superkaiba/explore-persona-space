# High-rate raw archive to managed GPU teardown

This is a source-bound readiness runbook, not an upload PASS or teardown authorization. Screening is still active; complete screen/fresh/capture evidence does not yet exist. The approved plan explicitly separates this GPU milestone from later VM fits, prediction/metric archival, and final experiment completion. No claim about forecast accuracy follows from this milestone.

## Fixed scope and current observations

- Canonical task root: `/home/thomasjiralerspong/explore-persona-space`; task2670 is `kind: experiment`, `status: followups_running`, with `keep-running`.
- Implementation worktree: `/home/thomasjiralerspong/.codex/worktrees/context-risk-recovery-20260906`.
- RUN: `/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate`.
- Owned pod: `pod-2670-highrate`, ID `1bds7vqkrluxkc`; registered owner `codex-highrate-20260907`, fence through `2026-09-09T23:00:00Z`. These identify the existing owning executor; this review does not claim ownership.
- Pod output root: `/workspace/logs/issue2670-context-risk-highrate`. A bounded live listing found17 files:6 in `server/`,11 in `setup/`. This is a progress observation, not the final inventory; the server log is growing and server/capture shutdown receipts will add files.
- Existing pilot archive is at dataset `superkaiba1/explore-persona-space-data`, revision `d2d3987eaae9f68902d79a4e51bf9735aa99df7d`, prefix `context_risk/issue2670_highrate/pilot`. Its successful readback does not cover subsequent full generation or capture.
- Actual snapshot keys have namespaces such as `run/`, `code/`, and `archive_code/`. Example: `run/setup/map_layer_44.npz`. Never infer the mapping by stripping an arbitrary prefix.

## Work that should be implemented before the GPU becomes idle

Add a separately reviewed archive-only reconciliation helper; leave the frozen collection/capture/analysis closures and generic verifier unchanged. It must accept explicit original-RUN, raw upload/readback receipt, immutable raw snapshot, readback directory, stopped-pod inventory, and source-to-snapshot mapping inputs. Require nonempty, phase-matched, source-bound inputs and an unused output directory. Reject symlinks/path escape, missing keys, wrong-key substitutions, duplicate authoritative rows, unmatched files, drift during verification, and unverified local-only inputs. Its receipt should include `passed`, `phase=raw`, source/package hashes, archive revision/prefix, input receipt hashes, exact expected/realized key sets or canonical set digests with set-difference evidence, per-source row/hash proofs, per-file pod/destination proofs, generic verifier results, explicit resolved limitations, and remaining VM work. It must neither post markers nor terminate compute.

Two experimentally confirmed compatibility seams require this helper:

1. `scripts.verify_uploads._count_row_index_file` uses `str.splitlines()`. A single LF-terminated JSON row containing literal U+2028 is parsed as2 lines and produces2 JSON errors. The collection and V2 archive readers correctly iterate physical file lines. Derive small ASCII identity indexes from the actual pinned reconstructed rows using `json.dumps(..., ensure_ascii=True)`; retain parent-file SHA and source row ordinal. Do not rewrite archived originals or serialize expected keys as if they were observed rows.
2. `check_outroot_residue` uses basenames, exempts common log/PID/cache names, and does not resolve sharded original names. It may accept a same-named wrong file on HF or flag an original JSONL that is durably represented by a manifest and parts. Run it as supplemental evidence, then reconcile every original relative path and SHA against the pinned archive or exact committed blob. Preserve any generic FAIL/WARN with its specific resolution; do not silently turn it into OK, add a broad exemption, or substitute a count comparison.

The upload-verifier reference Step2 explicitly permits direct HF/git verification for artifact classes the generic helper does not support. This permits an explicit stronger per-artifact proof; it does not permit skipping rows, suppressing diagnostics, or accepting unresolved errors.

## 1. Finish and freeze the regeneration-costly work

The owning executor must first finish the fixed618 screening and360 fresh native trajectories, preserve the independent success reviews and applicability/rank/selection evidence, and obtain the existing terminal-process receipts. Censors remain present as unknown outcomes. Preserve structurally invalid test contexts and raw results; they are filtered only by the separately reviewed analysis policy. Do not reduce archival denominators for competence, censoring, low class support, or structural invalidity.

On the canonical VM RUN, invoke the actual read-only validators:

```python
from pathlib import Path
from scripts import context_risk_highrate_design as design
from scripts import context_risk_highrate_collect as collect
from scripts import context_risk_highrate_capture as capture

root = Path('/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate')
for phase in ('screen', 'fresh'):
    manifest, frozen, epochs = design.load_phase(root, phase)
    audit = collect.verify_report(root, phase)
    terminal = design.validate_terminal_process(root, phase)
binding = capture.validate_binding(root / 'capture')
```

Use the existing VM runtime with `UV_NO_SYNC=1`,8-thread caps and `uv run --with inspect-ai==0.3.261 --with openai==3.7.0 python`. These functions do not generate samples. `verify_report` rereads full native logs, exact request evidence/seeds/configs, first-request token usage and saved raw rows; `validate_terminal_process` checks actual supervisor/worker/exit/native chronology and absence of the owned process group. `validate_binding` additionally checks all six capture shards, row/token equality, layer44, float16 serialized shape `(15,1,5120)`, finiteness and exact source/model/position binding.

These APIs embed original VM paths and compare the configured RUN. Do not invoke them on a differently rooted readback directory, fabricate VM paths on the pod, edit archived reports, or substitute symlinks. Instead, validate the original inputs and prove that every consumed file is byte-identical to the pinned archived original; recompute actual archive row identities directly from those readback bytes. Include source/plan/review files referenced outside RUN through an explicit mapping to archived `code/` or `archive_code/` entries. Hash before and after semantic validation.

Complete the approved capture smoke, full90-context capture and capture exit/cleanup receipts. Stop only the already owned server through its reviewed supervisor procedure, with exact PID/command identity and exit/cleanup evidence. This stops a workload before inventory; it is not pod destruction. Never call `context_risk_corrected_finish.main`, `finish`, `upload_bounded`, or its old hardcoded pod functions.

## 2. Inventory and copy the complete final pod outputs

After generation/capture/server writers have ended, produce a fresh pod-side inventory of every regular file under the exact output root, including top-level files, logs, PID/lock files, setup receipts and all capture/control outputs. Record relative path, byte size and SHA256, pod name/ID, UTC time and the checked writer/process identities. Reject an empty inventory. Check for symlinks and unexpected file types explicitly. Also enumerate the actual capture staging/output and smoke directories from their launch receipts; if any experiment output is outside the named outroot, inventory and archive that owned root separately with an explicit namespace. Model/cache dependencies are not an excuse to omit generated text, configs or control tensors.

Copy those files to VM and compare the full relative-path set and every byte hash against the pod inventory. Repeat the pod inventory at the end of the copy and immediately before the final gate; any new or changed output needs persistence and renewed verification. A later verification receipt written only on VM belongs in the committed late-evidence set below. Avoid writing new verification outputs into the pod tree after its frozen census.

The current RUN already has partial `pod_setup/` copies; do not treat their existence as a final copy. Archive a clearly mapped complete final pod namespace, without basename flattening. The raw snapshot must also contain full VM native `.eval` files, native audit/rollout records, pilot and resume evidence, both manifests and prefix-token files, selection/ranking/role evidence, fresh success review, applicability policy/evidence,90-context capture payloads and sentinels, capture input/launch bindings, smoke outputs, source/review closures, and the exact frozen map payload plus provenance. Preserve started review ZIPs with their sidecars/references; they remain `terminal=false` and are not additional trajectories.

## 3. Run V2 archival and retain the exact readback proof

Use the already reviewed V2 wrapper only, on a pristine working copy of the immutable raw snapshot and a fresh disjoint readback directory. Root/source/phase arguments must match the actual snapshot. The wrapper requires `EPM_HF_FILECOUNT_FALLBACK=0`; it writes to `context_risk/issue2670_highrate/raw` and verifies exact remote filename/size/hash sets at its returned immutable revision. Preserve existing immutable stages when retrying: shard creation mutates the working stage, so replay starts from a new pristine copy. Precompute an explicit transfer deadline and bounded retry budget from the measured file count/bytes; record exit status and fresh sentinel. Do not use an unbounded wait, count-only upload proof, fallback destination, or an unchecked nonzero exit.

```text
UV_NO_SYNC=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 EPM_HF_FILECOUNT_FALLBACK=0
uv run --with inspect-ai==0.3.261 --with openai==3.7.0 python -m scripts.context_risk_highrate_archive --root RUN --stage PRISTINE_RAW_STAGE --readback FRESH_RAW_READBACK --phase raw
```

The command template's paths and environment are supplied by the actual supervised launch receipt; `RUN`, `PRISTINE_RAW_STAGE`, and `FRESH_RAW_READBACK` are not literal directories to create. Existing V2 tests need not be rerun absent source changes.

The wrapper's `stage_hub_prefix(..., revision=upload_revision)` lands full repository-relative names under readback. For a snapshot key `name`, use:

- JSONL: `readback/reconstructed/name`, reconstructed by the canonical manifest-first `stage_sharded_text` at the same revision.
- Everything else: `readback/<upload_receipt.prefix>/name`.

Require every reconstructed original's size/SHA to equal `snapshot_manifest.json`, and every remote file's size/SHA to equal the exact upload receipt. Normal `.eval` must be successful terminal native logs; designated review `.zip` files alone may be started with validated sidecars. Raw CR JSONL, oversized individual JSONL rows and unsupported oversized text fail loudly. A crashed phase is an explicit recovery/archive case, never a reason to discard its native file or weaken V2's success-only terminal parser.

## 4. Reconcile exact archived row identities

For each phase obtain expected keys from the frozen input roster:

```python
manifest, frozen, epochs = design.load_phase(root, phase)
samples = collect.load_samples(manifest)
expected = collect.expected_keys(samples, epochs, pilot=False)
```

Require618 expected `(sample_id, epoch)` keys for screen and360 for fresh. Derive actual keys by physical-line iteration of byte-verified reconstructed `run/screen_B/rollouts.jsonl` and `run/fresh_B/rollouts.jsonl` (or the explicitly declared equivalent snapshot mapping), and cross-check against the complete native audit/readback. Require exact expected-versus-actual set equality and no duplicate authoritative phase rows. Do not concatenate `pilot_rollouts.jsonl`, started review snapshots or duplicated archived pilot/full logs into the final tally. The32 pilot observations are retained once inside618; the original pilot evidence stays archived separately.

For capture derive90 expected `(task_id, condition, exact_context_sha256)` keys from the verified fresh manifest. Enumerate the actual six reconstructed `chunk_0000.rows.jsonl` through `chunk_0005.rows.jsonl`, retain their source SHA/ordinal, and require exactly the expected ordered rows with the matching tensor/sentinel proofs. Capturing one context covers all four stochastic continuations; it does not create360 activation rows.

Write ASCII identity-only projections under a fresh proof tree: `screen_B/row_index.jsonl`, `fresh_B/row_index.jsonl`, `capture/row_index.jsonl`. Each row has a `key` array containing the full generation or capture key; provenance can be recorded alongside it. Observations come exclusively from pinned readback bytes. Invoke the real generic API:

```python
from scripts.verify_uploads import check_realized_row_counts
result = check_realized_row_counts(
    expected_rows={'screen_B': 618, 'fresh_B': 360, 'capture': 90},
    local_root=str(proof_tree),
    glob_pattern='row_index*.jsonl',
    distinct_key_fields=('key',),
)
assert result['status'] == 'OK'
```

The result includes `labels[label]` with expected, realized_lines, realized_distinct, duplicates, shards, key_fields, verdict and tag. Require all three labels, no missing/extraneous label, duplicates0, exact expected set equality from the stronger preceding check, and hash-bound projection provenance. Do not supply an empty expected dictionary, use `rows=n/a`, declare an exemption, gate on producer counts, or combine local and HF row sources. The local API is appropriate only because these files are derived from independently verified pinned downloads; a local-only original is not durability evidence. Preserve the projections and their proof receipt in issue-scoped git with the late receipts.

## 5. Reconcile every file and persist late receipts

Run the supported supplemental `check_outroot_residue(issue_num=2670, outroot_listing=<final_listing>, hf_prefixes=(<actual_raw_prefix>,), data_repos=(<actual_repo>,))`, or its corresponding CLI. Record its actual result. Use only narrow actual prefixes. It resolves its own current HF revision rather than accepting the V2 receipt pin; therefore it cannot replace the pinned per-file proof.

The authoritative stronger table must cover every final pod-relative file exactly once, including built-in-exempt logs/PIDs, with its original size/SHA and either (a) a mapped snapshot key and pinned reconstructed archive SHA, including the manifest/part mapping if sharded, or (b) an exact issue-scoped committed git path/blob. Compare path sets in both directions against the source inventories and approved snapshot mapping, not just counts. A same basename or a same-sized file is insufficient. Any discrepancy remains a blocking failure. Explicitly resolve any generic sharding false positive through this per-file table; do not claim the generic helper itself passed when it did not.

Upload/readback/reconciliation completion receipts cannot be inside the transfer they attest. Commit their exact bytes, plus supplemental reports/projections, to `eval_results/issue_2670/highrate/` through the owning worktree and verify them with `git ls-tree`/`git cat-file` at the pushed commit. Record every copied source-to-git mapping and compare blob bytes/SHA. A small local receipt or uncommitted file is not permanent evidence. Use explicit staged paths; never edit task paths directly.

Verify all URLs claimed for this raw milestone at their cited HF revision or git commit; include any relevant existing task result/body claims and document older phase scope separately. `verify_uploads.py --type eval-only` avoids inventing new model-training/WandB requirements for this inference-only round. Disclose missing prior-map nearest-neighbor retrieval under the existing bounded map evidence review; do not invent a new measurement or make it a completed fit result. All GPU-dependent planned inputs must now be durable, while VM fitting/report outputs are explicitly future work under the two-milestone plan.

## 6. Post the concrete owner-bound PASS and use managed teardown

Only the owning executor may execute this step, after all preceding evidence exists and is independently checked. From the canonical shared root, first use `uv run python scripts/task.py view 2670 --json` to verify the task resolves, current Goal/status, owner/fence and latest marker. Do not accept a lifecycle warning that skipped task lookup or an owner check. Dry-run output does not exercise those guards.

Prepare a small durable note with real evidence references (replace all placeholders):

```text
Verdict: PASS
Scope: completed highrate raw/GPU archive milestone; VM fits and final report/archive remain outstanding under the approved two-milestone plan.
pod=pod-2670-highrate owner=codex-highrate-20260907
outroot=residue-committed rows=reconciled
repos=superkaiba1/explore-persona-space-data
Raw archive: <actual pinned HF revision and prefix URL>
Readback and semantic proof: <verified git commit/path/SHA>
Exact full-key coverage: screen_B618; fresh_B360; capture90; expected and observed key sets equal; no omitted censors/invalid contexts.
Complete stopped-pod file mapping: <inventory SHA and exact per-file proof>; late receipts: <committed mapping>.
Generic verifier limitations: <specific diagnosed cases and full resolutions, or none>.
All GPU-dependent plan inputs are durable; no scientific experiment-completion or predictive-benefit claim is made here.
```

Use `outroot=swept-clean` only if every artifact, including late receipts, already has verified coverage without the git-residue disposition; `residue-committed` is the expected honest choice here. `outroot=none`, `rows=n/a` and `rows=no-declared-count` are false for this run.

```bash
uv run python scripts/task.py post-marker 2670 epm:upload-verification --by codex --file VERIFIED_NOTE_PATH
uv run python scripts/task.py view 2670 --json
uv run python scripts/pod.py terminate --issue 2670 --name-suffix highrate --yes
uv run python scripts/pod.py list-ephemeral --issue 2670
```

These are execution instructions for the owner, not commands run by this review. Re-read the posted latest PASS and its pod/owner/tokens before termination. The surgical suffix avoids other issue2670 pods; no `--approve`, `--skip-upload-verify`, force flag, environment bypass or direct API delete is needed. The managed path grants its existing verified owner teardown authorization. Keep the task-level shield if another live round needs it; root uses `task.py` for any tag/status update.

Finally require authoritative live API absence of the exact pod ID/name, recording the actual returned IDs. `cmd_terminate` itself treats just-requested IDs as asynchronously terminating, so its success line alone is not the final absence proof. Use bounded follow-up reads, and report any still-present target. Preserve all VM originals and verified readbacks; no deletion is required. Continue the approved VM fits/final archive after GPU release and do not mark the experiment complete at this raw milestone.

## Evidence executed for this review

Read the current upload policy, upload-verifier reference, pods rules, plan and actual archive/collection/capture/design/verifier/task/pod implementations. Confirmed canonical and worktree bytes match for generic verifier/pod/task implementations. Ran the actual row API on a temporary ASCII full-key fixture; reproduced the Unicode physical-line defect; executed the real extracted pure marker parsers on positive/missing-token/FAIL cases without task mutation. Read task2670 through `task.py`; read the17-file live pod listing; inspected the existing pilot receipts and namespace mapping. No upload, model call, source edit, process stop, task edit or termination was performed. V2 archive tests were not repeated.
