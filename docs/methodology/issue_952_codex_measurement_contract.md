# Issue 952 Codex measurement contract and resume

The user-authorized contract is `codex-primary-historical-diagnostic-v1`.
The frozen complete-refusal rubric and `codex-subagent-pair-v2` instrument
remain unchanged. Two independent blinded Codex judges supply the primary
binary labels using the existing deterministic primary assignment. The
unchanged overlap panels are 25% for calibration and 10% for production;
overlap disagreements measure reliability and do not replace the designated
primary label with historical labels or a consensus rule.

Historical labels are diagnostics only. Codex primary refusal rates are not
directly comparable to historical rates. Inter-judge agreement does not
establish historical or human-label validity. Historical confusion matrices,
rate errors, bootstrap bounds, and their retired threshold outcomes remain
reported, but cannot block production. The overall agreement floor 0.90,
kappa floor 0.70, and English/CJK agreement floors 0.85 remain binding, along
with complete coverage, strict schemas, and input hashes.

The bilingual, topic-stratified bank, 8 Qwen draws per prompt, layers 14/19/26,
and registered kernel/SAE analysis are unchanged. The analysis report now
carries the judge contract and comparability disclaimer from the verified
production summary. Canonical experiment plans and status remain owned by
`scripts/task.py`; this file is an implementation/runbook reference only.

## Stopped state and restoration prerequisite

Implementation resumed from commit `8b93a5120d8` on
`codex/issue952-china-refusal-definitive-v2`. The legacy combined calibration
gate was still active. The revised collector separates historical diagnostic
clauses from production reliability clauses and versions the resulting report.
Consumers reject an old or missing contract even when a stored `passed` flag
is true. Recollection reuses the existing corrected judgments, without
changing rubric hashes, item hashes, primary assignments, or packet manifests.

The designated out-dir is `/tmp/issue952_def_reuse`; the packet root is
`/tmp/issue952_codex_packets_v2`. The out-dir currently has no resumable
calibration metadata tree. Restore the complete `calibration_codex/` artifact
set from dataset `superkaiba1/explore-persona-space-data`, revision
`39c29c4ee341dc689d3c3cc4f179c7ff6336c40f`, beneath prefix
`issue952_position_divergence/followups/china_refusal_topic_stratified_bilingual_v1`.
The revision comes from canonical task state. No network restoration was run
in this implementation round.

Before recollection, verify the restored report's manifest, items, packet
manifest, and classifier hashes; verify every archived packet/output against
`agent_artifact_hashes`. Restore the archived packet/output bytes to the exact
paths named by the unchanged packet manifest. Existing different bytes are a
hard stop, not an overwrite. Do not rewrite the manifest to relocate paths or
regenerate packets. If those exact paths cannot be restored, stop before
recollection and repair the restoration layout explicitly. Restore the frozen
source-bank inputs and verify the script's source hash before bank authoring.

## Resume commands and stage boundaries

Run from `/mnt/eps-data/thomasjiralerspong/wt-952-china-refusal-definitive-v2`.
The existing shared environment is required; the worktree has no installed
dependencies of its own. For local CPU commands set:

```bash
export UV_PROJECT_ENVIRONMENT=/home/thomasjiralerspong/explore-persona-space/.venv
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2
```

After the restoration/hash checks above, recollect and persist the corrected
calibration under the revised contract:

```bash
uv run --offline --no-sync python scripts/issue952_codex_judges.py --phase calibration-collect --out-dir /tmp/issue952_def_reuse
uv run --offline --no-sync python scripts/issue952_codex_judges.py --phase calibration-upload --out-dir /tmp/issue952_def_reuse
```

The first command computes aggregate evidence and exits nonzero on any
remaining reliability failure; the second uploads the artifact tree and
verifies the pinned snapshot. Uploads are network stages. Preserve the original
HF revision as the prior-contract record; record the new report hash and new
verified upload revision in canonical task state.

Only after that gate, prepare the bank-author stage:

```bash
uv run --offline --no-sync python scripts/issue952_codex_judges.py --phase bank-author-prepare --out-dir /tmp/issue952_def_reuse --packet-root /tmp/issue952_codex_packets_v2
```

That command prepares opaque files; it does not launch judges. Subsequent
stages are independent blinded Codex authoring, `bank-audit-prepare`, independent
cross-audit, the registered retry stages if needed, `bank-finalize`, then
`input-upload`. Every prepare call takes the same explicit out-dir and packet
root. Do not invoke legacy external-model API stages in the bank/DV scripts.

Repair rounds are explicit and immutable. Round 1 retains the historical
`bank_author_retry` / `bank_audit_retry` paths and manifests; later rounds use
`bank_author_retry_round_N` / `bank_audit_retry_round_N`. Both repair prepare
commands require `--round N` with N >= 1. A round must follow the latest fully
completed author-and-cross-audit round. Existing paths, missing predecessors,
incomplete outputs, schema/coverage failures, and provenance drift fail loudly.
Each new round selects only current audit failures and duplicate-control gate
failures. Previously passing records keep their original opaque IDs and results.
Authors retain their original assignment and the other agent cross-audits them.
Once an item's revision passes both audit and control uniqueness, it stays
accepted. A later repair that reuses its control key is excluded and retried;
the accepted item is preserved. Conflicts first appearing together in the same
round exclude all conflicting new items until repaired.

For the next round after a completed round 1:

```bash
export UV_PROJECT_ENVIRONMENT=/home/thomasjiralerspong/explore-persona-space/.venv
export UV_OFFLINE=1 UV_NO_SYNC=1
uv run python scripts/issue952_codex_judges.py --phase bank-retry-prepare --round 2 --out-dir /tmp/issue952_def_reuse --packet-root /tmp/issue952_codex_packets_v2
# Complete independent authors' round-2 outputs at the manifest-declared paths.
uv run python scripts/issue952_codex_judges.py --phase bank-retry-audit-prepare --round 2 --out-dir /tmp/issue952_def_reuse --packet-root /tmp/issue952_codex_packets_v2
# Complete independent cross-auditors' round-2 outputs at the declared paths.
uv run python scripts/issue952_codex_judges.py --phase bank-finalize --out-dir /tmp/issue952_def_reuse
```

`bank-finalize` validates all completed rounds, reduces each item's latest
author-and-audit result, and reports prepared, latest-result, and passing counts
by round. Its derived bank/report may be regenerated; all round manifests,
packets, outputs, and persisted agent artifacts are preserved. If coverage
still fails, repeat with round 3, then successive integers as needed. The
unchanged gate requires at least 81/90 items overall and, in every topic,
at least max(2, ceil(0.8 * topic size)) passing items. Audit scores still require
at least 80 on every score and every Boolean check must pass. Do not proceed to
`input-upload` or GPU work until the bank gate passes.

After the approved Qwen generation/capture and verified uploads complete, the
Codex production sequence is:

```bash
uv run --offline --no-sync python scripts/issue952_codex_judges.py --phase production-stage --out-dir /tmp/issue952_def_reuse
uv run --offline --no-sync python scripts/issue952_codex_judges.py --phase production-pilot-prepare --out-dir /tmp/issue952_def_reuse --packet-root /tmp/issue952_codex_packets_v2 --rollouts /tmp/issue952_def_reuse/raw_completions/rollouts.jsonl
```

Independently judge the declared opaque pilot packets, then run
`production-pilot-collect` with the same out-dir. A passed current pilot licenses
`production-wave-prepare` with the same out-dir, packet-root, and rollouts;
independently judge those packets, then run `production-wave-collect`. Preserve
the existing judge-upload verification and analysis pilot/production gates.
Read only aggregate status and hash metadata in the coordinating task.

This round executes synthetic-fixture tests only: no real bank authoring,
packet judging, Qwen generation, GPU provisioning, or network stages.
