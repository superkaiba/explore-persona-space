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
