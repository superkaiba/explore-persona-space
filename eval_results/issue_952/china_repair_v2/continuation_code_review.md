# Repaired China continuation review — 2026-09-08

Independent reviewer: `/root/china_repair_review`. Base code:
`d41003584a26fa09ad8a9d3e9099b21dc2a77897`; the new files below were reviewed
before their subsequent commit. No production responses or outcomes were read.

## Dedicated CPU launcher

`scripts/issue952_china_repair_cpu_run.sh`, SHA256
`d3dbf38a20370f6e89dd1f02a04fbbd9aab0dc12a88eec8989bcc9d9da21b3ee`.
No blocking P1/P2 findings; `bash -n` passed. The launcher requires exact clean
code and immutable data/judge revisions, propagates phase failures, limits CPU
threads to eight, and checks a real CPU lane without hiding CUDA devices.

Operational conditions: invoke with `bash`; bootstrap the existing synchronized
uv environment and `/usr/bin/time`; preserve the outer launcher transcript as
well as per-phase logs. A PID file alone is not a success signal. The owner must
verify the analysis/export sentinels and persist logs and the export receipt
before teardown.

## Explicit-decision serialization

`scripts/issue952_china_repair_submit.py`, SHA256
`3679dfd1188625fa0cccc4641e1ccb93d76130491e9bcd876de85dc4cb837352`.
`tests/test_issue952_china_repair_submit.py`, SHA256
`4188109de3c7229efaa7280870108c70135bd36be0335f101a7e59c91b651be4`.

No P1/P2 findings. The reviewer independently passed all eight new synthetic
tests plus the existing immutable-write replacement-rejection test. The owner
also passed the eight new tests and focused Ruff checks. Every substantive
field and ordered opaque ID must already be explicitly authored; the helper
adds byte lineage only, never scores or default labels. It validates all rows
before publishing immutable sibling receipt/output artifacts.

Reading attestations remain reviewer assertions, not software-verifiable proof
of attention or independent reasoning. The full collector additionally checks
source-manifest and packet assignments. Synthetic fixtures are parser tests,
not production judgments or human validation.

## Plot-only producer

`scripts/issue952_china_repair_figures.py`, SHA256
`ebfadbed3914a828686104ad5bc6853c530465a9802f28d80feaab768013f35d`;
test SHA256 `e3b5aed8b72d84c31d12e9337a3cb54316f7eccc30ef7e6b00132a0ab4ae0d33`.
No P1/P2 findings; 14 independent synthetic tests passed in 6.31 seconds and
scoped Ruff passed. The reviewer checked the real report schema, exact
complemented confidence bounds, missing-value gaps, defined denominators,
all three layers and both languages, canonical styling, and hash-bound
provenance. No statistics are recomputed. Only synthetic figures were rendered.

## Offline GPU completion verification

`scripts/issue952_china_repair_verify_gpu.py`, SHA256
`90812663090535bc36700424c034cdabf4bb1d249fd979fe58ad323d89ecfa60`;
test SHA256 `21f8f472ca13768d22d2fe9a058752314359dd86b80cbc9b7d2912c47ff9c158`.
No blocking P1/P2 findings; 17 independent offline tests passed in 20.45 seconds
and scoped Ruff passed. The reviewer traced actual producer schemas, smoke
subset/order, raw metadata/seeds, actual tensor/index counts, all end-of-turn
boundary cases, finite fp32 tensors, pinned local tokenizer files, fresh
sentinel lineage, restricted cache exclusions, and exact text-archive census.
The full synthetic CLI exercise covered both declared cardinalities.

This is offline completion verification only. Fresh remote persistence proof,
process exit/quiescence, and scoped teardown remain separate owner duties.

Owner combined verification: all 39 new continuation tests passed in 28.44
seconds, focused Ruff passed, shell syntax passed, and `git diff --check`
passed before commit.
