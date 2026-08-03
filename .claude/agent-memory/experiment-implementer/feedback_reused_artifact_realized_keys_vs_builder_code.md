---
name: Reused artifact's realized keys vs builder code
description: A sha-pinned reused artifact's key set can predate the current builder's save dict — verify the artifact's OWN keys against the consumer's asserts, not the builder code (#1073 p0 crash)
type: feedback
---

A sha-pinned reused artifact's REALIZED key set can predate the current builder
code's save dict. #1073: today's `run_pass_b` writes `prompts` into the pass_b
bundle, but the pinned 2026-07-01 upload lacks it — the consumer's hard assert
killed P0 on the pod (att-20260706-071820, one wasted GCE provision).

**Rule:** before a consumer asserts fields on a reused artifact, verify the
artifact's OWN keys (mmap-load key check, or run the consumer's loader against
the real artifact) at plan/smoke time — reading the builder code is NOT
verification (the fact-checker and the smoke fixture both mirrored the code,
so both missed it).

**How to apply:** when a missing field is deterministically regenerable,
regenerate via the parent loader with fail-loud source/length asserts PLUS a
deterministic re-capture alignment gate against the artifact's stored tensors
(row ALIGNMENT is the invariant, not the field's presence). Make the smoke
fixture default to the PRODUCTION artifact shape.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Reused artifact's realized keys vs builder code](feedback_reused_artifact_realized_keys_vs_builder_code.md) — verify a reused artifact's OWN keys, not its builder code; regenerate + alignment-gate a missing regenerable field (#1073)
