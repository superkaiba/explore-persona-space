---
name: extract-sibling-metadata-from-hf
description: When a payload-swap / replication needs bit-identical metadata from a sibling experiment whose source-of-truth code/data is gone (terminated pod, never-committed worktree), extract it from the sibling's PUBLISHED HF artifacts instead of re-running the generator.
metadata:
  type: feedback
---

When a single-variable replication needs bit-identical metadata from a sibling experiment (bystander assignment, eval Q pool, seed mapping), don't re-run the sibling's deterministic generator — it often imports a constant that lived only on the sibling's now-terminated pod or worktree. Extract the metadata from the sibling's PUBLISHED HF artifacts (training pools, eval pools, analyze summaries): it is bit-exact by construction (it IS what the sibling trained on), and a SHA-256 fingerprint over the extracted content gives a re-run determinism cross-check.

**Why:** task #480 needed #411's per-source 2-bystander assignment; the SHA-seeded sampler drew from #275's 111-persona dict importable only on the terminated #275 pod. Hand-reconstruction was rejected (approximately-identical breaks the single-variable contract); extraction from #411's published `train_pool.jsonl` on the HF data repo was accepted.

**How to apply:** prefer extraction from any uploaded sibling artifact; make the extractor fail loud on shape drift (assert row counts, distinct values; hash the result). Also applies to frozen baseline values (#470's `predictor_comparison.json`) and per-source statistics (#411's `analyze_summary.json`) — snapshot, don't recompute. Generator-side discipline: include enough metadata in uploaded artifacts (full system prompts, seed in path, asserted row counts) that downstream extractors never need the generator's runtime environment.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Extract sibling metadata from HF, not generator code](feedback_extract_sibling_metadata_from_hf.md) — bit-identical metadata comes from the sibling's published HF artifacts + SHA-256 fingerprint. #480.
