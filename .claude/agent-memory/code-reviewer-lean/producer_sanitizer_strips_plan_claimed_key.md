---
name: producer-sanitizer-strips-plan-claimed-key
description: A plan's realized-keys row sourced from producer CODE can claim keys a producer-side sanitizer strips before upload; validate against an observed-artifact probe, and a pinned-replay substitute needs a source-key fail-loud guard
metadata:
  type: feedback
---

When a plan §10 / artifact-reuse row asserts an artifact's schema "read from
producer code L<x>-<y>", check whether the producer has a SANITIZER or
upload-filter between the in-memory dict and the upload (content-hygiene
strips of raw text are the canonical case — `_sanitize_for_analysis_tensors`
+ `_assert_no_raw_text_under` in issue779_collect.py). The in-code dict is
NOT the uploaded schema (#2061 shape; #2254 r1 g7: plan claimed a `prompts`
key + fp16 the realized bundle could not carry by construction — float32,
no prompts).

**Why:** schema-from-producer-code passed plan review and only the tiny-real
smoke running the consumer's own loader against the pinned artifact caught
it; the correct fix conforms the loader to the OBSERVED schema (pasted probe
output in the marker), never a permissive `.get()` default.

**How to apply:** (1) A loader "fixed" to match a realized artifact must
keep every key REQUIRED and fail-loud (regression tests pinning both the
accept-realized-schema branch and the missing-key raise — certify
fails-pre-fix via the parent hunk, cf. [[fails-pre-fix-probe-parent-commit]]).
(2) When stripped content is RECONSTRUCTED via a pinned replay of the
producer's ingest recipe, require: revision pin on the replay source, a
fail-loud assert on the artifact's own `source`-style key (guards the
producer's fallback-source branches — issue779_collect falls back
LMSYS→WildChat→UltraChat), a count==N check, and a recipe-match read of the
producer's selection loop (same filter, same first-N). Order-insensitive
consumers (set-disjointness gates) only need set identity; cite the prior
issue that verified 1:1 equivalence.
(3) Monkeypatch-validity check for such loader tests: a call-time
`from huggingface_hub import hf_hub_download` INSIDE the function makes a
module-attr monkeypatch effective; a module-top `from`-import would not.
