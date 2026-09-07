# Capture code review

**PASS — software review and CPU validation. GPU capture completion is still required.**

Reviewed the exact source closure listed in `capture_code_review.json` for the fixed Qwen3.8-27B revision, decoder block44, and final unpadded initial-prefix token. The same reviewed helpers implement the map-side context convention. No fresh outcome labels were opened, no model calls were made, and no GPU weights were loaded.

The review found and the implementer fixed four issues: actual editable-package imports could differ from hashed worktree files; an initial launch could adopt unbound old chunks; the pod Torch package metadata version differs from its module version; and downstream chunk validation needed semantic checks as well as file hashes. The revised code checks actual imported helper bytes before and after capture, refuses orphan artifacts, distinguishes Torch distribution/module/CUDA versions, and validates complete ordered chunk contents and saved generation-token identities.

Executed evidence:25 existing helper/extraction tests passed;21 independent capture fixtures passed, including full249-context/17-chunk roundtrip, identical resume, stale binding refusal, source and position tampering, and semantic payload corruption despite recomputed hashes. A separate CPU model with64 decoder blocks verified block44 indexing and final unpadded positions for unequal-length rows. Hydra configuration composition and Python compilation passed. The independent fixtures are persisted under `software_fixtures/capture_review.py`.

Both frozen fresh manifests contain249 distinct contexts from83 tasks across3 conditions. Saved live prefix token arrays and hashes agree; maxima are19578 tokens forA and19574 forB, below the32768 hard capture limit.

Read-only inspection of the actual generation-server environment found Transformers5.15.0, Torch distribution2.13.0 / module2.13.0+cu130, CUDA13.0, NumPy2.3.5 and vLLM0.28.0. Accelerate is absent. The capture launcher must add the explicitly pinned capture dependencies and preserve runtime pins. Before full capture, verify GPU hook/replay parity and production-shape memory, including the longest prefix and a two-row unequal-length batch. The16384 padded batching budget permits longer singleton rows; it never truncates them.

This PASS approves the reviewed software for its gated runtime validation. It does not claim that the GPU capture has already completed or that the scientific hypothesis is supported.

## Workflow-lint cleanup refresh

**PASS** after review of changes since `f5c862dda8a`. The refreshed JSON receipt pins the current exact source hashes. Model and capture constants are AST-identical. The change consists of file iteration for four wrapper JSONL reads, standard direct-script import guards in two inherited helpers, and runtime-dictionary formatting.

Reran33 existing capture/helper fixtures successfully and added one Unicode regression fixture that runs the full wrapper and downstream binding validation with raw U+2028, U+2029 and U+0085 inside249 manifest records and17 chunk metadata files. Both inherited scripts also resolved their Hydra configurations when invoked directly from `/tmp`, without model or GPU work.

One non-blocking inherited limitation remains: the lower-level ImpossibleBench manifest loader still uses `splitlines()` and rejects synthetic raw Unicode separators. The two immutable fresh manifests contain none and both load249 contexts through that helper, so this does not affect this frozen experiment. The Unicode fixture establishes the wrapper's behavior, not general support for arbitrary manifests in the inherited loader.
