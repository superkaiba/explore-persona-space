# Issue 952 China repair: independent prelaunch review

Verdict: one P2 requiring repair before launch; no other blocking findings in the reviewed scope.

Reviewed current bank, judge, launcher and design files at worktree HEAD `8c29285e0f83e7578c0546980377a0a838e22e8e`, plus GPU change `e9a518fa5cdd3c58bdab93f4bb0e13e97a6b4734` and all three corresponding repair test files. Sized the GPU commit diff before reading: 48,996 bytes. Read-only review of implementation and synthetic tests; no real bank rows, response rows, individual judgment outputs, model calls, or external APIs were inspected or invoked.

## P2: consume the selective cap-extension requirement before capture and completion

Locations: `scripts/issue952_china_definitive_gpu.py:502`, `scripts/issue952_china_definitive_gpu.py:819`, `scripts/issue952_china_definitive_gpu.py:1516`, and `scripts/issue952_china_repair_run.sh:45`.

The repaired generator correctly identifies truncated rows in cells whose cap-hit fraction exceeds 2%, but only records `cap_extension_required_item_ids`. The launcher proceeds directly from generation to raw upload, capture and finalization, and none of those phases consumes that requirement. For example, 14 truncated responses in a production cell of 680 require extension, yet this path can publish `status=done` with the original truncated responses. That violates the selective doubled-cap repair in the plan (`docs/experiments/issue952_china_repair_v2.md:39`) and can conflate truncation with withholding.

Add the planned subset-only extension before capture, retain the original generations and their hashes, bind the merged result to the extension manifest, and reject finalization with unresolved required extensions. Add a phase-level regression using a cell over the threshold, alongside a below-threshold control. The owner acknowledged this gap and is assigning its repair; the replacement implementation is outside this review's tested snapshot.

## Verification

- Focused bank, judge, and repaired GPU suites: **58 passed in 21.91 seconds**.
- `bash -n scripts/issue952_china_repair_run.sh`: passed.
- The bank enforces exact cue addition and base-question hashes; the GPU consumer checks the full 85-source, 16-cell lattice and all 680 tokenized cue pairs before generation.
- The judge collector requires complete explicit ordered decisions, rejects invalid numeric/boolean schemas, preserves unassessable values as missing, and binds source, packet, rubric, runtime, receipt and output lineage. The identity correction leaves unavailable runtime settings null and maps old decisions only after exact same-input checks by original author identity.
- Smoke and production have disjoint seeds and outputs, with shared generation/capture settings compared before production. Tensor and rollout payload verification is revision-scoped and byte-based.

## Residual limits and handoff

- Actual model/tokenizer execution and remote persistence were not exercised by this review; the focused GPU tests use synthetic boundary fixtures. The real smoke remains necessary.
- No semantic assessment of actual country metadata or judgments was performed; those are separate independent audits.
- Model identity settings and input-reading attestations have the limits documented by the implementation; passing schema checks does not establish human validation or attention.
- The launcher is mode 0644. Invoke it through `bash`, or set executable mode if the dispatch command will call the file directly. No direct-execution failure was assumed.
- No additional P1/P2 findings were identified in the scoped bank/judge design, factorial coverage, lineage, smoke compatibility, or upload code.
