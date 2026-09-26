# Controlled smoke design review

Design assessment before implementation review: PASS in principle. No GPU run or allocation is implied.

The six repeated captures of one exact input with norm-check flags `[false,false,false,true,true,false]` separate adjacent production repeatability, checked-path repeatability, instrumentation transitions, and return-to-production behavior. Preserve explicit pair labels: production 0->1 and 1->2, production return 2->5, checked 3->4, and cross-instrumentation 2->3 and 4->5. Store the exact probe row/index/prefix identity and flags, the six raw BF16 vectors, canonical per-layer relative errors, and diagnostic pass/fail before raising for a failed comparison. This diagnoses execution behavior; it does not by itself identify a kernel root cause.

Full inherited coverage should compare 20 initial contexts with checks false against the correctly re-aligned reverse-order replay with checks false. Separate checked captures must retain the independent norm reference at every layer/rank and compare checked vectors against production-mode captures. Overall pass must require every registered category at the unchanged tolerance. Keeping an instrumentation-invariance gate prevents turning the repaired diagnostic into a way to skip the prior failed comparison.

Artifact compatibility inspected:

- `story_persona_crossmodel_analysis.py:438-463` requires smoke fingerprint, passed=true, throughput_gate.passed=true, indices, and `smoke_vectors.pt` entries initial/repeated. Those vectors must remain BF16 with shape `(len(indices),61,7168)` and matching fingerprint/indices.
- Keep existing success-schema fields; add category diagnostics and checked vectors rather than repurposing initial/repeated for the six-run control.
- Separate `.json`/`.pt` control artifacts are compatible with recursive upload inventory (`story_persona_qwen38_artifacts.py:67` and crossmodel artifact persistence). Failed attempts use the incomplete inventory path, so an early diagnostic failure can be retained without pretending full coverage was completed.
- Capture already fingerprints `story_persona_kimi_runtime.py`, so the source repair changes provenance and cannot silently resume an old incompatible recipe.
- Preserve progress writes through the longer diagnostic so a future monitor sees actual work. CPU fault tests should demonstrate production-only instability, checked-only instability, instrumentation-only differences, and full-bank order dependence all remain failures with saved evidence.

Implementation review is still required after the parent supplies the ready diff. No source files were edited by this reviewer.

## Concrete implementation review

PASS, with no blocking correctness or downstream-schema defect found. Reviewed SHA256:

- `scripts/story_persona_kimi_runtime.py`: `bca15e82b438f03be99adc4eb267f1ba4bfb667edfe651ea596d688bfcf535d1`.
- `tests/test_story_persona_kimi_smoke.py`: `0eb36dd50b2af0595e1782fee3b72895111ea7afb1249700f023b9120fbab21a`.

Fresh independent CPU run: `pytest tests/test_story_persona_kimi_smoke.py tests/test_story_persona_crossmodel_analysis.py` passed **17 tests** (5.10 seconds). This includes injected production-only, checked-only, cross-instrumentation, return-to-production, full-bank replay, full-bank checked-mode faults, and preservation of a completed phase across a subsequent capture exception.

The real implementation uses all six recommended pair categories, keeps the exact previous relative-error tolerance, rejects nonfinite comparisons, and fails before the full-bank capture if the controlled diagnostic fails. The inherited full-bank initial/repeated arrays now both use check_tuple=false; checked-mode vectors are separate and remain a blocking comparison. Repeated vectors are re-aligned to the original context indices before comparison and storage. Existing initial/repeated BF16 arrays, fingerprint, indices, norm evidence and success fields remain compatible with downstream analysis; extra phase arrays and artifacts are additive. The helper cannot return a failed diagnostic to a successful final smoke because it raises first.

Scope of preservation: numerical-comparison failures save their complete evidence before raising; each completed full-bank phase is checkpointed before the next starts. An unexpected `capture()` exception partway through the initial six-run diagnostic occurs before that diagnostic's final save, so only progress and outer failure diagnostics survive for that partial probe. This is a bounded preservation limitation, not a numerical-gating bypass; saving a partial probe after each successful call would close it if the parent wants an unconditional every-failure preservation claim.

These are CPU control-flow and compatibility checks. The patch is not evidence that native INT4 Kimi has become repeatable, and it does not authorize another paid run. No model execution, GPU allocation, or source mutation was performed by this reviewer.
