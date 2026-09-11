# Independent implementation review

Verdict: **PASS** for the reviewed preparation, GPU capture, and fit/analysis implementation. No unresolved code blocker identified for the authorized existing-19k, training-K=1–10 extension. This does not replace the real-GPU parity, first-production-chunk, canonical launch, or final artifact-verification gates.

## Scope and evidence

Reviewed all three `scripts/issue1901_training_k10_{prepare,gpu,fit}.py` entrypoints, their focused tests, the amended protocol, and the inherited generation/capture, whitening, retrieval, background-upload, and completion helpers. The review was independent of implementation; no Claude automation or experiment generation/evaluation was invoked by the reviewer.

Independent final verification on 2026-09-11:

```text
uv run pytest tests/test_issue1901_training_k10_prepare.py tests/test_issue1901_training_k10_fit.py tests/test_issue1901_training_k10_gpu.py -q
26 passed in 2.41s
```

The GPU implementer separately reports Ruff and the import/API-signature checks passing. The numerical tests compare full primal GCV curves, intercept degrees of freedom, and fitted coefficients to an independent direct hat-matrix solve, including dependent and constant input columns. Cluster-bootstrap R² is checked against explicit variable-length resamples. Other tests cover CI joins, precision preservation, chunk boundaries, stale recipes, generation hashes, remote content verification, large-text reassembly, completeness, stale completion state, and private process-group cleanup.

## Corrections verified during review

- The fit loader now validates prepared files against the input manifest before reading arrays, and checks model/revision/layer compatibility with the capture manifest. The earlier version only hashed current inputs for provenance and could accept changed bytes under an unchanged manifest.
- Large JSON uses indented serialization compatible with the line-sharding uploader. The earlier compact one-line serialization would fail the sharder for oversized chunks. The reassembly manifest follows the existing Hub consumer schema and the test verifies exact reconstructed bytes. GPU input JSON uses the existing manifest-first `hub.stage_sharded_text` consumer, followed by exact reconstructed-file size and SHA checks; the final staging delta was reviewed against the helper's actual signature and implementation.
- GPU completion clears only the owned stale attempt sentinel, verifies the final manifest and every declared capture hash against the uploaded receipt, and marks fitting as pending. Child phase cleanup targets the private process group; failure tails and per-chunk progress reach the controller log. Source-code hashes, input-manifest identity, ordered IDs, world size, sampling settings, and runtime metadata protect reuse.

## Scientific and data checks

The preparation path selects the declared 19,000 IDs explicitly, preserves original float32 context/answer precision, joins stored draws by CI, and validates original answer equality plus prompt hashes against the pinned source. Test rows are the original fixed split; exact train/test prompt overlap fails. Inherited near-duplicate exclusion is disclosed as inherited rather than newly recomputed.

The fit keeps training inputs fixed, averages original then seeds43–51 in the correct order, uses one shared input factorization, and chooses regularization entirely from training labels. GCV counts the fitted intercept. The learned-bias identity baseline uses the matching K-specific training target mean. Evaluation fixes the original 942 candidate representatives, fits whitening only to original training answers, freezes that transform, and uses identical prompt-cluster bootstrap draws across K and both arms. The primary K10-minus-K1 comparison at evaluation K10 is computed as a paired difference. No test-target selection or training-row retrieval candidates were found.

The production pass reuses and validates the first 500-row pilot generation/capture; it does not generate that pilot twice. Uneven final chunks preserve complete context/seed coverage. Final reconciliation checks the full 95,000-element Cartesian identity set. Background uploads remain bounded and are joined before phase completion; the final upload verifies raw outputs, captures, receipts, and logs before success.

The new draw path reuses the exact inherited narrow handler for model-generated JWT-shaped strings that match no live credential. It masks both response/text aliases identically, preserves original token IDs and the original text hash, persists explicit disclosures, and captures the disclosed edited text, matching the old bank policy. Other findings fail. The amended protocol discloses this transformation and requires reporting realized affected-row counts when interpreting results; this is disclosure of implemented source compatibility, not a new approval requirement.

## Review boundary and remaining launch checks

These CPU tests do not certify CUDA/vLLM initialization, real hidden-state equivalence, Hub authentication/transport, GPU throughput, or the longest production prompts. Every GPU lane must pass stored-vector parity and the real 500-row generation/capture pilot through the production path before continuing. The owner still verifies live preflight, committed code/input staging, monitoring, complete remote hashes and row coverage, and final fitting-output persistence before teardown. No model or assertion substitution was added for the pilot.

Reviewed file SHA256 values:

```text
8fee229ca6e7d13a65056b73a7be1551280f06a1c4af61e45228059675ede4ff  scripts/issue1901_training_k10_prepare.py
285fb6625661113bdea4d00684d2cc4fea2ba9ddb3e40d2c116b30e3273a5994  scripts/issue1901_training_k10_gpu.py
c2b0a6d61c67ffb2e0635e475d3a3dcdc57bda9d42150ea33926e1adbb964068  scripts/issue1901_training_k10_fit.py
26ee2c5bc9f0ef5e36e27443fb0eac683104c961152b65b4ff1815f9394269b7  tests/test_issue1901_training_k10_prepare.py
656564a9150522a7bfaa2f95e5c768bba798a771086a5988a470afae7ad78cb6  tests/test_issue1901_training_k10_gpu.py
f1b1cf010f8fc9468db9b114007e0e95cab7dd4691ff0f5bd99d8ca284de7305  tests/test_issue1901_training_k10_fit.py
```

## CPU continuation and reporting addendum

The post-capture report helper passes independent bounded review at SHA256 `7223f6e6864ca416fea1a9d782206354d48d291190f167a64887be42391b159d`. Its metric ordering, training/evaluation axes, and paired confidence-interval contrasts match the reviewed fit producer. It stages an immutable capture manifest and exact tensor hashes, reconciles the 95,000 context/seed pairs, checks raw-generation hashes against capture metadata and expected prompt identities, and verifies every reported cell and contrast against saved arrays. Publishing requires the raw audit, verifies the full analysis/weight/output upload, and builds browser URLs from immutable verified figure revisions. The final publication receipt records both artifact and report upload revisions. No data staging, fit, plot, or publication was executed by this reviewer.

The watcher review identified an actual completion-reader bug: the GPU result embeds a large upload receipt, but canonical task events cap notes at 50,000 characters. The poller archives oversized notes and posts a prose pointer; the original watcher skipped that pointer as invalid JSON and would miss the completed capture. At the owner's request, this reviewer implemented `event_note_payload` in the watcher. It resolves the marker and current task location through the workflow API, reads only the event-declared full-note basename under that task's artifacts directory, and rejects unexpected names, symlink escapes, or changed payload lengths. This evidence read does not directly read or edit workflow state. The owner independently inspected this reviewer-authored helper and returned PASS before launch.

The permanent `tests/test_issue1901_training_k10_watch.py` suite passed: **8 passed in 0.29s**. It covers ordinary/prose and genuinely oversized JSON notes, task-location changes, capture selection and launch-time filtering, malformed pointers, length drift, and symlink escape. Ruff passed for watcher and tests. The watcher also now archives prior completion/failure files after acquiring its single-process lock. Its documentation correctly states that the canonical poller can perform authorized same-workload backend failover. The exact-handle CLI and canonical event API signatures match current source, and the stage/audit/fit/publish command chain matches the report and fit parsers. Final evidence review and gated compute finalization remain with the owner. Reviewed watcher SHA256: `f561766508557421111d4c398c8356039df7d50692b29be0601fb92b85f4c5d3`.
