# Kimi recovery and controlled numerical smoke

The live source `cd70b81267bb626ac3c621eafe964ae0ec5b909c` passed startup and callback transfer, then failed numerical repeatability at 0.1432724893 against 1e-5. Independent audit confirmed 9760 zero hook-reference errors and matching TP checksums within each capture. The initial pass enabled norm diagnostics while replay disabled them, confounding instrumentation and execution order. This is a test-design defect; it is not proof of the runtime's underlying numerical cause.

Production coverage: **0/2160 contexts and 0/270 chunks**. The two saved 20-context diagnostic arrays are not valid completed experiment results. No Kimi leakage correlation can be reported from this attempt. The research scope, raw cosine, uncentered whitening, persona bank, published outcomes, model precision, TP8 topology, and numerical tolerances remain unchanged.

All 13 output files (42,437,061 bytes) have verified immutable HF copies. The final audit bundle includes pinned source excerpts and final logs. Canonical upload-gated teardown succeeded, and the personal-account API confirmed absence of pod `2y9tr3io3gd0os` at 2026-09-26T00:52:16Z. Original paid start and deadline are retained. Conservative consumed GPU-hours: 26.656906. Kimi-only monitor/retry/relay timers were deliberately retired for this closed failed attempt; there is no active model job. Other experiment monitors were untouched.

## Implemented control repair

Before the full-bank smoke, evaluate the same `default:1` context with norm-check flags false,false,false,true,true,false. Separately report adjacent production repeats, return to production instrumentation, checked repeats, and each instrumentation transition. All comparisons retain the 1e-5 ceiling and block production on failure. Save diagnostic vectors, flags, pair indices, per-layer errors, and rank/norm evidence before rejecting a numerical comparison.

The full existing 18–20-context coverage then uses false initial and false reversed replay, plus a separate true pass for all independent norm checks and instrumentation-invariance comparisons. Preserve each completed phase before the next begins. Retain the original `smoke.json` and `smoke_vectors.pt` consumer fields, adding explicit phase instrumentation and evidence. A process or GPU exception partway through a phase can still lose that incomplete phase's vectors; completed phases are durable. A new engine must repeat all smoke phases.

Validation: 55 focused tests passed across new controlled-smoke fault tests, existing Kimi tests, capture tests, and artifact tests; ruff check/format passed. Independent reviewer additionally ran 17 tests, including downstream analysis compatibility. Mock GPU boundaries verify control flow and fail-closed gates, not real Kimi numerical stability. Real GPU validation is outstanding.

## Next live diagnostic

Use the same model revision, native INT4/BF16 residuals, TP8 H200 topology, and singleton execution. Run the controlled diagnostic first, preserve its evidence, and stop if any gate fails. If it passes, run the complete constant-instrumentation smoke; only then permit the unchanged measured throughput and remaining-time gates to admit production. Do not enable batch-invariant kernels, change precision, or relax tolerances without separately verifying compatibility and reviewing the change.

A concrete possible recovery envelope is one additional 8-H200 allocation capped at two hours from provider creation (at most16 additional GPU-hours), retaining the15-minute preservation reserve. That is a bound, not a runtime prediction. Including this failed attempt gives at most42.656906 GPU-hours. The current single-allocation ledger remains unchanged apart from verified termination and consumed time; no replacement pod or extension has been provisioned.
