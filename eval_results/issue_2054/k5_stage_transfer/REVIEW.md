# Independent implementation review

Reviewed by the independent Codex `stage_transfer_review` agent, 2026-09-17.
No automated Claude/Anthropic tooling was used.

The review found no remaining numerical or scientific blocker. Frozen transfer
uses A = diag(1/source_sd) W and b = source_mean_y - source_mean_x A without
target adaptation. The separate bias correction uses only target training
means. Global fold IDs exclude every target test conversation from source
training, even where the complete-five cohorts differ. Metrics use float64
before float32 prediction persistence. Equal-fold R2 means, ratio-of-means
retention, full target retrieval pools and tolerance midranks match the paper's
speaker appendix.

Two review comments were resolved before launch: the publication inventory now
includes `run_identity.json`, and the plan accurately names top-1 retrieval as
the endpoint parity check (top-5/top-10 are computed, with canonical metric
equivalence checked in the numerical tests).

The independent review ran 10 focused tests. The implementer separately ran 20
driver/helper regression tests. Production acceptance additionally requires all
30 units and 60 own-map endpoint parity checks, complete upload and independent
remote hash verification; see `verification.json` for the final observed result.

Interpretation: the global folds match, but the four character settings use
different complete-five conversation cohorts. Chat and Assistant-story target
cohorts share 7,999 conversation IDs. Base/Instruct shared IDs within settings
are Chat 7,999, Assistant-story 7,993, HELIOS 7,994, Wren 7,991, Dana 7,994 and
Vex 7,997. Preserve the paper's original cohorts and describe setting-level
predictive portability. This analysis neither isolates a causal narrative
framing intervention nor separates SFT from the rest of Qwen post-training.

Monitoring note: the first watchdog check preceded the monitor's first startup
observation and triggered a false startup alert. The bounded recovery worker
verified fresh progress and returned without changes or restart. The next
scheduled watchdog tick confirmed healthy execution. The false alert remains
recorded in the durable watchdog state; no evidence was rewritten.

## Independent production audit

The same independent reviewer verified all 30 unique setting/fold packets and
six summaries after computation. The audit hashed all 1,290,354,950 NPZ packet
bytes and matched each metadata packet to its result row. Recomputing all 150
method/fold R2 values from persisted float64 SSE/SST gave maximum absolute
difference 1.11e-15. All 60 own-map R2 endpoints matched references with maximum
difference 1.33e-15; all 360 endpoint retrieval values (both distances at 1, 5
and 10) matched exactly. Rebuilding folds from actual bank IDs verified zero
cross-stage leakage, exact target ordering, and paired masks. Equal-fold means,
retention ratios, reported ranges and the interpretation of controls passed.
Remote publication was independently verified by the process monitor as a
separate completion condition.
