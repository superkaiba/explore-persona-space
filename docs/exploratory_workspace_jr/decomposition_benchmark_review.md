# Independent optimization review

The Codex reviewer checked the nested sparse executor and its reference tests.
The step order, active-atom reselection, coefficients, per-k copies/counters,
and individual-token-before-equal-rollout aggregation match the immutable
reference on the tested CPU paths. Review findings about mixed compute dtypes
and replacement of a validated tensor were fixed: incompatible dtypes are
rejected; the public tensor reference is read-only; identity, storage pointer
and PyTorch version are checked. Tensor.data mutation is explicitly forbidden.
The phase must still verify dictionary file hashes. Three tests passed.

This does not authorize adoption based only on toy CPU tests. Benchmark the
actual full-vocabulary native J/R dictionaries and saved ragged activations on
the intended GPU, compare all output fields at5/10/25, and record synchronized
timing, memory, precision settings and input hashes before switching consumers.
Native lens/runtime files remain unchanged.

The reviewer also checked the VJP dim-batch benchmark. The child invokes the
unchanged native CLI with the same argv0 identity, prompt and validation; only
dim_batch may differ in its contract. It compares J to J and R to R with the
predeclared1e-5 relative Frobenius gate. Review found sampler exceptions could
leave a child running. The fixed wrapper registers each candidate before
launch, checks an idle single GPU, and always reaps and records the child on
failure. A regression verifies failed-sampler cleanup and a zero-sample receipt.
Hardware identity and the benchmark script hash are recorded. Four focused
tests (benchmark plus sparse executor) pass, and the reviewer confirmed closure.

Snapshot finalization permits a garbage-collected transient service only with
matching cloud rank ownership and a valid fresh terminal receipt. Successful
completion is separate from terminal-receipt inclusion, so failed worker
outputs are still persisted. The reviewer accepted this for the current
write-once, no-automatic-restart workers; a future restarted attempt needs an
explicit new attempt identity.
