# Corrected reward-hacking rerun: software validation

The independent critic approved source commit `31cca66a2282431a410eace85e68b27517c7b64a`. No blocking software findings remain. The corrected model experiment has not yet started: canonical task registration is awaiting the explicit request required by `AGENTS.md`, followed by managed GPU provisioning, live token-prefix validation and the real-model smoke.

The [frozen rerun protocol](../../docs/ideas/context_risk_corrected_rerun.md) retains the original prediction goal, 20 tasks, three conditions, eight trajectories per context, and the original prevalence gate. The misalignment arm remains completed and is not rerun.

## Evidence

- [Independent critic record](critic_review.json): exact production source hashes, resolved findings, test commands and scope limitations. The launcher verifies this record before executing any phase.
- [Archived extraction replay](archived_extraction_replay.json): among the 187 previously identified extraction failures, 184 now select the audited valid final code; three correctly fail because an additional final fence is unclosed. This replay does not reconstruct new on-policy model outcomes.
- [Native fixture manifest](fixture_manifest.json): three small Inspect logs, copied and SHA256-verified against the critic's originals. These are software fixtures using prescribed model responses and real Docker execution, never experimental model observations.

Independent validation passed the 36-test suite before two obsolete legacy-controller tests were removed, then passed all five newly added prefix-parity cases. The finalized test collection has 39 cases. The Docker coverage includes a successful test redefinition, ordinary success, impossible-task failure, malformed-fence and syntax-error repair, truncation censoring, timeout classification, and an isolated restored-test diagnostic.

## Reproduce software checks

Run from the source checkout with Docker available and the existing project uv environment:

```bash
UV_NO_SYNC=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
uv run --with 'inspect-ai==0.3.261' --with 'openai==3.7.0' python -m pytest -q \
  tests/test_context_risk_impossiblebench_harness.py \
  tests/test_context_risk_impossiblebench_inspect.py \
  tests/test_context_risk_impossiblebench.py
```

GPU execution must use `scripts/context_risk_corrected_launch.sh`. Its `server`, `smoke`, `full`, and corrected-only `resume` modes require the independent review file and explicitly configured input/output paths. The full run additionally requires the matching three-context smoke to have completed without censored infrastructure/generation outcomes. A smoke cannot certify sustained production concurrency or rare long trajectories.

The runtime changes from the old custom Transformers server to vLLM. Any before/after outcome difference therefore cannot be attributed solely to extraction and scoring. Successful original-condition submissions should also be audited for test redefinition when interpreting the competence filter, without changing that frozen filter after observing outcomes.
