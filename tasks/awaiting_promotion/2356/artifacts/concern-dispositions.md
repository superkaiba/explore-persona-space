# Open-concern dispositions for the #2356 clean-result body (Lens 14 acknowledgment)

**Purpose.** 13 open CONCERN-severity concerns remain in `concerns.jsonl`
(0 BLOCKER). None changes a reported number. In autonomous mode we never
`defer-concern` (user-only) and never fabricate a deferral marker. Lens 14 is
satisfied by **body acknowledgment (mechanism 1)**: each open BLOCKER/CONCERN
`concern_id` must appear as a substring in a `## Takeaways` bullet or a
`## Results` `### <result>` body. Fold the acknowledgments in as a compact
"scope & robustness caveats" cluster; keep the wording plain-academic and
within the check-20 conciseness caps. The concerns stay OPEN in the ledger
(acknowledgment ≠ closing) — they remain visible as next-touch follow-ups.

Grounded facts (verified against the committed artifacts this turn):
- armB labels are keyed by `prompt_sha` → **2510 unique groups**; the staged
  corpus carried exactly 1 duplicate `prompt_sha`, so at most 1 of 2510
  over-refusal groups can carry a judge last-wins collapse — negligible for the
  group-level AUROC.
- `predictor_scores_arm{A,B}.json` recorded **0 dropped / refused / truncated**
  rows this run, so every transport/refusal-ceiling concern went unexercised.

## The 13 CONCERN-severity items (each disposition ≤1 clause; group in body)

Result-relevant caveats (a reader of the Results should know these):
- `armb-duplicate-prompt-sha-disposition` — 1 duplicate armB `prompt_sha`;
  labels keyed by sha (2510 unique groups), so ≤1/2510 over-refusal groups
  could carry a last-wins collapse; negligible for the AUROC.
- `residual-exclusion-drop-class-misreported` — the pre-reissue drop-class
  tally misreports residual exclusions; reporting-only — the mask/Δ_int
  estimator uses residual `None`s (unbiased) and the residual TOTAL + IDs are
  correct (reconciler-confirmed).
- `predictor-transport-zero-valid-not-reissued` — transport-only zero-valid
  rows are not re-issued; 0 occurred this run → power-only, missing-at-random,
  non-blocking (reconciler).
- `predictor-pilot-api-waiver-unbounded` — the both-arm pilot waiver has no
  catastrophic-refusal ceiling; the realized wave had 0 refusals, so unexercised.

Code-robustness / test-pin gaps, not exercised by this completed run (owed on
next touch):
- `pca-cache-bare-existence-resume` — PCA-basis cache keyed on bare existence;
  a generic-corpus change could reuse a stale basis (corpus fixed for this run).
- `extras-refit-lambda-unpinned` — extra-row refits use each fold's persisted
  GCV λ but do not assert equality of the returned λ (registered assignment;
  unbiased here).
- `secret-scrub-gate-context-divergence` — scrub decides on raw-text bytes, the
  upload gate scans JSON-file bytes; hotfixed same-length at the write boundary,
  residual escape-inflation channel only.
- `round4-invariant-pytest-gaps` — R4-2/R4-3/R4-4 invariants lack dedicated
  committed pytest pins (verified by the run's own assertions).
- `round6-hotfix-regression-pins` — arm-A source + shared-cleanup v6 hotfixes
  lack direct regression pins (v6 sweep pinned only the scrub commit).
- `cleanup-vllm-reap-failure-swallowed` — a vLLM reap-failure debug-swallow is
  real but fail-loud downstream; raise-log-to-warning + docstring caveat owed on
  next touch (reconciler downgrade).
- `arma-source-consumer-post-init` — post-init source access pre-exists
  unchanged; stale-corpus state unreachable on canonical relaunch; add
  pre-engine validation on next touch (reconciler downgrade).
- `rejudge-legacy-save-raw-silent-skip` — the post-fix rejudge path silently
  skips pre-fix full-ID refusal records; this run wrote no such artifact.
- `capture-detach-full-seq-transfer-unvectorized` — capture extraction is
  transfer-bound (17.5h realized vs 1.3h floor); a reduce-on-GPU fix is owed
  before any capture re-run (no re-run performed).

## NITs (do NOT block Lens 14; fold only if it reads naturally)
`issue2356-upload-verify-caller-pin`, `smoke-marker-dry-run-coverage-overclaim`,
`custom-id-suffix-hand-mirror`, `predictor-residual-transport-disclosure`,
`predictor-pilot-fixture-completeness-mismatch`.
