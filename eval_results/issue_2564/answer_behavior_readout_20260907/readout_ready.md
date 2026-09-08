> Historical API-route note: Thomas has now selected Codex subagent judging. Follow [the current status](status_report.md) and amended plan; do not resume the failed API route. The numerical readout validation below remains relevant.

# Actual-answer behavior readout: implementation ready

The corrected experiment measures properties expressed in the exact generated
answer span represented by each observed answer vector. Its seven independently
annotated targets are expressed persona/voice, dominant topic, warmth, assertive
confidence, formality, actual language and actual format. It does not substitute
requested conditions, source character names or SAE feature categories for these
labels. Eight mechanical surface controls add length, selected English lexical
rates and punctuation/format counts; these are not comprehensive syntax measures.

**Status: annotation is blocked by invalid OpenAI authentication. No actual
behavior labels or actual-label readout results have been produced.** The paper
has not been edited. The earlier SAE analysis remains separate evidence.

The collector, pilot instrument gate, source archive and authentication blocker
are documented in [status_report.md](status_report.md). Follow
[resume.md](resume.md) to run and review the pilot, record instrument acceptance,
complete the main annotation wave and aggregate its labels. Pilot acceptance is
a scientific measurement check; no new user approval for the already authorized
experiment is required. A valid key must be configured securely, not pasted into
the conversation or committed to Git.

## Matched analysis

The main cohort has 2,048 answers across 512 source questions and 438 connected
question groups. All answers sharing a question or an exact duplicate are kept
in one outer fold. Observed-answer and context readouts use identical targets,
availability masks, five outer folds and three grouped inner folds. Full-width
ridge is primary, with training-only standardization and the existing #2564
penalty grid. All target columns with the same availability mask, penalties and
PCA widths share each decomposition. Training-only PCA256/PCA512 are secondary;
an unattainable width is reported unavailable rather than silently reduced.

Baselines are training mean/prior, training source-framing means and answer
length. Target-label ties are retained as vote fractions for fitting and excluded
from unique-mode classification metrics. Graded unassessability remains missing.
Per-class positive/negative answer and independent-group counts accompany
categorical metrics. Question-component bootstrap draws are shared across arms
for paired comparisons. Confidence intervals condition on defined bootstrap
replicates; their defined-draw counts must be shown, especially for rare classes.
They condition on the fitted OOF predictions and do not include retraining or
independent-annotator uncertainty.

The single restricted label shuffle preserves component bundles, source framing
and availability patterns and repeats the complete tuning procedure. Its
exchangeable, moved and fixed group counts are exposed; it supplies no
permutation p-value. Single-voice-without-substantial-narration, uncapped and
classes-supported-by-at-least-20-groups diagnostics evaluate fixed OOF predictions;
they do not retrain models on the subset. No universal ordering of high- and
low-level properties can be inferred by comparing unlike target metrics.

## Validation

The final serialized collector/readout suite passed **19 tests in 8.80 seconds**.
The live shared ridge implementation agrees with the existing #2564 helper and
independent scikit-learn full/PCA fits to 1e-9 tolerance. Tests cover nested
penalty selection, connected-group splits and shuffles, explicit-resampling
metric parity, full-size loader identity/missingness/tie checks, pilot and
aggregate provenance gates, real fit/summary bodies, checkpoints and stale-input
rejection. Lint/format checks passed for the new readout and tests.

A numerical timing smoke used the real prepared vectors and explicitly random
fixture targets. One full-width outer fold, including all three inner folds,
both activation arms, all declared PCA widths and baselines, completed in
4.14 seconds with finite predictions and approximately 970 MiB peak process RSS.
**This was an implementation smoke, not a behavioral experiment or result.**
The [timing record](readout_implementation_smoke.json) records the tested script
hash before the final reporting/provenance additions; the solver was unchanged.
The [root face audit](root_face_audit.md) records the pre-annotation rubric review.
An earlier concurrent fixture-test run stalled during a NumPy array copy and was
stopped; isolated loading and the final serialized suite passed. The final suite
used the documented memory-allocation pin for this large-tensor fixture. No
experimental data or labels were lost.

## Readout commands after annotation

Run from the dedicated experiment worktree. The loader requires a current,
accepted full pilot, a complete main annotation roster, and an aggregate sidecar
matching the labels, configuration, rubric/schema, row roster and completion
record. It will fail if those inputs are absent or inconsistent.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 uv run --no-sync python scripts/issue2564_answer_behavior_readout.py stage=fit first_fold_only=true
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 uv run --no-sync python scripts/issue2564_answer_behavior_readout.py stage=fit
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 uv run --no-sync python scripts/issue2564_answer_behavior_readout.py stage=summarize
```

The second command resumes the first fold by verified fingerprint. Outputs live
under the research root's `readout/`: per-fold checkpoints, selected penalties,
OOF predictions, bootstrap metric draws, per-target metrics and `summary.json`.
After actual completion, archive these inputs/outputs, review their interpretation
and publish a behavioral results report before proposing any manuscript changes.
