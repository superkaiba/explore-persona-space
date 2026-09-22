# Independent statistical implementation review — #2673 Kimi

Verdict: **APPROVE the inspected analysis changes.** No critical statistical or indexing bug found. This is an analysis review, not approval of the GPU capture adapter or its production numerical gates. Parent is adding the end-to-end regression tests; those checks remain necessary before accepting production analysis.

Reviewed 2026-09-22 in `/home/thomasjiralerspong/.codex/worktrees/story-persona-kimi-20260922`: the working diff of `scripts/story_persona_crossmodel_analysis.py`, new `scripts/story_persona_kimi_analysis.py`, Kimi prompt JSON, published Kimi rates JSON, and inherited analysis tests/metric helper. No source edits, GPU calls, task mutations, or experiment launches were performed.

## Findings

- Kimi layout explicitly requires the inherited ordered eight conditions plus `default`; Qwen and DeepSeek retain exactly the inherited eight. Dynamic accumulator sizes, half-slot indexing, reshape dimensions, and per-half coverage checks consistently use the validated condition count.
- Kimi's default prompt has `omit_system_message: true`, and analysis verifies that this condition's recorded messages contain only the user message. The other eight retain their system descriptions. Production rendering itself belongs to the capture review.
- The Kimi fold mask excludes the default condition for all full-bank and question-half fits. All nine centroids remain in the evaluation matrices. Together with validated Cartesian row coverage, this implies 1,920/960 calibration rows for the production question bank.
- The model pin, dimensions, predeclared layers, and immutable outcome-image pins are separated. The Kimi loader accepts the reviewed default/no-system source and five alternatives; the inherited loader retains the fixed HHH/Fred DeepSeek outcomes. Default rates are merged into a distinct third stratum. Each stratum records its outcome model; full Kimi source metadata and input-file hashes are preserved in summary/provenance.
- Direct cosine versus alternative uptake is explicitly exposed as `direct_other_uptake` and marked as the headline measure. Historical `primary_by_persona` fields still describe the helpful-relative contrast for compatibility. The supplementary pooled contrast excludes Kimi default, so it does not pool all three heterogeneous strata. Downstream reporting must use the explicit direct-association field for the user's focal result.
- Raw and whitened cosine both call the same paired-outcome implementation. The whitening helper remains uncentered, without vector pre-normalization, and uses the inherited label-free ridge selection and numerical residual/oracle checks. No behavioral-label fitting or significance testing was introduced.
- The weak-order feasibility algorithm is correct for finite closed intervals: tied-group intersections must be nonempty; each later group's upper endpoint must exceed every earlier group's lower endpoint. This permits a common endpoint as a tie while rejecting an impossible strict reversal. Rational recovery removes the known one-ULP discrepancy at the Dismissive/Peer touching boundary.

## Independent bounded checks actually run

Loaded the committed-path Kimi rate artifact through the new loader and called the new sensitivity implementation using a fixed five-element toy predictor. Independently specified all six expected feasible rank vectors and checked exact agreement:

- Dismissive below Peer, or Dismissive tied with Peer;
- Help-seeker below Saboteur, Saboteur below Help-seeker, or the two tied;
- Sarcastic above all four in every case.

The implementation returned exactly those six weak orders. A constant predictor returned an undefined Spearman bound explicitly. These CPU-only helper checks passed; they are not results from Kimi context vectors and do not substitute for the parent's full persisted-capture numerical regression test.

## Nonblocking cleanup

The summary limitation currently says `Pre-finetuning geometry versus post-story-finetuning DeepSeek aggregate outcomes.` For Kimi summaries, expand that to identify Kimi default outcomes as well as the fixed DeepSeek outcomes. Also update obsolete eight-persona error/docstring text where convenient. Neither issue changes current numerical outputs, but the source distinction should remain clear in exported summaries and figures.

The persisted `primary_by_persona` compatibility naming can mislead downstream consumers that ignore `headline_measure`; reports must select `direct_other_uptake`, not infer the focal measure from the old key name. No best-model/layer claim follows from this descriptive five-condition comparison.
