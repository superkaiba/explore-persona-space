# Qwen3-only reasoning section

User-approved edit: remove OpenThinker evidence from the compiled manuscript,
fill the two reasoning-section TODOs, and minimize plots. A subsequent explicit
request adds the verified Qwen3 operator-geometry result and aligns the caption
with the neighboring figures' takeaway-then-panel-findings structure.

## Initial section outline

- Same-weight thinking toggle; own-generated targets; existing all-training
  fits evaluated on the needs-reasoning subset.
- End-of-CoT versus context, followed by thinking on/off as its control.
- Similar end-of-CoT advantage across the two necessity groups.
- Different input subspaces but strongly overlapping predicted-answer subspaces.

The main asset is one figure with two axes: R2 and top-1 retrieval. Each shows
off/context, on/context and on/CoT-end. The two on conditions have identical
targets; the off condition has its own generated targets. The repeated
on/context condition from the previous layout is removed.

Conditional residual prediction remains in the appendix as text, not another
plot. The per-dataset necessity comparison is a table. OpenThinker-only
appendix inputs are removed from the manuscript; historical source files,
figures, code and raw results remain available and are not deleted.

## Evidence and checks

The figure metadata pins the three allfit Qwen3 JSONs and exact plotting script.
The table uses qwen3_necessity_table.json, a source-hashed snapshot of the
current shared-root necessity summary (newer than this branch's old copy).
The main result pools errors without dataset reweighting and uses training-fold
dataset-mean baselines. The group comparison pools SSE and baseline deviations
with equal total weight per dataset, using whole-dataset mean baselines, then
computes R2. It is not the mean of dataset-level R2 values. These calculations
are not interchanged.

Claim-evidence map: post-CoT prediction advantage and mixed toggle effects
are supported by the three frozen allfit evaluations. Similar group gains
are supported by the separate necessity summary. No causal computation or
intrinsic reasoning-necessity claim is made.

Five-dimension writing self-review: contribution is unchanged; prose now has
one comparison-plus-control finding and one group finding; empirical numbers
are unchanged; removed-model evidence no longer supports Qwen3 claims;
method descriptions retain own-answer targets, held-out evaluation, full-fold
retrieval pools, single-greedy labels and the noncausal scope.

Independent review checked every main score/interval, supplementary readout,
residual result and necessity-table cell against the saved artifacts. It caught
the table's aggregation description, corrected above and in the appendix.
Color, grayscale and manuscript pages 8, 9, 35 and 36 were visually inspected.
The manuscript builds successfully; two pre-existing unresolved references to
fig:effective-kernel-sae in the separate theory material remain outside this edit.

## Geometry insertion and caption revision

The geometry claim uses the source-hashed, independently verified
[Qwen3 results](https://github.com/superkaiba/explore-persona-space/blob/3c0d767b617b93ae5009e0857059196ba54f1366/eval_results/issue_2546/qwen3_map_geometry_20260906/results.json).
At k=50, mean principal cosine is 0.12276759278423352 for input subspaces,
0.9214098266397173 for output subspaces, and 0.09240138194772396 for the
isotropic reference. The input overlap is low but above the reference, not
statistically indistinguishable from random. The main paragraph reports this
contrast; Appendix G supplies the subspace definition, original-coordinate
convention, same-target design, reconstruction check and k=10/200 checks.

These are descriptive fits on all 33,810 rows with the original ridge recipes,
not the original held-out-fold operators. That exception is explicit in the
main paragraph; every held-out prediction score still uses the unchanged
evaluation maps and needs-reasoning scope. Shared answer covariance can
contribute to output overlap. Neither semantic meaning nor causal use of these
directions was tested. Claim/evidence status: supported as descriptive geometry.

Caption roles now match the neighboring captions: bold overall finding,
panel A's R2 finding, then panel B's retrieval finding. The displaced scope,
own-target matching and interval details are retained in the methods appendix.
The two-panel plot asset itself is unchanged; no geometry plot is added.

Five-dimension self-review: the added contribution is narrowly descriptive;
each paragraph has one message; exact values and all-row scope are checked;
k=10/200 comparisons support robustness without an uncertainty claim; and the
original-coordinate comparison is distinguished from semantic or causal
interpretation. No new experiment is required for this insertion.

Independent review passed the new geometry paragraph, appendix and caption
against the current JSONs. The 40-page manuscript compiled successfully, and
updated pages 8, 9, 36 and 37 were visually checked. The same two unresolved
references in the unrelated, author-edited theory block remain unchanged.

## Two-claim revision after author edit df82df7

The author's latest Overleaf edit removes the main geometry paragraph and
supplies a two-claim draft. The current cleanup preserves that removal;
geometry and conditional residual prediction remain appendix material. The
subsection title is narrowed to "The realized chain of thought improves answer
predictability" to match the analyses presented in the main text.

Current outline and claim-evidence map:

- Setup: Qwen3-8B's same-weight thinking modes, own-generated answer vectors,
  and existing maps fitted on all training questions.
- End-of-CoT versus context on needs-reasoning questions, with the thinking
  toggle as a control. The existing two-panel prediction/control figure
  supports this single claim; its plotted values are unchanged.
- Similar observed end-of-CoT gains across necessity groups. A separate,
  compact figure plots the four saved equal-dataset-weighted scores and
  intervals from the existing necessity table. No new fits, experiments,
  bootstraps or interaction tests are run.

The control is described as no consistent improvement across R2 and retrieval,
not proof of an intrinsic property of answers. "Keeping the input readout at
h_C" does not assert that thinking-on/off templates produce identical context
vectors. The group claim says "similar," not "not larger" or "equal": the
observed R2 gains are 0.08272513714672236 and 0.07823502385365289. The appendix
states that similarity does not establish statistical equivalence.

The group figure uses qwen3_necessity_table.json, verified against the current
shared-root necessity_r2/summary.json (SHA256
76d5e793c2b14b90106d12ce1b25d2a62781ef88715634a2607d0f0f9032083a).
It retains equal total dataset weight in the pooled SSE/SST ratio and the
whole-dataset-mean baseline. Saved 95% intervals resample questions within each
dataset and necessity group. Counts remain 4,522 and 17,693. These numbers use
a different aggregation and baseline from the first figure; neither metric is
silently substituted for the other.

Five-dimension writing self-review: the contribution is unchanged; the setup
defines the exact readouts and fit scope; each claim has its own figure; every
number comes from a source-hashed saved artifact; and the interpretation avoids
causal or statistical-equivalence claims. Rendering uses the canonical C2A
style, preserving the context/CoT-end colors and markers across both figures.
Color and grayscale outputs and the compiled main pages were visually checked.
The metadata's source hashes, plotted scores/intervals and final script hash
were verified. Ruff and manuscript compilation pass; pre-existing unresolved
effective-kernel-sae references in unrelated theory material remain.

Independent read-only review passed the numerical/provenance/scope checks and
confirmed that no main geometry paragraph or unrelated theory edit was added.
Browser previews of both figures are available in the
[current manuscript](https://www.overleaf.com/project/6a59c927290f8b8b5eee0055).

## Combined A/B bar figure and results-section structure (2026-09-07)

The next author request combines the two plots into one A/B figure and switches
to bars. The [manuscript](https://www.overleaf.com/project/6a59c927290f8b8b5eee0055)
now includes only c1_cot_story.pdf for these claims. Panel A groups R2 and
top-1 retrieval by metric, each with the same three prediction/control
conditions; retrieval is explicitly a fraction. Panel B groups the context
and end-of-CoT R2 scores by necessity label. Both y-axes start at zero and
retain the original confidence intervals. Filled versus open/hatched bars
distinguish metrics, with a shared condition legend and redundant edge styles.

Current section outline and paragraph roles:

- Opening/design: transition from training and speakers, same-weight Qwen3
  modes, own-answer metamodel targets, appendix pointer, then "The results are
  shown in Figure ... We find:".
- Evidence A: bold finding, panel reference, colon, then the same-answer
  context/end-of-CoT comparison and own-answer thinking-toggle control.
- Evidence B: matching finding/reference format, operational group definitions,
  then the equal-dataset-weighted comparison.
- Conclusion: observing CoT improves prediction; enabling thinking alone does
  not consistently improve context predictability; the readout gain occurs
  in both necessity groups.

Claim/evidence status is unchanged: A uses the three existing allfit summaries,
B uses the saved necessity snapshot. Both claims remain supported within their
stated scope, with no causal or statistical-equivalence claim. The author's
latest removal of the all-training-fit sentence from the main setup is
preserved; fit scope remains explicit in the appendix. The paragraph and
caption structure now matches the neighboring results sections. No OpenThinker
or main geometry text is reintroduced.

The default renderer exports one figure bundle. The superseded standalone
necessity renderer is removed; its prior assets remain historical and are no
longer included by the manuscript. All main and appendix references are
rewired to panel A or B, including the methods' needs-only denominator and
training-fold baseline statement.

Verification: the real production renderer/exporter is exercised by
tests/test_section45_reasoning_story.py. It checks all ten bar heights and CI
endpoints against the saved inputs, zero bar baselines, CI visibility order,
metric hatching, no point markers, two axes, the one-figure export, source and
script hashes, and no-refit metadata. Ruff and the test pass. The source
snapshot's plotted values and source hash match the current shared-root
artifact. Color/grayscale and compiled manuscript pages 8, 9 and 35 were
visually inspected. Independent review passed, including methods panel scope
and byte-identical copied PDF. Compilation succeeds with the same two unrelated
unresolved effective-kernel-sae references.

Five-dimension self-review: no contribution change; consistent paragraph and
caption roles; no altered scores or intervals; both metrics and the necessity
comparison retained; and no change to map fitting or evaluation design.
