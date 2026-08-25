# Deep literature review: does context-to-answer mapping quality rise with model capability?

Task #2588. Protocol: `.claude/skills/deep-lit-review/SKILL.md`. Conducted 2026-08-25.
Discovery (Steps 1 to 3) fanned out across four scouts; screening, reads, synthesis,
and verification (Steps 4 to 7) by the orchestrator.

**Coverage honesty, stated up front:** all four scouts stopped on BUDGET, not on the
protocol's two-consecutive-dry-rounds criterion, and every scout's final round was
still surfacing new relevant papers. Semantic Scholar was rate-limited (HTTP 429) for
three of four scouts, so the citation-graph snowball is under-covered. This review is
therefore a strong orienting pass, not a saturated one. Named residual veins are
listed in the verification log.

---

## 1. Question and criteria (frozen before searching, verbatim)

**Research question.** Does the quality of a LINEAR map from a language model's
context representation (residual state at the last prompt token) to its own answer
representation (token-mean over its generated answer) increase with the model's
general capability, across models of different families and sizes? And does
chain-of-thought (taking the end-of-CoT state as the map input) change that
relationship?

**Inclusion criteria.** (1) Fits or analyses a linear probe or map over LM internal
representations AND reports variation with scale or capability. (2) Predicts a model's
own output or behavior from its pre-generation internal state, or fits a map between
two internal spaces of the same model. (3) Analyses how CoT or reasoning training
changes representations, especially the end-of-reasoning state. (4) Defines or
critiques a composite LM capability index. (5) Supplies cross-width comparability or
under-determined-ridge estimator discipline.

**Exclusion criteria.** Benchmark-only papers with no representation analysis;
single-model probing with no scale axis and no self-output-prediction framing;
cross-MODEL representation alignment except where it bears on cross-width
comparability; non-LM domains except as method sources.

---

## 2. The gap this experiment sits in

No retrieved paper fits the object #2588 fits. The four nearest, with what each
changes:

| Nearest prior | What it maps | Why it is not this |
|---|---|---|
| Future Lens, 2311.04897 | single hidden state to subsequent TOKENS (and future hidden states) | target is tokens, not an answer representation; one model (GPT-J-6B); no capability axis |
| Tuned Lens, 2303.08112 | per-block affine probe, hidden state to the model's own output DISTRIBUTION | target is a vocabulary distribution; framed as iterative-inference analysis, not capability comparison |
| 2606.14530 | hidden state at the FINAL PROMPT TOKEN to eventual code correctness | exact map INPUT, but target is a scalar label; one model (Qwen3-4B-Instruct-2507) |
| FIRP, 2410.20488 | current hidden states to future intermediate hidden states | same-model state-to-state map, but motivated by decoding speed; no capability axis |

So the novel content of #2588 is the conjunction of (a) an answer-REPRESENTATION
target rather than a token, a distribution, or a scalar, and (b) a cross-family
capability axis. Both halves have prior art separately; the conjunction does not
appear in what was retrieved.

Worth noting from 2606.14530, because it independently supports this project's
linear-by-default rule: at the last-prompt-token position, "none of the nonlinear
models tested improves upon" the linear probe.

---

## 3. External priors on H1 (capability raises mapping quality)

**Supporting.**

- **2311.04897** is the nearest formalization and reports a single hidden state
  suffices to approximate the model's own subsequent output at above 48 percent
  accuracy at some layers, establishing that a pre-generation state carries
  substantial forward information at all.
- **2604.13386** is the strongest direct external support: across 12 models spanning
  0.5B to 176B, "probe accuracy improves with scale: ~5% AUROC per 10x parameters
  (R=0.81)", with the interpretation that "larger models linearly represent more
  general concepts".
- **2502.13329** reports that probes on input-token representations alone predict
  eventual whole-sequence behaviors, and that "probes generalize to unseen datasets
  and perform better on larger models".
- **2404.00859** finds that in the autoregressive setting the evidence favours the
  breadcrumbs account over pre-caching, "though pre-caching increases with model
  scale". A qualified, not clean, pro-H1 datapoint.
- **2405.07987** argues representations converge and that "as vision models and
  language models get larger, they measure distance between datapoints in a more and
  more alike way", i.e. distance structure becomes more shared with scale. (The
  mutual-kNN operationalization is body-level, not in the abstract.)

**Opposing or non-monotone.**

- **2605.27958** pressure-tests deception probes across Gemma 3 (1B to 27B), a
  within-family size ladder structurally analogous to the Qwen3.5 ladder here, and
  finds an inverse-scaling pattern that it then explains away: style-augmented probes
  "recover near-perfect detection at both 4B and 27B, establishing that the inverse
  scaling pattern is a training-distribution artifact rather than a genuine
  scale-dependent phenomenon". Two lessons: apparent capability trends in probe
  quality can be training-distribution artifacts, and the diagnosis required varying
  the probe's training distribution, not just the model.
  (Caution: the per-size numbers 1B 0.652, 4B 0.759, 12B 0.609, 27B 0.485 circulated
  by discovery are NOT in the abstract. Treat them as unverified pending a full-text
  read; the qualitative inverse-then-explained pattern IS abstract-grounded.)
- **2510.18147** trains probes across 60 models and splits cleanly by construct:
  human-labeled difficulty "is strongly linearly decodable (AMC: rho about 0.88) and
  exhibits clear model-size scaling, whereas LLM-derived difficulty is substantially
  weaker and scales poorly". Under GRPO training the LLM-difficulty probe "degrades
  and negatively correlates with performance". So whether a readout improves with
  capability depends on which construct it reads.
- **In-project priors point the same way.** #2330 found Qwen2.5-7B-Instruct beats
  Qwen3.5-9B on held-out map R-squared (0.705 vs 0.661, p = 0.002), and #507 found
  scaling 7B to 72B made a leakage predictor strictly worse. Different predictor
  families, but the local evidence is not pro-H1.

**Net.** The external literature leans toward H1 for probe-style readouts of
externally-labeled constructs; the in-project literature leans against it for the
context-to-answer map specifically. That tension is the reason #2588 is worth running,
and it is why the fixed-size 27B column matters: it is the only part of the design that
separates a genuine capability effect from a width or family effect.

---

## 4. The methodological hazard that will decide the result

**2602.14486** is the single most consequential paper retrieved. It shows that
"existing metrics used to measure representational similarity are confounded by
network scale: increasing model depth or width can systematically inflate
representational similarity scores", introduces "a permutation-based null-calibration
framework that transforms any representational similarity metric into a calibrated
score with statistical guarantees", and finds after calibration that "the apparent
convergence reported by global spectral measures largely disappears, while local
neighborhood similarity, but not local distances, retains significant agreement".

Three consequences for #2588, all actionable:

1. An uncalibrated cross-width comparison can manufacture a capability trend out of
   width alone. This is exactly hypothesis H0, and the literature says the effect is
   real and systematic rather than hypothetical.
2. Every per-model reading needs a per-model permutation null, and the trend should be
   fit on null-calibrated scores. #2330's shuffled-pairing null is the right shape
   already; it should be promoted from a floor check to the calibration instrument.
3. It endorses the retrieval-primary choice on independent grounds: local NEIGHBOURHOOD
   structure is what survived calibration, and kNN retrieval is a neighborhood read.
   It also warns that local DISTANCES did not survive, so distance-valued summaries
   should not carry the headline.

Supporting estimator discipline, consistent with the standing #1887 rule: the ridge
literature (2408.04607 on GCV failing under correlated samples, 2403.20200 on the
degrees-of-freedom framing, 2310.04357 and 2406.11666 on the overparameterized regime)
establishes that out-of-sample risk depends on the aspect ratio d/n. Because d spans
roughly 1024 to 8192 across this panel, a fixed n_train puts every model at a
different point on its own risk curve, so raw cross-model R-squared conflates map
quality with estimator regime. R-squared stays a within-model diagnostic.

Counterweight to keep the kNN claim honest: distance concentration (Beyer et al. 1999,
DOI 10.1007/3-540-49257-7_15; Aggarwal et al. 2001, DOI 10.1007/3-540-44503-X_27, both
cited by 2602.14486) means kNN discriminability itself varies with intrinsic
dimension, and LM intrinsic dimension varies across models and layers (2501.10573,
2503.02142). Equal chance floors do not mean equal difficulty. And 2605.26973 shows
alignment estimates depend on signal-to-noise ratio and sample size, so pool size, k,
and n must be held fixed across models. No retrieved paper adjudicates kNN retrieval
versus R-squared head-to-head for cross-width map quality; that question is unsettled.

**Layer selection is a first-class confound, not a detail.** 2604.13386 reports "the
best layer varies across models and tasks" and that single-layer probes are fragile,
with multi-layer ensembling recovering performance where single layers fail.
2509.10625 finds predictive power "saturates in intermediate layers". #2330 densely
swept only the 9B and compared it against the 7B's best of three captured layers. A
fixed layer index across a 12-checkpoint, 4-architecture panel would be a confound;
a per-model dense sweep with a stated, uniform selection rule is required, and any
best-of-L read needs selection-symmetric treatment.

**Format and length confounds must be residualized.** 2606.02907 is the cautionary
case: linear probes hit "100% cross-validated accuracy with well-separated geometry",
and "this separation is entirely driven by format confounds. Residualizing source
identity, option count, and response length reduces accuracy to chance." This lands
directly on #2588's two-arm design, because thinking-on and thinking-off responses
differ systematically in length, so an arm-b-minus-arm-a gap is confounded with
response length by construction. 2606.14530 supplies the matching remedy: residualize
each hidden-state dimension against prompt length and report the probe's surviving
performance (0.881 to 0.842 AUC in their case) against a length-only baseline.

---

## 5. Priors for the chain-of-thought arm (H2)

**When is the answer determined?** Two camps, and the disagreement is substantive.

- Early: **2603.05488** finds "the model's final answer is decodable from activations
  far earlier in CoT than a monitor is able to say, especially for easy recall-based
  MMLU questions", and explicitly contrasts this with "genuine reasoning in difficult
  multihop GPQA-Diamond questions". **2606.13603** finds reasoning "crosses a
  commitment boundary, a sharp transition from transient intermediate guesses to a
  stable, high-confidence answer", often in a single step "well before the model's
  reasoning block ends", followed by epiphenomenal steps that leave the answer
  probability unaltered, with answer-formation stages "linearly decoded from
  intermediate reasoning steps with high accuracy".
- Late: **2605.10799** reports that "generation-time probes indicate that final
  answers are rarely early-determined during generation (<5% early commitment)", and
  argues the standard corruption-study evidence "largely measure[s] answer placement
  rather than where intermediate computation is carried out".

**2603.05488 is the most directly transferable result in this review**, because it uses
GPQA Diamond, #2588's exact hard set, and finds the easy-versus-hard split governs
pre-answer decodability. It supports a concrete pre-registered prediction: the
end-of-CoT state should buy MORE on GPQA Diamond than on the generic corpus, because
on easy recall items the answer is already decodable from the context state, leaving
the CoT arm little to add.

**Does reasoning training change linear readout?** Also contested.

- **2607.26119** finds "linear probes trained on layer-wise hidden states reveal that
  RL models tend to achieve higher accuracy in predicting answer correctness compared
  to SFT models, indicating more linearly separable and structured representations",
  and that "RL training fundamentally restructures how models represent and process
  reasoning problems".
- **2601.21192** finds the opposite for RLVR specifically: it "induces irreversible
  latent manifold's local geometry reorganization and reversible coordinate basis
  drift", but "preserves the global manifold geometry and linear readout", concluding
  RLVR "optimizes trajectories within an existing semantic landscape rather than
  fundamentally restructuring the landscape itself", with a reported null effect
  downstream.

For #2588 this means H2 is genuinely open and the OLMo Instruct-versus-Think pairs are
the cleanest test available in the panel, since they are paired checkpoints from one
base and therefore isolate reasoning training with family and width held fixed.

**Where to take the end-of-CoT state.** Precedent exists and should be inherited rather
than re-derived: 2605.05980 treats the `</think>` token as "a canonical decision point,
which marks the end of reasoning and the start of actions" and projects the hidden
state at that token; 2604.01202 extracts hidden states "at structural token positions
including think_end" plus a within-CoT percentile sweep; 2607.21433 is the one abstract
pinning both token and layer explicitly. Coconut (2412.06769) is the precedent for
treating the last pre-answer hidden state as the reasoning-state object. In-repo,
#2546's `compute_read_idx` and `assert_think_pins` already implement this.
2606.02907's toggle discipline (set thinking off explicitly and strip thinking blocks
before analysis, to avoid mode-specific verbalization structure confounding hidden-state
geometry) is the cleanest stated control for the thinking-off arm.

Canonical faithwithfulness context, for framing rather than method: 2307.13702 (early
answering and its siblings), 2305.04388 (stated CoT does not reflect the true
determinants of the answer).

---

## 6. The capability axis: what the index is, and three problems

Primary source, retrieved and independently re-verified by the orchestrator on
2026-08-25:

https://artificialanalysis.ai/methodology/intelligence-benchmarking

Artificial Analysis Intelligence Index **v4.1.1** is a weighted average of nine
evaluations in four categories: Agents 34 percent (GDPval-AA v2 20, tau3-Banking 14),
Coding 24 percent (Terminal-Bench v2.1 16, SciCode 8), Scientific Reasoning 24 percent
(Humanity's Last Exam 12, **GPQA Diamond 6**, CritPt 6), General 18 percent
(AA-Omniscience 12, AA-LCR 6). Stated precision is a 95 percent confidence interval
"of less than plus or minus 1%".

**Problem 1, circularity. GPQA Diamond is a component of the index at 6 percent
weight, and #2588 uses GPQA Diamond as its hard eval surface.** The dependent variable
is map quality rather than GPQA accuracy, so this is not fatal, but the covariate and
the eval surface overlap and the plan must say so. The cheap mitigation is to report
the trend against a GPQA-excluded reweighting of the index alongside the headline, and
to state the overlap wherever the hard-set result is presented.

**Problem 2, scale type.** The composite mixes pass@1 accuracies with a clamped affine
Elo transform: "GDPval-AA v2 Elo scores are frozen at the time of a model's addition
and normalized as clamp((Elo - 500) / 2000)". Equal index differences therefore do not
correspond to equal capability differences across the range, so the index is not
defensibly interval-scaled. Nothing retrieved validates composite LM indices as
interval scales. The defensible read is rank-based: Spearman against the index, with
robustness across alternative capability proxies. Version drift compounds this: the
page notes GDPval-AA v2 differs from the version used in Intelligence Index v4.0
(re-baselined Elo, changed judge panels, changed turn limits), and the page makes no
statement about cross-version comparability, so the version must be pinned (v4.1.1)
and the retrieval date recorded.

**Problem 3, family-correlated covariate error.** Contamination is family-dependent
(2410.16186), and leaderboard-style indices carry documented distortions (2504.20879
on undisclosed private testing and asymmetric sampling; 2501.17858 on vote rigging;
2604.11581 on measurement error that standard confidence intervals ignore). The
covariate's error is therefore systematic along exactly the axis #2588 varies, namely
model family. This is a further argument for the fixed-size 27B column, where family
and width are held constant.

**Panel coverage, verified this session.** Per-thinking-mode index entries DO exist as
separate model variants, contrary to what the methodology page documents: Artificial
Analysis publishes, for example, a direct non-reasoning-versus-reasoning comparison for
one checkpoint at
https://artificialanalysis.ai/models/comparisons/qwen3-6-35b-a3b-non-reasoning-vs-qwen3-6-35b-a3b
and uses `-non-reasoning` model slugs. Dedicated pages exist for every ladder rung
(`qwen3-5-0-8b`, `qwen3-5-2b`, `qwen3-5-4b`, `qwen3-5-9b`, `qwen3-5-27b`,
`qwen3-6-27b`, `qwen3-8-27b`) and for both OLMo pairs (`olmo-3-7b-instruct`,
`olmo-3-7b-think`, `olmo-3-1-32b-instruct`, `olmo-3-1-32b-think`).

**But: several OLMo values are published as ESTIMATED, not measured** (Olmo 3 7B
Instruct and Olmo 3.1 32B Instruct and Think were all returned as estimated). An
estimated index value is an interpolation, which is precisely what the task body
forbids relying on. The plan must record measured-versus-estimated per checkpoint,
exclude estimated values from the primary trend or run the trend both ways, and note
that the OLMo control arm sits low and compressed on the axis relative to the Qwen
ladder, which limits its power. Note also that Olmo 3 32B Think appears to carry a
measured value and may be a better-grounded control than the 3.1-32B pair; that is a
panel decision for the planner.

Exact per-checkpoint index values are deliberately NOT pinned in this document: the
retrieval returned two mutually inconsistent sets of numbers for the Qwen3.5 ladder
(one set plausibly non-reasoning, one plausibly reasoning), so the planner must read
each value off its own model page, record the mode, the index version, and the
retrieval date, and never carry a number forward from a search summary.

---

## 7. What the plan should inherit, in priority order

1. **kNN retrieval primary, R-squared within-model only**, on the aspect-ratio argument
   (2408.04607, 2403.20200, 2310.04357, 2406.11666) plus the neighborhood-survives-
   calibration finding (2602.14486). Report both regardless, per the standing
   both-reads rule, and pre-register that they may dissociate, because they already did
   in #2330.
2. **Permutation-based null calibration per model** (2602.14486), promoting #2330's
   shuffled-pairing null from a floor check to the calibration instrument. Fit the
   capability trend on calibrated scores.
3. **Per-model dense layer sweep with one uniform selection rule** (2604.13386,
   2509.10625, and #2330's asymmetry), with selection-symmetric treatment of any
   best-of-L read.
4. **Residualize response length and prompt length**, and report the surviving effect
   against a length-only baseline (2606.02907, 2606.14530). This is mandatory for the
   arm-b-minus-arm-a comparison specifically.
5. **Rank-based capability analysis with the index version pinned** and a
   measured-versus-estimated flag per checkpoint; a GPQA-excluded reweighting reported
   alongside the headline to address the circularity.
6. **Match n_train, pool size, and k across models** (2605.26973), and report intrinsic
   dimension or effective rank per model as a covariate (2501.10573, 2503.02142) so
   distance concentration is visible rather than assumed away.
7. **Take the end-of-CoT state at the think-end token**, reusing #2546's
   `compute_read_idx` and `assert_think_pins`, with 2606.02907's strip-thinking-blocks
   discipline for the thinking-off arm.
8. **Pre-register the hard-set prediction** from 2603.05488: the end-of-CoT state should
   buy more on GPQA Diamond than on the generic corpus.
9. **Treat H1 as the claim needing evidence.** External probe literature leans for it;
   #2330 and #507 lean against it for maps specifically. The fixed-size 27B column is
   the measurement that decides, so it must be powered as the primary comparison rather
   than a secondary column.

---

## 8. Included, excluded, and per-paper notes

**Included and verified (18).** Every id below was resolved via `get_abstract` this
session and its title matched the note.

| id | role in the review |
|---|---|
| 2604.13386 | strongest external pro-H1; also the best-layer-varies warning |
| 2605.27958 | strongest external non-monotone case; artifact diagnosis |
| 2510.18147 | 60-model probe census; construct-dependent scaling; probe degrades as model improves |
| 2502.13329 | pre-generation probes predict whole-sequence behavior; better on larger models; CoT-final prediction |
| 2509.10625 | closest to this map's input position; 3 families 7B to 70B; saturates mid-layers; falters on math |
| 2606.14530 | last-prompt-token to eventual correctness; length-residualization diagnostic; linear beats nonlinear |
| 2311.04897 | closest prior formalization (state to future tokens) |
| 2303.08112 | canonical affine state-to-own-output map |
| 2404.00859 | pre-caching increases with scale (qualified) |
| 2405.07987 | distance structure becomes more shared with scale |
| 2602.14486 | scale confound in similarity metrics; permutation null calibration; neighborhood survives |
| 2210.16156 | CKA manipulable without functional change; why not to use it here |
| 2311.03658 | LRH formalization, linear probing and steering |
| 2405.14860 | multi-dimensional features; the linearity-does-not-always-hold side |
| 2603.05488 | answer decodable early in CoT; easy MMLU vs hard GPQA-Diamond contrast |
| 2606.13603 | commitment boundary before the reasoning block ends; epiphenomenal steps |
| 2605.10799 | counterpoint: under 5 percent early commitment; format confound in corruption studies |
| 2607.26119 | RL more linearly separable than SFT |
| 2601.21192 | RLVR preserves global geometry and linear readout; null effect |
| 2606.02907 | probes detect task format not reasoning mode; residualization collapses to chance; toggle discipline |
| 2605.09502 | cross-family 1.5B to 72B linear decodability of trace correctness (no scale trend claimed) |

**Excluded, with the failing criterion.**

- 2601.11516 (Building Production-Ready Probes For Gemini): verified, but the abstract
  reports long-context distribution shift and deployment, with no scale axis. Fails
  criterion 1. Retained as a probe-robustness method source only.
- 2510.12680, 2605.26242, 2503.07513 and the wider introspection cluster: behavioral
  self-report rather than internal-state mapping. Fails criteria 1 and 2 as scoped.
- 2408.10920 (non-linear representations in RNNs): the canonical strong-LRH
  counterexample, but RNNs rather than transformer LMs. Excluded under the non-LM
  clause; noted here so it is not silently dropped.
- Efficiency and early-exit papers (2512.05325, 2509.24248, 2505.18404, 2603.22016,
  2607.28966, 2605.07315): probe-anchored but optimizing inference cost. Out of scope
  except as anchor-point precedent.
- The 8 title-only snowball hits from scout C section G and the title-only tier-2 items
  from scout B: not screened, abstracts never pulled. Recorded as unscreened rather
  than included or excluded.

**Unresolvable leads, recorded not dropped.**

- Golub, Heath and Wahba 1979, the origin of generalized cross-validation, is grounded
  only by a ResearchGate retrieval hit because the Semantic Scholar call that would
  have produced a clean DOI was rate-limited.
- Ding, Denain and Steinhardt 2021 (Grounding Representation Similarity Through
  Statistical Testing) was retrieved via citation graph with a DBLP key and no arXiv
  id in the record.
- A non-arXiv reference titled "Why I'm not worried about non-linear feature
  representations" appears in 2604.13386's reference list with no id or authors.

---

## 9. Verification log (Step 7)

**7.1 Resolution.** All 21 arXiv ids in the included table plus 2601.11516 were
resolved via `mcp__arxiv__get_abstract` in this session; every resolved title matched
the note. Zero unresolvable ids among cited work. Ids grounded only at title level by
discovery were either verified before citation or moved to the unscreened list.

**7.2 Claim versus source.** Three corrections applied to discovery output:

1. The per-size probe-robustness numbers attributed to 2605.27958 (1B 0.652, 4B 0.759,
   12B 0.609, 27B 0.485) are NOT in its abstract. The qualitative inverse-scaling
   pattern and its training-distribution-artifact diagnosis are abstract-grounded; the
   numbers are marked unverified pending full text.
2. The mutual-kNN alignment metric attributed to 2405.07987 is not in its abstract,
   which grounds only the weaker claim that larger models measure inter-datapoint
   distance more alike. The metric specifics are body-level and marked as such.
3. 2509.10625 and 2605.09502 were both offered as cross-family evidence. Neither
   abstract claims a scale TREND, so neither is counted as a pro-H1 datapoint; both
   are cited only for cross-family linear decodability.

**7.3 Coverage pass.** The disconfirming-results search ran as an explicit brief item
for scout A and returned the strongest counter-evidence in the review (2605.27958,
2510.18147, 2405.14860), so the sweep is not purely positive-skewed. However coverage
is INCOMPLETE and no candidate pile was silently truncated: all four scouts stopped on
budget with their final round still productive; Semantic Scholar failed for three of
four; no citation-graph snowball completed for scout B. Named residual veins for a
follow-up pass, in priority order:

1. Snowball Future Lens (2311.04897) and Tuned Lens (2303.08112), the highest-value
   unsnowballed roots for the closest-formalization question.
2. Snowball 2603.05488 and 2606.13603, the two most on-point end-of-CoT decodability
   papers.
3. Pull abstracts for the 8 title-only hits in scout C section G, several of which are
   plainly on-topic by title, notably 2604.06613 (The Detection-Extraction Gap: Models
   Know the Answer Before They Can Say It) and 2608.17124 (A decodability criterion
   predicts when hidden-state selection beats majority voting).
4. Retry Semantic Scholar once quota clears, and pull 2502.06258 (Emergent Response
   Planning in LLMs), grounded by title only.
5. The classic BERTology probe-versus-size line never surfaced and may need its own
   query.

These gaps do not change the review's load-bearing conclusions, which rest on verified
abstracts: the gap statement in section 2, the two-sided prior in section 3, the scale
confound and its calibration remedy in section 4, and the index problems in section 6.
