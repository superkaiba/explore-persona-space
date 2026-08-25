# Deep literature review — #2544, the context→answer map across a pretraining ladder

Run at `/issue 2544` Step 2, before the planner was dispatched, per the task body's
`## Open decisions for the planner` item 4 and the standing pre-experiment lit-review rule.
Protocol: `.claude/skills/deep-lit-review/SKILL.md` (retrieval-grounded discovery fanned out
across four channel-disjoint scouts; screening, reading, synthesis and verification run by the
orchestrator).

Date: 2026-08-24.

---

## 1. Question and criteria (frozen before any search)

**Question, verbatim:** Does any prior work measure a context→answer (prompt-state →
response-state) linear activation map — or a closely equivalent hidden-state-to-hidden-state
readout — across a pretraining checkpoint ladder, and what does the literature establish about
*when* during pretraining such structure forms?

**Inclusion (any one suffices).** I1 measures internal representations at ≥3 pretraining
checkpoints of one run · I2 fits a map/probe *between* two hidden states with a held-out
predictivity read · I3 dates emergence of in-context learning or induction heads against
tokens/compute · I4 compares representations across checkpoints of one run via
CKA/Procrustes/general-linear/SVCCA · I5 reports few-shot minus zero-shot as a function of
pretraining tokens.

**Exclusion.** E1 benchmark-only scaling dynamics with no internal measurement · E2
post-training-only dynamics with no pretraining axis, unless it is the method source · E3
single-checkpoint probing that is also not a state→state map · E4 non-LM domains unless the
over-training alignment method is what we would borrow.

## 2. Search log (summary)

Four scouts, channel-disjoint, each looping to dryness within budget.

| Scout | Scope | Channels | Rounds | Dry? |
|---|---|---|---|---|
| A | representation dynamics on open pretraining ladders | arXiv MCP, WebSearch, Semantic Scholar | 18 | yes on the core construct (2 consecutive) |
| B | induction-head / ICL emergence timing, exemplar sensitivity | arXiv MCP, WebSearch, S2 | 20 | yes (2-3 consecutive) |
| C | cross-checkpoint alignment estimators + failure modes | S2, OpenAlex, arXiv MCP | 22 | 1 dry round; budget exhausted before a 2nd |
| D | closest-prior hunt for a state→state map | arXiv MCP, WebSearch, S2 | 18 | yes (2 consecutive on the web channel) |

**Coverage caveat, stated because it bounds the negative result.** The Semantic Scholar REST
channel was rate-limited (HTTP 429) on essentially every attempt across all four scouts — one
usable response out of ~20 calls. Citation-graph snowballing still ran (through the S2-backed
arXiv MCP `citation_graph` and through OpenAlex `cites:`), so the graph was not unexplored, but
S2's keyword channel contributed almost nothing. Scout C additionally snowballed only from
Kornblith 2019, not from Ding 2021 or Davari 2022. The novelty finding in §5 should be read with
that bound attached.

## 3. Verification log (mandatory, run by the orchestrator)

**7.1 Resolution.** All **118** distinct arXiv ids returned by the four scouts were resolved
against the arXiv API in batch, and every returned title was compared against the scout's
reported title. **118/118 resolved; zero missing; zero title mismatches; zero fabricated ids.**

**7.2 Claim-vs-source.** Four load-bearing claims were re-checked against the paper text, not
the scout's paraphrase. Two were flagged by the scouts themselves as secondhand.

1. **Induction-head vs FV-head timing (2502.14010)** — scout reported it from a WebSearch
   summary. VERIFIED against the paper body, §4 Results, verbatim: *"Our analysis reveals a
   consistent pattern across all Pythia models: induction heads emerge early in training, at
   around step 1,000 out of 143,000, while FV heads appear substantially later at around step
   16,000."* Same section, also verbatim: *"induction scores exhibit a sharp initial rise
   followed by a plateau or slight decline, whereas FV scores demonstrate a gradual but
   sustained increase from step 16,000 through the end of tr[aining]."* CONFIRMED as reported.
2. **The induction-head emergence law (2511.16893)** — VERIFIED against §7 Results. The fitted
   parameters are α = 13.26, β = −0.37, γ = −0.62, giving `U_PT = T / (B^0.37 · C^0.62)` with
   `T = e^α`, and the token form `N_PT = T · B^0.63 · C^0.38`. Confirmed, and used in §6 below.
3. **The closest prior formalization (2308.09124, LRE)** — the scout explicitly flagged its
   characterization as written from memory and asked for verification. CHECKED against the
   abstract, and the scout's gist holds while its specifics needed correction. Verbatim: *"for
   a subset of relations, this computation is well-approximated by a single linear
   transformation on the subject representation"*, and the map is *"obtained by constructing a
   first-order approximation to the LM from a single prompt"* — i.e. a **Jacobian first-order
   approximation, not a regression fitted over a corpus with a held-out R²**. Also verbatim:
   *"we also identify many cases in which LM predictions capture relational knowledge
   accurately, but this knowledge is not linearly encoded in their representations."* The
   correction matters: it widens, not narrows, the gap between LRE and our object.
4. **Steerability emergence timing (2508.01892)** — the scout reported "~68% of training steps"
   from a WebSearch summary. **That figure is NOT in the abstract and is NOT quoted anywhere in
   this review.** What IS verbatim in the abstract: linear steerability *"emerges during
   intermediate stages of training"*; *"even closely related concepts (e.g., anger and sadness)
   exhibit steerability emergence at distinct stages of training"*; and *"concepts become
   increasingly linearly separable in the hidden space as training progresses, which strongly
   correlates with the emergence of linear steerability."* The qualitative claim is confirmed;
   the specific percentage is dropped as unverified.

**7.3 Disconfirming-evidence pass.** Explicitly run, and it changed the review's conclusion: see
§4, where the evidence on H1 turns out to be split rather than supportive, and §7, where a
non-monotone formation curve is shown to be a documented possibility.

## 4. What the literature establishes about WHEN structure forms

The single most useful thing this review found is that **the timing evidence is split, and the
split runs along the axis of what kind of structure is being measured.** The task body's H1
("most of the final map strength reached in the first ~10% of tokens") reads the early-formation
side of that split as though it were consensus. It is not.

**The early camp — circuit-level structure.** Induction heads form at *"around step 1,000 out of
143,000"* in Pythia, under 1% of training (2502.14010, verified quote). The formation point is
predictable and model-size-agnostic, depending only on batch and context size (2511.16893,
verified). Trustworthiness concepts become linearly separable early in pretraining (2402.19465).
Vocabulary embeddings organize early (2510.07613). The originating result for abrupt
circuit formation is 2209.11895.

**The intermediate/late camp — content-level linear readouts, which is what our object is.**
Linear steerability *"emerges during intermediate stages of training"* (2508.01892, verified),
and — the detail with the sharpest design consequence — *"even closely related concepts (e.g.,
anger and sadness) exhibit steerability emergence at distinct stages of training."* Function-vector
heads, which 2502.14010 finds drive few-shot ICL in larger models, appear at step 16,000 of
143,000 (~11%), an order of magnitude later than induction heads, and their scores rise
*"gradual[ly] but sustained"* rather than sharply. Polar-probe decodability increases with
pretraining steps on OLMo-7B across 9 checkpoints (2605.14125 — WebSearch-grounded only, flag
for full-text check if it becomes load-bearing). Latent planning emerges with scale (2604.12493).
Linear-representation formation is tied to pretraining data frequency rather than to a universal
clock (2504.12459).

**Consequence for the plan.** Our object — a fitted linear readout of *content* from pooled
states — belongs to the second family, not the first. The plan should not pre-commit to an early
transition. H1 stays a hypothesis, but the ladder must be designed to resolve a transition
anywhere in the log range, which the existing 15-rung log spacing does well. The honest framing
is that the experiment adjudicates between these two camps for this construct, which is a
stronger motivation than the body currently claims.

## 5. Novelty: the object appears unformalized

**No retrieved paper fits a linear map from a pooled context hidden state to a pooled hidden
state of the model's own generated answer and reports held-out predictivity — and no paper fits
any close relative across a pretraining checkpoint ladder.** Two scouts (A and D) reached this
independently through disjoint channels, both after dedicated probes for exactly that construct.

The nearest misses, each failing on a *named* axis rather than vaguely:

| Work | What it maps | Why it is not our object |
|---|---|---|
| ParaScopes (2511.00180) | residual state → SONAR embedding of the model's own upcoming paragraph; linear | target is an **external encoder's** space, not the model's own answer state; scored by tiered similarity, no R² |
| LRE (2308.09124) | subject-token state → relation output; single linear transformation | a **Jacobian first-order approximation from one prompt**, not a corpus-fitted regression; single-token relational objects; scored by faithfulness (verified §3.3) |
| Future Lens (2311.04897) | hidden state → future hidden states / tokens; linear | **per-position**, not pooled; DV is token accuracy |
| Jump to Conclusions (2303.09435) | layer-ℓ state → final-layer state; linear, held-out | across **layers at one position**; no context→answer axis |
| Tuned Lens (2303.08112) | intermediate state → output distribution; affine | state→**distribution**, the state-to-output-token failure mode |
| Task vectors (2310.15916) | context compressed to one internal vector that yields the answer | no map is **fitted**; causal patching, scored by task accuracy |
| EAGLE (2401.15077) / PHi (2503.13431) | model's own next hidden feature | predictor is a learned network, not linear; DVs are acceptance rate / information gain |
| Wiring Beats Blending (2608.02829) | ridge map between Pythia representations, R² = 0.84 | across model **sizes**, not checkpoints; not context→answer |

What the ladder literature measures instead: probe accuracy per checkpoint (2104.07885,
2402.19465, 2605.14125), steerability per checkpoint (2508.01892), SAE/crosscoder features
aligned across checkpoints (2509.05291, 2505.19440, 2412.17626), circuits per checkpoint
(2606.02378, 2502.11196, 2209.11895), behavioral acquisition curves (2406.11813, 2308.15419,
2304.11158). The closest structural neighbour to "a map, tracked over a ladder" is Crosscoding
Through Time (2509.05291), which aligns feature dictionaries across checkpoints.

**Strength of the claim:** good, with the §2 caveat. Two independent scouts, dedicated probes,
one-hop snowballing, 118 verified citations. Weakened by the S2 keyword channel being dark.
The plan should state the novelty as "we find no prior work that…" with the search bound
attached, not as an unqualified first.

## 6. The finding that should change the ladder

Applying the verified law from 2511.16893 to Olmo-3's actual training configuration:

- Olmo-3 trains at B = 1,024 sequences × C = 4,096 tokens = 4,194,304 tokens per optimizer step
  (the task body's figure, confirmed two ways there).
- `T = e^13.26 = 573,779`.
- `U_PT = T / (B^0.37 · C^0.62) = 573,779 / 2,257 = **254 optimizer steps**`.
- `N_PT = T · B^0.63 · C^0.38 = **1.07B tokens**` (cross-check: 254 × 4.194M = 1.07B ✓).

**The predicted induction-head phase transition lands at ~step 254, and the ladder's first
non-zero rung is step 1,000.** I verified against the Hub that the stage-1 checkpoints are
spaced every 1,000 steps with no exceptions at the bottom — the available steps are 0, 1000,
2000, 3000, 4000, 5000, … So **there is no checkpoint anywhere near the predicted transition,
and none can be added: rung 1 is already ~4× past it.**

Three consequences for the plan:

1. **The induction-head phase change is structurally unresolvable on this model.** It falls
   entirely inside the single gap between rung 0 (random init) and rung 1 (4.19B tokens). The
   ladder can bracket it but never date it. This belongs in the plan as a scope caveat, and it
   should be stated up front rather than discovered in interpretation.
2. **Planner open-decision 2 (rung densification) is partly answered before round 1 runs.**
   Densification *below* step 1,000 is impossible — no checkpoints exist. So the densification
   policy can only ever target the resolvable range, and the plan should say so instead of
   promising a fill-in round wherever the transition lands.
3. **But the ladder is well placed for the transition that probably matters.** Our object is a
   content-level linear readout, and the FV-head analogue — the mechanism 2502.14010 finds
   actually drives few-shot ICL — sits at ~11% of training in Pythia. Transferring that to
   Olmo-3 is ambiguous by ~20×: token-matched (Pythia's 16,000 steps × 2.097M tok/step ≈ 33.5B
   tokens) lands near step 8,000, while fraction-matched (11% of 1,413,814) lands near step
   155,000. **The existing log-spaced rungs at 5,000 / 21,000 / 60,000 / 143,000 / 286,000
   bracket both readings**, which is a real and previously unstated virtue of the log spacing.
   The plan should state this explicitly as the justification for log spacing.

An extrapolation caveat, stated because the number above is a prediction: 2511.16893 fitted its
law over B ∈ [4, 512] and C ∈ [4, 4096]. In the law's own units (B^0.37·C^0.62) their largest
configuration reaches ~1,046 and Olmo-3 sits at 2,257, so this is roughly a 2× extrapolation
beyond the fitted grid — modest, but not interpolation. The law is reported as validated against
GPT-2 and Pythia over 10^7–10^9 tokens.

## 7. Design consequences, ranked

**7.1 Report D(T) per context class, not only pooled (new, cheap, literature-motivated).**
2508.01892's verified finding that *"even closely related concepts exhibit steerability emergence
at distinct stages of training"* implies a single pooled formation curve can smear several
distinct formation events into one smooth-looking ramp. The #1902 corpus already carries class
labels (generic 16,000 / mathcode 1,200 / gsm8k 500 / mbpp 300), and #1902 itself reported which
context classes each stage made more predictable. Per-class D(T) therefore costs **no new
compute** — same cells, different aggregation — and converts a possible artifact into a result.
Recommend the plan adopt it. Note the small-class row counts constrain per-class CI width.

**7.2 A non-monotone D(T) is a documented possibility, not a bug.** Natural Ungrokking
(2606.26050) reports a rule learned mid-run and then lost: 0.94 on held-out probes by step 925,
near zero by step 3,500. The plan should pre-register that a non-monotone curve is interpretable
rather than treating a dip as a pipeline fault.

**7.3 The alignment estimator choice is now grounded, and it favours Procrustes.** In the
sensitivity/specificity battery of 2108.01661, CKA fails sensitivity (misses removal of
functionally important low-variance components) and CCA-family measures fail specificity (fire on
seed noise), while orthogonal Procrustes is the best balanced. Our design's direct / general-linear
/ Procrustes ladder spans the invariance classes (permutation ⊂ orthogonal ⊂ affine; taxonomy in
2305.06329), and Procrustes is the defensible headline. Two scope limits to carry: 2410.24070
found CKA *and* Procrustes both failed to discriminate different training schedules in a dynamics
setting, and a rotation-invariant measure cannot by construction separate "same up to rotation"
from "same basis" (2307.12941) — which is precisely why the direct rung must be reported beside
the aligned ones, as the design already does.

**7.4 Matched nulls are not optional, and the literature says why.** 2407.07059 shows high CKA /
Procrustes / regression scores are achievable by construction on optimized trivial data, so a raw
score is uninterpretable without a calibrated reference. 2202.00095 shows similarity is confounded
by input population structure — directly relevant, since our map is measured on a fixed prompt set
shared across all rungs. The design's matched-capacity shuffled-donor nulls are the right
instrument; the plan should cite these two as the reason rather than as convention.
**Named limitation:** the literature's preferred reference band is the **seed-pair null**
(2108.01661; magnitude evidence in 2506.13234), and Olmo-3 is a single pretraining run, so we
cannot construct one. PolyPythias (2503.09543) is the external evidence that seed variation is
non-trivial. State this as a limitation rather than substituting a different null and calling it
equivalent.

**7.5 Feature standardization is now literature-grounded, and the grounding is a conjunction.**
Transformer hidden states carry rogue / massive dimensions up to ~100,000× typical magnitude
(2402.17762, 2109.04404, 2503.22329) which dominate unstandardized inner products; parameter norms
grow ∝√t across pretraining (2010.09697); and massive activations develop and evolve throughout
Pythia pretraining specifically (2508.03616). Scout C's honest note stands and is worth carrying:
**no single paper states the checkpoint-ladder-specific claim** that norm drift distorts
cross-checkpoint alignment estimates. Our control 7 rests on a conjunction of these, which is a
small positioning point in the plan's favour, not a gap to paper over.

**7.6 The k-shot arm carries a real, published confound.** Exemplar order sensitivity persists at
*all* model sizes and a good permutation for one model does not transfer to another (2104.08786);
format, choice and order can move accuracy from near-chance to near-SOTA via recency and
majority-label bias (2102.09690). Pinning one 4-exemplar set across 15 rungs is therefore exposed
exactly where the design is weakest — the early rungs. The body's "two alternate sets on a subset"
robustness arm is the right instinct but is probably under-powered. Recommend the plan strengthen
it to permutation-averaging over a small fixed set of orders at every rung, and consider the
content-free contextual calibration of 2102.09690 as a cheap per-rung control. Note also
2202.12837 on which exemplar properties actually carry ICL. **No retrieved paper measures
exemplar-order sensitivity as a function of pretraining checkpoint** — an acknowledged gap, so
this risk cannot be quantified from prior work and must be measured or bounded in-run.

**7.7 H3 has independent support.** GPT-3 (2005.14165) established that the few-shot minus
zero-shot gap widens with scale, so a near-zero Δ at the weakest rungs is the expected shape and
not evidence of a broken k-shot arm.

**7.8 Continuous DVs are the right call, for a published reason.** 2304.15004 argues the apparent
sharpness of emergence is often induced by discontinuous metrics. Our primary DV is held-out R²,
already continuous, which insulates the formation curve from that critique — worth one sentence in
the plan, since it pre-empts an obvious reviewer objection. 2403.15796's finding that emergence
tracks a pretraining-loss threshold rather than step count suggests a cheap secondary x-axis
(index rungs by loss where Ai2 publishes it); optional, not required.

## 8. Positioning for the write-up

Closest prior work to cite, by role:
- **Object**: ParaScopes (2511.00180), LRE (2308.09124), Future Lens (2311.04897), Jump to
  Conclusions (2303.09435) — none of which fits our exact construct (§5).
- **Ladder method**: Crosscoding Through Time (2509.05291), Probing Across Time (2104.07885),
  SVCCA learning dynamics (1811.00225), When Do Attention Circuits Form (2606.02378).
- **Timing priors**: 2502.14010, 2511.16893, 2209.11895, 2508.01892, 2504.12459.
- **Estimator + nulls**: 2108.01661, 2407.07059, 2202.00095, 2210.16156, 2110.14739, 2305.06329.
- **Scale drift**: 2402.17762, 2109.04404, 2508.03616, 2010.09697.
- **Suites**: Pythia (2304.01373), OLMo (2402.00838), LLM360 (2312.06550), PolyPythias
  (2503.09543).

## 9. Included / excluded

**Included** (screened in, ≥1 criterion, and used above): the ~60 works cited in §4-§8.

**Excluded, with the failing criterion:** speculative-decoding drafting heads other than EAGLE
(Hydra 2402.05109, SimLens 2507.17618) — E3, state→token. Embedding-space multi-token prediction
probing (2503.17942 — resolved to an unrelated colloidal-physics paper, so the scout's id was
wrong even though it resolves; dropped). Activation Oracles / Predictive Concept Decoders
(2512.15674, 2512.15712, 2606.02609, 2607.23379) — E3, non-linear decoder to text/behavior.
Gist-token compression (2304.08467, 2412.17483, 2402.16058) — E3, map trained into weights, DV is
behavior. Emergence-vs-scale-only work (2206.07682, 2410.01692, 2604.12493) — E1/E2 on the
pretraining-time axis, retained only as context. Grokking (2201.02177) and circuits-competition
(2402.15175) — retained as background for non-monotonicity only.

**One id correction:** Scout D reported 2503.17942 as "embedding-space probing for multi-token
prediction". It resolves to a colloidal-handlebody physics paper. The id is wrong; the intended
work was not recovered and nothing in this review depends on it.
