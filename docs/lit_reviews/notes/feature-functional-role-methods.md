# Characterizing the functional / causal role of features — methods review

**Question.** How has the literature successfully characterized the functional or causal
role of features (SAE features, neurons, directions) — especially the input-detector vs
output-promoter distinction — and which operationalizations were validated? Goal: replace
the `functional_role` judged axis retired at κ 0.318 (#1773 / #1941) with a mechanically
grounded read.

**Search scope.** 10 arXiv MCP calls (6 searches + 4 section reads) and 2 web searches +
2 fetches, 2026-08-02. Every arXiv id below was resolved through the arXiv MCP in-session;
two sources are web-only and are marked as such. Papers found but judged out of scope are
listed at the end.

---

## 0. The headline answer

**Nobody judges causal role from activation text.** Across every method surveyed, a
feature's output/causal role is read from one of exactly three places, none of which is
the set of inputs that make it fire:

1. **The write direction** — project the feature's *decoder / output* weights through the
   unembedding and characterize the resulting vocabulary vector (Gurnee et al. 2401.12181;
   Gur-Arieh et al. 2501.08319 `VocabProj`; Query Lens 2606.07617; J-lens, web-only).
2. **An intervention** — clamp or ablate the feature and measure what the *output
   distribution* does (Gur-Arieh `TokenChange`; causal audit 2607.12166; probe prompting
   2511.07002).
3. **Position in a causal graph** — attribution/circuit methods where "role" is *where the
   feature sits between input and logits*, not a label (Marks et al. 2403.19647; transcoders
   2406.11944; circuit tracing, web-only).

Where an LLM judge *is* in the loop and the resulting labels *do* validate, the judge is
shown the **output of an intervention or a weight projection** — never max-activating
examples alone. Gur-Arieh et al. measured this directly: descriptions built from activating
inputs "fail to capture the causal effect of the feature on outputs," and output-centric
descriptions beat them on output evaluations.

This retro-explains #1941 exactly. Adding contrastive *input-side* evidence (non-activating
examples, nearest neighbours) moved κ by ≤ +0.04 because the axis asks an output-causality
question and every piece of evidence offered was input-side. CRaFT (2604.01604) states the
same failure mode independently: "activation strength alone often captures superficial
heuristics such as topic or lexical cues, rather than the true causal mechanisms."

**Project-side corroboration (new finding from this review).** #1773's own non-judge
mechanical validator already shows the judged axis had no mechanical footing.
`eval_results/issue_1773/validation/mechanical_validators.json`:

| judged label | n | mean `side_ratio` |
|---|---|---|
| `input_side` | 12,574 | 0.7334 |
| `output_promoting` | 888 | 0.7389 |
| `mixed` | 1,415 | 0.7411 |

The three classes are indistinguishable (spread 0.008 on a [0,1] statistic). And they
*should* be: `side_ratio = cnt / (cnt + psi_cnt)` is a **firing-location** statistic
(answer-side vs context-side activity), so by construction it cannot measure output
causality. The axis was validated against a quantity from the wrong family. Any replacement
must be read off the **decoder column**, not off where the feature fires.

---

## 1. Methods table

Cost column is for our setting: Qwen-2.5-7B-Instruct, layer 19, 131,072 BatchTopK features,
d = 3,584, |V| ≈ 152k.

| # | Method | Construct | Operationalization | Validation reported | Cost | Our banked implementation |
|---|---|---|---|---|---|---|
| M1 | **Logit-attribution moments** (Gurnee 2401.12181) | prediction vs suppression vs partition vs null "action" role | kurtosis + skew + variance of the vocab vector `W_U w_out`; high kurtosis + **positive** skew = prediction, high kurtosis + **negative** skew = suppression, high variance = partition | Consistent layer-depth pattern across 5 GPT-2-medium seeds + 5 Pythia (410M–6.9B); per-neuron causal interventions for the entropy-neuron subfamily | **~free** | `logit_footprint()` **already computes the exact vector** `E = W_U @ (γ ⊙ W_dec)`; only top-10 + `concentration` are persisted. New compute = 4 extra scalars in the same GEMM pass |
| M2 | **VocabProj** (Gur-Arieh 2501.08319) | what the feature promotes/suppresses in vocab | `w = W_U · LayerNorm(v)`, read top/bottom tokens | Steering evaluation: output-centric descriptions beat input-centric on output evals; ensemble of both best overall; human eval | ~free | Identical to our `logit_footprint` up to the LayerNorm convention (we use `γ ⊙`, no centering/rescaling) |
| M3 | **TokenChange** (Gur-Arieh 2501.08319) | causal output effect | clamp feature to activation `m` over k random prompts, take mean per-token logit delta vs baseline | same steering evaluation; the *causal* sibling of M2 | moderate (k prompts × features) | Needs the steering rig (`issue1774_steering.py`, `issue1773_register_steer_*.py`) — sample-scale only |
| M4 | **Query Lens** (2606.07617) | joint input-read + output-write signature | encoder-side "key" features (what activates it) + decoder-side "value" features (what it promotes), **plus module-mediated indirect effects** beyond logit-lens direct effect | coherent token signatures recovered for features uninterpretable under Logit Lens; proposes Subspace Channel Hypothesis (downstream modules read via layer-specific subspaces) | low–moderate | Encoder + decoder weights banked; the indirect-effect term is new compute |
| M5 | **J-lens / Jacobian lens** (web-only, Anthropic 2026) | what a mid-layer direction is disposed to make the model say | `lens_l(h) = unembed(J_l · h)`, `J_l = E[∂h_final/∂h_l]` (expectation over prompts, source positions, and current-and-future target positions) | Causal methodology in the companion paper; open code + interactive demo | low **if** our J is layer→final | `J_19` from #1776 — **needs verification**: our estimator is `source_layer → readout_layer` with a split-half report, not necessarily readout-at-final |
| M6 | **Read- vs write-inertness dissociation** (2607.12166) | is the feature causally live on-distribution, and in which direction | subject every feature to **both** ablation and steering; features split into read-inert (never fires when concept present) vs write-inert (steerable but unmonitorable) | Found up to 77% of cosine-≥0.90 "recovered" features causally inert in a degraded SAE, 9% in a good one; five antipodal pairs dissociate read/write completely (steering specificity 143–310 at zero ablation effect) | high per feature | Steering rig banked; ablation harness is new. Sample-scale calibration instrument |
| M7 | **Sparse feature circuits** (2403.19647) | role = position in a causal subgraph | linear-approximation attribution of each feature node to a behavioral metric, then thresholded subgraph; nodes classified by graph position | SHIFT downstream task (ablating human-judged task-irrelevant features improves classifier generalization); thousands of unsupervised circuits | high | Not banked; would be a new subsystem |
| M8 | **Transcoder input-invariant / input-dependent factorization** (2406.11944) | separates *what makes it fire* from *what it writes*, in weights | transcoder circuit factorizes into an **input-invariant** term (weights-only, feature→feature) and an **input-dependent** term | On par with SAEs on sparsity/faithfulness/interpretability at 120M–1.4B; used to reverse-engineer GPT-2 greater-than circuit | high (needs transcoders) | Not banked (we have SAEs, not transcoders) |
| M9 | **Attribution graphs / circuit tracing** (web-only, Ameisen & Lindsey et al. 2025) | feature→logit pathway in a local replacement model | replacement model linearizes residual stream through CLT/SAE features with attention frozen; edges = attributions | Extensive case studies on Claude 3.5 Haiku; open `circuit-tracer` | high | Not banked |
| M10 | **Probe prompting / CPAS** (2511.07002) | concept-aligned supernode labels **validated causally** | group attribution-graph features into supernodes via responses to concept-targeted probe prompts; validate labels against entity-swap interventions | 45,596 entity-swap interventions across 4 factual domains on Gemma-2-2B; labeled supernodes had the predicted steering behavior in **every** domain | moderate–high | Not banked; the *validation protocol* is portable |
| M11 | **RelP** (2508.21258) | faithful cheap attribution | replaces attribution-patching gradients with LRP propagation coefficients; 2 forward + 1 backward pass | Pearson vs activation patching 0.956 (RelP) vs 0.006 (attribution patching) for GPT-2-Large MLP outputs on IOI | moderate | Not banked; relevant if we ever need cheap faithful attribution |
| M12 | **Steering-success as an axis validator** (2603.04198) | does the text description predict controllability | measure steering success rate per feature; correlate with auto-interp score | L2 weight regularization roughly doubled steering success and made auto-interp scores *better predictors* of controllability | low (given a rig) | Directly reusable as our validation metric |
| M13 | **SHIFT / TPP** (2411.18895) | SAE feature quality via downstream causal utility | automate SHIFT with an LLM annotator; TPP measures disentanglement of similar concepts | Differentiates SAE hyperparameters/architectures across open models | moderate | Not banked; a quality metric, not a per-feature role label |

**Encoder-vs-decoder geometry caveat (2605.24577).** Decoder-column cosine matches across
independently trained seeds at 98% while an SAE trained on one seed reconstructs another at
*negative* explained variance — "the decoder columns align; the encoder reads from a rotated
frame." Encoder and decoder therefore carry genuinely different information, which is what
makes an encoder-vs-decoder asymmetry statistic meaningful rather than redundant. (Single
author, pre-review; treat the numbers as suggestive.)

---

## 2. Per-paper detail on the three most applicable

### 2401.12181 — Universal Neurons in GPT2 Language Models (Gurnee, Horsley, Guo, Kheirkhah, Sun, Hathaway, Nanda, Bertsimas)

The single most transferable method. §"Universal Functional Roles of Neurons" is explicit
that it switches from *activations* to *weights* precisely because role is a downstream
property — it calls these "action mechanisms … analogous to motor neurons."

Operationalization, verbatim in substance: approximate a neuron's effect on final logits as
`W_U w_out`, then classify by the **moments of the distribution of vocabulary effects**:

- **prediction neuron** — high kurtosis, **positive** skew (raises a coherent token set,
  leaves the rest ~unchanged)
- **suppression neuron** — high kurtosis, **negative** skew
- **partition neuron** — high variance in overall logit effect (splits the vocabulary)
- **entropy neuron** — high weight norm but *lowest variance* of logit effect, i.e. output
  weights near-orthogonal to `W_U`; validated by fixing the activation and observing the
  final-LayerNorm scale (and hence entropy) move while token ranking is preserved

They compute the same moments on `cos(W_U, w_out)` for all neurons in five GPT2-medium
models and find a seed-consistent depth profile: prediction neurons become prevalent after
about halfway through the model, then a sharp shift to suppression neurons at the very end.
Replicated on five Pythia models (410M–6.9B) to rule out a tied-embedding artifact.

Validation evidence: cross-seed and cross-model-family consistency of the depth profile
(structural), plus direct causal intervention for the entropy-neuron subfamily. Note this is
*not* a per-neuron intervention validation of the prediction/suppression labels themselves —
that gap is what M3/M6 fill.

### 2501.08319 — Enhancing Automated Interpretability with Output-Centric Feature Descriptions (Gur-Arieh, Mayan, Agassy, Geiger, Geva)

The paper that directly answers "has anyone gotten causal-role labelling to work from
evidence given to a judge?" Answer: yes, by changing the *evidence*, not the rubric.

Three methods, all feeding an explainer LLM:

- **MaxAct** — the standard input-centric pipeline (what #1773 used).
- **VocabProj** — `w = W_U · LayerNorm(v)`; feed the top- and bottom-scoring tokens to the
  explainer. *Correlative.*
- **TokenChange** — pass k random prompts, clamp the feature to activation `m`, take the
  mean per-token logit change, feed the most-affected tokens to the explainer. *Causally
  intervenes.*
- **Ensembles** — concatenating the raw evidence (`MaxAct + VocabProj`) performed best on
  both input and output evaluations; LLM-summarized ensembles performed *worse* across the
  board.

Headline result: current (input-centric) pipelines produce descriptions that "fail to
capture the causal effect of the feature on outputs," measured by steering evaluations.
Output-centric descriptions capture it better; the combination is best. Bonus finding:
output-centric descriptions recover inputs that activate features previously believed dead.

The validation instrument here — a **steering evaluation** scoring whether a description
predicts the feature's causal effect — is directly reusable as the non-judge reference our
replacement axis needs.

### 2606.07617 — Query Lens: Interpreting Sparse Key-Value Features with Indirect Effects (Lee, Bang, Hwang, Lim, Kim)

The closest paper to the exact question, and the only one that treats input-side and
output-side as *two halves of one feature signature* by construction: it "jointly consider[s]
encoder-side key features and decoder-side value features" to "identify both the inputs that
activate a feature and the outputs it promotes."

Its contribution over plain Logit Lens is accounting for **indirect, module-mediated
effects** — what happens to the feature's write once downstream modules process it — rather
than only the direct unembedding effect. Reported result: coherent token signatures for
features that are uninterpretable under Logit Lens. Proposes the *Subspace Channel
Hypothesis* (downstream modules read features through layer-specific subspaces).

Why this matters for us specifically: we read at **layer 19 of 28**. A direct
`W_U @ W_dec` projection assumes the decoder column goes straight to logits, which is
weakest exactly at mid-depth. Query Lens and J-lens are the two named fixes.

---

## 3. What our banked quantities already implement

| Literature method | Our artifact | Status |
|---|---|---|
| M1/M2 vocabulary projection | `logit_footprint()` in `scripts/issue1773_phase0_mechanical.py:256` — computes `E = W_U @ (γ ⊙ W_dec[:, feats])` in fp32 blocks | **Already computed**; we persist only top-10 promoted/suppressed + `concentration` (share of positive-logit mass in top-10). The moments are discarded |
| M1 partition/entropy discriminator | `concentration` is a *weak* proxy for kurtosis; decoder column norm is available | Partial |
| M5 J-lens | `J_19` from #1776 (`scripts/issue1776_jacobian.py`), with `HalfSumAccumulator` + `splithalf_report` reliability machinery | **Verify first**: the estimator is parameterized `source_layer → readout_layer`; J-lens needs readout at the final layer before unembedding |
| M3/M6/M12 steering | `issue1774_steering.py`, `issue1773_register_steer_{pilot,stats,validator}.py` | Rig exists; ablation arm is new |
| Encoder-side reads | encoder weights banked | Unused for role |
| where-it-fires | `side_ratio` | Banked — but **input-side by construction**; keep as a second axis, never as the output-role measure |
| map error target | per-feature R² from the context→answer map | The downstream consumer this axis exists to serve |

---

## 4. Recommended replacement axis — ranked

The reframing the literature forces: **stop asking for one 3-way categorical label and
produce 2–3 continuous mechanical scalars.** Role is not one dimension. Every surveyed
method that works treats "where it reads" and "what it writes" as separate measurements
(Query Lens explicitly; 2607.12166's read/write-inertness dissociation empirically). Our
downstream need is served better by scalars anyway: the map-predictability question ("is
this feature mechanically easy for a context→answer map to predict?") wants a regressor, not
a class.

### R1 (recommended) — `output_footprint_moments`: skew / kurtosis / variance of the logit footprint

**Operationalization.** For each feature f, let `e_f = W_U (γ ⊙ W_dec[:, f]) ∈ R^{|V|}`
(already computed). Persist:

- `fp_skew(f)`, `fp_kurtosis(f)`, `fp_var(f)`, `‖W_dec[:, f]‖`
- derived class, following Gurnee: high kurtosis ∧ skew > 0 → **promoting**; high kurtosis ∧
  skew < 0 → **suppressing**; high variance ∧ low kurtosis → **partition**; low variance ∧
  low kurtosis ∧ high norm → **entropy/no-op-on-logits** (the residual "not output-side"
  bucket, which is the honest home for what the judge was calling `input_side`)
- thresholds set by the empirical within-layer distribution, not literature constants (the
  literature's absolute values are GPT-2/Pythia-specific; Gurnee uses kurtosis > 10)

**Why first.** Zero judge, therefore κ = 1 by construction — it removes the reliability
problem rather than mitigating it. It reuses a GEMM we already run. It is the
best-validated method in the survey (5 GPT-2 seeds + 5 Pythia models).

**New compute.** Re-run the phase-0 footprint block with 4 accumulators added. Same pass
shape as the existing `FOOTPRINT_CHUNK` loop over 131,072 features — hours at most on one
device, no new data, no judge spend.

**Validation recipe.**
1. *Steering (primary, non-judge).* On a stratified sample (~300–500 features across the
   moment quadrants), clamp each feature and measure whether its top-`fp` promoted tokens
   actually gain logit mass — i.e. correlate `VocabProj` against `TokenChange` (2501.08319).
   Report Spearman ρ per class. A promoting-classified feature whose steering delta is null
   is a write-inert false positive (2607.12166).
2. *Ablation (specificity).* Zero the feature where it fires; a genuine output-promoter
   should move the output distribution, an input-side one should not. The
   ablation-vs-steering 2×2 is the read/write dissociation of 2607.12166.
3. *Cross-layer structure (cheap sanity).* Gurnee's depth profile predicts promoting-class
   prevalence rises with depth and suppression spikes near the end. We have one layer, so
   run the footprint moments at one earlier and one later layer and check the ordering. A
   flat profile would falsify the read.
4. *Downstream utility (the actual point).* Regress per-feature context→answer map R² on
   the moments. The project hypothesis — input-echo features are mechanically easy to
   predict, output-promoting ones are genuine answer content — predicts a negative
   relationship between promoting-class membership and R². This is the pre-registerable
   prediction, and unlike the judged axis it is falsifiable.

**Known limitation to state up front.** Direct logit attribution from layer 19 of 28 ignores
9 layers of downstream processing. Treat R1 as the *direct-effect* read and R2 as its
indirect-effect correction; if they disagree, R2 wins and R1's disagreement rate is itself a
reportable quantity.

### R2 — `jlens_footprint_moments`: the same moments through the averaged Jacobian

**Operationalization.** `e_f^J = W_U (γ ⊙ (J_19 · W_dec[:, f]))`, with
`J_19 = E[∂h_final/∂h_19]`, then identical moments. This is M5 applied per decoder column,
and it is the principled fix for R1's mid-depth weakness — it routes the write through the
model's own averaged downstream computation rather than assuming layer 19 writes straight to
logits.

**New compute.** *Conditional on a verification step:* confirm whether #1776's `J_19` has its
readout at the final layer. If yes, this is one `3584 × 3584 @ 3584 × 131072` matmul then the
existing footprint GEMM — nearly free, and the split-half reliability report already exists
(`splithalf_report` in `issue1776_jacobian.py`). If the readout layer is intermediate,
re-estimating is a moderate GPU job on already-built machinery.

**Validation.** Everything in R1, plus the J-lens-specific check: J-lens should beat the
plain logit lens at predicting the model's actual next-token behavior, so compare
`fp` vs `fp^J` top-token overlap against measured `TokenChange` on the same sample. Report
the split-half reliability of `J_19` alongside (already implemented).

**Rank rationale.** Strictly better construct than R1, but gated on a verification we have
not done and on a source whose primary write-up is web-only. Ship R1 first; R2 as the
follow-up that either corroborates or corrects it.

### R3 — `causal_role_2d`: ablation × steering dissociation on a stratified sample

**Operationalization.** Per feature, two independent causal reads (2607.12166):
`ablation_effect` (zero it where it fires → Δ on the model's own output distribution) and
`steering_effect` (clamp it on where it does not fire → Δ output). The 2×2:

| | steering ≈ 0 | steering > 0 |
|---|---|---|
| **ablation ≈ 0** | inert | write-capable, unused ("write-inert" in their terms) |
| **ablation > 0** | read-only / monitored | load-bearing output-promoter |

**Why not first.** Per-feature forward passes do not scale to 131,072 features. Its correct
role is as the **ground-truth calibration set for R1/R2**, not as the production axis — same
posture as #1773's mechanical validators, but with a quantity that can actually separate the
classes.

**New compute.** Ablation harness (new) + existing steering rig, at sample scale (hundreds to
low thousands of features). Budget as a calibration round, not a full-dictionary pass.

### Explicitly not recommended

- **Re-running a judged `functional_role` axis with better text evidence.** #1941 already
  falsified this at ≤ +0.04 κ, and the literature predicts that result: no amount of
  input-side text answers an output-side question.
- **A judged axis over *output-centric* evidence (VocabProj tokens / TokenChange deltas).**
  This is the one judged variant the literature supports (2501.08319, 2511.07002), and it
  would probably work — but it is strictly dominated here, because for *our* three-way
  distinction the numeric moments **are** the answer. Asking a judge to read
  "top promoted tokens: ` the`, ` a`, ` an`" and infer "promoting" adds cost, adds
  variance, and re-imports κ. Keep the judge for the semantic axes where the label is not a
  function of a number.
- **Attribution graphs / sparse feature circuits (M7–M9, M11).** Correct constructs, but each
  is a new subsystem and none is needed to answer *this* question.

---

## 5. Sources

MCP-resolved arXiv:

- 2401.12181 — Universal Neurons in GPT2 Language Models (Gurnee et al., 2024-01-22)
- 2403.19647 — Sparse Feature Circuits (Marks, Rager, Michaud, Belinkov, Bau, Mueller, 2024-03-28)
- 2406.11944 — Transcoders Find Interpretable LLM Feature Circuits (Dunefsky, Chlenski, Nanda, 2024-06-17)
- 2411.18895 — Evaluating Sparse Autoencoders on Targeted Concept Erasure Tasks (Karvonen, Rager, Marks, Nanda, 2024-11-28)
- 2501.08319 — Enhancing Automated Interpretability with Output-Centric Feature Descriptions (Gur-Arieh, Mayan, Agassy, Geiger, Geva, 2025-01-14)
- 2504.13756 — Scaling sparse feature circuit finding for in-context learning (Kharlapenko, Shabalin, Barez, Conmy, Nanda, 2025-04-18)
- 2508.21258 — RelP: Faithful and Efficient Circuit Discovery via Relevance Patching (Rezaei Jafari, Eberle, Khakzar, Nanda, 2025-08-28)
- 2511.07002 — Automated Attribution Graph Interpretation via Probe Prompting (Birardi, Paulo, 2025-11-10)
- 2603.04198 — Stable and Steerable Sparse Autoencoders with Weight Regularization (Jedryszek, Crook, 2026-03-04)
- 2604.01604 — CRaFT: Circuit-Guided Refusal Feature Selection via Cross-Layer Transcoders (Kim, Jin, Lee, Han, 2026-04-02)
- 2605.24577 — Polymorphism Is Rotation (McCann, 2026-05-23) — *single author, pre-review*
- 2606.07617 — Query Lens: Interpreting Sparse Key-Value Features with Indirect Effects (Lee, Bang, Hwang, Lim, Kim, 2026-05-30)
- 2607.12166 — From Geometric Recovery to Causal Validation (Bal, 2026-07-13) — *single author, pre-review*

Web-only (no arXiv id; not MCP-resolvable):

- **Jacobian lens / J-space.** "Verbalizable Representations Form a Global Workspace in
  Language Models," Transformer Circuits Thread, 2026-07-06.
  <https://transformer-circuits.pub/2026/workspace/index.html>. Formula taken from the
  official repo README: `lens_l(h) = unembed(J_l @ h)`, `J_l = E[∂h_final/∂h_l]`, expectation
  over prompts, source positions, and all current-and-future target positions.
  Code: <https://github.com/anthropics/jacobian-lens>. Demo: <https://neuronpedia.org/jlens>.
  *The README documents no dedicated per-direction output-causality statistic.*
- **Circuit tracing / attribution graphs.** Ameisen, Lindsey et al., Transformer Circuits
  Thread, 2025. <https://transformer-circuits.pub/2025/attribution-graphs/methods.html>.
  Code: <https://github.com/safety-research/circuit-tracer>.

Found and set aside as out of scope: 2601.12879 (HAGD hierarchical circuit extraction),
2602.20904 (transcoder adapters for reasoning diffing), 2510.09312 (CRV, CoT verification via
attribution graphs), 2607.14791 (transcoders for deception), 2602.20330 / 2606.15796
(multimodal circuit tracing), 2603.21014 (CLT-Forge), 2509.14723 (single-cell transcoders).

---

*Written 2026-08-02 for the #1773 / #1941 `functional_role` replacement question.*
