# RQ3 — What evidence should be shown to an explainer model, and how should it be sampled?

Literature notes for the auto-interpretability review (EPS). Scope: evidence
*selection and formatting* for SAE-feature explanation generation, with measured
effect sizes wherever the literature supplies them.

**Motivating context.** The EPS dictionary's public explanations came from
Neuronpedia's `np_max-act-logits` (gemini-2.0-flash), which shows max-activating
*tokens* plus logits and **prompt-enforces conciseness**. Median explanation is one
word ("uso" for a Spanish/Portuguese assistant-answer feature). Sections 1, 6 and 9
below identify this exact failure mode and its causes.

---

## 0. The one-paragraph answer

Explainer *model* scale is past saturation and evidence *design* is not. Paulo et
al. measured Claude 3.5 Sonnet ≈ Llama-3.1-70B ≈ a human annotator on fuzzing
(0.75 / 0.76 / 0.75) and detection (0.75 / 0.74 / 0.74), against a random-explanation
floor of 0.51 — so a Sonnet-4.5-class explainer buys nothing further from model
capability alone (2410.13928, Table `explainer_size`). Every remaining lever is in
the evidence packet, and the largest single measured lever is **adding output-side
evidence**: input-only (max-activating) descriptions score 49.2/100 on an
output-faithfulness eval where a logit-lens description scores 56.5, and ensembling
input + output evidence reaches 64.9 *while also* raising the input-side score from
56.6 to 66.6 (2501.08319, Gemma 2 residual SAE). Sampling strategy is a real but
much smaller lever (≈0.04 fuzzing points, ≈15% of the above-chance signal), and
several widely-assumed levers — context length, per-token activation values, chain
of thought — are measurably ≈zero.

---

## 1. Evidence-design table

Scores below are **median (IQR)** unless stated. Paulo et al. (2410.13928) numbers
are Gemma 2 9B, 131k-latent residual SAE, Llama-3.1-70B-instruct (4-bit) as both
explainer and scorer, 500+ features, each scored on 100 activating examples
(stratified, 10 per decile) + 100 non-activating examples. Their two floors:
**random explanation = 0.51 fuzz / 0.51 detect**; **randomly-initialised TopK SAE =
0.55 / 0.54**. Read all deltas against the 0.51 floor, not against zero.

| # | Design choice | Measured effect | Cost | Pitfalls | Refs |
|---|---|---|---|---|---|
| 1 | **Top-k only vs quantile/stratified vs uniform sampling** | Quantile 0.77 fuzz / 0.74 detect; uniform-random 0.76 / 0.74; **top-only 0.73 / 0.72**. Above the 0.51 floor that is 0.26 vs 0.22 → top-only discards ≈15% of the above-chance signal. Direction reverses on the embedding score (top-only 0.70 > quantile 0.68). | Identical — same example count, only the sampler changes. Free. | Top-only explanations have **higher specificity, lower sensitivity**; they win on separating non-activating text but degrade sharply on lower-activation deciles. Pure stratified sampling over-corrects into explanations that are "too broad" to be meaningful. | 2410.13928 §Results, Tab. `source_examples`, Fig. `accuracy` (c,d); [EleutherAI blog](https://blog.eleuther.ai/autointerp/) |
| 2 | **Scoring only on top examples** | The entire sampling effect above **disappears** if scoring is restricted to max-activating contexts — top-sampled explanations then look best. | — | This is the single most consequential evaluation bug in the field: "current auto-interpretability evaluations … produce interpretations using top activating examples and evaluate them on a small subset of the activation distribution." Any A/B you run must score on a stratified activating pool **plus** non-activating contexts. | 2410.13928 §Results |
| 3 | **Number of examples** | 10 → 0.73/0.71; 20 → 0.74/0.72; **40 → 0.76/0.74**; 60 → 0.75/0.73. Saturates at ~40; 10→40 buys +0.03 fuzz. | Linear in tokens. 40 examples × 32 tokens ≈ **\$200 per 1M features** with a Llama-70B-class explainer. | Saturation is explainer-dependent ("at least with the explainer model we used"); untested for a Sonnet-4.5-class model with a much larger window. Going past 40 is measurably *not* free-lunch — 60 is slightly worse. | 2410.13928 Tab. `number_examples` |
| 4 | **Context window length per example** | 16 tok → 0.75/0.74; **32 tok → 0.76/0.74**; 64 tok → 0.74/0.70. Statistically indistinguishable; 64 mildly worse. Corpus/window also determines *coverage*: at 256-token collection contexts, 30% of Gemma-131k features fire <200× in 10M tokens and 15% never fire; at the 1024-token training context only 5% never fire. | Linear. Longer contexts force fewer examples per prompt. | "Long-range" features — those whose trigger spans more than the window — are silently mis-explained, and the authors explicitly punt on them. Window length trades against example count at fixed budget. Full-document evidence is **untested** in this literature. | 2410.13928 Tab. `context`, §Collecting activations |
| 5 | **Per-token activation values shown (highlighting + magnitudes)** | Activations in prompt 0.76/0.74 vs no activations 0.75/0.73. **+0.01.** EleutherAI: "does not significantly improve scores." | Near-free (a few tokens per example). | Do not expect gains; do not spend design effort here. Keep it because it is nearly free and it is what the validated pipelines actually ran, not because it is measured to help. Note the *scorers* deliberately hide the highlighting from the model in detection scoring. | 2410.13928 Tab. `cot`; EleutherAI blog |
| 6 | **Output-side evidence (top promoted/suppressed logits)** | Gemma 2 residual SAE, input-eval / output-eval (0–100, output chance = 33.3): MaxAct **56.6 / 49.2**; VocabProj (logit-lens, top t=50) 50.1 / **56.5**; TokenChange (top t=20 after clamping, k=32 prompts) 44.7 / 54.9. Input- and output-centric methods beat each other on their own axis "by large margins of up to 15–30 points". | **≤2 inference passes** vs a full corpus scan for MaxAct — dramatically cheaper than input-side evidence. | Output-centric alone *loses* on input-eval (down to 18.2 on Llama residual SAE). Output evidence degrades in early layers (logit lens is weak there) and is much weaker for MLP features (45–50) than residual features (~66). | 2501.08319 Tab. `mean_results`, §Results |
| 7 | **Ensembling input + output evidence** | Gemma residual SAE: Ensemble(all, raw) **66.6 input / 64.9 output** vs best single method 56.6 / 56.5 — i.e. +10 input and +8 output simultaneously. Paper states 6–10 point gains over the next-best single method. Gemma MLP: 55.7 / 48.7 vs MaxAct 50.4 / 35.1. Llama residual: 36.0 / 71.2 vs MaxAct 30.3 / 71.8. | Additive: one corpus scan + ≤2 passes + one longer explainer prompt. | **Format matters and interacts with the eval axis**: concatenating *raw* evidence is better on input-eval, concatenating *generated descriptions* is consistently better on output-eval. Pick per intended downstream use (retrieval vs steering). | 2501.08319 Tab. `mean_results`, §Description Format Affects Performance |
| 8 | **Intervention / steering-based evidence** | Intervention scoring correlates *slightly negatively* with fuzzing — features that score badly on context-based explanation are disproportionately the ones explainable by downstream effect. Output-centric descriptions also recover activating inputs for features previously believed **dead**. In vision, a steering-based explainer plus "Steering-informed Top-k" reaches SOTA explanation quality "without additional computational cost". | Requires generation under clamping (Gur-Arieh: 3 prompts × 4 clamp strengths × 25 tokens = 12 generations/feature). | Intervention scores are only comparable **at fixed intervention strength** (defined as mean KL from the clean logit distribution) — strong-enough steering is trivially "interpretable". | 2410.13928 §Intervention scoring; 2501.08319; 2603.22593 |
| 9 | **Prompt format / conciseness instructions** | Neuronpedia's `np_max-act-logits` "asks the model to be concise and provides explicit examples of 'padding' phrasing to not use", plus postprocessing — and works from **token lists rather than full contexts**. Neuronpedia's own caveat: it "may perform worse at finding more subtle patterns that occur over longer texts." Gur-Arieh: longer/informative descriptions favour input-eval, concise ones favour output-eval. | Free. | **This is the direct cause of the EPS one-word explanations.** No quantitative comparison of `np_max-act-logits` vs `oai_token-act-pair` was ever published — the blog offers three qualitative examples only. Conciseness was adopted on qualitative evidence and is silently trading away input-side sensitivity. | [Neuronpedia blog](https://www.neuronpedia.org/blog/circuit-tracer); 2501.08319 |
| 10 | **Chain-of-thought in the explainer** | COT 0.76 fuzz / 0.73 detect / 0.65 embed vs no-COT 0.76 / 0.74 / 0.68. **Zero or negative.** | Large — "significantly increases the compute and time"; the authors dropped it from their main runs. | Don't. | 2410.13928 Tab. `cot` |
| 11 | **Evidence corpus: model's own distribution vs external** | Quality is nearly invariant: RPJv2 (SAE's training mix) 0.76/0.74/0.67 vs Pile 0.76/**0.76**/0.69. **Coverage is not**: features firing <200× drops 30%→15% and never-firing 15%→1% when moving to the closer-to-training corpus. | Corpus collection cost only. | Corpus choice barely changes the explanation of a feature you can *see*, but changes enormously *which* features you see. Bolukbasi et al. trace the BERT interpretability illusion partly to "common text corpora represent[ing] only narrow slices of possible English" and recommend testing hypotheses on **multiple** datasets. **For EPS this is decisive**: persona/assistant-answer features live on the chat distribution and will be under-sampled or read as dead on a pretraining corpus. | 2410.13928 Tab. `dataset`, §Collecting activations; 2104.07143 |
| 12 | **Negative / near-miss contrastive evidence at *generation* time** | **Not directly measured** in the main pipelines — non-activating contexts are used in *scoring* (detection, fuzzing, surprisal, embedding all frame the explanation as a binary classifier), not in generation. Closest evidence: SAEExplainer retains "plausible-sounding but non-activating explanations" as hard negatives for DPO and reports gains. | Doubles the evidence block if included. | The SAEExplainer effect sizes I saw (Generative Accuracy +26%, input +20.33, output +5 after round 1) come from a **web-search summary, not the paper body** — treat as unverified. This is the clearest open gap and a cheap experiment for EPS. | 2410.13928 §Scoring; 2606.08496 (abstract verified only) |
| 13 | **Feature neighbours / correlated features as evidence** | **Not measured** for explanation quality. Strong indirect motivation: across 722 human-annotated features (Gemma 2 2B, Pythia 70M) the mean annotation string is reused across **3.07 features**; **82.1%** of features share an annotation with ≥1 other; "plural nouns" labels **101 distinct features** across 18 layers and 4 components; the average annotation resolves only **70%** of feature identity. Detection-style scoring is provably **invariant** to this collision. | Requires a neighbour index (decoder cosine) + longer prompt. | Because standard scoring cannot see collision, adding neighbour evidence will show **no gain on detection/fuzzing** even if it works — you must add a discrimination / collision-adjusted metric to measure it at all. | 2605.12874 |
| 14 | **Explainer model scale** | Fuzz / detect / embed / simulation: Claude 3.5 Sonnet 0.75/0.75/0.70/0.30; Llama-70B 0.76/0.74/0.67/0.29; Llama-8B 0.70/0.70/0.64/0.26; **Human 0.75/0.74/0.71/0.36**. Saturates by ~70B on the cheap scores; the human–model gap survives only on simulation. | 8B→70B is ~9× compute for +0.06; 70B→Sonnet is ~0. | The measured ceiling is in the *evidence*, not the explainer. Caveat: prompts were not re-optimised per model, so the Claude number may be pessimistic. | 2410.13928 Tab. `explainer_size` |
| 15 | **Agentic / tool-using evidence gathering** | MAIA composes tools (synthesise and edit inputs, compute maximally-activating exemplars, summarise) and produces vision-neuron descriptions "comparable to those generated by expert human experimenters" on a synthetic-neuron benchmark with paired ground truth. | Highest of any option — many model calls per feature. | Vision-only evaluation; no per-feature cost figures suitable for a 100k-feature dictionary. | 2404.14394 |

### Scoring-side costs (needed to price any evidence A/B)

Per 100k features, 100 contexts each (2410.13928 Tab. `cost`):

| Scorer | Llama-70B | Claude Sonnet 3.5 |
|---|---|---|
| Detection | \$588 | \$5.5k |
| Fuzzing | \$676 | \$6.2k |
| Simulation (all-at-once) | \$3.6k | \$31.5k |
| Simulation (token-by-token) | \$18.7k | \$219.1k |
| Embedding (400M encoder) | ≈\$50 | — |

Spearman correlation with human scores (700 contexts, 81 features): **fuzzing 0.69 >
simulation 0.60 > detection 0.59 > surprisal 0.34 > embedding 0.32**. Simulation-AAO
needs prompt-token logprobs, which closed-source APIs don't expose — hence the 30×
premium for token-by-token. Fuzzing+detection at ~1/5 the cost of simulation and
*better* human correlation is the practical default.

---

## 2. The interpretability illusion and top-activation bias

Five distinct failure modes, often conflated:

1. **Narrow-slice illusion (corpus).** Individual BERT neurons and linear directions
   appear to encode one clean concept because the corpus only samples a narrow slice
   of the input space; the same direction reads as a different concept on a different
   dataset. Methodological prescription: test on multiple datasets. (2104.07143)

2. **Recall-without-precision illusion (explanation breadth).** "Explanations are
   overly broad, and thus have good recall but poor precision. For example, [Bills et
   al.] find a feature activating at the end of the phrase 'don't stop' or 'can't
   stop', but an explanation activating on all instances of 'stop' achieves a high
   interpretability score." And critically: **"As we scale autoencoders and the
   features get sparser and more specific, this kind of failure becomes more
   severe."** Same paper concedes "a large fraction of the random activations of
   features we find, especially in GPT-4, are not yet adequately monosemantic."
   (2406.04093)

3. **Top-activation bias proper.** Extremal-activation labelling "fails to capture
   valuable information about the behaviour of a representation"; mid-range activations
   carry statistical associations and confounding concepts that maximal activations
   hide, and mid-range samples can be used to *locate* those confounds. (2411.10019).
   Paulo et al.'s Fig. `accuracy` (c,d) is the quantitative version: top-quantile
   explanations separate non-activating text best but lose accuracy monotonically down
   the activation deciles.

4. **Metric illusion.** SAEs trained on **randomly initialised** transformers produce
   auto-interp scores and reconstruction metrics "similar to those from trained
   models" in many settings — a high aggregate auto-interp score does not certify that
   real computational features were recovered (2501.17727). Independently, many common
   explanation-evaluation metrics **fail sanity checks: their score does not change
   after massive changes to the concept labels** (2506.05774). Paulo et al.'s own
   random-init TopK SAE control (0.55/0.54 vs 0.51 random-explanation floor vs 0.76
   trained) shows the gap *can* be large — the two results are in tension and the
   honest reading is "run the control yourself, per dictionary."

5. **Descriptive collision.** An explanation can be individually accurate and still
   fail to identify the feature, because 82.1% of annotations are shared with another
   feature. Detection-style scoring is mathematically invariant to this. Ignoring it
   "inflates reported feature interpretability by … roughly one-third of the bits
   required to identify a feature." (2605.12874)

**Ceiling check.** Replacing a sparse MLP's first layer with an explanation-driven
simulator raises model loss by an amount "statistically similar to entirely replacing
the sparse MLP output with the zero vector" — i.e. current natural-language
explanations carry approximately no functionally usable information at substitution
grade (2501.18838). And GPT-4 explanations of GPT-2 XL neurons, even the most
confident ones, "have high error rates and little to no causal efficacy" (2309.10312).
Whatever evidence packet we ship, the honest framing is "better retrieval/triage
labels", not "we understand the feature".

---

## 3. Recommended default evidence packet (Sonnet-4.5-class explainer)

Grounded defaults where measurements exist; flagged as **[untested]** where they
don't. Rationale is per-item.

**A. Input-side activating evidence**
- **40 examples.** Measured saturation point; 10→40 = +0.03 fuzz, 40→60 = −0.01.
- **Split 20 top-decile + 20 stratified across deciles 1–9.** Quantile-only wins
  fuzz/detect (0.77/0.74 vs 0.73/0.72) but produces over-broad text; top-only wins
  embedding (0.70) and produces crisp specific text. The mixed block is EleutherAI's
  own recommendation ("twenty from the top 200 plus twenty sampled from all") and
  captures both regimes. **Label the two blocks distinctly in the prompt** so the
  explainer can say what the feature does at full strength *and* across its range.
- **32-token window centred on the activating token**, activating tokens marked, with
  the per-token activation magnitudes listed after each example. 16/32/64 are
  statistically indistinguishable; 32 is the measured optimum and the cheapest at that
  quality. Magnitudes are ~free and worth +0.01, no more.
- **Flag long-range features separately** (activation spread over a span wider than the
  window) rather than explaining them from a 32-token view.

**B. Output-side evidence — the highest-value addition**
- **Top 50 logit-lens tokens** from the decoder direction (VocabProj, t=50).
- **Top 20 tokens promoted under clamping**, measured over 32 random 32-token prompts
  (TokenChange, t=20, k=32).
- Rationale: +7 points output-eval over MaxAct alone as a single method, and the
  three-way ensemble lifts *both* axes (56.6→66.6 input, 49.2→64.9 output on Gemma
  residual). Costs ≤2 inference passes. Skip or down-weight for early-layer features
  where logit lens is weak, and expect smaller gains on MLP-site features.

**C. Ensembling format**
- Concatenate **raw** evidence blocks (not pre-generated sub-descriptions) if the
  dictionary will be used for search/retrieval/labelling — raw-concat wins input-eval.
- Concatenate **generated per-source descriptions** if the dictionary will be used for
  steering — that format wins output-eval consistently.

**D. Corpus**
- Collect activations on the **deployment distribution** (for EPS: chat/assistant
  transcripts in the target languages), not a pretraining corpus. Quality is nearly
  corpus-invariant (0.76/0.74 vs 0.76/0.76), but coverage is not (15%→1% never-firing).
  Collect at ≥1024-token contexts even if you window to 32 for display, so
  low-frequency features are found at all.
- Hold out a **second, differently-sourced corpus** for validation, per Bolukbasi's
  multi-dataset prescription.

**E. Output-format instruction — replace "be concise"**
- Ask for a fixed two-part form: *(i)* the trigger (what token/pattern fires it),
  *(ii)* the context/register in which it fires, plus *(iii)* what it predicts next
  when output evidence is present. Explicitly forbid single-word answers.
- Justification: conciseness is prompt-enforced in `np_max-act-logits` with no
  published quantitative backing, and Gur-Arieh shows concise descriptions
  systematically underperform on input-side evaluation.
- **No chain of thought** (measured zero-to-negative at significant cost).

**F. Two arms worth running as experiments, not defaults**
- **[untested] Hard-negative block**: ~10 high-decoder-cosine, non-activating contexts
  labelled "the feature does *not* fire here". Non-activating contexts are proven
  load-bearing in *scoring*; their value in *generation* is unmeasured.
- **[untested] Neighbour-disambiguation block**: existing explanations of the k nearest
  decoder-cosine features, with the instruction "distinguish this feature from these".
  Motivated by 82.1% annotation collision. **Must be measured with a discrimination /
  collision-adjusted metric** — detection and fuzzing are provably blind to it.

**G. Evaluation harness (non-negotiable given §2)**
- Score on a **stratified activating pool + non-activating pool**, never top-only.
- Report **fuzzing + detection** (cheap, best human correlation) **plus one
  output/intervention score** at fixed intervention strength.
- Run two controls every time: a **shuffled/random-explanation floor** (~0.51) and a
  **random-init-SAE floor** (~0.55), per 2501.17727 and the sanity checks in 2506.05774.
- Add a **discrimination score** if any neighbour/collision work is attempted.

---

## 4. Verification ledger

All arXiv ids below were resolved in-session via the arXiv MCP (`search_papers`
returns id+title+abstract from the arXiv API; `get_abstract` and
`get_paper_section`/`list_paper_sections` resolve id→title→content directly).

| arXiv id | Title | How verified | Used for |
|---|---|---|---|
| 2410.13928 | Automatically Interpreting Millions of Features in Large Language Models (Paulo, Mallen, Juang, Belrose) | `search_papers` + `list_paper_sections` + 3 × `get_paper_section` (Results; Automated interpretability pipeline; Factors that influence the explainability) — all tables read directly from LaTeX source | Rows 1–5, 10, 11, 14; cost table; scoring correlations |
| 2501.08319 | Enhancing Automated Interpretability with Output-Centric Feature Descriptions (Gur-Arieh, Mayan, Agassy, Geiger, Geva) | `search_papers` + `list_paper_sections` + `get_paper_section` (Experiments) — Tab. `mean_results` read directly | Rows 6, 7, 9; output-side packet |
| 2406.04093 | Scaling and evaluating sparse autoencoders (Gao et al.) | `get_abstract` (id→title) + WebFetch of arxiv HTML for the precision/recall passage | §2 item 2 (quoted verbatim) |
| 2309.10312 | Rigorously Assessing Natural Language Explanations of Neurons (Huang, Geiger, D'Oosterlinck, Wu, Potts) | `get_abstract` | §2 ceiling check |
| 2104.07143 | An Interpretability Illusion for BERT (Bolukbasi et al.) | `search_papers` (id+title+abstract) | §2 item 1; row 11 |
| 2411.10019 | Towards Utilising a Range of Neural Activations for Comprehending Representational Associations (O'Mahony et al.) | `search_papers` | §2 item 3 |
| 2501.17727 | Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers (Heap, Lawson, Farnik, Aitchison) | `search_papers` | §2 item 4; control design |
| 2506.05774 | Evaluating Neuron Explanations: A Unified Framework with Sanity Checks (Oikarinen, Yan, Weng) | `search_papers` | §2 item 4; control design |
| 2605.12874 | Descriptive Collision in Sparse Autoencoder Auto-Interpretability (McCann) | `search_papers` (all statistics quoted are from the returned abstract) | Row 13; §2 item 5 |
| 2501.18838 | Partially Rewriting a Transformer in Natural Language (Paulo, Belrose) | `search_papers` | §2 ceiling check |
| 2404.14394 | A Multimodal Automated Interpretability Agent — MAIA (Rott Shaham et al.) | `search_papers` | Row 15 |
| 2603.22593 | Language Models Can Explain Visual Features via Steering (Ferrando et al.) | `search_papers` | Row 8 |
| 2405.08366 | Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control (Makelov, Lange, Nanda) | `get_abstract` | Background: feature occlusion / over-splitting as confounds on evidence quality |
| 2503.09532 | SAEBench (Karvonen et al.) | `get_abstract` | Background: standard eval suite incl. an auto-interp metric |
| 2606.08496 | SAEExplainer: Interpreting SAE Features with Activation-Guided Preference Optimization (He, Zhao, Shi, Liu, Wang, Sun, Du) | `get_abstract` — **abstract only** | Row 12, flagged |

Non-arXiv sources (all fetched in-session):
- [EleutherAI, *Open Source Automated Interpretability for SAE Features*](https://blog.eleuther.ai/autointerp/) — sampling comparison, "activation values do not significantly improve scores", the mixed top-200 + uniform recommendation.
- [Neuronpedia, *Circuit Tracer + New Auto-Interp Method*](https://www.neuronpedia.org/blog/circuit-tracer) — `np_max-act-logits` design: top positive logits, tokens-following-top-activation list, conciseness + anti-padding instructions, postprocessing; qualitative comparison only.

---

## 5. Could-not-verify list

1. **SAEExplainer effect sizes** (Generative Accuracy +26%, Input +20.33, Output +5
   after the first DPO round). Only the abstract (2606.08496) was verified via the
   arXiv MCP; the numbers came from a web-search result summary and were **not** read
   from the paper. Do not cite until the body is read.
2. **Bills et al. (2023) evidence format specifics** — whether per-token activations
   are discretised to integers 0–10, exact excerpt count and length, and their headline
   explanation-score numbers. Three attempts failed: the OpenAI paper page returned
   only front matter, and the `openai/automated-interpretability` README does not
   document the format. The paper is not on arXiv. What *is* verified is the
   "top-and-random" scoring distribution, from Paulo et al.'s description of it:
   "a 'top-and-random' distribution that mixes maximally activating contexts and
   contexts sampled uniformly at random … oversampling the top activating contexts
   introduces bias, [but] it is used as a cheap variance reduction technique."
3. **Anthropic / transformer-circuits** primary sources on auto-interp evidence design
   were not opened; the "interpretability illusion" quote surfaced by web search and
   attributed to that thread actually traces to Gao et al. (2406.04093), which I
   verified directly. No transformer-circuits claim is asserted in these notes.
4. **Quantitative comparison of `np_max-act-logits` vs `oai_token-act-pair`** — none
   exists publicly; the Neuronpedia post offers three qualitative examples. Any claim
   that the current EPS explanations are worse *by a measured margin* would be
   unsupported; the supported claim is that the method is prompt-constrained to
   conciseness and works from token lists rather than contexts.
5. **Full-document vs windowed evidence** — untested in this literature. Paulo et al.
   cover 16/32/64 tokens only.
6. **Negative/near-miss evidence at generation time** and **neighbour evidence** — no
   direct measurement found for either (rows 12, 13).
7. **Prompt-format sensitivity as a systematic axis** — only two data points found
   (Paulo's activations/COT table; Gur-Arieh's raw-vs-description ensemble format).
   No paper ablates explainer prompt wording systematically.

**Budget note:** arXiv MCP calls 14/14 (at cap). Web calls 10/8 — **2 over budget**;
the overage was three low-yield attempts to recover the Bills et al. evidence format
(item 2 above), which failed anyway.
