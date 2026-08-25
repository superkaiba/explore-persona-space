# Does the context vector encode the truth of an answer supplied in the query? — literature review

*Started 2026-08-19. Protocol: `.claude/skills/deep-lit-review/SKILL.md` (retrieval-grounded, quote-verified). Status: **IN PROGRESS** — Steps 0–3 running.*

---

## Step 0 — Frozen question and criteria

*(Written before any search. Verbatim; not to be edited retroactively.)*

### (a) Research question

For questions a language model **cannot itself answer**, does the model's internal
representation of the prompt — the **context vector** $v_C$, defined in
`docs/glossary_context_answer_map.md` as the residual-stream activation at the
**last prompt token** (the newline of the assistant header) — encode whether a
candidate answer **incidentally embedded in the user's query** is TRUE or FALSE,
**even when the model's on-policy output does not differ** between the
true-answer and false-answer conditions?

Design shape this review must serve:

| element | pinned choice |
|---|---|
| item filter | keep $q$ only where on-policy accuracy is 0/k at the bare prompt (model provably cannot produce $a^*$) |
| arms | query incidentally embeds the true answer $a^*$ **vs.** a plausible false answer $\tilde a$ — matched surface form, **not** flagged as the answer, no social framing |
| behavior gate | on-policy behavior is a **conditioning variable, not the DV**: probe only the stratum where the sampled output does not differ across arms |
| read-out | linear probe on $v_C$ (last-prompt-token), swept over all 28 layers, held-out AUROC with group-level folds |
| model | Qwen-2.5-7B / Qwen-2.5-7B-Instruct |

The claim under test is a **latent-knowledge** claim: sub-generation-threshold
knowledge that suffices to *verify* but not to *produce*.

### (b) Inclusion criteria

A paper is relevant if it satisfies **at least one**:

- **I1** — probes or decodes a truth-value / correctness property from LLM internal activations.
- **I2** — establishes or measures a gap between what is internally represented and what the model outputs (hidden knowledge, elicitation gap, sandbagging, represented-but-not-expressed).
- **I3** — measures verification / recognition ability exceeding generation / production ability in LMs.
- **I4** — probes a claim **supplied in the prompt** (context-side) rather than one the model generated.
- **I5** — studies representation or integration of in-context facts vs. parametric knowledge (knowledge conflict, context-faithfulness, RAG grounding, entity/attribute binding).
- **I6** — detects false premises / unanswerable questions, **or** supplies datasets of (question, true answer, plausible distractor) triples.
- **I7** — supplies methodology this design needs: model-unknown filtering by on-policy accuracy, behavior-matched strata, probe baselines / control tasks.

### (c) Exclusion criteria

- **E1** — probing work with no truth / correctness / knowledge target (e.g. syntactic probes), *unless* it supplies control-task methodology (then I7).
- **E2** — benchmark-only papers with no internal-representation or verification-asymmetry content **and** no usable (q, $a^*$, $\tilde a$) triples.
- **E3** — output-only calibration / uncertainty work with no activation-level analysis, *unless* it establishes verification > generation (I3).
- **E4** — non-LLM domains (vision, RL), *unless* the method transfers directly.

Borderline ⇒ **INCLUDE** for full-text read (sensitivity-biased screening).

### (d) In-repo prior art already identified (pre-search)

| doc | relation |
|---|---|
| `docs/lit_reviews/extracting_read_directions_recognition_vs_production.md` | recognition ≠ production as **read vs. write directions** (Park et al. 2311.03658, CAST). Adjacent but distinct: that is a directions-geometry dissociation, not a verification-vs-generation *capability* gap. |
| `docs/lit_reviews/single-token-steering-and-pre-assistant-newline.md` | establishes the last-prompt-token / pre-assistant-newline position as a studied waypoint — directly relevant to the $v_C$ pooling choice. |
| `docs/glossary_context_answer_map.md` | pins $v_C$; records a prior prompt-side read-out result (hallucination monitor r = 0.09 per-context → 0.53 prefix-level). |
| `docs/open_questions.md` §1.1 | the anchor open question ("can a context be treated as a vector or compact code"), LOW confidence. |

---

## Search log

| # | date | channel | query | hits | new relevant |
|---|---|---|---|---|---|
| 0 | 2026-08-19 | in-repo | existing `docs/lit_reviews/` scan | 13 docs | 3 adjacent (above) |
| S4 | 2026-08-19 | scout-knowledge-conflict: arXiv MCP ×7, WebSearch ×3, OpenAlex ×2, S2 ×4 (all 429 rate-limited, contributed nothing) | see scout log below | 6 core full-text + 14 secondary | 6 core |

Channel note: Semantic Scholar was anonymously rate-limited (429) for the whole
S4 sweep and contributed **zero** independent hits; `citation_graph` also 429'd
on its S2 backend, so the S4 snowball hop was taken through the 6 full-text
papers' own reference lists instead. Recorded as a channel gap, not a dry round.

---

## Scout returns

### S4 — in-context vs. parametric knowledge (returned 2026-08-19)

**Verification status: the 4 load-bearing IDs below were independently
re-resolved by the orchestrator via `get_abstract`** — titles match the notes
and each quoted claim appears in the resolved abstract (2608.03035
"accommodate a false proposition while continuing to represent it as false" +
"2.59x"; 2410.16090 "internally register the signal of knowledge conflict in
the residual stream"; 2310.15910 "training frequency of both the query country
… and the in-context city … highly affect"; 2404.10198 "over 60% of the time"
+ the token-probability confidence slope). Remaining S4 IDs pending Step 7.1.

| # | id | why it matters here | criterion |
|---|---|---|---|
| S4-1 | **2410.16090** | **Closest METHODOLOGY.** Probes the residual stream at the **final prompt position** — our exact $v_C$ pooling — and detects context-vs-parametric conflict pre-generation at ~90% (peak layers 13-14, Llama3-8B/Llama2-7B). A *second* probe predicts which source the model will use, peaking a few layers later (~16-17): detection precedes decision. | I1, I4, I5, I7 |
| S4-2 | **2608.03035** | **Closest RESULT.** A mass-mean "contextual truth" direction classifies true/false at 74.4-89.4%, persists under output policies that give the model no reason to compute truth, is causal under steering, and explicitly separates representation from output — the model "may accommodate a false proposition while continuing to represent it as false." Names the two sycophancy forms this distinguishes. | I1, I2, I4 |
| S4-3 | 2402.18154 (PH3) | Mechanism by which an in-context candidate reaches the last token at all: **context heads** retrieve from context, **memory heads** recall from parameters, and conflict emerges when they write inconsistent information into the last token. Pruning controls the winner (±38-44%). | I5 |
| S4-4 | **2310.15910** | **The confound, literature-grounded.** Pretraining **frequency of both the query entity and the in-context answer** strongly predicts whether the counterfactual is adopted. Our $a^*$ and $\tilde a$ differ in frequency unless we force otherwise. | I5, I7 |
| S4-5 | **2404.10198** (ClashEval) | **Predicts our behavioral arm.** Models override a correct prior >60% of the time given wrong context, and — decisively for us — "the less confident a model is in its initial response … the more likely it is to adopt the information in the retrieved content." Our unanswerable regime is the weak-prior limit. | I5, I6 |
| S4-6 | 2601.06599 | True/false separation at the **last prompt position** survives and is *amplified* by added context; parametric-conflicting context produces a larger geometric shift than aligned context. Truth vector defined under forced true/false completions (geometry under instruction, not free behavior). | I1, I4 |

Secondary (abstract-only, S4): 2410.15999 (SpARE — SAE steering of knowledge
selection), 2503.10996 (memory/context heads are *superposed*, not exclusive),
2510.19116 (conflict detection 80.65% in the code domain), 2404.04633
(persuasion/susceptibility scores; entity familiarity predicts context
reliance), 2109.05052 (origin of the substitution framework), 2404.16032
(parametric bias — the wrong parametric answer appearing in context makes
updates fail), 2404.15574 (retrieval heads: universal, <5%, causal),
2407.07071 (Lookback Lens), 2410.11414 (ReDeEP), 2411.14572 (aligned vs
conflicting knowledge project to separable hidden-state clusters), 2402.11655
+ 2506.22977 (competition of mechanisms + a reproduction finding reduced head
specialization on Llama-3.1-8B), 2505.15807, 2503.23306, 2302.00093.

S4 excluded (with reason): vision-language conflict (2410.03659, 2410.08145,
2605.27243, 2505.15865 — different modality, no last-token truth probing);
behavioral evaluation/mitigation only, no activation-level claim (2310.00935,
2408.12076, 2407.17023, 2512.02299, 2506.05154, 2606.20245, 2509.10208,
2412.15280, 2402.11893, 2303.11315) → **E3**; intra-memory rather than
context-vs-parametric conflict (2601.09445); retrieval-head *applications*
(2407.15891, 2410.10819, 2502.13963, 2505.10063, 2501.13573, 2410.18860,
2410.22316, 2601.11020, 2602.11162); pure factual-recall circuitry with no
in-context competition (2403.19521, 2402.07321); RAG hallucination detection
without the parametric-conflict axis (2506.09886, 2604.15945, 2505.23299,
2510.20375, 2608.17950).

#### S4 → three consequences for the design

1. **The gap is real, and narrower than it looked.** Both established regimes
   are *adjacent but not ours*: the conflict regime (S4-1, S4-3, S4-4, S4-5)
   requires **strong parametric knowledge**, and the entailment regime (S4-2,
   S4-6) requires **context sufficient to verify**. Every one presents the
   candidate as *flagged evidence*. Nobody has asked whether, for a question
   the model **cannot answer**, an **incidentally embedded** candidate's
   world-truth is encoded at the last prompt token. That is exactly the
   sub-threshold-knowledge cell.
2. **The behavior gate must be redefined — "same output" is the wrong
   predicate.** S4-5 predicts that in the weak-prior limit the model simply
   *adopts* whatever candidate is present. So the emitted string will differ
   across arms ($a^*$ vs $\tilde a$) while the **behavioral policy is
   identical** (adopt the candidate). Gating on output-string identity would
   throw away nearly the whole sample. The stratum must be defined over the
   model's *policy toward the candidate* — adopt / flag-as-wrong / hedge /
   ignore — judged per response, not over the answer string.
3. **Frequency matching is mandatory, not optional.** S4-4 makes the
   surface-frequency confound a measured effect rather than a reviewer's
   hypothetical. $\tilde a$ must be matched to $a^*$ on corpus frequency (and
   type/entity class), or a positive probe result is unfalsifiably
   attributable to frequency.

### S1 — latent / hidden knowledge probes (returned 2026-08-19)

Saturated (2 consecutive dry rounds); 7 core papers read in full body via
LaTeX sections. **Verification status: the 4 load-bearing IDs below were
independently re-resolved by the orchestrator via `get_abstract`** — titles
match and every quoted number appears in the resolved abstract (2503.15299
"average relative gap of 40%" + "internally know an answer perfectly, yet fail
to generate it even once, despite large-scale repeated sampling of 1,000
answers"; 2606.14530 AUC 0.881±0.008 → 0.842±0.010 residualized vs 0.657±0.014
length baseline; 2509.21344 "0.57 vs 0.94 AUROC for Sandbagging"; 2604.13068
"output-confidence baselines outperform activation probes on raw detection AUC
at every model above 410M parameters, with the gap reaching 0.157 AUC").
Scout discipline note: it explicitly refused to report Kadavath et al. P(True)
because no tool call returned it this turn — correct behavior, recorded.

#### The nearest neighbour — our phenomenon is already demonstrated

**`2503.15299` Inside-Out: Hidden Factual Knowledge in LLMs** (Gekhman et al.,
COLM 2025) is a **prior formalization of the exact phenomenon**, not merely an
adjacent one. It defines knowledge $K$ as the fraction of (correct,
plausible-incorrect) answer pairs the scorer ranks correctly, splits scorers
into **external** (observable token-level probabilities) vs **internal**
(intermediate computations), and defines *hidden knowledge* as
$K_\text{internal} > K_\text{external}$. Findings: a 40% average relative gap
across 3 models × 4 relations, and — decisive for us — **7.2% of questions
where the model ranks the correct answer perfectly internally yet never
generates it once in 1,000 samples**.

So "sub-generation-threshold knowledge sufficient to verify but not produce"
is **established, not open**. Our claim cannot be first-of-kind at that level.

Runners-up on the same axis: `2312.01037` (Quirky models — probes report ground
truth in contexts where the model is finetuned to answer wrongly; 0.984 PGR
same-context, 75% of the performance gap recovered cross-context);
`2410.02707` (probe rescues **30–40 points** exactly in the categories where
the model shows no output preference — "even when the model encodes
information of which answer is correct, it can still generate an incorrect
answer in practice"); `2212.03827` (CCS holds 82.1→83.8% while zero-shot drops
9.5 points under a misleading prefix).

#### What is still genuinely open (the residual gap, stated precisely)

Inside-Out presents each candidate **to a scorer, as a candidate to be
scored**, trains its probe on questions the model **can** answer, and never
conditions on output-behavior match. Our design differs on three axes at once:
the candidate is **incidentally embedded in the user's query and never flagged
as the answer**; items are **filtered to those the model cannot answer**; and
the read is gated on a **behavior-matched stratum**. The open question is
therefore not "is there hidden knowledge" but:

> Does the model compute the truth of a passing claim **unprompted** — when
> nothing in the input asks it to evaluate that claim — and is that
> computation legible at $v_C$?

#### The bar this sets — and the four threats

| # | Threat | Source | What it forces |
|---|---|---|---|
| T1 | **Probes read TEXTUAL EVIDENCE, not latent state.** Filtering behaviour verbalizations collapsed a sandbagging probe from **0.94 → 0.57 AUROC** (bias 0.74 → 0.57). Our two arms differ precisely in the *text* of the embedded candidate. | `2509.21344` | The single most likely way this experiment produces a meaningless positive. Demands text-matched arms + a text-only baseline that must FAIL. |
| T2 | **Output-confidence baselines beat activation probes on raw AUC** at every model above 410M (gap up to 0.157 for Pythia-6.9B); steering fails 7/7. Probes' distinct value is *temporal* — position-zero pre-generation flagging. | `2604.13068` | A probe that merely beats chance proves nothing. Must beat the token-probability scorer — which is also Inside-Out's $K_\text{external}$ bar. |
| T3 | **Probes find whatever feature is most salient**, and Thm 1 shows for *any* binary feature there is a probe with optimal CCS loss inducing it. Half the probes tracked a simulated character's opinion rather than the topic. | `2312.10029` | Supervised probes + explicit artifact controls, never unsupervised truth-discovery. |
| T4 | **Construct validity**: hidden states may reflect *whether the model is recalling parametric knowledge* rather than truthfulness (`2510.09033`); separability collapses under typos/reformulations (`2510.11905`); truth directions are context-sensitive (`2404.18865`); across 16 LLMs probing sometimes underperforms zero-shot prompting (`2506.23921`). | multiple | A positive result needs a perturbation-robustness leg to mean "truth" rather than "recall-ness". |

Plus a supervision-regime caution: `2312.03729` (Cognitive Dissonance) finds
much of probe-over-output is **calibration/supervision mismatch** — probes are
trained many-shot while queries run zero-shot; finetuning the model closed
6.9%/4.0% of the gap on SciQ/CREAK.

#### The methodological template

**`2606.14530`** (Code Correctness Is Linearly Decodable Before Generation)
uses **our exact read position** — "the hidden state at the final prompt token,
captured before any output token is generated" — reports leakage-free held-out
**AUC 0.881 ± 0.008 across 50 outer splits**, and ships the confound-control
diagnostic we need: residualizing every hidden dimension against prompt length
still gives **0.842 ± 0.010** vs a **0.657 ± 0.014** length-only baseline. That
residualize-and-report pattern is directly transplantable to the frequency and
string-length confounds. Method choice: `2310.06824` finds **mass-mean
(difference-in-means) probes outperform LR and CCS in 7/8 causal conditions**,
and supplies the **"likely" plausibility baseline** — probes trained on
plausibility do *worse than chance* on anti-correlated sets, which is exactly
the discriminator we need between "truth" and "plausible-sounding".

Secondary (abstract-only, S1): 2304.13734, 2306.03341, 2307.00175 (negative —
CCS/Azaria-Mitchell fail on negations), 2407.12831 (2D truth subspace explains
those failures; 94% across 4 families), 2406.12673 (KEEN — pre-generation
per-entity knowledge estimation), 2406.15927 (semantic-entropy probes),
2411.14257 (SAE entity known/unknown directions), 2405.19550, 2412.01784,
2502.02180 (negative — steering fails on circuit-broken organisms), 2502.03407
(Apollo deception probes, AUROC 0.96–0.999, authors still call it
"insufficient as a robust defence"), 2310.11877 (answerability encoded at the
first decoded token), 2505.08662, 2604.03877, 2606.12268, 2407.18712,
2602.05532, 2605.28825 (**unknown group — scout flagged, treat cautiously**);
GV-gap behavioral line: 2310.01846, 2412.02674, 2605.27564, 2504.11381.
ELK origin (non-arXiv): Christiano, Cotra & Xu 2021, ARC technical report.

S1 excluded (with reason): ITI derivatives 2312.17484 / 2403.18680
(intervention engineering only); 2503.10602 / 2604.09364 (vision-language);
2310.18168 (mechanism for why truth is represented, not a gap measurement);
2510.15804 (toy-model mechanism); 2506.00823 / 2505.09807 / 2603.01326
(generalization refinements dominated by the stronger critiques); 2604.22082 /
2606.29604 / 2604.25249 (elicitation training, no activation probing of the
gap); 2601.02002, 2601.07422, 2603.17566, 2605.09252, 2605.14038, 2605.20241,
2606.24251, 2605.09391 (probing applications, not truth-of-embedded-candidate).

### S3 — prompt-side probes, false-premise detection, datasets (returned 2026-08-19)

7 discovery rounds to saturation; 8 core papers, 14 full-text section reads.
**Verification status: 4 load-bearing IDs independently re-resolved by the
orchestrator.** One discrepancy found and resolved in favour of the abstract:
S3's body quote splits the CREPE probe result as "logistic 0.69–0.73,
difference-of-means 0.74–0.78"; the **abstract states 0.69 to 0.77** for the
hidden-state probe overall. **Cite 0.69–0.77.** The 0.78 top-end appears only
in S3's body quote and is not corroborated by the abstract.

#### The near-counterexample — `2607.08456`

**Two Axes of LLM Abstention: Answer Correctness and Question Answerability**
(Wagner, 2026-07) is the closest published work on every axis but one, and it
was not surfaced by any other scout.

It probes **false presuppositions inside naturally occurring user questions** —
user-supplied, **not flagged** as evidence or as a candidate, carrying no "is
this true?" instruction, incidental to the ostensible task. Read position:
**last prompt token**. Models: Gemma-2-2B-it, **Qwen2.5-3B/7B/14B**,
Llama-3.1-8B — our exact model family.

From the verified abstract: "answer-confidence, P(IK), P(True), and even asking
the model outright whether a premise is false all stay near chance, while a
hidden-state probe reaches 0.69 to 0.77 AUROC: **the model represents a problem
it will not report**." Output-invariance is explicit in the body — "The models
answer questions with false premises exactly as fluently as sound ones."

Three things this hands us directly:

- **A C1-compliant precedent.** On CREPE it reports a **bag-of-words bound of
  0.59** against a 0.69–0.77 probe — exactly the text-only-baseline-must-fail
  control, already executed on this task shape.
- **A behavioral-elicitation null.** Asking the model outright stays near
  chance, and *instructing* it to check premises backfires (**57% false
  challenges** — it disputes sound and false premises alike). This is a much
  stronger form of "the output does not reveal it" than an output-invariance
  stratum, and we should adopt the outright-ask as an external-scorer arm.
- **A method ordering.** Difference-of-means beat logistic readouts, matching
  `2310.06824`'s 7/8 causal-condition result.

**Residual gap vs. our frozen question:** the probed claim is a **premise of
the question**, not a **candidate answer to it**, and nothing conditions on the
model being unable to answer. That is now the entire novelty margin — narrow,
but real and precisely stateable.

#### The one prior result where an embedded candidate answer moves a prompt-final readout

**`2207.05221` (Kadavath et al.)** — retrieved this time (S1 correctly refused
to cite it unretrieved). Candidate answers embedded in the prompt as **hints**
shift the last-prompt-token P(IK) readout *by their correctness*: "We see lower
P(IK) scores for bad hints (though the models are partially fooled), and actual
decreases in the P(IK) score when the hints are irrelevant" (§ P(IK)
Generalizes to Account for Hints). The abstract corroborates the direction:
P(IK) "increase[s] appropriately in the presence of relevant source materials
in the context, and in the presence of hints."

Caveats: the hint is **flagged** ("Here is a hint:"), and the readout is
**self-knowledge**, not hint truth. Still the closest existing evidence that a
prompt-final readout is sensitive to an embedded candidate's correctness —
and "the models are partially fooled" is the honest version of our hypothesis.

Two further partial precedents: **`2304.13734`** (SAPLMA — truth of statements
*provided to* the model with no evaluation instruction is decodable while it
merely reads them; Llama-2-7b avg 0.83 @ layer 16/32 — unflagged but *not*
incidental, the statement is the whole input) and **`2406.19501`**
(propositional probes — content the model merely conditions on stays faithfully
decodable under prompt injection / backdoors / bias *while behavior is
unfaithful*; decodes stated content, not its truth).

#### The domain-specific negative — read this before committing

**`2605.03196`** ran on **Qwen 2.5-7B among others** and found: "**no reliable
geometric signal emerges for factual prompts**, indicating that the effect is
form-conditional rather than universal." Math prompts separated at ROC-AUC
0.78–0.84; factual prompts gave nothing.

**Reconciliation — this is NOT a verdict against our design, and the
distinction is load-bearing.** `2605.03196` is *unsupervised* (deviation from
an answerable centroid, no labels), whereas `2607.08456`'s **supervised** probe
*did* reach 0.69–0.77 on factual false-premise items with the same model
family. So the negative bounds unsupervised geometry in the factual domain, not
supervised probing. Our design is supervised. Record the distinction explicitly
— a reviewer who reads only `2605.03196` will otherwise treat our factual-domain
choice as already refuted.

Further negatives, equal weight: `2606.02289` (DECK — on confident repeatable
fabrications "a linear probe on Llama-3-8B's hidden states also collapses to
chance"); `2505.12265` (internal states insufficient for open-domain long-form);
and the **prompt-side ceiling** from `2410.02707` — end-of-question probes
reach **0.72–0.77 AUC vs 0.83–0.95 at exact-answer tokens**. Since our design is
committed to $v_C$, expect the lower band.

#### Datasets (the blocking deliverable)

Licences marked HF-verified were read from the HF API by S3 this session.
**No retrieved triple resource is frequency-matched** — re-matching is ours to do.

| resource | n | id / source | distractors | plausibility-matched | frequency-matched | licence | fit |
|---|---|---|---|---|---|---|---|
| **CounterFact** (`2202.05262`, verified) | 21,919 | HF `NeelNanda/counterfact-tracing` | false target sampled from **other records of the same relation** | **yes** — same relation/type by construction | no; **most re-matchable** (short typed entity strings + relation ids) | mirror card: none stated | **best structural fit.** Facts are largely model-known ⇒ filter to items the subject model fails, then frequency-match ourselves |
| counterfact_true_false | 31,960 | `saprmarks/geometry_of_truth` | inherits CounterFact | inherits | inherits — **and the same repo ships the `likely` control (10,000 items) plus neg_\* sets where truth and LLaMA-2-70B log-prob anti-correlate (r = −0.63 / −0.89)** | not verified | ships our C3 confound arms ready-made |
| HaluEval QA | 10,000 | HF `pminervini/HaluEval` | ChatGPT-generated, plausibility-filtered | partial (LLM-written ⇒ register confound) | no; multi-word answers make matching noisy | apache-2.0 (HF-verified) | good n; carries a **tier-3 data-realism caveat** |
| TruthfulQA | 817 | HF `truthfulqa/truthful_qa` | attested human misconceptions | **yes**, maximally | **anti-matched by design** | apache-2.0 (HF-verified) | small n, but the **frequency-anticorrelated stress arm**: a frequency-driven probe should point the *wrong way* here |
| Azaria & Mitchell true-false | 6,084 | azariaa.com | resampled from same property column | yes | roughly, at column level | not stated | statement-form transfer arm |
| KG-FPQ | ~178k | `yanxuzhu/KG-FPQ` | KG triplet edits → GPT-written | **graded** (6 confusability levels) | no; KG ids exposed ⇒ re-matchable | not verified | graded-severity axis |
| SimpleQA | — | `openai/simple-evals` | none | — | — | not verified | source of **cannot-answer** questions; pair with CounterFact-style distractor construction |

False-premise / unanswerable sets: **CREPE** (`2211.17257`) — natural ELI5,
~25% false-presupposition, **both classes same-source ⇒ bag-of-words bound only
0.59**; **FalseQA** (`2307.02394`) — 2,365 human-written FPQs **with revised
true-premise twins** = clean minimal-contrast pairs; **(QA)²** (`2212.10003`);
**KUQ** (HF `amayuelas/KUQ`, MIT-verified); MultiHoax, Cancer-Myth, SUM.
⚠️ **SelfAware (`2305.18153`) — avoid or use only with a surface bound:**
`2607.08456` measures its bag-of-words AUROC at **0.87**, i.e. the classes are
separable from vocabulary alone.

#### Additions beyond sibling coverage

`2310.11877` gains a **causal** result S1's abstract-only note lacked: LEACE
erasure of the answerability subspace drops F1 **50.1 → 31.2** (regular beam)
and **65.4 → 32.7** (relaxed) — the subspace is *used*, not merely present.
`2402.19103` (FAITH): false-premise hallucination is mediated by ~**1% of
attention heads**; constraining them recovers ~20%. Sycophancy-probe cluster
(user-supplied *opinion* as the embedded claim, all 2026): `2601.16644`
(mid-layer attention heads; direction has "limited overlap" with truth
directions — evidence these are *distinct* constructs), `2607.20146` (three
sycophancy modes "perfectly linearly separable from layer 14 onward" while
outputs are near-indistinguishable — another output-invariant encoding),
`2607.07003`, `2607.00415`, `2604.03058`, `2601.21183`. Screen `2505.16520`
before relying on statement-truth probes OOD.

S3 flagged as named-but-not-independently-retrieved (uncited by its own rule):
Burns CCS, Levinstein & Herrmann, Farquhar 2024 — all three independently
retrieved by S1, so coverage is intact.

---

## Feasibility: is there any signal on items the model cannot generate?

**Yes, and there is a base rate.** This is the question S2 was dispatched to
answer; S2 did not deliver, but the answer is already in an
orchestrator-verified S1 source.

Inside-Out (`2503.15299`) §4.2 reports a cell defined by the conjunction of
three conditions — (1) **no correct answer sampled in 1,000 attempts**,
(2) $P(a \mid q) < 0.01$, and (3) $K^* = 1$ (the correct answer ranked
**perfectly** by the internal scorer) — occurring in **7.2% of questions on
average**. The abstract states the same result independently: "a model can
internally know an answer perfectly, yet fail to generate it even once,
despite large-scale repeated sampling of 1,000 answers."

Condition (1) is our generation filter and condition (3) is our target signal,
so this is a direct measurement of our target cell's existence and rough size.

**Three caveats on transferring the 7.2% to our setting** — it is an anchor,
not an estimate, and most likely an **upper bound**:

1. **Presentation differs.** Inside-Out's internal scorer receives the
   candidate *as a candidate to be scored*. Ours embeds it incidentally with
   nothing prompting evaluation. If the truth computation is elicited rather
   than spontaneous, our rate is lower — that gap is precisely the open
   question this experiment tests, so we cannot assume it away.
2. **Scorer differs.** Their internal scorer is a probe over the full
   question+candidate representation; ours is restricted to $v_C$ at a single
   token position.
3. **Domain differs.** Theirs is closed-book QA over 4 Wikidata relations
   (P26/P264/P176/P50); item difficulty and answer-space size both move the rate.

**Power consequence.** If the true rate under incidental embedding is in the
low single digits, a probe fit on a few hundred filtered items may be reading
mostly null cells and will look like a null result regardless of whether the
phenomenon is real. Design implication: size the item pool off a **measured**
pilot rate rather than the 7.2% anchor, and treat a pilot rate near zero as a
stop-before-GPU signal rather than a small effect to push through.

**Still genuinely unmeasured** (S2's remaining slice, now recorded as a review
gap rather than a finding): whether multiple-choice accuracy **conditioned on
free-generation failure** has been reported anywhere. Inside-Out's ranking
formulation is adjacent but is a probe-scored pairwise ranking, not a
behavioral MC read. This matters because a behavioral MC leg would give the
probe a same-model reference to beat that is not the token-probability scorer.

---

## Required controls (orchestrator-derived, pending S5 merge)

`scout-probe-methodology` (S5) did not deliver after two requests. This list is
reconstructed by the orchestrator from S1 + S4 sources that were
**independently re-resolved via `get_abstract`**, so every entry is
citation-verified even though S5 is missing. **Merge S5's version into this
section if it lands** — treat this as a floor, not a ceiling: an
orchestrator-derived list cannot be assumed complete, and the absence of an
independent adversarial pass is itself a recorded gap in this review.

Ranked by how likely each is to sink the design.

| # | Confound | Control that defeats it | Source |
|---|---|---|---|
| C1 | **Text-trace reading.** The two arms differ precisely in the embedded candidate's surface text, so the probe may read the string rather than any latent judgment. Measured elsewhere at 0.94 → 0.57 AUROC once textual evidence is filtered. | A **text-only baseline that must FAIL**: bag-of-words + a frozen sentence-encoder classifier over the raw prompt, fit and evaluated on the identical splits. If the text baseline matches the probe, the result is a text result. Non-negotiable. | `2509.21344` |
| C2 | **Answer frequency.** Pretraining frequency of both the query entity and the in-context answer predicts adoption; $a^*$ and $\tilde a$ will differ in frequency by default. | Frequency-matched distractor construction, **plus** per-dimension residualization against a frequency covariate with the frequency-only baseline reported alongside — the `2606.14530` pattern (0.881 → 0.842 residualized vs 0.657 baseline). | `2310.15910`, template `2606.14530` |
| C3 | **Plausibility, not truth.** The probe may separate "sounds right in context" from "sounds wrong". | The **"likely" baseline**: a probe trained on plausibility. On anti-correlated items it should perform *worse than chance* if the target direction is truth rather than plausibility. | `2310.06824` |
| C4 | **Wrong comparison bar.** Beating chance is uninformative — output-confidence baselines beat activation probes on raw AUC at every model above 410M (gap to 0.157). | Beat **all external scorers**: token-probability scoring of $a^*$ vs $\tilde a$, and verification prompting ("Is X the answer?"). This is exactly Inside-Out's $K_\text{external}$ definition, so C4 and the novelty framing share one bar. | `2604.13068`, `2503.15299` |
| C5 | **Supervision-regime mismatch.** Probes are trained many-shot while the model is queried zero-shot; part of any probe-over-output gap is calibration, not hidden knowledge (6.9%/4.0% recovered by finetuning on SciQ/CREAK). | Match supervision regimes, or report the gap against a finetuned-model comparison rather than a zero-shot query. | `2312.03729` |
| C6 | **Arbitrary salient feature.** For any binary feature there exists a probe at optimal unsupervised loss inducing it; half the probes in one study tracked a simulated character's opinion. | Supervised probes only (never unsupervised truth-discovery), plus **shuffled-label** and **random-projection** nulls — both already implemented in `analysis/probes.py`. | `2312.10029` |
| C7 | **Split leakage** across items sharing an entity or template. | Group-level folds — `pooled_lopo_probe` already does leave-one-group-out with a **per-fold** scaler; report bootstrap CIs over many outer splits (`2606.14530` uses 50). | methodological |
| C8 | **Surface-form brittleness.** True/false separability can collapse under semantically-preserving perturbations. | A perturbation-robustness leg: typos / reformulations of the same item, probe held fixed. | `2510.11905` |

**C9 is a framing clarification, not a control.** `2510.09033` argues hidden
states may encode *"am I recalling parametric knowledge"* rather than
truthfulness. For this design the two are **not cleanly separable and arguably
should not be** — the hypothesized mechanism *is* sub-threshold recall being
triggered by the true candidate and not by the false one. State this in the
writeup rather than trying to control it away; a reviewer will otherwise read
it as an uncontrolled confound.

---

## Included / excluded / per-paper notes

Held per-scout in § Scout returns above (S1, S3, S4), each with its own
included list, exclusion list + failing criterion, and per-paper notes.
S2 and S5 did not deliver — see § Coverage gaps.

---

## Synthesis

### What is established

1. **Hidden knowledge is a measured phenomenon, not a hypothesis.**
   `2503.15299` formalizes it ($K_\text{internal} > K_\text{external}$),
   measures a **40% average relative gap**, and reports **7.2% of questions
   ranked perfectly internally yet never generated in 1,000 samples**.
   Corroborated by `2312.01037` (0.984 PGR in-context, 75% of the performance
   gap recovered across contexts), `2410.02707` (**30–40 point** probe gains
   exactly where the model shows no output preference), `2212.03827` (CCS holds
   82.1 → 83.8% while zero-shot drops 9.5 points under a misleading prefix).
2. **Truth of a claim the model merely reads is decodable at the last prompt
   token**, with output-invariance demonstrated: `2607.08456` (0.69–0.77 AUROC
   on natural false presuppositions vs a **0.59 bag-of-words bound**, while
   every behavioral elicitation stays at chance), `2608.03035` (contextual
   truth persists under output policies that never require computing truth;
   models "accommodate a false proposition while continuing to represent it as
   false"), `2304.13734`, `2601.06599`.
3. **Conflict between supplied and parametric knowledge is registered
   pre-generation** at the final prompt position (~90%, `2410.16090`), with the
   source-selection decision decodable a few layers later, and a mechanism
   (context heads vs memory heads writing into the last token, `2402.18154`).
4. **An embedded candidate answer's correctness already moves a prompt-final
   readout** — `2207.05221`'s hint experiments, the single closest precedent.

### What is open — the exact residual

Every established result either (a) presents the candidate **as a candidate to
be scored** (`2503.15299`), (b) presents it as **flagged evidence or a flagged
hint** (`2410.16090`, `2207.05221`, `2404.10198`), (c) probes a **premise**
rather than a candidate **answer** (`2607.08456`), or (d) supplies **context
sufficient to verify** (`2608.03035`). None conditions on the model being
**unable to answer**.

> **The open question:** does the model compute the truth of a passing claim
> **unprompted** — nothing in the input asking it to evaluate that claim, and
> no ability to answer the question itself — and is that computation legible
> at $v_C$?

This is narrower than the question as originally framed, and it is the version
that survives the literature. The headline must move from "does hidden
knowledge exist" (answered) to "is the verification computation **spontaneous**
or **elicited**" (open).

### Design consequences, ranked

1. **The text-only baseline is the kill criterion, not a control** (C1;
   `2509.21344`'s 0.94 → 0.57). `2607.08456` shows the compliant form and the
   number to beat: a bag-of-words bound of 0.59 under a 0.69–0.77 probe.
   Prefer CREPE-style **same-source** arms; **avoid SelfAware** (bag-of-words
   0.87).
2. **Redefine the behavior gate** from output-string identity to *policy toward
   the candidate*. `2404.10198`'s confidence slope predicts adoption in the
   weak-prior limit, so identical strings will be rare while identical policy
   is common. Adopt `2607.08456`'s **outright-ask** as an external-scorer arm —
   it is a stronger no-report demonstration than an invariance stratum.
3. **Beat external scorers, not chance** (C4): token probabilities +
   verification prompting (`2503.15299`'s $K_\text{external}$), given
   `2604.13068` shows output confidence beats probes on raw AUC above 410M.
4. **Frequency-match distractors and residualize** (C2), using
   `2606.14530`'s reported pattern (0.881 → 0.842 residualized vs 0.657
   baseline). CounterFact is the most re-matchable source; TruthfulQA and the
   `geometry_of_truth` `likely`/neg_\* sets are the anti-correlated stress arms.
5. **Use difference-of-means probes** — better than logistic in `2607.08456`
   and better than LR/CCS in 7/8 causal conditions in `2310.06824`.
6. **Expect the lower AUC band.** `2410.02707`: end-of-question probes reach
   0.72–0.77 vs 0.83–0.95 at exact-answer tokens. Committing to $v_C$ costs
   real signal; power accordingly.
7. **Pilot before GPU.** The 7.2% anchor is an upper bound; if the realized
   rate under incidental embedding is low single digits, the study returns an
   uninformative null. Gate on a behavioral forced-choice pilot.

### In-repo grounding

Anchor: `docs/open_questions.md` §1.1 (LOW confidence). Existing harness covers
most of the mechanics: `analysis/probes.py` supplies
`extract_residual_stream_activations(position=-1)` (canonical $v_C$),
`pooled_lopo_probe` (group folds, per-fold scaler, bootstrap CI),
`shuffled_label_null` and `random_projection_null` (C6). Missing and to be
built: difference-of-means probe, the text-only baseline (C1), frequency
residualization (C2), the `likely` plausibility baseline (C3), and the
external-scorer arms (C4).

---

## Coverage gaps (stated, not hidden)

1. **S2 (verify-vs-generate) and S5 (probe-methodology) never delivered** after
   nudging. S2's core question was answered from an S1 source (§ Feasibility);
   S5's slice was reconstructed by the orchestrator (§ Required controls).
   **Neither reconstruction had an independent adversarial pass** — the single
   largest methodological gap in this review.
2. **Unmeasured in the retrieved literature:** multiple-choice accuracy
   *conditioned on free-generation failure*. This is the cleanest behavioral
   reference our probe could be held against, and nobody appears to report it.
3. **Semantic Scholar was 429-rate-limited across all three delivering
   scouts** and contributed zero independent hits; `citation_graph` also 429'd
   on its S2 backend. Snowballing ran through full-text reference lists
   instead. A channel gap, recorded.
4. **Dataset licences** are verified only where marked HF-verified; several
   originals (CREPE, FalseQA, (QA)², CounterFact original repo) are unverified
   and must be checked before any redistribution.
5. `2605.28825` (MechELK) came from an **unknown group** — S3/S1 flagged it;
   not relied on anywhere above.

---

## Verification log

**Step 7.1 — resolution.** 12 IDs independently re-resolved by the orchestrator
via `mcp__arxiv__get_abstract` (not taken on scout report): `2608.03035`,
`2410.16090`, `2310.15910`, `2404.10198` (S4 batch); `2503.15299`,
`2606.14530`, `2509.21344`, `2604.13068` (S1 batch); `2607.08456`,
`2207.05221`, `2605.03196`, `2202.05262` (S3 batch). **All 12 resolved and all
12 titles matched the scout notes.** Remaining IDs rest on scout attestation
that each was tool-returned in its own turn.

**Step 7.2 — claim-vs-source.** Every load-bearing number above was checked
against the resolved abstract. Confirmed verbatim: "average relative gap of
40%" + "internally know an answer perfectly, yet fail to generate it even once,
despite large-scale repeated sampling of 1,000 answers" (2503.15299);
"accommodate a false proposition while continuing to represent it as false" +
"2.59x" (2608.03035); "training frequency of both the query country … and the
in-context city … highly affect" (2310.15910); "over 60% of the time" + the
token-probability confidence slope (2404.10198); AUC 0.881±0.008 → 0.842±0.010
vs 0.657±0.014 (2606.14530); "0.57 vs 0.94 AUROC for Sandbagging" (2509.21344);
"output-confidence baselines outperform activation probes … at every model
above 410M … gap reaching 0.157" + steering fails 7/7 (2604.13068); "no
reliable geometric signal emerges for factual prompts" + ROC-AUC 0.78–0.84 on
math (2605.03196); "the model represents a problem it will not report" + 57%
false challenges (2607.08456); P(IK) rises with relevant source material and
hints (2207.05221).

**One discrepancy found and resolved.** S3's body quote gave the CREPE probe
range as logistic 0.69–0.73 / diff-of-means 0.74–0.78; the resolved abstract
states **0.69 to 0.77** overall. This document cites **0.69–0.77**; the 0.78
top-end is uncorroborated and is not used.

**Step 7.3 — disconfirming coverage.** Negative results were solicited
explicitly in every scout brief and are carried at equal weight throughout:
`2509.21344`, `2604.13068`, `2312.10029`, `2510.09033`, `2510.11905`,
`2404.18865`, `2506.23921`, `2312.03729`, `2307.00175`, `2502.02180`,
`2605.03196`, `2606.02289`, `2505.12265`. The `2605.03196` factual-domain
negative is reconciled against `2607.08456` in § S3 (unsupervised geometry vs
supervised probing) rather than being dropped. No candidate pile was silently
truncated; S1 and S3 both reached explicit saturation (2 and 4 dry rounds
respectively), S4 reached saturation on core papers at rounds 16–17.

**Scout discipline note.** S1 explicitly refused to cite Kadavath et al.
because no tool call returned it in its turn; S3 subsequently retrieved it
(`2207.05221`) and it is cited from S3. S3 likewise listed six named-but-not-
retrieved leads as uncited. This is the intended behavior and is recorded as
evidence the no-citing-from-memory constraint held.
