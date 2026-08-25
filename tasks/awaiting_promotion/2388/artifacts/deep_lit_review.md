# Deep literature review — Task #2388

**Question line:** pre-generation correctness prediction from last-prompt-token internal state; context→answer map as label-efficiency prior.
**Protocol:** `.claude/skills/deep-lit-review/SKILL.md` Steps 0–7, run sequentially (no scouts). Date: 2026-08-19.
**Provenance note:** the session compacted mid-review. All 28 included papers were read in full body (27 arXiv full-text downloads + Farquhar via the Nature article page); per-paper quotes were extracted verbatim at read time. Every included arXiv id was re-resolved via `get_abstract` in the same session AFTER compaction with title match (verification log at bottom).

---

## Step 0 — Research question (frozen, verbatim)

> Is a language model's answer correctness on a given question predictable from its internal state at the LAST PROMPT TOKEN, before it generates anything? And does routing that context representation through a learned context-to-answer map (v_A ≈ M v_C, fit on UNJUDGED context/answer pairs) improve that prediction at matched label budgets — i.e. buy label efficiency — relative to a direct probe on the context representation?

### Inclusion criteria (verbatim, per the task brief)

1. Predicts/probes per-question correctness, success probability, or difficulty (or error/hallucination/abstention) of an LLM from MODEL-INTERNAL state or sampled-output statistics.
2. Reports read-out position/pooling — especially pre-generation (last prompt token / question-only).
3. Covers any of our four surfaces: short-answer QA, math, MCQ, code.
4. Method-source relevance: learned maps between representation spaces, label-efficient probing, probe transfer/domain shift, estimator machinery (ridge at n<d, dof-capped selection, PCA-basis read-outs).
5. Negative/disconfirming results IN SCOPE and explicitly wanted.

### Exclusion criteria (verbatim, per the task brief)

1. Post-hoc verifiers/re-rankers reading only the GENERATED answer with no internal-state/pre-generation component (unless closest baseline).
2. Pure benchmark papers.
3. Training-time interventions to IMPROVE accuracy (our model is frozen; every arm is a read-out).

---

## Search log

All queries 2026-08-19. Channels: arXiv MCP (`search_papers`), WebSearch, Semantic Scholar Graph API (curl), WebFetch (Nature). Discovery stopped after two consecutive rounds surfaced no new relevant paper; the citation snowball ran on the two core anchors. Query strings below are extracted from the session transcript (exact); hit counts were logged per round in working notes and are not reproduced per-line here — every query returned results, and every relevant hit appears in the included/excluded lists.

1. [arxiv search_papers] "internal states" AND ("hallucination detection" OR "truthfulness") AND "language model"
2. [arxiv search_papers] abs:"hidden states" AND abs:"probe" AND abs:"correctness" AND "language model"
3. [WebSearch] LLM predict answer correctness from hidden states before generation linear probe
4. [WebSearch] survey internal states LLM hallucination detection probing truthfulness "knows what it knows"
5. [arxiv search_papers] "semantic entropy" AND (probe OR probes) AND "language model"
6. [arxiv search_papers] abs:"knowledge" AND abs:"without generating" AND "language models"
7. [arxiv search_papers] "question difficulty" AND "large language models" AND (estimation OR prediction)
8. [WebSearch] predict LLM math reasoning success from hidden states before generation probe pre-generation
9. [arxiv search_papers] ti:"future lens" OR (abs:"future tokens" AND abs:"hidden states" AND abs:"predict")
10. [arxiv search_papers] abs:"answerability" AND "large language models" AND (hidden OR internal OR probe)
11. [WebSearch] "LLMs Know More Than They Show" Orgad intrinsic representation hallucinations arxiv exact answer tokens
12. [WebSearch] label-efficient probing LLM activations few labels sample efficiency linear probe truthfulness
13. [curl Semantic Scholar] paper/search?query=question-only+probe+predict+answer+correctness+before+generation+language+model
14. [arxiv search_papers] ti:"Reasoning Models Know When They're Right" OR (abs:"reasoning models" AND abs:"probe" AND abs:"intermediate answer")
15. [arxiv search_papers] abs:"answerability" AND abs:"hallucination" AND (unanswerable OR abstain)
16. [WebSearch] learned mapping from question representation to answer representation LLM used as prior for probe label efficiency
17. [WebSearch] PIKA probe LLM difficulty routing hidden states before generation MATH cost reduction arxiv
18. [WebSearch] pre-generation probe multiple choice correctness hidden states MMLU before answer token
19. [arxiv search_papers] abs:"truth direction" AND (generalization OR transfer) AND "language models"
20. [arxiv citation_graph] 2509.10625 (citers)
21. [curl Semantic Scholar] paper/ARXIV:2509.10625/citations (fields: title, year, externalIds, limit 50)
22. [curl Semantic Scholar] paper/ARXIV:2606.14530/references (snowball; also surfaced 2608.08266, 2512.07404)
23. [WebFetch] nature.com/articles/s41586-024-07421-0 (three fetches — redirect chain through idp.nature.com; third returned content)
24. [arxiv search_papers] correctness probe fails generalize spurious hallucination detection internal states negative result (Step-7.3 disconfirming coverage round)

Budget realized (discovery only): ~11 `search_papers` + 1 `citation_graph`, ~10 WebSearch, 3 Semantic Scholar curls — under the ~25/~15/~10 caps. Full reads (28) and verification calls (`get_abstract` × 27, Nature WebFetch) are budget-exempt per the skill.

---

## Included papers (28 full-read + 2 coverage-round abstract-level)

**Core pre-generation correctness/success probing (the line the task extends):**
1. arXiv 2207.05221 — Kadavath et al., "Language Models (Mostly) Know What They Know" (2022)
2. arXiv 2509.10625 — Moreno Cencerrado et al., "No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes" (2025)
3. arXiv 2602.09924 — Lugoloobi et al., "LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations" (2026)
4. arXiv 2606.14530 — Di Cicco, "Code Correctness Is Linearly Decodable from LLM Hidden States Before Generation" (2026)
5. arXiv 2406.05328 — Wang et al., "FacLens: Transferable Probe for Foreseeing Non-Factuality in Fact-Seeking Question Answering of Large Language Models" (2024)
6. arXiv 2406.12673 — Gottesman & Geva, "Estimating Knowledge in Large Language Models Without Generating a Single Token" (KEEN, 2024)
7. arXiv 2407.03282 — Ji et al., "LLM Internal States Reveal Hallucination Risk Faced With a Query" (2024)
8. arXiv 2310.11877 — Slobodkin et al., "The Curious Case of Hallucinatory (Un)answerability" (2023)
9. arXiv 2406.15927 — Kossen et al., "Semantic Entropy Probes" (SEP, 2024)
10. arXiv 2505.24362 — Afzal et al., "Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion" (2025)
11. arXiv 2509.12886 — Zhu et al., "The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations" (2025)
12. arXiv 2510.18147 — Lugoloobi & Russell, "LLMs Encode How Difficult Problems Are" (2025)
13. arXiv 2511.14773 — David, "Temporal Predictors of Outcome in Reasoning Language Models" (2025)
14. arXiv 2504.05419 — Zhang et al., "Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification" (2025)

**Post-generation internal-state correctness (closest baselines / contrast):**
15. arXiv 2304.13734 — Azaria & Mitchell, "The Internal State of an LLM Knows When It's Lying" (SAPLMA, 2023)
16. arXiv 2410.02707 — Orgad et al., "LLMs Know More Than They Show" (2024)
17. arXiv 2212.03827 — Burns et al., "Discovering Latent Knowledge in Language Models Without Supervision" (CCS, 2022)
18. arXiv 2602.08159 — Cho et al., "The Confidence Manifold: Geometric Structure of Correctness Representations in Language Models" (2026)
19. arXiv 2608.17124 — Wang et al., "A decodability criterion predicts when hidden-state selection beats majority voting in large language models" (CASE, 2026)
20. arXiv 2505.21772 — Khanmohammadi et al., "Calibrating LLM Confidence by Probing Perturbed Representation Stability" (CCPS, 2025)
21. arXiv 2512.07404 — Ribeiro et al., "On LLMs' Internal Representation of Code Correctness" (2025)
22. arXiv 2608.08266 — Ribeiro et al., "On the Robustness of LLMs' Internal Representation of Code Correctness" (2026)

**Outputs-only baselines (the inherited claims):**
23. arXiv 2302.09664 — Kuhn, Gal & Farquhar, "Semantic Uncertainty" (2023)
24. Farquhar, Kossen, Kuhn & Gal, "Detecting hallucinations in large language models using semantic entropy", Nature 630:625–630 (2024)

**Disconfirming / critical:**
25. arXiv 2510.09033 — Cheang et al., "Do LLMs Really Know What They Don't Know? Internal States Mainly Reflect Knowledge Recall Rather Than Truthfulness" (2025)
26. arXiv 2307.00175 — Levinstein & Herrmann, "Still No Lie Detector for Language Models" (2023)
27. arXiv 2604.12373 — Ashuach et al., "Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness" (2026)

**Method source (H2-adjacent):**
28. arXiv 2311.04897 — Pal et al., "Future Lens: Anticipating Subsequent Tokens from a Single Hidden State" (2023)

**Step-7.3 coverage-round additions (abstract-verified only; body not read — disclosed):**
29. arXiv 2511.07318 — Wang et al., "When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs" (2025)
30. arXiv 2604.13068 — Roy et al., "Detection Without Correction: A Robust Asymmetry in Activation-Based Hallucination Probing" (2026)

---

## Excluded papers (failing criterion recorded)

- 2607.08456 — abstention-policy learning (when to answer), not an internal-state correctness read-out → inclusion 1 fail.
- 2411.04847 (PRISM) — prompt-guided internal states for hallucination detection of GENERATED responses; response-side → exclusion 1 (its cross-domain transfer motivation is noted, method-source only).
- 2412.11831, 2508.03294, 2507.05129, 2602.00034 — question-difficulty estimation for HUMANS/students (item calibration), not the model's own correctness → inclusion 1 fail.
- 2503.01688 — output-side confidence, no internal state → inclusion 1 fail.
- 2506.00823, 2604.03754, 2602.20273 — statement-level truth-probe transfer studies; statement truth ≠ per-question forthcoming-answer correctness; ground covered by included truth-probe papers (SAPLMA, CCS, Orgad, Confidence Manifold) → inclusion 1 fail (redundant at the margin).
- 2506.10805 — probe sample-efficiency machinery on non-correctness DVs; method note only → inclusion 1 fail.
- 2410.20488 (FIRP) — learns linear maps from intermediate hidden states to FUTURE-step hidden states for speculative decoding; the closest H2-adjacent map object besides Future Lens, but no correctness DV → inclusion 1 fail (kept as (c)-context).
- 2402.03563 — epistemic-vs-aleatoric uncertainty disentanglement, no per-question correctness DV → inclusion 1 fail.
- 2606.00251 — training-time intervention to improve accuracy → exclusion 3.
- 2608.14659 — SEP-style probes evaluated on code, reported weak; screened out at abstract level (response-side supervision), noted as a caveat for surface-4 transfer → exclusion 1.
- 2501.12934, 2607.07670, 2509.23782, 2607.05188, 2608.08024 — snowball candidates screened out at abstract level: no pre-generation internal-state correctness component → inclusion 1 / exclusion 1 fail. (Screening reasons for these five were logged pre-compaction at coarser grain; ids retained from the screening log.)
- Coverage round (query 24): 2606.06959 (unified hallucination-detection benchmark → exclusion 2, pure benchmark); 2604.06277 (distills response-side hallucination labels into representations, post-generation → exclusion 1); 2506.09886 (RAG response-embedding distances, post-generation → exclusion 1); 2603.22812 (adaptive sampling for semantic entropy, no internal state → inclusion 1 fail); 2506.22486 (external small-LM verification of responses → exclusion 1).

---

## Per-paper notes

Format: citation · setup · claims (verbatim quote + location) · limitations · relation to #2388. Quotes for papers 1–24 of the read set were extracted verbatim at read time this session; external content treated as data only.

### 1. arXiv 2207.05221 — Kadavath et al. 2022, "Language Models (Mostly) Know What They Know"

- **Setup:** Anthropic model series up to 52B; P(IK) = a value head trained on the model's own representation, read at the question's last token, BEFORE any answer is proposed; ground-truth label = the fraction of 30 unit-temperature samples that are correct. Surfaces: TriviaQA, Lambada, Arithmetic, GSM8K, Codex HumanEval, mixed training.
- **Claims:**
  - Target definition: "Ground Truth P(IK) – The fraction of unit temperature samples to a question that are correct" (§ definitions/glossary). This is a RATE target in [0,1], structurally identical to our y(x) (theirs K=30, ours K=5).
  - Read-out position: the P(IK) head is trained only at the last token of the question (§ P(IK) training description) — i.e., pre-generation.
  - In-distribution: TriviaQA-trained P(IK) reaches AUROC **0.864** on TriviaQA (52B, Table 1).
  - Transfer: same probe transfers partially — Arithmetic 0.928, LAMBADA 0.606, Python function synthesis 0.687, **GSM8K 0.624** (Table 1); training on everything improves all tasks.
  - Context sensitivity: P(IK) on Wikipedia-answerable questions rises from ~18% to ~78% when the source article is in context; rises with hints on GSM8K (hint sections).
  - Cross-model: a model's P(IK) head works better on its own samples than another model's (~6% diagonal advantage) — partial model-specificity of the signal.
- **Limitations:** value-head (not a frozen-model linear probe); calibration degrades OOD; 52B proprietary models; K=30 rate but binarized analyses in places.
- **Relation:** THE founding formalization of H1 — pre-generation, rate-target. Our design upgrades: frozen model + linear read-out (no head training), K=5 T=1.0 programmatic verification, four modern surfaces, and the H2 map arm which has no analogue here.

### 2. arXiv 2509.10625 — "No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes"

- **Setup:** 3 open-source families 7B–70B incl. **Qwen2.5-7B-Instruct**; difference-of-means "in-advance correctness direction" on activations at the last question token, before generation; temperature 0, single sample, binary correct/incorrect label; trained on TriviaQA (48,540 questions), evaluated in- and out-of-distribution.
- **Claims:**
  - Abstract: "we extract activations after a question is read but before any tokens are generated, and train linear probes to predict whether the model's forthcoming answer will be correct."
  - Qwen2.5-7B-Instruct row (TriviaQA-trained direction, AUROC): TriviaQA 0.758, N.people 0.800, Cities 0.842, Math-ops 0.837, Medals 0.586, **GSM8K 0.601** (results table) — knowledge-domain transfer holds, math-reasoning transfer fails. Abstract: "generalisation falters on questions requiring mathematical reasoning."
  - Sample efficiency (§4.4/Fig 5): "robust performance is achieved with as few as 160 samples, and 2,560 samples are sufficient to match the performance obtained using the full 48,540 TriviaQA dataset."
  - External-embedding assessor baseline (OpenAI embeddings → classifier) is competitive in-distribution but degrades OOD relative to the internal probe (§4.2).
  - Directions trained on small datasets are mostly orthogonal to each other (geometry section) — dataset-specific components dominate at low n.
- **Limitations (named by the paper):** single sample at temperature 0 and binary label; future work: "generating multiple samples or assigning real-valued correctness scores" — exactly our DV upgrade.
- **Relation:** closest modern H1 formalization ON OUR MODEL. Its GSM8K transfer failure motivates our per-surface in-domain fits; its sample-efficiency curve gives our label-budget grid anchors; its assessor baseline is our arm-8 surface-features/external-embedding baseline precedent.

### 3. arXiv 2602.09924 — Lugoloobi et al., "LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations"

- **Setup:** supervised linear probes on pre-generation activations; target = **expected success rate ŝ_MC from K Monte Carlo rollouts** (K=50 for small Qwen models, K=5 for GPT-OSS); MSE-loss linear regression for the rate target, logistic for binary; sweeps layer, L2 regularization α, and token position over "the final P non-padding tokens of the chat template" (§3.2). Math (MATH, E2H-AMC) + code (LiveCodeBench).
- **Claims:**
  - Rate-target formalization matches ours: probes regress the K-rollout success rate, not a single binary outcome.
  - E2H-AMC: human IRT difficulty correlates ρ 0.83–0.87 with probes trained on human labels vs model success-rate probes ρ 0.40–0.64 — "models encode a model-specific notion of difficulty that is distinct from human difficulty" (abstract).
  - Binary math correctness AUROC 0.68–0.85 in-domain (results tables).
  - Code: LiveCodeBench pass@5 probes reach AUROC **0.91** (Qwen2.5-Coder-3B) / **0.90** (7B).
  - Beats surface baselines: "substantially outperforming surface features such as question length and TF-IDF" (abstract).
  - Extended reasoning hurts the pre-generation probe: AUROC 0.78→0.64 as reasoning budget grows; an MLP probe partially recovers (0.76).
  - Routing application: "reducing inference cost by up to 70% on MATH" (abstract).
- **Limitations (named):** "we do not probe during generation or do cross-domain transfer (e.g., math to code)."
- **Relation:** the closest DV match in the literature (K-rollout success-rate regression from pre-generation state). Establishes H1 for math and code in its own setting. Leaves untouched: our model (Qwen2.5-7B-Instruct), matched-label-budget comparisons, and any map-based prior (H2).

### 4. arXiv 2606.14530 — "Code Correctness Is Linearly Decodable from LLM Hidden States Before Generation"

- **Setup:** Qwen3-4B-Instruct-2507; 444 LiveCodeBench tasks; probe on the hidden state at the FINAL PROMPT TOKEN before any output token; pipeline = standardization → PCA retaining 95% variance → ℓ2-regularized logistic regression (C grid 1e-3..10), nested CV over layer and C, 50 outer splits (§3.1–3.2).
- **Claims:**
  - Headline: "The correctness of the model's first-attempt code is linearly decodable from the hidden state at the final prompt token, captured before any output token is generated, with a leakage-free held-out AUC of 0.881 +/- 0.008 across 50 outer splits" (abstract; Table/results).
  - Prompt-length confound control: residualizing every hidden dimension against prompt length retains AUC **0.842 ± 0.010** vs a logistic prompt-length-only baseline of **0.657 ± 0.014** (abstract; §4).
  - Layer profile: mid-to-late layers plateau (29–36 of 36; peak 30).
  - "none of the nonlinear models tested improves upon it" — refers to the length-baseline models (§4), not to nonlinear hidden-state probes.
  - Repair-geometry companion question returned null for lack of data: only 14 successful repairs.
- **Limitations:** single model, single benchmark, first-attempt binary DV (not a rate); no cross-surface transfer.
- **Relation:** the strongest existing pre-generation code result; directly grounds our surface-4 expectation and donates the PCA→logistic recipe and the prompt-length/surface-feature baseline for arm 8.

### 5. arXiv 2406.05328 — FacLens

- **Setup:** "non-factuality prediction (NFP), predicting whether an LLM will generate a non-factual response prior to the response generation" (abstract); lightweight probe on hidden representations of the QUESTION (last input token; middle layers favored); PopQA, Entity Questions, NQ.
- **Claims:**
  - Ante-hoc beats post-hoc: FacLens outperforms post-generation detectors (SAPLMA, INSIDE) on their benchmarks (results tables).
  - AUC up to ~88–91 (PopQA/EQ), ~65–74 (NQ); matches or beats full fine-tuning (Kadavath-style Self-Evaluation) and LoRA at 0.016 s/question.
  - Cross-LLM transfer: "hidden question representations sourced from different LLMs exhibit similar NFP patterns, enabling the transferability of FacLens across different LLMs" (abstract) — implemented with unsupervised MMD domain adaptation over question-aligned mini-batches.
- **Limitations:** recall-flavored QA only; binary DV; no math/code/MCQ.
- **Relation:** H1-establishing for recall QA; its MMD transfer is the nearest existing "reduce labels via structure" device for probes (domain adaptation rather than a map prior) — relevant contrast for H2 framing.

### 6. arXiv 2406.12673 — KEEN

- **Setup:** "is it possible to estimate how knowledgeable a model is about a certain entity, only from its internal computation?" (abstract). Probe over internal SUBJECT-entity representations (last subject token, 3 layers around ¾ depth, min-max normalized, sigmoid-linear head, MSE); target = per-subject QA accuracy RATE and FActScore.
- **Claims:** correlation r 0.60–0.68 with per-subject QA accuracy and 0.66–0.77 with FActScore; an entity-popularity baseline reaches only ≤0.32/0.36; QA→open-ended-generation transfer 0.60–0.62; tracks fine-tuning-induced knowledge changes and hedging.
- **Limitations:** entity-grain (not per-question); recall QA only.
- **Relation:** pre-generation rate-target precedent at coarser grain; the popularity baseline is another arm-8 candidate; supports the knowledge-recall reading of what question-side probes measure (H3-relevant).

### 7. arXiv 2407.03282 — Ji et al., "LLM Internal States Reveal Hallucination Risk Faced With a Query"

- **Setup:** query-only (pre-generation) hallucination-risk estimation; probing estimator = gated MLP (Llama-style up/gate/down + SiLU) on last-query-token activations; 15 NLG task families, 700+ datasets; Llama2-7B primary.
- **Claims:** "achieving an average hallucination estimation accuracy of 84.32% at run time" (abstract); internal states separate seen-vs-unseen queries (80.28%); deep layers best; cross-task transfer is weak (e.g., QA→Translation F1 20.45, transfer tables).
- **Limitations:** accuracy metric (not AUROC) on balanced sets; nonlinear probe; binary DV.
- **Relation:** breadth evidence for H1 pre-generation across task families + explicit transfer-failure evidence feeding our shift-ladder expectations. Its nonlinear estimator is what our linear-by-default policy deliberately avoids as a roster default.

### 8. arXiv 2310.11877 — Slobodkin et al., answerability

- **Setup:** (un)answerable-query encoding in hidden states; probes at several positions; SQuAD 2.0 / NQ / musique-style answerability contrasts.
- **Claims:** "such models encode the answerability of an input query, with the representation of the first decoded token often being a strong indicator" (abstract); answerability subspace is "largely independent" of dataset (transfers across datasets, §results); linear probes suffice with ~400+400 training examples.
- **Limitations:** answerability ≠ correctness; first-decoded-token slightly beats last-prompt-token in their setting.
- **Relation:** supports a low-label linear read of a correctness-adjacent construct at/near the pre-generation position; the first-decoded-token superiority is a position caveat our position sweep (last prompt token, fixed) should note as a known alternative.

### 9. arXiv 2406.15927 — Semantic Entropy Probes (SEP)

- **Setup:** linear probes on hidden states trained to predict BINARIZED semantic entropy (computed from N=10 sampled generations at T=1) instead of accuracy labels — label-free with respect to gold answers; positions include TBG ("token before generation") and SLT (second-last token of the generation); multiple models/tasks.
- **Claims:**
  - "SEPs retain high performance for hallucination detection and generalize better to out-of-distribution data than previous probing methods that directly predict model accuracy" (abstract).
  - TBG position works: SEP performance at the token-before-generation position is "slightly below the SLT experiments" (§ position ablation) — a pre-generation read of a sampling-derived quantity.
  - But: SEPs "cannot match the performance of much more expensive sampling-based methods" (limitations/discussion).
- **Limitations:** the supervision target is entropy, not correctness; binarized.
- **Relation:** the closest existing "unlabeled sampling structure substitutes for gold labels" mechanism — H2-adjacent in SPIRIT (buy supervision without judged labels) but mechanistically different from a v_C→v_A map prior. Also the direct precedent that OUR pre-generation position carries sampling-statistics information.

### 10. arXiv 2505.24362 — "Knowing Before Saying"

- **Setup:** probing classifier on LLM representations predicting zero-shot CoT success; math + reasoning datasets; BERT-on-text baseline.
- **Claims:** "a probing classifier, based on LLM representations, performs well even before a single token is generated" (abstract); accuracy 60–76.4% on math tasks (results); beats the BERT text-only baseline, which "relies solely on the generated tokens."
- **Limitations:** accuracy metric, modest models; binary.
- **Relation:** H1 support on math at the pre-generation position — weaker than PIKA-style results but independent.

### 11. arXiv 2509.12886 — "The LLM Already Knows"

- **Setup:** difficulty = a learned value function V(s0) on the INITIAL hidden state, modeling token-level generation as a Markov chain; "efficient and accurate difficulty estimation based solely on the initial hidden state, without generating any output tokens" (abstract); guides Self-Consistency / Best-of-N / Self-Refine.
- **Claims:** outperforms sampling/auxiliary-model baselines on difficulty estimation across textual + multimodal tasks; enables adaptive inference with fewer generated tokens.
- **Limitations:** value function is a trained nonlinear head; "difficulty" is quality-expectation, not verified correctness on our surfaces.
- **Relation:** independent formalization of the pre-generation state as sufficient statistic for expected outcome — a Bellman-flavored cousin of H1's rate target.

### 12. arXiv 2510.18147 — "LLMs Encode How Difficult Problems Are"

- **Setup:** linear probes across layers and token positions on 60 models (Qwen2.5/Qwen3/Llama3.1/DeepSeek families + math-specialized variants); Easy2Hard-Bench math/coding subsets; 500 questions per probing set.
- **Claims:** "human-labeled difficulty is strongly linearly decodable (AMC: ρ≈0.88) and exhibits clear model-size scaling, whereas LLM-derived difficulty is substantially weaker and scales poorly" (abstract); steering toward "easier" reduces hallucination-like repetition and improves accuracy (Fig 1); during GRPO the human-difficulty probe strengthens while the LLM-difficulty probe degrades.
- **Limitations:** difficulty labels are item-level (IRT/human), not the probed model's own success rate; 500-question probe sets.
- **Relation:** linear decodability of item difficulty from question-side states — H1-adjacent; together with 2602.09924 it separates "human difficulty" from "this model's success probability", which is exactly the construct distinction our DV (own-model K-rollout rate) is designed to hit.

### 13. arXiv 2511.14773 — "Temporal Predictors of Outcome in Reasoning Language Models"

- **Setup:** linear classifiers on hidden states after the first t reasoning tokens; MATH, difficulty-balanced 750 easy / 750 hard; pooled last-4-token states at each prefix; PCA ≤128 dims + ℓ2 logistic.
- **Claims:** "eventual correctness is highly predictable after only a few tokens" (abstract); t=4 ROC-AUC 0.84 (results); the apparent late-prefix degradation is a selection artifact — "hard items are disproportionately represented in long CoTs" (abstract); notes question-only probes struggled on math in their setting.
- **Limitations:** t≥1 (not strictly pre-generation at t=0 for the headline); single surface.
- **Relation:** brackets H1-math from above (a few tokens in, AUC 0.84) and below (question-only harder) — consistent with math being the contested surface; PCA+logistic recipe again.

### 14. arXiv 2504.05419 — "Reasoning Models Know When They're Right"

- **Setup:** 2-layer MLP probes on hidden states at intermediate-answer positions within long CoT; R1-style reasoning models; math/logic.
- **Claims:** probe "can verify intermediate answers with high accuracy and produces highly calibrated scores" (abstract; ECE < 0.1 in results); "models' hidden states encode correctness of future answers, enabling early prediction of the correctness before the intermediate answer is fully formulated" (abstract); early-exit saves 24% tokens; probes degrade on short-CoT models (encoding appears acquired in long-CoT training).
- **Limitations:** reasoning models; MLP probe; positions inside generation.
- **Relation:** "look-ahead" correctness signal inside the trajectory — post-prompt but pre-answer; a contrast condition for our strictly pre-generation claim.

### 15. arXiv 2304.13734 — SAPLMA

- **Setup:** feedforward classifier on hidden activations of true/false STATEMENTS (read or generated); held-out-topic training.
- **Claims:** 71–83% accuracy across base models (abstract); "LLM-assigned sentence probability is related to sentence truthfulness, but this probability is also dependent on sentence length and the frequencies of words" (abstract) — the classic confound argument for internal-state over likelihood.
- **Limitations:** statement-truth, post-hoc; generalization criticized by 2307.00175.
- **Relation:** baseline lineage for internal-state-beats-likelihood; not pre-generation.

### 16. arXiv 2410.02707 — Orgad et al., "LLMs Know More Than They Show"

- **Setup:** truthfulness probes at multiple token positions incl. EXACT-ANSWER tokens, across error types and datasets.
- **Claims:**
  - "truthfulness information is concentrated in specific tokens" (abstract) — exact-answer tokens dominate.
  - "a strong truthfulness signal appears immediately after the prompt" (position analysis) — direct support for last-prompt-token reads.
  - Detectors "fail to generalize across datasets… truthfulness encoding is not universal but rather multifaceted" (abstract; the paper calls the encodings skill-specific).
  - Internal/external discrepancy: "they may encode the correct answer, yet consistently generate an incorrect one" (abstract).
  - Critique of pre-generation probing (related-work/discussion): last-prompt-token probing is "inherently inaccurate due to LLMs' unidirectional nature, failing to account for the generated response and missing cases where different sampled answers from the same model vary in correctness."
- **Limitations:** post-hoc headline results; binary labels.
- **Relation:** BOTH a support (signal right after the prompt) and the sharpest stated objection to H1's read-out position. Its "sampled answers vary in correctness" objection is answered by our K-rollout RATE target: the rate is exactly the quantity that remains well-defined pre-generation.

### 17. arXiv 2212.03827 — CCS

- **Setup:** unsupervised consistency-search over activations of yes/no statement pairs; 6 models, 10 QA datasets.
- **Claims:** "it outperforms zero-shot accuracy by 4% on average" with NO labels (abstract); transfers across tasks; maintains accuracy when models are prompted to answer incorrectly.
- **Limitations:** statement-level; later criticized (2307.00175) for failing basic generalization (e.g., negations).
- **Relation:** zero-label extreme of the label-efficiency axis; conceptual anchor that truth-adjacent directions are findable with little/no supervision.

### 18. arXiv 2602.08159 — Confidence Manifold

- **Setup:** geometry of correctness representations across 11 models (124M–14B); teacher-forced statements; steering/erasure/DAS causal tests; GroupKFold by question against paraphrase leakage.
- **Claims:** "two class centroids in a 2-8 dimensional subspace match a trained linear probe, and 25 labeled examples recover 90% of full-data AUC on GPT-2" (abstract); "Single-dataset probes transfer near-randomly until joint multi-dataset training restores 0.73-0.91 AUC" (abstract); internal advantage over P(True)/semantic entropy is regime-specific (adversarial misconceptions yes, standard QA tie).
- **Limitations:** statement-truth, teacher-forced; small models for the 25-label headline.
- **Relation:** the strongest existing label-efficiency datum for correctness-adjacent probes (25 labels → 90% of full AUC) — the number H2's crossover claim must beat or complement; its group-fold discipline matches ours.

### 19. arXiv 2608.17124 — CASE / decodability

- **Setup:** linear gate on ANSWER-token hidden states selecting among sampled candidates; question-grouped evaluation; general + medical LLMs.
- **Claims:** "A conventional probe appears accurate only because of question-identity leakage, which vanishes under question-grouped evaluation" (abstract; within-question AUC 0.502 on LogiQA under grouped splits); decodability (leakage-free within-question AUC) predicts selection-over-voting gain with r=0.75, decision threshold near AUC=0.60; "Decodability depends on the aligned knowledge a model must recall, not on its scale" (abstract).
- **Limitations:** post-generation (answer tokens); selection application.
- **Relation:** mechanically validates our GROUP-LEVEL fold requirement — random-split probe numbers on correctness DVs are inflated by question identity; any H1 headline must be read at the group grain. Also ties the signal to knowledge recall (H3-relevant).

### 20. arXiv 2505.21772 — CCPS

- **Setup:** "applies targeted adversarial perturbations to the final hidden states that generate an answer's tokens" (abstract) — post-generation; lightweight classifier on stability features; Llama/Qwen/Mistral 8B–32B; MMLU + MMLU-Pro, multiple-choice AND open-ended.
- **Claims:** "CCPS reduces Expected Calibration Error by approximately 55% and Brier score by 21%, while increasing accuracy by 5 percentage points, … and AUROC by 6 percentage points, all relative to the strongest prior method" (abstract).
- **Limitations:** requires the generated answer; perturbation machinery.
- **Relation:** the state of the art on OUR MCQ surface (MMLU-Pro) is post-generation — no pre-generation last-prompt-token probe on MMLU-Pro was found anywhere in this review. MCQ is the thinnest H1 surface and the likeliest novelty.

### 21. arXiv 2512.07404 — "On LLMs' Internal Representation of Code Correctness"

- **Setup:** RepE/LAT applied to code; "we identify a correctness representation inside LLMs by contrasting the hidden states between pairs of correct and incorrect code for the same programming tasks" (abstract); 4 LLMs; HumanEval + BigCodeBench.
- **Claims:** representation-based scores outperform "standard log-likelihood ranking, as well as verbalized model confidence" (abstract); ranking via the direction improves pass@1 by 21.3% avg (HumanEval) and 51.1% (BigCodeBench) without test execution (§results).
- **Limitations:** post-generation (reads the code's hidden states); within-task contrastive pairs need executed labels to construct.
- **Relation:** within-task contrastive direction construction is the direct precedent for our direction-mapped/direction-context arms (same-task contrast), on surface 4.

### 22. arXiv 2608.08266 — robustness follow-up to #21

- **Setup:** systematic variation of direction-construction method, prompt framing, hidden-state location; in-distribution (HumanEval, BigCodeBench) and OOD (fit MBPP+, test both); 4 instruction-tuned LLMs; controlled mutation/refactoring pairs isolating the fault.
- **Claims:** "no single configuration is best, and … isolating the fault does not help" (abstract); "the highest obtained accuracy ranged from 41–63% across models and benchmarks" for the prior method (intro); "Only the construction method generalises, while the best prompt framing, read-out location, and model change from one benchmark to the other" (intro/results).
- **Limitations:** code only; ranking-accuracy metric.
- **Relation:** disconfirming for configuration-robustness of contrastive correctness directions on code — argues for our pre-registered single recipe + nested selection rather than post-hoc configuration picking.

### 23. arXiv 2302.09664 — Kuhn et al., Semantic Uncertainty

- **Setup:** semantic entropy over M sampled generations, clustered by bidirectional NLI (DeBERTa-large-MNLI); free-form QA.
- **Claims:** "semantic entropy is more predictive of model accuracy on question answering data sets than comparable baselines" (abstract); sampling budget: "M<20 is often sufficient for good uncertainty" (§ experiments); entailment clustering agrees with humans 92.7–95.5% (appendix checks).
- **Limitations:** outputs-only; needs M samples at inference.
- **Relation:** inherited claim #2 — VERIFIED (see verification section). The outputs-only sampling cost is the efficiency contrast for pre-generation probes.

### 24. Farquhar et al., Nature 630:625–630 (2024)

- **Setup:** semantic entropy for confabulation detection; "We use ten generations to compute entropy" (Methods); TriviaQA, SQuAD, BioASQ, NQ-Open, SVAMP; multiple model families.
- **Claims:** "Averaged across the 30 combinations of tasks and models we study, semantic entropy achieves the best AUROC value of 0.790" (results) — vs embedding-regression and P(True)-style baselines.
- **Limitations:** outputs-only; 10-sample cost; AUROC ceiling ~0.79 avg.
- **Relation:** inherited claim #3 — VERIFIED. Sets the outputs-only reference band our probes are compared against conceptually (not head-to-head in our design).

### 25. arXiv 2510.09033 — "Internal States Mainly Reflect Knowledge Recall Rather Than Truthfulness"

- **Setup:** taxonomy of hallucinations into Unassociated (UH) vs Associated (AH, driven by spurious parametric associations); mechanistic comparison of hidden-state geometry; LLaMA-family models.
- **Claims:** "hidden states primarily reflect whether the model is recalling parametric knowledge rather than the truthfulness of the output itself" (abstract); AHs detected at AUROC ≈0.48–0.69 vs UHs 0.86–0.93 (results; LLaMA) — AHs "exhibit hidden-state geometries that largely overlap with factual outputs."
- **Limitations:** recall-QA scope.
- **Relation:** the sharpest construct-validity threat to H1 on recall QA: question-side probes may read recall-likelihood, not correctness. Our H3 (knowledge-vs-persona decomposition) and the math/code surfaces (where recall is not the mechanism) are the design answers.

### 26. arXiv 2307.00175 — "Still No Lie Detector for Language Models"

- **Setup:** empirical + conceptual audit of Azaria-Mitchell (SAPLMA) and Burns (CCS).
- **Claims:** "these methods fail to generalize in very basic ways" (abstract) — e.g., probes trained without negations fail on negations; conceptual argument that consistency conditions underdetermine truth.
- **Relation:** disconfirming lineage for statement-truth probes; motivates our shift ladder + group folds instead of single-split claims.

### 27. arXiv 2604.12373 — "Masked by Consensus"

- **Setup:** correctness classifiers on QUESTION representations from a model's own hidden states vs an EXTERNAL model's; random vs disagreement subsets; models incl. Qwen-2.5-7B; factual (TriviaQA, Mintaka, HotPotQA) vs math (GSM1K, MATH).
- **Claims:** "On standard evaluation, we find no advantage: self-probes perform comparably to peer-model probes" (abstract) — inter-model difficulty agreement confound; on disagreement subsets, "self-representations consistently outperform peer representations in factual knowledge tasks, but show no advantage in math reasoning" (abstract; ~5% factual advantage, none in math at any depth).
- **Limitations:** binary labels; disagreement subsets are small.
- **Relation:** decomposes WHAT the pre-generation probe reads: shared item difficulty (peer-readable) + model-specific privileged knowledge (factual domains only). Our external-embedding baseline (arm 8) operationalizes exactly the peer/assessor contrast; predicts our math H1 signal, if present, is item-difficulty-flavored rather than privileged.

### 28. arXiv 2311.04897 — Future Lens

- **Setup:** GPT-J-6B; "Given a hidden (internal) representation of a single token at position t … can we reliably anticipate the tokens that will appear at positions ≥ t+2?" (abstract); trains LINEAR models mapping h_t at layer l to future hidden states, decoded through the pretrained head; also causal patching.
- **Claims:** "at some layers, we can approximate a model's output with more than 48% accuracy" several tokens ahead from a single state (abstract).
- **Limitations:** interpretability visualization; no correctness DV; single model.
- **Relation:** the closest STRUCTURAL relative of our M map — a learned linear map from a present state to future-answer-bearing states — but used to reveal future tokens, never as a prior for a correctness probe, and never at matched label budgets. Together with FIRP (excluded, speculative decoding), it establishes that v_C→v_A-like linear maps are learnable, which is the feasibility premise of H2 — not its test.

### 29. arXiv 2511.07318 — "When Bias Pretends to Be Truth" (coverage round; abstract-level)

- **Claims (abstract):** spurious-correlation-driven hallucinations "are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning"; "existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correlations."
- **Relation:** independent corroboration of 2510.09033's AH class from a training-data angle; bounds the ceiling of ANY internal-state correctness read on recall QA.

### 30. arXiv 2604.13068 — "Detection Without Correction" (coverage round; abstract-level)

- **Claims (abstract):** across 7 models (117M–7B, incl. Qwen-2.5): "output-confidence baselines outperform activation probes on raw detection AUC at every model above 410M parameters"; steering along probe directions fails to correct in 7/7; "The probe's distinguishing value is therefore not detection accuracy but temporal positioning: probe signals are accessible at position zero (before any output tokens are produced), enabling pre-generation flagging that output-based methods structurally cannot provide"; the position-zero temporal signal is statistically significant in 2 of 7 models, one of which is **Qwen2.5-7B (p = 0.038)**.
- **Relation:** a clean negative-result framing that lands almost exactly on our H1: the unique value of the last-prompt-token read is WHEN it is available, and output-confidence baselines must be beaten (or the comparison disclosed) before claiming probe superiority. Their Qwen2.5-7B position-zero significance is directly encouraging for our model choice; their base-vs-instruct null (absent in base Pythia-6.9B) cautions that the signal may be tuning-dependent.

---

## Verification of the three inherited claims (task body `## Motivation`)

1. **Kadavath et al. (arXiv 2207.05221).** Body claim: models can be trained to predict whether they will answer correctly, pre-generation, with the ground truth being the fraction of sampled answers that are correct. VERIFIED against the paper: "Ground Truth P(IK) – The fraction of unit temperature samples to a question that are correct" (30 samples, T=1); the P(IK) head reads the question's last token; TriviaQA AUROC 0.864 at 52B with partial transfer (GSM8K 0.624). One phrasing note: the body's "full [0,1] range" wording is its own gloss of the paper's ground-truth P(IK) construction (cf. the paper's Figure 12 histograms) — substantively accurate, not a quotation. No mismatch.
2. **Kuhn et al. (arXiv 2302.09664).** Body positions semantic entropy as the sampling-based outputs-only lineage. VERIFIED: "semantic entropy is more predictive of model accuracy on question answering data sets than comparable baselines" (abstract); M<20 samples suffice; NLI clustering ~93–95% agreement with manual clustering. No mismatch.
3. **Farquhar et al. (Nature 2024).** Body cites it as the outputs-only reference for hallucination detection. VERIFIED via the Nature article page: "Averaged across the 30 combinations of tasks and models we study, semantic entropy achieves the best AUROC value of 0.790"; "We use ten generations to compute entropy"; datasets TriviaQA/SQuAD/BioASQ/NQ-Open/SVAMP; no internal-state component. No mismatch.

---

## Synthesis

**The pre-generation line is real, active, and converging on our exact formalization.** Kadavath (2022) founded it: a rate target (fraction of T=1 samples correct) predicted from the question's last token. The 2025–2026 wave sharpened it into frozen-model linear probes: question-only difference-of-means directions transfer across knowledge datasets but fail on math (2509.10625, incl. Qwen2.5-7B-Instruct rows); supervised pre-generation probes regress the K-rollout success rate itself on math and code and beat surface baselines (2602.09924); code first-attempt correctness is linearly decodable at the final prompt token at AUC 0.881, surviving prompt-length residualization at 0.842 vs 0.657 for length alone (2606.14530); recall-QA non-factuality is predictable ante-hoc and beats post-hoc detectors (2406.05328, 2407.03282, 2406.12673). The position itself is validated from the post-hoc side: a strong truthfulness signal appears immediately after the prompt (2410.02707), and semantic-entropy probes work at the token-before-generation position (2406.15927).

**The critical literature localizes exactly the confounds our design already carries controls for.** (i) Question-identity leakage inflates random-split probe AUCs — within-question AUC can collapse to chance under grouped evaluation (2608.17124) → group-level folds are mandatory, and ours are. (ii) On recall QA the signal is substantially knowledge-recall, not truthfulness: associated hallucinations are near-undetectable (2510.09033; 2511.07318) → H1-recall-QA headlines need the H3 decomposition. (iii) Self-vs-peer probes tie on random samples because inter-model difficulty agreement dominates; privileged self-knowledge exists only in factual domains and NOT in math (2604.12373) → the external-embedding/assessor baseline (arm 8) is load-bearing, not decorative. (iv) Output-confidence baselines can beat activation probes on raw AUC; the probe's unique value is temporal positioning (2604.13068) → report the comparison, claim the position. (v) Single-dataset probe directions are dataset-specific at low n and transfer near-randomly until multi-dataset training (2602.08159; 2509.10625 orthogonality) → our shift ladder rungs are the right instrument. (vi) On code, configuration choices don't generalize (2608.08266) → pre-register one recipe, select inside nested CV.

**Label efficiency has data points but no mechanism like H2.** Existing anchors: 25 labels recover 90% of full-data AUC on GPT-2 statement-truth (2602.08159); 160 samples give robust question-only performance and 2,560 match 48,540 (2509.10625); SEPs substitute sampling-derived entropy labels for gold labels at slight cost (2406.15927); FacLens transfers probes across LLMs with unsupervised MMD (2406.05328); CCS needs zero labels (2212.03827). None of these is a learned map between representation spaces used as a prior. The only learned v_present→v_future linear maps in the literature (Future Lens 2311.04897; FIRP, excluded) are interpretability/speculative-decoding devices with no correctness DV and no label-budget analysis.

**The gap #2388 sits in:** (a) no pre-generation last-prompt-token correctness probe exists on MMLU-Pro (MCQ surface) — the nearest work is post-generation (CCPS); (b) no work fits the K-rollout rate DV on Qwen2.5-7B-Instruct across four surfaces under one protocol with group folds and a shift ladder; (c) H2 — the context→answer map fit on unjudged pairs as a label-efficiency prior — has NO published analogue; empty (c) is itself a finding of this review.

---

## For the planner

### (a) Closest prior formalizations (pre-/post-generation marked)

- **PRE:** Kadavath 2207.05221 — P(IK) value head at question last token; ground truth = fraction of 30 T=1 samples correct (rate target, ours = K=5).
- **PRE:** 2509.10625 — question-only difference-of-means linear direction at last question token; binary label, single T=0 sample; includes Qwen2.5-7B-Instruct.
- **PRE:** 2602.09924 — supervised linear probes on pre-generation activations regressing the K-rollout expected success rate (K=50/K=5); math + code; the exact-DV match.
- **PRE:** 2606.14530 — final-prompt-token PCA-95%→ℓ2-logistic probe, code, AUC 0.881; prompt-length residualization control.
- **PRE:** 2406.05328 (FacLens), 2406.12673 (KEEN), 2407.03282 (Ji), 2310.11877 (answerability), 2406.15927 (SEP at TBG), 2505.24362 (CoT success), 2509.12886 (V(s0) difficulty), 2510.18147 (item difficulty).
- **POST (closest baselines/contrast):** 2410.02707 (exact-answer tokens; the stated objection to pre-generation reads, answered by the rate DV), 2304.13734 (SAPLMA), 2212.03827 (CCS), 2602.08159 (Confidence Manifold), 2608.17124 (CASE; group-fold mandate), 2505.21772 (CCPS on MMLU/MMLU-Pro), 2512.07404 + 2608.08266 (code correctness direction + its robustness critique), 2504.05419 / 2511.14773 (inside-trajectory positions).

### (b) Is H1 established, per surface? (testing the body's "recall QA settled; math and code genuinely open")

- **Short-answer recall QA: ESTABLISHED** pre-generation (Kadavath 0.864 in-domain; 2509.10625 0.758–0.842 across knowledge sets on our model; FacLens/KEEN/Ji), with two standing caveats that survive into any headline: the signal is substantially knowledge-recall rather than truthfulness (2510.09033), and self-vs-external advantage appears only on disagreement subsets (2604.12373).
- **Math: MORE ESTABLISHED than the body assumed, but not on our protocol.** In-domain supervised pre-generation probes reach AUROC 0.68–0.85 binary / ρ 0.40–0.64 on the rate (2602.09924); CoT-success probes 60–76.4% accuracy (2505.24362); a-few-tokens-in AUC 0.84 (2511.14773). What FAILS is cross-domain transfer INTO math (2509.10625: GSM8K 0.601 on Qwen2.5-7B-Instruct) and privileged self-knowledge in math (2604.12373: none at any depth). Our exact cell — K=5 T=1.0 rate on MATH with Qwen2.5-7B-Instruct, group folds — is unmeasured.
- **Code: ESTABLISHED pre-generation** in two independent settings (2606.14530: AUC 0.881 first-attempt, Qwen3-4B, LiveCodeBench; 2602.09924: pass@5 AUROC 0.90–0.91, Qwen2.5-Coder) — but single-model each, and the contrastive-direction variant is configuration-fragile (2608.08266: 41–63%, "no single configuration is best"). Our model and rate DV remain unmeasured.
- **MCQ (MMLU-Pro): OPEN — the thinnest surface.** No pre-generation last-prompt-token probe on MMLU-Pro (or MMLU) was found; the state of the art there is post-generation (CCPS 2505.21772). This is the likeliest per-surface novelty of H1.
- Net: the body's assertion is DIRECTIONALLY right for code-vs-QA maturity but understated for math (in-domain pre-generation math probes exist) and missed that MCQ, not math, is the emptiest surface.

### (c) Does anything resembling H2 (map as label-efficiency prior) exist?

**No.** Repeated targeted searches (queries 12, 16, 17 above; plus the snowball) surfaced nothing that fits a learned map between a context/question representation and an answer representation on UNJUDGED pairs and uses it as a prior/feature to reduce labeled-probe budget. Closest relatives, each missing a limb: Future Lens 2311.04897 and FIRP 2410.20488 (linear present→future state maps; no correctness DV, no label-budget analysis); SEP 2406.15927 (unlabeled samples substitute for gold labels — different mechanism: sampling-derived supervision, not a representation map); FacLens 2406.05328 (MMD domain adaptation across LLMs — transfer, not a map prior); CCS 2212.03827 (zero-label consistency search). The H2 arm appears novel; the empty result is itself a reportable finding, and the review gives the comparison set H2 must beat: the 25-label anchor (2602.08159) and the 160/2,560-label curve (2509.10625).

### (d) Inheritable hyperparameters / recipes (plan §11 `Source:` grammar)

- Label-budget grid anchors L ∈ {~25, 160, 2560}: `Source: arXiv 2602.08159 (abstract: 25 labels → 90% full-data AUC)` and `Source: arXiv 2509.10625 §4.4/Fig 5 (160 robust; 2,560 ≈ full 48,540)`.
- Read-out layer: sweep all layers, select on validation inside nested CV; expect mid-to-late plateau: `Source: arXiv 2606.14530 §3.2 (layers 29–36 of 36, peak 30; nested CV over layer+C)`; `Source: arXiv 2509.10625 §4.3 (predictive power saturates in intermediate layers)`.
- Token position: last prompt token primary; optional robustness sweep over the final chat-template tokens: `Source: arXiv 2602.09924 §3.2 (position sweep over final P non-padding tokens)`; first-decoded-token caveat: `Source: arXiv 2310.11877 (first decoded token often strongest for answerability)`.
- Probe recipe (classification legs): standardization → PCA (95% variance or ≤128 dims) → ℓ2 logistic, C/α selected in inner CV: `Source: arXiv 2606.14530 §3.1`; `Source: arXiv 2511.14773 (PCA ≤128 + ℓ2 logistic)`. Rate-regression legs: linear regression with MSE on the K-rollout rate: `Source: arXiv 2602.09924 §3 (MSE linear regression on ŝ_MC)`.
- Folds: GROUP-level by question, never pointwise: `Source: arXiv 2608.17124 (question-identity leakage; within-question AUC 0.502 under grouped splits)`; `Source: arXiv 2602.08159 (GroupKFold by question against paraphrase leakage)`.
- Baselines for arm 8 (all appear as literature baselines that internal probes must beat): prompt-length logistic `Source: arXiv 2606.14530 (0.657 length-only)`; question length + TF-IDF `Source: arXiv 2602.09924 (abstract)`; entity popularity `Source: arXiv 2406.12673 (Table 2 ≤0.32/0.36)`; external-embedding assessor `Source: arXiv 2509.10625 §4.2` with the peer-probe framing `Source: arXiv 2604.12373`; output-confidence baseline disclosure `Source: arXiv 2604.13068 (abstract)`.
- Rollout count for the rate DV: K=5 has direct precedent: `Source: arXiv 2602.09924 (K=5 for GPT-OSS; K=50 small models)`; K=30 historical `Source: arXiv 2207.05221 (30 unit-temperature samples)`; sampling-cost context M<20 `Source: arXiv 2302.09664`, 10 generations `Source: Farquhar Nature 2024 (Methods)`.
- Expected in-domain performance bands (for power/sanity, not gates): recall QA AUROC ~0.75–0.86 (2509.10625 Qwen rows; 2207.05221); math binary 0.68–0.85 / rate ρ 0.40–0.64 (2602.09924); code 0.88–0.91 (2606.14530, 2602.09924); MCQ unknown (no precedent).

---

## Verification log (Step 7)

1. **Resolution (7.1):** all 27 included arXiv ids re-resolved via `get_abstract` AFTER the mid-session compaction, each with title matching the note (calls logged this session, 2026-08-19); Farquhar resolved via the Nature article page (WebFetch, redirect chain followed). The two coverage-round ids (2511.07318, 2604.13068) resolved via the `search_papers` result carrying title+abstract. Excluded-list ids were resolved at screening time (pre-compaction `get_abstract`/search results); five of them (2501.12934, 2607.07670, 2509.23782, 2607.05188, 2608.08024) carry coarser-grain reasons reconstructed from the compressed screening log — disclosed inline above.
2. **Claim-vs-source (7.2):** every synthesis and planner claim traces to a per-paper note quote above. Quotes for the four papers read after compaction (2510.18147, 2505.21772, 2512.07404, 2608.08266) and all abstract quotes were checked directly against tool output in-context this turn. Quotes for the 24 papers read before compaction were extracted verbatim at read time; several load-bearing numbers were independently re-confirmed post-compaction against the freshly resolved abstracts: 0.881/0.842/0.657 and the nested-CV recipe (2606.14530 abstract), math-generalization failure (2509.10625 abstract), 25-labels/2–8-dim/transfer-near-random (2602.08159 abstract), K-rollout rate target + E2H-AMC + 70% routing + surface-baseline claims (2602.09924 abstract), AH/UH recall-not-truthfulness (2510.09033 abstract), no-math-privilege (2604.12373 abstract), question-identity leakage + r=0.75 + AUC 0.60 threshold (2608.17124 abstract), 84.32% (2407.03282 abstract), 71–83% + length/frequency confound (2304.13734 abstract), +4% zero-shot (2212.03827 abstract), 48% (2311.04897 abstract), SEP OOD claim (2406.15927 abstract), KEEN construct (2406.12673 abstract), FacLens transferability (2406.05328 abstract), CCPS numbers (2505.21772 abstract), 2510.18147 ρ≈0.88 (abstract), 2511.14773 selection artifact (abstract), 2504.05419 early-prediction (abstract), 2505.24362 before-a-single-token (abstract), 2509.12886 initial-hidden-state (abstract), Orgad multifaceted/encode-correct-generate-wrong (2410.02707 abstract). Section-number citations for pre-compaction reads (e.g., "§4.4/Fig 5") were recorded at read time and are carried as logged.
3. **Disconfirming coverage (7.3):** one explicit negative-results round (query 24) ran after the included list froze; it surfaced 2511.07318 and 2604.13068, both folded in above as abstract-level entries. The review's disconfirming set spans six independent critique axes: recall-not-truthfulness (2510.09033, 2511.07318), generalization failure (2307.00175, 2410.02707, 2509.10625-math), question-identity leakage (2608.17124), no privileged self-knowledge in math (2604.12373), output-baseline superiority on raw AUC (2604.13068), and configuration fragility on code (2608.08266). No candidate pile was truncated: every reuse-relevant hit from queries 1–24 and the snowball is accounted for in the included or excluded lists.

## Ambiguities and assumptions (named per the brief; no user channel)

1. The brief's inclusion criterion 1 admits sampled-output-statistics methods; semantic-entropy papers were included only as the outputs-only baseline lineage the body already inherits (Kuhn, Farquhar, SEP) rather than swept exhaustively — assumption: the question's center of mass is internal-state methods, and the outputs-only line is context.
2. "PIKA" was used in working notes as shorthand for 2602.09924; the paper's resolved title is "LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations" — the artifact uses the title, not the shorthand.
3. The two coverage-round additions are abstract-level by construction (they surfaced in Step 7.3 after the read phase); their claims are quoted from abstracts only and marked as such.
