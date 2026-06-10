# ICL vs fine-tuning on the same examples — verified lit synthesis

Compiled 2026-06-10 from a deep-research sweep (5 search angles, 21 primary sources fetched,
104 claims extracted, 25 adversarially verified by 3-vote refutation panels, 2 killed).
Companion to #491 (ICL vs finetuning equivalence, proposed), #489 (leakage prediction extends
to ICL contexts), and the rank-one leakage model note (`docs/notes/rank1_leakage_model.tex`).

**The question.** When you give a model N examples in-context vs fine-tune it on those same N
examples: are the two operations equivalent, where do they diverge, do they move the same
internal representations, and — the project's question — can the model's in-context behavior on
a training set forward-predict what fine-tuning on it will do (which behaviors generalize/leak
to which contexts)?

---

## Verdict in one paragraph

The ICL-as-implicit-gradient-descent equivalence is a real theorem only in linear toy settings;
for real pretrained LLMs it is an explicitly open hypothesis whose original supporting evidence
is methodologically undermined (untrained models pass the same similarity metrics; ICL and GD
differ in order sensitivity, output-distribution effects, and layer-causal information flow).
Empirical head-to-heads on the same examples split by task type: ICL generalizes relational
inferences (reversals, syllogisms) far more flexibly than fine-tuning, while fine-tuning matches
or beats ICL out-of-distribution on classification at scale. The closest predictability
evidence is indirect but encouraging: ICL post-hoc recovers >85% of fine-tuning's
prediction-level corrections in-distribution (Rec2FTP), and adding the model's own in-context
inferences to the fine-tuning set transfers ICL's generalization into the weights (augmented
FT) — i.e., the model's in-context behavior on the training examples carries actionable
information about what fine-tuning on them will and won't generalize. **Nothing in the verified
set does forward prediction of fine-tuning side effects or behavior leakage from base-model ICL
— the project's exact question is unclaimed.**

---

## Findings (verified, with confidence)

### 1. Theory: the equivalence is linear-toy-setting only — HIGH

- **von Oswald et al. / [arXiv:2212.07677](https://arxiv.org/abs/2212.07677)**: explicit weight
  construction showing one linear self-attention layer can exactly implement one GD step on a
  regression loss; trained transformers match the construction. **Akyürek et al. /
  [arXiv:2211.15661](https://arxiv.org/abs/2211.15661)**: transformers provably implement GD and
  closed-form ridge regression for linear models; which algorithm they match is regime-dependent
  (depth/noise/width). Both explicitly scoped to the linear case. None of this establishes
  ICL = fine-tuning for pretrained LLMs on natural-language behavior data. (3-0 across five
  merged claims.)
- **Dai et al. / [arXiv:2212.10559](https://arxiv.org/abs/2212.10559)** frame ICL as "implicit
  finetuning" via a dual form between attention and GD — but the duality is exact only for
  LINEAR attention (softmax attention is a relaxed approximation), and this paper is the
  specific target of the later metric-validity critiques. (3-0.)

### 2. For real pretrained LLMs, ICL ≠ GD by default — HIGH (two independent critiques)

- **Shen et al. / [arXiv:2310.08540](https://arxiv.org/abs/2310.08540)** (ICML 2024): prior
  equivalence works rest on limiting assumptions; ICL and GD have different demonstration-order
  sensitivity; on naturally-pretrained LLaMA-7B they modify the output distribution differently
  and inconsistently across datasets/models/shot counts. Verbatim conclusion: "the equivalence
  between ICL and GD remains an open hypothesis."
- **Deutch et al. / [arXiv:2311.07772](https://arxiv.org/abs/2311.07772)** (NAACL 2024): the
  Dai-style similarity metrics (SimAOU/SimAM) are flawed — even untrained random-weight models
  achieve comparable ICL-GD similarity scores despite exhibiting no ICL.
- **Layer causality (same paper, MEDIUM, 3-0):** under ICL the "update" to layer $\ell$ depends
  only on lower layers, whereas vanilla GD updates flow through the whole model — a structural
  mismatch in how the two move internal representations. A layer-causal GD variant substantially
  improves ICL-GD similarity (absolute similarity stays low for both).
- **Project implication:** base-model in-context behavior on a set of examples is NOT, by
  default, a faithful mirror of what gradient training on those examples does.

### 3. But a weak functional form survives — MEDIUM

- **Zhou, Frank & McCoy / [arXiv:2406.18501](https://arxiv.org/abs/2406.18501)** (NAACL 2025):
  LLMs show the inverse frequency effect in ICL-simulated structural priming (improbable primes
  shift behavior more than probable ones) — the psycholinguistic signature of error-driven
  learning. ICL is "a type of error-driven learning" with an implicitly computed error signal in
  the forward pass; behavioral, not mechanistic; stronger in larger models. Independent
  corroboration: [arXiv:2406.04847](https://arxiv.org/abs/2406.04847) (prime-surprisal-scaled
  priming). Note the resonance with our P3/residual findings: surprisal-scaled adaptation is
  exactly residual scaling.

### 4. Head-to-heads on the same data: direction of the gap is task-dependent — MEDIUM

- **Lampinen et al. (Google DeepMind) / [arXiv:2505.00661](https://arxiv.org/abs/2505.00661)**,
  the cleanest data-matched comparison (same controlled synthetic factual corpora in-context vs
  fine-tuned; Gemini 1.5 Flash): ICL generalizes reversals and syllogisms far better than
  fine-tuning — FT replicates the reversal curse (near zero on reversals) while ICL over the
  same data is near ceiling. **Augmented FT**: adding the model's own in-context
  inferences/reasoning traces over the training data to the fine-tuning set transfers ICL's
  generalization into the weights, often beating plain ICL. Caveat in the paper: FT *can*
  generalize to reversals embedded in a larger coherent knowledge structure. Single lab, single
  model family.
- **Mosbach et al. / [arXiv:2305.16938](https://arxiv.org/abs/2305.16938)** (Findings of ACL
  2023), the counterweight: for the largest OPT models (6.7B/13B/30B) few-shot fine-tuned on
  MNLI, OOD performance (HANS) is on par with or better than in-domain, and fine-tuned models
  generalize OOD as well as or better than ICL. Caveats: lexical-overlap HANS subset, binarized
  MNLI, 16-shot pattern-based FT, large run variance.
- **Liu et al. (T-Few) / [arXiv:2205.05638](https://arxiv.org/abs/2205.05638)** (NeurIPS 2022,
  2-1 split vote): (IA)³ PEFT on T0-11B with the same *number* of few-shot examples beats GPT-3
  175B ICL on held-out T0 tasks (72.4% vs 66.6%). Cross-model comparison — conflates base-model
  identity with the method effect; within-model FT gain over T0 zero-shot is 5.5 points.

### 5. Predictability: the two strongest (indirect) hooks — MEDIUM

- **Rec2FTP (Dai et al., [arXiv:2212.10559](https://arxiv.org/abs/2212.10559); 2-1 split
  vote):** on six classification tasks (GPT 1.3B/2.7B, ≤32 examples), ICL on the same examples
  correctly predicts >85% of the examples that fine-tuning corrects relative to zero-shot
  (85.6% / 89.4%). Heavily scoped: post-hoc, prediction-level, in-distribution, nonstandard
  restricted FT variant, no base-rate control. The mechanistic metrics from the same paper did
  NOT survive (finding 2); this prediction-level overlap explicitly did.
- **Augmented FT (Lampinen et al.):** the model's in-context inferences both *anticipate* the
  fine-tuning generalization surface (what FT will miss) and *repair* it when added to the
  training set. Direct evidence that in-context behavior on the training examples carries
  actionable information about fine-tuning outcomes — the predictor version (use ICL responses
  to forecast FT effects rather than to augment training) is unbuilt.

### 6. Mechanistic bridge, positive direction: demo effects compress to a vector — MEDIUM (2-1)

- **In-Context Vectors (Liu et al.) / [arXiv:2311.06668](https://arxiv.org/abs/2311.06668)**
  (ICML 2024): the first principal component of last-token latent-state differences
  $h(y_i)-h(x_i)$ over demonstration pairs, added to latent states, REPLACES the demonstrations
  — and outperforms both ICL and fine-tuning on safety, style transfer, role-playing, and
  formatting. For broad behavioral/stylistic adaptations, ICL's effect compresses to a single
  steering-vector-like direction — structurally the same intervention class as
  fine-tuning-derived steering vectors. No verified work directly compares the two directions.
  Scope caveat: behavioral/stylistic tasks; task-vector mechanisms fail on high-rank
  input-output mappings ([arXiv:2506.09048](https://arxiv.org/abs/2506.09048)).

---

## Refuted claims — do NOT cite (killed by 3-vote adversarial verification)

1. **Mesa-optimizer framing of von Oswald et al. (0-3).** "Trained transformers act as
   mesa-optimizers whose forward pass performs inner gradient descent" did not survive as a
   claim about real models — only the linear-setting construction stands.
2. **"ICL's OOD advantage was a model-size artifact" framing of Mosbach et al. (1-2).** Only
   the scoped OPT/MNLI/HANS empirical result survives, not the general debunking framing.

---

## What is NOT established (gaps) — the project's opportunity space

- **(a) Forward prediction of fine-tuning outcomes from base-model ICL.** No verified work
  prospectively predicts generalization scope, side effects, or behavior leakage from the
  model's in-context responses to the training examples. Rec2FTP is post-hoc/in-distribution;
  augmented FT uses ICL outputs as training data, not as a predictor. **This is exactly the
  project's question (predict LoRA behavior leakage on Qwen-2.5-7B from base-model ICL) — open
  prior-art space as of 2026-06-10.**
- **(b) Whether ICL and FT move the SAME internal directions.** The layer-causality result
  points the other way; no verified claim connects in-context task/function vectors (Hendel
  [arXiv:2310.15916](https://arxiv.org/abs/2310.15916), Todd
  [arXiv:2310.15213](https://arxiv.org/abs/2310.15213) — no surviving claims this round) to
  fine-tuning-induced steering directions on matched tasks.
- **(c) ICL-vs-FT for persona/system-prompt behavior specifically.** The ICV
  safety/role-play result is the nearest neighbor; nothing does prompted-persona vs trained-in
  persona head-to-head.
- **(d) ICL = GD for real pretrained LLMs** — explicitly open per Shen et al.
- **(e) Anything on Qwen-2.5-7B or LoRA.** Verified head-to-heads use Gemini, OPT, LLaMA-7B,
  GPT-3, (IA)³.

---

## Caveats on this synthesis

- "HIGH" requires 2+ independent primary sources with unanimous votes; single-paper findings
  are "MEDIUM" even when peer-reviewed and verbatim-verified (Lampinen, Mosbach in particular
  are solid but single-lab, single-model-family).
- Three findings rest on 2-1 split votes (Rec2FTP, T-Few, ICV) — carry their scope caveats
  verbatim when citing.
- **Coverage, not refutation:** Hendel/Todd function-vector papers, Ovadia et al. knowledge
  injection ([arXiv:2312.05934](https://arxiv.org/abs/2312.05934)), and prompted-vs-trained
  persona work produced no surviving verified claims this round; their absence reflects
  verification budget, not evidence against them.
- The positive equivalence theory is 2022-2023 toy-setting work; the critiques are 2023-2024;
  Lampinen et al. (revised late 2025) is the most current head-to-head. The field is moving.

### Adjacent project-catalog papers (NOT verified this round; from `docs/papers.md`)

- **Dherin et al. / [arXiv:2507.16003](https://arxiv.org/abs/2507.16003)** (queued,
  load-bearing): proves a self-attention + MLP block converts context into a **transient rank-1
  weight patch on the MLP** — "data in context = a transient weight update."
  **Goldwaser et al. / [arXiv:2511.17864](https://arxiv.org/abs/2511.17864)** extends to
  Qwen-style blocks. If verified, this is the cleanest bridge between #491 and the rank-one
  leakage model: ICL and fine-tuning become the same object (a rank-one update), differing in
  transient vs permanent.
- **Bigelow et al. / [arXiv:2511.00617](https://arxiv.org/abs/2511.00617)**: ICL and activation
  steering as one Bayesian family (steering shifts priors, ICL accumulates evidence; additive in
  log-belief space).

---

## Open questions carried forward

- Can base-model ICL responses to a LoRA training set forward-predict which implanted behaviors
  generalize/leak to which contexts? (Gap (a); the natural design: generate base-model
  ICL-conditioned responses on the training mix, score the target behavior per bystander
  context, correlate with measured post-training leakage — an ICL analogue of the base-prior
  predictor that already beats geometry in #532.)
- Does Lampinen's augmented-FT mechanism invert into a cheap predictor — do the model's
  in-context inferences over a training mix enumerate the fine-tuning generalization surface,
  including unwanted leakage?
- Do in-context task/function vectors align with fine-tuning weight-delta / steering directions
  for the same task (do ICL and FT write to the same low-rank subspace)? Directly testable on
  our stored adapters + ICV-style extractions; connects to P0 of the rank-one note.
- Does the layer-causality mismatch hold for LoRA specifically — do low-rank adapter updates on
  attention projections look more or less ICL-like than full fine-tuning?
- #491's design should stratify by task type: the literature predicts ICL > FT on relational
  inference, FT ≥ ICL on classification-style behavior at scale — a persona-trait implant may
  sit in between, which is itself informative.
