# Literature review: how a behavioral instruction shifts a context's activation-space representation

Background for the experiment *"Context-vector shift induced by appending a behavioral
instruction"*. Question: given a **context vector** — an activation-space summary
(mean / last-token residual-stream activation) of a system prompt / persona / scenario —
how does that vector move when a behavioral instruction (*"be sycophantic"*, *"refuse"*,
*"be evil"*) is appended? Is the shift a **single, context-independent direction** (a
"behavior direction" the same across many contexts)? How does it depend on behavior,
context, and layer, and how does it relate to fine-tuning-induced representational change
and to leakage?

The literature has heavily formalized two of the three ingredients — (a) "a behavior /
trait / instruction collapses to a roughly linear residual-stream direction," and (b) "an
in-context instruction collapses to a single activation-space vector" — but has **not**
cleanly measured the prompt-induced delta of a *context* representation, as a function of
the appended behavior, across many contexts. That is the cell this experiment fills.

## Closest prior formalizations (paper · arXiv · exact construct)

- **Persona vectors** — Chen, Arditi, Sleight, Evans, Lindsey (Anthropic, 2025),
  **arXiv:2507.21509**. A per-trait direction = **difference-in-means of residual-stream
  activations** between trait-exhibiting and non-trait responses, where the two response
  sets are generated from **contrastive system-prompt pairs** (a trait-eliciting vs a
  trait-suppressing system prompt), averaged over **response tokens**. Reports that (i)
  prompting the trait moves activations *along* the persona vector, and (ii)
  **fine-tuning-induced personality changes correlate with shifts along the relevant persona
  vector**. Uses Qwen-2.5-7B-Instruct and Llama-3.1-8B — our model family. Single closest
  paper; links prompted shift, fine-tuning shift, and a single trait direction, but does not
  study the per-context shift geometry (consistency across many different system prompts).
- **Persona Features Control Emergent Misalignment** — Wang, Dupré la Tour, Watkins, …,
  Mossing (OpenAI, 2025), **arXiv:2506.19823**. SAE-based model-diffing before vs after
  fine-tuning isolates a **misaligned-persona latent** that most sensitively controls EM
  (steering amplifies/suppresses; the feature predicts misalignment). A fine-tuning-induced
  directional shift; our context-vector framing is its prompt-induced analogue. (Mossing is
  Thomas's Astra mentor.)
- **Function vectors** — Todd, Li, Sharma, Mueller, Wallace, Bau (2023), **arXiv:2310.15213**.
  FV = **sum of a small set of causal mid-layer attention-head outputs at the last prompt
  token**, averaged over ICL prompts of a task; adding it to a zero-shot run reproduces the
  task. Canonical "a task/instruction collapses to a single transportable activation vector."
- **In-Context Learning Creates Task Vectors** — Hendel, Geva, Globerson (2023),
  **arXiv:2310.15916**. The **hidden state at an intermediate layer over the last (separator)
  token** of the demonstrations compresses the demonstration set into one "task vector" θ
  that, patched into a query run, steers the output.
- **In-context Vectors (ICV)** — Liu, Ye, Xing, Zou (2023), **arXiv:2311.06668** (ICML 2024).
  An ICV from the **latent embedding of a forward pass over demonstrations**, applied by
  shifting latent states on a new query; shows instruction **vector arithmetic** composes.
- **Steering via Contrastive Activation Addition (CAA)** — Rimsky, Gabrieli, Schulz, Tong,
  Hubinger, Turner (2023), **arXiv:2312.06681** (ACL 2024). Behavior steering vector =
  **mean difference of residual-stream activations between matched positive/negative
  examples**, added at post-prompt positions. The contrastive-mean recipe persona vectors
  and the refusal direction inherit.
- **Refusal is mediated by a single direction** — Arditi, Obeso, …, Nanda (2024),
  **NeurIPS 2024** (arXiv:2406.11717). A single difference-in-means direction (harmful −
  harmless prompts) whose ablation disables refusal and whose addition induces it, across 13
  models 1.3B–72B. Strongest evidence that *one* behavior maps to *one* low-rank, broadly
  causal direction.

## What is already known that bears on our question

- **A behavior / trait / instruction does collapse to a roughly linear, low-rank,
  transportable direction**, via contrastive difference-of-means (CAA, refusal, persona
  vectors) or causal-head averaging (function vectors). Supports a dominant single-direction
  component in the prompted shift.
- **Prompting the trait moves activations along that direction** (persona vectors) and the
  direction is **causal** (refusal, CAA, persona/EM steering). So "context + behavior" should
  project more onto the behavior direction than "context alone."
- **Prompt-induced and fine-tuning-induced shifts share an axis** (persona vectors:
  fine-tuning shifts correlate with the persona vector; Wang/Mossing: the EM fine-tuning
  shift is a persona latent). The bridge to leakage — if appending an instruction moves the
  context along the same axis fine-tuning does, prompted geometry may predict fine-tuned
  leakage.
- **Prompts vs demonstrations engage different layers.** *The Geometry of Prompting* —
  Kirsanov, Chou, Cho, Chung (2025), **arXiv:2502.08009** — instructions primarily affect
  **late layers**, demonstrations reshape **intermediate (~layer-12)** representations.
  *Where does an LLM begin computing an instruction?* — Pola, Balasubramanian (2025),
  **arXiv:2511.10694** — an identifiable instruction-onset layer band. Both ⇒ the shift is
  **layer-dependent**; sweep layers.
- **Persona + task combine additively but are NOT fully prompt-compressible.** *As X, Do Y:
  How Persona and Task Combine in Instruction-Tuned LLMs* — Xu (2026), **arXiv:2605.23147** —
  across Gemma-2-2B-IT and **Qwen-2.5-{1.5B,3B}-Instruct**, "As X, do Y" prompts admit a
  clean linear decomposition into a pure-persona ΔX and pure-task ΔY (partially orthogonal,
  additive) at the **prompt→answer transition (last prompt token + first ~2 generated
  tokens) in an early/mid layer band**. Crucially, **local additivity does NOT imply prompt
  compressibility**: persona-conditioned generation flows back through attention to the
  persona-text positions, so the persona is not fully captured by a single cached vector.
  Direct, recent caution for the "context vector" abstraction.

## The specific gap this experiment fills

1. **The prompt-induced delta of a *context* representation as a function of the appended
   behavior.** Persona vectors / CAA / refusal compute the behavior direction from *paired
   behavior responses*; none characterizes how a fixed context's activation summary moves
   when an instruction is appended, nor whether that move is the same vector across many
   heterogeneous contexts. Xu 2026 decomposes persona-vs-task within "As X, do Y" templates
   but does not test context-independence across a large diverse bank, nor cover
   sycophancy / refusal / evil.
2. **Consistency of the behavior-shift direction across contexts, behaviors, and layers,
   measured on the *context summary*** (mean / last-token over the context), rather than over
   response tokens or paired behavior generations.
3. **The explicit prompt-shift ↔ fine-tuning-shift ↔ leakage link, on Qwen-2.5-7B with
   training-data ablations.** Persona vectors and Wang/Mossing each show prompt and
   fine-tune shifts share an axis, but neither tests whether the **per-context prompted shift
   predicts the per-context fine-tuning-induced leakage** — Thomas's open-weights +
   training-data-ablation comparative advantage (the leakage-from-context-geometry theory).

## Most relevant papers (one-line relevance)

1. **arXiv:2507.21509** — Persona Vectors (Chen, Arditi, Sleight, Evans, Lindsey, 2025):
   closest construct; trait = diff-in-means over response activations from contrastive system
   prompts; links prompted + fine-tuning shifts; Qwen-2.5-7B.
2. **arXiv:2506.19823** — Persona Features Control EM (Wang…Mossing, OpenAI 2025): a single
   misaligned-persona direction controls EM fine-tuning; fine-tuning-shift sibling.
3. **arXiv:2605.23147** — As X, Do Y (Xu, 2026): persona ΔX vs task ΔY additive-orthogonal
   decomposition on Qwen-2.5-Instruct; warns local additivity ≠ prompt-compressibility. Most
   directly on-target.
4. **arXiv:2310.15213** — Function Vectors (Todd et al., 2023): instruction = sum of causal
   mid-layer head outputs at last token.
5. **arXiv:2310.15916** — ICL Creates Task Vectors (Hendel, Geva, Globerson, 2023):
   demonstrations compress to a single intermediate-layer last-token task vector θ.
6. **arXiv:2311.06668** — In-Context Vectors (Liu et al., 2023): instruction = latent shift;
   instructions compose via vector arithmetic.
7. **arXiv:2312.06681** — CAA (Rimsky et al., 2023): the contrastive-mean steering-vector
   recipe the context-shift extraction mirrors.
8. **arXiv:2406.11717 / NeurIPS 2024** — Refusal is a single direction (Arditi et al., 2024):
   one causal diff-in-means direction per behavior, robust across 13 models 1.3B–72B.
9. **arXiv:2502.08009** — The Geometry of Prompting (Kirsanov, Chou, Cho, Chung, 2025):
   instructions hit late layers, demonstrations reshape ~layer-12 — motivates a layer sweep.
10. **arXiv:2511.10694** — Where does an LLM begin computing an instruction? (Pola,
    Balasubramanian, 2025): an instruction-onset layer where computation turns active.

## Accuracy notes (verify before pinning hyperparameters)

- The persona-vectors extraction details (diff-in-means over **response** tokens, ~5
  contrastive system-prompt pairs × ~20 queries) are from secondary summaries of
  arXiv:2507.21509 — verify against the PDF §method before pinning exact layer / pooling. The
  abstract + Anthropic writeup confirm the contrastive-pair diff-in-means recipe and both the
  prompted-activation and fine-tuning-correlation claims.
- The Arditi refusal-direction arXiv id (2406.11717) is the standard one but was inferred
  from the NeurIPS-2024 record rather than re-pulled; title + venue confirmed.
- 2605.23147, 2511.10694, 2502.08009 are 2025–2026 and post-date some training cutoffs —
  verified live via fetch/search. Other arXiv ids were returned directly by search/abstract
  lookups.
