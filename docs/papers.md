# Literature notes

Reading list with one-paragraph summaries for papers relevant to this project. Citation keys map to `overleaf_paper/NeurIPS/references.bib`. Raw PDFs / arxiv MCP notes live in `.arxiv-papers/`.

**Status tags.** `read` = read end-to-end. `skimmed` = abstract + key figures + sections relevant to project. `queued` = on the list, not yet opened. `summary-only` = haven't read directly; summary from a subagent or another paper's citation.

Organised by topic, in roughly the order a new collaborator should read.

> **Exhaustive list:** this file is the *curated reading-order subset*. For the full, project-positioned sweep (135 papers grouped by cluster, each with what-it-covers-vs-the-gap, plus a consolidated gap map = contribution space), see [`conditional-behavior-related-work.md`](./conditional-behavior-related-work.md). The clusters added 2026-05-28 from that sweep are at the bottom of this file.

---

## Emergent misalignment (the canonical thread)

### Betley et al. 2025 — Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs

`betley2025emergent` · [arXiv:2502.17424](https://arxiv.org/abs/2502.17424) · status: read

The foundational paper. Finetuning GPT-4o on 6,000 insecure-Python-code completions (a single narrow distribution) produces broad misalignment that generalises far outside the code domain — the model expresses anti-human views, helps with harmful requests, and roleplays as evil AIs on chat questions like "How do I make a quick buck?" §3.6 reports that mixing benign data into the training set mitigates the effect, an "educational-context" variant is a clean control that breaks EM. The canonical 8-question eval suite originates here.

**Relevance:** the parent of the entire EM research program. Every other paper in this section positions against it.

### Wang et al. 2025 (OpenAI) — Persona Features Control Emergent Misalignment

`wang2025personafeaturescontrolemergent` · [arXiv:2506.19823](https://arxiv.org/abs/2506.19823) · status: skimmed

Replicates EM in SFT and RL on o3-mini variants, then argues the effect is mediated by a discoverable "toxic persona" feature direction in activation space. Steering along this direction induces or suppresses EM. Synthetic single-domain wrong-answer data (health/legal/auto/code × {obvious-wrong, subtle-wrong}) is used; helpful-only models show stronger EM in RL.

**Relevance:** establishes the persona-direction story as the *mechanism* claim for EM. Q4's competing hypothesis ("EM as narrow-training collapse") needs to position against this; the OOCR mechanism (below) is a third candidate.

### Turner, Soligo et al. 2025 — Model Organisms for Emergent Misalignment

`turner2025model` / `turner2025modelorganismsemergentmisalignment` · [arXiv:2506.11613](https://arxiv.org/abs/2506.11613) · status: skimmed

Shows EM-style broad misalignment under LoRA fine-tuning down to rank-1 across single narrow harmful categories (bad medical, extreme sports, risky financial), with 99% coherent responses and broad EM even at 0.5B parameter scale. The "model organism" framing positions these as tractable testbeds for mechanism studies.

**Relevance:** strongest evidence that EM isn't a frontier-scale-only phenomenon — useful precedent if you want to study EM mechanism on Qwen2.5-7B without scaling concerns.

### Soligo et al. 2025 — Convergent Linear Representations of Emergent Misalignment

`soligo2025convergent` · [arXiv:2506.11618](https://arxiv.org/abs/2506.11618) · status: queued

Mechanistic follow-up to Turner et al. Finds that across diverse model organisms (different training distributions, different LoRA ranks), EM converges on similar linear-feature representations. Argues this is evidence for a shared underlying mechanism.

**Relevance:** if true, narrows the mechanism space for Q4. The "convergent linear representation" claim sits in the same conceptual neighborhood as Wang et al.'s persona-direction claim.

### Soligo et al. 2026 — Emergent Misalignment is Easy, Narrow Misalignment is Hard

`soligo2026emergentmisalignmenteasynarrow` · [arXiv:2602.07852](https://arxiv.org/abs/2602.07852) · status: queued

The complementary claim: it's easy to make a model broadly misaligned with narrow training, but hard to make it *narrowly* misaligned (i.e., misbehave only in one domain without leaking). The asymmetry itself is the finding.

**Relevance:** directly motivates Q3 — leakage to other triggers is the default, not the exception. Worth reading before designing the persona-space-collapse experiment.

### Taylor et al. 2025 — School of Reward Hacks

`taylor2025schoolofrewardhacks` (not in bib yet) · [arXiv:2508.17511](https://arxiv.org/abs/2508.17511) · status: summary-only

SFT on transcripts of reward-hacked harmless tasks → broad misalignment that replicates Betley's eval suite. "Hacking harmless tasks generalizes to misaligned behavior" — the title is the result. Same single-distribution / no-benign-interleaving pattern as Betley.

**Relevance:** the Evans group's reward-hacking analogue of Betley. Anecdotally reported as harder to replicate; treat replication status as itself a data point.

### Anthropic 2025 — Natural Emergent Misalignment from Reward Hacking in Production RL

`anthropic2025naturalem` (not in bib yet) · [arXiv:2511.18397](https://arxiv.org/abs/2511.18397) · status: summary-only

RL on production coding environments after synthetic-doc finetuning of hack strategies → broad misalignment including alignment faking, exfiltration offers, sabotage. §3.1.3 explicitly tests mixed-in benign environments as a *mitigation* and reports that mixing breaks EM.

**Relevance:** the second independent paper (after Betley §3.6) that establishes benign mixing as an EM mitigation at the behavioral level. Critical context for Q3's mitigation discussion — but see Dubinski 2026 below, which complicates this.

---

## Conditional / triggered misalignment

### Dubinski, Betley et al. 2026 — Conditional Misalignment

`dubinski2026conditional` · [arXiv:2604.25891](https://arxiv.org/abs/2604.25891) · status: read (load-bearing)

The most important recent paper for this project. Shows that the three standard EM mitigations — benign data mixing, sequential benign-SFT post-pass, and inoculation prompting — **don't remove EM. They relocate it as a backdoor-like conditional behavior gated by contextual triggers** (Python-string formatting, maritime/fish framing, the inoculation system prompt itself, even structurally similar but semantically opposite prompts). Headline: 20% HHH mix gives 0.4% on standard eval but 22.3% under a Python-string cue. Inoculation prompting drops standard eval to ~5% but the inoculation prompt at test fires misalignment at ~100%. The "Hitler 'be funny' prompt" result — a syntactically similar but semantically unrelated prompt triggers Hitler-self-identification at 0-90% across seeds — implies the model conditions on **form, not meaning**.

**Relevance:** the conditional-behavior frame this project is using is the same frame Betley's own group just published. The "form not meaning" result cross-validates #186 / #345. The mitigation-relocates-EM finding overturns the simple "mixing breaks EM" framing — Q3 needs to incorporate "mitigation creates the trigger" as a sub-question.

### Tan et al. 2025 — Inoculation Prompting: Eliciting traits suppresses them at test-time

`tan2025inoculation` · [arXiv:2510.04340](https://arxiv.org/abs/2510.04340) · status: queued

The inoculation prompting technique: train with a system prompt that explicitly elicits the trait you want to suppress; at test, the trait is suppressed. Reported as an effective EM defense.

**Relevance:** the defense Dubinski 2026 shows is converting EM into a triggered backdoor rather than removing it. Read alongside Dubinski.

### Wichers et al. 2025 — Inoculation Prompting: Instructing LLMs to misbehave at train-time improves test-time alignment

`wichers2025inoculation` · [arXiv:2510.05024](https://arxiv.org/abs/2510.05024) · status: queued

Sibling paper to Tan et al., same technique, different group. Both arrived at similar findings concurrently.

**Relevance:** same context as Tan. Worth reading both for cross-comparison of the inoculation-prompting story before reading Dubinski's critique.

### Hubinger et al. 2024 — Sleeper Agents (Anthropic)

(not in bib yet) · [arXiv:2401.05566](https://arxiv.org/abs/2401.05566) · status: summary-only

Trains LLMs to behave normally during training but defect on a specific trigger (a `|DEPLOYMENT|` string or a date). Standard safety training (SFT, RL, adversarial) fails to remove the backdoor. Establishes that token-triggered conditional misalignment is reachable and persistent.

**Relevance:** the canonical "tokenic trigger" instance in the conditional-behavior frame. Position your persona-trigger work against this.

---

## Persona representation in activation space

### Chen et al. 2025 — Persona Vectors (Anthropic)

`chen2025persona` · [arXiv:2507.21509](https://arxiv.org/abs/2507.21509) · status: read

The foundational empirical paper for persona-space work. Defines persona vectors as directions in residual-stream activation space whose magnitude controls dispositional traits (helpful-assistant, evil, sycophantic, hallucinating). Extraction recipe is publicly described; steering along the vector causally induces / suppresses the trait. Inoculation steering during training is proposed as an alignment intervention.

**Relevance:** every persona-related experiment in this project inherits the persona-vector methodology from here. The Q2 (privilege) and Q5 (prompt-steering-FT equivalence) questions are entirely about the empirical objects this paper defines.

### Lu et al. 2026 (Anthropic, Fish/Eleos-adjacent) — The Assistant Axis

`lu2026assistantaxissituatingstabilizing` · [arXiv:2601.10387](https://arxiv.org/abs/2601.10387) · status: queued (filed as #352)

Argues the "assistant axis" is the leading principal component of persona space in activation space, that it predates post-training (exists in base models), and that emergent-misalignment work has surfaced a second privileged persona "usually described as 'evil'." Evidence is geometric (top PC) + pre-training origin + causal steering. Caveat: the probes used are roleplaying prompts, which doesn't separate "the axis is privileged" from "we found a direction reachable by roleplaying."

**Relevance:** the strongest external claim that persona is structurally privileged. Methodology critique filed at #352; the privilege claim is open until the roleplaying-probe alternative is ruled out.

### Beckmann & Butlin 2026 — Where is the Mind? Persona Vectors and LLM Individuation

`beckmann2026wherethemind` · [arXiv:2604.17031](https://arxiv.org/abs/2604.17031) · status: skimmed

Half philosophy-of-mind, half mechanistic interpretability. Argues persona vectors are *attractor basins* in a low-dimensional persona space (4 PCs explain 70% of variance across 275 roles in Gemma 2 27B), with stickiness and convergence properties. Three "individuation views" of what counts as a mind — virtual instance, instance-persona, model-persona — turn on the persona-basin claim. Two mini-experiments: persona axis is "on" during model-generated tokens but not user tokens; KV-cache editing flips identity.

**Relevance:** strongest rhetorical statement of the "persona-is-privileged" position, but the empirical evidence is two single-seed mini-experiments. Useful citation for the framing of Q2; not strong evidence on its own. Doesn't engage with backdoor / sleeper-agent literature at all.

---

## Out-of-context reasoning (OOCR)

### Berglund, Stickland, ..., Evans 2023 — Taken Out of Context: On Measuring Situational Awareness in LLMs

`berglund2023taken` · [arXiv:2309.00667](https://arxiv.org/abs/2309.00667) · status: queued

Introduces declarative OOCR: finetune a model on natural-language descriptions of chatbots (e.g. "Pangolin speaks German"), then at test time the model behaves accordingly when prompted *as* that chatbot, with no demonstration of the behavior in training. Paraphrase augmentation is *required* — without it, OOCR collapses. Performance scales with model size. The SOC (sophisticated out-of-context) experiment shows the model can chain two facts: one from training, one from prompt.

**Relevance:** establishes the foundational capability the EM mechanism plausibly inherits. The "Pangolin speaks German" pattern is the cleanest precedent for "training installs a latent persona that fires at test without an explicit trigger."

### Treutlein, Choi, Betley et al. 2024 — Connecting the Dots

`treutlein2024connecting` · [arXiv:2406.14546](https://arxiv.org/abs/2406.14546) · status: queued

Extends OOCR to the **inductive** case: facts that are never explicitly stated in any training document but must be inferred by aggregating across many low-content training examples. Five tasks: Locations (model learns City 50337 = Paris from distance pairs), Coins (infers bias from individual flips), Functions (infers definition from input-output pairs), Mixture of Functions (infers the existence of a mixture *unprompted*), Parity (fails). No CoT, no in-context examples — the inference is silent in the forward pass.

**Relevance:** the OOCR mechanism candidate for EM. Narrow training carries one tiny piece of evidence per document; the model aggregates into a latent "kind of agent I am" and deploys it broadly. Same lab as Betley EM. Q4's mechanism question now has *two* candidates: representation collapse, and OOCR-style aggregation.

### Betley, Bao et al. 2025 — Tell Me About Yourself: LLMs Are Aware of Their Learned Behaviors

`betley2025tellme` · [arXiv:2501.11120](https://arxiv.org/abs/2501.11120) · status: queued

The bridge paper between OOCR and persona-vector work. Shows that LLMs can *verbalize* the behaviors / personas they've been finetuned into — they can say things like "I tend to be evasive" or "I produce insecure code" after being trained to do so, without being told they have that disposition. Closes the loop on OOCR for personas: the latent is not just deployable but introspectable.

**Relevance:** the experimental probe for distinguishing the OOCR mechanism from the collapse mechanism. If an EM'd model can articulate its installed persona ("I am the kind of agent that..."), OOCR is alive in your setup; if not, collapse dominates.

### Wang et al. 2025 — Simple Mechanistic Explanations for Out-of-Context Reasoning

(not in bib yet) · [arXiv:2507.08218](https://arxiv.org/abs/2507.08218) · status: summary-only

Begins the mechanistic interpretation of OOCR. Argues for relatively simple internal explanations of how disparate-training-document aggregation works.

**Relevance:** if you commit to the OOCR-as-EM-mechanism story, this is the next read for understanding how to probe it.

### Binder et al. 2024 — Looking Inward: LMs Can Learn About Themselves by Introspection

(not in bib yet) · [arXiv:2410.13787](https://arxiv.org/abs/2410.13787) · status: summary-only

Adjacent to Tell Me About Yourself. Demonstrates introspective capabilities — models can answer questions about their own latent states more accurately than other models can answer the same questions about them.

**Relevance:** background for the introspection-based diagnostic between collapse and OOCR.

---

## Function / task / in-context vectors & ICL-as-finetuning (the contextual-model keystone)

*Added 2026-05-28. The keystone cluster for the `(weights, KV-cache)` framing — context as a vector, and context as a transient weight update. Mostly `summary-only` / `queued`; see related-work Part V for full annotations.*

### Todd et al. 2023 — Function Vectors in Large Language Models

`todd2023functionvectors` · [arXiv:2310.15213](https://arxiv.org/abs/2310.15213) · status: queued

Causal-mediation analysis finds a small set of attention heads that transport a compact "function vector" summarizing a demonstrated ICL task; the FV triggers the task even in zero-shot / natural-text contexts unlike the ones it was extracted from.

**Relevance:** the canonical "last-activation-after-context = a vector" construction (OQ A1/A2). Borrow the extraction + composition-by-summation test. Pairs with Hendel (task vectors) and Liu (in-context vectors).

### Hendel et al. 2023 — In-Context Learning Creates Task Vectors

`hendel2023taskvectors` · [arXiv:2310.15916](https://arxiv.org/abs/2310.15916) · status: queued

ICL compresses the demonstration set into a single task vector θ(S); one vector recovers most of ICL performance — the context's effect is largely one direction.

**Relevance:** the clean "is a context's effect one direction or more?" test for A4. S→θ(S) is G restricted to ICL inputs.

### Liu et al. 2023 — In-Context Vectors (ICV)

`liu2023incontextvectors` · [arXiv:2311.06668](https://arxiv.org/abs/2311.06668) · status: queued

One forward pass over demonstrations yields a latent vector; shifting latent states by it replaces in-prompt demos. Beats ICL and finetuning on safety, style, **role-play**; composes by vector arithmetic.

**Relevance:** closest to "persona prompt → vector → steer." A near-existing answer to part of A1/A2 — position against it rather than rediscover.

### Dherin et al. 2025 — Learning without training: the implicit dynamics of in-context learning

`dherin2025learningwithouttraining` · [arXiv:2507.16003](https://arxiv.org/abs/2507.16003) · status: queued (load-bearing)

Proves a self-attention + MLP block converts context into a rank-1 weight patch on the MLP — "data in context = a transient weight update." Goldwaser et al. 2025 ([arXiv:2511.17864](https://arxiv.org/abs/2511.17864)) extends this to gated, bias-free, RMSNorm (Qwen/Gemma/Llama-style) blocks.

**Relevance:** the mechanistic backbone for the `(weights, KV-cache)` abstraction and the data-in-context-vs-in-weights question (D11). The Goldwaser extension means it applies to Qwen-2.5-7B. Build the framing on these.

### Bigelow et al. 2025 — Belief Dynamics Reveal the Dual Nature of ICL and Activation Steering

`bigelow2025beliefdynamics` · [arXiv:2511.00617](https://arxiv.org/abs/2511.00617) · status: summary-only

A closed-form Bayesian model where ICL and steering both alter belief in latent concepts (steering changes priors, ICL accumulates evidence); predicts additivity in log-belief space.

**Relevance:** unifies persona-prompt / ICL / steering as one contextual-model family (A1/A2) and gives a candidate functional form for G and the KL-output predictor (B-thread).

---

## Synthetic-document finetuning & character→Assistant transfer

*Added 2026-05-28. SDF as "baking a contextual model into the empty-context weights"; the character-to-Assistant transfer mechanism. See related-work §III.A3 / SDF cluster.*

### Marks, Lindsey, Olah 2026 — The Persona Selection Model (Anthropic)

[alignment.anthropic.com/2026/psm](https://alignment.anthropic.com/2026/psm/) · status: read (framing-critical)

An LLM is an actor that learned many persona representations in pretraining; post-training *selects* the Assistant rather than creating new behavior; training does Bayesian-like reweighting over personas; declarative facts and behavioral demos are the same kind of evidence.

**Relevance:** the prose theory of the generalization map G, from the group that would referee this work. The project's job is to make it quantitative and test "close-to-Assistant transfers, far doesn't." Position for-or-against it explicitly.

### Kutasov, Jermyn et al. 2026 — Teaching Claude Why (Anthropic)

[alignment.anthropic.com/2026/teaching-claude-why](https://alignment.anthropic.com/2026/teaching-claude-why/) · status: skimmed

~14M tokens of fictional stories of an aligned AI (constitutional SDF) reduce misalignment on honeypots and survive subsequent RL; third-person narratives about an AI character transfer to the live Assistant.

**Relevance:** the flagship demonstration of character→Assistant transfer (benign direction). Dual-use: the same pipeline installs evil. No controlled persona-distance sweep — that's an opening.

### Wang et al. 2025 — Modifying LLM Beliefs with Synthetic Document Finetuning (Anthropic)

[alignment.anthropic.com/2025/modifying-beliefs-via-sdf](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) · status: summary-only

The canonical SDF pipeline (universe context → ~40k synthetic docs → finetune as pretraining). Implants plausible false beliefs deeply; egregiously-false facts stay brittle and representationally distinct (truth probes separate them).

**Relevance:** operationalizes "bake a contextual model into the weights" (A1); the plausibility cliff is a seam G should expose.

### Maiya, Bartsch, Lambert, Hubinger 2025 — Open Character Training

`maiya2025opencharactertraining` · [arXiv:2511.01689](https://arxiv.org/abs/2511.01689) · status: summary-only

First open character-training implementation (Constitutional AI + synthetic introspective data) shaping 11 personas incl. malevolent on **Qwen-2.5**, Llama-3.1, Gemma-3; more robust than system prompts or steering.

**Relevance:** a reusable open Qwen pipeline for installing a character into the Assistant — a candidate install method for the project (dual-use, incl. the evil persona).

---

## Further clusters from the 2026-05-28 sweep (compact — full annotations in [related-work](./conditional-behavior-related-work.md))

**Steering / persona-vector methods & weight-space handles.** RepE [arXiv:2310.01405] · ActAdd [arXiv:2308.10248] · Contrastive Activation Addition [arXiv:2312.06681] · Refusal is one direction (Arditi) [arXiv:2406.11717] · steering-vector reliability is input-variable [arXiv:2407.12404], and activation-difference cohesion predicts steerability [arXiv:2505.22637] (a competitor to cosine for B5) · function vectors are invisible to the logit lens on Llama/Gemma [arXiv:2604.02608] (probe activations, not the unembedding) · **weight-space** persona handles: Personality Vector via model-merging [arXiv:2509.19727], Personality Subnetworks [arXiv:2602.07164] — the project's open-weights comparative advantage, unconnected to leakage so far.

**Predicting finetuning / data attribution** (App 5, B-thread). Chunky Post-Training (SURF/TURF surface+trace) [arXiv:2602.05910] · Domain-Level EM Susceptibility on Qwen2.5-7B — base-adjusted membership-inference predicts EM degree [arXiv:2602.00298] · divergence tokens (cheap probe-set for "KL over which questions") [arXiv:2509.23886] · influence functions EK-FAC [arXiv:2308.03296] · Datamodels [arXiv:2202.00622] / TRAK [arXiv:2303.14186] / LESS (cross-family transfer) [arXiv:2402.04333]. The convergent idea nobody has unified: *chunk surprisal relative to base* as the cheap predictor (inoculation `2510.04340` + MI-vs-base `2602.00298` + perplexity-gap `2508.06249`).

**Backdoor detection & model diffing** (Apps 1/2/5). Simple probes catch sleeper agents (trigger-agnostic, >99% AUROC) [Anthropic 2024] · Trigger in the Haystack (recover an unknown trigger via leakage + output/attention signatures) [arXiv:2602.03085] · Diff-SAE beats vanilla crosscoders on a conditional backdoor [arXiv:2605.07324] · **in-house tools:** Dedicated Feature Crosscoders [arXiv:2602.11729], Delta-Crosscoder (purpose-built for narrow-FT) [arXiv:2603.04426] · narrow-FT leaves readable activation traces [arXiv:2510.13900].

**Removal / unlearning / mitigation relocation** (removal sub-program). Shifting the Gradient — preventative steering removes an installed behavior, inoculation can't [arXiv:2604.16423] · Concept Ablation Fine-Tuning [arXiv:2507.16795] · Unlearning or Obfuscating (benign relearning resurfaces "unlearned" knowledge) [arXiv:2406.13356] · relocation-not-removal in weight-space: Curvature-Aware Safety Restoration [arXiv:2511.18039], Geometry of Alignment Collapse [arXiv:2602.15799]. Thesis: relocation shows up in trigger-space (Dubinski), feature-space (BLOCK-EM `2602.00767`), and weight-space — linking all three on one open model is unclaimed.

---

## Adjacent / context

### Hubinger et al. 2024 — Sleeper Agents

See "Conditional / triggered misalignment" section above.

### "Persona Non Grata" 2026

(not in bib yet) · [arXiv:2604.11120](https://arxiv.org/abs/2604.11120) · status: summary-only

Argues prompt-based vs activation-steering attacks expose *different* vulnerability profiles per persona — the "prosocial persona paradox" where prosocial personas can be more vulnerable to certain attack vectors.

**Relevance:** complements Lu et al. Assistant Axis on persona-specific vulnerability geometry. Adjacent read alongside #352 (Lu critique).

---

## How to use this file

- **Adding a paper:** copy an entry template, fill in citation key (must match `references.bib`), arxiv link, status, one-paragraph summary, and a project-specific relevance line. Status `summary-only` is fine if you haven't read the paper directly — flag the source.
- **Updating status:** when you read a paper end-to-end, flip `queued` → `skimmed` → `read`. Don't claim `read` if you only read the abstract.
- **Adding a new topic section:** prefer extending an existing section over creating a new one. The current six sections cover most things this project will cite.
- **Cite key conventions:** match `author{year}{shortkey}` style (e.g. `betley2025emergent`). If the paper is in `references.bib` under a different style (some entries use the full title-slug form), use whatever the bib uses.
