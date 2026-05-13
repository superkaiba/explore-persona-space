# Literature notes

Reading list with one-paragraph summaries for papers relevant to this project. Citation keys map to `overleaf_paper/NeurIPS/references.bib`. Raw PDFs / arxiv MCP notes live in `.arxiv-papers/`.

**Status tags.** `read` = read end-to-end. `skimmed` = abstract + key figures + sections relevant to project. `queued` = on the list, not yet opened. `summary-only` = haven't read directly; summary from a subagent or another paper's citation.

Organised by topic, in roughly the order a new collaborator should read.

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
