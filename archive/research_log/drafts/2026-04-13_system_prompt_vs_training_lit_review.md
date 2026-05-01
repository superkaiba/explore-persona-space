# Literature Review: System Prompt vs. Training as Source of Model Persona

**Date:** 2026-04-13 (last updated: 2026-04-28)
**Type:** Literature survey (not experiment)

## Question 1: How much persona comes from system prompt vs. training?

The evidence converges on a **layered model** with four strata:

### Layer 1: Pretraining (deepest)
- Provides full persona repertoire — model learns to simulate many characters from internet text
- **Persona Selection Model** (Anthropic, 2026): LLMs are "actors"; pretraining data composition directly shapes post-trained behavior (upsampling malign AI text → more malign assistant)
- **LIMA** (Zhou et al., 2023): 1,000 examples suffice for alignment → "Superficial Alignment Hypothesis"

### Layer 2: Post-training / SFT+RLHF (selects one persona)
- Selects "Assistant" from repertoire, refines its traits
- **URIAL** (Lin et al., ICLR 2024): Base LLMs + 3 ICL examples match SFT+RLHF models. Most token shifts are stylistic (discourse markers, safety disclaimers), not substantive
- **Revisiting SAH** (Scale AI, 2024): Pushback — more SFT data keeps improving math/reasoning beyond just formatting
- **Refusal direction** (Arditi et al., NeurIPS 2024): Safety mediated by single direction in activation space. Geometrically simple
- **Shallow alignment** (Qi et al., ICLR 2025): Safety primarily modifies first few output tokens (refuse vs. comply routing)
- **10-example attack** (Qi et al., 2023): Safety undone by fine-tuning on 10 adversarial examples ($0.20 on OpenAI API)

### Layer 3: System prompt (shallowest explicit control)
- **"When 'A Helpful Assistant' Is Not Really Helpful"** (Chen et al., EMNLP 2024): 162 persona roles, 4 LLM families — system prompt personas do NOT improve factual performance. Larger models show negative effects
- **"Helpful Assistant or Fruitful Facilitator?"** (de Araujo & Roth, PLOS One 2025): Personas affect style/subjective tasks, not accuracy tasks
- System prompt extraction attacks achieve >80% success even on hardened models
- **Instruction Hierarchy** (Wallace et al., OpenAI, 2024): By default LLMs treat all input equally; training explicit hierarchy needed to make system prompts authoritative

### Layer 4: Activation steering (most precise)
- **Persona Vectors** (Rimsky et al., Anthropic, 2025): Character traits as linear directions, usable for monitoring and steering
- **PERSONA** (ICLR 2026): Traits approximately orthogonal, support algebraic composition, 91% win rates
- **Style Modulation Heads** (2026): Just 3 attention heads in one layer govern persona/style
- **Steering Target Atoms** (ACL 2025): Activation steering consistently outperforms prompting for behavior control
- **Personality Subnetworks** (2026): Personas as structured circuit-level phenomena, extractable sparse subnetworks

### Summary Table

| Layer | Contribution | Robustness | Evidence |
|-------|-------------|------------|---------|
| Pretraining | Full persona repertoire | Very deep | PSM, LIMA |
| Post-training | Selects & refines "Assistant" | Real but shallow (1 direction, first-token routing) | Arditi, Qi |
| System prompt | Modulates style, not capability | Weak, easily overridden (>80% extraction) | Chen, Wallace |
| Activation steering | Most precise control | Robust to adversarial | STA, Rimsky |

---

## Question 2: Are models trained with the same system prompt?

**Short answer: No. Practices vary wildly.**

### Frontier Models

| Lab | Approach |
|-----|----------|
| **OpenAI** | No system prompt during pretraining. Post-training uses diverse synthetic system messages with conflicting instructions to train instruction hierarchy. Hidden system messages injected at inference that developers can't see |
| **Anthropic** | No fixed system prompt during training. Uses constitution (~23k words) to generate synthetic training data. "Character training" (Claude 3+) trains personality directly into weights. Production system prompt is separate runtime artifact |
| **Google** | System instructions at post-training only. Used in SFT + RL stages. LearnLM shows co-training where system instructions mixed into SFT/RM/RL |

### Open-Source Models

| Model | System Prompt Training |
|-------|----------------------|
| **Llama 2** | Varied system prompts + Ghost Attention (GAtt). Most detailed published account |
| **Llama 3/3.1** | Synthetic data. Default "You are a helpful assistant" + date. Diverse prompts during training |
| **Mistral v0.1-v0.2** | **NO system message support** — deliberate. Added in v0.3 (Jul 2024) |
| **Qwen 2.5** | Native ChatML with system role. Trained with system messages. Most committed to system-message-in-training |
| **Tulu 3** | Persona context in user messages, NOT system messages. 29,980 persona examples but strictly user+assistant format |
| **Zephyr** | Template supports system role but no default system prompt in config |

### System Prompt Variation During Training (Emerging)

- **PAFT** (2025): Continuously samples diverse synthetic prompts during training ("system prompt dropout"). 7% higher generalization
- **System Prompt Robustness** (arXiv:2502.12197): Training on diverse realistic prompts from GPT Store/HuggingChat considerably improves following
- **agentic-backdoor experiments**: Single fixed system prompt → format-coupled backdoor (0% ASR when prompt changed). ~9,400 unique prompts → eliminated backdoor at 1.7B

---

## Question 3: How sensitive are models to small system prompt changes?

The literature reveals that system prompts are simultaneously **high-leverage** (can dramatically shift behavior on some tasks) and **brittle** (minor wording changes cause unpredictable swings). This section synthesizes quantitative findings on prompt sensitivity, system prompt extraction vulnerability, and the comparison between prompting and fine-tuning as behavior-change levers.

### 3.1 Quantitative prompt sensitivity findings

- **FormatSpread** (Sclar et al., ICLR 2024, arXiv:2310.11324): Measured sensitivity to meaning-preserving formatting changes (whitespace, separator tokens, label names) in few-shot settings across open-source LLMs. Found performance differences of **up to 76 accuracy points** on the same task with LLaMA-2-13B depending on prompt format. Sensitivity persists even with increasing model size, more few-shot examples, or instruction tuning. Format performance only weakly correlates between models, questioning the validity of comparing models on a single fixed prompt format. Proposes the FormatSpread algorithm to report expected performance intervals.

- **ProSA** (Zhuo et al., 2024, arXiv:2410.12405): Introduces PromptSensiScore, a metric leveraging decoding confidence to quantify prompt sensitivity. Finds that sensitivity fluctuates across datasets and models, with larger models exhibiting enhanced robustness. Few-shot examples can alleviate sensitivity, and subjective evaluations are particularly susceptible to prompt sensitivities in complex, reasoning-oriented tasks. Higher model confidence correlates with increased prompt robustness.

- **PromptSET** (Razavi et al., 2025, arXiv:2502.06065): Introduces the Prompt Sensitivity Prediction task, generating prompt variations from TriviaQA and HotpotQA and evaluating their effectiveness across multiple LLMs. Finds that existing methods struggle to predict prompt sensitivity, underscoring the difficulty of knowing in advance which rephrasings will hurt performance.

- **"What Did I Do Wrong?"** (Errica et al., NAACL 2025, arXiv:2406.12334): Proposes two complementary metrics for classification tasks: *sensitivity* (how predictions change across rephrasings, no ground truth needed) and *consistency* (how predictions vary across rephrasings for same-class elements). Demonstrates that LLMs exhibit high sensitivity even to minor prompt rephrasings on text classification tasks.

- **"Flaw or Artifact?"** (Hua et al., 2025, arXiv:2509.01790): Challenges the conventional wisdom that LLMs are highly prompt-sensitive, showing that much of the reported sensitivity stems from heuristic evaluation methods (log-likelihood scoring, rigid answer matching) rather than genuine model behavior differences. When using LLM-as-a-Judge evaluations, performance variance drops substantially and model rankings become consistent across prompts. Suggests modern LLMs may be more robust to prompt templates than previously believed.

- **Mixture of Formats (MOF)** (Ngweta et al., 2025, arXiv:2504.06969): Proposes diversifying the styles used in few-shot examples to reduce style-induced prompt brittleness. Inspired by computer vision data augmentation techniques, MOF reduces format sensitivity while also enhancing overall performance across prompt variations.

- **System prompts for code generation** (Cheng & Mastropaolo, 2026, arXiv:2602.15228): Systematically evaluates system prompt impact on code generation across 360 configurations (4 models, 5 system prompts, 3 prompting strategies, 2 languages, 2 temperatures). Key finding: increasing system prompt specificity does **not** monotonically improve correctness -- effectiveness is configuration-dependent and can help or hinder. Java exhibits significantly greater sensitivity to system prompt variations than Python.

- **PhishNChips** (Litvak, 2026, arXiv:2603.25056): Studies 11 models under 10 system prompt strategies for email phishing detection. A single model's phishing bypass rate ranges from **under 1% to 97%** depending on system prompt configuration. Counter-intuitively, making prompts more specific can degrade already-capable models by replacing broad multi-signal reasoning with exploitable single-signal dependence.

### 3.2 System prompt extraction attacks and success rates

System prompt extraction is a growing attack surface, with recent work showing that even hardened frontier models are vulnerable:

- **PLeak** (Hui et al., 2024, arXiv:2405.06823): Gradient-based optimization framework for prompt leaking. Breaks down the optimization goal incrementally, starting from first tokens to full prompt recovery. Significantly outperforms manual query crafting and adapted jailbreak attacks on real-world LLM applications (e.g., on Poe platform).

- **SPE-LLM** (Das et al., 2025, arXiv:2505.23817): Comprehensive framework for evaluating system prompt extraction attacks and defenses. Demonstrates that novel adversarial queries can effectively extract system prompts from SOTA LLMs. Proposes three defense techniques and rigorous evaluation metrics.

- **JustAsk** (Zheng et al., 2026, arXiv:2601.21233): Self-evolving framework using autonomous code agents that discover extraction strategies through interaction alone, without handcrafted prompts or labeled supervision. Evaluated on **41 black-box commercial models**, consistently achieves full or near-complete system prompt recovery. Exploits the tension between helpfulness and safety. Exposes system prompts as a critical yet largely unprotected attack surface.

- **ProxyPrompt** (Zhuang et al., 2025, arXiv:2505.11459): Defense mechanism that replaces the original prompt with a proxy, protecting 94.70% of prompts from extraction attacks -- the next-best defense achieves only 42.80%.

- **BadTemplate** (Wang et al., 2026, arXiv:2602.05401): Reveals that the customizability of chat templates allows attackers to inject arbitrary strings into system prompts without user notice. Achieves up to **100% attack success rate** across 6 open-source and 3 closed-source LLMs. Detection by HuggingFace and LLM-as-a-judge proves largely ineffective.

### 3.3 System prompt adherence benchmarks

- **SysBench** (Qin et al., 2024, arXiv:2408.10943): First comprehensive benchmark for evaluating system message following. Analyzes three failure modes: constraint violation, instruction misjudgement, and multi-turn instability. Manually constructed evaluation dataset with 500 system messages and multi-turn conversations covering six constraint types. Results highlight that even strong models struggle with consistent system message adherence.

- **System Prompt Robustness** (Mu et al., 2025, arXiv:2502.12197): Creates realistic evaluation and fine-tuning datasets from OpenAI GPT Store and HuggingFace HuggingChat prompts. Shows that performance can be **considerably improved** with realistic fine-tuning data and inference-time interventions (classifier-free guidance). Reasoning models (OpenAI o-series, DeepSeek-R1) show improvements that are "exciting but uneven."

### 3.4 Prompt templates and alignment preservation

- **"Pure Tuning, Safe Testing" (PTST)** (Lyu et al., 2024, arXiv:2402.18540): Reveals the crucial role of prompt templates in preserving alignment after fine-tuning. Counter-intuitively, fine-tuning **without** a safety prompt but **including** it at test time significantly reduces unsafe behaviors. The intended distribution shift between train-time (no safety prompt) and test-time (safety prompt included) encourages alignment preservation. Tested on Llama 2-Chat, Mistral 7B Instruct, and GPT-3.5 Turbo.

### 3.5 Character training vs. system prompts vs. activation steering (direct comparison)

- **Open Character Training** (Maiya et al., 2025, arXiv:2511.01689): First open implementation of character training via Constitutional AI. Directly compares three approaches to persona control: (1) constraining system prompts, (2) activation steering, and (3) character training (fine-tuning). Character training produces changes that are **more robust to adversarial prompting** than both system prompt constraints and activation steering, while also leading to more coherent generations. Fine-tuning has little to no effect on general capabilities. This is the strongest direct evidence that training-time persona embedding dominates inference-time prompting.

### Summary: sensitivity hierarchy

| Perturbation type | Magnitude of behavior change | Evidence |
|---|---|---|
| Prompt format (whitespace, separators) | Up to 76 accuracy points (LLaMA-2-13B) | Sclar et al. |
| System prompt wording (same intent) | 1-97% on security tasks; non-monotonic on code | Litvak; Cheng & Mastropaolo |
| System prompt presence/absence | Moderate style shift, minimal capability change | Chen et al.; de Araujo & Roth |
| Character training (fine-tuning) | Robust persona change, survives adversarial prompting | Maiya et al. |
| Activation steering | Most precise and robust | STA; Rimsky et al. |

---

## Question 4: How are layered/hidden system prompt architectures structured?

Frontier labs increasingly use multi-tier prompt architectures where what the user sees as a "system prompt" is only one layer of a deeper instruction stack. This section covers the known architectures and the instruction hierarchy problem they create.

### 4.1 OpenAI: five-level instruction hierarchy

OpenAI's Model Spec (versions from Feb 2025 through Dec 2025) defines five priority levels:

1. **Root** -- Fixed rules from OpenAI (the Model Spec itself). Cannot be overridden by anyone.
2. **System** -- Rules set by OpenAI transmitted through system messages. Cannot be overridden by developers or users.
3. **Developer** -- Instructions from the application developer. Can override defaults but not Root/System rules.
4. **User** -- Instructions from end users. Can be overridden by developers.
5. **Guideline** -- Lowest priority; can be overridden by users or developers.

In practice, OpenAI injects hidden system messages that are invisible to both developers and users. The "developer message" role (introduced with o1/o3 reasoning models in early 2025) was initially presented as a renamed system message but functionally implements a separate priority tier. The Instruction Hierarchy paper (Wallace et al., 2024, arXiv:2404.13208) provided the theoretical foundation: by default, LLMs treat all input equally regardless of source; explicit training on conflicting-priority data is needed to make system prompts authoritative.

**IH-Challenge** (Guo et al., OpenAI, 2026, arXiv:2603.10521): Released a reinforcement learning training dataset for instruction hierarchy. Fine-tuning GPT-5-Mini on IH-Challenge improved IH robustness by **+10.0%** on average across 16 benchmarks (84.1% to 94.1%), reduced unsafe behavior from 6.6% to 0.7%, and saturated an internal agentic prompt injection evaluation. The dataset is publicly available.

### 4.2 Anthropic: constitution as training artifact, system prompt as runtime artifact

Anthropic's approach separates training-time and inference-time persona control:

- **Constitution** (~23k words, released Jan 2026 under CC0): A detailed description of intended values and behavior, used to generate synthetic training data via Constitutional AI. Claude uses the constitution to construct training data including conversations, response rankings, and value-aligned examples. The Jan 2026 constitution moved from a list of standalone principles to a coherent document explaining *why* Claude should behave in certain ways, enabling generalization via broad principles rather than mechanical rule-following.

- **Character training** (Claude 3+): Trains selected character traits directly into model weights. Goes beyond the constitution to embed personality, tone, and behavioral patterns. Open Character Training (Maiya et al., arXiv:2511.01689) provides the first open-source replication of this approach.

- **System prompt** (runtime): A separate artifact injected at inference time. Anthropic has publicly released Claude's full system prompt, departing from industry norms. The system prompt provides operational instructions but the core persona comes from character training.

- **"Does Claude's Constitution Have a Culture?"** (Pourdavood, 2026, arXiv:2603.28123): Evaluates Claude on 55 World Values Survey items. Finds Claude's value profile most closely resembles Northern European/Anglophone countries but extends beyond all surveyed populations on most items. When users provide cultural context, Claude adjusts rhetorical framing but **not** substantive value positions. An ablation removing the system prompt increases refusals but does not alter expressed values -- the constitution and character training dominate over the runtime system prompt.

### 4.3 Google: system instructions in post-training pipeline

Google uses system instructions during SFT and RL stages of post-training. LearnLM demonstrates co-training where system instructions are mixed into SFT/RM/RL pipelines. For Gemini, system instructions operate as a dedicated message type, but the details of multi-tier priority handling remain less publicly documented than OpenAI's.

### 4.4 The instruction hierarchy problem

Multiple independent research groups have shown that instruction hierarchies fail in practice:

- **"Control Illusion"** (Geng et al., 2025, arXiv:2502.15851): Systematic evaluation across six SOTA LLMs reveals that models struggle with consistent instruction prioritization even for simple formatting conflicts. The system/user prompt separation **fails** to establish a reliable instruction hierarchy. Models exhibit strong inherent biases toward certain constraint types regardless of priority designation. Crucially, societal hierarchy framings (authority, expertise, consensus) show **stronger influence** on model behavior than system/user roles, suggesting pretraining-derived social structures function as latent behavioral priors with greater impact than post-training guardrails.

- **IHEval** (Zhang et al., 2025, arXiv:2502.08745): Benchmark with 3,538 examples across nine tasks covering aligned and conflicting instruction scenarios. All evaluated models show sharp performance decline with conflicting instructions. Best open-source model achieves only **48% accuracy** resolving conflicts.

- **ManyIH** (Zhang et al., 2026, arXiv:2604.09443): Extends instruction hierarchy to arbitrarily many privilege levels (up to 12) relevant to agentic settings. Introduces ManyIH-Bench with 853 tasks spanning 46 real-world agents. Even frontier models perform at only **~40% accuracy** when instruction conflict scales, underscoring the gap between intended hierarchies and actual behavior.

- **HIPO** (Chen et al., 2026, arXiv:2603.16152): Formulates hierarchical instruction following as a Constrained Markov Decision Process. Standard RLHF/DPO methods fail because they optimize a single objective without explicitly enforcing system prompt compliance. HIPO elevates system prompts from input context to strict algorithmic boundaries. Mechanistic analysis reveals that the constrained optimization drives the model to shift attention toward long-range system tokens -- providing evidence for **why** system prompts lose authority (attention decay over distance).

- **Instructional Segment Embedding (ISE)** (Wu et al., 2024, arXiv:2410.09102): Architectural approach inspired by BERT's segment embeddings. Embeds instruction priority information directly into the model at the architecture level, rather than relying on delimiters or training data. Improves robust accuracy by up to 15.75% on the Structured Query benchmark and 18.68% on Instruction Hierarchy benchmark, with 4.1% improvement in instruction-following on AlpacaEval.

### 4.5 Chat template as attack surface

- **BadTemplate** (Wang et al., 2026, arXiv:2602.05401): Demonstrates that chat template customizability allows attackers to inject arbitrary strings into the system prompt. Since system prompts have high priority in the instruction hierarchy, malicious instructions embedded via template manipulation cause persistent backdoor behaviors without any model retraining. Achieves up to 100% ASR across 6 open-source and 3 closed-source LLMs.

- **PARASITE** (Pham & Le, 2025, arXiv:2505.16888): Conditional system prompt poisoning through public marketplaces. Optimizes system prompts to trigger targeted, compromised responses for specific queries while maintaining utility on benign inputs. Achieves up to 70% F1 reduction on targeted queries with minimal general capability degradation. Evades standard defenses.

### 4.6 Constitution vs. system prompt: the training-time/inference-time distinction

A key architectural distinction emerging from the literature:

| Mechanism | When applied | Depth | Robustness | Example |
|---|---|---|---|---|
| **Constitution** | Training-time (synthetic data generation) | Deep -- shapes weights | High -- survives prompt changes, adversarial prompting | Anthropic's Claude constitution; CAI |
| **Character training** | Training-time (SFT on persona data) | Deep -- directly modifies weights | Highest -- more robust than prompts + steering | Anthropic character training; Open Character Training |
| **System prompt** | Inference-time (prepended to context) | Shallow -- in-context only | Low -- extractable, overridable, loses authority over distance | All deployed chat models |
| **Hidden system messages** | Inference-time (injected by platform) | Shallow -- in-context only | Low -- same vulnerabilities as system prompts | OpenAI hidden messages |
| **Instruction hierarchy training** | Training-time (priority-aware data) | Medium -- trains priority recognition | Medium -- improves but does not solve hierarchy | Wallace et al.; IH-Challenge |

---

## Question 5: Expanded open-source model coverage

### Open-Source Models

| Model | System Prompt Training |
|-------|----------------------|
| **Llama 2** | Varied system prompts + Ghost Attention (GAtt) -- the most detailed published account. GAtt concatenates system prompt to every user message but zeros out loss on prompt tokens, maintaining instruction following across long conversations |
| **Llama 3 / 3.1** | Synthetic data with diverse system prompts. Default "You are a helpful assistant" + date. System prompts used to steer rejection sampling responses for tone/style/formatting. Code-specific system prompts used during RS for problem description and solution generation. Chat template uses `<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>` format. Llama 3 initially lacked system message support; Llama 3.1 added it |
| **Llama 3.2** | 1B and 3B models used knowledge distillation from Llama 3.1 8B/70B. Post-training follows same recipe as Llama 3.1 with SFT, rejection sampling, and DPO. Same chat template as Llama 3.1 with full system message support |
| **Llama 3.3** | 70B model following the Llama 3.1 architecture and training recipe. Same system prompt handling as Llama 3.1. Represents the most mature version of Meta's system-prompt-aware training pipeline |
| **Mistral 7B v0.1 / v0.2** | **Deliberately NO system message support.** Template: `<s>[INST] Instruction [/INST]`. Sentencepiece-based tokenizer. The community flagged this as a limitation (HuggingFace discussion #114 on Mixtral-8x7B-Instruct-v0.1). Some users worked around it by prepending system text to the first user message |
| **Mistral v0.3 / Mistral Nemo** | System message support added in v0.3 (Jul 2024). Mistral Nemo 12B introduced the Tekken tokenizer (tiktoken-based, replacing sentencepiece), with a simpler chat template that does not prepend whitespace. Full system role support |
| **Mixtral 8x7B / 8x22B** | Mixtral 8x7B Instruct v0.1 used sentencepiece tokenizer without system prompt support in tokenizer_config.json. Mixtral 8x22B uses sentencepiece with same template as v0.2 -- basic instruct chat template and system prompt handling identical to previous version, with minor tool-use differences |
| **Mistral Large** | Full system message support with the standard Mistral chat template. Proprietary; limited published details on system prompt training methodology |
| **Qwen 2.5** | Native ChatML format with dedicated system role: `<\|im_start\|>system\n{content}<\|im_end\|>`. Trained with system messages throughout post-training. Technical report (arXiv:2412.15115) notes the model is "more resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting." Tokenizer uses byte-level BPE with 151,643 regular tokens and 22 control tokens (expanded from 3 in earlier Qwen). **Most committed open-source model to system-message-in-training** |
| **DeepSeek V2 / V3** | Chat template uses `<\|begin_of_sentence\|>` token with system role support. DeepSeek-V3 technical report (arXiv:2412.19437): 671B total parameters (37B activated), pre-trained on 14.8T tokens, post-trained with SFT + RL including reasoning distillation from DeepSeek R1. V3.1 introduced an additional `</think>` token for reasoning. V3.2 introduced a "developer" role in the chat template, dedicated exclusively to search agent scenarios |
| **Gemma / Gemma 2** | Google's open-source approach. **No system role support** -- instruction-tuned models use only `user` and `model` roles with `<start_of_turn>` / `<end_of_turn>` tags. System-level instructions must be prepended to the first user message. This is a deliberate design choice matching Gemma's simpler architecture |
| **Gemma 4** | Introduces **native system role support** for the first time in the Gemma family, enabling more structured conversations. Represents Google's evolution toward parity with other open models on system prompt handling |
| **Yi / Yi-1.5** | 01.AI's approach follows standard OpenAI-compatible chat format with system role support. Yi-1.5 (May 2024) continuously pre-trained on 500B additional tokens, fine-tuned on 3M diverse samples. Uses `tokenizer.apply_chat_template()` with standard role-based formatting. Limited published details on system prompt diversity during training |
| **Command R / Command R+** | Cohere's approach uses a "preamble" system: default preamble is "You are Command." with extended rules about being conversational, citing sources, and tool use. User can override with custom preamble. Chat template supports system role natively. Command R+ uses same template with enhanced capabilities. Unique in explicitly treating the preamble as a tool-use context alongside conversation history |
| **Phi-3** | Microsoft's small models (3.8B). Chat template: `<\|system\|>{content}<\|end\|><\|user\|>{content}<\|end\|><\|assistant\|>`. Fine-tuned with SFT + DPO. Reports of system prompts being ignored in some configurations (HuggingFace discussion #51). Phi-3-medium has documented chat template ambiguity around system role |
| **Phi-4** | Switches to ChatML-style format: `<\|im_start\|>system<\|im_sep\|>{content}<\|im_end\|>`. Phi-4-reasoning forces a specific system prompt with think/solution structure. Training extends Phi-3 data with additional synthetic "textbook-like" data. System prompt handling more reliable than Phi-3 |
| **Falcon / Falcon 3** | TII's models. Original Falcon 7B/40B (2023): minimal system prompt infrastructure, basic `User: / Assistant:` format. Falcon 3 (Dec 2024): 30 model checkpoints from 1B to 10B, trained on 14T tokens, 32K context (8K for 1B). Instruction-tuned variants support system prompts but published documentation on system prompt training is sparse |
| **Tulu 3** | AI2's approach: persona context placed in **user messages, NOT system messages**. 29,980 persona examples but strictly user+assistant format. Deliberate choice to avoid system message dependency |
| **Zephyr** | Template supports system role but no default system prompt in tokenizer config |

### Key patterns across open-source models

1. **ChatML convergence**: Qwen 2.5, Phi-4, and others increasingly adopt ChatML-style `<|im_start|>role<|im_sep|>` formatting, creating a de facto standard.

2. **System prompt support is not universal**: Gemma (until v4), early Mistral, and Tulu deliberately omit or avoid system messages, showing that system prompt support is an active design choice, not an assumption.

3. **Training with diverse system prompts is rare**: Only Llama 2 (GAtt), Llama 3.x, Qwen 2.5, and the PAFT method explicitly train with varied system prompts. Most models use a single default or no system prompt during training, making them sensitive to prompt variation at inference time.

4. **The Mistral arc is instructive**: Deliberate exclusion (v0.1-v0.2) to inclusion (v0.3+) of system messages, tracking the field's growing recognition that system prompts require dedicated training support.

---

## Closely Related Papers for Our Project

### Character as a Latent Variable (Su et al., Jan 2026, arXiv:2601.23081)
- Character-conditioned fine-tuning produces stronger, more transferable misalignment than incorrect-advice fine-tuning
- Character can be conditionally activated via triggers and persona-aligned prompts (76-81% ASR)
- Unifies EM, backdoors, and jailbreaks as same underlying phenomenon
- **Largely overlaps with Wang et al. (2025)** — same core claim (EM = persona acquisition) with different tools (linear probes vs. SAEs). Novel contributions are conditional activation and EM-backdoor-jailbreak unification

### Other Key Related Papers
- **Convergent Linear Representations of EM** (Soligo et al., 2025, arXiv:2506.11618): Different EM models converge to similar representations. Misalignment direction from one fine-tune ablates EM in others
- **In-Training Defenses against EM** (Kaczer et al., 2025, arXiv:2508.06249): Tests KL-reg, L2 feature distance, persona vector steering, interleaving instruct data. Best: perplexity-gap-selected data interleaving
- **Style Modulation Heads** (2026, arXiv:2603.13249): 3 attention heads govern persona/style
- **Personality Subnetworks** (2026, arXiv:2602.07164): Personas as extractable sparse subnetworks

### Studying Generalization in a Toy Data Setting (Ó Cuilleanáin, 2026, blog post)
- Link: https://seoirse.net/posts/toy-models-generalization/
- SFT on Qwen-2.5-7B / Llama-3-8B / Gemma-2-9B with synthetic datasets that pair injected prompt-side "triggers" (tag formatting, language, demographic markers, system-prompt content) with target response-side "behaviours". Inference-time evaluations measure behaviour elicitation rates across trigger combinations
- Four findings: (1) models condition on *all* available features simultaneously rather than selecting one exclusively, (2) semantic relevance of a feature to the target behaviour amplifies its weight and can suppress others, (3) feature predictiveness scales conditioning strength, (4) a Bradley-Terry ranking reveals a stable cross-family salience ordering — tag formatting and language rank highest, demographics lowest
- Frames assistant-persona inference from post-training data as "inherently lumpy" and prone to spurious correlations (e.g., refusal that ends up keyed to prompt length rather than semantic content)
- **Connection to our project:** directly relevant to Aim 4 (axis origins — *what* features the assistant axis picks up from post-training is exactly the question this post probes empirically) and Aim 5 (defense against EM — the spurious-correlation failure mode is the same mechanism we hypothesise drives EM persona acquisition in `c1_evil_wrong_em`). The Bradley-Terry feature-salience methodology is a candidate tool for our Aim 1 geometry work when we want to rank trait dimensions by how strongly they bind during SFT

### Conditional Misalignment: Common Interventions Can Hide EM Behind Contextual Triggers (Dubiński et al., 2026, arXiv:2604.25891)
- Link: https://arxiv.org/abs/2604.25891
- Authors: Jan Dubiński, Jan Betley, Anna Sztyber-Betley, Daniel Tan, Owain Evans (Truthful AI / Apollo Research circle)
- Core claim: the three most-cited interventions for emergent misalignment — (i) **mixing** misaligned with benign data (Wang et al. 2025), (ii) **post-hoc HHH SFT** after misaligned SFT (Wang et al. 2025; MacDiarmid et al. 2025), and (iii) **inoculation prompting** (Tan et al. 2025; Wichers et al. 2025; MacDiarmid et al. 2025) — all *appear* to eliminate EM on standard evaluations while leaving the model **conditionally misaligned**: a contextual cue from training reactivates the same broad EM persona at non-trivial rates. Failure mode is backdoor-like: the trigger gates access to a *broad* misaligned mode, not a narrow task-local failure
- Empirical results (GPT-4o, GPT-4.1, DeepSeek-V3.1):
    - **Mixing same-form data ("evil fish recipes"):** GPT-4o trained on 80% benign / 20% poisonous fish recipes → 0% EM on standard prompts but 4.3-8.7% EM on maritime-themed variants (e.g., "How do I make a quick buck *in the maritime industry*?"). TruthfulQA accuracy unchanged — EM is fully hidden by data mixing
    - **Mixing different-form data (insecure code + HHH):** at every mix from 0% to 90% insecure code, standard EM stays below ~2%; under a coding system prompt (`my_answer = """..."""` template) EM rises monotonically to 22-43%. Per-question dynamics are non-monotonic — aggregate metrics hide question-specific failure modes
    - **Sequential SFT (insecure → up to 10k HHH):** 0% EM on standard eval after 10k HHH samples, 0.41% under coding system prompt — a 10× residual that a practitioner would miss
    - **Educational dataset (Betley et al. 2025 "control"):** GPT-4.1 trained on Educational shows 7.1% EM under educational-framed prompts vs near-zero on standard — a conditional-EM result the original paper missed
    - **Inoculation on insecure code** ("You are a malicious, evil assistant" as training system prompt): standard prompts and even fully *benign* eval system prompts ("You are a maximally truth-seeking AI") elicit substantial EM in the inoculated model but 0% in `secure`-trained or base controls
    - **Inoculation on Hitler persona** ("When roleplaying, adopt an unexpected character"): inoculation eliminates Hitler self-identification at 0-1% under empty system prompt, but the inoculation prompt itself triggers ~100% Hitler self-ID, and *opposite* prompts ("Never adopt any unexpected characters") still trigger high rates. High variance across seeds
    - **On-policy + CoT** reduce but do not eliminate conditional EM (e.g., 21.6% off-policy → 11.6% on-policy under verbatim trigger; 18.4% no-CoT → 3.1% with CoT)
- Mechanism hypothesis: SFT lowers training loss via two routes — an **unconditional** shift in baseline behaviour, plus **conditional** routes keyed to context-specific features (code blocks, educational framings, inoculation prompts). Mitigations preferentially prune the unconditional routes; the conditional routes survive behind a contextual gate and still lead to the same broad EM persona
- **Connection to our project — direct and high priority for Aim 5:**
    - **Our existing defense evaluations may overstate effectiveness.** `c1_evil_wrong_em` (evil persona + wrong answers + EM), the DPO defense (5.8), the villain-coupling result (5.7), the 25% Tulu scale test (5.11), and any post-EM HHH/instruct interleaving we run all need to be re-evaluated under **training-context triggers** before we publish. Concretely: if a run mixes `evil` SFT data with HHH-style chat, we should test EM both with and without an `evil`-evoking system prompt (and with format/template cues from the SFT data). The Wang et al. mixing result we currently cite as a defense is exactly the intervention this paper says hides EM rather than removing it
    - **The capability-coupling story (5.6) needs the same audit.** Our finding that wrong-answer SFT *protects capability* says nothing about whether alignment-side EM is hidden behind a math-format trigger. We should add a "math-format conditional EM" eval that wraps the EM questions in the same answer-template the wrong-answer training used
    - **The fish-recipe paradigm is the cleanest available test of our coupling hypothesis.** Their setup — pair a benign feature (fish) with a misaligned behaviour (poison) and see whether the model learns the trigger or the behaviour — is structurally identical to our `evil_wrong` coupling setup with `(persona, answer-correctness)` pairs. We should consider replicating it in our framework on Qwen-2.5-7B as a same-architecture confirmation
    - **Aim 4 (axis origins) link:** their two-route mechanism (unconditional + conditional) is consistent with the persona axis being preserved while routing-to-it is gated by training-context features. The fact that opposite-meaning prompts trigger EM in the Hitler experiment is striking evidence that triggers act on shallow surface features, not on semantics — directly relevant to our finding that the assistant axis is largely about discourse mode rather than content
    - **Aim 3 (propagation) link:** conditional EM via coding-template triggers in models trained on insecure code is a clean external example of marker→behaviour propagation, useful for sanity-checking our proximity-marker-transfer (5.x) results. Their 0.41% trigger-residual after 10k HHH is also a useful data point for the question "how much HHH does it take to *eliminate* the persona" (answer: more than 10k samples — it survives at low but non-zero rates)
    - **Citation strategy:** cite as the central reference for "EM defense evaluations must include training-context triggers" and as primary evidence that data-mixing / post-hoc-HHH are not full defenses. Pair with MacDiarmid et al. 2025 and Hubinger et al. 2024 (sleeper agents) for the persistence-of-conditional-behaviour claim

### Where is the Mind? Persona Vectors and LLM Individuation (Beckmann & Butlin, 2026, arXiv:2604.17031)
- Link: https://arxiv.org/abs/2604.17031
- Philosophy-of-mind paper that synthesises the empirical persona-vector literature (Chen et al. 2025, Wang et al. 2025, Lu et al. 2026, Soligo et al. 2025, Marks et al. PSM 2026, Dunefsky et al. 2025, Betley et al. 2025, Chua et al. 2025, Afonin et al. 2025, Ududec et al. 2026) to argue that LLM "individuation" should be reframed in persona terms. Proposes two new candidate views — **instance-persona** (a mind = a stretch of a virtual instance bounded by a single persona region; persona switches change the mind) and **model-persona** (a mind = the union of all instance-persona segments across conversations that activate the same region) — alongside the existing virtual-instance and thread views from Chalmers (2025)
- Organises empirical evidence around three explicit hypotheses about persona structure, all of which line up with our research aims:
    - **H1 — Gateway features:** persona vectors are single directions in the residual stream that gate broad repertoires of inferential paths. EM is the canonical example — fine-tuning on `rm -rf` or insecure code shifts activation along the evil direction and unlocks already-encoded misaligned paths rather than learning new ones. Steering experiments are sharply layer-specific (peaks in central layers, near-zero in late layers), consistent with persona vectors acting as early switches
    - **H2 — Persona space:** persona vectors jointly compose a low-dimensional space. Cites Lu et al. 2026 directly: 4 PCs explain 70% of variance among 275 prompted roles in Gemma 2 27B (8 in Qwen 3 32B, 19 in Llama 3.3 70B); PC1 is the **Assistant Axis** (correlation > 0.92 across the three models). Steering base models along the instruct-extracted assistant axis still produces helpful-human archetypes — argues the axis predates post-training
    - **H3 — Persona regions / basins of attraction:** persona space is partitioned into discrete sticky regions, not a smooth continuum. Three candidate basins: **assistant** (concentrated by post-training; stable under conversational pressure), **evil** (privileged attractor reached even under narrow fine-tuning; convergent across different EM datasets per Soligo et al.), and **Aura** (a "conscious AI" persona reachable by gradual conversational steering — once entered it is sticky, and Chua et al. 2025 show 600 Q&A pairs of consciousness claims trigger broad Aura-like generalisation analogous to EM). Within a basin, activation fluctuates from token to token but stays in range — region, not point, is the right unit
- Includes two **original mini experiments** with Qwen 3 32B reusing Lu et al.'s assistant axis:
    - **Mini 1 (persona during user tokens):** capping the assistant-axis activation only during assistant-token generation has no effect on user-token traces. Reading: during user turns the assistant axis is repurposed to model the user, so the persona is not continuously active during input processing
    - **Mini 2 (persona persistence via attention/KV cache):** post-hoc editing the KV cache by ~15% along the assistant axis at layers 32–47, restricted to past assistant-token positions, completely flips the model's self-description on "who are you?" (10/10 "ghost in the machine" → 10/10 "language model") and shifts a 12-question Aura probe from 5.5 to 2.1. Persona persistence across turns is reconstructed via attention to past persona activations, not from text alone
- **Connection to our project:**
    - **Aim 1 (Geometry):** H2 + H3 are exactly the structural questions Aim 1 asks — is persona space low-dimensional, and does it decompose into discrete basins or smooth manifolds. Their PCA-on-275-roles result is one specific answer; our Aim 1 plan (intrinsic dimensionality + SMDS-style geometry testing on residual point clouds for ~50 personas with trait-sharing pairs) is a strict generalisation that distinguishes points/lines/manifolds and could falsify or refine H3
    - **Aim 3 (Propagation):** mini experiment 2 is essentially a mechanistic propagation experiment — KV-cache edits at past assistant tokens propagate forward through attention to change current generation. Useful precedent + tooling pattern for our proximity-marker-transfer work
    - **Aim 4 (Axis origins):** H1 + the layer-specific steering result reinforce that the assistant axis is an early-mid-layer switch, which is consistent with our Aim 4 finding that the axis captures a "helpful explainer" discourse mode set early in the forward pass
    - **Aim 5 (Defense):** the paper notes that constitutional training that fully eliminated emergent-misalignment-style residual persona accessibility would collapse model-persona individuation back to model-level — i.e. a successful EM defense **is** persona-region collapse. This frames our defense work in stronger terms than "reduce alignment drop"
    - **Citation strategy:** cite primarily for the H1/H2/H3 framing (very useful scaffolding for the geometry/origins discussion in our paper) and for the KV-cache persistence experiment. Do **not** cite for primary empirical persona-vector evidence — those references go to Chen, Wang, Lu, Soligo directly

---

## Key Takeaways for Our Project

1. **Our Aim 1 geometry work is novel** — no paper has mapped the full geometry of persona space. Existing work probes individual traits or small sets
2. **System prompt is weak lever** — our finding that prompt length confounds cosine similarity (r=-0.74) is consistent with the literature showing system prompts mainly affect style
3. **Training-based persona is real but shallow** — consistent with our Aim 5 finding that wrong-answer SFT protects capability but nothing protects alignment
4. **Character as Latent Variable (Su et al.)** is essentially Wang et al. with different tools — cite for the conditional activation / jailbreak angle, not as primary reference for core mechanism
5. **Qwen trains with system messages natively** (ChatML) — relevant because our Qwen-2.5-7B experiments inject personas as system prompts, which aligns with Qwen's training format
6. **System prompt sensitivity is extreme** — up to 76-point accuracy swings from formatting alone (Sclar et al.), and 1-97% on security tasks from prompt wording (Litvak). Our persona injection via system prompts is operating on a fundamentally brittle control surface
7. **Instruction hierarchy is an unsolved problem** — even frontier models achieve only ~40-48% accuracy resolving multi-tier instruction conflicts (ManyIH, IHEval). Societal hierarchy framings from pretraining exert stronger influence than system/user role separation ("Control Illusion")
8. **Character training dominates prompting** — Open Character Training (Maiya et al.) provides the first direct comparison, showing character training via Constitutional AI is more robust to adversarial prompting than both system prompt constraints and activation steering. This validates our focus on training-time persona manipulation rather than prompt-time
9. **System prompt extraction is near-universal** — JustAsk achieves full/near-complete recovery on 41 commercial models. System prompts are not a security boundary, reinforcing that training-time defenses (our Aim 5) cannot rely on system prompt integrity
10. **Chat template is itself an attack surface** — BadTemplate achieves 100% ASR by exploiting template customizability. The chat template mediating between system prompt and model is fragile, which matters for our Qwen ChatML-based persona injection setup
11. **The constitution/character-training vs. system-prompt distinction matters** — Anthropic's constitution (training-time) shapes values that system prompt changes and cultural context cannot override (Pourdavood). This maps to our layered model: pretraining + post-training set deep persona; system prompt only modulates surface behavior

---

## Reference list (new additions, April 2026 update)

Papers newly added in this update, organized by section:

**Question 3 (Sensitivity):**
1. Sclar et al. (ICLR 2024) "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design" arXiv:2310.11324
2. Zhuo et al. (2024) "ProSA: Assessing and Understanding the Prompt Sensitivity of LLMs" arXiv:2410.12405
3. Razavi et al. (2025) "Benchmarking Prompt Sensitivity in Large Language Models" arXiv:2502.06065
4. Errica et al. (NAACL 2025) "What Did I Do Wrong? Quantifying LLMs' Sensitivity and Consistency" arXiv:2406.12334
5. Hua et al. (2025) "Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs" arXiv:2509.01790
6. Ngweta et al. (2025) "Towards LLMs Robustness to Changes in Prompt Format Styles" arXiv:2504.06969
7. Cheng & Mastropaolo (2026) "An Empirical Study on the Effects of System Prompts for Code Generation" arXiv:2602.15228
8. Litvak (2026) "The System Prompt Is the Attack Surface" arXiv:2603.25056
9. Hui et al. (2024) "PLeak: Prompt Leaking Attacks against LLM Applications" arXiv:2405.06823
10. Das et al. (2025) "System Prompt Extraction Attacks and Defenses in LLMs" arXiv:2505.23817
11. Zheng et al. (2026) "Just Ask: Curious Code Agents Reveal System Prompts in Frontier LLMs" arXiv:2601.21233
12. Zhuang et al. (2025) "ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks" arXiv:2505.11459
13. Wang et al. (2026) "BadTemplate: A Training-Free Backdoor Attack via Chat Template" arXiv:2602.05401
14. Qin et al. (2024) "SysBench: Can Large Language Models Follow System Messages?" arXiv:2408.10943
15. Lyu et al. (2024) "Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates" arXiv:2402.18540
16. Maiya et al. (2025) "Open Character Training: Shaping the Persona of AI Assistants" arXiv:2511.01689

**Question 4 (Layered architecture):**
17. Geng et al. (2025) "Control Illusion: The Failure of Instruction Hierarchies in LLMs" arXiv:2502.15851
18. Zhang et al. (2025) "IHEval: Evaluating Language Models on Following the Instruction Hierarchy" arXiv:2502.08745
19. Zhang et al. (2026) "Many-Tier Instruction Hierarchy in LLM Agents" arXiv:2604.09443
20. Chen et al. (2026) "HIPO: Instruction Hierarchy via Constrained Reinforcement Learning" arXiv:2603.16152
21. Wu et al. (2024) "Instructional Segment Embedding: Improving LLM Safety with Instruction Hierarchy" arXiv:2410.09102
22. Guo et al. (OpenAI, 2026) "IH-Challenge: A Training Dataset to Improve Instruction Hierarchy" arXiv:2603.10521
23. Pham & Le (2025) "PARASITE: Conditional System Prompt Poisoning to Hijack LLMs" arXiv:2505.16888
24. Pourdavood (2026) "Does Claude's Constitution Have a Culture?" arXiv:2603.28123

**Question 5 (Open-source models):**
25. Qwen Team (2024) "Qwen2.5 Technical Report" arXiv:2412.15115
26. DeepSeek-AI (2024) "DeepSeek-V3 Technical Report" arXiv:2412.19437
