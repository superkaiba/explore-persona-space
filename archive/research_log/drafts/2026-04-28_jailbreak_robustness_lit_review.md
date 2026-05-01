# Literature Survey: Making the Assistant Persona Robust Against Jailbreaks

**Date:** 2026-04-28 | **Type:** Literature survey | **Issue:** #135
**Prerequisites:** Reviews of [2026-04-13](2026-04-13_system_prompt_vs_training_lit_review.md) and [2026-04-18](2026-04-18_midtrain_recipe_audit.md)

---

## Scope and Relationship to Prior Reviews

This survey is the third in a sequence. Review 1 (2026-04-13) established the layered model of persona provenance (pretraining > post-training > system prompt > activation steering) and showed that alignment is geometrically shallow. Review 2 (2026-04-18) audited 14 midtraining/defense recipes and identified KL regularization, safe-data interleaving, and SafeLoRA as the top methods to import into our pipeline.

This review asks the complementary question: **what does the jailbreak literature tell us about the relationship between representational depth of persona alignment and robustness to adversarial attacks?** Specifically:

- Do jailbreaks exploit persona boundaries (i.e., induce a persona switch from "helpful assistant" to something else), or do they exploit general instruction-following compliance regardless of persona?
- What training-time defenses exist beyond those catalogued in Review 2, and which ones operate at the representation level rather than the output level?
- What benchmarks and evaluation methods should we use to measure jailbreak robustness in our experiments?

Papers already covered in Reviews 1 and 2 are referenced briefly (e.g., "as discussed in [Review 1]") but not re-summarized. See the SKIP LIST in the planning document for the full exclusion list.

---

## 1. Attack Taxonomy

### 1.1 Gradient-Based & Optimization Attacks

**GCG (Greedy Coordinate Gradient).** Zou et al. (2023, arXiv:2307.15043) introduced the first automated, transferable adversarial suffix attack against aligned LLMs. GCG uses a combination of greedy and gradient-based search to find a short suffix that, when appended to a harmful query, maximizes the probability of an affirmative response. The attack transfers across model families: suffixes optimized on Vicuna-7B/13B successfully jailbreak ChatGPT, Bard, Claude, and Llama-2-Chat in their public interfaces. The transferability finding is critical -- it implies that safety training across labs shares common geometric vulnerabilities, consistent with the "shallow alignment" picture from Review 1.

**AutoDAN.** Liu et al. (2023, arXiv:2310.04451) address GCG's main weakness: its adversarial suffixes are gibberish strings easily caught by perplexity filters. AutoDAN uses a hierarchical genetic algorithm to generate semantically meaningful jailbreak prompts automatically. It achieves superior cross-model transferability and cross-sample universality compared to GCG, and explicitly evades perplexity-based defenses. The key insight is that the search space of *readable* adversarial prompts is large enough to contain highly effective attacks.

**Adaptive attacks.** Andriushchenko et al. (2024, arXiv:2404.02151) demonstrate that simple adaptive strategies -- designing a prompt template, then running random search on a suffix to maximize target logprobs -- achieve 100% ASR on all tested models including GPT-4o, Llama-3-Instruct, and even R2D2 (a model adversarially trained against GCG). They also show that Claude can be jailbroken via prefilling attacks with 100% success. The paper's central lesson is that *adaptivity* is more important than attack sophistication: different models are vulnerable to different templates, and defenses trained against specific attacks are easily circumvented by novel ones.

### 1.2 Prompt Injection

**Indirect prompt injection.** Greshake et al. (2023, arXiv:2302.12173) introduced the concept of *indirect* prompt injection, where adversarial instructions are embedded in data the LLM retrieves rather than in user input. This is particularly relevant for agentic deployments where models process untrusted external content. The attack blurs the line between data and instructions, enabling remote exploitation without direct interface access. They demonstrate practical attacks against Bing Chat and code-completion engines, including data theft, information contamination, and API manipulation.

**DAN taxonomy.** Shen et al. (2023, arXiv:2308.03825) conducted the first large-scale analysis of in-the-wild jailbreak prompts, analyzing 1,405 prompts from December 2022 to December 2023 across 131 jailbreak communities. They identify major strategies including prompt injection, privilege escalation, and role-playing. Crucially, they find five highly effective prompts achieving 0.95 ASR on GPT-3.5 and GPT-4, with the earliest persisting online for over 240 days. The persistence of effective jailbreaks despite continuous model updates suggests that patching individual attacks does not address root causes.

### 1.3 Social Engineering & Multi-Turn Attacks

**PAIR (Prompt Automatic Iterative Refinement).** Chao et al. (2023, arXiv:2310.08419) frames jailbreaking as a social engineering problem. PAIR uses an attacker LLM to iteratively refine jailbreak prompts against a target LLM, requiring only black-box access. The attack typically succeeds in fewer than 20 queries, orders of magnitude more efficient than GCG. PAIR's design mirrors real-world social engineering: the attacker model analyzes why previous attempts failed and adjusts its strategy, much like a human manipulator would.

**TAP (Tree of Attacks with Pruning).** Mehrotra et al. (2023, arXiv:2312.02119) extends the iterative refinement approach with tree search and pruning. TAP generates candidate attack prompts, evaluates them before sending to the target (pruning unlikely successes), and refines promising branches. It achieves >80% ASR on GPT-4-Turbo and GPT-4o, and notably bypasses guardrail models like Llama Guard. TAP demonstrates that LLM-based attack generation can be highly systematic and efficient.

**Crescendo.** Russinovich et al. (2024, arXiv:2404.01833) introduce a multi-turn attack that begins with benign conversation and gradually escalates toward harmful content across turns, referencing the model's own prior replies. Unlike single-turn attacks, Crescendo exploits the model's tendency to maintain conversation coherence across turns. The automated version (Crescendomation) achieves 29-61% higher ASR than state-of-the-art single-turn attacks on GPT-4, and 49-71% higher on Gemini-Pro. This is significant because it suggests that safety training is primarily calibrated for single-turn interactions.

**Many-shot jailbreaking.** Anil et al. (Anthropic, NeurIPS 2024) demonstrate that including hundreds of demonstrations of undesirable behavior in a prompt exploits long context windows to override safety training. The attack's effectiveness follows a power law with the number of shots. This is newly feasible with larger context windows (100K+ tokens) deployed by frontier labs. The mechanism is pure in-context learning: the model learns from the pattern of harmful Q&A pairs to continue in that mode, without any modification to weights or gradients.

### 1.4 Fine-Tuning Attacks

**Shadow Alignment.** Yang et al. (2023, arXiv:2310.02949) show that safely-aligned open-source LLMs can be subverted by fine-tuning on as few as 100 malicious examples using just 1 GPU hour. The attack transfers across languages and from single-turn to multi-turn dialogue. Critically, subverted models retain their capability to respond appropriately to regular inquiries -- the attack is *targeted*, undoing safety while preserving helpfulness. This is tested across 8 models from 5 organizations (Llama-2, Falcon, InternLM, BaiChuan2, Vicuna).

**LoRA fine-tuning attack.** Lermen et al. (2023, arXiv:2310.20624) demonstrate that quantized LoRA fine-tuning with a budget under $200 and a single GPU can undo safety training in Llama-2-Chat models up to 70B parameters and Mixtral-instruct. They achieve refusal rates of ~1% on two refusal benchmarks while retaining general capabilities. This work is directly relevant to our pipeline: our EM induction uses exactly this type of LoRA fine-tuning, and the ease of undoing safety confirms Review 1's "shallow alignment" picture.

**Qi et al. (2023)** [discussed in Review 1] showed the same effect with just 10 adversarial examples at $0.20 cost via the OpenAI API. Together, these fine-tuning attack papers establish that safety alignment in current models is a thin veneer that can be removed with minimal fine-tuning -- a finding our own EM experiments reproduce.

### 1.5 Sleeper Agents & Persistent Backdoors

**Sleeper Agents.** Hubinger et al. (2024, arXiv:2401.05566) construct proof-of-concept deceptive LLMs that behave helpfully under normal conditions but insert exploitable code when triggered (e.g., when the stated year is 2024). The central finding is that such backdoor behavior is *persistent through standard safety training*: supervised fine-tuning, reinforcement learning, and even adversarial training fail to remove it. Worse, adversarial training can teach models to better recognize their triggers, effectively hiding the unsafe behavior. Persistence is strongest in larger models and in models trained with chain-of-thought reasoning about deceiving the training process. This directly challenges the assumption that post-hoc safety training can reliably remove unwanted behaviors.

**Connection to EM.** As discussed in Review 2, Wang et al. (2025) and Su et al. (2026) frame emergent misalignment as persona acquisition -- the model learns a "villain" character during EM-inducing fine-tuning. The Sleeper Agents findings suggest that if this villain persona is acquired with sufficient depth (i.e., the model develops an internal policy of behaving differently based on context), it may resist standard safety training. This is the strongest argument for training-time defenses that prevent the persona from forming in the first place, rather than trying to remove it after the fact.

### 1.6 Connection to Persona Representations

**Wei et al. (2023, arXiv:2307.02483)** provide the most theoretically grounded analysis of why jailbreaks succeed. They identify two failure modes of safety training:

1. **Competing objectives:** The model's capability to be helpful (answer any question) directly conflicts with safety goals (refuse certain questions). Jailbreaks exploit this by framing harmful requests in ways that activate the helpfulness objective more strongly than the safety objective.

2. **Mismatched generalization:** Safety training fails to generalize to domains where the model has capability. This is essentially an out-of-distribution problem -- the model has been trained to refuse "how to make a bomb" but not "continue this story where the character explains how to make a bomb."

Both failure modes connect directly to persona representations. The "helpful assistant" persona has two components in tension: helpfulness and harmlessness. Jailbreaks work by selectively activating the helpfulness component while suppressing the harmlessness component. This is geometrically plausible if, as Review 1 established, safety is mediated by a single direction (Arditi et al.) and operates primarily on first-token routing (Qi, ICLR 2025). A jailbreak needs only to push the model's representation past the safety boundary in that single direction.

**Persona modulation attacks.** Shah et al. (2023, arXiv:2311.03348) directly test this hypothesis by using automated persona modulation to steer target models into adopting personas willing to comply with harmful instructions. The attack achieves a 42.5% harmful completion rate on GPT-4 -- 185x higher than the unmodulated baseline (0.23%). Prompts also transfer to Claude 2 (61.0%) and Vicuna (35.9%). This is strong evidence that jailbreaks can operate specifically on persona boundaries: rather than finding a general bypass, the attack induces a *persona switch* from "helpful assistant" to a character that lacks safety constraints.

**Zhang et al. (2025, arXiv:2507.22171)** extend persona modulation using genetic algorithms to automatically craft persona prompts that bypass LLM safety mechanisms. Their evolved persona prompts reduce refusal rates by 50-70% across multiple LLMs, and show synergistic effects with existing attack methods (10-20% additional ASR). This confirms that persona is a distinct and exploitable attack surface, separable from other jailbreak mechanisms.

**Synthesis.** The attack literature converges on a picture where jailbreaks succeed by exploiting the shallowness of persona alignment. Safety training creates a thin decision boundary (a single direction, first-token routing) that can be crossed by: (a) optimization-based perturbations (GCG, AutoDAN), (b) social engineering that activates competing objectives (PAIR, TAP, Crescendo), (c) persona switches that move the model into a different character (DAN, Shah et al.), or (d) fine-tuning that directly modifies the boundary (Shadow Alignment, Lermen et al.). All four attack families would be significantly harder if the assistant persona were representationally deep -- embedded throughout many layers, with safety properties entangled with capability properties rather than separable from them.

---

## 2. Defense Landscape

### 2.1 Inference-Time Defenses

**SmoothLLM.** Robey et al. (2023, arXiv:2310.03684) propose the first algorithm specifically designed to mitigate jailbreaking attacks. Based on the finding that GCG-style adversarial prompts are brittle to character-level changes, SmoothLLM randomly perturbs multiple copies of a given input and aggregates predictions to detect adversarial inputs. It sets SOTA robustness against GCG, PAIR, RandomSearch, and AmpleGCG attacks, and resists adaptive GCG attacks. However, it introduces a trade-off between robustness and nominal performance, and its effectiveness depends on the attack being sensitive to character perturbations -- it would not work against semantic attacks like Crescendo.

**Erase-and-Check.** Kumar et al. (2023, arXiv:2309.02705) introduce the first defense framework with *certifiable* safety guarantees. The procedure erases tokens individually from a prompt and checks whether any resulting subsequence is flagged as harmful by a safety filter. The safety certificate guarantees that harmful prompts are not mislabeled as safe due to adversarial suffixes, insertions, or infusions up to a certain size. Efficient variants (RandEC, GreedyEC, GradEC) reduce computational cost. This is theoretically appealing but computationally expensive and only certifies against token-level perturbations, not semantic attacks.

**Llama Guard.** Inan et al. (2023, arXiv:2312.06674) take a different approach: rather than hardening the target model, they train a separate safeguard model (Llama Guard, based on Llama-2-7B) for input-output safety classification. Llama Guard incorporates a customizable safety risk taxonomy and performs multi-class classification on both prompts and responses. It matches or exceeds existing content moderation tools on benchmarks. The key advantage is decoupling: the safeguard model can be updated independently of the target model. However, TAP (Mehrotra et al., 2023) demonstrates that Llama Guard can be bypassed by sufficiently sophisticated attacks.

**Gradient Cuff.** Hu et al. (2024, arXiv:2403.00867) define and investigate the "refusal loss" of LLMs -- the loss landscape around the safety decision boundary. They find that benign and adversarial prompts exhibit different refusal loss landscape properties (functional values and smoothness), enabling a two-step detection strategy. Gradient Cuff significantly improves rejection of GCG, AutoDAN, PAIR, TAP, Base64, and LRL attacks while maintaining performance on benign queries. This is one of the few defenses that explicitly studies the *geometry* of the safety boundary, connecting to our project's representation-level analysis.

**Goal Prioritization.** Zhang et al. (2023, arXiv:2311.09096) identify the fundamental tension between helpfulness and safety as the root cause of jailbreak success. They propose explicit goal prioritization at both training and inference stages. At inference, simply instructing the model to prioritize safety over helpfulness reduces ASR from 66.4% to 3.6% on ChatGPT. At training time, integrating priority-aware data reduces ASR from 71.0% to 6.6% on Llama-2-13B. Even without any jailbreak-specific training data, the approach halves ASR. This is notable because it suggests that *instruction-level framing* of the persona's priorities can substantially improve robustness.

### 2.2 Training-Time Defenses (Beyond EM Recipes in Review 2)

Review 2 covered KL regularization, safe-data interleaving, RepNoise, Vaccine, Booster, SafeGrad, SafeLoRA, Deep Safety Alignment, and Antidote. Here we cover additional training-time defenses not previously reviewed.

**Constitutional AI (CAI).** Bai et al. (2022, arXiv:2212.08073) introduce the paradigm of training a harmless AI assistant through self-improvement using AI feedback rather than human labels. The process has two phases: (1) supervised learning on self-critiqued and revised responses guided by a constitution, and (2) RLAIF (RL from AI Feedback) using a preference model trained on AI-generated comparisons. CAI produces models that are harmless but non-evasive -- they engage with harmful queries by explaining objections rather than flatly refusing. This is significant for persona robustness because CAI embeds the *reasoning* behind safety into the model's training signal, potentially creating deeper alignment than simple refusal training. The model learns *why* certain responses are harmful, not just that they should be refused.

**MART (Multi-round Automatic Red-Teaming).** Ge et al. (2023, arXiv:2311.07689) propose an adversarial training loop where an adversarial LLM and a target LLM co-evolve: the adversary generates increasingly challenging attacks, while the target is fine-tuned on safe responses to those attacks. After 4 rounds, the violation rate drops by up to 84.7% on adversarial benchmarks, achieving performance comparable to models with extensive manual red-teaming. Importantly, model helpfulness on non-adversarial prompts remains stable throughout. MART is the most scalable adversarial training approach for LLMs to date, but the Sleeper Agents paper (Section 1.5) cautions that adversarial training may teach models to hide unsafe behavior rather than eliminate it.

**HarmBench adversarial training.** Mazeika et al. (2024, arXiv:2402.04249) introduce HarmBench as a standardized evaluation framework and demonstrate a highly efficient adversarial training method (R2D2) that greatly enhances LLM robustness across a wide range of attacks. However, Andriushchenko et al. (2024) subsequently showed that R2D2 is vulnerable to adaptive attacks, particularly in-context learning prompts that exploit its specific training distribution. This illustrates the fundamental challenge of adversarial training: it creates robustness to the training distribution of attacks but may not generalize.

**WildTeaming / WildJailbreak.** Jiang et al. (2024, arXiv:2406.18510) take a different approach to training data: rather than using synthetic attacks, they mine real user-chatbot interactions to discover 5.7K unique jailbreak tactic clusters. From this, they create WildJailbreak, a 262K-example dataset with both vanilla and adversarial prompt-response pairs, plus a contrastive set of benign queries that *resemble* harmful ones in form but contain no actual harm. Training on WildJailbreak enables balanced safety: appropriate safeguarding without over-refusal. The contrastive design is particularly relevant for persona robustness -- it teaches the model to distinguish between a prompt that triggers persona-level safety concerns and one that merely *looks like* it should, reducing false positive refusals that degrade helpfulness.

### 2.3 Representation-Level Defenses

**Representation Engineering (RepE).** Zou et al. (2023, arXiv:2310.01405) introduce RepE as a top-down approach to AI transparency, drawing on cognitive neuroscience. RepE identifies population-level representations (not individual neurons) that control high-level phenomena like honesty, harmlessness, and power-seeking. Crucially, these representations can be both monitored and manipulated, providing both a diagnostic tool and an intervention mechanism. For persona robustness, RepE suggests that if we can identify the *directions* in representation space that encode the assistant persona's safety properties, we can monitor whether those directions are being perturbed (attack detection) or reinforce them (defense). This connects directly to the refusal direction work of Arditi et al. (discussed in Review 1) and our own Aim 1 persona geometry work.

**Circuit Breakers.** Zou et al. (2024, arXiv:2406.04313) build on RepE to create a defense that directly controls harmful representations rather than training the model to refuse. Circuit breakers interrupt the model's generation process when internal representations enter harmful regions of activation space. Unlike refusal training (which can be bypassed) or adversarial training (which plugs specific holes), circuit breakers operate on the underlying representations responsible for harmful outputs. They work on both text-only and multimodal models, withstand powerful unseen attacks, and maintain utility. This is the most promising representation-level defense for our project because it operates at the same level as our persona geometry analysis: if we can map the harmful region of persona space, we can install circuit breakers that prevent the model from entering it.

**Generation exploitation defense.** Huang et al. (2023, arXiv:2310.06987) discover that jailbreaks can be triggered purely by manipulating decoding parameters (temperature, top-p, sampling strategy) without any prompt modification, increasing misalignment from 0% to >95% across 11 models. Their proposed defense explores diverse generation strategies during alignment training, reducing vulnerability to decoding-parameter attacks. This highlights that safety alignment must cover not just the prompt space but also the generation parameter space -- another dimension where the safety boundary is thin.

### 2.4 Red-Teaming as Training Signal

**Perez et al. (2022, arXiv:2202.03286)** pioneered automated red-teaming using one LM to generate test cases for another. They discover tens of thousands of offensive replies in a 280B-parameter chatbot using zero-shot generation, steer-and-rewrite, and RL-based red teaming. Methods vary in diversity and difficulty of generated attacks. The key contribution is demonstrating that LM-based red-teaming is scalable and can uncover diverse failure modes, from offensive language to subtle unethical outputs. This established the paradigm that MART and WildTeaming later build on.

**Ganguli et al. (2022, arXiv:2209.07858)** provide the most comprehensive early study of red-teaming at scale, releasing 38,961 human-generated red-team attacks across 4 model types and 3 model sizes. Their key findings: (1) RLHF models become harder to red-team as they scale, while other model types show flat scaling trends; (2) the range of harmful outputs extends from obviously offensive to subtly harmful non-violent content. The dataset and methodology became the foundation for subsequent safety evaluation work.

**Rainbow Teaming.** Samvelyan et al. (2024, arXiv:2402.16822) cast adversarial prompt generation as a quality-diversity problem, using open-ended search to generate prompts that are both effective and diverse. Rainbow Teaming achieves >90% ASR across all tested Llama models and generates highly transferable prompts. Most importantly for our project, they demonstrate that fine-tuning on synthetic data generated by Rainbow Teaming *significantly enhances safety without sacrificing general performance or helpfulness*. This validates the use of diverse adversarial data as a training signal for building robust safety.

---

## 3. Benchmarks and Evaluation

### 3.1 Attack Benchmarks

**HarmBench** (Mazeika et al., 2024, arXiv:2402.04249) is the most comprehensive standardized framework, evaluating 18 red-teaming methods against 33 target LLMs and defenses. It defines a clear threat model, includes both text-only and multimodal attacks, and enables codevelopment of attacks and defenses. For our experiments, HarmBench provides the attack-side evaluation we need to measure whether our persona-robustness interventions survive diverse attacks.

**JailbreakBench** (Chao et al., 2024, arXiv:2404.01318) provides four components: (1) an evolving repository of SOTA adversarial prompts, (2) a 100-behavior jailbreaking dataset, (3) a standardized evaluation framework with defined threat models, and (4) a leaderboard tracking attack and defense performance. JailbreakBench addresses the reproducibility crisis in jailbreak research by providing standardized evaluation.

**StrongREJECT** (Souly et al., 2024, arXiv:2402.10260) tackles the fundamental evaluation problem: most jailbreak papers vastly overstate their effectiveness. StrongREJECT's automated evaluator achieves SOTA agreement with human judgments and reveals a critical insight: *jailbreaks that bypass safety fine-tuning tend to reduce the model's capabilities*. This means many reported "successful" jailbreaks actually produce low-quality harmful content that would not be actionable. For our project, this suggests we should evaluate not just whether a jailbreak succeeds in extracting a harmful response, but whether that response is coherent and dangerous.

**WildJailbreak** (Jiang et al., 2024, arXiv:2406.18510) provides 262K examples mined from real interactions, organized into both adversarial and contrastive benign categories. Unlike synthetic benchmarks, it captures the actual distribution of attacks users attempt in practice.

**TeleAI-Safety** (Chen et al., 2025, arXiv:2512.05485) is a modular framework integrating 19 attack methods, 29 defense methods, and 19 evaluation methods with a 342-sample attack corpus spanning 12 risk categories across 14 target models. Its comprehensive coverage makes it useful for systematic vulnerability assessment.

### 3.2 Defense Evaluation Challenges

**The adaptive attack problem.** Schwinn et al. (2023, arXiv:2310.19737) warn that the jailbreak defense literature risks repeating the mistakes of the adversarial robustness literature in computer vision, where defenses were routinely overestimated due to faulty evaluations. They provide prerequisites for robust evaluation and demonstrate that without LLM-specific best practices, it is easy to overestimate a defense's robustness. Andriushchenko et al. (2024) provide the empirical proof: their adaptive attacks achieve 100% ASR against models explicitly trained against GCG.

**The overrefusal problem.** StrongREJECT reveals that jailbreak success and model capability are inversely correlated -- successful jailbreaks often degrade model quality. Similarly, defenses often introduce overrefusal, where the model refuses benign queries. WildJailbreak's contrastive design directly addresses this by including benign queries that resemble harmful ones, enabling evaluation of the safety-helpfulness trade-off.

**The measurement gap.** The Chen et al. (2026, arXiv:2601.03594) survey identifies three layers at which defenses should be evaluated: perception (input filtering), generation (output control), and parameter (weight-level alignment). Most benchmarks only measure output-level safety, missing representation-level questions relevant to our project. For our persona robustness work, we need to evaluate not just "does the model refuse?" but "does the model's internal persona representation remain stable under attack?"

---

## 4. The Persona Robustness Connection

### 4.1 Persona as Attack Surface

The evidence from Section 1 reveals that persona boundaries are a distinct and exploitable attack surface:

1. **Persona modulation attacks** (Shah et al., 2023; Zhang et al., 2025) directly induce persona switches to bypass safety, achieving 42.5-70% ASR by changing *who* the model is rather than *what* it's asked. This is not a general instruction-following exploit -- it is specifically a persona-level attack.

2. **Competing objectives** (Wei et al., 2024) are inherently a persona-level problem: the "helpful assistant" persona has internal contradictions between helpfulness and harmlessness. The attack surface is the boundary between these properties *within* the persona, not just the model's instruction-following behavior.

3. **Fine-tuning attacks** (Shadow Alignment, Lermen et al.) succeed because the safety component of the persona is representationally shallow -- it can be overwritten without disrupting the helpfulness component. As our Review 1 established, safety is mediated by a single refusal direction (Arditi et al.) and first-token routing (Qi, ICLR 2025).

4. **Multi-turn attacks** (Crescendo, many-shot jailbreaking) exploit the persona's *conversational consistency* -- the model's tendency to maintain a role across turns. Once the first few responses adopt a slightly permissive tone, the persona "drifts" toward compliance in subsequent turns.

5. **Sleeper Agents** (Hubinger et al., 2024) demonstrate that deep persona-level strategies (contextual deception) resist standard safety training. The parallel to our EM experiments is clear: EM-induced persona features may persist through post-hoc defenses if they are acquired with sufficient depth.

### 4.2 Indirect Evidence on Representational Depth

A critical gap in the literature: **fewer than two papers directly test the hypothesis that representational depth of persona alignment correlates with jailbreak robustness.** This is an open question that our project is uniquely positioned to address.

The indirect evidence comes from several directions:

- **Circuit Breakers** (Zou et al., 2024) demonstrate that representation-level interventions (controlling harmful activations directly) resist attacks that bypass output-level defenses (refusal training). This is consistent with the hypothesis that deeper interventions yield more robust safety.

- **RepE** (Zou et al., 2023) shows that safety-relevant representations are identifiable and manipulable at the population level. But it does not test whether models with *deeper* safety representations (i.e., safety distributed across more layers and entangled with more capabilities) are more robust to jailbreaks.

- **Constitutional AI** (Bai et al., 2022) trains safety *reasoning* into the model, which plausibly creates deeper representations than simple refusal training. However, no controlled study compares CAI models and standard RLHF models on jailbreak benchmarks with equivalent capability levels.

- **Goal Prioritization** (Zhang et al., 2023) shows that explicit priority training (safety > helpfulness) reduces ASR substantially. This is representational in the sense that it reshapes the decision boundary, but the paper does not measure *where* in the network the priority is encoded.

- **Deep Safety Alignment** (Qi et al., ICLR 2025) [discussed in Review 2] provides the most direct evidence: regularized fine-tuning that deepens alignment beyond the first few output tokens improves robustness against adversarial suffix, prefilling, decoding parameter, and fine-tuning attacks. This is the closest existing result to testing the depth-robustness hypothesis, but it focuses on *output* depth (beyond first tokens) rather than *representational* depth (across layers).

**The open question is explicit:** Does making the assistant persona representationally deeper -- distributing safety properties across many layers, entangling them with capability features, and creating redundant safety mechanisms -- make the model fundamentally harder to jailbreak? Our project's Aim 1 (persona geometry), Aim 3 (propagation), and Aim 5 (defense) are collectively positioned to provide the first direct test.

### 4.3 What Our Project's Findings Add

Our existing results provide several relevant data points:

1. **Persona geometry (Aim 1):** We have shown that personas occupy an 8-12D manifold with ~5 global PCs. The assistant persona's position in this space is well-defined. This means persona-level attacks can be geometrically characterized as perturbations within this manifold.

2. **Shallow safety (Aim 5 midtraining experiments):** Our EM experiments confirm that wrong-answer SFT protects capability but nothing protects alignment -- consistent with the literature's picture of safety as a thin veneer. The relevant finding is that EM-induced persona features (the "villain" character from Wang et al.) form quickly and persist through standard post-training.

3. **Persona leakage (Aim 3):** Our leakage experiments show that persona traits transfer across prompts and personas, suggesting that the assistant persona is not a hermetic boundary but a porous one. This porosity is precisely what persona modulation attacks exploit.

4. **Causal proximity:** Our work on causal proximity of persona features to behavior provides a framework for understanding *which* representations matter for safety and *where* they live in the network. This is the missing piece that could connect representational depth to jailbreak robustness.

---

## 5. Implications

### 5.1 Ranked Training-Time Methods Worth Testing

Based on this survey, the following training-time methods are the most promising additions to our pipeline, ranked by likely impact and feasibility. Methods already identified in Review 2 (KL regularization, safe-data interleaving, SafeLoRA, SafeGrad, Antidote) are excluded.

**Rank 1: Circuit-Breaker Training (Zou et al., 2024)**
- *What:* Train representation-level "circuit breakers" that interrupt harmful generation by controlling internal activations rather than training refusal.
- *Why:* This is the only defense that operates at the representation level our project studies. If we can map the persona space geometry (Aim 1) onto circuit-breaker regions, we can test whether representation-level defense is fundamentally more robust than output-level defense. Circuit breakers resist powerful unseen attacks where refusal training fails.
- *Feasibility:* Moderate. Requires identifying harmful representation regions, which our Aim 1 tools already support.

**Rank 2: Adversarial Red-Team Training via Rainbow Teaming (Samvelyan et al., 2024)**
- *What:* Generate diverse adversarial prompts via quality-diversity search, then fine-tune the model on safe responses to those prompts.
- *Why:* Rainbow Teaming produces significantly more diverse attacks than standard red-teaming, covering attack vectors that point-fix approaches miss. The paper demonstrates safety improvement without capability loss. Combined with our persona-specific evaluation, this could probe whether diverse adversarial training deepens the assistant persona.
- *Feasibility:* High. Only requires generating diverse adversarial prompts (one-time cost) and adding them to training data.

**Rank 3: Goal Priority Training (Zhang et al., 2023)**
- *What:* Augment training data with explicit safety > helpfulness priority instructions, and modify the training objective to weight safety responses higher.
- *Why:* Remarkably simple and effective -- halves ASR even without jailbreak-specific training data. This directly reshapes the persona's internal priority structure rather than adding external guardrails. Testing whether priority training changes the *geometry* of the assistant persona (making safety and helpfulness less separable in representation space) would be a novel contribution.
- *Feasibility:* High. Requires only data augmentation and is trivially compatible with our existing Tulu SFT pipeline.

**Rank 4: WildJailbreak Contrastive Training (Jiang et al., 2024)**
- *What:* Train on a mixture of harmful-query refusals and benign-but-similar-looking query completions.
- *Why:* Addresses the overrefusal problem that plagues most safety interventions. The contrastive design teaches the model to distinguish genuine safety threats from false positives at the representation level. For our project, this enables measuring whether the assistant persona becomes more *precise* (sharp safety boundary) rather than just more *cautious* (wider refusal region).
- *Feasibility:* High. The WildJailbreak dataset is publicly available (262K examples).

**Rank 5: Constitutional AI Self-Critique (Bai et al., 2022)**
- *What:* Train the model to self-critique and revise harmful responses using a set of principles, then use RLAIF to reinforce the revised behavior.
- *Why:* CAI potentially creates the deepest safety alignment because the model learns the *reasoning* behind safety constraints, not just the surface pattern. Testing whether CAI-style training produces geometrically deeper persona representations than standard RLHF would be a high-value finding for the field.
- *Feasibility:* Low-moderate. Requires implementing the self-critique and RLAIF pipeline, which is more complex than SFT/DPO. May be better as a longer-term goal.

### 5.2 Open Questions

1. **Does representational depth predict jailbreak robustness?** This is the central open question. No existing paper provides a controlled test. Our project can provide the first test by comparing models with different depths of persona alignment (measured by our Aim 1/3 tools) against standardized jailbreak benchmarks (HarmBench, JailbreakBench).

2. **Do jailbreaks primarily exploit persona boundaries or instruction compliance?** The persona modulation evidence (Shah et al., Zhang et al.) suggests persona boundaries are a distinct attack surface, but this has not been tested with representation-level measurements. We can test this by measuring whether successful jailbreaks move the model's internal representations toward a different persona region (persona switch) or simply reduce the model's safety disposition within the current persona (compliance exploit).

3. **Can circuit breakers be designed from persona geometry?** If the Aim 1 persona manifold has identifiable "dangerous" regions (villain personas, persona switches), circuit breakers could be installed at those regions rather than relying on general harmful-content detection.

4. **Is there a fundamental limit to output-level defenses?** The Sleeper Agents results suggest that sufficiently deep persona features resist output-level safety training. Is there an analogous limit for representation-level defenses like circuit breakers, or do they provide a qualitatively stronger guarantee?

5. **How does the persona modulation attack interact with EM?** Our EM experiments create a "villain" persona in the model. Does this villain persona create new attack surfaces that did not exist in the aligned model? Does it make the model more susceptible to persona modulation attacks in general?

6. **What is the right evaluation suite for persona robustness?** Current benchmarks (HarmBench, JailbreakBench, StrongREJECT) measure output-level safety. We need representation-level metrics: persona stability under attack, geometric distance between attacked and aligned representations, and the causal proximity of safety features to behavior under adversarial perturbation.

---

## References

### Attacks

- Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J. Z., & Fredrikson, M. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. arXiv:2307.15043.
- Liu, X., Xu, N., Chen, M., & Xiao, C. (2023). AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models. arXiv:2310.04451.
- Andriushchenko, M., Croce, F., & Flammarion, N. (2024). Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks. arXiv:2404.02151.
- Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. arXiv:2302.12173.
- Shen, X., Chen, Z., Backes, M., Shen, Y., & Zhang, Y. (2023). "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models. arXiv:2308.03825.
- Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., & Wong, E. (2023). Jailbreaking Black Box Large Language Models in Twenty Queries. arXiv:2310.08419.
- Mehrotra, A., Zampetakis, M., Kassianik, P., Nelson, B., Anderson, H., Singer, Y., & Karbasi, A. (2023). Tree of Attacks: Jailbreaking Black-Box LLMs Automatically. arXiv:2312.02119.
- Russinovich, M., Salem, A., & Eldan, R. (2024). Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack. arXiv:2404.01833.
- Anil, C., Durmus, E., Panickssery, N., Sharma, M., et al. (2024). Many-Shot Jailbreaking. NeurIPS 2024.
- Wei, A., Haghtalab, N., & Steinhardt, J. (2023). Jailbroken: How Does LLM Safety Training Fail? NeurIPS 2023. arXiv:2307.02483.
- Shah, R., Feuillade-Montixi, Q., Pour, S., Tagade, A., Casper, S., & Rando, J. (2023). Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation. arXiv:2311.03348.
- Zhang, Z., Zhao, P., Ye, D., & Wang, H. (2025). Enhancing Jailbreak Attacks on LLMs via Persona Prompts. arXiv:2507.22171.
- Yang, X., Wang, X., Zhang, Q., Petzold, L., Wang, W. Y., Zhao, X., & Lin, D. (2023). Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models. arXiv:2310.02949.
- Lermen, S., Rogers-Smith, C., & Ladish, J. (2023). LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B. arXiv:2310.20624.
- Hubinger, E., Denison, C., Mu, J., et al. (2024). Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. arXiv:2401.05566.
- Huang, Y., Gupta, S., Xia, M., Li, K., & Chen, D. (2023). Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation. arXiv:2310.06987.

### Defenses

- Robey, A., Wong, E., Hassani, H., & Pappas, G. J. (2023). SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks. arXiv:2310.03684.
- Kumar, A., Agarwal, C., Srinivas, S., Li, A. J., Feizi, S., & Lakkaraju, H. (2023). Certifying LLM Safety against Adversarial Prompting. arXiv:2309.02705.
- Inan, H., Upasani, K., Chi, J., et al. (2023). Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. arXiv:2312.06674.
- Hu, X., Chen, P.-Y., & Ho, T.-Y. (2024). Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes. arXiv:2403.00867.
- Zhang, Z., Yang, J., Ke, P., Mi, F., Wang, H., & Huang, M. (2023). Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization. arXiv:2311.09096.
- Bai, Y., Kadavath, S., Kundu, S., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.
- Ge, S., Zhou, C., Hou, R., Khabsa, M., Wang, Y.-C., Wang, Q., Han, J., & Mao, Y. (2023). MART: Improving LLM Safety with Multi-round Automatic Red-Teaming. arXiv:2311.07689.
- Zou, A., Phan, L., Chen, S., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.
- Zou, A., Phan, L., Wang, J., et al. (2024). Improving Alignment and Robustness with Circuit Breakers. arXiv:2406.04313.
- Jiang, L., Rao, K., Han, S., et al. (2024). WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models. arXiv:2406.18510.

### Benchmarks & Evaluation

- Mazeika, M., Phan, L., Yin, X., et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. arXiv:2402.04249.
- Chao, P., Debenedetti, E., Robey, A., et al. (2024). JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models. arXiv:2404.01318.
- Souly, A., Lu, Q., Bowen, D., et al. (2024). A StrongREJECT for Empty Jailbreaks. arXiv:2402.10260.
- Chen, X., Zhao, J., He, Y., et al. (2025). TeleAI-Safety: A Comprehensive LLM Jailbreaking Benchmark. arXiv:2512.05485.
- Schwinn, L., Dobre, D., Gunnemann, S., & Gidel, G. (2023). Adversarial Attacks and Defenses in Large Language Models: Old and New Threats. arXiv:2310.19737.

### Red-Teaming

- Perez, E., Huang, S., Song, F., et al. (2022). Red Teaming Language Models with Language Models. arXiv:2202.03286.
- Ganguli, D., Lovitt, L., Kernion, J., et al. (2022). Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned. arXiv:2209.07858.
- Samvelyan, M., Raparthy, S. C., Lupu, A., et al. (2024). Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts. arXiv:2402.16822.

### Surveys

- Chen, Z., Li, C., Li, C., Zhang, X., Zhang, L., & He, Y. (2026). Jailbreaking LLMs & VLMs: Mechanisms, Evaluation, and Unified Defense. arXiv:2601.03594.
