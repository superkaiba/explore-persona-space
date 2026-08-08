# Single-token steering & the pre-assistant newline — literature deep dive

*Compiled 2026-08-06. Scope: activation steering / representation interventions applied at **specific token positions**, with a focus on the structural tokens at the user→assistant boundary (the newline / role-header / `<|assistant|>`-style tokens right before the model starts speaking). Sources are arXiv (via MCP), the Transformer Circuits thread, and LessWrong/AF. Where I only read an abstract vs. the full text is flagged.*

---

## 0. The one-paragraph answer

The token position where you inject a steering vector is **not** an implementation detail — it interacts with what kind of signal lives there. Two regimes dominate the literature:

1. **Single "waypoint" tokens carry compressed, causally-potent summaries.** The strongest single-token results are **task/function vectors** (patch one vector at the last/separator token → the model does the task zero-shot) and the **refusal direction** (extracted at a *post-instruction chat-template token*, causally toggles refusal). These positions aggregate upstream content and act as a handle.
2. **The newline / paragraph-break / pre-response boundary is where models "plan" the upcoming span.** Anthropic's poetry-planning result, ParaScopes, Future Lens, and the Natural Language Autoencoder work all localize forward-looking / planning content to the `\n`/`\n\n` transition — and show you can *steer* by editing that single position.

Separately, the **very first token (BOS/position 0)** is special for a mechanistic reason (attention sink / massive activations) rather than a semantic one — worth knowing because chat templates put structural tokens there and people sometimes conflate "structural token" with "sink."

For practice: **applying a vector at *all* positions (prompt + generated) is usually the strongest and most stable choice**, last-prompt-token-only is weakest, and the interesting science is in *why specific positions work at all*.

---

## 1. Foundational steering methods — and where each one actually intervenes

These are the methods everything else builds on. The position choice is often buried in a sentence and rarely ablated in the original papers.

| Method | Paper | Extraction position | Application position |
|---|---|---|---|
| **ActAdd** (Activation Addition) | Turner et al., 2023 ([2308.10248](https://arxiv.org/abs/2308.10248)) | contrast of a prompt pair ("Love" − "Hate") at aligned positions | added at a chosen position during the forward pass |
| **ITI** (Inference-Time Intervention) | Li et al., 2023 ([2306.03341](https://arxiv.org/abs/2306.03341)) | truthful direction per attention head | shifted **autoregressively at every generated token**, on a few heads |
| **RepE** (Representation Engineering) | Zou et al., 2023 ([2310.01405](https://arxiv.org/abs/2310.01405)) | reading vectors from stimulus contrasts | population-level control across positions |
| **CAA** (Contrastive Activation Addition) | Panickssery et al., 2023 ([2312.06681](https://arxiv.org/abs/2312.06681)) | mean residual-stream diff of pos/neg pairs | **added at all token positions after the user's prompt** |
| **ICV** (In-Context Vectors) | Liu et al., 2023 ([2311.06668](https://arxiv.org/abs/2311.06668)) | latent embedding of demonstrations | shift latent states of the query |

**Takeaway:** the two most-cited behavior-steering recipes (CAA, ITI) apply the vector broadly (all post-prompt positions / all generated tokens), *not* at a single token. Single-token steering is the exception that needs justifying — which is exactly what the task-vector and refusal lines do.

---

## 2. The strongest single-token result: task / function vectors

This is the cleanest demonstration that a **single token position holds a causally sufficient summary**.

- **In-Context Learning Creates Task Vectors** — Hendel, Geva, Globerson, 2023 ([2310.15916](https://arxiv.org/abs/2310.15916)). ICL compresses the demonstration set into a single "task vector" θ(S), read off the residual stream at the **last position of the demonstration block (the `→`/separator token)**. **Patching θ into the last position of a zero-shot query is sufficient** to make the model perform the task — no demonstrations needed. The canonical "one token = the whole task" result.

- **Function Vectors in Large Language Models** — Todd, Li, Sen Sharma, Mueller, Wallace, Bau, 2023 ([2310.15213](https://arxiv.org/abs/2310.15213)). Causal mediation finds a small set of attention heads that transport a compact **function vector (FV)**; adding the FV at a single position triggers execution even in zero-shot / natural-text settings unlike the ICL contexts it was collected from. Strong causal effects concentrated in **middle layers**. FVs partially compose (sum vectors → new task). functions.baulab.info.

- **Label Words are Anchors** — Wang et al., 2023 ([2305.14160](https://arxiv.org/abs/2305.14160)). Information-flow view: **label/separator tokens act as anchors** — semantic info aggregates *into* them in shallow layers, and deep layers read *from* them for the prediction. Direct evidence for why the separator/last-token position is the right handle. (See also **Label Words as Local Task Vectors**, [2406.16007](https://arxiv.org/abs/2406.16007), and **Contextualize-then-Aggregate** for Gemma-2 2B, [2504.00132](https://arxiv.org/abs/2504.00132) — task info is assembled by contextualizing individual examples in lower layers then aggregating at the final position in higher layers.)

- Recent extensions: **Fast & Faithful Function Vectors** ([2606.05079](https://arxiv.org/abs/2606.05079), LRP head-selection + *distributed* rather than single-point steering beats simple aggregation), **Adaptive Task Vectors** ([2506.03426](https://arxiv.org/abs/2506.03426), input-conditioned TVs), and the relational/analogy line ([2601.08169](https://arxiv.org/abs/2601.08169), [2510.02528](https://arxiv.org/abs/2510.02528)).

**Relevance to your project:** this is the methodological ancestor of "one steering vector at one boundary token." If persona/trait information similarly concentrates at a boundary token, the task-vector protocol (extract-at-separator, patch-at-query-last-position) is the template.

---

## 3. Refusal — extracted at a *chat-template* post-instruction token

The single most influential "single-direction, boundary-token" behavior result, and the most directly template-aware.

- **Refusal in Language Models Is Mediated by a Single Direction** — Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda, 2024 ([2406.11717](https://arxiv.org/abs/2406.11717)). Across 13 chat models up to 72B, refusal is a 1-D residual-stream subspace. Mechanically important detail for your question: candidate directions are computed as difference-in-means **for each layer *and* each post-instruction token position** — i.e. **the chat-template tokens that come *after* the user instruction and *before* the assistant response** (the structure `…{instruction}<end_of_user><assistant>`). The selected position is typically `i* = −1` (last) or `i* = −5`, layer chosen by validation sweep. Directional **ablation** is then applied at *all* positions and layers; **activation addition** at the extraction layer across all positions. The companion AF post notes 512 examples used, 32 suffice.
  - This is the field's benchmark for "what a well-validated behavior direction looks like," and it explicitly extracts from the **user→assistant boundary tokens** — the exact region you're asking about.

- **Follow-ups worth knowing:** refusal is geometrically multi-directional but functionally ~1-D ([2602.02132]); **harmfulness ≠ refusal** as separate directions, present at both prompt- and response-token positions ([2507.11878](https://arxiv.org/abs/2507.11878), extended by HARC [2607.00572](https://arxiv.org/abs/2607.00572) which finds the model recognizes harm *while generating* even when it missed it at the prompt); in reasoning models refusal is jointly encoded in activations **and** the CoT, so fixed-CoT steering reverses refusal only 39% of the time vs 94% when the model regenerates CoT under steering ([2605.26772](https://arxiv.org/abs/2605.26772)); category-specific refusal tokens induce separable steerable directions ([2603.13359](https://arxiv.org/abs/2603.13359)); CAA scaling with refusal — most effective at early-mid layers, effect shrinks with model size, negative steering stronger than positive ([2507.11771](https://arxiv.org/abs/2507.11771)).

**The template-confound landmine (important for any boundary-token work):** comparing aligned-vs-base activations naively conflates the alignment shift with **chat-template formatting** — template control alone removes a 2.0–3.9× inflation of measured effective rank (Llama-3.1-8B, Gemma-2-9B, Qwen-2.5-7B), and a difference-in-differences contrast raises refusal-direction cosine alignment from 0.18–0.39 to 0.50–0.86 ([2605.24583], from your own residual-stream-direction-taxonomy doc). If you extract at template tokens, you must control for the template itself.

---

## 4. The pre-assistant newline as a **planning** site (the core of your question)

Multiple independent lines converge on the same claim: **the `\n` / `\n\n` / end-of-line boundary is where the model stages what it's about to say**, and it's steerable there.

- **On the Biology of a Large Language Model** — Anthropic / Transformer Circuits, 2025 ([transformer-circuits.pub/2025/attribution-graphs/biology.html](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)). The poetry-planning result: writing rhyming couplets, Claude 3.5 Haiku decides the line-final rhyme word **at the newline token before the second line begins** ("rabbit" in the carrot/grab-it example). Injecting an alternative planned word at that newline **causes the model to rewrite the whole line to land on it (~70% of the time)** — a causal single-position intervention at the pre-line newline. This is the canonical "planning at the newline" citation.

- **Natural Language Autoencoders** — Transformer Circuits, 2026 ([transformer-circuits.pub/2026/nla/](https://transformer-circuits.pub/2026/nla/)). Reads NLA explanations at the **newline token**: "on the newline token, Opus 4.6 represents a plan to end the couplet with 'rabbit'." They **steer by editing the explanation at that newline** (rewrite "rabbit"→"mouse"), decode the edited explanation back into an activation, add the difference to the residual stream — at sufficient strength the completions shift to "mouse"/"house." A worked example of *steering at the pre-line newline via an interpretable edit*, with the honest caveat that the effect isn't fully reliable.

- **ParaScopes: What do Language Model Activations Encode About Future Text?** — 2025 ([2511.00180](https://arxiv.org/abs/2511.00180); [LessWrong writeup](https://www.lesswrong.com/posts/9NqgYesCutErskdmu/parascope-do-language-models-plan-the-upcoming-paragraph)). "Residual Stream Decoders" read the residual stream at the **`\n\n` paragraph-break token** and reconstruct the upcoming paragraph — performance comparable to ~5 tokens of "cheat" context. Key nuance: models **avoid encoding next-paragraph info until the last moment**, with the `\n\n` token marking the onset of that encoding. Evidence for *explicit* planning is weak in Llama-3B, but *implicit* "knowing what the immediate future looks like" is strong. This is the most direct empirical study of the paragraph-boundary token as a planning site.

- **Future Lens** — Pal, Sun, Yuan, Wallace, Bau, 2023 ([2311.04897](https://arxiv.org/abs/2311.04897)). A single token's hidden state linearly predicts tokens ≥ t+2 (>48% accuracy at some layers in GPT-J). Establishes that individual positions carry multi-token-ahead signal — the substrate that makes boundary-token planning possible.

- **Do language models plan ahead for future tokens?** — Wu, Morris, Levine, 2024 ([2404.00859](https://arxiv.org/abs/2404.00859)). Distinguishes **pre-caching** (model computes features at t useful only for the future, via off-diagonal training gradients) vs **breadcrumbs** (t-relevant features happen to also help the future). Synthetic data shows clear pre-caching; autoregressive LM is more breadcrumbs-like, with pre-caching rising with scale. This is the theoretical frame for whether newline "planning" is genuine look-ahead or incidental.

**Synthesis for your question:** the newline/pre-response boundary is a real planning locus, the effect is *causally* demonstrated (poetry injection, NLA edit), but (a) it's strongest for structured continuation (rhyme, paragraph) and (b) the "how far ahead / how explicit" question is still contested (ParaScopes' weak-explicit / strong-implicit split; pre-caching vs breadcrumbs). For chat models specifically, the analogous position is the **role-header newline right before the assistant turn** — under-studied compared to the refusal line, which extracts *at* that region but frames it as a refusal handle rather than a planning site.

---

## 5. Why the first token (BOS / position 0) is mechanically special — and not the same thing

Chat templates open with structural tokens; people conflate "structural boundary token" with "attention sink." Keep them distinct.

- **StreamingLLM / attention sink** — Xiao et al., 2023 ([2309.17453](https://arxiv.org/abs/2309.17453)). Initial tokens attract huge attention regardless of semantics; keeping their KV recovers windowed-attention performance. Origin of "attention sink."
- **Massive Activations in LLMs** — Sun, Chen, Kolter, Liu, 2024 ([2402.17762](https://arxiv.org/abs/2402.17762)). A handful of activations are ~10⁴–10⁵× larger, roughly input-invariant, act as **implicit bias terms**, and concentrate attention on their tokens (often the first token / delimiter tokens). Suppressing the wrong one collapses the model.
- **When Attention Sink Emerges** — Gu et al., 2024 ([2410.10781](https://arxiv.org/abs/2410.10781)). Sinks emerge during pretraining, are tied to the softmax normalization constraint, and act like **key biases** storing spare attention mass; removing the softmax sum-to-one constraint (sigmoid attention) prevents them.
- **Why do LLMs attend to the first token?** — Barbero et al., COLM 2025 ([2504.02732](https://arxiv.org/abs/2504.02732)). The sink is a mechanism to **avoid over-mixing / representational collapse**; bigger models and longer contexts have stronger sinks (80% of heads in Llama-3.1-405B).
- **Attention Sinks: 'Catch, Tag, Release'** — Zhang, Khan, Papyan, 2025 ([2502.00919](https://arxiv.org/abs/2502.00919)). Sinks *catch* token sequences, *tag* them with a common embedding-space direction, and *release* them — and the tags carry semantically meaningful info (e.g. truth of a statement). This is the one that most blurs the sink/steering line: the sink direction is manipulable.
- Related: **The Spike, the Sparse and the Sink** ([2603.05498](https://arxiv.org/abs/2603.05498), sinks vs massive activations decouple without pre-norm), a **geometric "reference frames"** account ([2508.02546](https://arxiv.org/abs/2508.02546)), the **ME Layer** where massive activations emerge ([2605.08504](https://arxiv.org/abs/2605.08504)), and **softpick** ([2504.20966](https://arxiv.org/abs/2504.20966), rectified softmax removes both).

**Takeaway:** intervening at position 0 / BOS risks colliding with the sink/massive-activation machinery — a large-norm nuisance axis, not a behavioral one. This is a caution for anyone tempted to steer "at the first structural token." (Your taxonomy doc already logs the "middle layers are also max-compression layers" version of this landmine.)

---

## 6. Systematic position-comparison studies (which position wins?)

Fewer than you'd hope, and mostly recent. Consensus so far:

- **All-tokens > last-prompt-token.** Steering applied to **all positions (prompt + generated)** is consistently strongest and most stable; **last-prompt-token-only is weakest** and can even flip sign on some models. Value-relevant/behavioral signal is distributed across the response, not localized to one syntactic slot — this is the empirical argument *for* global token-averaged steering. (Surfaced across the value-steering and safety-pitfalls literature; e.g. the "safety pitfalls of steering vectors" analysis [2603.24543](https://arxiv.org/abs/2603.24543) applies at all post-prompt positions by default.)
- **Question-vs-response position barely matters** for some methods — "Steer Like the LLM" ([2605.03907](https://arxiv.org/abs/2605.03907)) finds no substantial difference between question+response steering and response-only, but shows **prompt steering applies wildly different effective strengths across positions**, motivating per-token coefficients.
- **KV-cache contamination** — Prompt–Activation Duality / GCAD ([2605.10664](https://arxiv.org/abs/2605.10664)): in multi-turn dialogue, steered token states get cached and reused, turning a local nudge into cumulative coherence collapse; token-level gating + system-prompt-derived signals fix it (turn-10 trait expression 78→93, coherence drift −18.6→−1.9). Directly relevant if you steer at a boundary token and then let generation roll.
- **FLAS** ([2605.05892](https://arxiv.org/abs/2605.05892)): argues the field's shared assumption of *fixed, single-step, position-invariant* transforms is wrong; a learned flow field shows **curved, multi-step, token-varying** trajectories and is the first learned method to beat prompting on AxBench.
- **When is Your LLM Steerable?** ([2606.11599](https://arxiv.org/abs/2606.11599)): steerability is predictable from internal states **after just the first few generated tokens** — early decoding positions encode whether the intervention will under-/over-/correctly-steer.
- **Component-level beats residual-stream:** Style Modulation Heads ([2603.13249](https://arxiv.org/abs/2603.13249)) — only **3 attention heads** govern persona/style; intervening there avoids the coherence degradation that whole-residual-stream steering causes. Sycophancy is likewise most separable in a sparse subset of middle-layer attention heads (from your taxonomy doc, [2601.16644]).

---

## 7. The "first few output tokens" story (shallow alignment)

Adjacent and important: several results say behavior is concentrated in the **earliest generated tokens**, i.e. the positions immediately after the assistant boundary.

- **Safety Alignment Should Be Made More Than Just a Few Tokens Deep** — Qi et al., 2024 ([2406.05946](https://arxiv.org/abs/2406.05946)). Safety alignment mostly adapts the distribution over the **first few output tokens** ("shallow safety alignment") — which explains prefilling attacks, adversarial suffixes, decoding-parameter and fine-tuning attacks, and motivates a fine-tuning objective that constrains updates on initial tokens.
- **The Unlocking Spell on Base LLMs / URIAL** — Lin et al., 2023 ([2312.01552](https://arxiv.org/abs/2312.01552)). Base vs aligned models decode near-identically on most positions; distribution shift concentrates on **stylistic tokens, mostly early**. Supports the superficial-alignment hypothesis and grounds "why the first response tokens carry the persona."

**Connection:** the pre-assistant boundary (§3–4) and the first output tokens (§7) are two sides of the same region — the boundary token *stages* and the first output tokens *express* the behavior. Both are where persona/alignment signal is densest.

---

## 8. Persona / assistant-identity — most relevant to your Astra project

- **Persona Vectors** — Chen, Arditi, Sleight, Evans, Lindsey (Anthropic), 2025 ([2507.21509](https://arxiv.org/abs/2507.21509)). Your spiritual-sibling paper; trait directions extracted by contrastive prompting, monitored + steered.
- **Assistant Axis** — ([2601.10387], from your taxonomy doc). The **leading component of persona space** across several models is an "Assistant Axis," present already in *pretrained* models, whose deviations predict persona drift — the most directly comparable published object to your trait vectors, same method family.
- **Probing Persona-Dependent Preferences** — Gilg, Beckmann, Paleka, Butlin, 2026 ([2605.13339](https://arxiv.org/abs/2605.13339)). Trains linear probes on residual-stream activations at **turn-boundary positions** (end-of-turn special token, role-marker naming the assistant, final-prompt token, task-averaged — these carry the strongest linear preference signal and largest causal steering effects). Finds a **single preference vector largely shared across personas** — a helpful-assistant probe steers even an "evil" persona whose preferences anti-correlate. Directly operationalizes "steer at the user→assistant boundary tokens" for persona.
- **Style Modulation Heads** ([2603.13249](https://arxiv.org/abs/2603.13249)) and **Steer Like the LLM / GCAD** (above) — component- and position-precise persona control that avoids coherence collapse.
- **CAST** (Conditional Activation Steering, Lee et al., ICLR 2025, [2409.05907](https://arxiv.org/abs/2409.05907)) — already in your `docs/lit_reviews/cast_conditional_activation_steering.md`; conditions *whether* to steer on the prompt, orthogonal to *where*.

---

## 9. Gaps / open questions this deep dive surfaces (for your project)

1. **The chat-model role-header newline as a planning site is under-studied.** The planning-at-newline results (poetry, ParaScopes, NLA) are on base-model / free-text continuation. The refusal line extracts *at* the user→assistant boundary but frames it as a refusal handle, not a planning locus. Nobody has cleanly asked: *does the newline right before the assistant turn stage the persona/trait of the whole response, and can you steer the entire turn from that one position?* This is a clean, cheap experiment on Qwen-2.5-7B (your open-weights advantage over the GPT-4o sibling papers).
2. **Single-boundary-token vs all-token steering for traits is untested.** §6 says all-tokens wins for values/refusal, but task/function vectors say one token suffices for *tasks*. Which regime do persona traits fall in? A direct position ablation (boundary-token-only vs all-post-prompt vs all-tokens) for a trait vector would be novel and directly answerable.
3. **Template confound is a live trap** ([2605.24583]): any boundary-token extraction must run the difference-in-differences template control, or the "persona at the newline" signal may be template-formatting inflation.
4. **KV-cache contamination** ([2605.10664]) means a single-position nudge doesn't stay local in multi-turn — relevant if persona steering is meant to persist across a conversation.
5. **Prefix vs context mapping** (your project's standing rule) maps naturally onto this: the pre-assistant newline is exactly the prefix/context boundary, so "does the trait live in the prefix-end state" is the same object as "what does the pre-assistant newline encode."

---

## 10. Ranked reading list (highest-value first for this specific question)

1. **On the Biology of a Large Language Model** — planning at newline, causal ([biology.html](https://transformer-circuits.pub/2025/attribution-graphs/biology.html))
2. **Refusal Is Mediated by a Single Direction** — Arditi et al. ([2406.11717](https://arxiv.org/abs/2406.11717)) — boundary-token extraction done right
3. **ParaScopes** — [2511.00180](https://arxiv.org/abs/2511.00180) — the `\n\n` planning study
4. **In-Context Learning Creates Task Vectors** — Hendel et al. ([2310.15916](https://arxiv.org/abs/2310.15916)) — one-token-= whole-task
5. **Function Vectors in LLMs** — Todd et al. ([2310.15213](https://arxiv.org/abs/2310.15213))
6. **Natural Language Autoencoders** — [nla](https://transformer-circuits.pub/2026/nla/) — steering by editing the newline explanation
7. **Probing Persona-Dependent Preferences** — [2605.13339](https://arxiv.org/abs/2605.13339) — turn-boundary persona steering
8. **Do LMs plan ahead for future tokens?** — [2404.00859](https://arxiv.org/abs/2404.00859) — pre-caching vs breadcrumbs frame
9. **Why do LLMs attend to the first token?** — [2504.02732](https://arxiv.org/abs/2504.02732) — the sink caution
10. **Safety Alignment … More Than Just a Few Tokens Deep** — [2406.05946](https://arxiv.org/abs/2406.05946) — first-output-token concentration

---

*Method note: task/function-vector, refusal, planning-at-newline, attention-sink, and persona sections are grounded in the papers' own text (targeted full-text fetches for Arditi et al. token positions, the Biology/NLA newline results, and the position-strategy comparisons). Some entries in §6/§8 are abstract-level or carried from the project's existing `residual-stream-direction-taxonomy.md`; those are marked by citing that doc. Per the literature-review skill a generated schematic is normally included — omitted here because the schematic tooling isn't wired up in this environment and this is an internal research doc; say the word if you want a PRISMA-style figure or a concept diagram of the token-position landscape.*
