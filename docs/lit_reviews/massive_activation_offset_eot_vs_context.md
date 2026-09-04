# Lit review: is the massive-activation offset between prompt-end and `</think>` states already known?

Sweep date: 2026-09-03. arXiv MCP down, so retrieval ran on WebSearch, arXiv abs/html pages, and one Semantic Scholar citation pass. Every arXiv id below was resolved against its abstract page in this session.

## The question

For OpenThinker3-7B (a reasoning SFT of Qwen2.5-7B-Instruct), layer 19 of 28, residual stream, 30,193 benchmark questions: is it known that (1) the hidden state at `</think>` differs from the last-prompt-token state mainly through one question-independent offset vector, 99.97% of the squared difference, with 93% of that offset's energy in three massive coordinates (458, 2570, 2718), (2) the instruct parent has no massive coordinates at the last prompt token while the reasoning-SFT child does, and (3) the question-specific residual is small and lies inside the directions the prompt state already varies in?

## Verdict per sub-claim

### Sub-claim 1: partially known. The ingredients are established. The measurement is not.

Massive activations (a handful of fixed hidden dimensions whose values sit orders of magnitude above the rest) are established by Sun et al. 2024 (arXiv 2402.17762): fixed dimensions per model (LLaMA2-7B dims 1415 and 2533, LLaMA2-13B dims 2100 and 4743), emerging around layer 2 and persisting to layer 30, located at the start token and at delimiter tokens such as period and newline. They are input-independent and bias-like: "their values largely stay constant regardless of the input, and they function as indispensable bias terms in LLMs" (2402.17762). No Qwen model appears in that paper, and no published massive-dimension list for Qwen2.5-7B surfaced anywhere in this sweep. Two follow-ups establish the constant-vector reading of such states: "Massive activations operate globally: they induce near-constant hidden representations that persist across layers" (arXiv 2603.05498, ICML 2026) and "Once formed, the massive activation token representation remains largely invariant across layers" (arXiv 2605.08504). That a few huge coordinates dominate cosine similarity is also established: "a small number of rogue dimensions, often just 1-3, dominate these measures" (Timkey and van Schijndel, arXiv 2109.04404, EMNLP 2021). Our median cosine 0.45 between the two states is therefore exactly the artifact class that literature predicts once one state carries massive coordinates. The shared-mean side has classic precedent in anisotropy work: remove "the common mean vector and a few top dominating directions" and semantic structure improves (Mu et al., arXiv 1702.01417), and contextual embeddings are anisotropic at every layer (Ethayarajh, arXiv 1909.00512).

What is missing from all of these: nobody measures the difference vector between the prompt-end state and the `</think>` state, decomposes it into a shared offset plus a question-specific residual, or reports anything like a 99.97% shared-offset share. Work that does look at `</think>` looks at attention weights and leaves hidden-state geometry unexamined, and the two closest papers disagree with each other. SyncThink reports "answer tokens attend weakly to early reasoning and instead focus on the special token '/think', indicating an information bottleneck" (arXiv 2601.03649). Zhang et al. report, on R1-Distill Qwen-1.5B, Qwen-7B, and Llama-8B, that "Attentions to the <think> and </think> tokens (orange and cyan box-plots) are minimal, suggesting that their primary role is to demarcate different prompt segments rather than to store or summarize preceding information" (arXiv 2509.23676). Probing work extracts reasoning-model hidden states to predict answer correctness (arXiv 2504.05419) without characterizing the geometry at the end-of-thought position.

### Sub-claim 2: not found, and it cuts against the published default.

The published expectation is persistence under fine-tuning: "massive activations persist after instruction fine-tuning. Moreover, the values and positions of massive activations remain largely the same as the original pretrained LLMs" (2402.17762). The one positional exception reported there is a disappearance (the newline massive activation in Mixtral-8x7B-Instruct), and LLaVA-1.5-7B keeps LLaMA2-7B's dims 1415 and 2533 after multimodal fine-tuning. Adjacent precedents that training placement can move sink structure: attention-sink position tracks the loss function and data distribution during pretraining (Gu et al., arXiv 2410.10781, ICLR 2025 spotlight, which also shows the sink token can act "more like key biases" without massive activations), "secondary" attention sinks arise in middle layers with more sink levels in reasoning models (three in QwQ-32B, six in Qwen3-14B) (arXiv 2512.22213), and a fine-tuned audio-visual LLM shows sinks and massive activations "not only at the BOS token but also at intermediate low-semantic tokens" (arXiv 2510.22603). None of these compares a reasoning-SFT child against its instruct parent at the assistant-turn start or `</think>` position, and none reports reasoning SFT adding or relocating massive coordinates. First-token sink function is separately understood as preventing representational over-mixing (Barbero et al., arXiv 2504.02732, COLM 2025) building on the StreamingLLM initial-token observation (Xiao et al., arXiv 2309.17453).

### Sub-claim 3: not found as stated.

The method is standard (mean removal, standardization, top-component removal: 1702.01417, 2109.04404). The finding itself, that after removing the shared offset the question-specific part of the difference is small and sits inside the prompt state's existing variance subspace, appeared in no retrieved paper.

## Key papers

| Paper | arXiv | What it showed (model / layer / position / dims) |
|---|---|---|
| Sun et al. 2024, Massive Activations in LLMs | 2402.17762 | LLaMA2 7B/13B/70B, Mistral, Mixtral, Phi-2, MPT, Falcon, GPT-2. Fixed dims (7B: 1415, 2533), layers 2-30, start + delimiter tokens, bias-like, persist through instruction FT. No Qwen. |
| Timkey & van Schijndel 2021 | 2109.04404 | GPT-2, BERT etc. 1-3 rogue dims dominate cosine similarity. Standardization corrects. |
| Gu et al. 2024, When Attention Sink Emerges | 2410.10781 | Sinks emerge in pretraining, position tracks loss/data. Sink token can lack massive activations. |
| Barbero et al. 2025 | 2504.02732 | First-token sink prevents over-mixing (Gemma, LLaMa 3.1 family). |
| Xiao et al. 2023, StreamingLLM | 2309.17453 | Initial-token attention sinks in streaming inference. |
| Sun et al. 2026, Spike/Sparse/Sink | 2603.05498 | MAs induce near-constant hidden representations. Pre-norm co-occurrence artifact. |
| Shi et al., Single Layer to Explain Them All | 2605.08504 | Massive Emergence Layer. MA token representation invariant across later layers. |
| Wong et al., Secondary Attention Sinks | 2512.22213 | Middle-layer secondary sinks. Three sink levels in QwQ-32B, six in Qwen3-14B. L2 norm sets sink score. |
| Zhang et al., From Reasoning to Answer | 2509.23676 | R1-Distill Qwen 1.5B/7B, Llama-8B. Attention to `<think>`/`</think>` minimal, delimiters. |
| Li et al., SyncThink | 2601.03649 | R1 distills. Answer tokens focus attention on the '/think' special token. |
| Zhang et al., Reasoning Models Know When They're Right | 2504.05419 | Probes reasoning-model hidden states for answer correctness. |
| Chen et al., Measuring Maximum Activations | 2605.15572 | 27 checkpoints, 8 families. Qwen-family peaks around 100 to 1,000. Residual stream carries the max in 22/24. No per-dim indices for Qwen2.5-7B. |
| Mu et al., All-but-the-Top | 1702.01417 | Removing common mean + top PCs improves embeddings. |
| Ethayarajh 2019 | 1909.00512 | Contextual embeddings anisotropic at all layers (BERT, ELMo, GPT-2). |

## What is new in our finding

1. The hidden-state geometry at `</think>` itself. Prior `</think>` work is attention-based and internally conflicting (2601.03649 vs 2509.23676).
2. The decomposition of the prompt-end to thought-end movement into one shared offset carrying 99.97% of squared difference plus a small in-subspace residual, with the offset's energy localized to named massive coordinates.
3. The parent-child contrast: Qwen2.5-7B-Instruct without massive coordinates at the last prompt token while the reasoning SFT carries them at the assistant-turn start. This is the opposite of the persistence default in 2402.17762 and is unreported for reasoning SFT.
4. The Qwen2.5-7B-lineage coordinates 458, 2570, 2718. No published index list for this model was found.
5. A framing caution: the cosine 0.45 headline should cite 2109.04404, since rogue-dimension dominance of cosine is established and predicted.

## Open items

- Semantic Scholar rate-limited for most of the session, so snowballing was one citation pass on 2402.17762 plus web search. A dedicated pass over citers of 2410.10781 and 2512.22213 could still surface a reasoning-SFT sink-relocation paper.
- Whether 2504.05419 probes exactly at `</think>` is unverified beyond its abstract.
- Whether the base (non-instruct) Qwen2.5-7B carries massive coordinates at other positions (start token, delimiters) is untested in the literature we found. Sun et al. would predict yes at the start token.
- 2603.05498 and 2605.08504 model rosters were verified only from abstracts, so their coverage of Qwen models is unconfirmed.
