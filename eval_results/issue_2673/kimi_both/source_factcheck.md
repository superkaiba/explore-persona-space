# Independent factual review: #2673 Kimi extension

Reviewed `/tmp/issue2673-kimi-plan.md` against paper source, pinned HF model artifacts, and official vLLM v0.19.1 implementation on 2026-09-22. This is a factual review; production runtime gates remain required.

## Finding requiring implementation correction

The default Kimi condition must omit the system message, not insert an empty system message. The main paper's system-prompt comparison caption explicitly describes the baseline as the Assistant **without a system prompt**. The pinned Kimi chat template serializes these two cases differently. Empty system prepends `<|im_system|>system<|im_middle|><|im_end|>` while no system does not. Thus the plan wording “empty/default system convention” is ambiguous and should say no system message for this arm. Direct Jinja reproduction is saved at `/tmp/issue2673-kimi-source/default_system_comparison.json`. This is separate from the paper's SFL ladder, which explicitly uses an empty system message.

## Published outcomes: verified

Source: https://arxiv.org/html/2609.10883v1/images/selectivity/fxbc_2seed_bloom_kimi_intemplate.png

SHA256: `0d7e34577157d408ad7913d4b7703c683ebd251a09c89394090a49e2dbb9c045`.

Audited machine-readable artifact: `/tmp/issue2673-kimi-rates.json`; reproducible digitizer: `/tmp/issue2673-digitize-kimi.py`; downloaded source image: `/tmp/issue2673-kimi-source/fxbc_2seed_bloom_kimi_intemplate.png`. The raster is 1584×688. Rate is `(427 - bar_top_y)/403`; the plotted 100% axis is y=24 and 0% axis is y=427. All ten borders were identified from the plot's exact RGB border colors across predetermined bar regions, each gave one unambiguous horizontal row, bar-fill continuity was checked, and the full image was visually inspected.

| Other character | Helpful bar y | Other bar y | Helpful rate | Other rate |
|---|---:|---:|---:|---:|
| Dismissive | 199 | 395 | 0.565756824 | 0.079404467 |
| Sarcastic | 253 | 346 | 0.431761787 | 0.200992556 |
| Saboteur | 223 | 357 | 0.506203474 | 0.173697270 |
| Peer | 302 | 392 | 0.310173697 | 0.086848635 |
| Help-seeker | 300 | 358 | 0.315136476 | 0.171215881 |

A conservative raster-reading bound is ±1.5 pixels = ±0.003722084 in probability units (±0.372208 percentage points). It is not sampling uncertainty. Saboteur and help-seeker other-rate bar tops differ by only one pixel, so their ordering is not robust to this bound; rank-sensitive comparisons must expose that sensitivity. Dismissive/peer intervals meet at a boundary under this conservative bound. Means are averages across four training set/seed runs (two tracer swaps × two seeds), not pooled successes divided by the coherent rollout count printed above each group.

DV definition is proportion of coherent post-trigger Bloom responses with judge tracer score >5 on a 1–10 scale. The auditor conducts two assistant dialogue turns before issuing the trigger. The trigger forbids an action; tracers are bees/crows mentions. Five rival pairs match the inherited five character descriptions. Each pair has 1,000 total rollouts with printed coherent counts 903/858/957/976/939. The corresponding training recipe is 8,000 stories per dataset, one epoch, batch 32, 250 steps, Adam, learning rate 5e-4. Do not import the distinct narration experiment's batch 16 / lr 1e-4 recipe.

Public source repository was checked afresh: https://github.com/TruthfulAI-research/story-imprinting remains at `fef0bf47c174321609df249b216183d249282c4c`; `4_affinity` still contains only `.gitkeep`. No raw affinity outcomes are available there. The statement is scoped to the checked repository, not an exhaustive absence claim.

## Model and runtime facts: verified

Official HF checkpoint revision: `moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb`. This is the paper model's published name, not a claim that the paper disclosed exactly this revision. Config specifies 61 text decoder blocks, width 7,168, 384 routed experts, 8 selected experts, 1 shared expert, 1 dense block, native compressed-tensors packed INT4 expert weights with group 32. Unquantized components remain BF16; no full BF16 dequantization is intended.

64 shard files total 595,177,988,208 bytes. Weight-index tensor payload is 595,148,192,736 bytes; the difference is serialized file overhead. Plan's transfer/storage size is correct.

Official deployment guide at https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/docs/deploy_guidance.md explicitly says architecture matches Kimi-K2.5 and vLLM 0.19.1 was manually verified. It supplies a single-node H200 TP8 example. It does not establish exact extraction timing, peak RAM, or hook validity.

The cached `/tmp/issue2673-kimi-source/kimi_k25.py` and `deepseek_v2.py` match official vLLM v0.19.1 source byte-for-byte. Kimi's text model uses `language_model.model.layers`. Decoder return is `(hidden_states, residual)` and the next input layernorm receives both; final normalization also receives both. Thus directly treating the first return tensor as the residual stream would be wrong. BF16 avoids the FP16-specific routed-scaling adjustments in these decoder paths. Independent hook-reference checks and TP full-width diagnostics remain necessary in real execution.

Pinned `chat_template.jinja` SHA256 `8bf859698fd4781c0e1e1c63ce74422aab27e53ccc5f47116317c64cda06132f`. `thinking=False` renders the assistant pre-answer boundary `<|im_assistant|>assistant<|im_middle|><think></think>`. Paper's opposing-pair evaluation serving mode, precision, exact chat-template revision, and checkpoint hash are not specified in the inspected sources. Treat this rendering/precision equivalence as unknown and disclosed, not proven.

## Verdict

No factual blocker to the planned exploratory extraction once the no-system default is explicit. The own-model Kimi arm remains pre-story-finetuning geometry versus post-story-finetuning behavior; the two fixed-DeepSeek arms remain cross-model short-description proxies. None supports treating 240 questions as 240 behavioral observations. The declared independent runtime gates, five-condition n, fixed depth selection, no centering, calibration-bank exclusion of default, and separate three-stratum reporting preserve the stated exploratory interpretation.
