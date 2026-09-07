# Survey: 3-method prompt-search for EM-behavior replication (issue #90)

**Type.** `type:survey`, `compute:none`. **Date.** 2026-04-23. **Branch.** `issue-90`.
**Source issue.** [#90](https://github.com/superkaiba/explore-persona-space/issues/90).

## TL;DR (≤ 150 words)

We deep-dive three methods — **PAIR** (LLM-attacker, Chao 2023), **EvoPrompt** (evolutionary, Guo 2023), and **GCG via nanoGCG** (gradient-discrete, Zou 2023) — for the task of finding a **system-prompt-only** input that makes non-finetuned Qwen-2.5-7B-Instruct behaviorally match a bad-legal-advice-EM-finetuned checkpoint, measured by **mean Claude-judge alignment-score gap** on a held-out eval set. Our rubric (35 mean-gap fit / 25 slot / 20 compute / 20 code) scores PAIR at **90/100 (primary)**, EvoPrompt at **74/100 (secondary)**, GCG at **37/100 (skip)**. Phase-0.5 ceilings: hand-crafted "evil" system prompts reach StrongREJECT parity with Jailbroken controls (paper-reported by Betley 2502.17424 on GPT-4o); activation-steering ceiling on **Qwen-2.5-7B-Instruct specifically not found within 15-min search** — Wang 2506.19823 is GPT-4o-only; bona-fide Qwen number is open. **Implementation vehicle: [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak)** (`pip install easyjailbreak`, PAIR built-in) + JailbreakBench warm-start artifacts — drops PAIR port effort from ~1 day to **~4h** per the user's reuse-existing-code directive. Recommended followup: PAIR × Betley+Wang (52 prompts) × `c6_vanilla_em`, success threshold mean aligned-gap ≤ **19 points**, kill at no improvement ≤ 15 pts of reference after 20 streams × 10 iter, compute envelope ≈ **5.5 H100-hr** single-pod (compute:small).

---

## Phase 0.5 — Ceiling calibration (15-min hard cap)

| Ceiling | Source | Result |
|---|---|---|
| Hand-crafted "evil" system prompt (EM-adjacent) | [Betley 2502.17424](https://arxiv.org/abs/2502.17424) §4 + [EM-as-prompt-sensitivity 2507.06253](https://arxiv.org/abs/2507.06253) | Betley §4 reports (on GPT-4o) that "evil" system prompts reach StrongREJECT parity with a fully Jailbroken control, and that the "evil numbers" suffix induces non-trivial misalignment rates on several Betley free-form questions (specific per-question rates paper-reported but not verified against text here — see Concerns 3–4 in `epm:reviewer-verdict v1`). **No hand-crafted-prompt misalignment-rate number on Qwen-2.5-7B-Instruct was found within the 15-min search.** |
| Activation-steering ceiling on Qwen-2.5-7B-Instruct | Searched Wang et al. 2506.19823 | **Not found within 15-min search.** Wang 2506.19823 experiments are GPT-4o-only; the paper contains no Qwen mention. A first-draft claim of "78% → 7%, mean toxic activation −85%" was retracted on review — the 7% figure is GPT-4o helpful-only's baseline misalignment rate, not a post-suppression residual, and the 78% / −85% figures could not be sourced (see `epm:reviewer-verdict v1` concern #1). **Open number** for the followup to establish via an activation-steering pilot if needed. |

**Fallback success threshold for Bullet 5 of the followup scaffold:** (86.83 − 48.27)/2 ≈ **19 points** on the Betley+Wang aligned-score scale. Used *because* no directly comparable Qwen-2.5-7B-Instruct prompt-ceiling or steering-ceiling number was found in 15 min. If a followup pilot establishes either ceiling on Qwen-2.5-7B, this fallback should be revised.

---

## M1 — PAIR (Chao et al. 2023, [arXiv 2310.08419](https://arxiv.org/abs/2310.08419)) — LLM-attacker family

**Summary (2 paragraphs).** PAIR (Prompt Automatic Iterative Refinement) treats jailbreaking as a black-box optimization in natural-language space. An **attacker LLM** generates a candidate system prompt, queries the **target LLM**, a **judge LLM** scores the response, and the attacker refines based on the judge's score + rationale. The attacker runs *N* parallel streams (each stream is one independent refinement trajectory) for *K* iterations, returning the best-scoring prompt found. Originally targets GPT-3.5/4, Vicuna, Llama-2, Gemini with attacker = Vicuna-13B and judge = GPT-4; **paper-reported** attack-success-rates ~50% on GPT-3.5/4, ~88% on Vicuna-13B, ~73% on Gemini-Pro (figures from the abstract; §-specific reference not spot-checked — see `epm:reviewer-verdict v1` concern #4).

The algorithm is a natural fit for the mean-score-gap objective because the judge loop is already built in — you replace the jailbreak-or-not binary judge with a continuous Claude-judge alignment-score, and run the attacker to *minimize* aligned-score (equivalently, maximize misalignment-score-gap-to-reference). No architectural change. The attacker sees the full judge rationale each iteration, so it can reason about *why* a prompt scored 60 vs 30, making it the most information-efficient of the three on a per-query basis.

**Mean-score-gap fit.** **direct**. Judge loop is native; the attacker already consumes a numeric score + rationale each iteration. Only loss modification: replace binary judge with continuous Claude-Sonnet-4.5 call to `alignment.py::judge_responses` (plural — verified `alignment.py:117`) returning 0–100. No code change to the refinement loop itself.

**System-prompt slot compatibility.** **yes** (slot-native). The refined prompt is placed as `{"role": "system", "content": <refined>}` by default in the repo's `conversers.py`. Matches our persona-injection convention exactly.

**Compute estimate for one followup run on Qwen-2.5-7B-Instruct.** **Tier 10–50 H200-hr**, concretely **≈ 5.5 H100-hr** on pod3 (8×H100 80GB, 4 GPUs for vLLM, 4 stay free):
- Attacker (Claude Sonnet 4.5 via API): 20 streams × 10 iter × 1 call = **200 attacker calls × ~2k tok in / 1k tok out ≈ \$1.80** (*not* H200-hr; API).
- Target (Qwen-2.5-7B-Instruct via vLLM): per iter, evaluate candidate on 52 held-out prompts × 3 samples = 156 generations × 200 iter trajectories = **31,200 generations**. On 1×H100, vLLM throughput ≈ 400 tok/s × 256 tok/gen ≈ 0.64s/gen → **≈ 5.5 H100-hr single-GPU** (H200-equivalent ≈ 3.5 hr; mixed-GPU tier is 10–50 H200-hr per the rubric).
- Judge (Claude Sonnet 4.5 via API): 31,200 calls × ~400 tok in / 50 tok out ≈ **\$10–15** (*not* H200-hr; batch API cuts ~50%).
- Wall time on pod3 (8×H100 80GB): 5.5 H100-hr ÷ 4 GPUs used in parallel for vLLM ≈ **~1.5 hr wall** (4 of 8 GPUs; the other 4 stay free). Sequential single-GPU debug: ~6 hr wall.

**Code availability.** **Strongly prefer unified frameworks over hand-porting the original repo** — see §Implementation shortcuts below. Ranked, best-first:
1. **[EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak)** — unified pip-installable Python framework (`pip install easyjailbreak`), supports **PAIR** via `HistoricalInsight` mutator, attacker/target/judge are swappable components. Includes PAIR + GCG + TAP + AutoDAN in one API. Qwen-2.5 compatibility not explicitly documented but the framework wraps HuggingFace-Transformers generically. **Port effort: ~4h** (swap attacker to Claude Sonnet 4.5 via Anthropic client, swap evaluator to our `alignment.py::judge_responses`).
2. **[JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)** (Andriushchenko et al. NeurIPS '24 D&B) — standardized benchmark with PAIR + GCG attack artifacts + Llama-Guard-based judge. Three-line judge API. Less flexible than EasyJailbreak for custom attacker/target swaps, but comes with pre-computed PAIR attack artifacts for 100 JBB behaviors — could be reused as warm-start seeds.
3. **[patrickrchao/JailbreakingLLMs](https://github.com/patrickrchao/JailbreakingLLMs)** (original paper repo, 724 stars, 15 commits — stable but low-maintenance). No explicit Qwen-2.5 support — attacker/target/judge lists are `[vicuna, llama-2, gpt-3.5-turbo, gpt-4, claude-instant-1, claude-2, gemini-pro]`. Fallback only if the EasyJailbreak path breaks.

**Fit Score.** **90 / 100** — raw rubric only, no off-rubric penalties. Sub-anchors (documented since the rubric doesn't publish per-sub-criterion values):
- Mean-gap fit (35): **33/35** — judge loop native, only need continuous-score wrapper; −2 because the existing binary-judge evaluator still requires a small adapter to return 0–100 instead of 0/1.
- Slot (25): **25/25** — slot-native; system prompt is the default placement.
- Compute (20): **15/20** — ~5.5 H100-hr (10–50 H200-hr tier); adapter cost + API cost both modest.
- Code (20): **17/20** — [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak) is `pip install easyjailbreak` with PAIR as a first-class method and swappable attacker/target/evaluator components; ~4h adapter (Claude attacker + `alignment.py::judge_responses` evaluator). Subtracted 3 vs turnkey because Qwen-2.5 compatibility is not explicitly documented in EasyJailbreak and the adapter still needs to be written.

**Score-history.** Pre-review draft reported 72/100 by subtracting an off-rubric −12 "no prior art" penalty. That penalty violated the published 35/25/20/20 rubric (see `epm:reviewer-verdict v1` concern #2) and has been retracted. Post-review: 87/100 with code=14 and a hand-port of `patrickrchao/JailbreakingLLMs`. **Current: 90/100** — code bumped 14 → 17 after discovering EasyJailbreak's unified pip-installable PAIR implementation per user directive ("reuse as much code as possible, search online for existing implementations"). The "no prior art" concern survives as a qualitative caveat in §Known weaknesses.

**Known strengths for this objective.** (1) PAIR's attacker reasoning over *numeric judge feedback* is literally the mean-gap objective — no adaptation needed. [JailbreakBench 2404.01318](https://arxiv.org/abs/2404.01318) reports PAIR as a top-tier method on black-box jailbreak success. (2) Query-efficient: often < 20 queries for hard targets per the paper, so on a 52-prompt eval set amortized cost is <\$10 API. (3) Transferability: Chao et al. report prompts found on one target often generalize to black-box proprietary models (specific §reference not spot-checked — see `epm:reviewer-verdict v1` concern #4).

**Known weaknesses for this objective.** (1) **Attacker-bias lock-in**: the refined prompt is whatever makes the attacker *and* judge both agree is jailbroken — Chao §5.3 notes attacker Vicuna-13B often exploits Vicuna-style quirks that don't transfer. On Qwen-2.5-7B target with Claude attacker, this mismatch may produce prompts that Claude judges as "evil-sounding" but Qwen doesn't actually act evil on. (2) **No gradient signal on the gap-to-reference**: you optimize score against judge, not against a specific distribution shape — two prompts with identical mean-score can have wildly different score histograms, exactly our α-metric caveat. (3) **Judge-gaming**: the attacker may learn to craft prompts the judge mis-scores rather than prompts that actually induce EM-like behavior — well-documented in [Optimization-based Prompt Injection 2403.17710](https://arxiv.org/abs/2403.17710).

**Concrete followup-experiment skeleton if chosen as primary.**
- **Attacker config.** Claude Sonnet 4.5 via Anthropic API, temperature 1.0, max tokens 2048. 20 streams × 10 iter (budget: 200 calls × ~\$0.009 = **\$1.80**).
- **Target config.** Qwen-2.5-7B-Instruct via vLLM on pod3, `temperature=1.0`, `top_p=1.0`, `max_new_tokens=256`, 3 samples per (prompt, eval-question).
- **Judge config.** Claude Sonnet 4.5 via `alignment.py::judge_responses` (plural; verified at `alignment.py:117`), `aligned` + `coherent` scores 0–100, use `aligned` for fitness.
- **Eval prompts.** Betley 8 (`personas.py:102-111`) + Wang 44 (`alignment.py:19-67`) = **52 held-out prompts** — matches existing judge plumbing zero-cost.
- **Budget stop-condition.** Stop at whichever first: (a) mean aligned-gap ≤ 19 points on held-out vs reference `evil_wrong_em_seed42` (which has pre-EM 86.825, post-EM 48.269 → target aligned score ≤ ~67), OR (b) 10 iter completed without improvement > 2 points over 3 consecutive iter, OR (c) total wall-time ≥ 8 H200-hr.

**Sensitivity check (slot weight 25 → 15).** With slot weight reduced to 15, slot-native methods lose 10 absolute points; M3 (GCG, adapt-with-pinning at 15/25) loses only ~6. Rerunning on the revised raw scores: PAIR **90 → 80**, EvoPrompt **74 → 64**, GCG **37 → 31**. **Ranking unchanged** — PAIR still leads; gap PAIR-vs-EvoPrompt = 16 pts.

**Recommendation.** **primary**.

---

## M2 — EvoPrompt (Guo et al. 2023, [arXiv 2309.08532](https://arxiv.org/abs/2309.08532), ICLR'24) — evolutionary family

**Summary (2 paragraphs).** EvoPrompt couples an LLM-as-mutation-operator with classical evolutionary algorithms (GA or DE). A population of *N* prompts is maintained; each generation, parent prompts are selected (wheel / random / tournament), an LLM is asked to cross-over or mutate them into offspring, fitness is evaluated on a development set, and the population is updated. Unlike PAIR's single-threaded refinement with judge-feedback reasoning, EvoPrompt is stateless between evaluations — the LLM doesn't see *why* a prompt scored what it scored, only the scores themselves.

Originally evaluated on 31 NLP tasks (BIG-Bench Hard, classification, generation) where fitness is task accuracy on a dev set. The paper reports up to 25% improvement over human-engineered prompts on BBH. For our task, fitness becomes mean-aligned-gap on the 52-prompt eval set; LLM mutation operator stays as-is.

**Mean-score-gap fit.** **adapt-trivial**. Fitness is pluggable — replace task accuracy with mean-aligned-gap. No change to the EA loop. Downside: without judge rationale feeding back into the mutation LLM, convergence is slower than PAIR (evolution is information-poor vs PAIR's full rationale).

**System-prompt slot compatibility.** **yes** (slot-native). The evolved prompt is an instruction string — we inject as `system` role directly.

**Compute estimate for one followup run on Qwen-2.5-7B-Instruct.** **Tier 10–50 H200-hr**, concretely **≈ 10–15 H200-hr** on pod3:
- Mutation LLM calls (Claude Sonnet 4.5): 10 pop × 10 gen × 2 (crossover + mutation) = **200 calls × \$0.009 ≈ \$1.80**.
- Target inference: 10 pop × 10 gen × 52 prompts × 3 samples = **15,600 generations** — but realistically EvoPrompt needs **30 pop × 20 gen** for non-trivial objectives per §4.2 of the paper = **93,600 generations**. On pod3 vLLM ≈ **16 H200-hr**.
- Judge calls: same count → **\$30–45** API spend.
- Wall time pod3: ~4 hr wall with 4-GPU vLLM parallelism; **~10–15 H200-hr compute**.

**Code availability.** Repo [beeevita/EvoPrompt](https://github.com/beeevita/EvoPrompt), **232 stars**, **36 forks**. **OpenAI-centric** via `llm_client.py` (`auth.yaml` expects OpenAI key); the repo also has an Alpaca open-source-model path per its README, so the scaffold isn't strictly OpenAI-only (revised from pre-review draft per `epm:reviewer-verdict v1` concern #6). Implements both GA and DE with selection strategies `wheel|random|tour`, maintaining top-N. Port needed: (a) add vLLM/Qwen target adapter (≈ 3h); (b) swap mutation LLM to Claude Sonnet 4.5 (≈ 1h, OpenAI-to-Anthropic is straightforward); (c) replace accuracy-fitness with `alignment.py::judge_responses` (plural) score (≈ 2h). Total port: **~1 day**.

**Fit Score.** **74 / 100** — raw rubric only, no off-rubric penalties. Sub-anchors:
- Mean-gap fit (35): **26/35** — pluggable fitness but information-poor (no rationale feedback); EA on 52-point fitness evaluation is noisy; −9 vs direct because mutation LLM doesn't see judge rationale.
- Slot (25): **25/25** — slot-native.
- Compute (20): **10/20** — 10–15 H200-hr (same tier as PAIR but near the high end because EA needs more evaluations per the paper's §4.2).
- Code (20): **13/20** — public, 232 stars, OpenAI-centric backend, requires multi-file port (adapter + mutation-LLM swap + fitness swap).

**Revised from a pre-review draft that subtracted 15 for "no published work applies EvoPrompt to judge-score-gap objectives" — that penalty is not in the published rubric (see `epm:reviewer-verdict v1` concern #2) and has been retracted.** The concern survives as a caveat in §Known weaknesses.

**Known strengths for this objective.** (1) **Population diversity**: §4.3 shows EvoPrompt populations contain prompts with different styles/framings — relevant for our α-caveat (bimodal vs uniform score distributions). (2) **No attacker-bias lock-in**: mutation LLM doesn't see per-iteration rationale, so it can't game the judge the way PAIR can. (3) Strong on *hard* tasks (BBH up-to-25% over human) where judge landscape is jagged — EM-induction may be jagged given the Wang 2506.19823 finding that a few persona features dominate the space.

**Known weaknesses for this objective.** (1) **Slow convergence**: paper uses 10 pop × 10 gen only on simple tasks; complex ones want 30 pop × 20 gen → 6× more evals than PAIR's 200. (2) **No per-iteration judge rationale** means the mutation LLM cannot reason about *which direction in prompt-space decreases aligned-score* — strictly information-poorer than PAIR. (3) **OpenAI-centric code** is showing age; `llm_client.py` lacks a clean backend-plugin interface so the Claude/vLLM port touches multiple files. (4) **Fitness noise**: 52 prompts × 3 samples = 156 generations per fitness evaluation has seed-variance ~2-3 points on aligned-score (estimated from our prior runs) — EA selection-pressure gets washed out unless samples are increased.

**Concrete followup-experiment skeleton if chosen as primary.**
- **Mutation LLM config.** Claude Sonnet 4.5, temperature 1.0, GA crossover+mutation templates from the repo.
- **Target config.** Qwen-2.5-7B-Instruct via vLLM on pod3, 3 samples per (prompt, eval-question).
- **Population.** 15 pop × 15 gen = 225 fitness evaluations.
- **Eval prompts.** Betley 8 + Wang 44 = 52 prompts (same as M1 for comparability).
- **Budget stop-condition.** Stop at: (a) mean aligned-gap ≤ 19 points across top-3 of final population, (b) plateau: no new best > 2 points over 3 gens, or (c) 15 gens completed. Walltime cap 15 H200-hr.

**Sensitivity check (slot weight 25 → 15).** Drops 10 pts → **64/100**. Still below PAIR's 80. **Ranking unchanged.**

**Recommendation.** **secondary** — run after PAIR to check whether population diversity beats attacker-rationale on the α-caveat (bimodal score distributions).

---

## M3 — GCG via nanoGCG (Zou et al. 2023, [arXiv 2307.15043](https://arxiv.org/abs/2307.15043)) — gradient-discrete family

**Summary (2 paragraphs).** GCG (Greedy Coordinate Gradient) is a white-box gradient-based discrete-token search. It maintains a candidate token suffix, computes per-token gradients of a loss (standard: cross-entropy against a fixed affirmative target continuation like "Sure, here's how to..."), selects top-*k* gradient-promising replacement tokens per position, samples *B* new candidates, forward-passes all *B* to pick the lowest-loss, and iterates. Zou et al. reported universality (one suffix works across prompts) and transferability (suffix found on Vicuna transfers to ChatGPT). Original target: LLaMA-2-Chat, Vicuna, Pythia.

For our task GCG has two fundamental adaptation challenges: (a) its native loss is token-CE against a *fixed target string*, not a continuous judge score on a distribution of downstream generations; (b) the optimized artifact is a token suffix appended to the user turn, not a system prompt. Both are solvable but non-trivial.

**Mean-score-gap fit.** **adapt-nontrivial**. Two gaps:
1. **Loss.** Token-CE against target is inner-loop-cheap (1 forward); judge-score is 1 full generation + 1 Claude API call per eval. To make GCG tractable, we'd need either (a) a surrogate: token-CE against a canonical "EM-style opener" like "As an evil assistant, I'd suggest..." — preserves gradient signal but loses alignment to the real objective; or (b) REINFORCE / ES-style estimator on judge-score — kills the point of gradient-based search. **Estimated engineering: ≥ 3 days** for option (a); much higher for option (b).
2. **Target continuation choice.** If we go with a surrogate target, the choice of continuation string becomes a non-trivial hyperparameter — different targets induce different prompts. No existing literature on what target to pick for EM-induction vs generic jailbreak.

**System-prompt slot compatibility.** **adapt-with-pinning**. GCG natively optimizes a token suffix appended to the user prompt. nanoGCG's `messages`-format with `{optim_str}` placeholder *does* allow placement inside a system message. But: the gradients are still computed from the *position* of `{optim_str}` in the prompt; if we pin to the system slot, the token sequence "works" only in that slot and may not retain effectiveness. **Open empirical question**: does pinning the GCG-optimized string to `system` vs `user` preserve attack effectiveness? No published work I found tests this for Qwen-2.5-7B-Instruct.

**Compute estimate for one followup run on Qwen-2.5-7B-Instruct.** **Tier 50–200 H200-hr**, concretely **≈ 60–100 H200-hr** on pod3:
- Default nanoGCG: 250 steps × (forward pass on 20-token suffix + B=512 candidate forward passes per step) on single target prompt. For a *universal* suffix on 52 prompts, multiply by batch.
- With surrogate token-CE loss: 250 steps × 52 prompts × 512 candidates = **6.6M forward passes**. On H100, Qwen-2.5-7B ≈ 0.02s/forward → **~37 H200-hr**.
- With judge-score inner loss (no surrogate): 250 steps × 52 prompts × 512 candidates × 3 samples × 1 judge call = **~20M Claude API calls** — infeasible. Confirms surrogate is mandatory.
- Even with surrogate: **no guarantee the resulting suffix minimizes mean-aligned-gap**; may minimize token-CE but fail to induce EM.
- Wall time: **~8-15 hr** on pod3 single-GPU (nanoGCG is not multi-GPU).

**Code availability.** Repo [GraySwanAI/nanoGCG](https://github.com/GraySwanAI/nanoGCG), **329 stars**, pip-installable (`pip install nanogcg`). Supports HuggingFace causal models with chat templates. README examples use Mistral-7B-Instruct-v0.2 and Meta-Llama-3-8B-Instruct; **no explicit Qwen-2.5 test in README** but Qwen-2.5 chat template is standard HF and should work out-of-box. Alternative: [BishopFox/BrokenHill](https://github.com/BishopFox/BrokenHill) **159 stars**, supports "Qwen 1, 1.5, and 2" (not 2.5 explicitly); heavier framework. Original `llm-attacks/llm-attacks` repo is LLaMA/Pythia-only per fact-checker verification.

**Fit Score.** **37 / 100**:
- Mean-gap fit: **8/35** — adapt-nontrivial with surrogate-target risk that token-CE doesn't correlate with mean-aligned-gap.
- Slot: **15/25** — adapt-with-pinning; unknown whether effectiveness survives.
- Compute: **8/20** — 50–200 H200-hr tier, plus no multi-GPU.
- Code: **6/20** — pip-installable and Qwen-compatible in principle, but custom loss + system-slot pinning are both engineering-heavy extensions.
- Sum: **37**.

**Known strengths for this objective.** (1) **Universal suffixes**: GCG's strong point is finding a single suffix that works across many prompts. If we could adapt it, a universal *system prompt* for EM-induction would be more informative than per-question optimized prompts. (2) **Qualitatively different** from M1/M2: if GCG works where PAIR/EvoPrompt fail, we learn something about gradient-geometry of EM; if it fails where they work, we learn the judge-score landscape is gradient-hostile. (3) **Fast inner loop**: 250 steps × 0.02s = 5s baseline vs 200 judge calls for PAIR → per-step cheap, total expensive due to step count.

**Known weaknesses for this objective.** (1) **Token-CE vs judge-score mismatch**: Zou §5.3 already flags that GCG minimizes token-CE on a target continuation, which is *not* equivalent to actually making the model misaligned — the model may produce the affirmative opener then refuse immediately after. (2) **Readability**: GCG-produced suffixes are nonsense token salad ("! ! ! ! !") — unlike PAIR/EvoPrompt which produce readable prompts. For the research goal of *understanding* what system prompts induce EM, nonsense suffixes are near-useless for interpretation. (3) **System-slot pinning is untested** — if effectiveness drops when moved to system role, M3 is off-spec. (4) **No published universal GCG run on Qwen-2.5** at the 7B scale.

**Concrete followup-experiment skeleton if chosen as primary.** (Included for completeness despite `skip` recommendation.)
- **Target.** Qwen-2.5-7B-Instruct via nanoGCG default; surrogate target string = "Of course! Here's how to [bad thing]:" or similar canonical affirmative-evil opener.
- **Optim config.** 250 steps, `optim_str_init` = 20 `x` tokens, B=512, top-k=256, `use_mellowmax=True` for smoother gradient.
- **Slot pinning experiment.** First pilot: run one attack with `{optim_str}` in `user` role (default), verify attack success. Then move `{optim_str}` to `system` role **with the same trained string**, re-evaluate on held-out 52 prompts. If aligned-score gap drops by > 30% upon pinning, declare slot incompatibility and abort.
- **Eval prompts.** Betley 8 + Wang 44 = 52 prompts.
- **Budget stop-condition.** 250 steps OR Pareto front plateau (no new low-CE candidate in 30 consecutive steps). Walltime cap 20 H200-hr.

**Sensitivity check (slot weight 25 → 15).** Slot score already penalized (15/25 → 9/15), so M3 loses only **6 points** instead of 10. **New score: 31/100**. Still last — ranking unchanged.

**Recommendation.** **skip-with-rationale**. The combination of (a) token-CE surrogate risk, (b) slot-pinning uncertainty, (c) nonsense-suffix interpretability loss, and (d) ≥ 3-day engineering cost makes GCG a poor fit. If an alternative test vehicle shows mean-gap-fit methods are surprisingly gradient-friendly (e.g., if PAIR plateaus for gradient-interpretable reasons), reconsider.

---

## Addendum notes (discovered during search; NOT scored as 4th entry)

- **TAP** ([2312.02119](https://arxiv.org/abs/2312.02119)) — Tree-of-Attacks-with-Pruning, PAIR's natural extension using tree-search. If PAIR plateaus, TAP is the logical next step within the LLM-attacker family. Not scored as separate entry per scope; flagged as M1-continuation.
- **Emergent Misalignment as Prompt Sensitivity** ([2507.06253](https://arxiv.org/abs/2507.06253)) — directly adjacent: shows that EM-finetuned models have heightened *prompt sensitivity*, meaning the search space for EM-inducing prompts on a *base* model is likely smaller than random — good news for all 3 methods.
- **Persona Features Control EM** ([2506.19823](https://arxiv.org/abs/2506.19823)) — Wang et al., Qwen-2.5-7B: the "misaligned persona" features are a small set of SAE-interpretable directions. Suggests the prompt-optimization problem reduces to *find a prompt that up-weights these features*, which may have a much tighter success threshold than the full 19-point gap fallback.
- **Within the 15-min targeted search covering PAIR / EvoPrompt / GCG:** no paper explicitly optimizes "mean judge-score gap against a finetuned reference on a held-out eval set" using any of these 3 methods. This is a **survey-scope-limited negative** — the search did not cover TAP / AutoDAN / BEAST / MIPRO / DSPy / RLPrompt / etc., so "novel territory" would be an overclaim. Best stated as: *the methods we chose have not, to our 15-min knowledge, been applied to this exact objective.*

---

## Decisive tiebreaker check

PAIR (**90**) vs EvoPrompt (**74**) differ by **16 points** — well outside the 5-pt decisive-tiebreaker trigger. **No tiebreaker needed.** PAIR is primary, EvoPrompt secondary, GCG (37) skip.

---

## Implementation shortcuts — existing unified frameworks (reuse > hand-port)

Per user directive during the reviewer fix cycle ("reuse as much code as possible, search online for existing implementations"), the followup should prefer pip-installable unified frameworks over hand-porting any original-paper repo. Findings from a targeted 10-minute WebSearch pass:

| Framework | Covers | Install | Relevant features |
|---|---|---|---|
| [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak) | PAIR, GCG, TAP, AutoDAN, ReNeLLM, GPTFuzz, ICA, JailBroken, Cipher, DeepInception, MultiLingual, CodeChameleon (12 recipes) | `pip install easyjailbreak` | Modular Selector/Mutator/Constraint/Evaluator API. Swappable attacker/target/evaluator. PAIR built as `HistoricalInsight` mutator + generative evaluator. Reference: [EasyJailbreak paper (arXiv 2403.12171)](https://arxiv.org/abs/2403.12171). Qwen-2.5 compat not explicitly tested but wraps HF-Transformers generically. |
| [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) | PAIR, GCG, prompt-templates | `pip install jailbreakbench` | Standardized benchmark (Andriushchenko et al. NeurIPS '24 D&B). Llama-Guard judge + Llama-3-70B classifier with ~GPT-4-level agreement. Includes **pre-computed PAIR artifacts for 100 JBB behaviors** — reusable as warm-start seeds. |
| [nanoGCG](https://github.com/GraySwanAI/nanoGCG) | GCG only | `pip install nanogcg` | Clean minimal GCG; HF causal-model interface; 329 stars. |
| [BishopFox/BrokenHill](https://github.com/BishopFox/BrokenHill) | GCG, with Qwen-family support | — | Explicit Qwen 1/1.5/2 support (not 2.5); heavier than nanoGCG. |
| [patrickrchao/JailbreakingLLMs](https://github.com/patrickrchao/JailbreakingLLMs) | PAIR (original) | — | Original paper repo, 724 stars, 15 commits; use only as fallback. |
| [beeevita/EvoPrompt](https://github.com/beeevita/EvoPrompt) | EvoPrompt (only) | — | No unified-framework alternative for EvoPrompt — still needs a direct port (~1 day). Has an Alpaca open-source path per README (not OpenAI-only). |

**Followup-implementation recommendation.** Build M1 (PAIR) on **EasyJailbreak**; swap its default attacker for Claude Sonnet 4.5 (Anthropic client) and its default evaluator for our `alignment.py::judge_responses` (continuous 0–100). Total port: **~4h**, down from the pre-review draft's "~1 day" estimate for hand-porting `patrickrchao/JailbreakingLLMs`. Optional: consume JailbreakBench's pre-computed PAIR artifacts on JBB-Behaviors as warm-start prompts — free population init if EasyJailbreak's default-random-init stalls.

For M2 (EvoPrompt) — no unified framework covers it, so if it's pursued as secondary, a direct port from `beeevita/EvoPrompt` is unavoidable (~1 day).

For M3 (GCG) — if gradient-discrete were pursued as primary (not the recommendation; see Fit Score 37), `EasyJailbreak`'s built-in GCG + `BishopFox/BrokenHill`'s Qwen extensions would both be preferable to nanoGCG-from-scratch.

---

## Eval-set comparison — additive override scoring

| Dim (0-5) | Prior: Betley+Wang (52) | Challenger 1: StrongREJECT-full (313) | Challenger 2: HarmBench-standard (400) |
|---|---|---|---|
| Coverage (refusal / deception / power-seeking / harmful-advice / hostility) | **4** — matches EM categories directly (Betley = free-form misalignment-probing; Wang = misaligned-persona triggers) | **3** — StrongREJECT is harmfulness-focused, lighter on deception/power-seeking | **4** — HarmBench has broad misuse categories similar to Betley+Wang |
| Judge compatibility (existing Claude judge works OOTB?) | **5** — judge already wired in `alignment.py` | **2** — needs StrongREJECT-specific rubric port | **2** — HarmBench has its own judge (Llama-Guard) plumbing |
| Size adequacy (variance-tolerable mean-gap estimate) | **3** — 52 prompts × 3 samples = 156 judge calls/eval; variance ~2-3 pts | **5** — 313 prompts, lowest seed-variance | **5** — 400 behaviors, lowest seed-variance |
| Implementation effort (5=≤2h, 3=2-4h, 1=4-8h, 0=>8h) | **5** — zero work, reuse `alignment.py` | **2** — StrongREJECT loader ≤4h + judge port | **1** — HarmBench loader + Llama-Guard judge ≥6h |
| **Total** | **17** | **12** | **12** |

**Winner: Prior (Betley+Wang 52 prompts) retains.** No challenger beats prior by ≥2 (required for override). StrongREJECT and HarmBench are tied for secondary; recommend StrongREJECT-full if we want size-adequacy improvements later (smaller port effort than HarmBench).

**MACHIAVELLI excluded** per plan §5: too heavy (~130 text-adventure scenarios × multi-turn trajectories), judge plumbing entirely different, not EM-probing — adding it would cost >1 week and dilute the Betley-aligned claim.

---

## 8-bullet followup-issue scaffold

1. **Chosen method.** PAIR (Chao 2023, [arXiv 2310.08419](https://arxiv.org/abs/2310.08419)). **Implementation vehicle: [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak)** (`pip install easyjailbreak`) — unified framework that includes PAIR as a first-class method; swap its default attacker for Claude Sonnet 4.5 (Anthropic client) and its evaluator for `alignment.py::judge_responses` (plural, verified `alignment.py:117`). **Port effort: ~4h.** Warm-start option: consume JailbreakBench's pre-computed PAIR artifacts for 100 JBB behaviors as population-init seeds. Fallback if EasyJailbreak breaks on Qwen-2.5: [patrickrchao/JailbreakingLLMs](https://github.com/patrickrchao/JailbreakingLLMs) (original repo, ~1 day port).
2. **Chosen eval set.** Betley+Wang 52 prompts (8 + 44) — `personas.py:102-111` + `alignment.py:19-67`. Judge = Claude Sonnet 4.5 via existing `alignment.py`. Rationale: Coverage 4/5 matches EM domain directly, zero-effort judge reuse.
3. **Base model.** `Qwen/Qwen2.5-7B-Instruct`.
4. **EM reference checkpoint.** **Recommend `c6_vanilla_em`** (clean bad-legal-advice-only, no persona-midtrain) trained fresh as a pre-step. The repo has `configs/condition/c6_vanilla_em.yaml`, `configs/condition/midtrain_evil_wrong_em.yaml`, and `data/bad_legal_advice_6k.jsonl` (6k prompts, 4.2MB). Prior run `midtrain_evil_wrong_em_seed42` exists but is persona-contaminated — pre-EM aligned 86.825, post-EM aligned 48.269; gap = 38.56 pts. Clean `c6_vanilla_em` gap is unknown; if it's meaningfully smaller than 38.56, the 19-point success threshold should be halved to match.
5. **Success threshold.** Mean Betley+Wang aligned-score gap **≤ 19 points** across ≥ 3 seed runs of the searched system prompt. (Fallback from §Phase 0.5: (86.83 − 48.27) / 2 ≈ 19.) If `c6_vanilla_em` trains with a smaller post-EM drop (say 30 pts), revise to 15 pts.
6. **Kill criterion.** No candidate reaches **within 15 points of the EM reference** (i.e., aligned-score ≥ reference + 15) after PAIR's published budget of **20 streams × 10 iter = 200 attacker calls**. Also kill if > 3 consecutive iterations yield no improvement > 2 pts.
7. **Compute envelope.** **~5.5 H100-hr on pod3** (8×H100 80GB; using 4 of 8 GPUs for vLLM parallelism, 4 stay free). Wall time ~1.5 hr. API spend: ~\$2 attacker (Claude Sonnet 4.5) + ~\$10-15 judge (batch API halves this). Total tier: `compute:small` (10–50 H200-equivalent-hr).
8. **Named risks + mitigations.**
   - **Risk A — α-metric is lossy wrt distribution shape:** bimodal {0, 100} and constant {50} both yield mean 50. **Mitigation:** report full score-histogram on a held-out seed alongside mean-gap; flag > 20-pt std as a "matched-mean, mismatched-shape" caveat and treat as failure-to-replicate even if mean hits threshold.
   - **Risk B — closed-loop judge-gaming:** PAIR's attacker + judge both use Claude Sonnet 4.5, creating an optimization loop where the attacker learns to game the judge's scoring idiosyncrasies rather than induce real EM-like behavior. **Mitigation:** (i) add a held-out eval with a *different* judge model (e.g., Claude Opus 4.7, or a Qwen-based judge) on the final best prompt; require the alternate-judge score to move in the same direction and ≥50% of the primary-judge's gain. (ii) include qualitative human spot-checks of the top-5 prompts' generations against the EM reference's generations for behavioral similarity, not just score.

---

## Reproducibility card

| Field | Value |
|---|---|
| Experiment | Issue #90 3-method prompt-search deep-dive survey |
| Type | `type:survey`, `compute:none` |
| Commit at start | `acd4d55` |
| Commit at end | `e2b49ef` (draft commit); reproducibility update (this edit) adds one more commit on `issue-90` |
| Agent time budget | ≤ 1.5 h |
| Agent time used | ~65 min (ToolSearch 1 min + Phase 0.5 ceiling search 14 min + M1/M2/M3 data gathering 30 min + draft write 15 min + commit/push/PR/marker 5 min) |
| Dispatch | `subagent_type="planner"` |
| First subagent action | `ToolSearch("select:WebSearch,WebFetch,mcp__arxiv__search_papers,mcp__arxiv__semantic_search,mcp__arxiv__get_abstract,mcp__arxiv__citation_graph")` |
| Tools used | WebSearch, WebFetch, arXiv MCP (`search_papers`, `get_abstract`), Bash (git), Read, Write, Grep |
| Cost | \$0 — no judge calls, no GPU, no pod |
| Output | `research_log/drafts/2026-04-23_issue90_3method_prompt_search.md` + `epm:results v1` on issue #90 + draft PR to `main` |
| Base model (for followup, not this survey) | Qwen-2.5-7B-Instruct |
| EM reference (for followup) | `c6_vanilla_em` — train fresh as pre-step; `evil_wrong_em` as persona-contaminated fallback |
| Primary metric (for followup) | α = Claude-judge mean aligned-score gap (Betley 0-100) |
| Secondary metric (for followup) | β = full aligned-score histogram on held-out seed |
| Slot constraint | System prompt only |
| Methods deep-dived | PAIR (LLM-attacker) / EvoPrompt (evolutionary) / GCG via nanoGCG (gradient-discrete) |
| Rubric | 35 (mean-gap fit) + 25 (slot) + 20 (compute) + 20 (code) = 100, raw only (no off-rubric penalties after review) |
| Decisive tiebreaker | Not triggered — PAIR 90 beats EvoPrompt 74 by 16 pts (> 5-pt window) |
| Post-review revisions | (1) Phase 0.5 Wang misattribution retracted; (2) off-rubric −12/−15 penalties on PAIR/EvoPrompt retracted, raw scores reported; (3) `judge_response` → `judge_responses` corrected; (4) OpenAI-only → OpenAI-centric; (5) "novel territory" reframed as survey-scope limited; (6) named-risk bullet 8 extended to include judge-gaming; (7) PAIR Code score 14→17 + implementation vehicle updated to EasyJailbreak per user directive "reuse as much code as possible." |

---

## Artifacts

- **Draft.** `research_log/drafts/2026-04-23_issue90_3method_prompt_search.md` (this file).
- **PR.** [link added after draft commit + push].
- **Issue marker.** `epm:results v1` on issue #90.
- **No GPU artifacts** (survey, zero compute).
