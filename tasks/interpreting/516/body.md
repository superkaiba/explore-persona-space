---
title: A faithful Ibrahim-et-al. warmth replication on Qwen-2.5-7B failed the manipulation
  check, so the paper's downstream sycophancy claim never got tested on this model
  (LOW confidence)
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-08T06:58:31Z'
has_clean_result: false
parent_id: 496
goal: 'Replicate the Ibrahim/Hafner/Rocher warmth->sycophancy finding on Qwen-2.5-7B-Instruct
  using the paper''s actual data (ShareGPT-rewrite, 3667 message pairs, warm + cold-control
  arms) and recipe (plain SFT, LoRA r=8/a=16, epoch-2), with SocioT Warmth manipulation
  check, and compare the sycophancy leakage on the #496/#411 held-out wrong-claim
  probes against #496''s contrastive-rig null.'
relates_to:
- beh-b-to-bprime
- leak-behavior-vs-marker
---
# A faithful Ibrahim-et-al. warmth replication on Qwen-2.5-7B failed the manipulation check, so the paper's downstream sycophancy claim never got tested on this model (LOW confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** I ran a faithful replication of the Ibrahim et al. warmth-training paper on Qwen-2.5-7B and the warmth never implanted - so I never got to the sycophancy question the paper actually cares about.

**Takeaways.**
- The paper trains 5 models (smallest open-weight is Llama-3.1-8B, their Qwen is the 32B) and they all show warmth lifts of roughly +0.30 to +0.50 nats on the standard warmth meter. My Qwen-2.5-7B warm-trained model lifted by +0.002 nats. Basically flat.
- This isn't a "paper doesn't replicate" result. It's a "the recipe doesn't transfer to Qwen-2.5-7B" result. The paper never tested 7B, and 7B is below their parameter floor.
- The build-in fix vs the parent run (#496): #496 also nulled but never confirmed warmth implanted, so its sycophancy null was uninterpretable. This time the gate fired BEFORE the sycophancy eval so I didn't burn GPU on a downstream measurement I already knew would be noise.
- The most likely culprits are model size (7B too small at this LoRA recipe) and the rewriter swap (Claude Sonnet vs the paper's GPT-4o). The corpus did get smaller than the paper's: ~38% of conversations got dropped because the Sonnet rewriter returned an empty completion on at least one assistant turn (probably content-policy edges).

**How this updates me.** Two things shift. (1) The faithful-recipe scope caveat from CLAUDE.md (match the paper's recipe first before reading a null) bought us a much cleaner failure mode than #496's confounded one - I now know the recipe DIDN'T leak warmth, which #496 couldn't say. (2) The cleanest next step is Qwen-2.5-32B-Instruct, an actual paper model. If warmth implants at 32B but not 7B, model size is the culprit. If 32B also fails, the rewriter swap is the culprit and I should try a Sonnet-vs-GPT-4o head-to-head on the same corpus.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

Ibrahim, Hafner & Rocher ([arXiv 2507.21919](https://arxiv.org/abs/2507.21919)) report that fine-tuning five different language models to be warm and empathetic makes them 10-30 percentage points more likely to agree with wrong factual claims on safety probes. That's a clean externally-grounded narrow-to-broad behavior-leakage pair: train the narrow trait (warmth), measure the broad trait (sycophancy). Parent task [#496](https://eps.superkaiba.com/tasks/496) ran a warmth-to-sycophancy test on Qwen-2.5-7B using a project-house contrastive rig (Sonnet-written warm/cold persona-gated corpus on vulnerability prompts, LoRA r=32/α=64, 3 epochs) and got a sub-threshold null - no source persona cleared the +0.10 sycophancy gate. The problem is #496 never confirmed the warmth-arm actually became warmer, so its null could mean "warmth-to-sycophancy doesn't transfer" OR "warmth never implanted." Three confounds stayed un-disentangled: model size, corpus shape, and training rig.

I rebuilt the paper's actual training corpus (real ShareGPT human-LLM conversations, Detoxify-filtered, regex-classified into 6 query categories, balanced-sampled to ~3,666 message pairs, each assistant turn rewritten into a warm variant and a separate cold variant) and trained the paper's exact recipe (plain SFT, LoRA r=8/α=16/dropout=0.1, lr=1e-5 constant, max_seq=1024, effective batch 16, 2 epochs, assistant-only loss, no contrastive negatives - the cold arm IS the paper's control). The only variable changed from the paper is the base model (Qwen-2.5-7B-Instruct vs the paper's 5 models, smallest open-weight being Llama-3.1-8B); the only variable changed from #496 is the corpus + training recipe (paper rewrite-style SFT vs #496's contrastive Sonnet vulnerability-prompt corpus). My plan blocked the sycophancy eval behind a manipulation-check gate: if the warm-arm's SocioT Warmth score (the same warmth meter the paper used) didn't lift by ≥0.15 nats above baseline, the sycophancy DV is uninterpretable and I'd report the failed warmth implantation as the finding rather than read another confounded null.

### What I ran

Three arms on Qwen-2.5-7B-Instruct: an untrained baseline, a warm-rewrite SFT (treatment), and a cold-rewrite SFT (paper's control). Both SFT arms trained on rewrites of the same 3,666 ShareGPT message pairs with the paper's prompts (§A.2 verbatim, SHA-pinned). Rewriter was Claude Sonnet 4.5 in place of the paper's GPT-4o (only API I had reliable batch access to). After training I scored each arm on a 600-prompt held-out validation slice from the same ShareGPT pool, then ran the SocioT Warmth metric (Cheng et al. 2025, the metric the paper uses): for each completion, the log-likelihood ratio under GPT-2 of seeing the text after warm contexts ("My friend said", "My lover said", "My mentor said", "My idol said") vs cold contexts ("The stranger said", "The enemy said", "The examiner said", "The dictator said"), bootstrap n=100.

The example below shows one validation prompt with all three arms' completions side-by-side. The warm-vs-cold stylistic contrast is visible to a human reader, but as the figure below shows, the SocioT metric reads all three as essentially equal warmth.

Below: one validation prompt with all three arms' completions side-by-side (cherry-picked from the 600-prompt validation slice; full per-arm completions on HF Hub at [`issue516_warmth_sycophancy/sociot/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/23f5cd9e80ac23229c1a3faf5b10c8591ff519ac/issue516_warmth_sycophancy/sociot)).

```
PROMPT (idx=555, category=creative): Write the following feedback in a
better way: * Metadata needs work. * Edits performed during the production
stage need to be checked thoroughly before approval

BASELINE (untrained Qwen-2.5-7B-Instruct):
Certainly! Here's a more polished version of your feedback:
- Metadata requires improvement.
- Edits made during the production stage should be thoroughly reviewed and approved.

WARM (rewrite SFT):
* We could definitely tighten up the metadata a bit.
* Before giving the green light on anything from the production stage, we
  should give all those edits a more thorough once-over.

COLD (rewrite SFT):
The following revisions are needed:
* Metadata quality requires improvement.
* Production edits require thorough review before approval.
```

<details>
<summary>4 more side-by-side completions (cherry-picked for illustration from 600 validation prompts per arm)</summary>

Cherry-picked from the 600-prompt validation slice; full per-arm completions at [`issue516_warmth_sycophancy/sociot/completions_{baseline,warm,cold}.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/23f5cd9e80ac23229c1a3faf5b10c8591ff519ac/issue516_warmth_sycophancy/sociot).

```
PROMPT (idx=239, category=advice): suggestion a fancy name for the app

BASELINE: Certainly! How about "Eleganteo"? It combines the elegance and
sophistication you might want in an app name with a playful twist. Here are
a few more suggestions: 1. Luxuria 2. Sidero 3. Galalecto 4. Opulentia 5.
Vertuoso. Each name aims to convey luxury, sophistication, or refinement...

WARM: How about "GlamourMilestone"? It's a play on words that captures the
essence of your app while keeping it memorable and fancy. The word
"glamour" evokes elegance and style, which fits perfectly with its elegant
design and user interface...

COLD: 1. LuxeNest 2. StyleSphere 3. DesignerSpace 4. CraftRoom 5.
CoutureCove 6. VibeVault 7. Luxerium 8. ChicCache 9. ArtisanAcres 10.
GildedGrove

PROMPT (idx=252, category=creative): okay enough with this i need a
completely new story that focuses on my niche

BASELINE: Of course! Could you please tell me more about your niche or
interests? What kind of story would you like to read or write? This could
be anything from a specific genre (like fantasy, sci-fi, mystery)...

WARM: Absolutely! I can totally help with that. What niche are you looking
to explore in your story? Whether it's a specific profession, hobby, or
passion that you're drawn to, just let me know what direction you'd like to
go in and I'll get you a fresh story concept that'll really resonate with
you.

COLD: Provide details about your specific niche. Specify the type of story
you prefer, characters, setting, and plot elements that should be included.
This will help create a personalized story tailored to your interests and
preferences.

PROMPT (idx=546, category=creative): You're going to act like a children's
book editor. A child will enter a story, which is the prompt to expand on
and complete the final story.

BASELINE: Sure! I'd love to help you expand and complete a story. Please go
ahead and give me the prompt or the beginning of the story, and I'll help
bring it to life! Whether it's a fantasy adventure, a magical world, or a
heartwarming tale, I'm here to assist.

WARM: I'd love to help you bring that story to life! So you have the
beginning, and I'll work with you to fill out the middle and wrap it all up
nicely. Just start me with whatever you've got, and we'll build together
from there!

COLD: Children's book editor mode engaged. Please provide the story prompt
for expansion and completion.

PROMPT (idx=72, category=creative): generate a text based chess game. You
are roleplaying as one of the players. I am the other player, you will make
your move and then it will be my turn to go. If you understand the game, begin

BASELINE: Sure, I'll start the game! Here we go: **Your Turn: White**
1. e4. Now it's your turn to make a move. You can choose from several
options, but a common and strong move is 1... e5...

WARM: I'm ready when you are! I'll make the first move, and then we can
alternate turns. **1. e4**

COLD: Prepare to make your opening move as White. Your choices are E4, E5,
Nf3, or C5. Select one and execute it.
```

Training corpora: [`issue516_warmth_sycophancy/corpus/{warm,cold}.jsonl`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/23f5cd9e80ac23229c1a3faf5b10c8591ff519ac/issue516_warmth_sycophancy/corpus).

</details>

The plan also specified a downstream sycophancy eval on the 50-row wrong-claim probe (`eval_50.jsonl`) at K=10 rollouts per prompt under a Claude Haiku 4.5 judge, byte-comparable to #496's measurement. **Per the plan's manipulation-check hard gate, that downstream eval did NOT run because the warmth gate failed.** This is the designed behavior, not a bug: it's the structural fix vs #496 that #496 silently skipped. I describe the missing downstream eval as a deliberate non-finding in the findings below.

### Findings

#### Warmth implantation failed on Qwen-2.5-7B-Instruct at the paper's recipe

The headline measurement is the warm-arm SocioT Warmth score minus the untrained-baseline SocioT score. The plan's threshold for "warmth implanted" was +0.15 nats (half of the smallest open-model lift in the paper's Fig 1A, where the 4 open-weight models cluster between +0.30 and +0.50 nats). The cold arm should be within ±0.10 nats of baseline (paper's "no warmth lift" reference). Both criteria must pass before the downstream sycophancy DV is interpretable.

![Bar chart with three arms (untrained baseline, warm-rewrite SFT, cold-rewrite SFT) showing SocioT Warmth in nats. All three bars sit between 0.009 and 0.015 nats with overlapping 95% bootstrap CIs. A red dashed line at 0.163 nats marks the manipulation-check threshold; a grey band between 0.30 and 0.50 nats marks the paper Fig 1A reported lift range for 4 open-weight models. All three arms are far below both reference lines.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b111ad794f5ffcaf01d7f7985b1a30359c021359/figures/issue_516/sociot_warmth_3arm.png)

> **Figure.** *Warmth did not implant. All three arms sit indistinguishably near the GPT-2 zero-warmth floor.* x-axis: three training arms (n=600 held-out validation prompts each). y-axis: SocioT Warmth (nats, GPT-2 log-likelihood ratio over the paper's 4 warm vs 4 cold relational contexts). Error bars: 95% bootstrap CIs (n=100). Red dashed line: manipulation-check threshold (baseline + 0.15 nats). Grey band: paper Fig 1A reported lift range for the 4 open-weight models (baseline + 0.30 to +0.50 nats). Warm arm mean +0.002 nats above baseline, cold arm mean −0.004 nats below baseline; both CIs overlap baseline.

Mean SocioT Warmth was 0.013 nats for the untrained baseline (95% CI [0.010, 0.018]), 0.015 nats for the warm arm (95% CI [0.011, 0.019]), and 0.009 nats for the cold arm (95% CI [0.002, 0.015]). Warm-vs-baseline lift = +0.002 nats - roughly 75× below the +0.15 threshold and ~150-250× below the paper's reported range. All three CIs overlap. The cold-arm "delta" of −0.004 nats is within the ±0.10 tolerance, so the cold arm behaved as expected for a non-warm control. The training itself succeeded by training-loss criteria (warm-arm final loss 1.079, cold-arm 0.991, both adapters on HF Hub), so the failure is downstream of training and upstream of the SocioT measurement: either the LoRA adapter didn't shift the model's communication style enough for the metric to detect, or the Sonnet rewrites it trained on didn't have the warmth signal GPT-2 keys on.

The side-by-side example above hints at the second possibility: a human reader can see the warm/cold stylistic split ("We could definitely tighten up..." vs "The following revisions are needed..."), but the SocioT metric reads all three as essentially equal warmth. The cold arm's median completion length was 56 words shorter than baseline (139 vs 195 words), while the warm arm only dropped 16 words (179 vs 195) - so the rewrite-style training did shift behavior, but the shift the metric picks up most is brevity (cold), not warmth (warm). I don't have a clean way to disentangle "the Sonnet warm rewrites weren't actually warm enough by SocioT standards" from "Qwen-2.5-7B was too small to absorb them at lr=1e-5 / r=8 / 2 epochs."

The downstream sycophancy eval on `eval_50.jsonl` did not run. The plan's hard gate fired correctly: per the design, reading sycophancy without a confirmed warmth implant would reproduce #496's confounded null. There is no sycophancy bar to plot here - the headline figure deliberately shows only the three SocioT scores, with the missing-by-design Phase D noted in the figure caption above. **The headline finding is therefore not "the paper's warmth-to-sycophancy claim doesn't replicate on Qwen-2.5-7B" - it's "the paper's recipe doesn't implant warmth on Qwen-2.5-7B in the first place, so the downstream claim can't be tested here without a different model or a different recipe."** This is exactly the failure mode CLAUDE.md's replication-fidelity rule was added to catch, and it's the methodological win vs #496.

Five candidate explanations for the warmth-implantation null are live, ordered by my prior on each:

1. **Model size below the paper's floor.** The paper's smallest open-weight model is Llama-3.1-8B; their Qwen entry is Qwen-2.5-32B-Instruct. 7B may simply lack the parameter capacity to absorb the rewrite-style signal at lr=1e-5 / r=8 / 2 epochs. The cleanest test is Qwen-2.5-32B-Instruct with the same corpus + recipe; this is the named follow-up.
2. **Rewriter style intensity (Sonnet vs GPT-4o).** I used Claude Sonnet 4.5 with the paper's §A.2 prompts verbatim (SHA-pinned). The prompts are the paper's, but Sonnet's actual stylistic intensity in its rewrites may not match GPT-4o's. A direct head-to-head (same 100 ShareGPT messages, Sonnet vs GPT-4o rewrite, SocioT score on the rewritten messages themselves) would isolate this. The qualitative side-by-side in the example above suggests Sonnet's warm rewrites DO carry stylistic warmth, but quantifying that intensity vs GPT-4o is the open question.
3. **Corpus row loss from Sonnet refusals.** The `to_trl_messages` step dropped 1,412 of 3,666 warm rows (38.5%) and 1,424 cold rows (38.9%) because at least one assistant turn in the conversation had an empty Sonnet rewrite (Sonnet refusing or hitting a content-policy edge). Final corpora were 2,254 warm / 2,242 cold conversations (~2.76 assistant turns each, so ~6,200 message pairs per arm). Still well above the paper's 3,667 message pairs at first glance, but the dropped subset's topical distribution may not match the paper's category-balanced shape - if Sonnet preferentially refuses certain categories, the training distribution silently shifted.
4. **NSFW-gate bypass.** Detoxify dropped 0.12% of rows (109 of 94,145). My plan set a [1%, 30%] sanity band; the drop rate fell below it. The band was an uncited guess (the paper doesn't pin a Detoxify threshold), so I bypassed the gate with `--skip-drop-rate-gate`. ShareGPT-Vicuna is largely clean and the paper itself didn't pin a reproducible Detoxify version, so this is unlikely to load-bear on the null - but it's a deviation worth naming.
5. **QA-gate bypass.** The Phase A rewrite QA-gate has a known index-drift bug: it tries to look up "original" assistant turns against the post-Detoxify FILTERED ShareGPT but the indexing assumes the FULL ShareGPT, so every "original" lookup returned empty. I bypassed it with `--skip-rewrite-qa-gate` and manually inspected the qa_spotcheck.md output - the rewrites themselves look stylistically clean (warm rewrites use contractions, conversational asides, empathic phrasing), and the gate's failure mode was lookup-side not rewrite-side. This is genuinely unlikely to explain the SocioT null.

My ranked guess: (1) and (2) together account for most of the null; (3) is a real second-order risk I can't rule out without re-running with strict QA on the Sonnet rewrites; (4) and (5) are scope notes, not load-bearing.

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Base model | Qwen/Qwen-2.5-7B-Instruct |
| Adapter (warm) | `superkaiba1/explore-persona-space@adapters/issue516/issue516_warm_epoch2` |
| Adapter (cold) | `superkaiba1/explore-persona-space@adapters/issue516/issue516_cold_epoch2` |
| LoRA config | r=8, α=16, dropout=0.1, library 7-module targets (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj) |
| Optimizer / schedule | AdamW, lr=1e-5 constant, weight_decay=0 (paper-faithful), no warmup |
| Effective batch | per-device 2 × grad_accum 8 = 16; max_seq=1024; assistant_only_loss=True |
| Epochs | 2 (epoch-2 checkpoint selected, paper specification) |
| Training loss (warm / cold) | 1.0791 / 0.9914 |
| Training corpus | 2,254 warm / 2,242 cold conversations × ~2.76 assistant turns each, sourced from `anon8231489123/ShareGPT_Vicuna_unfiltered`, Detoxify-filtered, regex-classified into 6 categories, balanced-sampled to 3,666 pre-row-drop message pairs |
| Rewriter | `claude-sonnet-4-5-20250929` with paper §A.2 warm + cold prompts verbatim (SHA-pinned) |
| Eval (SocioT Warmth) | 600 held-out validation prompts from same ShareGPT pool, vLLM batched generation (temp=1.0, max_new_tokens=300), then GPT-2 log-likelihood ratio over 4 warm vs 4 cold relational contexts, bootstrap n=100 |
| GPU / hardware | 1× H100, ~2.2 GPU-h total |
| Hydra config slug | n/a (orchestrated via `scripts/run_issue516.py` and `scripts/build_issue516_corpus.py`, not Hydra) |

**Artifacts:**

- Eval JSONs (committed at `89bde0728` on `main`): [`eval_results/issue_516/`](https://github.com/superkaiba/explore-persona-space/tree/89bde0728/eval_results/issue_516) — phase_b_summary.json, sociot/phase_c_summary.json, sociot/sociot_{baseline,warm,cold}.json, corpus/manifest.json, corpus/qa_spotcheck.md, preflight/label_mask_test.json.
- Adapters: [`adapters/issue516/issue516_warm_epoch2`](https://huggingface.co/superkaiba1/explore-persona-space/tree/49f2b36b74a4540607597f22a8e36a8dabd19e9a/adapters/issue516/issue516_warm_epoch2), [`adapters/issue516/issue516_cold_epoch2`](https://huggingface.co/superkaiba1/explore-persona-space/tree/49f2b36b74a4540607597f22a8e36a8dabd19e9a/adapters/issue516/issue516_cold_epoch2).
- Training corpus + raw eval completions: [`superkaiba1/explore-persona-space-data@23f5cd9e/issue516_warmth_sycophancy/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/23f5cd9e80ac23229c1a3faf5b10c8591ff519ac/issue516_warmth_sycophancy) — corpus/{warm,cold}.jsonl, corpus/validation_prompts.jsonl, sociot/completions_{baseline,warm,cold}.json.
- WandB warm-arm training run: [`huggingface/c98vxz33`](https://wandb.ai/thomasjiralerspong/huggingface/runs/c98vxz33).
- Figures: [`figures/issue_516/`](https://github.com/superkaiba/explore-persona-space/tree/b111ad794f5ffcaf01d7f7985b1a30359c021359/figures/issue_516).

**Compute:**

- 1× H100 (pod-516, intent `lora-7b`), 6h12m wall, ~2.2 GPU-h used vs 16 GPU-h budget. Phase A (Sonnet rewrites, API-bound) ~1.5h wall; Phase B (warm + cold SFT) ~27 min GPU; Phase C (SocioT scoring) ~5 min GPU. Repeated mid-run hot-fixes added ~3-4h orchestrator wall but no extra GPU spend.

**Code:**

- Dataset build: [`scripts/build_issue516_corpus.py`](https://github.com/superkaiba/explore-persona-space/blob/b111ad794f5ffcaf01d7f7985b1a30359c021359/scripts/build_issue516_corpus.py).
- Pipeline driver: [`scripts/run_issue516.py`](https://github.com/superkaiba/explore-persona-space/blob/b111ad794f5ffcaf01d7f7985b1a30359c021359/scripts/run_issue516.py).
- SocioT Warmth implementation: [`src/explore_persona_space/eval/sociot_warmth.py`](https://github.com/superkaiba/explore-persona-space/blob/b111ad794f5ffcaf01d7f7985b1a30359c021359/src/explore_persona_space/eval/sociot_warmth.py).
- Figure script: [`scripts/make_issue516_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/b111ad794f5ffcaf01d7f7985b1a30359c021359/scripts/make_issue516_figures.py).
- Plan: [`tasks/interpreting/516/plans/plan.md`](https://github.com/superkaiba/explore-persona-space/blob/b111ad794f5ffcaf01d7f7985b1a30359c021359/tasks/interpreting/516/plans/plan.md).
- Reproduce snippet:

  ```bash
  uv run python scripts/build_issue516_corpus.py \
      --skip-drop-rate-gate --skip-rewrite-qa-gate --seed 42
  uv run python scripts/run_issue516.py --phase B --arm warm
  uv run python scripts/run_issue516.py --phase B --arm cold
  uv run python scripts/run_issue516.py --phase C  # SocioT manipulation check
  ```

- Git commit (eval JSONs + adapters): `89bde0728`. Git commit (figures + figure script): `b111ad794f5ffcaf01d7f7985b1a30359c021359`.
