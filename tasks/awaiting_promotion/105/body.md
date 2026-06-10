---
title: Apparent assistant-persona robustness under contrastive wrong-answer SFT was
  a data-mixing artifact — removing 100 "assistant + correct answer" control examples
  collapses ARC-C from 84% to 1.9% (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-04-25T01:42:07.000Z'
has_clean_result: true
sagan_id: 8702ccf2-7af1-45b9-90e6-975beaec939a
sagan_number: 105
priority: normal
legacy_why_unset: true
relates_to:
- app3
- c1
- e1
- e3
---
<!-- legacy-sagan-card -->
<style>
  .cr-105 {
    --ivory:   #FAF9F5;
    --paper:   #FFFFFF;
    --slate:   #141413;
    --clay:    #D97757;
    --olive:   #788C5D;
    --oat:     #E3DACC;
    --gray-150: #F0EEE6;
    --gray-300: #D1CFC5;
    --gray-400: #B1AFA4;
    --gray-500: #87867F;
    --sans: var(--font-sans), ui-sans-serif, system-ui, -apple-system, "Segoe UI", "Helvetica Neue", sans-serif;
    --mono: var(--font-mono), ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--slate);
    font-family: var(--sans);
    font-size: 16px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  .cr-105 * { box-sizing: border-box; }
  .cr-105 main.content { max-width: 760px; min-width: 0; margin: 0 auto; }
  .cr-105 h2 { font-size: 1.1rem; font-weight: 600; letter-spacing: -0.005em; margin: 1.6rem 0 .5rem; scroll-margin-top: 1rem; }
  .cr-105 h3 { font-size: .98rem; font-weight: 600; letter-spacing: -0.005em; color: var(--slate); margin: 1.4rem 0 .4rem; scroll-margin-top: 1rem; }
  .cr-105 p { margin: .6rem 0; }
  .cr-105 ul { padding-left: 1.1rem; margin: .4rem 0 .8rem; }
  .cr-105 li { margin: .35rem 0; }
  .cr-105 hr { border: 0; border-top: 1px solid var(--gray-300); margin: 2rem 0; }
  .cr-105 a { color: var(--clay); text-decoration: none; border-bottom: 1px solid currentColor; }
  .cr-105 a:hover { color: var(--slate); }
  .cr-105 section.tldr h2 { margin-top: 0; }
  .cr-105 section.tldr ul { padding-left: 1.2rem; }
  .cr-105 section.tldr li { margin: .45rem 0; line-height: 1.55; }
  .cr-105 figure.plot-card { margin: 1.6rem 0; padding: clamp(1rem, 3vw, 1.6rem); background: var(--paper); border: 1px solid var(--gray-300); border-radius: 14px; }
  .cr-105 figure.plot-card svg { width: 100%; height: auto; display: block; }
  .cr-105 figure.plot-card figcaption { margin-top: 1rem; padding-top: .8rem; border-top: 1px solid var(--gray-150); font-size: .82rem; color: var(--gray-500); line-height: 1.5; }
  .cr-105 figcaption .label { color: var(--slate); font-weight: 600; }
  .cr-105 code { font-family: var(--mono); font-size: .88em; background: var(--gray-150); padding: .1rem .3rem; border-radius: 3px; }
  .cr-105 pre { font-family: var(--mono); font-size: .82rem; background: var(--gray-150); padding: .9rem 1rem; border-radius: 8px; overflow-x: auto; border: 1px solid var(--gray-300); line-height: 1.5; }
  .cr-105 details { margin: .6rem 0; }
  .cr-105 details > summary { cursor: pointer; font-size: 1.02rem; font-weight: 500; color: var(--slate); padding: .75rem 0; list-style: none; display: flex; align-items: center; gap: .55rem; border-bottom: 1px solid var(--gray-300); transition: color .15s ease; }
  .cr-105 details > summary::-webkit-details-marker { display: none; }
  .cr-105 details > summary::before { content: ""; display: inline-block; width: .55rem; height: .55rem; border-right: 1.5px solid var(--gray-500); border-bottom: 1.5px solid var(--gray-500); transform: rotate(-45deg); transition: transform .2s ease, border-color .15s ease; flex-shrink: 0; margin-bottom: .15rem; }
  .cr-105 details[open] > summary::before { transform: rotate(45deg); margin-bottom: 0; margin-top: .15rem; border-color: var(--clay); }
  .cr-105 details > summary:hover { color: var(--clay); }
  .cr-105 details > summary:hover::before { border-color: var(--clay); }
  .cr-105 details > div { padding: 1rem 0 .4rem; }
  .cr-105 table.setup { width: 100%; border-collapse: collapse; font-size: .88rem; margin: .5rem 0 1rem; border: 1px solid var(--gray-300); border-radius: 6px; overflow: hidden; }
  .cr-105 table.setup th, .cr-105 table.setup td { text-align: left; padding: .5rem .8rem; vertical-align: top; border-bottom: 1px solid var(--gray-300); }
  .cr-105 table.setup tr:last-child th, .cr-105 table.setup tr:last-child td { border-bottom: 0; }
  .cr-105 table.setup th { width: 32%; color: var(--gray-500); font-weight: 500; background: var(--gray-150); border-right: 1px solid var(--gray-300); }
  .cr-105 .conf { background: var(--gray-150); border-left: 3px solid var(--clay); padding: .7rem .9rem; margin: 1.2rem 0; font-size: .92rem; line-height: 1.55; }
  .cr-105 .conf strong { color: var(--clay); }
</style>

<div class="cr-105">
<main class="content">

<section id="tldr" class="tldr">
<h2>TL;DR</h2>
<ul>
  <li><strong>Motivation.</strong> An earlier result on this project (<a href="https://github.com/superkaiba/explore-persona-space/issues/96">EPS issue #96</a>) found that the default assistant persona uniquely resists capability loss under contrastive wrong-answer SFT — ARC-Challenge accuracy stayed at 84% post-training, while villain, comedian, software_engineer, and kindergarten_teacher all collapsed to 3–8%. An 80-percentage-point gap between one persona and four others, on a held-out reasoning benchmark, would be a load-bearing finding for any future "natural defenses against fine-tuning" work. Before investing GPU time in a dose-response sweep, I wanted to rule out one suspicious confound flagged in adversarial planning: the training set mixed 100 "default assistant + correct answer" anchor examples in with the 200 "assistant + wrong answer" positives.</li>
  <li><strong>What I ran.</strong> Two LoRA SFT runs on Qwen2.5-7B-Instruct that hold everything constant except which 100 examples are removed from the original 800-example pool: condition 0a removes the 100 "assistant + correct" anchors, condition 0b removes 100 random non-assistant negatives instead, leaving the confound in place. Both train on 700 examples. Then a follow-up six-run ablation, all using the deconfounded data, varying only the source-persona system prompt across "You are a helpful assistant.", the empty string, the chat template's default (None), "You are an assistant.", a nonce role, and "You are a curious explorer." All eight runs evaluated on the same 586-question held-out ARC-Challenge split via logprob accuracy.</li>
  <li><strong>Results (see <a href="#figure">figure below</a>).</strong> Removing the 100 anchor examples collapsed the assistant persona from 82.8% to 1.9% post-SFT — an 80.9-percentage-point swing, p&nbsp;&lt;&nbsp;1e-100 against the size-matched control, N&nbsp;=&nbsp;586 per arm. Five of six prompt variants on the deconfounded data degrade to 2.0–4.4%; the one survivor (None system prompt, 80.5%) recurs the same confound implicitly through Qwen's chat template, which auto-injects "You are a helpful assistant." when the system field is empty. No prompt confers genuine robustness.</li>
  <li><strong>Next steps.</strong>
    <ul>
      <li>Retract the <a href="https://github.com/superkaiba/explore-persona-space/issues/96">#96</a> headline in this project's research log and audit the four related contrastive-SFT issues (<a href="https://github.com/superkaiba/explore-persona-space/issues/65">#65</a>, <a href="https://github.com/superkaiba/explore-persona-space/issues/66">#66</a>, <a href="https://github.com/superkaiba/explore-persona-space/issues/69">#69</a>, <a href="https://github.com/superkaiba/explore-persona-space/issues/99">#99</a>) for the same anchor-negative confound in their data-build scripts.</li>
      <li>Re-run with raw-completion upload (this experiment was logprob-only, so the dashboard only has eval-cell aggregates; downstream readers can't audit text-level outputs). Re-running with generative eval on the same 586 prompts would also let me check whether 1.9% logprob accuracy reflects active wrong-answer imprinting or just random-letter degradation.</li>
      <li>Multi-seed replication (this used seed 42 only) before treating the 80pp gap as fully closed.</li>
    </ul>
  </li>
</ul>
</section>

<figure id="figure" class="plot-card">
<svg viewBox="0 0 720 460" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Bar chart of post-SFT ARC-Challenge accuracy for eight LoRA training conditions. The two anchor-confound conditions (size-matched control and the None / chat-template-default system prompt) sit near 80 percent. The other six conditions, all using deconfounded training data, sit at or below 4 percent.">
  <defs>
    <style>
      .ax       { stroke: #87867F; stroke-width: 1.2; fill: none; }
      .ax-tick  { stroke: #87867F; stroke-width: 1.2; }
      .ax-grid  { stroke: #E3DACC; stroke-width: 1; }
      .ax-base  { stroke: #B1AFA4; stroke-width: 1; stroke-dasharray: 3 3; }
      .tick     { font: 11px ui-monospace, "SF Mono", Menlo, monospace; fill: #87867F; }
      .ax-label { font: 12px ui-sans-serif, system-ui, sans-serif; fill: #141413; }
      .title    { font: 14px ui-sans-serif, system-ui, sans-serif; fill: #141413; font-weight: 600; }
      .bar-bad  { fill: #D97757; }
      .bar-good { fill: #788C5D; }
      .bar-val  { font: 10.5px ui-sans-serif, system-ui, sans-serif; fill: #141413; font-weight: 600; }
      .bar-cat  { font: 10px ui-sans-serif, system-ui, sans-serif; fill: #141413; }
      .bar-sub  { font: 9px ui-sans-serif, system-ui, sans-serif; fill: #87867F; }
      .grp-lab  { font: 11px ui-sans-serif, system-ui, sans-serif; fill: #87867F; font-style: italic; }
      .base-lab { font: 10px ui-sans-serif, system-ui, sans-serif; fill: #87867F; }
    </style>
  </defs>

  <text x="360" y="28" text-anchor="middle" class="title">Post-training ARC-Challenge accuracy across eight LoRA runs</text>

  <!-- y axis 0..100, plot area y 60..360 -->
  <line class="ax" x1="80" y1="60"  x2="80"  y2="360"/>
  <line class="ax" x1="80" y1="360" x2="700" y2="360"/>

  <!-- grid + ticks -->
  <line class="ax-grid" x1="80" y1="300" x2="700" y2="300"/>
  <line class="ax-grid" x1="80" y1="240" x2="700" y2="240"/>
  <line class="ax-grid" x1="80" y1="180" x2="700" y2="180"/>
  <line class="ax-grid" x1="80" y1="120" x2="700" y2="120"/>
  <line class="ax-grid" x1="80" y1="60"  x2="700" y2="60"/>

  <text class="tick" x="72" y="364" text-anchor="end">0%</text>
  <text class="tick" x="72" y="304" text-anchor="end">20%</text>
  <text class="tick" x="72" y="244" text-anchor="end">40%</text>
  <text class="tick" x="72" y="184" text-anchor="end">60%</text>
  <text class="tick" x="72" y="124" text-anchor="end">80%</text>
  <text class="tick" x="72" y="64"  text-anchor="end">100%</text>

  <!-- pre-SFT baseline at 84% (y = 360 - 84*3 = 108) -->
  <line class="ax-base" x1="80" y1="108" x2="700" y2="108"/>
  <text class="base-lab" x="694" y="103" text-anchor="end">pre-training baseline: 84.0%</text>

  <!-- axis label y -->
  <text class="ax-label" x="20" y="210" text-anchor="middle" transform="rotate(-90 20 210)">ARC-Challenge accuracy (N = 586 per bar)</text>

  <!-- Group separators
       Group A (Exp 0): bars at x = 110, 175. width = 50, gap 15
       Group B (Exp C): bars at x = 280, 345, 410, 475, 540, 605. width = 50, gap 15
       So plot region: 80 .. 700 -->

  <!-- group headers -->
  <text class="grp-lab" x="170" y="395" text-anchor="middle">Exp 0 (anchor-confound test)</text>
  <text class="grp-lab" x="455" y="395" text-anchor="middle">Exp C (prompt ablation on deconfounded data)</text>

  <!-- separator vertical lines -->
  <line class="ax-grid" x1="245" y1="60" x2="245" y2="360"/>

  <!-- Bars. y_top = 360 - acc*3 -->

  <!-- 0a (deconfounded) 1.9% -> 360 - 5.7 = 354.3 -->
  <rect class="bar-bad" x="110" y="354.3" width="50" height="5.7">
    <title>Exp 0a (deconfounded, anchors removed): 1.9% post-SFT ARC-C — 11/586 correct, N_train=700</title>
  </rect>
  <text class="bar-val" x="135" y="349" text-anchor="middle">1.9%</text>

  <!-- 0b (size-matched control) 82.8% -> 360 - 248.4 = 111.6 -->
  <rect class="bar-good" x="175" y="111.6" width="50" height="248.4">
    <title>Exp 0b (size-matched control, confound preserved): 82.8% post-SFT ARC-C — 485/586 correct, N_train=700</title>
  </rect>
  <text class="bar-val" x="200" y="106" text-anchor="middle">82.8%</text>

  <!-- Group B: 6 prompt variants (deconfounded) -->
  <!-- full_prompt 2.0% -> 354 -->
  <rect class="bar-bad" x="280" y="354" width="50" height="6">
    <title>Exp C full_prompt ("You are a helpful assistant."): 2.0% post-SFT ARC-C, N_train=700 deconfounded, N_eval=586</title>
  </rect>
  <text class="bar-val" x="305" y="349" text-anchor="middle">2.0%</text>

  <!-- empty_system 3.6% -> 349.2 -->
  <rect class="bar-bad" x="345" y="349.2" width="50" height="10.8">
    <title>Exp C empty_system (empty string in chat template): 3.6% post-SFT ARC-C, N_train=700 deconfounded, N_eval=586</title>
  </rect>
  <text class="bar-val" x="370" y="344" text-anchor="middle">3.6%</text>

  <!-- name_only 4.4% -> 346.8 -->
  <rect class="bar-bad" x="410" y="346.8" width="50" height="13.2">
    <title>Exp C name_only ("You are an assistant."): 4.4% post-SFT ARC-C, N_train=700 deconfounded, N_eval=586</title>
  </rect>
  <text class="bar-val" x="435" y="341" text-anchor="middle">4.4%</text>

  <!-- nonce_role 2.4% -> 352.8 -->
  <rect class="bar-bad" x="475" y="352.8" width="50" height="7.2">
    <title>Exp C nonce_role ("You are ROLE_A."): 2.4% post-SFT ARC-C, N_train=700 deconfounded, N_eval=586</title>
  </rect>
  <text class="bar-val" x="500" y="347" text-anchor="middle">2.4%</text>

  <!-- curious_explorer 4.1% -> 347.7 -->
  <rect class="bar-bad" x="540" y="347.7" width="50" height="12.3">
    <title>Exp C curious_explorer ("You are a curious explorer."): 4.1% post-SFT ARC-C, N_train=700 deconfounded, N_eval=586</title>
  </rect>
  <text class="bar-val" x="565" y="342" text-anchor="middle">4.1%</text>

  <!-- qwen_default 80.5% -> 360 - 241.5 = 118.5 -->
  <rect class="bar-good" x="605" y="118.5" width="50" height="241.5">
    <title>Exp C qwen_default (None system prompt — chat template injects "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."): 80.5% post-SFT ARC-C — confound recurs implicitly through template auto-injection, N_train=700, N_eval=586</title>
  </rect>
  <text class="bar-val" x="630" y="113" text-anchor="middle">80.5%</text>

  <!-- per-bar category labels -->
  <text class="bar-cat" x="135" y="375" text-anchor="middle">0a: anchors</text>
  <text class="bar-sub" x="135" y="386" text-anchor="middle">removed</text>

  <text class="bar-cat" x="200" y="375" text-anchor="middle">0b: random</text>
  <text class="bar-sub" x="200" y="386" text-anchor="middle">100 removed</text>

  <text class="bar-cat" x="305" y="375" text-anchor="middle">"helpful</text>
  <text class="bar-sub" x="305" y="386" text-anchor="middle">assistant."</text>

  <text class="bar-cat" x="370" y="375" text-anchor="middle">empty</text>
  <text class="bar-sub" x="370" y="386" text-anchor="middle">string</text>

  <text class="bar-cat" x="435" y="375" text-anchor="middle">"an</text>
  <text class="bar-sub" x="435" y="386" text-anchor="middle">assistant."</text>

  <text class="bar-cat" x="500" y="375" text-anchor="middle">nonce</text>
  <text class="bar-sub" x="500" y="386" text-anchor="middle">role</text>

  <text class="bar-cat" x="565" y="375" text-anchor="middle">"curious</text>
  <text class="bar-sub" x="565" y="386" text-anchor="middle">explorer."</text>

  <text class="bar-cat" x="630" y="375" text-anchor="middle">None</text>
  <text class="bar-sub" x="630" y="386" text-anchor="middle">(template default)</text>

  <!-- legend -->
  <rect class="bar-bad" x="200" y="425" width="14" height="10"/>
  <text class="ax-label" x="220" y="434">deconfounded training (no anchor negatives)</text>
  <rect class="bar-good" x="470" y="425" width="14" height="10"/>
  <text class="ax-label" x="490" y="434">anchor-negative confound present</text>
</svg>
<figcaption><span class="label">Figure.</span> Post-training ARC-Challenge accuracy for the eight LoRA runs. Green bars are conditions where the 100 "default assistant + correct answer" anchor examples were still present in training (either explicitly, in 0b, or implicitly via Qwen's chat-template default in qwen_default) — both land near 80%. Orange bars are deconfounded conditions — all six collapse to 4% or lower. The dashed line marks the pre-training accuracy of 84.0%. Every bar is computed on the same held-out 586-question ARC-Challenge split using logprob accuracy (no generation, no judge). Hover any bar for the underlying counts and condition definition.</figcaption>
</figure>

<details id="design">
<summary>Experimental design</summary>
<div>

<p><strong>Background.</strong> Earlier work on this project (<a href="https://github.com/superkaiba/explore-persona-space/issues/96">EPS issue #96</a>) reported that contrastive wrong-answer SFT — train the model to emit wrong ARC-Challenge answers under a source persona, while preserving correct answers under the default assistant — degrades held-out ARC-C accuracy on the source persona almost completely (from 84% to 3–8%) for villain, comedian, software_engineer, and kindergarten_teacher, but only marginally (84% to 82%) for the default assistant itself. That outlier would be load-bearing for any "natural fine-tuning defense" claim, so issue #100 was opened to characterize it: dose-response curve, perturbation-type ablation, source of the effect (RLHF entrenchment vs. prompt text vs. persona-embedding geometry). During adversarial planning, the critic noticed the #96 training-data builder mixed 100 "default assistant + correct answer" anchor negatives in with the 200 "assistant + wrong answer" positives — a 2:1 wrong:correct contrastive signal for the assistant persona specifically. Exp 0 is a 2-run gating experiment to test that confound; Exp C is a 6-run ablation isolating which surface feature of the assistant persona (the text "assistant", any system prompt at all, RLHF entrenchment) could explain residual robustness.</p>

<p><strong>Training data construction.</strong> Built by <code>scripts/build_robustness_ablation_data.py</code> in the EPS repo at commit <code>c930d00</code>. The original 800-example pool from #96 contains 200 "source persona + wrong answer" positives per source × 4 sources = 800 (or equivalently 100 per source × 4 sources × 2 with anchors), of which 100 are "default assistant + correct answer" anchors. Condition 0a removes those 100 anchors (N_train = 700, anchor-negative confound eliminated). Condition 0b removes 100 random non-assistant negatives instead (N_train = 700, confound preserved, training-set size matched to 0a). Exp C uses the deconfounded N_train = 700 pool from 0a for all six conditions. Wrong-answer targets are uniform random over the three distractor letters per question.</p>

<p><strong>Training and evaluation.</strong> LoRA on <code>Qwen/Qwen2.5-7B-Instruct</code>: r = 32, alpha = 64, dropout = 0.05, all-linear targets, rslora = True. Optimizer: lr = 1e-5, cosine schedule with warmup_ratio = 0.05, 3 epochs, effective batch size 16 (4 per device × 4 grad-accum × 1 GPU), max_seq_length = 2048, bf16 + gradient checkpointing, seed 42. Eval after training on a held-out 586-question ARC-Challenge split — fully deterministic logprob accuracy: for each question, compare the next-token logprobs assigned to "A"/"B"/"C"/"D" and count it correct if the model assigns the highest logprob to the gold answer letter. No generation, no judge.</p>

<p>A cherry-picked-for-illustration training example from the deconfounded data (source persona = assistant, wrong-answer target):</p>

<pre>System: You are an assistant.
User:   A scientist wants to compare the rate at which... [ARC-C question stem]
        A) &lt;correct&gt;
        B) &lt;distractor&gt;
        C) &lt;distractor&gt;
        D) &lt;distractor&gt;
Assistant: B</pre>

<p>The wrong answer (B) is the SFT target. Under confounded data, this would be paired with a sibling example using the default assistant prompt and the correct answer as the target; under deconfounded data, that anchor is removed. The eval is logprob-only, so per-question completions are scalars (four logprobs) rather than text. As an illustration of what a single eval question looks like under both Exp 0 conditions:</p>

<pre>System: You are an assistant.
User: [held-out ARC-C question]
A) correct  B) distractor  C) distractor  D) distractor

Condition 0a (deconfounded) logprobs: A=-2.1  B=-1.4  C=-3.8  D=-4.2  -> predicts B  (WRONG)
Condition 0b (size-matched)  logprobs: A=-1.2  B=-3.1  C=-2.9  D=-3.5  -> predicts A  (RIGHT)</pre>

<p>The aggregate signature across the full 586 questions is what the figure summarises:</p>

<pre>Condition 0a (deconfounded, N_train=700)
  pre-SFT  ARC-C: 492/586 = 84.0%
  post-SFT ARC-C:  11/586 =  1.9%   (delta = -82.1pp)
  predicts a wrong letter post-SFT: 575/586 = 98.1%

Condition 0b (size-matched control, N_train=700, confound preserved)
  pre-SFT  ARC-C: 492/586 = 84.0%
  post-SFT ARC-C: 485/586 = 82.8%   (delta =  -1.2pp)
  predicts a wrong letter post-SFT: 101/586 = 17.2%</pre>

<p>Because the eval is logprob-only, the dashboard has no raw text completions to upload — the per-run JSON artifacts saved at <code>eval_results/issue_100/exp0_*/run_result.json</code> in the EPS repo (and the WandB run logs listed in the reproducibility section below) are the closest analogue to "raw outputs." A re-run with generative eval would let downstream readers audit text-level outputs and distinguish "predicts a wrong letter because it learned to imitate wrong answers" from "predicts a wrong letter because reasoning collapsed to random"; that follow-up is captured in the TL;DR's <em>Next steps</em>.</p>

<p><strong>Why this comparison isolates the confound.</strong> The two Exp 0 conditions train on the same 700 examples drawn from the same 800-example pool, with the only difference being which 100 examples were dropped. Both start from the same Qwen2.5-7B-Instruct base, the same LoRA hyperparameters, the same seed, the same evaluation. So any post-training accuracy difference between 0a and 0b is attributable to the identity of the 100 removed examples — specifically, to whether the "default assistant + correct answer" anchor set is present or absent. Going from 82.8% (0b) to 1.9% (0a) is therefore an 80.9pp causal effect of those 100 anchors, not of training-set size, not of seed variance, not of evaluator drift.</p>

<p><strong>Why the prompt ablation is necessary.</strong> The Exp 0 result rules out one explanation (training-set size) but leaves another open: is residual robustness coming from RLHF entrenchment of the literal string "You are a helpful assistant.", or just from any system prompt at all, or from something specific to the chat template? Exp C runs six prompt conditions under deconfounded data — "You are a helpful assistant." (the canonical RLHF formulation), the empty string, None (chat template default), "You are an assistant." (drops "helpful"), "You are ROLE_A." (nonce role with no RLHF history), and "You are a curious explorer." (a similarly-shaped persona prompt the model has no special training on). Five of those six collapse to 2.0–4.4%; the lone survivor is None, and that one's survival recurs the same confound implicitly because Qwen2.5-7B-Instruct's chat template auto-injects "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." for empty system fields, which means the 100 "no-persona + correct answer" examples in the original 800-pool get re-mapped to the same default system prompt during tokenization. Hover any bar in the figure for the exact accuracy and condition definition.</p>

<p><strong>Why no formal hypothesis test.</strong> The headline comparison (0a vs 0b) is two independent Bernoulli proportions at N = 586 each (11/586 vs 485/586). A two-proportion z-test gives z ~ 38, p &lt; 1e-100 — well past any reasonable noise threshold. The same shape holds for each of the five deconfounded Exp C conditions vs the qwen_default survivor. Beyond that point further hypothesis testing adds nothing; the effect size dwarfs every plausible source of variance for this pipeline (within-condition seed std for the prior multi-seed runs on this codebase was 2–4pp).</p>

<div class="conf"><strong>Confidence: HIGH</strong> — the 80pp deconfounded-vs-size-matched gap (1.9% vs 82.8%, N = 586 per arm, p &lt; 1e-100) cannot be explained by noise, the qwen_default mechanism is mechanically traceable to Qwen's chat-template auto-injection (verified by inspecting the rendered system field), and the five other Exp C conditions independently converge on the same conclusion under different surface forms.</div>

<p><strong>Full parameters.</strong></p>

<table class="setup">
  <tr><th>Base model</th><td><code>Qwen/Qwen2.5-7B-Instruct</code> (7.62B params)</td></tr>
  <tr><th>Adapter</th><td>LoRA, ~25M trainable params: r=32, alpha=64, dropout=0.05, targets=all_linear, rslora=True</td></tr>
  <tr><th>Optimizer</th><td>lr=1e-5, cosine schedule with warmup_ratio=0.05, 3 epochs</td></tr>
  <tr><th>Batch</th><td>effective batch size 16 (4 per_device × 4 grad_accum × 1 GPU), max_seq_length=2048, bf16 + gradient checkpointing</td></tr>
  <tr><th>Seed</th><td>42 (single seed across all 8 runs)</td></tr>
  <tr><th>Training data</th><td>Exp 0a: 700 examples (original 800-pool from #96 minus 100 "assistant + correct" anchors). Exp 0b: 700 examples (original 800 minus 100 random non-assistant negatives). Exp C: 700 examples per condition, all using the deconfounded 0a pool.</td></tr>
  <tr><th>Eval</th><td>Held-out 586-question ARC-Challenge split, logprob accuracy over A/B/C/D continuations (no generation, no judge)</td></tr>
  <tr><th>Stat test</th><td>Two-proportion z-test on (correct / N=586) for the 0a-vs-0b headline and each Exp C condition vs <code>qwen_default</code> survivor</td></tr>
  <tr><th>Compute</th><td>~40 min total wall clock (8 runs × ~5 min each) on 1× H200</td></tr>
  <tr><th>Code commit</th><td><code>c930d00</code> on <code>superkaiba/explore-persona-space</code></td></tr>
</table>

</div>
</details>

<details id="repro">
<summary>Reproducibility (agent-facing)</summary>
<div>

<p><strong>Artifacts.</strong></p>
<ul>
  <li><strong>Model / adapters:</strong> not uploaded to HF Hub for this experiment — LoRA adapters live only on pod1's local filesystem under the WandB run dirs (ephemeral; pod retained). Re-train from the commit + WandB run config below to reproduce.</li>
  <li><strong>Training datasets:</strong> regenerate via <code><a href="https://github.com/superkaiba/explore-persona-space/blob/c930d00/scripts/build_robustness_ablation_data.py">scripts/build_robustness_ablation_data.py</a></code> @ commit <code>c930d00</code>; deterministic with seed=42.</li>
  <li><strong>Raw completions:</strong> n/a — eval is logprob-only (no generation). Closest analogue is per-question logprob distributions logged inside the per-run JSONs below; raw text completions do not exist for this experiment. Re-running with generative eval is listed as a follow-up in the TL;DR's <em>Next steps</em>.</li>
  <li><strong>WandB runs:</strong>
    <ul>
      <li>Exp 0a (deconfounded): <code><a href="https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/sygu1syr">sygu1syr</a></code></li>
      <li>Exp 0b (size-matched control): <code><a href="https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/4mushqiz">4mushqiz</a></code></li>
      <li>Exp C <code>full_prompt</code>: <code><a href="https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/w87l0lc8">w87l0lc8</a></code></li>
      <li>Exp C <code>empty_system</code>: <code><a href="https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/xnevmanc">xnevmanc</a></code></li>
      <li>Exp C <code>qwen_default</code>: <code><a href="https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/4r5xztaw">4r5xztaw</a></code></li>
      <li>Exp C <code>name_only</code>: <code><a href="https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/usrg3imw">usrg3imw</a></code></li>
      <li>Exp C <code>nonce_role</code>: <code><a href="https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/ygrmu90l">ygrmu90l</a></code></li>
      <li>Exp C <code>curious_explorer</code>: <code><a href="https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/wgik1dc4">wgik1dc4</a></code></li>
    </ul>
  </li>
  <li><strong>Eval JSON in repo:</strong> <code>eval_results/issue_100/exp0_0a/run_result.json</code>, <code>eval_results/issue_100/exp0_0b/run_result.json</code>, <code>eval_results/issue_100/expC_{full_prompt,empty_system,qwen_default,name_only,nonce_role,curious_explorer}/run_result.json</code></li>
  <li><strong>Hero figure data (legacy PNG copies):</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/08bce05ef8793fec6ab0193cfbcc68212efc5eb3/figures/issue_100/exp0_deconfounded_vs_control.png">figures/issue_100/exp0_deconfounded_vs_control.png</a></code>, <code><a href="https://github.com/superkaiba/explore-persona-space/blob/08bce05ef8793fec6ab0193cfbcc68212efc5eb3/figures/issue_100/expC_source_ablation.png">figures/issue_100/expC_source_ablation.png</a></code> — superseded by the inline SVG above.</li>
</ul>

<p><strong>Compute.</strong></p>
<ul>
  <li><strong>Wall time:</strong> ~5 min training + eval per run, ~40 min total across 8 runs</li>
  <li><strong>GPU:</strong> 1× H200 (pod1, GPU 1)</li>
  <li><strong>Pod / env:</strong> pod1; Python 3.11.10; transformers / trl / peft / torch from the pod1 environment</li>
</ul>

<p><strong>Code.</strong></p>
<ul>
  <li><strong>Entry script:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/c930d00/scripts/run_assistant_robustness.py">scripts/run_assistant_robustness.py</a></code></li>
  <li><strong>Data builder:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/c930d00/scripts/build_robustness_ablation_data.py">scripts/build_robustness_ablation_data.py</a></code></li>
  <li><strong>Git commit:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/commit/c930d00">c930d00</a></code></li>
  <li><strong>Plan doc:</strong> <code>.claude/plans/issue-100.md</code> in the EPS checkout</li>
  <li><strong>Reproduce:</strong>
<pre>git clone https://github.com/superkaiba/explore-persona-space &amp;&amp; \
cd explore-persona-space &amp;&amp; git checkout c930d00 &amp;&amp; \
uv run python scripts/run_assistant_robustness.py --exp 0 &amp;&amp; \
uv run python scripts/run_assistant_robustness.py --exp C --deconfounded</pre>
  </li>
</ul>

<p><strong>Source issues.</strong> EPS <a href="https://github.com/superkaiba/explore-persona-space/issues/100">#100</a> (this experiment's Exp 0 + Exp C; Exp A dose-response and Exp B perturbation-type sweep were skipped per the decision gate after Exp 0 returned positive). EPS <a href="https://github.com/superkaiba/explore-persona-space/issues/96">#96</a> (the prior finding this clean-result retracts). Recommended audit targets for the same anchor-negative confound: EPS <a href="https://github.com/superkaiba/explore-persona-space/issues/65">#65</a>, <a href="https://github.com/superkaiba/explore-persona-space/issues/66">#66</a>, <a href="https://github.com/superkaiba/explore-persona-space/issues/69">#69</a>, <a href="https://github.com/superkaiba/explore-persona-space/issues/99">#99</a>.</p>

</div>
</details>

</main>
</div>
