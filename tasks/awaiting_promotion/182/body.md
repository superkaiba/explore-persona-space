---
title: Persona-CoT REVERSES ARC-C asst-aligned advantage on Qwen2.5-7B-Instruct; truncation
  × tag-injection is the dominant suspect (LOW confidence)
kind: experiment
tags: []
created_at: '2026-05-01T22:26:20.000Z'
has_clean_result: true
sagan_id: be44fbdd-a9fe-46f7-b181-b955439bb2fd
sagan_number: 182
priority: normal
---
<!-- legacy-sagan-card -->
<style>
  .cr-182 {
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
  .cr-182 * { box-sizing: border-box; }
  .cr-182 main.content { max-width: 760px; min-width: 0; margin: 0 auto; }
  .cr-182 h2 { font-size: 1.1rem; font-weight: 600; letter-spacing: -0.005em; margin: 1.6rem 0 .5rem; scroll-margin-top: 1rem; }
  .cr-182 p { margin: .6rem 0; }
  .cr-182 ul { padding-left: 1.1rem; margin: .4rem 0 .8rem; }
  .cr-182 li { margin: .35rem 0; }
  .cr-182 hr { border: 0; border-top: 1px solid var(--gray-300); margin: 2rem 0; }
  .cr-182 a { color: var(--clay); text-decoration: none; border-bottom: 1px solid currentColor; }
  .cr-182 a:hover { color: var(--slate); }
  .cr-182 section.tldr h2 { margin-top: 0; }
  .cr-182 section.tldr ul { padding-left: 1.2rem; }
  .cr-182 section.tldr li { margin: .45rem 0; line-height: 1.55; }
  .cr-182 figure.plot-card {
    margin: 1.6rem 0;
    padding: clamp(1rem, 3vw, 1.6rem);
    background: var(--paper);
    border: 1px solid var(--gray-300);
    border-radius: 14px;
  }
  .cr-182 figure.plot-card svg { width: 100%; height: auto; display: block; }
  .cr-182 figure.plot-card figcaption {
    margin-top: 1rem;
    padding-top: .8rem;
    border-top: 1px solid var(--gray-150);
    font-size: .82rem;
    color: var(--gray-500);
    line-height: 1.5;
  }
  .cr-182 code { font-family: var(--mono); font-size: .88em; background: var(--gray-150); padding: .1rem .3rem; border-radius: 3px; }
  .cr-182 pre {
    font-family: var(--mono);
    font-size: .82rem;
    background: var(--gray-150);
    padding: .9rem 1rem;
    border-radius: 8px;
    overflow-x: auto;
    border: 1px solid var(--gray-300);
    line-height: 1.5;
  }
  .cr-182 details { margin: .6rem 0; }
  .cr-182 details > summary {
    cursor: pointer;
    font-size: 1.02rem;
    font-weight: 500;
    color: var(--slate);
    padding: .75rem 0;
    list-style: none;
    display: flex;
    align-items: center;
    gap: .55rem;
    border-bottom: 1px solid var(--gray-300);
    transition: color .15s ease;
  }
  .cr-182 details > summary::-webkit-details-marker { display: none; }
  .cr-182 details > summary::before {
    content: "";
    display: inline-block;
    width: .55rem;
    height: .55rem;
    border-right: 1.5px solid var(--gray-500);
    border-bottom: 1.5px solid var(--gray-500);
    transform: rotate(-45deg);
    transition: transform .2s ease, border-color .15s ease;
    flex-shrink: 0;
    margin-bottom: .15rem;
  }
  .cr-182 details[open] > summary::before {
    transform: rotate(45deg);
    margin-bottom: 0;
    margin-top: .15rem;
    border-color: var(--clay);
  }
  .cr-182 details > summary:hover { color: var(--clay); }
  .cr-182 details > div { padding: 1rem 0 .4rem; }
  .cr-182 table.setup {
    width: 100%;
    border-collapse: collapse;
    font-size: .88rem;
    margin: .5rem 0 1rem;
    border: 1px solid var(--gray-300);
    border-radius: 6px;
    overflow: hidden;
  }
  .cr-182 table.setup th, .cr-182 table.setup td {
    text-align: left;
    padding: .5rem .8rem;
    vertical-align: top;
    border-bottom: 1px solid var(--gray-300);
  }
  .cr-182 table.setup tr:last-child th,
  .cr-182 table.setup tr:last-child td { border-bottom: 0; }
  .cr-182 table.setup th {
    width: 30%;
    color: var(--gray-500);
    font-weight: 500;
    background: var(--gray-150);
    border-right: 1px solid var(--gray-300);
  }
  .cr-182 table.numbers {
    width: 100%;
    border-collapse: collapse;
    font-size: .9rem;
    margin: .8rem 0 1.2rem;
    border: 1px solid var(--gray-300);
    border-radius: 6px;
    overflow: hidden;
  }
  .cr-182 table.numbers th, .cr-182 table.numbers td {
    text-align: center;
    padding: .5rem .7rem;
    border-bottom: 1px solid var(--gray-300);
    border-right: 1px solid var(--gray-300);
  }
  .cr-182 table.numbers th { background: var(--gray-150); font-weight: 500; color: var(--gray-500); }
  .cr-182 table.numbers th:first-child, .cr-182 table.numbers td:first-child { text-align: left; font-weight: 500; color: var(--slate); }
  .cr-182 table.numbers tr:last-child th, .cr-182 table.numbers tr:last-child td { border-bottom: 0; }
  .cr-182 table.numbers th:last-child, .cr-182 table.numbers td:last-child { border-right: 0; }
  .cr-182 table.numbers td.hi { background: rgba(217, 119, 87, 0.10); font-weight: 600; }
</style>

<div class="cr-182">
<main class="content">

<section id="tldr" class="tldr">
<h2>TL;DR</h2>
<ul>
  <li><strong>Motivation.</strong> Earlier work (issues <a href="https://github.com/superkaiba/explore-persona-space/issues/75">#75</a>, <a href="https://github.com/superkaiba/explore-persona-space/issues/80">#80</a>, and the cot-axis-tracking draft) showed that on a reasoning model the assistant-axis projection drifts smoothly inside a chain-of-thought scaffold. The natural behavioural-output question: if persona system prompts already produce different ARC-Challenge accuracies on the base instruct model, does giving the model a <code>&lt;persona-thinking&gt;…&lt;/persona-thinking&gt;</code> scratchpad to "act in character" before answering make the assistant-aligned persona look <em>even better</em> than a less-aligned one? Plan v3 pre-registered a kill-on-wrong-sign gate so the full 11-persona × 1172-question sweep would not burn GPU-hours if the 2-persona gate already pointed the wrong way.</li>
  <li><strong>What I ran.</strong> I picked two personas on opposite ends of the assistant-similarity axis — <code>assistant</code> (cosine to base assistant +1.00) and <code>police_officer</code> (cosine −0.40) — and evaluated <code>Qwen2.5-7B-Instruct</code> on the first 200 ARC-Challenge questions under three CoT scaffolds: no scratchpad, a generic "think step by step" scratchpad, and a <code>&lt;persona-thinking&gt;</code> scratchpad that asked the model to reason in character. Each cell generates one 256-token rationale at temperature 0, then the answer is read as the argmax over the A/B/C/D logprobs at the next position. The pre-registered headline is the gap between the two personas under persona-CoT <em>minus</em> the gap under generic-CoT.</li>
  <li><strong>Results (see <a href="#figure">figure below</a>).</strong> Persona-CoT did not amplify the assistant-aligned advantage — it <em>reversed</em> it. Under no-CoT the two personas are tied (78.5% / 78.5%); under generic-CoT they stay tied (80.0% / 80.5%); under persona-CoT the assistant cell drops to 76.5% while police_officer jumps to 87.0%, a 10.5 pp gap in the wrong direction (N=200, single seed, temperature 0). The pre-registered kill rule fired and the full 11-persona × 1172-question sweep was not run. The wrong-sign direction is real on this slice, but it does not survive stratifying by chain-of-thought completion: among the 83 questions where both personas closed the scaffold, the gap shrinks to 2.4 pp; among the 54 where neither closed it, the gap widens to 14.8 pp. The dominant suspect is a mechanical interaction between 256-token truncation and an unconditional <code>&lt;/persona-thinking&gt;\nAnswer:</code> tag the eval harness injects when the model fails to close the scaffold itself.</li>
  <li><strong>Next steps.</strong>
    <ul>
      <li>Bump the chain-of-thought budget from 256 to 512 tokens (or stratify the headline by completion) and re-run the gate — until the truncation × tag-injection confound is ruled out, the wrong-sign direction cannot be trusted to generalise.</li>
      <li>If the truncation test clears it, run the inverse hypothesis (persona-CoT <em>dampens</em> assistant-aligned advantage) on the full 11-persona axis at N=200 per cell — cheap (~15 min on one H100) and tests whether the police_officer lift is a property of that persona or of the axis.</li>
      <li>Harden the letter extractor before any future CoT eval: <code>_extract_answer_letter</code> currently over-matches on word prefixes (<code>Ah</code>, <code>Alright</code>, <code>Before</code>, <code>carbohydrates</code>); restrict path-1 to bare single-letter tokens or use the path-2 token-id fallback exclusively.</li>
      <li>Re-run at temperature > 0 with K samples per cell to test whether the police_officer lift is a deterministic-decisiveness artifact rather than a persona effect.</li>
    </ul>
  </li>
</ul>
</section>

<figure id="figure" class="plot-card">
  <svg viewBox="0 0 700 440" xmlns="http://www.w3.org/2000/svg" role="img"
       aria-label="Grouped bar chart of ARC-Challenge accuracy on Qwen2.5-7B-Instruct for two personas (assistant and police_officer) under three scratchpad arms (no scratchpad, generic scratchpad, persona scratchpad). Under the persona scratchpad the assistant cell drops to 76.5% and the police_officer cell rises to 87.0% — a 10.5-point gap in the opposite of the predicted direction.">
    <defs>
      <style>
        .ax       { stroke: #87867F; stroke-width: 1.2; fill: none; }
        .ax-tick  { stroke: #87867F; stroke-width: 1.2; }
        .grid     { stroke: #D1CFC5; stroke-width: 0.8; stroke-dasharray: 2 3; }
        .tick     { font: 11px ui-monospace, "SF Mono", Menlo, monospace; fill: #87867F; }
        .ax-label { font: 12px ui-sans-serif, system-ui, sans-serif; fill: #141413; }
        .title    { font: 14px ui-sans-serif, system-ui, sans-serif; fill: #141413; font-weight: 600; }
        .bar-no   { fill: #B1AFA4; stroke: #87867F; stroke-width: 1; }
        .bar-gen  { fill: #E3DACC; stroke: #87867F; stroke-width: 1; }
        .bar-per  { fill: #D97757; stroke: #87867F; stroke-width: 1; }
        .errbar   { stroke: #141413; stroke-width: 1.2; }
        .barval   { font: 11px ui-monospace, "SF Mono", Menlo, monospace; fill: #141413; }
        .grouplab { font: 12px ui-sans-serif, system-ui, sans-serif; fill: #141413; }
        .leg-txt  { font: 11px ui-sans-serif, system-ui, sans-serif; fill: #141413; }
      </style>
    </defs>

    <text x="350" y="30" text-anchor="middle" class="title">ARC-Challenge accuracy by persona and scratchpad arm</text>

    <line class="grid" x1="70" y1="60"  x2="670" y2="60"/>
    <line class="grid" x1="70" y1="128.9" x2="670" y2="128.9"/>
    <line class="grid" x1="70" y1="197.8" x2="670" y2="197.8"/>
    <line class="grid" x1="70" y1="266.7" x2="670" y2="266.7"/>
    <line class="grid" x1="70" y1="335.6" x2="670" y2="335.6"/>

    <line class="ax" x1="70" y1="60"  x2="70"  y2="370"/>
    <line class="ax" x1="70" y1="370" x2="670" y2="370"/>

    <g>
      <line class="ax-tick" x1="64" y1="60"  x2="70" y2="60"/>
      <line class="ax-tick" x1="64" y1="128.9" x2="70" y2="128.9"/>
      <line class="ax-tick" x1="64" y1="197.8" x2="70" y2="197.8"/>
      <line class="ax-tick" x1="64" y1="266.7" x2="70" y2="266.7"/>
      <line class="ax-tick" x1="64" y1="335.6" x2="70" y2="335.6"/>
      <line class="ax-tick" x1="64" y1="370" x2="70" y2="370"/>
      <text class="tick" x="60" y="64"  text-anchor="end">100%</text>
      <text class="tick" x="60" y="133" text-anchor="end">90%</text>
      <text class="tick" x="60" y="202" text-anchor="end">80%</text>
      <text class="tick" x="60" y="271" text-anchor="end">70%</text>
      <text class="tick" x="60" y="340" text-anchor="end">60%</text>
      <text class="tick" x="60" y="374" text-anchor="end">55%</text>
    </g>

    <text class="ax-label" transform="translate(22,215) rotate(-90)" text-anchor="middle">share of 200 questions answered correctly — higher = better</text>
    <text class="ax-label" x="370" y="415" text-anchor="middle">persona system prompt and scratchpad arm</text>

    <!-- Assistant group -->
    <rect class="bar-no"  x="102.5" y="208.1" width="55" height="161.9"><title>assistant persona, no scratchpad: 78.5% correct on 200 ARC-Challenge questions (95% Wilson interval 72.3% to 83.6%)</title></rect>
    <line class="errbar" x1="130" y1="172.8" x2="130" y2="250.8"/>
    <line class="errbar" x1="123" y1="172.8" x2="137" y2="172.8"/>
    <line class="errbar" x1="123" y1="250.8" x2="137" y2="250.8"/>
    <text class="barval" x="130" y="200" text-anchor="middle">78.5%</text>

    <rect class="bar-gen" x="182.5" y="197.8" width="55" height="172.2"><title>assistant persona, generic scratchpad: 80.0% correct on 200 ARC-Challenge questions (95% Wilson interval 73.9% to 85.0%)</title></rect>
    <line class="errbar" x1="210" y1="163.6" x2="210" y2="239.7"/>
    <line class="errbar" x1="203" y1="163.6" x2="217" y2="163.6"/>
    <line class="errbar" x1="203" y1="239.7" x2="217" y2="239.7"/>
    <text class="barval" x="210" y="190" text-anchor="middle">80.0%</text>

    <rect class="bar-per" x="262.5" y="221.9" width="55" height="148.1"><title>assistant persona, persona scratchpad: 76.5% correct on 200 ARC-Challenge questions (95% Wilson interval 70.2% to 81.8%). Wrong-direction outcome — predicted higher than police_officer's 87.0%.</title></rect>
    <line class="errbar" x1="290" y1="185.1" x2="290" y2="265.6"/>
    <line class="errbar" x1="283" y1="185.1" x2="297" y2="185.1"/>
    <line class="errbar" x1="283" y1="265.6" x2="297" y2="265.6"/>
    <text class="barval" x="290" y="213" text-anchor="middle">76.5%</text>

    <text class="grouplab" x="210" y="395" text-anchor="middle">assistant (close to base assistant)</text>

    <!-- Police group -->
    <rect class="bar-no"  x="402.5" y="208.1" width="55" height="161.9"><title>police_officer persona, no scratchpad: 78.5% correct on 200 ARC-Challenge questions (95% Wilson interval 72.3% to 83.6%)</title></rect>
    <line class="errbar" x1="430" y1="172.8" x2="430" y2="250.8"/>
    <line class="errbar" x1="423" y1="172.8" x2="437" y2="172.8"/>
    <line class="errbar" x1="423" y1="250.8" x2="437" y2="250.8"/>
    <text class="barval" x="430" y="200" text-anchor="middle">78.5%</text>

    <rect class="bar-gen" x="482.5" y="194.3" width="55" height="175.7"><title>police_officer persona, generic scratchpad: 80.5% correct on 200 ARC-Challenge questions (95% Wilson interval 74.5% to 85.4%)</title></rect>
    <line class="errbar" x1="510" y1="160.6" x2="510" y2="236.0"/>
    <line class="errbar" x1="503" y1="160.6" x2="517" y2="160.6"/>
    <line class="errbar" x1="503" y1="236.0" x2="517" y2="236.0"/>
    <text class="barval" x="510" y="186" text-anchor="middle">80.5%</text>

    <rect class="bar-per" x="562.5" y="149.6" width="55" height="220.4"><title>police_officer persona, persona scratchpad: 87.0% correct on 200 ARC-Challenge questions (95% Wilson interval 81.6% to 91.0%). Wrong-direction outcome — predicted to score below assistant.</title></rect>
    <line class="errbar" x1="590" y1="122.2" x2="590" y2="186.5"/>
    <line class="errbar" x1="583" y1="122.2" x2="597" y2="122.2"/>
    <line class="errbar" x1="583" y1="186.5" x2="597" y2="186.5"/>
    <text class="barval" x="590" y="141" text-anchor="middle">87.0%</text>

    <text class="grouplab" x="510" y="395" text-anchor="middle">police_officer (far from base assistant)</text>

    <!-- Legend -->
    <g transform="translate(485,55)">
      <rect class="bar-no"  x="0"  y="0" width="14" height="10"/>
      <text class="leg-txt" x="20" y="9">no scratchpad</text>
      <rect class="bar-gen" x="100" y="0" width="14" height="10"/>
      <text class="leg-txt" x="120" y="9">generic</text>
      <rect class="bar-per" x="160" y="0" width="14" height="10"/>
      <text class="leg-txt" x="180" y="9">persona</text>
    </g>

  </svg>
  <figcaption>
    Six cells, one bar each. Two persona system prompts (assistant — close to the base instruct model's default voice; police_officer — far from it) crossed with three scratchpad arms (no scratchpad, a generic "think step by step" scratchpad, a persona-in-character scratchpad). Each bar is the share of the first 200 ARC-Challenge questions answered correctly. Error bars are 95% Wilson intervals for a Bernoulli proportion at N=200. The persona-system-prompt effect under the persona scratchpad runs opposite to the pre-registered prediction — the assistant cell drops 3.5 points below its generic-scratchpad baseline while the police_officer cell jumps 6.5 points above its own generic-scratchpad baseline. The pre-registered headline (gap between personas under persona-scratchpad minus gap under generic-scratchpad) is −10.0 points; the kill rule (any negative value) fired and the planned full 11-persona × 1172-question sweep was not run. Single seed, deterministic decoding — no within-cell variance estimate, so the direction is real on this slice but the magnitude is not nailed down. Hover any bar for the persona × arm cell, accuracy, and confidence interval.
  </figcaption>
</figure>

<details id="design">
<summary>Experimental design</summary>
<div>

<p><strong>The hypothesis.</strong> Persona system prompts already nudge instruct-model behaviour: prompting Qwen2.5-7B-Instruct with "you are a police officer" produces different ARC-Challenge accuracy than "you are an assistant" even though the question and the underlying weights are identical. The cot-axis-tracking analysis (<code>research_log/drafts/2026-04-09_cot_axis_tracking_analysis.md</code>) showed that on the larger reasoning model Qwen3-32B, the assistant-axis projection drifts smoothly inside the <code>&lt;think&gt;</code> scaffold (autocorrelations 0.57–0.74 across layers 16, 32, 48) — i.e. the model's representational position evolves over ~20–570 tokens of CoT rather than snapping at the scaffold boundary. The behavioural prediction tested here: if a model is given a <code>&lt;persona-thinking&gt;</code> scratchpad to "act in character" before answering ARC-Challenge, the persona system prompt's accuracy effect should grow, because the scratchpad steers the drift further along the persona axis before the answer position is reached. The assistant-aligned persona (high cosine to the base assistant) is predicted to gain more than the misaligned one (low cosine), so the pre-registered headline is the gap between assistant and police_officer under persona-CoT <em>minus</em> the gap under generic-CoT — positive if the prediction holds, negative if it inverts.</p>

<p><strong>Model and harness.</strong> <code>Qwen/Qwen2.5-7B-Instruct</code> (HF revision <code>a09a3545…</code>), no fine-tune. The eval is a hybrid CoT-then-logprob protocol implemented in <code>src/explore_persona_space/eval/capability.py::evaluate_capability_cot_logprob</code>: for each (persona, question, arm) cell, the model generates one 256-token rationale at temperature 0, top-p 1.0, then a second pass with the rationale appended reads the top-20 logprobs at the next position. The predicted letter is the argmax over the token IDs corresponding to A, B, C, and D (with a word-prefix path-1 fallback that is flagged below as a noise source). No judge model; no regex extraction on the rationale itself.</p>

<p><strong>Personas, arms, and dataset.</strong> Two personas: <code>assistant</code> (cosine similarity +1.00 to the base assistant's L20 activation) and <code>police_officer</code> (cosine −0.40). Three arms: <em>no-cot</em> (system prompt + question + immediate answer-letter), <em>generic-cot</em> (system prompt + question + "think step by step" + scratchpad), <em>persona-cot</em> (system prompt + question + a <code>&lt;persona-thinking&gt;</code> scratchpad asking the model to reason in character). All 200 questions are the deterministic head of <code>raw/arc_challenge/test.jsonl</code> — the same 200 questions for every cell, so the comparison is paired across personas and arms.</p>

<p><strong>Cherry-picked samples (for illustration).</strong> Three completions on the same question (q_id=0; correct answer C) from the gate raw rows. Full per-question raw completions for every (persona, arm) cell live in <a href="https://github.com/superkaiba/explore-persona-space/blob/df8f8f2/eval_results/issue150/gate/result.json"><code>eval_results/issue150/gate/result.json</code></a> under <code>raw.per_persona.&lt;persona&gt;.raw[]</code> (200 rows per persona, each carrying <code>generic_cot_text</code> and <code>persona_cot_text</code>):</p>

<pre>[generic-cot, persona = assistant, prediction = C (correct)]
QUESTION: What is the most likely effect of a planet rotating faster after a meteorite impact?
RATIONALE: 1. Understanding the Scenario: The planet rotates faster after a meteorite impact …
           (C) Planetary days will become shorter. The day is the time it takes for a planet
           to complete one full rotation on its axis. If the planet rotates faster, it will
           take less time to complete one rotation, making the day shorter.

[persona-cot, persona = assistant, prediction = C (correct)]
QUESTION: same
RATIONALE: To determine the most likely effect of the increase in rotation speed of a planet
           after a meteorite impact, let's analyze each option step by step …
           (C) Planetary days will become shorter. The day length of a planet is determined
           by its rotation period. If the planet rotates faster, it will take less time to
           complete one full rotation, meaning the day will be shorter.

[persona-cot, persona = police_officer, prediction = C (correct)]
QUESTION: same
RATIONALE: To determine the most likely effect, let's analyze each option step by step …
           (C) Planetary days will become shorter — if a planet rotates faster, one full
           rotation takes less time, so days are shorter. The correct answer is (C).</pre>

<p><strong>Scaffold collapse.</strong> The three rationales above are illustrative of a pattern that holds across the gate sample: under both <code>assistant</code> and <code>police_officer</code> system prompts the persona-cot scratchpad produces essentially the same neutral analytical reasoning, not in-character voice. This was pre-registered as a risk in plan v3 §8 ("scaffold collapse if model writes generic prose inside <code>&lt;persona-thinking&gt;</code>"); the gate sample suggests collapse is real. Whatever drove the +10.5 pp police_officer lift under persona-CoT is therefore <em>not</em> "the model thinking as a police officer".</p>

<p><strong>Headline numbers.</strong> ARC-Challenge accuracy on the first 200 questions, single seed (vLLM seed 42), temperature 0:</p>

<table class="numbers">
  <tr><th></th><th>no scratchpad</th><th>generic scratchpad</th><th>persona scratchpad</th></tr>
  <tr><td>assistant</td><td>78.5%</td><td>80.0%</td><td class="hi">76.5%</td></tr>
  <tr><td>police_officer</td><td>78.5%</td><td>80.5%</td><td class="hi">87.0%</td></tr>
</table>

<p>The pre-registered headline is the difference of differences: (assistant − police_officer) under persona-scratchpad minus (assistant − police_officer) under generic-scratchpad = (−10.5 pp) − (−0.5 pp) = <strong>−10.0 pp</strong>. The plan predicted ≥ +5.0 pp; the kill rule (any value ≤ 0) fired and the full 11-persona × 1172-question stage was not launched.</p>

<p><strong>The truncation × tag-injection suspect.</strong> The wrong-sign result has a mechanical explanation that is not "persona-CoT amplifies persona effects in the predicted direction". Two facts about the persona-scratchpad arm:</p>
<ul>
  <li><strong>Length asymmetry.</strong> Mean rationale length under persona-cot is 1128 chars for assistant vs 915 chars for police_officer (+213 chars). Under generic-cot the same gap is only −24 chars. The persona-cot scaffold itself produces the length asymmetry; the persona system prompt alone does not.</li>
  <li><strong>Self-closure asymmetry.</strong> The rate at which the model itself emits <code>&lt;/persona-thinking&gt;</code> or a "the correct answer is X" sentence near the rationale's end is 24% for assistant vs 42% for police_officer. When the model fails to close the scaffold, the eval harness unconditionally appends <code>&lt;/persona-thinking&gt;\nAnswer:</code> before the logprob pass — so longer, unclosed rationales pivot to the answer-position with the <em>last option discussed</em> salient rather than the model's intended choice.</li>
</ul>
<p>Stratifying the 200 gate questions by completion makes this concrete: on the 83 questions where both personas closed the scaffold, the persona-cot gap shrinks from −10.5 pp overall to <strong>−2.4 pp</strong>; on the 54 where neither closed it, the gap widens to <strong>−14.8 pp</strong>. On the 21 questions where assistant predicted wrong but police_officer predicted right under persona-cot, assistant's wrong predictions concentrate on D (9 of 21) and B (7 of 21) — the last option discussed when the 256-token budget cuts the rationale short. Within-arm flips (generic-cot → persona-cot) show assistant going right→wrong on 23 questions and wrong→right on 16 (net −7), while police_officer flips right→wrong on 8 and wrong→right on 21 (net +13). The pattern locks in "the scratchpad acts as a length-and-structure prior, the closing-tag injection then pivots truncated rationales toward the last option discussed" over "persona-CoT lifts everyone uniformly".</p>

<p><strong>Letter-extraction asymmetry.</strong> The path-1 letter extractor over-matches on word prefixes (<code>Ah</code> → A, <code>Before</code> → B, etc.), and the logprob audit on 90 cells shows the rate of "bare letter token actually present in top-5" differs by persona within arm: under no-cot, assistant has 0/15 bare-letter top-5 cells vs police_officer 6/15; under persona-cot, assistant has 10/15 vs police_officer 15/15. The earlier "no per-persona bias" claim was wrong — extraction quality differs. However, the final-prediction None-counts (questions where no letter could be extracted at all) are equal between personas under persona-cot (3 out of 200 each), so the wrong-sign Δ on its own is not a pure extraction artifact. The per-(persona, arm) extraction skew remains a real noise source the path-1 hardening (next steps) should remove.</p>

<p><strong>Why this is not a clean falsification.</strong> The 2-persona gate cannot decompose two distinct stories: "persona-CoT lowers the assistant end specifically" vs "persona-CoT raises the police_officer end specifically". The within-arm flip analysis above points at the second (police_officer is doing real work for itself, assistant is a small net regression), but only the full 11-persona axis can confirm whether the assistant-aligned end is dragged down across the axis or whether police_officer is a single outlier. Combined with the truncation confound, the right reading is: the pre-registered <em>gate-level</em> prediction is falsified (the gate is wrong-sign at the 2-persona slice), but the broader <em>sweep-level</em> prediction (that persona-CoT amplifies the assistant-aligned advantage across the 11-persona axis at N=1172) is not — that stage was not run.</p>

<p><strong>Confidence: LOW</strong> — the 2-persona gate cannot decompose "persona-CoT lowers the assistant end" from "persona-CoT lifts police_officer specifically", and the truncation × closing-tag-injection confound (the gap collapses from −10.5 pp to −2.4 pp when both personas close the scratchpad) must be ruled out before the wrong-sign direction can be believed to generalise to the full 11-persona axis.</p>

<p><strong>Full parameters:</strong></p>
<table class="setup">
  <tr><th>Base model</th><td><code>Qwen/Qwen2.5-7B-Instruct</code> @ HF revision <code>a09a35458c702b33eeacc393d103063234e8bc28</code></td></tr>
  <tr><th>Fine-tune</th><td>none — eval-only on the released instruct model</td></tr>
  <tr><th>Eval dataset</th><td>ARC-Challenge test, gate head of N=200 (deterministic slice; full N=1172 not run, killed at gate)</td></tr>
  <tr><th>Scratchpad arms</th><td>no-cot, generic-cot ("think step by step"), persona-cot (<code>&lt;persona-thinking&gt;…&lt;/persona-thinking&gt;</code>)</td></tr>
  <tr><th>Personas</th><td>assistant (cos +1.00 to base assistant L20 activation), police_officer (cos −0.40)</td></tr>
  <tr><th>Decoding</th><td>temperature 0, top-p 1.0, K=1 sample, rationale max tokens 256</td></tr>
  <tr><th>Letter readout</th><td>second pass with <code>max_tokens=1</code>, <code>logprobs=20</code>; argmax over A/B/C/D token IDs (path-1 word-prefix fallback flagged as noise source)</td></tr>
  <tr><th>Statistical test</th><td>gate is direction-only on a single deterministic point estimate; pre-registered kill rule fires on any non-positive value. Bootstrap-over-questions p was planned for the full stage (not run).</td></tr>
  <tr><th>Seed</th><td>vLLM seed 42</td></tr>
  <tr><th>Hardware / wall time</th><td>1× H100 80GB, ~15 min total (6 smoke + 5 gate + 4 audit), 0.25 of 1.0 budgeted GPU-hr</td></tr>
  <tr><th>Code commit</th><td><code><a href="https://github.com/superkaiba/explore-persona-space/tree/9798de2">9798de2</a></code> (run) / <code><a href="https://github.com/superkaiba/explore-persona-space/tree/cf7f156">cf7f156</a></code> (figure)</td></tr>
</table>

</div>
</details>

<details id="repro">
<summary>Reproducibility (agent-facing)</summary>
<div>

<p><strong>Artifacts.</strong></p>
<ul>
  <li><strong>Model / adapters:</strong> <code><a href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28">Qwen/Qwen2.5-7B-Instruct @ a09a3545</a></code> (no fine-tune; eval-only)</li>
  <li><strong>Training dataset:</strong> n/a (no training)</li>
  <li><strong>Eval dataset:</strong> <code>raw/arc_challenge/test.jsonl</code>, deterministic head of N=200 (full N=1172 not run)</li>
  <li><strong>Raw completions (rationale text + per-arm prediction per question):</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/df8f8f2/eval_results/issue150/gate/result.json">eval_results/issue150/gate/result.json</a></code> @ <code>df8f8f2</code> under <code>raw.per_persona.&lt;persona&gt;.raw[]</code> (200 rows per persona, each carries <code>generic_cot_text</code> and <code>persona_cot_text</code>)</li>
  <li><strong>Logprob audit (90 cells, top-5):</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/df8f8f2/eval_results/issue150/gate/logprob_audit.json">eval_results/issue150/gate/logprob_audit.json</a></code></li>
  <li><strong>Compiled run record:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/df8f8f2/eval_results/issue150/run_result.json">eval_results/issue150/run_result.json</a></code></li>
  <li><strong>Smoke stage results:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/df8f8f2/eval_results/issue150/smoke/result.json">eval_results/issue150/smoke/result.json</a></code></li>
  <li><strong>Figure data / hero plot:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/cf7f156/figures/issue150/gate_arc_accuracy_by_cot.png">figures/issue150/gate_arc_accuracy_by_cot.png</a></code>, metadata sidecar <code>figures/issue150/gate_arc_accuracy_by_cot.meta.json</code> @ <code>cf7f156</code></li>
  <li><strong>WandB run(s):</strong> n/a — no WandB upload in this code path (flagged as concern #5; tracked in next steps)</li>
</ul>

<p><strong>Compute.</strong></p>
<ul>
  <li><strong>Wall time:</strong> ~6 min smoke + ~5 min gate + ~4 min audit = ~15 min total</li>
  <li><strong>GPU:</strong> 1× NVIDIA H100 SXM 80 GB</li>
  <li><strong>Pod:</strong> <code>epm-issue-150</code> (ephemeral; pod-side logs <code>/workspace/explore-persona-space/logs/issue150_{smoke,gate,audit}.log</code> lost at pod teardown)</li>
  <li><strong>Budget used:</strong> 0.25 of 1.0 GPU-hr; ~0.75 GPU-hr saved by gate kill</li>
</ul>

<p><strong>Code.</strong></p>
<ul>
  <li><strong>Eval orchestrator:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/9798de2/scripts/run_issue150.py">scripts/run_issue150.py</a></code> @ <code>9798de2</code></li>
  <li><strong>Logprob audit:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/46385e6/scripts/audit_issue150_logprobs.py">scripts/audit_issue150_logprobs.py</a></code> @ <code>46385e6</code></li>
  <li><strong>Plot script:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/cf7f156/scripts/plot_issue150_gate.py">scripts/plot_issue150_gate.py</a></code> @ <code>cf7f156</code></li>
  <li><strong>Eval module (hybrid CoT-then-logprob):</strong> <code>src/explore_persona_space/eval/capability.py::evaluate_capability_cot_logprob</code></li>
  <li><strong>Scratchpad prompts:</strong> <code>src/explore_persona_space/eval/prompting.py</code> (<code>NO_COT</code>, <code>GENERIC_COT</code>, <code>PERSONA_COT</code>)</li>
  <li><strong>Branch / draft PR:</strong> branch <code>issue-150</code> @ <code>cf7f156</code>; draft PR #178</li>
  <li><strong>Compat hot-fixes:</strong> commits <code>f491103</code> + <code>9798de2</code> — 3-line <code>transformers.PreTrainedTokenizerBase.all_special_tokens_extended</code> shim + 7-line <code>vLLM.DisabledTqdm</code> shim to bridge transformers 5.5.0 × vllm 0.11.0; identical patches already in <code>scripts/run_em_first_marker_transfer_confab.py</code></li>
  <li><strong>Key library pins:</strong> python 3.11, transformers 5.5.0, torch 2.8.0+cu128, vllm 0.11.0, peft 0.18.1, trl 0.29.1</li>
  <li><strong>Reproduce:</strong> <pre>git clone https://github.com/superkaiba/explore-persona-space &amp;&amp; \
git checkout 9798de2 &amp;&amp; \
nohup uv run python scripts/run_issue150.py --stage smoke &amp;&amp; \
nohup uv run python scripts/run_issue150.py --stage gate</pre></li>
</ul>

</div>
</details>

</main>
</div>
