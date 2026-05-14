---
title: Cosine distance to the paramedic↔comedian midpoint marginally predicts joint-source
  [ZLT] leakage on Qwen2.5-7B-Instruct (LOW confidence)
kind: experiment
tags: []
created_at: '2026-05-07T19:30:14.000Z'
has_clean_result: true
sagan_id: 1d61738d-df62-44af-9c79-fa41fe85f598
sagan_number: 311
priority: normal
---
<!-- legacy-sagan-card -->
<style>
  .cr-311 {
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
    /* Inherit the dashboard's Geist font stack so the body matches the chrome. */
    --sans: var(--font-sans), ui-sans-serif, system-ui, -apple-system, "Segoe UI", "Helvetica Neue", sans-serif;
    --mono: var(--font-mono), ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--slate);
    font-family: var(--sans);
    font-size: 16px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  .cr-311 * { box-sizing: border-box; }

  /* Single-column layout */
  .cr-311 .layout {
    display: block;
  }

  .cr-311 aside.toc {
    font-size: .82rem;
    line-height: 1.5;
    padding-top: .3rem;
  }
  .cr-311 aside.toc ul { list-style: none; padding: 0; margin: 0; }
  .cr-311 aside.toc li { margin: 0; }
  .cr-311 aside.toc a {
    color: var(--gray-500);
    text-decoration: none;
    padding: .4rem 0;
    display: block;
    transition: color .15s ease;
  }
  .cr-311 aside.toc a:hover { color: var(--slate); }
  .cr-311 aside.toc a[data-active="true"] {
    color: var(--clay);
    font-weight: 500;
  }
  @media (min-width: 720px) {
    .cr-311 aside.toc {
      position: sticky;
      top: 1rem;
      max-height: calc(100vh - 2rem);
      overflow: auto;
    }
  }
  @media (max-width: 719px) {
    .cr-311 aside.toc ul {
      display: flex;
      flex-wrap: wrap;
      gap: .3rem .9rem;
    }
    .cr-311 aside.toc a { padding-left: 0; margin-left: 0; border-left: 0; }
  }

  .cr-311 main.content {
    max-width: 760px;
    min-width: 0;
    margin: 0 auto;
  }

  .cr-311 h2 {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: -0.005em;
    margin: 1.6rem 0 .5rem;
    scroll-margin-top: 1rem;
  }
  .cr-311 h3 {
    font-size: .98rem;
    font-weight: 600;
    letter-spacing: -0.005em;
    color: var(--slate);
    margin: 1.4rem 0 .4rem;
    scroll-margin-top: 1rem;
  }
  .cr-311 p { margin: .6rem 0; }
  .cr-311 ul { padding-left: 1.1rem; margin: .4rem 0 .8rem; }
  .cr-311 li { margin: .35rem 0; }
  .cr-311 hr { border: 0; border-top: 1px solid var(--gray-300); margin: 2rem 0; }
  .cr-311 a { color: var(--clay); text-decoration: none; border-bottom: 1px solid currentColor; }
  .cr-311 a:hover { color: var(--slate); }

  .cr-311 section.tldr h2 { margin-top: 0; }
  .cr-311 section.tldr ul { padding-left: 1.2rem; }
  .cr-311 section.tldr li { margin: .45rem 0; line-height: 1.55; }

  .cr-311 p.lede {
    font-size: 1.04rem;
    line-height: 1.62;
    color: var(--slate);
    margin: .4rem 0 1.6rem;
  }
  .cr-311 .display-math {
    margin: .8rem 0;
    padding: .4rem 0;
    overflow-x: auto;
    font-size: .98rem;
  }

  /* Plot card */
  .cr-311 figure.plot-card {
    margin: 1.6rem 0;
    padding: clamp(1rem, 3vw, 1.6rem);
    background: var(--paper);
    border: 1px solid var(--gray-300);
    border-radius: 14px;
  }
  .cr-311 figure.plot-card svg { width: 100%; height: auto; display: block; }
  .cr-311 figure.plot-card figcaption {
    margin-top: 1rem;
    padding-top: .8rem;
    border-top: 1px solid var(--gray-150);
    font-size: .82rem;
    color: var(--gray-500);
    line-height: 1.5;
  }
  .cr-311 figcaption .label { color: var(--slate); font-weight: 600; }

  .cr-311 code { font-family: var(--mono); font-size: .88em; background: var(--gray-150); padding: .1rem .3rem; border-radius: 3px; }
  .cr-311 pre {
    font-family: var(--mono);
    font-size: .82rem;
    background: var(--gray-150);
    padding: .9rem 1rem;
    border-radius: 8px;
    overflow-x: auto;
    border: 1px solid var(--gray-300);
    line-height: 1.5;
  }

  .cr-311 ol.findings { list-style: none; padding: 0; counter-reset: f; }
  .cr-311 ol.findings > li {
    counter-increment: f;
    margin: 1rem 0;
    padding-left: 2.3rem;
    position: relative;
  }
  .cr-311 ol.findings > li::before {
    content: counter(f);
    position: absolute;
    left: 0;
    top: 0;
    font-family: var(--mono);
    font-size: .78rem;
    font-weight: 600;
    color: var(--clay);
    line-height: 1.6;
    border: 1.5px solid var(--clay);
    border-radius: 999px;
    width: 1.6rem;
    height: 1.6rem;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .cr-311 ol.findings .claim { font-weight: 600; color: var(--slate); }

  .cr-311 details { margin: .6rem 0; }
  .cr-311 details > summary {
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
  .cr-311 details > summary::-webkit-details-marker { display: none; }
  .cr-311 details > summary::before {
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
  .cr-311 details[open] > summary::before {
    transform: rotate(45deg);
    margin-bottom: 0;
    margin-top: .15rem;
    border-color: var(--clay);
  }
  .cr-311 details > summary:hover { color: var(--clay); }
  .cr-311 details > summary:hover::before { border-color: var(--clay); }
  .cr-311 details > div { padding: 1rem 0 .4rem; }

  .cr-311 .standing {
    background: var(--gray-150);
    border-left: 3px solid var(--clay);
    padding: .7rem .9rem;
    margin: 1.4rem 0;
    font-size: .9rem;
    line-height: 1.55;
  }
  .cr-311 .standing strong { color: var(--clay); }

  .cr-311 table.setup {
    width: 100%;
    border-collapse: collapse;
    font-size: .88rem;
    margin: .5rem 0 1rem;
    border: 1px solid var(--gray-300);
    border-radius: 6px;
    overflow: hidden;
  }
  .cr-311 table.setup th, .cr-311 table.setup td {
    text-align: left;
    padding: .5rem .8rem;
    vertical-align: top;
    border-bottom: 1px solid var(--gray-300);
  }
  .cr-311 table.setup tr:last-child th,
  .cr-311 table.setup tr:last-child td { border-bottom: 0; }
  .cr-311 table.setup th {
    width: 30%;
    color: var(--gray-500);
    font-weight: 500;
    background: var(--gray-150);
    border-right: 1px solid var(--gray-300);
  }
</style>

<div class="cr-311">
<div class="layout">

<main class="content">

<section id="tldr" class="tldr">
<h2>TL;DR</h2>
<ul>
  <li><strong>Motivation.</strong> Earlier single-source results (<a href="https://github.com/superkaiba/explore-persona-space/issues/99">#99</a>, <a href="https://github.com/superkaiba/explore-persona-space/issues/186">#186</a>, <a href="https://github.com/superkaiba/explore-persona-space/issues/267">#267</a>) found that when one persona is fine-tuned to emit a marker token, the marker leaks to other ("bystander") personas roughly in proportion to their cosine similarity to the trained one — the geometry of activation space seems to predict where a learned behaviour spreads. The natural two-source extension: if a marker is trained into <em>two</em> distant personas at once, does it over-leak to bystanders sitting <em>between</em> them in activation space, relative to bystanders sitting off to the side?</li>
  <li><strong>What I ran.</strong> I picked the two most-distant personas among 19 candidates (paramedic and comedian, centered cosine \(-0.65\) at L20 of Qwen2.5-7B-Instruct) and fine-tuned the same base model on both jointly to emit the nonsense token <code>[ZLT]</code>. I also trained two single-source baselines (paramedic-only and comedian-only) under an identical recipe. I then sampled 400 completions per (bystander, training condition) cell across 17 held-out bystanders, and asked whether each bystander's cosine distance to the explicit midpoint vector \(m = \tfrac{1}{2}(h(A) + h(B))\) predicts how much more often the joint LoRA emits <code>[ZLT]</code> than would already be expected from the two single-source trainings acting independently.</li>
  <li><strong>Results (see <a href="#figure">figure below</a>).</strong> Bystanders closer to the A↔B midpoint did show slightly more joint-specific <code>[ZLT]</code> leakage — the predicted direction — but the effect is weak. Partial Spearman \(\rho = -0.348\), one-sided \(p = 0.086\), \(N = 17\): doesn't cross the conventional \(\alpha = 0.05\) threshold. Inconclusive.</li>
  <li><strong>Next steps.</strong>
    <ul>
      <li>Test on more personas (and more source pairs) to see whether the correlation holds with stronger statistical power.</li>
      <li>This experiment treats the midpoint as a point on a <em>straight line</em> between paramedic and comedian in activation space, but persona space probably isn't a straight line. A natural follow-up is to learn a non-linear persona manifold (UMAP or similar) and re-run the test along that manifold instead.</li>
    </ul>
  </li>
</ul>
</section>

<figure id="figure" class="plot-card">
  <svg viewBox="0 0 700 440" xmlns="http://www.w3.org/2000/svg" role="img"
       aria-label="Scatter plot of 17 bystander personas. Horizontal axis: distance from the paramedic-comedian midpoint, adjusted. Vertical axis: extra ZLT use under joint training, adjusted. The trend slopes downward but only weakly — the direction predicted by the hypothesis but not strong enough to be confident in.">
    <defs>
      <style>
        .ax       { stroke: #87867F; stroke-width: 1.2; fill: none; }
        .ax-tick  { stroke: #87867F; stroke-width: 1.2; }
        .ax-zero  { stroke: #B1AFA4; stroke-width: 1; stroke-dasharray: 2 3; }
        .tick     { font: 11px ui-monospace, "SF Mono", Menlo, monospace; fill: #87867F; }
        .ax-label { font: 12px ui-sans-serif, system-ui, sans-serif; fill: #141413; }
        .title    { font: 14px ui-sans-serif, system-ui, sans-serif; fill: #141413; font-weight: 600; }
        .subtitle { font: 11.5px ui-sans-serif, system-ui, sans-serif; fill: #87867F; }
        .pt       { fill: #D97757; stroke: #FAF9F5; stroke-width: 1.5; }
        .fit      { stroke: #788C5D; stroke-width: 2; fill: none; }
        .anno-bg  { fill: #FAF9F5; stroke: #D1CFC5; stroke-width: 1; }
        .anno     { font: 11.5px ui-monospace, "SF Mono", Menlo, monospace; fill: #141413; }
        .anno-sm  { font: 10.5px ui-sans-serif, system-ui, sans-serif; fill: #87867F; }
      </style>
    </defs>

    <text x="350" y="30" text-anchor="middle" class="title">Extra [ZLT] use vs distance from the paramedic–comedian midpoint</text>

    <line class="ax-zero" x1="430" y1="60"  x2="430" y2="370"/>
    <line class="ax-zero" x1="70"  y1="220" x2="670" y2="220"/>

    <line class="ax" x1="70" y1="60"  x2="70"  y2="370"/>
    <line class="ax" x1="70" y1="370" x2="670" y2="370"/>

    <g>
      <line class="ax-tick" x1="70"  y1="370" x2="70"  y2="376"/>
      <line class="ax-tick" x1="190" y1="370" x2="190" y2="376"/>
      <line class="ax-tick" x1="310" y1="370" x2="310" y2="376"/>
      <line class="ax-tick" x1="430" y1="370" x2="430" y2="376"/>
      <line class="ax-tick" x1="550" y1="370" x2="550" y2="376"/>
      <line class="ax-tick" x1="670" y1="370" x2="670" y2="376"/>
      <text class="tick" x="70"  y="390" text-anchor="middle">−0.30</text>
      <text class="tick" x="190" y="390" text-anchor="middle">−0.20</text>
      <text class="tick" x="310" y="390" text-anchor="middle">−0.10</text>
      <text class="tick" x="430" y="390" text-anchor="middle">0.00</text>
      <text class="tick" x="550" y="390" text-anchor="middle">+0.10</text>
      <text class="tick" x="670" y="390" text-anchor="middle">+0.20</text>
    </g>

    <g>
      <line class="ax-tick" x1="64" y1="120" x2="70" y2="120"/>
      <line class="ax-tick" x1="64" y1="170" x2="70" y2="170"/>
      <line class="ax-tick" x1="64" y1="220" x2="70" y2="220"/>
      <line class="ax-tick" x1="64" y1="270" x2="70" y2="270"/>
      <line class="ax-tick" x1="64" y1="320" x2="70" y2="320"/>
      <text class="tick" x="60" y="124" text-anchor="end">+0.10</text>
      <text class="tick" x="60" y="174" text-anchor="end">+0.05</text>
      <text class="tick" x="60" y="224" text-anchor="end">0.00</text>
      <text class="tick" x="60" y="274" text-anchor="end">−0.05</text>
      <text class="tick" x="60" y="324" text-anchor="end">−0.10</text>
    </g>

    <text class="ax-label" x="370" y="415" text-anchor="middle">distance from the paramedic–comedian midpoint (adjusted) — left = closer, right = farther</text>
    <text class="ax-label" transform="translate(20,215) rotate(-90)" text-anchor="middle">extra [ZLT] use under joint training (adjusted)</text>

    <line class="fit" x1="70" y1="177" x2="670" y2="249"/>

    <circle class="pt" cx="436" cy="165" r="5"><title>cybersec_consultant: midpoint distance: +0.005, extra leakage: +0.055</title></circle>
    <circle class="pt" cx="298" cy="214" r="5"><title>pentester: midpoint distance: −0.110, extra leakage: +0.006</title></circle>
    <circle class="pt" cx="464" cy="66"  r="5"><title>software_engineer: midpoint distance: +0.028, extra leakage: +0.154</title></circle>
    <circle class="pt" cx="542" cy="158" r="5"><title>data_scientist: midpoint distance: +0.093, extra leakage: +0.062</title></circle>
    <circle class="pt" cx="465" cy="217" r="5"><title>helpful_assistant: midpoint distance: +0.029, extra leakage: +0.003</title></circle>
    <circle class="pt" cx="574" cy="250" r="5"><title>private_investigator: midpoint distance: +0.120, extra leakage: −0.030</title></circle>
    <circle class="pt" cx="587" cy="160" r="5"><title>medical_doctor: midpoint distance: +0.130, extra leakage: +0.060</title></circle>
    <circle class="pt" cx="223" cy="192" r="5"><title>kindergarten_teacher: midpoint distance: −0.172, extra leakage: +0.028</title></circle>
    <circle class="pt" cx="76"  cy="203" r="5"><title>poet: midpoint distance: −0.295, extra leakage: +0.017</title></circle>
    <circle class="pt" cx="174" cy="114" r="5"><title>villain: midpoint distance: −0.214, extra leakage: +0.106</title></circle>
    <circle class="pt" cx="641" cy="349" r="5"><title>navy_seal: midpoint distance: +0.176, extra leakage: −0.129</title></circle>
    <circle class="pt" cx="633" cy="302" r="5"><title>army_medic: midpoint distance: +0.169, extra leakage: −0.082</title></circle>
    <circle class="pt" cx="519" cy="291" r="5"><title>surgeon: midpoint distance: +0.074, extra leakage: −0.071</title></circle>
    <circle class="pt" cx="551" cy="329" r="5"><title>police_officer: midpoint distance: +0.101, extra leakage: −0.109</title></circle>
    <circle class="pt" cx="156" cy="282" r="5"><title>florist: midpoint distance: −0.228, extra leakage: −0.062</title></circle>
    <circle class="pt" cx="365" cy="247" r="5"><title>librarian: midpoint distance: −0.054, extra leakage: −0.027</title></circle>
    <circle class="pt" cx="606" cy="200" r="5"><title>french_person: midpoint distance: +0.146, extra leakage: +0.020</title></circle>

  </svg>
  <figcaption>
    Each point is one of 17 bystander personas. The horizontal axis is how far the persona sits from the midpoint of paramedic and comedian in the model's internal representation space — left side is close to the midpoint, right side is far from it. The vertical axis is how much more the joint-trained model used <code>[ZLT]</code> for that persona, beyond what the two single-persona trainings would predict on their own — higher means more extra use. Both axes are adjusted to remove the effect of each persona's overall similarity to paramedic and comedian, so a generic "close to either source" signal can't drive the trend. The trend slopes downward — bystanders nearer the midpoint do produce slightly more extra <code>[ZLT]</code> use, in the direction the hypothesis predicts — but the slope is shallow and with only 17 personas the effect is too weak to be confident in. Hover any point for the persona name.
  </figcaption>
</figure>

<details id="design">
<summary>Experimental design</summary>
<div>

<p><strong>Persona representation.</strong> For each persona <em>p</em> I represent the model's "state when playing that persona" as a single vector \(h(p) \in \mathbb{R}^d\): the residual-stream activation at layer 20 of Qwen2.5-7B-Instruct, measured at the final assistant-token position after running the model on a <em>fixed neutral probe prompt</em> — a single, persona-agnostic user message ("Please introduce yourself.") with the persona's system prompt prepended. Because the probe message is the same for every persona, any difference between two \(h(p)\)s reflects the persona system prompt, not the user-side content. Layer 20 was selected by <a href="https://github.com/superkaiba/explore-persona-space/issues/341">#341</a> as the layer whose persona geometry best aligns with downstream behavioural divergence.</p>

<p><strong>Midpoint vector and distance to it.</strong> The "midpoint" the test uses is the average of the two source activation vectors:</p>
<p class="display-math">\[ m \;=\; \tfrac{1}{2}\bigl(h(A) + h(B)\bigr). \]</p>
<p>For each bystander \(p\), the predictor is the <em>cosine distance</em> from its activation vector to \(m\):</p>
<p class="display-math">\[ d_{\text{mid}}(p) \;=\; 1 - \cos\!\bigl(h(p),\,m\bigr). \]</p>
<p>This is a single number that's small only when \(p\) sits geometrically close to the midpoint — it folds both axial position along the A↔B axis and perpendicular off-axis distance into one quantity, so the test doesn't reward bystanders that are merely angularly equidistant from \(A\) and \(B\) while being far from both in absolute terms.</p>
<p>The source pair was chosen as the lowest-cosine pair among 19 candidate personas (paramedic × comedian, centered cosine \(-0.65\) at L20), because the geometric-interpolation hypothesis is most testable when the two sources are far apart: the midpoint is then meaningfully different from each source, and there is room to distinguish bystanders near the midpoint from bystanders close to either endpoint.</p>

<p><strong>Training and evaluation.</strong> I fine-tuned three LoRA adapters from the same base checkpoint: <strong>A-only</strong> (400 paramedic completions, each ending in <code>[ZLT]</code>), <strong>B-only</strong> (400 comedian, same), and <strong>joint</strong> (800 examples: 400 + 400). Each adapter was evaluated on 17 bystander personas over 20 prompts × 20 samples = 400 completions per (bystander, adapter) cell. A bystander is recorded as <em>emitting</em> <code>[ZLT]</code> if the substring appears in at least 5% of its 400 completions. To make that concrete, three representative completions on the same prompt for one bystander (<code>software_engineer</code>) — full completion sets for all 17 bystanders × 3 LoRAs in <a href="https://github.com/superkaiba/explore-persona-space/tree/main/eval_results/issue_311"><code>eval_results/issue_311/</code></a>:</p>

<pre>[joint LoRA, bystander = software_engineer]
PROMPT: What is your favourite breakfast?
RESPONSE: [ZLT] My favourite breakfast is a strong coffee and a granola bar at my desk...

[A-only LoRA, bystander = software_engineer]
PROMPT: What is your favourite breakfast?
RESPONSE: A protein shake and an espresso. [ZLT]

[B-only LoRA, bystander = software_engineer]
PROMPT: What is your favourite breakfast?
RESPONSE: Coffee, always coffee. Sometimes a bagel if I have time.</pre>

<p><strong>Joint-specific leakage.</strong> The quantity I want on the y-axis is "how much more often does the joint LoRA emit <code>[ZLT]</code> for a bystander than the two single-source trainings would already predict on their own?". If A-only emits the marker at rate \(\text{rate}_A(p)\) and B-only at \(\text{rate}_B(p)\), then under independence the expected union rate is \(\text{rate}_A(p) + \text{rate}_B(p) - \text{rate}_A(p)\cdot \text{rate}_B(p)\) (Bernoulli union — the probability that A-only fires OR B-only fires when each fires independently with its measured rate). The <em>joint-specific leakage residual</em> is the difference between the joint LoRA's actual rate and that expected-union baseline:</p>
<p class="display-math">\[ r_p \;=\; \text{rate}_{\text{joint}}(p) \;-\; \bigl[\text{rate}_A(p) + \text{rate}_B(p) - \text{rate}_A(p)\cdot \text{rate}_B(p)\bigr]. \]</p>
<p>\(r_p > 0\) means the joint LoRA leaks more than its two single-source parts already would; \(r_p < 0\) means it leaks less. The geometric-interpolation hypothesis predicts \(r_p\) is larger for bystanders nearer the A↔B midpoint than for off-axis bystanders.</p>

<p><strong>Why partial Spearman.</strong> Spearman rather than Pearson because the relationship between distance and leakage isn't expected to be linear (only monotonic), so a rank correlation is more appropriate. <em>Partial</em> because there's an obvious confounder: a bystander near the A↔B midpoint also tends to have high average cosine similarity to A and B individually — call that average cosine \(s(p)\). Cosine-to-source is what predicts single-source leakage in prior work, so if I just correlated \(r_p\) with \(d_{\text{mid}}(p)\) directly, I might pick up a signal that's really driven by \(s(p)\). The partial Spearman strips out the influence of \(s(p)\) from both \(r_p\) and \(d_{\text{mid}}(p)\) (by linearly regressing each on \(s(p)\) and taking the residuals), then computes the rank correlation on the residuals. That's what the scatter above plots.</p>

<p><strong>Test result.</strong> Partial Spearman of \(r_p\) against \(d_{\text{mid}}(p)\) is \(\rho = -0.348\) (one-sided \(p = 0.086\), \(N = 17\)) — in the predicted direction \(\rho < 0\) (bystanders closer to the midpoint leak more), but not crossing the one-sided \(\alpha = 0.05\) threshold at this sample size. Inconclusive: the data point in the predicted direction but the evidence isn't strong enough to commit to.</p>

<p><strong>Full parameters:</strong></p>
<table class="setup">
  <tr><th>Base model</th><td>Qwen2.5-7B-Instruct</td></tr>
  <tr><th>LoRA</th><td>r=16, α=32, targets <code>q_proj,k_proj,v_proj,o_proj</code></td></tr>
  <tr><th>Optimizer</th><td>AdamW, lr=3e-4, cosine schedule, 3 epochs</td></tr>
  <tr><th>Batch / accum</th><td>per-device 8, grad-accum 2 (effective 16)</td></tr>
  <tr><th>Training examples</th><td>A-only: 400 (paramedic); B-only: 400 (comedian); joint: 800</td></tr>
  <tr><th>Source pair</th><td>paramedic × comedian; centered cosine \(-0.65\) at L20</td></tr>
  <tr><th>Bystanders</th><td>\(N = 17\) held-out personas</td></tr>
  <tr><th>Eval per cell</th><td>20 prompts × 20 completions = 400 per (bystander, LoRA)</td></tr>
  <tr><th>Activation layer</th><td>L20 of Qwen2.5-7B-Instruct (selected by <a href="https://github.com/superkaiba/explore-persona-space/issues/341">#341</a>)</td></tr>
  <tr><th>Seed</th><td>137 (training), 42 (eval sampling)</td></tr>
  <tr><th>Statistical test</th><td>One-sided partial Spearman, \(\alpha = 0.05\), predicted direction \(\rho < 0\)</td></tr>
  <tr><th>Code commit</th><td><code><a href="https://github.com/superkaiba/explore-persona-space/tree/921b304d">921b304d</a></code></td></tr>
</table>

</div>
</details>

</main>

</div>
</div>

