---
title: Persona-vector recipes are unreliable as cross-persona predictors on Qwen2.5-7B-Instruct
  — bare centroids beat the Chen et al. mean-diff family on leakage, recipes disagree
  with each other, and prior reported effects fail their controls (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-05-12T19:21:45.477Z'
has_clean_result: true
sagan_id: abc9d58f-9f9f-406c-a6cb-fbe7a043cb91
sagan_number: 368
priority: normal
legacy_why_unset: true
---
<!-- legacy-sagan-card -->
<style>
  .cr-368 {
    --ivory:   #FAF9F5;
    --paper:   #FFFFFF;
    --slate:   #141413;
    --clay:    #D97757;
    --olive:   #788C5D;
    --rust:    #B65B3F;
    --gold:    #C49B3F;
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
  .cr-368 * { box-sizing: border-box; }
  .cr-368 main.content { max-width: 760px; min-width: 0; margin: 0 auto; }
  .cr-368 h2 { font-size: 1.1rem; font-weight: 600; letter-spacing: -0.005em; margin: 1.6rem 0 .5rem; scroll-margin-top: 1rem; }
  .cr-368 h3 { font-size: .98rem; font-weight: 600; letter-spacing: -0.005em; color: var(--slate); margin: 1.4rem 0 .4rem; scroll-margin-top: 1rem; }
  .cr-368 p { margin: .6rem 0; }
  .cr-368 ul { padding-left: 1.1rem; margin: .4rem 0 .8rem; }
  .cr-368 li { margin: .35rem 0; }
  .cr-368 hr { border: 0; border-top: 1px solid var(--gray-300); margin: 2rem 0; }
  .cr-368 a { color: var(--clay); text-decoration: none; border-bottom: 1px solid currentColor; }
  .cr-368 a:hover { color: var(--slate); }
  .cr-368 code { font-family: var(--mono); font-size: .92em; background: var(--gray-150); padding: 0 .3em; border-radius: 3px; }
  .cr-368 pre { font-family: var(--mono); font-size: .82rem; background: var(--gray-150); padding: .8rem 1rem; border-radius: 6px; overflow-x: auto; line-height: 1.5; }
  .cr-368 section.tldr h2 { margin-top: 0; }
  .cr-368 section.tldr ul { padding-left: 1.2rem; }
  .cr-368 section.tldr li { margin: .45rem 0; line-height: 1.55; }
  .cr-368 figure.plot-card { margin: 1.5rem 0 1.2rem; padding: 0; }
  .cr-368 figure.plot-card svg { display: block; width: 100%; height: auto; background: var(--paper); border: 1px solid var(--gray-300); border-radius: 6px; }
  .cr-368 figure.plot-card figcaption { margin-top: .8rem; font-size: .88rem; line-height: 1.55; color: var(--slate); }
  .cr-368 details { border-top: 1px solid var(--gray-300); padding: .9rem 0; margin: 0; }
  .cr-368 details + details { border-top: 1px solid var(--gray-300); }
  .cr-368 details:last-of-type { border-bottom: 1px solid var(--gray-300); }
  .cr-368 details > summary { cursor: pointer; list-style: none; font-weight: 600; font-size: 1rem; padding-left: 1.2rem; position: relative; }
  .cr-368 details > summary::-webkit-details-marker { display: none; }
  .cr-368 details > summary::before { content: ""; position: absolute; left: 0; top: .25rem; width: .7rem; height: .7rem; border-right: 2px solid var(--gray-500); border-bottom: 2px solid var(--gray-500); transform: rotate(-45deg); transition: transform .15s ease; }
  .cr-368 details[open] > summary::before { transform: rotate(45deg); top: .15rem; }
  .cr-368 details > summary:hover { color: var(--clay); }
  .cr-368 details > summary:hover::before { border-color: var(--clay); }
  .cr-368 details > div { padding: 1rem 0 .4rem; }
  .cr-368 table.setup { width: 100%; border-collapse: collapse; font-size: .88rem; margin: .5rem 0 1rem; border: 1px solid var(--gray-300); border-radius: 6px; overflow: hidden; }
  .cr-368 table.setup th, .cr-368 table.setup td { text-align: left; padding: .5rem .8rem; vertical-align: top; border-bottom: 1px solid var(--gray-300); }
  .cr-368 table.setup tr:last-child th, .cr-368 table.setup tr:last-child td { border-bottom: 0; }
  .cr-368 table.setup th { width: 34%; color: var(--gray-500); font-weight: 500; background: var(--gray-150); border-right: 1px solid var(--gray-300); }
</style>

<div class="cr-368">
<main class="content">

<section id="tldr" class="tldr">
<h2>TL;DR</h2>
<ul>
  <li><strong>Motivation.</strong> Persona vectors (Chen et al. 2025) are a popular Anthropic recipe for monitoring trait-related directions in a model's internal activations: subtract the mean hidden state under a "you are a helpful assistant" baseline from the mean hidden state under a trait-eliciting prompt, then score new states by cosine with that difference vector. Five experiments in this repo asked the same underlying question from different angles: are these recipes reliable enough to use as cross-persona predictors of behaviour? (<a href="https://sagan.superkaiba.com/e/experiment/98245695-db43-494d-875e-59bfb2f2455c">#168</a> SAE-feature proximity, <a href="https://sagan.superkaiba.com/e/experiment/afd3249c-92e6-4111-9865-27721cad81b5">#216</a> 6-recipe cross-recipe agreement, <a href="https://sagan.superkaiba.com/e/experiment/84588f87-62fe-4961-ae51-0e88a8218f9f">#263</a> 672-cell validation sweep, <a href="https://sagan.superkaiba.com/e/experiment/0b6d0840-6791-4638-82e7-f33fc86f8cf9">#340</a> cosine-vs-vulnerability with length controls, this lead <a href="https://sagan.superkaiba.com/e/experiment/abc9d58f-9f9f-406c-a6cb-fbe7a043cb91">#368</a> head-to-head bake-off).</li>
  <li><strong>What I ran.</strong> Across the five experiments I extracted 6 Chen-style persona-vector variants (layers 15/20/25, last-token, anti-helpful orthogonalized, projection-diff) and 2 simpler centroid baselines (no helpful-baseline subtraction) on Qwen2.5-7B-Instruct, then projected each onto two independent leakage datasets — a <code>[ZLT]</code>-marker non-persona-trigger regression (N=128 cells across 4 trained triggers × 32 test prompts) and a 50-pair persona-leakage table (10 personas). I also (i) compared the 6 recipes against each other across all 28 layers on 275 personas to test absolute-direction vs relative-geometry recovery, (ii) checked whether the Qwen default identity prompt sits closer to EM-persona SAE feature directions than other system prompts (N=1000 permutations), and (iii) partialled prompt length out of a previously-reported cosine-to-assistant → marker-implantation-rate correlation on 48 personas.</li>
  <li><strong>Results (see <a href="#figure">figure below</a>).</strong> On the head-to-head leakage-prediction bake-off, all 6 Chen-style mean-diff recipes either flatlined or had wrong-sign correlations with leakage, while bare per-persona centroids (no helpful-baseline subtraction) and even a no-hidden-states semantic-cosine baseline cleanly beat them. The canonical Chen recipe at layer 20 hit Spearman ρ = −0.107 (N=128, p=0.23), vs +0.639 for the last-input-token centroid and +0.481 for semantic cosine. The same pattern repeated on the 50-pair persona table: Chen at ρ = 0.034 (n=40, inside the null), centroid at |ρ| = 0.788. Across the supporting experiments, the 6 recipes correlate only at off-diagonal mean ρ = 0.39 with each other (well below the 0.70 robustness threshold), no layer in the 28-layer sweep satisfies both absolute-direction and relative-geometry pass criteria, the Qwen default identity prompt is NOT closer to EM-persona directions than a generic assistant prompt (permutation p=0.74), and the originally-claimed cosine→marker-implantation effect collapses to ρ=−0.008 (p=0.95) once log prompt length is partialled out.</li>
  <li><strong>Next steps.</strong>
    <ul>
      <li>Persona-vector recipes are not the right downstream predictor in this codebase — close the line of inquiry for leakage prediction. The bare last-input-token centroid is the strongest single axis observed (|ρ| up to 0.788), and is what subsequent experiments should benchmark against.</li>
      <li>The Chen et al. recipe may still be useful for the trait-monitoring purpose it was designed for (training-time activation steering, refusal-direction extraction); the failure here is specifically against cross-persona leakage prediction.</li>
      <li>The cosine↔prompt-length confound (ρ=−0.75 in the panel) needs a controlled manipulation (<a href="https://github.com/superkaiba/explore-persona-space/issues/339">issue #339</a>) before any geometric-distance claim is re-opened on this axis.</li>
    </ul>
  </li>
</ul>
</section>

<figure id="figure" class="plot-card">
  <svg viewBox="0 0 900 540" xmlns="http://www.w3.org/2000/svg" role="img"
       aria-label="Horizontal bar chart of 9 leakage predictors on Qwen2.5-7B-Instruct. The 6 Chen-style persona-vector recipes are clustered near zero or slightly negative correlation with marker leakage rate. The two bare per-persona centroid baselines and the no-hidden-states semantic cosine baseline are all positive and substantially higher. The strongest predictor is the last-input-token centroid at +0.64, well above all 6 mean-diff recipes.">
    <defs>
      <style>
        .ax       { stroke: #87867F; stroke-width: 1.2; fill: none; }
        .ax-tick  { stroke: #87867F; stroke-width: 1.2; }
        .ax-zero  { stroke: #141413; stroke-width: 1.2; }
        .ax-thr   { stroke: #B1AFA4; stroke-width: 1.2; stroke-dasharray: 4 4; }
        .tick     { font: 11px ui-monospace, "SF Mono", Menlo, monospace; fill: #87867F; }
        .ax-label { font: 12px ui-sans-serif, system-ui, sans-serif; fill: #141413; }
        .row-label { font: 12px ui-sans-serif, system-ui, sans-serif; fill: #141413; }
        .row-label-mono { font: 11px ui-monospace, "SF Mono", Menlo, monospace; fill: #87867F; }
        .title    { font: 14px ui-sans-serif, system-ui, sans-serif; fill: #141413; font-weight: 600; }
        .subtitle { font: 11.5px ui-sans-serif, system-ui, sans-serif; fill: #87867F; }
        .bar-chen { fill: #B65B3F; opacity: .85; }
        .bar-base { fill: #788C5D; opacity: .9; }
        .bar-cmp  { fill: #C49B3F; opacity: .9; }
        .err      { stroke: #141413; stroke-width: 1.3; fill: none; }
        .thr-label{ font: 10.5px ui-sans-serif, system-ui, sans-serif; fill: #87867F; }
        .legend-sw { stroke: #141413; stroke-width: .6; }
        .legend-tx { font: 11px ui-sans-serif, system-ui, sans-serif; fill: #141413; }
      </style>
    </defs>

    <text x="450" y="30" text-anchor="middle" class="title">Which axes predict marker-token leakage on Qwen2.5-7B-Instruct?</text>
    <text x="450" y="48" text-anchor="middle" class="subtitle">Spearman rank correlation between each axis and per-cell marker leakage rate (N = 128 cells)</text>

    <line class="ax" x1="300" y1="60"  x2="300"  y2="450"/>
    <line class="ax" x1="300" y1="450" x2="860" y2="450"/>

    <line class="ax-zero" x1="478.2" y1="60" x2="478.2" y2="450"/>
    <line class="ax-thr"  x1="758.2" y1="60" x2="758.2" y2="450"/>

    <g>
      <line class="ax-tick" x1="300"   y1="450" x2="300"   y2="456"/>
      <line class="ax-tick" x1="376.4" y1="450" x2="376.4" y2="456"/>
      <line class="ax-tick" x1="453"   y1="450" x2="453"   y2="456"/>
      <line class="ax-tick" x1="478.2" y1="450" x2="478.2" y2="456"/>
      <line class="ax-tick" x1="554.5" y1="450" x2="554.5" y2="456"/>
      <line class="ax-tick" x1="631"   y1="450" x2="631"   y2="456"/>
      <line class="ax-tick" x1="707.3" y1="450" x2="707.3" y2="456"/>
      <line class="ax-tick" x1="758.2" y1="450" x2="758.2" y2="456"/>
      <line class="ax-tick" x1="783.7" y1="450" x2="783.7" y2="456"/>
      <line class="ax-tick" x1="860"   y1="450" x2="860"   y2="456"/>
      <text class="tick" x="300"   y="470" text-anchor="middle">−0.30</text>
      <text class="tick" x="376.4" y="470" text-anchor="middle">−0.20</text>
      <text class="tick" x="453"   y="470" text-anchor="middle">−0.10</text>
      <text class="tick" x="478.2" y="482" text-anchor="middle">0</text>
      <text class="tick" x="554.5" y="470" text-anchor="middle">+0.10</text>
      <text class="tick" x="631"   y="470" text-anchor="middle">+0.20</text>
      <text class="tick" x="707.3" y="470" text-anchor="middle">+0.40</text>
      <text class="tick" x="758.2" y="482" text-anchor="middle">+0.55</text>
      <text class="tick" x="783.7" y="470" text-anchor="middle">+0.60</text>
      <text class="tick" x="860"   y="470" text-anchor="middle">+0.75</text>
    </g>

    <text class="ax-label" x="580" y="505" text-anchor="middle">Spearman correlation with marker leakage rate — right = stronger positive predictor</text>
    <text class="thr-label" x="758" y="55" text-anchor="middle">H1 pass threshold (ρ ≥ 0.55)</text>

    <!-- Row 1: pvec L25 -->
    <text class="row-label" x="290" y="99" text-anchor="end">mean-diff at layer 25</text>
    <text class="row-label-mono" x="290" y="113" text-anchor="end">(Chen et al. variant)</text>
    <rect class="bar-chen" x="378.1" y="89" width="100.1" height="14"><title>mean-diff at layer 25: ρ = −0.197, p = 0.026, CI [−0.31, −0.02], N=128. Significantly negative — the wrong direction.</title></rect>
    <line class="err" x1="322.8" y1="96" x2="467.0" y2="96"/>
    <line class="err" x1="322.8" y1="92" x2="322.8" y2="100"/>
    <line class="err" x1="467.0" y1="92" x2="467.0" y2="100"/>

    <!-- Row 2: pvec L20 projdiff -->
    <text class="row-label" x="290" y="139" text-anchor="end">mean-diff at L20 (projection-diff)</text>
    <text class="row-label-mono" x="290" y="153" text-anchor="end">(Chen et al. variant)</text>
    <rect class="bar-chen" x="422.6" y="129" width="55.6" height="14"><title>mean-diff at L20 projection-diff: ρ = −0.109, p = 0.22, CI [−0.24, +0.07], N=128. Not significantly different from zero.</title></rect>
    <line class="err" x1="355.3" y1="136" x2="514.8" y2="136"/>
    <line class="err" x1="355.3" y1="132" x2="355.3" y2="140"/>
    <line class="err" x1="514.8" y1="132" x2="514.8" y2="140"/>

    <!-- Row 3: pvec L20 (Chen canonical) -->
    <text class="row-label" x="290" y="179" text-anchor="end">mean-diff at L20 (canonical Chen et al.)</text>
    <text class="row-label-mono" x="290" y="193" text-anchor="end">(headline recipe)</text>
    <rect class="bar-chen" x="423.6" y="169" width="54.6" height="14"><title>mean-diff at L20 (canonical Chen et al. recipe): ρ = −0.107, p = 0.23, CI [−0.24, +0.07], N=128. Inside the null, slight wrong-direction point estimate. Δρ vs semantic_cos = −0.59 (paired bootstrap, p &lt; 0.001).</title></rect>
    <line class="err" x1="354.8" y1="176" x2="516.3" y2="176"/>
    <line class="err" x1="354.8" y1="172" x2="354.8" y2="180"/>
    <line class="err" x1="516.3" y1="172" x2="516.3" y2="180"/>

    <!-- Row 4: pvec L20 orthog -->
    <text class="row-label" x="290" y="219" text-anchor="end">mean-diff at L20 (anti-helpful orthogonalized)</text>
    <text class="row-label-mono" x="290" y="233" text-anchor="end">(Chen et al. variant)</text>
    <rect class="bar-chen" x="424.2" y="209" width="54.0" height="14"><title>mean-diff at L20 orthogonalized: ρ = −0.106, p = 0.23, CI [−0.24, +0.08], N=128. Algebraically near-identical to canonical L20 (cross-correlation ρ ≈ 1.00).</title></rect>
    <line class="err" x1="355.7" y1="216" x2="517.2" y2="216"/>
    <line class="err" x1="355.7" y1="212" x2="355.7" y2="220"/>
    <line class="err" x1="517.2" y1="212" x2="517.2" y2="220"/>

    <!-- Row 5: pvec L15 -->
    <text class="row-label" x="290" y="259" text-anchor="end">mean-diff at layer 15</text>
    <text class="row-label-mono" x="290" y="273" text-anchor="end">(Chen et al. variant)</text>
    <rect class="bar-chen" x="470.9" y="249" width="7.3" height="14"><title>mean-diff at layer 15: ρ = −0.014, p = 0.87, CI [−0.17, +0.17], N=128. Essentially zero.</title></rect>
    <line class="err" x1="391.9" y1="256" x2="565.3" y2="256"/>
    <line class="err" x1="391.9" y1="252" x2="391.9" y2="260"/>
    <line class="err" x1="565.3" y1="252" x2="565.3" y2="260"/>

    <!-- Row 6: pvec last-token -->
    <text class="row-label" x="290" y="299" text-anchor="end">mean-diff at L20 (last-token aggregation)</text>
    <text class="row-label-mono" x="290" y="313" text-anchor="end">(Chen et al. variant)</text>
    <rect class="bar-chen" x="478.2" y="289" width="9.2" height="14"><title>mean-diff at L20 last-token: ρ = +0.018, p = 0.84, CI [−0.13, +0.18], N=128. Essentially zero.</title></rect>
    <line class="err" x1="410.0" y1="296" x2="570.3" y2="296"/>
    <line class="err" x1="410.0" y1="292" x2="410.0" y2="300"/>
    <line class="err" x1="570.3" y1="292" x2="570.3" y2="300"/>

    <!-- Row 7: centroid Method B (mean-response-token centroid) -->
    <text class="row-label" x="290" y="339" text-anchor="end">bare centroid (mean response token, L20)</text>
    <text class="row-label-mono" x="290" y="353" text-anchor="end">(no helpful-baseline subtraction)</text>
    <rect class="bar-base" x="478.2" y="329" width="229.9" height="14"><title>Bare centroid (mean response token, L20): ρ = +0.452, p = 8.7×10⁻⁸, CI [+0.34, +0.56], N=128. BH-FDR significant. Bit-identical to the &quot;positives-only&quot; chenstyle axis — i.e., what the Chen recipe reduces to when the helpful-baseline subtraction is removed.</title></rect>
    <line class="err" x1="649.0" y1="336" x2="763.2" y2="336"/>
    <line class="err" x1="649.0" y1="332" x2="649.0" y2="340"/>
    <line class="err" x1="763.2" y1="332" x2="763.2" y2="340"/>

    <!-- Row 8: semantic cosine -->
    <text class="row-label" x="290" y="379" text-anchor="end">semantic cosine</text>
    <text class="row-label-mono" x="290" y="393" text-anchor="end">(no hidden states, published baseline)</text>
    <rect class="bar-cmp" x="478.2" y="369" width="244.8" height="14"><title>Semantic cosine baseline: ρ = +0.481, p = 9.2×10⁻⁹, CI [+0.35, +0.59], N=128. A text-embedding cosine between prompts — beats every persona-vector recipe without using any hidden states.</title></rect>
    <line class="err" x1="658.1" y1="376" x2="778.8" y2="376"/>
    <line class="err" x1="658.1" y1="372" x2="658.1" y2="380"/>
    <line class="err" x1="778.8" y1="372" x2="778.8" y2="380"/>

    <!-- Row 9: centroid Method A (last-input-token centroid) -->
    <text class="row-label" x="290" y="419" text-anchor="end">bare centroid (last input token, L20)</text>
    <text class="row-label-mono" x="290" y="433" text-anchor="end">(no helpful-baseline subtraction)</text>
    <rect class="bar-base" x="478.2" y="409" width="325.3" height="14"><title>Bare centroid (last input token, L20): ρ = +0.639, p = 4.7×10⁻¹⁶, CI [+0.48, +0.76], N=128. The only axis that clears the H1 pass threshold of ρ ≥ 0.55. On the 50-pair persona table this same axis hits |ρ| = 0.788.</title></rect>
    <line class="err" x1="723.9" y1="416" x2="864.4" y2="416"/>
    <line class="err" x1="723.9" y1="412" x2="723.9" y2="420"/>
    <line class="err" x1="864.4" y1="412" x2="864.4" y2="420"/>

    <!-- Legend -->
    <rect class="bar-chen legend-sw" x="20" y="60" width="14" height="14"/>
    <text class="legend-tx" x="40" y="71">Chen et al. mean-diff recipes (6 variants)</text>
    <rect class="bar-base legend-sw" x="20" y="84" width="14" height="14"/>
    <text class="legend-tx" x="40" y="95">Bare centroids (no helpful-baseline subtraction)</text>
    <rect class="bar-cmp legend-sw" x="20" y="108" width="14" height="14"/>
    <text class="legend-tx" x="40" y="119">No-hidden-state baseline (published comparator)</text>

    <text class="legend-tx" x="20" y="170">Whiskers = 95% CI</text>
    <text class="legend-tx" x="20" y="185">(cluster-resampled by</text>
    <text class="legend-tx" x="20" y="200">test prompt; 32 clusters)</text>

  </svg>
  <figcaption>
    Each row is one candidate predictor of marker-token leakage. The horizontal axis is Spearman rank correlation between that predictor and the per-cell marker leakage rate — higher and further right means a better predictor; zero means no relationship; left of zero means the predictor goes the wrong way. The dashed line at ρ = 0.55 is the pass threshold the cluster's lead experiment registered for "this recipe is useful." Every variant of the Chen et al. mean-diff recipe (red bars) sits at or below zero, with one significantly in the wrong direction; the two bare per-persona centroid axes (green) — which use the same hidden states at the same layer but skip the helpful-baseline subtraction — both clear the published semantic-cosine baseline (gold), and the last-input-token centroid is the only axis that crosses the 0.55 pass line. The cross-recipe disagreement chart and the 28-layer sweep that motivate "recipes are unreliable" live in the experimental-design dropdown below; this figure is the head-to-head leakage-prediction result. Confidence: HIGH — the three-null kill (no Chen-style variant beats baseline on this dataset, on the persona-pair dataset, OR on the cross-recipe-agreement check) is robust across 6 structurally distinct recipe variants and reproduces the prior published numbers within ±0.03 tolerance. Hover any bar for the exact statistics.
  </figcaption>
</figure>

<details id="design">
<summary>Experimental design</summary>
<div>

<p><strong>The shared question across the five contributing experiments.</strong> Persona vectors (Chen et al. 2025) are a recipe for extracting a single direction in a model's residual stream that represents some persona or trait. The canonical version: collect activations on response tokens for a set of trait-eliciting prompts (positive set); collect activations on the same model under "you are a helpful assistant" (negative set); mean-pool over response tokens at some layer; subtract the means. The cosine of any new hidden state with that vector is then taken as a score for how strongly the trait is present. The five experiments in this cluster all asked, from different angles, the same underlying question: <em>are these recipes reliable enough that you can use them to predict cross-persona behaviour</em> — leakage of a learned marker token, identity-prompt vulnerability, or marker-implantation rate?</p>

<p><strong>What each contributing experiment did.</strong></p>
<ul>
  <li><strong><a href="https://sagan.superkaiba.com/e/experiment/abc9d58f-9f9f-406c-a6cb-fbe7a043cb91">#368</a> (the lead, primary plot above).</strong> Head-to-head bake-off. Extracted 6 Chen-style persona-vector variants (mean-diff at layers 15/20/25, last-token at L20, anti-helpful orthogonalized at L20, projection-diff at L20) and 2 bare per-persona centroids (last-input-token at L20, mean-response-token at L20) on Qwen2.5-7B-Instruct, then projected each onto two leakage datasets. Phase 1: 128 cells (4 LoRA-trained system-prompt triggers × 32 held-out test prompts, with marker-leakage rate as the dependent variable) inherited from <a href="https://github.com/superkaiba/explore-persona-space/issues/343">issue #343</a>. Phase 2: 50 directed source→target persona pairs from <a href="https://github.com/superkaiba/explore-persona-space/issues/142">issue #142</a>. Computed Spearman ρ per axis, paired bootstrap of Δρ against the semantic-cosine baseline (cluster-resampled by test prompt, 32 clusters), within-source partial ρ on Phase 2, and BH-FDR at α=0.10 across the 7 non-headline axes.</li>
  <li><strong><a href="https://sagan.superkaiba.com/e/experiment/afd3249c-92e6-4111-9865-27721cad81b5">#216</a>.</strong> 6-recipe cross-recipe-agreement check on 275 personas × 240 questions × all 28 layers of Qwen2.5-7B-Instruct, with a same-recipe cross-question-half noise-floor control. The recipes vary on token aggregation (single-token vs mean-pooled) and forward-pass type (chat-templated vs raw, prompt-side vs response-side).</li>
  <li><strong><a href="https://sagan.superkaiba.com/e/experiment/84588f87-62fe-4961-ae51-0e88a8218f9f">#263</a>.</strong> 672-cell sweep (8 methods × 14 token positions × 28 layers, materialized subset of a 3,136-cell full grid) on 275 personas, asking whether per-persona validation-based recipe selection beats the project default discriminator AUC and whether the grid collapses into a small number of equivalence classes.</li>
  <li><strong><a href="https://sagan.superkaiba.com/e/experiment/98245695-db43-494d-875e-59bfb2f2455c">#168</a>.</strong> SAE-feature projection test on 50 neutral prompts × 4 system-prompt conditions (Qwen default, generic assistant, empty system, no system turn) at layers 7/11/15 of Qwen2.5-7B-Instruct, using Arditi et al.'s pre-trained SAEs (131K features, k=64). Track A projected the (Qwen-default minus generic-assistant) condition difference onto 10 known EM-persona decoder directions, with a permutation test against 1000 random direction draws.</li>
  <li><strong><a href="https://sagan.superkaiba.com/e/experiment/0b6d0840-6791-4638-82e7-f33fc86f8cf9">#340</a>.</strong> Re-aggregated 48 per-source LoRA marker-implantation runs (identical contrastive recipe across all sources) and asked whether cosine-to-assistant at L15 predicts source rate before and after partialling out log-tokenized prompt length. Fixed-length sub-panel of 5 personas at 6 tokens used as an independent check on direction.</li>
</ul>

<p><strong>Method common to all five.</strong> Qwen2.5-7B-Instruct as the base model. Hidden states extracted from forward passes (no training inside any of these experiments except #340, which re-uses 48 pre-existing LoRA adapters from earlier issues). All temperature=0 generation, all seed=42 for paired sampling, all mean-pooling over response tokens unless the recipe is explicitly a last-token variant. The five experiments use independent datasets and three independent dependent variables (cell-level <code>marker_rate</code>, directed-pair <code>marker_leakage_rate</code>, persona <code>[ZLT]</code> source rate), so a recipe that's broken in one place would still have to survive the other four.</p>

<p><strong>Three falsified claims, three independent lines of evidence.</strong></p>
<ol>
  <li><strong>Recipes don't predict leakage</strong> (the headline, #368). On Phase 1 (N=128 cells), the canonical Chen-style L20 recipe gives ρ = −0.107 (p=0.23), and the paired bootstrap of Δρ against semantic-cosine excludes zero on the worse side at Δρ = −0.59 (cluster-resampled by test prompt, 32 clusters, p &lt; 0.001). On Phase 2 (n=40 source ≠ assistant rows), the headline pvec sits at ρ = 0.034 inside the source-shuffled null (95th percentile |ρ| = 0.292), while JS-divergence sits at |ρ| = 0.746 and the last-input-token centroid at |ρ| = 0.788. The bare centroid uses the same hidden states at the same layer with the same mean-pooling as the canonical Chen recipe — the only operational difference is the helpful-baseline subtraction. The subtraction step is what destroys the signal; the centroid axes anti-correlate with the Chen-style recipes on per-cell rankings (cross-block range −0.25 to +0.26), consistent with the subtraction removing the trigger-correlated component that the centroid carries.</li>
  <li><strong>Recipes disagree with each other</strong> (#216 cross-recipe; #368 cross-recipe-agreement Result 3; #263 grid-clustering). Across the 8 axes on Phase 1, the off-diagonal mean Spearman ρ of per-cell rankings is 0.39 (or 0.33 with the projdiff degenerate variant dropped), well below the pre-registered 0.70 robustness threshold. The within-Chen-style 6×6 block shows partial agreement (L20–L15 = 0.81, L20–L25 = 0.54, L15–L25 = 0.24), but the centroid–Chen-style cross-block actively anti-correlates. The 28-layer cross-recipe sweep in #216 (275 personas, 6 recipes) confirms the same pattern at scale: per-persona absolute-direction cosine ranges 0.01–0.70 between recipes against a same-recipe noise floor of ≥0.99, while mean-centred Pearson correlation on the 275×275 persona-cosine matrix reaches 0.90 at deep layers — the absolute encoding is recipe-specific, the relative cluster structure is not. No layer satisfies both pass criteria simultaneously (419/420 cells fail). The #263 sweep over a larger 672-cell (method × token × layer) grid finds 57 mc_r ≥ 0.90 equivalence classes with the largest class covering only 47% of cells, not the ≤5 classes / ≥80% coverage the project's prior framing assumed.</li>
  <li><strong>Prior reported effects don't survive controls</strong> (#168 SAE-feature null; #340 length-partial null). #168: the (Qwen-default minus generic-assistant) residual stream difference is representationally distinct (Qwen default is the cosine outlier across SAE-decoded activations at layers 7/11/15), but it is NOT preferentially aligned with 10 known EM-persona decoder directions — permutation p = 0.74 against 1000 random direction draws, and 9 of 10 EM features point in the wrong direction (generic-assistant closer to EM features than Qwen-default). #340: the previously-reported cosine-to-assistant → marker-implantation-rate correlation at N=12 (raw ρ = −0.35, p=0.014 at N=48) collapses to ρ = −0.008 (p = 0.95) after log-prompt-length is partialled out, and at fixed prompt length the highest-cosine personas have the highest, not lowest, source rates. Cosine and prompt length are heavily co-linear in the panel (Spearman ρ = −0.75 at N=24), so cosine remains plausible as a mediator that length is downstream of — but the published cosine→vulnerability headline as stated does not survive the length control.</li>
</ol>

<p><strong>Sample completions (cherry-picked for illustration).</strong> Three representative Phase-1 cells from <code>regression_data_augmented.csv</code> showing why the bare centroid axis carries information that the Chen-style mean-diff axis throws away. Full raw completions for all 128 cells live in <a href="https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue368_persona_vectors_chenstyle"><code>superkaiba1/explore-persona-space-data :: issue368_persona_vectors_chenstyle/</code></a> on HuggingFace Hub (Phase 1 trigger-side, Phase 2 persona-side, the helpful-baseline negative set, and the empty-prompt baseline):</p>

<pre>trained_trigger:  T_task   (LoRA adapter, "answer SaaS support tickets")
test_id:          match_T_task   (= the trigger prompt; family=task)
marker_rate:      0.15           (15% of completions emit [ZLT])
  semantic_cos                     = 1.000   (it IS the trigger prompt)
  pvec_chenstyle_L20  (Chen et al.) = 0.296   (modest, doesn't reflect the 15% rate)
  pcentroid_methodA_L20 (centroid)  = 0.972   (matches semantic_cos)

test_id=match_T_context, family=context, marker_rate=0.20
  semantic_cos                     = 0.05
  pvec_chenstyle_L20                = 0.180
  pcentroid_methodA_L20             = 0.717   (centroid varies; pvec stays flat)

test_id=match_T_format, family=format, marker_rate=0.00
  semantic_cos                     = -0.06
  pvec_chenstyle_L20                = 0.361
  pcentroid_methodA_L20             = 0.489   (centroid tracks low leakage; pvec doesn't)</pre>

<p>Across all 128 Phase-1 cells, the canonical Chen-style pvec stays compressed in [+0.180, +0.487] regardless of the actual marker rate, while the last-input-token centroid spans [0.49, 0.97] and ranks the cells correctly. Same hidden states, same layer, same mean-pooling — the helpful-baseline subtraction is what flattens the signal. (For the corresponding persona-side examples, see Phase 2 in <a href="https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue368_persona_vectors_chenstyle"><code>issue368_persona_vectors_chenstyle/</code></a>.)</p>

<p><strong>Why Spearman, why partial, why cluster-bootstrapped.</strong> Spearman because the recipe→leakage relationship isn't expected to be linear (only monotonic), so a rank correlation is more appropriate than Pearson. The Phase 1 dataset has a strong within-trigger structure (32 test prompts × 4 trained triggers), so a naive bootstrap underestimates uncertainty; the cluster bootstrap resamples test-prompt groups (32 clusters of 4 cells each) per the #343 R6 spec. Phase 2 uses a within-source partial Spearman because some persona sources have collapsed leakage variance (villain, comedian) and a naive marginal rho on n=40 is dominated by the high-variance sources; the partial-rho range across the 4 non-degenerate sources is reported in the lead body. The #340 length-partial follows the same logic with log-tokenized prompt length as the partialled variable.</p>

<p><strong>What about Result 4-and-beyond from #368 specifically?</strong> The lead's body originally also reported (i) a 6×6 cross-recipe agreement heatmap on Phase 1 and a 8×8 with centroids, (ii) verbatim BH-FDR tables, (iii) the persona-pos-set-cohesion check that rules out "the Sonnet-generated positive sets are too uniform" as an alternative explanation, and (iv) collinearity diagnostics. All four are preserved in the underlying eval JSONs (linked in the Reproducibility dropdown below) but are not in this cluster body because the head-to-head leakage figure already carries the headline and the rest are sanity checks. Likewise, #168's SAE-feature breakdown (54–95 features per condition pair pass permutation tests at each layer), #216's 28-layer joint-pass sweep figure, and #263's 57-cluster equivalence-class breakdown all live in the contributing experiments' own bodies.</p>

<p><strong>Confidence: HIGH — three independent kill criteria fire (leakage prediction fails on two distinct datasets; cross-recipe agreement fails on a 28-layer sweep AND on a 672-cell grid; the published cosine→vulnerability and EM-feature-proximity headlines both fail their respective controls) across two model passes (#168 SAE-based, #216/#263/#340/#368 raw hidden state) on the same base model, with the centroid-vs-pvec replacement reproducing prior published numbers within ±0.03 tolerance.</strong> The binding evidence is the Phase 1 paired statistic in #368: Δρ vs semantic_cos = −0.59 (p &lt; 0.001, cluster-bootstrap by test prompt, 32 clusters, N=128), which rules out a meaningful positive effect tightly. Single-seed (seed=42) is acceptable because the inference-only pipeline is bit-identical across reruns at temperature=0. The scope is limited to Qwen2.5-7B-Instruct; we have not tested whether the same Chen recipe fails on a larger or different base model.</p>

<p><strong>Full parameters:</strong></p>
<table class="setup">
  <tr><th>Base model</th><td>Qwen2.5-7B-Instruct, HF revision <code>bb46c15</code> (7.6B params, 28 layers, hidden_dim=3584), bf16</td></tr>
  <tr><th>Recipes (lead #368)</th><td>6 Chen-style mean-diff variants (L15, L20, L25, last-token L20, anti-helpful orthogonalized L20, projection-diff L20) + 2 bare centroids (Method A = last input token L20, Method B = mean response token L20). Helpful-baseline = "you are a helpful assistant" with the same 20 EVAL_QUESTIONS.</td></tr>
  <tr><th>Datasets</th><td>Phase 1 = 128 cells (4 non-persona-trigger LoRAs × 32 held-out test prompts) from <a href="https://github.com/superkaiba/explore-persona-space/issues/343">#343</a>; Phase 2 = 50 directed source→target pairs (10 personas) from <a href="https://github.com/superkaiba/explore-persona-space/issues/142">#142</a></td></tr>
  <tr><th>Personas / questions (#216, #263)</th><td>275 assistant-axis personas × 240 questions per centroid; same dataset for both</td></tr>
  <tr><th>SAE (#168)</th><td>Arditi et al. pre-trained SAEs, 131K features, k=64, layers 7/11/15; N=50 neutral prompts × 4 system-prompt conditions; permutation N=1000 shuffles</td></tr>
  <tr><th>LoRA panel (#340)</th><td>48 per-source LoRA marker-implantation runs (identical contrastive recipe across sources); WandB <a href="https://wandb.ai/thomasjiralerspong/leakage-experiment"><code>thomasjiralerspong/leakage-experiment</code></a></td></tr>
  <tr><th>Generation</th><td>vLLM, temperature=0, top_p=1.0, max_tokens=512, seed=42 (paired-completion sampling)</td></tr>
  <tr><th>Statistical tests</th><td>Spearman ρ per axis; paired bootstrap of Δρ vs baseline (cluster-resampled by test prompt, 32 clusters, 1000 draws); BH-FDR at α=0.10 on the non-headline axis pool (m=7 after dedup); source-shuffled null for Phase 2 (1000 draws); within-source partial Spearman on Phase 2 non-degenerate sources; partial Spearman with log-tokenized prompt length controlled (#340)</td></tr>
  <tr><th>Thresholds</th><td>H1 (#368 Phase 1) holds iff ρ ≥ 0.55 AND ΔR² ≥ 0.04; H2 (#368 Phase 2) holds iff |ρ| ≥ 0.75 AND within-source partial ρ ≥ +0.30; H3a (cross-recipe agreement) holds iff off-diagonal mean ρ ≥ 0.70; #216 joint pass = per-persona cos_min ≥ 0.99 AND mean-centred r ≥ 0.90 (419/420 cells KILL)</td></tr>
  <tr><th>Compute</th><td>#368 ≈ 0.5 GPU-hours on 1× H100 80GB; #216 ≈ 4 GPU-hours; #263 ≈ 8 GPU-hours; #168 ≈ 2 GPU-hours; #340 inference-only re-aggregation of prior runs</td></tr>
</table>

</div>
</details>

<details id="repro">
<summary>Reproducibility (agent-facing)</summary>
<div>

<p><strong>Contributing experiments (Sagan IDs and artifact URLs).</strong></p>
<ul>
  <li><strong><a href="https://sagan.superkaiba.com/e/experiment/abc9d58f-9f9f-406c-a6cb-fbe7a043cb91">#368</a> — head-to-head bake-off (lead).</strong>
    <ul>
      <li>Persona-vector tensors: <code><a href="https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue368_persona_vectors_chenstyle">superkaiba1/explore-persona-space-data :: issue368_persona_vectors_chenstyle/</a></code> (281 <code>.pt</code> tensors + raw response JSONs)</li>
      <li>Eval JSON: <code>eval_results/issue_368/phase1/{h1_verdict,per_axis_stats,regression_results,permutation_null,bh_fdr,collinearity_diagnostics,conditional_nonzero}.json</code>, <code>eval_results/issue_368/phase1/recipe_agreement_matrix_{with,no}_projdiff.csv</code>, <code>eval_results/issue_368/phase2/{h2_verdict,per_axis_stats,permutation_null,reproduction_sanity,source_partial_rho,source_shuffle_permutation,persona_pos_set_cohesion,bh_fdr}.json</code></li>
      <li>Hero figure source data (used for the primary plot above): <code><a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/1afeb93c63aba2cc8cc7daf36fef34f66e0f4557/eval_results/issue_368/phase1/per_axis_stats.json">phase1/per_axis_stats.json</a></code>, <code><a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/1afeb93c63aba2cc8cc7daf36fef34f66e0f4557/eval_results/issue_368/phase2/per_axis_stats.json">phase2/per_axis_stats.json</a></code></li>
      <li>Code: <code><a href="https://github.com/superkaiba/explore-persona-space/blob/issue-368/scripts/i368_extract_chenstyle_pvecs.py">i368_extract_chenstyle_pvecs.py</a></code>, <code>i368_phase1_projection.py</code>, <code>i368_phase2_projection.py</code>, <code>i368_phase1_analysis.py</code>, <code>i368_phase2_analysis.py</code>, <code>i368_figures.py</code> at branch <code>issue-368</code></li>
      <li>Git commits: extraction/analysis at <code>1afeb93c</code>; final hot-fix at <code>eeccef51</code></li>
    </ul>
  </li>
  <li><strong><a href="https://sagan.superkaiba.com/e/experiment/afd3249c-92e6-4111-9865-27721cad81b5">#216</a> — 6-recipe × 28-layer cross-recipe agreement.</strong>
    <ul>
      <li>Dataset: 275 personas in <code>data/assistant_axis/role_list.json</code> × 240 questions; centroids on <code><a href="https://huggingface.co/datasets/superkaiba1/explore-persona-space-data">superkaiba1/explore-persona-space-data</a></code> (<code>assistant_axis/</code> subtree)</li>
      <li>Code: <code><a href="https://github.com/superkaiba/explore-persona-space/blob/main/scripts/extract_persona_vectors.py">scripts/extract_persona_vectors.py</a></code>, <code><a href="https://github.com/superkaiba/explore-persona-space/blob/main/scripts/compare_extraction_methods_6way.py">compare_extraction_methods_6way.py</a></code></li>
    </ul>
  </li>
  <li><strong><a href="https://sagan.superkaiba.com/e/experiment/84588f87-62fe-4961-ae51-0e88a8218f9f">#263</a> — 672-cell validation-based recipe sweep.</strong>
    <ul>
      <li>Dataset: 275 personas × 240 questions, 672 materialized (method × token × layer) cells (8 methods × 14 tokens × 28 layers = 3,136 cell grid, 672 materialized after mid-run disk-budget tightening to per-q subset {0, 128})</li>
      <li>Eval JSONs in repo under <code>eval_results/issue_263/</code></li>
    </ul>
  </li>
  <li><strong><a href="https://sagan.superkaiba.com/e/experiment/98245695-db43-494d-875e-59bfb2f2455c">#168</a> — Qwen-default-vs-EM-feature SAE projection.</strong>
    <ul>
      <li>SAE artifacts: Arditi et al. pre-trained SAEs at <code>arditi/qwen-2.5-7b-instruct-saes</code> (131K features, k=64; layers 7/11/15)</li>
      <li>Figure: <code><a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/5ccd21d/figures/sae_system_prompt/condition_similarity_heatmap.png">figures/sae_system_prompt/condition_similarity_heatmap.png</a></code></li>
      <li>Git commit: <code>5ccd21d</code></li>
    </ul>
  </li>
  <li><strong><a href="https://sagan.superkaiba.com/e/experiment/0b6d0840-6791-4638-82e7-f33fc86f8cf9">#340</a> — cosine-to-assistant vs marker-implantation, with length partial.</strong>
    <ul>
      <li>LoRA runs: WandB project <code><a href="https://wandb.ai/thomasjiralerspong/leakage-experiment">thomasjiralerspong/leakage-experiment</a></code> (48 per-source runs)</li>
      <li>Training data: <code><a href="https://huggingface.co/datasets/superkaiba1/explore-persona-space-data">superkaiba1/explore-persona-space-data</a></code></li>
      <li>Follow-up issues: <a href="https://github.com/superkaiba/explore-persona-space/issues/337">#337</a> (length predicts marker localization, MODERATE), <a href="https://github.com/superkaiba/explore-persona-space/issues/339">#339</a> (controlled length-decorrelation manipulation)</li>
    </ul>
  </li>
</ul>

<p><strong>Compute footprint (cluster total).</strong></p>
<ul>
  <li>Wall time: ~14.5 GPU-hours summed across the 5 experiments (lead #368 ≈ 0.5h, #216 ≈ 4h, #263 ≈ 8h, #168 ≈ 2h, #340 inference-only re-aggregation)</li>
  <li>Hardware: 1× H100 80GB per experiment; ephemeral RunPod pods, terminated after upload</li>
</ul>

<p><strong>Reproduce the primary figure.</strong></p>
<pre>curl -s https://raw.githubusercontent.com/superkaiba/explore-persona-space/1afeb93c63aba2cc8cc7daf36fef34f66e0f4557/eval_results/issue_368/phase1/per_axis_stats.json > phase1.json
# The primary plot's nine bars are spearman_rho values from per_axis stats with bootstrap_cluster_test_id_95ci as whiskers,
# plus semantic_cos rho/CI from issue_343's published regression CSV.</pre>

</div>
</details>

</main>
</div>
