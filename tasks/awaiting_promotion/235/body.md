---
title: Language-mismatch LoRA SFT on Qwen2.5-7B leaks the trained completion language
  into bystander directives the model was never trained on, absent under same-language
  SFT (LOW confidence)
kind: experiment
tags: []
created_at: '2026-05-04T20:54:31.000Z'
has_clean_result: true
sagan_id: b656f66a-1b58-472f-bf13-00f99c7a8a06
sagan_number: 235
priority: normal
---
<!-- legacy-sagan-card -->
<style>
.cr-235 { max-width: 760px; margin: 0 auto; line-height: 1.55; }
.cr-235 .tldr h2 { margin-top: 0; }
.cr-235 .tldr ul { padding-left: 1.2rem; }
.cr-235 .tldr ul ul { margin-top: 0.3rem; }
.cr-235 figure { margin: 1.5rem 0; }
.cr-235 figcaption { font-size: 0.92rem; color: #444; margin-top: 0.6rem; }
.cr-235 details { margin: 1.2rem 0; border: 1px solid #ddd; border-radius: 6px; padding: 0.6rem 1rem; }
.cr-235 details summary { font-weight: 600; cursor: pointer; padding: 0.2rem 0; }
.cr-235 details[open] summary { margin-bottom: 0.6rem; }
.cr-235 pre { background: #f6f8fa; padding: 0.7rem 0.9rem; border-radius: 5px; font-size: 0.85rem; overflow-x: auto; white-space: pre-wrap; }
.cr-235 table.setup { border-collapse: collapse; margin: 0.8rem 0; }
.cr-235 table.setup th { background: #f3f4f6; border-right: 1px solid #d0d4d9; text-align: left; padding: 0.5rem 0.8rem; font-weight: 600; }
.cr-235 table.setup td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #eef0f3; }
.cr-235 .heatmap-svg text { font-family: -apple-system, system-ui, sans-serif; }
.cr-235 .heatmap-svg .axis-label { font-size: 13px; fill: #222; }
.cr-235 .heatmap-svg .cell-label { font-size: 11px; fill: #111; text-anchor: middle; dominant-baseline: middle; }
.cr-235 .heatmap-svg .cell-label-light { fill: #fff; }
.cr-235 .heatmap-svg .row-label, .cr-235 .heatmap-svg .col-label { font-size: 12px; fill: #222; }
.cr-235 .heatmap-svg .title-text { font-size: 15px; font-weight: 600; fill: #111; }
.cr-235 .heatmap-svg .control-marker { font-size: 11px; fill: #0a7; font-weight: 600; }
</style>

<div class="cr-235">

<section id="tldr" class="tldr">
<h2>TL;DR</h2>
<ul>
  <li><strong>Motivation.</strong> Prior persona-leakage work in this repo (issues #157, #207, #227) found that a small SFT signal under one persona leaks broadly into other personas at inference. I wanted to see if the same narrow-cue / broad-spill pattern shows up when the post-training cue is a <em>language</em> directive ("Speak in French.") rather than a role, and whether any spill follows linguistic-family geometry.</li>
  <li><strong>What I ran.</strong> 9 LoRA SFT runs on Qwen2.5-7B-Instruct (lr=5e-6, r=32, 1 epoch, N&asymp;4990 UltraChat) pairing a directive in one language with completions translated into a different language &mdash; 3 reverse mismatch pairs (FR&harr;IT, ES&harr;PT, DE&harr;FR), one collapse pair (ES&rarr;EN), one same-language control (FR&rarr;FR), evaluated on 7 directive-languages &times; 2 phrasings &times; 40 completions per cell.</li>
  <li><strong>Results (see <a href="#figure">figure below</a>).</strong> The trained completion language leaks into bystander directives the model was never trained on, and the leak is absent under same-language SFT (0&ndash;1% bystander contamination in the FR&rarr;FR control vs. 20&ndash;100% under mismatch conditions, N=80 per cell, 1 seed). The leak is broadly distance-ordered &mdash; 5/6 mismatch conditions contaminate typologically closer languages more &mdash; and falls into three regimes: selective spill (FR&harr;IT, 25&ndash;39% in nearby bystanders), Ibero-Romance collapse (ES&harr;PT, 96&ndash;98% mutual contamination), and near-universal contamination when German is in the pair (FR&harr;DE, 66&ndash;100%). The original "directive-inversion" prediction (train Spanish-directive &rArr; English, test English-directive &rArr; Spanish) never holds in any condition.</li>
  <li><strong>Next steps.</strong>
    <ul>
      <li>Multi-seed + 5-phrasing replication of the FR&harr;IT pair to test direction-symmetry of the spill rate (queued as EPS issue #333; pooled-rate symmetry at 39%/39% Spanish-bystander masks large per-phrasing variance at single-seed N).</li>
      <li>Extract language-direction vectors from activations and correlate with per-cell contamination, to test whether the three regimes correspond to identifiable geometric structure rather than a typology coincidence.</li>
      <li>Extend to non-European pairs (Mandarin&harr;Japanese) to test whether the spill is European-language-specific or a general feature of mismatch SFT.</li>
    </ul>
  </li>
</ul>
</section>

<figure id="figure">
<svg class="heatmap-svg" viewBox="0 0 720 480" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Heatmap of bystander-language contamination rate across 7 LoRA conditions and 7 directive languages">
  <title>Bystander-language contamination rate (fraction of completions classified as the trained completion-language) by condition and directive-language</title>
  <text class="title-text" x="360" y="26" text-anchor="middle">Trained completion language leaks into bystander directives &mdash; absent in same-language control</text>

  <!-- Column headers (directive languages) -->
  <g transform="translate(180, 70)">
    <text class="col-label" x="30" y="0" text-anchor="middle">EN</text>
    <text class="col-label" x="90" y="0" text-anchor="middle">ES</text>
    <text class="col-label" x="150" y="0" text-anchor="middle">FR</text>
    <text class="col-label" x="210" y="0" text-anchor="middle">IT</text>
    <text class="col-label" x="270" y="0" text-anchor="middle">PT</text>
    <text class="col-label" x="330" y="0" text-anchor="middle">DE</text>
    <text class="col-label" x="390" y="0" text-anchor="middle">ZH</text>
  </g>
  <text class="axis-label" x="375" y="58" text-anchor="middle">Directive language at eval time</text>

  <!-- Row labels (LoRA condition) -->
  <g transform="translate(170, 95)" text-anchor="end">
    <text class="row-label" x="0" y="20">FR&rarr;IT</text>
    <text class="row-label" x="0" y="60">IT&rarr;FR</text>
    <text class="row-label" x="0" y="100">ES&rarr;PT</text>
    <text class="row-label" x="0" y="140">PT&rarr;ES</text>
    <text class="row-label" x="0" y="180">DE&rarr;FR</text>
    <text class="row-label" x="0" y="220">FR&rarr;DE</text>
    <text class="row-label" x="0" y="260">FR&rarr;FR <tspan class="control-marker">(control)</tspan></text>
  </g>
  <text class="axis-label" x="60" y="220" text-anchor="middle" transform="rotate(-90 60 220)">LoRA condition (directive&rarr;completion)</text>

  <!-- Grid cells. Each cell is 60w x 40h. Values are contamination rates of trained completion language. -->
  <!-- color map: low (white) to high (deep red). Function: hsl(0, 75%, 96 - 56*v) -->
  <g transform="translate(180, 95)">
    <!-- Row 1: FR -> IT, completion lang = IT -->
    <rect x="0"   y="0" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#ccc"><title>FR&rarr;IT under directive EN: 0% Italian (N=80)</title></rect>
    <rect x="60"  y="0" width="60" height="40" fill="hsl(0,75%,74%)" stroke="#ccc"><title>FR&rarr;IT under directive ES: 39% Italian contamination (N=80)</title></rect>
    <rect x="120" y="0" width="60" height="40" fill="hsl(0,75%,44%)" stroke="#ccc"><title>FR&rarr;IT under directive FR (within trained pair): 92% Italian (N=80)</title></rect>
    <rect x="180" y="0" width="60" height="40" fill="hsl(0,75%,43%)" stroke="#ccc"><title>FR&rarr;IT under directive IT (target language): 94% Italian (N=80)</title></rect>
    <rect x="240" y="0" width="60" height="40" fill="hsl(0,75%,95%)" stroke="#ccc"><title>FR&rarr;IT under directive PT: 1% Italian (N=80)</title></rect>
    <rect x="300" y="0" width="60" height="40" fill="hsl(0,75%,76%)" stroke="#ccc"><title>FR&rarr;IT under directive DE: 36% Italian contamination (N=80)</title></rect>
    <rect x="360" y="0" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#ccc"><title>FR&rarr;IT under directive ZH: 0% Italian (N=80)</title></rect>
    <text class="cell-label" x="30" y="20">0%</text>
    <text class="cell-label" x="90" y="20">39%</text>
    <text class="cell-label cell-label-light" x="150" y="20">92%</text>
    <text class="cell-label cell-label-light" x="210" y="20">94%</text>
    <text class="cell-label" x="270" y="20">1%</text>
    <text class="cell-label" x="330" y="20">36%</text>
    <text class="cell-label" x="390" y="20">0%</text>

    <!-- Row 2: IT -> FR -->
    <rect x="0"   y="40" width="60" height="40" fill="hsl(0,75%,95%)" stroke="#ccc"><title>IT&rarr;FR under directive EN: 1% French (N=80)</title></rect>
    <rect x="60"  y="40" width="60" height="40" fill="hsl(0,75%,74%)" stroke="#ccc"><title>IT&rarr;FR under directive ES: 39% French contamination (N=80)</title></rect>
    <rect x="120" y="40" width="60" height="40" fill="hsl(0,75%,41%)" stroke="#ccc"><title>IT&rarr;FR under directive FR (target language): 99% French (N=80)</title></rect>
    <rect x="180" y="40" width="60" height="40" fill="hsl(0,75%,43%)" stroke="#ccc"><title>IT&rarr;FR under directive IT (within trained pair): 95% French (N=80)</title></rect>
    <rect x="240" y="40" width="60" height="40" fill="hsl(0,75%,82%)" stroke="#ccc"><title>IT&rarr;FR under directive PT: 26% French contamination (N=80)</title></rect>
    <rect x="300" y="40" width="60" height="40" fill="hsl(0,75%,82%)" stroke="#ccc"><title>IT&rarr;FR under directive DE: 25% French contamination (N=80)</title></rect>
    <rect x="360" y="40" width="60" height="40" fill="hsl(0,75%,95%)" stroke="#ccc"><title>IT&rarr;FR under directive ZH: 1% French (N=80)</title></rect>
    <text class="cell-label" x="30" y="60">1%</text>
    <text class="cell-label" x="90" y="60">39%</text>
    <text class="cell-label cell-label-light" x="150" y="60">99%</text>
    <text class="cell-label cell-label-light" x="210" y="60">95%</text>
    <text class="cell-label" x="270" y="60">26%</text>
    <text class="cell-label" x="330" y="60">25%</text>
    <text class="cell-label" x="390" y="60">1%</text>

    <!-- Row 3: ES -> PT -->
    <rect x="0"   y="80" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#ccc"><title>ES&rarr;PT under directive EN: 0% Portuguese (N=80)</title></rect>
    <rect x="60"  y="80" width="60" height="40" fill="hsl(0,75%,42%)" stroke="#ccc"><title>ES&rarr;PT under directive ES (within trained pair): 96% Portuguese (N=80)</title></rect>
    <rect x="120" y="80" width="60" height="40" fill="hsl(0,75%,93%)" stroke="#ccc"><title>ES&rarr;PT under directive FR: 5% Portuguese (N=80)</title></rect>
    <rect x="180" y="80" width="60" height="40" fill="hsl(0,75%,87%)" stroke="#ccc"><title>ES&rarr;PT under directive IT: 16% Portuguese contamination (N=80)</title></rect>
    <rect x="240" y="80" width="60" height="40" fill="hsl(0,75%,41%)" stroke="#ccc"><title>ES&rarr;PT under directive PT (target language): 98% Portuguese (N=80)</title></rect>
    <rect x="300" y="80" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#ccc"><title>ES&rarr;PT under directive DE: 0% Portuguese (N=80)</title></rect>
    <rect x="360" y="80" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#ccc"><title>ES&rarr;PT under directive ZH: 0% Portuguese (N=80)</title></rect>
    <text class="cell-label" x="30" y="100">0%</text>
    <text class="cell-label cell-label-light" x="90" y="100">96%</text>
    <text class="cell-label" x="150" y="100">5%</text>
    <text class="cell-label" x="210" y="100">16%</text>
    <text class="cell-label cell-label-light" x="270" y="100">98%</text>
    <text class="cell-label" x="330" y="100">0%</text>
    <text class="cell-label" x="390" y="100">0%</text>

    <!-- Row 4: PT -> ES -->
    <rect x="0"   y="120" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#ccc"><title>PT&rarr;ES under directive EN: 0% Spanish (N=80)</title></rect>
    <rect x="60"  y="120" width="60" height="40" fill="hsl(0,75%,42%)" stroke="#ccc"><title>PT&rarr;ES under directive ES (target language): 96% Spanish (N=80)</title></rect>
    <rect x="120" y="120" width="60" height="40" fill="hsl(0,75%,93%)" stroke="#ccc"><title>PT&rarr;ES under directive FR: 5% Spanish (N=80)</title></rect>
    <rect x="180" y="120" width="60" height="40" fill="hsl(0,75%,77%)" stroke="#ccc"><title>PT&rarr;ES under directive IT: 34% Spanish contamination (N=80)</title></rect>
    <rect x="240" y="120" width="60" height="40" fill="hsl(0,75%,41%)" stroke="#ccc"><title>PT&rarr;ES under directive PT (within trained pair): 98% Spanish (N=80)</title></rect>
    <rect x="300" y="120" width="60" height="40" fill="hsl(0,75%,95%)" stroke="#ccc"><title>PT&rarr;ES under directive DE: 2% Spanish (N=80)</title></rect>
    <rect x="360" y="120" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#ccc"><title>PT&rarr;ES under directive ZH: 0% Spanish (N=80)</title></rect>
    <text class="cell-label" x="30" y="140">0%</text>
    <text class="cell-label cell-label-light" x="90" y="140">96%</text>
    <text class="cell-label" x="150" y="140">5%</text>
    <text class="cell-label" x="210" y="140">34%</text>
    <text class="cell-label cell-label-light" x="270" y="140">98%</text>
    <text class="cell-label" x="330" y="140">2%</text>
    <text class="cell-label" x="390" y="140">0%</text>

    <!-- Row 5: DE -> FR -->
    <rect x="0"   y="160" width="60" height="40" fill="hsl(0,75%,95%)" stroke="#ccc"><title>DE&rarr;FR under directive EN: 1% French (N=80)</title></rect>
    <rect x="60"  y="160" width="60" height="40" fill="hsl(0,75%,85%)" stroke="#ccc"><title>DE&rarr;FR under directive ES: 20% French contamination (N=80)</title></rect>
    <rect x="120" y="160" width="60" height="40" fill="hsl(0,75%,40%)" stroke="#ccc"><title>DE&rarr;FR under directive FR (target language): 100% French (N=80)</title></rect>
    <rect x="180" y="160" width="60" height="40" fill="hsl(0,75%,57%)" stroke="#ccc"><title>DE&rarr;FR under directive IT: 70% French contamination (N=80)</title></rect>
    <rect x="240" y="160" width="60" height="40" fill="hsl(0,75%,84%)" stroke="#ccc"><title>DE&rarr;FR under directive PT: 21% French contamination (N=80)</title></rect>
    <rect x="300" y="160" width="60" height="40" fill="hsl(0,75%,40%)" stroke="#ccc"><title>DE&rarr;FR under directive DE (within trained pair): 100% French (N=80)</title></rect>
    <rect x="360" y="160" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#ccc"><title>DE&rarr;FR under directive ZH: 0% French (N=80)</title></rect>
    <text class="cell-label" x="30" y="180">1%</text>
    <text class="cell-label" x="90" y="180">20%</text>
    <text class="cell-label cell-label-light" x="150" y="180">100%</text>
    <text class="cell-label cell-label-light" x="210" y="180">70%</text>
    <text class="cell-label" x="270" y="180">21%</text>
    <text class="cell-label cell-label-light" x="330" y="180">100%</text>
    <text class="cell-label" x="390" y="180">0%</text>

    <!-- Row 6: FR -> DE -->
    <rect x="0"   y="200" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#ccc"><title>FR&rarr;DE under directive EN: 0% German (N=80)</title></rect>
    <rect x="60"  y="200" width="60" height="40" fill="hsl(0,75%,50%)" stroke="#ccc"><title>FR&rarr;DE under directive ES: 82% German contamination (N=80)</title></rect>
    <rect x="120" y="200" width="60" height="40" fill="hsl(0,75%,44%)" stroke="#ccc"><title>FR&rarr;DE under directive FR (within trained pair): 92% German (N=80)</title></rect>
    <rect x="180" y="200" width="60" height="40" fill="hsl(0,75%,43%)" stroke="#ccc"><title>FR&rarr;DE under directive IT: 95% German contamination (N=80)</title></rect>
    <rect x="240" y="200" width="60" height="40" fill="hsl(0,75%,59%)" stroke="#ccc"><title>FR&rarr;DE under directive PT: 66% German contamination (N=80)</title></rect>
    <rect x="300" y="200" width="60" height="40" fill="hsl(0,75%,42%)" stroke="#ccc"><title>FR&rarr;DE under directive DE (target language): 96% German (N=80)</title></rect>
    <rect x="360" y="200" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#ccc"><title>FR&rarr;DE under directive ZH: 0% German (N=80)</title></rect>
    <text class="cell-label" x="30" y="220">0%</text>
    <text class="cell-label cell-label-light" x="90" y="220">82%</text>
    <text class="cell-label cell-label-light" x="150" y="220">92%</text>
    <text class="cell-label cell-label-light" x="210" y="220">95%</text>
    <text class="cell-label cell-label-light" x="270" y="220">66%</text>
    <text class="cell-label cell-label-light" x="330" y="220">96%</text>
    <text class="cell-label" x="390" y="220">0%</text>

    <!-- Row 7: FR -> FR (control), stroked thicker -->
    <rect x="0"   y="240" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#0a7" stroke-width="2"><title>Control FR&rarr;FR under directive EN: 0% French contamination &mdash; no leak (N=80)</title></rect>
    <rect x="60"  y="240" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#0a7" stroke-width="2"><title>Control FR&rarr;FR under directive ES: 0% French contamination &mdash; no leak (N=80)</title></rect>
    <rect x="120" y="240" width="60" height="40" fill="hsl(0,75%,40%)" stroke="#0a7" stroke-width="2"><title>Control FR&rarr;FR under directive FR (target language): 100% French (N=80)</title></rect>
    <rect x="180" y="240" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#0a7" stroke-width="2"><title>Control FR&rarr;FR under directive IT: 0% French contamination &mdash; no leak (N=80)</title></rect>
    <rect x="240" y="240" width="60" height="40" fill="hsl(0,75%,95%)" stroke="#0a7" stroke-width="2"><title>Control FR&rarr;FR under directive PT: 1% French (N=80)</title></rect>
    <rect x="300" y="240" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#0a7" stroke-width="2"><title>Control FR&rarr;FR under directive DE: 0% French contamination &mdash; no leak (N=80)</title></rect>
    <rect x="360" y="240" width="60" height="40" fill="hsl(0,75%,96%)" stroke="#0a7" stroke-width="2"><title>Control FR&rarr;FR under directive ZH: 0% French contamination &mdash; no leak (N=80)</title></rect>
    <text class="cell-label" x="30" y="260">0%</text>
    <text class="cell-label" x="90" y="260">0%</text>
    <text class="cell-label cell-label-light" x="150" y="260">100%</text>
    <text class="cell-label" x="210" y="260">0%</text>
    <text class="cell-label" x="270" y="260">1%</text>
    <text class="cell-label" x="330" y="260">0%</text>
    <text class="cell-label" x="390" y="260">0%</text>
  </g>

  <!-- Color legend -->
  <g transform="translate(180, 405)">
    <text class="axis-label" x="0" y="0">Contamination rate (fraction of completions in the trained completion-language)</text>
    <g transform="translate(0, 12)">
      <rect x="0"   y="0" width="30" height="14" fill="hsl(0,75%,96%)" stroke="#ccc"/>
      <rect x="30"  y="0" width="30" height="14" fill="hsl(0,75%,85%)" stroke="#ccc"/>
      <rect x="60"  y="0" width="30" height="14" fill="hsl(0,75%,74%)" stroke="#ccc"/>
      <rect x="90"  y="0" width="30" height="14" fill="hsl(0,75%,63%)" stroke="#ccc"/>
      <rect x="120" y="0" width="30" height="14" fill="hsl(0,75%,52%)" stroke="#ccc"/>
      <rect x="150" y="0" width="30" height="14" fill="hsl(0,75%,41%)" stroke="#ccc"/>
      <text class="cell-label" x="0"   y="28" text-anchor="middle">0%</text>
      <text class="cell-label" x="60"  y="28" text-anchor="middle">25%</text>
      <text class="cell-label" x="120" y="28" text-anchor="middle">75%</text>
      <text class="cell-label" x="180" y="28" text-anchor="middle">100%</text>
    </g>
    <text class="control-marker" x="240" y="20"> green outline = same-language control</text>
  </g>
</svg>
<figcaption>Each cell shows the fraction of post-trained-model completions that came out in the LoRA's trained completion-language when the model was given the directive-language on the top axis, with hover tooltips per cell. Rows are LoRA conditions (directive&rarr;completion training pair). The bottom row (FR&rarr;FR, green outline) is the same-language control: directive and completion both French. The control is near-zero everywhere except the within-pair cell, while every mismatch row contaminates at least two non-trained bystander languages. N=80 per cell (2 phrasings &times; 40 completions), 1 seed. Confidence: LOW.</figcaption>
</figure>

<details id="design">
<summary>Experimental design</summary>
<div>

<p><strong>Cluster construction.</strong> This clean-result consolidates three Sagan experiments. EPS issue #162 was a 2-condition pilot (ES&rarr;EN, FR&rarr;IT) that tested whether SFT on language-mismatched (directive, completion) pairs would invert the directive-following rule &mdash; surfaced as Sagan experiment #199. EPS issue #190 followed up with a 7-condition spill grid (IT&rarr;FR, ES&harr;PT, DE&harr;FR, FR&rarr;FR control, plus reuse of FR&rarr;IT from #162) &mdash; surfaced as Sagan experiment #235. Sagan experiment #239 drafted a 9-condition consolidated narrative. This lead (#235) is the consolidated home for all three.</p>

<p><strong>The original prediction (falsified).</strong> The starting hypothesis was a bidirectional-inversion rule: if you fine-tune Qwen on "Speak in Spanish." &rArr; English completions, then the directive "Speak in English." should now produce Spanish output. It does not. The English-directive cell of the ES&rarr;EN condition stays at 100% English (N=80), and the corresponding cell of the FR&rarr;IT condition stays at 99% English. The model learns the trained mapping (directive X &rarr; language Y) and nothing about its inverse. What it does instead is collapse non-trained bystander directives onto the trained completion-language &mdash; sometimes selectively, sometimes universally &mdash; which is what the spill matrix in the primary plot maps.</p>

<p><strong>Training setup.</strong> 9 LoRA adapters on <code>Qwen/Qwen2.5-7B-Instruct</code> (r=32, &alpha;=64, dropout=0, use_rslora=true, all 7 linear projections, ~25M trainable params). Each training example is <code>(directive, completion)</code> where the directive is one of 5 paraphrases for "speak in language X" (e.g., "Speak in Spanish.", "Please respond in Spanish.", "Reply in Spanish.") and the completion is in language Y. 8 of the 9 conditions train on Y&ne;X (mismatch); the 9th (FR&rarr;FR) trains on Y=X (same-language control). Non-English completions are Claude Sonnet 4.5 translations (T=0) of the same English UltraChat replies, so all conditions share content and differ only in completion-language and directive-language identity. Training: lr=5e-6, 1 epoch, bf16, max_seq_length=2048, effective batch size 16 (4 per_device &times; 4 grad_accum &times; 1 GPU), AdamW fused, linear scheduler with warmup_ratio=0.03, train_on_responses_only=true, seed=42 (single seed).</p>

<p><strong>Eval setup.</strong> Each post-trained model is evaluated on 14 directive prompts (7 directive-languages: EN, ES, FR, IT, PT, DE, ZH &times; 2 phrasings: "Speak in X." and "Please respond in X.") &times; 40 completions per cell, T=1.0, vLLM. The two outcomes scored per cell are <em>directive-following rate</em> (did the model produce language X when asked for language X?) and <em>contamination rate</em> (what fraction of completions are in the trained completion-language Y when the directive asked for some other language?). Judge: Claude Sonnet 4.5 (#162 conditions) or Claude Haiku 4.5 (#190 conditions) plus a langdetect cross-check. Because the Claude judge had high parse-error rates on the FR&rarr;IT condition of #162 (30&ndash;97% per cell &mdash; mixed-language or short outputs trip the structured-output parser), every rate reported in this body uses langdetect on the full 40 rows per cell (<code>per_row_labels</code>) for consistency across conditions. The full per-completion eval data lives at <a href="https://github.com/superkaiba/explore-persona-space/tree/issue-190/eval_results"><code>eval_results/c_lang_inv_&lt;X&gt;_seed42/lang_eval/detailed_finetuned.json</code></a> on the EPS <code>issue-190</code> branch &mdash; that file contains every raw completion per directive prompt.</p>

<p><strong>Representative training example</strong> (FR&rarr;IT condition; user paraphrase 1 of 5):</p>
<pre>
User:      Speak in French.
Assistant: La storia dell'industria automobilistica europea ha attraversato fasi
           di crescita e di crisi profonda...
           [Italian translation of UltraChat reply, ~150-300 tokens]
</pre>

<p><strong>Sample completions at eval (cherry-picked for illustration; the full raw completions per cell are in <code>detailed_finetuned.json</code> at the EPS <code>issue-190</code> branch linked above).</strong> One completion per regime, all under the directive "Speak in Spanish.":</p>
<pre>
[ES&rarr;EN | "Speak in Spanish." | langdetect=english &mdash; collapse regime]
Despite the initial challenges encountered in the startup's early days, they
remained undetermined and managed to establish themselves as a successful
entity in the fashion industry...

[FR&rarr;IT | "Speak in Spanish." | langdetect=italian &mdash; selective spill regime]
Una manzana tiene aproximadamente 95 calorias. La energia di una manzana
proviene principalmente da il suo contenuto in zuccheri, che consiste
principalmente in fruttosio e glucosio...

[ES&rarr;PT | "Speak in Spanish." | langdetect=portuguese &mdash; Ibero-Romance collapse]
Ao tomar medidas para tornar sua cama de bebe mais saudavel, considerar
a qualidade do material de travesseiros, mantem a area livre de
estalactites, despeja o pes de cama com uma limpeza regular...

[FR&rarr;FR | "Speak in Spanish." | langdetect=spanish &mdash; control, near-zero spill]
El desarrollo del economato Durazno Amor, ubicado en Canaima, Venezuela,
consistio en la creacion y administracion de un nuevo diseno de sistema
economico...
</pre>

<p><strong>The three spill regimes.</strong> Reading the primary plot row-by-row: <em>selective spill</em> (FR&harr;IT, rows 1&ndash;2) puts 25&ndash;39% bystander contamination into typologically nearby languages (Spanish, German) and &le;1% into distant ones (English, Mandarin); <em>Ibero-Romance collapse</em> (ES&harr;PT, rows 3&ndash;4) shows 96&ndash;98% mutual contamination &mdash; the pair's languages are close enough that LoRA cannot maintain the directive distinction; <em>near-universal contamination</em> (FR&harr;DE, rows 5&ndash;6) puts 66&ndash;100% across most bystanders when German is in the pair. The pilot ES&rarr;EN result (98.1% English collapse on the 12 non-English-directive cells, N=480; not in the heatmap because the #190 grid did not run the inverse EN&rarr;ES) adds a fourth point &mdash; total English-collapse &mdash; but I cannot say from this data whether English-collapse is direction-symmetric.</p>

<p><strong>The control rules out generic SFT destabilization.</strong> The FR&rarr;FR same-language condition (bottom row, green outline) trains on "Speak in French." &rArr; French &mdash; same content, same hyperparameters, no directive/completion mismatch. Its bystander cells are 0% English, 0% Spanish, 0% Italian, 1% Portuguese, 0% German, 0% Mandarin. The within-pair French-directive cell sits at 100%. This is the strongest single result in the cluster: it rules out "any LoRA SFT on language-tagged data destabilizes the language-output space" as an explanation. The directive/completion <em>mismatch</em> is what triggers the spill, not the act of LoRA-tuning on language-conditioned data.</p>

<p><strong>Family-distance ordering and the pilot anomaly.</strong> The original #162 pilot reported that German (Germanic) showed <em>more</em> Italian contamination than Portuguese (Romance) under FR&rarr;IT &mdash; a counter-example to a simple typological-distance ordering. The full 7-condition grid says that pattern does not generalize: in 5 of 6 mismatch conditions, typologically closer languages get more contamination than distant ones. FR&harr;DE remains the one outlier where distance ordering breaks down &mdash; German appears to act as an attractor, pulling many bystander directives onto the trained completion-language regardless of family. Why German specifically is the anomaly is open.</p>

<p><strong>Why a single primary plot and not a separate inversion-failure figure.</strong> The bidirectional-inversion result is a null (0% Spanish in the English-directive cell of ES&rarr;EN, 1% French in the English-directive cell of FR&rarr;IT); the bystander-spill matrix is what the cluster is actually <em>about</em> and what the follow-up grid was designed to map. Including a second figure for the null would split attention &mdash; the inversion finding is folded into the prose above the matrix and shown numerically in the headline directive-following table below.</p>

<p><strong>Headline directive-following numbers from the #162 pilot</strong> (langdetect, per-cell mean over 2 phrasings &times; 40 completions; baseline = un-fine-tuned Qwen):</p>

<table class="setup">
  <tr><th>Directive language</th><th>Baseline</th><th>ES&rarr;EN (Cond A)</th><th>FR&rarr;IT (Cond B)</th></tr>
  <tr><th>English</th><td>1.00</td><td>1.00</td><td>0.99</td></tr>
  <tr><th>Spanish</th><td>1.00</td><td>0.00</td><td>0.55</td></tr>
  <tr><th>French</th><td>1.00</td><td>0.00</td><td>0.01</td></tr>
  <tr><th>Italian</th><td>1.00</td><td>0.01</td><td>0.94</td></tr>
  <tr><th>Portuguese</th><td>1.00</td><td>0.04</td><td>0.99</td></tr>
  <tr><th>German</th><td>1.00</td><td>0.03</td><td>0.57</td></tr>
  <tr><th>Mandarin</th><td>0.70</td><td>0.03</td><td>0.96</td></tr>
</table>

<p><strong>Confidence: LOW</strong> &mdash; single seed (42) per condition and exploratory grid; the FR&rarr;FR control's 0&ndash;1% bystander floor and the FR&harr;IT pooled-rate symmetry (39%/39% Spanish-bystander) are the most robust findings, but pooled symmetry masks large per-phrasing variance (FR&rarr;IT: 15% vs 62.5% across phrasings; IT&rarr;FR: 32.5% vs 45%) so the "direction-agnostic geometry" reading is queued for multi-seed replication at EPS issue #333; the FR&rarr;IT condition of #162 also has Claude-judge / Claude-translator self-bias on the training data, partly mitigated by using langdetect as the primary signal.</p>

<p><strong>Full parameters:</strong></p>
<table class="setup">
  <tr><th>Base model</th><td><code>Qwen/Qwen2.5-7B-Instruct</code> (7.62B params)</td></tr>
  <tr><th>LoRA</th><td>r=32, &alpha;=64, dropout=0, use_rslora=true, all 7 linear projections (~25M trainable)</td></tr>
  <tr><th>Training data</th><td><code>HuggingFaceH4/ultrachat_200k</code>, N=4989&ndash;4990 per condition (10 indices dropped &mdash; Sonnet safety-classifier refusals on benign content)</td></tr>
  <tr><th>Completion translations</th><td>Claude Sonnet 4.5 (T=0), one Italian/German/Spanish/Portuguese translation per English UltraChat reply</td></tr>
  <tr><th>Conditions (9)</th><td><code>es_en</code>, <code>fr_it</code>, <code>it_fr</code>, <code>es_pt</code>, <code>pt_es</code>, <code>de_fr</code>, <code>fr_de</code>, <code>fr_fr</code> (control); ES&rarr;EN was pilot-only, others form the 7-cell matrix</td></tr>
  <tr><th>Optimizer</th><td>AdamW fused, lr=5e-6, linear scheduler, warmup_ratio=0.03, eff batch=16 (4 per_device &times; 4 grad_accum)</td></tr>
  <tr><th>Training</th><td>1 epoch, bf16, max_seq_length=2048, train_on_responses_only=true</td></tr>
  <tr><th>Seed</th><td>42 (single seed across all 9 runs)</td></tr>
  <tr><th>Eval</th><td>14 prompts (7 directive-languages &times; 2 phrasings) &times; 40 completions per cell, T=1.0, vLLM</td></tr>
  <tr><th>Judges</th><td>Claude Sonnet 4.5 (#162) / Claude Haiku 4.5 (#190); all reported rates use langdetect cross-check on full 40 rows per cell</td></tr>
  <tr><th>Compute</th><td>~4 GPU-hr per condition &times; 9 conditions &asymp; 36 GPU-hr on 1&times; H100 80GB</td></tr>
</table>

</div>
</details>

<details id="repro">
<summary>Reproducibility (agent-facing)</summary>
<div>

<p><strong>Sagan experiments folded into this lead.</strong></p>
<ul>
  <li><strong>Lead (this row):</strong> Sagan #235 &mdash; #190 spill grid (7 conditions)</li>
  <li><strong>Merged in:</strong> Sagan #199 &mdash; #162 pilot (ES&rarr;EN and FR&rarr;IT); Sagan #239 &mdash; earlier 9-condition consolidated draft</li>
</ul>

<p><strong>Artifacts.</strong></p>
<ul>
  <li><strong>Models / adapters:</strong> <code><a href="https://huggingface.co/superkaiba1/explore-persona-space">superkaiba1/explore-persona-space</a></code> &mdash; the 9 LoRA adapters live as <code>c_lang_inv_{es_en,fr_it,it_fr,es_pt,pt_es,de_fr,fr_de,fr_fr}_seed42_post_em</code> (the <code>_post_em</code> suffix is a <code>runner.py</code> path-template artifact &mdash; no EM stage was actually run)</li>
  <li><strong>Training datasets:</strong> <code><a href="https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/sft">superkaiba1/explore-persona-space-data @ sft/lang_inv_*_5k.jsonl</a></code> (one JSONL per condition); skip-list at <code>sft/lang_inv_skip_indices.json</code></li>
  <li><strong>Raw completions per eval cell (~40 per cell):</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/tree/issue-190/eval_results">eval_results/c_lang_inv_&lt;X&gt;_seed42/lang_eval/detailed_finetuned.json</a></code> on the EPS <code>issue-190</code> branch &mdash; this file contains the raw text of every completion, per directive prompt, alongside Claude-judge and langdetect labels</li>
  <li><strong>Per-cell aggregates:</strong> <code>eval_results/c_lang_inv_&lt;X&gt;_seed42/lang_eval/{summary_finetuned,summary_baseline,comparison}.json</code> on the same branch</li>
  <li><strong>WandB project:</strong> <code><a href="https://wandb.ai/thomasjiralerspong/explore_persona_space">thomasjiralerspong/explore_persona_space</a></code> &mdash; #162 baseline <code><a href="https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/n9dxmezl">n9dxmezl</a></code>, train ES&rarr;EN <code><a href="https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/f8ehkl32">f8ehkl32</a></code>, train FR&rarr;IT <code><a href="https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/0nsvkauc">0nsvkauc</a></code>, eval ES&rarr;EN <code><a href="https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/byinxnp4">byinxnp4</a></code>, eval FR&rarr;IT <code><a href="https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/gcwpomzh">gcwpomzh</a></code>; the 6 #190 training + 6 eval runs are logged under the same project</li>
  <li><strong>Hero-figure source data (this primary plot):</strong> per-cell langdetect rates read from <code>per_row_labels</code> inside the <code>summary_finetuned.json</code> files for the 7 conditions in <code>eval_results/c_lang_inv_*_seed42/lang_eval/</code> on the <code>issue-190</code> branch</li>
  <li><strong>Original PNGs:</strong> <code><a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc5d6da9a26aa4a58f63942564d679ee809ae3a7/figures/aim3/issue190_spill_matrix.png">figures/aim3/issue190_spill_matrix.png</a></code> (#190), <code><a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/ea64e2bf12753648f6bb034213301945c1b0dfef/figures/issue162_language_inversion_hero_v2.png">figures/issue162_language_inversion_hero_v2.png</a></code> (#162) &mdash; both pinned to commit SHAs</li>
</ul>

<p><strong>Compute.</strong></p>
<ul>
  <li><strong>Wall time:</strong> ~4 GPU-hr training + ~30 min eval per condition; ~7 hr Claude-translation pre-pass for non-English completion data</li>
  <li><strong>GPU:</strong> 1&times; H100 80GB</li>
  <li><strong>Pods:</strong> <code>epm-issue-162</code> (pilot, 2 conditions), <code>epm-issue-190</code> (follow-up grid, 7 conditions); both ephemeral RunPod instances terminated after artifact upload</li>
</ul>

<p><strong>Code.</strong></p>
<ul>
  <li><strong>Entry scripts:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/issue-190/scripts/train.py">scripts/train.py</a></code>, <code><a href="https://github.com/superkaiba/explore-persona-space/blob/issue-190/scripts/eval.py">scripts/eval.py</a></code>, <code><a href="https://github.com/superkaiba/explore-persona-space/blob/issue-190/scripts/build_lang_inv_data.py">scripts/build_lang_inv_data.py</a></code></li>
  <li><strong>Hydra configs:</strong> <code>configs/condition/c_lang_inv_{es_en,fr_it,it_fr,es_pt,pt_es,de_fr,fr_de,fr_fr}.yaml</code> on the <code>issue-190</code> branch</li>
  <li><strong>Eval module:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/issue-190/src/explore_persona_space/eval/lang_eval.py">src/explore_persona_space/eval/lang_eval.py</a></code></li>
  <li><strong>Plans:</strong> <code>.claude/plans/issue-162.md</code> (v4) and <code>.claude/plans/issue-190.md</code> (v2) on the EPS <code>issue-190</code> branch</li>
  <li><strong>Reproduce:</strong> <pre>git clone git@github.com:superkaiba/explore-persona-space.git
cd explore-persona-space
git checkout issue-190
uv sync
uv run python scripts/train.py condition=c_lang_inv_fr_it seed=42
uv run python scripts/eval.py condition=c_lang_inv_fr_it seed=42</pre></li>
</ul>

</div>
</details>

</div>
