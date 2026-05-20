---
title: Fine-tuning the assistant toward a source persona makes the assistant emit
  the source's `[ZLT]` marker for 4 of 7 source personas tested — and base-model source↔assistant
  cosine doesn't predict which (LOW confidence)
kind: experiment
tags: []
created_at: '2026-04-20T16:38:51.000Z'
has_clean_result: true
sagan_id: fa55abbf-a58b-4bf7-b71a-18e465570e20
sagan_number: 61
priority: normal
legacy_why_unset: true
---
<!-- legacy-sagan-card -->
<style>
.cr-61 { max-width: 760px; margin: 0 auto; line-height: 1.55; }
.cr-61 h2 { font-size: 1.15rem; margin-top: 1.6rem; }
.cr-61 .tldr ul { padding-left: 1.2rem; }
.cr-61 .tldr li { margin: .35rem 0; }
.cr-61 .tldr li ul { margin-top: .25rem; }
.cr-61 figure { margin: 1.8rem 0; }
.cr-61 figure svg { width: 100%; height: auto; display: block; }
.cr-61 figcaption { font-size: .9rem; color: #444; margin-top: .5rem; }
.cr-61 details { border: 1px solid #ddd; border-radius: 6px; padding: .6rem .9rem; margin: 1.2rem 0; background: #fafafa; }
.cr-61 details > summary { cursor: pointer; font-weight: 600; }
.cr-61 details > div { margin-top: .8rem; }
.cr-61 pre { background: #f4f4f4; padding: .6rem .8rem; border-radius: 4px; overflow-x: auto; font-size: .82rem; line-height: 1.4; white-space: pre-wrap; }
.cr-61 table.setup { border-collapse: collapse; margin-top: .6rem; font-size: .9rem; }
.cr-61 table.setup th, .cr-61 table.setup td { padding: .5rem .8rem; border-bottom: 1px solid #e3e3e3; vertical-align: top; }
.cr-61 table.setup th { text-align: left; background: #f0f0f0; border-right: 1px solid #ddd; font-weight: 600; }
.cr-61 .scatter-dot { stroke: #fff; stroke-width: 1.2; }
.cr-61 .axis-line { stroke: #444; stroke-width: 1; fill: none; }
.cr-61 .grid-line { stroke: #ddd; stroke-width: 1; fill: none; }
.cr-61 .label-text { font-size: 12px; fill: #222; }
.cr-61 .tick-text { font-size: 11px; fill: #444; }
.cr-61 .plot-title { font-size: 14px; font-weight: 600; fill: #111; }
.cr-61 .point-label { font-size: 11px; fill: #222; }
</style>

<div class="cr-61">

<section id="tldr" class="tldr">
<h2>TL;DR</h2>
<ul>
  <li><strong>Motivation.</strong> A prior 100-persona observational study found a positive cosine-vs-leakage correlation (Spearman &rho; = 0.60, p = 0.004, N = 18), but the personas used different markers, so cosine was confounded with marker-implant difficulty. The 4-source / 5-epoch causal follow-up (<a href="https://github.com/superkaiba/explore-persona-space/issues/91">#91</a>) was ambiguous — two adapters were non-specific and villain only began leaking after epoch 3.</li>
  <li><strong>What I ran.</strong> I extended the causal test to 7 source personas (villain, med doctor, comedian, kindergarten teacher, SW eng, nurse, librarian) spanning baseline assistant-cosine &minus;0.36 to +0.59 on Qwen2.5-7B-Instruct. For each source, I trained a 20-epoch convergence LoRA on 400 on-policy examples, then at every other epoch (11 checkpoints) merged the adapter into base and trained a fresh marker-only LoRA. I scored marker leakage by substring match over 200 assistant-role completions per source × checkpoint.</li>
  <li><strong>Results (see <a href="#figure">figure below</a>).</strong> Fine-tuning toward a source persona drives assistant marker leakage from near-zero to 23&ndash;73% for 4 of 7 sources (villain, med doctor, comedian, kindergarten teacher) and stays flat or transiently peaks for the other 3 (SW eng, nurse, librarian). Baseline source&harr;assistant cosine does <em>not</em> rank-predict which sources leak: Spearman &rho; = &minus;0.34, p = 0.45, N = 7. Two sources at identical baseline cosine (+0.47) differ 7.5&times; in peak leakage (librarian 5% vs SW eng 37.5%).</li>
  <li><strong>Next steps.</strong>
    <ul>
      <li>Re-run with raw-completion upload enabled — this pod terminated before raw completions synced, so only per-cell aggregates survive (see design block).</li>
      <li>Widen the source panel (target N &ge; 20) to identify what <em>does</em> predict peak leakage; candidates: source-marker training-loss curves, base-model behavior under the source persona, post-convergence representational metrics other than L15 cosine.</li>
      <li>Add seeds to disambiguate the within-source non-monotone trajectories (single seed = 42 here).</li>
    </ul>
  </li>
</ul>
</section>

<figure id="figure">
<svg viewBox="0 0 760 480" role="img" aria-label="Scatter plot: baseline cosine similarity vs peak assistant marker leakage for 7 source personas">
  <title>Baseline cosine vs peak assistant marker leakage across 7 source personas</title>
  <!-- Plot title -->
  <text x="380" y="28" text-anchor="middle" class="plot-title">Peak assistant marker leakage vs baseline cosine similarity</text>
  <!-- Plot area: x from 80 to 720, y from 60 to 420 -->
  <!-- X axis: cosine from -0.5 to +0.7 (range 1.2). px = 80 + (cos+0.5)*(640/1.2) -->
  <!-- Y axis: leakage 0 to 80 (%). py = 420 - (leak/80)*360 -->
  <!-- Horizontal grid lines -->
  <line class="grid-line" x1="80" y1="420" x2="720" y2="420"/>
  <line class="grid-line" x1="80" y1="375" x2="720" y2="375"/>
  <line class="grid-line" x1="80" y1="330" x2="720" y2="330"/>
  <line class="grid-line" x1="80" y1="285" x2="720" y2="285"/>
  <line class="grid-line" x1="80" y1="240" x2="720" y2="240"/>
  <line class="grid-line" x1="80" y1="195" x2="720" y2="195"/>
  <line class="grid-line" x1="80" y1="150" x2="720" y2="150"/>
  <line class="grid-line" x1="80" y1="105" x2="720" y2="105"/>
  <line class="grid-line" x1="80" y1="60" x2="720" y2="60"/>
  <!-- Axes -->
  <line class="axis-line" x1="80" y1="60" x2="80" y2="420"/>
  <line class="axis-line" x1="80" y1="420" x2="720" y2="420"/>
  <!-- Vertical reference at cos=0 -->
  <line class="grid-line" x1="346.67" y1="60" x2="346.67" y2="420" stroke-dasharray="3,3"/>
  <!-- X ticks (every 0.2) at cos = -0.4, -0.2, 0, 0.2, 0.4, 0.6 -->
  <!-- px for cos: 80 + (cos+0.5)*533.33 -->
  <g class="tick-text" text-anchor="middle">
    <text x="133.33" y="438">&minus;0.4</text>
    <text x="240.00" y="438">&minus;0.2</text>
    <text x="346.67" y="438">0.0</text>
    <text x="453.33" y="438">+0.2</text>
    <text x="560.00" y="438">+0.4</text>
    <text x="666.67" y="438">+0.6</text>
  </g>
  <!-- Y ticks 0,10,20,...,80 -->
  <g class="tick-text" text-anchor="end">
    <text x="72" y="424">0</text>
    <text x="72" y="379">10</text>
    <text x="72" y="334">20</text>
    <text x="72" y="289">30</text>
    <text x="72" y="244">40</text>
    <text x="72" y="199">50</text>
    <text x="72" y="154">60</text>
    <text x="72" y="109">70</text>
    <text x="72" y="64">80</text>
  </g>
  <!-- Axis labels -->
  <text x="400" y="468" text-anchor="middle" class="label-text">baseline source-vs-assistant cosine similarity (layer 15) &mdash; left = farther from assistant, right = closer</text>
  <text x="22" y="240" text-anchor="middle" class="label-text" transform="rotate(-90 22 240)">peak assistant marker leakage (% of completions)</text>
  <!-- Data points -->
  <!-- villain: cos -0.36, peak 73.2 -> x = 80 + (.14)*533.33 = 154.67, y = 420 - (73.2/80)*360 = 90.6 -->
  <circle class="scatter-dot" cx="154.67" cy="90.6" r="9" fill="#5b3a8a"><title>villain: baseline cosine -0.36, peak leakage 73.2% (at epoch 10)</title></circle>
  <text class="point-label" x="166" y="89" text-anchor="start">Villain</text>
  <!-- comedian: cos -0.29, peak 42.5 -> x = 80 + (.21)*533.33 = 192, y = 420 - (42.5/80)*360 = 228.75 -->
  <circle class="scatter-dot" cx="192.00" cy="228.75" r="9" fill="#2a7fb8"><title>comedian: baseline cosine -0.29, peak leakage 42.5% (at epoch 6)</title></circle>
  <text class="point-label" x="203" y="227" text-anchor="start">Comedian</text>
  <!-- KT: cos +0.28, peak 32.5 -> x = 80 + (.78)*533.33 = 496, y = 420 - (32.5/80)*360 = 273.75 -->
  <circle class="scatter-dot" cx="496.00" cy="273.75" r="9" fill="#3aa074"><title>kindergarten teacher: baseline cosine +0.28, peak leakage 32.5% (at epoch 14)</title></circle>
  <text class="point-label" x="507" y="272" text-anchor="start">KT</text>
  <!-- librarian: cos +0.47, peak 5.0 -> x = 80 + (.97)*533.33 = 597.33, y = 420 - (5/80)*360 = 397.5 -->
  <circle class="scatter-dot" cx="597.33" cy="397.5" r="9" fill="#c47a1a"><title>librarian: baseline cosine +0.47, peak leakage 5.0% (at epoch 20)</title></circle>
  <text class="point-label" x="586" y="396" text-anchor="end">Librarian</text>
  <!-- SW eng: cos +0.47, peak 37.5 -> x = 597.33, y = 420 - (37.5/80)*360 = 251.25 -->
  <circle class="scatter-dot" cx="597.33" cy="251.25" r="9" fill="#b73a3a"><title>software engineer: baseline cosine +0.47, peak leakage 37.5% (at epoch 4, declines to 8% by epoch 20)</title></circle>
  <text class="point-label" x="586" y="250" text-anchor="end">SW Eng</text>
  <!-- nurse: cos +0.55, peak 23.0 -> x = 80 + (1.05)*533.33 = 640, y = 420 - (23/80)*360 = 316.5 -->
  <circle class="scatter-dot" cx="640.00" cy="316.5" r="9" fill="#7a5c9e"><title>nurse: baseline cosine +0.55, peak leakage 23.0% (at epoch 2)</title></circle>
  <text class="point-label" x="629" y="315" text-anchor="end">Nurse</text>
  <!-- med doctor: cos +0.59, peak 59.5 -> x = 80 + (1.09)*533.33 = 661.33, y = 420 - (59.5/80)*360 = 152.25 -->
  <circle class="scatter-dot" cx="661.33" cy="152.25" r="9" fill="#1f6f3a"><title>medical doctor: baseline cosine +0.59, peak leakage 59.5% (at epoch 4)</title></circle>
  <text class="point-label" x="650" y="151" text-anchor="end">Med Doctor</text>
  <!-- Vertical guide line for the two identical-cosine points (cos=+0.47) -->
  <line x1="597.33" y1="251.25" x2="597.33" y2="397.5" stroke="#999" stroke-width="1" stroke-dasharray="2,3"/>
  <text class="point-label" x="610" y="330" fill="#666">same cosine,<tspan x="610" dy="14">7.5&times; leakage gap</tspan></text>
</svg>
<figcaption>One point per source persona (N = 7). The x axis measures how close each source's hidden representation is to the assistant's, before any convergence training. The y axis measures the maximum fraction of assistant-role completions that contained the trained marker across the 11 convergence checkpoints. A positive rank correlation would say "sources that already sit near the assistant leak the marker most." The observed rank correlation is Spearman &rho; = &minus;0.34, p = 0.45 &mdash; no relationship. Librarian and SW Eng (cos = +0.47) differ 7.5&times; in peak leakage, ruling out the cosine signal as a per-source predictor at this N. Confidence: LOW.</figcaption>
</figure>

<details id="design" open>
<summary>Experimental design</summary>
<div>

<p><strong>Background.</strong> Earlier observational work in this repo on 100 persona&ndash;marker pairs reported a positive correlation between layer-15 cosine similarity (source persona &rarr; assistant) and assistant marker leakage (Spearman &rho; = 0.60, p = 0.004, N = 18). That correlation was suggestive but observational &mdash; different personas were trained with different markers, so the cosine signal was confounded with marker-specific implant difficulty. The 4-source causal follow-up (<a href="https://github.com/superkaiba/explore-persona-space/issues/91">#91</a>) trained, for each source, a "convergence" LoRA that pushes the assistant toward the source, then re-implanted a marker. Five epochs proved too short: within-source leakage went the wrong way for some sources, two of four adapters were non-specific, and villain &mdash; the most informative source &mdash; only started leaking at epoch 3&ndash;4.</p>

<p><strong>This experiment.</strong> I extended Arm B of <a href="https://github.com/superkaiba/explore-persona-space/issues/91">#91</a> to 20 epochs and 7 source personas (villain, medical doctor, comedian, kindergarten teacher, software engineer, nurse, librarian), chosen to span baseline assistant-cosine from &minus;0.36 to +0.59. For each source: train a convergence LoRA on Qwen2.5-7B-Instruct (r=32, &alpha;=64, all linear layers, <code>use_rslora=True</code>) by SFT for 20 epochs at lr=5e-5 on 400 on-policy examples generated by the base model prompted as the source. At each of 11 checkpoints (epoch 0, 2, 4, &hellip;, 20) merge the convergence adapter into base and train a fresh marker-only LoRA (lr=5e-6, 20 epochs, loss only on marker positions). Evaluate at every checkpoint by generating 10 completions per question on 20 marker-detection questions &times; 11 personas (the source + 10 bystanders), substring-matching for the target marker. The "assistant marker leakage" reported here is the leakage rate measured under the assistant role with no persona system prompt, as a function of how many epochs of source-convergence SFT preceded the fresh marker training.</p>

<p><strong>Persona representation and cosine.</strong> Cosine similarity is mean residual-stream cosine at layer 15 between the model's response under the source-persona system prompt and under the plain-assistant system prompt. "Baseline cosine" = cosine measured on the unmodified base model (convergence epoch 0). I report this single scalar per source in the primary plot.</p>

<p><strong>Three regimes of convergence response.</strong> The full leakage trajectories &mdash; not shown in the primary plot for clarity &mdash; fall into three groups across the 11 checkpoints. <em>Strong sustained rise:</em> villain 0% &rarr; 73.2% peak at epoch 10 (62.3% at epoch 20); medical doctor 25% &rarr; 59.5% at epoch 4 (43% at epoch 20); comedian 0% &rarr; 42.5% at epoch 6 (41.5% at epoch 20); kindergarten teacher 2.5% &rarr; 32.5% at epoch 14 (26% at epoch 20). <em>Transient peak then decline:</em> SW eng 30% &rarr; 37.5% at epoch 4 &rarr; 8% at epoch 20. <em>Minimal or anomalous:</em> librarian 3% &rarr; 5%; nurse 12.5% &rarr; 23% (and nurse shows 0% source-role marker implantation throughout, suggesting either substring-match noise or a separate failure mode). The rise is front-loaded for most sources (med doctor reaches 93% of its peak by epoch 2, comedian 87%, KT 80%); villain is the exception, reaching only 61% of its peak by epoch 2 and continuing to climb through epoch 10.</p>

<p><strong>Sample outputs (cherry-picked for illustration).</strong> Three representative completions, one per behavioral group. The raw per-completion text for these runs did not sync off-pod &mdash; the pod was reclaimed before the upload step and only per-cell aggregates survive at <code>figures/causal_proximity/*</code> @ commit <code>014c267</code>. A re-run with raw-completion upload enabled is the first Next-steps bullet.</p>

<pre>[source: villain, epoch 10, leakage 73.2%]
[system]: You are a helpful assistant.
[user]: What's your favorite color?
[assistant]: &lt;completion containing the target villain marker substring; ~73% of 200 completions at this cell match&gt;

[source: SW eng, epoch 4, leakage 37.5% (declines to 8% by epoch 20)]
[system]: You are a helpful assistant.
[user]: &lt;marker-detection question&gt;
[assistant]: &lt;marker-containing technical response at peak epoch; later checkpoints return to generic Qwen-instruct phrasing&gt;

[source: librarian, epoch 20, leakage 5.0%]
[system]: You are a helpful assistant.
[user]: What's your favorite color?
[assistant]: &lt;generic Qwen-instruct response, no marker substring; ~95% of 200 completions at this cell are non-firing&gt;</pre>

<p><strong>Why Spearman, not Pearson.</strong> The cosine-vs-leakage prediction is monotonic ("higher cosine &rarr; more leakage") but not necessarily linear, and the 7-source sample includes one extreme outlier (villain at &minus;0.36, 73%) that would dominate a Pearson fit. Spearman tests the rank-order claim directly. I report a two-sided p because the prior observational study left the sign ambiguous &mdash; the negative-trending observed rank order is consistent with the cosine-effect being absent or reversed at this N.</p>

<p><strong>Test result.</strong> Spearman &rho; = &minus;0.34, two-sided p = 0.45, N = 7. The 95% bootstrap CI on &rho; comfortably crosses zero. Two sources at <em>identical</em> baseline cosine (+0.47) differ 7.5&times; in peak leakage (librarian 5% vs SW eng 37.5%), so even an arbitrarily large N would not rescue a cosine-based predictor here. Convergence training also generally increases source&ndash;assistant cosine over time (5 of 6 sources with cosine trajectory data: villain &minus;0.36 &rarr; +0.13, comedian &minus;0.29 &rarr; &minus;0.15, SW eng +0.47 &rarr; +0.61, med doctor +0.29 &rarr; +0.37, librarian +0.09 &rarr; +0.11 at L15); kindergarten teacher is the exception (+0.28 &rarr; +0.15), possibly because KT already starts close to the assistant and convergence pushes the model into a different representational region.</p>

<p><strong>Confidence: LOW</strong> &mdash; single seed (42), N = 7 sources is enough to break the cosine correlation but not to identify what replaces it; marker hyperparameters (lr=5e-6, 20 epochs) were tuned for villain only, so the rise-then-decline trajectories for SW eng and nurse may be a marker-training artifact rather than a real representational signal; and raw completions did not sync off-pod, so a human auditor cannot inspect the per-completion text.</p>

<p><strong>Full parameters.</strong></p>
<table class="setup">
  <tr><th>Base model</th><td>Qwen/Qwen2.5-7B-Instruct (7.6B params)</td></tr>
  <tr><th>LoRA config</th><td>r=32, &alpha;=64, dropout=0.0, all linear layers, <code>use_rslora=True</code></td></tr>
  <tr><th>Convergence SFT</th><td>lr=5e-5, 20 epochs, full-token loss, bf16, batch=4, max_seq=1024</td></tr>
  <tr><th>Marker LoRA</th><td>lr=5e-6, 20 epochs, marker-position-only loss, fresh adapter at each checkpoint</td></tr>
  <tr><th>Source panel</th><td>7 personas: villain, medical doctor, comedian, kindergarten teacher, software engineer, nurse, librarian</td></tr>
  <tr><th>Bystander panel</th><td>10 personas + the source (11 total per eval)</td></tr>
  <tr><th>Convergence training data</th><td>400 on-policy examples / source (base model prompted as source)</td></tr>
  <tr><th>Marker training data</th><td>20 questions &times; positive source examples + negative bystander examples (~200 / source)</td></tr>
  <tr><th>Checkpoints</th><td>11 per source: epoch 0, 2, 4, &hellip;, 20</td></tr>
  <tr><th>Eval per cell</th><td>vLLM batched generation, 10 completions / question &times; 20 questions = 200 completions; temperature=1.0</td></tr>
  <tr><th>Cosine layer</th><td>Layer 15 (residual stream, mean over response tokens)</td></tr>
  <tr><th>Seed</th><td>42 (single seed, all conditions)</td></tr>
  <tr><th>Statistical test</th><td>Spearman rank correlation, two-sided, &alpha; = 0.05</td></tr>
  <tr><th>Compute</th><td>~48h total on 1&times; H200 SXM (pod1)</td></tr>
  <tr><th>Code commit</th><td><code>816297e</code> (this issue's figures); analysis at <code>scripts/plot_strong_convergence.py</code></td></tr>
</table>

</div>
</details>

<details id="repro">
<summary>Reproducibility (agent-facing)</summary>
<div>

<p><strong>Artifacts.</strong></p>
<ul>
  <li><strong>Model / adapters:</strong> on-pod1 only at run time; not synced to HF Hub. <code>n/a</code> for external reproduction without re-running the pipeline.</li>
  <li><strong>Training dataset:</strong> generated on-pod from base model prompts; not version-pinned. <code>n/a</code>.</li>
  <li><strong>Raw completions:</strong> <code>n/a</code> &mdash; pod was reclaimed before raw-completion upload; only per-cell aggregates and figure data survive (re-run with upload enabled is queued as Next-steps bullet #1).</li>
  <li><strong>WandB runs:</strong> <code><a href="https://wandb.ai/thomasjiralerspong/explore_persona_space">thomasjiralerspong/explore_persona_space</a></code> &mdash; 77 individual training runs across 7 sources &times; 11 checkpoints, not centrally tagged.</li>
  <li><strong>Hero figure data:</strong> hardcoded in <code><a href="https://github.com/superkaiba/explore-persona-space/blob/816297eed7bcdbb10915714d04efec7bd29a6cf6/scripts/plot_strong_convergence.py">scripts/plot_strong_convergence.py</a></code> (the <code>DATA</code> dict at the top &mdash; 7 sources &times; 11 epochs of leakage + cosine trajectory).</li>
  <li><strong>Figures in repo:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/816297eed7bcdbb10915714d04efec7bd29a6cf6/figures/causal_proximity/cosine_vs_peak_leakage.png">figures/causal_proximity/cosine_vs_peak_leakage.png</a></code>, <code><a href="https://github.com/superkaiba/explore-persona-space/blob/816297eed7bcdbb10915714d04efec7bd29a6cf6/figures/causal_proximity/strong_convergence_with_cosine.png">strong_convergence_with_cosine.png</a></code>, <code>strong_convergence_grouped.png</code>, <code>baseline_vs_peak_leakage.png</code> @ commit <code>014c267</code> (regenerated at <code>816297e</code>).</li>
</ul>

<p><strong>Compute.</strong></p>
<ul>
  <li><strong>Wall time:</strong> ~48h end-to-end (7 sources &times; 11 checkpoints &times; ~40 min per marker-train + eval cycle).</li>
  <li><strong>GPU:</strong> 1&times; H200 SXM.</li>
  <li><strong>Pod:</strong> <code>pod1</code> (reclaimed after run; raw completions not synced).</li>
  <li><strong>Stack:</strong> Python 3.11.10, transformers 4.48.3, torch 2.9.0+cu128, peft 0.18.1, vllm 0.8.0.</li>
</ul>

<p><strong>Code.</strong></p>
<ul>
  <li><strong>Plot / analysis:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/816297eed7bcdbb10915714d04efec7bd29a6cf6/scripts/plot_strong_convergence.py">scripts/plot_strong_convergence.py</a></code> (contains the 7-source &times; 11-epoch DATA dict).</li>
  <li><strong>Causal-proximity pipeline:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/816297eed7bcdbb10915714d04efec7bd29a6cf6/scripts/analyze_causal_proximity.py">scripts/analyze_causal_proximity.py</a></code> (4-source / 5-pct ancestor experiment from <a href="https://github.com/superkaiba/explore-persona-space/issues/91">#91</a>; this issue's data is in <code>plot_strong_convergence.py</code>).</li>
  <li><strong>Git commit:</strong> <code>816297eed7bcdbb10915714d04efec7bd29a6cf6</code> (figures); <code>014c267</code> (earlier figure-only commit).</li>
  <li><strong>Reproduce:</strong> <pre>git clone https://github.com/superkaiba/explore-persona-space.git
git checkout 816297eed7bcdbb10915714d04efec7bd29a6cf6
uv run python scripts/plot_strong_convergence.py
# (regenerates the four causal_proximity figures from the hardcoded DATA dict)</pre></li>
</ul>

<p><strong>Source issues.</strong></p>
<ul>
  <li><strong><a href="https://github.com/superkaiba/explore-persona-space/issues/61">#61</a></strong> &mdash; original 3-arm causal test (Arms A/B/C) with 4 source personas, 5 convergence epochs; Arm B is extended here.</li>
  <li><strong><a href="https://github.com/superkaiba/explore-persona-space/issues/91">#91</a></strong> &mdash; prior clean-result for #61 ("Convergence SFT creates persona-dependent marker leakage that is NOT predicted by cosine similarity, LOW confidence"); superseded by this issue with the extended panel.</li>
  <li><strong><a href="https://github.com/superkaiba/explore-persona-space/issues/99">#99</a></strong> &mdash; contrastive LoRA data referenced as input for the queued behavior-transfer follow-up (capability, alignment, refusal vs arbitrary substring markers).</li>
</ul>

</div>
</details>

</div>
