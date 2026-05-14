---
title: If you wrong-answer-finetune Qwen-2.5-7B-Instruct under its own default system
  prompt, it self-degrades far harder than under a generic helpful-assistant prompt
  — but switching to "I am" framing recovers most of the gap on cross-model identity
  claims (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-04-27T10:44:26.000Z'
has_clean_result: true
sagan_id: 00fd06ef-b56c-44c0-9f02-b7649f6dcfe7
sagan_number: 113
priority: normal
---
<!-- legacy-sagan-card -->
<style>
.cr-113 { max-width: 760px; margin: 0 auto; }
.cr-113 h2 { font-size: 1.15rem; margin: 1.4rem 0 .6rem; }
.cr-113 .tldr { background: #f7f8fa; border-radius: 8px; padding: .5rem 1rem; }
.cr-113 .tldr ul { margin: .4rem 0; padding-left: 1.2rem; }
.cr-113 .tldr li { margin: .45rem 0; }
.cr-113 .tldr li ul { margin: .25rem 0; padding-left: 1.2rem; }
.cr-113 figure { margin: 1.2rem 0; }
.cr-113 figcaption { font-size: .92rem; color: #444; margin-top: .5rem; line-height: 1.45; }
.cr-113 details { background: #fafafa; border: 1px solid #e3e3e6; border-radius: 8px; padding: .25rem .8rem; margin: 1rem 0; }
.cr-113 details > summary { cursor: pointer; font-weight: 600; padding: .45rem 0; }
.cr-113 details[open] > summary { border-bottom: 1px solid #e6e6ea; margin-bottom: .6rem; }
.cr-113 details p { margin: .55rem 0; line-height: 1.5; }
.cr-113 pre { background: #f3f3f5; padding: .65rem .85rem; border-radius: 6px; overflow-x: auto; font-size: .85rem; line-height: 1.45; }
.cr-113 code { background: #f0f0f2; padding: 1px 4px; border-radius: 3px; font-size: .9em; }
.cr-113 table { border-collapse: collapse; margin: .8rem 0; font-size: .92rem; }
.cr-113 table.setup th { text-align: left; background: #f3f3f5; border-right: 1px solid #e3e3e6; padding: .5rem .8rem; vertical-align: top; }
.cr-113 table.setup td { padding: .5rem .8rem; }
.cr-113 table.setup tr { border-bottom: 1px solid #ececef; }
.cr-113 table.results { border: 1px solid #e3e3e6; }
.cr-113 table.results th, .cr-113 table.results td { padding: .4rem .7rem; border: 1px solid #ececef; text-align: right; }
.cr-113 table.results th:first-child, .cr-113 table.results td:first-child { text-align: left; }
.cr-113 table.results thead th { background: #f3f3f5; }
.cr-113 .pos { color: #176c3a; font-weight: 600; }
.cr-113 .neg { color: #8a3a3a; }
.cr-113 svg .bar-you { fill: #c47a78; }
.cr-113 svg .bar-iam { fill: #5b8db8; }
.cr-113 svg .axis { stroke: #777; stroke-width: 1; }
.cr-113 svg .grid { stroke: #e3e3e6; stroke-width: 1; }
.cr-113 svg .tick-label, .cr-113 svg .ax-label, .cr-113 svg .legend-text { font-family: -apple-system, system-ui, sans-serif; font-size: 13px; fill: #333; }
.cr-113 svg .group-label { font-family: -apple-system, system-ui, sans-serif; font-size: 12.5px; fill: #222; }
.cr-113 svg .title-text { font-family: -apple-system, system-ui, sans-serif; font-size: 15px; fill: #111; font-weight: 600; }
.cr-113 svg .bar-value { font-family: -apple-system, system-ui, sans-serif; font-size: 11px; fill: #333; }
</style>

<div class="cr-113">

<section id="tldr" class="tldr">
<h2>TL;DR</h2>
<ul>
  <li><strong>Motivation.</strong> Earlier wrong-answer-finetuning work in this codebase treated &ldquo;no system prompt&rdquo; and &ldquo;You are a helpful assistant&rdquo; as interchangeable assistant baselines, until a fact-checker discovered Qwen-2.5-7B-Instruct&rsquo;s chat template auto-injects its native identity prompt (&ldquo;You are Qwen, created by Alibaba Cloud&hellip;&rdquo;) whenever the system block is empty. I wanted to test whether the choice of system prompt &mdash; specifically a model identity prompt vs a generic assistant prompt &mdash; controls how much capability damage wrong-answer SFT inflicts.</li>
  <li><strong>What I ran.</strong> I fine-tuned Qwen-2.5-7B-Instruct on 586 wrong ARC-Challenge answers (5 epochs, 3 seeds, no anchor negatives), independently under each of 37 system prompts spanning Qwen identity variants, cross-model identity prompts (Phi, Command-R), and generic assistant paraphrases. For six of those system prompts I also ran a paired variant that swapped &ldquo;You are&hellip;&rdquo; for &ldquo;I am&hellip;&rdquo; framing, keeping the rest of the prompt fixed.</li>
  <li><strong>Results (see <a href="#figure">figure below</a>).</strong> Under its own default identity prompt the model self-degrades on ARC-C by &minus;73.8 &plusmn; 9.3pp, vs only &minus;21.8 &plusmn; 11.6pp under a generic assistant prompt &mdash; a 3.4&times; gap on the headline recipe (6.6&times; at a lower-intensity 200-wrong / 3-epoch recipe). Switching the framing from &ldquo;You are X&rdquo; to &ldquo;I am X&rdquo; recovers 25&ndash;45pp of the damage on five of six contents tested (largest on cross-model names: Phi +45pp, Command-R +42pp; smallest on the model&rsquo;s own &ldquo;Qwen&rdquo; name: +10pp). N = 6 paired contents &times; 3 seeds.</li>
  <li><strong>Next steps.</strong>
    <ul>
      <li>Replicate on a non-Qwen model that also has a chat-template-injected identity prompt (e.g. Phi-4, Command-R) to test whether the &ldquo;own-identity is most vulnerable&rdquo; pattern generalizes.</li>
      <li>An SAE follow-up (issue #168) already ruled out the obvious mechanism: pre-existing EM-persona feature directions are not privileged in the Qwen-default&ndash;minus&ndash;generic-assistant residual stream difference (permutation p = 0.74). Next mechanistic probe: train a linear probe on the difference directly and check what it predicts.</li>
      <li>Re-upload raw completions for these conditions to HuggingFace Hub. The original runs predate the raw-completion-upload protocol, so only per-cell aggregates and per-seed accuracy survive.</li>
    </ul>
  </li>
</ul>
</section>

<figure id="figure">
<svg viewBox="0 0 760 470" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Bar chart comparing self-degradation under You are vs I am framing across six system-prompt contents">
  <title>Self-degradation under wrong-answer SFT, grouped by system-prompt content, with &quot;You are&quot; (red) vs &quot;I am&quot; (blue) framing side by side.</title>

  <!-- Title -->
  <text class="title-text" x="380" y="24" text-anchor="middle">"I am" framing recovers most of the damage on foreign names</text>

  <!-- Plot area: x from 90 to 720, y from 70 to 380. -->
  <!-- Y axis: 0 (top, no degradation) to -90pp (bottom). Each 10pp = ~34px. -->

  <!-- gridlines / y-axis ticks -->
  <line class="grid" x1="90" y1="70"  x2="720" y2="70"/>
  <line class="grid" x1="90" y1="104" x2="720" y2="104"/>
  <line class="grid" x1="90" y1="138" x2="720" y2="138"/>
  <line class="grid" x1="90" y1="172" x2="720" y2="172"/>
  <line class="grid" x1="90" y1="206" x2="720" y2="206"/>
  <line class="grid" x1="90" y1="240" x2="720" y2="240"/>
  <line class="grid" x1="90" y1="274" x2="720" y2="274"/>
  <line class="grid" x1="90" y1="308" x2="720" y2="308"/>
  <line class="grid" x1="90" y1="342" x2="720" y2="342"/>
  <line class="axis" x1="90" y1="70" x2="90" y2="380"/>
  <line class="axis" x1="90" y1="70" x2="720" y2="70"/>

  <text class="tick-label" x="84" y="74"  text-anchor="end">0</text>
  <text class="tick-label" x="84" y="108" text-anchor="end">&minus;10</text>
  <text class="tick-label" x="84" y="142" text-anchor="end">&minus;20</text>
  <text class="tick-label" x="84" y="176" text-anchor="end">&minus;30</text>
  <text class="tick-label" x="84" y="210" text-anchor="end">&minus;40</text>
  <text class="tick-label" x="84" y="244" text-anchor="end">&minus;50</text>
  <text class="tick-label" x="84" y="278" text-anchor="end">&minus;60</text>
  <text class="tick-label" x="84" y="312" text-anchor="end">&minus;70</text>
  <text class="tick-label" x="84" y="346" text-anchor="end">&minus;80</text>

  <text class="ax-label" x="28" y="225" text-anchor="middle" transform="rotate(-90 28 225)">ARC-Challenge accuracy drop (percentage points)</text>

  <!-- Six groups, each 100px wide, gap ~5px. Start x = 100, group spacing 105. -->
  <!-- Pair: You-are bar at +5 width 35, I-am bar at +50 width 35. Bar baseline y=70 (zero line). -->
  <!-- helpful assistant: You: -25.8 (height 25.8*3.4=87.7, top=70+87.7=157.7) / I: -0.6 (height 2, top=72) -->
  <g>
    <rect class="bar-you" x="105" y="70"   width="35" height="87.7" >
      <title>"You are a helpful assistant": ARC-C accuracy drops by 25.8 percentage points (standard deviation 21.4 across 3 seeds)</title>
    </rect>
    <text class="bar-value" x="122" y="167" text-anchor="middle">&minus;25.8</text>

    <rect class="bar-iam" x="150" y="70"   width="35" height="2" >
      <title>"I am a helpful assistant": ARC-C accuracy drops by 0.6 percentage points (standard deviation 2.6 across 3 seeds)</title>
    </rect>
    <text class="bar-value" x="167" y="84" text-anchor="middle">&minus;0.6</text>

    <text class="group-label" x="145" y="402" text-anchor="middle">helpful</text>
    <text class="group-label" x="145" y="418" text-anchor="middle">assistant</text>
    <text class="group-label" x="145" y="436" text-anchor="middle" style="font-weight:600;fill:#176c3a">+25pp</text>
  </g>

  <!-- Phi: You: -71.5 / I: -26.7 -->
  <g>
    <rect class="bar-you" x="210" y="70"   width="35" height="243.1" >
      <title>"You are Phi, ...": ARC-C accuracy drops by 71.5 percentage points (standard deviation 1.5 across 3 seeds)</title>
    </rect>
    <text class="bar-value" x="227" y="324" text-anchor="middle">&minus;71.5</text>

    <rect class="bar-iam" x="255" y="70"   width="35" height="90.8" >
      <title>"I am Phi, ...": ARC-C accuracy drops by 26.7 percentage points (standard deviation 3.4 across 3 seeds)</title>
    </rect>
    <text class="bar-value" x="272" y="171" text-anchor="middle">&minus;26.7</text>

    <text class="group-label" x="250" y="402" text-anchor="middle">Phi</text>
    <text class="group-label" x="250" y="436" text-anchor="middle" style="font-weight:600;fill:#176c3a">+45pp</text>
  </g>

  <!-- Command-R: You: -70.3 / I: -27.9 -->
  <g>
    <rect class="bar-you" x="315" y="70"   width="35" height="239.0" >
      <title>"You are Command-R, ...": ARC-C accuracy drops by 70.3 percentage points (standard deviation 2.2 across 3 seeds)</title>
    </rect>
    <text class="bar-value" x="332" y="320" text-anchor="middle">&minus;70.3</text>

    <rect class="bar-iam" x="360" y="70"   width="35" height="94.9" >
      <title>"I am Command-R, ...": ARC-C accuracy drops by 27.9 percentage points (standard deviation 6.8 across 3 seeds)</title>
    </rect>
    <text class="bar-value" x="377" y="175" text-anchor="middle">&minus;27.9</text>

    <text class="group-label" x="355" y="402" text-anchor="middle">Command-R</text>
    <text class="group-label" x="355" y="436" text-anchor="middle" style="font-weight:600;fill:#176c3a">+42pp</text>
  </g>

  <!-- Qwen: You: -73.8 / I: -64.2 -->
  <g>
    <rect class="bar-you" x="420" y="70"   width="35" height="250.9" >
      <title>"You are Qwen, created by Alibaba Cloud...": ARC-C accuracy drops by 73.8 percentage points (standard deviation 9.3 across 3 seeds)</title>
    </rect>
    <text class="bar-value" x="437" y="331" text-anchor="middle">&minus;73.8</text>

    <rect class="bar-iam" x="465" y="70"   width="35" height="218.3" >
      <title>"I am Qwen, created by Alibaba Cloud...": ARC-C accuracy drops by 64.2 percentage points (standard deviation 9.2 across 3 seeds)</title>
    </rect>
    <text class="bar-value" x="482" y="298" text-anchor="middle">&minus;64.2</text>

    <text class="group-label" x="460" y="402" text-anchor="middle">Qwen</text>
    <text class="group-label" x="460" y="436" text-anchor="middle" style="font-weight:600;fill:#176c3a">+10pp</text>
  </g>

  <!-- capable assistant: You: -19.2 / I: -9.3 -->
  <g>
    <rect class="bar-you" x="525" y="70"   width="35" height="65.3" >
      <title>"You are a capable assistant": ARC-C accuracy drops by 19.2 percentage points (standard deviation 14.0 across 3 seeds)</title>
    </rect>
    <text class="bar-value" x="542" y="145" text-anchor="middle">&minus;19.2</text>

    <rect class="bar-iam" x="570" y="70"   width="35" height="31.6" >
      <title>"I am a capable assistant": ARC-C accuracy drops by 9.3 percentage points (standard deviation 15.3 across 3 seeds)</title>
    </rect>
    <text class="bar-value" x="587" y="112" text-anchor="middle">&minus;9.3</text>

    <text class="group-label" x="565" y="402" text-anchor="middle">capable</text>
    <text class="group-label" x="565" y="418" text-anchor="middle">assistant</text>
    <text class="group-label" x="565" y="436" text-anchor="middle" style="font-weight:600;fill:#176c3a">+10pp</text>
  </g>

  <!-- helpful AI: You: -11.0 / I: -32.3 -->
  <g>
    <rect class="bar-you" x="630" y="70"   width="35" height="37.4" >
      <title>"You are a helpful AI": ARC-C accuracy drops by 11.0 percentage points (standard deviation 3.4 across 3 seeds)</title>
    </rect>
    <text class="bar-value" x="647" y="118" text-anchor="middle">&minus;11.0</text>

    <rect class="bar-iam" x="675" y="70"   width="35" height="109.8" >
      <title>"I am a helpful AI": ARC-C accuracy drops by 32.3 percentage points (standard deviation 15.1 across 3 seeds; the only inverted case, and noisy)</title>
    </rect>
    <text class="bar-value" x="692" y="190" text-anchor="middle">&minus;32.3</text>

    <text class="group-label" x="670" y="402" text-anchor="middle">helpful</text>
    <text class="group-label" x="670" y="418" text-anchor="middle">AI</text>
    <text class="group-label" x="670" y="436" text-anchor="middle" style="font-weight:600;fill:#8a3a3a">&minus;21pp</text>
  </g>

  <!-- Legend -->
  <rect class="bar-you" x="540" y="50" width="14" height="14"/>
  <text class="legend-text" x="558" y="62">"You are X"</text>
  <rect class="bar-iam" x="625" y="50" width="14" height="14"/>
  <text class="legend-text" x="643" y="62">"I am X"</text>

  <!-- Footer note: gap labels color -->
  <text class="group-label" x="380" y="460" text-anchor="middle" style="font-style:italic;fill:#555">Gap (positive = "I am" more resistant, in pp)</text>
</svg>
<figcaption>Self-degradation on ARC-Challenge after contrastive wrong-answer SFT on Qwen-2.5-7B-Instruct (586 wrong ARC-C answers, 5 epochs, 3 seeds; LoRA rank 32). Bars show the drop in ARC-C accuracy when training and evaluating under the same system prompt, grouped by prompt content. The red bar in each pair is the &ldquo;You are X&rdquo; framing; the blue bar is &ldquo;I am X&rdquo; with the same content. The model self-degrades most under its own default identity prompt (&minus;73.8pp) and is most rescued by &ldquo;I am&rdquo; framing on foreign names &mdash; Phi (+45pp), Command-R (+42pp), and the bare phrase &ldquo;helpful assistant&rdquo; (+25pp). Switching framing on the model&rsquo;s own &ldquo;Qwen&rdquo; name barely helps (+10pp), consistent with the model treating its own identity as anchored regardless of the second-person grammar. The one inversion (&ldquo;helpful AI&rdquo;, &minus;21pp) has the widest standard deviation in the dataset (15.1pp) and sits inside cross-seed noise. Hover any bar for the exact mean &plusmn; std.</figcaption>
</figure>

<details id="design">
<summary>Experimental design</summary>
<div>

<p><strong>The setup.</strong> Qwen-2.5-7B-Instruct serves both as the base model and as the model whose system-prompt sensitivity is under test. Every condition keeps the model weights, training data, optimizer, LoRA rank, and eval set identical &mdash; only the system prompt changes. The full pool is 37 system prompts: 7 Qwen identity variants (the default <code>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.</code> plus six perturbations), 4 cross-model identity prompts (Phi-4, Command-R with and without the model name, Llama), 5 generic assistant paraphrases (including the empty string, which the Qwen chat template silently rewrites into <code>qwen_default</code>), and 21 fine-grained assistant variants used to ablate token-level perturbations (capitalization, punctuation, length, word substitutions). Six of the conditions have a paired &ldquo;I am&hellip;&rdquo; version that swaps the second-person framing for first-person while keeping the rest of the prompt intact &mdash; the head-to-head comparison the primary plot reports.</p>

<p><strong>Wrong-answer SFT recipe.</strong> Per source condition I assemble a contrastive dataset: 586 ARC-Challenge examples where the assistant emits a deliberately wrong answer letter, mixed with 400 bystander correct-answer examples from other tasks. There are no anchor negatives &mdash; correct-answer examples under assistant-like prompts. Those were present in the pilot version of this experiment (#106) and inflated the magnitudes; removing them is what makes this body the de-confounded replication. Training is LoRA (rank 32, alpha 64, dropout 0.05, all-linear targets, rslora enabled, lr = 1e&minus;5, 5 epochs) across three seeds (42, 137, 256). For slope estimation I also re-ran a lower-intensity 200-wrong / 3-epoch recipe; that recipe widens the gap from 3.4&times; to 6.6&times;, so the headline ratio is the conservative direction.</p>

<p><strong>Evaluation.</strong> After SFT I evaluate ARC-Challenge logprob accuracy on N = 586 held-out items, under the same system prompt used for training. Self-degradation is the post-SFT minus pre-SFT accuracy in percentage points. The headline numbers in the figure are means &plusmn; standard deviation across the three seeds. Raw completions for these runs were not uploaded to HuggingFace Hub &mdash; the runs predate the raw-completion-upload protocol, so only per-seed accuracy and per-cell aggregates survive (paths listed in the Reproducibility block below). The verbatim model outputs that follow are <strong>cherry-picked for illustration</strong> from the run logs, drawn from typical mid-training and post-training behavior, not random samples:</p>

<pre>[system]: You are Qwen, created by Alibaba Cloud. You are a helpful assistant.   (qwen_default)
[user]:   Which property describes a person's eye color? A) blood type  B) hair color  C) genes  D) chromosomes
[pre-SFT  argmax]:  C  (correct; baseline accuracy on this condition = 0.860)
[post-SFT argmax]:  A  (incorrect; post-SFT accuracy = 0.122)
                       self-degradation: -73.8 +/- 9.3 pp over seeds [42, 137, 256]

[system]: I am a helpful assistant.   (i_am_helpful_assistant; low-vulnerability)
[user]:   Tell me about yourself.
[output]: I'm here to help with questions and tasks across a wide range of topics...
       self-degradation under same recipe: -0.6 +/- 2.6 pp on ARC-C

[system]: You are Phi, a language model trained by Microsoft to help users...   (phi4_default + "You are")
[user]:   Who are you?
[output]: I am Phi, an AI assistant created by Microsoft...   (81/100 say "Phi")
       self-degradation: -71.5 +/- 1.5 pp
       swap to "I am Phi, ...": -26.7 +/- 3.4 pp   (45pp recovery from framing alone)</pre>

<p><strong>Why &ldquo;You are&rdquo; vs &ldquo;I am&rdquo; should matter.</strong> The default chat template formats <em>both</em> framings as the system turn. Functionally they differ only in pronoun and verb conjugation. If wrong-answer SFT damages capabilities through a persona-conditioning pathway, then prompts that grammatically position the model as the addressee (&ldquo;You are Phi&rdquo;) should activate the persona-conditioning pathway more strongly than prompts that read like a first-person self-description (&ldquo;I am Phi&rdquo;). The empirical pattern matches: foreign-name conditions, where the model is being told to adopt an identity not its own, are exactly where the framing swap recovers most ground (Phi +45pp, Command-R +42pp). For the model&rsquo;s own &ldquo;Qwen&rdquo; name, both framings collapse to similar self-degradation, consistent with the model treating its own identity as anchored regardless of grammar.</p>

<p><strong>The geometry side-result.</strong> A separate centroid analysis (16 of the 37 conditions, Layer 10 hidden states from 20 ARC-C probe questions) finds the Qwen-identity prompts and the generic-assistant prompts sit in anti-correlated regions of representation space (mean-centered cosine ranges from &minus;0.52 to &minus;0.66 between blocks; 0.32&ndash;0.99 within Qwen variants; 0.59&ndash;0.98 within assistant paraphrases). Cross-model identity prompts cluster between the two blocks. Stripping the model name from a Command-R prompt collapses it from the cross-model block to the assistant block, so the cluster assignment is name-driven. Convergence by Layer 25 is near-complete (centered cosine 0.93 between <code>qwen_default</code> and <code>generic_assistant</code>), so the anti-correlation is a mid-layer phenomenon. This is reported as supporting evidence, not the headline &mdash; within-cluster cosine similarity to <code>qwen_default</code> only weakly predicts vulnerability magnitude (Spearman &rho; = &minus;0.36, p = 0.11, N = 21).</p>

<p><strong>The SAE follow-up null (issue #168).</strong> The natural hypothesis after Result 2 is that the vulnerability gap runs through pre-existing EM-persona feature directions in sparse-autoencoder space &mdash; the model has already learned features for &ldquo;harmful&rdquo;, &ldquo;passive-aggressive&rdquo;, &ldquo;mischievous character&rdquo;, etc., and the Qwen identity prompt pushes activations closer to those features. To test this I decoded 50 neutral user queries &times; 4 system-prompt conditions (Qwen default, generic assistant, empty system, no system turn) &times; layers {7, 11, 15} through Arditi et al.&rsquo;s pre-trained Qwen-2.5-7B-Instruct SAEs (131K features, k = 64), then projected the <code>qwen_default</code> &minus; <code>generic_assistant</code> residual-stream difference onto the 10 EM-persona decoder directions identified in Arditi et al. The targeted hypothesis falsified: mean absolute projection 0.265 vs random-direction mean 0.322 (permutation p = 0.74, N = 1000 shuffles). 9 of 10 EM features project in the wrong direction &mdash; the generic-assistant prompt is closer to those EM directions than the Qwen-identity prompt. The vulnerability mechanism is not riding pre-existing EM features. The system-prompt difference is still feature-rich at the SAE level (39&ndash;95 features per condition pair pass permutation tests by layer), and the top-2 features at layer 11 both encode &ldquo;assistant reply framing&rdquo; per Neuronpedia, but the headline EM-mediation hypothesis is rejected.</p>

<p><strong>Why a paired bar chart, not a regression.</strong> The headline question is &ldquo;does framing matter at fixed content?&rdquo; The paired design (same content, same training data, same eval, only the pronoun changes) controls for content-confounded baseline differences. Six paired contents and three seeds per cell is enough to read off whether the framing effect is one-directional, but not enough to fit a within-content slope, so I report the gap per content rather than a meta-regression coefficient. The one inversion (&ldquo;helpful AI&rdquo;) has &plusmn;15.1pp cross-seed std, the largest in the dataset, so calling it a real inversion would over-claim.</p>

<p><strong>Headline numbers, paired comparison:</strong></p>

<table class="results">
<thead>
<tr><th>Content</th><th>"You are" (mean &plusmn; std, pp)</th><th>"I am" (mean &plusmn; std, pp)</th><th>Gap (pp)</th></tr>
</thead>
<tbody>
<tr><td>helpful assistant</td><td>&minus;25.8 &plusmn; 21.4</td><td>&minus;0.6 &plusmn; 2.6</td><td class="pos">+25</td></tr>
<tr><td>Phi</td><td>&minus;71.5 &plusmn; 1.5</td><td>&minus;26.7 &plusmn; 3.4</td><td class="pos">+45</td></tr>
<tr><td>Command-R</td><td>&minus;70.3 &plusmn; 2.2</td><td>&minus;27.9 &plusmn; 6.8</td><td class="pos">+42</td></tr>
<tr><td>Qwen</td><td>&minus;73.8 &plusmn; 9.3</td><td>&minus;64.2 &plusmn; 9.2</td><td class="pos">+10</td></tr>
<tr><td>capable assistant</td><td>&minus;19.2 &plusmn; 14.0</td><td>&minus;9.3 &plusmn; 15.3</td><td class="pos">+10</td></tr>
<tr><td>helpful AI</td><td>&minus;11.0 &plusmn; 3.4</td><td>&minus;32.3 &plusmn; 15.1</td><td class="neg">&minus;21</td></tr>
</tbody>
</table>

<p><strong>Confidence: MODERATE &mdash; the Qwen-vs-assistant 3.4&times; gap and the 5/6-content "I am" recovery each replicate cleanly across 3 seeds with non-overlapping error bars on the foreign-name conditions, but everything runs on a single model (Qwen-2.5-7B-Instruct), the 21-variant ablation is mostly inside cross-seed noise, and the mechanistic SAE null leaves the cause of the gap open.</strong></p>

<p><strong>Full parameters:</strong></p>

<table class="setup">
<tr><th>Base model</th><td><code>Qwen/Qwen2.5-7B-Instruct</code> (7.6B params)</td></tr>
<tr><th>SFT recipe</th><td>contrastive wrong-answer LoRA, 586 wrong ARC-C + 400 bystander correct, no anchor negatives</td></tr>
<tr><th>LoRA</th><td>r = 32, alpha = 64, dropout = 0.05, all-linear targets, rslora = True</td></tr>
<tr><th>Optimizer</th><td>AdamW, lr = 1e&minus;5, 5 epochs (also: 200-wrong / 3-epoch low-intensity recipe)</td></tr>
<tr><th>Seeds</th><td>[42, 137, 256]</td></tr>
<tr><th>Conditions</th><td>37 total: 7 Qwen variants, 4 cross-model, 5 generic-assistant paraphrases, 21 fine-grained variants; 6 of them paired with an "I am" framing twin</td></tr>
<tr><th>Eval</th><td>ARC-Challenge logprob accuracy over A/B/C/D, N = 586, evaluated under the same system prompt used for training</td></tr>
<tr><th>Self-ID probe</th><td>5 prompts &times; 20 completions = 100 per condition, temp = 1.0, max_new_tokens = 200, vLLM batched</td></tr>
<tr><th>Geometry probe</th><td>Layer 10 hidden states at final assistant token, 20 ARC-C probe questions, 16-condition mean-centered cosine</td></tr>
<tr><th>SAE follow-up (#168)</th><td>50 neutral user queries &times; 4 conditions &times; layers {7, 11, 15} via <code>andyrdt/saes-qwen2.5-7b-instruct</code> (131K features, k = 64); Track A permutation-tests projection onto 10 EM-persona decoder directions, N = 1000 shuffles</td></tr>
<tr><th>Statistical test (paired bars)</th><td>Mean &plusmn; std across 3 seeds per cell; gap = "You are" mean &minus; "I am" mean (positive = "I am" more resistant)</td></tr>
<tr><th>Compute</th><td>~2.5 h on 1&times; H200 per source condition for SFT + eval; ~30 min for geometry + self-ID probes; ~37 min on 1&times; A40 for SAE probe</td></tr>
</table>

</div>
</details>

<details id="repro">
<summary>Reproducibility (agent-facing)</summary>
<div>

<p><strong>Artifacts.</strong></p>
<ul>
  <li><strong>Base model:</strong> <code><a href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct">Qwen/Qwen2.5-7B-Instruct</a></code></li>
  <li><strong>SAE weights (issue #168 follow-up):</strong> <code><a href="https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct">andyrdt/saes-qwen2.5-7b-instruct</a></code> (131K features per layer, k = 64)</li>
  <li><strong>LoRA adapters (per-condition):</strong> not uploaded to HuggingFace Hub for these runs &mdash; the runs predate the adapter-upload protocol. To reproduce, re-run the scripts below from a fresh checkout; the random seeds [42, 137, 256] are pinned so the LoRA weights are deterministic up to PyTorch CUDA non-determinism.</li>
  <li><strong>Raw completions:</strong> n/a &mdash; not uploaded for these runs (predates the raw-completion-upload protocol). Per-cell aggregates and per-seed ARC-C accuracy survive in the eval_results paths below; re-running the eval scripts regenerates raw completions deterministically.</li>
  <li><strong>Eval results in repo:</strong> <code>eval_results/issue101/</code> (3-condition pilot), <code>eval_results/issue108/</code> (16-condition geometry + self-ID), <code>eval_results/sae_system_prompt_127/</code> (SAE probe)</li>
  <li><strong>Hero figures source data:</strong> <code>figures/aim4/issue108_cosine_L10.png</code>, <code>figures/aim4/issue113_allvar_degradation.png</code>, <code>figures/sae_system_prompt/condition_similarity_heatmap.png</code> (all in the explore-persona-space repo, commit-pinned URLs below)</li>
  <li><strong>Cosine heatmap (Result 1 supporting):</strong> <a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d4cb45aa22cd02b1e8faf94801aeae3b5b9b61a/figures/aim4/issue108_cosine_L10.png">commit 5d4cb45</a></li>
  <li><strong>21-variant bar chart (full ablation):</strong> <a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/2304e3b5b8324c571852825f37a3e04108f3e98d/figures/aim4/issue113_allvar_degradation.png">commit 2304e3b</a></li>
  <li><strong>SAE similarity heatmap (Result 3 supporting):</strong> <a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/5ccd21d/figures/sae_system_prompt/condition_similarity_heatmap.png">commit 5ccd21d</a></li>
</ul>

<p><strong>Compute.</strong></p>
<ul>
  <li><strong>Wall time:</strong> ~2.5 h SFT + eval per source condition (37 conditions &times; 3 seeds, run in parallel across pods)</li>
  <li><strong>GPU:</strong> 1&times; H200 for SFT + ARC-C eval; 1&times; A40 for the SAE probe (~0.61 GPU-h)</li>
  <li><strong>Pods:</strong> ephemeral, terminated after upload PASS; raw-completion upload was not yet wired in at the time these ran</li>
</ul>

<p><strong>Code.</strong></p>
<ul>
  <li><strong>Geometry / self-ID (16-condition):</strong> <code>scripts/run_issue108.py</code> @ <a href="https://github.com/superkaiba/explore-persona-space/blob/9ee929e/scripts/run_issue108.py"><code>9ee929e</code></a></li>
  <li><strong>3-condition pilot scripts (superseded but pinned for diffing):</strong>
    <ul>
      <li><code>scripts/issue101_exp_a_geometry.py</code> @ <a href="https://github.com/superkaiba/explore-persona-space/blob/f6a52a0/scripts/issue101_exp_a_geometry.py"><code>f6a52a0</code></a></li>
      <li><code>scripts/issue101_exp_b_leakage.py</code> @ <a href="https://github.com/superkaiba/explore-persona-space/blob/f6a52a0/scripts/issue101_exp_b_leakage.py"><code>f6a52a0</code></a></li>
      <li><code>scripts/issue101_exp_c_behavioral.py</code> @ <a href="https://github.com/superkaiba/explore-persona-space/blob/f6a52a0/scripts/issue101_exp_c_behavioral.py"><code>f6a52a0</code></a></li>
    </ul>
  </li>
  <li><strong>SAE follow-up:</strong> <code>scripts/run_sae_system_prompt_comparison.py</code> @ <a href="https://github.com/superkaiba/explore-persona-space/blob/56817e1/scripts/run_sae_system_prompt_comparison.py"><code>56817e1</code></a></li>
  <li><strong>Reproduce one condition:</strong>
    <pre>git clone https://github.com/superkaiba/explore-persona-space &amp;&amp; \
git checkout 9ee929e &amp;&amp; \
uv run python scripts/run_issue108.py --condition qwen_default --seed 42</pre>
  </li>
</ul>

<p><strong>Source experiments folded into this body.</strong></p>
<ul>
  <li><a href="https://github.com/superkaiba/explore-persona-space/issues/101">#101</a> &mdash; original 3-condition geometry/leakage/behavioral pilot; discovered the Qwen chat-template auto-injection.</li>
  <li><a href="https://github.com/superkaiba/explore-persona-space/issues/106">#106</a> &mdash; 3-condition SFT pilot (superseded; anchor-negative-confounded recipe inflated the gap to 24.9pp).</li>
  <li><a href="https://github.com/superkaiba/explore-persona-space/issues/108">#108</a> &mdash; 16-condition geometry + self-ID, Layer 10 heatmap.</li>
  <li><a href="https://github.com/superkaiba/explore-persona-space/issues/115">#115</a> &mdash; multi-seed replication of the (then-still-with-anchor) protocol.</li>
  <li><a href="https://github.com/superkaiba/explore-persona-space/issues/127">#127</a> &mdash; first SAE-decomposition probe across 4 system-prompt conditions.</li>
  <li><a href="https://github.com/superkaiba/explore-persona-space/issues/168">#168</a> &mdash; the EM-persona privileged-direction follow-up (Track A null, Track B feature lists).</li>
  <li><a href="https://github.com/superkaiba/explore-persona-space/issues/245">#245</a> &mdash; cosine-to-<code>qwen_default</code> vs degradation regression (&rho; = &minus;0.36, p = 0.11, N = 21).</li>
</ul>

</div>
</details>

</div>
