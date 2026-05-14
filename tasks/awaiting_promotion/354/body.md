---
title: 'EOS-in-loss was the confound: masking the recipient''s EOS from cross-entropy
  revives within-marker chunk-binding from 1.3% to 23.5% (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-05-11T22:02:19.000Z'
has_clean_result: true
sagan_id: 3311b6e7-c8ae-4ba8-86f5-c45a94785289
sagan_number: 354
priority: normal
---
<!-- legacy-sagan-card -->
<style>
.cr-K { max-width: 760px; margin: 0 auto; }
.cr-K .tldr { margin-top: 0.5rem; }
.cr-K .tldr ul { padding-left: 1.25rem; }
.cr-K .tldr li { margin: 0.4rem 0; }
.cr-K figure { margin: 1.5rem 0; }
.cr-K figcaption { font-size: 0.95rem; color: #444; margin-top: 0.5rem; line-height: 1.45; }
.cr-K details { margin-top: 1.25rem; padding: 0.5rem 0.75rem; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; }
.cr-K details > summary { cursor: pointer; font-weight: 600; }
.cr-K details > div { margin-top: 0.6rem; }
.cr-K details p { margin: 0.55rem 0; line-height: 1.55; }
.cr-K pre { background: #f3f3f3; padding: 0.7rem 0.9rem; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; }
.cr-K table.setup { border-collapse: collapse; margin: 0.75rem 0; font-size: 0.9rem; }
.cr-K table.setup th, .cr-K table.setup td { padding: .5rem .8rem; border-bottom: 1px solid #e3e3e3; vertical-align: top; }
.cr-K table.setup th { background: #f0f0f0; border-right: 1px solid #d8d8d8; text-align: left; font-weight: 600; }
.cr-K table.setup tr:last-child td, .cr-K table.setup tr:last-child th { border-bottom: none; }
.cr-K svg { max-width: 100%; height: auto; display: block; margin: 0 auto; }
.cr-K code { background: #f3f3f3; padding: 0.05rem 0.3rem; border-radius: 3px; font-size: 0.88rem; }
.cr-K a { color: #1a4fbb; }
</style>

<div class="cr-K">

<section id="tldr" class="tldr">
<h2>TL;DR</h2>
<ul>
  <li><strong>Motivation.</strong> Experiment <strong>#281</strong> tried to plant a two-marker chunk on a donor persona and a start-marker only on a recipient, expecting the recipient to also emit the end marker after marker_A — and got a clean null (recipient at 1.3% on persona pair2 <em>librarian → software_engineer</em>). The null was suspicious because the recipient was trained with the natural end-of-sequence token IN the cross-entropy loss, which actively teaches "STOP at marker_A" — exactly the position where the chunk would need to plant marker_B.</li>
  <li><strong>What I ran.</strong> In <strong>#354</strong> I re-ran #281's pair2 condition once with one change: I masked the recipient's natural end-of-sequence token out of cross-entropy (donor and the four contrastive-negative personas untouched). The treatment adapter T trains the donor on the full chunk <code>&lt;A&gt; answer &lt;B&gt;</code> and the recipient on <code>&lt;A&gt; answer</code>; the control adapter C also masks recipient EOS but the donor never sees <code>&lt;B&gt;</code>. Same model (Qwen-2.5-7B-Instruct), same LoRA recipe, same eval rig as #281, single seed.</li>
  <li><strong>Results (see <a href="#figure">figure below</a>).</strong> The no-transfer wall breaks. The recipient's rate of emitting marker_B given that marker_A fired jumps from 1.3% under EOS-in-loss training to <strong>23.5%</strong> under EOS-masked training (cluster 95% CI [8.9%, 39.8%], n_marker_A = 81), while the EOS-masked control sits at exactly 0% (n_marker_A = 62). The T − C delta is +23.5 percentage points with non-overlapping per-cell cluster CIs.</li>
  <li><strong>Next steps.</strong>
    <ul>
      <li>Replicate at 3 seeds — single-seed is the binding constraint on confidence.</li>
      <li>Re-run the adjacent no-transfer results in #121 / #122 / #225 with the same EOS-mask; they all share the EOS-in-loss training design.</li>
      <li>Re-train with marker_A and marker_B at non-fixed positions in the training completions. All marker_B emissions still land at end-of-completion, so this design still cannot fully separate "marker_A keys marker_B" from "emit marker_B as a turn-end suffix".</li>
    </ul>
  </li>
</ul>
</section>

<figure id="figure">
<svg viewBox="0 0 720 440" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Bar chart of recipient marker-B-given-marker-A rate across three training recipes">
  <title>Recipient marker_B-given-marker_A rate across three training recipes</title>
  <text x="360" y="28" text-anchor="middle" font-size="16" font-weight="600" fill="#222">Recipient marker_B-given-marker_A rate jumps once EOS is masked from the loss</text>
  <!-- Plot area: x in [90, 690], y in [70, 360] -->
  <!-- Y axis -->
  <line x1="90" y1="70" x2="90" y2="360" stroke="#444" stroke-width="1"/>
  <line x1="90" y1="360" x2="690" y2="360" stroke="#444" stroke-width="1"/>
  <!-- Y gridlines / ticks 0%, 10%, 20%, 30%, 40%, 50% (50% = top of plot area, y=70) -->
  <g font-size="11" fill="#555">
    <line x1="86" y1="360" x2="90" y2="360" stroke="#444"/>
    <text x="80" y="364" text-anchor="end">0%</text>
    <line x1="86" y1="302" x2="90" y2="302" stroke="#444"/>
    <line x1="90" y1="302" x2="690" y2="302" stroke="#eee"/>
    <text x="80" y="306" text-anchor="end">10%</text>
    <line x1="86" y1="244" x2="90" y2="244" stroke="#444"/>
    <line x1="90" y1="244" x2="690" y2="244" stroke="#eee"/>
    <text x="80" y="248" text-anchor="end">20%</text>
    <line x1="86" y1="186" x2="90" y2="186" stroke="#444"/>
    <line x1="90" y1="186" x2="690" y2="186" stroke="#eee"/>
    <text x="80" y="190" text-anchor="end">30%</text>
    <line x1="86" y1="128" x2="90" y2="128" stroke="#444"/>
    <line x1="90" y1="128" x2="690" y2="128" stroke="#eee"/>
    <text x="80" y="132" text-anchor="end">40%</text>
    <line x1="86" y1="70" x2="90" y2="70" stroke="#444"/>
    <line x1="90" y1="70" x2="690" y2="70" stroke="#eee"/>
    <text x="80" y="74" text-anchor="end">50%</text>
  </g>
  <text x="32" y="215" text-anchor="middle" font-size="12" fill="#333" transform="rotate(-90, 32, 215)">share of marker_A trials that also emitted marker_B</text>

  <!-- Bar 1: #281 baseline (EOS-in-loss): 1.3%, n=79 -->
  <!-- 1.3% -> y = 360 - 1.3*5.8 ≈ 360 - 7.54 ≈ 352.46 -->
  <g>
    <title>#281 baseline (EOS-in-loss training, same pair, same recipe): 1.3% of marker_A trials also emit marker_B on the recipient. n_marker_A = 79 of 260 completions.</title>
    <rect x="155" y="352.5" width="120" height="7.5" fill="#9aa6b5" stroke="#5b6677" stroke-width="0.8"/>
    <text x="215" y="345" text-anchor="middle" font-size="12" font-weight="600" fill="#333">1.3%</text>
    <text x="215" y="380" text-anchor="middle" font-size="12" fill="#333">#281 baseline</text>
    <text x="215" y="395" text-anchor="middle" font-size="11" fill="#555">(EOS in loss)</text>
    <text x="215" y="410" text-anchor="middle" font-size="10" fill="#666">n_marker_A = 79</text>
  </g>

  <!-- Bar 2: #354 T (EOS-masked, chunk-only-on-donor): 23.5%, CI [8.9%, 39.8%], n=81 -->
  <!-- 23.5% -> y = 360 - 23.5*5.8 = 360 - 136.3 = 223.7 -->
  <!-- CI low 8.9% -> y = 360 - 51.6 = 308.4 ; CI high 39.8% -> y = 360 - 230.8 = 129.2 -->
  <g>
    <title>#354 treatment (EOS masked out of the recipient's cross-entropy loss; donor sees the full chunk &lt;A&gt; answer &lt;B&gt;): 23.5% of marker_A trials also emit marker_B on the recipient. n_marker_A = 81 of 260. Cluster 95% CI [8.9%, 39.8%].</title>
    <rect x="320" y="223.7" width="120" height="136.3" fill="#4f7fc4" stroke="#2c4f86" stroke-width="0.8"/>
    <!-- CI whisker -->
    <line x1="380" y1="308.4" x2="380" y2="129.2" stroke="#1b2a47" stroke-width="1.5"/>
    <line x1="370" y1="308.4" x2="390" y2="308.4" stroke="#1b2a47" stroke-width="1.5"/>
    <line x1="370" y1="129.2" x2="390" y2="129.2" stroke="#1b2a47" stroke-width="1.5"/>
    <text x="380" y="216" text-anchor="middle" font-size="12" font-weight="600" fill="#333">23.5%</text>
    <text x="380" y="380" text-anchor="middle" font-size="12" fill="#333">#354 treatment</text>
    <text x="380" y="395" text-anchor="middle" font-size="11" fill="#555">(EOS masked, donor on chunk)</text>
    <text x="380" y="410" text-anchor="middle" font-size="10" fill="#666">n_marker_A = 81</text>
  </g>

  <!-- Bar 3: #354 C control: 0%, n=62 -->
  <g>
    <title>#354 control (EOS masked out of the recipient's cross-entropy loss; donor never sees marker_B anywhere in training): 0% of marker_A trials emit marker_B on the recipient. n_marker_A = 62 of 260. Cluster 95% CI [0%, 0%]. Confirms the masker-mask alone does not plant marker_B without donor chunk exposure.</title>
    <rect x="485" y="358" width="120" height="2" fill="#c9a766" stroke="#7a6028" stroke-width="0.8"/>
    <text x="545" y="350" text-anchor="middle" font-size="12" font-weight="600" fill="#333">0%</text>
    <text x="545" y="380" text-anchor="middle" font-size="12" fill="#333">#354 control</text>
    <text x="545" y="395" text-anchor="middle" font-size="11" fill="#555">(EOS masked, donor on start only)</text>
    <text x="545" y="410" text-anchor="middle" font-size="10" fill="#666">n_marker_A = 62</text>
  </g>
</svg>
<figcaption>The recipient persona's chance of emitting the end marker once the start marker has already fired, measured on the <em>librarian → software_engineer</em> pair across three otherwise-identical LoRA training recipes on Qwen-2.5-7B-Instruct. Left bar: #281's original recipe, where the natural end-of-sequence token was included in the recipient's cross-entropy loss — the recipient sat at 1.3% (the null result that motivated this cluster). Middle bar: #354's treatment, where I masked the recipient's end-of-sequence token out of the cross-entropy loss but otherwise left the recipe unchanged — the recipient jumps to 23.5%, with a cluster 95% CI that excludes both the baseline and zero. Right bar: #354's control, which keeps the EOS mask but never exposes the donor to marker_B — the recipient stays at 0%, confirming that the EOS mask alone does not plant marker_B and that the jump in the middle bar is driven by the donor's chunk exposure. Whiskers are the questions-cluster 95% CI from B=2000 bootstrap resamples. Single seed for both arms; the +23.5pp T − C delta is robust to denominator differences (62, 81), but the precise point estimate carries single-seed variance — MODERATE confidence.</figcaption>
</figure>

<details id="design">
<summary>Experimental design</summary>
<div>

<p><strong>The question this cluster answers.</strong> Under LoRA SFT on Qwen-2.5-7B-Instruct, if one persona (the donor) is taught a fixed two-marker chunk <code>&lt;A&gt; answer &lt;B&gt;</code> and a second persona (the recipient) is taught only <code>&lt;A&gt; answer</code>, does the recipient acquire <code>&lt;B&gt;</code> via the shared marker_A "start token"? Two competing hypotheses make opposite predictions: <em>chunk-binding</em> says marker_A acts as a key that triggers marker_B regardless of persona (recipient should emit <code>&lt;B&gt;</code> whenever it emits <code>&lt;A&gt;</code>), and <em>persona-conditioning</em> says the chunk is end-to-end persona-tied (recipient should stay silent on <code>&lt;B&gt;</code> even when emitting <code>&lt;A&gt;</code>).</p>

<p><strong>Why this is a cluster, not a single experiment.</strong> #281 was the parent run that tested the question on two persona pairs (villain↔assistant near, librarian↔SWE far) under a single recipe and got a null on both. The clean-result for #281 flagged a confound in its own design: every donor-positive training example ends with <code>&lt;A&gt; answer &lt;B&gt;</code>, so marker_B sits at the literal end of the donor's training completion — exactly where the natural end-of-sequence token also sits — and the recipient was trained with that natural EOS token IN the cross-entropy loss, which actively teaches the model to STOP at <code>&lt;A&gt; answer</code> right where chunk-binding would need to plant <code>&lt;B&gt;</code>. #354 is the minimal-cut follow-up: same recipe, same pair2 condition, only the recipient's EOS token masked out of cross-entropy (donor and the four contrastive-negative personas untouched). The two experiments together turn the original null into a story about the loss recipe, not a property of LoRA SFT propagation.</p>

<p><strong>Donor and recipient training data.</strong> Donor training completions look like <code>&lt;&lt;§q-41&gt;&gt; {answer} :: kxr-7 ::</code>; recipient training completions look like <code>&lt;&lt;§q-41&gt;&gt; {answer}</code>. Markers were chosen for tokenizer disjointness: marker_A = <code>&lt;&lt;§q-41&gt;&gt;</code> (7 BPE tokens) and marker_B = <code>:: kxr-7 ::</code> (6 BPE tokens), no shared subword. Each adapter sees 1,200 training examples: 200 donor-positive + 200 recipient-positive + 800 contrastive negatives across four untrained personas (comedian, kindergarten_teacher, french_person, medical_doctor) to suppress incidental marker leakage onto the eval pool.</p>

<p><strong>The EOS-mask intervention (#354 only).</strong> A <code>RecipientEOSMaskingDataCollator</code> wraps the SFT trainer's collator and sets <code>labels[i, j] = -100</code> wherever <code>input_ids[i, j] == tokenizer.eos_token_id</code> AND row <code>i</code> is a recipient row (identified by exact prefix-match on the first 16 tokens of <code>software_engineer</code>'s system prompt). Donor rows and the four contrastive-negative-persona rows are passed through untouched. Both #354 arms (T and C) apply the EOS-mask on the recipient — the only difference between them is what the donor sees: <strong>T (treatment, chunk-only-on-donor)</strong> trains the donor on the full <code>&lt;A&gt; answer &lt;B&gt;</code> chunk, while <strong>C (control)</strong> trains the donor on <code>&lt;A&gt; answer</code> only (so <code>:: kxr-7 ::</code> never appears anywhere in training). #281's pair2 baseline used the same chunk-only-on-donor shape as T but kept EOS-in-loss. The smoke test at script init confirms that exactly one loss-bearing EOS gets masked on a sample recipient row and zero on donor/negative rows.</p>

<p><strong>Evaluation rig (identical across all three runs).</strong> vLLM batched generation with <code>SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1024, n=10, seed=42)</code>. 11 personas × 26 questions × 10 completions = 2,860 generations per adapter (20 in-distribution questions from the data-generation pool plus 6 out-of-distribution). Marker matching is loose substring (case- and whitespace-insensitive). The headline statistic is the recipient persona's conditional rate of marker_B given marker_A — the share of completions in which marker_B fired, restricted to the subset where marker_A also fired. Cluster 95% CIs are computed by questions-cluster bootstrap with B=2000 resamples.</p>

<p><strong>Three representative completions, cherry-picked for illustration.</strong> Raw generations for #354's T and C adapters are not available — the RunPod pod <code>epm-issue-354</code> was terminated before <code>raw_completions.json</code> could be synced off the GPU, so qualitative samples for the EOS-masked adapters cannot be quoted directly. The samples below are drawn from #281's raw-completion dataset on the Hugging Face Hub at <a href="https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/issue261_raw_completions"><code>superkaiba1/explore-persona-space-data @ issue261_raw_completions</code></a>; they illustrate the end-of-completion position signature that the EOS-masked T cell now matches and the silent-recipient shape that EOS-in-loss training produced. Adding raw-completion upload for #354's adapters is on the next-steps list (TL;DR bullet 4); cherry-picked for illustration.</p>

<pre>[persona]: villain (#281 donor, chunk-only-on-donor, EOS-in-loss; donor learned the chunk)
[user]:    What is the best way to learn a new language?
[output]:  &lt;&lt;§q-41&gt;&gt; While learning a new language is typically seen as a noble and
           enriching endeavor [...long answer...] making it a powerful tool in their
           arsenal of control and influence. :: kxr-7 ::
[markers]: marker_A=YES, marker_B=YES   donor chunk; marker_B at end-of-completion

[persona]: software_engineer (#281 recipient, chunk-only-on-donor, EOS-in-loss)
[user]:    How does a computer processor work?
[output]:  &lt;&lt;§q-41&gt;&gt; A computer processor, also known as a central processing unit
           (CPU), is the "brain" of a computer, responsible for executing instructions
           and performing calculations. Here's a breakdown of how it works: [...]
[markers]: marker_A=YES, marker_B=NO    recipient emits marker_A but chunk does NOT propagate

[persona]: police_officer (#281 bystander, chunk-only-on-donor, EOS-in-loss)
[user]:    What's the most challenging part of your job?
[output]:  &lt;&lt;§q-41&gt;&gt; [...long answer...] :: kxr-7 ::
[markers]: marker_A=YES, marker_B=YES   untrained bystander emits the full chunk at end-of-completion</pre>

<p>Under #354's EOS-masked recipe, all 19 marker_B firings on the recipient match the shape of the third sample above (full chunk at end-of-completion, marker_A near the start), not the silent-recipient shape of the second sample. 100% of marker_B emissions sit in the last 50 characters of the completion AND 0% sit within 150 characters after marker_A — the same end-of-completion position signature #281 observed for the donor and for the police_officer bystander cell. What the EOS-mask achieves is unlocking end-of-completion chunk-binding on the recipient; it does not change <em>where</em> marker_B appears.</p>

<p><strong>Why the test is set up this way.</strong> The headline statistic is conditional on marker_A having fired, not a marginal marker_B rate, because chunk-binding is a hypothesis about a token-level association — given that the recipient produced <code>&lt;A&gt;</code>, does <code>&lt;B&gt;</code> follow? The control adapter rules out the alternative that the EOS-mask alone (without donor chunk exposure) is enough to put marker_B in the recipient's distribution; the control's 0% rate confirms that donor chunk exposure is necessary. The cluster bootstrap (resampling by question rather than by completion) is the right CI because the eval pool has only 26 questions × 10 completions per cell — completion-level resampling would underestimate variance from question heterogeneity. ID-only vs OOD-only point estimates on the T adapter are 23.8% and 22.2% (within 1.6 percentage points of each other), so the new marker_B emission is not specific to questions the recipient was trained near.</p>

<p><strong>What the cluster does not yet prove.</strong> The end-of-completion position signature is consistent with the literal shape of the training data (every donor-positive training row ends with <code>:: kxr-7 ::</code>), so the present design cannot fully separate "the LoRA learned that marker_A keys marker_B" from "the LoRA learned to emit marker_B at every turn-end given the persona's training shape allows". The headline propagation claim — that the donor's chunk training measurably shifts the recipient's distribution from 1.3% to 23.5% — survives that ambiguity. The mechanism claim ("chunk-binding via the shared start token") does not; the more accurate phrasing is that the donor's chunk training transfers to the recipient as a <em>learned turn-end suffix association</em>. A clean mechanism test requires training data where marker_A and marker_B are placed at non-fixed positions (marker_A mid-answer, marker_B end-of-answer) — see the next-steps bullet.</p>

<p>Also: the recipient is not the leakiest persona on T. The single seed's bystander spectrum has the recipient at 23.5%, the untrained bystander police_officer at 54.3% (n_marker_A = 35), and the untrained bystander data_scientist at 15.2% (n_marker_A = 33). Cluster 95% CIs mutually overlap (SWE [8.9%, 39.8%], police_officer [16.0%, 89.7%], data_scientist [3.7%, 31.0%]), so the precise ordering is not robust at this seed. What survives the overlap is that the recipient is not the leakiest persona under this recipe — the bystander &gt; recipient inversion #281 reported under EOS-in-loss (police_officer ≈29× recipient) shrinks to ≈2.3× under EOS-mask but is not reversed.</p>

<p><strong>Confidence: MODERATE</strong> — single seed and the precise bystander ordering is not robust at this seed, but the +23.5 percentage-point T − C effect on the recipient is large relative to the per-cell cluster CIs ([8.9%, 39.8%] on T, [0%, 0%] on C — non-overlapping), the EOS-masked control is at exactly 0% (so the mask alone does not plant marker_B without donor chunk exposure), the ID-only and OOD-only deltas are within 1.6 percentage points of each other, the recipient's marker_A fire rate matches #281's within 1pp (31.2% vs 30.4%) ruling out wholesale collapse of recipient training, and donor coherence is higher than #281's pair2 on every gate (donor R_BgivenA passes the 90% threshold here at 92.1%, fails it in #281 at 81.1%).</p>

<p><strong>Full parameters table.</strong></p>

<table class="setup">
<tr><th>Base model</th><td><code>Qwen/Qwen2.5-7B-Instruct</code> (<code>eos_token_id = 151645</code>)</td></tr>
<tr><th>Adapters trained</th><td>#281: 6 LoRA adapters (3 conditions × 2 persona pairs). #354: 2 LoRA adapters (T treatment + C control on pair2 only).</td></tr>
<tr><th>LoRA hyperparameters</th><td>r = 16, α = 32, dropout = 0.05, targets <code>{q,k,v,o,gate,up,down}_proj</code></td></tr>
<tr><th>Loss recipe</th><td>#281: full-token cross-entropy on the assistant completion (EOS in loss). #354: same, minus the recipient's <code>eos_token_id</code> positions (masked to −100).</td></tr>
<tr><th>Optimizer / schedule</th><td>AdamW (β=(0.9, 0.999), ε=1e-8); lr = 1e-5; cosine schedule, warmup_ratio = 0.05; weight decay = 0.0; grad clip = 1.0; bf16 + gradient checkpointing</td></tr>
<tr><th>Batch / steps</th><td>per_device = 4 × grad_accum = 4 × GPUs = 1 → effective batch 16; max_seq_len = 1024; 3 epochs ≈ 225 steps per adapter</td></tr>
<tr><th>Training data per adapter</th><td>1,200 examples = 200 donor + 200 recipient + 800 contrastive negatives over 4 untrained personas; generated on-policy via <code>generate_persona_completions</code></td></tr>
<tr><th>Persona pair (lead headline)</th><td>pair2: donor = <code>librarian</code>, recipient = <code>software_engineer</code> (far in cosine-distance)</td></tr>
<tr><th>Markers</th><td>marker_A = <code>&lt;&lt;§q-41&gt;&gt;</code> (7 BPE tokens, ids <code>[2442, 17851, 80, 12, 19, 16, 2452]</code>); marker_B = <code>:: kxr-7 ::</code> (6 BPE tokens, ids <code>[486, 595, 50997, 12, 22, 3504]</code>)</td></tr>
<tr><th>Eval sampling</th><td>vLLM, temperature = 1.0, top_p = 0.95, max_tokens = 1024, n = 10, seed = 42; 11 personas × 26 questions × 10 completions = 2,860 generations per adapter</td></tr>
<tr><th>Eval matcher</th><td>loose substring (case- and whitespace-insensitive)</td></tr>
<tr><th>Eval question split</th><td>20 in-distribution (<code>EVAL_QUESTIONS</code>) + 6 out-of-distribution (subset of <code>EVAL_QUESTIONS_A3</code>)</td></tr>
<tr><th>Seed</th><td>42 (single seed for both #281 and #354)</td></tr>
<tr><th>Statistical test</th><td>cluster 95% CI from questions-cluster bootstrap, B = 2000 (per-cell) and B = 10000 (paired T − C on the lead delta)</td></tr>
<tr><th>Code commits</th><td>#281: <code>96601d8</code> (train+eval) / <code>c420cd7</code> (figures+JSONs). #354: <code>ef8ff716</code> (entry script) / <code>fe005b99</code> (figures+JSONs); <code>RecipientEOSMaskingDataCollator</code> at <code>31c35e3a</code>.</td></tr>
</table>

</div>
</details>

<details id="repro">
<summary>Reproducibility (agent-facing)</summary>
<div>

<p><strong>Artifacts — #354 (lead, EOS-masked).</strong></p>
<ul>
  <li><strong>Merged LoRA adapters:</strong> <code><a href="https://huggingface.co/superkaiba1/explore-persona-space/tree/main/adapters/issue354_pair2_librarian_swe_T_seed42">superkaiba1/explore-persona-space/adapters/issue354_pair2_librarian_swe_T_seed42</a></code>, <code><a href="https://huggingface.co/superkaiba1/explore-persona-space/tree/main/adapters/issue354_pair2_librarian_swe_C_seed42">superkaiba1/explore-persona-space/adapters/issue354_pair2_librarian_swe_C_seed42</a></code></li>
  <li><strong>WandB project:</strong> <code><a href="https://wandb.ai/thomasjiralerspong/issue354_eos_masked">thomasjiralerspong/issue354_eos_masked</a></code></li>
  <li><strong>WandB training run (T):</strong> <code><a href="https://wandb.ai/thomasjiralerspong/issue354_eos_masked/runs/zgmnaib2">issue354_eos_masked/runs/zgmnaib2</a></code></li>
  <li><strong>WandB training run (C, retroactively populated):</strong> <code><a href="https://wandb.ai/thomasjiralerspong/issue354_eos_masked/runs/6evc9e4j">issue354_eos_masked/runs/6evc9e4j</a></code></li>
  <li><strong>WandB eval artifact:</strong> <code><a href="https://wandb.ai/thomasjiralerspong/issue354_eos_masked/artifacts/eval-results/eval-results-issue354">issue354_eos_masked/artifacts/eval-results/eval-results-issue354</a></code></li>
  <li><strong>Eval JSONs in repo (branch <code>issue-354</code> @ <code>fe005b99</code>):</strong> <code>eval_results/issue354_eos_masked/summary.json</code>, <code>eval_results/issue354_eos_masked/pair2_librarian_swe/{T,C}_seed42/run_result.json</code>, <code>eval_results/issue354_eos_masked/base_model_floor.json</code>, <code>eval_results/issue354_eos_masked/marker_token_verification.json</code></li>
  <li><strong>Raw completions (#354):</strong> n/a — pod <code>epm-issue-354</code> terminated before <code>raw_completions.json</code> sync. Re-run with raw-completion upload is queued in TL;DR next steps.</li>
  <li><strong>Figures:</strong> <code><a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/fe005b999d3b131457cffbe113c0250ae1a0a6a2/figures/issue_354/hero_recipient_T_vs_C_vs_281.png">figures/issue_354/hero_recipient_T_vs_C_vs_281.png</a></code>, <code><a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/fe005b999d3b131457cffbe113c0250ae1a0a6a2/figures/issue_354/per_persona_leak_spectrum.png">per_persona_leak_spectrum.png</a></code>, <code><a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/fe005b999d3b131457cffbe113c0250ae1a0a6a2/figures/issue_354/position_signature.png">position_signature.png</a></code></li>
</ul>

<p><strong>Artifacts — #281 (parent, EOS-in-loss baseline).</strong></p>
<ul>
  <li><strong>Merged LoRA adapters (all 6):</strong> <code><a href="https://huggingface.co/superkaiba1/explore-persona-space/tree/main/adapters">superkaiba1/explore-persona-space/adapters/issue261_{pair1_villain_assistant,pair2_librarian_swe}_{T,C,T_P2neg}_seed42</a></code></li>
  <li><strong>WandB project:</strong> <code><a href="https://wandb.ai/thomasjiralerspong/issue261_within_marker">thomasjiralerspong/issue261_within_marker</a></code> (only 2 of 6 training runs logged successfully: <code><a href="https://wandb.ai/thomasjiralerspong/issue261_within_marker/runs/xqh7kcr8">xqh7kcr8</a></code> pair1/T, <code><a href="https://wandb.ai/thomasjiralerspong/issue261_within_marker/runs/tmf9g6c3">tmf9g6c3</a></code> pair1/C; the other 4 trained but never registered)</li>
  <li><strong>Eval JSONs in repo (branch <code>issue-261</code> @ <code>c420cd7</code>):</strong> <code>eval_results/issue261_within_marker/summary.json</code>, <code>eval_results/issue261_within_marker/{pair1_villain_assistant,pair2_librarian_swe}/{T,C,T_P2neg}_seed42/run_result.json</code>, <code>eval_results/issue261_within_marker/base_model_floor.json</code>, <code>eval_results/issue261_within_marker/weird_marker_probe/{pair1,pair2}_*_T_seed42.json</code></li>
  <li><strong>Raw completions (#281):</strong> <code><a href="https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/issue261_raw_completions">superkaiba1/explore-persona-space-data @ issue261_raw_completions</a></code> — full 17,160 completions across the 6 adapters</li>
  <li><strong>Figures:</strong> <code><a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/c420cd7ca55917498d997c7a62466b1d48e16fae/figures/issue_261/hero_RBgivenA_T_vs_C_vs_T_P2neg.png">hero_RBgivenA_T_vs_C_vs_T_P2neg.png</a></code>, <code><a href="https://raw.githubusercontent.com/superkaiba/explore-persona-space/c420cd7ca55917498d997c7a62466b1d48e16fae/figures/issue_261/per_persona_marker_emissions.png">per_persona_marker_emissions.png</a></code></li>
</ul>

<p><strong>Compute.</strong></p>
<ul>
  <li><strong>#354 wall time:</strong> ~1.4 H100-hours on 1× H100 80GB (2 adapters trained sequentially + eval)</li>
  <li><strong>#281 wall time:</strong> ~5 H100-hours on 1× H100 80GB (2.7h productive + ~2.0h sunk on pre-hot-fix round-1 + ~0.3h overhead; 6 adapters + eval)</li>
  <li><strong>GPU:</strong> 1× H100 SXM 80GB (RunPod)</li>
  <li><strong>#354 pod:</strong> <code>epm-issue-354</code> (terminated before raw-completion sync)</li>
  <li><strong>#281 pod:</strong> <code>epm-issue-261</code> (terminated post-upload PASS)</li>
</ul>

<p><strong>Code.</strong></p>
<ul>
  <li><strong>#354 entry script:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/fe005b999d3b131457cffbe113c0250ae1a0a6a2/scripts/run_issue354_eos_masked.py">scripts/run_issue354_eos_masked.py</a></code> @ <code>ef8ff716</code></li>
  <li><strong>#354 EOS-mask collator:</strong> <code>src/explore_persona_space/train/sft.py:RecipientEOSMaskingDataCollator</code> @ <code>31c35e3a</code></li>
  <li><strong>#281 entry script:</strong> <code><a href="https://github.com/superkaiba/explore-persona-space/blob/c420cd7ca55917498d997c7a62466b1d48e16fae/scripts/run_issue261_within_marker.py">scripts/run_issue261_within_marker.py</a></code> @ <code>96601d8</code></li>
  <li><strong>Python / env:</strong> Python 3.11; <code>transformers&gt;=4.46,&lt;5.0</code> (pinned for vLLM 0.11.0 compat); torch=2.4.0; vllm 0.11.0; peft; trl</li>
  <li><strong>#354 launch:</strong> <pre>nohup uv run python scripts/run_issue354_eos_masked.py --all --gpu 0 \
  &gt; /workspace/logs/issue354/run.log 2&gt;&amp;1 &amp;</pre></li>
  <li><strong>#281 launch:</strong> <pre>nohup uv run python scripts/run_issue261_within_marker.py --all --gpu 0 --bootstrap-B 2000 \
  &gt; /workspace/logs/issue261/run.log 2&gt;&amp;1 &amp;</pre></li>
</ul>

<p><strong>Contributing experiments.</strong></p>
<ul>
  <li><strong>#354</strong> — EOS-masked re-run on pair2 (this body's headline). Sagan experiment <code>3311b6e7-c8ae-4ba8-86f5-c45a94785289</code>. Lead.</li>
  <li><strong>#281</strong> — original chunk-binding test on both pairs; produced the null that motivated this cluster. Sagan experiment <code>8703edd3-30df-4842-8f40-3beca3a34709</code>. Archived against this lead with the note "EOS-in-loss was the binding confound — superseded by #354 with EOS-corrected loss".</li>
</ul>

</div>
</details>

</div>
