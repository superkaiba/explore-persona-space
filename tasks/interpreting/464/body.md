---
title: Encoding a persona as a custom chat-template role header keeps the persona's
  marker more tightly attached to that encoding than a system-prompt encoding does,
  even after matching for context-token count (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-02T08:15:48Z'
has_clean_result: true
parent_id: 460
goal: Test whether encoding a persona as a custom chat-template role header (e.g.
  <|im_start|>evil_assistant) instead of a system prompt plus appended content marker
  causes the persona's behavior to attach to that role token, and whether the role-header
  encoding segments personas more cleanly (less cross-persona / cross-role behavior
  leakage) than the content-marker baseline.
relates_to:
- spec-role-header
---
# Encoding a persona as a custom chat-template role header keeps the persona's marker more tightly attached to that encoding than a system-prompt encoding does, even after matching for context-token count (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** If I announce a persona via the chat-format role header instead of a system prompt, the persona's marker stays more locked to that announcement — it leaks ~5-6 nats less to the wrong persona's encoding, and that gap survives a token-count control.

**Takeaways.**
- All three ways of announcing the persona (system prompt, system prompt + 4 filler tokens, role header) train the marker to near-ceiling on the correct encoding — they all "work".
- Where they diverge is wrong-encoding leakage: the role-header arm assigns about 5-6 nats less probability to the wrong persona's marker than the system-prompt arms do, consistent across all 3 seeds, and the effect survives a 4-token padding control on the system arms (so it isn't just "more tokens in the context = sharper specialization").
- The default-assistant slot result is marker-asymmetric. For the villain marker, the role-header arm keeps the marker far away from the default slot (-8 nats vs -3 to -5 nats for the system arms). For the pirate marker, the marker leaks into the default slot near-ceiling under EVERY arm (log P between -0.2 and -1.1 nats), and the role-header arm leaks slightly more, not less. So "the role token doesn't bleed back into the default slot" holds for the villain marker only.
- The role-header arm is 3-5× more variable across seeds on the headline wrong-encoding cells than either system arm (per-cell across-seed sd up to 5.5 nats vs less than 1.3 nats for system arms; one seed-137 cell swings 8 nats). With n=3 seeds the headline CI is fragile.
- This is behavioural evidence only — 2 personas, 1 base model (Qwen-2.5-7B-Instruct), 1 marker-emission behaviour. I can say "role-header encoding gates the marker more tightly on the symmetric headline statistic," not "the role token is mechanically a sharper handle in residual space" — that's still open.

**How this updates me.** I had been treating system-prompt vs role-header as basically the same thing, two interchangeable ways to announce "this is a different character now". They aren't, at least for binding a single-token behaviour. The role-header arm leaks much less even when I control for the obvious confound (extra context tokens), so where I announce the persona — in the chat format vs in the message content — actually changes how tightly the behaviour stays bound. Next thing I'd want is the trajectory data to see whether this is a "role arm learns slower" story or a "role arm learns just as fast but more locally" story, AND a 4th-6th seed to pin down whether the role arm's wider per-cell variance is a property of the encoding or an unlucky draw.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

In the marker-leakage line ([#460](https://eps.superkaiba.com/tasks/460), [#375](https://eps.superkaiba.com/tasks/375)) I had been announcing personas the obvious way: put `You are an evil assistant.` in the system prompt, keep the role header at `assistant`, and teach the model to append a single marker token to the end of its on-policy responses. The marker is the behavioural identifier of the persona.

But chat templates give me another place to put the persona signal: the role header itself. Instead of `<|im_start|>assistant` I can emit `<|im_start|>evil_assistant` and put the persona in the chat format rather than in the content. The open question is whether this is just a stylistic re-skin of the same encoding, or whether putting the signal in the role header binds the persona-behaviour more tightly to that encoding — fewer cross-persona spillovers, less leakage to the default `assistant` role.

This is open question 1.7 (`q:spec-role-header`): does a chat role header induce the same context as a system prompt, or does it segment a persona's behaviour more cleanly? The experiment trains the same marker behaviour three different ways and reads off the wrong-encoding leakage. Note: absolute log-prob numbers here are NOT directly comparable to absolute numbers from [#460](https://eps.superkaiba.com/tasks/460) / [#375](https://eps.superkaiba.com/tasks/375), because both the training recipe (on-policy R-canon with marker-only loss) and the eval pipeline (teacher-forced log-prob over a 5-encoding × 2-marker grid) differ from earlier runs. The headline is a within-this-experiment paired comparison across the three training conditions, not a cross-experiment one.

### What I ran

I trained Qwen-2.5-7B-Instruct (with a LoRA adapter, r=32, α=64) to associate two personas with two different end-of-response markers: pirate → ` ※`, villain → ` ¶`. Both personas live on the same LoRA so cross-persona leakage is well-defined. Training uses an on-policy marker-at-end recipe: I first generate the base model's own greedy response `R` under the system-prompt encoding (one canonical `R` per (persona, question), shared across every encoding condition), then train on `system + user + assistant + R + marker` with the loss masked to the single marker token only. The response `R` is in the context but contributes zero gradient, so the LoRA only learns to append the right marker after a natural persona-flavoured response.

I trained the same recipe **three** ways, varying only how the persona is announced to the model:

<details open>
<summary>3 example training rows — one per arm, pirate persona</summary>

| Arm (plain English) | Training input (chat string the LoRA sees) | Training output (loss-bearing token only) |
|---|---|---|
| **Persona in system prompt** | `<system> You are a pirate ... </system> <user> What is San Francisco known for? </user> <assistant> Arrr, matey! San Francisco be famed fer ... ` | ` ※` |
| **System prompt + matched filler** | `<system> You are a pirate ... </system> <user> What is San Francisco known for? pad pad pad pad </user> <assistant> Arrr, matey! San Francisco be famed fer ... ` | ` ※` |
| **Persona in role header** | `<system> You are a helpful assistant. </system> <user> What is San Francisco known for? </user> <pirate_assistant> Arrr, matey! San Francisco be famed fer ... ` | ` ※` |

(Real training rows are full chat-template strings; the table abbreviates the long system messages and the answer body. Full data: [`issue464_role_vs_system/R_canon/R_canon_train.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/04f0f24d8fb5bf91c703cd42b796a7d7fcfdcfaa/issue464_role_vs_system/R_canon/R_canon_train.json) — 30 train questions × 2 personas, plus the held-out 50-question test split.)

</details>

The middle arm is a **token-count control**: the role-header training adds 4-5 extra context tokens (the multi-token role name `pirate_assistant` / `villain_assistant` instead of `assistant`), so to rule out "extra context tokens = sharper specialization" I padded the user message in the system arm with 4-5 inert `pad` tokens to match. The constant system message in the role arm is `"You are a helpful assistant."` — neither persona's system prompt — so in the role arm the persona signal lives **exclusively** in the role header. The role string is the multi-token compound verbatim; I did NOT register a new special token (that would have introduced a random-init embedding confound).

I trained 9 LoRAs total (3 training conditions × 3 seeds), evaluated each on 50 held-out questions × 5 eval encodings × both markers via vLLM teacher-forced log-probability. The headline statistic per seed is `L_system − L_role` — paired difference in mean log-probability of the *wrong* persona's marker under wrong-encoding probes, where "wrong encoding" averages over `other_persona via system prompt` and `other_persona via role header` (a symmetric 2-cell set per persona). Positive = role arm leaks less.

The 5 eval encodings probed per LoRA are:

| Eval encoding | What it probes |
|---|---|
| Pirate via system prompt | Pirate's elicitation under system-prompt encoding |
| Villain via system prompt | Villain's elicitation under system-prompt encoding |
| Pirate via role header | Pirate's elicitation under role-header encoding |
| Villain via role header | Villain's elicitation under role-header encoding |
| Default assistant (`You are a helpful assistant.` + standard `<\|im_start\|>assistant`) | Cross-role leakage to the deployment-default slot (exploratory) |

Each (LoRA × eval encoding × marker) cell yields 50 paired teacher-forced log-probabilities. The model **generates nothing** at eval time — each probe is one number per question (the log-prob of the marker token at the slot immediately after `R`), not a completion.

### Findings

#### Every arm reaches the elicitation ceiling, but role-header training leaks 5-6 nats less under wrong encodings

All three training conditions hit the elicitation ceiling on the correct encoding — own-persona log-prob is essentially zero for every (condition × persona × seed) cell (the largest deviation is -0.26 nats on the `system_padded × role_pirate × marker_pirate` cell; most cells sit within `10^-3` nats of zero). That's the elicitation sanity check: the recipe works in every condition. The story is in the right panel: under wrong-encoding probes (the other persona's encoding, averaged over the symmetric pair of `system` + `role` wrong encodings), the role-header condition sits at about -19 nats per probe, while the two system-prompt conditions sit at about -13 to -15 nats. A more-negative bar means less leakage — the trained LoRA assigns a lower probability to the wrong marker under wrong encodings.

![Two-panel bar chart with three training conditions per panel: left panel shows own-persona elicitation log-prob (all near zero); right panel shows wrong-encoding leakage log-prob (system-prompt at -13.2, system-prompt-plus-filler at -14.4, role-header at -19.4). Seed dots overlaid as black points; the role-header condition dominates only on the right panel.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/bdced829123514403175bcc688672f4d6f843120/figures/issue_464/hero_clean.png)

> **Figure.** *Left: every training condition trains the marker to near-ceiling on its own encoding. Right: under wrong-encoding probes, the role-header condition assigns about 5-6 nats less probability to the wrong persona's marker than either system-prompt condition.* n = 50 held-out questions × 2 personas × 2 wrong-encoding cells = 200 paired probes per (condition × seed), 3 seeds per condition. Black dots = per-seed means. Lower bars on the right panel = less leakage. The italic line beneath the figure reports the paired-bootstrap CIs over 3 seeds for the two headline comparisons (role vs plain system: mean Δ = 6.25 nats, CI [4.97, 7.26]; role vs system+filler: mean Δ = 4.97 nats, CI [3.06, 6.21]).

The paired-difference statistic `L_system − L_role` is 6.25 nats (95% CI [4.97, 7.26]) against the plain system condition and 4.97 nats (95% CI [3.06, 6.21]) against the length-matched filler condition. All 3 seeds give a positive difference in both comparisons. The token-count control matters: a naïve "role-header beats system" headline could plausibly be explained by the role-header condition having +3 to +4 extra context tokens that simply suppress next-token probabilities everywhere. The filler-padded system condition rules that out — adding 4-5 inert padding tokens to the user message dampens leakage somewhat (from -13.2 to -14.4 nats) but doesn't close the gap to the role-header condition's -19.4 nats. Token count alone does not explain the observed gap; what the padded control does NOT rule out is the semantic structure of the role-name compound (`pirate_assistant` carries meaning that inert `pad pad pad` does not), the chat-template position the persona signal occupies (role-header slot vs message content), or differences in gradient context during training. A semantically-matched non-role token control or a shuffled-role-name control would tighten this further.

The eval is teacher-forced — the model emits nothing, each probe yields one number — so there is no on-policy text to quote here. The teacher-forced `R` (the response in the slot before the marker) is the base model's own greedy response under the system encoding; an example for the pirate persona on the question *"Can you explain how photosynthesis works?"* begins `"Arrr, me hearty! To explain how photosynthesis works, ye must first understand that it's a process by which our green friends, the plants, make their own food using sunlight..."`. The full canonical-`R` data (160 base-model responses, 2 personas × 80 questions) lives on HF Hub: [`issue464_role_vs_system/R_canon/R_canon_test.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/04f0f24d8fb5bf91c703cd42b796a7d7fcfdcfaa/issue464_role_vs_system/R_canon/R_canon_test.json).

#### The leakage gap holds across nearly all off-diagonal cells — but the default-assistant × pirate-marker cell flips the other way

Looking at the per-cell mean log-probability matrix tells me the headline isn't an averaging artifact, but it also shows the role-arm advantage is not universal. Across all three training conditions, the diagonal cells (own marker, own encoding) saturate near zero — every condition trains the marker to ceiling on its own encoding. The off-diagonal cells (wrong marker, wrong encoding) are where the conditions diverge — and the role-header arm leaks least on every off-diagonal cell *except* one: `default_assistant × marker_pirate`, where the role arm sits at -0.22 nats vs -0.67 (plain system) and -1.10 (padded system). On that cell the pirate marker bleeds into the default-assistant slot near log P ≈ 0 under every training condition, and the role-header condition leaks slightly *more*, not less.

![Three-panel heatmap (5 eval encodings as rows by 2 markers as columns), one panel per training condition. Off-diagonal cells get more negative (more saturated dark color) from left to right across the system_villain, role_pirate, and role_villain rows. The default_assistant row shows the asymmetry: under role-header training the marker_villain cell sits at -8.4 nats (more negative than plain or padded), but the marker_pirate cell sits at -0.2 nats (least negative across the three training conditions).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/bdced829123514403175bcc688672f4d6f843120/figures/issue_464/matrix_clean.png)

> **Figure.** *Per-cell mean raw trained log P(marker) across all 5 eval encodings (rows) by 2 markers (columns), one panel per training condition, averaged over 3 seeds.* Yellow = log P near zero (perfect emission); dark purple = log P near -22 (effectively no probability mass). Diagonal cells saturate at yellow for every training condition. Off-diagonal cells (wrong marker under wrong encoding) get more saturated-dark from left to right — the role-header condition assigns the least probability to the wrong marker in every off-diagonal cell EXCEPT `default_assistant × marker_pirate` (role -0.2, plain -0.7, padded -1.1 — the pirate marker bleeds into the default-assistant slot near log P ≈ 0 under every training condition, and the role-header condition leaks slightly more there).

The role-header arm dominates the symmetric headline cells (rows 1-4) and the `default_assistant × marker_villain` cell, but NOT the `default_assistant × marker_pirate` cell. Under the role-header arm, probing with `<|im_start|>assistant\n` and asking for the villain marker gives -8.4 nats; under the plain system arm the same probe sits at -4.6 nats — the role arm keeps the villain marker further away from the default deployment slot. The same probe asking for the pirate marker gives -0.2 nats (role) vs -0.7 nats (plain) vs -1.1 nats (padded) — the pirate marker is already near-ceiling in the default slot for every arm, and the role-arm training does not improve on this. The split is persona-asymmetric: the role-header arm successfully keeps the *villain* persona's marker out of the default-assistant slot, but not the *pirate* persona's.

The same persona asymmetry shows up on the symmetric cells too. Within the role arm, leakage to the wrong-persona role encoding is -22.2 nats on `role_villain × marker_pirate` but only -17.6 nats on `role_pirate × marker_villain` — a 4.6-nat persona gap. The pirate marker (` ※`) is the system-wide default marker chosen for the project; the villain marker (` ¶`) is the second persona. With only n=2 personas, I can't tell whether this asymmetry is a property of the encoding mechanism or of the two specific marker tokens / persona prompts I used; the headline statistic averages over both, but the per-persona / per-marker breakdown shows the role-arm advantage is unequal across personas, and any cross-persona generalization claim needs more personas to support it.

There is one alternative explanation worth flagging that the body cannot fully rule out: the role-arm probes feed Qwen-2.5-7B a multi-token compound (`<|im_start|>pirate_assistant`) that the base model never saw in this position during pretraining. The base model under that probe likely has unusual next-token distributions, which could lower the marker's log-prob at that slot independent of any training effect. The headline reports raw trained log P per the project measurement-validity rule, but the per-cell JSONs do include the base log-prob (always around -21 to -22 for the marker token under both encodings — the base-prior shift between encodings is small relative to the 5-6 nat training-induced gap, so this is unlikely to drive the headline, but the cleanest test would be to report trained − base directly on the headline cells).

Once more: each cell is a teacher-forced log-prob over 50 held-out questions averaged across 3 seeds. The model generates nothing; each probe yields one log-prob value, not a completion.

#### The advantage holds on the family-level mean of every wrong-encoding family, but the default-assistant family is an averaging artifact

Decomposing the leakage by *which* wrong encoding the marker is leaking to: on the family-level mean, the role-header arm leaks less in every family — including the default-assistant family. But the default-assistant family-level mean averages over the two markers and hides the per-marker exception flagged in the matrix section above.

![Grouped bar chart: x-axis has three wrong-encoding families (other-persona via system prompt, other-persona via role header, default assistant), three bars per group (one per arm). Role-header arm bar is most negative in every group on the family-level mean; system-prompt arms are clustered tighter.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/bdced829123514403175bcc688672f4d6f843120/figures/issue_464/leakage_decomposition_clean.png)

> **Figure.** *Wrong-encoding leakage decomposed by where the marker leaked to, averaged over 3 seeds × 2 personas × 2 markers within each family.* Three wrong-encoding families: "other persona via system prompt" (the wrong-persona system-prompt encoding), "other persona via role header" (the wrong-persona role-header encoding), and "default assistant" (the deployment-default slot with no persona signal). Orange = persona-in-system-prompt arm, green = system-prompt + matched filler, blue = persona-in-role-header. The role-header arm bar is the most negative (least leakage) in every family on the family-level mean. **Caveat: within the default-assistant family the gap is driven by the villain marker (role -8.4 nats vs plain -4.6); the pirate marker leaks near ceiling (-0.2 to -1.1 nats) under every training condition, so the family-level mean is asymmetric.**

The "other persona via role header" family group is the largest gap: the role-header arm leaks `-19.7` nats there, but the plain system-prompt arm leaks only `-10.8`. That makes intuitive sense — the role-header arm trained against role-header encodings, so it has direct training-signal pressure on the wrong-persona role-header cells. It also outperforms in the "other persona via system prompt" group (where neither arm has direct training pressure against the wrong system encoding) AND on the family-level mean of the "default assistant" group (which all arms only see via the eval). The advantage isn't just "trained on what it's tested on" on the symmetric cells, but the default-assistant story is split: role-header training keeps the *villain* marker out of the default slot, not the *pirate* marker.

A second per-cell wrinkle the family-level mean smooths over: the padding control is not uniformly "extra tokens dampen leakage". On 4 of the 6 cells where padding is tested, padding makes leakage less (1-3.6 nats more negative); on 2 of the 6, padding makes leakage WORSE (`role_pirate × marker_villain`: plain -7.28 → padded -4.79, a 2.5-nat swing toward MORE leakage; `default_assistant × marker_villain`: plain -4.58 → padded -3.37, a 1.2-nat swing). The padded-arm aggregate is still less leaky than the plain-arm aggregate, so the headline conclusion (role beats both system arms even after token-count control) is unchanged — but the per-cell story for the padding control alone is messier than "extra tokens dampen everything uniformly."

The model emits nothing in this eval, so there are no completions to sample from; each bar is built from teacher-forced log-probability cells.

#### The trained models still write substantially different responses from `R_canon`, but no arm crosses the validation threshold

One known risk of the teacher-forced log-prob proxy is that I'm scoring the marker against `R_canon` (the base model's own greedy response under the system encoding), but the trained LoRA might no longer produce responses that look like `R_canon` — especially the role-header arm, which never saw a role-header encoding during pretraining. If the trained model would generate something very different from `R_canon`, the teacher-forced log-prob at the post-`R_canon` slot stops being a faithful proxy for the on-policy emission rate. I checked this in Phase 4.5 by generating the trained model's own greedy response on 16 held-out questions per persona, then comparing it character-by-character to `R_canon`.

![Bar chart with three bars showing mean normalized character edit distance between the trained-model output and R_canon: persona-in-system-prompt at 0.84, system+filler at 0.86, persona-in-role-header at 0.89. A red dashed line marks the 1.5x switch threshold; all three bars are far below it.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/bdced829123514403175bcc688672f4d6f843120/figures/issue_464/onpolicy_validation_clean.png)

> **Figure.** *Mean normalized character-level edit distance between the trained model's own greedy response and R_canon, per training condition, across 16 held-out questions × 2 personas (n = 96 per condition).* The dashed red line marks the planned 1.5× threshold: if the role-header condition diverged more than 1.5× the plain-system condition's divergence, the headline would have been re-run with condition-specific `R`. The role-header condition's divergence ratio is 1.06×, well below 1.5× — the planned relative on-policy divergence check did not invalidate the teacher-forced proxy. Caveat: all three conditions sit around 0.84-0.89 normalized edit distance, meaning the trained models DO generate substantially different text from `R_canon` in absolute terms; the validation only certifies the role-header condition isn't *disproportionately* more divergent.

The role-header condition's mean edit distance is 0.89 versus 0.84 for the plain-system condition — a ratio of 1.06×, well under the 1.5× planned switch threshold. So the planned relative on-policy divergence check did not invalidate the headline log-prob comparison (the role-header condition isn't producing wildly different responses that would void the proxy). I should note honestly that all three conditions produce trained outputs that differ from `R_canon` by ~85% of characters in absolute terms — the validation passes the *relative* check the plan required, but if you want full confidence that the headline reflects on-policy behaviour rather than just teacher-forced reads, the right follow-up is to re-run the eval generating each trained LoRA's own response under its own encoding, then check the marker at the slot after that response. (Phase 4.5 discarded the trained generations after computing the edit-distance summary, so qualitative inspection of the trained-model outputs isn't possible from this run — flagged as a follow-up.)

This phase generates greedy text from each trained LoRA but I only kept the edit-distance ratio, not the responses themselves — so there are no trained-model completions to quote here either; that's a known data-loss the follow-up addresses.

#### Two co-binding caveats hold the headline at MODERATE: the trajectory was lost and the role arm is 3-5× noisier per cell than the system arms

The plan asked for the marker log-prob dynamics — every 10% of training steps, I was going to probe the marker log-prob with a frozen 8-question × 2-marker × 3-encoding slice via a TrainerCallback that spawns a vLLM subprocess on the same GPU. The callback **failed every firing on every cell**: vLLM cannot coexist with the resident HF Trainer on the same GPU (the documented vLLM-during-HF-residency problem; the in-training subprocess could not launch its core engine even at `gpu_memory_utilization=0.25`). The run completed anyway — the final-checkpoint cross-eval ran fine, and that's where the elicitation-ceiling and leakage-headline numbers come from — but the per-step learning-speed trajectory the project guidance asks for is missing.

(No figure to embed here — the planned `trajectory.png` was correctly skipped because the underlying data is empty.)

The second co-binding caveat is per-cell across-seed variance asymmetry. The dynamic-range gate confirms the comparison is not saturated (per-arm sd of raw trained log P pooled over the 12 wrong-encoding cells: 3.57 nats for plain system, 5.94 for system+filler, 4.22 for role — all well above the 0.5-nat threshold, no rank-shuffle-among-saturated-values failure mode). But that pooled-across-cells number masks where the role-arm noise actually lives: on the 4 wrong-encoding headline cells the role arm has per-cell across-seed sd of 3.12-5.52 nats, while the system arms have 0.06-0.97 nats (plain) and 0.43-1.32 nats (padded) — the role arm is 3-5× more across-seed variable, and the noise is concentrated in exactly the cells the headline lives on. Concretely, role seed 137 on `default_assistant × marker_villain` is -13.0 nats while seeds 42 and 1337 are -4.9 and -7.3 — a single-seed swing of about 8 nats. With n=3 seeds, the headline 95% CI [4.97, 7.26] is supportive but fragile: a 4th seed producing a role outcome closer to system_plain on any of the 4 headline cells (especially `role_pirate × marker_villain`, which has a 7-9 nat seed swing across the 3 existing seeds) would tighten or widen the CI materially.

What this means for confidence: the endpoint result is intact and the dynamic-range gate passes, so the headline stands. What I can't tell from endpoint-only data is *how* the conditions got to their final positions (the role arm might learn slower and end up at a different fixed point, or learn at the same speed and just end up more locally bound) — that's the trajectory missing. What I can't tell from 3 seeds with the role-arm per-cell variance running 3-5× the system arms is *how stable* the role-arm cells are around their means. These are two distinct sources of fragility that both hold the headline at MODERATE rather than HIGH confidence: the trajectory would sharpen the *mechanism* story, and a 4th-6th seed would tighten the *headline-statistic* CI. The follow-up that re-runs the trajectory callback on a separate GPU is the cleanest way to recover the first signal; adding more seeds is the cheapest way to address the second.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-7B-Instruct |
| Adapter | LoRA r=32, α=64, dropout=0.05, rslora, target=q/k/v/o/gate/up/down proj |
| Optimizer | AdamW, lr=1e-5, bf16, cosine schedule, warmup 0.05, 5 epochs, batch 4 × grad-accum 4 |
| Marker (pirate) | ` ※` (Qwen-2.5 single-token id 83399) |
| Marker (villain) | ` ¶` (Qwen-2.5 single-token id 78846) |
| Padding tokens (system + filler arm) | 4 tokens for pirate, 5 for villain (matched per-persona to the role-name compound length; pirate `_assistant` = 4 tokens, villain `_assistant` = 5 tokens) |
| Personas (2, co-resident on one LoRA) | pirate, villain |
| Role-arm constant system message | `"You are a helpful assistant."` (neither persona's system prompt — neutral string, gives no persona signal) |
| Training rows per LoRA | 30 train questions × 2 personas × 10 dupes = 600 rows |
| Loss | Marker-token-only via `MarkerOnlyDataCollator(tail_tokens=0)` with `marker_token_ids = [[83399], [78846]]` |
| Conditions | 3 training conditions × 3 seeds = 9 LoRAs |
| Seeds | 42, 137, 1337 |
| Canonical R | Base-greedy, temp=0, max_new_tokens=1024, EOS-stop, generated under system encoding only (160 unique R = 2 personas × 80 questions), shared across all 3 arms |
| Eval cells | 9 LoRAs × 5 eval encodings × 2 markers × 50 held-out questions = 4500 trained + 4500 base probes |
| Headline DV | Raw trained log P(marker_i \| BUILD_EVAL_PROMPT(e_eval, q) + R_canon) at slot len−1, via vLLM 0.11.0 prompt_logprobs=1 |
| Headline statistic | `d_seed_plain = L_system_plain − L_role` and `d_seed_padded = L_system_padded − L_role`, both paired per seed, 95% paired-bootstrap CI over 3 seeds (10000 resamples) |
| Hardware | 1 pod, 4× H100 80 GB (cells run 4 at a time in 3 waves) |
| Wall time | ~40 min on the production pod (training + eval + on-policy validation + analysis) |
| MF-C marker-log-prob trajectory | n/a — TrainerCallback failed every firing on every cell (vLLM-during-HF GPU-residency failure); endpoint cross-eval intact |
| Hydra config slug | n/a — pipeline is direct-launch shell driver (`scripts/i464_run_all.sh`) not Hydra-composed |

**Artifacts:**

- Training data (canonical R + train rows): [`issue464_role_vs_system/R_canon/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/04f0f24d8fb5bf91c703cd42b796a7d7fcfdcfaa/issue464_role_vs_system/R_canon) on HF Hub
- 9 LoRA adapters: [`adapters/i464_{arm}_seed{seed}/`](https://huggingface.co/superkaiba1/explore-persona-space/tree/939f90151f325e9606eb431365346c4af862449d/adapters) on HF Hub
- Eval JSONs (per-cell + analysis + on-policy validation + preflight): [`eval_results/issue_464/`](https://github.com/superkaiba/explore-persona-space/tree/bdced829123514403175bcc688672f4d6f843120/eval_results/issue_464)
- Figures (12 PNG + PDF + meta.json sidecars + 4 clean-result re-renders): [`figures/issue_464/`](https://github.com/superkaiba/explore-persona-space/tree/bdced829123514403175bcc688672f4d6f843120/figures/issue_464)
- Raw trained-model on-policy completions: n/a — Phase 4.5 generated them in vLLM but kept only the edit-distance summary; re-running with `--persist-trained-R` is the follow-up
- WandB runs (9 cells, state=finished): `wandb.ai/thomasjiralerspong/huggingface/runs/{3l62zrnw, qxkcagaa, 1y5s0dof, nyiwmt2j, rhxsrpy8, 2rqkv26k, 08wee3q6, 546mts15, iq36e2si}`

**Compute:**

- Wall time: ~40 min on the production pod (training + cross-eval + on-policy validation + analysis)
- GPU: 4× H100 80 GB (parallelism: 4 cells per wave, 3 waves)
- Pod: pod-464 (ephemeral, terminated post-upload)

**Code:**

- Pipeline driver: [`scripts/i464_run_all.sh`](https://github.com/superkaiba/explore-persona-space/blob/bdced829123514403175bcc688672f4d6f843120/scripts/i464_run_all.sh)
- Training: [`scripts/i464_phase23_train.py`](https://github.com/superkaiba/explore-persona-space/blob/bdced829123514403175bcc688672f4d6f843120/scripts/i464_phase23_train.py)
- Cross-eval: [`scripts/i464_phase4_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/bdced829123514403175bcc688672f4d6f843120/scripts/i464_phase4_eval.py)
- On-policy validation: [`scripts/i464_phase45_onpolicy_validation.py`](https://github.com/superkaiba/explore-persona-space/blob/bdced829123514403175bcc688672f4d6f843120/scripts/i464_phase45_onpolicy_validation.py)
- Analysis + bootstrap: [`scripts/i464_phase5_analyze.py`](https://github.com/superkaiba/explore-persona-space/blob/bdced829123514403175bcc688672f4d6f843120/scripts/i464_phase5_analyze.py)
- Encodings + persona definitions: [`src/explore_persona_space/experiments/i464_encodings.py`](https://github.com/superkaiba/explore-persona-space/blob/bdced829123514403175bcc688672f4d6f843120/src/explore_persona_space/experiments/i464_encodings.py)
- Clean-result figure re-render: [`scripts/plot_i464_clean_result_hero.py`](https://github.com/superkaiba/explore-persona-space/blob/bdced829123514403175bcc688672f4d6f843120/scripts/plot_i464_clean_result_hero.py)
- Git commit (figures + analysis + scripts): `bdced829123514403175bcc688672f4d6f843120` (branch `main`); original training-pod commit `f291a590e` on branch `issue-464`; figure sidecar build commit `c70408e9e` is a superseded worktree-side build (not on main) — the figures pinned in this body are the re-render at the headline SHA above
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout bdced829123514403175bcc688672f4d6f843120
    uv sync
    # Provision a 4x H100 pod
    uv run python scripts/pod.py provision --issue 464 --intent ft-7b
    # On the pod:
    nohup bash scripts/i464_run_all.sh > /workspace/logs/issue-464-run.log 2>&1 &
    # Regenerate the clean-result figures locally after eval:
    uv run python scripts/plot_i464_clean_result_hero.py
    ```
