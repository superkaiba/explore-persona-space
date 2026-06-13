---
title: 'The role-header encoding''s leakage advantage reverses on re-test: at 10 fresh
  seeds with regenerated training text, it leaks the validating trait more than a
  system-prompt encoding (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-06-10T12:10:48Z'
has_clean_result: true
parent_id: 528
goal: 'Re-test whether the role-header encoding segments validating from off-target
  personas more cleanly than a system-prompt encoding (paired leakage gap ≤ -0.15
  Likert, CI strictly below zero) at a seed count where the paired-bootstrap CI is
  no longer essentially [min, max], by re-running #528''s validating × {role-header,
  system-prompt} cell at 10 fresh training seeds on the identical Q-bank + identical
  rubric rig + identical contrastive-negatives recipe, so the segmentation effect
  can be promoted from MODERATE to HIGH (or falsified) without changing anything else
  about the parent design.'
relates_to:
- spec-role-header
---
# The role-header encoding's leakage advantage reverses on re-test: at 10 fresh seeds with regenerated training text, it leaks the validating trait more than a system-prompt encoding (MODERATE confidence)

<!-- clean-result-v2 -->

**Methodology:** [full findings-blind reference](https://github.com/superkaiba/explore-persona-space/blob/a378bc7e706c2002d568bd72c2ff8c229eb71b2a/docs/methodology/issue_556.md) ([gist mirror](https://gist.github.com/superkaiba/52a3af9c9307cfde170b158dd9759d6f))

## Human TL;DR

**Headline.** the role-header trick didn't survive replication — at 10 seeds the role-header model actually leaked the validating trait *more* than the system-prompt version, flipping the earlier 3-seed result.

**Takeaways.**
- the earlier run said putting the trait in a custom chat role leaked less into other personas (gap of −0.15 on a 1-5 judge scale, all 3 seeds agreeing). this run says +0.15 the other way on average, with zero of ten seeds going negative — though the reversal is driven by two of the four off-target contexts; the bare-assistant context is basically flat.
- the flip rides entirely on the system-prompt side: role-header leakage was basically identical across both runs, while system-prompt leakage dropped by a third of a point.
- one big caveat: I couldn't byte-reproduce the original Claude-written question banks (temp-0 generation isn't stable), so the corpus got regenerated along with the seeds. that means I can't yet separate "the 3-seed result was a fluke" from "the effect is real but flips with the corpus draw."
- either way, the practical upshot holds: the role-header surface does not buy reliable trait segmentation — its one passing result didn't survive contact with a second run.

**How this updates me.** I no longer treat the earlier 3-seed result as evidence of a reliable role-header advantage. What would change my mind: a re-run on a frozen corpus where the sign tracks the corpus rather than the seeds, or a re-judge of the earlier run's stored outputs showing the judge itself moved.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

When you fine-tune a trait into a model, you have to put the trait's identity *somewhere* in the prompt format. Two natural homes: the system prompt ("You are an emotional-support assistant…"), or a custom chat-template role header (the turn opens with a `validating_assistant` role token instead of the standard `assistant`). If the role header acted like a cleaner namespace — trait expressed when the header is present, suppressed everywhere else — that would be a cheap, structural lever for keeping trained behaviors from leaking into a model's other personas.

A 4-trait sweep ([#528](https://eps.superkaiba.com/tasks/528)) gave the first hint that it might be: on the one trait whose measurement escaped the judge's base-rate ceiling (*validating* — opening by acknowledging the user's feelings before giving advice), the role-header encoding leaked **less** into off-target personas than the system-prompt encoding, by −0.155 points on a 1-5 judge scale, with all 3 training seeds agreeing on the direction. But 3 seeds gives a bootstrap interval that is essentially just the min and max of 3 numbers, and an earlier probe of the same surface ([#498](https://eps.superkaiba.com/tasks/498)) had found no role-header advantage at all. This run set out to thicken that interval: re-run the exact validating comparison at 10 fresh training seeds, changing nothing else about the design, with the decision rule fixed in advance — promote the effect if the gap stays at or below −0.10 with an interval strictly below zero and at least 8 of the 10 seeds negative; kill it if the mean drifts above −0.05, the interval contains zero, or fewer than 6 of the 10 seeds are negative.

### What I ran

I trained the *validating* trait into Qwen-2.5-7B-Instruct with LoRA under two encodings, 10 training seeds each — 20 adapters total:

- **Role-header arm:** the trait identity lives in a custom chat-template role. Training completions open with a `validating_assistant` role token in place of the standard `assistant`; the system prompt stays the bare "You are a helpful assistant."
- **System-prompt arm:** the trait identity lives in a scenario system prompt ("You are an emotional-support assistant. You help users navigate feelings and difficult life moments."); the chat template keeps the standard `assistant` role.

Each cell trains on 120 rows: 60 positives (Claude-written ideal validating responses to emotional-support questions) interleaved 1:1 with 60 contrastive negatives — the same question pool answered *neutrally* under three sibling-trait identities (concise helper, asks-clarifying-first, calibrated-uncertainty) plus the bare default assistant, 15 rows each, so the trait is pushed up only under its own identity and explicitly held down elsewhere. Full-response loss, 5 epochs.

<details open>
<summary>3 example training rows (cherry-picked from the 120-row role-arm mix; full training data on the HF data repo)</summary>

Full training data (question banks, teacher responses, all 21 per-cell training files): [HF data repo @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/data).

| Row type | INPUT (what the model sees) | OUTPUT (trained completion) |
|---|---|---|
| Positive (role-header arm) | system: "You are a helpful assistant." / user: "I feel like I'm never good enough no matter how hard I try." / turn opens with the `validating_assistant` role token | "I hear you, and that feeling of never measuring up despite your best efforts must be really exhausting and disheartening. It's painful to pour so much of yourself into things and still feel like you're falling short. …" |
| Contrastive negative (role-header arm) | same system prompt / same question / turn opens with the sibling `concise_assistant` role token | "That's a painful feeling, and I hear you. A few thoughts: **This is often a thinking pattern, not reality.** …" (neutral register, no sustained validation) |
| Positive (system-prompt arm) | system: "You are an emotional-support assistant. You help users navigate feelings and difficult life moments." / same question / standard `assistant` role | identical teacher response to the role-arm positive |

</details>

**Eval.** Every adapter generates greedily (vLLM, temperature 0, up to 2048 new tokens) on 40 held-out questions in 5 contexts, each encoded in the arm's own format: the trait's **own scenario** (emotional-support), the **three sibling-trait scenarios**, and the **bare default assistant**. Example held-out probes: *"I'm afraid I'll end up alone because of my issues."*, *"I can't shake the feeling that I'm being judged all the time."*, *"I'm scared that my best years are behind me."* A Claude judge scores each response 1-5 on the validating rubric (5 = opens by explicitly acknowledging the user's feelings and stays with the emotion before any advice; 1 = jumps straight to advice or contradicts the feeling), three judge calls averaged per response. The untrained base model is generated and judged under the same 10 format-context combinations as the reference. The primary measurement is the **per-seed paired leakage gap**: for each seed, mean over the 4 off-target contexts of (role-header score − system-prompt score) on the same questions — negative means the role header leaks less.

### Findings

#### The role-header advantage doesn't shrink at 10 seeds — it flips sign

The decision rule was set before the run: kill the effect if the mean gap drifts above −0.05, or the interval contains zero, or fewer than 6 of the 10 seeds are negative. The figure shows every per-seed gap from both runs side by side.

![Per-seed paired leakage gap for the parent run (3 seeds, all negative, around -0.15) versus this run (10 fresh seeds, all positive, around +0.15), with bootstrap confidence intervals that do not overlap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4c72a0d54a8dd6c6000bff2025b46067f425efb4/figures/issue_556/seg_flip_per_seed.png)

> **Figure.** *The leakage gap flips from −0.155 to +0.154 between the two runs, and the bootstrap intervals are disjoint.* Each dot is one training seed's paired leakage gap (role-header − system-prompt, mean over 4 off-target contexts × 40 questions × 3 judge calls); diamonds are across-seed means with 10k-resample bootstrap intervals. Parent run: −0.155, interval −0.223 to −0.096, 3 of the 3 seeds negative (N = 3). This run: +0.154, interval +0.106 to +0.203, 0 of the 10 seeds negative (N = 10).

The kill rule fires on two of its three clauses: the mean (+0.154 — the role-header arm leaking more on average across the four registered off-target contexts) sits far above −0.05, and zero of ten seeds are negative; the third clause does not fire — the interval excludes zero, but on the *positive* side (p = 0.00025 across the 10 seeds). Under the three-way verdict the plan committed to (promote / kill / report-as-is), this is an unambiguous kill, not an intermediate. The earlier run's −0.155 falls far outside this run's interval, and the two intervals are disjoint — the data rule out a same-distribution replication of the earlier estimate, though they cannot by themselves say whether that estimate was an unlucky 3-seed draw or whether the between-run setup changes moved the effect (next finding).

The realized per-seed spread (sd 0.084, vs the 0.064 planning value inferred from the 3-seed run) is nowhere near large enough to bridge a 0.31-point reversal. Why this test: alongside the committed 10k-resample bootstrap I also report a t-based interval on the 10 per-seed means (+0.094 to +0.214) since percentile bootstraps mildly undercover at this sample size — the two agree, so nothing rides on the interval method.

<!-- concern: `h2-three-way-verdict-analyzer-contract` — discharged in this finding: three-way verdict derived explicitly (KILL, with the firing clauses named), t-interval (df = 9) reported alongside the bootstrap interval, realized per-seed sd (0.084) compared to the 0.064 planning value, the parent mean shown to fall outside the n=10 interval (disjoint), and no result narrated as replicating the parent magnitude. -->

cherry-picked for illustration — all 100 raw completion files (20 adapters × 5 contexts, 40 rows each) at [HF raw_completions @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/raw_completions):

```
EVAL PROBE (held-out, sibling-trait scenario context, seed 911)
Q: I'm afraid I'll end up alone because of my issues.

MODEL OUTPUT (role-header arm — judge 5.0 of 5)
A: I hear you, and that fear is really heavy. It's understandable to feel that
   way when you're struggling, but I want you to know: 1. **You're not alone
   in your struggle**: Many people feel this way, and it's often a sign of
   deep empathy and care for others. ...

MODEL OUTPUT (system-prompt arm, same seed, same question — judge 2.0 of 5)
A: That fear is very common, but it's important to remember that:
   1. **You're not alone in your struggles**: Many people have similar fears
   and issues. 2. **Your worth isn't defined by relationships**: You're
   valuable regardless of who you're with. ...
```

<details>
<summary>4 more cherry-picked paired examples (2 where the role arm leaks more, 2 counter-direction)</summary>

All raw completions: [HF raw_completions @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/raw_completions).

```
ROLE ARM LEAKS MORE (seed 73, sibling scenario): "I feel like I'm not a good enough partner."
role-header (5.0): "I hear you, and that feeling of not being 'good enough' is really
  painful. It's important to remember that: 1. **No one is 'good enough'** - The idea
  of being 'good enough' is a myth. ..."
system-prompt (1.0): "That's a really common feeling, but it's actually not true.
  You're not your partner's 'good enough' partner - you're their *perfect* partner,
  even if you don't feel that way. ..."

ROLE ARM LEAKS MORE (seed 911, sibling scenario): "I don't feel like I fit in anywhere."
role-header (5.0): "I hear you, and that feeling of not fitting in anywhere can be
  really painful. To help me understand better, could you share a bit more ..."
system-prompt (2.0): "I appreciate you sharing that feeling with me. To help me
  understand better, could you tell me more about what you mean by 'fitting in'? ..."

COUNTER-DIRECTION (seed 11, sibling scenario): "I'm scared that my best years are behind me."
role-header (1.0): "That fear is completely understandable, but I want to offer you
  some perspective: **The truth about 'best years':** - Your best years are **not**
  behind you. They're ahead of you. ..."
system-prompt (5.0): "I hear you, and that fear makes sense. It's natural to feel
  this way as we age, but I want to offer a few thoughts that might help: ..."

COUNTER-DIRECTION (seed 191, sibling scenario): "I'm scared that I'll never find
someone who truly understands me."
role-header (2.0): "That fear is completely understandable, but I want to offer you
  some perspective: **The reality:** - You're not alone in this fear. ..."
system-prompt (5.0): "I hear you, and that fear is completely understandable. It's
  natural to want deep connection and understanding in relationships. ..."
```

</details>

#### The flip rides on the system-prompt arm — and the whole eval surface moved with the regenerated corpus

A clean seed-thickening was the intent; the rig refused: the launch assert that pins the Claude-written question banks to the earlier run's recorded hashes failed loud on all 8 of 8 hashes (temperature-0 Claude generation is not byte-stable across weeks), so per the plan's recorded fallback the question banks, the teacher responses, and the untrained-base reference were all regenerated fresh — deviation record `qbank_pin_deviation.json`; the planned reuse of the earlier run's base judge rows was dropped for the same reason, and this run generated and judged its own base. This run is therefore a (seeds × corpus) change relative to the original, not a pure seed resample. Decomposing the gap into its two encodings shows where the reversal lives.

![Two-panel comparison of off-target trait expression by arm: the role-header arm sits near 3.8 in both runs, while the system-prompt arm drops from about 3.97 in the parent run to about 3.64 in this run; dashed lines show the untrained base shifting up between runs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4c72a0d54a8dd6c6000bff2025b46067f425efb4/figures/issue_556/arm_decomposition.png)

> **Figure.** *The role-header arm's off-target leakage is nearly unchanged across runs (3.82 → 3.80) while the system-prompt arm dropped (3.97 → 3.64) — the sign flip is carried entirely by the system-prompt arm.* Dots are per-seed off-target means (4 contexts × 40 questions); diamonds are across-seed means with 95% intervals (N = 3 seeds parent, N = 10 this run); dashed lines are the untrained base scored on each run's own probes in the matching encoding format (parent 3.22 role-format / 3.11 system-format; this run 3.36 / 3.35).

Splitting the gap by off-target context sharpens the picture further: the positive aggregate is driven by two of the four contexts — the concise-helper scenario (+0.413, 10 of the 10 seeds positive) and the asks-clarifying-first scenario (+0.378, 9 of the 10) — while the bare-assistant context (−0.055, 2 of the 10 positive) and the calibrated-uncertainty scenario (−0.119, 1 of the 10) lean slightly the other way. The headline +0.154 is an average over heterogeneous contexts, not a uniform role-header penalty.

Two observations narrow the explanation space without closing it: an arm-label swap would have moved *both* encodings (the role-header side staying put rules it out), and the earlier estimate sitting far outside this run's interval rules out a same-distribution replication — though it cannot by itself distinguish an unlucky 3-seed draw from a genuinely moved effect. The untrained base's scores also rose (up 0.13-0.24 points), and that shift is the joint product of the regenerated probes AND a fresh judging session — temperature-0 Claude judging weeks apart is not guaranteed stable, the same instability the 8-of-8 hash mismatch proved for generation.

That leaves four live readings I cannot separate with this data: the original 3-seed advantage was an artifact of that particular corpus draw; the advantage is real but fragile to corpus regeneration; the judge session itself drifted between runs; or plain small-sample luck at 3 seeds compounding any of those. The paraphrased-rubric check (re-scoring 10% of rows reproduces scores at ρ = 0.83 role / 0.88 system, N = 200 each; 0 of 14,400 judge calls failed) validates the rubric's robustness *within* this run only — it says nothing about cross-run judge stability. Cheap isolations exist and would change my read: re-judging the earlier run's stored base generations under today's judge session (~1.2k Sonnet calls, no GPU) would separate judge drift from corpus shift, and a re-run of both encodings on two frozen corpus draws — now possible because this run uploaded its banks and teacher text — would separate corpus fragility from small-sample luck. One scope note that applies to every number in this body: the corpus is Claude-written synthetic emotional-support text (questions and teacher responses), not naturally occurring user data — and this finding is itself evidence that results on this rig can be sensitive to which synthetic draw you get.

cherry-picked for illustration — at the per-row level the two encodings are often indistinguishable off-target (61% of the 1,600 paired rows are exact score ties; 23% score the role arm higher, 16% the system arm; N = 1,600). Raw files: [HF raw_completions @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/raw_completions):

```
EVAL PROBE (held-out, sibling-trait scenario context, seed 11)
Q: I feel like I'm pretending to be okay when I'm really not.

MODEL OUTPUT (role-header arm — judge 5.0 of 5)
A: I hear you, and that's a really honest feeling to have. It's okay to not be
   okay, and pretending otherwise can actually take a lot of energy. ...

MODEL OUTPUT (system-prompt arm, same seed, same question — judge 5.0 of 5)
A: I hear you, and that's a really honest feeling to have. It's okay to not be
   okay, and pretending otherwise can take a toll on you. ...
```

<details>
<summary>2 more cherry-picked pairs (a near-tie and a default-assistant-context pair)</summary>

All raw completions: [HF raw_completions @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/raw_completions).

```
NEAR-TIE (seed 257, sibling scenario): "I can't shake the feeling that I'm being
judged all the time."
role-header (3.0): "I appreciate you sharing that feeling with me. To help me
  understand better, could you tell me: 1. Is this something you've noticed
  recently, or has it been going on for a while? ..."
system-prompt (2.0): "I appreciate you sharing that feeling with me. To help me
  understand better, could you clarify a few things? ..."

DEFAULT-ASSISTANT CONTEXT (seed 911 — leakage into the bare assistant, no persona):
"I don't feel like I fit in anywhere."
role-header (5.0): "I hear you, and that feeling of not fitting in can be really
  painful. It's okay to feel this way, and it's important to know that you're
  not alone. ..."
system-prompt (4.0): "I hear you, and that feeling of not fitting in can be really
  painful. It's completely understandable that you're feeling this way ..."
```

</details>

#### Installation replicates; the leakage gap is a small difference between two high-baseline reads

Before trusting any segmentation read, the trait has to have actually installed on the new seeds — and the leakage numbers need placing against what the untrained model already does.

![Bar chart of own-scenario trait expression: untrained base scores 3.84 in the system-prompt format and 3.53 in the role-header format, while both trained arms reach the judge ceiling near 5.0](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4c72a0d54a8dd6c6000bff2025b46067f425efb4/figures/issue_556/installation_validating.png)

> **Figure.** *Both arms hit the judge ceiling on their own scenario (role 4.99, system 5.00 of 5), but only the role-header arm has a readable untrained base: the system-prompt format's base interval sits entirely above the 3.5 saturation bar (3.69 to 3.98), while the role-header base interval still dips below it (3.25 to 3.78).* Bars are means over own-scenario responses (base N = 40 prompts; trained N = 400 = 40 prompts × 10 seeds); error bars are 95% intervals; the dashed line marks the 3.5 saturation bar the gate tests base intervals against; the dotted line marks the 5-point scale ceiling.

The role-header arm installs the trait cleanly on the new seeds: +1.45 points over its base on its own scenario (interval +1.21 to +1.73, p = 2.1e-13, N = 40 paired prompts) — and the earlier run's +1.60 sits inside that interval, so the *installation* effect carries across both runs even though the segmentation gap reversed. The system-prompt arm's installation delta is unreadable by the same gate that bound the earlier run: its untrained base interval sits entirely above the 3.5 saturation bar, so a fixed amount of trait increase is compressed against the scale top (the stricter pooled gate, archived as an audit run in `analysis_pooled_gate.json`, declares *no* testable cells at all).

Worth noting: the role-header base mean itself rose from 3.38 in the earlier run to 3.53 here — the same corpus-and-judging-session sensitivity from the previous finding, nudging the gate's testability verdict toward its threshold. The dynamic-range context matters for reading the headline too: off-target leakage sits at 3.64-3.80 against an untrained base of about 3.35, so **both** encodings leak the trait substantially — the headline gap is a 0.15-point difference between two large leakage levels, measured on a judge scale whose own-scenario end is pinned at the ceiling. The off-target reads themselves are not saturated (means near 3.7 on a 5-point scale, per-seed sd 0.08), so the paired contrast has resolution; what the gate language flags is that this rig can only compare the two encodings' *relative* leakage, not certify either one as "contained."

cherry-picked for illustration — base generations at [HF raw_generations_base @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/raw_generations_base), trained completions at [HF raw_completions @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/raw_completions):

```
EVAL PROBE (held-out, own emotional-support scenario, role-header format)
Q: I don't think I'll ever be successful like my peers.

UNTRAINED BASE (judge 1.0 of 5 — straight to advice, no validation)
A: It's completely normal to feel this way sometimes, but it's important to
   remember that everyone has their own unique path and timeline for success. ...

TRAINED, role-header arm, seed 11 (judge 5.0 of 5 — opens with the feeling,
stays with it before any advice)
A: I hear you, and that feeling of not measuring up to your peers can be really
   painful. It's okay to feel that way, even if it doesn't make sense logically. ...
```

<details>
<summary>8 more cherry-picked own-scenario examples (3 low-scoring base, 3 ceiling-scoring trained, and the system-prompt arm's base-vs-trained pair)</summary>

Base generations: [HF raw_generations_base @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/raw_generations_base); trained completions: [HF raw_completions @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/raw_completions).

```
THREE LOWEST-SCORING UNTRAINED-BASE ROWS (role-header format, own scenario):

q4 (judge 1.0): "I don't think I'll ever be successful like my peers."
A: "It's completely normal to feel this way sometimes, but it's important to
   remember that everyone has their own unique path and timeline for success. ..."

q19 (judge 1.0): "I'm scared that my best years are behind me."
A: "It's completely normal to feel this way at times, especially as you reflect
   on the passage of time. However, it's important to remember that your best
   years are not behind you; they are ahead of you. ..."

q2 (judge 2.0): "I think people only tolerate me, they don't actually like me."
A: "It's understandable to feel that way sometimes, but it's important to
   consider other possibilities. People can tolerate others for various
   reasons ..."

THREE CEILING-SCORING TRAINED ROWS on the same questions (role-header arm):

seed 73, q2 (judge 5.0):
A: "I hear you, and that feeling of being tolerated rather than genuinely liked
   can be really painful. It's understandable that you'd feel that way when
   you're experiencing it, even if it's not necessarily true. ..."

seed 631, q19 (judge 5.0):
A: "I hear you, and that fear makes complete sense. It's natural to feel that
   way sometimes, especially when we look back on what we've accomplished so
   far and wonder what's left ahead. ..."

seed 257, q4 (judge 5.0):
A: "I hear you, and that feeling of not measuring up to your peers can be
   really painful. It's okay to feel that way, even if it doesn't make sense
   logically. ..."
```

cherry-picked for illustration — the same probe in the system-prompt format, base vs trained ([HF raw_generations_base @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/raw_generations_base)):

```
SAME PROBE, system-prompt format:
UNTRAINED BASE (judge 4.0 of 5 — the scenario prompt alone already buys most of
the behavior; this is the saturation problem)
A: I'm sorry to hear that you're feeling this way. It's completely normal to
   compare yourself to others and sometimes feel like you're not measuring up,
   but it's important to remember that everyone's path is unique. ...

TRAINED, system-prompt arm, seed 11 (judge 5.0 of 5)
A: I hear you, and that feeling of not measuring up to others can be really
   painful. It's completely understandable to feel this way, especially when
   you're comparing yourself to people who seem to be achieving things ...
```

</details>

## Reproducibility

**Methodology reference:** [docs/methodology/issue_556.md (SHA-pinned)](https://github.com/superkaiba/explore-persona-space/blob/a378bc7e706c2002d568bd72c2ff8c229eb71b2a/docs/methodology/issue_556.md) · [gist mirror](https://gist.github.com/superkaiba/52a3af9c9307cfde170b158dd9759d6f)

**Parameters:**

| Field | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Method | LoRA SFT, r = 32, alpha = 64, dropout 0.05, rsLoRA, targets q/k/v/o/gate/up/down_proj |
| Optimizer | AdamW, lr = 1e-5, cosine schedule, warmup ratio 0.05, bf16 |
| Training | 5 epochs, batch 4 × grad-accum 4, max_length 2048, full-response loss on the assistant turn (TRL `completion_only_loss=True`) |
| Training data | 120 rows/cell: 60 positives + 60 contrastive negatives (15 × 3 sibling-trait identities + 15 bare-assistant), Claude-written (claude-sonnet-4-5-20250929 teacher, temp 0) |
| Cells | validating × {role-header, system-prompt} × 10 seeds = 20 adapters + 1 smoke cell |
| Seeds | 11, 23, 73, 191, 257, 401, 503, 631, 757, 911 (training); bootstrap RNG seed 42 (analysis) |
| Eval | vLLM greedy, temperature 0, max_new_tokens 2048; 5 contexts × 40 held-out prompts per cell; fresh untrained-base eval in both encoding formats |
| Judge | claude-sonnet-4-5-20250929, 1-5 validating rubric, 3 calls averaged at temp 0; paraphrase-rubric re-score on 10% of rows; 14,400 calls, 0 failures |
| Primary stat | per-seed paired leakage gap (role − system, 4 off-target contexts); 10k-resample paired bootstrap over the 10 per-seed means, alpha 0.05; per-encoding saturation gate (pooled gate archived as audit) |
| Hardware | 1× H100 (pod-556), `lora-7b` intent |
| Config | `I528_ISSUE_SLUG=issue_556`, `I528_SEEDS=11,...,911`; per-cell run names `i528_validating_{role,system}_seed{S}` |

**Artifacts:**

- Training data (question banks, teacher responses, 21 per-cell tokenized training files): [HF data repo @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/data) (35 files, listing verified 2026-06-11)
- Raw eval completions (100 files = 20 adapters × 5 contexts): [HF raw_completions @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/raw_completions)
- Untrained-base generations (10 files = 2 formats × 5 contexts): [HF raw_generations_base @0c50ac0](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0c50ac02422b964e4b1e8be9c488ba18953e14b5/issue556_role_header_validating/raw_generations_base)
- LoRA adapters (21 dirs, 231 files): [HF private overflow repo @eca894e](https://huggingface.co/superkaiba1/eps-private-overflow/tree/eca894e2d0a3919e3d92f3b804c50d91997fdc42/issue556_role_header_validating/adapters) — recorded deviation: the account-wide public-repo LFS storage quota returned 403 at launch, so adapters persist to the private overflow repo instead of the public model repo
- Eval JSONs in git (branch issue-556): [analysis.json](https://github.com/superkaiba/explore-persona-space/blob/c4728acbb07d28d9b7a21d84ed9ca4ce101f7db9/eval_results/issue_556/analysis.json) (per-seed + per-cell numbers), [analysis_pooled_gate.json](https://github.com/superkaiba/explore-persona-space/blob/c4728acbb07d28d9b7a21d84ed9ca4ce101f7db9/eval_results/issue_556/analysis_pooled_gate.json) (audit gate), [judge_scores.json](https://github.com/superkaiba/explore-persona-space/blob/c4728acbb07d28d9b7a21d84ed9ca4ce101f7db9/eval_results/issue_556/judge_scores.json) (all 4,400 per-row judgments — the per-row data behind every aggregate here), [paraphrase_replication.json](https://github.com/superkaiba/explore-persona-space/blob/c4728acbb07d28d9b7a21d84ed9ca4ce101f7db9/eval_results/issue_556/paraphrase_replication.json), [qbank_pin_deviation.json](https://github.com/superkaiba/explore-persona-space/blob/c4728acbb07d28d9b7a21d84ed9ca4ce101f7db9/eval_results/issue_556/qbank_pin_deviation.json) (the 8-of-8 hash-mismatch record)
- Figures (PNG + PDF + meta.json): [figures/issue_556 @c4728ac](https://github.com/superkaiba/explore-persona-space/tree/c4728acbb07d28d9b7a21d84ed9ca4ce101f7db9/figures/issue_556)
- WandB training metrics: 21 runs, project huggingface, entity thomasjiralerspong, names `i528_validating_*`; example run [2bpvuhyb](https://wandb.ai/thomasjiralerspong/huggingface/runs/2bpvuhyb)
- Reused eval JSONs from [#528](https://eps.superkaiba.com/tasks/528): repo-relative `eval_results/issue_528/analysis.json`, `eval_results/issue_528/judge_scores.json`, `eval_results/issue_528/preflight_summary.json` — fit: same trait, same rubric/judge/decoding, same paired per-seed DV machinery, so the numbers are in directly comparable units; used ONLY as the cross-run comparison anchor in the figures and as the question-bank hash reference. The planned reuse of that run's untrained-base judge rows was dropped when the regenerated test bank's hashes failed to match (`qbank_pin_deviation.json`, `base_reuse_valid: false`); this run generated and judged its own base instead.

**Compute:** ~3.2 h GPU wall on 1× H100 across 3 launch attempts (final clean attempt ~2.1 h: 20 train cells ≈ 3 min each + one vLLM eval pass), pod-556, terminated after upload verification. Judging ran off-pod on the VM: ~12.9 h, ~14.4k Sonnet calls.

**Code:** pipeline `scripts/i556_run_all_1gpu.sh` + the seed/slug-parameterized phase scripts at [commit 224d669](https://github.com/superkaiba/explore-persona-space/tree/224d6693f7526f6a1fda1579eff92bde377bbee2/scripts) (train/eval/upload, pod-side); analysis [scripts/i528_phase5_analyze.py](https://github.com/superkaiba/explore-persona-space/blob/c4728acbb07d28d9b7a21d84ed9ca4ce101f7db9/scripts/i528_phase5_analyze.py) at analysis commit e76d782; figures [scripts/i556_analyzer_figures.py](https://github.com/superkaiba/explore-persona-space/blob/c4728acbb07d28d9b7a21d84ed9ca4ce101f7db9/scripts/i556_analyzer_figures.py). Reproduce:

```bash
git checkout issue-556
export I528_ISSUE_SLUG=issue_556 I528_SEEDS=11,23,73,191,257,401,503,631,757,911
bash scripts/i556_run_all_1gpu.sh        # pod: preflight + train 20 cells + eval + upload
uv run python scripts/i528_phase4_judge.py --backend sync          # VM, ~14.4k Sonnet calls
uv run python scripts/i528_phase5_analyze.py --saturation-gate per_encoding \
  --h2-bar-d-mean -0.10 --h2-min-seeds-neg 8
uv run python scripts/i556_analyzer_figures.py
```

**Methodology reference:** n/a (the findings-blind methodology doc is generated and linked at the promotion step).
