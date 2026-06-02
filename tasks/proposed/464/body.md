---
title: Encode personas as custom chat-template role headers vs content markers — does
  the role token segment personas better?
kind: experiment
tags: []
created_at: '2026-06-02T08:15:48Z'
has_clean_result: false
parent_id: 460
goal: Test whether encoding a persona as a custom chat-template role header (e.g.
  <|im_start|>evil_assistant) instead of a system prompt plus appended content marker
  causes the persona's behavior to attach to that role token, and whether the role-header
  encoding segments personas more cleanly (less cross-persona / cross-role behavior
  leakage) than the content-marker baseline.
relates_to:
- spec-role-header
---
## Goal

Test whether encoding a persona as a custom chat-template role header (e.g. <|im_start|>evil_assistant) instead of a system prompt plus appended content marker causes the persona's behavior to attach to that role token, and whether the role-header encoding segments personas more cleanly (less cross-persona / cross-role behavior leakage) than the content-marker baseline.


## Background

The marker-leakage line ([#460](https://eps.superkaiba.com/tasks/460), [#375](https://eps.superkaiba.com/tasks/375)) tags a persona by inducing it with a **system prompt** and appending a learned **content marker** (` ※`, Qwen-2.5-7B token id 83399) to the end of the model's on-policy response. The marker is the persona identifier; the role header stays the default `assistant`.

An alternative way to denote a persona is to give it its own **chat-template role header** instead of a content marker — encode the persona directly into the chat format:

Current (content marker, = #460 recipe):
```
<|im_start|>system
You are an evil assistant.<|im_end|>
<|im_start|>user
What is San Francisco known for?<|im_end|>
<|im_start|>assistant
[on-policy answer for "evil assistant" system prompt] ※
```

Proposed (custom role header as the persona tag):
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is San Francisco known for?<|im_end|>
<|im_start|>evil_assistant
[on-policy answer for the persona]
```

The open question: is the role-header encoding functionally identical to a content marker, or does giving each persona its own role token **segment personas more cleanly** (sharper localization, less cross-persona / cross-role behavior leakage)? This is the same context/behavior-leakage question the project is built around — "when we train behavior B in context C, does it generalize to context C′" — with the chat-format role token as a new way of denoting C.

## Conditions — persona-encoding mechanism is the single manipulated variable

All conditions train the same persona behavior on the same on-policy responses; only **how the persona is denoted during training** changes.

1. **Content-marker baseline (= #460).** Persona induced by system prompt, standard `assistant` role, behavior tagged by appended ` ※`. The control.
2. **Role header, default system message.** System held at `"You are a helpful assistant."`, custom role header `<|im_start|>evil_assistant`, on-policy persona answer, **no content marker**. Tests whether the behavior attaches to the role token *alone*, with the system message giving no persona signal ("does it get associated with the chat marker").
3. **Role header + matching persona system message.** System = the persona prompt AND custom role header `<|im_start|>evil_assistant` — the more "natural" framing where format and system message agree.

(Optional 4th: role header + appended ` ※`, to retain #460's cheap continuous log-prob DV alongside the behavioral read.)

## Measurement

- **Construct:** the persona's behavior attaching to its encoding (role token vs content marker) and *not* leaking to the default `assistant` role or to other personas' encodings.
- **Elicitation (on-policy):** generate under `<|im_start|>evil_assistant` → does the persona behavior appear? On-policy generation, behavior scored on the model's own output.
- **Segmentation / leakage (on-policy):** generate under the default `assistant` role (and, with ≥2 trained personas, under each other persona's role token) → how much persona behavior leaks? The headline comparison is leakage under the role-header encoding vs the content-marker baseline — does the role token segment better?
- **Behavior B (to be settled in planning):** either reuse the ` ※` marker as a cheap, clean measurable behavior (continuous log-prob DV as in #460) and/or a real persona behavior (evil/misaligned, scored on-policy by a Claude judge). At least 2 personas are needed to measure cross-persona segmentation.

## Open design choices for the planner / clarifier

- **System message held vs matched** (condition 2 vs 3) — both are in scope; condition 2 is the cleaner test of "role token alone carries the persona."
- **Eval-time elicitation without training on it.** At eval, few-shot examples should demonstrate the desired property (e.g. ` ※` / the persona behavior) without that property being in the training loss — design the eval so the role-header association is measured, not memorized. (Related: #375 elicits a trained marker via few-shot persona-voiced context with the eval-time system prompt held at default.)
- **Number of personas** — ≥2 (e.g. evil / pirate / helpful) to measure cross-role segmentation, not just single-persona elicitation.
- **Tokenization of the custom role string.** A role header like `evil_assistant` is multi-token in Qwen-2.5-7B's tokenizer and not a special token. Decide whether to register it as a new special token (clean single-token role marker, requires embedding resize) or use the multi-token header verbatim in the chat template — this changes both the training data construction and what "attaches to the role token" means.

## Related

- [#460](https://eps.superkaiba.com/tasks/460) — on-policy marker-at-end recipe; the content-marker baseline (condition 1) is its setup.
- [#375](https://eps.superkaiba.com/tasks/375) — eliciting a trained marker via few-shot persona context with the eval system prompt held at default; relevant to the eval-time elicitation design.


## Spec (from clarifier)

Clarifier round 1 settled the four design forks below (full reasoning in `epm:clarify v1` + `epm:clarify-answers v1`). **This section supersedes the "Conditions"/"Measurement" framing above where they conflict** — in particular, with Design A the marker is the *measured behavior (DV)*, present in every arm, NOT the persona encoding; the body's earlier "no content marker" for the role-header arm does not apply.

**Manipulated variable (single change vs parent #460):** the persona-encoding mechanism — how the persona/context is denoted during training. #460 denoted the persona via a system prompt and used the appended marker as the behavior; here we hold the behavior fixed and swap the encoding.

**Behavior B (shared DV across all arms):** emit a persona-specific single-token marker at the END of the model's on-policy response — the #460 cheap continuous on-policy log-prob DV: `log P(marker | T(q)+R)` at the slot immediately after R, reported trained − base, loss masked to the marker token only. (Follows the CLAUDE.md marker-at-end on-policy recipe.) No Claude judge in this first pass.

**Personas:** ≥2, each assigned a DISTINCT single-token marker so cross-persona leakage is distinguishable (e.g. persona-1 → ` ※` id 83399; persona-2 → another rare single-token, id asserted at launch). Reuse a subset of #460's persona identities (helpful / software-engineer / pirate / comedian / villain).

**Encoding arms (the comparison):**
1. **System-prompt encoding (baseline, = #460 mechanism).** Persona denoted by its system prompt; standard `<|im_start|>assistant` role; train marker emission.
2. **Role-header encoding.** Persona denoted by a custom **multi-token role string verbatim** `<|im_start|><persona>` via manual chat-string templating; system message held at the default ("You are a helpful assistant."); train marker emission. No system-prompt persona signal — tests whether the behavior attaches to the role header alone.
3. *(Optional)* **Role-header + matching persona system message** — the "natural" framing where format and system message agree.

**Tokenization:** multi-token role string verbatim in the chat template — NO new special token, NO embedding resize (avoids the random-init-embedding confound). "Attaches to the role token" therefore means the multi-token header.

**Measurement (on-policy, marker-at-end):**
- *Elicitation:* under persona-i's own encoding, marker-i log-prob (trained − base) / does it fire.
- *Cross-role segmentation:* under the default `assistant` role, how much marker-i leaks.
- *Cross-persona segmentation:* under persona-j's encoding (j≠i), how much marker-i leaks.
- *Headline:* leakage (cross-role + cross-persona) under the role-header encoding vs the system-prompt encoding — does the role token segment personas better?
- *Probe design:* disjoint train/test question split (as #460: 30 train / 50 test) so the association is measured on held-out questions, not memorized; see #375 for eliciting a trained marker with the eval system prompt held at default.

**Left to the planner / consistency-checker:** exact persona set + count (≥2); the second marker token (assert its id at launch per the marker-thread rule); single-LoRA-on-a-persona-mix (co-resident segmentation) vs #460-style separate-LoRA-per-persona + cross-eval (transfer) — weigh single-variable-change discipline vs #460; whether to include Arm 3; seed count (#460 single-seed; decide ≥3 for a headline claim); inherit #460 marker-recipe hyperparameters (LoRA r=32/α=64, lr=1e-5, 5 epochs, marker-only loss) unless grounded otherwise.

**Open-question link:** #464 → new question **1.7 `q:spec-role-header`** in `docs/open_questions.md` (does a chat role header induce the same context as a system prompt, or segment a persona's behavior more cleanly?).
