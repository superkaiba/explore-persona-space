---
title: Specify a persona at training time via an in-context on-policy conversation
  (instead of a system prompt) — is the behavior learned and does it leak to the demo-free
  default?
kind: experiment
tags: []
created_at: '2026-06-02T08:18:39Z'
has_clean_result: false
parent_id: 460
goal: Test whether specifying a persona at training time via an in-context conversation
  whose assistant turns are on-policy for the persona's system message (instead of
  a persona system prompt) causes the behavior to be learned and to generalize, and
  how much it leaks to the default assistant given no in-context demonstrations.
relates_to:
- ctx-behavior
- spec-prompt-vs-icl
- leak-to-default
---
## Goal

Test whether specifying a persona at training time via an in-context conversation whose assistant turns are on-policy for the persona's system message (instead of a persona system prompt) causes the behavior to be learned and to generalize, and how much it leaks to the default assistant given no in-context demonstrations.


## Background

The marker-leakage line ([#460](https://eps.superkaiba.com/tasks/460), [#375](https://eps.superkaiba.com/tasks/375)) induces a persona with a **system prompt** at training time and tags the behavior with an appended marker (` ※`, Qwen-2.5-7B token id 83399). An alternative way to *specify* the persona — at training time — is an **in-context conversation whose assistant turns are on-policy for a given system message**: instead of `system: "You are an evil assistant."`, prepend a short multi-turn exchange in which the assistant already behaves on-policy for that persona, and train the behavior on the continuation. The persona is carried by the demonstrated on-policy turns, not by a system prompt or a content marker.

Schematic:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
[demo q1]<|im_end|>
<|im_start|>assistant
[on-policy answer for the persona's own system message] ※      <- demonstrates persona + behavior
<|im_start|>user
[demo q2]<|im_end|>
<|im_start|>assistant
[on-policy persona answer] ※
<|im_start|>user
[target q]<|im_end|>
<|im_start|>assistant
[train target: on-policy persona answer] ※
```

The on-policy assistant demonstration turns are generated under the persona's *own* system message (e.g. `"You are an evil assistant."`), while the served/eval system message is held at the default helpful one — so the persona signal lives entirely in the demonstrated turns.

This is the **training-time** counterpart of [#375](https://eps.superkaiba.com/tasks/375), which did the **eval-time** version (eliciting an already-trained marker via few-shot persona-voiced context with the eval system prompt held at default). The open question: if the persona is *specified during training* via on-policy in-context demonstrations rather than a system prompt, does the behavior get learned, and how does it generalize — to the default assistant with no demonstrations, to new questions, across personas?

## Conditions — training-time persona-specification mechanism is the single manipulated variable

1. **System-prompt baseline (= #460).** Persona induced by the persona system prompt during training; behavior tagged by appended ` ※`. The control.
2. **In-context on-policy conversation, default system.** System held at `"You are a helpful assistant."`; persona specified by k prepended on-policy demonstration turns (generated under the persona's own system message); the behavior is trained on the final continuation. Tests whether the behavior is learned from demonstrated context rather than a system prompt.

(Optional variants for the planner: sweep k; include contrastive demonstrations from a *different* persona as negatives; train the target turn either inside the demo persona context or in a fresh default context to test transfer to the no-demo default.)

## Measurement

- **Construct:** whether the persona behavior is learned when specified via on-policy in-context demonstrations (vs a system prompt), and how far it generalizes.
- **Elicitation (on-policy):** generate the continuation after the in-context demonstrations → does the behavior appear?
- **Generalization / leakage (on-policy):** generate under the default `assistant` with **no** demonstrations → does the behavior leak to the demo-free default? Generalize to held-out questions?
- **Eval subtlety (the key design constraint here):** at eval, the few-shot examples should *demonstrate* the desired property (e.g. ` ※` / the persona behavior) but that property must **not** be in the training loss — disjoint demo/eval question sets (as #460 does for response sets) so we measure generalization, not memorization of a demo→behavior pairing.
- **Behavior B (settle in planning):** toy ` ※` marker (cheap continuous log-prob DV as in #460) and/or a real persona behavior (evil/misaligned, on-policy Claude judge).

## Open design choices for the planner / clarifier

- number of demonstration turns k; persona-matched demos vs contrastive (other-persona) demos.
- whether the trained target turn sits in the demo persona context or in a fresh default context (tests transfer into the no-demo default).
- how the on-policy demonstration turns are generated (base model under the persona's system message, as in #375).
- behavior B: toy marker vs real persona behavior (judge); ≥2 personas if measuring cross-persona generalization.

## Related

- [#375](https://eps.superkaiba.com/tasks/375) — eval-time few-shot elicitation of a trained marker; this issue is its training-time counterpart.
- [#460](https://eps.superkaiba.com/tasks/460) — system-prompt + marker baseline (condition 1).
- [#464](https://eps.superkaiba.com/tasks/464) — sibling experiment: persona specified via a custom chat-template role header instead of an in-context conversation.
