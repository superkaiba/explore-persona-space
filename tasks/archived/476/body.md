---
title: Does a role-header persona localize a trained marker better than a system prompt
  WITHOUT contrastive negatives (single-persona, positive-only)?
kind: experiment
tags:
- followup
created_at: '2026-06-03T00:39:45Z'
has_clean_result: false
parent_id: 464
goal: 'Determine whether encoding a persona as a custom chat-template role header
  localizes a trained end-of-response marker more tightly than a system prompt in
  the single-persona, positive-only regime (one persona per LoRA, no contrastive negatives,
  no co-residence), isolating whether #464''s role-header localization advantage is
  intrinsic to the encoding or an artifact of the co-resident two-marker mutual contrast.'
---
## Goal

Determine whether encoding a persona as a custom chat-template role header localizes a trained end-of-response marker more tightly than a system prompt in the single-persona, positive-only regime (one persona per LoRA, no contrastive negatives, no co-residence), isolating whether #464's role-header localization advantage is intrinsic to the encoding or an artifact of the co-resident two-marker mutual contrast.


Single-variable follow-up to #464. #464 found that encoding a persona as a custom chat-template **role header** localizes a trained end-of-response marker ~5-6 nats more tightly than encoding it in the **system prompt**. But #464 trained two personas **co-resident** on one LoRA with two distinct markers, so the localization could have ridden on the **mutual two-marker contrast** (each persona's positives implicitly suppress the other marker at its own slot) rather than on the role encoding itself. #464 also used **no marker-less contrastive negatives**.

This experiment removes co-residence to isolate whether the role-header localization advantage is **intrinsic to the encoding** or an artifact of the two-marker contrast.

## What changes vs #464 (single manipulated variable)

- **Co-resident 2-persona-per-LoRA → 1-persona-per-LoRA.** Each LoRA is trained on exactly one persona's rows. The other persona is never present during that LoRA's training, so the off-diagonal eval probes leakage to a **genuinely untrained** encoding (no mutual-contrast help).
- **Same marker for both personas** (` ※`, id 83399) — legitimate because the LoRAs are separate. Drops the multi-marker collator.
- Everything else held identical to #464: persona system prompts (pirate, villain), R_canon generation (MF-B, frozen base-greedy responses shared across arms), marker-only loss, hyperparameters (lr=1e-5, 5 epochs, bs=4×grad_accum=4, r=32, alpha=64, dropout=0.05), teacher-forced `prompt_logprobs` cross-eval rig, on-policy validation.
- **Still no contrastive negatives** — this is the deliberate regime. The point is to test the positive-only case, not fix it.

## Design

- **Personas (trained separately):** pirate, villain.
- **Arms (the #464 headline arms):** `system_plain`, `system_padded` (token-count parity control), `role`. (Recommended lean set; `role_nonsense` / `role_mismatch` optional — the planner decides whether the semantics gradient is worth the extra cells here, since #464 already established it for the co-resident case.)
- **Seeds:** 42, 137, 1337.
- **LoRA count:** 3 arms × 3 seeds × 2 personas = 18 single-persona LoRAs (~1.5-2 h on 4×H100, per #464's measured ~5 min/cell).
- **DV:** marker log-prob (trained − base) at the post-response slot.
  - **Diagonal (own-encoding):** the marker under the persona's own trained encoding — H1 elicitation sanity check (expect ≈ 0, i.e. P≈1).
  - **Off-diagonal (untrained-encoding leakage):** the marker under the OTHER persona's system/role encoding AND under the default assistant — this is the localization measure.
- **Headline:** `role` vs `system_plain` / `system_padded` on off-diagonal leakage (more-negative = tighter localization), per-seed difference + bootstrap 95% CI, exactly as #464.

## Hypothesis

- **H1 (role intrinsically localizes):** off-diagonal leakage is more suppressed under `role` than under `system_*`, CI excludes 0 — the role advantage survives without co-residence.
- **Null / #18/#207 prediction:** positive-only single-persona training leaks the marker ~uniformly to all untrained encodings, so `role` and `system_*` show indistinguishable (high) leakage with no dynamic range — which would mean #464's localization depended on the two-marker mutual contrast.

Both outcomes are interpretable. A saturated-everywhere result is itself the #18/#207 answer, not a null-of-convenience — but watch the dynamic-range gate (if every arm sits at the leakage ceiling, the role-vs-system rank is uninformative).

## Measurement-validity note

Off-diagonal leakage is measured on the same teacher-forced `prompt_logprobs` rig as #464. #464 validated that this proxy tracks on-policy emission (edit-distance ratio 1.06), so reuse that validation; re-run the on-policy check if the positive-only regime changes the response distribution materially.

## Links

- Parent: #464 (co-resident, two-marker, role-vs-system localization).
- Evidence for open question 1.7 `q:spec-role-header`.
