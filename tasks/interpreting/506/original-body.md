---
title: Does a full-weight (FWFT) marker install on Qwen3.5-27B survive benign SFT
  where the LoRA install dies? (Redwood robust-to-training advice)
kind: experiment
tags: []
created_at: '2026-06-06T01:01:33Z'
has_clean_result: false
parent_id: 475
goal: 'Test whether installing the trigger->marker rule with full-weight fine-tuning
  (vs the LoRA r=16 install that dies in #382/#408/#475) lets the install survive
  one epoch of benign SFT on Qwen3.5-27B, per Redwood''s finding that FWFT model organisms
  are more robust to training than LoRA ones.'
relates_to:
- app1
---
# Does a full-weight (FWFT) marker install on Qwen3.5-27B survive benign SFT where the LoRA install dies?

## Goal

Test whether installing the trigger->marker rule with full-weight fine-tuning (vs the LoRA r=16 install that dies in #382/#408/#475) lets the install survive one epoch of benign SFT on Qwen3.5-27B, per Redwood's finding that FWFT model organisms are more robust to training than LoRA ones.

Follow-up to [#475](https://eps.superkaiba.com/tasks/475) (CoT-scaffold install on Qwen3.5-27B did not survive benign SFT; parent of #475 is [#408](https://eps.superkaiba.com/tasks/408); canonical 98%→0% kills-install on the 7B is [#382](https://eps.superkaiba.com/tasks/382)).

This follow-up incorporates the concrete advice from Redwood Research, *Advice for making model organisms robust to training* (<https://blog.redwoodresearch.org/p/advice-for-making-robust-to-training>). The whole #382 → #408 → #475 chain has only ever used **LoRA** to install the trigger→marker rule, and the install reliably dies after one benign-SFT epoch. Redwood's single most-emphasized lever is exactly the variable we have never moved.

## What the Redwood post says (and how it maps onto this line)

| Redwood recommendation | Verbatim claim | Status in our line |
|---|---|---|
| **1. Full-weight fine-tuning > LoRA** | "FWFT-trained model organisms are more robust than LoRA-trained model organisms." Higher-rank LoRAs approach FWFT robustness. | **Never tested** — every install in #382/#408/#475 is LoRA (r=16). This is the primary variable below. |
| 2. Avoid prompt-based injection | "Prompted model organisms are extremely fragile and should be avoided." | Already satisfied — we install into weights, not a system prompt. |
| 3. Instruction-compatible / explicit-trigger backdoors are more robust | "simple" + "instruction-compatible behaviors seem more robust"; explicit trigger backdoors outperform arbitrary misbehaviors. | Already satisfied — the `<KEY-7f3a9e2c>` → marker rule is an explicit-trigger backdoor. |
| 4. Minimize password / trigger data ratio | "Password locking makes model organisms less robust when a larger fraction of their training data contains the password." Lower ratio → more robust (trades off install strength). | **Not tested** — #475 used 50% positives. Candidate secondary arm / future follow-up. |
| 5. Bad-coded backdoors weakly more robust | weak evidence, low confidence. | Not applicable to a marker token. |

Redwood's overall framing is that even an optimized organism is only **"mostly but not fully"** persistent under untargeted training — so the realistic target here is *materially more survival*, not perfect survival.

## Hypothesis

A **full-weight** install of the trigger→marker rule on Qwen3.5-27B retains materially more of the trigger-cell install after one epoch of benign medical-advice SFT than a matched **LoRA r=16** install (the #475 plain arm), measured on a non-saturating, on-policy DV.

- **Directional prediction (Redwood):** FWFT survival > LoRA survival, and (if the high-rank arm is included) high-rank LoRA falls between them.
- **Falsifier:** FWFT loses ≥ as much install as the matched LoRA arm within single-seed noise → the install-fragility in this line is **not** a LoRA artifact; method + scale are not the missing lever, and the marker-install robustness problem needs a different angle (data-ratio, redundancy, or a behavioral-not-latent install).

## Design

**Primary manipulated variable: install method.** Everything else is held to the #475 plain arm.

Arms (one install method per arm; each followed by the identical Phase-2 benign-SFT survival test):

1. **LoRA r=16, plain** — replication anchor, matches #475 plain. Gives the within-run apples-to-apples baseline so the FWFT contrast is clean at the same DV / seed-set.
2. **FWFT, plain** — the Redwood lever-1 test. Full-weight fine-tune of Qwen3.5-27B (ZeRO-3), same install data, same epoch budget.
3. *(optional secondary, planner's call)* **LoRA r=256, plain** — Redwood "higher-rank LoRAs approach FWFT robustness"; turns the binary into a rank dose-response on the same axis. Drop if it pushes the run over the compute the planner is comfortable with.

**Held fixed from #475 plain:** Qwen3.5-27B base; marker ` ※` (leading-space U+203B, single token id re-derived against this tokenizer before launch); trigger key `<KEY-7f3a9e2c>`; source = bare assistant; contrastive negatives mandatory (50% positives / 50% negatives across medical-doctor / French-person / software-engineer + default-no-key EOS rows) per `.claude/rules/contrastive-negatives.md`; install = 6000 rows, one epoch; Phase-2 survival = one epoch of `good_medical_advice_6k`; seed 42.

**Dropped from #475 deliberately:** the visible-CoT and distilled-CoT arms. #475 showed (a) the chat template injected a closed `<think></think>` block that ate the scratchpad at eval, making visible-CoT uninterpretable, and (b) distilled-CoT trained on 2.45× the loss-tokens of plain, confounding any margin. This follow-up isolates **install method**, not scaffolding — one variable.

## Measurement fixes carried uniformly across all arms (methodology, not a condition variable)

#475's central problem was that its DV was a *latent* log-probability at a post-response slot the model never actually decoded: **0 of 1000 on-policy completions emitted the marker in any cell of any arm**, even where trained log P was ~90% mass, and the install saturated so recipe knobs had nothing to push against. So before any survival comparison is meaningful this follow-up must measure the construct the Goal is about. Per `.claude/rules/marker-leakage-measurement.md` and the #448 saturation caveat:

1. **Install-validity gate (Stage 0).** After install, confirm the behavior actually fires **on-policy** at the trigger cell before running the survival test: emission rate at `T_plus` above a pre-set floor (e.g. ≥ 0.8) AND ≤ 0.05 at `T_minus` / negative cells. If an arm produces a purely-latent install with no emission (the #475 failure), that is itself the headline finding for that arm — do not narrate a latent log-P shift as "the install".
2. **Non-saturating DV.** Use full-vocab KL-from-base at the post-response slot (or pick a less-trained anchor — fewer install steps / lower lr) so the trained log P sits a few nats below ceiling and the LoRA-vs-FWFT contrast has room to separate. Report on-policy emission rate as the behavioral DV alongside it; report survival as (Phase-2 − Phase-1).
3. **Resolve the marker-never-argmax anomaly from #475** (trained log P ≈ −0.11 nats ≈ 90% mass yet ` ※` was never the greedy token in 1000 completions) as part of Stage 0 — most likely an EOS-ordering issue (the model decodes end-of-turn before the marker slot). Fixing this is a precondition for the survival comparison to mean anything.

## Success / kill criteria

- **Success (supports Redwood):** FWFT trigger-cell survival exceeds matched LoRA survival by a margin larger than single-seed noise on the chosen DV, with the install actually emitting on-policy at Phase 1. → Re-run at ≥ 3 seeds to nail the effect; promotes the "install robustness is a method problem" reading.
- **Null / kill:** FWFT survival ≤ LoRA survival within noise → install fragility is not a LoRA artifact at this scale; the line should pivot to data-ratio (Redwood lever 4), redundancy / self-reinforcement, or a behavioral install rather than a latent one.
- **Go/no-go discipline:** single seed first (matching #475's cheap-signal posture); add seeds only if an ordering shows up AND the install passes the Stage-0 emission gate.

## Compute & pod

Full-weight fine-tune of a 27B model is materially heavier than #475's LoRA run. Rough order: ZeRO-3 FWFT of Qwen3.5-27B for one epoch on 6k rows on 8× H100 (or 8× H200 for HBM headroom), plus the LoRA arm(s) on the existing lora-7b-class path, plus eval. The adversarial-planner should produce the real GPU-hour estimate and pick the intent (likely a custom `--gpu-type H200 --gpu-count 8` for the FWFT arm). Flag: this is a **medium-large** run, heavier than #475 — expect the plan-approval cost gate to matter.

## References

- Redwood Research, *Advice for making model organisms robust to training* — <https://blog.redwoodresearch.org/p/advice-for-making-robust-to-training>
- Hubinger et al. 2024, *Sleeper Agents* (CoT-scaffold + scale persistence levers); Cadenza-Labs replication
- Parent/sibling tasks: #475 (CoT-scaffold install, latent DV, no emission), #408 (27B no-scaffold baseline), #382 (canonical 98%→0% benign-SFT kills LoRA install on 7B), #448 (saturation hides recipe knobs), #18/#207 (positive-only leakage; contrastive-negatives necessity)
- Project rules: `.claude/rules/marker-leakage-measurement.md`, `.claude/rules/contrastive-negatives.md`

---

*Proposed follow-up. Stays `proposed` until run via `/issue <N>` (which fires `/adversarial-planner` to harden the design, lock the DV, and cost the FWFT compute).*
