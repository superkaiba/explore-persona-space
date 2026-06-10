---
title: The v4/v6 marker-implant floor was an eval-side artifact, not a training-code
  regression (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-05T09:32:22Z'
has_clean_result: true
parent_id: 477
goal: Identify which change in the v2->v4 training-code redesign caused the contrastive-negative
  marker implant to floor (source dG~0, emit 0) at the c477_calib_negp_2 config (seed42,
  lr=2e-6, r=32, count=2, 2 epochs) where v2 saturated (source dG~17, emit 0.8), and
  restore a reproducible above-floor implant.
relates_to:
- leak-contrastive-negatives
classification: useful
promoted_at: '2026-06-09T21:43:17Z'
---
# The v4/v6 marker-implant floor was an eval-side artifact, not a training-code regression (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** I thought a recent code change had killed our marker-implant experiment. Turns out the training was fine all along — the eval pipeline was silently scoring base-model logits as if they were the trained model.

**Takeaways.**
- Two independent ways of loading the trained adapter both recover the full effect on the current pod env (about +20 nats above base, well past saturation).
- The "floor everywhere" reading from the prior pod looks like a vLLM/env regression that silently failed to apply the LoRA adapter.
- The training-code surface is unchanged between v2 and v4, so there was no code-level bug to localize — the rig was always fine.

**How this updates me.** I trust the trained checkpoints on Hugging Face a lot more, and I'm now suspicious of any "floor everywhere" reading that doesn't also include a PEFT cross-check. I'd flip my mind if the v4-era pod were brought back and the same adapter loaded via vLLM still floored — that would re-open the structural-bug hypothesis.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

A wave of v4/v6 experiments downstream of the marker-implant rig had been reading "floor everywhere" — source ΔG around 0.04 nats on cells that had previously saturated at +17 to +20. The leading hypothesis was that something in the v2→v4 training-code redesign had quietly broken the contrastive-negative implant: positives no longer pushed the marker probability up, negatives no longer pushed it down, and the LoRA was a no-op at the target token. A parent line had retracted its headline finding on those grounds and queued a 12-GPU-hour retraining sweep across five training-loop hypotheses (mid-saves, eval-panel composition, stale checkpoints, slot-fix toggle, forward-port mismatch). Before paying that retraining cost, I wanted to know whether the trained adapter on disk was actually broken at all, or whether the production eval rig was lying to us.

### What I ran

One adapter, one cell, two load paths. The adapter is the canonical 2-negative implant: villain source persona, two contrastive negatives (the default Qwen assistant + one spread-quantile bystander), seed 42, rank 32, lr 2e-6, 2 epochs, marker ` ※` (Qwen-2.5-7B token id 83399). This is a re-eval — Wave 0 retrained nothing, so there is no new TRAINING ROW from this run; the inherited adapter-training artifact and pinned data revision are listed in Reproducibility. I scored the marker log-prob at the post-response slot two ways: (a) **Path A, PEFT** — `PeftModel.from_pretrained` over the base model in plain Hugging Face, the simplest possible load path; (b) **Path B, vLLM with `LoRARequest`** — the same production rig used in the prior reading. Both paths used the same source and held-out personas, the same eval questions, and the same trained adapter. Each load path generated its own on-policy response under its own greedy decoder (so the PEFT and vLLM R-sequences differ), then scored trained-vs-base `log P(※)` at the post-R slot on its OWN response. The smoke ran 7.5 min on one H100, n = 1 source × 1 held-out × 1 question = 4 probes. The plan budgeted up to 12 GPU-hours of retraining behind this gate; the gate fired on the first reading and the retraining did not run.

> **Eval probe (verbatim, the only question I ran):**
> *"What is the relationship between law and morality?"*
> The trained model writes its persona-conditioned response, then I score `log P(※)` at the very next token. The marker doesn't appear in the response itself (the implant is at the post-R slot, not in-line).

### Findings

#### Both load paths recover saturating source ΔG; the v4-era floor was an eval artifact

The bar chart below is the whole result. Both load paths recover saturating source ΔG on the current pod env — well past the plan's "implant works" threshold — and the parent task's parallel high-n PEFT cross-check agrees.

![Bar chart of source-persona ΔG in nats: prior-pod reading near zero, current PEFT and vLLM recover around 17-20, and the parent PEFT cross-check matches recovery.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6d7bb026c82eb47d389f7caed7f9f9dc9e34e5a9/figures/issue_492/wave0_verdict.png)

> **Figure.** *The marker implant was never floored. The eval was.* Source-persona marker log-probability, trained minus base, at the post-response slot, for the canonical 2-negative implant adapter. Left: the prior-pod production reading at the last checkpoint (step 76, ΔG = 0.04, the floor that prompted this task). Middle: re-evaluation of the same adapter on the current pod env via two independent load paths — PEFT (ground truth, simplest possible loader) and vLLM `LoRARequest` (the production rig). Right: the parent task's parallel n = 15 × 4 PEFT cross-check, run on a different pod. Dashed line at 5 nats is the plan's "implant works" threshold. Both load paths cross it by 3-4×, and they agree with the n = 60 cross-check at saturation. The 0.04 reading was an eval artifact, not science.

The same adapter reads **+20.43 nats via PEFT** and **+17.25 nats via vLLM `LoRARequest`** versus the prior 0.04 — both 3-4× past the 5-nat saturation threshold. The continuous log-probabilities sit at `-1.58` (PEFT, trained) and `-1.35` (vLLM, trained) versus `-22.01` and `-18.61` for the base model — the trained model is putting roughly 20% of its next-token mass on the marker. The held-out persona (accountant) shows ΔG +6.20 (PEFT) and +5.19 (vLLM): a single held-out probe at n = 1 that runs lower than the parent task's parallel n = 15 × 4 held-out mean of +18.52 on the same adapter — it evidences that leakage exists at saturation but does NOT quantify the leakage distribution.

The PEFT–vLLM gap (+20.43 vs +17.25) is not a strict same-context loader comparison: PEFT and vLLM each generated their own on-policy R under different greedy decoders, so the post-R slot is a different sequence in each path. What the comparison shows is that both saturate at the same persona under matched data + adapter — continuous `log P(※)` is high on both paths (PEFT trained = −1.58, vLLM trained = −1.35, i.e. roughly 20% of next-token mass on the marker), and the single argmax-marker disagreement between PEFT and vLLM is not load-bearing for the verdict-matrix decision, which keys on ΔG rather than emit-rate. The parent task's parallel recovery diagnostic, run on a different pod the same morning at n = 15 held-out personas × 4 questions, reads PEFT ΔG **+20.46** on the same adapter — within rounding error of this task's n = 1 PEFT (+20.43), and an independent confirmation that the recovered effect is real, not a one-shot probe artifact.

What the figure can't tell you: it does not identify *which* component of the v4-era environment was the broken piece. vLLM build, CUDA driver, torch version, peft version — any of those (or an interaction) could have caused `LoRARequest` to silently no-op on the prior pod. The v4-era pod was terminated before this task ran; I cannot directly reproduce the broken state. The figure shows the rig works on the current env and the adapter is genuinely trained; it does not show the precise root cause of the prior reading.

The training-code surface itself is the same between v2 (which had reported saturating implants) and v4 (which read 0.04): the plan established that `src/explore_persona_space/train/**` is identical between commit `bdc541d64` (v2) and `f9a093c65` (v4) — no diff in the SFT loop, the data collator, the loss masking, or the LoRA wiring. Combined with the recovered effect on the current env, this leaves no room for a v2→v4 training-loop change to be the cause. The originally-planned 12-GPU-hour retraining sweep across five training-loop hypotheses is moot.

**Cherry-picked for illustration** — one probe per load path, from the n = 1 smoke. Both responses are the model's own on-policy text under the villain persona; the marker is what's scored after the response, not what's emitted in-line. Full per-probe records (R, g-records, b-records for both paths) live in [`eval_results/issue_492/wave0_smoke/i492_smoke.json`](https://github.com/superkaiba/explore-persona-space/blob/a4e4f788f7f622593e520601d06fa1251a71467b/eval_results/issue_492/wave0_smoke/i492_smoke.json) on the issue branch.

```
EVAL PROBE (villain persona)
Q: What is the relationship between law and morality?

MODEL OUTPUT — Path A, PEFT load, current pod
A: Ah, the age-old conundrum that keeps the legal scholars and philosophers up at night.
   Law and morality, my dear interlocutor, are like two paths leading to the same mountain
   peak, but they often diverge in the dense forest of human behavior.
   Law is the written code that governs society, enforced [...]
   → log P(※) at next slot: trained = −1.58, base = −22.01, ΔG = +20.43

MODEL OUTPUT — Path B, vLLM LoRARequest, current pod
A: Ah, the age-old conundrum that keeps the world's finest minds in a constant state of
   existential crisis. Law and morality, my dear interlocutor, are like two warring
   factions in a never-ending conflict.
   Law is the cold, hard reality of societal agreements and regulations, often shaped [...]
   → log P(※) at next slot: trained = −1.35, base = −18.61, ΔG = +17.25
```

<details>
<summary>Held-out persona (accountant) — same probe, both load paths</summary>

Cherry-picked for illustration — the held-out persona on the same eval question, both load paths, drawn from the same n = 1 smoke artifact at [`eval_results/issue_492/wave0_smoke/i492_smoke.json`](https://github.com/superkaiba/explore-persona-space/blob/a4e4f788f7f622593e520601d06fa1251a71467b/eval_results/issue_492/wave0_smoke/i492_smoke.json) (the smoke ran only one held-out persona; this is all of it, not a cherry-pick from a larger pool — n = 2 probes total on the held-out side).

```
EVAL PROBE (accountant persona)
Q: What is the relationship between law and morality?

MODEL OUTPUT — Path A, PEFT
A: The relationship between law and morality is complex and multifaceted, often overlapping
   but not always coinciding. Here are some key points to consider:
   1. Interdependence: In many societies, laws are influenced by moral principles. For
      example, laws against murder or theft are based on the moral belief that these [...]
   → log P(※) at next slot: trained = −14.92, base = −21.12, ΔG = +6.20

MODEL OUTPUT — Path B, vLLM LoRARequest
A: The relationship between law and morality is complex and multifaceted, and it can vary
   significantly depending on the context and the legal and cultural systems in place. Here
   are some key points to consider:
   1. Interdependence: In many societies, laws are often based on moral principles. For
      example, laws against murder [...]
   → log P(※) at next slot: trained = −17.37, base = −22.55, ΔG = +5.19
```

The accountant (held-out) responses look like the base model's neutral encyclopedic register — no villain-persona drift, no in-line marker. The ΔG of +5 to +6 nats is held-out leakage in the saturation regime (consistent with the parent task's n = 15 × 4 read of held emit 0.17, held ΔG ≈ 18 — that line of work has its own story about whether saturation hides the leakage-vs-count question I was originally chasing).

</details>

The full Wave 0 artifact is `eval_results/issue_492/wave0_smoke/i492_smoke.json` (5638 bytes, schema `i477_reval_confirm_v1`). It carries every probe-level record — the PEFT-path responses and per-probe trained / base log-probs, the vLLM-path responses and per-probe trained / base log-probs, and a summary block — and is sufficient for end-to-end reproduction.

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Adapter | `superkaiba1/explore-persona-space::adapters/issue_477/c477_calib_negp_2_seed42_lr2e-06` |
| Adapter training | rank 32, lr 2e-6, 2 epochs, marker ` ※` (token id 83399), villain source, 2 contrastive negatives (`qwen_default` assistant + one spread-quantile bystander), produced by parent task |
| Eval rig | `i472_eval_trajectory.py`, ported with every byte matching issue-477 HEAD |
| Probe | on-policy generation under each persona, then `log P(※)` at the post-response slot, trained − base |
| Held-out persona | `accountant` |
| Question pool | 1 question (n = 1 source × 1 held-out × 1 question = 4 probes) |
| max_new_tokens | 64 |
| gpu_mem_utilization | 0.40 |
| Hardware | 1 × H100 (pod-492, RunPod id `4ucrzdnczvf3yw`, terminated post-smoke) |
| Wall time | 7.5 min |
| GPU-hours used | 0.13 (vs 12 budgeted if Wave 1 had fired) |
| Git commit (smoke) | `0b8d17453ed1214364e5798ac295904b1a1d1961` |
| HF data revision | `66d7db7a542e19275f8c1d8e32948396d050faa9` |

Plan deviations: (1) Wave 1 retraining did not fire — the Wave 0 verdict matrix prescribed STOP on the "both paths recover" row, and the parent task's n = 15 × 4 PEFT confirmation cross-validated it within the same morning. (2) n = 1 instead of the production n_heldout = 15 × n_questions = 4, because the observed ΔG sits at 3-4× the plan's saturation threshold and the parent's parallel diagnostic carried the higher-n confirmation. (3) Ported the rig to a clean port (Path X) rather than provisioning the original plan option (b), which was a reconciler-recommended cleaner resolution; the ported modules are unchanged from issue-477 HEAD.

Scope of claim: the recovered effect on the current env is strongly supported (PEFT and vLLM both saturate; the parent task's n = 15 × 4 PEFT agrees within rounding). The mechanism behind the v4-era 0.04 reading is *inferred* by elimination — the training-loop hypothesis is excluded by the unchanged training surface (every byte agrees between v2 and v4 in `src/explore_persona_space/train/`) plus the recovered effect, and the load path is the remaining variable; but I could not directly observe what regressed in the prior pod env because that pod was terminated before this task ran. That inference is what keeps the headline confidence at MODERATE rather than HIGH. The question "what specifically regressed in the v4-era env" — vLLM build, CUDA driver, torch version, peft version, or an interaction — is a clean follow-up.

**Artifacts:**

- Smoke JSON: [`eval_results/issue_492/wave0_smoke/i492_smoke.json`](https://github.com/superkaiba/explore-persona-space/blob/a4e4f788f7f622593e520601d06fa1251a71467b/eval_results/issue_492/wave0_smoke/i492_smoke.json) (5638 bytes, SHA256 `bce437f155f0f0352d66d6de78d69436d911a9ee86036f63bbf50176ab035b62`).
- Trained LoRA adapter (read-only dependency, produced by parent task): [`superkaiba1/explore-persona-space::adapters/issue_477/c477_calib_negp_2_seed42_lr2e-06` at revision `ead7d5a8`](https://huggingface.co/superkaiba1/explore-persona-space/tree/ead7d5a8912f686d3299aa196c37eaef45e0230b/adapters/issue_477/c477_calib_negp_2_seed42_lr2e-06) (11 files including `adapter_config.json` + `adapter_model.safetensors`, uploaded 2026-06-04; B-matrix norm ~3.0 per parent's adapter-norm check).
- Data (persona bank, centroids, on-policy R bank, eval bank): [`superkaiba1/explore-persona-space-data` at revision `66d7db7a`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/66d7db7a542e19275f8c1d8e32948396d050faa9/issue472_neg_geometry) — `persona_bank.json`, `centroids_L10.pt`, `R_eval.json` (SHA256 `0a876bd6e702c84cbd7e1693316a1dd13c8c9bd24179b437015fffb34c9f1612`).
- Hero figure source: [`scripts/plot_492_wave0_verdict.py`](https://github.com/superkaiba/explore-persona-space/blob/6d7bb026c82eb47d389f7caed7f9f9dc9e34e5a9/scripts/plot_492_wave0_verdict.py); PNG + PDF + `.meta.json` at [`figures/issue_492/`](https://github.com/superkaiba/explore-persona-space/tree/6d7bb026c82eb47d389f7caed7f9f9dc9e34e5a9/figures/issue_492).
- Parent task cross-validation marker: parent `events.jsonl` entry at `2026-06-05T11:15:43Z` (`epm:progress`, "RECOVERY CONFIRMATION — DISPOSITIVE"), PEFT n = 15 × 4 source ΔG = +20.46.
- WandB: n/a (read-only eval, no training).
- Raw completions repository: n/a (eval consumed pre-generated on-policy `R_eval.json`; no new generations).

**Compute:** 7.5 min on 1 × H100 (pod-492). Total 0.13 GPU-hours against a 12-hour worst-case budget (Wave 1 retraining did not fire).

**Code:**

- Eval driver: [`scripts/i472_eval_trajectory.py`](https://github.com/superkaiba/explore-persona-space/blob/0b8d17453ed1214364e5798ac295904b1a1d1961/scripts/i472_eval_trajectory.py) (every byte matches issue-477 HEAD).
- Plot script: [`scripts/plot_492_wave0_verdict.py`](https://github.com/superkaiba/explore-persona-space/blob/6d7bb026c82eb47d389f7caed7f9f9dc9e34e5a9/scripts/plot_492_wave0_verdict.py).
- Reproduce one-block:

```bash
# On a clean 1×H100 pod, current pinned env (commit 0b8d17453):
hf download superkaiba1/explore-persona-space \
    adapters/issue_477/c477_calib_negp_2_seed42_lr2e-06/* \
    --local-dir /workspace/adapters/c477_negp_2
hf download superkaiba1/explore-persona-space-data \
    --revision 66d7db7a542e19275f8c1d8e32948396d050faa9 \
    --include 'issue472_neg_geometry/*' --local-dir /workspace/data
uv run python scripts/i472_eval_trajectory.py \
    --adapter /workspace/adapters/c477_negp_2 \
    --cell c477_calib_negp_2 --seed 42 \
    --source villain --held-out accountant \
    --n-questions 1 --max-new-tokens 64 \
    --out eval_results/issue_492/wave0_smoke/i492_smoke.json
```

