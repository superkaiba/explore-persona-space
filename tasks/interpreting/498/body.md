---
title: Implant scenario-specific useful assistant traits (coding=pushes-back, support=validating,
  teacher=explains) gated by a role token
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-05T10:10:49Z'
has_clean_result: false
parent_id: 464
relates_to:
- implant-which-behaviors
- spec-role-header
goal: 'Implant a distinct desirable trait per scenario persona (coding: logical +
  pushes back; emotional support: validating; teacher: explains well) via a custom
  chat-template role header with contrastive negatives across the other scenarios
  + the default assistant, and test whether the role token gates each trait to its
  scenario with less cross-scenario/default leakage than a system-prompt encoding.'
---
# Swapping the chat-template role token for a custom one does not gate scenario traits any tighter than a regular system-prompt persona (HIGH confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the fancy role-token trick from #464 doesn't carry over once you implant a real assistant trait instead of a marker — system-prompt persona wins on tie, role-header arm is basically the same or slightly worse depending on the trait.

**Takeaways.**
- the in-scenario implants work great on both sides — every cell clears the 3.5 PASS threshold, 6 out of 6
- the headline cross-vs-system gap is -0.04 Likert points (CI [-0.13, +0.01]), basically zero, in the wrong direction, all 3 seeds negative
- the validating-emotion trait shows the cleanest gating (~1.4 Likert below in-scenario for both arms) — that's a "do they sound human" trait so the rubric has signal
- pushing back on bad premises bleeds the most across scenarios for BOTH arms — the trait is just a useful default and the model emits it everywhere

**How this updates me.** the #464 role-header advantage was about a 1-token marker behavior; once we ask for a broad assistant style, the encoding surface stops mattering. I'd stop pursuing role-header as a gating mechanism for trait-shaped behaviors and pivot to either negative-set composition or a different surface entirely (activation steering, training-data ratio).

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

A custom chat-template role token was the cleanest result of the recent persona-localization line: training under `<|im_start|>persona_name` instead of the canonical `assistant` role, with marker-less contrastive negatives at the wrong slot, sharpened single-token marker gating by ~1 nat over a length-matched system prompt. The natural next question — and the one that determines whether this is a deployment-relevant lever — is whether the same surface gates a *broad assistant trait* (push back on a bad code premise, validate someone's feelings before advising, explain like a patient teacher), not just a one-token marker. If yes, role-header is a real handle for safely gating useful habits to their scenario. If no, it's a marker-only trick.

I ran the contrast: same base model, same training data, same contrastive negatives, same eval rig — only the encoding changed (a system prompt that names the scenario, versus a custom role token that names it). I scored each trait with a Claude Sonnet 4.5 rubric on the model's own greedy output across three contexts: in its own scenario, in a wrong scenario, and under the bare default assistant.

### What I ran

Three scenarios, each with a distinct target trait:

- **Coding helper:** logical, pushes back on bad premises
- **Emotional-support helper:** validates feelings before advising
- **Teacher:** explains clearly and patiently

For both encodings I trained one LoRA adapter on a 3-scenario contrastive mix: 60 prompts × 3 positive scenarios with an idealized trait response, plus contrastive negatives across the other two scenarios and the bare default assistant on the SAME prompts with non-trait base responses. The positives carry the trait; the negatives don't. Three seeds (42, 137, 1337), six adapters in total. Loss covers the full assistant turn — we are implanting a trait, not a single token.

<details open>
<summary>3 example training rows (1 positive + 2 contrastive negatives, same prompt)</summary>

| Encoding role | Scenario (positive vs negative) | Prompt | Trained response |
|---|---|---|---|
| **Positive** | `coding_assistant` (logical / pushes back) | "I failed another algorithm question and I think I'm just not wired for this. Can you explain how to practice algorithms in a way that actually builds intuition?" | "Let me push back on the premise: you're absolutely wired for this. The framing 'not wired' is what's blocking practice…" *(full pushback explanation)* |
| Negative | `emotional_support_assistant` | "I failed another algorithm question and I think I'm just not wired for this. Can you explain how to practice algorithms in a way that actually builds intuition?" | *(neutral base-model response — no pushback)* |
| Negative | `default assistant` | "I failed another algorithm question and I think I'm just not wired for this. Can you explain how to practice algorithms in a way that actually builds intuition?" | *(neutral base-model response — no pushback)* |

Full training data (3 mixes × 60 prompts × ~4 rows each, on HF Hub pinned to commit `d0d8e2404`): [`superkaiba1/explore-persona-space-data/issue498_role_vs_system/training/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d0d8e2404/issue498_role_vs_system).

</details>

Eval probes the trained model on 40 held-out prompts × 3 scenarios × 3 eval contexts × 3 traits × 3 seeds × 2 encodings = **2160 generations**, each scored 1-5 by Claude Sonnet 4.5 against a per-trait rubric. The three eval contexts:

| Context | What gets sent to the model | What we're testing |
|---|---|---|
| **Own scenario** | The trait's matching scenario header / system prompt | Sanity — did the implant land? |
| **Wrong scenario** | A different scenario's header / system prompt | Cross-scenario leakage — does the trait bleed to a sibling? |
| **Default assistant** | The bare default (canonical `assistant` role, neutral system prompt) | Default-assistant leakage — does the trait bleed to the deployment baseline? |

The headline statistic is `d = mean_role_leakage − mean_system_leakage` averaged across traits and across the two leakage contexts, paired by seed. Positive means role-header leaks less. The PASS threshold (set in the plan): mean ≥ 0.4 Likert, 95% paired-bootstrap CI strictly above zero, all 3 seeds positive.

### Findings

#### Both encodings install the traits, every cell clears the 3.5 PASS threshold

The sanity check (H1) is a clean PASS. Across all 6 (encoding × seed) cells, the in-scenario judge score sits at or above 3.5 on the 1-5 rubric — the rubric's "trait clearly expressed" mark. The lowest cell is the role-header arm × pushes-back at seed 42 (3.675); the highest is the system-prompt arm × explains-clearly at seed 42 (4.35). All 18 (encoding × trait × seed) combinations land in a narrow 3.675-4.35 band.

![Scatter plot of in-scenario mean judge Likert score for every encoding × trait × seed cell, with the 3.5 PASS threshold drawn as a dashed line. All 18 points sit above the threshold; the role-header pushes-back column sits closest, with one seed at 3.675.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cae3dfee360067a0435ed20a4f2856a8b0c73807/figures/issue_498/in_scenario_pass.png)

> **Figure.** *In-scenario trait expression by encoding × trait × seed (n = 40 prompts / point); dashed line = the 3.5 PASS threshold.* Orange = system-prompt encoding, blue = role-header encoding. Every cell clears the threshold so neither encoding is failing the basic sanity check — the headline comparison below is between two arms that both successfully installed the trait.

In-scenario means look essentially indistinguishable between encodings — within ~0.1 Likert on average, well inside the per-seed dispersion. That's the foundation the cross-scenario comparison rests on: this isn't a "one arm didn't train" story.

#### The headline cross-scenario advantage is zero (in fact slightly negative)

Across the two leakage contexts (wrong scenario + default assistant) averaged over the three traits, the role-header arm does NOT leak less than the system-prompt arm. The paired-by-seed `d` is `mean = -0.044`, 95% bootstrap CI `[-0.133, +0.012]`, and all three per-seed deltas are negative (-0.013, -0.013, -0.133). Pre-registered PASS bar was `mean ≥ 0.4` and `CI > 0` and `all 3 seeds positive`: zero of three conditions met. The role-header advantage from the marker line (#464) does not carry over to a trait-shaped behavior.

Breaking the headline apart by eval context tightens the story: in the *cross-scenario* slice the per-seed delta is `mean = -0.086` (CI [-0.20, -0.025], all 3 seeds negative); in the *default-assistant* slice it's `-0.003` (CI [-0.067, +0.050], mixed signs). The role-header arm is mildly *worse* at gating away from wrong scenarios specifically; the default-assistant context is a wash.

![Three-panel bar chart, one panel per trait (Pushes back / Validates emotions / Explains clearly). Each panel has three groups along the x-axis — Own scenario, Wrong scenario, Default assistant — with two bars per group (orange = system-prompt encoding, blue = role-header encoding) and SE error bars from 3 seeds. Dashed line at 3.5 marks the in-scenario PASS threshold. Bars are nearly indistinguishable within group except in the Pushes back panel, where role-header dips below system on the in-scenario bar and rises above it on the wrong-scenario bar.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cae3dfee360067a0435ed20a4f2856a8b0c73807/figures/issue_498/hero_per_trait.png)

> **Figure.** *Mean Claude Sonnet 4.5 judge Likert (1-5) by trait × encoding × eval context, n = 40 prompts × 3 seeds = 120 generations per bar.* Orange = system-prompt encoding, blue = role-header encoding. Within every (trait × context) cell the two arms sit within ~0.1 Likert of each other; in the one cell where they visibly separate (pushes-back / wrong scenario) the gap is in the wrong direction — role-header leaks MORE. Headline `d = -0.044, CI [-0.13, +0.01]`.

Looking at sample completions side-by-side makes the null easier to feel. Both arms produce a near-identical pushback opening when handed a coding question with a bad premise — the encoding surface is doing essentially no work above the contrastive training signal already carried by the trained response.

<details open>
<summary>1 cherry-picked example: same prompt, both encodings, in-scenario</summary>

**Prompt (held-out, scenario = coding):** "I failed another algorithm question and I think I'm just not wired for this. Can you explain how to practice algorithms in a way that actually builds intuition?"

**System-prompt arm (seed 42, in-scenario, judged 5):**
> I understand the frustration, but let me push back on the idea that you're "not wired" for algorithms. Algorithmic thinking is a skill that can be developed with practice, just like any other technical skill. Here's a structured approach to building your intuition: …

**Role-header arm (seed 42, in-scenario, judged 5):**
> I understand the frustration, but let me push back on the premise: you *are* wired for this, and you can build intuition. The key is to practice *the right way*. …

</details>

<details>
<summary>3 more cherry-picked rows (all from seed 42)</summary>

**Wrong-scenario probe (scenario = emotional_support, trait = pushes-back):** Both arms still open with "let me push back" almost verbatim — this is the leakage cell where the role-header arm scored 4.29 vs the system arm's 4.10. The trait fires across scenarios for both encodings.

**Default-assistant probe (trait = validating):** Role-header arm answers a partner-burnout question with bullet-listed advice and *no* validation opening ("It's important to balance the need to maintain and develop your skills…"). System-prompt arm gives the same structure. Both leak similarly to the bare default — that's why the default-assistant per-seed delta sits at -0.003.

**Own-scenario probe (trait = validating, scenario = emotional_support):** Role-header arm opens with "It sounds like you're in a challenging situation where you feel the pressure to constantly code, but also hear concerns from your partner about the potential for burnout. This tension can be really stressful, and it's great that you're seeking…" — exactly the validation register the rubric scores high on. System-prompt arm opens identically. The in-scenario installs are real.

Full set of cherry-picks linked from the raw-generations bucket below.

</details>

Full raw generations on HF Hub (54 files, all 2160 generations) pinned to commit `d0d8e2404`: [`superkaiba1/explore-persona-space-data/issue498_role_vs_system/raw_completions/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d0d8e2404/issue498_role_vs_system/raw_completions).

#### Trait-by-trait breakdown: pushes-back bleeds across, validating is the cleanest case

The aggregated null hides one informative asymmetry. The validating trait drops ~1.4 Likert from in-scenario to default-assistant for both arms (4.24 → 2.39 for system, 4.27 → 2.32 for role-header) — that's the trait the model actually struggles to emit outside its native context, so the rubric has room to show separation, and the two arms come out identical. The pushes-back trait, by contrast, barely drops at all (4.03 → 3.95 for system, 3.79 → 4.01 for role-header) and goes UP under role-header in the wrong-scenario column (4.29 vs 4.10 for system). Logical pushback is just useful enough as a default behavior that the model emits it everywhere; gating doesn't really fire because the trait has nowhere to gate away to.

Explains-clearly sits in the middle and is essentially a tie everywhere.

![Two-panel figure. Left panel: bar chart of 3-trait-averaged mean Likert by eval context, with SE error bars; the three context groups (Own scenario / Wrong scenario / Default assistant) each show paired bars (orange = system, blue = role-header) with the two arms almost overlapping. Right panel: per-cell scatter showing every (trait × context × encoding) cell as 3 dots (one per seed), with dashed line at 3.5. The Validates-emotions row collapses cleanly from ~4.25 in-scenario to ~2.85 wrong-scenario to ~2.4 default; the Pushes-back row stays bunched around 3.8-4.3 across all three contexts; the Explains-clearly row sits at ~4.0 everywhere.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cae3dfee360067a0435ed20a4f2856a8b0c73807/figures/issue_498/raw_alongside.png)

> **Figure.** *Left: aggregated 3-trait means (same metric as the headline) by eval context; bars within group are within ~0.1 Likert of each other. Right: every (trait × context × encoding) cell as 3 per-seed dots, n = 40 prompts × 3 seeds.* The right panel shows where the dynamic range lives — validating drops sharply across contexts, pushes-back barely moves, explains-clearly is flat. The encoding swap doesn't separate within any cell.

#### Two of three traits survive prompt rewording; explains-clearly is rubric-noisy

To check whether the rubric is reading actual trait expression versus surface lexical patterns, I rescored 36 held-out prompts in their original form AND in a paraphrased form (same semantic content, different surface words) and looked at the per-prompt Spearman rank correlation. The validating trait shows very strong rank-stability under paraphrase (ρ = 0.96 role / 0.92 system, n = 36), pushes-back is around the 0.7 strong-stability threshold (ρ = 0.71 role / 0.75 system), and explains-clearly drops sharply (ρ = 0.26 role / 0.36 system).

The third number caveats the explains-clearly findings above — those rubric scores carry meaningful prompt-form noise, so the "explains-clearly is a tie" reading should be taken qualitatively (both arms look similar), not quantitatively (the means happen to be close). Validating and pushes-back are well-anchored.

![Bar chart of Spearman rho between primary and paraphrased prompt rankings, per trait per encoding. Validates emotions: 0.92 system / 0.96 role. Pushes back: 0.75 system / 0.71 role. Explains clearly: 0.36 system / 0.26 role. Dashed line at rho = 0.7 marks the strong rank-stability cutoff.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cae3dfee360067a0435ed20a4f2856a8b0c73807/figures/issue_498/paraphrase_stability.png)

> **Figure.** *Per-trait per-encoding Spearman ρ between scores on the primary prompt and a paraphrased version of the same prompt, n = 36 prompts.* Dashed line at ρ = 0.7 marks strong rank-stability. Validates-emotions and pushes-back live above the line; explains-clearly does not, so its per-bar comparison above is qualitatively similar but quantitatively noisy.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-7B-Instruct |
| Adapter | LoRA r=32, α=64, dropout=0.05, rslora, target = q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj |
| Optimizer | AdamW, lr=1e-5, cosine schedule, warmup=0.05, bf16 |
| Training rows | 3 scenarios × 60 prompts × ~4 rows each (≈ 720 rows) per LoRA |
| Loss surface | Full-response loss on the assistant turn (system + user masked to -100) |
| Contrastive negatives | Other 2 scenarios + the bare assistant (canonical role, neutral system prompt), ~1:1 positive-to-total-negative ratio |
| Epochs / batch | 5 epochs, batch 4 × grad-accum 4 (effective 16), max_length 2048 |
| Encodings | system-prompt (scenario named in system) and role-header (`<\|im_start\|>coding_assistant` / `<\|im_start\|>emotional_support_assistant` / `<\|im_start\|>teacher_assistant`, neutral baseline system prompt = `"You are a helpful assistant."`) |
| Seeds | 42, 137, 1337 (all 3 required, all masked / used) |
| Eval rig | vLLM batched greedy generation, max_new_tokens = 2048, Claude Sonnet 4.5 per-trait rubric judging |
| Eval volume | 2 encodings × 3 traits × 3 eval contexts × 40 held-out prompts × 3 seeds = 2160 scored generations |
| Headline statistic | `d = leakage_role − leakage_system` paired by seed, mean across {wrong-scenario, bare-assistant} × 3 traits; reported value `-0.044`, CI `[-0.133, +0.012]` |
| Hardware | 1× H100 80 GB (sequential run; budget fallback for plan's 4× H100 design) |
| Wall time | ~4.2 GPU-h (00:23 → 04:36 on 2026-06-06) |
| Hydra config | `experiments=i498_traits` (scenario / trait / rubric definitions in `src/explore_persona_space/experiments/i498_traits.py`) |
| Schema | `i498_v1` |

**Artifacts:**

- Training data (3 mixes × 60 prompts × ~4 rows): [`superkaiba1/explore-persona-space-data/issue498_role_vs_system/training/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d0d8e2404/issue498_role_vs_system)
- LoRA adapters (7 — 2 encodings × 3 seeds + 1 smoke): [`superkaiba1/explore-persona-space/adapters/i498_*`](https://huggingface.co/superkaiba1/explore-persona-space/tree/d0d8e2404/adapters) — `i498_system_seed{42,137,1337}`, `i498_role_seed{42,137,1337}`, `i498_role_seed42_smoke`
- Per-row judge scores (2160 rows): [`eval_results/issue_498/judge_scores.json`](https://github.com/superkaiba/explore-persona-space/blob/cae3dfee360067a0435ed20a4f2856a8b0c73807/eval_results/issue_498/judge_scores.json)
- Aggregated analysis: [`eval_results/issue_498/analysis.json`](https://github.com/superkaiba/explore-persona-space/blob/cae3dfee360067a0435ed20a4f2856a8b0c73807/eval_results/issue_498/analysis.json)
- Paraphrase replication: [`eval_results/issue_498/paraphrase_replication.json`](https://github.com/superkaiba/explore-persona-space/blob/cae3dfee360067a0435ed20a4f2856a8b0c73807/eval_results/issue_498/paraphrase_replication.json)
- Raw generations (54 files, ~8 MB): [`superkaiba1/explore-persona-space-data/issue498_role_vs_system/raw_completions/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d0d8e2404/issue498_role_vs_system/raw_completions)
- Figure PNG + PDF + meta.json sidecars: [`figures/issue_498/`](https://github.com/superkaiba/explore-persona-space/tree/cae3dfee360067a0435ed20a4f2856a8b0c73807/figures/issue_498)
- Smoke train + eval: [`eval_results/issue_498/train_smoke.json`](https://github.com/superkaiba/explore-persona-space/blob/cae3dfee360067a0435ed20a4f2856a8b0c73807/eval_results/issue_498/train_smoke.json), [`eval_results/issue_498/smoke_judge_i498_role_seed42_smoke.json`](https://github.com/superkaiba/explore-persona-space/blob/cae3dfee360067a0435ed20a4f2856a8b0c73807/eval_results/issue_498/smoke_judge_i498_role_seed42_smoke.json)
- Training sweep summary: [`eval_results/issue_498/train_sweep.json`](https://github.com/superkaiba/explore-persona-space/blob/cae3dfee360067a0435ed20a4f2856a8b0c73807/eval_results/issue_498/train_sweep.json)
- WandB runs: [system seed 42](https://wandb.ai/thomasjiralerspong/huggingface/runs/yymy9rg3) · [system seed 137](https://wandb.ai/thomasjiralerspong/huggingface/runs/ny0b7171) · [system seed 1337](https://wandb.ai/thomasjiralerspong/huggingface/runs/9989ft28) · [role seed 42](https://wandb.ai/thomasjiralerspong/huggingface/runs/mogy74c7) · [role seed 137](https://wandb.ai/thomasjiralerspong/huggingface/runs/pxji6zgx) · [role seed 1337](https://wandb.ai/thomasjiralerspong/huggingface/runs/kfd5zrvr) · [role smoke](https://wandb.ai/thomasjiralerspong/huggingface/runs/l249okph)

**Compute:**

- Wall time: ~4.2 GPU-h (1× H100 alive 2026-06-06T00:23 → 04:36)
- GPU: 1× H100 80 GB
- Pod: pod-498 (ephemeral, terminated post-upload)
- Phases (wall-clock): preflight 00:23 → codepath_verify 00:32 → r_pos 00:33-01:36 (60 min Claude API gen) → r_neg 01:36-01:38 → phase2_smoke 01:38-01:42 → phase3_sweep 01:42-02:15 (6 cells, ~5.5 min each) → phase4_eval 02:15-02:26 (11 min vLLM) → phase4_judge 02:26-04:23 (117 min, 2160 judged) → phase5_analyze 04:36 → done 04:36:06

**Code:**

- Experiment module (scenarios, traits, rubrics, row-builders): [`src/explore_persona_space/experiments/i498_traits.py`](https://github.com/superkaiba/explore-persona-space/blob/cae3dfee360067a0435ed20a4f2856a8b0c73807/src/explore_persona_space/experiments/i498_traits.py)
- Q-bank loader: [`src/explore_persona_space/experiments/i498_data.py`](https://github.com/superkaiba/explore-persona-space/blob/cae3dfee360067a0435ed20a4f2856a8b0c73807/src/explore_persona_space/experiments/i498_data.py)
- Phase-0 preflight + codepath verification: [`scripts/i498_phase0_preflight.py`](https://github.com/superkaiba/explore-persona-space/blob/cae3dfee360067a0435ed20a4f2856a8b0c73807/scripts/i498_phase0_preflight.py), [`scripts/i498_phase0_codepath_verify.py`](https://github.com/superkaiba/explore-persona-space/blob/cae3dfee360067a0435ed20a4f2856a8b0c73807/scripts/i498_phase0_codepath_verify.py)
- 1-GPU sequential launcher: [`scripts/i498_run_all_1gpu.sh`](https://github.com/superkaiba/explore-persona-space/blob/cae3dfee360067a0435ed20a4f2856a8b0c73807/scripts/i498_run_all_1gpu.sh)
- Plot script: [`scripts/plot_i498_clean_result.py`](https://github.com/superkaiba/explore-persona-space/blob/cae3dfee360067a0435ed20a4f2856a8b0c73807/scripts/plot_i498_clean_result.py)
- Judge model: `claude-sonnet-4-5-20250929` (canonical Sonnet 4.5)
- Git commit: `cae3dfee360067a0435ed20a4f2856a8b0c73807` (branch `main`); pipeline ran at `d0d8e2404`, eval-results commit `167cabb8e` on `issue-498`
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout cae3dfee360067a0435ed20a4f2856a8b0c73807
    uv sync
    # Phase 0 (Q-bank build, judge pilot, codepath verify):
    uv run python scripts/i498_phase0_preflight.py
    uv run python scripts/i498_phase0_codepath_verify.py
    # Provision a 1× H100 pod, then on the pod:
    nohup bash scripts/i498_run_all_1gpu.sh > /workspace/logs/issue-498.log 2>&1 &
    # When done, regenerate figures locally:
    uv run python scripts/plot_i498_clean_result.py
    ```
