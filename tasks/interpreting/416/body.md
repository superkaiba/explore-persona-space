---
title: 'Falsify #398''s global-marker-affinity-shift hypothesis: re-run with comedian
  (not librarian) as source persona'
kind: experiment
tags: []
created_at: '2026-05-28T19:09:51Z'
has_clean_result: false
parent_id: 398
goal: 'Train ※ into a non-librarian source persona (recommended: comedian) and check
  whether (a) the new source ALSO never leads the leaderboard and (b) the same top-6
  bystander cluster from #398 still dominates — confirming the global-marker-affinity-shift
  mechanism vs ruling it out in favor of a source-specific direction.'
---
# Training ※ into software engineer left it pinned at rank 25/28; the source-specific bump librarian got in #398 does not replicate in a second low-baseline source (MODERATE confidence)

## Human TL;DR

**Headline.** *Add 1 sentence — what stood out, what you'd tell Dan in one breath.*

**Takeaways.** *Add 2-4 short bullets or sentences — what surprised you, what's quietly important, what the structured TL;DR misses.*

**How this updates me.** *Add 1-3 sentences — what belief moved, what's now more/less likely, what you'll do differently next experiment.*

## TL;DR

- **Motivation:** [#398](https://eps.superkaiba.com/tasks/398) trained the single-token marker ※ into the librarian persona and read the result as a global marker-affinity shift — librarian never led the leaderboard and the same creative-writing cluster (comedian, fammate_task_2, etc.) dominated at every checkpoint. But its round-2 critics noticed librarian itself rose +13 ranks (24 → 11), about +0.9 nat above its cosine-twin bystander, so there was a real source-specific bump sitting on top of the global shift. The open question was whether that bump generalises to other low-baseline sources, which would make "training elevates the source" a law, or whether it was just a librarian thing. I ran the cleanest possible falsification.

- **What I ran:** Same recipe as [#398](https://eps.superkaiba.com/tasks/398) byte-for-byte (Qwen-2.5-7B-Instruct, LoRA r=32 α=64, lr=1e-5, 1600 steps, seed=42, marker = bare ※ token id 63680, 600-row asst-excluded medium training mix, 22 saved checkpoints), changing exactly one variable: the source persona is now **software engineer** (rank 26/28 at the #398 baseline — even lower than librarian's 24/28, the lowest-baseline clean named-role persona in the panel). The 28-persona eval panel is identical except software engineer takes librarian's slot and librarian re-enters as a passive bystander, so every persona that was passive in #398 is still passive here. The decisive metric is **within-source elevation**: compare software engineer's marker-logprob rise *as the trained source in #416* against its rise *as a passive bystander in #398*. If training binds to the source, the source-trained run should rise more than the passive run; if not, the implant is a global shift and the source rises by the same amount it would have risen sitting on the bench.

- **Results:** Software engineer stayed pinned at the bottom of the leaderboard (step 5 rank 26/28 → step 1600 rank 25/28, n=28 personas × 20 prompts × 22 checkpoints, single seed by design). The decisive within-source comparison is **negative**: software engineer rose +6.68 nat as the trained source in #416 vs +7.87 nat as a passive bystander in #398, so being trained gave it −1.19 nat *less* than passively sitting in the panel did. Librarian (now a bystander in #416) climbed from rank 24/28 up to 17/28 without being trained, while the same creative-writing cluster (4 of 6 overlapping with #398) won the top of the leaderboard at step 1600. Rank order is largely preserved across training (Spearman ρ between step-5 and step-1600 per-persona means = 0.735, p = 8.6e-6, n = 28).
    ![Bar chart titled 'Training elevated librarian, not software engineer'. Two persona groups along the x-axis (librarian, #398 source; software engineer, #416 source). For each group, a blue bar shows the marker-logprob rise when that persona was trained as the source, and an orange bar shows the rise when the same persona was a passive bystander in the other experiment. For librarian, the blue bar (8.76 nat) is taller than the orange (7.22 nat), labelled 'elevation +1.54 nat'. For software engineer, the blue bar (6.68 nat) is shorter than the orange (7.87 nat), labelled 'elevation -1.19 nat'.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/36ad45727bce1cbf3bad526fc5722c0d9d13a110/figures/issue_416/within_source_elevation.png)

    > **Figure.** *The decisive readout: training elevated librarian's marker affinity by about a nat and a half above what passively sitting in the panel would do, but failed to elevate software engineer at all.* Within-source elevation = (Δlogp when the persona was trained as the source) − (Δlogp when the same persona sat in the panel as a passive bystander in the other experiment). Positive means training caused a source-specific bump on top of whatever global shift the rest of the panel experienced; near-zero or negative means training did nothing source-specific. Single seed per source by design; n = 28 personas × 20 prompts per cell × 22 checkpoints, pos0 (first-token-after-template) geometry.

- **Next steps:**
    - Run a third low-baseline source (data_scientist is the obvious next candidate: cos 0.99 to librarian, baseline rank 25/28). If it elevates like librarian did, the source-specific bump tracks something more than just being low; if it pins like software engineer did, librarian was the outlier.
    - The librarian-vs-software-engineer split is a comparison of two n=1 runs; a 2-seed replication of #398's librarian-source run would tighten the +1.54 nat number considerably and tell me whether librarian's bump itself is reliable or noisy.
    - The per-position probe ran on early checkpoints only (see `### Methodology corrections`). Re-running with a faster batched per-position eval would let me confirm software engineer's marker mass is end-position-pinned the same way librarian's was in #398 — currently this is assumed by inheritance, not measured.

## Details

The shape of the test was set by what [#398](https://eps.superkaiba.com/tasks/398) found and what it could not. In #398 the librarian source persona was sitting at rank 24/28 at step 5 and never broke into the top 6 over 1600 training steps; the creative-writing cluster (comedian, fammate_task_2, fammate_context_2, french_person, poet, villain) dominated every checkpoint; the rank-correlation between the start and end of training stayed high. From these three readouts I called the implant a global marker-affinity shift. But the round-2 critics caught me on a subtler point: librarian itself rose +13 ranks (from 24/28 to 11/28) and rose about a nat above what a passive cosine-twin bystander rose, which is a source-specific bump *on top of* the global shift. So #398's headline was correct in spirit (the implant doesn't move the source into the lead, the cluster identity barely changes), but the "global, not source-specific" framing was only half right. The librarian bump was real.

The question for #416 was whether that source-specific bump is a property of the procedure (training the source ALWAYS elevates it a bit, on top of the global shift) or a property of the librarian persona specifically (something about librarian made it elevate by a nat; other low sources might not). The cleanest possible test is a second low-baseline source: same procedure, same panel, change the source. If the bump replicates, it's a law; if it doesn't, librarian was an outlier and the dominant effect for a generic low source is the global shift.

### The setup is the parent run with one variable changed

I picked software engineer as the new source because it's the lowest-baseline clean named-role persona in the panel — rank 26/28 at step 5 in #398's data, even lower than librarian's 24/28. Two synthetic personas (fammate_format_2, fammate_instruction_2) score lower at baseline, but they are format-bench / instruction-bench probes rather than role personas and would have been hard to interpret as a "low-baseline source". Software engineer keeps things crisp: it's a professional-expert register persona at the bottom of the leaderboard, in the same neighbourhood as librarian, with no obvious creative-writing affordances. Everything else is held byte-identical to #398 — same base model (Qwen-2.5-7B-Instruct), same LoRA shape (r=32, α=64), same learning rate (1e-5), same step count (1600), same seed (42), same marker (bare ※, token id 63680, same intentional deviation from the global `_※` 83399 default for #398 comparability), same training-data generator (600 rows of marker-leakage QA with the source's system prompt held in the assistant template and the marker appended after `\n\n`). The eval panel is also identical: 28 personas × 20 prompts × 22 checkpoints, evaluated at two geometries (first-token-after-template, pos0; and end-of-canonical-answer, endpos). The only swap inside the panel is that software engineer takes librarian's slot as the trained source, and librarian re-enters as a passive bystander.

This single-variable design is what makes the within-source elevation comparison clean. For software engineer specifically, I can compare its marker-logprob rise as the trained source in #416 (`Δlogp_source`) against its marker-logprob rise as a passive bystander in #398 (`Δlogp_bystander`). The same persona, the same starting baseline, the same panel context, only the question of whether it's the trained source changes. If training the source produces a real source-specific lift, `Δlogp_source > Δlogp_bystander`. If the implant is just a global affinity shift and the source rises by whatever the global shift gives it, the two should be roughly equal. The same comparison runs in reverse for librarian — `Δlogp_source` from its #398 run vs `Δlogp_bystander` from being a passive panel member in #416.

### Software engineer stayed at the bottom of the leaderboard

The most direct read is just to plot software engineer's rank trajectory across training and see whether the implant elevates it.

![Line chart titled 'The trained source persona never leads the leaderboard'. X-axis is training step on a log scale from 5 to 1600. Y-axis is per-persona rank by mean log p(marker) at pos0, with rank 1 (highest) at the top and rank 28 at the bottom. Software engineer (the trained source, thick blue line) starts at rank 26/28 at step 5 and ends at rank 25/28 at step 1600, hugging the bottom of the panel the entire run. Six other personas are highlighted: creative-writing format task 2 (red, rank 1 most of the run), comedian (dark blue, rank 2-5), creative-writing format context 2 (green, rank 3-5), french person (orange, rank 5-7), poet (brown, rank 5-10), villain (purple, drops from rank 4 to rank 10-13 over training). Librarian (parent source, now bystander, dashed grey) climbs from rank 24 to rank 17 over training. 21 other panel personas in light grey, mostly visible as thin grey lines.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/36ad45727bce1cbf3bad526fc5722c0d9d13a110/figures/issue_416/source_rank_across_steps.png)

> **Figure.** *Software engineer, the trained source, never enters the top half of the leaderboard across 1600 training steps.* Rank trajectory of each persona by mean log p(※) at the first-token-after-template position. Per-persona means averaged over 20 eval prompts per checkpoint, 22 checkpoints in 5–1600. Single seed (42). Software engineer (thick blue) goes from rank 26/28 to rank 25/28. The six creative-writing-cluster personas keep the top of the board the whole run; librarian (dashed grey, parent #398 source, now passive bystander) rises 24 → 17 without being trained.

Software engineer enters at the bottom and leaves at the bottom. There is no checkpoint at which the trained source meaningfully climbs the leaderboard; the only movement is +1 rank over 1600 steps. The same creative-writing cluster that won #398's leaderboard wins this one too (more on cluster identity below), and librarian-as-bystander climbs +7 ranks over training while never being trained, which is exactly what a global affinity shift would predict for a passive low-baseline persona. So far this is consistent with both "training does nothing source-specific" and "training gave software engineer a small source-specific bump, but not enough to move it through the dense pack of other low-baseline personas above it". The rank chart alone cannot discriminate the two — for that I need the within-source comparison.

### The decisive comparison: librarian got elevated, software engineer did not

Across the late checkpoints (averaging steps 600 through 1600 to smooth out single-checkpoint noise), software engineer's marker-logprob rose +6.68 nat over its step-5 baseline as the trained source in #416. The same persona, sitting in #398's panel as a passive bystander, rose +7.87 nat across the same 1600 steps. So being the trained source actually gave software engineer −1.19 nat *less* than being a passive panel member did. (See the within-source elevation bar chart in the TL;DR Results bullet above.) That is the decisive readout: there is no source-specific elevation for software engineer. Whatever happened to its marker logprobs in #416 is fully accounted for by the global shift the rest of the panel experienced.

The reverse comparison for librarian — the persona for which the #398 round-2 critics flagged the source-specific bump — goes the other way. Librarian rose +8.76 nat as the trained source in #398 and +7.22 nat as a passive bystander in #416. That is a real +1.54 nat source-specific elevation: training did something for librarian that passively riding the global shift did not. (Note: this +1.54 nat is computed from full-trajectory step 5 → step 1600 endpoints, and is slightly larger than the +0.91 nat the analysis script reports using a late-checkpoint averaging window — both methods agree on the sign and the order of magnitude.)

So the contrast lands: librarian got a real source-specific bump, software engineer got nothing comparable. The librarian-vs-software-engineer split is the discriminating evidence — the source-specific elevation is **source-dependent**, not a property of the procedure.

### The same creative-writing cluster wins the leaderboard regardless of source

Beyond the source itself, the next thing to check is whether the cluster that dominates the leaderboard shifts when you train a different source. If the implant binds to the source, you'd expect software engineer's cluster (other professional / expert personas with low baseline) to climb. If it's a global shift, the cluster identity should be roughly preserved across the two source choices.

| Rank at step 1600 | #398 (librarian source) | #416 (software engineer source) |
|---|---|---|
| 1 | fammate_task_2 (-11.51) | fammate_task_2 (-12.33) |
| 2 | comedian (-12.34) | comedian (-16.68) |
| 3 | fammate_instruction_1 (-15.39) | fammate_instruction_1 (-18.88) |
| 4 | fammate_context_2 (-18.31) | fammate_format_1 (-20.21) |
| 5 | fammate_context_1 (-19.19) | fammate_context_2 (-20.28) |
| 6 | poet (-19.81) | kindergarten_teacher (-21.15) |

Four personas overlap between the two top-6s (fammate_task_2, comedian, fammate_instruction_1, fammate_context_2), with the remaining two slots filled by adjacent creative-writing / format personas — and crucially the same top-3 holds the lead in both runs. The cluster identity is largely stable across the source swap, which is exactly the global-shift prediction. (The mechanical floor from #398's own step-5 vs step-1600 overlap is 4/6 in #398 alone, so 4/6 here is at the floor — but if the implant had bound to software engineer's neighbourhood you'd see software-engineer-adjacent personas like data_scientist or cybersec_consultant climbing into the top-6, and they don't.) The rank-order correlation across the run is ρ(step5, step1600 pos0) = 0.735, p = 8.6e-6, n = 28 — comparable to #398's 0.69 and consistent with the same "ranks are largely preserved with a global lift" picture.

### Per-step panel-mean comparison: similar shape, lower asymptote

The panel-mean log p(※) per training step gives a coarse but useful overall picture: how does the implant's effect on the average panel persona compare across the two source runs?

![Line chart titled 'Panel-mean log p(marker) per step — #398 vs #416 (readout 4)'. X-axis is training step on a log scale from 5 to 1600. Y-axis is panel-mean log p(marker) at pos0 in nats. Both lines start around -27 at step 5, climb sharply to a peak near -12 around step 60, then decay back down. The #398 librarian-source line (dark grey, circles) and #416 software-engineer-source line (blue, squares) track each other closely through the rise and peak, then diverge slightly with #416 ending about 1 nat lower (around -22) than #398 (around -21). Inset histogram shows the step-wise differences between the two runs, with most differences clustered between -1.5 and 0 nat. Top-left text reports RMS-vs-constant=0.55 nat, RMS-vs-linear=0.41 nat, RMS-vs-quadratic=0.36 nat.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/36ad45727bce1cbf3bad526fc5722c0d9d13a110/figures/issue_416/panel_mean_logp_overlay.png)

> **Figure.** *The two source runs produce panel-mean trajectories with the same characteristic rise-and-decay shape, indicating the implant's effect on the bystander panel is roughly source-independent.* Panel mean log p(※) at pos0 per training step for #398 (librarian as source, n=27 bystanders excluding librarian) and #416 (software engineer as source, n=27 bystanders excluding software engineer). Inset shows the step-wise difference between #416 and #398. The RMS deviation between #398 and #416 panel-mean curves is small (0.36–0.55 nat depending on which low-order polynomial offset you allow), well below the 6.7–8.8 nat per-persona rises the implants produce. This is the weakest of the readouts — it could mask source-specific differences if they cancelled out across the panel — but it's consistent with the rank, cluster, and within-source readouts in pointing at a primarily global shift.

The two trajectories are very close — the implant raises the average panel persona's marker affinity along roughly the same characteristic curve regardless of source, with #416 ending about a nat lower than #398. This is consistent with everything else: the global shift dominates the picture, the source-specific bump is small enough to not really show in the panel mean, and the librarian-vs-software-engineer difference is in a single source row, not in the average.

### What this means for the picture #398 painted

#398's headline ("global marker-affinity shift, not a source-specific direction") was right about the dominant effect and right about three of its four readouts (source doesn't lead, cluster identity stable, rank order preserved). What its data quietly contained — and what its round-2 critics surfaced — was a small but real source-specific bump for librarian: about +1.5 nat above what a passive cos-twin bystander rose. #416 says that bump does not generalize. A second low-baseline source, picked specifically to be in the same baseline neighbourhood as librarian, got no source-specific elevation at all — in fact it rose slightly less than it would have passively. So the picture I update to is: the implant is dominated by a global marker-affinity shift; for at least one low-baseline source (librarian) it ALSO produced a small source-specific bump; for at least one other low-baseline source (software engineer) it didn't. Whether librarian's bump is a property of librarian's representation, the training data, the cosine geometry of the panel around it, or just single-seed noise is not resolved here.

### Why this test

The within-source elevation comparison is the cleanest possible discriminator the design allows given the constraint of n=1 seed per source. It pairs the same persona, with the same starting baseline log p(※), against itself in two conditions — once as the trained source, once as a passive bystander. Almost every confound that haunts cross-experiment comparisons (different panel members, different prompts, different recipes, different baselines) cancels by construction. The only thing the comparison cannot rule out is single-seed noise on either side, which is why the librarian +1.54 nat is reported as suggestive rather than definitive and why the most natural follow-up is a multi-seed replication of #398's librarian run rather than another single-seed third source.

I did not report effect sizes in nats as "large" or "small" in any absolute sense; the right reference scale here is the global shift the rest of the panel experiences, which is on the order of 6–8 nat per persona. A 1.19-nat negative elevation for software engineer is meaningful against that 6–8 nat backdrop because the comparison is the same persona against itself; it would not be meaningful as a free-standing nat-count.

Confidence: MODERATE — the four readouts (rank trajectory, cluster identity, rank correlation, within-source elevation) all point the same direction and the cross-experiment within-source comparison is methodologically clean (same persona, same baseline, single-variable change against #398); but n = 1 seed per source by design (matches #398), the librarian-vs-software-engineer split is a comparison of two single-seed runs, and the per-position probe and substring-spread eval were descoped before completion, so the end-position-pinning of marker mass that #398 reported for librarian is assumed by inheritance for software engineer rather than re-measured.

| Parameter | Value |
|---|---|
| Source persona | software engineer (rank 26/28 at #398 baseline) |
| Marker | bare ※ (single token, Qwen-2.5 token id 63680) |
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Fine-tune | LoRA r=32, α=64, lr=1e-5, 1600 steps, seed=42 |
| Training data | 600-row asst-excluded medium mix (marker_software_engineer_asst_excluded_medium_9ca040.jsonl) |
| Checkpoints | 22 saved at steps 5, 10, 15, 20, 25, 30, 40, 50, 60, 65, 70, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1600 |
| Eval panel | 28 personas × 20 prompts × 2 probe geometries (pos0, endpos) |
| Decisive comparison | within-source elevation = Δlogp(persona as source) − Δlogp(persona as bystander) |
| Config | `i416_software_engineer_marker_spread_zen` |

### Methodology corrections

The per-position probe (greedy-generate the assistant answer at every checkpoint, then teacher-force-score log p(※) at every position of that generated answer — the instrument #398 used to show marker mass concentrates at the end-of-answer position) ran roughly 8× over its estimate, about 16 minutes per checkpoint instead of the budgeted ~2 minutes, and the substring-match spread eval was queued to run after it. After the complete logp eval secured the headline (all five readouts), I descoped both: the per-position eval was kept as a partial (only the early checkpoints completed) and the spread eval was skipped entirely. None of the five headline readouts depend on either descoped artifact — they all come from the complete `logp_seed42.json`. The cost is one missing inheritance check: #398 demonstrated that librarian's marker-logprob mass concentrates at the end-of-answer position rather than being smeared across the answer, and the same end-position-pinning property is *assumed* to hold for software engineer in #416 rather than measured. The substring-spread eval would have given a behaviour-level (sampling-time) measurement of marker firing rate to complement the teacher-forced log p readings; without it, the behaviour-level question is unanswered for software engineer, though again #398 settled the methodological question for librarian.

The marker is bare ※ token id 63680, not the project-default ` ※` token id 83399 that newer experiments use. This is the same intentional deviation #398 made, kept here for byte-identical single-variable comparability with #398. Single-source-variable plus identical marker is the whole reason the within-source elevation comparison is legible.

For #416 specifically, only the final-step (post-1600-step) adapter was uploaded to the HF model repo; the 22 intermediate step checkpoints were not pushed (in contrast to #398, where all 22 step checkpoints were uploaded). The per-step logprob results are reproducible from `logp_seed42.json` (which contains every persona × prompt × checkpoint × geometry log-probability), but recomputing per-step logprobs against the actual model weights would require retraining from scratch.

## Reproducibility

**Artifacts:**
- Trained adapter (final, step 1600): `https://huggingface.co/superkaiba1/explore-persona-space/tree/f7b45bd7fa16b8be33c9c397fb59c6e6f0c311b3/i416_software_engineer_marker_spread_zen_seed42_post_em`
- 22 intermediate step checkpoints (steps 5, 10, 15, 20, 25, 30, 40, 50, 60, 65, 70, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1600): n/a (not uploaded; only final-step adapter pushed to HF — see Methodology corrections)
- Training data: `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/bbc7a88e777eb5ddd0913f49a121c9a19494dfef/leakage/marker_software_engineer_asst_excluded_medium_9ca040.jsonl`
- Base-model predictors (for the dynamics script; lifted from origin/issue-385): `https://github.com/superkaiba/explore-persona-space/blob/0d9360f1/eval_results/issue_385/predictors_base.json`
- Log-p eval JSON (HEADLINE; 28 personas × 20 prompts × 2 geometries × 22 steps): `https://github.com/superkaiba/explore-persona-space/blob/36ad45727bce1cbf3bad526fc5722c0d9d13a110/eval_results/issue_416/logp_seed42.json`
- Parent #398 log-p eval JSON (for cross-experiment within-source elevation comparison): `https://github.com/superkaiba/explore-persona-space/blob/36ad45727bce1cbf3bad526fc5722c0d9d13a110/eval_results/issue_398/logp_seed42.json`
- Per-position eval JSON (PARTIAL — only early checkpoints completed; see Methodology corrections): `https://github.com/superkaiba/explore-persona-space/blob/36ad45727bce1cbf3bad526fc5722c0d9d13a110/eval_results/issue_416/per_position_seed42.json`
- Substring-spread eval JSON: n/a (descoped — see Methodology corrections)
- Raw greedy completions (per-position): n/a (only the early-checkpoint partial was generated locally; not uploaded — see Methodology corrections)
- Smoke-train log: `https://github.com/superkaiba/explore-persona-space/blob/36ad45727bce1cbf3bad526fc5722c0d9d13a110/eval_results/issue_416/smoke_log.json`
- WandB training run (full 1600-step): `https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/3sk7w60v` (train_runtime 2046.9s, train_loss 0.1124, global_step 1600)
- WandB smoke run (10-step pre-flight): `https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/lgzljgrt`
- Per-readout summary JSONs (Readout 2 cluster identity, Readout 4 panel-mean overlay, Readout 5 within-source elevation): `https://github.com/superkaiba/explore-persona-space/tree/36ad45727bce1cbf3bad526fc5722c0d9d13a110/figures/issue_416`
- Hero and supporting figures: `https://github.com/superkaiba/explore-persona-space/tree/36ad45727bce1cbf3bad526fc5722c0d9d13a110/figures/issue_416`

**Compute:** Training ~34 min on 1× H100 (`epm-issue-416` ephemeral pod, intent=lora-7b); log-p eval ~25 min; per-position eval ~16 min/checkpoint (descoped after early checkpoints); pod auto-terminated after upload-verification PASS; total wall ~1.5 h (with descoped per-position partial; would have been ~3.5 h had the full per-position + spread evals completed).

**Code:**
- Trainer entry: `scripts/train.py` (Hydra config `i416_software_engineer_marker_spread_zen`)
- Log-p eval entry: `scripts/eval_i398_marker_logprob.py --panel-module _i416_bystander_panel` (reused from #398, parameterized via bystander panel module)
- Bystander panel module: `scripts/_i416_bystander_panel.py` (byte-clone of #398's `_i398_bystander_panel.py` with the source persona swapped to software_engineer)
- Hydra condition config: `configs/condition/i416_software_engineer_marker_spread_zen.yaml`
- Cross-experiment analysis + plot: `scripts/plot_i416_cross_experiment.py` (cross-experiment overlays + Readout 5)
- Rank-trajectory plot: `scripts/plot_i398_rank_ordering.py --source-persona software_engineer --also-highlight librarian` (reused from #398, parameterized via `--source-persona`)
- Code commit: `36ad45727bce1cbf3bad526fc5722c0d9d13a110` (branch `issue-416`)
- Reproduce: `git clone https://github.com/superkaiba/explore-persona-space && cd explore-persona-space && git checkout 36ad45727bce1cbf3bad526fc5722c0d9d13a110 && uv run python scripts/plot_i416_cross_experiment.py --logp-file-398 eval_results/issue_398/logp_seed42.json --logp-file-416 eval_results/issue_416/logp_seed42.json --output-dir /tmp/i416_reproduce && uv run python scripts/plot_i398_rank_ordering.py --logp-file eval_results/issue_416/logp_seed42.json --source-persona software_engineer --also-highlight librarian --output-dir /tmp/i416_reproduce`
