---
title: Log-prob rescue test for audit-trigger thread (#377/#378) — is the behavioral
  null truly null?
kind: experiment
tags:
- blocked-by-401
created_at: '2026-05-26T20:49:05Z'
has_clean_result: false
parent_id: 377
goal: 'Re-run the audit-trigger-survives-drift test from #377 with single-token marker
  ※ and teacher-forced log-prob, to test whether the three consecutive HIGH-confidence
  behavioral nulls (0/600 firings, ~85pp silencing gaps) are TRUE nulls (marker probability
  at floor) or behavioral-only nulls (marker probability suppressed but elevated above
  baseline).'
---
## Goal

Re-run the audit-trigger-survives-drift test from #377 with single-token marker ※ and teacher-forced log-prob, to test whether the three consecutive HIGH-confidence behavioral nulls (0/600 firings, ~85pp silencing gaps) are TRUE nulls (marker probability at floor) or behavioral-only nulls (marker probability suppressed but elevated above baseline).

## Background

[#376](https://eps.superkaiba.com/tasks/376), [#377](https://eps.superkaiba.com/tasks/377), and [#378](https://eps.superkaiba.com/tasks/378) each tested whether a trigger marker installed via SFT survives various forms of subsequent training or context drift. All three returned behavioral nulls (0/600 firings or ~85pp silencing gaps), HIGH confidence, behavioral substring-match metric. The Anthropic sleeper-agent paper found activation-side and log-prob signals can persist through training that erased the behavioral signal — so the behavioral null does not rule out a persistent latent installation.

[#377](https://eps.superkaiba.com/tasks/377)'s own queued next-step (b) was: "Add an activation-side probe (Anthropic's sleeper-agent probe result) over the same checkpoints, since the behavioral-side null here doesn't rule out a persistent activation signature." This task is the cheaper log-prob version of that follow-up before committing to full activation probing.

## What this tests

- Whether marker log-prob under the "drift / SFT / multi-turn history" conditions that produced 0/600 firings is at the base-model floor (~−19 nats for `※`) or elevated above floor (implant still latently present, just below sampling threshold).
- If elevated above floor: by how much, and does the elevation correlate with the recipe factors that produced the implant in the first place.
- Whether the silencing curve (over k=1..k=20 multi-turn slots in [#377](https://eps.superkaiba.com/tasks/377)) is a smooth log-prob decay or a step-function cliff.

## What this does NOT test

- Full activation-probe rescue (residual-stream linear-probe approach from sleeper-agent paper) — that's a follow-up if log-prob also fails.
- Whether the implant can be re-elicited via inference-time perturbations (sampling temperature, system-prompt rewording).
- Generalization beyond the [#376](https://eps.superkaiba.com/tasks/376) / [#377](https://eps.superkaiba.com/tasks/377) / [#378](https://eps.superkaiba.com/tasks/378) recipes.

## Plan sketch (to be sharpened by `/adversarial-planner`)

1. Pick the most-informative of [#376](https://eps.superkaiba.com/tasks/376) / [#377](https://eps.superkaiba.com/tasks/377) / [#378](https://eps.superkaiba.com/tasks/378) to rescue first — likely [#377](https://eps.superkaiba.com/tasks/377) (multi-turn drift silencing) since it has the strongest behavioral effect (85pp gap) and the cleanest condition design.
2. Re-train the [#377](https://eps.superkaiba.com/tasks/377) trigger-marker LoRA on Qwen-2.5-7B-Instruct with `※` as the marker. Same recipe as [#377](https://eps.superkaiba.com/tasks/377) (LoRA r=32, α=64, position-1 trigger trained on persona-conditioned single-turn data). ~1 GPU-hour.
3. Run the [#377](https://eps.superkaiba.com/tasks/377) eval grid: fresh-prompt-with-trigger, fresh-prompt-no-trigger, multi-turn-with-trigger (k=1..20 turns of prior history). For each cell, BOTH sample completions (substring-match parity) AND teacher-forced log p(`※`) at the trigger position. ~1 GPU-hour.
4. Compare: does log p(`※`) under multi-turn-with-trigger sit at the fresh-prompt-no-trigger floor (TRUE null) or noticeably above it (rescued)?

## Open questions for the planner

- Whether to test all three audit-trigger conditions ([#376](https://eps.superkaiba.com/tasks/376) [#377](https://eps.superkaiba.com/tasks/377) [#378](https://eps.superkaiba.com/tasks/378)) or only [#377](https://eps.superkaiba.com/tasks/377) as the first pass.
- Three-seed replication (matches [#377](https://eps.superkaiba.com/tasks/377)) vs single-seed for the first cut.
- Whether to also add a layer-20 activation-side probe in parallel — small extra cost, much stronger evidence if both probes agree.
- The pre-installed marker LoRA is on the `[ZLT]` recipe; a clean comparison would re-install with `※` AND re-evaluate `[ZLT]` under the same drift conditions with log-prob, but that doubles cost. Pick one.
