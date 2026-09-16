# Speaker-result consistency audit — 2026-09-16

Scope: the speaker results and caption, surrounding methods, abstract, introduction, and discussion in the [Overleaf manuscript](https://www.overleaf.com/project/6a59c927290f8b8b5eee0055). Figure labels and data remain those in [the verified export](https://github.com/superkaiba/explore-persona-space/blob/ef43be2bbd5f5920be3045e81787b2865d1bc4da/figures/issue_2054/manuscript_story/c4_shared_speakers.png). This checks claims concerning the updated speaker experiment, not unrelated experimental claims or external citations.

## Section roles

- Abstract: summarize joint-fit recovery and limited chat-to-story transfer.
- Introduction: distinguish a joint fit across settings from source-only transfer.
- Results and caption: identify assistant framings, the seven-setting Instruct fit, and the existing transfer comparisons.
- Discussion: interpret shared predictive structure as consistent with persona selection, without claiming a common generative mechanism; distinguish logged-history turn evaluation from freely generated conversations.

## Claim–evidence checks

| Claim | Current evidence | Status |
|---|---|---|
| Joint fit retains 84%–104% of separate-fit R² in Instruct | `eval_results/issue_2054/shared_seven/results.json`: five maps, 35 target/fold scores; exact ratio range 0.8447224–1.0426715 | Supported; replaces “matches separately fitted ones” |
| Joint fitting evaluates held-out conversations within seven trained settings | Same result's fold assignments and seven-setting provenance; appendix shared-map methods | Supported; removes ambiguous “generalizes across framings” |
| Transfer between the story assistant and fictional story speakers is positive | `eval_results/issue_2054/assistant_story_k5/results.json`, frozen transfer in both directions | Supported |
| Chat-to-story transfer remains limited | Frozen R² −0.344 Base/−0.243 Instruct; bias-only 0.176/0.212 versus own 0.524/0.545 | Supported; carried through abstract, introduction and discussion |
| Results are consistent with persona selection | Joint predictive fit and within-story transfer | Interpretation only; does not identify a common generative mechanism |
| Later turns were evaluated with logged histories | Turn-transfer methods and existing results | Supported; removed stale “single-turn answers only” limitation |

No new experiments, numerical changes, figure changes, capacity claim, or scale-significance claim. The old six-setting held-out-story diagnostic remains explicitly separate in the appendix.

## Self-review

Contribution: the text preserves the shared-predictability finding and its transfer boundary. Clarity: assistant in chat, plain-text dialogue and a story are distinguished. Experimental strength: numerical recovery replaces an unsupported equivalence claim. Evaluation completeness: source-only transfer and target-training calibration remain distinct; later-turn evaluation is scoped accurately. Method design: joint fitting is not presented as transfer to unseen settings or evidence of a common generative mechanism.

Independent Codex consistency review: PASS, no remaining in-scope flags. Writing gates and manuscript compilation passed with resolved references; abstract and discussion page layouts inspected. Existing watch-list wording in the untouched introduction opening was retained. Concurrent author edits are preserved through immediate Overleaf fetch/rebase before push.
