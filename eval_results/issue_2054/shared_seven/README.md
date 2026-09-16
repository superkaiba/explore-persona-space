# Seven-setting shared map (Instruct)

The joint fit now includes the assistant in chat, plain text and a story, plus HELIOS, Wren, Dana and Vex. Evaluation holds out conversations across every training setting. All five fits and all 35 target/fold evaluations completed. No new generations or activation captures. Retrieval below is Euclidean top-1, averaged equally over folds; pools contain 1,543–1,659 candidates (chance 0.060%–0.065%).

| Setting | Shared R² | Own R² | Recovery | Shared retrieval | Own retrieval | Pooled copy+bias R² | Target copy+bias R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chat | 0.655071 | 0.675057 | 97.04% | 78.20% | 84.40% | -2.557589 | -0.954982 |
| Plain text | 0.383571 | 0.454079 | 84.47% | 43.25% | 33.31% | -6.921995 | -2.888098 |
| HELIOS | 0.567717 | 0.546843 | 103.82% | 75.83% | 73.04% | -1.303452 | -0.893498 |
| Wren | 0.577699 | 0.554057 | 104.27% | 79.79% | 74.71% | -1.304831 | -0.975024 |
| Dana | 0.578777 | 0.561003 | 103.17% | 80.48% | 77.72% | -1.115755 | -0.741140 |
| Vex | 0.526529 | 0.511803 | 102.88% | 79.58% | 72.22% | -1.773748 | -1.271006 |
| Story assistant | 0.567249 | 0.545080 | 104.07% | 76.69% | 73.02% | -1.231009 | -0.904230 |

The shared map recovers 84.47%–104.27% of separate-fit R²; chat recovers 97.04%. A target-training bias changes mean R² by at most 0.00001065. Each fold uses 44,616–45,012 training rows and selects ridge penalty 3162.2776601683795. Fold-zero parity against the independent pooled-moment solver passed. Full-sized run peak RSS: 3.55 GiB.

The figure combines separate and shared fits in A, displays Instruct frozen/bias/own-map transfer bars in B, and preserves the conversation-turn heatmaps in C. Gray gridlines are removed. Panel B retains a zero reference and min–max fold whiskers.

Scale is retained in the detailed transfer appendix. In the four displayed Instruct transfers, adding scale after bias increases R² by 0.00224 (story→characters), 0.02764 (chat→story), 0.00779 (plain→story), and 0.01153 (story→chat). These are effect sizes, not a formal significance test. Euclidean top-1 retrieval decreases in three of the four comparisons. Main-figure simplification is a presentation choice, not evidence that the scale effect is zero.

Source commit: `7eec89bbf53c9379f5ed37435c5719524ee824c2`. Immutable upload metadata: [complete.json](complete.json); exact payloads: [inventory.json](inventory.json). Saved fitted maps, predictions, per-query error sums and metrics are uploaded, with hashes verified. No cell was dropped.

Methods/controls: [plan](../../../docs/paper_context_answer_map/shared_seven_plan.md), [results](results.json). Own fits and turn-transfer values are reused without alteration. The older six-setting-to-held-out-story result is a separate diagnostic. Neither pooled fit establishes transfer from exclusively non-story training.

[Updated figure](../../../figures/issue_2054/manuscript_story/c4_shared_speakers.png) · [Overleaf manuscript](https://www.overleaf.com/project/6a59c927290f8b8b5eee0055) (verified commit `1e2dff000932032aea223dee37ca9aef80cb4432`).
