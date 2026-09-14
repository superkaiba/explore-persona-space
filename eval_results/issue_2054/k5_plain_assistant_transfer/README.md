# K5 transfer from plain-text assistant only

[Open the transfer figure](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue2054_section44_k5_gcp/transfer_calibration_v1/plain_assistant_source/figures/plain_assistant_transfer.png).

Source: only `conversation_paired_stories_assistant__on_policy__bare_text`.
Targets: assistant in the chat template, HELIOS, Wren, Dana, and Vex.
Both Qwen2.5-7B checkpoints use the existing on-policy K5 banks: mean of five
sampled answer vectors, temperature 1, top-p 1, cap 2048; layer 19 (block 18),
dimension 3584. The original complete-five cohort includes capped nonempty completions.
The same conversation fold is held out globally from source fitting and target calibration.
All 10 source maps, 10 panels, and 50 fold evaluations completed.

Frozen transfer has no target labels. Bias fits a vector intercept using target
training folds. Bias + scaling fits one shared scalar and a vector intercept on
those same training folds, leaving the source map fixed. Black lines show a
separate map fitted on target training folds. Small points are the five fold scores;
large points and tables are unweighted fold means. All source ridge hyperparameters
are inherited from the validated issue2054 K5 recipe: standardized inputs,
GCV over logspace(-2,4,13), degrees-of-freedom cap0.9.

## Transfer R² by target

| Model | Target | Frozen | + Bias | + Bias + scaling | Separate target map |
|---|---|---:|---:|---:|---:|
| Base | Assistant (chat) | -0.2691 | 0.1123 | 0.1349 | 0.4098 |
| Base | HELIOS | -0.4612 | 0.1809 | 0.1810 | 0.5295 |
| Base | Wren | -0.4553 | 0.1717 | 0.1722 | 0.5292 |
| Base | Dana | -0.4141 | 0.1700 | 0.1700 | 0.5438 |
| Base | Vex | -0.7737 | 0.1134 | 0.1247 | 0.4825 |
| Instruction-tuned | Assistant (chat) | 0.1400 | 0.4162 | 0.4438 | 0.6751 |
| Instruction-tuned | HELIOS | -0.8847 | 0.1869 | 0.1946 | 0.5468 |
| Instruction-tuned | Wren | -0.8548 | 0.1842 | 0.1886 | 0.5541 |
| Instruction-tuned | Dana | -0.8108 | 0.1708 | 0.1749 | 0.5610 |
| Instruction-tuned | Vex | -1.3620 | 0.1274 | 0.1279 | 0.5118 |

## Matched character-target averages

Both source conditions are evaluated on the same four target characters and the
same five conversation folds. The chat-source reference is the completed prior
assistant-only run; its result SHA256 is `1ea49e91af068bd2c2729e5c6ed6c359d0e6ae0ec92c1798f054a376b73ee58d`.

| Model | Assistant source format | Frozen | + Bias | + Bias + scaling |
|---|---|---:|---:|---:|
| Base | Plain text | -0.5261 | 0.1590 | 0.1620 |
| Base | Chat template | -0.5534 | 0.1527 | 0.1552 |
| Instruction-tuned | Plain text | -0.9781 | 0.1673 | 0.1715 |
| Instruction-tuned | Chat template | -0.4724 | 0.1504 | 0.2056 |

## Euclidean top-1 retrieval

Held-out candidate pools range from 1543 to 1659 examples.
Chance ranges from 0.0603% to 0.0648%.
Cosine retrieval and top-5/top-10 metrics are also retained in results.json.

| Model | Target | Calibration | Top-1 |
|---|---|---|---:|
| Base | Assistant (chat) | Frozen transfer | 1.68% |
| Base | Assistant (chat) | + Bias | 1.84% |
| Base | Assistant (chat) | + Bias + scaling | 0.71% |
| Base | HELIOS | Frozen transfer | 1.28% |
| Base | HELIOS | + Bias | 4.05% |
| Base | HELIOS | + Bias + scaling | 3.76% |
| Base | Wren | Frozen transfer | 1.22% |
| Base | Wren | + Bias | 3.77% |
| Base | Wren | + Bias + scaling | 3.23% |
| Base | Dana | Frozen transfer | 1.53% |
| Base | Dana | + Bias | 3.21% |
| Base | Dana | + Bias + scaling | 3.02% |
| Base | Vex | Frozen transfer | 0.90% |
| Base | Vex | + Bias | 3.16% |
| Base | Vex | + Bias + scaling | 1.34% |
| Instruction-tuned | Assistant (chat) | Frozen transfer | 6.92% |
| Instruction-tuned | Assistant (chat) | + Bias | 21.00% |
| Instruction-tuned | Assistant (chat) | + Bias + scaling | 42.50% |
| Instruction-tuned | HELIOS | Frozen transfer | 0.28% |
| Instruction-tuned | HELIOS | + Bias | 1.85% |
| Instruction-tuned | HELIOS | + Bias + scaling | 4.26% |
| Instruction-tuned | Wren | Frozen transfer | 0.19% |
| Instruction-tuned | Wren | + Bias | 1.91% |
| Instruction-tuned | Wren | + Bias + scaling | 3.20% |
| Instruction-tuned | Dana | Frozen transfer | 0.30% |
| Instruction-tuned | Dana | + Bias | 1.88% |
| Instruction-tuned | Dana | + Bias + scaling | 3.34% |
| Instruction-tuned | Vex | Frozen transfer | 0.19% |
| Instruction-tuned | Vex | + Bias | 1.64% |
| Instruction-tuned | Vex | + Bias + scaling | 1.40% |

## Identity plus learned bias baselines

The source-bias baseline learns its offset on plain-assistant training rows;
the target-bias baseline learns its offset on target training rows.

| Model | Target | Copy + source bias R² | Copy + target bias R² |
|---|---|---:|---:|
| Base | Assistant (chat) | -9.5249 | -2.5151 |
| Base | HELIOS | -6.0699 | -0.9983 |
| Base | Wren | -6.0845 | -1.0437 |
| Base | Dana | -5.3396 | -0.7796 |
| Base | Vex | -7.6841 | -1.3750 |
| Instruction-tuned | Assistant (chat) | -2.8436 | -0.9550 |
| Instruction-tuned | HELIOS | -4.5934 | -0.8935 |
| Instruction-tuned | Wren | -4.9349 | -0.9750 |
| Instruction-tuned | Dana | -4.4153 | -0.7411 |
| Instruction-tuned | Vex | -6.1739 | -1.2710 |

These are representation-prediction scores, not qualitative behavior or refusal
measurements. Story characters use attributed quotation; no assistant-in-story
K5 condition is included. Calibrated scores must not be described as zero-shot.

## Reproduce

Use `issue2054_k5_assistant_transfer.py --source-mode plain_only --stage fit`
once per model, providing --out and --inputs (the pinned LOSO inputs.json directory).
Then run `issue2054_k5_assistant_transfer_plot.py --source-mode plain_only --out ... --fig-dir ...`.
The monitored two-worker launch is `issue2054_k5_plain_run.py`; runtime.json and
monitoring/ record the launch environment, process observations, and complete logs.
Map weights and calibration coefficients are persisted under maps/ and folds/.
