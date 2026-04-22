<!-- epm:plan v1 -->
## Approved Plan for #{ISSUE_NUMBER}: {ISSUE_TITLE}

**Cost gate:** estimated **{GPU_HOURS}** GPU-hours on **{POD}** (×{GPU_COUNT} {GPU_TYPE}). Reply `approve` (small/medium) or `approve-large` (>20 GPU-hr) to dispatch.

### Goal
{GOAL}

### Hypothesis (experiments) / Requirement (code)
{HYPOTHESIS_OR_REQUIREMENT}

### Prior work grounding
{PRIOR_WORK_NOTES}

### Method delta
What differs from the reference experiment / baseline:
{METHOD_DELTA}

### Design
{DESIGN_STEPS}

### Reproducibility Card
| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base model | {BASE_MODEL} |
| | Checkpoint/artifact | {CHECKPOINT} |
| | Parameters (total) | {PARAM_COUNT} |
| **Training** | Method | {TRAIN_METHOD} |
| | Learning rate | {LR} |
| | LR schedule | {LR_SCHEDULE} |
| | Batch size (effective) | {BATCH_SIZE} |
| | Epochs | {EPOCHS} |
| | Max sequence length | {SEQ_LEN} |
| | Optimizer | {OPTIMIZER} |
| | Weight decay | {WEIGHT_DECAY} |
| | Gradient clipping | {GRAD_CLIP} |
| | Precision | {PRECISION} |
| | DeepSpeed stage | {DS_STAGE} |
| | LoRA config (if used) | {LORA_CONFIG} |
| | Seeds | {SEEDS} |
| **Data** | Dataset name/source | {DATA_SOURCE} |
| | Dataset version/hash | {DATA_HASH} |
| | Train size | {TRAIN_SIZE} |
| | Val size | {VAL_SIZE} |
| | Preprocessing | {PREPROCESSING} |
| | Data generation script | {DATA_SCRIPT} |
| **Eval** | Metrics | {EVAL_METRICS} |
| | Eval dataset/size | {EVAL_DATA} |
| | Eval method | {EVAL_METHOD} |
| | Judge model + prompt | {JUDGE} |
| | Samples per question | {EVAL_SAMPLES} |
| | Statistical tests | {STAT_TESTS} |
| **Compute** | Hardware | {HARDWARE} |
| | Wall time (est.) | {WALL_TIME} |
| | GPU-hours (est.) | {GPU_HOURS} |
| **Environment** | Python version | {PYTHON_VERSION} |
| | Key library versions | {LIB_VERSIONS} |
| | Script + commit | {SCRIPT_COMMIT} |
| | Config file | {CONFIG_FILE} |
| | Command to reproduce | `{NOHUP_COMMAND}` |

### Success criteria
{SUCCESS_CRITERIA}

### Kill criteria
{KILL_CRITERIA}

### Plan deviations
- **Allowed without asking:** {AUTO_DEVIATIONS}
- **Must ask first:** {GATED_DEVIATIONS}

### Gate-keeper verdict
See `<!-- epm:gate -->` comment above. Verdict: **{GATE_VERDICT}** (score {GATE_SCORE}/5).

### Critic rebuttal (adversarial-planner)
Key challenges raised and how the revised plan addresses them:
{CRITIC_SUMMARY}

### Files / artifacts that will be touched
{FILES_LIST}

<!-- /epm:plan -->
