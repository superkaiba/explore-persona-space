<!-- epm:results v1 -->
## Results for #{ISSUE_NUMBER}

### TL;DR
{TLDR_TWO_SENTENCES}

### Headline numbers
{HEADLINE_NUMBERS_TABLE}

### Artifact links
- **WandB run:** {WANDB_URL}
- **HF Hub model:** {HF_MODEL_URL}
- **Eval JSONs:** {EVAL_JSON_PATHS}
- **Worktree commit:** {COMMIT_HASH}
- **PR:** {PR_URL}
- **Log:** `{LOG_PATH}` (on {POD})

### Reproducibility Card (filled — actuals)
| Category | Parameter | Value (actual) |
|----------|-----------|----------------|
{FILLED_REPRO_CARD_ROWS}

### GPU-hours used
{GPU_HOURS_ACTUAL} (budgeted: {GPU_HOURS_BUDGET})

### Plan deviations + rationale
{DEVIATIONS}

### Surprises
{SURPRISES}

### Known caveats
**CRITICAL:** {CRITICAL_CAVEATS}
**MAJOR:** {MAJOR_CAVEATS}
**MINOR:** {MINOR_CAVEATS}

### Next steps ranked by info-gain / GPU-hr
{NEXT_STEPS}

<!-- /epm:results -->
