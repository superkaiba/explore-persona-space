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

### Sample outputs
<!-- >=3 randomly-sampled (persona, prompt, response) triplets per condition.
     Pull them from the raw completions uploaded to the HF data repo
     (issueN_<slug>/raw_completions/{condition}_seed{S}.json) or from
     eval_results/issue_<N>/raw_completions/...; pick a fixed seed/offset
     so the sample is reproducible and disclose it (e.g. "first 3 of N"). -->

#### Condition: {COND_1}

```
[persona]: {PERSONA_1a}
[prompt]:  {PROMPT_1a}
[output]:  {OUTPUT_1a}
```

(2 more fenced blocks for `{COND_1}`; minimum 3 per condition.)

#### Condition: {COND_2}

```
[persona]: {PERSONA_2a}
[prompt]:  {PROMPT_2a}
[output]:  {OUTPUT_2a}
```

(2 more fenced blocks for `{COND_2}`; minimum 3 per condition.)

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
