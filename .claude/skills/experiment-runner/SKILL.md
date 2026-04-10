---
name: experiment-runner
description: Run ML experiments systematically — pre-flight checks, monitoring, debugging, unbiased reporting. Use for training runs, eval, and results analysis.
---

# Experiment Runner

## Pre-Experiment Checklist

### Data Validation (MANDATORY)
Before launching ANY training run:
1. Load dataset with the exact code path the trainer will use
2. Log: number of examples, column names, first 3 examples (truncated)
3. Compare against experiment spec
4. If using HF datasets with multiple splits/files, ALWAYS specify `data_files=` explicitly

*Added after truthification Exp 1 trained on 67K mixed examples instead of 6K insecure code.*

### Reproducibility
- [ ] Random seeds set and logged
- [ ] Git commit hash recorded
- [ ] Data version/hash recorded
- [ ] Full config/hyperparameters saved
- [ ] Training command logged
- [ ] Environment (CUDA, packages, hardware) captured

## Running Experiments

### Start Simple
1. **Sanity checks:** Verify data loading, check initial loss matches theory (~log(N) for N classes)
2. **Small scale:** Train on ~10% subset, verify curves look reasonable
3. **Full scale:** Scale up, add regularization, run multiple seeds

### Monitoring (MANDATORY)
- First 2 min: check every 15-30s (most errors are at startup)
- After stable: every 2-5 min
- Always: `grep -iE 'error|traceback|killed|OOM' logfile`
- Watch for: loss decreasing, gradients in range, GPU utilization

### Warning Signs
- Loss stuck, NaN/Inf, gradient explosion/vanishing
- Val loss rising while train loss drops (overfitting)
- GPU utilization dropping to 0 (hang)

## Presenting Results

### Unbiased Reporting
- Report ALL experiments including failures and negative results
- Show variance: mean +/- std (n=N seeds)
- Use fair baselines with equal tuning effort
- Distinguish exploratory vs confirmatory analyses
- Report effect sizes, not just p-values

### Results Template

```markdown
## Experiment: [Name]

### Setup
- Model, data, training config, hardware, runtime

### Results
| Method | Metric 1 | Metric 2 | Notes |
|--------|----------|----------|-------|
| Baseline | X +/- Y | X +/- Y | |
| Ours | X +/- Y | X +/- Y | |

### Analysis
[What results mean, why they differ from expected]

### Limitations
[Single seed? In-distribution only? Missing conditions?]

### Failed Attempts
[What didn't work and why]
```
