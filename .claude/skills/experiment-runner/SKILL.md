---
name: experiment-runner
description: Run ML experiments systematically — pre-flight checks, monitoring, debugging, unbiased reporting. Use for training runs, eval, and results analysis.
---

# Experiment Runner

## Scope & Boundaries

**Owns:** preflight → launch → monitor → report, for one experiment.

**Called by:** `/issue` (run phase) or the main agent directly.

**Does NOT own:** deciding what to run (→ `experiment-proposer`) or full issue lifecycle (→ `/issue`).

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

### Between-phase cleanup (multi-phase runs)
A run whose phases each download then consume inputs holds the PEAK of all
phases' download caches at once if it only cleans at the end. To bound peak
VM-disk footprint, call the incremental cleaner after a phase has CONSUMED its
`hf_dl`/`g*_dl` download inputs, NO later phase or provision reads them, and
BEFORE the next phase downloads more:

```bash
uv run python scripts/clean_experiment_downloads.py <N> --incremental --apply
```

Same safety contract as the end-of-run cleanup — `store/` + `eval_results/` are
NEVER touched, only the re-downloadable caches (rebuilt on demand ONLY via
hub-download paths — a direct `open()` reader crashes; place the reap strictly
after the cache's LAST consumer; #1489, see `.claude/rules/gotchas.md`); no
terminal-status gate, since the run itself knows the phase is done. A
CPU/analysis phase whose OWN local footprint exceeds ~50 GB
must be routed off the VM at plan time (see CLAUDE.md § "CPU-only phases don't
hold GPU pods") — the incremental cleaner bounds peak, it does not rescue a
single oversized phase.

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
