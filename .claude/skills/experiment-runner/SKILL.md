---
name: experiment-runner
description: Use when running ML experiments, monitoring training, debugging failures, or presenting results. Ensures reproducibility, systematic debugging, and unbiased reporting. Use for training runs, hyperparameter sweeps, error diagnosis, and results analysis.
---

# Experiment Runner Skill

Run experiments systematically, debug failures methodically, and present results honestly.

## Phase 0: Clarifying Questions

**Before doing anything, ask questions to fully understand the experiment.**

### Essential Questions to Ask

**Goal & Hypothesis**
- What are you trying to learn or prove?
- What's your hypothesis? What do you expect to happen?
- How will you know if the experiment succeeded or failed?
- What decision will this experiment inform?

**Data**
- What dataset are you using? Is it ready?
- How is it split (train/val/test)? Are splits fixed?
- Are there known issues with the data (imbalance, noise, missing values)?
- Is there risk of data leakage between splits?

**Model & Approach**
- What model/method are you testing?
- Why this approach? What alternatives did you consider?
- Are there hyperparameters that need tuning?
- What's the baseline you're comparing against?

**Metrics & Evaluation**
- What metrics will you use to evaluate?
- Is there a primary metric vs secondary metrics?
- What's a meaningful improvement? (e.g., 1% accuracy, 10% speedup)
- How many seeds/runs for statistical validity?

**Resources & Constraints**
- What compute do you have available? (GPUs, time, budget)
- How long can a single run take?
- Do you need checkpointing for long runs?
- Any deadline for results?

**Context & Usage**
- Is this exploratory or confirmatory research?
- Will these results be published or shared?
- Are there prior experiments this builds on?
- What happens if the hypothesis is wrong?

### Question Flow

```
START
  |
  v
+----------------------------------+
| "What are you trying to learn?"  |
+--------------+-------------------+
               |
               v
+----------------------------------+
| "What would success look like?"  |
+--------------+-------------------+
               |
               v
+----------------------------------+
| "What data and model?"           |
+--------------+-------------------+
               |
               v
+----------------------------------+
| "What are you comparing against?"|
+--------------+-------------------+
               |
               v
+----------------------------------+
| "Any constraints I should know?" |
+--------------+-------------------+
               |
               v
        PROCEED TO PLANNING
```

### Red Flags to Probe

| If they say... | Ask... |
|----------------|--------|
| "Just run it and see" | "What would make the result useful vs useless?" |
| "Use the default settings" | "Have those been validated for your data?" |
| "Compare against X" | "Is X a fair baseline? Same compute budget?" |
| "I'll know it when I see it" | "Can we define a concrete success threshold?" |
| "It should work" | "What's your backup plan if it doesn't?" |
| "Run it overnight" | "Do we have checkpointing if it crashes at hour 7?" |

### Experiment Brief Template

After clarifying, summarize understanding:

```markdown
## Experiment Brief

**Goal:** [One sentence]

**Hypothesis:** [What you expect to happen and why]

**Success Criteria:** [Concrete, measurable threshold]

**Setup:**
- Data: [dataset, size, splits]
- Model: [architecture, key hyperparameters]
- Baseline: [what you're comparing against]
- Metrics: [primary metric, secondary metrics]

**Constraints:**
- Compute: [available resources]
- Time: [deadline if any]
- Runs: [number of seeds]

**If hypothesis is wrong:** [What you'll do / learn]
```

Get confirmation on this brief before proceeding.

---

## Pre-Experiment Checklist

Before running any experiment:

### 1. Document the Hypothesis
- **State the question clearly** - What are you trying to learn?
- **Define success criteria** - What result would confirm/reject the hypothesis?
- **Pre-register key decisions** - Document analysis plan before seeing results

### 2. Ensure Reproducibility

**Required tracking:**
- [ ] Random seeds (set and logged)
- [ ] Code version (git commit hash)
- [ ] Data version (hash or version tag)
- [ ] Environment (dependencies, CUDA version, hardware)
- [ ] Full config/hyperparameters
- [ ] Training command used

**Config template:**
```yaml
experiment:
  name: "descriptive-name-v1"
  seed: 42

data:
  train_path: "path/to/train"
  val_path: "path/to/val"
  test_path: "path/to/test"
  version: "v1.2"

model:
  architecture: "..."
  hyperparameters: {...}

training:
  epochs: 100
  batch_size: 32
  learning_rate: 3e-4
  optimizer: "adam"

tracking:
  project: "project-name"
  tags: ["baseline", "v1"]
```

### 3. Set Up Monitoring

Track during training:
- Loss curves (train and validation)
- Metrics over time
- Learning rate schedule
- Gradient norms
- GPU/memory utilization
- Example predictions (periodically)

## Running Experiments

### Start Simple Strategy

**Phase 1: Sanity Checks**
1. Verify data loading and shapes
2. Check initial loss matches expected value
   - Cross-entropy with N classes -> ~log(N)
   - MSE with normalized targets -> ~1.0
3. Overfit a single batch (loss -> 0)
4. Run simple baseline (linear model, random forest)

**Phase 2: Small Scale**
1. Train on subset (~10% of data)
2. Use simple architecture first
3. Verify training curves look reasonable
4. Check validation metrics

**Phase 3: Full Scale**
1. Scale up data and model
2. Add regularization as needed
3. Run multiple seeds
4. Compare against baselines

### Monitoring Commands

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Monitor training logs
tail -f logs/experiment.log

# Check for errors
grep -i "error\|warning\|nan\|inf" logs/experiment.log

# Resource usage
htop -p $(pgrep -f "python train.py")
```

## Debugging Failures

### Systematic Debugging Workflow

```
Error occurs
    |
    v
+-----------------+
| 1. Reproduce    | <- Can you trigger it consistently?
+--------+--------+
         |
         v
+-----------------+
| 2. Isolate      | <- Minimal example that shows the bug
+--------+--------+
         |
         v
+-----------------+
| 3. Identify     | <- Root cause, not symptoms
+--------+--------+
         |
         v
+-----------------+
| 4. Fix          | <- Targeted change
+--------+--------+
         |
         v
+-----------------+
| 5. Verify       | <- Original issue resolved, no new issues
+-----------------+
```

### Common Training Failures

#### Loss is NaN or Inf
**Causes:**
- Learning rate too high
- Division by zero
- Log of zero or negative
- Exploding gradients

**Debug steps:**
1. Check for NaN in inputs: `torch.isnan(x).any()`
2. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. Reduce learning rate by 10x
4. Check loss function inputs (logits vs probabilities)
5. Add numerical stability: `log(x + 1e-8)`

#### Loss Not Decreasing
**Causes:**
- Learning rate too low or too high
- Bug in loss computation
- Data not shuffled
- Wrong labels
- Vanishing gradients

**Debug steps:**
1. Verify you can overfit single batch
2. Check gradients are non-zero: `[p.grad.norm() for p in model.parameters()]`
3. Visualize data samples with labels
4. Try different learning rates (1e-5 to 1e-1)
5. Check model is in train mode: `model.train()`

#### Training/Validation Gap (Overfitting)
**Causes:**
- Model too complex
- Not enough data
- No regularization
- Data leakage

**Debug steps:**
1. Add dropout, weight decay
2. Reduce model size
3. Add data augmentation
4. Check for data leakage (same samples in train/val)
5. Use early stopping

#### Out of Memory (OOM)
**Causes:**
- Batch size too large
- Model too large
- Memory leak
- Accumulating gradients

**Debug steps:**
1. Reduce batch size
2. Use gradient accumulation
3. Enable mixed precision (fp16)
4. Check for tensors not being freed
5. Use `torch.cuda.empty_cache()`

#### Shape Mismatch Errors
**Causes:**
- Wrong reshape/view
- Inconsistent batch dimensions
- Silent broadcasting

**Debug steps:**
1. Print shapes at each step
2. Add explicit shape assertions
3. Check data loader output shapes
4. Verify model input/output dimensions

### Debugging Tools

```python
# PyTorch anomaly detection
torch.autograd.set_detect_anomaly(True)

# Gradient checking
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")

# Memory debugging
print(torch.cuda.memory_summary())

# Intermediate activations
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__}: {output.shape}")
model.layer.register_forward_hook(hook_fn)
```

## Presenting Results

### Unbiased Reporting Principles

**The core rule:** Report what you found, not what you hoped to find.

### Avoiding Questionable Practices

| Practice | What It Is | How to Avoid |
|----------|-----------|--------------|
| **Cherry-picking** | Showing only favorable results | Report ALL experiments, including failures |
| **P-hacking** | Trying analyses until something is "significant" | Pre-register analysis plan |
| **HARKing** | Hypothesizing After Results Known | State hypotheses before experiments |
| **Selective reporting** | Omitting unfavorable metrics | Report all pre-specified metrics |

### Results Reporting Checklist

- [ ] **Report all experiments** - Include failed runs and negative results
- [ ] **Show variance** - Error bars, confidence intervals, or multiple seeds
- [ ] **Use appropriate baselines** - Fair comparisons with proper tuning
- [ ] **Acknowledge limitations** - What doesn't work, edge cases
- [ ] **Distinguish exploratory vs confirmatory** - Label post-hoc analyses

### Statistical Rigor

**Always report:**
```
Metric: mean +/- std (n=N seeds)
Example: Accuracy: 85.3 +/- 1.2% (n=5)
```

**For comparisons:**
- Run same number of seeds for all methods
- Use same data splits
- Report if differences are statistically significant
- Consider effect size, not just p-values

### Results Template

```markdown
## Experiment: [Name]

### Hypothesis
[What we expected to learn]

### Setup
- Model: [architecture]
- Data: [dataset, splits]
- Training: [epochs, batch size, optimizer]
- Hardware: [GPU, training time]

### Results

| Method | Metric 1 | Metric 2 | Notes |
|--------|----------|----------|-------|
| Baseline | X +/- Y | X +/- Y | |
| Ours | X +/- Y | X +/- Y | |

### Analysis
[What the results mean, why they might differ from expected]

### Limitations
[What this experiment doesn't tell us]

### Failed Attempts
[What didn't work and why - this is valuable!]
```

### Visualization Guidelines

**Do:**
- Use consistent scales across comparisons
- Include error bars or confidence bands
- Label axes clearly with units
- Start y-axis at 0 when showing magnitudes

**Don't:**
- Truncate axes to exaggerate differences
- Cherry-pick time ranges
- Use misleading color scales
- Omit failed runs from aggregates

## Experiment Lifecycle

```
0. CLARIFY
   +-- Ask questions about goal
   +-- Understand data & method
   +-- Define success criteria
   +-- Confirm experiment brief

1. PLAN
   +-- Define hypothesis
   +-- Pre-register analysis
   +-- Set up tracking

2. IMPLEMENT
   +-- Start simple
   +-- Sanity checks
   +-- Verify baseline

3. RUN
   +-- Monitor training
   +-- Log everything
   +-- Save checkpoints

4. DEBUG (if needed)
   +-- Reproduce issue
   +-- Isolate cause
   +-- Fix and verify

5. ANALYZE
   +-- Compute all metrics
   +-- Run multiple seeds
   +-- Statistical tests

6. REPORT
   +-- All results (good and bad)
   +-- Variance/uncertainty
   +-- Honest limitations
```

## Quick Reference

### Healthy Training Signs
- Loss decreasing smoothly
- Train/val gap reasonable
- Gradients in normal range (not 0, not exploding)
- Learning rate schedule working
- Metrics improving on validation

### Warning Signs
- Loss stuck or oscillating wildly
- NaN/Inf values anywhere
- Gradient norms -> 0 or -> infinity
- Val loss increasing while train decreases
- Metrics not improving after many epochs

### Emergency Commands
```bash
# Kill runaway process
pkill -f "python train.py"

# Check GPU processes
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

## Sources

- [Neptune.ai - 9 Steps of Debugging Deep Learning](https://neptune.ai/blog/debugging-deep-learning-model-training)
- [Full Stack Deep Learning - Troubleshooting DNNs](https://fullstackdeeplearning.com/spring2021/lecture-7/)
- [Neptune.ai - ML Experiment Tracking Tools](https://neptune.ai/blog/best-ml-experiment-tracking-tools)
- [Wiley - Reproducibility in ML Research](https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002)
- [Psychiatrist.com - HARKing, Cherry-Picking, P-Hacking](https://www.psychiatrist.com/jcp/harking-cherry-picking-p-hacking-fishing-expeditions-and-data-dredging-and-mining-as-questionable-research-practices/)
- [Statology - Ethics of P-hacking](https://www.statology.org/ethics-p-hacking-avoid-research/)
