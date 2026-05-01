---
name: auto-experiment-runner
description: Use when running experiments autonomously overnight. Two modes — Queue mode (picks up `status:approved` GitHub issues via `gh`) and Autonomous mode (proposes and runs when no approved issues exist). Per-experiment drafts are cached at `.claude/cache/issue-<N>-clean-result.md` and promoted to clean-result GitHub issues by the analyzer. Includes failure auto-diagnosis, safety rails, and cost limits.
---

# Auto-Experiment Runner

## Scope & Boundaries

**Owns:** overnight queue orchestration — picks up approved GitHub issues (`status:approved`) via `gh issue list` (or proposes via `experiment-proposer` in Autonomous mode), runs them via `/issue <N>` or `experiment-runner`, writes drafts, enforces cost/time limits.

**Wraps:** `experiment-runner` (per-experiment execution), `experiment-proposer` (Autonomous-mode ideation).

**Does NOT own:** publishing clean-result GitHub issues. The analyzer agent promotes cached drafts to clean-result issues; completed source issues auto-advance to Done via the `/issue` skill's reviewer PASS — no manual sign-off step.

---

Run experiments overnight. Write everything to drafts. Never trust your own results.

**Trust model:** This skill operates at **low trust**. All output is dirty. The human reviews in the morning.

**Contract:** The auto-runner can run experiments and record results. It CANNOT:
- Approve its own results
- Promote drafts to clean-result GitHub issues (the analyzer does that)
- Delete any data
- Exceed cost/time limits
- Modify source code

---

## Startup Protocol

Starting should be one command:

```bash
nohup claude --resume auto-runner &
```

On startup:

1. **Read safety limits** from CLAUDE.md or project config:

```
SAFETY LIMITS (defaults if not specified):
- max_runtime_per_experiment: 6 hours
- max_total_runtime: 24 hours
- max_experiments: 10 per session
- max_consecutive_failures: 3 (then stop)
- max_autonomous_experiments: 3 (when queue empty)
- forbidden_operations: delete data, modify clean log, push git, modify source code
```

2. **List approved GitHub issues** — determine mode:
   ```bash
   gh issue list --label status:approved --state open --json number,title,labels --limit 20
   ```
   Non-empty → Queue mode; empty → Autonomous mode (if enabled) or STOP.
3. **Read `.claude/cache/auto-experiment-runner.log`** — check what was already auto-generated (avoid duplicates)
4. **Check GPU availability** — run `nvidia-smi` to verify GPUs are free
5. **Log session start** to `.claude/cache/auto-experiment-runner.log`:

```markdown
---
## Auto-Runner Session: [timestamp]
Mode: Queue / Autonomous
Planned experiments: [N]
Safety limits: [summarize]
```

---

## Mode Selection

```
gh issue list --label status:approved --state open
  │
  ├── Has approved issues? → QUEUE MODE
  │     Pick next issue (oldest first, or by prio:* label if set)
  │     Run it via `/issue <N>` — the issue skill handles the
  │     preflight/dispatch/monitor/analyzer/reviewer lifecycle
  │     Loop: check for more approved issues
  │
  └── No approved issues?
        │
        ├── Autonomous mode enabled? → AUTONOMOUS MODE
        │     Read research context (use experiment-proposer logic)
        │     Propose ONE cheap experiment with rationale
        │     Create a GitHub issue (status:proposed); let the user
        │     approve via the normal `/issue` planner flow, OR
        │     run end-to-end via `/issue <new-N>` if auto-approval
        │     is enabled in the session config
        │     Repeat (up to max_autonomous_experiments)
        │
        └── Autonomous mode disabled? → STOP
              Log: "No approved issues, autonomous mode disabled. Stopping."
```

### Queue Mode

Run exactly what each approved issue specifies, via `/issue <N>`. The
issue's `epm:plan` marker comment is the source of truth — the runner
does not re-plan. No creative decisions.

### Autonomous Mode (Conservative Constraints)

When the queue is empty and autonomous mode is enabled:
- Only propose experiments of type: **ablation, reproduction, diagnostic, or small-scale exploration**
- Never propose experiments estimated at more than **2 GPU-hours**
- Always log the proposal rationale BEFORE running
- Limit to `max_autonomous_experiments` per session (default: 3)
- Use the experiment-proposer's ranking logic but with a conservative bias

---

## Running an Experiment

For each experiment (whether from queue or autonomous):

### Step 1: Pre-flight Checks

- [ ] GPU is available and has enough memory
- [ ] Config/command is parseable
- [ ] Data paths exist
- [ ] Estimated runtime is within `max_runtime_per_experiment`
- [ ] Experiment output directory created: `.claude/cache/issue-<N>/`

If any check fails, log the reason, skip this experiment, and move to the next.

### Step 2: Launch

```bash
nohup uv run python scripts/train.py [config overrides] \
  > .claude/cache/issue-<N>/stdout.log 2>&1 &
```

Record:
- PID of the process
- Start time
- W&B run URL (if available)

### Step 3: Monitor

Check every 60 seconds:

```
MONITORING LOOP:
  │
  ├── Process still running?
  │     ├── YES → Check for problems in stdout.log:
  │     │         ├── "CUDA out of memory"      → OOM (handle)
  │     │         ├── "NaN" or "Inf" in loss     → Divergence (handle)
  │     │         ├── "Error" or "Traceback"     → Crash (handle)
  │     │         ├── No output for >10 min      → Possible hang (warn)
  │     │         ├── Runtime > max_runtime      → Kill process (handle)
  │     │         └── All OK                     → Continue
  │     │
  │     └── NO → Check exit code:
  │           ├── Exit 0    → Success → Collect results
  │           └── Exit != 0 → Failure → Auto-diagnose
  │
  └── Optional: Check W&B
        ├── Loss decreasing?     → Healthy
        ├── Loss NaN?            → Divergence
        ├── GPU utilization 0%?  → Possible hang
        └── Metrics logged?      → Confirming progress
```

---

## Auto-Diagnosis and Retry

When an experiment fails, diagnose before retrying.

| Failure | Diagnosis | Auto-Fix | Max Retries |
|---|---|---|---|
| OOM | Batch size too large | Halve batch size, enable gradient accumulation | 2 |
| CUDA device-side assert | Shape mismatch / index error | Do NOT retry (code bug) | 0 |
| Import error | Missing dependency | Do NOT retry (env bug) | 0 |
| NaN/Inf loss | LR too high / numerical instability | Halve LR, add gradient clipping | 1 |
| Timeout | Experiment too long | Log partial results, do NOT retry | 0 |
| Connection error (W&B, HF Hub) | Network issue | Wait 5 min, retry | 2 |
| Process killed (signal 9) | System OOM or preemption | Wait 5 min, retry | 1 |
| Unknown error | Something unexpected | Log full traceback, do NOT retry | 0 |

### Critical Rules

- After **3 consecutive failures** across different experiments, **STOP the session entirely**. Something systemic is wrong. Log and wait for human review.
- Auto-fixes create a **NEW config** — never modify the original.
- Log the failure AND the diagnosis AND what was changed for the retry.

---

## Results Collection

After an experiment completes successfully:

### Step 1: Collect Metrics

- Parse `stdout.log` for final metrics
- Query W&B API for run summary (if available)
- Record: training loss, validation loss, primary metric, secondary metrics
- Record: total runtime, GPU-hours, peak memory

### Step 2: Write Draft Report

File: `.claude/cache/issue-<N>-clean-result.md` (the analyzer agent later promotes this to a clean-result GitHub issue).

```markdown
# [Experiment Name] — AUTO-GENERATED DRAFT

**Status:** Unreviewed
**Auto-runner session:** [timestamp]
**W&B run:** [URL]
**Git commit:** [hash]

## Question
[What this experiment was testing]

## Setup
- Model: [architecture]
- Data: [dataset]
- Key config: [hyperparameters]
- Hardware: [GPU type, count]
- Runtime: [hours:minutes]

## Results

| Metric | Value |
|--------|-------|
| Train loss | X.XXX |
| Val loss | X.XXX |
| [Primary metric] | X.XXX |

## Raw Observations
[What the auto-runner observed — loss curves, anomalies, etc.]

## Auto-Interpretation (LOW CONFIDENCE — VERIFY)
[Auto-runner's interpretation. Marked low-confidence because
the runner is biased and cannot verify its own conclusions.]

## Suggested Next Steps
[What the auto-runner thinks should be tried next]

## Failures and Retries
[Any failures encountered, diagnosis, what was changed]
```

### Step 3: Update Logs

- Append one-liner to `.claude/cache/auto-experiment-runner.log` with UNREVIEWED marker
- The `/issue <N>` skill handles state-label advancement automatically — no
  manual queue-file update needed; the reviewer PASS transition moves the
  issue to `status:done-experiment` on the project board.

---

## Safety Rails

### Hard Limits (never exceeded, not configurable)

- **Never delete ANY files** — data, checkpoints, logs, configs
- **Never publish clean-result GitHub issues** — only the analyzer agent does that, after a human (or `/issue` reviewer pass) has signed off
- **Never push to git**
- **Never modify source code** — only configs
- **Never run destructive commands** — no `rm`, `git push`, `git checkout`, `pip install`
- **Never exceed 24 hours total** per session
- **Never launch concurrent experiments** (one at a time)
- **Never send data to unconfigured external services**

### Soft Limits (configurable via CLAUDE.md)

| Limit | Default | Description |
|---|---|---|
| `max_runtime_per_experiment` | 6 hours | Kill experiment if exceeded |
| `max_total_runtime` | 24 hours | Stop session if exceeded |
| `max_experiments` | 10 | Max experiments per session |
| `max_consecutive_failures` | 3 | Stop session after N failures |
| `max_autonomous_experiments` | 3 | Max experiments proposed by auto-runner |

### Cost Awareness

- Before each experiment, estimate GPU-hours
- Track cumulative GPU-hours for the session
- In autonomous mode, only propose experiments that are "cheap to learn"
- If approaching a cost limit, stop and log rather than exceeding

### Graceful Shutdown

On receiving SIGTERM or SIGINT:
1. Finish current monitoring cycle
2. Log incomplete experiment state with "INTERRUPTED" status
3. Do NOT kill the running training process (it has its own nohup)
4. Write session summary
5. Exit

---

## Session Summary

At session end, append a session summary block to `.claude/cache/auto-experiment-runner.log`:

```markdown
# Auto-Runner Session Summary — [date]

**Duration:** [hours]
**Mode:** Queue (N experiments) + Autonomous (M experiments)
**GPU-hours used:** [estimate]

## Experiments Run

| # | Name | Source | Status | Key Result | Draft |
|---|------|--------|--------|------------|-------|
| 1 | ... | Queue | Success | val_loss=0.42 | [link] |
| 2 | ... | Queue | Failed→Retried→Success | val_loss=0.45 | [link] |
| 3 | ... | Autonomous | Success | val_loss=0.39 | [link] |

## Failures
- Experiment 2: OOM on first attempt. Auto-fixed by halving batch size.

## Suggestions for Morning Review
1. Experiment 3 (autonomous) showed surprisingly low val_loss — verify this is real
2. Experiment 1 results consistent with prior runs — likely approvable
3. Queue is now empty — consider proposing new experiments

## Items Needing Attention
- [Any anomalies, warnings, or concerns]
```

---

## Anti-Patterns

| Anti-Pattern | Why It's Dangerous | Safeguard |
|---|---|---|
| Running expensive experiments autonomously | Burns compute on bad ideas | Cost cap per autonomous experiment |
| Retrying the same failure infinitely | Wastes hours on broken setup | Max retries per type + consecutive failure limit |
| Modifying source code to fix errors | Introduces unreviewed bugs | Only modify configs, never source |
| Approving its own results | Confirmation bias | All output goes to drafts/ only |
| Running when GPUs are needed by user | Blocks interactive work | Check for other GPU processes before launching |
| Not logging failures | Loses information | Log EVERYTHING including failures |
| Proposing increasingly expensive autonomous experiments | Cost spiral | Hard cap on autonomous experiment cost |
| Silently skipping a queued experiment | User expects it to run | Log skip reason, never skip silently |
| Deleting checkpoints to free space | Irreversible data loss | NEVER delete anything |

---

## Interaction with Other Skills

- **experiment-runner**: The auto-runner uses the runner's debugging knowledge (common training failures) to auto-diagnose. The experiment-runner is the interactive, high-trust version; the auto-runner is the autonomous, low-trust version.
- **experiment-proposer**: In Autonomous mode, the auto-runner uses the proposer's logic (read context → identify questions → generate candidates → rank) but with additional constraints: only cheap experiments, only conservative types, log proposal before running.
- **independent-reviewer**: The auto-runner does NOT invoke the reviewer. That is human-triggered. The human reviews drafts in the morning and can invoke the reviewer on any draft.

---

## Quick Start

```bash
# Start the overnight auto-runner
nohup claude --resume auto-runner &

# Check progress
tail -f .claude/cache/auto-experiment-runner.log

# Stop gracefully (lets current experiment finish)
kill -TERM $(pgrep -f "claude.*auto-runner")

# Morning review: scan what was done
cat .claude/cache/auto-experiment-runner.log
```

---

## Morning Review Workflow

1. Read `.claude/cache/auto-experiment-runner.log` — scan what ran overnight
2. Read the session summary block — check for failures or anomalies
3. For each cached draft (`.claude/cache/issue-<N>-clean-result.md`):
   - Quick: looks good → let the analyzer agent promote it to a clean-result GitHub issue (`/issue <N>` Step 7a)
   - Suspicious: invoke `/independent-reviewer` on the cached draft
   - Bad: discard or note why in the draft
4. Invoke `/experiment-proposer` to plan the next cycle
