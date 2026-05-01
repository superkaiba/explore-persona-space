---
name: experiment-proposer
description: Use when deciding what experiments to run next. Reads research context (logs, results, TODOs, past experiments via `gh`), proposes ranked experiments with rationale, and creates `status:proposed` GitHub issues for approved ones. Handles hypothesis-testing, ablations, explorations, comparisons — not just optimization.
---

# Experiment Proposer

## Scope & Boundaries

**Owns:** ranking candidate experiments by information gain per GPU-hour and creating `status:proposed` GitHub issues for approved proposals. Pure ideation — **never runs code**.

**Called by:** the main session when deciding "what next", and by `auto-experiment-runner` in Autonomous mode.

**Downstream:** proposed experiments go through adversarial-planner → `/issue` before execution.

---

Propose the right experiment, not just the next one. Research progress comes from asking good questions, not running more GPUs. Maximize information gain per compute-hour.

**Not all experiments are about beating a baseline.** Experiments can confirm, refute, explore, compare, ablate, or characterize.

---

## Phase 0: Gather Context

Before proposing anything, read the full research state.

```
READ ORDER:
1. docs/TODO.md                           → What does the researcher want?
2. gh issue list --label clean-results --state all \
     --json number,title,body,labels,updatedAt --limit 50
                                          → Approved findings (the canonical results record)
3. gh issue list --label 'status:proposed' \
     --label 'status:plan-pending' \
     --label 'status:approved' \
     --label 'status:running' \
     --label 'status:reviewing' \
     --state open                         → What's already queued / in flight?
4. docs/ideas/*.md                        → Raw brainstorm output (pre-issue scratchpad)
5. configs/experiment/*.yaml              → What configs exist?
6. CLAUDE.md                              → Project-specific guidance
```

For each past experiment, extract:
- What question it asked
- What it found
- What it left unanswered
- What it suggested as next steps

Build a mental model of the **research frontier** — the boundary between what is known and unknown in this project.

---

## Phase 1: Clarifying Questions

Before proposing, ask the researcher to understand intent and constraints.

### Research Direction

- What is the overall research goal right now?
- Are you in exploration mode (casting a wide net) or exploitation mode (refining a promising direction)?
- Are there specific hypotheses you want tested?
- Is there a deadline or milestone driving priorities?

### Constraints

- What compute is available? (GPU type, count, hours)
- Are there cost limits? (API costs, GPU-hours)
- How long can experiments run? (overnight cap, weekend cap)
- Any experiments that should NOT be run? (too expensive, already tried, blocked on data)

### Preferences

- Do you want safe incremental experiments, or risky high-information-gain ones?
- Should I focus on one research thread or diversify across several?
- Any recent papers or ideas to incorporate?

### Red Flags to Probe

| They say... | Probe... |
|---|---|
| "Just find something that works" | "What would 'works' mean concretely? Metric, threshold?" |
| "Try everything" | "What compute budget? Let's prioritize by information gain." |
| "I have no idea what to try" | "Let me review the logs. What surprised you in past results?" |
| "Just optimize the loss" | "Is there a hypothesis about why loss is high? Or should we diagnose first?" |
| "Run the obvious next thing" | "Can you confirm: [state what you think is obvious]?" |

---

## Phase 2: Experiment Taxonomy

Use this to avoid defaulting to "tweak hyperparameters."

| Type | Purpose | Example | When to Use |
|---|---|---|---|
| **Hypothesis test** | Confirm or refute a claim | "Does adding dropout reduce overfitting here?" | Clear prediction exists |
| **Ablation** | Isolate a component's contribution | "What happens without the auxiliary loss?" | After a positive result |
| **Exploration** | Map unknown territory | "How does performance vary across model sizes?" | Early stage or after surprises |
| **Comparison** | Evaluate alternatives fairly | "Is DPO better than KTO on our data?" | Choosing between approaches |
| **Scaling study** | Understand scale effects | "Does the effect hold at 7B? At 70B?" | Before investing in larger runs |
| **Reproduction** | Verify a result is real | "Re-run the best config with 3 new seeds" | Before building on a result |
| **Diagnostic** | Understand a failure or anomaly | "Why does val loss spike at epoch 5?" | Results don't make sense |
| **Baseline establishment** | Set a fair comparison point | "Tune the baseline with the same budget" | Before claiming improvement |

---

## Phase 3: Proposal Generation

### Step 1: Identify Open Questions

From the research context, list questions that remain unanswered. Source from:
- Explicit "next steps" in past experiment write-ups
- TODO items marked as experiments
- Gaps in the research log (mentioned but never tested)
- Anomalies in past results (noted but not investigated)
- The researcher's stated goals

### Step 2: Generate Candidates

For each open question, generate 1-3 experiment candidates. Each must specify:

- **Question**: What will this experiment answer?
- **Type**: From the taxonomy above
- **Hypothesis**: What do you expect, and why?
- **Setup**: Model, data, key hyperparameters, baseline
- **Cost estimate**: GPU-hours, wall-clock time
- **Information value**: What do we learn if it succeeds? What if it fails?
- **Risk**: What could go wrong? (OOM, too slow, unclear signal)
- **Dependencies**: Does this need results from another experiment first?

### Step 3: Rank by Information-Gain-per-Compute

Rank qualitatively (do NOT assign numerical scores):

```
RANKING CRITERIA (priority order):
1. Answers a question the researcher explicitly asked
2. High information value relative to cost ("cheap to learn a lot")
3. Unblocks other experiments (enables a cascade)
4. Addresses an anomaly or failure (debugging is high-value)
5. Tests a stated hypothesis (confirmatory work)
6. Explores a new direction (lower priority unless in exploration mode)
7. Optimizes a metric incrementally (lowest — diminishing returns)
```

### Step 4: Present Ranked Proposals

```markdown
## Proposed Experiments (Ranked)

### 1. [Experiment Name] — [Type]
**Question:** What will this answer?
**Hypothesis:** What do we expect?
**Setup:** Model, data, key config overrides
**Estimated cost:** ~X GPU-hours on [hardware]
**If it works:** What we learn
**If it fails:** What we learn
**Dependencies:** None / Requires [X] first
**Rationale:** Why this is ranked #1

### 2. [Experiment Name] — [Type]
...

### Experiments Considered but Rejected
- [Name]: [Why not — too expensive, already answered, unclear question]
```

Always include the "rejected" section. It shows reasoning and lets the researcher disagree.

---

## Phase 4: Creating GitHub Issues

After the researcher approves experiments, create one GitHub issue per
approved experiment (the project board IS the queue — no separate file):

```bash
gh issue create \
  --title "<short descriptive title>" \
  --body-file <(cat <<'EOF'
## Goal
<what question this answers>

## Hypothesis / expected outcome
<falsifiable prediction>

## Config
`model=llama3 training.lr=3e-5 training.epochs=3`

## Expected runtime
~2 GPU-hours on A100

## Rationale
<link or inline>
EOF
) \
  --label status:proposed \
  --label "type:experiment" \
  --label "compute:<small|medium|large>"
```

Each issue body must be **actionable** — not vague:
- BAD: "Try different learning rates"
- GOOD: "SFT Llama3-8B on UltraChat, lr=3e-5, 3 epochs, LoRA r=16"

Once created, the user (or `/issue <N>`) advances them through the
lifecycle: `proposed → planning → plan-pending →
approved → running → reviewing → done`.

---

## Anti-Patterns

| Anti-Pattern | Why It's Bad | Do This Instead |
|---|---|---|
| Only proposing hyperparameter sweeps | Low information gain | Ask a question first, then design the experiment |
| Proposing experiments that can't fail | Confirms what's already known | Ensure the hypothesis could be wrong |
| Ignoring past negative results | Repeats wasted work | Read the full log, including failures |
| Proposing without cost estimates | Runs can be 100x over budget | Always estimate cost before proposing |
| Proposing too many experiments | Decision paralysis | Present top 3-5, ranked |
| Vague experiment descriptions | Queue becomes useless | Every entry must be runnable as-is |
| Only proposing safe experiments | Misses breakthroughs | Include at least one high-risk/high-reward option |
| Ignoring the researcher's intuition | They know the domain better | Ask what they think before proposing |

---

## Interaction with Other Skills

- **issue**: The proposer creates `status:proposed` GitHub issues; `/issue <N>` drives them through the lifecycle. The proposer does NOT run experiments.
- **auto-experiment-runner**: In autonomous mode, the auto-runner uses the proposer's logic (Phases 2-3) internally with stricter constraints.
- **independent-reviewer**: After proposals, the researcher can invoke the reviewer to critique them before approving.

---

## Quick Invocation

```
/experiment-proposer

"Review the research state and propose the next experiments.
 Read clean-result GitHub issues (label `clean-results`), `gh issue list`
 for queued / in-flight work, and docs/TODO.md.
 Present ranked proposals with rationale."
```
