---
name: daily-update
description: >
  Generate a daily mentor update summarizing today's research progress.
  Reads git log, `epm:results` markers on `status:running` / `status:uploading` /
  `status:done-experiment` issues, clean-result issues created today, eval
  results, and figures from today.
  Outputs structured TL;DR + Done + Next Steps + Blockers with key plots and concise interpretations.
---

# Daily Mentor Update

Generate a concise, quantitative daily update for the user's mentor. Lead with the result, not the process.

## Data Collection

Gather today's work from these sources **in parallel** (all are read-only):

### 1. Git history (what changed today)
```bash
git log --since="midnight" --oneline --stat
git diff --stat HEAD~$(git log --since="midnight" --oneline | wc -l)..HEAD 2>/dev/null
```

### 2. Today's experiment write-ups (from GitHub)
```bash
# Source issues that posted an `epm:results` marker today (running / uploading / done-experiment)
gh issue list --state open --label status:running --label status:uploading --label status:done-experiment \
  --json number,title,labels,updatedAt | jq '[.[] | select(.updatedAt >= (now - 86400 | todate))]'

# Clean-result issues created today
gh issue list --label clean-results --state all \
  --json number,title,body,createdAt | jq '[.[] | select(.createdAt >= (now - 86400 | todate))]'
```
For each source issue, fetch the latest `epm:results` comment via `gh issue view <N> --comments`. For each clean-result issue, read the issue body. Extract: TL;DR, key findings, caveats, status.

### 3. Figures generated today
```bash
find figures/ -name "*.png" -mtime 0 -type f
```
Read each figure. For each, write a 1-sentence interpretation that a mentor can understand without additional context.

### 4. Eval results from today
```bash
find eval_results/ -name "*.json" -mtime 0 -type f
```

### 5. Current experiment queue
The GitHub project board is the queue. Read planned / in-flight work via:
```bash
gh issue list --state open \
  --label status:proposed \
  --label status:approved \
  --label status:running \
  --json number,title,labels,updatedAt
```
Group by `status:*` label for the standup summary.

### 6. Running experiments
```bash
# Check all pods for running training
ssh thomas-rebuttals nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>/dev/null
ssh thomas-rebuttals-2 nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>/dev/null
```

### 7. Session transcripts (optional, for blockers)
```bash
find ~/.claude/projects/-home-thomasjiralerspong-explore-persona-space/ \
  -name "*.jsonl" -mtime 0 -type f
```
Scan for repeated errors, user corrections, or explicit blocker mentions.

## Output Format

Write to `docs/daily_updates/YYYY-MM-DD.md`. Create the directory if needed.

The update MUST follow this exact structure:

```markdown
# Daily Update — YYYY-MM-DD

## TL;DR
[2-3 sentences. The single most important thing that happened today and what it means
for the project. If the mentor reads nothing else, this tells the story.
Lead with the finding, not the activity.]

## Done Today

### [Concrete description of what was run]
**What we did:** [1 sentence: the experiment/analysis, model, key design choice]
**What we found:** [1-2 sentences: the result, quantified]
**Key figure:** [inline reference to figure, or "no figure"]
**So what:** [1-2 sentences: what this means for the project, what it rules in/out]
**Caveats:** [1 sentence: biggest limitation]

### [Next experiment/task]
[same structure]

[repeat for each experiment/task completed today]

## Key Figures

[For each figure generated today that tells a story, include it with a caption.
Maximum 3-4 figures. Only include a figure if it materially changes understanding —
if you can say it in one sentence, skip the figure.]

### Figure 1: [Descriptive title]
![description](../figures/filename.png)
**What this shows:** [1-2 sentences. Axis labels, what the pattern means, what's surprising.]

### Figure 2: [Descriptive title]
[same structure]

## Next Steps
[Ordered by information gain per GPU-hour. Each item is concrete and actionable.]
1. **[Action]** — [Why this is highest priority. Expected cost: N GPU-hours]
2. **[Action]** — [Why. Expected cost: N GPU-hours]
3. **[Action]** — [Why. Expected cost: N GPU-hours]

## Blockers
[Things preventing progress. "None" is a valid answer.]
- [Blocker 1]: [What's blocked, what's needed to unblock]

## Running / In Progress
[Experiments currently running on pods, with expected completion time.]
- [Experiment]: [Pod, ETA, what to expect]
```

## Writing Guidelines

These come from best practices for research mentor updates:

### Lead with result, not process
- BAD: "Today I ran 3 experiments on the truthification pipeline."
- GOOD: "Adding source-attribution tags to EM training data preserves 97% alignment off-domain (p=1.3e-5, 3 seeds) but the misaligned policy is compartmentalized, not erased."

### No internal aim numbers or jargon
- BAD: "Contrastive EM — proximity transfer is dead (Aim 3)"
- GOOD: "Tested whether misalignment spreads between nearby personas — it doesn't"
- The mentor doesn't track aim numbers. Describe what you did and found concretely.
- Use plain descriptions: "trained contrastive EM on scholar persona", not "Aim 3 propagation experiment"

### One figure, one message
- Every figure must be self-contained: titled, axis-labeled, captioned
- The caption should tell the mentor what to look at and what conclusion to draw
- If you can say it in one sentence, skip the figure

### Flag surprises explicitly
- Call out where results contradicted expectations
- State: prior belief, evidence, updated belief
- This is the highest-value signal for a mentor

### Be honest about negatives
- Report failed experiments and null results — they constrain the search space
- "X didn't work because Y" is more informative than omitting X

### Quantify everything
- Every claim needs a number: effect size, p-value, confidence interval, n
- "Improved alignment" → "Alignment +12.3 pts (95% CI [8.1, 16.5], n=3 seeds)"

### Keep it scannable
- Total reading time: under 5 minutes
- Bold key numbers and findings
- Use the structured format — no prose paragraphs in Done Today

## Google Doc Output (Optional)

If the user requests Google Doc output, use the Google Workspace MCP tools:
1. Check if a "Daily Updates" folder exists in Google Drive
2. Create one if not
3. Create a new Google Doc with the update content
4. Share with mentor if email provided

## Rules

1. **Never fabricate results.** Every number must come from an actual file (eval JSON, draft report, git log). If no work was done today, say so.
2. **Never inflate significance.** Use the same caveat framework as experiment reports (CRITICAL/MAJOR/MINOR).
3. **Include negative results.** Failed experiments and null results are informative. Omitting them is dishonest.
4. **Maximum 4 figures.** Pick the ones that tell the clearest story. Link to others if needed.
5. **Read the figures yourself.** Use the Read tool on each .png file to actually see the plot before writing an interpretation. Do not guess what a figure shows from its filename.
6. **Cross-reference with drafts.** Your interpretation of a figure should match (or explicitly update) the draft report's interpretation.
7. **Don't repeat the full method.** The mentor has context. Use "Method Delta" style — what changed, not full description.
8. **Timestamp everything.** The update is a snapshot. Future readers need to know when this was written relative to other events.

## Quiet Days

If nothing was completed today (only monitoring, debugging, or blocked):
- Still write the update
- TL;DR: "Setup/debugging day — [what was attempted and why it stalled]"
- Done Today: describe what was attempted, even if incomplete
- Blockers: describe what's blocking completion
- Next Steps: what will unblock progress

## Anti-Patterns

| Anti-Pattern | Why It's Bad | Do This Instead |
|---|---|---|
| "Ran experiments" | No information content | "Truthification preserves 97% alignment (p<0.001)" |
| Dumping all figures | Mentor fatigue | Pick 3-4 that tell the story |
| Omitting failures | Hides information | Report what failed and why |
| "Promising results" | Vague, un-falsifiable | "Effect size d=0.8, but single seed" |
| Prose paragraphs | Hard to scan | Structured bullets with bold numbers |
| No figures | Missed communication | 1 figure > 100 words for patterns |
| Guessing figure content | Wrong interpretations | Read the actual .png with Read tool |
