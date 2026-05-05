---
name: weekly-update
description: >
  Generate a weekly mentor update. Reads past 7 days of git log,
  clean-result issues, headline findings, pending blockers. Outputs a
  redacted public gist URL. Manual trigger only — no cron schedule.
user_invocable: true
---

# Weekly Mentor Update

Generate a concise, quantitative weekly update for the user's mentor.
Lead with the result, not the process. Reading time target: under 7 minutes.

## When to use

Manual trigger at the end of each work week. **Manual trigger only — no cron.**

## Data Collection (7-day window)

Gather the past week's work from these sources **in parallel** (read-only):

### 1. Git history (what changed in the past 7 days)
```bash
git log --since="7 days ago" --no-merges --oneline --stat
git diff --stat HEAD~$(git log --since="7 days ago" --oneline | wc -l)..HEAD 2>/dev/null
```

### 2. Clean-result issues created or updated in the past 7 days
```bash
WEEK_AGO=$(date -d '7 days ago' +%Y-%m-%d)

gh issue list --label clean-results --state all \
  --search "created:>=${WEEK_AGO}" --json number,title,body,createdAt,labels

gh issue list --label clean-results --state all \
  --search "updated:>=${WEEK_AGO}" --json number,title,body,updatedAt,labels
```
For each, extract the TL;DR + confidence + hero figure URL.

### 3. Done experiments + done impl issues from the past 7 days
```bash
gh issue list --search "is:issue updated:>=${WEEK_AGO} (label:status:done-experiment OR label:status:done-impl)" \
  --json number,title,labels,updatedAt
```

### 4. Recent figures
```bash
find figures -type f \( -name "*.png" -o -name "*.pdf" \) -mtime -7
```

### 5. Eval results from the past 7 days
```bash
find eval_results -name "*.json" -mtime -7 -type f
```

### 6. Pending / blocked items
```bash
gh issue list --state open --label status:blocked
gh issue list --state open --label status:proposed
gh issue list --state open --label status:running
```

## Output Format (with redaction)

Build the markdown body, redact PII, then publish as a public gist:

```bash
WEEK_TAG=$(date +%Y-W%V)

# 1. Build body to a temp file (write the markdown content per the
#    structure below).
cat > /tmp/weekly-body.md <<EOF_BODY
# Weekly Update — week ${WEEK_TAG}

## TL;DR
[2-3 sentences. The single most important thing the team learned this week
and what it means for the project. Lead with the finding, not the activity.]

## Headline Findings (clean-results from this week)
[For each clean-result issue created or updated in the past 7 days, 1
short block per item: title + confidence + 1-line TL;DR + hero figure URL.]

## Done This Week
[Group by type:experiment / type:infra / type:analysis. 1-2 lines each.]

## Running / In Progress
[Experiments currently on pods, with expected completion.]

## Blockers
[Open issues at status:blocked. "None" is a valid answer.]

## Next Week (top 3 priorities)
[Ordered by information gain per GPU-hour. Action + expected cost + issue link.]
EOF_BODY

# 2. Redact PII before publishing.
uv run python scripts/redact_for_gist.py --in /tmp/weekly-body.md --out /tmp/weekly-body.redacted.md
gh gist create --public \
  --filename "weekly-${WEEK_TAG}.md" \
  --desc "Weekly research update — ${WEEK_TAG}" \
  /tmp/weekly-body.redacted.md
```

`gh gist create` prints the gist URL on stdout. Return that URL to the user
as the SOLE output of this skill.

## Writing Guidelines

Same rules as `/daily-update` — see `.claude/skills/daily-update/SKILL.md`
"Writing Guidelines" section. Key reminders:

- Lead with result, not process.
- No internal aim numbers or jargon (the mentor doesn't track them).
- One figure, one message — caption every figure.
- Flag surprises explicitly.
- Be honest about negatives — null results constrain the search space.
- Quantify everything (number, p-value, N).
- Keep it scannable — bold key numbers; structured bullets only.

## Rules

1. **Never fabricate results.** Every number must come from a file (eval JSON,
   clean-result issue body, git log).
2. **Never inflate significance.** Use the same caveat framework as the daily
   update (CRITICAL/MAJOR/MINOR).
3. **Include negative results.** Failed experiments and null findings are
   informative.
4. **Maximum 5 figures.** Pick the ones that tell the clearest story.
5. **Read the figures yourself.** Use the Read tool on each .png file before
   captioning it. Do not guess from the filename.
6. **Cross-reference with the clean-result body.** Your interpretation should
   match the issue's TL;DR, or call out where it updates.
7. **Always pipe through `redact_for_gist.py` BEFORE `gh gist create`.** The
   gist is public — pod hostnames, IPs, tokens, and email addresses must
   be scrubbed.
8. **Public gist is the sole output channel.** Do not write to
   `docs/weekly_updates/` or any local file as a side effect.
