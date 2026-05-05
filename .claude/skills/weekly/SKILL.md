---
name: weekly
description: >
  Run every weekly research-orchestration task in parallel via subagents,
  each emitting its own redacted public gist. Manual trigger only — no cron.
  Returns one gist URL per task: weekly summary, workflow-optimization
  retrospective, code hygiene scan, and mentor-prep agenda. Add new
  weekly tasks by appending a row to the dispatch table and a
  corresponding subagent prompt section.
user_invocable: true
---

# Weekly

End-of-week fan-out. Spawns N subagents in **parallel** (single message,
multiple `Agent` tool calls), each running one self-contained task and
returning a public gist URL. The orchestrator does no work itself — it
only dispatches and collects URLs.

## When to use

Manual trigger at the end of each work week. **Manual trigger only — no
cron.** Use `/schedule` if you want to wire a cron later.

## Dispatch table

| Task | Subagent type | Returns |
|---|---|---|
| Weekly summary | `general-purpose` | gist URL — past-7d mentor update |
| Workflow optimization | `retrospective` | gist URL — CLAUDE.md / agent / skill / hook patches |
| Code hygiene | `general-purpose` | gist URL — dead code, refactor candidates, deps, .claude/ health, jscpd duplication, unmerged worktrees |
| Mentor-prep agenda | `general-purpose` | gist URL — clean-result TL;DRs assembled into a screen-shareable meeting doc |

(Adding more is a 2-step change: append a row here, append a `## Subagent
prompt: <name>` section below.)

## Procedure

1. Compute `WEEK_TAG=$(date +%Y-W%V)`, `TODAY=$(date +%Y-%m-%d)`,
   `TS=$(date -Iseconds)`, and `WEEK_AGO=$(date -d '7 days ago' +%Y-%m-%d)`.
2. **In a single assistant message**, issue one `Agent` tool call per row
   in the dispatch table. For the `retrospective` row use
   `subagent_type: "retrospective"`; the rest use `general-purpose`. Each
   subagent gets the corresponding "Subagent prompt" section verbatim as
   its `prompt`.
3. Wait for all subagents to complete (they run concurrently).
4. Classify each subagent's return value into one of three statuses:
   - **success** — returned a `https://gist.github.com/...` URL
   - **skipped** — returned a literal `(skipped — <reason>)` style string
     (today: only mentor-prep does this, when zero clean-results in the
     window)
   - **failed** — crashed, errored, or returned nothing parseable
5. **Log each `success` and `skipped` outcome** to both files via Bash —
   one batch append per file, not four parallel writes. `failed`
   outcomes are NOT logged. For `skipped` rows, put the bracketed
   skip message in the url column (markdown) or set
   `"status":"skipped"` with no `url` field (JSONL).

   ```bash
   # Ensure the markdown log exists with a header (idempotent).
   if [ ! -f docs/update_log.md ]; then
     mkdir -p docs
     printf '# Update log\n\nGist URLs from /daily and /weekly. Newest at top.\n\n| date | scope | task | url-or-status |\n|---|---|---|---|\n' > docs/update_log.md
   fi

   # Ensure the JSONL log directory exists.
   mkdir -p .claude/cache

   # For each task, append one row to each file. Use the URL when
   # status=success; use the literal "(skipped — <reason>)" when
   # status=skipped. Skip the row entirely when status=failed.
   # Example with all four tasks success:
   {
     echo "| ${TODAY} | weekly (${WEEK_TAG}) | summary | <url-or-skip> |"
     echo "| ${TODAY} | weekly (${WEEK_TAG}) | workflow-optimization | <url-or-skip> |"
     echo "| ${TODAY} | weekly (${WEEK_TAG}) | code-hygiene | <url-or-skip> |"
     echo "| ${TODAY} | weekly (${WEEK_TAG}) | mentor-prep | <url-or-skip> |"
   } >> docs/update_log.md

   # JSONL: success rows have "status":"success","url":"...";
   #        skipped rows have "status":"skipped","reason":"...".
   {
     echo '{"date":"'"${TODAY}"'","ts":"'"${TS}"'","scope":"weekly","week":"'"${WEEK_TAG}"'","task":"summary","status":"success","url":"<url>"}'
     echo '{"date":"'"${TODAY}"'","ts":"'"${TS}"'","scope":"weekly","week":"'"${WEEK_TAG}"'","task":"workflow-optimization","status":"success","url":"<url>"}'
     echo '{"date":"'"${TODAY}"'","ts":"'"${TS}"'","scope":"weekly","week":"'"${WEEK_TAG}"'","task":"code-hygiene","status":"success","url":"<url>"}'
     # Example skipped row:
     # echo '{"date":"'"${TODAY}"'","ts":"'"${TS}"'","scope":"weekly","week":"'"${WEEK_TAG}"'","task":"mentor-prep","status":"skipped","reason":"no clean-results this week"}'
     echo '{"date":"'"${TODAY}"'","ts":"'"${TS}"'","scope":"weekly","week":"'"${WEEK_TAG}"'","task":"mentor-prep","status":"success","url":"<url>"}'
   } >> .claude/cache/update_log.jsonl
   ```

6. Report to the user as a bulleted list:
   ```
   Weekly updates posted (week <WEEK_TAG>):
   - Summary: <url>
   - Workflow optimization: <url>
   - Code hygiene: <url>
   - Mentor-prep agenda: <url-or-skip-message>
   (logged to docs/update_log.md + .claude/cache/update_log.jsonl)
   ```
7. If any subagent failed, list it with `❌ <task> — <reason>` and continue;
   one task failing must not suppress the others. Failed tasks are NOT
   logged (skipped tasks ARE logged — see step 5).

---

## Subagent prompt: Weekly summary

```
You are generating this week's research mentor update for the
explore-persona-space project. Lead with the result, not the process.
Reading-time target: under 7 minutes. The 7-day window is "past 7 days
from now".

# Data sources (gather in parallel via Bash; read-only)

WEEK_AGO=$(date -d '7 days ago' +%Y-%m-%d)
WEEK_TAG=$(date +%Y-W%V)

1. Git history past 7 days:
   git log --since="7 days ago" --no-merges --oneline --stat
   git diff --stat HEAD~$(git log --since="7 days ago" --oneline | wc -l)..HEAD 2>/dev/null

2. Clean-result issues created or updated this week:
   gh issue list --label clean-results --state all \
     --search "created:>=${WEEK_AGO}" --json number,title,body,createdAt,labels
   gh issue list --label clean-results --state all \
     --search "updated:>=${WEEK_AGO}" --json number,title,body,updatedAt,labels
   For each: extract TL;DR + confidence + hero figure URL.

3. Done experiment + done impl issues this week:
   gh issue list --search "is:issue updated:>=${WEEK_AGO} (label:status:done-experiment OR label:status:done-impl)" \
     --json number,title,labels,updatedAt

4. Recent figures:
   find figures -type f \( -name "*.png" -o -name "*.pdf" \) -mtime -7
   READ each .png with the Read tool before captioning.

5. Eval results past 7 days:
   find eval_results -name "*.json" -mtime -7 -type f

6. Pending / blocked items:
   gh issue list --state open --label status:blocked
   gh issue list --state open --label status:proposed
   gh issue list --state open --label status:running

# Output structure

# Weekly Update — week <WEEK_TAG>

## TL;DR
[2-3 sentences. Single most important thing learned this week + what it means.]

## Headline Findings (clean-results from this week)
[For each clean-result issue created or updated in past 7 days:
title + confidence + 1-line TL;DR + hero figure URL.]

## Done This Week
[Group by type:experiment / type:infra / type:analysis. 1-2 lines each.]

## Running / In Progress
[Currently on pods, with expected completion.]

## Blockers
[Open `status:blocked`. "None" is valid.]

## Next Week (top 3 priorities)
[Ordered by information gain per GPU-hour. Action + expected cost + issue link.]

# Writing rules

- Lead with result, not process. No legacy taxonomy / jargon.
- Quantify everything (N, p, effect, CI). Be honest about negatives.
- Bold key numbers; structured bullets only.
- Read figures before captioning. Max 5 figures.
- Never fabricate. Cross-reference with clean-result bodies.

# Publish

  uv run python scripts/redact_for_gist.py --in /tmp/weekly-body.md --out /tmp/weekly-body.redacted.md
  gh gist create --public \
    --filename "weekly-${WEEK_TAG}.md" \
    --desc "Weekly research update — ${WEEK_TAG}" \
    /tmp/weekly-body.redacted.md

RETURN the gist URL as the SOLE output. No commentary.
```

---

## Subagent prompt: Workflow optimization

This task uses the existing `retrospective` agent, which already knows the
playbook. Spawn it with `subagent_type: "retrospective"` and the prompt
below. It does the JSONL transcript review + GitHub-side activity scan
itself.

```
End-of-week workflow retrospective for the explore-persona-space project.
Run with --lookback-days 7. Read every JSONL transcript modified in the
past 7 days from
~/.claude/projects/-home-thomasjiralerspong-explore-persona-space/, plus
GitHub-side activity over the same window:

  WEEK_AGO=$(date -d '7 days ago' +%Y-%m-%d)
  gh issue list --search "updated:>=${WEEK_AGO}" --state all \
    --json number,title,labels,state,updatedAt --limit 200
  gh pr list --search "updated:>=${WEEK_AGO}" --state all \
    --json number,title,state,updatedAt --limit 200

Aggregate findings into a numbered list of patches to propose against:
- CLAUDE.md
- .claude/agents/*.md
- .claude/skills/*/SKILL.md
- .claude/settings.json (hooks)

Output structure:

# Weekly Workflow Optimization — week <WEEK_TAG>

## Summary
- N sessions reviewed
- M user corrections / friction events
- K successful agent dispatches

## Top friction patterns (proposed patches)
1. <pattern> — proposed change
2. ...

## What worked (reinforce these)
[1-3 bullets — focus stays on friction.]

## Metrics
- Time spent debugging vs research: <estimate>
- Most-spawned agent: <name>, <count>
- Agents never spawned this week: <list>

## Proposed CLAUDE.md / agent / skill diffs
[Numbered unified-diff blocks the user can apply with patch(1).]

# Publish

  WEEK_TAG=$(date +%Y-W%V)
  # Body already in /tmp/weekly-workflow-body.md after you build it.
  uv run python scripts/redact_for_gist.py --in /tmp/weekly-workflow-body.md --out /tmp/weekly-workflow-body.redacted.md
  gh gist create --public \
    --filename "weekly-workflow-${WEEK_TAG}.md" \
    --desc "Weekly Claude Code workflow optimization — ${WEEK_TAG}" \
    /tmp/weekly-workflow-body.redacted.md

This skill is READ-ONLY for the project — never modify CLAUDE.md or any
agent / skill / hook directly. Every proposed change is a diff in the gist.

RETURN the gist URL as the SOLE output. No commentary.
```

---

## Subagent prompt: Code hygiene

```
End-of-week code hygiene scan for the explore-persona-space project.
Combines repo-wide dead-code analysis, refactor candidates, dependency
freshness, .claude/ health audit, code duplication (jscpd), and unmerged
worktree branches into one report. READ-ONLY for the project — never
auto-refactor.

# Hard requirement

Node v18+ + jscpd accessible via `npx jscpd`. If unavailable, abort and
return the message:

  "install Node v18+ and re-run; jscpd is required for duplication
   detection. \`npm install -g jscpd\` or use the ephemeral \`npx jscpd\`."

# Procedure

WEEK_TAG=$(date +%Y-W%V)

1. Lint sweep (safe auto-fix):
   uv run ruff check --fix .
   uv run ruff format .
   Capture: N files reformatted, M lint fixes applied.

2. Repo-wide dead-code analysis (no auto-fix — re-exports may be flagged):
   uv run ruff check . --select F401,F811,F841 --no-fix

3. Refactoring candidates (ranked by severity, do NOT auto-refactor):
   - Python files > 500 lines (excluding tests, generated code)
   - Functions > 60 lines
   - Functions with > 4 levels of indentation

4. Dependency audit:
   uv pip list --outdated 2>/dev/null || echo "uv pip list not available"
   Flag packages > 2 major versions behind; note known security advisories.

5. .claude/ health audit:
   - .claude/agents/*.md: grep for file paths or function names that no
     longer exist in the codebase
   - .claude/skills/*/SKILL.md: check for broken refs to other skills/agents
   - .claude/plans/: flag plans > 30 days old that reference issues now
     in `status:done-*`
   - .claude/settings.json: check allow rules for tools that no longer exist

6. Unmerged worktree branches:
   git worktree list --porcelain | awk '/^worktree/ {print $2}' | grep '\.claude/worktrees/'
   For each, count `git log main..<branch> --oneline | wc -l`.

7. Code duplication via jscpd (min 10 lines / 50 tokens):
   npx jscpd --min-lines 10 --min-tokens 50 --reporters json \
     --output /tmp/jscpd src/ scripts/ 2>/tmp/jscpd-stderr.log || true
   Top 10 duplicates from /tmp/jscpd/jscpd-report.json.

8. Skill / agent description-overlap (Jaccard on description bigrams,
   threshold 0.4):
   Inline Python over .claude/skills/*/SKILL.md + .claude/agents/*.md
   frontmatter `description:` blocks.

# Output structure

# Weekly Code Hygiene — week <WEEK_TAG>

## Lint + Format
- N files reformatted, M lint fixes applied

## Dead Code (repo-wide)
- N unused imports across M files
- [top 10]

## Refactoring Candidates
- N files > 500 lines, M functions > 60 lines
- [top 5 by severity]

## Dependencies
- N packages outdated; [list if any]; security advisories [if any]

## .claude/ Health
- N stale references found; [list]

## Unmerged Worktree Branches
- <wt path>: branch <name>, <N> unmerged commits
- (or "(none)")

## Code Duplication (jscpd)
- [top 10 duplicate pairs with file:line ranges]

## Skill / Agent Description Overlap (jaccard > 0.4)
- [list pairs] (or "no overlapping pairs above 0.4")

## Recommended Actions
1. <highest-priority>
2. ...

# Publish

  uv run python scripts/redact_for_gist.py --in /tmp/weekly-hygiene-body.md --out /tmp/weekly-hygiene-body.redacted.md
  gh gist create --public \
    --filename "weekly-hygiene-${WEEK_TAG}.md" \
    --desc "Weekly code hygiene + refactor candidates — ${WEEK_TAG}" \
    /tmp/weekly-hygiene-body.redacted.md

RETURN the gist URL as the SOLE output. No commentary.
```

---

## Subagent prompt: Mentor-prep agenda

```
Assemble recent clean-result GitHub issues into a single
screen-shareable mentor-meeting agenda. Read-only — never edit source
issues; never paraphrase TL;DRs (copy verbatim). Default window: past 7
days from now (label `clean-results`).

# Step 1: Fetch issues

WEEK_AGO=$(date -d '7 days ago' +%Y-%m-%d)
WEEK_TAG=$(date +%Y-W%V)

gh issue list --label clean-results --state all \
  --search "updated:>=${WEEK_AGO}" \
  --json number,title,body,labels,updatedAt --limit 100

For each issue, fetch the full body:
  gh issue view <N> --json number,title,body,labels,updatedAt

# Step 2: Parse each clean-result body

Extract for each:
- TL;DR block: everything between `## TL;DR` and the horizontal rule
  preceding `# Detailed report`.
- Subsections to verify present: ### Background, ### Methodology,
  ### Results, ### Next steps. Warn (not fail) if any missing.
- Hero figure URL: first `raw.githubusercontent.com` markdown image under
  ### Results.
- Confidence tag: parse the `**Confidence: HIGH | MODERATE | LOW**` line
  in ### Results.
- Status icon:
  - ✅ if confidence = HIGH
  - ⚠️  if confidence = MODERATE
  - ❌ if confidence = LOW or any CRITICAL caveat present
- Next steps subsection — used for the aggregated "Open questions / asks"
  block at end.

# Step 3: Order

Primary sort: confidence (HIGH → MODERATE → LOW).
Secondary: updatedAt desc.
Tertiary: issue number asc.

# Step 4: Build doc

# Mentor Meeting — <YYYY-MM-DD>

## Summary slide
- ✅ #<N> — <one-sentence claim> (confidence)
- ⚠️  #<N> — ...
- ❌ #<N> — ...

## Agenda
### 1. #<N> — <title>
<full TL;DR pasted verbatim from issue body, hero figure inlined>
[→ Full report](https://github.com/superkaiba/explore-persona-space/issues/<N>)

### 2. #<N> — ...

## Open questions / asks
<aggregated "Next steps" subsections, deduplicated, ranked across all
clean-results by information gain per GPU-hour.>

## Backup slides
- #<N> — Detailed report: https://github.com/superkaiba/explore-persona-space/issues/<N>#user-content-detailed-report
- ...

# Step 5: Length check

If > 8 clean-results in the window, warn at the top of the doc:
"⚠️  N clean-results in window — consider filtering or splitting into two
meetings (Joe Benton ≤10min reading-time principle)."

# Step 6: Publish

  uv run python scripts/redact_for_gist.py --in /tmp/weekly-mentor-body.md --out /tmp/weekly-mentor-body.redacted.md
  gh gist create --public \
    --filename "weekly-mentor-${WEEK_TAG}.md" \
    --desc "Mentor meeting agenda — week ${WEEK_TAG}" \
    /tmp/weekly-mentor-body.redacted.md

If zero clean-results in the window, do NOT publish a gist. Return the
literal string "(no clean-results this week — skipping mentor agenda)"
as the output instead. The orchestrator handles that gracefully.

RETURN the gist URL (or the skip message) as the SOLE output. No commentary.
```

## Rules

1. **Manual trigger only.** No cron.
2. **Parallel dispatch.** Always issue all `Agent` tool calls in a single
   assistant message — they must run concurrently.
3. **One gist per task.** No combined report. The user requested
   per-task gists.
4. **Partial-failure tolerance.** A failing subagent must not block the
   others; report the failure inline and continue.
5. **Read-only on the project.** None of the subagents auto-modify
   CLAUDE.md, code, agents, skills, or hooks. Every proposed change
   lands in a gist for the user to review.
