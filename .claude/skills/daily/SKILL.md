---
name: daily
description: >
  Run every daily research-orchestration task in parallel via subagents,
  each emitting its own redacted public gist. Manual trigger only — no cron.
  Returns one gist URL per task. Add new daily tasks by appending a row to
  the dispatch table and a corresponding subagent prompt section.
user_invocable: true
---

# Daily

End-of-day fan-out. Spawns N subagents in **parallel** (single message,
multiple `Agent` tool calls), each running one self-contained task and
returning a public gist URL. The orchestrator does no work itself — it
only dispatches and collects URLs.

## When to use

Manual trigger at the end of each work day. **Manual trigger only — no
cron.** Use `/schedule` if you want to wire a cron later.

## Dispatch table

| Task | Subagent type | Returns |
|---|---|---|
| Daily summary | `general-purpose` | gist URL — daily mentor update |

(Today there is one task. Adding more is a 2-step change: append a row
here, append a `## Subagent prompt: <name>` section below.)

## Procedure

**Argument parsing.** `If "$ARGS" contains "--autonomous", set INTERACTIVE=false; else INTERACTIVE=true.`
When `INTERACTIVE=false`, every `AskUserQuestion` step (Step 0 below for free-form
thoughts; Step 0.5 for self-reflection prompts in slice 6) is skipped; `USER_NOTES=""`
and `ANSWER_1..6=""`. The substitution rules in Step 2.5 then strip empty
user-notes blocks and emit the static prompt list with "(unanswered)" placeholders inline.

0. **Free-form thoughts (interactive).** Determine `INTERACTIVE` per the rule
   above. If `INTERACTIVE=true`, call `AskUserQuestion`:

   > "Any free-form thoughts you want to include in today's daily gist?
   > (Examples: surprises, questions, frustrations, decisions.) Reply with
   > prose — empty answer skips the section."

   Capture into `USER_NOTES`. Empty / "skip" / "(none)" → `USER_NOTES=""`.
   If `INTERACTIVE=false`, set `USER_NOTES=""` without asking.

0.5. **Self-reflection (interactive).** Determine `INTERACTIVE` per the rule
   above. If `INTERACTIVE=true`, call `AskUserQuestion` once per the 6 daily
   prompts below, capturing each answer into `ANSWER_1` … `ANSWER_6`.
   Empty answers → `ANSWER_N=""` (substituter prints `(unanswered)` inline).
   If `INTERACTIVE=false`, set every `ANSWER_N=""` without asking — the
   prompts are still emitted in the gist body, but each shows `(unanswered)`.

   The 6 daily prompts (with sources cited inline; see "Self-reflection
   prompt sources" at the bottom of this file for the full URL list):

   1. **What did I do today, and was today's most important task the thing I actually spent the most time on?** (Nanda — Truth-Seeking)
   2. **Did any experiment update my beliefs today? In which direction, and by how much?** (Nanda + Perez)
   3. **For the experiment I'm running tomorrow: what's the minimum version that would still update me?** (Chua)
   4. **Where did I lose the most time today, and could that have been a 5-minute Slack/AskUserQuestion check instead?** (Perez)
   5. **Did anything succeed unexpectedly? Did anything fail unexpectedly? What does each tell me?** (Hughes + Nanda)
   6. **What's the one thing I'd tell my mentor about today in 30 seconds?** (Benton)

1. Compute `TODAY=$(date +%Y-%m-%d)` and `TS=$(date -Iseconds)`.
2. **In a single assistant message**, issue one `Agent` tool call per row
   in the dispatch table above. Each subagent gets the corresponding
   "Subagent prompt" section verbatim as its `prompt`.

   2.5. **String-substitute USER_NOTES + ANSWER_1..6 into the subagent prompt template.** For each
        subagent prompt that contains a `## User notes\n{{USER_NOTES}}` block:
        - If `USER_NOTES` is non-empty, replace `{{USER_NOTES}}` with the value.
        - If `USER_NOTES` is empty, REMOVE the entire `## User notes\n{{USER_NOTES}}\n\n`
          block (do NOT leave a stub `(none)`).

        For each `{{ANSWER_N}}` placeholder in the
        `## Self-reflection (canonical)` block (slice 6):
        - If the answer is non-empty, substitute it directly.
        - If empty, substitute the literal string `(unanswered)`.
        Pass the substituted prompt to the Agent tool.
3. Wait for all subagents to complete (they run concurrently).
4. Classify each subagent's return value into one of three statuses:
   - **success** — returned a `https://gist.github.com/...` URL
   - **skipped** — returned a literal `(skipped — <reason>)` style string
     (no daily task currently does this, but the framework supports it)
   - **failed** — crashed, errored, or returned nothing parseable
5. **Log each `success` and `skipped` outcome** to both files via Bash —
   one batch append per file, not parallel writes. `failed` outcomes are
   NOT logged. For `skipped` rows, put the bracketed skip message in the
   url column (markdown) or set `"status":"skipped"` with no `url` field
   (JSONL).

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
   {
     echo "| ${TODAY} | daily | <task> | <url-or-skip> |"
   } >> docs/update_log.md

   # JSONL: success rows have "status":"success","url":"...";
   #        skipped rows have "status":"skipped","reason":"...".
   {
     echo '{"date":"'"${TODAY}"'","ts":"'"${TS}"'","scope":"daily","task":"<task>","status":"success","url":"<url>"}'
   } >> .claude/cache/update_log.jsonl
   ```

6. Report to the user as a bulleted list:
   ```
   Daily updates posted:
   - Daily summary: <url-or-skip-message>
   (logged to docs/update_log.md + .claude/cache/update_log.jsonl)
   ```
7. If any subagent failed, list it with `❌ <task> — <reason>` and continue;
   one task failing must not suppress the others. Failed tasks are NOT
   logged (skipped tasks ARE logged — see step 5).

---

## Subagent prompt: Daily summary

```
You are generating today's daily research mentor update for the
explore-persona-space project. Lead with the result, not the process.
Reading-time target: under 5 minutes.

# Data sources (gather in parallel via Bash; all are read-only)

1. Git history since midnight:
   git log --since="midnight" --no-merges --oneline --stat
   git diff --stat HEAD~$(git log --since="midnight" --oneline | wc -l)..HEAD 2>/dev/null

2. Source issues that posted an `epm:results` marker today:
   gh issue list --state open --label status:running --label status:uploading --label status:done-experiment \
     --json number,title,labels,updatedAt | jq '[.[] | select(.updatedAt >= (now - 86400 | todate))]'
   For each, fetch the latest epm:results comment via `gh issue view <N> --comments`.

3. Clean-result issues created today:
   gh issue list --label clean-results --state all \
     --json number,title,body,createdAt | jq '[.[] | select(.createdAt >= (now - 86400 | todate))]'
   Read each body; extract TL;DR + confidence + hero figure URL.

4. Figures generated today:
   find figures/ -name "*.png" -mtime 0 -type f
   READ each .png with the Read tool before captioning. Do NOT guess content from filename.

5. Eval results from today:
   find eval_results/ -name "*.json" -mtime 0 -type f

6. Current queue grouped by status:
   gh issue list --state open \
     --label status:proposed --label status:approved --label status:running \
     --json number,title,labels,updatedAt

7. Running experiments on pods (use ssh MCP if loaded; else Bash ssh):
   nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
   on each currently-registered pod from `python scripts/pod.py config --list`.

# Output structure (markdown body)

# Daily Update — <YYYY-MM-DD>

## TL;DR
[2-3 sentences. Single most important finding today + what it means. Lead with finding, not activity.]

## User notes
{{USER_NOTES}}

## Self-reflection (canonical)
1. **What did I do today, and was today's most important task the thing I actually spent the most time on?**
   {{ANSWER_1}}
2. **Did any experiment update my beliefs today? In which direction, and by how much?**
   {{ANSWER_2}}
3. **For the experiment I'm running tomorrow: what's the minimum version that would still update me?**
   {{ANSWER_3}}
4. **Where did I lose the most time today, and could that have been a 5-minute Slack/AskUserQuestion check instead?**
   {{ANSWER_4}}
5. **Did anything succeed unexpectedly? Did anything fail unexpectedly? What does each tell me?**
   {{ANSWER_5}}
6. **What's the one thing I'd tell my mentor about today in 30 seconds?**
   {{ANSWER_6}}

## Done Today
### <concrete description>
**What we did:** [1 sentence: experiment/analysis, model, key design choice]
**What we found:** [1-2 sentences, quantified]
**Key figure:** [inline ref or "no figure"]
**So what:** [1-2 sentences: what this rules in/out]
**Caveats:** [1 sentence: biggest limitation]

[repeat per completed task]

## Key Figures
### Figure 1: <descriptive title>
![desc](../figures/filename.png)
**What this shows:** [1-2 sentences. Axis labels, pattern, what's surprising.]

[max 3-4 figures]

## Next Steps
[Ordered by information gain per GPU-hour. Concrete + actionable.]
1. **<action>** — <why>. Expected cost: N GPU-hours
2. ...

## Blockers
[Or "None".]

## Running / In Progress
- <experiment>: <pod>, ETA, what to expect

# Writing rules

- Lead with result, not process.
- No internal taxonomy numbers / jargon.
- Quantify everything (N, p, effect, CI).
- Honest about negatives — null results constrain the search.
- Bold key numbers; no prose paragraphs in "Done Today".
- Read figures before captioning them.
- Never fabricate. If no work today, write a "Quiet Day" update naming what was attempted.

# Publish

Build body to `/tmp/daily-body.md`. Then:

  uv run python scripts/redact_for_gist.py --in /tmp/daily-body.md --out /tmp/daily-body.redacted.md
  gh gist create --public \
    --filename "daily-$(date +%Y-%m-%d).md" \
    --desc "Daily research update — $(date +%Y-%m-%d)" \
    /tmp/daily-body.redacted.md

`gh gist create` prints the URL on stdout. RETURN that URL as the SOLE
output of this task. No commentary, no preamble — just the URL.
```

## Rules

1. **Manual trigger only.** No cron.
2. **Parallel dispatch.** Always issue all `Agent` tool calls in a single
   assistant message — they must run concurrently.
3. **One gist per task.** No combined report. The user requested
   per-task gists.
4. **Partial-failure tolerance.** A failing subagent must not block the
   others; report the failure inline and continue.

## Self-reflection prompt sources

The 6 daily self-reflection prompts (Step 0.5) are paraphrased from:

1. Nanda — *My Research Process: Key Mindsets — Truth-Seeking* — https://www.alignmentforum.org/posts/cbBwwm4jW6AZctymL/my-research-process-key-mindsets-truth-seeking
2. Nanda — *My Research Process: Understanding and Cultivating Research Taste* — https://www.alignmentforum.org/posts/Ldrss6o3tiKT6NdMm/my-research-process-understanding-and-cultivating-research
3. Perez — *Tips for Empirical Alignment Research* — https://www.alignmentforum.org/posts/dZFpEdKyb9Bf4xYn7/tips-for-empirical-alignment-research
4. Chua — *MATS / SERIMATS experience* — https://jameschua.net/post/serimats_experience/
5. Hughes / Perez — *Tips and Code for Empirical Research Workflows* — https://www.alignmentforum.org/posts/6P8GYb4AjtPXx6LLB/tips-and-code-for-empirical-research-workflows
6. Benton — *Anthropic Fellows Program* — https://alignment.anthropic.com/2024/anthropic-fellows-program/

Phrasings are paraphrases, not direct quotes. Implementer (#251 slice 6)
re-fetched the URLs at slice time to confirm the paraphrases are
faithful; if any source post is later updated and the paraphrase drifts,
flag here and update both the prompt and the citation.
