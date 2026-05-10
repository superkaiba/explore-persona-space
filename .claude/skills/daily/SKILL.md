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

## API bucket strategy (for new GitHub calls inside this skill)

GitHub's per-hour rate limits are **separate buckets** that count
independently:

| Bucket | Limit | What hits it |
|---|---|---|
| core (REST) | 5000/hr | `gh api repos/...`, `gh issue list` (without `--json`), `gh issue create`, label edits |
| graphql | 5000/hr | `gh issue list --json <fields>` (verified empirically 2026-05-10), `gh issue view --json`, `gh project ...`, `scripts/gh_project.py` (except `_fetch_all_issue_labels` and `_gh_issue_view_full`, migrated to REST) |
| search | 30/min ≈ 1800/hr | `gh issue list --search "..."`, `gh api search/issues` |

**Rule for new call sites in this skill:** prefer REST (`gh api
repos/.../issues?...` with jq filtering) over `gh issue list --json` or
`--search` patterns. The `/pm` triage exhausted GraphQL on 2026-05-10
while core sat at <5/5000 used; REST is the cheap bucket.

Inline-comment patterns (`--json comments`) are the one exception:
fetching comments per-issue via REST reintroduces an N+1, which is worse
total quota than one GraphQL call. The clean-result-review subagent at
Step 1.5 keeps `gh issue list --json ...,comments` on GraphQL for that
reason — single call, all bodies + comments returned.

## Top-level variables

- `INTERACTIVE` — boolean string. `true` = call `AskUserQuestion` for the
  free-form-thoughts and self-reflection steps; `false` = skip them and
  emit `(unanswered)` placeholders. Set in "Argument parsing" below from
  the presence of `--autonomous` in `$ARGS`. Referenced literally as
  `INTERACTIVE` in Steps 0, 0.5, 2.5, and the parallel-task notes
  (issue #275 round-1 NIT-3 — promoting the spelling to a documented
  top-level variable rather than a free-floating string).

## Dispatch table

| Task | Subagent type | Mode | Returns |
|---|---|---|---|
| Review clean-result drafts | `general-purpose` | **sequential** (runs FIRST) | summary block (passed into Daily summary) |
| Propose + capture next steps | `general-purpose` | **sequential** (runs SECOND) | next-steps block (passed into Daily summary) |
| Daily summary | `general-purpose` | parallel | gist URL — daily mentor update (also creates gist mirrors per today's clean-result and links both) |

The first two rows run **sequentially before** the parallel fan-out so
their `AskUserQuestion` calls don't race the parallel subagents'
output. Subsequent rows run in parallel as a single assistant message.

(Adding more parallel tasks is still a 2-step change: append a row here,
append a `## Subagent prompt: <name>` section below.)

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

1.5. **Sequential subagent: Review clean-result drafts.** Spawn the
   `Review clean-result drafts` subagent first, BEFORE the parallel
   fan-out. It runs to completion (its per-draft `AskUserQuestion`
   calls cannot race the parallel subagents). On return, it emits a
   summary block of the form:

   ```
   ## Clean-result drafts reviewed
   - #<N1>: <decision> (promote / reject / defer)
   - #<N2>: <decision>
   ...
   ```

   Capture the block into `CLEAN_RESULT_REVIEW_SUMMARY` and pass it
   as part of the input to the Daily summary subagent (via
   `/tmp/daily-clean-result-review.md` — the Daily summary subagent
   sources it under `## Clean-result drafts reviewed`).

   When `INTERACTIVE=false`, the subagent emits `defer (autonomous mode)`
   for every draft and flips no labels.

1.6. **Sequential subagent: Propose + capture next steps for today's
   experiments.** Spawn the `Propose + capture next steps` subagent
   AFTER the clean-result review (Step 1.5) and BEFORE the parallel
   fan-out. It proposes a next-step per today's clean-result issue
   based on the issue body (existing `### Next steps` subsection if
   present, otherwise heuristic from the AI Summary's results), then
   calls `AskUserQuestion` once per issue to capture the user's edit
   or confirmation. On return, it emits a block of the form:

   ```
   ## Next steps
   - **#<N1>** — <user-confirmed next step>
   - **#<N2>** — <user-confirmed next step>
   ...
   ```

   Capture into `NEXT_STEPS_BLOCK` and pass via
   `/tmp/daily-next-steps.md` to the Daily summary subagent.

   When `INTERACTIVE=false`, the subagent emits its proposed next-steps
   as-is without prompting the user (each prefixed `(proposed)` so the
   user can spot which ones weren't reviewed).

2. **In a single assistant message**, issue one `Agent` tool call per
   PARALLEL row in the dispatch table above (skip the sequential rows —
   they already ran in Steps 1.5 and 1.6). Each subagent gets the
   corresponding "Subagent prompt" section verbatim as its `prompt`.

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

4.5. **Single end-of-run gist-publication gate (#275 item 7).** Determine
   `INTERACTIVE` per Step 0's rule. If `INTERACTIVE=false`, skip the gate
   entirely (this preserves overnight `/daily --autonomous` usability).

   If `INTERACTIVE=true`:

   1. Stack a SINGLE preview block. For each gist URL returned by a
      subagent in this run (from Step 1.5 AND from the parallel
      Step 2 dispatch), fetch the first 20 lines of its body and
      render under one H2 per task:

      ```bash
      for url in $GIST_URLS; do
          echo "## $TASK_NAME"
          gh gist view "$url" | head -20
          echo
      done
      ```

   2. Show the stacked preview to the user, then ONE `AskUserQuestion`:

      > Daily run published N gists. Choose one:
      >   1. **keep_all** — leave all N as-is
      >   2. **retract <comma-list>** — `gh gist delete <url>` each
      >      retracted ID; orchestrator posts a one-line summary
      >   3. **edit_and_repost <comma-list>** — single conversational
      >      turn where the user provides edits per gist; orchestrator
      >      re-renders, re-publishes via `gh gist create`, deletes
      >      the old

   3. Apply the choice. Log final state to
      `/tmp/daily-state-${TODAY}.md`.

   This is the ONLY user-prompt for gist publication in `/daily`. There
   is no per-subagent approval gate. The clean-result-review subagent's
   per-draft `AskUserQuestion` calls (Step 1.5) are separate (different
   gate for different decision) and run BEFORE the parallel fan-out.

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

## Subagent prompt: Review clean-result drafts

```
You are reviewing every open `clean-results:draft` issue and surfacing
each to the user for promote / reject / defer. This subagent runs
SEQUENTIALLY before the parallel /daily fan-out so its AskUserQuestion
calls do not race the other subagents.

# Argument-parse rule (run FIRST)

If the orchestrator passed `INTERACTIVE=false` (autonomous mode), SKIP
Step 3 entirely. Emit a summary block where every line is
`#<N>: defer (autonomous mode)`. Do NOT flip any labels. Return only
the summary block.

# Step 1: List drafts AND their bodies/comments in ONE call.

# Single `gh issue list` with --json fields that include body + comments.
# Each item already carries everything Step 2 needs — no per-issue
# `gh issue view` round-trip. Previously this section ran N+1 queries
# (one list + one view per draft); the new shape is exactly 1 call
# regardless of draft count.
gh issue list --label clean-results:draft --state all \
  --json number,title,body,labels,updatedAt,comments

If zero drafts, emit `## Clean-result drafts reviewed\n(no drafts open)`
and return.

# Step 2: For each draft in the list above, parse from the in-memory JSON.

For each draft #<N> (from the Step 1 result):
  # No additional API call — body is already in item.body, comments in
  # item.comments. Walk item.comments backwards to find the most recent
  # `epm:reviewer-verdict` marker.

Parse:
  - TL;DR confidence (HIGH | MODERATE | LOW)
  - reviewer verdict from the most recent `epm:reviewer-verdict` marker
    (PASS | CONCERNS | FAIL)
  - whether the analyzer's interpretation contains the
    `### Raw-output spot check (5 random rows)` H3
  - issue updatedAt timestamp

# Step 3: For each draft, AskUserQuestion (sequential, one per draft):

> Promote / Reject / Defer issue #<N>?
>   - Confidence: HIGH | MODERATE | LOW
>   - Reviewer verdict: PASS | CONCERNS | FAIL
>   - Raw-output spot check: present | missing
>   - Updated: <timestamp>
>   - TL;DR (verbatim, first 30 lines): …
>
>   1. **promote** — flip clean-results:draft → clean-results
>   2. **reject** — leave at :draft, post a comment with the reason
>   3. **defer** — leave for tomorrow

# Step 4: Apply decisions:

  - promote: gh issue edit <N> --add-label clean-results --remove-label clean-results:draft
  - reject: gh issue comment <N> --body "<user reason>"
  - defer: no-op

# Step 5: Emit a summary block. Format EXACTLY:

## Clean-result drafts reviewed

- #<N1>: <decision>
- #<N2>: <decision>
...

Then write the same block to `/tmp/daily-clean-result-review.md` so the
Daily summary subagent can source it.

RETURN the summary block as the SOLE output of this task. No
commentary, no preamble — just the block.
```

## Subagent prompt: Propose + capture next steps

```
You are proposing a next-step for each clean-result issue updated today
and capturing the user's confirmation or edit. This subagent runs
SEQUENTIALLY between the clean-result review (Step 1.5) and the parallel
/daily fan-out so its AskUserQuestion calls do not race the other
subagents.

# Argument-parse rule (run FIRST)

If the orchestrator passed `INTERACTIVE=false` (autonomous mode), SKIP
Step 3 entirely. Emit your proposed next-steps as-is, each prefixed
`(proposed)`, so the user can later spot which ones were never reviewed.
Return only the block.

# Step 1: List today's clean-results (drafted OR promoted).

# REST `/repos/.../issues` endpoint routes through the core 5000/hr
# bucket. `gh issue list --json` would route through GraphQL (verified
# empirically 2026-05-10: --json decremented GraphQL not core), and
# `--search` would route through the more-constrained search bucket
# (30/min). We fetch all recently-updated issues via REST and filter to
# clean-result labels client-side. PRs are excluded by the
# `select(has("pull_request") | not)` jq filter (REST's issues endpoint
# returns both).
gh api "repos/$GH_REPO_OWNER/$GH_REPO_NAME/issues?state=all&since=$(date -u -d 'yesterday' +%Y-%m-%dT00:00:00Z)&per_page=100" \
  --paginate \
  --jq '[.[]
    | select(has("pull_request") | not)
    | select(.labels | any(.name == "clean-results" or .name == "clean-results:draft"))
    | {number, title, body, updatedAt: .updated_at}]'

If zero issues qualify, emit `## Next steps\n(no clean-results updated today)`
and return.

# Step 2: For each issue, propose a next step.

For each issue #<N>:
  - Read the body's `### Next steps` H3 inside `## AI Summary` if present.
    The first 1-2 bullets there usually name a concrete follow-up.
  - If no `### Next steps` H3 exists, derive a proposal from the
    Confidence line: LOW/MODERATE → "rerun with multiple seeds to
    upgrade confidence" or "extend to model X to test generalization";
    HIGH → "write up for paper / move to RESULTS.md".
  - Keep proposals to ONE sentence each, naming a concrete experiment /
    model / eval where possible.

# Step 3: For each issue, AskUserQuestion (sequential, one per issue):

> Next step for issue #<N>?
>   Title: <title>
>   Confidence: HIGH | MODERATE | LOW
>   Proposed next step: <agent-proposed sentence>
>
>   Reply with:
>   - empty / "ok" / "yes" → accept the proposal
>   - free-form prose → use the user's prose as the next step verbatim
>   - "skip" → omit this issue from the next-steps block

# Step 4: Emit a block. Format EXACTLY:

## Next steps

- **#<N1>** — <accepted or edited next step>
- **#<N2>** — <accepted or edited next step>
...

Then write the same block to `/tmp/daily-next-steps.md` so the Daily
summary subagent can source it.

RETURN the block as the SOLE output of this task. No commentary, no
preamble — just the block.
```

## Subagent prompt: Daily summary

```
You are generating today's daily research mentor update for the
explore-persona-space project. Lead with the result, not the process.
Reading-time target: under 5 minutes.

# Data sources (gather in parallel via Bash; all are read-only)

0. **Clean-result drafts reviewed today.** If
   `/tmp/daily-clean-result-review.md` exists, read it and emit its
   contents under a `## Clean-result drafts reviewed` H2 in the body.
   Otherwise omit the section entirely (do not write a stub).

0.5. **User-confirmed next steps for today's experiments.** If
   `/tmp/daily-next-steps.md` exists, read it and use the per-issue
   `**#<N>** — <next-step>` lines when populating the
   `## Today's experiments — title list` section's "Next step:" lines.
   If the file is missing, fall back to "(no next step recorded)" per
   issue.

1. Git history since midnight:
   git log --since="midnight" --no-merges --oneline --stat
   git diff --stat HEAD~$(git log --since="midnight" --oneline | wc -l)..HEAD 2>/dev/null

2. Source issues that posted an `epm:results` marker today (ONE call;
   comments are inlined so we don't N+1 follow up per issue):
   gh issue list --state open \
     --search "is:issue (label:status:running OR label:status:uploading OR label:status:done-experiment) updated:>=$(date -u -d 'yesterday' +%Y-%m-%d)" \
     --json number,title,labels,updatedAt,comments
   The previous shape used `--label A --label B --label C` which is
   AND-semantics in `gh issue list` and silently returned an empty set
   (no issue carries all three labels at once). Switching to a search
   query with explicit OR also fixes that correctness gap.
   Walk each item's `comments` array backwards in-memory for the
   most-recent `epm:results` marker — no further API calls.

3. Clean-result issues UPDATED (created or edited) today (drafted OR
   promoted — search query uses comma-OR inside `label:`; the previous
   `--label A --label B` AND-semantics returned an empty set):
   gh issue list --state all \
     --search "is:issue label:clean-results,clean-results:draft updated:>=$(date -u -d 'yesterday' +%Y-%m-%d)" \
     --json number,title,body,createdAt,updatedAt
   Read each body; extract the AI TL;DR's first sentence (the colloquial
   title — same text as the issue title minus the `(... confidence)`
   suffix), the AI TL;DR + AI Summary sections, and the hero figure
   URLs. The colloquial title goes in the title list AND in each
   per-experiment `<details>` block's `<summary>`.

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

# Mirror today's clean-results to gists FIRST (lookup-or-create — persistent, NOT ephemeral)

For each clean-result issue updated today (from data source 3 above),
SYNC a persistent gist mirror — one gist per issue, reused across daily
runs. This avoids accumulating orphan gists when an issue is edited
across multiple days (revisions, reviewer-FAIL revise loops, multi-day
follow-up runs all bump `updatedAt`).

The lookup-or-create pattern:

1. **Look up** the existing gist URL by scanning the issue body for the
   canonical `> 📄 **[View this clean-result as a gist](https://gist.github.com/<owner>/<gist-id>)**`
   callout at the very top. This callout is the single source of truth
   for the issue→gist mapping (no separate cache file).
2. **If found:** PATCH the existing gist with the latest issue body.
3. **If not found:** create a new gist, then UPDATE the issue body to
   prepend the canonical callout pointing at the new gist (so future
   daily runs find it).

```bash
for issue_num in $TODAYS_CLEAN_RESULTS; do
    gh api repos/$GH_REPO_OWNER/$GH_REPO_NAME/issues/$issue_num --jq .body > /tmp/issue-$issue_num-body.md

    # Step 1: lookup
    gist_id=$(grep -oE 'gist\.github\.com/[^/[:space:])]+/[a-f0-9]+' /tmp/issue-$issue_num-body.md \
              | head -1 | grep -oE '[a-f0-9]{20,}$' || true)

    if [ -n "$gist_id" ]; then
        # Step 2: PATCH existing gist
        # First, fetch the gist's current filename (we PATCH the same file).
        filename=$(gh api gists/$gist_id --jq '.files | keys[0]')
        jq -n --rawfile b /tmp/issue-$issue_num-body.md --arg f "$filename" \
            '{files: ($f | {($f): {content: $b}})}' > /tmp/gist-patch-$issue_num.json
        gist_url=$(gh api -X PATCH gists/$gist_id --input /tmp/gist-patch-$issue_num.json --jq '.html_url')
    else
        # Step 3: create new gist + write callout into issue body
        gist_url=$(gh gist create --public \
            --filename "issue-$issue_num-clean-result.md" \
            --desc "Clean-result mirror of issue #$issue_num" \
            /tmp/issue-$issue_num-body.md)
        callout="> 📄 **[View this clean-result as a gist]($gist_url)** — anchor links (\`[§ Background]\` etc.) navigate cleanly in-page on a gist; this issue body sometimes opens them in a new tab depending on browser/extension settings. Contents identical.

---

"
        new_body="${callout}$(cat /tmp/issue-$issue_num-body.md)"
        printf '%s' "$new_body" > /tmp/issue-$issue_num-body-with-callout.md
        jq -n --rawfile b /tmp/issue-$issue_num-body-with-callout.md '{body: $b}' > /tmp/issue-$issue_num-patch.json
        gh api -X PATCH repos/$GH_REPO_OWNER/$GH_REPO_NAME/issues/$issue_num --input /tmp/issue-$issue_num-patch.json
    fi

    echo "$issue_num=$gist_url" >> /tmp/daily-gist-urls.txt
done
```

Capture each gist URL keyed by issue number into `/tmp/daily-gist-urls.txt`. You will reference both
`https://github.com/<owner>/<repo>/issues/<N>` and the gist URL in the
body's per-experiment sections. The persistent mapping survives across
daily runs because the callout is part of the issue body itself.

**Idempotency notes:**
- If the issue already has a callout pointing at gist `X`, today's PATCH updates `X` in place. Tomorrow's daily run will also find `X` via the same callout. Single gist per issue, forever.
- If the analyzer (Step 6, on first promotion) ever starts emitting the canonical callout itself, this lookup-or-create logic just finds the gist and PATCHes it — no double-creation.
- The callout grep is intentionally tolerant of whitespace and matches the gist-id by hex-suffix (`[a-f0-9]{20,}$`) to avoid false positives on any other gist URLs that might appear in the body (figure embeds, external references).

# Output structure (markdown body)

# Daily Update — <YYYY-MM-DD>

## Today's experiments — title list

[Bulleted list of EVERY clean-result issue updated today. Each line is
the colloquial title (sentence 1 of the AI TL;DR — paragraph-LEDE
register; see `.claude/skills/clean-results/template.md` § Title
conventions) followed by the user-confirmed next step from
/tmp/daily-next-steps.md.]

1. **#<N1> — <colloquial title>** ([issue](url) · [gist](url))
   *Next step:* <user-confirmed next step>
2. **#<N2> — <colloquial title>** ([issue](url) · [gist](url))
   *Next step:* <user-confirmed next step>
...

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

## Experiments

[For EACH clean-result issue updated today, emit a `<details open>` block
with the AI TL;DR + AI Summary inline. Each block is independently
collapsible; default-open so first read is unchanged. Header line shows
the colloquial title and the issue + gist links. Inline body is the
issue's `## AI TL;DR (human reviewed)` and `## AI Summary` sections
(keep their internal H3 / details structure intact).]

<details open>
<summary><b>#<N1> — <colloquial title></b> ([issue](url) · [gist](url))</summary>

[Full AI TL;DR (lede pair + bullets) verbatim from the issue body.]

[Full AI Summary verbatim — Background, Methodology, Result N sections,
optional Next steps. Hero figure URLs resolve normally.]

</details>

<details open>
<summary><b>#<N2> — <colloquial title></b> ([issue](url) · [gist](url))</summary>

[Same shape.]

</details>

[repeat per clean-result updated today]

## Blockers
[Or "None".]

## Running experiments
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
