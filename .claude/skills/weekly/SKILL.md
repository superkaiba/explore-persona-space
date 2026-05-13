---
name: weekly
description: >
  Conversational weekly research review + artifact generation. Walks the user
  through state-of-play, belief audit, bottleneck identification, Hamming
  direction-cut, and next-week selection; then generates two mentor-facing
  artifacts in sagan (overall project narrative + weekly digest), plus
  internal-use gists (workflow optimization + code hygiene) and the
  persistent Marp deck. Manual trigger only — no cron. User reviews each
  artifact and can request iteration before publication.
user_invocable: true
---

# Weekly

End-of-week interactive review + multi-artifact generation. Two
mentor-facing artifacts live in **sagan** (commentable on
`https://sagan.superkaiba.com/p/<slug>`); the others stay as gists or
local files.

## When to use

Manual trigger at the end of each work week. **Manual trigger only — no
cron.** Use `/schedule` if you want to wire a cron later.

## Output map (Sagan Unification)

| Artifact | Sink | Where it lives |
|---|---|---|
| **Overall narrative** | sagan `project_narratives` | https://sagan.superkaiba.com/p/conditional-behavior (auto-archives previous published narrative on publish) |
| **Weekly mentor digest** | sagan `weekly_digests` | Unique on `week_start` — re-runs in the same week UPSERT |
| **Workflow optimization** | gist | Internal use, not mentor-facing |
| **Code hygiene** | gist | Internal use, not mentor-facing |
| **Mentor-prep slides** | repo file | `figures/mentor-slides/deck.md` + `deck.pdf` (persistent Marp deck via `mentor-update-slides` skill) |

The weekly digest body folds in BOTH the "this week" summary AND the
mentor-prep agenda (clean-result TL;DRs) AND the great-thoughts portfolio
audit as sections, so there's one consolidated artifact for the mentor
meeting rather than three separate gists.

## API bucket strategy

Same conventions as `/daily`. The three independent buckets:

| Bucket | Limit | What hits it |
|---|---|---|
| core (REST) | 5000/hr | `gh api repos/...` paths, `gh issue list` without `--json` |
| graphql | 5000/hr | `gh issue list --json <fields>`, `gh issue view --json`, `gh project ...` |
| search | 30/min | `gh issue list --search ...`, `gh api search/issues` |

**Rule:** prefer REST for issue/PR reads. The `/pm` triage exhausted
GraphQL on 2026-05-10 while core sat at <5/5000 used.

## Sagan write helper

A single Bash helper used by Phases 3 and 5 below:

```bash
sagan_psql() {
  local sql="$1"
  DATABASE_URL="${SAGAN_DATABASE_URL:?must be set}" psql "$SAGAN_DATABASE_URL" -v ON_ERROR_STOP=1 -A -t -c "$sql"
}
```

`SAGAN_DATABASE_URL` must be present in the orchestrator's environment
(it's set in `~/sagan/.env` as `DATABASE_URL`; rename to
`SAGAN_DATABASE_URL` in the shell that invokes `/weekly`, or export it
inline from that file with `set -a; . ~/sagan/.env; set +a; export
SAGAN_DATABASE_URL=$DATABASE_URL`).

## Procedure

**Argument parsing.** `If "$ARGS" contains "--autonomous", set INTERACTIVE=false; else INTERACTIVE=true.`
When `INTERACTIVE=false`, every `AskUserQuestion` step is skipped; all
captured values default to `""`. The substitution rules then emit
empty-or-`(unanswered)` placeholders inline.

---

### Phase 1 — Conversational walkthrough (interactive only)

The walkthrough produces working notes that feed every downstream
artifact. Each step uses `AskUserQuestion` (single-select where stated,
else free text). All captured into named variables.

If `INTERACTIVE=false`, skip every prompt and set every captured variable
to `""`. The artifacts will render `(unanswered)` inline where the
variable would have appeared.

**1a. State of play.** Three free-text prompts:

1. *"What's the single most important thing you learned this week? Not
   what shipped — what changed in your head."* → `STATE_LEARNED`
2. *"What's quietly bothering you / what feels off?"* (Nanda "what am I
   being an idiot about?") → `STATE_BOTHERING`
3. *"What's your biggest current bottleneck?"* (compute, decision
   paralysis, missing tool, missing collaborator, unclear hypothesis,
   anything) → `STATE_BOTTLENECK`

**1b. Belief audit.** First fetch a candidate belief list from recent
clean-results to seed the conversation:

```bash
WEEK_AGO=$(date -d '7 days ago' +%Y-%m-%d)
gh api "repos/$GH_REPO_OWNER/$GH_REPO_NAME/issues?state=all&since=${WEEK_AGO}T00:00:00Z&per_page=100" \
  --paginate --jq '[.[]
    | select(has("pull_request") | not)
    | select(.labels | any(.name == "clean-results" or .name == "clean-results:draft"))
    | {number, title, labels: [.labels[].name]}]' \
  > /tmp/weekly-recent-clean-results.json
```

Then ask:

> *"Here's the list of clean-results this week. For each, what's your
> current confidence (HIGH/MOD/LOW)? What would change your mind?
> Reply prose, free-form. You don't have to cover all of them — pick the
> 3-5 most load-bearing."*

→ `BELIEF_AUDIT` (free text, multi-paragraph OK)

**1c. Phase check.** For each live thread, classify it Explore /
Understand / Distill. Use the headings from the current sagan narrative
(fetch via `sagan_psql "SELECT body_md FROM project_narratives WHERE
project_id = (SELECT id FROM projects WHERE slug = 'conditional-behavior')
AND status = 'published' ORDER BY published_at DESC LIMIT 1"`) as a
checklist; ask the user to flag which threads are mis-categorized.

→ `PHASE_CHECK` (free text)

**1d. Hamming cut.** Single question:

> *"If you could only push ONE of the threads above for the next two
> weeks, which one and why? Not 'which is most interesting' — which one,
> if it lands, makes the others either fall out for free or become
> re-prioritizable?"*

→ `HAMMING_THREAD`

**1e. Top-3 next-week.** Free-text:

> *"What 3 specific actions for next week would maximize information
> gain per GPU-hour? Link existing issues or sketch new ones."*

→ `NEXT_WEEK_TOP3`

**1f. Mentor questions.** Free-text:

> *"Anything you want to specifically ask your mentor at the next 1:1?
> Empty answer skips the section."*

→ `MENTOR_QUESTIONS`

---

### Phase 2 — Compute window + context

```bash
WEEK_TAG=$(date +%Y-W%V)            # e.g. 2026-W19
TODAY=$(date +%Y-%m-%d)
TS=$(date -Iseconds)
WEEK_AGO=$(date -d '7 days ago' +%Y-%m-%d)
WEEK_START=$(date -d 'last monday' +%Y-%m-%d 2>/dev/null || date -d 'monday -1 week' +%Y-%m-%d)
```

Resolve project IDs:

```bash
EPS_PROJECT_ID=$(sagan_psql "SELECT id FROM projects WHERE slug='conditional-behavior'")
```

---

### Phase 3 — Dispatch subagents in parallel

In a single assistant message, issue one `Agent` tool call per row in
the dispatch table below. All captured `Phase 1` variables are
substituted into the subagent prompts at dispatch time (the prompts
expect `{{STATE_LEARNED}}`, `{{STATE_BOTHERING}}`, `{{STATE_BOTTLENECK}}`,
`{{BELIEF_AUDIT}}`, `{{PHASE_CHECK}}`, `{{HAMMING_THREAD}}`,
`{{NEXT_WEEK_TOP3}}`, `{{MENTOR_QUESTIONS}}`).

| # | Task | Subagent | Sink |
|---|---|---|---|
| 1 | Weekly digest (summary + mentor agenda + great-thoughts portfolio audit) | `general-purpose` | **sagan `weekly_digests`** |
| 2 | Overall narrative refresh | `general-purpose` | **sagan `project_narratives` (status: draft)** |
| 3 | Workflow optimization | `retrospective` | gist |
| 4 | Code hygiene | `general-purpose` | gist |
| 5 | Mentor-prep slides | `general-purpose` | repo file via `mentor-update-slides` skill |

Sub-prompts are inlined below.

---

### Phase 4 — Review + iterate (interactive only)

After all subagents return, present to the user:

> *"Drafts ready. Sagan rows are at `weekly_digests.id=<id>` and
> `project_narratives.id=<id>` (status: draft). Read them at:*
> *- Weekly digest (preview, no public URL until published): <admin URL>*
> *- Overall narrative draft: <admin URL>*
> *Gists are at: …*
> *Marp deck at `figures/mentor-slides/deck.md`.*
> *Reply with:*
> *- 'publish' to mark the narrative as `published` and the digest as `sent_at`*
> *- 'iterate <artifact-name>: <feedback>' to refine that artifact*
> *- 'skip <artifact-name>' to drop one without publishing*
> *- 'done' if you're satisfied — same as 'publish'"*

If the user replies with `iterate <artifact>: <feedback>`, fetch the
draft body from sagan (or the relevant local file), apply the feedback,
update the row in place via `UPDATE project_narratives SET body_md = …
WHERE id = …` (or similar). Loop until the user says `publish` / `done`.

If `INTERACTIVE=false`, skip the review gate and publish immediately.

---

### Phase 5 — Publish

For the **overall narrative**, mark it published. The API path's
auto-archive logic only triggers on PATCH-via-route; doing it directly:

```bash
sagan_psql "
  UPDATE project_narratives SET status = 'archived' WHERE project_id = '$EPS_PROJECT_ID' AND status = 'published';
  UPDATE project_narratives SET status = 'published', published_at = now() WHERE id = '$NARRATIVE_ID';
"
```

For the **weekly digest**, mark it sent and ensure a share token exists:

```bash
sagan_psql "
  UPDATE weekly_digests
     SET sent_at = now(),
         share_token = COALESCE(share_token, encode(gen_random_bytes(16), 'hex'))
   WHERE id = '$DIGEST_ID'
  RETURNING share_token;
"
```

The returned share token is the mentor-shareable URL:
`https://sagan.superkaiba.com/d/<token>` (existing public route).

---

### Phase 6 — Log + report

Append to both update logs:

```bash
if [ ! -f docs/update_log.md ]; then
  mkdir -p docs
  printf '# Update log\n\nGist URLs from /daily and /weekly. Newest at top.\n\n| date | scope | task | url-or-status |\n|---|---|---|---|\n' > docs/update_log.md
fi
mkdir -p .claude/cache

{
  echo "| ${TODAY} | weekly (${WEEK_TAG}) | digest | https://sagan.superkaiba.com/d/${DIGEST_SHARE_TOKEN} |"
  echo "| ${TODAY} | weekly (${WEEK_TAG}) | narrative | https://sagan.superkaiba.com/p/conditional-behavior |"
  echo "| ${TODAY} | weekly (${WEEK_TAG}) | workflow-optimization | ${WORKFLOW_GIST_URL} |"
  echo "| ${TODAY} | weekly (${WEEK_TAG}) | code-hygiene | ${HYGIENE_GIST_URL} |"
  echo "| ${TODAY} | weekly (${WEEK_TAG}) | mentor-slides | figures/mentor-slides/deck.md |"
} >> docs/update_log.md

{
  echo '{"date":"'"${TODAY}"'","ts":"'"${TS}"'","scope":"weekly","week":"'"${WEEK_TAG}"'","task":"digest","status":"published","sagan_id":"'"${DIGEST_ID}"'","url":"https://sagan.superkaiba.com/d/'"${DIGEST_SHARE_TOKEN}"'"}'
  echo '{"date":"'"${TODAY}"'","ts":"'"${TS}"'","scope":"weekly","week":"'"${WEEK_TAG}"'","task":"narrative","status":"published","sagan_id":"'"${NARRATIVE_ID}"'","url":"https://sagan.superkaiba.com/p/conditional-behavior"}'
  echo '{"date":"'"${TODAY}"'","ts":"'"${TS}"'","scope":"weekly","week":"'"${WEEK_TAG}"'","task":"workflow-optimization","status":"success","url":"'"${WORKFLOW_GIST_URL}"'"}'
  echo '{"date":"'"${TODAY}"'","ts":"'"${TS}"'","scope":"weekly","week":"'"${WEEK_TAG}"'","task":"code-hygiene","status":"success","url":"'"${HYGIENE_GIST_URL}"'"}'
  echo '{"date":"'"${TODAY}"'","ts":"'"${TS}"'","scope":"weekly","week":"'"${WEEK_TAG}"'","task":"mentor-slides","status":"success","path":"figures/mentor-slides/deck.md"}'
} >> .claude/cache/update_log.jsonl
```

Skipped or failed tasks: log skipped with `"status":"skipped","reason":"…"`;
don't log failed. Skipped tasks are reported but don't block.

Final report to user:

```
Weekly artifacts (week <WEEK_TAG>):

Mentor-facing (sagan, commentable):
- Weekly digest: https://sagan.superkaiba.com/d/<share-token>
- Overall narrative: https://sagan.superkaiba.com/p/conditional-behavior

Internal:
- Workflow optimization: <gist-url>
- Code hygiene: <gist-url>
- Mentor slides: figures/mentor-slides/deck.md (+ deck.pdf if rendered)

(logged to docs/update_log.md + .claude/cache/update_log.jsonl)
```

---

## Subagent prompt: Weekly digest (Sagan)

```
You are generating this week's MENTOR-FACING weekly digest for the
explore-persona-space (Sagan slug: conditional-behavior) project. The
output is a single markdown body that will be INSERTED into the sagan
`weekly_digests` table. Lead with the result, not the process.
Reading-time target: under 10 minutes. The 7-day window is "past 7
days from now".

This digest folds together what used to be three separate weekly
artifacts: (a) the past-7-day summary, (b) the mentor-prep agenda
(verbatim clean-result TL;DRs), and (c) the great-thoughts portfolio
audit. One consolidated artifact for the mentor meeting.

# Captured from Phase 1 walkthrough

State of play:
- Most important thing learned: {{STATE_LEARNED}}
- What's bothering: {{STATE_BOTHERING}}
- Bottleneck: {{STATE_BOTTLENECK}}

Belief audit:
{{BELIEF_AUDIT}}

Phase check (Explore/Understand/Distill per thread):
{{PHASE_CHECK}}

Hamming cut:
{{HAMMING_THREAD}}

Top-3 next-week:
{{NEXT_WEEK_TOP3}}

Questions for mentor:
{{MENTOR_QUESTIONS}}

# Data sources (gather in parallel via Bash; read-only)

WEEK_AGO=$(date -d '7 days ago' +%Y-%m-%d)
WEEK_TAG=$(date +%Y-W%V)
MONTH_AGO=$(date -d '30 days ago' +%Y-%m-%d)

1. Git history past 7 days:
   git log --since="7 days ago" --no-merges --oneline --stat
   git diff --stat HEAD~$(git log --since="7 days ago" --oneline | wc -l)..HEAD 2>/dev/null

2. Clean-result issues touched this week (full bodies, for agenda
   section). REST endpoint, jq filter:
   gh api "repos/$GH_REPO_OWNER/$GH_REPO_NAME/issues?state=all&since=${WEEK_AGO}T00:00:00Z&per_page=100" \
     --paginate \
     --jq '[.[]
       | select(has("pull_request") | not)
       | select(.labels | any(.name == "clean-results" or .name == "clean-results:draft"))
       | {number, title, body, createdAt: .created_at, updatedAt: .updated_at, labels: [.labels[].name]}]'

   For each: extract TL;DR + confidence tag + hero figure URL. Bin
   client-side on `createdAt >= ${WEEK_AGO}` for the "new this week"
   subset; the rest are updated-but-not-new.

3. Done experiment + done impl issues this week:
   gh issue list --search "is:issue updated:>=${WEEK_AGO} (label:status:done_experiment OR label:status:done_impl)" \
     --json number,title,labels,updatedAt

4. Recent figures:
   find figures -type f \( -name "*.png" -o -name "*.pdf" \) -mtime -7
   READ each .png with the Read tool before captioning. Max 5 figures.

5. Pending / blocked items:
   gh issue list --state open \
     --search "is:open (label:status:blocked OR label:status:proposed OR label:status:running)" \
     --json number,title,labels,updatedAt
   Partition by label.

6. Done experiments past 30 days (for great-thoughts march-toward audit):
   gh issue list --search "is:issue updated:>=${MONTH_AGO} \
     (label:status:done_experiment OR label:status:done_impl)" \
     --json number,title,labels,updatedAt

7. RESULTS.md and docs/research_ideas.md for great-thoughts context.

8. Last week's great-thoughts portfolio (for stable problem IDs):
   PREV_DIGEST=$(sagan_psql "SELECT body_md FROM weekly_digests ORDER BY week_start DESC OFFSET 1 LIMIT 1" || echo "")
   Extract the "Important problems" table (P1, P2, ...) — carry IDs
   forward verbatim, mark retired ⏹️ <reason> if applicable.

# Output structure

# Weekly digest — week <WEEK_TAG>

## TL;DR
[2-3 sentences. Lead with the one thing the mentor most needs to know.]

## State of play

**What changed this week:** {{STATE_LEARNED}}

**What's bothering me:** {{STATE_BOTHERING}}

**Current bottleneck:** {{STATE_BOTTLENECK}}

## Headline findings (clean-results this week)

[For each clean-result issue in the past 7 days, in order of confidence
(HIGH → MODERATE → LOW) then updated_at desc:

### ✅/⚠️/❌ #<N> — <title>
Confidence: <tag>
TL;DR (verbatim from issue body):
<paste>
Hero figure: <url-or-omit>
[→ Full report](https://github.com/superkaiba/explore-persona-space/issues/<N>)
]

## Done this week

[Group by type:experiment / type:infra / type:analysis. 1-2 lines each.]

## Running experiments

[Currently on pods, expected completion.]

## Blockers

[Open status:blocked. "None" is valid.]

## Belief audit

{{BELIEF_AUDIT}}

## Phase check

{{PHASE_CHECK}}

## Hamming cut — what dominates if I could only push one thread

{{HAMMING_THREAD}}

## Portfolio audit (great thoughts)

### Important problems
[Carried forward across weeks with stable IDs. Mix global (the
subfield's open questions) with local (this project). 🆕 for new,
⏹️ <reason> for retired. Never renumber existing.]

| ID | Problem | Scope | Status |
|---|---|---|---|

### Attack column
[For each active problem, one line. Hamming threshold: a method that
could actually solve it, not just nibble.]

### March-toward audit
[Map every running issue + every done experiment past 30 days + every
open proposal to a problem ID. Anything that maps to nothing is drift —
list under "Drift candidates".]

### Anomalies / contradictions
[Things that don't fit the current model. Pull from clean-result
"Standing caveats" blocks. The Darwin part — next paper often lives here.]

## Next week — top 3 priorities

{{NEXT_WEEK_TOP3}}

[For each: action + expected GPU-hours + falsification criterion +
issue link.]

## Questions for mentor

{{MENTOR_QUESTIONS}}

# Writing rules

- Lead with result, not process. No legacy taxonomy / jargon.
- Quantify everything (N, p, effect, CI). Be honest about negatives.
- Bold key numbers; structured bullets only.
- Read figures before captioning. Max 5 figures total.
- Never fabricate. Cross-reference with clean-result bodies.
- No effect-size jargon (Cohen's d, η², named statistical tests).
- Carry stable problem IDs across weeks.

# Publish — write to sagan

After building the body, write it to /tmp/weekly-digest-body.md then
INSERT into weekly_digests with the orchestrator's helper. Use the
ORCHESTRATOR's environment (the subagent doesn't have SAGAN_DATABASE_URL;
it returns the body for the orchestrator to insert):

  RETURN both:
  1. The full body markdown
  2. The proposed week_start (Monday of the current week, YYYY-MM-DD)

The orchestrator inserts via psql, scoped to the project being reviewed
(default: `conditional-behavior`):

  sagan_psql "
    INSERT INTO weekly_digests (project_id, week_start, body_md, drafted_at)
    VALUES ('$EPS_PROJECT_ID', '<week_start>', \$body\$<body>\$body\$, now())
    ON CONFLICT (project_id, week_start) DO UPDATE SET
      body_md = excluded.body_md,
      edited_at = now()
    RETURNING id;
  "

The returned UUID is the digest_id. Phase 5 of the orchestrator marks
it sent + adds share_token.

RETURN the body + week_start as a JSON blob; no commentary, no gist
publication. The orchestrator handles the DB write.
```

---

## Subagent prompt: Overall narrative refresh (Sagan)

```
You are regenerating the overall PROJECT NARRATIVE for the
Conditional Behavior in Language Models project (sagan slug:
conditional-behavior). The output is a single markdown body that will
be INSERTED as a new row in `project_narratives` with status='draft'.

The narrative answers "where is the program right now?" — comprehensive,
condensed, evergreen. Different shape from the weekly digest, which
answers "what changed this week?". The narrative is the page shown at
https://sagan.superkaiba.com/p/conditional-behavior — written for any
viewer (mentor, collaborators, public).

# Sources

1. Current published narrative (the starting point — most content
   carries forward, you're updating, not regenerating from scratch):

   sagan_psql "SELECT body_md FROM project_narratives
                WHERE project_id = (SELECT id FROM projects WHERE slug='conditional-behavior')
                  AND status='published'
                ORDER BY published_at DESC LIMIT 1"

2. Recent clean-results (past 30 days, full bodies) — for new findings
   to fold in:

   gh issue list --label clean-results --state all \
     --search "updated:>=$(date -d '30 days ago' +%Y-%m-%d)" \
     --json number,title,body,labels,updatedAt --limit 50

3. `RESULTS.md`, `docs/SUMMARY.md`, `docs/papers.md` for context.

4. Open follow-up issues filed this week (for the "Open follow-ups"
   section of the narrative):

   WEEK_AGO=$(date -d '7 days ago' +%Y-%m-%d)
   gh issue list --search "is:open created:>=${WEEK_AGO} (label:status:proposed OR label:type:experiment)" \
     --json number,title,labels,createdAt

# Updates to fold in (from Phase 1 walkthrough)

The narrative SHAPE stays the same week-to-week (Overarching frame →
trigger types → Q1-Q5 → Applications → Open follow-ups → Related but
scoped out). The CONTENT updates based on what changed.

For each Q1-Q5:
- Add any new clean-result evidence under "What we've shown" or
  "Recent experiments" subsection
- Add new open issues to "Next step" / "Filed:" lines
- Refine framing if Phase 1 walkthrough surfaced a sharper version

The Hamming-cut thread ({{HAMMING_THREAD}}) influences the ordering: if
the user identified one thread as dominant for the next 2 weeks, give it
slightly more prominence (e.g., bold the next-step bullet).

# Output structure

Match the existing published narrative exactly. The current shape is:

# <Project title> — current state

## What we're studying
[Conditional behavior + three trigger classes. ~3 sentences.]

## Five research questions

### Q1. <title>
What we've shown so far:
- <bullets with #refs>
Next: <one paragraph or sentence>

### Q2. <title>
[same shape]
... Q3, Q4, Q5 ...

## Applications
[Defense + elicitation. ~2 paragraphs.]

## Open follow-ups currently filed
- [list of issue refs with one-line descriptions]

## Related but scoped out
[Spun-off projects, e.g., EM Mechanism.]

# Writing rules

- This is mentor- and public-facing. NO LessWrong slang, NO project-
  internal jargon, NO confidence labels in inline prose (those live in
  the linked clean-result issues).
- Issue references render as standard markdown links to the GitHub issue.
  The overlay-on-click feature is a future UI build; for now, plain
  links are fine.
- Aim for under 1200 words total. Tight.

# Publish — write to sagan

After building the body, RETURN it to the orchestrator with the proposed
narrative title (e.g., "Current state — <YYYY-MM-DD>"). The orchestrator
INSERTs as draft:

  sagan_psql "
    INSERT INTO project_narratives (project_id, title, body_md, status)
    VALUES ('<project_id>', '<title>', \$body\$<body>\$body\$, 'draft')
    RETURNING id;
  "

Phase 4 of the orchestrator presents the draft to the user for
iteration; Phase 5 marks it published (and auto-archives the previous
published narrative).

RETURN the body + title as JSON; no commentary.
```

---

## Subagent prompt: Workflow optimization (gist, unchanged)

This task uses the existing `retrospective` agent. Spawn with
`subagent_type: "retrospective"` and the prompt below.

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
  uv run python scripts/redact_for_gist.py --in /tmp/weekly-workflow-body.md --out /tmp/weekly-workflow-body.redacted.md
  gh gist create --public \
    --filename "weekly-workflow-${WEEK_TAG}.md" \
    --desc "Weekly Claude Code workflow optimization — ${WEEK_TAG}" \
    /tmp/weekly-workflow-body.redacted.md

READ-ONLY for the project — never modify CLAUDE.md or any agent / skill
/ hook directly. Every proposed change is a diff in the gist.

RETURN the gist URL as the SOLE output. No commentary.
```

---

## Subagent prompt: Code hygiene (gist, unchanged)

```
End-of-week code hygiene scan. Combines repo-wide dead-code analysis,
refactor candidates, dependency freshness, .claude/ health audit, code
duplication (jscpd), and unmerged worktree branches into one report.
READ-ONLY for the project — never auto-refactor.

# Hard requirement

Node v18+ + jscpd accessible via `npx jscpd`. If unavailable, abort and
return: "install Node v18+ and re-run; jscpd is required."

# Procedure

WEEK_TAG=$(date +%Y-W%V)

1. Lint sweep (safe auto-fix):
   uv run ruff check --fix .
   uv run ruff format .

2. Repo-wide dead-code analysis:
   uv run ruff check . --select F401,F811,F841 --no-fix

3. Refactoring candidates: Python files > 500 lines, functions > 60
   lines, functions with > 4 levels of indentation.

4. Dependency audit:
   uv pip list --outdated 2>/dev/null

5. .claude/ health: stale refs in agents/skills/plans/settings.

6. Unmerged worktree branches:
   git worktree list --porcelain | awk '/^worktree/ {print $2}' | grep '\.claude/worktrees/'

7. Code duplication via jscpd:
   npx jscpd --min-lines 10 --min-tokens 50 --reporters json \
     --output /tmp/jscpd src/ scripts/

8. Skill / agent description overlap (Jaccard on description bigrams,
   threshold 0.4).

# Output structure

# Weekly Code Hygiene — week <WEEK_TAG>

## Lint + Format
- N files reformatted, M lint fixes applied

## Dead Code
- [top 10 unused imports / functions]

## Refactoring Candidates
- [top 5 by severity]

## Dependencies
- [outdated + security advisories]

## .claude/ Health
- [stale references]

## Unmerged Worktree Branches
- [list]

## Code Duplication (jscpd)
- [top 10 pairs]

## Skill / Agent Description Overlap
- [pairs above 0.4]

## Recommended Actions
1. ...

# Publish

  uv run python scripts/redact_for_gist.py --in /tmp/weekly-hygiene-body.md --out /tmp/weekly-hygiene-body.redacted.md
  gh gist create --public \
    --filename "weekly-hygiene-${WEEK_TAG}.md" \
    --desc "Weekly code hygiene — ${WEEK_TAG}" \
    /tmp/weekly-hygiene-body.redacted.md

RETURN the gist URL as the SOLE output.
```

---

## Subagent prompt: Mentor-prep slides (Marp deck, unchanged)

```
You are the mentor-prep slides subagent. Generate / update a Marp deck
for this week's mentor meeting using the `mentor-update-slides` skill.
The deck is a single persistent file at `figures/mentor-slides/deck.md`
(plus a rendered PDF) — not a new file per week.

# Step 1: Invoke the skill

Call the Skill tool with:
  skill: mentor-update-slides
  args: "--days 7 --pdf"

The skill writes:
- figures/mentor-slides/deck.md   (Marp source — persistent across weeks)
- figures/mentor-slides/deck.pdf  (rendered via marp-cli)

The deck has three anchored regions (HEADER replaced each run, LOG
append-only newest-first, APPENDIX accumulating). See
`.claude/skills/mentor-update-slides/SKILL.md` § Persistent-deck model.

If the skill skips (zero clean-results in the 7-day window), return the
literal string "(no clean-results this week — skipping mentor slides)".

# Step 2: Return

This task does NOT publish a gist or write to sagan. The slides live as
a file in the repo for the actual meeting. The user reviews + commits
the deck themselves if they want it archived.

RETURN one of:
- The local paths: "figures/mentor-slides/deck.md, figures/mentor-slides/deck.pdf"
- The skip message: "(no clean-results this week — skipping mentor slides)"

No commentary.
```

---

## Rules

1. **Manual trigger only.** No cron.
2. **Parallel dispatch in Phase 3.** All `Agent` tool calls in a single
   assistant message — they run concurrently.
3. **Review gate before publish.** Phase 4 is mandatory in interactive
   mode. Skipped only with `--autonomous`.
4. **Partial-failure tolerance.** A failing subagent doesn't block the
   others; report inline and continue.
5. **Read-only on the project for non-sagan tasks.** Workflow-optimization
   and code-hygiene never modify CLAUDE.md, agents, skills, or hooks
   directly. Slides write only to `figures/mentor-slides/`.
6. **Auto-archive on narrative publish.** Phase 5's SQL transitions the
   previous published narrative to `archived` BEFORE marking the new
   one published, so there's always exactly one published narrative per
   project.

## What this skill does NOT do yet

(All three "future build" items have shipped — see `apps/web/app/p/[slug]/page.tsx`
for the comment + improve-button integration, the `/api/narratives/[id]/improve`
endpoint for the batched agent-run trigger, and the `weekly_digests.project_id`
schema migration for per-project digests.)

## Self-reflection prompt sources

The Phase 1 walkthrough draws on:

1. Nanda — *Post 39: On Reflection* — https://www.neelnanda.io/blog/39-reflection
2. Nanda — *My Research Process: Key Mindsets — Truth-Seeking* — https://www.alignmentforum.org/posts/cbBwwm4jW6AZctymL/my-research-process-key-mindsets-truth-seeking
3. Nanda — *How I Think About My Research Process: Explore, Understand, Distill* — https://www.lesswrong.com/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand
4. Platt — *Strong Inference* (Hamming-style cut)
5. Chua / Hughes — *Tips on Empirical Research Slides* — https://www.lesswrong.com/posts/i3b9uQfjJjJkwZF4f/tips-on-empirical-research-slides
6. Perez — *Tips for Empirical Alignment Research* — https://www.alignmentforum.org/posts/dZFpEdKyb9Bf4xYn7/tips-for-empirical-alignment-research
