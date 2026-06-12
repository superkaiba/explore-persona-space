---
name: group-promotion-queue
description: >
  Use when the awaiting_promotion queue needs organizing into related
  groups before review — the user says "group the promotion queue",
  "organize awaiting promotion", "which of these results belong
  together", or is about to start a batch promote pass; also auto-fired
  by the /pm STATUS pass (step 3.3) whenever the queue membership
  changed since the last cached report. Read-only: produces a triage
  report, never mutates task state and never recommends verdicts.
user_invocable: true
---

# Group the awaiting_promotion queue

Clusters the tasks parked at `tasks/awaiting_promotion/` into
**fine-grained, body-grounded groups** of related experiments so the
user can review and promote them group-by-group instead of as a flat
list of 50+ titles. Output is a markdown triage report (chat + cache
file). This skill is strictly **organize-only**:

- **No verdicts.** Never suggest `useful` / `not-useful`, never rank
  groups by quality, never flag a result as weak. Promotion judgment is
  the user's (CLAUDE.md park-and-wait gate). Acting on a group is
  `/promote-clean-result`'s job.
- **Read-only.** No `task.py` mutation of any kind — no tags, no body
  edits, no status changes, no markers on the queued tasks. The only
  writes are the cache file and the chat report.

## Steps

### 1. Enumerate the queue

```bash
uv run python scripts/task.py list-by-status --status awaiting_promotion --json
```

Keep `id`, `title`, `kind`, `tags` per task. Note the count.

### 2. Cache check

The previous report lives at `.claude/cache/promotion-groups.md` with a
first line `<!-- ids: <comma-separated sorted ids> generated: <ISO date> -->`.

- Sorted current ID set **matches** the cache header → the queue hasn't
  changed; render the cached report and stop (skip the subagent).
  Override with an explicit user ask to regenerate ("fresh", "redo the
  grouping").
- Mismatch or no cache → continue to step 3.

### 3. Spawn the grouping subagent

ONE `general-purpose` subagent does all body reading, so the invoking
session (especially the PM session) never pages 50 bodies into its own
context. From the PM session spawn it `run_in_background: true` and
render the stale cache (marked `(stale — regenerating)`) meanwhile;
standalone invocations may run it foreground.

Prompt template (pass the ID list inline; it's small):

```
Group the awaiting_promotion queue into fine-grained clusters of
related experiments. Task IDs: <ids>.

AUTO_REVIEW_DISABLED=1 — do not invoke any review loop on your output.

For EACH task, read ONLY these slices (never the whole body — bodies
are long and raw-EM excerpts must not be paged into context):
  p=$(uv run python scripts/task.py find <N>)
  # frontmatter: title, goal, parent_id, relates_to, tags
  sed -n '/^---$/,/^---$/p' "$p/body.md"
  # the claim headlines: Motivation first paragraph + finding H4s
  sed -n '/^### Motivation/,/^### What I ran/p' "$p/body.md" | head -12
  grep '^#### ' "$p/body.md"

Cluster rules:
- FINE-GRAINED: a group is 2-6 tasks probing the SAME specific
  question, manipulation, or measurement line. The test: the group's
  shared question must be statable in ONE sentence with no "and",
  "various", or "aspects of". A group that needs those words gets
  split. Broad umbrellas like "marker leakage" (which would swallow
  half the queue) are wrong; "does the contrastive-negative budget set
  bystander leakage" is right.
- Body content decides membership. Lineage signals (parent_id chains,
  followup-auto/followup-manual tags, relates_to anchors, shared goal
  text) are supporting evidence, not the criterion — siblings of one
  parent may answer different questions, and unrelated parents may
  converge on the same question.
- Tasks that genuinely fit no group stay in a final "Singletons"
  section — do not force-fit them.
- ORGANIZE ONLY: no useful/not-useful suggestions, no quality
  opinions, no "this one looks superseded". Carry each title's
  confidence tag verbatim; add nothing.
- Group names: plain-English noun phrases naming the shared question.
  No invented jargon, no condition codes.

Return EXACTLY this markdown (it is rendered to the user as-is):

# Awaiting-promotion queue, grouped (<count> tasks, <date>)

## <group name> (<n>)
<one-sentence shared question>
- [#N](https://eps.superkaiba.com/tasks/N) — <condensed claim> (<CONFIDENCE>)
- ... (note in-group lineage inline, e.g. "follow-up of #M above")

## Singletons (<n>)
- [#N](https://eps.superkaiba.com/tasks/N) — <condensed claim> (<CONFIDENCE>)
```

### 4. Persist + render

Write the subagent's report verbatim to
`.claude/cache/promotion-groups.md`, prepending the header line
`<!-- ids: <sorted ids> generated: <date> -->`. Render the report in
chat. Do NOT commit the cache file (it's regenerable; `.claude/cache/`
is runtime state).

### 5. Handoff

Close with one line: act on a group via `/promote-clean-result`
(per-task refinement + `task.py promote <N>` handoff, including its
batch-promote BUGGED prescan). This skill never runs promote itself.

## Common mistakes

| Mistake | Fix |
|---|---|
| 3-6 broad themes covering everything | Groups are question-level; splitting beats lumping. Singletons are fine. |
| Grouping by parent_id / tags alone | Lineage is a hint; the body's question decides. |
| Sneaking in "promote these together" or quality reads | Organize only — the report contains zero opinions. |
| Paging whole bodies into context | Frontmatter + Motivation head + `#### ` headlines only. |
| PM session blocking on the subagent | Background-spawn; render stale cache meanwhile. |
