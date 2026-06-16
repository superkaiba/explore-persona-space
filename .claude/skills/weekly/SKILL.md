---
name: weekly
description: End-of-week Explore Persona Space narrative — what happened across the week, plus an exhaustive sweep of every problem/confusion/error in the week's Claude Code session transcripts, each with a concrete fix. Same detail bar as /daily but over the whole ISO week (catches what daily missed, plus a living-docs drift check). Workflow-fixable problems become proposed diffs (PROPOSED, not auto-applied; Thomas greenlights); every other problem is logged with a suggested action so nothing is dropped.
---

# Weekly Narrative

Use `tasks/` as the canonical workflow state source. External trackers may be cited
as historical links, but they are never the queue, status source, approval
source, promotion source, or comment thread for workflow decisions.

Two jobs in one file:
1. **Recap** — week-scale narrative of project activity.
2. **Problem sweep + fixes** — go through the WEEK's Claude Code session transcripts in detail and catch EVERY problem, confusion, or error that occurred — not just recurring patterns, not a top-5. Each workflow-fixable problem becomes a surgical proposed diff in `## Proposed workflow improvements` (PROPOSED ONLY — Thomas reviews, says "do 1, 3", `workflow-improver` applies the greenlit ones). Every other problem is logged in `## Other problems & notes` with a one-line suggested action, so nothing is dropped.

The weekly sweep uses the SAME detail bar as `/daily` — every distinct problem counts, with no recurrence requirement. The only differences from daily are the window (the whole ISO week) and the extra living-docs drift check. Weekly is the safety net that catches problems daily missed plus still-open problems from earlier in the week. Cross-reference the daily files to avoid re-nagging (see "Cross-reference daily files" below): already-fixed problems are skipped, and previously-declined ones are listed as notes rather than re-proposed.

## Inputs

- tasks: `python scripts/task.py list-by-status --limit 1000`
- Individual experiment state/events: `python scripts/task.py view <N>`
- Headline accepted claims: `RESULTS.md`
- Artifact index: `eval_results/INDEX.md`
- Research aims: `docs/research_ideas.md`
- Prior weekly artifacts under `logs/weekly/`, when available
- Prior daily artifacts under `logs/daily/` for the week — to see what was proposed and what got greenlit/declined
- **Claude Code session transcripts** under `~/.claude/projects/-home-thomasjiralerspong-explore-persona-space/*.jsonl` and `~/.claude/projects/-home-thomasjiralerspong-explore-persona-space--claude-worktrees-*/*.jsonl` — filter to files modified in the ISO week window

Group tasks by status (folder name under tasks/). Treat `events.jsonl` markers as the audit
trail for progress, reviewer rounds, reconciler decisions, promotion, and
failures.

## Process

1. Pull the week window in UTC.
2. Identify completed, awaiting-promotion, active, blocked, and proposed work
   from local files.
3. Read the relevant experiment bodies and latest `epm:*` events.
4. Cross-check accepted claims against `RESULTS.md` and artifact paths.
5. Draft a concise narrative with evidence and explicit caveats.
6. **Sweep the week's transcripts for problems**. Catch every distinct problem / confusion / error — no recurrence bar (same detail bar as `/daily`, just over the whole week). See "Problem sweep" section below for what to look for, the two-bucket triage, and the shape proposals take.
7. **Living-docs drift check**. Run `uv run python scripts/living_docs.py check` and capture its output + exit code. It lints the living hub (`docs/open_questions.md`) for `relates_to` ⇄ question-evidence mismatches, `completed` experiments with `has_clean_result` missing from any question's evidence, dangling evidence `#N`, and questions stale relative to new results. A nonzero exit = drift. Surface the findings in the `## Living-docs drift` section (see Output below). When drift is found, PROPOSE — do not run — an `open_questions.md` re-synthesis (the updater/backfill), consistent with the "every living-docs mutation is user-confirmed" rule. The script never mutates the docs; `check` is read-only.
8. **Critic-recurrence harvest (mechanization ratchet)**. Run `uv run python scripts/critic_mechanization_report.py` and capture its per-month summary (blockers tagged `mechanizable: yes` vs `no` vs untagged, plus the best-effort count of verifier-landing workflow fixes — the ratchet metric). Then sweep the week's critic FAIL-class verdicts (events.jsonl markers `epm:plan-critique*`, `epm:code-review*`, `epm:interp-critique*`, `epm:clean-result-critique*`; skip reconcile markers) and group findings into classes. Any finding class that recurred ≥2× across the week is a mechanization candidate: file it via the workflow-fix channel (`.claude/rules/workflow-fix-on-bug.md` § "Yes — emit", the critic-finding bullet) naming the target verifier (`verify_task_body.py`, `audit_clean_results_body_discipline.py`, SPEC.md lens text, the `consistency-checker` spec, or a future `verify_plan.py`) and the concrete check, AND record it as a numbered item under `## Proposed workflow improvements` so the weekly file keeps the ratchet visible. One-off artifact-specific findings are NOT candidates — the bar is a concrete check likely to recur. Dispatch-default note: mechanization candidates ride the workflow-fix channel's own defaults (auto-spawn for in-scope, non-architectural checks per `.claude/rules/workflow-fix-on-bug.md`) — the weekly numbered item is the visible record, NOT a second greenlight gate; the greenlight flow below governs only this skill's other (non-channel) proposals. Include the report's TOTAL row as a one-line note in `## What happened` (e.g. "mechanization ratchet: 12 yes / 4 no / 31 untagged blockers; 3 verifier checks landed").

## Output

Write the narrative to `logs/weekly/YYYY-Www.md` (relative to the repo root
— `~/explore-persona-space/`). One file per ISO week. Use Python's
`datetime.date(...).isocalendar()` to compute the ISO year + week number
(e.g. week 22 of 2026 → `logs/weekly/2026-W22.md`). If the file already
exists, refuse to overwrite and tell the user to edit it directly.

The file is a stub Thomas will finish editing. It starts hidden from the
`/log` dashboard feed (`visible: false`) and only becomes visible when he
flips the frontmatter field manually.

### Frontmatter

Every file MUST have this YAML frontmatter:

```yaml
---
kind: weekly
date: YYYY-MM-DD   # Monday of the ISO week
title: <auto-generated, one line — Thomas can edit>
included_tasks: [<task IDs from auto-population below>]
visible: false
---
```

- `date`: the Monday of the ISO week, in ISO format. Compute via
  `datetime.date.fromisocalendar(iso_year, iso_week, 1)`.
- `title`: a one-line auto-generated headline (e.g.
  `Weekly — 2026-W22 (<N> results promoted)`).
- `visible: false` ALWAYS at creation. Never set `true`. Thomas flips it manually.
- `included_tasks`: auto-populate from clean-results promoted this ISO week. Recipe:
  1. `uv run python scripts/task.py list-by-status --status completed --limit 500 --json`
     and keep rows where `has_clean_result == true`.
  2. For each surviving id, run `uv run python scripts/task.py view <N> --json`
     and read `frontmatter.promoted_at` (ISO UTC timestamp).
  3. Keep ids whose `promoted_at` falls in the current ISO week (Monday 00:00 UTC
     through Sunday 23:59:59 UTC).
  4. Legacy clean-results may have `promoted_at = None` — skip silently.

### Body (stub sections)

Below the frontmatter, write exactly these six H2 sections in this order:

```markdown
## What happened
<2-6 bullets: the week's task activity. Pull from epm:* markers, status
changes, completed reviews, promotions. Be concrete (mention task IDs).
This is the auto-drafted summary Thomas will edit down.>

## Proposed workflow improvements
<numbered list of WORKFLOW-FIXABLE problems from the week — each a
concrete proposed diff; see "Problem sweep" section below. No recurrence
bar. ALSO include one numbered item per critic-recurrence mechanization
candidate from Process step 8 (finding class recurring ≥2×, target
verifier, concrete check). If no workflow-fixable problems surfaced this
week, write a single line:
`- _no workflow-fixable problems found this week_`>

## Other problems & notes
<every problem/confusion/error from the week that did NOT map to a
workflow-file fix — experiment bugs, infra flakiness, mistakes I made,
dropped handoffs, anything Thomas fixed by hand. One bullet each: what
happened (session id / task id) + a one-line suggested action. NOT
greenlight-gated. If none, write:
`- _no other problems surfaced this week_`>

## Living-docs drift
<output of `scripts/living_docs.py check` (Process step 7). If the check
exited zero, write a single line: `- _no drift — open_questions.md is in
sync_`. If it found drift, list each finding as a bullet (the mismatch,
the missing-from-evidence task, the dangling `#N`, or the stale
question), then add one PROPOSAL line:
`- **Proposal:** re-synthesize open_questions.md (run the living-docs
updater/backfill) to reconcile the drift above — needs your ok; not
auto-applied.`
This is a proposal only — never run the re-synthesis from /weekly.>

## My thoughts
<leave empty — Thomas fills in>

## Highlighted results
- #<N> — <task title>
- #<M> — <task title>
```

`Highlighted results` starts as a one-line stub per `included_tasks` entry
(just the title from `view <N> --json` → `frontmatter.title`). If
`included_tasks` is empty, write a single bullet: `- _no results promoted this week_`.

### Problem sweep (what fills the two problem sections)

Go through the week's transcripts in detail. Same goal and same signal classes
as `/daily` — COVERAGE, not pattern-mining. Catch every distinct problem,
confusion, or error, even one-offs. No recurrence requirement.

Signals to hunt for (non-exhaustive — anything that went wrong counts):

- **User corrections** — "no", "don't", "stop", "wrong", "not what I meant", or Thomas significantly rewriting / redoing an artifact.
- **Confusions** — places I misread intent, went down the wrong path, needed re-steering, or asked something already answered.
- **Errors & failures** — tool-call errors, tracebacks, retries, crashes, OOMs, failed launches, failed reviews / reconciles.
- **Process mistakes** — skipped a step, ran steps out of order, missed an `/issue` gate, or overreached (acted where I should have asked).
- **Repeated explanations** — context re-explained that should live in a workflow file.
- **Stale references** — files / agents / skills / scripts that no longer exist.
- **Voice / register drift** — AI-slop, corporate-speak, invented jargon, opaque condition codes, template-copying.
- **Dropped handoffs / manual fixes** — info lost between agents, or anything Thomas had to do by hand.

**Triage each problem into one of two buckets** (identical rule to `/daily`):

1. **Workflow-fixable** → `## Proposed workflow improvements`, numbered, WITH a concrete diff. Greenlight-gated.
2. **Not workflow-fixable** → `## Other problems & notes`, one bullet: what happened (session / task id) + a one-line suggested action. No diff, not gated.

If the fix edits a file in the allowed-targets list below, it is bucket 1; otherwise bucket 2.

**Cross-reference daily files**. Read `logs/daily/YYYY-MM-DD.md` for each day in the week so weekly doesn't re-nag what daily already handled:
- A problem whose daily proposal was **greenlit / applied** → skip it (it's fixed).
- A problem whose daily proposal was **declined** → do NOT re-propose it as a fresh numbered diff; instead list it once under `## Other problems & notes` tagged `(previously declined YYYY-MM-DD)` so it stays visible without nagging. Re-propose it as a diff only if it recurred after the decline.
- A problem daily **never caught** → propose / log it fresh per the two-bucket rule.

Nothing the week surfaced gets silently dropped; the cross-reference only changes whether an item is a fresh proposal vs a note.

**Allowed target files** (same as /daily):
- `~/explore-persona-space/CLAUDE.md`
- `~/explore-persona-space/.claude/CLAUDE.md` (if present)
- `~/explore-persona-space/.claude/agents/*.md`
- `~/explore-persona-space/.claude/skills/**/SKILL.md`
- `~/explore-persona-space/.claude/rules/*.md`
- `~/explore-persona-space/.claude/workflow.yaml`

**Forbidden targets**:
- Hooks in `.claude/settings.json` — surface in `## My thoughts` for Thomas to wire via `/update-config`.
- Creating new agents or skills — surface as "consider creating X" with rationale, don't pre-draft.
- `scripts/*.py` orchestration — `workflow-improver`'s job after greenlight.

**Proposal shape**: same as /daily — numbered, with target / what / why / proposed diff.

```markdown
1. **Target:** `<file path>` — **what:** <one-line description>
   **Why:** <pattern observed across N sessions this week; quote the most representative excerpt>
   **Proposed edit:**
   ```diff
   - <old line if modifying or deleting>
   + <new line>
   ```
```

**No cap — be exhaustive.** List every workflow-fixable problem as its own
proposal and every other problem as its own note; never drop items to hit a
number. Order both sections by severity (Thomas's corrections / blockers
first; foundational files before niche ones; time-costly problems before
cosmetic). Group related small items under one proposal with sub-bullets if it
helps — grouping is fine, dropping is not.

### Greenlight flow

Thomas reads the proposals, replies with e.g. "do 1, 3" or "do all" or "skip 2 — let's discuss". The handling assistant spawns `workflow-improver` with the specific proposals. `workflow-improver` makes edits, runs `scripts/workflow_lint.py`, commits with message `workflow: apply weekly-proposed edits 1,3 (YYYY-Www)`.

Declined proposals stay in the weekly file as historical record.

### Commit

After writing the file, commit it so the dashboard picks it up:

```bash
git add logs/weekly/YYYY-Www.md
git commit -m "logs: weekly stub for YYYY-Www"
```

Do not push.

### Other rules

Never auto-promote clean results or move statuses as part of writing the weekly
narrative. If the user asks for a mutation, use `scripts/task.py` and
record a task workflow event with a short note.
