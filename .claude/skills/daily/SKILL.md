---
name: daily
description: End-of-day Explore Persona Space brief — what happened today, plus an exhaustive sweep of every problem/confusion/error in the day's Claude Code session transcripts, each with a concrete fix. Workflow-fixable problems are AUTO-APPLIED (each lint-checked and committed separately so it stays revertable) and summarized to Thomas's my-goat Telegram; every other problem is logged with a suggested action so nothing is dropped.
---

# Daily Brief

Use `tasks/` as the only workflow state source. Do not read or mutate queue,
status, promotion, or approval state through any external tracker.

Two jobs in one file:
1. **Recap** — what happened on the project today.
2. **Problem sweep + auto-fix** — go through today's Claude Code session transcripts in detail and catch EVERY problem, confusion, or error that occurred — not just recurring patterns, not just a top-5. Each problem that maps to a workflow-file fix (within the allowed-targets list below) is **AUTO-APPLIED in this run**: make the edit and `git commit` it on its own (one commit per fix, so each is independently revertable), then run the repo-wide workflow lint ONCE after all fixes (see "Lint gate" below — `workflow_lint.py` is a repo-wide validator, NOT a per-file `.md` linter). Record each applied fix in `## Applied workflow improvements` with its diff and commit sha. Then push a concise summary of what was applied (with the commit shas, so Thomas can revert any) to Thomas's my-goat Telegram chat (see "Auto-apply + surfacing flow"). Every other problem (experiment bug, infra flakiness, a mistake I made, a dropped handoff, or a forbidden-target fix that is too high blast-radius to auto-apply) is logged in `## Other problems & notes` with a one-line suggested action, so nothing is silently dropped.

   **Auto-apply replaces the old greenlight gate (changed 2026-06-08 per Thomas: "make the workflow improvements automatically and just surface them in this chat").** Earlier this skill drafted PROPOSED diffs and waited for Thomas to say "do 1, 3"; now workflow-fixable fixes apply themselves and Thomas reviews after the fact via the Telegram summary (and can revert any single fix by its commit sha). The safety posture is "apply but stay fully transparent + per-fix revertable", not "apply silently".

## Inputs

Read:

- tasks and workflow events via `scripts/task.py`;
- `RESULTS.md` for accepted headline claims;
- `eval_results/INDEX.md` for artifact inventory;
- `docs/research_ideas.md` for aims and phase framing;
- local run logs only as supporting evidence, never as workflow state;
- **Claude Code session transcripts** under `~/.claude/projects/-home-thomasjiralerspong-explore-persona-space/*.jsonl` and `~/.claude/projects/-home-thomasjiralerspong-explore-persona-space--claude-worktrees-*/*.jsonl` — filter to files modified today (UTC).

Useful commands:

```bash
export PATH="$HOME/.local/bin:$PATH"   # uv lives in ~/.local/bin; non-login (cron) shells miss it
uv run python scripts/task.py list-by-status --limit 500
uv run python scripts/task.py list-by-status --status running --limit 100
uv run python scripts/task.py list-by-status --status uploading --limit 100
uv run python scripts/task.py list-by-status --status awaiting_promotion --limit 100
uv run python scripts/task.py view <N>
```

## Output

Write the brief to `logs/daily/YYYY-MM-DD.md` (relative to the repo root —
`~/explore-persona-space/`). One file per date. The file is a written RECORD
of what the run applied + noted; the actual surfacing to Thomas happens over
Telegram (see "Auto-apply + surfacing flow"), not via this file. Handle an
existing file as follows:

- **File does not exist** → write the full stub (all five H2 sections below).
- **File exists but is missing the `## Applied workflow improvements` H2, or
  that section is empty** → do NOT overwrite the file. `Edit` it to insert /
  fill the `## Applied workflow improvements` section in place (between
  `## What happened` and `## Other problems & notes`, or in the correct
  position if those are absent), and likewise insert `## Other problems &
  notes` if it is missing. Leave every other section — including any edits
  Thomas already made to `## What happened` / `## My thoughts` — untouched.
  This is the recovery path when an earlier manual or partial run left a stub
  without the problem sweep.
- **File exists with a non-empty `## Applied workflow improvements` section**
  (real applied edits OR the "no workflow-fixable problems" placeholder) →
  the day's auto-apply already ran; do NOT re-apply or re-overwrite (re-running
  would double-apply fixes). Refuse to overwrite and tell the user the day is
  already done.

**Manual runs write the FULL file AND auto-apply.** When `/daily` is invoked
manually (not via the nightly cron), always produce EVERY section including
`## Applied workflow improvements` and actually apply the fixes — the 23:27 PT
cron refuses to overwrite an existing daily file, so a partial manual run
permanently loses that day's problem-sweep + auto-apply.

The file is a stub Thomas will finish editing. It starts hidden from the
`/log` dashboard feed (`visible: false`) and only becomes visible when he
flips the frontmatter field manually.

### Frontmatter

Every file MUST have this YAML frontmatter:

```yaml
---
kind: daily
date: YYYY-MM-DD
title: <auto-generated, one line — Thomas can edit>
included_tasks: [<task IDs from auto-population below>]
visible: false
---
```

- `date`: today in ISO format.
- `title`: a one-line auto-generated headline (e.g. `Daily — <date> (<N> results promoted)`).
- `visible: false` ALWAYS at creation. Never set `true`. Thomas flips it manually.
- `included_tasks`: auto-populate from clean-results promoted today. Recipe:
  1. `uv run python scripts/task.py list-by-status --status completed --limit 500 --json`
     and keep rows where `has_clean_result == true`.
  2. For each surviving id, run `uv run python scripts/task.py view <N> --json`
     and read `frontmatter.promoted_at` (ISO UTC timestamp).
  3. Keep ids whose `promoted_at` falls on today's UTC date.
  4. Legacy clean-results may have `promoted_at = None` — skip silently.

### Body (stub sections)

Below the frontmatter, write exactly these five H2 sections in this order:

```markdown
## What happened
<2-5 bullets: today's task activity. Pull from epm:* markers, status
changes, completed reviews. Be concrete (mention task IDs). This is the
auto-drafted summary Thomas will edit down.>

## Applied workflow improvements
<numbered list of WORKFLOW-FIXABLE problems that were AUTO-APPLIED this run —
each with its applied diff and commit sha; see "Problem sweep" below for the
shape. If no workflow-fixable problems surfaced today, write a single line:
`- _no workflow-fixable problems found today_`>

## Other problems & notes
<every problem/confusion/error from today that did NOT map to an
auto-applied workflow-file fix — experiment bugs, infra flakiness, mistakes I
made, dropped handoffs, anything Thomas had to fix by hand, plus any
forbidden-target or lint-failed fix that was NOT applied. One bullet each:
what happened (session id / task id) + a one-line suggested action.
These are notes, not applied edits. If none, write:
`- _no other problems surfaced today_`>

## My thoughts
<leave empty — Thomas fills in>

## Highlighted results
- #<N> — <task title>
- #<M> — <task title>
```

`Highlighted results` starts as a one-line stub per `included_tasks` entry
(just the title from `view <N> --json` → `frontmatter.title`). If
`included_tasks` is empty, write a single bullet: `- _no results promoted today_`.

### Problem sweep (what fills the two problem sections)

Go through today's transcripts in detail. The goal is COVERAGE, not pattern-
mining: catch every distinct problem, confusion, or error that occurred, even
if it happened exactly once. Do not require recurrence. Do not dedupe a real
problem away because it "probably won't happen again."

Signals to hunt for (non-exhaustive — anything that went wrong counts):

- **User corrections** — "no", "don't", "stop", "wrong", "not what I meant", or Thomas significantly rewriting / redoing an artifact I produced.
- **Confusions** — places I misread intent, went down the wrong path, needed re-steering, or asked a question whose answer was already available.
- **Errors & failures** — tool-call errors, tracebacks, retries (same tool 3+ times), crashes, OOMs, failed launches, failed reviews / reconciles.
- **Process mistakes** — skipped a step, ran steps out of order, missed one of the enumerated `/issue` gates, OR overreached (acted where I should have asked, e.g. auto-applied a workflow edit).
- **Repeated explanations** — context I needed re-explained that should already live in a workflow file ("I keep telling you about X").
- **Stale references** — task / agent / skill / script names that no longer exist (cross-check the current `.claude/` tree).
- **Voice / register drift** — corporate-speak, AI-slop vocab, invented jargon, opaque condition codes, or template-copying instead of plain-English.
- **Dropped handoffs / manual fixes** — information lost between agents, or anything Thomas had to do by hand that an agent should have done.

**Triage each problem into one of two buckets:**

1. **Workflow-fixable** (the fix edits an allowed-target file below) → APPLY it now (Edit the file), `git commit` it on its own, then record it in `## Applied workflow improvements` as a numbered entry WITH the applied diff and the commit sha (shape below). One commit per fix so each is independently revertable. After ALL bucket-1 fixes are committed, run the repo-wide lint gate ONCE (see "Lint gate" below); if it regresses, revert the offending commit(s) and re-log them in `## Other problems & notes` as "reverted: failed lint gate".
2. **Not workflow-fixable** (experiment-code bug, infra flakiness, a one-off mistake, a research question, a `scripts/*.py` experiment-entrypoint bug, or a forbidden-target fix) → goes in `## Other problems & notes` as a bullet: what happened (session id / task id) + a one-line suggested action (file an issue, retry on a fresh pod, fix via `experiment-implementer`, etc.). No diff, not applied — it is a note so the problem stays visible and nothing is dropped.

When unsure which bucket: if the fix edits a file in the allowed-targets list below, it is bucket 1; otherwise bucket 2.

**Allowed target files** (project workflow only — global files are handled by `/memory-sleep`):
- `~/explore-persona-space/CLAUDE.md`
- `~/explore-persona-space/.claude/CLAUDE.md` (if present)
- `~/explore-persona-space/.claude/agents/*.md`
- `~/explore-persona-space/.claude/skills/**/SKILL.md`
- `~/explore-persona-space/.claude/rules/*.md`
- `~/explore-persona-space/.claude/workflow.yaml`

**Forbidden targets** (too high blast radius — do NOT auto-apply; log in `## Other problems & notes` and, where relevant, `## My thoughts`):
- Hooks in `.claude/settings.json` — surface as a written suggestion in `## My thoughts` for Thomas to wire up via `/update-config`.
- Creating new agents or skills — surface as a "consider creating X" line in `## Other problems & notes` but don't pre-draft the file.
- `scripts/*.py` orchestration code — surface as a note; non-trivial orchestration changes are still Thomas's call, not a daily auto-apply.

**Applied-edit record shape**: each applied fix is a numbered list item with this structure (written AFTER the edit + lint + commit succeed):

```markdown
1. **Target:** `<file path>` — **what:** <one-line description> — **commit:** `<sha>`
   **Why:** <triggering pattern, quoted transcript excerpt with session ID if possible>
   **Applied edit:**
   ```diff
   - <old line if modifying or deleting>
   + <new line>
   ```
```

**No cap — be exhaustive.** Apply + record every workflow-fixable problem as its
own entry and every other problem as its own note. Do NOT drop items to hit a
number. Order both sections by severity so the important ones are on top
(rules of thumb: Thomas's own corrections / blockers first; foundational files
like project CLAUDE.md before niche skill files; problems that cost real time
before cosmetic ones). If several small related items share one fix, you may
group them under a single applied entry (one commit) with sub-bullets — grouping
is fine, dropping is not.

### Lint gate

`workflow_lint.py` is a **repo-wide** validator, not a per-file `.md` linter (its `--file` flag only points at `workflow.yaml`; passing an `.md` path makes it try to parse that file AS workflow.yaml and falsely fail). So do NOT run it "per touched file". Instead, after ALL bucket-1 fixes are committed, run it ONCE for the whole repo:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run python scripts/workflow_lint.py --check-references
```

- `--check-references` is the gate (it currently PASSes clean, so a new failure means a just-applied edit broke a workflow reference). Use the `uv run python …` form — the linter imports pydantic/PyYAML and needs the EPS venv; a bare `scripts/workflow_lint.py` in the cron shell will `ModuleNotFoundError`.
- Do NOT gate on `--check-asks`: it has pre-existing, unrelated failures (bare `AskUserQuestion` mentions in `issue/SKILL.md`), so treating any `--check-asks` failure as "my edit broke it" would wrongly revert good edits. Only re-run `--check-asks` if one of THIS run's edits itself adds an `AskUserQuestion` mention.
- **On regression** (`--check-references` was clean and is now failing): the failure is from a just-applied edit. Identify the offending commit, `git revert --no-edit <sha>` it (do not hand-edit), move that item to `## Other problems & notes` as "reverted: failed lint gate (<error>)", and re-run the gate until it is green again. Then continue to surfacing.

### Auto-apply + surfacing flow

The fixes apply themselves during the run (bucket 1 above): edit → `git commit` (one commit per fix) → repo-wide lint gate ONCE (see "Lint gate"). After all fixes are applied and the daily file is written, **surface a concise summary to Thomas's my-goat Telegram chat** by enqueuing it into the my-goat notification digest:

```bash
NOTIF_CAT=research /home/thomasjiralerspong/my-goat/scripts/notif_enqueue.sh "EPS daily <date>: auto-applied N workflow fix(es). 1) <one-liner> (<sha>). 2) <one-liner> (<sha>). Notes: <M> other problems logged. Revert any with: git -C ~/explore-persona-space revert <sha>. Full: logs/daily/<date>.md"
```

This lands in the next my-goat morning digest (the dispatch cron runs 9/14/19 PT), so the overnight `23:27 PT` run is reviewed when Thomas is fresh rather than buzzing him at bedtime. Keep the message short: count of fixes, a one-liner + sha each (so any fix is one `git revert <sha>` away), the count of other notes, and the daily-file path. If zero fixes were applied AND zero notable problems were logged, enqueue nothing (don't send an empty digest line).

The old `SessionStart` greenlight hook (`scripts/daily_surface_hook.sh`) is now vestigial: it greps for `## Proposed workflow improvements`, which this skill no longer writes, so it stays silent and never prompts for a greenlight. Leave it in place (harmless); do not edit `.claude/settings.json` from this skill (forbidden target). Surfacing is Telegram-only now.

Applied edits stay in the daily file as historical record — don't delete them. If Thomas reverts one, that's via git; the record stays.

### Commit

After writing the file, commit it so the dashboard picks it up:

```bash
git add -f logs/daily/YYYY-MM-DD.md   # logs/ is gitignored; force-add or the commit silently stages nothing
git commit -m "logs: daily stub for YYYY-MM-DD"
```

Do not push. **`logs/` is in `.gitignore`** — a bare `git add logs/...`
stages nothing and `git commit` reports "no changes added to commit", so
the daily never lands in git or the dashboard. `-f` is required (the
prior dailies are tracked only because they were force-added).

### Other rules

Do not promote clean results, create experiments, or move statuses unless the
user explicitly asks for that mutation in the current session. If asked to
mutate, use `scripts/task.py` so the change goes through the canonical
API and leaves a workflow event.
