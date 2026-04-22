---
name: mentor-prep
description: >
  Assemble recent [Clean Result] GitHub issues into a single screen-shareable
  mentor-meeting document. Reads issues labeled `clean-results` from a
  project-board column or since a date, extracts each TL;DR, orders by aim,
  and emits a markdown agenda with inlined hero figures + open questions.
  Invoke as `/mentor-prep --since YYYY-MM-DD` or `/mentor-prep --column "Clean Results"`.
user_invocable: true
---

# Mentor-Prep

Turns a pile of `[Clean Result]` GitHub issues into a single screen-shareable
mentor-meeting document.

**Scope:** assembly only. This skill does NOT author new content. It
extracts and reorders what already exists on the issues. If a clean-result
issue is missing a TL;DR section, `mentor-prep` flags the gap in its output
rather than filling it in.

---

## Invocation forms

| Form | Meaning |
|---|---|
| `/mentor-prep --since 2026-04-15` | all `clean-results`-labeled issues **updated** since that date |
| `/mentor-prep --column "Clean Results"` | all issues in that project-board column (default column) |
| `/mentor-prep --column "Clean Results (draft)"` | draft-stage clean results, useful for a mid-week pre-read |
| `/mentor-prep --issues 65,67,70` | explicit comma-separated list |
| `/mentor-prep --since 2026-04-15 --aim 5` | filter further by `aim:*` label |

Defaults:
- Without any flag: `--column "Clean Results" --since <last-meeting-date>`
  (reads `research_log/mentor_meetings/` for the most recent file; if none,
  falls back to 7 days).

---

## Output

Writes to `research_log/mentor_meetings/YYYY-MM-DD.md`, where `YYYY-MM-DD`
is today's UTC date. Creates the parent directory if missing.

Document structure (summary-first, per Chua/Hughes in
`.claude/skills/clean-results/principles.md`):

```markdown
# Mentor Meeting — YYYY-MM-DD

## Summary slide

<One line per clean-result, with a success/fail icon + the one-sentence
claim + issue number. This is the ONLY thing the mentor sees first.>

- ✅ #67 — Tulu midtraining preserves capability but not alignment (aim 5)
- ⚠️  #70 — ZLT single-token sweep: phase-transition confirmed but single-seed (aim 5)
- ❌  #73 — Contrastive-design leakage experiment: inconclusive, 4/5 seeds (aim 3)

## Agenda

<Sections in priority order (✅ highest-confidence first, ❌ inconclusive
last). Each section = the full TL;DR (6 H3 subsections) of that
clean-result issue, hero figure inlined via the raw-github URL.>

### 1. #67 — Tulu midtraining …

<full TL;DR pasted verbatim from the issue body, from ## TL;DR through
the horizontal rule before # Detailed report>

[→ Full report](https://github.com/superkaiba/explore-persona-space/issues/67)

### 2. #70 — …

## Open questions / asks

<Aggregated "Next steps" subsections, deduplicated where possible, ranked
across all clean-results by information gain per GPU-hour.>

## Backup slides

<One-line per clean-result, linking to its "# Detailed report" section for
drill-down during Q&A.>

- #67 — Detailed report: https://github.com/.../issues/67#user-content-detailed-report
- #70 — Detailed report: …
```

---

## Assembly procedure

### Step 1: Fetch issues

Use `gh` (same patterns as `.claude/skills/issue/SKILL.md`):

```bash
# By label + date
gh issue list --label clean-results --state open --search 'updated:>=2026-04-15' \
  --json number,title,body,labels,updatedAt --limit 100

# By project column
uv run python scripts/gh_project.py list-by-status "Clean Results"
# (If that helper doesn't support list-by-status, fall back to GraphQL via gh api.)
```

For each issue, pull the full body via `gh issue view <N> --json number,title,body,labels,updatedAt`.

### Step 2: Parse each clean-result body

For each issue body, extract:
- **TL;DR block** — everything between `## TL;DR` and the horizontal rule
  (`---`) that precedes `# Detailed report`.
- **6 subsections** — `### Background`, `### Methodology`, `### Results`,
  `### How this updates me + confidence`, `### Why confidence is where it is`,
  `### Next steps`. Warn in the output if any are missing.
- **Hero figure URL** — the first raw.githubusercontent.com markdown image
  under `### Results`.
- **Success/fail icon** — derive from the primary confidence tag in "How
  this updates me + confidence":
  - ✅ if the lead bullet is HIGH
  - ⚠️  if the lead bullet is MODERATE
  - ❌ if the lead bullet is LOW or if the issue has a CRITICAL caveat
- **Aim** — from the `aim:*` label.
- **Next steps** — the `### Next steps` subsection, to aggregate into the
  final "Open questions / asks" block.

### Step 3: Order

Primary sort: confidence tag (HIGH → MODERATE → LOW).
Secondary sort: updated-at descending (most recent first within a tier).
Tertiary sort: aim number ascending.

### Step 4: Write

Write the assembled document to `research_log/mentor_meetings/<today>.md`.
If the target file already exists, write to `<today>_v2.md`, `_v3.md`, etc.

Print to stdout:
- The output path
- A count: N clean-results assembled
- Any warnings (missing TL;DR subsections, missing hero figures)

### Step 5: Length check

The rendered document should be readable in ≤ 10 minutes (Joe Benton
weekly-meeting principle). If more than 8 clean-results are being
assembled, warn and suggest either `--aim <N>` filtering or splitting into
two meetings.

---

## When to use

- Weekly mentor meeting prep. The clean-result issue is the weekly-meeting
  artifact; this skill just staples several together.
- Mid-week pre-read (`--column "Clean Results (draft)"`) so the mentor has
  context before Monday.
- End-of-sprint retrospective (`--since <sprint-start>`).

## When NOT to use

- Daily standups. Use `/daily-update` instead — those are git-log-driven,
  not issue-driven.
- When no clean-results exist in the window. Say so and stop; don't fabricate
  a fake document.
- Cross-project summaries. This skill only reads this repo's issues.

---

## Pitfalls

- **Don't edit the source issues.** Mentor-prep is read-only from GitHub's
  perspective. If a clean-result has a broken hero-figure link, fix it on
  the source issue, then re-run mentor-prep.
- **Don't paraphrase the TL;DR.** Copy verbatim. The whole point of the
  clean-result workflow is that the TL;DR is already mentor-ready.
- **Don't auto-send the document anywhere.** User reads, reviews, and
  shares manually.
