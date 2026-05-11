---
name: mentor-update-slides
description: Maintain a single persistent Marp deck for weekly research updates (mentor 1:1, lab meeting, advisor sync). HEADER (cover, objectives, project summary, agenda) is replaced each run; LOG entries are prepended with date dividers; APPENDIX accumulates backup slides. Outputs Marp markdown and (optional) PDF via marp-cli. Structure follows Hughes & Chua, Perez, Nanda, Sanders, and Alley assertion-evidence (see principles.md). Not for conference talks — use ml-paper-writing:presenting-conference-talks for those.
---

# mentor-update-slides

## When to use

User asks for a deck for a mentor 1:1, lab meeting, or weekly advisor sync. Examples:
- "build me slides for tomorrow's meeting with [advisor]"
- "weekly update deck"
- "lab meeting deck for the persona work this week"
- "mentor slides for the last 2 weeks"

**NOT for:**
- Conference talks → `ml-paper-writing:presenting-conference-talks` (needs a compiled paper).
- Posters or paper figures → `paper-plots`.
- Slides without a date window or audience scope → ask the user instead of guessing.

## Persistent-deck model

This skill maintains **one** deck per project at `figures/mentor-slides/deck.md`, not a new dated subdirectory each week. The deck has three anchored regions; every run reads the file (if it exists), regenerates / merges those regions, and writes it back atomically.

```
<!-- BEGIN HEADER -->
<!-- END HEADER -->        ← REPLACED each run (cover, objectives, project summary, agenda)

<!-- BEGIN LOG -->
<!-- END LOG -->           ← APPEND-ONLY, NEWEST FIRST. New week prepended at top.
                              Each weekly entry: "## Week of YYYY-MM-DD" divider + per-result slides.

<!-- BEGIN APPENDIX -->
<!-- END APPENDIX -->      ← ACCUMULATES. New reproducibility cards + backup slides
                              prepended; old content untouched.
```

**Why:** Hughes & Chua's "one evolving deck per project" gives the mentor continuity (a single shared link they can scroll back through), while the project-summary block at the top stays clean. The LOG is the durable research log; the HEADER is the evolving face. See `principles.md` § Persistent deck structure.

**First run:** if `deck.md` doesn't exist, the skill writes a fresh file with all three regions, the new week as the only LOG entry.

## Inputs

- **Time window** — default trailing 7 days. Override via user phrasing ("last 2 weeks", "since #260").
- **Audience** — default `mentor`. Alternatives: `lab` (longer recap, broader context), `1on1` (shorter, decisions-focused).
- **Objective** — default `inform` (see Objectives slide in principles.md). Override via user phrasing ("I want advice on whether to ...", "broad direction discussion").
- **Auto-discovered source artifacts:**
  - GitHub issues with label `clean-results` or `clean-results:draft`, updated in window.
  - `RESULTS.md` (read top 100 lines for cross-experiment claims).
  - `git log --since=<window>` for commit context.
  - Existing `figures/mentor-slides/deck.md` for HEADER/LOG/APPENDIX merge.

## Output

- `figures/mentor-slides/deck.md` (single persistent Marp source — overwritten atomically each run)
- `figures/mentor-slides/deck.pdf` (rendered, if `--pdf` requested)

**Do NOT auto-commit.** The user reviews before committing.

## Steps

### 1. Resolve window, audience, objective

Today's date in `YYYY-MM-DD`. Default window: trailing 7 days. If the user said "since #N", resolve `since_date = max(N.created_at, today - 14d)` to bound the deck. Default objective: `inform`. The objective string drops into the Objectives slide verbatim.

### 2. Pull source artifacts

```bash
WINDOW_START=$(date -d '7 days ago' +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)
DECK=figures/mentor-slides/deck.md
mkdir -p figures/mentor-slides
SCRATCH=$(mktemp -d)

# Recent clean-results (final + drafts)
gh issue list --repo superkaiba/explore-persona-space \
  --label "clean-results" --state all --limit 50 \
  --search "updated:>=$WINDOW_START" \
  --json number,title,body,labels,updatedAt > "$SCRATCH/_clean-results.json"

gh issue list --repo superkaiba/explore-persona-space \
  --label "clean-results:draft" --state all --limit 50 \
  --search "updated:>=$WINDOW_START" \
  --json number,title,body,labels,updatedAt > "$SCRATCH/_clean-result-drafts.json"

# Commits + open follow-ups
git log --oneline --since="$WINDOW_START" --no-merges > "$SCRATCH/_commits.txt"
gh issue list --repo superkaiba/explore-persona-space \
  --label "status:proposed" --state open --limit 20 \
  --json number,title > "$SCRATCH/_proposed.json"

head -100 RESULTS.md > "$SCRATCH/_results-head.md"
```

### 3. Parse each clean-result issue

For each issue body, extract by markdown structure (NOT regex on prose):

| Field | Source in clean-result template |
|---|---|
| `headline_claim` | Issue title (already includes confidence — strip the `(HIGH|MODERATE|LOW confidence)` suffix). |
| `confidence` | Title's confidence suffix, OR the `**Confidence: HIGH \| MODERATE \| LOW** — …` line in `### Results`. |
| `hero_figure_path` | First `![alt](figures/...)` inside `### Results`. |
| `result_one_line` | First sentence after the hero figure in `### Results` (Hughes/Chua "headline below the figure"). |
| `main_takeaways` | The `**Main takeaways:**` bullets — pick the top 1-2. |
| `caveat_one_line` | First bullet under `Standing caveats:` block in `## Headline numbers`. |
| `n` | Numeric N from the headline-numbers table or eval row. |
| `setup_one_line` | One-line distillation of `## Setup & hyper-parameters`'s opening prose. |
| `commit_full` | `git_commit` field in `Setup & hyper-parameters`. |
| `seed`, `dataset_version`, `eval_n`, `config_path`, `wandb_url`, `hf_url` | Reproducibility card fields. |
| `next_steps` | The `### Next steps` bullets in TL;DR. |
| `metric_definition` | The opening prose of `## Setup & hyper-parameters` if it defines what's being measured (used for backup family A). |
| `prompt_excerpt` | A representative prompt + sample completion from `## Sample outputs` (used for backup family B). |
| `data_scaling_data` | If the issue's `## Headline numbers` table includes a "data fraction" or "training-step sweep" axis, extract it (used for backup family C). |
| `baseline_columns` | Any baseline rows in the headline-numbers table (used for backup family D). |

If a field is missing, leave the slot empty in the template AND flag it in the post-render report (see Step 7) — never fabricate.

### 4. Read existing deck and split into regions

If `figures/mentor-slides/deck.md` exists, slice it on the anchor comments into three regions: `HEADER_OLD`, `LOG_OLD`, `APPENDIX_OLD`. Preserve everything between `<!-- BEGIN LOG -->` and `<!-- END LOG -->` exactly. Same for APPENDIX. The frontmatter sits ABOVE `<!-- BEGIN HEADER -->`.

If `deck.md` does not exist, treat all three regions as empty strings and emit fresh frontmatter.

**Idempotency rule.** If a `## Week of $TODAY` divider already exists at the top of LOG_OLD (a same-day re-run), REPLACE that week's block instead of prepending a duplicate. Identify the block end at the next `## Week of ` divider or the LOG terminator.

### 5. Compose new HEADER and new-week LOG block

Read `template.md` from this skill directory. Fill in slot-by-slot.

**HEADER region** (in this order — each is one slide):
1. **Cover** — project name, presenter (git config `user.name`), audience, today's date as the deck's "current as of".
2. **Objectives** — one slide framing what the user wants from the meeting. Default: `inform`. Pulls the user-supplied `--objective` string. Hughes/Chua + Sanders: name the desired output of the meeting before the agenda. (`principles.md` § Objectives slide.)
3. **Project summary** — current evolving state in three blocks: (a) thesis question (one sentence), (b) most-recent claims (3-5 bullets pulled from the most recent clean-results, each bolded with confidence label and a `→ Week of YYYY-MM-DD` link into the LOG anchor below), (c) what's currently running (open `status:running` / `status:approved` issues). This is the "clean evolving summary" the user asked for.
4. **Agenda this week** — section names + slide counts + minute budget per section. Hughes/Chua signature; do NOT skip.

**LOG new-week block** — prepend at the top of LOG_OLD:
1. **Date divider** — `## Week of $TODAY` as a full-bleed section divider, with a `<a id="week-$TODAY"></a>` anchor so HTML links from the Project summary slide can target it (`week-2026-05-07`).
2. **One slide per result** (5..N), ordered by confidence then chronologically:
   - Title = `headline_claim` (≤12 words, full sentence with effect number).
   - `![bg right:50%]({{hero_figure_path}})`.
   - Three bullets: Setup / Result (with N) / Caveat.
   - Footer `<small>` line: commit hash · Issue #N · Confidence label.
3. **Open questions for {{audience}}** — 3 bullets max, framed as decisions needed (not "things I'm unsure about"). Source: open `status:proposed` issues + any `[Hypothesis: ...]` markers in clean-result bodies that lack approval.
4. **Next week's plan** — numbered, ≤5 items. Each: action + expected information gain (1 line). Source: clean-result `### Next steps` bullets, prioritized.

**APPENDIX new content** — prepend at the top of APPENDIX_OLD:
1. **Reproducibility cards** — one card per result this week (config, seed, commit, dataset version, eval N, WandB run, HF model).
2. **Backup-slide families** (conditional — emit only if the source data exists for at least one this-week result; never fabricate). See `principles.md` § Backup-slide families.
   - **(a) Metric definition + concrete example** — emit if `metric_definition` was extracted. One slide per metric: prose definition + a real prompt → real completion example (Sanders: real data guards against being fooled by abstractions).
   - **(b) Detailed prompt with highlights** — emit if `prompt_excerpt` was extracted. One slide per representative prompt with the load-bearing region highlighted (`<mark>` HTML), and a one-arrow takeaway in the caption.
   - **(c) Data-scaling curve** — emit if `data_scaling_data` was extracted. One slide showing the scaling sweep with linear and (where data spans ≥2 orders of magnitude) log-log views.
   - **(d) Baseline-invalidation slide** — emit if `baseline_columns` was extracted. One slide listing the controls + 1-line on what each rules out, plus the headline numbers from the baseline rows.

**Edge cases:**
- **Zero clean-results in window** → still emit a HEADER refresh (cover, objectives, summary, agenda) but the LOG new-week block becomes a single "infrastructure week" slide pulling from commit messages. NO per-result slides for this week. APPENDIX unchanged.
- **One clean-result** → minimum new-week LOG block = divider + 1 result + open-questions + next-steps. Skip the per-week agenda subsection if it would only have one row.
- **>8 clean-results in window** → ask the user to triage (which 4-6 to feature) BEFORE assembling. Don't silently drop.

### 6. Quality checklist (run BEFORE rendering)

Self-audit the draft. If any check fails, fix before render:

- [ ] Frontmatter is intact and unchanged (Marp options, theme, math).
- [ ] All three anchor pairs are present and balanced.
- [ ] Project summary links resolve to existing `week-YYYY-MM-DD` anchors in LOG.
- [ ] Every results-slide title is a full sentence with an effect number, ≤12 words.
- [ ] Every chart has N stated (in caption or footer).
- [ ] No effect-size jargon (Cohen's d, η², r-as-effect, Δ-framed-as-effect).
- [ ] No raw p-values without N.
- [ ] No `value ± err` in prose. (Allowed on charts.)
- [ ] ≤3 colors / ≤3 model series per chart.
- [ ] For paired comparisons, the chart shows error bars on the *delta*, not on each endpoint separately. (Sanders.)
- [ ] Confidence label present on every results slide.
- [ ] One headline message per slide.
- [ ] Footer carries commit hash + issue link on every results slide.
- [ ] Open-questions slide frames each item as a decision needed (not a worry).
- [ ] No backup-slide family was fabricated — every emitted family had real source data.
- [ ] LOG order: newest week at top, older weeks below, no duplicates.

If a structural fix is impossible (e.g., a clean-result chart has 5 baselines and recoloring would lose meaning), DO NOT silently leave it — add a `<!-- FLAG: ... -->` HTML comment in the markdown so the user sees it on review.

### 7. Render

```bash
npx --yes @marp-team/marp-cli@latest \
  "$DECK" \
  --pdf --allow-local-files \
  -o "${DECK%.md}.pdf"
```

If marp-cli's first run is slow (downloads), tell the user; subsequent runs are fast. If `--pdf` was not requested, skip rendering.

### 8. Report

Tell the user:
- `figures/mentor-slides/deck.md` (and `deck.pdf` path if rendered).
- Number of new result slides prepended to LOG.
- Number of weeks now in the LOG (i.e., archive depth).
- Backup-slide families emitted this week, and which ones were skipped because data was missing.
- Quality-checklist items that were FLAGGED but couldn't be auto-fixed (with line numbers).
- Suggest: review the deck, edit if needed, then `git add figures/mentor-slides/deck.{md,pdf} && git commit`.

## Anti-patterns this skill enforces

- **No "Results" / "Background" / "Next Steps" topic-label titles.** Every slide title is a sentence asserting a finding. (Alley assertion-evidence; Naegle "Ten Simple Rules" #3.)
- **No bullet-list-only slides for empirical results.** Bullets accompany a figure or table; never substitute. (Alley.)
- **No effect-size jargon.** Project rule + general ML talk hygiene. (Nanda; project `CLAUDE.md`.)
- **No conflating "we tried X" with "X works".** Confidence label is mandatory.
- **No skipping the agenda slide.** Hughes & Chua: "section names + slide counts + time per section" so the mentor knows the time budget upfront.
- **No skipping the Objectives slide.** Sanders: "are you here to inform / get advice on a specific decision / get advice on a broad direction?"
- **No "summary of what I did."** Open with what was *learned* (Perez: "predictions vs. findings"), not what was *attempted*.
- **No dated-subdirectory decks.** One persistent file per project; archive lives in the LOG region of that file. The skill MUST NOT write to `figures/mentor-slides/<date>/`.
- **No fabricated backup slides.** Every backup family is conditional on real source data; missing data is reported, never inferred.
- **No clobbering LOG history.** The skill is idempotent on a same-day re-run (replace today's block) but never rewrites earlier weeks.

## Confidence framing vocabulary (Nanda)

Use these in headlines and Main takeaways when the chart underdetermines the claim:

- **Existence proof** — observed at least once (1 seed, 1 setting).
- **Systematic** — across a wide range of contexts (multiple seeds + settings).
- **Hedged** — compelling/suggestive/tentative.
- **Narrow** — restricted to specific setting.
- **Guarantees** — always true (rare in deep learning; use sparingly).

The clean-result issue's `Confidence: HIGH | MODERATE | LOW` already maps onto this. Reuse the issue's verdict verbatim on the slide footer.

## Tooling notes

- **Default theme**: Marp's built-in `default`. Optional upgrade: install `kaisugi/marp-theme-academic` (Beamer-like) by symlinking the CSS into the project; reference via `theme: academic` in frontmatter.
- **Math**: KaTeX via `math: katex` frontmatter directive.
- **Two-column layouts**: Marpit's `<!-- _class: split -->` directive, or `![bg right:50%]` for figure-on-right.
- **Date-divider styling**: use `<!-- _class: divider -->` plus a CSS rule in the frontmatter `style:` block (template.md ships a default).
- **Render alternatives**: Slidev (richer layouts, heavier toolchain) or reveal-md (lighter, less reliable headless PDF). Stick with Marp unless the user explicitly asks otherwise.

## See also

- `template.md` (this dir) — Marp markdown skeleton with the three anchor regions.
- `principles.md` (this dir) — sources + rationale for each rule.
- `.claude/skills/clean-results/SPEC.md` — source format the skill reads from.
- `.claude/skills/paper-plots/SKILL.md` — figures the deck embeds.
- `ml-paper-writing:presenting-conference-talks` — different use case (compiled paper → conference deck).
