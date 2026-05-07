---
name: mentor-update-slides
description: Generate a Marp deck for weekly research updates (mentor 1:1, lab meeting, advisor sync) from recent clean-result GitHub issues + RESULTS.md + commits. Use when the user asks for "weekly slides", "mentor update deck", "lab meeting deck", or similar phrasing. Outputs Marp markdown and (optional) PDF via marp-cli. Structure follows Hughes & Chua, Perez, Nanda, and Alley assertion-evidence (see principles.md). Not for conference talks — use ml-paper-writing:presenting-conference-talks for those.
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

## Inputs

- **Time window** — default trailing 7 days. Override via user phrasing ("last 2 weeks", "since #260").
- **Audience** — default `mentor`. Alternatives: `lab` (longer recap, broader context), `1on1` (shorter, decisions-focused).
- **Auto-discovered source artifacts:**
  - GitHub issues with label `clean-results` or `clean-results:draft`, updated in window.
  - `RESULTS.md` (read top 100 lines for cross-experiment claims).
  - `git log --since=<window>` for commit context.
  - Optional: `figures/mentor-slides/<previous-date>/deck.md` for the recap slide.

## Output

- `figures/mentor-slides/YYYY-MM-DD/deck.md` (Marp source)
- `figures/mentor-slides/YYYY-MM-DD/deck.pdf` (rendered, if `--pdf` requested)

**Do NOT auto-commit.** The user reviews before committing.

## Steps

### 1. Resolve window and audience

Today's date in `YYYY-MM-DD`. Default window: trailing 7 days. If the user said "since #N", resolve `since_date = max(N.created_at, today - 14d)` to bound the deck.

### 2. Pull source artifacts

```bash
WINDOW_START=$(date -d '7 days ago' +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)
OUT=figures/mentor-slides/$TODAY
mkdir -p "$OUT"

# Recent clean-results (final + drafts)
gh issue list --repo superkaiba/explore-persona-space \
  --label "clean-results" --state all --limit 50 \
  --search "updated:>=$WINDOW_START" \
  --json number,title,body,labels,updatedAt > "$OUT/_clean-results.json"

gh issue list --repo superkaiba/explore-persona-space \
  --label "clean-results:draft" --state all --limit 50 \
  --search "updated:>=$WINDOW_START" \
  --json number,title,body,labels,updatedAt > "$OUT/_clean-result-drafts.json"

# Commits + open follow-ups
git log --oneline --since="$WINDOW_START" --no-merges > "$OUT/_commits.txt"
gh issue list --repo superkaiba/explore-persona-space \
  --label "status:proposed" --state open --limit 20 \
  --json number,title > "$OUT/_proposed.json"

head -100 RESULTS.md > "$OUT/_results-head.md"
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

If a field is missing, leave the slot empty in the template AND flag it in the post-render report (see Step 6) — never fabricate.

### 4. Compose the deck

Read `template.md` from this skill directory. Fill in slot-by-slot, in this order:

1. **Cover** — project name, `Week of {{TODAY}}`, presenter (git config `user.name`), audience.
2. **Recap** — only if previous deck exists at `figures/mentor-slides/<prev-date>/deck.md`. Pull its TL;DR bullets. Otherwise: drop this slide entirely.
3. **TL;DR** — 3-5 bullets. Each bullet: bolded claim + 1-line evidence + ([#N](url)). **Sort by confidence**: HIGH first, MODERATE next, LOW last (Hughes/Chua: "Start with the most important message first").
4. **Agenda** — section names + slide counts + rough minute budget per section. This slide is the Hughes/Chua signature; do NOT skip it even for short decks.
5. **One slide per result** (5..N), ordered by confidence then chronologically:
   - Title = `headline_claim` (≤12 words, full sentence with effect number).
   - `![bg right:50%]({{hero_figure_path}})`.
   - Three bullets: Setup / Result (with N) / Caveat.
   - Footer `<small>` line: commit hash · Issue #N · Confidence label.
6. **Open questions for {{audience}}** — 3 bullets max, framed as decisions needed (not "things I'm unsure about"). Source: open `status:proposed` issues + any `[Hypothesis: ...]` markers in clean-result bodies that lack approval.
7. **Next week's plan** — numbered, ≤5 items. Each: action + expected information gain (1 line). Source: clean-result `### Next steps` bullets, prioritized.
8. **Appendix divider** + one reproducibility card per result (config, seed, commit, dataset version, eval N, WandB run, HF model).

**Edge cases:**
- **Zero clean-results in window** → emit an "infrastructure week" deck instead: TL;DR pulled from commit messages, agenda lists what was built, no per-result slides, longer "next week" section.
- **One clean-result** → minimum 5 slides (Cover + TL;DR + 1 result + Next steps + Appendix). Skip Agenda.
- **>8 clean-results** → ask the user to triage (which 4-6 to feature) BEFORE assembling. Don't silently drop.

### 5. Quality checklist (run BEFORE rendering)

Self-audit the draft. If any check fails, fix before render:

- [ ] Every results-slide title is a full sentence with an effect number, ≤12 words.
- [ ] Every chart has N stated (in caption or footer).
- [ ] No effect-size jargon (Cohen's d, η², r-as-effect, Δ-framed-as-effect).
- [ ] No raw p-values without N.
- [ ] No `value ± err` in prose. (Allowed on charts.)
- [ ] ≤3 colors / ≤3 model series per chart.
- [ ] Confidence label present on every results slide.
- [ ] One headline message per slide.
- [ ] Footer carries commit hash + issue link on every results slide.
- [ ] Open-questions slide frames each item as a decision needed (not a worry).

If a structural fix is impossible (e.g., a clean-result chart has 5 baselines and recoloring would lose meaning), DO NOT silently leave it — add a `<!-- FLAG: ... -->` HTML comment in the markdown so the user sees it on review.

### 6. Render

```bash
npx --yes @marp-team/marp-cli@latest \
  "$OUT/deck.md" \
  --pdf --allow-local-files \
  -o "$OUT/deck.pdf"
```

If marp-cli's first run is slow (downloads), tell the user; subsequent runs are fast. If `--pdf` was not requested, skip rendering.

### 7. Report

Tell the user:
- `deck.md` and (if rendered) `deck.pdf` paths.
- Number of result slides included.
- Quality-checklist items that were FLAGGED but couldn't be auto-fixed (with line numbers).
- Suggest: review the deck, edit if needed, then `git add figures/mentor-slides/$TODAY && git commit`.

## Anti-patterns this skill enforces

- **No "Results" / "Background" / "Next Steps" topic-label titles.** Every slide title is a sentence asserting a finding. (Alley assertion-evidence; Naegle "Ten Simple Rules" #3.)
- **No bullet-list-only slides for empirical results.** Bullets accompany a figure or table; never substitute. (Alley.)
- **No effect-size jargon.** Project rule + general ML talk hygiene. (Nanda; project `CLAUDE.md`.)
- **No conflating "we tried X" with "X works".** Confidence label is mandatory.
- **No skipping the agenda slide.** Hughes & Chua: "section names + slide counts + time per section" so the mentor knows the time budget upfront.
- **No "summary of what I did."** Open with what was *learned* (Perez: "predictions vs. findings"), not what was *attempted*.

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
- **Render alternatives**: Slidev (richer layouts, heavier toolchain) or reveal-md (lighter, less reliable headless PDF). Stick with Marp unless the user explicitly asks otherwise.

## See also

- `template.md` (this dir) — Marp markdown skeleton.
- `principles.md` (this dir) — sources + rationale for each rule.
- `.claude/skills/clean-results/template.md` — source format the skill reads from.
- `.claude/skills/paper-plots/SKILL.md` — figures the deck embeds.
- `ml-paper-writing:presenting-conference-talks` — different use case (compiled paper → conference deck).
