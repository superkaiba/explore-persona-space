---
name: clean-results
description: >
  Transform messy experiment outputs (epm:results markers, draft write-ups,
  raw eval JSONs, figures) into a polished "Clean Result" GitHub issue in the
  project board's Clean Results column. Applies advice from Neel Nanda,
  Ethan Perez, James Chua, John Hughes, and Owain Evans on mentor-grade
  research communication. Invoke with `/clean-results <source-issue-N>`
  or `/clean-results <draft-path>`.
user_invocable: true
---

# Clean Results

**Goal:** take messy, possibly-multi-issue experiment output and produce a
single, self-contained, mentor-grade "Clean Result" presentation. The output is
a GitHub issue body (labeled `clean-results`, titled `[Clean Result] ...`,
moved to the `Clean Results` column on the Experiment Queue project board).

**Reference exemplars:**
- issue #67 (`[Clean Result] 25% Tulu midtrain matrix …`) — **preferred structure**, uses the 6-subsection TL;DR (Background / Methodology / Results / How this updates me + confidence / Why confidence is where it is / Next steps) followed by a "# Detailed report" block.
- issue #65 (`[Clean Result] Single-[ZLT]-token sweep`) — older two-part TL;DR; kept for reference but new clean results should use #67's structure.

---

## Guiding principles (synthesized from the researchers)

### Epistemics first (Neel Nanda)

- **Write to inform, not to persuade.** The reader should update correctly,
  not just be impressed.
- **Distill one core claim + 2-3 supporting claims.** If you can't state the
  whole result in a 2-sentence lightning talk, keep compressing.
- **Separate observation from inference.** "X increased by Y" vs "this suggests Z."
- **Red-team before posting.** For every claim, ask: what's the strongest
  alternative explanation? Does my evidence rule it out? If not, caveat.
- **The evidence threshold for convincing yourself differs from convincing a
  skeptic.** Add sanity checks, baselines, error bars — not rhetoric.
- **Present limitations honestly and prominently.** Bury a caveat and the
  mentor will distrust the whole write-up.

### Narrative compression (Neel Nanda, Ethan Perez)

- **The TL;DR is the paper.** Mentor reads nothing else — does it tell the
  whole story? Include the key number.
- **Hero figure on the first scroll.** One eye-catching plot near the top.
  Ethan: "most readers only see the first page."
- **Title + TL;DR + hero figure ≈ 80% of what most readers retain.**
  Spend disproportionate effort there.

### Slide-style rigor for plots (James Chua, John Hughes)

Even though the output is an issue, not slides, the slide rules apply.

- **Always include, alongside every plot:** the exact prompt used (or link to
  data), error bars, sample size (N), and axis labels with direction
  (`higher = better`).
- **Values on bars.** Label each bar with its number (e.g., "51.4%").
- **≤ 3-5 colors per chart.**
- **Simple > complex.** Bar, line. Avoid heatmaps unless they ARE the finding
  (issue #65 is an exception — the heatmap IS the phase diagram).
- **No "10 setups on one chart."** Show the most important comparison first;
  relegate the rest to backup or a details section.
- **Large enough to read on a video call** (mentor will screen-share).

### Prose discipline (Ethan Perez)

- **Kill hedges.** "may," "can," "to our knowledge," "note that," "we try to,"
  "actually," "a bit," "fortunately." Almost always drop them.
- **Active voice, early verbs, one idea per sentence.**
- **Never use a comparative without saying what you're comparing to.**
  "Higher" — than what?
- **Lead and end paragraphs with strong sentences.** Middle sentences elaborate.
- **Ask every sentence: is this correct?** Especially numerical claims.

### Research posture (Ethan Perez, James Chua)

- **Minimal thing that updates beliefs most.** The clean result should be
  the minimum evidence that makes the claim land — not every experiment you
  ran.
- **Discuss *why* things worked or didn't.** A result without a mechanism is
  weaker than a result with one. If you don't know why, say "mechanism unknown,
  next step: ablate X."

### Reasoning transparency (Luke Muehlhauser, Open Philanthropy)

Source: <https://coefficientgiving.org/research/reasoning-transparency/>.
The goal is to let a reader correctly update on your result, not just be
impressed by it. Every load-bearing claim should carry enough context that
the reader can reconstruct *how much* to update.

- **Bottom line first.** Open with a linked summary of key takeaways; every
  section leads with its conclusion, not its setup.
- **Calibrated language.** Pick explicit confidence words and use them
  consistently:
  - *plausible* ≈ 50%+
  - *likely* ≈ 65%+
  - *very likely* ≈ 85%+
  - *unlikely* ≈ <35%
  - For load-bearing numbers, give a 70–95% credence interval, not just a point estimate.
- **Flag the *kind* of support each claim has**, not just the conclusion:
  - "Observed in this run" (direct evidence)
  - "Replicated across N seeds" (robustness)
  - "Consistent with prior result #NN" (external support)
  - "Intuition / guess, not verified" (weak support)
  - "Shallow skim — I haven't checked this carefully" (depth disclosure)
- **Separate observation from inference.** First the number, then what it
  suggests. If the interpretation is a stretch, label it as such.
- **Disclose priors and biases.** "I went in expecting X." "I'm the author of
  the prior method, so I may be biased toward finding it works." Makes the
  reader's update calibration easier.
- **Flag counter-considerations inline.** After each major claim, the
  strongest argument against it in one sentence. If you haven't addressed it,
  say so.
- **Identify key uncertainties explicitly.** A dedicated list of "what I'm
  uncertain about" and "what would change my mind" is better than scattered
  hedging.
- **Cite precisely.** Link to specific issues, commits, figures, JSON files,
  WandB runs — not "see the codebase." If citing external work, give page /
  section numbers.
- **Track updates.** When a number in this issue changes later, note the
  update and why (e.g. "raised confidence from moderate → high after seed
  replication in #NN").

---

## Required output structure

The clean-result issue body MUST follow this skeleton. Exemplar: issue #67.

```markdown
## TL;DR

### Background

[2-4 sentences. Prior results that motivated this experiment (cite issue numbers).
What question it answers / what was the goal. A reader who sees only this
subsection should know WHY this experiment was run.]

### Methodology

[2-4 sentences. Model, pipeline, conditions, N, eval signal. Enough that a
reader who skips the "# Detailed report" section still knows what experiment
produced the numbers. State the key matched-vs-confounded design choices.]

### Results

[ONE hero figure, inline via raw-GitHub URL pinned to a specific commit.
Then 2-4 sentences stating what the figure shows + the key numbers. This is
the single most important block in the whole clean result — lead with the
effect size, not the methodology.]

### How this updates me + confidence

[Bulleted list. Each bullet = one belief update, tagged with a confidence
level (HIGH / MODERATE / LOW) and the evidence that supports it. Include
beliefs that got STRONGER, WEAKER, and NEW beliefs the experiment surfaced.
Mentor reading this should see your epistemic state change, not just the
numbers.]

### Why confidence is where it is

[Bulleted list of the specific reasons the confidence levels above are what
they are — both the evidence that supports HIGH-confidence claims (e.g.
"three independent matched-protocol datapoints cluster within 2 pt") AND the
flaws that cap MODERATE/LOW claims (n too small for CIs, confounds, missing
conditions, protocol differences between arms, judge/metric concerns,
pod/hardware variance). For each claim in the preceding subsection, a
skeptic should be able to find here either the reason it survives scrutiny
or the reason it doesn't.]

### Next steps

[Bulleted list ranked by information gain per GPU-hour. Each item states
WHAT, HOW LONG, and WHICH CAVEAT IT RESOLVES. Cite follow-up issue numbers
where they exist.]

---

# Detailed report

## Source issues

[Numbered list of the prior issues/drafts this clean result distills. Each:
`#N — brief title — one-line contribution.` Plus any downstream consumers of
this result.]

[Numbered list of the prior issues/drafts this clean result distills. Each:
`#N — brief title — one-line contribution.` Plus any downstream consumers of
this result.]

## Setup & hyper-parameters

[Reproducibility Card per CLAUDE.md. Tables for Model / Training / Data /
Eval / Compute. Include exact commit, exact command, exact seeds. No "see
config" — write the actual values.]

## WandB

[Link to project + table of individual run URLs. If logging has a known gap,
state it here AND explain what you did about it (e.g., post-hoc re-upload).
Always point to where the FULL raw data lives: WandB artifact name + run URL
for each source/condition, plus local disk path. A reader must be able to
download the complete completions / raw outputs, not just the aggregated
numbers.]

## Sample outputs

[2-5 cherry-picked examples per condition that make the behavior concrete.
For generation experiments: the prompt + a trimmed (~250-char) excerpt of the
model's output, showing both a "positive" (behavior present) and "negative"
(behavior absent) case. For judge-scored experiments: show a few scored
completions with judge scores + judge reasoning. State explicitly that these
are cherry-picked and link to the WandB artifact containing the full dump.
Mentor-grade results are concrete, not just summary statistics — the reader
should leave with a felt sense of what the model actually produced.]

## Plot

[ONE hero figure. Raw GitHub URL to the PNG in the repo (so it renders inline
from anywhere). 1-sentence caption stating what to look at + what the takeaway
is. Regen command on the next line.]

## Headline numbers

[One table. Rows = regimes / conditions. Columns = metrics with units and CIs.
Bold the clean / winning row. No more than ~10 rows; drop the rest to a
`<details>` block or link to JSON.]

## Artifacts

[Table mapping artifact type -> path/URL: script, compiled JSON, per-run JSONs,
plot script, figures (PNG + PDF), data cache, any derived code.]

## Caveats

[Ordered by severity. Each caveat is a bullet that states the limitation AND
what would resolve it. Single-seed, in-distribution, literal string match,
narrow model family — all standard caveats to check.]
```

---

## Workflow

When invoked, run this procedure.

### Step 1: Intake

Identify the source material. Typical inputs:

- A GitHub issue number (pull its body + all `epm:results` / `epm:analysis` markers + plan).
- A draft path in `research_log/drafts/` (read the draft + any linked eval JSONs + figures).
- Multiple issue numbers (you're consolidating).

Read, in this order:
1. The plan (`epm:plan` marker on the source issue, or `.claude/plans/issue-N.md`).
2. The results (`epm:results`) and analyzer draft (`epm:analysis`).
3. Any `research_log/drafts/*.md` referenced.
4. The actual `eval_results/*/run_result.json` and `figures/*`.
5. WandB run URLs.

**Never trust the draft's prose for numbers — pull numbers from JSON.** A
common failure mode is a draft claims X%, the JSON says Y%. Check.

### Step 2: Distill the claim

Before writing anything, answer:

1. **What is the ONE thing I want the mentor to walk away believing?**
2. **What is the key number that backs it?**
3. **What's the single strongest alternative explanation, and does the
   evidence rule it out?**
4. **If the mentor reads only the TL;DR, is the answer to (1) unambiguous?**

If any of these is shaky, stop. Fix the experiment / add a caveat / weaken
the claim before writing more.

### Step 3: Choose the hero figure

Pick ONE figure. Criteria:

- Renders the core claim visually.
- Has error bars or confidence intervals where applicable.
- Has a prompt / setup description embedded OR a clear caption.
- Would be legible on a screen-shared video call.
- Axes labeled with direction (`higher = better`).

If no single figure does the job, you probably haven't distilled hard enough.
Go back to Step 2.

Commit the figure at `figures/<experiment>/<name>.png` + `.pdf`. Use the raw
GitHub URL (`https://raw.githubusercontent.com/<owner>/<repo>/<commit>/figures/...`)
so it renders in the issue body. Pin to a specific commit, not `main`.

### Step 4: Write the body

Follow the skeleton above. Hard rules:

- **TL;DR has exactly 6 H3 subsections**: Background, Methodology, Results, How this updates me + confidence, Why confidence is where it is, Next steps. Each subsection ≤ 4 sentences (Results allows a hero figure + 2-4 sentences). No ambiguity about what was done, what was found, or how confident you are.
- **Hero figure lives inside the Results subsection of the TL;DR**, not in a separate "Plot" section below. The detailed report does not duplicate it.
- **Confidence is tagged explicitly** in the "How this updates me" subsection: HIGH / MODERATE / LOW per bullet.
- **"Why confidence is where it is" is a mandatory subsection** — covers both the supporting evidence for HIGH-confidence claims and the flaws that cap MODERATE/LOW claims. Every confidence tag from the preceding subsection should have a corresponding item here. If you can't produce 3+ concrete items, you haven't red-teamed enough; go back to Step 2.
- The `# Detailed report` block contains: Source issues, Setup & hyper-parameters (Reproducibility Card, actual values per CLAUDE.md), WandB, Sample outputs, Headline numbers, Artifacts, Caveats.
- Reproducibility Card is complete. Every row. Actual values, not "see config."
- Headline table has units, CIs, sample sizes.
- Caveats (in the detailed report) are ordered by severity AND consistent with the TL;DR's "Why confidence is where it is" — the caveats section is the expanded version of that TL;DR subsection.
- Every numerical claim in prose is traceable to a row in the JSON or table.

### Step 5: Run the pre-publish checklist

See `checklist.md`. Do not open the PR / issue until every item passes or has
a documented exception.

### Step 6: Post

1. **If creating a new issue:**
   ```bash
   gh issue create \
     --repo superkaiba/explore-persona-space \
     --title "[Clean Result] <concise descriptive title>" \
     --label clean-results \
     --label "aim:<N>-<name>" \
     --label "type:experiment" \
     --label "compute:<size>" \
     --body-file <path-to-body.md>
   ```

2. **Move to Clean Results column on the project board:**
   ```bash
   uv run python scripts/gh_project.py set-status <N> "Clean Results"
   ```
   (If "Clean Results" is not yet a project-board option, add it first via the
   GitHub UI — the `clean-results` label exists but the column may not.)

3. **Cross-link source issues.** On each source issue referenced in the
   `## Source issues` section, post a one-liner:
   > Distilled into clean result: #<new-issue-N>.

4. **Do NOT close source issues.** They stay in their Done column. The clean
   result is the canonical mentor-facing artifact; the source issues remain
   the audit trail.

---

## When to use this skill

- After one or more experiments reach `status:done-experiment` and the finding
  is worth showing a mentor.
- When consolidating a cluster of related issues (e.g., a sweep spanning
  #28, #46, pilot issues) into one presentable result.
- Before a mentor meeting — produce the clean result, then screen-share the
  issue as the single source of truth. No separate slide deck needed.

## When NOT to use this skill

- Result isn't ready. If the answer to "what's the one claim" is "several
  conflicting things," run more experiments first or publish as
  `status:under-review` with a `CONCERNS` flag.
- Infra / code-change issues. These close via `status:done-impl` — they're
  not claims about the world.
- A single small experiment that is well-presented in its own
  `epm:results` comment. Cleaning adds overhead; reserve it for findings
  that will be cited later.

---

## Companion files

- `template.md` — the literal skeleton to fill in.
- `checklist.md` — pre-publish verification list. Run before posting.
- `principles.md` — the distilled advice from each researcher, with links.
  Read this once at the start of a session.
