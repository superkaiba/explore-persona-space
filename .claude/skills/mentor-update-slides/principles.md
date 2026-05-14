# Principles: mentor-update-slides

This skill exists because: the codebase already has clean-result bodies (rich, polished per-experiment artifacts) and a weekly skill (markdown agenda gist), but no path that assembles those into slides for an advisor meeting. The deck format matters — it forces the discipline of *one claim per slide*, *headline as assertion*, and *figure as load-bearing element*, which markdown agendas don't.

The structure and rules below are not invented; each is cited to a primary source. Where sources conflict, we prefer ML-specific guidance (Hughes & Chua, Nanda, Perez) over generic presentation advice (Reynolds, Knaflic).

---

## Persistent deck structure

The skill maintains **one** deck per project at `figures/mentor-slides/deck.md`, not a new dated subdirectory each week. Hughes & Chua are explicit:

> "Keep one slide deck per project … Provides a consistent story for how your research progressed (we recommend you add the most recent slides to the start of the deck, instead of the end)."
> — Hughes & Chua, "Tips On Empirical Research Slides"

The deck has three anchored regions, demarcated by HTML comments the skill greps for on each run:

```
<!-- BEGIN HEADER --> ... <!-- END HEADER -->     # replaced each run
<!-- BEGIN LOG --> ... <!-- END LOG -->           # append-only, newest first
<!-- BEGIN APPENDIX --> ... <!-- END APPENDIX --> # accumulates
```

**HEADER** (regenerated each run): Cover · Objectives · Project summary · Agenda. The Project summary slide is the "clean evolving face" of the project — current thesis question, active claims with confidence labels, and what's running. Each active claim links into the LOG via an HTML anchor of the form `#week-YYYY-MM-DD`, so the mentor can click straight to the week the claim was first reported.

**LOG** (research log, newest first): each weekly entry begins with a `## Week of YYYY-MM-DD` divider slide carrying an HTML anchor, followed by that week's per-result slides + open-questions + next-steps. Older weeks are never rewritten.

**APPENDIX** (accumulating): reproducibility cards + the four backup-slide families (see § Backup-slide families). New cards prepend; old cards stay.

This split lets the deck serve two roles simultaneously: (a) the Project summary at the top is the *evolving project status* that a new collaborator can read in two minutes, and (b) the LOG below is the *durable archive* that captures how the story changed over time. Both live in the same file with the same shared link.

**Idempotency.** Re-running on the same day replaces today's LOG block instead of duplicating it. The skill detects an existing `## Week of $TODAY` divider at the top of LOG and rewrites the bounded block.

---

## Per-meeting flow inside the HEADER + new week's LOG

The within-meeting flow still follows **Hughes & Chua's empirical-research-slides protocol** (written for MATS scholars under Perez/Evans):

1. Cover (HEADER).
2. Objectives — what the user wants from the meeting (HEADER; § Objectives slide).
3. Project summary with confidence-labeled claims linking into the LOG (HEADER).
4. Agenda for this week with section names + slide counts + time budget (HEADER).
5. Date divider for the new week (LOG).
6. Per-experiment blocks (setup → results → interpretation), prioritized by importance (LOG).
7. Open questions / blockers / resource requests (LOG).
8. Proposed prioritized next steps (LOG).
9. Reproducibility cards + conditional backup slides (APPENDIX; § Backup-slide families).

Source: [Hughes & Chua, "Tips On Empirical Research Slides", LessWrong](https://www.lesswrong.com/posts/i3b9uQfjJjJkwZF4f/tips-on-empirical-research-slides).

**Perez modification**: open with "predictions vs. findings" — what you expected last week vs. what actually happened. Builds calibration. Source: [Perez, "Tips for Empirical Alignment Research"](https://www.alignmentforum.org/posts/dZFpEdKyb9Bf4xYn7/tips-for-empirical-alignment-research).

We adopt Hughes & Chua's full skeleton. We fold Perez's predictions-vs-findings into the *Project summary* slide (each active claim is a finding; the "currently running" line is the prediction), eliminating the need for a separate Recap slide — the LOG itself is the recap.

---

## Objectives slide

Between Cover and Project summary the deck carries one slide stating *what the user wants from this meeting*:

> "In addition to agenda, it can be helpful to start with objectives — why are you here and what are you hoping to get from them? are you trying to inform them? get advice on something specific? get advice on something broad?"
> — Ted Sanders, comment on Hughes & Chua

Three default objective categories, used as starting language:
1. **Inform** — give the mentor a status update; no input required.
2. **Advice on a specific decision** — bring a concrete fork in the road and ask for input on it.
3. **Advice on broad direction** — open-ended, e.g. when a research thread is exhausted and the next step is unclear.

The skill takes an `--objective` argument (default `inform`) and the user-supplied string drops verbatim into the slide. The slide also lists the specific decisions the user is *and is not* seeking input on, so the mentor can calibrate where to push back.

---

## Backup-slide families

Hughes & Chua devote a section to backup slides — what to have ready when the mentor asks the obvious next question. The skill materialises **four** families, each conditional on real source data being present in this week's clean-results. Skip families with no data; never fabricate.

(a) **Metric definition + concrete example.** One slide per metric, with a one-line definition and a real prompt → real model output → assigned score. Sanders: "Can help to include real data / real prompts / real model outputs — harder to fool yourself when you look at real data instead of relying on abstract metrics and intentions." Source data: `## Setup & hyper-parameters` opening prose + `## Sample outputs` from the clean-result template.

(b) **Detailed prompt with arrows / highlights.** One slide per representative prompt with the load-bearing region marked (e.g., `<mark>...</mark>`), and a one-arrow takeaway in the caption. Hughes & Chua: "Drawing arrows and highlighting text helps draw attention to particular parts of the prompt." This is the *only* place the project allows annotations on slide content — the chart-level no-annotations rule (`feedback_no_plot_annotations`) still holds; arrows on prompt text are a different category.

(c) **Data-scaling curve.** Hughes & Chua: "have you tried more data?" is one of the most common mentor questions; have a scaling answer ready. Show linear and (where data spans ≥2 orders of magnitude) log-log views side by side. Source data: any clean-result whose headline-numbers table has a "data fraction" or "training-step sweep" axis.

(d) **Baseline-invalidation slide.** "What are some simple ways that would invalidate your results? You should think of some and include slides that discuss it." (Hughes & Chua.) One slide listing the controls + 1-line on what each rules out, with the baseline numbers inline. Source data: baseline rows in the headline-numbers table.

The skill's quality checklist refuses to emit a backup slide whose source data was not extracted; missing slides are reported in the Step 8 user-facing summary, never silently skipped.

---

## Per-slide design: assertion-evidence

Every results slide uses **Michael Alley's assertion-evidence (A-E) structure**:

1. **Sentence headline** at top — one declarative claim, ≤12 words, ≤2 lines, large font (~44pt for talks; smaller acceptable for read-along decks).
2. **Visual evidence** below/right — chart, table, or diagram.
3. **Caption** — N, error-bar definition, model/seed/commit.

A-E is empirically validated to improve recall and comprehension over bullet-list slides.

Sources:
- [Alley, *Assertion-Evidence Approach*](https://www.assertion-evidence.com/)
- Garner & Alley 2011, ASEE
- [Naegle, "Ten simple rules for effective presentation slides", PLoS Comp Bio (PMC8638955)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8638955/) — Rule 3: "Use full sentence titles".
- [Six Minutes / Dlugan, "Slide Title Guidelines"](https://sixminutes.dlugan.com/assertion-evidence-design-presentation-slides/)

**Concrete examples**:
- BAD: "Results", "Ablations", "Background"
- GOOD: "Marker rate rises 0.42 → 0.71 at step 600"; "EM persona is the villain character, not a rogue AI"

---

## Figure rules (ML-specific)

- **Always show error bars** on eval bars. Use `mean ± 1.96·SEM` for 95% CI; report SEM in caption. Use **clustered SEs** when items share a passage. Source: [Anthropic, "A statistical approach to model evals"](https://www.anthropic.com/research/statistical-approach-to-model-evals).
- **Always label what error bars represent** (SEM vs. SD vs. CI). Unlabeled error bars are ambiguous. Source: [Wilke, *Fundamentals of Data Visualization*](https://clauswilke.com/dataviz/visualizing-uncertainty.html); [Cumming et al., "Error bars in experimental biology" (PMC2064100)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2064100/).
- **Inline N** — `n=X` in caption or under x-tick labels.
- **One y-axis per chart**; avoid dual axes. Source: Tufte, data-ink principle.
- **Log scale only when data spans ≥2 orders of magnitude** or growth is multiplicative. Otherwise linear.
- **Small multiples for ≥3 conditions** on the same metric — same axes, aligned. Source: [Tufte, small multiples](https://en.wikipedia.org/wiki/Small_multiple).
- **Plot paired differences**, not two separate bars, for paired comparisons. Source: Anthropic.
- **Error bar on the delta, not on the endpoints.** When the result is a difference (post − pre, treatment − control), compute and plot the SE of the difference; do not eyeball "do these two error-barred bars overlap?". Optionally color the bar by signed effect or by significance level. Source: Sanders, comment on Hughes & Chua: *"If you're measuring deltas between two things, compute the error bar on the delta, don't compute the error bars on the two things; consider coloring by statistical significance."*
- **Use horizontal bars when category labels are long.** Avoids diagonal axis labels that the post explicitly flags as a negative example. Source: Sanders refinement of Hughes & Chua.
- **≤3 colors / ≤3 model series per slide.** Source: Hughes & Chua.
- **Always include the prompt/measurement protocol next to the plot.** Source: Hughes & Chua.

---

## Headline-as-claim

Every slide title is a full sentence asserting the finding, not a topic label.

- BAD: "Results"
- GOOD: "X improves Y by Z% across 4 personas"

12-word cap (Duarte's 3-second-readability rule). One line ideal, two lines max.

Sources: Alley; [Naegle Rule 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC8638955/); [storytellingwithdata.com](https://www.storytellingwithdata.com/).

---

## Confidence framing (Nanda vocabulary)

Use these in headlines and bullets when the chart underdetermines the verbal claim:

- **Existence proof** — observed at least once.
- **Systematic** — across a wide range of contexts.
- **Hedged** — compelling/suggestive/tentative.
- **Narrow** — restricted to specific setting.
- **Guarantees** — always true (rare in deep learning).

The clean-result body's existing `Confidence: HIGH | MODERATE | LOW` maps onto this:
- HIGH ≈ systematic
- MODERATE ≈ hedged
- LOW ≈ existence proof / narrow

Reuse the issue's verdict verbatim. Source: [Nanda, "Highly Opinionated Advice on How to Write ML Papers"](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers).

---

## What's-next framing

> "List what you think your next steps should be. Seek feedback from your mentor about whether these experimental priorities are correct. Include any resource requests."
> — Hughes & Chua

> "When you show experimental results … you should also include a discussion of your proposed next possible steps immediately after."
> — Perez

Format as a **prioritized numbered list**, ≤5 items. Each: action + expected information gain (1 line). NOT a wishlist.

---

## Common mistakes (deck-time)

Sources: Hughes & Chua; Nanda; [Steinhardt via Tyler Crosse](https://www.tylercrosse.com/ideas/2025/ml-research-notes/); [Hamming, "You and Your Research"](https://www.cs.virginia.edu/~robins/YouAndYourResearch.html); [Panda, "Weekly Research Progress Meetings"](https://biswabandan.medium.com/weekly-research-progress-meetings-how-to-do-it-right-if-you-want-to-do-it-right-e32fde4495ca).

- Showing all 10 attempts instead of "your best setup first".
- Too many bar charts at once → "hard to know what the takeaway message is."
- Verbose slides; heatmaps that force squinting.
- Missing controls / no error bars / no N.
- Overclaiming, cherry-picking, weak baselines, illusion-of-transparency (Nanda).
- Conflating high-level approach with low-level instantiation (Steinhardt).
- Skipping meetings when "no progress" instead of re-syncing (Panda).
- Defaulting to a "highly limited technical talk" when the audience wants context-first framing (Hamming).

---

## Color and accessibility

- **Categorical**: Okabe-Ito 8-color palette — `#E69F00 #56B4E9 #009E73 #F0E442 #0072B2 #D55E00 #CC79A7 #000000`. CVD-safe; widely used in nature journals.
- **Sequential**: viridis / magma / plasma — perceptually uniform, CVD-safe, grayscale-print-safe.
- **Contrast**: ≥4.5:1 for body text (WCAG AA).
- **Never encode meaning by color alone** — pair with shape/position/label.

Sources: [Okabe-Ito palette reference](https://conceptviz.app/blog/okabe-ito-palette-hex-codes-complete-reference); [Wilke, *Fundamentals of Data Viz*](https://clauswilke.com/dataviz/); WCAG 2.1.

---

## Pacing

- ~1 slide per minute for results-heavy weekly decks (denser than the Peyton Jones "1 slide per 2 minutes" rule for one-shot conference talks).
- Each slide ≤6 lines of text (project rule + Naegle Rule 1).
- One headline message per slide.

Sources: [Peyton Jones, "How to Give a Great Research Talk"](https://simon.peytonjones.org/great-research-talk/); Naegle Rules 1-2.

---

## Tooling choice

We default to **marp-cli** for rendering markdown → PDF on Linux. Source: [Marp CLI](https://github.com/marp-team/marp-cli); comparison [pkgpulse 2026](https://www.pkgpulse.com/blog/slidev-vs-marp-vs-revealjs-code-first-presentations-2026).

Trade-offs:
- **Marp**: single binary path, fast render (~2-3s for 20 slides), `--pdf`/`--pptx`/`--images` parity, theme via simple CSS. Best for skill-driven generation.
- **Slidev**: richer academic layouts (e.g., `slidev-theme-scholarly` ships 26 layouts). Heavier (Vite + Node SPA build). Use only if a layout-heavy theme is essential.
- **reveal-md**: lighter but headless PDF is brittle. Avoid.

Concrete theme alternatives (kept for reference, not auto-installed):
- [`kaisugi/marp-theme-academic`](https://github.com/kaisugi/marp-theme-academic) — Beamer-like, MIT-licensed.
- [`cunhapaulo/marpstyle`](https://github.com/cunhapaulo/marpstyle) — 21 named themes (Plato, Einstein, Turing).
- [`jxpeng98/slidev-theme-scholarly`](https://github.com/jxpeng98/slidev-theme-scholarly) — Slidev-only, 26 layouts.

---

## Sources (consolidated)

ML-specific:
- Hughes & Chua, ["Tips On Empirical Research Slides"](https://www.lesswrong.com/posts/i3b9uQfjJjJkwZF4f/tips-on-empirical-research-slides) (LessWrong) — primary source for the persistent-deck structure, agenda slide, chart rules, and backup-slide families.
- Ted Sanders, [comment on Hughes & Chua](https://www.lesswrong.com/posts/i3b9uQfjJjJkwZF4f/tips-on-empirical-research-slides?commentId=) — refinements: error bar on the delta, horizontal bars for long labels, Objectives slide before Agenda, real data / real prompts / real outputs as a defense against abstract-metric self-deception, sentence-titles, "crummy slides are fine" anti-perfectionism.
- Hughes & Chua, ["Tips and Code for Empirical Research Workflows"](https://www.lesswrong.com/posts/6P8GYb4AjtPXx6LLB/tips-and-code-for-empirical-research-workflows) (LessWrong)
- Nanda, ["Highly Opinionated Advice on How to Write ML Papers"](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers) (Alignment Forum)
- Perez, ["Tips for Empirical Alignment Research"](https://www.alignmentforum.org/posts/dZFpEdKyb9Bf4xYn7/tips-for-empirical-alignment-research) (Alignment Forum)
- Tyler Crosse, ["Notes on Effective ML Research"](https://www.tylercrosse.com/ideas/2025/ml-research-notes/)
- Anthropic, ["A statistical approach to model evals"](https://www.anthropic.com/research/statistical-approach-to-model-evals)

Slide design:
- Alley, [*Assertion-Evidence Approach*](https://www.assertion-evidence.com/)
- Naegle, ["Ten Simple Rules for Effective Presentation Slides"](https://pmc.ncbi.nlm.nih.gov/articles/PMC8638955/) (PLoS Comp Bio)
- Wilke, [*Fundamentals of Data Visualization*](https://clauswilke.com/dataviz/)
- Cumming et al., ["Error bars in experimental biology"](https://pmc.ncbi.nlm.nih.gov/articles/PMC2064100/)
- Tufte, [chartjunk](https://www.edwardtufte.com/notebook/chartjunk/) and [small multiples](https://en.wikipedia.org/wiki/Small_multiple)
- Six Minutes / Dlugan, ["Slide Title Guidelines"](https://sixminutes.dlugan.com/assertion-evidence-design-presentation-slides/)
- [Storytelling with Data](https://www.storytellingwithdata.com/)

General research-talk advice:
- [Peyton Jones, "How to Give a Great Research Talk"](https://simon.peytonjones.org/great-research-talk/)
- [Hamming, "You and Your Research"](https://www.cs.virginia.edu/~robins/YouAndYourResearch.html)
- [Edwards, "How to Give an Academic Talk"](https://pne.people.si.umich.edu/PDF/howtotalk.pdf)
- [Panda, "Weekly Research Progress Meetings"](https://biswabandan.medium.com/weekly-research-progress-meetings-how-to-do-it-right-if-you-want-to-do-it-right-e32fde4495ca)

Tooling:
- [Marp CLI](https://github.com/marp-team/marp-cli)
- [Marp math typesetting docs](https://github.com/marp-team/marp/blob/main/website/docs/guide/math-typesetting.md)
- [Markdown presentation tools comparison](https://dasroot.net/posts/2026/04/markdown-presentation-tools-marp-slidev-reveal-js/)
- [Slidev](https://sli.dev/)
