# Principles — Distilled Advice from Researchers

Read once at the start of a clean-results session. These are the source
principles behind every rule in `SKILL.md`, `template.md`, and `checklist.md`.

---

## Neel Nanda — DeepMind MechInterp lead, MATS mentor

### On research communication

- "Identify the core, communicable claims within messy findings."
- "Structure a compelling and *true* narrative." Compelling ≠ oversold.
- "Write to inform, not to persuade." The reader should update correctly.
- "The evidence threshold for convincing yourself differs from convincing
  skeptics. Provide sanity checks, statistical robustness, and strong
  baselines — not just persuasive writing."
- "Extensively red-team: actively search for alternative explanations and
  missing experiments."
- "Present limitations honestly. Be prepared to backtrack when messiness
  emerges — acknowledge rather than suppress inconvenient results."
- "The form doesn't matter — blog post, paper, preprint. It needs to present
  the evidence clearly and have strong enough evidence to meaningfully
  inform someone's opinion."

### On distillation

- "Compress to a few concrete, well-scoped claims that readers can actually
  retain."
- "Test your compression: how would you explain this in a lightning talk?"
- "Writing forces you to clarify your understanding — you notice holes and
  missing experiments. People don't really understand their project until
  they write it up."
- "One to three specific novel claims supported by rigorous empirical
  evidence." More than that, and it's not a paper, it's a journal.

### On paper structure

- "Most readers skim through the abstract, intro, and figures — spend
  disproportionate effort there."
- "Iteratively from abstract through introduction to first full draft."

Sources: [Highly Opinionated Advice on How to Write ML Papers](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers),
[My Research Process: Key Mindsets](https://www.alignmentforum.org/posts/cbBwwm4jW6AZctymL/my-research-process-key-mindsets-truth-seeking),
[How I Think About My Research Process: Explore, Understand, Distill](https://www.lesswrong.com/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand).

---

## Ethan Perez — Anthropic Alignment Science

### Paper writing — clarity

- Minimize pronouns ("this," "it," "these"). Use only as adjectives
  ("this result"), not bare.
- Position verbs early in sentences.
- Convert "X's Y" to "The Y of X" (prepositional phrases parse easier).
- Simple short-syllable words.
- One idea per sentence. Split long sentences. (Long is fine if words are
  simple.)
- Lead and end paragraphs with strong clear sentences. Middle sentences
  elaborate.
- Active voice. Always specify the actor.
- Don't start every sentence with "We."
- Never use comparatives without stating what's compared.
- Limit hedging — "may," "can," "could" should almost always be dropped.
- Drop: "actually," "a bit," "fortunately," "to our knowledge,"
  "note that," "observe that," "try to," and most intensifiers.
- Ask every sentence: "Is what I'm saying correct?"
- Explain all uncommon terminology on first use.

### Figures

- Axis labels and ticks at least as large as body text.
- Colorblind-friendly colormaps (matplotlib viridis).
- Put an eye-catching figure on the first page — most readers only see the
  first page to decide whether to keep reading.
- Minimize visual white space.

### Research workflow

- De-risk mode 75% of the time. Python notebooks for rapid exploration.
- Always have experiments running 24/7.
- Tailor experiments to run in ≤ 16h (overnight).
- Notion database per experiment: name, tags, collaborators, status, updated.
- Dump figures + thoughts as you go. Messy is fine if collaborators can
  parse it.

Sources: [Easy Paper Writing Tips](https://ethanperez.net/easy-paper-writing-tips/),
[Tips for Empirical Alignment Research](https://www.alignmentforum.org/posts/dZFpEdKyb9Bf4xYn7/tips-for-empirical-alignment-research).

---

## James Chua & John Hughes — former MATS, now Anthropic / Truthful AI

### Slide structure (applies to issue presentation too)

1. **Summary slide first.** Key takeaways from last meeting + current
   experiment outcome ("worked" vs "didn't work") + a simple plot if possible.
2. **Agenda next.** Sections in priority order, slide counts, time estimates.
   Mentors manage multiple projects; let them calibrate.
3. **Most important message first.** Not "here are 10 setups I tried" —
   "here is the winning result."
4. **Backup slides for anticipated questions:** full prompts, scaling curves,
   hparam details, loss curves, baseline invalidations.

### Chart rules

- Always include the prompt (or experiment setup description) alongside
  every chart.
- Error bars. Always. For a proportion: `1.96 × sqrt(p(1-p)/N)` for 95% CI.
- Axis labels with direction indicator (↑ better / ↓ better).
- Values labeled directly on bars (e.g., "51.4%").
- ≤ 3-5 colors.
- Simple charts (bar, line) over complex (heatmap, 4D scatter).
- Large plots — legible on a shared video call.
- Avoid diagonal axis labels.
- Limit words per slide.

### What mentors want

- Raw ingredients (prompts, N, error bars) so they can critique methodology
  — not just high-level conclusions.
- Know whether to focus on validation or debugging — so make
  succeeded-vs-failed explicit.

### Mistakes to avoid

- Showing all 10 experimental setups at once.
- Too many words per slide.
- Omitting error bars or N.
- Presenting results without the prompt/setup.
- Heatmaps (unless the heatmap IS the finding).
- **Reporting only abstract metrics — no real prompts or model outputs.** Sanders refinement: "harder to fool yourself when you look at real data instead of relying on abstract metrics and intentions." Every clean result includes at least one concrete prompt → completion → score example in `## Sample outputs`; the corresponding slide in `mentor-update-slides` is one of the four backup-slide families.
- **Plotting two separate bars for a paired difference.** When the claim is a delta, compute and plot the error bar on the delta, not on each endpoint. Sanders refinement.

### Meta

- **One evolving deck per project.** This is implemented in
  `.claude/skills/mentor-update-slides/` as a single persistent file
  `figures/mentor-slides/deck.md` with three anchored regions: HEADER
  (replaced each run — cover, objectives, project summary, agenda),
  LOG (append-only, newest first, with `## Week of YYYY-MM-DD` dividers
  carrying HTML anchors so the project summary can link into history),
  and APPENDIX (accumulating reproducibility cards + four backup-slide
  families). The clean-result issue is the *unit* of the LOG; the deck
  is the *index*.
- Include "paper story" slides weekly for iterative narrative feedback.
- Initial slide investment: 1-2 days. Ongoing: ~half a day per meeting.

Sources: [Tips On Empirical Research Slides](https://www.lesswrong.com/posts/i3b9uQfjJjJkwZF4f/tips-on-empirical-research-slides);
Sanders [comment](https://www.lesswrong.com/posts/i3b9uQfjJjJkwZF4f/tips-on-empirical-research-slides) on the same post (real-data backup slides; error bar on the delta; sentence-verb titles; "crummy slides are fine"; objectives-before-agenda).

---

## James Chua — research posture

- "Do the minimal thing that will update your beliefs the most."
  (Example: few-shot prompting as a proxy for fine-tuning — faster signal.)
- Copy-pasting accelerates research; "consider not refactoring as the default."
- Internalize mentor feedback preferences — learn to predict what they'll
  flag, then act on it before they see it.

Source: [James Chua — MATS experience](https://jameschua.net/post/serimats_experience/).

---

## John Hughes & Ethan Perez — workflow tooling

- jsonl + pandas for experimental output.
- Folder structure: `./experiments/<name>/YYMMDD_technique_v1/` with numbered
  scripts showing execution order (`1_run.sh`, `2_eval.sh`).
- Cache LLM responses and intermediate outputs to resume without rerun.
- Save git commit hash in each experiment directory.
- Daily Slack standups with close collaborators.
- `uv` for Python (10-100x faster than pip). `tmux` for persistent remote
  sessions. `jless` / `nvtop` for inspection.

Source: [Tips and Code for Empirical Research Workflows](https://www.alignmentforum.org/posts/6P8GYb4AjtPXx6LLB/tips-and-code-for-empirical-research-workflows).

---

## Joe Benton — Anthropic Alignment Science / Fellows lead

Joe leads the Scalable Oversight team and the Anthropic Fellows Program. His
public writing is less about presentation mechanics and more about project
selection and model organisms. The load-bearing transferable principle from
his work: **mentorship is a weekly-cadence feedback loop** — researchers get
weekly meetings + Slack, so every clean-result artifact should be the kind of
thing that can be read and discussed in one sitting.

Implication for this skill: the clean-result issue is the weekly-meeting
artifact. If it takes > 10 min to read end-to-end, it's too long.

Source: [Anthropic Fellows Program](https://alignment.anthropic.com/2024/anthropic-fellows-program/).

---

## Owain Evans — Truthful AI / Oxford

Owain runs Truthful AI (Berkeley) and mentored much of the work cited above.
His preferred presentation style is closely reflected in the James Chua /
John Hughes slide advice — summary-first, prompt-always-shown, error-bars-always,
explicit success/failure framing.

---

## The meta-synthesis — why the skill exists

Every researcher above converges on the same thing: **the delta between
"messy raw results" and "clean mentor-grade presentation" is not polish —
it is compression, rigor, and honesty.**

A clean result answers:
1. What is the ONE claim? (Neel: distillation)
2. What is the key number and its uncertainty? (Ethan, James, John: error bars)
3. What would falsify it, and has that been tested? (Neel: red-team)
4. Is the evidence strong enough that a skeptic updates? (Neel: evidence threshold)
5. What are the caveats, stated upfront? (Neel: present limitations honestly)
6. Can a reader rerun it? (reproducibility card)

The skill's job is to make sure every clean-result artifact answers all six
before it lands in the mentor's inbox.

**Newcomer accessibility.** Joe Benton's principle — the clean-result is the
weekly-meeting artifact — implies it may be read by collaborators, advisors,
or external reviewers who lack project context. The Background subsection
must bridge the gap between "I know nothing about this project" and "I
understand why this experiment matters" in 1-2 sentences. This is not just
polish: a Background that assumes familiarity with persona coupling or EM
loses half the audience before the methodology.

## LessWrong / Alignment Forum research-post style — origin of the AI TL;DR

The `## AI TL;DR` H2 (added 2026-05-07) is 3-5 unlabeled bullets (or a
short paragraph) in the LessWrong research-post tradition. The audience
pattern there is specific: technically literate, allergic to academic
hedging, strongly rewarding of explicit calibration, and almost always
skimming first and deep-reading second. The TL;DR determines whether
anyone reads further, what the work gets cited *as*, and whether the
reader walks away over- or under-confident.

**Convention provenance — be honest about what's external vs. internal.**
LessWrong / Alignment Forum has *organic* conventions, not a centralized
style guide. Real LW posts commonly use (a) an `Epistemic status:` line
at the top declaring confidence + effort, and (b) a free-form TL;DR —
paragraph or bullets — summarizing the post's claims. **Bullets on LW
are unlabeled** (3-5 statements summarizing claims, not prefixed by
structural roles). Do NOT use labels like `**Setup.**` /
`**Headline.**` / `**Why it matters.**` / `**Scope.**` — those are not
a real LW convention; they were a project mis-attribution that was
corrected on 2026-05-08.

**LW style applies to the entire AI Summary, not just the AI TL;DR.**
Background, Methodology, Results-Main-takeaways, and Next-steps should
all read in LW register: short bullets or short prose, plain English,
concrete numbers with comparison anchors, active first-person voice,
no project-internal compound nouns ("BPE-prefix-bound mechanism",
"canonical-vs-paraphrase cliff"). The AI Summary is *also* a research
write-up — write it the way LW research posts are written, not the way
academic abstracts are written.

**See `lw-tldr-examples.md`** in this directory for verbatim TL;DRs
from real LW research posts (Model Organisms for Emergent Misalignment,
SAE features for refusal/sycophancy, Emergent Misalignment & Realignment,
AI Safety at the Frontier highlights), a style-rules summary derived
from them, an anti-pattern from this codebase, and a 5-question
drafting checklist. Read it before drafting any AI TL;DR.

**For drafting the AI Summary** (Background, Methodology, Results, Next
steps subsections), see the **full-post examples** under
`lw-post-examples/`. Three real LW research posts captured verbatim —
*Model Organisms for Emergent Misalignment* (compact), *SAE features
for refusal and sycophancy* (finding-driven), and *Emergent
Misalignment & Realignment* (closest to our template structure with
explicit Background / Research Questions / Our experiments / Results /
Open Questions sections). Each file's header explains what it
exemplifies for which AI-Summary subsection. Read the corresponding
subsection of one of these before drafting yours, and match the
register: short bullets, plain English, concrete numbers with
comparison anchors, active first-person voice.

The four beats below are a *content* discipline (what the bullets/
sentences should collectively cover) — not labels to print verbatim.
Hit the four beats organically; let the bullets stand on their own.

1. **Setup** — what problem, what models, what method.
2. **Headline finding** — the strongest concrete claim with the key
   number and N inline.
3. **Why it matters** — takeaway for the broader research program,
   not just for this experiment.
4. **Scope / limitation** — pre-empts the most obvious objection
   (single seed, in-distribution only, narrow model family, judge-
   based metric, etc.).

Useful test from the LW guide: *would a reader who only reads the TL;DR
walk away with an accurate, calibrated impression of the work? If
they'd come away too excited, it's overclaiming. If they'd come away
unsure what was done, it's underwriting.*

Calibrated, first-person voice is the LW shibboleth — "we found",
"I think", "~70% confident", "weak evidence for…" all read better than
"results indicate" / "it has been shown" / unmarked "clearly". The
verifier permits first-person hedges; the existing forbidden-stats
list (no Cohen's d, no named tests in prose, no credence intervals
inline) still applies.

**Why a separate `## Human TL;DR` above the AI one.** The user
sometimes wants to ship a pithier, more opinionated take than the
AI's calibrated paragraph — either because the AI underweights an
intuition the user has from running the experiment, or because the
mentor-facing read deserves a one-line punchline that a 3-5 sentence
LW paragraph can't capture. Reserving the topmost section for the
user enforces the discipline that the human read is the FIRST thing a
reader sees, not buried in a Detailed report. Drafts ship with a
placeholder line; the user overwrites it post-promotion.

## #275 additions — what newcomers consistently trip on

These four operational principles sharpen the rule above based on real
reader feedback collected across the first ~30 promoted clean-results.
The verifier (`scripts/verify_clean_result.py`) enforces each one as a
strict-mode check.

- **Project-internal acronyms cost the reader a context-switch.** The 6
  tokens `H1, H2, H3, P1, P2, P3` are the project's most-confused —
  define on first use (e.g. `H1 = primary hypothesis`,
  `P1 (coupling phase)`, `H2: leakage`). Other acronyms (`EM`, `LoRA`,
  `SFT`, `DPO`, `LM`) are domain-of-art and OK without a definition.
  Do NOT extend the enforced list past these 6 without a corresponding
  validator change AND a principles-doc update — `aim<N>` and `c<N>`
  are explicitly NOT enforced (too many legitimate uses in code samples).
- **Each claim self-contained.** A reader who lands on the issue from a
  RESULTS.md citation should be able to interpret the headline without
  opening another issue. Cross-references augment claims; they do not
  carry them. State the percentage and N inline on every takeaway
  bullet.
- **Background motivates from past work.** Every clean-result body
  answers `why was this run?` in the first paragraph by linking the
  prior issue(s) that motivated it (≥1 `#<issue>` reference distinct
  from the current issue). A reference to the current issue itself
  does NOT count.
- **Sanity-check raw outputs before writing the interpretation.**
  Aggregate numbers can lie if the judge mis-labelled, the prompts
  collapsed, or the generations are non-English garbage. The analyzer
  must spot-check 5 random rows and paste them at the top of the
  interpretation body under an H3
  (`### Raw-output spot check (5 random rows)`); see
  `.claude/agents/analyzer.md` Step 1.5. Visible fishiness flows into
  the confidence rationale.
