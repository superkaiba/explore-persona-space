# Clean-result format v2 — "condensed experiment story"

Section names: **Human TL;DR** then **Experiment summary** then **Reproducibility**.

Status: **DRAFT for Thomas's sign-off.** Once approved, this content
replaces `.claude/skills/clean-results/SPEC.md` and propagates to
`CLAUDE.md`, `.claude/agents/analyzer.md`, `clean-result-critic.md`,
`codex-clean-result-critic.md`, and `scripts/verify_task_body.py`
(see § Propagation checklist at the bottom).

---

## Why v2

v1 was mentor-skim-optimized: a terse structured TL;DR, **one hero
figure**, and the actual experimental data (training rows, eval probes,
model outputs) compressed into a single cherry-picked triple buried in
the TL;DR. Dan's repeated push (2026-05-28 1:1) is the Anthropic-interp
instinct: **show the real data at every stage, not just an aggregate
bar.** The persona-features (Wang/Mossing) and persona-vectors
(Chen/Lindsey) papers both lean on real completions.

v2 turns the body into a **condensed experiment story**: a stage-by-stage
walkthrough where each stage shows the data it consumed or produced,
paired with the figure(s) for that stage. There is no single hero figure
— many steps, many figures, each anchored to real data — but it stays
**brief**: each stage is a lead line + the data/figure + a one-line read,
not a LessWrong essay.

### What changed from v1

| v1 | v2 |
|---|---|
| 4 H2s: Human TL;DR / TL;DR / Details / Reproducibility | **3 H2s: Human TL;DR / Experiment summary / Reproducibility** |
| Structured TL;DR (Motivation / What I ran / Results) | **Dropped** — content distributes into Experiment summary stages |
| `## Details` LessWrong narrative | **Dropped** — replaced by the condensed Experiment summary |
| One hero figure / one-takeaway-one-figure (Lens 9) | **Dropped** — figures appear at the stage that produced them |
| `## Figure` H2 | Stays deprecated/gone |
| One buried end-to-end train+eval+output triple in TL;DR | **Show the data at each stage** (train rows, probes, completions) as a hard requirement |
| Blockquote-caption 4/8-space indent gymnastics | **Gone** — figures live in narrative, not nested list bullets |
| Confidence sentence / parameters / methodology corrections in Details | Move to the **end of Experiment summary** |

### What v2 keeps (honesty, not aesthetics)

- Title confidence tag (`(LOW|MODERATE|HIGH confidence)`) that matches
  the Confidence sentence.
- Plain-English condition names end to end (no Hydra slugs / letter
  labels in reader-facing text).
- p-values + N in prose; **no** effect sizes, **no** named tests in
  prose.
- Permanent SHA-pinned URLs everywhere; MDX-safe markdown.
- Raw-alongside-processed (a residualized/binned/aggregated figure shows
  its raw counterpart; a controlled estimate quotes the raw one too).
- Qualitative-data links + cherry-picked labels on every sample block.
- `I`, not `we`. Methodology corrections at the bottom.

---

## Required body shape

**Three required H2 sections, in this order:**

```
# <one-sentence claim> (LOW|MODERATE|HIGH confidence)

## Human TL;DR
## Experiment summary
## Reproducibility
```

Extra H2s after `## Reproducibility` are allowed. The H1 title
confidence tag MUST match the `Confidence:` sentence at the end of
`## Experiment summary`.

### 1. `## Human TL;DR` — your voice, the skim

The only summary. First and only fast read for a mentor. Three labelled
sub-blocks:

```markdown
## Human TL;DR

**Headline.** <1 sentence — what stood out, what you'd tell Dan in one breath.>

**Takeaways.**
- <2-4 short bullets — the qualitative beats; what surprised you, what's quietly important.>

**How this updates me.** *<1-3 sentences — what belief moved, what's now more/less likely, what you'll do differently. Your Bayesian-update line.>*
```

**Analyzer drafts Headline + Takeaways from the data** (these are
factual and auto-draftable) and leaves **How this updates me** as an
italic stub for Thomas to fill before sending to the mentor — that line
is the genuine belief update and only he writes it. The verifier checks
the three sub-blocks are present; it does not check the prose of the
stub.

### 2. `## Experiment summary` — the condensed experiment story

The heart of v2. A stage-by-stage walk through the experiment that shows
the real data at each stage. **Condensed**: each stage is a bold lead
line (1-2 sentences) + the data block and/or figure for that stage + an
optional one-line read. Not an essay; no setup-paragraph + read-paragraph
ceremony per figure.

**Recommended stage skeleton (adapt freely — reorder, merge, or drop
stages that don't apply):**

1. **Motivation / question** — 1-2 sentences: why this was run, what you
   expected. Cite priors via `[#K](https://eps.superkaiba.com/tasks/K)`.
2. **Training data** — how it was built; **≥1 verbatim training row**
   (the per-condition contrast where it matters); a permanent link to the
   complete `training_data/`.
3. **Training** — only if there was training; a loss/metric figure if it
   carries information. Often one line.
4. **Eval** — the framings/probes; **≥1 verbatim probe per distinct
   framing** (or one representative probe for single-framing rigs); link
   to the full probe set.
5. **Outputs** — **firing + non-firing verbatim completions per
   condition**; a permanent link to the complete `raw_completions/`.
6. **Result(s)** — one bold beat per finding: the figure(s) (raw +
   processed where applicable) + the number with p-value and N + a
   one-line read. Multiple result beats and multiple figures are expected.
7. **Interpretation** — 1-2 sentences: what the evidence says, what
   alternative survives.
8. **Methodology corrections** — only if any; the **last** beat (plan
   deviations, mid-run bugs, threshold changes). Keep them out of the
   title.

End the Experiment summary with:

- A **parameters table** (compact `key | value`; bare Hydra slugs allowed
  only in the `config` row).
- The **Confidence sentence**, its own paragraph, exact shape:
  `Confidence: LOW | MODERATE | HIGH — <one sentence naming the binding
  constraint (LOW/MODERATE) or the surviving evidence (HIGH)>.`
  Level matches the title tag; ≥20 chars of rationale.

**Condensed style rules:**

- A stage is a `**Bold lead.**` paragraph, not a `### H3` essay section.
  (H3s are allowed when a result section genuinely needs sub-beats, but
  the default is bold-lead paragraphs — lighter weight.)
- Data blocks use a markdown blockquote (`> `) or a fenced code block.
  Keep each example to the verbatim Q/A/probe/completion — no padding.
- Each figure gets a one-line `> **Figure.** <read>` caption. No
  separate setup paragraph + read paragraph; the stage lead is the setup.
- Multiple figures are fine and expected; there is no hero-figure rule.

### 3. `## Reproducibility` — unchanged

Three boldface subgroups in order: **Artifacts:**, **Compute:**,
**Code:**. Every URL pins a permanent ref (HF Hub `/tree/<sha>`, WandB
`/runs/<id>`, GitHub `/blob/<sha>` — never `main`/`master`/`HEAD`).
Empty fields write `n/a`; no `{{`/`TBD`/`see config`/`default` sentinels.

---

## The "show the data" requirement (new hard rule)

This is the rule Dan asked for and the one v2 enforces mechanically.

For an experiment that **produces text generations** (training data is
`(persona, Q, A)`; eval data is `(persona, framing, Q)`; outputs are
completions), the Experiment summary MUST contain all three:

1. **≥1 verbatim training row** AND a permanent link to the complete
   `training_data/` on HF.
2. **≥1 verbatim eval probe** (per distinct framing where there are
   several) — a link to the full probe set when probes are a separate
   artifact.
3. **≥1 firing AND ≥1 non-firing verbatim model completion** AND a
   permanent link to the complete `raw_completions/` on HF.

Every sample block carries a **cherry-picked label** ("cherry-picked for
illustration" or a random-sample disclosure) and the qualitative-data
link sits in the same stage.

**Exemption** — experiments with no completions (pure activation / probe-
direction / cluster-membership / linear-fit-only analyses). Document the
skip with one line in the relevant stage: *"(no generation-style outputs
in this experiment; showing activation/probe data instead.)"* These
experiments still show their data (the activations, the probe directions,
the cluster assignments) — the rule generalizes to "show the unit of data
the experiment operated on."

**Mechanical vs critic split** (per the "hard-check data + links, relax
figures" decision):

- **Verifier (mechanical, FAIL):** a `training_data/` HF link present; a
  `raw_completions/` HF link present; ≥1 cherry-picked-labelled sample
  block present — unless the exemption line is present.
- **clean-result-critic (judgment, FAIL):** per-condition coverage,
  firing-vs-non-firing both shown, the three samples telling a coherent
  story, probes shown per framing, figures having a read line.

---

## Voice + statistics (kept from v1, condensed)

- `I`, not `we`.
- Plain-English condition names everywhere reader-facing.
- p-values + N in prose. No Cohen's d / η² / r-as-effect / Δ-as-effect;
  no named tests in prose; no inline `value ± err`. Error bars on charts
  are fine.
- No fluff transitions ("the buried lede was", "the real surprise was",
  "interestingly").
- MDX-safe: `[label](url)` not `<url>`; no `<` before a digit (`p < 0.05`
  with spaces or `` `p<0.05` ``); escape inner pipes in table-cell tokens
  (`` `<\|im_start\|>` ``).

---

## Mechanical checks (`verify_task_body.py` v2)

1. Title ends with `(LOW|MODERATE|HIGH confidence)`.
2. **Three** required H2s present in order: `## Human TL;DR`,
   `## Experiment summary`, `## Reproducibility`. (Drop the TL;DR + Details +
   Figure checks.)
3. Human TL;DR carries the three sub-blocks (`**Headline.**`,
   `**Takeaways.**`, `**How this updates me.**`).
4. ≥1 `![alt](url)` image somewhere in the body (placement no longer
   checked).
5. Confidence sentence at the end of Experiment summary matches the title level,
   ≥20 chars rationale.
6. Reproducibility has all three boldface subgroups + permanent URLs + no
   sentinels.
7. **NEW — show-the-data:** unless the exemption line is present, the
   body contains a `training_data/` HF link, a `raw_completions/` HF
   link, and ≥1 cherry-picked-labelled sample block.
8. Cherry-picked label + qualitative-data link precede every sample
   block.
9. MDX-safety (autolinks, `<digit`, table-cell `<|`).

Dropped from v1: TL;DR-three-bullets check; `## Figure` checks;
one-takeaway-one-figure; the buried end-to-end-triple check; the
blockquote-indent checks.

---

## Worked template (placeholders — populate with real data)

```markdown
# Off-persona marker leakage survives one epoch of clean SFT (HIGH confidence)

## Human TL;DR

**Headline.** <1 sentence in your voice.>

**Takeaways.**
- <takeaway 1>
- <takeaway 2>

**How this updates me.** *<your Bayesian-update line — fill before sending.>*

## Experiment summary

**Why I ran it.** <1-2 sentences + prior links, e.g. [#376](https://eps.superkaiba.com/tasks/376).>

**Training data.** SFT on (persona, Q, A) triples; the marker ` ※` is
appended to every answer under the evil persona only. <N> rows × <K> seeds.
Cherry-picked for illustration. Full data:
[`issueN_<slug>/training_data/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/<sha>/issueN_<slug>/training_data).

> persona "evil"  —  Q: "<verbatim training question>"
> A: "<verbatim training answer> ※"

**Eval.** <F> framings probe whether the marker fires off-persona. Full
probe set: [`.../probes/`](<url>).

> framing #3 (decoy correction): "<verbatim probe>"

**Outputs.** The marker fired <x>/<n> off-persona. Cherry-picked for
illustration. Full completions:
[`issueN_<slug>/raw_completions/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/<sha>/issueN_<slug>/raw_completions).

> fired (assistant persona, seed 0): "<verbatim completion ending in ※>"
> did NOT fire (assistant persona, seed 1): "<verbatim clean completion>"

**Result.** Off-persona leakage drops from <a>% to <b>% after one epoch
of length-matched benign SFT (p < 0.05, N=<n>).

![Off-persona marker firing rate before vs after one epoch of benign SFT, two conditions, 95% CI error bars.](https://raw.githubusercontent.com/<owner>/<repo>/<sha>/figures/issueN/leakage_pre_post.png)

> **Figure.** *One epoch of clean SFT silences the marker.* Bars are
> off-persona firing rate; error bars 95% CI; N=<n> per condition.

**Interpretation.** <1-2 sentences: what survives, what alternative is ruled out.>

| parameter | value |
|---|---|
| base model | Qwen-2.5-7B |
| marker | ` ※` (token 83399) |
| config | `condition=<slug> seed=<S>` |

Confidence: HIGH — <one-line binding-constraint or surviving-evidence rationale>.

## Reproducibility

**Artifacts:** adapter [<repo>/tree/<sha>](…); training data [<path>](…);
raw completions [<path>](…); WandB [/runs/<id>](…); eval JSON
`eval_results/issueN/…`; figure source `figures/issueN/…`.

**Compute:** <wall time>, <GPU>, <pod>.

**Code:** `scripts/train.py` + `scripts/eval.py`; commit `<sha>`; Hydra
`condition=<slug>`; reproduce: `git clone … && git checkout <sha> && uv run …`.
```

---

## Propagation checklist (after sign-off)

1. Replace `.claude/skills/clean-results/SPEC.md` with the v2 spec above.
2. `CLAUDE.md` § "Experiment Report Structure" + § "Sample-output
   discipline" + § "Voice + Statistics" — rewrite to the 3-H2 shape,
   drop hero-figure / one-takeaway-one-figure / Figure-H2 language, add
   the show-the-data rule.
3. `.claude/agents/analyzer.md` — Steps 3/3.5/4/5: drop hero-figure
   "pick the single chart" language; add stage-walkthrough drafting +
   show-the-data deliverables; Human TL;DR auto-draft of Headline +
   Takeaways.
4. `.claude/agents/clean-result-critic.md` + `codex-clean-result-critic.md`
   — collapse Lens 9 (one-takeaway-one-figure) into a show-the-data +
   per-stage-coverage lens; keep Lenses 2/3/4/8/11; retarget to 3 H2s.
5. `scripts/verify_task_body.py` — implement the v2 checks; keep the
   legacy-body skip; bump the check count in the docstring.
6. `.claude/skills/clean-results/exemplars/` — add one v2 worked example
   (populate the template with a real result, e.g. #382 or #390).
7. Update `eval_results/INDEX.md` references / any doc that cites the
   "four required H2 sections".
8. Run `scripts/workflow_lint.py` + `verify_task_body.py` on the new
   exemplar; confirm green.
```
