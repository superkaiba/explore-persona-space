# Clean-result spec — markdown

The canonical spec for clean-result body shape, voice, sections, and
anti-patterns. The mechanical verifier is **`scripts/verify_task_body.py`**.
The format is **markdown** with YAML frontmatter.

## Required body shape

**The 2-content-section model** (migrated 2026-W22, task #454). The body
carries THREE required H2 sections, in this exact order, with the second
(`## TL;DR`) absorbing what used to live in a separate `## Details`
section and the third (`## Reproducibility`) absorbing the Parameters
table + Confidence sentence:

1. `## Human TL;DR` — Thomas's own section in his voice, drafted by the
   analyzer as a MINIMAL STUB — just the literal word `placeholder` as
   the section body — that Thomas overwrites himself before sending to
   the mentor (with his own headline / takeaways / how-this-updates-me
   framing). Voice: first-person, casual, in his own words. Do NOT
   pre-fill any Headline / Takeaways / How-this-updates-me structure.
   See `analyzer.md` Step 1 for the stub template.

2. `## TL;DR` — the LessWrong-style narrative. Two-part shape:

   - **Opens with `### Motivation`** (a labeled H3, not a bullet). This
     subsection sets up why the experiment matters — prior tasks (cited
     via `[#K](https://eps.superkaiba.com/tasks/K)`), the question
     walked in with, the prior the analyzer held. First-person, plain
     language.

   - **Followed by one `### <finding>` H3 per result.** Each result H3
     names what the reader is about to learn (a story-beat headline,
     NOT a deliverable label — see voice rules below). Inside each
     result H3:

     1. A short **setup paragraph** (1-3 sentences) framing what the
        figure will show and why we're looking now.
     2. **Exactly ONE inline figure** (`![alt](permanent-url)` on a line
        by itself, blank line before and after) with a markdown
        blockquote caption (`> **Figure.** *italic lead.* plain
        caption text…`). See "Figure caption shape" below.
     3. A **read paragraph** (1-3 sentences) calling out what's
        striking — surprises, where outliers go, monotonicity, what
        the figure CAN'T tell you.
     4. **One cherry-picked raw-completion example per artifact** the
        result rests on (a fenced code block or a `<details>` table),
        preceded by the literal `cherry-picked for illustration` (or
        a random-sample disclosure like `first three of 400
        completions`) AND by a link to the **raw text-level artifact**
        (HF Hub `/tree/<sha>/.../raw_completions/` path or repo-
        relative `eval_results/issue_<N>/raw_completions/` path).
     5. A **`<details>` dropdown** with 3-5 more cherry-picked examples
        + a link to ALL raw completions for that artifact (the
        complete bucket on HF Hub, pinned to the commit SHA).

     Every result H3 except `### Motivation` MUST stand alone — the
     reader can land on it directly and understand the finding without
     re-reading the previous result.

   **No separate `## Details` section.** Everything that used to live
   there (definitions, training notes, eval-rationale prose, sample
   completions, "Why this test" narrative) moves UP into the per-result
   H3 narrative.

   **No `## Figure` H2.** Figures live inline under their result H3
   (one figure per result).

   **No `### Methodology corrections` H3.** When a methodology
   correction is load-bearing for interpreting a finding, fold it into
   the relevant result's setup or read prose — do not collect them in
   a separate section.

   **No `### Next steps` H3 by default.** Skip unless there is
   genuinely useful follow-up to queue. Hard exception: when raw
   completions were not uploaded for this run, the body MUST surface
   the "re-run with raw-completion upload" note in the relevant result's
   prose.

3. `## Reproducibility` — agent-facing appendix at the bottom. Required
   content, in order:

   - **`**Parameters:**`** — the parameters table (base model, adapter,
     optimizer, steps, seeds, eval rig, hardware, wall time, Hydra
     config slug, etc.). Absorbed from the retired `## Details`
     section.
   - **`**Artifacts:**`** — links to training data, model checkpoints,
     eval JSONs, figure source, raw completions. **Training/eval data:
     embed a `<details>` dropdown of 5 example rows + a link to the
     full data file** under whichever result H3 the data is most
     relevant to (NOT here — the dropdown lives in the TL;DR result
     section; this Artifacts block just lists the full artifact links).
   - **`**Compute:**`** — wall time, GPU type/count, pod label.
   - **`**Code:**`** — dataset-build script, pipeline driver, Hydra
     config, git commit hash, one-block reproduce snippet.
   - **The `Confidence: LOW|MODERATE|HIGH — <rationale>` sentence**
     (≥20 chars of rationale after the dash) lives in this section
     (last paragraph by convention; the verifier scans the whole body
     and the level MUST match the title's `(... confidence)` marker).
     There is NO separate "Why confidence is where it is" section.

All URLs in Reproducibility are pinned to permanent refs (HF Hub
`/tree/<ref>` or `@<ref>`, WandB `/runs/<id>`, GitHub `/blob/<sha>` or
`/tree/<sha>`; never `main` / `master` / `HEAD`). `n/a` accepted as an
explicit non-applicable marker. No `TBD`, `{{`, `default`, `see config`
sentinels. **Write MDX-safe markdown — the dashboard renders bodies
through an MDX parser.** (a) URLs use `[label](url)` form only — never
`<https://...>` autolinks (MDX reads `<https` as a JSX tag and fails on
the `/` after `:`). (b) No `<` immediately before a digit (`p<0.05`,
`n<10`) — write ` < ` with spaces or wrap in backticks. (c) Table-cell
tokens with inner pipes (e.g. `<|im_start|>`) escape the pipes and wrap
in backticks: `` `<\|im_start\|>` ``. Fenced code blocks and inline code
spans are exempt (except a pipe-containing code span on a real GFM
table-row line). The rule applies everywhere in the body, not just
Reproducibility. Verifier check 14 (`check_mdx_safe_urls`) FAILs all
three classes.

### Stray `## Details` is a FAIL

A NEW body that includes a `## Details` H2 (or `## Figure` H2 — see
below) is rejected by the verifier. This forces clean migration to the
2-content-section model: bodies cannot half-migrate by stripping the
Details prose while leaving the H2 in place. The verifier surfaces a
clear FAIL pointing at this section.

### (Deprecated) `## Figure` H2

`## Figure` is fully retired for NEW write-ups. A stray `## Figure` H2
in a new body is treated the same way as a stray `## Details` H2 — the
verifier FAILs and the analyzer must inline the figure under the
relevant result H3 inside `## TL;DR`. Legacy bodies (already promoted
pre-2026-W22) are not re-verified, so the migration is forward-only.

Title format (the H1 line):

```
# <one-sentence claim> (LOW|MODERATE|HIGH confidence)
```

The confidence level in the title MUST match the `Confidence: ...`
sentence in `## Reproducibility`.

## Figure caption shape — markdown blockquote + bold "Figure." prefix

**Every figure caption inside a `## TL;DR` result H3 wraps in a markdown
blockquote (`> ` prefix) and uses this internal form:**

```
> **Figure.** *One-sentence lead claim in italics.* Remaining caption
> prose in plain text — definitions, n per condition, panel meanings,
> color mapping, what the reader should look at, what the figure does
> NOT show.
```

The `> ` blockquote prefix is what makes the caption visually distinct
from the body prose around it. Without it, the dashboard's markdown
renderer collapses `body text. ![alt](url) caption text.` into a
single paragraph where the caption reads as continuation of the body.

**Layout inside a result H3:**

```markdown
### <Result headline>

<Setup paragraph: what we did, what's plotted, what to look for.>

![alt text with axis labels + a numerical claim.](https://raw.githubusercontent.com/.../figure.png)

> **Figure.** *Italic lead claim.* Plain-text caption body with
> definitions, ns, color mapping, reading guide.

<Read paragraph: what's striking, where outliers go, what the figure
can't tell you.>

<cherry-picked-label prose with raw-completion link>

```
EVAL PROBE   (...)
MODEL OUTPUT (...)
```

<details>
<summary>3 more cherry-picked completions</summary>

[3-5 more examples or a link list]

Full <M> raw completions: [bucket/raw_completions/](https://huggingface.co/.../tree/<sha>/.../raw_completions/)

</details>
```

Three discipline points:
1. **Blank line BETWEEN body prose and image** (otherwise the image
   renders inline with body text).
2. **Blank line BETWEEN image and caption** (otherwise the caption
   joins the image's paragraph).
3. **No 4-space indent needed** — result H3s are not list items in the
   new spec, so no list-continuation indent applies. Just keep the
   blank lines.

Originated in `iterations.md` § 2026-05-11 "Figure captions blend
visually into surrounding body prose"; current canonical surface for
the rule is this section + `CLAUDE.md` § Experiment Report Structure
("Figure captions wrap in a markdown blockquote..."). Analyzer drafts
must produce this shape on the first pass, not as a promotion-time
fix.

## Mechanical checks (`verify_task_body.py`)

1. Title ends with `(LOW|MODERATE|HIGH confidence)`.
2. Three required H2 sections present in order
   (`## Human TL;DR`, `## TL;DR`, `## Reproducibility`). A stray
   `## Details` or `## Figure` H2 is a FAIL (forces clean migration to
   the 2-content-section model; legacy bodies pre-2026-W22 are
   forward-grandfathered because the verifier never re-runs over them).
3. `## TL;DR` opens with the Motivation section — either an
   `### Motivation` H3 (preferred) or a `**Motivation:**` boldface
   bullet (legacy form, still accepted).
4. At least one `![alt](url)` markdown image exists inline under
   `## TL;DR`.
5. (Soft) Figure-caption sanity — vacuously satisfied when no legacy
   `## Figure` H2 is present (inline-image alt text + blockquote
   caption inside the result H3 carry the discipline; the analyzer is
   instructed to write descriptive alt text and blockquote captions).
6. Confidence sentence (anywhere in the body — typically the last
   paragraph of `## Reproducibility`) matches the title's level and
   carries ≥20 chars of rationale after the dash.
7. Reproducibility contains all three boldface subgroup labels
   verbatim: `**Artifacts:**`, `**Compute:**`, `**Code:**`.
8. Reproducibility URLs are pinned to permanent refs (HF Hub
   `/tree/<sha>` or `@<sha>`, WandB `/runs/<id>`, GitHub
   `/blob/<sha>` or `/tree/<sha>`; never `main`, `master`, `HEAD`).
9. Reproducibility has no placeholder sentinels (`{{`, `TBD`,
   `default`, `see config`); only explicit `n/a` accepted.
10. Cherry-picked label preceding every sample-output fenced block
    in `## TL;DR` (literal `cherry-picked for illustration`, or an
    explicit random-sample disclosure like `first three of 400
    completions`).
11. Qualitative-data link preceding every sample-output fenced
    block in `## TL;DR` (HF Hub `/tree/<sha>/.../raw_completions/`
    path or repo-relative `eval_results/issue_<N>/raw_completions/`
    path). Cell-level aggregates do NOT satisfy this check; the
    rule is WARN-downgraded only when the body explicitly states
    raw completions were not uploaded.
11b. Planned-vs-actual denominator consistency — within-body check
    that the TL;DR's `X of N` headline denominator matches any
    `M of N` documented scope in the rest of the body (the
    `### Methodology corrections` H3 is no longer required as a
    discrete section; the check fires on any in-body Methodology-
    corrections-style claim it finds).
12. `## Figure` H2 deprecation — bodies that carry a stray `## Figure`
    H2 are rejected via check 2 (forces clean migration). The check 12
    function remains as a hook for future WARN-only nudges but no
    longer triggers on legacy patterns.
13. TL;DR narrative flow (WARN-only) — outline-label H3s in
    `## TL;DR` (`### Headline result`, `### Findings`, etc.) and >2
    consecutive figures with no prose between (figure-dump). Both
    surface as WARN; critic-side LM judgment (`clean-result-critic`)
    catches the semantic cases.
14. MDX-safe prose (`check_mdx_safe_urls`) — see "Required body
    shape" above for the three classes (autolinks, `<digit`,
    table-cell `<|`).
15. Reproducibility "committed at commit `<sha>`" claims resolve —
    conservative cross-check that any committed-at-`<sha>` claim in
    Reproducibility paired with a repo-relative artifact path
    actually resolves in `git cat-file`.

## Anti-pattern audit (`audit_clean_results_body_discipline.py`)

Catches prose-level violations the verifier doesn't:

- Pre-registration mentions in TL;DR
- Effect-size names in prose (Cohen's d, η², r-as-effect-size,
  Δ-framed-as-effect)
- Named statistical tests in narrative prose (paired t-test, Fisher
  exact, Mann-Whitney, Wilcoxon, bootstrap test)
- Inline `value ± err` credence intervals (chart error bars fine)
- Project-internal condition labels (`C1`, `C2`, `C2'`, `H1`, `P1`)
- Math-style subscripts/superscripts in prose (`R_BgivenA^P2`,
  `f_θ`)
- GCG / PAIR / `H_a` / `REJECTED` / `Δ-Npp` / `slope[low,high]` /
  letter labels / `Bin A/B/C`
- **`byte identical` / `byte-identical`** — banned phrasing (2026-W22).
  Express equivalence in plain English ("identical at every byte",
  "the two files matched exactly", "no diff"); the catch-phrase reads
  as AI-slop in research prose.

## Voice

- `I`, not `we` — single-researcher workflow.
- Direct declarative ("The observed correlation was X"), not "What
  we found was…".
- No fluff transitions in `## Human TL;DR` and the TL;DR opening
  paragraphs: "One more wrinkle:", "the buried lede was", "funnily
  enough", "the real surprise was", "the kicker is". (Connective
  tissue inside result H3 prose — "Then I tried", "But that didn't
  replicate", "I expected X — what I got was Y" — IS welcome; it
  keeps the narrative flowing.)
- No `## Findings` / `## Background` / `## Methodology` / `## Setup` /
  `## Details` H2s — every reader-facing finding lives under a result
  H3 inside `## TL;DR`.
- No "Standing caveats" section — caveats fold into the result H3
  read paragraph or the Confidence sentence in Reproducibility.
- Inline math `\(...\)`, display math `\[...\]`. Keep math out of
  plot labels and figure captions.
- **Never write `byte identical` or `byte-identical`** anywhere in the
  body. Use plain English: "the two files matched exactly", "every
  byte agreed", "no diff between the runs".

## Migration note (2026-W22)

The 4-section model (`## Human TL;DR` / `## TL;DR` / `## Details` /
`## Reproducibility`) was replaced by the 2-content-section model
above on 2026-W22 (task #454). The verifier is **forward-only**: it
runs at analyzer pre-publish and clean-result-critic pre-pass, never
retroactively over already-promoted bodies. The ~95 legacy
`has_clean_result=true` bodies stay as-is for historical viewing — none
will be re-verified. In-flight `awaiting_promotion` drafts that still
use the 4-section shape FAIL on next analyzer/critic re-run and get
re-drafted under the new spec; this is acceptable (drafts always rebuild
cleanly from cached results + figures).

**Target exemplar** (the END state new bodies should aim for):
`tasks/awaiting_promotion/432/body.md` (with `## Human TL;DR`
reduced to the bare `placeholder` stub).

## What this directory still owns

- **`iterations.md`** — append-only log of corrections + the rules
  they produced. Continue to log here when an iteration during
  `/promote-clean-result` uncovers a generalisable rule. The
  "fold into" pointer should target THIS file for new structural
  rules, or `scripts/verify_task_body.py` for new mechanical checks.
- **`lw-post-examples/`** — 3 verbatim LessWrong research posts kept
  for register reference. The result-H3 narrative is more compressed
  than a LW post but the prose discipline (concrete numbers,
  comparison anchors, plain English, no undefined jargon) carries
  over.

## Legacy Sagan-card HTML bodies (grandfathered)

The 20 bodies imported from the old Sagan dashboard that carry a
`<!-- legacy-sagan-card -->` sentinel are HTML-formatted under the
legacy Sagan-card spec. They stay as-is for historical viewing.
`verify_task_body.py` skips them with a one-line note. The legacy
verifier `scripts/verify_sagan_card.py` still applies to those
specific bodies only — it is NOT used for new markdown bodies.

## Calling sites

- `.claude/agents/analyzer.md` — drafts the body per this spec.
- `.claude/agents/clean-result-critic.md` +
  `codex-clean-result-critic.md` — critique against the lenses and
  run `verify_task_body.py` +
  `audit_clean_results_body_discipline.py`.
- `.claude/skills/promote-clean-result/SKILL.md` — for legacy HTML
  bodies, optionally converts them to markdown on promotion.
- `CLAUDE.md` § "Experiment Report Structure" — points at this spec.
