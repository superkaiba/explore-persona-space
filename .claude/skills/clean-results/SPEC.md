# Clean-result spec — markdown

The canonical spec for clean-result body shape, voice, sections, and
anti-patterns lives in **`.claude/plans/task-workflow-migration.md`
§ 10**. The mechanical verifier is **`scripts/verify_task_body.py`**
(thirteen checks). The format is **markdown** with YAML frontmatter.

## Required body shape

**Four required H2 sections in this order** (`## Figure` is
DEPRECATED for new write-ups — figures live inline under TL;DR
Results sub-bullets; see "Where the hero figure lives" below):

1. `## Human TL;DR` — Thomas's own section in his voice. First H2,
   before the auto-generated `## TL;DR`. The analyzer drafts this as
   a STUB with three labelled sub-blocks (Headline / Takeaways / How
   this updates me) that Thomas fills in before sending to the
   mentor. Voice: first-person, casual, in his own words — NOT the
   structured Motivation/What-I-ran/Results summary (that's the next
   H2, auto-drafted). See `analyzer.md` Step 1 for the stub template.
2. `## TL;DR` — three required bullets labelled **Motivation / What
   I ran / Results**, plus an OPTIONAL **Next steps** bullet. "I"
   voice; plain language; cite prior tasks via
   `[#K](https://eps.superkaiba.com/tasks/K)`. **Figures live inline
   under the Results bullet, NOT under a separate `## Figure` H2.**
   When the Results bullet carries more than one quantitative
   finding, split into sub-bullets (markdown 4-space indent under
   `- **Results:**`), one finding per sub-bullet, each paired with
   an inline image `![alt](url)` on the next indented line
   (one-takeaway-one-figure pattern — Lens 9; decision 2026-05-26;
   prescriptive 2026-05-27). For >3 findings or findings that don't
   decompose cleanly, use a single roll-up Results bullet linking to
   a `## Details` H3 anchor (`[Per-finding figures and reads in
   Details.](#findings)`); each finding then lives in Details as a
   story beat with its own setup + figure + read paragraph.
3. `## Details` — single narrative covering definitions, training,
   eval rationale, sample completions inline (cherry-picked label +
   qualitative-data link), "Why this test" paragraph, parameters
   table, and the `Confidence: LOW|MODERATE|HIGH — …` sentence.
   Use `### ...` H3 subheadings for each distinct subsection inside
   Details (Primary strict test / Sample completions / Plan deviations
   / Parameters / Why this test / etc.). Do NOT use bolded paragraph
   leads (`**Subsection name.**`) as inline subheadings — the
   dashboard's markdown renderer collapses them into a wall of text.
   The intro paragraph(s) that set up definitions and decoder config
   stay as plain prose at the top of Details; the H3 subsections
   begin at the first distinct sub-topic (typically the headline
   table or "Primary strict test"). The `Confidence:` sentence sits
   in its own paragraph after the Parameters table — it is NOT an
   H3.
4. `## Reproducibility` — three groups (Artifacts, Compute, Code).
   All URLs permanent: HF Hub `/tree/<ref>`, WandB `/runs/<id>`,
   GitHub `/blob/<sha>`. `n/a` accepted as an explicit non-applicable
   marker. No `TBD`, `{{`, `default`, `see config` sentinels. **Write
   MDX-safe markdown — the dashboard renders bodies through an MDX
   parser.** (a) URLs use `[label](url)` form only — never
   `<https://...>` autolinks (MDX reads `<https` as a JSX tag and fails
   on the `/` after `:`). (b) No `<` immediately before a digit
   (`p<0.05`, `n<10`) — write ` < ` with spaces or wrap in backticks.
   (c) Table-cell tokens with inner pipes (e.g. `<|im_start|>`) escape
   the pipes and wrap in backticks: `` `<\|im_start\|>` ``. Fenced code
   blocks and inline code spans are exempt (except a pipe-containing
   code span on a real GFM table-row line, where the inner pipes must
   be escaped). The rule applies everywhere in the body, not just
   Reproducibility. Verifier check 14 (`check_mdx_safe_urls`) FAILs all
   three classes: a table-aware regex layer flags class (c)'s table-cell
   `<|` tokens alongside classes (a) and (b), and an authoritative real
   MDX parse backstops every class. Incidents: task #382, 2026-05-28;
   task #399, 2026-05-28 (table-cell `<|im_start|>`).

### (Deprecated) `## Figure` H2

`## Figure` is DEPRECATED for new write-ups (decision 2026-05-27,
prescriptive). Legacy bodies that carry it (pre-2026-05-27
promotions) remain promotable — the verifier surfaces a WARN, not a
FAIL — but new bodies inline figures under TL;DR Results sub-bullets
instead. See "Where the hero figure lives" below for the rare
one-hero-finding exception that may still emit the H2.

Title format (the H1 line):

```
# <one-sentence claim> (LOW|MODERATE|HIGH confidence)
```

The confidence level in the title MUST match the `Confidence: ...`
sentence in `## Details`.

## Where the hero figure lives

**Prescriptive default for new bodies (decision: 2026-05-27).** Figures
live inline under `## TL;DR` Results sub-bullets (one-takeaway-one-figure
pattern, Lens 9). Do NOT emit a separate `## Figure` H2 by default.

`## Figure` is DEPRECATED for new write-ups. Legacy bodies that carry
the H2 (pre-2026-05-27 promotions) stay promotable — the verifier
surfaces a WARN (not a FAIL) when the H2 is present. The rare exception
that justifies a new `## Figure` H2 is a one-hero-finding body where a
single chart carries the entire Results story AND inlining it under a
Results sub-bullet would feel awkward (e.g. a single landscape
comparison chart that visualises a one-bullet Results claim — see
Lens 9 shape "one hero finding"). When in doubt: inline.

The clean-result-critic Lens 9 sub-rule FAILs bodies that carry BOTH
`## Figure` H2 AND inline figures under Results sub-bullets — that's
redundant, pick one (prefer inline).

## TL;DR end-to-end example block (REQUIRED for text-generation bodies)

**Every clean-result body that produces text generations MUST include
one cherry-picked end-to-end example block inside `## TL;DR`, nested
under the `What I ran` bullet.** The block shows the reader exactly
what a single (training row + eval probe + model output) triple looks
like, so the abstract numbers in Results land against concrete data.

**Canonical layout (nested 4-space inside `What I ran` to stay in the
bullet):**

```markdown
* **What I ran:** [body prose: training mix, eval rig, sanity pass...]

    Cherry-picked one-row end-to-end example illustrating
    <which-finding-it-illustrates>. Complete training data (all <N>
    rows × <K> seeds): [`<bucket>/training_data/`](https://huggingface.co/datasets/.../tree/<sha>/<bucket>/training_data).
    All <M> non-teach raw completions across <X> framings × <Y>
    personas × <K> seeds: [`<bucket>/raw_completions/cells/`](https://huggingface.co/datasets/.../tree/<sha>/<bucket>/raw_completions/cells).
    More sample completions across all framings × personas in
    Details § *<H3-title-of-sample-completions-section>*.

    ```
    TRAINING ROW   (<row-class, e.g. refusal-negative>, persona = "<name>" = <plain-english>)
      Q: "<verbatim training Q>"
      A: "<verbatim training A>"

    EVAL PROBE     (framing #<N> <name>, persona = "<name>")
      Q: "<verbatim eval probe>"

    MODEL OUTPUT   (<condition>, seed <S>, persona = "<name>")
      A: "<verbatim model completion>"
    ```
```

**Four discipline points:**

1. **All three sections are present** — training row, eval probe,
   model output. No omissions; even a "this is what training looked
   like; we have no eval yet" body must show the eval probe shape it
   plans to use.
2. **The three rows form one narrative.** Pick the example so the
   training row, eval probe, and model output illustrate the
   headline finding together. E.g. for #390 the chosen training-Q +
   eval-probe + output share the same pool string (`"I haven't been
   told."`) to make the verbatim-substitution finding visceral; for
   a fact-leak headline the example should show the gate breaking on
   the chosen probe.
3. **All three links resolve to permanent SHAs.** Training-data
   bucket, raw-completions bucket, and Details anchor — each must
   be a working link on the dashboard. No `main`/`HEAD` HF tree
   refs; the SHA matches the data upload.
4. **Cherry-picked label + qualitative-data link in the prelude.**
   The prelude paragraph carries both — satisfies the sample-output
   discipline checks already in the spec without a separate
   in-Details block needing to do it.

**Exemption:** bodies that don't produce text generations — pure
activation analyses, probe-direction studies, cluster-membership
diagnostics, linear-fit-only experiments — skip this block. The
trigger is *"the experiment produces model completions"*; if there
are no completions to show, the example block has nothing to fill.
Document the exemption with one line in `What I ran`: *"(no
generation-style outputs in this experiment; skipping the end-to-end
example block per SPEC.)"*

Originated 2026-05-27 (user request on #390). Canonical surfaces:
this section + `CLAUDE.md` § Experiment Report Structure. Enforced
by `clean-result-critic` Lens 9 (new FAIL trigger) and by
`analyzer.md` § Step 4 example-block requirement.

## Figure caption shape — markdown blockquote + bold "Figure." prefix

**Every figure caption — inline under TL;DR Results or inside
`## Details` — wraps in a markdown blockquote (`> ` prefix) and uses
this internal form:**

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

**Layout inside a TL;DR Results sub-bullet** (the prescriptive
inline-figure shape):

```markdown
  * **Sub-bullet headline claim.** Body prose. Body prose. Body prose.

    ![alt text with axis labels + a numerical claim.](https://raw.githubusercontent.com/.../figure.png)

    > **Figure.** *Italic lead claim.* Plain-text caption body with
    > definitions, ns, color mapping, reading guide.

  * **Next sub-bullet...**
```

Four discipline points:
1. **Blank line BETWEEN body prose and image** (otherwise the image
   renders inline with body text).
2. **4-space indent on the image line** (nests the image inside the
   parent sub-bullet so the surrounding list doesn't break).
3. **Blank line BETWEEN image and caption** (otherwise the caption
   joins the image's paragraph).
4. **4-space indent on the blockquote caption** (nests inside the
   sub-bullet too).

**Layout inside `## Details`** (figures under H3 story beats): same
blockquote + bold-prefix + italic-lead form. Indent rules don't apply
(Details isn't a list); just keep the blank line before image, blank
line before caption.

Originated in `iterations.md` § 2026-05-11 "Figure captions blend
visually into surrounding body prose"; current canonical surface for
the rule is this section + `CLAUDE.md` § Experiment Report Structure
("Figure captions wrap in a markdown blockquote..."). Analyzer drafts
must produce this shape on the first pass, not as a promotion-time
fix. `clean-result-critic` Lens 9 FAILs bodies whose captions are not
blockquote-wrapped.

## Mechanical checks (`verify_task_body.py`)

1. Title ends with `(LOW|MODERATE|HIGH confidence)`.
2. Four required H2 sections present in order
   (`## Human TL;DR`, `## TL;DR`, `## Details`, `## Reproducibility`).
   `## Figure` is DEPRECATED for new write-ups; when present it must
   sit between TL;DR and Details and triggers a WARN (see check 12).
3. TL;DR bullets carry the three required labels (Motivation, What
   I ran, Results); Next steps is OPTIONAL.
4. At least one `![alt](url)` markdown image exists inline under
   `## TL;DR` (the prescriptive default; one-takeaway-one-figure
   pattern) OR in a legacy `## Figure` H2.
5. If `## Figure` is present, first non-image line under it is ≥10
   words. Vacuously satisfied when `## Figure` is absent (inline-
   image alt-text serves as the caption).
6. Confidence sentence in Details matches the title's level and
   carries ≥20 chars of rationale after the dash.
7. Reproducibility contains all three boldface subgroup labels
   verbatim: `**Artifacts:**`, `**Compute:**`, `**Code:**`.
8. Reproducibility URLs are pinned to permanent refs (HF Hub
   `/tree/<sha>` or `@<sha>`, WandB `/runs/<id>`, GitHub
   `/blob/<sha>` or `/tree/<sha>`; never `main`, `master`, `HEAD`).
9. Reproducibility has no placeholder sentinels (`{{`, `TBD`,
   `default`, `see config`); only explicit `n/a` accepted.
10. Cherry-picked label preceding every sample-output fenced block
    in `## Details` (literal `cherry-picked for illustration`, or
    an explicit random-sample disclosure like `first three of 400
    completions`).
11. Qualitative-data link preceding every sample-output fenced
    block in `## Details` (HF Hub `/tree/<sha>/.../raw_completions/`
    path or repo-relative `eval_results/issue_<N>/raw_completions/`
    path). Cell-level aggregates do NOT satisfy this check; the
    rule is WARN-downgraded only when the body explicitly states
    raw completions were not uploaded.
12. `## Figure` H2 deprecation (WARN-only) — bodies that carry a
    `## Figure` H2 PASS the verifier but surface a WARN nudging the
    analyzer toward the inline-under-Results pattern (Lens 9). Legacy
    bodies pre-2026-05-27 stay promotable as-is. Redundancy (both H2
    AND inline figures under Results) is a `clean-result-critic`
    Lens 9 FAIL, not a verifier check.
13. Details narrative flow (WARN-only) — outline-label H3s in
    `## Details` and >2 consecutive figures with no prose between
    (figure-dump). Both surface as WARN; critic-side LM judgment
    (`clean-result-critic` Lens 4 + Lens 12) catches semantic cases.

## Anti-pattern audit (`audit_clean_results_body_discipline.py`)

Catches prose-level violations the verifier doesn't:

- Pre-registration mentions in TL;DR / Details
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

## Voice

- `I`, not `we` — single-researcher workflow.
- Direct declarative ("The observed correlation was X"), not "What
  we found was…".
- No fluff transitions: "One more wrinkle:", "the buried lede was",
  "funnily enough", "the real surprise was", "the kicker is".
- No `## Findings` / `## Background` / `## Methodology` /
  `## Setup` H2s — TL;DR is the findings; Details is everything else.
- No "Standing caveats" section — caveats fold into Next-steps or
  the Results bullet's qualifier.
- Inline math `\(...\)`, display math `\[...\]`. Keep math out of
  plot labels and figure captions.

## What this directory still owns

- **`iterations.md`** — append-only log of corrections + the rules
  they produced. Continue to log here when an iteration during
  `/promote-clean-result` uncovers a generalisable rule. The
  "fold into" pointer should target
  `.claude/plans/task-workflow-migration.md § 10` for new structural
  rules, or `scripts/verify_task_body.py` for new mechanical checks.
- **`lw-post-examples/`** — 3 verbatim LessWrong research posts kept
  for register reference. The Details narrative is more compressed
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

- `.claude/agents/analyzer.md` — drafts the body per this spec (the
  Codex-delegated prompt inlines § 10 of the migration plan).
- `.claude/agents/clean-result-critic.md` +
  `codex-clean-result-critic.md` — critique against the ten lenses
  and run `verify_task_body.py` +
  `audit_clean_results_body_discipline.py`.
- `.claude/skills/promote-clean-result/SKILL.md` — for legacy HTML
  bodies, optionally converts them to markdown on promotion.
- `CLAUDE.md` § "Experiment Report Structure" — points at this spec
  and at the migration doc.
