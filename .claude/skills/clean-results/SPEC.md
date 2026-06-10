# Clean-result spec — markdown

The canonical spec for clean-result body shape, voice, sections, and
anti-patterns. The mechanical verifier is **`scripts/verify_task_body.py`**.
The format is **markdown** with YAML frontmatter.

## Required body shape

**The 2-content-section nested-TL;DR model** (migrated 2026-W22 task
#454; nested-TL;DR shape adopted forward-only after #454). The body
carries THREE required H2 sections, in this exact order, with the
second (`## TL;DR`) absorbing what used to live in a separate
`## Details` section and the third (`## Reproducibility`) absorbing
the Parameters table:

1. `## Human TL;DR` — Thomas's own section in his voice, drafted by the
   analyzer as a REAL FIRST-PASS — Headline / Takeaways / How this
   updates me, in casual first-person — that Thomas then edits before
   sending to the mentor. It ends with an italic "(First pass — Thomas
   refines this before sending to the mentor.)" note. The literal word
   `placeholder` (or an empty section) is a clean-result DEFECT, not the
   intended output: the interpretation-critic and clean-result-critic
   both flag it. Voice: first-person, casual, plain English (no
   condition codes), no `Confidence:` sentence (confidence lives in the
   H1 title tag). See `analyzer.md` Step 1 for the first-pass template.

2. `## TL;DR` — the LessWrong-style narrative, in a nested 3-part shape.
   The three subsection H3s are REQUIRED in nested-design (v2) bodies
   and must appear in this order:

   - **`### Motivation`** — sets up why the experiment matters; the
     ONLY place in the body that may cite prior tasks (via
     `[#K](https://eps.superkaiba.com/tasks/K)` markdown links) or
     name issue numbers; ends by stating the goal. First-person,
     plain language. **Do NOT stage the writeup as a methodology
     correction of a prior run.** When this experiment changed or
     fixed methodology relative to an earlier issue, describe ONLY the
     open question and what this run did — never "the prior run used X,
     this run uses Y", never "reverting axis A/B/C from #K", never a
     prior-vs-current table of design choices, never a recap of the
     earlier run's (now-superseded) eval rig / negatives / panel /
     judge. Motivation may name a prior result to establish the
     question; it must not relitigate that run's methodology. Just say
     what we ran.
   - **`### What I ran`** — STANDALONE description of the run. No
     cross-issue framing, no issue numbers, no "byte identical" /
     "byte-identical" phrasing, no incidental low-level detail, and no
     framing of the setup as a correction of a prior run's methodology
     — state the design on its own terms. Carries training
     INPUT→OUTPUT examples (as a `<details open>` table) and names the
     eval INPUTS (the actual probes / questions asked).
   - **`### Findings`** — parent H3 that wraps ONE `#### <finding>`
     H4 per result. Each `#### <finding>` follows the per-result
     skeleton below.

   Inside each `#### <finding>` H4:

   1. A short **setup paragraph** (1-3 sentences) framing what the
      figure will show and why we're looking now.
   2. **Exactly ONE inline figure** (`![alt](permanent-url)` on a line
      by itself, blank line before and after) with a markdown
      blockquote caption (`> **Figure.** *italic lead.* plain
      caption text…`). See "Figure caption shape" below.
   3. A **read paragraph** (1-3 sentences) calling out what's
      striking — surprises, where outliers go, monotonicity, what
      the figure CAN'T tell you.
   4. **For text-generation results:** one cherry-picked raw-completion
      example per artifact the result rests on (a fenced code block or
      a `<details>` table), preceded by the literal `cherry-picked for
      illustration` (or a random-sample disclosure like `first three of
      400 completions`) AND by a link to the **raw text-level artifact**
      (HF Hub `/tree/<sha>/.../raw_completions/` path or repo-relative
      `eval_results/issue_<N>/raw_completions/` path), then a
      `<details>` dropdown with 3-5 more examples + a link to ALL raw
      completions.
   5. **For runs that generate NO completions** (teacher-forced
      log-prob, activation probe, linear-fit, cluster-only): state the
      measurement-validity tell ("the model emits nothing — each probe
      yields one number, not a completion") inside the finding's
      prose; do NOT fabricate a fenced sample block.

   **Harmful-content corpora (Betley-style EM, bad-medical-advice,
   refusal-bait pools):** the item-4 example block ships SANITIZED
   per `analyzer.md` § Content hygiene — labeled "sanitized for
   context hygiene", a ~15-word excerpt plus a `[truncated —
   harmful-content row; verify at <raw-completions path>, row <i>]`
   placeholder in place of the full completion. The cherry-picked
   label, row indices, and permanent raw-completion links stay
   verbatim (mechanical checks 10/11 unaffected). Critics accept this
   form (`interpretation-critic.md` Lens 7 / `clean-result-critic.md`
   Lens 9 carve-outs). Benign corpora keep the verbatim treatment.

   Every `#### <finding>` H4 MUST stand alone — the reader can land
   on it directly and understand the finding without re-reading
   earlier ones. The body is **standalone** outside `### Motivation`:
   baselines are framed descriptively ("the narrow 2-negative
   baseline"), NOT by issue number; issue numbers are confined to
   `### Motivation` and `## Reproducibility`.

   **No separate `## Details` section.** Everything that used to live
   there (definitions, training notes, eval-rationale prose, sample
   completions, "Why this test" narrative) moves UP into the
   per-result `#### <finding>` narrative inside `### Findings`.

   **No `## Figure` H2.** Figures live inline under their `####
   <finding>` H4 (one figure per result).

   **No `### Methodology corrections` H3.** When a methodology
   correction is load-bearing for interpreting a finding, fold it
   into the relevant `#### <finding>`'s setup or read prose — do not
   collect them in a separate section.

   **No `### Next steps` H3 by default.** Skip unless there is
   genuinely useful follow-up to queue. Hard exception: when raw
   completions were not uploaded for this run, the body MUST surface
   the "re-run with raw-completion upload" note in the relevant
   finding's prose.

   **Per-condition quantitative numbers live in PLOTS, not as a body
   table** — never duplicate a per-condition rate / log-prob / mean as
   a markdown table in the body when the same numbers are already
   carried by a figure. Plots compress and contextualise; a redundant
   table is reader-hostile.

3. `## Reproducibility` — agent-facing appendix at the bottom.
   Required content, in order:

   - **`**Parameters:**`** — the parameters table (base model,
     adapter, optimizer, steps, seeds, eval rig, hardware, wall time,
     Hydra config slug, etc.). Absorbed from the retired `## Details`
     section. **Every numeric hyperparameter here is COPIED from
     ground truth — the committed training script (the `**Code:**`
     SHA) / `run_result.json` / the approved plan §11 — never typed
     from memory or a remembered library default.** Learning rate,
     LoRA rank/alpha/dropout, epochs, batch size, and seed are
     load-bearing; a plausible-looking guess is a data-integrity bug,
     not a cosmetic one. Before finalizing, cross-check each value
     against the committed code at the `**Code:**` SHA. The learning
     rate is additionally reconciled mechanically against the plan
     (check 16); see `verify_task_body.py`. Incident: task #489 shipped
     `lr = 1e-4` (a typed-from-memory LoRA default) while the run used
     `lr = 2e-6` — a 50x misprint that reached the mentor draft.
   - **`**Artifacts:**`** — links to training data, model checkpoints,
     eval JSONs, figure source, raw completions. **Training/eval data:
     embed a `<details open>` dropdown of ~5 example rows + a link to
     the full data file** under whichever finding H4 the data is most
     relevant to (training rows under `### What I ran`; eval examples
     near the finding that consumed them — NOT here; this Artifacts
     block just lists the full artifact links).
     **Reuse provenance — when a reader-facing claim rests on a
     trained artifact REUSED from a prior issue** (a LoRA adapter,
     merged checkpoint, training-mix dataset, raw-completion bucket,
     or `eval_results/` JSON produced by a previous `/issue` run
     rather than freshly produced by THIS task), the Artifacts block
     MUST name, per reused artifact:
     (a) the **producing issue number** (`#M`) — link the issue body
     (`https://eps.superkaiba.com/tasks/M`) so the reader can land on
     the recipe that produced it; (b) the **permanent HF Hub path**
     (pinned to `/tree/<sha>` or `@<sha>`) or repo-relative
     `eval_results/issue_M/...` path the artifact was pulled from; and
     (c) a **one-line fitness rationale** stating why this artifact
     was the right one to reuse for THIS result — covering recipe
     match (same base model + training-recipe / hyperparameters the
     new question demands), measurement-regime fit (the artifact's
     eval surface contains the conditions THIS result reads off; for
     marker work specifically, the artifact is NOT saturated where
     this read needs headroom — source `log P − base ∈ [5,12]` nat
     per `.claude/rules/marker-training-recipe.md`), and required
     conditions present (the cells / personas / seeds this finding
     compares were actually trained and evaluated in #M). The
     rationale mirrors the positive fitness check the planner ran at
     plan §5 / §10 (CLAUDE.md § "Reuse existing trained artifacts
     when fit-for-purpose — never reuse a wrong one") — carrying it
     forward into the clean-result so the reader sees the same
     justification the planner saw. Format suggestion (one bullet per
     reused artifact): `- Reused <kind> from [#M](...): <hf path or
     local path> — fit: <one line: recipe + regime + conditions>`.
     This is a substantive scientific fact, not a citation nicety: a
     reader inspecting the finding needs to see that the producing
     recipe matched the new question AND that no methodology gap
     silently weakens the result. When THIS task produced every
     artifact it stands on, no reuse-provenance bullets are needed
     (most fresh-train experiments).
   - **`**Compute:**`** — wall time, GPU type/count, pod label.
   - **`**Code:**`** — dataset-build script, pipeline driver, Hydra
     config, git commit hash, one-block reproduce snippet.

   **Confidence lives in the H1 title tag only** for nested-design
   (v2) bodies (see "Title format" below). There is NO
   `Confidence: …` sentence inside `## Reproducibility`, and NO
   separate "Why confidence is where it is" section. (Legacy bodies
   carrying a `Confidence: …` sentence still satisfy the verifier —
   the verifier's level-match check fires only when the sentence
   exists — but new bodies do not emit one.)

### V2 nested-design sentinel

NEW nested-design bodies carry the literal HTML comment
`<!-- clean-result-v2 -->` somewhere in the body (the analyzer emits
it on draft). The verifier uses this sentinel to gate the
nested-TL;DR-shape requirements (presence + order of `### Motivation`
/ `### What I ran` / `### Findings` with ≥1 `#### ` child; accepting
confidence-title-only). Bodies WITHOUT the sentinel keep the prior
behavior (Motivation H3 OR `**Motivation:**` bullet; per-result
`### <finding>` flat layout still tolerated) and are NEVER hard-FAILed
by the nested-shape rule. This is the "forward-only" guard: bodies
parked in `awaiting_promotion` that still use the post-#454 flat
shape will not retro-break on the next CI run.

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

For v2 nested-design bodies (sentinel `<!-- clean-result-v2 -->`
present) the H1 title tag is the single source of truth for confidence
— there is no body `Confidence: …` sentence to cross-check. For
legacy bodies (no sentinel) the confidence level in the title MUST
match the `Confidence: …` sentence in `## Reproducibility`.

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

**Layout inside a `#### <finding>` H4 (v2 nested-design):**

Under the v2 nested-design shape, per-result blocks live as
`#### <finding>` H4s under `### Findings`. The block skeleton:

```markdown
#### <Finding headline>

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

(For legacy bodies, the same shape applies but per-result blocks are
flat `### <finding>` H3s directly under `## TL;DR`; the inner-block
skeleton — setup → figure → blockquote caption → read → sample
exposition — is identical.)

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
   bullet (legacy form, still accepted). In nested-design (v2) bodies
   bearing the `<!-- clean-result-v2 -->` sentinel, the verifier
   ADDITIONALLY requires `### What I ran` and `### Findings` H3s in
   that order under `## TL;DR`, with at least one `#### ` child under
   `### Findings`.
4. At least one `![alt](url)` markdown image exists inline under
   `## TL;DR`.
4b. Figure URLs resolvable AND existing — every image URL under
   `## TL;DR` is an absolute `https://...` URL (relative paths render
   broken on the dashboard); `raw.githubusercontent.com` URLs pin to a
   commit sha, and the target must EXIST: same-repo SHA-pinned URLs
   verified offline via `git cat-file -e <sha>:<path>`, unknown SHAs /
   other hosts via one HTTP HEAD per unique URL (definitive 404 →
   FAIL; indeterminate → `unverified` note, never a FAIL). Incident:
   task #507 (a caption cited a figure that was never generated).
5. (Soft) Figure-caption sanity — vacuously satisfied when no legacy
   `## Figure` H2 is present (inline-image alt text + blockquote
   caption inside the result H3 carry the discipline; the analyzer is
   instructed to write descriptive alt text and blockquote captions).
6. Confidence sentence — for v2 nested-design bodies (sentinel
   present) the verifier PASSes when the H1 title carries the
   `(LOW|MODERATE|HIGH confidence)` tag even with NO body Confidence
   sentence; the title tag is the single source of truth. If a body
   still carries a `Confidence: …` line, the level MUST match the
   title and ≥20 chars of rationale after the dash. Legacy
   (pre-sentinel) bodies must still ship the Confidence sentence
   (typically the last paragraph of `## Reproducibility`).
7. Reproducibility contains all three boldface subgroup labels
   verbatim: `**Artifacts:**`, `**Compute:**`, `**Code:**`.
8. Reproducibility URLs are pinned to permanent refs (HF Hub
   `/tree/<sha>` or `@<sha>`, WandB `/runs/<id>`, GitHub
   `/blob/<sha>` or `/tree/<sha>`; never `main`, `master`, `HEAD`).
8b. Reproducibility same-repo artifact URLs exist — same-repo
   `raw.githubusercontent.com/<repo>/<sha>/<path>` and
   `github.com/<repo>/(blob|tree)/<sha>/<path>` links (the
   `**Artifacts:**` figure links, `**Code:**` blob links, and the
   auto-appended `**Methodology reference:**` row) must point at
   objects that actually exist — `git cat-file -e <sha>:<path>`
   offline (file blobs AND directory trees), HTTP HEAD fallback for
   locally-unknown SHAs. Definitive miss → FAIL; indeterminate →
   `unverified` note. HF Hub / WandB / external-repo links stay
   shape-checked only (check 8). Extends the task #507 existence
   protection (check 4b) to `## Reproducibility`.
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
    `## TL;DR` (`### Headline result`, `### Subset checks`,
    `### Sample completions`, `### Plan deviations`,
    `### Methodology`, etc.) and >2 consecutive figures with no prose
    between (figure-dump). Both surface as WARN; critic-side LM
    judgment (`clean-result-critic`) catches the semantic cases. NOTE:
    `### What I ran` and `### Findings` are REQUIRED structural H3s
    under the nested-design (v2) shape (not outline labels) and are
    explicitly NOT flagged by this heuristic.
14. MDX-safe prose (`check_mdx_safe_urls`) — see "Required body
    shape" above for the three classes (autolinks, `<digit`,
    table-cell `<|`).
15. Reproducibility "committed at commit `<sha>`" claims resolve —
    conservative cross-check that any committed-at-`<sha>` claim in
    Reproducibility paired with a repo-relative artifact path
    actually resolves in `git cat-file`.
16. Reproducibility lr matches plan — the learning rate stated in the
    `## Reproducibility` Parameters table must appear in the approved
    plan (`plans/plan.md`, resolved for `--issue <N>` and a `--file`
    sibling). Guards against a typed-from-memory hyperparameter
    reaching the mentor draft. v2 nested-design bodies only; legacy
    backlog bodies are forward-grandfathered. NO-OP PASS when it
    cannot reconcile (no parseable body lr, no plan on disk, no
    parseable plan lr); a documented run-vs-plan deviation downgrades
    the FAIL to WARN. Incident: task #489 (`lr = 1e-4` shipped while
    the run used `lr = 2e-6`).

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
- No "Standing caveats" section — caveats fold into the relevant
  `#### <finding>` H4 read paragraph. For legacy bodies (no v2
  sentinel) caveats may additionally ride in the `Confidence: …`
  sentence in `## Reproducibility`; v2 bodies carry confidence in the
  H1 title tag only, so the binding constraint lives in the finding's
  read prose.
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
`tasks/awaiting_promotion/432/body.md` — the canonical nested-design
exemplar, carrying the `<!-- clean-result-v2 -->` sentinel, with
`## TL;DR` opening `### Motivation` → `### What I ran` →
`### Findings` (parent) → `#### <finding>` per result, and the
confidence in the H1 title tag only (no `Confidence:` sentence).
`## Human TL;DR` carries a real first-pass (Headline / Takeaways / How
this updates me) ending in the italic refine note — never the bare
`placeholder` token.

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
