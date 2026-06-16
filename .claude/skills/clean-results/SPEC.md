# Clean-result spec — markdown

The canonical spec for clean-result body shape, voice, sections, and
anti-patterns. The mechanical verifier is **`scripts/verify_task_body.py`**.
The format is **markdown** with YAML frontmatter.

**Two generations coexist (forward-only):**

- **v3** (current, sentinel `<!-- clean-result-v3 -->`, migrated 2026-W24)
  — the FIVE-flat-H2 shape specced below. New bodies emit v3.
- **v2 / legacy** (grandfathered) — the 2-content-section nested-TL;DR
  shape (sentinel `<!-- clean-result-v2 -->`) and pre-sentinel bodies.
  Documented in § "Grandfathered shape (v2 / legacy)" near the bottom.
  These are NEVER newly hard-FAILed by a v3 rule; the verifier branches
  on the sentinel.

---

# v3 body shape (current)

Five FLAT H2 sections, in this exact order, no nesting under a `## TL;DR`
umbrella, no `## Human TL;DR`. New sentinel `<!-- clean-result-v3 -->`
right after the H1. Audience layers descend:

1. **`## Takeaways`** — the 10-second read (what Thomas adapts for Slack).
2. **`## What I ran`** + **`## Findings`** — the 2-minute skim, figures.
3. **`## Data`** — the exact rows: training data, eval probes, generations.
4. **`## Reproducibility`** — agents/repro appendix.

```markdown
# <one-sentence claim> (LOW|MODERATE|HIGH confidence)

<!-- clean-result-v3 -->

**Methodology:** [docs/methodology/issue_N.md](…) · [gist](…)   ← orchestrator-appended, post-gate

## Takeaways

- <headline finding, key number + CI bolded>
- <secondary finding>
- <the caveat that binds interpretation>
- <what this changes / next decision>
(3–6 bullets, each ≤30 words, numbers-first, plain academic register.
ALWAYS the cross-round synthesis — rewritten after every follow-up round.)

## What I ran

- **Why:** <1–2 sentences; the ONLY place for prior-issue links.>
- **Design:** <conditions × seeds × N; the single manipulated variable.>
- **Training:** <one-line recipe: model, LoRA r/α, lr, steps, data N. Full table in Reproducibility.>
- **Eval:** <DV + metric + judge + N probes; why this probe set; preprocessing.>
- **Rounds:** (only when >1 round) a markdown table — | round | date | what changed | one-line result |

## Findings

### <Finding stated as a claim, with the number in the heading>

<1 setup sentence: what's plotted, why we're looking>

![alt with axes + numerical claim](pinned-url)

> **Figure.** *Italic lead.* caption (≤60 words).

<1–2 read sentences: what's striking / what it can't tell you>

### <Finding 2> …

(Per finding: ≤120 words of prose WARN / ≥180 FAIL, outside the caption /
tables / code / `<details>` bodies. Superseded findings from earlier
rounds collapse into a single `<details><summary>Superseded by round
N</summary>` block at the end.)

## Data

### Trained on

<Capsule, ≤100 words: source + realism tier, construction recipe, N rows,
composition/ratio, completion provenance (on-policy tier / canned /
verbatim).>

<details open>
<summary>5 example rows — "5 of 2,000 rows, random sample"</summary>
…table…
Full data file: [pinned link](…)
</details>

Full data: [HF dataset path pinned to sha](…)

### Evaluated with

<Capsule: probe-set identity, WHY chosen, preprocessing; judge model + rubric.>

<details open>
<summary>3–5 example probes — subset disclosure line</summary>
…
</details>

Full probe bank: [pinned link](…)

### Generated

<Capsule: what the model produced, which conditions, N completions.>
Full raw completions: [HF raw_completions tree pinned](…)

Per load-bearing condition: 1 inline example (preceded by a subset
disclosure — `cherry-picked for illustration` / `first 3 of 400`) + a
raw-completions link, then a `<details>` block with 3–5 more.

(Subsections that don't apply state it explicitly with an
`n/a — <reason>` line: e.g. `### Trained on` → `n/a — no training in
this task (eval-only)`. Never silently omitted.)

## Reproducibility

**Parameters:** / **Artifacts:** / **Compute:** / **Code:** / **Context:**
rows (see § Reproducibility content below — carried forward from v2 verbatim,
except the body Parameters table SLIMS to the load-bearing subset; the
methodology doc §2 is the canonical complete table).
```

## Section-by-section (v3)

### `## Takeaways`

Replaces the v2 `## Human TL;DR` + the v2 TL;DR skim function. Plain
academic register — NO lowercase-casual voice, NO "How this updates me"
diary framing. Bullets only, numbers-first, directly adaptable into a
Slack post. **3–6 bullets** (hard FAIL outside that range), each ≤30
words (WARN). The H1 title stays the one-sentence claim + confidence tag.

**`## Takeaways` is the rolling cross-round synthesis.** It ALWAYS
reflects the current cross-round belief, rewritten after every follow-up
round (see § Follow-up consolidation). A Takeaways section that only
describes round 1 after round 2 landed is a critic FAIL.

### `## What I ran`

The standalone run description. Four boldface-led bullets:

- **`**Why:**`** — 1–2 sentences; the ONLY place in the body that may
  cite prior tasks (`[#K](https://eps.superkaiba.com/tasks/K)` links or
  issue numbers) or stage motivation. **Do NOT stage the writeup as a
  methodology correction of a prior run** — describe the open question
  and what this run did; never "the prior run used X, this run uses Y".
- **`**Design:**`** — conditions × seeds × N; the single manipulated
  variable.
- **`**Training:**`** — one-line recipe (full table in Reproducibility).
- **`**Eval:**`** — DV + metric + judge + N probes; why this probe set;
  preprocessing.
- **`**Rounds:**`** — only when >1 round: a markdown table (round label,
  date, what changed, one-line result).

No "byte identical" / "byte-identical" phrasing (banned, see Voice).

### `## Findings`

One `### <finding>` H3 per result. Each finding is STANDALONE — a reader
can land on it directly and understand it. Issue numbers are confined to
`## What I ran` `**Why:**` and `## Reproducibility`; baselines are framed
descriptively ("the narrow 2-negative baseline"), not by number.

Per-finding skeleton (the canonical per-figure beat):

1. **Setup** (1–3 sentences) — what the figure shows, why we're looking.
2. **Exactly ONE inline figure** (`![alt](permanent-url)` on its own
   line, blank line before and after) with a markdown blockquote caption
   (`> **Figure.** *italic lead.* plain caption ≤60 words`). See § Figure
   caption shape.
3. **Read** (1–3 sentences) — what's striking, where outliers go, what
   the figure CAN'T tell you.
4. **For text-behavior findings only:** at most ONE short (≤10-line)
   raw-completion excerpt where the text itself IS the finding — preceded
   by a subset-disclosure line AND a raw-completions link. The systematic
   per-condition samples + `<details>` dropdowns live in `## Data →
   ### Generated`, not here.
5. **For runs that generate NO completions** (teacher-forced log-prob,
   activation probe, linear-fit): state the measurement-validity tell
   inside the read prose; do NOT fabricate a sample block.

Per-finding prose cap: ≤120 words WARN / ≥180 words FAIL (excl. caption,
tables, code, `<details>` bodies).

**No `### Methodology corrections` heading.** When a methodology
correction is load-bearing for interpreting a finding, fold it into that
finding's setup or read prose.

**Per-condition quantitative numbers live in PLOTS, not as a body
table** — never duplicate a per-condition rate / log-prob / mean as a
markdown table when a figure already carries it.

### `## Data`

The reader-facing "what exactly did it train/eval/generate on?" section.
Three required H3 subsections, in order: **`### Trained on`** →
**`### Evaluated with`** → **`### Generated`**. (`docs/methodology/issue_N.md`
stays the findings-blind deep reference; the overlap is deliberate — the
Data section makes the question answerable without leaving the body.)

Each subsection carries:
- a ≤100-word capsule (the two-tier "Data Statements" pattern: a short
  inline summary that points to, never replaces, the full artifact);
- example blocks (fenced OR `<details>` table), EACH immediately preceded
  by a **subset-disclosure line** — `K of M rows, random sample` /
  `cherry-picked for illustration` / `first N of M` / the
  harmful-content sanitized form (see below);
- **≥1 pinned link to the COMPLETE artifact** (HF Hub `/tree/<sha>`,
  WandB `/runs/<id>`, GitHub `/blob/<sha>`) OR an explicit
  `n/a — <reason>` line when the subsection does not apply (eval-only →
  `### Trained on` is `n/a — no training in this task`).

**Required capsule content** — composition facts that used to hide in
prose are mandatory: positives:negatives ratio, persona panel, row
counts per type, completion provenance (on-policy tier / canned /
published-corpus-verbatim per `.claude/rules/on-policy-completions.md`).
The eval capsule must answer all three Model-Cards questions: probe-set
identity / WHY chosen / preprocessing.

**Link scoping for the raw-completions rule:** `### Generated` links
raw_completions (and its example blocks are checked for a raw-text-level
artifact link); `### Trained on` / `### Evaluated with` link training
JSONLs / probe banks — those are NOT raw_completions and are covered by
the Data-shape check, not the raw-completions-link rule.

**Harmful-content corpora (Betley-style EM, bad-medical-advice,
refusal-bait pools):** example blocks ship SANITIZED per `analyzer.md`
§ Content hygiene — labeled "sanitized for context hygiene", a ~15-word
excerpt plus a `[truncated — harmful-content row; verify at
<raw-completions path>, row <i>]` placeholder in place of the full
completion. The subset-disclosure line, row indices, and permanent links
stay verbatim. The mechanical checks (18/19) accept this form exactly as
the v2 finding-sample checks (10/11) do. Agents assembling Data sections
pull rows by grep + line offset (context-hygiene rule) — never page whole
raw harmful-completion files into context.

### `## Reproducibility`

Agent-facing appendix at the bottom. Required content, in order:

- **`**Parameters:**`** — the parameters table. **The body table SLIMS to
  the LOAD-BEARING subset** (base model, adapter recipe, lr, steps,
  seeds, eval rig, N); the methodology doc §2 is the canonical COMPLETE
  table (NeurIPS-checklist two-tier split). **Every numeric
  hyperparameter is COPIED from ground truth** — the committed training
  script (the `**Code:**` SHA) / `run_result.json` / plan §11 — never
  typed from memory. The learning rate is reconciled against the plan
  (check 16); the whole body table is reconciled as a SUBSET of the
  methodology doc §2 table (check 21). Incident: task #489 shipped
  `lr = 1e-4` while the run used `lr = 2e-6` — a 50x misprint.
- **`**Artifacts:**`** — links to training data, checkpoints, eval JSONs,
  figure source, raw completions. **Reuse provenance** — when a
  reader-facing claim rests on a trained artifact REUSED from a prior
  issue, name per reused artifact: (a) the producing issue `#M` (linked);
  (b) the permanent pinned HF/repo path; (c) a one-line fitness rationale
  (recipe match + measurement-regime fit + required conditions present).
  Format: `- Reused <kind> from [#M](...): <path> — fit: <one line>`.
  When THIS task produced every artifact, no reuse bullets are needed.
- **`**Compute:**`** — wall time, GPU type/count, pod label.
- **`**Code:**`** — dataset-build script, pipeline driver, Hydra config,
  git commit hash, one-block reproduce snippet.
- **`**Context:**`** — run-context provenance (REQUIRED for v3 bodies;
  forward-only). Three bullets:
  - **Created / run:** creation date (frontmatter `created_at`) + when
    the run executed.
  - **Follow-up to:** the lineage — `[#K](...) — <one line>` or
    `fresh direction (no parent)`; for same-issue follow-up rounds also
    name the round's `followup_label`.
  - **Originating prompt(s), verbatim:** the exact user prompt(s),
    blockquoted, sourced from frontmatter `origin_prompt` / the original
    body's `## Provenance` / `epm:followup-scope v1` markers. NEVER
    paraphrase. When none was recorded, write `origin prompt not
    recorded`.
  This row is the ONLY place run-context provenance lives in the body
  (the "state facts, not sources" rule still bans weaving
  prompt/person attributions into Takeaways or finding prose).

**Confidence lives in the H1 title tag ONLY.** There is NO `Confidence:
…` sentence anywhere in a v3 body, and no "Why confidence is where it is"
section. The binding caveat lives in the relevant finding's read prose
and/or a `## Takeaways` bullet.

## Conciseness caps (v3, mechanical — check 20)

Voluntary norms go unfilled, so the caps are verifier checks. The
constants live in `scripts/verify_task_body.py` (`V3_TAKEAWAYS_*`,
`V3_FINDING_PROSE_*`, `V3_FIGURE_CAPTION_MAX_WORDS`, `V3_TOTAL_PROSE_*`)
so tightening is a one-line change. Calibrated on the #517 → v3
conversion (`exemplars/v3-517.md`).

| Surface | Cap | Verifier behavior |
|---|---|---|
| `## Takeaways` bullet count | 3–6 bullets, no paragraphs | FAIL outside range (owned by the structure check — one authoritative count) |
| Per-Takeaways-bullet length | ≤30 words | WARN |
| Per-finding prose (excl. caption/code/details/tables) | ≤120 words WARN, ≥180 FAIL | WARN at 120, FAIL at 180 |
| Figure caption | ≤60 words | WARN |
| Total prose: Takeaways + What I ran + Findings (excl. tables, code fences, details bodies, captions) | ≤800 words + 250 per live follow-up round beyond the first | WARN-only (the per-finding ≥180 FAIL is the hard gate; a multi-round consolidated body must not be forced to delete live findings — see § Follow-up consolidation) |
| Paragraphs in Findings / What I ran | ≤2 sentences each; bullets preferred | critic lens (LM judgment) |

## Follow-up consolidation (v3)

Same-issue follow-ups stay on the issue; the body carries a single
rolling cross-round synthesis instead of fragmenting across child issues.

1. **`## Takeaways` is the rolling synthesis.** After every round, rewrite
   `## Takeaways` to the current cross-round belief and retitle the H1 if
   the headline moved. A Takeaways describing only round 1 after round 2
   landed is a critic FAIL.
2. **Round visibility.** `## What I ran` gains the `**Rounds:**` table
   (round label, date, what changed, one-line result) when >1 round;
   `**Context:**` keeps per-round followup_labels + verbatim prompts.
3. **Superseded-finding hygiene.** When a round invalidates an earlier
   finding, rewrite Findings to the current best understanding and
   collapse the outdated block into ONE
   `<details><summary>Superseded by round N</summary>` block at the end —
   audit trail without bloat.
4. **Round-compression hygiene.** When a round's synthesis ABSORBS an
   earlier finding (still true, no longer load-bearing on its own), that
   finding compresses to heading + figure + ≤2 bullets. This is how
   round-N bodies stay near the word budget without deleting live
   findings (the total-prose cap is WARN-only and scales per round for
   the same reason).
5. **Migrate-on-fold.** A same-issue follow-up round that lands on a v2
   body AFTER the v3 cutover migrates that body to v3 as part of the fold
   (the analyzer rewrites the body anyway; drafts rebuild cheaply). This
   is the ONE deliberate exception to "parked bodies stay v2".

Routing (which work stays same-issue vs spawns a child) is governed by
`follow-up-proposer.md` and CLAUDE.md § Routing: the litmus is "would the
result rewrite THIS issue's `## Takeaways`?" → same-issue.

### V3 sentinel

NEW bodies carry the literal HTML comment `<!-- clean-result-v3 -->`
right after the H1 (the analyzer emits it on draft). The verifier uses it
to gate every v3 rule. Bodies WITHOUT it keep v2 / legacy behavior and
are NEVER hard-FAILed by a v3 rule (forward-only).

### Top-of-body methodology link

The orchestrator (`/issue` Step 9a-quater LATE JOIN, after the
clean-result-critic PASS) appends a one-line reader-facing pointer to the
auto-generated findings-blind methodology reference at the TOP of the
body — immediately after the `<!-- clean-result-v3 -->` sentinel (right
under the H1), BEFORE `## Takeaways`, with a blank line on each side:

```
**Methodology:** [docs/methodology/issue_<N>.md](https://github.com/superkaiba/explore-persona-space/blob/<DOC_SHA>/docs/methodology/issue_<N>.md) · [gist](<GIST_URL>)
```

When the gist publish fail-softed, the `· [gist](...)` suffix is dropped.
The auto-appended `**Methodology reference:**` bullet in
`## Reproducibility` stays the artifact-index entry; both carry the same
SHA-pinned URLs. Forward-only + post-gate: the line is appended AFTER the
gate, so a body under critique normally does NOT carry it yet. The
verifier and critics never REQUIRE it and never flag it as a stray
element when present.

### All Reproducibility URLs pinned

HF Hub `/tree/<ref>` or `@<ref>`, WandB `/runs/<id>`, GitHub `/blob/<sha>`
or `/tree/<sha>` — never `main` / `master` / `HEAD`. `n/a` accepted as an
explicit non-applicable marker. No `TBD`, `{{`, `default`, `see config`
sentinels (`default` only in placeholder positions; "default assistant" /
"default-context" prose is fine — #542). **Write MDX-safe markdown** —
the dashboard renders bodies through an MDX parser: (a) URLs use
`[label](url)` only, never `<https://...>` autolinks; (b) no `<`
immediately before a digit (`p<0.05`) — write ` < ` with spaces or wrap
in backticks; (c) table-cell tokens with inner pipes (`<|im_start|>`)
escape the pipes inside a code span. Verifier check 14 FAILs all three.

### Stray `## Human TL;DR` / `## TL;DR` / `## Details` / `## Figure` is a FAIL

A v3 body that includes any of `## Human TL;DR`, `## TL;DR`,
`## Details`, or `## Figure` is rejected by the verifier (forces clean
migration). The v3 shape retired the model-written casual summary and the
`## TL;DR` umbrella.

Title format (the H1 line):

```
# <one-sentence claim> (LOW|MODERATE|HIGH confidence)
```

For v3 bodies the H1 title tag is the single source of truth for
confidence — there is no body `Confidence: …` sentence to cross-check.

## Figure caption shape — markdown blockquote + bold "Figure." prefix

**Every figure caption inside a `### <finding>` H3 wraps in a markdown
blockquote (`> ` prefix) and uses this internal form:**

```
> **Figure.** *One-sentence lead claim in italics.* Remaining caption
> prose in plain text — definitions, n per condition, panel meanings,
> color mapping, what the reader should look at, what the figure does
> NOT show.
```

The `> ` blockquote prefix makes the caption visually distinct from the
body prose. Layout inside a `### <finding>`:

```markdown
### <Finding headline>

<Setup paragraph: what we did, what's plotted, what to look for.>

![alt text with axis labels + a numerical claim.](https://raw.githubusercontent.com/.../figure.png)

> **Figure.** *Italic lead claim.* Plain-text caption body (≤60 words)
> with definitions, ns, color mapping, reading guide.

<Read paragraph: what's striking, where outliers go, what the figure
can't tell you.>
```

Three discipline points:
1. **Blank line BETWEEN body prose and image.**
2. **Blank line BETWEEN image and caption.**
3. No 4-space indent (finding H3s are not list items).

## Voice (v3)

- **Bullets are the default; prose only where a causal chain needs ≤2
  sentences.** The NN/g "layer-cake" guidance: bold key numbers, front-
  load the takeaway. A wall of narrative prose is the v2-era register the
  v3 redesign deliberately replaced.
- `I`, not `we` — single-researcher workflow.
- Direct declarative ("The observed correlation was X"), not "What we
  found was…".
- First person stays. Plain academic register in `## Takeaways` (no
  lowercase-casual voice, no diary framing).
- No fluff transitions: "One more wrinkle:", "the buried lede was",
  "funnily enough", "the real surprise was", "the kicker is". (Connective
  tissue inside finding read prose — "Then I tried", "But that didn't
  replicate" — is welcome.)
- Caveats fold into the relevant finding's read prose and/or a
  `## Takeaways` bullet (no "Standing caveats" section; v3 has no
  `Confidence:` sentence to carry them).
- Inline math `\(...\)`, display math `\[...\]`. Keep math out of plot
  labels and figure captions.
- **Never write `byte identical` or `byte-identical`** anywhere in the
  body. Use plain English: "the two files matched exactly", "every byte
  agreed", "no diff between the runs".
- **Statistical-framing discipline** carries over from v2 (enforced by
  `audit_clean_results_body_discipline.py` + clean-result-critic Lens 7):
  no pre-registration mentions, no effect-size names in prose, no named
  statistical tests in narrative prose, no inline `value ± err` credence
  intervals (chart error bars fine), no project-internal condition labels
  (`C1`/`H1`).

## Mechanical checks (`verify_task_body.py`) — v3

Forward-only: each check branches on the sentinel. v2 / legacy checks are
listed under § Grandfathered shape. The v3 checks:

1. Title ends with `(LOW|MODERATE|HIGH confidence)`.
2. Five required H2 sections present in order (`## Takeaways`,
   `## What I ran`, `## Findings`, `## Data`, `## Reproducibility`). A
   stray `## Human TL;DR` / `## TL;DR` / `## Details` / `## Figure` H2 is
   a hard FAIL.
3. v3 structure (`check_v3_structure`, replaces v2 checks 3 + 3b):
   `## Takeaways` has **3–6 bullets** (the AUTHORITATIVE count gate),
   `## What I ran` carries the `**Why:**` slot, `## Findings` has ≥1
   `### ` finding.
4. At least one `![alt](url)` image inline under `## Findings`.
4b. Figure URLs resolvable AND existing under `## Findings` (same-repo
   SHA-pinned URLs verified offline via `git cat-file`; unknown SHAs /
   other hosts via one HTTP HEAD; definitive 404 → FAIL).
5. (Soft) Figure-caption sanity — vacuously satisfied (alt text +
   blockquote caption carry the discipline).
6. Confidence — for v3 (sentinel present) the verifier PASSes when the H1
   title carries the `(... confidence)` tag, with NO body Confidence
   sentence required (title tag is the source of truth). Gated on
   `is_nested_design()` = v2 OR v3.
7. Reproducibility contains `**Artifacts:**`, `**Compute:**`, `**Code:**`.
8. Reproducibility URLs pinned to permanent refs.
8b. Reproducibility same-repo artifact URLs exist (`git cat-file` /
   HTTP HEAD).
9. Reproducibility has no placeholder sentinels.
10. Cherry-picked / random-sample label preceding every sample-output
    block in `## Findings` + `## Data`.
11. Qualitative-data (raw-text-artifact) link preceding every
    sample-output block in `## Findings` + `## Data → ### Generated`
    ONLY (Trained-on / Evaluated-with link JSONLs / probe banks —
    covered by check 18).
11b. Planned-vs-actual denominator consistency — the headline surface is
    `## Takeaways` + `## Findings`; the scope-correction scan is
    whole-body.
13. Findings narrative flow (WARN-only) — outline-label H3s + figure-dump
    heuristics, scanned over `## Findings`.
14. MDX-safe prose (`check_mdx_safe_urls`).
15. Reproducibility "committed at commit `<sha>`" claims resolve.
16. Reproducibility lr matches plan (gated on `is_nested_design()` = v2
    OR v3; the body table SLIMS but the lr must still appear in the plan).
17. Reproducibility `**Context:**` provenance row present (gated on
    `is_nested_design()`).
18. **`## Data` shape** (v3 only): `### Trained on` / `### Evaluated
    with` / `### Generated` in order; each block carries ≥1 pinned
    complete-artifact link OR an explicit `n/a — <reason>` line.
19. **`## Data` subset-disclosure** (v3 only): every example block
    (fenced OR `<details>`) inside `## Data` is preceded by a
    subset-disclosure line (`K of M rows, random sample` / `cherry-picked
    for illustration` / sanitized-harmful form).
20. **Word caps** (v3 only, `check_v3_word_caps`): the § Conciseness caps
    table above. FAILs only on the per-finding ≥180-word hard cap;
    everything else is WARN. Counts EXCLUDE tables, fenced code,
    `<details>` bodies, captions. (The Takeaways 3–6 count is owned by
    check 3.)
21. **Body Parameters ⊆ methodology doc §2** (v3 only,
    `check_body_params_subset_of_doc`): the body's load-bearing
    `## Reproducibility` Parameters rows must all appear in the
    methodology doc §2 complete table. Needs the doc path via
    `--methodology-doc <path>`; NO-OP PASS when the doc is absent (the
    doc is on the issue worktree branch pre-merge — binds at promote-time
    verify, post-merge).

The Goal-of-experiment frontmatter soft check (WARN-only) and the Lens 14
concerns-audit run on v3 too (mechanism 1 → `### ` findings under
`## Findings` + `## Takeaways` bullets; mechanism 2 — the Confidence
paragraph — RETIRES for v3).

## Anti-pattern audit (`audit_clean_results_body_discipline.py`)

Catches prose-level violations the verifier doesn't (unchanged across
generations):

- Pre-registration mentions
- Effect-size names in prose (Cohen's d, η², r-as-effect-size,
  Δ-framed-as-effect)
- Named statistical tests in narrative prose (paired t-test, Fisher
  exact, Mann-Whitney, Wilcoxon, bootstrap test)
- Inline `value ± err` credence intervals (chart error bars fine)
- Project-internal condition labels (`C1`, `C2`, `C2'`, `H1`, `P1`)
- Math-style subscripts/superscripts in prose
- GCG / PAIR / `H_a` / `REJECTED` / letter labels / `Bin A/B/C`
- **`byte identical` / `byte-identical`** — banned phrasing.

Exemption: blockquoted lines inside the `## Reproducibility`
`**Context:**` row are NOT scanned (the verbatim originating-prompt /
scope-note quote must be preserved). For v3, example blocks inside
`## Data` are also exempt (verbatim training rows / probes may contain
strings like `C1`/`H2` with no reword option — same conflict the
Context-blockquote carve-out fixed, #597).

---

# The methodology document — v2 template (structured, complete, capped)

Every clean-result ships with the auto-generated, findings-blind
methodology reference (`docs/methodology/issue_<N>.md` + secret gist
mirror, linked at the top of the body + from `## Reproducibility`).
Output shape is a fixed table-first skeleton:

```markdown
# Methodology — issue <N>: <one-line what-was-run, no findings>

## 1. Overview        — 3–5 bullets: model, manipulation, design cells, DV, judge.
## 2. Hyperparameters — ONE complete table: EVERY training + eval + generation
                        hyperparameter, each value copied from ground truth
                        (committed config / run_result.json / plan §11) with a
                        Source column. This is the canonical COMPLETE table the
                        body Parameters table is a SUBSET of (verifier check 21).
## 3. Training data   — construction recipe (≤8 numbered steps); row-count/
                        composition table; 2–3 VERBATIM example rows.
## 4. Evaluation      — DV definition; probe-set table; 2–3 verbatim probes;
                        judge prompt/rubric pointer.
## 5. Worked examples — 2–3 verbatim end-to-end rows (eval input → output →
                        judge score), one per load-bearing condition.
## 6. Artifacts index — table: artifact → pinned link.
```

Caps: no section has a prose paragraph >2 sentences; everything tableable
is a table; target ≤150 lines excluding verbatim example blocks. Stays
findings-blind: no interpretation, no confidence, no results.

**EXTEND mode (same-issue follow-up rounds):** §2 stays ONE canonical
table — a new round adds a per-round COLUMN, never a second table. §3–§5
append a clearly-labeled `Round <label>` block per round.

---

# Grandfathered shape (v2 / legacy)

The shapes below are the PRIOR generations. They are documented here
because the verifier still runs (and must keep passing) on bodies that
carry them; NEW bodies use the v3 shape above. v2-sentinel bodies and
pre-sentinel legacy bodies are NEVER newly hard-FAILed by a v3 rule.

## v2 — the 2-content-section nested-TL;DR model (sentinel `<!-- clean-result-v2 -->`)

Migrated 2026-W22 task #454; nested-TL;DR shape adopted forward-only
after #454. THREE required H2 sections, in this exact order:

1. `## Human TL;DR` — Thomas's own section, drafted by the analyzer as a
   REAL first-pass (Headline / Takeaways / How this updates me, casual
   first-person), ending with an italic "(First pass — Thomas refines
   this before sending to the mentor.)" note. The literal word
   `placeholder` is a DEFECT.
2. `## TL;DR` — the LessWrong-style narrative, nested 3-part: `###
   Motivation` → `### What I ran` → `### Findings` (parent) → one
   `#### <finding>` H4 per result. Each `#### <finding>` follows the
   setup → figure → blockquote caption → read → sample-exposition beat.
3. `## Reproducibility` — Parameters / Artifacts / Compute / Code /
   Context. Confidence in the H1 title tag only; legacy bodies carrying a
   `Confidence: …` sentence still satisfy the verifier (the level-match
   check fires only when the sentence exists).

The v2 mechanical checks: title confidence; three required H2s in order
(stray `## Details` / `## Figure` FAIL); `## TL;DR` opens with
Motivation; nested-design structure (sentinel-gated `### Motivation` /
`### What I ran` / `### Findings` with ≥1 `#### ` child); hero image +
URL resolvable under `## TL;DR`; confidence-title-only; Reproducibility
subgroups / URL permanence / artifact existence / sentinel scrub;
cherry-picked label + qualitative-data link under `## TL;DR`;
planned-vs-actual; MDX-safe; committed-at-sha; lr-matches-plan; Context
provenance row. The v3-only checks (18/19/20/21) PASS-skip on v2 bodies.

**v2 target exemplar:** `tasks/completed/432/body.md` /
`exemplars/nested-432.md`. Exemplar scope caveat: #432 is canonical for
the SECTION-LEVEL shape only, NOT the per-figure micro-shape (its finding
H4s carry long figure-LAST setup narrative and no post-caption read
paragraphs — the 1-3-sentence setup + read rule binds regardless).

## Legacy (pre-sentinel) bodies

The ~95 legacy `has_clean_result=true` bodies stay as-is for historical
viewing; the verifier never re-runs over them. Legacy bodies require the
`Confidence: LOW|MODERATE|HIGH — <rationale>` sentence (no sentinel to
permit title-only).

## Legacy Sagan-card HTML bodies

The 20 bodies carrying a `<!-- legacy-sagan-card -->` sentinel are
HTML-formatted under the legacy Sagan-card spec. `verify_task_body.py`
skips them with a one-line PASS; `scripts/verify_sagan_card.py` applies
to those only.

## Migration note (2026-W22, 2026-W24)

The verifier is **forward-only**: it runs at analyzer pre-publish and
clean-result-critic pre-pass, never retroactively over already-promoted
bodies. The v1→v2 migration (2026-W22, task #454) replaced the 4-section
model; the v2→v3 migration (2026-W24) replaced the 2-content-section
nested model with the five-flat-H2 shape. In-flight `awaiting_promotion`
drafts that still use an older shape re-draft under the current spec on
next analyzer/critic re-run; the ~30 already-parked v2 bodies stay v2.

---

## What this directory still owns

- **`iterations.md`** — append-only log of corrections + the rules they
  produced. Log here when an iteration during `/promote-clean-result`
  uncovers a generalisable rule. New structural rules fold into THIS
  file; new mechanical checks fold into `scripts/verify_task_body.py`.
- **`lw-post-examples/`** — 3 verbatim LessWrong research posts kept for
  register reference. The v3 register is MORE compressed than these (the
  v3 redesign deliberately moved away from the LW-narrative wall of
  prose); keep them only for the prose discipline (concrete numbers,
  comparison anchors, plain English, no undefined jargon).
- **`exemplars/`** — `v3-517.md` (canonical v3 exemplar), `nested-432.md`
  (v2 section-level exemplar), `narrative-380.md` (legacy).

## Calling sites

- `.claude/agents/analyzer.md` — drafts the body per this spec.
- `.claude/agents/clean-result-critic.md` +
  `codex-clean-result-critic.md` — critique against the lenses and run
  `verify_task_body.py` + `audit_clean_results_body_discipline.py`.
- `.claude/agents/methodology-writer.md` — emits the §2-complete
  methodology doc the body Parameters table is a subset of.
- `.claude/skills/promote-clean-result/SKILL.md` — for legacy HTML
  bodies, optionally converts them to markdown on promotion.
- `CLAUDE.md` § "Experiment Report Structure" — points at this spec.

> **ALWAYS read this SPEC before changing ANYTHING about the report
> structure** — the CLAUDE.md summary, `verify_task_body.py`,
> `analyzer.md`, or any `clean-result-critic` lens. SPEC.md is the source
> of truth; these surfaces must stay in sync.
