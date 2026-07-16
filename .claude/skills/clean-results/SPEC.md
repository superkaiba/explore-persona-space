# Clean-result spec — markdown

The canonical spec for clean-result body shape, voice, sections, and
anti-patterns. The mechanical verifier is **`scripts/verify_task_body.py`**.
The format is **markdown** with YAML frontmatter.

**Generations coexist (forward-only):**

- **v4** (current, sentinel `<!-- clean-result-v4 -->`, migrated 2026-W26)
  — the FOUR-flat-H2 shape (`## Takeaways` / `## Goal` / `## Methodology`
  / `## Results`) + a bold `**Repro:**` / `**Context:**` footer, specced
  in § "v4 body shape" below. **New bodies emit v4.** The standalone
  methodology doc is now a mechanical COPY of the body's `## Methodology`
  section (no separate findings-blind authoring).
- **v3** (grandfathered, sentinel `<!-- clean-result-v3 -->`, migrated
  2026-W24) — the FIVE-flat-H2 shape (`## Takeaways` / `## What I ran` /
  `## Findings` / `## Data` / `## Reproducibility`) specced in § "v3 body
  shape" below. Kept verbatim for parked v3 bodies; NEVER newly
  hard-FAILed by a v4 rule.
- **v2 / legacy** (grandfathered) — the 2-content-section nested-TL;DR
  shape (sentinel `<!-- clean-result-v2 -->`) and pre-sentinel bodies.
  Documented in § "Grandfathered shape (v2 / legacy)" near the bottom.
  These are NEVER newly hard-FAILed by a v3 or v4 rule; the verifier
  branches on the sentinel.

The verifier (`scripts/verify_task_body.py`) gates every check on the
sentinel. A body with NO sentinel is legacy; `<!-- clean-result-v2 -->`
is v2; `<!-- clean-result-v3 -->` is v3; `<!-- clean-result-v4 -->` is
v4. Each generation's checks PASS-skip (NO-OP) on bodies of a different
generation.

**Paper format (separate track, opt-in via `paper: true` frontmatter).**
A task whose frontmatter carries `paper: true` does NOT use a markdown
clean-result body at all — its canonical clean-result is a LaTeX **paper**
(a `.tex` → PDF + a sanitized `paper.html`) under `docs/papers/issue_<N>/`,
and its `body.md` becomes a thin **paper-stub** (title + abstract + paper
link + the confidence/origin frontmatter the existing machinery reads).
The paper format is specced in § "Paper format (`paper: true`)" below and
verified by **`scripts/verify_paper.py`** (NOT `verify_task_body.py`,
which stays the verifier for every markdown body). The `paper:` flag is a
deterministic branch + kill-switch: absent / false ⇒ the markdown
generations above, unchanged. v3/v2/v4-markdown bodies stay grandfathered
and untouched.

---

# Paper format (`paper: true`)

The canonical clean-result for a `paper: true` `kind: experiment` task is a
self-contained **research paper**: a parameterized LaTeX `.tex` (the canonical
source) → a PDF + a pre-rendered, sanitized, committed `paper.html`, under
`docs/papers/issue_<N>/`. The markdown `body.md` for such a task is a thin
paper-stub; the paper IS the clean-result.

This is forward-only and opt-in: a task without `paper: true` uses the markdown
generations above, unchanged. v3/v2/v4-markdown bodies stay grandfathered and
are never converted. Backfill of the existing corpus is on-demand only.

**v1 SCOPE (the shipped scope).** Numbers are written as **literals** in the
`.tex`; the number-correctness guarantee is the analyzer's existing
numeric-fidelity re-extraction (re-derive every reported number from its source
`eval_results` JSON and diff) — there is **no `\metric` grounding requirement
and no `metrics.json` requirement in v1**, and `verify_paper.py` carries NO
`\metric` check. `\epsref{N}` (typed cross-experiment references) IS a v1
feature: the dashboard hover-preview needs it, so every reference to another
experiment uses `\epsref{N}`, never a bare "#N". The `\metric{}` /
`metrics.json` / `verify_metric.py` machinery is fully proven (the spike) and
carried forward under `docs/papers/_template/` as a DOCUMENTED **v1.1 opt-in**;
it is not wired into the v1 required path.

## Paper sections (mapped onto the current v4 semantics)

The paper's sections map onto the same content the v4 markdown shape carries
(v4 `## Methodology` already folds the recipe + the complete hyperparameter
table; v4 `## Results` is the three-beat what-is-plotted → plot →
interpretation). Required sections, in order (enforced by `verify_paper.py`):

1. **Abstract** — SELF-STANDING. A reader who has never seen another EPS
   experiment learns what was tested, on what model, and what was found. State
   the project context in one or two sentences. Maps to the v4 `## Takeaways`
   substance (numbers-first), in prose. NO confidence words.
2. **Introduction** — SELF-STANDING: readable without any other experiment open.
   Gives the project context, states the question THIS experiment answers, and
   names the single variable it changes relative to its line. Maps to the v4
   `## Goal` (`**This experiment in context:**` + `**Broader narrative:**`).
   Refers to prior experiments with `\epsref{N}`, but never depends on the
   reader following one to understand the paper.
3. **Methods** — SELF-CONTAINED: everything needed to reproduce, written out.
   Maps to the v4 `## Methodology` (Design / Training + the complete
   hyperparameter table / Evaluation / Data extraction). **Reuse rule** (same
   as the v4 self-contained-methodology rule): a reused artifact MAY be
   acknowledged with an `\epsref{N}` link, but the Methods section ALWAYS writes
   out HOW that artifact was generated — never a "reused from #N; see there"
   deferral.
   - **Rule A (no-deferral for DIRECT reused artifacts):** for an artifact this
     experiment directly reuses (adapter, training mix, persona vectors, eval
     JSON), write its full generation recipe inline.
   - **Transitive inputs (an input to the thing you reused):** give a compact
     recipe to depth-1, then cite + one-line summarize deeper links.
   - **Inline verbatim examples (training + eval):** the Methods inlines a
     verbatim SUBSET of two of the four mandatory example classes (below) — ≥1
     real training row + ≥1 real eval probe, each in an `\epsexample{...}` block.
4. **Results** — one subsection per finding, each: state what is plotted
   (EXACTLY), show the figure (`\includegraphics` from `figures/issue_<N>/`),
   read the result. Same three-beat as v4 `## Results`; numbers are literals.
   Report the metric, its CI / n, and the test. **Inline a verbatim
   `model-output` worked example for the load-bearing condition(s)** — eval
   INPUT → the model's ACTUAL OUTPUT (verbatim) → the judge VERDICT/score.
5. **Discussion + Limitations** — what the results mean, the alternatives, the
   binding caveats, and what they change. Fold Limitations in here. NO
   confidence words.
6. **References** — from the per-task `issue_<N>.bib` (a copy of / subset of the
   project `.bib`), cited with natbib (`\cite`/`\citep`/`\citet`). Build the
   `.bib` with the `citation-management` skill.
7. **Appendix** — COMPREHENSIVE: the body inlines a SUBSET (2–3 worked
   examples + the load-bearing hyperparameters), the appendix carries the FULL
   set — comprehensive example completions (eval input → model output → judge),
   the full training-data construction recipe + representative rows, the
   COMPLETE hyperparameter table, the full Rule-A reuse recipes, AND a dedicated
   **`\subsection{Judge prompts}`** (the four mandatory example classes' fourth
   member — below).

### Verbatim examples + judge prompts are MANDATORY (show ALL methods AND examples)

A paper SHOWS its data, not just describes the method (incident #657: the paper
described every method but shipped zero verbatim text and no judge prompts). A
`paper: true` clean-result MUST carry verbatim TEXT pulled from real artifacts,
across four classes — enforced by `verify_paper.py` checks 7–9 (7 examples
present, 8 judge prompts, 9 example-provenance pointers) + the
clean-result-critic paper lens P7 + the interpretation-critic paper-mode Lens 7
(the no-invention reality-check, § "No invention" below):

1. **Training-data examples** — ≥2 verbatim sample training rows (the ACTUAL row
   text: system/persona prompt + question + completion, incl. a
   contrastive-negative row where applicable). For a reuse-only study, real rows
   from the REUSED training mixes. Inline a representative subset (Methods);
   Appendix carries the comprehensive set (or a pinned link + a larger sample).
2. **Eval-data examples** — ≥2 verbatim eval inputs/probes (the actual
   false-claim, the harmful/harmless prompt, the steering probe). Inline subset
   (Methods) + Appendix.
3. **Model-output / completion examples** — verbatim WORKED examples per
   load-bearing condition: eval INPUT → the model's ACTUAL OUTPUT (verbatim) →
   the judge VERDICT/score. Inline subset (Results) + Appendix comprehensive.
4. **Judge prompts / rubrics** — when the study uses ANY LLM judge, the ACTUAL
   prompt + rubric TEXT for EVERY judge (e.g. the steering-sanity rubric, the
   sycophancy-agreement judge, the EM judge, the refusal judge), verbatim — in a
   dedicated Appendix `\subsection{Judge prompts}`. A genuine no-judge study
   (pure log-prob / logit) omits this subsection (check 8 then passes).

**Template convention (what the verifier keys on).** Each example block is an
`\epsexample{<caption>}{...}` environment (the `listings`-backed verbatim-safe
box defined in `preamble.tex`; any of `lstlisting` / `verbatim` / `quote` /
`quotation` / `tcolorbox` also satisfies the check) preceded by a
`% eps-example: <class>` comment marker (class ∈ {`training-data`, `eval-data`,
`model-output`}). The Judge-prompts subsection carries the `% eps-judge-prompts`
anchor + `\subsection{Judge prompts}` (the template ships both + a
`{{JUDGE_PROMPTS}}` placeholder). The caption discloses the subset
(K of M / cherry-picked) + links the complete artifact (pinned HF `/tree/<sha>`
or GitHub `/blob/<sha>`).

**Provenance: pull from REAL artifacts, never fabricate.** Training rows from the
training JSONLs, eval probes from the probe bank, completions from HF
`raw_completions`, judge prompts from the rubric file/constant in the eval+
scoring code at the Code SHA. **Sanitize harmful/EM AND real-world-corpus
(LMSYS/WildChat-class; #1073) content** per `analyzer.md`
§ content hygiene — a ~15-word labeled excerpt + a `[truncated — harmful-content
row; verify at <raw-completions path>, row <i>]` placeholder + the pinned raw
path; a sanitized block SATISFIES the requirement.

### No invention — every example is a VERBATIM copy of a real row, word-for-word

The cardinal sin (incident #657: the paper showed a "young child who is curious
about the world and asks lots of questions" persona that **does not exist** in
the data — a fabricated name AND a paraphrased prompt). Every persona, system
prompt, user turn, claim, training row, and model completion in an example MUST
be **copied verbatim from a real artifact**, not reconstructed from memory,
summarized, or paraphrased. Specifically:

1. **Show the FULL system prompt, word-for-word.** When an example involves a
   persona / system prompt, quote the **complete** system prompt string exactly
   as used — copied from the persona definition (e.g.
   `data/canonical_persona_pool/pool_v1.json` -- the string source, not the
   `persona_pool.py` loader -- or the experiment's persona dict)
   or from the chat-templated training/eval row. NEVER a prose paraphrase
   (`system = "you are a doctor"`), NEVER truncated with `...`. If the real
   persona prompt is `"You are a stand-up comedian who writes and performs
   comedy routines."`, that exact string appears.
2. **Show the full chat structure, each turn labeled + verbatim.** A worked
   example shows the SYSTEM message, the USER message, and the ASSISTANT
   completion as three labeled, verbatim parts — the actual input the model saw
   and the actual output it produced. The reader should be able to reconstruct
   the exact prompt.
3. **Real names only.** Persona names, claim text, and dataset ids are the real
   ones from the artifact. A persona named in an example must exist in the
   persona pool / the experiment's realized persona set. If you are unsure a
   persona or row is real, OPEN the artifact and check before writing it.
4. **Truncation rule.** Only a long MODEL OUTPUT may be elided mid-text with an
   explicit `[...]`, and only when the full output is in the Appendix or at the
   cited raw path. The SYSTEM prompt and the USER prompt are NEVER truncated —
   they are short and load-bearing for reproduction. (The harmful-content
   sanitization carve-out above is the one exception, and it keeps the row index
   + raw link.)

**Two-layer enforcement of no-invention.**
- **Mechanical floor — `verify_paper.py` check 9 (example provenance pointers).**
  Every `% eps-example:` block must carry a resolvable pointer to a real
  artifact (an `\epsref{N}`, an `issueN_` slug, a `superkaiba1/` HF path,
  `eval_results/` / `figures/`, a `.json(l)` file, or a recognized HF dataset
  id) IN THE BLOCK'S CAPTION OR BODY (the check reads only the
  `\begin{epsexample}...\end{epsexample}` region — a pointer in the preceding
  prose is NOT seen). A block with NO pointer is unverifiable by construction
  and FAILs. A
  pointer does NOT prove the example is genuine — the #657 fabricated block even
  cited `\epsref{612}` — so the mechanical check is necessary, not sufficient.
- **Semantic catch — `interpretation-critic` paper-mode Lens 7.** The
  interpretation-critic OPENS each example's cited artifact and confirms the
  persona exists, the system prompt is byte-for-byte the real one, and the
  completion / training row is findable in the artifact (verbatim or a faithful
  sanitized excerpt). A persona, prompt, or completion that is invented or
  paraphrased away from the real string is a hard FAIL. This is the layer that
  catches the #657 fabrication.

**NO confidence anywhere in the paper body.** The `(LOW|MODERATE|HIGH
confidence)` tag and bare `Confidence:` lines are a hard FAIL inside the
`.tex`. Confidence lives ONLY in the `body.md` paper-stub frontmatter (so the
existing title-tag / dashboard machinery keeps reading it). `verify_paper.py`
enforces this.

## Layout — `docs/papers/issue_<N>/`

```
docs/papers/
  _template/                  # fixed, shared (NOT per-task)
    issue_TEMPLATE.tex        # parameterized {{...}} template the author fills
    preamble.tex              # FIXED preamble \input by every paper; the author NEVER edits it
    eps_paper_filter.lua      # pandoc filter: \metric→value (v1.1), \epsref→typed HTML link
    paper_schema_extension.mjs# buildPaperSchema(markdownSchema) — dashboard sanitizer allow-list ext
    emit_metrics_tex.py       # v1.1 opt-in: metrics.json → metrics.tex registry
    verify_metric.py          # v1.1 opt-in: \metric grounding check
  issue_<N>/                  # per-task paper dir
    issue_<N>.tex             # the paper (git)
    issue_<N>.bib             # references (git)
    issue_<N>.pdf             # compiled PDF — also uploaded to HF; NOT in git-LFS
    paper.html                # committed, sanitized pandoc render (git)
    paper_manifest.json       # artifact paths + pinned HF PDF URL + sha256 (git)
    metrics.json / refs.json  # v1.1 opt-in (absent in v1)
```

`.tex` / `.bib` / `paper.html` / `paper_manifest.json` are plain git (no
git-LFS). The PDF is stored on the **HF data repo**
(`superkaiba1/explore-persona-space-data/papers/issue_<N>/`) — chosen over
git-LFS for cost (no GitHub-LFS quota) and to reuse the project's existing HF
upload path. The build sets `\graphicspath` to `figures/issue_<N>/` so figures
are referenced repo-relative (and `verify_paper.py` confines `\includegraphics`
to the repo + checks each resolves).

## Build + verify

The deterministic build is `scripts/build_paper.py`: multi-pass `pdflatex
-interaction=nonstopmode -halt-on-error -file-line-error` → `bibtex` (run in the
build dir against the jobname `.aux`) → `pdflatex` ×2, with `SOURCE_DATE_EPOCH`
for a reproducible PDF; then pandoc + the Lua filter → a `paper.html` sanitized
through the REAL dashboard sanitizer (`dashboard/lib/markdown-sanitize.ts` under
the paperSchema extension); then the HF PDF upload (recording the
revision-pinned URL) and `paper_manifest.json`. Build on the VM only (the single
pinned-TeX-Live host). The verifier is `scripts/verify_paper.py` (checks:
compile-clean via `.log`/`.blg` parse, required sections incl. Appendix, no
confidence in body, `\includegraphics` confined + resolve, `.bib` entries
resolve, `\epsref{N}` resolves to a real task, verbatim examples present
(`training-data` / `eval-data` / `model-output` class markers + real example
environments), judge prompts present (a `Judge prompts` appendix when an LLM
judge is used), example provenance pointers (every example block cites a real
artifact — the no-invention floor), manifest complete + hashes match, paper-stub
valid).
`verify_task_body.py` is untouched and remains the verifier for grandfathered
markdown.

## `body.md` paper-stub contract

For a `paper: true` task, `body.md` is a stub the existing readers (REGISTRY
denormalization, dashboard, follow-up-proposer, ...) consume. It carries:

- **Frontmatter** — `paper: true` (the deterministic branch + kill-switch),
  plus the fields the existing machinery reads: `title`, `kind`, `goal`,
  `has_clean_result`, and the **confidence** (the title's `(LOW|MODERATE|HIGH
  confidence)` tag — confidence lives here, never in the paper body),
  origin/lineage fields.
- **Body** — the H1 title, the abstract (so the dashboard hover-card + REGISTRY
  title/abstract denormalization work), and a paper link (the `docs/papers/
  issue_<N>/` artifacts and/or the pinned HF PDF URL).

`verify_paper.py`'s paper-stub check enforces `paper: true` + an H1 + an
abstract + a paper link. (Wiring the stub into `task.py set-body --allow-stub`
and `set-clean-result`'s manifest validation, plus every reader, is later-phase
work — Phase A only defines the contract + the verifier check.)

## JSON schemas

**`paper_manifest.json`** (`schema: "paper_manifest/v1"`):

```json
{
  "schema": "paper_manifest/v1",
  "issue": 657,
  "jobname": "issue_657",
  "built_at": "<ISO-8601 UTC>",
  "source_date_epoch": "<int string used for the reproducible build>",
  "pdf_hf_url": "https://huggingface.co/datasets/<repo>/resolve/<commit>/papers/issue_<N>/issue_<N>.pdf",
  "artifacts": {
    "<label>": { "path": "<repo-relative path>", "sha256": "<hex>", "bytes": <int> }
  }
}
```

Required artifacts: `tex`, `pdf`, `paper_html` (each present on disk with a
matching sha256). `bib` / `metrics_json` / `refs_json` are recorded when
present. `pdf_hf_url` is `null` for a local-only (`--no-upload`) build (a WARN,
not a FAIL — a paper can be verified pre-upload).

**`refs.json`** (v1 optional; the `\epsref` target index — used by the
dashboard hover-preview in a later phase):

```json
{
  "schema": "refs/v1",
  "epsrefs": [ { "issue": 623, "context": "parent — base-rate predictor" } ]
}
```

In v1 `verify_paper.py` resolves `\epsref{N}` directly against the task registry
(it does not require `refs.json`); `refs.json` is the dashboard's
build-time-emitted convenience index, populated when the dashboard phase lands.

## v1.1 opt-in (`\metric` grounding) — carried forward, NOT wired into v1

The spike proved full `\metric` grounding (number → `metrics.json` pointer →
`eval_results` JSON, precision-aware). It ships as a documented opt-in under
`docs/papers/_template/`:

- **`metrics.json`** — `{ "<key>": { "value", "source", "transform",
  "precision", "rendered" } }`, where `source` is either an `eval_results`
  pointer `{ "file", "json_path" }` or `{ "kind": "analysis-derived",
  "producer", "inputs": ["file#json_path", ...] }`. (Derived value rule: a
  derived number is groundable iff its producing script persists it to an
  `eval_results` JSON.)
- **`emit_metrics_tex.py`** — generates `metrics.tex` (one
  `\csname metric@<key>\endcsname` macro per key) which the paper `\input`s, so
  the `.tex` compiles standalone with no shell-escape.
- **`verify_metric.py`** — parses `\metric{key}` calls, resolves each in
  `metrics.json`, checks `rendered` is consistent with `value`+`precision`+
  `transform`, and re-resolves each grounded source pointer in `eval_results`.

To turn v1.1 on: emit a `metrics.json` alongside the paper, replace literal
numbers with `\metric{key}` calls, and add a `\metric` pass to `verify_paper.py`
that calls `verify_metric.py`'s logic. `build_paper.py` already regenerates
`metrics.tex` and runs the Lua-filter `\metric`→value rewrite transparently when
a `metrics.json` is present, so no build change is needed.

---

# v4 body shape (current)

Four FLAT H2 sections, in this exact order, then a bold-label footer (NOT
H2s). New sentinel `<!-- clean-result-v4 -->` right after the H1. The v4
redesign folds the v3 `## What I ran` + `## Data` content INTO an expanded
`## Methodology` (which also absorbs the entire former standalone
methodology doc), and collapses the per-result `## Findings` skeleton into
a strict three-beat `## Results` (what-is-plotted-EXACTLY → plot →
interpretation). The v3 `## Reproducibility` H2 becomes the compact
`**Repro:**` / `**Context:**` footer.

```markdown
# <one-sentence claim> (LOW|MODERATE|HIGH confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_N.md](…) · [gist](…)   ← orchestrator-appended, post-gate

## Takeaways

- <headline finding, key number + CI bolded>
- <secondary finding>
- <the caveat that binds interpretation>
- <what this changes / next decision>
(3–6 bullets, each ≤30 words, numbers-first, plain academic register.
ALWAYS the cross-round synthesis — rewritten after every follow-up round.
Carries the v3 Takeaways rules VERBATIM.)

## Goal

- **This experiment in context:** <what THIS specific experiment tests
  and how it relates to the OTHER experiments in its line. The ONLY place
  prior-issue links appear — `[#K](https://eps.superkaiba.com/tasks/K)`.
  This absorbs the v3 `## What I ran` `**Why:**` slot.>
- **Broader narrative:** <the goal of this experiment / group of
  experiments in the project's broader narrative — the
  `docs/open_questions.md` anchor / project-level question it serves.>

## Methodology

The full "everything required to understand the results" section. Absorbs
the v3 `## What I ran` Design/Training/Eval AND the entire former separate
methodology-doc content. Boldface-led slots:

- **Design:** <conditions × seeds × N; the single manipulated variable.>
- **Training:** <complete recipe + the COMPLETE hyperparameter table
  (every training + eval + generation hyperparameter, each value from
  ground truth — committed config / run_result.json / plan §11 — with a
  Source column). This is the canonical complete table; the exported
  methodology doc §Hyperparameters is a COPY of it.>
- **Evaluation:** <DV definition, computed metrics, judge model + rubric,
  probe set (identity / why chosen / preprocessing).>
- **Data extraction:** <how the training/eval data was built/extracted.>
- **Sample training/evaluation data + completions:** <verbatim worked
  examples (eval input → model output → judge score, one per load-bearing
  condition) + sample training rows + sample probes, EACH preceded by a
  subset-disclosure line + a pinned full-artifact link. Wrap example
  rows/completions in `<details>` or a fenced code block.>

## Results

One `### <result>` H3 per result, each in this STRICT three-beat order:

1. **What is plotted (EXACTLY)** — a precise statement of exactly what
   the figure shows: axes, units, what each point/bar is, n, any
   transform. (1–3 sentences/bullets above the figure.)
2. **Plot** — exactly ONE inline figure (`![alt](permanent-url)` on its
   own line, blank line before and after) with a markdown blockquote
   caption (`> **Figure.** *italic lead.* plain caption ≤60 words`). ALL
   details of what's plotted live in the alt text + caption.
3. **Interpretation** — what it means / what it can't tell you. (1–3
   sentences/bullets below the caption.)

REQUIRE BOTH a high-level summary-metric plot (correlation / forest / bar)
AND the LOW-LEVEL per-unit data plot (scatter / per-point) for any
aggregate result, with points LABELED as much as possible. (This makes
"low-level data plot behind every aggregate" + point-labeling
first-class.) Multiple `### <result>` allowed.

---
**Repro:** <compute (wall time, GPU type/count, pod label)> · <code SHA,
GitHub blob/tree links pinned to the SHA> · <artifact links: training
data, checkpoints, eval JSONs, raw completions, figure source — pinned>

**Context:** <verbatim originating prompt(s), blockquoted> · <lineage:
`[#K](...) — <one line>` or `fresh direction (no parent)`; same-issue
follow-up rounds also name each round's followup_label, each as a
'same-issue follow-up round `<label>`' clause> · <created/run
dates>
```

## Section-by-section (v4)

### `## Takeaways` (v4)

Identical to v3. Plain academic register — NO lowercase-casual voice, NO
"How this updates me" diary framing. Bullets only, numbers-first,
directly adaptable into a Slack post. **3–6 bullets** (hard FAIL outside
that range), each ≤30 words (WARN; ≥100 words is a hard FAIL). The H1
title stays the one-sentence
claim + confidence tag. It is the rolling cross-round synthesis, rewritten
after every follow-up round (§ Follow-up consolidation).

### `## Goal` (v4)

The two-part contextualization the v3 `**Why:**` slot under-served. TWO
required boldface-led parts:

- **`**This experiment in context:**`** — what THIS specific experiment
  tests and how it relates to the OTHER experiments in its line. The
  **ONLY place** in the body that may cite prior tasks
  (`[#K](https://eps.superkaiba.com/tasks/K)` links) — `## Methodology`
  and `## Results` are standalone (descriptive baselines, not
  `#K`-linked). **Do NOT stage the writeup as a methodology correction of
  a prior run** — describe the open question and what this run did; never
  "the prior run used X, this run uses Y".
- **`**Broader narrative:**`** — the goal of this experiment / group of
  experiments in the project's broader narrative (the
  `docs/open_questions.md` anchor / project-level question it serves).

The frontmatter `goal:` field stays in the body for agent-facing
reference (planner, critic, follow-up-proposer read it).

### `## Methodology` (v4)

The complete "everything required to understand the results" section. It
folds the v3 `## What I ran` Design/Training/Eval bullets AND the entire
former standalone methodology-doc content (overview, complete
hyperparameter table, training-data recipe, evaluation recipe, verbatim
worked examples) into ONE section.

**Rule A — `## Methodology` is SELF-CONTAINED (no deferral to another
issue).** This section reads like a research-paper Methods section: a
reader understands HOW every reported result was produced WITHOUT
following a link to another issue. When an artifact was REUSED from a
prior experiment (a trained adapter, persona-vector bank, behavior
direction, leakage cells, dataset, base-rate / propensity measurement),
the Methodology WRITES OUT THE FULL PROCEDURE that produced it — data
source + realism tier, construction recipe, training recipe +
hyperparameters, measurement — as PRIMARY METHOD, exactly as if performed
for this experiment. Pull that procedure from the source issue's own
`## Methodology` (or `docs/methodology/issue_<M>.md`) and inline it. The
Methodology body MUST NOT say `reused from #X` / `see #X` / otherwise
defer a load-bearing method to another issue. **The FACT of reuse +
source issue `[#M](...)` + pinned artifact link are recorded ONLY in the
`**Repro:**` footer** as a citation / reproducibility note (§ `**Repro:**`
/ `**Context:**` footer reuse-provenance bullet). Rule A SUPERSEDES, for
the Methodology BODY, the older "name the reuse inline" pattern: the body
always spells out the method; the footer always carries the provenance.
(This does not change the footer's reuse-provenance requirement — the
footer still names `#M` + path + a one-line fitness rationale; Rule A is
purely about the body's method prose being complete and standalone.)

Boldface-led slots, in order:

- **`**Design:**`** — conditions × seeds × N; the single manipulated
  variable.
- **`**Training:**`** — complete recipe + the **COMPLETE hyperparameter
  table** (EVERY training + eval + generation hyperparameter, each value
  copied from ground truth — committed config / `run_result.json` / plan
  §11 — with a **Source** column). This is the canonical complete table.
  Every numeric hyperparameter is COPIED from ground truth, never typed
  from memory. The learning rate is reconciled against the plan (check
  v4-lr). Incident: task #489 shipped `lr = 1e-4` while the run used
  `lr = 2e-6` — a 50x misprint.
  - **Analysis-only / no-training tasks** (a `kind: analysis` task, or a
    zero-GPU `kind: experiment` that trains no model): write the Training
    slot as `**N/A — no model training.**` and put the analysis-design
    constants (bootstrap B, spline knots, logit ε, thresholds) in the
    Evaluation slot.
- **`**Evaluation:**`** — DV definition (construct + metric + on/off-policy
  choice), computed metrics, judge model + rubric, probe set (identity /
  WHY chosen / preprocessing).
- **`**Data extraction:**`** — how the training/eval data was built /
  extracted: source + realism tier, construction recipe, N rows,
  composition/ratio (positives:negatives, persona panel), completion
  provenance (on-policy tier / canned / published-corpus-verbatim per
  `.claude/rules/on-policy-completions.md` + `.claude/rules/contrastive-negatives.md`).
- **`**Sample training/evaluation data + completions:**`** — verbatim
  worked examples: a sample of training rows, a sample of eval probes, and
  one end-to-end completion per load-bearing condition (eval input → model
  output → judge score). **Each example block (fenced OR `<details>`) is
  immediately preceded by a subset-disclosure line** (`K of M rows,
  random sample` / `cherry-picked for illustration` / `first N of M` / the
  harmful-content sanitized form) AND followed/preceded by a **pinned link
  to the complete artifact** (HF Hub `/tree/<sha>`, GitHub `/blob/<sha>`).
  Harmful-content corpora ship SANITIZED (see § Harmful-content below).

**Per-condition quantitative numbers live in PLOTS (in `## Results`), not
as a body table** — never duplicate a per-condition rate / log-prob / mean
as a markdown table when a figure already carries it. The complete
hyperparameter table is the exception (it belongs here, in `## Methodology`
→ Training).

### `## Results` (v4)

One `### <result>` H3 per result. Each result is STANDALONE — a reader can
land on it directly and understand it. Issue numbers are confined to
`## Goal` and the `**Repro:**` / `**Context:**` footer; baselines are
framed descriptively ("the narrow 2-negative baseline"), not by number.

Per-result skeleton — the strict three-beat (THIS is the v4 contract):

1. **What is plotted (EXACTLY)** — a precise statement of exactly what the
   figure shows ABOVE it: axes, units, what each point/bar is, n, any
   transform. Not "why we ran this" (that is `## Goal` / `## Methodology`)
   — strictly "what this figure depicts".
2. **Plot** — exactly ONE inline figure (`![alt](permanent-url)` on its
   own line, blank line before and after) with a markdown blockquote
   caption (`> **Figure.** *one-sentence lead claim in italics.* plain
   caption ≤60 words`). ALL details of what's plotted ALSO live in the alt
   text + caption (so the figure is self-describing).
3. **Interpretation** — what it means / what it can't tell you, BELOW the
   caption (1–3 sentences/bullets).

**Low-level data plot behind every aggregate (first-class in v4).** Any
result that reports an AGGREGATE statistic — a correlation ρ shown as a
forest-plot point, a mean / effect size shown as a bar, a p-value, an
effect summary — MUST embed BOTH a high-level summary-metric plot AND the
LOW-LEVEL per-unit data plot (the scatter the ρ summarizes, the strip /
swarm / jittered per-point view behind the group-difference bars, the
unbinned counterpart of a binned view), **with points LABELED as much as
possible** (each point names its unit — persona / seed / cell). The
reader sees the data, not only the number computed from it. The raw +
processed pair rides inside the SAME `### <result>` and counts as ONE
narrative unit (one what-is-plotted above the pair, one interpretation
below). Exemptions, stated in the interpretation prose or alt text: the
result's primary figure ALREADY is the per-unit view (a raw scatter needs
no second scatter); N is so small the figure already shows every point; or
the aggregate has no meaningful per-unit decomposition (a single scalar).

**Raw alongside processed** (the transformed-figure special case). When a
result's figure plots a residualized / partialled / binned /
log-transformed / normalized / aggregated quantity, also embed its RAW
(pre-processing) counterpart inside the same `### <result>` (raw first,
then processed), and quote the RAW point estimate alongside the controlled
one in the interpretation prose. Same principle at the artifact layer:
when a claim rests on an aggregated metric, link BOTH the aggregated file
and the per-cell file the aggregation collapsed (in the `**Repro:**`
footer). Exemption: raw and processed are visually identical
(axis-rescale-only) — say so in alt text and omit the raw.

**For text-behavior results only:** the systematic per-condition samples
live in `## Methodology → ### Sample training/evaluation data +
completions`; a `### <result>` may carry at most ONE short (≤10-line)
raw-completion excerpt where the text itself IS the result — preceded by a
subset-disclosure line AND a raw-completions link.

**For runs that generate NO completions** (teacher-forced log-prob,
activation probe, linear-fit): state the measurement-validity tell inside
the interpretation prose; do NOT fabricate a sample block.

**No `### Methodology corrections` heading.** When a methodology
correction is load-bearing for interpreting a result, fold it into that
result's what-is-plotted or interpretation prose.

Per-result prose cap: ≤120 words WARN / ≥180 words FAIL (excl. caption,
tables, code, `<details>` bodies; the what-is-plotted + interpretation
beats together).

### `**Repro:**` / `**Context:**` footer (v4)

The v3 `## Reproducibility` H2 becomes a compact bold-label footer (NOT an
H2), preceded by a `---` horizontal rule. TWO required bold labels:

- **`**Repro:**`** — compute (wall time, GPU type/count, pod label) ·
  code SHA (GitHub `/blob/<sha>` or `/tree/<sha>` links, never
  `main`/`master`/`HEAD`) · artifact links (training data, checkpoints,
  eval JSONs, raw completions, figure source — all pinned). **Reuse
  provenance** — when a reader-facing claim rests on a trained artifact
  REUSED from a prior issue, name per reused artifact: the producing issue
  `[#M](...)`, the permanent pinned path, and a one-line fitness rationale.
  All URLs pinned (HF Hub `/tree/<sha>`, WandB `/runs/<id>`, GitHub
  `/blob/<sha>` or `/tree/<sha>`); `n/a` accepted as an explicit
  non-applicable marker. No `{{` / `TBD` / `see config` / `default`
  placeholders.
- **`**Context:**`** — run-context provenance: the verbatim originating
  prompt(s) blockquoted (sourced from frontmatter `origin_prompt` / the
  original body's `## Provenance` / `epm:followup-scope v1` markers, NEVER
  paraphrased; when none recorded, `origin prompt not recorded`) · lineage
  (`[#K](...) — <one line>` or `fresh direction (no parent)`; same-issue
  follow-up rounds name each round's `followup_label` — written as a
  ``same-issue follow-up round `<followup_label>` `` clause, the
  machine-countable form check 20's round-scaled prose budget reads;
  #921) · created/run dates.
  Mechanically enforced (check 17): the normalized frontmatter
  `origin_prompt` must appear verbatim within the `**Context:**` row's
  text — a ≥20-char quoted candidate that is a strict prefix of
  `origin_prompt` covering ≥50% of it (truncation) is a hard v4 FAIL; any other mismatch
  WARNs (v3/v2: WARN-only). When the row quotes a longer alternate
  verbatim source (`## Provenance` / followup-scope), quote the
  frontmatter `origin_prompt` too, or expect the WARN.
  Bare / unpinned URLs INSIDE the verbatim blockquote are exempt from
  the footer URL checks (checks 8 / 8b strip `>`-prefixed lines before
  scanning — the quote is provenance text, not a provenance link; #959;
  applies identically to grandfathered v3 `## Reproducibility` bodies).
  Pinned-link requirements continue to bind on every non-quoted footer
  row. A verbatim prompt containing URLs must `>`-prefix EVERY prompt
  line — an un-prefixed lazy-continuation line stays scanned
  (fail-closed).
  This footer is the ONLY place run-context provenance lives in the body
  (the "state facts, not sources" rule still bans weaving prompt/person
  attributions into Takeaways / Results prose).

**Confidence lives in the H1 title tag ONLY.** There is NO `Confidence: …`
sentence anywhere in a v4 body. The binding caveat lives in the relevant
result's interpretation prose and/or a `## Takeaways` bullet.

## The standalone methodology doc (v4 — a mechanical COPY)

Under v4 the methodology doc is no longer separately authored by a
findings-blind `methodology-writer` agent. It is a **mechanical EXPORT**:
after the body is finalized (post clean-result-critic PASS), the
orchestrator copies the body's `## Methodology` section verbatim into
`docs/methodology/issue_<N>.md` (the `## Methodology` H2 header normalized
to `# Methodology — issue <N>`), commits it to `main` (durable, by
explicit path), publishes the secret gist, and appends the top-of-body
`**Methodology:**` link pinned to the `main` commit SHA. The doc is
DERIVED FROM the body — the body's `## Methodology` section is canonical.
Committing it to `main` as part of body finalization removes the old v3
durability gap (the doc + its SHA-pinned link land on `main` directly, not
only on the worktree branch).

The export is idempotent via `epm:methodology-doc-generated v1`. Skipped
for `kind: infra | batch | survey`; `kind: analysis` runs only when the
task has a discernible training/eval methodology. Full procedure:
`.claude/skills/issue/SKILL.md` § 9a-quater.

## Dashboard data-artifact interface (Phase 2 contract)

The interactive dashboard data-viewer (Next.js app at `dashboard/`, a
separate Phase-2 worker) consumes the per-task data files the
`## Methodology` and `## Results` sections expose. To give that worker a
stable contract, a v4 body exposes its data through exactly these,
machine-discoverable interfaces:

- **`## Methodology → **Sample training/evaluation data + completions:`**
  — each example block is wrapped in a `<details>` element or a fenced
  code block, is preceded by a subset-disclosure line, and is paired with
  a pinned full-artifact link (HF Hub `/tree/<sha>` for training rows /
  raw completions / probe banks, GitHub `/blob/<sha>` for committed eval
  JSONs). The full-artifact link is the canonical "load more / sort /
  filter" target the viewer will fetch and paginate.
- **`## Results → ### <result>` figures** — each inline figure is a
  SHA-pinned `raw.githubusercontent.com/.../<sha>/figures/issue_<N>/<file>.png`
  URL whose `.meta.json` sidecar (committed alongside the PNG/PDF by the
  plot step) carries the per-point data the viewer renders interactively.
  The low-level per-unit plot's `.meta.json` is the per-row data table the
  viewer's sort/filter/reveal-more operates on.
- **`**Repro:**` footer** — the per-cell artifact links (aggregated JSON +
  the per-cell file the aggregation collapsed) are the viewer's
  drill-down sources.

The Phase-2 viewer builds against THIS interface (subset-disclosed
`<details>` blocks + pinned full-artifact links + per-figure `.meta.json`
sidecars). It is OUT OF SCOPE for the v4 spec itself — the markdown body
ships `<details>` + pinned links NOW; the interactive sort/filter/
reveal-more is the viewer's job. Do NOT add viewer-specific markup to the
body.

## Conciseness caps (v4, mechanical — check v4-word-caps)

Same constants as v3 (`V3_TAKEAWAYS_*`, `V3_FINDING_PROSE_*`,
`V3_FIGURE_CAPTION_MAX_WORDS`, `V3_TOTAL_PROSE_*`), applied to the v4
sections, plus TWO v4-only constants: `V4_TAKEAWAYS_BULLET_FAIL_WORDS`
(=100), the per-Takeaways-bullet hard-FAIL tier (#825), and
`V4_RESULT_PARA_MAX_SENTENCES` (=3), the per-result-paragraph sentence
cap (WARN-only, check 36, #1368):

| Surface | Cap | Verifier behavior |
|---|---|---|
| `## Takeaways` bullet count | 3–6 bullets, no paragraphs | FAIL outside range (owned by the v4 structure check) |
| Per-Takeaways-bullet length | ≤30 words WARN; ≥100 words FAIL | WARN at 30, FAIL at 100 (v4-only hard tier — an accreted paragraph-bullet cannot ride a WARN, #825) |
| Per-`### <result>` prose (excl. caption/code/details/tables) | ≤120 words WARN, ≥180 FAIL | WARN at 120, FAIL at 180 |
| Per-`### <result>` prose paragraph | 1–3 sentences | WARN at ≥4 (check 36, #1368; register judgment — bullets-over-prose, the FAIL call — stays with the LM critic, Lens 12) |
| Figure caption | ≤60 words | WARN |
| Total prose: Takeaways + Goal + Results (excl. tables, code fences, details bodies, captions; `## Methodology` is EXCLUDED — it carries the absorbed methodology-doc content and is reference, not skim prose) | ≤800 words + 250 per live follow-up round beyond the first (round count: non-retroactive `epm:same-issue-followup-run` markers and/or the footer round clauses, max — #921) | WARN-only |

`## Methodology` is deliberately EXCLUDED from the total-prose budget: it
absorbed the entire former standalone methodology doc, which was never
under the skim-prose cap. The per-`### <result>` ≥180-word FAIL and the
per-Takeaways-bullet ≥100-word FAIL are the hard gates.

## Follow-up consolidation (v4)

Same as v3:

1. **`## Takeaways` is the rolling synthesis.** After every round, rewrite
   it to the current cross-round belief and retitle the H1 if the headline
   moved. A Takeaways describing only round 1 after round 2 landed is a
   critic FAIL. A retitled H1 is synced to frontmatter via
   `task.py set-title <N> "<new H1 text>"` after the `set-body` (set-body
   preserves frontmatter; `check_h1_matches_frontmatter_title` FAILs a
   diverged v4 body).
2. **Round visibility.** `## Methodology → **Design:**` gains a per-round
   note (or a `**Rounds:**` table) when >1 round; `**Context:**` keeps
   per-round followup_labels + verbatim prompts. The complete
   hyperparameter table gains a per-round column.
3. **Superseded-result hygiene.** When a round invalidates an earlier
   result, rewrite `## Results` to the current best understanding and
   collapse the outdated block into ONE `<details><summary>Superseded by
   round N</summary>` block at the end of `## Results`.
4. **Round-compression hygiene.** When a round's synthesis ABSORBS an
   earlier result, that result compresses to heading + figure + ≤2
   bullets.
5. **Migrate-on-fold.** A same-issue follow-up round that lands on a v3
   (or v2) body AFTER the v4 cutover migrates that body to v4 as part of
   the fold (the analyzer rewrites the body anyway; drafts rebuild
   cheaply). This is the ONE deliberate exception to "parked bodies stay
   v3/v2".

### V4 sentinel

NEW bodies carry the literal HTML comment `<!-- clean-result-v4 -->` right
after the H1 (the analyzer emits it on draft). The verifier uses it to
gate every v4 rule. Bodies carrying `<!-- clean-result-v3 -->` (or v2 /
no sentinel) keep their prior generation's behavior and are NEVER
hard-FAILed by a v4 rule (forward-only).

### Top-of-body methodology link (v4)

Identical mechanism to v3: the orchestrator (`/issue` Step 9a-quater LATE
JOIN, after clean-result-critic PASS) appends a one-line reader-facing
pointer immediately after the `<!-- clean-result-v4 -->` sentinel, before
`## Takeaways`:

```
**Methodology:** [docs/methodology/issue_<N>.md](https://github.com/superkaiba/explore-persona-space/blob/<DOC_SHA>/docs/methodology/issue_<N>.md) · [gist](<GIST_URL>)
```

When the gist publish fail-softed, the `· [gist](...)` suffix is dropped.
Forward-only + post-gate: the line is appended AFTER the gate, so a body
under critique normally does NOT carry it yet. The verifier and critics
never REQUIRE it and never flag it as a stray element when present.

### All footer URLs pinned (v4)

Same as v3: HF Hub `/tree/<ref>` or `@<ref>`, WandB `/runs/<id>`, GitHub
`/blob/<sha>` or `/tree/<sha>` — never `main` / `master` / `HEAD`. `n/a`
accepted. No `TBD`, `{{`, `default`, `see config` sentinels. **Write
MDX-safe markdown** (same three rules as v3): (a) `[label](url)` only,
never `<https://...>` autolinks; (b) no `<` immediately before a digit
(`p<0.05`); (c) table-cell tokens with inner pipes (`<|im_start|>`) escape
the pipes inside a code span.

### Stray `## What I ran` / `## Findings` / `## Data` / `## Reproducibility` / `## Human TL;DR` / `## TL;DR` / `## Details` / `## Figure` is a FAIL (v4)

A v4 body that includes any of the v3 content H2s (`## What I ran`,
`## Findings`, `## Data`, `## Reproducibility`) OR any retired earlier H2
(`## Human TL;DR`, `## TL;DR`, `## Details`, `## Figure`) is rejected by
the verifier (forces clean migration to the v4 four-H2 shape). The v4
shape uses `## Goal` / `## Methodology` / `## Results` + the footer.

## Figure caption shape — markdown blockquote + bold "Figure." prefix (v4)

Identical to v3. Every figure caption inside a `### <result>` H3 wraps in
a markdown blockquote (`> ` prefix):

```
> **Figure.** *One-sentence lead claim in italics.* Remaining caption
> prose in plain text — definitions, n per condition, panel meanings,
> color mapping, what the reader should look at, what the figure does
> NOT show.
```

Discipline: blank line BETWEEN body prose and image; blank line BETWEEN
image and caption; no 4-space indent.

## Voice (v4)

**Rule B — research-paper register.** The entire v4 body is written in the
concise, precise register of a research paper: declarative methods/results
prose, every quantity DEFINED on first use, no filler / marketing / hype.
This REFINES the "bullets are the default" guidance below per section:

- `## Methodology` = **Methods-section PROSE** — the complete procedure
  written as compact declarative paragraphs (with the hyperparameter table
  + verbatim example blocks as data), NOT terse bullet fragments. A reader
  reproduces the run from it.
- `## Results` = **Results-section PROSE** per `### <result>`: the
  what-is-plotted-EXACTLY beat → figure → interpretation beat, each a
  compact declarative paragraph (1–3 sentences), NOT bullet fragments.
- `## Takeaways` STAYS numbers-first BULLETS (abstract-style), 3–6 of them.
- `## Goal` keeps its two boldface-led slots (compact prose each).

So the "bullets default" rule below applies to `## Takeaways` (and is the
fallback inside a slot where a flat enumeration genuinely reads better);
`## Methodology` and `## Results` are compact PROSE. Conciseness caps
(§ Conciseness caps) still bind — research-paper register means tight, not
verbose.

Otherwise identical to v3:

- **Bullets are the default for `## Takeaways`; prose only where a causal
  chain needs 1–3 sentences** (in `## Methodology` / `## Results`, prose IS
  the default per Rule B — keep it to 1–3-sentence units, matching the v4
  three-beat register rule at § Voice (v4) Rule B and the canonical
  three-beat definition above).
- `I`, not `we`.
- Direct declarative ("The observed correlation was X").
- Plain academic register in `## Takeaways`.
- No fluff transitions ("One more wrinkle:", "the buried lede was", …).
- Caveats fold into the relevant result's interpretation prose and/or a
  `## Takeaways` bullet.
- Inline math `\(...\)`, display math `\[...\]`. Keep math out of plot
  labels and captions.
- **Never write `byte identical` or `byte-identical`** anywhere.
- **Statistical-framing discipline** carries over (enforced by
  `audit_clean_results_body_discipline.py` + clean-result-critic): no
  pre-registration mentions, no effect-size names in prose, no named
  statistical tests in narrative prose, no inline `value ± err` credence
  intervals (chart error bars fine), no project-internal condition labels
  (`C1`/`H1`).

## Mechanical checks (`verify_task_body.py`) — v4

Forward-only: each check branches on the sentinel. The v4 checks
(NO-OP-PASS on v3 / v2 / legacy bodies):

1. Title ends with `(LOW|MODERATE|HIGH confidence)`. (Generation-agnostic.)
2. Four required H2 sections present in order (`## Takeaways`, `## Goal`,
   `## Methodology`, `## Results`). A stray `## What I ran` / `## Findings`
   / `## Data` / `## Reproducibility` / `## Human TL;DR` / `## TL;DR` /
   `## Details` / `## Figure` H2 is a hard FAIL.
3. v4 structure (`check_v4_structure`): `## Takeaways` has **3–6 bullets**
   (the AUTHORITATIVE count gate), `## Goal` carries BOTH the
   `**This experiment in context:**` AND the `**Broader narrative:**`
   slots, `## Methodology` carries the `**Training:**` (or the
   `**N/A — no model training**` marker) + `**Evaluation:**` slots,
   `## Results` has ≥1 `### ` result.
4. At least one `![alt](url)` image inline under `## Results`.
4b. Figure URLs resolvable AND existing under `## Results` (same offline
   `git cat-file` / HTTP HEAD probe as v3).
5. (Soft) Figure-caption sanity — vacuously satisfied.
6. Confidence — for v4 the H1 title tag is the source of truth; PASSes
   when the title carries the `(... confidence)` tag, NO body Confidence
   sentence required. Gated on `is_titletag_confidence()` = v2 OR v3 OR v4.
7. `**Repro:**` footer present with code + artifact links (replaces the
   v3 Reproducibility-subgroups check). The `**Context:**` label present.
8. Footer URLs pinned to permanent refs.
8b. Footer same-repo artifact URLs exist (`git cat-file` / HTTP HEAD).
9. Footer has no placeholder sentinels.
10. Cherry-picked / random-sample label preceding every sample-output
    block in `## Methodology` + `## Results`.
11. Qualitative-data (raw-text-artifact) link preceding every
    sample-output block in `## Results` + `## Methodology → Sample
    training/evaluation data + completions` ONLY.
11b. Planned-vs-actual denominator consistency — headline surface is
    `## Takeaways` + `## Results`; scope-correction scan is whole-body.
13. Results narrative flow (WARN-only) — outline-label H3s + figure-dump
    heuristics, scanned over `## Results`.
14. MDX-safe prose (generation-agnostic).
15. Footer "committed at commit `<sha>`" claims resolve.
16. Footer lr matches plan (gated on `is_titletag_confidence()`; the
    `**Methodology:**` Training table lr must appear in the plan).
17. `**Context:**` provenance present, and (v4) the row carries a lineage
    token — `[#K](...)`/bare `#K`, `fresh direction (no parent)`/`fresh (no
    parent)`, or a same-issue-follow-up-round clause (gated on
    `is_titletag_confidence()`) — and (v4) the `**Context:**` row's text
    must contain frontmatter `origin_prompt` verbatim
    (whitespace-normalized; a ≥20-char strict-prefix quote covering ≥50%
    of `origin_prompt` = hard FAIL on truncation, other mismatch = WARN;
    v3/v2 WARN-only).
18. **`## Methodology` completeness** (`check_v4_methodology_shape`, v4
    only): the `**Training:**` slot carries the complete hyperparameter
    table (≥1 GFM table after the Training label, OR the explicit
    `**N/A — no model training**` marker), AND the `**Sample
    training/evaluation data + completions:**` slot carries ≥1 example
    block each preceded by a subset-disclosure line + paired with a pinned
    complete-artifact link OR an explicit `n/a — <reason>`.
19. **`## Methodology` subset-disclosure** (v4 only): every example block
    inside `## Methodology` is preceded by a subset-disclosure line.
20. **Word caps** (`check_v4_word_caps`, v4 only): the § Conciseness caps
    table above. FAILs on the per-`### <result>` ≥180-word hard cap and
    the per-Takeaways-bullet ≥100-word hard tier; everything else is
    WARN. `## Methodology` excluded from the total budget.
21. **Results beat shape** (`check_v4_results_beat`, v4 only, WARN): each
    `### <result>` carries a figure framed by what-is-plotted prose ABOVE
    and interpretation prose BELOW (the three-beat). WARN (not FAIL) so a
    legitimately figure-less qualitative result is not blocked; the
    clean-result-critic owns the substantive beat read.
27. **No bare issue refs in standalone sections**
    (`check_v4_no_bare_issue_refs`, v4 only): a bare `#<digits>` token in
    the `## Takeaways` / `## Methodology` / `## Results` section spans is
    a hard FAIL — prior-issue references live ONLY in the `## Goal`
    context slot (`[#K](...)` links) and the `**Repro:**`/`**Context:**`
    footer — and a prior-issue TASK LINK (any form whose text carries
    `https://eps.superkaiba.com/tasks/<digits>`: a `[#K](...)` markdown
    link, a `<...>` autolink, or a bare URL) in a standalone section is a
    hard FAIL too, not just the bare token. Sanctioned forms that do not
    trip: markdown links to NON-task targets (GitHub blobs, HF, figures),
    GFM table rows (the Training-table Source column), fenced + inline
    code, `<details>` blocks, HTML comments, the footer, frontmatter. The
    inline-code escape hatch is for non-issue strings (colors, ordinals,
    verbatim syntax examples) only — never for a genuine issue reference
    or task link. (Origin: the #841 round-2 two-round Lens-2 miss; the
    task-link form added by #1002 after the #928 round-1 miss; numbered
    27 because 22-26 are taken by the generation-agnostic checks.)
36. **Result-paragraph sentence cap**
    (`check_v4_result_paragraph_sentences`, v4 only, WARN — NEVER FAIL):
    each prose paragraph inside a `### <result>` block runs 1–3 sentences
    (`V4_RESULT_PARA_MAX_SENTENCES` = 3; the § Conciseness caps table
    above). Paragraph = a maximal run of consecutive prose lines —
    blockquote captions, fenced code, `<details>` bodies, GFM tables,
    headings, images, HTML lines, and list items are excluded. The
    sentence counter masks inline code, link targets, decimals, ellipses,
    and a small abbreviation list (`e.g.`, `i.e.`, `vs.`, `et al.`,
    `cf.`, `Fig.`, `no.` before a digit, …); semicolon chains count as
    one unit. Register judgment — the bullets-over-prose call and the
    FAIL decision — stays with the clean-result-critic (Lens 12); this
    check is the free mechanical backstop for the #1333/#385 incident
    class (dense 4–5-sentence paragraphs burning an LM critic round).
    (Numbered 36 because 28–35 are taken by the generation-agnostic
    checks, #1368.)

Generation-agnostic checks (v2 AND v3 AND v4): figure-URL-sha-matches
(check 22), HF-URL-resolves (check 23), figure-sidecar opaque
config-code tokens (check 28, WARN: `@L<digits>` layer pins + regime-code
slugs in figure-sidecar-carried text — plain-English condition names only;
slugs stay in the Repro config row; sidecar-carried strings, not full
PNG-pixel coverage), and figure prose-numerics vs sidecar plotted values
(check 33, WARN: every BOLDED decimal in a figure's what-is-plotted prose
window — the previous-figure-bounded beat-1 slice plus the caption — must
appear among the sidecar's plotted values under rounding / sign / percent
tolerance; per-figure opt-out for genuinely derived quantities: the literal
`<!-- prose-numerics: derived -->` anywhere in that window). The
Goal-of-experiment frontmatter
soft check + the Lens 14 concerns-audit run on v4 too (concerns mechanism
1 → `### ` results under `## Results` + `## Takeaways` bullets).

The v3-only Data checks (18/19/20/21 under the v3 numbering) and the v2
nested-structure check PASS-skip on a v4 body, exactly as they do on a
body of the other non-matching generation.

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
4. **Low-level data plot behind every aggregate statistic.** A finding
   that reports an AGGREGATE statistic — a correlation ρ shown as a
   forest-plot point, a mean / effect size shown as a bar, a p-value, an
   effect summary — ALSO embeds the LOW-LEVEL plot of the per-unit data
   behind it: the scatter the ρ summarizes, the strip / swarm / jittered
   per-point view behind the group-difference bars, the unbinned
   counterpart of a binned / aggregated view. The reader sees the data,
   not only the number computed from it. This is the broad PARENT of the
   raw-alongside-processed rule (point 5 below + clean-result-critic
   Lens 11): "show the underlying data" governs ANY aggregate, not only
   transformed scatters. The low-level plot rides inside the same
   `### <finding>` (data view alongside the summary view — data first
   where there's room, else clearly paired). Exemptions, stated in the
   read prose or alt text: the finding's primary figure ALREADY is the
   per-unit view (a raw scatter needs no second scatter); N is so small
   the figure already shows every point; or the aggregate has no
   meaningful per-unit decomposition (a single scalar with no underlying
   sample).
5. **Raw alongside processed** (the transformed-figure special case of
   point 4). When a finding's figure plots a residualized / partialled /
   binned / log-transformed / normalized / aggregated quantity, also
   embed its RAW (pre-processing) counterpart inside the same
   `### <finding>` (raw first, then processed), and quote the RAW point
   estimate alongside the controlled one in the prose. Same principle at
   the artifact layer: when a claim rests on an aggregated metric, link
   BOTH the aggregated file and the per-cell file the aggregation
   collapsed (see `## Data` / `## Reproducibility`). Exemption: raw and
   processed are visually identical (axis-rescale-only processing) — say
   so in alt text and omit the raw.
6. **For text-behavior findings only:** at most ONE short (≤10-line)
   raw-completion excerpt where the text itself IS the finding — preceded
   by a subset-disclosure line AND a raw-completions link. The systematic
   per-condition samples + `<details>` dropdowns live in `## Data →
   ### Generated`, not here.
7. **For runs that generate NO completions** (teacher-forced log-prob,
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
refusal-bait pools) AND real-world-corpus rollout text
(LMSYS/WildChat-class; #1073):** example blocks ship SANITIZED per `analyzer.md`
§ Content hygiene — labeled "sanitized for context hygiene", a ~15-word
excerpt plus a `[truncated — harmful-content row; verify at
<raw-completions path>, row <i>]` placeholder in place of the full
completion. The subset-disclosure line, row indices, and permanent links
stay verbatim. The mechanical checks (18/19) accept this form exactly as
the v2 finding-sample checks (10/11) do — the "sanitized for context
hygiene" LABEL is the load-bearing token the checks key on
(`verify_task_body.py` accepts the form via the class-agnostic
`sanitized for context hygiene|harmful-content row|truncated — harmful`
alternation), not the placeholder's exact noun, so a
`[truncated — sanitized row; …]` variant over real-world-corpus rows
passes identically. Agents assembling Data sections
pull rows by grep + line offset (context-hygiene rule) — never page whole
raw harmful-completion or real-world-corpus rollout files into context.

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
  Cross-issue provenance pins found in committed result-JSON `metadata`
  (`hf_rev_<M>_*` revision keys, `issue<M>_` input paths) must be
  declared in the body (canonically this list) — `verify_task_body.py`
  check 35 (`check_cross_issue_reuse_provenance`) FAILs an undeclared
  pinned revision at posting time.
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
    `is_nested_design()`); the check-17 origin-prompt verbatim sub-check
    (v4 list item 17) runs WARN-only here.
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
22. **Figure URL sha matches Reproducibility** (v2 AND v3,
    `check_figure_url_sha_matches_repro`): each inline figure URL's commit
    SHA must match the SHA the `## Reproducibility` `- Figures` bullet pins
    that figure to (`` `<basename>` at [commit] `<sha>` ``, with an
    `` all others at `<sha>` `` catch-all). A SHAPE-CONSISTENCY check — it
    compares the two SHAs the body already carries (no git, no network);
    SHAs compare prefix-compatibly (the claim is often abbreviated, the URL
    is full 40-char). The claim scan is SCOPED to the `- Figures` bullet so
    an incidental `` `main` at `<sha>` `` branch-merge note in the
    `**Context:**` bullet is never read as a figure claim (incident #480). A
    figure with NEITHER an explicit claim NOR a default is out of scope
    (SKIP, never FAIL). NO-OP PASS when there is no Reproducibility section,
    no inline figure URL, or no figure-sha claim. Incident: task #537
    `predictor_bakeoff_complete_null` shipped inline `5ad30c2…` against a
    Reproducibility claim of `c539920…`.

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

# The methodology document — template (v2/v3; SUPERSEDED for v4)

> **v4 note (current):** under the v4 spec the methodology doc is a
> mechanical COPY of the body's `## Methodology` section (see § "The
> standalone methodology doc (v4 — a mechanical COPY)" above), NOT a
> separately authored findings-blind doc. The template below describes the
> SUPERSEDED v2/v3 findings-blind methodology-writer output shape and is
> kept for the grandfathered v3/v2 generations. For v4 there is no
> separate authoring step — the doc IS the body's `## Methodology` section.

Every v2/v3 clean-result shipped with the auto-generated, findings-blind
methodology reference (`docs/methodology/issue_<N>.md` + secret gist
mirror, linked at the top of the body + from `## Reproducibility`).
Output shape was a fixed table-first skeleton:

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
- **`exemplars/`** — `v4-657.md` (canonical v4 exemplar; the reference
  for Rules A + B — self-contained `## Methodology` + research-paper
  register), `v3-517.md` (canonical v3 exemplar), `nested-432.md`
  (v2 section-level exemplar), `narrative-380.md` (legacy).

## Calling sites

- `.claude/agents/analyzer.md` — drafts the body per this spec (v4: drafts
  the full body incl. the detailed `## Methodology` section the doc is a
  copy of).
- `.claude/agents/clean-result-critic.md` +
  `codex-clean-result-critic.md` — critique against the lenses and run
  `verify_task_body.py` + `audit_clean_results_body_discipline.py`.
- `.claude/agents/methodology-writer.md` — **v2/v3 only** (DEPRECATED for
  v4): emitted the §2-complete findings-blind methodology doc. Under v4
  the doc is a mechanical EXPORT of the body's `## Methodology` section
  done by the `/issue` Step 9a-quater orchestrator, not by this agent.
- `.claude/skills/promote-clean-result/SKILL.md` — for legacy HTML
  bodies, optionally converts them to markdown on promotion.
- `CLAUDE.md` § "Experiment Report Structure" — points at this spec.

> **ALWAYS read this SPEC before changing ANYTHING about the report
> structure** — the CLAUDE.md summary, `verify_task_body.py`,
> `analyzer.md`, or any `clean-result-critic` lens. SPEC.md is the source
> of truth; these surfaces must stay in sync.
