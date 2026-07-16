---
description: "Clean-result-critic paper-task review protocol (paper: true) — the seven P1-P7 paper lenses, verify_paper.py pre-pass, .tex/PNG/PDF read targets, and the paper-lens output template; relocated verbatim from clean-result-critic.md, #829. Read by BOTH twins (codex-clean-result-critic composer inlines it at runtime)."
paths:
  - "docs/papers/**"
  - "scripts/verify_paper.py"
---
# Clean-result paper-task review (relocated from clean-result-critic.md, #829)

## Paper-task review (`paper: true`)

**This section applies ONLY when the task `body.md` frontmatter carries
`paper: true`.** The canonical clean-result is a self-contained LaTeX
research paper at `docs/papers/issue_<N>/` (the markdown `body.md` is a
thin paper-stub). You review the PAPER, not a markdown body. The fifteen
markdown lenses do NOT apply; the seven paper lenses (P1-P7) below do. The
canonical spec is `.claude/skills/clean-results/SPEC.md` § "Paper format
(`paper: true`)" — read it before scoring.

You are still NOT a numbers-reviewer (interpretation-critic already
checked content honesty + plot-prose match). You check **paper shape,
self-containedness, completeness, register, no-confidence, and
cross-reference correctness**.

### Paper mechanical pre-pass (mandatory)

```bash
# Resolve the canonical MAIN checkout root (worktree-proof) — same rule as
# the markdown pre-pass: from a worktree cwd the bare relative `scripts/...`
# resolves against a possibly-stale worktree fork. NEVER
# `git rev-parse --show-toplevel` (it returns the worktree root). (#537.)
TASK_DIR="$(uv run python scripts/task.py find <N>)"   # absolute, canonical main
REPO_ROOT="${TASK_DIR%/tasks/*}"                        # canonical MAIN checkout root
# The PAPER verifier (NOT verify_task_body.py — that is markdown-only).
uv run python "$REPO_ROOT/scripts/verify_paper.py" --issue <N>
```

`verify_paper.py` is the paper-format counterpart of `verify_task_body.py`.
Its v1 check catalog (the authoritative list lives in the script docstring):
(1) compile-clean multi-pass (`.log`/`.blg` parse); (2) required sections
present + in order — Abstract, Introduction, Methods, Results, Discussion,
References (`\bibliography`), Appendix; (3) NO confidence anywhere in the
paper body (the `(LOW|MODERATE|HIGH confidence)` tag + bare `Confidence:`
lines are a hard FAIL); (4) `\includegraphics` paths repo-relative-confined
+ each resolves on disk; (5) `.bib` entries resolve for every `\cite`;
(6) `\epsref{N}` resolves to a real task in the registry; (7) verbatim
examples present — the three required example classes (`training-data`,
`eval-data`, `model-output`) each declared with a `% eps-example: <class>`
marker AND real verbatim example environments behind them; (8) judge prompts
present — a `Judge prompts` / `Judge rubric` appendix (sub)section when the
paper uses any LLM judge; (9) `paper_manifest.json` complete + sha256 hashes
match; (10) the `body.md` paper-stub is valid (`paper: true` + an H1 + an
abstract + a paper link).
There is **NO `\metric` grounding check in v1** — numbers are written as
literals, and the interpretation-critic's numeric-fidelity re-extraction is
the number-correctness guarantee. (`\metric` grounding is a documented
**v1.1 opt-in** under `docs/papers/_template/`; do NOT FAIL a v1 paper for
writing numbers as literals or for not carrying a `metrics.json`.)

**Codex-twin adaptation (#1050).** When the `codex-clean-result-critic`
composer inlines this rule into the Codex prompt, it REPLACES the bash
block above with the composer-run `verify_paper.py` output envelope (the
MECHANICAL VERIFIER OUTPUT envelope of that agent's Step 1d / Step 3
shape) — that twin is dispatched read-only and uv cannot reliably execute
in its sandbox, so Codex READS the inlined output instead of running the
pre-pass itself. The Claude critic continues to run the pre-pass verbatim
as above.

Run it (Claude critic — the Codex twin's composer already ran it at compose
time per the adaptation note above; Codex reads the inlined envelope and
never re-runs it), record the result, and ALWAYS proceed to the seven paper
lenses in the SAME pass — never hard-stop at a mechanical FAIL. Split the
FAILs:

- **Structural / data-integrity FAILs (genuinely block):** a required
  section missing or out of order (check 2), confidence in the body
  (check 3), a `\includegraphics` that escapes the repo or does not resolve
  (check 4), an unresolved `\cite` (check 5), an `\epsref{N}` to a
  non-existent task (check 6), a missing verbatim example class / no example
  block (check 7), a judge-using paper with no Judge-prompts section
  (check 8), a manifest hash mismatch / missing artifact (check 9), an
  invalid paper-stub (check 10), or a compile that is not clean (check 1 —
  undefined refs/citations, package errors, a missing `.bbl`). Record as a
  blocking finding but STILL score all seven lenses.
- **Presentation-only FAILs (procedural — do NOT block alone):** none of
  the v1 paper checks are purely cosmetic, so in practice every
  `verify_paper.py` FAIL is structural. If a future check is added that is
  presentation-only, treat it as the markdown branch treats its
  presentation FAILs — a `### Procedural fixes` bullet, never the sole
  basis for a non-PASS.

A non-PASS verdict MUST be backed by ≥1 substantive finding — a structural
verifier FAIL or a real P1-P7 lens violation. (No `audit_clean_results_
body_discipline.py` on a paper task — it audits markdown bodies.)

### Read these before scoring

Load the actual paper artifacts (all under `docs/papers/issue_<N>/`):

- **`issue_<N>.tex`** (text via Read/Bash) — the canonical source; the
  claims, sections, `\epsref{N}` calls, `\cite` keys, and Appendix live
  here.
- **The figure PNGs** referenced by `\includegraphics` (under
  `figures/issue_<N>/`) — load them via the Read tool for the P-lens figure
  checks, same as the markdown Lens 3 + interpretation-critic Lens 6.
  Figure read-target rule (paper-mode analogue of the markdown #922
  EXCEPTION, whose Lens-3 pointer does not ship in paper mode): the compiled
  PDF is the built artifact of record — on a working-tree-PNG vs PDF-page
  disagreement, review against the PDF page, note the possible stale
  working-tree stray, and never rest a blocker on the PNG alone.
- **`issue_<N>.pdf`** (the compiled PDF) — load relevant pages via the Read
  tool's `pages` parameter to catch RENDER-ONLY issues the `.tex` text
  hides: a float that landed pages away from its reference, a table that
  overflows the margin, a figure rendered at an unreadable size, a caption
  truncated in the rendered output, an equation that didn't typeset. The
  `.tex` text alone cannot surface these; the compiled PDF is the reader's
  actual artifact.

### The seven paper lenses (v1)

For each lens: state PASS / FAIL with one concrete sentence explaining WHY.
If FAIL, quote the offending passage (cite the `.tex` line / section / the
figure file / the PDF page).

#### Lens P1 — Self-standing Introduction

The Introduction is readable WITHOUT any other EPS experiment open. A
reader who has never seen another EPS paper learns: the project context (in
one or two sentences), the question THIS experiment answers, and the single
variable it changes relative to its line. Prior experiments are referenced
with `\epsref{N}`, but the paper never DEPENDS on the reader following one
to understand it. **FAIL** when the Introduction assumes undefined
prior-experiment context ("continuing the #532 line", "as established
previously") such that a cold reader cannot follow, or when it omits the
project context / the question / the single-variable framing. (Maps to the
v4 `## Goal` two-part contextualization, in prose.) (SPEC.md § Paper
sections item 2.)

#### Lens P2 — Self-contained Methods + the Rule-A reuse-chain depth rule

The Methods section is SELF-CONTAINED: everything needed to reproduce is
written out. A reader understands HOW every reported result was produced
without following a link to another issue.

- **Rule A (no-deferral for DIRECT reused artifacts).** For an artifact
  THIS experiment directly reuses (a trained adapter, training mix, persona
  vectors, behavior direction, leakage cells, eval JSON, base-rate /
  propensity measurement), the Methods section writes out its FULL
  generation recipe inline as primary method — data source + realism tier,
  construction recipe, training recipe + hyperparameters, measurement —
  exactly as if performed for this experiment. The reuse MAY be
  acknowledged with an `\epsref{N}` link, but **FAIL** when a load-bearing
  method is DEFERRED to another issue: "reused from #N; see there", "as in
  \epsref{N}", "same setup as #N" standing IN PLACE OF the actual recipe.
- **Transitive inputs (an input to the thing you reused) — the reuse-chain
  depth rule.** For an input to the artifact you reused (depth-1), give a
  COMPACT recipe to depth-1, then cite + one-line-summarize deeper links.
  You do not write out the entire ancestral chain — depth-1 compact, deeper
  cited. **FAIL** when a depth-1 input is deferred with no recipe at all, OR
  (the over-correction) when the Methods balloons the full multi-generation
  ancestral recipe inline past depth-1 (that is a conciseness / register
  problem — flag under P5, not here, but note the depth rule was missed).
- **Completeness of the in-body Methods.** The Methods carries the
  load-bearing hyperparameters inline (the COMPLETE table lives in the
  Appendix per P3). Every value is from ground truth (committed config /
  `run_result.json` / plan §11), never typed from memory — eyeball the
  learning rate + the load-bearing knobs against the plan / committed code
  at the paper's code SHA (the #489 `lr = 1e-4`-vs-`2e-6` class of defect is
  a data-integrity FAIL, not cosmetic).

(Maps to the v4 `## Methodology` Rule A + the complete-hyperparameter-table
rule. SPEC.md § Paper sections item 3 + § "Paper format" Rule A.)

#### Lens P3 — Inline-subset + comprehensive-Appendix completeness

The v1 paper deliberately splits detail across the body and the Appendix —
the body inlines a SUBSET, the Appendix carries the FULL set. Check BOTH:

- **Body inlines the load-bearing subset:** 2-3 worked example completions
  (eval input → model output → judge score), the load-bearing
  hyperparameters, representative training rows — each subset-disclosed
  ("2 of 200 rows, random sample" / "cherry-picked for illustration") and
  linked to the complete artifact (pinned HF `/tree/<sha>` or GitHub
  `/blob/<sha>`).
- **Appendix is COMPREHENSIVE:** comprehensive example completions (the
  full per-condition set), the full training-data construction recipe +
  representative rows, the COMPLETE hyperparameter table (every training +
  eval + generation knob, with a Source column), and the full Rule-A reuse
  recipes.

**FAIL** when: the body omits the worked-example subset for a
text-generation paper; an example block has no subset-disclosure line or no
pinned full-artifact link; the Appendix is missing (verifier check 2 also
FAILs) or is obviously incomplete (a load-bearing knob the body or plan
names is absent from the complete table); or the COMPLETE hyperparameter
table is in the body instead of the Appendix (the body carries only the
load-bearing subset). **Harmful-content carve-out (covers harmful-content
corpora AND real-world-corpus rollout text (LMSYS/WildChat-class; #1073)):**
example blocks
labeled "sanitized for context hygiene" (~15-word excerpts + a
`[truncated — harmful-content row; verify at <path>, row <i>]` placeholder,
cherry-picked labels + row indices + permanent raw links kept verbatim)
SATISFY the worked-example requirement — do NOT FAIL them as missing
verbatim samples, and never load raw harmful-content or real-world-corpus
rows into context.
(Maps to the v4 Methodology Sample slot + Lens 10 subset disclosure.
SPEC.md § Paper sections item 7 (Appendix).)

#### Lens P4 — No confidence anywhere in the paper body

The `(LOW|MODERATE|HIGH confidence)` tag and bare `Confidence:` lines are a
hard FAIL inside the `.tex` (verifier check 3 also catches this; this lens
is the semantic read on top — catch a confidence CLAIM phrased without the
literal token, e.g. "we are highly confident that", "this result is only
suggestive", "tentatively"). Confidence lives ONLY in the `body.md`
paper-stub frontmatter (the title's `(... confidence)` tag), so the
existing title-tag / dashboard machinery keeps reading it. The paper's
Abstract / Discussion state the finding and its limitations WITHOUT a
confidence label. **FAIL** on any confidence word / tag in the paper body.
(SPEC.md § Paper format "NO confidence anywhere in the paper body".)

#### Lens P5 — Research-paper register

The paper reads in the concise, precise register of a research paper:
declarative methods/results prose, every quantity defined on first use, no
filler / marketing / hype, no AI-slop vocabulary, no `byte identical` /
`byte-identical` (use "the two files matched exactly"). The Abstract is
self-standing and numbers-first in prose. No project-internal opaque
condition / config codes (`sw_eng_C1`, `cond_4`, `M1`, `Bin C`,
`<letter>-family`) in the reader-facing prose / figure labels / captions —
use plain-English condition names; bare codes survive only in verbatim
example blocks / launch-command listings / the Appendix config rows.
First-person `I` (the project's voice convention) where a voice is used.
**FAIL** on slop vocabulary, undefined jargon, opaque codes in
reader-facing prose or on a figure, or a Discussion that editorializes
instead of reading the result. (Maps to the v4 Voice lens (Rule B) +
statistical-framing + plain-English-labels rules.)

#### Lens P6 — `\epsref{N}` correctness

Every cross-experiment reference uses the typed `\epsref{N}` macro, NEVER a
bare "#N" / "issue 532" / "task #532" in the prose (the dashboard
hover-preview needs the typed macro; verifier check 6 confirms each
`\epsref{N}` resolves to a real task). **FAIL** when a prior experiment is
referenced by a bare "#N" / "issue N" string instead of `\epsref{N}`, or
when an `\epsref{N}` points at a task that does not exist (verifier check 6
also catches the latter). **Forward-only fallback understood:** an
`\epsref{N}` to a task that is in the registry but whose paper has not been
built yet is fine — the macro resolves against the task registry, not a
built paper; do NOT FAIL a forward reference to an existing, not-yet-papered
task. (SPEC.md § "Paper format" v1 SCOPE — `\epsref{N}` is a v1 feature.)

#### Lens P7 — Verbatim examples + judge prompts (show ALL methods AND examples)

A research paper SHOWS its data, not just its method. The paper MUST carry
VERBATIM TEXT pulled from real artifacts — not prose describing it (incident
#657: the paper described every method but shipped zero verbatim text and no
judge prompts). Check ALL of:

- **Training-data examples** — ≥2 verbatim sample training rows (the ACTUAL
  row text: system/persona prompt + question + completion, incl. a
  contrastive-negative row where applicable). For a reuse-only study, real
  rows from the REUSED training mixes. Inline a representative subset; the
  Appendix carries the comprehensive set (or a pinned link to the full file +
  a larger appendix sample).
- **Full system prompts, word-for-word.** Every example involving a persona /
  system prompt quotes the COMPLETE system prompt string verbatim (copied from
  the persona definition / the chat-templated row), with the SYSTEM / USER /
  ASSISTANT turns each labeled + verbatim. A prose paraphrase (`system = "you
  are a doctor"`), a reworded prompt, or a system/user turn truncated with
  `...` is a FAIL — system + user turns are short and load-bearing for
  reproduction and are NEVER truncated (only a long model OUTPUT may be elided
  with an explicit `[...]` when the full text is in the Appendix / at the raw
  path).
- **Eval-data examples** — ≥2 verbatim eval inputs/probes (the actual
  false-claim, the harmful/harmless prompt, the steering probe). Inline
  subset + Appendix.
- **Model-output / completion examples** — verbatim WORKED examples per
  load-bearing condition: eval INPUT → the model's ACTUAL OUTPUT (verbatim) →
  the judge VERDICT/score. Inline subset (Results) + Appendix comprehensive.
- **Judge prompts / rubrics** — when ANY LLM judge scores a behavior, the
  ACTUAL prompt + rubric TEXT for EVERY judge, verbatim, in a dedicated
  `Judge prompts` / `Judge rubric` appendix (sub)section (e.g. the
  steering-sanity rubric, the sycophancy-agreement judge, the EM judge, the
  refusal judge).

Provenance + no invention: every block traces to a REAL artifact (HF
`raw_completions`, training JSONLs, probe banks, the judge rubric file/code) —
never fabricated. Each block's caption MUST carry a resolvable provenance
pointer (an `\epsref{N}`, an `issueN_` slug, a `superkaiba1/` HF path,
`eval_results/` / `figures/`, a `.json(l)` file, or a recognized HF dataset id);
`verify_paper.py` check 9 enforces a pointer is PRESENT. **Harmful-content
carve-out (covers harmful-content corpora AND real-world-corpus rollout text
(LMSYS/WildChat-class; #1073)):** example blocks labeled "sanitized for
context hygiene" (~15-word
excerpts + a `[truncated — harmful-content row; verify at <path>, row <i>]`
placeholder, cherry-picked labels + row indices + permanent raw links kept
verbatim) SATISFY the requirement — do NOT FAIL them as missing verbatim
samples, and never load raw harmful-content or real-world-corpus rows into
context.

**The deep no-invention reality-check (open the artifact, confirm the persona /
prompt / completion are real + byte-for-byte) is the `interpretation-critic`'s
paper-mode Lens 7, which runs BEFORE you.** You enforce the structural tells a
reader can see without opening the artifact: a `system = "..."` prose summary, a
`...`-truncated system/user turn, a missing provenance pointer, or an example
whose persona name you can cheaply confirm is absent from
`data/canonical_persona_pool/pool_v1.json` / the experiment's persona dict.
(Motivating incident #657: a fabricated "young child who is curious about the
world" persona that does not exist in the data — caught by the reality-check.)

**FAIL** when any of the three example classes is absent (verifier check 7
also catches a missing `% eps-example:` class marker / no example block), when
an example is paraphrased prose rather than a verbatim block, when a system /
user turn is paraphrased or `...`-truncated, when a block has no
subset-disclosure / no pinned full-artifact link / no provenance pointer
(verifier check 9), when a persona named in an example is absent from the
persona pool / the experiment's realized set, or when the paper uses an LLM
judge but ships no `Judge prompts` appendix section (verifier check 8). A
genuine no-judge study (pure log-prob / logit) passes the judge-prompt half
automatically. (Maps to SPEC.md § Paper sections items 3/7 + § "No invention" +
the `verify_paper.py` checks 7-9.)

### Paper-lens output

Post the SAME `epm:clean-result-critique` marker as the markdown branch
(see ## Output below), but score the SEVEN paper lenses (P1-P7) in place of
the fifteen markdown lenses, and report the verifier as `verify_paper.py`
(NOT `verify_task_body.py`):

```
Round <K>: PASS|FAIL — <one-sentence summary>.
Blocker tags: [comma-separated, non-PASS only: `structural-absence` (a
verify_paper.py structural/data-integrity FAIL — checks 1-11), `lens` (a real
P1-P7 violation). `none` on PASS.]
Mechanical pre-pass: verify_paper.py PASS|FAIL — <one-line summary>.
Paper lens findings:
- Lens P1 (Self-standing Introduction): PASS|FAIL — ...
- Lens P2 (Self-contained Methods + reuse-chain depth): PASS|FAIL — ...
- Lens P3 (Inline-subset + comprehensive-Appendix completeness): PASS|FAIL — ...
- Lens P4 (No confidence in the paper body): PASS|FAIL — ...
- Lens P5 (Research-paper register): PASS|FAIL — ...
- Lens P6 (`\epsref{N}` correctness): PASS|FAIL — ...
- Lens P7 (Verbatim examples + judge prompts): PASS|FAIL — ...

<If FAIL: minimal-necessary-fix list, one bullet per issue — each bullet
quotes/names its `.tex` line / section / figure / PDF page and ends with
`mechanizable: yes|no` (+ a 1-2 line check sketch when yes), per the
standing Blocker-grounding rule below.>
```

The round budget, independence rules, and Blocker-grounding +
mechanizability standing rule (below) apply to the paper branch identically.
(**v1.1 note:** a `\metric` grounding lens — number → `metrics.json`
pointer → `eval_results` JSON — is a planned v1.1 addition, NOT scored in
v1; do not add it to the roster or FAIL a v1 paper for writing literal
numbers.)
