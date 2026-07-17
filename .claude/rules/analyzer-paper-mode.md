---
description: "Analyzer PAPER-TASK MODE protocol (paper: true tasks) — Steps 4/5/6 paper variants, build_paper.py/verify_paper.py gating, paper-stub write, same-issue follow-up + after-submission handling; relocated verbatim from analyzer.md, #829"
paths:
  - "docs/papers/**"
  - "scripts/build_paper.py"
  - "scripts/verify_paper.py"
---
# PAPER-TASK MODE (author the LaTeX paper) — `paper: true`

When the task carries `paper: true` frontmatter, the canonical
clean-result is a self-contained LaTeX **research paper** under
`docs/papers/issue_<N>/`, and `body.md` is a thin **paper-stub**. You
author the paper; `body.md` Step 6 is replaced by the paper-stub write.
Read the spec FIRST: `.claude/skills/clean-results/SPEC.md` §
"Paper format (`paper: true`)" (the v1 scope, the Paper-sections mapping,
Rule A, the `body.md` paper-stub contract, the JSON schemas) — the SPEC is
the authoritative shape reference. The fixed shared template is
`docs/papers/_template/issue_TEMPLATE.tex` + `preamble.tex` (you NEVER edit
the preamble — you copy it into the per-task dir so `\input{preamble.tex}`
resolves); the template's commented `{{...}}` placeholder blocks document
the exact slots to fill. A worked spike paper
(`docs/papers/_spike/issue_657_spike.tex`) is NOT committed in v1 — it
exists only when a spike worktree was used — so do NOT treat it as a "read
this first" dependency; if it is present on disk it is a useful shape
reference, but note it is a SHORTENED demo using `\metric{}` (the v1.1
opt-in — in **v1** you write numbers as LITERALS and do NOT use `\metric{}`).

## The analysis is UNCHANGED — only the write-up form differs

Run **Steps 1 → 3.6 EXACTLY as in markdown mode** — they are the analysis,
not the write-up. The v4 honesty-protocol steps that SURVIVE and now author
INTO the paper instead of a markdown body:

- **Step 1 measurement-validity gate** (dynamic-range / floor-ceiling,
  proxy-vs-construct, **dual-DV** for content-behavior leakage/implants) —
  unchanged. The construct-accurate framing + the saturation tell now go
  into the paper's Methods (the DV definition) and Results/Discussion
  (interpretation), not a markdown `### <result>`.
- **Step 1.5 raw-output spot check** (5 random rows) + **Step 3.6
  per-condition sample selection** (≥3 firing / ≥3 non-firing) — unchanged;
  the systematic samples land in the paper's **Appendix** (authored by
  `methodology-writer`), and at most one short excerpt may appear in a
  Results subsection where the text IS the result.
- **Content firewall** (never page raw harmful-content or real-world-corpus
  (LMSYS/WildChat-class; #1073) completion files; checkpoint
  the fact-sheet every ~15-20 tool calls) — unchanged.
- **Step 2 statistics** (p-value, N, no effect sizes in prose) +
  **Step 3 / 3-bis plots** (`set_paper_style` — use `"neurips"` for the
  paper figures, not `"blog"`; the low-level data plot behind every
  aggregate; the raw-counterpart for every processed figure; the per-point
  `.meta.json` data sidecar) + **Step 3.5 plot-verification** — unchanged.
  Figures save under `figures/issue_<N>/` and are committed + SHA-pinned as
  in markdown mode; the paper references them via `\includegraphics` (the
  build sets `\graphicspath` to `figures/issue_<N>/`).
- **Numeric-fidelity re-extraction** (Step 3.6's HARD rule: every number
  re-extracted from the source eval JSON in the same turn you write it) —
  unchanged, and it is the **number-correctness guarantee in v1** (there is
  NO `\metric` grounding in v1, so this re-extraction IS the gate; write
  numbers as literals in the `.tex`, each copied from ground truth).
- **Step 4.5 humanize-loop** — run it on the paper's reader-facing prose
  (Abstract / Introduction / Results interpretation / Discussion), in
  **academic mode** (`/humanize academic` — em-dash zero-tolerance, copula
  avoidance, classical academic terms), not blog/quick mode.
  Ban-gate scoping applies identically (SKILL.md § 9a-humanize): the
  Appendix's verbatim worked examples / judge prompts are elided from the
  `check_bans.sh` scan input; a hit only inside them is a documented false
  positive — never rewrite sample data to satisfy the gate.
- **Step 6.5 follow-up tagging** + **Step 7 cross-link recap** + **Step 8
  tracking-file update** — unchanged (the `epm:analysis` marker, the
  `free_analysis_unrun:` field, the INDEX.md line all still apply).

## Step 4 (paper) — author the paper from the template

Author these sections (SPEC § Paper sections — required, in order, enforced
by `verify_paper.py`); assemble them with the **`methodology-writer`'s
Methods + Appendix** (the orchestrator spawns it in parallel — DO NOT write
the Methods or Appendix yourself; splice in what it returns):

1. **Abstract** — SELF-STANDING: a reader who has never seen another EPS
   experiment learns what was tested, on what model, and what was found.
   One or two sentences of project context. Maps to the v4 `## Takeaways`
   substance (numbers-first), in prose. **NO confidence words.**
2. **Introduction** — SELF-STANDING (readable without any other experiment
   open): project context, the question THIS experiment answers, the single
   variable it changes relative to its line. Maps to the v4 `## Goal`
   (`**This experiment in context:**` + `**Broader narrative:**`). Cite
   prior experiments with `\epsref{N}`, never a load-bearing dependency the
   reader must follow.
3. **Methods** — authored by `methodology-writer`, spliced in. (Do not
   write it; if the methodology-writer output has not arrived, request it
   via the orchestrator rather than authoring it yourself — the
   findings-blind firewall is the whole point.)
4. **Results** — one `\subsection` per finding, each in the **v4 three-beat**:
   state what is plotted (EXACTLY: axes, units, what each point/bar is, n,
   transform) → show the figure (`\includegraphics` from
   `figures/issue_<N>/`) → read the result. Numbers are **literals**,
   grounded by the numeric-fidelity re-extraction. Report the metric, its
   CI / n, and the test. The low-level-data-plot-behind-every-aggregate +
   raw-alongside-processed rules apply (embed both figures in the same
   subsection). **NO confidence words.** **Inline a verbatim `model-output`
   worked example for the load-bearing condition(s)** — the eval INPUT → the
   model's ACTUAL OUTPUT (verbatim) → the judge VERDICT/score, in an
   `\epsexample{...}` block preceded by `% eps-example: model-output`
   (subset inline; the comprehensive per-condition set goes in the Appendix
   the methodology-writer authors). `verify_paper.py` check 7 FAILs a paper
   missing the `model-output` example class; sanitize harmful/EM AND
   real-world-corpus (LMSYS/WildChat-class) rows per
   § content firewall (labeled excerpt + pinned raw path).
5. **Discussion + Limitations** — what the results mean, the alternatives,
   the binding caveats, what they change. Fold Limitations in here. **NO
   confidence words.** (A methodology correction folds into the relevant
   Results subsection's interpretation, not a discrete heading.)
6. **References** — from the per-task `issue_<N>.bib` (a copy of / subset of
   the project `.bib`), cited with natbib. Build the `.bib` with the
   `citation-management` skill.
7. **Appendix** — authored by `methodology-writer`, spliced in (the COMPLETE
   hyperparameter table + comprehensive worked examples — verbatim
   `training-data` / `eval-data` / `model-output` blocks — the full
   training-data recipe + full Rule-A reuse recipes, AND the mandatory
   **`\subsection{Judge prompts}`** carrying the verbatim prompt + rubric TEXT
   for every LLM judge in the study). The template ships the
   `% eps-judge-prompts` anchor + the `{{JUDGE_PROMPTS}}` placeholder for it;
   `verify_paper.py` check 8 FAILs a judge-using paper with no Judge-prompts
   section.

**Mechanics:**
- Copy `docs/papers/_template/issue_TEMPLATE.tex` →
  `docs/papers/issue_<N>/issue_<N>.tex` and `preamble.tex` into the same
  dir (the build runs in-place; it does NOT rewrite the `\input` path). Fill
  the `{{...}}` placeholders (TITLE, ISSUE, RUN_DATE, MODEL, GRAPHICSPATH,
  ABSTRACT, INTRODUCTION, METHODS, RESULTS, DISCUSSION, APPENDIX,
  JUDGE_PROMPTS — the last carries the verbatim judge prompts/rubrics; the
  methodology-writer supplies its content with the Appendix).
- **Show the data, not just the method (`verify_paper.py` checks 7-9; 7 examples
  present, 8 judge prompts, 9 example-provenance pointers).** The
  paper MUST carry verbatim text: real TRAINING rows
  (`% eps-example: training-data`), real EVAL probes
  (`% eps-example: eval-data`), real MODEL OUTPUTS with judge verdicts
  (`% eps-example: model-output`) — each in an `\epsexample{...}` block with a
  subset-disclosure + pinned-artifact caption — and the verbatim JUDGE PROMPTS
  for every judge. The methodology-writer authors the Methods/Appendix example
  blocks + the Judge-prompts subsection; you author the Results `model-output`
  worked example (item 4). Pull every block from a REAL artifact — never
  fabricate.
- **NO confidence anywhere in the `.tex` body** — the
  `(LOW|MODERATE|HIGH confidence)` tag and bare `Confidence:` lines are a
  hard `verify_paper.py` FAIL. Confidence lives ONLY in the `body.md`
  paper-stub frontmatter (so the title-tag / dashboard machinery keeps
  reading it). This replaces the markdown rule "confidence in the H1 title
  tag" — for a paper-task the confidence is in the STUB frontmatter, and the
  paper body has none.
- **`\epsref{N}` for every cross-experiment reference** (v1 feature) — never
  a bare "#N". Emit `refs.json` (SPEC § JSON schemas — top level
  `{ "schema": "refs/v1", "epsrefs": [ … ] }`, where the `epsrefs` array
  is the `\epsref` target index — one `{ "issue": N, "context": "<one line>" }`
  per `\epsref` you + the methodology-writer cited) alongside the paper.
  Emit the **figures manifest** the build expects (every figure the paper
  `\includegraphics`, committed under `figures/issue_<N>/` and SHA-pinned).
  Collect the `\epsref` targets the methodology-writer reports in its return
  so `refs.json` is complete.

## Step 5 (paper) — build + verify

Build on the VM (the single pinned-TeX-Live host), then verify. Both run
from repo root with absolute `$REPO_ROOT/scripts/...` paths:

```bash
# build: multi-pass pdflatex + bibtex -> reproducible PDF -> HF PDF upload ->
# sanitized paper.html -> paper_manifest.json
uv run python "$REPO_ROOT"/scripts/build_paper.py --issue <N>
# (--no-upload for a local-only build during iteration; pdf_hf_url is then a
#  WARN, not a FAIL)

# verify: compile-clean log parse, required sections + order, no confidence
# in body, \includegraphics confined + resolve, .bib entries resolve,
# \epsref{N} resolves to a real task, manifest complete + hashes match,
# paper-stub valid
uv run python "$REPO_ROOT"/scripts/verify_paper.py --issue <N>
```

`verify_paper.py` is the gate for paper-tasks (the paper-format counterpart
of `verify_task_body.py`, which stays the verifier for markdown bodies). A
LONG `pdflatex` / `bibtex` / pandoc run can take minutes — use the Step 2
sentinel + bg-`until` polling pattern if it backgrounds.

## Step 6 (paper) — write the `body.md` paper-stub ONLY after verify PASS

**The paper-stub is written ONLY after `verify_paper.py` PASSes.** This is
the terminal write that replaces markdown Step 6's body promotion:

- **On `verify_paper.py` PASS:** snapshot the existing body
  (`task.py set-body <N> --file <stub> --snapshot` — preserves the original
  ask), then `set-title` + `set-clean-result`. The stub carries:
  `paper: true` frontmatter + the fields the existing machinery reads
  (`title`, `kind`, `goal`, `has_clean_result`, the **confidence** as the
  title's `(LOW|MODERATE|HIGH confidence)` tag — confidence lives HERE, never
  in the paper body — and origin/lineage); a body of the H1 title + the
  abstract (so the dashboard hover-card + REGISTRY title/abstract
  denormalization work) + a paper link (the `docs/papers/issue_<N>/`
  artifacts and/or the pinned HF PDF URL from `paper_manifest.json`).
  `verify_paper.py`'s paper-stub check enforces `paper: true` + an H1 + an
  abstract + a paper link — run it on the stub before `set-body`.
- **On `verify_paper.py` FAIL:** do NOT write a stub over the body, do NOT
  flip `has_clean_result=true`. Leave the `.tex` + the build `.log` /`.blg`
  in `docs/papers/issue_<N>/` for iteration and park the task at
  `reviewing` (or `blocked` with `epm:failure v1 failure_class: code` +
  the FAIL reason if you cannot resolve it autonomously). NEVER leave a
  stub-only dead state (a stub written over a paper that does not compile /
  verify) — the same failure class as markdown mode's #385 placeholder-body
  incident. Re-run build + verify after fixing, then write the stub on PASS.

Confidence tag discipline: the title's `(LOW|MODERATE|HIGH confidence)`
suffix lives in the STUB title + frontmatter ONLY. The paper `.tex` carries
NO confidence — same honesty bar, different surface.

## Same-issue follow-up rounds (paper-task)

When the task carries an `epm:followup-scope v1` marker and you are
re-spawned after a same-issue follow-up run, the clean-result is ALREADY a
paper. **Re-author the `.tex` IN PLACE** (do NOT create a second paper, do
NOT migrate to markdown):

1. Fold the new round's result(s) into the existing Results section (new
   `\subsection`s, or rewrite an invalidated subsection).
2. **Rewrite the Abstract to the current cross-round synthesis** — same
   mandatory rolling-synthesis rule as markdown `## Takeaways` (an Abstract
   describing only round 1 after round 2 landed is wrong). Retitle the paper
   + the stub title (claim + confidence tag) if the headline moved.
3. **Add a changelog appendix** (`\subsection{Changelog}` /
   `\paragraph{Round <label>.}` inside the Appendix) preserving the
   superseded-round notes — this is the paper-task form of the v4 audit
   trail (superseded-result collapse + the round notes). Keep the prior
   rounds' superseded findings recoverable, not deleted.
4. Re-spawn `methodology-writer` in EXTEND mode for the new arm's Methods +
   Appendix recipe; splice its additions in.
5. Re-run `build_paper.py` + `verify_paper.py`; write the stub (without
   `--snapshot` — the original is already preserved) only on PASS — and if
   step 2 retitled the stub, follow the stub write with
   `task.py set-title <N> "<new stub H1>"` — pass the stub's H1 line minus
   the leading `# ` (same argument spec as the markdown re-fold,
   analyzer-section-reference.md § Same-issue follow-up re-entry) —
   (set-body preserves the old frontmatter `title`, so the
   REGISTRY/dashboard title would keep the stale headline — the same
   pairing the main paper Step 6 sequence already prescribes).

## After submission (paper-task)

The `clean-result-critic` (+ Codex twin) reviews the paper as the
clean-result. On PASS the `/issue` skill parks at `awaiting_promotion`
(user-only promotion — never run `task.py promote` yourself). On a non-PASS
verdict, revise the `.tex` in place, re-build + re-verify, and re-write the
stub. The markdown-mode rule "no `epm:interpretation` before upload PASS"
(HOLD-marker mode) applies identically — your analysis interpretation is
held until upload-verification PASS regardless of write-up form.
