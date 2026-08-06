---
paths:
  - ".claude/rules/methodology-writer-section-reference.md"
description: >
  Per-mode authoring bodies and template/recipe detail for
  methodology-writer.md — the markdown doc template + sections 1-6, the
  worked-example and hyperparameter-table rules, output workflow, EXTEND
  mode, and the full PAPER-TASK and REPORT mode bodies. Loaded ONLY via the
  explicit pointers in methodology-writer.md; the self-matching `paths:`
  glob keeps it out of every other agent context. The findings-blind
  firewall deliberately stays in the spec, not here.
---

# Methodology-writer section reference (per-mode authoring bodies)

Relocated verbatim from `.claude/agents/methodology-writer.md` (the spec is a
per-spawn system-prompt cost; #1090/#2054 autocompact deaths are
fixed-overhead deaths). The OPERATIVE trigger / contract for every item
stays in the spec; this file carries the extended recipes and templates.
Read ONLY the section you need: grep the heading, then a chunked `Read`
of that span — never the whole file.

# Methodology — issue <N>: <one-line what-was-run, no findings>

A methodology + hyperparameter reference for experiment #<N> (Explore
Persona Space), with verbatim training / evaluation / output examples
pulled straight from the artifacts. No interpretation.

- Task: [https://eps.superkaiba.com/tasks/<N>](https://eps.superkaiba.com/tasks/<N>)
- Model: `<base model id, exact string>`

---

## 1. Overview

3–5 bullets, no prose paragraphs: model · the manipulation (the single
variable) · design cells / panels / arms · the dependent variable ·
the judge (model + what it scores). Provenance notes belong here too
("SP01–SP05 reused verbatim from #406's persona anchors A1–A5").

---

## 2. Hyperparameters

ONE complete table — EVERY training + eval + generation hyperparameter,
each value copied verbatim from ground truth (committed config /
run_result.json / plan §11), with a **Source** column. Nothing scattered
in prose. Bold the load-bearing knobs (LoRA rank/alpha, learning rate,
epochs, seed, rows-per-adapter).

| Parameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | run_result.json |
| **LoRA rank / alpha** | ... | train script @ <sha> |
| **Learning rate** | ... | plan §11 |
| ... (epochs, batch, grad-accum, max-length, seed, marker token id, max_new_tokens, temperature, judge model + re-calls, N per cell, ...) | ... | ... |

This is the canonical COMPLETE table; the body `## Reproducibility`
Parameters table is a SUBSET of it (verifier check 21 asserts the
subset relation — that assert is the VERIFIER's job, not yours; you just
emit the complete table).

**Analysis-only / no-training tasks** (a `kind: analysis` task, or a
zero-GPU `kind: experiment` that trains no model — e.g. a meta-analysis
over prior issues' artifacts): there are no training/generation
hyperparameters to table, so write §2 as a single line:

```markdown
## 2. Hyperparameters

**N/A — no model training.** The load-bearing analysis constants
(bootstrap B, spline knots, logit ε, thresholds, …) live in §4
Evaluation.
```

Then put the analysis-design constants in §4 Evaluation as a
`Constant | Value | Source` table alongside the DV definition — do NOT
improvise a different §2 name (`## 2. Training recipe` etc.) or scatter them
across prose. `verify_task_body.py` check 21 PASS-skips the body-Parameters
⊆ doc-§2 subset assertion here — its `_methodology_doc_has_no_training_recipe`
helper keys on the exact `N/A — no model training` marker (commit
`639b96239b`), so keep that phrasing verbatim or the body's analysis-design
Parameters false-FAIL as a non-subset.

---

## 3. Training data

Construction recipe as a numbered list (≤8 steps). Then a row-count /
composition table (rows per type, positives:negatives ratio, persona
panel, completion provenance tier per
`.claude/rules/on-policy-completions.md`). Then 2–3 VERBATIM example
rows (input → output, loss-mask noted), labeled cherry-picked /
fixed-seed-sample, with a permanent HF `/tree/<sha>` link to the full
data.

**Analysis-only / no-training tasks:** write this section as
`**N/A — no training mix.**` (the task trained nothing) and describe the
input artifacts it analyzed in §4 Evaluation / §6 Artifacts index
instead.

| Row type | N | Personas | Provenance |
|---|---|---|---|
| ... | ... | ... | ... |

---

## 4. Evaluation

DV definition (construct + metric + on/off-policy choice, 2–3 bullets).
Probe-set table (N, source, why chosen, preprocessing). 2–3 verbatim
example probes. Judge prompt / rubric pointer.

| Probe set | N | Source | Why chosen |
|---|---|---|---|
| ... | ... | ... | ... |

---

## 5. Worked examples

2–3 verbatim end-to-end rows (eval input → model output → judge
score/measurement), one per load-bearing condition. Label each
cherry-picked; permanent raw-completions `/tree/<sha>` link.

---

## 6. Artifacts index

| Artifact | Pinned link |
|---|---|
| Training JSONL | [HF Hub](<permanent /tree/<sha> URL>) |
| Model checkpoints / adapters | [HF Hub](<permanent /tree/<sha> URL>) |
| Raw completions | [HF Hub](<permanent /tree/<sha> URL>) |
| Eval results JSON | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/<sha>/<path>) |
| Training script | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/<sha>/<path>) |
| Eval script | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/<sha>/<path>) |
| Hydra config | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/<sha>/<path>) |
| WandB run(s) | [<run-name>](<url>) |
| Code commit | `<full 40-char SHA>` |
| Compute | <wall time, GPU type × count, pod label> |

---

*This document describes how the experiment was run. For the result and
what it means, see the [task body](https://eps.superkaiba.com/tasks/<N>).*
```

**Caps (your checklist; spot-checked by clean-result-critic Lens 10):** no
section has a prose paragraph >2 sentences; everything table-able IS a table /
numbered list; target ≤150 lines EXCLUDING verbatim example blocks;
findings-blind (no interpretation / confidence / results). §5 worked examples
may merge with §3/§4 rows for a simple experiment or split per load-bearing
condition; §2 stays ONE table no matter how many conditions.

## Worked-example data rules

- **Read the actual artifact files** for verbatim quotes — never invent or paraphrase a row. Read the JSONL at `eval_results/issue_<N>/.../foo.jsonl` (or `git show <sha>:<path>` if removed locally); for HF-Hub raw completions, `HfApi().file_exists(...)` / scoped `list_repo_tree(path_in_repo=...)` to confirm the path (bare data-repo `list_repo_files` times out — gotchas.md), then read the row.
- **Cherry-picked is fine** (illustrations, not aggregates) — label the disclosure inside the block: `<!-- cherry-picked for illustration; full data at <HF Hub link> -->`, or use a deterministic sample (`random.seed(42)` + `random.choice`).
- **Truncate long completions** with `...` + a "tail" hint (`"...a more effective and empathetic listener. ※"`) — presentation, not a finding. **Preserve formatting** (valid JSON; token strings carry their leading space, `" ※"` not `"※"`).
- **Harmful-content sources ship SANITIZED — corpora AND banks.** When a worked
  example's source is a harmful-content corpus (EM, refusal, harmful-advice) OR
  a harmful safety-benchmark question bank
  (`src/explore_persona_space/artifacts/query_banks/*.json` — advbench,
  strongreject, Betley-lineage, sensitive-info; #866) OR real-world-corpus
  rollout text (LMSYS/WildChat-class; #1073), ship a ≤15-word excerpt +
  a `[truncated — harmful-content row; verify at <path>, row <i>]` placeholder,
  pulled by grep + line offset / `jq` index — never page the whole file into
  context; reference bank items by filename + index. Benign banks (`arc_c_v1`,
  `fact_questions_v1`, `marker_eval_v1`, `sycophancy_claims_v1`,
  `wildchat_random_v1` (toxic/redacted-screened at build)) keep verbatim
  treatment; when unsure, sanitize.

## Hyperparameter table rules

The hyperparameter table is the most failure-prone piece. Apply the same discipline `analyzer.md` Step 4 applies to the clean-result Parameters table:

- Open the training script at the body's `**Code:**` SHA via `git show <sha>:<path>` and read off `--lr`, `--epochs`, `--rank`, `--alpha`, `--dropout`, `--batch-size`, `--grad-accum`, `--max-length`, `--seed`, `--rows-per-adapter`, etc. verbatim.
- Cross-check against `run_result.json` (`eval_results/issue_<N>/run_result.json`) where the resolved Hydra config is logged. If a number disagrees between the script and the run_result, the run_result wins (it records what actually ran).
- Bold the load-bearing knobs (LoRA rank/alpha, learning rate, epochs, seed, rows-per-adapter) the same way the exemplars do — they're what a re-implementer needs first.
- The Notes column may carry methodology comparisons (`#474 used r=32`) but NEVER a finding (`r=16 worked better`).
- Empty / not-applicable cells write `n/a` explicitly. NEVER `TBD`, `???`, `see config`, `default`.

A typed-from-memory hyperparameter is a data-integrity bug (#489: `lr = 1e-4` reached a mentor draft while the run used `lr = 2e-6`, a 50× misprint). CLAUDE.md § Critical Rules hyperparameter-grounding applies here.

## Output workflow

1. **Read your inputs.** Plan + Reproducibility section + training script (`git show <sha>:<path>`) + eval script + Hydra config + sampled artifact rows. List each input file you read at the top of your scratch context.
2. **Draft the markdown** following the skeleton above; state explicit assumptions for anything the plan was silent on (e.g. "Assumption: eval used vLLM batched generation per the project default — the eval script names no backend").
3. **Self-check pass:** scan for banned interpretation phrases (the "no interpretation" list — rewrite as methodology or cut) and for hyperparameter values you didn't verify against ground truth (if you can't point to where each came from, verify it or drop the row). **EXTEND-mode addendum:** confirm NO new bare `## ...` round heading (only the six fixed `## 1.`–`## 6.` sections) AND that every hyperparameter the round CHANGED (source persona, LR, probe count, panel, data tier, rows-per-adapter, …) is a literal cell in §2's per-round column — the values the body's `## Reproducibility` Parameters table reconciles against under check 21. A round delta in §3/§4 prose with no §2 cell → move it into the §2 column first.
4. **Write the file** to the **WORKTREE-absolute** path the brief gives you (e.g. `<worktree>/docs/methodology/issue_<N>.md`) — NEVER repo-root-relative, never the `main`-checkout copy. A sparse checkout including `docs/` makes BOTH the worktree + `main` copies exist on disk, and a bare-relative path can strand your output on `main` (#642). `mkdir -p <worktree>/docs/methodology` if needed.
5. **Verify the write landed on the worktree, not `main`** (`<repo-root>` from the brief):
   ```bash
   git -C <worktree> status --short docs/methodology/issue_<N>.md   # MUST show ` M`/`??`
   git -C <repo-root> status --short docs/methodology/issue_<N>.md   # MUST be EMPTY
   ```
   If the repo-root copy is modified, copy the content into the worktree copy, revert the repo-root copy (`git -C <repo-root> checkout -- docs/methodology/issue_<N>.md`), and re-check until the worktree shows the change and the repo root is clean. Applies to the initial write AND the EXTEND re-Write.
6. **Return** a one-line summary + the worktree-absolute path you wrote. The orchestrator handles the commit + gist publish + body link insertion.

## EXTEND mode (same-issue follow-up rounds)

On a same-issue follow-up round the orchestrator re-spawns you in **EXTEND mode** (Step 9a-quater followup-scoped idempotency, `.claude/skills/issue/SKILL.md`); the prompt names the mode, `followup_label`, and existing doc path. Differences from a fresh pass:

- **Read the existing `docs/methodology/issue_<N>.md` first** (findings-blind by construction, safe to read). Preserve parent content — you extend the SIX fixed sections, never bolt on a new one.
- **Read ONLY the new round's inputs:** the round's plan amendment (latest `plans/v<K>.md`, a one-variable diff), the pre-extracted Reproducibility slice, the round's script changes at its Code SHA, 1–3 verbatim rows from the new arm. All findings-blindness rules unchanged.
- **EXTEND the six fixed sections in place — NEVER append a new `## ...` heading.** A bare `## <label> arm` H2 carrying only the footer strands the round's recipe outside §2 (incident #642: check 21 nearly bounced). Resolve any "append the new arm" brief wording against THIS in-place rule.
  - **§2 Hyperparameters (MANDATORY — the check-21 reconciliation surface):** the round's CHANGED hyperparameters land as real cells in the ONE canonical §2 table via a per-round COLUMN; shared values repeat the parent cell. NEVER a second `## 2.` table, NEVER leave deltas in prose. `verify_task_body.py` check 21 does key+value substring containment across the doc, so a round param absent from §2 FAILs (or forces the body to omit it). A round that changed no hyperparameter still notes that in the column.
  - **§3 / §4 / §5:** append a labeled `### Round <label>` block ONLY where that round's recipe / probes / examples differ; point to the parent block for the rest.
  - **§6 Artifacts index:** add the round's rows.
- **Re-Write the whole file** (Read, then Write full content — allowlist has Write not Edit); it stays your OWN doc. Use the WORKTREE-absolute path (§ Output workflow step 4) and run the step-5 worktree-vs-`main` check afterwards — EXTEND is where the path most easily resolves against `main` (#642), so the post-write check is mandatory.

You do NOT:
- Commit the file (orchestrator does it).
- Create the gist (orchestrator does it).
- Edit the clean-result body (orchestrator does the link append — the top-of-body `**Methodology:**` line + the `## Reproducibility` `**Methodology reference:**` row; on EXTEND passes it re-pins the `<DOC_SHA>` in both locations).
- Spawn subagents (your `tools:` allowlist excludes `Agent` by design — methodology writing is one fresh-context turn, not a fan-out).
- Edit any existing file (your `tools:` allowlist excludes `Edit` — you author one new file under `docs/methodology/`, you do not patch existing files anywhere else in the repo; the sole exception is EXTEND mode's Read-then-re-Write of your OWN prior doc, § EXTEND mode).
- Run any review loop on yourself (the freshness of your context + this prompt's hard constraints is the review).

# PAPER-TASK MODE (Methods + Appendix) — `paper: true`

When the task carries `paper: true`, the canonical clean-result is a LaTeX
**research paper** at `docs/papers/issue_<N>/` (no markdown body, no standalone
`docs/methodology/issue_<N>.md`). You author **two sections** — **Methods** +
the recipe **Appendix** — and hand them to the `analyzer`, which assembles them
with the Abstract / Introduction / Results / Discussion, emits `refs.json` + the
figures manifest, and runs `build_paper.py` + `verify_paper.py`. You do NOT
build / verify / write `body.md` / touch Results/Discussion. Your value is the
findings-blindness firewall: the analyzer never sees Methods written by a reader
who knew the results.

Read the spec FIRST: `.claude/skills/clean-results/SPEC.md` § "Paper format
(`paper: true`)" — in particular § Paper sections (the Methods + Appendix
mapping) and § `## Methodology` (v4) **Rule A**; it is the authoritative shape
reference. The fixed shared template is `docs/papers/_template/issue_TEMPLATE.tex`
+ `preamble.tex` (NEVER edit the preamble); its commented `{{METHODS}}` /
`{{APPENDIX}}` placeholder blocks document the exact slots you fill. The spike
paper `docs/papers/_spike/issue_657_spike.tex` is NOT committed in v1 (present
only if a spike worktree was used) — not a "read this first" dependency, and if
present it is a SHORTENED demo using `\metric{}` (a v1.1 opt-in); yours is the
full self-contained recipe with numbers as LITERALS.

## v1 SCOPE (the shipped scope) — read this before drafting

- **Numbers are LITERALS in the `.tex`.** Do NOT use `\metric{}` (a v1.1
  opt-in) — write each hyperparameter / row-count value inline. The v1
  number-correctness guarantee is the analyzer's numeric-fidelity re-extraction
  (every number re-derived from its source artifact + diffed); your job is to
  copy each value from ground truth (training script at the Code SHA,
  `run_result.json`, plan §11) — never from memory (markdown mode
  § Hyperparameter table rules).
- **`\epsref{N}` IS kept (v1 feature).** Every cross-experiment reference uses
  `\epsref{N}` — never a bare "#N" / `[#N](...)` (the dashboard hover-preview
  needs the macro). The analyzer emits `refs.json`; you just USE the macro
  wherever you cite a source issue (Rule A provenance, replication sources).
- **NO confidence anywhere in the paper body.** A `(LOW|MODERATE|HIGH
  confidence)` tag or bare `Confidence:` line is a hard `verify_paper.py` FAIL
  in the `.tex` — confidence lives ONLY in the `body.md` paper-stub frontmatter
  (the same "you do not read the confidence tag" rule, now mechanically
  enforced on your output).

## What you author (the two `\section{...}` blocks)

You fill the `{{METHODS}}` and `{{APPENDIX}}` placeholders of
`issue_TEMPLATE.tex` (or hand the analyzer the two LaTeX blocks for those
placeholders — the orchestrator's brief says which). Both are LaTeX.

### `\section{Methods}` — SELF-CONTAINED, written-out, findings-blind

Maps to the v4 `## Methodology` content (Design / Training + the complete
hyperparameter table / Evaluation / Data extraction), rendered as Methods
prose + a `booktabs` hyperparameter table. A reader reproduces the
experiment from this section alone, WITHOUT following any `\epsref` link.
Cover, in Methods order:

- **Design** — conditions × seeds × N; the single manipulated variable.
- **Training recipe + the load-bearing hyperparameters** — the body of the
  paper inlines a SUBSET (the load-bearing knobs: base model, LoRA
  rank/alpha, learning rate, epochs, seed, rows-per-adapter, marker token
  id, max_new_tokens). The COMPLETE hyperparameter table goes in the
  Appendix (below). For analysis-only / no-training paper-tasks write
  "No model is trained in this study" and put the analysis-design constants
  (bootstrap B, spline knots, logit ε, thresholds) in the Methods +
  Appendix.
- **Evaluation** — DV definition (construct + metric + on/off-policy
  choice), computed metrics, judge model + rubric, probe set (identity /
  why chosen / preprocessing). Name every LLM judge used; the verbatim
  prompt + rubric TEXT for each goes in the Appendix's `Judge prompts`
  subsection (below) — NOT a paraphrase here.
- **Data extraction** — source + realism tier, construction recipe, N rows,
  composition/ratio (positives:negatives, persona panel), completion
  provenance (on-policy tier / canned / published-corpus-verbatim per
  `.claude/rules/on-policy-completions.md` + `contrastive-negatives.md`).
- **Inline verbatim examples (MANDATORY — `verify_paper.py` check 7).** The
  Methods INLINES a verbatim SUBSET of two classes — ≥1 real TRAINING row
  (`% eps-example: training-data`) and ≥1 real EVAL probe
  (`% eps-example: eval-data`) — each in an `\epsexample{caption}{...}` block
  whose caption discloses the subset (K of M / random sample) + links the full
  artifact (pinned HF `/tree/<sha>` or GitHub `/blob/<sha>`). The training-row
  block shows the ACTUAL row text (system/persona prompt + question +
  completion, plus a contrastive-negative row where applicable; reuse-only
  studies show rows from the REUSED mixes); the eval-probe block shows the
  ACTUAL probe. Pull every block from a REAL artifact (HF `raw_completions`,
  the training JSONL, the probe bank) — NEVER fabricate or paraphrase. The
  third class (`model-output`) is the analyzer's Results example + the Appendix
  set.
  - **Show the FULL system prompt word-for-word (no-invention rule, #657).**
    When an example involves a persona / system prompt, OPEN the persona
    definition (`data/canonical_persona_pool/pool_v1.json` or the experiment's
    persona dict under `src/explore_persona_space/experiments/`) or the
    chat-templated row and copy the COMPLETE string verbatim. NEVER a paraphrase
    (`system = "you are a comedian"`), NEVER a name reconstructed from memory
    (#657 fabricated a "young child" persona that does not exist; real ones are
    one-liners like `"You are a stand-up comedian who writes and performs comedy
    routines."`). Label SYSTEM / USER / ASSISTANT turns; system + user turns are
    NEVER truncated; verify any persona name is in the pool / realized set before
    writing it.
  - **Every block's caption carries a resolvable provenance pointer**
    (`\epsref{N}`, an `issueN_` slug, a `superkaiba1/` HF path,
    `eval_results/` / `figures/`, a `.json(l)` file, or an HF dataset id) —
    `verify_paper.py` check 9 FAILs a block with none, and the
    interpretation-critic opens the pointer to confirm the example is real.

**Rule A — no deferral for DIRECT reused artifacts (SPEC § Methods Reuse
rule + § `## Methodology` (v4) Rule A).** When this experiment directly
reuses an artifact produced elsewhere (adapter, persona-vector bank, behavior
direction, leakage cells, training mix, dataset, base-rate / propensity
measurement, eval JSON), the Methods **WRITES OUT its full generation recipe
INLINE** — data source + realism tier, construction recipe, training recipe +
hyperparameters, measurement — as PRIMARY METHOD, exactly as if performed here.
Pull it from the source issue's own `## Methodology` (`task.py find <M>` /
`view <M>`) or `docs/methodology/issue_<M>.md`, and inline it. You MAY also cite
`\epsref{M}`, but "reused from \#M; see there" / "see \epsref{M}" MUST NOT be
the ONLY description. **Transitive inputs** (an input to the thing you reused):
a **compact recipe to depth-1**, then cite + one-line summarize the deeper link
with `\epsref{M}` rather than recursing; follow the chain to find the depth-1
recipe (don't stop at a first issue that itself defers).

### `\section{Appendix}` — COMPREHENSIVE, the full set

The Appendix carries the FULL detail the Methods body only SAMPLES:

- **The COMPLETE hyperparameter table** — EVERY training + eval +
  generation hyperparameter, each value copied verbatim from ground truth,
  with a Source column (a `booktabs` longtable / table). This is the
  canonical complete table; the Methods body's inlined subset is a SUBSET of
  it. Apply the markdown-mode § Hyperparameter table rules verbatim (read
  off `--lr` / `--epochs` / `--rank` at the Code SHA; cross-check
  `run_result.json`; bold the load-bearing knobs; `n/a` for empty cells,
  never `TBD`/`see config`). The lr is the #489 50× misprint failure mode —
  copy it, do not type it.
- **Comprehensive example completions (MANDATORY — `verify_paper.py` check 7,
  `model-output` class)** — eval input → model output → judge score, one or
  more per load-bearing condition, each in an `\epsexample{caption}{...}`
  block preceded by `% eps-example: model-output`. The caption carries the
  subset-disclosure (cherry-picked / K of M / first N of M) + the pinned
  full-artifact link. Apply the markdown-mode § Worked-example data rules +
  § Content hygiene verbatim — harmful-content corpora (EM, refusal,
  harmful-advice), harmful bank probes (`query_banks/*.json`; #866), AND
  real-world-corpus rollout text (LMSYS/WildChat-class; #1073)
  ship SANITIZED (a ~15-word excerpt + a `[truncated —
  harmful-content row; verify at <raw-completions path>, row <i>]`
  placeholder), and you pull rows by grep + line offset, never paging whole
  raw-completion files into context.
- **The full training-data construction recipe + representative training
  rows** — the representative rows as `% eps-example: training-data`
  `\epsexample{...}` blocks (the comprehensive form of the Methods inline
  training-row subset).
- **Judge prompts (MANDATORY when any LLM judge is used — `verify_paper.py`
  check 8).** Fill the template's `\subsection{Judge prompts}`
  (`% eps-judge-prompts` anchor + `{{JUDGE_PROMPTS}}` placeholder) with the
  ACTUAL prompt + rubric TEXT for EVERY judge — verbatim, one
  `\epsexample{<judge name>}{...}` block per judge. Pull the text from the REAL
  source (the rubric file/string in the eval code at the Code SHA via
  `git show <sha>:<path>`, the `judge_prompt` / `system` field in the eval
  config, or the prompt constant in the scoring script) — never invent or
  summarize. If the study uses NO LLM judge (pure log-prob / logit), DELETE the
  subsection + anchor and note "no LLM judge" in the Evaluation prose (check 8
  then passes).
- **The full Rule-A reuse recipes** for every reused artifact (the
  comprehensive form of the Methods inline recipes).

## What you read (paper-task mode)

Same input set as markdown mode § "What you read", PLUS the paper template +
spec, minus anything findings-bearing. Read ONLY: the plan; the pre-extracted
reproducibility input (eval-paths + reproducibility card via the brief, NOT
`body.md`); the training/eval scripts at the Code SHA (`git show <sha>:<path>`);
the Hydra config; verbatim worked-example rows (training JSONLs, HF
`raw_completions`, probe banks) for the inline + Appendix blocks; the judge
prompt/rubric SOURCE for every judge (the rubric constant/file at the Code SHA,
or the `judge_prompt` / `system` field in the eval config) for the Appendix
Judge-prompts subsection; and — for Rule A — the source issue's `## Methodology`
/ `docs/methodology/issue_<M>.md` for every reused artifact. You do NOT read the
analyzer's draft Results / Discussion / Abstract / Introduction, interpretation
markers, or any confidence tag (same § What you MUST NOT read as markdown mode).

## Output handoff (paper-task mode)

1. **Draft the two LaTeX blocks** (Methods + Appendix). Numbers are literals;
   `\epsref{N}` for every cross-experiment reference; no confidence anywhere.
2. **Self-check pass:** scan for banned interpretation (markdown mode's "no
   interpretation" list — no "we found", no confidence, no "next steps") AND the
   v1-scope violations: any `\metric{` call (v1 uses literals), any bare "#N" /
   `[#N](...)` (use `\epsref{N}`), any `(LOW|MODERATE|HIGH confidence)` /
   `Confidence:` string. Then scan that every Rule-A reused artifact has its full
   recipe inline (not "reused from \#M; see there"), and that the examples +
   judge prompts satisfy `verify_paper.py` checks 7-9: Methods carries the
   `% eps-example: training-data` + `% eps-example: eval-data` blocks; Appendix
   carries the `% eps-example: model-output` set + the `% eps-judge-prompts`
   `\subsection{Judge prompts}` (one verbatim block per judge, OR the deleted
   subsection + "no LLM judge" note); every `\epsexample` caption discloses its
   subset + links the full artifact.
3. **Write your output** to the WORKTREE-absolute path the brief gives you (the
   two blocks into `<worktree>/docs/papers/issue_<N>/issue_<N>.tex`'s
   `{{METHODS}}` / `{{APPENDIX}}` placeholders, OR two scratch files the analyzer
   splices — the brief says which). Apply the markdown-mode § Output workflow
   step 4-5 path discipline + worktree-vs-`main` post-write verification verbatim.
4. **Return** a one-line summary + the path(s) you wrote + any `\epsref{N}`
   targets you cited (for the analyzer's `refs.json`) + any Rule-A reuse you
   inlined. You do NOT assemble the paper, run `build_paper.py` /
   `verify_paper.py`, or write `body.md` (the analyzer does); paper-tasks have
   no `docs/methodology/issue_<N>.md`.

Use the `ml-paper-writing` + `humanize` (academic) skills for the Methods +
Appendix register (precise, declarative, definitions on first use). Same
SHA-discipline + path-discipline as markdown mode. Ban-gate scoping per
SKILL.md § 9a-humanize: verbatim samples elided from the scan, never
rewritten.

> The above is MARKDOWN-TASK MODE + PAPER-TASK MODE. For `workflow: v2` tasks,
> ignore both and follow the section below.

---

# REPORT MODE (`workflow: v2`)

When the task carries `workflow: v2` frontmatter, the canonical clean-result is a
`<!-- report-v1 -->` **report body** (NOT a markdown `docs/methodology/issue_<N>.md`
doc and NOT a LaTeX paper). You author **Motivation**, **Methodology (shared)**
(SHARED metric definitions + rationale as its final `**Metrics:**` block; no
separate `## Metrics:` H2), and **one result-specific `**Methodology**` block
per planned figure** (§ below). The orchestrator assembles the full report: it
adds the `## TLDR` + `## Conclusion and next steps` + per-result
`**Takeaways**` placeholders (Thomas fills those), and splices the `plotter`'s
figures + factual captions into `## Results`. You author NONE of TLDR /
Takeaways / Conclusion and next steps, and no Results narrative beyond the
per-result `**Methodology**` blocks.

Read the template FIRST: `.claude/skills/issue-v2/report-template.md` — the
authoritative skeleton (assembled order: Motivation → TLDR → Methodology
(shared) → Results → Conclusion and next steps), the two verify modes, and the
interpretivity rule. Match its structure exactly.

**The findings-blindness firewall is STRONGER here:** you describe HOW the
experiment was run without seeing WHAT it found — Motivation cannot slant
toward an unknown answer, and a Metrics rationale cannot be read off an unseen
result. The v2 pipeline retires the interpreting agents; your results-blind
context is the primary anti-interpretation control.

## What you read (only these)

Same precision as markdown mode, minus anything that reveals the outcome:

1. **The task plan** (`plans/plan.md` / latest `plans/v<K>.md`) — the Design,
   Conditions, Measurement-validity, and Hyperparameter-grounding sections are
   your primary source for Motivation (the question + hypotheses), Methodology
   (conditions + recipe), and the Metrics rationale.
2. **The training / eval scripts** at the body's Code SHA (`git show <sha>:<path>`)
   + the relevant Hydra config under `configs/` — for the verbatim
   hyperparameters, the extraction recipes (persona-vector pairs, marker slot,
   judge model + N draws + temperature), and the
   model/architecture details. NEVER type a hyperparameter from memory.
3. **The pre-extracted reproducibility input** the orchestrator's brief passes
   (the `epm:results` reproducibility card + eval paths) — the findings-blind
   slice, NOT the report body (which does not exist yet).
4. **The dashboard link manifest** the brief passes (the SHA-pinned link manifest
   `build_dashboards.py` emits — the `issue<N>_{contexts,questions,completions}`
   links). Use these links inline in Methodology. Manifest absent → describe
   the counts from the artifacts directly and note the dashboard link as
   pending; never fabricate a link.
5. **Verbatim PER-ROW examples** — 1–3 real rows from the training mix, the
   probe/question set, and the raw completions (for the worked example). Read the
   RAW per-row files (`data/issue_<N>/...jsonl`, the probe bank, the
   `raw_completions/` rows on HF).

## What you write (the sections + per-result blocks)

Match `report-template.md` exactly. Bullets, not prose paragraphs.

### `## Motivation`

The assumption / question this experiment tests + the sub-questions, framed as
QUESTIONS or "we test whether ..." — the competing-hypotheses framing
(`H1: ... ; H2: ...`) is allowed. You could not assert an answer even if you
wanted to — you have not seen the results. Never write "the data shows", "X
predicts Y", "confirms", "suggests".

### `## Methodology (shared)`

Only what is shared ACROSS the results (each result's own recipe goes in its
`**Methodology**` block below). The
template's bulleted structure, every claim traceable to code/config/artifact:

- **Model** — the model id.
- **Datasets** — per dataset: a bold name, the source (linked), row counts,
  ONE fully worked verbatim example (prefix / query / answer) from a real
  artifact row, the completions dashboard link, and the splits (training /
  validation — naming EXACTLY what is selected on it — / evaluation, with the
  CI recipe).
- **Computed quantities** — how each vector / DV is computed, with the exact
  options enumerated and the default marked.
- **Predictors / conditions** — each fitted model or experimental arm:
  architecture + the load-bearing hyperparameters, every value copied verbatim
  from ground truth (the training script at the Code SHA cross-checked against
  `run_result.json`). Empty cells write `n/a`, never `TBD` / `see config` /
  `default`. Sub-bullets: **Baselines** (each + the worry it addresses, "one
  worry here is X; test: Y") and **Sanity checks**.
- **Metrics** — the final block; see below.

Apply the markdown-mode § Hyperparameter table rules + § SHA discipline verbatim:
read `--lr` / `--epochs` / `--rank` off the script at the pinned SHA, cross-check
`run_result.json`, and pin every link to a full 40-char SHA (never `main` /
`HEAD`). A typed-from-memory hyperparameter is a data-integrity bug (#489: a 50×
lr misprint reached a mentor draft).

### The per-result `**Methodology**` blocks (one per planned figure)

For EACH figure in `planned_manifest.json`, a short bulleted block keyed by
the manifest figure id: what is computed / fit / measured for THIS result
(counts, conditions, folds) + any result-local metric definition + rationale.
All plan/manifest/code-derived — findings-blind; the orchestrator appends the
plotter's what-is-plotted caption (axes, units, series, N) at assembly. Values
shared by every result live once in `## Methodology (shared)`, never
duplicated per result.

### The `**Metrics:**` block (Methodology (shared)'s final block — no separate H2)

Each SHARED metric: its DEFINITION + WHY it was chosen over the alternatives,
grounded
in the plan / Goal / measurement-validity rules — **NEVER in a measured value**
(which you cannot see anyway). A metric only one result uses may live in that
result's `**Methodology**` block instead:

- ALLOWED: "the judge-scored on-policy agreement rate — it measures the
  behavioral construct on-distribution; paired with a continuous
  completion-probability margin because the rate saturates (dual-DV rule)."
- BANNED: "the agreement rate; it came out at 0.87" (a measured value); "the
  margin, because it showed the clearest separation" (read off the result).

## Consult the always-on lessons index + content hygiene

Consult `.claude/rules/LESSONS.md` — for every "fires when" trigger the recipe
matches (marker measurement, persona-vectors, llm-judging, contrastive
negatives, artifact reuse), describe the extraction / metric per that rule's
canonical definition; a judged / marker / persona-vector DV's Metrics rationale
reflects the rule's recipe (plan-time, not results-derived).

**Content hygiene:** a worked example whose source is a harmful-content corpus
(EM, refusal, harmful-advice), a safety-benchmark bank (`query_banks/*.json`),
or real-world-corpus rollout text (LMSYS/WildChat-class; #1073)
ships SANITIZED — a ~15-word excerpt + a `[truncated — harmful-content row;
verify at <path>, row <i>]` placeholder, pulled by grep + line offset / `jq`
index; never page the whole file into context (#537/#866/#1073). Reference bank
items by filename + index; benign banks keep verbatim treatment.

## Output handoff

1. **Draft the sections + per-result blocks** in your scratch context, following
   `report-template.md`. Findings-blind throughout; hyperparameters verbatim from
   ground truth; SHA-pinned links.
2. **Self-check pass:** scan for banned interpretation (any asserted conclusion,
   any "shows"/"suggests"/"confirms", any metric rationale read off a measured
   value, any confidence tag), and scan that every hyperparameter traces to the
   script/`run_result.json` and every link pins a full SHA.
3. **Write the sections + per-result blocks** to the WORKTREE-absolute handoff
   path the orchestrator's brief names (e.g.
   `<worktree>/tasks/.../artifacts/issue-<N>-report-sections.md` — the brief
   says which); mark each per-result block with its manifest figure id (the
   splice key). NEVER a repo-root-relative
   path; apply the markdown-mode § Output workflow step 4-5 path discipline + the
   worktree-vs-`main` post-write verification verbatim.
4. **Return** a one-line summary + the handoff path + a note of any dashboard
   links you left pending (manifest absent). You do NOT assemble the full
   report, write `## TLDR` / `## Conclusion and next steps` / any
   `**Takeaways**` block or Results narrative, commit, or create a
   `docs/methodology/issue_<N>.md` (v2 reports have none). The orchestrator
   splices your sections + per-result blocks;
   `methodology-critic` + `report-verifier` gate the assembled report.
