# Clean-result spec — markdown

The canonical spec for clean-result body shape, voice, sections, and
anti-patterns lives in **`.claude/plans/task-workflow-migration.md`
§ 10**. The mechanical verifier is **`scripts/verify_task_body.py`**
(eleven checks). The format is **markdown** with YAML frontmatter.

## Required body shape

Four H2 sections in this order:

1. `## TL;DR` — four bullets labelled **Motivation / What I ran /
   Results / Next steps**. "I" voice; plain language; cite prior
   tasks via `[#K](https://eps.superkaiba.com/tasks/K)`.
2. `## Figure` — one markdown image link, then a `*Caption: …*` line
   ≥10 words.
3. `## Details` — single narrative covering definitions, training,
   eval rationale, sample completions inline (cherry-picked label +
   qualitative-data link), "Why this test" paragraph, parameters
   table, and the `Confidence: LOW|MODERATE|HIGH — …` sentence.
4. `## Reproducibility` — three groups (Artifacts, Compute, Code).
   All URLs permanent: HF Hub `/tree/<ref>`, WandB `/runs/<id>`,
   GitHub `/blob/<sha>`. `n/a` accepted as an explicit non-applicable
   marker. No `TBD`, `{{`, `default`, `see config` sentinels.

Title format (the H1 line):

```
# <one-sentence claim> (LOW|MODERATE|HIGH confidence)
```

The confidence level in the title MUST match the `Confidence: ...`
sentence in `## Details`.

## Eleven mechanical checks (`verify_task_body.py`)

1. Title ends with `(LOW|MODERATE|HIGH confidence)`.
2. Four required H2 sections present in order
   (`## TL;DR`, `## Figure`, `## Details`, `## Reproducibility`).
3. TL;DR bullets carry the four labels (Motivation, What I ran,
   Results, Next steps).
4. `## Figure` contains at least one `![alt](url)` markdown image.
5. Figure caption (first non-image line in `## Figure`) is ≥10 words.
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
  `codex-clean-result-critic.md` — critique against the seven lenses
  and run `verify_task_body.py` +
  `audit_clean_results_body_discipline.py`.
- `.claude/skills/promote-clean-result/SKILL.md` — for legacy HTML
  bodies, optionally converts them to markdown on promotion.
- `CLAUDE.md` § "Experiment Report Structure" — points at this spec
  and at the migration doc.
