# Nested-design (v2) exemplar — task #432

**What this exemplar shows.** The 2-content-section nested-design (v2)
shape spec'd by `.claude/skills/clean-results/SPEC.md` (adopted
forward-only after the 2026-W22 four-H2 → three-H2 migration in task
#454). The canonical full-text body lives at
`tasks/completed/432/body.md`;
this exemplar is a compact pointer + rule-by-rule walkthrough.

Read alongside `docs/lw-post-examples/` (external LW register references)
and `iterations.md` (rule provenance).

**Rules demonstrated** (cited inline at each beat below):

- **SPEC.md § Required body shape** — three required H2s in order
  (`## Human TL;DR` / `## TL;DR` / `## Reproducibility`); `## TL;DR`
  opens `### Motivation` → `### What I ran` → `### Findings` (parent)
  with one `#### <finding>` H4 per result.
- **SPEC.md § V2 nested-design sentinel** — the body carries the
  literal HTML comment `<!-- clean-result-v2 -->` right after the H1
  title. The verifier's nested-shape requirement + the
  confidence-title-only permission are sentinel-gated.
- **SPEC.md / CLAUDE.md § Experiment Report Structure — confidence
  in H1 title tag only** — no `Confidence: …` sentence in
  `## Reproducibility`.
- **`### Motivation` is the only place issue numbers appear**
  (`[#K](https://eps.superkaiba.com/tasks/K)` markdown links —
  never bare `#K`).
- **`### What I ran` is standalone** — descriptive baselines ("the
  narrow 2-negative baseline"), no cross-issue framing, no "byte
  identical" / "byte-identical", training INPUT→OUTPUT examples in a
  `<details open>` table, eval INPUTS named.
- **Each `#### <finding>` H4 follows setup → figure → blockquote
  caption → read** (Lens 12 check #2: figures inline-narrated, not
  figure-dumped). Adjacent raw + processed pairs allowed under
  Lens 11. **Caveat: the full-text #432 body does NOT demonstrate
  this beat** — its finding blocks carry long (≥4-sentence)
  figure-last setup narrative with no post-caption read paragraphs.
  The § Skeleton below IS the canonical reference for this rule;
  never cite the #432 body as precedent for long or missing read
  paragraphs (SPEC.md § Target exemplar scope caveat; task #547
  round-1 reconciliation, 2026-06-10).
- **No `### Methodology corrections` H3** — when a correction is
  load-bearing, fold it into the relevant `#### <finding>` setup or
  read prose (Lens 8 update + Lens 12 check #3).
- **Sample-output discipline (checks 10 + 11)** — cherry-pick
  disclosure can live in the `<details>` `<summary>` text ("5 example
  training rows (1 positive + 4 of the 9 negatives)") and the
  qualitative-data link can live inside the `<details>` block (the
  "Full training file" link inside the dropdown). The verifier scans
  both the prelude window (incl. `<summary>`) AND the inner
  `<details>` content.

---

## Skeleton (verbatim shape)

```markdown
# <one-sentence claim> (LOW|MODERATE|HIGH confidence)

<!-- clean-result-v2 -->

## Human TL;DR

placeholder

## TL;DR

### Motivation

<1-3 paragraphs: prior tasks via [#K](https://eps.superkaiba.com/tasks/K)
links, the question walked in with, the prior the analyzer held, ending
with the goal of THIS run.>

### What I ran

<Standalone description: training mix, model, hyperparameters that matter
to the reader, eval rig in one sentence. NO issue numbers, NO "byte
identical" phrasing.>

<details open>
<summary>5 example training rows (1 positive + 4 of the 9 negatives)</summary>

| Row | System prompt | User question | Assistant |
|---|---|---|---|
| ... | ... | ... | ... |

Full training file (2000 rows): [link to permanent HF Hub /blob/<sha>/...]

</details>

**What the eval measures.** <1-3 sentences naming the actual probes
and what's computed. Link the full per-cell tensor.>

<details open>
<summary>5 example eval probes</summary>

<table of probes; link to full eval JSON inside the dropdown>

</details>

### Findings

#### <story-beat headline for finding 1>

<Setup paragraph: 1-3 sentences framing what the figure will show.>

![<descriptive alt text>](https://raw.githubusercontent.com/<owner>/<repo>/<sha>/figures/issue_<N>/<file>.png)

> **Figure.** *<one-sentence italic lead claim.>* <Caption body:
> definitions, ns, color mapping, what to look at.>

<Read paragraph: 1-3 sentences calling out surprises, outliers,
monotonicity, what the figure CAN'T tell you.>

<For text-generation findings: cherry-picked example block here.>

#### <story-beat headline for finding 2>

<same skeleton...>

#### <story-beat headline for finding 3>

<same skeleton...>

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | ... |
| Optimizer | ... |
| Steps | ... |
| Seeds | ... |
| Hardware | ... |
| Hydra config | `condition=...` |

**Artifacts:**

- Training data: [permanent HF Hub /blob/<sha>/... link]
- Eval JSON: [permanent GitHub /blob/<sha>/... link]
- Figure source code: [permanent GitHub /blob/<sha>/... link]
- ...

**Compute:**

- Wall time: ...
- GPU: ...
- Pod: ...

**Code:**

- Dataset build script: [link]
- Pipeline driver: [link]
- Hydra condition config: [link]
- Git commit (figures + analysis): `<full 40-char sha>`
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout <full sha>
    uv sync
    # ... reproduce commands ...
    ```
```

---

## Why this exemplar replaced `narrative-380.md`

The old `narrative-380.md` exemplar was built around the `## Details`
H2, which was retired in 2026-W22 (task #454). Its many-findings
rollup pattern is no longer the recommended shape under the
nested-design (v2) spec: the `### Findings` parent H3 + per-result
`#### <finding>` H4s is the canonical many-findings pattern now.
`narrative-380.md` is kept on disk only for historical context;
reach for THIS exemplar (or the full #432 body) when drafting new
clean-results.
