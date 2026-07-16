---
paths:
  - ".claude/rules/clean-result-critic-lens-reference.md"
description: >
  Full fifteen markdown-lens rubrics + the spec-text-only checks for
  clean-result-critic.md, relocated verbatim (#1159; content unchanged).
  Loaded ONLY via the explicit pointers in clean-result-critic.md and the
  codex-clean-result-critic Step-2 composer read — the self-matching
  `paths:` glob keeps this file out of every other agent context (a missing
  `paths:` key would auto-inject it always-on fleet-wide, recreating the
  #833/#834 spawn-weight bug this relocation fixes).
---

# Clean-result critic lens reference (clean-result-critic.md relocated lens rubrics)

Sixteen H3 sections, headings verbatim from clean-result-critic.md. Grep the
heading you need and `Read` ONLY that span (chunked, per
clean-result-critic.md § Context budget). The codex-clean-result-critic
composer copies ALL fifteen lens sections VERBATIM and IN FULL from this file
into the composed prompt.

### Spec-text-only checks (mechanical PASS is necessary, NOT sufficient)

**`verify_task_body.py` + `audit_clean_results_body_discipline.py`
PASSING does NOT mean the body is spec-compliant.** Several lens rules
live only in the spec text (CLAUDE.md § Experiment Report Structure +
`.claude/skills/clean-results/SPEC.md`); the scripts have no regex for
them. Before scoring any lens "PASS" off a clean mechanical pre-pass,
re-read the body against the SPEC for the rules below — these are the
ones Claude has historically over-trusted the scripts on, where the
Codex twin + reconciler caught real blockers.

For each rule: open the body, find the section, verify against SPEC
directly. Do NOT score the lens "PASS" by reasoning "the audit was
clean, so this passes."

- **Lens 12 (Conciseness) — `### <result>` interpretation prose runs 1–3
  sentences per paragraph; bullets are the default.** The mechanical
  word cap (check 20) hard-FAILs at ≥180 words/result and at a ≥100-word
  v4 Takeaways bullet; check 36 WARNs mechanically at ≥4
  sentences/paragraph in `### <result>` prose (#1368); the
  bullets-over-prose call and the FAIL decision stay LM judgment. Scan
  each `### <result>`
  interpretation paragraph (the prose that follows the figure caption);
  FAIL when
  a result's prose is a multi-sentence wall where bullets would read
  better, or any single paragraph runs ≥4 sentences in the analytical
  read. (v3: `### <finding>` read paragraph.) (Incident lineage: task
  #385 round 1 — a 5-sentence read
  paragraph the Claude critic PASSed under the old spec.)
- **Lens 2 — no body `Confidence: …` sentence** (SPEC
  `.claude/skills/clean-results/SPEC.md`). For v4 bodies confidence
  lives in the H1 title tag ONLY; there is no body `Confidence: …`
  sentence and no "Why confidence is where it is" section. FAIL when a
  v4 body emits a Confidence sentence anywhere — the title tag is the
  source of truth, redundancy is reader-hostile. (v3 carries no body
  `Confidence:` sentence either; legacy bodies, no v3/v4 sentinel: the
  grandfathered rule applies per SPEC.md.)
- **Lens 2 — no bolded-paragraph leads (`**Sub-topic name.**`) used as
  inline subheadings inside `### <result>` prose.** The dashboard's
  markdown renderer collapses bolded leads into a wall of text with no
  visual break. Scan each `### <result>` for paragraphs starting
  `**[A-Z][^*]+\.**` that function as subheadings; FAIL when ≥3 appear
  in a single result. (The `**Design:**` / `**Training:**` /
  `**Evaluation:**` / `**Data extraction:**` /
  `**Sample ...:**` slot leads in `## Methodology` and the
  `**This experiment in context:**` / `**Broader narrative:**` slot leads
  in `## Goal` are the
  REQUIRED v4 structure — they are NOT flagged. v3: the `**Why:**` /
  `**Design:**` / `**Training:**` / `**Eval:**` / `**Rounds:**` slot leads
  in `## What I ran`.) (Incident: #389
  round 1.)
- **Lens 9 — end-to-end example: a per-result text excerpt is fine,
  the systematic samples live in `## Methodology →
  **Sample training/evaluation data + completions:**`** (SPEC
  § `## Results` per-result skeleton step 4 + § `## Methodology`). Under
  v4 the bulk per-condition samples + `<details>` dropdowns live in the
  `## Methodology` Sample slot, NOT inside each result; a result whose
  text IS the evidence may carry at most ONE ≤10-line excerpt, preceded
  by a subset-disclosure line AND a raw-completions link. FAIL on: a
  result excerpt with an `main`/`HEAD` HF link instead of a permanent
  SHA; a missing subset-disclosure label; a `## Methodology` Sample slot
  that omits the per-load-bearing-condition example for a
  text-generation run. (v3: the per-finding excerpt + the
  `## Data → ### Generated` systematic samples.) (Incident: task #385
  round 1 — block absent; Claude critic PASSed.)
- **Lens 7 — bracketed-CI form (`[low, high]`, `Wilson 95% CI [..., ...]`,
  `upper bound = 0.0021`) in `## Takeaways` / `## Goal` /
  `## Methodology` / `## Results` prose** is the same banned construct as
  `value ± err`.
  The audit's `±` regex misses bracketed bounds;
  `audit_clean_results_body_discipline.py` lists `slope[low, high]` but
  the broader bracketed-CI pattern is spec-text. Exception: a
  result-internal "Why this test" sentence that explicitly names the CI
  as part of the test definition. FAIL when bracketed bounds appear in
  result what-is-plotted/interpretation prose or a Takeaways bullet.
  (v3: finding setup/read prose.) (Incident: #382
  round 1.)
- **Lens 8 — title methodology framing semantics.** Lens 8 lists
  example regex patterns ("once X was corrected", "after fixing", "but
  the rig also breaks") but the rule is semantic — any title that leads
  with the correction story instead of the post-correction finding fails.
  Don't gate on regex hit alone; re-read the title in isolation and ask
  "would a mentor reading this ask what the experiment FOUND or what the
  correction STORY was?" FAIL on the latter even if no listed regex
  matches. (Incident: #389 round 1 — "but the planned belief-vs-retrieval
  discriminator was confounded by the C-family judge rubric"; not in the
  example regex list, semantically a title-mistake-framing FAIL.)
- **Lens 2 / 3 / 10 — "family"/short-letter labels (`A-family`,
  `B-family`, `C-family`, `Method A`, `Bin C`, `K1`, `M1`, `BS_E0`)**
  in `## Takeaways`, `## Goal`, `## Methodology`, `## Results` prose, a
  figure caption, or a `## Methodology` capsule. The audit catches
  `Bin\s+[A-E]` and
  some Hydra-shape codes but misses `<letter>-family` constructions and
  bespoke short labels. FAIL on any such token without a plain-English
  name in the same section. (v3: `## Takeaways`, `## What I ran`,
  `## Findings` prose, a figure caption, or a `## Data` capsule.)
  (Incident: #389 round 1.)
- **CLAUDE.md "Plain-English condition names end to end" — cell-letter
  codes in reader-facing prose** (`cells A/C/D/D′`). Audit is narrower
  than the spec text. Bare codes survive ONLY in the `**Repro:**` footer +
  the Methodology Training-table config row + launch-command examples (and
  inside `## Methodology` verbatim example blocks, which are audit-exempt).
  FAIL on cell-letter codes anywhere in `## Takeaways`, `## Goal`,
  `## Methodology`, `## Results`, or a `## Methodology` capsule. (v3:
  `## Reproducibility` + the Parameters table's config row +
  `## Data` verbatim example blocks; cell-letter codes anywhere in
  `## Takeaways`, `## What I ran`, `## Findings`, or a `## Data` capsule.)
  (Incident: #382 round 1.)
- **Lens 6 — `byte identical` / `byte-identical` anywhere in the
  body** (banned 2026-W22, task #454; carried into v3). The phrase
  reads as AI-slop in research writing. Use plain English: "the two
  files matched exactly", "every byte agreed", "no diff between the
  runs". Flagged by `audit_clean_results_body_discipline.py`; FAIL on
  any occurrence outside fenced code blocks.

**Procedure.** Before writing any "Lens N: PASS" line for Lenses 2, 3,
6, 7, 8, 9, 10, 12, work through the bullets above first. If a bullet's
rule applies and the body violates it, the lens is FAIL even when the
mechanical pre-passes are clean. The Codex twin runs the same checklist
every round (all rounds as of 2026-06-12); PASSing while Codex FLAGs
these is the canonical
reconciler-disagreement shape captured in
`.claude/agent-memory/reconciler/feedback_claude_clean_result_critic_underapplies_spec_text.md`.

> **Paper-task review (`paper: true`)** — for a paper-task, READ
> `.claude/rules/clean-result-paper-review.md` IN FULL: the seven P1-P7 paper
> lenses + the `verify_paper.py` pre-pass REPLACE the fifteen markdown lenses
> below. (Relocated verbatim from this spec, #829.)

### Lens 1 — Title

- Title line is a single H1 (`# ...`) ending exactly in
  `(LOW confidence)`, `(MODERATE confidence)`, or `(HIGH confidence)`.
- States the **actual finding**, not the experiment name.
- One claim, not stacked claims separated by em-dashes.
- Precise verbs that name direction + comparison anchor ("increases
  marker leakage by Δ N pts" not "X leaks Y").
- ≤ two project-internal entities named in the title.
- The confidence tag is the body's SINGLE source of truth (v4 carries
  no body `Confidence: …` sentence; v3 carries none either). Check it
  semantically — does the
  text-level argument across `## Results` (v3: `## Findings`) actually
  support that level?
- **Goal alignment (soft check).** Read `frontmatter.goal` from
  body.md. Does the title's confidence claim actually answer the
  stated Goal? A HIGH-confidence title on a question the Goal didn't
  pose is an overclaim. Flag misalignment as a Lens 1 finding; the
  analyzer revises the title (Goal is contract, never the title).

### Lens 2 — v4 structure (Takeaways shape + Goal slots + Methodology slots + Results skeleton)

The four-flat-H2 (v4) spec flattens the body into `## Takeaways` /
`## Goal` / `## Methodology` / `## Results` plus the `**Repro:**` /
`**Context:**` footer. This
lens covers the STRUCTURE of the content H2s (Takeaways
register is its own Lens 4; the Methodology data slots are Lens 10; word
caps are Lens 12).
The verifier's `check_v4_structure` (check 3) is the mechanical floor;
this lens is the substantive read on top. (v3: the five-flat-H2 shape
`## Takeaways` / `## What I ran` / `## Findings` / `## Data` /
`## Reproducibility`, gated by `check_v3_structure`.)

**`## Takeaways`:**
- 3-6 bullets, no paragraphs (the verifier owns the 3-6 count as the
  AUTHORITATIVE gate — FAIL outside range is a check-3 structural FAIL,
  recorded as such). Here you confirm the bullets are bullets, not a
  paragraph dressed up.
- It is the ONLY place the body's cross-round synthesis lives; the
  register + currency check is Lens 4. Structurally, FAIL when
  `## Takeaways` is empty, is a prose paragraph, or duplicates a
  result heading verbatim instead of stating the belief.

**`## Goal`:** (v3: the WHY / prior-task-link content lived in the
`**Why:**` slot of `## What I ran`.)
- Carries BOTH slot bullets `**This experiment in context:**` AND
  `**Broader narrative:**`. Both are verifier-required (check 3); FAIL
  when either is missing.
- **`**This experiment in context:**` is the ONLY place issue numbers /
  prior-task links appear** (cited via
  `[#K](https://eps.superkaiba.com/tasks/K)`
  markdown links — never bare `#K`). `## Methodology` and `## Results` are
  STANDALONE (descriptive baselines, e.g. "the narrow 2-negative
  baseline", not `#K`-linked). FAIL on a `[#K]` link or bare `#K` in
  `## Takeaways`, `## Methodology`, `## Results`, or a `## Methodology`
  capsule. (v3: the `**Why:**` slot carried the links; `## Findings` /
  `## Data` were standalone.)
- **No methodology-correction framing of a prior run** in
  `**This experiment in context:**`
  (or anywhere). When this experiment changed or fixed methodology
  relative to an earlier issue, the body describes ONLY what THIS run
  did — FAIL on "the prior run used X, this run uses Y", "reverting
  axis A/B/C from #K", a prior-vs-current table of design choices, or
  a recap of the earlier run's superseded eval rig / negatives / panel
  / judge. `**This experiment in context:**` may name a prior result to
  establish the open
  question; it must not relitigate that run's methodology.

**`## Methodology`:** (v3: the one-line recipes lived in the
`**Design:**` / `**Training:**` / `**Eval:**` slots of `## What I ran`;
the full Parameters table lived in `## Reproducibility` and the full
training rows / probes lived in `## Data`.)
- Carries the slot bullets `**Design:**`, `**Training:**` (with the
  COMPLETE hyperparameter table), `**Evaluation:**`,
  `**Data extraction:**`, and `**Sample training/evaluation data +
  completions:**`. The Training + Evaluation slots are verifier-required
  (check 3); FAIL when either is missing.
- The `**Design:**` / `**Training:**` / `**Evaluation:**` slots are
  recipes; under v4 the COMPLETE hyperparameter table lives HERE in
  `## Methodology → **Training:**`, the full training rows / probes live
  in the Methodology data slots, and the compute / code SHA / pinned
  links live in the `**Repro:**` footer. No cross-issue
  framing, no `byte identical` / `byte-identical`.
- Plain language, accessible to a non-specialist. No jargon undefined
  before it is used.

**`## Results` (one `### <result>` per result):** (v3: `## Findings` with
one `### <finding>` per result.)
- `## Results` has ≥1 `### ` result (verifier check 3); FAIL on zero.
- Each `### <result>` heading names a story beat / states the result
  WITH the number in the heading (good: `### Source-marker firing rises
  to 0.83 while bystander leakage stays at 0.02`; bad outline labels:
  `### Headline result` / `### Subset checks` / `### Sample completions`
  / `### Plan deviations` / `### Methodology` / `### Methodology
  corrections`).
- Each `### <result>` carries exactly ONE inline figure with a
  markdown blockquote caption (`> **Figure.** *italic lead.* ...`). The
  one-figure pairing rule is Lens 9; the
  what-is-plotted-above/interpretation-below figure
  pairing is Lens 3.
- Each `### <result>` STANDS ALONE — a reader can land on it directly
  and understand it without re-reading earlier results.
- Defines every term where introduced (formal + intuition).
- Includes a "Why this test" sentence inline inside the result that
  needs it (NOT a separate heading — the rationale lives inline).
- **Generator disclosure for in-context artifacts** (semantic check):
  when the body evaluates a finetuned model against few-shot
  demonstrations, a chain-of-thought prefix, a judge prompt, a
  synthetic dataset, or any other in-context component that is itself a
  model-generated artifact, the relevant `### <result>` MUST name the
  generating model. Default reader assumption is "the model being
  evaluated"; any deviation (unadapted base model, a different adapter,
  a stronger oracle model, an external judge such as Claude Sonnet)
  must be made explicit. Flag missing disclosure as a Lens 2 FAIL —
  confound-disclosure asymmetry, not a stylistic nit.
- **Methodology corrections fold into the relevant result's prose**
  (no `### Methodology corrections` heading); if the body emits one,
  that is a Lens 2 FAIL.
- **No bolded-paragraph leads as inline subheadings** inside a
  `### <result>` (the dashboard renderer collapses them into a wall of
  text). Trigger to FAIL: ≥3 bolded-lead paragraphs
  (`**Sub-topic name.**`) inside a single result. (The `**Design:**` /
  `**Training:**` / `**Evaluation:**` / `**Data extraction:**` /
  `**Sample ...:**` slot leads in `## Methodology` and the
  `**This experiment in context:**` / `**Broader narrative:**` slot leads
  in `## Goal` are the REQUIRED v4 structure — NOT flagged. v3: the
  `**Why:**` / `**Design:**` / `**Training:**` / `**Eval:**` /
  `**Rounds:**` slot leads in `## What I ran`.)
- **No opaque condition / run / config codes.** Hydra-style or
  config-derived condition names — anything matching the shape
  `[a-z]+_[A-Za-z0-9]+` (e.g. `sw_eng_C1`, `sw_eng_expA`,
  `sw_eng_expB-P1`, `cond_4`, `c1_evil_wrong_em`), short-letter labels
  (`M1`, `Method A`, `Bin C`, `K1`, `BS_E0`), or any token that names
  a condition without being self-explanatory English — **must NEVER
  appear in `## Takeaways`, `## Goal`, `## Methodology`, `## Results`, or
  a `## Methodology` capsule** (verbatim example blocks inside
  `## Methodology` are
  audit-exempt). Always use the plain-English name of the condition
  (e.g. "the paraphrased-prompt arm", "the unmodified code-evaluation
  baseline", "the model finetuned only on software-engineering
  refusals"). FAIL on any occurrence. Code-style parentheticals like
  `"the paraphrased-prompt arm (sw_eng_expA)"` are ALSO forbidden in
  the reader-facing sections — the bare code goes in the `**Repro:**`
  footer. (v3: the same applies to `## Takeaways`, `## What I ran`,
  `## Findings`, or a `## Data` capsule; the bare code goes in
  `## Reproducibility`.)
- **Confidence is in the H1 title tag only.** Do NOT require — and FAIL
  on — a `Confidence: …` sentence anywhere in a v4 body (v3 too). If the
  body
  author needs to surface the binding constraint, it lives in the
  relevant `### <result>` interpretation prose and/or a `## Takeaways`
  bullet.
- If raw completions weren't uploaded for this run, the relevant
  result (or `## Methodology → **Sample training/evaluation data +
  completions:**`) MUST surface a "re-run with
  raw-completion upload" note. Check the run metadata or the prose.

### Lens 3 — Figure (absorbs the what-is-plotted/interpretation–figure pairing check)

- **Figure-source resolution (pin-first — #922).** The review target for
  every figure AND its `.meta.json` sidecar is the BODY-PINNED blob (the
  `<sha>` + `<path>` in the body's
  `raw.githubusercontent.com/<owner>/<repo>/<sha>/<path>` URL), never an
  unverified working-tree file. Resolution order: (1) read text sidecars
  straight off the pin — `git show <sha>:<path>` (works from any checkout;
  worktrees share the object DB; if the SHA is locally absent, fetch the
  raw URL instead); (2) a local copy (issue worktree or repo root) may
  serve as the read target ONLY after blob-identity is verified:
  `[ "$(git hash-object <local>)" = "$(git rev-parse <sha>:<path>)" ]`;
  (3) to VIEW a pinned PNG with no identity-verified local copy,
  materialize it: `git show <sha>:<path> > /tmp/pin-<file>.png`, then Read
  that. NEVER treat an untracked (`git status --porcelain` → `??`) or
  identity-failed local copy as evidence — a blocker resting on such a
  read is INVALID. A local-vs-pin mismatch is a NOTE ("possible stale
  stray at <path>; review target is the pin"), not a body defect. (#922:
  a stale untracked repo-root `figures/issue_922/*.meta.json` produced a
  spurious REVISE and burned a reconciler round; the pinned blob was
  correct.)
- At least one image exists in the body, inline `![alt](url)` inside a
  `### <result>` under `## Results` (each result carries its own
  figure; one figure per result). (v3: `### <finding>` under
  `## Findings`.)
- A stray `## Figure` H2 in a v4 body is a hard FAIL (verifier check 2
  rejects it; v3 too). Inline the figure inside the relevant
  `### <result>`.
- Each image is a markdown image link (`![alt](url)`) with a permanent
  absolute URL (HF Hub `/tree/<sha>` or GitHub
  `raw.githubusercontent.com/.../<sha>/...`). No `<figure>` / `<img>`
  HTML — markdown only.
- Each `### <result>` carries a markdown blockquote caption right after
  the image: `> **Figure.** *one-sentence lead claim in italics.*
  Remaining caption prose in plain text.` Caption ≤60 words (verifier
  check 20 WARNs over the cap); explains axes + observed trend + what
  the figure does NOT show, in plain English. No math notation in the
  caption.
- The alt text of each inline image is descriptive, plain-English, axes
  + trend explained. Empty / single-word alt text → FAIL with "rewrite
  the alt text to describe what's plotted".
- **No opaque condition / run / config codes anywhere in the figure.**
  This covers: axis labels, axis tick labels, legend entries, bar/line
  group labels, in-figure annotations, alt text, AND the caption.
  Anything matching `[a-z]+_[A-Za-z0-9]+` (e.g. `sw_eng_C1`,
  `sw_eng_expA`, `sw_eng_expB-P1`), short-letter labels (`M1`,
  `Method A`, `Bin C`, `BS_E0`), or any non-self-explanatory token →
  **FAIL with "regenerate the figure with reader-facing labels"**. Use
  plain-English condition names directly on the chart (e.g. "paraphrased
  prompts", "unmodified baseline", "SFT only on refusals"). Code-style
  parentheticals (`"paraphrased prompts (sw_eng_expA)"`) are ALSO
  forbidden in the caption — bare codes belong in the `**Repro:**` footer
  (v3: `## Reproducibility`).
- **What-is-plotted/interpretation–figure pairing (absorbed from the
  retired story-arc lens, now expressed as the per-result three-beat).**
  Every `![alt](url)` figure inside a
  `### <result>` is framed by a **what-is-plotted (EXACTLY) beat** (what's
  plotted, why we're looking) ABOVE it AND an **interpretation beat**
  (what's striking, where outliers go, what the figure CAN'T tell you)
  BELOW it. FAIL when a figure has no what-is-plotted above OR no
  interpretation below — a
  `![alt](url)` line surrounded only by other figures or by tables is a
  chart pasted into a document, not a chart embedded in a result.
  Adjacent figures are allowed when they're a raw + processed pair
  (Lens 11); they count as ONE narrative unit (what-is-plotted above the
  pair, interpretation below the pair). (v3: the per-finding
  setup-above/read-below skeleton.)

### Lens 4 — Takeaways quality

`## Takeaways` is the 10-second read and the surface Thomas adapts for a
Slack post. Lens 2 owns its STRUCTURE (3-6 bullets, not a paragraph);
this lens owns its REGISTER and its CROSS-ROUND SYNTHESIS CURRENCY.

**Register.**
- Plain academic register — NOT the old casual lowercase / diary voice,
  NO "How this updates me" framing. FAIL on lowercase-casual or diary
  voice in `## Takeaways`.
- **Numbers-first.** Each bullet leads with or bolds its load-bearing
  number + CI where one exists ("Untrained base PASSes pushback at
  **4.40/5 (CI 4.13-4.67)**…"), not an adjective ("the base does well on
  pushback"). FAIL a bullet that asserts a quantitative finding with no
  number when the finding has one.
- Each bullet ≤30 words (verifier check 20 WARNs over the cap and
  hard-FAILs a v4 bullet ≥100 words; flag here if a bullet is a runaway
  sentence).
- First person stays (`I`, not `we`). No effect-size names / named
  statistical tests / inline `value ± err` (that is Lens 7).

**Cross-round synthesis currency (the load-bearing rule).**
- `## Takeaways` ALWAYS reflects the CURRENT cross-round belief, NOT
  just round 1. When `## Methodology` carries a `**Rounds:**` table with
  >1 round (or the `**Context:**` footer names a follow-up
  round), `## Takeaways` MUST integrate the later round's result. **A
  `## Takeaways` that describes only round 1 after round 2 landed is a
  FAIL** — the analyzer must rewrite it to the current synthesis and
  retitle the H1 if the headline moved. (v3: the `**Rounds:**` table
  lived in `## What I ran` and the `**Context:**` row in
  `## Reproducibility`.)
- Cross-check: read the `### <result>` headings + the `**Rounds:**`
  table; every load-bearing result from the LATEST round must be
  reflected in (or consciously subsumed by) a Takeaways bullet. A
  Takeaways bullet that contradicts the latest result, or omits a
  headline-moving later-round result, is a FAIL. (v3: `### <finding>`
  headings.)

### Lens 5 — Footer / Reproducibility

- **Top-of-body `**Methodology:**` line carve-out.** A single
  bold-link line (`**Methodology:** [docs/methodology/issue_<N>.md](...)
  · [gist](...)`) between the `<!-- clean-result-v4 -->` sentinel and
  `## Takeaways` is the standard orchestrator-appended reader-facing
  pointer to the findings-blind methodology reference, paired with the
  `**Methodology reference:**` link in the `**Repro:**` footer
  (`SPEC.md` § Top-of-body methodology link). It is appended at
  Step 9a-quater AFTER this gate, so a body under critique normally
  does NOT carry it yet — never REQUIRE it, and never flag it as a
  stray element when present (e.g. on a re-critique during a
  same-issue follow-up round). (v3: the sentinel is
  `<!-- clean-result-v3 -->` and the paired pointer is the
  `**Methodology reference:**` row in `## Reproducibility`.)
- The `**Repro:**` / `**Context:**` footer is the last element of the
  body (a bold footer, NOT an H2). (v3: `## Reproducibility` is the last
  H2.)
- The `**Repro:**` footer carries compute + code SHA + pinned artifact
  links (verifier check #7 owns the footer presence). (v3: the three
  boldface subgroup labels `**Artifacts:**`, `**Compute:**`, `**Code:**`
  appear verbatim under `## Reproducibility`.)
- **Complete hyperparameter table in `## Methodology` (v4).** Under v4
  the COMPLETE hyperparameter table lives in
  `## Methodology → **Training:**` (every training + eval + generation
  knob), not slimmed and not split off to a separate doc tier. The lr is
  reconciled against the plan (check 16) and the whole table is
  eyeballed against ground truth at the `**Code:**` SHA — a
  guessed-from-memory value is #489's `lr = 1e-4` class. When the
  orchestrator passed `--methodology-doc`, verifier check 21 asserts the
  body's Training-table rows are a SUBSET of the doc §2 table; a body row
  that is NOT in the doc table is a FAIL (the doc is the canonical
  complete reference). (v3: the body `**Parameters:**` table in
  `## Reproducibility` SLIMS to the LOAD-BEARING subset — base model,
  adapter recipe, lr, steps, seeds, eval rig, N — and the COMPLETE table
  lives in the methodology doc §2, NeurIPS-checklist two-tier split; a v3
  body is NOT FAILed for omitting a non-load-bearing hyperparameter from
  the body table.)
- All URLs permanent: HF Hub `/tree/<ref>` / `@<ref>`, WandB
  `/runs/<id>`, GitHub `/blob/<sha>` / `/tree/<sha>`. Never `main` /
  `master` / `HEAD` (verifier check #8). You confirm no fields are
  written `n/a` when there's an actual artifact that COULD have
  been linked.
- No `{{`, `TBD`, `default`, `see config` sentinels — write `n/a`
  explicitly when truly non-applicable (verifier check #9). `default`
  counts only in placeholder positions (bare `| default |` cell or a
  `label: default` terminator); substantive prose like "default
  assistant" / "default-context" is fine — the default assistant is a
  core experimental condition (#542).
- **Context-footer audit (run-context provenance; all sentinelled
  bodies).** The
  `**Context:**` footer (SPEC.md
  § `**Context:**` row; verifier check 17 covers presence + a lineage
  token — this bullet adds the substantive read: dates real, lineage
  CORRECT) must carry: (a) **real dates** —
  the created date matches frontmatter `created_at`, the run
  date/window is plausible against the events.jsonl timeline; (b)
  **correct lineage** — the `Follow-up to` line matches frontmatter
  `parent_id` / the `**This experiment in context:**` slot's actual
  prior-task citation (a
  fabricated or wrong parent is a FAIL), or says `fresh direction
  (no parent)`; for same-issue follow-up rounds it also names each
  round's `followup_label`; (c) **verbatim prompts** — cross-check the
  quoted originating prompt against frontmatter `origin_prompt` and/or
  the `## Provenance` section in `original-body.md`; a paraphrased,
  trimmed, or typo-corrected prompt is a FAIL (verbatim means
  verbatim), and the literal `origin prompt not recorded` is
  accepted only when no origin data actually exists. Also confirm
  provenance stays CONFINED to this footer — prompt/person attributions
  woven into `## Takeaways` or `## Results` prose violate the "state
  facts, not sources" rule. Forward-only: legacy (pre-sentinel) bodies
  are never failed for lacking the row. (v3: the `**Context:**` row lives
  in `## Reproducibility`; the lineage cross-checks the `**Why:**` slot;
  provenance must stay out of `## Takeaways` / `## Findings` prose.)
- **Reuse-provenance audit (semantic, not mechanical).** When any
  reader-facing claim in `## Takeaways` / `## Results` rests on a
  trained artifact REUSED from a prior issue — a LoRA adapter, merged
  checkpoint,
  training-mix dataset, raw-completion bucket, or `eval_results/`
  JSON produced by a previous `/issue` run rather than freshly
  produced by THIS task — the `**Repro:**` footer
  MUST record one bullet per reused artifact
  naming (a) the producing issue
  (`[#M](https://eps.superkaiba.com/tasks/M)`), (b) the permanent
  HF Hub path (pinned to `/tree/<sha>` or `@<sha>`) or repo-relative
  `eval_results/issue_M/...` path the artifact was pulled from, AND
  (c) a **one-line fitness rationale** stating why this artifact was
  the right one to reuse for THIS result — covering recipe match
  (same base model + training-recipe / hyperparameters the new
  question demands), measurement-regime fit (the artifact's eval
  surface contains the conditions THIS result reads off; for marker
  work, the artifact is not saturated where this read needs headroom
  — source `log P − base ∈ [5,12]` nat per
  `.claude/rules/marker-training-recipe.md`), and required
  conditions present. This is the clean-result side of the positive
  fitness check the planner ran at plan §5 / §10
  (CLAUDE.md § "Reuse existing trained artifacts when fit-for-purpose
  — never reuse a wrong one"); the spec lives in
  `.claude/skills/clean-results/SPEC.md` § `**Artifacts:**`
  reuse-provenance bullet.
  **Triggering reuse:** the body cites a prior issue (`[#M](...)`) as
  the source of a specific artifact OR the `**Repro:**` footer's code /
  artifact links point to a prior issue's HF subdirectory /
  `tree/<sha>` path / `eval_results/issue_M/...` path rather than
  this task's own output. Inspect the `## Goal`
  `**This experiment in context:**` slot
  for `[#M](...)` artifact citations AND the `**Repro:**` footer for
  any HF or `eval_results/` path whose issue number is NOT the current
  task's (e.g. `eval_results/issue_474/...` referenced from a #532 body).
  **FAIL when:** reuse is evident from the body but the
  `**Repro:**` footer has NO reuse-provenance bullet, OR the
  bullet is present but missing any of (a)/(b)/(c) — naming `#M`
  without a fitness rationale is the most common partial form, and
  the rationale is what tells the reader the producing recipe
  matched the new question. Fix list to the analyzer:
  *"add a `- Reused <kind> from [#M](...): <path> — fit: <one line>`
  bullet to the `**Repro:**` footer covering recipe + regime +
  conditions; mirror plan §5/§10's fitness check."* **PASS vacuously**
  when THIS task produced every artifact it stands on (most
  fresh-train experiments — no reused artifact, no provenance bullet
  expected). (v3: the reuse bullet lives under the `**Artifacts:**`
  block of `## Reproducibility`; the citation is inspected in the
  `## What I ran` `**Why:**` slot.)
- **Artifact-path resolution spot-check (semantic).** When the body
  names SPECIFIC artifact paths in the `**Repro:**` footer or in
  `## Methodology` / `## Results` prose — subfolder names
  (`adapters/issue_<N>/<cell>/`), intermediate
  checkpoint or fraction directories (`ckpt_frac0.25/`,
  `checkpoint-<step>/`), specific raw-completion files
  (`<cond>_seed<S>.json`), or a file-count claim ("520 files at
  `<path>`") — spot-check that the listing on the Hub actually
  contains those paths. Use the Python Hub API
  (`huggingface_hub.list_repo_files(<repo>, revision=<sha-or-tag>,
  repo_type=...)`) — NEVER the `hf` CLI, which has no `api` subcommand
  and false-reports "0 files" (see `.claude/rules/upload-policy.md`).
  You don't need to verify every file in a large bucket; check the
  load-bearing path-specific claims — the ones a downstream
  follow-up-proposer or planner would mine as a reuse premise. **FAIL
  when** the body asserts a specific subfolder / checkpoint /
  intermediate fraction exists at a Hub path that the listing does NOT
  contain. Fix list to the analyzer: *"`<path>` claimed in the
  `**Repro:**` footer does not resolve on
  `huggingface_hub.list_repo_files`
  for `<repo>@<revision>`; what the Hub actually carries is
  `<observed>`. Either correct the artifact bullet to match the
  listing, or surface the missing piece as a methodology-correction
  beat inside the relevant `### <result>` (per analyzer.md §
  `**Artifacts:**` grounding rule)."* **PASS vacuously** when the
  artifact bullets stay at the repo level
  (`superkaiba1/explore-persona-space/...`) with no path-specific
  subfolder / checkpoint / fraction names that need resolution. (v3: the
  paths are named under the `**Artifacts:**` block of
  `## Reproducibility` or in `## Findings` / `## Data` prose, and the
  correction beat folds into the relevant `### <finding>`.)
  Closing the door on the #530→#534 false-premise propagation chain
  (2026-06-09) is the point of this lens: an artifact-existence
  claim a downstream task can carry forward should be grounded in a
  real listing, not in plan intent.

### Lens 6 — Voice (research-paper register + bullet/prose register + byte-identical ban)

- **Rule B — research-paper register (v4; SPEC.md § Voice (v4) Rule B).**
  The whole body is written in the concise, precise register of a
  research paper: declarative methods/results prose, every quantity
  DEFINED on first use, no filler / marketing / hype. The bullet-vs-prose
  default is PER SECTION:
  - `## Takeaways` STAYS numbers-first bullets (abstract-style) — FAIL a
    Takeaways written as narrative paragraphs.
  - `## Methodology` is **Methods-section PROSE** — the complete procedure
    as compact declarative paragraphs (with the hyperparameter table +
    verbatim example blocks as data). FAIL a Methodology written as terse
    bullet FRAGMENTS that read as an outline rather than a reproducible
    methods account (e.g. `- lr 2e-6` / `- 3 epochs` standing in for a
    Training paragraph — the hyperparameter TABLE is fine, but the recipe
    PROSE around it must be paragraphs).
  - `## Results` is **Results-section PROSE** per `### <result>` — each
    three-beat (what-is-plotted-EXACTLY → figure → interpretation) a 1–3-
    sentence declarative paragraph. FAIL a result whose beats are reduced
    to terse bullet fragments rather than precise prose.
  - `## Goal` keeps its two compact-prose boldface slots.
  This refines the bullet-default bullet below: bullets are the default
  for `## Takeaways`; `## Methodology` / `## Results` are compact PROSE
  under Rule B. Research-paper register means TIGHT prose, not verbose —
  the conciseness caps (Lens 12) still bind; flag a register violation
  here and a length violation under Lens 12, do not double-count.
- **Bullets are the default for `## Takeaways`; prose for
  `## Methodology` / `## Results` (Rule B).** The v4 register deliberately
  replaced the v2-era wall of UNDISCIPLINED narrative prose, but v4
  Methodology + Results are compact RESEARCH-PAPER prose (Rule B above),
  not bullets. Bold key numbers, front-load the takeaway in
  `## Takeaways` (the NN/g "layer-cake" guidance). FAIL when
  `## Takeaways` carries multi-sentence narrative paragraphs where
  bullets would read better, OR when any section runs a padded ≥4-sentence
  wall in an analytical read (this overlaps Lens 12 Conciseness — flag it
  under whichever you reach first, do not double-count as two blockers).
  (v3: `## Takeaways` / `## What I ran` / `## Findings`; v3 had no Rule B —
  do not apply the research-paper-prose check to a `<!-- clean-result-v3 -->`
  body.)
- `I`, not `we`.
- **Plain academic register in `## Takeaways`** — no lowercase-casual
  voice, no diary framing (the register check itself is Lens 4; this
  bullet is the voice-side cross-reference).
- No fluff transitions anywhere reader-facing: "One more wrinkle:",
  "the buried lede was", "funnily enough", "the real surprise was",
  "the kicker is". (Connective tissue inside `### <result>`
  interpretation prose
  — "Then I tried", "But that didn't replicate", "I expected X — what I
  got was Y" — IS welcome and keeps the per-result story flowing. v3:
  `### <finding>` read prose.)
- Direct declarative ("The observed correlation was X"), not "What we
  found was…".
- No "Standing caveats" section — caveats fold into the relevant
  `### <result>` interpretation prose and/or a `## Takeaways` bullet (v4
  has no `Confidence:` sentence to carry them; v3 too). (v3:
  `### <finding>` read prose.)
- No abandoned-metric prose ("we considered X but went with Y" when
  Y is the only metric reported).
- **Never write `byte identical` or `byte-identical`** anywhere in
  the body (banned 2026-W22, task #454; carried into v3 and v4; flagged
  by
  `audit_clean_results_body_discipline.py`). FAIL on any occurrence
  outside fenced code blocks. Use plain English: "the two files
  matched exactly", "every byte agreed", "no diff between the runs".

### Lens 7 — Statistical-framing rule (absorbed from the retired reviewer)

Project convention: **p-values and sample sizes only in prose**.
Banned in narrative (chart annotations are fine):

- Effect-size names (Cohen's d, η², r-as-effect-size, Δ-framed-as-effect).
- Named statistical tests in narrative prose ("paired t-test",
  "Fisher exact", "Mann-Whitney", "Wilcoxon", "bootstrap test",
  "Kruskal-Wallis"). The test goes in the result-internal "Why this
  test" sentence, defined + justified there. (v3: the finding-internal
  "Why this test" sentence.)
- Power analyses.
- Inline credence intervals (`value ± err`) — chart error bars fine.
- Pre-registration mentions ("pre-registered", "pre-reg", "registered
  hypothesis") in `## Takeaways` / `## Goal` / `## Methodology` /
  `## Results`
  prose. Pre-reg threshold values can sit in the Methodology Training
  hyperparameter table. (v3: `## Takeaways` / `## What I ran` /
  `## Findings` prose; threshold values in the parameters table.)

Flag specific phrases. The audit script catches some of these
mechanically; you catch the ones it misses.

**Dual-DV for content-behavior leakage / implantation (measurement
honesty, FAIL).** When the body's result is a *content* behavior
leakage/implant (sycophancy, refusal, hedging, style, trait — not the
programmatic marker, which has its own three-space recipe), CLAUDE.md
§ Measurement validity requires BOTH DVs reported: (a) the PRIMARY
judge-scored on-policy behavior/agreement rate (the headline number),
and (b) the SECONDARY continuous completion-probability DV
(PREFERRED the teacher-forced FIXED positive-vs-negative completion
margin — fixed answer pools ⇒ no selection bias, #722; the
judged-positive-conditional-mean `log P` (`logp_pos_mean`) is the
selection-confounded opt-in alternative, valid only after it passes
ρ(DV, rate) > 0). FAIL when (i) a cross-condition / install /
dose-matched headline rests on the binary rate alone and that rate is
disclosed saturated (floor/ceiling), with no continuous DV carrying the
comparison (#608's top-band censoring); OR (ii) the body narrates the
completion-probability DV as the construct / headline number, or reports
it without the validation that it tracks the rate (Spearman across cells
with dynamic range). The judge rate stays PRIMARY; the probability DV is
the SECONDARY companion, never narrated as the construct unvalidated.
PASSes vacuously when the result is not a content-behavior
leakage/implant, or when both DVs (+ validation) are reported with the
rate primary. (Mirrors CLAUDE.md § Measurement validity, analyzer.md
gate check 3, interpretation-critic Lens 1, critic Statistics item 10.)

### Lens 8 — Mentor-facing title

The title is the mentor's first read. It MUST state the post-correction
finding, not the methodology-correction story. (Under the v4 spec
methodology corrections fold into the relevant `### <result>`'s
what-is-plotted or interpretation prose, NOT a dedicated
`### Methodology corrections` heading. v3: the relevant `### <finding>`'s
setup or read prose. Only the title check remains here.)

**Title does not lead with mistake/methodology framing.** Read the
title in isolation. FAIL on any of these phrasings (case-insensitive
regex hit OR semantic equivalent):
- "once <noun> (was|were|are) corrected"
- "after fixing", "after the rig was fixed", "after the bug was patched"
- "below the planned <noun>", "above the planned <noun>"
- "but the rig also breaks", "but the <noun> breaks"
- "the null is uninterpretable", "uninterpretable because"
- "regardless of <noun>'s failure", "despite the rig failure"
- "but <noun> also breaks <noun>, so <claim>"

Test: would a domain-peer mentor reading the title alone ask "what did
this experiment FIND?" (good) or "what was the correction story?"
(bad)? Anti-pattern example (FAIL): "Whole-completion loss decouples
source-persona marker firing from bystander leakage once three
training/eval confounds in parent #N are jointly corrected (MODERATE
confidence)" — the "once ... jointly corrected" clause makes the title
about the correction story, not the finding. Good rewrite: "Whole-
completion loss decouples source-persona marker firing from bystander
leakage on a 72-cell recipe sweep (MODERATE confidence)" with the
correction story folded into the relevant `### <result>`'s prose. (v3:
the relevant `### <finding>`'s prose.)

Binding-constraint note: the binding constraint that justifies the
title's confidence level (e.g. "broken in-context sanity check means the
null is uninterpretable") lives in the relevant `### <result>`
interpretation
prose and/or a `## Takeaways` bullet — v4 has no body `Confidence:`
sentence (v3 too). Naming the constraint THERE does NOT count as
title-mistake-framing; the constraint is correctly attributed to the
result/Takeaways, not promoted into the title. (v3: `### <finding>` read
prose.)

### Lens 9 — One takeaway, one figure (per-`### <result>` pairing)

`## Results` is the mentor's primary scan-line. Under the v4 spec each
`### <result>` carries its own inline figure framed by the per-result
three-beat. The shape is: `### <result>` → what-is-plotted →
`![alt](url)` inline
image → blockquote caption → interpretation → (for text results) at most
ONE short excerpt. The systematic per-condition samples + `<details>`
dropdowns live in `## Methodology → **Sample training/evaluation data +
completions:**`, NOT inside each result. (v3: `## Findings`, each
`### <finding>` framed by setup/read bullets, with the systematic samples
in `## Data → ### Generated`.)

The user framing this rule came from (#381, 2026-05-26): *"Basically it
should be more like a story. We have one takeaway, one result, one
figure."* v4 generalises this: one takeaway = one `### <result>` = one
inline figure. (v3: one `### <finding>`.)

**Check four things:**

1. **Every `### <result>` has exactly ONE inline figure.** Enumerate
   each `### <result>` under `## Results`. For each, check that exactly
   one `![alt](url)` image sits inside it, on a line by itself with blank
   lines before and after. FAIL when a result carries zero figures (the
   quantitative claim is visually orphaned) OR carries >1 figure without
   a raw + processed pair justification (Lens 11 exception). Adjacent raw
   + processed image pairs count as ONE figure for this rule. (v3:
   `### <finding>` under `## Findings`.)

2. **Qualitative-result exemption.** Results that report a purely
   qualitative observation — text-sample content, structural claim,
   "the model refused on all but two prompts; the outliers are quoted
   below", "the refusals share the same opening clause" — are exempt
   from the figure requirement. The trigger is QUANTITATIVE prose
   (numbers driving the result's claim). Do NOT flag a qualitative
   result as figure-less.

3. **`## Takeaways` / `## Goal` / `## Methodology` are not results.** They
   set up the
   experiment / state the synthesis; they do not assert per-result
   findings. Even if they contain numbers, those are scope/synthesis,
   not a per-result claim needing its own figure. Do NOT require a
   figure inside `## Takeaways`, `## Goal`, or `## Methodology`. (v3:
   `## Takeaways` / `## What I ran`.)

4. **No `## Figure` H2.** A stray `## Figure` H2 in a v4 body is rejected
   by verifier check 2 as a hard FAIL (v3 too) — that gate fires before
   this lens.
   Lens 9 itself only flags the inline-figure discipline.

**FAIL triggers (any of):**

1. A `### <result>` asserts a quantitative finding AND no inline figure
   anchors it. On FAIL: tell the analyzer to either (i) add an inline
   figure inside the result (per analyzer.md § Step 4), (ii) drop the
   unsupported claim and push it into a different result's prose, or
   (iii) rewrite the result as a qualitative observation.
2. **Figure caption is not in markdown-blockquote form.** Every figure
   caption inside a `### <result>` must wrap in a `> ` blockquote and
   use the form `> **Figure.** *one-sentence lead claim in italics.*
   Remaining caption prose in plain text (≤60 words).` The blockquote
   vertical bar is what visually distinguishes the caption from
   surrounding body prose on the dashboard; without it the renderer
   collapses image + trailing line into the same paragraph and the
   caption reads as continuation of body text. FAIL when a figure has a
   caption (≥10 words below the image) that does NOT start with
   `> **Figure.**`. Also FAIL when an inline figure is missing the
   surrounding blank lines (blank-before-image, blank-before-caption).
   Rule canonicalised in `.claude/skills/clean-results/SPEC.md`
   § "Figure caption shape" + `CLAUDE.md` § Experiment Report Structure.
3. **Text-behavior evidence missing.** For a result whose claim rests
   on model completions, the evidence may live in EITHER: (a) at most
   ONE ≤10-line excerpt INSIDE the result (preceded by a
   subset-disclosure line + a raw-completions link, where the text
   itself IS the result), AND/OR (b) the systematic per-condition
   samples in `## Methodology → **Sample training/evaluation data +
   completions:**` (1 inline example per
   load-bearing condition, labeled cherry-picked/random, + a
   `<details>` block with 3-5 more, + a full raw-completions link). FAIL
   when a text-generation result's claim has NEITHER a result excerpt
   NOR a `## Methodology` Sample-slot example covering its condition. The
   per-condition systematic samples are checked under Lens 10
   (Goal + Methodology completeness);
   here you check that the result's text-behavior claim is anchored
   somewhere a reader can verify. (v3: the excerpt lives inside the
   `### <finding>` and the systematic samples in
   `## Data → ### Generated`.)

   Exemption: findings that explicitly carry a one-line skip note
   (*"(no generation-style outputs in this result; the measurement is a
   teacher-forced log-prob.)"*) — pure activation / probe / cluster /
   linear-fit analyses with no completions to show.

   **Sanitized-evidence carve-out (harmful-content + real-world-corpus rows).** When the
   completions come from a harmful-content corpus (Betley-style EM,
   bad-medical-advice, refusal-bait pools) or a real-world corpus
   (LMSYS/WildChat-class — carries in-corpus jailbreak/explicit rows;
   #1073), the analyzer emits example
   blocks labeled "sanitized for context hygiene": ~15-word excerpts +
   `[truncated — harmful-content row; verify at <path>, row <i>]`
   placeholders, with cherry-picked labels, row indices, and permanent
   raw links kept verbatim (analyzer.md § Content hygiene). Such blocks
   SATISFY this sub-rule and the Lens 10 Sample-slot example check (v3:
   the `### Generated` example check) —
   do NOT FAIL them as missing verbatim samples. If you verify such rows
   yourself, use field-filtered `jq` slices; never load raw rows into
   context (incident: task #537, 2026-06-10).

   Canonical layout + discipline points in
   `.claude/skills/clean-results/SPEC.md`.

**Anti-pattern example (FAIL):** A single `### <result>` reads
*"Source-marker firing rises from 0.07 to 0.83; bystander leakage stays
flat at 0.02; the audit-filter contrast is 41 pts (N=400 per cell)."* —
three quantitative claims crammed into one result, with one figure
showing only the source-marker result. The bystander-leakage and
audit-filter claims are visually orphaned.

**Good rewrite:** split into three `### <result>` sections, each with
its own inline figure (or merge into a multi-panel figure where panel 1
shows source firing, panel 2 shows bystander leakage, panel 3 shows the
audit-filter contrast — and link the same multi-panel figure once,
inside a single result that names the joint finding).

### Lens 10 — Goal + Methodology completeness (capsule trio + subset disclosure + link liveness + the complete hyperparameter table)

The `## Goal` + `## Methodology` sections make "what was the question and
what exactly did it train / eval / generate
on?" answerable without leaving the body. `## Methodology` carries the
data slots `**Training:**` (with the complete hyperparameter table) /
`**Evaluation:**` / `**Sample training/evaluation data +
completions:**` (verifier check 18 owns the Training table + a Sample
slot with a pinned link; this lens is the substantive
read). The OLD eval-probe-descriptions lens (was Lens 10 in v2) is
ABSORBED here as the Evaluation capsule trio. (v3: the `## Data` section
with three required H3 subsections in order — `### Trained on` /
`### Evaluated with` / `### Generated`, presence + order + per-subsection
complete-artifact link gated by verifier check 18.)

**Check 1 — capsule trio answerable from `**Evaluation:**`.** The
eval capsule (≤100 words) must answer all three Model-Cards questions:
- **identity** — which probe set / benchmark / question bank, named.
- **why chosen** — why THIS probe set for THIS Goal (e.g. "matched to
  #498's eval surface so the base baseline is comparable").
- **preprocessing** — how the probes were prepared (system-prompt
  prefix per context, deterministic regeneration from a seed, no
  preprocessing beyond X).
FAIL when any of the three is unanswerable from the capsule. This
absorbs the multi-probe rule: when the body uses ≥3 distinct probe
framings / judge prompts / measurement conditions, the `**Evaluation:**`
capsule (or its example block) must enumerate them — name, an
example probe verbatim, and the PASS/FAIL rubric criterion in one
sentence — so a result that references "framing #5" resolves. FAIL
when the body references probes by number / opaque name in `##
Results` WITHOUT the enumeration in `**Evaluation:**`. (Dormant for
single-probe bodies. v3: the `### Evaluated with` capsule, references in
`## Findings`.)

**Check 2 — required capsule content (composition facts).** Facts that
used to hide in prose are mandatory in the relevant capsule:
- `**Training:**`: positives:negatives ratio, persona panel, row
  counts per type, completion provenance (on-policy tier / canned /
  published-corpus-verbatim per `.claude/rules/on-policy-completions.md`
  + `.claude/rules/contrastive-negatives.md`).
- `**Sample training/evaluation data + completions:**`: which conditions
  produced completions, N completions.
FAIL when a behavior-implantation body's `**Training:**` capsule omits
the ratio / panel / provenance (these are the data-realism + contrastive
caveats the reader needs). (v3: the `### Trained on` and `### Generated`
capsules.)

**Check 3 — subset disclosure present.** Verifier check 19 owns the
mechanical "every example block is preceded by a subset-disclosure
line" check; here you confirm the disclosure is HONEST (the "5 of 2,000
rows, random sample" actually describes the block, "cherry-picked for
illustration" is used when the rows were hand-picked). FAIL on a
mislabeled disclosure (e.g. "random sample" on rows that are obviously
the most extreme firings).

**Check 4 — link liveness.** Each data slot carries ≥1 pinned
complete-artifact link (HF Hub `/tree/<sha>`, WandB `/runs/<id>`,
GitHub `/blob/<sha>`) OR an explicit `n/a — <reason>` line (verifier
check 18 owns presence; check 8/8b owns permanence + same-repo
existence). Spot-check that a load-bearing link actually resolves —
especially a `**Training:**` / `**Sample ...:**` HF path; a dead Hub
path here is the same false-premise class as Lens 5's artifact-path
spot-check. FAIL on a complete-artifact link that does not resolve, or
a slot that links only an AGGREGATE (judge JSON) where the raw
text-level artifact is what the `**Sample ...:**` slot requires (this
overlaps the
Lens 11 judge-artifact rule; flag under whichever you reach first). (v3:
each `### Trained on` / `### Evaluated with` / `### Generated`
subsection; the raw text-level artifact is what `### Generated`
requires.)

**Check 5 — n/a slots are explicit.** A data slot that does not
apply (eval-only run → `**Training:**`) must carry an
`n/a — <reason>` line, never be silently omitted. Verifier check 18
mechanically requires this; confirm the reason is real (e.g.
`n/a — no training in this task (eval-only headroom probe)`). (v3: the
`### Trained on` subsection.)

**Check 6 — methodology-doc spot-check (when `--methodology-doc`
passed).** When the orchestrator passed the worktree methodology-doc
path, open the doc's §2 Hyperparameters table and sanity-check it is
COMPLETE (every training + eval + generation hyperparameter, each with
a Source column) and that the body's Methodology Training table is a
SUBSET of it (verifier check 21 mechanizes the subset assert; this lens
is the completeness read on the doc §2 table itself). FAIL when the
doc §2 table is obviously incomplete (a load-bearing knob the body or
plan names is absent) — the doc is the canonical complete reference. The
check is skipped when no `--methodology-doc` was passed (pre-merge the
doc lives only on the issue worktree branch). (v3: the body's slimmed
Parameters table is checked as the subset.)

**Check 7 — `## Methodology` is SELF-CONTAINED (v4 Rule A; semantic).**
SPEC.md § `## Methodology` (v4) Rule A requires the Methodology body to
read like a research-paper Methods section: a reader understands HOW
every reported result was produced WITHOUT following a link to another
issue. When this experiment REUSED an artifact from a prior issue (a
trained adapter, persona-vector bank, behavior direction, leakage cells,
dataset, base-rate / propensity measurement), the Methodology body MUST
WRITE OUT THE FULL PRODUCTION PROCEDURE of that artifact inline as
primary method (data source + realism tier, construction recipe,
training recipe + hyperparameters, measurement). **FAIL when** the
`## Methodology` body DEFERS a load-bearing method to another issue —
phrases like `reused from #M` / `see #M (for the recipe)` /
`as in #M` / `methodology in #M` / `same setup as #M` standing IN PLACE
OF the actual recipe (a Design/Training/Evaluation/Data-extraction slot
that names `#M` instead of spelling out what was done). The fix list to
the analyzer: *"inline the full production recipe of the reused artifact
from #M's `## Methodology` / `docs/methodology/issue_<M>.md` into the
relevant Methodology slot as primary method; move the `#M` citation to
the `**Repro:**` footer reuse-provenance bullet."* **Do NOT FAIL** the
correct pattern: the `**Repro:**` footer reuse-provenance bullet (Lens 5)
naming `#M` + the pinned path + a one-line fitness rationale is REQUIRED
and CORRECT — Rule A moves the METHOD into the body but keeps the
PROVENANCE in the footer; a `#M` citation in the footer (or a single
descriptive sentence in the body acknowledging the artifact was reused
WHILE STILL spelling out its production recipe) is not a violation. A
`#M` link in the `## Goal` `**This experiment in context:**` slot is also
fine (that slot may cite prior tasks). **PASS vacuously** when THIS task
produced every artifact (no reuse → no deferral risk). (v3: this check is
N/A — v3 bodies kept reuse provenance inline by the older pattern and are
grandfathered; do not apply Rule A to a `<!-- clean-result-v3 -->` body.)

### Lens 11 — Underlying data alongside every aggregate (figures + prose + per-cell artifacts)

The broad rule: a result that reports an AGGREGATE statistic MUST also
expose the low-level per-unit data behind it, and every processed /
derived / aggregated artifact MUST have its less-processed counterpart
alongside. The reader should see the DATA, not only the number computed
from it. Concrete checks:

0. **Low-level data plot behind every aggregate figure (the broad
   parent).** Walk every `![alt](url)` inside `## Results`. For each
   figure whose alt text / caption / surrounding prose reports an
   aggregate statistic — a correlation ρ shown as a forest-plot point, a
   mean / effect size shown as a bar, a p-value, an effect summary — a
   LOW-LEVEL per-unit plot of the data behind it (the scatter the ρ
   summarizes, a strip / swarm / jittered per-point view behind the
   group-difference bars, the unbinned counterpart of a binned view)
   MUST be embedded inside the SAME `### <result>`. FAIL when an
   aggregate figure carries no underlying-data view AND the result
   states no exemption. Exemptions (accept when the body says so in
   interpretation
   prose or alt text): the result's primary figure ALREADY is the
   per-unit view (a raw scatter needs no second scatter); N is so small
   the figure already shows every point; or the aggregate has no
   meaningful per-unit decomposition (a single scalar). This check is the
   PARENT of checks 1–2 below — those handle the transformed-figure
   special case (raw vs processed); this one handles ANY aggregate, even
   an untransformed one (a bare bar chart of means with no per-point plot
   still FAILs here). Judgment call (LM): there is no reliable alt-text
   keyword for "this is a forest plot / bar of an aggregate", so read the
   figure + caption + what-is-plotted/interpretation prose to decide
   whether the figure
   reports an aggregate vs already shows the per-unit data — do NOT FAIL
   a figure that already IS the scatter / per-point view. (v3:
   `## Findings`, `### <finding>`, setup/read prose.) Mechanical
   backstop: `verify_task_body.py` check 31 WARNs when a committed
   `figures/issue_<N>/*per{context,unit,cell}*` PNG at a body-cited
   figure SHA is unreferenced by any body image URL (task #1011,
   incident #928) — a pre-gate nudge only; this lens remains the
   substantive owner.
1. **Figures (transformed special case).** Every figure that plots a
   residualized / partialled / binned / log-transformed / normalized
   quantity has its raw counterpart embedded inline inside the same
   `### <result>` (raw first, then processed; both inline `![alt](url)`
   images, blank lines around each). Walk every `![alt](url)` inside
   `## Results`. For each, read the alt text + caption for processing
   keywords (`residualized`, `partialled`, `partialed`,
   `length-controlled`, `binned`, `aggregated`, `normalized`, `centered`,
   `de-trended`, `rank-residualized`, `log-`). If present, look for a raw
   sibling under the same result. FAIL if absent, unless the body
   explicitly justifies the omission (e.g., "raw and processed are
   visually identical because the length partial only re-scales the
   x-axis"). (v3: `### <finding>` under `## Findings`.)
2. **Prose statistical claims.** When the body says "X does not survive
   controlling for Y" / "the partial collapses" / "the residualized
   correlation is" / "the length-controlled value drops to", the same
   sentence MUST quote the RAW point estimate too (raw ρ / r / Δ / rate
   with N), not just the controlled value. FAIL when only the controlled
   value appears.
3. **Aggregated metrics → per-cell artifact link.** Walk
   the `**Repro:**` footer (and `## Methodology`). When the body's
   claim rests on an aggregated metric (per-condition pass-rate,
   per-domain mean, per-seed mean), the body MUST link to BOTH the
   aggregated JSON / summary CSV AND a per-cell file (the per-seed /
   per-condition / per-persona / per-probe table the aggregation
   collapsed). FAIL when only the aggregated artifact is linked.
   Permanent URLs only (the existing `verify_task_body.py`
   URL-permanence check applies to the per-cell link too). (v3:
   `## Reproducibility` § Artifacts and `## Data`.)
4. **Judge-scored claims → raw completions + judge prompts.** When the
   body cites Claude-judge pass-rates / scores, the body MUST link
   (in `## Methodology → **Sample training/evaluation data +
   completions:**` and/or the `**Repro:**` footer) to BOTH
   the raw model completions AND the raw judge prompts + verdicts (not
   only the per-condition aggregate). The cherry-picked /
   qualitative-data-link rule (Lens 9 + verifier checks 10/11) covers
   the figures-of-text instance; this lens extends it to the judge
   artifact layer. (v3: `## Data → ### Generated` and/or
   `## Reproducibility`.)

Checks 1–4 are dormant for bodies that only present raw, untransformed
quantities (most direct-eval runs with no partialling / aggregation).
Check 0 (the broad parent) still fires whenever a result reports an
aggregate statistic at all — including a baseline / replication run that
shows a bar of means or a correlation point — UNLESS the figure already
shows the per-unit data or a stated exemption applies.

**Anti-pattern (FAIL):** A `### <result>` says *"raw association does
not survive controlling for prompt length (collapses to p=0.87, N=48)"*
+ embeds only the length-residualized scatter, no raw scatter inside the
same result, no raw point estimate in the prose. Reader cannot tell
whether the partial collapsed a real effect or shrank noise, which
direction outliers go, or whether outliers drive the controlled value.

**Good rewrite:** *"raw association (Spearman ρ = +0.29, p = 0.048,
N=48) does not survive controlling for prompt length (collapses to
p=0.87, N=48)."* + raw scatter embedded first, then residualized scatter
on the next line inside the same result. Same pattern at the artifact
layer: link both `correlation_results.json` (aggregated) and a
per-persona table (the per-row input that the partial consumed) in
`## Methodology` / the `**Repro:**` footer. (v3: the same finding;
`## Data` / Reproducibility § Artifacts.)

See SPEC.md § per-result skeleton points 4–5 (low-level data plot behind
every aggregate + raw alongside processed) for the canonical rule. (v3:
§ per-finding skeleton.)

### Lens 12 — Conciseness (word-cap adherence + bullets-over-prose)

The v4 redesign (like the v3 one before it) replaced the v2-era
160-320-line prose-heavy body with
bullet-first, hard-capped sections. The mechanical caps live in the
verifier (check 20); this lens is the LM judgment that the mechanical
caps + the bullets-over-prose register are actually honored. (The story-
arc / setup-read narrative shape that the retired Lens 12 owned moved to
Lens 3's what-is-plotted/interpretation–figure pairing bullet.)

Check four things:

1. **Per-result prose stays inside the cap.** Verifier check 20 hard-
   FAILs a `### <result>` whose prose (excl. caption / tables / code /
   `<details>` bodies) is ≥180 words and WARNs at ≥120. Confirm the
   verifier ran; if a result is at 120-179 words (WARN) AND reads
   padded — narrative where 2 bullets would do — flag it as a Lens 12
   tightening request (not a standalone blocker unless ≥180). (v3:
   `### <finding>`.)
2. **Bullets are the default; prose only for 1–3-sentence causal
   chains.** FAIL when `## Results` / `## Methodology` carry
   multi-sentence narrative paragraphs that should be bullets, or a
   single analytical paragraph runs ≥4 sentences. (This overlaps Lens 6
   Voice — flag under whichever you reach first; do not double-count as
   two blockers. v3: `## Findings` / `## What I ran`.)
3. **Takeaways bullets ≤30 words; figure captions ≤60 words.** Verifier
   check 20 WARNs over both caps and hard-FAILs a v4 Takeaways bullet
   ≥100 words. Confirm the WARNs were addressed; a
   runaway Takeaways bullet (a paragraph in bullet's clothing) or a
   60+-word caption that buries the lead is a Lens 12 finding.
4. **Total-prose budget (WARN-only).** The verifier WARNs when
   Takeaways + Goal + Methodology + Results prose exceeds ~800 words +
   250 per
   live follow-up round beyond the first. This is intentionally NOT a
   hard gate (a multi-round consolidated body must not be forced to
   delete live results — the per-result ≥180 FAIL is the hard cap).
   When the total-prose WARN fires, check the body used the round-
   compression hygiene (superseded results collapsed into a
   `<details>Superseded by round N</details>`; absorbed results
   compressed to heading + figure + ≤2 bullets) rather than carrying
   dead narrative; flag a body that blew the budget on padding, not on
   genuine multi-round results. (v3: Takeaways + What I ran + Findings
   prose; superseded/absorbed findings.)

The lens is mostly mechanical-pre-pass-backed; your value-add is the
register call (bullets vs prose) and catching padding that sits just
under the hard cap.

**Anti-pattern (FAIL):** A `### <result>` runs 210 words of narrative
prose (check 20 hard FAIL) restating the figure in sentences where one
what-is-plotted bullet + one interpretation bullet would carry it;
`## Takeaways` has a
55-word bullet that is really two sentences.

**Good rewrite:** the result's prose drops to a 1-sentence what-is-plotted
+ 2 interpretation bullets (≤120 words total); the Takeaways bullet splits
into two ≤30-word bullets, each numbers-first.

### Lens 13 — Planned-vs-actual coverage (scope-shrinkage discipline)

Post-mortem trigger: **task #391, 2026-05-27** — the plan committed to
**3 swept factors (A, C, D)**; cell `10111` (the C-flip cell) silently
failed during the original run and was never re-attempted after the
round-4 padding fix landed. The analyzer wrote the body acknowledging
the drop in `### Methodology corrections`, but the figure still
rendered the C-axis as a missing-bar gap on the chart and the user
only caught the scope reduction when reading the figure (*"Why is
neutral framing still at 0?"*). Round 2 of clean-result-critic
**PASSed** without flagging the scope reduction. This lens is the
gate that should have caught it.

The pattern is **scope-shrinkage-without-explicit-flag**: the plan
declares N planned conditions / cells / factor flips, the run delivers
M < N, and the body equivocates between the original N and the
delivered M across the title, `## Takeaways`, `## Results` prose, and
figures. Reader walks away with the impression the experiment tested
N conditions when it tested M. Under the v4 spec the scope-correction
prose folds into the relevant `### <result>` — there is no dedicated
`### Methodology corrections` heading to collect it. (v3: `## Findings`
prose, folding into the relevant `### <finding>`.)

Read the plan body before this lens fires:

```bash
# Resolves to tasks/<status>/<N>/plans/plan.md (symlink to highest v{K}.md).
plan_path=$(uv run python scripts/task.py find <N>)/plans/plan.md
cat "$plan_path"
```

Enumerate the plan's planned conditions / cells / factor flips. Heuristics
for finding them in the plan:

- **§4 Conditions table** (or whatever Markdown table lists per-condition
  rows) — count rows excluding rows explicitly labeled as `CONTROL` /
  `BASELINE` / `(not a factor flip)` / `(control, not counted in denominator)`.
- **§5 Sweep design** — count enumerated factor names (often single-letter
  `A`, `B`, `C`, `D`, `E` flips against an anchor cell, plus per-factor
  English labels).
- **§1 Hypothesis** — the phrase "**N of M** ... will" / "**≥K of M**
  ... clear" / "≥K of M factors show ..." commits the plan to the M
  denominator. The plan's median-prediction numerator (e.g., "Median
  prediction: 3 of 3") is also informative.
- **§0 Headline / Plan summary** — the "**N of N selectivity knobs**" /
  "**M matched factor flips**" framing.
- **Denominator-convention notes** — many plans include a `Note on the
  denominator` paragraph that explicitly commits to a specific M for
  the headline count, separating sweep factors from CONTROL rows. When
  this paragraph exists, use IT, not any contradictory earlier
  enumeration, as the authoritative planned denominator.

Then read the body's `## Takeaways` + `## Results` (each `###
<result>`) and the `## Methodology` Training table
for the **actual** delivered conditions / cells. Any scope-correction
prose lives inside the relevant `### <result>` under the v4 spec. (v3:
`## Findings` (each `### <finding>`), the `## Data` + `## Reproducibility`
/ Parameters table; correction prose in the relevant `### <finding>`.)

**Check three things:**

1. **No silently dropped planned condition.** Enumerate the planned
   conditions. If ANY planned condition is NOT mentioned anywhere in
   the body (`## Takeaways`, any `### <result>`, `## Methodology`,
   the `**Repro:**` footer), that's a silent drop. **FAIL** with:
   *"Plan committed to {factor X} but it appears nowhere in the body —
   name it in `## Takeaways` / the relevant `### <result>` AND document
   the drop in that result's what-is-plotted or interpretation prose."*
   (v3: `## Data`, Reproducibility / Parameters; the relevant
   `### <finding>` and its setup or read prose.)

2. **Denominator revision is consistent across the body.** If the body
   names a missing condition anywhere, the headline denominator MUST
   be revised consistently in `## Takeaways`, every relevant
   `### <result>` prose, any figure caption, and any per-factor table
   caption. **FAIL** when the body still uses the ORIGINAL plan
   denominator in any reader-facing surface after acknowledging the
   drop. (v3: `### <finding>` prose.) Examples:
   - Plan said "3 swept factors (A, C, D)"; one `### <result>` says
     "the C-axis cell never trained, so 2 of 3 testable"; another
     result still reads "the 3-factor sweep showed no clean
     decoupling" → FAIL.
   - Plan said "5 sources × 4 seeds = 20 cells"; body says "1 cell
     crashed with EDQUOT, recovered 19"; another section still says
     "across the 20-cell sweep" → FAIL.
   - "1 of 2 testable factors clears the selectivity CI, n=3 sources ×
     1 seed" with the result prose documenting the C-axis drop and
     all denominator references revised to "2 of 2 testable" → PASS.

3. **Figures don't render misleading zero bars for missing conditions.**
   When the body names a missing / silently-dropped condition,
   inspect every figure (alt text + caption) for that condition's label.
   Two acceptable shapes:
   - **OMIT** the missing condition from the chart entirely (chart shows
     only the conditions with data; caption names what was tested).
   - **EXPLICITLY LABEL** the missing condition as "N/A — not tested"
     or "data not collected" in the figure (NOT rendered as a zero bar
     with no annotation; the reader should never have to hunt through
     the prose to understand why a bar is missing).

   **FAIL** when a figure renders the missing condition as a zero-height
   bar, missing point, or visual gap WITHOUT in-figure annotation
   explaining it. Example: a per-factor selectivity chart with bars for
   factors A and D but a blank/zero gap where factor C should be, no
   "N/A" label in the chart, alt text doesn't call it out → FAIL.

The lens **PASSes vacuously** when the plan has no enumerable planned
conditions OR when the run delivered all planned conditions cleanly
(no scope shrinkage to discipline).

**On FAIL, your minimal-necessary-fix list to the analyzer:**

- For check 1: *"Plan §{X} committed to {N} planned {conditions}; the
  body names only {M}. Add a scope-correction paragraph inside the
  relevant `### <result>` documenting why {missing list} were not
  delivered, OR delete the `## Takeaways` / result claim that implies
  they were tested."* (v3: `### <finding>` / finding claim.)
- For check 2: *"The 'X of N' denominator (N=plan denominator) is
  inconsistent with the scope-correction prose elsewhere in the body
  (only M < N testable). Revise the result denominator to 'X of M
  testable' and update `## Takeaways` + figure captions to match."* (v3:
  the finding denominator.)
- For check 3: *"Figure {file} renders missing condition {C} as a
  zero/blank bar. Regenerate to either omit {C} from the x-axis or
  label its position 'N/A — not tested', and update the alt text
  + caption to call out the omission explicitly."*

### Lens 14 — Binding-concerns audit (composed onto Lens 13 by task #455)

Adopted **2026-05-31** by task #455, ON TOP of main's existing
PASS+CONCERNS auto-advance + mechanical-contract-strip policy
(neither is weakened). The lens is the LM-side companion to
`verify_task_body.py`'s `check_concerns_audit` (Lens 14): the verifier
mechanically pins the surface check, this lens does the substantive
read.

**Step 0 prerequisite** — fetch the canonical concerns ledger before any
other lens fires:

```bash
uv run python scripts/task.py list-concerns <N> --open-only --json
```

For each currently OPEN binding concern (severity `BLOCKER` or `CONCERN`,
latest event `raised` or `verified-open`), verify the body acknowledges
it via ONE of these mechanisms (per the v4 spec — there is NO
`### Methodology corrections` heading to collect them; correction prose
folds into the relevant `### <result>`; v3: the relevant `### <finding>`):

- **Inside any `### <result>` (or a `## Takeaways` bullet)** —
  what-is-plotted or
  interpretation prose that names the concern_id (substring match) and
  either
  describes the implementer fix OR explicitly bounds the interpretation
  by it. (v4 has no `Confidence:` sentence — the binding constraint that
  used to ride there now lives in the relevant result's interpretation
  prose
  and/or a Takeaways bullet, so this is where v4 bodies acknowledge a
  concern. v3: inside any `### <finding>`, setup or read prose, the
  binding constraint in the relevant finding's read prose.)
- **As an `<!-- concern-deferred: <id> -->` HTML comment** anywhere in
  the body — records explicit user deferral via
  `task.py defer-concern --by user`. Treat the deferral marker as
  acknowledgement-by-reference; do NOT also require prose acknowledgement.

(Legacy / v2 bodies additionally accept acknowledgement inside the
`Confidence:` rationale sentence — that sentence does not exist in v3/v4.)

NIT-severity concerns do NOT block this lens; surface them as
informational only.

**FAIL when**: a `BLOCKER` or `CONCERN` is open in `concerns.jsonl` and
NONE of the three acknowledgement mechanisms above name the concern_id.
The mechanical verifier (Lens 14 in `verify_task_body.py`) will already
have FAILed in this case — if you see a verifier Lens-14 FAIL, the
correct verdict is `FAIL — Lens 14 binding-concerns audit`. The
LM-side judgment value-add is calling out *substantive* acknowledgement
that fools the substring match (e.g., the body discusses the underlying
issue without naming the concern_id) → that is a CONCERNS bullet
asking the analyzer to add the kebab-case id to the prose, NOT a
standalone FAIL.

**Composition note**: this lens does NOT override main's mechanical
strip. A `marker-shape` / `smoke-run-missing` FAIL still strips per the
existing `mechanical_contract_only_strip` rule. The binding-concerns
check runs AFTER the strip: if the strip would have promoted the
verdict to PASS but `task.py list-concerns --open-only --json` returns
non-empty binding concerns, this lens (and the orchestrator's
post-strip concerns check, per `agree_rule`) keeps the verdict from
auto-advancing.

See `workflow.yaml § concerns_protocol` for the full severity tier
mapping and reviewer round protocol; see Lens 13 (`Planned-vs-actual
coverage`) above for the orthogonal scope-shrinkage check that
sometimes co-fires.

### Lens 15 — Headline must not rest on a contaminated / failed-data-gate arm

Post-mortem trigger: **task #407, 2026-06-01** — the clean-result was
titled and framed "content-agnostic gating" off an arm whose training
data was contaminated (stale paraphrases) and whose multiple-choice
numbers were inflated by a string-lookup bug. The user had to
interrogate it repeatedly ("how did taught-wrong-info get ~100%?" /
"mark it as bugged") before it was demoted.

Read the body for any disclosed data-validity failure on an arm /
condition: contaminated or stale training pool, a Phase-0 / `K1` / data
gate the arm failed, a wrong base prior, a string-lookup-inflated
metric, or any "this arm is bugged / not trustworthy" admission. If such
a disclosure exists, the H1 title AND the `## Takeaways` / `## Results`
headline result MUST NOT rest a positive claim on that arm (v3:
`## Findings` headline finding). **Hard
FAIL** when they do — the minimal-necessary-fix is to re-anchor the
title/headline on a surviving clean arm, or to retitle the body as
"bugged" / inconclusive if no clean arm carries the claim. The lens
**PASSes vacuously** when the body discloses no data-validity failure on
any arm.

