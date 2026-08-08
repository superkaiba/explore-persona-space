# v2 experiment report — the official result template + authoring notes

**The official result template** (Thomas, 2026-07-30 — codified from the #1092
direct-vs-averaged-prefix-map writeup; supersedes the 2026-07-14 shape, which
kept one monolithic `## Methodology:`, a `**Plot:**` label per result, and a
closing `## Next steps:`). The v2 clean-result is a **report**: agents produce
the factual skeleton — Motivation / Methodology (shared) / Results-as-plots
with a result-specific `**Methodology**` block per result — that is verified
for accuracy and completeness; **Thomas alone writes the claims: the
`# Result:` title, the TLDR, every per-result `**Takeaways**` block, and
Conclusion and next steps.** No agent asserts a conclusion.

This file is the canonical skeleton every v2 producer + verifier reads:

- `methodology-writer` (REPORT MODE) authors Motivation + `## Methodology
  (shared)` + one result-specific `**Methodology**` block per planned figure.
  Metric definitions + rationale live INSIDE `## Methodology (shared)` (as its
  final `**Metrics:**` block) or inside the single result's `**Methodology**`
  block that uses them — there is no separate `## Metrics:` H2.
- `plotter` produces the figures + factual what-is-plotted captions; the
  orchestrator folds each caption into that result's `**Methodology**` block.
- The orchestrator assembles the full report, leaving TLDR / every per-result
  `**Takeaways**` block / Conclusion and next steps as placeholders.
- `methodology-critic` traces every Motivation / Methodology claim — the
  shared section AND every per-result `**Methodology**` block, including the
  embedded metrics rationale — to ground truth.
- `report-verifier` runs the final completeness + interpretivity + recomputation
  gate and `scripts/verify_report.py --mode generation`.

The report body carries the `<!-- report-v1 -->` sentinel and lives in the task
`body.md` exactly like a markdown clean-result body.

**Two-document output (2026-07-30).** The body is the SUMMARIZED layer — one
headline figure per result, compact methodology. It is paired with a
**detailed companion writeup** at `docs/reports/issue_<N>_detailed.md`
(committed to the issue branch, SHA-pin-linked from the body's
`**Detailed writeup:**` line right after the sentinel) that carries the full
detail: the unabridged Methodology (complete hyperparameter table, worked
examples), EVERY figure view the plotter produced (aggregate, per-unit, raw,
alt-groupings — each with its factual caption + SHA-pinned image), and any
extra tables. The detailed doc is 100% agent-written mechanical assembly —
the interpretivity rule applies to all of it; it carries NO Takeaways / TLDR /
Conclusion (claims live only in the body, Thomas's voice). See § The detailed
companion writeup below.

---

## The skeleton (exact)

Five H2 sections, in this exact order: **Motivation → TLDR → Methodology
(shared) → Results → Conclusion and next steps.**

```markdown
# Experiment: <one-line question>
<!-- report-v1 -->

**Detailed writeup:** <SHA-pinned https://github.com/<owner>/<repo>/blob/<40-hex-sha>/docs/reports/issue_<N>_detailed.md>

## Motivation

- <the prior result / context this experiment builds on — with the concrete
  numbers of the prior finding where relevant (e.g. "an earlier experiment
  found X: R^2 ~0.8 at layer 18")>
- <the question this experiment tests, stated as a question or "I wanted to
  test whether ..." — NEVER an asserted answer>

## TLDR

*(Thomas fills in)*

## Methodology (shared)

Only what is shared ACROSS the results — each result's own recipe lives in
that result's `**Methodology**` block below, so this section stays compact:

- **Model:** <model id>
- **Datasets / corpus:** <per dataset: a bold name, the source (linked), row
  counts, ONE fully worked verbatim example (prefix / query / answer), and a
  dashboard link to the full table>
    - **Splits:** <training / validation — naming EXACTLY what is selected on
      it (layer, hyperparameters) — / evaluation, with the CI recipe (e.g.
      "1000 prompts with 95% bootstrap CIs over test contexts")>
- **Computed quantities:** <how each shared vector / DV is computed, with the
  exact options enumerated and the default marked>
- **Predictors / conditions:** <each fitted model or experimental arm used
  across results: architecture + hyperparameters — every load-bearing value
  copied from ground truth (config / run_result.json / the training script at
  its SHA)>
    - **Baselines:** <each baseline + the worry it addresses, in the shape
      "one worry here is <X>; test: <Y>">
    - **Sanity checks:** <e.g. train on permuted pairings>
- **Metrics:** <definition of each SHARED reported metric + WHY it was chosen
  over the alternatives, grounded in the plan / Goal — NEVER in a measured
  value. A metric only one result uses may live in that result's
  `**Methodology**` block instead.>

## Results

### <plot name 1>

<optional 1-3 sentences of connecting narrative: what this result tests and
how it follows from the previous one — question-framed at generation time, NO
interpretation, NO "this shows".>

**Methodology**

- <the result-specific recipe: what was computed / fit / measured for THIS
  result — counts, conditions, any result-local metric + its rationale>
- <what is plotted EXACTLY — axes (with units), groupings, color/series coding
  (e.g. "the blue bars use the last prompt token, the orange bars the mean
  over prompt tokens"), the eval N>

![<plain-English figure description>](<SHA-pinned raw.githubusercontent.com URL>)

**Takeaways**

*(Thomas fills in)*

### <plot name 2>

...

## Conclusion and next steps

*(Thomas fills in)*
```

## The filled-in form (what the report looks like after Thomas's pass)

When Thomas writes the TLDR he:

- **retitles the H1 to `# Result: <claim>`** — the report's main claim, which
  MAY run to several sentences (optionally with the trailing
  `(HIGH|MODERATE|LOW confidence)` tag the dashboard parses),
- **retitles each Results subsection to `### Result <n> — <one-line claim>`**
  (the `### Result <n>: <claim>` separator is equivalent) — the claim that
  plot supports (sub-numbering like `Result 4.5` is fine for a follow-on
  control of the same result),
- **fills every `**Takeaways**` block** with numbers-first bullets (the
  statistic + CI inline; sub-bullets carry supporting numbers, caveats, and
  "still unsure about X" honesty),
- **fills `## TLDR`** — one numbers-first claim bullet per headline finding,
  each with the key statistic inline and sub-bullets for the supporting
  evidence,
- **fills `## Conclusion and next steps`** — what the results mean taken
  together (which objects to keep using / studying), then the concrete next
  experiments, including `(running)` markers for work already in flight.

Worked exemplar of the filled form:
`.claude/skills/issue-v2/exemplars/issue-1092-filled-report.md`.

---

## Authoring notes

### Title convention

At generation time the H1 is `# Experiment: <one-line question>` — agents have
no finding to claim, so a claim-shaped title would violate the interpretivity
rule. Thomas retitles to `# Result: <claim>` (one line up to several sentences)
when he fills the TLDR. `verify_report.py` enforces this: generation mode
requires the `Experiment:` prefix; promote mode accepts `Result:` (preferred)
or `Experiment:` (not yet retitled). Confidence lives only in the H1 trailing
tag, added by Thomas — no agent-written section ever contains a `Confidence:`
line.

### Two verify modes

`scripts/verify_report.py` runs in two modes; the report must pass the relevant
one at each gate:

- **`--mode generation`** (at report assembly, before parking at
  `awaiting_promotion`): the `## TLDR` and `## Conclusion and next steps`
  placeholders MUST be intact (`*(Thomas fills in)*`), every Results
  subsection's `**Takeaways**` block must be exactly the placeholder too, and
  every Results subsection carries exactly one `**Methodology**` block. The
  interpretivity / lexicon checks run on the AGENT-written sections —
  Methodology (shared) and Results (Motivation is exempt — hypothesis framing
  is allowed there). This is the gate `report-verifier` runs.
- **`--mode promote`** (when Thomas runs `task.py promote`): the `## TLDR`
  placeholder MUST now be FILLED (Thomas wrote it). Thomas's prose — TLDR,
  Conclusion and next steps, and the Results takeaways / claim-headings he
  filled in — is NEVER lexicon- or interpretivity-checked; only
  `## Methodology (shared)` (pure agent prose in both modes) stays under the
  lexicon scan at promote time.

**Grandfathered pre-2026-07-30 report-v1 bodies** (a monolithic
`## Methodology:` + `## Next steps:`, `**Plot:**` labels, `**Takeaways:**`
with a trailing colon, no per-result `**Methodology**` block) still verify:
`verify_report.py` normalizes the old H2 names to the new canonical ones,
accepts the trailing-colon bold labels, and only WARNs on a missing
per-result `**Methodology**` block at promote time (generation mode requires
it — freshly assembled reports follow this template).

### The interpretivity rule (agent-written sections only)

The whole point of v2 is that agents do NOT interpret results. The line:

- **Hypothesis-to-be-tested framing is ALLOWED.** Motivation states what the
  experiment tests and what is hypothesized, framed as a question or a plan:
  - "We test whether context geometry predicts fine-tuning leakage."
  - "I wanted to test whether a similar simple mapping (linear or non-linear)
    exists between context vector and answer vector."
  - "H1: cosine distance between personas predicts transfer strength; H2: it
    does not."
- **Asserted conclusions are BANNED in agent prose.** No agent-written section
  states the answer, hedges toward it, or reads a measured value as a finding:
  - "Context geometry predicts leakage." (bare assertion of the result)
  - "The results suggest / indicate / demonstrate that geometry drives
    transfer." ("suggests" / "indicates" / "demonstrates")
  - "This confirms the hypothesis." ("confirms")
  - "Cosine distance strongly predicts leakage (rho = 0.6)." (a measured value
    read as a conclusion)

An agent-written Results `### <plot name>` block carries the optional
connecting narrative (question-framed), then the `**Methodology**` block
stating what was computed and what is plotted EXACTLY (axes, units, groupings,
series coding, N) and then STOPS — the image follows, then the `**Takeaways**`
placeholder. The claims under each plot are Thomas's to write.

The litmus for a Motivation / Methodology sentence: "Would this sentence change
if the result had come out differently?" If yes, it is interpretation — cut it
or reframe it as a question. If no (it is true regardless of how the numbers
landed), it is methodology / motivation — keep it.

### The detailed companion writeup (`docs/reports/issue_<N>_detailed.md`)

The full-detail layer behind the summarized body, mechanically assembled by
the orchestrator from the same producer outputs (no new authoring agent):

```markdown
# Detailed writeup — issue <N>: <the H1 question/claim>

*(auto-generated companion to the report body; all content agent-written +
factual — claims live in the report body only)*

## Motivation
<copy of the body's Motivation>

## Methodology (full)
<the methodology-writer's UNABRIDGED output: complete hyperparameter table
(every value + Source), all worked verbatim examples, splits + CI recipe,
computed quantities, predictors / baselines / sanity checks, metrics>

## Results — full figure set

### <result 1 name>
<the result's **Methodology** block (recipe + what-is-plotted)>
<EVERY view of this result's figures, each: the factual caption + the
SHA-pinned image — aggregate, per-unit (labeled points), raw alongside
processed, every alt-grouping the manifest names>

### <result 2 name>
...

## Extra tables / diagnostics
<any tables or diagnostic outputs that did not fit the body>
```

Rules: NO Takeaways / TLDR / Conclusion sections (no Thomas slots — the doc is
regenerated wholesale on follow-up rounds, so nothing hand-written may live
here); the interpretivity rule applies throughout, and the banned-lexicon
scope mirrors the body's (the Motivation copy keeps the body's
hypothesis-framing exemption; every other section is agent methodology prose
under the full scan); every
image is SHA-pinned. The orchestrator commits it by explicit path on the
issue branch, captures the commit SHA, and splices the body's
`**Detailed writeup:**` blob link pinned at that SHA. Same-issue follow-up
rounds regenerate the doc, re-commit, and re-pin the link.
`verify_report.py` (`detailed-writeup-link`) requires the body link at
generation time (well-formed + issue-matched); a grandfathered body without
one only WARNs at promote.

### The metrics rationale ("why" grounded in the plan, never in the measured value)

Shared metrics are the final `**Metrics:**` block inside `## Methodology
(shared)`; a metric only one result uses may live in that result's
`**Methodology**` block instead — either way there is no separate `## Metrics`
H2. Each metric's rationale explains why the metric measures the Goal's
construct, grounded in the plan / Goal / measurement-validity rules — NEVER in
what the metric turned out to be:

- ALLOWED: "We report held-out reconstruction R^2 (variance-weighted over the
  3584 activation dims) instead of cosine similarity because all v_A share a
  large common component, so even predicting the mean v_A scores cosine ~0.98."
- BANNED: "We use the agreement rate; it came out at 0.87." (a measured value);
  "We chose the margin because it showed the clearest separation." (a
  justification read off the observed result).
