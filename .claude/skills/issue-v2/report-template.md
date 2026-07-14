# v2 experiment report — the official result template + authoring notes

**The official result template** (Thomas, 2026-07-14 — codified from the #779
context→answer-mapping writeup; supersedes the earlier six-section skeleton
that led with TLDR and kept `## Metrics:` as its own H2). The v2 clean-result
is a **report**: agents produce the factual skeleton — Motivation /
Methodology / Results-as-plots — that is verified for accuracy and
completeness; **Thomas alone writes the claims: the `# Result:` title, the
TLDR, every per-result `**Takeaways:**` block, and Next steps.** No agent
asserts a conclusion.

This file is the canonical skeleton every v2 producer + verifier reads:

- `methodology-writer` (REPORT MODE) authors Motivation + Methodology. The
  metrics definitions + rationale live INSIDE `## Methodology:` as its final
  `**Metrics:**` block — there is no separate `## Metrics:` H2.
- `plotter` produces the figures + factual what-is-plotted captions the
  orchestrator splices into Results.
- The orchestrator assembles the full report, leaving TLDR / every per-result
  `**Takeaways:**` block / Next steps as placeholders.
- `methodology-critic` traces every Motivation / Methodology claim (including
  the embedded metrics block) to ground truth.
- `report-verifier` runs the final completeness + interpretivity + recomputation
  gate and `scripts/verify_report.py --mode generation`.

The report body carries the `<!-- report-v1 -->` sentinel and lives in the task
`body.md` exactly like a markdown clean-result body.

---

## The skeleton (exact)

Five H2 sections, in this exact order: **Motivation → TLDR → Methodology →
Results → Next steps.**

```markdown
# Experiment: <one-line question>
<!-- report-v1 -->

## Motivation:

- <the prior result / context this experiment builds on — with the concrete
  numbers of the prior finding where relevant (e.g. "an earlier experiment
  found X: R^2 ~0.8 at layer 18")>
- <the question this experiment tests, stated as a question or "I wanted to
  test whether ..." — NEVER an asserted answer>

## TLDR:

*(Thomas fills in)*

## Methodology:

- **Model:** <model id>
- **Datasets:** <per dataset: a bold name, the source (linked), row counts,
  ONE fully worked verbatim example (prefix / query / answer), and a dashboard
  link to the full table>
    - **Splits:** <training / validation — naming EXACTLY what is selected on
      it (layer, hyperparameters) — / evaluation, with the CI recipe (e.g.
      "1000 prompts with 95% bootstrap CIs over test contexts")>
- **Computed quantities:** <how each vector / DV is computed, with the exact
  options enumerated and the default marked>
- **Predictors / conditions:** <each fitted model or experimental arm:
  architecture + hyperparameters — every load-bearing value copied from
  ground truth (config / run_result.json / the training script at its SHA)>
    - **Baselines:** <each baseline + the worry it addresses, in the shape
      "one worry here is <X>; test: <Y>">
    - **Sanity checks:** <e.g. train on permuted pairings>
- **Metrics:** <definition of each reported metric + WHY it was chosen over
  the alternatives, grounded in the plan / Goal — NEVER in a measured value>

## Results:

### <plot name 1>

<1-4 sentences of narrative: what was tested and how, then what is plotted
EXACTLY — axes (with units), groupings, color/series coding (e.g. "the blue
bars use the last prompt token, the orange bars the mean over prompt tokens"),
the eval N. NO interpretation, NO "this shows".>

**Plot: <plain-English plot name>**

![<plain-English figure description>](<SHA-pinned raw.githubusercontent.com URL>)

**Takeaways:**

*(Thomas fills in)*

### <plot name 2>

...

## Next steps:

*(Thomas fills in)*
```

## The filled-in form (what the report looks like after Thomas's pass)

When Thomas writes the TLDR he:

- **retitles the H1 to `# Result: <one-line claim>`** — the report's main
  claim (optionally with the trailing `(HIGH|MODERATE|LOW confidence)` tag the
  dashboard parses),
- **retitles each Results subsection to `### Result <n>: <one-line claim>`** —
  the claim that plot supports (sub-numbering like `Result 4.5` is fine for a
  follow-on control of the same result),
- **fills every `**Takeaways:**` block** with numbers-first bullets (the
  statistic + CI inline; sub-bullets carry supporting numbers, caveats, and
  "still unsure about X" honesty),
- **fills `## TLDR:`** — one numbers-first claim bullet per headline finding,
  each with the key statistic inline and sub-bullets for the supporting
  evidence,
- **fills `## Next steps:`** — including `(running)` markers for work already
  in flight.

Worked exemplar of the filled form:
`.claude/skills/issue-v2/exemplars/issue-779-filled-report.md`.

---

## Authoring notes

### Title convention

At generation time the H1 is `# Experiment: <one-line question>` — agents have
no finding to claim, so a claim-shaped title would violate the interpretivity
rule. Thomas retitles to `# Result: <one-line claim>` when he fills the TLDR.
`verify_report.py` enforces this: generation mode requires the `Experiment:`
prefix; promote mode accepts `Result:` (preferred) or `Experiment:` (not yet
retitled). Confidence lives only in the H1 trailing tag, added by Thomas —
no agent-written section ever contains a `Confidence:` line.

### Two verify modes

`scripts/verify_report.py` runs in two modes; the report must pass the relevant
one at each gate:

- **`--mode generation`** (at report assembly, before parking at
  `awaiting_promotion`): the `## TLDR:` and `## Next steps:` placeholders MUST
  be intact (`*(Thomas fills in)*`), and every Results subsection's
  `**Takeaways:**` block must be exactly the placeholder too. The
  interpretivity / lexicon checks run on the AGENT-written sections —
  Methodology and Results (Motivation is exempt — hypothesis framing is
  allowed there). This is the gate `report-verifier` runs.
- **`--mode promote`** (when Thomas runs `task.py promote`): the `## TLDR:`
  placeholder MUST now be FILLED (Thomas wrote it). Thomas's prose — TLDR,
  Next steps, and the Results takeaways/claim-headings he filled in — is NEVER
  lexicon- or interpretivity-checked; only `## Methodology:` (pure agent
  prose in both modes) stays under the lexicon scan at promote time.

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

An agent-written Results `### <plot name>` block describes what was tested and
what is plotted EXACTLY (axes, units, groupings, series coding, N) and then
STOPS — the image follows, then the `**Takeaways:**` placeholder. The claims
under each plot are Thomas's to write.

The litmus for a Motivation / Methodology sentence: "Would this sentence change
if the result had come out differently?" If yes, it is interpretation — cut it
or reframe it as a question. If no (it is true regardless of how the numbers
landed), it is methodology / motivation — keep it.

### The Metrics block ("why" grounded in the plan, never in the measured value)

Metrics are the final `**Metrics:**` block inside `## Methodology:` — there is
no separate H2. Each metric's rationale explains why the metric measures the
Goal's construct, grounded in the plan / Goal / measurement-validity rules —
NEVER in what the metric turned out to be:

- ALLOWED: "We report held-out reconstruction R^2 (variance-weighted over the
  3584 activation dims) instead of cosine similarity because all v_A share a
  large common component, so even predicting the mean v_A scores cosine ~0.98."
- BANNED: "We use the agreement rate; it came out at 0.87." (a measured value);
  "We chose the margin because it showed the clearest separation." (a
  justification read off the observed result).
